// probe-sa-viewer.mjs — END-TO-END check that the LIVE viewer engages GPU series
// approximation on the deep fe/rescaled path and stays correct + fast. Unlike
// validate-gpu (which computes the SA coeffs itself in the harness), this drives the
// real app: render() → worker computeRef (computes params.sa) → _renderGpuPerturb →
// renderPerturbRescaled with the seed. It asserts the done status reports a real skip,
// the engine is gpu-perturb-fe, the image is structured + glitch-free, and that turning
// SA OFF yields the SAME picture (fingerprint) — just slower.
//
//   GPU=1 node tools/probe-sa-viewer.mjs      (real GPU — the meaningful run)
import { chromium } from '@playwright/test';
import { launchOpts } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8149;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/index.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const BITS = Number(process.env.BITS || 271);

// Downsample the canvas to a G×G grid of RGB samples (so SA-on vs SA-off can be compared
// by BULK fraction — the project's standard — tolerating the measure-zero ill-conditioned
// boundary pixels rather than demanding an exact match). Also report distinct colors.
function gridEvalSrc(G) {
  return (GG) => {
    const c = document.getElementById('view'); const g = c.getContext('2d');
    const { data } = g.getImageData(0, 0, c.width, c.height);
    const rgb = [], distinct = new Set();
    for (let gy = 0; gy < GG; gy++) for (let gx = 0; gx < GG; gx++) {
      const px = Math.floor((gx + 0.5) / GG * c.width), py = Math.floor((gy + 0.5) / GG * c.height);
      const o = (py * c.width + px) * 4;
      rgb.push(data[o], data[o + 1], data[o + 2]);
      distinct.add((data[o] >> 4 << 8) | (data[o + 1] >> 4 << 4) | (data[o + 2] >> 4));
    }
    return { rgb, distinct: distinct.size };
  };
}

let fail = false;
try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage({ viewport: { width: 600, height: 600 } });
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail = true; });
  await page.goto(baseURL + '/index.html');
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 30000 });
  await page.evaluate(() => window.__viewer.setSupersample(1));   // ss=1: 1× pixels for a fast probe

  const radius = 1.5 * 2 ** -BITS;
  async function renderWith(series) {
    const c0 = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate((on) => window.__viewer.setSeries(on), series);
    const c1 = await page.evaluate(() => window.__doneCount || 0);
    // setSeries re-renders only if the flag changed; force a fresh render at the coordinate.
    await page.evaluate((vv) => window.__viewer.setState(vv), { cx: RE, cy: IM, radius });
    const t0 = Date.now();
    await page.waitForFunction((p) => (window.__doneCount || 0) > p, Math.max(c0, c1), { timeout: 180000 });
    const ms = Date.now() - t0;
    const info = await page.evaluate(() => ({
      engine: window.__lastDone?.engine, saSkip: window.__lastDone?.saSkip,
      glitches: window.__lastDone?.glitches, refLen: window.__lastDone?.refLen,
    }));
    const grid = await page.evaluate(gridEvalSrc(48), 48);
    return { ...info, ms, ...grid };
  }

  console.log('renderer:', (await page.evaluate(() => {
    const gl = window.__viewer.gpu?.gl; const d = gl?.getExtension('WEBGL_debug_renderer_info');
    return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : 'n/a';
  })));
  console.log(`\nlive viewer @ 2^-${BITS}, deep boundary coordinate (ss=1, 600×600)\n`);

  const on = await renderWith(true);
  const off = await renderWith(false);
  // Bulk fraction of grid samples whose color differs (per-channel > 24/255) between SA
  // on and off — the measure-zero ill-conditioned boundary pixels may differ; the picture
  // as a whole must match.
  let diff = 0; const n = on.rgb.length / 3;
  for (let i = 0; i < on.rgb.length; i += 3) {
    if (Math.abs(on.rgb[i] - off.rgb[i]) > 24 || Math.abs(on.rgb[i + 1] - off.rgb[i + 1]) > 24 ||
        Math.abs(on.rgb[i + 2] - off.rgb[i + 2]) > 24) diff++;
  }
  const diffFrac = diff / n;
  const brief = (o) => ({ engine: o.engine, saSkip: o.saSkip, glitches: o.glitches, refLen: o.refLen, distinct: o.distinct, ms: o.ms });
  console.log('SA on :', JSON.stringify(brief(on)));
  console.log('SA off:', JSON.stringify(brief(off)));
  console.log(`grid-sample diff fraction (SA on vs off): ${(diffFrac * 100).toFixed(2)}%`);

  const ok =
    on.engine === 'gpu-perturb-fe' && off.engine === 'gpu-perturb-fe' &&   // deep fe path
    on.saSkip > 0 && off.saSkip === 0 &&                                   // SA engaged only when on
    on.distinct > 8 &&                                                     // structured (not blank)
    on.glitches === 0 &&                                                   // glitch-free
    diffFrac < 0.02;                                                       // same picture (bulk)
  if (!ok) { fail = true; console.log('\nFAIL — see fields above'); }
  else console.log(`\nPASS — SA engaged (skip ${on.saSkip}), same picture SA on/off (${(diffFrac * 100).toFixed(2)}% diff), ` +
                   `${(off.ms / Math.max(1, on.ms)).toFixed(2)}× faster (${off.ms}ms → ${on.ms}ms incl. ref build)`);
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e); fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
