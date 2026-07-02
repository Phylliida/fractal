// probe-refcache.mjs — REFERENCE-REUSE cache (Spawn 30): live-viewer timing + correctness.
//
// Simulates the core deep-zoom interaction — successive zoom-in ticks at a fixed
// point — and A/Bs the settle-render time WITH the reference cache (production)
// vs COLD (cache force-disabled): warm ticks should skip the BigInt rebuild
// (extend-only, ~ms) while cold ticks pay the full reference build every time.
// CORRECTNESS: the warm path renders against a cached (higher-prec, possibly
// drifted) reference — a different-but-equally-valid realization — so the final
// frames are compared by BULK grid samples (the project's standard for reference
// realization changes), plus the engine/skip/refLen sanity fields.
//
//   GPU=1 node tools/probe-refcache.mjs
//   GPU=1 BITS=350 TICKS=6 node tools/probe-refcache.mjs
import { chromium } from '@playwright/test';
import { launchOpts, gpuMode } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8154;
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
const BITS = Number(process.env.BITS || 350);
const TICKS = Number(process.env.TICKS || 6);

function gridEvalSrc() {
  return (G) => {
    const c = document.getElementById('view'); const g = c.getContext('2d');
    const { data } = g.getImageData(0, 0, c.width, c.height);
    const rgb = [];
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
      const px = Math.floor((gx + 0.5) / G * c.width), py = Math.floor((gy + 0.5) / G * c.height);
      const o = (py * c.width + px) * 4;
      rgb.push(data[o], data[o + 1], data[o + 2]);
    }
    return rgb;
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
  await page.evaluate(() => window.__viewer.setSupersample(1));
  console.log('mode=' + gpuMode() + '  zoom-tick sequence at the deep boundary coordinate, start 2^-' + BITS + ', ' + TICKS + ' ticks ×0.5\n');

  async function waitDone(c0) {
    await page.waitForFunction((p) => (window.__doneCount || 0) > p, c0, { timeout: 300000 });
  }
  async function runSequence(cold, drift) {
    // fresh state at the anchor depth
    await page.evaluate((v) => {
      const V = window.__viewer;
      V._refCache = null;
      V.__coldMode = v.cold;
      if (v.cold && !V.__origUsable) V.__origUsable = V._refCacheUsable.bind(V);
      V._refCacheUsable = v.cold ? () => null : (V.__origUsable || V._refCacheUsable.bind(V));
    }, { cold });
    let c0 = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate((vv) => window.__viewer.setState(vv), { cx: RE, cy: IM, radius: 1.5 * 2 ** -BITS });
    await waitDone(c0);
    const ticks = [];
    for (let t = 0; t < TICKS; t++) {
      c0 = await page.evaluate(() => window.__doneCount || 0);
      const t0 = Date.now();
      // drift=false: zoom at the EXACT canvas center — drift stays 0.0 (the fast path).
      // drift=true: zoom at an OFF-CENTER pixel, like a real wheel/click zoom — the
      // view center then DRIFTS from the cached reference point every tick. This is
      // the case that catches offset-sign/mirroring bugs in the cache's dc synthesis
      // (the Spawn-31 "blue screen / teleports on click" bug): a centered-only probe
      // is geometrically blind to them.
      await page.evaluate((d) => {
        const V = window.__viewer;
        const px = V.backingW / 2 + (d ? V.backingW * 0.05 : 0);
        const py = V.backingH / 2 + (d ? -V.backingH * 0.04 : 0);
        V.zoomAt(px, py, 0.5);
        V.render();
      }, drift);
      await waitDone(c0);
      const info = await page.evaluate(() => ({
        cached: !!(window.__viewer._refMeta && window.__viewer._refMeta.refCached),
        refLen: window.__viewer._refMeta?.refLen, skip: window.__viewer._saSkip,
        engine: window.__viewer._refMeta?.engine,
      }));
      ticks.push({ ms: Date.now() - t0, ...info });
    }
    const rgb = await page.evaluate(gridEvalSrc(), 48);
    return { ticks, rgb };
  }

  const warm = await runSequence(false, false);
  const cold = await runSequence(true, false);
  const warmD = await runSequence(false, true);   // drifted (real-zoom-like) sequence
  const coldD = await runSequence(true, true);

  console.log('tick |  warm ms (cached? skip)      |  cold ms (skip)');
  let wSum = 0, cSum = 0;
  for (let t = 0; t < TICKS; t++) {
    const w = warm.ticks[t], c = cold.ticks[t];
    wSum += w.ms; cSum += c.ms;
    console.log(` ${String(t + 1).padStart(3)} | ${String(w.ms).padStart(6)}  (${w.cached ? 'HIT ' : 'MISS'} skip ${String(w.skip).padStart(6)}) | ${String(c.ms).padStart(6)}  (skip ${String(c.skip).padStart(6)})`);
  }
  // bulk correctness: same final view, warm vs cold realization
  const bulkDiff = (a, b) => {
    let diff = 0; const n = a.length / 3;
    for (let i = 0; i < a.length; i += 3) {
      if (Math.abs(a[i] - b[i]) > 24 || Math.abs(a[i + 1] - b[i + 1]) > 24 ||
          Math.abs(a[i + 2] - b[i + 2]) > 24) diff++;
    }
    return diff / n;
  };
  const diffFrac = bulkDiff(warm.rgb, cold.rgb);
  const diffFracD = bulkDiff(warmD.rgb, coldD.rgb);
  const hits = warm.ticks.filter((t) => t.cached).length;
  const hitsD = warmD.ticks.filter((t) => t.cached).length;
  const wSumD = warmD.ticks.reduce((s, t) => s + t.ms, 0);
  const cSumD = coldD.ticks.reduce((s, t) => s + t.ms, 0);
  console.log(`\ncentered: warm ${wSum}ms vs cold ${cSum}ms → ${(cSum / Math.max(1, wSum)).toFixed(2)}× faster ticks; ` +
              `${hits}/${TICKS} hits; final-frame bulk diff ${(diffFrac * 100).toFixed(2)}%`);
  console.log(`drifted : warm ${wSumD}ms vs cold ${cSumD}ms → ${(cSumD / Math.max(1, wSumD)).toFixed(2)}× faster ticks; ` +
              `${hitsD}/${TICKS} hits; final-frame bulk diff ${(diffFracD * 100).toFixed(2)}%  (MUST be small — mirror-bug guard)`);
  const ok = hits === TICKS && diffFrac < 0.02 && diffFracD < 0.02 && hitsD >= 1 &&
             warm.ticks.every((t) => t.engine === 'gpu-perturb-fe') && wSum < cSum;
  if (!ok) { fail = true; console.log('FAIL — see fields above'); }
  else console.log('PASS');
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e); fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
