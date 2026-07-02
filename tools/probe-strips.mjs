// probe-strips.mjs — measure the STRIP-TILING ORCHESTRATION OVERHEAD of a deep
// viewer render (Spawn 28; the M6-FUTURE item).
//
// _stripRows() sizes strips from a WORST-CASE budget 4e8/(W·maxIter) pixel-iterations
// — but it ignores the SA skip. SA seeds EVERY pixel at iteration `skip` (~90-95% of
// maxIter deep), so the true worst case is maxIter−skip and the viewer cuts ~10-20×
// more strips than the budget intends. Each strip pays fixed costs (main path: draw +
// flush + full colorize + blit + a vsync-locked rAF ≈ 16.7ms; worker path: draw +
// full-canvas colorize + full-canvas ImageBitmap + postMessage + ack). This probe
// drives the LIVE viewer at a deep boundary coordinate and A/Bs the SAME render with
// production strips vs coarser overrides (×8, single-strip) by monkey-patching
// _stripRows. The delta is pure orchestration overhead (the CPU ref build and all GPU
// math are identical across arms — strip partition is bit-identical, crosscheck-tiled).
//
//   GPU=1 node tools/probe-strips.mjs                 (real GPU)
//   GPU=1 BITS=400,700 ROUNDS=3 node tools/probe-strips.mjs
//   OFFSCREEN=0 forces the on-main-thread path (rAF yields — the worst case)
import { chromium } from '@playwright/test';
import { launchOpts, gpuMode } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8153;
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
const BITSET = (process.env.BITS || '271,400,700').split(',').map(Number);
const ROUNDS = Number(process.env.ROUNDS || 3);
const OFFSCREEN = process.env.OFFSCREEN !== '0';

let fail = false;
try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage({ viewport: { width: 600, height: 600 } });
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail = true; });
  await page.goto(baseURL + `/index.html${OFFSCREEN ? '' : '?offscreen=0'}`);
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 30000 });
  await page.evaluate(() => window.__viewer.setSupersample(1));
  console.log('mode=' + gpuMode() + '  offscreen=' + OFFSCREEN + '  renderer:', await page.evaluate(() => {
    const gl = window.__viewer.gpu?.gl; const d = gl?.getExtension('WEBGL_debug_renderer_info');
    return d ? gl.getParameter(d.UNMASKED_RENDERER_WEBGL) : '(worker-owned GL)';
  }));

  // stash the original once; overrides multiply it or force a single strip
  await page.evaluate(() => {
    const v = window.__viewer;
    if (!v.__origStripRows) v.__origStripRows = v._stripRows.bind(v);
  });
  const setStrips = (mode) => page.evaluate((m) => {
    const v = window.__viewer;
    if (m === 'prod') v._stripRows = v.__origStripRows;
    else if (m === 'full') v._stripRows = () => v.cH;
    else v._stripRows = () => Math.min(v.cH, v.__origStripRows() * Number(m)); // '8' → ×8
  }, mode);

  async function renderOnce(bits) {
    const radius = 1.5 * 2 ** -bits;
    const c0 = await page.evaluate(() => window.__doneCount || 0);
    // jiggle the radius by an exact tiny factor so setState always re-renders
    await page.evaluate((vv) => window.__viewer.setState(vv), { cx: RE, cy: IM, radius });
    const t0 = Date.now();
    await page.waitForFunction((p) => (window.__doneCount || 0) > p, c0, { timeout: 300000 });
    const ms = Date.now() - t0;
    const info = await page.evaluate(() => ({
      engine: window.__lastDone?.engine, saSkip: window.__lastDone?.saSkip, maxIter: window.__viewer.maxIter,
      // what the render actually used: the skip-aware strip height (Spawn 28)
      stripH: window.__viewer._stripRows(window.__lastDone?.saSkip || 0),
      cH: window.__viewer.cH, cW: window.__viewer.cW,
    }));
    return { ms, ...info };
  }

  console.log('\nlive viewer, deep boundary coordinate, ss=1 — wall-clock per config (min of rounds; incl. CPU ref build, identical across arms)\n');
  console.log('depth    maxIter  skip%  | cfg   stripH  #strips | min ms');
  for (const bits of BITSET) {
    for (const mode of ['prod', '8', 'full']) {
      await setStrips(mode);
      let best = null;
      for (let r = 0; r < ROUNDS; r++) {
        const o = await renderOnce(bits);
        if (o.engine !== 'gpu-perturb-fe') { console.log(`  !! engine=${o.engine} (expected gpu-perturb-fe) — aborting case`); fail = true; break; }
        if (!best || o.ms < best.ms) best = o;
      }
      if (!best) continue;
      const nStrips = Math.ceil(best.cH / best.stripH);
      const pct = best.saSkip ? (100 * best.saSkip / best.maxIter).toFixed(1) : '0.0';
      console.log(
        `2^-${String(bits).padEnd(4)} ${String(best.maxIter).padStart(7)} ${pct.padStart(5)}% | ` +
        `${mode.padEnd(5)} ${String(best.stripH).padStart(6)} ${String(nStrips).padStart(8)} | ${String(best.ms).padStart(6)}`);
    }
  }
  console.log('\n(prod vs full delta = pure strip-orchestration overhead; ×8 shows the shape.');
  console.log(' The ref build is repeated identically in every render, so deltas are clean.)');
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e); fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
