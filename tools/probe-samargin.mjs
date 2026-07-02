// probe-samargin.mjs — how low can the SA safety MARGIN go at the deep band while staying
// bit-exact on the real GPU? (Spawn 22 measure-first probe.)
//
// computeSeries backs the skip off by marginFrac (default 0.05) of the grid-validated validN —
// a guard because the validation runs in f64 but the GPU seeds in df64 (~46-bit), which can
// diverge "a touch sooner." At 2^-120 that 5% margin discards ~1380 iters/pixel ≈ a THIRD of
// the entire post-SA budget (~4190 iters) — so shaving it is a uniform iteration-count cut on
// EVERY pixel (its benefit is robust to the GPU's throughput-vs-divergence binding, unlike a
// per-pixel prune). probe-saorder-gpu showed the deep-band df64 seed is ~1000× under the gate,
// so the margin is likely over-conservative DEEP. This sweeps marginFrac and reports skip% +
// the bit-exact mism (GPU-SA vs the no-SA CPU oracle) + the smooth-count drift. The gate: mism
// stays 0 (no over-skip past an escape/rebase); meanΔsn stays tiny (df64 precision, allowed).
//
//   GPU=1 node tools/probe-samargin.mjs
//   GPU=1 BITS=120,218,271 MARGINS=0.05,0.03,0.02,0.01,0.005,0 W=48 node tools/probe-samargin.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8151;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));
const BITS = (process.env.BITS || '120,218,271').split(',').map(Number);
const MARGINS = (process.env.MARGINS || '0.05,0.03,0.02,0.01,0.005,0').split(',').map(Number);
const W = Number(process.env.W || 48), H = Number(process.env.H || 48);

try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); });
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const init = await page.evaluate(() => window.__gpu.init(128, 128));
  if (!init.supported) throw new Error('GPU not supported');
  console.log('renderer:', JSON.parse(JSON.stringify(await page.evaluate(() => window.__gpu.info()))).renderer);
  console.log(`\nGPU rescaled+SA(order5) vs no-SA oracle, deep boundary coord (${W}×${H}). GATE: mism 0, meanΔsn tiny.`);
  for (const bits of BITS) {
    const r = 1.5 * 2 ** -bits, it = autoIter(bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: W, height: H, checkStep: 1, glitchTol: 0, rs: true };
    console.log(`\n2^-${bits}  maxIter ${it}`);
    console.log('  margin | skip%   postSA  | mism     meanΔsn  maxΔsn   | verdict');
    for (const mg of MARGINS) {
      const o = await page.evaluate((q) => window.__gpu.comparePerturb(q),
        { ...base, series: { order: 5, marginFrac: mg } });
      const frac = 100 * (o.compared ? o.mism / o.compared : 0);
      const postSA = it - o.saSkip;
      const ok = o.mism === 0;
      console.log(
        `  ${mg.toFixed(3)}  | ${(100 * o.saSkip / it).toFixed(1).padStart(5)}% ${String(postSA).padStart(7)} | ` +
        `${frac.toFixed(3).padStart(6)}% ${o.meanAbs.toExponential(2).padStart(9)} ${o.maxAbs.toExponential(2).padStart(9)} | ` +
        `${ok ? 'bit-exact ✓' : 'OVER-SKIP ✗'}`);
    }
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
