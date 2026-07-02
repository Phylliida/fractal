// probe-saorder-gpu.mjs — find where order-5 SA becomes df64-SAFE on the real GPU (Spawn 21).
// Order-5 raises the skip%, but a larger skip seeds dz at a later iteration with larger
// coefficients, so the df64 (~46-bit) in-shader Horner has a larger ABSOLUTE seed error that
// then amplifies through the chaotic post-seed continuation. At the shallow chaotic seahorse
// band this pushed meanΔsn over the validate-gpu gate; at deep zoom there's 3+ orders of
// headroom. This locates the crossover on the genuine deep boundary coordinate (chaotic
// filament) so the depth-adaptive order threshold can be set from data, not guesswork.
//
//   GPU=1 node tools/probe-saorder-gpu.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8147;
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
const BITS = (process.env.BITS || '100,110,115,120,130').split(',').map(Number);

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
  console.log('\nGPU rescaled+SA vs no-SA oracle, deep boundary coord — order 3 vs 5 (gate: meanΔsn<1.0, mism<2%)\n');
  console.log('depth   | o3 skip%  mism    meanΔsn | o5 skip%  mism    meanΔsn');
  for (const bits of BITS) {
    const r = 1.5 * 2 ** -bits, it = autoIter(bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: 64, height: 64, checkStep: 1, glitchTol: 0, rs: true };
    const o3 = await page.evaluate((q) => window.__gpu.comparePerturb(q), { ...base, series: { order: 3 } });
    const o5 = await page.evaluate((q) => window.__gpu.comparePerturb(q), { ...base, series: { order: 5 } });
    const frac = (o) => 100 * (o.compared ? o.mism / o.compared : 0);
    console.log(
      `2^-${String(bits).padEnd(4)}| ${(100 * o3.saSkip / it).toFixed(1).padStart(6)}% ${frac(o3).toFixed(3).padStart(6)}% ${o3.meanAbs.toExponential(2).padStart(9)} | ` +
      `${(100 * o5.saSkip / it).toFixed(1).padStart(6)}% ${frac(o5).toFixed(3).padStart(6)}% ${o5.meanAbs.toExponential(2).padStart(9)}`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
