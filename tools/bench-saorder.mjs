// bench-saorder.mjs — DECISIVE wall-clock A/B for higher-order series approximation (Spawn 21).
// The rescaled deep engine rendered with the in-shader SA seed at ORDER 3 vs ORDER 5, plus a
// no-SA baseline, on the genuine deep boundary coordinate. Order 5 raises the skip% most at the
// moderate-deep band (the documented speed gap). This measures whether that skip% gain actually
// moves wall-clock, or is swallowed by the post-rebase re-growth floor. Run on the real GPU
// with GPU=1 (SwiftShader timing is unreliable on a contended host — see NOTES).
//
//   GPU=1 node tools/bench-saorder.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8146;
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
const CASES = (process.env.BITS
  ? process.env.BITS.split(',').map((b) => ({ bits: Number(b), W: 128, H: 128 }))
  : [{ bits: 120, W: 128, H: 128 }, { bits: 218, W: 112, H: 112 },
     { bits: 271, W: 96, H: 96 }, { bits: 400, W: 64, H: 64 }]);

try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); });
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const init = await page.evaluate(() => window.__gpu.init(256, 256));
  if (!init.supported) throw new Error('GPU not supported');
  console.log('renderer:', JSON.parse(JSON.stringify(await page.evaluate(() => window.__gpu.info()))).renderer);
  console.log('\nrescaled deep engine — SA order 3 vs order 5 (deep boundary coordinate)\n');
  console.log('depth    size     maxIter | o3 skip%  o3 ms | o5 skip%  o5 ms | o5/o3 | o5 vs noSA');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: 5, rs: true };
    const off = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base });
    const o3 = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, series: { order: 3 } });
    const o5 = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, series: { order: 5 } });
    const p3 = (100 * o3.saSkip / it).toFixed(1), p5 = (100 * o5.saSkip / it).toFixed(1);
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${String(c.W + 'x' + c.H).padEnd(8)} ${String(it).padStart(7)} | ` +
      `${p3.padStart(6)}% ${o3.ms.toFixed(1).padStart(6)} | ` +
      `${p5.padStart(6)}% ${o5.ms.toFixed(1).padStart(6)} | ` +
      `${(o3.ms / Math.max(0.01, o5.ms)).toFixed(2).padStart(5)} | ` +
      `${(off.ms / Math.max(0.01, o5.ms)).toFixed(2).padStart(5)}x`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
