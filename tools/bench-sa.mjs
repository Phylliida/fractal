// bench-sa.mjs — measure the GPU series-approximation speedup: the rescaled deep engine
// rendered WITH the in-shader SA seed vs WITHOUT, on the genuine deep boundary coordinate
// (real high-count escapes where SA engages a large skip). Pure GPU render time (a draw +
// a 1-px readback to force the pipeline). Run on the real GPU with GPU=1.
//
//   node tools/bench-sa.mjs
//   GPU=1 node tools/bench-sa.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8143;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

// The genuine deep boundary coordinate (tools/gen-deep-coord.mjs) — a real filament with
// high-count chaotic escapes, so SA skips ~85-90% of the leading iterations.
const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));
const CASES = [
  { bits: 120, W: 128, H: 128 },
  { bits: 271, W: 96, H: 96 },
  { bits: 400, W: 64, H: 64 },
];

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
  console.log('\nrescaled deep engine — SA off vs SA on (deep boundary coordinate)\n');
  console.log('depth    size     maxIter  skipN   skip%  | off ms   on ms  speedup');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: 3, rs: true };
    const off = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base });
    const on = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, series: true });
    const pct = (100 * on.saSkip / it).toFixed(1);
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${String(c.W + 'x' + c.H).padEnd(8)} ${String(it).padStart(7)}  ` +
      `${String(on.saSkip).padStart(6)} ${pct.padStart(5)}% | ${off.ms.toFixed(1).padStart(6)} ${on.ms.toFixed(1).padStart(6)}  ` +
      `${(off.ms / Math.max(0.01, on.ms)).toFixed(2).padStart(5)}x`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
