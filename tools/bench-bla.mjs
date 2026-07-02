// bench-bla.mjs — measure the GPU BLA speedup on the rescaled deep engine: pure GPU render
// time (a draw + 1-px readback to force the pipeline) on the genuine deep boundary coordinate,
// across three configs — no-opt (off), series approximation (SA), and SA+BLA. BLA jumps RUNS of
// linear iterations THROUGHOUT the orbit (incl. post-rebase) — the part of the ~SA floor that SA
// cannot touch — so SA+BLA is the headline path. Run on the real GPU with GPU=1.
//
//   node tools/bench-bla.mjs
//   GPU=1 node tools/bench-bla.mjs
//   GPU=1 BITS=120,271,400 EPS=30 node tools/bench-bla.mjs
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
const EPS = 2 ** -Number(process.env.EPS || 30);
const BITS = (process.env.BITS || '120,271,400').split(',').map(Number);
const SIZE = { 120: 128, 271: 96, 400: 64 };

try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); });
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const init = await page.evaluate(() => window.__gpu.init(256, 256));
  if (!init.supported) throw new Error('GPU not supported');
  console.log('renderer:', (await page.evaluate(() => window.__gpu.info())).renderer);
  console.log(`\nrescaled deep engine — off vs SA vs SA+BLA (deep boundary coordinate, eps 2^${Math.round(Math.log2(EPS))})\n`);
  console.log('depth    size     maxIter | off ms   SA ms  SA+BLA |  SA×   SA+BLA×   (vs off)');
  for (const bits of BITS) {
    const r = 1.5 * 2 ** -bits, it = autoIter(bits), S = SIZE[bits] || 64;
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: S, height: S, reps: 3, rs: true };
    const off = await page.evaluate((q) => window.__gpu.benchPerturb(q), { ...base });
    const sa = await page.evaluate((q) => window.__gpu.benchPerturb(q), { ...base, series: true });
    const sb = await page.evaluate((q) => window.__gpu.benchPerturb(q), { ...base, series: true, bla: { eps: EPS } });
    console.log(
      `2^-${String(bits).padEnd(4)} ${String(S + 'x' + S).padEnd(8)} ${String(it).padStart(7)} | ` +
      `${off.ms.toFixed(1).padStart(6)} ${sa.ms.toFixed(1).padStart(6)} ${sb.ms.toFixed(1).padStart(6)}  | ` +
      `${(off.ms / Math.max(0.01, sa.ms)).toFixed(2).padStart(5)}× ` +
      `${(off.ms / Math.max(0.01, sb.ms)).toFixed(2).padStart(6)}×   lvls=${sb.blaMaxLevel}`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
