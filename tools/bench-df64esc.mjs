// bench-df64esc.mjs — measure the df64-escape-fast-path speedup in the rescaled deep engine.
// The per-iteration escape/rebase/glitch test used to run in floatexp every iteration; it now
// runs in plain df64 in the common case (|dz|,|Z_m| >= ~2^-100), falling back to fe only near a
// reference minimum (uDf64Esc, glsl.js). This A/Bs the SAME deep render with df64Esc OFF (forced
// fe, the old behaviour) vs ON (default), BOTH with series approximation on (the production path).
// Pure GPU render time (draw + 1-px readback to force the pipeline). Run on the real GPU: GPU=1.
//
//   node tools/bench-df64esc.mjs
//   GPU=1 node tools/bench-df64esc.mjs
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

// The genuine deep boundary coordinate (tools/gen-deep-coord.mjs) — a real filament with
// high-count chaotic escapes, where the escape/rebase block runs nearly every iteration.
const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));
// SIZE env overrides W/H (a larger render is compute-bound: per-iteration shader cost
// dominates the fixed draw+readback overhead, so any ALU win shows up clearly).
const SZ = process.env.SIZE ? parseInt(process.env.SIZE, 10) : 0;
const NOSA = process.env.NOSA === '1';   // run the escape block EVERY iteration (no SA skip)
const CASES = [
  { bits: 120, W: SZ || 128, H: SZ || 128 },
  { bits: 271, W: SZ || 96, H: SZ || 96 },
  { bits: 400, W: SZ || 64, H: SZ || 64 },
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
  console.log('\nrescaled deep engine + SA — escape/rebase in floatexp (off) vs df64 (on)\n');
  console.log('depth    size     maxIter  skipN   skip%  |  fe ms  df64 ms  speedup');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: 5, rs: true, series: !NOSA };
    const off = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, df64Esc: 0 });
    const on = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, df64Esc: 1 });
    const pct = (100 * on.saSkip / it).toFixed(1);
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${String(c.W + 'x' + c.H).padEnd(8)} ${String(it).padStart(7)}  ` +
      `${String(on.saSkip).padStart(6)} ${pct.padStart(5)}% | ${off.ms.toFixed(1).padStart(6)} ${on.ms.toFixed(1).padStart(7)}  ` +
      `${(off.ms / Math.max(0.01, on.ms)).toFixed(2).padStart(5)}x`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
