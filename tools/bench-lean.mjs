// bench-lean.mjs — DECISIVE wall-clock A/B for the LEAN rescaled kernel (Spawn 22).
//
// The production deep render runs with BLA OFF (a measured GPU net-loss) and df64-escape OFF
// (measured neutral), but those uniform-gated blocks are still COMPILED into the kernel. A GPU
// sets occupancy (active warps) from a kernel's PEAK live-register count across ALL branches, so
// the never-taken BLA scan + df64-escape blocks may throttle occupancy for EVERY pixel. The LEAN
// variant (perturbFragRescaled({bla:false,df64esc:false})) excludes them at build time — output
// BIT-IDENTICAL in production config, but a smaller kernel that may hit higher occupancy.
//
// This renders the SAME deep view with the FULL kernel (lean:false, the production default) and
// the LEAN kernel (lean:true), production config (SA on, no BLA, no df64-escape), reports wall-clock.
// Run on the real GPU with GPU=1 (SwiftShader is a CPU rasterizer with no occupancy notion +
// an unreliable clock on a contended host — see NOTES).
//
//   GPU=1 node tools/bench-lean.mjs
//   GPU=1 BITS=120,218,271,400 node tools/bench-lean.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8149;
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
  ? process.env.BITS.split(',').map((b) => ({ bits: Number(b), W: 160, H: 160 }))
  : [{ bits: 120, W: 192, H: 192 }, { bits: 218, W: 160, H: 160 },
     { bits: 271, W: 144, H: 144 }, { bits: 400, W: 112, H: 112 }]);
const REPS = Number(process.env.REPS || 9);

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
  console.log('\nrescaled deep engine, production config (SA on) — FULL kernel vs LEAN kernel');
  console.log('(LEAN excludes the never-taken BLA + df64-escape blocks; output bit-identical)\n');
  console.log('depth    size      maxIter | skip%  | full ms | lean ms | full/lean (speedup)');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: REPS, rs: true, series: true };
    // interleave full/lean a couple times to average out any thermal/scheduling drift
    let fms = Infinity, lms = Infinity, skip = 0;
    for (let rep = 0; rep < 2; rep++) {
      const full = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, lean: false });
      const lean = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, lean: true });
      fms = Math.min(fms, full.ms); lms = Math.min(lms, lean.ms); skip = full.saSkip;
    }
    const pct = (100 * skip / it).toFixed(1);
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${String(c.W + 'x' + c.H).padEnd(9)} ${String(it).padStart(7)} | ` +
      `${pct.padStart(5)}% | ${fms.toFixed(2).padStart(7)} | ${lms.toFixed(2).padStart(7)} | ` +
      `${(fms / Math.max(0.01, lms)).toFixed(3).padStart(6)}×`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
