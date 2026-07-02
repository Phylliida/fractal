// bench-prefetch.mjs — wall-clock A/B for the SOFTWARE-PIPELINED reference prefetch
// (Spawn 27) in the rescaled deep engine.
//
// The production loop texelFetches Z[m] and uses it IMMEDIATELY (escape test), so if
// the warp scheduler cannot hide the fetch latency behind other warps, it is exposed
// serially every iteration. The prefetch variant (perturbFragRescaled({prefetch:true}))
// carries Z[m+1] in registers: each fetch is issued a full iteration before first use,
// overlapping the escape-test + next-update ALU. Same fetched values, same arithmetic
// -> output must be BIT-IDENTICAL (crossArgs gate: nDiff === 0). Cost: 2 vec2 registers
// (the companion-LLM's caveat: watch for an occupancy drop turning this negative).
//
//   GPU=1 node tools/bench-prefetch.mjs
//   GPU=1 BITS=120,218,271,400 ROUNDS=5 REPS=11 node tools/bench-prefetch.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8152;
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
// ENGINE=df64|fe|rs (default rs) selects the engine for BITS-specified cases; the
// default case list covers the df64 medium band + the rescaled deep band.
const ENGINE = process.env.ENGINE || 'rs';
const CASES = (process.env.BITS
  ? process.env.BITS.split(',').map((b) => ({ bits: Number(b), W: 160, H: 160, engine: ENGINE }))
  : [{ bits: 50, W: 160, H: 160, engine: 'df64' }, { bits: 120, W: 192, H: 192, engine: 'rs' },
     { bits: 218, W: 160, H: 160, engine: 'rs' }, { bits: 271, W: 144, H: 144, engine: 'rs' },
     { bits: 400, W: 112, H: 112, engine: 'rs' }]);
const REPS = Number(process.env.REPS || 11);
const ROUNDS = Number(process.env.ROUNDS || 5);

try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); });
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const init = await page.evaluate(() => window.__gpu.init(256, 256));
  if (!init.supported) throw new Error('GPU not supported');
  console.log('mode=' + gpuMode() + '  renderer:', JSON.parse(JSON.stringify(await page.evaluate(() => window.__gpu.info()))).renderer);
  console.log('\nsoftware-pipelined getZ prefetch vs production — rescaled engine, SA on');
  console.log('(min-of-min over interleaved rounds: the companion LLM shares this GPU;');
  console.log(' contention only ADDS time, so min converges to the true frame cost)\n');
  console.log('depth    engine size      maxIter | skip%  | pf ms   | base ms | base/pf | sn diff (MUST be 0)');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const eng = c.engine === 'df64' ? { df64: true } : c.engine === 'fe' ? { fe: true } : { rs: true };
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: REPS,
      ...eng, series: c.engine === 'rs' };
    let pms = Infinity, bms = Infinity, skip = 0;
    for (let rep = 0; rep < ROUNDS; rep++) {
      const p = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, prefetch: true });
      const b = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, prefetch: false });
      pms = Math.min(pms, p.min); bms = Math.min(bms, b.min); skip = p.saSkip;
    }
    const d = await page.evaluate((qq) => window.__gpu.crossArgs(qq.q, qq.a, qq.b),
      { q: base, a: { prefetch: true }, b: { prefetch: false } });
    const pct = (100 * skip / it).toFixed(1);
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${c.engine.padEnd(6)} ${String(c.W + 'x' + c.H).padEnd(9)} ${String(it).padStart(7)} | ` +
      `${pct.padStart(5)}% | ${pms.toFixed(2).padStart(7)} | ${bms.toFixed(2).padStart(7)} | ` +
      `${(bms / Math.max(0.01, pms)).toFixed(3).padStart(6)}× | ` +
      `${d.nDiff}/${d.total} (flips ${d.flips}, maxΔsn ${d.maxDsn.toExponential(1)})`);
  }
  console.log('\n(base/pf > 1 means prefetch is FASTER. sn diff must be 0 — same values, same math.)');
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
