// bench-sqr.mjs — wall-clock A/B for the dedicated df64 SQUARING (Spawn 27).
//
// ds_sqr/fe_sqr (q.sqrOn) replace ds_mul(a,a)/fe_mul(a,a) at the ~6 self-product
// sites of the per-iteration loop (update quad term + escape-test magnitudes): ONE
// Veltkamp split instead of two, 6 barriers vs 8, ~3 fewer mults. MEASURED VERDICT
// (Spawn 27): 0.999–1.031× full-shader on the real RTX 3090 (the driver CSEs the
// duplicate split — isolated per-op is 1.00× too) and 0.95–0.98× on SwiftShader →
// ds_sqr is OPT-IN (p.sqrOn), NOT the production default. This bench is the gate to
// re-decide on new hardware. Reports the sn diff between the two shaders (NOT
// required to be bit-identical — ~1-ulp low-word scheduling freedom — the
// correctness gate is validate:gpu[:real]).
//
//   GPU=1 node tools/bench-sqr.mjs
//   GPU=1 BITS=120,218,271,400 node tools/bench-sqr.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

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
// deep cases run the rescaled engine (the production deep path); the 50-bit case
// runs the plain-df64 engine (the medium band — it has the MOST squares/iter: 6).
const CASES = (process.env.BITS
  ? process.env.BITS.split(',').map((b) => ({ bits: Number(b), W: 160, H: 160 }))
  : [{ bits: 50, W: 192, H: 192, df64: true }, { bits: 120, W: 192, H: 192 },
     { bits: 218, W: 160, H: 160 }, { bits: 271, W: 144, H: 144 }, { bits: 400, W: 112, H: 112 }]);
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
  console.log('mode=' + gpuMode() + '  renderer:', JSON.parse(JSON.stringify(await page.evaluate(() => window.__gpu.info()))).renderer);
  console.log('\nds_sqr (dedicated df64 squaring) vs ds_mul(a,a) — production config (SA on, deep=rescaled)');
  console.log('depth    engine size      maxIter | skip%  | sqr ms  | mul ms  | mul/sqr | sn diff (of total)');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const engine = c.df64 ? { df64: true } : { rs: true };
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: REPS,
      ...engine, series: !c.df64 };
    // Interleave MANY rounds and take the MIN-of-min: the companion LLM shares this
    // GPU, and its inference bursts contaminate medians asymmetrically. Contention
    // only ever ADDS time, so the min over rounds converges to the true frame cost.
    const ROUNDS = Number(process.env.ROUNDS || 4);
    let sms = Infinity, mms = Infinity, skip = 0;
    for (let rep = 0; rep < ROUNDS; rep++) {
      const s = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, sqrOn: true });
      const m = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, sqrOn: false });
      sms = Math.min(sms, s.min); mms = Math.min(mms, m.min); skip = s.saSkip;
    }
    const d = await page.evaluate((qq) => window.__gpu.crossSqr(qq), base);
    const pct = (100 * skip / it).toFixed(1);
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${(c.df64 ? 'df64' : 'rs').padEnd(6)} ${String(c.W + 'x' + c.H).padEnd(9)} ${String(it).padStart(7)} | ` +
      `${pct.padStart(5)}% | ${sms.toFixed(2).padStart(7)} | ${mms.toFixed(2).padStart(7)} | ` +
      `${(mms / Math.max(0.01, sms)).toFixed(3).padStart(6)}× | ` +
      `${d.nDiff}/${d.total} (flips ${d.flips}, maxΔsn ${d.maxDsn.toExponential(1)})`);
  }
  console.log('\n(mul/sqr > 1 means the dedicated squaring is FASTER. sn diff is informational —');
  console.log(' the two shaders differ by ~1 ulp of the df64 low word; validate:gpu[:real] is the gate.)');
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
