// bench-interior-oracle.mjs — the UPPER-BOUND on interior detection (Spawn 23). Measure-first
// gate BEFORE committing to a multi-spawn attracting-cycle detector.
//
// At the deep minibrot coordinate, 58–90% of pixels are INTERIOR (run to maxIter). probe-interior
// also showed the ESCAPING pixels run ~81% of the same post-SA budget — they escape LATE — so on a
// GPU, where a warp runs until its slowest lane, pruning interior pixels only helps where whole
// warps are interior. This bench uses the CPU oracle to build an EXACT per-pixel interior mask,
// feeds it to the rescaled+SA shader, and times the frame while bailing oracle-interior pixels at
// n = skip + frac·(maxIter−skip):
//   frac = 1.0  -> baseline (mask fetch paid, prune never fires)
//   frac = 0.0  -> a PERFECT, FREE, zero-latency interior detector = the absolute CEILING
//   frac 0.5/0.25/0.1 -> "if a real detector fires by this fraction of the tail, you get this much"
// A real attracting-cycle detector is strictly WORSE than frac=0 (it has detection latency + a
// per-iteration tax + fires later than `skip`), so if frac=0 shows ~1.0×, interior detection is
// DEAD on this GPU and we lock in the current state. The 'nomask' column (uPrune=0, no fetch)
// isolates the mask-fetch overhead from the work actually saved.
//
//   GPU=1 node tools/bench-interior-oracle.mjs
//   GPU=1 BITS=120,271 REPS=15 node tools/bench-interior-oracle.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8154;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

// The genuine deep boundary minibrot coordinate (same as probe-interior / bench-samargin).
const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));
const CASES = (process.env.BITS
  ? process.env.BITS.split(',').map((b) => ({ bits: Number(b), W: 160, H: 160 }))
  : [{ bits: 120, W: 192, H: 192 }, { bits: 218, W: 160, H: 160 },
     { bits: 271, W: 144, H: 144 }, { bits: 400, W: 112, H: 112 }]);
const FRACS = (process.env.FRACS || '1.0,0.5,0.25,0.1,0.0').split(',').map(Number);
const REPS = Number(process.env.REPS || 11);

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
  console.log('\nINTERIOR-PRUNE upper bound — rescaled+SA(order5), oracle interior mask, real wall-clock');
  console.log('frac = fraction of the post-SA tail (skip→maxIter) interior pixels are allowed to run');
  console.log('frac=1.0 baseline (never fires) · frac=0.0 = perfect free detector = CEILING\n');
  const hdr = FRACS.map((f) => `f=${f.toFixed(2)}`).join(' | ');
  console.log(`depth   size     int%  nomask | ${hdr} | ceiling(f=0)  flips/escDiff`);
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const q = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: REPS,
                series: { order: 5 }, fracs: FRACS };
    let o = null;
    for (let rep = 0; rep < 2; rep++) {                  // best-of-2 over the whole point-set
      const x = await page.evaluate((qq) => window.__gpu.benchPrune(qq), q);
      if (!o) o = x; else { o.points.forEach((p, k) => { p.ms = Math.min(p.ms, x.points[k].ms); }); o.nomask = Math.min(o.nomask, x.nomask); }
    }
    const baseMs = o.points.find((p) => p.frac === 1.0)?.ms ?? o.points[0].ms;
    const cells = o.points.map((p) => p.ms.toFixed(2).padStart(6)).join(' | ');
    const ceil = (baseMs / Math.max(0.01, o.points.find((p) => p.frac === 0.0)?.ms ?? baseMs));
    console.log(
      `2^-${String(c.bits).padEnd(3)} ${String(c.W + 'x' + c.H).padEnd(8)} ${(100 * o.interior).toFixed(0).padStart(3)}% ${o.nomask.toFixed(2).padStart(6)} | ` +
      `${cells} | ${ceil.toFixed(2)}×        ${o.insideFlip}/${o.escDiff}`);
  }
  console.log('\nflips/escDiff = baseline-vs-max-prune sn diff (mask alignment + correctness crosscheck;');
  console.log('flips are bounded by the CPU/GPU inside-classification boundary, escDiff must be 0).');
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
