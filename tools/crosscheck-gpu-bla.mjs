// crosscheck-gpu-bla.mjs — A/B the GPU BLA against the GPU no-BLA rescaled engine on the
// genuine deep boundary coordinate. The no-BLA rescaled engine is already validated vs the
// CPU oracle (validate-gpu.mjs), so this ISOLATES the GPU BLA contribution: BLA must not
// change the picture beyond its small truncation drift (it drops dz²). Unlike crosscheck-skip
// (which gates 0-diff, the skip being bit-identical) this gates a SMALL bounded drift matching
// the CPU crosscheck-bla / arbiter-bla envelope (a few near-maxIter pixels off by ≤~19).
//
//   node tools/crosscheck-gpu-bla.mjs
//   GPU=1 node tools/crosscheck-gpu-bla.mjs           # the real GPU (the gate that matters)
//   BITS=120,271,400 EPS=30 SERIES=1 node tools/crosscheck-gpu-bla.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8145;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const BITS = (process.env.BITS || '120,271,400').split(',').map(Number);
const EPS = 2 ** -Number(process.env.EPS || 30);
const SERIES = process.env.SERIES !== '0';     // compose SA by default (the production path)
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));
const SIZE = { 120: 96, 271: 72, 400: 56 };

let failed = 0;
try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); failed++; });
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const init = await page.evaluate(() => window.__gpu.init(256, 256));
  if (!init.supported) throw new Error('GPU not supported');
  console.log('renderer:', (await page.evaluate(() => window.__gpu.info())).renderer);
  console.log(`BLA on-vs-off drift (deep boundary coordinate)  eps 2^${Math.round(Math.log2(EPS))}  SA ${SERIES ? 'on' : 'off'}\n`);
  console.log('depth    size    refLen  lvls  | iterDiff  (≤2)  maxΔn  insMism  meanΔsn   verdict');

  for (const bits of BITS) {
    const r = 1.5 * 2 ** -bits, it = autoIter(bits), S = SIZE[bits] || 64;
    const q = { re: RE, im: IM, radius: r, maxIter: it, width: S, height: S,
                rs: true, bla: { eps: EPS }, series: SERIES };
    const d = await page.evaluate((qq) => window.__gpu.crossCheckBla(qq), q);
    // Gate: drift is BLA truncation, must be a small bulk fraction AND never a missed escape
    // (a missed escape flips by hundreds; cap maxΔn). Mirrors crosscheck-bla's bulk gate.
    const frac = d.iterDiff / Math.max(1, d.n);
    const ok = frac <= 0.02 && d.insideDiff <= 4 && d.maxIterDiff <= 60;
    if (!ok) failed++;
    console.log(
      `2^-${String(bits).padEnd(4)} ${String(S + 'x' + S).padEnd(7)} ${String(d.refLen).padStart(6)}  ` +
      `${String(d.blaMaxLevel).padStart(4)}  | ${String(d.iterDiff).padStart(7)} ${String(d.near).padStart(5)} ` +
      `${String(d.maxIterDiff).padStart(6)} ${String(d.insideDiff).padStart(7)}  ${d.meanAbs.toExponential(2)}  ` +
      `${ok ? 'PASS' : 'FAIL'} (${(frac * 100).toFixed(3)}%)`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
  failed++;
} finally {
  server.kill();
}
console.log(`\n${failed ? 'FAILURES: ' + failed : 'GPU BLA drift within envelope — PASS'}`);
process.exit(failed ? 1 : 0);
