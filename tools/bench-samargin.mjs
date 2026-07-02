// bench-samargin.mjs — wall-clock win of a SMALLER SA safety margin at the deep band (Spawn 22).
// probe-samargin.mjs proved margin 0.05 → 0.01 stays BIT-EXACT (mism 0, identical drift) on the
// real GPU while cutting post-SA iterations 27–39%. Margin reduction is a UNIFORM per-pixel
// iteration cut (every pixel skips more), so its benefit is robust to throughput-vs-divergence.
// This confirms the post-SA cut becomes real wall-clock. Run with GPU=1.
//
//   GPU=1 node tools/bench-samargin.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

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
const CASES = (process.env.BITS
  ? process.env.BITS.split(',').map((b) => ({ bits: Number(b), W: 160, H: 160 }))
  : [{ bits: 120, W: 192, H: 192 }, { bits: 218, W: 160, H: 160 },
     { bits: 271, W: 144, H: 144 }, { bits: 400, W: 112, H: 112 }]);
const MARGINS = (process.env.MARGINS || '0.05,0.02,0.01').split(',').map(Number);
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
  console.log(`\nrescaled+SA(order5) deep render — wall-clock vs SA margin (baseline ${MARGINS[0]})\n`);
  const hdr = MARGINS.map((m) => `m=${m} ms`).join(' | ');
  console.log(`depth    size      maxIter | ${hdr} | speedup vs m=${MARGINS[0]}`);
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits, it = autoIter(c.bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: c.W, height: c.H, reps: REPS, rs: true };
    const res = [];
    for (const mg of MARGINS) {
      let ms = Infinity, skip = 0;
      for (let rep = 0; rep < 2; rep++) {
        const o = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, series: { order: 5, marginFrac: mg } });
        ms = Math.min(ms, o.ms); skip = o.saSkip;
      }
      res.push({ mg, ms, skip });
    }
    const b = res[0].ms;
    const speed = res.slice(1).map((x) => `m=${x.mg}: ${(b / Math.max(0.01, x.ms)).toFixed(3)}×`).join('  ');
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${String(c.W + 'x' + c.H).padEnd(9)} ${String(it).padStart(7)} | ` +
      res.map((x) => x.ms.toFixed(2).padStart(7)).join(' | ') + ` | ${speed}`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
