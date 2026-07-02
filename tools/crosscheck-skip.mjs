// crosscheck-skip.mjs — prove the perturb fast-skip (uFastSkip) is BIT-IDENTICAL
// to the exact escape/rebase path. Renders the same deep view twice (skip on/off)
// and asserts the full sn/iter/glitch buffers match exactly (maxAbs=0, no diffs).
// The skip only elides provably-inert iterations, so ANY nonzero diff is a bug in
// the skip bound. Covers df64 + floatexp across depths incl. the chaotic seahorse
// valley (where rebasing is frequent — the case most likely to expose a bad bound).
//
//   node tools/crosscheck-skip.mjs
import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';
import { readdirSync, existsSync } from 'node:fs';

function resolveChromium() {
  if (process.env.CHROMIUM_PATH && existsSync(process.env.CHROMIUM_PATH)) return process.env.CHROMIUM_PATH;
  const dirs = readdirSync('/nix/store').filter((d) => /-chromium-\d/.test(d) && !d.includes('sandbox')).sort().reverse();
  for (const d of dirs) { const p = `/nix/store/${d}/bin/chromium`; if (existsSync(p)) return p; }
  return undefined;
}

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

const SH = '-0.743643887037158704752191506114774', SI = '0.131825904205311970493132056385139';
// Seahorse valley = chaotic escapes + frequent rebasing (the bound's worst case).
const CASES = [
  { tag: 'df64 2^-20 valley', re: SH, im: SI, bits: 20,  df64: true, W: 96, H: 96, it: 2000 },
  { tag: 'df64 2^-50 valley', re: SH, im: SI, bits: 50,  df64: true, W: 96, H: 96, it: 4000 },
  { tag: 'df64 2^-90 valley', re: SH, im: SI, bits: 90,  df64: true, W: 96, H: 96, it: 6000 },
  { tag: 'fe   2^-80 valley', re: SH, im: SI, bits: 80,  fe: true,   W: 96, H: 96, it: 6000 },
  { tag: 'fe   2^-150 valley', re: SH, im: SI, bits: 150, fe: true,  W: 96, H: 96, it: 6000 },
  { tag: 'fe   2^-270 valley', re: SH, im: SI, bits: 270, fe: true,  W: 96, H: 96, it: 6000 },
  { tag: 'rs   2^-80 valley', re: SH, im: SI, bits: 80,  rs: true,   W: 96, H: 96, it: 6000 },
  { tag: 'rs   2^-150 valley', re: SH, im: SI, bits: 150, rs: true,  W: 96, H: 96, it: 6000 },
  { tag: 'rs   2^-270 valley', re: SH, im: SI, bits: 270, rs: true,  W: 96, H: 96, it: 6000 },
  { tag: 'rs   2^-130 exter', re: '0.36', im: '0.09', bits: 130, rs: true, W: 64, H: 64, it: 1200 },
  { tag: 'rs   2^-270 exter', re: '0.36', im: '0.09', bits: 270, rs: true, W: 64, H: 64, it: 1200 },
];

let anyFail = false;
try {
  await waitServer();
  const browser = await chromium.launch({
    executablePath: resolveChromium(),
    args: chromiumArgs(),
  });
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); });
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const init = await page.evaluate(() => window.__gpu.init(256, 256));
  if (!init.supported) throw new Error('GPU not supported');
  console.log('renderer:', (await page.evaluate(() => window.__gpu.info())).renderer);
  console.log('');
  console.log('case                  size    maxIter  refLen   iterDiff snDiff glitchDiff maxAbs   verdict');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits;
    const q = { re: c.re, im: c.im, radius: r, maxIter: c.it, width: c.W, height: c.H,
                df64: !!c.df64, fe: !!c.fe };
    const d = await page.evaluate((qq) => window.__gpu.crossCheckSkip(qq), q);
    const ok = d.iterDiff === 0 && d.snDiff === 0 && d.glitchDiff === 0 && d.maxAbs === 0;
    if (!ok) anyFail = true;
    console.log(
      `${c.tag.padEnd(20)} ${String(c.W + 'x' + c.H).padEnd(7)} ${String(c.it).padStart(7)}  ` +
      `${String(d.refLen).padStart(6)}   ${String(d.iterDiff).padStart(7)} ${String(d.snDiff).padStart(6)} ` +
      `${String(d.glitchDiff).padStart(9)}  ${d.maxAbs.toExponential(1).padStart(7)}  ${ok ? 'IDENTICAL' : 'MISMATCH!!'}`
    );
  }
  await browser.close();
  console.log('');
  console.log(anyFail ? 'FAIL: fast-skip is NOT bit-identical — bound is wrong' : 'PASS: fast-skip is bit-identical to the exact path');
} catch (e) {
  console.error('ERROR:', e.stack || e);
  anyFail = true;
} finally {
  server.kill();
}
process.exit(anyFail ? 1 : 0);
