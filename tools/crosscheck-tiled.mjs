// crosscheck-tiled.mjs — prove the strip-tiled deep render (the fix for "can't zoom
// past ~2^218") is BIT-IDENTICAL to a single full-frame draw. Renders the same view
// once as one draw, then again split into horizontal scissor strips, and asserts the
// full sn/iter/glitch buffers match exactly (iterDiff=0, snDiff=0, maxAbs=0). The
// scissor restricts WHICH rows are written but keeps gl_FragCoord global, so tiling
// must not change a single pixel. Covers df64 + floatexp + rescaled across depths
// (incl. the chaotic seahorse valley and below the float32 floor), and a few strip
// heights (incl. 1-row strips and a strip larger than the frame = the single-draw case).
//
//   node tools/crosscheck-tiled.mjs
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

const PORT = process.env.PORT || 8148;
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
const CASES = [
  { tag: 'naive 2^-1  h=7',  re: '-0.5', im: '0', bits: 1,  W: 80, H: 80, it: 800,  stripH: 7 },
  { tag: 'df64 2^-50  h=13', re: SH, im: SI, bits: 50,  df64: true, W: 80, H: 80, it: 4000, stripH: 13 },
  { tag: 'df64 2^-90  h=1',  re: SH, im: SI, bits: 90,  df64: true, W: 64, H: 64, it: 6000, stripH: 1 },
  { tag: 'fe   2^-150 h=11', re: SH, im: SI, bits: 150, fe: true,   W: 64, H: 64, it: 6000, stripH: 11 },
  { tag: 'fe   2^-270 h=9',  re: SH, im: SI, bits: 270, fe: true,   W: 64, H: 64, it: 6000, stripH: 9 },
  { tag: 'rs   2^-90  h=8',  re: SH, im: SI, bits: 90,  rs: true,   W: 80, H: 80, it: 6000, stripH: 8 },
  { tag: 'rs   2^-150 h=1',  re: SH, im: SI, bits: 150, rs: true,   W: 56, H: 56, it: 6000, stripH: 1 },
  { tag: 'rs   2^-218 h=13', re: SH, im: SI, bits: 218, rs: true,   W: 64, H: 64, it: 6000, stripH: 13 },
  { tag: 'rs   2^-270 h=99', re: SH, im: SI, bits: 270, rs: true,   W: 64, H: 64, it: 6000, stripH: 99 }, // > H (one strip)
  { tag: 'rs   2^-130 exter', re: '0.36', im: '0.09', bits: 130, rs: true, W: 64, H: 64, it: 1200, stripH: 10 },
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
  if (!(await page.evaluate(() => window.__gpu.init(256, 256))).supported) throw new Error('GPU not supported');
  console.log('renderer:', (await page.evaluate(() => window.__gpu.info())).renderer);
  console.log('');
  console.log('case                 size    maxIter stripH refLen iterDiff snDiff glitchDiff maxAbs   verdict');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits;
    const q = { re: c.re, im: c.im, radius: r, maxIter: c.it, width: c.W, height: c.H,
                df64: !!c.df64, fe: !!c.fe, rs: !!c.rs, stripH: c.stripH };
    const d = await page.evaluate((qq) => window.__gpu.crossCheckTiled(qq), q);
    const ok = d.iterDiff === 0 && d.snDiff === 0 && d.glitchDiff === 0 && d.maxAbs === 0;
    if (!ok) anyFail = true;
    console.log(
      `${c.tag.padEnd(19)} ${String(c.W + 'x' + c.H).padEnd(7)} ${String(c.it).padStart(7)} ` +
      `${String(d.stripH).padStart(6)} ${String(d.refLen).padStart(6)} ${String(d.iterDiff).padStart(8)} ` +
      `${String(d.snDiff).padStart(6)} ${String(d.glitchDiff).padStart(9)}  ${d.maxAbs.toExponential(1).padStart(7)}  ${ok ? 'IDENTICAL' : 'MISMATCH!!'}`
    );
  }
  await browser.close();
  console.log('');
  console.log(anyFail ? 'FAIL: strip-tiling changed pixels — scissor/viewport bug' : 'PASS: strip-tiled render is bit-identical to a single draw');
} catch (e) {
  console.error('ERROR:', e.stack || e);
  anyFail = true;
} finally {
  server.kill();
}
process.exit(anyFail ? 1 : 0);
