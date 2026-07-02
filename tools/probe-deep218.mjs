// probe-deep218.mjs — measure what actually happens to the GPU deep perturbation
// engines (rescaled + floatexp) at CHAOTIC high-iteration pixels BELOW the float32
// floor, around and past 2^-218. The existing validate-gpu/probe-rescaled only test
// chaotic escapes to 2^-90 and SMOOTH (low-iter) exterior patches at 2^-150..-340 —
// nobody has measured chaotic, high-maxIter correctness deep. That is exactly the
// regime the user hits when "zooming past 2^218".
//
// Uses a REAL deep coordinate (Danielle's "ultra" 2^-271 location) so the pixels are
// genuinely on the boundary and escape at high counts. Compares the GPU engine vs the
// CPU perturbation oracle (escapePerturb, 53-bit doubles, validated to 2^400 vs BigInt).
// NOTE: the GPU and the CPU oracle share the SAME 53-bit reference orbit, so this
// isolates the df64(46-bit)-delta gap. It also times the reference build (the other
// candidate for "feels broken deep").
//   node tools/probe-deep218.mjs
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

const PORT = process.env.PORT || 8146;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

// Danielle's "ultra" deep location (~115 sig digits ≈ 380 bits; real boundary point).
const RE = '-1.369078017863660784890619576747781310848768032841633323730495873496232879296538490243106365484246242476783355722';
const IM = '-0.071817675972918479944583194368632476442138106251769795140812120871593742404751576456750164324645880810732436640';
// ~250 iters/octave is the viewer's autoMaxIter; use it so the test matches reality.
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

// Increasing depth across and past the reported 2^218 wall. 40×40 chaotic grid.
const CASES = [
  { bits: 90 },   // above the float32 floor (df64 would also work) — baseline
  { bits: 120 },  // just below the df64 floor
  { bits: 150 },
  { bits: 190 },
  { bits: 218 },  // the reported wall
  { bits: 250 },
  { bits: 271 },  // the "ultra" reference depth
];
const W = 40, H = 40;

let fail = false;
try {
  await waitServer();
  const browser = await chromium.launch({
    executablePath: resolveChromium(),
    args: chromiumArgs(),
  });
  const page = await browser.newPage();
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail = true; });
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  if (!(await page.evaluate(() => window.__gpu.init(256, 256))).supported) throw new Error('GPU not supported');
  console.log('renderer:', (await page.evaluate(() => window.__gpu.info())).renderer);
  console.log(`grid ${W}x${H}, chaotic deep coordinate, maxIter = autoMaxIter(bits)\n`);
  console.log('depth     maxIter ref  refMs  esc  | rs mism   fe mism  | rs maxΔsn  rs insideMism');

  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits;
    const it = autoIter(c.bits);
    const base = { re: RE, im: IM, radius: r, maxIter: it, width: W, height: H, checkStep: 1, glitchTol: 0 };
    const t0 = Date.now();
    const dr = await page.evaluate((qq) => window.__gpu.comparePerturb(qq), { ...base, rs: true });
    const rsMs = Date.now() - t0;
    const df = await page.evaluate((qq) => window.__gpu.comparePerturb(qq), { ...base, fe: true });
    const esc = Math.max(1, dr.compared - dr.interiorCount);
    const rp = 100 * dr.mism / esc, fp = 100 * df.mism / esc;
    console.log(
      `2^-${String(c.bits).padEnd(4)} ${String(it).padStart(7)} ${String(dr.refLen).padStart(5)} ` +
      `${String(rsMs).padStart(5)} ${String(esc).padStart(4)}  | ${rp.toFixed(3).padStart(7)}% ${fp.toFixed(3).padStart(7)}% ` +
      `| ${dr.maxAbs.toExponential(2)}  ${dr.insideMismatch}`);
  }

  await browser.close();
  console.log('\n(measurement only — rs vs CPU-oracle shares the 53-bit reference, so this is the df64-delta gap)');
} catch (e) {
  console.error('ERROR:', e.stack || e);
  fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
