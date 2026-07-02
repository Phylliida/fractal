// probe-rescaled.mjs — validate the rescaled perturbation engine: (1) vs the CPU
// perturbation oracle (must match fe's tolerance), (2) direct agreement with the fe
// engine on the same view. Run after editing perturbFragRescaled.
//   node tools/probe-rescaled.mjs
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

const PORT = process.env.PORT || 8144;
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
// Overlap band (real chaotic escapes) + below-floor exterior, mirroring validate-gpu's fe section.
const ORACLE = [
  { tag: 'rs seahorse 2^-70 ', re: SH, im: SI, bits: 70, W: 64, H: 64, it: 20000, step: 1 },
  { tag: 'rs seahorse 2^-90 ', re: SH, im: SI, bits: 90, W: 64, H: 64, it: 26000, step: 1 },
  { tag: 'rs exterior 2^-150', re: SH, im: SI, bits: 150, W: 48, H: 48, it: 10000, step: 1 },
  { tag: 'rs exterior 2^-270', re: SH, im: SI, bits: 270, W: 48, H: 48, it: 10000, step: 1 },
];
const AGREE = [
  { tag: 'rs~fe 2^-90 it=8k ', re: SH, im: SI, bits: 90, W: 64, H: 64, it: 8000 },
  { tag: 'rs~fe 2^-90 it=15k', re: SH, im: SI, bits: 90, W: 64, H: 64, it: 15000 },
  { tag: 'rs~fe 2^-90 it=26k', re: SH, im: SI, bits: 90, W: 64, H: 64, it: 26000 },
  { tag: 'rs~fe 2^-70 it=20k', re: SH, im: SI, bits: 70, W: 64, H: 64, it: 20000 },
];

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

  console.log('\n== vs CPU perturbation oracle (rescaled vs floatexp, identical params) ==');
  for (const c of ORACLE) {
    const r = 1.5 * 2 ** -c.bits;
    const base = { re: c.re, im: c.im, radius: r, maxIter: c.it, width: c.W, height: c.H,
                   checkStep: c.step, glitchTol: 0 };
    const dr = await page.evaluate((qq) => window.__gpu.comparePerturb(qq), { ...base, rs: true });
    const df = await page.evaluate((qq) => window.__gpu.comparePerturb(qq), { ...base, fe: true });
    const esc = Math.max(1, dr.compared - dr.interiorCount);
    const rp = 100 * dr.mism / esc, fp = 100 * df.mism / esc;
    // rescaled is OK iff it is no worse than fe (within a hair) — fe is the validated reference.
    const ok = (rp <= fp + 0.05) && (dr.insideMismatch <= df.insideMismatch + 1);
    if (!ok) fail = true;
    console.log(`${c.tag} ref=${String(dr.refLen).padStart(5)} esc=${String(dr.compared - dr.interiorCount).padStart(4)}  ` +
      `rs mism=${rp.toFixed(3).padStart(6)}%  fe mism=${fp.toFixed(3).padStart(6)}%  ` +
      `rs maxΔsn=${dr.maxAbs.toExponential(2)}  ${ok ? 'PASS (<=fe)' : 'FAIL (worse than fe)'}`);
  }

  console.log('\n== rescaled vs floatexp agreement (same view) ==');
  for (const c of AGREE) {
    const r = 1.5 * 2 ** -c.bits;
    const q = { re: c.re, im: c.im, radius: r, maxIter: c.it, width: c.W, height: c.H };
    const d = await page.evaluate((qq) => window.__gpu.compareRescaledVsFe(qq), q);
    const pct = (100 * d.iterDiff / Math.max(1, d.n));
    const ok = pct < 2.0 && d.insideDiff <= 3;
    if (!ok) fail = true;
    console.log(`${c.tag} ref=${String(d.refLen).padStart(5)} iterDiff=${String(d.iterDiff).padStart(4)} ` +
      `(${pct.toFixed(3)}%) near≤2=${String(d.near).padStart(4)} insideDiff=${String(d.insideDiff).padStart(3)} ` +
      `meanΔsn=${d.meanAbs.toExponential(2)} maxΔsn=${d.maxAbs.toExponential(2)}  ${ok ? 'PASS' : 'FAIL'}`);
  }

  await browser.close();
  console.log('\n' + (fail ? 'FAIL: rescaled engine has a problem' : 'PASS: rescaled engine validates'));
} catch (e) {
  console.error('ERROR:', e.stack || e);
  fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
