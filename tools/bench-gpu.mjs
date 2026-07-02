// bench-gpu.mjs — time the GPU perturbation shaders (df64 + floatexp) on this
// host's GL (SwiftShader here) across a range of depths, so shader optimizations
// can be measured, not guessed. Pure GPU render time: a draw + gl.finish() per
// frame (no CPU oracle, no readback-to-stats). Prints median ms / frame and a
// derived throughput (Miter-pixels/s = width*height*maxIter / ms / 1e3).
//
//   node tools/bench-gpu.mjs
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8141;
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
// Seahorse-valley cases exercise the real chaotic escape work (the expensive path).
// Sizes/iters kept modest so the heavy fe shader finishes in reasonable time on
// SwiftShader. Each row is one (engine, depth) bench point.
// Each (depth) is benched on BOTH the floatexp ("fe") and rescaled ("rs") engines
// so the fe->rs speedup is a within-run, matched-load ratio.
const CASES = [
  { tag: '2^-80 ', re: SH, im: SI, bits: 80,  W: 128, H: 128, it: 6000 },
  { tag: '2^-150', re: SH, im: SI, bits: 150, W: 128, H: 128, it: 6000 },
  { tag: '2^-270', re: SH, im: SI, bits: 270, W: 96,  H: 96,  it: 6000 },
];

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
  console.log('');
  // floatexp ("fe") vs rescaled ("rs") on the SAME view (matched-load within-run ratio).
  console.log('case     size     maxIter  refLen   fe ms    rs ms   fe Mit/s  rs Mit/s  speedup');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits;
    const base = { re: c.re, im: c.im, radius: r, maxIter: c.it, width: c.W, height: c.H, reps: 3 };
    const fe = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, fe: true });
    const rs = await page.evaluate((qq) => window.__gpu.benchPerturb(qq), { ...base, rs: true });
    const thru = (ms) => (c.W * c.H * c.it) / ms / 1e3; // Mit-px/s
    console.log(
      `${c.tag}  ${String(c.W + 'x' + c.H).padEnd(8)} ${String(c.it).padStart(7)}  ` +
      `${String(fe.refLen).padStart(6)}  ${fe.ms.toFixed(1).padStart(7)} ${rs.ms.toFixed(1).padStart(7)}  ` +
      `${thru(fe.ms).toFixed(1).padStart(8)}  ${thru(rs.ms).toFixed(1).padStart(8)}  ${(fe.ms / rs.ms).toFixed(2).padStart(5)}x`
    );
  }
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e);
} finally {
  server.kill();
}
