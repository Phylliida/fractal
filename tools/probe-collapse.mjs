// probe-collapse.mjs — render the REAL perturbFragDf64 and the f32 perturb shader
// on the same view and measure how much they disagree. df64 intact => they differ
// on the chaotic boundary; df64 collapsed-to-f32 => nearly identical.
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8157;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}
const SH = '-0.743643887037158704752191506114774', SI = '0.131825904205311970493132056385139';
const CASES = [
  { tag: '2^-22', bits: 22, it: 5900 },
  { tag: '2^-50', bits: 50, it: 12900 },
];
await waitServer();
const browser = await chromium.launch(launchOpts());
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  await page.evaluate(() => window.__gpu.init(160, 160));
  console.log(`mode=${gpuMode()}  renderer:`, (await page.evaluate(() => window.__gpu.info())).renderer);
  console.log('\nIf df64 collapsed to f32, df64 and f32 engines AGREE (mismFrac ~ 0).');
  console.log('If df64 intact, they disagree on the chaotic boundary (mismFrac > 0).\n');
  for (const c of CASES) {
    const r = 1.5 * 2 ** -c.bits;
    const q = { re: SH, im: SI, radius: r, maxIter: c.it, width: 160, height: 160 };
    const res = await page.evaluate((qq) => window.__gpu.compareDf64VsF32(qq), q);
    console.log(`${c.tag}: df64-vs-f32 iterMismFrac=${(res.iterMismFrac*100).toFixed(2)}%  maxDiff=${res.maxDiff}  ${res.iterMismFrac < 0.01 ? '=> df64 COLLAPSED (== f32)' : '=> df64 INTACT (differs from f32)'}`);
  }
} finally {
  await browser.close();
  server.kill();
}
