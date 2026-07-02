// probe-xbackend.mjs — render the SAME df64 perturbation view on SwiftShader AND
// on the real GPU and compare the iteration buffers pixel-for-pixel. Both run the
// identical df64 algorithm; if the GPU's df64 is bit-faithful they should match
// almost exactly (the chaotic boundary may differ by a tiny fraction from
// genuinely different last-bit rounding, but NOT by tens of percent).
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { resolveChromium } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8158;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch {}
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}
const SWIFT = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu','--enable-unsafe-swiftshader','--headless=new'];
const VULKAN = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--use-angle=vulkan','--use-gl=angle','--enable-features=Vulkan','--ignore-gpu-blocklist','--enable-gpu','--headless=new'];

const SH = '-0.743643887037158704752191506114774', SI = '0.131825904205311970493132056385139';
const W = 160, H = 160;

async function renderOn(args, q, engine) {
  const browser = await chromium.launch({ executablePath: resolveChromium(), args });
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
    await page.goto(baseURL + '/test/gpu/harness.html');
    await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
    await page.evaluate(([w,h]) => window.__gpu.init(w,h), [W, H]);
    const renderer = (await page.evaluate(() => window.__gpu.info())).renderer;
    const res = await page.evaluate((qq) => window.__gpu.renderIter(qq), { ...q, [engine]: true });
    return { renderer, iter: res.iter, sn: res.sn };
  } finally { await browser.close(); }
}

await waitServer();
try {
  for (const fastSkip of [1, 0]) {
    console.log(`\n--- fastSkip=${fastSkip} ---`);
    for (const [tag, bits, it] of [['2^-22',22,5900],['2^-50',50,12900]]) {
      const q = { re: SH, im: SI, radius: 1.5 * 2 ** -bits, maxIter: it, width: W, height: H, fastSkip };
      const swift = await renderOn(SWIFT, q, 'df64');
      const gpu = await renderOn(VULKAN, q, 'df64');
      let n = 0, diff = 0, maxD = 0, sumD = 0;
      for (let i = 0; i < swift.iter.length; i++) {
        n++; const d = Math.abs(swift.iter[i] - gpu.iter[i]);
        if (d > 0) diff++; if (d > maxD) maxD = d; sumD += d;
      }
      console.log(`${tag}: df64 SwiftShader vs GPU  mismFrac=${(100*diff/n).toFixed(2)}%  meanΔiter=${(sumD/n).toFixed(2)}  maxΔiter=${maxD}`);
    }
  }
} finally { server.kill(); }
