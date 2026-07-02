// arbiter-gpu.mjs — for GPU-df64 vs CPU-double naive mismatches, consult the
// BigInt-exact oracle to classify each as chaos (both finite-precision wrong) or
// a genuine GPU bug (GPU wrong where double is right).
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
const PORT = process.env.PORT || 8141;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) { try { const r = await fetch(baseURL + '/test/gpu/harness.html'); if (r.ok) return; } catch {} await new Promise((r) => setTimeout(r, 100)); }
  throw new Error('server did not start');
}
try {
  await waitServer();
  const browser = await chromium.launch({ executablePath: resolveChromium(),
    args: chromiumArgs() });
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
  await page.goto(baseURL + '/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const W = 120, H = 120;
  await page.evaluate(([w, h]) => window.__gpu.init(w, h), [W, H]);

  for (const [name, cx, cy, r, it] of [
    ['df64 2^-18', -0.745, 0.113, 1.5 * 2 ** -18, 1200],
    ['df64 2^-14', -0.745, 0.113, 1.5 * 2 ** -14, 800],
    ['f32  2^0   ', -0.5, 0.0, 1.5, 300],
  ]) {
    const scale = (2 * r) / H;
    const p = { ox: cx - r * (W / H), oy: cy - r, scale, maxIter: it, width: W, height: H,
                df64: name.startsWith('df64'), checkStep: 1, arbPrec: 120 };
    const res = await page.evaluate((pp) => window.__gpu.arbitrateNaive(pp), p);
    console.log(`${name}: mism=${res.mism} chaos(both wrong)=${res.chaosBoth} ` +
      `gpu==big=${res.gpuMatchesBig} dbl==big=${res.doubleMatchesBig} ` +
      `GPU-UNIQUELY-WRONG=${res.gpuUniquelyWrong}`);
    if (res.gpuUniquelyWrong) console.log('   examples:', JSON.stringify(res.examples.slice(0, 5)));
  }
  await browser.close();
} catch (e) { console.error('ERROR:', e.stack || e); } finally { server.kill(); }
