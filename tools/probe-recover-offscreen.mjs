// probe-recover-offscreen.mjs — the OffscreenCanvas analogue of probe-recover.mjs (Spawn 24).
// Drives the REAL viewer on the DEFAULT (offscreen worker) GPU path through a WebGL context
// loss/restore cycle and confirms the mobile recovery still works when the GL context lives
// in the worker: (1) the worker forwards webglcontextlost → the viewer falls back to the CPU
// pool, (2) on restore the viewer renders on the GPU worker again with a non-blank image.
// The loss is simulated via the worker-side WEBGL_lose_context hook (gpuWorker.loseContextForTest).
//
//   node tools/probe-recover-offscreen.mjs        (SwiftShader)
//   GPU=1 node tools/probe-recover-offscreen.mjs  (real GPU)
import { chromium } from '@playwright/test';
import { chromiumArgs, resolveChromium } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8169;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const base = `http://127.0.0.1:${PORT}`;
async function ws() { for (let i = 0; i < 100; i++) { try { const r = await fetch(base + '/index.html'); if (r.ok) return; } catch { /* retry */ } await new Promise((r) => setTimeout(r, 100)); } throw new Error('no server'); }

let fail = 0;
const log = (ok, msg) => { console.log(`${ok ? 'OK  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };

try {
  await ws();
  const browser = await chromium.launch({ executablePath: resolveChromium(), args: chromiumArgs() });
  const page = await browser.newPage({ viewport: { width: 420, height: 760 } });
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail++; });
  await page.goto(base + '/index.html');           // default → offscreen worker path

  await page.waitForFunction(() => (window.__doneCount || 0) > 0, { timeout: 20000 });
  const capable = await page.evaluate(() => window.__viewer._offscreenCapable());
  if (!capable) { console.log('SKIP: OffscreenCanvas/WebGL2 unavailable here'); server.kill(); process.exit(0); }
  const home = await page.evaluate(() => window.__lastDone.engine);
  log(home === 'gpu-naive', `home engine = ${home} (expect gpu-naive)`);
  log(await page.evaluate(() => window.__viewer._useOffscreen()), 'offscreen worker is the active GPU path');

  // ---- Case A: lose the worker's context and DO NOT restore -> CPU fallback ----
  let prevDone = await page.evaluate(() => window.__doneCount || 0);
  await page.evaluate(() => window.__viewer.gpuWorker.loseContextForTest());
  await page.waitForFunction(() => window.__viewer.gpuWorker && window.__viewer.gpuWorker.lost, { timeout: 6000 });
  log(true, 'worker context-loss forwarded to the viewer (gpuWorker.lost)');
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 25000 });
  const lostEngine = await page.evaluate(() => window.__lastDone.engine);
  log(!String(lostEngine).startsWith('gpu-'), `while lost, rendered on CPU engine = ${lostEngine}`);

  // ---- Case B: restore -> GPU worker render returns, image non-blank ----
  prevDone = await page.evaluate(() => window.__doneCount || 0);
  await page.waitForTimeout(80);                    // tick between lose and restore
  await page.evaluate(() => window.__viewer.gpuWorker.restoreContextForTest());
  await page.waitForFunction(() => window.__viewer.gpuWorker && !window.__viewer.gpuWorker.lost, { timeout: 10000 });
  log(true, 'worker context restored (gpuWorker.lost cleared)');
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 25000 });
  const restoredEngine = await page.evaluate(() => window.__lastDone.engine);
  log(String(restoredEngine).startsWith('gpu-'), `after restore, rendered on GPU engine = ${restoredEngine}`);
  log(await page.evaluate(() => window.__viewer._useOffscreen()), 'offscreen worker is active again');

  const stats = await page.evaluate(() => {
    const c = document.getElementById('view'); const g = c.getContext('2d');
    const { data } = g.getImageData(0, 0, c.width, c.height);
    let max = 0; const seen = new Set();
    for (let i = 0; i < data.length; i += 4) { max = Math.max(max, data[i], data[i + 1], data[i + 2]); if (seen.size < 100) seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]); }
    return { max, distinct: seen.size };
  });
  log(stats.max > 0 && stats.distinct > 3, `recovered image non-blank (max ${stats.max}, distinct ${stats.distinct})`);

  await browser.close();
} catch (e) { console.error('ERROR', e); fail++; } finally { server.kill(); }
console.log(fail ? `\nRESULT: ${fail} FAIL` : '\nRESULT: ALL PASS');
process.exit(fail ? 1 : 0);
