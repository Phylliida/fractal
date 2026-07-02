// probe-recover.mjs — drive the REAL viewer through a WebGL context loss/restore
// cycle and confirm: (1) while lost it renders on the CPU pool, (2) once restored it
// renders on the GPU again with a non-blank image. This is the end-to-end gate for
// the mobile context-loss recovery (renderer + viewer). Run on SwiftShader (default)
// or the real GPU (GPU=1).
import { chromium } from '@playwright/test';
import { chromiumArgs, resolveChromium } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8155;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const base = `http://127.0.0.1:${PORT}`;
async function ws() { for (let i = 0; i < 100; i++) { try { const r = await fetch(base + '/index.html'); if (r.ok) return; } catch {} await new Promise((r) => setTimeout(r, 100)); } throw new Error('no server'); }

let fail = 0;
const log = (ok, msg) => { console.log(`${ok ? 'OK  ' : 'FAIL'} ${msg}`); if (!ok) fail++; };

try {
  await ws();
  const browser = await chromium.launch({ executablePath: resolveChromium(), args: chromiumArgs() });
  const page = await browser.newPage({ viewport: { width: 420, height: 760 } });
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail++; });
  page.on('console', (m) => { if (m.type() === 'error') console.error('console.error:', m.text()); });
  await page.goto(base + '/index.html');

  // initial home render on the GPU
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, { timeout: 20000 });
  const home = await page.evaluate(() => window.__lastDone.engine);
  log(home === 'gpu-naive', `home engine = ${home} (expect gpu-naive)`);
  log(await page.evaluate(() => !!window.__viewer.gpu && !window.__viewer.gpu.lost), 'gpu present and not lost');

  // grab the lose_context extension on the viewer's live GL context
  await page.evaluate(() => { window.__loseExt = window.__viewer.gpu.gl.getExtension('WEBGL_lose_context'); });
  log(await page.evaluate(() => !!window.__loseExt), 'WEBGL_lose_context available');

  // ---- Case A: lose and DO NOT restore -> CPU fallback after the grace period ----
  let prevDone = await page.evaluate(() => window.__doneCount || 0);
  await page.evaluate(() => window.__loseExt.loseContext());
  await page.waitForFunction(() => window.__viewer.gpu && window.__viewer.gpu.lost, { timeout: 5000 });
  log(true, 'context lost flagged on the renderer');
  // wait for the grace-period CPU render to complete
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 20000 });
  const lostEngine = await page.evaluate(() => window.__lastDone.engine);
  log(!String(lostEngine).startsWith('gpu-'), `while lost, rendered on CPU engine = ${lostEngine}`);

  // ---- Case B: restore -> GPU render returns, image non-blank ----
  prevDone = await page.evaluate(() => window.__doneCount || 0);
  // a tick is required between loseContext and restoreContext (proven by probe-ctxloss)
  await page.waitForTimeout(60);
  await page.evaluate(() => window.__loseExt.restoreContext());
  await page.waitForFunction(() => window.__viewer.gpu && !window.__viewer.gpu.lost, { timeout: 8000 });
  log(true, 'context restored flagged on the renderer');
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 20000 });
  const restoredEngine = await page.evaluate(() => window.__lastDone.engine);
  log(String(restoredEngine).startsWith('gpu-'), `after restore, rendered on GPU engine = ${restoredEngine}`);

  // image must be non-blank (mean luma > 0 over a sample)
  const meanLuma = await page.evaluate(() => {
    const c = document.getElementById('view');
    const t = document.createElement('canvas'); t.width = 64; t.height = 64;
    const x = t.getContext('2d'); x.drawImage(c, 0, 0, 64, 64);
    const d = x.getImageData(0, 0, 64, 64).data; let s = 0;
    for (let i = 0; i < d.length; i += 4) s += d[i] + d[i + 1] + d[i + 2];
    return s / (64 * 64 * 3);
  });
  log(meanLuma > 1, `restored image non-blank (mean luma ${meanLuma.toFixed(1)})`);

  // ---- Case C: lose/restore at DEEP zoom (gpu-perturb) — exercises the reference
  // texture re-upload on restore, not just the shallow naive path ----
  prevDone = await page.evaluate(() => window.__doneCount || 0);
  await page.evaluate(() => window.__viewer.setState({ cx: '-0.743643887037158704752191506114774', cy: '0.131825904205311970493132056385139', radius: 5e-13 }));
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 45000 });
  const deepEngine = await page.evaluate(() => window.__lastDone.engine);
  log(deepEngine === 'gpu-perturb', `deep seahorse engine = ${deepEngine} (expect gpu-perturb)`);
  await page.evaluate(() => { window.__loseExt = window.__viewer.gpu.gl.getExtension('WEBGL_lose_context'); window.__loseExt.loseContext(); });
  await page.waitForFunction(() => window.__viewer.gpu && window.__viewer.gpu.lost, { timeout: 5000 });
  prevDone = await page.evaluate(() => window.__doneCount || 0);
  await page.waitForTimeout(60);
  await page.evaluate(() => window.__loseExt.restoreContext());
  await page.waitForFunction(() => window.__viewer.gpu && !window.__viewer.gpu.lost, { timeout: 8000 });
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 45000 });
  const deepRestored = await page.evaluate(() => window.__lastDone.engine);
  log(deepRestored === 'gpu-perturb', `deep render returned to GPU engine = ${deepRestored} (ref texture re-uploaded)`);
  const deepLuma = await page.evaluate(() => {
    const c = document.getElementById('view');
    const t = document.createElement('canvas'); t.width = 64; t.height = 64;
    const x = t.getContext('2d'); x.drawImage(c, 0, 0, 64, 64);
    const d = x.getImageData(0, 0, 64, 64).data; let s = 0;
    for (let i = 0; i < d.length; i += 4) s += d[i] + d[i + 1] + d[i + 2];
    return s / (64 * 64 * 3);
  });
  log(deepLuma > 1, `deep restored image non-blank (mean luma ${deepLuma.toFixed(1)})`);

  // ---- Case D: repeated losses -> give up on the GPU for the session (anti flip-flop) ----
  // Reset the cumulative loss counter so this case independently verifies give-up at
  // exactly GPU_MAX_LOSSES (cases A+C already incremented it).
  await page.evaluate(() => { window.__viewer._gpuLossCount = 0; });
  prevDone = await page.evaluate(() => window.__doneCount || 0);
  await page.evaluate(() => window.__viewer.setState({ cx: '-0.5', cy: '0', radius: 1.5 }));
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 20000 });
  for (let i = 0; i < 3; i++) {
    if (!(await page.evaluate(() => !!window.__viewer.gpu))) break;   // gave up already
    await page.evaluate(() => { window.__loseExt = window.__viewer.gpu.gl.getExtension('WEBGL_lose_context'); window.__loseExt.loseContext(); });
    await page.waitForFunction(() => window.__viewer.gpu && window.__viewer.gpu.lost, { timeout: 5000 });
    await page.waitForTimeout(60);
    await page.evaluate(() => { try { window.__loseExt.restoreContext(); } catch { /* gpu may be disposed on give-up */ } });
    await page.waitForFunction(() => !window.__viewer.gpu || !window.__viewer.gpu.lost, { timeout: 8000 });
  }
  const g = await page.evaluate(() => ({ gaveUp: window.__viewer._gpuGaveUp, gpu: !!window.__viewer.gpu, losses: window.__viewer._gpuLossCount }));
  log(g.gaveUp && !g.gpu, `after 3 losses gave up on GPU (gaveUp=${g.gaveUp}, gpu=${g.gpu}, losses=${g.losses})`);
  prevDone = await page.evaluate(() => window.__doneCount || 0);
  await page.evaluate(() => window.__viewer.render());
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prevDone, { timeout: 20000 });
  const afterGiveUp = await page.evaluate(() => window.__lastDone.engine);
  log(!String(afterGiveUp).startsWith('gpu-'), `post-give-up render stays on CPU = ${afterGiveUp}`);

  await browser.close();
} catch (e) {
  console.error('THREW:', e && (e.stack || e.message || e));
  fail++;
} finally {
  server.kill();
}

console.log(fail ? `\n${fail} CHECK(S) FAILED` : '\nALL CHECKS PASSED');
process.exit(fail ? 1 : 0);
