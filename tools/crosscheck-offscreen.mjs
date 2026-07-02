// crosscheck-offscreen.mjs — prove the OffscreenCanvas GPU worker path (Spawn 24) is
// PIXEL-IDENTICAL to the legacy on-main-thread GpuRenderer path. Both run the same WebGL2
// shaders; the worker just relocates the raster+color loop off the main thread and ships
// finished ImageBitmap strips back. The composited image must therefore be byte-identical
// (RGBA8 round-trip through an ImageBitmap is lossless).
//
// It loads the real viewer twice — `?offscreen=1` (worker) and `?offscreen=0` (main thread)
// — renders the SAME view at each of the three GPU engines (naive shallow / df64 medium /
// rescaled-fe deep), reads back the full canvas, and reports the differing-pixel fraction
// and max channel delta. Expect 0.000% / 0 everywhere.
//
//   node tools/crosscheck-offscreen.mjs            (SwiftShader)
//   GPU=1 node tools/crosscheck-offscreen.mjs      (real GPU)
import { chromium } from '@playwright/test';
import { launchOpts } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8161;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/index.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

// A genuine deep boundary coordinate (the project's standard deep test point).
const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

// view list: [label, expected engine, radius]   (home stays at the default center)
const VIEWS = [
  { label: 'naive (home 2^0)', engine: 'gpu-naive', center: 'home', bits: 0 },
  { label: 'df64 (2^-50)', engine: 'gpu-perturb', center: 'deep', bits: 50 },
  { label: 'fe/rescaled (2^-120)', engine: 'gpu-perturb-fe', center: 'deep', bits: 120 },
];

async function renderView(page, v) {
  const before = await page.evaluate(() => window.__doneCount || 0);
  await page.evaluate(() => window.__viewer.setSupersample(1));   // ss=1 for a deterministic 1× compare
  if (v.center === 'home') {
    await page.evaluate(() => window.__viewer.setState({ cx: '-0.5', cy: '0', radius: 1.5 }));
  } else {
    const radius = 1.5 * 2 ** -v.bits;
    await page.evaluate((p) => window.__viewer.setState(p), { cx: RE, cy: IM, radius });
  }
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, before, { timeout: 180000 });
  return page.evaluate(() => {
    const c = document.getElementById('view');
    const g = c.getContext('2d');
    const { data } = g.getImageData(0, 0, c.width, c.height);
    // hash + raw copy (small viewport) for an exact diff on the Node side
    return { w: c.width, h: c.height, data: Array.from(data),
             engine: window.__lastDone?.engine, offscreen: window.__viewer._useOffscreen() };
  });
}

function diff(a, b) {
  if (a.w !== b.w || a.h !== b.h) return { mismatch: 1, note: `size ${a.w}x${a.h} vs ${b.w}x${b.h}` };
  let nDiff = 0, maxD = 0;
  const n = a.data.length;
  for (let i = 0; i < n; i += 4) {
    let d = 0;
    for (let k = 0; k < 3; k++) d = Math.max(d, Math.abs(a.data[i + k] - b.data[i + k]));
    if (d > 0) { nDiff++; if (d > maxD) maxD = d; }
  }
  return { mismatch: nDiff / (n / 4), maxD, nDiff, total: n / 4 };
}

let fail = false;
try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  // smaller viewport keeps the deep renders quick and the full-pixel diff cheap
  const vp = { width: 256, height: 256 };

  const offPage = await browser.newPage({ viewport: vp });
  offPage.on('pageerror', (e) => { console.error('OFFSCREEN PAGE ERROR:', e.message); fail = true; });
  await offPage.goto(baseURL + '/index.html?offscreen=1');
  await offPage.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 30000 });

  const mainPage = await browser.newPage({ viewport: vp });
  mainPage.on('pageerror', (e) => { console.error('MAIN PAGE ERROR:', e.message); fail = true; });
  await mainPage.goto(baseURL + '/index.html?offscreen=0');
  await mainPage.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 30000 });

  console.log('view                       engine(off/main)            mismatch   maxΔ');
  for (const v of VIEWS) {
    const off = await renderView(offPage, v);
    const main = await renderView(mainPage, v);
    const d = diff(off, main);
    const ok = d.mismatch === 0 && (d.maxD || 0) === 0
      && off.engine === v.engine && main.engine === v.engine
      && off.offscreen === true && main.offscreen === false;
    if (!ok) fail = true;
    const pct = (100 * d.mismatch).toFixed(3) + '%';
    console.log(`${v.label.padEnd(26)} ${String(off.engine + '/' + main.engine).padEnd(27)} ${pct.padStart(8)}  ${String(d.maxD ?? '-').padStart(4)}  ${ok ? 'OK' : 'FAIL ' + JSON.stringify(d)}`);
  }
  await browser.close();
} catch (e) {
  console.error('ERROR', e);
  fail = true;
} finally {
  server.kill();
}
console.log(fail ? '\nRESULT: FAIL' : '\nRESULT: ALL PIXEL-IDENTICAL');
process.exit(fail ? 1 : 0);
