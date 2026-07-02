// probe-repro-url.mjs — reproduce Danielle's Spawn-31 report: "fine details slightly
// off at a view I zoomed to; zooming further fixes them up".
//
// Hypothesis: SA-coefficient REUSE across zoom ticks. The reused coeffs' truncation
// validity nests (new dc box ⊂ validated box) but the ESCAPE-GUARD is a 13×13 probe
// grid over the OLD box — after a few ticks the view sits in a tiny corner of it,
// sampled by ~a probe or two, so an isolated early-escaping pixel can be skipped
// past → subtly wrong fine detail until the next SA refresh/rebuild.
//
// Method (her exact coordinate): render the SAME final view three ways and diff on a
// FINE grid (96×96 samples):
//   COLD   — direct setState at the target (fresh reference + fresh SA: ground truth)
//   WARM   — arrive by zoom ticks from 5 octaves shallower (cache + SA reuse engaged)
//   NOSA   — cold with series approximation off (the SA-vs-truth noise floor)
// diff(WARM, COLD) >> diff(NOSA, COLD)  ⇒ the cache/SA-reuse path is adding error.
//
//   GPU=1 node tools/probe-repro-url.mjs
import { chromium } from '@playwright/test';
import { launchOpts, gpuMode } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8155;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/index.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

// Danielle's reported coordinate (r = 2.99709927e-76 ≈ 2^-250.9)
const RE = '-0.013723903427109727127929602217373808131181131374105015124101269177883848493753482931960573952845398672419';
const IM = '0.720216709412699393589401716000260792609362764831527728273017596899498993965708580806931908277598917947246';
const R = 2.99709927e-76;
const TICKS = Number(process.env.TICKS || 5);

const G = 96;
function gridEvalSrc() {
  return (GG) => {
    const c = document.getElementById('view'); const g = c.getContext('2d');
    const { data } = g.getImageData(0, 0, c.width, c.height);
    const rgb = [];
    for (let gy = 0; gy < GG; gy++) for (let gx = 0; gx < GG; gx++) {
      const px = Math.floor((gx + 0.5) / GG * c.width), py = Math.floor((gy + 0.5) / GG * c.height);
      const o = (py * c.width + px) * 4;
      rgb.push(data[o], data[o + 1], data[o + 2]);
    }
    return rgb;
  };
}
const bulkDiff = (a, b) => {
  let d = 0; const n = a.length / 3;
  for (let i = 0; i < a.length; i += 3) {
    if (Math.abs(a[i] - b[i]) > 24 || Math.abs(a[i + 1] - b[i + 1]) > 24 ||
        Math.abs(a[i + 2] - b[i + 2]) > 24) d++;
  }
  return d / n;
};

let fail = false;
try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage({ viewport: { width: 600, height: 600 } });
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail = true; });
  await page.goto(baseURL + '/index.html');
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 30000 });
  await page.evaluate(() => window.__viewer.setSupersample(1));
  console.log('mode=' + gpuMode() + '  repro at re=-0.01372…, im=0.72021…, r=2.997e-76 (≈2^-251)\n');

  const waitDone = async (c0) =>
    page.waitForFunction((p) => (window.__doneCount || 0) > p, c0, { timeout: 300000 });
  const state = async () => page.evaluate(() => ({
    maxIter: window.__viewer.maxIter, skip: window.__viewer._saSkip,
    refLen: window.__viewer._refMeta?.refLen, cached: !!window.__viewer._refMeta?.refCached,
    engine: window.__viewer._refMeta?.engine,
  }));

  async function render(cx, cy, radius, { series = true, clearCache = true } = {}) {
    await page.evaluate((o) => {
      const V = window.__viewer;
      if (o.clearCache) V._refCache = null;
      V.setSeries(o.series);
    }, { series, clearCache });
    const c0 = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate((vv) => window.__viewer.setState(vv), { cx, cy, radius });
    await waitDone(c0);
    return state();
  }

  // COLD ground truth at the exact target
  const cold = await render(RE, IM, R);
  const coldRgb = await page.evaluate(gridEvalSrc(), G);
  console.log('COLD :', JSON.stringify(cold));

  // NOSA cold baseline (the SA-vs-truth noise floor at this view)
  const nosa = await render(RE, IM, R, { series: false });
  const nosaRgb = await page.evaluate(gridEvalSrc(), G);
  console.log('NOSA :', JSON.stringify(nosa));

  // WARM arrival: start TICKS octaves shallower (same center), then centered
  // zoom ticks down to the target radius — cache + SA reuse engage per tick.
  await render(RE, IM, R * 2 ** TICKS);            // anchor (fills the cache fresh)
  for (let t = 0; t < TICKS; t++) {
    const c0 = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate(() => {
      const V = window.__viewer;
      V.zoomAt(V.backingW / 2, V.backingH / 2, 0.5);
      V.render();
    });
    await waitDone(c0);
    console.log(`  tick ${t + 1}:`, JSON.stringify(await state()));
  }
  const warmRgb = await page.evaluate(gridEvalSrc(), G);

  const dWarm = bulkDiff(warmRgb, coldRgb);
  const dNosa = bulkDiff(nosaRgb, coldRgb);
  console.log(`\nfine-grid (${G}×${G}) diffs vs COLD ground truth:`);
  console.log(`  WARM(cache+SA-reuse) : ${(dWarm * 100).toFixed(2)}%`);
  console.log(`  NOSA (noise floor)   : ${(dNosa * 100).toFixed(2)}%`);
  console.log(dWarm > Math.max(0.01, 3 * dNosa)
    ? '\n=> WARM adds error beyond the SA noise floor — the reuse path is implicated. REPRODUCED'
    : '\n=> WARM within the noise floor — reuse path NOT implicated at this view. NOT REPRODUCED');
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e); fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
