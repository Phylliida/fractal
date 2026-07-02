// probe-interrupt.mjs — mid-render INTERRUPT latency (Spawn 33, Danielle's report:
// "if it's halfway rendering and I zoom in I have to wait for it to finish").
//
// ROOT CAUSE: _beginPreview/render() bumped gen (main drops stale strips) but never
// sent the GPU worker a `cancel` — the stale frame kept rendering EVERY remaining
// strip at full GPU cost and the settle render queued behind it (a tail that
// full-resolution frames stretched to seconds). Plus the worker's strip loop never
// yielded a macrotask between ack and the next strip, so even an arriving cancel
// was seen one whole strip late.
//
// Scenario measured: full-res-ish viewport, deep view; start a render; at ~30% of
// its expected duration send a zoom tick (preview + settle); measure wall-clock from
// the ZOOM to the settled new frame. A/B: production vs the OLD behavior (client
// cancel stubbed to a no-op — gen guards stay, message just never sent).
//
//   GPU=1 node tools/probe-interrupt.mjs
import { chromium } from '@playwright/test';
import { launchOpts, gpuMode } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8160;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
for (let i = 0; i < 100; i++) { try { const r = await fetch(baseURL + '/index.html'); if (r.ok) break; } catch { /* retry */ } await new Promise((r) => setTimeout(r, 100)); }

const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const BITS = Number(process.env.BITS || 300);
const ROUNDS = Number(process.env.ROUNDS || 3);

let fail = false;
try {
  const browser = await chromium.launch(launchOpts());
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 } });
  page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail = true; });
  await page.goto(baseURL + '/index.html');
  await page.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 60000 });
  await page.evaluate(() => window.__viewer.setSupersample(1));
  console.log('mode=' + gpuMode() + '  1400×900 full-res, deep 2^-' + BITS + ' — zoom-interrupt at ~30% of the render\n');

  const waitDone = (c0, t = 300000) =>
    page.waitForFunction((p) => (window.__doneCount || 0) > p, c0, { timeout: t });

  async function measure(stubbed) {
    // WARM scenario (Danielle's real interaction): the ref cache is hot, so a settle
    // render dispatches in ~ms and, in the OLD behavior, queues straight behind the
    // stale frame's strips. Interrupt lands mid-STRIP-phase (the frame is GPU-bound
    // on a warm hit). Also counts stale strips still arriving after the zoom — the
    // direct signature of the missing cancel.
    await page.evaluate((s) => {
      const V = window.__viewer;
      const cl = V.gpuWorker;
      if (cl) {
        if (!cl.__origCancel) cl.__origCancel = cl.cancel.bind(cl);
        // OLD behavior: gen guards intact, but the cancel message never reaches the worker
        cl.cancel = s ? (gen) => { cl._gen = gen; cl._handlers = null; } : cl.__origCancel;
        if (!cl.__tapped) {
          cl.__tapped = true;
          cl.worker.addEventListener('message', (e) => {
            if (e.data && e.data.type === 'strip' && window.__staleWatch && e.data.gen === window.__staleWatch.gen) {
              window.__staleWatch.count++; window.__staleWatch.last = performance.now();
            }
          });
        }
      }
      window.__staleWatch = null;
    }, stubbed);
    // cold anchor render fills the ref cache; time a warm frame for the interrupt point
    let c0 = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate((vv) => { window.__viewer._refCache = null; window.__viewer.setState(vv); }, { cx: RE, cy: IM, radius: 1.5 * 2 ** -BITS });
    await waitDone(c0);
    c0 = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate(() => window.__viewer.render());     // warm re-render (cache hit)
    const tW = Date.now();
    await waitDone(c0);
    const frameMs = Date.now() - tW;                          // warm frame ≈ strip phase
    // warm render again, interrupt mid-strips
    c0 = await page.evaluate(() => window.__doneCount || 0);
    await page.evaluate(() => {
      window.__staleWatch = { gen: window.__viewer.gen + 1, count: 0, last: 0 };
      window.__viewer.render();
    });
    await new Promise((r) => setTimeout(r, Math.max(80, frameMs * 0.4)));
    const tZoom = Date.now();
    await page.evaluate(() => {
      window.__staleWatch.t0 = performance.now();
      const V = window.__viewer;
      V.zoomBy(0.5, V.backingW / 2 + 40, V.backingH / 2 - 25);   // wheel-like zoom tick
    });
    await waitDone(c0);   // the SETTLED new frame (the interrupted one never reports done)
    const interruptMs = Date.now() - tZoom;
    const stale = await page.evaluate(() => {
      const w = window.__staleWatch;
      return { tail: w && w.last > w.t0 ? Math.round(w.last - w.t0) : 0, count: w ? w.count : 0 };
    });
    return { frameMs, interruptMs, staleTailMs: stale.tail, staleAfterZoom: stale.count };
  }

  console.log('round | fixed: warmfrm zoom→settled staleTail | OLD: warmfrm zoom→settled staleTail');
  let f = Infinity, o = Infinity, ft = 0, ot = 0;
  for (let r = 0; r < ROUNDS; r++) {
    const a = await measure(false);
    const b = await measure(true);
    f = Math.min(f, a.interruptMs); o = Math.min(o, b.interruptMs);
    ft = Math.max(ft, a.staleTailMs); ot = Math.max(ot, b.staleTailMs);
    console.log(`  ${r + 1}   |       ${String(a.frameMs).padStart(6)} ${String(a.interruptMs).padStart(9)} ${String(a.staleTailMs).padStart(7)}ms |      ${String(b.frameMs).padStart(6)} ${String(b.interruptMs).padStart(9)} ${String(b.staleTailMs).padStart(7)}ms`);
  }
  console.log(`\nmin zoom→settled: FIXED ${f}ms vs OLD ${o}ms; worst stale-strip tail after zoom: FIXED ${ft}ms vs OLD ${ot}ms`);
  if (!(f <= o && ft <= ot)) { fail = true; console.log('FAIL: fix did not improve interruption'); }
  else console.log('PASS');
  await browser.close();
} catch (e) {
  console.error('ERROR:', e.stack || e); fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
