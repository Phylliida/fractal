// bench-offscreen-jank.mjs — measure the OffscreenCanvas win (Spawn 24): how much the GPU
// raster BLOCKS THE MAIN THREAD during a deep render, main-thread path vs offscreen worker.
//
// The throughput frontier is closed (the deep frame is warp-divergence bound), so the
// offscreen pivot doesn't aim for a faster FRAME — it aims for a RESPONSIVE main thread
// (the win is input/scroll latency, not ms/frame). The right metric, per the project's
// measure-first discipline, is therefore main-thread JANK, not render time:
//   - Long Tasks: count + total + max duration of main-thread tasks >50ms during the render
//     (the W3C Long Tasks API; worker-thread work is NOT reported, which is exactly the point).
//   - Heartbeat: a rAF loop's worst inter-frame gap — the stall a UI animation would feel.
// It also reports wall-clock render time to confirm the offscreen path isn't slower.
//
// On SwiftShader the GL work is pure CPU, so moving it to the worker thread frees the main
// thread dramatically (the clearest demonstration); GPU=1 confirms on real hardware.
//
//   node tools/bench-offscreen-jank.mjs            (SwiftShader)
//   GPU=1 node tools/bench-offscreen-jank.mjs      (real RTX 3090)
//   BITS=271 VP=400 node tools/bench-offscreen-jank.mjs
import { chromium } from '@playwright/test';
import { launchOpts } from './chromium-launch.mjs';
import { spawn } from 'node:child_process';

const PORT = process.env.PORT || 8163;
const BITS = Number(process.env.BITS || 271);
const VP = Number(process.env.VP || 400);
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/index.html'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

// Install the main-thread instrumentation: a Long Tasks observer + a rAF heartbeat that
// records the worst gap. Returns nothing; results are read from window.__jank later.
function instrumentSrc() {
  window.__jank = { long: [], maxGap: 0, ticks: 0 };
  try {
    const obs = new PerformanceObserver((list) => {
      for (const e of list.getEntries()) window.__jank.long.push(e.duration);
    });
    obs.observe({ entryTypes: ['longtask'] });
  } catch { /* longtask unsupported -> heartbeat only */ }
  let last = performance.now();
  function tick() {
    const now = performance.now();
    const gap = now - last; last = now;
    if (window.__jank.ticks > 0 && gap > window.__jank.maxGap) window.__jank.maxGap = gap;
    window.__jank.ticks++;
    window.__jank._raf = requestAnimationFrame(tick);
  }
  tick();
}

async function measure(page, label) {
  await page.evaluate(() => window.__viewer.setSupersample(1));
  // settle: render the deep view ONCE (warm shaders / reference) before measuring.
  const radius = 1.5 * 2 ** -BITS;
  let before = await page.evaluate(() => window.__doneCount || 0);
  await page.evaluate((p) => window.__viewer.setState(p), { cx: RE, cy: IM, radius });
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, before, { timeout: 180000 });

  // now arm instrumentation and re-render the SAME deep view; measure that render.
  await page.evaluate(instrumentSrc);
  before = await page.evaluate(() => window.__doneCount || 0);
  const t0 = Date.now();
  // nudge to a sibling coordinate so render() actually recomputes (radius identical → same cost)
  await page.evaluate((p) => window.__viewer.setState(p), { cx: RE, cy: IM, radius: radius * 1.0000001 });
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, before, { timeout: 180000 });
  const ms = Date.now() - t0;
  const j = await page.evaluate(() => {
    cancelAnimationFrame(window.__jank._raf);
    const L = window.__jank.long;
    return {
      engine: window.__lastDone?.engine, offscreen: window.__viewer._useOffscreen(),
      longCount: L.length, longTotal: Math.round(L.reduce((a, b) => a + b, 0)),
      longMax: Math.round(Math.max(0, ...L)), maxGap: Math.round(window.__jank.maxGap), ticks: window.__jank.ticks,
    };
  });
  return { label, ms, ...j };
}

let fail = false;
try {
  await waitServer();
  const browser = await chromium.launch(launchOpts());
  const vp = { width: VP, height: VP };

  async function run(offscreen) {
    const page = await browser.newPage({ viewport: vp });
    page.on('pageerror', (e) => { console.error('PAGE ERROR:', e.message); fail = true; });
    await page.goto(baseURL + '/index.html?offscreen=' + (offscreen ? '1' : '0'));
    await page.waitForFunction(() => (window.__doneCount || 0) > 0, null, { timeout: 30000 });
    const r = await measure(page, offscreen ? 'offscreen (worker)' : 'main-thread');
    await page.close();
    return r;
  }

  const off = await run(true);
  const main = await run(false);

  const fmt = (r) => `${r.label.padEnd(20)} engine=${(r.engine || '?').padEnd(15)} render=${String(r.ms + 'ms').padStart(8)}  longTasks=${String(r.longCount).padStart(3)} total=${String(r.longTotal + 'ms').padStart(7)} max=${String(r.longMax + 'ms').padStart(7)}  heartbeatMaxGap=${String(r.maxGap + 'ms').padStart(7)} (${r.ticks} ticks)`;
  console.log(`\ndeep render 2^-${BITS}, ${VP}x${VP}, ss=1, ${process.env.GPU ? 'REAL GPU' : 'SwiftShader'}\n`);
  console.log(fmt(off));
  console.log(fmt(main));
  if (off.offscreen !== true || main.offscreen !== false) {
    console.log('\n⚠ path flags wrong (off.offscreen=' + off.offscreen + ', main.offscreen=' + main.offscreen + ')'); fail = true;
  }
  const blockReduction = main.longTotal > 0 ? (main.longTotal / Math.max(1, off.longTotal)).toFixed(1) : 'n/a';
  console.log(`\nmain-thread blocked-time (Long Task total): main ${main.longTotal}ms -> offscreen ${off.longTotal}ms  (${blockReduction}× less blocking)`);
  console.log(`worst heartbeat stall:                       main ${main.maxGap}ms -> offscreen ${off.maxGap}ms`);
  await browser.close();
} catch (e) {
  console.error('ERROR', e); fail = true;
} finally {
  server.kill();
}
process.exit(fail ? 1 : 0);
