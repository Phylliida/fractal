import { test, expect } from '@playwright/test';
import { waitDone, doneCount, lastDone, canvasStats } from './helpers.mjs';

// These run the GPU validation harness (GPU vs CPU oracle, headless via
// SwiftShader) and confirm the live app dispatches to the GPU engines. The
// exhaustive depth sweep lives in tools/validate-gpu.mjs; here we assert the
// key invariants cheaply so a regression fails CI.

test('WebGL2 is available and the GPU renderer initializes', async ({ page }) => {
  await page.goto('/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  const init = await page.evaluate(() => window.__gpu.init(120, 120));
  expect(init.ok).toBe(true);
  expect(init.supported).toBe(true);
});

test('GPU naive f32 matches the CPU naive oracle at home (bulk)', async ({ page }) => {
  await page.goto('/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  await page.evaluate(() => window.__gpu.init(160, 160));
  const res = await page.evaluate(() => {
    const r = 1.5, W = 160, H = 160, scale = (2 * r) / H;
    return window.__gpu.compareNaive({ ox: -0.5 - r * (W / H), oy: 0 - r, scale, maxIter: 400, width: W, height: H, df64: false, checkStep: 1 });
  });
  // bulk agreement: mean smooth-count diff tiny, few-count mismatches rare
  expect(res.meanAbs).toBeLessThan(0.3);
  expect(res.mism / res.compared).toBeLessThan(0.02);
});

test('GPU perturb df64 matches the CPU perturbation oracle deep (2^-30, 2^-60)', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/test/gpu/harness.html');
  await page.waitForFunction(() => window.__ready === true, { timeout: 15000 });
  await page.evaluate(() => window.__gpu.init(120, 120));
  const SH = '-0.743643887037158704752191506114774', SI = '0.131825904205311970493132056385139';
  for (const [bits, it] of [[30, 8000], [60, 15000]]) {
    const res = await page.evaluate(([re, im, r, mi]) => window.__gpu.comparePerturb(
      { re, im, radius: r, maxIter: mi, width: 120, height: 120, checkStep: 2, glitchTol: 0, df64: true }),
      [SH, SI, 1.5 * 2 ** -bits, it]);
    expect(res.meanAbs).toBeLessThan(1.0);          // boundary-level only
    expect(res.mism / res.compared).toBeLessThan(0.02);
    expect(res.insideMismatch).toBeLessThan(10);
  }
});

test('app dispatches gpu-naive at home and gpu-perturb when deep', async ({ page }) => {
  page.on('pageerror', (e) => { throw e; });
  await page.goto('/');
  await waitDone(page, 0);
  expect((await lastDone(page)).engine).toBe('gpu-naive');

  // Disable supersampling for this dispatch check: it 4×'s the pixel work, which
  // is slow under SwiftShader (software GL) headless. AA quality is covered elsewhere.
  const cs = await page.evaluate(() => window.__doneCount);
  await page.evaluate(() => window.__viewer.setSupersample(1));
  await waitDone(page, cs);

  const c0 = await page.evaluate(() => window.__doneCount);
  await page.evaluate(() => window.__viewer.setState({
    cx: '-0.743643887037158704752191506114774',
    cy: '0.131825904205311970493132056385139', radius: 5e-13, maxIter: 8000,
  }));
  await waitDone(page, c0, 45000);
  expect((await lastDone(page)).engine).toBe('gpu-perturb');
});

test('forceCpu falls back to the CPU worker engines and still renders', async ({ page }) => {
  page.on('pageerror', (e) => { throw e; });
  await page.goto('/');
  await waitDone(page, 0);
  const c0 = await page.evaluate(() => {
    const v = window.__viewer; v.forceCpu = true; v._gpuChecked = false; v.gpu = null;
    const n = window.__doneCount; v.render(); return n;
  });
  await waitDone(page, c0, 30000);
  expect((await lastDone(page)).engine).toBe('naive'); // CPU naive at home
});

// Mobile GPUs lose the WebGL context under memory pressure / when backgrounded.
// The renderer flags itself lost (-> render on the CPU pool) and re-renders on the
// GPU once the browser restores it. Simulated here via WEBGL_lose_context, whose
// loseContext()/restoreContext() drive the same webglcontextlost/restored events a
// real device fires. (A tick is required between lose and restore — see
// tools/probe-ctxloss.mjs.)
test('recovers from WebGL context loss: CPU while lost, GPU once restored', async ({ page }) => {
  test.setTimeout(40000);
  page.on('pageerror', (e) => { throw e; });
  // Pin the on-main-thread GPU renderer: this test simulates loss via loseContext() on
  // viewer.gpu's GL context. The default offscreen path keeps the context in the worker
  // (viewer.gpu is null) and forwards loss worker→main (see probe-recover-offscreen).
  await page.goto('/?offscreen=0');
  await waitDone(page, 0);
  // Only meaningful when the home view is on a GPU engine (the headless default).
  test.skip(!(await lastDone(page)).engine.startsWith('gpu-'), 'no GPU path in this environment');

  const hasExt = await page.evaluate(() => {
    const ext = window.__viewer.gpu.gl.getExtension('WEBGL_lose_context');
    window.__loseExt = ext; return !!ext;
  });
  test.skip(!hasExt, 'WEBGL_lose_context unavailable');

  // Lose the context (do not restore yet): the renderer flags lost and, after a short
  // grace period, the view re-renders on the CPU worker pool.
  let c0 = await page.evaluate(() => window.__doneCount);
  await page.evaluate(() => window.__loseExt.loseContext());
  await page.waitForFunction(() => window.__viewer.gpu && window.__viewer.gpu.lost, { timeout: 5000 });
  await waitDone(page, c0, 30000);
  expect((await lastDone(page)).engine.startsWith('gpu-')).toBe(false); // CPU while lost

  // Restore: the renderer recreates its GL objects and the view re-renders on the GPU.
  c0 = await page.evaluate(() => window.__doneCount);
  await page.waitForTimeout(60);                  // tick required between lose and restore
  await page.evaluate(() => window.__loseExt.restoreContext());
  await page.waitForFunction(() => window.__viewer.gpu && !window.__viewer.gpu.lost, { timeout: 8000 });
  await waitDone(page, c0, 30000);
  expect((await lastDone(page)).engine.startsWith('gpu-')).toBe(true); // GPU restored

  const stats = await canvasStats(page);          // and the recovered image isn't blank
  expect(stats.max).toBeGreaterThan(0);
  expect(stats.distinctColors).toBeGreaterThan(3);
});

// The GPU perturbation shaders emit a Pauldelbrot glitch flag in the sn texture's .b
// channel, but normal renders pass glitchTol=0 so it's never set (the GPU reports a
// structural 0). The "Glitch overlay" debug toggle turns on the diagnostic — PURELY
// diagnostic: it sets the flag but does NOT change escape/rebase or the rendered fractal —
// tints flagged pixels magenta in the color pass, and reads back the real flagged count.
// This proves: (a) the toggle re-renders and reports a real finite count, and (b) a forced
// huge tolerance lights up the flag → readback count → magenta tint end-to-end, then
// turning the overlay off removes the tint.
test('glitch overlay: honest GPU glitch count + magenta tint when forced', async ({ page }) => {
  test.setTimeout(120000);     // three sequential deep renders under SwiftShader (software GL)
  page.on('pageerror', (e) => { throw e; });
  await page.goto('/');
  await waitDone(page, 0);
  test.skip(!(await lastDone(page)).engine.startsWith('gpu-'), 'no GPU path in this environment');

  // Count overlay-magenta-ish pixels: the overlay mixes flagged pixels 0.6 toward
  // (255,0,255), so a flagged pixel has high R, high B, low G.
  const countMagenta = () => page.evaluate(() => {
    const c = document.getElementById('view');
    const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
    let m = 0;
    for (let i = 0; i < data.length; i += 4)
      if (data[i] > 150 && data[i + 2] > 150 && data[i + 1] < 110) m++;
    return m;
  });

  // A deep view (perturbation engine), ss=1 for speed under SwiftShader (software GL).
  let c0 = await page.evaluate(() => window.__doneCount);
  await page.evaluate(() => {
    window.__viewer.setSupersample(1);
    window.__viewer.setState({
      cx: '-0.743643887037158704752191506114774',
      cy: '0.131825904205311970493132056385139', radius: 5e-13, maxIter: 1500,
    });
  });
  await waitDone(page, c0, 45000);
  expect((await lastDone(page)).engine).toBe('gpu-perturb');
  const mBase = await countMagenta();             // incidental palette purples, the baseline

  // Enable the overlay → re-render with the Pauldelbrot diagnostic on. A glitch-free
  // render reports a finite count (typically 0); the image is unchanged (nothing flagged).
  c0 = await page.evaluate(() => window.__doneCount);
  await page.evaluate(() => window.__viewer.setShowGlitches(true));
  await waitDone(page, c0, 45000);
  let d = await lastDone(page);
  expect(d.engine).toBe('gpu-perturb');
  expect(Number.isFinite(d.glitches)).toBe(true);
  expect(d.glitches).toBeGreaterThanOrEqual(0);

  // Force detection: a huge tolerance flags essentially every escaping pixel
  // (|z|^2 < tol·|Z_m|^2), exercising the flag → readback → tint path end-to-end.
  c0 = await page.evaluate(() => window.__doneCount);
  await page.evaluate(() => { window.__viewer.glitchTol = 1e9; window.__viewer.render(); });
  await waitDone(page, c0, 45000);
  d = await lastDone(page);
  expect(d.glitches).toBeGreaterThan(100);        // many flagged pixels read back
  const mForced = await countMagenta();
  expect(mForced).toBeGreaterThan(mBase + 100);   // and they're visibly tinted

  // Turn the overlay off → the tint is dropped (cheap recolor; the rendered fractal returns).
  await page.evaluate(() => { window.__viewer.glitchTol = 1e-6; window.__viewer.setShowGlitches(false); });
  await page.waitForTimeout(200);
  expect(await countMagenta()).toBeLessThan(mBase + 100);
});

// The strip-tiled GPU render must REPLACE the low-res preview with high-res tiles as
// they arrive — it must NOT clear the whole canvas to interior-black first (Danielle:
// "don't render black screen instead replace low res with high res as it comes in").
// This is the mechanism: _blitGpuStrip copies ONLY its own strip's display rows.
test('strip blit replaces only its rows (preview survives below the tile)', async ({ page }) => {
  // main-thread GPU path: drives viewer.gpu.canvas + _blitGpuStrip directly. The offscreen
  // path composites worker-pushed ImageBitmap strips (final image: crosscheck-offscreen).
  await page.goto('/?offscreen=0');
  await waitDone(page, 0);
  test.skip(!(await lastDone(page)).engine.startsWith('gpu-'), 'no GPU path in this environment');
  // ss=1 keeps the GPU canvas 1:1 with the display (simple row math) and is fast.
  let cs = await doneCount(page);
  await page.evaluate(() => window.__viewer.setSupersample(1));
  await waitDone(page, cs);

  const res = await page.evaluate(() => {
    const v = window.__viewer;
    const c = document.getElementById('view');
    const g = c.getContext('2d');
    const W = c.width, H = c.height;
    // The home render left a complete frame in v.gpu.canvas. Paint the DISPLAY canvas a
    // sentinel (stand-in for the low-res preview) and blit only a top strip; everywhere
    // the strip didn't cover must still be the sentinel (preview), not black.
    g.fillStyle = 'rgb(0,128,255)'; g.fillRect(0, 0, W, H);
    const px = (x, y) => { const d = g.getImageData(x | 0, y | 0, 1, 1).data; return [d[0], d[1], d[2]]; };
    const isSentinel = (p) => p[0] === 0 && p[1] === 128 && p[2] === 255;
    const ss = v._effSS;                       // 1 here
    v._blitGpuStrip(0, Math.round(H * 0.4) * ss);   // blit the top ~40% of display rows
    const top = px(W / 2, H * 0.1);            // inside the blitted strip
    const bot = px(W / 2, H * 0.9);            // below it
    return { topReplaced: !isSentinel(top), botPreserved: isSentinel(bot), top, bot };
  });
  expect(res.topReplaced).toBe(true);    // high-res tile arrived in the strip
  expect(res.botPreserved).toBe(true);   // preview preserved below it (NOT black-voided)
});

// Integration: during a REAL multi-strip render, after the first strip blits, the rows
// below it still show the prior frame (preview), proving the loop reveals high-res over
// the preview rather than over a black void. Deterministic: a hook captures a below-strip
// pixel at the exact moment of the first strip blit.
test('multi-strip render keeps the preview below the first tile', async ({ page }) => {
  test.setTimeout(60000);
  // main-thread GPU path: hooks viewer._blitGpuStrip. The offscreen path reveals strips via
  // onStrip → drawImage(bitmap); its final image is crosschecked pixel-identical.
  await page.goto('/?offscreen=0');
  await waitDone(page, 0);
  test.skip(!(await lastDone(page)).engine.startsWith('gpu-'), 'no GPU path in this environment');
  let cs = await doneCount(page);
  await page.evaluate(() => window.__viewer.setSupersample(1));
  await waitDone(page, cs);

  // A mostly-exterior shallow view (gpu-naive, fast escapes → quick under SwiftShader).
  cs = await doneCount(page);
  await page.evaluate(() => window.__viewer.setState({ cx: '1.5', cy: '0', radius: 0.5, maxIter: 400 }));
  await waitDone(page, cs);

  const res = await page.evaluate(async () => {
    const v = window.__viewer;
    const c = document.getElementById('view');
    const g = c.getContext('2d');
    // Force several short strips: pick maxIter so _stripRows() ≈ cH/4 (a high cap only
    // shrinks the strips; the actual escapes here are still fast since the view is exterior).
    v.autoIter = false;
    v.maxIter = Math.max(800, Math.ceil(1.6e9 / (v.cW * v.cH)));
    const strips = Math.ceil(v.cH / v._stripRows());
    // Capture a below-strip pixel at the FIRST strip blit (the real loop calls this).
    let firstBottom = null, n = 0;
    const orig = v._blitGpuStrip.bind(v);
    v._blitGpuStrip = function (y, h) {
      orig(y, h);
      if (++n === 1) {
        const d = g.getImageData((c.width / 2) | 0, (c.height * 0.95) | 0, 1, 1).data;
        firstBottom = [d[0], d[1], d[2]];
      }
    };
    // Paint the sentinel preview, then render WITHOUT touching the canvas first.
    g.fillStyle = 'rgb(0,128,255)'; g.fillRect(0, 0, c.width, c.height);
    v.render();
    // Wait for completion.
    await new Promise((r) => { const t = setInterval(() => { if (!v.rendering) { clearInterval(t); r(); } }, 30); });
    v._blitGpuStrip = orig;
    const final = g.getImageData((c.width / 2) | 0, (c.height * 0.95) | 0, 1, 1).data;
    return { strips, firstBottom, finalBottom: [final[0], final[1], final[2]] };
  });
  const isSentinel = (p) => p[0] === 0 && p[1] === 128 && p[2] === 255;
  expect(res.strips).toBeGreaterThan(1);             // genuinely multi-strip (test is meaningful)
  expect(isSentinel(res.firstBottom)).toBe(true);    // after strip 1, below it still shows the preview
  expect(isSentinel(res.finalBottom)).toBe(false);   // and the high-res render eventually fills it in
});

// OffscreenCanvas GPU worker (Spawn 24) is the DEFAULT raster path when supported: the GPU
// raster+color loop runs in a worker that pushes finished ImageBitmap strips back, keeping
// the heavy GL sync off the main thread. This asserts the default path actually engages and
// renders correctly across the engines; pixel-equivalence to the main-thread path is the
// dedicated crosscheck-offscreen.mjs gate. (Skips where OffscreenCanvas/WebGL2 is absent.)
test('OffscreenCanvas worker path is the default and renders all engines', async ({ page }) => {
  test.setTimeout(90000);
  page.on('pageerror', (e) => { throw e; });
  await page.goto('/');                              // no ?offscreen → default (worker when capable)
  await waitDone(page, 0);
  const usable = await page.evaluate(() => window.__viewer._offscreenCapable());
  test.skip(!usable, 'OffscreenCanvas/WebGL2 unavailable in this environment');

  // Home: gpu-naive in the worker.
  expect(await page.evaluate(() => window.__viewer._useOffscreen())).toBe(true);
  expect((await lastDone(page)).engine).toBe('gpu-naive');
  let stats = await canvasStats(page);
  expect(stats.distinctColors).toBeGreaterThan(50);

  // ss=1 keeps the deep render fast under SwiftShader (software GL).
  let cs = await doneCount(page);
  await page.evaluate(() => window.__viewer.setSupersample(1));
  await waitDone(page, cs);

  // Deep df64 (Seahorse Valley, ~2^-41): gpu-perturb in the worker, structured.
  cs = await doneCount(page);
  await page.evaluate(() => window.__viewer.setState({
    cx: '-0.743643887037158704752191506114774',
    cy: '0.131825904205311970493132056385139', radius: 5e-13, maxIter: 8000,
  }));
  await waitDone(page, cs, 45000);
  expect((await lastDone(page)).engine).toBe('gpu-perturb');
  expect(await page.evaluate(() => window.__viewer._useOffscreen())).toBe(true);
  stats = await canvasStats(page);
  expect(stats.max).toBeGreaterThan(0);
  expect(stats.distinctColors).toBeGreaterThan(3);

  // An instant palette recolor (no recompute) still works on the worker path.
  cs = await doneCount(page);
  await page.evaluate(() => window.__viewer.setPalette({ cycle: 120 }));
  // recolor doesn't bump doneCount; just assert the canvas stays non-blank + structured.
  await page.waitForTimeout(200);
  stats = await canvasStats(page);
  expect(stats.distinctColors).toBeGreaterThan(3);
});
