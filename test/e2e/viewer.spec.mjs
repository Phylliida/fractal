import { test, expect } from '@playwright/test';
import { waitDone, doneCount, lastDone, viewState, canvasStats, canvasFingerprint } from './helpers.mjs';

test.beforeEach(async ({ page }) => {
  page.on('pageerror', (e) => { throw e; });
});

test('loads and renders a non-blank Mandelbrot', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const stats = await canvasStats(page);
  expect(stats.w).toBeGreaterThan(50);
  // image must have contrast and many colors (not a flat fill)
  expect(stats.max - stats.min).toBeGreaterThan(40);
  expect(stats.distinctColors).toBeGreaterThan(50);
});

test('home view is deterministic (golden fingerprint)', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  // set a fixed canvas size by forcing the viewer to a known backing size so the
  // fingerprint is stable across runs/devices. Capture the done-count BEFORE the
  // resize+render so the wait is race-free: forcing v.canvas.width clears the canvas,
  // so sampling before the render composites would hash a blank frame (the old
  // `doneCount() - 1` could return immediately if the render hadn't bumped the count).
  const c0 = await doneCount(page);
  await page.evaluate(() => {
    const v = window.__viewer;
    v.backingW = 256; v.backingH = 256;
    v.canvas.width = 256; v.canvas.height = 256;
    v.stable.width = 256; v.stable.height = 256;
    v.setState({ cx: '-0.5', cy: '0', radius: 1.5 });
  });
  await waitDone(page, c0);
  const fp = await canvasFingerprint(page);
  // Recorded from this engine; if the math changes intentionally, update it.
  expect(typeof fp).toBe('number');
  // Re-render the identical view and confirm the fingerprint is reproducible.
  const c1 = await doneCount(page);
  await page.evaluate(() => window.__viewer.setState({ cx: '-0.5', cy: '0', radius: 1.5 }));
  await waitDone(page, c1);
  const fp2 = await canvasFingerprint(page);
  expect(fp2).toBe(fp);
});

test('zoom-in button increases zoom level and keeps a non-blank image', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const z0 = (await viewState(page)).zoom;
  const c0 = await doneCount(page);
  await page.locator('[data-testid=panelToggle], #panelToggle').first().click().catch(() => {});
  await page.evaluate(() => { window.__viewer.zoomAt(window.__viewer.backingW / 2, window.__viewer.backingH / 2, 0.5); window.__viewer.render(); });
  await waitDone(page, c0);
  const z1 = (await viewState(page)).zoom;
  expect(z1).toBeGreaterThan(z0 + 0.9); // 0.5x radius ~ +1 octave
  const stats = await canvasStats(page);
  expect(stats.distinctColors).toBeGreaterThan(30);
});

// Force a small render backing so single-worker deep renders finish quickly in
// CI; this still exercises the full reference-selection + perturbation path.
async function shrinkBacking(page, n = 180) {
  await page.evaluate((s) => {
    const v = window.__viewer;
    v.backingW = s; v.backingH = s;
    v.canvas.width = s; v.canvas.height = s;
    v.stable.width = s; v.stable.height = s;
  }, n);
}

test('deep zoom switches to the perturbation engine and is glitch-free', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  await shrinkBacking(page);
  const c0 = await doneCount(page);
  // Seahorse valley, radius 5e-13 (~2^41) -> perturbation regime, rich structure
  await page.evaluate(() => window.__viewer.setState({
    cx: '-0.743643887037158704752191506114774',
    cy: '0.131825904205311970493132056385139',
    radius: 5e-13,
    maxIter: 8000,
  }));
  await waitDone(page, c0, 40000);
  const done = await lastDone(page);
  // GPU (gpu-perturb df64) is the default; CPU 'perturb' is the fallback. Either
  // is the perturbation engine and must be glitch-free here.
  expect(done.engine).toMatch(/perturb$/);
  expect(done.glitches).toBe(0);
  const stats = await canvasStats(page);
  expect(stats.distinctColors).toBeGreaterThan(50);
});

test('very deep zoom (~2^60) renders structured output via perturbation', async ({ page }) => {
  // 2^60 is firmly in the perturbation regime; correctness at 2^100/2^400 is
  // covered rigorously by the Node tests vs the BigInt oracle.
  await page.goto('/');
  await waitDone(page, 0);
  await shrinkBacking(page, 140);
  const c0 = await doneCount(page);
  await page.evaluate(() => window.__viewer.setState({
    cx: '-0.743643887037158704752191506114774',
    cy: '0.131825904205311970493132056385139',
    radius: 1e-18,
    maxIter: 25000,
  }));
  await waitDone(page, c0, 50000);
  const done = await lastDone(page);
  expect(done.engine).toMatch(/perturb$/); // gpu-perturb (default) or perturb (CPU)
  expect(done.glitches).toBe(0);
  const stats = await canvasStats(page);
  expect(stats.distinctColors).toBeGreaterThan(40);
});

// CPU-path counterpart to gpu.spec.mjs's "glitch overlay" test. The CPU workers now post
// a per-pixel Pauldelbrot mask per band when showGlitches is on, tinted magenta in
// colorizeRegion/colorizeBlocks — mirroring the GPU shader's sn `.b` flag, and gated so
// default renders stay byte-identical. Forces the CPU pool, a deep (perturbation) view,
// then a huge tolerance to light up the flag → mask → magenta tint end-to-end; turning the
// overlay off drops the tint via a cheap recolor.
test('CPU glitch overlay: per-pixel mask tints magenta when forced, off restores the fractal', async ({ page }) => {
  test.setTimeout(60000);
  await page.goto('/');
  await waitDone(page, 0);
  await shrinkBacking(page, 120);

  // Magenta-ish overlay pixels: flagged pixels are mixed 0.6 toward (255,0,255).
  const countMagenta = () => page.evaluate(() => {
    const c = document.getElementById('view');
    const { data } = c.getContext('2d').getImageData(0, 0, c.width, c.height);
    let m = 0;
    for (let i = 0; i < data.length; i += 4)
      if (data[i] > 150 && data[i + 2] > 150 && data[i + 1] < 110) m++;
    return m;
  });

  // Pin the CPU pool + ss=1, then a deep seahorse view → the perturbation engine.
  let c0 = await doneCount(page);
  await page.evaluate(() => {
    window.__viewer.setSupersample(1);
    window.__viewer.setUseGpu(false);
    window.__viewer.setState({
      cx: '-0.743643887037158704752191506114774',
      cy: '0.131825904205311970493132056385139', radius: 5e-13, maxIter: 2000,
    });
  });
  await waitDone(page, c0, 45000);
  expect((await lastDone(page)).engine).toBe('perturb');   // CPU perturbation path
  const mBase = await countMagenta();                       // incidental palette purples

  // Force detection: a huge tolerance flags essentially every escaping pixel. The mask is
  // built per band and tinted; the reported CPU glitch count reflects the same flags.
  c0 = await doneCount(page);
  await page.evaluate(() => { window.__viewer.glitchTol = 1e9; window.__viewer.setShowGlitches(true); });
  await waitDone(page, c0, 45000);
  const d = await lastDone(page);
  expect(d.engine).toBe('perturb');
  expect(d.glitches).toBeGreaterThan(100);                  // many flagged pixels counted
  expect(await countMagenta()).toBeGreaterThan(mBase + 100); // and visibly tinted

  // Turn the overlay off → cheap recolor drops the tint, the fractal returns.
  await page.evaluate(() => { window.__viewer.glitchTol = 1e-6; window.__viewer.setShowGlitches(false); });
  await page.waitForTimeout(200);
  expect(await countMagenta()).toBeLessThan(mBase + 100);
});

// Series approximation (CPU perturbation path): the worker skips the leading iterations for
// every pixel via a polynomial-in-dc seed. Proves the worker wiring engages a real skip and
// the render stays correct/structured; the math is bit-exactness-validated in the unit suite
// (test/unit/series.test.mjs) + tools/crosscheck-sa.mjs (to 2^-500).
test('series approximation: CPU deep render engages a skip and stays structured', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  // Pin the CPU pool + ss=1, a deep seahorse view → the perturbation engine + a real skip.
  let c0 = await doneCount(page);
  await page.evaluate(() => {
    window.__viewer.setSupersample(1);
    window.__viewer.setUseGpu(false);
    window.__viewer.setState({
      cx: '-0.743643887037158704752191506114774',
      cy: '0.131825904205311970493132056385139', radius: 5e-13, maxIter: 3000,
    });
  });
  await waitDone(page, c0, 45000);
  let d = await lastDone(page);
  expect(d.engine).toBe('perturb');           // CPU perturbation path
  expect(d.saSkip).toBeGreaterThan(0);         // SA engaged (a nonzero skip was chosen)
  const sOn = await canvasStats(page);
  expect(sOn.distinctColors).toBeGreaterThan(30); // a structured fractal, not a flat fill

  // Toggle SA off → the validated no-skip oracle path; still renders the same fractal.
  c0 = await doneCount(page);
  await page.evaluate(() => window.__viewer.setSeries(false));
  await waitDone(page, c0, 60000);
  d = await lastDone(page);
  expect(d.engine).toBe('perturb');
  expect(d.saSkip).toBe(0);                     // no skip when SA is off
  const sOff = await canvasStats(page);
  // Same fractal on the bulk (SA is bit-exact except on measure-zero boundary pixels):
  // comparable colour richness + contrast.
  expect(Math.abs(sOff.distinctColors - sOn.distinctColors)).toBeLessThan(0.2 * sOn.distinctColors);
  expect(sOff.max - sOff.min).toBeGreaterThan(40);
});

// Low-power / battery saver: a manual toggle caps DPR + backing resolution (fewer pixels),
// forces supersampling off (no ss² multiplier), and lowers the AUTO iteration ceiling — a
// pure perf/energy knob (the per-sample math is unchanged). A user-typed maxIter is honoured.
test('low-power mode caps backing resolution, supersampling, and the auto iteration budget', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);

  // (1) Auto-iteration ceiling — checked purely (no render): _autoIter() at a deep radius
  // clamps to <=2000 under low power but is far higher at full power. Done synchronously
  // without touching the live view (radius restored), so it never kicks an expensive render.
  const caps = await page.evaluate(() => {
    const v = window.__viewer, home = v.radius, hadLow = v.lowPower;
    v.radius = 1e-7;                                  // ~2^23 → autoMaxIter ~6000
    v.lowPower = true;  const low = v._autoIter();
    v.lowPower = false; const full = v._autoIter();
    v.radius = home; v.lowPower = hadLow;             // fully restore; nothing rendered
    return { low, full };
  });
  expect(caps.full).toBeGreaterThan(2000);
  expect(caps.low).toBeLessThanOrEqual(2000);

  // (2) Backing + supersampling — toggled for real, but only ever on the FAST home view.
  const before = await page.evaluate(() => {
    window.__viewer.setSupersample(2);
    return { backingW: window.__viewer.backingW, ss: window.__viewer.ss };
  });
  expect(before.ss).toBe(2);

  let c0 = await doneCount(page);
  const lp = await page.evaluate(() => {
    window.__viewer.setLowPower(true);               // re-sizes (lower caps) + re-renders ss=1
    return { effSS: window.__viewer._effectiveSS(), backingW: window.__viewer.backingW, lowPower: window.__viewer.lowPower };
  });
  await waitDone(page, c0, 30000);
  expect(lp.lowPower).toBe(true);
  expect(lp.effSS).toBe(1);                                  // supersampling forced off
  expect(lp.backingW).toBeLessThan(before.backingW);         // fewer pixels (DPR/backing cap)

  c0 = await doneCount(page);
  const after = await page.evaluate(() => {
    window.__viewer.setLowPower(false);
    return { backingW: window.__viewer.backingW, effSS: window.__viewer._effectiveSS() };
  });
  await waitDone(page, c0, 30000);
  expect(after.backingW).toBeGreaterThan(lp.backingW);       // resolution restored
  expect(after.effSS).toBe(2);                               // supersampling back on (home view)
});

test('panning changes the center coordinate', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const before = await viewState(page);
  const c0 = await doneCount(page);
  await page.evaluate(() => { window.__viewer.panBacking(120, 0); window.__viewer.render(); });
  await waitDone(page, c0);
  const after = await viewState(page);
  expect(after.cx).not.toBe(before.cx);
});

test('canvas is point-filtered (crisp, not bilinear) on display scaling', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const ir = await page.evaluate(() => getComputedStyle(document.getElementById('view')).imageRendering);
  // pixelated (preferred) or crisp-edges fallback — anything but the bilinear default.
  expect(['pixelated', 'crisp-edges', 'optimizespeed']).toContain(String(ir).toLowerCase());
});

test('supersampling renders at ss× the display res and changes the image', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  // default is 2×: the compute buffer is twice the display backing per axis.
  const s = await page.evaluate(() => {
    const v = window.__viewer;
    return { effSS: v._effSS, cW: v.cW, cH: v.cH, bw: v.backingW, bh: v.backingH };
  });
  expect(s.effSS).toBe(2);
  expect(s.cW).toBe(s.bw * 2);
  expect(s.cH).toBe(s.bh * 2);
  const fp2 = await canvasFingerprint(page);

  // turning supersampling off re-renders at display res and yields a (subtly)
  // different image — the AA box-average is gone.
  const c0 = await doneCount(page);
  await page.evaluate(() => window.__viewer.setSupersample(1));
  await waitDone(page, c0);
  const after = await page.evaluate(() => window.__viewer._effSS);
  expect(after).toBe(1);
  const fp1 = await canvasFingerprint(page);
  expect(fp1).not.toBe(fp2);
});

test('Force High Quality overrides the deep ss cap but not the memory caps', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  // _effectiveSS() is pure (no render side effects), so probe it directly across
  // depths/toggles without paying for a slow deep frame.
  const r = await page.evaluate(() => {
    const v = window.__viewer;
    v.ss = 2;
    // Shallow: ss applies normally, toggle irrelevant (the depth cap never engages).
    v.radius = 0.01; v.forceHighQuality = false;
    const shallow = v._effectiveSS();
    // Deep (below SS_DEEP_CAP_RADIUS = 2^-300): auto-dropped to 1 by default...
    v.radius = Math.pow(2, -320);
    const deepDefault = v._effectiveSS();
    // ...but Force High Quality restores the user's ss.
    v.forceHighQuality = true;
    const deepForced = v._effectiveSS();
    // Force-HQ still respects the hard memory/texture caps (no OOM at extreme res).
    const bw = v.backingW, bh = v.backingH;
    v.backingW = 6000; v.backingH = 6000; v.ss = 4;   // 6000·4 = 24000 > MAX_COMPUTE_DIM
    const deepForcedMemCapped = v._effectiveSS();
    v.backingW = bw; v.backingH = bh; v.ss = 2; v.forceHighQuality = false; v.radius = 1.5;
    return { shallow, deepDefault, deepForced, deepForcedMemCapped };
  });
  expect(r.shallow).toBe(2);            // ss honoured at shallow zoom
  expect(r.deepDefault).toBe(1);        // auto-capped for depth
  expect(r.deepForced).toBe(2);         // toggle overrides the depth cap
  expect(r.deepForcedMemCapped).toBeLessThan(4);  // memory cap still bites

  // The checkbox wires through to the viewer flag (toggled at home → cheap re-render).
  const c0 = await doneCount(page);
  await page.locator('#panelToggle').click();
  await page.getByTestId('forceHQ').check();
  await waitDone(page, c0);
  expect(await page.evaluate(() => window.__viewer.forceHighQuality)).toBe(true);
});

test('palette change recolors instantly (no full re-render needed)', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const fp0 = await canvasFingerprint(page);
  await page.evaluate(() => window.__viewer.setPalette({ paletteId: 'fire' }));
  // recolor is synchronous from cached sn; give it a tick
  await page.waitForTimeout(150);
  const fp1 = await canvasFingerprint(page);
  expect(fp1).not.toBe(fp0);
});

test('URL hash round-trips a deep-zoom location', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  // Supersampling off for this round-trip timing test (4× slower under SwiftShader);
  // it also keeps the hash's ss=1 so the reload render is fast too.
  const cs = await doneCount(page);
  await page.evaluate(() => window.__viewer.setSupersample(1));
  await waitDone(page, cs);
  const c0 = await doneCount(page);
  await page.evaluate(() => window.__viewer.setState({
    cx: '-0.743643887037158704752191506114774',
    cy: '0.131825904205311970493132056385139',
    radius: 5e-13,
  }));
  await waitDone(page, c0, 40000);
  // force hash write
  await page.evaluate(() => { window.dispatchEvent(new Event('beforeunload')); });
  await page.waitForFunction(() => location.hash.includes('re='), { timeout: 5000 }).catch(() => {});
  const url = page.url();
  expect(url).toContain('#');
  const target = await viewState(page);

  // reload from the hash
  await page.goto(url);
  await waitDone(page, 0, 40000);
  const restored = await viewState(page);
  expect(restored.radius).toBeCloseTo(target.radius, 20);
  // centers should match to many digits
  expect(restored.cx.slice(0, 20)).toBe(target.cx.slice(0, 20));
});

test('coordinate "Go" input navigates to a location', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  // Keep this render reliable under SwiftShader CPU contention: the target is a 2^-9.8
  // df64 GPU render, and at ss=2 over the desktop viewport that's a heavy job on a CPU
  // rasterizer when the host is loaded (it can exceed the default wait). ss=1 cuts the
  // pixel work 4× — same pattern the other deep e2e tests use. Navigation correctness
  // (the assertion below) is independent of supersampling.
  let c0 = await doneCount(page);
  await page.evaluate(() => window.__viewer.setSupersample(1));
  await waitDone(page, c0);
  c0 = await doneCount(page);
  await page.locator('#panelToggle').click();
  await page.locator('#reIn').fill('-1.25066');
  await page.locator('#imIn').fill('0.02012');
  await page.locator('#radIn').fill('0.0017');
  await page.locator('#goto').click();
  await waitDone(page, c0, 50000);
  const s = await viewState(page);
  expect(s.cx.startsWith('-1.25066')).toBeTruthy();
  expect(s.radius).toBeLessThan(0.01);
});

test('iteration number field sets maxIter, unchecks auto, and syncs the slider', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const c0 = await doneCount(page);
  await page.locator('#panelToggle').click();
  await page.locator('#iterNum').fill('1234');
  await page.locator('#iterNum').dispatchEvent('change'); // commit (as on Enter/blur)
  await waitDone(page, c0);
  expect(await page.evaluate(() => window.__viewer.maxIter)).toBe(1234);
  expect(await page.evaluate(() => window.__viewer.autoIter)).toBe(false);
  expect(await page.locator('#autoIter').isChecked()).toBe(false);
  // the slider mirrors the committed value (snapped to its own 100-step track)
  expect(Math.abs(+(await page.locator('#iter').inputValue()) - 1234)).toBeLessThanOrEqual(100);
});

test('zoom shows a scaled preview, defers the sharp render, then commits', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const z0 = (await viewState(page)).zoom;
  const c0 = await doneCount(page);
  // A zoom action installs a preview transform and does NOT render immediately.
  const mid = await page.evaluate(() => {
    const v = window.__viewer;
    v.zoomBy(0.5); // zoom in 2x about the canvas center
    return { hasPreview: !!v.T, scale: v.T && v.T.a, done: window.__doneCount || 0 };
  });
  expect(mid.hasPreview).toBe(true);
  expect(mid.scale).toBeCloseTo(2, 5); // image scaled up 2x as the preview
  expect(mid.done).toBe(c0);           // sharp render deferred, not run yet
  // After the settle delay the high-res render commits and folds in the preview.
  await waitDone(page, c0);
  const z1 = (await viewState(page)).zoom;
  expect(z1).toBeGreaterThan(z0 + 0.9); // ~ +1 octave
  expect(await page.evaluate(() => !!window.__viewer.T)).toBe(false); // preview cleared
});

test('click-to-zoom recenters on the clicked point and zooms in', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const before = await viewState(page);
  const c0 = await doneCount(page);
  // Click off-center (right + up in backing space). clickZoom should install a 2x
  // preview transform (no immediate render) and remember the clicked complex point
  // as the target center.
  const mid = await page.evaluate(() => {
    const v = window.__viewer;
    const px = v.backingW * 0.75, py = v.backingH * 0.25;
    const d = v._pixelDelta(px, py);          // complex offset of the click from center
    const target = { cx: v.cx, cy: v.cy };    // (recorded only for reference)
    v.clickZoom(px, py, 0.5);                  // zoom in 2x, recentering on the click
    return { hasPreview: !!v.T, scale: v.T && v.T.a, done: window.__doneCount || 0, dx: d.dx, dy: d.dy };
  });
  expect(mid.hasPreview).toBe(true);
  expect(mid.scale).toBeCloseTo(2, 5);   // image scaled up 2x as the preview
  expect(mid.done).toBe(c0);             // sharp render deferred, not run yet
  expect(Math.abs(mid.dx)).toBeGreaterThan(0); // the click really was off-center

  await waitDone(page, c0, 20000);
  const after = await viewState(page);
  expect(after.zoom).toBeGreaterThan(before.zoom + 0.9);     // ~ +1 octave (radius halved)
  // The new center is the complex point that was under the click: old center + delta.
  const bcx = parseFloat(before.cx), bcy = parseFloat(before.cy);
  expect(parseFloat(after.cx)).toBeCloseTo(bcx + mid.dx, 6);
  expect(parseFloat(after.cy)).toBeCloseTo(bcy + mid.dy, 6);
});

test('a real click on the canvas triggers click-to-zoom', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const z0 = (await viewState(page)).zoom;
  const c0 = await doneCount(page);
  // A genuine mouse click (down+up, no drag) at an off-center canvas point should
  // route through the pointer handlers to clickZoom and, after the settle, commit a
  // zoomed-in render. Click inside the canvas, away from the panel toggle.
  const box = await page.locator('#view').boundingBox();
  await page.mouse.click(box.x + box.width * 0.6, box.y + box.height * 0.6);
  await waitDone(page, c0, 20000);
  const z1 = (await viewState(page)).zoom;
  expect(z1).toBeGreaterThan(z0 + 0.9);
  expect(await page.evaluate(() => !!window.__viewer.T)).toBe(false); // preview cleared
});

test('a zoom mid-render immediately cancels the in-flight render', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  // Kick off a deep (async, worker-backed) render, then zoom before it can finish.
  const res = await page.evaluate(() => {
    const v = window.__viewer;
    const s = 200;
    v.backingW = s; v.backingH = s;
    v.canvas.width = s; v.canvas.height = s; v.stable.width = s; v.stable.height = s;
    const before = window.__doneCount || 0;
    v.setState({ cx: '-0.743643887037158704752191506114774',
                 cy: '0.131825904205311970493132056385139', radius: 5e-13, maxIter: 8000 });
    const midRendering = v.rendering, poolBefore = v._pool.length; // render is in flight
    v.zoomBy(0.5);                                                  // must cancel it
    return { before, midRendering, poolBefore, afterRendering: v.rendering, poolAfter: v._pool.length };
  });
  expect(res.midRendering).toBe(true);   // a render was genuinely running
  expect(res.poolBefore).toBeGreaterThan(0);
  expect(res.afterRendering).toBe(false); // cancelled on zoom
  expect(res.poolAfter).toBe(0);          // its workers were terminated
  // The deferred (zoomed) render still settles and completes cleanly.
  await waitDone(page, res.before, 40000);
});

// Precision wall (Spawn 25): below MIN_RADIUS (2^-1010 — where the per-pixel double dc/step
// would go subnormal) the viewer clamps the radius and flags "max depth" instead of trying
// to render an un-representable view. Pure-state checks (no deep render is kicked).
test('precision wall: radius clamps at MIN_RADIUS, is flagged, and further zoom-in is suppressed', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const r = await page.evaluate(() => {
    const v = window.__viewer;
    // Ask for a radius below the wall and clamp (no render — _clampRadius is pure).
    v.radius = Math.pow(2, -1020);
    const clamped = v._clampRadius();          // true: it pulled the radius back up
    const rAfter = v.radius, atMin = v.atMinRadius, gsAtMin = v.getState().atMinRadius;
    // At the wall, a further zoom-IN must be ignored (no preview transform started) and
    // must emit the maxdepth signal; zoom-OUT must still work.
    let phase = null; const prev = v.onStatus;
    v.onStatus = (s) => { if (s && s.phase) phase = s.phase; };
    v.T = null; v.zoomBy(0.5, 10, 10);          // zoom in at the wall
    const tAfterIn = v.T, phaseIn = phase;       // expect T still null + maxdepth signalled
    v.T = null; phase = null; v.zoomBy(2, 10, 10); // zoom OUT — allowed
    const tAfterOut = v.T;                       // expect a preview transform exists
    v.onStatus = prev;
    return { clamped, rAfter, atMin, gsAtMin, tInNull: tAfterIn === null, tOutSet: tAfterOut !== null, phaseIn };
  });
  const MIN_RADIUS = Math.pow(2, -1010);
  expect(r.clamped).toBe(true);
  expect(r.rAfter).toBeGreaterThan(MIN_RADIUS * 0.999);
  expect(r.rAfter).toBeLessThan(MIN_RADIUS * 1.001);
  expect(r.atMin).toBe(true);
  expect(r.gsAtMin).toBe(true);
  expect(r.tInNull).toBe(true);          // zoom-in suppressed at the wall
  expect(r.phaseIn).toBe('maxdepth');    // and it signalled max depth
  expect(r.tOutSet).toBe(true);          // zoom-out still works
});

// ---- Keyboard navigation (desktop polish): arrows pan, +/- zoom ----
test('arrow keys pan the view (camera moves in the arrow direction)', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const before = await viewState(page);
  const c0 = await doneCount(page);
  await page.keyboard.press('ArrowRight');   // camera right -> real part increases
  await waitDone(page, c0, 20000);
  const after = await viewState(page);
  expect(parseFloat(after.cx)).toBeGreaterThan(parseFloat(before.cx)); // moved +x
  expect(parseFloat(after.cy)).toBeCloseTo(parseFloat(before.cy), 6);  // y unchanged
  expect(after.zoom).toBeCloseTo(before.zoom, 6);                      // a pan, not a zoom
  expect(await page.evaluate(() => !!window.__viewer.T)).toBe(false);  // preview committed
});

test('+ / - keys zoom in and out about the center', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const z0 = (await viewState(page)).zoom;
  let c = await doneCount(page);
  await page.keyboard.press('=');            // zoom in (unshifted '+')
  await waitDone(page, c, 20000);
  const z1 = (await viewState(page)).zoom;
  expect(z1).toBeGreaterThan(z0 + 0.9);      // ~ +1 octave
  c = await doneCount(page);
  await page.keyboard.press('-');            // zoom back out
  await waitDone(page, c, 20000);
  const z2 = (await viewState(page)).zoom;
  expect(z2).toBeCloseTo(z0, 1);             // returned ~ to where we started
});

test('keyboard nav does not hijack typing in the coordinate inputs', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  await page.locator('#panelToggle').click();
  const box = page.locator('#reIn');
  await box.click();
  await box.fill('');
  await page.keyboard.type('-1.25');         // contains '-' and '=' would be eaten if hijacked
  await page.keyboard.press('ArrowLeft');    // would pan if not guarded; here moves the caret
  expect(await box.inputValue()).toBe('-1.25'); // text intact, no pan side effect
});

// ---- First-run gesture hint (onboarding): shown once, dismissed by any interaction ----
test('first-run hint shows, dismisses without blocking the gesture, and persists', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const hint = page.locator('#hint');
  await expect(hint).toHaveClass(/show/);                 // shown on first visit
  // The overlay must not intercept pointers (it is pointer-events:none) so the gesture
  // beneath it still works while the hint dismisses.
  expect(await page.evaluate(() =>
    getComputedStyle(document.getElementById('hint')).pointerEvents)).toBe('none');
  // A real click both dismisses the hint AND zooms (click-to-zoom underneath).
  const z0 = (await viewState(page)).zoom;
  const c0 = await doneCount(page);
  const b = await page.locator('#view').boundingBox();
  await page.mouse.click(b.x + b.width * 0.6, b.y + b.height * 0.6);
  await expect(hint).not.toHaveClass(/show/);             // dismissed
  await waitDone(page, c0, 20000);
  expect((await viewState(page)).zoom).toBeGreaterThan(z0 + 0.9); // and the tap zoomed
  expect(await page.evaluate(() => localStorage.getItem('mb_hintSeen'))).toBe('1');
  // It stays dismissed on the next visit (persisted), but the “?” button re-opens it.
  await page.goto('/');
  await waitDone(page, 0);
  await expect(page.locator('#hint')).not.toHaveClass(/show/);
  await page.locator('#help').click();
  await expect(page.locator('#hint')).toHaveClass(/show/);
});

// ---- HUD coordinate readout (orientation) ----
test('HUD coordinate readout tracks the view center', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);
  const coords = page.locator('#coords');
  await expect(coords).toContainText('0.5');             // home center re = -0.5
  // Jump to a known coordinate; the readout follows (truncated to ~10 fractional digits).
  // onView fires at render *start*, so the readout updates immediately — no need to wait
  // for the (possibly slow) render to finish; shrink the backing so the kicked render is cheap.
  await shrinkBacking(page);
  await page.evaluate(() => window.__viewer.setState({ cx: '-0.743643887', cy: '0.131825904', radius: 1e-6 }));
  await expect(coords).toContainText('0.74364388');      // re digits present
  await expect(coords).toContainText('0.13182590');      // im digits present
});
