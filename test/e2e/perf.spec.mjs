import { test, expect } from '@playwright/test';
import { waitDone, doneCount } from './helpers.mjs';

// Performance-budget assertions (M8 — "time-to-first-pixel, frame budget").
//
// These are deliberately GENEROUS catastrophic-regression guards, NOT tight perf
// gates. The e2e backend is SwiftShader (a CPU software rasterizer) and this host
// is often shared/contended (the companion LLM runs alongside), so a tight wall-
// clock threshold would flake. A normal home render finishes many× under these
// ceilings; the budgets only trip on a GROSS regression — workers failing to spawn
// (deep render goes single-threaded), a maxIter blow-up, an accidental O(n²), or GPU
// init hanging. Deep-path catastrophic regressions are already caught by the explicit
// timeouts on the deep-zoom/Go-navigation e2e tests; this file guards the shallow
// home path (the default engine + the time-to-first-pixel UX) that nothing else times.

// Wall-clock ceilings (ms). See the note above on why they're loose. Observed on
// this host (SwiftShader, lightly contended): home first-pixel ~0.7–1.0s, full
// ~0.7–1.0s, CPU-pool home ~0.5–0.6s. The ceilings sit ~8–24× above that so a CPU
// contention burst (the companion mid-inference pins the cores) can't flake them,
// while an order-of-magnitude regression still trips.
const FIRST_PIXEL_BUDGET = 8_000;    // user must see *something* fast (progressive UX)
const HOME_FULL_BUDGET   = 12_000;   // first complete GPU/default home frame
const CPU_HOME_BUDGET    = 15_000;   // same on the CPU worker pool (the fallback/oracle)

// Poll the canvas until it shows real structure (more than a couple of distinct
// colors over a small corner sample) — i.e. the first painted (coarse) pixels.
async function waitFirstPixel(page, timeout) {
  await page.waitForFunction(() => {
    const c = document.getElementById('view');
    if (!c || !c.width) return false;
    const g = c.getContext('2d');
    const n = Math.min(c.width, 80), m = Math.min(c.height, 80);
    const d = g.getImageData(0, 0, n, m).data;
    const seen = new Set();
    for (let i = 0; i < d.length; i += 4) seen.add((d[i] << 16) | (d[i + 1] << 8) | d[i + 2]);
    return seen.size > 2; // non-blank, has fractal structure (not a flat fill)
  }, null, { timeout });
}

test('perf: home view — time-to-first-pixel and full-frame within budget', async ({ page }) => {
  const t0 = Date.now();
  await page.goto('/');
  await waitFirstPixel(page, FIRST_PIXEL_BUDGET);
  const firstPixelMs = Date.now() - t0;
  await waitDone(page, 0, HOME_FULL_BUDGET);
  const fullMs = Date.now() - t0;
  console.log(`[perf] home: first-pixel ${firstPixelMs}ms, full-frame ${fullMs}ms`);
  expect(firstPixelMs).toBeLessThan(FIRST_PIXEL_BUDGET);
  expect(fullMs).toBeLessThan(HOME_FULL_BUDGET);
  expect(firstPixelMs).toBeLessThanOrEqual(fullMs); // progressive: first pixels never after the full frame
});

test('perf: CPU worker-pool home render within budget (the fallback path)', async ({ page }) => {
  await page.goto('/');
  await waitDone(page, 0);              // initial (default-engine) frame
  const c0 = await doneCount(page);
  const t0 = Date.now();
  await page.evaluate(() => window.__viewer.setUseGpu(false)); // force the CPU pool
  await waitDone(page, c0, CPU_HOME_BUDGET);
  const ms = Date.now() - t0;
  console.log(`[perf] cpu-pool home full-frame ${ms}ms (poolSize=${await page.evaluate(() => window.__viewer.poolSize)})`);
  expect(ms).toBeLessThan(CPU_HOME_BUDGET);
});
