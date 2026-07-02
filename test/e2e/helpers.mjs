// Shared helpers for e2e tests.

// Wait until at least one render has completed (window.__doneCount > prev).
export async function waitDone(page, prev = 0, timeout = 30000) {
  await page.waitForFunction((p) => (window.__doneCount || 0) > p, prev, { timeout });
  return page.evaluate(() => window.__doneCount || 0);
}

export async function doneCount(page) {
  return page.evaluate(() => window.__doneCount || 0);
}

export async function lastDone(page) {
  return page.evaluate(() => window.__lastDone || null);
}

export async function viewState(page) {
  return page.evaluate(() => window.__viewer.getState());
}

// Read the canvas pixels and return basic stats to assert it isn't blank/uniform.
export async function canvasStats(page) {
  return page.evaluate(() => {
    const c = document.getElementById('view');
    const g = c.getContext('2d');
    const { data } = g.getImageData(0, 0, c.width, c.height);
    let min = 255, max = 0, sum = 0, n = data.length / 4;
    const seen = new Set();
    for (let i = 0; i < data.length; i += 4) {
      const lum = (data[i] + data[i + 1] + data[i + 2]) / 3;
      if (lum < min) min = lum; if (lum > max) max = lum; sum += lum;
      if (seen.size < 5000) seen.add((data[i] << 16) | (data[i + 1] << 8) | data[i + 2]);
    }
    return { min, max, mean: sum / n, distinctColors: seen.size, w: c.width, h: c.height };
  });
}

// Simple FNV-1a hash of a downscaled, quantized canvas — a stable "golden" id
// that tolerates tiny AA differences (we quantize to 5 bits/channel on a grid).
export async function canvasFingerprint(page, grid = 16) {
  return page.evaluate((G) => {
    const c = document.getElementById('view');
    const g = c.getContext('2d');
    const { data } = g.getImageData(0, 0, c.width, c.height);
    let h = 0x811c9dc5;
    for (let gy = 0; gy < G; gy++) {
      for (let gx = 0; gx < G; gx++) {
        const px = Math.floor((gx + 0.5) / G * c.width);
        const py = Math.floor((gy + 0.5) / G * c.height);
        const o = (py * c.width + px) * 4;
        const q = ((data[o] >> 3) << 10) | ((data[o + 1] >> 3) << 5) | (data[o + 2] >> 3);
        h ^= q; h = Math.imul(h, 0x01000193) >>> 0;
      }
    }
    return h >>> 0;
  }, grid);
}
