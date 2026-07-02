// naive.js — ground-truth escape-time Mandelbrot in plain double precision.
//
// This module is the ORACLE. At shallow zoom (radius >= ~2^-40) it is exact and
// every higher-precision engine (reference orbit, perturbation) is validated
// against it. Keep it dead simple and obviously correct. No DOM, pure ESM.

export const BAILOUT = 2;
export const BAILOUT_SQ = BAILOUT * BAILOUT; // |z|^2 > 4 means escaped

// Smooth (continuous) iteration count for a single point c = (cx, cy).
// Returns { n, sn } where:
//   n  = integer escape iteration (n === maxIter means "did not escape" / inside)
//   sn = smooth/normalized iteration count (fractional), NaN if not escaped
// Uses a bailout radius large enough that the smoothing formula is accurate.
export function escapeNaive(cx, cy, maxIter, bailoutSq = 1 << 16) {
  // Cardioid / period-2 bulb quick rejection (big speedup, exact, optional).
  // Points inside the main cardioid or the period-2 bulb never escape.
  const xm = cx - 0.25;
  const q = xm * xm + cy * cy;
  if (q * (q + xm) <= 0.25 * cy * cy) return { n: maxIter, sn: NaN };
  const xp = cx + 1;
  if (xp * xp + cy * cy <= 0.0625) return { n: maxIter, sn: NaN };

  let zx = 0, zy = 0;
  let zx2 = 0, zy2 = 0; // zx*zx, zy*zy
  let n = 0;
  while (n < maxIter && zx2 + zy2 <= bailoutSq) {
    zy = 2 * zx * zy + cy;
    zx = zx2 - zy2 + cx;
    zx2 = zx * zx;
    zy2 = zy * zy;
    n++;
  }
  if (n >= maxIter) return { n: maxIter, sn: NaN };
  // Smooth coloring: nu = n + 1 - log2(log(|z|)) (using log of modulus).
  const logZn = Math.log(zx2 + zy2) / 2;
  const nu = Math.log(logZn / Math.log(2)) / Math.log(2);
  const sn = n + 1 - nu;
  return { n, sn };
}

// Raw orbit (no bailout-radius smoothing) — used to compare against the
// high-precision reference orbit. Returns arrays of zx, zy up to escape/maxIter.
// escapeIdx = index at which |z|^2 first exceeded 4 (or maxIter if none).
export function orbitNaive(cx, cy, maxIter) {
  const zx = new Float64Array(maxIter + 1);
  const zy = new Float64Array(maxIter + 1);
  let x = 0, y = 0;
  let i = 0;
  zx[0] = 0; zy[0] = 0;
  for (; i < maxIter; i++) {
    const nx = x * x - y * y + cx;
    const ny = 2 * x * y + cy;
    x = nx; y = ny;
    zx[i + 1] = x; zy[i + 1] = y;
    if (x * x + y * y > 4) { i++; break; }
  }
  return { zx, zy, len: i, escapeIdx: i };
}

// Render a full grid into a Float64Array of smooth counts (NaN = inside).
// view: { centerX, centerY, radius, width, height } radius = half-height.
// Returns { sn: Float64Array, iters: Int32Array } row-major, length w*h.
export function renderNaive(view, maxIter) {
  const { centerX, centerY, radius, width, height } = view;
  const sn = new Float64Array(width * height);
  const iters = new Int32Array(width * height);
  const aspect = width / height;
  const scale = (2 * radius) / height; // complex units per pixel
  const x0 = centerX - radius * aspect;
  const y0 = centerY - radius;
  for (let py = 0; py < height; py++) {
    const cy = y0 + py * scale;
    for (let px = 0; px < width; px++) {
      const cx = x0 + px * scale;
      const r = escapeNaive(cx, cy, maxIter);
      const idx = py * width + px;
      iters[idx] = r.n;
      sn[idx] = r.n >= maxIter ? -1 : r.sn;
    }
  }
  return { sn, iters };
}
