// bla.js — Bivariate Linear Approximation (BLA) for perturbation Mandelbrot.
//
// Series approximation (series.js) skips the LEADING run of perturbation iterations
// once, while the delta dz is in its initial linear regime. But the deep-zoom orbit
// is a SEQUENCE of linear regimes: every Zhuoran rebase resets dz to a small true
// value (dz = z, m = 0) from which it grows linearly AGAIN before the next rebase /
// escape. SA cannot touch those post-rebase re-growth phases — they are exactly the
// "kept" chaotic iterations that dominate a deep render after SA. BLA skips them too.
//
// THE IDEA. The exact delta step is
//     dz_{n+1} = 2·Z_n·dz_n + dz_n² + dc.
// While |dz_n| is small the quadratic dz_n² is negligible and the step is LINEAR in
// (dz_n, dc). A run of L linear steps from reference index m composes to a single
// bivariate-linear map
//     dz_{m+L} = A·dz_m + B·dc
// with complex A, B precomputed FROM THE REFERENCE ALONE (shared by every pixel). A
// pixel sitting at (m, dz) with |dz| inside the run's validity radius r can jump L
// iterations in one A·dz+B·dc evaluation instead of L true steps. After a rebase it
// re-engages from m=0. This is the Kalles-Fraktaler / Zhuoran BLA.
//
// CONSTRUCTION (a binary merge tree; all formulas cross-checked + validated against
// the no-BLA escape-count oracle, see tools/crosscheck-bla.mjs):
//   level 0 (one step at m):  A = 2·Z_m,  B = 1,  r = blaEps·|2·Z_m|
//       r is the radius where the dropped |dz|² ≤ blaEps·|2·Z_m·dz| (the linear term).
//       r is forced to 0 when |Z_m| ≥ ZMAX so no BLA run can approach the bailout
//       radius (then |z| = |Z+dz| ≈ |Z| < 2 ≪ 256, no escape can be skipped).
//   merge x (l_x steps from m) then y (l_y steps from m+l_x):
//       A = A_y·A_x,  B = A_y·B_x + B_y,
//       r = min( r_x, (r_y − |B_x|·dcMax) / |A_x| )   clamped to ≥ 0
//       (after x, dz' = A_x·dz + B_x·dc must stay within r_y for ANY |dc| ≤ dcMax;
//        the triangle inequality gives the second term). |A| grows like 2^L and can
//        overflow a double — any non-finite coefficient forces r = 0 so that BLA is
//        never applied (the radius check screens it out).
//
// SAFETY of skipping the escape + rebase tests inside a run (validated):
//   - no rebase: a rebase needs |z| < |dz|, i.e. |Z| < 2|dz|; but |dz| ≤ r = blaEps·
//     |2·Z| ≪ |Z|, so the orbit never drifts far enough to rebase mid-run.
//   - no escape: the ZMAX guard keeps |Z| < 2 across the whole run, so |z| ≈ |Z| < 2,
//     far below the bailout — no escape can be hidden. Jumps are also clamped so they
//     never overshoot maxIter or the reference length.

// |Z_m| ≥ this ⇒ no BLA at m (keeps every run's |z| far from the bailout radius).
const ZMAX = 2.0;
const ZMAX2 = ZMAX * ZMAX;

// Build the BLA table for a reference orbit.
//   ref    : { zx, zy, len }  (zx/zy index n = Z_n; valid 0..len)
//   dcMax  : max |dc| over the image box (the corner) — bounds dz growth in a run
//   opts.eps : validity tolerance blaEps (default 2^-30). Smaller = safer / less skip.
//     The accuracy/speed knee, measured vs the BigInt-exact oracle on the deep boundary
//     coordinate (tools/crosscheck-bla.mjs + arbiter-bla.mjs), per-pixel escape counts:
//       2^-32 : ~0 mism, ~2.0–2.7× work reduction ON TOP OF series approximation
//       2^-30 : ≤3 pixels off by ≤~19 (near-maxIter, visually invisible), ~2.5–3.3×
//       2^-28 : ~0.3% of pixels off by a few, ~3.1–4.0×
//     The drift is BLA truncation (the dropped dz²) accumulating over a long orbit, NOT a
//     missed escape (the ZMAX guard + radius forbid that). It is FAR below the GPU df64
//     mismatch envelope (0.3–1%), so the eventual GPU port can run a looser eps gated on
//     "GPU+BLA vs CPU oracle stays in the df64 envelope" (see NOTES "BLA").
// Returns { levels, maxLevel, eps } where levels[l] = { Ax,Ay,Bx,By,r2 } (Float64Array
// indexed by start m, covering 2^l steps; r2 = r², 0 where unusable). levels[l][m] is
// defined for m in [0, len − 2^l]; outside that range r2 is 0.
export function buildBLA(ref, dcMax, opts = {}) {
  const eps = opts.eps ?? Math.pow(2, -30);
  const { zx, zy, len } = ref;
  // Highest level whose run still fits: 2^maxLevel ≤ len.
  let maxLevel = 0;
  while ((1 << (maxLevel + 1)) <= len) maxLevel++;

  // ---- level 0: single steps -------------------------------------------------
  const n0 = len;                       // starts m = 0 .. len-1 (step to m+1 ≤ len)
  const Ax0 = new Float64Array(n0), Ay0 = new Float64Array(n0);
  const Bx0 = new Float64Array(n0), By0 = new Float64Array(n0);
  const r20 = new Float64Array(n0);
  for (let m = 0; m < n0; m++) {
    const zr = zx[m], zi = zy[m];
    Ax0[m] = 2 * zr; Ay0[m] = 2 * zi;   // A = 2 Z_m
    Bx0[m] = 1; By0[m] = 0;             // B = 1
    const z2 = zr * zr + zi * zi;
    if (z2 < ZMAX2) {
      const r = eps * 2 * Math.sqrt(z2);  // blaEps·|2 Z_m|
      r20[m] = r * r;
    } // else r2 stays 0 (no BLA here)
  }
  const levels = [{ Ax: Ax0, Ay: Ay0, Bx: Bx0, By: By0, r2: r20 }];

  // ---- merge upward ----------------------------------------------------------
  for (let l = 1; l <= maxLevel; l++) {
    const half = 1 << (l - 1);
    const span = 1 << l;
    const count = len - span + 1;       // valid starts m = 0 .. len-span
    if (count <= 0) { maxLevel = l - 1; break; }
    const Ax = new Float64Array(count), Ay = new Float64Array(count);
    const Bx = new Float64Array(count), By = new Float64Array(count);
    const r2 = new Float64Array(count);
    const lo = levels[l - 1];
    for (let m = 0; m < count; m++) {
      // x = lo[m]  (steps m..m+half), y = lo[m+half] (steps m+half..m+span)
      const axx = lo.Ax[m], axy = lo.Ay[m], bxx = lo.Bx[m], bxy = lo.By[m], rx2 = lo.r2[m];
      const k = m + half;
      const ayx = lo.Ax[k], ayy = lo.Ay[k], byx = lo.Bx[k], byy = lo.By[k], ry2 = lo.r2[k];
      // A = A_y · A_x  (complex)
      const nAx = ayx * axx - ayy * axy, nAy = ayx * axy + ayy * axx;
      // B = A_y · B_x + B_y
      const nBx = ayx * bxx - ayy * bxy + byx, nBy = ayx * bxy + ayy * bxx + byy;
      Ax[m] = nAx; Ay[m] = nAy; Bx[m] = nBx; By[m] = nBy;
      // r = min( r_x, (r_y − |B_x|·dcMax) / |A_x| ), clamped ≥ 0; 0 if any coeff blew up.
      let r = 0;
      if (rx2 > 0 && ry2 > 0 && isFinite(nAx) && isFinite(nAy) && isFinite(nBx) && isFinite(nBy)) {
        const ax = Math.hypot(axx, axy);          // |A_x|
        const bx = Math.hypot(bxx, bxy);          // |B_x|
        const ry = Math.sqrt(ry2);
        const cand = (ry - bx * dcMax) / ax;       // input radius the y-part allows
        const rx = Math.sqrt(rx2);
        r = Math.min(rx, cand);
        if (!(r > 0)) r = 0;                        // NaN / ≤0 → unusable
      }
      r2[m] = r * r;
    }
    levels.push({ Ax, Ay, Bx, By, r2 });
  }
  return { levels, maxLevel, eps, len };
}

// --- GPU texture packing ----------------------------------------------------
//
// Pack the BLA table for upload as an RGBA32F texture the rescaled deep shader can
// texelFetch. The shader scans levels 1..maxLevel (level 0 is single-step — only the
// merge tree uses it, never a jump), so only those are uploaded.
//
// WHY FLOATEXP, NOT PLAIN df64 (measured — tools/probe-bla-mag.mjs): at deep zoom the
// USABLE/APPLIED BLA coefficients run far outside the float32 exponent range — at 2^-400
// the jumps a real render applies hit |A|~2^228, |B|~2^231, r²~2^-576. Plain df64 (float32
// hi/lo) would overflow A,B and underflow r²; the table is only useful BELOW the df64 floor,
// exactly where that happens. So every coefficient carries an int exponent (floatexp): each
// complex A,B as a df64 mantissa pair under ONE shared exponent (normalized so the larger
// component's |mantissa| is in [0.5,1), matching the shader's rescaled dz form), and r² as a
// single-float mantissa + exponent. The shader fe_norm's each component after the fetch.
//
// Layout — 3 RGBA32F texels per (level,m) entry, entry index E = (l−1)·len + m (uniform
// stride `len` per level; invalid m ≥ len−2^l left as r²=0 → never applied):
//   texel 0: Ax.hi, Ax.lo, Ay.hi, Ay.lo      (A mantissas, shared exponent Ae)
//   texel 1: Bx.hi, Bx.lo, By.hi, By.lo      (B mantissas, shared exponent Be)
//   texel 2: Ae,    Be,    r2e,   r2m        (exponents as exact-int floats; r2 single-float mantissa)
// Returns { data:Float32Array, texW, texH, maxLevel, len, entries }.
const FE_ZERO_E = -2000000;             // exponent sentinel for value 0 (matches shader intent)

// Complex (cx,cy) double -> shared-exponent floatexp: [mxHi,mxLo,myHi,myLo,e] with
// value cx = (mxHi+mxLo)·2^e, cy = (myHi+myLo)·2^e and max(|mantissa|) in [0.5,1).
// Scaling is done incrementally (never forms 2^±e) so it can't overflow even when r²-class
// exponents reach ~±660; non-finite (overflowed-merge) coeff -> zero (its entry has r²=0).
function feSharedComplex(cx, cy) {
  if (!isFinite(cx) || !isFinite(cy)) return [0, 0, 0, 0, FE_ZERO_E];
  let a = Math.max(Math.abs(cx), Math.abs(cy));
  if (a === 0) return [0, 0, 0, 0, FE_ZERO_E];
  let e = 0, mx = cx, my = cy, s = a;
  while (s >= 1) { s *= 0.5; mx *= 0.5; my *= 0.5; e++; }
  while (s < 0.5) { s *= 2; mx *= 2; my *= 2; e--; }
  const mxHi = Math.fround(mx), myHi = Math.fround(my);
  return [mxHi, Math.fround(mx - mxHi), myHi, Math.fround(my - myHi), e];
}
// Non-negative scalar v -> floatexp [m,e] with v = m·2^e, m in [0.5,1) (single float).
function feScalar(v) {
  if (!(v > 0) || !isFinite(v)) return [0, FE_ZERO_E];
  let e = 0, s = v;
  while (s >= 1) { s *= 0.5; e++; }
  while (s < 0.5) { s *= 2; e--; }
  return [Math.fround(s), e];
}

export function blaToFloat32(bla, texW = 2048) {
  const { levels, maxLevel, len } = bla;
  const entries = maxLevel * len;                 // levels 1..maxLevel, `len` slots each
  const totalTexels = entries * 3;
  const texH = Math.max(1, Math.ceil(totalTexels / texW));
  const data = new Float32Array(texW * texH * 4); // RGBA; zero-init => r²=0 (unusable) by default
  for (let l = 1; l <= maxLevel; l++) {
    const lv = levels[l];
    const span = 1 << l;
    const validCount = len - span + 1;            // valid m = 0 .. len−2^l
    const base = (l - 1) * len;
    for (let m = 0; m < validCount; m++) {
      const r2 = lv.r2[m];
      const t0 = (base + m) * 3;                  // base texel of this entry
      const A = feSharedComplex(lv.Ax[m], lv.Ay[m]);
      const B = feSharedComplex(lv.Bx[m], lv.By[m]);
      const R = feScalar(r2);                     // [r2m, r2e]; r2m=0 => never applies
      let o = t0 * 4;
      data[o] = A[0]; data[o + 1] = A[1]; data[o + 2] = A[2]; data[o + 3] = A[3];
      o = (t0 + 1) * 4;
      data[o] = B[0]; data[o + 1] = B[1]; data[o + 2] = B[2]; data[o + 3] = B[3];
      o = (t0 + 2) * 4;
      data[o] = A[4]; data[o + 1] = B[4]; data[o + 2] = R[1]; data[o + 3] = R[0];
    }
  }
  return { data, texW, texH, maxLevel, len, entries };
}
