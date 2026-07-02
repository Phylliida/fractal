// perturb.js — double-precision perturbation iteration with Zhuoran rebasing.
//
// Given a reference orbit (Z_n as Float64Arrays, from reference.js) and a
// per-pixel delta dc = c - C (a small double complex), compute the escape count
// of c via the delta recurrence:
//
//     dz_{n+1} = 2 * Z_m * dz_n + dz_n^2 + dc          (m = reference index)
//     true value:  z_{n+1} = Z_{m+1} + dz_{n+1}
//
// Rebasing (Zhuoran): whenever |z| < |dz| OR the reference is exhausted, reset
//     dz := z ;  m := 0
// Because Z_0 = 0, this restarts the delta against the orbit's start using the
// true value; on exhaustion it gracefully degrades to direct iteration (correct).
//
// Escape uses the TRUE value |z|, with the SAME bailout as naive.js so escape
// counts are directly comparable to the oracle.

const LN2 = Math.log(2);

// Smooth escape count for one pixel. Returns { n, sn, glitched }.
//   n   = integer escape iteration (maxIter => inside)
//   sn  = smooth count (NaN if inside)
//   glitched = Pauldelbrot diagnostic flag (rendering does NOT rely on this;
//              rebasing handles glitches — it's exposed for tests/overlays)
// Optional `sa` (from series.computeSeries) skips the first sa.skip iterations: the delta
// is seeded with the SA polynomial dz = a·u + b·u² + c·u³ (u = dc/R, evaluated by Horner)
// and the loop starts at iteration sa.skip+1 with reference index m = sa.skip. This is
// exact (the true value z = Z_skip + dz_skip is representation-independent) and provably
// safe within the probed dc box (see series.js); with sa null/skip 0 the path is
// bit-identical to the original.
export function escapePerturb(ref, dcx, dcy, maxIter, bailoutSq = 1 << 16, glitchTol = 1e-6, sa = null, bla = null, stats = null) {
  const { zx, zy, z2, len } = ref;
  // computeReference always stores at least z_0 and z_1 for maxIter >= 1, so
  // len >= 1 in every real call. Guard the degenerate len === 0 (maxIter 0).
  if (len < 1) return { n: maxIter, sn: NaN, glitched: false };
  let dx = 0, dy = 0;   // dz, relative to Z_m
  let m = 0;            // reference index; invariant at loop top: 0 <= m <= len-1
  let start = 0;        // last iteration already accounted for (SA skip; 0 = none)
  let glitched = false;

  if (sa && sa.skip > 0 && sa.skip < len && sa.skip < maxIter) {
    const Ux = dcx * sa.invR, Uy = dcy * sa.invR;   // u = dc / R
    // Order-5 Horner ((((e·u + d)·u + c)·u + b)·u + a)·u; d,e are 0 for an order-3 sa.
    let hx = sa.ex ?? 0, hy = sa.ey ?? 0, tx, ty;
    tx = hx * Ux - hy * Uy + (sa.dx ?? 0); ty = hx * Uy + hy * Ux + (sa.dy ?? 0); hx = tx; hy = ty;
    tx = hx * Ux - hy * Uy + sa.cx; ty = hx * Uy + hy * Ux + sa.cy; hx = tx; hy = ty;
    tx = hx * Ux - hy * Uy + sa.bx; ty = hx * Uy + hy * Ux + sa.by; hx = tx; hy = ty;
    tx = hx * Ux - hy * Uy + sa.ax; ty = hx * Uy + hy * Ux + sa.ay; hx = tx; hy = ty;
    dx = hx * Ux - hy * Uy; dy = hx * Uy + hy * Ux; // dz_skip relative to Z_skip
    m = sa.skip; start = sa.skip;                   // true value is z = Z_m + dz, m = skip
  }

  // BLA path: skip RUNS of linear iterations via the precomputed bivariate-linear
  // maps (bla.js). Identical escape/rebase logic, just fewer true steps. Lives in its
  // own loop so the no-BLA path below stays bit-identical to the original.
  if (bla) {
    const levels = bla.levels, maxLevel = bla.maxLevel;
    let n = start;
    while (n < maxIter) {
      // Try the largest BLA jump valid at (m, dz): radius covers |dz|², lands at
      // m+L ≤ len-1 (so the post-jump single step's z = Z_{m+L+1} is in range), and
      // n+L ≤ maxIter (don't skip past the iteration cap for an interior pixel).
      const mag2 = dx * dx + dy * dy;
      let jumped = false;
      for (let l = maxLevel; l >= 1; l--) {
        const L = 1 << l;
        if (m + L > len - 1 || n + L > maxIter) continue;
        const lv = levels[l];
        if (m >= lv.r2.length || lv.r2[m] < mag2 || lv.r2[m] === 0) continue;
        // dz_{n+L} = A·dz + B·dc  (complex); A,B from the reference alone.
        const Ax = lv.Ax[m], Ay = lv.Ay[m], Bx = lv.Bx[m], By = lv.By[m];
        const ndx = Ax * dx - Ay * dy + Bx * dcx - By * dcy;
        const ndy = Ax * dy + Ay * dx + Bx * dcy + By * dcx;
        dx = ndx; dy = ndy; m += L; n += L;
        if (stats) { stats.jumps++; stats.jumpIters += L; }
        jumped = true; break;
      }
      if (jumped) continue;
      if (stats) stats.steps++;
      // No BLA valid → one true perturbation step + escape/rebase (the chaotic regime).
      const Zx = zx[m], Zy = zy[m];
      const ndx = 2 * (Zx * dx - Zy * dy) + (dx * dx - dy * dy) + dcx;
      const ndy = 2 * (Zx * dy + Zy * dx) + (2 * dx * dy) + dcy;
      dx = ndx; dy = ndy;
      m++; n++;
      const zfx = zx[m] + dx, zfy = zy[m] + dy;
      const mag2z = zfx * zfx + zfy * zfy;
      if (mag2z > bailoutSq) {
        const logZn = Math.log(mag2z) / 2;
        const nu = Math.log(logZn / LN2) / LN2;
        return { n, sn: n + 1 - nu, glitched };
      }
      const dz2 = dx * dx + dy * dy;
      const rebase = mag2z < dz2 || m === len;
      if (!rebase && z2[m] > 0 && mag2z < glitchTol * z2[m]) glitched = true;
      if (rebase) { dx = zfx; dy = zfy; m = 0; }
    }
    return { n: maxIter, sn: NaN, glitched };
  }

  for (let n = start + 1; n <= maxIter; n++) {
    if (stats) stats.steps++;
    const Zx = zx[m], Zy = zy[m];
    // dz_{n+1} = 2 Z_m dz + dz^2 + dc  (now relative to Z_{m+1})
    const ndx = 2 * (Zx * dx - Zy * dy) + (dx * dx - dy * dy) + dcx;
    const ndy = 2 * (Zx * dy + Zy * dx) + (2 * dx * dy) + dcy;
    dx = ndx; dy = ndy;
    m++;                                 // 1 <= m <= len  (Z_m valid)

    const zfx = zx[m] + dx;              // true value z = Z_m + dz
    const zfy = zy[m] + dy;
    const mag2 = zfx * zfx + zfy * zfy;

    if (mag2 > bailoutSq) {
      const logZn = Math.log(mag2) / 2;
      const nu = Math.log(logZn / LN2) / LN2;
      return { n, sn: n + 1 - nu, glitched };
    }

    // Zhuoran rebasing: restart delta from Z_0=0 using the true value when
    //   (a) |z| < |dz|  (cancellation), or
    //   (b) we reached the end of the reference (must restart to keep Z_m valid).
    const dz2 = dx * dx + dy * dy;
    const rebase = mag2 < dz2 || m === len;

    // Glitch diagnostic (Pauldelbrot): the true value fell far below the
    // reference. A trip implies dz ~= -Z_m so |z| < |dz|, i.e. rebasing fires
    // and FIXES it — so we only flag an *unhandled* glitch (trip without a
    // rebase), which should not occur with rebasing. This is an honest "did
    // anything slip through?" counter, not a count of rebase events.
    if (!rebase && z2[m] > 0 && mag2 < glitchTol * z2[m]) glitched = true;

    if (rebase) { dx = zfx; dy = zfy; m = 0; }
  }
  return { n: maxIter, sn: NaN, glitched };
}

// Render a rectangular region with perturbation. dc per pixel is computed in
// doubles relative to the (high-precision) center, using the SAME pixel->plane
// mapping as naive.renderNaive so results are directly comparable.
//   ref     : reference orbit at the view center
//   view    : { radius, width, height }  (center lives in the reference)
//   region  : { x0, y0, w, h } sub-rectangle in pixels (defaults to full)
// Returns { sn: Float64Array, iters: Int32Array, glitches: number } for region.
export function renderPerturb(ref, view, maxIter, region) {
  const { radius, width, height } = view;
  const r = region || { x0: 0, y0: 0, w: width, h: height };
  const aspect = width / height;
  const scale = (2 * radius) / height;
  const offX = -radius * aspect; // dc.x at px=0
  const offY = -radius;          // dc.y at py=0
  const sn = new Float64Array(r.w * r.h);
  const iters = new Int32Array(r.w * r.h);
  let glitches = 0;
  for (let j = 0; j < r.h; j++) {
    const py = r.y0 + j;
    const dcy = offY + py * scale;
    for (let i = 0; i < r.w; i++) {
      const px = r.x0 + i;
      const dcx = offX + px * scale;
      const res = escapePerturb(ref, dcx, dcy, maxIter);
      const idx = j * r.w + i;
      iters[idx] = res.n;
      sn[idx] = res.n >= maxIter ? -1 : res.sn;
      if (res.glitched) glitches++;
    }
  }
  return { sn, iters, glitches };
}

// Compute dc (double) for a pixel under the same mapping (helper for tests).
export function pixelDelta(view, px, py) {
  const aspect = view.width / view.height;
  const scale = (2 * view.radius) / view.height;
  return { dcx: -view.radius * aspect + px * scale, dcy: -view.radius + py * scale };
}
