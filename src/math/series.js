// series.js — series approximation (SA): skip the first N perturbation iterations
// for EVERY pixel by approximating the delta orbit as a polynomial in dc.
//
// The delta recurrence dz_{n+1} = 2 Z_n dz_n + dz_n² + dc, expanded as a power series
// in dc with Z_0 = 0, dz_0 = 0, has coefficients (complex):
//     dz_n ≈ A_n·dc + B_n·dc² + C_n·dc³ + D_n·dc⁴ + E_n·dc⁵ + …
//     A_{n+1} = 2 Z_n A_n + 1
//     B_{n+1} = 2 Z_n B_n + A_n²
//     C_{n+1} = 2 Z_n C_n + 2 A_n B_n
//     D_{n+1} = 2 Z_n D_n + (2 A_n C_n + B_n²)
//     E_{n+1} = 2 Z_n E_n + (2 A_n D_n + 2 B_n C_n)
//
// ORDER (Spawn 21): production SA is now ORDER 5 (through E). The skip N is bounded by
// the FIRST of (a) polynomial truncation > tol·|dz| or (b) the escape guard |dz|>0.25.
// Measured (tools/probe-saorder.mjs): truncation ALWAYS binds at production tol=1e-10, and
// raising order 3→5 raises the skip MOST at the moderate-deep band (+4.7% at 2^-120, the
// documented speed gap) tapering to +0.5% at 2^-400. A higher-order seed is strictly more
// accurate (more terms of the SAME exact series), so it can only INCREASE the safe skip,
// never corrupt — the bit-exact escape-count crosscheck (crosscheck-sa) still gates it.
// Order 5 is the bang-for-buck knee (order 7+ adds <2% at 2^-120 and risks df64 precision /
// GPU register pressure). `opts.order` (3|4|5) overrides; the DEFAULT is depth-adaptive —
// order 5 below 2^-112 (the GPU df64→rescaled dispatch floor, where SA is the production path
// and order-5 is df64-clean), order 3 in the shallow overlap band (see SA_ORDER5_RADIUS).
// Order 3 reproduces the historical behavior bit-for-bit (d=e=0).
//
// At DEEP zoom dc is tiny, so dz stays in the LINEAR regime (dz ≈ A_n dc, the quadratic
// term negligible) for almost the whole orbit — the quadratic/cubic only bite near escape.
// So SA can skip a huge fraction of the iterations (measured: 94–98% below 2^-120). It is
// orthogonal to Zhuoran rebasing: rebasing manages precision (glitches), SA manages
// iteration COUNT (speed). Starting every pixel at iteration N with dz = SA(dc) is exact
// because the true value z = Z_N + dz_N is the same regardless of how dz is represented;
// the per-pixel rebased iteration then continues from N normally.
//
// OVERFLOW: the raw A_n is the orbit derivative; near a deep boundary point it grows past
// 2^1024 and overflows a double. We track SCALED coefficients with u = dc/R (R = radius,
// |u| ~ O(1)), a_n = A_n·R, b_n = B_n·R², c_n = C_n·R³:
//     a_{n+1} = 2 Z_n a_n + R   (a_1 = R)
//     b_{n+1} = 2 Z_n b_n + a_n²
//     c_{n+1} = 2 Z_n c_n + 2 a_n b_n
//     dz_n ≈ a_n·u + b_n·u² + c_n·u³        (evaluated by Horner)
// The scaled coefficients stay O(|dz_n|) (≤ ~O(1)) → representable in doubles.
//
// SKIP SELECTION (probe GRID — the robust Kalles-Fraktaler-style method): iterate the
// FULL non-rebased delta dz for a grid of dc across the image box, and take the largest N
// where the SA polynomial still tracks the true dz for EVERY probe. A probe ends its
// validity at the first of: (a) the polynomial truncation grows past a relative tolerance,
// or (b) |dz| leaves the linear regime (≳ ESCAPE_GUARD) — meaning that pixel is approaching
// escape/rebase, so a no-rebase SA must not skip past it. A grid (not just corners) is
// REQUIRED: at moderate zoom the image still contains fast-escaping exterior regions that
// are NOT at the corners; a pixel that escapes before N would get a wrong (too-high) count
// if SA skipped past its escape. The grid catches those wherever they are; (b) is the hard
// guard that any probe NEAR such a region trips before the escape. A safety margin backs
// off from N (the coefficients are f64 here but the GPU runs df64 ~46-bit, which diverges a
// touch sooner). The escape-count crosscheck (tools/crosscheck-sa.mjs) is the final gate:
// it proves the chosen N yields IDENTICAL escape counts to a no-skip render.

// |dz| beyond this is out of the linear regime (escape/rebase imminent) — SA stops there.
const ESCAPE_GUARD = 0.25;

// Order-5 SA seeds dz at a LATER iteration than order-3 (larger skip), where the scaled
// coefficients are larger, so the GPU's df64 (~46-bit) in-shader Horner has a larger ABSOLUTE
// seed error. On a shallow CHAOTIC region (esc counts ~17k at 2^-90) that error amplifies
// through the post-seed continuation — measured: 2^-90 seahorse meanΔsn 0.9→1.4, over the
// validate-gpu gate. BUT below 2^-112 (GPU_PERTURB_FLOOR — the GPU df64→rescaled/fe dispatch
// boundary, i.e. where SA actually runs in production) order-5 is df64-clean even on the
// genuine chaotic deep boundary coordinate (tools/probe-saorder-gpu.mjs: 0% mism, meanΔsn
// ~5e-4 at 2^-100..-130, 1000× under the gate). So order-5 is the DEFAULT only below that
// floor (where it both pays off — the moderate-deep skip gap — and is safe); the shallow
// overlap band (where production renders df64-no-SA anyway, and SA's gain is small) keeps
// order 3. opts.order overrides (the order A/B benches force a fixed order at any depth).
const SA_ORDER5_RADIUS = Math.pow(2, -112);

// DEPTH-ADAPTIVE SAFETY MARGIN (Spawn 22). The skip is backed off marginFrac·validN as a guard,
// motivated entirely by the GPU df64 (~46-bit) seed diverging "a touch sooner" than the f64 grid
// validation (validN). The shipping 0.05 was over-conservative DEEP: below 2^-112 — the band where
// the GPU runs SA (rescaled/fe engine) — the df64 seed is measured ~1000× under the gate, so the
// margin can shrink with ZERO precision cost. MEASURED on the real RTX 3090 (tools/probe-samargin
// + bench-samargin): margin 0.05→0.02 (and down to 0) stays BIT-EXACT (mism 0, IDENTICAL meanΔsn/
// maxΔsn) on the deep boundary coordinate while cutting post-SA iterations ~20–31% → **1.12× /
// 1.19× / 1.18× / 1.40× wall-clock at 2^-120/-218/-271/-400** (0.01 gives ~1.18–1.52×, kept as the
// further measured-safe option — opts.marginFrac overrides). The 2% buffer keeps a comfortable
// guard for untested coordinates; the isolated-early-escaper case is covered SEPARATELY by the
// worker's skipCap (coarse-pass min-escape), NOT this margin, and the truncation worst case (the
// dc-box corner) is sampled EXACTLY by the probe grid. Shallow/moderate band keeps 0.05 (CPU-SA
// territory, not swept here; SA's gain there is small). Gated bit-exact by crosscheck-sa +
// validate:gpu. Below 2^-112 = exactly SA_ORDER5_RADIUS (the df64→rescaled GPU-SA dispatch floor).
const SA_DEEP_MARGIN = 0.02;

// Compute the SA coefficients and choose a safe skip N for a view.
//   ref            : reference orbit { zx, zy, len } (from computeReference)
//   radius         : view radius (half-height) — the coefficient scale R
//   dcHalfX,dcHalfY: half-extents of the dc box (max |dc.x|, |dc.y| over the pixel grid)
//   opts.tol       : relative SA-vs-full agreement tolerance (default 1e-3)
//   opts.marginFrac: back off this fraction of the valid skip for safety (default: depth-adaptive,
//                    SA_DEEP_MARGIN below 2^-112 else 0.05 — see SA_DEEP_MARGIN).
//   opts.minSkip   : don't bother below this many iterations (default 32)
//   opts.maxIter   : cap (default ref.len)
//   opts.order     : SA polynomial order 3|4|5 (default: depth-adaptive, see SA_ORDER5_RADIUS).
// Returns { skip, invR, ax..ey } (coeffs AT index `skip`; dx,dy,ex,ey are 0 below `order`),
// or { skip: 0 } if SA isn't worthwhile. Pass the result as the `sa` arg to escapePerturb.
export function computeSeries(ref, radius, dcHalfX, dcHalfY, opts = {}) {
  // tol governs SHALLOW-band safety: there the 3-term series has real truncation error, so
  // a loose tol over-skips and the error amplifies through the chaotic continuation (1e-3 →
  // counts off by tens at 2^-50). 1e-10 keeps the chosen skip bit-exact at every depth
  // (crosscheck-sa: 0 mism 2^-50…2^-500). Deep zoom is essentially tol-independent — the
  // linear term dominates so the series is exact until it breaks abruptly near escape.
  const tol = opts.tol ?? 1e-10;
  const marginFrac = opts.marginFrac ?? (radius < SA_ORDER5_RADIUS ? SA_DEEP_MARGIN : 0.05);
  const minSkip = opts.minSkip ?? 32;
  const gridN = opts.gridN ?? 13;       // gridN×gridN probes across the dc box
  const order = opts.order ?? (radius < SA_ORDER5_RADIUS ? 5 : 3);   // depth-adaptive (3|4|5)
  const eg = opts.escapeGuard ?? ESCAPE_GUARD;
  const guard2 = eg * eg;
  const tol2 = tol * tol;
  // Hard upper bound on the skip — independent of the probe grid. The caller (worker)
  // passes the minimum escape iteration found in its dense no-SA coarse pass, so the skip
  // can never pass an isolated early-escaping pixel that falls between the grid probes. A
  // no-op (Infinity) for callers without a coarse pass (the grid escape-guard still applies).
  const skipCap = opts.skipCap ?? Infinity;
  const { zx, zy, len } = ref;
  const cap = Math.min(opts.maxIter ?? len, len);
  const R = radius;
  if (cap < minSkip || R <= 0 || !(dcHalfX > 0 || dcHalfY > 0)) return { skip: 0 };

  // Probe grid over the box: u_x ∈ [-dcHalfX/R, …], u_y ∈ [-dcHalfY/R, …]. Includes the
  // corners (i,j at the ends) — the SA-truncation worst case — and the interior, where a
  // fast-escaping region may sit at moderate zoom.
  const invR = 1 / R;
  const P = gridN * gridN;
  const dcx = new Float64Array(P), dcy = new Float64Array(P);
  const ux = new Float64Array(P), uy = new Float64Array(P);
  const px = new Float64Array(P), py = new Float64Array(P); // full non-rebased dz per probe
  for (let j = 0; j < gridN; j++) {
    const fy = gridN === 1 ? 0 : (j / (gridN - 1)) * 2 - 1;
    for (let i = 0; i < gridN; i++) {
      const fx = gridN === 1 ? 0 : (i / (gridN - 1)) * 2 - 1;
      const p = j * gridN + i;
      dcx[p] = fx * dcHalfX; dcy[p] = fy * dcHalfY;
      ux[p] = dcx[p] * invR; uy[p] = dcy[p] * invR;
    }
  }

  // Pass 1 — advance the scaled coefficients and the probe deltas together. The skip is the
  // largest N at which EVERY probe is still in its valid (linear, low-truncation) regime;
  // we stop at the FIRST iteration where any probe fails (its min failure point).
  let ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0, dx = 0, dy = 0, ex = 0, ey = 0;
  let validN = 0;
  for (let n = 1; n <= cap; n++) {
    const Zx = zx[n - 1], Zy = zy[n - 1];
    const a2x = ax * ax - ay * ay, a2y = 2 * ax * ay;          // a²   (old a)
    const abx = ax * bx - ay * by, aby = ax * by + ay * bx;     // a·b  (old a,b)
    const nax = 2 * (Zx * ax - Zy * ay) + R, nay = 2 * (Zx * ay + Zy * ax);
    const nbx = 2 * (Zx * bx - Zy * by) + a2x, nby = 2 * (Zx * by + Zy * bx) + a2y;
    const ncx = 2 * (Zx * cx - Zy * cy) + 2 * abx, ncy = 2 * (Zx * cy + Zy * cx) + 2 * aby;
    if (order >= 4) {
      // d' = 2Z d + (2 a c + b²) ; e' = 2Z e + (2 a d + 2 b c)   (all old coeffs)
      const acx = ax * cx - ay * cy, acy = ax * cy + ay * cx;
      const b2x = bx * bx - by * by, b2y = 2 * bx * by;
      const ndx = 2 * (Zx * dx - Zy * dy) + (2 * acx + b2x), ndy = 2 * (Zx * dy + Zy * dx) + (2 * acy + b2y);
      if (order >= 5) {
        const adx = ax * dx - ay * dy, ady = ax * dy + ay * dx;
        const bcx = bx * cx - by * cy, bcy = bx * cy + by * cx;
        const nex = 2 * (Zx * ex - Zy * ey) + (2 * adx + 2 * bcx), ney = 2 * (Zx * ey + Zy * ex) + (2 * ady + 2 * bcy);
        ex = nex; ey = ney;
      }
      dx = ndx; dy = ndy;
    }
    ax = nax; ay = nay; bx = nbx; by = nby; cx = ncx; cy = ncy;

    let ok = true;
    for (let p = 0; p < P; p++) {
      const dxp = px[p], dyp = py[p];
      const ndx = 2 * (Zx * dxp - Zy * dyp) + (dxp * dxp - dyp * dyp) + dcx[p];
      const ndy = 2 * (Zx * dyp + Zy * dxp) + (2 * dxp * dyp) + dcy[p];
      px[p] = ndx; py[p] = ndy;
      // (b) escape guard: |dz| left the linear regime → escape/rebase imminent, stop here.
      const mag2 = ndx * ndx + ndy * ndy;
      if (mag2 > guard2) { ok = false; break; }
      // (a) SA estimate via Horner: (((e·u + d)·u + c)·u + b)·u + a)·u, vs the true dz.
      const Ux = ux[p], Uy = uy[p];
      let hx, hy, tx, ty;
      if (order >= 5) { hx = ex; hy = ey; } else if (order >= 4) { hx = dx; hy = dy; } else { hx = cx; hy = cy; }
      if (order >= 5) { tx = hx * Ux - hy * Uy + dx; ty = hx * Uy + hy * Ux + dy; hx = tx; hy = ty; }
      if (order >= 4) { tx = hx * Ux - hy * Uy + cx; ty = hx * Uy + hy * Ux + cy; hx = tx; hy = ty; }
      tx = hx * Ux - hy * Uy + bx; ty = hx * Uy + hy * Ux + by; hx = tx; hy = ty;
      tx = hx * Ux - hy * Uy + ax; ty = hx * Uy + hy * Ux + ay; hx = tx; hy = ty;
      const sx = hx * Ux - hy * Uy, sy = hx * Uy + hy * Ux;
      const erx = sx - ndx, ery = sy - ndy;
      const err2 = erx * erx + ery * ery;
      // relative truncation: |SA − dz| > tol·|dz|. (At dz=0 — the centre probe — both sides
      // are 0, so 0 > 0 is false and the probe stays valid, as it should.)
      if (err2 > tol2 * mag2) { ok = false; break; }
    }
    if (!ok) break;
    validN = n;
  }

  // Back off a margin from the tighter of (grid-validated N, caller's escape cap).
  const bound = Math.min(validN, skipCap);
  let skip = Math.max(0, Math.floor(bound * (1 - marginFrac)));
  if (skip < minSkip) return { skip: 0 };

  // Pass 2 — recompute the coefficients up to the chosen skip and return their values.
  ax = 0; ay = 0; bx = 0; by = 0; cx = 0; cy = 0; dx = 0; dy = 0; ex = 0; ey = 0;
  for (let n = 1; n <= skip; n++) {
    const Zx = zx[n - 1], Zy = zy[n - 1];
    const a2x = ax * ax - ay * ay, a2y = 2 * ax * ay;
    const abx = ax * bx - ay * by, aby = ax * by + ay * bx;
    const nax = 2 * (Zx * ax - Zy * ay) + R, nay = 2 * (Zx * ay + Zy * ax);
    const nbx = 2 * (Zx * bx - Zy * by) + a2x, nby = 2 * (Zx * by + Zy * bx) + a2y;
    const ncx = 2 * (Zx * cx - Zy * cy) + 2 * abx, ncy = 2 * (Zx * cy + Zy * cx) + 2 * aby;
    if (order >= 4) {
      const acx = ax * cx - ay * cy, acy = ax * cy + ay * cx;
      const b2x = bx * bx - by * by, b2y = 2 * bx * by;
      const ndx = 2 * (Zx * dx - Zy * dy) + (2 * acx + b2x), ndy = 2 * (Zx * dy + Zy * dx) + (2 * acy + b2y);
      if (order >= 5) {
        const adx = ax * dx - ay * dy, ady = ax * dy + ay * dx;
        const bcx = bx * cx - by * cy, bcy = bx * cy + by * cx;
        const nex = 2 * (Zx * ex - Zy * ey) + (2 * adx + 2 * bcx), ney = 2 * (Zx * ey + Zy * ex) + (2 * ady + 2 * bcy);
        ex = nex; ey = ney;
      }
      dx = ndx; dy = ndy;
    }
    ax = nax; ay = nay; bx = nbx; by = nby; cx = ncx; cy = ncy;
  }
  return { skip, invR, ax, ay, bx, by, cx, cy, dx, dy, ex, ey };
}

// Evaluate the SA polynomial dz_skip = a·u + b·u² + c·u³ + d·u⁴ + e·u⁵ for a pixel's dc
// (u = dc/R), via Horner. Coeffs d,e are 0 for an order-3 sa (so this reduces to the cubic).
// Returns [dzx, dzy]. (escapePerturb inlines this on its hot path; this helper is for tests/tools.)
export function saStartDelta(sa, dcx, dcy) {
  const Ux = dcx * sa.invR, Uy = dcy * sa.invR;
  let hx = sa.ex ?? 0, hy = sa.ey ?? 0, tx, ty;
  tx = hx * Ux - hy * Uy + (sa.dx ?? 0); ty = hx * Uy + hy * Ux + (sa.dy ?? 0); hx = tx; hy = ty;
  tx = hx * Ux - hy * Uy + sa.cx; ty = hx * Uy + hy * Ux + sa.cy; hx = tx; hy = ty;
  tx = hx * Ux - hy * Uy + sa.bx; ty = hx * Uy + hy * Ux + sa.by; hx = tx; hy = ty;
  tx = hx * Ux - hy * Uy + sa.ax; ty = hx * Uy + hy * Ux + sa.ay; hx = tx; hy = ty;
  return [hx * Ux - hy * Uy, hx * Uy + hy * Ux];
}
