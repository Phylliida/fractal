// probe-sa.mjs — MEASURE the achievable series-approximation (SA) skip at depth.
//
// Series approximation skips the first N perturbation iterations for EVERY pixel by
// approximating the delta orbit as a polynomial in dc:
//     dz_n ≈ A_n·dc + B_n·dc² + C_n·dc³ + …
// where the complex coefficients obey (with Z_0=0, dz_0=0):
//     A_{n+1} = 2 Z_n A_n + 1
//     B_{n+1} = 2 Z_n B_n + A_n²
//     C_{n+1} = 2 Z_n C_n + 2 A_n B_n
//
// PROBLEM: raw A_n is the orbit derivative; near a deep boundary point it grows past
// 2^1024 and overflows a double. FIX (scaled coefficients): with u = dc/R (R = image
// radius, so |u| ≤ aspect, O(1)), track a_n=A_n·R, b_n=B_n·R², c_n=C_n·R³:
//     a_{n+1} = 2 Z_n a_n + R       (a_1 = R)
//     b_{n+1} = 2 Z_n b_n + a_n²
//     c_{n+1} = 2 Z_n c_n + 2 a_n b_n
// then dz_n ≈ a_n·u + b_n·u² + c_n·u³. These stay O(|dz_n|) (≤ ~O(1)) → safe in doubles.
//
// SKIP SELECTION (probe points, the robust Kalles-Fraktaler-style method): pick u at the
// image-box corners/edges (largest |u|), iterate the FULL non-rebased delta dz for each,
// and find the largest N where the SA polynomial still matches the full dz for ALL probes
// within a relative tolerance. Starting every pixel's real (rebased) iteration at N with
// dz = SA(u) is then accurate for any pixel inside the probe box.
//
// This tool reports N / maxIter across depths — the make-or-break number for whether SA
// is worth implementing. SA only helps if it skips a worthwhile fraction of the iterations.
//
//   node tools/probe-sa.mjs
//   BITS=50,120,271,400,500 TOL=1e-3 node tools/probe-sa.mjs
import { fromDecimalString, precForRadius } from '../src/math/bignum.js';
import { computeReference } from '../src/math/reference.js';

// The genuine deep boundary coordinate (good to ~2^-520) from tools/gen-deep-coord.mjs,
// the same one tools/probe-deep500.mjs validates the GPU engines against.
const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

const BITS = (process.env.BITS || '50,120,218,271,340,400,500').split(',').map(Number);
const TOL = Number(process.env.TOL || 1e-3);    // relative agreement tolerance for SA-vs-full
const ASPECT = Number(process.env.ASPECT || 1.5);
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

// Probe u points: the box is u_x ∈ [-aspect, aspect], u_y ∈ [-1, 1]. The worst-case
// (largest |u|, where SA breaks down first) is the corners; include edge mids + a couple
// of off-axis points for robustness against a coefficient happening to vanish on an axis.
function probeUs(aspect) {
  const us = [];
  for (const sx of [-1, 0, 1]) for (const sy of [-1, 0, 1]) {
    if (sx === 0 && sy === 0) continue;
    us.push([sx * aspect, sy]);
  }
  us.push([0.7 * aspect, 0.7], [-0.7 * aspect, 0.6], [0.55 * aspect, -0.8]);
  return us;
}

// Full NON-rebased delta iteration for one probe dc, returning dz_n at each n (we only
// keep the running value and compare against SA on the fly).
function measureSkip(ref, R, aspect, maxIter, tol) {
  const { zx, zy, len } = ref;
  const us = probeUs(aspect);
  const P = us.length;
  // Per-probe full dz (non-rebased), relative to Z_n.
  const dx = new Float64Array(P), dy = new Float64Array(P);
  const dcx = us.map(([ux]) => ux * R), dcy = us.map(([, uy]) => uy * R);
  // Scaled SA coefficients (complex), index advances with n.
  let ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0;
  let skip = 0;          // largest n proven valid for all probes
  let broke = false;
  const cap = Math.min(maxIter, len);

  for (let n = 1; n <= cap; n++) {
    const Zx = zx[n - 1], Zy = zy[n - 1];
    // Advance SA coefficients: coeff_{n} from coeff_{n-1} using Z_{n-1}.
    //   a' = 2 Z a + R ; b' = 2 Z b + a² ; c' = 2 Z c + 2 a b   (complex)
    const a2x = ax * ax - ay * ay, a2y = 2 * ax * ay;            // a²  (use OLD a)
    const abx = ax * bx - ay * by, aby = ax * by + ay * bx;      // a·b (use OLD a,b)
    const nax = 2 * (Zx * ax - Zy * ay) + R;
    const nay = 2 * (Zx * ay + Zy * ax);
    const nbx = 2 * (Zx * bx - Zy * by) + a2x;
    const nby = 2 * (Zx * by + Zy * bx) + a2y;
    const ncx = 2 * (Zx * cx - Zy * cy) + 2 * abx;
    const ncy = 2 * (Zx * cy + Zy * cx) + 2 * aby;
    ax = nax; ay = nay; bx = nbx; by = nby; cx = ncx; cy = ncy;

    // Advance each probe's full (non-rebased) delta to index n, compare to SA.
    let allGood = true;
    for (let p = 0; p < P; p++) {
      const px = dx[p], py = dy[p];
      const ndx = 2 * (Zx * px - Zy * py) + (px * px - py * py) + dcx[p];
      const ndy = 2 * (Zx * py + Zy * px) + (2 * px * py) + dcy[p];
      dx[p] = ndx; dy[p] = ndy;
      // SA estimate dz_n = a·u + b·u² + c·u³  (u = (ux,uy) complex)
      const [ux, uy] = us[p];
      const u2x = ux * ux - uy * uy, u2y = 2 * ux * uy;
      const u3x = u2x * ux - u2y * uy, u3y = u2x * uy + u2y * ux;
      const sx = ax * ux - ay * uy + bx * u2x - by * u2y + cx * u3x - cy * u3y;
      const sy = ax * uy + ay * ux + bx * u2y + by * u2x + cx * u3y + cy * u3x;
      const ex = sx - ndx, ey = sy - ndy;
      const err = Math.hypot(ex, ey), mag = Math.hypot(ndx, ndy);
      if (err > tol * mag && err > 1e-300) { allGood = false; }
    }
    if (allGood && !broke) skip = n;
    else { broke = true; break; }   // once it diverges it stays diverged — stop early
  }
  return { skip, cap };
}

console.log(`SA skip measurement on the deep boundary coordinate (aspect ${ASPECT}, relTol ${TOL})`);
console.log('coords:', RE.slice(0, 24) + '…', IM.slice(0, 24) + '…\n');
console.log('depth     maxIter   refLen  skipN    skip%   refMs');

for (const bits of BITS) {
  const radius = 1.5 * 2 ** -bits;
  const maxIter = autoIter(bits);
  const prec = precForRadius(radius, 80);
  const cx = fromDecimalString(RE, prec), cy = fromDecimalString(IM, prec);
  const t0 = Date.now();
  const ref = computeReference({ x: cx, y: cy, prec }, maxIter);
  const refMs = Date.now() - t0;
  const { skip } = measureSkip(ref, radius, ASPECT, maxIter, TOL);
  const pct = (100 * skip / maxIter).toFixed(1);
  console.log(
    `2^-${String(bits).padEnd(4)} ${String(maxIter).padStart(7)} ${String(ref.len).padStart(7)} ` +
    `${String(skip).padStart(6)}  ${pct.padStart(6)}%  ${String(refMs).padStart(5)}`);
}
console.log('\nskip% = fraction of iterations SA skips for every pixel. >~15% ⇒ worth a GPU port;');
console.log('tiny ⇒ rebasing/derivative growth kills the early-skip and SA is not worth it here.');
