// probe-saorder.mjs — does a HIGHER-ORDER series approximation close the moderate-depth
// skip gap? (Spawn 21 measure-first probe.)
//
// Production SA (src/math/series.js) is ORDER 3: dz ≈ a·u + b·u² + c·u³. The skip it can
// claim is bounded by the FIRST of two constraints at the binding probe:
//   (a) TRUNCATION: |SA − dz_full| > tol·|dz_full|   (production tol = 1e-10)
//   (b) ESCAPE GUARD: |dz_full| > 0.25                (linear regime ends / rebase imminent)
// If (a) binds at moderate depth, a higher-order series (order 5) extends the skip — directly
// closing the documented 82%→92% skip gap (2^-120 vs 2^-400). If (b) binds, order is moot:
// no polynomial in dc can skip past the pixel's first rebase/escape.
//
// This probe runs the EXACT production skip-selection loop at order 3 AND order 5, with the
// production tol + escape guard, and reports the chosen validN, the skip%, AND which
// constraint tripped at the break for each order. Decisive, cheap, before any implementation.
//
//   node tools/probe-saorder.mjs
//   BITS=50,120,271,400 TOL=1e-10 node tools/probe-saorder.mjs
import { fromDecimalString, precForRadius } from '../src/math/bignum.js';
import { computeReference } from '../src/math/reference.js';

const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

const BITS = (process.env.BITS || '50,120,218,271,400').split(',').map(Number);
const TOL = Number(process.env.TOL || 1e-10);
const MARGIN = Number(process.env.MARGIN || 0.05);
const GUARD = Number(process.env.GUARD || 0.25);
const ASPECT = Number(process.env.ASPECT || 1.5);
const GRIDN = Number(process.env.GRIDN || 13);
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

// Run the production skip loop at a given polynomial ORDER (3 or 5). Returns the largest
// validN and, at the FIRST failing iteration, which constraint tripped + at which probe.
function measure(ref, R, order, maxIter) {
  const { zx, zy, len } = ref;
  const cap = Math.min(maxIter, len);
  const invR = 1 / R;
  const guard2 = GUARD * GUARD, tol2 = TOL * TOL;
  // dc box: u_x ∈ [-aspect, aspect], u_y ∈ [-1, 1].
  const dcHalfX = ASPECT * R, dcHalfY = R;
  const P = GRIDN * GRIDN;
  const dcx = new Float64Array(P), dcy = new Float64Array(P);
  const ux = new Float64Array(P), uy = new Float64Array(P);
  const px = new Float64Array(P), py = new Float64Array(P);
  for (let j = 0; j < GRIDN; j++) {
    const fy = GRIDN === 1 ? 0 : (j / (GRIDN - 1)) * 2 - 1;
    for (let i = 0; i < GRIDN; i++) {
      const fx = GRIDN === 1 ? 0 : (i / (GRIDN - 1)) * 2 - 1;
      const p = j * GRIDN + i;
      dcx[p] = fx * dcHalfX; dcy[p] = fy * dcHalfY;
      ux[p] = dcx[p] * invR; uy[p] = dcy[p] * invR;
    }
  }
  // Scaled coeffs up to order 5 (only the first `order` are used in the SA estimate).
  let ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0, dx = 0, dy = 0, ex = 0, ey = 0;
  let validN = 0, reason = 'none', failP = -1;
  for (let n = 1; n <= cap; n++) {
    const Zx = zx[n - 1], Zy = zy[n - 1];
    const a2x = ax * ax - ay * ay, a2y = 2 * ax * ay;          // a²
    const abx = ax * bx - ay * by, aby = ax * by + ay * bx;     // a·b
    const acx = ax * cx - ay * cy, acy = ax * cy + ay * cx;     // a·c
    const bcx = bx * cx - by * cy, bcy = bx * cy + by * cx;     // b·c
    const adx = ax * dx - ay * dy, ady = ax * dy + ay * dx;     // a·d
    const b2x = bx * bx - by * by, b2y = 2 * bx * by;           // b²
    const nax = 2 * (Zx * ax - Zy * ay) + R,  nay = 2 * (Zx * ay + Zy * ax);
    const nbx = 2 * (Zx * bx - Zy * by) + a2x, nby = 2 * (Zx * by + Zy * bx) + a2y;
    const ncx = 2 * (Zx * cx - Zy * cy) + 2 * abx, ncy = 2 * (Zx * cy + Zy * cx) + 2 * aby;
    const ndx = 2 * (Zx * dx - Zy * dy) + (2 * acx + b2x), ndy = 2 * (Zx * dy + Zy * dx) + (2 * acy + b2y);
    const nex = 2 * (Zx * ex - Zy * ey) + (2 * adx + 2 * bcx), ney = 2 * (Zx * ey + Zy * ex) + (2 * ady + 2 * bcy);
    ax = nax; ay = nay; bx = nbx; by = nby; cx = ncx; cy = ncy; dx = ndx; dy = ndy; ex = nex; ey = ney;

    let ok = true, why = '', wp = -1;
    for (let p = 0; p < P; p++) {
      const dxp = px[p], dyp = py[p];
      const tdx = 2 * (Zx * dxp - Zy * dyp) + (dxp * dxp - dyp * dyp) + dcx[p];
      const tdy = 2 * (Zx * dyp + Zy * dxp) + (2 * dxp * dyp) + dcy[p];
      px[p] = tdx; py[p] = tdy;
      const mag2 = tdx * tdx + tdy * tdy;
      if (mag2 > guard2) { ok = false; why = 'guard'; wp = p; break; }
      // SA estimate via Horner up to `order`.
      const Ux = ux[p], Uy = uy[p];
      let hx, hy, tx, ty;
      if (order >= 5) { hx = ex; hy = ey; } else { hx = cx; hy = cy; }
      if (order >= 5) { tx = hx * Ux - hy * Uy + dx; ty = hx * Uy + hy * Ux + dy; hx = tx; hy = ty; }
      if (order >= 4) { tx = hx * Ux - hy * Uy + cx; ty = hx * Uy + hy * Ux + cy; hx = tx; hy = ty; }
      tx = hx * Ux - hy * Uy + bx; ty = hx * Uy + hy * Ux + by; hx = tx; hy = ty;
      tx = hx * Ux - hy * Uy + ax; ty = hx * Uy + hy * Ux + ay; hx = tx; hy = ty;
      const sx = hx * Ux - hy * Uy, sy = hx * Uy + hy * Ux;
      const erx = sx - tdx, ery = sy - tdy;
      const err2 = erx * erx + ery * ery;
      if (err2 > tol2 * mag2) { ok = false; why = 'trunc'; wp = p; break; }
    }
    if (!ok) { reason = why; failP = wp; break; }
    validN = n;
  }
  const skip = Math.max(0, Math.floor(validN * (1 - MARGIN)));
  return { validN, skip, reason, failP };
}

console.log(`SA order-3 vs order-5 skip (tol ${TOL}, guard ${GUARD}, margin ${MARGIN}, grid ${GRIDN}², aspect ${ASPECT})`);
console.log('coords:', RE.slice(0, 22) + '…\n');
console.log('depth    maxIter  refLen | o3 validN  skip%   bind  | o5 validN  skip%   bind  |  Δskip%');
for (const bits of BITS) {
  const radius = 1.5 * 2 ** -bits;
  const maxIter = autoIter(bits);
  const prec = precForRadius(radius, 80);
  const cx = fromDecimalString(RE, prec), cy = fromDecimalString(IM, prec);
  const ref = computeReference({ x: cx, y: cy, prec }, maxIter);
  const o3 = measure(ref, radius, 3, maxIter);
  const o5 = measure(ref, radius, 5, maxIter);
  const p3 = (100 * o3.skip / maxIter), p5 = (100 * o5.skip / maxIter);
  console.log(
    `2^-${String(bits).padEnd(4)}${String(maxIter).padStart(8)}${String(ref.len).padStart(8)} | ` +
    `${String(o3.validN).padStart(8)} ${p3.toFixed(1).padStart(6)}%  ${o3.reason.padEnd(5)} | ` +
    `${String(o5.validN).padStart(8)} ${p5.toFixed(1).padStart(6)}%  ${o5.reason.padEnd(5)} | ` +
    `${(p5 - p3 >= 0 ? '+' : '') + (p5 - p3).toFixed(1)}`);
}
console.log('\nbind = which constraint stopped the skip: "trunc" (truncation, higher order helps)');
console.log('or "guard" (|dz|>0.25, escape/rebase imminent — order is MOOT past this point).');
