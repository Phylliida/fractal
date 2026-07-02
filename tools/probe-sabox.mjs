// probe-sabox.mjs — does a SMALLER (centered) dc sub-box yield a meaningfully larger SA skip?
// (measure-first probe for a region-adaptive SA skip.)
//
// The production SA skip is GLOBAL: one skip N for the whole frame, bounded by the WORST
// probe — the corner pixel (max |dc|). Central pixels have tiny |dc|, so they rebase/escape
// far later and could be SA-skipped much further. If a centered sub-box (scale s of the full
// half-extents) skips substantially more, a region-adaptive scheme (concentric boxes / 2D
// tiles, each its own skip+coeffs) would close part of the moderate-deep speed gap.
//
// This runs the EXACT production order-5 skip loop for several box scales and reports the
// chosen skip%, the absolute skip, and which constraint bound it (trunc vs guard).
//
//   node tools/probe-sabox.mjs
//   BITS=120,218,271,400 SCALES=1,0.7,0.5,0.25 node tools/probe-sabox.mjs
import { fromDecimalString, precForRadius } from '../src/math/bignum.js';
import { computeReference } from '../src/math/reference.js';

const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

const BITS = (process.env.BITS || '120,218,271,400').split(',').map(Number);
const SCALES = (process.env.SCALES || '1,0.7,0.5,0.25').split(',').map(Number);
const TOL = Number(process.env.TOL || 1e-10);
const MARGIN = Number(process.env.MARGIN || 0.05);
const GUARD = Number(process.env.GUARD || 0.25);
const ASPECT = Number(process.env.ASPECT || 1.5);
const GRIDN = Number(process.env.GRIDN || 13);
const ORDER = Number(process.env.ORDER || 5);
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

// Production order-5 skip loop over a dc box of half-extents (hx,hy). Returns validN + bind.
function measure(ref, R, hx, hy, order, maxIter) {
  const { zx, zy, len } = ref;
  const cap = Math.min(maxIter, len);
  const invR = 1 / R;
  const guard2 = GUARD * GUARD, tol2 = TOL * TOL;
  const P = GRIDN * GRIDN;
  const dcx = new Float64Array(P), dcy = new Float64Array(P);
  const ux = new Float64Array(P), uy = new Float64Array(P);
  const px = new Float64Array(P), py = new Float64Array(P);
  for (let j = 0; j < GRIDN; j++) {
    const fy = GRIDN === 1 ? 0 : (j / (GRIDN - 1)) * 2 - 1;
    for (let i = 0; i < GRIDN; i++) {
      const fx = GRIDN === 1 ? 0 : (i / (GRIDN - 1)) * 2 - 1;
      const p = j * GRIDN + i;
      dcx[p] = fx * hx; dcy[p] = fy * hy;
      ux[p] = dcx[p] * invR; uy[p] = dcy[p] * invR;
    }
  }
  let ax = 0, ay = 0, bx = 0, by = 0, cx = 0, cy = 0, dx = 0, dy = 0, ex = 0, ey = 0;
  let validN = 0, reason = 'none';
  for (let n = 1; n <= cap; n++) {
    const Zx = zx[n - 1], Zy = zy[n - 1];
    const a2x = ax * ax - ay * ay, a2y = 2 * ax * ay;
    const abx = ax * bx - ay * by, aby = ax * by + ay * bx;
    const acx = ax * cx - ay * cy, acy = ax * cy + ay * cx;
    const bcx = bx * cx - by * cy, bcy = bx * cy + by * cx;
    const adx = ax * dx - ay * dy, ady = ax * dy + ay * dx;
    const b2x = bx * bx - by * by, b2y = 2 * bx * by;
    const nax = 2 * (Zx * ax - Zy * ay) + R,  nay = 2 * (Zx * ay + Zy * ax);
    const nbx = 2 * (Zx * bx - Zy * by) + a2x, nby = 2 * (Zx * by + Zy * bx) + a2y;
    const ncx = 2 * (Zx * cx - Zy * cy) + 2 * abx, ncy = 2 * (Zx * cy + Zy * cx) + 2 * aby;
    const ndx = 2 * (Zx * dx - Zy * dy) + (2 * acx + b2x), ndy = 2 * (Zx * dy + Zy * dx) + (2 * acy + b2y);
    const nex = 2 * (Zx * ex - Zy * ey) + (2 * adx + 2 * bcx), ney = 2 * (Zx * ey + Zy * ex) + (2 * ady + 2 * bcy);
    ax = nax; ay = nay; bx = nbx; by = nby; cx = ncx; cy = ncy; dx = ndx; dy = ndy; ex = nex; ey = ney;

    let ok = true, why = '';
    for (let p = 0; p < P; p++) {
      const dxp = px[p], dyp = py[p];
      const tdx = 2 * (Zx * dxp - Zy * dyp) + (dxp * dxp - dyp * dyp) + dcx[p];
      const tdy = 2 * (Zx * dyp + Zy * dxp) + (2 * dxp * dyp) + dcy[p];
      px[p] = tdx; py[p] = tdy;
      const mag2 = tdx * tdx + tdy * tdy;
      if (mag2 > guard2) { ok = false; why = 'guard'; break; }
      const Ux = ux[p], Uy = uy[p];
      let hxx, hyy, tx, ty;
      if (order >= 5) { hxx = ex; hyy = ey; } else { hxx = cx; hyy = cy; }
      if (order >= 5) { tx = hxx * Ux - hyy * Uy + dx; ty = hxx * Uy + hyy * Ux + dy; hxx = tx; hyy = ty; }
      if (order >= 4) { tx = hxx * Ux - hyy * Uy + cx; ty = hxx * Uy + hyy * Ux + cy; hxx = tx; hyy = ty; }
      tx = hxx * Ux - hyy * Uy + bx; ty = hxx * Uy + hyy * Ux + by; hxx = tx; hyy = ty;
      tx = hxx * Ux - hyy * Uy + ax; ty = hxx * Uy + hyy * Ux + ay; hxx = tx; hyy = ty;
      const sx = hxx * Ux - hyy * Uy, sy = hxx * Uy + hyy * Ux;
      const erx = sx - tdx, ery = sy - tdy;
      if (erx * erx + ery * ery > tol2 * mag2) { ok = false; why = 'trunc'; break; }
    }
    if (!ok) { reason = why; break; }
    validN = n;
  }
  return { validN, skip: Math.max(0, Math.floor(validN * (1 - MARGIN))), reason };
}

console.log(`SA skip vs dc-box scale (order ${ORDER}, tol ${TOL}, guard ${GUARD}, grid ${GRIDN}², aspect ${ASPECT})`);
console.log('A centered sub-box of scale s uses half-extents (s·aspect·R, s·R). skip% = skip/maxIter.\n');
for (const bits of BITS) {
  const radius = 1.5 * 2 ** -bits;
  const maxIter = autoIter(bits);
  const prec = precForRadius(radius, 80);
  const cx = fromDecimalString(RE, prec), cy = fromDecimalString(IM, prec);
  const ref = computeReference({ x: cx, y: cy, prec }, maxIter);
  const cells = SCALES.map((s) => {
    const r = measure(ref, radius, s * ASPECT * radius, s * radius, ORDER, maxIter);
    return { s, ...r, pct: 100 * r.skip / maxIter };
  });
  const base = cells[0];
  console.log(`2^-${bits}  maxIter ${maxIter}  refLen ${ref.len}`);
  for (const c of cells) {
    const dPct = c.pct - base.pct;
    console.log(
      `   scale ${c.s.toFixed(2)}:  validN ${String(c.validN).padStart(8)}  skip ${String(c.skip).padStart(8)}` +
      `  skip% ${c.pct.toFixed(1).padStart(5)}  bind ${c.reason.padEnd(5)}` +
      `  (Δ vs full ${(dPct >= 0 ? '+' : '') + dPct.toFixed(1)}%)`);
  }
  console.log('');
}
