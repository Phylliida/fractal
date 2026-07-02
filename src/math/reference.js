// reference.js — high-precision reference orbit for perturbation.
//
// Given a center C = (cx, cy) as fixed-point BigInts at precision `prec`, iterate
// Z_{n+1} = Z_n^2 + C in high precision and export the orbit as Float64Arrays
// (Z_n are O(1) magnitude, so doubles capture them with full relative precision).
//
// Output feeds the double-precision perturbation engine (perturb.js).

import { mulShift, toDouble } from './bignum.js';

// Exact escape count of a single point computed entirely in BigInt fixed point.
// This is the DEEP-ZOOM ground-truth oracle (slow, but exact at any depth).
// Uses the same bailout as naive.js / perturb.js so counts are comparable.
//   cx, cy : fixed-point BigInts at precision `prec`
// Returns the integer escape iteration (maxIter => did not escape).
export function escapeBigInt(cx, cy, prec, maxIter, bailoutSq = 1 << 16) {
  const bail = BigInt(bailoutSq) << BigInt(prec);
  // zx2/zy2 = zx²/zy² carried across iterations (Spawn 29): the escape test's
  // squares of the NEW z ARE the next update's squares — computing them once cuts
  // 5 BigInt mults/iter to 3. Bit-exact: identical mulShift calls, identical values.
  let zx = 0n, zy = 0n, zx2 = 0n, zy2 = 0n;
  for (let n = 1; n <= maxIter; n++) {
    const ny = mulShift(2n * zx, zy, prec) + cy;
    const nx = zx2 - zy2 + cx;
    zx = nx; zy = ny;
    zx2 = mulShift(zx, zx, prec);
    zy2 = mulShift(zy, zy, prec);
    if ((zx2 + zy2) > bail) return n;
  }
  return maxIter;
}

// Shared orbit stepper (Spawn 29/30): iterate Z ← Z² + C in BigInt fixed point
// from state (bx, by) = Z_fromN, for n = fromN+1 .. maxIter, calling
// store(n, dx, dy, m2) with the double export of each new Z_n. Both
// computeReference (fromN = 0) and extendReference (resume from a cached tail)
// run THIS loop, so a resumed orbit is BIT-IDENTICAL to an uninterrupted one
// (same BigInt ops in the same order; unit-tested).
//
// 2-mult complex square (Spawn 29): re = (bx+by)(bx−by), im = (2bx)·by — one
// BigInt mult fewer per iteration than bx²,by²,bx·by (~33% of the build; the
// build dominates a deep frame now that the GPU raster is fast). (bx+by)(bx−by)
// rounds ONCE where bx²−by² rounded twice — a ≤1-ulp-at-2^-prec difference per
// step, the same magnitude class as the existing mulShift rounding; per-pixel
// escape counts are gated bit-exact vs the BigInt-exact oracle by probe-wall.
//
// The escape test uses the PREVIOUS z's squares (|Z_{n-1}|² > 4, tested after
// storing Z_n — the historical semantics) — those squares no longer exist under
// the 2-mult form, so the decision runs on pm2 (the previous iteration's double
// |z|², relative error ~2^-51) with an EXACT BigInt re-check only within ±1e-12
// of the threshold (a sliver the double error cannot bridge; hit ~never).
// pm2 at entry must be |Z_fromN|² as the DOUBLE m2 of that iteration (z2[fromN]).
function iterateOrbit(cx, cy, prec, bx, by, pm2, fromN, maxIter, store, onProgress) {
  const four = 4n << BigInt(prec);
  let len = fromN;
  let escaped = false;
  let pbx = 0n, pby = 0n; // previous z (for the rare exact re-check)
  for (let n = fromN + 1; n <= maxIter; n++) {
    // Z = Z^2 + C  (complex, 2 mults)
    const newX = mulShift(bx + by, bx - by, prec) + cx;
    const newY = mulShift(2n * bx, by, prec) + cy;
    pbx = bx; pby = by;
    bx = newX; by = newY;

    const dx = toDouble(bx, prec);
    const dy = toDouble(by, prec);
    const m2 = dx * dx + dy * dy;
    store(n, dx, dy, m2);
    len = n;

    if (pm2 > 4 - 1e-12 &&
        (pm2 > 4 + 1e-12 ||
         (mulShift(pbx, pbx, prec) + mulShift(pby, pby, prec)) > four)) { escaped = true; break; }
    pm2 = m2;
    if (onProgress && (n & 4095) === 0) onProgress(n, maxIter);
  }
  return { bx, by, len, escaped };
}

// Compute the reference orbit.
//   center: { x: BigInt, y: BigInt, prec: number }
//   maxIter: cap on orbit length
//   onProgress(i, maxIter): optional, called every ~4096 iterations
// Returns { zx, zy, z2, len, escaped, tailX, tailY }
//   zx,zy : Float64Array of the orbit (index n = z_n), length = len+1 incl z_0=0
//   z2    : Float64Array of |z_n|^2 (for glitch detection)
//   len   : number of valid iterations (z_0..z_len); escaped at z_len if escaped
//   escaped: whether the reference escaped before maxIter
//   tailX/tailY: the BigInt state Z_len — resume material for extendReference
//   (the reference-reuse cache extends an orbit instead of rebuilding it).
export function computeReference(center, maxIter, onProgress) {
  const { x: cx, y: cy, prec } = center;
  const cap = maxIter + 1;
  const zx = new Float64Array(cap);
  const zy = new Float64Array(cap);
  const z2 = new Float64Array(cap);
  zx[0] = 0; zy[0] = 0; z2[0] = 0;

  const r = iterateOrbit(cx, cy, prec, 0n, 0n, 0, 0, maxIter,
    (n, dx, dy, m2) => { zx[n] = dx; zy[n] = dy; z2[n] = m2; }, onProgress);
  if (onProgress) onProgress(r.len, maxIter);

  return {
    zx: zx.subarray(0, r.len + 1),
    zy: zy.subarray(0, r.len + 1),
    z2: z2.subarray(0, r.len + 1),
    len: r.len,
    escaped: r.escaped,
    tailX: r.bx,
    tailY: r.by,
  };
}

// Extend a reference orbit from a cached tail state (Spawn 30 — reference reuse).
//   center  : { x, y, prec } — the ORIGINAL reference point at the ORIGINAL prec
//   tail    : { bx: BigInt, by: BigInt, pm2: number } — Z_fromN and its double |Z|²
//             (pm2 MUST be the z2[fromN] the original run stored — it feeds the
//             guarded escape test exactly as the uninterrupted loop would)
//   fromN   : the iteration tail represents (== the cached len)
//   maxIter : extend the orbit to this iteration
// Returns { zx, zy, z2, len, escaped, tailX, tailY } where the arrays hold ONLY
// the NEW segment: index k = iteration fromN+1+k, length len−fromN. len is the
// absolute last valid iteration. Resuming through iterateOrbit makes the merged
// orbit BIT-IDENTICAL to a single computeReference(center, maxIter) run.
export function extendReference(center, tail, fromN, maxIter, onProgress) {
  const { x: cx, y: cy, prec } = center;
  const segCap = Math.max(0, maxIter - fromN);
  const zx = new Float64Array(segCap);
  const zy = new Float64Array(segCap);
  const z2 = new Float64Array(segCap);

  const r = iterateOrbit(cx, cy, prec, tail.bx, tail.by, tail.pm2, fromN, maxIter,
    (n, dx, dy, m2) => { const k = n - fromN - 1; zx[k] = dx; zy[k] = dy; z2[k] = m2; }, onProgress);

  const segLen = r.len - fromN;
  return {
    zx: zx.subarray(0, segLen),
    zy: zy.subarray(0, segLen),
    z2: z2.subarray(0, segLen),
    len: r.len,
    escaped: r.escaped,
    tailX: r.bx,
    tailY: r.by,
  };
}
