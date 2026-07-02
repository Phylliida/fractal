// bench-ref.mjs — time the CPU BigInt REFERENCE ORBIT build (Spawn 29).
//
// After the skip-aware strip fix, the reference build dominates a deep frame
// (~450ms of the ~726ms at 2^-400 — the GPU raster is ~180ms). This times
// computeReference at deep-render parameters (prec = zoomBits+64, autoMaxIter)
// and escapeBigInt at probe-wall-like parameters, pure Node (no browser).
//
//   node tools/bench-ref.mjs
//   BITS=271,400,700 REPS=3 node tools/bench-ref.mjs
import { fromDecimalString } from '../src/math/bignum.js';
import { computeReference, escapeBigInt } from '../src/math/reference.js';

const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const BITSET = (process.env.BITS || '271,400').split(',').map(Number);
const REPS = Number(process.env.REPS || 3);
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

console.log('reference orbit build (computeReference) — pure Node, min of ' + REPS);
console.log('depth    prec  maxIter |    ms    | len     escaped | iters/ms');
for (const bits of BITSET) {
  const prec = bits + 64;
  const maxIter = autoIter(bits);
  const center = { x: fromDecimalString(RE, prec), y: fromDecimalString(IM, prec), prec };
  let best = Infinity, out = null;
  for (let r = 0; r < REPS; r++) {
    const t0 = performance.now();
    out = computeReference(center, maxIter);
    best = Math.min(best, performance.now() - t0);
  }
  console.log(`2^-${String(bits).padEnd(4)} ${String(prec).padStart(4)} ${String(maxIter).padStart(8)} | ${best.toFixed(1).padStart(8)} | ${String(out.len).padStart(7)} ${String(out.escaped).padEnd(7)} | ${(out.len / best).toFixed(0)}`);
}

// escapeBigInt at a modest depth (the per-pixel exact oracle — probe-wall grids call
// this hundreds of times; its speed sets the probe's wall-clock).
{
  const bits = 120, prec = bits + 96, maxIter = 30400;
  const cx = fromDecimalString(RE, prec), cy = fromDecimalString(IM, prec);
  let best = Infinity, n = 0;
  for (let r = 0; r < REPS; r++) {
    const t0 = performance.now();
    n = escapeBigInt(cx, cy, prec, maxIter);
    best = Math.min(best, performance.now() - t0);
  }
  console.log(`\nescapeBigInt @2^-${bits} prec ${prec} maxIter ${maxIter}: ${best.toFixed(1)}ms (n=${n})`);
}
