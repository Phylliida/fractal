// probe-wall.mjs — find the TRUE deep-zoom precision wall of the double-precision
// perturbation renderer by comparing it, per pixel, to the BigInt-EXACT oracle.
//
// Why this is different from probe-deep500.mjs: that tool compares the GPU fe/rescaled
// engine to the CPU perturbation oracle — but BOTH share the SAME double-precision dc
// (pixelDelta returns doubles) and the same double delta-iteration, so a failure of the
// double representation itself is INVISIBLE to it (both would be wrong identically).
// This tool compares the renderer's double path to escapeBigInt computed at the pixel's
// coordinate in full BigInt precision — the authoritative ground-truth oracle used by
// the M4 correctness tests (B/C/C2 at 2^400). It extends that methodology into the deep
// band to answer: down to what radius does the double-dc perturbation engine still match
// ground truth, and WHERE / WHY does it break?
//
// What the double path can lose deep:
//  (1) dc REPRESENTATION: dc ~ radius is a normal double down to ~2^-1022; below that it
//      goes subnormal and loses mantissa bits; the per-pixel step (2*radius/height)
//      underflows even a little earlier (so adjacent pixels stop being distinguishable).
//  (2) delta-ITERATION rounding: dz iterates in double; rebasing should bound the
//      accumulation independent of maxIter (this is the perturbation correctness claim).
// To see BOTH, the oracle escape count is taken at the pixel's TRUE geometric offset,
// computed in BigInt directly from the (double) radius and integer pixel index — NOT via
// the double dc. So the mismatch combines (2) the delta-iteration error AND (1) the
// double-dc rounding/underflow of the true offset: at normal depth it's ~0 (the double
// offset is accurate to 53 bits), and it SPIKES once dc/step go subnormal — that spike
// IS the wall. The "dc normal?" column shows the representation status analytically too.
//
//   node tools/probe-wall.mjs                         (defaults; uses the deep coord below)
//   BITS=700,1000,1022 GRID=10 node tools/probe-wall.mjs
//   RE=… IM=… BITS=… node tools/probe-wall.mjs
import { fromDecimalString, fromDouble, precForRadius } from '../src/math/bignum.js';
import { computeReference, escapeBigInt } from '../src/math/reference.js';
import { escapePerturb, pixelDelta } from '../src/math/perturb.js';

// Round-to-nearest BigInt division (negligible at prec ~1100 bits, but exact-ish).
function divRound(num, den) {
  const q = num / den, r = num % den;
  const twice = (r < 0n ? -r : r) * 2n;
  if (twice >= (den < 0n ? -den : den)) return q + (((num < 0n) === (den < 0n)) ? 1n : -1n);
  return q;
}

// Deep boundary coordinate (good to ~2^-1010), from tools/gen-deep-coord.mjs TARGET=1010.
// Override via env. Falls back to the 2^-520 coordinate if the deeper one isn't wired yet.
const RE = process.env.RE || '-1.36907801852007942475642195255842390507898755054806063192697077066791713514355984183138458726946811817965776911647030941610560712351758961905260413267881816077787468843042104473375047052708330910725794245355803318429014122085514645861629120057518448513757126947453394807972239491257672294147147329605986816418497391255';
const IM = process.env.IM || '-0.07181767684485165923310781763936598852851027483522667720017724564851969024779139147729278915346152412576593892660509339580393958961060780193289065314335145345663719866400233165980940228073309024466767811760569409475826736402394026213081093650425780165523756647946667328862361557720732262149429441085094730412969264871';
// Default sweep spans the clean band → the wall. NOTE: the generated coordinate is a
// genuine boundary filament only to ~2^-1010 (below that the descent's own double dc
// goes subnormal). Depths ≥ ~1012 use a slightly-stale center, but the dc-REPRESENTATION
// wall (subnormal dc / non-distinct step) is coordinate-independent, so they still
// demonstrate exactly where/why the double path collapses.
const BITS = (process.env.BITS || '700,900,1000,1010,1016,1022,1028').split(',').map(Number);
const GRID = Number(process.env.GRID || 10);
const VIEWSCALE = Number(process.env.VIEWSCALE || 1.5); // half-height factor (matches probe-deep500)
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));
const MINNORMAL = 2.2250738585072014e-308; // smallest normal double

const maxBits = Math.max(...BITS);
const prec = precForRadius(2 ** -maxBits, 160);
const cx = fromDecimalString(RE, prec);
const cy = fromDecimalString(IM, prec);

console.log(`probe-wall: double perturbation vs BigInt-EXACT oracle, prec=${prec} bits, grid ${GRID}²`);
console.log(`coordinate good to ~2^-1010 (gen-deep-coord). The signal: mism% FLAT & low (= the`);
console.log(`ill-conditioned boundary floor) down to some depth, then RISING = the precision wall.\n`);
console.log('depth     maxIter  refLen | esc  | mism  mism%  maxΔn near-cap | dcRelErr  dcStep      dc?');

for (const bits of BITS) {
  const radius = VIEWSCALE * 2 ** -bits;
  const maxIter = autoIter(bits);
  const ref = computeReference({ x: cx, y: cy, prec }, maxIter);
  const view = { radius, width: GRID, height: GRID }; // square grid → aspect = 1
  const step = (2 * radius) / GRID;
  const Rbig = fromDouble(radius, prec);              // the (double) radius, exact in BigInt
  const Rmag = Rbig < 0n ? -Rbig : Rbig;
  const Gden = BigInt(GRID);

  let esc = 0, mism = 0, maxDn = 0, nearCap = 0, minAbsDc = Infinity, subnormal = false;
  let distinct = new Set(), maxRelErr = 0;
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      // What the renderer uses: the per-pixel dc as a DOUBLE (pixelDelta).
      const { dcx, dcy } = pixelDelta(view, gx, gy);
      const a = Math.abs(dcx), b = Math.abs(dcy);
      if (a > 0) minAbsDc = Math.min(minAbsDc, a);
      if (b > 0) minAbsDc = Math.min(minAbsDc, b);
      if ((a > 0 && a < MINNORMAL) || (b > 0 && b < MINNORMAL)) subnormal = true;
      distinct.add(dcx + ',' + dcy);

      // The TRUE geometric offset for this pixel, in BigInt, straight from the (double)
      // radius and integer index — NOT through the double dc. offset = radius·(2g−GRID)/GRID.
      const trueX = divRound(Rbig * BigInt(2 * gx - GRID), Gden);
      const trueY = divRound(Rbig * BigInt(2 * gy - GRID), Gden);
      // Quantify the double's offset error vs the true offset, as a fraction of the view
      // RADIUS (a stable denominator — unlike the per-pixel true magnitude, which is ~0 at
      // the grid center). Computed entirely in BigInt: round the (double) dc back to fixed
      // point, subtract the exact offset, and divide by |radius|. (toDouble underflows at
      // prec > 1074, so this ratio must NOT go through a double.) ~2^-52 when dc is a normal
      // double; rises to O(1) once dc/step go subnormal and the double can't place the
      // pixel — that IS the wall.
      const errX = fromDouble(dcx, prec) - trueX, errY = fromDouble(dcy, prec) - trueY;
      const errMag = (errX < 0n ? -errX : errX) + (errY < 0n ? -errY : errY); // L1, order-of-magnitude
      const relErr = Rmag > 0n ? Number((errMag << 53n) / Rmag) / 2 ** 53 : 0;
      if (relErr > maxRelErr) maxRelErr = relErr;

      const np = escapePerturb(ref, dcx, dcy, maxIter).n;            // renderer (double dc)
      const ex = escapeBigInt(cx + trueX, cy + trueY, prec, maxIter); // ground truth (exact offset)
      if (np < maxIter || ex < maxIter) esc++;
      if (np !== ex) {
        mism++;
        const d = Math.abs(np - ex);
        if (d > maxDn) maxDn = d;
        if (Math.min(np, ex) > maxIter - maxIter * 0.02) nearCap++; // both near the cap = ill-conditioned
      }
    }
  }
  const total = GRID * GRID;
  const denom = Math.max(1, esc);
  const mismPct = (100 * mism) / denom;
  const distinctNote = distinct.size < total ? ` ⚠ ${distinct.size}/${total} distinct` : '';
  console.log(
    `2^-${String(bits).padEnd(4)} ${String(maxIter).padStart(7)} ${String(ref.len).padStart(6)} | ` +
    `${String(esc).padStart(3)}/${total} | ${String(mism).padStart(4)} ${mismPct.toFixed(2).padStart(6)}% ${String(maxDn).padStart(5)} ${String(nearCap).padStart(3)}/${mism || 0} | ` +
    `${maxRelErr.toExponential(1)}  ${step.toExponential(1)} ${subnormal ? 'SUBNORM✗' : 'normal'}${distinctNote}`
  );
}

console.log('\nInterpretation:');
console.log(' • dcRelErr ~1e-16 + maxΔn small + mismatches "near-cap" ⇒ the diffs are ill-conditioned');
console.log('   boundary pixels (the measure-zero set any 53-bit method is noisy on), NOT a wall.');
console.log(' • dcRelErr blowing up + mism% rising + maxΔn large + "SUBNORM✗"/non-distinct dc ⇒ the WALL:');
console.log('   the double dc/step has run out of mantissa. That radius is the true precision floor.');
