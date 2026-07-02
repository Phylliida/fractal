// probe-sa-cap.mjs — does the coarse-pass min-escape CAP fix the moderate-zoom SA
// mismatches, or are they cap-independent ill-conditioned boundary pixels?
//
// For each depth on a STRUCTURED coordinate (seahorse valley, which still has filaments
// at 2^-35..-90 where grid+escape-guard alone showed mismatches), render:
//   - no-SA            (the oracle)
//   - SA, no cap       (grid + escape-guard only — what renderImage/GPU-without-coarse does)
//   - SA, true-min cap (skipCap = the TRUE min escape over ALL pixels — tightest safe cap)
//   - SA, coarse cap   (skipCap = min escape over the step-8 grid — what the WORKER ships)
// and report the escape-count mism for each. If the true-min cap drops mism to ~0, the
// mismatches are MISSED EARLY ESCAPES and the GPU SA needs a cap; if the cap leaves the
// same mism, they are ill-conditioned pixels (cap-independent) and no-cap is as safe as
// the CPU oracle there.
//
//   node tools/probe-sa-cap.mjs
//   BITS=35,50,70 W=80 H=80 node tools/probe-sa-cap.mjs
import { fromDecimalString, precForRadius } from '../src/math/bignum.js';
import { renderImage } from '../src/math/render.js';

const RE = process.env.RE || '-0.743643887037158704752191506114774';
const IM = process.env.IM || '0.131825904205311970493132056385139';
const BITS = (process.env.BITS || '22,35,50,70,90').split(',').map(Number);
const W = Number(process.env.W || 80), H = Number(process.env.H || 80);
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

function minEscape(iters, w, h, step) {
  let m = Infinity;
  for (let y = 0; y < h; y += step) for (let x = 0; x < w; x += step) {
    const v = iters[y * w + x];
    if (v > 0 && v < m) m = v;          // escaped pixel (iters < maxIter); v>0 guards interior=maxIter sentinel
  }
  return m;
}
function countMism(a, b) {
  let mism = 0, maxd = 0;
  for (let i = 0; i < a.length; i++) { const d = Math.abs(a[i] - b[i]); if (d > 0) { mism++; if (d > maxd) maxd = d; } }
  return { mism, maxd };
}

console.log(`SA cap probe — ${W}×${H} seahorse valley\n`);
console.log('depth   maxIter | skipNoCap  trueMinCap coarseCap | mismNoCap  mismTrueCap  mismCoarseCap');
for (const bits of BITS) {
  const radius = 1.5 * 2 ** -bits;
  const maxIter = autoIter(bits);
  const prec = precForRadius(radius, 80);
  const x = fromDecimalString(RE, prec), y = fromDecimalString(IM, prec);
  const view = { x, y, prec, radius, width: W, height: H };

  const off = renderImage(view, maxIter, {});
  const trueMin = minEscape(off.iters, W, H, 1);          // tightest possible cap
  const coarseMin = minEscape(off.iters, W, H, 8);        // what the step-8 worker coarse sees

  const noCap = renderImage(view, maxIter, { series: {} });
  const trueCap = renderImage(view, maxIter, { series: { skipCap: Number.isFinite(trueMin) ? Math.floor(trueMin) : Infinity } });
  const coarseCap = renderImage(view, maxIter, { series: { skipCap: Number.isFinite(coarseMin) ? Math.floor(coarseMin) : Infinity } });

  const a = countMism(off.iters, noCap.iters);
  const b = countMism(off.iters, trueCap.iters);
  const c = countMism(off.iters, coarseCap.iters);
  console.log(
    `2^-${String(bits).padEnd(3)} ${String(maxIter).padStart(7)} | ` +
    `${String(noCap.saSkip).padStart(8)} ${String(trueCap.saSkip).padStart(10)} ${String(coarseCap.saSkip).padStart(9)} | ` +
    `${String(a.mism).padStart(4)}(Δ${a.maxd}) ${String(b.mism).padStart(8)}(Δ${b.maxd}) ${String(c.mism).padStart(8)}(Δ${c.maxd})`);
}
