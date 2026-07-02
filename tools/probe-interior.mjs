// probe-interior.mjs — where does the post-SA iteration work actually go? (measure-first)
//
// Hypothesis: after SA skips the leading run, the dominant remaining cost is INTERIOR
// pixels — they run from `skip` all the way to maxIter (tens of thousands of steps) doing
// nothing useful, and on a GPU one interior lane drags its whole warp to maxIter. If interior
// pixels consume most of the post-SA work, a cheap interior/period test is a real lever; if
// the frame is mostly moderate-count escaping pixels, it is not.
//
// Renders a real deep boundary view with the PRODUCTION config (order-5 SA on) using the CPU
// oracle, counting actual steps per pixel (escapePerturb stats). Reports: interior fraction,
// share of total post-SA steps consumed by interior pixels, and the per-pixel step histogram.
//
//   node tools/probe-interior.mjs
//   BITS=120 W=200 H=150 node tools/probe-interior.mjs
import { fromDecimalString, precForRadius } from '../src/math/bignum.js';
import { computeReference } from '../src/math/reference.js';
import { computeSeries } from '../src/math/series.js';
import { escapePerturb, pixelDelta } from '../src/math/perturb.js';

const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

const BITS = (process.env.BITS || '120,271,400').split(',').map(Number);
const W = Number(process.env.W || 160);
const H = Number(process.env.H || 120);
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

for (const bits of BITS) {
  const radius = 1.5 * 2 ** -bits;
  const maxIter = autoIter(bits);
  const prec = precForRadius(radius, 80);
  const cx = fromDecimalString(RE, prec), cy = fromDecimalString(IM, prec);
  const ref = computeReference({ x: cx, y: cy, prec }, maxIter);

  const view = { radius, width: W, height: H };
  const dcHalfX = radius * (W / H), dcHalfY = radius;
  const sa = computeSeries(ref, radius, dcHalfX, dcHalfY, { maxIter });
  const skip = sa.skip || 0;

  let nInterior = 0, totalSteps = 0, interiorSteps = 0, maxSteps = 0;
  let escSteps = 0, escCount = 0;
  const buckets = [0, 0, 0, 0, 0, 0]; // step ranges (relative to a fraction of maxIter)
  const bnd = [0.01, 0.05, 0.2, 0.5, 0.9, 1.01].map((f) => f * maxIter);
  for (let j = 0; j < H; j++) {
    for (let i = 0; i < W; i++) {
      const { dcx, dcy } = pixelDelta(view, i, j);
      const stats = { steps: 0, jumps: 0, jumpIters: 0 };
      const res = escapePerturb(ref, dcx, dcy, maxIter, 1 << 16, 1e-6, sa, null, stats);
      const steps = stats.steps;
      totalSteps += steps;
      if (steps > maxSteps) maxSteps = steps;
      const interior = res.n >= maxIter;
      if (interior) { nInterior++; interiorSteps += steps; }
      else { escSteps += steps; escCount++; }
      for (let b = 0; b < bnd.length; b++) { if (steps <= bnd[b]) { buckets[b]++; break; } }
    }
  }
  const N = W * H;
  console.log(`\n=== 2^-${bits}  (${W}×${H} px, maxIter ${maxIter}, refLen ${ref.len}, SA skip ${skip} = ${(100*skip/maxIter).toFixed(1)}%) ===`);
  console.log(`interior pixels:        ${nInterior}/${N} = ${(100*nInterior/N).toFixed(1)}%`);
  console.log(`total post-SA steps:    ${totalSteps}`);
  console.log(`  from interior pixels: ${interiorSteps} = ${(100*interiorSteps/totalSteps).toFixed(1)}%  (each ~${nInterior?Math.round(interiorSteps/nInterior):0} steps)`);
  console.log(`  from escaping pixels: ${escSteps} = ${(100*escSteps/totalSteps).toFixed(1)}%  (each ~${escCount?Math.round(escSteps/escCount):0} steps avg)`);
  console.log(`max steps any pixel:    ${maxSteps}  (= the warp-divergence ceiling; maxIter-skip = ${maxIter-skip})`);
  const labels = ['≤1%', '≤5%', '≤20%', '≤50%', '≤90%', '≤100%'];
  console.log(`step histogram (as % of maxIter): ` + buckets.map((c, b) => `${labels[b]}:${(100*c/N).toFixed(0)}%`).join('  '));
}
