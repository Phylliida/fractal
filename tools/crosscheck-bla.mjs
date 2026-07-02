// crosscheck-bla.mjs — prove BLA is CORRECT and measure its iteration-reduction ceiling.
//
// BLA (bla.js) skips RUNS of linear perturbation iterations throughout the orbit — in
// particular the post-rebase re-growth phases that series approximation (SA) cannot
// touch. This tool renders the genuine deep boundary coordinate at several depths in
// four combos and (1) asserts BLA does not change escape counts vs the no-BLA oracle,
// (2) reports the total per-pixel WORK (true steps + BLA jumps) so the speedup ceiling
// is the work-reduction ratio. The kept-iteration cost is what the GPU's ~8ms floor is
// made of (see NOTES), so the SA+BLA vs SA work ratio bounds the eventual GPU win.
//
//   node tools/crosscheck-bla.mjs
//   BITS=120,271,400 W=48 H=48 EPS=24 node tools/crosscheck-bla.mjs   # EPS = -log2(blaEps)
import { fromDecimalString, precForRadius } from '../src/math/bignum.js';
import { renderImage } from '../src/math/render.js';

const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

const BITS = (process.env.BITS || '120,271,400').split(',').map(Number);
const W = Number(process.env.W || 48), H = Number(process.env.H || 48);
const EPS = (process.env.EPS || '24').split(',').map(Number).map((e) => 2 ** -e);  // blaEps sweep
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

// Total per-pixel work = true steps + BLA jumps (a jump ≈ a true step in cost: one
// complex mul-add). The oracle's work is just its true steps. Speedup ceiling = ratio.
function work(stats) { return stats.steps + stats.jumps; }

console.log(`BLA correctness + work-reduction — ${W}×${H}, deep boundary coordinate`);
console.log(`blaEps sweep: ${EPS.map((e) => `2^${Math.round(Math.log2(e))}`).join(', ')}\n`);

let fail = false;
for (const bits of BITS) {
  const radius = 1.5 * 2 ** -bits;
  const maxIter = autoIter(bits);
  const prec = precForRadius(radius, 80);
  const x = fromDecimalString(RE, prec), y = fromDecimalString(IM, prec);
  const view = { x, y, prec, radius, width: W, height: H };

  // Oracle: no SA, no BLA — the validated escape counts + the baseline work.
  const sBase = { steps: 0, jumps: 0, jumpIters: 0 };
  const oracle = renderImage(view, maxIter, { stats: sBase });
  const baseWork = work(sBase);

  // SA only (the current shipping path) — for the marginal-BLA-on-top-of-SA number.
  const sSA = { steps: 0, jumps: 0, jumpIters: 0 };
  const saOnly = renderImage(view, maxIter, { series: true, stats: sSA });

  console.log(`2^-${bits}   maxIter ${maxIter}   refLen ${oracle.refLen}   SA skip ${saOnly.saSkip} (${(100 * saOnly.saSkip / maxIter).toFixed(1)}%)`);
  console.log(`  baseline work ${baseWork}   |   SA-only work ${work(sSA)}  (${(baseWork / work(sSA)).toFixed(2)}× vs baseline)`);
  console.log('  blaEps    combo        mism maxΔn |   work    vs-base  vs-SA   lvls');

  for (const eps of EPS) {
    for (const combo of [{ name: 'BLA only', o: { bla: { eps } } },
                         { name: 'SA + BLA', o: { series: true, bla: { eps } } }]) {
      const st = { steps: 0, jumps: 0, jumpIters: 0 };
      const r = renderImage(view, maxIter, { ...combo.o, stats: st });
      let mism = 0, maxDn = 0;
      for (let i = 0; i < oracle.iters.length; i++) {
        const d = Math.abs(oracle.iters[i] - r.iters[i]);
        if (d > 0) { mism++; if (d > maxDn) maxDn = d; }
      }
      // Gate on the bulk fraction (project standard); a missed escape flips a whole region.
      const bad = mism > Math.ceil(0.01 * oracle.iters.length);
      if (bad) fail = true;
      const w = work(st);
      console.log(
        `  2^${String(Math.round(Math.log2(eps))).padStart(3)}  ${combo.name.padEnd(10)} ` +
        `${String(mism).padStart(5)} ${String(maxDn).padStart(5)} | ${String(w).padStart(8)}  ` +
        `${(baseWork / w).toFixed(2).padStart(6)}× ${(work(sSA) / w).toFixed(2).padStart(5)}×  ${r.blaLevels}` +
        `${bad ? '  <-- FAIL (escape counts changed)' : ''}`);
    }
  }
  console.log('');
}
console.log('mism = pixels whose escape count differs from the no-BLA oracle. A correct BLA leaves');
console.log('them ~0 (measure-zero ill-conditioned boundary pixels only). "vs-SA" is the marginal');
console.log('work reduction BLA buys ON TOP of SA — it bounds the additional GPU-escape speedup.');
process.exit(fail ? 1 : 0);
