import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderImage } from '../../src/math/render.js';
import { computeReference, escapeBigInt } from '../../src/math/reference.js';
import { computeSeries } from '../../src/math/series.js';
import { pixelDelta } from '../../src/math/perturb.js';
import { fromDecimalString, fromDouble, precForRadius } from '../../src/math/bignum.js';

// Series approximation skips the leading perturbation iterations for every pixel via a
// polynomial-in-dc seed. Its correctness gate is simple and strict: the escape counts must
// be IDENTICAL to a no-SA render (the validated oracle path). See src/math/series.js and
// tools/crosscheck-sa.mjs (the across-depth version, to 2^-500).
const cxStr = '-0.743643887037158704752191506114774';
const cyStr = '0.131825904205311970493132056385139';

function view(radius, W, H) {
  const prec = precForRadius(radius, 96);
  return { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius, width: W, height: H };
}

test('series approximation: bit-exact escape counts + a large skip at 2^100', () => {
  const radius = 1e-30, maxIter = 20000, W = 28, H = 28;
  const v = view(radius, W, H);
  const off = renderImage(v, maxIter);                      // SA off (default)
  const on = renderImage(v, maxIter, { series: true });     // SA on
  assert.ok(on.saSkip > maxIter * 0.5, `expected a large deep skip, got ${on.saSkip}/${maxIter}`);
  let mism = 0, maxd = 0;
  for (let i = 0; i < off.iters.length; i++) {
    const d = Math.abs(off.iters[i] - on.iters[i]);
    if (d) { mism++; if (d > maxd) maxd = d; }
  }
  assert.equal(mism, 0, `${mism} pixels differ (maxΔ ${maxd}) — SA changed the escape counts`);
  assert.equal(on.glitches, 0);
});

test('series approximation: 2^41 band — bulk-exact, and any diff is an ill-conditioned pixel', () => {
  // At the shallow end of the perturbation band dc is larger, so the 3-term series carries
  // real truncation; a rare ultra-sensitive boundary pixel can differ — the SAME measure-zero
  // effect the project documents for df64-vs-double / naive-vs-perturb (NOTES: "validate on
  // bulk metrics, not max"). The strong check: every differing pixel is one where the NO-SA
  // double render ALSO disagrees with the BigInt oracle — so SA never corrupts a
  // well-conditioned pixel, it only re-perturbs ones no double method gets right.
  const radius = 5e-13, maxIter = 8000, W = 32, H = 32;
  const prec = precForRadius(radius, 160);
  const v = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius, width: W, height: H };
  const off = renderImage(v, maxIter);
  const on = renderImage(v, maxIter, { series: true });
  const diffs = [];
  for (let i = 0; i < off.iters.length; i++) if (off.iters[i] !== on.iters[i]) diffs.push(i);
  assert.ok(diffs.length <= Math.ceil(0.01 * off.iters.length), `${diffs.length} pixels differ — too many to be measure-zero boundary pixels (a missed escape would flip a whole region)`);
  for (const i of diffs) {
    const px = i % W, py = (i / W) | 0;
    const { dcx, dcy } = pixelDelta({ radius, width: W, height: H }, px, py);
    const truth = escapeBigInt(fromDecimalString(cxStr, prec) + fromDouble(dcx, prec),
                               fromDecimalString(cyStr, prec) + fromDouble(dcy, prec), prec, maxIter);
    assert.ok(Math.abs(off.iters[i] - truth) > 1,
      `SA differs at a WELL-CONDITIONED pixel (${px},${py}): noSA ${off.iters[i]} == BigInt ${truth}, SA ${on.iters[i]} — that is a real SA bug`);
  }
});

test('series approximation: order 5 skips strictly more than order 3, still bit-exact', () => {
  // Higher-order SA (Spawn 21) extends the leading-run skip — the truncation constraint binds
  // before the escape guard at production tol, so more terms track the true dz further. The
  // skip must INCREASE with order, and an order-5 render must stay bit-exact vs no-SA.
  const radius = 1e-30, maxIter = 20000, W = 28, H = 28;
  const v = view(radius, W, H);
  const prec = precForRadius(radius, 96);
  const ref = computeReference({ x: v.x, y: v.y, prec }, maxIter);
  const o3 = computeSeries(ref, radius, radius * 1.5, radius, { order: 3, maxIter });
  const o5 = computeSeries(ref, radius, radius * 1.5, radius, { order: 5, maxIter });
  assert.ok(o5.skip > o3.skip, `order 5 should skip strictly more (got ${o5.skip} vs ${o3.skip})`);
  assert.equal(o3.dx, 0); assert.equal(o3.ex, 0);                        // order-3 carries no d,e
  assert.ok(Number.isFinite(o5.dx) && Number.isFinite(o5.ex) && o5.ex !== 0); // order-5 does
  // Forcing order 5 must not change the escape counts (more accurate seed, larger but safe skip).
  const off = renderImage(v, maxIter);
  const on5 = renderImage(v, maxIter, { series: { order: 5 } });
  let mism = 0;
  for (let i = 0; i < off.iters.length; i++) if (off.iters[i] !== on5.iters[i]) mism++;
  assert.equal(mism, 0, `${mism} pixels differ with order-5 SA — it changed the escape counts`);
});

test('series approximation: depth-adaptive safety margin (Spawn 22) — smaller deep, bit-exact', () => {
  // The skip is backed off marginFrac·validN. Below 2^-112 (the df64-clean GPU-SA band) the
  // default margin drops 0.05 → 0.02 (measured bit-exact, ~1.1–1.5× faster — probe/bench-samargin),
  // so the DEFAULT deep skip must exceed an explicit margin-0.05 skip; above 2^-112 it stays 0.05.
  const deepR = 1e-36, maxIter = 26000, W = 22, H = 22;     // 1e-36 ≈ 2^-119.6 (< 2^-112)
  const prec = precForRadius(deepR, 140);
  const dv = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius: deepR, width: W, height: H };
  const ref = computeReference({ x: dv.x, y: dv.y, prec }, maxIter);
  const def = computeSeries(ref, deepR, deepR * 1.5, deepR, { maxIter });                 // depth-adaptive default
  const m05 = computeSeries(ref, deepR, deepR * 1.5, deepR, { maxIter, marginFrac: 0.05 });
  assert.ok(def.skip > m05.skip, `deep default margin should out-skip margin 0.05 (got ${def.skip} vs ${m05.skip})`);

  // Shallow side (> 2^-112) keeps the conservative 0.05 default — same skip as an explicit 0.05.
  const shR = 1e-30;                                          // ≈ 2^-99.7 (> 2^-112)
  const sp = precForRadius(shR, 120);
  const sref = computeReference({ x: fromDecimalString(cxStr, sp), y: fromDecimalString(cyStr, sp), prec: sp }, maxIter);
  const sdef = computeSeries(sref, shR, shR * 1.5, shR, { maxIter });
  const s05 = computeSeries(sref, shR, shR * 1.5, shR, { maxIter, marginFrac: 0.05 });
  assert.equal(sdef.skip, s05.skip, 'shallow band must keep the 0.05 default margin');

  // The larger deep default skip must stay BIT-EXACT vs a no-SA render (the real gate).
  const off = renderImage(dv, maxIter);
  const on = renderImage(dv, maxIter, { series: true });
  let mism = 0;
  for (let i = 0; i < off.iters.length; i++) if (off.iters[i] !== on.iters[i]) mism++;
  assert.equal(mism, 0, `${mism} pixels differ with the deep default margin — over-skip`);
});

test('computeSeries declines (skip 0) when not worthwhile / degenerate', () => {
  const radius = 5e-13, maxIter = 8000;
  const prec = precForRadius(radius, 96);
  const ref = computeReference({ x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec }, maxIter);
  assert.equal(computeSeries(ref, radius, 0, 0).skip, 0);                                 // zero-size box -> no probes
  assert.equal(computeSeries(ref, radius, radius, radius, { minSkip: maxIter + 1 }).skip, 0); // unreachable minSkip
  const ok = computeSeries(ref, radius, radius, radius);
  assert.ok(ok.skip >= 0 && Number.isFinite(ok.invR === undefined ? 0 : ok.invR));        // well-formed result
});
