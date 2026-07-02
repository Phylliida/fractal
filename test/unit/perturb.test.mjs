import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderNaive } from '../../src/math/naive.js';
import { computeReference, escapeBigInt } from '../../src/math/reference.js';
import { escapePerturb, pixelDelta } from '../../src/math/perturb.js';
import { renderImage, engineForRadius, autoMaxIter, PERTURB_RADIUS } from '../../src/math/render.js';
import { fromDecimalString, fromDouble, precForRadius } from '../../src/math/bignum.js';

// BigInt-exact escape count of a pixel, using the SAME double dc the perturbation
// engine uses, lifted into high precision. This is the authoritative oracle.
function truthAt(centerStrX, centerStrY, view, px, py, maxIter, prec) {
  const { dcx, dcy } = pixelDelta(view, px, py);
  const ex = fromDecimalString(centerStrX, prec) + fromDouble(dcx, prec);
  const ey = fromDecimalString(centerStrY, prec) + fromDouble(dcy, prec);
  return escapeBigInt(ex, ey, prec, maxIter);
}

// ---- A: full render pipeline (auto reference selection) matches BigInt ----
// Moderate-deep zoom in the seahorse valley — the regime perturbation is for.
test('A: renderImage matches BigInt-exact over a grid (zoom ~2^30, seahorse)', () => {
  const radius = Math.pow(2, -30);
  const maxIter = 3000;
  // A boundary location (mix of interior + exterior pixels at this radius).
  const cxStr = '-0.74364388703715874', cyStr = '0.13182590420531197';
  const W = 24, H = 24;
  const prec = precForRadius(radius, 160); // generous prec for an exact oracle
  const viewHP = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius, width: W, height: H };
  const out = renderImage(viewHP, maxIter);

  const view = { radius, width: W, height: H };
  let checked = 0, mismatch = 0, escaped = 0, inside = 0;
  for (let py = 0; py < H; py += 4) {
    for (let px = 0; px < W; px += 4) {
      const truth = truthAt(cxStr, cyStr, view, px, py, maxIter, prec);
      const got = out.iters[py * W + px];
      if (Math.abs(got - truth) > 1) mismatch++;
      if (truth >= maxIter) inside++; else escaped++;
      checked++;
    }
  }
  assert.equal(out.glitches, 0, 'no glitches expected');
  assert.equal(mismatch, 0, `${mismatch}/${checked} disagree with BigInt`);
  assert.ok(escaped > 0 && inside > 0, `non-trivial field: escaped=${escaped} inside=${inside}`);
});

// ---- B: direct perturbation vs BigInt below double's limit (~2^45) ----
test('B: perturb == BigInt-exact over a region at zoom ~2^45', () => {
  const radius = Math.pow(2, -45);
  const maxIter = 4000;
  const cxStr = '-0.74364388703715874';
  const cyStr = '0.13182590420531197';
  const prec = precForRadius(radius);
  const center = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec };
  const ref = computeReference(center, maxIter);
  const W = 24, H = 24;
  const view = { radius, width: W, height: H };
  let mismatch = 0, escaped = 0, inside = 0;
  for (let py = 0; py < H; py += 3) {
    for (let px = 0; px < W; px += 3) {
      const { dcx, dcy } = pixelDelta(view, px, py);
      const p = escapePerturb(ref, dcx, dcy, maxIter).n;
      const truth = escapeBigInt(center.x + fromDouble(dcx, prec), center.y + fromDouble(dcy, prec), prec, maxIter);
      if (Math.abs(p - truth) > 1) mismatch++;
      if (truth >= maxIter) inside++; else escaped++;
    }
  }
  assert.equal(mismatch, 0, `${mismatch} pixels disagree with BigInt oracle`);
  assert.ok(escaped > 0 && inside > 0, `escaped=${escaped} inside=${inside}`);
});

// ---- C: deep-zoom — perturb tracks BigInt-exact at ~2^120 ----
test('C: perturb == BigInt-exact at points, zoom ~2^120', () => {
  const radius = Math.pow(2, -120);
  const maxIter = 6000;
  const cxStr = '-1.7689249285417043775265190984183';
  const cyStr = '0.0000000000000000005263331599528940';
  const prec = precForRadius(radius);
  const center = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec };
  const ref = computeReference(center, maxIter);
  const view = { radius, width: 32, height: 32 };
  let mismatch = 0;
  for (const [px, py] of [[0, 0], [8, 24], [16, 16], [31, 31], [24, 4], [4, 28]]) {
    const { dcx, dcy } = pixelDelta(view, px, py);
    const p = escapePerturb(ref, dcx, dcy, maxIter).n;
    const truth = escapeBigInt(center.x + fromDouble(dcx, prec), center.y + fromDouble(dcy, prec), prec, maxIter);
    if (Math.abs(p - truth) > 1) mismatch++;
  }
  assert.equal(mismatch, 0, `${mismatch} deep points disagree`);
});

// ---- C2: very deep — the headline capability, ~2^400 ----
// We cannot run a full BigInt grid here cheaply, but we CAN verify a handful of
// pixels against the BigInt oracle to prove correctness at extreme depth.
test('C2: perturb == BigInt-exact at a few points, zoom ~2^400', () => {
  const radius = Math.pow(2, -400);
  const maxIter = 1500;
  // A center deep inside the set (period-doubling cascade on the real axis is a
  // safe interior locus to high precision). Use many digits of -7/4 region.
  const cxStr = '-1.74999999999999999999999999999999999999999999999999999999999999';
  const cyStr = '0.0';
  const prec = precForRadius(radius); // ~464 bits
  const center = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec };
  const ref = computeReference(center, maxIter);
  const view = { radius, width: 16, height: 16 };
  let checked = 0, mismatch = 0;
  for (const [px, py] of [[0, 0], [8, 8], [15, 15], [3, 12]]) {
    const { dcx, dcy } = pixelDelta(view, px, py);
    const p = escapePerturb(ref, dcx, dcy, maxIter).n;
    const truth = escapeBigInt(center.x + fromDouble(dcx, prec), center.y + fromDouble(dcy, prec), prec, maxIter);
    if (Math.abs(p - truth) > 1) mismatch++;
    checked++;
  }
  assert.equal(mismatch, 0, `${mismatch}/${checked} disagree at 2^400`);
});

// ---- D: HP reference orbit agrees with naive double orbit where double is
// valid (early iterations, before chaotic divergence). Validates HP arithmetic.
test('D: HP reference orbit matches naive double orbit for early iterations', () => {
  const maxIter = 60;
  const prec = precForRadius(0.5);
  const center = { x: fromDecimalString('-0.745', prec), y: fromDecimalString('0.113', prec), prec };
  const ref = computeReference(center, maxIter);
  let x = 0, y = 0;
  for (let n = 1; n <= Math.min(45, ref.len); n++) {
    const nx = x * x - y * y - 0.745;
    const ny = 2 * x * y + 0.113;
    x = nx; y = ny;
    assert.ok(Math.abs(ref.zx[n] - x) < 1e-11, `re@${n}: ${ref.zx[n]} vs ${x}`);
    assert.ok(Math.abs(ref.zy[n] - y) < 1e-11, `im@${n}`);
  }
});

// ---- E: rebasing — an orbit that passes near zero still matches the oracle ----
test('E: a point whose orbit dips near zero matches the oracle (rebasing)', () => {
  const maxIter = 2000;
  const prec = precForRadius(1e-6);
  const center = { x: fromDecimalString('-1.0', prec), y: fromDecimalString('0.0', prec), prec };
  const ref = computeReference(center, maxIter);
  const view = { radius: 1e-6, width: 16, height: 16 };
  for (let py = 0; py < 16; py += 4) {
    for (let px = 0; px < 16; px += 4) {
      const { dcx, dcy } = pixelDelta(view, px, py);
      const p = escapePerturb(ref, dcx, dcy, maxIter).n;
      const truth = escapeBigInt(center.x + fromDouble(dcx, prec), center.y + fromDouble(dcy, prec), prec, maxIter);
      assert.ok(Math.abs(p - truth) <= 1, `px${px},py${py}: ${p} vs ${truth}`);
    }
  }
});

// ---- F: bulk agreement with naive at shallow zoom ----
// At shallow zoom both are ~53-bit; they agree on the vast majority and only
// differ on ill-conditioned boundary pixels (where BigInt is the arbiter).
test('F: perturb agrees with naive for the bulk at shallow zoom', () => {
  const maxIter = 800;
  const radius = 0.01;
  const cxStr = '-0.745', cyStr = '0.113';
  const W = 48, H = 48;
  const prec = precForRadius(radius);
  const viewHP = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius, width: W, height: H };
  const out = renderImage(viewHP, maxIter);
  const oracle = renderNaive({ centerX: -0.745, centerY: 0.113, radius, width: W, height: H }, maxIter);
  let within1 = 0;
  for (let i = 0; i < W * H; i++) if (Math.abs(out.iters[i] - oracle.iters[i]) <= 1) within1++;
  const rate = within1 / (W * H);
  assert.ok(rate >= 0.97, `only ${(rate * 100).toFixed(2)}% agree within 1`);
});

// ---- G: engine dispatch + iteration heuristics ----
test('G: engineForRadius/autoMaxIter behave sanely', () => {
  assert.equal(engineForRadius(0.5), 'naive');
  assert.equal(engineForRadius(PERTURB_RADIUS / 2), 'perturb');
  assert.ok(autoMaxIter(Math.pow(2, -400)) > autoMaxIter(0.5));
  assert.ok(autoMaxIter(0.5) >= 256);
});
