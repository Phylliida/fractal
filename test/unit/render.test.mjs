import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderImage } from '../../src/math/render.js';
import { escapeBigInt } from '../../src/math/reference.js';
import { pixelDelta } from '../../src/math/perturb.js';
import { fromDecimalString, fromDouble, precForRadius } from '../../src/math/bignum.js';

// Full production render path (reference auto-selection + perturbation +
// relocation offset) validated against the BigInt-exact oracle at a deep,
// structured location (the seahorse valley spiral).
test('renderImage deep (2^41 seahorse) matches BigInt and shows structure', () => {
  const radius = 5e-13;
  const maxIter = 8000;
  const cxStr = '-0.743643887037158704752191506114774';
  const cyStr = '0.131825904205311970493132056385139';
  const W = 40, H = 40;
  const prec = precForRadius(radius, 96);
  const viewHP = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius, width: W, height: H };
  const out = renderImage(viewHP, maxIter);

  // structure: many distinct escape counts, both escaped + (near-)inside present
  const set = new Set(out.iters);
  assert.ok(set.size > 40, `expected rich structure, got ${set.size} distinct iters`);
  assert.equal(out.glitches, 0, `unhandled glitches: ${out.glitches}`);

  // correctness: sampled pixels match the BigInt oracle within +/-1
  const view = { radius, width: W, height: H };
  let mismatch = 0, checked = 0;
  for (let py = 0; py < H; py += 6) {
    for (let px = 0; px < W; px += 6) {
      const { dcx, dcy } = pixelDelta(view, px, py);
      const ex = fromDecimalString(cxStr, prec) + fromDouble(dcx, prec);
      const ey = fromDecimalString(cyStr, prec) + fromDouble(dcy, prec);
      const truth = escapeBigInt(ex, ey, prec, maxIter);
      if (Math.abs(out.iters[py * W + px] - truth) > 1) mismatch++;
      checked++;
    }
  }
  assert.equal(mismatch, 0, `${mismatch}/${checked} pixels disagree with BigInt`);
});

// Deeper still: 2^100 — proves the pipeline produces correct, structured output
// well beyond any double-only method, matching BigInt on samples.
test('renderImage very deep (2^100) matches BigInt on samples', () => {
  const radius = 1e-30;
  const maxIter = 20000;
  const cxStr = '-0.743643887037158704752191506114774';
  const cyStr = '0.131825904205311970493132056385139';
  const W = 24, H = 24;
  const prec = precForRadius(radius, 96);
  const viewHP = { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius, width: W, height: H };
  const out = renderImage(viewHP, maxIter);
  assert.equal(out.glitches, 0);

  const view = { radius, width: W, height: H };
  let mismatch = 0;
  for (const [px, py] of [[0, 0], [12, 12], [23, 5], [5, 20], [18, 18]]) {
    const { dcx, dcy } = pixelDelta(view, px, py);
    const ex = fromDecimalString(cxStr, prec) + fromDouble(dcx, prec);
    const ey = fromDecimalString(cyStr, prec) + fromDouble(dcy, prec);
    const truth = escapeBigInt(ex, ey, prec, maxIter);
    if (Math.abs(out.iters[py * W + px] - truth) > 1) mismatch++;
  }
  assert.equal(mismatch, 0, `${mismatch} deep pixels disagree with BigInt`);
});
