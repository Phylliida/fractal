// refextend.test.mjs — extendReference (Spawn 30, reference-reuse cache) must be
// BIT-IDENTICAL to an uninterrupted computeReference run: both drive the same
// iterateOrbit core, and resuming from the exact BigInt tail state reproduces the
// identical mulShift sequence. This is the correctness foundation of the viewer's
// reference cache (a cached orbit + extension == the orbit a fresh render builds).
import test from 'node:test';
import assert from 'node:assert/strict';
import { computeReference, extendReference } from '../../src/math/reference.js';
import { fromDecimalString } from '../../src/math/bignum.js';

const RE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

function merged(center, k, m) {
  // build to k, extend k -> m, merge — the cache's exact code path
  const a = computeReference(center, k);
  assert.equal(a.len, k, 'first segment must not escape in this test');
  const b = extendReference(center, { bx: a.tailX, by: a.tailY, pm2: a.z2[a.len] }, a.len, m);
  const zx = new Float64Array(b.len + 1), zy = new Float64Array(b.len + 1), z2 = new Float64Array(b.len + 1);
  zx.set(a.zx); zy.set(a.zy); z2.set(a.z2);
  zx.set(b.zx, a.len + 1); zy.set(b.zy, a.len + 1); z2.set(b.z2, a.len + 1);
  return { zx, zy, z2, len: b.len, escaped: b.escaped, tailX: b.tailX, tailY: b.tailY };
}

test('extendReference: resume == uninterrupted, deep boundary center (2^-120 prec)', () => {
  const prec = 120 + 64;
  const center = { x: fromDecimalString(RE, prec), y: fromDecimalString(IM, prec), prec };
  const M = 5000, K = 1700;
  const full = computeReference(center, M);
  const two = merged(center, K, M);
  assert.equal(two.len, full.len);
  assert.equal(two.escaped, full.escaped);
  assert.equal(two.tailX, full.tailX);       // BigInt state identical -> further
  assert.equal(two.tailY, full.tailY);       // extensions stay identical too
  for (let n = 0; n <= full.len; n++) {
    assert.equal(two.zx[n], full.zx[n], `zx[${n}]`);
    assert.equal(two.zy[n], full.zy[n], `zy[${n}]`);
    assert.equal(two.z2[n], full.z2[n], `z2[${n}]`);
  }
});

test('extendReference: chained double extension == uninterrupted', () => {
  const prec = 200;
  const center = { x: fromDecimalString(RE, prec), y: fromDecimalString(IM, prec), prec };
  const full = computeReference(center, 4000);
  const a = computeReference(center, 1000);
  const b = extendReference(center, { bx: a.tailX, by: a.tailY, pm2: a.z2[a.len] }, a.len, 2500);
  const c = extendReference(center, { bx: b.tailX, by: b.tailY, pm2: b.z2[b.len - a.len - 1] }, b.len, 4000);
  assert.equal(c.len, full.len);
  assert.equal(c.tailX, full.tailX);
  assert.equal(c.tailY, full.tailY);
  // spot-check the last segment's doubles against the full run
  for (let n = b.len + 1; n <= full.len; n++) {
    assert.equal(c.zx[n - b.len - 1], full.zx[n], `zx[${n}]`);
    assert.equal(c.zy[n - b.len - 1], full.zy[n], `zy[${n}]`);
  }
});

test('extendReference: escape mid-extension matches the uninterrupted escape', () => {
  // a center OUTSIDE the set (escapes quickly): resume across the escape boundary
  const prec = 96;
  const center = { x: fromDecimalString('0.26', prec), y: fromDecimalString('0.0', prec), prec };
  const full = computeReference(center, 1000);
  assert.equal(full.escaped, true, 'test point must escape');
  const K = Math.floor(full.len / 2);
  const two = merged(center, K, 1000);
  assert.equal(two.len, full.len);
  assert.equal(two.escaped, true);
  for (let n = 0; n <= full.len; n++) {
    assert.equal(two.zx[n], full.zx[n], `zx[${n}]`);
  }
});
