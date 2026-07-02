import { test } from 'node:test';
import assert from 'node:assert/strict';
import { escapeNaive, orbitNaive } from '../../src/math/naive.js';

test('origin is inside the set (never escapes)', () => {
  const { n } = escapeNaive(0, 0, 1000);
  assert.equal(n, 1000);
});

test('main cardioid point -0.5+0i is inside', () => {
  assert.equal(escapeNaive(-0.5, 0, 5000).n, 5000);
});

test('period-2 bulb center -1+0i is inside', () => {
  assert.equal(escapeNaive(-1, 0, 5000).n, 5000);
});

test('far exterior point escapes fast (bailout radius 256)', () => {
  // With a smooth-coloring bailout of |z|^2 > 65536 it takes a few iterations
  // even for far points; what matters is it escapes well before maxIter.
  const { n } = escapeNaive(2, 2, 1000);
  assert.ok(n > 0 && n <= 8, `escaped at ${n}`);
});

test('c=1+0i escapes fast and gives finite smooth count', () => {
  const { n, sn } = escapeNaive(1, 0, 1000);
  assert.ok(n > 0 && n < 10);
  assert.ok(Number.isFinite(sn));
});

test('orbitNaive of c=0 stays at 0', () => {
  const { zx, zy, len } = orbitNaive(0, 0, 50);
  assert.equal(len, 50);
  for (let i = 0; i <= 50; i++) { assert.equal(zx[i], 0); assert.equal(zy[i], 0); }
});

test('orbitNaive of c=-1 cycles 0,-1,0,-1', () => {
  const { zx } = orbitNaive(-1, 0, 10);
  assert.equal(zx[0], 0);
  assert.equal(zx[1], -1);
  assert.equal(zx[2], 0);
  assert.equal(zx[3], -1);
});

test('smooth count is monotone-ish: deeper escapes have larger n', () => {
  const a = escapeNaive(0.30, 0.0, 5000).n; // closer to set, escapes later
  const b = escapeNaive(0.40, 0.0, 5000).n; // farther, escapes sooner
  assert.ok(a > b, `${a} vs ${b}`);
});
