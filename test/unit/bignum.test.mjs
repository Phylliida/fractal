import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  mulShift, toDouble, fromDouble, fromDecimalString, toDecimalString, precForRadius,
} from '../../src/math/bignum.js';

test('mulShift: 1.5 * 2.0 = 3.0 at prec 32', () => {
  const prec = 32;
  const a = fromDouble(1.5, prec);
  const b = fromDouble(2.0, prec);
  assert.equal(toDouble(mulShift(a, b, prec), prec), 3.0);
});

test('mulShift: handles negatives, 0.25 * -8 = -2', () => {
  const prec = 40;
  const a = fromDouble(0.25, prec);
  const b = fromDouble(-8, prec);
  assert.equal(toDouble(mulShift(a, b, prec), prec), -2.0);
});

test('fromDouble/toDouble round-trip across magnitudes', () => {
  const prec = 120;
  for (const v of [0, 1, -1, 0.5, -0.333333333333, 1.7976931348623157, 3.141592653589793, -2.5e-10]) {
    const r = toDouble(fromDouble(v, prec), prec);
    if (v === 0) { assert.equal(r, 0); continue; }
    assert.ok(Math.abs(r - v) <= Math.abs(v) * 1e-15 + 1e-300, `${v} -> ${r}`);
  }
});

test('fromDecimalString matches known fraction', () => {
  const prec = 200;
  // 0.5 exactly
  assert.equal(toDouble(fromDecimalString('0.5', prec), prec), 0.5);
  // -0.75 exactly
  assert.equal(toDouble(fromDecimalString('-0.75', prec), prec), -0.75);
  // pi to double accuracy
  const pi = fromDecimalString('3.14159265358979311599796346854418516159057617187500', prec);
  assert.ok(Math.abs(toDouble(pi, prec) - Math.PI) < 1e-15);
});

test('toDecimalString round-trips a deep coordinate to many digits', () => {
  const prec = 512;
  const s = '-0.74364388703715874252191506114774127287531851176856';
  const m = fromDecimalString(s, prec);
  const back = toDecimalString(m, prec, 50);
  assert.equal(back, s);
});

test('fromDecimalString handles exponent notation', () => {
  const prec = 200;
  const a = fromDecimalString('1.25e-3', prec);
  assert.ok(Math.abs(toDouble(a, prec) - 0.00125) < 1e-18);
});

test('toDouble does not overflow at very large prec', () => {
  const prec = 1500; // ~2^1400 zoom
  const m = fromDecimalString('0.3601', prec);
  const d = toDouble(m, prec);
  assert.ok(Math.abs(d - 0.3601) < 1e-12, `got ${d}`);
});

test('precForRadius scales with zoom depth', () => {
  assert.ok(precForRadius(1) >= 64);
  assert.ok(precForRadius(Math.pow(2, -400)) >= 400 + 64);
  assert.ok(precForRadius(Math.pow(2, -100)) >= 100 + 64);
});
