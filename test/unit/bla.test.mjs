import { test } from 'node:test';
import assert from 'node:assert/strict';
import { renderImage } from '../../src/math/render.js';
import { computeReference } from '../../src/math/reference.js';
import { buildBLA, blaToFloat32 } from '../../src/math/bla.js';
import { fromDecimalString, precForRadius } from '../../src/math/bignum.js';

// BLA (bivariate linear approximation, src/math/bla.js) skips RUNS of linear perturbation
// iterations throughout the orbit — including the post-rebase re-growth phases that series
// approximation cannot touch. Unlike SA it is an APPROXIMATION (it drops the dz² term), so
// the gate is: (a) it must not change escape counts beyond a small truncation drift, and a
// missed escape (BLA stepping past a real escape) would flip pixels by HUNDREDS — that must
// never happen; (b) it must actually reduce the work. See tools/crosscheck-bla.mjs (the
// across-depth measurement) and tools/arbiter-bla.mjs (BigInt classification of the drift).
const cxStr = '-0.743643887037158704752191506114774';
const cyStr = '0.131825904205311970493132056385139';

function view(radius, W, H) {
  const prec = precForRadius(radius, 96);
  return { x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec, radius, width: W, height: H };
}

test('BLA: escape counts track the no-BLA oracle and cut the work substantially', () => {
  const radius = 1e-30, maxIter = 20000, W = 28, H = 28;   // ~2^-100 deep field
  const v = view(radius, W, H);
  const sBase = { steps: 0, jumps: 0, jumpIters: 0 };
  const off = renderImage(v, maxIter, { stats: sBase });            // no BLA — the oracle
  const sBla = { steps: 0, jumps: 0, jumpIters: 0 };
  const on = renderImage(v, maxIter, { bla: true, stats: sBla });   // BLA, default eps

  // Correctness: at this depth the default eps is bit-exact; tolerate at most a few ±small
  // truncation flips, and HARD-cap maxΔ so a missed escape (off by hundreds) fails loudly.
  let mism = 0, maxd = 0;
  for (let i = 0; i < off.iters.length; i++) {
    const d = Math.abs(off.iters[i] - on.iters[i]);
    if (d) { mism++; if (d > maxd) maxd = d; }
  }
  assert.ok(mism <= Math.ceil(0.01 * off.iters.length), `${mism}/${off.iters.length} pixels differ — too many for BLA truncation (a missed escape flips a whole region)`);
  assert.ok(maxd <= 50, `maxΔ ${maxd} too large — looks like a skipped escape, not truncation drift`);
  assert.equal(on.glitches, 0);

  // Work reduction: BLA must take real jumps and cut the total work well below the no-BLA
  // step count (the whole point — this bounds the eventual GPU-escape speedup).
  assert.ok(sBla.jumps > 0, 'BLA took no jumps');
  const baseWork = sBase.steps + sBase.jumps;
  const blaWork = sBla.steps + sBla.jumps;
  assert.ok(blaWork * 2 < baseWork, `BLA work ${blaWork} not < half of baseline ${baseWork}`);
});

test('BLA: composes with series approximation (SA seed + BLA jumps)', () => {
  const radius = 1e-30, maxIter = 20000, W = 24, H = 24;
  const v = view(radius, W, H);
  const off = renderImage(v, maxIter);                                   // oracle
  const sBoth = { steps: 0, jumps: 0, jumpIters: 0 };
  const both = renderImage(v, maxIter, { series: true, bla: true, stats: sBoth });
  assert.ok(both.saSkip > 0 && both.blaLevels > 0, 'SA+BLA both engaged');
  let maxd = 0;
  for (let i = 0; i < off.iters.length; i++) maxd = Math.max(maxd, Math.abs(off.iters[i] - both.iters[i]));
  assert.ok(maxd <= 50, `SA+BLA maxΔ ${maxd} — a missed escape, not truncation`);
});

test('BLA: table is well-formed (finite coeffs, non-negative radii, ZMAX guard)', () => {
  const radius = 1e-30, maxIter = 4000;
  const prec = precForRadius(radius, 96);
  const ref = computeReference({ x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec }, maxIter);
  const bla = buildBLA(ref, radius * 1.5, { eps: Math.pow(2, -30) });
  assert.ok(bla.maxLevel >= 1 && bla.levels.length === bla.maxLevel + 1);
  for (const lv of bla.levels) {
    for (let m = 0; m < lv.r2.length; m++) {
      assert.ok(lv.r2[m] >= 0 && isFinite(lv.r2[m]), 'radius² must be finite and ≥ 0');
      // A usable BLA (r2>0) must have finite coefficients — an overflowed run is forced to r2=0.
      if (lv.r2[m] > 0) {
        assert.ok(isFinite(lv.Ax[m]) && isFinite(lv.Ay[m]) && isFinite(lv.Bx[m]) && isFinite(lv.By[m]),
          'a usable BLA must have finite A,B');
      }
    }
  }
  // Level-0 radius is 0 exactly where |Z_m| ≥ ZMAX (=2) — the escape-safety guard.
  const lv0 = bla.levels[0];
  for (let m = 0; m < lv0.r2.length; m++) {
    const z2 = ref.zx[m] * ref.zx[m] + ref.zy[m] * ref.zy[m];
    if (z2 >= 4) assert.equal(lv0.r2[m], 0, `level-0 radius must be 0 where |Z|≥2 (m=${m})`);
  }
});

// Decode one packed entry the way the GPU shader does: value = (hi+lo)·2^exp, with A,B sharing
// one exponent per complex coeff and r² a single-float mantissa·2^r2e (r2m==0 ⇒ unusable).
function decodeEntry(packed, l, m) {
  const d = packed.data, t0 = ((l - 1) * packed.len + m) * 3;
  const o0 = t0 * 4, o1 = (t0 + 1) * 4, o2 = (t0 + 2) * 4;
  const Ae = d[o2], Be = d[o2 + 1], r2e = d[o2 + 2], r2m = d[o2 + 3];
  const pa = 2 ** Ae, pb = 2 ** Be;
  return { Ax: (d[o0] + d[o0 + 1]) * pa, Ay: (d[o0 + 2] + d[o0 + 3]) * pa,
           Bx: (d[o1] + d[o1 + 1]) * pb, By: (d[o1 + 2] + d[o1 + 3]) * pb,
           r2: r2m === 0 ? 0 : r2m * 2 ** r2e, Ae, Be, r2e };
}
const rel = (a, b) => Math.abs(a - b) / Math.max(Math.abs(b), 1e-300);

test('BLA→float32: packed texels round-trip A,B,r² to df64 (~46-bit) precision', () => {
  const radius = 1e-30, maxIter = 6000;   // ~2^-100
  const prec = precForRadius(radius, 96);
  const ref = computeReference({ x: fromDecimalString(cxStr, prec), y: fromDecimalString(cyStr, prec), prec }, maxIter);
  const bla = buildBLA(ref, radius * 1.5, { eps: 2 ** -30 });
  const packed = blaToFloat32(bla);
  assert.equal(packed.maxLevel, bla.maxLevel);
  assert.equal(packed.len, bla.len);
  // texture must hold every level-1..maxLevel entry (3 texels each, RGBA)
  assert.ok(packed.data.length >= bla.maxLevel * bla.len * 3 * 4);

  let checked = 0;
  for (let l = 1; l <= bla.maxLevel; l++) {
    const lv = bla.levels[l], valid = bla.len - (1 << l) + 1;
    const stride = Math.max(1, Math.floor(valid / 64));     // sample ~64 entries/level
    for (let m = 0; m < valid; m += stride) {
      const got = decodeEntry(packed, l, m);
      // r²: exact mapping of usability + ~24-bit mantissa on the value.
      if (lv.r2[m] === 0) assert.equal(got.r2, 0, `r²=0 entry must pack as unusable (l${l} m${m})`);
      else assert.ok(rel(got.r2, lv.r2[m]) < 1e-6, `r² mismatch l${l} m${m}: ${got.r2} vs ${lv.r2[m]}`);
      // A,B: ~46-bit df64 mantissa, so ~1e-13 relative. Only meaningful for finite coeffs.
      if (isFinite(lv.Ax[m]) && isFinite(lv.Ay[m])) {
        assert.ok(rel(got.Ax, lv.Ax[m]) < 1e-11 && rel(got.Ay, lv.Ay[m]) < 1e-11, `A mismatch l${l} m${m}`);
      }
      if (isFinite(lv.Bx[m]) && isFinite(lv.By[m])) {
        assert.ok(rel(got.Bx, lv.Bx[m]) < 1e-11 && rel(got.By, lv.By[m]) < 1e-11, `B mismatch l${l} m${m}`);
      }
      checked++;
    }
  }
  assert.ok(checked > 50, 'sampled too few entries');
});

test('BLA→float32: floatexp packing survives exponents OUTSIDE the float32 range (deep)', () => {
  // The deep boundary coordinate at ~2^-271: applied BLA coeffs reach |A|~2^120+, r²~2^-360 —
  // far outside float32 (±127). Plain df64/float would overflow/underflow; floatexp must not.
  const DRE = '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
  const DIM = '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
  const radius = 1.5 * 2 ** -271, maxIter = 68150;
  const prec = precForRadius(radius, 80);
  const ref = computeReference({ x: fromDecimalString(DRE, prec), y: fromDecimalString(DIM, prec), prec }, maxIter);
  const bla = buildBLA(ref, Math.hypot(radius * (1), radius), { eps: 2 ** -30 });
  const packed = blaToFloat32(bla);

  let sawBigExp = false, sawTinyR = false, checked = 0, worstA = 0, worstR = 0;
  for (let l = 1; l <= bla.maxLevel; l++) {
    const lv = bla.levels[l], valid = bla.len - (1 << l) + 1;
    const stride = Math.max(1, Math.floor(valid / 200));
    for (let m = 0; m < valid; m += stride) {
      if (!(lv.r2[m] > 0) || !isFinite(lv.Ax[m]) || !isFinite(lv.Ay[m])) continue;
      const got = decodeEntry(packed, l, m);
      if (Math.abs(got.Ae) > 127) sawBigExp = true;
      if (got.r2e < -126) sawTinyR = true;
      worstA = Math.max(worstA, rel(got.Ax, lv.Ax[m]), rel(got.Ay, lv.Ay[m]));
      worstR = Math.max(worstR, rel(got.r2, lv.r2[m]));
      checked++;
    }
  }
  assert.ok(sawBigExp, 'expected some |A| exponent outside float32 range (the reason for floatexp)');
  assert.ok(sawTinyR, 'expected some r² exponent below the float32 floor (2^-126)');
  assert.ok(worstA < 1e-11, `A round-trip degraded deep: worst rel ${worstA}`);
  assert.ok(worstR < 1e-6, `r² round-trip degraded deep: worst rel ${worstR}`);
  assert.ok(checked > 100, `sampled too few finite entries (${checked})`);
});
