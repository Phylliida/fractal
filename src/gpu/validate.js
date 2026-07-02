// validate.js — compare GPU renders against the CPU oracles, in-page (so big
// pixel buffers never cross the CDP boundary; only summary stats are returned).
// Used by tools/validate-gpu.mjs and the Playwright GPU spec.
//
// The oracle is fed the SAME (ox, oy, scale) doubles the GPU received, and
// computes each pixel's c in full float64. Mismatches therefore reflect the GPU
// path's TOTAL precision loss (coordinate rounding + iteration), which is exactly
// what we use to choose the float32 / df64 / perturb depth thresholds.

import { escapeNaive } from '../math/naive.js';
import { computeReference, escapeBigInt } from '../math/reference.js';
import { escapePerturb } from '../math/perturb.js';
import { fromDouble } from '../math/bignum.js';

// c for the texel read back at (i, j): the shader used gl_FragCoord = texel+0.5,
// and readPixels row 0 is the GL-bottom row, so this matches the GPU exactly.
function texelC(p, i, j) {
  return { cx: p.ox + (i + 0.5) * p.scale, cy: p.oy + (j + 0.5) * p.scale };
}

export function compareNaive(gpu, p) {
  gpu.renderNaive(p);
  const { sn, iter, w, h } = gpu.readSn();
  const step = p.checkStep || 1;
  let compared = 0, mism = 0, insideMismatch = 0, maxAbs = 0, sumAbs = 0, snCount = 0;
  const examples = [];
  for (let j = 0; j < h; j += step) {
    for (let i = 0; i < w; i += step) {
      const { cx, cy } = texelC(p, i, j);
      const o = escapeNaive(cx, cy, p.maxIter);
      const o4 = j * w + i;
      const gi = iter[o4], gsn = sn[o4];
      const cpuInside = o.n >= p.maxIter, gpuInside = gsn < 0;
      compared++;
      if (cpuInside !== gpuInside) {
        insideMismatch++;
        if (examples.length < 10) examples.push({ i, j, cx, cy, cpu: o.n, gpu: gi, kind: 'inside' });
        continue;
      }
      if (cpuInside) continue;
      const d = Math.abs(gi - o.n);
      if (d > 0) { mism++; if (examples.length < 10) examples.push({ i, j, cx, cy, cpu: o.n, gpu: gi }); }
      const sd = Math.abs(gsn - o.sn);
      if (isFinite(sd)) { if (sd > maxAbs) maxAbs = sd; sumAbs += sd; snCount++; }
    }
  }
  return { engine: p.df64 ? 'naive64' : 'naive32', compared, mism, insideMismatch,
           maxAbs, meanAbs: snCount ? sumAbs / snCount : 0, examples, w, h };
}

// Arbiter: for GPU-df64-vs-CPU-double mismatches, ask the BigInt-exact oracle
// who is right. If CPU double is ALSO wrong vs BigInt at those pixels, the
// disagreement is boundary chaos (finite precision on a sensitive pixel), not a
// GPU bug. Returns counts: how many mismatches had double-also-wrong vs
// gpu-uniquely-wrong (the only worrying category).
export function arbitrateNaive(gpu, p) {
  gpu.renderNaive(p);
  const { sn, iter, w, h } = gpu.readSn();
  const prec = p.arbPrec || 96;
  const step = p.checkStep || 1;
  let mism = 0, chaosBoth = 0, gpuMatchesBig = 0, doubleMatchesBig = 0, gpuUniquelyWrong = 0;
  const examples = [];
  for (let j = 0; j < h; j += step) {
    for (let i = 0; i < w; i += step) {
      const cx = p.ox + (i + 0.5) * p.scale, cy = p.oy + (j + 0.5) * p.scale;
      const dbl = escapeNaive(cx, cy, p.maxIter);
      const o4 = j * w + i;
      const gi = iter[o4], gsn = sn[o4];
      const dblInside = dbl.n >= p.maxIter, gpuInside = gsn < 0;
      const gpuN = gpuInside ? p.maxIter : gi;
      const dblN = dblInside ? p.maxIter : dbl.n;
      if (gpuN === dblN) continue;            // agree -> nothing to arbitrate
      mism++;
      const bx = fromDouble(cx, prec), by = fromDouble(cy, prec);
      const big = escapeBigInt(bx, by, prec, p.maxIter);   // exact
      const gpuRight = gpuN === big, dblRight = dblN === big;
      if (gpuRight) gpuMatchesBig++;
      if (dblRight) doubleMatchesBig++;
      if (!gpuRight && !dblRight) chaosBoth++;
      if (gpuRight && !dblRight) { /* GPU actually better */ }
      if (!gpuRight && dblRight) {
        gpuUniquelyWrong++;
        if (examples.length < 12) examples.push({ i, j, cx, cy, gpu: gpuN, dbl: dblN, big });
      }
    }
  }
  return { mism, chaosBoth, gpuMatchesBig, doubleMatchesBig, gpuUniquelyWrong, examples, w, h };
}

// Compare GPU perturbation against the CPU perturbation oracle (same reference).
// p: { zx, zy, refLen, ox, oy, scale, maxIter, bailoutSq, width, height, checkStep }
export function comparePerturb(gpu, p) {
  const args = { ox: p.ox, oy: p.oy, scale: p.scale, refLen: p.refLen,
                 maxIter: p.maxIter, bailoutSq: p.bailoutSq, glitchTol: p.glitchTol ?? 0,
                 width: p.width, height: p.height, df64Esc: p.df64Esc, sa: p.sa || null, bla: !!p.bla };
  if (p.rs) { gpu.uploadReferenceDf64(p.zx, p.zy, p.refW || 2048); gpu.renderPerturbRescaled(args); }
  else if (p.fe) { gpu.uploadReferenceDf64(p.zx, p.zy, p.refW || 2048); gpu.renderPerturbFloatexp(args); }
  else if (p.df64) { gpu.uploadReferenceDf64(p.zx, p.zy, p.refW || 2048); gpu.renderPerturbDf64(args); }
  else { gpu.uploadReference(p.zx, p.zy, p.refW || 2048); gpu.renderPerturb(args); }
  const { sn, iter, w, h } = gpu.readSn();
  const ref = { zx: p.zx, zy: p.zy, z2: p.z2, len: p.refLen };
  const step = p.checkStep || 1;
  let compared = 0, mism = 0, insideMismatch = 0, maxAbs = 0, sumAbs = 0, snCount = 0, near = 0, interiorCount = 0;
  const examples = [];
  for (let j = 0; j < h; j += step) {
    for (let i = 0; i < w; i += step) {
      const dcx = p.ox + (i + 0.5) * p.scale;
      const dcy = p.oy + (j + 0.5) * p.scale;
      // The oracle is the no-SA/no-BLA escape count (the true value) UNLESS oracleSa/oracleBla
      // are supplied — then it mirrors the GPU's own SA/BLA so the comparison isolates the
      // GPU-vs-CPU df64 precision gap (the "faithfulness" gate) rather than BLA's own drift.
      const o = escapePerturb(ref, dcx, dcy, p.maxIter, p.bailoutSq ?? (1 << 16), 0,
                              p.oracleSa || null, p.oracleBla || null);
      const o4 = j * w + i;
      const gi = iter[o4], gsn = sn[o4];
      const cpuInside = o.n >= p.maxIter, gpuInside = gsn < 0;
      compared++;
      if (cpuInside !== gpuInside) {
        insideMismatch++;
        if (examples.length < 10) examples.push({ i, j, dcx, dcy, cpu: o.n, gpu: gi, kind: 'inside' });
        continue;
      }
      if (cpuInside) { interiorCount++; continue; }
      const d = Math.abs(gi - o.n);
      if (d > 0) { mism++; if (d <= 2) near++; if (examples.length < 10) examples.push({ i, j, dcx, dcy, cpu: o.n, gpu: gi }); }
      const sd = Math.abs(gsn - o.sn);
      if (isFinite(sd)) { if (sd > maxAbs) maxAbs = sd; sumAbs += sd; snCount++; }
    }
  }
  return { engine: 'perturb', compared, mism, near, insideMismatch, interiorCount, maxAbs,
           meanAbs: snCount ? sumAbs / snCount : 0, examples, w, h };
}

export { computeReference };
