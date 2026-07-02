// arbiter-bla.mjs — classify BLA-vs-oracle escape-count mismatches with the BigInt-exact
// oracle. For each pixel where the BLA render differs from the no-BLA f64 render, compute
// the EXACT count (escapeBigInt at high precision) and report whether BLA matches BigInt,
// the f64 oracle matches BigInt, or neither. A correct BLA differs from the f64 oracle only
// on measure-zero ILL-CONDITIONED pixels where the oracle is ALSO wrong (the project's
// philosophy — see NOTES "validate vs BigInt, not naive / bulk metrics, not max").
//
//   BITS=400 EPS=30 node tools/arbiter-bla.mjs
import { fromDecimalString, fromDouble, precForRadius } from '../src/math/bignum.js';
import { escapeBigInt } from '../src/math/reference.js';
import { renderImage } from '../src/math/render.js';

const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';
const bits = Number(process.env.BITS || 400);
const eps = 2 ** -Number(process.env.EPS || 30);
const W = Number(process.env.W || 48), H = Number(process.env.H || 48);
const MAXPX = Number(process.env.MAXPX || 12);   // arbitrate at most this many worst pixels
const autoIter = (b) => Math.min(2_000_000, Math.round(400 + b * 250));

const radius = 1.5 * 2 ** -bits;
const maxIter = autoIter(bits);
const prec = precForRadius(radius, 80);
const arbPrec = precForRadius(radius, 200);   // generous guard for the exact oracle
const cx = fromDecimalString(RE, prec), cy = fromDecimalString(IM, prec);
const cxA = fromDecimalString(RE, arbPrec), cyA = fromDecimalString(IM, arbPrec);
const view = { x: cx, y: cy, prec, radius, width: W, height: H };

console.log(`arbiter-bla — 2^-${bits}, ${W}×${H}, blaEps 2^${Math.round(Math.log2(eps))}, maxIter ${maxIter}\n`);
const oracle = renderImage(view, maxIter, {});
const blaR = renderImage(view, maxIter, { bla: { eps } });

// pixel -> absolute coordinate (view-center-relative dc, in fixed point at arbPrec)
const aspect = W / H, scale = (2 * radius) / H;
const offX = -radius * aspect, offY = -radius;

const diffs = [];
for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) {
  const idx = j * W + i;
  const d = Math.abs(oracle.iters[idx] - blaR.iters[idx]);
  if (d > 0) diffs.push({ i, j, idx, d, o: oracle.iters[idx], b: blaR.iters[idx] });
}
diffs.sort((a, b) => b.d - a.d);
console.log(`total mismatched pixels: ${diffs.length} (arbitrating worst ${Math.min(MAXPX, diffs.length)})\n`);

let blaRight = 0, oracleRight = 0, neither = 0;
for (const px of diffs.slice(0, MAXPX)) {
  const pcx = cxA + fromDouble(offX + px.i * scale, arbPrec);
  const pcy = cyA + fromDouble(offY + px.j * scale, arbPrec);
  const big = escapeBigInt(pcx, pcy, arbPrec, maxIter);
  const bMatch = big === px.b, oMatch = big === px.o;
  if (bMatch) blaRight++; if (oMatch) oracleRight++; if (!bMatch && !oMatch) neither++;
  console.log(`  (${px.i},${px.j})  oracle ${px.o}  BLA ${px.b}  BigInt ${big}   ` +
    `${bMatch ? 'BLA✓' : ''}${oMatch ? ' oracle✓' : ''}${!bMatch && !oMatch ? ' both-wrong(chaos)' : ''}`);
}
console.log(`\nsummary: BLA matches BigInt ${blaRight}, oracle matches BigInt ${oracleRight}, ` +
  `neither ${neither}  (of ${Math.min(MAXPX, diffs.length)} arbitrated)`);
console.log('If "oracle✓" dominates the mismatches, BLA is degrading correct pixels (a real bug).');
console.log('If the mismatches split / are "both-wrong", they are ill-conditioned — BLA is sound.');
