// probe-bla-mag.mjs — MEASURE the magnitude distribution of the BLA table coefficients
// (A, B) and validity radii (r²) at deep zoom, to decide the GPU texture format.
//
// The NOTES "GPU PORT" plan tentatively proposed storing A,B as plain df64 (float32
// hi/lo) + r² as a single float32. But float32 has min normal 2^-126 / max ~2^127, and
// at deep zoom dc,dz ~ 2^-400 while a USABLE high-level BLA run can have |A|,|B| up to
// ~2^240 (the radius merge tolerates |B|·dcMax < r_y ~ 2^-30, so |B| < 2^-30/2^-400 =
// 2^370). If A/B/r² exceed the float32 exponent range, plain df64/float storage would
// overflow/underflow and the GPU BLA would be wrong. This probe answers: do we need
// FLOATEXP (df64 mantissa + int exponent) for A, B, r², or is plain df64 enough?
//
// It reports, over the entries that are actually APPLICABLE (r² > 0) AND over the jumps
// that an actual deep render APPLIES, the exponent range of |A|, |B|, r (= sqrt r²).
//
//   node tools/probe-bla-mag.mjs
//   BITS=120,271,400 EPS=30 node tools/probe-bla-mag.mjs
import { fromDecimalString, precForRadius } from '../src/math/bignum.js';
import { chooseReference } from '../src/math/render.js';
import { buildBLA } from '../src/math/bla.js';
import { toDouble } from '../src/math/bignum.js';

const RE = process.env.RE || '-1.3690780185200794247564219525584239050789875505480606319269707706679171351435598418313845872694681181796577691164703094161056071235175896190526041326788181608562069022572';
const IM = process.env.IM || '-0.0718176768448516592331078176393659885285102748352266772001772456485196902477913914772927891534615241257659389266050933958039395896106078019328906531433514534019106897806';

const BITS = (process.env.BITS || '120,271,400').split(',').map(Number);
const EPS = 2 ** -Number(process.env.EPS || 30);
const W = Number(process.env.W || 48), H = Number(process.env.H || 48);
const autoIter = (bits) => Math.min(2_000_000, Math.round(400 + bits * 250));

// frexp exponent: the k with |x|·2^-k in [0.5,1). 0 -> -Infinity sentinel.
function ilog2(x) { return x === 0 ? null : Math.floor(Math.log2(Math.abs(x))) + 1; }

function range() { return { min: Infinity, max: -Infinity, n: 0 }; }
function note(r, e) { if (e === null) return; if (e < r.min) r.min = e; if (e > r.max) r.max = e; r.n++; }
function fmt(r) { return r.n ? `[2^${r.min}, 2^${r.max}] (${r.n})` : '(none)'; }

for (const bits of BITS) {
  const radius = 1.5 * 2 ** -bits;
  const maxIter = autoIter(bits);
  const prec = precForRadius(radius, 80);
  const x = fromDecimalString(RE, prec), y = fromDecimalString(IM, prec);
  const viewHP = { x, y, prec, radius, width: W, height: H };
  const sel = chooseReference(viewHP, maxIter);
  const ref = sel.ref, center = sel.center;
  const aspect = W / H, scale = (2 * radius) / H;
  const refOffX = toDouble(center.x - x, prec), refOffY = toDouble(center.y - y, prec);
  const offX = -radius * aspect - refOffX, offY = -radius - refOffY;
  const dcxFar = offX + (W - 1) * scale, dcyFar = offY + (H - 1) * scale;
  const dcHalfX = Math.max(Math.abs(offX), Math.abs(dcxFar));
  const dcHalfY = Math.max(Math.abs(offY), Math.abs(dcyFar));
  const dcMax = Math.hypot(dcHalfX, dcHalfY);

  const bla = buildBLA(ref, dcMax, { eps: EPS });

  // (1) over ALL usable table entries (r² > 0): exponent ranges of |A|, |B|, r.
  const Aall = range(), Ball = range(), Rall = range();
  let usable = 0, total = 0;
  for (let l = 0; l <= bla.maxLevel; l++) {
    const lv = bla.levels[l];
    for (let m = 0; m < lv.r2.length; m++) {
      total++;
      if (!(lv.r2[m] > 0)) continue;
      usable++;
      note(Aall, ilog2(Math.hypot(lv.Ax[m], lv.Ay[m])));
      note(Ball, ilog2(Math.hypot(lv.Bx[m], lv.By[m])));
      note(Rall, ilog2(Math.sqrt(lv.r2[m])));
    }
  }

  // (2) over the jumps an ACTUAL render applies: replay the BLA stepping for every
  // pixel and log each applied (level,m)'s |A|,|B|,r exponents + the dz magnitude.
  const Aapp = range(), Bapp = range(), Rapp = range(), Dzapp = range();
  const levelHist = new Array(bla.maxLevel + 1).fill(0);
  const { zx, zy, z2, len } = ref;
  for (let j = 0; j < H; j++) {
    const dcy = offY + j * scale;
    for (let i = 0; i < W; i++) {
      const dcx = offX + i * scale;
      let dx = 0, dy = 0, m = 0, n = 0;
      while (n < maxIter) {
        const mag2 = dx * dx + dy * dy;
        let jumped = false;
        for (let l = bla.maxLevel; l >= 1; l--) {
          const L = 1 << l;
          if (m + L > len - 1 || n + L > maxIter) continue;
          const lv = bla.levels[l];
          if (m >= lv.r2.length || lv.r2[m] < mag2 || lv.r2[m] === 0) continue;
          note(Aapp, ilog2(Math.hypot(lv.Ax[m], lv.Ay[m])));
          note(Bapp, ilog2(Math.hypot(lv.Bx[m], lv.By[m])));
          note(Rapp, ilog2(Math.sqrt(lv.r2[m])));
          note(Dzapp, ilog2(Math.sqrt(mag2)));
          levelHist[l]++;
          const Ax = lv.Ax[m], Ay = lv.Ay[m], Bx = lv.Bx[m], By = lv.By[m];
          const ndx = Ax * dx - Ay * dy + Bx * dcx - By * dcy;
          const ndy = Ax * dy + Ay * dx + Bx * dcy + By * dcx;
          dx = ndx; dy = ndy; m += L; n += L; jumped = true; break;
        }
        if (jumped) continue;
        const Zx = zx[m], Zy = zy[m];
        const ndx = 2 * (Zx * dx - Zy * dy) + (dx * dx - dy * dy) + dcx;
        const ndy = 2 * (Zx * dy + Zy * dx) + (2 * dx * dy) + dcy;
        dx = ndx; dy = ndy; m++; n++;
        const zfx = zx[m] + dx, zfy = zy[m] + dy;
        const mag2z = zfx * zfx + zfy * zfy;
        if (mag2z > (1 << 16)) break;
        const dz2 = dx * dx + dy * dy;
        if (mag2z < dz2 || m === len) { dx = zfx; dy = zfy; m = 0; }
      }
    }
  }

  console.log(`\n=== 2^-${bits}  maxIter ${maxIter}  refLen ${len}  maxLevel ${bla.maxLevel}  eps 2^${Math.round(Math.log2(EPS))} ===`);
  console.log(`  dcMax 2^${ilog2(dcMax)}   usable entries ${usable}/${total}`);
  console.log(`  ALL usable:  |A| ${fmt(Aall)}   |B| ${fmt(Ball)}   r ${fmt(Rall)}`);
  console.log(`  APPLIED:     |A| ${fmt(Aapp)}   |B| ${fmt(Bapp)}   r ${fmt(Rapp)}   |dz| ${fmt(Dzapp)}`);
  console.log(`  level histogram (applied jumps): ${levelHist.map((c, l) => c ? `L${l}:${c}` : '').filter(Boolean).join(' ')}`);
  // float32 range check: exponent must be within ~[-126, 127] to store as plain df64/float.
  const needFE = (r) => r.n && (r.min < -120 || r.max > 120);
  const verdict = [['|A|', Aall], ['|B|', Ball], ['r', Rall]].filter(([, r]) => needFE(r)).map(([k]) => k);
  console.log(`  >>> needs FLOATEXP (exp outside ±120): ${verdict.length ? verdict.join(', ') : 'NONE — plain df64/float OK'}`);
}
