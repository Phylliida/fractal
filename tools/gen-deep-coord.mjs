// gen-deep-coord.mjs — generate a genuine deep boundary coordinate (to ~2^-520) for
// validating the GPU deep engines past the old 2^-340 floor. The existing test
// coordinates run out of true boundary structure below ~2^-340 (their decimal strings
// are only ~380 bits), so a 2^-500 patch around them is a trivially-uniform escaping
// region — useless for testing chaotic high-iteration correctness.
//
// METHOD (standard "auto-deepen"): start at a known boundary point, and repeatedly
// (a) build the high-precision reference at the current center, (b) probe a small grid
// over a box of half-width = current radius for the DEEPEST escaping pixel (the densest
// filament), (c) relocate the BigInt center onto it, (d) shrink the radius. Following
// the deepest pixel down tracks a boundary filament, so the center stays a real boundary
// point while accumulating true low-order bits in BigInt. Prints the coordinate (enough
// decimal digits for the target depth) and a self-check of the chaotic mix at the bottom.
//
//   node tools/gen-deep-coord.mjs            (defaults: descend to ~2^-520)
//   TARGET=560 STEP=2 GRID=13 node tools/gen-deep-coord.mjs
import { fromDecimalString, toDecimalString, precForRadius, fromDouble, toDouble } from '../src/math/bignum.js';
import { computeReference } from '../src/math/reference.js';
import { escapePerturb, pixelDelta } from '../src/math/perturb.js';

// A known boundary point (true to ~2^-340). We only use it as a STARTING filament;
// everything below ~2^-340 is genuinely discovered by the descent below.
const RE0 = process.env.RE || '-1.369078017863660784890619576747781310848768032841633323730495873496232879296538490243106365484246242476783355722';
const IM0 = process.env.IM || '-0.071817675972918479944583194368632476442138106251769795140812120871593742404751576456750164324645880810732436640';

const TARGET = Number(process.env.TARGET || 520);   // descend to ~2^-TARGET
const STEP = Number(process.env.STEP || 2);         // octaves per descent step (factor 2^STEP)
const GRID = Number(process.env.GRID || 13);        // probe grid (GRID×GRID)
// maxIter cap during descent. MUST scale with TARGET: the boundary at 2^-b needs ~250·b
// iterations to resolve interior-vs-escaping, so a fixed cap (the old 140000) makes every
// probe pixel read as INTERIOR past ~2^-560 — the descent then thinks the whole box is
// inside the set and drifts off the filament (Spawn 25 bug). Default to the depth's own
// autoMaxIter + margin so the descent stays on the boundary all the way down.
const ITERCAP = Number(process.env.ITERCAP || Math.max(140000, Math.round(400 + TARGET * 250) + 8000));
const autoIter = (bits) => Math.min(ITERCAP, Math.round(400 + bits * 250));

const prec = precForRadius(2 ** -TARGET, 120);      // generous guard for the BigInt center
console.log(`descending to 2^-${TARGET}, step=${STEP} oct, grid=${GRID}², prec=${prec} bits, iterCap=${ITERCAP}`);

let cx = fromDecimalString(RE0, prec);
let cy = fromDecimalString(IM0, prec);
const t0 = Date.now();

for (let bits = 28; bits <= TARGET; bits += STEP) {
  const radius = 2 ** -bits;
  const maxIter = autoIter(bits);
  const ref = computeReference({ x: cx, y: cy, prec }, maxIter);
  const view = { radius, width: GRID, height: GRID };
  // Scan the grid: record each pixel's escape count (interior = maxIter) and the
  // hottest ESCAPING tip (the boundary is right at it). To DEEPEN while staying on
  // the boundary, we relocate onto the INTERIOR pixel nearest that tip — i.e. just
  // INSIDE the set, hugging the edge — NOT the escaping tip itself (chasing the tip
  // drifts the center to the exterior side and eventually the whole box escapes).
  const N = [];
  let hot = -1, hx = 0, hy = 0, escaped = 0, interior = 0;
  for (let gy = 0; gy < GRID; gy++) {
    for (let gx = 0; gx < GRID; gx++) {
      const { dcx, dcy } = pixelDelta(view, gx, gy);
      const r = escapePerturb(ref, dcx, dcy, maxIter);
      N.push({ gx, gy, dcx, dcy, n: r.n, esc: r.n < maxIter });
      if (r.n < maxIter) { escaped++; if (r.n > hot) { hot = r.n; hx = gx; hy = gy; } }
      else interior++;
    }
  }
  // Pick the relocation target. Three regimes:
  //  - straddle (interior+escaping): hug the boundary from the INSIDE — the interior
  //    pixel nearest the hottest escaping tip. This inward bias is what stops the
  //    center from drifting to the exterior side and falling off the filament deep.
  //  - fully exterior (coarse early steps): no fine structure visible yet — descend
  //    toward the hottest escaping tip to find the boundary.
  //  - fully interior (overshot inward): step back toward the box edge.
  let bdcx, bdcy, regime;
  if (escaped > 0 && interior > 0) {
    let bestD = Infinity; regime = 'straddle';
    for (const p of N) {
      if (p.esc) continue;
      const d = (p.gx - hx) * (p.gx - hx) + (p.gy - hy) * (p.gy - hy);
      if (d < bestD) { bestD = d; bdcx = p.dcx; bdcy = p.dcy; }
    }
  } else if (escaped > 0) {                          // fully exterior: chase the hottest tip
    regime = 'exterior';
    const p = N[hy * GRID + hx]; bdcx = p.dcx; bdcy = p.dcy;
  } else {                                           // fully interior: drift toward an edge
    regime = 'interior';
    const p = N[0]; bdcx = p.dcx; bdcy = p.dcy;
  }
  const best = hot < 0 ? maxIter : hot;
  cx += fromDouble(bdcx, prec);
  cy += fromDouble(bdcy, prec);
  if (bits % 20 === 0 || bits + STEP > TARGET) {
    console.log(`2^-${String(bits).padStart(3)}  ref.len=${String(ref.len).padStart(6)}  deepest=${String(best).padStart(6)}  esc=${String(escaped).padStart(3)}/${GRID * GRID}  ${regime.padEnd(8)} ${((Date.now() - t0) / 1000).toFixed(1)}s`);
  }
}

const digits = Math.ceil(TARGET / 3.32) + 12;       // enough decimals for the depth + margin
const reStr = toDecimalString(cx, prec, digits);
const imStr = toDecimalString(cy, prec, digits);
console.log('\n=== deep boundary coordinate ===');
console.log('RE =', reStr);
console.log('IM =', imStr);

// Self-check: render a grid at the bottom depth at FULL autoIter and report the
// interior/exterior mix + a sense of divergence (a real boundary view straddles both).
const checkBits = TARGET - 20;
const cr = 1.5 * 2 ** -checkBits;
const cIter = Math.min(2_000_000, Math.round(400 + checkBits * 250));
const cref = computeReference({ x: cx, y: cy, prec }, cIter);
const cview = { radius: cr, width: 24, height: 24 };
let inside = 0, esc = 0, minN = 1e18, maxN = -1;
for (let gy = 0; gy < 24; gy++) {
  for (let gx = 0; gx < 24; gx++) {
    const { dcx, dcy } = pixelDelta(cview, gx, gy);
    const r = escapePerturb(cref, dcx, dcy, cIter);
    if (r.n >= cIter) inside++; else { esc++; minN = Math.min(minN, r.n); maxN = Math.max(maxN, r.n); }
  }
}
console.log(`\nself-check @2^-${checkBits} (autoIter=${cIter}, 24² grid): interior=${inside} escaping=${esc} ` +
  `escN∈[${esc ? minN : '-'},${esc ? maxN : '-'}]  ref.len=${cref.len}`);
console.log(esc > 20 && inside > 20
  ? '✓ straddles interior+exterior with a wide escape-count spread → genuine chaotic boundary'
  : '⚠ not a clean boundary straddle — consider a different start or smaller STEP');
