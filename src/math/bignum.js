// bignum.js — fixed-point high-precision reals using BigInt.
//
// A real value v is stored as a BigInt m with  v = m / 2^prec.
// `prec` (bits after the binary point) is carried alongside / passed in.
// Sign is the BigInt's own sign (two's-complement not needed — BigInt is signed).
//
// We only need what the reference orbit and view-state require:
//   mulShift (rounded fixed mul), conversions to/from double & decimal string,
//   and a complex-square+add step. Pure ESM, no DOM. Node + browser.

// round((a*b) / 2^prec)  — round-to-nearest (ties toward +inf, negligible bias).
export function mulShift(a, b, prec) {
  const prod = a * b;
  if (prec === 0) return prod;
  const half = 1n << BigInt(prec - 1);
  return (prod + half) >> BigInt(prec);
}

// Convert fixed-point BigInt -> JS double. Handles arbitrarily large prec
// (where Number(m) alone would overflow to Infinity).
//
// FAST PATH (Spawn 29): for prec <= 1000, Number(m) rounds correctly to nearest
// in ONE step with no toString bit-length probe (the old path built an O(prec)
// binary string per call — at 2 calls per reference-orbit iteration that was a
// top cost of the deep ref build), and cannot overflow for the O(1)-magnitude
// values this project stores (overflow needs |v| >= 2^1024-prec ~ 2^24, checked
// anyway). Also MORE accurate: one correct rounding vs the general path's
// truncate-to-60-bits-then-round (up to 1 ulp apart — an accepted realization
// change, same class as mulShift rounding; gated by the unit suite + probe-wall).
// prec > 1000 (zooms beyond ~2^-936, approaching the 2^-1010 wall) and any
// overflowing magnitude fall back to the general path unchanged.
export function toDouble(m, prec) {
  if (m === 0n) return 0;
  if (prec <= 1000) {
    const v = Number(m);
    if (v !== Infinity && v !== -Infinity) return v * Math.pow(2, -prec);
  }
  const neg = m < 0n;
  let a = neg ? -m : m;
  const bits = a.toString(2).length;
  let e = 0;
  if (bits > 60) {
    const s = bits - 60;
    a = a >> BigInt(s);
    e = s;
  }
  const v = Number(a) * Math.pow(2, e - prec);
  return neg ? -v : v;
}

// Convert a JS double -> fixed-point BigInt at the given prec.
export function fromDouble(v, prec) {
  if (v === 0 || !isFinite(v)) return 0n;
  const neg = v < 0;
  v = Math.abs(v);
  let e = Math.floor(Math.log2(v));
  const mant = v / Math.pow(2, e); // in [1, 2)
  const mantInt = BigInt(Math.round(mant * Math.pow(2, 52))); // 53-bit integer
  const shift = e + prec - 52;
  let result;
  if (shift >= 0) result = mantInt << BigInt(shift);
  else result = mantInt >> BigInt(-shift);
  return neg ? -result : result;
}

// Parse a decimal string ("-0.7436438870371587") -> fixed-point BigInt.
// Supports an optional exponent ("1.25e-3").
export function fromDecimalString(s, prec) {
  s = String(s).trim();
  let neg = false;
  if (s[0] === '-') { neg = true; s = s.slice(1); }
  else if (s[0] === '+') s = s.slice(1);
  let exp = 0;
  const eIdx = s.search(/[eE]/);
  if (eIdx >= 0) { exp = parseInt(s.slice(eIdx + 1), 10) || 0; s = s.slice(0, eIdx); }
  let [ip, fp = ''] = s.split('.');
  ip = ip || '0';
  // value = (ip.fp) * 10^exp
  let digits = BigInt(ip + fp);
  let tenPow = fp.length - exp; // value = digits / 10^tenPow
  // fixed = round(digits * 2^prec / 10^tenPow)
  let num = digits << BigInt(prec);
  let den;
  if (tenPow >= 0) {
    den = 10n ** BigInt(tenPow);
  } else {
    num = num * (10n ** BigInt(-tenPow));
    den = 1n;
  }
  let q = num / den;
  const r = num % den;
  if (r * 2n >= den) q += 1n; // round to nearest
  return neg ? -q : q;
}

// Convert fixed-point BigInt -> decimal string with `digits` fractional places.
export function toDecimalString(m, prec, digits = 60) {
  const neg = m < 0n;
  let a = neg ? -m : m;
  // round-to-nearest at the requested number of decimal places
  const num = a * (10n ** BigInt(digits));
  const p = BigInt(prec);
  const half = 1n << (p - 1n);
  let scaled = (num + half) >> p;
  let s = scaled.toString().padStart(digits + 1, '0');
  const cut = s.length - digits;
  const ip = s.slice(0, cut) || '0';
  const fp = s.slice(cut);
  return (neg ? '-' : '') + ip + (digits > 0 ? '.' + fp : '');
}

// Choose a working precision (bits after point) for a given view radius.
// radius ~ 2^-zoomBits ; we add guard bits for iteration error headroom.
export function precForRadius(radius, guard = 64) {
  const zoomBits = radius > 0 ? Math.max(0, -Math.log2(radius)) : 0;
  return Math.ceil(zoomBits) + guard;
}
