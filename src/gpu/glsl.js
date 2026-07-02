// glsl.js — GLSL ES 3.00 (WebGL2) shader sources for the GPU Mandelbrot engines.
//
// Three fragment programs are produced here:
//   - naive  (float32)      : escape-time, shallow zoom (radius >= ~2^-22)
//   - naive  (df64)         : escape-time in double-single, medium zoom (~2^-44)
//   - perturb(float32)      : delta iteration + Zhuoran rebasing, deep zoom
//   - color                 : map the smooth-count float texture -> RGBA via a LUT
//
// All escape shaders write a vec4 into an RGBA32F attachment:
//   .r = smooth count sn  (-1.0 == interior / did-not-escape)
//   .g = integer iteration n (for debug / glitch overlay)
//   .b = glitch flag (perturb only; 1.0 == Pauldelbrot glitch suspected)
//   .a = 1.0
//
// The coordinate mapping is the single source of truth and is mirrored in JS for
// validation:  c = uOrigin + gl_FragCoord.xy * uScale   (gl_FragCoord = texel+0.5)
// so a test can recompute the exact c for any texel and compare to the CPU oracle.

// Fullscreen-triangle vertex shader (shared by every program).
export const VERT = `#version 300 es
in vec2 aPos;
void main(){ gl_Position = vec4(aPos, 0.0, 1.0); }
`;

// Double-single (df64) arithmetic on vec2(hi,lo); value = hi + lo with
// |lo| <= 0.5 ulp(hi). Standard GPU technique (Thall / Thasler).
//
// CRITICAL (Spawn 8 — found only once a REAL GPU was used; SwiftShader hid it):
// the Dekker/Veltkamp error terms rely on EXACT IEEE-754 float32 ops and are
// catastrophically fragile to the compiler's freedom to reassociate FP and to
// contract a*b±c into FMA. The NVIDIA ANGLE compiler (BOTH the Vulkan and the
// native-GLES backends) applies the algebraically-valid-but-FP-invalid identity
// `ca - (ca - x) -> x`, which zeroes the split (a_lo == 0) and collapses the WHOLE
// df64 to plain float32 (~24-bit) — silently, so the fractal just goes wrong deep.
// GLSL ES 3.00 (WebGL2) has neither the `precise` qualifier (3.20+) nor `fma`, so
// the only portable defense is an OPTIMIZATION BARRIER on every rounded result:
// `ob(x)` round-trips x through the integer domain, XOR-ing with uOptBarrier (a
// uniform == 0 the compiler cannot prove is zero), which forbids reassociation /
// contraction across that point. A plain intBitsToFloat(floatBitsToInt(x)) is NOT
// enough — the compiler folds the round-trip away; the XOR-with-an-opaque-zero is
// what actually holds. Proven exact on RTX 3090 (Vulkan + GLES) by tools/probe-df64.mjs;
// a partial/minimal placement still collapses, so EVERY op result is wrapped.
// uOptBarrier must be declared + set to 0 by every program that uses these ops
// (the renderer sets it in _program()); SwiftShader is unaffected either way.
//
// CRITICAL #2 (Spawn 9 — the OTHER half, and the real cause of the residual deep-GPU
// wrongness Spawn 8 left open): the df64 reference orbit is read per-iteration from a
// float32 texture via texelFetch. A sampler with NO precision qualifier defaults to
// mediump in the fragment shader, and NVIDIA honours mediump literally — it returns
// the texel rounded to ~fp16 (~10-bit mantissa). That ~10-bit Z destroys df64 (the lo
// word becomes meaningless), so the deep engines were ~90% wrong on real hardware even
// WITH the barrier above. SwiftShader implements mediump as fp32, which is why it (and
// every prior spawn) never saw it. THE FIX: declare every reference sampler `highp
// sampler2D` (uRef in all four perturb shaders; uSn in the color pass). Proven by
// tools/probe-localize/state/texz/fix.mjs (localized: per-iteration texelFetch'd Z is
// the only operand that collapses; uniform/constant/arithmetic operands are fine) and
// by probe-xbackend (90% -> 0.00% GPU-vs-SwiftShader) + validate:gpu:real (12 FAIL -> 0).
// NOTE: the residual was NEVER the driver defeating the per-op barrier in the big
// shader (Spawn 8's hypothesis) — the integer-XOR barrier is genuinely opaque and was
// always intact; the leak was sampler precision. The barrier is still required for the
// split (CRITICAL #1, a separate, texture-independent reassociation — proven in isolation).
export const DF64_LIB = `
uniform int uOptBarrier;   // == 0; opaque to the optimizer (see ob() below)
float ob(float x){ return intBitsToFloat(floatBitsToInt(x) ^ uOptBarrier); }
vec2 ds_set(float a){ return vec2(a, 0.0); }
float ds_tofloat(vec2 a){ return a.x + a.y; }
// BARRIER PLACEMENT — MINIMAL, not maximal (Spawn 20). The earlier shipping version
// wrapped EVERY intermediate (8/add, 24/mul); that was a ~2.6× over-tax (the barrier
// destroys FMA pipelining + the instruction scheduler, not just the XOR cost). Only
// the precision-critical points actually need a barrier, and only TWO compiler
// freedoms must be blocked: (i) reassociation that CANCELS -- s-(s-x)->x, ca-(ca-x)->x,
// (p+e)-p->e -- collapsing the Veltkamp split / TwoSum; (ii) FMA contraction a*b+-c,
// which de-rounds the Dekker product (the killer of the error term a_hi*b_hi-p).
// The leftover additive intermediates (small cross-products, the e-sum chain) only
// perturb the LOW correction word by ~1 ulp and are safe to let the compiler schedule.
// MEASURED INTACT on the real RTX 3090 (ANGLE/Vulkan) at ~1e-14 worst relerr — same as
// the full placement, 4 orders below the 1e-10 collapse gate — by tools/probe-barrier.mjs
// (the per-op regression gate; run with GPU=1 after ANY change here). Confirmed end-to-end
// by probe-xbackend (GPU≡SwiftShader 0.00%) + validate:gpu:real. The bare ops still collapse
// (probe-df64's CUR/A/C, probe-barrier's none/minC/lean5 controls) — the barrier IS required.
vec2 ds_add(vec2 a, vec2 b){
  float s  = ob(a.x + b.x);            // materialize: blocks v=(a.x+b.x)-a.x → b.x
  float v  = ob(s - a.x);              // materialize: blocks s-(s-a.x) → a.x
  float e  = (a.x - ob(s - v)) + (b.x - v);
  e = e + (a.y + b.y);
  float hi = ob(s + e);                // materialize: blocks (s+e)-s → e in lo
  float lo = e - ob(hi - s);
  return vec2(hi, lo);
}
vec2 ds_sub(vec2 a, vec2 b){ return ds_add(a, vec2(-b.x, -b.y)); }
vec2 ds_mul(vec2 a, vec2 b){
  const float SPLIT = 4097.0;            // 2^12 + 1
  float p  = ob(a.x * b.x);              // round the product (subtracted below; blocks fma into −p)
  float ca = ob(SPLIT * a.x); float a_hi = ca - ob(ca - a.x); float a_lo = a.x - a_hi;
  float cb = ob(SPLIT * b.x); float b_hi = cb - ob(cb - b.x); float b_lo = b.x - b_hi;
  float e = (ob(a_hi*b_hi) - p) + a_hi*b_lo + a_lo*b_hi + a_lo*b_lo;  // ob() de-fuses a_hi*b_hi−p
  e += a.x*b.y + a.y*b.x;
  float hi = ob(p + e);                  // materialize: blocks (p+e)-p → e in lo
  float lo = e - ob(hi - p);
  return vec2(hi, lo);
}
// Dedicated df64 SQUARING (Spawn 27): Dekker a*a needs only ONE Veltkamp split
// (the second operand's split is identical) and the cross terms collapse
// (a_hi*b_lo + a_lo*b_hi -> 2*a_hi*a_lo exactly; a.x*b.y + a.y*b.x -> 2*a.x*a.y
// exactly — both are a shared product doubled, and doubling is exact in IEEE-754).
// 6 barriers vs ds_mul's 8, ~3 fewer mults. Value differs from ds_mul(a,a) by at
// most ~1 ulp of the LOW word ((x+h)+h vs x+2h rounding — same freedom the lean
// barrier already grants the compiler's scheduler); the CPU-oracle bulk gates
// (validate:gpu[:real]) are the arbiter, as for every shader change.
// MEASURED (probe-barrier sqr ladder, RTX 3090/Vulkan): INTACT at 2.1e-15 single /
// 2.4e-11 attracting-chained (sqr_none control collapses at 1.2e-4). Speed
// (bench-sqr.mjs, contention-robust protocol): isolated 1.00x AND full-shader
// 0.999–1.031x on the real GPU (the driver CSEs the duplicate split even in the
// big shader), 0.95–0.98x on SwiftShader. VERDICT: OPT-IN (opts.sqr === true /
// p.sqrOn), NOT the default — per the lean-kernel/df64-escape precedent the
// default stays the measured-fastest config. Kept validated for a future
// register/ALU-constrained GPU where one fewer split might matter.
vec2 ds_sqr(vec2 a){
  const float SPLIT = 4097.0;            // 2^12 + 1
  float p  = ob(a.x * a.x);
  float ca = ob(SPLIT * a.x); float a_hi = ca - ob(ca - a.x); float a_lo = a.x - a_hi;
  float e = (ob(a_hi*a_hi) - p) + 2.0*(a_hi*a_lo) + a_lo*a_lo;
  e += 2.0*(a.x*a.y);
  float hi = ob(p + e);
  float lo = e - ob(hi - p);
  return vec2(hi, lo);
}
// frexp exponent of |x|: the k with |x|*2^-k in [0.5,1) for nonzero x; |x| >= 2^(k-1).
// Reads the IEEE-754 biased exponent field directly (sign masked off). For x==0 it
// returns -126 (treats 0 as the smallest normal) — callers handle 0 explicitly.
// Used by the perturb fast-skip (cheap magnitude exponent) and by FE_LIB.
int ilogb1(float x){ return ((floatBitsToInt(x) & 0x7fffffff) >> 23) - 126; }
// Multiply a df64 by 2^p EXACTLY (both hi and lo), for the rescaled perturb engine.
// p < -100 returns 0: a term scaled that far below an O(1) frame is below df64's
// ~46-bit precision (dropped by the next ds_add anyway) — and the cutoff keeps every
// scale factor a normal float (no subnormal, since our mantissas have |hi| ~ O(1)).
vec2 ds_scale2(vec2 a, int p){
  if (p < -100) return vec2(0.0);
  float s = intBitsToFloat((p + 127) << 23);
  return a * s;
}
`;

// floatexp ("fe") library: a real value is  m * 2^e  where m is a df64 (vec2
// hi+lo, ~46-bit) normalized so |m.x| in [0.5,1), and e is an int. This extends
// the df64 mantissa with an unbounded exponent, so the per-pixel deltas dc, dz —
// which are ~2^-270 at deep zoom and would UNDERFLOW float32 (min normal 2^-126),
// flooring the plain-df64 perturb path at ~2^-112 — keep full precision. The
// reference orbit Z stays df64 (it is O(1)); only the small deltas use fe.
//
// Relies only on the df64 ds_* ops + IEEE-754 bit reinterpretation (WebGL2's
// floatBitsToInt / intBitsToFloat). Depends on DF64_LIB being included first.
//
// PERF: normalization used to estimate the binary exponent with log2() and apply
// it with exp2() — two software transcendentals per fe_norm, and fe_add called
// exp2 three more times. The hot perturbation loop runs ~20 fe ops per pixel-
// iteration, so that was ~60+ transcendental calls/iteration — the dominant cost
// on a software rasterizer (SwiftShader). fe_pow2/fe_ilogb1 below do the same job
// by reading and writing the float's exponent field directly: exact (no ±1 log2
// rounding, so no correction step and no vendor-approx "sparkle" risk on mobile)
// and far cheaper. Valid because our mantissas stay in [~2^-50, 2], so every
// exponent we touch is well inside the normal-float range [-126,127].
const FE_LIB = `
struct fe { vec2 m; int e; };
const int FE_ZERO_E = -100000;          // exponent sentinel for the value 0

// 2^k as an exact float, built straight into the IEEE-754 exponent field
// (replaces exp2). k must be in [-126,127] — true for every k we feed it.
float fe_pow2(int k){ return intBitsToFloat((k + 127) << 23); }
// frexp exponent: the k with |x|*2^-k in [0.5,1), for nonzero x (replaces log2).
// Shared bit-twiddle lives in DF64_LIB (ilogb1), included before FE_LIB.
int fe_ilogb1(float x){ return ilogb1(x); }

fe fe_norm(vec2 m, int e){
  if (m.x == 0.0) return fe(vec2(0.0), FE_ZERO_E);
  int k = fe_ilogb1(m.x);                // exact: |m.x|*2^-k lands in [0.5,1)
  return fe(m * fe_pow2(-k), e + k);     // power-of-two scale is exact for hi AND lo
}
fe   fe_fromds(vec2 d){ return fe_norm(d, 0); }
fe   fe_fromf(float f){ return fe_norm(vec2(f, 0.0), 0); }
// value = (hi+lo)*2^e. For very negative e the value is far below O(1) and
// negligible against the O(1) reference when converted to a float — return 0
// (fe_pow2 only covers e >= -126; below that it would underflow to 0 anyway).
float fe_tof(fe a){
  if (a.e < -126) return 0.0;
  float s = fe_pow2(a.e);
  return a.m.x * s + a.m.y * s;
}
fe   fe_neg(fe a){ return fe(vec2(-a.m.x, -a.m.y), a.e); }
fe   fe_dbl(fe a){ if (a.m.x == 0.0) return a; return fe(a.m, a.e + 1); }

fe fe_add(fe a, fe b){
  if (a.m.x == 0.0) return b;
  if (b.m.x == 0.0) return a;
  int E = a.e >= b.e ? a.e : b.e;
  if (E - a.e > 52) return b;           // a negligible beside b
  if (E - b.e > 52) return a;           // b negligible beside a
  vec2 am = a.m * fe_pow2(a.e - E);     // scale to common exponent (exact, <= 1)
  vec2 bm = b.m * fe_pow2(b.e - E);
  return fe_norm(ds_add(am, bm), E);
}
fe fe_sub(fe a, fe b){ return fe_add(a, fe_neg(b)); }
fe fe_mul(fe a, fe b){
  if (a.m.x == 0.0 || b.m.x == 0.0) return fe(vec2(0.0), FE_ZERO_E);
  return fe_norm(ds_mul(a.m, b.m), a.e + b.e);
}
fe fe_mulds(fe a, vec2 d){              // fe * arbitrary df64 (e.g. reference Z)
  if (a.m.x == 0.0 || d.x == 0.0) return fe(vec2(0.0), FE_ZERO_E);
  return fe_norm(ds_mul(a.m, d), a.e);
}
fe fe_sqr(fe a){                        // a*a via the dedicated df64 squaring
  if (a.m.x == 0.0) return fe(vec2(0.0), FE_ZERO_E);
  return fe_norm(ds_sqr(a.m), a.e + a.e);
}
// magnitude compare for NON-NEGATIVE fe (squared magnitudes): is a < b ?
bool fe_lt(fe a, fe b){
  if (a.m.x == 0.0) return (b.m.x != 0.0);
  if (b.m.x == 0.0) return false;
  if (a.e != b.e) return a.e < b.e;
  if (a.m.x != b.m.x) return a.m.x < b.m.x;
  return a.m.y < b.m.y;
}
`;

// Call-site squaring macros. DEFAULT = the historical ds_mul(a,a)/fe_mul(a,a)
// source text (bit-for-bit the pre-Spawn-27 shader). opts.sqr === true compiles
// the dedicated ds_sqr/fe_sqr instead — MEASURED (Spawn 27, robust protocol):
// 0.999–1.031× on the real RTX 3090 (the driver CSEs the duplicate split, so the
// op cut buys ~nothing) and 0.95–0.98× on SwiftShader (slightly NEGATIVE on the
// fallback backend where speed matters most) → kept OPT-IN, not the default, per
// the lean-kernel/df64-escape precedent. Gate for enabling it on a future GPU:
// probe:barrier sqr ladder + validate:gpu:real + bench:sqr.
const sqrDefs = (opts = {}) => (opts.sqr === true
  ? `#define DS_SQR(a) ds_sqr(a)
#define FE_SQR(a) fe_sqr(a)`
  : `#define DS_SQR(a) ds_mul(a,a)
#define FE_SQR(a) fe_mul(a,a)`);

// Smooth-count helper — bit-identical to naive.js / perturb.js:
//   logZn = 0.5*ln(mag2);  nu = log2(logZn / ln2);  sn = n + 1 - nu
const SMOOTH = `
float smoothCount(float n, float mag2){
  float logZn = 0.5 * log(mag2);
  float nu = log(logZn / 0.6931471805599453) / 0.6931471805599453;
  return n + 1.0 - nu;
}
`;

// Build the naive escape shader. df64=true -> double-single coordinates.
export function naiveFrag(df64, opts = {}) {
  const typedefs = df64
    ? `#define REAL vec2
       #define R(x) ds_set(x)
       #define ADD(a,b) ds_add(a,b)
       #define SUB(a,b) ds_sub(a,b)
       #define MUL(a,b) ds_mul(a,b)
       #define SQR(a) ${opts.sqr === true ? 'ds_sqr(a)' : 'ds_mul(a,a)'}
       #define TOF(a) ds_tofloat(a)`
    : `#define REAL float
       #define R(x) (x)
       #define ADD(a,b) ((a)+(b))
       #define SUB(a,b) ((a)-(b))
       #define MUL(a,b) ((a)*(b))
       #define SQR(a) ((a)*(a))
       #define TOF(a) (a)`;
  return `#version 300 es
precision highp float;
precision highp int;
${df64 ? DF64_LIB : ''}
${SMOOTH}
${typedefs}
uniform REAL uOx, uOy, uScale;   // c = (uOx,uOy) + frag.xy*uScale
uniform int  uMaxIter;
uniform float uBailoutSq;
out vec4 frag;
void main(){
  REAL fx = R(gl_FragCoord.x);
  REAL fy = R(gl_FragCoord.y);
  REAL cx = ADD(uOx, MUL(fx, uScale));
  REAL cy = ADD(uOy, MUL(fy, uScale));
  REAL zx = R(0.0), zy = R(0.0), zx2 = R(0.0), zy2 = R(0.0);
  int n = 0;
  for (int i = 0; i < 100000000; i++) {
    float m2 = TOF(ADD(zx2, zy2));
    if (n >= uMaxIter || m2 > uBailoutSq) break;
    zy = ADD(MUL(ADD(zx, zx), zy), cy);   // 2*zx*zy + cy
    zx = ADD(SUB(zx2, zy2), cx);          // zx2 - zy2 + cx
    zx2 = SQR(zx);
    zy2 = SQR(zy);
    n++;
  }
  float mag2 = TOF(ADD(zx2, zy2));
  if (n >= uMaxIter) { frag = vec4(-1.0, float(n), 0.0, 1.0); return; }
  frag = vec4(smoothCount(float(n), mag2), float(n), 0.0, 1.0);
}
`;
}

// Build the perturbation escape shader (float32 deltas). The reference orbit is
// sampled from an RG32F texture uRef of width uRefW: texel index n -> (Zx,Zy).
// Per-pixel delta dc = (uOx,uOy) + frag.xy*uScale (all small float32 quantities,
// valid where dc is float32-representable, i.e. radius >~ 2^-100).
export function perturbFrag() {
  return `#version 300 es
precision highp float;
precision highp int;
${SMOOTH}
uniform highp sampler2D uRef;     // RG32F: (Zx, Zy) per iteration index (highp: see df64 note)
uniform int uRefW;          // texture width (texels per row)
uniform int uRefLen;        // number of valid reference iterations (z_0..z_len)
uniform float uOx, uOy, uScale;  // dc = (uOx,uOy) + frag.xy*uScale
uniform int uMaxIter;
uniform float uBailoutSq;
uniform float uGlitchTol;   // Pauldelbrot tol (e.g. 1e-6), 0 to disable
out vec4 frag;

vec2 refZ(int m){
  int x = m % uRefW;
  int y = m / uRefW;
  return texelFetch(uRef, ivec2(x, y), 0).rg;
}

void main(){
  float dcx = uOx + gl_FragCoord.x * uScale;
  float dcy = uOy + gl_FragCoord.y * uScale;
  float dx = 0.0, dy = 0.0;   // dz relative to Z_m
  int m = 0;                  // reference index
  int n = 0;
  float glitch = 0.0;
  vec2 Z = refZ(0);           // Z[m] carried across iterations (one fetch per iter)
  for (int i = 0; i < 100000000; i++) {
    if (n >= uMaxIter) break;
    // Z = Z[m].  dz' = 2*Z_m*dz + dz^2 + dc
    float ndx = 2.0*(Z.x*dx - Z.y*dy) + (dx*dx - dy*dy) + dcx;
    float ndy = 2.0*(Z.x*dy + Z.y*dx) + (2.0*dx*dy) + dcy;
    dx = ndx; dy = ndy;
    m++; n++;
    Z = refZ(m);              // now Z[m] (= Zm for z = Zm + dz)
    float zfx = Z.x + dx;     // true value z = Z_m + dz
    float zfy = Z.y + dy;
    float mag2 = zfx*zfx + zfy*zfy;
    if (mag2 > uBailoutSq) {
      frag = vec4(smoothCount(float(n), mag2), float(n), glitch, 1.0);
      return;
    }
    float dz2 = dx*dx + dy*dy;
    bool rebase = (mag2 < dz2) || (m == uRefLen);
    if (uGlitchTol > 0.0 && !rebase) {
      float zref2 = Z.x*Z.x + Z.y*Z.y;
      if (zref2 > 0.0 && mag2 < uGlitchTol * zref2) glitch = 1.0;
    }
    if (rebase) { dx = zfx; dy = zfy; m = 0; Z = refZ(0); }
  }
  frag = vec4(-1.0, float(n), glitch, 1.0);   // interior
}
`;
}

// df64 perturbation: like perturbFrag but the reference orbit AND the deltas are
// double-single (~46-bit). Fixes the f32 breakdown at high maxIter (the f32
// reference's ~2^-24 reconstruction error amplifies on chaotic high-count
// pixels). Reference texture is RGBA32F: (Zx.hi, Zx.lo, Zy.hi, Zy.lo).
export function perturbFragDf64(opts = {}) {
  // Software-pipelined reference prefetch (Spawn 27; same restructure as the rescaled
  // engine, where it measured bit-identical + 1.01–1.08× GPU / ~2.0× SwiftShader):
  // carry Z[m+1] so the per-iteration fetch is issued a full iteration before use.
  const PF = opts.prefetch === true;
  return `#version 300 es
precision highp float;
precision highp int;
${DF64_LIB}
${sqrDefs(opts)}
${SMOOTH}
uniform highp sampler2D uRef;   // highp REQUIRED: a mediump sampler truncates the
uniform int uRefW, uRefLen, uMaxIter;   // texelFetch'd df64 reference to fp16 on NVIDIA
uniform vec2 uOx, uOy, uScale;   // df64 dc origin + per-pixel step
uniform float uBailoutSq;
uniform float uGlitchTol;
uniform int uFastSkip;           // 1 = skip the provably-inert escape/rebase block
out vec4 frag;

void getZ(int m, out vec2 Zx, out vec2 Zy){
  vec4 v = texelFetch(uRef, ivec2(m % uRefW, m / uRefW), 0);
  Zx = v.xy; Zy = v.zw;
}

void main(){
  vec2 dcx = ds_add(uOx, ds_mul(ds_set(gl_FragCoord.x), uScale));
  vec2 dcy = ds_add(uOy, ds_mul(ds_set(gl_FragCoord.y), uScale));
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  int m = 0, n = 0;
  float glitch = 0.0;
  // Z[m] is carried across iterations: the Z[m+1] fetched for the z=Z+dz test below
  // IS the next iteration's Z[m], so we fetch the reference once per iteration (plus
  // a re-fetch of Z[0] on rebase) instead of twice. Bit-identical to a per-iter fetch.
  vec2 Zx, Zy; getZ(0, Zx, Zy);
${PF ? `  vec2 Znx, Zny; getZ(min(1, uRefLen), Znx, Zny);   // prefetched Z[m+1]` : ''}
  for (int i = 0; i < 100000000; i++) {
    if (n >= uMaxIter) break;
    // Zx,Zy = Z[m].   dz' = 2*Z*dz + dz^2 + dc
    vec2 t1 = ds_sub(ds_mul(Zx, dx), ds_mul(Zy, dy));   // Zx*dx - Zy*dy
    vec2 t2 = ds_add(ds_mul(Zx, dy), ds_mul(Zy, dx));   // Zx*dy + Zy*dx
    vec2 sx = ds_sub(DS_SQR(dx), DS_SQR(dy));           // dx^2 - dy^2
    vec2 sy = ds_mul(dx, dy);                           // dx*dy
    vec2 ndx = ds_add(ds_add(ds_add(t1, t1), sx), dcx);            // 2*t1 + sx + dc
    vec2 ndy = ds_add(ds_add(ds_add(t2, t2), ds_add(sy, sy)), dcy);// 2*t2 + 2*sy + dc
    dx = ndx; dy = ndy;
    m++; n++;
${PF ? `    Zx = Znx; Zy = Zny;                       // Z[m] — fetched one iteration ago
    getZ(min(m + 1, uRefLen), Znx, Zny);      // prefetch Z[m+1]; used next iteration` : `    getZ(m, Zx, Zy);                                    // now Z[m] (= Zm for z = Zm+dz)`}
    // Fast skip (bit-identical): when dz is far below the O(1) reference Z_m, the
    // true value z = Z_m + dz can neither escape (|z| <= |Z_m|+|dz| << bailout) nor
    // rebase (|z| >= |Z_m|-|dz| > |dz|), and the Pauldelbrot glitch test is false
    // (mag2 ~ |Z_m|^2). So the whole escape/rebase/glitch block leaves state
    // unchanged — skip it. |dz| < 2^(sdz+1); |Z_m| >= 2^(ezm-1); ezm >= sdz+4 =>
    // |Z_m| > 2|dz| (no rebase); ezm <= 6 => |z| < 256 (no escape); m != uRefLen
    // keeps the forced end-of-reference rebase.
    if (uFastSkip == 1 && m != uRefLen) {
      int sdz = max(ilogb1(dx.x), ilogb1(dy.x));
      int ezm = max(ilogb1(Zx.x), ilogb1(Zy.x));
      if (ezm <= 6 && ezm >= sdz + 4) continue;
    }
    vec2 zfx = ds_add(Zx, dx);
    vec2 zfy = ds_add(Zy, dy);
    float mag2 = ds_tofloat(ds_add(DS_SQR(zfx), DS_SQR(zfy)));
    if (mag2 > uBailoutSq) {
      frag = vec4(smoothCount(float(n), mag2), float(n), glitch, 1.0);
      return;
    }
    float dz2 = ds_tofloat(ds_add(DS_SQR(dx), DS_SQR(dy)));
    bool rebase = (mag2 < dz2) || (m == uRefLen);
    if (uGlitchTol > 0.0 && !rebase) {
      float zref2 = ds_tofloat(ds_add(DS_SQR(Zx), DS_SQR(Zy)));
      if (zref2 > 0.0 && mag2 < uGlitchTol * zref2) glitch = 1.0;
    }
    if (rebase) { dx = zfx; dy = zfy; m = 0; getZ(0, Zx, Zy);${PF ? ' getZ(min(1, uRefLen), Znx, Zny);' : ''} }
  }
  frag = vec4(-1.0, float(n), glitch, 1.0);
}
`;
}

// floatexp perturbation: like perturbFragDf64 but the per-pixel deltas dc, dz are
// floatexp (df64 mantissa + int exponent), so the shader works below the df64
// float32-exponent floor (~2^-112) all the way down to the double exponent range
// (~2^-1000) — this is what lets the GPU render ~2^270 zooms. The reference orbit
// is the SAME df64 texture (Zx.hi,Zx.lo,Zy.hi,Zy.lo); only dc/dz carry an exponent.
// dc origin/step come in as fe: mantissa (df64 vec2) + int exponent uniforms.
export function perturbFragFloatexp(opts = {}) {
  // Software-pipelined reference prefetch — see perturbFragDf64/perturbFragRescaled.
  const PF = opts.prefetch === true;
  return `#version 300 es
precision highp float;
precision highp int;
${DF64_LIB}
${FE_LIB}
${sqrDefs(opts)}
${SMOOTH}
uniform highp sampler2D uRef;   // highp REQUIRED: a mediump sampler truncates the
uniform int uRefW, uRefLen, uMaxIter;   // texelFetch'd df64 reference to fp16 on NVIDIA
uniform vec2 uOxm, uOym, uScalem;   // fe mantissas (df64) of dc origin + per-pixel step
uniform int  uOxe, uOye, uScalee;   // matching fe exponents
uniform float uBailoutSq;
uniform float uGlitchTol;
uniform int uFastSkip;              // 1 = skip the provably-inert escape/rebase block
out vec4 frag;

void getZ(int m, out vec2 Zx, out vec2 Zy){
  vec4 v = texelFetch(uRef, ivec2(m % uRefW, m / uRefW), 0);
  Zx = v.xy; Zy = v.zw;
}

void main(){
  fe sc = fe(uScalem, uScalee);
  fe dcx = fe_add(fe(uOxm, uOxe), fe_mul(fe_fromf(gl_FragCoord.x), sc));
  fe dcy = fe_add(fe(uOym, uOye), fe_mul(fe_fromf(gl_FragCoord.y), sc));
  fe dx = fe(vec2(0.0), FE_ZERO_E), dy = fe(vec2(0.0), FE_ZERO_E);
  int m = 0, n = 0;
  float glitch = 0.0;
  // Z[m] carried across iterations (one reference fetch per iter; re-fetch Z[0] on
  // rebase) — the Z[m+1] needed for z=Z+dz IS next iteration's Z[m]. Bit-identical.
  vec2 Zx, Zy; getZ(0, Zx, Zy);
${PF ? `  vec2 Znx, Zny; getZ(min(1, uRefLen), Znx, Zny);   // prefetched Z[m+1]` : ''}
  for (int i = 0; i < 100000000; i++) {
    if (n >= uMaxIter) break;
    // Zx,Zy = Z[m].  dz' = 2*Z*dz + dz^2 + dc   (Z is df64, dz/dc are fe)
    fe t1 = fe_sub(fe_mulds(dx, Zx), fe_mulds(dy, Zy));   // Zx*dx - Zy*dy
    fe t2 = fe_add(fe_mulds(dy, Zx), fe_mulds(dx, Zy));   // Zx*dy + Zy*dx
    fe sx = fe_sub(FE_SQR(dx), FE_SQR(dy));               // dx^2 - dy^2
    fe sy = fe_mul(dx, dy);                               // dx*dy
    fe ndx = fe_add(fe_add(fe_dbl(t1), sx), dcx);                 // 2*t1 + sx + dc
    fe ndy = fe_add(fe_add(fe_dbl(t2), fe_dbl(sy)), dcy);         // 2*t2 + 2*sy + dc
    dx = ndx; dy = ndy;
    m++; n++;
${PF ? `    Zx = Znx; Zy = Zny;                       // Z[m] — fetched one iteration ago
    getZ(min(m + 1, uRefLen), Znx, Zny);      // prefetch Z[m+1]; used next iteration` : `    getZ(m, Zx, Zy);                                      // now Z[m] (= Zm)`}
    // Fast skip (bit-identical): when dz is far below the O(1) reference Z_m, the
    // true value z = Z_m + dz can neither escape (|z| <= |Z_m|+|dz| << bailout) nor
    // rebase (|z| >= |Z_m|-|dz| > |dz|), and the Pauldelbrot glitch test is false.
    // So the whole escape/rebase/glitch block leaves state unchanged — skip it.
    // |dz| < 2^(sdz+1); |Z_m| >= 2^(ezm-1); ezm >= sdz+4 => |Z_m| > 2|dz| (no
    // rebase); ezm <= 6 => |z| < 256 (no escape). Below the df64 floor Z_m reads 0
    // (ezm=-126) and the exact path wouldn't rebase either (z=dz, |z|=|dz| not <),
    // so the skip still matches. m != uRefLen keeps the forced end-of-ref rebase.
    if (uFastSkip == 1 && m != uRefLen) {
      int sdz = max(dx.e, dy.e);
      int ezm = max(ilogb1(Zx.x), ilogb1(Zy.x));
      if (ezm <= 6 && ezm >= sdz + 4) continue;
    }
    fe zfx = fe_add(fe_fromds(Zx), dx);                   // true z = Z_m + dz
    fe zfy = fe_add(fe_fromds(Zy), dy);
    fe mag2 = fe_add(FE_SQR(zfx), FE_SQR(zfy));
    float mag2f = fe_tof(mag2);
    if (mag2f > uBailoutSq) {
      frag = vec4(smoothCount(float(n), mag2f), float(n), glitch, 1.0);
      return;
    }
    fe dz2 = fe_add(FE_SQR(dx), FE_SQR(dy));
    bool rebase = fe_lt(mag2, dz2) || (m == uRefLen);
    if (uGlitchTol > 0.0 && !rebase) {
      fe zr = fe_add(FE_SQR(fe_fromds(Zx)), FE_SQR(fe_fromds(Zy)));
      if (zr.m.x != 0.0 && fe_lt(mag2, fe_mul(fe_fromf(uGlitchTol), zr))) glitch = 1.0;
    }
    if (rebase) { dx = zfx; dy = zfy; m = 0; getZ(0, Zx, Zy);${PF ? ' getZ(min(1, uRefLen), Znx, Zny);' : ''} }
  }
  frag = vec4(-1.0, float(n), glitch, 1.0);
}
`;
}

// Rescaled single-exponent perturbation — same depth range as the floatexp engine
// (works far below the float32 floor) but ~1.3-1.5× faster on the per-iteration UPDATE.
//
// The floatexp engine carries a SEPARATE exponent on every delta component and
// renormalizes after EVERY arithmetic op (~14 fe ops/iteration, each a normalize +
// struct/branch). Here the delta dz = (Dx,Dy)·2^S keeps df64 mantissas Dx,Dy under
// ONE shared int exponent S, so the update dz' = 2·Z·dz + dz² + dc runs in raw df64:
//   - linear  L = 2·(Zx·Dx − Zy·Dy, Zx·Dy + Zy·Dx),  exponent S       (Z is df64, O(1))
//   - quad    (Dx²−Dy², 2·Dx·Dy),                     exponent 2S      (dropped when
//             >52 bits below the frame — same point fe_add drops it; at deep zoom
//             dz²~2^-540 vs linear~2^-270 it's invisible, so this is free precision)
//   - dc      (Cx,Cy)·2^Sc                            (collapsed once before the loop)
// align all three to the frame W = max(S,Sc) with exact power-of-two scalings, sum in
// df64, and renormalize the shared exponent ONCE. The escape/rebase test still runs in
// exact floatexp (convert dz→fe per component) so the Zhuoran rebase DECISION is
// bit-for-bit the same logic as the floatexp engine — the part that needs exact
// magnitudes when the true z passes near 0 is untouched. Validated vs the CPU oracle
// (tools/validate-gpu.mjs) and cross-checked to agree with the fe engine.
export function perturbFragRescaled(opts = {}) {
  // Build-time variant selection (Spawn 22). The production deep path runs with BLA OFF
  // (a measured GPU net-loss) and df64-escape OFF (measured neutral) — but a uniform-gated
  // `if` block is still COMPILED, and a GPU sets occupancy from a kernel's PEAK live-register
  // count across ALL branches, so the never-taken BLA scan + df64-escape blocks can throttle
  // occupancy (active warps) for EVERY pixel. Excluding them at shader-build time yields a LEAN
  // kernel that is BIT-IDENTICAL in production (the removed code is exactly the uBlaMaxLevel==0
  // / uDf64Esc==0 branches that never run) but may raise occupancy → faster. `bla`/`df64esc`
  // default ON (the full shader, for the BLA/df64-escape tools + validation); the renderer
  // compiles the LEAN variant for the normal deep render. Gate measured by bench-lean.mjs.
  const BLA = opts.bla !== false;       // include the BLA scan + table fetches
  const DESC = opts.df64esc !== false;  // include the df64-escape fast-path
  // SOFTWARE-PIPELINED reference prefetch (Spawn 27, bench variant): carry Z[m+1] in
  // registers so the per-iteration texelFetch is ISSUED a full iteration before its
  // first use (the escape test + the next update's ALU overlap the latency) instead
  // of being used immediately. Fetches the SAME indices (plus one extra per rebase /
  // SA seed, both rare) -> BIT-IDENTICAL output; costs 2 vec2 registers + a min().
  const PF = opts.prefetch === true;
  return `#version 300 es
precision highp float;
precision highp int;
${DF64_LIB}
${FE_LIB}
${sqrDefs(opts)}
${SMOOTH}
uniform highp sampler2D uRef;   // highp REQUIRED: a mediump sampler truncates the
uniform int uRefW, uRefLen, uMaxIter;   // texelFetch'd df64 reference to fp16 on NVIDIA
uniform vec2 uOxm, uOym, uScalem;   // fe mantissas (df64) of dc origin + per-pixel step
uniform int  uOxe, uOye, uScalee;   // matching fe exponents
uniform float uBailoutSq;
uniform float uGlitchTol;
uniform int uFastSkip;
// INTERIOR-PRUNE (experimental, measurement-only — Spawn 23). uPrune==1 makes pixels flagged
// in uPruneMask (an oracle-built interior mask: .r==1.0 means "this pixel runs to maxIter, i.e.
// is interior") BREAK the iteration loop early at n==uPruneIter instead of running to uMaxIter.
// This measures the wall-clock UPPER BOUND of interior detection (a perfect, free, zero-latency
// detector): it cuts ONLY interior pixels' tail and leaves escaping pixels untouched, so GPU
// warp divergence decides the real saving (a mixed warp still runs to its slowest escaper).
// Sweeping uPruneIter from skip→maxIter traces "if a real detector fires by iteration X, the
// frame speeds up by Y". uPrune==0 (renderer default) short-circuits the texelFetch so the loop
// is BIT-IDENTICAL to production. NOT a shipping feature — see NOTES "INTERIOR DETECTION".
uniform int uPrune;
uniform int uPruneIter;
uniform highp sampler2D uPruneMask;
${DESC ? `// uDf64Esc == 1: run the escape/rebase/glitch test in plain df64 in the common case
// (|dz| and |Z_m| both >= ~2^-100), falling back to floatexp only near a reference
// minimum. floatexp's only edge over df64 is exponent RANGE — same ~46-bit mantissa — and
// this block only runs once |dz| has grown to O(1) (fast-skip drops the tiny-dz steps),
// so df64 is bit-for-bit as precise there but uses fewer ALU ops (no per-op fe normalize).
// MEASURED PERFORMANCE-NEUTRAL (~1.00×) on SwiftShader + the real RTX 3090, so it is OPT-IN
// (renderer default 0 = the original always-floatexp path). Kept for future weak/mobile-GPU
// measurement. See NOTES "df64 ESCAPE". 0 reproduces the floatexp path bit-for-bit.
uniform int uDf64Esc;` : `// (df64-escape fast-path excluded from this LEAN build — see perturbFragRescaled opts.)`}
// Series approximation: when uSASkip > 0, seed dz at iteration uSASkip with the SA
// polynomial dz = a·u + b·u² + c·u³ (u = dc/R) instead of iterating the first uSASkip
// steps. The scaled coeffs a,b,c (computed on the CPU worker, O(|dz_skip|) ~ 2^-270 at
// deep zoom) and 1/R come in as floatexp; the Horner runs in fe so the ~2^-270 seed
// keeps full precision. uSASkip == 0 reproduces the original no-SA path bit-for-bit.
uniform int  uSASkip;
uniform vec2 uSAm[10];              // SA coeffs as fe mantissas: [ax,ay,bx,by,cx,cy,dx,dy,ex,ey]
uniform int  uSAe[10];              // matching fe exponents (order-5 series; d,e are 0 for order≤3)
uniform vec2 uInvRm; uniform int uInvRe;   // 1/R (= invR) as fe; u = dc·invR
${BLA ? `// BLA (bivariate linear approximation): jump RUNS of linear iterations via a precomputed
// table (bla.js) uploaded as a highp NEAREST RGBA32F texture. A run of L iterations from
// reference index m composes to ONE map dz_{m+L} = A·dz + B·dc, valid while |dz|² ≤ r². The
// coefficients reach far outside the float32 exponent range deep (|A|~2^228, r²~2^-576 at
// 2^-400), so each complex A,B is stored as a df64 mantissa pair under ONE shared int exponent
// and r² as a single-float mantissa + exponent — all floatexp. Entry (level l, index m) is 3
// texels at E=(l−1)·uBlaLen+m: t0 = A mantissas (Ax.hi,Ax.lo,Ay.hi,Ay.lo), t1 = B mantissas,
// t2 = (Ae,Be,r2e,r2m). uBlaMaxLevel == 0 disables BLA (the scan never runs) so the no-BLA path
// is bit-identical. highp + NEAREST are REQUIRED — a filtered/mediump fetch corrupts the df64
// coeffs (same class as the Spawn 9 mediump-sampler disaster).
uniform highp sampler2D uBla;
uniform int uBlaW, uBlaLen, uBlaMaxLevel;` : `// (BLA scan + table excluded from this LEAN build — see perturbFragRescaled opts.)`}
out vec4 frag;

const int S_ZERO = -2000000000;     // shared-exponent sentinel for dz == 0

void getZ(int m, out vec2 Zx, out vec2 Zy){
  vec4 v = texelFetch(uRef, ivec2(m % uRefW, m / uRefW), 0);
  Zx = v.xy; Zy = v.zw;
}
${BLA ? `
// The BLA table is read in two tiers to keep the per-iteration scan CHEAP on the GPU (texture
// fetches are latency-bound — a 3-texel-per-level scan every iteration made BLA SLOWER than no
// BLA). Tier 1 (getBLAr2) fetches ONLY texel 2 = (Ae, Be, r2e, r2m): the radius probe + the two
// shared exponents, one fetch. Tier 2 (getBLAab) fetches the A,B mantissa texels (0,1) ONLY when
// a level is actually chosen, using the Ae,Be already in hand. So a rejected iteration costs ONE
// fetch and a jump costs (#levels probed)+2.
vec4 getBLAr2(int l, int m){
  int t = ((l - 1) * uBlaLen + m) * 3 + 2;
  return texelFetch(uBla, ivec2(t % uBlaW, t / uBlaW), 0);   // .x=Ae .y=Be .z=r2e .w=r2m
}
void getBLAab(int l, int m, int Ae, int Be, out fe Ax, out fe Ay, out fe Bx, out fe By){
  int t = ((l - 1) * uBlaLen + m) * 3;
  vec4 a = texelFetch(uBla, ivec2(t % uBlaW, t / uBlaW), 0);
  int t1 = t + 1; vec4 b = texelFetch(uBla, ivec2(t1 % uBlaW, t1 / uBlaW), 0);
  Ax = fe_norm(a.xy, Ae); Ay = fe_norm(a.zw, Ae);     // shared exponent per complex coeff
  Bx = fe_norm(b.xy, Be); By = fe_norm(b.zw, Be);
}
// r² (a non-negative fe) from a texel-2 fetch; r2m == 0 ⇒ unusable (run overflowed or |Z|≥2).
fe blaR2(vec4 e){ if (e.w == 0.0) return fe(vec2(0.0), FE_ZERO_E); return fe(vec2(e.w, 0.0), int(e.z)); }` : ''}

void main(){
  // dc = origin + frag·step (in fe), then collapse to one shared exponent Sc.
  fe sc = fe(uScalem, uScalee);
  fe dcxf = fe_add(fe(uOxm, uOxe), fe_mul(fe_fromf(gl_FragCoord.x), sc));
  fe dcyf = fe_add(fe(uOym, uOye), fe_mul(fe_fromf(gl_FragCoord.y), sc));
  int Sc = max(dcxf.e, dcyf.e);
  vec2 Cx = ds_scale2(dcxf.m, dcxf.e - Sc);   // dc = (Cx,Cy)·2^Sc
  vec2 Cy = ds_scale2(dcyf.m, dcyf.e - Sc);

  vec2 Dx = vec2(0.0), Dy = vec2(0.0);        // dz = (Dx,Dy)·2^S
  int S = S_ZERO;
  int m = 0, n = 0;
  float glitch = 0.0;
  vec2 Zx, Zy; getZ(0, Zx, Zy);
${PF ? `  vec2 Znx, Zny; getZ(min(1, uRefLen), Znx, Zny);   // prefetched Z[m+1]` : ''}
${BLA ? `  // |dz|² carried across iterations for the BLA early-out — the escape block already computes it
  // (dz2) every step, so the BLA scan reuses it instead of recomputing fe_norm+fe_mul each
  // iteration. Identical value (same Dx,Dy,S), it just avoids the redundant work on the hot path.
  fe curMag2 = fe(vec2(0.0), FE_ZERO_E);` : ''}

  // Series-approximation seed: jump dz to its value at iteration uSASkip. Identical math
  // to the CPU escapePerturb seed (series.js): complex Horner ((c·u+b)·u+a)·u with u=dc·invR,
  // all in fe. The true value is z = Z[skip] + dz, so m = n = uSASkip (no rebase has occurred
  // yet — SA is only valid in the pre-rebase linear regime). The escape loop then continues
  // from there exactly as if it had iterated 0..skip.
  if (uSASkip > 0) {
    fe ux = fe_mul(dcxf, fe(uInvRm, uInvRe));
    fe uy = fe_mul(dcyf, fe(uInvRm, uInvRe));
    // Order-5 Horner ((((e·u + d)·u + c)·u + b)·u + a)·u; d,e are 0 for an order≤3 sa, so
    // this reduces to the historical cubic seed bit-for-bit (extra fe ops on zero coeffs).
    fe hx = fe(uSAm[8], uSAe[8]), hy = fe(uSAm[9], uSAe[9]);   // h = e
    fe rx, ry;
    // h = h·u + d
    rx = fe_sub(fe_mul(hx, ux), fe_mul(hy, uy));
    ry = fe_add(fe_mul(hx, uy), fe_mul(hy, ux));
    hx = fe_add(rx, fe(uSAm[6], uSAe[6])); hy = fe_add(ry, fe(uSAm[7], uSAe[7]));
    // h = h·u + c
    rx = fe_sub(fe_mul(hx, ux), fe_mul(hy, uy));
    ry = fe_add(fe_mul(hx, uy), fe_mul(hy, ux));
    hx = fe_add(rx, fe(uSAm[4], uSAe[4])); hy = fe_add(ry, fe(uSAm[5], uSAe[5]));
    // h = h·u + b
    rx = fe_sub(fe_mul(hx, ux), fe_mul(hy, uy));
    ry = fe_add(fe_mul(hx, uy), fe_mul(hy, ux));
    hx = fe_add(rx, fe(uSAm[2], uSAe[2])); hy = fe_add(ry, fe(uSAm[3], uSAe[3]));
    // h = h·u + a
    rx = fe_sub(fe_mul(hx, ux), fe_mul(hy, uy));
    ry = fe_add(fe_mul(hx, uy), fe_mul(hy, ux));
    hx = fe_add(rx, fe(uSAm[0], uSAe[0])); hy = fe_add(ry, fe(uSAm[1], uSAe[1]));
    // dz = h·u
    fe dzx = fe_sub(fe_mul(hx, ux), fe_mul(hy, uy));
    fe dzy = fe_add(fe_mul(hx, uy), fe_mul(hy, ux));
    int Sz = max(dzx.e, dzy.e);
    if (dzx.m.x == 0.0 && dzy.m.x == 0.0) { Dx = vec2(0.0); Dy = vec2(0.0); S = S_ZERO; }
    else { Dx = ds_scale2(dzx.m, dzx.e - Sz); Dy = ds_scale2(dzy.m, dzy.e - Sz); S = Sz; }
    m = uSASkip; n = uSASkip;
    getZ(m, Zx, Zy);
${PF ? `    getZ(min(m + 1, uRefLen), Znx, Zny);` : ''}
${BLA ? `    curMag2 = fe_add(FE_SQR(dzx), FE_SQR(dzy));   // seed |dz|² for the first BLA check` : ''}
  }
  // Interior-prune flag (Spawn 23, measurement-only): fetched ONCE here, not per-iteration. The
  // && short-circuits when uPrune==0, so production pays no fetch and stays bit-identical.
  bool pruneInt = (uPrune == 1) && (texelFetch(uPruneMask, ivec2(gl_FragCoord.xy), 0).r > 0.5);
  for (int i = 0; i < 100000000; i++) {
    if (n >= uMaxIter) break;
    if (pruneInt && n >= uPruneIter) break;   // interior-prune: bail this oracle-interior pixel early
${BLA ? `    // ---- BLA: jump the largest valid run of linear iterations (else fall through to a
    // single true step). Identical decision logic to the CPU bla loop (escapePerturb):
    // scan levels high→low; a level-l run (L=2^l) applies when |dz|² ≤ r²[l][m] AND it
    // overshoots neither the reference (m+L ≤ len−1, so the post-jump z=Z_{m+L} is in range)
    // nor the iteration cap (n+L ≤ maxIter). The jump dz' = A·dz + B·dc runs in floatexp
    // (A,B reach 2^±hundreds deep), then collapses back to the shared-exponent (Dx,Dy,S) form.
    // uBlaMaxLevel == 0 (no table) skips this entirely → the no-BLA path is unchanged. dz must
    // be nonzero: at m=0 (Z_0=0) every level has r²=0 anyway, so this never fires there.
    if (uBlaMaxLevel > 0 && S != S_ZERO) {
      // CHEAP EARLY-OUT: r² is monotone non-increasing in level (a merge can only shrink the
      // validity radius), so level 1 has the LARGEST radius at this m. One probe against the
      // carried |dz|²: if even level 1 can't cover it, no level can — reject with a single fetch
      // (the common chaotic case after SA/rebase, where dz is large). This + the carried magnitude
      // + the binary search below are what make BLA cheap enough to matter on the GPU.
      vec4 e1 = getBLAr2(1, m);
      if (e1.w != 0.0 && !fe_lt(blaR2(e1), curMag2)) {
        // Some jump is possible. Find the LARGEST level whose run is valid — radius covers |dz|²
        // (r² monotone ↓ in level) AND bounds fit (m+L ≤ len−1, n+L ≤ maxIter; L=2^l, also
        // monotone). Both predicates are monotone in l, so valid(l) is a step function — BINARY
        // SEARCH it (~4 probes, branch-coherent) instead of a high→low linear scan (~maxLevel
        // probes with per-lane divergence on the chosen level — that scan made deep BLA erratic).
        int bestL = 0; vec4 be = e1;
        int loL = 1, hiL = uBlaMaxLevel;
        for (int it = 0; it < 5; it++) {              // ceil(log2(maxLevel)) ≤ 5 for maxLevel ≤ 32
          if (loL > hiL) break;
          int mid = (loL + hiL) / 2, L = 1 << mid;
          vec4 el = (mid == 1) ? e1 : getBLAr2(mid, m);
          bool ok = (m + L <= uRefLen - 1) && (n + L <= uMaxIter) && (el.w != 0.0) && !fe_lt(blaR2(el), curMag2);
          if (ok) { bestL = mid; be = el; loL = mid + 1; }
          else hiL = mid - 1;
        }
        if (bestL > 0) {
          int L = 1 << bestL;
          fe dxf0 = fe_norm(Dx, S), dyf0 = fe_norm(Dy, S);   // dz as fe — only needed on a jump
          fe Ax, Ay, Bx, By;
          getBLAab(bestL, m, int(be.x), int(be.y), Ax, Ay, Bx, By);
          // dz' = A·dz + B·dc  (complex; A,B,dz,dc all fe). dc = (dcxf,dcyf) in scope.
          fe ndxf = fe_add(fe_sub(fe_mul(Ax, dxf0), fe_mul(Ay, dyf0)),
                           fe_sub(fe_mul(Bx, dcxf), fe_mul(By, dcyf)));
          fe ndyf = fe_add(fe_add(fe_mul(Ax, dyf0), fe_mul(Ay, dxf0)),
                           fe_add(fe_mul(Bx, dcyf), fe_mul(By, dcxf)));
          curMag2 = fe_add(FE_SQR(ndxf), FE_SQR(ndyf));   // |dz'|² for the next check
          int Sz = max(ndxf.e, ndyf.e);
          if (ndxf.m.x == 0.0 && ndyf.m.x == 0.0) { Dx = vec2(0.0); Dy = vec2(0.0); S = S_ZERO; }
          else { Dx = ds_scale2(ndxf.m, ndxf.e - Sz); Dy = ds_scale2(ndyf.m, ndyf.e - Sz); S = Sz; }
          m += L; n += L;
          getZ(m, Zx, Zy);
${PF ? `          getZ(min(m + 1, uRefLen), Znx, Zny);` : ''}
          continue;
        }
      }
    }` : ''}
    // ---- rescaled delta update: dz' = 2·Z·dz + dz² + dc ----
    if (S == S_ZERO) {                        // dz == 0  ->  dz' = dc
      Dx = Cx; Dy = Cy; S = Sc;
    } else {
      vec2 lx = ds_sub(ds_mul(Zx, Dx), ds_mul(Zy, Dy));   // Zx·Dx − Zy·Dy
      vec2 ly = ds_add(ds_mul(Zx, Dy), ds_mul(Zy, Dx));   // Zx·Dy + Zy·Dx
      lx = ds_scale2(lx, 1); ly = ds_scale2(ly, 1);       // ·2  (linear ≈ 2·Z·dz, exponent S)
      // The combine frame W must reflect each term's TRUE exponent. The linear term's
      // is eL = S + (its mantissa's ilogb): using S alone lets the un-normalized linear
      // (|2·Z·D| up to ~6) inflate the frame and cost the dc/dz² addends ~2-3 low bits
      // vs the floatexp engine. The dz² exponent qe = 2S must also be in W: when the
      // reference Z_m ≈ 0 (exactly so at m=0 after every rebase, since Z_0=0) the linear
      // vanishes and dz² is DOMINANT (dz' = dz² + dc); leaving qe out lets a vanished
      // linear (ilogb1(0) = -126) pick a bogus frame that mis-scales dz², so the orbit
      // never escapes. (Scaling lx by 2^(S-W) folds the normalization in — no separate step.)
      int eL = S + max(ilogb1(lx.x), ilogb1(ly.x));
      int qe = S + S;                                     // dz² exponent
      int W = max(max(eL, qe), Sc);
      // Accumulate (linear + dz²) + dc, matching the floatexp engine's add order.
      vec2 ax = ds_scale2(lx, S - W);
      vec2 ay = ds_scale2(ly, S - W);
      if (qe - W > -52) {                                 // else dz² negligible (as fe_add)
        vec2 qx = ds_sub(DS_SQR(Dx), DS_SQR(Dy));
        vec2 qy = ds_scale2(ds_mul(Dx, Dy), 1);           // 2·Dx·Dy
        ax = ds_add(ax, ds_scale2(qx, qe - W));
        ay = ds_add(ay, ds_scale2(qy, qe - W));
      }
      ax = ds_add(ax, ds_scale2(Cx, Sc - W));
      ay = ds_add(ay, ds_scale2(Cy, Sc - W));
      if (ax.x == 0.0 && ay.x == 0.0) { Dx = vec2(0.0); Dy = vec2(0.0); S = S_ZERO; }
      else {
        int k = max(ilogb1(ax.x), ilogb1(ay.x));          // larger component -> [0.5,1)
        Dx = ds_scale2(ax, -k); Dy = ds_scale2(ay, -k); S = W + k;
      }
    }
    m++; n++;
${PF ? `    Zx = Znx; Zy = Zny;                       // Z[m] — fetched one iteration ago
    getZ(min(m + 1, uRefLen), Znx, Zny);      // prefetch Z[m+1]; used next iteration` : `    getZ(m, Zx, Zy);`}
    // Fast skip (same proof as the fe engine; |dz| < 2^(S+1), |Z_m| >= 2^(ezm-1)).${BLA ? ` Disabled when
    // BLA is on: BLA's early-out already handles the tiny-dz case (better — it JUMPS the run), and
    // skipping the escape block here would leave the carried curMag2 stale for the next BLA probe.` : ''}
    if (uFastSkip == 1 ${BLA ? `&& uBlaMaxLevel == 0 ` : ``}&& m != uRefLen) {
      int ezm = max(ilogb1(Zx.x), ilogb1(Zy.x));
      if (ezm <= 6 && ezm >= S + 4) continue;
    }
${DESC ? `    // ---- escape / rebase / glitch test ----
    // z = Z_m + dz. The rescaled UPDATE above is df64-cheap, so this test is the dominant
    // per-iteration cost on chaotic deep views — and it ran in floatexp every iteration.
    // But this block only runs once |dz| has grown to O(1) (the fast-skip drops the tiny-dz
    // steps, regardless of zoom depth), so |dz| is comfortably inside df64's float32 exponent
    // range. floatexp's only advantage there is range, NOT precision (same ~46-bit mantissa),
    // so we run the test in plain df64 — bit-for-bit as precise, fewer ALU ops — and fall back
    // to floatexp only when |dz| or |Z_m| < ~2^-100 (near a reference minimum, where the
    // |z|<|dz| rebase compare is genuinely range-critical and ds_scale2 would underflow).
    if (uDf64Esc == 1 ${BLA ? `&& uBlaMaxLevel == 0 ` : ``}&& S != S_ZERO && S >= -100) {
      int ezm = max(ilogb1(Zx.x), ilogb1(Zy.x));
      if (ezm >= -100) {                          // Z_m normal (not a deep reference minimum)
        vec2 dx = ds_scale2(Dx, S);               // dz at its true scale (exact for S >= -100)
        vec2 dy = ds_scale2(Dy, S);
        vec2 zfx = ds_add(Zx, dx);                // z = Z_m + dz  (all O(1) df64)
        vec2 zfy = ds_add(Zy, dy);
        float mag2 = ds_tofloat(ds_add(DS_SQR(zfx), DS_SQR(zfy)));
        if (mag2 > uBailoutSq) {
          frag = vec4(smoothCount(float(n), mag2), float(n), glitch, 1.0);
          return;
        }
        float dz2 = ds_tofloat(ds_add(DS_SQR(dx), DS_SQR(dy)));
        bool rebase = (mag2 < dz2) || (m == uRefLen);
        if (uGlitchTol > 0.0 && !rebase) {
          float zref2 = ds_tofloat(ds_add(DS_SQR(Zx), DS_SQR(Zy)));
          if (zref2 > 0.0 && mag2 < uGlitchTol * zref2) glitch = 1.0;
        }
        if (rebase) {                             // dz = z; re-normalize to shared (Dx,Dy,S)
          int Sz = max(ilogb1(zfx.x), ilogb1(zfy.x));
          if (zfx.x == 0.0 && zfy.x == 0.0) { Dx = vec2(0.0); Dy = vec2(0.0); S = S_ZERO; }
          else { Dx = ds_scale2(zfx, -Sz); Dy = ds_scale2(zfy, -Sz); S = Sz; }
          m = 0; getZ(0, Zx, Zy);
${PF ? `          getZ(min(1, uRefLen), Znx, Zny);` : ''}
        }
        continue;
      }
    }
` : ``}    // ---- exact escape / rebase in floatexp (identical logic to perturbFragFloatexp) ----
    fe dxf = fe_norm(Dx, S);     // S==S_ZERO -> Dx==0 -> fe zero (e ignored)
    fe dyf = fe_norm(Dy, S);
    fe zfx = fe_add(fe_fromds(Zx), dxf);
    fe zfy = fe_add(fe_fromds(Zy), dyf);
    fe mag2 = fe_add(FE_SQR(zfx), FE_SQR(zfy));
    float mag2f = fe_tof(mag2);
    if (mag2f > uBailoutSq) {
      frag = vec4(smoothCount(float(n), mag2f), float(n), glitch, 1.0);
      return;
    }
    fe dz2 = fe_add(FE_SQR(dxf), FE_SQR(dyf));
${BLA ? `    curMag2 = dz2;               // carry |dz|² for the next iteration's BLA early-out (non-rebase)` : ``}
    bool rebase = fe_lt(mag2, dz2) || (m == uRefLen);
    if (uGlitchTol > 0.0 && !rebase) {
      fe zr = fe_add(FE_SQR(fe_fromds(Zx)), FE_SQR(fe_fromds(Zy)));
      if (zr.m.x != 0.0 && fe_lt(mag2, fe_mul(fe_fromf(uGlitchTol), zr))) glitch = 1.0;
    }
    if (rebase) {                // dz = z; re-collapse z (fe per component) to shared form
      int Sz = max(zfx.e, zfy.e);
      if (zfx.m.x == 0.0 && zfy.m.x == 0.0) { Dx = vec2(0.0); Dy = vec2(0.0); S = S_ZERO; }
      else { Dx = ds_scale2(zfx.m, zfx.e - Sz); Dy = ds_scale2(zfy.m, zfy.e - Sz); S = Sz; }
      m = 0; getZ(0, Zx, Zy);
${PF ? `      getZ(min(1, uRefLen), Znx, Zny);` : ''}
${BLA ? `      curMag2 = mag2;            // after rebase dz = z, so |dz|² = |z|² = mag2 (already computed)` : ``}
    }
  }
  frag = vec4(-1.0, float(n), glitch, 1.0);
}
`;
}

// Color pass: sample the smooth-count texture + a 1-D palette LUT -> RGBA8 canvas.
// Mirrors palette.colorFor: t = sn/uCycle + uShift; rgb = LUT(fract(t)); sn<0 -> interior.
//
// Supersampling: the sn texture is rendered at uSS× the output resolution. Each
// output pixel box-averages the COLORS of its uSS×uSS subsamples (averaging the
// final RGB, not the cyclic smooth-count sn — averaging sn would bleed hues where
// the palette wraps). uSS=1 reduces to a plain point sample (old behavior).
export const COLOR_FRAG = `#version 300 es
precision highp float;
precision highp int;
uniform highp sampler2D uSn; // RGBA32F (compute res = uSS * output): .r = smooth count
uniform sampler2D uPalette;  // 1xN RGBA8 gradient (LINEAR, REPEAT)
uniform float uCycle;
uniform float uShift;
uniform vec3 uInterior;
uniform int uSS;             // supersample factor (>= 1)
uniform int uShowGlitch;     // 0 = off; 1 = tint Pauldelbrot-flagged (.b) pixels (debug overlay)
out vec4 frag;

vec3 colorOf(float sn){
  if (sn < 0.0) return uInterior;          // interior / did-not-escape
  float t = sn / uCycle + uShift;
  return texture(uPalette, vec2(fract(t), 0.5)).rgb;
}

void main(){
  // Flip Y: the iteration shaders write sn bottom-up; output pixel (xo,yo) (GL,
  // bottom-up) maps to the compute block whose top row is csize.y-(yo+1)*uSS.
  // For uSS=1 this is csize.y-1-yo, matching the original point-sampled mapping.
  ivec2 csize = textureSize(uSn, 0);       // (cW, cH) compute resolution
  int bx = int(gl_FragCoord.x) * uSS;
  int by = csize.y - (int(gl_FragCoord.y) + 1) * uSS;
  vec3 acc = vec3(0.0);
  float glitch = 0.0;                       // OR the .b glitch flag across the ss×ss block
  for (int sy = 0; sy < uSS; sy++) {
    for (int sx = 0; sx < uSS; sx++) {
      vec4 t = texelFetch(uSn, ivec2(bx + sx, by + sy), 0);
      acc += colorOf(t.r);
      glitch = max(glitch, t.b);
    }
  }
  vec3 col = acc / float(uSS * uSS);
  // Debug overlay: blend flagged pixels toward magenta (the fractal stays visible
  // underneath). Only the .b channel is read; the rendered color is unchanged when off.
  if (uShowGlitch != 0 && glitch > 0.5) col = mix(col, vec3(1.0, 0.0, 1.0), 0.6);
  frag = vec4(col, 1.0);
}
`;
