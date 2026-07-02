// probe-barrier.mjs — find the MINIMAL ob() optimization-barrier placement that
// keeps df64 (double-single) arithmetic INTACT on the real GPU.
//
// Background: NVIDIA's ANGLE compiler breaks df64 two ways — (1) FP reassociation
// `ca-(ca-x) -> x` collapsing the Veltkamp split, (2) FMA contraction `a*b±c ->
// fma(...)` defeating the Dekker error terms. The shipping fix wraps EVERY
// intermediate in an opaque XOR barrier ob() (~24/mul, ~8/add), measured at ~2.6x.
// Spawn 19 found the barriered ds_mul squarings are the dominant per-iteration GPU
// cost. THIS probe searches for a cheaper placement that still passes (relerr<1e-10).
//
// Method (isolated, NO texture — so the Spawn-9 mediump-sampler bug is not involved):
// compile many ds_mul / ds_add variants, feed realistic df64 inputs (nonzero lo),
// compare hi+lo against the double-precision truth. INTACT -> ~1e-13..1e-16;
// COLLAPSED-to-float32 -> ~1e-7. Also runs a CHAINED accumulation (Horner-like) to
// surface collapse that a single op might hide. Reports worst relerr + barrier count.
//
//   node tools/probe-barrier.mjs            # SwiftShader (always intact — control)
//   GPU=1 node tools/probe-barrier.mjs      # real GPU (ANGLE/Vulkan) — the real test
import { chromium } from '@playwright/test';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

// ---------- df64 op variants, as GLSL source fragments ----------
// obB(x) = XOR-with-opaque-zero barrier (defined in the shader preamble).
// Each variant defines mul_<NAME>(vec2,vec2) and/or add_<NAME>(vec2,vec2).

const MULS = {
  // current shipping placement (24 barriers) — the INTACT reference.
  full: { n: 24, src: `
vec2 mul_full(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p  = obB(a.x * b.x);
  float ca = obB(SPLIT * a.x); float a_hi = obB(ca - obB(ca - a.x)); float a_lo = obB(a.x - a_hi);
  float cb = obB(SPLIT * b.x); float b_hi = obB(cb - obB(cb - b.x)); float b_lo = obB(b.x - b_hi);
  float e = obB(obB(obB(obB(obB(a_hi*b_hi) - p) + obB(a_hi*b_lo)) + obB(a_lo*b_hi)) + obB(a_lo*b_lo));
  e = obB(e + obB(obB(a.x*b.y) + obB(a.y*b.x)));
  float hi = obB(p + e);
  float lo = obB(e - obB(hi - p));
  return vec2(hi, lo);
}` },
  // minimal (=probe-df64's C): split inner-sub + final renorm only. Blocks reassoc
  // but NOT the FMA contraction of a_hi*b_hi-p. Known to COLLAPSE — kept as a control.
  minC: { n: 3, src: `
vec2 mul_minC(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p = a.x * b.x;
  float ca = SPLIT * a.x; float a_hi = ca - obB(ca - a.x); float a_lo = a.x - a_hi;
  float cb = SPLIT * b.x; float b_hi = cb - obB(cb - b.x); float b_lo = b.x - b_hi;
  float e = ((a_hi*b_hi - p) + a_hi*b_lo + a_lo*b_hi) + a_lo*b_lo;
  e += a.x*b.y + a.y*b.x;
  float hi = p + e;
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}` },
  // LEAN-8: barrier only the precision-critical points —
  //  p (rounded product, subtracted), ca/cb (split mul, rounds before sub),
  //  ca-a.x/cb-b.x (split reassoc), a_hi*b_hi (the FMA-into-(-p) killer),
  //  p+e (so hi materialized) + hi-p (renorm reassoc). Small products left to fuse.
  lean8: { n: 8, src: `
vec2 mul_lean8(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p  = obB(a.x * b.x);
  float ca = obB(SPLIT * a.x); float a_hi = ca - obB(ca - a.x); float a_lo = a.x - a_hi;
  float cb = obB(SPLIT * b.x); float b_hi = cb - obB(cb - b.x); float b_lo = b.x - b_hi;
  float e = (obB(a_hi*b_hi) - p) + a_hi*b_lo + a_lo*b_hi + a_lo*b_lo;
  e += a.x*b.y + a.y*b.x;
  float hi = obB(p + e);
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}` },
  // LEAN-10: lean8 + materialize a_hi,b_hi (block a_lo=a.x-a_hi reassoc through the split).
  lean10: { n: 10, src: `
vec2 mul_lean10(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p  = obB(a.x * b.x);
  float ca = obB(SPLIT * a.x); float a_hi = obB(ca - obB(ca - a.x)); float a_lo = a.x - a_hi;
  float cb = obB(SPLIT * b.x); float b_hi = obB(cb - obB(cb - b.x)); float b_lo = b.x - b_hi;
  float e = (obB(a_hi*b_hi) - p) + a_hi*b_lo + a_lo*b_hi + a_lo*b_lo;
  e += a.x*b.y + a.y*b.x;
  float hi = obB(p + e);
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}` },
  // LEAN-14: barrier every product that feeds a +/- (block ALL FMA) + reassoc points,
  // but NOT the additive-chain intermediates (let the compiler schedule the sums).
  lean14: { n: 14, src: `
vec2 mul_lean14(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p  = obB(a.x * b.x);
  float ca = obB(SPLIT * a.x); float a_hi = obB(ca - obB(ca - a.x)); float a_lo = a.x - a_hi;
  float cb = obB(SPLIT * b.x); float b_hi = obB(cb - obB(cb - b.x)); float b_lo = b.x - b_hi;
  float e = (obB(a_hi*b_hi) - p) + obB(a_hi*b_lo) + obB(a_lo*b_hi) + obB(a_lo*b_lo);
  e += obB(a.x*b.y) + obB(a.y*b.x);
  float hi = obB(p + e);
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}` },
  // CONTROL: lean8 WITHOUT the obB(p) barrier — p is then a bare product a.x*b.x,
  // so `obB(a_hi*b_hi) - a.x*b.x` can fma-contract. Should COLLAPSE -> proves p needs it.
  lean8_nop: { n: 7, src: `
vec2 mul_lean8_nop(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p  = a.x * b.x;
  float ca = obB(SPLIT * a.x); float a_hi = ca - obB(ca - a.x); float a_lo = a.x - a_hi;
  float cb = obB(SPLIT * b.x); float b_hi = cb - obB(cb - b.x); float b_lo = b.x - b_hi;
  float e = (obB(a_hi*b_hi) - p) + a_hi*b_lo + a_lo*b_hi + a_lo*b_lo;
  e += a.x*b.y + a.y*b.x;
  float hi = obB(p + e);
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}` },
};

const ADDS = {
  // current shipping placement (8 barriers) — INTACT reference. No multiplies, so the
  // only risk is reassociation (TwoSum cancellations + (s+e)-s renorm).
  full: { n: 8, src: `
vec2 add_full(vec2 a, vec2 b){
  float s  = obB(a.x + b.x);
  float v  = obB(s - a.x);
  float e  = obB(obB(a.x - obB(s - v)) + obB(b.x - v));
  e = obB(e + obB(a.y + b.y));
  float hi = obB(s + e);
  float lo = obB(e - obB(hi - s));
  return vec2(hi, lo);
}` },
  // LEAN-5: barrier only the reassociation cancellation points —
  //  s (block v=s-a.x -> b.x), s-v (block a.x-(s-v) -> ...), s+e (hi materialized),
  //  hi-s (renorm reassoc). v itself left transparent.
  lean5: { n: 5, src: `
vec2 add_lean5(vec2 a, vec2 b){
  float s  = obB(a.x + b.x);
  float v  = s - a.x;
  float e  = (a.x - obB(s - v)) + (b.x - v);
  e = e + (a.y + b.y);
  float hi = obB(s + e);
  float lo = e - obB(hi - s);
  return vec2(hi, lo);
}` },
  // LEAN-6: lean5 + barrier v (block s-v -> a.x reassoc more firmly).
  lean6: { n: 6, src: `
vec2 add_lean6(vec2 a, vec2 b){
  float s  = obB(a.x + b.x);
  float v  = obB(s - a.x);
  float e  = (a.x - obB(s - v)) + (b.x - v);
  e = e + (a.y + b.y);
  float hi = obB(s + e);
  float lo = e - obB(hi - s);
  return vec2(hi, lo);
}` },
  // CONTROL: no barriers at all — should COLLAPSE on NVIDIA.
  none: { n: 0, src: `
vec2 add_none(vec2 a, vec2 b){
  float s  = a.x + b.x;
  float v  = s - a.x;
  float e  = (a.x - (s - v)) + (b.x - v);
  e = e + (a.y + b.y);
  float hi = s + e;
  float lo = e - (hi - s);
  return vec2(hi, lo);
}` },
};

// ---------- ds_sqr variants (Spawn 27) ----------
// A dedicated df64 SQUARING: Dekker/Veltkamp needs only ONE split for a*a (the
// second operand's split is identical), and the cross terms collapse
// (a_hi*b_lo + a_lo*b_hi -> 2*a_hi*a_lo; a.x*b.y + a.y*b.x -> 2*a.x*a.y).
// ~6 of the perturbation loop's ~11 per-iteration ds_mul are self-products, so if
// the compiler does NOT already CSE the duplicate split in ds_mul(a,a), this is a
// uniform per-iteration ALU cut. Controls prove the kept barriers are load-bearing.
const SQRS = {
  // current behavior: the shipping lean8 mul applied to (a,a). The reference.
  mulref: { n: 8, src: `
vec2 sqr_mulref(vec2 a){ return mul_lean8(a, a); }` },
  // the candidate: one split, 6 barriers (same placement logic as lean8 —
  // p, split mul, split inner-sub, the fma-critical a_hi*a_hi, p+e, hi-p).
  sqr6: { n: 6, src: `
vec2 sqr_sqr6(vec2 a){
  const float SPLIT = 4097.0;
  float p  = obB(a.x * a.x);
  float ca = obB(SPLIT * a.x); float a_hi = ca - obB(ca - a.x); float a_lo = a.x - a_hi;
  float e = (obB(a_hi*a_hi) - p) + 2.0*(a_hi*a_lo) + a_lo*a_lo;
  e += 2.0*(a.x*a.y);
  float hi = obB(p + e);
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}` },
  // CONTROL: no p barrier — a_hi*a_hi - p can fma-contract. Expect degraded/collapsed.
  sqr6_nop: { n: 5, src: `
vec2 sqr_sqr6_nop(vec2 a){
  const float SPLIT = 4097.0;
  float p  = a.x * a.x;
  float ca = obB(SPLIT * a.x); float a_hi = ca - obB(ca - a.x); float a_lo = a.x - a_hi;
  float e = (obB(a_hi*a_hi) - p) + 2.0*(a_hi*a_lo) + a_lo*a_lo;
  e += 2.0*(a.x*a.y);
  float hi = obB(p + e);
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}` },
  // CONTROL: no barriers at all. Expect COLLAPSED.
  sqr_none: { n: 0, src: `
vec2 sqr_sqr_none(vec2 a){
  const float SPLIT = 4097.0;
  float p  = a.x * a.x;
  float ca = SPLIT * a.x; float a_hi = ca - (ca - a.x); float a_lo = a.x - a_hi;
  float e = (a_hi*a_hi - p) + 2.0*(a_hi*a_lo) + a_lo*a_lo;
  e += 2.0*(a.x*a.y);
  float hi = p + e;
  float lo = e - (hi - p);
  return vec2(hi, lo);
}` },
};

// Squaring frag: uMode 0 -> single sqr(a); uMode 2 -> chained t <- sqr(t) + b.
// The chained quadratic map MUST run in an ATTRACTING regime (see input list):
// a chaotic c would amplify even the legitimate df64-vs-double representation gap
// (~2^-49/step, doubling per iteration) past the collapse gate — a false COLLAPSED.
// In an attracting orbit errors CONTRACT, so intact (~1e-14) and collapsed
// (~per-step-2^-24 / (1-contraction) ~ 4e-7) stay cleanly separated.
function buildSqrFrag(sqrName, addName) {
  return `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform vec4 uIn;   // mode 0: a=(uIn.xy) df64 ; mode 2: t0=(uIn.xy), c=(uIn.zw)
uniform int uMode;
uniform int uZero;  // == 0, opaque
float obB(float x){ return intBitsToFloat(floatBitsToInt(x) ^ uZero); }
${MULS.lean8.src}
${SQRS[sqrName].src}
${ADDS[addName].src}
#define SQR(a) sqr_${sqrName}(a)
#define ADD(a,b) add_${addName}(a,b)
void main(){
  vec2 a = uIn.xy, b = uIn.zw;
  vec2 r;
  if (uMode == 0) r = SQR(a);
  else {
    vec2 t = a;
    for (int i = 0; i < 24; i++) t = ADD(SQR(t), b);
    r = t;
  }
  outColor = vec4(r.x, r.y, 0.0, 1.0);
}`;
}

function truthSqr(inp, mode) {
  const a = inp[0] + inp[1], b = inp[2] + inp[3];
  if (mode === 0) return a * a;
  let t = a;
  for (let i = 0; i < 24; i++) t = t * t + b;
  return t;
}

function buildFrag(mulName, addName) {
  return `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform vec4 uIn;   // a=(uIn.x,uIn.y) df64, b=(uIn.z,uIn.w) df64
uniform int uMode;  // 0 = single mul, 1 = single add, 2 = chained (Horner-ish)
uniform int uZero;  // == 0, opaque
float obB(float x){ return intBitsToFloat(floatBitsToInt(x) ^ uZero); }
${MULS[mulName].src}
${ADDS[addName].src}
#define MUL(a,b) mul_${mulName}(a,b)
#define ADD(a,b) add_${addName}(a,b)
void main(){
  vec2 a = uIn.xy, b = uIn.zw;
  vec2 r;
  if (uMode == 0) r = MUL(a, b);
  else if (uMode == 1) r = ADD(a, b);
  else {
    // chained: t <- t*a + b, 24 times (stresses accumulation like the perturb loop).
    vec2 t = a;
    for (int i = 0; i < 24; i++) t = ADD(MUL(t, a), b);
    r = t;
  }
  outColor = vec4(r.x, r.y, 0.0, 1.0);
}`;
}

// ---------- run one fragment, return [hi,lo] for given inputs+mode ----------
const RUN = ({ FRAG, inputs, mode }) => {
  const c = document.createElement('canvas'); c.width = 1; c.height = 1;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl) return { err: 'no webgl2' };
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  function sh(t, s) {
    const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
    if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('compile: ' + gl.getShaderInfoLog(o));
    return o;
  }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, 1, 1);
  const uIn = gl.getUniformLocation(prog, 'uIn');
  const uMode = gl.getUniformLocation(prog, 'uMode');
  const uZero = gl.getUniformLocation(prog, 'uZero');
  gl.uniform1i(uZero, 0);
  gl.uniform1i(uMode, mode);
  const out = [];
  for (const inp of inputs) {
    gl.uniform4f(uIn, inp[0], inp[1], inp[2], inp[3]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    out.push([px[0], px[1]]);
  }
  return { renderer, out };
};

// ---------- build df64 inputs (hi,lo proper split of a double) ----------
const fr = Math.fround;
function df(value) { const hi = fr(value); const lo = fr(value - hi); return [hi, lo]; }
// truth value of a df64 [hi,lo] as a double (exact: both are float32)
function val(d) { return d[0] + d[1]; }

// input pairs (a,b) chosen O(1) — the regime where the escape block runs — with
// nonzero lo words so the cross-terms matter; includes squarings (a==b).
const rawPairs = [
  [1.0 + Math.pow(2, -13), 1.0 + Math.pow(2, -13)],
  [1.2999999523162842 + 1e-9, 0.800000011920929 - 1e-9],
  [Math.PI / 2, Math.E / 2],
  [1.0000001192092896, 1.0000002384185791],
  [-1.4142135623730951, 1.7320508075688772],
  [0.6180339887498949, 0.6180339887498949],   // squaring
  [1.9318516525781366, 1.9318516525781366],    // squaring near 2
  [-0.7071067811865476, -0.7071067811865476],  // squaring
  [1.0 + Math.pow(2, -22), 1.0 - Math.pow(2, -22)],
  [0.31622776601683794, 3.1622776601683795],
];
const inputs = rawPairs.map(([a, b]) => { const A = df(a), B = df(b); return [A[0], A[1], B[0], B[1]]; });

// chained-sqr inputs: (t0, c) with c in an ATTRACTING window of z <- z^2 + c
// (|f'| < 1 at the cycle), t0 in the basin. Nonzero lo words via irrational values.
const sqrChainPairs = [
  [0.3141592653589793, -0.5],                    // attracting fixed point ~ -0.366
  [0.1234567890123456, 0.2],                     // attracting fixed point ~ 0.276
  [0.0, -0.9],                                   // attracting 2-cycle
  [0.5772156649015329, -1.0 + Math.pow(2, -13)], // 2-cycle, near-superattracting
];
const sqrChainInputs = sqrChainPairs.map(([t, c]) => { const T = df(t), C = df(c); return [T[0], T[1], C[0], C[1]]; });

// truth: mode 0 -> a*b ; mode 1 -> a+b ; mode 2 -> chained t<-t*a+b 24x
function truthFor(inp, mode) {
  const a = inp[0] + inp[1], b = inp[2] + inp[3];
  if (mode === 0) return a * b;
  if (mode === 1) return a + b;
  let t = a;
  for (let i = 0; i < 24; i++) t = t * a + b;
  return t;
}

const VERD = (w) => (w < 1e-10 ? 'INTACT  ' : 'COLLAPSED');

// ---------- TIMING harness (TIME=1): fair per-op speed A/B on the real GPU ----------
// Each fragment runs a FIXED chained loop (no early exit), so full and lean execute
// identical control flow — the collapse that breaks full's *correctness* does NOT bias
// the *timing*. Isolates the per-op ALU/scheduler cost the barrier placement controls.
function buildTimeFrag(mulName, addName, loops) {
  return `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform vec4 uIn;
uniform int uZero;
float obB(float x){ return intBitsToFloat(floatBitsToInt(x) ^ uZero); }
${MULS[mulName].src}
${ADDS[addName].src}
#define MUL(a,b) mul_${mulName}(a,b)
#define ADD(a,b) add_${addName}(a,b)
void main(){
  // per-fragment offset so the compiler can't hoist the loop out of the draw
  vec2 a = uIn.xy + vec2(gl_FragCoord.x * 1e-7, 0.0);
  vec2 b = uIn.zw;
  vec2 t = a;
  for (int i = 0; i < ${loops}; i++) t = ADD(MUL(t, a), b);
  outColor = vec4(t.x, t.y, 0.0, 1.0);
}`;
}

const TIME_RUN = ({ FRAG, W, H, reps, inp }) => {
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl) return { err: 'no webgl2' };
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  function sh(t, s) { const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
    if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('compile: ' + gl.getShaderInfoLog(o)); return o; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0, 0, W, H);
  gl.uniform1i(gl.getUniformLocation(prog, 'uZero'), 0);
  gl.uniform4f(gl.getUniformLocation(prog, 'uIn'), inp[0], inp[1], inp[2], inp[3]);
  const px = new Float32Array(4);
  // warm-up (compile/upload) + sync
  gl.drawArrays(gl.TRIANGLES, 0, 3);
  gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
  const ts = [];
  for (let r = 0; r < reps; r++) {
    const t0 = performance.now();
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px); // forces finish
    ts.push(performance.now() - t0);
  }
  ts.sort((x, y) => x - y);
  return { renderer, median: ts[ts.length >> 1], min: ts[0] };
};

async function timeMain() {
  const W = 512, H = 512, LOOPS = 3000, REPS = 25;
  const inp = (() => { const A = df(1.0000001192092896), B = df(0.49999997); return [A[0], A[1], B[0], B[1]]; })();
  const browser = await chromium.launch(launchOpts());
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
    await page.goto('about:blank');
    const variants = [['full', 'full'], ['lean8', 'lean6'], ['lean10', 'lean6']];
    console.log(`mode=${gpuMode()}  ${W}x${H}, ${LOOPS} chained (mul+add)/frag, ${REPS} reps\n`);
    console.log('mul        add        barriers   median ms   Mop/s        rel');
    let baseMs = null;
    for (const [m, a] of variants) {
      const FRAG = buildTimeFrag(m, a, LOOPS);
      const r = await page.evaluate(TIME_RUN, { FRAG, W, H, reps: REPS, inp });
      if (r.err) throw new Error(r.err);
      if (baseMs === null) baseMs = r.median;
      const ops = (W * H * LOOPS * 2) / r.median / 1e3; // mul+add per loop = 2 ops, Mop/s
      const bar = MULS[m].n + ADDS[a].n;
      console.log(`${m.padEnd(10)} ${a.padEnd(10)} ${String(bar).padStart(4)}      ${r.median.toFixed(2).padStart(8)}   ${ops.toFixed(0).padStart(8)}     ${(baseMs / r.median).toFixed(2)}x`);
    }
    console.log('\n(rel = speedup vs full/full. >1 means the lean placement is faster.)');

    // ---- squaring A/B: the SAME chained quadratic t <- SQR(t)+c, only the op differs.
    // mulref = shipping ds_mul(t,t); sqr6 = the dedicated one-split squaring. If the
    // compiler already CSEs the duplicate split in mul(a,a), the ratio will be ~1.00.
    const sqrInp = (() => { const T = df(0.3141592653589793), C = df(-0.5); return [T[0], T[1], C[0], C[1]]; })();
    function buildTimeSqrFrag(sqrName, addName, loops) {
      return `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform vec4 uIn;
uniform int uZero;
float obB(float x){ return intBitsToFloat(floatBitsToInt(x) ^ uZero); }
${MULS.lean8.src}
${SQRS[sqrName].src}
${ADDS[addName].src}
#define SQR(a) sqr_${sqrName}(a)
#define ADD(a,b) add_${addName}(a,b)
void main(){
  vec2 t = uIn.xy + vec2(gl_FragCoord.x * 1e-7, 0.0);
  vec2 b = uIn.zw;
  for (int i = 0; i < ${loops}; i++) t = ADD(SQR(t), b);
  outColor = vec4(t.x, t.y, 0.0, 1.0);
}`;
    }
    console.log('\nsquaring A/B (chained t←SQR(t)+c, attracting c=-0.5):');
    console.log('sqr        add        barriers   median ms   Mop/s        rel');
    let sqrBase = null;
    for (const s of ['mulref', 'sqr6']) {
      const FRAG = buildTimeSqrFrag(s, 'lean6', LOOPS);
      const r = await page.evaluate(TIME_RUN, { FRAG, W, H, reps: REPS, inp: sqrInp });
      if (r.err) throw new Error(r.err);
      if (sqrBase === null) sqrBase = r.median;
      const ops = (W * H * LOOPS * 2) / r.median / 1e3;
      const bar = SQRS[s].n + ADDS.lean6.n;
      console.log(`${s.padEnd(10)} ${'lean6'.padEnd(10)} ${String(bar).padStart(4)}      ${r.median.toFixed(2).padStart(8)}   ${ops.toFixed(0).padStart(8)}     ${(sqrBase / r.median).toFixed(2)}x`);
    }
    console.log('(rel > 1 means the dedicated ds_sqr is faster than ds_mul(a,a).)');
  } finally {
    await browser.close();
  }
}

async function main() {
  if (process.env.TIME) return timeMain();
  const browser = await chromium.launch(launchOpts());
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
    await page.goto('about:blank');

    let rendererPrinted = false;
    async function worst(mulName, addName, mode) {
      const FRAG = buildFrag(mulName, addName);
      const r = await page.evaluate(RUN, { FRAG, inputs, mode });
      if (r.err) throw new Error(r.err);
      if (!rendererPrinted) { console.log(`mode=${gpuMode()}  renderer: ${r.renderer}\n`); rendererPrinted = true; }
      let w = 0;
      r.out.forEach((v, i) => {
        const truth = truthFor(inputs[i], mode);
        const re = Math.abs((v[0] + v[1]) - truth) / Math.max(Math.abs(truth), 1e-30);
        w = Math.max(w, re);
      });
      return w;
    }

    console.log('=== ds_mul variants (mode 0 single, mode 2 chained x24 with add_full) ===');
    console.log('variant     barriers   single relerr        chained relerr       verdict');
    for (const name of Object.keys(MULS)) {
      const w0 = await worst(name, 'full', 0);
      const w2 = await worst(name, 'full', 2);
      const v = Math.max(w0, w2);
      console.log(`${name.padEnd(11)} ${String(MULS[name].n).padStart(4)}      ${w0.toExponential(2)}            ${w2.toExponential(2)}            ${VERD(v)} (worst ${v.toExponential(1)})`);
    }

    console.log('\n=== ds_add variants (mode 1 single, mode 2 chained x24 with mul_full) ===');
    console.log('variant     barriers   single relerr        chained relerr       verdict');
    for (const name of Object.keys(ADDS)) {
      const w1 = await worst('full', name, 1);
      const w2 = await worst('full', name, 2);
      const v = Math.max(w1, w2);
      console.log(`${name.padEnd(11)} ${String(ADDS[name].n).padStart(4)}      ${w1.toExponential(2)}            ${w2.toExponential(2)}            ${VERD(v)} (worst ${v.toExponential(1)})`);
    }

    console.log('\n=== ds_sqr variants (mode 0 single a², mode 2 attracting chained t←t²+c ×24 with add_lean6) ===');
    console.log('variant     barriers   single relerr        chained relerr       verdict');
    async function worstSqr(sqrName, mode) {
      const FRAG = buildSqrFrag(sqrName, 'lean6');
      const r = await page.evaluate(RUN, { FRAG, inputs: mode === 0 ? inputs : sqrChainInputs, mode });
      if (r.err) throw new Error(r.err);
      let w = 0;
      r.out.forEach((v, i) => {
        const truth = truthSqr((mode === 0 ? inputs : sqrChainInputs)[i], mode);
        const re = Math.abs((v[0] + v[1]) - truth) / Math.max(Math.abs(truth), 1e-30);
        w = Math.max(w, re);
      });
      return w;
    }
    for (const name of Object.keys(SQRS)) {
      const w0 = await worstSqr(name, 0);
      const w2 = await worstSqr(name, 2);
      const v = Math.max(w0, w2);
      console.log(`${name.padEnd(11)} ${String(SQRS[name].n).padStart(4)}      ${w0.toExponential(2)}            ${w2.toExponential(2)}            ${VERD(v)} (worst ${v.toExponential(1)})`);
    }

    // The combined candidate: leanest mul + leanest add together, chained — the real test.
    console.log('\n=== combined chained (mul x add) — the integration check ===');
    const combos = [['full', 'full'], ['lean8', 'lean5'], ['lean10', 'lean6'], ['lean14', 'lean6']];
    console.log('mul        add        barriers   chained relerr       verdict');
    for (const [m, a] of combos) {
      const w = await worst(m, a, 2);
      console.log(`${m.padEnd(10)} ${a.padEnd(10)} ${String(MULS[m].n + ADDS[a].n).padStart(4)}      ${w.toExponential(2)}            ${VERD(w)}`);
    }
  } finally {
    await browser.close();
  }
}
main();
