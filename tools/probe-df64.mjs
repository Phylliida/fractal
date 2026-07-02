// probe-df64.mjs — isolate whether the GPU's df64 (double-single) arithmetic
// keeps ~46-bit precision, or silently collapses to float32 (~24-bit) because the
// shader compiler reassociates the Dekker/Veltkamp error terms (ca-(ca-x) -> x)
// and/or contracts a*b-c into FMA.
//
// Self-contained: builds a raw WebGL2 context on about:blank, compiles a shader
// with the CURRENT ds_mul (from src/gpu/glsl.js) plus two candidate FIXES, and
// compares hi+lo against the exact product of the FLOAT32-ROUNDED inputs (what a
// correct df64 must reproduce). A correct df64 -> relerr ~1e-13..1e-16; a
// collapsed one -> ~1e-7 (= plain float32).
//
//   node tools/probe-df64.mjs            # SwiftShader baseline
//   GPU=1 node tools/probe-df64.mjs      # real GPU (ANGLE/Vulkan)
//   GPU=gl node tools/probe-df64.mjs     # real GPU (ANGLE native GLES)
import { chromium } from '@playwright/test';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

const FRAG = `#version 300 es
precision highp float;
precision highp int;
out vec4 outColor;
uniform float uA;
uniform float uB;
uniform int uTest;
uniform int uZero;   // == 0, but the compiler can't prove it (defeats folding)

// ---- CURRENT two-product (copied from src/gpu/glsl.js DF64_LIB) ----
vec2 ds_mul(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p = a.x * b.x;
  float ca = SPLIT * a.x; float a_hi = ca - (ca - a.x); float a_lo = a.x - a_hi;
  float cb = SPLIT * b.x; float b_hi = cb - (cb - b.x); float b_lo = b.x - b_hi;
  float e = ((a_hi*b_hi - p) + a_hi*b_lo + a_lo*b_hi) + a_lo*b_lo;
  e += a.x*b.y + a.y*b.x;
  float hi = p + e;
  float lo = e - (hi - p);
  return vec2(hi, lo);
}

// ---- barrier A: plain int round-trip ----
float obA(float x){ return intBitsToFloat(floatBitsToInt(x)); }
// ---- barrier B: XOR with a uniform 0 (compiler can't prove it's identity) ----
float obB(float x){ return intBitsToFloat(floatBitsToInt(x) ^ uZero); }

vec2 ds_mul_A(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p  = obA(a.x * b.x);
  float ca = obA(SPLIT * a.x); float a_hi = obA(ca - obA(ca - a.x)); float a_lo = obA(a.x - a_hi);
  float cb = obA(SPLIT * b.x); float b_hi = obA(cb - obA(cb - b.x)); float b_lo = obA(b.x - b_hi);
  float e = obA(obA(obA(obA(obA(a_hi*b_hi) - p) + obA(a_hi*b_lo)) + obA(a_lo*b_hi)) + obA(a_lo*b_lo));
  e = obA(e + obA(obA(a.x*b.y) + obA(a.y*b.x)));
  float hi = obA(p + e);
  float lo = obA(e - obA(hi - p));
  return vec2(hi, lo);
}
vec2 ds_mul_B(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p  = obB(a.x * b.x);
  float ca = obB(SPLIT * a.x); float a_hi = obB(ca - obB(ca - a.x)); float a_lo = obB(a.x - a_hi);
  float cb = obB(SPLIT * b.x); float b_hi = obB(cb - obB(cb - b.x)); float b_lo = obB(b.x - b_hi);
  float e = obB(obB(obB(obB(obB(a_hi*b_hi) - p) + obB(a_hi*b_lo)) + obB(a_lo*b_hi)) + obB(a_lo*b_lo));
  e = obB(e + obB(obB(a.x*b.y) + obB(a.y*b.x)));
  float hi = obB(p + e);
  float lo = obB(e - obB(hi - p));
  return vec2(hi, lo);
}

// ---- barrier C: MINIMAL placement with the XOR barrier — only the cancellation
// points (split inner-subtraction + final renorm). Cheapest if it suffices. ----
vec2 ds_mul_C(vec2 a, vec2 b){
  const float SPLIT = 4097.0;
  float p = a.x * b.x;
  float ca = SPLIT * a.x; float a_hi = ca - obB(ca - a.x); float a_lo = a.x - a_hi;
  float cb = SPLIT * b.x; float b_hi = cb - obB(cb - b.x); float b_lo = b.x - b_hi;
  float e = ((a_hi*b_hi - p) + a_hi*b_lo + a_lo*b_hi) + a_lo*b_lo;
  e += a.x*b.y + a.y*b.x;
  float hi = p + e;
  float lo = e - obB(hi - p);
  return vec2(hi, lo);
}

void main(){
  vec2 a = vec2(uA, 0.0), b = vec2(uB, 0.0);
  vec2 r;
  if (uTest == 0) r = ds_mul(a, b);
  else if (uTest == 1) r = ds_mul_A(a, b);
  else if (uTest == 2) r = ds_mul_B(a, b);
  else r = ds_mul_C(a, b);
  outColor = vec4(r.x, r.y, 0.0, 1.0);
}`;

const RUN = (FRAG) => {
  const c = document.createElement('canvas'); c.width = 1; c.height = 1;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl) return { err: 'no webgl2' };
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  function sh(t,s){ const o=gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw new Error('compile: '+gl.getShaderInfoLog(o)); return o; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: '+gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog,'p'); gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,1,1,0,gl.RGBA,gl.FLOAT,null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0,0,1,1);
  const uA = gl.getUniformLocation(prog,'uA'), uB = gl.getUniformLocation(prog,'uB');
  const uTest = gl.getUniformLocation(prog,'uTest'), uZero = gl.getUniformLocation(prog,'uZero');
  gl.uniform1i(uZero, 0);
  function evalOne(a,b,test){
    gl.uniform1f(uA,a); gl.uniform1f(uB,b); gl.uniform1i(uTest,test);
    gl.drawArrays(gl.TRIANGLES,0,3);
    const px = new Float32Array(4); gl.readPixels(0,0,1,1,gl.RGBA,gl.FLOAT,px);
    return [px[0], px[1]];
  }
  // Inputs whose float32-rounded product needs >24 bits (sub-float32 tail nonzero).
  const cases = [
    [1.0 + Math.pow(2,-13), 1.0 + Math.pow(2,-13)],
    [1.0 + Math.pow(2,-20), 1.0 - Math.pow(2,-20)],
    [1.3000001, 0.7999999],
    [Math.PI, Math.E],
    [1.0000001192, 1.0000002384],
  ];
  const out = { renderer, results: [] };
  for (const [a,b] of cases) {
    out.results.push({ a, b, cur: evalOne(a,b,0), A: evalOne(a,b,1), B: evalOne(a,b,2), C: evalOne(a,b,3) });
  }
  return out;
};

const browser = await chromium.launch(launchOpts());
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
  await page.goto('about:blank');
  const r = await page.evaluate(RUN, FRAG);
  if (r.err) { console.error('ERROR:', r.err); }
  else {
    const fr = Math.fround; // float32 round, to mirror the uniform upload
    console.log(`mode=${gpuMode()}  renderer: ${r.renderer}\n`);
    console.log('truth=fround(a)*fround(b)   CUR        A(bitcast)  B(xor full) C(xor min)');
    let wc = 0, wa = 0, wb = 0, wcc = 0;
    for (const res of r.results) {
      const truth = fr(res.a) * fr(res.b);                 // exact product of the rounded inputs, in double
      const re = (v) => Math.abs((v[0] + v[1]) - truth) / Math.abs(truth);
      const rc = re(res.cur), ra = re(res.A), rb = re(res.B), rcc = re(res.C);
      wc = Math.max(wc, rc); wa = Math.max(wa, ra); wb = Math.max(wb, rb); wcc = Math.max(wcc, rcc);
      console.log(`${truth.toExponential(6).padStart(16)}   ${rc.toExponential(1)}    ${ra.toExponential(1)}    ${rb.toExponential(1)}    ${rcc.toExponential(1)}`);
    }
    const verdict = (w) => w < 1e-10 ? 'INTACT (df64 ~46-bit)' : 'COLLAPSED to float32';
    console.log(`\nworst:  CUR                          =${wc.toExponential(2)} [${verdict(wc)}]`);
    console.log(`        A (bitcast round-trip)       =${wa.toExponential(2)} [${verdict(wa)}]`);
    console.log(`        B (xor uniform-0, full)      =${wb.toExponential(2)} [${verdict(wb)}]`);
    console.log(`        C (xor uniform-0, minimal)   =${wcc.toExponential(2)} [${verdict(wcc)}]`);
  }
} finally {
  await browser.close();
}
