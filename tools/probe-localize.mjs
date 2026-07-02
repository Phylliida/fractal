// probe-localize.mjs — localize WHERE perturbFragDf64 diverges GPU-vs-SwiftShader.
//
// Established (probe-df64-real): every ISOLATED DF64_LIB op (ds_mul/ds_add, a
// 40-iter update, big+tiny add) is bit-faithful on the real GPU — the XOR barrier
// holds, and an integer XOR with an unknown uniform CANNOT be reassociated across.
// Yet the full perturbFragDf64 still diverges (probe-xbackend: 21%@2^-22, 90%@2^-50).
// So the leak is NOT in the barriered df64 ops; it's in the UN-barriered float32
// logic (the escape/rebase reductions `ds_tofloat`, the mag2<dz2 decision) where the
// NVIDIA compiler is free to contract a*b+c into FMA / reassociate, flipping a
// knife-edge rebase decision that then cascades on chaotic pixels.
//
// This probe runs the REAL df64 perturbation inner loop with a JS-built reference
// orbit (uploaded IDENTICALLY to both backends, so any reference error is common-mode
// and cancels). It compiles SEPARATE shader programs for several VARIANTS and reports
// GPU-vs-SwiftShader mismatch for each. The variant that drops the mismatch to ~0
// localizes the construct; a CPU-double oracle says which side is actually correct.
//
//   node tools/probe-localize.mjs            # runs BOTH backends, compares
import { chromium } from '@playwright/test';
import { resolveChromium } from './chromium-launch.mjs';
import { DF64_LIB } from '../src/gpu/glsl.js';

const SWIFT = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu','--enable-unsafe-swiftshader','--headless=new'];
const VULKAN = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--use-angle=vulkan','--use-gl=angle','--enable-features=Vulkan','--ignore-gpu-blocklist','--enable-gpu','--headless=new'];

const CRE = -0.743643887037158704752191506114774;
const CIM =  0.131825904205311970493132056385139;
const W = 96, H = 96;

// ---- JS reference orbit at the center, plain doubles (common-mode to both backends) ----
function buildRef(maxIter) {
  const zx = new Float32Array(maxIter + 1), zy = new Float32Array(maxIter + 1);
  const zxlo = new Float32Array(maxIter + 1), zylo = new Float32Array(maxIter + 1);
  let x = 0, y = 0, n = 0;
  for (; n <= maxIter; n++) {
    const hx = Math.fround(x), hy = Math.fround(y);
    zx[n] = hx; zxlo[n] = Math.fround(x - hx);
    zy[n] = hy; zylo[n] = Math.fround(y - hy);
    const nx = x * x - y * y + CRE, ny = 2 * x * y + CIM;
    x = nx; y = ny;
    if (x * x + y * y > 4) { n++; break; }
  }
  return { zx, zy, zxlo, zylo, len: n - 1 };
}

// The inner loop, mirroring perturbFragDf64, with #ifdef knobs.
// VARIANT macros:
//   BARR_RED  : wrap the float32 escape/rebase reductions in ob() (block FMA/reassoc)
//   NO_REBASE : never rebase mid-orbit (only the forced end-of-reference rebase)
function frag(defs) {
  return `#version 300 es
precision highp float;
precision highp int;
${defs}
${DF64_LIB}
uniform sampler2D uRef;
uniform int uRefW, uRefLen, uMaxIter;
uniform vec2 uOx, uOy, uScale;
uniform float uBailoutSq;
out vec4 frag;

void getZ(int m, out vec2 Zx, out vec2 Zy){
  vec4 v = texelFetch(uRef, ivec2(m % uRefW, m / uRefW), 0);
  Zx = v.xy; Zy = v.zw;
}
#ifdef BARR_RED
  float red(vec2 a){ return ob(a.x) + ob(a.y); }   // barrier the reduction terms
#else
  float red(vec2 a){ return a.x + a.y; }
#endif

void main(){
  vec2 dcx = ds_add(uOx, ds_mul(ds_set(gl_FragCoord.x), uScale));
  vec2 dcy = ds_add(uOy, ds_mul(ds_set(gl_FragCoord.y), uScale));
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  int m = 0, n = 0;
  vec2 Zx, Zy; getZ(0, Zx, Zy);
  for (int i = 0; i < 100000000; i++) {
    if (n >= uMaxIter) break;
    vec2 t1 = ds_sub(ds_mul(Zx, dx), ds_mul(Zy, dy));
    vec2 t2 = ds_add(ds_mul(Zx, dy), ds_mul(Zy, dx));
    vec2 sx = ds_sub(ds_mul(dx, dx), ds_mul(dy, dy));
    vec2 sy = ds_mul(dx, dy);
    vec2 ndx = ds_add(ds_add(ds_add(t1, t1), sx), dcx);
    vec2 ndy = ds_add(ds_add(ds_add(t2, t2), ds_add(sy, sy)), dcy);
    dx = ndx; dy = ndy;
    m++; n++;
    getZ(m, Zx, Zy);
    vec2 zfx = ds_add(Zx, dx);
    vec2 zfy = ds_add(Zy, dy);
    float mag2 = red(ds_add(ds_mul(zfx, zfx), ds_mul(zfy, zfy)));
    if (mag2 > uBailoutSq) { frag = vec4(float(n), float(n), 0.0, 1.0); return; }
    float dz2 = red(ds_add(ds_mul(dx, dx), ds_mul(dy, dy)));
#ifdef NO_REBASE
    bool rebase = (m == uRefLen);
#else
    bool rebase = (mag2 < dz2) || (m == uRefLen);
#endif
    if (rebase) { dx = zfx; dy = zfy; m = 0; getZ(0, Zx, Zy); }
  }
  frag = vec4(-1.0, float(n), 0.0, 1.0);
}`;
}

const VARIANTS = [
  { name: 'BASE',        defs: '' },
  { name: 'BARR_RED',    defs: '#define BARR_RED' },
  { name: 'NO_REBASE',   defs: '#define NO_REBASE' },
];

const RUN = (args) => {
  const { variants, ref, W, H, ox, oy, scale, maxIter, refLen, bail } = args;
  const c = document.createElement('canvas'); c.width = W; c.height = H;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  function sh(t, s){ const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
    if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('compile: ' + gl.getShaderInfoLog(o)); return o; }
  // reference texture (RGBA32F): Zx.hi, Zx.lo, Zy.hi, Zy.lo
  const RW = 2048, RH = Math.ceil((refLen + 1) / RW);
  const tex = new Float32Array(RW * RH * 4);
  for (let i = 0; i <= refLen; i++) {
    tex[i*4+0] = ref.zx[i]; tex[i*4+1] = ref.zxlo[i]; tex[i*4+2] = ref.zy[i]; tex[i*4+3] = ref.zylo[i];
  }
  const rt = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, rt);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, RW, RH, 0, gl.RGBA, gl.FLOAT, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  // render target
  const ct = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, ct);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, W, H, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, ct, 0);
  gl.viewport(0, 0, W, H);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);

  const out = { renderer, results: {} };
  for (const v of variants) {
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS));
    gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, v.frag));
    gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link ' + v.name + ': ' + gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, rt);
    gl.uniform1i(gl.getUniformLocation(prog, 'uRef'), 0);
    gl.uniform1i(gl.getUniformLocation(prog, 'uRefW'), RW);
    gl.uniform1i(gl.getUniformLocation(prog, 'uRefLen'), refLen);
    gl.uniform1i(gl.getUniformLocation(prog, 'uMaxIter'), maxIter);
    const ob = gl.getUniformLocation(prog, 'uOptBarrier'); if (ob) gl.uniform1i(ob, 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'uOx'), ox[0], ox[1]);
    gl.uniform2f(gl.getUniformLocation(prog, 'uOy'), oy[0], oy[1]);
    gl.uniform2f(gl.getUniformLocation(prog, 'uScale'), scale[0], scale[1]);
    gl.uniform1f(gl.getUniformLocation(prog, 'uBailoutSq'), bail);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.viewport(0, 0, W, H);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Float32Array(W * H * 4); gl.readPixels(0, 0, W, H, gl.RGBA, gl.FLOAT, px);
    const iter = new Array(W * H);
    for (let i = 0; i < W * H; i++) iter[i] = px[i*4+1];   // .g = iteration count
    out.results[v.name] = iter;
  }
  return out;
};

// CPU double-precision oracle of the SAME algorithm (the "truth").
function cpuOracle(ref, ox, oy, scale, maxIter, refLen, bail) {
  const iter = new Array(W * H);
  const sp = (v) => v[0] + v[1];
  const oxd = sp(ox), oyd = sp(oy), scd = sp(scale);
  for (let py = 0; py < H; py++) for (let px = 0; px < W; px++) {
    const fx = px + 0.5, fy = py + 0.5;
    const dcx = oxd + fx * scd, dcy = oyd + fy * scd;
    let dx = 0, dy = 0, m = 0, n = 0, res = -1;
    for (;;) {
      if (n >= maxIter) break;
      const t1 = sp([ref.zx[m], ref.zxlo[m]]) * dx - sp([ref.zy[m], ref.zylo[m]]) * dy;
      const t2 = sp([ref.zx[m], ref.zxlo[m]]) * dy + sp([ref.zy[m], ref.zylo[m]]) * dx;
      const sx = dx*dx - dy*dy, sy = dx*dy;
      const ndx = 2*t1 + sx + dcx, ndy = 2*t2 + 2*sy + dcy;
      dx = ndx; dy = ndy; m++; n++;
      const Zx = sp([ref.zx[m], ref.zxlo[m]]), Zy = sp([ref.zy[m], ref.zylo[m]]);
      const zfx = Zx + dx, zfy = Zy + dy;
      const mag2 = zfx*zfx + zfy*zfy;
      if (mag2 > bail) { res = n; break; }
      const dz2 = dx*dx + dy*dy;
      const rebase = (mag2 < dz2) || (m === refLen);
      if (rebase) { dx = zfx; dy = zfy; m = 0; }
    }
    iter[py*W + px] = res < 0 ? -1 : res;
  }
  return iter;
}

function df64Split(v) { const hi = Math.fround(v); return [hi, Math.fround(v - hi)]; }

async function renderOn(args, defsList) {
  const browser = await chromium.launch({ executablePath: resolveChromium(), args });
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
    await page.goto('about:blank');
    return await page.evaluate(RUN, defsList);
  } finally { await browser.close(); }
}

function compare(a, b) {
  let n = 0, diff = 0, sum = 0, max = 0;
  for (let i = 0; i < a.length; i++) {
    n++; const d = Math.abs(a[i] - b[i]);
    if (d > 0) diff++; sum += d; if (d > max) max = d;
  }
  return { mism: (100*diff/n).toFixed(2), mean: (sum/n).toFixed(2), max };
}

for (const [tag, bits, maxIter] of [['2^-22', 22, 5900], ['2^-50', 50, 12900]]) {
  const radius = 1.5 * 2 ** -bits;
  const ref = buildRef(maxIter);
  const scale = 2 * radius / H;
  const ox = df64Split(-(W / 2) * scale), oy = df64Split(-(H / 2) * scale), sc = df64Split(scale);
  const bail = 1 << 16;
  const variants = VARIANTS.map((v) => ({ name: v.name, frag: frag(v.defs) }));
  const payload = { variants, ref, W, H, ox, oy, scale: sc, maxIter, refLen: ref.len, bail };

  const swift = await renderOn(SWIFT, payload);
  const gpu = await renderOn(VULKAN, payload);
  const oracle = cpuOracle(ref, ox, oy, sc, maxIter, ref.len, bail);

  console.log(`\n=== ${tag}  (refLen=${ref.len}, maxIter=${maxIter}, ${W}x${H}) ===`);
  console.log(`renderer: ${gpu.renderer}`);
  for (const v of VARIANTS) {
    const sv = swift.results[v.name], gv = gpu.results[v.name];
    const gs = compare(gv, sv);           // GPU vs SwiftShader (compiler-divergence signal)
    const go = compare(gv, oracle);        // GPU vs double oracle (is GPU right?)
    const so = compare(sv, oracle);        // SwiftShader vs double oracle (is Swift right?)
    console.log(`  ${v.name.padEnd(10)}  GPU-vs-Swift mism=${gs.mism}% mean=${gs.mean}  | GPU-vs-oracle ${go.mism}%  Swift-vs-oracle ${so.mism}%`);
  }
}
