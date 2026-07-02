// probe-state.mjs — does the df64 dz STATE itself diverge on the GPU, and at which
// iteration? Runs the real df64 perturbation update (with rebasing) for a SINGLE
// pixel up to n == uDumpIter (no escape exit), then outputs the raw df64 dz =
// (dx.hi,dx.lo,dy.hi,dy.lo). Compares GPU vs SwiftShader vs a double-precision CPU
// oracle as a function of the iteration count. The isolated ops are intact, so if
// the state diverges this pins down the construct that leaks in the full loop.
import { chromium } from '@playwright/test';
import { resolveChromium } from './chromium-launch.mjs';
import { DF64_LIB } from '../src/gpu/glsl.js';

const SWIFT = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu','--enable-unsafe-swiftshader','--headless=new'];
const VULKAN = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--use-angle=vulkan','--use-gl=angle','--enable-features=Vulkan','--ignore-gpu-blocklist','--enable-gpu','--headless=new'];

const CRE = -0.743643887037158704752191506114774;
const CIM =  0.131825904205311970493132056385139;

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

const FRAG = `#version 300 es
precision highp float;
precision highp int;
${DF64_LIB}
uniform sampler2D uRef;
uniform int uRefW, uRefLen, uDumpIter;
uniform vec2 uDcx, uDcy;      // df64 dc for the single probed pixel
out vec4 frag;
void getZ(int m, out vec2 Zx, out vec2 Zy){
  vec4 v = texelFetch(uRef, ivec2(m % uRefW, m / uRefW), 0);
  Zx = v.xy; Zy = v.zw;
}
out vec4 unused;     // placeholder (single attachment)
void main(){
  vec2 dcx = uDcx, dcy = uDcy;
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  int m = 0, n = 0;
  vec2 Zx, Zy; getZ(0, Zx, Zy);
  for (int i = 0; i < 100000000; i++) {
    if (n >= uDumpIter) break;
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
    float mag2 = ds_tofloat(ds_add(ds_mul(zfx, zfx), ds_mul(zfy, zfy)));
    float dz2 = ds_tofloat(ds_add(ds_mul(dx, dx), ds_mul(dy, dy)));
    bool rebase = (mag2 < dz2) || (m == uRefLen);
    if (rebase) { dx = zfx; dy = zfy; m = 0; getZ(0, Zx, Zy); }
  }
  frag = vec4(dx.x, dx.y, dy.x, dy.y);
}`;
// note: the duplicate `out` above would not compile; build sets exactly one. Fixed below.

const SHADER = FRAG.replace('out vec4 unused;     // placeholder (single attachment)\n', '');

const RUN = (args) => {
  const { shader, ref, refLen, dcx, dcy, dumps } = args;
  const c = document.createElement('canvas'); c.width = 1; c.height = 1;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  function sh(t, s){ const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
    if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('compile: ' + gl.getShaderInfoLog(o)); return o; }
  const RW = 2048, RH = Math.ceil((refLen + 1) / RW);
  const tex = new Float32Array(RW * RH * 4);
  for (let i = 0; i <= refLen; i++) { tex[i*4] = ref.zx[i]; tex[i*4+1] = ref.zxlo[i]; tex[i*4+2] = ref.zy[i]; tex[i*4+3] = ref.zylo[i]; }
  const rt = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, rt);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, RW, RH, 0, gl.RGBA, gl.FLOAT, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
  const ct = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, ct);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, ct, 0);
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, shader));
  gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, rt);
  gl.uniform1i(gl.getUniformLocation(prog, 'uRef'), 0);
  gl.uniform1i(gl.getUniformLocation(prog, 'uRefW'), RW);
  gl.uniform1i(gl.getUniformLocation(prog, 'uRefLen'), refLen);
  const ob = gl.getUniformLocation(prog, 'uOptBarrier'); if (ob) gl.uniform1i(ob, 0);
  gl.uniform2f(gl.getUniformLocation(prog, 'uDcx'), dcx[0], dcx[1]);
  gl.uniform2f(gl.getUniformLocation(prog, 'uDcy'), dcy[0], dcy[1]);
  const uDump = gl.getUniformLocation(prog, 'uDumpIter');
  gl.viewport(0, 0, 1, 1); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const out = { renderer, states: {} };
  for (const K of dumps) {
    gl.uniform1i(uDump, K);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    out.states[K] = [px[0], px[1], px[2], px[3]];   // dx.hi,dx.lo,dy.hi,dy.lo
  }
  return out;
};

function cpuOracle(ref, refLen, dcxd, dcyd, dumps) {
  const sp = (h, l) => h + l;
  const res = {};
  let dx = 0, dy = 0, m = 0, n = 0;
  const want = new Set(dumps);
  for (; n <= Math.max(...dumps); ) {
    if (want.has(n)) res[n] = [dx, dy];
    const Zx = sp(ref.zx[m], ref.zxlo[m]), Zy = sp(ref.zy[m], ref.zylo[m]);
    const t1 = Zx*dx - Zy*dy, t2 = Zx*dy + Zy*dx;
    const sx = dx*dx - dy*dy, sy = dx*dy;
    dx = 2*t1 + sx + dcxd; dy = 2*t2 + 2*sy + dcyd; m++; n++;
    const Zx2 = sp(ref.zx[m], ref.zxlo[m]), Zy2 = sp(ref.zy[m], ref.zylo[m]);
    const zfx = Zx2 + dx, zfy = Zy2 + dy;
    const mag2 = zfx*zfx + zfy*zfy, dz2 = dx*dx + dy*dy;
    if ((mag2 < dz2) || (m === refLen)) { dx = zfx; dy = zfy; m = 0; }
  }
  if (want.has(n)) res[n] = [dx, dy];
  return res;
}

function df64Split(v) { const hi = Math.fround(v); return [hi, Math.fround(v - hi)]; }

async function renderOn(args, payload) {
  const browser = await chromium.launch({ executablePath: resolveChromium(), args });
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
    await page.goto('about:blank');
    return await page.evaluate(RUN, payload);
  } finally { await browser.close(); }
}

const bits = 50, radius = 1.5 * 2 ** -bits, maxIter = 13000;
const ref = buildRef(maxIter);
const scale = 2 * radius / 96;
// probe a near-center pixel (small dc, so dz stays small and rebases happen)
const dcxd = 7 * scale, dcyd = -5 * scale;
const dcx = df64Split(dcxd), dcy = df64Split(dcyd);
const dumps = [1, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 12000];
const payload = { shader: SHADER, ref, refLen: ref.len, dcx, dcy, dumps };

const swift = await renderOn(SWIFT, payload);
const gpu = await renderOn(VULKAN, payload);
const oracle = cpuOracle(ref, ref.len, dcxd, dcyd, dumps);

console.log(`refLen=${ref.len} scale=${scale.toExponential(3)} dc=(${dcxd.toExponential(3)},${dcyd.toExponential(3)})`);
console.log(`renderer: ${gpu.renderer}\n`);
console.log('iter     |dz|        GPUvsSwift   GPUvsOracle  SwiftvsOracle');
for (const K of dumps) {
  const g = gpu.states[K], s = swift.states[K], o = oracle[K];
  const gdx = g[0]+g[1], gdy = g[2]+g[3], sdx = s[0]+s[1], sdy = s[2]+s[3];
  const odx = o[0], ody = o[1];
  const mag = Math.hypot(odx, ody) || 1;
  const gs = Math.hypot(gdx-sdx, gdy-sdy) / mag;
  const go = Math.hypot(gdx-odx, gdy-ody) / mag;
  const so = Math.hypot(sdx-odx, sdy-ody) / mag;
  console.log(`${String(K).padStart(5)}  ${mag.toExponential(2)}   ${gs.toExponential(2)}     ${go.toExponential(2)}     ${so.toExponential(2)}`);
}
