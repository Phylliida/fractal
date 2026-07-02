// probe-loop.mjs — THE decisive test. The isolated df64 update is INTACT in a
// CONSTANT-bound loop (`for i<40`, fully unrollable — probe-df64-real) but the real
// shader COLLAPSES (probe-state: dz wrong from iter 5). The one structural difference
// is the loop: the real shader uses a DYNAMIC bound with a runtime `break`, which the
// compiler cannot unroll. Hypothesis: in a rolled loop the NVIDIA backend defeats the
// per-op barrier (re-materializes / reschedules the ob() round-trip across the
// loop-carried df64 values), collapsing precision.
//
// Same fixed Z, same dc, same update arithmetic — run K iterations TWO ways and
// compare each to a double-precision CPU oracle:
//   CONST : for (i=0;i<K;i++)            (K a compile-time constant per program)
//   DYN   : for (i=0;i<BIG;i++){if(i>=uK)break;}   (uK a uniform — not unrollable)
import { chromium } from '@playwright/test';
import { resolveChromium } from './chromium-launch.mjs';
import { DF64_LIB } from '../src/gpu/glsl.js';

const SWIFT = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu','--enable-unsafe-swiftshader','--headless=new'];
const VULKAN = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--use-angle=vulkan','--use-gl=angle','--enable-features=Vulkan','--ignore-gpu-blocklist','--enable-gpu','--headless=new'];

// update-only inner body (no texture, no rebase) — exactly probe-df64-real mode 1.
const BODY = `
    vec2 t1 = ds_sub(ds_mul(Zx, dx), ds_mul(Zy, dy));
    vec2 t2 = ds_add(ds_mul(Zx, dy), ds_mul(Zy, dx));
    vec2 sx = ds_sub(ds_mul(dx, dx), ds_mul(dy, dy));
    vec2 sy = ds_mul(dx, dy);
    dx = ds_add(ds_add(ds_add(t1, t1), sx), dcx);
    dy = ds_add(ds_add(ds_add(t2, t2), ds_add(sy, sy)), dcy);`;

function fragConst(K) {
  return `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
uniform float uDcx, uDcy;
out vec4 frag;
void main(){
  vec2 Zx = ds_set(0.31), Zy = ds_set(0.43);
  vec2 dcx = ds_set(uDcx), dcy = ds_set(uDcy);
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  for (int i = 0; i < ${K}; i++) { ${BODY} }
  frag = vec4(dx.x, dx.y, dy.x, dy.y);
}`;
}
function fragDyn() {
  return `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
uniform float uDcx, uDcy;
uniform int uK;
out vec4 frag;
void main(){
  vec2 Zx = ds_set(0.31), Zy = ds_set(0.43);
  vec2 dcx = ds_set(uDcx), dcy = ds_set(uDcy);
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  for (int i = 0; i < 100000000; i++) { if (i >= uK) break; ${BODY} }
  frag = vec4(dx.x, dx.y, dy.x, dy.y);
}`;
}

const RUN = (args) => {
  const { progs, dcx, dcy } = args;
  const c = document.createElement('canvas'); c.width = 1; c.height = 1;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  function sh(t, s){ const o = gl.createShader(t); gl.shaderSource(o, s); gl.compileShader(o);
    if (!gl.getShaderParameter(o, gl.COMPILE_STATUS)) throw new Error('compile: ' + gl.getShaderInfoLog(o)); return o; }
  const ct = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, ct);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, ct, 0);
  gl.viewport(0, 0, 1, 1);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
  const out = { renderer, res: {} };
  for (const p of progs) {
    const prog = gl.createProgram();
    gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, p.frag));
    gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link ' + p.name + ': ' + gl.getProgramInfoLog(prog));
    gl.useProgram(prog);
    gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    const ob = gl.getUniformLocation(prog, 'uOptBarrier'); if (ob) gl.uniform1i(ob, 0);
    gl.uniform1f(gl.getUniformLocation(prog, 'uDcx'), dcx);
    gl.uniform1f(gl.getUniformLocation(prog, 'uDcy'), dcy);
    const uK = gl.getUniformLocation(prog, 'uK'); if (uK) gl.uniform1i(uK, p.K);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.viewport(0, 0, 1, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    out.res[p.name] = [px[0], px[1], px[2], px[3]];
  }
  return out;
};

function cpuIter(K, dcxd, dcyd) {
  const Zx = Math.fround(0.31), Zy = Math.fround(0.43);
  let dx = 0, dy = 0;
  for (let i = 0; i < K; i++) {
    const t1 = Zx*dx - Zy*dy, t2 = Zx*dy + Zy*dx;
    const sx = dx*dx - dy*dy, sy = dx*dy;
    dx = 2*t1 + sx + dcxd; dy = 2*t2 + 2*sy + dcyd;
  }
  return [dx, dy];
}

async function renderOn(args, payload) {
  const browser = await chromium.launch({ executablePath: resolveChromium(), args });
  try {
    const page = await browser.newPage();
    page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
    await page.goto('about:blank');
    return await page.evaluate(RUN, payload);
  } finally { await browser.close(); }
}

const dcxd = Math.fround(1e-3), dcyd = Math.fround(7e-4);
const Ks = [10, 20, 40];
const progs = [];
for (const K of Ks) {
  progs.push({ name: `CONST${K}`, frag: fragConst(K) });
  progs.push({ name: `DYN${K}`, frag: fragDyn(), K });
}
const payload = { progs, dcx: dcxd, dcy: dcyd };

const swift = await renderOn(SWIFT, payload);
const gpu = await renderOn(VULKAN, payload);
console.log(`renderer: ${gpu.renderer}\n`);
console.log('variant    GPU-relerr-vs-oracle   Swift-relerr-vs-oracle');
for (const K of Ks) {
  const [odx, ody] = cpuIter(K, dcxd, dcyd);
  const mag = Math.hypot(odx, ody) || 1;
  for (const name of [`CONST${K}`, `DYN${K}`]) {
    const g = gpu.res[name], s = swift.res[name];
    const gerr = Math.hypot((g[0]+g[1])-odx, (g[2]+g[3])-ody) / mag;
    const serr = Math.hypot((s[0]+s[1])-odx, (s[2]+s[3])-ody) / mag;
    const v = (w) => w < 1e-10 ? 'INTACT' : (w < 1e-4 ? 'PARTIAL' : 'COLLAPSED');
    console.log(`${name.padEnd(10)} ${gerr.toExponential(2)} [${v(gerr)}]      ${serr.toExponential(2)} [${v(serr)}]`);
  }
}
