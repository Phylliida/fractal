// probe-mag.mjs — is the df64 collapse triggered by tiny operand MAGNITUDE? The
// INTACT probes used dz~1e-3; the COLLAPSING full shader has dz~2^-50. Same fixed
// const Z, update-only, DYN loop (the INTACT config from probe-matrix) but sweep the
// dc magnitude so dz settles at ~2.6·dc across 1e-3 … 1e-18. If small dz collapses on
// the GPU but not SwiftShader, magnitude is the trigger.
import { chromium } from '@playwright/test';
import { resolveChromium } from './chromium-launch.mjs';
import { DF64_LIB } from '../src/gpu/glsl.js';

const SWIFT = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu','--enable-unsafe-swiftshader','--headless=new'];
const VULKAN = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--use-angle=vulkan','--use-gl=angle','--enable-features=Vulkan','--ignore-gpu-blocklist','--enable-gpu','--headless=new'];

const FRAG = `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
uniform float uDcx, uDcy;
uniform int uK;
out vec4 frag;
void main(){
  vec2 Zx = ds_set(0.31), Zy = ds_set(0.43);
  vec2 dcx = ds_set(uDcx), dcy = ds_set(uDcy);
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  for (int i = 0; i < 100000000; i++) {
    if (i >= uK) break;
    vec2 t1 = ds_sub(ds_mul(Zx, dx), ds_mul(Zy, dy));
    vec2 t2 = ds_add(ds_mul(Zx, dy), ds_mul(Zy, dx));
    vec2 sx = ds_sub(ds_mul(dx, dx), ds_mul(dy, dy));
    vec2 sy = ds_mul(dx, dy);
    dx = ds_add(ds_add(ds_add(t1, t1), sx), dcx);
    dy = ds_add(ds_add(ds_add(t2, t2), ds_add(sy, sy)), dcy);
  }
  frag = vec4(dx.x, dx.y, dy.x, dy.y);
}`;

const RUN = (args) => {
  const { frag, cases, K } = args;
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
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, frag));
  gl.bindAttribLocation(prog, 0, 'p'); gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: ' + gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const ob = gl.getUniformLocation(prog, 'uOptBarrier'); if (ob) gl.uniform1i(ob, 0);
  gl.uniform1i(gl.getUniformLocation(prog, 'uK'), K);
  const uDcx = gl.getUniformLocation(prog, 'uDcx'), uDcy = gl.getUniformLocation(prog, 'uDcy');
  gl.viewport(0, 0, 1, 1); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  const out = { renderer, res: [] };
  for (const [dcx, dcy] of cases) {
    gl.uniform1f(uDcx, dcx); gl.uniform1f(uDcy, dcy);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    out.res.push([px[0], px[1], px[2], px[3]]);
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

const K = 60;
const mags = [1e-3, 1e-6, 1e-9, 1e-12, 1e-15, 1e-16, 1e-17, 1e-18];
const cases = mags.map((mg) => [Math.fround(mg), Math.fround(mg * 0.7)]);
const payload = { frag: FRAG, cases, K };

const swift = await renderOn(SWIFT, payload);
const gpu = await renderOn(VULKAN, payload);
console.log(`renderer: ${gpu.renderer}   (K=${K})\n`);
console.log('dc-mag     |dz|       GPU relerr            Swift relerr');
for (let i = 0; i < mags.length; i++) {
  const [dcxd, dcyd] = cases[i];
  const [odx, ody] = cpuIter(K, dcxd, dcyd);
  const mag = Math.hypot(odx, ody) || 1;
  const g = gpu.res[i], s = swift.res[i];
  const gerr = Math.hypot((g[0]+g[1])-odx, (g[2]+g[3])-ody) / mag;
  const serr = Math.hypot((s[0]+s[1])-odx, (s[2]+s[3])-ody) / mag;
  const v = (w) => w < 1e-10 ? 'INTACT' : (w < 1e-4 ? 'PARTIAL' : 'COLLAPSED');
  console.log(`${mags[i].toExponential(0).padEnd(8)} ${mag.toExponential(2)}  ${gerr.toExponential(2)} [${v(gerr).padEnd(9)}]   ${serr.toExponential(2)} [${v(serr)}]`);
}
