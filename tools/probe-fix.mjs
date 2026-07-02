// probe-fix.mjs — try fixes for the texture-Z df64 collapse (root cause from
// probe-texz: a per-iteration texelFetch'd reference operand defeats the per-op
// barrier inside the df64 update on NVIDIA, though a uniform/constant operand does
// not). Same update-only loop with per-iteration getZ; several Z-laundering
// strategies; report relerr vs the double oracle at increasing K. INTACT to high K
// wins.
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
    zx[n] = hx; zxlo[n] = Math.fround(x - hx); zy[n] = hy; zylo[n] = Math.fround(y - hy);
    const nx = x * x - y * y + CRE, ny = 2 * x * y + CIM; x = nx; y = ny;
    if (x * x + y * y > 4) { n++; break; }
  }
  return { zx, zy, zxlo, zylo, len: n - 1 };
}

// getZ body variants (how the texelFetch result is laundered before df64 use)
const GETZ = {
  BASE:   `Zx = v.xy; Zy = v.zw;`,                                   // current (collapses)
  OB:     `Zx = vec2(ob(v.x), ob(v.y)); Zy = vec2(ob(v.z), ob(v.w));`, // barrier each component
  HIGHP:  `Zx = v.xy; Zy = v.zw;`,                                   // BASE body, but sampler is highp
};

function frag(getzBody, highpSampler) {
  return `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
uniform ${highpSampler ? 'highp ' : ''}sampler2D uRef;
uniform int uRefW, uK;
uniform vec2 uDcx, uDcy;
out vec4 frag;
void getZ(int m, out vec2 Zx, out vec2 Zy){
  vec4 v = texelFetch(uRef, ivec2(m % uRefW, m / uRefW), 0); ${getzBody}
}
void main(){
  vec2 dcx = uDcx, dcy = uDcy;
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  int m = 0;
  vec2 Zx, Zy; getZ(0, Zx, Zy);
  for (int i = 0; i < 100000000; i++) {
    if (i >= uK) break;
    vec2 t1 = ds_sub(ds_mul(Zx, dx), ds_mul(Zy, dy));
    vec2 t2 = ds_add(ds_mul(Zx, dy), ds_mul(Zy, dx));
    vec2 sx = ds_sub(ds_mul(dx, dx), ds_mul(dy, dy));
    vec2 sy = ds_mul(dx, dy);
    dx = ds_add(ds_add(ds_add(t1, t1), sx), dcx);
    dy = ds_add(ds_add(ds_add(t2, t2), ds_add(sy, sy)), dcy);
    m++; getZ(m, Zx, Zy);
  }
  frag = vec4(dx.x, dx.y, dy.x, dy.y);
}`;
}

const RUN = (args) => {
  const { progs, ref, refLen, dcx, dcy, Ks } = args;
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
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, rt);
    gl.uniform1i(gl.getUniformLocation(prog, 'uRef'), 0);
    gl.uniform1i(gl.getUniformLocation(prog, 'uRefW'), RW);
    const ob = gl.getUniformLocation(prog, 'uOptBarrier'); if (ob) gl.uniform1i(ob, 0);
    gl.uniform2f(gl.getUniformLocation(prog, 'uDcx'), dcx[0], dcx[1]);
    gl.uniform2f(gl.getUniformLocation(prog, 'uDcy'), dcy[0], dcy[1]);
    const uK = gl.getUniformLocation(prog, 'uK');
    gl.viewport(0, 0, 1, 1); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    const r = {};
    for (const K of Ks) {
      gl.uniform1i(uK, K); gl.drawArrays(gl.TRIANGLES, 0, 3);
      const px = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
      r[K] = [px[0], px[1], px[2], px[3]];
    }
    out.res[p.name] = r;
  }
  return out;
};

function cpuIter(ref, K, dcxd, dcyd) {
  const sp = (h, l) => h + l;
  let dx = 0, dy = 0, m = 0;
  for (let i = 0; i < K; i++) {
    const Zx = sp(ref.zx[m], ref.zxlo[m]), Zy = sp(ref.zy[m], ref.zylo[m]);
    const t1 = Zx*dx - Zy*dy, t2 = Zx*dy + Zy*dx;
    const sx = dx*dx - dy*dy, sy = dx*dy;
    dx = 2*t1 + sx + dcxd; dy = 2*t2 + 2*sy + dcyd; m++;
  }
  return [dx, dy];
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

const radius = 1.5 * 2 ** -50;
const ref = buildRef(200);
const scale = 2 * radius / 96;
const dcxd = 7 * scale, dcyd = -5 * scale;
const dcx = df64Split(dcxd), dcy = df64Split(dcyd);
const Ks = [5, 20, 100];
const progs = Object.entries(GETZ).map(([name, body]) => ({ name, frag: frag(body, name === 'HIGHP') }));
const payload = { progs, ref, refLen: ref.len, dcx, dcy, Ks };

const gpu = await renderOn(VULKAN, payload);
console.log(`renderer: ${gpu.renderer}   refLen=${ref.len}\n`);
console.log('variant   ' + Ks.map((K) => `K=${K}`.padEnd(22)).join(''));
for (const name of Object.keys(GETZ)) {
  const cells = Ks.map((K) => {
    const [odx, ody] = cpuIter(ref, K, dcxd, dcyd);
    const mag = Math.hypot(odx, ody) || 1;
    const g = gpu.res[name][K];
    const err = Math.hypot((g[0]+g[1])-odx, (g[2]+g[3])-ody) / mag;
    const v = err < 1e-10 ? 'INTACT' : (err < 1e-4 ? 'PARTIAL' : 'COLLAPSED');
    return `${err.toExponential(1)} [${v}]`.padEnd(22);
  });
  console.log(name.padEnd(9) + ' ' + cells.join(''));
}
