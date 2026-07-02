// probe-matrix.mjs — bisect what makes the df64 update collapse in the full loop.
// Established: update-only + fixed-const Z + DYN loop is INTACT (probe-loop), but
// update + texture-Z + escape/rebase block COLLAPSES from iter 5 (probe-state). Test
// the 2x2 matrix to find the trigger. All DYN-bound (rolled) loops; K=40; fixed O(1)
// Z so the escape block, when present, NEVER rebases (|Z|>>|dz|) — its mere PRESENCE
// (what the compiler sees) is the variable, not its runtime behavior.
//   Z source : CONST (ds_set 0.31)  vs  UNIFORM (runtime vec2 with a real lo word)
//   body     : UPDATE only          vs  UPDATE + escape/rebase block
import { chromium } from '@playwright/test';
import { resolveChromium } from './chromium-launch.mjs';
import { DF64_LIB } from '../src/gpu/glsl.js';

const SWIFT = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--disable-gpu','--enable-unsafe-swiftshader','--headless=new'];
const VULKAN = ['--no-sandbox','--disable-setuid-sandbox','--disable-dev-shm-usage','--use-angle=vulkan','--use-gl=angle','--enable-features=Vulkan','--ignore-gpu-blocklist','--enable-gpu','--headless=new'];

const UPDATE = `
    vec2 t1 = ds_sub(ds_mul(Zx, dx), ds_mul(Zy, dy));
    vec2 t2 = ds_add(ds_mul(Zx, dy), ds_mul(Zy, dx));
    vec2 sx = ds_sub(ds_mul(dx, dx), ds_mul(dy, dy));
    vec2 sy = ds_mul(dx, dy);
    dx = ds_add(ds_add(ds_add(t1, t1), sx), dcx);
    dy = ds_add(ds_add(ds_add(t2, t2), ds_add(sy, sy)), dcy);`;
const BLOCK = `
    vec2 zfx = ds_add(Zx, dx);
    vec2 zfy = ds_add(Zy, dy);
    float mag2 = ds_tofloat(ds_add(ds_mul(zfx, zfx), ds_mul(zfy, zfy)));
    float dz2 = ds_tofloat(ds_add(ds_mul(dx, dx), ds_mul(dy, dy)));
    if (mag2 < dz2) { dx = zfx; dy = zfy; }`;   // never true here (|Z| O(1) >> |dz|)

function frag(zUniform, withBlock) {
  return `#version 300 es
precision highp float; precision highp int;
${DF64_LIB}
uniform float uDcx, uDcy;
uniform int uK;
${zUniform ? 'uniform vec2 uZx, uZy;' : ''}
out vec4 frag;
void main(){
  ${zUniform ? 'vec2 Zx = uZx, Zy = uZy;' : 'vec2 Zx = ds_set(0.31), Zy = ds_set(0.43);'}
  vec2 dcx = ds_set(uDcx), dcy = ds_set(uDcy);
  vec2 dx = ds_set(0.0), dy = ds_set(0.0);
  for (int i = 0; i < 100000000; i++) {
    if (i >= uK) break;
${UPDATE}
${withBlock ? BLOCK : ''}
  }
  frag = vec4(dx.x, dx.y, dy.x, dy.y);
}`;
}

const CONFIGS = [
  { name: 'const,update    ', zU: false, block: false },
  { name: 'uniform,update  ', zU: true,  block: false },
  { name: 'const,+block    ', zU: false, block: true },
  { name: 'uniform,+block  ', zU: true,  block: true },
];

const RUN = (args) => {
  const { progs, dcx, dcy, K, zx, zxlo, zy, zylo } = args;
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
    gl.uniform1i(gl.getUniformLocation(prog, 'uK'), K);
    const lzx = gl.getUniformLocation(prog, 'uZx'); if (lzx) gl.uniform2f(lzx, zx, zxlo);
    const lzy = gl.getUniformLocation(prog, 'uZy'); if (lzy) gl.uniform2f(lzy, zy, zylo);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo); gl.viewport(0, 0, 1, 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    const px = new Float32Array(4); gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, px);
    out.res[p.name] = [px[0], px[1], px[2], px[3]];
  }
  return out;
};

function cpuIter(K, Zxd, Zyd, dcxd, dcyd, block) {
  let dx = 0, dy = 0;
  for (let i = 0; i < K; i++) {
    const t1 = Zxd*dx - Zyd*dy, t2 = Zxd*dy + Zyd*dx;
    const sx = dx*dx - dy*dy, sy = dx*dy;
    dx = 2*t1 + sx + dcxd; dy = 2*t2 + 2*sy + dcyd;
    if (block) {
      const zfx = Zxd+dx, zfy = Zyd+dy;
      const mag2 = zfx*zfx+zfy*zfy, dz2 = dx*dx+dy*dy;
      if (mag2 < dz2) { dx = zfx; dy = zfy; }
    }
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

const K = 40;
const dcxd = Math.fround(1e-3), dcyd = Math.fround(7e-4);
// uniform Z = 0.31,0.43 but with a realistic nonzero df64 lo word (like a texture ref)
const Zxd = 0.31, Zyd = 0.43;
const zx = Math.fround(Zxd), zxlo = Math.fround(Zxd - zx);
const zy = Math.fround(Zyd), zylo = Math.fround(Zyd - zy);
const progs = CONFIGS.map((cf) => ({ name: cf.name, frag: frag(cf.zU, cf.block) }));
const payload = { progs, dcx: dcxd, dcy: dcyd, K, zx, zxlo, zy, zylo };

const swift = await renderOn(SWIFT, payload);
const gpu = await renderOn(VULKAN, payload);
console.log(`renderer: ${gpu.renderer}   (K=${K})\n`);
console.log('config            GPU relerr-vs-oracle      Swift relerr-vs-oracle');
for (const cf of CONFIGS) {
  // const-Z uses fround(0.31); uniform-Z uses the full-double 0.31 reconstructed via lo
  const useZx = cf.zU ? (zx + zxlo) : Math.fround(0.31);
  const useZy = cf.zU ? (zy + zylo) : Math.fround(0.43);
  const [odx, ody] = cpuIter(K, useZx, useZy, dcxd, dcyd, cf.block);
  const mag = Math.hypot(odx, ody) || 1;
  const g = gpu.res[cf.name], s = swift.res[cf.name];
  const gerr = Math.hypot((g[0]+g[1])-odx, (g[2]+g[3])-ody) / mag;
  const serr = Math.hypot((s[0]+s[1])-odx, (s[2]+s[3])-ody) / mag;
  const v = (w) => w < 1e-10 ? 'INTACT' : (w < 1e-4 ? 'PARTIAL' : 'COLLAPSED');
  console.log(`${cf.name}  ${gerr.toExponential(2)} [${v(gerr).padEnd(9)}]    ${serr.toExponential(2)} [${v(serr)}]`);
}
