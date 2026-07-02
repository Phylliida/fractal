// probe-df64-real.mjs — test the ACTUAL shipping DF64_LIB (imported from
// src/gpu/glsl.js) on the selected backend, mirroring how the renderer compiles
// and sets uOptBarrier=0. Tells us definitively whether the real ds_mul keeps
// ~46-bit precision on this GPU, isolating the df64 collapse from the rest of the
// perturbation pipeline.
//
//   node tools/probe-df64-real.mjs           # SwiftShader
//   GPU=1 node tools/probe-df64-real.mjs     # real GPU (Vulkan)
import { chromium } from '@playwright/test';
import { spawn } from 'node:child_process';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

const PORT = process.env.PORT || 8156;
const server = spawn(process.execPath, ['tools/serve.mjs'], { env: { ...process.env, PORT }, stdio: 'ignore' });
const baseURL = `http://127.0.0.1:${PORT}`;
async function waitServer() {
  for (let i = 0; i < 100; i++) {
    try { const r = await fetch(baseURL + '/src/gpu/glsl.js'); if (r.ok) return; } catch { /* retry */ }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error('server did not start');
}

const RUN = async (setBarrier) => {
  const { DF64_LIB, VERT } = await import('/src/gpu/glsl.js');
  const c = document.createElement('canvas'); c.width = 1; c.height = 1;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const FRAG = `#version 300 es
precision highp float;
precision highp int;
${DF64_LIB}
uniform float uA, uB;
uniform int uMode;     // 0 = single ds_mul; 1 = 40-iter update; 2 = add tiny to O(1) (z=Z+dz path)
out vec4 outColor;
void main(){
  if (uMode == 0) {
    vec2 r = ds_mul(vec2(uA,0.0), vec2(uB,0.0));
    outColor = vec4(r.x, r.y, 0.0, 1.0);
  } else if (uMode == 2) {
    // The z = Z_m + dz path: add a TINY df64 to an O(1) df64 and recover it. This
    // is where the deep escape/rebase test lives (Z ~ O(1), dz ~ 2^-50). The two-sum
    // must keep the tiny addend; if it collapses, the recovered value is wrong.
    vec2 big = ds_set(1.0);
    vec2 tiny = ds_mul(vec2(uA,0.0), vec2(uB,0.0));   // ~1e-14 with a real low word
    vec2 z = ds_add(big, tiny);
    vec2 back = ds_sub(z, big);                       // should recover the tiny addend
    outColor = vec4(back.x, back.y, 0.0, 1.0);
  } else {
    // mimic perturbFragDf64's inner update with fixed Z and dc (no rebase)
    vec2 Zx = ds_set(0.31), Zy = ds_set(0.43);
    vec2 dcx = ds_set(uA), dcy = ds_set(uB);
    vec2 dx = ds_set(0.0), dy = ds_set(0.0);
    for (int i = 0; i < 40; i++) {
      vec2 t1 = ds_sub(ds_mul(Zx, dx), ds_mul(Zy, dy));
      vec2 t2 = ds_add(ds_mul(Zx, dy), ds_mul(Zy, dx));
      vec2 sx = ds_sub(ds_mul(dx, dx), ds_mul(dy, dy));
      vec2 sy = ds_mul(dx, dy);
      dx = ds_add(ds_add(ds_add(t1, t1), sx), dcx);
      dy = ds_add(ds_add(ds_add(t2, t2), ds_add(sy, sy)), dcy);
    }
    outColor = vec4(dx.x, dx.y, dy.x, dy.y);
  }
}`;
  function sh(t,s){ const o=gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw new Error('compile: '+gl.getShaderInfoLog(o)); return o; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VERT)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.bindAttribLocation(prog, 0, 'aPos'); gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: '+gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const obLoc = gl.getUniformLocation(prog, 'uOptBarrier');
  const obFound = obLoc !== null;
  if (obFound && setBarrier) gl.uniform1i(obLoc, 0);   // mirror the renderer
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0,2,gl.FLOAT,false,0,0);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,1,1,0,gl.RGBA,gl.FLOAT,null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0,0,1,1);
  const uA = gl.getUniformLocation(prog,'uA'), uB = gl.getUniformLocation(prog,'uB');
  const uMode = gl.getUniformLocation(prog,'uMode');
  function evalOne(a,b,mode){
    gl.uniform1f(uA,a); gl.uniform1f(uB,b); gl.uniform1i(uMode, mode);
    gl.drawArrays(gl.TRIANGLES,0,3);
    const px = new Float32Array(4); gl.readPixels(0,0,1,1,gl.RGBA,gl.FLOAT,px);
    return [px[0], px[1], px[2], px[3]];
  }
  const cases = [[1.0+Math.pow(2,-13),1.0+Math.pow(2,-13)],[1.3000001,0.7999999],[Math.PI,Math.E]];
  // dc for the iteration test: small offsets (float32-rounded on upload)
  const dc = [Math.fround(1e-3), Math.fround(7e-4)];
  // tiny addend test inputs (product ~1.4e-14, has sub-float32 structure)
  const tin = [Math.fround(1.3e-7), Math.fround(1.1e-7)];
  return { renderer, obFound,
    mul: cases.map(([a,b]) => ({ a, b, r: evalOne(a,b,0) })),
    iter: { dc, r: evalOne(dc[0], dc[1], 1) },
    tiny: { tin, r: evalOne(tin[0], tin[1], 2) } };
};

// CPU reference for the iteration test, in JS double (the truth).
function cpuIter(dcx, dcy) {
  const Zx = Math.fround(0.31), Zy = Math.fround(0.43);
  let dx = 0, dy = 0;
  for (let i = 0; i < 40; i++) {
    const t1 = Zx*dx - Zy*dy, t2 = Zx*dy + Zy*dx;
    const sx = dx*dx - dy*dy, sy = dx*dy;
    const ndx = 2*t1 + sx + dcx, ndy = 2*t2 + 2*sy + dcy;
    dx = ndx; dy = ndy;
  }
  return [dx, dy];
}

await waitServer();
const browser = await chromium.launch(launchOpts());
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
  await page.goto(baseURL + '/test/gpu/harness.html');   // a page on the served origin (for the dynamic import)
  const r = await page.evaluate(RUN, true);
  if (r.err) { console.error('ERROR:', r.err); }
  else {
    const fr = Math.fround;
    let mulWorst = 0;
    for (const res of r.mul) {
      const truth = fr(res.a) * fr(res.b);
      mulWorst = Math.max(mulWorst, Math.abs((res.r[0]+res.r[1]) - truth) / Math.abs(truth));
    }
    const [tdx, tdy] = cpuIter(r.iter.dc[0], r.iter.dc[1]);
    const gdx = r.iter.r[0] + r.iter.r[1], gdy = r.iter.r[2] + r.iter.r[3];
    const iterErr = Math.max(Math.abs(gdx - tdx) / Math.abs(tdx), Math.abs(gdy - tdy) / Math.abs(tdy));
    const tTruth = fr(r.tiny.tin[0]) * fr(r.tiny.tin[1]);
    const tBack = r.tiny.r[0] + r.tiny.r[1];
    const tinyErr = Math.abs(tBack - tTruth) / Math.abs(tTruth);
    const v = (w) => w < 1e-10 ? 'INTACT' : (w < 1e-5 ? 'PARTIAL' : 'COLLAPSED to f32');
    console.log(`mode=${gpuMode()} uOptBarrierFound=${r.obFound}   ${r.renderer}`);
    console.log(`  ds_mul single:        worst relerr=${mulWorst.toExponential(2)}  [${v(mulWorst)}]`);
    console.log(`  40-iter update (add): worst relerr=${iterErr.toExponential(2)}  [${v(iterErr)}]`);
    console.log(`  add tiny to O(1):     relerr=${tinyErr.toExponential(2)}  [${v(tinyErr)}]   (recovered ${tBack.toExponential(4)} vs ${tTruth.toExponential(4)})`);
  }
} finally {
  await browser.close();
  server.kill();
}
