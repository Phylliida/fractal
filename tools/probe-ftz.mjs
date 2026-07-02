// probe-ftz.mjs — does this GL backend flush subnormal float32 to zero (FTZ)?
// NVIDIA GPUs typically do; SwiftShader typically doesn't. df64 error terms can
// land in the subnormal range at deep zoom, so FTZ silently destroys precision.
// Self-contained (about:blank).
import { chromium } from '@playwright/test';
import { launchOpts, gpuMode } from './chromium-launch.mjs';

const FRAG = `#version 300 es
precision highp float;
precision highp int;
uniform float uA, uB;
uniform int uMode;
out vec4 outColor;
void main(){
  float r;
  if (uMode == 0) r = uA * uB;              // product underflows to subnormal? (uA=uB=1e-20 -> 1e-40)
  else if (uMode == 1) r = uA + uB;         // subnormal + 0 preserved?
  else r = uA;                              // can a subnormal even be loaded?
  outColor = vec4(r, 0.0, 0.0, 1.0);
}`;

const RUN = (FRAG) => {
  const c = document.createElement('canvas'); c.width = 1; c.height = 1;
  const gl = c.getContext('webgl2', { antialias: false });
  if (!gl.getExtension('EXT_color_buffer_float')) return { err: 'no float color buffer' };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const renderer = dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?';
  const VS = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  function sh(t,s){ const o=gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw new Error('compile: '+gl.getShaderInfoLog(o)); return o; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, VS)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, FRAG));
  gl.linkProgram(prog); gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,3,-1,-1,3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog,'p'); gl.enableVertexAttribArray(loc); gl.vertexAttribPointer(loc,2,gl.FLOAT,false,0,0);
  const tex = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,1,1,0,gl.RGBA,gl.FLOAT,null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  const fbo = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  gl.viewport(0,0,1,1);
  const uA = gl.getUniformLocation(prog,'uA'), uB = gl.getUniformLocation(prog,'uB'), uMode = gl.getUniformLocation(prog,'uMode');
  function evalOne(a,b,mode){
    gl.uniform1f(uA,a); gl.uniform1f(uB,b); gl.uniform1i(uMode,mode);
    gl.drawArrays(gl.TRIANGLES,0,3);
    const px = new Float32Array(4); gl.readPixels(0,0,1,1,gl.RGBA,gl.FLOAT,px);
    return px[0];
  }
  const sub = 1e-40;        // a subnormal float32 (min normal ~1.18e-38)
  return { renderer,
    loadSub: evalOne(sub, 0, 2),       // can a subnormal be carried at all?
    addSub: evalOne(sub, 0, 1),        // subnormal + 0
    mulUnderflow: evalOne(1e-20, 1e-20, 0),  // 1e-40 subnormal product
    mulNormal: evalOne(1e-15, 1e-15, 0),     // 1e-30 normal product (control)
  };
};

const browser = await chromium.launch(launchOpts());
try {
  const page = await browser.newPage();
  page.on('pageerror', (e) => console.error('PAGE ERROR:', e.message));
  await page.goto('about:blank');
  const r = await page.evaluate(RUN, FRAG);
  if (r.err) { console.error('ERROR:', r.err); }
  else {
    const ftz = (r.loadSub === 0 || r.mulUnderflow === 0);
    console.log(`mode=${gpuMode()}  ${r.renderer}`);
    console.log(`  load subnormal 1e-40  -> ${r.loadSub}`);
    console.log(`  subnormal + 0         -> ${r.addSub}`);
    console.log(`  1e-20 * 1e-20 (=1e-40)-> ${r.mulUnderflow}   ${r.mulUnderflow === 0 ? '(FLUSHED)' : '(preserved)'}`);
    console.log(`  1e-15 * 1e-15 (=1e-30)-> ${r.mulNormal}   (control, normal)`);
    console.log(`  => ${ftz ? 'FTZ: subnormals are FLUSHED to zero on this backend' : 'subnormals PRESERVED on this backend'}`);
  }
} finally { await browser.close(); }
