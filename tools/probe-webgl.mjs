// Probe: does WebGL2 (+ float textures, render-to-float) work in this sandbox's
// headless chromium via SwiftShader? Load-bearing for the GPU renderer plan.
import { chromium } from '@playwright/test';
import { chromiumArgs } from './chromium-launch.mjs';
import { readdirSync, existsSync } from 'node:fs';

function resolveChromium() {
  if (process.env.CHROMIUM_PATH && existsSync(process.env.CHROMIUM_PATH)) return process.env.CHROMIUM_PATH;
  const dirs = readdirSync('/nix/store').filter((d) => /-chromium-\d/.test(d) && !d.includes('sandbox')).sort().reverse();
  for (const d of dirs) { const p = `/nix/store/${d}/bin/chromium`; if (existsSync(p)) return p; }
  return undefined;
}

const browser = await chromium.launch({
  executablePath: resolveChromium(),
  args: chromiumArgs(),
});
const page = await browser.newPage();
page.on('console', (m) => console.log('  [page]', m.text()));
const result = await page.evaluate(() => {
  const c = document.createElement('canvas');
  c.width = 64; c.height = 64;
  const gl = c.getContext('webgl2', { antialias: false, preserveDrawingBuffer: true });
  if (!gl) return { webgl2: false };
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  const info = {
    webgl2: true,
    vendor: dbg ? gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) : '?',
    renderer: dbg ? gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : '?',
    maxTexSize: gl.getParameter(gl.MAX_TEXTURE_SIZE),
    maxTexUnits: gl.getParameter(gl.MAX_COMBINED_TEXTURE_IMAGE_UNITS),
    colorBufferFloat: !!gl.getExtension('EXT_color_buffer_float'),
    floatBlend: !!gl.getExtension('EXT_float_blend'),
    highpFragment: gl.getShaderPrecisionFormat(gl.FRAGMENT_SHADER, gl.HIGH_FLOAT).precision,
  };
  // Render a trivial shader to the default framebuffer, read a pixel back.
  const vs = `#version 300 es
  in vec2 p; void main(){ gl_Position = vec4(p,0.,1.); }`;
  const fs = `#version 300 es
  precision highp float; out vec4 o; uniform vec2 res;
  void main(){ vec2 uv = gl_FragCoord.xy/res; o = vec4(uv, 0.5, 1.0); }`;
  function sh(t, s){ const o = gl.createShader(t); gl.shaderSource(o,s); gl.compileShader(o);
    if(!gl.getShaderParameter(o,gl.COMPILE_STATUS)) throw new Error('shader: '+gl.getShaderInfoLog(o)); return o; }
  const prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(prog);
  if(!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error('link: '+gl.getProgramInfoLog(prog));
  gl.useProgram(prog);
  const buf = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 3,-1, -1,3]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'p'); gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.uniform2f(gl.getUniformLocation(prog,'res'), 64, 64);
  gl.viewport(0,0,64,64); gl.drawArrays(gl.TRIANGLES, 0, 3);
  const px = new Uint8Array(4); gl.readPixels(32,32,1,1,gl.RGBA,gl.UNSIGNED_BYTE,px);
  info.centerPixel = Array.from(px);
  return info;
});
console.log(JSON.stringify(result, null, 2));
await browser.close();
