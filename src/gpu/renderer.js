// renderer.js — WebGL2 Mandelbrot renderer (naive float32 / naive df64 / perturb).
//
// Renders an escape program into an RGBA32F float framebuffer (the "sn texture":
// .r = smooth count, .g = iteration, .b = glitch). Two consumers:
//   - colorize(): a cheap color pass maps the sn texture -> RGBA on the GL canvas
//     for display (no CPU readback — the whole point of the GPU path).
//   - readSn(): reads the float buffer back to the CPU for validation against the
//     naive / perturbation / BigInt oracles.
//
// The renderer owns its own canvas + GL context so it can run standalone (tests)
// or be blitted into the viewer's 2-D canvas.

import { VERT, naiveFrag, perturbFrag, perturbFragDf64, perturbFragFloatexp, perturbFragRescaled, COLOR_FRAG } from './glsl.js';

// Split a double into two float32 (df64 hi/lo) preserving ~46 bits.
export function df64Split(v) {
  const hi = Math.fround(v);
  const lo = Math.fround(v - hi);
  return [hi, lo];
}

// Split a double into floatexp { hi, lo, e }: value = (hi+lo)*2^e with the df64
// mantissa (hi,lo) normalized so |hi| in [0.5,1). Mirrors the shader's fe form so
// tiny (~2^-270) deltas survive being passed in as float32 mantissa + int exponent.
export function feSplit(v) {
  if (v === 0 || !Number.isFinite(v)) return { hi: 0, lo: 0, e: -100000 };
  let a = Math.abs(v), e = 0;
  while (a >= 1) { a *= 0.5; e++; }      // exact (powers of two)
  while (a < 0.5) { a *= 2; e--; }
  const mant = v < 0 ? -a : a;
  const hi = Math.fround(mant);
  const lo = Math.fround(mant - hi);
  return { hi, lo, e };
}

function makeCanvas(w, h) {
  if (typeof OffscreenCanvas !== 'undefined' && typeof document === 'undefined') {
    return new OffscreenCanvas(w, h);
  }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  return c;
}

export class GpuRenderer {
  constructor(opts = {}) {
    this.canvas = opts.canvas || makeCanvas(opts.width || 1, opts.height || 1);
    // WebGL context loss/restore callbacks. Mobile GPUs drop the context under
    // memory pressure or when the tab is backgrounded; the viewer wires these to
    // fall back to the CPU worker pool while lost and re-render on the GPU once
    // restored. Default null so the renderer still runs standalone in tools/tests.
    this.onContextLost = opts.onContextLost || null;
    this.onContextRestored = opts.onContextRestored || null;
    this._contextLost = false;
    this._disposed = false;
    const attrs = { antialias: false, depth: false, stencil: false, preserveDrawingBuffer: true, premultipliedAlpha: false };
    this.gl = this.canvas.getContext('webgl2', attrs);
    this.ok = !!this.gl;
    this._programs = {};
    this._fboW = 0; this._fboH = 0;
    if (this.ok) {
      this._initGLObjects();
      this._installContextListeners();
    }
  }

  get supported() { return this.ok && !!this.extColorFloat; }

  // True while the GL context is lost. The viewer gates GPU engine-selection on
  // this so renders route to the CPU pool until the context is restored.
  get lost() { return this._contextLost; }

  // (Re)acquire the float-render extension + the fullscreen-triangle VAO. Called at
  // construction and again after the context is restored (every prior GL object —
  // programs, textures, FBO, VAO — is invalid once the context was lost).
  _initGLObjects() {
    const gl = this.gl;
    this.extColorFloat = gl.getExtension('EXT_color_buffer_float');
    this.extFloatLinear = gl.getExtension('OES_texture_float_linear');
    this._initQuad();
  }

  _installContextListeners() {
    if (typeof this.canvas.addEventListener !== 'function') return;
    this._onCtxLost = (e) => {
      if (this._disposed) return;          // intentional teardown (dispose) — let it die
      e.preventDefault();                  // REQUIRED so the browser will fire 'restored'
      this._contextLost = true;
      this._dropGLState();                 // every cached GL handle is now invalid
      if (this.onContextLost) { try { this.onContextLost(); } catch { /* noop */ } }
    };
    this._onCtxRestored = () => {
      if (this._disposed) return;
      this._initGLObjects();               // re-fetch extensions + recreate the VAO
      this._contextLost = false;           // programs/textures/FBO recreated lazily on next render
      if (this.onContextRestored) { try { this.onContextRestored(); } catch { /* noop */ } }
    };
    this.canvas.addEventListener('webglcontextlost', this._onCtxLost, false);
    this.canvas.addEventListener('webglcontextrestored', this._onCtxRestored, false);
  }

  // Forget every cached GL object handle — after a context loss they are all invalid.
  // Nulling them (not just zeroing sizes) lets the lazy-init guards in _program /
  // _ensureFbo / uploadReference* / setPaletteLUT recreate them on the next render.
  _dropGLState() {
    this._programs = {};
    this._vao = null;
    this._snTex = null; this._fbo = null; this._fboW = 0; this._fboH = 0;
    this._refTex = null; this._refW = 0;
    this._palTex = null;
    this._blaTex = null; this._blaW = 0; this._blaLen = 0; this._blaMaxLevel = 0;
    this._dummy = null;
    this._maskTex = null; this._maskW = 0; this._maskH = 0;
  }

  _initQuad() {
    const gl = this.gl;
    this._vao = gl.createVertexArray();
    gl.bindVertexArray(this._vao);
    const buf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, buf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    // aPos is location 0 in every program (we bind it explicitly at link).
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  _compile(type, src) {
    const gl = this.gl;
    const sh = gl.createShader(type);
    gl.shaderSource(sh, src);
    gl.compileShader(sh);
    if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
      const log = gl.getShaderInfoLog(sh);
      gl.deleteShader(sh);
      throw new Error('shader compile failed: ' + log + '\n' + numberLines(src));
    }
    return sh;
  }

  _program(key, fragSrc) {
    if (this._programs[key]) return this._programs[key];
    const gl = this.gl;
    const prog = gl.createProgram();
    gl.attachShader(prog, this._compile(gl.VERTEX_SHADER, VERT));
    gl.attachShader(prog, this._compile(gl.FRAGMENT_SHADER, fragSrc));
    gl.bindAttribLocation(prog, 0, 'aPos');
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      const log = gl.getProgramInfoLog(prog);
      throw new Error('program link failed: ' + log);
    }
    // cache uniform locations on demand via a proxy map
    const cache = {};
    const u = (name) => (cache[name] ??= gl.getUniformLocation(prog, name));
    // df64 optimization barrier (glsl.js ob()): uOptBarrier must be 0. Set ONCE
    // here — it is per-program state that persists across useProgram switches.
    // Programs without df64 don't declare it (location null -> silent no-op).
    const obLoc = gl.getUniformLocation(prog, 'uOptBarrier');
    if (obLoc) { gl.useProgram(prog); gl.uniform1i(obLoc, 0); }
    this._programs[key] = { prog, u };
    return this._programs[key];
  }

  // (Re)allocate the RGBA32F render target at w×h. This is the COMPUTE resolution
  // (uSS× the display res when supersampling); the display canvas is sized
  // separately in colorize(). We deliberately do NOT resize the canvas here.
  _ensureFbo(w, h) {
    const gl = this.gl;
    if (this._fboW === w && this._fboH === h && this._fbo) return;
    if (this._snTex) gl.deleteTexture(this._snTex);
    if (this._fbo) gl.deleteFramebuffer(this._fbo);
    this._snTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._snTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, w, h, 0, gl.RGBA, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._fbo = gl.createFramebuffer();
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, this._snTex, 0);
    const st = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
    if (st !== gl.FRAMEBUFFER_COMPLETE) throw new Error('float FBO incomplete: 0x' + st.toString(16));
    this._fboW = w; this._fboH = h;
  }

  _setReal(u, name, df64, value) {
    const gl = this.gl;
    if (df64) { const [hi, lo] = df64Split(value); gl.uniform2f(u(name), hi, lo); }
    else gl.uniform1f(u(name), value);
  }

  // Bind the sn FBO as the escape render target. The VIEWPORT always covers the
  // full FBO (so gl_FragCoord — and therefore the per-pixel c — is identical to a
  // single full-frame draw), and an optional SCISSOR rectangle restricts which rows
  // are actually written. That lets a long deep escape pass be split into short
  // horizontal strips (p.stripY/p.stripH) across multiple draw calls — each well
  // under a GPU watchdog (TDR) timeout — while staying BIT-IDENTICAL to one big draw.
  _bindEscapeTarget(p) {
    const gl = this.gl;
    this._ensureFbo(p.width, p.height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
    gl.viewport(0, 0, p.width, p.height);
    if (p.stripH != null) { gl.enable(gl.SCISSOR_TEST); gl.scissor(0, p.stripY | 0, p.width, p.stripH); }
    else gl.disable(gl.SCISSOR_TEST);
  }

  // Clear the sn target to an interior sentinel (sn = -1, iter = 0). Strip-tiled
  // renders clear once up front so not-yet-computed rows read as interior (giving a
  // clean top-to-bottom progressive reveal instead of garbage).
  clearSn(width, height, snVal = -1) {
    const gl = this.gl;
    this._ensureFbo(width, height);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
    gl.disable(gl.SCISSOR_TEST);
    gl.viewport(0, 0, width, height);
    gl.clearColor(snVal, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
  }

  // Render the naive escape program into the sn texture.
  //   params: { ox, oy, scale, maxIter, bailoutSq=65536, df64=false }
  renderNaive(p) {
    const gl = this.gl;
    const df64 = !!p.df64;
    const { prog, u } = this._program(df64 ? 'naive64' : 'naive32', naiveFrag(df64));
    this._bindEscapeTarget(p);
    gl.useProgram(prog);
    gl.bindVertexArray(this._vao);
    this._setReal(u, 'uOx', df64, p.ox);
    this._setReal(u, 'uOy', df64, p.oy);
    this._setReal(u, 'uScale', df64, p.scale);
    gl.uniform1i(u('uMaxIter'), p.maxIter);
    gl.uniform1f(u('uBailoutSq'), p.bailoutSq ?? (1 << 16));
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Upload the reference orbit (Float32Array zx, zy) into an RG32F texture.
  uploadReference(zx, zy, refW = 2048) {
    const gl = this.gl;
    const len = zx.length;
    const rows = Math.ceil(len / refW);
    const data = new Float32Array(refW * rows * 2);
    for (let i = 0; i < len; i++) { data[i * 2] = zx[i]; data[i * 2 + 1] = zy[i]; }
    if (this._refTex) gl.deleteTexture(this._refTex);
    this._refTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._refTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG32F, refW, rows, 0, gl.RG, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._refW = refW;
    return { refW, rows };
  }

  // Upload the reference orbit as df64 (RGBA32F: Zx.hi,Zx.lo,Zy.hi,Zy.lo) for the
  // double-single perturbation shader. Each double Z is split into two float32.
  uploadReferenceDf64(zx, zy, refW = 2048) {
    const gl = this.gl;
    const len = zx.length;
    const rows = Math.ceil(len / refW);
    const data = new Float32Array(refW * rows * 4);
    for (let i = 0; i < len; i++) {
      const xh = Math.fround(zx[i]); const yh = Math.fround(zy[i]);
      data[i * 4] = xh; data[i * 4 + 1] = Math.fround(zx[i] - xh);
      data[i * 4 + 2] = yh; data[i * 4 + 3] = Math.fround(zy[i] - yh);
    }
    if (this._refTex) gl.deleteTexture(this._refTex);
    this._refTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._refTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, refW, rows, 0, gl.RGBA, gl.FLOAT, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._refW = refW;
    return { refW, rows };
  }

  // Upload the BLA table (bla.blaToFloat32 output: { data, texW, texH, maxLevel, len })
  // as a highp NEAREST RGBA32F texture for the rescaled deep shader's BLA scan. NEAREST +
  // the shader's `highp sampler2D uBla` are REQUIRED — a filtered/mediump fetch corrupts the
  // df64 coefficients (Spawn 9 class bug). The table is large at extreme depth (~tens of MB);
  // delete the prior one first. Stays uploaded across strip draws of one view.
  uploadBLA(packed) {
    const gl = this.gl;
    if (this._blaTex) gl.deleteTexture(this._blaTex);
    this._blaTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._blaTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, packed.texW, packed.texH, 0, gl.RGBA, gl.FLOAT, packed.data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._blaW = packed.texW; this._blaLen = packed.len; this._blaMaxLevel = packed.maxLevel;
    return { texW: packed.texW, texH: packed.texH, maxLevel: packed.maxLevel };
  }

  // Upload the per-pixel interior-prune mask (Spawn 23, measurement-only). `mask` is a
  // Float32Array of length w*h, row-major with row 0 = the GL-BOTTOM row (matching readSn /
  // validate.js texelC), value 1.0 = "interior pixel (runs to maxIter)". Stored as a NEAREST
  // R32F texture sampled once per pixel by perturbFragRescaled's uPruneMask when uPrune==1.
  uploadPruneMask(mask, w, h) {
    const gl = this.gl;
    if (this._maskTex) gl.deleteTexture(this._maskTex);
    this._maskTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._maskTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, w, h, 0, gl.RED, gl.FLOAT, mask);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this._maskW = w; this._maskH = h;
  }

  // A 1×1 RGBA32F texture to bind to the uBla sampler when BLA is OFF (a sampler must have a
  // valid texture bound even though the shader never fetches it with uBlaMaxLevel == 0).
  _dummyTex() {
    if (this._dummy) return this._dummy;
    const gl = this.gl;
    this._dummy = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._dummy);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 1, 1, 0, gl.RGBA, gl.FLOAT, new Float32Array(4));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    return this._dummy;
  }

  // df64 perturbation. Requires uploadReferenceDf64() first. Same params as
  // renderPerturb; ox/oy/scale are split to df64 internally.
  renderPerturbDf64(p) {
    const gl = this.gl;
    // p.sqrOn / p.prefetch compile the ds_sqr / software-pipelined-getZ variants —
    // BOTH OPT-IN here (measured Spawn 27: sqr 1.00× GPU / 0.95–0.98× SwiftShader;
    // prefetch 0.997× GPU / 0.957× SwiftShader in THIS engine — its tight loop has
    // too little work to overlap, unlike the rescaled engine where prefetch is ~2×
    // on SwiftShader and IS the default). bench-sqr / bench-prefetch are the A/Bs.
    const dOpts = {
      ...(p.sqrOn === true ? { sqr: true } : {}),
      ...(p.prefetch === true ? { prefetch: true } : {}),
    };
    const dk = (p.sqrOn === true ? 'sqr' : '') + (p.prefetch === true ? 'pf' : '');
    const { prog, u } = this._program('perturb64' + dk, perturbFragDf64(dOpts));
    this._bindEscapeTarget(p);
    gl.useProgram(prog);
    gl.bindVertexArray(this._vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._refTex);
    gl.uniform1i(u('uRef'), 0);
    gl.uniform1i(u('uRefW'), this._refW);
    gl.uniform1i(u('uRefLen'), p.refLen);
    this._setReal(u, 'uOx', true, p.ox);
    this._setReal(u, 'uOy', true, p.oy);
    this._setReal(u, 'uScale', true, p.scale);
    gl.uniform1i(u('uMaxIter'), p.maxIter);
    gl.uniform1f(u('uBailoutSq'), p.bailoutSq ?? (1 << 16));
    gl.uniform1f(u('uGlitchTol'), p.glitchTol ?? 0);
    gl.uniform1i(u('uFastSkip'), p.fastSkip === 0 ? 0 : 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  _setFe(u, mName, eName, value) {
    const { hi, lo, e } = feSplit(value);
    this.gl.uniform2f(u(mName), hi, lo);
    this.gl.uniform1i(u(eName), e);
  }

  // Series-approximation uniforms (rescaled deep engine). `sa` is computeSeries()'s
  // result { skip, invR, ax..ey }; the shader seeds dz at iteration `skip` via an order-5
  // fe Horner. Coeffs + 1/R are passed as floatexp (they reach ~2^-270 at deep zoom).
  // uSASkip=0 (sa null, skip 0, or skip out of range) = the no-SA path.
  _setSA(u, sa, maxIter) {
    const gl = this.gl;
    const active = !!(sa && sa.skip > 0 && sa.skip < maxIter);
    gl.uniform1i(u('uSASkip'), active ? sa.skip : 0);
    if (!active) return;
    // Order-5 SA: a,b,c,d,e (10 reals). d,e are 0 for an order≤3 sa → bit-identical seed.
    const coeffs = [sa.ax, sa.ay, sa.bx, sa.by, sa.cx, sa.cy,
                    sa.dx ?? 0, sa.dy ?? 0, sa.ex ?? 0, sa.ey ?? 0];
    const m = new Float32Array(20), e = new Int32Array(10);
    for (let i = 0; i < 10; i++) { const f = feSplit(coeffs[i]); m[i * 2] = f.hi; m[i * 2 + 1] = f.lo; e[i] = f.e; }
    gl.uniform2fv(u('uSAm'), m);
    gl.uniform1iv(u('uSAe'), e);
    const ir = feSplit(sa.invR);
    gl.uniform2f(u('uInvRm'), ir.hi, ir.lo); gl.uniform1i(u('uInvRe'), ir.e);
  }

  // floatexp perturbation: like renderPerturbDf64 but dc origin/step are passed as
  // floatexp (df64 mantissa + int exponent), so it works below the df64 float32-
  // exponent floor (~2^-112) down to the double exponent range — the GPU deep path
  // for ~2^270 zooms. Requires uploadReferenceDf64() first (same df64 reference).
  renderPerturbFloatexp(p) {
    const gl = this.gl;
    // p.sqrOn / p.prefetch — both OPT-IN here, see renderPerturbDf64 (the fe engine is
    // the non-default oracle/fallback path; nothing production-critical to win).
    const feOpts = {
      ...(p.sqrOn === true ? { sqr: true } : {}),
      ...(p.prefetch === true ? { prefetch: true } : {}),
    };
    const fek = (p.sqrOn === true ? 'sqr' : '') + (p.prefetch === true ? 'pf' : '');
    const { prog, u } = this._program('perturbFE' + fek, perturbFragFloatexp(feOpts));
    this._bindEscapeTarget(p);
    gl.useProgram(prog);
    gl.bindVertexArray(this._vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._refTex);
    gl.uniform1i(u('uRef'), 0);
    gl.uniform1i(u('uRefW'), this._refW);
    gl.uniform1i(u('uRefLen'), p.refLen);
    this._setFe(u, 'uOxm', 'uOxe', p.ox);
    this._setFe(u, 'uOym', 'uOye', p.oy);
    this._setFe(u, 'uScalem', 'uScalee', p.scale);
    gl.uniform1i(u('uMaxIter'), p.maxIter);
    gl.uniform1f(u('uBailoutSq'), p.bailoutSq ?? (1 << 16));
    gl.uniform1f(u('uGlitchTol'), p.glitchTol ?? 0);
    gl.uniform1i(u('uFastSkip'), p.fastSkip === 0 ? 0 : 1);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Rescaled single-exponent perturbation: same depth range + uniforms as
  // renderPerturbFloatexp (df64 reference, fe dc origin/step), but the faster
  // shared-exponent update. Requires uploadReferenceDf64() first. Optional p.sa
  // (computeSeries result) seeds dz at iteration sa.skip (series approximation) —
  // a huge deep-zoom win; null/skip 0 = the original full-iteration path.
  renderPerturbRescaled(p) {
    const gl = this.gl;
    // Shader variant (Spawn 22). The FULL kernel is the production default — it MEASURED FASTEST
    // on the real RTX 3090. A LEAN variant (the never-taken BLA scan + df64-escape blocks excluded
    // at build time; BIT-IDENTICAL output in production config) was built to test whether those
    // dead blocks throttle occupancy via the kernel's PEAK register count — but on the RTX 3090 it
    // was NEUTRAL-to-slightly-SLOWER (bench-lean.mjs: ~0.93–1.00×, reproducibly ~0.8× at 2^-271 —
    // the compiler's codegen for the smaller shader is no better, occupancy isn't the bottleneck).
    // So lean is OPT-IN (`p.lean`), kept only for future measurement on a register-constrained
    // mobile GPU where occupancy MIGHT bind. See NOTES "LEAN KERNEL". The full kernel is also
    // REQUIRED whenever BLA or df64-escape is actually used (those blocks live only in it).
    const useBla = !!p.bla && !!this._blaTex && this._blaMaxLevel > 0;
    const useDesc = p.df64Esc === 1;
    const useLean = p.lean === true && !useBla && !useDesc;
    const full = !useLean;
    // p.sqrOn compiles the dedicated ds_sqr/fe_sqr variant (opt-in — measured neutral
    // on the RTX 3090, slightly negative on SwiftShader; bench-sqr.mjs is the A/B).
    // Software-pipelined getZ prefetch is the DEFAULT (Spawn 27): bit-identical output
    // (same fetched values, same arithmetic — crossArgs sn diff 0 at every depth),
    // measured 1.01–1.08× on the RTX 3090 (growing with depth) and ~2.0× on
    // SwiftShader. p.prefetch === false compiles the old immediate-use fetch
    // placement (A/B baseline — bench-prefetch.mjs).
    const usePf = p.prefetch !== false;
    const vOpts = {
      ...(p.sqrOn === true ? { sqr: true } : {}),
      ...(usePf ? { prefetch: true } : {}),
    };
    const sk = (p.sqrOn === true ? 'sqr' : '') + (usePf ? 'pf' : '');
    const { prog, u } = useLean
      ? this._program('perturbRSlean' + sk, perturbFragRescaled({ bla: false, df64esc: false, ...vOpts }))
      : this._program('perturbRS' + sk, perturbFragRescaled(vOpts));
    this._bindEscapeTarget(p);
    gl.useProgram(prog);
    gl.bindVertexArray(this._vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._refTex);
    gl.uniform1i(u('uRef'), 0);
    gl.uniform1i(u('uRefW'), this._refW);
    gl.uniform1i(u('uRefLen'), p.refLen);
    this._setFe(u, 'uOxm', 'uOxe', p.ox);
    this._setFe(u, 'uOym', 'uOye', p.oy);
    this._setFe(u, 'uScalem', 'uScalee', p.scale);
    gl.uniform1i(u('uMaxIter'), p.maxIter);
    gl.uniform1f(u('uBailoutSq'), p.bailoutSq ?? (1 << 16));
    gl.uniform1f(u('uGlitchTol'), p.glitchTol ?? 0);
    gl.uniform1i(u('uFastSkip'), p.fastSkip === 0 ? 0 : 1);
    this._setSA(u, p.sa, p.maxIter);
    // Interior-prune (Spawn 23, measurement-only). p.prune off (default) → uPrune=0, the shader
    // short-circuits the mask fetch → bit-identical to production. When on, bind the oracle mask
    // (uploadPruneMask) to unit 2; pixels flagged interior bail at p.pruneIter. See NOTES.
    gl.uniform1i(u('uPrune'), p.prune ? 1 : 0);
    gl.uniform1i(u('uPruneIter'), p.pruneIter ?? p.maxIter);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, (p.prune && this._maskTex) ? this._maskTex : this._dummyTex());
    gl.uniform1i(u('uPruneMask'), 2);
    // The BLA + df64-escape uniforms exist only in the FULL kernel (the lean kernel omits them;
    // their uniform locations are null, so the calls would be silent no-ops anyway — but only
    // the full kernel ever needs them, and only the full kernel fetches the uBla sampler).
    if (full) {
      // df64 escape/rebase fast-path. OPT-IN (default off): measured bit-identical-correct on
      // SwiftShader + the real RTX 3090 but performance-NEUTRAL (~1.00×). See NOTES "df64 ESCAPE".
      gl.uniform1i(u('uDf64Esc'), useDesc ? 1 : 0);
      // BLA: bind the uploaded table to unit 1 when requested (p.bla truthy + uploadBLA done),
      // else bind a dummy + uBlaMaxLevel 0 so the shader's BLA scan never runs (no-BLA path).
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, useBla ? this._blaTex : this._dummyTex());
      gl.uniform1i(u('uBla'), 1);
      gl.uniform1i(u('uBlaW'), useBla ? this._blaW : 1);
      gl.uniform1i(u('uBlaLen'), useBla ? this._blaLen : 1);
      gl.uniform1i(u('uBlaMaxLevel'), useBla ? this._blaMaxLevel : 0);
    }
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Render the perturbation program. Requires uploadReference() first.
  //   params: { ox, oy, scale, refLen, maxIter, bailoutSq, glitchTol }
  renderPerturb(p) {
    const gl = this.gl;
    const { prog, u } = this._program('perturb', perturbFrag());
    this._bindEscapeTarget(p);
    gl.useProgram(prog);
    gl.bindVertexArray(this._vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this._refTex);
    gl.uniform1i(u('uRef'), 0);
    gl.uniform1i(u('uRefW'), this._refW);
    gl.uniform1i(u('uRefLen'), p.refLen);
    gl.uniform1f(u('uOx'), p.ox);
    gl.uniform1f(u('uOy'), p.oy);
    gl.uniform1f(u('uScale'), p.scale);
    gl.uniform1i(u('uMaxIter'), p.maxIter);
    gl.uniform1f(u('uBailoutSq'), p.bailoutSq ?? (1 << 16));
    gl.uniform1f(u('uGlitchTol'), p.glitchTol ?? 0);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Read the float sn buffer back (RGBA32F). Returns { sn, iter, glitch } as
  // Float32Arrays length w*h, in GL row order (row 0 = bottom).
  readSn() {
    const gl = this.gl;
    const w = this._fboW, h = this._fboH;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
    const buf = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    const sn = new Float32Array(w * h), iter = new Float32Array(w * h), glitch = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) { sn[i] = buf[i * 4]; iter[i] = buf[i * 4 + 1]; glitch[i] = buf[i * 4 + 2]; }
    return { sn, iter, glitch, w, h };
  }

  // Build a 1×N RGBA8 palette LUT texture from a colorFn(t in [0,1)) -> [r,g,b].
  setPaletteLUT(rgbAt, n = 1024) {
    const gl = this.gl;
    const data = new Uint8Array(n * 4);
    for (let i = 0; i < n; i++) {
      const [r, g, b] = rgbAt((i + 0.5) / n);
      data[i * 4] = r; data[i * 4 + 1] = g; data[i * 4 + 2] = b; data[i * 4 + 3] = 255;
    }
    if (!this._palTex) this._palTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, this._palTex);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, n, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, data);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  // Color pass: sn texture + palette LUT -> RGBA on the GL canvas (the default
  // framebuffer). opts: { cycle, shift, interior:[r,g,b] 0..255, ss }.
  //   ss (supersample factor, default 1): the sn texture is ss× the display res;
  //   each output pixel box-averages its ss×ss colored subsamples. The output
  //   (canvas) size is the compute size / ss.
  colorize(opts) {
    const gl = this.gl;
    const ss = Math.max(1, Math.round(opts.ss || 1));
    const outW = Math.max(1, Math.round(this._fboW / ss));
    const outH = Math.max(1, Math.round(this._fboH / ss));
    if (this.canvas.width !== outW || this.canvas.height !== outH) {
      this.canvas.width = outW; this.canvas.height = outH;
    }
    const { prog, u } = this._program('color', COLOR_FRAG);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.disable(gl.SCISSOR_TEST);   // a prior strip-tiled escape pass may have left it on
    gl.viewport(0, 0, outW, outH);
    gl.useProgram(prog);
    gl.bindVertexArray(this._vao);
    gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, this._snTex);
    gl.uniform1i(u('uSn'), 0);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this._palTex);
    gl.uniform1i(u('uPalette'), 1);
    gl.uniform1f(u('uCycle'), opts.cycle || 64);
    gl.uniform1f(u('uShift'), opts.shift || 0);
    gl.uniform1i(u('uSS'), ss);
    gl.uniform1i(u('uShowGlitch'), opts.showGlitch ? 1 : 0);
    const it = opts.interior || [0, 0, 0];
    gl.uniform3f(u('uInterior'), it[0] / 255, it[1] / 255, it[2] / 255);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  // Count pixels the perturbation shader flagged as a Pauldelbrot glitch (.b > 0.5)
  // in the current sn FBO. A full RGBA32F readback (slow) — call only when the glitch
  // debug overlay is enabled. Counts at COMPUTE resolution (ss× display), matching the
  // CPU worker's per-compute-pixel glitch count. Returns 0 if there is no sn buffer yet.
  countGlitches() {
    const gl = this.gl;
    if (!this._fbo) return 0;
    const w = this._fboW, h = this._fboH;
    gl.bindFramebuffer(gl.FRAMEBUFFER, this._fbo);
    const buf = new Float32Array(w * h * 4);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.FLOAT, buf);
    let n = 0;
    for (let i = 0; i < w * h; i++) if (buf[i * 4 + 2] > 0.5) n++;
    return n;
  }

  dispose() {
    this._disposed = true;   // so the loseContext() below doesn't trip the recovery path
    if (typeof this.canvas.removeEventListener === 'function' && this._onCtxLost) {
      this.canvas.removeEventListener('webglcontextlost', this._onCtxLost);
      this.canvas.removeEventListener('webglcontextrestored', this._onCtxRestored);
    }
    const gl = this.gl;
    if (!gl) return;
    if (this._snTex) gl.deleteTexture(this._snTex);
    if (this._refTex) gl.deleteTexture(this._refTex);
    if (this._palTex) gl.deleteTexture(this._palTex);
    if (this._blaTex) gl.deleteTexture(this._blaTex);
    if (this._dummy) gl.deleteTexture(this._dummy);
    if (this._fbo) gl.deleteFramebuffer(this._fbo);
    const ext = gl.getExtension('WEBGL_lose_context');
    if (ext) ext.loseContext();
  }
}

function numberLines(src) {
  return src.split('\n').map((l, i) => (i + 1) + ': ' + l).join('\n');
}
