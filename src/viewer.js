// viewer.js — canvas + high-precision view state + gestures + progressive render.
//
// View state:
//   cx, cy : BigInt fixed-point center at precision `prec`
//   radius : double, half-height of the view in the complex plane
//   maxIter
// Gestures manipulate a preview transform of the last stable image (instant
// feedback); on gesture end the transform is folded into the view state and a
// real render is kicked off in the worker.

import {
  fromDecimalString, toDecimalString, fromDouble, toDouble, precForRadius,
} from './math/bignum.js';
import { engineForRadius, autoMaxIter, gpuEngineForRadius } from './math/render.js';
import { colorizeBlocks, colorizeRegion, paletteRgbAt } from './palette.js';
import { GpuRenderer } from './gpu/renderer.js';
import { GpuWorkerClient } from './gpu/gpu-worker-client.js';

const GUARD_BITS = 80;
// Backing-resolution safety budget (Spawn 32). The old MAX_BACKING=1100 long-edge cap
// (a mobile-era guard) silently rendered BELOW screen resolution on any larger window
// — with the crisp pixelated upscale that reads as BLOCKY fine detail (Danielle's
// report). The backing store now follows the TRUE canvas resolution (css × dpr) by
// default, guarded only by a total-pixel budget (~4K fullscreen; canvas RGBA ≈ 36MB)
// — plus the user Resolution setting (resScale 1|2|3 = full/half/third) for speed,
// and the low-power mode's stricter caps which still apply as before.
const MAX_BACKING_PIXELS = 9e6;
// Reference-reuse cache (Spawn 30). The BigInt reference orbit dominates a deep
// frame's CPU cost; Zhuoran rebasing means ANY nearby reference is a VALID
// reference, so the viewer caches the last GPU-path orbit (+ its BigInt tail) and
// reuses it while the view stays in its neighborhood:
//  - fresh builds carry REF_HEADROOM extra precision bits, so a zoom-IN sequence
//    keeps hitting until zoomBits grows past the headroom (~64 doublings);
//  - when only maxIter grew (every zoom-in tick), the orbit is EXTENDED from its
//    BigInt tail — costing only the NEW iterations (~ms) instead of a full rebuild;
//  - the view center may DRIFT from the cached reference point by up to
//    REF_MAX_DRIFT view-radii (wheel zooms, small pans, click-recenters): the
//    per-pixel dc is computed against the CACHED point exactly like the worker's
//    relocated-reference offsets. Beyond that the SA dc-box (and skip) degrades,
//    so we rebuild centered instead.
// A cache hit renders with a DIFFERENT (equally valid) reference realization than
// a cold render — per-pixel low-bit sn may differ on ill-conditioned boundary
// pixels, the same accepted class as reference relocation itself (bulk-gated by
// tools/probe-refcache.mjs; cold loads — bookmarks/URLs — are unaffected).
const REF_HEADROOM = 64;    // extra prec bits baked into fresh builds (~+15% build cost)
const REF_MAX_DRIFT = 8;    // max |view center − cached ref| in current view radii
// SA-coefficient reuse staleness bound: cached coeffs stay VALID on zoom-in (their dc
// box shrinks inside the validated one) but their frozen skip goes conservative as
// maxIter grows — refresh (in the worker) after this much maxIter growth. ~1000 iters
// ≈ 4 zoom octaves; bounds the extra post-SA GPU work to ~1000 anchor iterations.
const SA_STALE_ITERS = 1000;
// (the old MAX_BACKING=1100 long-edge display cap is gone — see MAX_BACKING_PIXELS)
// Supersampling caps: the fractal is computed at ss× the display res then box-
// averaged down. ss² multiplies pixel work and (GPU) the float sn-texture size,
// so bound both the longest edge and the total compute pixels (sn tex bytes =
// 16 * pixels) to stay within mobile GPU memory / sane CPU time.
const MAX_COMPUTE_DIM = 8192;
const MAX_COMPUTE_PIXELS = 12e6;
// Auto-drop supersampling below this radius. Supersampling computes ss² subsamples
// per output pixel through the FULL escape shader, and at extreme depth that shader is
// the heavy fe/rescaled perturbation loop at maxIter ~250/octave — so ss=2 makes a deep
// frame ~4× slower (a 2^500 frame goes from ~19s to ~75s). Past this depth each frame is
// already a slow deliberate dive and the fractal's density hides most of the AA benefit,
// so we cap the EFFECTIVE ss to 1 for snappy deep zoom (the user's ss SELECT is unchanged;
// the debug line shows the cap). 2^-300 sits well past the usual deep-exploration range
// (e.g. the 2^-270 "ultra" bookmark keeps full ss) and inside the heavy fe/rescaled band.
export const SS_DEEP_CAP_RADIUS = Math.pow(2, -300);
// Hard zoom floor — the precision wall. Every per-pixel quantity (dc, dz) is a normal
// double; dc ~ radius and the per-pixel step = 2*radius/computeHeight must both stay
// NORMAL (>= 2^-1022) or the offsets lose mantissa bits (adjacent pixels merge / the
// escape math degrades). The binding constraint is the step at the largest compute
// height (MAX_COMPUTE_DIM = 8192 ≈ 2^13): radius >= 8192 * 2^-1023 = 2^-1010. The deep
// perturbation engines are VALIDATED bit-clean against the BigInt-exact oracle at a
// genuine boundary coordinate down to this radius (tools/probe-wall.mjs — escape-count
// mism stays at the ill-conditioned-boundary floor, no representation failure; and
// tools/probe-deep500.mjs — GPU fe/rescaled vs the CPU oracle, 0% flat). Below it the
// double representation runs out of mantissa, so we CLAMP the radius here and signal
// "max depth" rather than silently render garbage. 2^1010 zoom is ~10^304× — far past
// any real use (the original target was 2^400). See NOTES "DEEP PRECISION WALL".
export const MIN_RADIUS = Math.pow(2, -1010);
// WebGL context-loss recovery. On loss we wait this long for the browser to restore
// the context (usually near-instant) before falling back to a CPU render, so a brief
// loss/restore doesn't cause a wasteful CPU detour. After this many losses we give up
// on the GPU for the session (a device that keeps dropping our context is better off
// staying on the CPU pool than flip-flopping).
const GPU_LOST_GRACE_MS = 600;
const GPU_MAX_LOSSES = 3;

// Low-power / battery-saver caps. When `lowPower` is on the viewer trades detail for
// fewer pixel-iterations per frame: a lower DPR + backing-resolution ceiling (fewer
// pixels), supersampling forced off (no ss² multiplier), and a lower AUTO iteration
// budget. A manual toggle is authoritative; main.js may also auto-enable it from
// navigator.getBattery (discharging + low level). Correctness is untouched — the math
// per sample is identical; there are just fewer, coarser samples. A user-typed (manual)
// maxIter is still honoured (only the AUTO budget is capped), so an explicit deep-detail
// request isn't silently overridden.
const LOW_POWER_DPR_CAP = 1;        // vs the normal min(devicePixelRatio, 2)
const LOW_POWER_MAX_BACKING = 700;  // long-edge cap while low-power (full res otherwise)
const LOW_POWER_MAX_ITER = 2000;    // ceiling on the auto iteration budget

export class Viewer {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.stable = document.createElement('canvas'); // last completed render
    this.sctx = this.stable.getContext('2d', { alpha: false });

    // view state
    this.prec = precForRadius(1.5, GUARD_BITS);
    this.cx = fromDecimalString('-0.5', this.prec);
    this.cy = fromDecimalString('0', this.prec);
    this.radius = 1.5;
    // Low-power / battery saver (see the LOW_POWER_* constants). Off by default; a manual
    // toggle or main.js's battery auto-detect flips it. Affects DPR/backing (resize), the
    // auto iteration budget (_autoIter), and supersampling (_effectiveSS). Not URL-persisted
    // (it's a per-device/session energy choice, like Force HQ / the glitch overlay).
    this.lowPower = !!opts.lowPower;
    this.maxIter = this._autoIter();
    this.autoIter = true;

    // coloring
    this.paletteOpts = { paletteId: 'ultra', cycle: 48, shift: 0, interior: [0, 0, 0] };

    // supersampling: render the fractal at ss× the display res, box-average the
    // colors down. Anti-aliases the boundary filaments (the "ultra" smooth look).
    // GPU averages in the color shader; CPU renders to an offscreen compute canvas
    // and downscale-blits. ss=1 = off (and the fast direct-to-display CPU path).
    this.ss = Math.max(1, Math.round(opts.ss || 2));
    // Force High Quality: when on, the user's ss is honoured even below the deep
    // SS_DEEP_CAP_RADIUS (the auto-drop to ss=1 that keeps extreme-depth frames
    // snappy). For deliberate slow high-AA deep captures. The hard memory/texture
    // caps (MAX_COMPUTE_DIM/PIXELS) STILL apply — this only overrides the depth cap.
    this.forceHighQuality = !!opts.forceHighQuality;
    this._effSS = 1;            // ss after the compute-size caps (set per render)
    this.cW = 0; this.cH = 0;   // compute resolution = _effSS × backing
    this._compute = document.createElement('canvas'); // offscreen compute target (CPU, ss>1)
    this._cctx = this._compute.getContext('2d', { alpha: false });
    this._paintCtx = this.ctx;  // where bands putImageData (display ctx, or compute ctx)
    this._presentRAF = 0;

    // GPU renderer (lazy; null if unsupported or disabled). The per-pixel raster
    // runs in WebGL shaders; the high-precision reference orbit is still computed
    // on the CPU worker and uploaded as a texture. CPU pool is the fallback/oracle.
    this.gpu = null;
    this._gpuChecked = false;
    this._gpuPaletteId = null;
    this._lastGpu = false;
    // OffscreenCanvas GPU path (Spawn 24): when supported, the GPU raster + color loop
    // runs in a dedicated worker (gpu-worker.js) that owns an OffscreenCanvas and pushes
    // finished ImageBitmap strips back — so the heavy GL flush/sync (the per-strip
    // ctx.drawImage(gpu.canvas) that janks the main thread today) happens OFF the main
    // thread. Default on when capable (feature-detected in _ensureGpu); opts.offscreen
    // === false forces the legacy on-main-thread GpuRenderer. `offscreen` is the runtime
    // flag (a fallback/give-up flips it off); `_offscreenPref` is the user's preference,
    // restored when the GPU toggle is re-enabled.
    this._offscreenPref = opts.offscreen !== false;
    this.offscreen = this._offscreenPref;
    this.gpuWorker = null;

    // Glitch debug overlay (off by default). When on, the GPU perturbation shaders run
    // the Pauldelbrot diagnostic (glitchTol > 0 — PURELY diagnostic: it sets the .b flag
    // but does NOT change escape/rebase or the rendered fractal), flagged pixels are tinted
    // magenta in the color pass, and the real flagged-pixel count is read back for the
    // status line — instead of the structural 0 the GPU reports otherwise (glitchTol was
    // hard-0). glitchTol is the standard Pauldelbrot ratio: flag when |z|^2 < tol·|Z_m|^2.
    this.showGlitches = false;
    this.glitchTol = 1e-6;
    // Series approximation (CPU perturbation path): skip the leading iterations for every
    // pixel via a polynomial-in-dc seed (~90%+ skip / ~2.5× faster below 2^-120). On by
    // default — bit-exact escape counts (tools/crosscheck-sa.mjs). The worker computes the
    // skip adaptively (skip 0 when not worthwhile) and disables it under the glitch overlay.
    // Not URL-persisted (a pure-speed internal, like the other debug toggles). The GPU deep
    // path doesn't use SA yet — see NOTES "SERIES APPROXIMATION".
    this.series = opts.series !== false;
    this._saSkip = 0;                  // last render's chosen skip (for the debug line)
    this.forceCpu = !!opts.forceCpu;   // tests / fallback can pin the CPU path
    this._mode = 'cpu';
    // WebGL context-loss recovery (mobile): while the GPU context is lost we render
    // on the CPU pool; once it's restored we re-render on the GPU. A run of repeated
    // losses (e.g. a device too memory-starved for our textures) makes us give up on
    // the GPU for the session to avoid a CPU<->GPU flip-flop. See _onGpuContextLost.
    this._gpuLostTimer = 0;
    this._gpuLossCount = 0;
    this._gpuGaveUp = false;

    // render plumbing (worker pool)
    this._pool = [];
    this.poolSize = Math.max(1, Math.min(opts.poolSize || (navigator.hardwareConcurrency || 4), 12));
    this.bandRows = 16;
    this.gen = 0;
    this.img = null;          // ImageData of current render
    this.dpr = 1;
    this.resScale = 1;      // Resolution setting: 1=full, 2=half, 3=third (setResScale)
    this.backingW = 0; this.backingH = 0;
    this.rendering = false;
    this._tilesLeft = 0;
    this._glitchAcc = 0;
    this._refMeta = null;
    this.onStatus = opts.onStatus || (() => {});
    this.onView = opts.onView || (() => {});

    // gesture transform (backing-pixel space): displayed = a*orig + (e,f)
    this.T = null;
    this._pointers = new Map();
    this._pinch = null;
    this._settleTimer = 0;   // debounce: real render fires once zoom motion stops

    this._installGestures();
  }

  // Auto iteration budget for the current radius, with the low-power ceiling applied.
  // (Manual maxIter goes through setMaxIter and bypasses this — only the auto budget is
  // capped in low-power, so an explicit deep-detail request is never silently overridden.)
  _autoIter() {
    const it = autoMaxIter(this.radius);
    return this.lowPower ? Math.min(it, LOW_POWER_MAX_ITER) : it;
  }

  // ---- sizing ----
  resize() {
    const rect = this.canvas.getBoundingClientRect();
    // Low-power caps fewer/coarser pixels: a lower DPR ceiling and a smaller backing cap.
    const dpr = Math.min(window.devicePixelRatio || 1, this.lowPower ? LOW_POWER_DPR_CAP : 2);
    // Render at TRUE canvas resolution by default (÷ the user Resolution setting);
    // the pixelated upscale of anything less reads as blocky fine detail. Safety
    // caps only: a total-pixel budget (memory; ~4K fullscreen) and low-power's
    // stricter long-edge cap. See MAX_BACKING_PIXELS.
    let w = Math.round(rect.width * dpr / this.resScale);
    let h = Math.round(rect.height * dpr / this.resScale);
    let scale = Math.min(1, Math.sqrt(MAX_BACKING_PIXELS / Math.max(1, w * h)));
    if (this.lowPower) scale = Math.min(scale, LOW_POWER_MAX_BACKING / Math.max(w, h));
    w = Math.max(1, Math.round(w * scale));
    h = Math.max(1, Math.round(h * scale));
    this.dpr = (dpr / this.resScale) * scale;
    this.backingW = w; this.backingH = h;
    this.canvas.width = w; this.canvas.height = h;
    this.stable.width = w; this.stable.height = h;
    this.cssW = rect.width; this.cssH = rect.height;
    this.render();
  }

  // Resolution setting: 1 = full canvas resolution (default), 2|3 = half/third
  // (per-device speed choice — deliberately NOT URL-persisted, like lowPower/HQ,
  // so a shared link never carries a degraded-resolution trap).
  setResScale(v) {
    const r = Math.max(1, Math.min(4, Math.round(v) || 1));
    if (r === this.resScale) return;
    this.resScale = r;
    this.resize();
  }

  // CSS pointer coords -> backing pixel coords
  _toBacking(clientX, clientY) {
    const rect = this.canvas.getBoundingClientRect();
    return {
      x: (clientX - rect.left) / rect.width * this.backingW,
      y: (clientY - rect.top) / rect.height * this.backingH,
    };
  }

  // complex offset (double) of a backing pixel from the view center
  _pixelDelta(px, py) {
    const aspect = this.backingW / this.backingH;
    const scale = (2 * this.radius) / this.backingH;
    return { dx: -this.radius * aspect + px * scale, dy: -this.radius + py * scale };
  }

  _setPrec(newPrec) {
    if (newPrec === this.prec) return;
    if (newPrec > this.prec) {
      const s = BigInt(newPrec - this.prec);
      this.cx <<= s; this.cy <<= s;
    } else {
      const s = BigInt(this.prec - newPrec);
      this.cx >>= s; this.cy >>= s;
    }
    this.prec = newPrec;
  }

  // ---- view operations ----
  zoomAt(px, py, factor) {
    // keep the complex point under (px,py) fixed while radius *= factor
    const { dx, dy } = this._pixelDelta(px, py);
    this.cx += fromDouble(dx * (1 - factor), this.prec);
    this.cy += fromDouble(dy * (1 - factor), this.prec);
    this.radius *= factor;
    this._afterRadiusChange();
  }

  panBacking(dxPix, dyPix) {
    const scale = (2 * this.radius) / this.backingH;
    this.cx -= fromDouble(dxPix * scale, this.prec);
    this.cy -= fromDouble(dyPix * scale, this.prec);
  }

  _afterRadiusChange() {
    this._clampRadius();
    const want = precForRadius(this.radius, GUARD_BITS);
    if (want > this.prec) this._setPrec(want);
    if (this.autoIter) this.maxIter = this._autoIter();
  }

  // Enforce the precision wall (MIN_RADIUS). Below it the double-precision dc/step run
  // out of mantissa, so we never let the committed radius go deeper — we clamp and set
  // `atMinRadius` so the UI can say "max depth" instead of rendering garbage. Returns
  // true if a clamp happened (the caller asked to go deeper than the wall).
  _clampRadius() {
    if (this.radius < MIN_RADIUS) { this.radius = MIN_RADIUS; this.atMinRadius = true; return true; }
    this.atMinRadius = this.radius <= MIN_RADIUS;
    return false;
  }

  zoomLevel() { return Math.log2(1.5 / this.radius); }

  getState() {
    return {
      cx: toDecimalString(this.cx, this.prec, Math.ceil(this.prec / 3.32) + 5),
      cy: toDecimalString(this.cy, this.prec, Math.ceil(this.prec / 3.32) + 5),
      radius: this.radius,
      maxIter: this.maxIter,
      zoom: this.zoomLevel(),
      atMinRadius: !!this.atMinRadius,   // at the precision wall (clamped) — UI shows "max depth"
    };
  }

  setState(s) {
    if (s.radius) this.radius = +s.radius;
    this._clampRadius();                 // a deep bookmark/URL/Go below the wall is clamped here too
    this.prec = precForRadius(this.radius, GUARD_BITS);
    if (s.cx != null) this.cx = fromDecimalString(String(s.cx), this.prec);
    if (s.cy != null) this.cy = fromDecimalString(String(s.cy), this.prec);
    if (s.maxIter) { this.maxIter = +s.maxIter; this.autoIter = false; }
    else if (this.autoIter) this.maxIter = this._autoIter();
    this.render();
  }

  // Toggle the GPU path. Turning it on re-probes WebGL support; off pins the CPU
  // worker engines (the validated oracle/fallback). Either way, re-render.
  setUseGpu(on) {
    this.forceCpu = !on;
    this._gpuChecked = false;
    // Explicit re-enable clears any prior give-up/loss state (fresh probe + counters).
    this._gpuGaveUp = false; this._gpuLossCount = 0;
    clearTimeout(this._gpuLostTimer); this._gpuLostTimer = 0;
    this._disposeGpuBackend();
    this.offscreen = on ? this._offscreenPref : false;
    this.render();
  }

  // Tear down whichever GPU backend is live (main-thread renderer or offscreen worker).
  _disposeGpuBackend() {
    if (this.gpuWorker) { try { this.gpuWorker.dispose(); } catch { /* noop */ } this.gpuWorker = null; }
    if (this.gpu) { try { this.gpu.dispose(); } catch { /* noop */ } this.gpu = null; }
  }

  // True when the offscreen GPU worker is the active raster backend (constructed, not
  // reported-unsupported, and not context-lost). render() routes here vs the main-thread
  // GpuRenderer; the CPU pool is the fallback for both.
  _useOffscreen() {
    return this.offscreen && !!this.gpuWorker && this.gpuWorker.supported !== false && !this.gpuWorker.lost;
  }
  // GPU backend lost? (offscreen worker or main-thread renderer) — gates the CPU detour.
  _gpuLost() { return this.gpuWorker ? this.gpuWorker.lost : !!(this.gpu && this.gpu.lost); }

  // For the debug readout: which GPU is active (null if CPU path / context lost).
  gpuInfo() {
    if (this._useOffscreen()) return this.gpuWorker.info || 'WebGL2 (worker)';
    if (!this.gpu || !this.gpu.gl || this.gpu.lost) return null;
    const gl = this.gpu.gl;
    const dbg = gl.getExtension('WEBGL_debug_renderer_info');
    return dbg ? String(gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL)) : 'WebGL2';
  }

  // Set the supersample factor (1 = off). Re-renders at the new compute res.
  setSupersample(n) {
    const s = Math.max(1, Math.min(4, Math.round(+n || 1)));
    if (s === this.ss) return;
    this.ss = s;
    this.render();
  }

  // Force High Quality: honour the user's ss even below SS_DEEP_CAP_RADIUS (override
  // the extreme-depth auto-drop to ss=1). Re-renders so the change takes effect — at a
  // deep view this is a deliberate slow high-AA capture; at shallow zoom it's a no-op
  // visually (the cap never engaged there) but we re-render for parity with setSupersample.
  setForceHighQuality(on) {
    on = !!on;
    if (on === this.forceHighQuality) return;
    this.forceHighQuality = on;
    this.render();
  }

  // Toggle the glitch debug overlay. Turning ON re-renders the escape pass with the
  // Pauldelbrot diagnostic enabled (the GPU `.b` flags / CPU per-pixel mask must be
  // (re)computed); turning OFF only needs to drop the tint, so a cheap recolor suffices
  // when the current frame is already painted (GPU: reuse the sn FBO; CPU: re-tint from
  // the cached sn + glitch mask). Both paths now carry a per-pixel overlay.
  setShowGlitches(on) {
    on = !!on;
    if (on === this.showGlitches) return;
    this.showGlitches = on;
    if (!on && !this.rendering && this._lastGpu && this._useOffscreen()) this._recolorGpuOffscreen();
    else if (!on && !this.rendering && this._lastGpu && this.gpu && !this.gpu.lost) this._recolorGpu();
    else if (!on && !this.rendering && !this._lastGpu && this.img && this._sn) this._recolor();
    else this.render();
  }

  // Toggle series approximation (CPU perturbation path). Pure speed; bit-exact output, so a
  // re-render is the only effect. A debug/safety switch (default on).
  setSeries(on) {
    on = !!on;
    if (on === this.series) return;
    this.series = on;
    this.render();
  }

  setMaxIter(v) { this.maxIter = v; this.autoIter = false; this.render(); }
  setAutoIter(on) { this.autoIter = on; if (on) { this.maxIter = this._autoIter(); this.render(); } }

  // Toggle low-power / battery-saver mode. Recomputes the auto iteration budget (if on
  // auto) and re-sizes — resize() re-derives the backing resolution under the new DPR/
  // backing caps and re-renders (also picking up the ss=1 cap via _effectiveSS). When the
  // canvas isn't laid out yet (no backing) just render. See the LOW_POWER_* constants.
  setLowPower(on) {
    on = !!on;
    if (on === this.lowPower) return;
    this.lowPower = on;
    if (this.autoIter) this.maxIter = this._autoIter();
    if (this.backingW) this.resize(); else this.render();
  }
  setPalette(opts) {
    Object.assign(this.paletteOpts, opts);
    // recolor instantly without recomputing the fractal
    if (this._lastGpu && this._useOffscreen() && !this.rendering) this._recolorGpuOffscreen();
    else if (this._lastGpu && this.gpu && !this.rendering) this._recolorGpu();
    else if (this.img && this._sn && !this._lastGpu) this._recolor();
    else this.render();
  }

  // ---- rendering (worker pool) ----
  _terminatePool() {
    for (const w of this._pool) { try { w.terminate(); } catch { /* noop */ } }
    this._pool = [];
  }

  _spawnPool(n) {
    this._terminatePool();
    const count = n || this.poolSize;
    for (let i = 0; i < count; i++) {
      const w = new Worker(new URL('./worker.js', import.meta.url), { type: 'module' });
      const gen = this.gen;
      w.onmessage = (e) => this._onWorker(e.data, gen);
      w.onerror = (err) => this.onStatus({ error: String(err.message || err) });
      this._pool.push(w);
    }
  }

  // Effective supersample factor after the memory/size caps, given the display res.
  // Extreme deep zooms also drop to ss=1 (see SS_DEEP_CAP_RADIUS) so a 2^500 frame isn't
  // paying ss² on the heavy fe/rescaled shader; this is a pure perf/quality knob (the
  // math per sample is unchanged) and the debug line surfaces the cap.
  _effectiveSS() {
    let s = Math.max(1, Math.round(this.ss || 1));
    // Low-power forces ss=1 (no ss² pixel multiplier) — and wins over Force HQ, since
    // it's the explicit energy choice.
    if (this.lowPower) return 1;
    // Extreme depth: snappy over AA — unless Force High Quality overrides the cap.
    if (s > 1 && this.radius < SS_DEEP_CAP_RADIUS && !this.forceHighQuality) return 1;
    const maxDim = Math.max(this.backingW, this.backingH, 1);
    while (s > 1 && (maxDim * s > MAX_COMPUTE_DIM ||
                     this.backingW * s * this.backingH * s > MAX_COMPUTE_PIXELS)) {
      s--;
    }
    return s;
  }

  // Recompute compute resolution + the CPU paint target for the current ss.
  // ss==1: paint straight to the display canvas (fast path, no extra blit).
  // ss>1 : paint to an offscreen compute canvas, downscale-blit on present().
  _updateComputeSize() {
    this._effSS = this._effectiveSS();
    this.cW = this.backingW * this._effSS;
    this.cH = this.backingH * this._effSS;
    if (this._effSS > 1) {
      if (this._compute.width !== this.cW || this._compute.height !== this.cH) {
        this._compute.width = this.cW; this._compute.height = this.cH;
      }
      this._paintCtx = this._cctx;   // CPU bands paint into the offscreen compute canvas
    } else {
      this._paintCtx = this.ctx;     // ss==1: paint straight to the display canvas
    }
  }

  // Downscale the compute canvas onto the display canvas (the supersample average).
  // Smoothing ON here (area/box filter = the AA); the display->screen upscale stays
  // point-filtered (CSS image-rendering: pixelated) for crispness. No-op at ss==1.
  _presentNow() {
    if (this._effSS <= 1) return;
    this.ctx.imageSmoothingEnabled = true;
    this.ctx.imageSmoothingQuality = 'high';
    this.ctx.drawImage(this._compute, 0, 0, this.cW, this.cH, 0, 0, this.backingW, this.backingH);
  }
  _schedulePresent() {
    if (this._effSS <= 1 || this._presentRAF) return;
    const raf = (typeof requestAnimationFrame === 'function')
      ? requestAnimationFrame : (cb) => setTimeout(cb, 16);
    this._presentRAF = raf(() => { this._presentRAF = 0; this._presentNow(); });
  }
  _clearPresent() {
    if (this._presentRAF && typeof cancelAnimationFrame === 'function') cancelAnimationFrame(this._presentRAF);
    this._presentRAF = 0;
  }

  render() {
    if (!this.backingW) return;
    this._clearSettle();
    this._clearPresent();
    this._updateComputeSize();
    this.gen++;
    this.T = null; // clear preview transform; we render fresh
    this.rendering = true;
    this._glitchAcc = 0;
    this._refMeta = null;
    this._saSkip = 0;   // set by refReady on the perturb paths; 0 for naive (no SA)
    this.onView(this.getState());

    // Pick the GPU engine if available + in range; else fall back to CPU workers.
    // `this.gpu.lost` gates out a lost WebGL context (mobile) — render on CPU until
    // it's restored (see _onGpuContextLost / _onGpuContextRestored).
    const gpuReady = this._ensureGpu() && this._gpuUsable();
    const gpuEngine = gpuReady ? gpuEngineForRadius(this.radius) : null;
    if (gpuEngine === 'naive') {
      this._mode = 'gpu-naive';
      if (this._useOffscreen()) this._renderGpuNaiveWorker(); else this._renderGpuNaive();
      return;
    }
    if (gpuEngine === 'perturb' || gpuEngine === 'perturb-fe') {
      this._mode = gpuEngine === 'perturb-fe' ? 'gpu-perturb-fe' : 'gpu-perturb';
      this._startGpuPerturb(); return;   // CPU worker computes the reference; refReady routes the raster
    }

    // ---- CPU path (worker pool) ----
    this._mode = 'cpu';
    this._lastGpu = false;
    this.img = this.ctx.createImageData(this.cW, this.cH);     // at compute res
    this._sn = new Float64Array(this.cW * this.cH);            // compute-res sn cache
    this._sn.fill(-2); // -2 = not yet computed
    // Glitch-overlay mask cache (compute res), built only when the overlay is on so a
    // palette recolor / overlay-off can re-tint or drop the tint without recomputing.
    // Null on default renders -> colorize gets no mask -> output byte-identical.
    this._glitch = this.showGlitches ? new Uint8Array(this.cW * this.cH) : null;
    // Bands aligned to the supersample factor so each band maps to whole display
    // rows on present (no partial-block downscale seams).
    this.bandRows = Math.max(this._effSS, Math.round(16 / this._effSS) * this._effSS);
    const engine = engineForRadius(this.radius);
    this._spawnPool();
    // worker[0] computes the reference (or naive params) + a coarse pass
    this._pool[0].postMessage({
      type: 'computeRef', gen: this.gen,
      cxRaw: this.cx.toString(), cyRaw: this.cy.toString(), prec: this.prec,
      radius: this.radius, width: this.cW, height: this.cH,
      maxIter: this.maxIter, engine, wantCoarse: true,
      // Glitch overlay (debug): ask the workers for a per-pixel mask + tune the diagnostic
      // tolerance. Purely diagnostic — the mask never affects sn, so the render is unchanged.
      showGlitch: this.showGlitches, glitchTol: this.glitchTol,
      // Series approximation (perturbation only; the worker skips it under the overlay).
      series: this.series,
    });
    this.onStatus({ phase: 'start', engine, maxIter: this.maxIter, zoom: this.zoomLevel() });
  }

  // ---- GPU rendering ----
  // Is a GPU backend present + usable for this render? (offscreen worker or main renderer)
  _gpuUsable() {
    if (this._useOffscreen()) return true;
    return !!this.gpu && !this.gpu.lost;
  }

  // Synchronous main-thread capability probe for the offscreen path: OffscreenCanvas +
  // a worker + a WebGL2 context with the float-render extension. If this passes on the
  // main thread it will pass in the worker (same browser GL), so render() can route to
  // the worker optimistically before its async init handshake returns.
  _offscreenCapable() {
    if (typeof Worker === 'undefined' || typeof OffscreenCanvas === 'undefined') return false;
    if (typeof OffscreenCanvas.prototype.transferToImageBitmap !== 'function') return false;
    try {
      const c = new OffscreenCanvas(1, 1);
      const gl = c.getContext('webgl2', { antialias: false, depth: false, stencil: false });
      return !!gl && !!gl.getExtension('EXT_color_buffer_float');
    } catch { return false; }
  }

  _ensureGpu() {
    if (this._gpuChecked) {
      if (this._gpuGaveUp) return false;
      return this._useOffscreen() ? true : !!this.gpu;
    }
    this._gpuChecked = true;
    if (this.forceCpu || this._gpuGaveUp) { this.gpu = null; this.offscreen = false; return false; }
    // Preferred: the offscreen GPU worker (raster off the main thread). Falls through to
    // the on-main-thread GpuRenderer when OffscreenCanvas/WebGL2 isn't available.
    if (this.offscreen && this._offscreenCapable()) {
      try {
        this.gpuWorker = new GpuWorkerClient({
          onContextLost: () => this._onGpuContextLost(),
          onContextRestored: () => this._onGpuContextRestored(),
          onFail: () => this._onOffscreenFail(),
        });
        return true;
      } catch (e) { this.offscreen = false; this.gpuWorker = null; }
    } else {
      this.offscreen = false;
    }
    try {
      const g = new GpuRenderer({
        width: this.backingW || 1, height: this.backingH || 1,
        onContextLost: () => this._onGpuContextLost(),
        onContextRestored: () => this._onGpuContextRestored(),
      });
      this.gpu = g.supported ? g : null;
      this._gpuPaletteId = null;
    } catch (e) { this.gpu = null; }
    return !!this.gpu;
  }

  // The offscreen worker reported no usable GPU (rare — the sync probe passed). Drop the
  // worker, re-probe (→ main-thread GpuRenderer, else CPU), and re-render this view.
  _onOffscreenFail() {
    this.offscreen = false;
    if (this.gpuWorker) { try { this.gpuWorker.dispose(); } catch { /* noop */ } this.gpuWorker = null; }
    this._gpuChecked = false;
    this.render();
  }

  _setGpuLUT() {
    const pid = this.paletteOpts.paletteId;
    if (this._gpuPaletteId === pid) return;
    this.gpu.setPaletteLUT((u) => paletteRgbAt(pid, u), 1024);
    this._gpuPaletteId = pid;
  }

  _gpuColorOpts() {
    return { cycle: this.paletteOpts.cycle, shift: this.paletteOpts.shift,
             interior: this.paletteOpts.interior || [0, 0, 0], ss: this._effSS,
             showGlitch: this.showGlitches };
  }

  // The GPU color pass already supersampled (averaged) down to the display res,
  // so its canvas is 1:1 with ours — point-copy it (no smoothing). Used by
  // _recolorGpu (a complete frame); the strip loop uses _blitGpuStrip instead.
  _blitGpu() { this.ctx.imageSmoothingEnabled = false; this.ctx.drawImage(this.gpu.canvas, 0, 0); }

  // Blit ONLY the display rows covered by a just-computed compute-row strip [y, y+h),
  // leaving every other row untouched (still showing the scaled preview / prior frame).
  // This is what makes a deep render REPLACE the low-res preview with high-res tiles as
  // they arrive (top-to-bottom) instead of clearing the whole canvas to interior-black
  // and revealing over a void. The color pass flips Y and box-averages ss×ss, so compute
  // rows [y, y+h) map to display rows [y/ss, (y+h)/ss) at the SAME offset in source and
  // dest. Strips are ss-aligned (see _stripRows), so the division is exact.
  _blitGpuStrip(y, h) {
    const ss = this._effSS || 1;
    const y0 = Math.round(y / ss), h0 = Math.round(h / ss);
    if (h0 <= 0) return;
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.drawImage(this.gpu.canvas, 0, y0, this.backingW, h0, 0, y0, this.backingW, h0);
  }

  _finishGpu(engine, glitches) {
    this.rendering = false;
    this._lastGpu = true;
    this.sctx.drawImage(this.canvas, 0, 0); // snapshot for gesture previews
    const meta = this._refMeta || {};
    // GPU renders normally report 0 glitches (the .b diagnostic is off — glitchTol 0). When
    // the debug overlay is on, read back the real flagged-pixel count for the perturb engines
    // (the naive shallow engine has no perturbation reference, hence no glitch concept).
    let g = glitches;
    if (this.showGlitches && this.gpu && engine.startsWith('gpu-perturb')) {
      try { g = this.gpu.countGlitches(); } catch { /* keep the passed value on readback failure */ }
    }
    this.onStatus({ phase: 'done', engine, glitches: g, refLen: meta.refLen || 0,
                    relocations: meta.relocations || 0, zoom: this.zoomLevel(),
                    saSkip: this._saSkip });
  }

  _gpuFail(e) {
    // A GPU render threw. Drop the active backend and re-render. If the offscreen worker
    // failed, also disable it + re-probe so we fall to the main-thread GpuRenderer (still
    // GPU) and only then to the CPU pool — rather than re-routing to the failed worker.
    // eslint-disable-next-line no-console
    console.warn('GPU render failed; falling back:', e && (e.message || e));
    if (this.gpuWorker) {
      try { this.gpuWorker.dispose(); } catch { /* noop */ }
      this.gpuWorker = null; this.offscreen = false; this._gpuChecked = false;
    }
    this.gpu = null; this._lastGpu = false;
    this.render();
  }

  // ---- WebGL context-loss recovery ----
  // The GPU context can be lost on mobile (memory pressure, tab backgrounded, driver
  // reset). The GpuRenderer flags itself `lost` (so render() routes to the CPU pool)
  // and calls these. Lost: cancel the in-flight GPU render, then after a short grace
  // period render on the CPU if the context still hasn't returned — the grace avoids a
  // wasteful CPU detour when the browser restores the context almost immediately (the
  // common case) yet bounds how long the view sits frozen if it doesn't.
  _onGpuContextLost() {
    this._gpuLossCount++;
    this.gen++;                  // invalidate the in-flight render's async continuations
    this.rendering = false;
    this._terminatePool();
    if (this.gpuWorker) this.gpuWorker.cancel(this.gen);   // supersede any in-flight worker strips
    this._lastGpu = false;
    this.onStatus({ phase: 'start', engine: 'recovering GPU…', maxIter: this.maxIter, zoom: this.zoomLevel() });
    clearTimeout(this._gpuLostTimer);
    this._gpuLostTimer = setTimeout(() => {
      this._gpuLostTimer = 0;
      if (this._gpuLost()) this.render();   // still lost -> CPU (the lost flag gates engine pick)
    }, GPU_LOST_GRACE_MS);
  }

  // Restored: the renderer recreated its GL objects. Force a palette-LUT re-upload
  // (its texture was lost) and re-render on the GPU. If the context has dropped too
  // many times this session, give up on the GPU and stay on the CPU pool rather than
  // flip-flop a memory-starved device between the two engines.
  _onGpuContextRestored() {
    clearTimeout(this._gpuLostTimer); this._gpuLostTimer = 0;
    this._gpuPaletteId = null;   // palette texture was lost; _setGpuLUT must re-upload it
    if (this._gpuLossCount >= GPU_MAX_LOSSES) {
      this._gpuGaveUp = true;
      this._disposeGpuBackend();
      this.offscreen = false;
      this.render();             // CPU from here on
      return;
    }
    this.render();               // the lost flag is false now -> GPU path
  }

  _raf() { return new Promise((r) => requestAnimationFrame(r)); }

  // Strip height (compute rows per GPU escape draw), aligned to the supersample
  // factor so each strip maps to whole display rows. Sized so one strip's worst-case
  // work (rows × width × iterations pixel-iterations) stays well under a GPU watchdog
  // (~2s) even on a modest mobile GPU, while keeping the strip count (per-strip
  // overhead) reasonable. Cheap/shallow views collapse to a single full-frame strip.
  //
  // skipIters (Spawn 28): the SA seed jumps EVERY pixel to iteration `skip`, so the
  // true per-pixel worst case is maxIter − skip, not maxIter. Ignoring the skip cut
  // deep frames into ~10–20× more strips than the budget intends (2^-400: 100 strips
  // of 6 rows), and the per-strip fixed costs (full-canvas colorize + ImageBitmap +
  // post/ack — or flush + blit + a vsync rAF on the main-thread path) were ~35% of
  // the frame (tools/probe-strips.mjs: 1008→640ms @2^-400). Budgeting on the
  // POST-SKIP iterations restores the intended strip count at an UNCHANGED worst-case
  // per-strip duration — the same 4e8 envelope shipping since Spawn 7 (pre-SA, deep
  // frames genuinely ran maxIter/pixel and mobile was the design target). Partition
  // does not affect pixels (bit-identical, crosscheck-tiled).
  _stripRows(skipIters = 0) {
    const W = Math.max(1, this.cW), ss = this._effSS;
    const budget = 4e8;                         // pixel-iterations per strip (watchdog headroom)
    const effIter = Math.max(1, this.maxIter - Math.max(0, skipIters | 0));
    let rows = Math.floor(budget / (W * effIter));
    rows = Math.max(ss, Math.round(rows / ss) * ss);   // >= ss, aligned to ss
    return Math.min(rows, this.cH);
  }

  // Run a GPU escape pass in horizontal strips, yielding a frame between draws. This
  // is the fix for "can't zoom past ~2^218": a deep frame needs maxIter ~55k, and a
  // SINGLE full-screen draw at that count runs long enough to trip the GPU watchdog
  // (TDR) on real hardware → context loss → CPU fallback (minutes). Splitting it into
  // short per-strip draws keeps every draw under the watchdog, lets the page stay
  // responsive, and reveals the image top-to-bottom. `drawStrip(y, h)` issues one
  // strip's escape draw (rows [y, y+h)). The scissor approach keeps gl_FragCoord
  // identical to one big draw, so the result is BIT-IDENTICAL. Returns true if it ran
  // to completion, false if a newer render/gesture superseded it (gen guard).
  //
  // We blit each strip over ONLY its own rows (_blitGpuStrip), so the not-yet-computed
  // rows keep showing the scaled preview / prior frame: the high-res tiles REPLACE the
  // low-res preview as they arrive instead of the canvas flashing to a black void first.
  // (clearSn still defines the FBO's unwritten rows so colorize is well-formed; those
  // rows are simply never blitted while uncomputed.)
  async _drawTiledEscape(drawStrip, gen, skipIters = 0) {
    const W = this.cW, H = this.cH;
    this.gpu.clearSn(W, H, -1);                 // define unwritten rows (not blitted while uncomputed)
    this._setGpuLUT();
    const stripH = this._stripRows(skipIters);
    for (let y = 0; y < H; y += stripH) {
      if (gen !== this.gen) return false;       // superseded -> stop, don't touch the canvas
      const h = Math.min(stripH, H - y);
      drawStrip(y, h);
      this.gpu.gl.flush();                       // submit this strip as its own GPU command
      this.gpu.colorize(this._gpuColorOpts());
      this._blitGpuStrip(y, h);                  // reveal THIS strip over the preview (no black void)
      await this._raf();                         // yield: let the GPU drain + the page paint/respond
    }
    return gen === this.gen;
  }

  async _renderGpuNaive() {
    const W = this.cW, H = this.cH, gen = this.gen;   // compute res (ss × display)
    const aspect = W / H, scale = (2 * this.radius) / H;
    const cx = toDouble(this.cx, this.prec), cy = toDouble(this.cy, this.prec);
    const ox = cx - this.radius * aspect, oy = cy - this.radius;
    this.onStatus({ phase: 'start', engine: 'gpu-naive', maxIter: this.maxIter, zoom: this.zoomLevel() });
    try {
      const done = await this._drawTiledEscape(
        (y, h) => this.gpu.renderNaive({ ox, oy, scale, maxIter: this.maxIter,
                                         width: W, height: H, df64: false, stripY: y, stripH: h }), gen);
      if (!done) return;
      this._refMeta = { engine: 'gpu-naive', refLen: 0, relocations: 0 };
      this._finishGpu('gpu-naive', 0);
    } catch (e) { if (gen === this.gen) this._gpuFail(e); }
  }

  _startGpuPerturb() {
    // ---- reference-reuse cache (Spawn 30; see the REF_HEADROOM block up top) ----
    const use = this._refCacheUsable();
    if (use) {
      const c = this._refCache;
      const wantSeries = this.series && this._mode === 'gpu-perturb-fe' && !this.showGlitches;
      const saOk = !wantSeries || this._cachedSaFits();
      if (use === 'ready' && saOk) { this._gpuPerturbFromCache(); return; }
      this._spawnPool(1);
      if (use === 'extend') {
        // Only maxIter grew (the normal zoom-in tick): extend the cached orbit from
        // its BigInt tail — the worker computes ONLY the new iterations (~ms). When
        // the cached SA has gone stale (or is missing), bundle its refresh into the
        // same worker trip: clone the old arrays in, get merged arrays + fresh sa back.
        let withSA = null;
        if (wantSeries && !saOk) {
          const b = this._cacheDcBox();
          withSA = { radius: this.radius, dcHalfX: b.dcHalfX, dcHalfY: b.dcHalfY, maxIter: this.maxIter };
        }
        this._pool[0].postMessage({
          type: 'extendRef', gen: this.gen,
          cxRaw: c.cx.toString(), cyRaw: c.cy.toString(), prec: c.prec,
          tailBx: c.tailBx, tailBy: c.tailBy, tailPm2: c.z2[c.len],
          fromN: c.len, maxIter: this.maxIter,
          withSA, ...(withSA ? { zxOld: c.zx, zyOld: c.zy, z2Old: c.z2 } : {}),
        });
      } else {
        // 'ready' but the SA coeffs need (re)computation — run it in the worker
        // (computeSeries is hundreds of ms at extreme depth; never on main).
        const b = this._cacheDcBox();
        this._pool[0].postMessage({
          type: 'computeSA', gen: this.gen,
          zx: c.zx, zy: c.zy, len: c.len,
          radius: this.radius, dcHalfX: b.dcHalfX, dcHalfY: b.dcHalfY, maxIter: this.maxIter,
        });
      }
      this.onStatus({ phase: 'start', engine: 'gpu-perturb', maxIter: this.maxIter, zoom: this.zoomLevel() });
      return;
    }
    this._spawnPool(1); // one worker computes the reference orbit; GPU rasterizes
    // Series approximation for the GPU deep path: only the fe/rescaled engine seeds dz
    // in-shader (the df64 shallow band renders fast and doesn't apply SA), and never under
    // the glitch overlay (which must see every iteration — mirrors the CPU SA gating). The
    // worker computes the coeffs right after the reference (grid + escape-guard; no coarse
    // pass — its min-escape cap is non-binding here, see tools/probe-sa-cap.mjs) — ~tens of
    // ms even at 2^500, then they ride in params.sa to _renderGpuPerturb.
    const wantSeries = this.series && this._mode === 'gpu-perturb-fe' && !this.showGlitches;
    // Send the COMPUTE resolution: the worker derives scale=(2r)/height for the
    // supersampled grid (offX/offY are resolution-independent).
    // Fresh builds run at prec + REF_HEADROOM (the center BigInts shifted to match)
    // so subsequent zoom-ins can REUSE this orbit instead of rebuilding. The extra
    // bits cost ~10-15% on the build once; every reuse saves the whole build.
    const hb = BigInt(REF_HEADROOM);
    this._pool[0].postMessage({
      type: 'computeRef', gen: this.gen,
      cxRaw: (this.cx << hb).toString(), cyRaw: (this.cy << hb).toString(),
      prec: this.prec + REF_HEADROOM,
      radius: this.radius, width: this.cW, height: this.cH,
      maxIter: this.maxIter, engine: 'perturb', wantCoarse: false,
      series: wantSeries,
    });
    this.onStatus({ phase: 'start', engine: 'gpu-perturb', maxIter: this.maxIter, zoom: this.zoomLevel() });
  }

  // Can the cached reference serve the CURRENT view? 'ready' = render right now;
  // 'extend' = same orbit, needs more iterations (extendRef); null = rebuild.
  // Side effect on non-null: stashes the exact HP offset view-center − ref-point
  // (as doubles) in this._refCacheOff for _gpuPerturbFromCache's dc origin.
  _refCacheUsable() {
    const c = this._refCache;
    if (!c) return null;
    if (c.prec < this.prec) return null;                 // zoomed past the prec headroom
    const s = BigInt(c.prec - this.prec);
    // refOff = CACHED REF POINT − VIEW CENTER — the SAME convention as worker.js
    // computeRef's refOffX = center.x − cx, consumed as offX = −r·aspect − refOffX.
    // (A drift-sign flip here mirrors the rendered location about the reference and
    // silently invalidates the SA box check — the Spawn-31 "blue screen / teleport"
    // bug. probe-refcache's DRIFTED sequence now guards this: centered zooms have
    // drift exactly 0 and cannot catch it.)
    const offX = toDouble(c.cx - (this.cx << s), c.prec);
    const offY = toDouble(c.cy - (this.cy << s), c.prec);
    if (!(Math.hypot(offX, offY) <= REF_MAX_DRIFT * this.radius)) return null;
    this._refCacheOff = { offX, offY };
    if (this.maxIter <= c.len) return 'ready';
    if (!c.escaped) return 'extend';
    return null;   // escaped ref, too short: a fresh build (with relocation) is better
  }

  // The CURRENT view's dc geometry relative to the CACHED reference point
  // (requires this._refCacheOff from a preceding _refCacheUsable call).
  _cacheDcBox() {
    const { offX: refOffX, offY: refOffY } = this._refCacheOff;
    const aspect = this.cW / this.cH, scale = (2 * this.radius) / this.cH;
    const offX = -this.radius * aspect - refOffX;
    const offY = -this.radius - refOffY;
    const dcFarX = offX + (this.cW - 1) * scale, dcFarY = offY + (this.cH - 1) * scale;
    return { offX, offY, scale,
      dcHalfX: Math.max(Math.abs(offX), Math.abs(dcFarX)),
      dcHalfY: Math.max(Math.abs(offY), Math.abs(dcFarY)) };
  }

  // Can the CACHED SA coefficients serve the current view? They are properties of
  // the reference orbit + their validation dc box, so they remain VALID for any view
  // whose dc box fits INSIDE that box (every zoom-in tick; sa.invR is the seed's own
  // normalization, independent of the current radius). The frozen skip only goes
  // conservative as maxIter grows (fresh coeffs would skip a bit further), costing
  // post-SA GPU iterations — so refresh (in the worker) once maxIter has outgrown
  // the coeffs by SA_STALE_ITERS. Refresh cost amortizes to ~nothing per tick;
  // staleness loss stays bounded to ~SA_STALE_ITERS anchor iterations.
  _cachedSaFits() {
    const c = this._refCache;
    if (!c || !c.sa || !(c.sa.skip > 0) || !c.saBox) return false;
    if (c.sa.skip >= Math.min(c.len, this.maxIter)) return false;
    if (this.maxIter - c.saAtIter > SA_STALE_ITERS) return false;      // stale skip
    const b = this._cacheDcBox();
    return b.dcHalfX <= c.saBox.dcHalfX && b.dcHalfY <= c.saBox.dcHalfY;
  }

  // Render the GPU perturbation frame straight from the cached reference — no pool
  // worker, no BigInt build, no SA scan (the cached coeffs are reused; callers ensure
  // they fit or have refreshed them). Mirrors worker.js computeRef's params synthesis
  // for a (possibly drifted) reference point: offX/offY are the dc of pixel (0,0)
  // relative to the CACHED point.
  _gpuPerturbFromCache() {
    const c = this._refCache;
    const b = this._cacheDcBox();
    const wantSeries = this.series && this._mode === 'gpu-perturb-fe' && !this.showGlitches;
    const sa = (wantSeries && c.sa && c.sa.skip > 0 && c.sa.skip < Math.min(c.len, this.maxIter))
      ? c.sa : null;
    // slices: _renderGpuPerturbWorker TRANSFERS the arrays to the GPU worker — the
    // cache must keep its own copies for the next hit/extension.
    const params = { engine: 'perturb', zx: c.zx.slice(), zy: c.zy.slice(),
      len: c.len, offX: b.offX, offY: b.offY, scale: b.scale,
      maxIter: this.maxIter, width: this.cW, sa };
    this._refMeta = { engine: this._mode, refLen: c.len, relocations: 0, refCached: true };
    this._saSkip = (sa && sa.skip) || 0;
    this.onStatus({ phase: 'start', engine: 'gpu-perturb', maxIter: this.maxIter, zoom: this.zoomLevel() });
    if (this._useOffscreen()) this._renderGpuPerturbWorker(params);
    else this._renderGpuPerturb(params);
  }

  async _renderGpuPerturb(p) {
    const W = this.cW, H = this.cH, gen = this.gen;   // compute res (ss × display)
    const fe = this._mode === 'gpu-perturb-fe';       // floatexp-precision band, below the df64 floor
    try {
      this.gpu.uploadReferenceDf64(p.zx, p.zy);       // reference is df64 in both paths
      // Glitch overlay (debug): enable the Pauldelbrot diagnostic so the .b flag is set
      // (purely diagnostic — escape/rebase and the rendered image are unchanged), and turn
      // OFF the fast-skip so the test runs every iteration. (The skip is only valid where no
      // glitch could fire, so it's safe at the 1e-6 tol, but we compute exhaustively when the
      // user has explicitly asked to see glitches.) Default render: glitchTol 0, skip on.
      const g = this.showGlitches;
      // Series approximation (deep fe/rescaled engine only): p.sa carries the CPU-computed
      // skip + seed coeffs; the rescaled shader jumps dz to iteration sa.skip. Forced off
      // under the glitch overlay so the diagnostic sees every iteration (matches the CPU).
      const args = { ox: p.offX, oy: p.offY, scale: p.scale, refLen: p.len,
                     maxIter: this.maxIter, glitchTol: g ? this.glitchTol : 0,
                     fastSkip: g ? 0 : 1, width: W, height: H, sa: g ? null : p.sa };
      // The deep floatexp-precision band uses the RESCALED engine: same depth range and
      // ~46-bit precision as renderPerturbFloatexp (it still does escape/rebase in exact
      // floatexp), but a ~1.3-2× faster shared-exponent update. renderPerturbFloatexp
      // remains in the renderer as the reference/oracle (tools/validate-gpu.mjs gates both).
      // Both draw in horizontal strips (_drawTiledEscape) so a deep frame (maxIter ~55k
      // at 2^218) never exceeds the GPU watchdog — the actual barrier past ~2^218.
      const drawStrip = fe
        ? (y, h) => this.gpu.renderPerturbRescaled({ ...args, stripY: y, stripH: h })
        : (y, h) => this.gpu.renderPerturbDf64({ ...args, stripY: y, stripH: h });
      // Strip budget on the POST-SA-SKIP worst case (mirrors _setSA's active condition).
      const skip = (args.sa && args.sa.skip > 0 && args.sa.skip < this.maxIter) ? args.sa.skip : 0;
      const done = await this._drawTiledEscape(drawStrip, gen, skip);
      if (!done) return;                              // superseded; a newer render owns the canvas
      this._terminatePool();
      this._finishGpu(fe ? 'gpu-perturb-fe' : 'gpu-perturb', 0);
    } catch (e) { if (gen === this.gen) this._gpuFail(e); }
  }

  // ---- offscreen GPU path (raster in the worker) ----
  // Hand a self-contained render `plan` to the GPU worker and composite the strips it
  // pushes back. Each strip is an already-rasterized display-res ImageBitmap; the main
  // thread only draws it (no GL sync) over THIS strip's display rows — exactly the
  // _blitGpuStrip compositing the on-main-thread path does, but without the per-strip
  // GPU flush that janks input. onDone finalizes via the same _finishGpu as the main path.
  _dispatchWorkerRender(plan, transfers, engineLabel) {
    const gen = plan.gen;
    this.gpuWorker.renderEscape(plan, transfers, {
      onStrip: (bitmap, y0, h0) => {
        if (gen !== this.gen) { bitmap.close?.(); return; }
        this.ctx.imageSmoothingEnabled = false;
        this.ctx.drawImage(bitmap, 0, y0, this.backingW, h0, 0, y0, this.backingW, h0);
        bitmap.close?.();
      },
      onDone: (glitches) => { if (gen === this.gen) this._finishGpu(engineLabel, glitches); },
      onError: (e) => { if (gen === this.gen) this._gpuFail(e); },
    });
  }

  _renderGpuNaiveWorker() {
    const W = this.cW, H = this.cH, gen = this.gen;
    const aspect = W / H, scale = (2 * this.radius) / H;
    const cx = toDouble(this.cx, this.prec), cy = toDouble(this.cy, this.prec);
    const ox = cx - this.radius * aspect, oy = cy - this.radius;
    this.onStatus({ phase: 'start', engine: 'gpu-naive', maxIter: this.maxIter, zoom: this.zoomLevel() });
    this._refMeta = { engine: 'gpu-naive', refLen: 0, relocations: 0 };
    this._dispatchWorkerRender({
      gen, engine: 'naive', W, H, ss: this._effSS, maxIter: this.maxIter,
      stripRows: this._stripRows(), paletteId: this.paletteOpts.paletteId,
      color: this._gpuColorOpts(), naive: { ox, oy, scale, df64: false }, wantGlitchCount: false,
    }, [], 'gpu-naive');
  }

  // Deep/medium GPU perturbation in the worker. `p` is the CPU worker's refReady params
  // (df64 reference zx/zy, dc origin offX/offY, scale, len, sa). The reference typed arrays
  // are TRANSFERRED to the GPU worker (the main thread no longer needs them — a view change
  // recomputes the reference fresh). Mirrors _renderGpuPerturb's glitch/SA gating exactly.
  _renderGpuPerturbWorker(p) {
    const W = this.cW, H = this.cH, gen = this.gen;
    const fe = this._mode === 'gpu-perturb-fe';
    const g = this.showGlitches;
    const sa = g ? null : p.sa;
    // Strip budget on the POST-SA-SKIP worst case (mirrors _setSA's active condition —
    // an inactive/absent seed means pixels really do run from iteration 0). See _stripRows.
    const skip = (sa && sa.skip > 0 && sa.skip < this.maxIter) ? sa.skip : 0;
    this._dispatchWorkerRender({
      gen, engine: fe ? 'perturb-fe' : 'perturb', W, H, ss: this._effSS, maxIter: this.maxIter,
      stripRows: this._stripRows(skip), paletteId: this.paletteOpts.paletteId, color: this._gpuColorOpts(),
      perturb: { ox: p.offX, oy: p.offY, scale: p.scale, refLen: p.len,
                 glitchTol: g ? this.glitchTol : 0, fastSkip: g ? 0 : 1, sa },
      refZx: p.zx, refZy: p.zy, wantGlitchCount: g,
    }, [p.zx.buffer, p.zy.buffer], fe ? 'gpu-perturb-fe' : 'gpu-perturb');
  }

  // Recolor the worker's existing sn buffer (palette / overlay-off, no recompute) and draw
  // the single full-frame bitmap it returns. Mirrors _recolorGpu for the offscreen path.
  _recolorGpuOffscreen() {
    const gen = this.gen;
    const color = { ...this._gpuColorOpts(), paletteId: this.paletteOpts.paletteId };
    this.gpuWorker.recolor(gen, color, (bitmap) => {
      if (!bitmap) { this.render(); return; }
      if (gen !== this.gen) { bitmap.close?.(); return; }
      this.ctx.imageSmoothingEnabled = false;
      this.ctx.drawImage(bitmap, 0, 0);
      bitmap.close?.();
      this.sctx.drawImage(this.canvas, 0, 0);
    });
  }

  _recolorGpu() {
    try {
      this._setGpuLUT();
      this.gpu.colorize(this._gpuColorOpts());
      this._blitGpu();
      this.sctx.drawImage(this.canvas, 0, 0);
    } catch (e) { this.render(); }
  }

  _distributeBands(params) {
    // round-robin row-bands across the pool for load balance (interleaved so
    // each worker gets a spread of cheap + expensive rows). At compute res.
    const H = this.cH, P = this._pool.length, B = this.bandRows;
    const starts = [];
    for (let y = 0; y < H; y += B) starts.push(y);
    this._tilesLeft = this._pool.length;
    for (let w = 0; w < P; w++) {
      const bands = [];
      for (let k = w; k < starts.length; k += P) bands.push(starts[k]);
      // postMessage structured-clones `params` (incl. the ref arrays) per send,
      // so each worker gets its own copy — no manual cloning, no transfer.
      this._pool[w].postMessage({
        type: 'render', gen: this.gen, params, bands, bandRows: B,
        width: this.cW, height: this.cH,
      });
    }
  }

  _onWorker(m, gen) {
    if (gen !== this.gen) return; // stale message from a superseded render
    if (m.type === 'progress') {
      this.onStatus({ phase: m.phase, i: m.i, total: m.total });
    } else if (m.type === 'refReady') {
      this._refMeta = { engine: m.engine, refLen: m.refLen, relocations: m.relocations };
      this._saSkip = (m.params && m.params.sa && m.params.sa.skip) || 0;
      if (this._mode === 'gpu-perturb' || this._mode === 'gpu-perturb-fe') { // GPU rasterizes from the worker's reference
        this._refMeta.engine = this._mode;
        // Stow the reference-reuse cache BEFORE dispatch (the offscreen path
        // TRANSFERS params.zx/zy away). Copies + the BigInt tail let later
        // same-neighborhood renders skip or merely extend this build.
        if (m.refCache) {
          this._refCache = {
            cx: BigInt(m.refCache.cx), cy: BigInt(m.refCache.cy), prec: m.refCache.prec,
            len: m.params.len, escaped: m.refCache.escaped,
            tailBx: m.refCache.tailBx, tailBy: m.refCache.tailBy,
            zx: m.params.zx.slice(), zy: m.params.zy.slice(), z2: m.params.z2.slice(),
            sa: m.params.sa || null, saBox: m.refCache.saBox || null, saAtIter: this.maxIter,
          };
        }
        if (this._useOffscreen()) this._renderGpuPerturbWorker(m.params);
        else this._renderGpuPerturb(m.params);
        return;
      }
      const c = m.coarse;
      colorizeBlocks(this.img.data, c.sn, this.cW, this.cH, c.snW, c.snH, c.step, this.paletteOpts,
                     this.showGlitches ? c.glitch : null);
      this._paintCtx.putImageData(this.img, 0, 0);
      this._schedulePresent();
      this._distributeBands(m.params); // params keeps the (transferred-back) arrays
    } else if (m.type === 'refExtended') {
      // Reference-reuse cache: merge the extension segment into the cached orbit,
      // then render from the (now long enough) cache. If the orbit ESCAPED during
      // the extension, it is still a valid (Zhuoran-rebased) reference for THIS
      // render; the next deeper render will miss (escaped + short) and rebuild
      // with relocation, matching chooseReference's normal behavior.
      const c = this._refCache;
      if (!c) return;
      if (m.merged) {
        // SA-refresh trip: the worker merged the arrays and recomputed the coeffs
        c.zx = m.zx; c.zy = m.zy; c.z2 = m.z2;
        c.sa = m.sa || null;
        const bb = this._cacheDcBox();
        c.saBox = { dcHalfX: bb.dcHalfX, dcHalfY: bb.dcHalfY };
        c.saAtIter = this.maxIter;
      } else {
        const grow = (old, seg) => {
          const a = new Float64Array(m.len + 1);
          a.set(old); a.set(seg, c.len + 1);
          return a;
        };
        c.zx = grow(c.zx, m.zx); c.zy = grow(c.zy, m.zy); c.z2 = grow(c.z2, m.z2);
      }
      c.len = m.len; c.escaped = m.escaped;
      c.tailBx = m.tailBx; c.tailBy = m.tailBy;
      this._gpuPerturbFromCache();
    } else if (m.type === 'saReady') {
      // worker-side computeSeries for a cache-hit render ('ready' + missing/stale SA)
      const c = this._refCache;
      if (!c) return;
      c.sa = m.sa || null;
      const bb = this._cacheDcBox();
      c.saBox = { dcHalfX: bb.dcHalfX, dcHalfY: bb.dcHalfY };
      c.saAtIter = this.maxIter;
      this._gpuPerturbFromCache();
    } else if (m.type === 'band') {
      const sn = m.sn, gl = m.glitch;
      for (let j = 0; j < m.h; j++) {
        const dst = (m.y0 + j) * this.cW;
        const src = j * m.w;
        for (let i = 0; i < m.w; i++) this._sn[dst + i] = sn[src + i];
        if (gl && this._glitch) for (let i = 0; i < m.w; i++) this._glitch[dst + i] = gl[src + i];
      }
      colorizeRegion(this.img.data, sn, this.cW, { x0: m.x0, y0: m.y0, w: m.w, h: m.h }, this.paletteOpts,
                     this.showGlitches ? gl : null);
      this._paintCtx.putImageData(this.img, 0, 0, m.x0, m.y0, m.w, m.h);
      this._schedulePresent();
    } else if (m.type === 'tilesDone') {
      this._glitchAcc += m.glitches;
      this._tilesLeft--;
      if (this._tilesLeft <= 0) {
        this.rendering = false;
        this._clearPresent();
        this._presentNow();                       // flush final supersample downscale
        this.sctx.drawImage(this.canvas, 0, 0); // snapshot for gesture previews
        const meta = this._refMeta || {};
        this.onStatus({ phase: 'done', glitches: this._glitchAcc, refLen: meta.refLen || 0, relocations: meta.relocations || 0, engine: meta.engine, zoom: this.zoomLevel(), saSkip: this._saSkip });
      }
    } else if (m.type === 'error') {
      this.rendering = false;
      this.onStatus({ error: m.message });
    }
  }

  _recolor() {
    // recolor whole image from cached sn (instant palette change / overlay toggle), at
    // compute res. Re-tints from the cached glitch mask when the overlay is on; passes
    // null when off so the clean fractal returns (byte-identical to a fresh render).
    colorizeRegion(this.img.data, this._sn, this.cW,
      { x0: 0, y0: 0, w: this.cW, h: this.cH }, this.paletteOpts,
      this.showGlitches ? this._glitch : null);
    this._paintCtx.putImageData(this.img, 0, 0);
    this._presentNow();
    this.sctx.drawImage(this.canvas, 0, 0);
  }

  // ---- gesture preview ----
  _applyPreview() {
    const t = this.T;
    this.ctx.save();
    this.ctx.fillStyle = '#000';
    this.ctx.fillRect(0, 0, this.backingW, this.backingH);
    // Point filtering (nearest), not bilinear: scaling the last frame during a
    // zoom/pan gesture stays crisp instead of going soft. The sharp re-render
    // replaces it once the gesture settles.
    this.ctx.imageSmoothingEnabled = false;
    this.ctx.setTransform(t.a, 0, 0, t.a, t.e, t.f);
    this.ctx.drawImage(this.stable, 0, 0);
    this.ctx.restore();
  }

  // Start a preview: snapshot the current frame as the scale base and cancel any
  // in-flight render so zoom/pan is instant and nothing renders mid-gesture. The
  // real (high-res) render is deferred to _endGesture (touch up) or the settle
  // timer (wheel/buttons). Idempotent — only the first call of a gesture acts.
  _beginPreview() {
    if (this.T) return;
    this._clearSettle();
    this.sctx.drawImage(this.canvas, 0, 0); // current frame becomes the base image
    this.T = { a: 1, e: 0, f: 0 };
    this.gen++;                 // invalidate any in-flight worker messages
    this._terminatePool();
    this.rendering = false;
  }

  _clearSettle() { if (this._settleTimer) { clearTimeout(this._settleTimer); this._settleTimer = 0; } }

  // Debounced commit for momentum-free zoom sources (wheel, buttons, click-to-zoom):
  // keep showing the scaled preview, then render sharply once motion stops.
  _scheduleSettle(delay = 220) {
    this._clearSettle();
    this._settleTimer = setTimeout(() => { this._settleTimer = 0; this._endGesture(); }, delay);
  }

  // Zoom by `factor` (radius *= factor) about backing pixel (px,py), keeping that
  // point fixed. Renders the current image scaled (preview) immediately and defers
  // the real render until the zoom settles. Used by wheel and the zoom buttons.
  zoomBy(factor, px = this.backingW / 2, py = this.backingH / 2) {
    // At the precision wall, ignore further zoom-IN (factor < 1) so the preview doesn't
    // over-scale into a blurry image that snaps back on settle. Zoom-OUT still works.
    if (factor < 1 && this.atMinRadius) { this.onStatus({ phase: 'maxdepth' }); return; }
    this._beginPreview();
    const s = 1 / factor;            // display scale (radius shrinks -> image grows)
    this.T.a *= s;
    this.T.e = px * (1 - s) + s * this.T.e;
    this.T.f = py * (1 - s) + s * this.T.f;
    this._applyPreview();
    this._scheduleSettle();
  }

  // Pan by (dxPix, dyPix) backing pixels using the SAME preview machinery as a drag,
  // so repeated keyboard / programmatic pans stay smooth: translate the snapshot now,
  // render sharply once motion settles. (dxPix, dyPix) is how far the IMAGE moves on
  // screen — matching the drag model (T.e/T.f are the image translation): a positive
  // dxPix slides the image right (revealing content to the left), exactly like a finger
  // drag to the right. _endGesture folds the pure translation (a=1) into the HP center.
  panByPreview(dxPix, dyPix) {
    if (!this.backingW) return;
    this._beginPreview();
    this.T.e += dxPix;
    this.T.f += dyPix;
    this._applyPreview();
    this._scheduleSettle();
  }

  // Click-to-zoom: recenter the clicked backing pixel (px,py) to the screen center
  // AND zoom by `factor` (radius *= factor; <1 zooms in). The clicked complex point
  // becomes the new view center. Like zoomBy it shows the scaled preview instantly
  // and defers the sharp render to the settle timer. The transform G applied in
  // displayed space is "scale by s=1/factor about (px,py), then translate (px,py)
  // to center"; composing G∘T gives the update below (so it composes with any
  // preview already in progress, and _endGesture folds it into the HP view state:
  // new center = the complex point at (px,py), new radius = radius*factor).
  clickZoom(px, py, factor = 0.5) {
    if (!this.backingW) return;
    if (factor < 1 && this.atMinRadius) { this.onStatus({ phase: 'maxdepth' }); return; }
    this._beginPreview();
    const s = 1 / factor;
    const t = this.T;
    t.e = s * t.e + this.backingW / 2 - s * px;   // uses the OLD t.e/t.f
    t.f = s * t.f + this.backingH / 2 - s * py;
    t.a = s * t.a;
    this._applyPreview();
    this._scheduleSettle();
  }

  _endGesture() {
    if (!this.T) return;
    const t = this.T;
    const W = this.backingW, H = this.backingH;
    // complex point now shown at screen center, from the OLD stable image
    const ox = (W / 2 - t.e) / t.a;
    const oy = (H / 2 - t.f) / t.a;
    const { dx, dy } = this._pixelDelta(ox, oy);
    this.cx += fromDouble(dx, this.prec);
    this.cy += fromDouble(dy, this.prec);
    this.radius = this.radius / t.a;
    this._afterRadiusChange();
    this.T = null;
    this.render();
  }

  _installGestures() {
    const c = this.canvas;
    c.style.touchAction = 'none';

    const down = (ev) => {
      c.setPointerCapture?.(ev.pointerId);
      this._pointers.set(ev.pointerId, this._toBacking(ev.clientX, ev.clientY));
      if (this._pointers.size === 2) {
        const pts = [...this._pointers.values()];
        this._pinch = {
          startDist: dist(pts[0], pts[1]),
          startMid: mid(pts[0], pts[1]),
        };
      }
      // A single pointerdown starts nothing yet: a tap with no movement must not
      // cancel an in-flight render. Pan/pinch previews begin lazily on the first
      // real move; a tap that never moves becomes a click-to-zoom on pointerup.
    };
    const move = (ev) => {
      if (!this._pointers.has(ev.pointerId)) return;
      const prev = this._pointers.get(ev.pointerId);
      const cur = this._toBacking(ev.clientX, ev.clientY);
      this._pointers.set(ev.pointerId, cur);
      if (this._pointers.size === 1) {
        const dx = cur.x - prev.x, dy = cur.y - prev.y;
        if (!this.T && Math.hypot(dx, dy) < 1) return; // ignore sub-pixel jitter
        this._beginPreview(); // lazy: snapshot + cancel in-flight render on real motion
        // pan
        this.T.e += dx;
        this.T.f += dy;
        this._applyPreview();
      } else if (this._pointers.size === 2 && this._pinch) {
        this._beginPreview();
        const pts = [...this._pointers.values()];
        const d = dist(pts[0], pts[1]);
        const mp = mid(pts[0], pts[1]);
        const k = d / (this._pinch._lastDist || this._pinch.startDist);
        // scale around current midpoint
        this.T.a *= k;
        this.T.e = mp.x + (this.T.e - mp.x) * k;
        this.T.f = mp.y + (this.T.f - mp.y) * k;
        // translate by midpoint movement
        if (this._pinch._lastMid) {
          this.T.e += mp.x - this._pinch._lastMid.x;
          this.T.f += mp.y - this._pinch._lastMid.y;
        }
        this._pinch._lastDist = d;
        this._pinch._lastMid = mp;
        this._applyPreview();
      }
    };
    const up = (ev) => {
      if (!this._pointers.has(ev.pointerId)) return;
      const downPos = this._pointers.get(ev.pointerId); // ~tap location (down≈up)
      this._pointers.delete(ev.pointerId);
      if (this._pointers.size === 0) {
        this._pinch = null;
        if (this.T) { this._endGesture(); return; } // a pan/pinch preview was active
        // No preview started -> a tap/click with no drag: click-to-zoom, recentering
        // on the point. Shift / Ctrl / right-button zoom OUT instead of IN.
        if (ev.type !== 'pointercancel') {
          const out = ev.shiftKey || ev.ctrlKey || ev.button === 2;
          this.clickZoom(downPos.x, downPos.y, out ? 2 : 0.5);
        }
      } else if (this._pointers.size === 1) {
        // transition pinch -> pan; reset pinch refs
        this._pinch = null;
      }
    };
    c.addEventListener('pointerdown', down);
    c.addEventListener('pointermove', move);
    c.addEventListener('pointerup', up);
    c.addEventListener('pointercancel', up);
    // right-click zooms out; suppress the browser context menu on the canvas
    c.addEventListener('contextmenu', (ev) => ev.preventDefault());

    // wheel zoom (desktop): scaled preview now, sharp render once scrolling stops.
    // Wheel zooms about the cursor (keeps the point under it fixed) rather than
    // recentering — the expected scroll-to-zoom feel; a click recenters instead.
    c.addEventListener('wheel', (ev) => {
      ev.preventDefault();
      const p = this._toBacking(ev.clientX, ev.clientY);
      const factor = ev.deltaY > 0 ? 1.2 : 1 / 1.2;
      this.zoomBy(factor, p.x, p.y);
    }, { passive: false });
  }
}

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }
function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 }; }
