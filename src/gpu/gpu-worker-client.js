// gpu-worker-client.js — main-thread proxy for the OffscreenCanvas GPU worker.
//
// Why this exists (Spawn 24): the GPU raster runs on the MAIN thread today. Even with
// strip-tiling + rAF yields, every strip's `ctx.drawImage(gpu.canvas)` (viewer
// _blitGpuStrip) forces a SYNCHRONOUS GPU flush/readback on the main thread — a
// "synchronous pull" that janks input/scroll during a deep render. This client moves
// the whole escape+color loop into a dedicated worker that owns an OffscreenCanvas; the
// worker pushes finished, already-rasterized ImageBitmap strips back (transferable, no
// GPU sync on the main thread) — an "asynchronous push". The main thread only draws a
// ready bitmap onto its 2-D display canvas, so the heavy GL sync is off-main-thread.
//
// The WebGL context, the GpuRenderer, the df64/fe/rescaled shaders, and the per-strip
// watchdog tiling all live in the worker now; this is just the message bridge. The CPU
// reference-orbit worker pool (worker.js) is unchanged — it still computes the reference
// and hands it to the viewer, which forwards it here for rasterization.
//
// Back-pressure: the worker waits for an `ack` before drawing the next strip, so at most
// one in-flight ImageBitmap sits in the message queue (no unbounded bitmap backlog if the
// main thread can't keep up). Cancellation: a newer render (higher gen) supersedes; the
// client ignores stale-gen messages and the worker bails its loop on a gen bump.

export class GpuWorkerClient {
  constructor(opts = {}) {
    this.onContextLost = opts.onContextLost || null;
    this.onContextRestored = opts.onContextRestored || null;
    this.onFail = opts.onFail || null;     // worker unsupported / fatal — viewer falls back
    // supported: undefined until the worker's init handshake replies. We construct this
    // client only after a synchronous main-thread probe confirms OffscreenCanvas+WebGL2,
    // so render() may optimistically route here before `ready` resolves.
    this.supported = undefined;
    this.info = 'WebGL2 (worker)';
    this.lost = false;
    this._gen = -1;                        // current render generation (viewer's gen)
    this._handlers = null;                 // { onStrip, onDone, onError } for the live render
    this._readyResolvers = [];
    this._disposed = false;

    this.worker = new Worker(new URL('./gpu-worker.js', import.meta.url), { type: 'module' });
    this.worker.onmessage = (e) => this._onMessage(e.data);
    this.worker.onerror = (err) => this._fail(String((err && err.message) || err));
    this.worker.postMessage({ type: 'init' });
  }

  // Resolves to true/false once the worker reports whether it has a usable WebGL2 GPU.
  ready() {
    if (this.supported !== undefined) return Promise.resolve(this.supported);
    return new Promise((res) => this._readyResolvers.push(res));
  }

  _onMessage(m) {
    if (this._disposed) return;
    switch (m.type) {
      case 'ready':
        this.supported = !!m.supported;
        if (m.info) this.info = m.info;
        for (const r of this._readyResolvers) r(this.supported);
        this._readyResolvers = [];
        if (!this.supported && this.onFail) this.onFail('worker GPU unsupported');
        break;
      case 'strip':
        if (m.gen !== this._gen || !this._handlers) { this._ack(m.gen); break; }
        // Draw the finished strip onto the display canvas, then ack so the worker
        // proceeds. onStrip MUST close the bitmap (it owns it after transfer).
        try { this._handlers.onStrip(m.bitmap, m.y0, m.h0); }
        finally { this._ack(m.gen); }
        break;
      case 'done':
        if (m.gen !== this._gen || !this._handlers) break;
        { const h = this._handlers; this._handlers = null; h.onDone(m.glitches | 0); }
        break;
      case 'recolored':
        // Only the LATEST recolor's reply is drawn (rapid palette-slider drags post several;
        // out-of-order/older replies are discarded so the final frame is the newest palette).
        if (m.gen !== this._gen || m.seq !== this._recolorSeq || !this._recolorCb) { if (m.bitmap) m.bitmap.close?.(); break; }
        { const cb = this._recolorCb; this._recolorCb = null; cb(m.bitmap); }
        break;
      case 'snSample':
        if (this._sampleCb) { const cb = this._sampleCb; this._sampleCb = null; cb(m.sn ? { sn: m.sn, iter: m.iter } : null); }
        break;
      case 'contextlost':
        this.lost = true;
        if (this.onContextLost) { try { this.onContextLost(); } catch { /* noop */ } }
        break;
      case 'contextrestored':
        this.lost = false;
        if (this.onContextRestored) { try { this.onContextRestored(); } catch { /* noop */ } }
        break;
      case 'error':
        if (m.gen === this._gen && this._handlers) {
          const h = this._handlers; this._handlers = null; h.onError(new Error(m.message));
        }
        break;
      default: break;
    }
  }

  _ack(gen) { if (!this._disposed) this.worker.postMessage({ type: 'ack', gen }); }

  _fail(msg) {
    this.supported = false;
    for (const r of this._readyResolvers) r(false);
    this._readyResolvers = [];
    if (this._handlers) { const h = this._handlers; this._handlers = null; try { h.onError(new Error(msg)); } catch { /* noop */ } }
    if (this.onFail) this.onFail(msg);
  }

  // Start an escape render in the worker. `plan` is the self-contained render description
  // (engine, geometry, maxIter, strip plan, palette, color opts, and — for perturb — the
  // df64 reference arrays). `transfers` lists buffers to transfer (the reference typed
  // arrays). handlers = { onStrip(bitmap,y0,h0), onDone(glitches), onError(err) }.
  renderEscape(plan, transfers, handlers) {
    this._gen = plan.gen;
    this._handlers = handlers;
    this.worker.postMessage({ type: 'render', ...plan }, transfers || []);
  }

  // Re-colorize the worker's existing sn buffer (palette / overlay toggle, no recompute)
  // and hand back ONE full-frame display bitmap via cb. Mirrors the main-thread
  // _recolorGpu fast path. gen must match the last render so the worker's sn is current.
  recolor(gen, color, cb) {
    this._gen = gen;
    this._recolorCb = cb;
    this._recolorSeq = (this._recolorSeq || 0) + 1;
    this.worker.postMessage({ type: 'recolor', gen, color, seq: this._recolorSeq });
  }

  // Supersede any in-flight render (the worker bails its strip loop on a gen bump).
  cancel(gen) { this._gen = gen; this._handlers = null; if (!this._disposed) this.worker.postMessage({ type: 'cancel', gen }); }

  // Sample sn/iter at scattered texels of the last completed frame (Spawn 33 — the
  // runtime GPU precision self-test). Resolves null if the frame was superseded.
  sampleSn(gen, points) {
    if (this._disposed) return Promise.resolve(null);
    return new Promise((res) => {
      this._sampleCb = res;
      this.worker.postMessage({ type: 'sampleSn', gen, points });
    });
  }

  // Test-only: simulate a GPU context loss / restore on the worker's OffscreenCanvas, so the
  // mobile recovery path (worker→main forwarding → CPU detour → GPU on restore) is exercisable
  // end-to-end (tools/probe-recover-offscreen.mjs). The main-thread GpuRenderer has no canvas
  // here, so the WEBGL_lose_context handle lives in the worker.
  loseContextForTest() { if (!this._disposed) this.worker.postMessage({ type: '__lose' }); }
  restoreContextForTest() { if (!this._disposed) this.worker.postMessage({ type: '__restore' }); }

  dispose() {
    this._disposed = true;
    this._handlers = null;
    try { this.worker.postMessage({ type: 'dispose' }); } catch { /* noop */ }
    try { this.worker.terminate(); } catch { /* noop */ }
  }
}
