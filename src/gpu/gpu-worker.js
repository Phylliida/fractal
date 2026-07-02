// gpu-worker.js — the OffscreenCanvas GPU render worker (Spawn 24).
//
// Owns an OffscreenCanvas + a GpuRenderer (the same WebGL2 df64/floatexp/rescaled
// shaders the main thread used to run). Runs the ENTIRE escape+color strip loop here —
// the loop that used to live in viewer._drawTiledEscape — and pushes each finished strip
// back to the main thread as an already-rasterized ImageBitmap (transferable). The main
// thread just draws the bitmap onto its 2-D display canvas: no GL sync, no jank.
//
// makeCanvas() in renderer.js already returns an OffscreenCanvas when `document` is
// undefined, so the GpuRenderer runs unchanged here. glsl.js + palette.js are pure ESM
// and import cleanly in a module worker.
//
// Protocol: see gpu-worker-client.js. Strips are gated by an `ack` from the main thread
// (back-pressure: ≤1 bitmap in flight). A newer render (higher gen) supersedes the
// in-flight loop — every loop bails its next iteration when this._gen changes.

import { GpuRenderer } from './renderer.js';
import { paletteRgbAt } from '../palette.js';

let gpu = null;
let gen = -1;                       // current render generation (from the viewer)
let paletteId = null;              // last LUT uploaded (skip re-upload when unchanged)
let loseExt = null;                // WEBGL_lose_context handle (test-only loss simulation)
const ackWaiters = new Map();      // gen -> resolve(): the per-strip back-pressure gate

function post(msg, transfer) { self.postMessage(msg, transfer || []); }

function ensureGpu() {
  if (gpu) return gpu;
  // 1×1 to start; colorize() resizes the canvas to the display resolution.
  gpu = new GpuRenderer({
    width: 1, height: 1,
    onContextLost: () => post({ type: 'contextlost', gen }),
    onContextRestored: () => { paletteId = null; post({ type: 'contextrestored' }); },
  });
  return gpu;
}

self.onmessage = (e) => {
  const m = e.data;
  try {
    if (m.type === 'init') {
      const g = ensureGpu();
      let info = 'WebGL2 (worker)';
      if (g.supported && g.gl) {
        const dbg = g.gl.getExtension('WEBGL_debug_renderer_info');
        if (dbg) info = String(g.gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL));
      }
      post({ type: 'ready', supported: !!(g && g.supported), info });
    } else if (m.type === 'render') {
      gen = m.gen;
      wakeAckWaiters();            // wake any stale loop so it bails
      runRender(m).catch((err) => { if (m.gen === gen) post({ type: 'error', gen: m.gen, message: String((err && err.stack) || err) }); });
    } else if (m.type === 'ack') {
      const w = ackWaiters.get(m.gen);
      if (w) { ackWaiters.delete(m.gen); w(); }
    } else if (m.type === 'recolor') {
      gen = m.gen;
      recolor(m);
    } else if (m.type === 'cancel') {
      gen = m.gen;
      wakeAckWaiters();
    } else if (m.type === 'dispose') {
      if (gpu) { try { gpu.dispose(); } catch { /* noop */ } gpu = null; }
    } else if (m.type === '__lose') {
      // Test-only: simulate a mobile GPU context loss on the worker's OffscreenCanvas. The
      // renderer's webglcontextlost listener fires → onContextLost → 'contextlost' to main.
      try { loseExt = gpu && gpu.gl && gpu.gl.getExtension('WEBGL_lose_context'); if (loseExt) loseExt.loseContext(); } catch { /* noop */ }
    } else if (m.type === '__restore') {
      try { if (loseExt) loseExt.restoreContext(); } catch { /* noop */ }
    }
  } catch (err) {
    post({ type: 'error', gen: m && m.gen, message: String((err && err.stack) || err) });
  }
};

function wakeAckWaiters() { for (const w of ackWaiters.values()) w(); ackWaiters.clear(); }
function waitAck(myGen) { return new Promise((resolve) => ackWaiters.set(myGen, resolve)); }

function setLUT(pid) {
  if (paletteId === pid) return;
  gpu.setPaletteLUT((u) => paletteRgbAt(pid, u), 1024);
  paletteId = pid;
}

// Build the per-engine strip draw call. Mirrors viewer._renderGpuNaive / _renderGpuPerturb:
// the same params, the same scissor-strip (stripY/stripH) bit-identical to one big draw.
function makeDrawStrip(plan) {
  const W = plan.W, H = plan.H;
  if (plan.engine === 'naive') {
    const n = plan.naive;
    return (y, h) => gpu.renderNaive({ ox: n.ox, oy: n.oy, scale: n.scale, maxIter: plan.maxIter,
                                       width: W, height: H, df64: !!n.df64, stripY: y, stripH: h });
  }
  // perturb (df64 shallow band) or perturb-fe (rescaled deep band)
  const p = plan.perturb;
  gpu.uploadReferenceDf64(plan.refZx, plan.refZy);
  const args = { ox: p.ox, oy: p.oy, scale: p.scale, refLen: p.refLen, maxIter: plan.maxIter,
                 glitchTol: p.glitchTol || 0, fastSkip: p.fastSkip === 0 ? 0 : 1, width: W, height: H, sa: p.sa || null };
  return plan.engine === 'perturb-fe'
    ? (y, h) => gpu.renderPerturbRescaled({ ...args, stripY: y, stripH: h })
    : (y, h) => gpu.renderPerturbDf64({ ...args, stripY: y, stripH: h });
}

// The escape+color strip loop (was viewer._drawTiledEscape). Renders one strip, colorizes
// the full canvas, transfers it to an ImageBitmap, posts it, and waits for the main thread's
// ack before the next strip. Bails immediately (before any GL) whenever a newer render
// superseded this one (gen changed). y0/h0 are reported in DISPLAY rows (the bitmap is the
// colorized canvas, already box-averaged down by ss), matching the main thread's strip blit.
async function runRender(plan) {
  const myGen = plan.gen;
  const g = ensureGpu();
  if (!g.supported) { post({ type: 'error', gen: myGen, message: 'worker GPU unsupported' }); return; }
  const W = plan.W, H = plan.H, ss = Math.max(1, plan.ss | 0 || 1);
  const color = { ...plan.color, ss };
  const drawStrip = makeDrawStrip(plan);   // also uploads the reference for perturb
  g.clearSn(W, H, -1);                       // unwritten rows read as interior (clean reveal)
  setLUT(plan.paletteId);
  const stripH = Math.max(ss, plan.stripRows | 0 || H);
  for (let y = 0; y < H; y += stripH) {
    if (myGen !== gen) return;               // superseded — stop, leave the canvas to the new render
    const h = Math.min(stripH, H - y);
    drawStrip(y, h);
    g.gl.flush();                            // submit this strip as its own GPU command
    g.colorize(color);                       // full-canvas color pass (reads accumulated sn)
    const bitmap = g.canvas.transferToImageBitmap();   // strip's GPU work completes here, off-main
    const y0 = Math.round(y / ss), h0 = Math.round(h / ss);
    post({ type: 'strip', gen: myGen, y0, h0, bitmap }, [bitmap]);
    await waitAck(myGen);                     // back-pressure: ≤1 bitmap in flight
  }
  if (myGen !== gen) return;
  let glitches = 0;
  if (plan.wantGlitchCount && plan.engine !== 'naive') { try { glitches = g.countGlitches(); } catch { /* keep 0 */ } }
  post({ type: 'done', gen: myGen, glitches });
}

// Re-colorize the existing sn buffer (palette / overlay toggle without recompute) and
// hand back ONE full-frame display bitmap. Mirrors viewer._recolorGpu.
function recolor(m) {
  const g = ensureGpu();
  if (!g.supported || !g._fbo) { post({ type: 'recolored', gen: m.gen, seq: m.seq, bitmap: null }); return; }
  setLUT(m.color.paletteId ?? paletteId);
  g.colorize(m.color);
  const bitmap = g.canvas.transferToImageBitmap();
  post({ type: 'recolored', gen: m.gen, seq: m.seq, bitmap }, [bitmap]);
}
