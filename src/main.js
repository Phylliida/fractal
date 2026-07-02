// main.js — wire the Viewer to the DOM controls, status line, and URL hash.
import { Viewer, SS_DEEP_CAP_RADIUS } from './viewer.js';
import { autoMaxIter } from './math/render.js';

const $ = (id) => document.getElementById(id);
const canvas = $('view');

// `?offscreen=0` forces the legacy on-main-thread GPU renderer (default: the OffscreenCanvas
// worker path when supported). A query param, not a hash key — the hash is reserved for the
// shareable view state, and this is a session/debug choice, not part of a bookmark.
const _q = new URLSearchParams(location.search);
const viewer = new Viewer(canvas, {
  onStatus: updateStatus,
  onView: (s) => { updateView(s); scheduleHash(); },
  offscreen: _q.get('offscreen') === '0' ? false : undefined,
});

// expose for tests / debugging
window.__viewer = viewer;

// ---------- status / readouts ----------
function updateStatus(s) {
  const el = $('status');
  if (s.error) { el.textContent = '⚠ ' + s.error.split('\n')[0]; return; }
  if (s.phase === 'maxdepth') { el.textContent = '⚠ maximum zoom depth reached (double-precision limit)'; return; }
  if (s.phase === 'gpuverify' && s.ok === false) {
    el.textContent = `⚠ GPU precision self-test failed (${Math.round(s.mismFrac * 100)}% of sampled pixels ` +
      `differ from the CPU reference) — deep detail may be off on this GPU; toggle "GPU acceleration" off to compare`;
    return;
  }
  if (s.phase === 'start') el.textContent = `rendering · ${s.engine} · ${s.maxIter} it`;
  else if (s.phase === 'reference') el.textContent = `reference orbit ${pct(s.i, s.total)}`;
  else if (s.phase === 'render') el.textContent = `rendering ${pct(s.i, s.total)}`;
  else if (s.phase === 'done') {
    el.textContent = `${s.engine} · done${s.glitches ? ' · ' + s.glitches + ' glitch?' : ''}` +
      (s.gpuVerify === 'fail' ? ' · ⚠ GPU self-test failed' : '');
    $('debug').textContent = debugText(s);
    window.__lastDone = s;
    window.__doneCount = (window.__doneCount || 0) + 1; // test sync signal
  }
}
function pct(i, t) { return t ? Math.min(100, Math.round((i / t) * 100)) + '%' : ''; }
// Annotate the supersample readout: capped down for depth, or HQ-forced past the cap.
function ssNote() {
  if (viewer._effSS < viewer.ss) return ` (capped from ${viewer.ss}× for depth)`;
  if (viewer.forceHighQuality && viewer.radius < SS_DEEP_CAP_RADIUS && viewer._effSS > 1) {
    return ' (HQ: depth cap overridden)';
  }
  return '';
}
function debugText(s) {
  const v = viewer.getState();
  const gpu = viewer.gpuInfo();
  return [
    `engine     ${s.engine}`,
    `gpu        ${gpu ? gpu.replace(/^ANGLE \(/, '').slice(0, 48) : 'off (CPU workers)'}`,
    `zoom       2^${v.zoom.toFixed(2)}  (radius ${v.radius.toExponential(3)})${v.atMinRadius ? '  · AT PRECISION WALL' : ''}`,
    `maxIter    ${v.maxIter}`,
    `supersample ${viewer._effSS}×${ssNote()}  (compute ${viewer.cW}×${viewer.cH})`,
    `precision  ${viewer.prec} bits`,
    `refLen     ${s.refLen}  relocations ${s.relocations}${s.saSkip ? `  · SA skip ${s.saSkip} (${(100 * s.saSkip / Math.max(1, v.maxIter)).toFixed(0)}%)` : ''}`,
    `glitches   ${s.glitches}`,
    `backing    ${viewer.backingW}×${viewer.backingH} @dpr${viewer.dpr.toFixed(2)}${viewer.lowPower ? '  · low power' : ''}`,
  ].join('\n');
}
function updateView(s) {
  $('zoom').textContent = '2^' + s.zoom.toFixed(1) + (s.atMinRadius ? ' (max)' : '');
  // Always-on center readout in complex "a + bi" form — orientation while exploring;
  // the panel's Re/Im fields keep the full precision.
  const re = shortCoord(s.cx), im = shortCoord(s.cy);
  const op = im.startsWith('−') ? '−' : '+';
  $('coords').textContent = `${re} ${op} ${im.replace(/^−/, '')}i`;
  if (document.activeElement !== $('reIn')) $('reIn').value = s.cx;
  if (document.activeElement !== $('imIn')) $('imIn').value = s.cy;
  if (document.activeElement !== $('radIn')) $('radIn').value = s.radius.toExponential(6);
  setIterUI(s.maxIter);
}

// Compact a high-precision decimal coord string for the always-on HUD readout. Keeps the
// sign + integer part + up to `digits` fractional digits, trims trailing zeros, and only
// appends an ellipsis when a *significant* (non-zero) digit was actually dropped — so an
// exact value like -0.5 shows "−0.5", not "−0.5000000000…". Real minus sign; "0" for zero.
function shortCoord(str, digits = 10) {
  const s = String(str).trim();
  const neg = s.startsWith('-');
  const body = neg ? s.slice(1) : s;
  const dot = body.indexOf('.');
  const intPart = dot < 0 ? body : body.slice(0, dot);
  const frac = dot < 0 ? '' : body.slice(dot + 1);
  const kept = frac.slice(0, digits).replace(/0+$/, '');   // shown digits, trailing zeros trimmed
  const lost = /[1-9]/.test(frac.slice(digits));            // dropped a non-zero digit -> ellipsis
  if (intPart.replace(/^0+/, '') === '' && kept === '' && !lost) return '0'; // true zero
  return (neg ? '−' : '') + intPart + (kept ? '.' + kept : '') + (lost ? '…' : '');
}

// Keep the iteration slider + number field in sync with the current maxIter,
// without clobbering whichever control the user is actively editing. The slider
// is clamped to its track; the number field shows the true value (can exceed it).
function setIterUI(v) {
  const slider = $('iter'), num = $('iterNum');
  if (document.activeElement !== slider) {
    slider.value = Math.min(+slider.max, Math.max(+slider.min, v));
  }
  if (document.activeElement !== num) num.value = v;
}

// ---------- URL hash (bookmarks / shareable deep coords) ----------
let hashTimer = 0;
function scheduleHash() { clearTimeout(hashTimer); hashTimer = setTimeout(writeHash, 400); }
function writeHash() {
  const s = viewer.getState();
  const p = new URLSearchParams();
  p.set('re', s.cx); p.set('im', s.cy); p.set('r', s.radius.toExponential(8));
  p.set('i', s.maxIter); p.set('p', viewer.paletteOpts.paletteId);
  p.set('cy', viewer.paletteOpts.cycle); p.set('sh', viewer.paletteOpts.shift);
  p.set('ss', viewer.ss);
  p.set('res', viewer.resScale);
  history.replaceState(null, '', '#' + p.toString());
}
function readHash() {
  if (!location.hash || location.hash.length < 2) return false;
  const p = new URLSearchParams(location.hash.slice(1));
  if (!p.get('re')) return false;
  if (p.get('p')) viewer.paletteOpts.paletteId = p.get('p');
  if (p.get('cy')) viewer.paletteOpts.cycle = +p.get('cy');
  if (p.get('sh')) viewer.paletteOpts.shift = +p.get('sh');
  if (p.get('ss')) viewer.ss = Math.max(1, Math.min(4, +p.get('ss')));
  // Resolution setting rides in the URL too (Danielle's request — originally kept
  // per-device to avoid shared-link degradation traps; the panel select shows the
  // state, so a recipient can see + undo it). A change needs a RESIZE (backing dims),
  // not just a re-render — done below after setState; the interim render is cancelled.
  let needResize = false;
  if (p.get('res')) {
    const r = Math.max(1, Math.min(4, Math.round(+p.get('res')) || 1));
    if (r !== viewer.resScale) { viewer.resScale = r; needResize = true; }
  }
  // writeHash records i= on EVERY url, so for almost all shared/bookmarked views the
  // recorded i is just the AUTO value at capture time. Pinning autoIter=false on every
  // load silently froze the iteration budget — zoom deeper from a loaded URL and the
  // view under-iterates (fine detail reads as interior), the Spawn-31 trap Danielle
  // hit live (her session carried i=26100 = auto-for-2^-103 down to 2^-251). Only pin
  // manual when i clearly DIFFERS from auto for this radius — a deliberate choice.
  const radius = +p.get('r');
  const iRaw = p.get('i') ? +p.get('i') : undefined;
  const auto = autoMaxIter(radius);
  const manualPin = iRaw !== undefined && Math.abs(iRaw - auto) > 0.02 * auto;
  viewer.setState({ cx: p.get('re'), cy: p.get('im'), radius,
                    maxIter: manualPin ? iRaw : undefined });
  if (manualPin) { viewer.autoIter = false; $('autoIter').checked = false; }
  else if (iRaw !== undefined) { viewer.autoIter = true; $('autoIter').checked = true; }
  if (needResize) viewer.resize();   // re-sizes the backing + re-renders at the new res
  syncControls();
  return true;
}

// ---------- controls ----------
function syncControls() {
  $('palette').value = viewer.paletteOpts.paletteId;
  $('cycle').value = viewer.paletteOpts.cycle;
  $('shift').value = viewer.paletteOpts.shift;
  $('cycleVal').textContent = viewer.paletteOpts.cycle;
  $('autoIter').checked = viewer.autoIter;
  $('ss').value = String(viewer.ss);
  $('resScale').value = String(viewer.resScale);
  $('showGlitch').checked = viewer.showGlitches;
  $('forceHQ').checked = viewer.forceHighQuality;
  $('lowPower').checked = viewer.lowPower;
  $('series').checked = viewer.series;
}

$('panelToggle').addEventListener('click', () => $('panel').classList.toggle('open'));
$('panelClose').addEventListener('click', () => $('panel').classList.remove('open'));

$('zoomIn').addEventListener('click', () => viewer.zoomBy(0.5));
$('zoomOut').addEventListener('click', () => viewer.zoomBy(2));
$('reset').addEventListener('click', () => {
  viewer.setState({ cx: '-0.5', cy: '0', radius: 1.5 });
  viewer.autoIter = true; $('autoIter').checked = true;
});

// Iterations: slider for quick scrubbing, number field for precise/large values.
// Both commit on `change` (slider release / Enter / blur); `input` just mirrors
// the live value to the sibling control so they always agree, without rendering.
function commitIter(v) {
  if (!isFinite(v) || v < 1) return;
  viewer.setMaxIter(Math.round(v)); // sets autoIter=false + re-renders
  $('autoIter').checked = false;
}
$('iter').addEventListener('input', (e) => { $('iterNum').value = e.target.value; });
$('iter').addEventListener('change', (e) => commitIter(+e.target.value));
$('iterNum').addEventListener('input', (e) => {
  const v = +e.target.value;
  if (isFinite(v) && v > 0) $('iter').value = Math.min(+$('iter').max, Math.max(+$('iter').min, v));
});
$('iterNum').addEventListener('change', (e) => commitIter(+e.target.value));
$('autoIter').addEventListener('change', (e) => viewer.setAutoIter(e.target.checked));
$('useGpu').addEventListener('change', (e) => viewer.setUseGpu(e.target.checked));
$('showGlitch').addEventListener('change', (e) => viewer.setShowGlitches(e.target.checked));
$('forceHQ').addEventListener('change', (e) => viewer.setForceHighQuality(e.target.checked));
$('lowPower').addEventListener('change', (e) => { viewer.setLowPower(e.target.checked); markLowPowerManual(); });
$('series').addEventListener('change', (e) => viewer.setSeries(e.target.checked));
$('ss').addEventListener('change', (e) => { viewer.setSupersample(+e.target.value); scheduleHash(); });
// Resolution (render pixels): full/half/third of the true canvas resolution.
// URL-persisted at Danielle's request (Spawn 34) — like ss, it rides in the hash so a
// bookmarked view reproduces its full look/perf; the panel select keeps it visible.
$('resScale').addEventListener('change', (e) => { viewer.setResScale(+e.target.value); scheduleHash(); });

$('palette').addEventListener('change', (e) => { viewer.setPalette({ paletteId: e.target.value }); scheduleHash(); });
$('cycle').addEventListener('input', (e) => { $('cycleVal').textContent = e.target.value; viewer.setPalette({ cycle: +e.target.value }); scheduleHash(); });
$('shift').addEventListener('input', (e) => { viewer.setPalette({ shift: +e.target.value }); scheduleHash(); });

$('goto').addEventListener('click', () => {
  const re = $('reIn').value.trim(), im = $('imIn').value.trim();
  let r = parseFloat($('radIn').value.trim());
  if (!isFinite(r) || r <= 0) r = viewer.radius;
  viewer.setState({ cx: re, cy: im, radius: r });
  $('panel').classList.remove('open');
});
$('copyLink').addEventListener('click', async () => {
  writeHash();
  try { await navigator.clipboard.writeText(location.href); $('status').textContent = 'link copied'; }
  catch { $('status').textContent = location.href; }
});
$('save').addEventListener('click', () => {
  const a = document.createElement('a');
  a.download = `mandelbrot_2e${viewer.zoomLevel().toFixed(0)}.png`;
  a.href = canvas.toDataURL('image/png');
  a.click();
});

document.querySelectorAll('.place').forEach((b) => b.addEventListener('click', () => {
  viewer.setState({ cx: b.dataset.cx, cy: b.dataset.cy, radius: parseFloat(b.dataset.r) });
  $('panel').classList.remove('open');
}));

// ---------- low-power / battery auto-detect ----------
// A manual toggle is authoritative: once the user touches the Low-power checkbox we stop
// auto-managing it (so we never fight their choice). Until then, if the Battery Status API
// is available, auto-enable low-power while the device is discharging and low, and lift it
// when charging / recovered. The API is absent in many browsers (incl. headless Chromium and
// Firefox) — it's a progressive enhancement; the manual toggle is the real control.
let lowPowerManual = false;
function markLowPowerManual() { lowPowerManual = true; }
const LOW_BATTERY_LEVEL = 0.2;
async function initBatteryDetect() {
  if (typeof navigator === 'undefined' || !navigator.getBattery) return;
  let battery;
  try { battery = await navigator.getBattery(); } catch { return; }
  const apply = () => {
    if (lowPowerManual) return;                    // user took over; don't auto-manage
    const want = !battery.charging && battery.level <= LOW_BATTERY_LEVEL;
    if (want !== viewer.lowPower) { viewer.setLowPower(want); syncControls(); }
  };
  battery.addEventListener('levelchange', apply);
  battery.addEventListener('chargingchange', apply);
  apply();
}

// ---------- keyboard navigation (desktop) ----------
// Arrow keys pan, +/- zoom, f fullscreen, ? help. Discrete (auto-repeat ignored) so a
// held key can't fly the preview snapshot off into black; each press settles to a sharp
// render via the same machinery as the wheel/buttons. Never hijacks a focused form field
// (so typing into the coord/iteration inputs works) or a browser shortcut (ctrl/meta/alt).
const PAN_STEP = 0.18; // fraction of the viewport panned per arrow press
window.addEventListener('keydown', (e) => {
  const tag = e.target && e.target.tagName;
  if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
  if (e.ctrlKey || e.metaKey || e.altKey) return;
  let handled = true;
  switch (e.key) {
    case 'ArrowLeft':  if (!e.repeat) viewer.panByPreview(+PAN_STEP * viewer.backingW, 0); break;
    case 'ArrowRight': if (!e.repeat) viewer.panByPreview(-PAN_STEP * viewer.backingW, 0); break;
    case 'ArrowUp':    if (!e.repeat) viewer.panByPreview(0, +PAN_STEP * viewer.backingH); break;
    case 'ArrowDown':  if (!e.repeat) viewer.panByPreview(0, -PAN_STEP * viewer.backingH); break;
    case '+': case '=': if (!e.repeat) viewer.zoomBy(0.5); break;
    case '-': case '_': if (!e.repeat) viewer.zoomBy(2); break;
    case 'f': case 'F': if (!e.repeat) toggleFullscreen(); break;
    case '?':           if (!e.repeat) showHint(); break;
    default: handled = false;
  }
  if (handled) e.preventDefault();
});

// ---------- fullscreen ----------
function toggleFullscreen() {
  try {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen?.();
    else document.exitFullscreen?.();
  } catch { /* fullscreen may be blocked (no user gesture / unsupported) — ignore */ }
}
$('fullscreen').addEventListener('click', toggleFullscreen);

// ---------- first-run gesture hint ----------
// Shown once per device (localStorage), re-openable via the “?” button or the “?” key.
// The overlay is pointer-events:none, so the first tap/drag/scroll/key both dismisses it
// AND performs its action — we listen in the capture phase and never consume the event.
const HINT_KEY = 'mb_hintSeen';
function showHint() { $('hint').classList.add('show'); }
function hideHint(persist = true) {
  const el = $('hint');
  if (!el.classList.contains('show')) return;
  el.classList.remove('show');
  if (persist) { try { localStorage.setItem(HINT_KEY, '1'); } catch { /* private mode */ } }
}
['pointerdown', 'keydown', 'wheel'].forEach((ev) =>
  window.addEventListener(ev, () => hideHint(), { capture: true, passive: true }));
$('help').addEventListener('click', showHint);
try { if (!localStorage.getItem(HINT_KEY)) showHint(); }
catch { showHint(); }   // storage blocked: still show it (just won't persist the dismissal)

// ---------- boot ----------
syncControls();
setIterUI(viewer.maxIter);
let resizeTimer = 0;
window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => viewer.resize(), 150); });
// initial size + render (after layout), then apply any bookmarked hash
requestAnimationFrame(() => {
  viewer.resize();   // sizes the canvas and renders the default view
  readHash();        // if a deep-zoom link is present, override and re-render
  initBatteryDetect(); // progressive: auto-enable low-power on a low/discharging battery
});
window.addEventListener('hashchange', () => readHash());
