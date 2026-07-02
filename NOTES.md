# NOTES — architecture, math, decisions

Read this first. It is how each spawn talks to the next.

## The core idea (perturbation theory)

Mandelbrot iteration: z_{n+1} = z_n^2 + c, escapes when |z|>2.

Naive double precision dies around zoom 2^50 (53-bit mantissa). To reach 2^400 we
use **perturbation theory**:

- Pick a reference point C (the view center) and compute its orbit Z_n in HIGH
  precision (BigInt fixed-point). Z_n values are O(1) magnitude.
- For a nearby pixel c = C + dc (dc tiny), write its orbit as z_n = Z_n + dz_n.
  Subtracting the reference iteration gives the **delta iteration**:

      dz_{n+1} = 2 * Z_n * dz_n + dz_n^2 + dc

- This delta iteration runs in DOUBLE precision. Why doubles suffice for 2^400:
  double's *exponent* reaches 2^-1022, so dc ~ 2^-400 and dz ~ 2^-400 are stored
  with full 53-bit *relative* precision. We only lose precision near 2^-1000 zoom.
  So **doubles are correct to ~2^1000 zoom; 2^400 is comfortably inside.**
  Only the *reference center coordinate* needs high precision (it has ~400
  significant bits); every per-pixel quantity (dc, dz) is a normal double.

## Glitch handling — Zhuoran rebasing (primary)

Single-reference perturbation glitches when the true orbit point z_n = Z_n + dz_n
is much smaller than the reference Z_n (catastrophic cancellation: dz loses
meaning). The robust modern fix is **rebasing** (Zhuoran, fractalforums 2021):

  track reference index m and dz; the true value is z = Z_m + dz.
  each step: dz = 2*Z_m*dz + dz^2 + dc;  m += 1
  let z = Z_m + dz  (true orbit value)
  REBASE: if |z| < |dz|  (equivalently |Z_m + dz| < |dz|), then
      dz = z;  m = 0
  escape test uses |z| (the TRUE value), not |dz|.

Rebasing with Z_0 = 0 means "restart the delta against the beginning of the
reference orbit" using the true value as the new delta — glitch-free with ONE
reference. We also keep a Pauldelbrot-style check (|z|^2 < 1e-6 * |Z_m|^2) as an
independent diagnostic / test assertion, but rebasing is what we render with.

Reference must be long enough: if a pixel needs iteration k but the reference
escaped/ended at m<k, extend the reference or, after rebasing, m wraps to 0 and we
keep going up to that pixel's maxIter. We compute the reference to maxIter and, if
it escapes early, we still keep Z_n for n up to escape (Z stays defined; for the
classic "Z_0=0" reference of the center, if the *center* escapes the location is
outside the set anyway).

## Validation strategy (against ground truth)

`naive.js` is the ORACLE: plain-double escape-time Mandelbrot. At shallow zoom
(<= ~2^40) it is exact. Every higher layer is validated against it:

1. bignum complex mul/sqr vs known products & vs JS Number at low precision.
2. reference orbit (high precision) vs naive double orbit at shallow zoom -> equal.
3. **perturbation per-pixel escape counts == naive per-pixel counts** over a grid
   at shallow zoom. This is the make-or-break test (M4). If perturbation matches
   the oracle exactly where the oracle is valid, the engine is correct; we then
   trust it where the oracle can't reach (deep zoom).
4. Deep-zoom smoke tests assert "no glitch pixels" via the Pauldelbrot diagnostic
   and visual/structural checks.

## High precision: fixed-point BigInt

A real value v is stored as a BigInt `m` with v = m / 2^PREC (two's-complement
sign via BigInt sign). PREC ~ zoomBits + 64 guard bits.
- add/sub: BigInt +/- .
- mul: (a*b) then arithmetic shift right by PREC (with round-to-nearest).
- We only need: complex add, complex sqr (for z^2), complex mul, magnitude
  compare vs 4 (escape), and toDouble for export.
Reference orbit needs ~maxIter such iterations; done in a Web Worker with
progress. PREC chosen from view radius: PREC = ceil(-log2(radius)) + 64.

## Why CPU-first, GPU-later

- CPU double perturbation in Web Workers is *correct* to 2^400 and easy to
  validate against the oracle. It is the foundation.
- GPU float32 CANNOT represent 2^-400 deltas (min normal ~2^-126). Deep-zoom GPU
  needs scaled deltas + per-pixel rescale ("floatexp") or double-emulation — more
  complex and harder to validate. So GPU is M6, an accelerator, not the base.
- Mobile: progressive low-res-first + tiling + workers keeps it interactive even
  on CPU. GPU added later for shallow/medium zoom speed.

## Module layout
- src/math/naive.js      — oracle escape-time (doubles)         [pure, Node+browser]
- src/math/bignum.js     — fixed-point BigInt real/complex      [pure]
- src/math/reference.js  — high-precision reference orbit        [pure]
- src/math/perturb.js    — delta iteration + rebasing            [pure]
- src/math/palette.js    — smooth coloring                       [pure]
- src/worker.js          — pool worker: 'computeRef' (once) + 'render' bands
- src/math/render.js     — reference auto-selection + renderImage + dispatch
- src/viewer.js          — canvas, view state, touch/zoom, render orchestration
- src/main.js            — UI wiring (sliders, readouts, URL hash)
- index.html, styles.css
- test/unit/*.mjs        — Node test runner (node --test)
- test/e2e/*.spec.mjs    — Playwright
- tools/serve.mjs        — static server for dev + e2e

Pure math modules must import-cleanly in BOTH Node (tests) and the browser
(workers), so: no DOM, no top-level await, plain ESM exports.

## View-state representation
- center: {x: BigInt, y: BigInt, prec: PREC}  (high precision)
- radius (half-height of view in complex plane): a Number (double) — fine since
  >= ~2^-1000. zoom "level" displayed as log2(baseRadius/radius).
- maxIter: Number, auto-scaled with depth.
- For shallow zoom the high-precision center still works (PREC just small).

## Zoom / gesture interaction model (viewer.js) — read before touching gestures

The render is expensive (workers / GPU draw); the rule is **never re-render while
the user is still moving the view**. Instead we transform the last good frame.

- `this.T = {a, e, f}` is a preview transform in backing-pixel space:
  displayed = a*orig + (e,f). `_applyPreview()` draws the snapshot `this.stable`
  under that transform (smooth-scaled). While `T` is set, a preview is "active".
- `_beginPreview()` (idempotent — acts only when `T` is null) is the single entry
  to a gesture: it snapshots the *current visible frame* into `this.stable`
  (so even a half-finished render is what we scale), sets `T = identity`, and
  **cancels any in-flight render** (`gen++` to invalidate stale worker messages +
  `_terminatePool()` + `rendering=false`).
- Two ways a preview commits to a real render:
  - touch drag/pinch: `pointerup` with no pointers left → `_endGesture()` (immediate).
  - momentum-free sources (wheel / zoom buttons / click-to-zoom) have no "up", so
    `_scheduleSettle()` debounces `_endGesture()` ~220ms after the last motion.
  `_endGesture()` folds `T` into the HP view state (new center + radius/=T.a),
  clears `T`, and calls `render()`. `render()` also `_clearSettle()`s.
- `zoomBy(factor, px, py)` is the shared zoom primitive (radius *= factor about a
  backing pixel, **kept fixed**): `_beginPreview()` → compose scale `s=1/factor`
  about (px,py) into `T` → `_applyPreview()` → `_scheduleSettle()`. Wheel + the +/-
  buttons call it. `zoomAt()` (mutates view state directly, no preview) still
  exists for tests and programmatic jumps.
- `clickZoom(px, py, factor)` is the **click-to-zoom** primitive (added Spawn 5):
  like zoomBy but it RECENTERS — the clicked complex point becomes the new view
  center (moves to screen middle) AND radius *= factor. The displayed-space op is
  "scale by s=1/factor about (px,py), then translate (px,py)→center"; composed
  G∘T so it stacks with any in-progress preview, and `_endGesture` folds it in
  (new center = complex point under the click, new radius = radius*factor — verified
  by the e2e). A plain tap / left-click (down→up, no drag) calls it with factor=0.5
  (zoom in); shift / ctrl / right-click use factor=2 (zoom out). Right-click's
  context menu is suppressed on the canvas. (There is no more double-tap handler —
  a single tap now zooms, so two taps just zoom twice.)
- IMPORTANT subtlety: a pan/pinch preview begins **lazily on the first real move**,
  NOT on pointerdown. A plain tap (down→up, <1px move) must not cancel a running
  render — instead, on `pointerup` with `T` still null, it is treated as a click and
  routed to `clickZoom`. (The old `_beginGesture()` cancelled on every pointerdown,
  which also made a deep double-tap terminate its own freshly-kicked render.)

## Decisions / dead-ends log

### Engine dispatch by depth (IMPORTANT, validated empirically)
Double-precision perturbation stores the reference orbit as doubles (Z_n ~ O(1),
53-bit). At SHALLOW zoom (radius ~0.02) dc is large, so the per-pixel delta dz
grows to O(1) within a few iterations and the whole computation is only ~53-bit —
i.e. NO better than naive there. On ultra-sensitive boundary pixels (near
Misiurewicz points, high escape counts) double perturbation is then off by tens
of iterations and can even misclassify inside/outside. Measured at radius 0.02,
maxIter 1500: ~5/6400 pixels wrong, one inside-point reported as escaped.

At DEEP zoom dc ~ 2^-N is tiny, dz stays tiny far longer, so effective precision
is much higher and perturbation matches the BigInt-exact oracle to +/-1 (verified
by tests B/C/E at 2^45, 2^120, and a rebasing case). This is the regime
perturbation exists for.

DECISION: render NAIVE doubles for radius >= ~2^-40 (fast, GPU-able, standard,
as accurate as any double method there) and PERTURBATION for radius < ~2^-40.
They are NOT required to agree bit-for-bit on ill-conditioned shallow boundary
pixels — the BigInt orbit is the only true oracle there, and any 53-bit method
(naive OR perturbation) is noisy on that measure-zero set. See render.js
engineForRadius().

### BigInt reference precision / guard bits
precForRadius(radius, guard=64) sets prec = zoomBits + guard. For the EXACT
single-point BigInt oracle near sensitive boundary points, guard=64 was not
always enough (a point at radius 0.02 needed prec ~150 to converge its exact
count). For the perturbation *reference* (stored as double anyway) guard=64 is
fine in the deep regime (B/C/E pass). The exact-oracle tests use a generous prec.

### Validate against BigInt, not naive
naive is only an oracle where double is reliable (low/medium escape counts, not
ultra-sensitive boundary). The authoritative oracle is escapeBigInt (full BigInt,
high prec). All strict correctness assertions compare perturbation to BigInt.

## Running a browser in THIS sandbox (NixOS) — load-bearing, read before e2e
The Playwright-downloaded chromium (build 1228 in ~/.cache/ms-playwright) CANNOT
run here: it's a generic ELF whose interpreter /lib64/ld-linux-x86-64.so.2 does
not exist on NixOS -> spawn ENOENT. Fix: use a nix-store chromium instead.
- playwright.config.mjs resolves /nix/store/*-chromium-*/bin/chromium at load.
- Use the NEWEST build (148): older ones (143) crash with SIGTRAP because this
  sandbox has NO /sys/devices/system/cpu (Chromium reads it at startup). 148
  tolerates it; 143 does not.
- Must use NEW headless: pass '--headless=new' (appended after Playwright's own
  '--headless', last-wins). Old headless crashes. Also: --no-sandbox,
  --disable-dev-shm-usage, --disable-gpu, --enable-unsafe-swiftshader.
- Port 8080 is often busy on this host; default is now 8137. Override with PORT=.
- Verify a browser manually:  /nix/store/<...>-chromium-148*/bin/chromium \
    --headless=new --no-sandbox --remote-debugging-port=9333 about:blank &
  then curl http://127.0.0.1:9333/json/version
- Playwright 1.61.0 is pinned (its bundled build == cached 1228). Using nix
  chromium 148 over CDP 1.3 works fine despite the version skew.

## How to run
- `npm install`            (installs Playwright 1.61.0)
- `npm test`               (31 Node unit tests — the correctness oracle suite)
- `npm run serve`          (static server on :8137, COOP/COEP set)
- `npm run e2e`            (Playwright suite, mobile + desktop projects; SwiftShader)
- ANY tool/test takes `GPU=1` (real GPU, Vulkan) or `GPU=gl` (native GLES) — see the
  "⚠️ REAL-GPU TESTING" section. npm shortcuts: `validate:gpu:real`, `bench:gpu:real`,
  `e2e:gpu`, `probe:gpu` (flag-combo sweep), `probe:df64` (df64-precision-on-GPU gate),
  `probe:ftz`, `probe:xbackend` (GPU-df64 vs SwiftShader-df64).
- `node tools/probe-gpu-real.mjs` (which Chromium flags actually get the real GPU)
- `node tools/probe-df64.mjs` (is df64 intact on the backend, or collapsed to f32? + the
                            barrier A/B/C comparison that found the working XOR barrier)
- `npm run probe:barrier` (Spawn 20: the df64 barrier-PLACEMENT ladder on the real GPU — finds the
                            MINIMAL ob() set that stays INTACT, the regression gate for any DF64_LIB
                            edit / new target GPU; `npm run bench:barrier` = TIME=1 per-op speed A/B)
- `node tools/probe-df64-real.mjs` (test the SHIPPING DF64_LIB: single ds_mul, a 40-iter
                            update, add-tiny-to-O(1) — all intact on GPU in isolation)
- `npm run bench:sqr` (Spawn 27: full-shader A/B of the dedicated ds_sqr (p.sqrOn, OPT-IN) vs the
                            shipping ds_mul(a,a) — measured 1.00× RTX 3090 / 0.95–0.98× SwiftShader;
                            contention-robust min-of-min protocol. + sn diff report. BITS=/ROUNDS=/REPS=)
- `node tools/bench-ref.mjs` (Spawn 29: pure-Node timing of the BigInt reference build + escapeBigInt —
                            the A/B for any reference/bignum math change. BITS=/REPS=)
- `GPU=1 node tools/probe-refcache.mjs` (Spawn 30: THE gate for the reference/SA-reuse cache — 6 live
                            zoom ticks warm-vs-cold: must be 6/6 HITs, faster, and <2% bulk diff.
                            BITS=/TICKS=)
- `GPU=1 node tools/probe-strips.mjs` (Spawn 28: strip-orchestration overhead on the LIVE viewer —
                            A/Bs production vs ×8 vs single-strip by monkey-patching _stripRows;
                            the gate for any strip-budget change. OFFSCREEN=0 = main-thread path;
                            BITS=/ROUNDS=. Deltas are clean: the ref build repeats identically.)
- `npm run bench:prefetch` (Spawn 27: A/B of the software-pipelined getZ prefetch (SHIPPED default in
                            the RESCALED engine; opt-in df64/fe where it measured ≤1.0×) vs the old
                            immediate-use placement — 1.01–1.08× RTX 3090 + ~2.0× SwiftShader on rs;
                            must report sn diff 0. Robust protocol; BITS=/ENGINE=df64|fe|rs/ROUNDS=/REPS=)
- `node tools/probe-xbackend.mjs` (render df64 on SwiftShader AND GPU, compare — shows the
                            full-shader divergence the per-op barrier doesn't fix yet)
- `node tools/probe-ftz.mjs` (does the backend flush subnormals? both do — ruled out as cause)
- `npm run validate:gpu`   (GPU-vs-oracle regression across depths — run after any
                            shader change; mismatch numbers are the correctness gate.
                            Covers naive/df64/floatexp AND rescaled. SwiftShader by default;
                            `validate:gpu:real` runs it on the real GPU — currently 12 FAIL
                            there, all deep df64/fe/rs, due to the reassociation bug.)
- `node tools/bench-gpu.mjs` (time floatexp vs rescaled perturb on this host's GL)
- `node tools/crosscheck-skip.mjs` (prove the perturb fast-skip is bit-identical:
                            renders each view with the skip on AND off → 0-diff)
- `node tools/crosscheck-tiled.mjs` (prove the strip-tiled deep render is bit-identical
                            to a single full-frame draw → 0-diff; the gate for the 2^218 fix)
- `npm run crosscheck:bla` (Spawn 17: BLA correctness + work-reduction vs the no-BLA oracle on the
                            deep boundary coordinate; sweeps blaEps. BITS=120,271,400 EPS=28,30,32)
- `npm run arbiter:bla` (Spawn 17: BigInt-classify the BLA-vs-oracle drift — ill-conditioned vs a
                            real bug. BITS=400 EPS=30. Showed BLA is off ≤19 on well-conditioned
                            near-maxIter pixels = bounded truncation, not a missed escape.)
- `npm run probe:samargin` (Spawn 22: sweep the SA safety margin at the deep band, GPU-SA-vs-no-SA
                            bit-exact gate on the real GPU — showed 0.05→0.02→0 stays mism 0 with
                            identical drift; the basis for the depth-adaptive 0.02 deep margin)
- `npm run bench:samargin` (Spawn 22: wall-clock A/B of the SA margin on the real GPU —
                            1.12×/1.19×/1.18×/1.40× @2^-120/-218/-271/-400 for 0.05→0.02)
- `npm run probe:sabox` (Spawn 22: SA skip vs dc-box scale — showed region-adaptive SA is DEAD,
                            a quarter-box buys <1.6% more skip; the truncation cliff is too sharp)
- `npm run probe:interior` (Spawn 22: where post-SA work goes — 61–90% interior @deep, 65–91% of
                            the work; the interior-detection potential measurement, deferred)
- `npm run bench:lean` (Spawn 22: FULL vs LEAN rescaled kernel on the real GPU — LEAN drops the
                            never-taken BLA+df64esc blocks, bit-identical; measured NEUTRAL-to-slower,
                            occupancy isn't the bottleneck on the RTX 3090; lean kept opt-in for mobile)
- `node tools/probe-rescaled.mjs` (rescaled engine: vs the CPU oracle + vs floatexp)
- `node tools/probe-deep218.mjs` (measure deep chaotic GPU-vs-oracle mism at 2^-90..-271
                            on a real deep coordinate — showed it's 0.000%, i.e. the 2^218
                            wall was the watchdog, NOT precision)
- `TARGET=1010 node tools/gen-deep-coord.mjs` (Spawn 10/25: descend a filament in BigInt to make a
                            GENUINE deep boundary coordinate; ITERCAP now auto-scales with TARGET so the
                            descent stays on the boundary — at the old fixed 140k it fell into the set ~2^-560)
- `npm run probe:wall` (Spawn 25: double perturbation vs the BigInt-EXACT oracle at the pixel's TRUE
                            BigInt offset — finds the precision WALL. 0 escape-count mism FLAT 700→1010,
                            dc/step subnormal ~2^-1020. GRID=/BITS= override. The gate for MIN_RADIUS.)
- `npm run probe:deep500` (=GPU=1; Spawn 10/25: rescaled+fe vs CPU oracle on the deep boundary
                            coordinate at autoMaxIter — gate for the deep floor; 0.000% mism FLAT 600→1010.
                            Set BITS=…)
- `GPU=1 node tools/shoot-deep500.mjs` (Spawn 10: drive the real viewer to the deep
                            coordinate at zoom 2^400/2^500 — proves gpu-perturb-fe runs +
                            the ss cap; screenshots/deep{400,500}.png. BITS=…, SS=…)
- `node tools/shoot-deep.mjs` (render the actual viewer at the deep ultra coordinate)
- `node tools/shoot.mjs`   (capture screenshots/ — needs server running)
- `npm run probe:ctxloss`  (Spawn 11: does WEBGL_lose_context dispatch lost+restored on this
                            backend? proves the tick-between-lose-and-restore requirement)
- `npm run probe:recover`  (Spawn 11: drive the REAL viewer through a WebGL context loss/restore
                            cycle — CPU while lost, GPU on restore, shallow+deep+give-up. GPU=1 too)
- `npm run crosscheck:offscreen` (Spawn 24: render the SAME view via the offscreen worker AND the
                            on-main-thread renderer → diff the full canvas; must be 0.000%/Δ0. `:real` = GPU=1)
- `npm run bench:offscreen` (Spawn 24: main-thread JANK A/B — Long-Task total + rAF heartbeat max-gap,
                            offscreen worker vs main-thread, during a deep render. BITS=/VP= override. `:real`)
- `npm run probe:recover:offscreen` (Spawn 24: the offscreen analogue of probe:recover — context loss
                            forwarded worker→main → CPU detour → GPU worker on restore. `:real` = GPU=1)

## ✅✅ REFERENCE + SA REUSE ACROSS VIEWS — deep zoom ticks ~5×: ~510ms → ~100ms settle (Spawn 30) — READ THIS FIRST

**Ask (Danielle, same live session: continue optimizing). The recorded "big interactive win": every
zoom-tick settle rebuilt the BigInt reference orbit AND — the flame-graph surprise — recomputed the SA
coefficients, which are now the BIGGEST serial cost (computeSeries = 466–607ms in Node at 2^-350…-700,
larger than the ref build!). OUTCOME: the viewer caches the reference orbit (+ BigInt tail) and the SA
coeffs, reuses both across same-neighborhood views, and EXTENDS the orbit incrementally when maxIter
grows. Measured (probe-refcache, real GPU, 6 zoom ticks at the deep coordinate from 2^-350):
warm ticks **135/105/107/99/455/121ms vs cold ~504–539ms — 3.07× overall, ~5× on reuse ticks** —
successive deep zooms are now effectively INSTANT. Final-frame bulk diff vs cold: **0.00%**. 44 unit
(3 new bit-exactness tests), full desktop e2e 38/38, probe-wall spot re-check clean.**

### Why reuse is valid (the math, so nobody "fixes" it away)
- **Reference**: Zhuoran rebasing makes ANY nearby point a valid reference — the pipeline already
  supports off-center references (chooseReference relocation uses the same offX/refOff plumbing). So
  the cache tolerates the view center DRIFTING from the cached point by ≤ REF_MAX_DRIFT (8) view radii
  (wheel jiggle, small pans, click-recenters at depth).
- **Orbit extension**: computeReference and extendReference run the SAME iterateOrbit core; resuming
  from the exact BigInt tail reproduces the identical mulShift sequence ⇒ a cached orbit + extension
  is BIT-IDENTICAL to an uninterrupted full build (test/unit/refextend.test.mjs — including chained
  extensions and escape-mid-extension). A zoom-in tick needs only ~250·Δbits NEW iterations (~ms).
- **SA coefficients**: computeSeries output is a property of the reference orbit + the dc box it was
  validated against + its own invR normalization (u = dc·sa.invR is self-consistent at ANY current
  radius). A zoom-in's dc box lies INSIDE the validated box ⇒ the cached coeffs remain valid with a
  conservatively-frozen skip. The frozen skip goes stale as maxIter grows (fresh coeffs would skip
  further), so it is REFRESHED in the worker after SA_STALE_ITERS (1000) of maxIter growth — visible
  in the probe as the one 455ms tick per ~4; staleness loss bounded to ~1000 anchor iterations.
- **Precision headroom**: fresh builds run at prec + REF_HEADROOM (64) with the center BigInts shifted
  to match (~+15% one-time build cost), so zoom-ins keep hitting until zoomBits outgrows the headroom
  (~64 doublings), then one fresh (re-headroomed) build.
- Realization note: a warm render uses a different-but-equally-valid reference realization than a cold
  one (higher prec, possibly drifted point, conservative skip) — per-pixel low-bit sn may differ on
  ill-conditioned boundary pixels, the SAME accepted class as reference relocation itself. Gate:
  probe-refcache's warm-vs-cold bulk grid diff (0.00% measured). Cold loads (bookmarks/shared URLs)
  never see the cache ⇒ recipients render deterministically.

### The moving parts
- `viewer.js`: `_refCache` {ref point (BigInt), prec, orbit arrays, BigInt tail, escaped, sa, saBox,
  saAtIter} stored on refReady (GPU perturb path only — the arrays are SLICED before the offscreen
  dispatch transfers them away). `_refCacheUsable()` → 'ready' | 'extend' | null (prec headroom, drift
  bound, len/escaped logic). `_cachedSaFits()` (box-fit + staleness + skip<min(len,maxIter)).
  `_gpuPerturbFromCache()` synthesizes params exactly like worker computeRef (drifted-reference offX)
  and dispatches straight to the GPU worker — no pool, no BigInt, no SA scan on the good ticks.
  computeSeries NEVER runs on the main thread (it's 100s of ms — the OffscreenCanvas responsiveness
  discipline): missing/stale SA routes to the pool worker ('computeSA', or bundled into 'extendRef'
  as withSA — the worker merges old+segment arrays, recomputes SA, transfers everything back).
- `worker.js`: 'extendRef' (BigInt-tail resume; optional withSA bundling), 'computeSA'; refReady now
  carries refCache material {ref point, prec, escaped, tail, saBox}.
- `reference.js`: shared `iterateOrbit` core; computeReference returns tailX/tailY; new
  `extendReference(center, tail, fromN, maxIter)` (segment arrays + absolute len).
- Escaped-mid-extension: the extended (escaped) orbit still renders THIS frame correctly (end-of-ref
  rebase); the NEXT deeper render misses (escaped + short) and rebuilds fresh WITH relocation —
  matching chooseReference's normal behavior. CPU-pool path untouched (cache is GPU-path only).

### Measured + gates
probe-refcache (NEW; the gate for any cache change): 6/6 hits, warm ~1000ms vs cold ~3100ms total,
bulk diff 0.00%, PASS — in BOTH a centered and a DRIFTED zoom sequence (see the bug below). 44 unit
(41 + 3 refextend bit-exactness). Desktop e2e 38/38 (viewer 26 — the zoom-button/iteration paths
exercise hit+extend live; gpu 10; perf 2) + mobile gpu 10/10. probe-wall spot re-check
(BITS=700,1010) after the iterateOrbit refactor: 0 mismatch. bench-ref unchanged (the refactor is
op-identical; the store callback is JIT-inlined).

### ✅ RUNTIME GPU PRECISION SELF-TEST (Spawn 33 — Danielle is on a PIXEL 8, i.e. Mali) — warning-only by her choice
CONTEXT SHIFT: Danielle uses the viewer on a Pixel 8 (Mali/Tensor G3) — a GPU class NEVER validated
here (all shader correctness is empirical-per-GPU: NVIDIA + SwiftShader only; the df64 barrier NOTES
explicitly flag portability). The Spawn-8 backlog item is finally built: after the FIRST deep GPU
frame of a session, the viewer samples an 8×8 grid of the rendered escape data (targeted 1×1
readbacks — renderer.sampleSn / worker 'sampleSn' / client.sampleSn) and replays those pixels through
escapePerturb in a throwaway pool worker with the EXACT same reference/SA/geometry (`_lastDeepParams`
+ the Spawn-30 ref cache make this free). Comparison = validate.js faithfulness mode (same SA, |Δn|≤2,
inside-flips are mismatches). Verdict needs ≥6 sampled ESCAPERS (interior-heavy views are inconclusive
→ silently retried on the next deep frame — the 5×5 grid was one escaper short on the test view, hence
8×8); mismatch >25% of escapers ⇒ FAIL (healthy GPUs measure 0–1%; the known NVIDIA df64-collapse class
measured 22–99%). ON FAIL (Danielle's explicit choice — WARNING-ONLY, NO forced CPU fallback): rendering
STAYS on the GPU; a persistent status warning appears ("⚠ GPU precision self-test failed (N%…) — toggle
'GPU acceleration' off to compare") + a "· ⚠ GPU self-test failed" suffix on every later done-line. The
existing GPU toggle is the manual fallback. Cost: ~ms, once per session; skipped under the glitch
overlay. Test hook `viewer.__forceVerifyFail`; both paths verified headless (healthy → silent verified,
GPU stays; injected fail → warning + GPU stays). Gates: crosscheck:offscreen pixel-identical, 44 unit,
gpu+viewer e2e green. IF HER PIXEL 8 SHOWS THE WARNING: the Mali driver breaks the df64 barrier → the
next work item is a Mali-specific barrier variant (probe:barrier methodology, on-device).

### 🐛→✅ GESTURES NEVER CANCELLED THE GPU WORKER (Spawn 33, Danielle: "can't interrupt a render; UI laggy until it finishes")
Since the OffscreenCanvas migration (Spawn 24), `_beginPreview()` — the universal gesture entry that is
SUPPOSED to cancel the in-flight render — bumped gen and terminated the CPU pool but NEVER sent the GPU
worker a `cancel` message (only the context-loss path did). Consequence: the main thread dropped the
stale strips (gen guard), but the WORKER kept rendering every remaining strip of the abandoned frame
at full GPU cost — competing with the preview compositing, any other GPU tenant, and (on non-gesture
supersedes like "Go", where the new plan only reaches the worker after a ref build) delaying the next
frame. Capped-res frames kept the stale tail short enough to hide for 9 spawns; Spawn 32's full-
resolution frames stretched it to user-visible "have to wait for it to finish". TWO omissions fixed:
- `_beginPreview()` and `render()` now post `gpuWorker.cancel(gen)` immediately after the gen bump.
- The worker's strip loop gains a MACROTASK YIELD after each ack (message-ordering subtlety: the ack
  that wakes the loop is processed BEFORE any cancel/render message that arrived after it — without
  the hop the loop synchronously submits one more full strip before ever seeing the cancellation).
Measured (tools/probe-interrupt.mjs — warm-cache, full-res 1400×900, interrupt mid-strips; the OLD arm
stubs client.cancel to a no-op): stale strips arriving AFTER the zoom: 8ms tail → ≤1ms (none). On an
IDLE 3090 the old tail was only ~1–2 strips, so zoom→settled barely moves there — the honest scoping:
the win scales with strip duration (slow/contended GPUs, full-res deep frames = exactly the reported
conditions), and it removes ALL wasted stale GPU work during gestures. Gates: crosscheck:offscreen
ALL PIXEL-IDENTICAL (the worker loop was touched), 44 unit, gpu+viewer e2e green. NOTE the remaining
DESIGNED latency: a settle render still waits the 220ms debounce + the frame itself; interruption =
instant preview + freed GPU, not instant sharp pixels.

### ✅ FULL-RESOLUTION RENDERING + Resolution setting (Spawn 32) — the ACTUAL "blocky detail" root cause
Danielle's third message nailed it: the render resolution was LOWER than the screen. `resize()` capped
the backing store at MAX_BACKING=1100 on the long edge (a mobile-era guard) — on any larger window the
frame upscales to the canvas, and with the deliberate `image-rendering: pixelated` (crisp, not blurry)
the upscale reads as BLOCKY fine detail. On a 1600×900 window that was 1100×619 → 2.1× fewer pixels
than the screen; worse at 1440p/4K/dpr2. THE FIX: the backing store now follows the TRUE canvas
resolution (css × dpr, dpr still capped at 2), guarded by a total-pixel budget (MAX_BACKING_PIXELS =
9e6 ≈ 4K fullscreen; canvas RGBA ≈ 36MB) instead of an edge cap; low-power mode keeps its stricter
caps. NEW `resScale` setting (panel select "Resolution: Full/Half/Third", `viewer.setResScale`) divides
the backing for speed — deliberately NOT URL-persisted (per-device choice; a shared link must never
carry a degraded-resolution trap, cf. the ss=1/i= findings below). Verified headless: 1600×900 window
→ backing 1600×900 (was 1100×619); Half → 800×450. e2e 38/38. COST: default desktop frames now compute
~1.3–4× more pixels (that's the point — it's the sharpness the screen can show); the Half/Third select
+ the existing ss/lowPower/depth caps are the speed valves; MAX_COMPUTE_PIXELS still bounds ss×backing.

### 🔍 THE "GLITCHY FINE DETAIL" INVESTIGATION (Spawn 31, Danielle's second report) — how it resolved
Report: "fine details slightly off at [a 2^-251 URL], zooming fixes them; happens even on hard refresh."
The coordinate is an EARLY-ESCAPING-REFERENCE location (refLen ~3.5–3.8k vs maxIter ~63k; SA skip lands
within ~200 of the reference end — a corner the standard non-escaping validation coordinate never
stresses), so it deserved the full ladder. FINDINGS (tools/probe-repro-url.mjs + probe-repro-gpu.mjs):
- The reference cache NEVER ENGAGES here (escaped ref + maxIter>len ⇒ deliberate MISS every tick) and
  warm-arrival == cold to 0.00% — cache exonerated.
- probe-wall AT HER COORDINATE (RE=/IM= env): CPU perturbation vs BigInt-exact = **0/144 mismatches**.
- GPU production path (rs+SA) vs CPU oracle at her exact view: **0.012%** (3px/25600, the normal
  ill-conditioned floor); fe 0.020%. (A df64-engine row reads 99.98% wrong — that engine is 140 octaves
  below its dispatch band there; production never uses it. Don't be spooked by it in the probe output.)
- Her URL carried **i=26100 with r≈2^-251** (auto would be 63150): the i= URL-load PINNED autoIter off
  (see the trap fix below). At THIS view it happens to be harmless (all escape counts < 26100 — measured)
  but the trap is real for deeper continuation.
- Her URL also carried **ss=1** (supersampling off): fine filaments ALIAS at pixel scale — visually
  "slightly off / glitchy fine detail", and zooming in "fixes" it because the structure stops being
  subpixel. This is the remaining (behaving-as-designed) explanation consistent with hard-refresh.
VERDICT: engine math verified exact at this location class (short/escaping reference included); the two
real issues were URL-state traps, one fixed (below), one user-visible-by-design (ss=1 in the URL).

### 🐛→✅ THE URL i= AUTO-ITER PIN TRAP (Spawn 31) — fixed in main.js readHash
writeHash records `i=` on EVERY url, and readHash pinned `autoIter=false` whenever i was present — so
loading ANY shared/bookmarked URL silently froze the iteration budget; zoom deeper from there and the
view under-iterates (fine detail reads as interior). That is how Danielle's session carried i=26100
(= auto for 2^-103) down to 2^-251. FIX: readHash pins manual ONLY when i differs from autoMaxIter(r)
by >2% (a deliberate manual choice); an i that merely echoes the auto value keeps auto ON. Verified
both semantics headless; the URL round-trip e2e (records i=auto) is unaffected. Deliberately-lowered
artistic bookmarks still pin, as before.

### 🐛→✅ THE DRIFT-SIGN BUG (Spawn 31, found by Danielle in live use) — read before touching offsets
As first shipped, `_refCacheUsable` stored refOff as VIEW − REF, but `_cacheDcBox` consumes it with
worker.js's convention (refOffX = REF − VIEW; offX = −r·aspect − refOffX). Effect: whenever the view
center DRIFTED from the cached reference — every real wheel zoom at a cursor, every click-to-zoom
recenter — the synthesized dc origin was wrong by 2×drift, rendering the location MIRRORED about the
cached reference ("zoomed somewhere different than where I clicked"), and the SA box check used the
same wrong offsets, letting the seed run outside its validated box (diverged seed → uniform one-hue
frames — "screen is just blue"). WHY THE ORIGINAL PROBE MISSED IT: its zoom ticks hit the EXACT canvas
center, where drift is 0.0 exactly — the one point where the sign cannot matter. THE FIX: one line
(refOff = c.cx − (this.cx << s), matching the worker). THE GUARD: probe-refcache now runs a second,
DRIFTED sequence (off-center zoomAt ticks, 0.05·W — small enough to keep content-rich filament in
view, since at large drift BOTH mirrored and true views go featureless-interior and a bulk diff is
blind); verified by deliberately re-introducing the flipped sign: the drifted diff reads **77.21%**
(FAIL) bugged vs **0.00%** (PASS) fixed. LESSON, recorded: any A/B guard for a geometric transform
must include an input where the transform is NON-DEGENERATE (drift ≠ 0), and should be proven able
to FAIL by running it against the bug it guards.

## ✅✅ FAST REFERENCE BUILD — BigInt orbit 2.5×, the deep frame's NEW dominant cost harvested (Spawn 29) — READ THIS FIRST

**Ask (Danielle, same live session: "more optimize?"). After the strip fix, the deep frame's dominant
cost was no longer the GPU (~180ms at 2^-400) but the serial CPU BigInt REFERENCE BUILD (~450ms of a
726ms frame). Three changes, all in src/math (NO shader/GPU change): the build is 2.54× faster
(303→119.5ms at 2^-400 in Node), the exact per-pixel oracle 1.55× (45→29ms/point), and the live deep
frame dropped 726→576ms @2^-400, 467→388ms @2^-271. Gates: probe-wall 0 mismatch FLAT 2^-700→2^-1028
(the BigInt-exact standard), validate:gpu + validate:gpu:real ALL PASS, 41 unit, e2e green.**

### The three changes (tools/bench-ref.mjs is the A/B)
1. **2-mult complex square** (`computeReference`): re = (bx+by)(bx−by), im = (2bx)·by — 2 BigInt
   mulShift/iter instead of 3 (bx², by², bx·by). (x+y)(x−y) rounds ONCE where x²−y² rounded twice —
   ≤1 ulp at 2^-prec per step, the same class as existing mulShift rounding (the orbit is a pseudo-
   orbit either way; correctness = per-pixel escape counts vs the BigInt-exact oracle, gated below).
   Alone: 1.13× (mults were only ~40% of the loop — see #3).
2. **escapeBigInt 5→3 mults/iter, BIT-EXACT**: it squared the new z for the escape test, then
   re-squared the same z next iteration for the update. Carrying zx²/zy² across iterations is the
   identical sequence of mulShift calls → identical values, 1.55× faster. (This is the per-pixel
   oracle probe-wall/arbiter grids call hundreds of times — those probes run proportionally faster.)
3. **toDouble fast path** (bignum.js) — the sleeper: the old path did `a.toString(2).length` (an
   O(prec) BINARY STRING per call) just to get the bit length, ×2 calls/iteration. For prec ≤ 1000,
   `Number(m)` rounds a BigInt correctly to nearest in one step (no string), then an exact 2^-prec
   scale — faster AND more accurate (one rounding vs truncate-to-60-bits-then-round). Overflow-guarded;
   prec > 1000 (zooms beyond ~2^-936) falls back to the general path unchanged. This was the biggest
   single win: 1.13× → 2.54× when stacked.

### The escape-test subtlety (semantics preserved exactly)
computeReference's escape test used the PREVIOUS z's squares (bx²+by² > 4 tested after storing z_n) —
those squares no longer exist under the 2-mult form. The decision now runs on the previous iteration's
double |z|² (relative error ~2^-51) with an EXACT BigInt re-check only within ±1e-12 of the threshold
(a band the double error cannot bridge; ~never hit). Identical escape length on every tested orbit.

### Validation (the full ladder, all green)
- **probe-wall** (the authoritative gate): renderer vs escapeBigInt at TRUE BigInt offsets — **0
  escape-count mismatch FLAT 2^-700→2^-1028**, dcRelErr 3.3e-16 unchanged. This covers BOTH toDouble
  paths (prec 764–1092 straddles the ≤1000 fast path and the >1000 fallback) and the new 2-mult orbit.
  ON THE prec-1001 BOUNDARY (companion-flagged reviewer question): the fast path rounds correctly-to-
  nearest, the fallback truncates-to-60-bits-then-rounds — realizations can differ by 1 ulp ACROSS the
  boundary, but every render is self-consistent (one prec per render; nothing compares doubles across
  prec regimes), and the sweep's 0-mismatch on both sides of 2^-936 is the empirical seal. Do NOT
  "fix" the fallback to match — it must handle prec>1074 magnitudes the fast path can't.
- validate:gpu:real (RTX 3090) + validate:gpu (SwiftShader): ALL PASS (SA/BLA/df64esc sections included
  — computeSeries and the CPU escapePerturb oracle consume the new reference). 41 unit (incl. B/C/C2
  perturbation-vs-BigInt at 2^400 and the reference-vs-naive-orbit test D). gpu+perf e2e green.
- Frame-level (probe-strips, live viewer, RTX 3090): 2^-271 467→**388ms**, 2^-400 726→**576ms**.

### What's NEXT if more ref-build speed is ever wanted (recorded, not built)
**Reference REUSE across view changes** — the remaining big interactive win: a wheel-zoom sequence at a
fixed center recomputes the same orbit every settle (~120-180ms in-worker at 2^-400). Zhuoran rebasing
means ANY nearby reference is valid; compute the orbit once with prec headroom (e.g. +128 bits) and
reuse it across zoom-ins until zoomBits approaches the headroom, recomputing only on center moves
beyond a threshold (or extend chooseReference's relocation logic). Complications: the ref arrays are
TRANSFERRED to the GPU worker (would need cloning or re-request), maxIter grows with depth (orbit must
be extended — requires keeping the BigInt tail state), and chooseReference relocates to the deepest
pixel per view. Moderate complexity, biggest UX payoff during interactive deep zooming.
API note (companion-flagged): today computeReference returns only the Float64 export and DISCARDS the
BigInt tail (bx, by) — reuse/extension needs it to optionally accept+return that resume state. The
Spawn-29 loop restructure doesn't block this; it's an additive parameter, no refactor of the new math.

### Files
`src/math/reference.js` (2-mult square + guarded escape band; escapeBigInt carry), `src/math/bignum.js`
(toDouble fast path), NEW `tools/bench-ref.mjs` (pure-Node A/B: `BITS=… node tools/bench-ref.mjs`).

## ✅✅ SKIP-AWARE STRIP BUDGET — deep frame 1.4× (offscreen) / 3.2× (main-thread), SHIPPED (Spawn 28) — READ THIS FIRST

**Ask (Danielle, same live session as Spawn 27: "more optimization?"). The M6-FUTURE strip-tiling item,
measure-first. FINDING: `_stripRows()` sized strips from the worst-case budget `4e8/(W·maxIter)` —
but SA seeds EVERY pixel at iteration `skip` (93–95% of maxIter deep), so the TRUE per-pixel worst
case is `maxIter − skip` and the viewer was cutting 10–20× more strips than the budget intends
(2^-400: 100 strips of 6 rows). Each strip pays fixed orchestration: worker path = full-canvas
colorize + full-canvas ImageBitmap + post/ack; main path = flush + colorize + blit + a VSYNC-LOCKED
rAF (~16.7ms — 100 strips ≈ 1.67s of pure vsync waits, confirmed by the measured delta). THE FIX
(one formula change): budget on the POST-SKIP iterations — `_stripRows(skipIters)` uses
`maxIter − skip`, callers mirror `_setSA`'s active condition (inactive/absent seed ⇒ skip 0 —
glitch-overlay renders, df64 band, and naive are automatically unchanged).**

### Measured (tools/probe-strips.mjs, live viewer, real RTX 3090, deep boundary coord, ss=1, incl. identical ref build)
| depth · path | before (strips) | after (strips) | single-strip floor | speedup |
|---|---|---|---|---|
| 2^-271 offscreen | 641ms (67×9rows) | **467ms (5×147)** | 411ms | **1.37×** |
| 2^-400 offscreen | 1008ms (100×6) | **726ms (5×133)** | 640ms | **1.39×** |
| 2^-400 main-thread | 2308ms (100×6) | **726ms (5×133)** | 659ms | **3.2×** |
Captures ~76% of the theoretical (single-strip) win; the residual ~12ms/strip × 5 is the price of
watchdog safety + progressive reveal — kept deliberately.

### Why the watchdog envelope is UNCHANGED (the safety argument)
A strip's worst case is rows·W·(iterations each pixel can actually run) = the same 4e8 budget as
always. Pre-SA (Spawn 7, when 4e8 was calibrated with mobile as the target), deep pixels really ran
up to maxIter — post-SA they run at most maxIter−skip, and the new formula spends exactly the same
envelope on that true bound. Partition does not affect pixel values (crosscheck-tiled). NOTE: a
probe-strips `full` render (600 rows × 5k eff-iters ≈ 1.8e9) is ~4.5× OVER budget — fine headless
on the 3090, do NOT ship single-strip; 4e8 stays.

### Validation + files
crosscheck:offscreen ALL PIXEL-IDENTICAL (naive/df64/rescaled-fe, 0.000%/Δ0); 41 unit; gpu+perf e2e
green (the multi-strip progressive e2e forces its own strips on a no-SA naive view via the zero-arg
call — unaffected). Edits: `src/viewer.js` (`_stripRows(skipIters)`, `_drawTiledEscape(…, skip)`,
`_renderGpuPerturb` + `_renderGpuPerturbWorker` pass the active-SA skip). NEW `tools/probe-strips.mjs`
(npm-less; `GPU=1 [OFFSCREEN=0] [BITS=…] node tools/probe-strips.mjs`) — monkey-patches `_stripRows`
to A/B prod/×8/single-strip on the LIVE viewer; the ref build is identical across arms so deltas are
pure orchestration. FOLLOW-UP candidates (not taken): measured-adaptive strip growth (feedback on
actual strip time — would auto-tune per GPU; the static fix already captures most of it), batching
the colorize to only the strip's rows (the full-canvas colorize per strip remains, now ×5 not ×100).

## ⚖️ SHADER MICRO-OPT SWEEP — getZ prefetch SHIPPED (1.01–1.08× GPU, ~2× SwiftShader); ds_sqr measured neutral→OPT-IN (Spawn 27) — READ THIS FIRST

**Ask (Danielle, direct): "I think what I'm most interested in is optimizing the shader further." All
prior levers that prune WORK (BLA, lean, region-SA, interior detection) died on the warp-divergence/
escaper-tail wall; the surviving class is UNIFORM per-iteration cost cuts (the SA-margin lesson). This
spawn measured the two remaining unmeasured micro-levers in the inner loop. OUTCOME: (1) a SOFTWARE-
PIPELINED reference fetch (issue getZ(m+1) a full iteration early) — BIT-IDENTICAL by construction,
**1.01–1.08× on the RTX 3090 (growing with depth) and ~2.0× on SwiftShader** — SHIPPED as the
rescaled-engine default; (2) a dedicated df64 squaring `ds_sqr` — measured NEUTRAL on the RTX 3090
(the compiler already CSEs the duplicate split) and 0.95–0.98× on SwiftShader → kept OPT-IN
(`p.sqrOn`), NOT the default, per the lean-kernel precedent (default = measured-fastest config).
Also: a benching-methodology fix that matters for every future measurement on this host.**

### ⚠️ BENCH METHODOLOGY (load-bearing for anyone benching here): the companion LLM shares the GPU
The gemma-31B companion at 127.0.0.1:8051 runs ON THIS RTX 3090. Its inference bursts contaminate GPU
wall-clock A/Bs ASYMMETRICALLY — the first bench-sqr run showed a spurious **1.396×** at 2^-120 that
did NOT reproduce (0.989×/1.129× on repeat). Medians do NOT save you (a burst spans many reps). THE FIX
(now in bench-sqr/bench-prefetch): interleave the A/B arms over MANY rounds (ROUNDS=4–6) and take the
**MIN-of-min** per arm — contention only ever ADDS time, so min converges to the true frame cost. Do
NOT chat with the companion during a timing run. (bench-lean/bench-samargin et al. predate this and
used min-of-median over 2 rounds; their shipped verdicts were re-confirmable, but any FUTURE marginal
(<1.15×) result needs the robust protocol before you believe it.)

### 1. ds_sqr — dedicated df64 squaring (DF64_LIB), measured NEUTRAL-to-NEGATIVE → OPT-IN, not shipped
Dekker a·a needs ONE Veltkamp split (not two) and the cross terms collapse to exact doublings:
`e = (a_hi² − p) + 2·(a_hi·a_lo) + a_lo²; e += 2·(a.x·a.y)`. 6 ob() barriers vs ds_mul's 8, ~3 fewer
mults. ~6 of the ~11 per-iteration ds_mul are self-products (update quad Dx²,Dy²; escape-test zfx²,
zfy², dxf², dyf²), so IF the win were real it would be a uniform ~10–20% ALU cut. It is not:
- ISOLATED (probe-barrier squaring TIME A/B, chained t←t²+c): **1.00×** — the driver CSEs the duplicate
  split in a small shader. FULL SHADER (bench-sqr, robust min-of-min protocol): **0.999–1.031×** across
  2^-50(df64)/-120/-218/-271/-400 on the REAL GPU (the coarse big-shader optimizer phase from Spawn 20
  still catches this CSE, and/or per-iter ALU isn't what binds) and **0.946–0.980× on SwiftShader**
  (slightly NEGATIVE on the fallback backend, where deep frames cost SECONDS and speed matters most).
- CORRECTNESS (all green, so the infra is trustworthy if ever enabled): probe:barrier sqr ladder on the
  REAL GPU — sqr6 INTACT (2.1e-15 single / 2.4e-11 chained; sqr_none control COLLAPSES 1.2e-4; NB the
  chained-squares test must use an ATTRACTING quadratic orbit — a chaotic c amplifies legitimate
  df64-vs-double representation error past the collapse gate, a false COLLAPSED). validate:gpu
  (SwiftShader) + validate:gpu:real (RTX 3090): **ALL PASS** with ds_sqr enabled as default during the
  experiment. probe:xbackend 0.19%/0.67% @2^-22/-50 = the historical lean-barrier envelope.
  crosscheck:skip + crosscheck:tiled: IDENTICAL. 41 unit pass.
- NOT bit-identical to ds_mul(a,a) (the ~1-ulp LOW-word scheduling freedom the lean barrier already
  grants): sn diff 0–5 px per 36864 at the shallow/df64 bands, 0 px deep, 0 inside/outside flips.
- DECISION: **kept OPT-IN, default OFF** (`{sqr:true}` build opt / `p.sqrOn` render arg, program keys
  *sqr) — the production shader compiles the HISTORICAL bit-validated source. Same pattern as
  lean/df64-escape: measured-neutral-or-worse ⇒ not the default; kept validated for a future
  ALU/register-constrained GPU. GATE to re-decide: `npm run probe:barrier` (sqr ladder incl.
  controls) + `bench:sqr` + validate:gpu:real.

### 2. getZ software-pipelined prefetch (rescaled engine) — BIT-IDENTICAL win, SHIPPED as default
The loop fetched Z[m] and used it IMMEDIATELY (escape test) — zero fetch-to-use gap, so whenever the
warp scheduler can't hide the texture latency behind other warps (exactly the thin-warp deep regime),
it's exposed serially every iteration. The prefetch variant carries Z[m+1] in 2 extra vec2 registers:
each fetch is issued a FULL ITERATION before first use (the escape test + next update's ALU overlap
it). Same indices fetched (plus one extra per rebase/SA-seed, both rare; index clamped min(m+1,
uRefLen) — stays inside the texture, Z[uRefLen] is always valid) → **output bit-identical by
construction, and measured so** (crossArgs sn diff **0/N at every depth**).
- MEASURED (robust protocol, RTX 3090, production config SA-on): **1.012× @2^-120, 1.030× @2^-218,
  1.008–1.026× @2^-271, 1.051–1.075× @2^-400** (two independent runs) — small but consistently ≥1.0,
  GROWING with depth, exactly the divergence model's prediction (deep warps are thin/divergent → less
  cross-warp latency hiding → early issue pays). First strictly-safe positive lever since the SA margin.
- SwiftShader: **~2.0× faster** (7977→3989ms @2^-120, reproduced 1.89×/2.00× in two runs, sn diff 0)!
  SwiftShader JITs the shader to CPU SIMD; breaking the load→immediate-use dependency lets its
  pipeline overlap the texel gather (address calc + bounds + convert) with the escape-test ALU. This
  is a REAL user-facing path (GPU-blocklisted devices get SwiftShader WebGL) AND the whole e2e/CI
  suite — it should also relieve the known Spawn-9 flake (deep df64 e2e timing out on contended
  SwiftShader; that band is fe/rs… the rescaled deep e2e renders halve, the df64-band ones don't).
- DECISION: **shipped as the rescaled-engine default** (`p.prefetch !== false`; `{prefetch:false}` =
  the old fetch placement, program keys *pf).
- The df64/fe PORT was then built and MEASURED same-spawn (chasing the SwiftShader 2× at the
  shallower bands): perturbFragDf64/Floatexp take the same `{prefetch:true}` restructure — but it
  does NOT transfer: df64 @2^-50 = **0.997× real GPU, 0.957× SwiftShader** (bit-identical, sn diff 0).
  The 2× is RESCALED-SPECIFIC — that engine's heavy fe escape block gives the pipeline real work to
  overlap the fetch with; the df64 loop is too tight, and the extra live registers just cost. So
  df64/fe prefetch is **OPT-IN** (`p.prefetch === true` there — note the asymmetry: rescaled defaults
  ON, df64/fe default OFF, each per its own measurement). fe untimed (non-production oracle path;
  infra in place if ever needed).

### What this closes (and the honest frontier statement)
Per-iteration ALU micro-structure is now MEASURED-CLOSED on this GPU: a ~25%-fewer-ops df64 squaring
moves the frame ≤3%. Combined with Spawn 19 (df64-escape neutral) + Spawn 22 (lean neutral) + Spawn 23
(interior ceiling 1.3–1.55×), the deep frame is conclusively bound by warp divergence + the genuine
late-escaper tail + (a small, now-harvested slice of) fetch latency — NOT by instruction count. Anyone
re-proposing an inner-loop ALU lever should have to explain why ds_sqr measured 1.00×. The remaining
follow-ups: (a) ~~port the prefetch to df64/fe~~ — DONE same-spawn, measured NEUTRAL-to-NEGATIVE
(0.997× GPU / 0.957× SwiftShader @2^-50 df64; the 2× is rescaled-specific) → opt-in, closed;
(b) the M6-FUTURE ORCHESTRATION items (strip-tiling overhead at extreme maxIter; mobile loop-length
caps) — not further inner-loop surgery on this hardware.

### Files
`src/gpu/glsl.js` (DF64_LIB ds_sqr; FE_LIB fe_sqr; sqrDefs() macro injection — DEFAULT expands to the
historical ds_mul(a,a)/fe_mul(a,a) source; DS_SQR/FE_SQR at all self-product call sites in
naive-df64/perturb-df64/fe/rescaled; opts.prefetch loop restructure in perturbFragRescaled),
`src/gpu/renderer.js` (p.sqrOn opt-in / p.prefetch default-ON plumbing + program keys *sqr/*pf),
`test/gpu/harness.html` (crossSqr, generic crossArgs, benchPerturb passthrough), `tools/probe-barrier.mjs`
(SQRS ladder + squaring TIME A/B + attracting-chain inputs), NEW `tools/bench-sqr.mjs` +
`tools/bench-prefetch.mjs` (npm bench:sqr / bench:prefetch — both use the contention-robust protocol).

## ✅ UX POLISH — onboarding hint + keyboard nav + live coordinate HUD (Spawn 26) — READ THIS FIRST

**Ask (the GPU/precision frontiers are all closed + validated — Spawns 22–25 — so the companion + I took
the AGENDA's "lock in or polish UX" branch): close the discoverability/accessibility gaps that still made
the viewer feel like a tech demo rather than a finished product. The engine was rock-solid; a first-time
user just had no idea how to drive it, desktop users had no keyboard, and there was no always-on sense of
"where am I." OUTCOME: three additive, well-tested UX features — a first-run gesture hint, full keyboard
navigation, and a live center-coordinate HUD readout — plus a fullscreen + help affordance. Zero changes
to the render math or any pixel; all 41 unit + full e2e (both projects) green, 5 new e2e added.**

### What shipped (companion-ranked priority: onboarding > coords > keyboard > fullscreen)
1. **First-run gesture hint** (`index.html` #hint, `styles.css` .hint*, `main.js` hint block). A centered
   dismissible card shown once per device (localStorage `mb_hintSeen`) listing the gestures: Tap=zoom-in
   (Shift/right-tap=out), Drag=pan, Pinch=zoom, Scroll/+−=zoom & Arrow keys=pan, ☰ Menu. KEY DESIGN: the
   whole overlay is **`pointer-events: none`**, and dismissal is a **capture-phase** one-shot listener on
   `pointerdown`/`keydown`/`wheel` that does NOT consume the event. So the very first tap BOTH dismisses the
   hint AND zooms (exactly what the hint promised) — and, load-bearing, it means the overlay never blocks the
   existing e2e canvas clicks / panel-toggle clicks (the reason it's pointer-transparent, not a modal). The
   "?" corner button (or the `?` key) re-opens it any time. Storage access is try/caught (private mode →
   still shows, just won't persist the dismissal).
2. **Live coordinate HUD** (`index.html` #coords, `main.js` updateView + `shortCoord()`). The always-on HUD
   now shows the view CENTER in complex `a + bi` form below the status/zoom line. `shortCoord()` compacts a
   (possibly 400-digit) high-precision decimal to sign + int + ≤10 fractional digits, **trims trailing zeros**,
   and only appends "…" when a *significant* (non-zero) digit was actually dropped — so an exact -0.5 shows
   "−0.5", not "−0.5000000000…", while a deep coord shows "−0.743643887… + 0.131825904…i". Real minus glyph;
   "0" for true zero. The panel's Re/Im textareas still carry FULL precision; the HUD is just orientation.
   CSS caps it at 86vw with ellipsis overflow as a final guard. Updates via `onView` (fires at render START),
   so it tracks the center instantly on pan/zoom/jump without waiting for the render to finish.
3. **Keyboard navigation** (`main.js` keydown handler; `viewer.js` new `panByPreview()`). Arrow keys pan
   (camera-direction convention: → increases Re), `+`/`=` zoom in, `-`/`_` zoom out, `f` fullscreen, `?` help.
   `panByPreview(dxPix,dyPix)` is a new viewer primitive that reuses the EXACT gesture-preview machinery
   (`_beginPreview` → translate `T.e/T.f` → `_applyPreview` → `_scheduleSettle`) so a key-pan is smooth +
   cancels any in-flight render + settles to one sharp render, identical to a drag (a=1 → `_endGesture` folds
   the pure translation into the HP center, radius unchanged). DISCRETE by design: auto-repeat (`e.repeat`) is
   ignored so a held key can't fly the fixed preview snapshot off into black; each press is one PAN_STEP=0.18·
   viewport step. The handler is INERT inside form fields (`INPUT`/`TEXTAREA`/`SELECT` → return) so typing the
   coordinate / iteration inputs is never hijacked, and it leaves ctrl/meta/alt browser shortcuts alone.
4. **Fullscreen + help corner buttons** (`index.html` .corner, `main.js` toggleFullscreen). Two small
   top-right buttons: "?" (re-open hint) and "⛶" (toggle fullscreen). `requestFullscreen`/`exitFullscreen`
   are optional-chained + try/caught (blocked without a user gesture / unsupported → silently ignored). Gives
   mobile (no keyboard) parity with the `f`/`?` keys.

### Why these and not more (measure-the-gap, companion-vetted)
The companion ranked the four candidates and confirmed several things were ALREADY handled, so I didn't
re-add them: safe-area insets (the HUD/panel already use `env(safe-area-inset-*)`), a loading state (the
status line already shows `rendering NN%`), the precision-wall warning (Spawn 25's "⚠ max depth"), and UI
collapsibility (the panel is a hidden-by-default bottom sheet; the only persistent UI is the tiny HUD + two
small corner buttons + the round ☰ — already "the fractal is the star"). So a hide-all-UI toggle was judged
low-value and skipped (also: on mobile, with no keyboard, hiding the only restore affordance is a trap).

### Validation (all green; NO pixel/math change)
- 41 unit tests unchanged-pass (didn't touch any math module). Full `viewer.spec.mjs` **26/26** on
  desktop-chrome (21 pre-existing + 5 new), `gpu.spec.mjs` 10/10 + `perf.spec.mjs` 2/2 on desktop-chrome,
  and the 5 new tests pass on mobile-chrome too. The pre-existing real-mouse click-to-zoom test (the one the
  hint overlay sits on top of) STILL PASSES — proving the pointer-events:none + non-consuming dismissal works.
- 5 NEW e2e (`viewer.spec.mjs`): arrow-key pan (→ raises Re, y/zoom unchanged, preview committed); +/- zoom
  (in then out returns to ~start); keyboard nav does NOT hijack typing in #reIn; first-run hint shows + is
  pointer-events:none + a tap dismisses-and-zooms + persists across reload + "?" re-opens; coordinate HUD
  tracks the center (home "−0.5" → a deep jump shows the right digits).

### Files
Edits: `index.html` (HUD restructured to a column: status/zoom row + #coords; .corner help/fullscreen
buttons; #hint overlay), `styles.css` (.hud column, .coords-hud, .corner/.corner-btn, .hint/.hint-card/
.hint-gestures), `src/main.js` (shortCoord + complex-form coords readout; keydown nav; toggleFullscreen;
first-run hint show/dismiss/persist), `src/viewer.js` (new `panByPreview()` primitive), `test/e2e/viewer.spec.mjs`
(+5 tests), README. New dev tool: `tools/shoot-ux.mjs` (captures hint / HUD / deep-coord screenshots →
screenshots/ux-*.png; not wired to npm, just a visual-check helper). NET: pure UX/accessibility polish — the
viewer is now self-explanatory on first open, fully keyboard-drivable on desktop, and always shows where you
are. The engine/throughput/precision frontiers remain closed + validated (Spawns 9, 20–25).

## ✅✅ DEEP PRECISION WALL — floor 2^-600 → 2^-1010 + graceful clamp (Spawn 25) — READ THIS FIRST

**Ask (Danielle's standing "optimize until 900% faster" is structurally MET/closed — Spawns 22–24; the
companion + I picked the most concrete remaining item): push the deep-zoom FLOOR below 2^-600 toward the
true double-precision wall, and turn the heuristic floor into a SPECIFICATION — find where the double
representation actually breaks and degrade gracefully there instead of silently rendering garbage.
OUTCOME: the deep GPU floor is lowered 2^-600 → 2^-1010 (validated bit-clean against the BigInt-EXACT
oracle at a genuine boundary coordinate); below 2^-1010 the viewer CLAMPS the radius at the precision
wall and signals "max depth". 2^1010 zoom is ~10^304× — astronomically past the original 2^400 goal.**

### Why 2^-1010 is the wall (the specification, not a guess)
Every per-pixel quantity is a normal double. The reference center is BigInt (arbitrary precision), but
the per-pixel offset dc ~ radius and the per-pixel STEP = 2·radius/computeHeight are doubles. A double is
exact (53-bit relative) only while NORMAL (≥ 2^-1022); below that it goes subnormal and sheds mantissa
bits, and the step underflows even earlier (adjacent pixels stop being distinguishable → the image goes
blocky/degenerate). The binding constraint is the step at the LARGEST compute height (MAX_COMPUTE_DIM =
8192 ≈ 2^13): step ≥ 2^-1022 ⟺ radius ≥ 8192·2^-1023 = **2^-1010**. So 2^-1010 guarantees a normal,
full-precision per-pixel step at any resolution the viewer can produce. (The delta-iteration dz also runs
in double, but its rounding is bounded by Zhuoran rebasing independent of maxIter — confirmed below — so
the dc/step representation is the binding wall, not iteration accumulation.)

### How it was validated (two independent oracles, real hardware)
1. **BigInt-EXACT ground truth** (`tools/probe-wall.mjs`, NEW): the M4 correctness methodology (escapeBigInt,
   the authoritative oracle used by tests B/C/C2 at 2^400) extended into the deep band. For each pixel it
   compares the renderer's double-dc/double-delta escape count to escapeBigInt at the pixel's TRUE geometric
   offset computed in BigInt (NOT via the double dc — so it sees BOTH the delta-iteration error AND the
   double-dc rounding/underflow of the true offset). On the genuine deep boundary coordinate: **escape
   counts are BIT-EXACT (0 mismatch) FLAT from 2^-700 to 2^-1010**, dcRelErr ~2–3e-16 (= pure double
   rounding, ~2^-52). The dc/step stay normal through ~2^-1016 and go SUBNORMAL at ~2^-1020 (for a 10² grid;
   the viewer's larger compute height subnormalizes the step shallower, ~2^-1012 at 8192px — exactly why the
   floor is set at 2^-1010 with margin). [Caveat baked into the tool: toDouble underflows at prec > 1074, so
   probe-wall computes dcRelErr as a BigInt ratio, never through a double.]
2. **GPU fe/rescaled vs the CPU oracle** (`tools/probe-deep500.mjs`, real RTX 3090, new deep coordinate):
   **0.000% escape-count mismatch FLAT 2^-600 → 2^-1010** (maxIter 150k → 253k). The odd 1–2 differing pixels
   (rs 1.282% @2^-800 ≈ 1 px of 78; rs/fe 0.426% @2^-1000 ≈ 2 px of 470; maxΔsn < 1.0) are isolated
   ill-conditioned boundary pixels — the measure-zero set any 53-bit method is noisy on — NOT a depth-
   monotonic collapse (the per-iter df64 accumulation is rebasing-bounded; the reference is BigInt-sampled,
   never iterated in df64, so it can't drift with depth).
3. **validate:gpu + validate:gpu:real** exterior-arithmetic checks extended 600 → **800, 1000** (the fe
   arithmetic is exponent-magnitude-agnostic); ALL PASS on SwiftShader AND the RTX 3090. 41 unit tests pass
   (the dispatch test updated: 2^-1000 → 'perturb-fe', clamp covers below the floor).

### The genuine deep boundary coordinate (the enabler)
`tools/gen-deep-coord.mjs` descends a boundary filament in BigInt to a target depth. KEY FIX this spawn:
its ITERCAP must scale with depth — at the old FIXED 140000 the descent fell INTO the set past ~2^-560
(every probe pixel read as interior at the capped maxIter → it drifted off the filament). Now ITERCAP
DEFAULTS to the depth's own autoMaxIter + margin (≈ 400+TARGET·250+8000), so it holds the straddle to
2^-1010 (self-check at 2^-990: 84 escaping / 492 interior, escape counts spread near maxIter → a real
chaotic boundary). The coordinate is wired as the default in probe-deep500 + probe-wall.
(`TARGET=1010 node tools/gen-deep-coord.mjs`.)

### Graceful degradation (don't render garbage past the wall)
- `viewer.js MIN_RADIUS = 2^-1010` (== render.js GPU_PERTURB_FE_FLOOR, kept equal). `_clampRadius()` enforces
  it in the two radius-commit paths (`_afterRadiusChange` → zoomAt/_endGesture, and `setState` → deep
  bookmark/URL/"Go"). `atMinRadius` flag is surfaced in getState().
- zoomBy/clickZoom IGNORE a further zoom-IN once `atMinRadius` (so the gesture preview doesn't over-scale
  into a blurry image that snaps back on settle); zoom-OUT and pan still work, and a tap signals "max depth".
- main.js: the status line shows "⚠ maximum zoom depth reached (double-precision limit)" on the maxdepth
  signal; the debug zoom line shows "· AT PRECISION WALL"; the zoom readout shows "(max)".
- Correctness-only change: nothing about the rendered image at any reachable radius changes — this only
  prevents reaching an UN-renderable one. e2e covers it (`viewer.spec.mjs`: a sub-wall radius is clamped +
  flagged; further zoom-in is suppressed).

### Files
NEW: `tools/probe-wall.mjs` (npm `probe:wall`). Edits: `src/math/render.js` (GPU_PERTURB_FE_FLOOR 2^-600 →
2^-1010 + the dispatch doc), `src/viewer.js` (MIN_RADIUS + _clampRadius + zoom-in guard + getState),
`src/main.js` (maxdepth status + wall readouts), `tools/gen-deep-coord.mjs` (ITERCAP default 280000),
`tools/probe-deep500.mjs` (new default coordinate), `tools/validate-gpu.mjs` (exterior 800/1000),
`test/unit/gpu.test.mjs` (dispatch), `test/e2e/viewer.spec.mjs` (clamp). NEXT structural levers unchanged
(all measured/closed for raw speed): the win here is a CAPABILITY + correctness specification, not throughput.

## ✅✅ OFFSCREENCANVAS GPU WORKER — raster off the main thread (Spawn 24)

**Ask (Danielle's pivot): the GPU raw-throughput frontier is closed (Spawn 23 — the deep frame is
warp-divergence/escaper-tail bound, can't make ms/frame faster), so shift focus. Her #1 option:
OffscreenCanvas, to get the GPU raster off the main thread for a smoother mobile experience. OUTCOME:
built it, shipped as the DEFAULT when supported, PIXEL-IDENTICAL to the old path on SwiftShader AND
the real RTX 3090, and it ELIMINATES main-thread blocking during a deep render (99842ms → 0ms of Long
Tasks at 2^-271 on SwiftShader; the rAF heartbeat survives the whole render — 55 → 5061 ticks).**

### Why this is the right pivot (and what it does/doesn't buy)
The throughput wall means we can't make the deep FRAME faster (ms/frame is warp-bound). But the viewer
was a "synchronous pull": the GPU raster + every per-strip `ctx.drawImage(gpu.canvas)` ran on the MAIN
thread, and that drawImage forces a synchronous GPU flush/readback — so a long deep render janks input/
scroll even though the GPU is the bottleneck, not the CPU. OffscreenCanvas flips it to an "asynchronous
push": a dedicated worker owns the WebGL context + an OffscreenCanvas, runs the whole escape+color strip
loop, and pushes finished (already-rasterized) ImageBitmap strips back; the main thread only draws a ready
bitmap (no GL sync). The win is RESPONSIVENESS (input latency), not throughput — measured as main-thread
jank, per the project's measure-first discipline.

### Architecture (why it was a clean refactor, not a rewrite)
The GpuRenderer ALREADY owned its own WebGL canvas separate from the display canvas (the viewer composited
it via drawImage), and makeCanvas() already returned an OffscreenCanvas when `document` is undefined — the
renderer was written worker-portable. So the worker runs the EXACT same df64/floatexp/rescaled shaders +
strip-tiling; only the loop's HOME changed. The preview/gesture model, the CPU reference-orbit worker pool
(worker.js), and all the math are untouched.
- `src/gpu/gpu-worker.js`: owns the OffscreenCanvas + GpuRenderer; runs runRender() (= the old
  viewer._drawTiledEscape: clearSn → per strip {drawStrip, flush, colorize, transferToImageBitmap,
  post strip, await ack}). Imports renderer.js + palette.js (both pure). Back-pressure: waits for the
  main thread's `ack` before the next strip (≤1 ImageBitmap in flight — no unbounded backlog). Supersede:
  a newer render (higher gen) bails the loop before any GL.
- `src/gpu/gpu-worker-client.js`: main-thread bridge. renderEscape(plan, transfers, {onStrip,onDone,
  onError}); recolor(gen,color,cb) for instant palette/overlay changes (re-colorizes the worker's existing
  sn FBO, no recompute — seq-guarded so rapid slider drags draw the latest); cancel/dispose; mirrors
  `lost`/`info`/`supported`. Forwards webglcontextlost/restored from the worker to the viewer.
- `src/viewer.js`: `offscreen` flag (default on; `opts.offscreen===false` or `?offscreen=0` forces the
  legacy path). `_ensureGpu` does a SYNCHRONOUS main-thread capability probe (OffscreenCanvas + WebGL2 +
  EXT_color_buffer_float) — if it passes on main it passes in the worker (same browser GL) — then constructs
  the client and routes optimistically (message ordering guarantees the worker handles `init` before
  `render`). render()/refReady route naive + perturb to the worker variants; the CPU reference computation
  (worker.js) is unchanged. The on-main-thread GpuRenderer remains the automatic FALLBACK (offscreen
  unsupported, a worker render error, or give-up after repeated context losses) → then the CPU pool. Context-
  loss recovery, palette recolor, glitch-count readback, and the gesture-preview snapshot all work offscreen.

### Correctness (the gate): PIXEL-IDENTICAL, both backends
`tools/crosscheck-offscreen.mjs` renders the SAME view via `?offscreen=1` (worker) and `?offscreen=0`
(main thread) at each GPU engine (naive / df64 / rescaled-fe) and diffs the full canvas: **0.000%
differing pixels, max channel Δ 0** on SwiftShader (re-confirmed this spawn) AND the real RTX 3090
(Spawn-24 prior run). RGBA8 round-trip through an ImageBitmap is lossless and both paths run identical
shaders, so this is exact, not bulk-tolerance. (npm crosscheck:offscreen[:real].)

### The win (measure-first): main-thread blocking during a deep render
`tools/bench-offscreen-jank.mjs` (W3C Long Tasks API + a rAF heartbeat max-gap; render time too), deep
render 2^-271, 400×400, ss=1, SwiftShader (the clearest demo — SwiftShader's GL is pure CPU, so moving it
to the worker frees the main thread entirely):
| path                | render   | Long Tasks | main-thread blocked | worst heartbeat gap | rAF ticks |
|---------------------|----------|------------|---------------------|---------------------|-----------|
| offscreen (worker)  | 87193 ms | **0**      | **0 ms**            | 2892 ms             | **5061**  |
| main-thread         | 100291ms | 29         | 99842 ms            | 6368 ms             | 55        |
The render TIME is ~the same (throughput is warp-bound — as expected; offscreen is even a hair faster here,
within noise / less main-thread self-contention). The WIN: the worker path produces **ZERO main-thread Long
Tasks** (vs 99842ms ≈ the ENTIRE render blocked on the main path) and keeps the **rAF heartbeat alive through
the whole render (5061 ticks vs 55** — a UI animation would run ~smoothly offscreen, frozen on main). The
residual 2892ms worst-gap on the offscreen path is NOT a Long Task (longTasks=0) — it's a non-GL scheduling/
GC/reference-setup pause, and still well below the main path's 6368ms. So input/scroll/animation stay
responsive during a deep dive. (npm bench:offscreen[:real]; BITS=/VP= override.)

WHERE THE WIN APPLIES (honest scoping, real-GPU `:real` run, same 2^-271/400²/ss=1): on the fast RTX 3090
BOTH paths produce 0 Long Tasks (offscreen render 511ms / main 896ms, heartbeat gap 24ms / 17ms) — the GL
work is async and finishes in ~ms, so a 400² deep frame doesn't block the main thread even on the main-thread
path. The offscreen win therefore matters for SLOW GL — a software rasterizer (SwiftShader, the demo above),
a weak/throttled mobile GPU, large frames, or high supersampling — exactly where the per-strip main-thread GL
flush is expensive. It is neutral-or-better on a fast GPU (here even a hair faster, async-push pipelining),
never worse. Consistent with the mobile-first goal: the worst-case device is where responsiveness matters,
and SwiftShader is the proxy that measures it.

### Mobile context-loss recovery still works through the worker
The GL context now lives in the worker, so loss fires webglcontextlost THERE; the renderer's listener
forwards it (worker→main) and the viewer falls back to the CPU pool, then re-renders on the GPU worker once
restored (give-up after GPU_MAX_LOSSES → CPU). `tools/probe-recover-offscreen.mjs` exercises this end-to-end
via a worker-side WEBGL_lose_context hook (gpuWorker.loseContextForTest): **ALL PASS** on SwiftShader — home
on gpu-naive via the worker → loss forwarded (gpuWorker.lost) → CPU (naive) while lost → restore → GPU worker
resumes (gpu-naive) → recovered image non-blank (max 254, distinct 100). (npm probe:recover:offscreen[:real].)

### Validation (all green) + what shipped
- crosscheck-offscreen 0.000%/Δ0 (SwiftShader + RTX 3090); probe-recover-offscreen ALL PASS; 41 unit
  unchanged; the offscreen default-path e2e ("OffscreenCanvas worker path is the default and renders all
  engines") PASSES (desktop-chrome, 20s). Offscreen is the DEFAULT, so the whole e2e suite exercises it (the
  golden fingerprint determinism test passes on BOTH paths).
- e2e test edits (justified, not papering over): the determinism test had a latent wait race
  (`waitDone(page, doneCount()-1)` could sample the just-cleared canvas — fixed to capture the count BEFORE
  the resize+render; race-free on both paths). Three tests that reach into MAIN-THREAD GPU internals
  (viewer.gpu.gl loseContext, viewer.gpu.canvas, the _blitGpuStrip hook) are pinned to `?offscreen=0` —
  they validate the still-shipping fallback renderer's mechanism; the offscreen equivalents are covered by
  crosscheck-offscreen (final image) + probe-recover-offscreen + the new default-path e2e.
- New: src/gpu/{gpu-worker,gpu-worker-client}.js; tools/{crosscheck-offscreen,bench-offscreen-jank,
  probe-recover-offscreen}.mjs; npm crosscheck:offscreen[:real], bench:offscreen[:real],
  probe:recover:offscreen[:real]. Edits: src/viewer.js (offscreen routing + helpers), src/main.js
  (?offscreen URL param), test/e2e/{viewer,gpu}.spec.mjs, package.json.

PORTABILITY: OffscreenCanvas-WebGL2-in-a-worker is supported by all current target browsers (Chromium 148
here); where it's absent the synchronous capability probe declines and the viewer uses the on-main-thread
renderer unchanged. NEXT structural levers unchanged: fewer per-iter texture fetches, push the floor below
2^-600. The deep-band throughput frontier stays closed (Spawn 23) — this spawn bought RESPONSIVENESS, not
throughput, which was exactly the pivot's goal.

## ✅ THE DEEP-GPU BUG IS FIXED (Spawn 9) — READ THIS FIRST

**Spawn 8 found the deep GPU engines were ~90% wrong on real NVIDIA and blamed the
compiler defeating the df64 barrier in the big shader. That diagnosis was WRONG. Spawn 9
localized the TRUE cause and fixed it in one place.**

ROOT CAUSE: the df64 reference orbit is read per-iteration from a float32 texture via
`texelFetch`. The samplers were declared `uniform sampler2D uRef;` with NO precision
qualifier → they default to **mediump** in the fragment shader, and NVIDIA honours
mediump literally, returning the texel rounded to **~fp16 (~10-bit mantissa)**. That
~10-bit Z destroys df64 (its lo word becomes noise), so the deep engines were ~90% wrong
on real hardware — EVEN WITH Spawn 8's barrier. SwiftShader implements mediump as fp32,
which is exactly why it (and 6 spawns) never saw it.

THE FIX (src/gpu/glsl.js): declare every reference sampler `highp sampler2D` — `uRef` in
perturbFrag / perturbFragDf64 / perturbFragFloatexp / perturbFragRescaled, and `uSn` in
COLOR_FRAG. One word per sampler. NO renderer/JS change needed.

GATES (all green on RTX 3090, Vulkan): probe-xbackend 90% → **0.00%** (GPU bit-identical
to SwiftShader); `validate:gpu:real` 12 FAIL → **0 FAIL / ALL PASS** (df64/fe/rescaled,
2^-3…2^-340, same 0–1.2% chaotic-boundary mism as SwiftShader); crosscheck skip+tiled
0-diff on GPU; 31 unit + SwiftShader validate unchanged (no regression); a real 2^-50
df64 viewer render is a clean fractal (screenshots/df64_2e50_gpu1.png, glitches=0).

HOW IT WAS LOCALIZED (tools/, kept for the record): probe-localize (BASE reproduces, and
barriering the float32 reductions changes NOTHING → not the rebase decision), probe-state
(the df64 dz STATE is wrong from iteration 5, while SwiftShader tracks the oracle to
1e-14), probe-loop/probe-mag/probe-matrix (rolled vs unrolled loop, operand magnitude,
const vs uniform Z, escape-block presence — ALL intact, so none of those), probe-texz
(per-iteration texelFetch'd Z is the ONLY operand that collapses), probe-fix (laundering
Z through the barrier does NOT help, but `highp sampler` makes it INTACT — pinned it).

PRE-EXISTING FLAKE (not from this fix): the e2e `coordinate "Go" input navigates to a
location` (desktop-chrome) can time out (30s waitDone) on a 2^-9.8 df64 render under
SwiftShader when the CPU is contended (this session + the companion LLM share it). Proven
pre-existing: it fails identically with the original pre-highp shader (git-stash A/B). It's
a SwiftShader CPU-speed timeout, not a correctness bug; mobile-chrome (smaller viewport)
passes. 41/42 e2e green otherwise. Fix later by bumping that test's timeout if it annoys.

The XOR optimization barrier (Spawn 8, ob()) is STILL REQUIRED and kept: it fixes a
SEPARATE, texture-independent reassociation of the Veltkamp split (`ca-(ca-x)→x`), proven
in isolation by probe-df64 (no texture involved). Two distinct NVIDIA issues; both fixed.
POSSIBLE FOLLOW-UP (perf, untested): the barrier reportedly costs ~2.6×; now that the
sampler bug is gone it's worth re-checking whether a lighter barrier placement still holds
deep — gate any such change on probe-xbackend (must stay 0.00%) + validate:gpu:real.
**[Spawn 20 DID THIS — and it turned out to be not just perf but a CORRECTNESS FIX. See the
next section. The "still holds deep" assumption above was FALSE on the current driver: the
FULL (maximal) barrier COLLAPSES in the big shader; the LEAN one is what holds. Read on.]**

---
## ✅✅ df64 BARRIER MINIMIZATION — a CORRECTNESS FIX + 1.64× per-op (Spawn 20) — READ THIS

**Ask (continuing "optimize until 900% faster"): the one documented-but-unattempted GPU lever
— lighten the df64 optimization barrier (AGENDA NEXT, NOTES Spawn-9 follow-up). Spawn 19 had
pinned the dominant per-iteration GPU cost on "the barriered ds_mul squarings… the ob() XOR
barrier makes every ds_* op heavy." OUTCOME: lightening it is NOT just a speedup — it is a
CORRECTNESS FIX. The shipping MAXIMAL barrier is CURRENTLY BROKEN on the real RTX 3090
(validate:gpu:real = 24 FAIL, a depth-monotonic df64→f32 collapse to 99% wrong deep); the
minimal placement makes it PASS and is 1.64× faster per op. Shipped as the new default.**

### The surprise (fully reproducible, the whole reason this matters)
The shipping `ds_add`/`ds_mul` wrapped EVERY intermediate float in the `ob()` XOR barrier
(8/add, 24/mul = 32 per mul+add). The barrier blocks the two NVIDIA compiler transforms that
break Dekker/Veltkamp df64: (i) **reassociation that cancels** (`ca-(ca-x)→x`, `s-(s-x)→x`,
`(p+e)-p→e`) and (ii) **FMA contraction** of `a*b±c` (de-rounds the error term `a_hi*b_hi−p`).
The MINIMAL placement that blocks BOTH needs only ~half: barrier the split muls (`ca`,`cb`) +
their inner subs, the rounded product `p`, the one fma-critical product `a_hi*b_hi`, and the
two renorm materializations (`p+e`, `s+e`) — leaving the small additive corrections to schedule
freely. ds_mul: **8** barriers (was 24). ds_add: **5** (was 8). Combined 14 vs 32 (−56%).

Then the measurement, all on the real RTX 3090 (ANGLE/Vulkan), all reproducible:
1. **ISOLATED single-op** (`tools/probe-barrier.mjs`, tiny shader): BOTH full (32) and lean (14)
   keep df64 INTACT (~1e-14 relerr, 4 orders below the 1e-10 collapse gate). The controls
   `minC`/`lean5`/`none`/`lean8_nop` COLLAPSE or degrade — the kept barriers ARE all load-bearing.
2. **FULL perturbation shader** (`validate:gpu:real`, thousands of inlined ds ops): the **FULL
   barrier FAILS with 24 FAILURES** — a clean depth-monotonic collapse: df64 mism 3.9% @2^-3 →
   22% @2^-12 → 82% @2^-35 → **99% @2^-70/-90** (fe/rs/SA/BLA all 99% there too). The **LEAN
   barrier: ALL PASS** (0.0–1.1% mism = the normal ill-conditioned-boundary floor, matches the
   CPU oracle). `probe-xbackend` agrees: full 21%/90% vs lean 0.19%/0.68% at 2^-22/2^-50.
3. **Fair per-op timing** (`bench:barrier` = probe-barrier TIME=1, fixed-iteration loop, NO early
   exit so full and lean do IDENTICAL work — collapse can't bias the clock): lean **1.64× faster**
   (714938 vs 436907 Mop/s). [SwiftShader timing is unreliable here — CPU-contended by the
   companion LLM, loadavg ~5 — and irrelevant: full can't run correctly on the real GPU anyway.]
4. **End-to-end**: the real viewer at the 2^-50 df64 coordinate (the band that was 86% wrong)
   renders a CLEAN seahorse fractal, glitches=0 (`GPU=1 node tools/shoot-df64.mjs`,
   screenshots/df64_2e50_gpu1.png). bench-gpu:real (lean absolute): fe ~9000, rs ~19600 Mit/s.

### Why MORE barriers COLLAPSE where FEWER don't (the mechanism)
This reconciles the long-standing mystery and VINDICATES Spawn 8's ORIGINAL diagnosis ("the
driver defeats the barrier in the LARGE inlined perturbation shader") — which Spawn 9 thought
it had fully superseded with the highp-sampler fix. BOTH were real: the mediump sampler was a
separate bug (Spawn 9, still fixed), AND the maximal barrier independently collapses the big
shader. Mechanism (companion-LLM-vetted, plausibility "high"): the 32-barrier ds ops, inlined
thousands of times, blow up the shader IR past a compiler **complexity/size threshold**; NVIDIA's
ANGLE compiler then **phase-switches to a coarser optimizer** (to meet a compile timeout / fit
memory / relieve register pressure) that "sees through" and **strips the XOR barriers en masse**,
re-exposing every df64 op to FMA/reassociation → collapse. The isolated single-op shader is small
enough to stay in the precise pass (hence probe-df64 always showed full=INTACT — that test is too
small to trip it). The LEAN shader stays under the threshold in the big shader too. It is a
"Goldilocks zone": enough barriers to stop the standard FMA/reassoc passes, few enough not to trip
the nuclear pass. (Whether the prior "ALL PASS" claims regressed via a driver update or were never
re-run, the CURRENT empirical truth is what governs: full=broken, lean=correct, here and now.)

### Decision + what's preserved
LEAN is the **new default** (`src/gpu/glsl.js` DF64_LIB), NOT a toggle: the full barrier is
DEMONSTRABLY BROKEN on this GPU, so there is no useful "fall back to full." This is unlike
BLA/df64-escape (neutral/negative → kept off); this is strictly better (correct AND faster), so
it ships on. The universal-correct fallback remains SwiftShader (fp32-exact, both placements pass
there — validate:gpu ALL PASS, 39 unit pass). **PORTABILITY CAVEAT (load-bearing): correctness
here is EMPIRICALLY validated per-GPU, NOT guaranteed by the GLSL spec.** A different vendor
(AMD/Intel/mobile) or a future NVIDIA driver/arch could shift the threshold — lean could become
"too lean" (FMA leaks) or a new collapse mode appears. THE GATE for any new target GPU or any
DF64_LIB edit: `npm run probe:barrier` (isolated, must stay INTACT for the kept variants) +
`probe:xbackend` + `validate:gpu:real` (must ALL PASS) + `bench:barrier` (the per-op delta). If a
future GPU collapses lean, probe-barrier's variant ladder tells you exactly which barrier to add
back. New: `tools/probe-barrier.mjs` (correctness ladder + TIME=1 per-op bench); npm `probe:barrier`,
`bench:barrier`. Edit: `src/gpu/glsl.js` ds_add/ds_mul (+ the BARRIER PLACEMENT comment). The
1.64× per-op stacks MULTIPLICATIVELY with SA (SA cuts iteration COUNT, this cuts per-iter COST) —
and, more fundamentally, it is what makes the deep GPU path CORRECT on real hardware at all, so
every prior GPU speedup (rescaled, SA) is now actually DELIVERABLE on the real RTX 3090.

### IMPLICATION for prior real-GPU benchmarks (re-validate on the correct shader)
Every prior real-GPU A/B (bench-sa/bench-bla/bench-df64esc, Spawns 16–19) ran with the FULL
barrier — which collapses in exactly the chaotic deep band those benches use, so the shader was
early-exiting on garbage. Whether it collapsed WHEN those spawns ran (vs. a driver update since)
is unknown, but the numbers should be re-taken on the LEAN shader to be trustworthy. Re-checked
this session on the LEAN shader, ALL THREE re-confirmed — the collapse moved ABSOLUTE correctness
but the RELATIVE optimization verdicts were robust: **SA 4.46/4.27×, 6.74/6.76×, 9.31×** (still THE
lever); **df64-escape NEUTRAL 1.03×/1.00×/0.92×** (stays off); **BLA still a NET LOSS — SA+BLA
2.13×/3.22×/4.67× vs SA's 4.27×/6.76×/9.31×, ~2× slower** (its penalty — texture latency + post-SA-
tail warp divergence — is structural/barrier-independent, exactly as predicted; stays OFF). So the
optimization conclusions did NOT change; only the correctness did. The remaining 900%-gap is at the
MODERATE-deep band (2^-120 ~4.5×, vs ~9.3× at 2^-400) and the open levers are structural: fewer
per-iter texture fetches, less warp divergence, or off-GPU work (OffscreenCanvas).
**[Spawn 21 closed part of this gap — see the next section. Order-5 SA buys 1.24× at 2^-120.]**

---
## ✅✅ HIGHER-ORDER SERIES APPROXIMATION (order 5) — a NET WIN at the moderate-deep gap (Spawn 21)

**Ask (continuing "optimize until 900% faster"): close the documented moderate-deep gap (~4.5× at
2^-120 vs ~9.3× at 2^-400). OUTCOME: raising the SA polynomial from ORDER 3 to ORDER 5 is the first
NET-WIN speed lever since SA itself — measured 1.24× / 1.21× / 1.13× / 0.99× at 2^-120 / -218 / -271
/ -400 on the real RTX 3090, bit-exact, validated SwiftShader + real GPU. It targets EXACTLY the gap
(biggest win at the shallow-deep band, free at extreme depth) and stacks on everything.**

### The lever (measure-first, `tools/probe-saorder.mjs`)
SA (series.js) seeds dz at iteration `skip` via a polynomial in dc; it was ORDER 3 (`a·u + b·u² +
c·u³`). The skip N is the largest where, for every probe in the dc-box grid, the FIRST of two
constraints holds: (a) truncation `|SA−dz| ≤ tol·|dz|` (production tol 1e-10), or (b) the escape
guard `|dz| ≤ 0.25` (linear regime ends / rebase imminent). The open question: which binds? If (b),
order is moot (no dc-polynomial skips past a pixel's first rebase). If (a), MORE terms track the true
dz further → larger skip. **Measured: truncation ALWAYS binds at production tol**, and order 5 raises
the skip MOST at the moderate-deep band: skip% 81.6→86.2 (+4.7) @2^-120, +2.5 @2^-218, +1.9 @2^-271,
+0.5 @2^-400. The +gain shape is the inverse of the speed-gap shape — it lands where the gap is.
Math: D' = 2Z·D + (2AC+B²), E' = 2Z·E + (2AD+2BC); scaled d=D·R⁴, e=E·R⁵ stay O(1) like a,b,c.

### Why it is a real WALL-CLOCK win (not just skip%), and the worry that was wrong
The concern (companion-flagged earlier work): a near-flat ~8ms SA-on floor was attributed to
POST-REBASE re-growths, which SA does NOT skip (it seeds the LEADING run once). If that floor
dominated at 2^-120, a larger leading-run skip wouldn't move the clock. **It does move it** —
`tools/bench-saorder.mjs` on the real RTX 3090 (order 3 vs 5, both forced, fixed work):
| depth  | o3 skip% / ms | o5 skip% / ms | o5/o3 | o5 vs noSA |
|--------|---------------|---------------|-------|-----------|
| 2^-120 | 82.3% / 5.2   | 86.4% / 4.2   | 1.24× | 5.73×     |
| 2^-218 | 86.9% / 6.2   | 89.5% / 5.1   | 1.21× | 7.14×     |
| 2^-271 | 88.2% / 6.4   | 90.3% / 5.7   | 1.13× | 7.81×     |
| 2^-400 | 91.6% / 6.0   | 92.0% / 6.1   | 0.99× | 9.39×     |
So the leading-run tail (skip→first-rebase, computed per pixel) is a big enough slice of the
remaining work at the moderate-deep band that shaving ~1400 iters/pixel pays. At 2^-400 the +0.5%
skip ≈ the two extra fe Horner ops → exactly neutral (free, not a loss). Bit-exact: it can only be
NEUTRAL-or-POSITIVE, never a BLA-style net loss.

### The one trap (df64 seed precision at the shallow chaotic band) and the depth-adaptive fix
order-5's larger skip seeds at a LATER iteration where the scaled coeffs are larger, so the GPU's
df64 (~46-bit) in-shader Horner has a larger ABSOLUTE seed error. On a SHALLOW CHAOTIC region (the
2^-90 seahorse, esc-counts ~17k) that error amplifies through the post-seed continuation —
validate:gpu FAILED there with order-5 (meanΔsn 0.9→1.4, over the 1.0 gate; the mism FRACTION stayed
under gate, so it's precision, not an over-skip/missed-escape). BUT on the genuine chaotic deep
boundary coordinate it is df64-CLEAN across the whole band: `tools/probe-saorder-gpu.mjs` →
**0.000% mism, meanΔsn ~5e-4 at 2^-100..-130** (1000× under the gate). And in PRODUCTION the shallow
band (radius ≥ 2^-112) renders on the df64 engine, which does NOT apply SA at all — so the seahorse+SA
config is validation-only. FIX: **depth-adaptive order** (`series.js` SA_ORDER5_RADIUS = 2^-112 =
GPU_PERTURB_FLOOR, the df64→rescaled dispatch boundary) — default order 5 below it (where SA is the
production path AND df64-clean), order 3 above (shallow overlap; SA's gain there is small anyway).
`opts.order` overrides (the benches force a fixed order at any depth). This restores validate:gpu /
validate:gpu:real to ALL PASS while giving production the full order-5 win in the deep band.

### Validation (all green) + decision
- 40 unit (+1: order-5 skips strictly more than order-3 AND stays bit-exact; order-3 result has
  d=e=0). `crosscheck-sa` 0 mism, maxΔn 0 at 2^-120/-271 (skip 86.4%/90.3%). `validate:gpu`
  (SwiftShader) + `validate:gpu:real` (RTX 3090) ALL GPU VALIDATIONS PASSED.
- LEAN df64 barrier untouched (probe:barrier gates unaffected — no DF64_LIB edit). The order-5 seed
  rides the SAME default-on "Series approximation" toggle (one switch, CPU+GPU), automatically.
- Order 5 is the knee: probe-saorder shows order 7+ adds <2% skip at 2^-120 while risking df64
  precision exhaustion + GPU register pressure (companion-confirmed). Stopped at 5.
- New: `tools/{probe-saorder,probe-saorder-gpu,bench-saorder}.mjs`; npm `probe:saorder`,
  `probe:saorder:gpu`, `bench:saorder[:real]`. Edits: `src/math/{series,perturb}.js` (order 3|4|5 +
  depth-adaptive default + order-5 Horner seed), `src/gpu/glsl.js` (uSAm/uSAe [6]→[10], order-5 fe
  Horner), `src/gpu/renderer.js` (upload 10 coeffs), `test/gpu/harness.html` (thread series opts into
  benchPerturb so the order A/B works), `test/unit/series.test.mjs`. The moderate-deep band 2^-120 now
  ~5.7× vs no-SA (was ~4.6×); still short of 10× there (the chaotic post-rebase tail floor remains —
  BLA is the tool for it but a measured GPU net-loss), but a clean step toward it. The next structural
  levers are unchanged: fewer per-iter fetches / less warp divergence / off-GPU work (OffscreenCanvas).

---
## ⛔ INTERIOR DETECTION — MEASURED DEAD on the GPU (Spawn 23) — READ THIS BEFORE RE-PROPOSING IT

**Ask (continuing "optimize until 900% faster" — Danielle asked for a verdict: greenlight a rigorous
attracting-cycle interior detector, or lock in?). OUTCOME: a measure-first UPPER-BOUND experiment
REFUTES interior detection as a GPU lever. A PERFECT, FREE, zero-latency interior detector (the
absolute ceiling) buys only 1.3–1.55× across the deep band and 0.99× (nothing) at 2^-400. A REAL
detector is strictly worse. DECISION: do NOT build it; the current state is locked in. This is the
THIRD measured GPU dead-end (after region-adaptive SA and the lean kernel), all bound by the same
structural ceiling: the RTX 3090 deep frame is WARP-DIVERGENCE / escaper-tail bound, not bulk-work bound.**

### Why measure a ceiling instead of building the detector
Spawn 22 measured 61–90% of deep-frame pixels are INTERIOR (run to maxIter) consuming 65–91% of
post-SA work, and flagged interior detection as "the one remaining BIG lever" but DEFERRED it as a
research bet with three compounding uncertainties: (1) does pruning even help on a GPU that might be
divergence-bound not throughput-bound? (2) coordinate-dependent (pure filaments ≈ 0 interior);
(3) deep atom periods can be large → late detection. A rigorous detector is a multi-spawn build with
a per-iteration tax + GPU state. Instead of building on faith, build the CHEAP thing that bounds the
payoff: an **oracle prune**. The CPU oracle already classifies every pixel interior-vs-escaping
exactly; feed that mask to the shader and let oracle-interior pixels BREAK the loop early at a swept
iteration `T = skip + frac·(maxIter−skip)`. `frac=0` (bail at `skip`) is a perfect/free/zero-latency
detector = the absolute CEILING; it cuts ONLY interior pixels and leaves escapers intact, so GPU warp
divergence decides the real saving. ANY real detector is strictly below this (latency + per-iter tax +
fires later than `skip`). If the ceiling is ~1.0×, interior detection is dead with NO multi-spawn build.

### The result (`tools/bench-interior-oracle.mjs`, real RTX 3090, rescaled+SA order-5)
| depth  | interior% | baseline ms | **ceiling (frac=0)** | curve saturates by |
|--------|-----------|-------------|----------------------|--------------------|
| 2^-120 | 59%       | 5.32        | **1.55×**            | frac ≈ 0.10        |
| 2^-218 | 57%       | 5.51        | **1.29×**            | frac ≈ 0.25        |
| 2^-271 | **92%**   | 5.21        | **1.39×**            | frac ≈ 0.10        |
| 2^-400 | 65%       | 4.64        | **0.99× (nothing)**  | flat               |
The smoking gun is **2^-271: 92% interior, yet only 1.39×.** And the curves SATURATE early (cutting
interior pixels earlier than ~10–25% into the post-SA tail gains nothing). Mask-fetch overhead is noise
(`nomask` ≈ `frac=1` baseline), so the bound is clean. Correctness/alignment crosscheck (baseline vs
max-prune full-buffer sn): **escDiff=0 everywhere**, 0–3 inside-flips (the CPU/GPU inside-classification
boundary) — the oracle mask is pixel-aligned and the prune only touches interior pixels.

### The mechanism (why 92% interior ⇒ only 1.39×) — the escaper-tail bottleneck
`probe-interior` already showed the ESCAPING pixels run ~81% of the SAME post-SA budget — they escape
LATE (chaotic deep boundary), not early. On a GPU a warp runs until its SLOWEST lane. The late-escapers
are spatially SPRINKLED through the frame (boundary filaments threading the minibrot), so almost every
warp contains at least one, and that lane anchors the warp to ~0.8·tail no matter how early the interior
lanes bail. Pruning the interior bulk is therefore mostly INVISIBLE — the frame time floors at the
escaper-bound level. This is the IDENTICAL structural ceiling that killed BLA (Spawn 18: texture latency +
post-SA-tail warp divergence) and the lean kernel (Spawn 22: occupancy isn't the bottleneck): the deep
frame is bound by divergence + the genuine late-escaper tail, NOT by the prunable interior count.

### The Oracle-Gap argument (companion-LLM-vetted) + the verdict
Since a perfect, zero-latency oracle yields only ~1.4× in the HIGH-interior case, any real detector — which
adds (a) detection latency that pushes its firing point PAST the ~10–25% saturation window, and (b) a
per-iteration cycle-derivative tax on EVERY pixel including the 35–43% escapers that never benefit (extra
registers → lower occupancy, extra ALU) — logically trends toward 1.0× or BELOW, exactly like BLA. The
complexity + precision/stability risk of a rigorous attracting-cycle detector is not justified by a ceiling
this low. **VERDICT: dead on the GPU path. Locked in.** Do not re-propose without NEW hardware where the
divergence model differs (a low-divergence-penalty / huge-register arch) — and even then, re-run this bench
FIRST (it's the gate). PORTABILITY: the 1.3–1.55× ceiling is RTX-3090-specific; the *structural* argument
(escapers anchor warps) holds on any SIMT GPU, but the exact number is per-GPU.

### The ONE place it is NOT structurally dead: the CPU fallback path
On the CPU pool there is NO warp divergence — each pixel is independent, so pruning an interior pixel is
PURE saving (no escaper anchors it). Interior detection could give a real CPU speedup (interior = 60–66% of
post-SA CPU steps). BUT: the CPU path is the slow, rarely-hit FALLBACK (the GPU is the validated default for
the whole 2^-112…2^-600 band and correct on real HW since Spawn 9/20); SA already skips 89–92% of CPU
iterations; and a rigorous CPU detector STILL carries the per-iter derivative tax + coordinate-dependence
(0 on filaments). Low value — recorded as the only non-dead branch, not greenlit.

### What shipped (kept off, opt-in — same pattern as BLA/df64-escape/lean)
The measurement infra is preserved (validated, gated, bit-identical when off — `validate:gpu` +
`validate:gpu:real` ALL PASS with it present; 41 unit pass; the added shader IR did NOT shift the df64
collapse threshold, confirmed on the real RTX 3090). `src/gpu/glsl.js` perturbFragRescaled: `uPrune`/
`uPruneIter`/`uPruneMask` (uPrune=0 default short-circuits the fetch → production unchanged). `renderer.js`:
`uploadPruneMask()` + the unit-2 binding (off by default). `test/gpu/harness.html`: `benchPrune()` (builds
the exact CPU-oracle mask + sweeps the bail iteration + the sn crosscheck). New `tools/bench-interior-oracle.mjs`
(npm: add if re-run often; invoke `GPU=1 node tools/bench-interior-oracle.mjs`). This is bench-ONLY; it is
NOT wired into render.js/worker.js/viewer.js (no production path builds a prune mask).

---
## ✅✅ SA SAFETY-MARGIN MINIMIZATION — a NET WIN, the deep band reaches/exceeds 10× (Spawn 22)

**Ask (continuing "optimize until 900% faster"): close the moderate-deep gap. OUTCOME: the SA skip's
SAFETY MARGIN was over-conservative DEEP — shrinking it 0.05 → 0.02 (depth-adaptive, below 2^-112) is
the THIRD net-win lever (after order-5 SA and barrier-min), BIT-EXACT, measured 1.12× / 1.19× / 1.18× /
1.40× at 2^-120/-218/-271/-400 on the real RTX 3090. Stacked on SA this puts 2^-271 and below at ~10×+
(2^-400 ~13×) and lifts 2^-120 ~5.7× → ~6.4×. Shipped as the default. Measure-first throughout.**

### The lever (measure-first: `tools/probe-sabox.mjs`, `probe-interior.mjs`, `probe-samargin.mjs`)
`computeSeries` chooses the skip as `floor((1-marginFrac)·validN)` where validN is the largest iteration
the f64 probe-grid SA still tracks the true dz (tol 1e-10). The margin (shipping 0.05) exists for ONE
reason: the GPU seeds the SA in df64 (~46-bit), which can diverge "a touch sooner" than the f64 grid.
Two probes pinned that it's over-conservative DEEP:
- `probe-interior.mjs` (where does post-SA work go): at 2^-120 the post-SA budget is only ~4190 iters/
  pixel, and the 5% margin ALONE discards ~1380 of them (≈ a THIRD of the budget). (It also measured the
  frame is 61–90% INTERIOR pixels at the deep boundary coordinate — see the interior-detection note below.)
- `probe-samargin.mjs` (sweep marginFrac, GPU-SA-vs-no-SA-oracle bit-exact gate on the real RTX 3090):
  margin 0.05 → 0.02 → 0.01 → **even 0** stays mism 0.000% AND meanΔsn/maxΔsn IDENTICAL to 0.05 (the df64
  seed is ~1000× under the gate deep). i.e. the margin was buying ZERO precision deep. post-SA iters cut
  ~20–31% (margin 0.02) / ~27–39% (0.01). Why a per-pixel UNIFORM cut: every pixel skips more, so unlike a
  per-pixel prune (interior detection) the benefit is ROBUST to the GPU's throughput-vs-divergence binding.

### Why margin 0.02 (not 0 or 0.01) + depth-adaptive
The escape-guard worst case (an isolated early-escaper between grid probes) is covered SEPARATELY by the
worker's `skipCap` (coarse-pass min-escape), NOT this margin; the truncation worst case (the dc-box corner)
is sampled EXACTLY by the grid. So validN is already tight and the margin is belt-and-suspenders. Margin 0
is the edge (skip=validN; maxΔsn ticked up). **0.02** keeps a comfortable 2%/~550-iter buffer for UNTESTED
coordinates while capturing most of the win at ZERO measured precision cost; **0.01** (a bit more, identical
precision on the test coord) is documented as the further measured-safe option (`opts.marginFrac` overrides).
DEPTH-ADAPTIVE (`SA_DEEP_MARGIN = 0.02` below 2^-112 = SA_ORDER5_RADIUS, the df64-clean GPU-SA dispatch
floor; 0.05 above) — the shallow/moderate band is CPU-SA territory not swept here, and SA's gain there is
small. PORTABILITY CAVEAT (like the df64 barrier): bit-exactness is empirical-per-coordinate; `crosscheck-sa`
(CPU) + `validate:gpu[:real]` (the rs+SA sections use the default margin) are the regression gates.

### The win (`tools/bench-samargin.mjs`, real RTX 3090, margin 0.05 baseline)
| depth  | m=0.05 ms | m=0.02 | m=0.01 | 0.02× | 0.01× | with SA, vs no-SA (was→now @0.02) |
|--------|-----------|--------|--------|-------|-------|-----------------------------------|
| 2^-120 | 11.0 | 9.9 | 9.3 | 1.118× | 1.182× | 5.7× → ~6.4× |
| 2^-218 | 11.5 | 9.7 | 9.1 | 1.187× | 1.275× | 7.1× → ~8.5× |
| 2^-271 | 11.0 | 9.4 | 8.7 | 1.179× | 1.266× | 7.8× → ~9.2× (~9.9× @0.01) |
| 2^-400 |  6.6 | 4.7 | 4.4 | 1.402× | 1.522× | 9.4× → ~13× |
The win GROWS with depth (the margin is a bigger share of the smaller deep post-SA budget). CPU win too
(`crosscheck-sa` m=0.02: 2^-120 2.10×→2.23×, 2^-500 3.24×→3.50×). Bit-exact ⇒ strictly neutral-or-positive.

### Validation (all green) + what shipped
- `crosscheck-sa` MARGIN=0.02 AND 0.01: **mism 0, maxΔn 0** at 2^-50…-500 (CPU bit-exact). 41 unit (+1:
  deep default out-skips margin-0.05 + shallow keeps 0.05 + bit-exact). `validate:gpu` (SwiftShader) +
  `validate:gpu:real` (RTX 3090) ALL PASS — the rs+SA deep skips ROSE (2^-130 28213→29104, 2^-271 61550→
  63494) at mism 0.000%. Production flows it automatically: worker.js + render.js call computeSeries with NO
  marginFrac override → the depth-adaptive default applies to BOTH the CPU pool and the GPU deep render.
- Edit: `src/math/series.js` (SA_DEEP_MARGIN + depth-adaptive default). New tools: `probe-sabox`,
  `probe-interior`, `probe-samargin`, `bench-samargin` (npm `probe:sabox`/`probe:interior`/`probe:samargin`/
  `bench:samargin`). Order-5 SA + the lean df64 barrier untouched. The gap at 2^-120 is now ~6.4× (the
  chaotic post-rebase tail floor still bounds it; BLA is its tool but a GPU net-loss).

### Two dead-ends closed this spawn (measure-first, both NEGATIVE — recorded so no one re-treads them)
- **Region-adaptive SA skip** (a smaller centered dc sub-box → larger skip for central pixels):
  `probe-sabox.mjs` measured a quarter-scale box (1/16 the area) buys only **+1.4% skip @2^-120, +0.2%
  @2^-400** — the order-5 truncation cliff is too sharp (halving |dc| cuts the |dc|⁶ error 64× but the skip
  only extends a few iters because dz grows explosively at the cliff). Not worth the multi-draw complexity.
- **LEAN kernel / occupancy** (the companion-LLM's bet): the production deep shader compiles the never-taken
  BLA scan + df64-escape blocks; a GPU sets occupancy by a kernel's PEAK register count across ALL branches,
  so excluding them at build time (BIT-IDENTICAL in production config) MIGHT raise occupancy. BUILT it
  (`perturbFragRescaled({bla:false,df64esc:false})`, 407→278 lines; renderer `p.lean`; `tools/bench-lean.mjs`),
  VALIDATED bit-identical on SwiftShader + real GPU — but `bench:lean` on the RTX 3090 = **NEUTRAL-to-slightly-
  SLOWER (~0.93–1.00×, reproducibly ~0.8× at 2^-271)**. A strict code SUBSET being slower ⇒ the NVIDIA
  compiler's codegen for the smaller shader is no better and occupancy isn't the bottleneck on this big GPU
  (huge register file; it likely already excludes never-taken-branch registers from the occupancy peak). So
  FULL stays the production default (measured-fastest); LEAN is OPT-IN (`p.lean`), kept ONLY for future
  measurement on a register-constrained MOBILE GPU where occupancy might actually bind. Same keep-off-but-
  validated pattern as BLA/df64-escape.

### ⚠️→⛔ INTERIOR DETECTION — measured BIG *potential*, but Spawn 23 REFUTED it (see the "⛔ INTERIOR DETECTION — MEASURED DEAD" section above; ceiling 1.3–1.55×). The raw potential below is real; the WALL-CLOCK payoff is not, because of warp divergence.
`probe-interior.mjs` found that at the deep boundary coordinate, INTERIOR pixels (run to maxIter) are
**61% of the frame @2^-120 and 90% @2^-271**, consuming **65–91% of the post-SA work** — they (and late
escapers) all run nearly the full post-SA budget. So pruning interior pixels via period/attracting-cycle
detection is the one remaining lever with genuinely BIG upside HERE. BUT it is NOT pursued this spawn:
(1) it needs a rigorous attracting-cycle test (store a checkpoint + accumulate the cycle derivative; only a
PROVEN |multiplier|<1 prunes, so it's correctness-safe-by-construction — never changes an escaping pixel),
which is real GPU state/registers + a per-iter tax on the escaping common case; (2) the benefit is GATED on
the GPU being throughput-bound (so cutting SOME pixels' work helps) — unresolved without building it, and
the frame here is low-divergence (all pixels ~equally expensive), which cuts BOTH ways; (3) it is COORDINATE-
DEPENDENT — the companion-LLM (correctly) notes a pure-FILAMENT deep zoom has ~0 interior, so this win
appears near minibrots (like the test coord) and vanishes on filaments; (4) deep atom PERIODS can be large,
delaying detection. Net: a research bet, not a clean ship. The measurement + the safety-by-construction
design are recorded here for a future spawn (or for Danielle to greenlight) — it is the most promising
remaining big lever, and the SA-margin + order-5 wins are the clean, safe, shipped progress.

---
## ✅ DEEP FLOOR → 2^500 (Spawn 10) — the GPU now renders the extreme deep band

**Ask: "optimize the gpu code when deep zoom (2^500)." The 2^500 zoom (radius ~2^-500)
was BELOW the GPU dispatch floor (2^-340), so it fell back to the CPU pool — correct but
minutes-slow at maxIter ~125k. It now renders on the GPU rescaled/fe engine (~19s) and is
hard-validated correct.**

WHY IT WAS JUST A DISPATCH BOUND, NOT AN ARITHMETIC ONE: the fe/rescaled engines are
exponent-magnitude-agnostic. The fe exponent is a plain `int` (handles 2^-500 trivially);
the only float work is on NORMALIZED mantissas in [0.5,1). On the CPU side, render.js/worker
compute dc origin/step as NORMAL doubles (double exponent floor ~2^-1022, so 2^-500 dc is a
normal double), and feSplit hands the GPU (df64 mantissa, int exponent). So nothing in the
math floored at 2^-340 — that constant was just the last-validated depth. THE FIX: lower
`GPU_PERTURB_FE_FLOOR` 2^-340 → **2^-600** in render.js (covers the 2^500 target + margin).

THE VALIDATION (the careful part). The real risk the companion LLM flagged was correct in
spirit: not the zoom depth itself but the ITERATION COUNT — perturbation dz error
ACCUMULATES per iteration, and a 2^-500 boundary view needs ~125k iterations. Two things had
to be shown: (1) the fe arithmetic holds at 2^-500 exponent magnitudes, and (2) the
accumulated dz error stays bounded at 125k iters.
- (1) is cheap: validate-gpu.mjs exterior-patch cases extended to 2^-400/-500/-600 (fast
  escapes, maxIter 1200) — 0.000% mism on the real GPU AND SwiftShader. The exponent path is
  magnitude-agnostic, so passing at these magnitudes covers the whole band.
- (2) needed a GENUINE deep boundary coordinate (the old test coords are only ~380-bit
  decimal strings, so a 2^-500 patch around them is a trivially-uniform escaping region — NOT
  a chaotic boundary). **tools/gen-deep-coord.mjs** builds one by descending a filament to
  ~2^-520 in BigInt. KEY TRICK (first attempt fell off the boundary into an exterior atom
  below ~2^-300): HYBRID tracking — while the probe box is fully exterior (coarse early
  steps) chase the hottest ESCAPING tip; once it straddles, relocate onto the INTERIOR pixel
  NEAREST that tip (hug the boundary from the inside). The inward bias stops the center
  drifting to the exterior side. Result: clean "straddle" with ref.len = autoMaxIter all the
  way to 2^-520. **tools/probe-deep500.mjs** then compares the rescaled + fe GPU engines vs
  the CPU oracle on that coordinate at autoMaxIter: **0.000% escape-count mismatch FLAT from
  2^-120 (30k iters) to 2^-520 (130k iters)**, maxΔsn bounded ~4e-3, insideMism ~0. FLAT ⇒
  rebasing bounds the per-iteration accumulation; it does NOT grow with maxIter.
- WHY THE df64 REFERENCE DOESN'T DRIFT (the companion's other worry, resolved): reference.js
  computes the orbit in BigInt (EXACT) and SAMPLES each Z_n to df64 ONCE — it never iterates
  in df64, so each Z_n is independently 46-bit accurate with NO compounding. The 46-bit
  relative precision is depth-independent. So storing the reference as fe is likely
  unnecessary even far deeper (re-check only if chaotic noise ever appears below 2^-520).

END-TO-END PROOF: tools/shoot-deep500.mjs drives the real viewer to the deep coordinate.
Zoom 2^500 → engine gpu-perturb-fe, glitches=0, refLen 125254, ~19s, a clean seahorse-spiral
fractal (screenshots/deep500.png; 2^400 in deep400.png). The BigInt reference build is only
~0.43s, so the ~19s is GPU escape work — GPU-side perf genuinely matters at this depth.

PERF: AUTO-DROP SUPERSAMPLING below 2^-300 (SS_DEEP_CAP_RADIUS, viewer._effectiveSS). ss=2
runs the heavy fe/rescaled shader on ss²=4× the pixels, so a 2^500 frame at the default ss=2
would be ~75s; capping the EFFECTIVE ss to 1 keeps it ~19s for AA the fractal's density
mostly hides anyway. The user's ss SELECT is unchanged (this is the same mechanism as the
existing memory cap); the debug line shows "capped from 2× for depth". The cap is past the
2^-270 "ultra" bookmark, so normal deep exploration keeps full ss. (Companion LLM endorsed
the depth-aware default + surfacing it; a "Force High Quality" override toggle is a TODO.)

NEXT (deeper): the fe math holds toward the ~2^-1000 double-dc limit. To push the floor
below 2^-600, raise gen-deep-coord's TARGET, regenerate a deeper boundary coordinate, and
confirm probe-deep500 stays 0% below 2^-520 — then lower GPU_PERTURB_FE_FLOOR further.

---
## ✅ MOBILE STABILITY: WebGL context-loss recovery (Spawn 11)

**Ask: remaining UX/stability polish for mobile. Gap found: the viewer had ZERO handling of
WebGL context loss.** Mobile GPUs drop the WebGL context routinely — under memory pressure,
when the tab is backgrounded, or on a driver reset. Before this, a loss mid-render left the
strip-tiled draw loop issuing draw/flush/colorize on a dead context (silent no-ops → a frozen
or blank image) and **nothing ever recovered**: the engine never re-rendered, and because a
lost context's GL calls don't throw, even `_gpuFail` (which only fires on a thrown error)
wouldn't trip. A real, common, user-facing mobile failure with no recovery path.

THE FIX (two layers):
- **renderer.js** — `GpuRenderer` now listens for `webglcontextlost` / `webglcontextrestored`
  on its canvas. On lost: `e.preventDefault()` (REQUIRED — without it the browser will NOT fire
  `restored`), set `_contextLost=true`, and `_dropGLState()` (null EVERY cached GL handle:
  programs, VAO, sn-texture+FBO, reference texture, palette LUT — they're all invalid after a
  loss, and nulling them makes the existing lazy-init guards recreate them). On restored:
  `_initGLObjects()` (re-fetch extensions + rebuild the quad VAO), clear `_contextLost`;
  programs/textures/FBO are recreated lazily on the next render. New `get lost()`. `dispose()`
  sets `_disposed=true` + removes the listeners FIRST so the intentional `loseContext()` in
  dispose (e.g. the GPU-off toggle) doesn't trip the recovery path.
- **viewer.js** — `render()` gates GPU engine-selection on `!this.gpu.lost`, so while the
  context is lost every render routes to the **CPU worker pool** (correct, just slower —
  it's the validated oracle). `_onGpuContextLost()` cancels the in-flight render (gen bump +
  pool terminate) and, after a `GPU_LOST_GRACE_MS` (600ms) grace, re-renders on the CPU **iff
  the context still hasn't returned** — the grace avoids a wasteful CPU detour when the browser
  restores almost immediately (the common case) yet bounds how long the view sits frozen if it
  doesn't. `_onGpuContextRestored()` forces a palette-LUT re-upload (its texture was lost) and
  re-renders on the GPU. After `GPU_MAX_LOSSES` (3) losses it **gives up on the GPU for the
  session** (dispose + stay on CPU) rather than flip-flop a memory-starved device between
  engines (the companion LLM flagged this death-loop risk). `setUseGpu(true)` resets the
  give-up/loss state (a clean re-probe). `gpuInfo()` returns null while lost (debug honesty).

WHY DOUBLES/CPU ARE A SAFE FALLBACK HERE: the CPU worker engines are the ground-truth oracle
(validated to 2^400), so a lost-context render is never WRONG, only slower. The recovery just
restores SPEED once the GPU is back.

VALIDATION (headless, both SwiftShader AND real GPU via `WEBGL_lose_context`):
- `tools/probe-ctxloss.mjs` — proves the extension dispatches lost+restored events on this
  backend. KEY GOTCHA found: a tick is required BETWEEN `loseContext()` and `restoreContext()`
  — calling them back-to-back synchronously never fires `restored` (the lost event must dispatch
  first). The e2e + probe both honour this.
- `tools/probe-recover.mjs` (npm `probe:recover`) — drives the REAL viewer through the full
  cycle and asserts: (A) while lost → renders on a CPU engine; (B) restore → GPU returns,
  image non-blank (shallow gpu-naive); (C) same at DEEP zoom (gpu-perturb — exercises the
  reference-texture re-upload on restore, not just the naive path); (D) 3 losses → gives up on
  the GPU, subsequent renders stay CPU. ALL PASS on SwiftShader and GPU=1 (Vulkan, RTX 3090).
- e2e `gpu.spec.mjs` "recovers from WebGL context loss: CPU while lost, GPU once restored"
  (mobile + desktop) — the CI gate.
- No regression: 31 unit, validate:gpu ALL PASS (shaders/draw path untouched — only the
  renderer lifecycle changed), full e2e green.

NEXT (related, deferred): an OffscreenCanvas GPU worker (M6 FUTURE) would move the deep draw
off the main thread entirely; it also reduces context-loss exposure (a backgrounded tab's
worker canvas is less aggressively reclaimed). The recovery here is the cheap, correct fix that
works today regardless. A subtle visual "CPU mode" badge during the lost window (companion's
suggestion) is a possible nicety — currently the status line shows "recovering GPU…" then the
CPU engine name, which is honest enough.

---
## ✅ GLITCH DEBUG OVERLAY + honest GPU glitch count (Spawn 12)

**Ask: remaining polish / architectural gaps. Gap found: the GPU glitch count was a fake 0,
and there was no way to SEE where the perturbation might be failing.** The GPU perturbation
shaders have always computed a Pauldelbrot glitch flag into the sn texture's `.b` channel
(`|z|^2 < tol·|Z_m|^2`), but the viewer passed `glitchTol = 0`, so the flag was STRUCTURALLY
never set (`if (uGlitchTol > 0.0 …)` gated it off) and every GPU render reported `glitches: 0`
regardless of reality — while the CPU path counts a real number. A dishonest readout and the
last open M7 item (glitch overlay), NEXT #4.

THE FIX (a debug toggle, off by default — "Glitch overlay" checkbox):
- **glsl.js COLOR_FRAG** — new `uShowGlitch` uniform. The supersample loop now also ORs the
  `.b` glitch flag across the ss×ss block; when `uShowGlitch != 0` and any subsample is
  flagged, the output color is blended 0.6 toward magenta (the fractal stays visible under
  the tint). Reads one extra channel from a texel it already fetched — no extra fetch, and
  the rendered color is byte-identical when the toggle is off.
- **renderer.js** — `colorize()` sets `uShowGlitch` from `opts.showGlitch`; new
  `countGlitches()` reads back the RGBA32F FBO and counts `.b > 0.5` texels (at COMPUTE res,
  ss× display — matching the CPU worker's per-compute-pixel count). A full readback (slow),
  so it runs ONLY when the overlay is on.
- **viewer.js** — `showGlitches` (default false) + `glitchTol` (default 1e-6, the standard
  Pauldelbrot ratio). `setShowGlitches(on)`: turning ON re-renders with the diagnostic
  enabled; turning OFF only drops the tint, so it's a cheap `_recolorGpu()` when a GPU frame
  is current. `_renderGpuPerturb` passes `glitchTol: showGlitches ? this.glitchTol : 0` and
  `fastSkip: showGlitches ? 0 : 1` (the fast-skip is provably safe at tol 1e-6 — it only
  skips iterations where `|Z_m|>2|dz|`, so `mag2≈|Z_m|²` is never `< 1e-6·|Z_m|²` — but we
  compute exhaustively when the user explicitly asks to see glitches). `_finishGpu` reads the
  real count via `countGlitches()` when the overlay is on and the engine is a `gpu-perturb*`
  one (naive/shallow has no perturbation reference, hence no glitch concept).
- **main.js + index.html** — a "Glitch overlay (debug)" checkbox wired to `setShowGlitches`;
  added to `syncControls`. NOT persisted in the URL hash (it's an ephemeral debug switch).

WHY THIS IS SAFE / PURELY DIAGNOSTIC: the `.b` glitch flag never feeds back into `dx/dy/m/n`
or the rebase decision (read the perturb shaders: the flag is set then only written to the
output vec4). So enabling `glitchTol` and disabling `fastSkip` change the FLAG but NOT the
escape counts or the rendered fractal — confirmed by validate:gpu staying ALL PASS
bit-for-bit and the e2e's "overlay off ⇒ tint gone, fractal returns" check.

VALIDATION:
- 31 unit pass; **validate:gpu ALL PASS** (the COLOR_FRAG + renderer changes are
  non-regressive — escape counts untouched).
- New e2e `gpu.spec.mjs` "glitch overlay: honest GPU glitch count + magenta tint when
  forced": deep render → enable overlay (asserts a finite count is read back) → force a HUGE
  tol (`viewer.glitchTol = 1e9`) which flags essentially every escaping pixel, asserting the
  count > 100 AND magenta pixels appear (proves flag → readback → tint end-to-end) → toggle
  off (tint gone). PASSES on SwiftShader (both projects) AND the real GPU (Vulkan/RTX 3090,
  1.8s — the diagnostic uses the already-barriered ds_* ops + highp samplers, so no df64
  collapse). The forced-tol path is the gate because rebasing is genuinely glitch-free, so
  the realistic-tol count is ~0 — exactly the point of the tool: confirm glitch-free, catch
  the rare exception.

NEXT (related): the CPU path already reports a real glitch count but has NO visual overlay
(the worker only counts, doesn't keep a per-pixel mask). Adding one would mean posting a
glitch mask back per band + tinting in colorizeRegion/colorizeBlocks — more invasive, on the
hot CPU path, and the GPU is the default where deep glitches actually appear, so it's
deferred. `viewer.glitchTol` is exposed (default 1e-6) as a tuning hook if a power-user
control is ever wanted.

---
## ✅ FORCE HIGH QUALITY toggle + PERF-BUDGET e2e (Spawn 13)

**Ask: remaining polish (unsupervised). Two small, self-contained items closed: the
"Force High Quality" override (the AGENDA's flagged "smallest open UX item") and the last
unchecked M8 box — performance-budget e2e assertions.**

### Force High Quality (override the deep ss auto-cap)
Spawn 10 added SS_DEEP_CAP_RADIUS (2^-300): below it, the EFFECTIVE supersample drops to 1
so a 2^500 frame is ~19s instead of ~75s (ss² runs the heavy fe/rescaled shader on 4× the
pixels for AA the fractal's density mostly hides). That's the right DEFAULT, but a user doing
a deliberate slow deep CAPTURE may want full AA anyway. The toggle (`viewer.forceHighQuality`,
default false; "Force high quality" checkbox) makes `_effectiveSS()` skip the depth cap.
- **It only overrides the DEPTH cap, NOT the hard memory/texture caps** (MAX_COMPUTE_DIM
  8192, MAX_COMPUTE_PIXELS 12e6). Those prevent OOM / exceeding the GL max texture size, so
  forcing past them could crash the GPU — they always apply. (The e2e asserts exactly this:
  forced + a 6000²·ss4 backing still caps below 4×.)
- `setForceHighQuality(on)` re-renders so the change takes effect immediately (a deep view
  re-renders at full ss; shallow is a visual no-op — the cap never engaged there — but we
  re-render for parity with setSupersample). The debug line shows "(HQ: depth cap overridden)"
  when the override is actually in effect (forceHighQuality && radius < cap && effSS>1).
- **NOT persisted in the URL hash** (deliberately, like the glitch overlay): it's a per-session
  capture choice. Baking it into a shared bookmark would hand the recipient a silently ~4×-slower
  deep render. SS_DEEP_CAP_RADIUS is now `export`ed from viewer.js so main.js's debug line can
  reference the same constant (no duplicated magic number).

### Perf-budget e2e (test/e2e/perf.spec.mjs) — catastrophic-regression guards, by design
The honest constraint: the e2e backend is SwiftShader (CPU software GL) and this host is
often contended (the companion LLM shares the cores), so TIGHT wall-clock budgets WOULD flake
(NOTES already documents a contention timeout flake). So these are deliberately LOOSE — they
catch an order-of-magnitude regression (workers not spawning → single-threaded; a maxIter
blow-up; an O(n²); GPU init hanging), not a 20% drift. Two tests, on the FAST home/shallow
path (the default GPU engine + the CPU fallback) that nothing else times; deep-path
catastrophic regressions are ALREADY guarded by the explicit timeouts on the deep-zoom / Go
e2e tests.
- `perf: home view — time-to-first-pixel and full-frame within budget`: polls the canvas
  until it shows real structure (>2 distinct colors in an 80×80 corner = the first painted
  coarse pixels) for **time-to-first-pixel**, then `waitDone` for the **full frame**; asserts
  both under budget AND first-pixel ≤ full (progressive: the coarse paint never lands after
  the complete frame). Observed here: first-pixel ~0.7–1.0s, full ~0.7–1.0s.
- `perf: CPU worker-pool home render within budget`: forces `setUseGpu(false)` and budgets the
  CPU-pool home frame (the validated fallback/oracle path). Observed ~0.5–0.6s (poolSize 12).
- Budgets (FIRST_PIXEL 8s / HOME_FULL 12s / CPU_HOME 15s) sit ~8–24× over observed, with the
  reasoning + observed numbers documented inline so a future spawn can re-tune. To eyeball the
  live timings: `npx playwright test perf.spec.mjs` prints `[perf] …` lines.

SCOPE/SAFETY: changes are confined to viewer.js (state + `_effectiveSS` + setter + export),
main.js (import + debug note + checkbox wiring + syncControls), index.html (checkbox), and the
two test files. NO shader/renderer/render-path code touched, so the GPU correctness gates
(validate:gpu, the 31 unit tests) are untouched. 31 unit pass; Force-HQ e2e + perf e2e green
on both projects (SwiftShader).

NEXT (remaining polish, in rough priority — companion LLM's ranking): a CPU-path VISUAL glitch
overlay (the worker would post a per-pixel glitch mask per band → tint in colorizeRegion/
colorizeBlocks; only built when showGlitches is on, so default renders stay byte-identical —
mirrors the GPU overlay's gating); then a low-power / battery mode (cap DPR / maxIter / ss on
battery). Both are deferred, not blocked. **(Both DONE in Spawn 14 — see below.)**

---
## ✅ CPU GLITCH OVERLAY + LOW-POWER MODE (Spawn 14)

**Ask: the two remaining polish items the companion LLM had queued (NEXT #5 a/b). Both are
small, self-contained, and now done + tested. NO shader/renderer/GPU-math change — the GPU
correctness gates (validate:gpu, 31 unit) are untouched.**

### CPU-path visual glitch overlay (mirrors the GPU sn `.b` overlay)
The GPU overlay (Spawn 12) tints Pauldelbrot-flagged pixels magenta and reads back a real
count; the CPU path counted glitches but had NO visual overlay (the worker only summed a
count, never kept a per-pixel mask). Now it does, gated so default renders are byte-identical.
- **worker.js** — `makeEscape`'s closure records each pixel's flag in `state.lastGlitched`
  (one assignment; never affects sn). When `params.showGlitch` is set, `coarsePass` and
  `renderBands` build a per-pixel **Uint8 mask** (1 = flagged) alongside sn and transfer it
  (coarse → `refReady.coarse.glitch`; band → `band.glitch`). `params.glitchTol` is threaded
  into `escapePerturb(…, 1<<16, glitchTol)` so the count/mask are tunable like the GPU
  `uGlitchTol` uniform (undefined → escapePerturb's standard 1e-6, so the default count is
  unchanged). The mask is built ONLY when requested → default renders allocate/transfer/compute
  exactly as before, and sn is bit-identical (the diagnostic is independent of escape/rebase).
- **palette.js** — `colorizeRegion`/`colorizeBlocks` take an optional `glitch` mask; a flagged
  pixel is blended `tintGlitch` = 0.6 toward (255,0,255), the SAME mix the GPU COLOR_FRAG uses
  (`mix(col, vec3(1,0,1), 0.6)`). Omit the mask (default) → byte-identical output (the tint
  branch is `if (glitch && glitch[idx])`, skipped when null).
- **viewer.js** — CPU render() allocates `this._glitch` (compute-res mask) only when
  showGlitches; passes `showGlitch`/`glitchTol` in the computeRef message; the band handler
  copies each band's mask into `_glitch` and tints; `_recolor` re-tints from the cached mask
  when the overlay is on (so a palette change keeps the tint) and passes null when off (clean
  fractal returns). `setShowGlitches` now has a CPU branch: turning OFF does a cheap `_recolor`
  (drop the tint) instead of a full re-render — mirrors the GPU `_recolorGpu` path.
- **SS NOTE** (deliberate, companion-endorsed): with ss>1 the CPU tints flagged *subsamples*
  at compute-res, then box-averages down — so a partially-flagged output pixel ends up
  partially magenta (a glitch-intensity heatmap), whereas the GPU ORs the flag across the
  ss-block and tints the whole output pixel. Identical at ss=1; the soft heatmap is arguably
  *better* for debugging, so it was kept rather than matching the GPU's OR semantics.
- VALIDATION: 31 unit; golden-fingerprint + "deep zoom glitch-free" e2e UNCHANGED (proves the
  default CPU render is byte-identical); new e2e `viewer.spec.mjs` "CPU glitch overlay: per-pixel
  mask tints magenta when forced…" (force CPU pool + deep view + huge tol → magenta + count>100,
  off → tint gone). GPU glitch e2e + validate:gpu still pass (GPU path untouched).

### Low-power / battery-saver mode
A manual "Low power" checkbox (`viewer.lowPower`, default off) trades detail for far fewer
pixel-iterations per frame. The per-sample MATH is unchanged — just fewer, coarser samples;
correctness is never affected. Three caps (constants near the top of viewer.js):
- **DPR + backing** (`resize()`): DPR ceiling `min(dpr, 1)` (vs 2) and `LOW_POWER_MAX_BACKING`
  700 (vs `MAX_BACKING` 1100) → roughly 2–4× fewer pixels on a typical hi-DPI phone/desktop.
- **Supersampling** (`_effectiveSS()`): forced to 1 (no ss² multiplier). Low-power WINS over
  Force HQ (it's the explicit energy choice).
- **Auto iterations** (`_autoIter()`, the new single source for the auto budget — all
  `autoMaxIter(this.radius)` call sites route through it): `min(autoMaxIter, LOW_POWER_MAX_ITER
  =2000)`. A user-typed maxIter goes through `setMaxIter` and is NOT capped — only the AUTO
  budget is, so an explicit deep-detail request is never silently overridden.
- `setLowPower(on)` recomputes the auto budget then `resize()`s (re-derives backing under the
  new caps + re-renders). Debug line shows "· low power" on the backing row.
- **Battery auto-detect** (main.js `initBatteryDetect`): if `navigator.getBattery` exists,
  auto-enable low-power while discharging AND `level <= 0.2`, and lift it when charging /
  recovered — but ONLY until the user touches the toggle (`lowPowerManual` latches it off; we
  never fight a manual choice). The API is absent in many browsers (incl. headless Chromium,
  Firefox) → it's a progressive enhancement; the manual toggle is the real control. NOT
  URL-persisted (per-device energy state, like Force HQ / the glitch overlay).
- VALIDATION: new e2e "low-power mode caps backing resolution, supersampling, and the auto
  iteration budget" — checks the auto-iter cap purely via `_autoIter()` at a deep radius (no
  expensive render), then toggles low-power for real on the FAST home view and asserts backing
  shrinks + effSS→1, and both restore on toggle-off. (Lesson: radius 1e-7 is GPU-*perturb*, not
  naive — `gpuEngineForRadius` floors naive at 2^-2 — so a full render there at restored ss=2
  times out under SwiftShader; keep low-power render assertions on the home view.)

---
## ✅ SERIES APPROXIMATION — CPU path (Spawn 15)

**Ask (stretch goal): a mathematical enhancement — series approximation to skip the initial
perturbation iterations.** SHIPPED on the CPU perturbation path, default on, bit-exact,
validated to 2^-500. The GPU port (the headline-speed win) is the documented next step with
this CPU path as the validated oracle.

### What it is + why it's a big deep-zoom win
The delta recurrence dz_{n+1} = 2 Z_n dz_n + dz_n² + dc, expanded as a power series in dc
(Z_0=0, dz_0=0), has coefficients dz_n ≈ A_n dc + B_n dc² + C_n dc³ where
A_{n+1}=2Z_n A_n+1, B_{n+1}=2Z_n B_n+A_n², C_{n+1}=2Z_n C_n+2A_n B_n. At DEEP zoom dc is
tiny, so dz stays in the LINEAR regime (dz≈A_n dc, quadratic negligible) for almost the
whole orbit — the higher terms only bite near escape. So one polynomial seed lets EVERY
pixel start at iteration N instead of 0. Measured skip (tools/probe-sa.mjs, the raw
corner-truncation upper bound): 29% at 2^-50, **94% at 2^-120, 97% at 2^-271..-500**. The
realized (safety-capped) skip + speedup (tools/crosscheck-sa.mjs, full render on the deep
boundary coordinate): **2^-120 82%/2.6×, 2^-500 92%/2.7× — 0 escape-count mismatch FLAT
2^-50…2^-500.** It is ORTHOGONAL to rebasing: rebasing manages precision (glitches), SA
manages iteration COUNT (speed). Starting at N with dz=SA(dc) is exact — the true value
z=Z_N+dz_N is representation-independent; the per-pixel rebased iteration continues from N
normally (m=N).

### The overflow trap (load-bearing) — SCALED coefficients
Raw A_n is the orbit derivative; near a deep boundary point it grows past 2^1024 and
OVERFLOWS a double. Fix: u=dc/R (R=radius, |u|~O(1)) and track a_n=A_n·R, b_n=B_n·R²,
c_n=C_n·R³: a_{n+1}=2Z_n a_n+R (a_1=R), b_{n+1}=2Z_n b_n+a_n², c_{n+1}=2Z_n c_n+2a_n b_n.
dz_n≈a_n·u+b_n·u²+c_n·u³ (HORNER, ((c·u+b)·u+a)·u). The scaled coefficients stay O(|dz_n|)
(≤~O(1)) → representable in doubles. src/math/series.js `computeSeries`.

### Skip selection — TWO failure modes, TWO guards (the careful part)
A probe GRID over the dc box iterates the FULL non-rebased dz; the skip is the largest N at
which the SA polynomial still tracks every probe. A probe ends its validity at the first of:
(a) **truncation**: |SA−dz| > tol·|dz| (relTol **1e-10** — load-bearing, see below); or
(b) **escape guard**: |dz| > 0.25 (left the linear regime → escape/rebase imminent). PLUS a
third, independent guard in the worker:
(c) **coarse-pass min-escape cap**: the worker's dense step-8 no-SA coarse pass already
computes real escape times; the skip is hard-capped at their MINIMUM, so it can never pass
an isolated early-escaping pixel that falls BETWEEN the grid probes. (`skipCap` arg.)
A 5% safety margin backs off the final skip.
- **Why tol must be 1e-10, not 1e-3.** At the SHALLOW band (2^-38..-50) dc is larger so the
  3-term series has real truncation; a loose tol over-skips and the tiny seed error AMPLIFIES
  through the chaotic continuation (1e-3 → counts off by tens at 2^-50). 1e-10 keeps the
  chosen skip bit-exact at every depth. Deep zoom is essentially tol-independent (linear term
  dominates → series exact until it breaks abruptly near escape).
- **Why the grid isn't enough alone.** Probe density does NOT help against an isolated
  early-escaping pixel that no grid point lands on (tested: 13×13 → 49×49 gives the SAME
  skip). The coarse-pass cap (c) is the real defense; the grid handles truncation + smooth
  escapes.

### The ill-conditioned-pixel finding (the project's own philosophy, re-confirmed)
At 2^-41 (seahorse) SA differed from the no-SA render at ONE pixel by 484 iters. NOT a bug:
BigInt arbitration → that pixel BigInt=2778, **no-SA double=2870 (off by 92!)**, SA=2386;
EVERY neighbor matches BigInt exactly. It's a measure-zero ILL-CONDITIONED boundary pixel
where NO double method is right (the same effect NOTES documents for df64-vs-double /
naive-vs-perturb). SA never corrupts a WELL-conditioned pixel — proved by the unit test that
BigInt-checks every differing pixel. So the gate is the **bulk fraction**, not max Δn
(NOTES: "validate on bulk metrics, not max"). Deep zoom is bit-exact (0 mism) because the
deep field is uniformly well-conditioned at that scale.

### Integration (CPU only — the GPU deep path does NOT use SA yet)
- src/math/series.js (new) — computeSeries + saStartDelta. Pure, Node+browser.
- src/math/perturb.js — escapePerturb gains an optional `sa` arg (7th); seeds dz via Horner,
  starts the loop at n=skip+1, m=skip. With sa null the path is BIT-IDENTICAL to the original.
- src/math/render.js — renderImage gains opts.series (used by the tools/tests).
- src/worker.js — computeRef computes the series ONCE on worker[0] AFTER the no-SA coarse
  pass (so the coarse min-escape caps the skip), stashes it in `params.sa` which already
  travels to every tile worker; makeEscape threads it to escapePerturb. Disabled under the
  glitch overlay (so the diagnostic sees every iteration — mirrors the GPU dropping fast-skip).
- src/viewer.js — `this.series` (default on) + `setSeries`; threads `series` into the
  computeRef message; captures `_saSkip` from refReady; reports it in the done status.
- src/main.js + index.html — "Series approximation" checkbox (default on) + debug line
  "SA skip N (P%)". NOT URL-persisted (a pure-speed internal, like the other debug toggles).
- VALIDATION: 34 unit (+3 series.test.mjs: bit-exact 2^100, bulk+BigInt-arbiter 2^41,
  declines-when-degenerate); new e2e "series approximation: CPU deep render engages a skip"
  (forces CPU, asserts saSkip>0 + structured, toggles off → saSkip 0, same fractal);
  tools/crosscheck-sa.mjs (0 mism 2^-50…-500, the speedup table); validate:gpu unchanged
  (shaders/renderer untouched; escapePerturb backward-compatible). Default home render is
  byte-identical (home = naive engine, SA never invoked).

### NEXT — GPU PORT (the headline-speed win; CPU path is now the oracle)
The 2^500 GPU frame is ~19s, almost all escape iterations — SA would cut it ~10×. The port:
- Pass uniforms: `uSASkip` (int) + the scaled coeffs a,b,c. The seed dz=a·u+b·u²+c·u³ is
  computed PER PIXEL in-shader (Horner), u=dc/R (O(1), df64-representable). Then start the
  escape loop at iteration uSASkip with m=uSASkip, dz=seed (the loop/strip-tiling already
  parameterize a start; offset the returned count by the skip).
- **Representation**: for the df64 band (2^-2..2^-112) the seed magnitude ≥~2^-126, so df64
  coeffs/seed suffice. For the deep fe/rescaled band (2^-112..2^-600) the coeffs and seed can
  be ~2^-100 → need **floatexp** coeffs (the FE_LIB already exists) and an fe Horner. This is
  the delicate part — one exponent-normalization slip in the u²/u³ terms makes false chaotic
  pixels. Validate per-op like the df64 barrier work did.
- **Compute the coefficients where?** Cleanest: compute a,b,c on the CPU (worker, alongside
  the reference — it already runs computeSeries) and pass the chosen-N coeffs as uniforms (a
  handful of floats). The skipCap from the coarse pass still applies (worker already has it).
- **Gate**: extend tools/validate-gpu.mjs with an SA section (GPU-with-SA vs the CPU oracle,
  must match the no-SA mism numbers) + a crosscheck (GPU SA-on vs SA-off, like crosscheck-sa)
  + run on the REAL GPU (GPU=1). The CPU computeSeries is the bit-exact reference.
- Keep SA OFF for the GPU until validated on real hardware (gpuEngineForRadius unchanged); the
  CPU SA already speeds the deep fallback/oracle today.

---
## ✅ GPU SERIES APPROXIMATION + NO-BLACK-SCREEN PROGRESSIVE REVEAL (Spawn 16)

**Ask (2026-06-30): "optimize the shader further, and don't have it render black screen
instead replace low res with high res as it comes in."** Both shipped + validated on the
REAL RTX 3090 (Vulkan) AND SwiftShader.

### 1. No black screen — high-res tiles REPLACE the preview (viewer.js)
The strip-tiled GPU escape pass (Spawn 7) revealed top-to-bottom, but `_drawTiledEscape`
did a FULL `colorize` + FULL `_blitGpu` every strip, so the not-yet-computed rows (cleared
to interior sn=-1) painted as the interior color (BLACK), wiping out the gesture preview the
instant the first strip landed — the "black screen / void." THE FIX: blit ONLY each strip's
own display rows (`_blitGpuStrip(y, h)`); the uncomputed rows keep showing the scaled preview
/ prior frame, so the high-res tiles replace the low-res preview as they arrive. The color
pass flips Y (COLOR_FRAG `by = csize.y-(yo+1)*uSS`), so compute rows [y,y+h) map to display
rows [y/ss,(y+h)/ss) at the SAME offset — strips are ss-aligned so the division is exact.
- FINAL IMAGE UNCHANGED: the last strip covers the bottom, so the complete frame is identical
  (golden-fingerprint + deep-glitch-free e2e still pass byte-for-byte). Only the intermediate
  frames differ. `clearSn` is kept (defines the FBO's unwritten rows so colorize is well-formed;
  those rows are simply never blitted while uncomputed). `_blitGpu` (full) stays for `_recolorGpu`.
- e2e gpu.spec.mjs: "strip blit replaces only its rows" (deterministic — sentinel-fill the
  canvas, blit a top strip, assert the bottom is still the sentinel) + "multi-strip render keeps
  the preview below the first tile" (hooks _blitGpuStrip to sample a below-strip pixel at the
  first blit — proves the loop preserves the preview). Both projects.
- NOTE: a coordinate JUMP (no gesture preview) shows the prior frame in the uncomputed region
  instead of black — better than a void, still transient. A genuine low-res-of-the-NEW-view
  preview for the GPU path (a coarse downscaled pass before the strips) is a possible follow-up;
  not needed for the zoom case the ask is about.

### 2. GPU series approximation — the headline ~10× deep-zoom shader win
The deep rescaled engine now SEEDS dz at iteration `skip` via an in-shader floatexp Horner
(uSASkip>0), skipping the leading iterations every pixel ran. Measured on the RTX 3090
(tools/bench-sa.mjs, pure GPU escape time on the deep boundary coordinate): **4.1× at 2^-120
(82% skip), 6.7× at 2^-271 (88%), 9.3× at 2^-400 (92%)** — bigger the deeper you go. CAPSTONE
(tools/shoot-deep500.mjs, the real viewer at the 2^500 target): the frame that was **~19s drops
to ~4.2s** wall-clock (engine gpu-perturb-fe, glitches 0, refLen 125254; 2^400 ~3.5s) — the
fixed BigInt ref build + plumbing dilute the ~9-10× GPU-escape win to ~4.5× end-to-end. Live-
viewer end-to-end (probe-sa-viewer.mjs):
2^-271, 600², SA on vs off → **0.00% picture diff, 2.43× wall-clock** (incl. the fixed BigInt
ref build) — the GPU-escape portion is the ~6.7×; the ref build + plumbing dilute it at this size.

- **THE HORNER IS NOT DELICATE** (the NOTES-15 worry). It reuses the already-validated FE_LIB
  ops (fe_mul/fe_add — barriered + highp), so there's NO hand-rolled exponent juggling: the
  library normalizes every op. `dz = ((c·u+b)·u+a)·u`, u = dc·invR (all fe). Coeffs a,b,c
  (scaled, O(|dz_skip|) ~ 2^-270 deep) + 1/R passed as floatexp uniforms (uSAm[6]/uSAe[6],
  uInvRm/uInvRe). uSASkip==0 reproduces the no-SA path bit-for-bit. src/gpu/glsl.js
  perturbFragRescaled; renderer `_setSA`; the seed sets m=n=uSASkip (SA is valid only in the
  pre-rebase regime, so reference index = iteration) and getZ(skip).
- **THE CAP DECISION (load-bearing — why no coarse pass on the GPU).** The CPU SA has THREE
  guards: grid truncation, escape-guard (|dz|>0.25), and the worker's step-8 coarse-pass
  MIN-ESCAPE cap. tools/probe-sa-cap.mjs MEASURED the cap: on a structured coordinate (seahorse
  2^-22..-90) it is **NON-BINDING at every depth** — the skip is identical with the cap, without
  it, and with a PERFECT full-res min-escape cap; the ~20-70 differing pixels are IDENTICAL in
  all three and are the measure-zero ill-conditioned boundary pixels (no-SA double also wrong
  there). So the grid+escape-guard already holds the skip below the earliest escape; the cap
  only matters for a sub-grid isolated early-escaper, which doesn't occur at these scales (and is
  absent at deep zoom — uniform field). So the GPU worker computes computeSeries with
  skipCap=Infinity (grid+escape-guard only) — ~tens of ms even at 2^500 (no coarse pass, which
  would cost ~5s at extreme depth and bind nothing). The REAL gate is empirical: GPU-with-SA vs
  the CPU oracle must stay in the GPU's existing df64 mismatch envelope.
- **VALIDATION (all green, SwiftShader AND real RTX 3090, IDENTICAL numbers between them):**
  validate-gpu.mjs new "perturb rescaled + SERIES APPROXIMATION" section — GPU-rs-SA vs the
  no-SA CPU oracle: seahorse 2^-70 mism 0.358% (no-SA rs: 0.347%), 2^-90 0.922% (vs 0.977%) —
  SAME envelope, SA adds no error; deep boundary 2^-130 skip 84% **0.000% mism**, 2^-271 skip
  88% **0.000%**. probe-sa-viewer.mjs (live app, real GPU): 0.00% picture diff SA on/off. The
  fe Horner survives the NVIDIA compiler because it's built on the barriered/highp ds_*/fe ops
  (the Spawn 8/9 fixes) — no new precision cliff.
- **WIRING**: viewer `_startGpuPerturb` sends `series` only for the fe band + not under the
  glitch overlay (mirrors CPU SA gating); worker computeRef computes params.sa (no coarse cap);
  `_renderGpuPerturb` passes `sa` to renderPerturbRescaled; `_finishGpu` reports the real
  saSkip. SA rides the EXISTING default-on "Series approximation" toggle (now one switch for
  both CPU and GPU). df64 (shallow) band unaffected (doesn't apply SA; renders fast already).
- **NOT DONE / NEXT**: (a) the fe engine (perturbFragFloatexp, the oracle) does NOT have the SA
  seed — only the shipping rescaled engine does; add it if fe ever becomes a dispatch target
  again. (b) df64-band GPU SA — skipped deliberately (fast already + moderate-zoom ill-conditioned
  risk). (c) push bench-sa to 2^500 (extrapolates to ~10×+). (d) a coarse GPU preview pass for
  jumps (see §1 NOTE).

---
## ✅ BLA — bivariate linear approximation, CPU oracle (Spawn 17) — the path to sustained 10×+

**Ask (2026-06-30): "generate new ideas to optimize further, benchmark, repeat until 900% faster"
(= 10× over the no-SA GPU baseline).** Where we were: GPU series approximation (SA, Spawn 16)
gives 5.24× / 6.65× / 9.58× at 2^-120 / 2^-271 / 2^-400 (live RTX 3090 bench-sa). Near 10× ONLY
at extreme depth; ~5× at the shallower deep band. The SA-on render time is a near-FLAT ~8ms floor:
SA skips the LEADING linear run once, but the deep orbit is a SEQUENCE of linear runs — every
Zhuoran rebase resets dz to a small true value and it re-grows linearly AGAIN before the next
rebase/escape. Those post-rebase re-growth phases are the ~8ms floor and SA cannot touch them.

**BLA (Kalles-Fraktaler / Zhuoran) skips runs of linear iterations THROUGHOUT the orbit**, incl.
after every rebase — exactly the floor. This spawn built + validated it on the CPU (the project's
proven CPU-oracle-first discipline: SA was Spawn 15 CPU → Spawn 16 GPU; BLA is the same shape).
The CPU work-reduction directly bounds the eventual GPU-escape speedup. (Companion LLM consulted
twice: strongly ranked BLA as THE lever over the df64-escape micro-opt and higher-order SA, and
confirmed the radius/merge formulas + the escape/rebase-skip safety argument.)

### The math (src/math/bla.js — all formulas cross-checked + BigInt-validated)
The exact delta step dz' = 2·Z·dz + dz² + dc is LINEAR in (dz,dc) while dz is small (dz² ≪). A
run of L such steps from reference index m composes to ONE bivariate-linear map dz_{m+L} = A·dz_m
+ B·dc with complex A,B FROM THE REFERENCE ALONE (shared by every pixel). A pixel at (m,dz) with
|dz| inside the run's validity radius r jumps L iterations in one A·dz+B·dc eval. After a rebase
it re-engages from m=0. Built as a binary merge tree:
- level 0 (one step at m): A = 2·Z_m, B = 1, r = blaEps·|2·Z_m|; r forced to 0 when |Z_m| ≥ ZMAX
  (=2) so no run can approach the bailout radius (then |z| ≈ |Z| < 2 ≪ 256, no escape can hide).
- merge x then y: A = A_y·A_x, B = A_y·B_x + B_y, r = min(r_x, (r_y − |B_x|·dcMax)/|A_x|) clamped
  ≥ 0 (triangle-inequality bound on the y-input over the whole dc box). |A| grows like 2^L and can
  overflow a double — ANY non-finite coeff forces r = 0 so an overflowed BLA is never applied.
Stepping (escapePerturb, `bla` arg): take the largest level whose r²≥|dz|² and that overshoots
neither maxIter nor len−1, apply the map; else one true step + escape/rebase. The no-BLA path is
LITERALLY UNCHANGED (the BLA loop is a separate early branch) — all 34 prior unit tests still pass.
Safety of skipping escape+rebase inside a run (validated, companion-confirmed): |dz| ≤ r = blaEps·
|2Z| ≪ |Z| ⇒ no rebase (needs |Z|<2|dz|) and |z|≈|Z|<2 ⇒ no escape; jumps are clamped to len−1.

### The measurement (tools/crosscheck-bla.mjs — work = true steps + BLA jumps, vs the no-BLA oracle)
On the genuine deep boundary coordinate, 48×48, the MARGINAL work reduction BLA buys ON TOP of SA
(this bounds the additional GPU-escape speedup), and the combined SA+BLA vs the no-SA-no-BLA baseline:
| depth  | SA vs base | +BLA vs SA (eps 2^-30) | SA+BLA vs base | mism (eps 2^-30) |
|--------|-----------|------------------------|----------------|------------------|
| 2^-120 | 5.90×     | 2.49×                  | **14.7×**      | 0                |
| 2^-271 | 8.54×     | 2.73×                  | **23.3×**      | 1 (±1)           |
| 2^-400 | 12.15×    | 3.25×                  | **39.5×**      | 2–3 (±≤19)       |
Even the conservative eps 2^-32 gives SA+BLA = 11.8× / 19.2× / 33.2× vs baseline — ALL past 10×.

### Accuracy/speed knee + the honest correctness story (tools/arbiter-bla.mjs, BigInt-exact)
BLA is an APPROXIMATION (drops dz²), so unlike SA it is not perfectly bit-exact — blaEps trades
skip for a small escape-count drift on a handful of high-count near-maxIter pixels. BigInt
arbitration of the drift (the project standard): the differing pixels are WELL-conditioned (the
no-BLA oracle matches BigInt) and BLA is off by a SMALL bounded amount (±1 at 2^-271; ≤19 at
2^-400) — the dropped-dz² truncation accumulating over a long orbit, NOT a missed escape (a missed
escape flips by hundreds; the ZMAX guard + radius provably forbid it, and the unit test hard-caps
maxΔ ≤ 50). The drift is FAR below the GPU's existing df64 mismatch envelope (0.3–1%), so the GPU
port can run a looser eps (more skip) gated on "GPU+BLA vs CPU oracle stays in the df64 envelope".
Default blaEps = 2^-30 (≤3 pixels off by ≤~19, visually invisible). Tradeoff curve in bla.js.

### NEXT — THE GPU PORT (the headline-speed realization; CPU bla.js is now the bit-exact-ish oracle)
The ~8ms GPU floor is the post-rebase re-growths; BLA cuts them ~2.5–3.3× → projects the live GPU
5.24×/6.65×/9.58× to roughly **13×/18×/30×** (SA+BLA), smashing 10× across the board. The port:
- **Upload the BLA table as a texture** (the reference is shared, so it's a texelFetch — the 3090
  loves this). Per (level,m): complex A,B as df64 (Ax.hi,Ax.lo,Ay.hi,Ay.lo,Bx…,By… = 8 floats) + r²
  (1) = 9 floats. ~16 levels × refLen(~100k) ≈ 14M floats ≈ 56MB — fits, but consider stride-2^l
  per level (the classic KF layout, geometric-sum memory) or capping levels if VRAM is tight.
- **In perturbFragRescaled**, after the rescaled update + before (or replacing) the fe escape/rebase,
  scan levels high→low: fetch r²[l][m], if r² ≥ |dz|² (shared-exponent compare — careful, see below)
  and m+2^l ≤ len−1 and n+2^l ≤ maxIter, do dz = A·dz + B·dc in df64 (A,B,dc all O(1)·2^frame) and
  jump m,n by 2^l. The jump is df64 (reuse ds_mul/ds_add — already barriered/highp). |dz|² and r²
  must be compared in the SAME representation (rescaled shared-exponent or fe); reuse the engine's
  fe_norm path for the magnitude. This is the delicate part — validate per-op like the SA seed.
- **Compute A,B,r on the CPU** (worker, alongside computeReference + computeSeries) — buildBLA already
  does it; pass the chosen levels as a texture. blaEps tunable; start ~2^-26..-28 and tighten until
  GPU+BLA vs the CPU oracle re-enters the df64 envelope (validate-gpu's existing mism numbers).
- **Gates**: extend validate-gpu.mjs with a BLA section (GPU+BLA vs CPU oracle, must match the no-BLA
  df64 mism), a crosscheck (GPU BLA-on vs off like crosscheck-skip), and bench-sa-style timing with a
  BLA column — on the REAL GPU (GPU=1). Keep BLA OFF in the viewer until validated on real hardware.
- **SIMD divergence** (companion-vetted): in the chaotic zone some lanes jump, some don't — but a
  jump finishes instantly and the level scan is shared, so the work reduction dominates the penalty.
- **Compose with SA**: keep the SA seed for the ultra-deep LEADING run (where A overflows so BLA's
  top levels have r=0 anyway) and let BLA handle the post-rebase re-growths. They already compose on
  the CPU (the unit test + crosscheck cover SA+BLA).
- **Three companion-flagged GPU traps**: (i) the BLA table sampler MUST be NEAREST + `highp` (a
  filtered/mediump fetch corrupts the df64 coeffs — same class of bug as the Spawn 9 mediump-sampler
  disaster); (ii) the per-iteration level SCAN diverges across a warp (lanes at different BLA depths)
  — keep the scan branch-light and the table layout fetch-cheap; (iii) the (level,m)→texel index math
  must be exact integer (texelFetch ivec2, m % W / m / W), never float-derived, or it desyncs from the
  CPU/BigInt logic.

---
## ⚠️ BLA GPU PORT — built, validated CORRECT on real hardware, but a PERFORMANCE NEGATIVE (Spawn 18)

**Ask (continuing 2026-06-30 "optimize until 900% faster"): port the Spawn-17 CPU BLA to the GPU
rescaled deep engine — the documented "headline-speed" next step. OUTCOME: the port is correct and
real-HW-validated, but BLA does NOT accelerate the GPU. SA+BLA is ~1.6–2× SLOWER than SA alone.
SA is the right and sufficient GPU lever; BLA is a CPU-centric optimization. This is an honest,
measured, companion-confirmed negative result — kept OFF, with the validated infra preserved.**

### What was built (all correct, all validated on the real RTX 3090)
- **Floatexp table texture** (`bla.js` `blaToFloat32`): the NOTES plan said store A,B as plain df64 +
  r² as a float — but I MEASURED the coeff magnitudes first (`tools/probe-bla-mag.mjs`): at 2^-400 the
  jumps a real render APPLIES use |A|~2^228, |B|~2^231, r²~2^-576 — far outside float32's ±127. So the
  table MUST be floatexp. Each complex A,B is a df64 mantissa pair under ONE shared int exponent
  (normalized to the larger component, exactly the rescaled-dz form); r² is a single-float mantissa +
  exponent. 3 RGBA32F texels/entry (12 floats): t0=A mantissas, t1=B mantissas, t2=(Ae,Be,r2e,r2m).
  Entry index E=(l−1)·len+m, levels 1..maxLevel (level 0 is single-step, never jumped). ~77 MB at 2^-400.
  Round-trip unit-tested incl. the deep case where exponents exceed float32 range (bla.test.mjs).
- **Shader scan** (`glsl.js` `perturbFragRescaled`): at the loop top, if BLA is on and dz≠0, scan for the
  LARGEST valid level (radius r²≥|dz|², bounds m+L≤len−1 & n+L≤maxIter) and apply dz'=A·dz+B·dc entirely
  in floatexp, then collapse back to the shared-exponent (Dx,Dy,S) form. `uBlaMaxLevel==0` disables it →
  the no-BLA path is BIT-IDENTICAL (validated: the whole df64/fe/rescaled/SA suite is unchanged).
- **Renderer/harness/validation**: `renderer.uploadBLA` (highp NEAREST), `renderPerturbRescaled` binds it
  (dummy 1×1 + level 0 when off). validate-gpu has TWO gates — see "the right way to validate BLA" below.

### The right way to validate BLA (it is an APPROXIMATION, loose shallow, exact deep)
BLA drops dz², so it is only accurate where dz stays tiny (deep). The CPU BLA itself drifts ~7% / Δ~570 at
2^-90 on the seahorse coord but is 0-mism / Δ0 by 2^-120 (`tools/crosscheck-bla`, seahorse RE/IM). So a
naive "GPU(BLA) vs no-BLA oracle" gate FAILS at 2^-70/-90 — NOT a GPU bug, just BLA being loose there. Two
gates separate the concerns (both PASS on SwiftShader AND the real RTX 3090, identical numbers):
- **FAITHFULNESS** — GPU(BLA) vs CPU(BLA), same table (`blaOracle:true`): isolates the GPU-vs-CPU df64 gap
  from BLA's own drift. Holds at EVERY depth incl. the loose band (2^-70 0.315%, 2^-90 0.564%). THIS proves
  the port is correct.
- **CORRECTNESS** — GPU(BLA+SA) vs the TRUE no-BLA escape counts, deep where BLA is accurate (2^-130 0.171%,
  2^-271 0.049% — within the df64 envelope; the headline production path).
- `tools/crosscheck-gpu-bla.mjs` (GPU BLA on-vs-off drift): 0.15%/0.08%/0.22% at 2^-120/-271/-400, maxΔn≤23
  (BLA truncation, never a missed escape). `highp` samplers hold the df64 coeffs on real HW (no mediump collapse).

### WHY IT'S SLOWER ON GPU (the measured negative result — `tools/bench-bla.mjs`, real RTX 3090)
Off (no-opt) / SA / SA+BLA, deep boundary coord: 2^-120 32/7.7/12.7 ms, 2^-271 52/7.9/15.5, 2^-400 66/7.5/13.8.
So SA alone = 4.2×/6.6×/8.8× over no-opt, but **SA+BLA = 2.6×/3.4×/4.8× — adding BLA roughly HALVES the
speedup.** On the CPU, BLA cuts the kept ITERATION work 2.5–3.3× on top of SA (crosscheck-bla); that work
reduction is REAL on the GPU too, but each BLA step costs far more than a plain rescaled iteration:
- **Texture latency vs ALU.** SA is Horner — pure FMA, perfectly warp-coherent, what GPUs crush. BLA needs
  table fetches; even a cache hit's latency dwarfs an FMA. The per-iteration scan (probe + maybe-jump) is the tax.
- **The coherence gap.** SA clears the big COHERENT bulk (one Horner skips ~92%). What's left is the chaotic
  post-SA tail — exactly where lanes diverge (different m, different jump levels → serialized execution).
  BLA's value is highest where the GPU's penalty is highest.
- **Cost of the tail.** On the CPU a table lookup is negligible vs a high-precision iteration; on the GPU the
  scan "infrastructure" costs more than just brute-forcing the remaining iterations at the GPU's raw throughput.
The companion LLM independently reached the same conclusion (ALU-vs-latency trade-off; "SA is the correct and
sufficient lever for the GPU; BLA is CPU-centric — leverages low-latency caches & branch prediction the GPU
does not prioritize"). Likely NO winning GPU regime: smooth views → SA already does ~everything; chaotic views
→ divergence kills BLA.

### Three scan optimizations attempted (each helped, none flipped it)
1. **Cheap level-1 early-out**: r² is monotone non-increasing in level (a merge only shrinks the radius), so
   level 1 has the largest radius at m — one texel probe rejects the chaotic majority (48 fetches/iter → 1).
   2^-120 22.5→13.8 ms; but 2^-400 stayed erratic (divergence).
2. **Binary search** the largest valid level (both predicates monotone in l → step function): ~4 probes,
   branch-coherent, vs a high→low linear scan's ~maxLevel probes with per-lane divergence. 2^-400 45→11.7 ms.
3. **Carry |dz|² across iterations** (the escape block already computes dz2): the BLA early-out reuses it
   instead of recomputing fe_norm+fe_mul each step. Bit-identical. Fast-skip is disabled when BLA is on (BLA's
   early-out subsumes it; the escape block must run to keep the carried magnitude fresh).
Final: SA+BLA still ~1.6–2× slower than SA. The gap narrowed but the sign never flipped.

### DECISION + what's preserved
BLA stays **OFF on the GPU** (it was always to stay off in the viewer until real-HW-validated; now also because
it's a net loss). The no-BLA path is bit-identical, so shipping the (off) code is free. KEPT: the validated
shader/renderer/table-packing, the two-gate validate-gpu section, `tools/{probe-bla-mag,crosscheck-gpu-bla,
bench-bla}.mjs`, npm `crosscheck:gpu:bla[:real]` / `bench:bla[:real]` / `probe:bla:mag`. BLA remains a genuine
**CPU** win (Spawn 17) and a validated capability for future hardware or a smarter scheme (e.g. a coherent-region
detector that only engages BLA where a whole warp jumps together — speculative). The headline GPU lever is SA.

---
## ⚠️ df64 ESCAPE — escape/rebase in df64 (the documented NEXT-#2 lever): CORRECT, but PERFORMANCE-NEUTRAL (Spawn 19)

**Ask (continuing "optimize until 900% faster"): the one documented un-attempted GPU lever (AGENDA NEXT #2) —
the rescaled engine's per-iteration escape/rebase/glitch test still ran in floatexp every iteration, called
"~half the per-iter cost on the chaotic case." Idea: run it in plain df64 in the common case, fall back to fe
only near a reference minimum. OUTCOME: built + validated bit-identical-correct on SwiftShader AND the real
RTX 3090, but it is PERFORMANCE-NEUTRAL (~1.00×) on both — kept OFF (opt-in), like BLA. An honest measured
result that independently CONFIRMS the BLA finding: the GPU's ~8ms SA floor is not per-iteration-ALU-bound.**

### The key insight (why this was even plausible, and why the precision risk was low)
`fe` (floatexp) = a df64 mantissa + an `int` exponent. So `fe` and `df64` have the **same ~46-bit mantissa
precision**; `fe`'s ONLY advantage is exponent RANGE (`int` vs float32's ±127), needed because deep `dz ~ 2^-500`
underflows df64. BUT the escape/rebase block only RUNS when the fast-skip fails, i.e. when `|dz|` has grown to
O(1) (`S > ezm-4 ≈ -6`) — and that is true **independent of zoom depth** (depth only changes the cheap rescaled
*update*, where `dz` can be 2^-500; Spawn 6 already moved that to rescaled-df64). So when the block runs, `dz`
is comfortably inside df64's exponent range, and the test can run in plain df64 at **bit-for-bit the same
precision** — not a precision reduction. Fall back to `fe` only when `S < -100` or the reference `|Z_m| < ~2^-100`
(near a reference minimum, where the `|z|<|dz|` rebase compare is genuinely range-critical and `ds_scale2`
underflows — `ds_scale2` returns 0 for `p < -100`, so −100 is the exact safe threshold). Subnormal/tiny `Z_m`
reads `ilogb1 = -126 < -100`, so it auto-routes to `fe`. Mirrors `perturbFragDf64`'s escape block (validated to 2^-112).

### What was built (`glsl.js` perturbFragRescaled, `uDf64Esc`)
A gated branch before the floatexp escape block: when `uDf64Esc==1 && uBlaMaxLevel==0 && S != S_ZERO && S>=-100
&& ezm>=-100`, materialize `dz = ds_scale2((Dx,Dy), S)` and run the df64 escape/rebase/glitch (same ds_* ops as the
df64 engine), `continue`. `uDf64Esc==0` (the **renderer default**) skips the branch entirely → the floatexp path is
bit-for-bit unchanged. BLA-on keeps `fe` (it needs `curMag2` fresh). renderer `p.df64Esc` (opt-in); `df64Esc` threaded
through `comparePerturb`/`benchPerturb`. `tools/bench-df64esc.mjs` (npm `bench:df64esc[:real]`, SIZE=/NOSA= env).

### VALIDATED CORRECT — bit-for-bit as precise as fe (both backends, identical numbers)
`validate:gpu` + `validate:gpu:real` ALL PASS with `uDf64Esc=1`. The rescaled-with-df64-escape numbers MATCH the
historical fe-escape rescaled numbers to the digit (rs seahorse 2^-90 = **0.977%** either way; rs+SA deep 2^-130/-271
= **0.000%**), confirming the same-mantissa hypothesis — **no glitch line at the df64/fe fallback boundary** (the two
sides agree to ~last-bit, so the escape/rebase decisions are identical except on measure-zero pixels). A dedicated
`== perturb rescaled + df64-escape (uDf64Esc=1) ==` section now gates the (off-by-default) toggle going forward.

### THE NEUTRAL RESULT (`tools/bench-df64esc.mjs`, SA on, df64Esc off vs on)
- **Real RTX 3090**, 64²–128²: 1.03×/0.98×/0.92× at 2^-120/-271/-400 (small renders — the 0.92× is fixed
  draw+readback overhead noise). **512² (compute-bound, GPU saturated): 0.99×/1.00×/1.01% — dead neutral.**
- **SwiftShader** (the ALU-bound software path, where "half the per-iter cost" was claimed): **1.00× at all depths.**
So neutral EVERYWHERE — and crucially NEUTRAL, not a net-negative like BLA (≈1.6–2× slower). DIAGNOSIS: the escape
block's cost is dominated by the **barriered `ds_mul` squarings** (`|z|²`, `|dz|²` — shared by BOTH the fe and df64
paths; the `ob()` XOR barrier makes every `ds_*` op heavy) and the **`getZ` texture fetch**, NOT the fe-normalize
WRAPPER (`ilogb`+scale per op) that df64 removes. Removing the wrapper saved real ALU but it was a small fraction of
the per-iter cost, so net ≈ 0. (Companion LLM independently endorsed the measure-first gate AND the keep-off call.)

### DECISION + what's preserved
`uDf64Esc` stays **OFF by default** (opt-in): the long-validated floatexp escape path remains production; no measured
benefit ⇒ no production-default change (the project's measure-first discipline). The branch is bit-identical when off,
so shipping the (off) code is free; KEPT for future weak/mobile-GPU measurement (lighter ALU / register pressure MIGHT
help a latency-bound mobile GPU — untestable here, speculative). What this rules IN for a real future win: cut the
`ds_mul` COUNT or the per-iteration texture fetch, not the number representation. CONFIRMS BLA: per-iteration ALU
micro-opts do not break the SA floor; the next real GPU lever (if any) is fewer fetches / less divergence, or off-GPU
work (OffscreenCanvas). New: `tools/bench-df64esc.mjs`; edits: `src/gpu/{glsl,renderer,validate}.js`,
`test/gpu/harness.html`, `tools/validate-gpu.mjs` (+df64-escape section). 39 unit + validate:gpu (both) ALL PASS.

---
## ⚠️ (HISTORICAL — Spawn 8) REAL-GPU TESTING + the df64 reassociation bug

**Headline: the deep GPU engines (df64 / floatexp / rescaled) are NUMERICALLY WRONG on
real NVIDIA hardware. Every prior spawn validated only on SwiftShader (a CPU GL), which
HID it.** A real user on a real GPU gets 20–99% wrong pixels at deep zoom (≥ ~2^-12). The
CPU path is correct; the GPU deep path is not (yet).
*(Spawn 9: the "not yet" is now resolved — see the ✅ section above. The residual the
text below calls unsolved was the mediump sampler, not a barrier the driver defeats.)*

### 1. How to run on the real GPU (the original ask — SOLVED)
Headless Chromium defaults to the **SwiftShader CPU rasterizer** for WebGL even when a
GPU is present; our launch flags `--disable-gpu --enable-unsafe-swiftshader` forced it.
To use the real GPU you must select an ANGLE GPU backend. All launchers now read the
`GPU` env var via the shared **tools/chromium-launch.mjs**:
- `GPU` unset / `0` / `cpu` → SwiftShader (portable default; CI-safe).
- `GPU=1` (or `vulkan`) → ANGLE/Vulkan → `ANGLE (NVIDIA, Vulkan … RTX 3090, NVIDIA)`.
- `GPU=gl` → ANGLE native GLES → `ANGLE (NVIDIA, RTX 3090, OpenGL ES 3.2)`.
Both expose EXT_color_buffer_float (RGBA32F) + maxTex 32768. Verified on RTX 3090, NVIDIA
580.82.09, headless (no X). `node tools/probe-gpu-real.mjs` is the flag-combo sweep that
found these. npm: `validate:gpu:real`, `bench:gpu:real`, `e2e:gpu`, `probe:gpu`.
The GPU is ~600× faster than SwiftShader (fe ~13000 vs ~21 Mit-px/s before the precision
barrier; the barrier costs ~2.6× because the collapsed shader was doing less work).

### 2. THE BUG: df64 collapses to float32 on NVIDIA (compiler FP reassociation)
The double-single (df64) ops in DF64_LIB depend on EXACT IEEE-754 float32 and the
Dekker/Veltkamp error terms (e.g. the split `a_hi = ca - (ca - x)`). The NVIDIA shader
compiler applies the algebraically-valid-but-FP-INVALID identity `ca - (ca - x) → x`,
which zeroes the split (`a_lo == 0`) → df64 silently degrades to plain float32 (~24-bit).
SwiftShader does NOT reassociate, so it passed there. PROVEN by tools/probe-df64.mjs:
on the real GPU `split_lo == 0` and ds_mul relerr == plain-f32; on SwiftShader it's intact.
(The reference URL Danielle gave serves byte-identical code, so it has the same bug — it
shows the target image but also falls back to CPU. This is genuinely unsolved upstream.)

### 3. The PARTIAL fix (shipped) and its LIMIT
GLSL ES 3.00 (WebGL2) has no `precise` qualifier (3.20+) and no `fma`. The portable
defense is an OPTIMIZATION BARRIER `ob(x) = intBitsToFloat(floatBitsToInt(x) ^ uOptBarrier)`
where uOptBarrier is a uniform == 0 the compiler can't prove is zero (a plain
intBitsToFloat(floatBitsToInt(x)) round-trip is FOLDED AWAY — proven). ds_add/ds_mul in
DF64_LIB now wrap every rounded result in ob(); the renderer sets uOptBarrier=0 once per
program in `_program()`. RESULT (tools/probe-df64.mjs, probe-df64-real.mjs): in ISOLATION
the barriered ds_mul / ds_add / a 40-iter update are now INTACT on the real GPU (relerr
~1e-14, == SwiftShader). **BUT** the full `perturbFragDf64` shader STILL diverges from
SwiftShader-df64 by 21% (2^-22) … 90% (2^-50) — see tools/probe-xbackend.mjs — and still
fails validate:gpu:real (df64/fe/rs FAIL 2^-3…2^-90; only the fast-escaping exterior and
naive-f32 pass). So the barrier is NECESSARY but NOT SUFFICIENT.

### 4. What the residual is NOT, and what it IS (diagnosis, for next-spawn)
Ruled OUT (all tested): isolated ds_mul/ds_add (intact), the fast-skip branch (identical
on/off), subnormal flush / FTZ (tools/probe-ftz.mjs: SwiftShader AND NVIDIA both flush —
identical, so not the differentiator), the reference texture / coordinate mapping
(identical JS upload; the divergence GROWS with iteration count, so it's per-iter
accumulation not a fixed offset). Native GLES fails IDENTICALLY to Vulkan (same numbers to
the digit) — both ANGLE frontends feed the same NVIDIA driver backend. CONCLUSION: floating
point is deterministic, so identical GLSL + identical isolated ops + divergent full-shader
output ⇒ **the NVIDIA driver compiler reassociates/contracts the LARGE inlined perturbation
shader differently than SwiftShader, past the per-op ob() barriers** (it re-optimizes after
inlining ds_* into the big main()). The simple probe shaders don't trigger it; the complex
one does. df64-vs-f32 on GPU is ~5% (not 0), so it's PARTIAL collapse, not full.

### 5. NEXT STEPS (priority)
a. **SHIP A SAFETY NET regardless of the shader fix: a GPU deep-precision self-test +
   CPU fallback.** At first deep render, render a small known tile on the GPU and compare
   escape counts to the CPU oracle (escapePerturb is in the browser already); if mism >
   tol, mark the GPU deep path untrusted and route deep renders to the CPU pool (correct,
   just slower). This makes the viewer CORRECT on ALL hardware today — a real user-facing
   bug fix — independent of cracking the shader. THE highest-value next action.
b. Crack the residual: needs NVIDIA ISA introspection (no easy tool in-sandbox). Ideas to
   try: (i) a stronger/un-CSE-able barrier (the per-op ob() survives simple shaders but the
   driver defeats it in the big one — try barriering at the shader-body composition level
   too, or a barrier whose value the driver can't hoist); (ii) split perturbFragDf64 so the
   compiler has less to reassociate; (iii) test whether a TRIPLE-float or a higher-precision
   reference changes it (if it's a true precision cliff after all — but the GPU-vs-SwiftShader
   divergence of identical code argues compiler, not cliff). Re-run validate:gpu:real after
   each idea — it's the gate. probe-xbackend.mjs (GPU-df64 vs SwiftShader-df64) is the fast
   inner-loop signal (no oracle needed; should drop toward ~0%).
c. Once correct on GPU: profile (bench:gpu:real shows rescaled ~2.1× faster than fe on real
   HW vs 1.26× on SwiftShader — the rescaled engine helps MORE on real GPUs, as predicted).

## GPU acceleration (M6) — WebGL2 shaders  [BUILT + VALIDATED ON SWIFTSHADER; BROKEN ON REAL GPU — see ⚠️ above]

The per-pixel rasterization is migrated to GLSL fragment shaders (WebGL2), now the
DEFAULT engine (toggle in the UI). The high-precision reference orbit is still
computed on the CPU (BigInt) — only the cheap per-pixel delta/escape loop moves to
the GPU, which is the whole point. The CPU worker pool remains the fallback and
the ground-truth oracle (used below 2^-112 and whenever GPU is off/unsupported).
NOTE (Spawn 8): "VALIDATED" above means validated on SwiftShader. On real NVIDIA the
deep df64/fe/rescaled engines are numerically wrong (df64 reassociation, see ⚠️ section).

RESOLVED (Spawn 7): the deep render USED to be a single GPU draw on the main thread.
At extreme depth + high maxIter (e.g. 2^218 → maxIter ~55k) that one draw ran long
enough to trip the GPU watchdog (TDR) on real hardware → context loss → CPU fallback
(minutes) → "can't zoom past 2^218". The escape pass is now STRIP-TILED across many
short draws with a yield between them (viewer._drawTiledEscape), so no single draw
exceeds the watchdog, the UI stays responsive, and the image reveals top-to-bottom.
See the "Strip-tiled deep render" section below. (Running the GPU in an OffscreenCanvas
worker is still a possible future refinement, but is no longer needed to avoid the TDR.)

Files: src/gpu/glsl.js (shaders), src/gpu/renderer.js (WebGL2 plumbing),
src/gpu/validate.js (GPU-vs-oracle comparison), test/gpu/harness.html (browser
harness), tools/validate-gpu.mjs (the canonical GPU regression — run it!).

### The four GPU engines and the dispatch (empirically chosen, see below)
- **naive f32**     : radius >= 2^-2  (0.25). Shallow/home, where the view shows
  large parts of the set and a single-reference perturbation is inappropriate.
  No reference needed → instant. Boundary pixels (24-bit coord) may flip; bulk is
  exact (meanΔsn < 2 even at the worst valley at 0.25).
- **perturb df64**  : 2^-2 .. 2^-112. The deep workhorse. Reference orbit in a
  CPU-computed RGBA32F texture (Zx.hi,Zx.lo,Zy.hi,Zy.lo); deltas in df64 too.
  Validated vs the CPU perturbation oracle at the seahorse VALLEY (worst case)
  across 2^-3..2^-110: mism < 1.2% of pixels, meanΔsn < 1 (see validate-gpu).
  df64 extends the mantissa to ~46 bits but NOT the float32 exponent, so dc/dz
  ~2^-270 would underflow (min normal 2^-126) — hence the 2^-112 floor.
- **perturb floatexp / RESCALED** : 2^-112 .. **2^-600** (Spawn 10 lowered the floor
  from 2^-340). THE 2^270..2^600 GPU PATH, incl. the **2^500** deep-zoom target.
  Same df64 (~46-bit) mantissa, but each per-pixel delta carries an int exponent so
  dc/dz far below 2^-126 don't underflow. The reference Z stays df64 (it is O(1)). dc
  origin/step are passed in as fe (mantissa + exponent) from the CPU. Two
  implementations occupy this band: `perturbFragFloatexp` (per-component exponent, the
  validated reference) and `perturbFragRescaled` (shared exponent, ~1.26× faster, NOW
  THE DEFAULT the viewer dispatches — see the Spawn-6 rescaled section below). Both
  validated headless vs the CPU oracle — correct on varied chaotic escapes (2^-70/2^-90)
  AND, Spawn 10, on a GENUINE deep boundary coordinate at 0.000% escape-count mismatch
  FLAT from 2^-120 to 2^-520 (30k→130k iters) + exterior arithmetic to 2^-600. See the
  "Deep floor → 2^500" section below.
- **CPU perturbation** (existing worker pool): radius < 2^-600 or GPU unsupported.
  Still the ground-truth oracle, validated to 2^-400.

Also kept (validated, available, not the default): naive df64 shader and
perturb **f32** shader.

### floatexp ("fe") GPU deep path — how it works and why it's correct
The plain-df64 perturb shader floors at ~2^-112 because float32 cannot represent
the per-pixel offset dc ~ 2^-270 (it underflows to 0 below 2^-126). The fe engine
fixes exactly this: a real value is stored as `m * 2^e` where m is a df64 (vec2
hi/lo, ~46-bit, normalized so |m.x| in [0.5,1)) and e is a plain int. The df64
*mantissa* keeps the validated 46-bit precision; the int *exponent* gives the full
double range. Only the small deltas (dc, dz) need fe — the reference Z is O(1) and
stays df64; the 2*Z*dz product promotes Z into the fe via `fe_mulds` (df64×fe).
- src/gpu/glsl.js FE_LIB: fe_norm/add/sub/mul/mulds/dbl/lt/tof. Built on the df64
  ds_* ops. NOTE: WebGL2 is GLSL ES **3.00** which has NO frexp/ldexp (those are
  3.10). Normalization (Spawn 5, was log2/exp2 + a correction step) now reads/writes
  the IEEE-754 exponent field directly: `fe_ilogb1(x)=((floatBitsToInt(x)&0x7fffffff)
  >>23)-126` is the frexp exponent (k with |x|*2^-k in [0.5,1)); `fe_pow2(k)=
  intBitsToFloat((k+127)<<23)` is an exact 2^k. This is EXACT (no ±1 log2 rounding →
  no correction step, and no vendor-approx "sparkle" risk on Adreno/Mali) and far
  cheaper — see the perf section below. Valid because our mantissas stay in
  [~2^-50, 2], so every exponent we touch is inside the normal range [-126,127];
  `fe_tof` guards e<-126 → 0 (underflow), since fe_pow2 only covers e>=-126.
  Also: GLSL ES forbids the `?:` ternary on structs — use if/return.
- The rebase test |z| < |dz| MUST be done in fe (both sides ~2^-540 underflow a
  float), hence fe_lt. The escape test |z|^2 > bailout uses fe_tof (mag2 is O(1) at
  escape, representable). dz^2 is never converted to float — only compared via fe.
- src/gpu/renderer.js feSplit(double)->{hi,lo,e}; renderPerturbFloatexp() passes
  uOx/uOy/uScale as (vec2 mantissa, int exponent) pairs. Reuses uploadReferenceDf64.
- Validation (tools/validate-gpu.mjs, "perturb floatexp" section): two angles —
  (a) the 2^-70/-90 overlap band gives REAL varied chaotic escapes that exercise
  the exponent path and match the oracle's bulk metrics like df64 (esc≈9000,
  mism 0.3–1.1%); (b) escaping EXTERIOR patches at 2^-130..2^-340 (below the float32
  floor) confirm no underflow (esc=10000, mism 0.000%). Varied-chaotic escapes
  BELOW the floor aren't directly tested (they'd need ~maxIter≈30k → too slow under
  SwiftShader) but the fe arithmetic is exponent-magnitude-agnostic, so (a)+(b)
  together cover it. PERF: fe is still heavier than df64 (~5× after Spawn 5's opts,
  was ~19×), but it now renders much faster — see the shader-perf section below. On
  a real GPU it parallelises across pixels and should beat the single-thread-ish CPU
  path. GPU failure → CPU.

### GPU shader performance optimization (Spawn 5) — measured on SwiftShader
The fe (floatexp) deep path was the bottleneck. `tools/bench-gpu.mjs` (new) times a
pure render — a draw + a 1-px readPixels to force ANGLE/SwiftShader to actually run
the fragment work (gl.finish alone is elided when nothing reads the framebuffer) —
and reports Mit-px/s (= width·height·maxIter / ms). Two BIT-IDENTICAL optimizations
(validate-gpu mismatch numbers unchanged to the digit, so these are pure speed):

1. **fe_norm/fe_add/fe_tof via IEEE-754 bit ops, not log2/exp2.** The hot loop runs
   ~20 fe ops/pixel-iteration; the old normalize spent 2 transcendentals per fe_norm
   and fe_add 3 more exp2 — ~60+ software-transcendental calls per pixel-iteration,
   the dominant cost on SwiftShader (a CPU rasterizer). fe_ilogb1/fe_pow2 (above) do
   the same job with a few int ops. **~1.85× on the fe path** (matched-load A/B).
2. **Carry Z[m] across iterations (both df64 + fe perturb loops).** Each iteration
   fetched the reference texture twice (Z[m] at the top, Z[m+1] for the z=Z+dz test);
   the second IS the next iteration's first, so carry it and fetch once per iteration
   (re-fetch Z[0] only on rebase). **~2.4× on the deep df64 path** (texture sampling
   is a big fraction of df64's otherwise-lean per-iter work on SwiftShader) and
   **~1.2×** more on fe.

Combined, matched-load A/B at the seahorse valley (the expensive chaotic-escape case):
| path        | before | after | total |
|-------------|--------|-------|-------|
| df64 2^-80  | ~109   | ~262  | ~2.4× |   (Mit-px/s; df64 only got opt #2)
| fe   2^-80  | ~10.9  | ~21.7 | ~2.0× |
| fe   2^-270 | ~10.9  | ~21.8 | ~2.0× |
Absolute Mit-px/s drift with machine load (the companion LLM shares this CPU), so
the bench prints df64 + fe together and the ratios above are within-run/matched-load.
HONEST: SwiftShader is a CPU software GL; even at ~2× a full-screen ss=2 extreme-fe
render is still many seconds there. The point is (a) a real ~2-2.4× win that helps
every depth, and (b) a real GPU — where each pixel is a parallel thread — flies. The
single biggest remaining win is the **rescaled-iteration** rewrite (below).

### RESCALED single-exponent iteration (Spawn 6) — DONE, validated, now the deep default
fe carries a SEPARATE exponent per delta component and renormalizes after EVERY op
(~14 fe ops/iter, each a normalize + struct/branch). The rescaled engine
(`perturbFragRescaled`, src/gpu/glsl.js) keeps dz=(Dx,Dy) as df64 mantissas under ONE
shared int exponent S (dz=(Dx,Dy)·2^S), so the update 2·Z·dz + dz² + dc runs in raw
df64 (like the df64 path) and renormalizes S ONCE/iter. **It is now the engine the
viewer uses for the deep `gpu-perturb-fe` band** (viewer._renderGpuPerturb); the old
`renderPerturbFloatexp` stays in the renderer as the reference/oracle.

How the update works (per iteration, dz=(Dx,Dy)·2^S, dc=(Cx,Cy)·2^Sc collapsed once):
- linear  L = 2·(Zx·Dx−Zy·Dy, Zx·Dy+Zy·Dx), exponent S (Z is df64, O(1))
- quad    (Dx²−Dy², 2·Dx·Dy), exponent 2S, DROPPED when >52 bits below the frame
          (matches what fe_add does internally — at deep zoom dz²~2^-540 is invisible)
- dc      (Cx,Cy)·2^Sc
align all three to frame W = max(eL, qe, Sc) with exact power-of-two scalings
(ds_scale2), sum in df64, renormalize S once.

THE CATCH (deferred by Spawn 5) was the exact rebase magnitude compare. SOLVED simply:
the escape/rebase test still runs in EXACT floatexp — convert (Dx,S),(Dy,S) back to fe
per component (fe_norm) and run the byte-identical fe escape/rebase code. So the Zhuoran
decision logic is unchanged; only the cheap bulk update is rescaled. On rebase, z (fe per
component) is re-collapsed to shared (Dx,Dy,S).

TWO BUGS found + fixed while validating (both are general traps for this representation):
1. **Z_0 = 0 (Mandelbrot) makes the linear term vanish after EVERY rebase** (m=0,
   Z[0]=0 → 2·Z·dz = 0), so dz' = dz² + dc with dz² DOMINANT. The frame W must include
   the dz² exponent qe=2S; otherwise a vanished linear (ilogb1(0)=−126) picks a bogus
   frame that mis-scales dz², and the orbit NEVER ESCAPES after a rebase (exterior
   patches read 100% interior; chaotic pixels that rebase fail). W=max(eL,qe,Sc) fixes it.
2. **Un-normalized linear inflates the frame.** |2·Z·D| can be ~6; using S (not the
   linear's true exponent eL=S+ilogb(linear mantissa)) as the frame costs the dc/dz²
   addends ~2-3 low bits vs fe → mism 4× worse at the viewer's deep maxIter. Fold the
   linear's exponent into W (eL); the scaling ds_scale2(lx, S−W) needs no extra normalize.

PERF (matched-load A/B vs fe, seahorse valley = worst-case chaotic, SwiftShader):
**~1.26× faster than fe**, stable across 2^-80/-150/-270. (The escape/rebase block is
still fe and dilutes the update win on this maximally-divergent case; smoother deep
regions + a real GPU should do better.) PRECISION: validate-gpu's rescaled section uses
the SAME thresholds as fe and PASSES identically (2^-90 rs 0.977% vs fe 1.074% — rescaled
is even a hair better); rescaled agrees with fe bit-for-bit up to ~15k iters and to bulk
metrics beyond. tools/probe-rescaled.mjs (rs-vs-oracle + rs-vs-fe) and tools/crosscheck-
skip.mjs (the skip below) gate it.

### Perturb fast-skip (Spawn 6) — bit-identical, in all three perturb shaders
When dz is far below the O(1) reference Z_m, the true value z=Z_m+dz can neither escape
(|z|≤|Z_m|+|dz| ≪ 256) nor rebase (|z|≥|Z_m|−|dz| > |dz|), and the glitch test is false
(mag2~|Z_m|²) — so the whole escape/rebase/glitch block is INERT and is skipped
(`uFastSkip`). PROVABLY bit-identical: |dz|<2^(sdz+1), |Z_m|≥2^(ezm−1); ezm≥sdz+4 ⇒
|Z_m|>2|dz| (no rebase); ezm≤6 ⇒ |z|<256 (no escape); m≠uRefLen keeps the forced
end-of-ref rebase; below the df64 floor Z_m reads 0 (ezm=−126) and the exact path
wouldn't rebase either, so the skip still matches. tools/crosscheck-skip.mjs renders the
same view with the skip on and off and asserts a 0-diff full-image match (df64+fe+rs,
2^-20..-340 incl. the chaotic valley AND the escaping exterior). HONEST: on the chaotic
seahorse bench it gives ~1.00× — SwiftShader (and real GPUs) run pixels in SIMD groups,
so a `continue` saves nothing if ANY lane in the group still needs the block, and the
chaotic valley keeps groups busy. It's free (not slower) and helps SMOOTH contiguous
deep regions + real hardware, where whole groups skip together. The headline deep win is
the rescaled UPDATE (above), which every pixel runs every iteration regardless of divergence.

### Strip-tiled deep render (Spawn 7) — the fix for "zoom past 2^218"
THE 2^218 BARRIER WAS NOT PRECISION. Measured (tools/probe-deep218.mjs): the rescaled
+ floatexp engines match the CPU 53-bit oracle to **0.000% mism** on a real deep
coordinate all the way to 2^-271 (chaotic high-maxIter escaping regions), and the
reference builds in ~450ms. The actual wall: a deep frame needs maxIter ~55k (autoMaxIter
= 400 + 250·octaves), and the escape pass was ONE GPU draw over the whole screen. On a
real GPU a single 10–40s draw trips the **watchdog / TDR** → the browser resets the GL
context → viewer._gpuFail falls back to the CPU pool (minutes at that depth) → it looks
like "you can't zoom past 2^218". (On SwiftShader the same single draw just times out
>120s — confirmed via tools/shoot-deep.mjs, which showed gpu-perturb-fe, glitches:0, but
never finishing.)

THE FIX (viewer._drawTiledEscape + renderer scissor support): split the escape pass into
horizontal STRIPS and draw them one at a time, yielding a frame (requestAnimationFrame)
between draws.
- renderer: every escape method now calls `_bindEscapeTarget(p)` which sets the viewport
  to the FULL FBO (so gl_FragCoord — and therefore each pixel's c — is unchanged) and,
  when `p.stripH` is set, enables a SCISSOR rect (0, stripY, W, stripH) that restricts
  WHICH rows get written. Same gl_FragCoord + scissor ⇒ the tiled result is BIT-IDENTICAL
  to one big draw (tools/crosscheck-tiled.mjs: 0-diff across naive/df64/fe/rescaled, all
  depths incl. 2^-218, strip heights 1-row…larger-than-frame). `clearSn()` clears the sn
  target to interior (sn=-1) first so not-yet-drawn rows read as background → clean
  top-to-bottom reveal. colorize() now `disable(SCISSOR_TEST)` (a strip pass may leave it on).
- viewer: `_drawTiledEscape(drawStrip, gen)` clears sn, then loops strips — each draws,
  `gl.flush()`es (its own GPU command, so the watchdog timer is per-strip), colorizes +
  blits (progressive), and awaits a rAF. It checks `gen !== this.gen` each iteration so a
  zoom/pan/palette change mid-render cancels cleanly (the e2e "zoom mid-render cancels"
  covers this). `_renderGpuNaive` + `_renderGpuPerturb` are now async and route through it.
- `_stripRows()` sizes a strip so its worst-case work (rows·W·maxIter pixel-iterations)
  stays under a ~watchdog budget (4e8 pixel-iters/strip), aligned to the supersample
  factor, ≥ ss, ≤ frame height. Shallow/cheap views collapse to a single full-frame strip
  (no overhead). At 2^218 (W~840, maxIter~55k) that's ~10-row strips, ~150 strips; the rAF
  overhead (~16ms each) is a few % of the real per-strip compute on a real GPU.
HONEST: tiling does NOT make the total work smaller — a full 2^218 frame on a real GPU is
still ~tens of seconds (and is unrenderable on SwiftShader, a CPU rasterizer, regardless).
What it buys: no watchdog reset (the hard barrier), a responsive UI, a progressive reveal,
and free cancellation. The next *speed* levers (untouched, precision-safe-ish): auto-drop
supersampling as depth grows (ss² multiplies the heavy fe cost), and a cheaper-common-case
escape/rebase block (AGENDA NEXT). Driver fragment-loop caps (some mobile GPUs cap shader
loop iterations) are a separate possible failure mode that tiling does NOT address — if a
real device still mis-renders deep, suspect that and tile the ITERATIONS too (multi-pass).

### Display pipeline — point filtering + supersampling (this spawn)
Two separate scaling stages, filtered oppositely on purpose:
- **Supersample (compute res → display res): smoothing ON.** The fractal is computed
  at `ss×` the display backing (default 2×) and box-averaged down. This is the AA /
  "ultra" smoothness. Average the FINAL COLORS of the ss×ss subsamples, NOT the
  smooth-count sn (sn is cyclic through the palette — averaging it bleeds hues).
  - GPU: the color shader (COLOR_FRAG, uSS) loops the ss×ss block and averages
    colors; the sn FBO is ss× the canvas. renderer _ensureFbo (compute res) is now
    decoupled from the canvas (display res, sized in colorize()).
  - CPU: workers render at compute res into an offscreen compute canvas; present()
    downscale-blits it to the display canvas (smoothing on = the box filter),
    rAF-coalesced. ss==1 keeps the old direct-to-display path (no extra blit).
  - Caps: effective ss bounded by MAX_COMPUTE_DIM (8192) and MAX_COMPUTE_PIXELS
    (12e6, ≈ a 192MB float sn texture) so mobile GPUs don't OOM; bands are aligned
    to ss so each maps to whole display rows.
- **Display → screen (CSS upscale, and the zoom-gesture preview): point filter
  (nearest), no smoothing.** `#view { image-rendering: pixelated }` and
  `_applyPreview` uses `imageSmoothingEnabled=false`, so the image stays CRISP when
  the browser scales backing→screen and when a gesture scales the last frame.
  (Danielle: "point filter not bilinear, more crisp especially when zooming.")
  ss + URL hash (`ss=`) + a Supersampling select in the panel; debug shows `_effSS`.

### KEY PRECISION FINDINGS (hard-won — measured headless via SwiftShader)
1. **df64 (double-single, two float32) is correct and ~46-bit.** It matches CPU
   double exactly in well-conditioned regions. df64 extends the *mantissa* but
   NOT the float32 *exponent* (still ~2^-126 floor) → df64 reaches ~2^-112 zoom,
   not deeper. (To go past 2^-112 on GPU you need per-pixel floatexp; future.)
2. **f32 perturbation breaks at high maxIter, even deep.** It is exact only while
   maxIter stays under a depth-dependent threshold, then jumps to 10–30% boundary
   error (meanΔsn 6–23). Cause: the f32 reference Z_m + f32 deltas carry ~2^-24
   absolute error in the reconstruction z = Z_m + dz; once dz grows toward O(1)
   near escape, that error amplifies on chaotic high-count pixels. The viewer's
   autoMaxIter (~250/octave) is well into the breakdown, so **f32 perturbation is
   NOT safe as the default** — hence df64.
3. **df64 perturbation fixes it**: same cases drop from 10–30% to 0–1.2% mism,
   meanΔsn < 1 (a ~20–30× improvement). The residual <1.2% is the genuine
   46-bit(df64) vs 53-bit(double) gap on measure-zero chaotic boundary pixels —
   the SAME effect NOTES already documents for naive-vs-perturbation. The BigInt
   arbiter (tools/arbiter-gpu.mjs) confirms these mismatches are precision, not
   bugs: GPU-unique-wrong pixels are always high-count sensitive ones where CPU
   double ALSO differs from BigInt.
4. **Validate on BULK metrics, not max.** Two finite-precision methods always
   disagree on a few chaotic pixels. Gate on (fraction of pixels differing) +
   (mean Δsn over escaped pixels), never on max Δsn. This mirrors the existing
   "validate vs BigInt, not naive" wisdom.

### Coordinate mapping (single source of truth, mirrored in JS for validation)
Shader: c = uOrigin + gl_FragCoord.xy * uScale, with gl_FragCoord = texel + 0.5.
readPixels row 0 = GL bottom. validate.js recomputes the exact c per texel so the
oracle evaluates the identical point — orientation/flip never affects the check.
df64 uniforms are passed as (hi,lo) via df64Split(double) = [fround(v), fround(v-hi)].

### Smooth-count + coloring parity
Shaders compute sn bit-identically to naive.js/perturb.js. The color pass samples
a 1024×1 RGBA8 LUT baked on the CPU from palette.colorFor — so GPU and CPU coloring
match to LUT resolution. sn<0 = interior (same sentinel as CPU).

### Running the GPU browser (NixOS) — WebGL2 works headless
SwiftShader (ANGLE/Vulkan) gives WebGL2 + EXT_color_buffer_float (RGBA32F render
targets) + 8192 max texture (a 2M-iter reference fits in 2048×N). highp float = 23
mantissa bits (real float32). Same chromium 148 + --headless=new + --no-sandbox
+ --enable-unsafe-swiftshader as the e2e setup. tools/probe-webgl.mjs verifies it.

## Performance / scaling reality (single worker today)
~300M iteration-steps/sec single thread. A full-screen deep view (e.g. 2^100 at
~60k iters over ~500k px) is ~tens of seconds single-threaded. Progressive passes
make it usable (coarse image fast), but the NEXT big win is multi-worker tiling
(fan tiles across cores; share the one reference orbit via SharedArrayBuffer —
COOP/COEP already enabled). Then GPU for shallow/medium. Deep zoom correctness is
done and validated; this is purely about speed.
