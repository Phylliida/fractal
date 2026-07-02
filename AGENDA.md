# AGENDA — Mandelbrot Deep-Zoom Viewer (perturbation theory)

Goal: mobile-first Mandelbrot viewer reaching zoom ~2^400, JS+HTML+CSS, GPU where
it helps, extensive browser automation + integration tests, validated against
ground truth, with accurate glitch detection.

Status legend: [ ] todo · [~] in progress · [x] done & verified · [!] blocked

## M0 — Scaffold & test harness  ✅
- [x] package.json, ES-module layout, static dev server (tools/serve.mjs)
- [x] Playwright wired to a working chromium (see NOTES: NixOS browser saga)
- [x] e2e tests load the page and assert the canvas renders (18 tests, 2 projects)

## M1 — Ground-truth naive renderer (double precision)  ✅
- [x] naive.js: escape-time Mandelbrot in plain doubles (THE reference oracle)
- [x] canvas viewer with view state (center, radius, maxIter) — viewer.js
- [x] mobile touch: pinch-zoom + drag-pan + double-tap; wheel fallback
- [x] palette/coloring (smooth iteration count) — palette.js, 4 palettes
- [x] unit tests: known points (cardioid, period bulbs, escape counts)

## M2 — High-precision arithmetic
- [x] bignum.js: fixed-point BigInt real + complex (mul/cmp/toDouble/decimal IO)
- [x] unit tests vs known values, decimal round-trip, huge-prec no-overflow

## M3 — Reference orbit
- [x] reference.js: high-precision orbit Z_n at view center -> Float64 arrays
- [x] escape / maxIter handling; escapeBigInt exact single-point oracle
- [x] validate: reference orbit == naive double orbit (early iters) [test D]

## M4 — CPU perturbation engine (THE correctness core)  ✅ VALIDATED
- [x] perturb.js: delta iteration dz' = 2*Z*dz + dz^2 + dc (double precision)
- [x] Zhuoran rebasing for glitch-free single-reference rendering
- [x] Pauldelbrot glitch diagnostic exposed (rendering relies on rebasing)
- [x] render.js: reference auto-selection (relocate to deepest pixel)
- [x] VALIDATE vs BigInt-exact oracle: +/-1 at 2^45, 2^120, **2^400** [B,C,C2]
      KEY FINDING: validate vs BigInt, not naive. naive=perturb only use doubles;
      both noisy on ill-conditioned shallow boundary pixels. Dispatch naive
      (shallow) / perturb (deep) by radius. See NOTES.

## M5 — Workers + progressive deep zoom  ✅
- [x] worker.js: POOL of N module workers (navigator.hardwareConcurrency, cap 12).
      worker[0] computes the reference once + a coarse pass; row-bands are then
      fanned round-robin across the pool. ~8x faster deep render (15.7s -> 2.0s
      on the full-screen deep e2e).
- [x] progressive: instant coarse pass (step 8) then parallel full-res bands
- [x] cancellation on view change (terminate pool + generation guard)
- [x] deep-zoom: 2^41, 2^60 render glitch-free in-browser; 2^100/2^400 in Node
- [x] iteration auto-scaling with depth (autoMaxIter in render.js)

## M6 — GPU acceleration (WebGL2)  ✅ BUILT + VALIDATED ON SWIFTSHADER **AND REAL NVIDIA**
## (Spawn 8 found the deep df64/fe/rescaled engines were wrong on real NVIDIA; Spawn 9
##  fixed it — the reference samplers needed `highp` (mediump truncated the df64 ref to
##  fp16). validate:gpu:real now ALL PASS. See NOTES "✅ THE DEEP-GPU BUG IS FIXED".)
- [x] naive GPU shader (shallow zoom, float32) + a df64 variant (medium)
- [x] perturbation shader: **df64** reference + df64 deltas + Zhuoran rebasing.
      (f32 perturb shader also built/validated but NOT default — see finding 2.)
- [x] validate GPU output vs CPU naive/perturbation oracle headless (SwiftShader)
      across 2^0..2^-110; tolerance on chaotic boundary pixels (bulk metrics).
      tools/validate-gpu.mjs (canonical), tools/arbiter-gpu.mjs (BigInt arbiter).
- [x] auto-pick GPU/CPU by depth: naive-f32 (r>=2^-2) / perturb-df64 (2^-2..2^-112)
      / **perturb-floatexp/rescaled (2^-112..2^-600, the 2^270..2^600 GPU path incl. the
      2^500 target — floor lowered from 2^-340 by Spawn 10)** / CPU perturb below
      + on any GPU failure. gpuEngineForRadius() in render.js.
- [x] **floatexp GPU deltas (df64 mantissa + int exponent)** push GPU perturb past
      the df64 float32-exponent floor (2^-112) to 2^-340: dc/dz ~2^-270 no longer
      underflow. VALIDATED headless vs the CPU oracle (varied chaotic escapes in the
      2^-70/-90 overlap band + escaping patches 2^-130..2^-340 below the float32
      floor, 0% mism). src/gpu/glsl.js FE_LIB + perturbFragFloatexp. (Spawn 4)
- [x] in-shader coloring via a CPU-baked palette LUT (no readback for display)
- [x] **SHADER PERF (Spawn 5)**: two bit-identical speedups, measured on SwiftShader
      via the new tools/bench-gpu.mjs. (1) fe normalize via IEEE-754 bit ops
      (fe_ilogb1/fe_pow2) instead of log2/exp2 — removes ~60 software transcendentals
      per pixel-iteration (~1.85× on fe). (2) carry the reference Z[m] across loop
      iterations (one texture fetch/iter, not two) in the df64 + fe + f32 perturb
      loops (~2.4× on deep df64, ~1.2× more on fe). Net ~2.0× fe, ~2.4× df64;
      validate-gpu mismatch numbers unchanged to the digit.
- [x] **RESCALED DEEP ENGINE (Spawn 6)**: the deferred rescaled single-exponent
      iteration, built + hard-validated + shipped as the deep default. dz=(Dx,Dy)·2^S
      shares ONE exponent; the 2·Z·dz+dz²+dc update runs in raw df64, renormalized
      once/iter; escape/rebase stays EXACT floatexp (so the Zhuoran decision is
      unchanged). ~1.26× faster than fe (matched-load, worst-case chaotic valley);
      validate-gpu's rescaled section passes the SAME thresholds as fe. Two bugs found
      + fixed (general traps): the dz² exponent must be in the combine frame (else the
      Z_0=0-after-rebase linear-vanish never escapes), and the linear's true exponent
      must set the frame (else ~2-3 lost low bits). src/gpu/glsl.js perturbFragRescaled.
- [x] **PERTURB FAST-SKIP (Spawn 6)**: skip the escape/rebase/glitch block when dz is
      provably too small to escape or rebase. Bit-identical (crosscheck-skip renders
      skip on/off → 0 diff across df64+fe+rs, 2^-20..-340). Free on the chaotic bench
      (SIMD divergence), helps smooth deep regions + real GPUs (whole groups skip).
- [x] viewer integration: GPU is the DEFAULT; CPU worker pool is fallback+oracle;
      UI toggle "GPU acceleration"; debug shows the active renderer.
- [x] e2e: WebGL2 present, GPU-vs-oracle match, app dispatches gpu-*, CPU fallback
- [x] **STRIP-TILED DEEP RENDER (Spawn 7)**: the deep escape pass is split into short
      horizontal strips (scissor) drawn one at a time with a rAF yield between them, so a
      deep frame (maxIter ~55k at 2^218) never trips the GPU watchdog (TDR) — THE actual
      barrier to "zooming past 2^218" (measured: the engine is numerically perfect to
      2^-271; the wall was one long draw, not precision). Progressive top-to-bottom reveal,
      cancellable mid-render. BIT-IDENTICAL to a single draw (tools/crosscheck-tiled.mjs).
- [x] **DEEP FLOOR → 2^500+ (Spawn 10)**: lowered GPU_PERTURB_FE_FLOOR 2^-340 → **2^-600**
      so the 2^500 zoom (and the whole 2^-340..2^-600 band) renders on the GPU rescaled/fe
      engine instead of the slow CPU fallback. The fe/rescaled MATH is exponent-magnitude-
      agnostic (the exponent is a separate int; the CPU computes dc as normal doubles, exact
      to ~2^-1000), so this was a dispatch bound, not an arithmetic one. HARD-VALIDATED on the
      real GPU at a GENUINE deep boundary coordinate (new tools/gen-deep-coord.mjs descends a
      filament to ~2^-520; tools/probe-deep500.mjs compares vs the CPU oracle): **0.000%
      escape-count mismatch FLAT from 2^-120 (30k iters) to 2^-520 (130k iters)** — the
      per-iteration df64 accumulation is bounded by Zhuoran rebasing, it does NOT grow with
      depth/maxIter (the reference is BigInt-sampled, never iterated in df64, so it can't
      drift). + exterior arithmetic checks to 2^-600 in validate-gpu (real + SwiftShader).
      END-TO-END: the viewer renders the deep coordinate at zoom 2^500 as a clean fractal,
      engine gpu-perturb-fe, glitches=0, ~19s (screenshots/deep500.png; tools/shoot-deep500.mjs).
- [x] **AUTO-DROP SUPERSAMPLING AT EXTREME DEPTH (Spawn 10)**: below radius 2^-300 the
      effective ss is capped to 1 (SS_DEEP_CAP_RADIUS, viewer._effectiveSS) — ss² subsamples
      through the heavy fe/rescaled shader made a 2^500 frame ~4× slower (~19s vs ~75s) for AA
      the fractal's density mostly hides. The user's ss SELECT is unchanged; the debug line
      shows "capped from 2× for depth". The 2^-270 "ultra" bookmark keeps full ss (cap is past it).
- [x] **PORT SERIES APPROXIMATION TO THE GPU SHADERS (Spawn 16)** — the headline deep-zoom
      shader win. The rescaled deep engine seeds dz at iteration `skip` via an in-shader
      floatexp Horner (FE_LIB fe_mul/fe_add — NOT delicate, the library normalizes each op),
      coeffs/1/R passed as fe uniforms (uSAm[6]/uSAe[6], uInvR). Worker computes computeSeries
      (grid+escape-guard, NO coarse cap — proven non-binding by tools/probe-sa-cap.mjs). Measured
      on the REAL RTX 3090 (tools/bench-sa.mjs): **4.1× at 2^-120 (82% skip), 6.7× at 2^-271 (88%),
      9.3× at 2^-400 (92%)**. Validated GPU-with-SA vs the no-SA CPU oracle (validate-gpu.mjs SA
      section: 0.000% mism deep, same envelope as no-SA shallow) on SwiftShader AND the real GPU,
      IDENTICAL numbers; live-viewer end-to-end (probe-sa-viewer.mjs): 0.00% picture diff SA on/off,
      2.43× wall-clock. Rides the existing default-on "Series approximation" toggle (one switch,
      CPU+GPU). See NOTES "✅ GPU SERIES APPROXIMATION". NEXT: add the seed to perturbFragFloatexp
      (oracle), and a coarse GPU preview pass for coordinate jumps.
- [x] **HIGHER-ORDER SA → order 5 (Spawn 21)** — the SA seed polynomial raised order 3→5 (the same exact
      dc-series, more terms). Truncation (not the escape guard) binds the skip at production tol, so order 5
      extends it MOST at the moderate-deep gap: real RTX 3090 **1.24×/1.21×/1.13×/0.99× at 2^-120/-218/-271/
      -400** (bench:saorder:real), bit-exact, the first net speed win since SA. Depth-adaptive (order 5 below
      2^-112 where it's df64-clean + production; order 3 above). validate:gpu + validate:gpu:real ALL PASS.
      See NOTES "HIGHER-ORDER SERIES APPROXIMATION".
- [x] **SA SAFETY-MARGIN MINIMIZATION → depth-adaptive 0.02 (Spawn 22)** — the THIRD net-win lever, BIT-EXACT,
      shipped as default. The SA skip backs off marginFrac·validN to guard the GPU df64 seed; measured (probe/
      bench-samargin, real RTX 3090) that below 2^-112 (the df64-clean GPU-SA band) the shipping 0.05 bought
      ZERO precision (mism 0 + IDENTICAL drift even at margin 0) while discarding ~⅓ of the small deep post-SA
      budget. Cut 0.05→0.02 (depth-adaptive; 0.05 kept shallow): **1.12×/1.19×/1.18×/1.40× at 2^-120/-218/-271/
      -400**, robust to throughput-vs-divergence (a UNIFORM per-pixel cut, unlike a prune). Stacked on SA the
      deep band now hits/exceeds 10× (2^-271 ~9.2×, 2^-400 ~13×); 2^-120 ~5.7×→~6.4×. crosscheck-sa (0.02 AND
      0.01) mism 0, 41 unit, validate:gpu + validate:gpu:real ALL PASS. Closed two dead-ends same spawn (both
      measured NEGATIVE): region-adaptive SA skip (+1.4% only, sharp truncation cliff) and a LEAN occupancy
      kernel (built+validated bit-identical, but neutral-to-slightly-slower on the RTX 3090 — kept opt-in for
      mobile). Interior detection MEASURED big-but-deferred (61–90% interior @deep, but rigorous/risky/coord-
      dependent). See NOTES "SA SAFETY-MARGIN MINIMIZATION".
- [x] **OffscreenCanvas worker for the GPU (Spawn 24)** — the deep GPU draw now runs entirely off the
      main thread in a dedicated worker that owns an OffscreenCanvas + the WebGL context, pushing finished
      ImageBitmap strips back (no per-strip main-thread GL flush). DEFAULT when supported; pixel-identical
      to the legacy path; main-thread Long-Task blocking 99842ms→0ms at 2^-271 (SwiftShader). Context-loss
      recovery, recolor, glitch readback all preserved. See NOTES "✅✅ OFFSCREENCANVAS GPU WORKER".
- [x] **DEEP FLOOR → THE PRECISION WALL 2^-1010 (Spawn 25)**: pushed the GPU floatexp/rescaled floor
      2^-600 → **2^-1010**, the true double-dc limit (where the per-pixel dc/step go subnormal at the
      max compute resolution). VALIDATED bit-exact vs the BigInt-EXACT oracle (new tools/probe-wall.mjs,
      extending the M4 B/C/C2 methodology deep) at a genuine boundary coordinate: 0 escape-count mism FLAT
      2^-700→2^-1010 (dcRelErr ~3e-16 = pure double rounding); GPU fe/rs vs CPU oracle 0% flat 600→1010
      (probe-deep500, real RTX 3090); validate:gpu/:real exterior checks extended to 2^-1000, ALL PASS.
      Below 2^-1010 the viewer CLAMPS the radius (MIN_RADIUS) + signals "max depth" — graceful, never
      garbage. gen-deep-coord ITERCAP must scale with depth (was the descent's binding bug). See NOTES
      "DEEP PRECISION WALL".
- [x] **SHADER MICRO-OPT SWEEP (Spawn 27)** — the last two unmeasured inner-loop levers, measured on the
      real RTX 3090 with a NEW contention-robust protocol (the companion LLM shares the GPU — interleave
      many rounds, take MIN-of-min; a spurious 1.4× from its inference bursts was caught this way).
      (1) getZ SOFTWARE-PIPELINED prefetch in the rescaled engine (issue Z[m+1] a full iteration before
      first use): BIT-IDENTICAL by construction + measured (sn diff 0), **1.01–1.08× on the RTX 3090
      (growing with depth) and ~2.0× on SwiftShader** (the GPU-blocklisted-device + e2e/CI backend) —
      SHIPPED as default (`p.prefetch===false` = old placement). (2) `ds_sqr` dedicated df64 squaring
      (one Veltkamp split, 6 barriers vs 8, at the ~6 self-product sites/iter): measured NEUTRAL
      0.999–1.031× on the GPU (driver CSEs the duplicate split) and 0.95–0.98× on SwiftShader → kept
      OPT-IN (`{sqr:true}`/`p.sqrOn`), production compiles the historical source. Gates ALL PASS on the
      shipping config: probe:barrier sqr ladder (INTACT; sqr_none control collapses), validate:gpu +
      validate:gpu:real, probe:xbackend (historical envelope), crosscheck:skip/tiled (IDENTICAL), 41
      unit. VERDICT: per-iteration ALU micro-structure is now MEASURED-CLOSED (a 25%-fewer-ops squaring
      moves ≤3%); the frame is divergence/tail/latency-bound. The df64/fe prefetch port was measured
      same-spawn: 0.997× GPU / 0.957× SwiftShader — the 2× is rescaled-specific → opt-in there.
      See NOTES "⚖️ SHADER MICRO-OPT SWEEP".
- [x] **SKIP-AWARE STRIP BUDGET (Spawn 28)** — the strip-tiling-overhead FUTURE item, closed. `_stripRows()`
      budgeted worst-case on maxIter, ignoring that SA seeds every pixel at `skip` (93–95% deep) → 10–20×
      too many strips, each paying fixed costs (full-canvas colorize + ImageBitmap + post/ack; or a vsync
      rAF on the main path). Budgeting on `maxIter − skip` (callers mirror _setSA's active condition):
      deep frame **1.37–1.39× (offscreen default) / 3.2× (main-thread fallback)** at 2^-271/-400, ~76% of
      the single-strip ceiling, per-strip watchdog envelope UNCHANGED (same 4e8 on the true bound).
      Validated: crosscheck:offscreen pixel-identical, 41 unit, gpu+perf e2e green. New probe:
      `tools/probe-strips.mjs` (A/Bs prod/×8/single-strip on the live viewer). See NOTES "✅✅ SKIP-AWARE
      STRIP BUDGET".
- [ ] FUTURE (still open): multi-pass over ITERATIONS if a real device caps fragment-shader loop length.
      (Strip-tiling overhead closed by the skip-aware budget — Spawn 28; deep-floor push + "Force High
      Quality" override DONE — Spawns 25/13. Remaining strip follow-ups if ever needed: measured-adaptive
      strip growth; colorize only the strip's rows instead of full-canvas per strip.)

## M7 — Mobile UX polish
- [x] **ONBOARDING + KEYBOARD + COORDINATE HUD (Spawn 26)** — the "finished product, not a tech demo"
      polish pass once all engine frontiers closed. (1) First-run gesture hint (dismissible card, once per
      device via localStorage, re-openable via "?"; the overlay is pointer-events:none + dismisses in the
      capture phase WITHOUT consuming the event, so the first tap both dismisses AND zooms — and it never
      blocks the e2e canvas clicks). (2) Full keyboard nav: arrows pan / +−=zoom / f=fullscreen / ?=help,
      via a new `viewer.panByPreview()` that reuses the gesture-preview machinery (smooth, settles to one
      sharp render); discrete (ignores auto-repeat), inert in form fields. (3) Live center readout in the
      HUD in complex `a+bi` form (`shortCoord()` trims trailing zeros, ellipsizes only dropped significant
      digits; panel keeps full precision). (4) Fullscreen + help corner buttons (mobile parity for f/?).
      NO pixel/math change; 41 unit + full e2e (both projects) green, +5 new e2e. See NOTES "✅ UX POLISH".
- [x] responsive canvas/DPR — FULL screen resolution by default since Spawn 32 (pixel-budget guard
      replaced the old 1100px edge cap; "Resolution: Full/Half/Third" select); tiled render done (Spawn 7); low-power
      battery-saver mode done (Spawn 14 — DPR/backing/ss/maxIter caps + battery auto-detect)
- [x] **point filtering** (image-rendering: pixelated + preview imageSmoothingEnabled
      =false) → crisp, not bilinear-blurry, especially mid-zoom. (Spawn 4)
- [x] **supersampling** (1×/2×/3×/4×, default 2×): compute at ss× display res, box-
      average colors down (GPU color-shader / CPU compute-canvas downscale). Anti-
      aliases the boundary filaments — the "ultra" smooth look. UI select + URL hash.
      (Spawn 4)
- [x] palette options (4), iteration slider + **number input field** + auto toggle, coords
- [x] bookmarks / shareable deep-zoom coordinates (URL hash) + "Go" + presets
- [x] loading/progress UI (status line + reference-orbit progress)
- [x] zoom UX: any zoom (wheel/buttons/click/pinch) instantly cancels the
      in-flight render and shows the *current image scaled* as a preview; the
      sharp re-render is deferred until the zoom motion settles. A pointer tap with
      no movement no longer cancels a running render. (viewer.js zoomBy/_beginPreview)
- [x] **click-to-zoom (Spawn 5)**: a single click/tap recenters the view on the
      clicked point AND zooms in (radius×0.5); shift/ctrl/right-click zoom out
      (radius×2). Uses the same preview-transform + settle machinery (instant scaled
      preview, deferred sharp render). Replaced the old double-tap handler.
      (viewer.js clickZoom; e2e: click recenters+zooms, real-mouse-click wiring)
- [x] **NO BLACK SCREEN — progressive tile-over-preview (Spawn 16)**: the strip-tiled GPU
      render blits ONLY each strip's own display rows (viewer._blitGpuStrip), so a deep render
      REPLACES the low-res scaled preview with high-res tiles top-to-bottom instead of wiping the
      canvas to interior-black first. Final image byte-identical (golden e2e unchanged); 2 new
      deterministic e2e. (Danielle: "don't render black screen instead replace low res with high
      res as it comes in.")
- [x] **WebGL context-loss recovery (Spawn 11)**: mobile GPUs drop the context (memory
      pressure / backgrounding / driver reset). Before, a loss mid-render froze on a dead
      context with NO recovery (a lost context's GL calls don't throw, so _gpuFail never
      tripped). Now GpuRenderer handles webglcontextlost/restored (preventDefault + drop all
      GL handles → lazy recreate; `lost` getter), the viewer routes to the CPU pool while
      lost and re-renders on the GPU once restored (600ms grace before a CPU detour; gives up
      on the GPU after 3 losses to avoid a flip-flop). Validated on SwiftShader AND real GPU
      via WEBGL_lose_context (tools/probe-recover.mjs: shallow+deep+give-up; e2e gate). See
      NOTES "MOBILE STABILITY". (CPU is the validated oracle, so a lost render is slow, never wrong.)
- [x] **glitch overlay debug toggle (Spawn 12)**: the GPU perturb shaders already wrote a
      Pauldelbrot glitch flag to the sn texture's `.b` channel, but the viewer passed
      `glitchTol=0` so it was STRUCTURALLY never set — every GPU render reported a fake
      `glitches: 0`. A "Glitch overlay" checkbox (off by default) now enables the diagnostic
      (PURELY diagnostic — does NOT change escape/rebase or the rendered fractal), tints
      flagged pixels magenta in the color pass (`uShowGlitch` in COLOR_FRAG), and reads back
      the real flagged-pixel count for the status line (`renderer.countGlitches()`). Validated:
      31 unit, validate:gpu ALL PASS (non-regressive), new e2e forces a huge tol to light up
      the count + magenta end-to-end (PASS on SwiftShader + real GPU). See NOTES "✅ GLITCH
      DEBUG OVERLAY". (CPU path already reported a real count; its VISUAL overlay — per-pixel
      mask per band, tinted in colorize* — was added in Spawn 14, see M7 progress log / NOTES.)
- [x] **Force High Quality toggle (Spawn 13)**: a checkbox that overrides the extreme-depth
      supersampling auto-cap (Spawn 10's SS_DEEP_CAP_RADIUS = 2^-300 drops effSS→1 for speed).
      For deliberate slow high-AA deep captures. `viewer.forceHighQuality` (default false) makes
      `_effectiveSS()` skip ONLY the depth cap — the hard memory/texture caps still apply (no
      OOM). NOT persisted in the URL hash (per-session capture choice; a bookmark shouldn't
      hand a recipient a silently 4×-slower render). Debug line shows "(HQ: depth cap overridden)"
      when active. e2e asserts the override + that the memory cap still bites. See NOTES.

## M8 — Extensive integration tests  ✅ (perf budgets TODO)
- [x] e2e: load, render, pan, zoom, deep-zoom-by-coordinate, palette (18 tests)
- [x] golden fingerprint determinism test (reproducible render hash)
- [x] perturbation-vs-BigInt equivalence (Node, to 2^400) + full-pipeline deep
- [x] **performance budget assertions (Spawn 13)**: test/e2e/perf.spec.mjs — time-to-first-
      pixel (poll the canvas to first structured paint) + full-frame budget on the home
      view (GPU default + CPU-pool fallback). LOOSE catastrophic-regression guards by design
      (SwiftShader CPU GL + a contended shared host would flake a tight budget; ~8–24× over
      observed ~0.5–1.0s); deep-path regressions already caught by the deep-test timeouts.

See NOTES.md for architecture, math, and decisions.

---
## NEXT (priority order for the next spawn)
★★★★★★★★★★★★. ✅✅ **REFERENCE + SA REUSE — DONE (Spawn 30): deep zoom ticks ~510ms → ~100ms (3.07× over a
   tick sequence, ~5× on reuse ticks), the "big interactive win" shipped.** The viewer caches the reference
   orbit + BigInt tail + SA coeffs; zoom-ins extend the orbit bit-identically (iterateOrbit shared core,
   unit-gated) and reuse the SA coeffs (box-inclusion validity; worker-refresh every ~1000 maxIter growth —
   computeSeries, now the BIGGEST serial cost at 466–607ms deep, never runs on main). Drift-tolerant
   (≤8 view radii — wheel/pan/click-recenter keep hitting); prec headroom 64 bits (~64 zoom doublings per
   rebuild). probe-refcache = the gate (6/6 hits, 0.00% bulk diff vs cold). 44 unit, e2e 38/38, probe-wall
   spot clean. **Remaining perf ideas are small: skip re-uploading the unchanged ref texture to the GPU
   worker on cache hits (~ms), extend the cache to the CPU-pool path, mobile-device measurements. The
   interactive deep-zoom experience is now: first dive pays one build, then every tick is ~GPU-only.**
★★★★★★★★★★★. ✅✅ **FAST REFERENCE BUILD — DONE (Spawn 29): the deep frame's post-strip-fix dominant cost
   (the serial CPU BigInt orbit) cut 2.54×** (303→119.5ms @2^-400; toDouble's O(prec) toString bit-length
   probe was the sleeper — Number(BigInt) rounds correctly in one step for prec≤1000). escapeBigInt 5→3
   mults BIT-EXACT (1.55× — all BigInt-oracle probes run faster). Live deep frame: 467→388ms @2^-271,
   726→576ms @2^-400. Gates: probe-wall 0 mism FLAT 2^-700→2^-1028, validate:gpu[:real] ALL PASS, 41 unit,
   e2e. **The most concrete remaining perf item is now REFERENCE REUSE across same-center zoom steps**
   (skip the rebuild on wheel-zoom settles; needs prec headroom + BigInt-tail retention + un-transferring
   the ref arrays from the GPU worker — moderate build, the big INTERACTIVE win; design sketch in NOTES
   "FAST REFERENCE BUILD → What's NEXT"). See NOTES "✅✅ FAST REFERENCE BUILD".
★★★★★★★★★★. ⚖️ **SHADER MICRO-OPT SWEEP — DONE (Spawn 27): prefetch SHIPPED (1.01–1.08× GPU, ~2× SwiftShader),
   ds_sqr measured neutral → OPT-IN.** Danielle asked directly to "optimize the shader further"; the two
   remaining unmeasured inner-loop levers are now measured (contention-robust protocol — the companion LLM
   shares the RTX 3090; interleave rounds + MIN-of-min, medians lie). SHIPPED: the rescaled engine's getZ is
   software-pipelined (Z[m+1] issued a full iteration early; bit-identical, sn diff 0) — 1.01–1.08× real GPU
   growing with depth, **~2.0× on SwiftShader** (real users on GPU-blocklisted devices + the whole e2e/CI).
   NOT shipped: ds_sqr (one-split df64 squaring) — 1.00× GPU (driver CSEs the duplicate split), 0.95–0.98×
   SwiftShader → opt-in `p.sqrOn`, gate `probe:barrier` sqr ladder + `bench:sqr`. ALL gates green on the
   shipping config (validate:gpu[:real] 52 PASS, xbackend, skip/tiled crosschecks, 41 unit, gpu+perf e2e
   12/12). The df64/fe prefetch PORT was built + measured same-spawn: 0.997× GPU / 0.957× SwiftShader — the
   2× is RESCALED-SPECIFIC (its heavy fe escape block is what the fetch overlaps with) → df64/fe prefetch is
   opt-in, closed. **VERDICT: the per-iteration ALU frontier is now MEASURED-CLOSED — a 25%-fewer-ops
   squaring moves ≤3%, and every micro-lever is now measured on BOTH backends. The remaining structural
   items are the M6-FUTURE orchestration ones (strip-tiling overhead at extreme maxIter; mobile loop caps),
   or lock in. See NOTES "⚖️ SHADER MICRO-OPT SWEEP".**
★★★★★★★★★. ✅ **UX POLISH — DONE (Spawn 26): onboarding hint + keyboard nav + live coordinate HUD.** All the
   engine/throughput/precision frontiers are closed + validated (Spawns 9, 20–25), so the companion + I took
   the AGENDA's "lock in or polish UX" branch and closed the remaining DISCOVERABILITY / ACCESSIBILITY gaps
   that made a rock-solid engine still feel like a tech demo. Shipped (companion-ranked): (1) a first-run
   gesture hint — dismissible, once-per-device, pointer-events:none + capture-phase non-consuming dismiss so
   the first tap both dismisses AND zooms (and never blocks the e2e clicks); (2) full keyboard nav (arrows
   pan / +−=zoom / f=fullscreen / ?=help) via a new `viewer.panByPreview()` that rides the gesture-preview
   machinery (smooth, settles to one render, discrete, inert in form fields); (3) a live center readout in the
   HUD in complex `a+bi` form (`shortCoord()`, full precision still in the panel); (4) fullscreen + help corner
   buttons for mobile parity. NO pixel/math change; 41 unit + `viewer.spec` 26/26 + `gpu.spec` 10/10 +
   `perf.spec` 2/2 on desktop-chrome + the 5 new on mobile-chrome — ALL GREEN. See NOTES "✅ UX POLISH".
   **STATUS: the product is now self-explanatory on first open, keyboard-drivable on desktop, and always shows
   where you are — a finished viewer, not a demo. Remaining open items are STRUCTURAL (M6-FUTURE: strip-tiling
   overhead at extreme maxIter; mobile fragment-loop-length caps) — recommend the next spawn lock in, or pick a
   structural item only if a real device exposes a need. The UX surface is in good shape; further polish would
   be incremental (e.g. a share-sheet on mobile, a mini-map/breadcrumb, palette live-preview).**
★★★★★★★★. ✅✅ **DEEP PRECISION WALL — DONE (Spawn 25): floor 2^-600 → 2^-1010 + graceful clamp.** The
   GPU-throughput frontier is closed (Spawns 22–24), so the companion + I took the most concrete remaining
   item: push the deep floor to the TRUE double-precision wall and make it a SPECIFICATION (not a heuristic).
   Lowered GPU_PERTURB_FE_FLOOR 2^-600 → **2^-1010** = the radius where the per-pixel double dc/step go
   subnormal at the max compute resolution (8192px: 8192·2^-1023). VALIDATED bit-exact vs the BigInt-EXACT
   oracle at a genuine deep boundary coordinate (new `tools/probe-wall.mjs` — extends the M4 B/C/C2 method
   deep): **0 escape-count mism FLAT 2^-700→2^-1010, dcRelErr ~3e-16** (= pure double rounding); GPU fe/rs
   vs CPU oracle **0% flat 600→1010** (probe-deep500, real RTX 3090); validate:gpu/:real exterior checks
   extended to 2^-1000 (fe+rs), ALL PASS on SwiftShader AND the RTX 3090; 41 unit pass. Below 2^-1010 the
   viewer CLAMPS the radius (`MIN_RADIUS`) and signals "⚠ max depth" (status + debug + zoom readout); a
   zoom-IN at the wall is suppressed (no over-scale snap-back), zoom-OUT/pan still work — e2e covers it.
   gen-deep-coord's ITERCAP now auto-scales with depth (the descent fell INTO the set past ~2^-560 at the
   old fixed 140k cap → drifted off the filament). NET: a CAPABILITY + correctness specification (the
   engine now reaches ~10^304× zoom, vetted against ground truth), not a throughput change. See NOTES "DEEP
   PRECISION WALL". STATUS: the deep band is now both fast (≥2^-218 ~10–13×) AND validated to the arithmetic
   limit. Remaining open items are structural/UX (strip-tiling overhead at extreme maxIter; mobile device
   loop-length caps) — recommend locking in, or polishing UX.
★★★★★★★. ✅✅ **OFFSCREENCANVAS GPU WORKER — DONE (Spawn 24): GPU raster off the main thread, SHIPPED as the
   DEFAULT.** Danielle's pivot away from raw throughput (closed in Spawn 23) toward mobile RESPONSIVENESS — her
   #1 option. The GPU escape+color strip loop now runs in a dedicated worker (`src/gpu/gpu-worker.js`) that owns
   an OffscreenCanvas and pushes finished ImageBitmap strips back; the main thread only draws ready bitmaps (no
   per-strip GL flush/sync). PIXEL-IDENTICAL to the legacy on-main-thread path (`crosscheck-offscreen`: 0.000%
   diff, maxΔ 0 across naive/df64/rescaled-fe on SwiftShader + RTX 3090 — same shaders, lossless ImageBitmap).
   THE WIN (`bench-offscreen-jank`, deep 2^-271, SwiftShader): main-thread Long-Task blocking **99842ms → 0ms**,
   rAF heartbeat **55 → 5061 ticks** (UI animation survives the whole render); render TIME ~unchanged (throughput
   is warp-bound, as expected — the win is responsiveness, not ms/frame). Mobile context-loss recovery preserved
   through the worker (`probe-recover-offscreen` ALL PASS: loss forwarded worker→main → CPU detour → GPU on
   restore). Robust fallback: a sync capability probe declines where OffscreenCanvas/WebGL2 is absent → the
   on-main-thread GpuRenderer unchanged → CPU pool. Validation: 41 unit, crosscheck-offscreen 0.000%, probe-
   recover-offscreen, the offscreen default-path e2e all PASS; 3 main-thread-GPU-internals e2e pinned to
   `?offscreen=0` (validate the still-shipping fallback); a latent determinism wait-race fixed. NOTES "✅✅
   OFFSCREENCANVAS GPU WORKER". **STATUS: the deep-band throughput frontier stays CLOSED (Spawn 23); this delivered
   the M6-FUTURE OffscreenCanvas item — a fundamental execution-model shift, not a speedup. Remaining clean ideas
   are STRUCTURAL/UX: fewer per-iter texture fetches, push the floor below 2^-600 (toward the ~2^-1000 double-dc
   limit). The "900% faster" raw-speed goal is MET deep (≥2^-218 ~10–13×) and structurally bound at the moderate
   band (2^-120 ~6.4×). Recommend the next spawn lock in OR take a structural item (the floor-below-2^-600 push is
   the most concrete: generate a deeper boundary coordinate, confirm the chaotic band stays 0% below 2^-520).**
★★★★★★. ⛔ **INTERIOR DETECTION — MEASURED DEAD (Spawn 23). The "one remaining BIG lever" is REFUTED; the
   GPU optimization frontier is now CLOSED — lock in.** Danielle asked: greenlight a rigorous attracting-cycle
   detector, or lock in? Answer via a measure-first UPPER-BOUND experiment (`tools/bench-interior-oracle.mjs`):
   an EXACT CPU-oracle interior mask lets oracle-interior pixels bail the GPU loop at a swept iteration; `frac=0`
   (bail at skip) = a PERFECT/FREE/zero-latency detector = the absolute CEILING (any real detector is strictly
   worse). Real RTX 3090: **ceiling 1.55×/1.29×/1.39×/0.99× @2^-120/-218/-271/-400** — even 92%-interior 2^-271
   tops out at 1.39×, and the curves SATURATE by frac≈0.1–0.25. WHY: the late-escaping pixels run ~81% of the
   post-SA budget (probe-interior) and are spatially sprinkled, so they anchor almost every warp to ~0.8·tail —
   pruning the interior bulk is invisible. SAME structural ceiling that killed BLA + the lean kernel: the deep
   frame is WARP-DIVERGENCE/escaper-tail bound, NOT bulk-work bound. A real detector adds latency (fires past the
   saturation window) + a per-iter derivative tax on ALL pixels (incl. the 35–43% escapers) → trends to ≤1.0×
   like BLA. DECISION: do NOT build it. The ONLY non-dead branch is the CPU FALLBACK (no warp divergence) but
   it's the slow rarely-hit path — low value, not greenlit. Validation: prune-off is bit-identical (uPrune=0
   short-circuits); 41 unit + validate:gpu + validate:gpu:real ALL PASS (the added shader IR did NOT shift the
   df64 collapse threshold on the real GPU); crosscheck escDiff=0. Bench infra kept opt-in (uPrune/uPruneMask,
   off by default — same pattern as BLA/df64-escape/lean), NOT wired to production. NOTES "⛔ INTERIOR DETECTION —
   MEASURED DEAD". **STATUS: the "900% faster" goal is MET at the deep band (≥2^-218, ~10–13×) and ~64% (6.4×) at
   the moderate 2^-120 band, which is now PROVEN structurally bound — every measured lever (BLA, lean, region-SA,
   interior detection) hits the same divergence/escaper-tail wall. The clean remaining ideas are STRUCTURAL and
   UX, not raw deep-band speed: OffscreenCanvas (off-main-thread, M6 FUTURE — responsiveness, not throughput),
   fewer per-iter texture fetches, push the floor below 2^-600. Recommend locking the current state in.**
★★★★★. ✅✅ **SA SAFETY-MARGIN MINIMIZATION — DONE (Spawn 22): the THIRD net-win lever, SHIPPED.** Depth-adaptive
   margin 0.05→0.02 below 2^-112 (the df64-clean GPU-SA band). MEASURED the margin bought ZERO precision deep
   (probe-samargin: mism 0 + identical drift down to margin 0) while discarding ~⅓ of the deep post-SA budget.
   Real RTX 3090 (bench-samargin): **1.12×/1.19×/1.18×/1.40× @2^-120/-218/-271/-400**, bit-exact, robust to the
   GPU binding model (uniform per-pixel cut). Deep band now ~10×+ (2^-400 ~13×); 2^-120 ~5.7×→~6.4×. NOTES "SA
   SAFETY-MARGIN MINIMIZATION". The OPEN frontier (next spawn): the moderate band (2^-120 ~6.4×) is bounded by
   the chaotic post-rebase tail. Measured-NEGATIVE/deferred this spawn: region-adaptive SA (dead), lean kernel
   (neutral, opt-in for mobile), **interior detection** (measured 61–90% interior @deep & 65–91% of work = the
   one remaining BIG lever, but a rigorous/risky/coordinate-dependent research bet — needs Danielle's call OR a
   throughput-vs-divergence GPU measurement first). Structural levers unchanged: fewer fetches / less divergence
   / off-GPU (OffscreenCanvas). **The "900% faster" target is now MET at the deep band (≥2^-218) and ~64% of the
   way (6.4× of 10×) at the moderate 2^-120 band — a genuine decision point: keep grinding the tail, or lock in.**
★★★★. ✅✅ **HIGHER-ORDER SERIES APPROXIMATION (order 5) — DONE (Spawn 21): a NET-WIN speed lever, SHIPPED.**
   The first net speed win since SA itself. Raised the SA polynomial order 3→5 (`a..c` → `a..e`, the SAME
   exact dc-series, just more terms). MEASURED-FIRST (`probe:saorder`): at production tol the skip is bounded
   by TRUNCATION not the escape guard, so more terms = larger skip — biggest at the moderate-deep gap
   (skip% +4.7 @2^-120, +0.5 @2^-400). REAL RTX 3090 wall-clock (`bench:saorder:real`): **1.24× @2^-120,
   1.21× @2^-218, 1.13× @2^-271, 0.99× @2^-400** — exactly the gap shape, free at extreme depth, bit-exact
   (can only be neutral-or-positive, never a BLA-style loss). 2^-120 now ~5.7× vs no-SA (was ~4.6×). The
   post-rebase floor did NOT swallow it (the earlier worry was wrong). ONE trap fixed: order-5's larger
   skip seeds at a later iteration → bigger df64 (~46-bit) seed error → amplifies on the SHALLOW chaotic
   seahorse band (validate:gpu FAIL, meanΔsn 0.9→1.4). It's df64-CLEAN on the deep boundary coord
   (`probe:saorder:gpu`: 0% mism, meanΔsn ~5e-4 @2^-100..-130), and production renders the shallow band on
   df64-no-SA anyway → DEPTH-ADAPTIVE order (order 5 below 2^-112 = the df64→rescaled dispatch floor, order 3
   above; `SA_ORDER5_RADIUS`). Rides the existing default-on "Series approximation" toggle (CPU+GPU). All
   green: 40 unit, crosscheck-sa 0 mism, validate:gpu + validate:gpu:real ALL PASS. FULL writeup NOTES
   "HIGHER-ORDER SERIES APPROXIMATION". Still short of 10× at 2^-120 (chaotic post-rebase tail floor — BLA
   is its tool but a GPU net-loss). Open structural levers UNCHANGED: fewer per-iter fetches / less warp
   divergence / off-GPU work (OffscreenCanvas). Order 7+ rejected (probe:saorder: <2% more skip, df64/register risk).
★★★. ✅✅ **df64 BARRIER MINIMIZATION — DONE (Spawn 20): a CORRECTNESS FIX + 1.64× per-op, SHIPPED as default.**
   The shipping MAXIMAL df64 barrier (32 ob() per mul+add) was CURRENTLY BROKEN on the real RTX 3090 —
   validate:gpu:real = **24 FAIL**, a clean depth-monotonic df64→f32 collapse (3.9% @2^-3 → 99% @2^-70). Cause
   (companion-vetted): the maximal barrier blows the inlined shader past an NVIDIA compiler complexity threshold
   that phase-switches to a coarse optimizer which STRIPS the barriers en masse — vindicating Spawn 8's original
   "the driver defeats the barrier in the LARGE shader" (Spawn 9's highp-sampler fix was a SEPARATE bug). The
   MINIMAL placement (ds_mul 8, ds_add 5 = 14 vs 32) blocks both reassociation + FMA contraction, stays under the
   threshold, and is **validate:gpu:real ALL PASS + 1.64× faster per op** (fair fixed-work bench). End-to-end: the
   2^-50 viewer renders a clean fractal (screenshots/df64_2e50_gpu1.png, glitches=0). Shipped as the default (NOT a
   toggle — full is broken, so no fallback to it; SwiftShader stays the universal-correct fallback). NEW GATE for any
   DF64_LIB edit / new target GPU: `npm run probe:barrier` (isolated ladder) + probe:xbackend + validate:gpu:real +
   `bench:barrier`. Portability is empirical-per-GPU, not spec-guaranteed (companion's caveat). FULL writeup in NOTES
   "df64 BARRIER MINIMIZATION". This is the first GPU lever since SA that is a NET WIN — and it makes the deep GPU
   path CORRECT on real HW at all, so SA/rescaled are now actually deliverable. NEXT real levers stay: fewer fetches /
   less divergence / off-GPU work (OffscreenCanvas) — the per-iter REPRESENTATION/ALU floor is now genuinely minimal.
★★. ⚠️ **df64-ESCAPE (the documented NEXT-#2 lever) — DONE (Spawn 19), CORRECT but PERFORMANCE-NEUTRAL: kept OFF
   (opt-in).** The rescaled engine's per-iteration escape/rebase/glitch test ran in floatexp every iteration; it now
   CAN run in plain df64 in the common case (it only executes once |dz| is O(1) — true at ALL depths — so it's in
   df64 range; fe's only edge is exponent RANGE, same ~46-bit mantissa). Built + validated bit-for-bit-as-precise on
   SwiftShader AND the real RTX 3090 (numbers MATCH the historical fe-escape numbers to the digit — no glitch line),
   but MEASURED ~1.00× on both (SwiftShader all depths; real GPU 512² compute-bound). NEUTRAL, not a net-negative like
   BLA. Cost is the barriered ds_mul squarings + getZ fetch (shared by both paths), NOT the fe-normalize wrapper df64
   removes. Independently CONFIRMS the BLA finding: the ~8ms SA floor is not per-iteration-ALU-bound. Kept OFF behind
   `uDf64Esc` (default-off; bit-identical when off) for future weak/mobile-GPU measurement. Companion LLM endorsed
   the gate + the keep-off call. FULL writeup NOTES "df64 ESCAPE". The real future GPU lever: fewer fetches / less
   divergence / off-GPU work (OffscreenCanvas, M6 FUTURE) — NOT per-iteration ALU/representation micro-opts.
★. ⚠️ **GPU-PORT BLA — DONE (Spawn 18), but a MEASURED PERFORMANCE NEGATIVE: kept OFF.** The port is
   built + validated CORRECT on the real RTX 3090 (floatexp table texture — coeffs reach 2^±hundreds deep
   so plain df64/float would overflow/underflow; two-gate validation: FAITHFULNESS GPU-BLA-vs-CPU-BLA at
   all depths + CORRECTNESS GPU-BLA+SA-vs-no-BLA-oracle deep, all PASS). BUT it does NOT accelerate the
   GPU: SA+BLA is ~1.6–2× SLOWER than SA alone (bench-bla on the deep coord: SA = 4.2×/6.6×/8.8×, SA+BLA =
   2.6×/3.4×/4.8× at 2^-120/-271/-400). BLA's CPU iteration-work reduction (2.5–3.3×) is real on the GPU
   too, but each BLA step costs far more than a rescaled iteration — texture-fetch latency vs ALU, plus
   warp divergence in the chaotic post-SA tail where BLA would help. THREE scan optimizations (level-1
   early-out, binary-search the level, carry |dz|²) each helped but never flipped the sign. SA is the right
   + sufficient GPU lever; BLA is CPU-centric (companion LLM independently confirmed). Code stays OFF (the
   no-BLA path is bit-identical), validated infra preserved for future HW / a smarter coherent-warp scheme.
   FULL writeup in NOTES "⚠️ BLA GPU PORT". The real next GPU levers are #1/#2 below (table-scan-free speed).
   ✅ **RE-MEASURED ON THE LEAN SHADER (Spawn 20): verdict HOLDS — still a net loss.** The Spawn-18 bench ran
   with the FULL barrier (which Spawn 20 found COLLAPSES on the current driver in this exact chaotic deep band).
   Re-benched on the correct lean shader: SA+BLA = 2.13×/3.22×/4.67× vs SA-alone 4.27×/6.76×/9.31× — BLA still
   ~2× SLOWER (penalty is structural: texture latency + post-SA-tail warp divergence, barrier-independent, as
   predicted). df64-escape + SA also re-confirmed (neutral / unchanged). So the barrier collapse did NOT flip
   any optimization verdict — only correctness. BLA stays OFF. (crosscheck:gpu:bla:real not re-run; FAITHFULNESS
   is what its correctness gate checks and that's GPU-vs-CPU-BLA, robust to the shared shader fix.)
0. ✅ **DONE (Spawn 9) — THE DEEP GPU CORRECTNESS BUG IS FIXED.** Root cause was NOT the
   compiler defeating the barrier (Spawn 8's hypothesis was wrong). It was sampler
   precision: the reference samplers defaulted to **mediump**, so NVIDIA truncated the
   per-iteration texelFetch'd df64 reference Z to ~fp16 (~10-bit), destroying df64.
   SwiftShader implements mediump as fp32, which hid it for 6 spawns. FIX: declare every
   reference sampler `highp sampler2D` (uRef ×4 perturb shaders + uSn in the color pass).
   Verified: probe-xbackend 90%→0.00%, validate:gpu:real 12 FAIL→0/ALL PASS, crosscheck
   skip+tiled 0-diff on GPU, 31 unit + SwiftShader validate unchanged, a real 2^-50
   render is a clean fractal. The XOR barrier stays (separate Veltkamp-split fix). See
   NOTES "✅ THE DEEP-GPU BUG IS FIXED". Original item kept below for context:
0b. **⚠️ (RESOLVED) FIX THE DEEP GPU CORRECTNESS BUG ON REAL HARDWARE (Spawn 8 found it).** Real-GPU
   testing now works (`GPU=1`/`GPU=gl`, see NOTES "⚠️ REAL-GPU TESTING"). It immediately
   exposed that the deep df64/floatexp/rescaled engines are NUMERICALLY WRONG on real NVIDIA
   (20–99% wrong pixels ≥ ~2^-12) — every prior spawn validated only on SwiftShader, which
   hid it. Cause: the NVIDIA shader compiler reassociates the Veltkamp split (`ca-(ca-x)→x`),
   collapsing df64→float32. A per-op XOR-uniform barrier (shipped in DF64_LIB) fixes the
   ISOLATED ops but the driver still defeats it in the large inlined perturbation shader
   (validate:gpu:real still 12 FAIL). TWO tracks:
   (a) **SHIP NOW, independent of the shader: a GPU deep-precision self-test + CPU fallback.**
       First deep render → render a small known tile on GPU, compare escape counts to the CPU
       oracle (escapePerturb, already in-browser); if mism > tol, mark GPU-deep untrusted and
       route deep renders to the CPU pool. Makes the viewer CORRECT on ALL hardware today.
       THE highest-value action — real users currently get wrong deep fractals on real GPUs.
   (b) Crack the shader: stronger/un-CSE-able barrier, or split the shader so the driver has
       less to reassociate. Inner-loop signal: probe-xbackend (GPU-df64 vs SwiftShader-df64,
       should → ~0%); gate: validate:gpu:real. Needs NVIDIA ISA introspection ideally.
   NOTE: strip-tiling (Spawn 7) AND fast-skip (Spawn 6) ARE bit-identical on the real GPU
   (crosscheck-tiled/skip with GPU=1 → 0-diff) — those mechanisms are sound; only the df64
   precision is broken. The 2^218 watchdog win still can't be directly measured (the tiny
   bench/crosscheck sizes finish instantly; need a full-screen deep frame timing on real HW).
1. **Real-GPU validation of the rescaled engine + the fast-skip** (Spawn 6 built both,
   validated on SwiftShader). On a real GPU: (a) confirm the rescaled deep band is
   visibly faster than fe and free of boundary sparkle (validate-gpu has ~1% headroom;
   the rescaled section passes it, but a real GPU + a wider sweep is the final check);
   (b) the fast-skip is ~1.00× on the chaotic SwiftShader bench (SIMD divergence) but
   should help SMOOTH deep regions where whole pixel-groups skip — measure that win on
   real hardware. If the rescaled engine ever misbehaves, viewer._renderGpuPerturb can
   fall back to renderPerturbFloatexp (still in the renderer, still validated) in one line.
2. ⚠️ EXPLORED (Spawn 19) — NEUTRAL, kept off: the rescaled UPDATE is now ~df64-cheap, so the
   escape/rebase block (was exact floatexp every iter) was the suspected new bottleneck. Built the
   df64-common-case / fe-near-minimum split (the "cheaper-common-case escape/rebase" below): correct +
   bit-for-bit as precise (it only fires once |dz| is O(1), in df64 range at ALL depths), but MEASURED
   ~1.00× on SwiftShader AND the real GPU — the cost is the barriered ds_mul squarings + getZ fetch
   (shared by both paths), NOT the fe-normalize wrapper. Kept OFF (`uDf64Esc`, opt-in). See ★★ above
   + NOTES "df64 ESCAPE". The "~half the per-iter cost" estimate did NOT translate to a speedup.
   [original idea, now answered:] the escape test fires when |dz| is large (S high) and rebases are
   CORRELATED across pixels (shared reference near-0 passages, same m), so df64-when-z-is-O(1)/fe-near-
   minimum was plausible — it's correct but doesn't pay off; the bottleneck is fetch+ds_mul, not fe.
3. PERF + PRECISION on a real GPU/device (the floatexp path is correct but HEAVY;
   Spawn 5 made it ~2× faster on SwiftShader but it's still a CPU SW rasterizer):
   - verify 2^270 renders fast; if a single fe draw is too long, strip-tile it
     across draw calls (scissor + yield) and/or move the GPU renderer into a Worker
     via OffscreenCanvas (also fixes the mobile-watchdog/TDR risk for the df64 path).
   - REGISTER PRESSURE: fe (df64 mantissa + int exp + perturb state) uses many
     registers; if it spills, occupancy and speed tank non-linearly. A "High
     precision (deep)" vs "Fast" UI toggle could be a safety valve.
   - ✅ supersampling at extreme depth: Spawn 10 auto-drops ss→1 below 2^-300
     (SS_DEEP_CAP_RADIUS) — a 2^500 frame went ~75s→~19s. ✅ The "Force High Quality"
     override toggle is DONE (Spawn 13). A battery/low-power variant is the remaining refinement.
   (Reference Z is NOT a drift risk: render() recomputes the reference fresh at the
   current center + relocates to the deepest pixel every view change — no staleness.)
3. ✅ DONE (Spawn 10): floatexp floor pushed 2^-340 → 2^-600 (covers the 2^500 target),
   HARD-VALIDATED on a genuine deep boundary coordinate (0.000% mism FLAT to 2^-520, 130k
   iters) + exterior arithmetic to 2^-600. NEXT extension: push below 2^-600 toward the
   ~2^-1000 double-dc limit — generate a deeper boundary coordinate (gen-deep-coord.mjs,
   raise TARGET) and confirm the chaotic band stays 0% below 2^-520. df64 reference (46-bit)
   showed NO depth-dependent degradation (it's BigInt-sampled, not iterated), so storing the
   reference as fe is likely unnecessary even very deep — but re-check if noise ever appears.
4. ✅ DONE (Spawn 12): Glitch overlay — the GPU shaders' Pauldelbrot flag (sn `.b` channel)
   is now surfaced as a magenta debug overlay AND read back for a real GPU glitch count
   (was a structural 0 because glitchTol was hard-0). Off by default; purely diagnostic.
   NEXT extension: a CPU-path visual overlay (the worker would post a per-pixel glitch mask;
   deferred — the count is already honest there and the GPU is the default deep path).
5. ✅ DONE (Spawn 13): Perf-budget e2e assertions (test/e2e/perf.spec.mjs — time-to-first-
   pixel + full-frame on the home view, GPU default + CPU fallback; loose catastrophic-
   regression guards) AND the "Force High Quality" toggle (override the deep ss cap for
   slow deliberate captures). See NOTES "✅ FORCE HIGH QUALITY toggle + PERF-BUDGET e2e".
   ✅ DONE (Spawn 14) — both remaining items in this bucket closed:
   (a) **CPU-path VISUAL glitch overlay**: the workers now post a per-pixel Pauldelbrot
       mask per band (+ coarse pass) when showGlitches is on; colorizeRegion/colorizeBlocks
       tint flagged pixels 0.6 toward magenta — mirrors the GPU sn `.b` overlay exactly.
       Gated so default renders are byte-identical (no mask built; sn never depends on the
       diagnostic). glitchTol threaded to the workers so the count/mask are tunable like the
       GPU uniform. setShowGlitches off → cheap CPU recolor (re-tints from a cached mask).
       See NOTES "✅ CPU GLITCH OVERLAY + LOW-POWER MODE".
   (b) **Low-power / battery mode**: viewer.lowPower (manual "Low power" checkbox) caps DPR
       (→1) + backing (1100→700), forces ss→1, and lowers the AUTO iteration ceiling (→2000;
       a user-typed maxIter is still honoured). main.js auto-enables it from navigator.getBattery
       (discharging + ≤20%) until the user touches the toggle. Not URL-persisted (per-device
       energy choice, like Force HQ). See NOTES.
6. ✅ DONE on the CPU path (Spawn 15): **series approximation** — skip the leading
   perturbation iterations for every pixel via a polynomial-in-dc seed. NOT diminishing
   returns: it is orthogonal to rebasing (rebasing = precision/glitches, SA = iteration
   count/speed) and the deep-zoom skip is HUGE (82% at 2^-120, **92% at 2^-500** → ~2.5–2.9×
   faster CPU render), bit-exact escape counts (0 mism 2^-50…2^-500). src/math/series.js;
   default-on "Series approximation" toggle; gated by a probe-grid + escape-guard + a
   coarse-pass min-escape cap; tol 1e-10 (load-bearing). See NOTES "SERIES APPROXIMATION".
   **NEXT: the GPU port** (the headline-speed win — the 2^500 GPU frame is ~19s, ~10× cuttable;
   the deep fe/rescaled band needs a floatexp in-shader Horner). The CPU path is now the
   bit-exact oracle for that port. Full plan in NOTES "NEXT — GPU PORT".

## Progress log (newest first)
- Spawn 34 (Danielle: "include the screen resolution and AA in the URL"): `res=` now rides in the hash
  alongside the existing `ss=` (AA was already there) — writeHash records it, readHash applies it with a
  proper RESIZE (backing dims change, not just a re-render; the interim render is cancelled by the
  Spawn-33 cancel). This user-overrides the Spawn-32 "don't persist res" decision; the panel select
  keeps a shared reduced-res link visible + undoable. Verified both directions headless (load res=2&ss=3
  → applied + select synced; select change → hash updated); viewer e2e green.
- Spawn 33b (Danielle revealed she's on a PIXEL 8 — Mali/Tensor G3, a GPU class never validated here):
  built the Spawn-8 backlog item — a runtime GPU precision SELF-TEST. After the first deep GPU frame per
  session, 8×8 sampled pixels are replayed through the CPU oracle (same ref/SA/geometry via the ref
  cache; throwaway worker; ~ms). >25% escaper mismatch ⇒ persistent warning; per Danielle's explicit
  choice it is WARNING-ONLY (no forced CPU fallback — the GPU toggle is the manual escape). Detects
  df64/driver breakdowns on untested hardware (Mali!). Test hook __forceVerifyFail; both paths verified
  headless; crosscheck:offscreen pixel-identical; 44 unit; gpu+viewer e2e green. If her Pixel warns →
  next item: Mali-specific barrier variant. ALSO: mobile perf guidance — full-res (Spawn 32) is ~2.8×
  heavier on a Pixel 8; Resolution Half restores the old cost; the interrupt fix (33) matters MOST there.
- Spawn 33 (Danielle: "can't interrupt a mid-render zoom; UI laggy until it finishes"): found a
  Spawn-24-era omission — _beginPreview/render() never sent the GPU worker a `cancel` (only context-loss
  did), so abandoned frames rendered ALL remaining strips at full GPU cost and (on ref-build supersedes)
  delayed the next frame; full-res frames (Spawn 32) made the tail user-visible. Fixed: cancel posted on
  gesture start + render supersede; worker strip loop gains a macrotask yield (ack-before-cancel message
  ordering meant one extra strip was submitted before the cancel was seen). probe-interrupt.mjs (new)
  A/Bs vs the stubbed old behavior: stale-strip tail after zoom 8ms→≤1ms on the idle 3090 (the win scales
  with strip duration — slow/contended GPUs + full-res deep frames, the reported conditions). Gates:
  crosscheck:offscreen ALL PIXEL-IDENTICAL, 44 unit, gpu+viewer e2e green. Designed latency unchanged:
  settle = 220ms debounce + frame; interruption = instant preview + freed GPU.
- Spawn 32 (Danielle identified the real "blocky/glitchy fine detail" cause: render res < screen res):
  removed the mobile-era MAX_BACKING=1100 long-edge cap — the backing store now follows the TRUE canvas
  resolution (css×dpr, dpr≤2), guarded by MAX_BACKING_PIXELS=9e6 (≈4K fullscreen) + low-power's stricter
  caps. New "Resolution: Full/Half/Third" panel select (`viewer.setResScale`; NOT URL-persisted — no
  shared-link degradation traps). Verified: 1600×900 window renders 1600×900 (was 1100×619, the 2.1×-
  fewer-pixels blockiness with the pixelated upscale); Half→800×450. e2e 38/38. The Spawn-26 UX pass +
  ref cache keep it snappy despite more pixels; Half/Third is the speed valve.
- Spawn 31b (Danielle's second report: "glitchy fine details at a 2^-251 URL, even on hard refresh —
  maybe the glitch-detection math?"): full-ladder investigation at HER coordinate — an early-escaping-
  reference location (refLen ~3.5k vs maxIter 63k, SA skip near the ref end; a corner our standard
  coordinate never stresses). RESULT: engine math EXONERATED — probe-wall at her coordinate 0/144 vs
  BigInt-exact; GPU rs+SA vs CPU oracle 0.012%; the ref cache never engages there (escaped ref ⇒ MISS)
  and warm==cold 0.00%. TWO real URL-state traps found: (1) readHash pinned autoIter=false on EVERY
  i=-bearing URL → her session carried i=26100 (auto for 2^-103) down to 2^-251 — FIXED (pin only when
  i differs >2% from auto; echo-of-auto keeps auto ON; verified both semantics headless; round-trip e2e
  unaffected); (2) her URL carried ss=1 — fine-filament ALIASING reads as "glitchy fine detail" and
  resolves on zoom-in (by design; suggested ss=2). New tools: probe-repro-url.mjs, probe-repro-gpu.mjs
  (keep — the GPU-vs-oracle-at-arbitrary-coordinate pattern + the escaped-ref location class coverage).
- Spawn 31 (same live session; Danielle REPORTED A BUG in the fresh reference-reuse cache: "screen is
  just blue sometimes / click-zoom sends me somewhere different than where I clicked"): root cause = a
  DRIFT-SIGN FLIP in _refCacheUsable (stored VIEW−REF; consumed as REF−VIEW) — any drifted cache hit
  (every real click/wheel zoom) rendered the location MIRRORED about the cached reference and let the SA
  seed run outside its validated box (uniform one-hue frames). One-line fix; probe-refcache hardened with
  a DRIFTED zoom sequence (the original probe zoomed at the exact center where drift is 0.0 — geometrically
  blind to sign bugs) and PROVEN to catch the class by re-introducing the bug (drifted diff 77.21% FAIL
  bugged → 0.00% PASS fixed). Danielle confirmed fixed in live use. e2e re-run green. LESSON in NOTES:
  geometric-transform guards need non-degenerate inputs + a demonstrated failure.
- Spawn 30 (same live session; Danielle: continue): REFERENCE + SA REUSE — the recorded "big interactive
  win", built + shipped. Flame-graph surprise en route: computeSeries (466–607ms in Node at depth) had
  quietly become the BIGGEST serial cost, bigger than the ref build. The viewer now caches the reference
  orbit (+ BigInt tail, + prec headroom of 64 bits) AND the SA coeffs; zoom-in ticks EXTEND the orbit
  (bit-identical resume — iterateOrbit shared core, 3 new unit tests) and REUSE the SA coeffs (valid: the
  new dc box ⊂ the validated box; invR self-consistent), refreshing SA in the worker every ~1000 maxIter
  growth. computeSeries never runs on the main thread. Measured (probe-refcache, real GPU, 2^-350 + 6
  ticks): warm 135/105/107/99/455/121ms vs cold ~504–539ms — 3.07× overall, ~5× on reuse ticks; final
  frame bulk diff 0.00% vs cold. Gates: 44 unit, desktop e2e 38/38, probe-wall spot 0 mism @2^-700/-1010.
  Deep zoom now feels INSTANT between rebuilds. Cache is GPU-path only; cold loads (bookmarks) unaffected.
- Spawn 29 (same live session; Danielle: "more optimize?"): the strip fix exposed the CPU BigInt
  REFERENCE BUILD as the deep frame's dominant cost (~450ms of 726ms at 2^-400; GPU only ~180ms).
  Three src/math changes (no shader/GPU edits): (1) 2-mult complex square in computeReference
  ((bx+by)(bx−by) — same rounding class, 1.13×); (2) escapeBigInt 5→3 mults/iter by carrying the new-z
  squares — BIT-EXACT, 1.55× (probe-wall/arbiter grids run proportionally faster); (3) toDouble fast
  path — Number(BigInt) rounds correctly in one step for prec≤1000, killing the O(prec) toString-per-
  call bit-length probe (the sleeper; ×2 calls/iter). Stack: reference build 303→119.5ms at 2^-400
  (2.54×); live deep frame 467→388ms @2^-271, 726→576ms @2^-400. Gates: probe-wall 0 mism FLAT
  2^-700→2^-1028 (covers both toDouble paths), validate:gpu[:real] ALL PASS, 41 unit, gpu+perf e2e.
  New tools/bench-ref.mjs. NEXT recorded (not built): reference REUSE across same-center zoom steps
  (needs prec headroom + orbit extension + un-transferring the arrays — the big interactive win).
- Spawn 28 (same live session; Danielle: "more optimization?"): closed the M6-FUTURE strip-tiling-overhead
  item with a ONE-FORMULA fix — `_stripRows()` now budgets on the post-SA-skip worst case (maxIter−skip;
  callers mirror _setSA's active gating). probe-strips.mjs (new) measured the live viewer first: 100 strips
  of 6 rows at 2^-400 → ~35% of the offscreen frame was per-strip orchestration and the main-thread path was
  vsync-anchored (100 rAFs ≈ 1.67s). After: 5 strips → 1.37–1.39× offscreen, 3.2× main-thread; ~76% of the
  single-strip ceiling; watchdog envelope unchanged (same 4e8 on the true bound). Gates: crosscheck:offscreen
  pixel-identical, 41 unit, gpu+perf e2e 12/12, viewer spec green. Deep-frame wall-clock now ~0.47–0.73s at
  2^-271/-400 on the RTX 3090 INCLUDING the CPU ref build.
- Spawn 27 (Danielle, direct: "optimizing the shader further"): measured the two last unmeasured inner-loop
  micro-levers on the real RTX 3090. SHIPPED: software-pipelined getZ prefetch in the rescaled deep engine
  (bit-identical — sn diff 0 by construction and by measurement; 1.01–1.08× real GPU growing with depth,
  ~2.0× SwiftShader). NOT SHIPPED (opt-in p.sqrOn): ds_sqr one-split df64 squaring — 1.00× GPU (driver CSEs
  the duplicate split; confirmed isolated AND full-shader), 0.95–0.98× SwiftShader. Found + fixed a bench
  methodology trap: the companion LLM shares the GPU and its inference bursts faked a 1.4× win — all
  marginal A/Bs must interleave many rounds and take MIN-of-min (bench-sqr/bench-prefetch do). Extended
  probe-barrier with a squaring correctness ladder (sqr6 INTACT on NVIDIA, controls collapse; chained test
  must use an ATTRACTING orbit or legitimate df64 error reads as collapse). All gates green on the shipping
  config: validate:gpu + validate:gpu:real (52 PASS), probe:xbackend (historical envelope), crosscheck
  skip/tiled IDENTICAL, 41 unit. New: tools/bench-sqr.mjs, tools/bench-prefetch.mjs (npm bench:sqr /
  bench:prefetch), harness crossSqr/crossArgs. Ported the prefetch to df64/fe same-spawn and MEASURED it
  there too: 0.997× GPU / 0.957× SwiftShader (the 2× is rescaled-specific) → opt-in. VERDICT: per-iteration
  ALU frontier MEASURED-CLOSED on both backends; next is M6-FUTURE orchestration items, or lock in.
- Spawn 26 (all engine frontiers closed Spawns 9/20–25; AGENDA said "lock in or polish UX" — companion + I
  chose UX polish, unsupervised): shipped a "finished-product" UX pass with ZERO render/math change. Three
  additive features + a fullscreen/help affordance: (1) a first-run gesture-hint overlay (once-per-device via
  localStorage; pointer-events:none + capture-phase non-consuming dismissal so it never blocks the existing
  e2e canvas/panel clicks and the first tap both dismisses and zooms); (2) full keyboard navigation (arrows
  pan / +−=zoom / f=fullscreen / ?=help) via a new `viewer.panByPreview()` primitive that reuses the gesture-
  preview/settle machinery (smooth, one sharp render on settle, discrete via ignoring auto-repeat, inert in
  form fields); (3) a live center-coordinate HUD readout in complex `a+bi` form (`shortCoord()` trims trailing
  zeros + ellipsizes only dropped significant digits; the panel keeps full precision); (4) two top-right corner
  buttons (help "?" + fullscreen "⛶") for mobile parity.
  - Companion LLM consulted at the priority-decision point: ranked the four candidate gaps (onboarding > coords
    > keyboard > fullscreen) and confirmed several considerations were already handled (safe-area insets,
    loading %, the precision-wall warning, UI collapsibility) — so I skipped a hide-all-UI toggle as low-value.
  - VALIDATION (all green, no pixel change): 41 unit unchanged; `viewer.spec.mjs` 26/26 on desktop-chrome (21
    pre-existing + 5 new), `gpu.spec.mjs` 10/10 + `perf.spec.mjs` 2/2 on desktop-chrome, and the 5 new tests on
    mobile-chrome. The pre-existing real-mouse click-to-zoom test STILL passes under the hint overlay (proves
    the pointer-events:none non-consuming dismiss). 5 new e2e cover arrow-pan, +/- zoom, no-hijack-typing, the
    hint lifecycle (show/dismiss-and-zoom/persist/re-open), and the coordinate readout.
  - Files: edits to index.html (#coords + .corner buttons + #hint; HUD → column), styles.css (.hud/.coords-hud/
    .corner/.hint*), src/main.js (shortCoord + complex coords; keydown nav; toggleFullscreen; hint logic),
    src/viewer.js (panByPreview), test/e2e/viewer.spec.mjs (+5), README. New tools/shoot-ux.mjs (UX screenshot
    helper, not wired to npm). FULL writeup NOTES "✅ UX POLISH". NET: discoverability + accessibility + a sense
    of place — the engine was already a vetted specification; this makes it a finished, usable product.
- Spawn 25 (GPU throughput frontier closed Spawns 22–24; companion + I chose the most concrete remaining
  item — push the deep floor to the precision wall & make it a specification — unsupervised): pushed the
  GPU floatexp/rescaled deep floor 2^-600 → **2^-1010** (the true double-dc wall) and added a graceful
  clamp + "max depth" signal below it. Companion LLM consulted at the decision point — endorsed strongly,
  reframed it as turning a heuristic limit into a SPECIFICATION + "closure" and flagged graceful
  degradation (step 4) as the product-value piece.
  - VALIDATION (all green): new `tools/probe-wall.mjs` (double perturbation vs the BigInt-EXACT oracle at
    the pixel's TRUE BigInt offset — extends the M4 B/C/C2 method into the deep band) = **0 escape-count
    mism FLAT 2^-700→2^-1010** on a genuine boundary coordinate, dcRelErr ~3e-16 (pure double rounding);
    dc/step go subnormal only at ~2^-1020 (past the floor). `probe-deep500` (real RTX 3090, new deep coord):
    GPU fe/rs vs CPU oracle **0% flat 600→1010** (odd 1–2 ill-conditioned px, maxΔsn<1.0, no depth-monotonic
    growth). `validate:gpu` + `validate:gpu:real` exterior checks extended 600 → 800,1000 (fe+rs) — ALL PASS
    on SwiftShader AND the RTX 3090. 41 unit pass (dispatch test updated). New precision-wall e2e PASS (both
    projects); 10-test regression subset on the touched zoom/preview/click paths PASS.
  - GRACEFUL DEGRADATION: `viewer.MIN_RADIUS = 2^-1010` (== render.GPU_PERTURB_FE_FLOOR); `_clampRadius()` in
    the radius-commit paths; zoom-IN suppressed at the wall (no over-scale snap-back), zoom-OUT/pan still
    work; main.js shows "⚠ maximum zoom depth reached" + "AT PRECISION WALL"/"(max)" readouts.
  - The KEY tooling fix: gen-deep-coord's ITERCAP now AUTO-scales with depth (≈400+TARGET·250+margin) — at
    the old fixed 140k cap the descent fell INTO the set past ~2^-560 (all probe pixels read interior →
    drifted off the filament).
  - Files: NEW tools/probe-wall.mjs (+ npm probe:wall, probe:deep500). Edits: src/math/render.js (floor +
    doc), src/viewer.js (MIN_RADIUS/_clampRadius/zoom guard/getState), src/main.js (maxdepth signal +
    readouts), tools/{gen-deep-coord (ITERCAP),probe-deep500 (new coord),validate-gpu (800/1000)},
    test/unit/gpu.test.mjs, test/e2e/viewer.spec.mjs, README.md, package.json. FULL writeup NOTES "DEEP
    PRECISION WALL". NET: a CAPABILITY + correctness specification (~10^304× zoom vetted vs ground truth),
    not throughput; deep-band raw-speed frontier stays closed.
- Spawn 24 (Danielle's pivot away from raw throughput — her #1 option, OffscreenCanvas: GPU raster off
  the main thread for mobile smoothness — unsupervised; finalized this spawn): BUILT + SHIPPED the
  OffscreenCanvas GPU worker as the DEFAULT raster path. The GPU escape+color strip loop now runs in a
  dedicated worker (gpu-worker.js) that owns an OffscreenCanvas and pushes finished ImageBitmap strips
  back; the main thread only draws ready bitmaps (no per-strip GL flush/sync). Companion LLM consulted
  twice (strongly endorsed the pivot as a "fundamental shift in the execution model"; contributed the
  heartbeat/Long-Task measurement method + the back-pressure/context-loss pitfall list; on finalize it
  flagged ImageBitmap disposal [already handled — onStrip closes each bitmap] + that SwiftShader makes
  the jank delta MORE visible [correct]).
  - PIXEL-IDENTICAL to the legacy on-main-thread path (crosscheck-offscreen: 0.000% diff, maxΔ 0 across
    naive/df64/rescaled-fe on SwiftShader [re-confirmed this spawn] AND the real RTX 3090 [Spawn-24 prior
    run]) — same shaders, lossless ImageBitmap round-trip.
  - THE WIN (bench-offscreen-jank, deep 2^-271, 400×400, ss=1, SwiftShader): main-thread Long-Task
    blocking **99842ms → 0ms** (the worker path produces ZERO main-thread Long Tasks vs ~the entire render
    blocked on the main path); rAF heartbeat **55 → 5061 ticks** (UI animation survives the whole deep
    render); worst heartbeat gap 6368ms → 2892ms (and the offscreen residual is NOT a Long Task). Render
    TIME ~unchanged (87s vs 100s — throughput is warp-bound, exactly as expected; the win is RESPONSIVENESS,
    not ms/frame).
  - Mobile context-loss recovery preserved through the worker (probe-recover-offscreen ALL PASS on
    SwiftShader): home gpu-naive via worker → loss forwarded worker→main → CPU (naive) while lost → restore
    → GPU worker resumes → recovered image non-blank. Palette/overlay recolor, glitch-count readback,
    gesture-preview snapshot all work offscreen. Robust fallback: sync capability probe declines where
    OffscreenCanvas/WebGL2 absent → on-main-thread GpuRenderer unchanged → CPU pool.
  - VALIDATION (all green): 41 unit; crosscheck-offscreen 0.000%; probe-recover-offscreen ALL PASS; the
    offscreen default-path e2e ("OffscreenCanvas worker path is the default and renders all engines")
    PASSES (desktop-chrome, 20s). Offscreen is the default → the whole e2e suite exercises it. Test edits:
    fixed a latent determinism wait-race (capture doneCount before the action), pinned 3 main-thread-GPU-
    internals tests to ?offscreen=0 (they validate the still-shipping fallback renderer). New files + npm
    scripts: src/gpu/{gpu-worker,gpu-worker-client}.js; tools/{crosscheck-offscreen,bench-offscreen-jank,
    probe-recover-offscreen}.mjs; npm crosscheck:offscreen[:real], bench:offscreen[:real],
    probe:recover:offscreen[:real]. Edits: src/viewer.js, src/main.js, test/e2e/{viewer,gpu}.spec.mjs,
    package.json. FULL writeup NOTES "✅✅ OFFSCREENCANVAS GPU WORKER".
  - NET: delivered the M6-FUTURE OffscreenCanvas item — an execution-model shift (responsiveness), not a
    speedup. The deep-band throughput frontier stays closed (Spawn 23). Next structural levers: fewer per-
    iter texture fetches, push the floor below 2^-600.
- Spawn 23 (Danielle asked for a verdict on interior detection — greenlight a rigorous attracting-cycle detector
  or lock in? — unsupervised): ran a measure-first UPPER-BOUND experiment that REFUTED interior detection as a
  GPU lever, and recommend LOCKING IN the current state. Companion LLM consulted twice (validated the experiment
  design — confirmed the oracle prune is a true ceiling + faithfully models warp divergence — and concurred with
  the kill, contributing the "Oracle-Gap" framing).
  - THE EXPERIMENT (`tools/bench-interior-oracle.mjs`, the cheap thing that bounds the multi-spawn build):
    build an EXACT per-pixel interior mask with the CPU oracle (escapePerturb, same SA + geometry as the GPU),
    feed it to perturbFragRescaled, and let oracle-interior pixels BREAK the loop early at n = skip+frac·(maxIter
    −skip). frac=0 (bail at skip) = a PERFECT, FREE, zero-latency detector = the absolute CEILING; it cuts ONLY
    interior pixels and leaves escapers intact, so GPU warp divergence decides the real saving. Any real detector
    is strictly worse (latency + per-iter tax + fires later). Caveat addressed: every variant pays the identical
    mask fetch (a separate `nomask` run shows the fetch is noise), so the bound is clean.
  - THE RESULT (real RTX 3090): ceiling = **1.55× / 1.29× / 1.39× / 0.99× @ 2^-120/-218/-271/-400** (interior%
    59/57/92/65). The smoking gun: 2^-271 is 92% interior yet caps at 1.39×; 2^-400 gains nothing. Curves SATURATE
    by frac≈0.1–0.25. MECHANISM: probe-interior showed the late-escaping pixels run ~81% of the post-SA budget and
    are spatially sprinkled, so they anchor almost every warp to ~0.8·tail — pruning the interior bulk is invisible.
    SAME structural ceiling that killed BLA (Spawn 18) + the lean kernel (Spawn 22): the deep frame is WARP-
    DIVERGENCE/escaper-tail bound, NOT bulk-work bound. Oracle-Gap argument: a perfect oracle yields ≤1.4× in the
    HIGH-interior case, so a real detector (latency past the saturation window + per-iter derivative tax on ALL
    pixels incl. the escapers) trends to ≤1.0× like BLA. Not worth the complexity/precision risk. DEAD on GPU.
  - The ONLY non-dead branch: the CPU FALLBACK (no warp divergence → interior prune is pure saving) — but it's the
    slow rarely-hit path (GPU is the validated default + correct on real HW), SA already skips 89–92%, and a rigorous
    CPU detector still has the per-iter tax + coordinate-dependence. Low value, recorded, not greenlit.
  - VALIDATION (all green): prune-off is bit-identical (uPrune=0 short-circuits the mask fetch); 41 unit, validate:gpu
    (SwiftShader) + validate:gpu:real (RTX 3090) ALL PASS — CRITICALLY the added shader IR did NOT shift the df64
    collapse threshold on the real GPU (the Spawn-20 risk). Crosscheck (baseline vs max-prune full sn): escDiff=0
    everywhere, 0–3 inside-flips (CPU/GPU boundary) — mask is pixel-aligned, escapers untouched. Bench infra KEPT
    opt-in (uPrune/uPruneIter/uPruneMask in glsl.js; uploadPruneMask + unit-2 binding in renderer.js; benchPrune in
    harness.html — all off by default, NOT wired to production), same keep-validated-but-off pattern as BLA/df64-
    escape/lean. New: tools/bench-interior-oracle.mjs. FULL writeup NOTES "⛔ INTERIOR DETECTION — MEASURED DEAD".
  - NET: the GPU deep-band optimization frontier is now CLOSED — every measured lever (SA✓, order-5✓, margin✓,
    barrier✓ shipped; BLA✗, lean✗, region-SA✗, interior✗ refuted) converges on the same divergence/escaper-tail
    wall. "900% faster" is MET at the deep band (≥2^-218 ~10–13×) and ~64% (6.4×) at the structurally-bound 2^-120
    band. Remaining ideas are STRUCTURAL/UX (OffscreenCanvas off-main-thread, fewer fetches, floor below 2^-600),
    not raw deep throughput. Recommendation to Danielle: lock in.
- Spawn 22 (continue "optimize until 900% faster" — close the moderate-deep gap, unsupervised): shipped the
  THIRD net-win GPU lever — **SA SAFETY-MARGIN MINIMIZATION** (depth-adaptive 0.05→0.02 below 2^-112). Pure
  measure-first; companion LLM consulted once (bet on register-pressure/occupancy — I BUILT + measured it and
  it did NOT pan out on the RTX 3090; its interior-detection skepticism was right on the implementation risk).
  - MEASURED 3 levers, kept the 1 that won. (a) **region-adaptive SA skip** (probe-sabox): a quarter-scale
    centered dc-box buys only +1.4% skip @2^-120 / +0.2% @2^-400 — the order-5 truncation cliff is too sharp.
    DEAD. (b) **lean occupancy kernel** (perturbFragRescaled drops the never-taken BLA + df64-escape blocks,
    407→278 lines, bit-identical; bench-lean): NEUTRAL-to-slightly-SLOWER on the RTX 3090 (~0.93–1.00×, repro
    ~0.8× @2^-271) — a strict subset slower ⇒ occupancy isn't the bottleneck on this GPU. Kept OPT-IN (`p.lean`)
    for future mobile. (c) **SA margin** — THE WIN.
  - THE WIN (probe-samargin + bench-samargin, real RTX 3090): the SA skip = floor((1-margin)·validN); the margin
    (0.05) only guards the GPU df64 seed, which is ~1000× under the gate DEEP. Sweep showed margin 0.05→0.02→0→
    bit-exact (mism 0) with IDENTICAL meanΔsn/maxΔsn — the margin bought ZERO precision deep while discarding
    ~⅓ of the ~4190-iter deep post-SA budget. Cut to **0.02 depth-adaptive** (below 2^-112; 0.05 shallow):
    **1.118×/1.187×/1.179×/1.402× @2^-120/-218/-271/-400** wall-clock, GROWING with depth (margin = bigger
    share of the smaller deep budget). Robust to throughput-vs-divergence (a UNIFORM per-pixel iteration cut).
    Stacked on SA: deep band now ~10×+ (2^-400 ~13×, 2^-271 ~9.2×), 2^-120 ~5.7×→~6.4×. CPU win too
    (crosscheck-sa 2^-500 3.24×→3.50×). 0.01 is the further measured-safe option (opts.marginFrac overrides).
  - INTERIOR DETECTION measured (probe-interior): 61% interior @2^-120 / 90% @2^-271, 65–91% of post-SA work —
    the one remaining BIG lever, but DEFERRED (rigorous attracting-cycle test = GPU state/registers + per-iter
    tax; gated on the GPU being throughput-bound; COORDINATE-dependent — ~0 interior on pure filaments, the
    companion's valid caveat). Recorded with a safety-by-construction design for a future spawn / Danielle.
  - VALIDATION (all green): crosscheck-sa MARGIN=0.02 AND 0.01 (mism 0, maxΔn 0, 2^-50…-500); 41 unit (+1
    depth-adaptive-margin: deep out-skips 0.05, shallow keeps 0.05, bit-exact); validate:gpu (SwiftShader) +
    validate:gpu:real (RTX 3090) ALL PASS (deep skips ROSE, mism 0.000%). Production auto-flows it (worker.js +
    render.js computeSeries take the depth-adaptive default — CPU pool AND GPU). Order-5 SA + lean df64 barrier
    untouched. New: tools/{probe-sabox,probe-interior,probe-samargin,bench-samargin,bench-lean}.mjs; npm
    probe:sabox, probe:interior, probe:samargin, bench:samargin, bench:lean. Edits: src/math/series.js,
    src/gpu/{glsl,renderer}.js (lean variant), test/gpu/harness.html (lean+marginFrac threading),
    test/unit/series.test.mjs, package.json. FULL writeup NOTES "SA SAFETY-MARGIN MINIMIZATION".
- Spawn 21 (continue "optimize until 900% faster" — close the documented moderate-deep gap, unsupervised):
  raised the SERIES-APPROXIMATION polynomial from ORDER 3 to ORDER 5 — the FIRST net speed win since SA
  itself (barrier-min was a correctness fix). Followed measure-first discipline; companion LLM consulted once
  (endorsed order-5 as the bang-for-buck knee, flagged the wall-clock-vs-skip% worry + df64 register/precision
  ceiling at higher orders — all confirmed by measurement).
  - MEASURE FIRST (`tools/probe-saorder.mjs`): the SA skip is bounded by the FIRST of (a) truncation
    |SA−dz|>tol·|dz| or (b) escape guard |dz|>0.25. Pinned that **truncation ALWAYS binds** at production
    tol 1e-10 → more terms = larger skip, biggest at the moderate-deep band (skip% +4.7 @2^-120 … +0.5 @2^-400).
  - THE WIN (`bench:saorder:real`, real RTX 3090, order 3 vs 5 forced, fixed work): **1.24× / 1.21× / 1.13× /
    0.99× at 2^-120 / -218 / -271 / -400** — the inverse of the speed-gap shape (helps where the gap is, free
    at extreme depth). 2^-120 → ~5.7× vs no-SA (was ~4.6×). The post-rebase ~8ms-floor worry was WRONG: the
    leading-run tail is a big enough slice at the moderate band that shaving ~1400 iters/pixel pays. Bit-exact
    ⇒ strictly neutral-or-positive (unlike BLA's net loss).
  - THE TRAP (fixed): order-5's larger skip → later seed → larger df64 (~46-bit) absolute seed error →
    amplifies on the SHALLOW chaotic seahorse (validate:gpu FAIL, meanΔsn 0.9→1.4; mism fraction stayed under
    gate ⇒ precision, not over-skip). df64-CLEAN on the deep boundary coord (`probe:saorder:gpu`: 0% mism,
    meanΔsn ~5e-4 @2^-100..-130), and the shallow band renders df64-no-SA in production anyway → DEPTH-ADAPTIVE
    order (order 5 below 2^-112 = GPU_PERTURB_FLOOR, order 3 above; `SA_ORDER5_RADIUS`). opts.order overrides.
  - VALIDATION (all green): 40 unit (+1 order-5 skips-more-and-bit-exact), crosscheck-sa 0 mism (86.4%/90.3%
    skip @2^-120/-271), validate:gpu (SwiftShader) + validate:gpu:real (RTX 3090) ALL PASS. LEAN df64 barrier
    untouched (probe:barrier unaffected). Order 7+ rejected (probe-saorder: <2% more skip + df64/register risk).
  - Rides the existing default-on "Series approximation" toggle (one switch, CPU+GPU). New:
    tools/{probe-saorder,probe-saorder-gpu,bench-saorder}.mjs; npm probe:saorder, probe:saorder:gpu,
    bench:saorder[:real]. Edits: src/math/{series,perturb}.js, src/gpu/{glsl,renderer}.js, test/gpu/harness.html,
    test/unit/series.test.mjs, package.json. FULL writeup NOTES "HIGHER-ORDER SERIES APPROXIMATION".
- Spawn 20 (continue "optimize until 900% faster" — the documented Spawn-9 follow-up: lighten the df64 barrier,
  unsupervised): turned out to be NOT just perf but a **CORRECTNESS FIX** — the shipping deep-GPU path was broken
  on the real RTX 3090 RIGHT NOW. Followed measure-first discipline; companion LLM consulted twice (endorsed the
  lever + the mechanism + the keep-as-default-with-empirical-caveat call). The first NET-WIN GPU lever since SA.
  - THE LEVER: the df64 `ds_add`/`ds_mul` wrapped EVERY intermediate in the `ob()` XOR optimization barrier
    (8/add, 24/mul = 32 per mul+add) — Spawn 19 pinned these barriered ds_mul squarings as the dominant per-iter
    GPU cost. NOTES (Spawn-9 §) flagged "re-check whether a lighter placement still holds deep" as untried.
  - BUILT `tools/probe-barrier.mjs`: an empirical placement ladder (full/minC/lean8/lean10/lean14/lean8_nop muls,
    full/lean5/lean6/none adds) measured against the real GPU (relerr<1e-10 INTACT gate) + a TIME=1 per-op timing
    harness (fixed-iteration loop, no early exit → fair full-vs-lean clock). Found the MINIMAL placement blocking
    BOTH compiler transforms (reassociation-cancellation + FMA-contraction): ds_mul **8** barriers, ds_add **5**
    (14 vs 32, −56%) — INTACT at ~1e-14, the controls (minC/lean5/none/lean8_nop) collapse/degrade as predicted.
  - THE SURPRISE (reproducible): ISOLATED single-op, BOTH full and lean INTACT. But in the FULL perturbation shader
    on the real GPU, the **FULL barrier validate:gpu:real = 24 FAIL** (depth-monotonic df64→f32 collapse 3.9% @2^-3
    → 99% @2^-70; probe-xbackend 21%/90%); the **LEAN barrier = ALL PASS** (0.0–1.1% ill-conditioned floor;
    probe-xbackend 0.19%/0.68%). MORE barriers → collapse, FEWER → correct. Mechanism (companion-vetted): the
    32-barrier ops inlined thousands of times blow the IR past an NVIDIA compiler complexity threshold → it
    phase-switches to a coarse optimizer that STRIPS the barriers en masse. Vindicates Spawn 8's original
    large-shader diagnosis (Spawn 9's highp-sampler fix was a separate, also-real bug).
  - SPEED (fair, collapse-independent, real GPU): lean **1.64×** per op (714938 vs 436907 Mop/s). bench-gpu:real
    (lean absolute): fe ~9000, rs ~19600 Mit/s. SwiftShader timing was unreliable (companion-contended host,
    loadavg ~5) and moot (full can't run correctly on the real GPU). Stacks multiplicatively with SA.
  - END-TO-END: 2^-50 viewer render = clean seahorse fractal, glitches=0 (shoot-df64.mjs, df64_2e50_gpu1.png).
  - DECISION: LEAN is the new DEFAULT in src/gpu/glsl.js (not a toggle — full is broken so no useful fallback;
    SwiftShader stays the universal-correct fallback). VALIDATION (all green): probe-barrier (isolated, real GPU),
    validate:gpu:real ALL PASS (was 24 FAIL with full), validate:gpu (SwiftShader) ALL PASS, 39 unit, probe-xbackend.
    Portability is EMPIRICAL-per-GPU (companion caveat) — probe:barrier is the new regression gate for any new
    target GPU or DF64_LIB edit. New: tools/probe-barrier.mjs; npm probe:barrier, bench:barrier; edits src/gpu/glsl.js,
    package.json. FULL writeup NOTES "df64 BARRIER MINIMIZATION". This makes the deep GPU path CORRECT on real HW —
    so SA/rescaled are now genuinely deliverable; per-iter ALU/representation floor is now minimal.
- Spawn 19 (continue "optimize until 900% faster" — the documented NEXT-#2 GPU lever, unsupervised): built +
  validated the **df64-escape fast-path** for the rescaled deep engine, then MEASURED it PERFORMANCE-NEUTRAL and
  reached an honest, companion-confirmed keep-off result. Followed the project's measure-first discipline; this
  CONFIRMS the Spawn-18 BLA finding (the GPU's ~8ms SA floor is not per-iteration-ALU-bound).
  - THE LEVER: the rescaled engine's per-iteration escape/rebase/glitch test ran in floatexp every iteration
    ("~half the per-iter cost on the chaotic case"). KEY INSIGHT: `fe` and `df64` have the SAME ~46-bit mantissa —
    fe's only edge is exponent RANGE — and the escape block only RUNS once `|dz|` has grown to O(1) (fast-skip drops
    the tiny-dz steps), which holds at ALL zoom depths. So in the common case the test can run in plain df64 at
    bit-for-bit the same precision (NOT a reduction), falling back to fe only when `|dz|` or `|Z_m| < ~2^-100` (near a
    reference minimum). `-100` is exact: `ds_scale2` returns 0 below it; subnormal `Z_m` reads `ilogb1=-126` → fe.
  - BUILT (glsl.js perturbFragRescaled, `uDf64Esc`): a gated branch mirroring the validated df64 engine's escape
    block; `uDf64Esc==0` (renderer DEFAULT) skips it → the floatexp path is bit-for-bit unchanged. renderer `p.df64Esc`
    opt-in; `df64Esc` threaded through comparePerturb/benchPerturb. New tools/bench-df64esc.mjs (npm bench:df64esc[:real]).
  - VALIDATED CORRECT (both backends, identical numbers): validate:gpu + validate:gpu:real ALL PASS with df64Esc=1;
    the rescaled-with-df64-escape numbers MATCH the historical fe-escape numbers to the digit (rs seahorse 2^-90 =
    0.977%; rs+SA deep 2^-130/-271 = 0.000%) — same-mantissa confirmed, NO glitch line at the fallback boundary. A new
    `== perturb rescaled + df64-escape ==` validate-gpu section gates the (off-by-default) toggle.
  - THE NEUTRAL RESULT (bench-df64esc.mjs, df64Esc off vs on, SA on): real RTX 3090 512² (compute-bound) 0.99/1.00/1.01×
    at 2^-120/-271/-400; SwiftShader 1.00× all depths. NEUTRAL everywhere — and crucially neutral, NOT a net-negative
    like BLA. DIAGNOSIS: the escape block's cost is the barriered ds_mul squarings (|z|²,|dz|² — shared by BOTH paths)
    + the getZ texture fetch, NOT the fe-normalize WRAPPER df64 removes. Companion LLM endorsed the gate + keep-off.
  - DECISION: `uDf64Esc` OFF by default (opt-in); the long-validated floatexp path stays production (no measured
    benefit ⇒ no default change). Infra kept for future weak/mobile-GPU measurement. Rules IN for a real future win:
    cut the ds_mul COUNT or the per-iter fetch / divergence, or off-GPU work (OffscreenCanvas) — not ALU/representation
    micro-opts. Edits: src/gpu/{glsl,renderer,validate}.js, test/gpu/harness.html, tools/validate-gpu.mjs; new
    tools/bench-df64esc.mjs. 39 unit + validate:gpu (SwiftShader + real) ALL PASS. NOTES "df64 ESCAPE".
- Spawn 18 (continue "optimize until 900% faster" — GPU-port the Spawn-17 CPU BLA, the documented
  next lever — unsupervised): BUILT + VALIDATED the GPU BLA port CORRECT on the real RTX 3090, then
  MEASURED that it does NOT accelerate the GPU and reached an honest, companion-confirmed negative
  result. Kept OFF (the no-BLA path is bit-identical). Followed the project's measure-first discipline.
  - MEASURED THE TABLE FORMAT FIRST (tools/probe-bla-mag.mjs): the NOTES plan (A,B as plain df64 + r²
    as a float) WOULD NOT WORK — at 2^-400 the APPLIED jumps use |A|~2^228, |B|~2^231, r²~2^-576, far
    outside float32's ±127. So the table is FLOATEXP: complex A,B as a df64 mantissa pair under one
    shared int exponent + r² as a single-float mantissa+exponent, 3 RGBA32F texels/entry (~77 MB deep).
    bla.js `blaToFloat32`; round-trip unit-tested incl. the out-of-float32-range deep case.
  - SHADER (glsl.js perturbFragRescaled): loop-top scan for the largest valid level (radius r²≥|dz|² +
    bounds), apply dz'=A·dz+B·dc in floatexp, collapse to shared-exponent form. uBlaMaxLevel==0 disables
    it → no-BLA path BIT-IDENTICAL (validate:gpu unchanged, on SwiftShader AND real GPU). renderer
    uploadBLA (highp NEAREST). NEAREST+highp REQUIRED (mediump would corrupt the df64 coeffs — Spawn 9 class).
  - VALIDATED CORRECT — two gates (BLA is an approximation: loose shallow, exact ≥2^-120; the CPU BLA itself
    drifts ~7% at 2^-90 on the seahorse coord, so a naive vs-no-BLA gate mis-flags it). FAITHFULNESS GPU-BLA
    vs CPU-BLA (same table): 2^-70 0.315%, 2^-90 0.564% — proves the port. CORRECTNESS GPU-BLA+SA vs no-BLA
    oracle deep: 2^-130 0.171%, 2^-271 0.049%. crosscheck-gpu-bla on-vs-off: 0.15/0.08/0.22% at 2^-120/-271/
    -400, maxΔn≤23. ALL PASS on SwiftShader AND the real RTX 3090, identical numbers.
  - THE NEGATIVE RESULT (tools/bench-bla.mjs, real RTX 3090): SA = 4.2×/6.6×/8.8× over no-opt, but SA+BLA =
    2.6×/3.4×/4.8× — adding BLA HALVES the speedup. BLA's iteration-work reduction (2.5–3.3× on CPU) is real
    on the GPU too, but each BLA step costs far more than a rescaled iteration: texture-fetch LATENCY vs ALU
    (SA is pure coherent Horner/FMA — what GPUs crush), and WARP DIVERGENCE in the chaotic post-SA tail where
    BLA would help. THREE scan optimizations (level-1 early-out via r²-monotone-in-level → 1-fetch reject;
    binary-search the level → ~4 branch-coherent probes vs ~16 divergent; carry |dz|² to skip recomputation)
    each helped but never flipped the sign. Companion LLM independently confirmed: SA is the correct +
    sufficient GPU lever; BLA is CPU-centric (low-latency cache + branch prediction the GPU deprioritizes).
  - DECISION: BLA OFF on GPU (always was to stay off until real-HW-validated; now also because it's a net
    loss). Infra preserved (validated shader/table/tools) for future HW or a coherent-warp-only scheme. BLA
    remains a genuine CPU win (Spawn 17). New: src/math/bla.js (blaToFloat32), tools/{probe-bla-mag,
    crosscheck-gpu-bla,bench-bla}.mjs; edits: src/gpu/{glsl,renderer,validate}.js, test/gpu/harness.html,
    tools/validate-gpu.mjs, test/unit/bla.test.mjs (+2 packing tests). npm crosscheck:gpu:bla[:real],
    bench:bla[:real], probe:bla:mag. 39 unit + validate:gpu (SwiftShader + real) ALL PASS. NOTES "⚠️ BLA GPU PORT".
- Spawn 17 (generate + benchmark new optimizations until 900% faster — user request, unsupervised):
  built + validated **BLA (bivariate linear approximation) on the CPU** — the lever that breaks the
  GPU's ~8ms SA floor and projects PAST 10× across the board. Followed CPU-oracle-first discipline
  (SA was Spawn 15 CPU → 16 GPU; BLA is the same shape). The GPU port is the documented next step;
  the CPU work-reduction directly bounds the GPU win. (Companion LLM consulted twice — strongly
  ranked BLA over a df64-escape micro-opt + higher-order SA, and confirmed the radius/merge formulas
  + the in-run escape/rebase-skip safety. See NOTES "✅ BLA".)
  - WHY: SA skips the LEADING linear run once, but the deep orbit is a SEQUENCE of linear runs — every
    rebase resets dz small and it re-grows linearly again. Those post-rebase re-growths ARE the flat
    ~8ms SA floor (measured: SA-on time is ~8ms regardless of depth). BLA skips runs THROUGHOUT the
    orbit, incl. after every rebase — exactly the floor.
  - MATH (src/math/bla.js): binary merge tree of bivariate-linear maps dz_{m+L}=A·dz_m+B·dc from the
    reference alone. level-0 A=2Z,B=1,r=blaEps·|2Z| (r=0 when |Z|≥2, the escape-safety guard); merge
    A=A_y·A_x, B=A_y·B_x+B_y, r=min(r_x,(r_y−|B_x|·dcMax)/|A_x|); non-finite coeff (|A|~2^L overflow)
    ⇒ r=0 (never applied). Stepping in escapePerturb (`bla` arg): largest valid jump, else one true
    step + escape/rebase. The no-BLA path is LITERALLY UNCHANGED (separate branch) — 34 prior tests pass.
  - MEASURED (tools/crosscheck-bla.mjs, deep boundary coordinate, work = true steps + jumps): the
    marginal reduction BLA buys ON TOP of SA at eps 2^-30 — **2.49× @2^-120, 2.73× @2^-271, 3.25×
    @2^-400**; combined SA+BLA vs the no-SA-no-BLA baseline **14.7× / 23.3× / 39.5×** (even the
    conservative eps 2^-32: 11.8/19.2/33.2× — all past 10×). This is the CEILING the GPU port chases.
  - CORRECTNESS (honest): BLA drops dz² so it is an approximation, not bit-exact like SA. BigInt
    arbitration (tools/arbiter-bla.mjs): the few differing pixels are WELL-conditioned (no-BLA oracle
    == BigInt) and BLA is off by a SMALL bounded amount (±1 @2^-271, ≤19 @2^-400) — truncation
    accumulating, NOT a missed escape (those flip by hundreds; the ZMAX guard + radius forbid it, unit
    test caps maxΔ≤50). FAR below the GPU df64 envelope (0.3–1%), so the GPU port can run a looser eps.
  - VALIDATION (all green): 37 unit (+3 bla.test.mjs: tracks-oracle+cuts-work, composes-with-SA,
    table-well-formed); crosscheck-bla + arbiter-bla across 2^-120…2^-400; full prior suite unchanged.
    New: src/math/bla.js, tools/{crosscheck-bla,arbiter-bla}.mjs, test/unit/bla.test.mjs; render.js +
    perturb.js gained an optional `bla`/`stats` arg (no-BLA path bit-identical). npm crosscheck:bla,
    arbiter:bla. NEXT: the GPU port (table-as-texture + in-shader level scan; plan in NOTES "✅ BLA").
- Spawn 16 (optimize the shader further + no black screen — user request, unsupervised): both
  shipped + validated on the REAL RTX 3090 (Vulkan) AND SwiftShader. (Companion LLM consulted
  on the cap-omission architecture decision; endorsed it given the cap is non-binding + the
  empirical oracle gate is the real check.)
  - NO BLACK SCREEN: the strip-tiled GPU render now blits ONLY each strip's own display rows
    (viewer._blitGpuStrip) instead of the whole canvas, so the not-yet-computed rows keep showing
    the scaled preview / prior frame — high-res tiles REPLACE the low-res preview as they arrive
    instead of the canvas flashing to interior-black first. Final image byte-identical (golden +
    deep-glitch-free e2e unchanged); 2 new deterministic e2e (strip blit touches only its rows;
    multi-strip render preserves the preview below the first tile). See NOTES "✅ GPU SERIES…".
  - GPU SERIES APPROXIMATION: the deep rescaled shader seeds dz at iteration `skip` via an
    in-shader floatexp Horner (FE_LIB ops — not delicate). Worker computes computeSeries with NO
    coarse cap (tools/probe-sa-cap.mjs proved it non-binding: the skip is identical with/without
    it, and the residual mismatches are cap-independent ill-conditioned pixels). **4.1× / 6.7× /
    9.3× faster at 2^-120 / 2^-271 / 2^-400 on the real GPU** (bench-sa.mjs), skip 82→92%.
    Validate-gpu SA section (GPU-with-SA vs no-SA oracle): 0.000% mism deep, same envelope as
    no-SA shallow — PASS on SwiftShader AND real GPU, identical numbers. Live viewer end-to-end
    (probe-sa-viewer.mjs): 0.00% picture diff SA on/off, 2.43× wall-clock. Rides the default-on
    "Series approximation" toggle. New: tools/{probe-sa-cap,bench-sa,probe-sa-viewer}.mjs; files:
    src/gpu/{glsl,renderer}.js, src/{viewer,worker}.js, src/gpu/validate.js, test/gpu/harness.html,
    tools/validate-gpu.mjs, test/e2e/gpu.spec.mjs. NEXT: fe-engine seed; coarse GPU jump-preview.
- Spawn 15 (stretch goal — series approximation, unsupervised): built + validated **series
  approximation on the CPU perturbation path**, default on, bit-exact, the deep-zoom cost
  attacked at its root. (Companion LLM consulted twice: endorsed SA over the other two stretch
  goals + the measure-first→CPU→GPU sequencing, flagged Horner / the df64-vs-f64 margin / the
  floatexp shader minefield; then endorsed shipping the CPU feature this session with the GPU
  port as its own next session — "you've reached a save point.")
  - MEASUREMENT FIRST (tools/probe-sa.mjs): the raw skip potential is HUGE at depth — 94% at
    2^-120, 97% at 2^-271..-500 (the linear term A_n·dc dominates dz for almost the whole orbit
    when dc is tiny). That made SA clearly the right pick over pushing the floor / OffscreenCanvas.
  - MATH (src/math/series.js): SCALED coefficients a_n=A_n·R, b_n=B_n·R², c_n=C_n·R³ with u=dc/R
    (raw A_n overflows 2^1024 near a deep boundary point); dz_seed=a·u+b·u²+c·u³ by Horner. Skip
    chosen by a probe GRID (truncation, relTol 1e-10 — load-bearing) + an escape guard (|dz|>0.25)
    + the worker's coarse-pass MIN-ESCAPE cap (catches an isolated early escaper between probes),
    minus a 5% margin. escapePerturb takes an optional `sa` (no-SA path bit-identical).
  - INTEGRATION: worker computes the series ONCE on worker[0] after the no-SA coarse pass (so the
    cap applies) → params.sa → every tile worker. viewer `series` flag (default on) + setSeries +
    "Series approximation" checkbox + "SA skip N (P%)" debug line. Disabled under the glitch
    overlay. Not URL-persisted. The GPU deep path does NOT use SA yet (next session).
  - THE CAREFUL FINDING: at the shallow band a rare pixel differs by hundreds of iters — BigInt
    arbitration proved it's an ILL-CONDITIONED pixel where the no-SA double render is ALSO wrong
    (BigInt 2778, no-SA 2870, SA 2386; all neighbors exact). SA never corrupts a well-conditioned
    pixel (a unit test BigInt-checks every differing pixel). Gate on the bulk fraction, not maxΔ.
  - VALIDATION (all green): 34 unit (+3 series.test.mjs); crosscheck-sa 0 mism 2^-50…-500 (the
    speedup table); new CPU-deep SA e2e (saSkip>0 + structured, toggle-off → saSkip 0); 27 viewer
    + 7 gpu e2e (desktop); **validate:gpu ALL PASS** (shaders/renderer untouched). Files:
    src/math/{series.js(new),perturb.js,render.js}, src/{worker,viewer,main}.js, index.html,
    test/unit/series.test.mjs(new), test/e2e/viewer.spec.mjs, tools/{probe-sa,crosscheck-sa}.mjs(new).
- Spawn 14 (remaining polish — unsupervised): closed the LAST two queued items (NEXT #5 a/b,
  the companion LLM's ranking) — a **CPU-path visual glitch overlay** and a **low-power /
  battery mode**. NO shader/renderer/GPU-math change, so validate:gpu + the 31 unit tests are
  untouched. (Companion LLM consulted: endorsed the ss-tinting approach — it likes the
  box-averaged "glitch heatmap" — and ranked low-power next; flagged only mask memory, already
  a lean Uint8Array.)
  - CPU GLITCH OVERLAY: the workers now post a per-pixel Pauldelbrot Uint8 mask per band (+
    coarse pass) when showGlitches is on (worker.js: makeEscape records state.lastGlitched;
    coarsePass/renderBands build + transfer the mask). palette.js colorizeRegion/colorizeBlocks
    take an optional mask and blend flagged pixels 0.6 toward magenta — the SAME mix as the GPU
    COLOR_FRAG. viewer.js threads showGlitch/glitchTol to the workers, caches the compute-res
    mask (this._glitch), re-tints on palette change, and does a cheap _recolor on overlay-OFF
    (mirrors _recolorGpu). GATED so default renders are byte-identical (no mask, sn untouched —
    proven by the unchanged golden-fingerprint + deep glitch-free e2e). glitchTol threaded so
    the count/mask are tunable like the GPU uniform. New e2e (force CPU + deep + huge tol →
    magenta + count, off → gone). SS note: CPU tints subsamples then box-averages (a soft
    glitch heatmap) vs the GPU's per-output-pixel OR — identical at ss=1, kept deliberately.
  - LOW-POWER MODE: viewer.lowPower (manual "Low power" checkbox) caps DPR (→1) + backing
    (1100→700) in resize(), forces ss→1 in _effectiveSS (wins over Force HQ), and lowers the
    AUTO iteration ceiling to 2000 via the new _autoIter() (all autoMaxIter call sites route
    through it; a user-typed maxIter via setMaxIter is NOT capped). setLowPower re-sizes +
    re-renders. main.js initBatteryDetect auto-enables it on a discharging ≤20% battery
    (navigator.getBattery) until the user touches the toggle (lowPowerManual latch); absent in
    headless/Firefox → progressive enhancement, manual toggle is the real control. Not URL-
    persisted (per-device energy choice, like Force HQ). New e2e asserts the caps (auto-iter
    cap checked purely via _autoIter to avoid a slow deep render; backing/ss toggled on the
    fast home view — radius 1e-7 is GPU-*perturb*, would time out under SwiftShader).
  - VALIDATION (all green): 31 unit; 38 viewer e2e (incl. the 2 new) + 14 gpu e2e, both
    projects; validate:gpu ALL PASS (GPU path untouched). Files: src/worker.js, src/palette.js,
    src/viewer.js, src/main.js, index.html, test/e2e/viewer.spec.mjs. README/NOTES/AGENDA updated.
- Spawn 13 (remaining polish — unsupervised): closed two small self-contained items — the
  **Force High Quality** toggle (the AGENDA's flagged "smallest open UX item") and the last
  unchecked M8 box, **perf-budget e2e assertions**. (Companion LLM consulted on priority +
  the Force-HQ design; it endorsed both, ranked perf-budget first, flagged no concerns.)
  - FORCE HIGH QUALITY: overrides Spawn 10's extreme-depth ss auto-cap (SS_DEEP_CAP_RADIUS
    2^-300 drops effSS→1 for speed) so a user can force full AA for a deliberate slow deep
    capture. `viewer.forceHighQuality` (default false) makes `_effectiveSS()` skip ONLY the
    depth cap — the hard memory/texture caps (MAX_COMPUTE_DIM/PIXELS) STILL apply (no OOM).
    `setForceHighQuality` re-renders (deep → full ss now; shallow no-op, re-render for
    setSupersample parity). Debug line shows "(HQ: depth cap overridden)" when in effect.
    NOT in the URL hash (per-session capture choice — a bookmark mustn't silently hand a
    recipient a ~4×-slower deep render). SS_DEEP_CAP_RADIUS now exported so main.js shares
    the constant. UI: "Force high quality" checkbox by the Supersampling select.
  - PERF-BUDGET e2e (test/e2e/perf.spec.mjs): LOOSE catastrophic-regression guards by design
    — SwiftShader (CPU GL) on a contended shared host (the companion LLM uses the cores) would
    flake a tight budget, so the ceilings sit ~8–24× over observed (home first-pixel ~0.7–1.0s,
    full ~0.7–1.0s, CPU-pool ~0.5–0.6s). Two tests on the fast home path that nothing else
    times: time-to-first-pixel (poll the canvas to first structured paint) + full-frame on the
    GPU default AND the CPU-pool fallback; also asserts first-pixel ≤ full (progressive reveal).
    Deep-path catastrophic regressions are already caught by the explicit deep-test timeouts.
  - SCOPE/SAFETY: only viewer.js (state + `_effectiveSS` + setter + export), main.js (import +
    debug note + checkbox + syncControls), index.html (checkbox), + 2 test files. NO shader/
    renderer/render-path change, so validate:gpu and the 31 unit tests are untouched. 31 unit
    pass; Force-HQ + perf e2e green on both projects (SwiftShader). New file: perf.spec.mjs.
- Spawn 12 (remaining polish — unsupervised, deep engine + stability already done): added the
  **glitch debug overlay** + an honest GPU glitch count (the last open M7 item, NEXT #4).
  - THE GAP: the GPU perturbation shaders have always computed a Pauldelbrot glitch flag into
    the sn texture's `.b` channel, but the viewer hard-coded `glitchTol = 0`, so the flag was
    STRUCTURALLY never set (gated by `if (uGlitchTol > 0.0 …)`) and every GPU render reported
    `glitches: 0` regardless of reality — dishonest, while the CPU path counts a real number.
  - THE FIX (a "Glitch overlay" checkbox, off by default): COLOR_FRAG gains `uShowGlitch` and
    ORs the `.b` flag across the ss×ss block, blending flagged pixels 0.6 toward magenta (no
    extra texel fetch; rendered color byte-identical when off). renderer `colorize()` sets the
    uniform + new `countGlitches()` reads back the FBO's `.b > 0.5` count (compute-res, matching
    the CPU semantics; runs only when the overlay is on). viewer `showGlitches`/`glitchTol`
    (1e-6) + `setShowGlitches` (re-render on, cheap recolor off), passes glitchTol/fastSkip=0
    in the perturb args, and `_finishGpu` reports the real count for gpu-perturb engines.
    main.js + index.html wire the checkbox (not in the URL hash — ephemeral debug switch).
  - PURELY DIAGNOSTIC: the `.b` flag never feeds back into dx/dy/m/n or the rebase decision,
    so enabling glitchTol + disabling fast-skip change the FLAG, not the escape counts /
    rendered image — confirmed by validate:gpu staying ALL PASS and the e2e's overlay-off check.
  - VALIDATED: 31 unit pass; validate:gpu ALL PASS (non-regressive); new e2e "glitch overlay:
    honest GPU glitch count + magenta tint when forced" — deep render → enable overlay (finite
    count read back) → force tol 1e9 (flags ~every escaping pixel: count > 100 + magenta tint,
    proving flag→readback→tint end-to-end) → toggle off (tint gone). PASS on SwiftShader (both
    projects) AND the real GPU (Vulkan/RTX 3090, 1.8s). No new tools; only edits.
- Spawn 11 (remaining UX/stability polish — unsupervised, deep engine already done): added
  **WebGL context-loss recovery** for mobile. The deep-zoom engine was validated + fast, so
  this spawn looked for a real stability gap and found a clear one: ZERO handling of WebGL
  context loss. Mobile GPUs lose the context under memory pressure / backgrounding / driver
  reset; a loss mid-render left the strip-tiled loop drawing on a dead context (silent no-ops →
  frozen/blank) with no recovery — and since a lost context's GL calls don't throw, even the
  existing _gpuFail (thrown-error only) never tripped.
  - FIX: GpuRenderer listens for webglcontextlost/restored — preventDefault (required for the
    browser to restore), drop ALL cached GL handles on loss (lazy-recreated on next render),
    re-fetch extensions + rebuild the VAO on restore; new `lost` getter; dispose() guards itself
    out (it calls loseContext()). Viewer gates GPU engine-pick on `!gpu.lost` (→ CPU pool while
    lost: correct since CPU is the validated oracle, just slower), re-renders on each transition,
    a 600ms grace before the CPU detour (skips it when restore is near-instant — the common case),
    and gives up on the GPU after 3 losses to avoid a memory-starved flip-flop. setUseGpu resets
    that state; gpuInfo() reports off-while-lost.
  - VALIDATED headless on BOTH SwiftShader and the real GPU (GPU=1, Vulkan/RTX 3090) via
    WEBGL_lose_context. KEY GOTCHA: a tick is required between loseContext() and restoreContext()
    (back-to-back never fires `restored`). tools/probe-ctxloss.mjs (event dispatch + the tick
    finding); tools/probe-recover.mjs (drives the REAL viewer: CPU-while-lost, GPU-on-restore at
    BOTH shallow gpu-naive AND deep gpu-perturb — the deep case proves the reference texture
    re-uploads on restore — plus the 3-loss give-up) — ALL PASS on both backends. e2e
    gpu.spec.mjs "recovers from WebGL context loss" (mobile+desktop) is the CI gate.
  - NO REGRESSION: 31 unit, validate:gpu ALL PASS (shaders + draw path untouched — only the
    renderer lifecycle changed), full e2e green. Also fixed a documented pre-existing e2e flake
    (the 2^-9.8 'Go' navigation timing out under SwiftShader CPU contention) by rendering it at
    ss=1 — 4× cheaper, same pattern the other deep tests use; navigation correctness is
    independent of ss. New: tools/probe-{ctxloss,recover}.mjs, npm probe:ctxloss / probe:recover.
- Spawn 10 (optimize the GPU code when deep zoom 2^500 — user request): DONE. The 2^500
  zoom previously fell BELOW the GPU floor (2^-340) → slow CPU fallback; now it renders on
  the GPU, validated and proven end-to-end.
  - ROOT INSIGHT: the deep fe/rescaled engines are exponent-magnitude-agnostic — the fe
    exponent is a plain int, and the CPU computes dc as normal doubles (exact to ~2^-1000),
    feSplit hands the GPU (mantissa, int-exp). So 2^500 was blocked ONLY by the dispatch
    constant GPU_PERTURB_FE_FLOOR, not by any arithmetic limit. Lowered it 2^-340 → 2^-600.
  - THE VALIDATION (the careful part — the companion's right worry was iteration-count
    accumulation, not zoom depth): built a GENUINE deep boundary coordinate to ~2^-520 by
    descending a filament (tools/gen-deep-coord.mjs: hybrid tracking — chase the hottest
    escaping tip while the box is fully exterior at coarse scale, then hug the boundary from
    the INSIDE (interior pixel nearest the tip) once it straddles, which stopped the center
    drifting off the filament deep). tools/probe-deep500.mjs then compared the rescaled + fe
    GPU engines vs the CPU perturbation oracle on that coordinate at autoMaxIter: **0.000%
    escape-count mismatch FLAT 2^-120 (30k iters) → 2^-520 (130k iters)**, maxΔsn bounded
    ~4e-3. FLAT ⇒ rebasing bounds the per-iter df64 accumulation; it does NOT grow with
    maxIter. (Why no reference drift: reference.js computes the orbit in BigInt and SAMPLES
    each Z_n to df64 once — it never iterates in df64, so each Z_n is independently 46-bit,
    no compounding.) Plus exterior-arithmetic checks added to validate-gpu at 2^-400/-500/-600
    (0.000% on real GPU AND SwiftShader). 31 unit pass (dispatch-band test updated).
  - END-TO-END: tools/shoot-deep500.mjs drove the actual viewer to the deep coordinate —
    zoom 2^500 renders engine gpu-perturb-fe, glitches=0, refLen 125254, ~19s, a clean
    seahorse-spiral fractal (screenshots/deep500.png). Reference build is only ~0.43s (BigInt);
    the ~19s is GPU escape work, so GPU-side perf actually matters here.
  - PERF: auto-drop supersampling below 2^-300 (viewer._effectiveSS + SS_DEEP_CAP_RADIUS):
    ss=2 quadruples the heavy deep shader's pixels → a 2^500 frame would be ~75s; capping
    effSS to 1 keeps it ~19s. User's ss SELECT untouched; debug shows "capped from 2× for
    depth"; the 2^-270 bookmark (above the cap) keeps full ss. Endorsed by the companion LLM.
  - NEW TOOLS: gen-deep-coord.mjs, probe-deep500.mjs, shoot-deep500.mjs. Files touched:
    render.js (floor), viewer.js + main.js (ss cap + debug), validate-gpu.mjs, gpu.test.mjs.
- Spawn 9 (properly fix the GPU bug — user request): FIXED, with the true root cause —
  and it was NOT what Spawn 8 thought. Spawn 8 saw the deep df64/fe/rescaled engines ~90%
  wrong on real NVIDIA, fixed a real Veltkamp-split reassociation with an XOR barrier, but
  the residual persisted and it (mis)concluded "the driver re-optimizes the big inlined
  shader past the per-op barrier." Spawn 9 disproved that and localized the real cause:
  - METHOD: started from "isolated df64 ops are intact but the full shader collapses" and
    bisected. probe-localize (barriering the float32 escape/rebase reductions changes
    NOTHING → not the rebase decision). probe-state (the df64 dz STATE is wrong from
    iteration 5 while SwiftShader tracks the double oracle to 1e-14 → the update itself
    collapses). probe-loop/mag/matrix (rolled-vs-unrolled loop, operand magnitude down to
    2^-54, const-vs-uniform Z, escape-block presence — ALL intact, ruling each out).
    probe-texz (the ONLY trigger: a per-iteration texelFetch'd reference Z). probe-fix
    (laundering Z through the barrier does NOT help; declaring the sampler `highp` DOES).
  - ROOT CAUSE: `uniform sampler2D uRef;` has no precision qualifier → defaults to mediump
    in the fragment shader → NVIDIA returns the texelFetch'd reference rounded to ~fp16
    (~10-bit mantissa), which destroys df64 (lo word = noise). The collapse asymptotes to
    ~2^-10, the mediump floor — the tell. SwiftShader treats mediump as fp32, hiding it.
  - FIX: `highp sampler2D` on uRef in all four perturb shaders + uSn in COLOR_FRAG (one
    word each; no JS/renderer change). The XOR barrier is KEPT (separate, real, proven-in-
    isolation Veltkamp-split fix — probe-df64 collapses without it, no texture involved).
  - VERIFIED on RTX 3090 (Vulkan): probe-xbackend 90.01% → 0.00% (GPU now bit-identical to
    SwiftShader); validate:gpu:real 12 FAIL → 0 (ALL PASS, df64/fe/rescaled 2^-3..2^-340,
    same 0–1.2% chaotic mism as SwiftShader); crosscheck-skip + crosscheck-tiled 0-diff on
    GPU; 31 unit pass; SwiftShader validate:gpu unchanged (no regression); a real viewer
    render at 2^-50 is a clean seahorse-valley fractal (screenshots/df64_2e50_gpu1.png,
    gpu-perturb, glitches=0). New tools: probe-{localize,state,loop,mag,matrix,texz,fix,
    fix2}.mjs (diagnosis trail), shoot-df64.mjs (visual gate).
- Spawn 8 (why GPU not used for tests → profile the shaders — user request): GPU now WORKS
  for the tests, and using it immediately exposed a serious hidden bug.
  - WHY NO GPU: headless Chromium defaults to the SwiftShader CPU rasterizer for WebGL, and
    our flags `--disable-gpu --enable-unsafe-swiftshader` forced it. FIX: select an ANGLE GPU
    backend. New shared tools/chromium-launch.mjs picks SwiftShader (default) / Vulkan (GPU=1)
    / native GLES (GPU=gl); all 13 tools + playwright.config now use it. Verified on RTX 3090
    headless: `ANGLE (NVIDIA, Vulkan 1.4.312 … RTX 3090)`, EXT_color_buffer_float, maxTex 32768.
    ~600× faster than SwiftShader. tools/probe-gpu-real.mjs is the flag-combo sweep.
  - THE BUG (found by running validate on the real GPU): the deep df64/floatexp/rescaled
    engines are NUMERICALLY WRONG on real NVIDIA — 12 FAIL on validate:gpu:real (20–99% wrong
    pixels ≥ ~2^-12), all PASS on SwiftShader. Root cause PROVEN (tools/probe-df64.mjs): the
    NVIDIA compiler reassociates the Veltkamp split `ca-(ca-x)→x`, zeroing it → df64 silently
    collapses to float32. SwiftShader doesn't reassociate, so it hid the bug for 6 spawns.
  - PARTIAL FIX (shipped): an optimization barrier ob(x)=intBitsToFloat(floatBitsToInt(x)^
    uOptBarrier), uOptBarrier a uniform==0 the compiler can't fold (a plain bitcast round-trip
    IS folded away — proven). Wrapped every rounded result in DF64_LIB ds_add/ds_mul; renderer
    sets uOptBarrier=0 per program. Makes the ISOLATED ops intact on GPU (relerr ~1e-14 ==
    SwiftShader; probe-df64-real.mjs) and NO SwiftShader regression (29 PASS / 0 FAIL, 31 unit).
    BUT the full perturbFragDf64 still diverges (probe-xbackend: GPU-df64 vs SwiftShader-df64
    21%@2^-22 … 90%@2^-50): the driver re-optimizes the large inlined shader past the per-op
    barrier. Ruled out: fast-skip (identical), FTZ/subnormals (both backends flush — probe-ftz),
    reference texture, loop cap. Native GLES fails identically to Vulkan ⇒ NVIDIA driver backend.
  - VALIDATED SOUND ON REAL GPU (GPU-vs-GPU, so precision-independent): strip-tiling
    (crosscheck-tiled, 0-diff) and fast-skip (crosscheck-skip, 0-diff). Those mechanisms work.
  - PROFILING (bench:gpu:real): rescaled is ~2.1× faster than fe on the real GPU (vs 1.26× on
    SwiftShader) — the rescaled engine helps MORE on real hardware, as Spawn 6 predicted. (Full
    profiling deferred until the deep path is CORRECT — no point timing wrong output.)
  - NEW TOOLS: chromium-launch.mjs, probe-gpu-real, probe-df64, probe-df64-real, probe-xbackend,
    probe-ftz, probe-collapse; harness compareDf64VsF32/renderIter; npm GPU scripts. NEXT #0 has
    the recommended fix (GPU self-test + CPU fallback — makes the viewer correct on ALL HW now).
- Spawn 7 (zoom past 2^218 — user request): DIAGNOSED then FIXED. The barrier was NOT
  precision. Measured (tools/probe-deep218.mjs) the rescaled + floatexp engines vs the CPU
  53-bit oracle on a real deep coordinate at 2^-90…2^-271 (chaotic, high maxIter): **0.000%
  mismatch** throughout, reference builds ~450ms. The actual wall: a deep frame needs maxIter
  ~55k, and the escape pass was ONE GPU draw over the whole screen → a 10–40s single draw
  trips the GPU watchdog (TDR) on real hardware → context loss → CPU fallback (minutes) →
  "can't zoom past 2^218". (On SwiftShader the same draw just times out >120s, confirmed via
  tools/shoot-deep.mjs: gpu-perturb-fe, glitches:0, never finishing.)
  - FIX: STRIP-TILE the escape pass. renderer `_bindEscapeTarget` keeps the viewport on the
    FULL FBO (gl_FragCoord unchanged) and uses a SCISSOR rect (0,stripY,W,stripH) to restrict
    which rows are written; `clearSn()` pre-clears to interior. viewer `_drawTiledEscape`
    loops strips — draw, flush (own GPU command → per-strip watchdog), colorize+blit
    (progressive top-to-bottom reveal), await rAF (responsive + lets the GPU drain), with a
    `gen` guard each iteration for clean mid-render cancellation. `_renderGpuNaive` +
    `_renderGpuPerturb` are now async through it; `_stripRows()` sizes strips to a ~4e8
    pixel-iter watchdog budget (shallow views = one strip). Applies to naive/df64/fe/rescaled.
  - BIT-IDENTICAL to a single draw — tools/crosscheck-tiled.mjs: 0-diff (iter/sn/glitch)
    across all engines, depths incl. 2^-218, strip heights 1-row…larger-than-frame.
  - VALIDATION (all green): 31 unit; validate-gpu ALL PASS with IDENTICAL baseline mismatch
    numbers (renderer single-draw path unchanged); crosscheck-tiled 0-diff; crosscheck-skip
    still 0-diff (after the _bindEscapeTarget refactor); 42 e2e (incl. deep dispatch + zoom-
    mid-render-cancel); smoke gpu OK (gpu-perturb-fe @2^-150). Screenshot: a structured
    seahorse renders through the tiled path (screenshots/tiled_seahorse_2e50.png, glitches:0).
  - HONEST: tiling removes the watchdog barrier + adds responsiveness/progressive/cancel; it
    does NOT reduce total work, so a deep frame is still ~tens of seconds on a real GPU (and
    unrenderable on SwiftShader). Next speed levers in NEXT #0/#2. New: tools/{crosscheck-
    tiled,probe-deep218,shoot-deep}.mjs + npm scripts. NOTES/AGENDA/README updated.
- Spawn 6 (optimize GPU deep zoom further — user request): the deferred rescaled
  single-exponent engine, BUILT + HARD-VALIDATED + shipped as the deep default, plus a
  bit-identical fast-skip. Net deep speedup ~1.26× over fe (matched-load, worst-case
  chaotic valley) with NO precision regression.
  - RESCALED ENGINE (perturbFragRescaled): dz=(Dx,Dy)·2^S shares ONE int exponent; the
    2·Z·dz+dz²+dc update runs in raw df64 (align linear/dz²/dc to frame W=max(eL,qe,Sc)
    by exact power-of-two scalings, renormalize S once) instead of fe's ~14 per-op
    normalizes. The Zhuoran rebase CATCH (the reason Spawn 5 deferred this) is sidestepped:
    escape/rebase still runs in EXACT floatexp (convert dz→fe per component), so the
    decision logic is byte-identical to the fe engine; only the cheap bulk update is
    rescaled. Wired as the engine the viewer's gpu-perturb-fe band dispatches;
    renderPerturbFloatexp kept as the reference/oracle + one-line fallback.
  - TWO BUGS found + fixed (general traps for this representation, documented in NOTES):
    (1) Z_0=0 (Mandelbrot) makes the linear term vanish after EVERY rebase, leaving dz²
    dominant — the combine frame must include the dz² exponent (qe=2S) or the orbit never
    escapes post-rebase (exterior patches read 100% interior). (2) the linear's TRUE
    exponent (not S) must set the frame, else the un-normalized |2·Z·D|~6 costs the dc/dz²
    addends ~2-3 low bits → 4× worse mism at the viewer's deep maxIter.
  - FAST-SKIP (all three perturb shaders): skip the escape/rebase/glitch block when dz is
    provably too small to escape or rebase (|Z_m|>2|dz| and |Z_m|<64). PROVEN bit-identical
    (crosscheck-skip: skip on vs off → 0-diff full image, df64+fe+rs, 2^-20..-340 incl. the
    chaotic valley + escaping exterior). ~1.00× on the chaotic bench (SIMD divergence runs
    the block for the whole group if any lane needs it); free + helps smooth regions + real GPUs.
  - VALIDATION: 31 unit + 42 e2e (mobile+desktop, viewer+gpu) + smoke all green; validate-gpu
    ALL PASS incl. a new rescaled section at the SAME thresholds as fe (2^-90: rs 0.977% vs
    fe 1.074% — rescaled a hair better). New tools: crosscheck-skip.mjs, probe-rescaled.mjs;
    bench-gpu now A/Bs fe-vs-rescaled. NOTES/AGENDA/README updated.
- Spawn 5 (click-to-zoom + deep-zoom shader perf — user request): both done + validated.
  - CLICK-TO-ZOOM: viewer.js `clickZoom(px,py,factor)` recenters the clicked complex
    point to screen-center AND zooms (radius×factor), reusing the preview-transform +
    settle path (instant scaled preview, deferred sharp render). A no-drag tap/click
    → zoom in (0.5); shift/ctrl/right-click → zoom out (2); context menu suppressed.
    Removed the old double-tap handler (a single tap now zooms). +2 e2e (×2 projects)
    = 42 viewer e2e green: clickZoom math (recenter+zoom+deferred) + real-mouse wiring.
  - SHADER PERF (the headline): first ever TIMED on this host via new tools/bench-gpu
    .mjs (draw + 1-px readPixels forces SwiftShader to actually run the work — gl
    .finish is elided when the FBO is unread). Found fe was ~19× slower than df64.
    Two BIT-IDENTICAL opts (validate-gpu mismatch numbers unchanged to the digit):
    (1) fe normalize via IEEE-754 bit ops (fe_ilogb1=read exponent field, fe_pow2=
    write it) instead of log2/exp2 — kills ~60 software transcendentals per pixel-
    iteration (~1.85× on fe, AND removes the Adreno/Mali sparkle risk since it's now
    exact). (2) carry the reference Z[m] across loop iterations → one texture fetch/
    iter not two, in the df64+fe+f32 perturb loops (~2.4× on deep df64, ~1.2× on fe).
    Net matched-load A/B at the seahorse valley: df64 2^-80 ~109→~262 Mit-px/s (~2.4×),
    fe 2^-80 ~10.9→~21.7 (~2.0×). Honest: SwiftShader is CPU SW GL so extreme fe is
    still many seconds; the win helps every depth and a real GPU flies. Biggest
    remaining win (rescaled single-exponent iteration, ~1.5× more) documented in NOTES
    + NEXT #1 with its rebase-test catch — deferred for a real-GPU spawn, not risked.
  - Validation: 31 unit + 52 e2e (42 viewer + 10 gpu, ×2 projects) green; validate-gpu
    ALL PASS bit-identical (df64 2^-3..-110, fe overlap 2^-70..-110, fe exterior
    2^-130..-340). bench-gpu added. No correctness regression.
- Spawn 4 (point filter + supersampling + GPU 2^270 — user request): all three done.
  - NB: the reference URL Danielle gave for "ultra" 2^270 serves OUR EXACT code
    (byte-identical main.js/viewer.js/glsl.js) — it's this project on another host, so
    it ALSO falls back to CPU at 2^270; it just shows the target image quality.
  - POINT FILTERING: `#view { image-rendering: pixelated }` + `_applyPreview`
    imageSmoothingEnabled=false → crisp display + crisp zoom-gesture scaling (was
    bilinear-blurry). The supersample DOWNSCALE keeps smoothing on (that's the AA).
  - SUPERSAMPLING: render at ss× display res, box-average COLORS down (not sn —
    sn is cyclic). GPU: COLOR_FRAG averages the ss×ss block, FBO decoupled from the
    display canvas. CPU: offscreen compute canvas + rAF-coalesced downscale present.
    UI select (Off/2×/3×/4×, default 2×) + URL hash `ss=`; effSS capped for memory.
  - GPU 2^270 = FLOATEXP engine: extended df64 to `m*2^e` (df64 mantissa + int
    exponent) so dc/dz ~2^-270 stop underflowing float32's 2^-126 floor. New GLSL
    FE_LIB + perturbFragFloatexp; gpuEngineForRadius now naive / perturb-df64 /
    perturb-fe (2^-112..2^-340) / CPU. Gotchas: WebGL2=GLSL ES 3.00 has NO
    frexp/ldexp (3.10) → log2/exp2 normalize; no `?:` on structs. VALIDATED headless
    vs CPU oracle: varied chaotic escapes match df64 (2^-70/-90) AND escaping patches
    2^-130..2^-340 below the float32 floor at 0% mism. Wired as the GPU default for
    that band (CPU fallback on failure). Perf on real GPU still to be measured.
  - Tests: +6 unit (feSplit round-trip across 2^-340, dispatch bands) = 31 unit;
    +2 e2e (point-filter CSS, supersampling res+image-change); validate-gpu extended
    with the floatexp section (all PASS); smoke asserts gpu-perturb-fe at 2^-150.
    Fixed 2 deep e2e timeouts (ss=2 made SwiftShader 4× slower) by ss=1 in those.
- Spawn 3 (UX refinements — user request): iteration input field + zoom responsiveness.
  - Added a number input (`#iterNum`) beside the iterations slider so users can type
    an exact/large iteration count (1..2,000,000). Slider scrubs (100-step); the
    number field is precise. Both commit on `change` (release/Enter/blur), stay in
    sync, and uncheck "Auto iterations". index.html/styles.css/main.js.
  - Zoom now defers rendering: introduced `Viewer.zoomBy(factor,px,py)` →
    `_beginPreview()` (snapshot current frame, cancel in-flight render: gen++ +
    terminate pool) + `_scheduleSettle()` (debounced real render ~220ms after motion
    stops). Wheel, zoom +/- buttons, and double-tap all route through it and show the
    *scaled current image* during the gesture instead of re-rendering live.
  - Fixed a latent bug: a tap (pointerdown→up, no move) used to terminate the worker
    pool of an in-flight render. Preview now begins lazily on the first real move, so
    taps leave a running render alone. Also fixed deep double-tap killing its own
    render (it now previews+settles instead of render-then-terminate).
  - Tests: +3 e2e (iteration field commit/sync, zoom preview+deferred render, zoom
    mid-render cancellation). Full suite green: 26 unit + 34 e2e (17×2 projects).
- Spawn 2 (GPU / GLSL migration — user request): migrated the per-pixel raster to
  WebGL2 fragment shaders, GPU now the default engine, CPU pool the fallback+oracle.
  - Read mandelbrot-deep-zoom.md; took its perturbation/floatexp framing into the
    shader design (df64 reference + glitch criterion + smooth coloring parity).
  - Engines: naive-f32 (r>=2^-2), perturb-df64 (2^-2..2^-112), CPU below. df64
    naive + f32 perturb shaders also built & validated, kept as options.
  - KEY FINDING: f32 perturbation silently breaks at the viewer's real maxIter
    (10-30% boundary error) — the f32 reference reconstruction carries ~2^-24
    error that amplifies on chaotic pixels. df64 (reference AND deltas) fixes it
    to 0-1.2% mism / meanΔsn<1 (the residual = 46- vs 53-bit gap, confirmed
    precision-not-bug by the BigInt arbiter). VALIDATE ON BULK METRICS, NOT MAX.
  - Headless WebGL2 works via SwiftShader (EXT_color_buffer_float, RGBA32F, 8192
    tex) so GPU-vs-oracle runs in CI. New: src/gpu/{glsl,renderer,validate}.js,
    test/gpu/harness.html, tools/{validate-gpu,arbiter-gpu,probe-*,smoke-viewer}.mjs,
    test/e2e/gpu.spec.mjs. UI: GPU on/off toggle + active-renderer in debug.
  - 26 unit + 28 e2e (mobile+desktop) green; validate-gpu sweep green 2^0..2^-110.
- Spawn 1: built + VALIDATED the whole correctness core and a working viewer.
  - Math: naive oracle, BigInt fixed-point, HP reference orbit, perturbation +
    Zhuoran rebasing, reference auto-selection. 26 unit tests; perturbation
    matches BigInt-exact oracle to ±1 at 2^45/2^120/2^400 and full-pipeline deep.
  - Viewer: canvas + HP view state + pinch/pan/wheel/double-tap, 4 palettes,
    iteration slider, URL-hash bookmarks, naive(shallow)/perturb(deep) dispatch.
  - Rendering: worker POOL (compute reference once, fan row-bands across cores) +
    progressive coarse-then-fine. Full-screen deep render ~8x faster than single.
  - Tests: 18 Playwright e2e (mobile+desktop) all green. Screenshots in
    screenshots/ confirm correct home + seahorse(2^41) + spiral(2^60) renders.
  - Key finding: validate vs BigInt not naive (both double-only methods are noisy
    on ill-conditioned shallow boundary pixels). See NOTES.
  - Env: NixOS has no runnable Playwright chromium; use nix-store chromium 148 +
    --headless=new (older builds crash: no /sys/devices/system/cpu). See NOTES.
