# How the Deep Mandelbrot Zoom Algorithm Works

This codebase's "deep zoom" is the **perturbation-theory** Mandelbrot algorithm (the K.I. Martin / "Pauldelbrot" method), implemented in `mandelbrotPerturbation.js` (float64) and `mandelbrotPerturbationExtFloat.js` (for zoom past ~10^300). Here is exactly how it works.

## 1. Coordinate representation (fixed point)

Plane coordinates are arbitrary-precision integers with an implicit binary exponent. A real number x is stored as a BigInt mantissa X = round(x · 2^s), where s = `task.precision` is the scale chosen for the current zoom depth. Multiplication is `(X*Y) >> s` (product is at scale 2^(2s), shift back to 2^s). The number of mantissa bits grows roughly linearly with zoom depth, so this representation is *exact* but the per-multiply cost is O(s^2) — too slow to run on every pixel.

## 2. The reference orbit (computed exactly)

For one carefully chosen point c_ref (initially the frame center, `mandelbrotPerturbation.js:74-79`), the full orbit is iterated in BigInt fixed point (`mandelbrot_high_precision`, lines 274-301):

    Z_0 = 0,    Z_{n+1} = Z_n^2 + c_ref.

In code, Im(Z^2) = 2·Z_r·Z_i is `(zr*zi >> (s-1))` (the shift by s-1 rather than s supplies the factor 2), and Re(Z^2) = Z_r^2 - Z_i^2 is `zrq - ziq`. Each Z_n is then downscaled to a float64 and stored in the flat `zs` array with stride 3: (Z_n^r, Z_n^i, errBound_n). Because |Z_n| is bounded by the bailout radius, float64 captures it to full relative precision.

## 3. The perturbation recurrence (the heart)

Every other pixel c = c_ref + δ is expressed as a *deviation* from the reference orbit: write z_n = Z_n + ε_n. Substituting into z_{n+1} = z_n^2 + c and cancelling the exact identity Z_{n+1} = Z_n^2 + c_ref gives an **exact** (un-truncated) recurrence for the deviation:

    ε_{n+1} = 2·Z_n·ε_n + ε_n^2 + δ = (2·Z_n + ε_n)·ε_n + δ,    ε_0 = δ.

This is the whole trick: the catastrophic cancellation z_n - Z_n is performed *symbolically*, so although δ may be ~10^-300, ε_n only ever appears in products with the O(1) quantity 2·Z_n and in sums — operations that preserve float64 relative accuracy. Thus the inner loop (`mandlebrot_perturbation`, lines 189-231) runs entirely in fast float64:

    zzr = zr + ezr;  zzi = zi + ezi          // z_n = Z_n + ε_n  (reconstructed)
    zr_ezr_2 = zr + zzr   // = 2·Z_r + ε_r  (computed as Z_r + z_r for numerical quality)
    ezr = (zr_ezr_2*ezr - zi_ezi_2*ezi) + δ_r
    ezi = (zr_ezr_2*ezi + zi_ezi_2*ezr) + δ_i

The escape test uses the **reconstructed** |z_n|^2 = |Z_n + ε_n|^2 (`zzq`), bailing when it exceeds the radius (4, or 128 when smooth-coloring). In Julia mode δ is only the initial seed of ε_0 and the additive per-step term is 0 (`adr=adi=0`, line 109); in Mandelbrot mode the additive term *is* δ.

## 4. Glitch detection and reference exhaustion

The perturbation series is invalid wherever |z_n| becomes tiny relative to |Z_n| — there ε_n ≈ -Z_n and the float64 reconstruction Z_n + ε_n loses all significant digits. This is the **Pauldelbrot glitch criterion**. The reference precomputes errBound_n = |Z_n|^2 · 10^-6 (line 259), and the inner loop declares a glitch when |z_n|^2 < 10^-6·|Z_n|^2 (i.e. |z_n| < 10^-3·|Z_n|), returning -1 (lines 216-219). A second failure mode is `iter >= numZs` (line 201): the reference orbit itself escaped and ended, so a pixel that hasn't escaped yet has no more reference data — also -1. Either way the pixel needs a *different, closer* reference.

## 5. Multi-reference LRU cache + cross-frame reuse

A list `referencePoints[]` holds several reference orbits. For each pixel the algorithm tries them in LRU order from `head` (lines 100-129); on a hit, that reference is promoted toward the most-recently-used end (spatial coherence means neighboring pixels share a reference). On a total miss, a brand-new reference orbit is computed *at that pixel* in full precision and inserted (lines 131-138). Net effect: only a handful of O(s^2) BigInt orbits per frame, plus one cheap float64 orbit per pixel. `updateCache` (lines 145-173) additionally **translates and reuses** references across frames when only panning at constant zoom — each kept reference's pixel-delta is shifted by the frame displacement (`dr - deltar`) rather than recomputed; any zoom (precision) or parameter change flushes the cache.

## 6. Extended-float exponent tracking (zoom beyond ~10^300)

Plain float64 underflows once δ ≲ 10^-308, so `MandelbrotPerturbationExtFloat` represents ε_n as **mantissa × a per-iteration shared exponent**: the true deviation is ε_n^true = ezr · 2^(eExp[n]). Since ε_n grows as the orbit separates from the reference, the exponent must ramp from -s (where δ ≈ 2^-s) up toward 0. The code *predicts* that ramp with a power law (`calculate_reference`, line 276):

    eExp[n] = round( (n/N)^1.75 · s - s ).

Each step it rescales the running mantissa from the previous exponent to the current one by `eExpDeltaFactor` = 2^(eExp[n-1] - eExp[n]) (rescaling δ in lockstep, lines 212-217), and reconstructs z_n = Z_n + ezr·2^(eExp[n]) for the bailout test (line 229). This keeps the mantissa O(1) — neither underflowing nor overflowing float64's ±1023-bit exponent window. The author flags this as a heuristic with **no proof** (line 274): if the predicted exponent drifts more than ~1000 from the true magnitude of ε_n, results corrupt. The reference values themselves are converted BigInt→float by a fast top-500-bits shift, falling back to exact conversion only near zero (lines 303-311).

## 7. Smooth coloring

On escape the integer count is refined to a continuous value (`smoothen`, `workerContext.js`) by the normalized-iteration formula μ = n + 1 - log2( log|z_n| / log 2 ), with the fractional part quantized to a 0–255 sub-index for the gradient. (The various +4/+5/`return 2` offsets on the raw counts are just bookkeeping to align the three engines' outputs for coloring.)

---

In short: **one exact high-precision orbit + a closed-form exact deviation recurrence run in cheap float64, guarded by a relative-magnitude glitch test, served from an LRU cache of references, with a predicted per-step exponent track to push past float64's range.** The pure-BigInt engine in `mandelbrotFxP.js` computes the same fractal exactly but is kept only as a slow reference implementation.
