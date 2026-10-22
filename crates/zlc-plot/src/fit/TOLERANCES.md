# Fit engine: agreement with the Python oracle

Written before the numerics were ported; the port is held to it.  The oracle is
`zlc_plot.fit` / `_fit_compiled` / `_fit_radial` / `_fit_units` / `_fit_prepare`
/ `fit_evaluation` / `evidence` running in the product's science environment
(CPython 3.13, NumPy 2.4, numba 0.67, SciPy-OpenBLAS 0.3.30 "Haswell" core for
numba's LAPACK, NumPy's own SciPy-OpenBLAS 0.3.31 for `numpy.linalg`, MSVC UCRT
libm) -- on the same machine; see "CPU dependence" for what changes elsewhere.

**Result: every class below that asks for bit-exact agreement has it; no
non-zero tolerance is in use for any compiled-lane or regular-image output.**

## What the screen shows, and why that bounds every tolerance

A fitted value and its standard error are painted by `compact_number`:
`%.4g`, falling back to `%.1e` -- four significant digits, a relative
resolution of 5e-5 at the rounding boundary.  The evidence line is
`BIC gain {:.1f}`; a threshold is `compact_number` too.  A tolerance of
relative `t` can change one of those digits only when the true value lies
within `t` of a rounding boundary, i.e. with probability about `t / 5e-5`.
Every non-zero tolerance below is therefore at most 1e-9 relative (a flip
probability under 2e-5 per number), and every golden test also compares the
painted text (`compact_number` of value and error, the evidence and message
strings) character for character; a text difference is a failure whatever
the numeric tolerance says.

## Classes of output

| Output | Required agreement | Measured |
|---|---|---|
| Catalogue: model ids, names, parameter names, symbols, labels, formulas, unit relations, domains, domain bounds (incl. `nextafter` ends), targets, defaults, capabilities, reductions, registry order, `models_for` order | exact (bytes / bits) | exact, all 13 built-ins |
| Option validation, budgets (`evaluation_budget`), error/refusal messages (`_too_few_points`, bound refusals, histogram role text with `%g`) | exact text | exact |
| Model values and analytic Jacobians (all 13 built-ins, incl. the Poisson-Gaussian trapezoid grid and its Euler-Maclaurin terms, Lambert-W) | bit-exact | bit-exact.  Same scalar IEEE operations in the same order; Rust never contracts `a*b+c` and numba does not either; libm calls (`exp log log1p expm1 sin cos atan lgamma erf sqrt pow`) go to the same UCRT entry points (40 000 arguments each: 0 differences between numba, UCRT and NumPy; CPython's own `math.lgamma` differs and the fit never uses it) |
| Cold seeds, data-derived bounds, histogram confinement, sigma weights, binned sufficient statistics, peak finding, Poisson deviance residuals, `_authored_seed` / candidate de-duplication | bit-exact | bit-exact (see "Reductions" for the sorts, sums and medians) |
| Warm starts | a warm start is a whole previous solution (what the product passes: `tuple(result.parameter_values)`), checked as `_initial_values` checks a row | exact.  A *partial* (named) warm start -- which made Python complete it with its Python-lane initializer -- is not reachable from the product and is not accepted (see "Open gaps") |
| Trust-region solve (compiled lane): free parameters, cost, raw RSS, status, `nfev`, `njev`, iterations, winning seed | bit-exact | bit-exact |
| Covariance, standard errors, `covariance_valid` (compiled lane: `dgeqrf` of the scaled Jacobian, `dgesdd` of R) | bit-exact target, 1e-10 interim bound | bit-exact; the interim bound was never needed |
| Reduced chi-square, final fitted values and residuals, `parameter_units`, `batch_revision`, array lengths | bit-exact | bit-exact |
| BIC gain (`evidence`), reduction verdict, settled one-population parameters | bit-exact | bit-exact |
| Classifier threshold, fractions, fidelity (`_bimodal_classifier_metrics`), component curves | bit-exact | bit-exact |
| Regular-image fits (`_fit_radial`): crop, proxy subsample, seeds, both stages, RSS, status, solver diagnostics, deferred per-pixel arrays | bit-exact target, 1e-9 interim bound | bit-exact (u8/u16/i16/f32/f64 storage, masks, crops, NaN samples, every robust loss through the striped objective, radial and anisotropic, up to 1200x1920 and 2048-wide frames) |
| Regular-image covariance (`_covariance_from_information`: NumPy `eigh` of the normalised information, then a matmul) | bit-exact target, 1e-10 interim bound | bit-exact |
| Unit crossings of parameters and errors, expression parse and text (`fit_expression_target` / `fit_expression_text`) | exact bits / exact text | exact (8 unit contexts, every model; CPython's `float(str)` -- any Unicode decimal digit, Unicode spaces, underscores --, `str.strip` and `repr(str)` from CPython's own Unicode 15.1 tables: `crate::pystr`) |
| Target resolution (`resolve_fit_target`, `fit_request_target`, `prepare_fit_request`) | exact (structure, values, failure text) | exact (6720 resolutions: 8 unit contexts x 3 previous requests x 2 previous expression failures) |
| `evaluate_fit` (per-target results, failure texts, units; only cancellation propagates) | exact | exact (2804 targets of 221 session evaluations) |
| Success / failure classification of every cell (result vs failure message, status code, message text) | exact | exact |
| Cancellation | the same checks, in the same order, at the same points | exact: check counts equal Python's (single Gaussian 4, bimodal 8, 64-cell bimodal 134, 256-cell Gaussian 259, camera frame 9, circular mask 17), and every cancel-after-k case matches |
| Deadlines | the same clock starts (per `_fit_compiled_batch`, per image batch) and propagation (`FitDeadlineExceeded` is a `TimeoutError`: it fails a mixed batch's image cell, not the batch) | exact semantics; *which* call runs out of time is wall-clock, so a faster port can finish where Python times out |
| Fallback solver for models with no compiled descriptor (SciPy `least_squares` lane) | semantics exact (seed set, candidate ranking and tie rule, invalid-residual sentinel, warm shortcut, covariance rank test); numbers: relative 1e-8 of `max(|p|, sigma_p)` on converged well-posed fits | the port solves with the compiled lane's TRF, not SciPy's (an SVD of the full Jacobian per step), so iterates differ in the last bits; its covariance step (`np.linalg.svd(J, full_matrices=False)`) is reproduced bit for bit.  No registered model reaches this lane; it exists for caller-built models |

Evidence for the "measured" column (all run by `cargo test --release --test
fit --test fit_api`; the large sets with `ZLC_FIT_GOLDEN_LARGE`):

- `tests/data/fit.json`: 207 synthetic cells over all 13 models, failures,
  binned curves, every robust loss, warm starts, regular images in five
  storages with masks, crops and NaN samples, camera frames (120x192 u16:
  cold, warm, circular and holed masks, anisotropic with every loss),
  facet batches including the masked union-grid layout -- 207/207
  bit-exact, worst deviation 0.  Every result also compares its units,
  batch revision, array lengths and the solver row its parameters came from.
- `golden_fit/fit_large.json`: 8 large cells (9001-point binned, 512x512
  images, 1200x1920 u16 camera frames cold/warm/circle/anisotropic/soft_l1)
  -- bit-exact.
- `golden_fit/captured.json`: every fit and every `evaluate_fit` the recorded
  sessions (corpus and recbase replays) made, 222 calls / 2805 cells and
  221 evaluations / 2804 targets -- all bit-exact; a stratified sample (one
  case per model, kind, storage, mask, loss, initial, bounds, warm and
  outcome: 48 strata, every one the full set has; 17 evaluations) is
  committed as `tests/data/fit_captured.json`.
- `tests/data/fit_api.json`: 27 non-ASCII expressions per model and context,
  780 `float(str)` and 2612 `repr(str)` probes, 6720 resolutions -- exact.
- 600 LAPACK probe matrices, 120 `ddot` probes, 286 `dgemm` probes (every
  square side 1..13 across the unroll and blocking seams) and 73 `dnrm2`
  probes (subnormal to 1.7e308) -- bit-exact.
- The image passes that read a plane in its own storage are checked against
  their definitions across the kernel's chunk and tail seams (widths to
  5000, odd heights, NaN samples, masks).
- An independent review re-ran these and 2510 fresh synthetic cells, 3456
  facet, 624 BIC, 1854 degenerate (subnormal and 1e-150-scaled data), 288
  cancellation, 72 anisotropic-probe and 16 camera-frame cells: all
  bit-exact (after the fixes listed in the commit history), except the
  partial warm starts the port no longer accepts.
- The scalar fallback of every vectorised loop (CPUs without AVX/FMA) is
  bit-exact on the same sets.

## BLAS and LAPACK: where the Python fit calls them and how the port matches

Summation order alone cannot reproduce these; each routine was identified by
probing the very library the oracle loads (random operands until a model of
its kernel matched every case, and, where that left cases over, recovering
the summation tree from the library itself with large/small cancellation
probes).  Generators in `tests/gen/`.

| Python call | Library routine reached | Reproduction (`blas.rs`, `linalg.rs`) |
|---|---|---|
| numba `np.linalg.eigh(trust_information)` in `_solve_seed` (when the Cholesky step is rejected; at most 6x6) | SciPy-OpenBLAS 0.3.30 `dsyevd` -> `dsytrd` (unblocked `dsytd2` at this size) -> `dsteqr` -> `dormtr` | reference LAPACK 3.12 `dsytd2` / `dsteqr` / `dorm2r` / `dlarfg` / `dlartg` (f90) / `dlaev2` / `dlasr` / `dlascl`, calling emulations of the Haswell `dsymv_L` and `dsyr2` kernels |
| `_GEQRF` (the direct `dgeqrf` binding) in `_finalize_one` | `dgeqrf` -> `dgeqr2` -> `dlarfg` (`dnrm2`: the x87 kernel at 53-bit precision with the x87 exponent range) and `dlarf` (`dgemv_t`, `dger`) | `qr_r`; `dnrm2`'s four interleaved sums, scaled by exact powers of two when the largest element would leave double's range (a subnormal largest element included), `dgemv_t`'s 4/2/1-column blocks with their own lane counts and FMA/no-FMA choice, `dger` as column `daxpy` |
| numba `np.linalg.svd(R)` in `_finalize_one` | `dgesdd` path 5: `dgebd2`, `dbdsdc` -> `dlasdq` -> `dbdsqr` (`dlasv2`, `dlas2`), `dormbr` | `svd_small` (singular values and V^T; U is computed as LAPACK does, it shares no arithmetic with V^T) |
| numba `np.dot(a, b)` (1-D) and `image @ vector` in `_fit_radial` | Haswell `ddot` (16-element FMA blocks, 4x4 lanes, pairwise lane reduction, plain tail); `dgemv` 'T' on the C-contiguous image (2048-column chunks, 4-row FMA blocks, `n & 3` rows plain, `m & 3` columns last) | `ddot` (its lanes can be carried across pieces of one sum); `dgemv_t`, and for image planes `plane::plane_products`: the same per-row operations on samples converted and centred as they are read, several basis vectors per read |
| NumPy `np.dot(a, a)` (1-D) in `fit.py`: binned rescale, all-fixed cells, evidence deviances, fallback RSS | NumPy's SciPy-OpenBLAS 0.3.31 `ddot` -- the same Haswell kernel | `ddot` |
| NumPy `jacobian.T @ weighted_jacobian` in the striped (masked / robust) image objective | 0.3.31 `dgemm`: K split into `GEMM_Q = 128` blocks (the last two halved, rounded to the unroll of 4), one FMA chain per element and block, blocks added in order -- except the 4x1 micro-tiles (the last row when the side is odd, against each full group of four columns): four lanes per block (K mod 4 over the block's multiple of eight, the rest into lane 0) summed `(l0 + l1) + (l2 + l3)`; a 1x1 product is `ddot` | `gemm_at_b` |
| NumPy stacked `np.linalg.eigh` and `(vectors / values) @ vectors^T` in `_covariance_from_information` | NumPy's `dsyevd` (same reference code and kernels as numba's) and a small `dgemm` | `eigh_lower`, then FMA accumulation in `dgemm`'s order |
| NumPy `np.linalg.svd(J, full_matrices=False)` in the fallback lane's covariance | `dgesdd` path 3: `dgeqrf` of J, then the path-5 work on R | `qr_r` then `svd_small` -- bit-identical to the full SVD's singular values and V^T |

The Python fit calls no `lstsq`, `cholesky` or `solve`: the TRF's Cholesky
step is numba scalar code, ported as such.  `np.fft.rfft` appears only in a
Python-lane initializer the product no longer reaches (see "Open gaps").

The reproduced kernels use hardware FMA where the library does (bit-identical
to the correctly rounded libm `fma`), and their inner loops run on AVX lanes
when the CPU has AVX and FMA: each lane performs the same IEEE operation, in
the same order, as the scalar loop it replaces.

## Reductions and parallel work

- NumPy's pairwise `np.sum` / `np.mean` (and the float32 / buffered-cast
  cases): `crate::numeric`, bit-identical to NumPy 2.4.
- numba's sequential `np.sum` / `np.mean`, numba's quicksort `np.sort` /
  `np.argsort` (unstable; NaN last), NumPy's `np.median` (partition, mean of
  the middle pair): `num.rs`.  A median by selection equals the sort's
  middle elements whenever they are not a signed zero; when they are, the
  full numba sort runs, so zero signs come out as numba's.
- `math.fsum`: a port of CPython's `msum`.
- Work is split across threads only where every output keeps its own
  sequence of operations: cells of a batch; buckets one after another (the
  oracle's checks between them), their cells in parallel; a large cell's
  seeds, weighed afterwards in seed order; image rows of `image @ x` (each a
  separate dot product); stripes of the centring moments, residual RSS and
  striped objective, combined in stripe order.  A sum the oracle forms in
  one sequence (the point-by-point image objective under a mask or a robust
  loss, a TRF iteration) stays on one thread.

## Decisions that sit on a threshold

`evidence > min_bic_gain`, `covariance_valid` (a strict singular-value rank
test), `status` (ftol/xtol/gtol tests) and the classifier's minimum component
shots are discontinuous.  Where the numbers are bit-exact so are the
decisions -- which is everywhere except the fallback lane, whose decisions
can differ only when the Python operand lies within 1e-8 of the threshold.

## Iteration counts

Compared exactly wherever the solve is bit-exact (the compiled lane and the
regular-image route).  Not compared on the fallback lane.

## Stalled regular-image solves (a deliberate departure)

A regular-image seed solve still short of its tolerances after 100
iterations (`trf::STALL_ITERATIONS`) is finished the port's own way, not the
oracle's: on the proxy it hands its point to the refinement as settled; in
the refinement (linear loss, every sample valid, every parameter free) it
goes on with the amplitude and offset solved out (variable projection, the
Schur complement of the information on the linear pair).  This is a
single Gaussian drawn onto one atom of a many-spot frame: the oracle creeps
there for thousands of iterations, often to the evaluation budget, and
stops unconverged.  Every solve that settles within 100 iterations -- every
golden and captured case -- is bit-exact as before.  On 120 live 372x460
frames of the 5x7 virtual array: the 7 the oracle left unconverged converge,
1.8-12% lower in cost; 37 that it converged slowly reach the same minimum
(cost within 3e-11, parameters within 6.2e-5 relative, the flat directions
the cost tolerance leaves); 2 reach a different, lower minimum; 1 lands on a
neighbouring atom of equal cost (+4.5e-6).

## CPU dependence (decision)

Python's own answers are a per-CPU fact.  OpenBLAS picks its kernel family
when it loads (Haswell on this machine) and UCRT picks an FMA3 or an SSE2
path for its libm.  Re-running the oracle with OpenBLAS forced to other
families (an independent review): Sandybridge, Nehalem and pre-AVX kernels
differ from Haswell in 175-177 of 197 synthetic cells -- mostly the last bit
of covariance entries, in 16 cells radial-image parameters by about 1.4e-8
relative, in 4 the termination message -- and UCRT without FMA3 differs
again.

The port reproduces the Haswell kernels and calls the same UCRT libm, so it
equals Python on the same machine for every CPU on which OpenBLAS runs its
Haswell kernels -- AVX2 + FMA processors: Intel Haswell and later client
cores, AMD Zen 1-3.  On those, "bit-exact" above holds as stated.  It is a
statement per CPU, not across CPUs: on an older CPU Python's numbers move by
the magnitudes just given while the port's do not (its scalar fallback still
computes the Haswell arithmetic, through libm's exact `fma`); on AVX-512
CPUs, where OpenBLAS selects its SkylakeX kernels, Python's numbers are
expected to differ from these in the same way, and the port then equals
Python-on-Haswell, not Python-on-that-CPU.

## Speed

`tests/gen/fit_bench.py` makes the inputs and times the Python engine (warm
numba cache); `examples/fit_speed.rs` times the port on the same inputs and
checks every answer to the bit; `tests/gen/fit_interleave.py` alternates
rounds of both.  "1t": numba with one thread (NumPy's BLAS keeps its own) and
a one-thread rayon pool; "16t"/"pool": numba's 16 threads and the default
rayon pool.  Median (minimum) ms per call over 15 calls in three interleaved
rounds, both sides warm, on an Intel Core Ultra 7 265H (hybrid P/E cores)
that other jobs kept 50-100 % busy -- repeat runs move by up to ~30 %:

| case | python 1t | python 16t | port 1t | port pool | x 1t | x pool | answers |
|---|---|---|---|---|---|---|---|
| one 201-point Gaussian | 0.971 (0.891) | 0.960 (0.881) | 0.109 (0.088) | 0.076 (0.055) | 8.9 | 12.6 | bit-identical |
| 40-cell facet batch, 201-point Gaussians | 12.762 (11.227) | 4.937 (4.390) | 2.989 (2.515) | 1.394 (0.996) | 4.3 | 3.5 | bit-identical |
| 24-cell bimodal histograms (with reduction) | 11.191 (9.439) | 8.015 (7.060) | 3.070 (2.501) | 1.262 (0.977) | 3.6 | 6.4 | bit-identical |
| 128x128 radial image | 9.606 (8.914) | 10.504 (9.878) | 1.477 (1.251) | 1.369 (1.080) | 6.5 | 7.7 | bit-identical |
| 512x512 radial image | 12.536 (10.943) | 12.696 (11.258) | 2.596 (2.017) | 2.735 (1.839) | 4.8 | 4.6 | bit-identical |
| 9001-point Gaussian, binned | 7.285 (6.498) | 7.132 (6.672) | 2.403 (2.098) | 1.821 (1.505) | 3.0 | 3.9 | bit-identical |
| camera frame, radial, cold | 36.100 (33.258) | 30.435 (28.956) | 13.848 (11.869) | 6.666 (5.273) | 2.6 | 4.6 | bit-identical |
| camera frame, radial, warm (previous answer) | 37.041 (33.225) | 30.997 (29.861) | 14.344 (12.078) | 6.507 (5.196) | 2.6 | 4.8 | bit-identical |
| camera frame, anisotropic | 42.748 (39.207) | 36.718 (35.332) | 16.640 (14.758) | 7.589 (6.064) | 2.6 | 4.8 | bit-identical |
| camera frame, radial, circular selector (r = 200 px) | 30.880 (27.519) | 27.168 (24.437) | 22.347 (19.402) | 15.544 (12.475) | 1.4 | 1.7 | bit-identical |

The camera frames are 1200x1920 uint16, handed over in their own type.

**Where a camera-frame fit's time goes, and its floor.**  The cold radial
fit reads the frame (4.6 MB) eleven times: the centring moments once, nine
refinement evaluations (six trials at one `image @ x` each, three
derivative calls at two or three -- an accepted trial's product reused), and
the residual RSS once: 51 MB, with about 30 M FMAs.  A plain read of the
frame on this machine takes 0.29 ms on one core and 0.07 ms on all of them,
so the frame passes alone need about 3.2 ms on one thread and 0.8 ms on the
pool (the FMAs, at two 4-lane FMAs a cycle, about 0.8 ms on one core).  The 129x129 proxy is a sequence of
trust-region steps (80 evaluations, 73 of them in the losing seed) that no
split shortens below its slowest seed, about 1.5 ms.  Floor: about 5 ms on
one thread and 2.5 ms on the pool, against 13.8 / 6.7 ms in the loaded
run above and 10.8 / 5.1 ms on a quieter one: 2-3x from it on both.  The masked fit is further: its objective is the
oracle's single point-by-point sequence over every valid pixel, one core.

Each replaced BLAS/LAPACK routine, timed the way the fit calls it, on
matrices from those fits (`tests/gen/fit_routines.py` / `examples/
fit_routines.rs`; best of seven batches, ns per call; every output
bit-identical):

| routine (Python call) | Python | port | ratio |
|---|---|---|---|
| numba `eigh` 4x4 (Gaussian) | 1562 | 1408 | 1.1 |
| numba `eigh` 6x6 (bimodal) | 3456 | 3405 | 1.0 |
| `dgeqrf` 201x4 | 1565 | 1583 | 1.0 |
| `dgeqrf` 4096x4 | 31787 | 27652 | 1.1 |
| `dgeqrf` 60x6 | 1487 | 1230 | 1.2 |
| numba `svd` of R, 4x4 | 3827 | 2383 | 1.6 |
| numba `svd` of R, 6x6 | 6660 | 4998 | 1.3 |
| numba `image @ vector` 512x512 | 43404 | 21052 | 2.1 |
| numba `image @ vector` 129x129 | 2571 | 1520 | 1.7 |
| numba `np.dot` 32768 (one image stripe) | 5092 | 3414 | 1.5 |
| NumPy `np.dot` 32768 | 6254 | 3395 | 1.8 |
| numba `np.dot` 201 | 45 | 19 | 2.4 |
| NumPy `np.dot` 201 (Python call overhead) | 732 | 14 | 50 |
| NumPy `J.T @ WJ` 32768x4 | 176588 | 43797 | 4.0 |
| NumPy stacked `eigh` 5x5 | 9770 | 2206 | 4.4 |
| NumPy `svd` 201x4 (fallback lane) | 18659 | 4076 | 4.6 |

Where LAPACK is close: OpenBLAS's `dsyevd` and `dgeqrf` are at parity with
the port at these sizes (1.0-1.2x); for matrices larger than any the fit
forms (LAPACK switches to blocked level-3 code from a few dozen columns) the
libraries would likely pull ahead.

Three Windows effects matter as much as arithmetic.  The process heap
decommits freed blocks once more than 64 KiB lie free, so every fresh
temporary that size pays a page fault per 4 KiB: `scratch.rs` keeps a few
large buffers per thread.  A closure does not inherit a function's
`target_feature`, so each parallel task's body is compiled for AVX2/FMA on
its own (a `mul_add` outside one is a libm call).  And the UCRT `exp`
(about 3-7 ns) dominates the point-model objectives; the port calls the same
function for bit-exactness and so pays the same price numba does.

## Open gaps

- Partial warm starts are refused.  Python completes a named warm start
  that misses parameters with its Python-lane initializer (which orders
  points with NumPy's unstable `np.argsort` and seeds a damped sine from
  `|np.fft.rfft|`); no product path makes one -- the product's warm starts
  are whole previous solutions and its operator guesses go through
  `_authored_seed` -- so the path was removed rather than reproduced.
- The fallback (SciPy `least_squares`) lane agrees to 1e-8, not bitwise;
  the Python-lane initializers it uses for a caller-built built-in model
  carry the same argsort and rfft caveats.  The product fits its registry's
  own models and reaches neither.
- Deadlines are wall-clock: which call exceeds one depends on speed.
