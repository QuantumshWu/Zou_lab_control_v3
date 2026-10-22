# Projection agreement policy

Written before any numerics were ported; it is the contract the golden tests
enforce. The oracle is the Python science package at the worktree's
`science/src` (numpy 2.4.2, numba 0.67, `ZLC_PLOT_KERNELS` unset, i.e. the
compiled kernels engaged) in the environment `run_chain.ps1` gives the app
(`harness/env.sh`: `NUMBA_NUM_THREADS=16`, `OPENBLAS_NUM_THREADS=1`).

Single-threaded BLAS is a precondition, not a detail: with
`OPENBLAS_NUM_THREADS=4` OpenBLAS splits the `np.dot` of
`_centred_moment_totals` across threads and Python's own rolling SEM moves by
1-12 ulp (5 of 56 single-frame cases). The agreement below is with the
single-threaded oracle; the native projection computes that dot itself and
has no BLAS threading to configure.

## What is compared

Every projection payload field the Python objects carry: `CurveData`,
`CurveSeries`, `ImageData`, `HistogramData`, `RollingHistory`, `FacetData`,
`FacetCell`, `SelectionSubject`, `QuantityArray`/`CoordinateArray` (canonical
AND display arrays, both units, label), `AxisValue` (index, canonical, display,
label), x labels, group keys, revision and generation, plus the histogram
domain state (`HistogramProjection`) that is carried between revisions and the
fit inputs (`FitSelection`).

Arrays are compared with their dtype and shape, element by element, as raw
bits. "Bit-exact" below means: same dtype, same shape, identical bit pattern
for every element (so `NaN` payloads, `-0.0` and `+0.0` are distinguished).

## Level 1 -- exact by construction (no arithmetic choice exists)

Counts, validity/presence masks, selected indices, bucket codes, coordinates
(domains are gathers and unit conversions are one IEEE operation per
element), first/last, histogram counts, integer sums, every label and
every text (axis labels, `AxisValue.label`, series labels, x tick labels:
Python `str`/`repr` of the scalar, reproduced by `pyfmt::repr`), selection
subjects, scope coordinates. Any difference is a bug.

## Level 2 -- floating reductions: bit-exact by replicating the order

Sums, means, centred moments and SEM depend on summation order. The port does
NOT pick its own order: it reproduces the order of the exact Python path that
would have run, which means it reproduces Python's path SELECTION too (dense
tensor / factored lattice / generic axis-code / segmented history), because
each path sums in a different order:

* Compiled kernels (`aggregate_axis_codes`, `masked_leading_float64`,
  `centred_moment_sums`, `_trailing_moment_windows`, `uniform_histogram`,
  `finite_extrema`) are sequential loops in a fixed order (their `prange` is
  over independent outputs, or over integer/order-free partials). Ported
  operation for operation, in the same dtypes.
* NumPy `add.reduce` (`np.sum`/`np.mean`, with or without `where=`), modelled
  and verified against numpy 2.4.2 on random data with full-mantissa,
  wide-exponent values (635 of 635 shape/axes/mask/dtype cases bit-equal):
  - axes are iterated in C order with size-1 axes dropped and adjacent axes
    coalesced when every operand's strides allow it (the output has stride 0
    on reduced axes);
  - if the innermost coalesced axis is reduced, each output element receives
    `out += pairwise(run)` per contiguous run (numpy's pairwise sum: below 8
    elements a left fold from `-0.0`; up to 128 eight accumulators combined
    `((r0+r1)+(r2+r3))+((r4+r5)+(r6+r7))` then the tail; above, split at
    `n/2` rounded down to a multiple of 8); otherwise the reduction is an
    elementwise sequential accumulation over the outer axes;
  - the output starts at the identity `+0.0`;
  - with `where=`, runs are the maximal runs of `True` inside the inner loop;
  - when the input dtype is not float64 (a cast is buffered) the inner
    reduced loop is cut into chunks of 8192 elements from the start of each
    inner row, and mask runs are cut at those chunk boundaries;
  - `np.mean` is that sum divided once by the count.
* NumPy `minimum`/`maximum` reductions (`np.min`/`np.max` with or without
  `where=`/`initial=`, `ufunc.reduceat`) select an operand, so only the
  choice between `-0.0` and `+0.0` and which NaN comes out can differ; the
  port runs numpy 2.4.2's AVX2 loops (`numeric::Extremes`): float64/float32
  `npyv_minn` on 4/8-lane vectors and in the scalar tail (a NaN first
  operand sticks, otherwise a tie keeps the SECOND operand); a contiguous
  reduced run folds eight vectors per step as a tree into the accumulator,
  then single vectors, then the horizontal halving tree (a NaN lane gives
  the canonical `0x7ff8...` NaN), then the scalar tail; float16 is a scalar
  loop whose ties keep the FIRST operand; without `initial` the first
  element along the reduced axes seeds the output and is skipped. Verified
  on 12000 contiguous, 3200 `reduceat`, 9256 multi-axis/masked/`initial`
  and 4872 float16 cases. The compiled kernels' `<=`/`>=` (ties keep the
  later sample) and `finite_extrema`'s `<` are ported as written, and
  Python's builtin `min`/`max` keep the first of equal arguments.
* Path selection includes the NaN of an empty bucket: numpy's MEAN divides
  `0/0` (x86's negative NaN, `0xfff8...`) where the fused leading kernel
  (float64, at least 32768 samples, two or more outputs, holes) writes
  `np.nan` (`0x7ff8...`).
* A broadcast validity plane is a path selector: `_centred_moment_sums`
  declines a mask that is not C-contiguous (validity components that leave
  a cell axis wider than one sample unnamed, a whole-block bool), and numpy's
  `where=` sums run instead of the compiled kernel. NumPy's masked
  reductions coalesce axes by the value and output strides only, whatever
  the mask's strides (verified on 1454 broadcast-mask sums and 5786
  broadcast-mask extremes).
* `np.bincount(weights=...)` accumulates in input order: sequential.
* Python `int / int` (integer histogram bin widths and counts) is the exact
  quotient rounded once (`histogram::true_divide`), not two float64
  roundings: an int64 span past 2^53 makes the difference.
* Elementwise numpy arithmetic (`/`, `*`, `-`, `np.square`, `np.sqrt`,
  `np.where`) is one correctly rounded IEEE operation per element: identical.
* `np.linspace`/`histogram_edges`: `i*step + start` with the last edge set to
  `stop` (numpy's formula, including the `step == 0` branch and float32 edges).

Integer-valued inputs (camera frames: `u8`, `u16`) make every sum exact in
float64 whatever the order, so these are Level 1 in practice; the replicated
order matters for float64/float32 data (fit outputs, derived signals).

## Level 3 -- declared tolerances (the only ones)

1. `np.dot` inside `_centred_moment_totals` (one-shot whole-revision rolling
   SEM, `uncertainty` on, no repeat/history axis): the BLAS `ddot` order
   depends on the OpenBLAS kernel the CPU selects. On this machine's x86-64-v3
   (AVX2) class it is eight two-lane FMA accumulators per 16 elements,
   combined pairwise, then the tail added in order; `numeric::blas_dot_self`
   replicates it (239/239 lengths bit-equal), so it is Level 2 here. On a CPU
   where numpy's OpenBLAS picks another kernel (AVX512, Zen-specific) the
   bound is |rel| <= 4e-16 * log2(n) on the second moment, hence on the
   variance, and half that on the SEM: a band of half-height
   `sem * px_per_unit` moves less than 1e-11 px on a 4096 px panel, four
   orders under Agg's 1/256 px subpixel grid, and the SEM is never printed
   with more than 6 significant digits.
2. Transcendental unit conversions (decibel and voltage-into-load units:
   `10**x`, `log10`, `sqrt` of a product) use the platform libm in both and
   agree bit for bit here (dBm, Vpp, mVrms goldens); numpy dispatches AVX512
   SVML loops for `log10`/`power` on CPUs that have them, where the bound is
   1 ulp of the displayed value. Linear and offset units are Level 1.
Refusals are part of the contract: every Python exception a projection, a
selector mask, a fit selection or a selection subject raises is raised with
the same exception class (no aliasing between `DataViewError` and its
subclass `AxisResolutionError`) and the same text, in Python's check order,
and the histogram domain a refused revision leaves behind is compared like an
accepted one's. This includes the refusals that are
Python-side accidents rather than design (numba refusing float16 planes in the
centred-moment and finite-extrema kernels, and the unbroadcast window mask of
a stride-0 validity plane meeting a float histogram's extrema pass).
`float32` and `float16` planes are centred in their own precision where
NEP 50 makes numpy do so (a Python float meeting a float32 array).

Any other difference fails the golden tests. A tolerance is never applied to
a Level 1 or Level 2 quantity to make a test pass; the order is fixed instead.

## Out of scope here (owned elsewhere)

Resolution of the configuration into `ProjectionPlan` (spec defaults, fate
table, `projection_scope`), fit solving and overlays (`src/fit/`), rendering.
