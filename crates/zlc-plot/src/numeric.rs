//! numpy's floating reductions, in numpy's operation order, so native sums
//! and means equal the Python ones bit for bit (numpy 2.4, x86-64).
//!
//! - A contiguous float64 (or float32) reduction is the pairwise sum numpy's
//!   add loop performs, started from the identity 0: eight accumulators per
//!   128-block, halving above it (`pairwise_sum` in loops_utils.h).
//! - A reduction that casts (float32 or integers summed with dtype=float64)
//!   runs through numpy's 8192-element buffer: each chunk is cast and summed
//!   pairwise, and the chunk sums accumulate in order.

const BLOCK: usize = 128;
/// numpy's default iterator buffer size, in elements.
pub const BUFFER: usize = 8192;

macro_rules! pairwise {
    ($name:ident, $t:ty) => {
        /// numpy's pairwise sum of a contiguous run (no starting value).
        pub fn $name(values: &[$t]) -> $t {
            let n = values.len();
            if n < 8 {
                let mut result: $t = 0.0;
                for &value in values { result += value; }
                result
            } else if n <= BLOCK {
                let mut r: [$t; 8] = [values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]];
                let mut i = 8;
                while i < n - (n % 8) {
                    for j in 0..8 { r[j] += values[i + j]; }
                    i += 8;
                }
                let mut result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
                while i < n { result += values[i]; i += 1; }
                result
            } else {
                let mut half = n / 2;
                half -= half % 8;
                $name(&values[..half]) + $name(&values[half..])
            }
        }
    };
}

pairwise!(pairwise_f64, f64);
pairwise!(pairwise_f32, f32);

/// `np.sum(values)` of a contiguous float64 array.
pub fn sum_f64(values: &[f64]) -> f64 { 0.0 + pairwise_f64(values) }

/// `np.sum(values)` of a contiguous float32 array (float32 accumulation).
pub fn sum_f32(values: &[f32]) -> f32 { 0.0 + pairwise_f32(values) }

/// `np.sum(values, dtype=np.float64)` of a contiguous array of another dtype.
pub fn sum_cast_f64<T: Copy + Into<f64>>(values: &[T]) -> f64 {
    let mut total = 0.0;
    let mut chunk = Vec::with_capacity(BUFFER.min(values.len()));
    for part in values.chunks(BUFFER) {
        chunk.clear();
        chunk.extend(part.iter().map(|&value| value.into()));
        total += pairwise_f64(&chunk);
    }
    total
}

/// `np.mean(values)` of a contiguous float64 array.
pub fn mean_f64(values: &[f64]) -> f64 { sum_f64(values) / values.len() as f64 }

/// `np.mean(values, dtype=np.float64)` of a contiguous array of another dtype.
pub fn mean_cast_f64<T: Copy + Into<f64>>(values: &[T]) -> f64 { sum_cast_f64(values) / values.len() as f64 }

/// An element numpy widens to float64 exactly (every real dtype).
pub trait Widen: Copy + Send + Sync {
    fn widen(self) -> f64;
}

macro_rules! widen { ($($t:ty),*) => {$(impl Widen for $t { #[inline(always)] fn widen(self) -> f64 { self as f64 } })*}; }
widen!(u8, i8, u16, i16, u32, i32, u64, i64, f32, f64);
impl Widen for bool { #[inline(always)] fn widen(self) -> f64 { if self { 1.0 } else { 0.0 } } }

/// [`pairwise_f64`] of `values` widened element by element: what numpy's
/// add loop computes on a buffered (cast) chunk.
pub fn pairwise_widened<T: Widen>(values: &[T]) -> f64 {
    let n = values.len();
    if n < 8 {
        let mut result = 0.0;
        for value in values { result += value.widen(); }
        result
    } else if n <= BLOCK {
        let mut r = [0.0_f64; 8];
        for j in 0..8 { r[j] = values[j].widen(); }
        let mut i = 8;
        while i < n - (n % 8) {
            for j in 0..8 { r[j] += values[i + j].widen(); }
            i += 8;
        }
        let mut result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n { result += values[i].widen(); i += 1; }
        result
    } else {
        let mut half = n / 2;
        half -= half % 8;
        pairwise_widened(&values[..half]) + pairwise_widened(&values[half..])
    }
}

/// `out += pairwise(run)` per buffer chunk (a cast) and per run of `True`.
fn reduce_run<T: Widen>(values: &[T], mask: Option<&[bool]>, buffered: bool, out: &mut f64) {
    let n = values.len();
    let chunk = if buffered { BUFFER } else { n.max(1) };
    let mut low = 0;
    while low < n {
        let high = (low + chunk).min(n);
        match mask {
            None => *out += pairwise_widened(&values[low..high]),
            Some(mask) => {
                let mut j = low;
                while j < high {
                    while j < high && !mask[j] { j += 1; }
                    let mut k = j;
                    while k < high && mask[k] { k += 1; }
                    if k > j { *out += pairwise_widened(&values[j..k]); }
                    j = k;
                }
            }
        }
        low = high;
    }
}

/// One iteration axis after numpy's coalescing: extent and the input and
/// output strides in elements (the output stride is 0 on a reduced axis).
#[derive(Clone, Copy, Debug)]
struct Dim { n: usize, input: usize, output: usize }

/// numpy's iterator over a C-contiguous input reduced over `axes`: C order,
/// size-1 axes dropped, adjacent axes merged where every stride allows.
fn coalesced(shape: &[usize], axes: &[usize]) -> (Vec<Dim>, usize) {
    let ndim = shape.len();
    let mut input = vec![1usize; ndim];
    for axis in (0..ndim.saturating_sub(1)).rev() { input[axis] = input[axis + 1] * shape[axis + 1]; }
    let mut output = vec![0usize; ndim];
    let mut kept = 1usize;
    for axis in (0..ndim).rev() {
        if !axes.contains(&axis) { output[axis] = kept; kept *= shape[axis]; }
    }
    let mut dims: Vec<Dim> = (0..ndim).filter(|axis| shape[*axis] != 1)
        .map(|axis| Dim { n: shape[axis], input: input[axis], output: output[axis] }).collect();
    if dims.is_empty() { dims.push(Dim { n: 1, input: 0, output: 0 }); }
    let mut merged = vec![*dims.last().unwrap()];
    for dim in dims.iter().rev().skip(1) {
        let inner = merged[0];
        if dim.input == inner.n * inner.input && dim.output == inner.n * inner.output {
            merged[0] = Dim { n: inner.n * dim.n, input: inner.input, output: inner.output };
        } else {
            merged.insert(0, *dim);
        }
    }
    (merged, kept)
}

/// `np.add.reduce(values.reshape(shape), axis=axes, dtype=float64, where=mask)`
/// for a C-contiguous input, in numpy's order: the output starts at 0; a
/// reduced innermost axis adds `pairwise(run)` per run of `True` (cut into
/// 8192-element buffer chunks when `buffered`, i.e. the input is cast);
/// otherwise rows accumulate elementwise in C order. `mask` is a full plane
/// over `shape`. Verified bit-equal to numpy 2.4.2 on 635 shape/axes/mask/
/// dtype cases (projection/AGREEMENT.md). The result is C-ordered over the
/// kept axes.
pub fn add_reduce<T: Widen>(values: &[T], shape: &[usize], axes: &[usize], mask: Option<&[bool]>, buffered: bool) -> Vec<f64> {
    let (dims, kept) = coalesced(shape, axes);
    let mut out = vec![0.0_f64; kept];
    let inner = *dims.last().unwrap();
    let outer = &dims[..dims.len() - 1];
    let total: usize = outer.iter().map(|d| d.n).product();
    let mut index = vec![0usize; outer.len()];
    let (mut ib, mut ob) = (0usize, 0usize);
    for _ in 0..total {
        if inner.output == 0 {
            let run = &values[ib..ib + inner.n];
            reduce_run(run, mask.map(|m| &m[ib..ib + inner.n]), buffered, &mut out[ob]);
        } else {
            match mask {
                None => for j in 0..inner.n { out[ob + j * inner.output] += values[ib + j].widen(); },
                Some(mask) => for j in 0..inner.n { if mask[ib + j] { out[ob + j * inner.output] += values[ib + j].widen(); } },
            }
        }
        for axis in (0..outer.len()).rev() {
            index[axis] += 1;
            ib += outer[axis].input;
            ob += outer[axis].output;
            if index[axis] < outer[axis].n { break; }
            ib -= outer[axis].input * outer[axis].n;
            ob -= outer[axis].output * outer[axis].n;
            index[axis] = 0;
        }
    }
    out
}

/// Element visits one parallel part has to carry to pay for itself. Smaller
/// work runs on the calling thread: waking a pool worker costs tens of
/// microseconds, and on a loaded machine one preempted worker stalls the
/// whole call (a 1.2 M-sample facet reduction: 0.2 ms on one thread, 0.65 to
/// 2.9 ms spread over 16 while other builds ran).
pub const PARALLEL_GRAIN: usize = 1 << 19;

/// How many parallel parts `work` element visits justify; 1 means inline.
pub fn parallel_parts(work: usize) -> usize { (work / PARALLEL_GRAIN).clamp(1, rayon::current_num_threads().max(1)) }

/// [`add_reduce`] of a `(pool, rest)` plane over its leading axis without a
/// mask, parallel over columns where that pays: every column still adds its
/// rows in order, so the result is the same bits.
pub fn leading_sum<T: Widen>(values: &[T], pool: usize, rest: usize, buffered: bool) -> Vec<f64> {
    let parts = parallel_parts(pool * rest);
    if rest == 1 || pool <= 1 || parts == 1 {
        return add_reduce(values, &[pool, rest], &[0], None, buffered);
    }
    use rayon::prelude::*;
    let mut out = vec![0.0_f64; rest];
    let chunk = rest.div_ceil(parts * 2).max(1024);
    out.par_chunks_mut(chunk).enumerate().for_each(|(index, part)| {
        let first = index * chunk;
        let width = part.len();
        for row in 0..pool {
            let base = row * rest + first;
            for (slot, value) in part.iter_mut().zip(&values[base..base + width]) { *slot += value.widen(); }
        }
    });
    out
}

// ------------------------------------------------------------------ extremes

/// How numpy 2.4.2's minimum/maximum loops pick between operands on this
/// CPU (AVX2). It only shows between +0.0 and -0.0 and in which NaN comes
/// out: float64/float32 run `npyv_minn` on 4/8-lane vectors and in the
/// scalar tail alike (a NaN first operand sticks, otherwise ties keep the
/// SECOND operand), and a horizontal vector reduction with a NaN lane gives
/// the canonical NaN; float16 runs a scalar loop whose ties keep the FIRST
/// operand. Integers cannot tell equal operands apart. Verified on 12000
/// contiguous, 3200 reduceat and 9256 multi-axis/masked/initial cases
/// (f64/f32) and 4872 float16 cases.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Extremes { Lanes(usize), Half }

impl Extremes {
    /// The loop numpy runs for a dtype of this kind and size.
    pub fn of(kind: char, itemsize: usize) -> Extremes {
        if kind == 'f' && itemsize == 2 { Extremes::Half } else { Extremes::Lanes(32 / itemsize) }
    }

    #[inline(always)]
    pub fn step(self, a: f64, b: f64, minimum: bool) -> f64 {
        if a.is_nan() { return a; }
        let keep = match self {
            Extremes::Half => if minimum { a <= b } else { a >= b },
            Extremes::Lanes(_) => if minimum { a < b } else { a > b },
        };
        if keep { a } else { b }
    }
}

/// numpy's contiguous reduce loop (`simd_reduce_c_minn`) folding `run` into
/// the output's current value `op1`: eight vectors per step combined as a
/// tree into the accumulator vector, then single vectors, the horizontal
/// halving tree, then the scalar tail.
pub fn reduce_contiguous<T: Widen>(op1: f64, run: &[T], rule: Extremes, minimum: bool) -> f64 {
    let step = |a: f64, b: f64| rule.step(a, b, minimum);
    let Extremes::Lanes(vstep) = rule else { return run.iter().fold(op1, |a, v| step(a, v.widen())) };
    let n = run.len();
    if n == 0 { return op1; }
    let mut acc = [op1; 32];
    let mut i = 0;
    while n - i >= vstep * 8 {
        for (lane, slot) in acc.iter_mut().enumerate().take(vstep) {
            let v = |j: usize| run[i + vstep * j + lane].widen();
            let (r01, r23, r45, r67) = (step(v(0), v(1)), step(v(2), v(3)), step(v(4), v(5)), step(v(6), v(7)));
            *slot = step(*slot, step(step(r01, r23), step(r45, r67)));
        }
        i += vstep * 8;
    }
    while n - i >= vstep {
        for (lane, slot) in acc.iter_mut().enumerate().take(vstep) { *slot = step(*slot, run[i + lane].widen()); }
        i += vstep;
    }
    let mut result = if acc[..vstep].iter().any(|a| a.is_nan()) { f64::from_bits(0x7ff8_0000_0000_0000) } else {
        let mut width = vstep;
        while width > 1 {
            width /= 2;
            for k in 0..width { acc[k] = step(acc[k], acc[k + width]); }
        }
        acc[0]
    };
    while i < n { result = step(result, run[i].widen()); i += 1; }
    result
}

/// `np.min/np.max(values.reshape(shape), axis=axes[, where=mask, initial=])`
/// over a C-contiguous input, widened to float64 (exact: widening keeps
/// order, signs and NaN). Without `initial` each output starts as the first
/// element along the reduced axes and the loop skips it (numpy's
/// first-visit rule); a masked reduction always has an `initial`.
pub fn extreme_reduce<T: Widen>(values: &[T], shape: &[usize], axes: &[usize], mask: Option<&[bool]>, initial: Option<f64>,
                                rule: Extremes, minimum: bool) -> Vec<f64> {
    let (dims, kept) = coalesced(shape, axes);
    let mut out = match initial {
        Some(value) => vec![value; kept],
        None => {
            let ndim = shape.len();
            let mut strides = vec![1usize; ndim];
            for axis in (0..ndim.saturating_sub(1)).rev() { strides[axis] = strides[axis + 1] * shape[axis + 1]; }
            let retained: Vec<usize> = (0..ndim).filter(|a| !axes.contains(a)).collect();
            let mut index = vec![0usize; retained.len()];
            (0..kept).map(|_| {
                let position: usize = retained.iter().zip(&index).map(|(a, i)| strides[*a] * i).sum();
                for k in (0..retained.len()).rev() { index[k] += 1; if index[k] < shape[retained[k]] { break; } index[k] = 0; }
                values[position].widen()
            }).collect()
        }
    };
    let inner = *dims.last().unwrap();
    let outer = &dims[..dims.len() - 1];
    let total: usize = outer.iter().map(|d| d.n).product();
    let mut index = vec![0usize; outer.len()];
    let (mut ib, mut ob) = (0usize, 0usize);
    for _ in 0..total {
        let first = initial.is_none() && outer.iter().zip(&index).all(|(d, i)| d.output != 0 || *i == 0);
        if inner.output == 0 {
            let low = ib + usize::from(first);
            let high = ib + inner.n;
            match mask {
                None => out[ob] = reduce_contiguous(out[ob], &values[low..high], rule, minimum),
                Some(mask) => {
                    let mut j = low;
                    while j < high {
                        while j < high && !mask[j] { j += 1; }
                        let mut k = j;
                        while k < high && mask[k] { k += 1; }
                        if k > j { out[ob] = reduce_contiguous(out[ob], &values[j..k], rule, minimum); }
                        j = k;
                    }
                }
            }
        } else if !first {
            for j in 0..inner.n {
                if mask.is_some_and(|m| !m[ib + j]) { continue; }
                let slot = &mut out[ob + j * inner.output];
                *slot = rule.step(*slot, values[ib + j].widen(), minimum);
            }
        }
        for axis in (0..outer.len()).rev() {
            index[axis] += 1;
            ib += outer[axis].input;
            ob += outer[axis].output;
            if index[axis] < outer[axis].n { break; }
            ib -= outer[axis].input * outer[axis].n;
            ob -= outer[axis].output * outer[axis].n;
            index[axis] = 0;
        }
    }
    out
}

/// `np.minimum/np.maximum.reduceat(values, starts)` widened to float64:
/// every segment reduces from its own first element.
pub fn extreme_reduceat<T: Widen>(values: &[T], starts: &[usize], rule: Extremes, minimum: bool) -> Vec<f64> {
    starts.iter().enumerate().map(|(k, start)| {
        let stop = starts.get(k + 1).copied().unwrap_or(values.len());
        reduce_contiguous(values[*start].widen(), &values[start + 1..stop], rule, minimum)
    }).collect()
}

#[cfg(test)]
mod axis_tests {
    use super::*;

    #[test]
    fn extremes_follow_numpys_loops() {
        // Signed zeros (n = -0.0, p = +0.0, 1 = 1.0) and numpy 2.4.2's choice.
        let zeros = |pattern: &str| -> Vec<f64> { pattern.chars().map(|c| match c { 'n' => -0.0, 'p' => 0.0, _ => 1.0 }).collect() };
        let x = zeros("pp1npnpnpppnnp1n1pnppppnppppnnppppnpppnnppnnn");
        let rule = Extremes::of('f', 8);
        assert!(reduce_contiguous(x[0], &x[1..], rule, true).is_sign_negative());
        let x: Vec<f32> = zeros("nn1pnpnpppnnn1npppnnpnnppnnppnpnpppnpnpnpppnnnnpnppppppnnpnppnpnnnnppppnnppnp").into_iter().map(|v| v as f32).collect();
        assert!(reduce_contiguous(x[0].widen(), &x[1..], Extremes::of('f', 4), true).is_sign_positive());
        // A leading-axis reduction keeps the later zero; float16 the earlier.
        assert_eq!(extreme_reduce(&[-0.0_f64, 0.0, 0.0, -0.0], &[2, 2], &[0], None, None, rule, true).iter().map(|v| v.is_sign_negative()).collect::<Vec<_>>(), vec![false, true]);
        assert_eq!(extreme_reduce(&[-0.0_f64, 0.0, 0.0, -0.0], &[2, 2], &[0], None, None, Extremes::Half, true).iter().map(|v| v.is_sign_negative()).collect::<Vec<_>>(), vec![true, false]);
        // A NaN lane of the horizontal reduction comes out as the canonical NaN.
        let mut x = vec![1.0_f64; 9];
        x[3] = f64::from_bits(0xfff8_0000_0000_0000);
        assert_eq!(reduce_contiguous(x[0], &x[1..], rule, true).to_bits(), 0x7ff8_0000_0000_0000);
    }

    #[test]
    fn axis_reductions_follow_numpys_iteration() {
        let values: Vec<f64> = (1..=6).map(f64::from).collect();
        assert_eq!(add_reduce(&values, &[2, 3], &[0], None, false), vec![5.0, 7.0, 9.0]);
        assert_eq!(add_reduce(&values, &[2, 3], &[1], None, false), vec![6.0, 15.0]);
        assert_eq!(add_reduce(&values, &[2, 3], &[0, 1], None, false), vec![21.0]);
        let mask = [true, false, true, true, true, false];
        assert_eq!(add_reduce(&values, &[2, 3], &[1], Some(&mask), false), vec![4.0, 9.0]);
        let long: Vec<f32> = (0..20000).map(|i| (i as f32).sin() * 1e7).collect();
        let cast: Vec<f64> = long.iter().map(|v| f64::from(*v)).collect();
        assert_eq!(add_reduce(&long, &[20000], &[0], None, true)[0].to_bits(), sum_cast_f64(&long).to_bits());
        assert_eq!(add_reduce(&cast, &[20000], &[0], None, false)[0].to_bits(), sum_f64(&cast).to_bits());
    }
}

/// numpy's integer `add.reduce` with its default accumulator (int64 for
/// signed and bool inputs, uint64 for unsigned), wrapping on overflow, which
/// makes it order-free. Returned as the accumulator's bits in an i128.
pub fn int_add_reduce<T: Copy>(values: &[T], shape: &[usize], axes: &[usize], mask: Option<&[bool]>, as_int: impl Fn(T) -> i128) -> Vec<i128> {
    let (dims, kept) = coalesced(shape, axes);
    let mut out = vec![0i128; kept];
    let total: usize = dims.iter().map(|d| d.n).product();
    let mut index = vec![0usize; dims.len()];
    let (mut ib, mut ob) = (0usize, 0usize);
    for _ in 0..total {
        if mask.is_none_or(|m| m[ib]) { out[ob] = out[ob].wrapping_add(as_int(values[ib])); }
        for axis in (0..dims.len()).rev() {
            index[axis] += 1;
            ib += dims[axis].input;
            ob += dims[axis].output;
            if index[axis] < dims[axis].n { break; }
            ib -= dims[axis].input * dims[axis].n;
            ob -= dims[axis].output * dims[axis].n;
            index[axis] = 0;
        }
    }
    out
}

/// `np.dot(x, x)` of a contiguous float64 vector as numpy's bundled OpenBLAS
/// computes it on an AVX2 (x86-64-v3) CPU: eight two-lane FMA accumulators
/// per 16 elements, combined pairwise, then the tail added in order.
/// Verified bit-equal on 239 lengths up to 3e5 (projection/AGREEMENT.md).
pub fn blas_dot_self(values: &[f64]) -> f64 {
    let n = values.len();
    let whole = n & !15;
    let mut acc = [[0.0_f64; 2]; 8];
    let mut i = 0;
    while i < whole {
        for (a, lanes) in acc.iter_mut().enumerate() {
            for (l, slot) in lanes.iter_mut().enumerate() {
                let v = values[i + 2 * a + l];
                *slot = v.mul_add(v, *slot);
            }
        }
        i += 16;
    }
    let pair = |x: [f64; 2], y: [f64; 2]| [x[0] + y[0], x[1] + y[1]];
    let quads = [pair(acc[0], acc[1]), pair(acc[2], acc[3]), pair(acc[4], acc[5]), pair(acc[6], acc[7])];
    let halves = [pair(quads[0], quads[1]), pair(quads[2], quads[3])];
    let total = pair(halves[0], halves[1]);
    let mut dot = if whole > 0 { total[0] + total[1] } else { 0.0 };
    for v in &values[whole..] { dot += v * v; }
    dot
}
