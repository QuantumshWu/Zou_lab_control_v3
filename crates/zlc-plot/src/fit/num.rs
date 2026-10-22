//! Scalar semantics the oracle's numerics depend on, written once.
//!
//! * numba compiles with `error_model="python"`: a float division by zero
//!   RAISES `ZeroDivisionError` instead of returning an infinity.  Every
//!   division ported from a numba kernel goes through [`div`], and the error
//!   travels exactly where the exception did.
//! * Python's (and numba's) `max(a, b)` is `b if b > a else a` -- not IEEE
//!   maxNum: zeros keep the first argument's sign and a NaN is kept or
//!   dropped by position.  [`pmax`] / [`pmin`] are that rule.
//! * libm: the UCRT entry points numba, CPython and NumPy all call.
//! * numba's own `np.sort` / `np.argsort` (an unstable quicksort) and
//!   sequential reductions; NumPy's pairwise `np.sum` is `crate::numeric`.

/// What a ported numba kernel raises: a division by zero (`ZeroDivisionError`)
/// or a table larger than [`ALLOCATION_LIMIT`] (`MemoryError`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelError { ZeroDivision, Allocation }

impl std::fmt::Display for KernelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KernelError::ZeroDivision => f.write_str("division by zero"),
            // numba's runtime, when an array cannot be allocated.
            KernelError::Allocation => f.write_str("Allocation failed (probably too large)."),
        }
    }
}

pub type Zr<T> = Result<T, KernelError>;

/// The most elements a kernel table may hold (128 MiB of f64). The oracle
/// sizes some tables from the data alone -- the Poisson grid grows with the
/// square root of the rate -- and allocates whatever that asks, tens of
/// gigabytes for a histogram of large values; a kernel refuses instead,
/// failing the fit as an allocation numba cannot make fails it.
pub const ALLOCATION_LIMIT: f64 = (1u64 << 24) as f64;

/// `a / b` under numba's Python error model.
#[inline(always)]
pub fn div(a: f64, b: f64) -> Zr<f64> {
    if b == 0.0 { Err(KernelError::ZeroDivision) } else { Ok(a / b) }
}

/// Python's `max(a, b)` for two floats.
#[inline(always)]
pub fn pmax(a: f64, b: f64) -> f64 { if b > a { b } else { a } }

/// Python's `min(a, b)` for two floats.
#[inline(always)]
pub fn pmin(a: f64, b: f64) -> f64 { if b < a { b } else { a } }

pub const EPSILON: f64 = f64::EPSILON;
pub const TINY: f64 = f64::MIN_POSITIVE;
pub const PI: f64 = std::f64::consts::PI;

/// C `nextafter`, which `np.nextafter` is.
pub fn nextafter(x: f64, y: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return x + y;
    }
    if x == y {
        return y;
    }
    if x == 0.0 {
        return if y > 0.0 { f64::from_bits(1) } else { -f64::from_bits(1) };
    }
    let bits = x.to_bits();
    let away = (y > x) == (x > 0.0);
    f64::from_bits(if away { bits + 1 } else { bits - 1 })
}

/// numba's `int(float)` on x86-64: truncation, and the `cvttsd2si`
/// "integer indefinite" value for anything out of range or NaN.
#[inline]
pub fn numba_int(x: f64) -> i64 {
    if x.is_nan() || x >= 9.223_372_036_854_775_807e18 || x < -9.223_372_036_854_775_808e18 {
        i64::MIN
    } else {
        x as i64
    }
}

unsafe extern "C" {
    #[link_name = "lgamma"]
    fn c_lgamma(x: f64) -> f64;
    #[link_name = "erf"]
    fn c_erf(x: f64) -> f64;
    #[link_name = "pow"]
    fn c_pow(x: f64, y: f64) -> f64;
}

/// UCRT `lgamma` -- what numba's `math.lgamma` calls.
#[inline]
pub fn lgamma(x: f64) -> f64 { unsafe { c_lgamma(x) } }

/// UCRT `erf` -- what CPython 3.13's `math.erf` and numba call.
#[inline]
pub fn erf(x: f64) -> f64 { unsafe { c_erf(x) } }

/// UCRT `pow` -- CPython's `float ** float` and NumPy's general power.
#[inline]
pub fn pow(x: f64, y: f64) -> f64 { unsafe { c_pow(x, y) } }

/// numba's `lt_floats`: NaNs sort last.
#[inline(always)]
fn lt_floats(a: f64, b: f64) -> bool { a < b || (b.is_nan() && !a.is_nan()) }

const SMALL_QUICKSORT: isize = 15;

/// numba's quicksort over an index permutation `r` of `keys`
/// (`numba.misc.quicksort.make_quicksort_impl`, iterative form).
fn numba_quicksort_by(keys: &[f64], r: &mut [usize]) {
    let n = r.len() as isize;
    if n < 2 {
        return;
    }
    let get = |r: &[usize], i: isize| keys[r[i as usize]];
    let mut stack: Vec<(isize, isize)> = Vec::with_capacity(64);
    stack.push((0, n - 1));
    while let Some((mut low, mut high)) = stack.pop() {
        while high - low >= SMALL_QUICKSORT {
            // partition
            let mid = (low + high) >> 1;
            if lt_floats(get(r, mid), get(r, low)) {
                r.swap(low as usize, mid as usize);
            }
            if lt_floats(get(r, high), get(r, mid)) {
                r.swap(high as usize, mid as usize);
            }
            if lt_floats(get(r, mid), get(r, low)) {
                r.swap(low as usize, mid as usize);
            }
            let pivot = get(r, mid);
            r.swap(high as usize, mid as usize);
            let mut i = low;
            let mut j = high - 1;
            loop {
                while i < high && lt_floats(get(r, i), pivot) {
                    i += 1;
                }
                while j >= low && lt_floats(pivot, get(r, j)) {
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                r.swap(i as usize, j as usize);
                i += 1;
                j -= 1;
            }
            r.swap(i as usize, high as usize);
            if high - i > i - low {
                if high > i {
                    stack.push((i + 1, high));
                }
                high = i - 1;
            } else {
                if i > low {
                    stack.push((low, i - 1));
                }
                low = i + 1;
            }
        }
        // insertion sort r[low..=high]
        if high > low {
            for i in (low + 1)..=high {
                let k = r[i as usize];
                let v = keys[k];
                let mut j = i;
                while j > low && lt_floats(v, get(r, j - 1)) {
                    r[j as usize] = r[(j - 1) as usize];
                    j -= 1;
                }
                r[j as usize] = k;
            }
        }
    }
}

/// numba's `np.argsort(values)` (quicksort; ties land where it puts them).
pub fn numba_argsort(values: &[f64]) -> Vec<usize> {
    let mut r: Vec<usize> = (0..values.len()).collect();
    numba_quicksort_by(values, &mut r);
    r
}

/// numba's `np.sort(values)`: the same quicksort moving values, so equal
/// keys that differ in bits (0.0 and -0.0) keep its order.
pub fn numba_sort(values: &[f64]) -> Vec<f64> {
    // Sorting values in place by the same comparisons is sorting the index
    // permutation of a copy and gathering: the algorithm only ever reads
    // keys through the permutation.
    let mut keys = values.to_vec();
    numba_quicksort_values(&mut keys);
    keys
}

fn numba_quicksort_values(a: &mut [f64]) {
    let n = a.len() as isize;
    if n < 2 {
        return;
    }
    let mut stack: Vec<(isize, isize)> = Vec::with_capacity(64);
    stack.push((0, n - 1));
    while let Some((mut low, mut high)) = stack.pop() {
        while high - low >= SMALL_QUICKSORT {
            let mid = (low + high) >> 1;
            if lt_floats(a[mid as usize], a[low as usize]) {
                a.swap(low as usize, mid as usize);
            }
            if lt_floats(a[high as usize], a[mid as usize]) {
                a.swap(high as usize, mid as usize);
            }
            if lt_floats(a[mid as usize], a[low as usize]) {
                a.swap(low as usize, mid as usize);
            }
            let pivot = a[mid as usize];
            a.swap(high as usize, mid as usize);
            let mut i = low;
            let mut j = high - 1;
            loop {
                while i < high && lt_floats(a[i as usize], pivot) {
                    i += 1;
                }
                while j >= low && lt_floats(pivot, a[j as usize]) {
                    j -= 1;
                }
                if i >= j {
                    break;
                }
                a.swap(i as usize, j as usize);
                i += 1;
                j -= 1;
            }
            a.swap(i as usize, high as usize);
            if high - i > i - low {
                if high > i {
                    stack.push((i + 1, high));
                }
                high = i - 1;
            } else {
                if i > low {
                    stack.push((low, i - 1));
                }
                low = i + 1;
            }
        }
        if high > low {
            for i in (low + 1)..=high {
                let v = a[i as usize];
                let mut j = i;
                while j > low && lt_floats(v, a[(j - 1) as usize]) {
                    a[j as usize] = a[(j - 1) as usize];
                    j -= 1;
                }
                a[j as usize] = v;
            }
        }
    }
}

/// `_fit_compiled._median`: numba sort, then the middle (pair).
///
/// The middle order statistics are selected, not sorted for: their VALUES
/// are the same under any correct ordering -- except a zero, whose sign
/// depends on where the quicksort leaves -0.0 against 0.0, so a zero in
/// the middle is answered by the full numba sort.
pub fn numba_median(values: &[f64]) -> f64 {
    let count = values.len();
    if count == 0 {
        return f64::NAN;
    }
    let mut work = super::scratch::Scratch::copied(values);
    let k = count / 2;
    let (left, middle, _) = work.select_nth_unstable_by(k, numba_order);
    let upper = *middle;
    let lower = if count & 1 == 1 { upper } else { left.iter().copied().fold(f64::NEG_INFINITY, |a, b| if numba_order(&b, &a).is_gt() { b } else { a }) };
    if upper == 0.0 || lower == 0.0 {
        let ordered = numba_sort(values);
        if count & 1 == 1 {
            return ordered[count / 2];
        }
        return 0.5 * (ordered[count / 2 - 1] + ordered[count / 2]);
    }
    if count & 1 == 1 { upper } else { 0.5 * (lower + upper) }
}

/// The order `lt_floats` sorts by (NaNs last; zeros equal).
pub fn numba_order(a: &f64, b: &f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match a.partial_cmp(b) {
        Some(order) => order,
        None => match (a.is_nan(), b.is_nan()) {
            (true, true) => Ordering::Equal,
            (true, false) => Ordering::Greater,
            _ => Ordering::Less,
        },
    }
}

/// numba's `np.mean` / `np.sum`: one sequential accumulator.
pub fn sequential_sum(values: &[f64]) -> f64 {
    let mut total = 0.0;
    for &v in values {
        total += v;
    }
    total
}

pub fn numba_mean(values: &[f64]) -> f64 { sequential_sum(values) / values.len() as f64 }

pub use crate::numeric::{mean_f64 as numpy_mean, sum_f64 as numpy_sum};

/// `np.median` of finite values: the middle order statistic(s); an even
/// count takes the NumPy mean of the middle pair, `(a + b) / 2`.
pub fn numpy_median(values: &[f64]) -> f64 {
    let mut ordered = values.to_vec();
    ordered.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = ordered.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 {
        ordered[n / 2]
    } else {
        (ordered[n / 2 - 1] + ordered[n / 2]) / 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nextafter_matches_c() {
        assert_eq!(nextafter(0.0, 1.0).to_bits(), 1);
        assert_eq!(nextafter(1.0, 2.0), 1.0 + f64::EPSILON);
        assert_eq!(nextafter(1.0, 0.0), 1.0 - f64::EPSILON / 2.0);
        assert_eq!(nextafter(-1.0, 0.0), -1.0 + f64::EPSILON / 2.0);
        assert_eq!(nextafter(f64::INFINITY, 0.0), f64::MAX);
    }

    #[test]
    fn quicksort_orders() {
        let values: Vec<f64> = (0..200).map(|i| ((i * 7919) % 211) as f64 - 50.0).collect();
        let order = numba_argsort(&values);
        for w in order.windows(2) {
            assert!(values[w[0]] <= values[w[1]]);
        }
        let sorted = numba_sort(&values);
        for w in sorted.windows(2) {
            assert!(w[0] <= w[1]);
        }
    }
}

/// Python's `repr(str)`.
pub fn str_repr(text: &str) -> String { crate::pystr::repr(text) }

/// Python's `repr` of a list of strings (`['a', 'b']`).
pub fn str_list_repr<S: AsRef<str>>(items: &[S]) -> String {
    let parts: Vec<String> = items.iter().map(|s| str_repr(s.as_ref())).collect();
    format!("[{}]", parts.join(", "))
}

/// CPython's `math.fsum` (Shewchuk partials, half-even across partials).
pub fn fsum(values: &[f64]) -> f64 {
    let mut partials: Vec<f64> = Vec::new();
    let mut special_sum = 0.0;
    let mut inf_sum = 0.0;
    let mut lo = 0.0;
    for &item in values {
        let mut x = item;
        let xsave = x;
        let mut i = 0;
        for j in 0..partials.len() {
            let mut y = partials[j];
            if x.abs() < y.abs() {
                std::mem::swap(&mut x, &mut y);
            }
            let hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                partials[i] = lo;
                i += 1;
            }
            x = hi;
        }
        partials.truncate(i);
        if x != 0.0 {
            if !x.is_finite() {
                if xsave.is_infinite() {
                    inf_sum += xsave;
                }
                special_sum += xsave;
                partials.clear();
            } else {
                partials.push(x);
            }
        }
    }
    if special_sum != 0.0 {
        let _ = inf_sum;
        return special_sum;
    }
    let mut n = partials.len();
    let mut hi = 0.0;
    if n > 0 {
        n -= 1;
        hi = partials[n];
        while n > 0 {
            let x = hi;
            n -= 1;
            let y = partials[n];
            hi = x + y;
            let yr = hi - x;
            lo = y - yr;
            if lo != 0.0 {
                break;
            }
        }
        if n > 0 && ((lo < 0.0 && partials[n - 1] < 0.0) || (lo > 0.0 && partials[n - 1] > 0.0)) {
            let y = lo * 2.0;
            let x = hi + y;
            let yr = x - hi;
            if y == yr {
                hi = x;
            }
        }
    }
    hi
}
