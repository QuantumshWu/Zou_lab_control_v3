//! CPython and x86 numeric semantics the height-bar geometry depends on.
//!
//! Python's builtins answer differently from Rust's: `max`/`min` keep the
//! first argument on ties, `%`/`//` floor towards minus infinity with
//! CPython's own snapping, and `math.hypot` is a compensated vector norm, not
//! the C library's hypot. numpy and numba cast floats to integers with the
//! x86 truncating convert, whose out-of-range answer is `i64::MIN`, where Rust
//! saturates. Each is written once here.

/// `math.radians`: one multiply by CPython's rounded π/180.
pub fn radians(degrees: f64) -> f64 {
    const DEG_TO_RAD: f64 = std::f64::consts::PI / 180.0;
    degrees * DEG_TO_RAD
}

/// Python `max(a, b)`: `b` replaces `a` only when strictly greater.
#[inline]
pub fn py_max(a: f64, b: f64) -> f64 { if b > a { b } else { a } }

/// Python `min(a, b)`: `b` replaces `a` only when strictly smaller.
#[inline]
pub fn py_min(a: f64, b: f64) -> f64 { if b < a { b } else { a } }

/// numpy `maximum`/`minimum` on finite data (NaN propagates from either side).
#[inline]
pub fn np_maximum(a: f64, b: f64) -> f64 { if a >= b || a.is_nan() { a } else { b } }
#[inline]
pub fn np_minimum(a: f64, b: f64) -> f64 { if a <= b || a.is_nan() { a } else { b } }

/// numpy `clip(x, low, high)` = minimum(maximum(x, low), high).
#[inline]
pub fn np_clip(x: f64, low: f64, high: f64) -> f64 { np_minimum(np_maximum(x, low), high) }

/// Python float `%`.
pub fn py_mod(vx: f64, wx: f64) -> f64 {
    let mut modulo = vx % wx;
    if modulo != 0.0 {
        if (wx < 0.0) != (modulo < 0.0) {
            modulo += wx;
        }
    } else {
        modulo = 0.0f64.copysign(wx);
    }
    modulo
}

/// Python float `//` (CPython `_float_div_mod`).
pub fn py_floordiv(vx: f64, wx: f64) -> f64 {
    let modulo = vx % wx;
    let mut div = (vx - modulo) / wx;
    if modulo != 0.0 && (wx < 0.0) != (modulo < 0.0) {
        div -= 1.0;
    }
    if div != 0.0 {
        let mut floordiv = div.floor();
        if div - floordiv > 0.5 {
            floordiv += 1.0;
        }
        floordiv
    } else {
        0.0f64.copysign(vx / wx)
    }
}

/// `frexp`: x = m * 2^e with |m| in [0.5, 1) for finite nonzero x.
fn frexp_exponent(x: f64) -> i32 {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // subnormal: normalise through an exact scaling
        let scaled = x * f64::from_bits(0x4350_0000_0000_0000); // 2^54
        return frexp_exponent(scaled) - 54;
    }
    biased - 1022
}

/// 2^k, exact, for k in the representable range.
fn pow2(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

#[inline]
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    (z, x.mul_add(y, -z))
}

#[inline]
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    (x, (a - x) + b)
}

fn vector_norm(vec: &mut [f64], max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    if max == 0.0 || vec.len() <= 1 {
        return max;
    }
    let max_e = frexp_exponent(max);
    if max_e < -1023 {
        for value in vec.iter_mut() {
            *value /= f64::MIN_POSITIVE;
        }
        return f64::MIN_POSITIVE * vector_norm(vec, max / f64::MIN_POSITIVE, found_nan);
    }
    let scale = pow2(-max_e);
    let (mut csum, mut frac1, mut frac2) = (1.0f64, 0.0f64, 0.0f64);
    for &value in vec.iter() {
        let x = value * scale;
        let pr = dl_mul(x, x);
        let sm = dl_fast_sum(csum, pr.0);
        csum = sm.0;
        frac1 += pr.1;
        frac2 += sm.1;
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let pr = dl_mul(-h, h);
    let sm = dl_fast_sum(csum, pr.0);
    csum = sm.0;
    frac1 += pr.1;
    frac2 += sm.1;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h);
    h / scale
}

/// Python `math.hypot(x, y)` (CPython 3.13 `vector_norm`).
pub fn py_hypot(x: f64, y: f64) -> f64 {
    let mut coordinates = [x.abs(), y.abs()];
    let mut max = 0.0f64;
    let mut found_nan = false;
    for &value in &coordinates {
        found_nan |= value.is_nan();
        if value > max {
            max = value;
        }
    }
    vector_norm(&mut coordinates, max, found_nan)
}

/// numpy `hypot`: the C library's.
#[inline]
pub fn np_hypot(x: f64, y: f64) -> f64 { x.hypot(y) }

/// numpy `astype(int64)` / numba `np.int64(x)`: the x86 truncating convert.
#[inline]
pub fn trunc_i64(x: f64) -> i64 {
    if x.is_nan() || x >= 9.223_372_036_854_775_808e18 || x < -9.223_372_036_854_775_808e18 {
        i64::MIN
    } else {
        x as i64
    }
}

/// Python `round(x)` for a float: ties to even, then int.
#[inline]
pub fn py_round(x: f64) -> f64 { x.round_ties_even() }
