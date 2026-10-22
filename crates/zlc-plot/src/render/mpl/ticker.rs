//! Matplotlib's default tick policy for an axis zlc never gave its own:
//! `AutoLocator` (MaxNLocator, nbins 'auto', steps [1, 2, 2.5, 5, 10]) and
//! `ScalarFormatter` with the default rc (offset on, threshold 4, power limits
//! (-5, 6), Unicode minus). Such an axis draws no visible label in zlc's
//! figures, but its locations, labels and offset are part of the figure.

/// Python's `divmod(x, y)` for floats (numpy's floor_divide/remainder agree).
pub fn py_divmod(x: f64, y: f64) -> (f64, f64) {
    let mut m = x % y;
    let mut div = (x - m) / y;
    if m != 0.0 {
        if (y < 0.0) != (m < 0.0) { m += y; div -= 1.0; }
    } else {
        m = 0.0f64.copysign(y);
    }
    let floordiv = if div != 0.0 {
        let mut f = div.floor();
        if div - f > 0.5 { f += 1.0; }
        f
    } else {
        0.0f64.copysign(x / y)
    };
    (floordiv, m)
}

pub fn floor_div(x: f64, y: f64) -> f64 { py_divmod(x, y).0 }

/// `mtransforms.nonsingular(vmin, vmax, expander, tiny, increasing=True)`.
pub fn nonsingular(v0: f64, v1: f64, expander: f64, tiny: f64) -> (f64, f64) {
    if !v0.is_finite() || !v1.is_finite() { return (-expander, expander); }
    let (mut vmin, mut vmax) = if v1 < v0 { (v1, v0) } else { (v0, v1) };
    let maxabsvalue = if vmax.abs() > vmin.abs() { vmax.abs() } else { vmin.abs() };
    if maxabsvalue < (1e6 / tiny) * f64::MIN_POSITIVE {
        vmin = -expander;
        vmax = expander;
    } else if vmax - vmin <= maxabsvalue * tiny {
        if vmax == 0.0 && vmin == 0.0 { vmin = -expander; vmax = expander; }
        else { vmin -= expander * vmin.abs(); vmax += expander * vmax.abs(); }
    }
    (vmin, vmax)
}

/// `scale_range(vmin, vmax, n)`.
fn scale_range(vmin: f64, vmax: f64, n: f64) -> (f64, f64) {
    let dv = (vmax - vmin).abs();
    let meanv = (vmax + vmin) / 2.0;
    let offset = if meanv.abs() / dv < 100.0 { 0.0 } else { 10f64.powf(floor_div(meanv.abs().log10(), 1.0)).copysign(meanv) };
    let scale = 10f64.powf(floor_div((dv / n).log10(), 1.0));
    (scale, offset)
}

struct EdgeInteger { step: f64, offset: f64 }

impl EdgeInteger {
    fn closeto(&self, ms: f64, edge: f64) -> bool {
        let tol = if self.offset > 0.0 {
            let digits = (self.offset / self.step).log10();
            let tol = 1e-10f64.max(10f64.powf(digits - 12.0));
            0.4999f64.min(tol)
        } else { 1e-10 };
        (ms - edge).abs() < tol
    }
    fn le(&self, x: f64) -> f64 {
        let (d, m) = py_divmod(x, self.step);
        if self.closeto(m / self.step, 1.0) { d + 1.0 } else { d }
    }
    fn ge(&self, x: f64) -> f64 {
        let (d, m) = py_divmod(x, self.step);
        if self.closeto(m / self.step, 0.0) { d } else { d + 1.0 }
    }
}

/// `AutoLocator().tick_values(vmin, vmax)` with `get_tick_space()` = `tick_space`.
pub fn auto_ticks(vmin: f64, vmax: f64, tick_space: i64) -> Vec<f64> {
    let (vmin, vmax) = nonsingular(vmin, vmax, 1e-13, 1e-14);
    raw_ticks(vmin, vmax, tick_space.clamp(1, 9) as f64, &[0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0, 20.0])
}

/// `MaxNLocator(nbins, prune="both" when asked).tick_values(vmin, vmax)` with
/// the default steps `[1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10]`.
pub fn max_n_ticks(vmin: f64, vmax: f64, nbins: usize, prune_both: bool) -> Vec<f64> {
    const STEPS: [f64; 10] = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    // `_staircase(steps)`: 0.1 * steps[:-1], steps, 10 * steps[1].
    let staircase: Vec<f64> = STEPS[..9].iter().map(|s| 0.1 * s).chain(STEPS).chain([10.0 * STEPS[1]]).collect();
    let (vmin, vmax) = nonsingular(vmin, vmax, 1e-13, 1e-14);
    let ticks = raw_ticks(vmin, vmax, nbins as f64, &staircase);
    match (prune_both, ticks.len()) {
        (false, _) => ticks,
        (true, n) if n > 2 => ticks[1..n - 1].to_vec(),
        (true, _) => Vec::new(),
    }
}

/// `MaxNLocator._raw_ticks(vmin, vmax)` over the extended staircase (`min_n_ticks` 2).
fn raw_ticks(vmin: f64, vmax: f64, nbins: f64, staircase: &[f64]) -> Vec<f64> {
    let (scale, offset) = scale_range(vmin, vmax, nbins);
    let (vmin_, vmax_) = (vmin - offset, vmax - offset);
    let steps: Vec<f64> = staircase.iter().map(|s| s * scale).collect();
    let raw_step = (vmax_ - vmin_) / nbins;
    let istep = steps.iter().position(|s| *s >= raw_step).unwrap_or(steps.len() - 1);
    let mut ticks = Vec::new();
    for &step in steps[..=istep].iter().rev() {
        let best_vmin = floor_div(vmin_, step) * step;
        let edge = EdgeInteger { step, offset: offset.abs() };
        let low = edge.le(vmin_ - best_vmin);
        let high = edge.ge(vmax_ - best_vmin);
        let count = (high + 1.0 - low).ceil().max(0.0) as usize;
        ticks = (0..count).map(|i| (low + i as f64) * step + best_vmin).collect();
        let nticks = ticks.iter().filter(|t| **t <= vmax_ && **t >= vmin_).count();
        if nticks >= 2 { break; }
    }
    ticks.iter().map(|t| t + offset).collect()
}

/// `get_tick_space()`: estimated number of ticks that fit (length in points).
pub fn tick_space(length_pt: f64, labelsize: f64, x: bool) -> i64 {
    let size = labelsize * if x { 3.0 } else { 2.0 };
    if size > 0.0 { (length_pt / size).floor() as i64 } else { (1i64 << 31) - 1 }
}

fn power_of_ten(n: i32) -> f64 {
    const P10: [f64; 9] = [1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8];
    if n < 9 { return P10[n as usize]; }
    let mut ret = 1e9;
    let mut k = n;
    while k > 9 { ret *= 10.0; k -= 1; }
    ret
}

/// `np.round(x, decimals)` for decimals >= 0.
fn np_round(x: f64, decimals: i32) -> f64 {
    let f = power_of_ten(decimals);
    (x * f).round_ties_even() / f
}

fn fix_minus(s: &str) -> String { s.replace('-', "\u{2212}") }

/// `ScalarFormatter` state after `set_locs`.
#[derive(Clone, Debug, Default)]
pub struct ScalarFormatter { pub offset: f64, pub order_of_magnitude: i32, pub sigfigs: usize, pub locs: Vec<f64> }

impl ScalarFormatter {
    /// `set_locs(locs)` with the axis' view interval.
    pub fn set_locs(locs: &[f64], view: (f64, f64)) -> ScalarFormatter {
        let mut f = ScalarFormatter { locs: locs.to_vec(), ..Default::default() };
        if locs.is_empty() { return f; }
        let (vmin, vmax) = if view.0 <= view.1 { view } else { (view.1, view.0) };
        let visible: Vec<f64> = locs.iter().copied().filter(|l| vmin <= *l && *l <= vmax).collect();
        f.offset = compute_offset(&visible);
        // _set_order_of_magnitude
        let abs: Vec<f64> = visible.iter().map(|v| v.abs()).collect();
        f.order_of_magnitude = if abs.is_empty() { 0 } else {
            let oom = if f.offset != 0.0 { (vmax - vmin).log10().floor() as i32 } else {
                let val = abs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                if val == 0.0 { 0 } else { val.log10().floor() as i32 }
            };
            if oom <= -5 || oom >= 6 { oom } else { 0 }
        };
        // _set_format
        let mut extended = locs.to_vec();
        if locs.len() < 2 { extended.push(view.0); extended.push(view.1); }
        let divisor = 10f64.powf(f.order_of_magnitude as f64);
        let mut scaled: Vec<f64> = extended.iter().map(|l| (l - f.offset) / divisor).collect();
        let max = scaled.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = scaled.iter().copied().fold(f64::INFINITY, f64::min);
        let mut loc_range = max - min;
        if loc_range == 0.0 { loc_range = scaled.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max); }
        if loc_range == 0.0 { loc_range = 1.0; }
        if locs.len() < 2 { scaled.truncate(scaled.len() - 2); }
        let loc_range_oom = loc_range.log10().floor() as i32;
        let mut sigfigs = (3 - loc_range_oom).max(0);
        let thresh = 1e-3 * 10f64.powf(loc_range_oom as f64);
        while sigfigs >= 0 {
            let worst = scaled.iter().map(|v| (v - np_round(*v, sigfigs)).abs()).fold(f64::NEG_INFINITY, f64::max);
            if worst < thresh { sigfigs -= 1; } else { break; }
        }
        f.sigfigs = (sigfigs + 1) as usize;
        f
    }

    /// `__call__(x)`.
    pub fn format(&self, x: f64) -> String {
        if self.locs.is_empty() { return String::new(); }
        let mut xp = (x - self.offset) / 10f64.powf(self.order_of_magnitude as f64);
        if xp.abs() < 1e-8 { xp = 0.0; }
        fix_minus(&format!("{:.*}", self.sigfigs, xp))
    }

    fn format_data(value: f64) -> String {
        let e = value.abs().log10().floor() as i32;
        let s = round10(value / 10f64.powf(e as f64));
        let significand = if s % 1.0 == 0.0 { format!("{}", s as i64) } else { crate::pyfmt::format_g(s, 10) };
        let significand = fix_minus(&significand);
        if e == 0 { return significand; }
        format!("{significand}e{}", fix_minus(&e.to_string()))
    }

    /// `get_offset()`.
    pub fn offset_text(&self) -> String {
        if self.locs.is_empty() || (self.order_of_magnitude == 0 && self.offset == 0.0) { return String::new(); }
        let mut offset = String::new();
        if self.offset != 0.0 {
            offset = Self::format_data(self.offset);
            if self.offset > 0.0 { offset = format!("+{offset}"); }
        }
        let sci = if self.order_of_magnitude != 0 { format!("1e{}", self.order_of_magnitude) } else { String::new() };
        fix_minus(&format!("{sci}{offset}"))
    }
}

/// Python's `round(x, 10)` for a float.
fn round10(x: f64) -> f64 {
    let text = format!("{x:.10}");
    text.parse().unwrap_or(x)
}

fn compute_offset(locs: &[f64]) -> f64 {
    if locs.is_empty() { return 0.0; }
    let lmin = locs.iter().copied().fold(f64::INFINITY, f64::min);
    let lmax = locs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if lmin == lmax || (lmin <= 0.0 && 0.0 <= lmax) { return 0.0; }
    let (abs_min, abs_max) = { let (a, b) = (lmin.abs(), lmax.abs()); if a <= b { (a, b) } else { (b, a) } };
    let sign = 1f64.copysign(lmin);
    let oom_max = abs_max.log10().ceil();
    let mut oom = oom_max;
    while floor_div(abs_min, 10f64.powf(oom)) == floor_div(abs_max, 10f64.powf(oom)) { oom -= 1.0; }
    oom += 1.0;
    if (abs_max - abs_min) / 10f64.powf(oom) <= 1e-2 {
        let mut o = oom_max;
        while !(floor_div(abs_max, 10f64.powf(o)) - floor_div(abs_min, 10f64.powf(o)) > 1.0) { o -= 1.0; }
        oom = o + 1.0;
    }
    let q = floor_div(abs_max, 10f64.powf(oom));
    if q >= 1000.0 { sign * q * 10f64.powf(oom) } else { 0.0 }
}
