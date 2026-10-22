//! The scene's z ticks: Matplotlib's `MaxNLocator(nbins=2).tick_values` over
//! the bar limits, kept inside them and labelled `f"{tick:g}"`, as
//! `_update_height_bars_artist` plans them. numpy/CPython semantics (floor
//! division, divmod, the C library's log10 and pow) are followed exactly.

use super::chrome::format_general6;
use super::pymath::{py_floordiv, py_max, py_min, py_mod};

/// `10 ** x` through the C library's pow (a literal base could be rewritten
/// into another function by the optimiser).
fn pow10(x: f64) -> f64 {
    std::hint::black_box(10.0f64).powf(x)
}

/// `matplotlib.transforms.nonsingular(vmin, vmax, expander, tiny)`.
fn nonsingular(mut vmin: f64, mut vmax: f64, expander: f64, tiny: f64) -> (f64, f64) {
    if !vmin.is_finite() || !vmax.is_finite() {
        return (-expander, expander);
    }
    if vmax < vmin {
        std::mem::swap(&mut vmin, &mut vmax);
    }
    let maxabsvalue = py_max(vmin.abs(), vmax.abs());
    if maxabsvalue < (1e6 / tiny) * f64::MIN_POSITIVE {
        vmin = -expander;
        vmax = expander;
    } else if vmax - vmin <= maxabsvalue * tiny {
        if vmax == 0.0 && vmin == 0.0 {
            vmin = -expander;
            vmax = expander;
        } else {
            vmin -= expander * vmin.abs();
            vmax += expander * vmax.abs();
        }
    }
    (vmin, vmax)
}

/// `matplotlib.ticker.scale_range(vmin, vmax, n)` with threshold 100.
fn scale_range(vmin: f64, vmax: f64, n: f64) -> (f64, f64) {
    let dv = (vmax - vmin).abs();
    let meanv = (vmax + vmin) / 2.0;
    let offset = if meanv.abs() / dv < 100.0 {
        0.0
    } else {
        pow10(py_floordiv(meanv.abs().log10(), 1.0)).copysign(meanv)
    };
    (pow10(py_floordiv((dv / n).log10(), 1.0)), offset)
}

/// `_Edge_integer`: tick indices with slop for a large offset.
struct Edge {
    step: f64,
    offset: f64,
}

impl Edge {
    fn closeto(&self, ms: f64, edge: f64) -> bool {
        let tol = if self.offset > 0.0 {
            let digits = (self.offset / self.step).log10();
            py_min(0.4999, py_max(1e-10, pow10(digits - 12.0)))
        } else {
            1e-10
        };
        (ms - edge).abs() < tol
    }

    fn le(&self, x: f64) -> f64 {
        let (d, m) = (py_floordiv(x, self.step), py_mod(x, self.step));
        if self.closeto(m / self.step, 1.0) { d + 1.0 } else { d }
    }

    fn ge(&self, x: f64) -> f64 {
        let (d, m) = (py_floordiv(x, self.step), py_mod(x, self.step));
        if self.closeto(m / self.step, 0.0) { d } else { d + 1.0 }
    }
}

/// `MaxNLocator(nbins).tick_values(vmin, vmax)` with its default steps
/// (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10), no integer/symmetric/prune options, min_n_ticks 2,
/// no 3D axis. `round_numbers` is `axes.autolimit_mode == "round_numbers"`.
pub fn max_n_locator(vmin: f64, vmax: f64, nbins: usize, round_numbers: bool) -> Vec<f64> {
    let (vmin, vmax) = nonsingular(vmin, vmax, 1e-13, 1e-14);
    let nbins_f = nbins as f64;
    let (scale, offset) = scale_range(vmin, vmax, nbins_f);
    let lo = vmin - offset;
    let hi = vmax - offset;
    // _staircase of the default steps: 0.1 * steps[:-1], steps, 10 * steps[1]
    let base = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0];
    let mut steps: Vec<f64> = base[..base.len() - 1].iter().map(|s| 0.1 * s).collect();
    steps.extend(base);
    steps.push(10.0 * base[1]);
    for step in steps.iter_mut() {
        *step *= scale;
    }
    let raw_step = (hi - lo) / nbins_f;
    let large = |step: f64| {
        let mut large = step >= raw_step;
        if round_numbers {
            let floored_vmin = py_floordiv(lo, step) * step;
            let floored_vmax = floored_vmin + step * nbins_f;
            large = large && floored_vmax >= hi;
        }
        large
    };
    let istep = steps.iter().position(|&step| large(step)).unwrap_or(steps.len() - 1);
    let mut ticks = Vec::new();
    for &step in steps[..=istep].iter().rev() {
        let best_vmin = py_floordiv(lo, step) * step;
        let edge = Edge { step, offset: offset.abs() };
        let low = edge.le(lo - best_vmin);
        let high = edge.ge(hi - best_vmin);
        // np.arange(low, high + 1): integral floats from low, step one
        let count = (high + 1.0 - low).ceil().max(0.0) as usize;
        ticks = (0..count).map(|i| (low + i as f64) * step + best_vmin).collect();
        let shown = ticks.iter().filter(|&&tick| tick <= hi && tick >= lo).count();
        if shown >= 2 {
            break;
        }
    }
    ticks.into_iter().map(|tick| tick + offset).collect()
}

/// The z ticks `_update_height_bars_artist` plans: two-bin ticks over the
/// bar limits, inside them (to 1e-12), each with its `:g` label.
pub fn z_ticks(low: f64, high: f64, round_numbers: bool) -> Vec<(f64, String)> {
    max_n_locator(low, high, 2, round_numbers)
        .into_iter()
        .filter(|&tick| low - 1e-12 <= tick && tick <= high + 1e-12)
        .map(|tick| (tick, format_general6(tick)))
        .collect()
}
