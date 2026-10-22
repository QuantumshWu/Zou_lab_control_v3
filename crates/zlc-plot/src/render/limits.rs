//! Axis limits a frame shows (`rendering.py` `_curve_x_limits`, `_data_limits`,
//! `_relim_retains`, `_autoscaled_limits`, `_select_display_limits`).

use crate::kinds::RelimMode;

fn finite_range(values: &[f64]) -> Option<(f64, f64)> {
    values.iter().filter(|v| v.is_finite()).fold(None, |range, &v| Some(match range {
        None => (v, v),
        Some((low, high)) => (if v < low { v } else { low }, if v > high { v } else { high }),
    }))
}

/// The authored curve coordinate span without autoscale padding.
pub fn curve_x_limits(values: &[f64]) -> Option<(f64, f64)> {
    let (low, high) = finite_range(values)?;
    if low == high {
        let half_span = 0.5f64.max(0.05 * low.abs());
        return Some((low - half_span, high + half_span));
    }
    Some((low, high))
}

pub fn data_limits(values: &[f64]) -> Option<(f64, f64)> { finite_range(values) }

/// `np.allclose(rtol=1e-12, atol=1e-15)` for scalars.
pub fn scalar_close(left: f64, right: f64) -> bool { (left - right).abs() <= 1e-15 + 1e-12 * right.abs() }

/// Whether an autoscaled axis may keep the limits it already shows.
pub fn relim_retains(mode: RelimMode, rules_geometry: bool) -> bool { mode != RelimMode::Tight || rules_geometry }

/// The limits one autoscaled axis takes for this data.
pub fn autoscaled_limits(data_range: (f64, f64), current: Option<(f64, f64)>, padding_fraction: f64,
                         deadband_fraction: f64, zero_based: bool, retain: bool) -> (f64, f64) {
    let (low, high) = data_range;
    let target = if zero_based && low >= 0.0 {
        (0.0, if high != 0.0 { high * 1.2 } else { 1.0 })
    } else {
        let span = high - low;
        let data_span = if span != 0.0 { span } else if high.abs() != 0.0 { high.abs() } else { 1.0 };
        let padding = padding_fraction * data_span;
        (low - padding, high + padding)
    };
    let Some((current_low, current_high)) = current.filter(|_| retain) else { return target };
    let span = current_high - current_low;
    if span <= 0.0 { return target; }
    let clips = low < current_low || high > current_high;
    let too_empty = if zero_based && low >= 0.0 && current_low == 0.0 {
        !(0.7 * current_high <= high) || current_high <= 0.0
    } else {
        high < current_high - deadband_fraction * span || low > current_low + deadband_fraction * span
    };
    if clips || too_empty { target } else { (current_low, current_high) }
}

/// The limits a display shows: automatic, or authored when fixed (or partially authored).
pub fn select_display_limits(mode: RelimMode, automatic: Option<(f64, f64)>, authored: (Option<f64>, Option<f64>),
                             allow_partial: bool) -> Result<(f64, f64), String> {
    let (authored_low, authored_high) = authored;
    if mode == RelimMode::Fixed && (authored_low.is_none() || authored_high.is_none()) {
        return Err("committed fixed limits are incomplete".into());
    }
    if mode != RelimMode::Fixed && !allow_partial { return Ok(automatic.unwrap_or((0.0, 1.0))); }
    let base = automatic.unwrap_or((0.0, 1.0));
    let mut low = authored_low.unwrap_or(base.0);
    let mut high = authored_high.unwrap_or(base.1);
    if low >= high {
        if authored_low.is_some() && authored_high.is_some() { return Err("committed display limits are not increasing".into()); }
        let span = low.abs().max(high.abs()).max(1.0) * 1e-12;
        (low, high) = if authored_low.is_some() { (low, low + span) } else { (high - span, high) };
    }
    Ok((low, high))
}
