//! What a dense trace looks like at pixel resolution (`rendering.py`
//! `_pixel_columns`, `_thinned_to_columns`, `_envelope_decimated`,
//! `_bounded_isolated_curve_points`): the one decision both the export draw and
//! the live stroke read, so they stroke the same polyline.

use crate::ticks::linspace;

const MIN_POINTS_PER_COLUMN: usize = 4;
const MIN_COLUMNS: f64 = 64.0;
const MAX_COLUMNS: f64 = 4096.0;

/// Two columns per pixel of the axes box, between a floor and a ceiling.
pub fn pixel_columns(box_width_px: f64) -> usize {
    MAX_COLUMNS.min(MIN_COLUMNS.max(box_width_px * 2.0)) as usize
}

/// `numpy.searchsorted(sorted, value, side)` for a sorted slice.
fn searchsorted(sorted: &[f64], value: f64, right: bool) -> usize {
    if right { sorted.partition_point(|&x| x <= value) } else { sorted.partition_point(|&x| x < value) }
}

/// `np.minimum`/`np.maximum` (NaN propagates).
fn nan_min(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else { a.min(b) } }
fn nan_max(a: f64, b: f64) -> f64 { if a.is_nan() || b.is_nan() { f64::NAN } else { a.max(b) } }

/// `ufunc.reduceat(values, indices)`.
fn reduceat(values: &[f64], indices: &[usize], combine: fn(f64, f64) -> f64) -> Vec<f64> {
    (0..indices.len()).map(|i| {
        let start = indices[i];
        let end = if i + 1 < indices.len() { indices[i + 1] } else { values.len() };
        if start < end { values[start + 1..end].iter().fold(values[start], |acc, &v| combine(acc, v)) } else { values[start] }
    }).collect()
}

/// The polyline as drawn: each column's extremes when denser than the columns.
pub fn thinned_to_columns(x: &[f64], y: &[f64], window: (f64, f64), columns: usize) -> (Vec<f64>, Vec<f64>) {
    if x.len() >= columns * MIN_POINTS_PER_COLUMN {
        if let Some(enveloped) = envelope_decimated(x, y, window, columns) { return enveloped; }
    }
    (x.to_vec(), y.to_vec())
}

/// Per-column min/max envelope of a dense polyline (x finite and sorted, gaps as NaN in y).
pub fn envelope_decimated(x: &[f64], y: &[f64], window: (f64, f64), columns: usize) -> Option<(Vec<f64>, Vec<f64>)> {
    let (low, high) = window;
    if !(low.is_finite() && high.is_finite()) || high <= low { return None; }
    let start = searchsorted(x, low, false);
    let stop = searchsorted(x, high, true);
    if stop.saturating_sub(start) < columns * MIN_POINTS_PER_COLUMN { return None; }
    let (x_view, y_view) = (&x[start..stop], &y[start..stop]);
    let finite: Vec<bool> = y_view.iter().map(|v| v.is_finite()).collect();
    if !finite.iter().any(|&f| f) { return None; }
    let mut run_edges = vec![0];
    run_edges.extend((1..finite.len()).filter(|&i| finite[i] != finite[i - 1]));
    run_edges.push(x_view.len());
    let edges = linspace(low, high, columns + 1);
    let (mut out_x, mut out_y) = (Vec::new(), Vec::new());
    if start > 0 { out_x.push(x[start - 1]); out_y.push(y[start - 1]); }
    for pair in run_edges.windows(2) {
        let (run_start, run_stop) = (pair[0], pair[1]);
        if !finite[run_start] {
            out_x.push(x_view[run_start]);
            out_y.push(f64::NAN);
            continue;
        }
        let (run_x, run_y) = (&x_view[run_start..run_stop], &y_view[run_start..run_stop]);
        let first = (searchsorted(&edges, run_x[0], true) as isize - 1).max(0) as usize;
        let last = ((searchsorted(&edges, run_x[run_x.len() - 1], true) as isize - 1).min(columns as isize - 1)) as usize;
        let view = &edges[first..last + 2];
        let starts: Vec<usize> = view[..view.len() - 1].iter().map(|&edge| searchsorted(run_x, edge, false)).collect();
        let counts: Vec<usize> = (0..starts.len()).map(|i| if i + 1 < starts.len() { starts[i + 1] - starts[i] } else { run_x.len() - starts[i] }).collect();
        let indices: Vec<usize> = starts.iter().map(|&s| s.min(run_x.len() - 1)).collect();
        let col_min = reduceat(run_y, &indices, nan_min);
        let col_max = reduceat(run_y, &indices, nan_max);
        for i in 0..starts.len() {
            let centre = 0.5 * (view[i] + view[i + 1]);
            let blank = counts[i] == 0;
            out_x.extend([centre; 3]);
            let (minimum, maximum) = if blank { (f64::NAN, f64::NAN) } else { (col_min[i], col_max[i]) };
            out_y.extend([minimum, maximum, maximum]);
        }
    }
    if stop < x.len() { out_x.push(x[stop]); out_y.push(y[stop]); }
    Some((out_x, out_y))
}

/// Valid vertices with no valid neighbour on either side.
pub fn isolated_mask(finite: &[bool]) -> Vec<bool> {
    (0..finite.len()).map(|i| {
        let left = i > 0 && finite[i - 1] && finite[i];
        let right = i + 1 < finite.len() && finite[i] && finite[i + 1];
        finite[i] && !(left || right)
    }).collect()
}

/// Visible singleton glyphs, bounded to two per display column.
pub fn bounded_isolated_points(x: &[f64], y: &[f64], window: (f64, f64), columns: usize) -> (Vec<f64>, Vec<f64>) {
    let (low, high) = if window.0 <= window.1 { window } else { (window.1, window.0) };
    let start = searchsorted(x, low, false).saturating_sub(1);
    let stop = (searchsorted(x, high, true) + 1).min(x.len());
    let (vx, vy) = (&x[start..stop], &y[start..stop]);
    let finite: Vec<bool> = vx.iter().zip(vy).map(|(a, b)| a.is_finite() && b.is_finite()).collect();
    let isolated = isolated_mask(&finite);
    if !isolated.iter().any(|&i| i) { return (Vec::new(), Vec::new()); }
    let (mut ix, mut iy) = (Vec::new(), Vec::new());
    for i in 0..vx.len() {
        if isolated[i] && vx[i] >= low && vx[i] <= high { ix.push(vx[i]); iy.push(vy[i]); }
    }
    if ix.len() <= 2 * columns || !(high > low) { return (ix, iy); }
    let bins: Vec<i64> = ix.iter().map(|&v| ((((v - low) * columns as f64) / (high - low)).floor() as i64).clamp(0, columns as i64 - 1)).collect();
    let starts: Vec<usize> = (0..bins.len()).filter(|&i| i == 0 || bins[i] != bins[i - 1]).collect();
    let minimum = reduceat(&iy, &starts, nan_min);
    let maximum = reduceat(&iy, &starts, nan_max);
    let (mut bx, mut by) = (Vec::new(), Vec::new());
    for (k, &s) in starts.iter().enumerate() {
        let centre = low + ((bins[s] as f64 + 0.5) * (high - low)) / columns as f64;
        bx.push(centre);
        by.push(minimum[k]);
        if maximum[k] != minimum[k] { bx.push(centre); by.push(maximum[k]); }
    }
    (bx, by)
}
