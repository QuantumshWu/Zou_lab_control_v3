//! Pixel geometry the renderer settles before drawing (`rendering.py`
//! helpers): axes boxes on whole pixels and on their aspect, image extents,
//! the image-to-viewport sample map and histogram bar vertices.

use super::scale::AxisScale;

/// Where a box that does not fill its slot sits in it, per anchor code.
pub fn anchor_fractions(anchor: &str) -> (f64, f64) {
    match anchor {
        "C" => (0.5, 0.5), "SW" => (0.0, 0.0), "S" => (0.5, 0.0), "SE" => (1.0, 0.0), "E" => (1.0, 0.5),
        "NE" => (1.0, 1.0), "N" => (0.5, 1.0), "NW" => (0.0, 1.0), "W" => (0.0, 0.5),
        _ => (0.5, 0.5),
    }
}

/// Python's `round()` of a float to an integer (half to even).
fn round_int(value: f64) -> i64 { value.round_ties_even() as i64 }

/// A span near `base` whose box measures exactly `extent` pixels from `origin`.
fn span_landing_on(origin: f64, extent: f64, scale: f64, base: f64) -> f64 {
    if scale == 0.0 { return base; }
    let low = origin * scale;
    let mut candidate = base;
    for _ in 0..24 {
        if (origin + candidate) * scale - low == extent { return candidate; }
        candidate = candidate.next_up();
    }
    candidate = base;
    for _ in 0..24 {
        candidate = candidate.next_down();
        if (origin + candidate) * scale - low == extent { return candidate; }
    }
    base
}

/// Fractions that put a box at `low` measuring exactly `high - low` pixels,
/// and whether such a pair was found.
pub fn exact_box_fractions(low: i64, high: i64, scale: f64) -> (f64, f64, bool) {
    let extent = (high - low) as f64;
    let base_left = low as f64 / scale;
    let base_span = (high - low) as f64 / scale;
    let mut origins = vec![base_left];
    let (mut upward, mut downward) = (base_left, base_left);
    for _ in 0..4 {
        upward = upward.next_up();
        downward = downward.next_down();
        origins.push(upward);
        origins.push(downward);
    }
    for origin in origins {
        let span = span_landing_on(origin, extent, scale, base_span);
        if (origin + span) * scale - origin * scale == extent { return (origin, span, true); }
    }
    (base_left, base_span, false)
}

/// The largest whole-pixel box inside `box_w x box_h` drawn at `ratio` (height/width).
pub fn box_on_aspect(box_w: i64, box_h: i64, ratio: f64) -> Option<(i64, i64)> {
    const SEARCH: i64 = 64;
    if box_w < 2 || box_h < 2 || !ratio.is_finite() || ratio <= 0.0 { return None; }
    let mut best: Option<(i64, i64)> = None;
    let mut width = box_w;
    while width >= (box_w - SEARCH).max(2) {
        let height = width as f64 * ratio;
        if (height - height.round_ties_even()).abs() < 1e-9 && (2..=box_h).contains(&round_int(height)) {
            best = Some((width, round_int(height)));
            break;
        }
        width -= 1;
    }
    let mut height = box_h;
    while height >= (box_h - SEARCH).max(2) {
        let width = height as f64 / ratio;
        if (width - width.round_ties_even()).abs() < 1e-9 && (2..=box_w).contains(&round_int(width)) {
            let candidate = (round_int(width), height);
            if best.is_none_or(|best| candidate.0 * candidate.1 > best.0 * best.1) { best = Some(candidate); }
            break;
        }
        height -= 1;
    }
    if best.is_some() { return best; }
    let height = box_h.min(round_int(box_w as f64 * ratio));
    let width = box_w.min(round_int(height as f64 / ratio));
    (width >= 2 && height >= 2).then_some((width, height))
}

/// One axes box snapped to whole physical pixels (`_pixel_quantized_bounds`):
/// the snapped (left, bottom, width, height) fractions and whether the box is
/// exactly integral, or the plan's bounds unchanged when no snap applies.
pub fn pixel_quantized_bounds(bounds: [f64; 4], figure_width: f64, figure_height: f64, ratio: Option<f64>,
                              facet_overview: bool, anchor: &str) -> ([f64; 4], bool) {
    if figure_width <= 1.0 || figure_height <= 1.0 { return (bounds, false); }
    let x0 = (bounds[0] * figure_width).floor() as i64;
    let y0 = (bounds[1] * figure_height).floor() as i64;
    let mut x1 = ((bounds[0] + bounds[2]) * figure_width).ceil() as i64;
    let mut y1 = ((bounds[1] + bounds[3]) * figure_height).ceil() as i64;
    if facet_overview {
        x1 = x0 + (bounds[2] * figure_width - 1e-9).ceil() as i64;
        y1 = y0 + (bounds[3] * figure_height - 1e-9).ceil() as i64;
    }
    if x1 - x0 < 2 || y1 - y0 < 2 { return (bounds, false); }
    let (mut x0, mut y0) = (x0, y0);
    if let Some(ratio) = ratio {
        let Some((width, height)) = box_on_aspect(x1 - x0, y1 - y0, ratio) else { return (bounds, false) };
        let (fraction_x, fraction_y) = anchor_fractions(anchor);
        // Python int() truncates toward zero.
        x0 += ((x1 - x0 - width) as f64 * fraction_x) as i64;
        y0 += ((y1 - y0 - height) as f64 * fraction_y) as i64;
        x1 = x0 + width;
        y1 = y0 + height;
    }
    let (left, span_x, exact_x) = exact_box_fractions(x0, x1, figure_width);
    let (bottom, span_y, exact_y) = exact_box_fractions(y0, y1, figure_height);
    ([left, bottom, span_x, span_y], exact_x && exact_y)
}

/// Whether the snapped box is already on its aspect within half a pixel (`_settle_owned_boxes`).
pub fn owns_box(bounds: [f64; 4], figure_width: f64, figure_height: f64, ratio: Option<f64>) -> bool {
    let Some(ratio) = ratio else { return true };
    let box_w = (bounds[0] + bounds[2]) * figure_width - bounds[0] * figure_width;
    let box_h = (bounds[1] + bounds[3]) * figure_height - bounds[1] * figure_height;
    (box_w * ratio - box_h).abs() <= 0.5 + 1e-9
}

/// Whether two boxes (x0, y0, x1, y1) share any pixel.
pub fn boxes_meet(one: [f64; 4], other: [f64; 4]) -> bool {
    !(one[2] <= other[0] || other[2] <= one[0] || one[3] <= other[1] || other[3] <= one[1])
}

/// Linear unless the coordinates are a nonuniform lattice (`_image_coordinate_scale`).
pub fn image_coordinate_scale(values: &[f64]) -> Result<AxisScale, String> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) { return Err("image coordinates must be non-empty and finite".into()); }
    if values.len() < 2 { return Ok(AxisScale::Linear); }
    let increasing = values.windows(2).all(|w| w[1] - w[0] > 0.0);
    let decreasing = values.windows(2).all(|w| w[1] - w[0] < 0.0);
    if !(increasing || decreasing) { return Err("image coordinates must be strictly monotonic".into()); }
    let n = values.len();
    let pitch = (values[n - 1] - values[0]) / (n - 1) as f64;
    let largest = values.iter().fold(f64::NEG_INFINITY, |acc, v| acc.max(v.abs()));
    let tolerance = 16.0 * f64::EPSILON * largest.max(pitch.abs());
    let regular = values.iter().enumerate().all(|(i, v)| (v - (values[0] + pitch * i as f64)).abs() <= tolerance);
    Ok(if regular { AxisScale::Linear } else { AxisScale::Samples(values.into()) })
}

/// Image extent (x0, x1, y0, y1) in drawing coordinates, half a cell past each end.
pub fn centers_extent(x: &[f64], y: &[f64], scales: (&AxisScale, &AxisScale)) -> (f64, f64, f64, f64) {
    fn edge(values: &[f64], scale: &AxisScale) -> (f64, f64) {
        let values: Vec<f64> = values.iter().map(|&v| scale.space(v)).collect();
        if values.len() == 1 { return (values[0] - 0.5, values[0] + 0.5); }
        let first = values[1] - values[0];
        let last = values[values.len() - 1] - values[values.len() - 2];
        (values[0] - first / 2.0, values[values.len() - 1] + last / 2.0)
    }
    let (x0, x1) = edge(x, scales.0);
    let (y0, y1) = edge(y, scales.1);
    (x0, x1, y0, y1)
}

/// Viewport pixel centres to nearest source samples: (start, stop, source index per pixel).
pub fn view_nearest_map(output_count: usize, source_count: usize, view: (f64, f64), source: (f64, f64)) -> Option<(usize, usize, Vec<usize>)> {
    let bounds = [view.0, view.1, source.0, source.1];
    if output_count < 1 || source_count < 1 || bounds.iter().any(|b| !b.is_finite()) || view.0 == view.1 || source.0 == source.1 {
        return None;
    }
    let step = (view.1 - view.0) / output_count as f64;
    let positions: Vec<f64> = (0..output_count).map(|i| {
        let centre = view.0 + ((i as f64 + 0.5) * step);
        (centre - source.0) / (source.1 - source.0)
    }).collect();
    let inside = |p: f64| p.is_finite() && p >= 0.0 && p < 1.0;
    let start = positions.iter().position(|&p| inside(p))?;
    let stop = positions.iter().rposition(|&p| inside(p))? + 1;
    let mapped = positions[start..stop].iter().map(|&p| {
        let index = (p * source_count as f64).floor();
        if index.is_nan() { 0 } else { (index.max(0.0) as usize).min(source_count - 1) }
    }).collect();
    Some((start, stop, mapped))
}

/// Where an image's extent lands in the canvas in whole pixels, (x0, y0, width, height), y up.
pub fn image_destination_rect(bbox: [f64; 4], extent: (f64, f64, f64, f64), x_limits: (f64, f64), y_limits: (f64, f64))
                              -> Option<(i64, i64, i64, i64)> {
    let (left, right, upper, lower) = extent;
    let (x_low, x_high) = (x_limits.0.min(x_limits.1), x_limits.0.max(x_limits.1));
    let (y_low, y_high) = (y_limits.0.min(y_limits.1), y_limits.0.max(y_limits.1));
    if x_high <= x_low || y_high <= y_low { return None; }
    let (bx0, by0, width, height) = (bbox[0], bbox[1], bbox[2], bbox[3]);
    let scale_x = width / (x_high - x_low);
    let scale_y = height / (y_high - y_low);
    let x_start = bx0 + (left.min(right) - x_low) * scale_x;
    let x_end = bx0 + (left.max(right) - x_low) * scale_x;
    let y_start = by0 + (upper.min(lower) - y_low) * scale_y;
    let y_end = by0 + (upper.max(lower) - y_low) * scale_y;
    if [x_start, y_start, x_end, y_end].iter().any(|v| (v - v.round_ties_even()).abs() > 1e-9) { return None; }
    let w = round_int(x_end) - round_int(x_start);
    let h = round_int(y_end) - round_int(y_start);
    (w >= 1 && h >= 1).then_some((round_int(x_start), round_int(y_start), w, h))
}

/// Letterbox one image footprint inside a fixed square viewport.
pub fn square_image_limits(extent: (f64, f64, f64, f64), coordinate_aspect: f64) -> Result<(f64, f64, f64, f64), String> {
    let (mut left, mut right, mut upper, mut lower) = extent;
    let scale = coordinate_aspect;
    let x_span = (right - left).abs();
    let y_span = (lower - upper).abs() * scale;
    if x_span <= 0.0 || y_span <= 0.0 || !scale.is_finite() || scale <= 0.0 { return Err("image extent spans must be positive".into()); }
    if x_span > y_span {
        let padding = (x_span - y_span) / (2.0 * scale);
        let direction = if lower > upper { 1.0 } else { -1.0 };
        upper -= direction * padding;
        lower += direction * padding;
    } else if y_span > x_span {
        let padding = (y_span - x_span) / 2.0;
        let direction = if right > left { 1.0 } else { -1.0 };
        left -= direction * padding;
        right += direction * padding;
    }
    Ok((left, right, upper, lower))
}

/// How wide one image axis is extent-wise: N samples cover N pitches.
pub fn image_axis_span(values: &[f64]) -> Option<f64> {
    if values.len() < 2 { return None; }
    let span = (values[values.len() - 1] - values[0]).abs();
    (span.is_finite() && span > 0.0).then(|| span * values.len() as f64 / (values.len() - 1) as f64)
}

/// Screen y/x scale that makes one x cell and one y cell equally long.
pub fn image_cell_aspect(x: &[f64], y: &[f64], scales: (&AxisScale, &AxisScale)) -> Option<f64> {
    let mut pitches = [0.0; 2];
    for (slot, (values, scale)) in [(x, scales.0), (y, scales.1)].into_iter().enumerate() {
        let values: Vec<f64> = values.iter().map(|&v| scale.space(v)).collect();
        if values.is_empty() { return None; }
        let pitch = if values.len() == 1 { 1.0 } else { image_axis_span(&values)? / values.len() as f64 };
        if !pitch.is_finite() || pitch <= 0.0 { return None; }
        pitches[slot] = pitch;
    }
    Some(pitches[0] / pitches[1])
}

/// Four vertices per bar: (left, 0), (left, count), (right, count), (right, 0).
pub fn histogram_vertices(edges: &[f64], counts: &[f64]) -> Result<Vec<[[f64; 2]; 4]>, String> {
    if edges.len() != counts.len() + 1 { return Err("histogram edges must contain one more value than counts".into()); }
    Ok(counts.iter().enumerate().map(|(i, &count)| [[edges[i], 0.0], [edges[i], count], [edges[i + 1], count], [edges[i + 1], 0.0]]).collect())
}
