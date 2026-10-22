//! The room's vector chrome as pure geometry over a rendered scene: the
//! occlusion sampler, the visible-run stitching, box edges, the ground
//! anchors and the axis/grid/tick/label layout of
//! `MatplotlibRenderer._update_height_bars_chrome`, the cage and readout of
//! the crosshair-selected bar, and the canvas<->cell mappings. Artists and
//! text measurement stay with the caller; only positions are decided here.

use super::pymath::{np_clip, np_hypot, np_maximum, np_minimum, py_max, py_min, trunc_i64};
use super::HeightBarScene;

/// One 3D edge: two folded (a, b, value) endpoints.
pub type Edge = [[f64; 3]; 2];

const OCCLUSION_PIXELS_PER_SAMPLE: f64 = 2.0;
const OCCLUSION_MIN_SAMPLES: f64 = 4.0;
const OCCLUSION_MAX_SAMPLES: f64 = 512.0;

/// `_height_bars_fraction`: scene pixel (top origin) -> axes fractions.
pub fn fraction(scene: &HeightBarScene, x: f64, y: f64) -> (f64, f64) {
    (x / scene.width.max(1) as f64, 1.0 - y / scene.height.max(1) as f64)
}

/// `_height_bars_ground_anchors`: the ground rectangle's sides by the DATA
/// on them (heatmap corners a, b, c, d clockwise from its top left).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroundAnchors {
    pub wall_a: f64,
    pub wall_b: f64,
    pub axis_a: f64,
    pub axis_b: f64,
    pub z_a: f64,
    pub z_b: f64,
    pub in_a: f64,
    pub in_b: f64,
}

pub fn ground_anchors(scene: &HeightBarScene) -> GroundAnchors {
    let sides = |row: i64, column: i64| -> (f64, f64) {
        let (a, b) = scene.fold_cell(row, column);
        (
            if a * 2 < scene.nx as i64 { 0.0 } else { scene.nx as f64 },
            if b * 2 < scene.ny as i64 { 0.0 } else { scene.ny as f64 },
        )
    };
    let top_row = if scene.flip_rows { 0 } else { scene.source_ny as i64 - 1 };
    let bottom_row = scene.source_ny as i64 - 1 - top_row;
    let (wall_a, wall_b) = sides(top_row, 0);
    let (axis_a, axis_b) = sides(bottom_row, scene.source_nx as i64 - 1);
    let (z_a, z_b) = sides(bottom_row, 0);
    GroundAnchors {
        wall_a,
        wall_b,
        axis_a,
        axis_b,
        z_a,
        z_b,
        in_a: if axis_a == 0.0 { 1.0 } else { -1.0 },
        in_b: if axis_b == 0.0 { 1.0 } else { -1.0 },
    }
}

/// `_height_bars_sampled_polyline`: `samples` points per edge, NaN where the
/// scene hides them, one NaN after each edge. Returns axes fractions.
pub fn sampled_polyline(scene: &HeightBarScene, edges: &[Edge], samples: usize) -> (Vec<f64>, Vec<f64>) {
    let stride = samples + 1;
    let mut xs = vec![f64::NAN; edges.len() * stride];
    let mut ys = vec![f64::NAN; edges.len() * stride];
    let (width, height) = (scene.width as i64, scene.height as i64);
    let (nx, ny) = (scene.nx as i64, scene.ny as i64);
    let (ca, sa, se, ce, z_unit) = (scene.ca, scene.sa, scene.se, scene.ce, scene.z_unit);
    let (x_low, y_high, scale) = (scene.x_low, scene.y_high, scene.scale);
    let z_slack = 0.5 / py_max(z_unit * ce * scale, 1e-9);
    let rise = se / (z_unit * ce);
    let delta = 1.0 / (samples as f64 - 1.0);
    let top_values = &scene.top_values;
    let mut hidden = vec![false; samples];
    for (e, edge) in edges.iter().enumerate() {
        let [a0, b0, z0] = edge[0];
        let (da, db, dz) = (edge[1][0] - a0, edge[1][1] - b0, edge[1][2] - z0);
        let base = e * stride;
        for n in 0..samples {
            // np.linspace(0, 1, samples): n * (1/(samples-1)), exact end;
            // a single sample is the start.
            let fraction = if samples == 1 { 0.0 } else if n == samples - 1 { 1.0 } else { n as f64 * delta };
            let ga = a0 + da * fraction;
            let gb = b0 + db * fraction;
            let gz = z0 + dz * fraction;
            let sx = ga * ca + gb * sa;
            let sy = -ga * sa * se + gb * ca * se + gz * z_unit * ce;
            let px = (sx - x_low) * scale;
            let py = (y_high - sy) * scale;
            let column = trunc_i64(px).clamp(0, width - 1);
            let row = trunc_i64(py).clamp(0, height - 1);
            let face = scene.id_plane[(row * width + column) as usize] as i64;
            let mut is_hidden = false;
            if face >= 4 {
                let shown = (face - 4) >> 2;
                let shown_a = shown.rem_euclid(nx);
                let shown_b = shown.div_euclid(nx);
                let sa_i = shown_a.clamp(0, nx - 1);
                let sb_i = shown_b.clamp(0, ny - 1);
                let shown_top = top_values[(sb_i * nx + sa_i) as usize];
                let enter_a = (shown_a as f64 - ga) / sa;
                let enter_b = (gb - shown_b as f64 - 1.0) / ca;
                let mut enter = enter_a;
                if enter_b > enter {
                    enter = enter_b;
                }
                if enter < 0.0 {
                    enter = 0.0;
                }
                let leave_a = ((shown_a + 1) as f64 - ga) / sa;
                let leave_b = (gb - shown_b as f64) / ca;
                let mut leave = leave_a;
                if leave_b < leave {
                    leave = leave_b;
                }
                if leave > 1e-9 && gz + enter * rise < shown_top - z_slack {
                    is_hidden = true;
                }
            }
            if !is_hidden {
                let cell_a = trunc_i64(ga.floor());
                let cell_b = trunc_i64(gb.ceil()).wrapping_sub(1);
                if 0 <= cell_a && cell_a < nx && 0 <= cell_b && cell_b < ny
                    && gz < top_values[(cell_b * nx + cell_a) as usize] - z_slack
                {
                    is_hidden = true;
                }
            }
            if !is_hidden {
                xs[base + n] = px / scene.width.max(1) as f64;
                ys[base + n] = 1.0 - py / scene.height.max(1) as f64;
            }
        }
        // isolated-visible erosion: a visible sample with hidden neighbours
        // on BOTH sides (edge ends count as hidden) becomes hidden. Read
        // from the written row, as the numba kernel reads it.
        for n in 0..samples {
            hidden[n] = xs[base + n].is_nan();
        }
        for n in 0..samples {
            if hidden[n] {
                continue;
            }
            let left_hidden = n == 0 || hidden[n - 1];
            let right_hidden = n == samples - 1 || hidden[n + 1];
            if left_hidden && right_hidden {
                xs[base + n] = f64::NAN;
                ys[base + n] = f64::NAN;
            }
        }
    }
    (xs, ys)
}

/// `_height_bars_visible_runs`: one (start, end, NaN) triple per visible run.
pub fn visible_runs(xs: &[f64], ys: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let keep: Vec<usize> = (0..xs.len()).filter(|&i| xs[i].is_finite()).collect();
    let mut drawn_x = Vec::new();
    let mut drawn_y = Vec::new();
    let mut index = 0;
    while index < keep.len() {
        let first = keep[index];
        let mut last = first;
        index += 1;
        while index < keep.len() && keep[index] == last + 1 {
            last = keep[index];
            index += 1;
        }
        drawn_x.extend([xs[first], xs[last], f64::NAN]);
        drawn_y.extend([ys[first], ys[last], f64::NAN]);
    }
    (drawn_x, drawn_y)
}

/// The power-of-two sample bucket an edge of `pixels` length asks for.
fn bucket(pixels: f64) -> usize {
    let wanted = np_clip((pixels / OCCLUSION_PIXELS_PER_SAMPLE).ceil() + 2.0, OCCLUSION_MIN_SAMPLES, OCCLUSION_MAX_SAMPLES);
    if wanted.is_nan() {
        return 0; // ceil(log2(nan)) casts to i64::MIN and 1 << that is 0
    }
    // ceil(log2(w)) for an integral w in [4, 512]: the least e with 2^e >= w.
    let mut exponent = 0u32;
    while ((1u64 << exponent) as f64) < wanted {
        exponent += 1;
    }
    1usize << exponent
}

/// `_height_bars_occluded_polyline`: every edge sampled at its own length,
/// grouped into power-of-two batches (ascending), stitched into runs.
pub fn occluded_polyline(scene: &HeightBarScene, edges: &[Edge]) -> (Vec<f64>, Vec<f64>) {
    if edges.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let buckets: Vec<usize> = edges
        .iter()
        .map(|edge| {
            let end = |p: &[f64; 3]| {
                (
                    p[0] * scene.ca + p[1] * scene.sa,
                    -p[0] * scene.sa * scene.se + p[1] * scene.ca * scene.se + p[2] * scene.z_unit * scene.ce,
                )
            };
            let (x0, y0) = end(&edge[0]);
            let (x1, y1) = end(&edge[1]);
            bucket(np_hypot(x1 - x0, y1 - y0) * scene.scale)
        })
        .collect();
    let mut unique = buckets.clone();
    unique.sort_unstable();
    unique.dedup();
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for samples in unique {
        let part: Vec<Edge> = edges.iter().zip(&buckets).filter(|(_, b)| **b == samples).map(|(e, _)| *e).collect();
        let (x, y) = sampled_polyline(scene, &part, samples);
        xs.extend(x);
        ys.extend(y);
    }
    visible_runs(&xs, &ys)
}

/// `_height_bars_box_edges`: the box edges of each (row, column) cell.
pub fn box_edges(scene: &HeightBarScene, cells: &[(i64, i64)], z_low: &[f64], z_high: &[f64]) -> Vec<Edge> {
    let mut edges = Vec::new();
    let pixel_z = 1.0 / py_max(scene.z_unit * scene.ce * scene.scale, 1e-9);
    for (index, &(row, column)) in cells.iter().enumerate() {
        let (a, b) = scene.fold_cell(row, column);
        let (a, b) = (a as f64, b as f64);
        let (low, high) = (z_low[index], z_high[index]);
        let corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)];
        let bottom = corners.map(|(da, db)| [a + da, b + db, low]);
        let top = corners.map(|(da, db)| [a + da, b + db, high]);
        let flat_box = (high - low) < pixel_z;
        for i in 0..4 {
            edges.push([top[i], top[(i + 1) % 4]]);
            if !flat_box {
                edges.push([bottom[i], bottom[(i + 1) % 4]]);
                edges.push([bottom[i], top[i]]);
            }
        }
    }
    edges
}

/// The heatmap the scene stands for: its extent (left, right, bottom, top)
/// and the source grid size (`_height_bars_data_frame`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DataFrame {
    pub extent: (f64, f64, f64, f64),
    pub source_nx: usize,
    pub source_ny: usize,
}

/// `_height_bars_cell_of` for coordinates already in axis space.
pub fn cell_of(frame: &DataFrame, x_space: f64, y_space: f64) -> Option<(i64, i64)> {
    let (left, right, bottom, top) = frame.extent;
    if right == left || bottom == top {
        return None;
    }
    let column = ((x_space - left) / (right - left) * frame.source_nx as f64).floor();
    let row = ((y_space - top) / (bottom - top) * frame.source_ny as f64).floor();
    if !(column.is_finite() && row.is_finite()) {
        return None; // Python's int() raises
    }
    if !(0.0 <= column && column < frame.source_nx as f64 && 0.0 <= row && row < frame.source_ny as f64) {
        return None;
    }
    Some((row as i64, column as i64))
}

/// `height_bars_pick`: canvas pixel (y up) -> the picked bar's coordinate in
/// axis space (the caller applies `axis_value`).
pub fn pick_axis_space(scene: &HeightBarScene, frame: &DataFrame, box_x0: f64, box_y1: f64, canvas_x: f64, canvas_y: f64) -> Option<(f64, f64)> {
    let (row, column) = scene.pick(canvas_x - box_x0, box_y1 - canvas_y)?;
    let (left, right, bottom, top) = frame.extent;
    let x_value = left + (column as f64 + 0.5) * (right - left) / frame.source_nx as f64;
    let y_value = top + (row as f64 + 0.5) * (bottom - top) / frame.source_ny as f64;
    Some((x_value, y_value))
}

/// The cage of `_update_height_bars_cage`: the selected cell's full-span box,
/// occluded like real geometry.
pub fn cage(scene: &HeightBarScene, cell: (i64, i64)) -> (Vec<f64>, Vec<f64>) {
    let z_low = [py_min(scene.value_low, 0.0)];
    let z_high = [py_max(scene.value_high, 0.0)];
    occluded_polyline(scene, &box_edges(scene, &[cell], &z_low, &z_high))
}

/// The tick room `_update_height_bars_artist` plans before rendering: tick
/// values and labels (z), source indices and labels (x, y), the per-axis
/// shared offsets, and the label metrics in canvas pixels.
#[derive(Clone, Debug, Default)]
pub struct ChromeLayout {
    pub z_ticks: Vec<(f64, String)>,
    pub x_ticks: Vec<(i64, String)>,
    pub y_ticks: Vec<(i64, String)>,
    pub x_offset: String,
    pub y_offset: String,
    pub tick_length_px: f64,
    pub label_gap_px: f64,
    pub box_w: usize,
    pub box_h: usize,
    /// The floor the base labels hug (`_height_bars_floor_value`); `None`
    /// when no data frame is known, which drops the base axes.
    pub floor_value: Option<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ChromeText {
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub ha: &'static str,
    pub va: &'static str,
}

/// What the chrome artists are handed: the grid polyline, the axis line
/// polyline (axes, a NaN, then tick marks), and every label in order --
/// the projected ones first (`projected_axes` names their axes for the
/// collision thinning), then the corner offsets.
#[derive(Clone, Debug, Default)]
pub struct ChromeGeometry {
    pub grid_x: Vec<f64>,
    pub grid_y: Vec<f64>,
    pub line_x: Vec<f64>,
    pub line_y: Vec<f64>,
    pub texts: Vec<ChromeText>,
    pub projected_axes: Vec<&'static str>,
}

struct Tick {
    anchor: [f64; 3],
    direction: (f64, f64),
    at: (f64, f64),
    text: String,
    axis: &'static str,
}

/// `_update_height_bars_chrome`, the geometry.
#[allow(clippy::too_many_lines)]
pub fn chrome(scene: &HeightBarScene, layout: &ChromeLayout) -> ChromeGeometry {
    let (box_w, box_h) = (layout.box_w as f64, layout.box_h as f64);
    let mut grid_edges: Vec<Edge> = Vec::new();
    let mut axis_edges: Vec<Edge> = Vec::new();
    let mut ticks: Vec<Tick> = Vec::new();
    let wall_low = py_min(scene.value_low, 0.0);
    let wall_high = py_max(scene.value_high, 0.0);
    let anchor = ground_anchors(scene);
    let (far_a, far_b) = (anchor.wall_a, anchor.wall_b);
    let (near_a, near_b) = (anchor.axis_a, anchor.axis_b);
    let (left_a, left_b) = (anchor.z_a, anchor.z_b);
    let (in_a, in_b) = (anchor.in_a, anchor.in_b);
    for (tick, _) in &layout.z_ticks {
        grid_edges.push([[near_a, far_b, *tick], [far_a, far_b, *tick]]);
        grid_edges.push([[far_a, near_b, *tick], [far_a, far_b, *tick]]);
    }
    let outward = |edge_point: (f64, f64), inner_point: (f64, f64)| -> ((f64, f64), (f64, f64)) {
        let f_edge = fraction(scene, edge_point.0, edge_point.1);
        let f_inner = fraction(scene, inner_point.0, inner_point.1);
        let vx = (f_edge.0 - f_inner.0) * box_w;
        let vy = (f_edge.1 - f_inner.1) * box_h;
        let length = np_hypot(vx, vy);
        if length <= 0.0 {
            return ((0.0, -1.0), f_edge);
        }
        ((vx / length, vy / length), f_edge)
    };
    axis_edges.push([[left_a, left_b, wall_low], [left_a, left_b, wall_high]]);
    for (tick, label) in &layout.z_ticks {
        let (direction, at) = outward(scene.project(left_a, left_b, *tick), scene.project(near_a, near_b, *tick));
        ticks.push(Tick { anchor: [left_a, left_b, *tick], direction, at, text: label.clone(), axis: "z" });
    }
    if let Some(base_value) = layout.floor_value {
        axis_edges.push([[far_a, near_b, base_value], [near_a, near_b, base_value]]);
        axis_edges.push([[near_a, near_b, base_value], [near_a, far_b, base_value]]);
        let a_tick = |grid: &mut Vec<Edge>, ticks: &mut Vec<Tick>, centre: f64, label: &str, axis: &'static str| {
            grid.push([[centre, far_b, wall_low], [centre, far_b, wall_high]]);
            grid.push([[centre, near_b, 0.0], [centre, far_b, 0.0]]);
            let (direction, at) = outward(scene.project(centre, near_b, base_value), scene.project(centre, near_b + in_b, base_value));
            ticks.push(Tick { anchor: [centre, near_b, base_value], direction, at, text: label.to_owned(), axis });
        };
        let b_tick = |grid: &mut Vec<Edge>, ticks: &mut Vec<Tick>, centre: f64, label: &str, axis: &'static str| {
            grid.push([[far_a, centre, wall_low], [far_a, centre, wall_high]]);
            grid.push([[far_a, centre, 0.0], [near_a, centre, 0.0]]);
            let (direction, at) = outward(scene.project(near_a, centre, base_value), scene.project(near_a + in_a, centre, base_value));
            ticks.push(Tick { anchor: [near_a, centre, base_value], direction, at, text: label.to_owned(), axis });
        };
        let even = scene.quadrant.rem_euclid(2) == 0;
        let mut x_ticks: &[(i64, String)] = &layout.x_ticks;
        let mut y_ticks: &[(i64, String)] = &layout.y_ticks;
        if even && y_ticks.len() > 2 {
            y_ticks = &y_ticks[..y_ticks.len() - 1];
        } else if !even && x_ticks.len() > 2 {
            x_ticks = &x_ticks[..x_ticks.len() - 1];
        }
        for (column, label) in x_ticks {
            let (a, b) = scene.fold_cell(0, *column);
            if even {
                a_tick(&mut grid_edges, &mut ticks, a as f64 + 0.5, label, "x");
            } else {
                b_tick(&mut grid_edges, &mut ticks, b as f64 + 0.5, label, "x");
            }
        }
        for (row, label) in y_ticks {
            let (a, b) = scene.fold_cell(*row, 0);
            if even {
                b_tick(&mut grid_edges, &mut ticks, b as f64 + 0.5, label, "y");
            } else {
                a_tick(&mut grid_edges, &mut ticks, a as f64 + 0.5, label, "y");
            }
        }
    }

    // ---- what the scene does not hide of its own ticks
    let anchors: Vec<Edge> = ticks.iter().map(|tick| [tick.anchor, tick.anchor]).collect();
    let (sampled, _) = sampled_polyline(scene, &anchors, 2);
    let mut segments_x = Vec::new();
    let mut segments_y = Vec::new();
    let mut geometry = ChromeGeometry::default();
    for (index, tick) in ticks.iter().enumerate() {
        if !sampled[index * 3].is_finite() {
            continue;
        }
        geometry.projected_axes.push(tick.axis);
        let (ux, uy) = tick.direction;
        let f = tick.at;
        segments_x.extend([f.0, f.0 + ux * layout.tick_length_px / box_w, f64::NAN]);
        segments_y.extend([f.1, f.1 + uy * layout.tick_length_px / box_h, f64::NAN]);
        let ha = if ux > 0.4 { "left" } else if ux < -0.4 { "right" } else { "center" };
        let va = if uy > 0.4 { "bottom" } else if uy < -0.4 { "top" } else { "center" };
        geometry.texts.push(ChromeText {
            x: f.0 + ux * layout.label_gap_px / box_w,
            y: f.1 + uy * layout.label_gap_px / box_h,
            text: tick.text.clone(),
            ha,
            va,
        });
    }
    if !grid_edges.is_empty() {
        (geometry.grid_x, geometry.grid_y) = occluded_polyline(scene, &grid_edges);
    }
    let (mut line_x, mut line_y) = occluded_polyline(scene, &axis_edges);
    line_x.push(f64::NAN);
    line_y.push(f64::NAN);
    line_x.extend(segments_x);
    line_y.extend(segments_y);
    (geometry.line_x, geometry.line_y) = (line_x, line_y);
    for (x, y, text, ha, va) in [
        (1.0, 0.0, &layout.x_offset, "right", "bottom"),
        (0.0, 1.0, &layout.y_offset, "left", "top"),
    ] {
        if !text.is_empty() {
            geometry.texts.push(ChromeText { x, y, text: text.clone(), ha, va });
        }
    }
    geometry
}

/// `_height_bars_floor_value`: the lowest drawn geometry the base labels hug.
pub fn floor_value(lowest: f64, low: f64) -> f64 {
    np_minimum(np_maximum(py_min(lowest, 0.0), low), 0.0)
}

// ---- the crosshair readout of `_update_height_bars_readout` -----------------

/// `_selector_precision`: decimals that resolve a thousandth of the span.
pub fn selector_precision(span: f64) -> Result<i64, String> {
    let gap = if span != 0.0 { span.abs() / 1000.0 } else { 0.01 };
    let exponent = gap.log10().ceil();
    if !exponent.is_finite() {
        return Err("selector precision of a non-finite span".into());
    }
    Ok((-(exponent as i64)).max(0))
}

/// Python `f"{value:.{precision}f}"`.
pub fn format_fixed(value: f64, precision: usize) -> String {
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf".into() } else { "-inf".into() };
    }
    format!("{value:.precision$}")
}

/// Python `f"{value:.6g}"`.
pub fn format_general6(value: f64) -> String {
    const PRECISION: i64 = 6;
    if value.is_nan() {
        return "nan".into();
    }
    if value.is_infinite() {
        return if value > 0.0 { "inf".into() } else { "-inf".into() };
    }
    if value == 0.0 {
        return if value.is_sign_negative() { "-0".into() } else { "0".into() };
    }
    let scientific = format!("{:.*e}", (PRECISION - 1) as usize, value);
    let (mantissa, exponent) = scientific.split_once('e').expect("exponent form");
    let exponent: i64 = exponent.parse().expect("exponent");
    let strip = |text: &str| -> String {
        if text.contains('.') {
            text.trim_end_matches('0').trim_end_matches('.').to_owned()
        } else {
            text.to_owned()
        }
    };
    if (-4..PRECISION).contains(&exponent) {
        strip(&format!("{:.*}", (PRECISION - 1 - exponent) as usize, value))
    } else {
        let sign = if exponent < 0 { '-' } else { '+' };
        format!("{}e{sign}{:02}", strip(mantissa), exponent.abs())
    }
}

/// `_selector_number`: fixed at the precision, `.6g` past it or past 1e9.
pub fn selector_number(value: f64, precision: i64) -> String {
    if precision <= 6 && value.abs() < 1.0e9 {
        format_fixed(value, precision.max(0) as usize)
    } else {
        format_general6(value)
    }
}

/// The readout text over the selected bar, for linear coordinate scales:
/// "(x, y, z)" at the precisions of the extent's spans and the limits.
pub fn readout_text(scene: &HeightBarScene, frame: &DataFrame, x: f64, y: f64, z: f64) -> Result<String, String> {
    let (left, right, bottom, top) = frame.extent;
    let xp = selector_precision((right - left).abs())?;
    let yp = selector_precision((top - bottom).abs())?;
    let zp = selector_precision((scene.value_high - scene.value_low).abs())?;
    Ok(format!("({}, {}, {})", selector_number(x, xp), selector_number(y, yp), selector_number(z, zp)))
}
