//! The Image kind's height-bar presentation (`zlc_plot._height3d_raster`).
//!
//! One (ny, nx) value grid drawn as shaded, outlined boxes standing on a
//! background floor between two background back panes, under an orthographic
//! camera whose z axis is vertical on screen. Every pixel column therefore
//! shares one footprint line through the ground grid, and the scene is a
//! per-column front-to-back walk ([`kernel`]); the creases of the drawn
//! surface are stroked from the face-id plane ([`rims`]); the room's vector
//! chrome is pure geometry over the published [`HeightBarScene`] ([`chrome`]).
//!
//! The Python module has two engines -- the numpy reference and the numba
//! scan-line walk -- bound bit-identical by contract. This port follows the
//! scan-line walk operation for operation, which is the reference's own
//! operation order in loop form, and its goldens are checked against both.

pub mod chrome;
mod kernel;
pub mod pymath;
pub mod rims;
pub mod ticks;

use std::sync::Arc;

use rayon::prelude::*;

use kernel::{CellZ, Columns, Kernel};
use pymath::{np_clip, np_maximum, np_minimum, py_floordiv, py_hypot, py_max, py_min, py_mod, radians};

/// The presentation camera: orbit angles plus a zoom factor, clamped as the
/// Python dataclass clamps them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeightBarCamera {
    pub azimuth_deg: f64,
    pub elevation_deg: f64,
    pub zoom: f64,
}

impl HeightBarCamera {
    pub fn new(azimuth_deg: f64, elevation_deg: f64, zoom: f64) -> Result<Self, String> {
        for (name, value) in [("azimuth_deg", azimuth_deg), ("elevation_deg", elevation_deg), ("zoom", zoom)] {
            if !value.is_finite() {
                return Err(format!("{name} must be a finite number"));
            }
        }
        Ok(Self {
            azimuth_deg,
            elevation_deg: py_min(py_max(elevation_deg, 8.0), 80.0),
            zoom: py_min(py_max(zoom, 0.4), 6.0),
        })
    }
}

impl Default for HeightBarCamera {
    /// The home view.
    fn default() -> Self {
        Self { azimuth_deg: 55.0, elevation_deg: 30.0, zoom: 1.0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Lower,
    Upper,
}

/// Everything `render_height_bars` takes. `heights` is row-major
/// `rows × columns`; NaN (or any non-finite value) is an absent bar.
/// `colours` is the colormap's 256-row float32 table.
#[derive(Clone, Copy)]
pub struct HeightBarRequest<'a> {
    pub heights: &'a [f64],
    pub rows: usize,
    pub columns: usize,
    pub colours: &'a [[f32; 3]],
    pub camera: HeightBarCamera,
    pub value_limits: (f64, f64),
    pub width: usize,
    pub height: usize,
    pub supersample: usize,
    pub side_shades: (f64, f64),
    pub background_rgb: [f64; 3],
    pub zero_rgb: Option<[f64; 3]>,
    pub z_fraction: f64,
    pub rim_rgb: [f64; 3],
    pub rim_width_px: f64,
    pub origin: Origin,
    pub inset_px: (f64, f64),
    /// Caller identity of `heights` + `colours`: the same key promises the
    /// same data, and lets a renderer reuse the derived planes across
    /// camera moves (Python keys them on the arrays' identity). `None`
    /// derives every frame.
    pub input_key: Option<u64>,
}

impl<'a> HeightBarRequest<'a> {
    /// The Python keyword defaults.
    pub fn new(heights: &'a [f64], rows: usize, columns: usize, colours: &'a [[f32; 3]], width: usize, height: usize) -> Self {
        Self {
            heights,
            rows,
            columns,
            colours,
            camera: HeightBarCamera::default(),
            value_limits: (0.0, 1.0),
            width,
            height,
            supersample: 1,
            side_shades: (1.0, 1.0),
            background_rgb: [1.0, 1.0, 1.0],
            zero_rgb: None,
            z_fraction: 0.55,
            rim_rgb: [0.0, 0.0, 0.0],
            rim_width_px: 0.0,
            origin: Origin::Lower,
            inset_px: (0.0, 0.0),
            input_key: None,
        }
    }
}

/// Everything an interaction needs to read one rendered scene. `project`
/// speaks FOLDED ground coordinates; `pick` the caller's original indices.
#[derive(Clone, Debug)]
pub struct HeightBarScene {
    pub quadrant: i64,
    pub ca: f64,
    pub sa: f64,
    pub se: f64,
    pub ce: f64,
    pub scale: f64,
    pub x_low: f64,
    pub y_high: f64,
    pub nx: usize,
    pub ny: usize,
    pub source_nx: usize,
    pub source_ny: usize,
    pub flip_rows: bool,
    pub z_unit: f64,
    pub value_low: f64,
    pub value_high: f64,
    pub width: usize,
    pub height: usize,
    /// (height × width) face ids: 0 sky, 1 floor, 2 panes, bars from 4.
    pub id_plane: Arc<Vec<i32>>,
    /// Folded (ny × nx) drawn top height per cell (-inf where no bar).
    pub top_values: Arc<Vec<f64>>,
    /// Folded (ny × nx) drawn base height per cell (+inf where no bar).
    pub base_values: Arc<Vec<f64>>,
}

impl HeightBarScene {
    /// Folded ground point + value -> pixel (x, y), y down.
    pub fn project(&self, a: f64, b: f64, value: f64) -> (f64, f64) {
        let sx = a * self.ca + b * self.sa;
        let sy = -a * self.sa * self.se + b * self.ca * self.se + value * self.z_unit * self.ce;
        ((sx - self.x_low) * self.scale, (self.y_high - sy) * self.scale)
    }

    /// Original grid indices -> folded (a_cell, b_cell).
    pub fn fold_cell(&self, row: i64, column: i64) -> (i64, i64) {
        let (col_p, mut row_p) = (column, row);
        if self.flip_rows {
            row_p = self.source_ny as i64 - 1 - row_p;
        }
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        match self.quadrant {
            0 => (col_p, row_p),
            1 => (row_p, ny - 1 - col_p),
            2 => (nx - 1 - col_p, ny - 1 - row_p),
            _ => (nx - 1 - row_p, col_p),
        }
    }

    /// Folded (a_cell, b_cell) -> the caller's (row, column).
    pub fn unfold_cell(&self, a: i64, b: i64) -> (i64, i64) {
        let (nx, ny) = (self.nx as i64, self.ny as i64);
        let (mut row_p, col_p) = match self.quadrant {
            0 => (b, a),
            1 => (a, ny - 1 - b),
            2 => (ny - 1 - b, nx - 1 - a),
            _ => (nx - 1 - a, b),
        };
        if self.flip_rows {
            row_p = self.source_ny as i64 - 1 - row_p;
        }
        (row_p, col_p)
    }

    /// Pixel -> original (row, column) of the bar drawn there.
    pub fn pick(&self, x: f64, y: f64) -> Option<(i64, i64)> {
        if !(x.is_finite() && y.is_finite()) {
            return None; // Python's int() raises here
        }
        let (column, row) = (x.trunc(), y.trunc());
        if !(0.0 <= row && row < self.height as f64 && 0.0 <= column && column < self.width as f64) {
            return None;
        }
        let face = self.id_plane[row as usize * self.width + column as usize] as i64;
        if face < 4 {
            return None;
        }
        let cell = (face - 4).div_euclid(4);
        let nx = self.nx as i64;
        Some(self.unfold_cell(cell.rem_euclid(nx), cell.div_euclid(nx)))
    }
}

/// One finished frame: (height × width × 4) opaque RGBA and its scene.
#[derive(Clone)]
pub struct HeightBarFrame {
    pub rgba: Arc<Vec<u8>>,
    pub scene: HeightBarScene,
}

/// The input-derived planes (folded), cached per input/limits/quadrant.
struct Derived {
    hz: Vec<f64>,
    finite: Vec<bool>,
    rgb: Vec<[f32; 3]>,
    top_values: Arc<Vec<f64>>,
    base_values: Arc<Vec<f64>>,
}

#[derive(Clone, Copy, PartialEq)]
struct DerivedKey {
    input: u64,
    flip_rows: bool,
    quadrant: i64,
    value_low: u64,
    value_high: u64,
    zero_rgb: Option<[u64; 3]>,
    z_fraction: u64,
    shape: (usize, usize),
}

/// A renderer that keeps what Python's `render_cache` keeps: the derived
/// planes across camera moves, the elevation-scaled planes across azimuth
/// and zoom moves, and two frame/id buffers it alternates between (reused
/// only when the frame published two renders ago has been dropped). Planes
/// a new derive replaces lend it their allocations once nothing holds them:
/// every element is written again, and fresh pages are what a live panel's
/// new shot would otherwise pay for.
#[derive(Default)]
pub struct HeightBarRenderer {
    derived: Option<(Option<DerivedKey>, Arc<Derived>)>,
    z_planes: Option<(Option<(DerivedKey, u64)>, Arc<Vec<CellZ>>)>,
    frames: [Option<Arc<Vec<u8>>>; 2],
    ids: [Option<Arc<Vec<i32>>>; 2],
    parity: usize,
}

fn buffer<T: Copy + Default>(slot: &mut Option<Arc<Vec<T>>>, len: usize) -> Arc<Vec<T>> {
    if let Some(existing) = slot.take() {
        let mut existing = existing;
        if existing.len() == len && Arc::get_mut(&mut existing).is_some() {
            return existing;
        }
    }
    Arc::new(vec![T::default(); len])
}

/// Allocations to write a derive into (empty when none can be lent).
#[derive(Default)]
struct Spare {
    hz: Vec<f64>,
    finite: Vec<bool>,
    rgb: Vec<[f32; 3]>,
    top: Vec<f64>,
    base: Vec<f64>,
    cells: Vec<CellZ>,
}

fn lend<T>(slot: Option<Arc<T>>) -> Option<T> {
    slot.and_then(|value| Arc::try_unwrap(value).ok())
}

/// An empty vector with room for `n`, reusing `spare`'s allocation.
fn room<T>(mut spare: Vec<T>, n: usize) -> Vec<T> {
    spare.clear();
    spare.reserve_exact(n);
    spare
}

/// Raw output planes for a parallel fill that writes every element once.
#[derive(Clone, Copy)]
struct Planes {
    hz: *mut f64,
    finite: *mut bool,
    rgb: *mut [f32; 3],
    top: *mut f64,
    base: *mut f64,
    cells: *mut CellZ,
}
// SAFETY: every task writes only its own rows.
unsafe impl Send for Planes {}
unsafe impl Sync for Planes {}

/// Folded rows a fill task owns: a fold that transposes reads a source row
/// across these, so they share its cache lines.
const FILL_ROWS: usize = 16;

/// The derivation of `_derive_planes` and `_derive_z_planes` in one pass:
/// clip, the finite branch, the colour code the heatmap picks, and the two
/// elevation-scaled planes.
#[allow(clippy::too_many_arguments)]
fn derive(
    request: &HeightBarRequest<'_>, flip_rows: bool, quadrant: i64, value_low: f64, value_high: f64, z_unit: f64,
    ce: f64, spare: Spare,
) -> (Derived, Vec<CellZ>) {
    let (rows, columns) = (request.rows, request.columns);
    let fold = quadrant.rem_euclid(4);
    let (ny, nx) = if fold % 2 == 1 { (columns, rows) } else { (rows, columns) };
    let span = 256.0 / (value_high - value_low);
    let lut = request.colours;
    let heights = request.heights;
    let n = nx * ny;
    let mut hz = room(spare.hz, n);
    let mut finite = room(spare.finite, n);
    let mut rgb = room(spare.rgb, n);
    let mut top = room(spare.top, n);
    let mut base = room(spare.base, n);
    let mut cells = room(spare.cells, n);
    let planes = Planes {
        hz: hz.as_mut_ptr(),
        finite: finite.as_mut_ptr(),
        rgb: rgb.as_mut_ptr(),
        top: top.as_mut_ptr(),
        base: base.as_mut_ptr(),
        cells: cells.as_mut_ptr(),
    };
    let fill = |task: usize| {
        let planes = planes; // the whole struct: its fields alone are not Sync
        let (b0, b1) = (task * FILL_ROWS, ((task + 1) * FILL_ROWS).min(ny));
        let cell = |b: usize, a: usize| {
            let (row_p, column) = match fold {
                0 => (b, a),
                1 => (a, columns - 1 - b),
                2 => (rows - 1 - b, columns - 1 - a),
                _ => (rows - 1 - a, b),
            };
            let row = if flip_rows { rows - 1 - row_p } else { row_p };
            let value = heights[row * columns + column];
            let mut clipped = value;
            if clipped < value_low {
                clipped = value_low;
            }
            if clipped > value_high {
                clipped = value_high;
            }
            let ok = value.is_finite();
            let (h, t, s) = if ok {
                (clipped * z_unit, if clipped > 0.0 { clipped } else { 0.0 }, if clipped < 0.0 { clipped } else { 0.0 })
            } else {
                (0.0, f64::NEG_INFINITY, f64::INFINITY)
            };
            let mut code_value = (clipped - value_low) * span;
            let code = if code_value.is_nan() {
                0
            } else {
                if code_value < 0.0 {
                    code_value = 0.0;
                } else if code_value > 255.0 {
                    code_value = 255.0;
                }
                code_value as usize
            };
            let scaled = h * ce;
            let z = CellZ {
                top: (if scaled > 0.0 { scaled } else { 0.0 }) as f32,
                bot: (if scaled < 0.0 { scaled } else { 0.0 }) as f32,
                finite: ok,
            };
            let index = b * nx + a;
            // SAFETY: index < n lies in rows [b0, b1), which this task owns.
            unsafe {
                planes.hz.add(index).write(h);
                planes.finite.add(index).write(ok);
                planes.rgb.add(index).write(lut[code]);
                planes.top.add(index).write(t);
                planes.base.add(index).write(s);
                planes.cells.add(index).write(z);
            }
        };
        if fold % 2 == 1 {
            for a in 0..nx {
                for b in b0..b1 {
                    cell(b, a);
                }
            }
        } else {
            for b in b0..b1 {
                for a in 0..nx {
                    cell(b, a);
                }
            }
        }
    };
    (0..ny.div_ceil(FILL_ROWS)).into_par_iter().for_each(fill);
    // SAFETY: the fill wrote every one of the n elements of each plane.
    unsafe {
        hz.set_len(n);
        finite.set_len(n);
        rgb.set_len(n);
        top.set_len(n);
        base.set_len(n);
        cells.set_len(n);
    }
    (Derived { hz, finite, rgb, top_values: Arc::new(top), base_values: Arc::new(base) }, cells)
}

/// `_derive_z_planes` alone: an elevation drag over cached derived planes.
fn z_planes(derived: &Derived, ce: f64, spare: Vec<CellZ>) -> Vec<CellZ> {
    let n = derived.hz.len();
    let mut cells = room(spare, n);
    cells.par_extend(derived.hz.par_iter().zip(derived.finite.par_iter()).map(|(&hz, &finite)| {
        let scaled = hz * ce;
        CellZ {
            top: (if scaled > 0.0 { scaled } else { 0.0 }) as f32,
            bot: (if scaled < 0.0 { scaled } else { 0.0 }) as f32,
            finite,
        }
    }));
    cells
}

// ---- what `_update_height_bars_artist` decides before it renders ----------

/// `_image_color_lut_rgb`: the colormap's uint8 table as float32 RGB.
pub fn colour_table(lut: &[[u8; 3]]) -> Vec<[f32; 3]> {
    lut.iter().map(|rgb| rgb.map(|v| v as f32 / 255.0f32)).collect()
}

/// The colour every side face's zero end takes: the table row the value 0
/// lands on under the limits, as float32 read back to float.
pub fn zero_rgb(lut: &[[u8; 3]], low: f64, high: f64) -> Result<[f64; 3], String> {
    let code = np_clip((0.0 - low) * (256.0 / (high - low)), 0.0, 255.0);
    if code.is_nan() {
        return Err("zero code is not a number".into());
    }
    Ok(lut[code as usize].map(|v| (v as f32 / 255.0f32) as f64))
}

/// The bar limits: valid colour limits, else the data range (or [0, 1]),
/// widened to one unit when empty.
pub fn bar_limits(color_limits: Option<(f64, f64)>, data_range: Option<(f64, f64)>) -> (f64, f64) {
    if let Some((low, high)) = color_limits {
        if high > low {
            return (low, high);
        }
    }
    let (low, mut high) = data_range.unwrap_or((0.0, 1.0));
    if high <= low {
        high = low + 1.0;
    }
    (low, high)
}

/// `max(int(round(extent)), 8)`: a box side in raster pixels.
pub fn box_pixels(extent: f64) -> usize {
    (pymath::py_round(extent) as i64).max(8) as usize
}

/// Horizontal taps: three per LOGICAL pixel, and no finer than a bar shows.
pub fn frame_supersample(device_pixel_ratio: f64, box_w: usize, box_h: usize, rows: usize, columns: usize) -> usize {
    let screen_taps = pymath::py_round(3.0 / py_max(device_pixel_ratio, 1e-9)) as i64;
    let bar_px = box_w.min(box_h) as f64 / py_max(1.0, pymath::np_hypot(columns as f64, rows as f64));
    let bar = match pymath::trunc_i64(bar_px) {
        0 => 1,
        value => value,
    };
    4.min(screen_taps).min(bar).max(1) as usize
}

/// The rim width in raster pixels: `axes.linewidth` points at the dpi.
pub fn rim_width_px(linewidth_pt: f64, dpi: f64) -> f64 {
    linewidth_pt * dpi / 72.0
}

/// The source indices a scene axis ticks: `round` of a six-point linspace.
pub fn coordinate_tick_indices(count: usize) -> Vec<i64> {
    let points = count.min(6);
    let stop = count as f64 - 1.0;
    let mut indices: Vec<i64> = (0..points)
        .map(|n| {
            let value = if points == 1 {
                0.0
            } else if n == points - 1 {
                stop
            } else {
                n as f64 * (stop / (points as f64 - 1.0)) + 0.0
            };
            pymath::py_round(value) as i64
        })
        .collect();
    indices.sort_unstable();
    indices.dedup();
    indices
}

/// The coordinate a scene tick labels on a LINEAR axis: the centre of
/// source cell `index` of `count` between the extent's `start` and `end`
/// (the caller applies `axis_value` for any other scale).
pub fn coordinate_tick_value(index: i64, start: f64, end: f64, count: usize) -> f64 {
    start + (index as f64 + 0.5) * (end - start) / count as f64
}

/// Render once, without keeping anything for the next frame.
pub fn render_height_bars(request: &HeightBarRequest<'_>) -> Result<HeightBarFrame, String> {
    HeightBarRenderer::default().render(request)
}

impl HeightBarRenderer {
    pub fn new() -> Self {
        Self::default()
    }

    /// `render_height_bars`: the grid as boxes -> (RGBA frame, scene map).
    #[allow(clippy::too_many_lines)]
    pub fn render(&mut self, request: &HeightBarRequest<'_>) -> Result<HeightBarFrame, String> {
        let supersample = request.supersample;
        if !(1..=4).contains(&supersample) {
            return Err("supersample must be 1, 2, 3 or 4".into());
        }
        let flip_rows = request.origin == Origin::Upper;
        let render_w = request.width * supersample;
        let render_h = request.height * supersample;
        if render_w < 8 || render_h < 8 {
            return Err("height-bar raster needs at least 8x8 pixels".into());
        }
        if request.colours.len() != 256 || request.rows == 0 || request.columns == 0
            || request.heights.len() != request.rows * request.columns
        {
            return Err("heights must be (ny, nx) and colours (256, 3)".into());
        }
        let (source_ny, source_nx) = (request.rows, request.columns);

        // ---- fold the azimuth into [0, 90) by rotating the grid
        let azimuth = py_mod(radians(request.camera.azimuth_deg), 2.0 * std::f64::consts::PI);
        let quarter = std::f64::consts::PI / 2.0;
        let quadrant = py_floordiv(azimuth, quarter) as i64;
        let local = azimuth - quadrant as f64 * quarter;
        let (ny, nx) = if quadrant.rem_euclid(4) % 2 == 1 { (source_nx, source_ny) } else { (source_ny, source_nx) };
        let ca = py_max(local.cos(), 1e-9);
        let sa = py_max(local.sin(), 1e-9);
        let elevation = radians(request.camera.elevation_deg);
        let (se, ce) = (elevation.sin(), elevation.cos());

        let (mut value_low, mut value_high) = request.value_limits;
        if !(value_low.is_finite() && value_high.is_finite()) {
            (value_low, value_high) = (0.0, 1.0);
        }
        if value_high <= value_low {
            value_high = value_low + 1.0;
        }
        let mut magnitude = py_max(value_low.abs(), value_high.abs());
        if magnitude <= 0.0 {
            magnitude = 1.0;
        }
        let z_unit = request.z_fraction * nx.max(ny) as f64 / magnitude;
        let pane_high = py_max(value_high, 0.0) * z_unit * ce;
        let pane_low = py_min(value_low, 0.0) * z_unit * ce;

        // ---- screen frame (y up during the math), fit-to-box then zoom
        let (ux, uy) = (ca, -sa * se);
        let (vx, vy) = (sa, ca * se);
        let (nxf, nyf) = (nx as f64, ny as f64);
        let ground_x = [0.0, nxf * ux, nyf * vx, nxf * ux + nyf * vx];
        let ground_y = [0.0, nxf * uy, nyf * vy, nxf * uy + nyf * vy];
        let min4 = |v: [f64; 4]| v.iter().copied().fold(f64::INFINITY, f64::min);
        let max4 = |v: [f64; 4]| v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut x_low = min4(ground_x);
        let x_high = max4(ground_x);
        let y_low = min4(ground_y) + pane_low;
        let mut y_high = max4(ground_y) + pane_high;
        let diagonal = py_hypot(nxf, nyf);
        let span_x = diagonal;
        let span_y = diagonal * se + pane_high - pane_low;
        let margin = 0.04 * py_max(span_x, span_y);
        let inner_w = py_max(1.0, render_w as f64 - 2.0 * request.inset_px.0 * supersample as f64);
        let inner_h = py_max(1.0, render_h as f64 - 2.0 * request.inset_px.1 * supersample as f64);
        let scale = py_min(inner_w / (span_x + 2.0 * margin), inner_h / (span_y + 2.0 * margin)) * request.camera.zoom;
        let x_mid = 0.5 * (x_low + x_high);
        let y_mid = 0.5 * (y_low + y_high);
        x_low = x_mid - 0.5 * render_w as f64 / scale;
        y_high = y_mid + 0.5 * render_h as f64 / scale;

        // ---- per-column footprint lines: a*ca + b*sa = X, direction (-sa, ca)
        let mut columns = Columns {
            a_at: vec![0.0; render_w],
            t_enter: vec![0.0; render_w],
            t_exit: vec![0.0; render_w],
            ia0: vec![0.0; render_w],
            ib0: vec![0.0; render_w],
            t_a0: vec![0.0; render_w],
            t_b0: vec![0.0; render_w],
            g0: vec![0.0; render_w],
            k_over_sa: (0..nx).map(|k| k as f64 / sa).collect(),
            m_over_ca: (0..ny).map(|m| m as f64 / ca).collect(),
        };
        for c in 0..render_w {
            let x = (c as f64 + 0.5) / scale + x_low;
            let a0 = x / ca;
            let t_enter = np_maximum((a0 - nxf) / sa, 0.0);
            let t_exit = np_maximum(np_minimum(a0 / sa, nyf / ca), t_enter);
            let a_at = a0 - t_enter * sa;
            let b_at = t_enter * ca;
            let ia0 = np_clip(a_at.ceil() - 1.0, 0.0, nxf - 1.0);
            let ib0 = np_clip(b_at.floor(), 0.0, nyf - 1.0);
            columns.a_at[c] = a_at;
            columns.t_enter[c] = t_enter;
            columns.t_exit[c] = t_exit;
            columns.ia0[c] = ia0;
            columns.ib0[c] = ib0;
            columns.t_a0[c] = t_enter + (a_at - ia0) / sa;
            columns.t_b0[c] = t_enter + ((ib0 + 1.0) - b_at) / ca;
            columns.g0[c] = (a0 * uy) as f32;
        }

        // ---- derived planes (cached on inputs/limits/quadrant), then the
        // elevation-scaled z planes (cached on those and ce)
        let key = DerivedKey {
            input: request.input_key.unwrap_or(0),
            flip_rows,
            quadrant,
            value_low: value_low.to_bits(),
            value_high: value_high.to_bits(),
            zero_rgb: request.zero_rgb.map(|v| v.map(f64::to_bits)),
            z_fraction: request.z_fraction.to_bits(),
            shape: (request.rows, request.columns),
        };
        let cacheable = request.input_key.is_some();
        let z_key = (key, ce.to_bits());
        let derived_hit = matches!(&self.derived, Some((Some(cached), _)) if cacheable && *cached == key);
        let z_hit = matches!(&self.z_planes, Some((Some(cached), _)) if cacheable && *cached == z_key);
        let (derived, cells) = if derived_hit {
            let derived = self.derived.as_ref().expect("cached derive").1.clone();
            let cells = if z_hit {
                self.z_planes.as_ref().expect("cached z planes").1.clone()
            } else {
                let spare = lend(self.z_planes.take().map(|(_, cells)| cells)).unwrap_or_default();
                Arc::new(z_planes(&derived, ce, spare))
            };
            (derived, cells)
        } else {
            let mut spare = Spare::default();
            if let Some(old) = lend(self.derived.take().map(|(_, derived)| derived)) {
                spare.hz = old.hz;
                spare.finite = old.finite;
                spare.rgb = old.rgb;
                spare.top = lend(Some(old.top_values)).unwrap_or_default();
                spare.base = lend(Some(old.base_values)).unwrap_or_default();
            }
            spare.cells = lend(self.z_planes.take().map(|(_, cells)| cells)).unwrap_or_default();
            let (derived, cells) = derive(request, flip_rows, quadrant, value_low, value_high, z_unit, ce, spare);
            (Arc::new(derived), Arc::new(cells))
        };
        self.derived = Some((cacheable.then_some(key), derived.clone()));
        self.z_planes = Some((cacheable.then_some(z_key), cells.clone()));

        let out_h = render_h / supersample;
        let out_w = request.width;
        self.parity ^= 1;
        let mut rgba = buffer(&mut self.frames[self.parity], out_h * out_w * 4);
        let mut ids = buffer(&mut self.ids[self.parity], out_h * out_w);
        let background32 = request.background_rgb.map(|v| v as f32);
        let kernel = Kernel {
            columns: &columns,
            cells: &cells,
            rgb: &derived.rgb,
            base: request.zero_rgb.map(|v| v.map(|c| c as f32)),
            nx,
            ny,
            shade_x: request.side_shades.0 as f32,
            shade_y: request.side_shades.1 as f32,
            ca,
            se32: se as f32,
            y_high,
            scale,
            pane_high32: pane_high as f32,
            pane_low32: py_min(pane_low, 0.0) as f32,
            background32,
            taps: supersample,
            render_h,
            out_h,
            out_w,
            mid_tap: supersample / 2,
        };
        {
            let rgba = Arc::get_mut(&mut rgba).expect("fresh frame buffer");
            let ids = Arc::get_mut(&mut ids).expect("fresh id buffer");
            kernel.run(rgba, ids);

            // ---- the creases of the drawn surface, drawn into it: whatever
            // width the bar can spare, up to the full line width.
            let scale_out = scale / supersample as f64;
            let step_a = py_hypot(ca, sa * se);
            let step_b = py_hypot(sa, ca * se);
            let bar_px = scale_out * py_min(step_a, step_b);
            let crease_px = py_min(request.rim_width_px, py_max(0.0, bar_px - 1.0));
            rims::stroke_rims(rgba, ids, out_w, out_h, request.rim_rgb, crease_px);
        }
        self.frames[self.parity] = Some(rgba.clone());
        self.ids[self.parity] = Some(ids.clone());

        let scene = HeightBarScene {
            quadrant,
            ca,
            sa,
            se,
            ce,
            scale: scale / supersample as f64,
            x_low,
            y_high,
            nx,
            ny,
            source_nx,
            source_ny,
            flip_rows,
            z_unit,
            value_low,
            value_high,
            width: out_w,
            height: request.height,
            id_plane: ids,
            top_values: derived.top_values.clone(),
            base_values: derived.base_values.clone(),
        };
        Ok(HeightBarFrame { rgba, scene })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The AVX2 walk and the portable walk are one computation per lane.
    #[test]
    fn simd_and_portable_walks_agree() {
        use std::sync::atomic::Ordering;
        let (rows, columns) = (23usize, 31usize);
        let heights: Vec<f64> = (0..rows * columns)
            .map(|i| if i % 17 == 3 { f64::NAN } else { ((i * 7919) % 1000) as f64 / 400.0 - 0.6 })
            .collect();
        let table: Vec<[f32; 3]> = (0..256).map(|i| [i as f32 / 255.0, 0.25, 1.0 - i as f32 / 255.0]).collect();
        for (supersample, azimuth, rim) in [(1, 55.0, 2.0), (2, 140.0, 0.0), (3, 260.0, 1.3), (4, 330.0, 3.0)] {
            let request = HeightBarRequest {
                camera: HeightBarCamera::new(azimuth, 35.0, 1.3).unwrap(),
                value_limits: (-0.5, 2.0),
                supersample,
                zero_rgb: Some([0.5, 0.5, 0.5]),
                rim_width_px: rim,
                side_shades: (0.9, 0.75),
                background_rgb: [0.97, 0.98, 1.0],
                ..HeightBarRequest::new(&heights, rows, columns, &table, 203, 141)
            };
            kernel::FORCE_PORTABLE.store(true, Ordering::Relaxed);
            let portable = render_height_bars(&request).unwrap();
            kernel::FORCE_PORTABLE.store(false, Ordering::Relaxed);
            let simd = render_height_bars(&request).unwrap();
            assert_eq!(portable.rgba, simd.rgba, "supersample {supersample}");
            assert_eq!(portable.scene.id_plane, simd.scene.id_plane, "supersample {supersample}");
        }
    }
}
