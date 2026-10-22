//! The FacetGrid kind (`_update_facets`) and the grid's chrome
//! (`_sync_axes_chrome`'s overview branch, `_refresh_facet_cell_chrome`).
//!
//! A grid is not one surface: every visible cell is drawn by the cell kind's
//! own machinery against limits pooled over the grid. In the overview the
//! cells' Axis objects keep the tick policy and draw nothing; their marks and
//! frames are strokes on the cell ([`Stroke`]), their titles and boundary
//! labels figure artists. One cell opened is a standalone surface of the
//! cell's kind over the grid's region.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use zlc_figure::ir::{HAlign, VAlign};

use super::core::{Kind, Renderer};
use super::curve::{apply_smart_ticks, curve_series, prepare_curve_series, FacetCommand, PreparedCurve, PreparedSeries};
use super::limits::{curve_x_limits, data_limits};
use super::mpl::axes::{AxesModel, AxisModel, Child, ChildId, FrameArtist, MarkArtist, Stroke, TickGeometry, TickParams, TickPolicy, SPINES};
use super::hashkey::Key;
use super::mpl::figure::{FigureArtist, FigureFrame, FigureText, LabelRef};
use crate::configuration::error::{PlotError, Result};
use crate::configuration::spec::CellPlot;
use crate::configuration::{DisplayState, PlotSpec, Value};
use crate::layout::Room;
use crate::projection::nd::{Array, DType, Scalar};
use super::frame::ViewLimits;
use super::geometry::{centers_extent, image_coordinate_scale};
use super::histogram::{histogram_arrays, histogram_payload, HistogramArrays};
use super::image::{all_valid, colormap_name, ImageTables};
use super::scale::AxisScale;
use crate::projection::output::{FacetCell, Payload};
use crate::session::FitOverlay;
use crate::ticks::compact_number;

/// The FacetGrid kind: its render resolves the cells from the painted
/// surfaces and draws each with the cell kind.
pub struct FacetKind;

impl Kind for FacetKind {
    fn render(&self, renderer: &mut Renderer, payload: &Arc<Payload>, state: &DisplayState, _axes: usize, _key: &str) {
        renderer.update_facets(payload, state);
    }
    fn key(&self) -> &'static str { "facet_grid" }
}

/// Whether the native renderer draws grids of this cell kind.
pub fn draws_cell(cell: &CellPlot) -> bool { matches!(cell, CellPlot::Curve(_) | CellPlot::Image(_) | CellPlot::Histogram(_)) }

/// The grid's prepared image scene (`image:prepared` of a grid): every
/// visible cell's plane, stacked, at the grid's one colour scale.
#[derive(Clone, Debug)]
pub struct FacetImages {
    pub values: Array,
    /// None: every sample of every cell valid.
    pub valid: Option<Array>,
    pub extents: Vec<[f64; 4]>,
    pub color_limits: (f64, f64),
    pub cmap: &'static str,
    pub show: bool,
    /// The cells it materializes from, and their state.
    pub payload: Arc<Payload>,
    pub state: Arc<DisplayState>,
}

/// What a cell's own render is a function of besides its data (`image:prepared_signature`).
#[derive(Clone, Debug, PartialEq)]
pub struct ImageSignature {
    revision: u64,
    extents: Vec<[f64; 4]>,
    view: Option<ViewLimits>,
    scales: Vec<(AxisScale, AxisScale)>,
}

/// One overview ellipse fit for the kernel: its centre and radii in canvas
/// pixels (y down), ring and centre colours and sizes, and its cell's clip.
struct FitEllipse {
    geometry: [f64; 4],
    ring: [u8; 4],
    ring_width: f64,
    center: [u8; 4],
    center_radius: f64,
    clip: [i64; 4],
}

/// `ellipse_boundary_distance(dx, dy, radius_x, radius_y)`: the distance from
/// a point (relative to the centre) to an axis-aligned ellipse's boundary --
/// Eberly's bracketed bisection in the normalised root variable.
fn ellipse_boundary_distance(dx: f64, dy: f64, radius_x: f64, radius_y: f64) -> f64 {
    let (mut y0, mut y1, mut e0, mut e1) = (dx.abs(), dy.abs(), radius_x, radius_y);
    if e0 < e1 {
        std::mem::swap(&mut y0, &mut y1);
        std::mem::swap(&mut e0, &mut e1);
    }
    if y1 > 0.0 {
        if y0 > 0.0 {
            let (z0, z1) = (y0 / e0, y1 / e1);
            let g = z0 * z0 + z1 * z1 - 1.0;
            if g == 0.0 { return 0.0; }
            let r0 = (e0 / e1) * (e0 / e1);
            let n0 = r0 * z0;
            let mut s0 = z1 - 1.0;
            let mut s1 = if g < 0.0 { 0.0 } else { (n0 * n0 + z1 * z1).sqrt() - 1.0 };
            let mut s = 0.0;
            for _ in 0..64 {
                s = 0.5 * (s0 + s1);
                if s == s0 || s == s1 { break; }
                let ratio0 = n0 / (s + r0);
                let ratio1 = z1 / (s + 1.0);
                let g = ratio0 * ratio0 + ratio1 * ratio1 - 1.0;
                if g > 0.0 { s0 = s; } else if g < 0.0 { s1 = s; } else { break; }
            }
            let x0 = r0 * y0 / (s + r0);
            let x1 = y1 / (s + 1.0);
            return ((x0 - y0) * (x0 - y0) + (x1 - y1) * (x1 - y1)).sqrt();
        }
        return (y1 - e1).abs();
    }
    // On the major axis the nearest point is the vertex only outside the
    // evolute's cusp; inside it the foot leaves the axis.
    let numerator = e0 * y0;
    let denominator = e0 * e0 - e1 * e1;
    if numerator < denominator {
        let fraction = numerator / denominator;
        let x0 = e0 * fraction;
        let x1 = e1 * (1.0 - fraction * fraction).sqrt();
        return ((x0 - y0) * (x0 - y0) + x1 * x1).sqrt();
    }
    (y0 - e0).abs()
}

/// Python's `max(a, b)` / `min(a, b)`: the first unless the second is strictly beyond it.
fn py_max(a: f64, b: f64) -> f64 { if b > a { b } else { a } }
fn py_min(a: f64, b: f64) -> f64 { if b < a { b } else { a } }

/// `raster_fit_ellipses`: each ring a stroke of its width centred on the
/// ellipse's boundary (coverage from the pixel centre's distance to it), and
/// a centre dot, blended over an opaque canvas.
fn raster_fit_ellipses(items: &[FitEllipse], out: &mut [u8], width: usize) {
    let blend = |out: &mut [u8], at: usize, colour: [u8; 4], amount: f64| {
        let alpha = colour[3] as f64 / 255.0 * amount;
        let inverse = 1.0 - alpha;
        for channel in 0..3 {
            let value = colour[channel] as f64 * alpha + out[at + channel] as f64 * inverse;
            out[at + channel] = py_min(255.0, (value + 0.5).floor()) as u8;
        }
        out[at + 3] = 255;
    };
    for item in items {
        let [center_x, center_y, rx, ry] = item.geometry;
        let (radius_x, radius_y) = (py_max(0.5, rx), py_max(0.5, ry));
        let ring_radius = py_max(0.5, item.ring_width * 0.5);
        let center_radius = py_max(0.5, item.center_radius);
        let (reach_x, reach_y) = (radius_x + ring_radius + 1.0, radius_y + ring_radius + 1.0);
        let left = item.clip[0].max((center_x - reach_x).floor() as i64);
        let right = item.clip[2].min((center_x + reach_x).ceil() as i64);
        let top = item.clip[1].max((center_y - reach_y).floor() as i64);
        let bottom = item.clip[3].min((center_y + reach_y).ceil() as i64);
        let scale = py_min(radius_x, radius_y);
        for row in top..bottom {
            let py = row as f64 + 0.5;
            for column in left..right {
                let px = column as f64 + 0.5;
                let (dx, dy) = (px - center_x, py - center_y);
                let normalized = ((dx / radius_x) * (dx / radius_x) + (dy / radius_y) * (dy / radius_y)).sqrt();
                let mut distance = (normalized - 1.0).abs() * scale;
                if radius_x != radius_y && distance < ring_radius + 0.5 {
                    distance = ellipse_boundary_distance(dx, dy, radius_x, radius_y);
                }
                let at = (row as usize * width + column as usize) * 4;
                let amount = py_min(1.0, ring_radius + 0.5 - distance);
                if amount > 0.0 { blend(out, at, item.ring, amount); }
                let center_distance = (dx * dx + dy * dy).sqrt();
                let center_amount = py_min(1.0, center_radius + 0.5 - center_distance);
                if center_amount > 0.0 { blend(out, at, item.center, center_amount); }
            }
        }
    }
}

/// Everything `Axis._update_ticks` answers from: the axis' direction, scale
/// and tick parameters, its policy's whole state, its view, and what its
/// painted box gives the ladder (extent, room, dpi).
fn tick_question(axis: &AxisModel, view: (f64, f64), g: &TickGeometry) -> String {
    let size = [g.bbox[2] - g.bbox[0], g.bbox[3] - g.bbox[1], g.dpi].map(f64::to_bits);
    format!("{}|{:?}|{size:?}|{:?}|{:?}|{}|{}|{:?}|{:?}", axis.x, [view.0.to_bits(), view.1.to_bits()], axis.geometry(g),
            axis.function, axis.log, axis.scale_serial, axis.policy, axis.major)
}

/// What `_update_ticks` left on an axis.
#[derive(Clone)]
struct TickAnswer {
    policy: TickPolicy,
    locs: Vec<f64>,
    labels: Vec<String>,
    offset_string: String,
    minor_locs: Vec<f64>,
    major: TickParams,
    offset_size: f64,
}

impl TickAnswer {
    fn of(axis: &AxisModel) -> TickAnswer {
        TickAnswer {
            policy: axis.policy.clone(), locs: axis.locs.clone(), labels: axis.labels.clone(), offset_string: axis.offset_string.clone(),
            minor_locs: axis.minor_locs.clone(), major: axis.major.clone(), offset_size: axis.offset_size,
        }
    }

    /// The answer on another axis that asked the same: its ticks move (its
    /// `majorTicks` keep their own past beyond them).
    fn adopt(&self, axis: &mut AxisModel) {
        axis.policy = self.policy.clone();
        axis.offset_string = self.offset_string.clone();
        axis.minor_locs = self.minor_locs.clone();
        axis.major = self.major.clone();
        axis.offset_size = self.offset_size;
        axis.set_ticks(self.locs.clone(), self.labels.clone());
    }
}

/// `np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8)`.
fn rgba_bytes(rgba: [f64; 4]) -> [u8; 4] {
    rgba.map(|c| (c * 255.0).round_ties_even().clamp(0.0, 255.0) as u8)
}

/// `np.stack(planes)`: one plane of the cells' dtype.
fn stack(planes: &[&Array]) -> Array {
    let (rows, columns) = (planes[0].shape()[0], planes[0].shape()[1]);
    let shape = vec![planes.len(), rows, columns];
    macro_rules! typed { ($t:ty) => { Array::from_vec(planes.iter().flat_map(|p| p.slice::<$t>().iter().copied()).collect::<Vec<$t>>(), shape) }; }
    match planes[0].dtype() {
        DType::Bool => Array::from_vec(planes.iter().flat_map(|p| p.bytes().iter().map(|&b| b != 0)).collect::<Vec<bool>>(), shape),
        DType::U8 => typed!(u8), DType::U16 => typed!(u16), DType::U32 => typed!(u32), DType::U64 => typed!(u64),
        DType::I8 => typed!(i8), DType::I16 => typed!(i16), DType::I32 => typed!(i32), DType::I64 => typed!(i64),
        DType::F32 => typed!(f32), DType::F64 => typed!(f64),
        other => panic!("a grid's image scene stacks no {} plane", other.name()),
    }
}

/// The grid's presentation and chrome (the renderer's facet fields).
#[derive(Clone, Debug, Default)]
pub struct FacetTables {
    /// `_focused_facet_index`: the selected cell.
    pub focused: Option<usize>,
    /// `_facet_focus_index`: the opened cell (None in the overview).
    pub focus: Option<usize>,
    /// `_visible_facet_count`.
    pub visible: usize,
    /// What each visible cell's title says (`_facet_cell_titles`).
    pub cell_titles: BTreeMap<usize, (String, f64)>,
    /// The chrome's figure artists: the cell titles, then the label carrier.
    pub chrome: Option<Vec<u64>>,
    /// What the chrome was built from (`_facet_chrome_signature`).
    pub chrome_signature: Option<u64>,
    /// The outer x and y labels (`facet:outer_x`, `facet:outer_y`).
    pub outer: Option<[u64; 2]>,
    /// The figure title (`figure:title`).
    pub title: Option<u64>,
    /// Every image surface's tables (`_artists[key]`, its range, mapping,
    /// rail and colorbar) but the one swapped in as the renderer's.
    pub images: HashMap<String, ImageTables>,
    /// Which surface's tables the renderer holds now.
    pub image_key: Option<String>,
    pub image_scene: Option<FacetImages>,
    pub image_signature: Option<ImageSignature>,
    /// `_facet_focus_chrome_index`: the cell owning the side rail and colorbar.
    pub focus_chrome: Option<usize>,
    /// `facet:fit_native`: an overview's fits, painted by the kernels.
    pub fit_native: Option<Vec<FitOverlay>>,
}

/// `_facet_cell_title(cell, fallback)`: the projected identity with a compact numeric coordinate.
pub fn facet_cell_title(cell: &FacetCell) -> String {
    let mut title = cell.label.clone();
    let numeric = match &cell.facet_value_display {
        Scalar::Int(v) => Some((*v as f64, v.to_string())),
        Scalar::Float(v) => Some((*v, Value::Float(*v).py_repr())),
        _ => None,
    };
    if let Some((value, raw)) = numeric {
        if value.is_finite() && title.contains(&raw) { title = title.replacen(&raw, &compact_number(value, None), 1); }
    }
    title
}

/// `fitted_facet_cell_title(label, typography, fonts, height_pt=...)` with the
/// renderer's text measure: the text and size a title may occupy.
fn fitted_title(measure: &super::fonts::TextToPath, label: &str, title_pt: f64, min_pt: f64, budget: f64, height_pt: f64)
    -> (String, f64) {
    let size = |text: &str, pt: f64| { let (w, h, _) = measure.width_height_descent(text, pt); (w, h) };
    let (width, height) = size(label, title_pt);
    let mut factor = 1.0f64;
    let by_width = if width > 0.0 { budget / width } else { 1.0 };
    let by_height = if height > 0.0 { height_pt / height } else { 1.0 };
    for candidate in [by_width, by_height] { if candidate < factor { factor = candidate; } }
    if factor == 1.0 { return (label.to_owned(), title_pt); }
    let fitted = title_pt * factor;
    if fitted >= min_pt { return (label.to_owned(), fitted); }
    let chars: Vec<char> = label.chars().collect();
    let shortened = |keep: usize| format!("{}\u{2026}", chars[..keep].iter().collect::<String>().trim_end());
    let (mut fits, mut too_long) = (0usize, chars.len());
    while too_long - fits > 1 {
        let keep = (fits + too_long) / 2;
        let (w, h) = size(&shortened(keep), min_pt);
        if w <= budget && h <= height_pt { fits = keep; } else { too_long = keep; }
    }
    (if fits > 0 { shortened(fits) } else { "\u{2026}".to_owned() }, min_pt)
}

/// A tick's parameters minus the label size (`_tick_params_key`).
#[allow(clippy::type_complexity)]
fn params_key(p: &TickParams) -> (f64, f64, f64, &str, &str, &str, [bool; 5]) {
    (p.size, p.width, p.pad, &p.color, &p.labelcolor, &p.tickdir, [p.tick1on, p.tick2on, p.label1on, p.label2on, p.grid_on])
}

fn rounded(values: &[f64]) -> Vec<u64> { values.iter().map(|v| ((v * 1e6).round_ties_even() / 1e6).to_bits()).collect() }

/// One lane of marks: every mark of one axis that strokes the same way.
#[derive(Clone, Debug)]
struct Lane { x: bool, grid: bool, side: u8, locs: Vec<f64>, color: String, marker: &'static str, size: f64, width: f64 }

/// `Tick` markers: (tick1, tick2) for the axis and direction.
fn tick_markers(x: bool, tickdir: &str) -> (&'static str, &'static str) {
    match (x, tickdir) {
        (true, "in") => ("2", "3"),
        (true, "out") => ("3", "2"),
        (false, "in") => ("1", "0"),
        (false, "out") => ("0", "1"),
        _ => ("|", "|"),
    }
}

impl Renderer {
    pub fn is_facet(&self) -> bool { matches!(self.spec, PlotSpec::FacetGrid(_)) }

    /// A grid showing all its cells.
    pub fn facet_overview(&self) -> bool { self.is_facet() && self.facet.focus.is_none() }

    pub(crate) fn facet_cells(&self) -> Vec<usize> { self.axes_of("facet_cell").to_vec() }

    /// `_consume_facet_presentation`'s refusals.
    pub fn check_facet_presentation(&self, facet_index: Option<usize>, focus: Option<usize>, visible_count: Option<usize>)
        -> Result<()> {
        if !self.is_facet() {
            if facet_index.is_some() || focus.is_some() {
                return Err(PlotError::type_error("facet presentation state requires FacetGridPlot"));
            }
            return Ok(());
        }
        let cells = self.axes_of("facet_cell").len();
        let count = visible_count.unwrap_or(cells);
        if count > cells { return Err(PlotError::value("visible facet count is outside the rendered grid")); }
        match facet_index {
            None => {
                if focus.is_some() { return Err(PlotError::value("an open facet requires a selected facet")); }
                if visible_count.is_some() && count > 0 { return Err(PlotError::value("a non-empty FacetGrid requires a selected facet")); }
            }
            Some(index) => {
                if index >= count { return Err(PlotError::index("facet index is outside the current grid")); }
                if focus.is_some_and(|f| f != index) { return Err(PlotError::value("the open facet must equal the selected facet")); }
            }
        }
        Ok(())
    }

    /// `_consume_facet_presentation`: mirror the session's facet state.
    pub(crate) fn consume_facet_presentation(&mut self, facet_index: Option<usize>, focus: Option<usize>, visible_count: Option<usize>) {
        if !self.is_facet() {
            self.facet.focused = None;
            self.facet.focus = None;
            self.facet.visible = 0;
            return;
        }
        let cells = self.facet_cells();
        if focus != self.facet.focus {
            // A focus change is a relayout: every cell's settled step goes.
            for &cell in &cells {
                for x in [true, false] { self.fig.axes[cell].axis_mut(x).forget_settled_step(); }
            }
        }
        self.facet.focused = facet_index;
        self.facet.focus = focus;
        self.facet.visible = visible_count.unwrap_or(cells.len());
    }

    /// `painted_surfaces` of a grid: the opened cell, or every visible cell.
    pub(crate) fn facet_surfaces(&self) -> Vec<(String, usize, Option<usize>)> {
        let cells = self.axes_of("facet_cell");
        match self.facet.focus {
            Some(index) => vec![(format!("facet:{index}"), cells[index], Some(index))],
            None => (0..self.facet.visible.min(cells.len())).map(|i| (format!("facet:{i}"), cells[i], Some(i))).collect(),
        }
    }

    /// `facet_index_for_axes(axes)`.
    pub fn facet_index_for_axes(&self, axes: usize) -> Option<usize> {
        if !self.is_facet() { return None; }
        self.axes_of("facet_cell").iter().position(|&cell| cell == axes).filter(|&index| index < self.facet.visible)
    }

    /// The cell a selector or a fit names -- its own `facet_index`, else the
    /// selected cell -- as `_selector_axis` and `_fit_target` resolve it.
    pub(crate) fn facet_cell_named(&self, index: Option<usize>, what: &str) -> usize {
        match index.or(self.facet.focused).filter(|&index| index < self.facet.visible) {
            Some(index) => self.axes_of("facet_cell")[index],
            None => panic!("{what} facet index is outside the current grid"),
        }
    }

    /// `_facet_mark_scale()`: how much of its full weight a mark keeps inside a cell.
    pub(crate) fn facet_mark_scale(&self) -> f64 {
        match &self.plan.facet_typography {
            Some(typography) if self.is_facet() => typography.scale,
            _ => 1.0,
        }
    }

    /// `_focused_fit_overlays(overlays)`: a focused grid paints the selected
    /// cell's fits (every group of it) and the grid-wide ones.
    pub(crate) fn focused_fit_overlays(&self, overlays: &[FitOverlay]) -> Vec<FitOverlay> {
        if !self.is_facet() { return overlays.to_vec(); }
        let Some(selected) = self.facet.focused else { return Vec::new() };
        overlays.iter().filter(|overlay| overlay.facet_index.is_none_or(|index| index == selected)).cloned().collect()
    }

    /// The image artist on `axes`, whichever surface's tables hold it
    /// (`_image_selector_artist`).
    pub(crate) fn image_artist_on(&self, axes: usize) -> Option<(usize, ChildId)> {
        self.painted_surfaces().into_iter().find(|(_, owner, _)| *owner == axes).and_then(|(key, _, _)| self.image_of(&key))
    }

    /// `_image_selector_payload(state)`: the picture a selector samples -- a
    /// grid's named (else selected) cell's.
    pub(crate) fn image_selector_payload(&self, facet_index: Option<usize>) -> Option<Arc<Payload>> {
        let payload = self.last_payload.clone()?;
        if !self.is_facet() { return Some(payload); }
        let Payload::Facet(facet) = &*payload else { return None };
        let index = facet_index.or(self.facet.focused)?;
        facet.cells.get(index).map(|cell| Arc::new(cell.payload.clone()))
    }

    /// `set_title("", y=1.0)` on a cell: empty, at the rc size and pad, pinned.
    pub(crate) fn clear_cell_title(&mut self, axes: usize) {
        let (size, color, pad) = (self.rc.axes_titlesize, self.rc.axes_titlecolor.clone(), self.rc.axes_titlepad);
        let model = &mut self.fig.axes[axes];
        model.titles[0].text.clear();
        model.titles[0].size = size;
        model.titles[0].color = color;
        model.title_pad = pad;
        model.autotitlepos = false;
    }

    /// A grid cell as the reserve builds it (`_build_axes`): marks at the
    /// cell length, labels off, its title empty and pinned at the compact pad.
    pub(crate) fn grid_cell(&self, bounds: [f64; 4]) -> AxesModel {
        let mut axes = AxesModel::new(&self.rc, bounds);
        let length = self.style.render.facet_cell_tick_length_pt;
        axes.xaxis.major.size = length;
        axes.yaxis.major.size = length;
        axes.xaxis.major.label1on = false;
        axes.yaxis.major.label1on = false;
        // Its first ticks were made before the length moved their labels.
        axes.xaxis.distinct_ticks = crate::ticks::TICKS_FLOOR;
        axes.yaxis.distinct_ticks = crate::ticks::TICKS_FLOOR;
        axes.title_pad = self.style.render.compact_axes_title_pad_pt;
        axes.autotitlepos = false;
        axes
    }

    /// `_position_facet_axes_for_frame(axes)`: the cells' final boxes.
    fn position_facet_axes_for_frame(&mut self, cells: &[usize]) {
        let image_cells = matches!(&self.spec, PlotSpec::FacetGrid(grid) if matches!(grid.cell, CellPlot::Image(_)));
        let focus_plans = self.plan.facet_focus_axes.clone().filter(|_| self.facet.focus.is_some() && image_cells);
        self.sync_facet_focus_chrome(if focus_plans.is_some() { self.facet.focus } else { None });
        let Some(selected) = self.facet.focus else {
            let ratio = self.planned_image_box_ratio("facet_cell");
            for (index, &axes) in cells.iter().enumerate() {
                let plan = self.plan.axes[index].clone();
                self.fig.axes[axes].room = Some(plan.room);
                let bounds = self.pixel_quantized_bounds(axes, plan.r#box.matplotlib_bounds(), ratio);
                self.quantized_bounds.insert(axes, bounds);
                // `get_position()` settles the aspect first.
                self.apply_aspect(axes);
                let p = self.fig.axes[axes].active;
                let current = [p[0], p[1], p[2] - p[0], p[3] - p[1]];
                if current.iter().zip(&bounds).any(|(now, wanted)| (now - wanted).abs() > 1e-12) {
                    self.fig.axes[axes].set_position(bounds);
                    self.chrome_dirty.insert(axes);
                }
                self.fig.axes[axes].visible = index < self.facet.visible;
                if self.fig.axes[axes].axison {
                    self.fig.axes[axes].axison = false;
                    self.clear_cell_title(axes);
                    self.chrome_dirty.insert(axes);
                }
            }
            return;
        };
        let (bounds, room) = match &focus_plans {
            // A focused image cell IS the standalone Image surface: the split's image box and room.
            Some(plans) => {
                let image = plans.iter().find(|plan| plan.role == "image").expect("an image box");
                (image.r#box.matplotlib_bounds(), image.room)
            }
            None => {
                let region = crate::layout::facet_focus_box(&self.plan).expect("a grid's region");
                (region.matplotlib_bounds(), Room::new(region.left, 1.0 - region.right, region.top, 1.0 - region.bottom).expect("margin room"))
            }
        };
        self.fig.axes[cells[selected]].room = Some(room);
        self.discard_facet_cell_chrome();
        for (index, &axes) in cells.iter().enumerate() {
            let visible = index == selected;
            self.fig.axes[axes].visible = visible;
            if self.fig.axes[axes].axison != visible {
                self.fig.axes[axes].axison = visible;
                self.chrome_dirty.insert(axes);
            }
            self.quantized_bounds.remove(&axes);
            self.box_exact.insert(axes, false);
        }
        self.apply_aspect(cells[selected]);
        let p = self.fig.axes[cells[selected]].active;
        if [p[0], p[1], p[2] - p[0], p[3] - p[1]] != bounds { self.fig.axes[cells[selected]].set_position(bounds); }
    }

    /// `_sync_facet_focus_chrome(index)`: the focused image cell's rail and
    /// colorbar axes exist exactly while it is focused.
    fn sync_facet_focus_chrome(&mut self, index: Option<usize>) {
        let previous = self.facet.focus_chrome;
        if previous == index { return; }
        if let Some(previous) = previous {
            // The cell's side chrome goes with its axes (`{key}:distribution...`).
            let key = format!("facet:{previous}");
            self.select_image(&key);
            let image = &mut self.image;
            image.distribution_limits = None;
            image.distribution_cache = None;
            image.distribution = (None, 0.0);
            image.guides = None;
            image.colorbar = None;
            let mut removed: Vec<usize> = ["distribution", "colorbar"].iter().flat_map(|role| self.axes_of(role).to_vec()).collect();
            self.roles.retain(|(role, _)| role != "distribution" && role != "colorbar");
            // A selector scene living on a removed axes (the rail's colour
            // handles, a crosshair's sample) dies with it, whole.
            for (kind, artists, topology) in std::mem::take(&mut self.selectors.kinds) {
                if artists.iter().any(|(axes, _)| removed.contains(axes)) {
                    self.remove_selector_artists(&artists);
                } else {
                    self.selectors.kinds.push((kind, artists, topology));
                }
            }
            removed.sort_unstable();
            for &axes in removed.iter().rev() {
                self.chrome_dirty.remove(&axes);
                self.fig.axes.remove(axes);
            }
        }
        if index.is_some() {
            let plans = self.plan.facet_focus_axes.clone().expect("a grid's focus plan");
            for plan in plans.iter().filter(|plan| plan.role != "image") {
                let mut axes = AxesModel::new(&self.rc, plan.r#box.matplotlib_bounds());
                axes.room = Some(plan.room);
                axes.role = plan.role.clone();
                axes.gid = plan.role.clone();
                self.fig.axes.push(axes);
                let at = self.fig.axes.len() - 1;
                match self.roles.iter_mut().find(|(role, _)| *role == plan.role) {
                    Some((_, list)) => list.push(at),
                    None => self.roles.push((plan.role.clone(), vec![at])),
                }
            }
        }
        self.facet.focus_chrome = index;
    }

    /// Swap image surface `key`'s tables in as the renderer's (the
    /// `_artists[key]`-keyed state of `_update_image`); the scene, the
    /// colour-limit candidate and the gesture stay the renderer's.
    pub(crate) fn select_image(&mut self, key: &str) {
        if self.facet.image_key.as_deref() == Some(key) { return; }
        let shared = (self.image.prepared.take(), self.image.candidate.take(), self.image.color_gesture);
        let incoming = self.facet.images.remove(key).unwrap_or_default();
        let outgoing = std::mem::replace(&mut self.image, incoming);
        if let Some(old) = self.facet.image_key.replace(key.to_owned()) { self.facet.images.insert(old, outgoing); }
        (self.image.prepared, self.image.candidate, self.image.color_gesture) = shared;
    }

    /// The grid's image scene from its cells: each cell's plane and range
    /// (measured under the cell's own key), the grid's colour scale.
    fn prepare_facet_images(&mut self, payload: &Arc<Payload>, state: &DisplayState, focused: bool)
        -> ((f64, f64), Option<ImageSignature>) {
        let Payload::Facet(facet) = &**payload else { unreachable!() };
        let mut pooled: Option<(f64, f64)> = None;
        let (mut planes, mut valid, mut extents, mut scales) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (index, cell) in facet.cells.iter().enumerate() {
            let Payload::Image(data) = &cell.payload else { panic!("an image grid's cell is an image") };
            let x = data.x.display.to_f64();
            let y = data.y.display.to_f64();
            let cell_scales = (image_coordinate_scale(&x).unwrap_or_else(|e| panic!("{e}")),
                               image_coordinate_scale(&y).unwrap_or_else(|e| panic!("{e}")));
            let (x0, x1, y0, y1) = centers_extent(&x, &y, (&cell_scales.0, &cell_scales.1));
            let cell_valid = (!all_valid(&data.valid)).then(|| data.valid.clone());
            self.select_image(&format!("facet:{index}"));
            if let Some(range) = self.cached_image_range(&data.z.display, cell_valid.as_ref()) {
                pooled = Some(match pooled { None => range, Some(p) => (if range.0 < p.0 { range.0 } else { p.0 }, if range.1 > p.1 { range.1 } else { p.1 }) });
            }
            planes.push(data.z.display.clone());
            valid.push(cell_valid);
            extents.push([x0, x1, y1, y0]);
            scales.push(cell_scales);
        }
        let limits = self.resolve_limits("facet:image", pooled, state, "color", false, true, false);
        if focused || planes.is_empty() {
            self.facet.image_scene = None;
            self.facet.image_signature = None;
            return (limits, None);
        }
        let refs: Vec<&Array> = planes.iter().collect();
        let stacked_valid = valid.iter().any(Option::is_some).then(|| {
            let masks: Vec<Array> = planes.iter().zip(&valid).map(|(plane, v)| v.clone()
                .unwrap_or_else(|| Array::from_vec(vec![true; plane.len()], plane.shape().to_vec()))).collect();
            stack(&masks.iter().collect::<Vec<_>>())
        });
        self.image.prepared = None;
        self.facet.image_scene = Some(FacetImages {
            values: stack(&refs), valid: stacked_valid, extents: extents.clone(), color_limits: limits, cmap: colormap_name(state),
            show: matches!(state.get("show_image"), Some(Value::Bool(true))), payload: payload.clone(), state: Arc::new(state.clone()),
        });
        (limits, Some(ImageSignature { revision: state.revision, extents, view: self.requested_view, scales }))
    }

    /// `_update_facets(payload, state)`.
    pub(crate) fn update_facets(&mut self, payload: &Arc<Payload>, state: &DisplayState) {
        let Payload::Facet(facet) = &**payload else { panic!("FacetGrid payload and rendered topology are inconsistent") };
        let cells = &facet.cells;
        assert_eq!(self.facet.visible, cells.len(), "FacetGrid payload and rendered topology are inconsistent");
        let focused = self.facet.focus.is_some();
        let cell_axes = self.facet_cells();
        let curve_cells = matches!(&self.spec, PlotSpec::FacetGrid(grid) if matches!(grid.cell, CellPlot::Curve(_)));
        if !curve_cells { self.curve.prepared = None; }
        self.position_facet_axes_for_frame(&cell_axes);
        // The grid's facts: one x span and one y span over every cell.
        let mut curve_series_by_cell: Vec<Vec<PreparedSeries>> = Vec::new();
        let mut curve_limits = None;
        let mut native_curve = false;
        if curve_cells {
            curve_series_by_cell = cells.iter().map(|cell| prepare_curve_series(curve_series(&cell.payload))).collect();
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            for series in &curve_series_by_cell {
                for item in series {
                    let s = item.summary;
                    if s.xlow.is_finite() {
                        xs.extend([s.xlow, s.xhigh]);
                        ys.extend([s.pooled.0, s.pooled.1]);
                    }
                }
            }
            if !xs.is_empty() {
                let x_target = curve_x_limits(&xs).expect("finite x");
                curve_limits = Some((x_target, self.resolve_curve_y_limits("facet:curve", data_limits(&ys), state)));
            }
            native_curve = !focused && curve_series_by_cell.iter().all(|series| self.native_curve_scene_supported(series));
            if native_curve {
                self.curve.prepared = Some(PreparedCurve {
                    series: curve_series_by_cell.clone(), state: Arc::new(state.clone()), key: String::new(),
                    x_label: String::new(), y_label: String::new(),
                    facet: Some(FacetCommand { payload: payload.clone(), limits: curve_limits }),
                });
                for (_, axes, _) in self.painted_surfaces() { self.withdraw_series_artists(axes); }
            } else {
                self.curve.prepared = None;
            }
        }
        // One colour scale for the grid; a cell renders again only when what
        // its render is a function of moved (the scene carries the planes).
        let image_cells = matches!(&self.spec, PlotSpec::FacetGrid(grid) if matches!(grid.cell, CellPlot::Image(_)));
        let (mut image_limits, mut signature, mut render_images) = (None, None, true);
        if image_cells {
            let (limits, scene) = self.prepare_facet_images(payload, state, focused);
            image_limits = Some(limits);
            render_images = scene.is_none() || self.facet.image_signature != scene;
            signature = scene;
        } else {
            self.facet.image_scene = None;
            self.facet.image_signature = None;
        }
        // One bin projection and one count range for the grid.
        let histogram_cells = matches!(&self.spec, PlotSpec::FacetGrid(grid) if matches!(grid.cell, CellPlot::Histogram(_)));
        let mut histogram_cell_arrays: Vec<HistogramArrays> = Vec::new();
        let mut histogram_limits = None;
        if histogram_cells {
            histogram_cell_arrays = cells.iter().map(|cell| histogram_arrays(histogram_payload(&cell.payload), state)).collect();
            let usable: Vec<&HistogramArrays> = histogram_cell_arrays.iter().filter(|arrays| arrays.edges.len() >= 2).collect();
            if let Some(first) = usable.first() {
                let shared = &first.edges;
                assert!(usable[1..].iter().all(|arrays| arrays.edges == *shared), "FacetGrid histogram cells must share one bin projection");
                // The shared edges already carry relim_mode's retention.
                let x = (shared[0], shared[shared.len() - 1]);
                let pooled: Vec<Arc<Vec<f64>>> = usable.iter().flat_map(|arrays| arrays.counts.iter().cloned()).collect();
                histogram_limits = Some((x, self.resolve_histogram_y_limits("facet:histogram", &pooled, state)));
            }
        }
        let show_image = matches!(state.get("show_image"), Some(Value::Bool(true)));
        let (outer_x, outer_y) = self.effective_labels(payload, state);
        let mut visible_axes: Vec<(usize, usize)> = Vec::new();
        for (key, axes, index) in self.painted_surfaces() {
            let index = index.expect("a grid surface is a cell");
            if native_curve {
                if let Some((x, y)) = curve_limits {
                    self.set_xlim(axes, x);
                    self.set_ylim(axes, y);
                }
            } else if curve_cells {
                let cell_payload = Arc::new(cells[index].payload.clone());
                self.update_curve(axes, &cell_payload, state, &key, curve_limits, Some(curve_series_by_cell[index].clone()), false);
            } else if image_cells && render_images {
                let Payload::Image(data) = &cells[index].payload else { unreachable!() };
                self.select_image(&key);
                self.update_image_data(axes, data, state, &key, image_limits, false);
            } else if histogram_cells {
                let arrays = histogram_cell_arrays[index].clone();
                self.update_histogram(axes, histogram_payload(&cells[index].payload), state, &key, Some(arrays), histogram_limits, false);
            }
            if image_cells {
                self.select_image(&key);
                if let Some((owner, id)) = self.image.image {
                    if let Some(Child::Image(image)) = self.fig.axes[owner].child_mut(id) { image.visible = show_image; }
                }
            }
            if !focused {
                let model = &mut self.fig.axes[axes];
                model.xaxis.label.clear();
                model.yaxis.label.clear();
            }
            visible_axes.push((index, axes));
        }
        if signature.is_some() { self.facet.image_signature = signature; }
        if image_cells {
            // The renderer's image tables are the primary surface's between frames.
            let (key, _) = self.primary_surface();
            if key.starts_with("facet:") { self.select_image(&key); }
        }
        let typography = self.plan.facet_typography.clone();
        let (rows, columns) = self.plan.facet_shape.map(|(r, c)| (r as usize, c as usize)).unwrap_or((1, cells.len().max(1)));
        let mut cell_titles = BTreeMap::new();
        let cell_tick_length = self.style.render.facet_cell_tick_length_pt;
        let cell_tick_pt = typography.as_ref().map_or(self.style.fonts.tick_pt, |t| t.tick_pt);
        for &(index, axes) in &visible_axes {
            if !focused {
                let size = typography.as_ref().map_or(self.style.fonts.tick_pt, |t| t.cell_title_pt);
                cell_titles.insert(index, (facet_cell_title(&cells[index]), size));
            }
            for x in [true, false] { self.fig.axes[axes].axis_mut(x).major.size = cell_tick_length; }
            if focused { continue; }
            let (row, column) = (index / columns, index % columns);
            let label_left = column == 0;
            let label_bottom = row == rows - 1 || index + columns >= cells.len();
            if curve_cells { self.apply_curve_ticks_at(axes, &curve_series_by_cell[index], cell_tick_pt); }
            else { self.apply_smart_ticks_both(axes, cell_tick_pt); }
            let model = &mut self.fig.axes[axes];
            model.yaxis.major.label1on = label_left;
            model.xaxis.major.label1on = label_bottom;
            let corner = label_bottom && label_left;
            model.xaxis.offset_visible = corner;
            model.yaxis.offset_visible = corner;
        }
        self.facet.cell_titles = cell_titles;
        // The outer labels: figure texts, shown in the overview.
        let outer_pt = typography.as_ref().map_or(self.style.fonts.axis_label_pt, |t| t.outer_axis_label_pt);
        let ids = match self.facet.outer {
            Some(ids) => ids,
            None => {
                let text = |x: f64, y: f64, ha: HAlign, va: VAlign, rotation: f64, color: &str| FigureText {
                    visible: true, zorder: 3.0, x, y, text: String::new(), size: outer_pt, color: color.to_owned(), ha, va, rotation,
                    frame: FigureFrame::Figure,
                };
                let color = self.rc.text_color.clone();
                let x_id = self.fig.add_text(text(0.5, 0.012, HAlign::Center, VAlign::Bottom, 0.0, &color));
                let y_id = self.fig.add_text(text(0.008, 0.5, HAlign::Left, VAlign::Center, 90.0, &color));
                self.facet.outer = Some([x_id, y_id]);
                [x_id, y_id]
            }
        };
        let overview = self.facet.focus.is_none();
        for (id, value) in ids.iter().zip([&outer_x, &outer_y]) {
            let text = self.fig.text_mut(*id).expect("outer label");
            text.text = value.clone();
            text.visible = overview;
        }
        let Some(selected) = self.facet.focus else { return };
        let axes = cell_axes[selected];
        let (label_pt, title_pt, pad, tick_pt) = (self.style.fonts.axis_label_pt, self.style.fonts.figure_title_pt,
                                                  self.style.render.compact_axes_title_pad_pt, self.style.fonts.tick_pt);
        let title = facet_cell_title(&cells[selected]);
        {
            let model = &mut self.fig.axes[axes];
            if model.xaxis.label != outer_x { model.xaxis.label = outer_x.clone(); model.xaxis.label_size = label_pt; }
            if model.yaxis.label != outer_y { model.yaxis.label = outer_y.clone(); model.yaxis.label_size = label_pt; }
        }
        if self.fig.axes[axes].titles[0].text != title || self.fig.axes[axes].titles[0].size != title_pt {
            let color = self.rc.axes_titlecolor.clone();
            let model = &mut self.fig.axes[axes];
            model.titles[0].text = title;
            model.titles[0].size = title_pt;
            model.titles[0].color = color;
            model.title_pad = pad;
            model.autotitlepos = false;
        }
        if curve_cells { self.apply_curve_ticks_at(axes, &curve_series_by_cell[selected], tick_pt); }
        if image_cells { self.apply_smart_ticks_both(axes, tick_pt); }
        let model = &mut self.fig.axes[axes];
        for x in [true, false] {
            let axis = model.axis_mut(x);
            if matches!(axis.policy, TickPolicy::Smart(_)) && axis.signature.is_some() && !axis.offset_visible { axis.offset_visible = true; }
        }
        if !model.xaxis.major.label1on || !model.yaxis.major.label1on {
            model.xaxis.major.label1on = true;
            model.yaxis.major.label1on = true;
        }
    }

    /// `_raster_prepared_images` of a grid's scene: every visible cell's
    /// plane in one kernel launch; false without a scene. The image artists
    /// it paints for are returned (they are not drawn again).
    pub(crate) fn raster_prepared_facet_images(&self, pixels: &mut [u8], width: usize, height: usize,
                                               figure: &zlc_figure::ir::Figure) -> Option<Vec<(usize, ChildId)>> {
        use zlc_render::composition::{raster_prepared_images, ImagePlan, ImageValues};
        let scene = self.facet.image_scene.as_ref()?;
        let surfaces = self.painted_surfaces();
        let artists: Vec<(usize, ChildId)> = surfaces.iter().filter_map(|(key, _, _)| self.image_of(key)).collect();
        if !scene.show { return Some(artists); }
        let (low, high) = scene.color_limits;
        let span = high - low;
        let shape = scene.values.shape();
        if !span.is_finite() || span <= 0.0 || surfaces.len() != shape[0] { return None; }
        let (rows, columns) = (shape[1], shape[2]);
        let upper = self.style.render.image_origin == "upper";
        let mut evaluator = zlc_figure::transform::Evaluator::new(figure);
        let (mut blits, mut clips, mut affines) = (vec![[0i64; 4]; shape[0]], vec![[0i64; 4]; shape[0]], vec![[0.0f64; 6]; shape[0]]);
        for (row, (key, axes, _)) in surfaces.iter().enumerate() {
            let (owner, id) = self.image_of(key)?;
            let index = self.fig.axes[owner].children.iter().position(|(child, _)| *child == id)?;
            let zlc_figure::ir::Artist::Image(image) = &figure.axes[owner].children[index] else { return None };
            let transform = evaluator.matrix(image.artist.transform.expect("an image transform")).ok()?;
            let bbox = self.fig.axes[*axes].bbox_exact(self.fig.size_inches, self.fig.dpi);
            if let Some((blit, clip, affine)) = super::image::image_scene_geometry(bbox, &transform, scene.extents[row], rows, columns, height, upper) {
                (blits[row], clips[row], affines[row]) = (blit, clip, affine);
            }
        }
        let values = &scene.values;
        let plane = match values.dtype() {
            DType::Bool => ImageValues::Bool(values.bytes()),
            DType::U8 => ImageValues::U8(values.slice()), DType::U16 => ImageValues::U16(values.slice()),
            DType::U32 => ImageValues::U32(values.slice()), DType::U64 => ImageValues::U64(values.slice()),
            DType::I8 => ImageValues::I8(values.slice()), DType::I16 => ImageValues::I16(values.slice()),
            DType::I32 => ImageValues::I32(values.slice()), DType::I64 => ImageValues::I64(values.slice()),
            DType::F32 => ImageValues::F32(values.slice()), DType::F64 => ImageValues::F64(values.slice()),
            other => panic!("the image kernel paints no {} plane", other.name()),
        };
        let table = super::image::color_lut(scene.cmap).expect("an image colormap");
        let mut lut = [[0u8; 4]; 256];
        for (slot, rgba) in lut.iter_mut().zip(table.chunks_exact(4)) { slot.copy_from_slice(rgba); }
        let single = matches!(values.dtype(), DType::Bool | DType::U8 | DType::U16 | DType::I8 | DType::I16 | DType::F16 | DType::F32);
        let plan = ImagePlan {
            values: plane, valid: scene.valid.as_ref().map(|v| v.bytes()), shape: [shape[0], rows, columns],
            blits: &blits, clips: &clips, affines: &affines, lut: &lut, vmin: low, span, single,
        };
        let mut canvas = zlc_render::Canvas::new(pixels, width, height).expect("canvas");
        let _span = super::profile::span("kernels.facet.images");
        raster_prepared_images(&mut canvas, &plan).expect("prepared images");
        Some(artists)
    }

    /// The image artist of surface `key`, wherever its tables are.
    fn image_of(&self, key: &str) -> Option<(usize, ChildId)> {
        if !self.is_facet() || self.facet.image_key.as_deref() == Some(key) { return self.image.image; }
        self.facet.images.get(key).and_then(|t| t.image)
    }

    /// `_materialize_prepared_images` of a grid's scene for an export: every
    /// visible cell renders again at the grid's scale, its artist holding
    /// the plane with the mask a draw needs.
    pub(crate) fn materialize_prepared_facet_images(&mut self) {
        let Some(scene) = self.facet.image_scene.take() else { return };
        let Payload::Facet(facet) = &*scene.payload else { return };
        for (key, axes, index) in self.painted_surfaces() {
            let Payload::Image(data) = &facet.cells[index.expect("a grid surface is a cell")].payload else { continue };
            self.select_image(&key);
            self.update_image_data(axes, data, &scene.state, &key, Some(scene.color_limits), false);
            let valid = (!all_valid(&data.valid)).then(|| data.valid.clone());
            self.mask_image_artist(&data.z.display, valid.as_ref());
        }
    }

    /// `Axis._update_ticks()` of a grid cell's axis. Labelled cells ask the
    /// ladder the same question cell after cell (one lattice, one box size,
    /// one policy history); the first answer is every such cell's.
    fn facet_axis_ticks(&mut self, axes: usize, x: bool, answers: &mut HashMap<String, TickAnswer>) {
        let _span = super::profile::span("facet.refresh.ticks");
        let geometry = self.tick_geometry(axes);
        let model = &self.fig.axes[axes];
        let question = tick_question(model.axis(x), model.view(x), &geometry);
        match answers.get(&question) {
            Some(answer) => answer.adopt(self.fig.axes[axes].axis_mut(x)),
            None => {
                self.update_axis_ticks(axes, x);
                answers.insert(question, TickAnswer::of(self.fig.axes[axes].axis(x)));
            }
        }
    }

    /// `_raster_facet_fit_ellipses(canvas)`: an overview paints its cells'
    /// ellipse fits (`facet:fit_native`) with the kernel. The fit artists it
    /// paints for are returned (they are not drawn again); None when it
    /// painted nothing.
    pub(crate) fn raster_facet_fit_ellipses(&self, pixels: &mut [u8], width: usize, height: usize,
                                            figure: &zlc_figure::ir::Figure) -> Option<Vec<(usize, ChildId)>> {
        if !self.facet_overview() { return None; }
        let overlays: Vec<&FitOverlay> = self.facet.fit_native.as_ref()?.iter()
            .filter(|overlay| overlay.success && overlay.ellipse_glyph.is_some() && overlay.facet_index.is_some()).collect();
        if overlays.is_empty() { return None; }
        let artists = &self.style.artists;
        let dpi = self.fig.dpi;
        let ring_token = &artists.point_occupied;
        let mut ring = super::mpl::color::rgba(ring_token.color);
        ring[3] *= ring_token.alpha;
        let ring = rgba_bytes(ring);
        let center = rgba_bytes(super::mpl::color::rgba(artists.fit_ellipse_color));
        let ring_width = py_max(1.0, ring_token.linewidth * dpi / 72.0);
        let center_radius = py_max(0.5, 0.5 * (artists.fit_ellipse_center_area_pt2.sqrt() + super::fit::CENTER_EDGE_PT) * dpi / 72.0);
        let mut evaluator = zlc_figure::transform::Evaluator::new(figure);
        let cells = self.axes_of("facet_cell");
        let canvas_h = height as f64;
        let items: Vec<FitEllipse> = overlays.iter().map(|overlay| {
            let axes = cells[overlay.facet_index.expect("a cell's fit")];
            let glyph = overlay.ellipse_glyph.expect("an ellipse fit");
            let data = figure.axes[axes].trans_data;
            let mut at = |x: f64, y: f64| evaluator.transform(data, x, y).expect("transData");
            let (cx, cy) = at(glyph.center_x, glyph.center_y);
            let (ex, _) = at(glyph.center_x + glyph.radius_x, glyph.center_y);
            let (_, ey) = at(glyph.center_x, glyph.center_y + glyph.radius_y);
            let [x0, y0, x1, y1] = self.fig.axes[axes].bbox_exact(self.fig.size_inches, dpi);
            FitEllipse {
                geometry: [cx, canvas_h - cy, (ex - cx).abs(), (ey - cy).abs()],
                ring, ring_width, center, center_radius,
                clip: [0i64.max(x0.floor() as i64), 0i64.max((canvas_h - y1).floor() as i64),
                       (width as i64).min(x1.ceil() as i64), (height as i64).min((canvas_h - y0).ceil() as i64)],
            }
        }).collect();
        let _span = super::profile::span("kernels.facet.ellipses");
        raster_fit_ellipses(&items, pixels, width);
        // `facet_ellipse_ids`: every shown ellipse fit's centre and ring.
        Some(self.fit.topologies.iter().filter(|(_, topology)| topology.family == "ellipse").flat_map(|(_, topology)| {
            [topology.slots.center, topology.slots.ring].into_iter().flatten().map(move |id| (topology.axes, id))
        }).filter(|&(axes, id)| self.fig.axes[axes].child(id).is_some_and(Child::visible)).collect())
    }

    /// `apply_smart_ticks(axes, label_pt=...)` on both axes.
    fn apply_smart_ticks_both(&mut self, axes: usize, label_pt: f64) {
        let model = &mut self.fig.axes[axes];
        apply_smart_ticks(model, true, label_pt);
        apply_smart_ticks(model, false, label_pt);
    }

    // ------------------------------------------------------------ chrome
    /// `_discard_facet_cell_chrome`: the strokes and the chrome's figure artists go.
    pub(crate) fn discard_facet_cell_chrome(&mut self) {
        if let Some(ids) = self.facet.chrome.take() {
            self.fig.artists.retain(|(id, _)| !ids.contains(id));
        }
        for cell in self.facet_cells() { self.fig.axes[cell].strokes.clear(); }
        self.facet.chrome_signature = None;
    }

    /// The overview branch of `_sync_axes_chrome`.
    pub(crate) fn sync_facet_chrome(&mut self) {
        let cells = self.facet_cells();
        let visible: Vec<(usize, usize)> = cells.iter().enumerate()
            .filter(|&(index, &axes)| index < self.facet.visible && self.fig.axes[axes].visible)
            .map(|(index, &axes)| (index, axes)).collect();
        let (w, h) = self.fig.size_px();
        let mut key = Key::new();
        key.add(&(w.round_ties_even(), h.round_ties_even()));
        for &(index, axes) in &visible {
            let a = &self.fig.axes[axes];
            let bbox = a.bbox_exact(self.fig.size_inches, self.fig.dpi);
            key.add(&(index, rounded(&bbox), [a.xlim.0, a.ylim.0, a.xlim.1, a.ylim.1], &a.xaxis.signature, &a.yaxis.signature,
                      params_key(&a.xaxis.major), params_key(&a.yaxis.major), a.xaxis.offset_visible, a.yaxis.offset_visible,
                      self.facet.cell_titles.get(&index)));
        }
        let signature = key.finish();
        if self.facet.chrome.is_some() && self.facet.chrome_signature == Some(signature) { return; }
        self.refresh_facet_cell_chrome(&visible, false);
        self.facet.chrome_signature = Some(signature);
    }

    /// `_refresh_facet_cell_chrome(visible, titles, tick_layout_ready=...)`.
    fn refresh_facet_cell_chrome(&mut self, visible: &[(usize, usize)], tick_layout_ready: bool) {
        let _span = super::profile::span("present.facet_chrome.refresh");
        let mut marks: Vec<(usize, Stroke)> = Vec::new();
        let mut frames: Vec<(usize, Stroke)> = Vec::new();
        let mut titles_plan: Vec<(usize, String, f64)> = Vec::new();
        let mut labels: Vec<LabelRef> = Vec::new();
        let mut shared: HashMap<u64, Vec<Lane>> = HashMap::new();
        let mut answers: HashMap<String, TickAnswer> = HashMap::new();
        let mut tick_sizes: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
        let axis_zorder = self.rc.axis_zorder;
        for &(index, axes) in visible {
            {
                let a = &self.fig.axes[axes];
                for (side, _) in SPINES.iter().enumerate() {
                    let vertices = match side {
                        0 => [[0.0, a.ylim.0], [0.0, a.ylim.1]],
                        1 => [[1.0, a.ylim.0], [1.0, a.ylim.1]],
                        2 => [[a.xlim.0, 0.0], [a.xlim.1, 0.0]],
                        _ => [[a.xlim.0, 1.0], [a.xlim.1, 1.0]],
                    };
                    frames.push((axes, Stroke::Frame(FrameArtist { side, zorder: 2.5, vertices })));
                }
            }
            let (share_key, labelled) = {
                let a = &self.fig.axes[axes];
                let bbox = a.bbox_exact(self.fig.size_inches, self.fig.dpi);
                let key = Key::new().add(&([a.xlim.0, a.ylim.0, a.xlim.1, a.ylim.1],
                    rounded(&[bbox[2] - bbox[0], bbox[3] - bbox[1]]), &a.xaxis.signature, &a.yaxis.signature,
                    params_key(&a.xaxis.major), params_key(&a.yaxis.major))).finish();
                let labelled = [&a.xaxis, &a.yaxis].iter().any(|axis| axis.major.label1on || axis.major.label2on);
                (key, labelled)
            };
            let lanes = match (labelled, shared.get(&share_key)) {
                (false, Some(lanes)) => lanes.clone(),
                _ => {
                    let mut lanes = Vec::new();
                    for x in [true, false] {
                        let measured = matches!(self.fig.axes[axes].axis(x).policy, TickPolicy::Smart(_) | TickPolicy::Declared { .. });
                        if measured && !tick_layout_ready { self.fig.axes[axes].axis_mut(x).forget_tick_memo(); }
                        self.facet_axis_ticks(axes, x, &mut answers);
                        let a = &self.fig.axes[axes];
                        let axis = a.axis(x);
                        if measured { tick_sizes[usize::from(!x)].push(axis.drawn_pt().expect("a measured axis")); }
                        let placed = axis.tick_indices_in_view(a.view(x));
                        let p = &axis.major;
                        let locs: Vec<f64> = placed.iter().map(|&i| axis.locs[i]).collect();
                        let (m1, m2) = tick_markers(x, &p.tickdir);
                        if p.grid_on && !locs.is_empty() {
                            lanes.push(Lane { x, grid: true, side: 0, locs: locs.clone(), color: String::new(), marker: "", size: 0.0, width: 0.0 });
                        }
                        for (side, on, marker) in [(1u8, p.tick1on, m1), (2u8, p.tick2on, m2)] {
                            if on && !locs.is_empty() {
                                lanes.push(Lane { x, grid: false, side, locs: locs.clone(), color: p.color.clone(), marker, size: p.size, width: p.width });
                            }
                        }
                        if labelled {
                            for &i in &placed {
                                if axis.labels[i].is_empty() { continue; }
                                for (side, on) in [(1u8, p.label1on), (2u8, p.label2on)] {
                                    if on { labels.push(LabelRef::Tick { axes, x, index: i, side }); }
                                }
                            }
                        }
                    }
                    if !labelled { shared.insert(share_key, lanes.clone()); }
                    lanes
                }
            };
            for lane in &lanes {
                if lane.grid {
                    for &loc in &lane.locs {
                        let (xs, ys) = if lane.x { (vec![loc, loc], vec![0.0, 1.0]) } else { (vec![0.0, 1.0], vec![loc, loc]) };
                        marks.push((axes, Stroke::Marks(MarkArtist { x: lane.x, grid: true, zorder: axis_zorder, xs, ys,
                            color: String::new(), marker: "", markersize: 0.0, markeredgewidth: 0.0 })));
                    }
                } else {
                    let across = if lane.side == 1 { 0.0 } else { 1.0 };
                    let (xs, ys) = if lane.x { (lane.locs.clone(), vec![across; lane.locs.len()]) }
                        else { (vec![across; lane.locs.len()], lane.locs.clone()) };
                    marks.push((axes, Stroke::Marks(MarkArtist { x: lane.x, grid: false, zorder: axis_zorder, xs, ys,
                        color: lane.color.clone(), marker: lane.marker, markersize: lane.size, markeredgewidth: lane.width })));
                }
            }
            for x in [true, false] {
                let axis = self.fig.axes[axes].axis_mut(x);
                if axis.offset_visible {
                    axis.offset_text = axis.offset_string.clone();
                    if !axis.offset_text.is_empty() { labels.push(LabelRef::Offset { axes, x }); }
                }
            }
            if let Some((text, size)) = self.facet.cell_titles.get(&index) { titles_plan.push((axes, text.clone(), *size)); }
        }
        // One common label size per direction, the smallest the locators priced.
        for (direction, sizes) in tick_sizes.iter().enumerate() {
            let Some(size) = sizes.iter().copied().fold(None, |m: Option<f64>, s| Some(m.map_or(s, |m| if s < m { s } else { m }))) else { continue };
            for &(_, axes) in visible {
                let axis = self.fig.axes[axes].axis_mut(direction == 0);
                let drawn = match &mut axis.policy {
                    TickPolicy::Smart(locator) => &mut locator.ladder.drawn_pt,
                    TickPolicy::Declared { locator } => &mut locator.ladder.drawn_pt,
                    _ => continue,
                };
                if (size - *drawn).abs() < 1e-9 { continue; }
                *drawn = size;
                axis.major.set_labelsize(size);
                axis.offset_size = size;
            }
        }
        // Titles stay inside the actual header: the narrowest cell, the lowest gutter.
        if let (Some(typography), false) = (self.plan.facet_typography.clone(), titles_plan.is_empty()) {
            let dots_per_point = self.fig.dpi / 72.0;
            let (_, fh) = self.fig.size_px();
            let width = titles_plan.iter().map(|&(axes, _, _)| {
                let b = self.fig.axes[axes].bbox_exact(self.fig.size_inches, self.fig.dpi);
                b[2] - b[0]
            }).fold(f64::INFINITY, |m, w| if w < m { w } else { m }) / dots_per_point;
            let budget = if width < typography.cell_title_max_width_pt { width } else { typography.cell_title_max_width_pt };
            let mut height = visible.iter().map(|&(index, _)| { let r = self.plan.axes[index].room; r.top + r.bottom })
                .fold(f64::INFINITY, |m, v| if v < m { v } else { m });
            height *= fh / dots_per_point;
            height -= self.style.render.compact_axes_title_pad_pt;
            let _span = super::profile::span("facet.refresh.titles");
            let fitted: Vec<(String, f64)> = titles_plan.iter()
                .map(|(_, text, _)| fitted_title(&self.measure, text, typography.cell_title_pt, typography.cell_title_min_pt, budget, height))
                .collect();
            let common = fitted.iter().map(|f| f.1).fold(f64::INFINITY, |m, s| if s < m { s } else { m });
            for (plan, (text, _)) in titles_plan.iter_mut().zip(fitted) { plan.1 = text; plan.2 = common; }
        }
        // Build the chrome afresh: strokes on their cells, titles then the carrier on the figure.
        self.discard_facet_cell_chrome();
        for (axes, stroke) in marks.into_iter().chain(frames) { self.fig.axes[axes].strokes.push(stroke); }
        let mut ids = Vec::new();
        for (axes, text, size) in titles_plan {
            let model = &self.fig.axes[axes];
            if !model.titles[0].text.is_empty() || model.autotitlepos {
                self.clear_cell_title(axes);
                self.fig.axes[axes].title_pad = self.style.render.compact_axes_title_pad_pt;
            }
            let title = &self.fig.axes[axes].titles[0];
            let color = if title.color == "auto" { self.rc.text_color.clone() } else { title.color.clone() };
            ids.push(self.fig.add_artist(FigureArtist::Text(FigureText {
                visible: true, zorder: 3.0, x: 0.5, y: 1.0, text, size, color, ha: HAlign::Center, va: VAlign::Baseline,
                rotation: 0.0, frame: FigureFrame::Title(axes),
            })));
        }
        ids.push(self.fig.add_artist(FigureArtist::Labels(labels)));
        self.facet.chrome = Some(ids);
        let axes: Vec<usize> = visible.iter().map(|v| v.1).collect();
        let refit = !tick_layout_ready && { let _span = super::profile::span("facet.refresh.fit_labels"); self.fit_tick_label_text(&axes, true) };
        if refit {
            // Refined locators also own the marks: lay the chrome once more on the final ticks.
            self.refresh_facet_cell_chrome(visible, true);
        }
    }

    /// The figure title of a grid (`figure:title`).
    pub(crate) fn update_facet_title(&mut self, title: &str) {
        let id = match self.facet.title {
            Some(id) => id,
            None => {
                let text = FigureText {
                    visible: true, zorder: 3.0, x: 0.5, y: self.style.render.figure_title_y, text: String::new(),
                    size: self.style.fonts.figure_title_pt, color: self.rc.text_color.clone(), ha: HAlign::Center, va: VAlign::Top,
                    rotation: 0.0, frame: FigureFrame::Figure,
                };
                let id = self.fig.add_text(text);
                self.facet.title = Some(id);
                id
            }
        };
        let text = self.fig.text_mut(id).expect("figure title");
        text.text = title.to_owned();
        text.visible = !title.is_empty();
    }

    /// `_update_text_artists` of a grid: the outer labels, the opened cell's labels.
    pub(crate) fn update_facet_text(&mut self, x_label: &str, y_label: &str) {
        if let Some(ids) = self.facet.outer {
            for (id, value) in ids.iter().zip([x_label, y_label]) {
                let text = self.fig.text_mut(*id).expect("outer label");
                if text.text != value { text.text = value.to_owned(); }
            }
        }
        if let Some(index) = self.facet.focus {
            let axes = self.facet_cells()[index];
            let size = self.style.fonts.axis_label_pt;
            let model = &mut self.fig.axes[axes];
            if model.xaxis.label != x_label { model.xaxis.label = x_label.to_owned(); model.xaxis.label_size = size; }
            if model.yaxis.label != y_label { model.yaxis.label = y_label.to_owned(); model.yaxis.label_size = size; }
        }
    }
}
