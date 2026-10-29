//! Curve series on one axes, as `rendering.py` draws them: prepared series
//! (`_prepare_curve_series` with the fused summary pass), the native prepared
//! scene the kernels stroke (`curve:prepared`), its materialization as Line2D
//! and error-bar artists (`_mutate_series_artists`), dense-trace envelopes and
//! isolated glyphs, series colours and slots, and series focus (hover, lock,
//! scroll, inspector). The Curve kind is one surface of this; Rolling is the
//! same series over a shot history.
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use super::blake2s::blake2s;
use super::core::{Kind, Renderer};
use super::envelope::{bounded_isolated_points, pixel_columns, thinned_to_columns};
use super::labels::{literal_text, quantity_label, state_label};
use super::limits::{curve_x_limits, data_limits};
use super::mpl::axes::{AxesModel, Child, ChildId, LineArtist, SegmentArtist, TextArtist, TextFrame, TickPolicy, TickSignature, OffsetPin};
use super::mpl::color;
use crate::configuration::{DisplayState, Interaction, PlotSpec, Presentation, SeriesTarget, Value};
use crate::projection::nd::Scalar;
use crate::projection::output::{AxisValue, CurveSeries, Payload, QuantityArray};
use zlc_figure::ir::{HAlign, VAlign};

/// `series_identity(group_key)`: (domain, axis id, repr(canonical)) per value.
pub type SeriesId = Vec<[String; 3]>;

pub fn series_identity(group_key: &[AxisValue]) -> SeriesId {
    group_key.iter().map(|value| [value.axis.domain.value().to_owned(), value.axis.axis_id.clone(), scalar_repr(&value.canonical)]).collect()
}

/// Python `repr` of a canonical coordinate.
pub fn scalar_repr(value: &Scalar) -> String {
    match value {
        Scalar::None => "None".into(),
        Scalar::Bool(v) => if *v { "True".into() } else { "False".into() },
        Scalar::Int(v) => v.to_string(),
        Scalar::Float(v) => Value::Float(*v).py_repr(),
        Scalar::Text(v) => Value::Text(v.clone()).py_repr(),
    }
}

fn str_repr(text: &str) -> String { Value::text(text).py_repr() }

/// `repr(series_identity(...))`: a tuple of 3-tuples of strings.
fn identity_repr(identity: &SeriesId) -> String {
    let parts: Vec<String> = identity.iter()
        .map(|part| format!("({}, {}, {})", str_repr(&part[0]), str_repr(&part[1]), str_repr(&part[2])))
        .collect();
    match parts.len() { 1 => format!("({},)", parts[0]), _ => format!("({})", parts.join(", ")) }
}

/// `_series_slot`: source coordinate ordinals stay stable when a displayed subset changes.
pub fn series_slot(group_key: &[AxisValue], count: usize) -> usize {
    if group_key.is_empty() { return 0; }
    if group_key.len() == 1 {
        if let Some(index) = group_key[0].index { return index % count; }
    }
    let digest = blake2s(identity_repr(&series_identity(group_key)).as_bytes(), 2);
    (u16::from_be_bytes([digest[0], digest[1]]) as usize) % count
}

/// `_effective_cap_width`: bars closer than a device pixel apart on average paint their stems alone.
pub fn effective_cap_width(pixel_x: &[f64], cap_width: f64) -> f64 {
    let finite: Vec<f64> = pixel_x.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.len() > 1 {
        let max = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let min = finite.iter().copied().fold(f64::INFINITY, f64::min);
        if max - min < (finite.len() - 1) as f64 { return 0.0; }
    }
    cap_width
}

/// The fused pass: xmin/xmax, low/high bounds, isolated, pooled value+SEM bounds.
#[derive(Clone, Copy, Debug)]
pub struct Summary { pub xlow: f64, pub xhigh: f64, pub ylow: f64, pub yhigh: f64, pub isolated: bool, pub pooled: (f64, f64) }

/// `_PreparedSeries`.
#[derive(Clone, Debug)]
pub struct PreparedSeries {
    pub x: Arc<Vec<f64>>,
    pub y: Arc<Vec<f64>>,
    pub valid: Arc<Vec<bool>>,
    pub label: String,
    pub identity: SeriesId,
    pub group_key: Vec<AxisValue>,
    pub x_labels: Option<Vec<String>>,
    pub band: Option<(Arc<Vec<f64>>, Arc<Vec<f64>>)>,
    pub summary: Summary,
}

fn py_min(a: f64, b: f64) -> f64 { if b < a { b } else { a } }
fn py_max(a: f64, b: f64) -> f64 { if b > a { b } else { a } }

/// `prepare_curve_summary`: validity, display bounds and singleton presence in one pass.
fn summarize(x: &[f64], y: &[f64], source_valid: &[bool], band: Option<(&[f64], &[f64])>) -> (Vec<bool>, Summary) {
    let mut valid = vec![false; x.len()];
    let (mut xmin, mut xmax, mut ymin, mut ymax) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    let (mut pooled_min, mut pooled_max) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut isolated = false;
    let mut previous = false;
    for point in 0..x.len() {
        let (xv, yv) = (x[point], y[point]);
        let usable = source_valid[point] && xv.is_finite() && yv.is_finite();
        valid[point] = usable;
        if usable {
            let (mut low, mut high) = match band { Some((l, h)) => (l[point], h[point]), None => (yv, yv) };
            if !low.is_finite() { low = yv; }
            if !high.is_finite() { high = yv; }
            xmin = py_min(xmin, xv);
            xmax = py_max(xmax, xv);
            ymin = py_min(ymin, low);
            ymax = py_max(ymax, high);
            pooled_min = py_min(py_min(py_min(pooled_min, yv), low), high);
            pooled_max = py_max(py_max(py_max(pooled_max, yv), low), high);
            if !previous {
                let following = point + 1 < x.len() && source_valid[point + 1] && x[point + 1].is_finite() && y[point + 1].is_finite();
                if !following { isolated = true; }
            }
        }
        previous = usable;
    }
    (valid, Summary { xlow: xmin, xhigh: xmax, ylow: ymin, yhigh: ymax, isolated, pooled: (pooled_min, pooled_max) })
}

/// `_series_band`: display-unit y+/-sem bounds of one projected series.
fn series_band(item: &CurveSeries) -> Option<(Vec<f64>, Vec<f64>)> {
    let spread = item.sem.as_ref()?.to_f64();
    if !spread.iter().any(|v| v.is_finite()) { return None; }
    let quantity: &QuantityArray = &item.y;
    let canonical = quantity.canonical.to_f64();
    let unit = &quantity.canonical_unit;
    let target = &quantity.display_unit;
    let convert = |values: Vec<f64>| -> Vec<f64> {
        if unit == target { return values; }
        values.into_iter().map(|v| target.from_base(unit.to_base(v))).collect()
    };
    let low = convert(canonical.iter().zip(&spread).map(|(c, s)| c - s).collect());
    let high = convert(canonical.iter().zip(&spread).map(|(c, s)| c + s).collect());
    Some((low, high))
}

/// `_prepare_curve_series(source)` (kernels engaged).
pub fn prepare_curve_series(source: &[CurveSeries]) -> Vec<PreparedSeries> {
    source.iter().map(|item| {
        let x = item.x.display.to_f64();
        let y = item.y.display.to_f64();
        let band = series_band(item);
        let source_valid: Vec<bool> = item.valid.to_f64().iter().map(|v| *v != 0.0).collect();
        let (valid, summary) = summarize(&x, &y, &source_valid, band.as_ref().map(|(l, h)| (l.as_slice(), h.as_slice())));
        PreparedSeries {
            x: Arc::new(x), y: Arc::new(y), valid: Arc::new(valid), label: item.label.clone(),
            identity: series_identity(&item.group_key), group_key: item.group_key.clone(),
            x_labels: item.x_labels.clone(), band: band.map(|(l, h)| (Arc::new(l), Arc::new(h))), summary,
        }
    }).collect()
}

fn text_of(state: &DisplayState, name: &str) -> Option<String> { super::core::text_value(state.get(name)) }

/// `_curve_labels(semantic, source, state)`: the x and y labels one curve shows.
pub fn curve_labels(spec: &PlotSpec, source: &[CurveSeries], state: &DisplayState) -> (String, String) {
    let labels = spec.semantic().labels();
    let nonempty = |v: &Option<Arc<str>>| v.as_deref().filter(|s| !s.is_empty()).map(str::to_owned);
    let explicit_x = state_label(text_of(state, "x_label").as_deref(), nonempty(&labels.x).as_deref());
    let mut explicit_y = state_label(text_of(state, "y_label").as_deref(), None);
    if explicit_y.is_none() {
        let fallback = nonempty(&labels.y).or_else(|| labels.value.as_deref().map(str::to_owned));
        explicit_y = state_label(text_of(state, "value_label").as_deref(), fallback.as_deref());
    }
    let label = |quantity: &QuantityArray, fallback: &str, explicit: &Option<String>| {
        quantity_label(Some(&quantity.label), Some(&quantity.display_unit.symbol), fallback, explicit.as_deref())
    };
    match source.first() {
        Some(first) => (label(&first.x, "x", &explicit_x), label(&first.y, "value", &explicit_y)),
        None => (explicit_x.unwrap_or_else(|| "x".into()), explicit_y.unwrap_or_else(|| "value".into())),
    }
}

/// The native prepared curve scene (`curve:prepared`).
#[derive(Clone, Debug)]
pub struct PreparedCurve {
    pub series: Vec<Vec<PreparedSeries>>,
    pub state: Arc<DisplayState>,
    pub key: String,
    pub x_label: String,
    pub y_label: String,
    /// A grid's scene: the cells it materializes from.
    pub facet: Option<FacetCommand>,
}

/// What a grid's prepared scene materializes from: the cells' payload and
/// the grid's pooled limits (the `cells`/`options` of the facet command).
#[derive(Clone, Debug)]
pub struct FacetCommand { pub payload: Arc<Payload>, pub limits: Option<Limits> }

/// The (x, y) limits one surface is drawn at.
pub type Limits = ((f64, f64), (f64, f64));

/// One registered line source (`_line_sources`): the raw data behind a line.
#[derive(Clone, Debug)]
pub struct LineSource { pub axes: usize, pub x: Arc<Vec<f64>>, pub y: Arc<Vec<f64>>, pub isolated_glyphs: bool }

/// A focused series: (axes, identity, label, anchor x, anchor y).
#[derive(Clone, Debug, PartialEq)]
pub struct Focus { pub axes: usize, pub identity: SeriesId, pub label: String, pub x: f64, pub y: f64 }

/// The curve renderer's tables (the `_artists`/`_series_*` dicts).
#[derive(Clone, Debug, Default)]
pub struct CurveTables {
    pub prepared: Option<PreparedCurve>,
    /// `_ensure_lines` key -> (axes, line ids).
    pub lines: BTreeMap<String, (usize, Vec<ChildId>)>,
    /// line id (per axes) -> source.
    pub sources: HashMap<(usize, ChildId), LineSource>,
    /// axes -> [(line, identity, label)].
    pub series_lines: BTreeMap<usize, Vec<(ChildId, SeriesId, String)>>,
    /// axes -> identity -> bar ids (insertion ordered).
    pub series_bars: BTreeMap<usize, Vec<(SeriesId, Vec<ChildId>)>>,
    pub series_indices: BTreeMap<usize, Vec<(SeriesId, usize)>>,
    pub hover: Option<Focus>,
    pub hover_frozen: bool,
    pub locked: Option<Focus>,
    /// Wheel travel short of a whole notch, and the locked series it was
    /// made on: it counts only while that series is still the locked one.
    pub wheel: Option<(usize, Option<usize>, SeriesId, f64)>,
    /// Axes and opened facet qualify a press as well as its group identity.
    pub press: Option<(usize, Option<usize>, f64, f64, Option<SeriesId>)>,
    /// axes -> inspector annotation text id.
    pub annotations: BTreeMap<usize, ChildId>,
    pub focus_applied: BTreeMap<usize, FocusToken>,
}

pub type FocusToken = (Option<Focus>, Option<Focus>, Vec<(ChildId, SeriesId)>, Vec<(SeriesId, Vec<ChildId>)>);

/// The Curve kind.
pub struct CurveKind;

impl Kind for CurveKind {
    fn render(&self, renderer: &mut Renderer, payload: &Arc<Payload>, state: &DisplayState, axes: usize, key: &str) {
        renderer.update_curve(axes, payload, state, key, None, None, true);
    }
    fn key(&self) -> &'static str { "curve" }
}

pub(crate) fn curve_series(payload: &Payload) -> &[CurveSeries] {
    match payload { Payload::Curve(curve) => &curve.series, _ => panic!("curve payload must expose a series tuple") }
}

impl Renderer {
    /// `_native_curve_scene_supported`.
    pub(crate) fn native_curve_scene_supported(&self, series: &[PreparedSeries]) -> bool {
        if series.is_empty() { return false; }
        let token = &self.style.artists.curve;
        matches!(token.linestyle, "-" | "solid") && matches!(token.marker, None | Some("None") | Some("") | Some(" "))
    }

    /// `_update_curve(axes, payload, state, key, limits=..., prepared_series=..., paint_labels=...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_curve(&mut self, axes: usize, payload: &Arc<Payload>, state: &DisplayState, key: &str,
                        limits: Option<Limits>, prepared: Option<Vec<PreparedSeries>>, paint_labels: bool) {
        let source = curve_series(payload);
        let series = prepared.unwrap_or_else(|| prepare_curve_series(source));
        let (x_label, y_label) = curve_labels(&self.spec, source, state);
        let native_direct = !matches!(self.spec, PlotSpec::FacetGrid(_))
            && self.curve.hover.is_none() && self.curve.locked.is_none() && self.fit_overlays == 0
            && self.native_curve_scene_supported(&series);
        if native_direct {
            let mut extremes = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
            for item in &series {
                let s = item.summary;
                extremes[0] = py_min(extremes[0], s.xlow);
                extremes[1] = py_max(extremes[1], s.xhigh);
                extremes[2] = py_min(extremes[2], s.ylow);
                extremes[3] = py_max(extremes[3], s.yhigh);
            }
            let mut x_limits = if extremes[0].is_finite() { curve_x_limits(&extremes[..2]) } else { None };
            let y_range = if extremes[2].is_finite() { data_limits(&extremes[2..]) } else { None };
            let y_limits = match limits {
                Some((x, y)) => { x_limits = Some(x); y }
                None => self.resolve_curve_y_limits(key, y_range, state),
            };
            if let Some(x_limits) = x_limits { self.set_xlim(axes, x_limits); }
            self.set_ylim(axes, y_limits);
            self.set_axis_labels(axes, &x_label, &y_label);
            self.apply_curve_ticks(axes, &series);
            self.curve.prepared = Some(PreparedCurve {
                series: vec![series], state: Arc::new(state.clone()), key: key.to_owned(), x_label, y_label, facet: None,
            });
            self.withdraw_series_artists(axes);
            return;
        }
        self.curve.prepared = None;
        self.mutate_series_artists(axes, &series, state, key, &x_label, &y_label, limits, None, paint_labels, true);
    }

    pub(crate) fn resolve_curve_y_limits(&mut self, key: &str, y_range: Option<(f64, f64)>, state: &DisplayState) -> (f64, f64) {
        self.resolve_data_limits(key, y_range, state, "y", true)
    }

    pub(crate) fn set_axis_labels(&mut self, axes: usize, x_label: &str, y_label: &str) {
        let a = &mut self.fig.axes[axes];
        if a.xaxis.label != x_label { a.xaxis.label = x_label.to_owned(); }
        if a.yaxis.label != y_label { a.yaxis.label = y_label.to_owned(); }
    }

    /// `_apply_curve_ticks(axes, series, label_pt=tick_pt)`.
    pub(crate) fn apply_curve_ticks(&mut self, axes: usize, series: &[PreparedSeries]) {
        self.apply_curve_ticks_at(axes, series, self.style.fonts.tick_pt);
    }

    /// `_apply_curve_ticks(axes, series, label_pt=...)`.
    pub(crate) fn apply_curve_ticks_at(&mut self, axes: usize, series: &[PreparedSeries], label_pt: f64) {
        let labelled = series.iter().find(|item| item.x_labels.is_some());
        let a = &mut self.fig.axes[axes];
        match labelled {
            None => { apply_smart_ticks(a, true, label_pt); apply_smart_ticks(a, false, label_pt); }
            Some(item) => {
                apply_smart_ticks(a, false, label_pt);
                let names: Vec<String> = item.x_labels.as_ref().unwrap().iter().map(|n| literal_text(n)).collect();
                apply_named_ticks(a, true, item.x.as_slice(), &names, label_pt);
            }
        }
    }

    /// `_withdraw_series_artists`: the native scene owns the picture.
    pub(crate) fn withdraw_series_artists(&mut self, axes: usize) {
        let records = self.curve.series_lines.get(&axes).cloned().unwrap_or_default();
        for (line, _, _) in &records {
            self.fig.axes[axes].line_mut(*line).visible = false;
            self.curve.sources.remove(&(axes, *line));
        }
        // A line the fit mode hid behind its source scatter has no picture to
        // come back to: the scene strokes its data.
        self.fit.hidden_source_lines.retain(|(owner, line, _)| !(*owner == axes && records.iter().any(|r| r.0 == *line)));
        if let Some(bars) = self.curve.series_bars.get(&axes).cloned() {
            for (_, ids) in bars {
                for id in ids { set_child_visible(&mut self.fig.axes[axes], id, false); }
            }
        }
    }

    /// `_ensure_lines(axes, count, key)`.
    fn ensure_lines(&mut self, axes: usize, count: usize, key: &str) -> Vec<ChildId> {
        let style = self.style.clone();
        let rc = self.rc.clone();
        let entry = self.curve.lines.entry(key.to_owned()).or_insert((axes, Vec::new()));
        let plotted = entry.1.len() < count;
        while entry.1.len() < count {
            let index = entry.1.len();
            let curve = &style.artists.curve;
            let line = LineArtist {
                visible: true, zorder: 2.0, alpha: Some(curve.alpha), color: style.palette.line_color(index).to_owned(),
                linewidth: curve.linewidth, linestyle: curve.linestyle.to_owned(),
                marker: curve.marker.unwrap_or("None").to_owned(), markersize: style.artists.curve_marker_size_pt,
                markeredgewidth: rc.lines_markeredgewidth, markevery: None, marker_colors: None, label: String::new(),
                x: Vec::new(), y: Vec::new(),
            };
            let id = self.fig.axes[axes].add(Child::Line(line));
            entry.1.push(id);
        }
        let lines = entry.1.clone();
        if plotted { self.plotted(axes); }
        for (index, &line) in lines.iter().enumerate() {
            let visible = index < count;
            let artist = self.fig.axes[axes].line_mut(line);
            if artist.visible != visible { artist.visible = visible; }
            if !visible { self.curve.sources.remove(&(axes, line)); }
        }
        lines
    }

    /// `_apply_line_data`.
    pub(crate) fn apply_line_data(&mut self, axes: usize, line: ChildId, x: Arc<Vec<f64>>, y: Arc<Vec<f64>>, isolated_glyphs: bool) {
        self.curve.sources.insert((axes, line), LineSource { axes, x: x.clone(), y: y.clone(), isolated_glyphs });
        self.set_enveloped_line(axes, line, &x, &y, isolated_glyphs);
    }

    /// `_pixel_columns(axes)`: the x window an axes shows and its envelope columns.
    pub(crate) fn pixel_columns(&self, axes: usize) -> ((f64, f64), usize) {
        let a = &self.fig.axes[axes];
        let b = a.bbox_exact(self.fig.size_inches, self.fig.dpi);
        (a.xlim, pixel_columns(b[2] - b[0]))
    }

    /// `_set_enveloped_line`.
    fn set_enveloped_line(&mut self, axes: usize, line: ChildId, x: &[f64], y: &[f64], isolated_glyphs: bool) {
        let (window, columns) = self.pixel_columns(axes);
        let (drawn_x, drawn_y) = thinned_to_columns(x, y, window, columns);
        let (point_x, point_y) = if isolated_glyphs { bounded_isolated_points(x, y, window, columns) } else { (Vec::new(), Vec::new()) };
        let marker_style = self.style.artists.curve.marker;
        let size = self.style.artists.curve_marker_size_pt;
        let artist = self.fig.axes[axes].line_mut(line);
        let wanted;
        if !point_x.is_empty() {
            let marker_start = drawn_x.len() + 1;
            let mut xs = drawn_x;
            let mut ys = drawn_y;
            for (px, py) in point_x.iter().zip(&point_y) {
                xs.push(f64::NAN); ys.push(f64::NAN);
                xs.push(*px); ys.push(*py);
            }
            let extra = point_x.len() * 2;
            artist.x = xs;
            artist.y = ys;
            wanted = "_".to_owned();
            artist.markeredgewidth = artist.linewidth;
            artist.markevery = Some((marker_start..marker_start + extra).step_by(2).map(|v| v as i64).collect());
        } else {
            artist.x = drawn_x;
            artist.y = drawn_y;
            wanted = marker_style.unwrap_or("None").to_owned();
            if artist.markevery.is_some() { artist.markevery = None; }
        }
        if artist.marker != wanted { artist.marker = wanted; }
        if artist.markersize != size { artist.markersize = size; }
    }

    /// `_refresh_enveloped_lines(axis)`.
    pub(crate) fn refresh_enveloped_lines(&mut self, axes: usize) {
        let sources: Vec<((usize, ChildId), LineSource)> = self.curve.sources.iter()
            .filter(|((owner, _), _)| *owner == axes).map(|(k, v)| (*k, v.clone())).collect();
        for ((_, line), source) in sources {
            self.set_enveloped_line(axes, line, &source.x, &source.y, source.isolated_glyphs);
        }
    }

    /// `_paint_error_bars`: one series' bars, reusing last revision's artist.
    fn paint_error_bars(&mut self, axes: usize, reused: Option<Vec<ChildId>>, segments: Vec<[f64; 3]>, colour: &str, zorder: f64) -> Vec<ChildId> {
        if let Some(reused) = reused {
            if let Some(Child::Segments(s)) = self.fig.axes[axes].child_mut(reused[0]) { s.segments = segments; }
            return reused;
        }
        let policy = &self.style.render;
        let mut facecolor = color::rgba(colour);
        facecolor[3] = policy.uncertainty_bar_alpha;
        let bars = SegmentArtist {
            visible: true, zorder, alpha: policy.uncertainty_bar_alpha, facecolor,
            linewidth: policy.uncertainty_bar_linewidth, capsize: policy.uncertainty_bar_capsize_pt, segments,
        };
        vec![self.fig.axes[axes].add(Child::Segments(bars))]
    }

    /// `_mutate_series_artists`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn mutate_series_artists(&mut self, axes: usize, series: &[PreparedSeries], state: &DisplayState, key: &str,
                                        x_label: &str, y_label: &str, limits: Option<((f64, f64), (f64, f64))>,
                                        x_limits: Option<(f64, f64)>, paint_labels: bool, isolated_glyphs: bool) {
        let lines = self.ensure_lines(axes, series.len(), key);
        let mut extremes = [f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY];
        let mut series_lines = Vec::new();
        let cycle = self.style.palette.line_cycle();
        let mut previous_bars = self.curve.series_bars.remove(&axes).unwrap_or_default();
        let mut bars_by_series: Vec<(SeriesId, Vec<ChildId>)> = Vec::new();
        for (index, item) in series.iter().enumerate() {
            let colour = cycle[series_slot(&item.group_key, cycle.len())];
            let plotted_y: Vec<f64> = item.y.iter().zip(item.valid.iter()).map(|(&y, &v)| if v { y } else { f64::NAN }).collect();
            self.apply_line_data(axes, lines[index], item.x.clone(), Arc::new(plotted_y), isolated_glyphs);
            {
                let line = self.fig.axes[axes].line_mut(lines[index]);
                if line.color != colour { line.color = colour.to_owned(); }
                if line.label != item.label { line.label = item.label.clone(); }
            }
            series_lines.push((lines[index], item.identity.clone(), item.label.clone()));
            if let Some((low, high)) = &item.band {
                let mut segments = Vec::new();
                for point in 0..item.x.len() {
                    let (l, h) = (low[point], high[point]);
                    if item.valid[point] && l.is_finite() && h.is_finite() && h > l {
                        segments.push([item.x[point], l, h]);
                    }
                }
                if !segments.is_empty() {
                    let reused = previous_bars.iter().position(|(id, _)| *id == item.identity).map(|at| previous_bars.remove(at).1);
                    let zorder = self.fig.axes[axes].line(lines[index]).zorder - 0.1;
                    let bars = self.paint_error_bars(axes, reused, segments, colour, zorder);
                    for &bar in &bars { set_child_visible(&mut self.fig.axes[axes], bar, true); }
                    match bars_by_series.iter_mut().find(|(id, _)| *id == item.identity) {
                        Some(slot) => slot.1 = bars,
                        None => bars_by_series.push((item.identity.clone(), bars)),
                    }
                }
            }
            if limits.is_none() {
                let s = item.summary;
                extremes[0] = py_min(extremes[0], s.xlow);
                extremes[1] = py_max(extremes[1], s.xhigh);
                extremes[2] = py_min(extremes[2], s.ylow);
                extremes[3] = py_max(extremes[3], s.yhigh);
            }
        }
        for (_, ids) in previous_bars {
            for id in ids { self.fig.axes[axes].remove(id); }
        }
        self.curve.series_indices.insert(axes, series_lines.iter().enumerate().map(|(i, (_, id, _))| (id.clone(), i)).collect());
        self.curve.series_lines.insert(axes, series_lines);
        if !bars_by_series.is_empty() { self.curve.series_bars.insert(axes, bars_by_series); }
        match limits {
            Some((x, y)) => { self.set_xlim(axes, x); self.set_ylim(axes, y); }
            None => {
                let xlim = x_limits.or_else(|| if extremes[0].is_finite() { curve_x_limits(&extremes[0..2]) } else { None });
                let y_range = if extremes[2].is_finite() { data_limits(&extremes[2..4]) } else { None };
                if let Some(xlim) = xlim { self.set_xlim(axes, xlim); }
                let y_limits = self.resolve_curve_y_limits(key, y_range, state);
                self.set_ylim(axes, y_limits);
            }
        }
        if paint_labels {
            self.set_axis_labels(axes, x_label, y_label);
            self.apply_curve_ticks(axes, series);
        }
        self.apply_series_focus(Some(axes));
    }

    /// `_materialize_prepared_curve`: the artists of the scene in force, at
    /// the view in force.
    pub(crate) fn materialize_prepared_curve(&mut self) {
        let Some(command) = self.curve.prepared.take() else { return };
        if let Some(facet) = &command.facet {
            let Payload::Facet(cells) = &*facet.payload else { return };
            let requested = self.requested_view;
            for (key, axes, index) in self.painted_surfaces() {
                let index = index.expect("a grid surface is a cell");
                let limits = if requested.is_some() { Some(self.resolved_view_limits(axes, requested)) } else { facet.limits };
                let cell = Arc::new(cells.cells[index].payload.clone());
                self.update_curve(axes, &cell, &command.state, &key, limits, Some(command.series[index].clone()), false);
            }
            return;
        }
        if command.series.len() != 1 { return; }
        let primary = self.primary_axes();
        let limits = self.resolved_view_limits(primary, self.requested_view);
        let state = command.state.clone();
        self.mutate_series_artists(primary, &command.series[0], &state, &command.key, &command.x_label, &command.y_label,
                                   Some(limits), None, true, true);
    }

    // --------------------------------------------------------- series focus
    /// `_series_focus_allowed(axis_id)`.
    pub(crate) fn series_focus_allowed(&self, axes: usize) -> bool {
        if self.is_facet() && (self.facet.focus.is_none() || axes != self.primary_axes()) { return false; }
        if let Some(prepared) = &self.curve.prepared {
            return axes == self.primary_axes() && prepared.series.len() == 1 && prepared.series[0].len() > 1;
        }
        self.curve.series_lines.get(&axes).map_or(0, Vec::len) > 1
    }

    pub(crate) fn accepts_series_focus(&self, axes: Option<usize>) -> bool {
        axes.is_some_and(|axes| self.fig.axes[axes].visible && self.series_focus_allowed(axes))
    }

    /// `series_interaction()`.
    pub fn series_interaction(&self) -> Interaction {
        let facet_index = self.facet.focus.map(|index| index as i128);
        Interaction { series_lock: self.curve.locked.as_ref().map(|focus| SeriesTarget { key: target_key(&focus.identity), facet_index }) }
    }

    /// `series_presentation()`.
    pub fn series_presentation(&self) -> Presentation {
        let active = self.curve.locked.as_ref().or(self.curve.hover.as_ref());
        let facet_index = self.facet.focus.map(|index| index as i128);
        Presentation {
            series_readout: active.map(|focus| crate::configuration::SeriesReadout {
                target: SeriesTarget { key: target_key(&focus.identity), facet_index },
                label: focus.label.as_str().into(),
                mode: if self.curve.locked.is_some() { "locked".into() } else { "hover".into() },
            }),
        }
    }

    /// The first point of `_series_xy(artist)`: a line's first datum, a
    /// histogram group's outline start (its first edge, its first top).
    fn series_start(&self, axes: usize, line: ChildId) -> Option<(f64, f64)> {
        match self.fig.axes[axes].child(line) {
            Some(Child::Bars(bars)) => (!bars.counts.is_empty()).then(|| (bars.edges[0], bars.counts[0])),
            _ => {
                let artist = self.fig.axes[axes].line(line);
                (!artist.x.is_empty() && !artist.y.is_empty()).then(|| (artist.x[0], artist.y[0]))
            }
        }
    }

    /// `restore_series_interaction(interaction, presentation)`.
    pub fn restore_series_interaction(&mut self, interaction: &Interaction, presentation: Option<&Presentation>) {
        let previous_target = self.series_interaction().series_lock;
        let readout = presentation.and_then(|p| p.series_readout.clone());
        let target = interaction.series_lock.clone();
        let selected = target.clone().or_else(|| readout.as_ref().map(|r| r.target.clone()));
        // The hover this renderer already shows stays its own, cleared by a leave; one restored into a
        // renderer that did not show it (a peer, a saved Figure) holds until the pointer moves here.
        let own = self.curve.hover.as_ref().filter(|_| !self.curve.hover_frozen).map(|focus| focus.identity.clone());
        self.curve.locked = None;
        self.curve.hover = None;
        self.curve.hover_frozen = false;
        if let Some(selected) = selected {
            if selected.facet_index == self.facet.focus.map(|index| index as i128) {
                self.materialize_prepared_curve();
                let entries: Vec<(usize, Vec<(ChildId, SeriesId, String)>)> =
                    self.curve.series_lines.iter().map(|(k, v)| (*k, v.clone())).collect();
                'found: for (axes, records) in entries {
                    if !self.series_focus_allowed(axes) { continue; }
                    for (line, identity, label) in records {
                        if target_key(&identity) == selected.key {
                            let anchor = self.series_start(axes, line).unwrap_or((0.0, 0.0));
                            let text = match &readout {
                                Some(r) if r.target == selected => r.label.to_string(),
                                _ => label.clone(),
                            };
                            let focus = Focus { axes, identity: identity.clone(), label: text, x: anchor.0, y: anchor.1 };
                            if target.is_some() || readout.as_ref().is_some_and(|r| &*r.mode == "locked") {
                                self.curve.locked = Some(focus);
                            } else {
                                self.curve.hover_frozen = own.as_ref() != Some(&identity);
                                self.curve.hover = Some(focus);
                            }
                            break 'found;
                        }
                    }
                }
            }
        }
        if previous_target != self.series_interaction().series_lock {
            self.curve.press = None;
            self.curve.wheel = None;
        }
        self.apply_series_focus(None);
    }

    /// `_apply_series_focus(only)`.
    pub(crate) fn apply_series_focus(&mut self, only: Option<usize>) {
        for locked in [true, false] {
            let state = if locked { &self.curve.locked } else { &self.curve.hover };
            if let Some(focus) = state {
                let keep = self.series_focus_allowed(focus.axes)
                    && self.curve.series_lines.get(&focus.axes).is_some_and(|e| e.iter().any(|(_, id, _)| *id == focus.identity));
                if !keep { if locked { self.curve.locked = None } else { self.curve.hover = None } }
            }
        }
        if self.curve.press.as_ref().is_some_and(|(axes, facet, _, _, identity)|
            *facet != self.facet.focus || !self.series_focus_allowed(*axes)
            || identity.as_ref().is_some_and(|identity| !self.curve.series_lines.get(axes)
                .is_some_and(|entries| entries.iter().any(|(_, candidate, _)| candidate == identity)))) {
            self.curve.press = None;
        }
        if self.curve.wheel.as_ref().is_some_and(|(axes, facet, identity, _)|
            *facet != self.facet.focus || self.curve.locked.as_ref()
                .is_none_or(|locked| locked.axes != *axes || locked.identity != *identity)) {
            self.curve.wheel = None;
        }
        let locked = self.curve.locked.clone();
        let active = locked.clone().or_else(|| self.curve.hover.clone());
        self.update_rolling_meter();
        let lines = &self.curve.series_lines;
        self.curve.focus_applied.retain(|axes, _| lines.contains_key(axes));
        let axes_to_style: Vec<usize> = match only { Some(axes) => vec![axes], None => self.painted_surfaces().into_iter().map(|s| s.1).collect() };
        let mut pending = Vec::new();
        for &axes in &axes_to_style {
            let entries = self.curve.series_lines.get(&axes).cloned().unwrap_or_default();
            let bars = self.curve.series_bars.get(&axes).cloned().unwrap_or_default();
            let token: FocusToken = (locked.clone(), self.curve.hover.clone(),
                                     entries.iter().map(|(line, id, _)| (*line, id.clone())).collect(), bars);
            if self.curve.focus_applied.get(&axes) != Some(&token) { pending.push((axes, token)); }
        }
        if pending.is_empty() { return; }
        let identity = active.as_ref().map(|focus| focus.identity.clone());
        let mut focus_line: Option<(usize, ChildId)> = None;
        let curve = self.style.artists.curve.clone();
        let bar_alpha = self.style.render.uncertainty_bar_alpha;
        let bar_linewidth = self.style.render.uncertainty_bar_linewidth;
        let fill_alpha = self.style.artists.histogram_fill_alpha;
        for (axes, token) in pending {
            let entries = self.curve.series_lines.get(&axes).cloned().unwrap_or_default();
            let bars = self.curve.series_bars.get(&axes).cloned().unwrap_or_default();
            for (line_id, series_id, _) in &entries {
                let focused = identity.as_ref().is_some_and(|id| id == series_id);
                let (width, alpha, zorder) = if locked.is_some() {
                    (curve.linewidth * if focused { 2.0 } else { 1.0 }, if focused { 1.0 } else { 0.18 }, if focused { 4.0 } else { 2.0 })
                } else if active.is_some() {
                    (curve.linewidth * if focused { 1.45 } else { 1.0 }, if focused { 1.0 } else { curve.alpha }, if focused { 3.0 } else { 2.0 })
                } else {
                    (curve.linewidth, curve.alpha, 2.0)
                };
                let (series_bar_alpha, series_bar_width) = if locked.is_some() {
                    (if focused { bar_alpha } else { 0.06 }, bar_linewidth * if focused { 1.6 } else { 1.0 })
                } else if active.is_some() {
                    (if focused { (bar_alpha * 1.4).min(1.0) } else { bar_alpha }, bar_linewidth * if focused { 1.3 } else { 1.0 })
                } else {
                    (bar_alpha, bar_linewidth)
                };
                let model = &mut self.fig.axes[axes];
                let line_zorder = match model.child_mut(*line_id) {
                    Some(Child::Line(line)) => {
                        line.linewidth = width;
                        line.alpha = Some(alpha);
                        line.zorder = zorder;
                        if line.marker == "_" { line.markeredgewidth = line.linewidth; }
                        zorder
                    }
                    Some(Child::Bars(bars)) => {
                        // A histogram group's fill carries its emphasis, always at the bottom.
                        let mut fill = fill_alpha;
                        if locked.is_some() { fill *= if focused { 1.6 } else { 0.25 }; }
                        else if active.is_some() && focused { fill *= 1.3; }
                        bars.linewidth = width;
                        bars.alpha = if fill < 1.0 { fill } else { 1.0 };
                        bars.zorder = 1.0;
                        1.0
                    }
                    _ => zorder,
                };
                if let Some((_, ids)) = bars.iter().find(|(id, _)| id == series_id) {
                    for &bar in ids {
                        if let Some(Child::Segments(s)) = model.child_mut(bar) {
                            s.alpha = series_bar_alpha;
                            s.facecolor[3] = series_bar_alpha;
                            s.zorder = line_zorder - 0.1;
                            s.linewidth = series_bar_width;
                        }
                    }
                }
                if focused && active.as_ref().is_some_and(|a| a.axes == axes) { focus_line = Some((axes, *line_id)); }
            }
            self.curve.focus_applied.insert(axes, token);
        }
        if only.is_none() { self.refresh_fit_focus(); }
        if let Some(only) = only {
            if active.as_ref().is_none_or(|a| a.axes != only) { return; }
        }
        let annotations: Vec<(usize, ChildId)> = self.curve.annotations.iter().map(|(k, v)| (*k, *v)).collect();
        for (axes, id) in annotations { set_child_visible(&mut self.fig.axes[axes], id, false); }
        if self.classifier_annotation_owns_inspector() { return; }
        let (Some(active), Some((axes, line))) = (active, focus_line) else { return };
        let colour = self.series_colour(axes, line);
        let size = self.style.fonts.annotation_pt;
        let id = match self.curve.annotations.get(&axes) {
            Some(&id) => id,
            None => {
                let text = TextArtist {
                    visible: true, zorder: 12.0, x: 0.98, y: 0.98, text: String::new(), color: self.rc.text_color.clone(),
                    size, ha: HAlign::Right, va: VAlign::Top, ma: None, clip_on: true, frame: TextFrame::Axes,
                    family: if self.in_style { None } else { Some(vec!["sans-serif".to_owned()]) },
                };
                let id = self.fig.axes[axes].add(Child::Text(text));
                self.curve.annotations.insert(axes, id);
                id
            }
        };
        let label = if active.label.is_empty() { "Series".to_owned() } else { literal_text(&active.label) };
        if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(id) {
            text.text = format!("{}{}", if locked.is_some() { "* " } else { "" }, label);
            text.color = colour;
            text.visible = true;
        }
    }
}

impl Renderer {
    /// `_series_hit(axes, px, py, radius)`: the series under a pointer (pixels, bottom-left origin).
    pub(crate) fn series_hit(&mut self, axes: Option<usize>, px: f64, py: f64, radius: f64) -> Option<Focus> {
        self.materialize_prepared_curve();
        let axes = axes?;
        if !self.fig.axes[axes].visible { return None; }
        let reach = radius * self.plan.device_pixel_ratio;
        let mut best = reach * reach;
        let mut hit = None;
        let current = self.curve.hover.as_ref().filter(|f| f.axes == axes).map(|f| f.identity.clone());
        let mut entries = self.curve.series_lines.get(&axes).cloned().unwrap_or_default();
        if let Some(current) = &current {
            let (first, rest): (Vec<_>, Vec<_>) = entries.into_iter().partition(|(_, id, _)| id == current);
            entries = first.into_iter().chain(rest).collect();
        }
        let m = self.fig.axes[axes].trans_data(self.fig.size_inches, self.fig.dpi);
        let (x_scale, y_scale) = (self.axis_scale(axes, true), self.axis_scale(axes, false));
        let xlim = self.fig.axes[axes].xlim;
        for (line_id, identity, label) in entries {
            if let Some(Child::Bars(bars)) = self.fig.axes[axes].child(line_id) {
                // A histogram group is hit inside one of its bars.
                if !bars.visible { continue; }
                // trans_data is only the affine part of transData. Undo both
                // scales too (including a FuncScale retained from image cells).
                let data_x = x_scale.value((px - m.0[0][2]) / m.0[0][0]);
                let data_y = y_scale.value((py - m.0[1][2]) / m.0[1][1]);
                if !data_x.is_finite() || !data_y.is_finite() { continue; }
                let index = bars.edges.partition_point(|&edge| edge <= data_x) as i64 - 1;
                if 0 <= index && (index as usize) < bars.counts.len() && 0.0 < data_y && data_y <= bars.counts[index as usize] {
                    hit = Some(Focus { axes, identity: identity.clone(), label: label.clone(), x: data_x, y: data_y });
                    best = 0.0;
                    if current.as_ref() == Some(&identity) { return hit; }
                }
                continue;
            }
            let line = self.fig.axes[axes].line(line_id);
            // In fit mode the first source line is replaced by a scatter.
            // v2 keeps that source selectable using exactly the visible dots,
            // without inventing connecting segments between those dots.
            let scatter = (!line.visible && self.fit.hidden_source_lines.iter()
                .any(|&(owner, source, visible)| owner == axes && source == line_id && visible))
                .then(|| self.fit.source_scatter).flatten()
                .filter(|(owner, _)| *owner == axes)
                .and_then(|(_, id)| match self.fig.axes[axes].child(id) {
                    Some(Child::Scatter(scatter)) if scatter.visible => Some(scatter), _ => None,
                });
            if !line.visible && scatter.is_none() { continue; }
            let (mut x, mut y): (Vec<f64>, Vec<f64>) = match scatter {
                Some(scatter) => scatter.offsets.iter().map(|point| (point[0], point[1])).unzip(),
                None => (line.x.clone(), line.y.clone()),
            };
            let mut sources: Option<Vec<usize>> = if scatter.is_none() && line.marker == "_" {
                line.markevery.as_ref().map(|m| m.iter().map(|v| *v as usize).collect())
            } else { None };
            if x.len() > 4096 * 4 && sources.is_none() && x.iter().all(|v| v.is_finite()) && x.windows(2).all(|w| w[1] >= w[0]) {
                let (low, high) = if xlim.0 <= xlim.1 { xlim } else { (xlim.1, xlim.0) };
                let start = x.partition_point(|v| *v < low).saturating_sub(1);
                let stop = (x.partition_point(|v| *v <= high) + 1).min(x.len());
                x = x[start..stop].to_vec();
                y = y[start..stop].to_vec();
            }
            let mut finite: Vec<bool> = x.iter().zip(&y).map(|(a, b)| a.is_finite() && b.is_finite()).collect();
            let pixels: Vec<(f64, f64)> = x.iter().zip(&y).zip(&finite)
                .map(|((&a, &b), &f)| if f {
                    let (sx, sy) = self.fig.axes[axes].scaled(a, b);
                    m.apply(sx, sy)
                } else { (f64::NAN, f64::NAN) }).collect();
            for (valid, point) in finite.iter_mut().zip(&pixels) { *valid &= point.0.is_finite() && point.1.is_finite(); }
            let mut low = (f64::INFINITY, f64::INFINITY);
            let mut high = (f64::NEG_INFINITY, f64::NEG_INFINITY);
            for (p, f) in pixels.iter().zip(&finite) {
                if !*f { continue; }
                if p.0 < low.0 { low.0 = p.0; }
                if p.1 < low.1 { low.1 = p.1; }
                if p.0 > high.0 { high.0 = p.0; }
                if p.1 > high.1 { high.1 = p.1; }
            }
            let finite_count = finite.iter().filter(|f| **f).count();
            if scatter.is_some() || sources.is_none() && finite_count == 1 {
                sources = Some(finite.iter().enumerate().filter(|(_, f)| **f).map(|(i, _)| i).collect());
            }
            let sources = sources.unwrap_or_default();
            if px < low.0 - reach || px > high.0 + reach || py < low.1 - reach || py > high.1 + reach { continue; }
            let adjacent: Vec<usize> = (0..finite.len().saturating_sub(1)).filter(|&i| finite[i] && finite[i + 1]).collect();
            if !adjacent.is_empty() && scatter.is_none() {
                let mut best_local = (f64::INFINITY, 0usize, 0.0f64);
                for &i in &adjacent {
                    let start = pixels[i];
                    let delta = (pixels[i + 1].0 - start.0, pixels[i + 1].1 - start.1);
                    let length = delta.0 * delta.0 + delta.1 * delta.1;
                    let mut t = if length > 0.0 { ((px - start.0) * delta.0 + (py - start.1) * delta.1) / length } else { 0.0 };
                    t = t.clamp(0.0, 1.0);
                    let closest = (start.0 + t * delta.0 - px, start.1 + t * delta.1 - py);
                    let distance = closest.0 * closest.0 + closest.1 * closest.1;
                    if distance < best_local.0 { best_local = (distance, i, t); }
                }
                let (distance, source, t) = best_local;
                if distance <= best {
                    best = distance;
                    hit = Some(Focus { axes, identity: identity.clone(), label: label.clone(),
                                       x: x[source] + t * (x[source + 1] - x[source]),
                                       y: y[source] + t * (y[source + 1] - y[source]) });
                    if current.as_ref() == Some(&identity) { return hit; }
                }
            }
            if !sources.is_empty() {
                let mut best_local = (f64::INFINITY, 0usize);
                for &s in &sources {
                    let p = pixels[s];
                    let distance = (p.0 - px) * (p.0 - px) + (p.1 - py) * (p.1 - py);
                    if distance < best_local.0 { best_local = (distance, s); }
                }
                let (distance, source) = best_local;
                if distance <= best {
                    best = distance;
                    hit = Some(Focus { axes, identity: identity.clone(), label: label.clone(), x: x[source], y: y[source] });
                    if current.as_ref() == Some(&identity) { return hit; }
                }
            }
        }
        hit
    }

    fn focus_state(&self) -> (&'static str, Option<(usize, SeriesId)>) {
        let active = self.curve.locked.as_ref().or(self.curve.hover.as_ref());
        (if self.curve.locked.is_some() { "locked" } else if self.curve.hover.is_some() { "hover" } else { "none" },
         active.map(|f| (f.axes, f.identity.clone())))
    }

    /// `series_focus(action, axes, px, py, hit_radius=..., click_radius=..., redraw=False)`.
    pub fn series_focus(&mut self, action: &str, axes: Option<usize>, px: f64, py: f64, hit_radius: f64, click_radius: f64) -> bool {
        if action == "leave" && self.curve.hover_frozen { return false; }
        if matches!(action, "move" | "press" | "clear") { self.curve.hover_frozen = false; }
        let before = self.curve.locked.clone().or_else(|| self.curve.hover.clone());
        let before_state = self.focus_state();
        let mut handled = false;
        match action {
            "press" => {
                if !self.accepts_series_focus(axes) { return false; }
                let hit = self.series_hit(axes, px, py, hit_radius);
                self.curve.press = Some((axes.unwrap(), self.facet.focus, px, py, hit.map(|h| h.identity)));
                return false;
            }
            "move" => {
                if self.curve.locked.is_some() { return false; }
                let hit = if self.accepts_series_focus(axes) { self.series_hit(axes, px, py, hit_radius) } else { None };
                if before.as_ref().map(|f| (f.axes, &f.identity)) == hit.as_ref().map(|f| (f.axes, &f.identity)) { return false; }
                self.curve.hover = hit;
            }
            "release" => {
                let press = self.curve.press.take();
                if !self.accepts_series_focus(axes) { return false; }
                let Some(press) = press else { return false };
                if axes != Some(press.0) || self.facet.focus != press.1 { return false; }
                if (px - press.2).hypot(py - press.3) > click_radius * self.plan.device_pixel_ratio { return false; }
                let hit = self.series_hit(axes, px, py, hit_radius);
                if press.4 != hit.as_ref().map(|h| h.identity.clone()) { return false; }
                handled = hit.is_some() || before.is_some();
                self.curve.hover = None;
                let same = hit.is_some() && self.curve.locked.is_some();
                self.curve.locked = match hit {
                    None => None,
                    Some(hit) => {
                        if same && self.curve.locked.as_ref().map(|l| (l.axes, &l.identity)) == Some((hit.axes, &hit.identity)) { None } else { Some(hit) }
                    }
                };
            }
            "leave" => {
                self.curve.press = None;
                if self.curve.locked.is_some() { return false; }
                self.curve.hover = None;
            }
            "clear" => {
                self.curve.press = None;
                self.curve.wheel = None;
                self.curve.hover = None;
                self.curve.locked = None;
            }
            _ => {}
        }
        if before_state == self.focus_state() { return handled; }
        self.apply_series_focus(None);
        if action == "release" { handled } else { true }
    }

    /// `series_focus_scroll(axes, step, redraw=False)`.
    pub fn series_focus_scroll(&mut self, axes: Option<usize>, step: f64) -> bool {
        let Some(locked) = self.curve.locked.clone() else { return false };
        if axes != Some(locked.axes) || locked.identity.is_empty() || !self.accepts_series_focus(axes) { return false; }
        let entries = self.curve.series_lines.get(&locked.axes).cloned().unwrap_or_default();
        let current = self.curve.series_indices.get(&locked.axes)
            .and_then(|v| v.iter().find(|(id, _)| *id == locked.identity).map(|(_, i)| *i));
        let Some(current) = current else { return false };
        if entries.is_empty() { return false; }
        // A whole notch is one series: a hi-res wheel or a touchpad sends a
        // notch as many small steps, each of which used to move a series.
        // The travel short of a notch is kept for the series it was made on.
        let travel = step + match &self.curve.wheel {
            Some((axes, facet, identity, rest)) if *axes == locked.axes && *facet == self.facet.focus && *identity == locked.identity => *rest,
            _ => 0.0,
        };
        // Steps summed in binary land a hair short of a whole notch (0.3 * 3 + 0.1).
        let notches = if (travel - travel.round()).abs() <= 1e-9 { travel.round() } else { travel.trunc() };
        let last = entries.len() as f64 - 1.0;
        let target = (current as f64 - notches).clamp(0.0, last);
        // At an end the travel past it is dropped: the first notch back moves.
        let beyond = (target == 0.0 && travel > 0.0) || (target == last && travel < 0.0);
        let rest = if beyond { 0.0 } else { travel - notches };
        let target = target as usize;
        if target == current {
            self.curve.wheel = Some((locked.axes, self.facet.focus, locked.identity, rest));
            return true;
        }
        let (line, identity, label) = entries[target].clone();
        let mut anchor = (locked.x, locked.y);
        if let Some(start) = self.series_start(locked.axes, line).filter(|s| s.0.is_finite() && s.1.is_finite()) { anchor = start; }
        self.curve.wheel = Some((locked.axes, self.facet.focus, identity.clone(), rest));
        self.curve.locked = Some(Focus { axes: locked.axes, identity, label, x: anchor.0, y: anchor.1 });
        self.apply_series_focus(None);
        true
    }
}

pub(crate) fn target_key(identity: &SeriesId) -> Vec<[Arc<str>; 3]> {
    identity.iter().map(|part| [Arc::from(part[0].as_str()), Arc::from(part[1].as_str()), Arc::from(part[2].as_str())]).collect()
}

pub(crate) fn set_child_visible(axes: &mut AxesModel, id: ChildId, visible: bool) {
    if let Some(child) = axes.child_mut(id) { child.set_visible(visible); }
}

/// `apply_smart_ticks(axes, name, label_pt=...)` for one axis.
pub fn apply_smart_ticks(axes: &mut AxesModel, x: bool, size_pt: f64) {
    let axis = axes.axis_mut(x);
    let scale = if axis.function.is_some() { "function" } else if axis.log { "log" } else { "linear" };
    let signature = TickSignature::Smart { name: if x { 'x' } else { 'y' }, scale: scale.into(), size_pt: size_pt.to_bits() };
    if axis.signature.as_ref() == Some(&signature) && matches!(axis.policy, TickPolicy::Smart(_)) { return; }
    axis.policy = TickPolicy::Smart(Box::new(crate::ticks::SmartOffsetLocator::new(size_pt)));
    // A log axis keeps its 2..9 ticks as unlabelled guides; a linear one has no minor ticks.
    axis.minor_log = axis.log;
    axis.offset_pin = Some(if x {
        OffsetPin { xy: (0.995, 0.008), ha: HAlign::Right, va: VAlign::Bottom }
    } else {
        OffsetPin { xy: (0.008, 0.995), ha: HAlign::Left, va: VAlign::Top }
    });
    axis.offset_visible = true;
    axis.offset_size = size_pt;
    axis.set_label_size_all(size_pt);
    axis.signature = Some(signature);
}

/// `apply_named_ticks(axes, name, positions, names, label_pt=...)`.
pub fn apply_named_ticks(axes: &mut AxesModel, x: bool, positions: &[f64], names: &[String], size_pt: f64) {
    let axis = axes.axis_mut(x);
    let signature = TickSignature::Named { ticks: positions.iter().map(|v| v.to_bits()).collect(), texts: names.to_vec(), size_pt: size_pt.to_bits() };
    if axis.signature.as_ref() == Some(&signature) { return; }
    axis.policy = TickPolicy::Named { ticks: positions.to_vec(), texts: names.to_vec() };
    axis.offset_visible = false;
    axis.set_label_size_all(size_pt);
    axis.signature = Some(signature);
}
