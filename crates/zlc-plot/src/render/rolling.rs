//! The Rolling kind (`_update_rolling`): one shot history drawn as curve
//! series on its history axes, framed by the configured window (index x) or
//! the displayed times; the live meter above it (`_update_rolling_meter`,
//! pinned with `_pin_readout`); and the side distribution of the window's
//! values beside it (`_update_horizontal_histogram`, declared count ticks).
use std::sync::Arc;

use zlc_figure::ir::{HAlign, VAlign};

use super::core::{Kind, Renderer};
use super::curve::{curve_labels, curve_series, prepare_curve_series, series_identity, series_slot, PreparedCurve};
use super::labels::literal_text;
use super::limits::{curve_x_limits, data_limits};
use super::mpl::axes::{Child, ChildId, DeclaredState, DeclaredText, PolyArtist, TextArtist, TextFrame, TickPolicy, TickSignature};
use super::mpl::color;
use crate::configuration::{DisplayState, PlotSpec, Value};
use crate::projection::histogram::linspace;
use crate::projection::kernels::uniform_histogram;
use crate::projection::output::Payload;

/// The rolling renderer's tables (`rolling:latest`, the distribution).
#[derive(Clone, Debug, Default)]
pub struct RollingTables {
    /// (history axes, meter text).
    pub latest: Option<(usize, ChildId)>,
    /// (distribution axes, collection).
    pub distribution: Option<(usize, ChildId)>,
    pub count_ceiling: f64,
}

/// Where the meter's widest text ends: above the frame, at its right.
pub const ROLLING_METER_RIGHT: f64 = 0.97;

pub struct RollingKind;

impl Kind for RollingKind {
    fn render(&self, renderer: &mut Renderer, payload: &Arc<Payload>, state: &DisplayState, axes: usize, key: &str) {
        renderer.update_rolling(axes, payload, state, key);
    }
    fn key(&self) -> &'static str { "rolling" }
}

fn int_value(state: &DisplayState, name: &str) -> i64 {
    match state.get(name) { Some(Value::Int(v)) => *v as i64, Some(Value::Float(v)) => *v as i64, other => panic!("{name}: {other:?}") }
}

/// numpy's `f"{value:.6g}"`.
fn format_6g(value: f64) -> String { crate::pyfmt::format_g(value, 6) }

/// `_DIGIT.sub("0", text)` (ASCII digits).
pub(crate) fn zero_digits(text: &str) -> String { text.chars().map(|c| if c.is_ascii_digit() { '0' } else { c }).collect() }

impl Renderer {
    /// `_update_rolling(axes, payload, state, key)`.
    pub fn update_rolling(&mut self, history: usize, payload: &Arc<Payload>, state: &DisplayState, key: &str) {
        let series = curve_series(payload);
        let sliced = prepare_curve_series(series);
        let (x_text, y_label) = curve_labels(&self.spec, series, state);
        let window = int_value(state, "window");
        let primary_index = matches!(&**payload, Payload::Curve(curve) if curve.x_role.as_deref() == Some("primary-index"));
        let frame = if primary_index {
            curve_x_limits(&[1.0 - window as f64, 0.0])
        } else {
            let extents: Vec<f64> = sliced.iter().flat_map(|item| [item.summary.xlow, item.summary.xhigh]).collect();
            curve_x_limits(&extents)
        };
        let native_direct = self.curve.hover.is_none() && self.curve.locked.is_none() && self.fit_overlays == 0
            && self.native_curve_scene_supported(&sliced);
        let history_key = format!("{key}:history");
        if native_direct {
            let mut extremes = [f64::INFINITY, f64::NEG_INFINITY];
            for item in &sliced {
                if item.summary.ylow < extremes[0] { extremes[0] = item.summary.ylow; }
                if item.summary.yhigh > extremes[1] { extremes[1] = item.summary.yhigh; }
            }
            let y_range = if extremes[0].is_finite() { data_limits(&extremes) } else { None };
            if let Some(frame) = frame { self.set_xlim(history, frame); }
            let y_limits = self.resolve_curve_y_limits(&history_key, y_range, state);
            self.set_ylim(history, y_limits);
            self.set_axis_labels(history, &x_text, &y_label);
            let tick_pt = self.style.fonts.tick_pt;
            super::curve::apply_smart_ticks(&mut self.fig.axes[history], true, tick_pt);
            super::curve::apply_smart_ticks(&mut self.fig.axes[history], false, tick_pt);
            self.curve.prepared = Some(PreparedCurve {
                series: vec![sliced.clone()], state: Arc::new(state.clone()), key: history_key.clone(),
                x_label: x_text.clone(), y_label: y_label.clone(), facet: None,
            });
            self.withdraw_series_artists(history);
        } else {
            self.curve.prepared = None;
            self.mutate_series_artists(history, &sliced, state, &history_key, &x_text, &y_label, None, frame, true, false);
        }
        if self.rolling.latest.is_none() {
            let text = TextArtist {
                visible: true, zorder: 3.0, x: ROLLING_METER_RIGHT, y: 1.0, text: String::new(),
                color: self.style.palette.readout.to_owned(), size: self.style.fonts.annotation_pt,
                ha: HAlign::Left, va: VAlign::Bottom, ma: None, clip_on: false,
                frame: TextFrame::AxesOffset(0.0, self.style.render.compact_axes_title_pad_pt / 72.0),
                family: None,
            };
            let id = self.fig.axes[history].add(Child::Text(text));
            self.rolling.latest = Some((history, id));
            self.update_rolling_meter();
        } else if native_direct {
            self.update_rolling_meter();
        }
        if let Some(&side) = self.axes_of("distribution").first() {
            let policy = self.style.render.clone();
            let mut values = Vec::new();
            let mut history_length = 0usize;
            for item in &sliced {
                let sample: Vec<f64> = item.y.iter().zip(item.valid.iter()).filter(|(_, v)| **v).map(|(y, _)| *y).collect();
                history_length = history_length.max(sample.len());
                values.extend(sample);
            }
            let requested_bins = int_value(state, "bin_count").max(0) as usize;
            let bin_count = policy.rolling_distribution_min_bins.max(requested_bins
                .min((history_length / policy.distribution_bin_divisor).max(policy.rolling_distribution_min_bins))
                .min(policy.distribution_max_bins));
            let y_limits = match self.requested_view { Some((_, Some(y))) => y, _ => self.fig.axes[history].ylim };
            let (low, high) = if y_limits.0 <= y_limits.1 { y_limits } else { (y_limits.1, y_limits.0) };
            let (counts, edges) = np_histogram(&values, bin_count, (low, high));
            let slot = (self.rolling.distribution, self.rolling.count_ceiling);
            (self.rolling.distribution, self.rolling.count_ceiling) = self.update_horizontal_histogram(side, slot, &edges, &counts, y_limits, false);
        }
    }

    /// `_update_horizontal_histogram(axes, key, edges, counts, y_limits, tick_profile=...)` on one
    /// rail slot (its collection and count ceiling), returned updated; `image` is the image profile.
    pub(crate) fn update_horizontal_histogram(&mut self, axes: usize, slot: (Option<(usize, ChildId)>, f64), edges: &[f64],
                                              counts: &[i64], y_limits: (f64, f64), image: bool) -> (Option<(usize, ChildId)>, f64) {
        let policy = self.style.render.clone();
        let verts = histogram_vertices_swapped(edges, counts);
        let (mut artist, mut ceiling) = slot;
        match artist {
            None => {
                let mut facecolor = color::rgba(self.style.palette.hist_fill);
                facecolor[3] = policy.side_distribution_fill_alpha;
                let id = self.fig.axes[axes].add(Child::Poly(PolyArtist {
                    visible: true, zorder: 1.0, alpha: Some(policy.side_distribution_fill_alpha), facecolor, verts,
                }));
                artist = Some((axes, id));
                let tick_pt = self.style.fonts.tick_pt;
                apply_declared_ticks(&mut self.fig.axes[axes], true, policy.distribution_tick_count, tick_pt);
                let model = &mut self.fig.axes[axes];
                if image {
                    model.xaxis.inward_ends = policy.image_distribution_counts_inside;
                    let params = &mut model.yaxis.major;
                    params.tick1on = true;
                    params.tick2on = false;
                    params.label1on = false;
                    params.label2on = false;
                } else {
                    model.xaxis.label = String::new();
                    model.yaxis.label = String::new();
                    for params in [&mut model.yaxis.major, &mut model.yaxis.minor] {
                        params.tick1on = false;
                        params.tick2on = false;
                        params.label1on = false;
                    }
                }
                // `set_tick_params` with a label switch shows the offset text with the labels.
                model.yaxis.offset_visible = model.yaxis.major.label1on || model.yaxis.major.label2on;
            }
            Some((owner, id)) => {
                if let Some(Child::Poly(poly)) = self.fig.axes[owner].child_mut(id) { poly.verts = verts; }
            }
        }
        let peak = counts.iter().copied().max().map_or(0.0, |v| v as f64);
        let grown = (peak + policy.distribution_count_headroom).max(peak * policy.distribution_count_growth);
        let wanted = (policy.distribution_count_floor as f64).max((grown as i64) as f64);
        if ceiling <= 0.0 || wanted > ceiling || wanted < policy.distribution_count_shrink_ratio * ceiling { ceiling = wanted; }
        let model = &mut self.fig.axes[axes];
        if model.xlim != (0.0, ceiling) { model.set_xlim(0.0, ceiling); }
        if model.ylim != y_limits { model.set_ylim(y_limits.0, y_limits.1); }
        (artist, ceiling)
    }

    /// `_update_rolling_meter`: the highlighted series' newest value (the locked one, else the
    /// hovered one, else the first), between frames too.
    pub(crate) fn update_rolling_meter(&mut self) {
        if !matches!(self.spec, PlotSpec::Rolling(_)) { return; }
        let Some((axes, id)) = self.rolling.latest else { return };
        let Some(payload) = self.last_payload.clone() else { return };
        let series = curve_series(&payload);
        let identity = self.curve.locked.as_ref().or(self.curve.hover.as_ref()).map(|focus| focus.identity.clone());
        let item = match &identity {
            Some(identity) => series.iter().find(|item| series_identity(&item.group_key) == *identity),
            None => series.first(),
        };
        let mut latest = None;
        let mut label = String::new();
        let mut colour = None;
        if let Some(item) = item {
            let values = item.y.display.to_f64();
            let valid: Vec<bool> = item.valid.to_f64().iter().zip(&values).map(|(v, y)| *v != 0.0 && y.is_finite()).collect();
            if let Some(last) = valid.iter().rposition(|v| *v) { latest = Some(values[last]); }
            if series.len() > 1 { label = item.label.clone(); }
            let cycle = self.style.palette.line_cycle();
            colour = Some(cycle[series_slot(&item.group_key, cycle.len())].to_owned());
        }
        let value = latest.map_or_else(|| "\u{2014}".to_owned(), format_6g);
        let words = if label.is_empty() { String::new() } else { format!("{} \u{b7} ", literal_text(&label)) };
        if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(id) {
            if let Some(colour) = colour { text.color = colour; }
            text.text = format!("{words}{value}");
        }
        let reserve = [format!("{words}-0.000000000"), format!("{words}-0.00000e+00")];
        self.pin_readout(axes, id, ROLLING_METER_RIGHT, &reserve);
    }

    /// `_pin_readout(artist, right, reserve=...)`: the words stand still while the number changes.
    pub(crate) fn pin_readout(&mut self, axes: usize, id: ChildId, right: f64, reserve: &[String]) {
        let dpi = self.fig.dpi;
        let bbox = self.fig.axes[axes].bbox_exact(self.fig.size_inches, dpi);
        let Some(Child::Text(text)) = self.fig.axes[axes].child(id) else { return };
        if text.text.is_empty() || dpi <= 0.0 { return; }
        let nominal = text.size;
        let measure = self.measure.clone();
        let measured = |content: &str| content.split('\n')
            .map(|line| measure.width(&zero_digits(&line.replace("\\$", "$")), nominal))
            .fold(f64::NEG_INFINITY, f64::max);
        let shown = measured(&text.text);
        let mut width = shown;
        for item in reserve { width = width.max(measured(item)); }
        let room = (bbox[2] - bbox[0]) * 72.0 / dpi;
        if room <= 0.0 { return; }
        let scale = nominal / nominal;
        let figure_edge = right * room + bbox[0] * 72.0 / dpi;
        let drawn = (width * scale).min(figure_edge.max(shown * scale));
        if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(id) {
            text.ha = HAlign::Left;
            text.ma = Some(HAlign::Left);
            text.x = right - drawn / room;
        }
    }
}

/// `_histogram_vertices(edges, counts)[..., ::-1]`: horizontal bars.
fn histogram_vertices_swapped(edges: &[f64], counts: &[i64]) -> Vec<Vec<[f64; 2]>> {
    (0..counts.len()).map(|i| {
        let c = counts[i] as f64;
        vec![[0.0, edges[i]], [c, edges[i]], [c, edges[i + 1]], [0.0, edges[i + 1]]]
    }).collect()
}

/// `np.histogram(values[isfinite(values)], bins=n, range=(low, high))`.
pub fn np_histogram(values: &[f64], bins: usize, range: (f64, f64)) -> (Vec<i64>, Vec<f64>) {
    let (mut first, mut last) = range;
    if first == last { first -= 0.5; last += 0.5; }
    let edges = linspace(first, last, bins + 1);
    let finite: Vec<f64> = values.iter().copied().filter(|v| v.is_finite()).collect();
    let counts = uniform_histogram(&finite, None, &[], 1, &edges, 1);
    (counts, edges)
}

/// `apply_declared_ticks(axes, which, count, label_pt=...)` with the count label rule.
pub fn apply_declared_ticks(axes: &mut super::mpl::axes::AxesModel, x: bool, count: usize, size_pt: f64) {
    let lengths = vec![None, Some(4), Some(3)];
    let signature = TickSignature::Declared { count, size_pt: size_pt.to_bits(), lengths: lengths.clone(), zero_optional: true, rule: DeclaredText::Count };
    let axis = axes.axis_mut(x);
    if axis.signature.as_ref() == Some(&signature) { return; }
    axis.policy = TickPolicy::Declared { locator: Box::new(DeclaredState::new(count, DeclaredText::Count, lengths, true, size_pt, None)) };
    axis.set_label_size_all(size_pt);
    axis.offset_visible = false;
    axis.signature = Some(signature);
}
