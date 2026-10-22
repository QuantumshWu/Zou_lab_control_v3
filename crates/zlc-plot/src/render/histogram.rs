//! The Histogram kind (`_update_histogram`): one `_BarBuffered` collection per
//! group on its axes, coloured by the group's series slot, its counts as the
//! state asks for them (`_histogram_arrays`: density, cumulative), x framed by
//! the bin edges, y by the count limits (`_resolve_histogram_y_limits`), on a
//! linear or logarithmic y scale. The bars are painted by the bar kernel
//! (`histogram:prepared`, see `front`), the collections stay what an export
//! draws.
use std::collections::HashMap;
use std::sync::Arc;

use super::core::{real_value, relim_mode, text_value, truthy, Kind, Renderer};
use super::curve::{apply_smart_ticks, series_identity, series_slot};
use super::labels::{quantity_label, state_label};
use super::limits::{relim_retains, select_display_limits};
use super::mpl::axes::{BarArtist, Child, ChildId};
use crate::configuration::{DisplayState, PlotSpec, Value};
use crate::kinds::RelimMode;
use crate::projection::output::{HistogramData, Payload};

/// The Histogram kind.
pub struct HistogramKind;

impl Kind for HistogramKind {
    fn render(&self, renderer: &mut Renderer, payload: &Arc<Payload>, state: &DisplayState, axes: usize, key: &str) {
        renderer.update_histogram(axes, histogram_payload(payload), state, key, None, None, true);
    }
    fn key(&self) -> &'static str { "histogram" }
}

/// The histogram a Histogram surface draws.
pub(crate) fn histogram_payload(payload: &Payload) -> &HistogramData {
    match payload { Payload::Histogram(histogram) => histogram, _ => panic!("histogram payload must expose edges and counts") }
}

/// One surface's bin edges (display unit) and counts per group, as drawn.
#[derive(Clone, Debug)]
pub struct HistogramArrays { pub edges: Arc<Vec<f64>>, pub counts: Vec<Arc<Vec<f64>>> }

/// What one count means: an automatic y range retained under other semantics
/// re-fits instead of riding the hysteresis (`count_semantics`).
type CountSemantics = (RelimMode, bool, bool, bool, i128);

/// The histogram renderer's tables.
#[derive(Clone, Debug, Default)]
pub struct HistogramTables {
    /// `_artists[key]`: each surface key's collections (axes, id), keys in insertion order.
    pub collections: Vec<(String, Vec<(usize, ChildId)>)>,
    /// `histogram:prepared`: the surface keys whose bars the kernel paints;
    /// None once the compose handed the picture back to the collections.
    pub prepared: Option<Vec<String>>,
    /// `{key}:histogram_limits` and `{key}:histogram_mode`.
    pub limits: HashMap<String, ((f64, f64), CountSemantics)>,
}

impl HistogramTables {
    /// Every collection, `_artists` order.
    pub fn bars(&self) -> impl Iterator<Item = (usize, ChildId)> + '_ {
        self.collections.iter().flat_map(|(_, ids)| ids.iter().copied())
    }

    /// The collections of one surface key.
    pub fn of(&self, key: &str) -> Option<&[(usize, ChildId)]> {
        self.collections.iter().find(|(k, _)| k == key).map(|(_, ids)| ids.as_slice())
    }
}

/// Python `bool(state[name])`.
fn flag(state: &DisplayState, name: &str) -> bool { state.get(name).is_some_and(truthy) }

fn bin_count(state: &DisplayState) -> i128 {
    match state.get("bin_count") { Some(Value::Int(v)) => *v, Some(Value::Float(v)) => *v as i128, other => panic!("bin_count: {other:?}") }
}

/// `_histogram_arrays(payload, state)`: the edges and the counts, cumulated
/// and/or normalized to a density as the state asks.
pub fn histogram_arrays(histogram: &HistogramData, state: &DisplayState) -> HistogramArrays {
    let edges = histogram.edges.display.to_f64();
    let shape = histogram.counts.shape();
    let (groups, bins) = (shape[0], shape[1]);
    let flat = histogram.counts.to_f64();
    let mut counts: Vec<Vec<f64>> = (0..groups).map(|g| flat[g * bins..(g + 1) * bins].to_vec()).collect();
    let density = flag(state, "density");
    if flag(state, "cumulative") {
        for row in &mut counts {
            let mut running = 0.0;
            for value in row.iter_mut() { running += *value; *value = running; }
        }
        if density && groups * bins > 0 {
            for row in &mut counts {
                let total = row[bins - 1];
                for value in row.iter_mut() { *value = if total > 0.0 { *value / total } else { 0.0 }; }
            }
        }
    } else if density {
        let widths: Vec<f64> = edges.windows(2).map(|w| w[1] - w[0]).collect();
        for row in &mut counts {
            let total = crate::numeric::sum_f64(row);
            for (value, width) in row.iter_mut().zip(&widths) {
                *value = if total > 0.0 { *value / (total * width) } else { 0.0 };
            }
        }
    }
    HistogramArrays { edges: Arc::new(edges), counts: counts.into_iter().map(Arc::new).collect() }
}

/// `_effective_labels` of a Histogram: the value axis' label and the count axis'.
pub fn histogram_labels(spec: &PlotSpec, histogram: &HistogramData, state: &DisplayState) -> (String, String) {
    // The semantic spec's labels: a grid's cell labels its axes.
    let labels = spec.semantic().labels();
    let text = |name: &str| text_value(state.get(name));
    let mut explicit_x = state_label(text("x_label").as_deref(), None);
    if explicit_x.is_none() {
        let fallback = labels.x.as_deref().filter(|x| !x.is_empty()).or(labels.value.as_deref());
        explicit_x = state_label(text("value_label").as_deref(), fallback);
    }
    let edges = &histogram.edges;
    let x_label = quantity_label(Some(&edges.label), Some(&edges.display_unit.symbol), "value", explicit_x.as_deref());
    let y_label = state_label(text("y_label").as_deref(), labels.y.as_deref().filter(|y| !y.is_empty()))
        .unwrap_or_else(|| if flag(state, "density") { "density".into() } else { "Shots".into() });
    (x_label, y_label)
}

impl Renderer {
    /// `_update_histogram(axes, payload, state, key, arrays=..., limits=..., paint_labels=...)`.
    #[allow(clippy::too_many_arguments)]
    pub fn update_histogram(&mut self, axes: usize, histogram: &HistogramData, state: &DisplayState, key: &str,
                            arrays: Option<HistogramArrays>, limits: Option<((f64, f64), (f64, f64))>, paint_labels: bool) {
        let HistogramArrays { edges, counts } = arrays.unwrap_or_else(|| histogram_arrays(histogram, state));
        let position = match self.histogram.collections.iter().position(|(k, _)| k == key) {
            Some(position) => position,
            None => { self.histogram.collections.push((key.to_owned(), Vec::new())); self.histogram.collections.len() - 1 }
        };
        while self.histogram.collections[position].1.len() > counts.len() {
            let (owner, id) = self.histogram.collections[position].1.pop().unwrap();
            self.fig.axes[owner].remove(id);
        }
        // `add_collection(autolim=False)`: no data limits, no autoscale.
        while self.histogram.collections[position].1.len() < counts.len() {
            let bars = BarArtist {
                visible: true, zorder: 1.0, alpha: self.style.artists.histogram_fill_alpha, color: self.style.palette.line_single.to_owned(),
                linewidth: self.rc.patch_linewidth, label: String::new(), edges: Arc::new(Vec::new()), counts: Arc::new(Vec::new()), kernel_painted: false,
            };
            let id = self.fig.axes[axes].add(Child::Bars(bars));
            self.histogram.collections[position].1.push((axes, id));
        }
        let labels = histogram.labels();
        assert!(histogram.group_keys.len() == counts.len() && labels.len() == counts.len(), "histogram groups differ from their counts");
        let cycle = self.style.palette.line_cycle();
        let mut records = Vec::new();
        for (index, &(owner, id)) in self.histogram.collections[position].1.clone().iter().enumerate() {
            let group_key = &histogram.group_keys[index];
            let identity = series_identity(group_key);
            let colour = cycle[series_slot(group_key, cycle.len())];
            if let Some(Child::Bars(bars)) = self.fig.axes[owner].child_mut(id) {
                bars.color = colour.to_owned();
                bars.edges = edges.clone();
                bars.counts = counts[index].clone();
                bars.label = labels[index].clone();
            }
            records.push((id, identity, labels[index].clone()));
        }
        self.curve.series_indices.insert(axes, records.iter().enumerate().map(|(i, (_, id, _))| (id.clone(), i)).collect());
        self.curve.series_lines.insert(axes, records);
        self.apply_series_focus(Some(axes));
        // The bars are painted by the kernel, from the collections' edges and counts.
        let prepared = self.histogram.prepared.get_or_insert_with(Vec::new);
        if !prepared.iter().any(|k| k == key) { prepared.push(key.to_owned()); }
        let (selected_x, selected_y) = match limits {
            Some((x, y)) => (Some(x), y),
            None => {
                let x = (edges.len() >= 2).then(|| (edges[0], edges[edges.len() - 1]));
                (x, self.resolve_histogram_y_limits(key, &counts, state))
            }
        };
        let log = flag(state, "log_y");
        assert!(!(log && selected_y.0 <= 0.0), "resolved logarithmic limits are not positive");
        if self.fig.axes[axes].yaxis.log != log {
            if log {
                // A linear axis whose lower bound is not positive cannot be
                // made logarithmic: a valid positive span first.
                let (a, b) = self.fig.axes[axes].ylim;
                let old_low = if b < a { b } else { a };
                if old_low <= 0.0 {
                    let high = if selected_y.1 > selected_y.0 * 10.0 { selected_y.1 } else { selected_y.0 * 10.0 };
                    self.fig.axes[axes].set_ylim(selected_y.0, high);
                }
            }
            self.fig.axes[axes].set_scale(false, log);
            self.chrome_dirty.insert(axes);
        }
        if let Some(x) = selected_x { self.set_xlim(axes, x); }
        self.set_ylim(axes, selected_y);
        if paint_labels {
            let (x_label, y_label) = histogram_labels(&self.spec, histogram, state);
            self.set_axis_labels(axes, &x_label, &y_label);
            let tick_pt = self.style.fonts.tick_pt;
            apply_smart_ticks(&mut self.fig.axes[axes], true, tick_pt);
            apply_smart_ticks(&mut self.fig.axes[axes], false, tick_pt);
        }
    }

    /// `_resolve_histogram_y_limits(key, counts, state)`: the count axis'
    /// automatic range expands when the data breaches it and shrinks when
    /// the data falls well inside it, re-fit whenever a count means
    /// something else.
    pub fn resolve_histogram_y_limits(&mut self, key: &str, counts: &[Arc<Vec<f64>>], state: &DisplayState) -> (f64, f64) {
        let mode = relim_mode(state);
        let logarithmic = flag(state, "log_y");
        let semantics: CountSemantics = (mode, logarithmic, flag(state, "density"), flag(state, "cumulative"), bin_count(state));
        let shrink_ratio = self.style.render.distribution_count_shrink_ratio;
        let state_key = format!("{key}:histogram_limits");
        let previous = self.histogram.limits.get(&state_key).copied();
        let automatic = if mode == RelimMode::Fixed { None } else {
            let finite: Vec<f64> = counts.iter().flat_map(|row| row.iter().copied()).filter(|v| v.is_finite()).collect();
            let positive: Vec<f64> = finite.iter().copied().filter(|v| *v > 0.0).collect();
            let min = |values: &[f64]| values.iter().copied().fold(f64::INFINITY, f64::min);
            let max = |values: &[f64]| values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let py_max = |a: f64, b: f64| if b > a { b } else { a };
            let peak = if finite.is_empty() { 0.0 } else { max(&finite) };
            let target = if logarithmic {
                if positive.is_empty() { (0.8, 1.2) } else { (py_max(min(&positive) * 0.8, f64::MIN_POSITIVE), py_max(max(&positive) * 1.2, 1.0)) }
            } else {
                (0.0, if peak > 0.0 { peak * 1.08 } else { 1.0 })
            };
            let force = previous.is_none_or(|(_, held)| held != semantics) || !relim_retains(mode, false);
            Some(if force { target } else {
                let ((prior_low, prior_high), _) = previous.unwrap();
                let (expand, expanded) = if logarithmic {
                    let breach_low = if positive.is_empty() { prior_low } else { min(&positive) };
                    let breach_high = if positive.is_empty() { prior_high } else { max(&positive) };
                    let low = if prior_low < target.0 { prior_low } else { target.0 };
                    (breach_low < prior_low || breach_high > prior_high, (low, py_max(breach_high * 1.5, prior_high)))
                } else {
                    (peak > prior_high, (0.0, peak * 1.25))
                };
                let shrink = target.1 < shrink_ratio * prior_high || (logarithmic && target.0 > prior_low / shrink_ratio);
                if expand { expanded } else if shrink { target } else { (prior_low, prior_high) }
            })
        };
        let authored = (real_value(state.get("y_min")), real_value(state.get("y_max")));
        let selected = select_display_limits(mode, automatic, authored, false).unwrap_or_else(|error| panic!("{error}"));
        assert!(!(logarithmic && selected.0 <= 0.0), "resolved logarithmic limits are not positive");
        self.histogram.limits.insert(state_key, (selected, semantics));
        selected
    }
}
