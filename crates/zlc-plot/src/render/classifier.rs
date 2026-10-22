//! The classifier layer (`_update_classifier`): a Distribution's two
//! population curves and their sum, the threshold bar and its reading,
//! painted apart from ordinary fits. Only a Histogram (or a grid of them)
//! classifies; for every other kind the layer holds nothing.
//! `_refresh_fit_focus` styles these artists with the series emphasis
//! (`Renderer::refresh_classifier_focus`).
use super::core::{Layer, Renderer};
use super::curve::{series_identity, SeriesId};
use super::fit::FitKey;
use super::frame::RenderFrame;
use super::labels::literal_text;
use super::mpl::axes::{Child, ChildId, LineArtist, TextArtist, TextFrame};
use super::mpl::color;
use crate::configuration::selectors::SelectorKind;
use crate::configuration::spec::Surface;
use super::core::truthy;
use crate::session::{FitOverlay, FitPolyline};
use crate::style::LineToken;
use zlc_figure::ir::{HAlign, VAlign};

/// One classified (cell, group)'s artists.
#[derive(Clone, Debug)]
pub struct ClassifierArtists { pub axes: usize, pub lines: [ChildId; 3], pub threshold: ChildId, pub label: ChildId }

/// The classifier layer's retained state.
#[derive(Clone, Debug, Default)]
pub struct ClassifierTables {
    /// `_classifier_artists`, in insertion order.
    pub artists: Vec<(FitKey, ClassifierArtists)>,
    /// `_classifier_labels`: each classified (cell, group)'s reading.
    pub labels: Vec<(FitKey, String)>,
}

impl ClassifierTables {
    /// Every classifier artist in `_dynamic_artists` order (lines, bar, label).
    pub fn artists(&self) -> impl Iterator<Item = (usize, ChildId)> + '_ {
        self.artists.iter().flat_map(|(_, a)| a.lines.iter().copied().chain([a.threshold, a.label]).map(move |id| (a.axes, id)))
    }

    pub fn label(&self, key: &FitKey) -> Option<&String> { self.labels.iter().find(|(k, _)| k == key).map(|(_, l)| l) }
}

/// The classifier layer of `present`.
pub struct ClassifierLayer;

impl Layer for ClassifierLayer {
    fn update(&self, renderer: &mut Renderer, frame: &RenderFrame) {
        let mut labels: Vec<(FitKey, String)> = Vec::new();
        for (overlay, label) in frame.classifier_overlays.iter().zip(&frame.classifier_labels) {
            let key = (overlay.facet_index, series_identity(&overlay.group_key));
            match labels.iter_mut().find(|(k, _)| *k == key) { Some(entry) => entry.1 = label.clone(), None => labels.push((key, label.clone())) }
        }
        renderer.classifier.labels = labels;
        let overlays: &[FitOverlay] = if renderer.height_bars.active { &[] } else { &frame.classifier_overlays };
        renderer.update_classifier(overlays, &frame.classifier_thresholds, &frame.classifier_labels);
    }
}

impl Renderer {
    fn classifier_line(&self, token: &LineToken) -> LineArtist {
        LineArtist {
            visible: true, zorder: token.zorder.unwrap_or(2.0), alpha: Some(token.alpha), color: token.color.to_owned(),
            linewidth: token.linewidth, linestyle: token.linestyle.to_owned(), marker: token.marker.unwrap_or("None").to_owned(),
            markersize: self.rc.lines_markersize, markeredgewidth: self.rc.lines_markeredgewidth, markevery: None,
            marker_colors: None, label: String::new(), x: Vec::new(), y: Vec::new(),
        }
    }

    /// `_update_classifier(overlays, thresholds, labels)`.
    pub(crate) fn update_classifier(&mut self, overlays: &[FitOverlay], thresholds: &[Option<f64>], labels: &[String]) {
        let mut active: Vec<(FitKey, Vec<FitPolyline>, f64, String)> = Vec::new();
        if matches!(self.spec.semantic(), Surface::Histogram(_)) {
            for ((overlay, threshold), label) in overlays.iter().zip(thresholds).zip(labels) {
                let mut curves: Vec<FitPolyline> = overlay.polylines.iter().filter(|p| p.role == "component").cloned()
                    .chain(overlay.polylines.iter().filter(|p| p.role == "total").cloned()).collect();
                if !overlay.success || curves.len() != 3 { curves.clear(); }
                let Some(threshold) = threshold else { continue };
                // A grid's classifier stands on a visible cell.
                if self.is_facet() && overlay.facet_index.is_none_or(|index| index >= self.facet.visible) { continue; }
                let group = literal_text(&overlay.group_key.iter().map(|v| v.label.as_str()).collect::<Vec<_>>().join(", "));
                let content = if group.is_empty() { label.clone() } else { format!("{group}\n{label}") };
                let key = (overlay.facet_index, series_identity(&overlay.group_key));
                let entry = (key.clone(), curves, *threshold, content);
                match active.iter_mut().find(|(k, ..)| *k == key) { Some(slot) => *slot = entry, None => active.push(entry) }
            }
        }
        let stale: Vec<(FitKey, ClassifierArtists)> = self.classifier.artists.iter()
            .filter(|(key, _)| !active.iter().any(|(k, ..)| k == key)).cloned().collect();
        for (key, artists) in stale {
            self.classifier.artists.retain(|(k, _)| *k != key);
            for id in artists.lines.iter().copied().chain([artists.threshold, artists.label]) {
                self.curve.sources.remove(&(artists.axes, id));
                self.fig.axes[artists.axes].remove(id);
            }
        }
        let style = self.style.clone();
        let inset = style.render.axes_text_inset_fraction;
        let label_pt = if self.is_facet() { style.fonts.facet_fit_annotation_pt } else { style.fonts.annotation_pt };
        let threshold_width = style.artists.classifier_threshold_line.linewidth * self.facet_mark_scale();
        for (key, curves, threshold, content) in active {
            let axes = match key.0 { Some(index) if self.is_facet() => self.axes_of("facet_cell")[index], _ => self.primary_axes() };
            let artists = match self.classifier.artists.iter().find(|(k, _)| *k == key) {
                Some((_, artists)) => artists.clone(),
                None => {
                    let lines = style.artists.bimodal_fit_lines.clone().map(|token| {
                        let line = self.classifier_line(&token);
                        self.fig.axes[axes].add(Child::Line(line))
                    });
                    let threshold_line = self.classifier_line(&style.artists.classifier_threshold_line);
                    let threshold = self.fig.axes[axes].add(Child::Line(threshold_line));
                    self.plotted(axes);
                    let label = self.fig.axes[axes].add(Child::Text(TextArtist {
                        visible: true, zorder: style.artists.fit_annotation_zorder, x: 1.0 - inset, y: 1.0 - inset, text: String::new(),
                        color: style.palette.threshold.to_owned(),
                        size: label_pt, ha: HAlign::Left, va: VAlign::Top,
                        ma: None, clip_on: true, frame: TextFrame::Axes, family: None,
                    }));
                    let artists = ClassifierArtists { axes, lines, threshold, label };
                    self.classifier.artists.push((key.clone(), artists.clone()));
                    artists
                }
            };
            for (position, &line) in artists.lines.iter().enumerate() {
                if let Some(curve) = curves.get(position) {
                    let curve = curve.clone();
                    self.set_classifier_curve(axes, line, &curve);
                } else if let Some(Child::Line(l)) = self.fig.axes[axes].child_mut(line) {
                    l.x.clear();
                    l.y.clear();
                    l.visible = false;
                }
            }
            let (low, high) = self.fig.axes[axes].ylim;
            if let Some(Child::Line(l)) = self.fig.axes[axes].child_mut(artists.threshold) {
                l.x = vec![threshold, threshold];
                l.y = vec![low, high];
                // A cell's marks are drawn at the cell's scale, like its text.
                l.linewidth = threshold_width;
            }
            if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(artists.label) { text.text = content; }
        }
    }

    /// `_set_fit_line(line, polyline)` then visible, for a classifier curve.
    fn set_classifier_curve(&mut self, axes: usize, line: ChildId, polyline: &FitPolyline) {
        let (mut x, mut y) = (polyline.x.to_vec(), polyline.y.to_vec());
        if x.len() > 1 && !x.windows(2).all(|w| w[1] >= w[0]) {
            let mut order: Vec<usize> = (0..x.len()).collect();
            order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
            x = order.iter().map(|&i| polyline.x[i]).collect();
            y = order.iter().map(|&i| polyline.y[i]).collect();
        }
        self.apply_line_data(axes, line, std::sync::Arc::new(x), std::sync::Arc::new(y), false);
        if let Some(child) = self.fig.axes[axes].child_mut(line) { child.set_visible(true); }
    }

    /// `_classifier_annotation_owns_inspector()`: a classifying histogram's
    /// threshold readout names the series; no inspector beside it.
    pub(crate) fn classifier_annotation_owns_inspector(&self) -> bool {
        matches!(self.spec.semantic(), Surface::Histogram(_))
            && self.last_state.as_ref().is_some_and(|state| state.get("threshold_classifier").is_some_and(truthy))
            && self.selectors.last.states().iter().any(|state| state.kind() == SelectorKind::Threshold)
    }

    /// `_classifier_series(axis)`: the authoring group -- the locked one, else the first.
    pub(crate) fn classifier_series(&self, axes: usize) -> Option<(ChildId, SeriesId, String)> {
        if self.classifier.label(&(None, Vec::new())).is_some() { return None; }
        let records = self.curve.series_lines.get(&axes)?;
        if records.is_empty() { return None; }
        let index = match &self.curve.locked {
            Some(locked) if locked.axes == axes => self.curve.series_indices.get(&axes)
                .and_then(|v| v.iter().find(|(id, _)| *id == locked.identity).map(|(_, i)| *i)).unwrap_or(0),
            _ => 0,
        };
        Some(records[index].clone())
    }

    /// The classifier part of `_refresh_fit_focus`.
    pub(crate) fn refresh_classifier_focus(&mut self) {
        let style = self.style.clone();
        let curve_width = style.artists.curve.linewidth;
        for (key, artists) in self.classifier.artists.clone() {
            let (_, identity) = &key;
            let axes = artists.axes;
            let data = self.curve.series_indices.get(&axes).and_then(|v| v.iter().find(|(id, _)| id == identity).map(|(_, i)| *i))
                .and_then(|index| self.curve.series_lines.get(&axes).map(|records| records[index].0));
            let (alpha, width, colour) = match data.and_then(|id| self.fig.axes[axes].child(id)) {
                Some(Child::Line(line)) => (line.alpha.unwrap_or(1.0), line.linewidth / curve_width, Some(line.color.clone())),
                Some(Child::Segments(bars)) => (bars.alpha, bars.linewidth / curve_width,
                                                Some(color::tuple_spec([bars.facecolor[0], bars.facecolor[1], bars.facecolor[2], 1.0]))),
                Some(Child::Bars(bars)) => {
                    let face = bars.facecolor();
                    (bars.alpha, bars.linewidth / curve_width, Some(color::tuple_spec([face[0], face[1], face[2], 1.0])))
                }
                _ => (1.0, 1.0, None),
            };
            let colour = if identity.is_empty() { None } else { colour };
            for (&line, token) in artists.lines.iter().zip(style.artists.bimodal_fit_lines.iter()) {
                if let Some(Child::Line(l)) = self.fig.axes[axes].child_mut(line) {
                    l.color = colour.clone().unwrap_or_else(|| token.color.to_owned());
                    l.alpha = Some(token.alpha * alpha);
                    l.linewidth = token.linewidth * width;
                }
            }
            if let Some(Child::Line(l)) = self.fig.axes[axes].child_mut(artists.threshold) {
                l.alpha = Some(style.artists.classifier_threshold_line.alpha * alpha);
                if let Some(colour) = &colour { l.color = colour.clone(); }
            }
            if let (Some(colour), Some(Child::Text(text))) = (&colour, self.fig.axes[axes].child_mut(artists.label)) { text.color = colour.clone(); }
            let interactive = !self.is_facet() || self.facet.focus == key.0;
            let authoring = self.classifier_series(axes).map(|s| s.1).unwrap_or_default();
            let chosen = identity.is_empty() || authoring == *identity;
            if let Some(child) = self.fig.axes[axes].child_mut(artists.threshold) { child.set_visible(!interactive || !chosen); }
            let shows = !interactive && chosen
                && matches!(self.fig.axes[axes].child(artists.label), Some(Child::Text(t)) if !t.text.is_empty());
            if let Some(child) = self.fig.axes[axes].child_mut(artists.label) { child.set_visible(shows); }
            if shows { self.pin_readout_held(axes, artists.label, 1.0 - style.render.axes_text_inset_fraction, true); }
        }
    }
}
