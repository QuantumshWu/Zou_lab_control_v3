//! The fit layer: `_set_fit_mode` (the source series as a scatter while a
//! fit is shown), `_update_fit` (one topology of artists per fitted (cell,
//! group): lines, an ellipse glyph, the annotation or the diagnostic) and
//! `_refresh_fit_focus` (fit styling that follows the series emphasis, and
//! the one annotation each axes shows).
use std::sync::Arc;

use super::core::{Layer, Renderer};
use super::curve::{series_identity, series_slot, SeriesId};
use super::frame::RenderFrame;
use super::labels::literal_text;
use super::mpl::axes::{Child, ChildId, EllipseArtist, LineArtist, ScatterArtist, TextArtist, TextFrame};
use super::mpl::color;
use crate::configuration::spec::Surface;
use crate::session::{FitOverlay, FitPolyline};
use crate::style::LineToken;
use crate::ticks::compact_number;
use zlc_figure::ir::{HAlign, VAlign};

const DIAGNOSTIC_SINGLE_MAX_CHARS: usize = 72;
const DIAGNOSTIC_FACET_MAX_CHARS: usize = 24;
/// `_FIT_CENTER_EDGE_PT`: the ellipse centre dot's edge, in its face colour.
pub(crate) const CENTER_EDGE_PT: f64 = 1.0;

/// `_FitAnnotationDetail`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Detail { None, Headline, Full }

/// One topology's named artists (`_fit_slots`).
#[derive(Clone, Debug)]
pub struct Slots {
    pub detail: Detail,
    pub diagnostic: Option<ChildId>,
    pub center: Option<ChildId>,
    pub ring: Option<ChildId>,
    /// A component family's lines.
    pub lines: Vec<ChildId>,
    /// A curve or histogram family's one line.
    pub line: Option<ChildId>,
    pub annotation: Option<ChildId>,
    pub annotation_value: Option<ChildId>,
}

impl Slots {
    fn new(detail: Detail) -> Slots {
        Slots { detail, diagnostic: None, center: None, ring: None, lines: Vec::new(), line: None, annotation: None, annotation_value: None }
    }

    /// The fit lines, in polyline order.
    fn fit_lines(&self) -> Vec<ChildId> { if self.lines.is_empty() { self.line.into_iter().collect() } else { self.lines.clone() } }

    fn texts(&self) -> [Option<ChildId>; 3] { [self.annotation, self.annotation_value, self.diagnostic] }
}

/// (facet index, series identity): one fitted (cell, group).
pub type FitKey = (Option<usize>, SeriesId);

#[derive(Clone, Debug)]
pub struct Topology { pub axes: usize, pub family: String, pub model: Option<String>, pub slots: Slots, pub artists: Vec<ChildId> }

/// The fit layer's retained state.
#[derive(Clone, Debug, Default)]
pub struct FitTables {
    /// `_fit_topologies`, in insertion order.
    pub topologies: Vec<(FitKey, Topology)>,
    /// `_fit_artists`, in creation order: (axes, artist).
    pub artists: Vec<(usize, ChildId)>,
    /// `_fit_source_scatter`.
    pub source_scatter: Option<(usize, ChildId)>,
    /// `_fit_hidden_source_lines`: (axes, line, was visible).
    pub hidden_source_lines: Vec<(usize, ChildId, bool)>,
    /// `_last_fit_overlays`: what this frame paints.
    pub last_overlays: Vec<FitOverlay>,
}

impl FitTables {
    fn topology(&self, key: &FitKey) -> Option<&Topology> { self.topologies.iter().find(|(k, _)| k == key).map(|(_, t)| t) }
}

/// The fit layer of `present` (`_set_fit_mode` + `_update_fit`).
pub struct FitLayer;

impl Layer for FitLayer {
    fn update(&self, renderer: &mut Renderer, frame: &RenderFrame) {
        let overview = renderer.fit_overview();
        let overlays: Vec<FitOverlay> = if overview { frame.fit_overlays.to_vec() } else { renderer.focused_fit_overlays(&frame.fit_overlays) };
        renderer.fit.last_overlays = overlays.clone();
        // A height-bar scene shows no 2D fit.
        let scene_3d = renderer.height_bars.active;
        renderer.set_fit_mode(!overlays.is_empty() && !overview && !scene_3d);
        renderer.update_fit(if scene_3d { &[] } else { &overlays }, frame.fit_model_id.as_deref(), overview);
    }
}

/// `_fit_family_for(semantic, overlay)`.
fn family_for(semantic: Surface<'_>, overlay: &FitOverlay) -> String {
    if !overlay.success { return "failure".into(); }
    if overlay.ellipse_glyph.is_some() { return "ellipse".into(); }
    if overlay.polylines.iter().any(|p| p.role == "component") { return format!("components:{}", overlay.polylines.len()); }
    if !overlay.polylines.is_empty() {
        return if matches!(semantic, Surface::Histogram(_)) { "histogram".into() } else { "curve".into() };
    }
    "annotation".into()
}

/// `_truncate_fit_diagnostic(message, maximum)`.
fn truncate_diagnostic(message: &str, maximum: usize) -> String {
    let text = message.trim();
    let text = if text.is_empty() { "fit failed" } else { text };
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= maximum { return text.to_owned(); }
    format!("{}...", chars[..maximum - 3].iter().collect::<String>())
}

/// `_fit_parameter_parts(parameter)`: the catalogue's symbol and the live value.
pub fn parameter_parts(parameter: &crate::fit::FitParameterDisplay) -> (String, String) {
    let uncertainty = match parameter.standard_error { None => "n/a".to_owned(), Some(error) => compact_number(error, None) };
    let unit = if parameter.unit.is_empty() { String::new() } else { format!(" {}", literal_text(&parameter.unit)) };
    (super::fonts::drawable_text(&parameter.label), format!(" = {} ± {uncertainty}{unit}", compact_number(parameter.value, None)))
}

/// `_fit_annotation_text(overlay)`.
fn annotation_text(overlay: &FitOverlay) -> String {
    let mut lines = Vec::new();
    if !overlay.formula.is_empty() { lines.push(super::fonts::drawable_text(&overlay.formula)); }
    for parameter in &overlay.parameter_display {
        let (symbol, value) = parameter_parts(parameter);
        lines.push(symbol + &value);
    }
    if !overlay.evidence.is_empty() { lines.push(super::fonts::drawable_text(&overlay.evidence)); }
    lines.join("\n")
}

fn group_label(overlay: &FitOverlay) -> String {
    literal_text(&overlay.group_key.iter().map(|v| v.label.as_str()).collect::<Vec<_>>().join(", "))
}

impl Renderer {
    /// A FacetGrid overview (no focused cell): each fitted cell shows its
    /// headline instead of the full annotation.
    fn fit_overview(&self) -> bool { self.facet_overview() }

    /// `_fit_target(overlay)`: the axes a fit is drawn on.
    fn fit_target(&self, overlay: &FitOverlay) -> usize {
        if self.is_facet() { self.facet_cell_named(overlay.facet_index, "fit") } else { self.primary_axes() }
    }

    /// `facet_index_for_axes(axes)`.
    fn fit_facet_index(&self, axes: usize) -> Option<usize> { self.facet_index_for_axes(axes) }

    /// `_fit_polyline_token(semantic, polyline)`.
    fn fit_polyline_token(&self, polyline: &FitPolyline) -> LineToken {
        let artists = &self.style.artists;
        if polyline.role == "component" {
            let tokens = &artists.bimodal_fit_lines[..2];
            return tokens[polyline.component_index % tokens.len()].clone();
        }
        if polyline.role == "total" || matches!(self.spec.semantic(), Surface::Histogram(_)) { return artists.bimodal_fit_lines[2].clone(); }
        artists.curve_fit_line.clone()
    }

    fn remove_fit_artists(&mut self, artists: &[ChildId], axes: usize) {
        for &id in artists {
            self.curve.sources.remove(&(axes, id));
            self.fig.axes[axes].remove(id);
        }
        self.fit.artists.retain(|entry| !(entry.0 == axes && artists.contains(&entry.1)));
    }

    /// `_restore_fit_source_lines`.
    fn restore_fit_source_lines(&mut self) {
        for (axes, line, was_visible) in std::mem::take(&mut self.fit.hidden_source_lines) {
            if let Some(child) = self.fig.axes[axes].child_mut(line) { child.set_visible(was_visible); }
        }
        if let Some((axes, id)) = self.fit.source_scatter {
            if let Some(Child::Scatter(scatter)) = self.fig.axes[axes].child_mut(id) {
                scatter.offsets.clear();
                scatter.visible = false;
            }
        }
    }

    /// `_show_fit_source_scatter(axis, source_lines)`.
    fn show_fit_source_scatter(&mut self, axes: usize, source_lines: &[ChildId]) {
        let hidden: Vec<(ChildId, bool)> = self.fit.hidden_source_lines.iter()
            .filter(|(a, line, _)| *a == axes && source_lines.contains(line)).map(|&(_, line, was)| (line, was)).collect();
        let visible: Vec<ChildId> = source_lines.iter().copied()
            .filter(|&line| self.fig.axes[axes].child(line).is_some_and(Child::visible)).collect();
        let active: Vec<ChildId> = if !visible.is_empty() {
            self.fit.hidden_source_lines = visible.iter().map(|&line| (axes, line, true)).collect();
            visible
        } else {
            hidden.iter().filter(|(_, was)| *was).map(|(line, _)| *line).collect()
        };
        let cap = self.style.render.fit_source_scatter_max_points;
        let mut pairs: Vec<(Arc<Vec<f64>>, Arc<Vec<f64>>)> = Vec::new();
        let mut count = 0usize;
        for line in active {
            let (x, y) = match self.curve.sources.get(&(axes, line)) {
                Some(source) => (source.x.clone(), source.y.clone()),
                None => {
                    let l = self.fig.axes[axes].line(line);
                    (Arc::new(l.x.clone()), Arc::new(l.y.clone()))
                }
            };
            if x.len() != y.len() { continue; }
            count += x.iter().zip(y.iter()).filter(|(a, b)| a.is_finite() && b.is_finite()).count();
            pairs.push((x, y));
            if count >= cap {
                self.restore_fit_source_lines();
                return;
            }
        }
        let points: Vec<[f64; 2]> = pairs.iter()
            .flat_map(|(x, y)| x.iter().zip(y.iter()).filter(|(a, b)| a.is_finite() && b.is_finite()).map(|(&a, &b)| [a, b]).collect::<Vec<_>>())
            .collect();
        if points.is_empty() {
            self.restore_fit_source_lines();
            return;
        }
        if let Some((owner, id)) = self.fit.source_scatter {
            if owner != axes {
                self.fig.axes[owner].remove(id);
                self.fit.source_scatter = None;
            }
        }
        let id = match self.fit.source_scatter {
            Some((_, id)) => id,
            None => {
                let artists = &self.style.artists;
                let id = self.fig.axes[axes].add(Child::Scatter(ScatterArtist {
                    visible: true, zorder: artists.fit_source_scatter_zorder, alpha: None, marker: "o", offsets: Vec::new(),
                    sizes: vec![artists.fit_source_scatter_area_pt2], facecolors: vec![color::rgba(self.style.palette.data_scatter)],
                    edgecolors: Vec::new(), linewidths: vec![self.rc.lines_linewidth],
                }));
                self.plotted(axes);
                self.fit.source_scatter = Some((axes, id));
                id
            }
        };
        if let Some(Child::Scatter(scatter)) = self.fig.axes[axes].child_mut(id) {
            scatter.offsets = points;
            scatter.visible = true;
        }
        for (owner, line, _) in self.fit.hidden_source_lines.clone() {
            if let Some(child) = self.fig.axes[owner].child_mut(line) { child.set_visible(false); }
        }
    }

    /// `_set_fit_mode(active)`.
    pub(crate) fn set_fit_mode(&mut self, active: bool) {
        if !active {
            self.restore_fit_source_lines();
            return;
        }
        self.materialize_prepared_curve();
        if !matches!(self.spec.semantic(), Surface::Curve(_) | Surface::Rolling(_)) {
            self.restore_fit_source_lines();
            return;
        }
        // A grid shows its selected cell's source.
        let axes = if self.is_facet() {
            match self.facet.focused.and_then(|index| self.axes_of("facet_cell").get(index).copied()) {
                Some(axes) => axes,
                None => {
                    self.restore_fit_source_lines();
                    return;
                }
            }
        } else {
            self.primary_axes()
        };
        // `_artists["curve"]` / `["rolling:history"]`: the first line on the axes.
        let first: Vec<ChildId> = self.curve.lines.iter()
            .filter(|(key, _)| key.as_str() == "curve" || key.as_str() == "rolling:history" || key.starts_with("facet:"))
            .flat_map(|(_, (owner, lines))| lines.iter().filter(move |_| *owner == axes).copied().collect::<Vec<_>>())
            .take(1).collect();
        self.show_fit_source_scatter(axes, &first);
    }

    fn fit_text(&self, size: f64, color: &str, ha_left: bool, visible: bool) -> TextArtist {
        let inset = self.style.render.axes_text_inset_fraction;
        TextArtist {
            visible, zorder: self.style.artists.fit_annotation_zorder, x: inset, y: 1.0 - inset, text: String::new(),
            color: color.to_owned(), size, ha: if ha_left { HAlign::Left } else { HAlign::Left }, va: VAlign::Top, ma: None,
            clip_on: true, frame: TextFrame::Axes, family: None,
        }
    }

    fn fit_line_artist(&self, token: &LineToken) -> LineArtist {
        LineArtist {
            visible: true, zorder: token.zorder.unwrap_or(2.0), alpha: Some(token.alpha), color: token.color.to_owned(),
            linewidth: token.linewidth, linestyle: token.linestyle.to_owned(), marker: token.marker.unwrap_or("None").to_owned(),
            markersize: self.rc.lines_markersize, markeredgewidth: self.rc.lines_markeredgewidth, markevery: None,
            marker_colors: None, label: String::new(), x: Vec::new(), y: Vec::new(),
        }
    }

    /// `_build_fit_topology(axis, family, semantic, overlay, annotation=detail)`.
    fn build_fit_topology(&mut self, axes: usize, family: &str, overlay: &FitOverlay, detail: Detail) -> Slots {
        let mut slots = Slots::new(detail);
        let fonts = &self.style.fonts;
        let size = if detail == Detail::Headline { fonts.facet_fit_annotation_pt } else { fonts.fit_annotation_pt };
        // `axis.plot` / `axis.scatter` request an autoscale; texts and `add_artist` do not.
        let add = |r: &mut Renderer, child: Child| {
            let plotted = matches!(child, Child::Line(_) | Child::Scatter(_));
            let id = r.fig.axes[axes].add(child);
            if plotted { r.plotted(axes); }
            r.fit.artists.push((axes, id));
            id
        };
        if family == "failure" {
            if detail == Detail::None { return slots; }
            let text = self.fit_text(size, self.style.artists.fit_failure_color, true, true);
            slots.diagnostic = Some(add(self, Child::Text(text)));
            return slots;
        }
        if family == "ellipse" {
            let artists = self.style.artists.clone();
            let mut center = self.fit_line_artist(&LineToken::new(artists.fit_ellipse_color, self.rc.lines_linewidth, 1.0));
            center.color = self.fig.axes[axes].next_line_color(&self.style.palette.line_cycle());
            center.alpha = None;
            center.linestyle = "None".into();
            center.marker = "o".into();
            center.markersize = artists.fit_ellipse_center_area_pt2.sqrt();
            center.markeredgewidth = CENTER_EDGE_PT;
            center.marker_colors = Some((artists.fit_ellipse_color.to_owned(), artists.fit_ellipse_color.to_owned()));
            center.zorder = artists.fit_ellipse_zorder;
            center.visible = false;
            let ring = EllipseArtist {
                visible: false, zorder: artists.fit_ellipse_zorder, alpha: Some(artists.point_occupied.alpha), center: [0.0, 0.0],
                width: 0.0, height: 0.0, edgecolor: artists.point_occupied.color.to_owned(), linewidth: artists.point_occupied.linewidth,
            };
            slots.center = Some(add(self, Child::Line(center)));
            slots.ring = Some(add(self, Child::Ellipse(ring)));
        } else if family.starts_with("components:") {
            for polyline in &overlay.polylines {
                let token = self.fit_polyline_token(polyline);
                let line = self.fit_line_artist(&token);
                let id = add(self, Child::Line(line));
                slots.lines.push(id);
            }
        } else if family == "histogram" || family == "curve" {
            let token = self.fit_polyline_token(&overlay.polylines[0]);
            let line = self.fit_line_artist(&token);
            slots.line = Some(add(self, Child::Line(line)));
        }
        if detail == Detail::None { return slots; }
        let text = self.fit_text(size, self.style.palette.fit_text, true, false);
        slots.annotation = Some(add(self, Child::Text(text.clone())));
        if detail == Detail::Headline { slots.annotation_value = Some(add(self, Child::Text(text))); }
        slots
    }

    /// `_set_fit_line(line, polyline)`.
    fn set_fit_line(&mut self, axes: usize, line: ChildId, polyline: &FitPolyline) {
        let (mut x, mut y) = (polyline.x.to_vec(), polyline.y.to_vec());
        if x.len() > 1 && !x.windows(2).all(|w| w[1] >= w[0]) {
            let mut order: Vec<usize> = (0..x.len()).collect();
            order.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap_or(std::cmp::Ordering::Equal));
            x = order.iter().map(|&i| polyline.x[i]).collect();
            y = order.iter().map(|&i| polyline.y[i]).collect();
        }
        self.apply_line_data(axes, line, Arc::new(x), Arc::new(y), false);
        if let Some(child) = self.fig.axes[axes].child_mut(line) { child.set_visible(true); }
    }

    /// `_update_fit_primitives(family, overlay)`.
    fn update_fit_primitives(&mut self, axes: usize, family: &str, slots: &Slots, overlay: &FitOverlay) {
        if family == "ellipse" {
            let glyph = overlay.ellipse_glyph.expect("ellipse fit family requires one ellipse glyph");
            if let Some(Child::Line(center)) = slots.center.and_then(|id| self.fig.axes[axes].child_mut(id)) {
                center.x = vec![glyph.center_x];
                center.y = vec![glyph.center_y];
                center.visible = true;
            }
            if let Some(Child::Ellipse(ring)) = slots.ring.and_then(|id| self.fig.axes[axes].child_mut(id)) {
                ring.center = [glyph.center_x, glyph.center_y];
                ring.width = 2.0 * glyph.radius_x;
                ring.height = 2.0 * glyph.radius_y;
                ring.visible = true;
            }
            return;
        }
        if family.starts_with("components:") {
            for (line, polyline) in slots.lines.iter().zip(&overlay.polylines) { self.set_fit_line(axes, *line, polyline); }
            return;
        }
        if family == "histogram" || family == "curve" {
            if let Some(line) = slots.line { self.set_fit_line(axes, line, &overlay.polylines[0]); }
        }
    }

    /// `_update_fit(overlays, overview=..., model_id=...)`.
    pub(crate) fn update_fit(&mut self, overlays: &[FitOverlay], model_id: Option<&str>, overview: bool) {
        let detail = if overview { Detail::Headline } else { Detail::Full };
        let mut targets: Vec<(FitKey, &FitOverlay, usize)> = Vec::new();
        for overlay in overlays {
            let axes = self.fit_target(overlay);
            let key = (self.fit_facet_index(axes), series_identity(&overlay.group_key));
            match targets.iter_mut().find(|(k, _, _)| *k == key) {
                Some(entry) => { entry.1 = overlay; entry.2 = axes; }
                None => targets.push((key, overlay, axes)),
            }
        }
        let stale: Vec<(FitKey, Topology)> = self.fit.topologies.iter().filter(|(k, _)| !targets.iter().any(|(t, _, _)| t == k)).cloned().collect();
        for (key, topology) in stale {
            self.fit.topologies.retain(|(k, _)| *k != key);
            self.remove_fit_artists(&topology.artists, topology.axes);
        }
        for (key, overlay, axes) in targets {
            let family = family_for(self.spec.semantic(), overlay);
            let current = self.fit.topology(&key).cloned();
            let rebuild = match &current {
                None => true,
                Some(t) => t.axes != axes || t.family != family || t.model.as_deref() != model_id || t.slots.detail != detail,
            };
            let topology = if rebuild {
                if let Some(old) = &current { self.remove_fit_artists(&old.artists, old.axes); }
                let first = self.fit.artists.len();
                let slots = self.build_fit_topology(axes, &family, overlay, detail);
                let artists = self.fit.artists[first..].iter().map(|&(_, id)| id).collect();
                let topology = Topology { axes, family: family.clone(), model: model_id.map(str::to_owned), slots, artists };
                match self.fit.topologies.iter_mut().find(|(k, _)| *k == key) {
                    Some(entry) => entry.1 = topology.clone(),
                    None => self.fit.topologies.push((key.clone(), topology.clone())),
                }
                topology
            } else {
                current.unwrap()
            };
            if family != "failure" { self.update_fit_primitives(axes, &family, &topology.slots, overlay); }
        }
        self.refresh_fit_focus();
        self.sync_fit_lines();
        // `facet:fit_native`: an overview's fits the kernels paint.
        let native = overview && matches!(self.spec.semantic(), Surface::Curve(_) | Surface::Image(_));
        self.facet.fit_native = native.then(|| overlays.to_vec());
    }

    /// The fit Line2Ds the kernels may stroke (`_native_curve_lines`' fit), in `_fit_artists` order.
    fn sync_fit_lines(&mut self) {
        self.fit_lines = self.fit.artists.iter().copied()
            .filter(|&(axes, id)| matches!(self.fig.axes[axes].child(id), Some(Child::Line(_)))).collect();
    }

    /// `_update_fit_annotation(overlay)`.
    fn update_fit_annotation(&mut self, axes: usize, slots: &Slots, overlay: &FitOverlay) {
        let mut content = annotation_text(overlay);
        if !overlay.group_key.is_empty() { content = format!("{}\n{content}", group_label(overlay)); }
        if let Some(Child::Text(text)) = slots.annotation.and_then(|id| self.fig.axes[axes].child_mut(id)) {
            text.visible = !content.is_empty();
            text.text = content;
        }
    }

    /// `_update_fit_headline_annotation(overlay, parameter_name)`: a cell's
    /// one parameter as two texts, the symbol and the live value after it.
    fn update_fit_headline_annotation(&mut self, axes: usize, slots: &Slots, overlay: &FitOverlay, parameter_name: Option<&str>) {
        let (Some(symbol_id), Some(value_id)) = (slots.annotation, slots.annotation_value) else { return };
        let parameter = parameter_name.and_then(|name| overlay.parameter_display.iter().find(|p| p.name == name))
            .or(overlay.headline_parameter.as_ref());
        let Some(parameter) = parameter else {
            for id in [symbol_id, value_id] { if let Some(child) = self.fig.axes[axes].child_mut(id) { child.set_visible(false); } }
            return;
        };
        let (mut symbol, value) = parameter_parts(parameter);
        if !overlay.group_key.is_empty() { symbol = format!("{}: {symbol}", group_label(overlay)); }
        let changed = !matches!(self.fig.axes[axes].child(symbol_id), Some(Child::Text(t)) if t.text == symbol);
        if changed {
            if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(symbol_id) { text.text = symbol.clone(); }
            let offset = self.text_advance_inches(axes, symbol_id);
            if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(value_id) { text.frame = TextFrame::AxesOffset(offset, 0.0); }
        }
        if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(value_id) { text.text = value; }
        for id in [symbol_id, value_id] { if let Some(child) = self.fig.axes[axes].child_mut(id) { child.set_visible(true); } }
    }

    /// `_transform_after(text)`'s offset: the text's width
    /// (`get_text_width_height_descent`), as `offset_copy(..., x=width pt)` holds it in inches.
    fn text_advance_inches(&self, axes: usize, id: ChildId) -> f64 {
        let Some(Child::Text(text)) = self.fig.axes[axes].child(id) else { return 0.0 };
        let _span = super::profile::span("fit.headline.advance");
        // Its content in its font at the figure's dpi: an empty figure carries
        // the dpi and the rc (a grid's whole IR, per headline, was the cost).
        let figure = super::mpl::figure::FigureModel::new(self.fig.size_inches, self.fig.dpi).to_ir(&self.rc);
        let font = match &text.family {
            Some(family) => super::mpl::artist::font_of(&self.rc, text.size, family),
            None => super::mpl::artist::font(&self.rc, text.size),
        };
        let mut fonts = self.fonts.lock().unwrap();
        let mut null = zlc_figure::query::NullBackend;
        let mut math = super::fonts::Math;
        let mut scene = zlc_figure::query::Scene::new(&figure, &mut null, &mut *fonts, &mut math).expect("scene");
        let ismath = zlc_figure::text::is_math_text(&text.text);
        let (width, _, _) = zlc_figure::text::text_width_height_descent(&mut scene.ctx, &text.text, &font, ismath)
            .expect("text width");
        (width * 72.0 / self.fig.dpi) / 72.0
    }

    /// `_refresh_fit_focus`.
    pub(crate) fn refresh_fit_focus(&mut self) {
        if self.fit.topologies.is_empty() && self.classifier.artists.is_empty() {
            self.sync_roi_text_visibility();
            return;
        }
        let mut selected: Vec<(usize, SeriesId)> = self.curve.series_lines.iter()
            .filter_map(|(&axes, records)| records.first().map(|r| (axes, r.1.clone()))).collect();
        let set = |selected: &mut Vec<(usize, SeriesId)>, axes: usize, identity: SeriesId, replace: bool| {
            match selected.iter_mut().find(|(a, _)| *a == axes) {
                Some(entry) => if replace { entry.1 = identity },
                None => selected.push((axes, identity)),
            }
        };
        if let Some(active) = self.curve.locked.clone().or_else(|| self.curve.hover.clone()) { set(&mut selected, active.axes, active.identity, true); }
        for (key, topology) in &self.fit.topologies { set(&mut selected, topology.axes, key.1.clone(), false); }
        let cycle = self.style.palette.line_cycle();
        let curve_width = self.style.artists.curve.linewidth;
        for overlay in self.fit.last_overlays.clone() {
            let axes = self.fit_target(&overlay);
            let identity = series_identity(&overlay.group_key);
            let Some(topology) = self.fit.topology(&(self.fit_facet_index(axes), identity.clone())).cloned() else { continue };
            let slots = &topology.slots;
            let chosen = selected.iter().find(|(a, _)| *a == axes).is_some_and(|(_, id)| *id == identity);
            let label = group_label(&overlay);
            let colour = (!overlay.group_key.is_empty()).then(|| cycle[series_slot(&overlay.group_key, cycle.len())].to_owned());
            let data_line = self.curve.series_indices.get(&axes).and_then(|v| v.iter().find(|(id, _)| *id == identity).map(|(_, i)| *i))
                .and_then(|index| self.curve.series_lines.get(&axes).map(|records| records[index].0));
            let (alpha, width) = match data_line.and_then(|line| self.fig.axes[axes].child(line)) {
                Some(Child::Line(line)) => (line.alpha.unwrap_or(1.0), line.linewidth / curve_width),
                Some(Child::Segments(bars)) => (bars.alpha, bars.linewidth / curve_width),
                Some(Child::Bars(bars)) => (bars.alpha, bars.linewidth / curve_width),
                _ => (1.0, 1.0),
            };
            for (line, polyline) in slots.fit_lines().into_iter().zip(&overlay.polylines) {
                let token = self.fit_polyline_token(polyline);
                if let Some(Child::Line(l)) = self.fig.axes[axes].child_mut(line) {
                    l.color = colour.clone().unwrap_or_else(|| token.color.to_owned());
                    l.alpha = Some(token.alpha * alpha);
                    l.linewidth = token.linewidth * width;
                }
            }
            for id in slots.texts().into_iter().flatten() {
                if let Some(child) = self.fig.axes[axes].child_mut(id) { child.set_visible(chosen); }
            }
            if !chosen { continue; }
            if topology.family == "failure" {
                let limit = if slots.detail == Detail::Headline { DIAGNOSTIC_FACET_MAX_CHARS } else { DIAGNOSTIC_SINGLE_MAX_CHARS };
                let content = truncate_diagnostic(&overlay.diagnostic, limit);
                let prefix = if label.is_empty() { String::new() } else { format!("{label}: ") };
                if let Some(Child::Text(text)) = slots.diagnostic.and_then(|id| self.fig.axes[axes].child_mut(id)) {
                    text.text = format!("{prefix}fit: {content}");
                }
            } else if slots.detail == Detail::Headline {
                let parameter = self.last_state.as_ref().and_then(|s| s.values.get(crate::configuration::display::FACET_FIT_PARAMETER))
                    .and_then(|v| v.as_str().map(str::to_owned));
                self.update_fit_headline_annotation(axes, slots, &overlay, parameter.as_deref());
            } else {
                self.update_fit_annotation(axes, slots, &overlay);
            }
            if let Some(colour) = &colour {
                for id in [slots.annotation, slots.annotation_value].into_iter().flatten() {
                    if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(id) { text.color = colour.clone(); }
                }
            }
        }
        self.refresh_classifier_focus();
        self.sync_roi_text_visibility();
    }
}
