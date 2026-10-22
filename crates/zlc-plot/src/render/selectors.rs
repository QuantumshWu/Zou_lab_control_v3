//! The selector layer (`_update_selectors`): the selector scene
//! (`interaction::scene`) built from each state's axes and the style, turned
//! into artists -- a Line2D per line, a scatter per handle row, a Text per
//! label -- that are rebuilt only when a kind's primitive topology changes.
//! A gesture's candidate replaces its kind's committed state until the
//! gesture ends. Right-aligned labels are pinned readouts.
use std::collections::HashMap;

use super::core::{Layer, Renderer};
use super::frame::RenderFrame;
use super::labels::literal_text;
use super::mpl::axes::{Child, ChildId, LineArtist, ScatterArtist, TextArtist, TextFrame};
use super::mpl::color;
use crate::configuration::selectors::{SelectorKind, SelectorState, SelectorValue};
use crate::interaction::controller::SelectorSnapshot;
use crate::configuration::spec::Surface;
use crate::interaction::scene::{SceneKind, SelectorItemContext, SelectorPrimitive, SelectorSceneOwner, SelectorSceneStyle, SelectorTarget};
use zlc_figure::ir::{HAlign, VAlign};

/// One primitive's topology: its type, axes and marker shape.
type Topology = (u8, SelectorTarget, Option<&'static str>);

/// The selector layer's retained state.
#[derive(Clone, Debug, Default)]
pub struct SelectorTables {
    /// `_selector_artists` with `_selector_topologies`, in insertion order.
    pub kinds: Vec<(SceneKind, Vec<(usize, ChildId)>, Vec<Topology>)>,
    /// `_last_selectors`: the frame's snapshot.
    pub last: SelectorSnapshot,
    /// `_selector_candidate`: a gesture's transient state.
    pub candidate: Option<SelectorState>,
    /// `_zlc_readout_widths` per pinned text: (words, size bits) -> widest width.
    pub readout_widths: HashMap<(usize, ChildId), HashMap<(String, u64), f64>>,
}

impl SelectorTables {
    /// Every selector artist, kind by kind (`_selector_artists.values()`).
    pub fn artists(&self) -> impl Iterator<Item = (usize, ChildId)> + '_ {
        self.kinds.iter().flat_map(|(_, artists, _)| artists.iter().copied())
    }
}

/// The selector layer of `present`.
pub struct SelectorLayer;

impl Layer for SelectorLayer {
    fn update(&self, renderer: &mut Renderer, frame: &RenderFrame) {
        // A height-bar scene shows no 2D selector (its state stays committed),
        // nor does a grid's overview.
        renderer.selectors.last = if renderer.height_bars.active || renderer.facet_overview() { Default::default() } else { frame.selectors.clone() };
        renderer.update_selectors();
    }
}

fn topology(primitive: &SelectorPrimitive) -> Topology {
    match primitive {
        SelectorPrimitive::Line(p) => (0, p.target.clone(), None),
        SelectorPrimitive::Markers(p) => (1, p.target.clone(), Some(p.shape)),
        SelectorPrimitive::Text(p) => (2, p.target.clone(), None),
    }
}

/// `_READOUT_NUMBER.sub("0", text)`: a readout's words with every number taken out.
pub fn readout_words(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let word = |c: Option<&char>| c.is_some_and(|c| c.is_ascii_alphanumeric() || *c == '_');
    let digits = |i: usize| { let mut j = i; while j < chars.len() && chars[j].is_ascii_digit() { j += 1; } j };
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        // [-+]?(?:\d+\.?\d*|\.\d+|inf)(?:e[-+]?\d+)?
        let number = (|| {
            for start in [i + usize::from(matches!(chars[i], '-' | '+')), i] {
                let mut j = start;
                let after = digits(j);
                if after > j {
                    j = after;
                    if j < chars.len() && chars[j] == '.' { j = digits(j + 1); }
                } else if j + 1 < chars.len() && chars[j] == '.' && chars[j + 1].is_ascii_digit() {
                    j = digits(j + 1);
                } else if chars[j..].starts_with(&['i', 'n', 'f']) {
                    j += 3;
                } else {
                    continue;
                }
                if j < chars.len() && chars[j] == 'e' {
                    let k = j + 1 + usize::from(j + 1 < chars.len() && matches!(chars[j + 1], '-' | '+'));
                    let end = digits(k);
                    if end > k { j = end; }
                }
                return Some(j);
            }
            None
        })();
        if let Some(end) = number {
            out.push('0');
            i = end;
            continue;
        }
        // \bnan\b | \bNaN\b
        let bounded = |token: &[char]| chars[i..].starts_with(token) && !word(i.checked_sub(1).map(|p| &chars[p]))
            && !word(chars.get(i + token.len()));
        if bounded(&['n', 'a', 'n']) || bounded(&['N', 'a', 'N']) {
            out.push('0');
            i += 3;
            continue;
        }
        if chars[i] == '\u{2014}' {
            out.push('0');
            i += 1;
            continue;
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

/// `f"{x:.1f}"`: the exact binary value rounded half-even to one decimal.
fn fixed1(x: f64) -> String {
    use crate::decimal::{Decimal, Integral};
    let scaled = Decimal::from_f64(x).scaleb(1);
    let floor = scaled.to_integral(Integral::Floor);
    let rest = scaled.sub(&Decimal::from_i128(floor));
    let n = match rest.cmp(&Decimal::from_f64(0.5)) {
        std::cmp::Ordering::Greater => floor + 1,
        std::cmp::Ordering::Less => floor,
        std::cmp::Ordering::Equal => floor + (floor % 2 != 0) as i128,
    };
    let sign = if n < 0 || (n == 0 && x.is_sign_negative()) { "-" } else { "" };
    format!("{sign}{}.{}", n.abs() / 10, n.abs() % 10)
}

impl Renderer {
    /// `_selector_target_for_axis(axis)`.
    pub(crate) fn selector_target(&self, axes: usize) -> SelectorTarget {
        let gid = &self.fig.axes[axes].gid;
        let (role, suffix) = match gid.split_once(':') { Some((r, s)) => (r, Some(s)), None => (gid.as_str(), None) };
        SelectorTarget {
            role: if role.is_empty() { "main".into() } else { role.into() },
            cell_index: suffix.filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())).and_then(|s| s.parse().ok()),
        }
    }

    fn axes_for_target(&self, target: &SelectorTarget) -> usize {
        (0..self.fig.axes.len()).find(|&i| self.selector_target(i) == *target).expect("selector target is not present")
    }

    /// `_selector_axis(state)`.
    fn selector_axes(&self, state: &SelectorState) -> usize {
        if self.is_facet() { self.facet_cell_named(state.facet_index.map(|i| i as usize), "selector") } else { self.primary_axes() }
    }

    /// `_make_selector_scene_owner(snapshot)` for the plot kinds this renderer draws.
    fn selector_scene_owner(&self, states: &[SelectorState]) -> SelectorSceneOwner {
        let style = &self.style;
        let threshold = &style.artists.threshold_line;
        let threshold_rgba = color::to_rgba_alpha(threshold.color, Some(threshold.alpha)).expect("threshold colour");
        let threshold_label = color::rgba(style.palette.threshold);
        let tuple = |c: [f64; 4]| (c[0], c[1], c[2], c[3]);
        let histogram = matches!(self.spec.semantic(), Surface::Histogram(_));
        let contexts = states.iter().map(|state| {
            let axes = self.selector_axes(state);
            let model = &self.fig.axes[axes];
            let selector_rgba = tuple(color::rgba(style.palette.line_single));
            let (mut item_threshold, mut item_label) = (threshold_rgba, threshold_label);
            if state.kind() == SelectorKind::Threshold && histogram {
                if let Some((line, identity, _)) = self.classifier_series(axes) {
                    if !identity.is_empty() {
                        let colour = self.series_colour(axes, line);
                        item_threshold = color::to_rgba_alpha(&colour, Some(threshold.alpha)).expect("series colour");
                        item_label = color::rgba(&colour);
                    }
                }
            }
            let mut context = SelectorItemContext {
                kind: SceneKind::Selector(state.kind()),
                target: self.selector_target(axes),
                label_target: self.selector_target(axes),
                x_limits: model.xlim,
                y_limits: model.ylim,
                selector_rgba,
                color_limit_rgba: (selector_rgba, selector_rgba),
                threshold_rgba: tuple(item_threshold),
                threshold_label_rgba: tuple(item_label),
                threshold_uses_x: histogram,
                threshold_text: self.histogram_threshold_text(state),
                x_scale: self.axis_scale(axes, true),
                y_scale: self.axis_scale(axes, false),
                x_label_factor: self.pulse_label_factor(),
                sample_value: None,
                sample_span: None,
                sample_target: None,
                sample_x_limits: None,
                sample_rgba: None,
            };
            if matches!(self.spec.semantic(), Surface::Image(_)) { self.image_selector_context(state, axes, &mut context); }
            context
        }).collect();
        let visual = &style.artists;
        SelectorSceneOwner {
            contexts,
            style: SelectorSceneStyle {
                line_width_pt: visual.selector_line_width,
                line_alpha: visual.selector_alpha,
                handle_size_pt: visual.selector_handle_size_pt,
                handle_edge_width_pt: visual.selector_handle_edge_width,
                crosshair_size_pt: visual.crosshair_marker_size_pt,
                annotation_pt: style.fonts.annotation_pt,
                side_distribution_width_pt: visual.side_distribution_linewidth,
                selector_zorder: visual.selector_zorder,
                font_family: style.fonts.family.to_owned(),
                label_inset_fraction: style.render.axes_text_inset_fraction,
                label_line_fraction: style.render.selector_label_line_fraction,
                threshold_width_pt: threshold.linewidth,
                threshold_linestyle: threshold.linestyle.to_owned(),
            },
            color_context: self.color_limit_context(tuple(threshold_rgba), tuple(threshold_label)),
        }
    }

    /// `_series_colour(artist)`: a line's colour, a bar collection's face without alpha.
    pub(crate) fn series_colour(&self, axes: usize, id: ChildId) -> String {
        match self.fig.axes[axes].child(id) {
            Some(Child::Line(line)) => line.color.clone(),
            Some(Child::Segments(bars)) => color::tuple_spec([bars.facecolor[0], bars.facecolor[1], bars.facecolor[2], 1.0]),
            Some(Child::Bars(bars)) => { let face = bars.facecolor(); color::tuple_spec([face[0], face[1], face[2], 1.0]) }
            _ => self.style.palette.line_single.to_owned(),
        }
    }

    /// `_histogram_threshold_text(state)`: one effective threshold read from the painted bins.
    fn histogram_threshold_text(&self, state: &SelectorState) -> String {
        let SelectorValue::Threshold(threshold) = state.value else { return String::new() };
        if !matches!(self.spec.semantic(), Surface::Histogram(_)) { return String::new(); }
        let axes = self.selector_axes(state);
        let selected = self.classifier_series(axes);
        let group_index = match &selected {
            None => 0,
            Some((_, identity, _)) => self.curve.series_indices.get(&axes)
                .and_then(|v| v.iter().find(|(id, _)| id == identity).map(|(_, i)| *i)).unwrap_or(0),
        };
        let mut caption = String::new();
        match &selected {
            Some((_, identity, label)) if !identity.is_empty() => {
                let locked = self.curve.locked.as_ref().is_some_and(|l| l.axes == axes);
                caption = format!("{}{}
", if locked { "* " } else { "" }, literal_text(label));
                if let Some(hover) = &self.curve.hover {
                    if !locked && hover.axes == axes && hover.identity != *identity {
                        caption = format!("{} (hover)
{}: ", literal_text(&hover.label), literal_text(label));
                    }
                }
            }
            Some(_) => {}
            None => {
                let active = self.curve.locked.as_ref().or(self.curve.hover.as_ref());
                if let Some(active) = active.filter(|a| a.axes == axes && !a.identity.is_empty()) {
                    let marker = if self.curve.locked.is_some() { "* " } else { "" };
                    caption = format!("{marker}{} (shared classifier)
", literal_text(&active.label));
                }
            }
        }
        // A grid's threshold reads its named (else selected) cell.
        let last = self.last_payload.clone();
        let (facet_index, payload) = match last.as_deref() {
            Some(crate::projection::output::Payload::Facet(facet)) if self.is_facet() => {
                match state.facet_index.map(|index| index as usize).or(self.facet.focused).filter(|&index| index < facet.cells.len()) {
                    Some(index) => (Some(index), Some(&facet.cells[index].payload)),
                    None => return String::new(),
                }
            }
            other => (None, other),
        };
        let key = (facet_index, selected.as_ref().map(|s| s.1.clone()).unwrap_or_default());
        if let Some(label) = self.classifier.label(&key) { return caption + label; }
        let Some(crate::projection::output::Payload::Histogram(histogram)) = payload else { return String::new() };
        let edges = histogram.edges.display.to_f64();
        let counts = histogram.counts.to_f64();
        let shape = histogram.counts.shape();
        let (groups, bins) = if shape.len() == 2 { (shape[0], shape[1]) } else { (1, counts.len()) };
        let weights: Vec<f64> = match selected {
            None => (0..bins).map(|b| (0..groups).map(|g| counts[g * bins + b]).sum()).collect(),
            Some(_) if group_index < groups => counts[group_index * bins..(group_index + 1) * bins].to_vec(),
            Some(_) => return String::new(),
        };
        if edges.len() != weights.len() + 1 || weights.is_empty() { return String::new(); }
        let total: f64 = weights.iter().sum();
        let left_fraction = if total <= 0.0 { 0.0 } else {
            let right = edges.partition_point(|&e| e <= threshold) as i64 - 1;
            let index = right.clamp(-1, weights.len() as i64);
            if index < 0 { 0.0 } else if index as usize >= weights.len() { 1.0 } else {
                let i = index as usize;
                let width = (edges[i + 1] - edges[i]).max(1.0e-300);
                let fraction = ((threshold - edges[i]) / width).clamp(0.0, 1.0);
                (weights[..i].iter().sum::<f64>() + weights[i] * fraction) / total
            }
        };
        format!("{caption}th={}
L/R={}%/{}%", crate::ticks::compact_number(threshold, None),
                fixed1(100.0 * left_fraction), fixed1(100.0 * (1.0 - left_fraction)))
    }

    /// `_new_selector_artist(primitive)`.
    fn new_selector_artist(&mut self, primitive: &SelectorPrimitive) -> (usize, ChildId) {
        let child = match primitive {
            SelectorPrimitive::Line(p) => {
                let axes = self.axes_for_target(&p.target);
                // `axis.plot((), ())`: the cycle's next colour, until the scene recolours it.
                let color = self.fig.axes[axes].next_line_color(&self.style.palette.line_cycle());
                (axes, Child::Line(LineArtist {
                    visible: true, zorder: 2.0, alpha: None, color, linewidth: self.rc.lines_linewidth,
                    linestyle: "-".into(), marker: "None".into(), markersize: self.rc.lines_markersize,
                    markeredgewidth: self.rc.lines_markeredgewidth, markevery: None, marker_colors: None, label: String::new(),
                    x: Vec::new(), y: Vec::new(),
                }))
            }
            SelectorPrimitive::Markers(p) => {
                let axes = self.axes_for_target(&p.target);
                (axes, Child::Scatter(ScatterArtist {
                    visible: true, zorder: 1.0, alpha: None, marker: match p.shape { "square" => "s", _ => "o" },
                    offsets: Vec::new(), sizes: Vec::new(), facecolors: Vec::new(), edgecolors: Vec::new(), linewidths: Vec::new(),
                }))
            }
            SelectorPrimitive::Text(p) => {
                let axes = self.axes_for_target(&p.target);
                (axes, Child::Text(TextArtist {
                    visible: true, zorder: 3.0, x: 0.0, y: 0.0, text: String::new(), color: "black".into(),
                    size: self.style.fonts.annotation_pt, ha: HAlign::Left, va: VAlign::Top, ma: None, clip_on: true,
                    frame: TextFrame::Axes, family: None,
                }))
            }
        };
        let (axes, child) = child;
        let plotted = !matches!(child, Child::Text(_));
        let id = self.fig.axes[axes].add(child);
        if plotted { self.plotted(axes); }
        (axes, id)
    }

    /// `_mutate_selector_artist(artist, primitive)`.
    fn mutate_selector_artist(&mut self, (axes, id): (usize, ChildId), primitive: &SelectorPrimitive) {
        let tuple = |c: (f64, f64, f64, f64)| [c.0, c.1, c.2, c.3];
        match (self.fig.axes[axes].child_mut(id), primitive) {
            (Some(Child::Line(line)), SelectorPrimitive::Line(p)) => {
                line.x = p.points.iter().map(|q| q.0).collect();
                line.y = p.points.iter().map(|q| q.1).collect();
                line.color = color::tuple_spec(tuple(p.color));
                line.linewidth = p.width_pt;
                line.linestyle = p.linestyle.clone();
                line.visible = true;
                line.zorder = p.zorder;
            }
            (Some(Child::Scatter(scatter)), SelectorPrimitive::Markers(p)) => {
                scatter.offsets = p.points.iter().map(|q| [q.0, q.1]).collect();
                scatter.sizes = vec![p.size_pt * p.size_pt; p.points.len()];
                scatter.facecolors = vec![tuple(p.facecolor)];
                scatter.edgecolors = p.edgecolor.map(|c| vec![tuple(c)]).unwrap_or_default();
                scatter.linewidths = vec![p.edge_width_pt];
                scatter.visible = true;
                scatter.zorder = p.zorder;
            }
            (Some(Child::Text(text)), SelectorPrimitive::Text(p)) => {
                text.text = p.text.clone();
                text.x = p.position.0;
                text.y = p.position.1;
                text.ha = match p.horizontal_alignment { "right" => HAlign::Right, "center" => HAlign::Center, _ => HAlign::Left };
                text.color = color::tuple_spec(tuple(p.color));
                text.family = Some(vec![p.font_family.clone()]);
                text.size = p.font_size_pt;
                text.clip_on = true;
                text.visible = true;
                text.zorder = p.zorder;
            }
            _ => panic!("selector artist {id} does not match its primitive"),
        }
    }

    /// `_update_selectors(self._last_selectors)`.
    pub(crate) fn update_selectors(&mut self) {
        let mut snapshot = self.selectors.last.clone();
        if let Some(candidate) = self.selectors.candidate { snapshot.candidate = Some(candidate); }
        let owner = self.selector_scene_owner(&snapshot.states());
        let scene = owner.build(&snapshot.states(), self.color_limit_state().as_ref()).expect("selector scene");
        let active = scene.selector_kinds();
        for (kind, artists, topology) in std::mem::take(&mut self.selectors.kinds) {
            if active.contains(&kind) {
                self.selectors.kinds.push((kind, artists, topology));
            } else {
                self.remove_selector_artists(&artists);
            }
        }
        for (kind, primitives) in &scene.groups {
            let wanted: Vec<Topology> = primitives.iter().map(topology).collect();
            let position = self.selectors.kinds.iter().position(|(k, _, _)| k == kind);
            let artists = match position {
                Some(p) if self.selector_topology(p) == wanted => self.selectors.kinds[p].1.clone(),
                _ => {
                    if let Some(p) = position {
                        let old = self.selectors.kinds[p].1.clone();
                        self.remove_selector_artists(&old);
                    }
                    let artists: Vec<(usize, ChildId)> = primitives.iter().map(|p| self.new_selector_artist(p)).collect();
                    match position {
                        Some(p) => { self.selectors.kinds[p].1 = artists.clone(); }
                        None => self.selectors.kinds.push((*kind, artists.clone(), Vec::new())),
                    }
                    artists
                }
            };
            let p = self.selectors.kinds.iter().position(|(k, _, _)| k == kind).unwrap();
            self.selectors.kinds[p].2 = wanted;
            for (&artist, primitive) in artists.iter().zip(primitives) {
                self.mutate_selector_artist(artist, primitive);
                if let SelectorPrimitive::Text(text) = primitive {
                    if text.horizontal_alignment == "right" { self.pin_readout_held(artist.0, artist.1, text.position.0, false); }
                }
            }
        }
        self.sync_roi_text_visibility();
    }

    fn selector_topology(&self, position: usize) -> Vec<Topology> { self.selectors.kinds[position].2.clone() }

    pub(crate) fn remove_selector_artists(&mut self, artists: &[(usize, ChildId)]) {
        for &(axes, id) in artists {
            self.curve.sources.remove(&(axes, id));
            self.fig.axes[axes].remove(id);
            self.selectors.readout_widths.remove(&(axes, id));
        }
    }

    /// `_sync_roi_text_visibility`: visible fit text takes precedence over ROI coordinates on its axes.
    pub(crate) fn sync_roi_text_visibility(&mut self) {
        let labels: Vec<(usize, ChildId)> = self.selectors.kinds.iter()
            .filter(|(kind, _, _)| matches!(kind, SceneKind::Selector(SelectorKind::Area | SelectorKind::XRange)))
            .flat_map(|(_, artists, _)| artists.iter().copied())
            .filter(|&(axes, id)| matches!(self.fig.axes[axes].child(id), Some(Child::Text(_))))
            .collect();
        if labels.is_empty() { return; }
        let fit_axes: Vec<usize> = self.fit.topologies.iter().filter(|(_, t)| {
            self.fig.axes[t.axes].visible && [t.slots.annotation, t.slots.annotation_value, t.slots.diagnostic].into_iter().flatten()
                .any(|id| matches!(self.fig.axes[t.axes].child(id), Some(Child::Text(text)) if text.visible && !text.text.is_empty()))
        }).map(|(_, t)| t.axes).collect();
        for (axes, id) in labels {
            let visible = !fit_axes.contains(&axes);
            if let Some(child) = self.fig.axes[axes].child_mut(id) { child.set_visible(visible); }
        }
    }

    /// `_pin_readout(artist, right, fit=fit)` of a readout that keeps its own
    /// widest width per set of words; `fit` shrinks it to the room it has.
    pub(crate) fn pin_readout_held(&mut self, axes: usize, id: ChildId, right: f64, fit: bool) {
        let dpi = self.fig.dpi;
        let bbox = self.fig.axes[axes].bbox_exact(self.fig.size_inches, dpi);
        let Some(Child::Text(text)) = self.fig.axes[axes].child(id) else { return };
        if text.text.is_empty() || dpi <= 0.0 { return; }
        let fit_pt = if self.is_facet() { self.style.fonts.facet_fit_annotation_pt } else { self.style.fonts.annotation_pt };
        let nominal = if fit { fit_pt } else { text.size };
        let measure = self.measure.clone();
        let measured = |content: &str| content.split('\n')
            .map(|line| measure.width(&super::rolling::zero_digits(&line.replace("\\$", "$")), nominal))
            .fold(f64::NEG_INFINITY, f64::max);
        let shown = measured(&text.text);
        let words = readout_words(&text.text);
        let held = self.selectors.readout_widths.entry((axes, id)).or_default();
        let key = (words, nominal.to_bits());
        let width = shown.max(held.get(&key).copied().unwrap_or(0.0));
        held.insert(key, width);
        let room = (bbox[2] - bbox[0]) * 72.0 / dpi;
        if room <= 0.0 { return; }
        let mut size = nominal;
        if fit {
            let usable = room * (1.0 - 2.0 * self.style.render.axes_text_inset_fraction);
            let fitted = if width > usable { nominal * usable / width } else { nominal };
            let minimum = self.style.fonts.facet_title_min_pt;
            size = if fitted > minimum { fitted } else { minimum };
            let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(id) else { return };
            text.size = size;
            if fitted < minimum {
                text.ha = HAlign::Right;
                text.ma = Some(HAlign::Right);
                text.x = right;
                return;
            }
        }
        let scale = size / nominal;
        let figure_edge = right * room + bbox[0] * 72.0 / dpi;
        let drawn = (width * scale).min(figure_edge.max(shown * scale));
        if let Some(Child::Text(text)) = self.fig.axes[axes].child_mut(id) {
            text.ha = HAlign::Left;
            text.ma = Some(HAlign::Left);
            text.x = right - drawn / room;
        }
    }

    /// `begin_selector_gesture(kind, compose)`: the scene is updated only
    /// for a compose that follows at once.
    pub fn begin_selector_gesture(&mut self, kind: SelectorKind, compose: bool) {
        self.selector_gesture = Some(kind);
        self.selectors.candidate = None;
        if compose { self.update_selectors(); }
    }

    /// `preview_selector(state)`: false when the state is not the gesture's kind.
    pub fn preview_selector(&mut self, state: &SelectorState) -> bool {
        if self.selector_gesture != Some(state.kind()) { return false; }
        self.selectors.candidate = Some(*state);
        self.update_selectors();
        true
    }

    /// `end_selector_gesture`.
    pub fn end_selector_gesture(&mut self) {
        self.selectors.candidate = None;
        self.selector_gesture = None;
    }
}
