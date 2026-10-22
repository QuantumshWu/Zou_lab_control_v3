//! A matplotlib `Axes` as the renderer keeps it: its box, view limits, two
//! `Axis` objects with their tick parameters and tick policy, title texts,
//! spines and its ordered children -- and the IR it resolves to.
use zlc_figure::ir::{self, Artist, HAlign, NodeId, RotationMode, VAlign};
use zlc_figure::transform::{bounds, transformed_points, M3};

/// `BboxTransformTo(bbox)`.
pub fn bbox_to(p: &[f64; 4]) -> M3 {
    let (l, b, w, h) = bounds(p);
    M3([[w, 0.0, l], [0.0, h, b], [0.0, 0.0, 1.0]])
}

/// `BboxTransformFrom(bbox)`.
pub fn bbox_from(p: &[f64; 4]) -> M3 {
    let (l, b, w, h) = bounds(p);
    let (xs, ys) = (1.0 / w, 1.0 / h);
    M3([[xs, 0.0, -l * xs], [0.0, ys, -b * ys], [0.0, 0.0, 1.0]])
}

use super::artist::{self, LineSpec};
use super::color;
use super::nodes::{AxesNodes, FigureNodes, Nodes};
use super::rc::Rc;
use crate::layout::Room;
use crate::ticks::{DeclaredLocator, Geometry, Measure, MemoKey, Scale, SmartOffsetLocator};

/// One family's (major or minor) tick parameters: `_major_tick_kw` merged
/// over the rc defaults a `Tick` reads.
#[derive(Clone, Debug, PartialEq)]
pub struct TickParams {
    pub size: f64,
    pub width: f64,
    pub pad: f64,
    pub color: String,
    pub labelsize: f64,
    /// Whether `tick_params` set the label size (`'labelsize' in _major_tick_kw`).
    pub labelsize_kw: bool,
    pub labelcolor: String,
    pub tickdir: String,
    pub tick1on: bool,
    pub tick2on: bool,
    pub label1on: bool,
    pub label2on: bool,
    pub grid_on: bool,
    /// `grid(color=..., linewidth=...)`: `_major_tick_kw['grid_color']` and
    /// `['grid_linewidth']`, kept once given; None: the rc's.
    pub grid_color: Option<String>,
    pub grid_linewidth: Option<f64>,
}

impl TickParams {
    /// `tick_params(labelsize=size)`.
    pub fn set_labelsize(&mut self, size: f64) {
        self.labelsize = size;
        self.labelsize_kw = true;
    }

    fn new(rc: &Rc, x: bool, major: bool) -> TickParams {
        let t = rc.tick(x);
        let labelcolor = if t.labelcolor == "inherit" { t.color.clone() } else { t.labelcolor.clone() };
        TickParams {
            size: if major { t.major_size } else { t.minor_size },
            width: if major { t.major_width } else { t.minor_width },
            pad: if major { t.major_pad } else { t.minor_pad },
            color: t.color.clone(),
            labelsize: t.labelsize,
            labelsize_kw: false,
            labelcolor,
            tickdir: t.direction.clone(),
            // Axes.clear: the major/minor visibility the rc says (minor ticks
            // and their labels are off in the style).
            tick1on: major && t.tick1,
            tick2on: major && t.tick2,
            label1on: major && t.label1,
            label2on: major && t.label2,
            grid_on: false,
            grid_color: None,
            grid_linewidth: None,
        }
    }

    /// `Tick._apply_tickdir`: the pad the labels stand at.
    pub fn label_pad(&self) -> f64 {
        let factor = match self.tickdir.as_str() { "in" => 0.0, "inout" => 0.5, _ => 1.0 };
        self.pad + self.size * factor
    }
}

/// The text rule of a declared tick ladder.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum DeclaredText { Count, Compact }

fn declared_text(rule: DeclaredText) -> &'static dyn Fn(f64, Option<usize>) -> String {
    match rule {
        DeclaredText::Count => &|value, length| crate::ticks::count_label(value, length),
        DeclaredText::Compact => &|value, length| crate::ticks::compact_number(value, length),
    }
}

/// What an axis' major locator and formatter are.
#[derive(Clone, Debug)]
pub enum TickPolicy {
    /// matplotlib's AutoLocator/ScalarFormatter (never drawn by this renderer).
    Default,
    Smart(Box<SmartOffsetLocator>),
    Named { ticks: Vec<f64>, texts: Vec<String> },
    Declared { locator: Box<DeclaredState> },
    /// A pulse timeline's time axis: `MaxNLocator(nbins, prune="both")` and a
    /// FuncFormatter printing `value * factor` to four significant digits,
    /// nothing for a negative value.
    PulseTime { nbins: usize, factor: f64 },
}

/// One `Tick` object's own label and mark properties where they differ from
/// what `_major_tick_kw` gives every tick: `set_ticklabels(fontsize=...)`, a
/// label's `set_fontsize`/`set_color`, a mark's `set_markeredgecolor`. A new
/// tick copies the first one's (`_copy_tick_props`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TickOverride {
    /// label1, label2.
    pub label_size: [Option<f64>; 2],
    pub label_color: [Option<String>; 2],
    pub mark_color: Option<String>,
}

/// A `DeclaredLocator`'s retained state (the text rule is a function).
#[derive(Clone, Debug)]
pub struct DeclaredState {
    pub count: usize,
    pub rule: DeclaredText,
    pub text_lengths: Vec<Option<usize>>,
    pub zero_optional: bool,
    pub label_pt: f64,
    pub span: Option<(f64, f64)>,
    pub ladder: crate::ticks::Ladder,
    pub memo: Option<MemoKey>,
}

impl DeclaredState {
    pub fn new(count: usize, rule: DeclaredText, text_lengths: Vec<Option<usize>>, zero_optional: bool, label_pt: f64,
               span: Option<(f64, f64)>) -> DeclaredState {
        let locator = DeclaredLocator::new(count, declared_text(rule), text_lengths.clone(), zero_optional, label_pt, span);
        DeclaredState { count, rule, text_lengths, zero_optional, label_pt, span, ladder: locator.ladder.clone(), memo: None }
    }

    pub fn with<T>(&mut self, body: impl FnOnce(&mut DeclaredLocator<'_>) -> T) -> T {
        let mut locator = DeclaredLocator::new(self.count, declared_text(self.rule), self.text_lengths.clone(),
                                               self.zero_optional, self.label_pt, self.span);
        locator.ladder = self.ladder.clone();
        locator.set_memo(self.memo.clone());
        let out = body(&mut locator);
        self.ladder = locator.ladder.clone();
        self.memo = locator.memo();
        out
    }
}

/// The once-per-configuration install signature (`_zlc_tick_signature`).
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub enum TickSignature {
    Smart { name: char, scale: String, size_pt: u64 },
    Named { ticks: Vec<u64>, texts: Vec<String>, size_pt: u64 },
    Declared { count: usize, size_pt: u64, lengths: Vec<Option<usize>>, zero_optional: bool, rule: DeclaredText },
}

/// zlc's SmartOffsetFormatter re-placement of the offset text.
#[derive(Clone, Debug, PartialEq)]
pub struct OffsetPin { pub xy: (f64, f64), pub ha: HAlign, pub va: VAlign }

/// What the painted geometry of an axis is, for its locator.
#[derive(Clone, Copy, Debug)]
pub struct TickGeometry {
    /// The axes box in figure pixels (x0, y0, x1, y1).
    pub bbox: [f64; 4],
    /// The figure box in pixels (width, height).
    pub figure: (f64, f64),
    pub dpi: f64,
    pub room: Option<Room>,
}

/// `axis_space` of values on a FuncScale lattice (None for a linear axis).
fn lattice_space(lattice: &Option<std::sync::Arc<[f64]>>, values: &[f64]) -> Option<Vec<f64>> {
    lattice.as_ref().map(|l| values.iter().map(|&v| zlc_figure::transform::piecewise(v, l)).collect())
}

#[derive(Clone, Debug)]
pub struct AxisModel {
    pub x: bool,
    /// `set_label_position`: "bottom"/"top" or "left"/"right".
    pub label_position: &'static str,
    /// A FuncScale over this nonuniform coordinate lattice (`axis_space`);
    /// None for a linear axis.
    pub function: Option<std::sync::Arc<[f64]>>,
    /// Which FuncScale object it is: every `set_xscale` makes a new one.
    pub scale_serial: u64,
    pub label: String,
    pub label_size: f64,
    pub label_color: String,
    pub labelpad: f64,
    pub major: TickParams,
    pub minor: TickParams,
    pub policy: TickPolicy,
    pub signature: Option<TickSignature>,
    pub offset_visible: bool,
    pub offset_size: f64,
    pub offset_color: String,
    pub offset_pin: Option<OffsetPin>,
    /// The last `_update_ticks`: locator output and formatted labels.
    pub locs: Vec<f64>,
    pub labels: Vec<String>,
    /// `majorTicks`, which only grows: an update moves the first
    /// `len(locs)` ticks onto the new (location, label) and leaves the rest
    /// where they were -- what a grid's label carrier still holds.
    pub tick_pool: Vec<(f64, String)>,
    /// What the formatter's offset is after the last `_update_ticks`.
    pub offset_string: String,
    /// The major `Tick` objects made so far, by index, as far as they differ
    /// from the tick parameters.
    pub ticks: Vec<TickOverride>,
    /// What the offset Text says: set when the axis draws, or when a grid's
    /// chrome shows it (`offsetText.set_text`).
    pub offset_text: String,
    /// How many leading Tick objects hold label transforms of their own (a
    /// `tick_params` that moves the labels re-made them per existing tick).
    pub distinct_ticks: usize,
    /// Where the last draw put the label and the unpinned offset text.
    pub drawn_label_xy: Option<(f64, f64)>,
    pub drawn_offset_xy: Option<(f64, f64)>,
    /// `get_scale() == "log"`: base 10, nonpositive values clipped.
    pub log: bool,
    /// The minor locator is the log policy's `LogLocator(10, subs=2..9,
    /// numticks=16)` with a NullFormatter; otherwise a NullLocator.
    pub minor_log: bool,
    /// The last `_update_ticks`' minor locations (`get_minorticklocs`).
    pub minor_locs: Vec<f64>,
    /// An x axis whose end labels align inside the axes (a narrow rail's
    /// count axis beside a colorbar): its labels stay over its own extent.
    pub inward_ends: bool,
}

/// `LogTransform(10, nonpositive="clip").transform_non_affine(value)`.
pub fn log_clip(value: f64) -> f64 { if value <= 0.0 { -1000.0 } else { value.log10() } }

/// `LogLocator(10.0, subs=(2.0, ..., 9.0), numticks=16).tick_values(vmin, vmax)`
/// for a positive view (a log axis' limits always are).
pub fn log_minor_ticks(vmin: f64, vmax: f64) -> Vec<f64> {
    let (vmin, vmax) = if vmax < vmin { (vmax, vmin) } else { (vmin, vmax) };
    let base = 10.0f64;
    let log_vmin = vmin.ln() / base.ln();
    let log_vmax = vmax.ln() / base.ln();
    let numdec = log_vmax.floor() as i64 - log_vmin.ceil() as i64;
    let mut stride = numdec.div_euclid(16) + 1;
    if stride >= numdec { stride = 1.max(numdec - 1); }
    if stride != 1 { return Vec::new(); }
    let (first, stop) = (log_vmin.floor() as i64 - stride, log_vmax.ceil() as i64 + 2 * stride);
    let ticks: Vec<f64> = (first..stop).flat_map(|decade| {
        let start = base.powf(decade as f64);
        (2..10).map(move |sub| sub as f64 * start)
    }).collect();
    if ticks.iter().filter(|&&t| vmin <= t && t <= vmax).count() <= 1 {
        return super::ticker::auto_ticks(vmin, vmax, 9);
    }
    ticks
}

impl AxisModel {
    pub fn new(rc: &Rc, x: bool) -> AxisModel {
        AxisModel {
            x,
            label_position: if x { "bottom" } else { "left" },
            function: None,
            scale_serial: 0,
            label: String::new(),
            label_size: rc.axes_labelsize,
            label_color: rc.axes_labelcolor.clone(),
            labelpad: rc.axes_labelpad,
            major: TickParams::new(rc, x, true),
            minor: TickParams::new(rc, x, false),
            policy: TickPolicy::Default,
            signature: None,
            offset_visible: true,
            offset_size: rc.tick(x).labelsize,
            offset_color: { let t = rc.tick(x); if t.labelcolor == "inherit" { t.color.clone() } else { t.labelcolor.clone() } },
            offset_pin: None,
            locs: Vec::new(),
            labels: Vec::new(),
            tick_pool: Vec::new(),
            offset_string: String::new(),
            ticks: Vec::new(),
            offset_text: String::new(),
            distinct_ticks: 0,
            drawn_label_xy: None,
            drawn_offset_xy: None,
            log: false,
            minor_log: false,
            minor_locs: Vec::new(),
            inward_ends: false,
        }
    }

    /// `get_major_ticks(count)`: at least `count` tick objects, a new one a
    /// copy of the first.
    pub fn tick_objects(&mut self, count: usize) -> &mut [TickOverride] {
        while self.ticks.len() < count {
            let copied = self.ticks.first().cloned().unwrap_or_default();
            self.ticks.push(copied);
        }
        &mut self.ticks[..count]
    }

    /// The size tick `index`'s label (1 or 2) is drawn at.
    pub fn label_size(&self, index: usize, side: usize) -> f64 {
        self.ticks.get(index).and_then(|tick| tick.label_size[side - 1]).unwrap_or(self.major.labelsize)
    }

    /// `tick_params(labelsize=size)`: every tick's labels, and the ones to come.
    pub fn set_label_size_all(&mut self, size: f64) {
        self.major.set_labelsize(size);
        for tick in &mut self.ticks { tick.label_size = [None, None]; }
    }

    pub fn name(&self) -> &'static str { if self.x { "x" } else { "y" } }

    /// `Axis._set_axes_scale("log" | "linear")`: the scale's default locators
    /// and formatters replace whatever policy was installed.
    pub fn set_scale(&mut self, log: bool) {
        self.log = log;
        self.policy = TickPolicy::Default;
        self.minor_log = false;
        self.minor_locs.clear();
    }

    /// `get_minorticklocs()`: the minor locator's locations less those that
    /// coincide with a major tick where the scale draws them.
    fn minor_ticks(&self, view: (f64, f64)) -> Vec<f64> {
        if !self.minor_log { return Vec::new(); }
        let minor = log_minor_ticks(view.0, view.1);
        let major: Vec<f64> = self.locs.iter().map(|&v| log_clip(v)).collect();
        let (a, b) = (log_clip(view.0), log_clip(view.1));
        let (lo, hi) = if b < a { (b, a) } else { (a, b) };
        let tol = (hi - lo) * 1e-5;
        minor.into_iter().filter(|&v| {
            let t = log_clip(v);
            !major.iter().any(|&m| ((t - m).abs() <= tol && m.is_finite()) || t == m)
        }).collect()
    }

    /// `Axis._get_ticks_position()` as `get_ticks_position()` names it.
    pub fn ticks_position(&self) -> &'static str {
        let ticks = [&self.major, &self.minor];
        if ticks.iter().all(|t| t.tick1on && !t.tick2on && t.label1on && !t.label2on) {
            if self.x { "bottom" } else { "left" }
        } else if ticks.iter().all(|t| t.tick2on && !t.tick1on && t.label2on && !t.label1on) {
            if self.x { "top" } else { "right" }
        } else if ticks.iter().all(|t| t.tick1on && t.tick2on && t.label1on && !t.label2on) {
            "default"
        } else {
            "unknown"
        }
    }

    /// `_room_pt` + `_geometry`: (extent, before, after, across, horizontal) in points.
    pub fn geometry(&self, g: &TickGeometry) -> Option<Geometry> {
        let dots_per_point = g.dpi / 72.0;
        if dots_per_point <= 0.0 { return None; }
        let side = self.ticks_position();
        let [x0, y0, x1, y1] = g.bbox;
        let (fw, fh) = g.figure;
        let across = match g.room {
            None => {
                let across = if self.x {
                    if side != "top" { y0 - 0.0 } else { fh - y1 }
                } else if side != "right" { x0 - 0.0 } else { fw - x1 };
                across / dots_per_point
            }
            Some(room) => {
                let width = fw / dots_per_point;
                let height = fh / dots_per_point;
                if self.x { (if side == "top" { room.top } else { room.bottom }) * height }
                else { (if side == "right" { room.right } else { room.left }) * width }
            }
        };
        let (before, after) = if self.x { (x0 - 0.0, fw - x1) } else { (y0 - 0.0, fh - y1) };
        let extent = if self.x { x1 - x0 } else { y1 - y0 };
        Some(Geometry { extent: extent / dots_per_point, before: before / dots_per_point, after: after / dots_per_point,
                        across: Some(across), horizontal: self.x, inward: self.x && self.inward_ends })
    }

    /// Whether this inward axis' labels did not fit inside it and are drawn centred (v2's way).
    pub fn labels_centred(&self) -> bool { matches!(&self.policy, TickPolicy::Declared { locator } if locator.ladder.centred) }

    /// The scale's non-affine transform of `value`: `axis_space(value)` on a
    /// FuncScale, `LogTransform(10, clip)` on a log axis, the value on a linear one.
    pub fn space(&self, value: f64) -> f64 {
        match &self.function {
            Some(lattice) => zlc_figure::transform::piecewise(value, lattice),
            None if self.log => log_clip(value),
            None => value,
        }
    }

    /// The axis' scale as its transform node.
    pub fn scale_node(&self) -> super::nodes::ScaleNode<'_> {
        match &self.function {
            Some(lattice) => super::nodes::ScaleNode::Lattice(lattice),
            None if self.log => super::nodes::ScaleNode::Log,
            None => super::nodes::ScaleNode::Linear,
        }
    }

    /// `body(scale)` with this axis' scale as the tick ladder prices it.
    pub fn with_scale<T>(&self, body: impl FnOnce(&Scale) -> T) -> T {
        let space = |values: &[f64]| lattice_space(&self.function, values);
        body(&if self.function.is_some() { Scale::Function(&space) } else if self.log { Scale::Log10 } else { Scale::Linear })
    }

    /// The tick label size the axis draws at (`_major_tick_kw['labelsize']`).
    pub fn labelsize(&self) -> f64 { self.major.labelsize }

    /// `_apply_drawn_size`: the ladder resized the labels.
    fn apply_drawn_size(&mut self, before: f64, after: f64) {
        if (after - before).abs() < 1.0e-9 { return; }
        self.set_label_size_all(after);
        self.offset_size = after;
    }

    /// The locator's drawn size (or None for a policy without a ladder).
    pub fn drawn_pt(&self) -> Option<f64> {
        match &self.policy {
            TickPolicy::Smart(locator) => Some(locator.ladder.drawn_pt),
            TickPolicy::Declared { locator } => Some(locator.ladder.drawn_pt),
            _ => None,
        }
    }

    /// `Axis._update_ticks()`'s locator and formatter half: the locations
    /// and labels; `view` is the axis' view interval as it is stored.
    pub fn update_ticks(&mut self, view: (f64, f64), g: &TickGeometry, measure: &dyn Measure) {
        let geometry = self.geometry(g);
        let horizontal = self.x;
        let (vmin, vmax) = view;
        let lattice = self.function.clone();
        let space = |values: &[f64]| lattice_space(&lattice, values);
        let scale = if self.function.is_some() { Scale::Function(&space) } else if self.log { Scale::Log10 } else { Scale::Linear };
        let serial = self.scale_serial;
        match &mut self.policy {
            TickPolicy::Default => {
                // matplotlib's AutoLocator + ScalarFormatter.
                let dpi = g.dpi;
                let inverse = M3([[1.0 / dpi, 0.0, 0.0], [0.0, 1.0 / dpi, 0.0], [0.0, 0.0, 1.0]]);
                let m = inverse.dot(&bbox_to(&g.bbox));
                let ends = zlc_figure::transform::bbox_transformed(&m, &[0.0, 0.0, 1.0, 1.0]);
                let length = if horizontal { (ends[2] - ends[0]) * 72.0 } else { (ends[3] - ends[1]) * 72.0 };
                let space = super::ticker::tick_space(length, self.major.labelsize, horizontal);
                let locs = super::ticker::auto_ticks(vmin, vmax, space);
                let formatter = super::ticker::ScalarFormatter::set_locs(&locs, view);
                self.labels = locs.iter().map(|&v| formatter.format(v)).collect();
                self.offset_string = formatter.offset_text();
                self.locs = locs;
            }
            TickPolicy::Smart(locator) => {
                let before = locator.ladder.drawn_pt;
                let locs = if vmax == vmin { Vec::new() }
                    else { locator.tick_values(vmin, vmax, geometry, &scale, serial, measure) };
                let labels = locs.iter().map(|&v| locator.ladder.text_for(v)).collect();
                let offset = locator.offset_text(horizontal);
                let after = locator.ladder.drawn_pt;
                self.locs = locs;
                self.labels = labels;
                self.offset_string = offset;
                self.apply_drawn_size(before, after);
            }
            TickPolicy::Named { ticks, texts } => {
                self.locs = ticks.clone();
                self.labels = (0..ticks.len()).map(|i| texts.get(i).cloned().unwrap_or_default()).collect();
                self.offset_string = String::new();
            }
            TickPolicy::Declared { locator } => {
                let before = locator.ladder.drawn_pt;
                let locs = if vmax == vmin { Vec::new() }
                    else { locator.with(|l| l.tick_values(vmin, vmax, geometry, &scale, serial, measure)) };
                let labels = locs.iter().map(|&v| locator.ladder.text_for(v)).collect();
                let after = locator.ladder.drawn_pt;
                self.locs = locs;
                self.labels = labels;
                self.offset_string = String::new();
                self.apply_drawn_size(before, after);
            }
            TickPolicy::PulseTime { nbins, factor } => {
                let locs = super::ticker::max_n_ticks(vmin, vmax, *nbins, true);
                let factor = *factor;
                self.labels = locs.iter().map(|&v| if v < 0.0 { String::new() } else { crate::pyfmt::format_g(v * factor, 4) }).collect();
                self.locs = locs;
                self.offset_string = String::new();
            }
        }
        let count = self.locs.len();
        self.tick_objects(count);
        self.pool_ticks();
        self.minor_locs = self.minor_ticks(view);
    }

    /// Ticks placed by another axis' answer to the same question (`_update_ticks`).
    pub fn set_ticks(&mut self, locs: Vec<f64>, labels: Vec<String>) {
        self.locs = locs;
        self.labels = labels;
        self.pool_ticks();
    }

    /// `majorTicks` take the new ticks: the first `len(locs)` move, the rest stay.
    fn pool_ticks(&mut self) {
        for (index, tick) in self.locs.iter().copied().zip(self.labels.iter().cloned()).enumerate() {
            match self.tick_pool.get_mut(index) {
                Some(slot) => *slot = tick,
                None => self.tick_pool.push(tick),
            }
        }
    }

    /// Forget the memo so the next `update_ticks` asks the ladder again.
    pub fn forget_tick_memo(&mut self) {
        match &mut self.policy {
            TickPolicy::Smart(locator) => locator.forget_memo(),
            TickPolicy::Declared { locator } => locator.memo = None,
            _ => {}
        }
    }

    pub fn forget_settled_step(&mut self) {
        if let TickPolicy::Smart(locator) = &mut self.policy { locator.forget_settled_step(); }
    }

    /// The indices (into `locs`) of the ticks inside the view interval.
    pub fn tick_indices_in_view(&self, view: (f64, f64)) -> Vec<usize> {
        let (mut low, mut high) = view;
        if low > high { std::mem::swap(&mut low, &mut high); }
        (0..self.locs.len()).filter(|&i| zlc_figure::axis::interval_contains_close((low, high), self.locs[i])).collect()
    }

    /// The ticks `_update_ticks` returns: those inside the view interval.
    /// Each as (tick object index, location, label).
    pub fn ticks_in_view(&self, view: (f64, f64)) -> Vec<(usize, f64, String)> {
        self.locs.iter().zip(&self.labels).enumerate().filter(|(_, (loc, _))| self.in_view(view, **loc))
            .map(|(index, (loc, label))| (index, *loc, label.clone())).collect()
    }

    /// Whether `_update_ticks` draws a tick at `loc`: inside the view, where the scale draws both.
    pub fn in_view(&self, view: (f64, f64), loc: f64) -> bool {
        let (mut low, mut high) = view;
        if low > high { std::mem::swap(&mut low, &mut high); }
        zlc_figure::axis::interval_contains_close((self.space(low), self.space(high)), self.space(loc))
    }
}

/// A Line2D child.
#[derive(Clone, Debug)]
pub struct LineArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: Option<f64>,
    pub color: String,
    pub linewidth: f64,
    pub linestyle: String,
    pub marker: String,
    pub markersize: f64,
    pub markeredgewidth: f64,
    pub markevery: Option<Vec<i64>>,
    /// Explicit `(markerfacecolor, markeredgecolor)`; None: the line colour.
    pub marker_colors: Option<(String, String)>,
    pub label: String,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
}

/// zlc's `_SegmentBuffered`: one closed error-bar glyph per segment.
#[derive(Clone, Debug)]
pub struct SegmentArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: f64,
    /// `to_rgba(colour, alpha)`.
    pub facecolor: [f64; 4],
    pub linewidth: f64,
    pub capsize: f64,
    /// (x, low, high) per bar, data coordinates.
    pub segments: Vec<[f64; 3]>,
}

/// A PolyCollection of fixed vertices (a histogram's bars).
#[derive(Clone, Debug)]
pub struct PolyArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: Option<f64>,
    pub facecolor: [f64; 4],
    pub verts: Vec<Vec<[f64; 2]>>,
}

/// zlc's `_BarBuffered`: one histogram group's bars, held as edges and tops
/// (a PolyCollection of closed quads from zero, on transData).
#[derive(Clone, Debug)]
pub struct BarArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: f64,
    /// The face colour as set (`set_facecolor`); drawn at `alpha`.
    pub color: String,
    pub linewidth: f64,
    pub label: String,
    /// `edges` is one longer than `counts`.
    pub edges: std::sync::Arc<Vec<f64>>,
    pub counts: std::sync::Arc<Vec<f64>>,
    /// Set while a compose builds its IR for bars the kernel paints: their
    /// paths would never be drawn, so none are made (`_BarBuffered` makes
    /// them only when something asks).
    pub kernel_painted: bool,
}

impl BarArtist {
    /// `to_rgba(color, alpha)`: the one face colour.
    pub fn facecolor(&self) -> [f64; 4] {
        let mut rgba = color::rgba(&self.color);
        rgba[3] = self.alpha;
        rgba
    }
}

/// Which transform a text child stands on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TextFrame {
    Data,
    Axes,
    /// transAxes + a ScaledTranslation of (dx, dy) inches.
    AxesOffset(f64, f64),
    /// `axes.annotate(text, xy, textcoords="offset points")`: the text's
    /// position is its offset in points from the data point `xy`; its
    /// transform is what the last draw or measure set (`update_positions`):
    /// (dpi/72, anchor pixel), identity before any.
    Annotation(f64, f64, Option<[f64; 3]>),
}

#[derive(Clone, Debug)]
pub struct TextArtist {
    pub visible: bool,
    pub zorder: f64,
    pub x: f64,
    pub y: f64,
    pub text: String,
    pub color: String,
    pub size: f64,
    pub ha: HAlign,
    pub va: VAlign,
    pub ma: Option<HAlign>,
    pub clip_on: bool,
    pub frame: TextFrame,
    /// The font family when the text was made outside the style (rc defaults).
    pub family: Option<Vec<String>>,
}

/// `axes.scatter(...)`: a PathCollection of one marker at data offsets.
#[derive(Clone, Debug)]
pub struct ScatterArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: Option<f64>,
    /// `"o"` or `"s"`.
    pub marker: &'static str,
    pub offsets: Vec<[f64; 2]>,
    /// Marker areas in points squared.
    pub sizes: Vec<f64>,
    pub facecolors: Vec<[f64; 4]>,
    /// Empty: `edgecolors="none"`.
    pub edgecolors: Vec<[f64; 4]>,
    pub linewidths: Vec<f64>,
}

/// A `patches.Ellipse` in data coordinates, unfilled (added with `add_artist`).
#[derive(Clone, Debug)]
pub struct EllipseArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: Option<f64>,
    pub center: [f64; 2],
    pub width: f64,
    pub height: f64,
    pub edgecolor: String,
    pub linewidth: f64,
}

/// A Line2D with what [`LineArtist`] does not say: an explicit dash sequence
/// (`set_linestyle((offset, seq))`), its solid cap style, or zlc's pulse
/// `RepeatBracketSide`, whose vertices its draw computes.
#[derive(Clone, Debug)]
pub struct StyledLineArtist {
    pub line: LineArtist,
    /// (offset, on-off sequence) in points, before the linewidth scales it.
    pub dashes: Option<(f64, Vec<f64>)>,
    pub solid_capstyle: ir::CapStyle,
    pub bracket: Option<ir::Bracket>,
}

/// A `patches.Rectangle` on transData (`add_patch`).
#[derive(Clone, Debug)]
pub struct RectangleArtist {
    pub visible: bool,
    pub zorder: f64,
    pub alpha: Option<f64>,
    pub xy: (f64, f64),
    pub width: f64,
    pub height: f64,
    pub facecolor: String,
    /// Set explicitly, so it is also the hatch colour.
    pub edgecolor: String,
    pub linewidth: f64,
    pub hatch: Option<String>,
}

/// `set_bbox(dict(boxstyle="round|circle,pad=...", facecolor=..., edgecolor="none"))`.
#[derive(Clone, Debug)]
pub struct Badge { pub circle: bool, pub pad: f64, pub facecolor: String }

/// A text with what [`TextArtist`] does not carry: its alpha, zlc's pulse
/// `SpanLabel` (printed only where its span holds it), a badge, or the data
/// point of `annotate(..., textcoords="offset points")` (whose `x, y` are
/// then the offset in points, and which no clip box clips).
#[derive(Clone, Debug)]
pub struct RichTextArtist {
    pub text: TextArtist,
    pub alpha: Option<f64>,
    pub span: Option<ir::SpanLabel>,
    pub badge: Option<Badge>,
    pub anchor: Option<(f64, f64)>,
}

#[derive(Clone, Debug)]
pub enum Child {
    Line(LineArtist),
    Segments(SegmentArtist),
    Poly(PolyArtist),
    Text(TextArtist),
    Scatter(ScatterArtist),
    Ellipse(EllipseArtist),
    StyledLine(StyledLineArtist),
    Rectangle(RectangleArtist),
    RichText(RichTextArtist),
    Bars(BarArtist),
    /// A line in (axes x, data y): `axhline`.
    AxLine(LineArtist),
    Image(Box<super::image::ImageArtist>),
    Mesh(Box<super::image::MeshArtist>),
    Rect(super::image::RectArtist),
    LineCollection(super::image::LineCollectionArtist),
    Polyline(super::image::PolylineArtist),
    Ellipses(Box<super::image::EllipsesArtist>),
    Union(Box<super::image::UnionArtist>),
}

impl Child {
    pub fn visible(&self) -> bool {
        match self {
            Child::Line(a) | Child::AxLine(a) => a.visible, Child::Segments(a) => a.visible, Child::Poly(a) => a.visible,
            Child::Text(a) => a.visible, Child::Scatter(a) => a.visible, Child::Ellipse(a) => a.visible,
            Child::StyledLine(a) => a.line.visible, Child::Rectangle(a) => a.visible, Child::RichText(a) => a.text.visible,
            Child::Bars(a) => a.visible,
            Child::Image(a) => a.visible, Child::Mesh(a) => a.visible, Child::Rect(a) => a.visible,
            Child::LineCollection(a) => a.visible, Child::Polyline(a) => a.visible, Child::Ellipses(a) => a.visible,
            Child::Union(a) => a.visible,
        }
    }

    pub fn set_visible(&mut self, visible: bool) {
        match self {
            Child::Line(a) | Child::AxLine(a) => a.visible = visible, Child::Segments(a) => a.visible = visible,
            Child::Poly(a) => a.visible = visible, Child::Text(a) => a.visible = visible, Child::Scatter(a) => a.visible = visible,
            Child::StyledLine(a) => a.line.visible = visible, Child::Rectangle(a) => a.visible = visible,
            Child::RichText(a) => a.text.visible = visible, Child::Bars(a) => a.visible = visible,
            Child::Ellipse(a) => a.visible = visible, Child::Image(a) => a.visible = visible, Child::Mesh(a) => a.visible = visible,
            Child::Rect(a) => a.visible = visible, Child::LineCollection(a) => a.visible = visible, Child::Polyline(a) => a.visible = visible,
            Child::Ellipses(a) => a.visible = visible, Child::Union(a) => a.visible = visible,
        }
    }

    pub fn zorder(&self) -> f64 {
        match self {
            Child::Line(a) | Child::AxLine(a) => a.zorder, Child::Segments(a) => a.zorder, Child::Poly(a) => a.zorder,
            Child::Text(a) => a.zorder, Child::Scatter(a) => a.zorder, Child::Ellipse(a) => a.zorder, Child::Image(_) => 0.0,
            Child::StyledLine(a) => a.line.zorder, Child::Rectangle(a) => a.zorder, Child::RichText(a) => a.text.zorder,
            Child::Bars(a) => a.zorder,
            Child::Mesh(a) => a.zorder, Child::Rect(a) => a.zorder, Child::LineCollection(a) => a.zorder, Child::Polyline(a) => a.zorder,
            Child::Ellipses(a) => a.zorder, Child::Union(a) => a.zorder,
        }
    }
}

/// One lane of a grid cell's tick marks (one Line2D carrying every mark of
/// one axis and side), or one gridline: a `_FacetChromeStroke` materialized.
#[derive(Clone, Debug)]
pub struct MarkArtist {
    /// Which axis' transform it stands on (`get_xaxis_transform` / `get_yaxis_transform`).
    pub x: bool,
    /// A gridline, clipped to the cell; tick marks are not clipped.
    pub grid: bool,
    pub zorder: f64,
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    /// The tick line it copies its style from (`update_from`).
    pub color: String,
    pub marker: &'static str,
    pub markersize: f64,
    pub markeredgewidth: f64,
}

/// One side of a grid cell's frame: a spine's path, frozen, as a PathPatch.
#[derive(Clone, Debug)]
pub struct FrameArtist { pub side: usize, pub zorder: f64, pub vertices: [[f64; 2]; 2] }

/// A grid cell's chrome stroke, appended after the cell's children (where
/// `_materialize_facet_chrome` adds it).
#[derive(Clone, Debug)]
pub enum Stroke { Marks(MarkArtist), Frame(FrameArtist) }

impl Stroke {
    pub fn zorder(&self) -> f64 { match self { Stroke::Marks(m) => m.zorder, Stroke::Frame(f) => f.zorder } }
}

pub const SPINES: [&str; 4] = ["left", "right", "bottom", "top"];

/// A stable handle on one child.
pub type ChildId = u64;

#[derive(Clone, Debug)]
pub struct TitleText { pub text: String, pub size: f64, pub color: String }

#[derive(Clone, Debug)]
pub struct AxesModel {
    pub role: String,
    pub gid: String,
    pub cell_index: Option<u32>,
    pub room: Option<Room>,
    /// `get_position(original=True)` as points (x0, y0, x1, y1), figure fractions.
    pub position: [f64; 4],
    /// The active position.
    pub active: [f64; 4],
    /// zlc's `apply_aspect` override is installed (the box is the renderer's).
    pub owned_box: bool,
    pub aspect: Option<f64>,
    /// `set_anchor`.
    pub anchor: &'static str,
    /// A colorbar's `_ColorbarAxesLocator` (extend 'neither').
    pub colorbar_locator: bool,
    /// A colorbar's outline spine (axes coordinates); the four spines are then hidden.
    pub outline: Option<Vec<[f64; 2]>>,
    pub visible: bool,
    pub axison: bool,
    pub frameon: bool,
    pub xlim: (f64, f64),
    pub ylim: (f64, f64),
    /// `get_autoscalex_on()` / `get_autoscaley_on()`: on until a limit is set.
    pub autoscale: [bool; 2],
    /// `_get_lines`' place in the property cycle: plots that took its colour.
    pub line_cycle: usize,
    /// `dataLim`'s x and y intervals; None while null.
    pub datalim: [Option<(f64, f64)>; 2],
    pub xaxis: AxisModel,
    pub yaxis: AxisModel,
    /// title, left, right.
    pub titles: [TitleText; 3],
    pub title_pad: f64,
    /// `_autotitlepos`: no `set_title(..., y=1.0)` pinned the titles yet.
    pub autotitlepos: bool,
    /// `set_axisbelow`: both axes' zorder; None: the rc's (`axes.axisbelow`).
    pub axis_zorder: Option<f64>,
    /// Spine visibility: left, right, bottom, top.
    pub spine_visible: [bool; 4],
    pub children: Vec<(ChildId, Child)>,
    /// A grid cell's chrome strokes (drawn after the children).
    pub strokes: Vec<Stroke>,
    next_child: ChildId,
}

/// `mtransforms.nonsingular(v0, v1, expander=0.05)` (increasing).
pub fn nonsingular(v0: f64, v1: f64) -> (f64, f64) { nonsingular_by(v0, v1, 0.05) }

/// `mtransforms.nonsingular(v0, v1, expander=expander)` (increasing).
pub fn nonsingular_by(v0: f64, v1: f64, expander: f64) -> (f64, f64) {
    let tiny = 1e-15;
    if !v0.is_finite() || !v1.is_finite() { return (-expander, expander); }
    let (mut vmin, mut vmax) = if v1 < v0 { (v1, v0) } else { (v0, v1) };
    let maxabsvalue = if vmax.abs() > vmin.abs() { vmax.abs() } else { vmin.abs() };
    if maxabsvalue < (1e6 / tiny) * f64::MIN_POSITIVE {
        vmin = -expander;
        vmax = expander;
    } else if vmax - vmin <= maxabsvalue * tiny {
        if vmax == 0.0 && vmin == 0.0 { vmin = -expander; vmax = expander; }
        else { vmin -= expander * vmin.abs(); vmax += expander * vmax.abs(); }
    }
    (vmin, vmax)
}

/// `Axis._set_lim(v0, v1)`: nonsingular, then the requested order again.
pub fn set_lim(v0: f64, v1: f64) -> (f64, f64) {
    let reverse = v0 > v1;
    let (low, high) = nonsingular(v0, v1);
    if reverse { (high, low) } else { (low, high) }
}

/// `Bbox.from_bounds(*bounds)` points of a `set_position(bounds)` box.
pub fn bounds_points(bounds: [f64; 4]) -> [f64; 4] {
    [bounds[0], bounds[1], bounds[0] + bounds[2], bounds[1] + bounds[3]]
}

impl AxesModel {
    /// `figure.add_axes(bounds)` under the style.
    pub fn new(rc: &Rc, bounds: [f64; 4]) -> AxesModel {
        let points = bounds_points(bounds);
        let title = |rc: &Rc| TitleText { text: String::new(), size: rc.axes_titlesize, color: rc.axes_titlecolor.clone() };
        AxesModel {
            role: String::new(),
            gid: String::new(),
            cell_index: None,
            room: None,
            position: points,
            active: points,
            owned_box: false,
            aspect: None,
            anchor: "C",
            colorbar_locator: false,
            outline: None,
            visible: true,
            axison: true,
            frameon: true,
            xlim: (0.0, 1.0),
            ylim: (0.0, 1.0),
            autoscale: [true, true],
            line_cycle: 0,
            datalim: [None, None],
            xaxis: AxisModel::new(rc, true),
            yaxis: AxisModel::new(rc, false),
            titles: [title(rc), title(rc), title(rc)],
            title_pad: rc.axes_titlepad,
            autotitlepos: true,
            axis_zorder: None,
            spine_visible: [true; 4],
            children: Vec::new(),
            strokes: Vec::new(),
            next_child: 0,
        }
    }

    /// `set_position(bounds)`: both the original and the active box.
    pub fn set_position(&mut self, bounds: [f64; 4]) {
        self.position = bounds_points(bounds);
        self.active = self.position;
    }

    /// `set_xlim(left, right)`: `Locator.nonsingular` keeps a degenerate span open.
    pub fn set_xlim(&mut self, left: f64, right: f64) {
        self.xlim = set_lim(left, right);
        self.autoscale[0] = false;
    }

    pub fn set_ylim(&mut self, bottom: f64, top: f64) {
        self.ylim = set_lim(bottom, top);
        self.autoscale[1] = false;
    }

    /// `update_datalim`: an artist added with data (`add_collection` with
    /// autolim, `add_line` of a line that has points) widens `dataLim`.
    pub fn update_datalim(&mut self, x: Option<(f64, f64)>, y: Option<(f64, f64)>) {
        for (index, range) in [(0usize, x), (1, y)] {
            let Some((lo, hi)) = range else { continue };
            self.datalim[index] = Some(match self.datalim[index] { Some((a, b)) => (a.min(lo), b.max(hi)), None => (lo, hi) });
        }
    }

    /// `axes.plot(...)` / `axes.scatter(...)` request an autoscale
    /// (`_request_autoscale_view`), which the next view read applies; it is
    /// applied here at once, which is the same view (`autoscale_view` without
    /// sticky edges): an autoscaling axis spans its `dataLim`, or -- with
    /// none -- the locator's default span unless its view was moved from
    /// (0, 1), made nonsingular and widened by the axes margin.
    pub fn request_autoscale(&mut self, margins: (f64, f64)) {
        for (index, margin) in [(0usize, margins.0), (1, margins.1)] {
            if !self.autoscale[index] { continue; }
            let view = if index == 0 { self.xlim } else { self.ylim };
            let (x0, x1) = match self.datalim[index] {
                Some(range) => range,
                None if view != (0.0, 1.0) => continue,
                None => (f64::NEG_INFINITY, f64::INFINITY),
            };
            let (x0, x1) = crate::render::mpl::ticker::nonsingular(x0, x1, 0.05, 1e-15);
            let delta = (x1 - x0) * margin;
            let limits = set_lim(x0 - delta, x1 + delta);
            if index == 0 { self.xlim = limits } else { self.ylim = limits }
        }
    }

    pub fn add(&mut self, child: Child) -> ChildId {
        let id = self.next_child;
        self.next_child += 1;
        self.children.push((id, child));
        id
    }

    pub fn remove(&mut self, id: ChildId) { self.children.retain(|(child, _)| *child != id); }

    pub fn child(&self, id: ChildId) -> Option<&Child> { self.children.iter().find(|(c, _)| *c == id).map(|(_, a)| a) }

    pub fn child_mut(&mut self, id: ChildId) -> Option<&mut Child> {
        self.children.iter_mut().find(|(c, _)| *c == id).map(|(_, a)| a)
    }

    pub fn line_mut(&mut self, id: ChildId) -> &mut LineArtist {
        match self.child_mut(id) { Some(Child::Line(line)) => line, _ => panic!("child {id} is not a line") }
    }

    pub fn line(&self, id: ChildId) -> &LineArtist {
        match self.child(id) { Some(Child::Line(line)) => line, _ => panic!("child {id} is not a line") }
    }

    /// The axes box in figure pixels.
    pub fn bbox_px(&self, figure: (f64, f64)) -> [f64; 4] {
        let p = self.active;
        [p[0] * figure.0, p[1] * figure.1, p[2] * figure.0, p[3] * figure.1]
    }

    /// `axes.bbox` as matplotlib computes it: TransformedBbox(position,
    /// transFigure), transFigure = BboxTransformTo(TransformedBbox(inches, dpi)).
    pub fn bbox_exact(&self, size_inches: [f64; 2], dpi: f64) -> [f64; 4] {
        let figure = transformed_points(&M3::IDENTITY.scale(dpi, dpi), &[0.0, 0.0, size_inches[0], size_inches[1]]);
        transformed_points(&bbox_to(&figure), &self.active)
    }

    /// `axes.transData`'s matrix: its affine part, which maps a point of the
    /// scales' space ([`AxesModel::scaled`]) -- the whole of it on linear scales.
    pub fn trans_data(&self, size_inches: [f64; 2], dpi: f64) -> M3 {
        let trans_axes = bbox_to(&self.bbox_exact(size_inches, dpi));
        let (x0, y0) = self.scaled(self.xlim.0, self.ylim.0);
        let (x1, y1) = self.scaled(self.xlim.1, self.ylim.1);
        let limits = bbox_from(&transformed_points(&M3::IDENTITY, &[x0, y0, x1, y1]));
        trans_axes.dot(&limits).dot(&M3::IDENTITY)
    }

    /// `transData.is_affine`: both axes on linear scales.
    pub fn is_linear(&self) -> bool {
        matches!((self.xaxis.scale_node(), self.yaxis.scale_node()), (super::nodes::ScaleNode::Linear, super::nodes::ScaleNode::Linear))
    }

    /// `transScale.transform_non_affine((x, y))`: the point where the scales draw it linearly.
    pub fn scaled(&self, x: f64, y: f64) -> (f64, f64) { (self.xaxis.space(x), self.yaxis.space(y)) }

    /// `transData.transform((x, y))`: display pixels (bottom-left origin).
    pub fn to_display(&self, size_inches: [f64; 2], dpi: f64, x: f64, y: f64) -> (f64, f64) {
        let (sx, sy) = self.scaled(x, y);
        self.trans_data(size_inches, dpi).apply(sx, sy)
    }

    /// `axes.set_yscale(...)` / `set_xscale(...)` (`Axis._set_axes_scale`):
    /// the axis takes the scale and its default locators.
    pub fn set_scale(&mut self, x: bool, log: bool) { self.axis_mut(x).set_scale(log); }

    /// `apply_aspect()`: Matplotlib shrinks the active box to the data aspect;
    /// an owned box (zlc's override) or an auto aspect keeps the original.
    pub fn apply_aspect(&mut self, size_inches: [f64; 2], dpi: f64) {
        let position = self.position;
        let Some(aspect) = self.aspect.filter(|_| !self.owned_box) else { self.active = position; return };
        let figure = transformed_points(&M3::IDENTITY.scale(dpi, dpi), &[0.0, 0.0, size_inches[0], size_inches[1]]);
        let fig_aspect = (figure[3] - figure[1]) / (figure[2] - figure[0]);
        let size = |axis: &AxisModel, (a, b): (f64, f64)| {
            let (low, high) = if a < b { (a, b) } else { (b, a) };
            (axis.space(high) - axis.space(low)).abs().max(1e-30)
        };
        let box_aspect = aspect * (size(&self.yaxis, self.ylim) / size(&self.xaxis, self.xlim));
        let (l, b, w, h) = bounds(&position);
        let mut height = w * box_aspect / fig_aspect;
        let width = if height <= h { w } else { height = h; h * fig_aspect / box_aspect };
        let shrunk = [position[0], position[1], position[0] + width, position[1] + height];
        let (_, _, sw, sh) = bounds(&shrunk);
        let (cx, cy) = super::super::geometry::anchor_fractions(self.anchor);
        let (dx, dy) = ((l + cx * (w - sw)) - shrunk[0], (b + cy * (h - sh)) - shrunk[1]);
        self.active = [shrunk[0] + dx, shrunk[1] + dy, shrunk[2] + dx, shrunk[3] + dy];
    }

    /// `axes.plot` without a colour: the property cycle's next one.
    pub fn next_line_color(&mut self, cycle: &[&str]) -> String {
        self.line_cycle += 1;
        cycle[(self.line_cycle - 1) % cycle.len()].to_owned()
    }

    pub fn axis(&self, x: bool) -> &AxisModel { if x { &self.xaxis } else { &self.yaxis } }
    pub fn axis_mut(&mut self, x: bool) -> &mut AxisModel { if x { &mut self.xaxis } else { &mut self.yaxis } }
    pub fn view(&self, x: bool) -> (f64, f64) { if x { self.xlim } else { self.ylim } }
}

/// Everything the emission needs beside the model.
pub struct EmitContext<'a> {
    pub rc: &'a Rc,
    /// Image planes carry their data (an export); a live figure's are painted by the kernel.
    pub materialize: bool,
    pub figure: &'a FigureNodes,
    pub dpi: f64,
    pub size_inches: [f64; 2],
}

fn tick_line(rc: &Rc, params: &TickParams, marker: &str, visible: bool, xy: [f64; 2], transform: NodeId) -> ir::Line2D {
    let spec = LineSpec {
        color: &params.color, linewidth: rc.lines_linewidth, linestyle: "None", alpha: None, marker,
        markersize: params.size, markeredgewidth: params.width, markeredgecolor: Some(&params.color), markerfacecolor: None,
        zorder: 0.0,
    };
    let mut line = artist::line(rc, &spec, vec![xy], transform);
    line.artist.visible = visible;
    line
}

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

impl AxesModel {
    /// Where an inward x axis' label at `loc` hangs: from an end of the view, inside it.
    fn inward_alignment(&self, loc: f64) -> HAlign {
        let (low, high) = if self.xlim.0 <= self.xlim.1 { self.xlim } else { (self.xlim.1, self.xlim.0) };
        let tolerance = 1e-9 * (high - low).abs();
        if (loc - low).abs() <= tolerance { HAlign::Left } else if (loc - high).abs() <= tolerance { HAlign::Right } else { HAlign::Center }
    }

    /// One major or minor tick of `axis` at `loc` with `label`; a major one is
    /// tick object `index`, with its own properties over the parameters.
    fn tick_template(&self, ctx: &EmitContext<'_>, nodes: &mut Nodes, an: &AxesNodes, axis: &AxisModel, major: bool,
                     index: usize, loc: f64, label: &str) -> ir::TickTemplate {
        let rc = ctx.rc;
        let p = if major { &axis.major } else { &axis.minor };
        let own = if major { axis.ticks.get(index).cloned().unwrap_or_default() } else { TickOverride::default() };
        // `Tick.__init__`: a major tick's lines stand a hair above Line2D's zorder.
        let zorder = if major { 2.0 + 0.01 } else { 2.0 };
        let (m1, m2) = tick_markers(axis.x, &p.tickdir);
        let line_transform = if axis.x { an.xaxis } else { an.yaxis };
        let (t1_xy, t2_xy, grid_xy) = if axis.x {
            ([loc, 0.0], [loc, 1.0], vec![[loc, 0.0], [loc, 1.0]])
        } else {
            ([0.0, loc], [1.0, loc], vec![[0.0, loc], [1.0, loc]])
        };
        let mut tick1 = tick_line(rc, p, m1, p.tick1on, t1_xy, line_transform);
        let mut tick2 = tick_line(rc, p, m2, p.tick2on, t2_xy, line_transform);
        tick1.artist.zorder = zorder;
        tick2.artist.zorder = zorder;
        if let Some(mark) = &own.mark_color {
            tick1.markeredgecolor = artist::colour(mark);
            tick1.mec_auto = false;
        }
        let grid_spec = LineSpec {
            color: p.grid_color.as_deref().unwrap_or(&rc.grid_color), linewidth: p.grid_linewidth.unwrap_or(rc.grid_linewidth),
            linestyle: &rc.grid_linestyle, alpha: Some(rc.grid_alpha),
            marker: "", markersize: rc.lines_markersize, markeredgewidth: rc.lines_markeredgewidth, markeredgecolor: None,
            markerfacecolor: None,
            zorder,
        };
        let mut grid = artist::line(rc, &grid_spec, grid_xy, line_transform);
        grid.artist.visible = p.grid_on;
        grid.artist.zorder = 2.0;
        grid.artist.clip_box = Some(an.patch_clip);
        ir::TickTemplate {
            cls: if axis.x { "XTick".into() } else { "YTick".into() },
            visible: true,
            size: p.size,
            tickdir: p.tickdir.clone(),
            tick1line: tick1,
            tick2line: tick2,
            gridline: grid,
            label1: {
                let mut label1 = Self::tick_label(ctx, nodes, an, axis, major, Some(index), loc, label, 1);
                if major && axis.x && axis.inward_ends && !axis.labels_centred() { label1.ha = self.inward_alignment(loc); }
                label1
            },
            label2: Self::tick_label(ctx, nodes, an, axis, major, Some(index), loc, label, 2),
        }
    }

    /// The drawn major ticks: the first one's template, and every one's when
    /// their own properties differ (`templates`) or their label transforms are
    /// distinct objects (a `tick_params` that moved the labels re-made them).
    fn tick_templates(&self, ctx: &EmitContext<'_>, nodes: &mut Nodes, an: &AxesNodes, axis: &AxisModel)
                      -> (Option<ir::TickTemplate>, Option<Vec<ir::TickTemplate>>) {
        if axis.locs.is_empty() { return (None, None); }
        let own = |index: usize| {
            let tick = axis.ticks.get(index).cloned().unwrap_or_default();
            let bits = |spec: &str| color::rgba(spec).map(f64::to_bits);
            ([axis.label_size(index, 1).to_bits(), axis.label_size(index, 2).to_bits()],
             tick.label_color.clone().map(|c| bits(c.as_deref().unwrap_or(&axis.major.labelcolor))),
             bits(tick.mark_color.as_deref().unwrap_or(&axis.major.color)))
        };
        let first = own(0);
        let alike = (1..axis.locs.len()).all(|index| own(index) == first);
        let distinct = (axis.locs.len() >= 2 && axis.distinct_ticks >= 2) || (axis.x && axis.inward_ends);
        let template = self.tick_template(ctx, nodes, an, axis, true, 0, axis.locs[0], &axis.labels[0]);
        if alike && !distinct { return (Some(template), None); }
        let all = (0..axis.locs.len()).map(|index| self.tick_template(ctx, nodes, an, axis, true, index, axis.locs[index], &axis.labels[index])).collect();
        (Some(template), Some(all))
    }

    /// A major or minor tick's `label1` or `label2` (`side`) at `loc`, as `Tick` places it;
    /// major tick object `index` draws its own label size and colour over the parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn tick_label(ctx: &EmitContext<'_>, nodes: &mut Nodes, an: &AxesNodes, axis: &AxisModel, major: bool, index: Option<usize>,
                      loc: f64, label: &str, side: u8) -> ir::Text {
        let p = if major { &axis.major } else { &axis.minor };
        let own = index.filter(|_| major).and_then(|i| axis.ticks.get(i)).cloned().unwrap_or_default();
        let pad = p.label_pad() * if side == 1 { -1.0 } else { 1.0 };
        let (transform, xy, ha, va) = if axis.x {
            (nodes.offset(an.xaxis, ctx.figure, 0.0, pad / 72.0), (loc, if side == 1 { 0.0 } else { 1.0 }), HAlign::Center,
             if side == 1 { VAlign::Top } else { VAlign::Bottom })
        } else {
            (nodes.offset(an.yaxis, ctx.figure, pad / 72.0, 0.0), (if side == 1 { 0.0 } else { 1.0 }, loc),
             if side == 1 { HAlign::Right } else { HAlign::Left }, VAlign::CenterBaseline)
        };
        let side = side as usize;
        let mut t = artist::text(ctx.rc, xy.0, xy.1, label, own.label_size[side - 1].unwrap_or(p.labelsize), transform);
        t.color = color::rgba(own.label_color[side - 1].as_deref().unwrap_or(&p.labelcolor));
        t.ha = ha;
        t.va = va;
        t.artist.visible = if side == 1 { p.label1on } else { p.label2on };
        t
    }

    /// The axis' offset text as it stands (`axis.offsetText`).
    pub fn offset_text_ir(ctx: &EmitContext<'_>, nodes: &mut Nodes, an: &AxesNodes, axis: &AxisModel) -> ir::Text {
        let rc = ctx.rc;
        let mut offset = match &axis.offset_pin {
            Some(pin) => {
                let mut t = artist::text(rc, pin.xy.0, pin.xy.1, &axis.offset_text, axis.offset_size, ctx.figure.trans_figure);
                t.ha = pin.ha;
                t.va = pin.va;
                t.artist.clip_on = false;
                t
            }
            None => {
                let transform = nodes.blended(an.trans_axes, ctx.figure.identity, false);
                let (xy, ha, va) = if axis.x {
                    ((1.0, 0.0), HAlign::Right, VAlign::Top)
                } else if axis.label_position == "right" {
                    // `set_ticks_position('right')` moves the offset with the ticks.
                    ((1.0, 0.5), HAlign::Right, VAlign::Baseline)
                } else {
                    ((0.0, 0.5), HAlign::Left, VAlign::Baseline)
                };
                let xy = axis.drawn_offset_xy.unwrap_or(xy);
                let mut t = artist::text(rc, xy.0, xy.1, &axis.offset_text, axis.offset_size, transform);
                t.ha = ha;
                t.va = va;
                t
            }
        };
        offset.color = color::rgba(&axis.offset_color);
        offset.artist.visible = axis.offset_visible;
        offset
    }

    fn axis_ir(&self, ctx: &EmitContext<'_>, nodes: &mut Nodes, an: &AxesNodes, axis: &AxisModel, drawn: bool) -> ir::Axis {
        let rc = ctx.rc;
        let (label_transform, label_xy, label_va, label_rotation) = if axis.x {
            (nodes.blended(an.trans_axes, ctx.figure.identity, false), (0.5, 0.0), VAlign::Top, 0.0)
        } else {
            (nodes.blended(ctx.figure.identity, an.trans_axes, false), (0.0, 0.5),
             if axis.label_position == "right" { VAlign::Top } else { VAlign::Bottom }, 90.0)
        };
        let label_xy = axis.drawn_label_xy.unwrap_or(label_xy);
        let mut label = artist::text(rc, label_xy.0, label_xy.1, &axis.label, axis.label_size, label_transform);
        label.color = color::rgba(&axis.label_color);
        label.ha = HAlign::Center;
        label.va = label_va;
        label.rotation = label_rotation;
        if !axis.x { label.rotation_mode = RotationMode::Anchor; }
        let offset = Self::offset_text_ir(ctx, nodes, an, axis);
        let major = if drawn {
            let (tick, per_tick) = self.tick_templates(ctx, nodes, an, axis);
            Some(ir::Ticks { locs: axis.locs.clone(), labels: axis.labels.clone(), tick, per_tick })
        } else { None };
        // A NullFormatter labels every minor tick with the empty string.
        let minor = if drawn {
            let tick = if axis.minor_locs.is_empty() { None }
                else { Some(self.tick_template(ctx, nodes, an, axis, false, 0, axis.minor_locs[0], "")) };
            Some(ir::Ticks { locs: axis.minor_locs.clone(), labels: vec![String::new(); axis.minor_locs.len()], tick, per_tick: None })
        } else { None };
        let in_view = |locs: &[f64]| locs.iter().filter(|&&loc| axis.in_view(self.view(axis.x), loc)).count();
        let (majors, minors) = if drawn { (in_view(&axis.locs), in_view(&axis.minor_locs)) } else { (0, 0) };
        ir::Axis {
            name: axis.name().into(),
            visible: true,
            zorder: self.axis_zorder.unwrap_or(rc.axis_zorder),
            label,
            label_position: axis.label_position.into(),
            labelpad: axis.labelpad,
            autolabelpos: true,
            offset_text: offset,
            offset_text_pinned: axis.offset_pin.is_some(),
            tick_position: if axis.x { Some("bottom".into()) } else { None },
            transform: if axis.x { an.x_scale } else { an.y_scale },
            drawn,
            major,
            minor,
            offset_string: if drawn { Some(axis.offset_string.clone()) } else { None },
            // `Spine.get_window_extent` pads by one drawn major and one drawn
            // minor tick (picked in set order; an axis' ticks pad alike), as
            // indices into the drawn ticks, majors first.
            spine_ticks: [(majors > 0).then_some(0), (minors > 0).then_some(majors)],
        }
    }

    fn spine(&self, ctx: &EmitContext<'_>, an: &AxesNodes, name: &str) -> ir::NamedSpine {
        let rc = ctx.rc;
        let (transform, vertices, axis) = match name {
            "left" => (an.yaxis, vec![[0.0, self.ylim.0], [0.0, self.ylim.1]], "y"),
            "right" => (an.yaxis, vec![[1.0, self.ylim.0], [1.0, self.ylim.1]], "y"),
            "bottom" => (an.xaxis, vec![[self.xlim.0, 0.0], [self.xlim.1, 0.0]], "x"),
            _ => (an.xaxis, vec![[self.xlim.0, 1.0], [self.xlim.1, 1.0]], "x"),
        };
        let mut patch = artist::rectangle([0.0; 4], color::rgba(&rc.axes_edgecolor), rc.axes_linewidth, true, transform);
        patch.artist.zorder = 2.5;
        patch.artist.visible = self.spine_visible[["left", "right", "bottom", "top"].iter().position(|side| *side == name).expect("spine name")];
        patch.capstyle = ir::CapStyle::Projecting;
        patch.shape = ir::PatchShape::Spine {
            spine_type: name.into(),
            path: artist::path(vertices, None, false),
            bounds: None,
            axis: Some(axis.into()),
        };
        ir::NamedSpine { name: name.into(), patch }
    }

    fn child_ir(&self, ctx: &EmitContext<'_>, nodes: &mut Nodes, an: &AxesNodes, index: usize, child: &Child,
                trans_data: &zlc_figure::transform::M3) -> Artist {
        let rc = ctx.rc;
        match child {
            Child::Line(l) => {
                let mut line = line_ir(ctx, an, index, l, &l.linestyle, l.x.iter().zip(&l.y).map(|(&x, &y)| [x, y]).collect());
                artist::recache(rc, &mut line, true);
                Artist::Line(line)
            }
            Child::Segments(s) => Artist::Collection(segments_ir(ctx, an, self, s, trans_data)),
            Child::Poly(p) => Artist::Collection(poly_ir(ctx, an, p)),
            Child::Bars(b) => Artist::Collection(bars_ir(ctx, an, b)),
            Child::Scatter(s) => Artist::Collection(scatter_ir(ctx, an, s)),
            Child::Ellipse(e) => Artist::Patch(ellipse_ir(an, e)),
            Child::Text(t) => Artist::Text(text_ir(ctx, nodes, an, index, t)),
            Child::StyledLine(s) => {
                let l = &s.line;
                // A bracket side's vertices are its draw's; until one, it has none.
                let xy = if s.bracket.is_some() { Vec::new() } else { l.x.iter().zip(&l.y).map(|(&x, &y)| [x, y]).collect() };
                let mut line = line_ir(ctx, an, index, l, if s.dashes.is_some() { "--" } else { &l.linestyle }, xy);
                if let Some((offset, sequence)) = &s.dashes {
                    // `_get_dash_pattern` (offset within one cycle), then `_scale_dashes`.
                    let total: f64 = sequence.iter().sum();
                    let offset = if total != 0.0 { offset.rem_euclid(total) } else { *offset };
                    line.dash_pattern = ir::Dash(offset * l.linewidth, Some(sequence.iter().map(|v| v * l.linewidth).collect()));
                }
                line.solid_capstyle = s.solid_capstyle;
                line.bracket = s.bracket.clone();
                artist::recache(rc, &mut line, true);
                Artist::Line(line)
            }
            Child::Rectangle(r) => Artist::Patch(rectangle_ir(ctx, an, r)),
            Child::RichText(t) => {
                let mut text = text_ir(ctx, nodes, an, index, &t.text);
                text.artist.alpha = t.alpha;
                text.span_label = t.span.clone();
                if let Some((x, y)) = t.anchor {
                    // `annotate` gives the text an identity transform (its draw
                    // places it from the anchor) and no clip box.
                    text.artist.transform = Some(ctx.figure.identity);
                    text.artist.clip_box = None;
                    text.annotation = Some(ir::AnnotationIr { xy: [x, y], xycoords: "data".into(), anncoords: "offset points".into(),
                                                              annotation_clip: None, axes: index });
                }
                if let Some(badge) = &t.badge {
                    // `set_bbox`: a FancyBboxPatch with the text's clip, placed by the text's draw.
                    let mut patch = artist::rectangle(color::rgba(&badge.facecolor), [0.0; 4], rc.patch_linewidth, true, ctx.figure.identity);
                    patch.artist.clip_on = text.artist.clip_on;
                    patch.artist.clip_box = text.artist.clip_box;
                    patch.hatch_linewidth = rc.ir.hatch_linewidth;
                    patch.shape = ir::PatchShape::FancyBbox {
                        boxstyle: ir::BoxStyle { name: if badge.circle { "Circle" } else { "Round" }.into(), pad: badge.pad, rounding_size: None },
                        mutation_aspect: 1.0,
                    };
                    text.bbox_patch = Some(Box::new(patch));
                }
                Artist::Text(text)
            }
            Child::AxLine(l) => {
                let spec = LineSpec {
                    color: &l.color, linewidth: l.linewidth, linestyle: &l.linestyle, alpha: l.alpha, marker: &l.marker,
                    markersize: l.markersize, markeredgewidth: l.markeredgewidth, markerfacecolor: None, markeredgecolor: None,
                    zorder: l.zorder,
                };
                let xy: Vec<[f64; 2]> = l.x.iter().zip(&l.y).map(|(&x, &y)| [x, y]).collect();
                let mut line = artist::line(rc, &spec, xy, an.yaxis);
                line.artist.visible = l.visible;
                line.artist.clip_box = Some(an.patch_clip);
                line.axes = Some(index);
                artist::recache(rc, &mut line, false);
                Artist::Line(line)
            }
            Child::Image(image) => super::image::image_ir(ctx, an, image),
            Child::Mesh(mesh) => super::image::mesh_ir(ctx, an, mesh),
            Child::Rect(rect) => super::image::rect_ir(ctx, an, rect),
            Child::LineCollection(lines) => super::image::line_collection_ir(ctx, an, lines),
            Child::Polyline(line) => super::image::polyline_ir(ctx, an, index, line),
            Child::Ellipses(rings) => super::image::ellipses_ir(ctx, nodes, an, index, self, rings),
            Child::Union(union) => super::image::union_ir(ctx, an, self, union),
        }
    }

    /// A chrome stroke's materialized artist (`_FacetChromeStroke.materialize`).
    fn stroke_ir(&self, ctx: &EmitContext<'_>, an: &AxesNodes, index: usize, stroke: &Stroke) -> Artist {
        let rc = ctx.rc;
        match stroke {
            Stroke::Marks(m) => {
                let transform = if m.x { an.xaxis } else { an.yaxis };
                let spec = if m.grid {
                    LineSpec {
                        color: &rc.grid_color, linewidth: rc.grid_linewidth, linestyle: &rc.grid_linestyle, alpha: Some(rc.grid_alpha),
                        marker: "", markersize: rc.lines_markersize, markeredgewidth: rc.lines_markeredgewidth,
                        markeredgecolor: None, markerfacecolor: None, zorder: m.zorder,
                    }
                } else {
                    LineSpec {
                        color: &m.color, linewidth: rc.lines_linewidth, linestyle: "None", alpha: None, marker: m.marker,
                        markersize: m.markersize, markeredgewidth: m.markeredgewidth, markeredgecolor: Some(&m.color), markerfacecolor: None,
                        zorder: m.zorder,
                    }
                };
                let xy: Vec<[f64; 2]> = m.xs.iter().zip(&m.ys).map(|(&x, &y)| [x, y]).collect();
                let mut line = artist::line(rc, &spec, xy, transform);
                if m.grid { line.artist.clip_box = Some(an.patch_clip); }
                line.axes = Some(index);
                artist::recache(rc, &mut line, false);
                Artist::Line(line)
            }
            Stroke::Frame(f) => {
                let transform = if f.side < 2 { an.yaxis } else { an.xaxis };
                let edge = color::rgba(&rc.axes_edgecolor);
                let mut patch = artist::rectangle([0.0; 4], edge, rc.axes_linewidth, true, transform);
                patch.artist.zorder = f.zorder;
                patch.capstyle = ir::CapStyle::Projecting;
                patch.shape = ir::PatchShape::Path { path: artist::path(f.vertices.to_vec(), None, false) };
                Artist::Patch(patch)
            }
        }
    }

    /// The IR of this axes (`figure.axes[index]`), its transform nodes and its
    /// title frame (`transAxes + titleOffsetTrans`).
    pub fn to_ir(&self, ctx: &EmitContext<'_>, nodes: &mut Nodes, index: usize, trans_data: &zlc_figure::transform::M3)
        -> (ir::Axes, AxesNodes, NodeId) {
        let rc = ctx.rc;
        let data = self.trans_data(ctx.size_inches, ctx.dpi);
        let axes_m = bbox_to(&self.bbox_exact(ctx.size_inches, ctx.dpi));
        // A non-affine transData equals no affine transform.
        let (x_scale, y_scale) = (self.xaxis.scale_node(), self.yaxis.scale_node());
        let linear = matches!((x_scale, y_scale), (super::nodes::ScaleNode::Linear, super::nodes::ScaleNode::Linear));
        let an = nodes.axes_scaled(ctx.figure, index, linear && data.0 == axes_m.0, x_scale, y_scale);
        let drawn = self.visible && self.axison;
        let xaxis = self.axis_ir(ctx, nodes, &an, &self.xaxis, drawn);
        let yaxis = self.axis_ir(ctx, nodes, &an, &self.yaxis, drawn);
        let mut spines: Vec<ir::NamedSpine> = ["left", "right", "bottom", "top"].iter().map(|name| self.spine(ctx, &an, name)).collect();
        if let Some(outline) = &self.outline {
            for spine in &mut spines { spine.patch.artist.visible = false; }
            spines.push(super::image::outline_spine(ctx, &an, outline));
        }
        let title_transform = nodes.offset(an.trans_axes, ctx.figure, 0.0, self.title_pad / 72.0);
        let titles = self.titles.iter().enumerate().map(|(k, title)| {
            let (x, ha) = match k { 0 => (0.5, HAlign::Center), 1 => (0.0, HAlign::Left), _ => (1.0, HAlign::Right) };
            let mut t = artist::text(rc, x, 1.0, &title.text, title.size, title_transform);
            if title.color != "auto" { t.color = color::rgba(&title.color); }
            t.ha = ha;
            t.va = VAlign::Baseline;
            t
        }).collect();
        let children = self.children.iter().map(|(_, child)| self.child_ir(ctx, nodes, &an, index, child, trans_data))
            .chain(self.strokes.iter().map(|stroke| self.stroke_ir(ctx, &an, index, stroke))).collect();
        let patch = artist::rectangle(color::rgba(&rc.axes_facecolor), [0.0; 4], 0.0, true, an.trans_axes);
        let (x0, x1) = self.xlim;
        let (y0, y1) = self.ylim;
        (ir::Axes {
            index,
            visible: self.visible,
            zorder: 0.0,
            position_original: self.position,
            position_active: self.active,
            aspect: match self.aspect { None => ir::Aspect::Auto("auto".into()), Some(v) => ir::Aspect::Value(v) },
            adjustable: "box".into(),
            anchor: ir::Anchor::Named(self.anchor.into()),
            box_aspect: None,
            apply_aspect: if self.owned_box { "position".into() } else { "matplotlib".into() },
            axes_locator: self.colorbar_locator.then(|| "colorbar_neither".into()),
            view_lim: [x0, y0, x1, y1],
            axison: self.axison,
            frameon: self.frameon,
            patch,
            spines,
            xaxis,
            yaxis,
            titles,
            autotitlepos: Some(self.autotitlepos),
            children,
            trans_data: an.trans_data,
            trans_axes: an.trans_axes,
            bbox: an.bbox,
        }, an, title_transform)
    }
}

/// `_SegmentBuffered`'s display-space glyphs: each bar through `transData`,
/// its scales' non-affine part (`model.scaled`) then `affine` (`get_affine()`).
fn segments_ir(ctx: &EmitContext<'_>, an: &AxesNodes, model: &AxesModel, s: &SegmentArtist, affine: &zlc_figure::transform::M3)
               -> ir::Collection {
    let dpi = ctx.dpi;
    let radius = (s.linewidth * dpi / 144.0).max(0.5);
    let points: Vec<(f64, f64, f64)> = s.segments.iter().map(|&[x, low, high]| {
        let (sx, s_low) = model.scaled(x, low);
        let (_, s_high) = model.scaled(x, high);
        let (px, p_low) = affine.apply(sx, s_low);
        let (_, p_high) = affine.apply(sx, s_high);
        (px, p_low.min(p_high), p_low.max(p_high))
    }).collect();
    let xs: Vec<f64> = points.iter().map(|p| p.0).collect();
    let cap = crate::render::curve::effective_cap_width(&xs, (s.capsize * dpi / 72.0).max(0.0));
    let paths = points.iter().map(|&(x, low, high)| {
        let vertices: Vec<[f64; 2]> = if cap <= 0.0 {
            vec![[x - radius, low], [x + radius, low], [x + radius, high], [x - radius, high]]
        } else {
            let widths = [-cap, cap, cap, radius, radius, cap, cap, -cap, -cap, -radius, -radius, -cap];
            let steps = if cap >= radius {
                [low - radius, low - radius, low + radius, low + radius, high - radius, high - radius, high + radius,
                 high + radius, high - radius, high - radius, low + radius, low + radius]
            } else {
                [low - radius, low - radius, low, low, high, high, high + radius, high + radius, high, high, low, low]
            };
            if cap >= radius && high - low <= 2.0 * radius {
                let short_w = [-cap, cap, cap, -cap, -cap, -cap, -cap, -cap, -cap, -cap, -cap, -cap];
                let short_s = [low - radius, low - radius, high + radius, high + radius, low - radius, low - radius,
                               low - radius, low - radius, low - radius, low - radius, low - radius, low - radius];
                (0..12).map(|k| [x + short_w[k], short_s[k]]).collect()
            } else {
                (0..12).map(|k| [x + widths[k], steps[k]]).collect()
            }
        };
        closed_path(vertices)
    }).collect();
    let mut artist = artist::props(s.zorder, Some(ctx.figure.identity));
    artist.visible = s.visible;
    artist.alpha = Some(s.alpha);
    artist.snap = Some(false);
    artist.clip_box = Some(an.patch_clip);
    ir::Collection {
        artist,
        cls: "_SegmentBuffered".into(),
        paths,
        offsets: vec![[0.0, 0.0]],
        offset_transform: ctx.figure.identity,
        facecolors: vec![s.facecolor],
        edgecolors: Vec::new(),
        linewidths: vec![s.linewidth],
        linestyles: vec![ir::Dash(0.0, None)],
        antialiaseds: vec![true],
        joinstyle: None,
        capstyle: None,
        mapped: false,
        sizes: Some(Vec::new()),
        factor: Some(1.0),
        transforms: None,
        ellipses: None,
    }
}

/// `PathCollection` of `axes.scatter`: the marker's path at every offset,
/// sized by area, through an identity transform.
fn scatter_ir(ctx: &EmitContext<'_>, an: &AxesNodes, s: &ScatterArtist) -> ir::Collection {
    let mut artist = artist::props(s.zorder, Some(ctx.figure.identity));
    artist.visible = s.visible;
    artist.alpha = s.alpha;
    artist.clip_box = Some(an.patch_clip);
    ir::Collection {
        artist,
        cls: "PathCollection".into(),
        paths: vec![artist::marker_path(s.marker)],
        offsets: s.offsets.clone(),
        offset_transform: an.trans_data,
        facecolors: s.facecolors.clone(),
        edgecolors: s.edgecolors.clone(),
        linewidths: s.linewidths.clone(),
        linestyles: vec![ir::Dash(0.0, None)],
        antialiaseds: vec![true],
        joinstyle: None,
        capstyle: None,
        mapped: false,
        sizes: Some(s.sizes.clone()),
        factor: Some(1.0),
        transforms: None,
        ellipses: None,
    }
}

/// A Line2D child on transData, before its recache.
fn line_ir(ctx: &EmitContext<'_>, an: &AxesNodes, index: usize, l: &LineArtist, linestyle: &str, xy: Vec<[f64; 2]>) -> ir::Line2D {
    let spec = LineSpec {
        color: &l.color, linewidth: l.linewidth, linestyle, alpha: l.alpha, marker: &l.marker,
        markersize: l.markersize, markeredgewidth: l.markeredgewidth,
        markerfacecolor: l.marker_colors.as_ref().map(|c| c.0.as_str()),
        markeredgecolor: l.marker_colors.as_ref().map(|c| c.1.as_str()), zorder: l.zorder,
    };
    let mut line = artist::line(ctx.rc, &spec, xy, an.trans_data);
    line.artist.visible = l.visible;
    line.artist.clip_box = Some(an.patch_clip);
    line.markevery = l.markevery.clone().map(ir::MarkEvery::Indices);
    line.axes = Some(index);
    line
}

/// A Text child on its frame, clipped by the axes patch.
fn text_ir(ctx: &EmitContext<'_>, nodes: &mut Nodes, an: &AxesNodes, index: usize, t: &TextArtist) -> ir::Text {
    let transform = match t.frame {
        TextFrame::Data => an.trans_data,
        TextFrame::Axes => an.trans_axes,
        TextFrame::AxesOffset(dx, dy) => nodes.offset(an.trans_axes, ctx.figure, dx, dy),
        TextFrame::Annotation(_, _, None) => ctx.figure.identity,
        TextFrame::Annotation(_, _, Some([s, px, py])) => nodes.affine([s, 0.0, px, 0.0, s, py]),
    };
    let mut text = artist::text(ctx.rc, t.x, t.y, &t.text, t.size, transform);
    if let TextFrame::Annotation(x, y, _) = t.frame {
        text.annotation = Some(ir::AnnotationIr { xy: [x, y], xycoords: "data".into(), anncoords: "offset points".into(),
                                                  annotation_clip: None, axes: index });
    }
    if let Some(family) = &t.family { text.font = artist::font_of(ctx.rc, t.size, family); }
    text.color = color::rgba(&t.color);
    text.ha = t.ha;
    text.va = t.va;
    text.ma = t.ma;
    text.artist.visible = t.visible;
    text.artist.zorder = t.zorder;
    text.artist.clip_on = t.clip_on;
    text.artist.clip_box = Some(an.patch_clip);
    text
}

/// `Rectangle(xy, width, height)` on transData with its colours, alpha and hatch.
fn rectangle_ir(ctx: &EmitContext<'_>, an: &AxesNodes, r: &RectangleArtist) -> ir::Patch {
    let face = color::to_rgba_alpha(&r.facecolor, r.alpha).expect("rectangle face colour");
    let edge = color::to_rgba_alpha(&r.edgecolor, r.alpha).expect("rectangle edge colour");
    let mut patch = artist::rectangle(face, edge, r.linewidth, true, an.trans_data);
    patch.artist.visible = r.visible;
    patch.artist.zorder = r.zorder;
    patch.artist.alpha = r.alpha;
    patch.artist.clip_box = Some(an.patch_clip);
    // `Path.hatch(pattern, 6.0)`: two vertices per line, six lines per stroke.
    patch.hatch = r.hatch.as_ref().map(|pattern| ir::Hatch {
        pattern: pattern.clone(), path_simplify: artist::simplify(ctx.rc, 2 * (6 * pattern.chars().count() + 1)),
    });
    patch.hatch_linewidth = ctx.rc.ir.hatch_linewidth;
    patch.shape = ir::PatchShape::Rectangle {
        x0: r.xy.0, y0: r.xy.1, width: r.width, height: r.height, angle: 0.0,
        rotation_point: ir::RotationPoint::Named("xy".into()),
        path_simplify: ir::Simplify { should_simplify: false, threshold: artist::SIMPLIFY_THRESHOLD },
    };
    patch
}

/// `Ellipse(center, width, height, facecolor="none", ...)` on transData.
fn ellipse_ir(an: &AxesNodes, e: &EllipseArtist) -> ir::Patch {
    let edge = color::to_rgba_alpha(&e.edgecolor, e.alpha).expect("ellipse edge colour");
    let mut patch = artist::rectangle([0.0; 4], edge, e.linewidth, true, an.trans_data);
    patch.artist.visible = e.visible;
    patch.artist.zorder = e.zorder;
    patch.artist.alpha = e.alpha;
    patch.artist.clip_box = Some(an.patch_clip);
    patch.shape = ir::PatchShape::Ellipse {
        center: e.center, width: e.width, height: e.height, angle: 0.0, aspect_ratio_correction: 1.0,
        // `Path.unit_circle()`: 26 Bezier vertices, never simplified.
        path_simplify: ir::Simplify { should_simplify: false, threshold: artist::SIMPLIFY_THRESHOLD },
    };
    patch
}

/// `Path(verts_pad, codes)` of `PolyCollection.set_verts(verts, closed=True)`.
pub fn closed_path(mut vertices: Vec<[f64; 2]>) -> ir::Path {
    let first = vertices[0];
    vertices.push(first);
    let n = vertices.len();
    let mut codes = vec![2u8; n];
    codes[0] = 1;
    codes[n - 1] = 79;
    artist::path(vertices, Some(codes), false)
}

fn poly_ir(ctx: &EmitContext<'_>, an: &AxesNodes, p: &PolyArtist) -> ir::Collection {
    let mut artist = artist::props(p.zorder, Some(an.trans_data));
    artist.visible = p.visible;
    artist.alpha = p.alpha;
    artist.clip_box = Some(an.patch_clip);
    ir::Collection {
        artist,
        cls: "PolyCollection".into(),
        paths: p.verts.iter().map(|v| closed_path(v.clone())).collect(),
        offsets: vec![[0.0, 0.0]],
        offset_transform: ctx.figure.identity,
        facecolors: vec![p.facecolor],
        edgecolors: Vec::new(),
        linewidths: vec![1.0],
        linestyles: vec![ir::Dash(0.0, None)],
        antialiaseds: vec![true],
        joinstyle: None,
        capstyle: None,
        mapped: false,
        sizes: Some(Vec::new()),
        factor: Some(1.0),
        transforms: None,
        ellipses: None,
    }
}

/// `_BarBuffered` as a draw materializes it: one closed quad per bin,
/// `(low, 0) (low, top) (high, top) (high, 0)` (`_histogram_vertices`).
fn bars_ir(ctx: &EmitContext<'_>, an: &AxesNodes, b: &BarArtist) -> ir::Collection {
    let mut artist = artist::props(b.zorder, Some(an.trans_data));
    artist.visible = b.visible;
    artist.alpha = Some(b.alpha);
    artist.clip_box = Some(an.patch_clip);
    let paths = if b.kernel_painted { Vec::new() } else {
        b.counts.iter().enumerate().map(|(i, &top)| {
            let (low, high) = (b.edges[i], b.edges[i + 1]);
            closed_path(vec![[low, 0.0], [low, top], [high, top], [high, 0.0]])
        }).collect()
    };
    ir::Collection {
        artist,
        cls: "_BarBuffered".into(),
        paths,
        offsets: vec![[0.0, 0.0]],
        offset_transform: ctx.figure.identity,
        facecolors: vec![b.facecolor()],
        edgecolors: Vec::new(),
        linewidths: vec![b.linewidth],
        linestyles: vec![ir::Dash(0.0, None)],
        antialiaseds: vec![true],
        joinstyle: None,
        capstyle: None,
        mapped: false,
        sizes: Some(Vec::new()),
        factor: Some(1.0),
        transforms: None,
        ellipses: None,
    }
}
