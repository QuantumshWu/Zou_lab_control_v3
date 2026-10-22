//! The selector scene (`_selector_scene`): what every committed or
//! transient selector and the image colour limits draw, as pure data --
//! lines, markers and labels with their formatted numbers, in data
//! coordinates of the axes they name -- for the renderer to turn into
//! artists.  Committed and transient selectors are one scene.

use crate::configuration::error::Result;
use crate::configuration::selectors::{NumericRange, SelectorKind, SelectorState, SelectorValue};
use crate::render::scale::AxisScale;

use super::labels::{selector_number, selector_precision};
use super::pymath::{midpoint, placed, py_max};

pub type Rgba = (f64, f64, f64, f64);
pub type Point = (f64, f64);

/// What a scene group draws: one selector kind, or the image colour limits.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum SceneKind { Selector(SelectorKind), ColorLimits }

impl SceneKind {
    pub fn value(self) -> &'static str { match self { SceneKind::Selector(kind) => kind.value(), SceneKind::ColorLimits => "color_limits" } }
}

/// The effective image colour scale beside an Image plot; a `candidate` is
/// the transient drag state (`ColorLimitCandidate`) and carries a label.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ColorLimitState { pub value: NumericRange, pub candidate: bool }

/// The axes a primitive is drawn on.
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct SelectorTarget { pub role: String, pub cell_index: Option<u64> }

#[derive(Clone, Debug, PartialEq)]
pub struct SelectorLine {
    pub key: &'static str,
    pub target: SelectorTarget,
    pub points: Vec<Point>,
    pub color: Rgba,
    pub width_pt: f64,
    pub zorder: f64,
    pub linestyle: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectorMarkers {
    pub key: &'static str,
    pub target: SelectorTarget,
    pub points: Vec<Point>,
    pub shape: &'static str,
    pub size_pt: f64,
    pub facecolor: Rgba,
    pub edgecolor: Option<Rgba>,
    pub edge_width_pt: f64,
    pub zorder: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectorText {
    pub key: &'static str,
    pub target: SelectorTarget,
    pub text: String,
    /// Axes fractions.
    pub position: Point,
    pub horizontal_alignment: &'static str,
    pub color: Rgba,
    pub font_family: String,
    pub font_size_pt: f64,
    pub zorder: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum SelectorPrimitive { Line(SelectorLine), Markers(SelectorMarkers), Text(SelectorText) }

#[derive(Clone, Debug, PartialEq, Default)]
pub struct SelectorScene { pub groups: Vec<(SceneKind, Vec<SelectorPrimitive>)> }

impl SelectorScene {
    pub fn selector_kinds(&self) -> Vec<SceneKind> { self.groups.iter().map(|(kind, _)| *kind).collect() }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectorSceneStyle {
    pub line_width_pt: f64,
    pub line_alpha: f64,
    pub handle_size_pt: f64,
    pub handle_edge_width_pt: f64,
    pub crosshair_size_pt: f64,
    pub annotation_pt: f64,
    pub side_distribution_width_pt: f64,
    pub selector_zorder: f64,
    pub font_family: String,
    pub label_inset_fraction: f64,
    pub label_line_fraction: f64,
    pub threshold_width_pt: f64,
    pub threshold_linestyle: String,
}

/// The image value under a crosshair: a number or the text already made.
#[derive(Clone, Debug, PartialEq)]
pub enum SampleValue { Number(f64), Text(String) }

/// Everything one selector kind (or the colour limits) is drawn with.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectorItemContext {
    pub kind: SceneKind,
    pub target: SelectorTarget,
    pub label_target: SelectorTarget,
    pub x_limits: Point,
    pub y_limits: Point,
    pub selector_rgba: Rgba,
    pub color_limit_rgba: (Rgba, Rgba),
    pub threshold_rgba: Rgba,
    pub threshold_label_rgba: Rgba,
    pub threshold_uses_x: bool,
    pub threshold_text: String,
    /// How each axis divides the space between its limits; where a handle
    /// belongs is a screen question.
    pub x_scale: AxisScale,
    pub y_scale: AxisScale,
    pub x_label_factor: f64,
    pub sample_value: Option<SampleValue>,
    pub sample_span: Option<f64>,
    pub sample_target: Option<SelectorTarget>,
    pub sample_x_limits: Option<Point>,
    pub sample_rgba: Option<Rgba>,
}

/// The scene owner a renderer holds per presentation.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectorSceneOwner {
    pub contexts: Vec<SelectorItemContext>,
    pub style: SelectorSceneStyle,
    pub color_context: Option<SelectorItemContext>,
}

/// Label slots already taken per (target, anchor): stacked labels move down.
type Slots = Vec<((SelectorTarget, &'static str), u64)>;

impl SelectorSceneOwner {
    /// The scene of `states` (a snapshot's states: committed with the
    /// candidate in its kind's place, in kind order) and the colour limits.
    pub fn build(&self, states: &[SelectorState], color_state: Option<&ColorLimitState>) -> Result<SelectorScene> {
        let mut slots: Slots = Vec::new();
        let mut groups = Vec::new();
        for state in states {
            let kind = SceneKind::Selector(state.kind());
            // The last context of a kind wins, as a dict built from them.
            if let Some(context) = self.contexts.iter().rev().find(|item| item.kind == kind) {
                let items = selector_primitives(state, context, &self.style, &mut slots)?;
                if !items.is_empty() { groups.push((kind, items)); }
            }
        }
        if let (Some(color_state), Some(context)) = (color_state, &self.color_context) {
            let items = color_limit_primitives(color_state, context, &self.style, &mut slots)?;
            if !items.is_empty() { groups.push((SceneKind::ColorLimits, items)); }
        }
        Ok(SelectorScene { groups })
    }
}

fn label(key: &'static str, text: String, target: &SelectorTarget, anchor: &'static str, color: Rgba, style: &SelectorSceneStyle, slots: &mut Slots) -> SelectorText {
    let lines = text.matches('\n').count() as u64 + 1;
    let slot = match slots.iter_mut().find(|(held, _)| held.0 == *target && held.1 == anchor) {
        Some((_, taken)) => { let slot = *taken; *taken = slot + lines.max(1); slot }
        None => { slots.push(((target.clone(), anchor), lines.max(1))); 0 }
    };
    let inset = style.label_inset_fraction;
    SelectorText {
        key, target: target.clone(), text,
        position: (if anchor == "left" { inset } else { 1.0 - inset }, py_max(inset, 1.0 - inset - style.label_line_fraction * slot as f64)),
        horizontal_alignment: anchor, color, font_family: style.font_family.clone(), font_size_pt: style.annotation_pt,
        zorder: style.selector_zorder,
    }
}

/// How many decimals a number deserves: what the axis resolves there.  A log
/// axis resolves a constant ratio, so its precision follows the value.
fn decimals(value: f64, span_precision: usize, scale: &AxisScale) -> Result<usize> {
    if *scale != AxisScale::Log { return Ok(span_precision); }
    selector_precision(value.abs())
}

fn selector_primitives(state: &SelectorState, context: &SelectorItemContext, style: &SelectorSceneStyle, slots: &mut Slots) -> Result<Vec<SelectorPrimitive>> {
    let (x0, x1) = context.x_limits;
    let (y0, y1) = context.y_limits;
    let xp = selector_precision((x1 - x0).abs() * context.x_label_factor)?;
    let yp = selector_precision((y1 - y0).abs())?;
    let fx = |value: f64| -> Result<String> {
        let scaled = value * context.x_label_factor;
        Ok(selector_number(scaled, decimals(scaled, xp, &context.x_scale)?))
    };
    let fy = |value: f64| -> Result<String> { Ok(selector_number(value, decimals(value, yp, &context.y_scale)?)) };
    let target = &context.target;
    let zorder = style.selector_zorder;
    let rgba = context.selector_rgba;
    let line_color = (rgba.0, rgba.1, rgba.2, rgba.3 * style.line_alpha);
    let line = |key: &'static str, points: Vec<Point>| SelectorPrimitive::Line(SelectorLine {
        key, target: target.clone(), points, color: line_color, width_pt: style.line_width_pt, zorder, linestyle: "-".into(),
    });
    let handles = |points: Vec<Point>| SelectorPrimitive::Markers(SelectorMarkers {
        key: "handles", target: target.clone(), points, shape: "square", size_pt: style.handle_size_pt,
        facecolor: (1.0, 1.0, 1.0, 1.0), edgecolor: Some(rgba), edge_width_pt: style.handle_edge_width_pt, zorder,
    });
    match state.value {
        SelectorValue::XRange(value) => {
            let middle_y = midpoint(&context.y_scale, y0, y1)?;
            let text = format!("({}, {})", fx(value.low)?, fx(value.high)?);
            Ok(vec![
                line("low", vec![(value.low, y0), (value.low, y1)]),
                line("high", vec![(value.high, y0), (value.high, y1)]),
                handles(vec![(value.low, middle_y), (value.high, middle_y)]),
                SelectorPrimitive::Text(label("label", text, &context.label_target, "right", rgba, style, slots)),
            ])
        }
        SelectorValue::Area(value) => {
            let (low_x, high_x, low_y, high_y) = (value.x.low, value.x.high, value.y.low, value.y.high);
            // Drawn, and grabbed, with an end the scale cannot place at the
            // wall it is clipped to; the label still reads the box's values.
            let (left, right) = (placed(&context.x_scale, low_x, context.x_limits), placed(&context.x_scale, high_x, context.x_limits));
            let (bottom, top) = (placed(&context.y_scale, low_y, context.y_limits), placed(&context.y_scale, high_y, context.y_limits));
            let middle_x = midpoint(&context.x_scale, left, right)?;
            let middle_y = midpoint(&context.y_scale, bottom, top)?;
            let corners = vec![(left, bottom), (middle_x, bottom), (right, bottom), (right, middle_y), (right, top), (middle_x, top),
                               (left, top), (left, middle_y)];
            let mut outline = corners.clone();
            outline.push(corners[0]);
            let text = format!("({}, {})\n({}, {})", fx(low_x)?, fy(low_y)?, fx(high_x)?, fy(high_y)?);
            Ok(vec![line("outline", outline), handles(corners),
                    SelectorPrimitive::Text(label("label", text, &context.label_target, "left", rgba, style, slots))])
        }
        SelectorValue::Crosshair(value) => {
            let sample_text = match &context.sample_value {
                Some(SampleValue::Number(sample)) => Some(selector_number(*sample, selector_precision(context.sample_span.unwrap_or(0.0))?)),
                Some(SampleValue::Text(text)) => Some(text.clone()),
                None => None,
            };
            let text = format!("({}, {}{})", fx(value.x)?, fy(value.y)?, sample_text.map_or(String::new(), |text| format!(", {text}")));
            let mut items = vec![
                line("vertical", vec![(value.x, y0), (value.x, y1)]),
                line("horizontal", vec![(x0, value.y), (x1, value.y)]),
                SelectorPrimitive::Markers(SelectorMarkers {
                    key: "point", target: target.clone(), points: vec![(value.x, value.y)], shape: "circle",
                    size_pt: style.crosshair_size_pt, facecolor: line_color, edgecolor: None, edge_width_pt: 0.0, zorder,
                }),
                SelectorPrimitive::Text(label("label", text, &context.label_target, "right", rgba, style, slots)),
            ];
            if let (Some(SampleValue::Number(sample)), Some(sample_target), Some((sample_x0, sample_x1)), Some(sample_rgba)) =
                (&context.sample_value, &context.sample_target, context.sample_x_limits, context.sample_rgba)
            {
                items.push(SelectorPrimitive::Line(SelectorLine {
                    key: "sample", target: sample_target.clone(), points: vec![(sample_x0, *sample), (sample_x1, *sample)],
                    color: sample_rgba, width_pt: style.annotation_pt / 4.0, zorder, linestyle: "-".into(),
                }));
            }
            Ok(items)
        }
        SelectorValue::Threshold(value) => {
            let points = if context.threshold_uses_x { vec![(value, y0), (value, y1)] } else { vec![(x0, value), (x1, value)] };
            let mut items = vec![SelectorPrimitive::Line(SelectorLine {
                key: "threshold", target: target.clone(), points, color: context.threshold_rgba, width_pt: style.threshold_width_pt,
                zorder, linestyle: style.threshold_linestyle.clone(),
            })];
            if !context.threshold_text.is_empty() {
                items.push(SelectorPrimitive::Text(label("label", context.threshold_text.clone(), &context.label_target, "right",
                                                         context.threshold_label_rgba, style, slots)));
            }
            Ok(items)
        }
    }
}

fn color_limit_primitives(state: &ColorLimitState, context: &SelectorItemContext, style: &SelectorSceneStyle, slots: &mut Slots) -> Result<Vec<SelectorPrimitive>> {
    let value = state.value;
    let (x0, x1) = context.x_limits;
    let (y0, y1) = context.y_limits;
    let precision = selector_precision((y1 - y0).abs())?;
    let (low_color, high_color) = context.color_limit_rgba;
    let line = |key: &'static str, level: f64, color: Rgba| SelectorPrimitive::Line(SelectorLine {
        key, target: context.target.clone(), points: vec![(x0, level), (x1, level)], color,
        width_pt: style.side_distribution_width_pt, zorder: style.selector_zorder, linestyle: "-".into(),
    });
    let mut items = vec![line("low", value.low, low_color), line("high", value.high, high_color)];
    if !state.candidate { return Ok(items); }
    let text = format!("H low={}  high={}", selector_number(value.low, precision), selector_number(value.high, precision));
    items.push(SelectorPrimitive::Text(label("label", text, &context.label_target, "left", context.threshold_label_rgba, style, slots)));
    Ok(items)
}
