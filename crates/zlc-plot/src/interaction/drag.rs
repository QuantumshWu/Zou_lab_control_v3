//! Drag handles and the one bounded numeric-range drag every selector and
//! colour-limit gesture resolves through (`selectors.DragHandle`,
//! `GestureState`, `_drag_numeric_range`, `_clamp`, `_clamp_range`,
//! `_dragged_value`, `_stored_state`).

use crate::configuration::error::{PlotError, Result};
use crate::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorState, SelectorValue};
use crate::render::scale::AxisScale;

use super::pymath::{axis_value, finite, placed, py_max, py_min, sorted2};

/// What part of a selector a press holds.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum DragHandle { New, Body, Low, High, Left, Right, Bottom, Top, BottomLeft, BottomRight, TopLeft, TopRight }

impl DragHandle {
    pub const ALL: [DragHandle; 12] = [DragHandle::New, DragHandle::Body, DragHandle::Low, DragHandle::High, DragHandle::Left,
        DragHandle::Right, DragHandle::Bottom, DragHandle::Top, DragHandle::BottomLeft, DragHandle::BottomRight,
        DragHandle::TopLeft, DragHandle::TopRight];
    pub fn value(self) -> &'static str {
        match self {
            DragHandle::New => "new", DragHandle::Body => "body", DragHandle::Low => "low", DragHandle::High => "high",
            DragHandle::Left => "left", DragHandle::Right => "right", DragHandle::Bottom => "bottom", DragHandle::Top => "top",
            DragHandle::BottomLeft => "bottom_left", DragHandle::BottomRight => "bottom_right",
            DragHandle::TopLeft => "top_left", DragHandle::TopRight => "top_right",
        }
    }
    pub fn parse(text: &str) -> Option<Self> { Self::ALL.into_iter().find(|handle| handle.value() == text) }
}

/// The controller's record of one pointer drag: what it holds, where it was
/// pressed, the value it started from and the candidate it has reached.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GestureState {
    pub kind: SelectorKind,
    pub handle: DragHandle,
    pub origin: CrosshairPoint,
    pub original_value: SelectorValue,
    pub candidate_value: SelectorValue,
    pub revision: u64,
}

/// `_clamp(value, bounds)`.
pub fn clamp(value: f64, bounds: Option<NumericRange>) -> f64 {
    match bounds { None => value, Some(bounds) => py_min(py_max(value, bounds.low), bounds.high) }
}

/// `_clamp_range(value, bounds)`: a range at least as wide as the bounds is
/// the bounds; any other is pushed back inside, keeping its span.
pub fn clamp_range(value: NumericRange, bounds: Option<NumericRange>) -> Result<NumericRange> {
    let Some(bounds) = bounds else { return Ok(value) };
    if value.span() >= bounds.span() { return Ok(bounds); }
    let (mut low, mut high) = (value.low, value.high);
    if low < bounds.low {
        high += bounds.low - low;
        low = bounds.low;
    }
    if high > bounds.high {
        low -= high - bounds.high;
        high = bounds.high;
    }
    NumericRange::new(py_max(low, bounds.low), py_min(high, bounds.high))
}

/// `NumericRange.shifted(delta)`.
pub fn shifted(range: NumericRange, delta: f64) -> Result<NumericRange> {
    let delta = finite(delta, "range shift")?;
    NumericRange::new(range.low + delta, range.high + delta)
}

/// Resolve every bounded numeric-range drag (`_drag_numeric_range`).
///
/// `minimum_span` is `None` for data selectors and explicit for colour
/// limits.  `scale` only matters to the BODY handle, which SLIDES where the
/// axis is straight; every other handle sets an end to the value under the
/// pointer.  `Ok(None)` when that slide carries an end where the unit has no
/// value.
pub fn drag_numeric_range(
    original: NumericRange,
    handle: DragHandle,
    origin: f64,
    position: f64,
    minimum_span: Option<f64>,
    bounds: Option<NumericRange>,
    scale: &AxisScale,
) -> Result<Option<NumericRange>> {
    let handle = match handle { DragHandle::Left => DragHandle::Low, DragHandle::Right => DragHandle::High, other => other };
    if !matches!(handle, DragHandle::New | DragHandle::Body | DragHandle::Low | DragHandle::High) {
        return Err(PlotError::value("range handle must be NEW, BODY, LOW, or HIGH"));
    }
    let origin = finite(origin, "range drag origin")?;
    let position = finite(position, "range drag position")?;
    let minimum = match minimum_span { None => 0.0, Some(minimum) => finite(minimum, "range drag minimum span")? };
    if minimum < 0.0 || (minimum_span.is_some() && minimum == 0.0) {
        return Err(PlotError::value("minimum_span must be positive"));
    }
    if let Some(bounds) = bounds {
        if bounds.span() <= 0.0 || minimum > bounds.span() {
            return Err(PlotError::value("bounds must contain the minimum span"));
        }
    }
    let slide = handle == DragHandle::Body && *scale != AxisScale::Linear;
    let original = if slide { original } else { clamp_range(original, bounds)? };
    if handle != DragHandle::New && original.span() < minimum {
        return Err(PlotError::value("original is narrower than minimum_span"));
    }
    match handle {
        DragHandle::New => {
            let candidate = if position < origin {
                NumericRange::new(py_min(position, origin - minimum), origin)?
            } else {
                NumericRange::new(origin, py_max(position, origin + minimum))?
            };
            Ok(Some(clamp_range(candidate, bounds)?))
        }
        DragHandle::Low => {
            let mut low = py_min(clamp(position, bounds), original.high - minimum);
            if let Some(bounds) = bounds { low = py_max(low, bounds.low); }
            Ok(Some(NumericRange::new(low, original.high)?))
        }
        DragHandle::High => {
            let mut high = py_max(clamp(position, bounds), original.low + minimum);
            if let Some(bounds) = bounds { high = py_min(high, bounds.high); }
            Ok(Some(NumericRange::new(original.low, high)?))
        }
        _ if slide => {
            // The same slide expressed where the axis is straight; the wall
            // stops it there too, from the box as it is.  An end the scale
            // cannot place slides from the low wall it is drawn at.
            let walls = bounds.map(|bounds| (bounds.low, bounds.high));
            let place = |end: f64| match walls { None => end, Some(walls) => placed(scale, end, walls) };
            let start = scale.space(place(original.low));
            let stop = scale.space(place(original.high));
            let mut moved = scale.space(position) - scale.space(origin);
            if let Some(bounds) = bounds {
                let (wall_low, wall_high) = sorted2(scale.space(bounds.low), scale.space(bounds.high));
                if (stop - start).abs() >= wall_high - wall_low { return Ok(Some(bounds)); }
                moved = py_min(py_max(moved, wall_low - py_min(start, stop)), wall_high - py_max(start, stop));
            }
            let low = axis_value(scale, start + moved)?;
            let high = axis_value(scale, stop + moved)?;
            if !(low.is_finite() && high.is_finite()) { return Ok(None); }
            Ok(Some(NumericRange::new(clamp(low, bounds), clamp(high, bounds))?))
        }
        _ => Ok(Some(clamp_range(shifted(original, position - origin)?, bounds)?)),
    }
}

/// `_stored_state`: a re-installed kind always advances its revision.
pub fn stored_state(state: SelectorState, previous_revision: Option<u64>) -> SelectorState {
    let revision = match previous_revision { Some(previous) => state.revision.max(previous + 1), None => state.revision };
    SelectorState { revision, ..state }
}

fn range_of(range: Option<NumericRange>) -> Result<NumericRange> {
    range.ok_or_else(|| PlotError::type_error("rectangle x and y must be NumericRange"))
}

/// The axes facts one move is resolved against: the canonical extent each
/// selector is held to and the scale each axis slides under.
#[derive(Clone, Copy, Debug)]
pub struct DragFrame<'a> {
    pub x_bounds: Option<NumericRange>,
    pub y_bounds: Option<NumericRange>,
    pub x_scale: &'a AxisScale,
    pub y_scale: &'a AxisScale,
}

impl DragFrame<'static> {
    /// Unbounded linear axes, the Python defaults.
    pub fn linear() -> Self { DragFrame { x_bounds: None, y_bounds: None, x_scale: &AxisScale::Linear, y_scale: &AxisScale::Linear } }
}

/// The value a selector of `kind` takes with the pointer at `point`
/// (`_dragged_value`).  A slide that would carry an end where the unit has no
/// value leaves that range where the last move drew it.
pub fn dragged_value(kind: SelectorKind, gesture: &GestureState, point: CrosshairPoint, frame: &DragFrame) -> Result<SelectorValue> {
    let (handle, origin, last) = (gesture.handle, gesture.origin, gesture.candidate_value);
    match (kind, gesture.original_value) {
        (SelectorKind::XRange, SelectorValue::XRange(original)) => {
            let slid = drag_numeric_range(original, handle, origin.x, point.x, None, frame.x_bounds, frame.x_scale)?;
            Ok(slid.map_or(last, SelectorValue::XRange))
        }
        (SelectorKind::Area, SelectorValue::Area(original)) => {
            let axis = |range: NumericRange, handle: DragHandle, horizontal: bool| {
                let (origin, position, bounds, scale) = if horizontal { (origin.x, point.x, frame.x_bounds, frame.x_scale) } else { (origin.y, point.y, frame.y_bounds, frame.y_scale) };
                drag_numeric_range(range, handle, origin, position, None, bounds, scale)
            };
            if handle == DragHandle::New {
                let x = range_of(axis(original.x, DragHandle::New, true)?)?;
                let y = range_of(axis(original.y, DragHandle::New, false)?)?;
                return Ok(SelectorValue::Area(RectangleRange { x, y }));
            }
            let (mut x, mut y) = (original.x, original.y);
            if handle == DragHandle::Body {
                let SelectorValue::Area(last) = last else { return Err(PlotError::runtime("area candidate is not a rectangle")) };
                let slid_x = axis(x, DragHandle::Body, true)?;
                let slid_y = axis(y, DragHandle::Body, false)?;
                // Against the end of what the unit can express the box stops
                // on that axis and still slides along the other.
                x = slid_x.unwrap_or(last.x);
                y = slid_y.unwrap_or(last.y);
            } else {
                use DragHandle::*;
                if matches!(handle, Left | BottomLeft | TopLeft) { x = range_of(axis(x, Low, true)?)?; }
                if matches!(handle, Right | BottomRight | TopRight) { x = range_of(axis(x, High, true)?)?; }
                if matches!(handle, Bottom | BottomLeft | BottomRight) { y = range_of(axis(y, Low, false)?)?; }
                if matches!(handle, Top | TopLeft | TopRight) { y = range_of(axis(y, High, false)?)?; }
            }
            Ok(SelectorValue::Area(RectangleRange { x, y }))
        }
        (SelectorKind::Crosshair, SelectorValue::Crosshair(_)) => {
            Ok(SelectorValue::Crosshair(CrosshairPoint::new(clamp(point.x, frame.x_bounds), clamp(point.y, frame.y_bounds))?))
        }
        (SelectorKind::Threshold, SelectorValue::Threshold(_)) => Ok(SelectorValue::Threshold(clamp(point.y, frame.y_bounds))),
        (kind, _) => Err(PlotError::runtime(format!("unsupported selector kind: {}", kind.value()))),
    }
}
