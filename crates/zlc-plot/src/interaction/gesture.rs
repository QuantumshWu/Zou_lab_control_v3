//! The pure rules of the session's pointer routing (`_session_gesture`,
//! `PlotSession._raster_pointer_event` / `_raster_pointer_state`): how a
//! pointer is read, which selector a press holds, when a press becomes a
//! drag or stays a click, what a double click is, how the wheel zooms, where
//! the colour-limit drag starts, what a release decides.  The session keeps
//! its own state (spec, focus, projection, renderer) and asks these.

use std::collections::HashMap;

use crate::configuration::error::{PlotError, Result};
use crate::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorState, SelectorValue, Viewport};
use crate::render::scale::AxisScale;

use super::controller::SelectorController;
use super::drag::{DragFrame, DragHandle};
use super::engine::{area_drag_handle, pan_rectangle, range_endpoint_hit, AxesId, ColorGesture, ColorLimitDrag, PanGesture, PointerGesture,
                    SelectorGesture};
use super::pymath::{axis_value, divide, hypot, placed, pow, py_max, py_min, sorted2};
use super::transform::{Axis, AxisTransform, CanvasSize};

/// The interaction policy (`zlc_plot.config.InteractionDefaults`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InteractionDefaults {
    pub double_click_interval_ms: i64,
    pub double_click_radius_px: f64,
    pub selector_hit_radius_fraction: f64,
    pub selector_handle_radius_px: f64,
    pub wheel_zoom_factor: f64,
}

impl Default for InteractionDefaults {
    /// The library defaults (`zlc_plot.config.DEFAULTS.interaction`).
    fn default() -> Self {
        Self { double_click_interval_ms: 500, double_click_radius_px: 6.0, selector_hit_radius_fraction: 0.035,
               selector_handle_radius_px: 10.0, wheel_zoom_factor: 1.1 }
    }
}

/// The canonical extent a selector is held to, or `None` where the drawn
/// axis ends on a value its canonical unit cannot express (`_drawn_bounds`).
pub fn drawn_bounds(limits: (f64, f64)) -> Option<NumericRange> {
    (limits.0.is_finite() && limits.1.is_finite()).then(|| NumericRange::new(limits.0, limits.1).expect("finite limits"))
}

/// A threshold drag on a histogram reads the pointer's x, from the press on
/// (`_threshold_gesture_on_histogram`).
pub fn threshold_gesture_on_histogram(gesture: Option<&PointerGesture>, histogram: bool) -> bool {
    matches!(gesture, Some(PointerGesture::Selector(g)) if g.kind == SelectorKind::Threshold) && histogram
}

/// The bounds a selector drag on `transform` is held to (`_selector_x_bounds`
/// and `_selector_y_bounds`): a histogram's threshold is held along x.
pub fn selector_bounds(transform: &AxisTransform, threshold_on_histogram: bool) -> (Option<NumericRange>, Option<NumericRange>) {
    let x = drawn_bounds(transform.canonical_x_limits);
    let y = if threshold_on_histogram { x } else { drawn_bounds(transform.canonical_y_limits) };
    (x, y)
}

/// The canonical point under a pointer event, or `None` where it reads
/// nothing (`_event_canonical`).  A drag (any handle but NEW) keeps, axis by
/// axis, the coordinate it last read where the canonical unit has no value;
/// a histogram threshold reads x as its level; the gesture's reading follows.
pub fn event_canonical(
    transform: Option<&AxisTransform>,
    pixel: Option<(f64, f64)>,
    canvas: CanvasSize,
    gesture: Option<&mut PointerGesture>,
    histogram: bool,
) -> Result<Option<CrosshairPoint>> {
    let Some(transform) = transform else { return Ok(None) };
    let Some(reading) = transform.canonical_reading(pixel, canvas)? else { return Ok(None) };
    match gesture {
        Some(PointerGesture::Selector(g)) => follow_reading(g, reading, histogram),
        _ => finite_point(reading),
    }
}

fn finite_point((x, y): (f64, f64)) -> Result<Option<CrosshairPoint>> {
    if !(x.is_finite() && y.is_finite()) { return Ok(None); }
    CrosshairPoint::new(x, y).map(Some)
}

/// A selector gesture's reading of `(x, y)`: the fallback, the histogram
/// threshold's x, and the reading it leaves behind.
fn follow_reading(gesture: &mut SelectorGesture, (mut x, mut y): (f64, f64), histogram: bool) -> Result<Option<CrosshairPoint>> {
    if gesture.handle != DragHandle::New {
        if !x.is_finite() { x = gesture.reading.x; }
        if !y.is_finite() { y = gesture.reading.y; }
    }
    if gesture.kind == SelectorKind::Threshold && histogram { y = x; }
    let point = finite_point((x, y))?;
    if let Some(point) = point { gesture.reading = point; }
    Ok(point)
}

/// The point under `pixel` for the gesture's own transform.
fn selector_point(gesture: &mut SelectorGesture, pixel: (f64, f64), canvas: CanvasSize, histogram: bool) -> Result<Option<CrosshairPoint>> {
    match gesture.transform.canonical_reading(Some(pixel), canvas)? {
        None => Ok(None),
        Some(reading) => follow_reading(gesture, reading, histogram),
    }
}

/// The selector gesture a left press arms once its point is read
/// (`_on_button_press`): the selector `hit` holds, or, on the active axes
/// only, a new Area from the press.  `selector_axes` resolves the axes a
/// state is painted on (`_selector_axes`).
pub fn selector_press(
    point: CrosshairPoint,
    pixel: (f64, f64),
    hit: Option<(SelectorState, DragHandle)>,
    on_active_axes: bool,
    focused_facet: Option<u64>,
    transform: &AxisTransform,
    selector_axes: impl FnOnce(&SelectorState) -> Option<AxesId>,
) -> Result<Option<SelectorGesture>> {
    let (state, handle) = match hit {
        Some(hit) => hit,
        None if on_active_axes => (new_area_draft(point, focused_facet)?, DragHandle::New),
        None => return Ok(None),
    };
    let Some(axes) = selector_axes(&state) else { return Ok(None) };
    Ok(Some(SelectorGesture { axes, transform: transform.clone(), kind: state.kind(), state, handle, origin: point, origin_px: pixel,
                              reading: point, started: false, moved: false }))
}

/// What one move of a selector gesture asks of the session.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectorStep {
    /// Nothing can be read under the hand: the candidate stays where it is.
    Unread,
    /// Armed by the press, the hand still on the press pixel: no candidate,
    /// no compose (a double click's first press must not flash an Area).
    Armed,
    /// The controller no longer holds the drag: cancel the gesture.
    Cancel,
    /// The candidate did not change.
    Unchanged,
    /// A new candidate: preview it (a threshold refreshes its classifier
    /// labels first) and announce it as UPDATED.
    Preview(SelectorState),
}

/// One move's answer.  `started` is true when this move started the drag
/// (the renderer begins the selector gesture); it is reported even when the
/// step failed, as Python had already begun it.
#[derive(Clone, Debug, PartialEq)]
pub struct SelectorMotion { pub started: bool, pub step: Result<SelectorStep> }

/// Follow one move of a selector gesture (`_on_motion`'s selector path):
/// the first move off the press pixel starts the controller's drag from the
/// captured state, every move drags under the press transform's bounds and
/// scales.
pub fn selector_motion(controller: &mut SelectorController, gesture: &mut SelectorGesture, pixel: (f64, f64), canvas: CanvasSize,
                       histogram: bool) -> SelectorMotion {
    if !gesture.moved { gesture.moved = gesture.leaves_press(pixel); }
    let mut started = false;
    let step = (|| {
        let Some(point) = selector_point(gesture, pixel, canvas, histogram)? else { return Ok(SelectorStep::Unread) };
        if !gesture.started {
            if !gesture.moved { return Ok(SelectorStep::Armed); }
            controller.pointer_down(gesture.kind, gesture.origin.x, gesture.origin.y, gesture.handle, Some(gesture.state))?;
            gesture.started = true;
            started = true;
        }
        if controller.active_gesture().is_none() { return Ok(SelectorStep::Cancel); }
        let Some(current) = controller.candidate_state()? else { return Ok(SelectorStep::Cancel) };
        let updated = controller.pointer_move(point.x, point.y, &drag_frame(gesture, histogram))?;
        Ok(match updated { Some(updated) if updated != current => SelectorStep::Preview(updated), _ => SelectorStep::Unchanged })
    })();
    SelectorMotion { started, step }
}

fn drag_frame(gesture: &SelectorGesture, histogram: bool) -> DragFrame<'_> {
    let (x_bounds, y_bounds) = selector_bounds(&gesture.transform, gesture.kind == SelectorKind::Threshold && histogram);
    DragFrame { x_bounds, y_bounds, x_scale: gesture.transform.interaction_scale(Axis::X), y_scale: gesture.transform.interaction_scale(Axis::Y) }
}

/// What letting go of a selector gesture asks of the session
/// (`_finish_selector_gesture`); the gesture is over in every case.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectorRelease {
    /// A press let go on its own pixel on blank space without ever moving:
    /// remove the committed selector of its kind, if any.  A click, not a
    /// zero-size Area: no candidate, no preview.
    BlankClick,
    /// A press that never started, let go elsewhere: nothing.
    Abandoned,
    /// The controller no longer holds the drag: cancel the gesture.
    Cancel,
    /// The drag ended without a candidate.
    Nothing,
    /// Commit this exact final geometry ([`SelectorController::commit_finished`])
    /// and announce it as COMMITTED.
    Commit(SelectorState),
    /// The final geometry has no extent: remove the committed selector of
    /// its kind, or, with none, repaint the overlay.
    Degenerate(SelectorState),
}

/// Let go of a selector gesture.  On an error the session cancels the
/// gesture ([`cancel_selector_gesture`]) and reports it, as Python does.
pub fn selector_release(controller: &mut SelectorController, gesture: &mut SelectorGesture, pixel: (f64, f64), canvas: CanvasSize,
                        histogram: bool) -> Result<SelectorRelease> {
    if !gesture.started {
        return Ok(if gesture.is_blank_click(pixel) { SelectorRelease::BlankClick } else { SelectorRelease::Abandoned });
    }
    if controller.active_gesture().is_none() { return Ok(SelectorRelease::Cancel); }
    let candidate = match selector_point(gesture, pixel, canvas, histogram)? {
        // Let go over what nothing can be read from, the box is where the
        // last reading put it.
        None => {
            let candidate = controller.candidate_state()?;
            controller.cancel()?;
            candidate
        }
        Some(point) => controller.finish_gesture(point.x, point.y, &drag_frame(gesture, histogram))?,
    };
    Ok(match candidate {
        None => SelectorRelease::Nothing,
        Some(candidate) if is_degenerate_selector(&candidate) => SelectorRelease::Degenerate(candidate),
        Some(candidate) => SelectorRelease::Commit(candidate),
    })
}

/// Cancel a selector gesture (`_cancel_gesture`): a started drag loses the
/// pointer capture and answers the state it dragged from (repaint the
/// committed overlay when there is one).
pub fn cancel_selector_gesture(controller: &mut SelectorController, gesture: &SelectorGesture) -> Result<Option<SelectorState>> {
    if gesture.started { controller.lost_pointer_capture() } else { Ok(None) }
}

/// Follow one move of a colour-limit drag; `true` when there is a candidate
/// to preview (`gesture.drag.candidate`).
pub fn color_motion(gesture: &mut ColorGesture, pixel: (f64, f64), canvas: CanvasSize) -> Result<bool> {
    let Some(point) = gesture.transform.canonical_reading(Some(pixel), canvas)?.map(finite_point).transpose()?.flatten() else { return Ok(false) };
    gesture.drag = gesture.drag.moved(point.x)?;
    Ok(true)
}

/// What letting go of a colour-limit drag asks of the session
/// (`_finish_color_gesture`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColorRelease {
    /// Nothing can be read under the hand: cancel the gesture.
    Cancel,
    /// Fix the colour limits to the candidate (canonical).
    Changed(NumericRange),
    /// The limits did not move: repaint the base geometry.
    Unchanged,
}

pub fn color_release(gesture: &mut ColorGesture, pixel: (f64, f64), canvas: CanvasSize) -> Result<ColorRelease> {
    let Some(point) = gesture.transform.canonical_reading(Some(pixel), canvas)?.map(finite_point).transpose()?.flatten() else { return Ok(ColorRelease::Cancel) };
    gesture.drag = gesture.drag.moved(point.x)?;
    Ok(if gesture.drag.changed() { ColorRelease::Changed(gesture.drag.candidate) } else { ColorRelease::Unchanged })
}

/// Recent single clicks of the middle and right buttons, for the timed
/// double click a native frontend does not report (`_click_history`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ClickHistory { clicks: HashMap<u8, (f64, f64, f64)> }

impl ClickHistory {
    pub fn new() -> Self { Self::default() }

    /// The last single click of `button`: `(time, x, y)`.
    pub fn last(&self, button: u8) -> Option<(f64, f64, f64)> { self.clicks.get(&button).copied() }

    /// Normalize native, browser-detail and timed middle/right double clicks
    /// (`_is_double_click`).  `browser_detail` is the browser click count
    /// (`None` where the event carries none or it is not a number);
    /// `pixel` the event position (`None` where it has none); `now` the
    /// monotonic clock in seconds.
    pub fn is_double_click(&mut self, button: u8, explicit: bool, browser_detail: Option<i64>, pixel: Option<(f64, f64)>, now: f64,
                           defaults: &InteractionDefaults) -> bool {
        if explicit || browser_detail.is_some_and(|detail| detail >= 2) {
            self.clicks.remove(&button);
            return true;
        }
        if button != 2 && button != 3 { return false; }
        let Some((x, y)) = pixel.filter(|(x, y)| x.is_finite() && y.is_finite()) else {
            self.clicks.remove(&button);
            return false;
        };
        let interval = defaults.double_click_interval_ms as f64 / 1000.0;
        if let Some((then, previous_x, previous_y)) = self.clicks.get(&button).copied() {
            if now >= then && now - then <= interval && hypot(x - previous_x, y - previous_y) <= defaults.double_click_radius_px {
                self.clicks.remove(&button);
                return true;
            }
        }
        self.clicks.insert(button, (now, x, y));
        false
    }
}

/// The area handle under a pixel for one displayed rectangle
/// (`_area_drag_handle`): handles are placed in pixels through the press'
/// transform, an end the scale cannot place at its wall.
pub fn session_area_hit(displayed: RectangleRange, pixel: Option<(f64, f64)>, transform: &AxisTransform, canvas: CanvasSize,
                        handle_radius: f64) -> Result<Option<(f64, DragHandle)>> {
    let Some(mouse) = pixel else { return Ok(None) };
    let pixel_point = |(x, y): (f64, f64)| -> Result<(f64, f64)> {
        Ok(transform.display_to_pixel(placed(&transform.x_scale, x, transform.x_limits), placed(&transform.y_scale, y, transform.y_limits), canvas))
    };
    area_drag_handle(displayed, mouse, pixel_point, handle_radius)
}

/// Which selector a press at `point` holds (`_hit_selector`).
///
/// `states` is the resolved snapshot's states in kind order; `on_axes`
/// says whether a state lives on the pressed axes; `area_hit` answers an
/// Area's pixel handle test ([`session_area_hit`] on its displayed value).
/// Ends and thresholds score in reaches of the drawn span where the axis is
/// straight; a threshold outranks everything, then the lowest score, then
/// the earliest state.
pub fn hit_selector(
    point: CrosshairPoint,
    states: &[SelectorState],
    mut on_axes: impl FnMut(&SelectorState) -> bool,
    focused_facet: Option<u64>,
    transform: &AxisTransform,
    hit_fraction: f64,
    histogram: bool,
    mut area_hit: impl FnMut(&SelectorState) -> Result<Option<(f64, DragHandle)>>,
) -> Result<Option<(SelectorState, DragHandle)>> {
    let x_scale = transform.interaction_scale(Axis::X);
    let y_scale = transform.interaction_scale(Axis::Y);
    let tx = reach(hit_fraction, transform.drawn_span(Axis::X));
    let ty = reach(hit_fraction, transform.drawn_span(Axis::Y));
    let mut best: Option<((u8, f64, usize), SelectorState, DragHandle)> = None;
    for (order, state) in states.iter().enumerate() {
        if !on_axes(state) || state.facet_index != focused_facet { continue; }
        let (mut score, mut handle) = (f64::INFINITY, DragHandle::Body);
        match state.value {
            SelectorValue::XRange(value) => {
                if let Some(endpoint) = range_endpoint_hit(value, point.x, tx, x_scale)? {
                    (score, handle) = endpoint;
                } else if value.low <= point.x && point.x <= value.high {
                    (score, handle) = (0.9, DragHandle::Body);
                }
            }
            SelectorValue::Area(_) => {
                if let Some(hit) = area_hit(state)? { (score, handle) = hit; }
            }
            SelectorValue::Threshold(level) => {
                let (coordinate, scale, reach) = if histogram { (point.x, x_scale, tx) } else { (point.y, y_scale, ty) };
                score = divide((scale.space(coordinate) - scale.space(level)).abs(), reach)?;
            }
            SelectorValue::Crosshair(_) => {}
        }
        if score <= 1.0 {
            let key = (if state.kind() == SelectorKind::Threshold { 0 } else { 1 }, score, order);
            let better = match &best { None => true, Some((held, _, _)) => key.0 < held.0 || (key.0 == held.0 && (key.1 < held.1 || (key.1 == held.1 && key.2 < held.2))) };
            if better { best = Some((key, *state, handle)); }
        }
    }
    Ok(best.map(|(_, state, handle)| (state, handle)))
}

/// The new Area a press on blank plot space arms (`_start_pointer_selection`).
pub fn new_area_draft(point: CrosshairPoint, focused_facet: Option<u64>) -> Result<SelectorState> {
    let x = NumericRange::new(point.x, point.x)?;
    let y = NumericRange::new(point.y, point.y)?;
    SelectorState::new(SelectorValue::Area(RectangleRange { x, y }), 0, focused_facet)
}

/// The colour-limit drag a press on the distribution rail starts
/// (`_begin_color_limit_pointer`), or `None` when it is near neither guide
/// (the press is still the rail's).  `bounds` is the rail's canonical
/// extent, `value` the effective colour limits.
pub fn begin_color_limit_drag(bounds: NumericRange, value: NumericRange, point_x: f64, hit_fraction: f64) -> Result<Option<ColorLimitDrag>> {
    let tolerance = reach(hit_fraction, bounds.span());
    let Some((_score, handle)) = range_endpoint_hit(value, point_x, tolerance, &AxisScale::Linear)? else { return Ok(None) };
    // Relative to the rail alone: v2's absolute 1e-12 floor made a rail of
    // smaller values (a current in A) its own minimum span, every drag of
    // limits narrower than the rail refused.
    let minimum_span = py_min(bounds.span(), py_max(value.span() * 1.0e-12, bounds.span() * 1.0e-12));
    Ok(Some(ColorLimitDrag { original: value, candidate: value, handle, origin: point_x, bounds, minimum_span }))
}

/// How far a press reaches along an axis spanning `span`: `fraction` of
/// it, never below its binary noise (1e-12 of it), and something even on
/// an axis of no span.  v2's floor was an absolute 1e-12, a whole axis of
/// small values (a current in A), where every press then held a selector
/// end.
fn reach(fraction: f64, span: f64) -> f64 { py_max(py_max(fraction * span, 1.0e-12 * span), f64::MIN_POSITIVE) }

impl SelectorGesture {
    /// Exact pixel inequality is the sole activation edge: no drag threshold.
    pub fn leaves_press(&self, pixel: (f64, f64)) -> bool { pixel != self.origin_px }

    /// A release that is a click on blank space: it removes the committed
    /// selector of the same kind and creates nothing.
    pub fn is_blank_click(&self, pixel: (f64, f64)) -> bool {
        self.handle == DragHandle::New && !self.moved && pixel == self.origin_px
    }
}

/// A scene press released within the double-click radius picks a bar.
pub fn is_pick_click(origin_px: (f64, f64), pixel: (f64, f64), defaults: &InteractionDefaults) -> bool {
    !(hypot(pixel.0 - origin_px.0, pixel.1 - origin_px.1) > defaults.double_click_radius_px)
}

/// An X range or Area without extent (`_is_degenerate_selector`).
pub fn is_degenerate_selector(state: &SelectorState) -> bool {
    match state.value {
        SelectorValue::XRange(value) => value.span() <= 0.0,
        SelectorValue::Area(value) => value.x.span() <= 0.0 || value.y.span() <= 0.0,
        _ => false,
    }
}

/// `_require_stable_selector`.
pub fn require_stable_selector(state: &SelectorState) -> Result<()> {
    if !is_degenerate_selector(state) { return Ok(()); }
    if state.kind() == SelectorKind::Area { return Err(PlotError::value("area selector requires non-degenerate x and y ranges")); }
    Err(PlotError::value(format!("{} selector requires a non-degenerate range", state.kind().value())))
}

/// Accumulated wheel ticks, positive up; a bare notch is one tick.
pub fn wheel_ticks(step: f64, up: bool) -> f64 { if step == 0.0 { if up { 1.0 } else { -1.0 } } else { step } }

/// The height-bar camera zoom after a wheel burst: `zoom * growth ** -ticks`.
pub fn wheel_camera_zoom(zoom: f64, growth: f64, ticks: f64) -> Result<f64> { Ok(zoom * pow(growth, -ticks)?) }

/// `range` zoomed by `factor` about its own centre, in the axis' drawn space.
pub fn wheel_zoom(range: NumericRange, scale: &AxisScale, factor: f64) -> Result<NumericRange> {
    let low = scale.space(range.low);
    let high = scale.space(range.high);
    let sum = low + high;
    let middle = if sum.is_finite() { sum / 2.0 } else { low / 2.0 + high / 2.0 };
    let span = high - low;
    let half = span * factor / 2.0;
    let half = if half.is_finite() { half } else { (high / 2.0 - low / 2.0) * factor };
    let mut selected = NumericRange::new(axis_value(scale, middle - half)?, axis_value(scale, middle + half)?)?;
    if factor > 1.0 && selected == range {
        // At the precision floor a fractional reverse notch can round both
        // ends back to themselves. Move one representable value outward so
        // fine wheels cannot get stuck at a numerical limit either.
        let (low, high) = (range.low.next_down(), range.high.next_up());
        if low.is_finite() && high.is_finite() { selected = NumericRange::new(low, high)?; }
    }
    Ok(selected)
}

/// The viewport one wheel burst asks for (`_on_scroll` on a 2-D surface).
///
/// Anchored on the committed continuous view, never on the drawn
/// limits (renders lag commits): `base_x` (already mapped to the axes'
/// coordinates) and `base_y`, each `None` where nothing is asked, when the
/// drawn limits stand in. Only an image zooms y. Source pixels impose no
/// navigation limit. `minimum_span` optionally adds a caller's lower bound
/// in drawn space; otherwise only floating-point resolution limits zoom.
#[allow(clippy::too_many_arguments)]
pub fn wheel_viewport(
    base_x: Option<NumericRange>,
    base_y: Option<NumericRange>,
    drawn_x: (f64, f64),
    drawn_y: (f64, f64),
    scales: (&AxisScale, &AxisScale),
    image_like: bool,
    zoom_factor: f64,
    ticks: f64,
    minimum_span: Option<(f64, f64)>,
) -> Result<(NumericRange, Option<NumericRange>)> {
    let mut factor = pow(zoom_factor, ticks)?;
    let sorted_range = |limits: (f64, f64)| { let (low, high) = sorted2(limits.0, limits.1); NumericRange::new(low, high) };
    let x_axes = match base_x { None => sorted_range(drawn_x)?, Some(x) => x };
    let y_axes = match base_y { None => sorted_range(drawn_y)?, Some(y) => y };
    if factor < 1.0 {
        let floor = |range: NumericRange, scale: &AxisScale, minimum: f64| {
            let (low, high) = (scale.space(range.low), scale.space(range.high));
            let span = (high - low).abs();
            let centre = low / 2.0 + high / 2.0;
            // Stay above the renderer's nonsingular-range expansion and
            // leave enough representable values for transforms and ticks.
            let mut resolution = (centre.abs() * (64.0 * f64::EPSILON))
                .max(1e22 * f64::MIN_POSITIVE).max(minimum);
            // A log axis around 1 has its drawn centre at zero. Its inverse
            // still needs distinct floating-point *values*, as do converted
            // or sample-lattice axes near their zero in drawn coordinates.
            // Use the centre's precision, not the old endpoints': otherwise a
            // large burst towards zero stops earlier than equivalent small steps.
            let value = scale.value(centre);
            let position = scale.space(value);
            let delta = value.abs() * (64.0 * f64::EPSILON);
            for neighbour in [value - delta, value + delta] {
                let mapped = (scale.space(neighbour) - position).abs();
                if mapped.is_finite() { resolution = resolution.max(mapped); }
            }
            if span > 0.0 && span.is_finite() { (resolution / span).min(1.0) }
            else {
                let half_span = (high / 2.0 - low / 2.0).abs();
                if half_span > 0.0 && half_span.is_finite() { (resolution / 2.0 / half_span).min(1.0) } else { 1.0 }
            }
        };
        let (minimum_x, minimum_y) = minimum_span.unwrap_or((0.0, 0.0));
        factor = py_max(factor, floor(x_axes, scales.0, minimum_x));
        if image_like { factor = py_max(factor, floor(y_axes, scales.1, minimum_y)); }
    }
    let zoomed_x = wheel_zoom(x_axes, scales.0, factor)?;
    let zoomed_y = if image_like { Some(wheel_zoom(y_axes, scales.1, factor)?) } else { base_y };
    Ok((zoomed_x, zoomed_y))
}

/// What a middle double click does (`_zoom_to_selection_or_reset`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SelectionZoom { Viewport(NumericRange, NumericRange), XLimits(f64, f64), Reset }

/// A committed Area wins over an X range, each only on the focused cell; a
/// degenerate one sends the view home.  `committed` answers the committed
/// state of a kind, `display` its displayed value.
pub fn zoom_to_selection(
    mut committed: impl FnMut(SelectorKind) -> Option<SelectorState>,
    focused_facet: Option<u64>,
    mut display: impl FnMut(&SelectorState) -> Result<SelectorValue>,
) -> Result<SelectionZoom> {
    for kind in [SelectorKind::Area, SelectorKind::XRange] {
        let Some(state) = committed(kind) else { continue };
        if state.facet_index != focused_facet { continue; }
        match display(&state)? {
            SelectorValue::Area(value) => {
                if value.x.span() <= 0.0 || value.y.span() <= 0.0 { break; }
                return Ok(SelectionZoom::Viewport(value.x, value.y));
            }
            SelectorValue::XRange(value) => {
                if value.span() <= 0.0 { break; }
                return Ok(SelectionZoom::XLimits(value.low, value.high));
            }
            _ => return Err(PlotError::runtime("displayed selector changed kind")),
        }
    }
    Ok(SelectionZoom::Reset)
}

/// Follow a pan to the pointer at `pixel` (`_update_pan`): `true` when the
/// view moved and the gesture holds a new candidate.  `x_from_axes` maps the
/// axes' x into the viewport's (a pulse timeline's display unit), `normalize`
/// completes the continuous viewport, `viewport_y` is the session's
/// own y override and `committed` its committed viewport.
pub fn update_pan(
    gesture: &mut PanGesture,
    pixel: Option<(f64, f64)>,
    canvas: CanvasSize,
    image_like: bool,
    viewport_y: Option<NumericRange>,
    committed: Option<Viewport>,
    x_from_axes: impl FnOnce(NumericRange) -> Result<NumericRange>,
    normalize: impl FnOnce(Option<Viewport>) -> Result<Option<Viewport>>,
) -> Result<bool> {
    let Some(point) = gesture.transform.display_point(pixel, canvas)? else { return Ok(false) };
    let Some(moved) = pan_rectangle(gesture.origin, point, gesture.x, gesture.y, image_like, &gesture.transform.x_scale,
                                    &gesture.transform.y_scale)? else { return Ok(false) };
    let selected: Viewport = (Some(x_from_axes(moved.x)?), if image_like { Some(moved.y) } else { viewport_y });
    let selected = normalize(Some(selected))?;
    let current = gesture.candidate.or(committed);
    if current == selected { return Ok(false); }
    gesture.candidate = selected;
    Ok(true)
}

/// One raster pointer action (`_raster_pointer_event`'s `action`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RasterAction { Press, Move, Release, Scroll, Key, Cancel, Leave }

impl RasterAction {
    pub fn parse(action: &str) -> Result<Self> {
        Ok(match action.trim().to_lowercase().as_str() {
            "press" => RasterAction::Press, "move" => RasterAction::Move, "release" => RasterAction::Release,
            "scroll" => RasterAction::Scroll, "key" => RasterAction::Key, "cancel" => RasterAction::Cancel, "leave" => RasterAction::Leave,
            _ => return Err(PlotError::value(format!("unknown raster pointer action {}", crate::configuration::value::repr_str(action)))),
        })
    }
}

/// Where a raster pointer at canvas fractions `(x, y)` (top origin) lands:
/// the exact bottom-origin physical pixel, and the whole pixel a synthetic
/// Matplotlib event carries (`int()` truncates toward zero).
pub fn raster_pixel(x: f64, y: f64, button: Option<u8>, canvas: CanvasSize) -> Result<((f64, f64), (f64, f64))> {
    if !(x.is_finite() && y.is_finite()) { return Err(PlotError::value("raster pointer coordinates must be finite")); }
    if button.is_some_and(|button| !(1..=3).contains(&button)) {
        return Err(PlotError::value("raster pointer button must be 1, 2, 3, or None"));
    }
    let exact = (x * canvas.width, (1.0 - y) * canvas.height);
    Ok((exact, (exact.0.trunc() + 0.0, exact.1.trunc() + 0.0)))
}

/// The pointer facts a raster frontend needs back (`_raster_pointer_state`
/// without the candidate itself): the axes the gesture holds, when the
/// button must stay latched; whether a pan-like gesture (pan, orbit, scene
/// pick) is active; whether a selector is armed but not started.
pub fn pointer_latch(gesture: Option<&PointerGesture>, has_candidate: bool) -> (Option<AxesId>, bool, bool) {
    let active_pan = matches!(gesture, Some(PointerGesture::Pan(_) | PointerGesture::Orbit(_) | PointerGesture::Pick(_)));
    let armed_selector = matches!(gesture, Some(PointerGesture::Selector(g)) if !g.started);
    let axes = gesture.filter(|_| has_candidate || active_pan || armed_selector).map(|g| g.axes().clone());
    (axes, active_pan, armed_selector)
}
