//! Raster pointer routing (`PlotSession._raster_pointer_event` and
//! `_session_gesture`): one normalized pointer message from a frontend,
//! resolved against the axes the frontend hit, through the interaction
//! model's rules into selector, colour, pan, orbit, pick, facet and series
//! actions.

use std::sync::OnceLock;
use std::time::Instant;

use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::parameters::RenderEffect;
use crate::configuration::selectors::{NumericRange, SelectorKind, SelectorState, SelectorValue};
use crate::configuration::value::{py_strip, Value, ValueMap};
use crate::interaction::engine::{AxesId, CameraAngles, ColorGesture, OrbitGesture, PanGesture, PickGesture, PointerGesture};
use crate::interaction::gesture::{begin_color_limit_drag, cancel_selector_gesture, color_motion, color_release, event_canonical, hit_selector,
                                  is_pick_click, selector_motion, selector_press, selector_release, session_area_hit, update_pan,
                                  wheel_camera_zoom, wheel_ticks, wheel_viewport, zoom_to_selection, ColorRelease, SelectionZoom, SelectorRelease,
                                  SelectorStep};
use crate::interaction::pymath::sorted2;
use crate::interaction::transform::AxisTransform;
use crate::kinds::PlotKind;

use super::core::PlotSession;
use super::renderer::SeriesAction;

/// One raster pointer action (`_raster_pointer_event`'s `action`).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PointerAction { Press, Move, Release, Scroll, Key, Cancel, Leave }

impl PointerAction {
    /// `action.strip().lower()`, with Python's whitespace.
    pub fn parse(action: &str) -> Result<Self> {
        Ok(match py_strip(action).to_lowercase().as_str() {
            "press" => PointerAction::Press, "move" => PointerAction::Move, "release" => PointerAction::Release,
            "scroll" => PointerAction::Scroll, "key" => PointerAction::Key, "cancel" => PointerAction::Cancel, "leave" => PointerAction::Leave,
            _ => return Err(PlotError::value(format!("unknown raster pointer action {}", crate::configuration::value::repr_str(action)))),
        })
    }
}

/// One pointer message.
#[derive(Clone, Debug)]
pub struct PointerEvent {
    pub action: PointerAction,
    /// Canvas fractions, top origin.
    pub x: f64,
    pub y: f64,
    /// 1 left, 2 middle, 3 right.
    pub button: Option<u8>,
    pub double: bool,
    pub step: f64,
    pub key: Option<String>,
    /// The axes the frontend hit, as the session described them.
    pub axes: Option<AxisTransform>,
    /// When the input happened (seconds of [`monotonic`]), for a replayed
    /// press; `None` is now.
    pub time: Option<f64>,
}

/// `int(pixel)`: a whole pixel, refusing an infinite one as Python does.
fn whole(value: f64) -> Result<f64> {
    if !value.is_finite() { return Err(PlotError::new(ErrorKind::Overflow, "cannot convert float infinity to integer")); }
    Ok(value.trunc() + 0.0)
}

/// The session's clock (`time.monotonic`), in seconds.
pub fn monotonic() -> f64 {
    static START: OnceLock<Instant> = OnceLock::new();
    START.get_or_init(Instant::now).elapsed().as_secs_f64()
}

impl PlotSession {
    /// `_raster_pointer_event`: route one pointer message; whether the
    /// front's pixels changed.
    pub fn raster_pointer_event(&mut self, event: &PointerEvent) -> Result<bool> {
        if !(event.x.is_finite() && event.y.is_finite()) { return Err(PlotError::value("raster pointer coordinates must be finite")); }
        if event.button.is_some_and(|button| !(1..=3).contains(&button)) {
            return Err(PlotError::value("raster pointer button must be 1, 2, 3, or None"));
        }
        self.assert_open_session()?;
        let epoch = self.scene.presentation_epoch;
        let generation = self.renderer.raster_generation();
        let canvas = self.renderer.canvas_size();
        let pixel = (event.x * canvas.width, (1.0 - event.y) * canvas.height);
        let event_axes = match &event.axes {
            Some(transform) => self.axis_for_transform(transform),
            None => self.renderer.interactive_axes_at(pixel.0, pixel.1),
        };
        let interaction = self.services.interaction;
        let mut series_changed = false;
        let mut series = |session: &mut PlotSession, action: SeriesAction, axes: Option<&AxesId>| -> Result<bool> {
            let changed = session.renderer.series_focus(action, axes, pixel.0, pixel.1, interaction.selector_handle_radius_px,
                                                        interaction.double_click_radius_px)?;
            series_changed |= changed;
            Ok(changed)
        };
        match (event.action, event.button) {
            (PointerAction::Cancel, _) => { series(self, SeriesAction::Clear, None)?; self.cancel_gesture()?; }
            (PointerAction::Leave, _) => { series(self, SeriesAction::Leave, None)?; }
            (PointerAction::Key, _) => {
                let key = event.key.clone().unwrap_or_default();
                if key.to_lowercase() == "escape" {
                    series(self, SeriesAction::Clear, None)?;
                    self.cancel_gesture()?;
                    if self.scene.spec.is_facet_grid() && self.scene.facet_focus_index.is_some() { self.show_facet_overview()?; }
                }
            }
            (PointerAction::Scroll, _) => {
                if event.step == 0.0 { return Ok(false); }
                if !self.renderer.series_focus_scroll(event_axes.as_ref(), event.step)? {
                    whole(pixel.0)?;
                    whole(pixel.1)?;
                    self.on_scroll(event.axes.as_ref(), event.step)?;
                }
            }
            (PointerAction::Move, None) => { series(self, SeriesAction::Move, event_axes.as_ref())?; }
            (action, button) => {
                let whole_pixel = (whole(pixel.0)?, whole(pixel.1)?);
                match action {
                    PointerAction::Press => {
                        if button == Some(1) && !event.double { series(self, SeriesAction::Press, event_axes.as_ref())?; }
                        self.on_button_press(button, event.double, whole_pixel, event.axes.as_ref(), event.time.unwrap_or_else(monotonic))?;
                    }
                    PointerAction::Release => {
                        let handled = button == Some(1) && series(self, SeriesAction::Release, event_axes.as_ref())?;
                        if handled { self.cancel_gesture()?; } else { self.on_button_release(button, whole_pixel)?; }
                    }
                    _ => self.on_motion(whole_pixel)?,
                }
            }
        }
        let interaction_state = self.renderer.series_interaction();
        let committed = self.scene.display.commit_interaction(&interaction_state);
        if series_changed || committed { self.render_current(RenderEffect::OVERLAY)?; }
        Ok(self.scene.presentation_epoch != epoch || self.renderer.raster_generation() != generation)
    }

    /// `_axis_for_transform`: the renderer's axes of a transform's role and cell.
    fn axis_for_transform(&self, transform: &AxisTransform) -> Option<AxesId> {
        self.renderer.axes(&transform.role).into_iter().find(|axes| axes.cell_index == transform.cell_index)
    }

    fn histogram(&self) -> bool { self.semantic_kind() == PlotKind::Histogram }

    /// `_on_button_press`.
    fn on_button_press(&mut self, button: Option<u8>, double: bool, pixel: (f64, f64), transform: Option<&AxisTransform>, now: f64) -> Result<()> {
        let Some(button) = button else { return Ok(()) };
        let Some(transform) = transform else { return Ok(()) };
        let event_axes = self.axis_for_transform(transform);
        let is_double = self.click_history.is_double_click(button, double, None, Some(pixel), now, &self.services.interaction);
        self.cancel_gesture()?;
        let active = self.primary_axes();
        let on_active = event_axes.as_ref() == Some(&active);
        if self.scene.spec.is_facet_grid() {
            if self.scene.facet_focus_index.is_none() {
                let Some(cell) = event_axes.as_ref().and_then(|axes| self.renderer.facet_index_for_axes(axes)) else { return Ok(()) };
                if button == 1 && is_double { self.focus_facet(cell)?; }
                return Ok(());
            } else if button == 1 && is_double && on_active {
                return self.show_facet_overview();
            }
        }
        let camera = if self.renderer.height_bars_scene() { self.renderer.height_bars_camera() } else { None };
        if let (Some(camera), true) = (camera, on_active) {
            if button == 2 && is_double {
                let schema = self.parameter_schema().clone();
                let values = ValueMap::from_pairs(["camera_azimuth", "camera_elevation", "camera_zoom"].map(|name|
                    (name, schema.get(name).map(|spec| spec.default.clone()).unwrap_or(Value::None))));
                return self.configure_parameters(values);
            }
            if button == 2 {
                self.scene.gesture = Some(PointerGesture::Orbit(OrbitGesture { axes: active, transform: transform.clone(), origin_px: pixel,
                                                                                start: camera, current: camera }));
                return self.renderer.set_height_bars_dragging(true);
            }
            if button == 1 && !is_double {
                self.scene.gesture = Some(PointerGesture::Pick(PickGesture { axes: active, transform: transform.clone(), origin_px: pixel }));
            }
            return Ok(());
        }
        if button == 1 && is_double { return Ok(()); }
        let canvas = self.renderer.canvas_size();
        if button == 2 {
            if !on_active { return Ok(()); }
            if is_double { return self.zoom_to_selection_or_reset(); }
            let Some(origin) = transform.display_point(Some(pixel), canvas)? else { return Ok(()) };
            let (x0, x1) = sorted2(transform.x_limits.0, transform.x_limits.1);
            let (y0, y1) = sorted2(transform.y_limits.0, transform.y_limits.1);
            self.scene.gesture = Some(PointerGesture::Pan(PanGesture { axes: active.clone(), transform: transform.clone(), origin,
                x: NumericRange::new(x0, x1)?, y: NumericRange::new(y0, y1)?, candidate: None }));
            return self.renderer.set_view_dragging(Some(&active));
        }
        let Some(point) = event_canonical(Some(transform), Some(pixel), canvas, None, self.histogram())? else { return Ok(()) };
        if button == 3 {
            if !on_active { return Ok(()); }
            let projected = self.projected();
            let current = projected.context().selector_state(SelectorKind::Crosshair);
            if is_double {
                if current.is_some() { self.remove_selector(SelectorKind::Crosshair)?; }
                return Ok(());
            }
            let state = SelectorState::new(SelectorValue::Crosshair(point), 0, self.scene.focused_facet_index.map(|i| i as u64))?;
            self.install_selector_state(state, false)?;
            return Ok(());
        }
        if event_axes.as_ref().is_some_and(|axes| self.renderer.axes("distribution").contains(axes)) {
            return self.begin_color_limit_pointer(point.x, event_axes.unwrap(), transform);
        }
        let states = self.resolved_selector_snapshot()?.states();
        let projected = self.projected();
        let focused = self.scene.focused_facet_index.map(|i| i as u64);
        let handle_radius = self.services.interaction.selector_handle_radius_px * self.renderer.plan().device_pixel_ratio;
        let hit = hit_selector(point, &states, |state| self.selector_axes(state) == event_axes, focused, transform,
                               self.services.interaction.selector_hit_radius_fraction, self.histogram(), |state| {
                                   let displayed = projected.display_selector_state(state)?;
                                   let SelectorValue::Area(area) = displayed.value else { return Ok(None) };
                                   session_area_hit(area, Some(pixel), transform, canvas, handle_radius)
                               })?;
        let gesture = selector_press(point, pixel, hit, on_active, focused, transform, |state| self.selector_axes(state))?;
        if let Some(gesture) = gesture { self.scene.gesture = Some(PointerGesture::Selector(gesture)); }
        Ok(())
    }

    /// `_color_selector_domain` and `_current_color_limit_value`, then the drag.
    fn begin_color_limit_pointer(&mut self, point_x: f64, axes: AxesId, transform: &AxisTransform) -> Result<()> {
        if self.semantic_kind() != PlotKind::Image { return Ok(()); }
        let rail = self.renderer.axes("distribution");
        let first = rail.first().ok_or_else(|| PlotError::runtime("color selector requires a side distribution"))?;
        let geometry = self.renderer.geometry(first).ok_or_else(|| PlotError::runtime("color selector requires a side distribution"))?;
        let quantity = self.projected().value_quantity();
        let canonical = |(low, high): (f64, f64)| -> Result<NumericRange> {
            let (low, high) = sorted2(low, high);
            let (low, high) = sorted2(quantity.to_canonical(low), quantity.to_canonical(high));
            NumericRange::new(low, high)
        };
        let bounds = canonical(geometry.y_limits)?;
        let limits = self.renderer.resolved_color_limits().ok_or_else(|| PlotError::type_error("color limits require an active Image artist"))?;
        let value = canonical(limits)?;
        let Some(drag) = begin_color_limit_drag(bounds, value, point_x, self.services.interaction.selector_hit_radius_fraction)? else { return Ok(()) };
        self.scene.gesture = Some(PointerGesture::Color(ColorGesture { axes, transform: transform.clone(), drag }));
        let candidate = self.display_color_candidate().expect("colour gesture")?;
        self.renderer.begin_color_limit_gesture(candidate)
    }

    /// `_display_color_limit_candidate`.
    fn display_color_candidate(&self) -> Option<Result<NumericRange>> {
        let Some(PointerGesture::Color(gesture)) = &self.scene.gesture else { return None };
        Some(self.projected().value_quantity().range_to_display(gesture.drag.candidate))
    }

    /// `_zoom_to_selection_or_reset`.
    fn zoom_to_selection_or_reset(&mut self) -> Result<()> {
        let projected = self.projected();
        let pulse = projected.is_pulse();
        let zoom = zoom_to_selection(|kind| projected.context().selector_state(kind), self.scene.focused_facet_index.map(|i| i as u64),
                                     |state| Ok(if pulse { self.pulse_display_selector_state(state)? } else { projected.display_selector_state(state)? }.value))?;
        match zoom {
            SelectionZoom::Viewport(x, y) => self.set_viewport(Some(x), Some(y)),
            SelectionZoom::XLimits(low, high) => self.set_x_limits(low, high),
            SelectionZoom::Reset => self.set_viewport_state(None).map(|_| ()),
        }
    }

    /// `_on_scroll`.
    fn on_scroll(&mut self, transform: Option<&AxisTransform>, step: f64) -> Result<()> {
        let Some(transform) = transform else { return Ok(()) };
        self.cancel_gesture()?;
        let axes = self.axis_for_transform(transform);
        let active = self.primary_axes();
        if self.scene.spec.is_facet_grid() && self.scene.facet_focus_index.is_none() { return Ok(()); }
        if axes.as_ref() != Some(&active) { return Ok(()); }
        let up = step > 0.0;
        let ticks = wheel_ticks(step, up);
        let factor = self.services.interaction.wheel_zoom_factor;
        if self.renderer.height_bars_scene() {
            let zoom = self.display_state().get("camera_zoom").and_then(Value::real).unwrap_or(1.0);
            let zoomed = wheel_camera_zoom(zoom, factor, ticks)?;
            return self.configure_parameters(ValueMap::from_pairs([("camera_zoom", Value::Float(zoomed))]));
        }
        let geometry = self.renderer.geometry(&active).ok_or_else(|| PlotError::runtime("active axes have no geometry"))?;
        // Navigation stays continuous in display coordinates. Snapping the
        // viewport to source pixels makes small notches disappear, especially
        // near a cell; raster sampling must not quantize the user's view.
        // Start from the committed view so bursts also work before a redraw.
        let base = self.scene.viewport.unwrap_or((None, None));
        let base_x = base.0.map(|x| self.viewport_x_to_axes(x)).transpose()?;
        let (x, y) = wheel_viewport(base_x, base.1, geometry.x_limits, geometry.y_limits, (&geometry.x_scale, &geometry.y_scale),
                                    self.semantic_kind() == PlotKind::Image, factor, ticks, None)?;
        let request = (Some(self.viewport_x_from_axes(x)?), y);
        self.set_viewport(request.0, request.1)?;
        Ok(())
    }

    /// `_on_motion`.
    fn on_motion(&mut self, pixel: (f64, f64)) -> Result<()> {
        let canvas = self.renderer.canvas_size();
        let histogram = self.histogram();
        match self.scene.gesture.take() {
            None => Ok(()),
            Some(PointerGesture::Pan(mut gesture)) => {
                let moved = self.follow_pan(&mut gesture, pixel);
                self.scene.gesture = Some(PointerGesture::Pan(gesture));
                if moved? { self.render_current(RenderEffect::AXIS_TRANSFORM | RenderEffect::OVERLAY)?; }
                Ok(())
            }
            Some(PointerGesture::Orbit(mut gesture)) => {
                let camera = gesture.follow(pixel);
                self.scene.gesture = Some(PointerGesture::Orbit(gesture));
                let camera: CameraAngles = camera?;
                self.configure_parameters(ValueMap::from_pairs([("camera_azimuth", Value::Float(camera.azimuth_deg)),
                                                                ("camera_elevation", Value::Float(camera.elevation_deg))]))
            }
            Some(PointerGesture::Color(mut gesture)) => {
                let moved = color_motion(&mut gesture, pixel, canvas);
                self.scene.gesture = Some(PointerGesture::Color(gesture));
                if moved? {
                    let candidate = self.display_color_candidate().expect("colour gesture")?;
                    self.renderer.preview_color_limit_candidate(candidate)?;
                }
                Ok(())
            }
            Some(PointerGesture::Selector(mut gesture)) => {
                let motion = selector_motion(&mut self.scene.selectors, &mut gesture, pixel, canvas, histogram);
                let kind = gesture.kind;
                self.scene.gesture = Some(PointerGesture::Selector(gesture));
                if motion.started { self.renderer.begin_selector_gesture(kind, false)?; }
                match motion.step? {
                    SelectorStep::Unread | SelectorStep::Armed | SelectorStep::Unchanged => Ok(()),
                    SelectorStep::Cancel => self.cancel_gesture().map(|_| ()),
                    SelectorStep::Preview(updated) => {
                        if kind == SelectorKind::Threshold {
                            let labels = self.classifier_frame_labels()?.1;
                            self.renderer.set_classifier_labels(&labels);
                        }
                        let painted = self.painted_selector_state(&updated)?;
                        self.renderer.preview_selector(&painted)?;
                        Ok(())
                    }
                }
            }
            Some(other) => { self.scene.gesture = Some(other); Ok(()) }
        }
    }

    /// `_update_pan`: follow the pan; whether the view moved.
    fn follow_pan(&self, gesture: &mut PanGesture, pixel: (f64, f64)) -> Result<bool> {
        let canvas = self.renderer.canvas_size();
        let image = self.semantic_kind() == PlotKind::Image;
        let committed = self.projected().viewport();
        update_pan(gesture, Some(pixel), canvas, image, self.scene.viewport.and_then(|view| view.1), committed,
                   |x| self.viewport_x_from_axes(x), |selected| self.committed_view(selected))
    }

    /// `_on_button_release`.
    fn on_button_release(&mut self, button: Option<u8>, pixel: (f64, f64)) -> Result<()> {
        let canvas = self.renderer.canvas_size();
        let histogram = self.histogram();
        let Some(gesture) = self.scene.gesture.clone() else { return Ok(()) };
        match gesture {
            PointerGesture::Orbit(_) => {
                self.clear_gesture()?;
                self.renderer.set_height_bars_dragging(false)?;
                self.renderer.capture_gesture_background()
            }
            PointerGesture::Pan(mut pan) if button == Some(2) => {
                let moved = self.follow_pan(&mut pan, pixel);
                self.scene.gesture = Some(PointerGesture::Pan(pan.clone()));
                self.clear_gesture()?;
                moved?;
                if pan.candidate.is_some() { self.set_viewport_state(pan.candidate)?; }
                Ok(())
            }
            _ if button == Some(2) => Ok(()),
            PointerGesture::Pan(_) => Ok(()),
            PointerGesture::Pick(pick) => {
                self.clear_gesture()?;
                if !is_pick_click(pick.origin_px, pixel, &self.services.interaction) { return Ok(()); }
                if let Some((x, y)) = self.renderer.height_bars_pick(pixel.0, pixel.1) {
                    self.set_crosshair_selector(x, y)?;
                } else if self.projected().context().selector_state(SelectorKind::Crosshair).is_some() {
                    self.remove_selector(SelectorKind::Crosshair)?;
                } else {
                    self.render_current(RenderEffect::BASE_GEOMETRY)?;
                }
                Ok(())
            }
            PointerGesture::Color(mut color) => {
                let release = color_release(&mut color, pixel, canvas)?;
                if release == ColorRelease::Cancel { return self.cancel_gesture().map(|_| ()); }
                self.scene.gesture = Some(PointerGesture::Color(color));
                let candidate = self.display_color_candidate().expect("colour gesture")?;
                self.clear_gesture()?;
                let outcome = match release {
                    ColorRelease::Changed(_) => self.set_color_limits(candidate.low, candidate.high),
                    _ => self.render_current(RenderEffect::BASE_GEOMETRY),
                };
                if let Err(error) = outcome {
                    self.render_current(RenderEffect::BASE_GEOMETRY)?;
                    return Err(error);
                }
                Ok(())
            }
            PointerGesture::Selector(mut selector) => {
                let release = selector_release(&mut self.scene.selectors, &mut selector, pixel, canvas, histogram);
                self.scene.gesture = Some(PointerGesture::Selector(selector.clone()));
                let release = match release {
                    Err(error) => { self.cancel_gesture()?; return Err(error); }
                    Ok(SelectorRelease::Cancel) => return self.cancel_gesture().map(|_| ()),
                    Ok(release) => release,
                };
                self.clear_gesture()?;
                match release {
                    SelectorRelease::BlankClick => {
                        if self.projected().context().selector_state(selector.kind).is_some() { self.remove_selector(selector.kind)?; }
                        Ok(())
                    }
                    SelectorRelease::Abandoned | SelectorRelease::Nothing | SelectorRelease::Cancel => Ok(()),
                    SelectorRelease::Commit(candidate) => { self.install_selector_state(candidate, true)?; Ok(()) }
                    SelectorRelease::Degenerate(candidate) => {
                        if self.projected().context().selector_state(candidate.kind()).is_some() {
                            self.remove_selector(candidate.kind())?;
                            return Ok(());
                        }
                        self.render_current(RenderEffect::OVERLAY)
                    }
                }
            }
        }
    }

    /// `_clear_gesture`: the gesture is over; lift its renderer partition.
    fn clear_gesture(&mut self) -> Result<()> {
        let Some(gesture) = self.scene.gesture.take() else { return Ok(()) };
        match gesture {
            PointerGesture::Pan(_) => self.renderer.set_view_dragging(None),
            PointerGesture::Selector(selector) if !selector.started => Ok(()),
            _ => { self.renderer.end_selector_gesture(); Ok(()) }
        }
    }

    /// `_cancel_gesture`: drop the gesture, repainting what it had changed.
    pub(crate) fn cancel_gesture(&mut self) -> Result<Option<SelectorState>> {
        let Some(gesture) = self.scene.gesture.clone() else { return Ok(None) };
        let cancelled = match &gesture {
            PointerGesture::Selector(selector) => cancel_selector_gesture(&mut self.scene.selectors, selector),
            PointerGesture::Orbit(_) => self.renderer.set_height_bars_dragging(false).map(|_| None),
            _ => Ok(None),
        };
        self.clear_gesture()?;
        let cancelled = cancelled?;
        if matches!(&gesture, PointerGesture::Pan(pan) if pan.candidate.is_some()) && !self.closed {
            self.render_current(RenderEffect::AXIS_TRANSFORM | RenderEffect::OVERLAY)?;
        }
        let color = matches!(gesture, PointerGesture::Color(_));
        if (color || cancelled.is_some()) && !self.closed {
            self.render_current(if color { RenderEffect::BASE_GEOMETRY } else { RenderEffect::OVERLAY })?;
        }
        Ok(cancelled)
    }
}
