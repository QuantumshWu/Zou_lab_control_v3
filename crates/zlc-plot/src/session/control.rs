//! The session's own edits (`PlotSession` setters the pointer reaches):
//! display parameters through the one configure path, selectors, the
//! continuous viewport and FacetGrid focus.

use std::sync::Arc;

use crate::configuration::error::{PlotError, Result};
use crate::configuration::parameters::RenderEffect;
use crate::configuration::resolve::ConfigurationUpdates;
use crate::configuration::selectors::{normalize_viewport, CrosshairPoint, NumericRange, SelectorKind, SelectorState, SelectorValue, Viewport};
use crate::configuration::value::{Value, ValueMap};
use crate::interaction::gesture::require_stable_selector;
use crate::interaction::pymath::sorted2;
use crate::kinds::PlotKind;
use crate::pulse::PulseTimelineData;

use super::core::PlotSession;
use super::projected::PlotInput;
use super::evaluation::classifier_threshold_target;

impl PlotSession {
    /// `_configure(parameters=values)`: resolve, prepare and accept through
    /// the asynchronous path; a failure restores the scene.
    pub fn configure_parameters(&mut self, values: ValueMap) -> Result<()> {
        let owns = self.pending_presentation().is_none();
        if owns { self.begin_presentation()?; }
        let result = (|| -> Result<()> {
            let input = self.data();
            let updates = ConfigurationUpdates { parameters: Some(values), ..Default::default() };
            let candidate = self.resolve_configuration(None, None, None, None, None, true, None, None, Some(updates), None)?;
            let packet = self.prepare_data(&input, Some(self.data_revision()), Some(candidate.clone()))?;
            self.present_prepared(packet, Some(candidate), false).map(|_| ())
        })();
        match result {
            Ok(()) => { if owns { self.finish_presentation(true, None); } Ok(()) }
            Err(error) => { self.finish_presentation(false, None); Err(error) }
        }
    }

    /// `update_data(data, revision=)` of a pulse timeline: a new authored
    /// timeline at a revision past the current one (by default the next).
    pub fn update_pulse(&mut self, data: Arc<PulseTimelineData>, revision: Option<u64>) -> Result<()> {
        if !self.scene.projection.is_pulse() { return Err(PlotError::type_error(format!(
            "{} requires zlc_data.OwnedSnapshot", crate::configuration::semantics::spec_type_name(&self.scene.spec)))); }
        self.assert_open_session()?;
        let selected = revision.unwrap_or(self.data_revision() + 1);
        if selected <= self.data_revision() {
            return Err(PlotError::value(format!("data revision must increase: {selected} <= {}", self.data_revision())));
        }
        let mut updates = ConfigurationUpdates { image_overlay: Some(None), ..Default::default() };
        if self.scene.live_fit_request.is_none() { updates.fit = Some(ValueMap::new()); updates.fit_live = Some(false); }
        let candidate = self.resolve_configuration(None, None, None, None, None, true, None, None, Some(updates), None)?;
        let owns = self.pending_presentation().is_none();
        if owns { self.begin_presentation()?; }
        let result = self.prepare_data(&PlotInput::Pulse(data), Some(selected), Some(candidate.clone()))
            .and_then(|packet| self.present_prepared(packet, Some(candidate), true).map(|_| ()));
        match result {
            Ok(()) => { if owns { self.finish_presentation(true, None); } Ok(()) }
            Err(error) => { self.finish_presentation(false, None); Err(error) }
        }
    }

    /// `set_size(preset)`: relayout at another preset, restored when it fails.
    pub fn set_size(&mut self, preset: &str) -> Result<()> {
        let selected = self.services.defaults.layout.validate_preset(preset).map_err(PlotError::value)?;
        self.assert_open_session()?;
        if selected == self.scene.size { return Ok(()); }
        self.cancel_gesture()?;
        let previous = (self.scene.size, self.scene.layout_revision, self.renderer.plan().clone());
        self.scene.size = selected;
        let plan = match self.resolve_plan() {
            Ok(plan) => plan,
            Err(error) => { self.scene.size = previous.0; return Err(error); }
        };
        self.scene.layout_revision += 1;
        self.relayout_or_restore(plan, previous.2, |session| (session.scene.size, session.scene.layout_revision) = (previous.0, previous.1))
    }

    /// `set_device_pixel_ratio(ratio)`: relayout at another ratio, restored when it fails.
    pub fn set_device_pixel_ratio(&mut self, ratio: f64) -> Result<()> {
        if !ratio.is_finite() || ratio <= 0.0 { return Err(PlotError::value("device pixel ratio must be a positive finite number")); }
        self.assert_open_session()?;
        if ratio == self.device_pixel_ratio { return Ok(()); }
        self.cancel_gesture()?;
        let previous = (self.device_pixel_ratio, self.scene.layout_revision, self.renderer.plan().clone());
        self.device_pixel_ratio = ratio;
        let plan = match self.resolve_plan() {
            Ok(plan) => plan,
            Err(error) => { self.device_pixel_ratio = previous.0; return Err(error); }
        };
        self.scene.layout_revision += 1;
        self.relayout_or_restore(plan, previous.2, |session| (session.device_pixel_ratio, session.scene.layout_revision) = (previous.0, previous.1))
    }

    fn relayout_or_restore(&mut self, plan: crate::layout::SurfacePlan, previous: crate::layout::SurfacePlan, restore: impl FnOnce(&mut Self))
                           -> Result<()> {
        let Err(error) = self.apply_layout_plan(plan, true, None) else { return Ok(()) };
        restore(self);
        if self.apply_layout_plan(previous, true, None).is_err() { let _ = self.redraw_surface(); }
        Err(error)
    }

    /// `reset_viewport()`.
    pub fn reset_viewport(&mut self) -> Result<()> { self.set_viewport_state(None).map(|_| ()) }

    /// `set_color_limits(low, high, fixed=True)` in display units.
    pub(crate) fn set_color_limits(&mut self, low: f64, high: f64) -> Result<()> {
        if !self.parameter_schema().contains("color_min") { return Err(PlotError::type_error("this plot kind has no color limits")); }
        self.configure_parameters(ValueMap::from_pairs([("color_min", Value::Float(low)), ("color_max", Value::Float(high)),
                                                        ("relim_mode", Value::text("fixed"))]))
    }

    // ------------------------------------------------------------- selectors
    /// `_install_selector_state(state, finished_gesture=)`: replace the one
    /// selector of `state`'s kind and repaint.
    pub(crate) fn install_selector_state(&mut self, state: SelectorState, finished: bool) -> Result<SelectorState> {
        require_stable_selector(&state)?;
        self.assert_open_session()?;
        if !finished && matches!(self.scene.gesture, Some(crate::interaction::engine::PointerGesture::Selector(_))) { self.cancel_gesture()?; }
        let fit_previous = self.scene.fit_context_generation;
        let (previous, stored) = if finished { self.scene.selectors.commit_finished(state)? } else { self.scene.selectors.install(state)? };
        let classifier_previous = self.scene.classifier_thresholds.clone();
        if stored.kind() == SelectorKind::Threshold && self.threshold_classifier_enabled() {
            if let Some(index) = self.classifier_distribution_index(stored.state().facet_index.map(|i| i as usize))? {
                if index < self.scene.classifier_results.len() {
                    let projected = self.projected();
                    let target = &super::evaluation::classifier_targets(&projected, &self.authored_classifier_keys())?[index];
                    let key = classifier_threshold_target(&projected, target, None, None)?.key_repr();
                    if let SelectorValue::Threshold(level) = stored.state().value { self.scene.classifier_thresholds.insert(key, Some(level)); }
                }
            }
        }
        if self.selector_change_affects_fit(stored.kind()) { self.scene.fit_context_generation += 1; }
        if let Err(error) = self.render_current(RenderEffect::OVERLAY) {
            self.scene.selectors.rollback_install(&stored, previous)?;
            self.scene.classifier_thresholds = classifier_previous;
            self.scene.fit_context_generation = fit_previous;
            if self.render_current(RenderEffect::OVERLAY).is_err() { let _ = self.redraw_surface(); }
            return Err(error);
        }
        Ok(stored.state())
    }

    /// `remove_selector(kind)`.
    pub(crate) fn remove_selector(&mut self, kind: SelectorKind) -> Result<SelectorState> {
        self.assert_open_session()?;
        let state = self.scene.selectors.state(kind)?;
        if matches!(&self.scene.gesture, Some(crate::interaction::engine::PointerGesture::Selector(gesture)) if gesture.kind == kind) {
            self.cancel_gesture()?;
        }
        let previous_fit = (self.scene.fit_context_generation, self.scene.fit_request_generation, self.scene.live_fit_request.clone(),
                            self.scene.accepted_fit.clone());
        let bound = self.scene.live_fit_request.as_ref().is_some_and(|request| request.selector_kind
            .is_some_and(|held| super::evaluation::selector_kind(held) == kind));
        let affects_fit = self.selector_change_affects_fit(kind);
        let removed = self.scene.selectors.remove(kind)?;
        let chosen_previous = self.scene.classifier_thresholds.clone();
        if kind == SelectorKind::Threshold && !self.scene.classifier_thresholds.is_empty() {
            if let Some(index) = self.classifier_distribution_index(state.facet_index.map(|i| i as usize))? {
                let projected = self.projected();
                let target = &super::evaluation::classifier_targets(&projected, &self.authored_classifier_keys())?[index];
                let key = classifier_threshold_target(&projected, target, None, None)?.key_repr();
                self.scene.classifier_thresholds.remove(&key);
            }
        }
        if affects_fit {
            self.scene.fit_context_generation += 1;
            if bound {
                self.scene.fit_request_generation += 1;
                self.scene.fit_warm_starts.clear();
                self.scene.live_fit_request = None;
                self.scene.accepted_fit = None;
            }
        }
        if let Err(error) = self.render_current(RenderEffect::OVERLAY) {
            self.scene.selectors.restore_removed(removed)?;
            self.scene.classifier_thresholds = chosen_previous;
            (self.scene.fit_context_generation, self.scene.fit_request_generation, self.scene.live_fit_request, self.scene.accepted_fit) = previous_fit;
            if self.render_current(RenderEffect::OVERLAY).is_err() { let _ = self.redraw_surface(); }
            return Err(error);
        }
        Ok(state)
    }

    fn authored_classifier_keys(&self) -> std::collections::BTreeSet<String> {
        self.scene.classifier_thresholds.keys().chain(self.scene.classifier_components.keys()).cloned().collect()
    }

    /// `set_crosshair_selector(x, y)` in display units.
    pub(crate) fn set_crosshair_selector(&mut self, x: f64, y: f64) -> Result<SelectorState> {
        self.assert_open_session()?;
        let projected = self.projected();
        let point = CrosshairPoint::new(projected.x_selector_quantity()?.to_canonical(x),
                                        if projected.is_histogram_plot() { y } else { projected.y_quantity()?.to_canonical(y) })?;
        let state = SelectorState::new(SelectorValue::Crosshair(point), 0, self.scene.focused_facet_index.map(|i| i as u64))?;
        self.install_selector_state(state, false)
    }

    // ------------------------------------------------------------- viewport
    /// `set_viewport(x, y)`.
    pub(crate) fn set_viewport(&mut self, x: Option<NumericRange>, y: Option<NumericRange>) -> Result<()> {
        self.set_viewport_state(normalize_viewport(Some((x, y))))?;
        Ok(())
    }

    /// `_viewport_x_to_axes`: a viewport's x in the axes' own units (a pulse
    /// timeline's axes are in its source time, its viewport in display time).
    pub(crate) fn viewport_x_to_axes(&self, value: NumericRange) -> Result<NumericRange> {
        if self.scene.projection.is_pulse() { self.projected().pulse_display_range_to_source(value) } else { Ok(value) }
    }

    /// `_viewport_x_from_axes`.
    pub(crate) fn viewport_x_from_axes(&self, value: NumericRange) -> Result<NumericRange> {
        if self.scene.projection.is_pulse() { self.projected().pulse_source_range_to_display(value) } else { Ok(value) }
    }

    /// `_special_display_selector_state`: a pulse selector's time in display units.
    pub(crate) fn pulse_display_selector_state(&self, state: &SelectorState) -> Result<SelectorState> {
        let projected = self.projected();
        let value = match state.value {
            SelectorValue::XRange(range) => SelectorValue::XRange(projected.pulse_source_range_to_display(range)?),
            SelectorValue::Area(area) => SelectorValue::Area(crate::configuration::selectors::RectangleRange {
                x: projected.pulse_source_range_to_display(area.x)?, y: area.y }),
            SelectorValue::Crosshair(point) => SelectorValue::Crosshair(CrosshairPoint::new(point.x * projected.pulse_x_factor()?, point.y)?),
            other => other,
        };
        Ok(SelectorState { value, ..*state })
    }

    /// `set_x_limits(low, high)`: the x range, the y view kept.
    pub fn set_x_limits(&mut self, low: f64, high: f64) -> Result<()> {
        let current = self.scene.viewport.unwrap_or((None, None));
        self.set_viewport(Some(NumericRange::new(low, high)?), current.1)
    }

    /// The viewport committing `selected` shows (`_set_viewport_state`'s
    /// normalized continuous view; sample selection is a separate concern).
    pub(crate) fn committed_view(&self, selected: Option<Viewport>) -> Result<Option<Viewport>> {
        let Some(mut selected) = normalize_viewport(selected) else { return Ok(None) };
        if self.semantic_kind() == PlotKind::Image && (selected.0.is_none() || selected.1.is_none()) {
            let (x, y) = self.current_display_limits()?;
            selected = (selected.0.or(Some(x)), selected.1.or(Some(y)));
        }
        Ok(Some(selected))
    }

    /// `_set_viewport_state`: commit the viewport and the fit authority
    /// after their frame draws.
    pub(crate) fn set_viewport_state(&mut self, selected: Option<Viewport>) -> Result<bool> {
        let selected = self.committed_view(selected)?;
        self.assert_open_session()?;
        if selected == self.scene.viewport { return Ok(false); }
        let previous = (self.scene.viewport, self.scene.fit_context_generation);
        self.scene.viewport = selected;
        self.scene.fit_context_generation += 1;
        let effects = RenderEffect::AXIS_TRANSFORM | RenderEffect::FIT_SELECTION | RenderEffect::OVERLAY;
        if let Err(error) = self.render_current(effects) {
            (self.scene.viewport, self.scene.fit_context_generation) = previous;
            if self.render_current(effects).is_err() { let _ = self.redraw_surface(); }
            return Err(error);
        }
        Ok(true)
    }

    /// `_current_display_limits`: the primary axes' sorted limits.
    pub(crate) fn current_display_limits(&self) -> Result<(NumericRange, NumericRange)> {
        let geometry = self.renderer.geometry(&self.renderer.primary_axes()).ok_or_else(|| PlotError::runtime("primary axes have no geometry"))?;
        let (x0, x1) = sorted2(geometry.x_limits.0, geometry.x_limits.1);
        let (y0, y1) = sorted2(geometry.y_limits.0, geometry.y_limits.1);
        Ok((NumericRange::new(x0, x1)?, NumericRange::new(y0, y1)?))
    }

    // ------------------------------------------------------------- facets
    /// `_select_facet(index)`: route cell-local state to `index`.
    fn select_facet(&mut self, index: usize) -> Result<bool> {
        if !self.scene.spec.is_facet_grid() { return Err(PlotError::type_error("facet selection is available only for FacetGridPlot")); }
        if index >= self.facet_cell_count().unwrap_or(0) { return Err(PlotError::index("facet index is outside the current grid")); }
        let changed = Some(index) != self.scene.focused_facet_index;
        if changed {
            self.cancel_gesture()?;
            self.scene.focused_facet_index = Some(index);
            self.retarget_selectors(index)?;
            self.scene.viewport = None;
            self.invalidate_fit_context();
        }
        Ok(changed)
    }

    /// `focus_facet(index)`: open one cell as the interactive surface.
    pub(crate) fn focus_facet(&mut self, index: usize) -> Result<()> {
        self.assert_open_session()?;
        let target_changed = self.select_facet(index)?;
        let presentation_changed = self.scene.facet_focus_index != Some(index);
        if presentation_changed {
            self.cancel_gesture()?;
            self.scene.facet_focus_index = Some(index);
        }
        if target_changed || presentation_changed {
            self.render_current(RenderEffect::BASE_GEOMETRY | RenderEffect::AXIS_TRANSFORM | RenderEffect::CHROME)?;
        }
        Ok(())
    }

    /// `show_facet_overview()`.
    pub(crate) fn show_facet_overview(&mut self) -> Result<()> {
        if !self.scene.spec.is_facet_grid() { return Err(PlotError::type_error("facet overview is available only for FacetGridPlot")); }
        self.assert_open_session()?;
        if self.scene.facet_focus_index.is_none() { return Ok(()); }
        self.cancel_gesture()?;
        let cleared = self.scene.viewport.is_some();
        self.scene.viewport = None;
        if cleared { self.invalidate_fit_context(); }
        self.scene.facet_focus_index = None;
        self.render_current(RenderEffect::BASE_GEOMETRY | RenderEffect::AXIS_TRANSFORM | RenderEffect::CHROME)
    }

    /// `_selector_axes(state)`: the axes a selector is painted on.
    pub(crate) fn selector_axes(&self, state: &SelectorState) -> Option<crate::interaction::engine::AxesId> {
        if self.scene.spec.is_facet_grid() {
            let axes = self.renderer.axes("facet_cell");
            let index = state.facet_index.map(|i| i as usize).or(self.scene.focused_facet_index)?;
            return axes.get(index).cloned();
        }
        Some(self.renderer.primary_axes())
    }

    pub(crate) fn assert_open_session(&self) -> Result<()> { if self.closed { Err(PlotError::runtime("plot session is closed")) } else { Ok(()) } }
}
