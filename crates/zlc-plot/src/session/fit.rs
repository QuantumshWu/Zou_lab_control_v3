//! The session's fit and classifier presentation (`_session_fit`): accepted
//! fit answers and their overlays, the threshold classifier's thresholds,
//! readings and derived selector, and the bounded fit table a reader asks
//! for.  Nothing here solves; every number comes from a preparation.

use std::sync::Arc;

use serde_json::{json, Value as Json};

use crate::configuration::error::{PlotError, Result};
use crate::configuration::selectors::{normalize_classifier_threshold_targets, thresholds_wire, SelectorKind, SelectorState, SelectorValue,
                                      ThresholdTarget};
use crate::configuration::value::Value;
use crate::fit::classifier::bimodal_classifier_metrics;
use crate::fit::target::LiveFitRequest;
use crate::fit::FitResult;
use crate::interaction::controller::SelectorSnapshot;
use crate::interaction::labels::series_identity;
use crate::pyfmt::format_g;

use super::core::{finite_or_null, AcceptedFit, FitAnswer, PlotSession};
use super::evaluation::{classifier_keys, classifier_targets, classifier_threshold_target, ClassifierEvaluation, FitEvaluation};
use super::fit_layer::fit_error;
use super::overlay::FitOverlay;
use super::projected::{scalar_value, FitCellTarget, FitProjection, FitSelection};

/// `_fit_overlay(projection, result, selection, target, failure)`: a
/// presentation failure never erases a computed number.
fn fit_overlay(projection: &FitProjection, result: Option<&FitResult>, selection: Option<&FitSelection>, target: &FitCellTarget,
               failure: Option<&str>) -> Result<FitOverlay> {
    let mut failure = failure.map(str::to_owned);
    if let (Some(result), Some(selection)) = (result, selection) {
        match projection.make_fit_overlay(result, selection) {
            Ok(overlay) => return Ok(overlay),
            Err(error) if error.kind == crate::configuration::error::ErrorKind::FitCancelled => return Err(error),
            Err(error) => failure = Some(super::evaluation::failure_text(&error)),
        }
    }
    Ok(match failure {
        None => FitOverlay { facet_index: target.facet_index, group_key: target.group_key.clone(), ..FitOverlay::default() },
        Some(failure) => FitOverlay { success: false, diagnostic: failure, facet_index: target.facet_index, group_key: target.group_key.clone(),
                                      ..FitOverlay::default() },
    })
}

/// Python's `round(value, 1)`.
fn round1(value: f64) -> f64 { format!("{value:.1}").parse().unwrap_or(value) }

impl PlotSession {
    // ------------------------------------------------------------- accepted fits
    /// `_invalid_fit_result`: the display shape of a failed single evaluation.
    fn invalid_fit_result(evaluation: &FitEvaluation) -> Result<FitResult> {
        let count = evaluation.model.parameters.len();
        let result = FitResult::new(evaluation.model.clone(), vec![f64::NAN; count], vec![f64::NAN; count], vec![f64::NAN; count * count],
                                    FitResult::eager(Vec::new(), Vec::new(), Vec::new()), evaluation.source_revision(), false,
                                    evaluation.failure_messages[0].as_deref().unwrap_or("fit failed"), 0.0, false, Vec::new())
            .map_err(fit_error)?;
        result.with_parameter_units(&evaluation.parameter_units).map_err(fit_error)
    }

    /// `_present_fit_evaluation`: adapt one numerical answer for painting.
    fn present_fit_evaluation(projection: &FitProjection, evaluation: &FitEvaluation) -> Result<(FitAnswer, Vec<FitOverlay>)> {
        if Some(&evaluation.source_ref) != projection.source_ref().as_ref() { return Err(PlotError::value("fit evaluation does not match this projection")); }
        let mut overlays = Vec::with_capacity(evaluation.targets.len());
        for (index, target) in evaluation.targets.iter().enumerate() {
            overlays.push(fit_overlay(projection, evaluation.results[index].as_ref(), evaluation.selections[index].as_deref(), target,
                                      evaluation.failure_messages[index].as_deref())?);
        }
        let answer = if evaluation.is_batch {
            FitAnswer::Batch { results: evaluation.results.clone(), failure_messages: evaluation.failure_messages.clone(),
                               source_revision: evaluation.source_revision(), parameter_units: evaluation.parameter_units.clone(),
                               sample_axes: evaluation.sample_axes.clone(), batch_revision: 0 }
        } else {
            FitAnswer::Single(match &evaluation.results[0] { Some(result) => result.clone(), None => Self::invalid_fit_result(evaluation)? })
        };
        Ok((answer, overlays))
    }

    /// `_resolved_accepted_fit`: the presentation of one numerical answer,
    /// numbered when accepted.
    pub(crate) fn resolved_accepted_fit(&mut self, evaluation: Arc<FitEvaluation>, request: LiveFitRequest, context_generation: u64,
                                        projection: &FitProjection) -> Result<AcceptedFit> {
        let (answer, overlays) = Self::present_fit_evaluation(projection, &evaluation)?;
        self.scene.fit_batch_revision += 1;
        let revision = self.scene.fit_batch_revision;
        let answer = match answer {
            FitAnswer::Single(result) => FitAnswer::Single(result.with_batch_revision(revision)),
            FitAnswer::Batch { results, failure_messages, source_revision, parameter_units, sample_axes, .. } =>
                FitAnswer::Batch { results, failure_messages, source_revision, parameter_units, sample_axes, batch_revision: revision },
        };
        let selection = if evaluation.is_batch { None } else { evaluation.selections[0].clone() };
        Ok(AcceptedFit {
            answer, selection, context_generation, source_generation: evaluation.source_ref.generation.clone(), request,
            selections: evaluation.selections.clone(), evaluation: Some(evaluation), overlays: overlays.into(),
        })
    }

    /// `_remember_fit_warm_starts`: successful parameters of the current request.
    pub(crate) fn remember_fit_warm_starts(&mut self, evaluation: &FitEvaluation, request_generation: i64) {
        if request_generation != self.scene.fit_request_generation { return; }
        let model = evaluation.model.model_id.clone();
        for (index, fit) in evaluation.results.iter().enumerate() {
            let key = (request_generation, model.clone(), evaluation.is_batch.then_some(index));
            match fit {
                Some(fit) if fit.success && !fit.reduced => { self.scene.fit_warm_starts.insert(key, fit.parameter_values.clone()); }
                _ => { self.scene.fit_warm_starts.remove(&key); }
            }
        }
    }

    /// `_refresh_accepted_fit_overlays`: repaint in the current display units.
    pub(crate) fn refresh_accepted_fit_overlays(&self, accepted: &AcceptedFit) -> Result<AcceptedFit> {
        let projection = self.projected();
        let overlays: Vec<FitOverlay> = match &accepted.answer {
            FitAnswer::Batch { results, .. } => {
                let mut refreshed = Vec::with_capacity(results.len());
                for ((result, selection), previous) in results.iter().zip(&accepted.selections).zip(accepted.overlays.iter()) {
                    match (result, selection) {
                        (Some(result), Some(selection)) => {
                            let mut context = (**projection.context()).clone();
                            context.focused_facet_index = selection.facet_index;
                            refreshed.push(projection.with_context(Arc::new(context)).make_fit_overlay(result, selection)?);
                        }
                        _ => refreshed.push(previous.clone()),
                    }
                }
                refreshed
            }
            FitAnswer::Single(result) => {
                let selection = accepted.selection.as_ref().ok_or_else(|| PlotError::runtime("single fit acceptance has no selection"))?;
                vec![projection.make_fit_overlay(result, selection)?]
            }
        };
        Ok(AcceptedFit { overlays: overlays.into(), ..accepted.clone() })
    }

    // ------------------------------------------------------------- classifier
    /// `_install_threshold_classifier`: present a numeric packet.
    pub(crate) fn install_threshold_classifier(&mut self, prepared: Arc<ClassifierEvaluation>) -> Result<()> {
        if !prepared.projection.same_built(&self.scene.projection) || !prepared.projection.same_payload(&self.scene.projection) {
            return Err(PlotError::value("classifier answer does not match the installed projection"));
        }
        if prepared.capture.model.is_none() {
            self.scene.classifier_evaluation = Some(prepared);
            self.scene.classifier_results = Arc::from(Vec::new());
            self.scene.classifier_overlays = Arc::from(Vec::new());
            self.scene.classifier_thresholds = Default::default();
            self.scene.classifier_components = Default::default();
            return Ok(());
        }
        let count = prepared.targets.len();
        let mut results: Vec<Option<FitResult>>;
        let mut overlays: Vec<Option<FitOverlay>>;
        if prepared.requested.len() == count {
            results = prepared.results.to_vec();
            overlays = vec![None; count];
        } else {
            if self.scene.classifier_results.len() != count { return Err(PlotError::value("partial classifier answer has no matching installed cells")); }
            results = self.scene.classifier_results.to_vec();
            overlays = self.scene.classifier_overlays.iter().cloned().map(Some).collect();
        }
        for index in prepared.requested.iter().copied() {
            results[index] = prepared.results[index].clone();
            overlays[index] = Some(fit_overlay(&prepared.projection, prepared.results[index].as_ref(), prepared.selections[index].as_deref(),
                                               &prepared.targets[index], prepared.failures[index].as_deref())?);
        }
        self.scene.classifier_results = results.into();
        self.scene.classifier_overlays = overlays.into_iter().map(|overlay| overlay.expect("every distribution painted")).collect::<Vec<_>>().into();
        self.scene.classifier_evaluation = Some(prepared);
        let _ = self.scene.selectors.remove(SelectorKind::Threshold);
        Ok(())
    }

    fn classifier_cell_targets(&self) -> Result<Vec<FitCellTarget>> {
        let authored = self.scene.classifier_thresholds.keys().chain(self.scene.classifier_components.keys()).cloned().collect();
        classifier_targets(&self.projected(), &authored)
    }

    /// `_classifier_thresholds_settled`: the choice, else the fit's threshold.
    pub(crate) fn classifier_thresholds_settled(&self) -> Result<Vec<Option<f64>>> {
        if self.scene.classifier_results.is_empty() { return Ok(Vec::new()); }
        let keys = classifier_keys(&self.projected(), &self.classifier_cell_targets()?)?;
        keys.iter().zip(self.scene.classifier_results.iter()).map(|(key, result)| match self.scene.classifier_thresholds.get(key) {
            Some(value) => Ok(*value),
            None => match result {
                Some(result) if result.success => Ok(bimodal_classifier_metrics(result, None).map_err(fit_error)?.0),
                _ => Ok(None),
            },
        }).collect()
    }

    fn classifier_thresholds_for_render(&self) -> Result<Vec<Option<f64>>> {
        let mut thresholds = self.classifier_thresholds_settled()?;
        if thresholds.is_empty() { return Ok(thresholds); }
        if let Some(candidate) = self.scene.selectors.candidate_state()?.filter(|state| state.kind() == SelectorKind::Threshold) {
            if let (Some(index), SelectorValue::Threshold(level)) = (self.classifier_distribution_index(candidate.facet_index.map(|i| i as usize))?, candidate.value) {
                if index < thresholds.len() { thresholds[index] = Some(level); }
            }
        }
        Ok(thresholds)
    }

    /// `_classifier_distribution_index(facet_index)`: the displayed group,
    /// through the series lock.
    pub(crate) fn classifier_distribution_index(&self, facet_index: Option<usize>) -> Result<Option<usize>> {
        let facet = if self.scene.spec.is_facet_grid() && facet_index.is_none() { Some(self.scene.focused_facet_index.unwrap_or(0)) } else { facet_index };
        let candidates: Vec<(usize, FitCellTarget)> = self.classifier_cell_targets()?.into_iter().enumerate()
            .filter(|(_, target)| target.facet_index == facet).collect();
        if let Some(locked) = &self.scene.display.state.interaction.series_lock {
            if locked.facet_index.map(|i| i as usize) == self.scene.focused_facet_index {
                for (index, target) in &candidates {
                    let key: Vec<(crate::kinds::AxisRef, Value)> = target.group_key.iter().map(|entry| (entry.axis.clone(), scalar_value(&entry.canonical))).collect();
                    let identity: Vec<[String; 3]> = series_identity(&key).into_iter().map(|(domain, axis, value)| [domain.to_owned(), axis, value]).collect();
                    let held: Vec<[String; 3]> = locked.key.iter().map(|part| [part[0].to_string(), part[1].to_string(), part[2].to_string()]).collect();
                    if identity == held { return Ok(Some(*index)); }
                }
            }
        }
        Ok(candidates.first().map(|(index, _)| *index))
    }

    fn classifier_labels(&self, thresholds: &[Option<f64>], display: &[Option<f64>]) -> Result<Vec<String>> {
        let mut labels = Vec::with_capacity(thresholds.len());
        for ((result, threshold), shown) in self.scene.classifier_results.iter().zip(thresholds).zip(display) {
            let (Some(threshold), Some(shown)) = (threshold, shown) else { labels.push(String::new()); continue };
            let reading = format!("Threshold {}", format_g(*shown, 4));
            let Some(result) = result.as_ref().filter(|result| result.success) else { labels.push(reading); continue };
            let (_, left, _, fidelity) = bimodal_classifier_metrics(result, Some(*threshold)).map_err(fit_error)?;
            let left_percent = round1(100.0 * left);
            let mut text = format!("{reading}\nL/R {}%/{}%", crate::fit::format_fixed(left_percent, 1), crate::fit::format_fixed(100.0 - left_percent, 1));
            if fidelity.is_finite() { text.push_str(&format!("\nFidelity {}%", crate::fit::format_fixed(100.0 * fidelity, 1))); }
            labels.push(text);
        }
        Ok(labels)
    }

    /// `_classifier_frame_labels`: each distribution's display threshold and reading.
    pub(crate) fn classifier_frame_labels(&self) -> Result<(Vec<Option<f64>>, Vec<String>)> {
        let thresholds = self.classifier_thresholds_for_render()?;
        // Asked per threshold, as Python's generator asks: no threshold, no quantity.
        let quantity = thresholds.iter().any(Option::is_some).then(|| self.projected().value_quantity());
        let display: Vec<Option<f64>> = thresholds.iter()
            .map(|value| value.map(|value| quantity.as_ref().expect("a threshold's quantity").to_display(value))).collect();
        let labels = self.classifier_labels(&thresholds, &display)?;
        Ok((display, labels))
    }

    /// `_derived_threshold_classifier_selector`.
    fn derived_threshold_selector(&self) -> Result<Option<SelectorState>> {
        let settled = self.classifier_thresholds_settled()?;
        if !self.threshold_classifier_enabled() || settled.is_empty() { return Ok(None); }
        let Some(index) = self.classifier_distribution_index(None)? else { return Ok(None) };
        let Some(Some(threshold)) = settled.get(index).copied() else { return Ok(None) };
        let facet = self.classifier_cell_targets()?[index].facet_index;
        Ok(Some(SelectorState::new(SelectorValue::Threshold(threshold), 0, facet.map(|i| i as u64))?))
    }

    /// `_classifier_threshold_targets_state(settled=)`.
    pub(crate) fn classifier_threshold_targets_state(&self, settled: bool) -> Result<Vec<ThresholdTarget>> {
        if !self.threshold_classifier_enabled() { return Ok(Vec::new()); }
        let projected = self.projected();
        let targets = self.classifier_cell_targets()?;
        let keys = classifier_keys(&projected, &targets)?;
        let values: Vec<Option<f64>> = if settled { self.classifier_thresholds_settled()? }
            else { keys.iter().map(|key| self.scene.classifier_thresholds.get(key).copied().flatten()).collect() };
        let subject = self.scene.projection.selection_subject(None)?;
        let mut records = Vec::new();
        for (index, (key, value)) in keys.iter().zip(values).enumerate() {
            if value.is_none() && !self.scene.classifier_thresholds.contains_key(key) { continue; }
            let mut target = classifier_threshold_target(&projected, &targets[index], value, Some(&subject))?;
            if let Some(Some(gaussian)) = self.scene.classifier_components.get(key) { target.gaussian_components = Some(*gaussian); }
            records.push(target);
        }
        normalize_classifier_threshold_targets(&thresholds_wire(&records))
    }

    // ------------------------------------------------------------- selectors
    /// `_resolved_selector_snapshot`: the sole painted/hit-tested state per kind.
    pub(crate) fn resolved_selector_snapshot(&self) -> Result<SelectorSnapshot> {
        let snapshot = self.scene.selectors.snapshot(true)?;
        if !self.threshold_classifier_enabled() { return Ok(snapshot); }
        let not_threshold = |state: &SelectorState| state.kind() != SelectorKind::Threshold;
        let committed: Vec<SelectorState> = snapshot.committed.iter().copied().filter(not_threshold).collect();
        let candidate = snapshot.candidate.filter(not_threshold);
        let Some(derived) = self.derived_threshold_selector()? else { return Ok(SelectorSnapshot { committed, candidate }) };
        let threshold = match snapshot.candidate {
            Some(held) if held.kind() == SelectorKind::Threshold && held.facet_index == derived.facet_index => held,
            _ => derived,
        };
        let mut committed = committed;
        committed.push(threshold);
        Ok(SelectorSnapshot { committed, candidate })
    }

    /// `_painted_selector_state`: a selector in the units its axes are drawn in.
    pub(crate) fn painted_selector_state(&self, state: &SelectorState) -> Result<SelectorState> { self.projected().display_selector_state(state) }

    /// `_painted_selector_snapshot`.
    pub(crate) fn painted_selector_snapshot(&self) -> Result<SelectorSnapshot> {
        let snapshot = self.resolved_selector_snapshot()?;
        let projected = self.projected();
        Ok(SelectorSnapshot {
            committed: snapshot.committed.iter().map(|state| projected.display_selector_state(state)).collect::<Result<_>>()?,
            candidate: snapshot.candidate.map(|state| projected.display_selector_state(&state)).transpose()?,
        })
    }

    /// `_fit_request_uses_selector`.
    pub(crate) fn fit_request_uses_selector(request: &LiveFitRequest, kind: SelectorKind) -> bool {
        match request.selector_kind {
            Some(held) => super::evaluation::selector_kind(held) == kind,
            None => matches!(kind, SelectorKind::Area | SelectorKind::XRange),
        }
    }

    /// `_selector_change_affects_fit`.
    pub(crate) fn selector_change_affects_fit(&self, kind: SelectorKind) -> bool {
        if let Some(request) = &self.scene.live_fit_request { return Self::fit_request_uses_selector(request, kind); }
        self.scene.accepted_fit.as_ref().is_some_and(|accepted| matches!(kind, SelectorKind::Area | SelectorKind::XRange)
            || accepted.selection.as_ref().is_some_and(|selection| selection.selector_kind == Some(kind)))
    }

    // ------------------------------------------------------------- fit table
    /// `fit_status`: whether the painted fit matches the data and context.
    fn fit_status(&self) -> Option<&'static str> {
        let accepted = self.scene.accepted_fit.as_ref()?;
        let current = Some(&*accepted.source_generation) == self.data_generation().as_deref() && accepted.answer.source_revision() == self.data_revision()
            && accepted.context_generation == self.scene.fit_context_generation;
        Some(if current { "current" } else { "lagging" })
    }

    /// `fit_summary(offset=, limit=)`: a bounded table of the accepted fit.
    pub fn fit_summary(&self, offset: i64, limit: i64) -> Result<Json> {
        if offset < 0 { return Err(PlotError::value("offset must be a non-negative integer")); }
        if !(1..=4096).contains(&limit) { return Err(PlotError::value("limit must be an integer from 1 to 4096")); }
        if self.closed { return Err(PlotError::runtime("plot session is closed")); }
        let (offset, limit) = (offset as usize, limit as usize);
        let generation = self.data_generation();
        let revision = self.data_revision();
        let configuration = self.fit_configuration();
        let Some(accepted) = &self.scene.accepted_fit else {
            let request = &self.scene.live_fit_request;
            return Ok(json!({
                "status": if request.is_none() { "not_enabled" } else { "pending" },
                "model_id": request.as_ref().map(|request| request.model.model_id.clone()),
                "rows": [], "total_rows": 0, "offset": offset, "next_offset": null,
                "data_generation": generation.as_deref(), "data_revision": revision,
            }));
        };
        let model = &accepted.request.model;
        let (outcomes, failures, axes, units, batch_revision, source_revision): (Vec<Option<&FitResult>>, Vec<Option<String>>, &[super::projected::SampleAxis], Vec<String>, u64, u64) =
            match &accepted.answer {
                FitAnswer::Single(result) => (vec![Some(result)], vec![None], &[], result.parameter_units.clone(), result.batch_revision, result.source_revision),
                FitAnswer::Batch { results, failure_messages, sample_axes, parameter_units, batch_revision, source_revision } =>
                    (results.iter().map(Option::as_ref).collect(), failure_messages.clone(), sample_axes,
                     model.parameters.iter().map(|p| parameter_units.iter().find(|(name, _)| *name == p.name).map(|(_, unit)| unit.clone()).unwrap_or_default()).collect(),
                     *batch_revision, *source_revision),
            };
        let fixed_authored: Vec<String> = configuration.get("fixed").and_then(Value::as_map).map(|map| map.keys().map(str::to_owned).collect()).unwrap_or_default();
        let stop = outcomes.len().min(offset + limit);
        let sizes: Vec<usize> = axes.iter().map(|axis| axis.size()).collect();
        let mut rows = Vec::new();
        for index in offset..stop {
            let outcome = outcomes[index];
            let mut coordinates = Vec::new();
            let mut rest = index;
            let mut positions = vec![0; sizes.len()];
            for (slot, size) in sizes.iter().enumerate().rev() { positions[slot] = rest % size; rest /= size; }
            for (axis, position) in axes.iter().zip(positions) { coordinates.push(scalar_value(&axis.coordinates[position]).to_json()); }
            let (mut values, mut errors, mut validities, mut error_validities, mut fixed) = (vec![], vec![], vec![], vec![], vec![]);
            for (parameter_index, parameter) in model.parameters.iter().enumerate() {
                let valid = outcome.is_some_and(|outcome| outcome.success);
                let value = outcome.map(|outcome| outcome.parameter_values[parameter_index]);
                let is_fixed = match outcome { None => fixed_authored.contains(&parameter.name), Some(outcome) => outcome.fixed_parameter_names.contains(&parameter.name) };
                let error_valid = valid && outcome.is_some_and(|outcome| outcome.covariance_valid && !outcome.fixed_parameter_names.contains(&parameter.name));
                values.push(value.filter(|v| v.is_finite()).map_or(Json::Null, |v| json!(v)));
                validities.push(valid && value.is_some_and(f64::is_finite));
                errors.push(if error_valid { json!(outcome.unwrap().standard_errors[parameter_index]) } else { Json::Null });
                error_validities.push(error_valid);
                if is_fixed { fixed.push(parameter.name.clone()); }
            }
            let message = match outcome { None => failures[index].clone().map_or(Json::Null, Json::String), Some(outcome) => json!(outcome.message) };
            rows.push(json!({
                "index": index, "coordinates": coordinates, "success": outcome.is_some_and(|outcome| outcome.success), "message": message,
                "values": values, "standard_errors": errors, "valid": validities, "error_valid": error_validities, "fixed_parameters": fixed,
                "covariance_valid": outcome.is_some_and(|outcome| outcome.covariance_valid),
                "reduced_chi_square": outcome.map_or(Json::Null, |outcome| finite_or_null(outcome.reduced_chi_square)),
            }));
        }
        Ok(json!({
            "status": self.fit_status(), "model_id": model.model_id, "source_generation": &*accepted.source_generation,
            "source_revision": source_revision, "batch_revision": batch_revision,
            "data_generation": generation.as_deref(), "data_revision": revision,
            "parameter_names": model.parameters.iter().map(|p| p.name.clone()).collect::<Vec<_>>(),
            "parameter_units": units,
            "sample_axes": axes.iter().map(|axis| json!({"domain": axis.domain.value(), "axis_id": &*axis.id, "name": &*axis.name,
                                                          "unit": axis.unit.as_deref()})).collect::<Vec<_>>(),
            "rows": rows, "total_rows": outcomes.len(), "offset": offset,
            "next_offset": if stop < outcomes.len() { json!(stop) } else { Json::Null },
        }))
    }

    /// `_invalidate_fit_context`: the accepted fit lags the interaction.
    pub(crate) fn invalidate_fit_context(&mut self) { self.scene.fit_context_generation += 1; }
}
