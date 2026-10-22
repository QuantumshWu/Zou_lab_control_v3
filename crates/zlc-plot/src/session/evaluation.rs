//! Renderer-free numerical fits over one prepared projection
//! (`fit_evaluation`, the solving half of `_fit_prepare`): one authored
//! request evaluated per distribution, and the threshold classifier's
//! bimodal fits or caller-authored Gaussian pairs.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::configuration::display::accepts_classifier_thresholds;
use crate::configuration::error::{PlotError, Result};
use crate::configuration::ordered::OrderedMap;
use crate::configuration::selectors::{normalize_classifier_threshold_targets, thresholds_wire, GaussianComponents, ThresholdTarget};
use crate::configuration::value::{Value, ValueMap};
use crate::fit::classifier::authored_classifier_result;
use crate::fit::engine::{CellData, FitCell, FitRequest as EngineRequest};
use crate::fit::evaluation::{evaluate_batch, evaluate_single, TargetCell};
use crate::fit::target::LiveFitRequest;
use crate::fit::{FitEngine, FitError, FitModelSpec, FitResult};

use super::fit_layer::fit_error;
use super::projected::{scalar_value, FitCellTarget, FitProjection, FitSelection, SampleAxis, SnapshotRef};

pub type Seed = Option<Vec<f64>>;

/// One exact numerical answer before presentation (`FitEvaluation`).
#[derive(Debug)]
pub struct FitEvaluation {
    pub source_ref: SnapshotRef,
    pub model: Arc<FitModelSpec>,
    pub is_batch: bool,
    pub results: Vec<Option<FitResult>>,
    pub failure_messages: Vec<Option<String>>,
    pub selections: Vec<Option<Arc<FitSelection>>>,
    pub targets: Vec<FitCellTarget>,
    pub sample_axes: Vec<SampleAxis>,
    pub parameter_units: Vec<(String, String)>,
}

impl FitEvaluation {
    pub fn source_revision(&self) -> u64 { self.source_ref.revision }
    pub fn success(&self) -> Vec<bool> { self.results.iter().map(|result| result.as_ref().is_some_and(|fit| fit.success)).collect() }
    pub fn parameter_names(&self) -> Vec<String> { self.model.parameters.iter().map(|parameter| parameter.name.clone()).collect() }
    /// The per-cell values of each parameter (NaN where a cell failed).
    pub fn parameter_values(&self) -> Vec<(String, Vec<f64>)> { crate::fit::result::batch_parameter_columns(&self.results, &self.model) }
    /// The per-cell standard errors (NaN where none was estimated).
    pub fn parameter_errors(&self) -> Vec<(String, Vec<f64>)> { crate::fit::result::batch_error_columns(&self.results, &self.model) }
}

fn cell(selection: &FitSelection, warm_start: Seed) -> FitCell {
    match &selection.regular_image {
        Some(image) => FitCell { data: CellData::Image(image.clone()), sigma: None, selected_indices: None,
                                 data_revision: selection.data_revision as i64, warm_start },
        None => FitCell {
            data: CellData::Points { coordinates: selection.coordinates.clone(), observations: Some(selection.observations.clone()) },
            sigma: selection.observation_sigma.clone(), selected_indices: selection.selected_indices.clone(),
            data_revision: selection.data_revision as i64, warm_start,
        },
    }
}

fn cancellation(cancelled: Option<&(dyn Fn() -> bool + Sync)>) -> Result<()> {
    if cancelled.is_some_and(|cancelled| cancelled()) { return Err(fit_error(FitError::Cancelled)); }
    Ok(())
}

/// The dataset revision a fit reads (a pulse timeline has no fit target).
fn fitted_source(projection: &FitProjection) -> Result<SnapshotRef> {
    projection.source_ref().ok_or_else(|| PlotError::type_error("a pulse timeline has no fit target"))
}

/// `evaluate_fit(projection, request.model, ...)` for a resolved request.
pub fn evaluate_fit(projection: &FitProjection, request: &LiveFitRequest, engine: &FitEngine, distribution_indices: Option<&[usize]>,
                    warm_starts: Option<Vec<Seed>>, selection: Option<Arc<FitSelection>>,
                    cancelled: Option<&(dyn Fn() -> bool + Sync)>) -> Result<FitEvaluation> {
    let model = &request.model;
    let engine_request = EngineRequest { initial: request.initial.clone(), bounds: request.bounds.clone(), options: request.options.clone(), cancelled };
    let source_ref = fitted_source(projection)?;
    let is_batch = request.all_facets || projection.has_grouped_histogram();
    if !is_batch {
        if distribution_indices.is_some() { return Err(PlotError::value("distribution_indices requires a batch fit")); }
        let exact = match selection {
            Some(selection) => selection,
            None => Arc::new(projection.fit_selection(model, request.selector_kind.map(selector_kind), None, None)?),
        };
        if exact.data_revision != projection.data_revision() { return Err(PlotError::value("prepared fit selection belongs to another revision")); }
        let seeds = warm_starts.unwrap_or_else(|| vec![None]);
        if seeds.len() != 1 { return Err(PlotError::value("single fit requires exactly one warm-start slot")); }
        let units = projection.fit_parameter_units(model)?;
        let result = evaluate_single(engine, model, &cell(&exact, seeds.into_iter().next().flatten()), &engine_request, &units).map_err(fit_error)?;
        let target = FitCellTarget { facet_index: exact.facet_index, group_index: None, group_key: exact.group_key.clone() };
        return Ok(FitEvaluation {
            source_ref, model: model.clone(), is_batch: false, results: vec![Some(result)],
            failure_messages: vec![None], selections: vec![Some(exact)], targets: vec![target], sample_axes: Vec::new(), parameter_units: units,
        });
    }
    if selection.is_some() { return Err(PlotError::value("a batch fit prepares its own per-target selections")); }
    let targets = projection.fit_targets(request.all_facets);
    if targets.is_empty() { return Err(PlotError::value("projection has no distributions to fit")); }
    let requested: BTreeSet<usize> = match distribution_indices { None => (0..targets.len()).collect(), Some(indices) => indices.iter().copied().collect() };
    let seeds = warm_starts.unwrap_or_else(|| vec![None; targets.len()]);
    if seeds.len() != targets.len() { return Err(PlotError::value("batch warm-start slots must match fit targets")); }
    let mut selections: Vec<Option<Arc<FitSelection>>> = vec![None; targets.len()];
    let mut cells: Vec<TargetCell> = Vec::with_capacity(targets.len());
    match projection.prepare_fit_selection(model, request.selector_kind.map(selector_kind)) {
        Err(error) => cells.extend(targets.iter().map(|_| TargetCell::Failed(failure_text(&error)))),
        Ok(prepared) => for (index, target) in targets.iter().enumerate() {
            if !requested.contains(&index) { cells.push(TargetCell::NotRequested); continue; }
            match prepared.select(projection, target.facet_index, target.group_index) {
                Ok(selection) => {
                    cells.push(TargetCell::Ready(cell(&selection, seeds[index].clone())));
                    selections[index] = Some(Arc::new(selection));
                }
                Err(error) => cells.push(TargetCell::Failed(failure_text(&error))),
            }
        },
    }
    let units = projection.fit_parameter_units(model)?;
    let evaluated = evaluate_batch(engine, model, cells, &engine_request, &units).map_err(fit_error)?;
    let sample_axes = projection.fit_sample_axes(&targets)?;
    Ok(FitEvaluation { source_ref, model: model.clone(), is_batch: true, results: evaluated.results, failure_messages: evaluated.failures, selections,
                       targets, sample_axes,
                       parameter_units: units })
}

/// `str(error) or type(error).__name__`.
pub fn failure_text(error: &PlotError) -> String { if error.message.is_empty() { error.python_name().to_owned() } else { error.message.clone() } }

/// The fit layer's selector kind as the configuration's.
pub fn selector_kind(kind: crate::fit::target::SelectorKind) -> crate::configuration::selectors::SelectorKind {
    use crate::configuration::selectors::SelectorKind as Kind;
    match kind {
        crate::fit::target::SelectorKind::XRange => Kind::XRange, crate::fit::target::SelectorKind::Area => Kind::Area,
        crate::fit::target::SelectorKind::Crosshair => Kind::Crosshair, crate::fit::target::SelectorKind::Threshold => Kind::Threshold,
    }
}

/// `failed_fit_evaluation`: a failed numerical answer for every cell.
pub fn failed_fit_evaluation(projection: &FitProjection, request: &LiveFitRequest, error: &PlotError, selection: Option<Arc<FitSelection>>) -> Result<FitEvaluation> {
    let batch = request.all_facets || projection.has_grouped_histogram();
    let targets = if batch { projection.fit_targets(request.all_facets) } else {
        vec![match &selection {
            Some(selection) => FitCellTarget { facet_index: selection.facet_index, group_index: None, group_key: selection.group_key.clone() },
            None => projection.fit_targets(false).into_iter().next().ok_or_else(|| PlotError::index("tuple index out of range"))?,
        }]
    };
    let message = failure_text(error);
    let count = targets.len();
    Ok(FitEvaluation {
        source_ref: fitted_source(projection)?, model: request.model.clone(), is_batch: batch, results: vec![None; count],
        failure_messages: vec![Some(message); count], selections: if batch { vec![None; count] } else { vec![selection] },
        sample_axes: if batch { projection.fit_sample_axes(&targets)? } else { Vec::new() }, targets,
        parameter_units: projection.fit_parameter_units(&request.model)?,
    })
}

// ------------------------------------------------------------------ classifier
/// One authored classifier scope's identity (`_classifier_threshold_key`).
pub type ThresholdKey = String;

/// The authored classifier policy and its seeds (`ClassifierCapture`).
#[derive(Debug, Default)]
pub struct ClassifierCapture {
    pub model: Option<Arc<FitModelSpec>>,
    pub thresholds: OrderedMap<ThresholdKey, Option<f64>>,
    /// `Some(None)` is an authored record that names no Gaussian pair.
    pub components: OrderedMap<ThresholdKey, Option<Option<GaussianComponents>>>,
    pub warm_starts: BTreeMap<usize, Vec<f64>>,
}

impl ClassifierCapture {
    /// The authored scopes (`thresholds.keys() | components.keys()`).
    pub fn authored_keys(&self) -> BTreeSet<ThresholdKey> { self.thresholds.keys().chain(self.components.keys()).cloned().collect() }
}

/// One classifier answer over one projection (`ClassifierEvaluation`).
pub struct ClassifierEvaluation {
    pub capture: Arc<ClassifierCapture>,
    pub projection: FitProjection,
    pub targets: Vec<FitCellTarget>,
    pub requested: BTreeSet<usize>,
    pub results: Arc<[Option<FitResult>]>,
    pub selections: Vec<Option<Arc<FitSelection>>>,
    pub failures: Vec<Option<String>>,
}

impl ClassifierEvaluation {
    /// The same answer read under another projection (`replace(..., projection=)`).
    pub fn with_projection(&self, projection: FitProjection) -> ClassifierEvaluation {
        ClassifierEvaluation { capture: self.capture.clone(), projection, targets: self.targets.clone(), requested: self.requested.clone(),
                               results: self.results.clone(), selections: self.selections.clone(), failures: self.failures.clone() }
    }
}

/// `classifier_threshold_target(projection, distribution, value, subject=)`.
pub fn classifier_threshold_target(projection: &FitProjection, target: &FitCellTarget, value: Option<f64>,
                                   subject: Option<&crate::projection::output::SelectionSubject>) -> Result<ThresholdTarget> {
    let owned;
    let subject = match subject { Some(subject) => subject, None => { owned = projection.selection_subject(None)?; &owned } };
    let mut scope: Vec<(crate::kinds::AxisRef, Value)> = subject.scope.iter().map(|(axis, value)| (axis.clone(), scalar_value(value))).collect();
    if let (Some(facet_index), Some(facet)) = (target.facet_index, projection.spec().facet()) {
        if let crate::projection::Payload::Facet(data) = &**projection.payload() {
            scope.push((facet.clone(), scalar_value(&data.cells[facet_index].facet_value_canonical)));
        }
    }
    scope.extend(target.group_key.iter().map(|entry| (entry.axis.clone(), scalar_value(&entry.canonical))));
    let record = ValueMap::from_pairs([
        ("value", value.map_or(Value::None, Value::Float)),
        ("scope", Value::list(scope.into_iter().map(|(axis, coordinate)| Value::Map(ValueMap::from_pairs([
            ("domain", Value::text(axis.domain.value())), ("axis_id", Value::text(axis.axis_id.as_str())), ("coordinate", coordinate)]))))),
    ]);
    Ok(normalize_classifier_threshold_targets(&Value::list([Value::Map(record)]))?.remove(0))
}

/// `classifier_targets(projection, authored)`: a facet whose shared scope is
/// authored classifies its groups pooled.
pub fn classifier_targets(projection: &FitProjection, authored: &BTreeSet<ThresholdKey>) -> Result<Vec<FitCellTarget>> {
    let targets = projection.fit_targets(true);
    if !projection.has_grouped_histogram() || authored.is_empty() { return Ok(targets); }
    let subject = projection.selection_subject(None)?;
    let (mut result, mut shared, mut facet_keys): (Vec<FitCellTarget>, BTreeSet<Option<usize>>, BTreeMap<Option<usize>, ThresholdKey>) = Default::default();
    for target in targets {
        let common = FitCellTarget { facet_index: target.facet_index, group_index: None, group_key: Vec::new() };
        if !facet_keys.contains_key(&target.facet_index) {
            facet_keys.insert(target.facet_index, classifier_threshold_target(projection, &common, None, Some(&subject))?.key_repr());
        }
        if authored.contains(&facet_keys[&target.facet_index]) {
            if shared.insert(target.facet_index) { result.push(common); }
        } else {
            result.push(target);
        }
    }
    Ok(result)
}

/// `classifier_keys(projection, targets)`.
pub fn classifier_keys(projection: &FitProjection, targets: &[FitCellTarget]) -> Result<Vec<ThresholdKey>> {
    let subject = projection.selection_subject(None)?;
    targets.iter().map(|target| Ok(classifier_threshold_target(projection, target, None, Some(&subject))?.key_repr())).collect()
}

/// `resolve_classifier_capture(projection, thresholds, ...)`.
pub fn resolve_classifier_capture(projection: &FitProjection, thresholds: &[ThresholdTarget], engine: &FitEngine,
                                  warm_starts: BTreeMap<usize, Vec<f64>>, discard_unmatched: bool) -> Result<ClassifierCapture> {
    if !accepts_classifier_thresholds(projection.spec(), &projection.display_state().values) { return Ok(ClassifierCapture::default()); }
    let selected = normalize_classifier_threshold_targets(&thresholds_wire(thresholds))?;
    let raw = projection.fit_targets(true);
    let full = classifier_keys(projection, &raw)?;
    let shared = classifier_keys(projection, &raw.iter().map(|target| FitCellTarget { facet_index: target.facet_index, group_index: None,
                                                                                       group_key: Vec::new() }).collect::<Vec<_>>())?;
    let provided: BTreeSet<ThresholdKey> = selected.iter().map(ThresholdTarget::key_repr).collect();
    let allowed: BTreeSet<ThresholdKey> = full.iter().chain(&shared).cloned().collect();
    if !discard_unmatched && provided.iter().any(|key| !allowed.contains(key)) {
        return Err(PlotError::value("classifier threshold target does not match a current distribution"));
    }
    for (full, shared) in full.iter().zip(&shared) {
        if full != shared && provided.contains(full) && provided.contains(shared) {
            return Err(PlotError::value("shared and independent classifier scopes overlap in one facet"));
        }
    }
    let mut capture = ClassifierCapture { warm_starts, ..ClassifierCapture::default() };
    for target in selected {
        let identity = target.key_repr();
        if !allowed.contains(&identity) { continue; }
        capture.thresholds.insert(identity.clone(), target.value);
        capture.components.insert(identity, target.gaussian_components);
    }
    capture.model = Some(engine.registry.get("bimodal_gaussian").map_err(|error| fit_error(error.into()))?);
    Ok(capture)
}

/// `evaluate_threshold_classifier(projection, captured, indices=, cancelled=)`.
pub fn evaluate_threshold_classifier(projection: &FitProjection, capture: Arc<ClassifierCapture>, engine: &FitEngine, indices: Option<BTreeSet<usize>>,
                                     cancelled: Option<&(dyn Fn() -> bool + Sync)>) -> Result<ClassifierEvaluation> {
    let Some(model) = capture.model.clone() else {
        return Ok(ClassifierEvaluation { capture, projection: projection.clone(), targets: Vec::new(), requested: BTreeSet::new(),
                                         results: Arc::from(Vec::new()), selections: Vec::new(), failures: Vec::new() });
    };
    let targets = classifier_targets(projection, &capture.authored_keys())?;
    let keys = classifier_keys(projection, &targets)?;
    let count = targets.len();
    let requested: BTreeSet<usize> = indices.unwrap_or_else(|| (0..count).collect());
    if requested.iter().any(|index| *index >= count) { return Err(PlotError::value("classifier indices do not identify current distributions")); }
    let mut results: Vec<Option<FitResult>> = vec![None; count];
    let mut selections: Vec<Option<Arc<FitSelection>>> = vec![None; count];
    let mut failures: Vec<Option<String>> = vec![None; count];
    let automatic: Vec<usize> = requested.iter().copied().filter(|index| matches!(capture.components.get(&keys[*index]), None | Some(None))).collect();
    let request = LiveFitRequest { model: model.clone(), selector_kind: None, initial: None, bounds: None, options: None, all_facets: true };
    for pooled in [false, true] {
        let selected: Vec<usize> = automatic.iter().copied()
            .filter(|index| (projection.has_grouped_histogram() && targets[*index].group_key.is_empty()) == pooled).collect();
        if selected.is_empty() { continue; }
        let view = if pooled { projection.pooled_histogram_groups()? } else { projection.clone() };
        let view_targets = view.fit_targets(true);
        let local: Vec<(usize, usize)> = selected.iter().map(|index| {
            let position = view_targets.iter().position(|target| target.facet_index == targets[*index].facet_index
                && target.group_key == targets[*index].group_key).expect("classifier distribution");
            (*index, position)
        }).collect();
        let seeds: Vec<Seed> = (0..view_targets.len()).map(|index| capture.warm_starts.get(&index).cloned()).collect();
        let distribution: Vec<usize> = local.iter().map(|(_, position)| *position).collect();
        let evaluation = evaluate_fit(&view, &request, engine, Some(&distribution), Some(seeds), None, cancelled)?;
        for (outer, position) in local {
            results[outer] = evaluation.results[position].clone();
            selections[outer] = evaluation.selections[position].clone();
            failures[outer] = evaluation.failure_messages[position].clone();
        }
    }
    let mut pooled_projection: Option<FitProjection> = None;
    for index in requested.iter().copied() {
        cancellation(cancelled)?;
        let Some(Some(Some(authored))) = capture.components.get(&keys[index]).copied() else { continue };
        let target = &targets[index];
        let outcome = (|| -> Result<(FitResult, FitSelection)> {
            let view = if target.group_index.is_none() && projection.has_grouped_histogram() {
                if pooled_projection.is_none() { pooled_projection = Some(projection.pooled_histogram_groups()?); }
                pooled_projection.clone().unwrap()
            } else { projection.clone() };
            let selection = view.fit_selection(&model, None, target.facet_index, target.group_index)?;
            let components = [("center", authored.center), ("sigma", authored.sigma), ("delta_center", authored.delta_center),
                              ("sigma_B", authored.sigma_b), ("ratio", authored.ratio)].map(|(name, value)| (name.to_owned(), value));
            let coordinates: Vec<Vec<f64>> = selection.coordinates.iter().map(|axis| axis.to_vec()).collect();
            let result = authored_classifier_result(&model, &coordinates, &selection.observations, selection.selected_indices.as_deref().unwrap_or(&[]),
                selection.data_revision, &components, &projection.fit_parameter_units(&model)?).map_err(fit_error)?;
            Ok((result, selection))
        })();
        match outcome {
            Ok((result, selection)) => { results[index] = Some(result); selections[index] = Some(Arc::new(selection)); }
            Err(error) if error.kind == crate::configuration::error::ErrorKind::Type || error.is_value_error() => failures[index] = Some(error.message),
            Err(error) => return Err(error),
        }
    }
    Ok(ClassifierEvaluation { capture, projection: projection.clone(), targets, requested, results: results.into(), selections, failures })
}
