//! One bounded scientific preparation template (`preparation.DataPreparation`):
//! projection, the authored fit and the threshold classifier of one admitted
//! read, carrying only the latest answer and small warm seeds.

use std::collections::BTreeMap;
use std::sync::Arc;

use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::parameters::RenderEffect;
use crate::configuration::resolve::{preparation_key, ResolvedConfiguration};
use crate::fit::target::LiveFitRequest;
use crate::fit::{FitEngine, FitResult};
use crate::interaction::controller::SelectorSnapshot;
use crate::projection::SchemaCache;
use crate::pulse::PulseTimelineData;

use super::evaluation::{evaluate_fit, evaluate_threshold_classifier, failed_fit_evaluation, resolve_classifier_capture, ClassifierEvaluation,
                        FitEvaluation, Seed};
use super::fit_layer::live_request;
use super::projected::{FitProjection, PlotInput, ProjectionContext, ProjectionSettings, SnapshotRef};

/// One exact answer paired with the configuration that requested it (`PreparedPlot`).
#[derive(Clone)]
pub struct PreparedPlot {
    pub configuration: Arc<ResolvedConfiguration>,
    pub projection: FitProjection,
    pub evaluation: Option<Arc<FitEvaluation>>,
    pub classifier: Arc<ClassifierEvaluation>,
    /// The dataset revision prepared; None for a pulse timeline.
    pub source_ref: Option<SnapshotRef>,
    pub request: Option<LiveFitRequest>,
    pub expression_failure: Option<(String, String)>,
}

impl PreparedPlot {
    pub fn with_configuration(&self, configuration: Arc<ResolvedConfiguration>) -> PreparedPlot { PreparedPlot { configuration, ..self.clone() } }
}

/// `ResolvedConfiguration.context`.
pub fn configuration_context(configuration: &ResolvedConfiguration) -> Arc<ProjectionContext> {
    Arc::new(ProjectionContext {
        display_state: Arc::new(configuration.display_state.clone()),
        selectors: SelectorSnapshot { committed: configuration.selectors.clone(), candidate: None },
        viewport: configuration.viewport,
        focused_facet_index: configuration.focused_facet_index.map(|index| index as usize),
    })
}

/// The native request of a configuration's fit target.
pub fn configuration_request(configuration: &ResolvedConfiguration) -> Option<LiveFitRequest> {
    configuration.fit_request.as_ref().and_then(live_request).cloned()
}

fn seeds(results: &[Option<FitResult>]) -> BTreeMap<usize, Vec<f64>> {
    results.iter().enumerate()
        .filter_map(|(index, result)| result.as_ref().filter(|fit| fit.success && !fit.reduced)
            .map(|fit| (index, fit.parameter_values.clone())))
        .collect()
}

/// Serial exact preparation: one latest projection and small warm seeds.
#[derive(Clone)]
pub struct DataPreparation {
    pub configuration: Arc<ResolvedConfiguration>,
    previous: Option<PreparedPlot>,
    request: Option<LiveFitRequest>,
    expression_failure: Option<(String, String)>,
    warm_starts: BTreeMap<usize, Vec<f64>>,
    classifier_seeds: BTreeMap<usize, Vec<f64>>,
}

impl DataPreparation {
    pub fn new(configuration: Arc<ResolvedConfiguration>, previous: Option<PreparedPlot>) -> Self {
        let previous = previous.filter(|_| configuration.same_source);
        let same = previous.as_ref().is_some_and(|previous| previous.configuration.signature == configuration.signature);
        let warm_starts = match previous.as_ref().and_then(|previous| previous.evaluation.as_ref()) {
            Some(evaluation) if same => seeds(&evaluation.results),
            _ => BTreeMap::new(),
        };
        let classifier_seeds = previous.as_ref().map_or_else(BTreeMap::new, |previous| seeds(&previous.classifier.results));
        DataPreparation { request: configuration_request(&configuration), expression_failure: configuration.expression_failure.clone(),
                          configuration, previous, warm_starts, classifier_seeds }
    }

    /// Prepare one admitted read (`DataPreparation.prepare`); a pulse
    /// timeline is prepared at `revision` (0 when none is given).
    pub fn prepare(&mut self, input: &PlotInput, revision: Option<u64>, schemas: &mut SchemaCache, settings: &ProjectionSettings,
                   engine: &FitEngine) -> Result<PreparedPlot> {
        let config = self.configuration.clone();
        if preparation_key(input.schema().as_ref())? != preparation_key(config.schema.as_ref())? {
            return Err(PlotError::value("data preparation requires the resolved scientific schema"));
        }
        if config.cancelled() { return Err(PlotError::new(ErrorKind::FitCancelled, "data preparation cancelled")); }
        let source = match input {
            PlotInput::Dataset(source) => source,
            PlotInput::Pulse(data) => return self.prepare_pulse(data, revision.unwrap_or(0), engine),
        };
        let source_ref = Some(SnapshotRef::of(source));
        let schema = config.schema.clone().ok_or_else(|| PlotError::type_error("data preparation requires a scientific plot input"))?;
        let context = configuration_context(&config);
        if let Some(previous) = self.previous.as_ref().filter(|previous| previous.configuration.signature == config.signature
            && previous.source_ref == source_ref && previous.projection.data_revision() == source.revision) {
            let changed: Vec<&'static str> = config.parameter_schema.names()
                .filter(|name| previous.configuration.display_state.values.get(name) != config.display_state.values.get(name)).collect();
            let effects = if changed.is_empty() { RenderEffect::NONE } else { config.parameter_schema.effects_for(changed.iter().copied())? };
            let mut projection = if effects.intersects(RenderEffect::VIEW_PROJECTION | RenderEffect::PAYLOAD_PROJECTION) {
                FitProjection::build(source.clone(), &schema, previous.projection.spec().clone(), context, settings.clone(), Some(&previous.projection), schemas)?
            } else {
                previous.projection.with_context(context)
            };
            if *projection.spec() != config.spec { projection = projection.with_spec(config.spec.clone()); }
            let prepared = PreparedPlot {
                configuration: config, classifier: Arc::new(previous.classifier.with_projection(projection.clone())), projection,
                evaluation: previous.evaluation.clone(), source_ref, request: self.request.clone(), expression_failure: self.expression_failure.clone(),
            };
            self.previous = Some(prepared.clone());
            return Ok(prepared);
        }
        let inherited = self.previous.as_ref().map(|previous| &previous.projection).filter(|projection| *projection.spec() == config.spec);
        let projection = FitProjection::build(source.clone(), &schema, config.spec.clone(), context, settings.clone(), inherited, schemas)?;
        let cancelled = || config.cancelled();
        let evaluation = match &self.request {
            None => None,
            Some(request) => {
                let count = if request.all_facets || projection.has_grouped_histogram() { projection.fit_targets(request.all_facets).len() } else { 1 };
                let warm: Vec<Seed> = (0..count).map(|index| self.warm_starts.get(&index).cloned()).collect();
                let _span = crate::render::profile::span("prepare.fit");
                Some(Arc::new(match evaluate_fit(&projection, request, engine, None, Some(warm), None, Some(&cancelled)) {
                    Ok(evaluation) => evaluation,
                    Err(error) if error.kind == ErrorKind::FitCancelled => return Err(error),
                    // A solve this data cannot support is the panel's fit result.
                    Err(error) => failed_fit_evaluation(&projection, request, &error, None)?,
                }))
            }
        };
        let capture = resolve_classifier_capture(&projection, &config.classifier_thresholds, engine, self.classifier_seeds.clone(),
                                                 config.discard_unmatched_classifier)?;
        let classifier = evaluate_threshold_classifier(&projection, Arc::new(capture), engine, None, Some(&cancelled))?;
        if config.cancelled() { return Err(PlotError::new(ErrorKind::FitCancelled, "data preparation cancelled")); }
        self.warm_starts = evaluation.as_ref().map_or_else(BTreeMap::new, |evaluation| seeds(&evaluation.results));
        self.classifier_seeds = seeds(&classifier.results);
        let prepared = PreparedPlot { configuration: config, projection, evaluation, classifier: Arc::new(classifier), source_ref,
                                      request: self.request.clone(), expression_failure: self.expression_failure.clone() };
        self.previous = Some(prepared.clone());
        Ok(prepared)
    }

    /// `prepare` of an authored pulse timeline: the timeline is its own
    /// payload and has no fit target; the previous answer is read again while
    /// it is the same timeline at the same revision.
    fn prepare_pulse(&mut self, data: &Arc<PulseTimelineData>, revision: u64, engine: &FitEngine) -> Result<PreparedPlot> {
        let config = self.configuration.clone();
        let context = configuration_context(&config);
        let input = PlotInput::Pulse(data.clone());
        if let Some(previous) = self.previous.as_ref().filter(|previous| previous.configuration.signature == config.signature
            && previous.source_ref.is_none() && previous.projection.data().same(&input) && previous.projection.data_revision() == revision) {
            let changed: Vec<&'static str> = config.parameter_schema.names()
                .filter(|name| previous.configuration.display_state.values.get(name) != config.display_state.values.get(name)).collect();
            let effects = if changed.is_empty() { RenderEffect::NONE } else { config.parameter_schema.effects_for(changed.iter().copied())? };
            let mut projection = if effects.intersects(RenderEffect::VIEW_PROJECTION | RenderEffect::PAYLOAD_PROJECTION) {
                FitProjection::pulse(data.clone(), revision, previous.projection.spec().clone(), context)?
            } else {
                previous.projection.with_context(context)
            };
            if *projection.spec() != config.spec { projection = projection.with_spec(config.spec.clone()); }
            let prepared = PreparedPlot {
                configuration: config, classifier: Arc::new(previous.classifier.with_projection(projection.clone())), projection,
                evaluation: previous.evaluation.clone(), source_ref: None, request: self.request.clone(),
                expression_failure: self.expression_failure.clone(),
            };
            self.previous = Some(prepared.clone());
            return Ok(prepared);
        }
        let projection = FitProjection::pulse(data.clone(), revision, config.spec.clone(), context)?;
        let cancelled = || config.cancelled();
        let capture = resolve_classifier_capture(&projection, &config.classifier_thresholds, engine, self.classifier_seeds.clone(),
                                                 config.discard_unmatched_classifier)?;
        let classifier = evaluate_threshold_classifier(&projection, Arc::new(capture), engine, None, Some(&cancelled))?;
        if config.cancelled() { return Err(PlotError::new(ErrorKind::FitCancelled, "data preparation cancelled")); }
        self.warm_starts = BTreeMap::new();
        self.classifier_seeds = seeds(&classifier.results);
        let prepared = PreparedPlot { configuration: config, projection, evaluation: None, classifier: Arc::new(classifier), source_ref: None,
                                      request: self.request.clone(), expression_failure: self.expression_failure.clone() };
        self.previous = Some(prepared.clone());
        Ok(prepared)
    }
}
