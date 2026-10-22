//! The configuration's fit layer over the native fit engine: the model
//! catalogue, target resolution and the expression text configuration asks
//! for, answered by `fit::target` exactly as `_fit_prepare` answers them.

use std::sync::Arc;

use crate::configuration::error::{ErrorKind, PlotError, Result};
use crate::configuration::fit::{FitLayer, FitModelInfo, FitRequest, FitResolution, FitTargetQuery, FitUnits, UnitRelation as InfoRelation};
use crate::configuration::handlers::FitTargetKind;
use crate::configuration::value::{Value, ValueMap};
use crate::fit::param_units::{FitParameterUnits, FitQuantityUnits};
use crate::fit::target::{fit_request_target, resolve_fit_target, LiveFitRequest, TargetFacts};
use crate::fit::{FitEngine, FitError, FitModelSpec, FitTarget, UnitRelation};

/// A fit error as the Python exception it is.
pub fn fit_error(error: FitError) -> PlotError {
    match error {
        FitError::Type(message) => PlotError::type_error(message),
        FitError::Runtime(message) => PlotError::runtime(message),
        FitError::Cancelled | FitError::Deadline => PlotError::new(ErrorKind::FitCancelled, error.to_string()),
        FitError::ZeroDivision => PlotError::new(ErrorKind::ZeroDivision, error.to_string()),
        FitError::Memory => PlotError::new(ErrorKind::Other("MemoryError"), error.to_string()),
        other => PlotError::value(other.to_string()),
    }
}

fn target_of(kind: FitTargetKind) -> FitTarget {
    match kind { FitTargetKind::Series => FitTarget::Series, FitTargetKind::Histogram => FitTarget::Histogram, FitTargetKind::Image => FitTarget::Image }
}

/// The catalogue facts of one model.
pub fn model_info(model: &FitModelSpec) -> FitModelInfo {
    FitModelInfo {
        model_id: model.model_id.clone(), display_name: model.display_name.clone(), independent_arity: model.independent_arity,
        coordinate_relations: model.coordinate_relations.iter().map(|relation| match relation {
            UnitRelation::Axis1 => InfoRelation::Axis1,
            _ => InfoRelation::Axis0,
        }).collect(),
        parameter_names: model.parameters.iter().map(|parameter| parameter.name.clone()).collect(),
        parameter_symbols: model.symbols().into_iter().map(str::to_owned).collect(),
    }
}

/// `FitParameterUnits` over configuration-resolved quantities.
pub fn parameter_units(units: &FitUnits) -> FitParameterUnits<'_> {
    let quantity = |held: &Option<crate::configuration::fit::FitQuantity>| held.as_ref()
        .map(|q| FitQuantityUnits { canonical_unit: q.canonical.clone(), display_unit: q.display.clone() });
    FitParameterUnits {
        registry: units.registry.as_deref(),
        quantities: Box::new(move |relation| match relation {
            UnitRelation::Value => quantity(&units.value),
            UnitRelation::Axis0 => quantity(&units.axis_0),
            UnitRelation::Axis1 => quantity(&units.axis_1),
            _ => None,
        }),
        histogram: units.histogram,
        rolling_primary: units.rolling_primary,
    }
}

/// The native request a configuration's fit request carries.
pub fn live_request(request: &FitRequest) -> Option<&LiveFitRequest> {
    request.native.as_ref().and_then(|native| native.downcast_ref::<LiveFitRequest>())
}

fn pairs(target: &ValueMap, key: &str) -> Vec<(String, f64)> {
    target.get(key).and_then(Value::as_map)
        .map(|map| map.iter().filter_map(|(name, value)| value.real().map(|number| (name.to_owned(), number))).collect())
        .unwrap_or_default()
}

/// One engine shared by every panel: a model is its registry's own object,
/// which the compiled lane requires.
pub struct EngineFitLayer { pub engine: Arc<FitEngine> }

impl EngineFitLayer {
    pub fn new(engine: Arc<FitEngine>) -> Self { Self { engine } }
}

impl FitLayer for EngineFitLayer {
    fn models_for(&self, target: FitTargetKind) -> Vec<Arc<FitModelInfo>> {
        self.engine.registry.models_for(target_of(target)).iter().map(|model| Arc::new(model_info(model))).collect()
    }

    fn resolve_target(&self, query: &FitTargetQuery<'_>) -> Result<FitResolution> {
        let units = parameter_units(query.units);
        let previous = query.previous_request.and_then(live_request);
        let (request, failure) = resolve_fit_target(&self.engine, &Value::Map(query.target.clone()).to_json(),
            TargetFacts { facet_grid: query.facet_grid }, &units, query.live, previous, query.previous_expression_failure)
            .map_err(fit_error)?;
        let request = request.map(|request| {
            let target = Value::from_json(&fit_request_target(Some(&request)));
            FitRequest { model: Arc::new(model_info(&request.model)), target: target.as_map().cloned().unwrap_or_default(),
                         native: Some(Arc::new(request)) }
        });
        Ok(FitResolution { request, expression_failure: failure })
    }

    fn expression_text(&self, units: &FitUnits, model: &FitModelInfo, target: &ValueMap) -> Result<String> {
        let model = self.engine.registry.get(&model.model_id).map_err(|error| fit_error(error.into()))?;
        parameter_units(units).fit_expression_text(&model, &pairs(target, "fixed"), &pairs(target, "initial")).map_err(fit_error)
    }
}
