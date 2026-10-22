//! The exact calls configuration makes into the fit layer, as one trait.
//!
//! Configuration never parses a fit expression or solves anything: it asks
//! the fit layer (the native fit engine, or a recorded oracle in tests) for
//! the model catalogue of a target, the canonical request a fit target
//! resolves to, and the expression text a request reads as.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use crate::units::{Unit, UnitRegistry};

use super::error::Result;
use super::handlers::FitTargetKind;
use super::value::ValueMap;

/// The fields a fit target may carry (`FIT_TARGET_FIELDS`).
pub const FIT_TARGET_FIELDS: [&str; 9] = ["model", "expression", "selector_kind", "fixed", "initial", "bounds", "options",
                                          "fit_all_facets", "min_bic_gain"];

/// Which plotted axis one independent model coordinate is read from.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum UnitRelation { Axis0, Axis1 }

/// The catalogue facts configuration reads of one fit model.
#[derive(Clone, Debug, PartialEq)]
pub struct FitModelInfo {
    pub model_id: String,
    pub display_name: String,
    /// One or two independent axes.
    pub independent_arity: usize,
    pub coordinate_relations: Vec<UnitRelation>,
    pub parameter_names: Vec<String>,
    /// The symbols its formula prints (what the Parameters row names, v2 `model.symbols`).
    pub parameter_symbols: Vec<String>,
}

/// One plotted quantity's canonical and display units.
#[derive(Clone, Debug, PartialEq)]
pub struct FitQuantity { pub canonical: Arc<Unit>, pub display: Arc<Unit> }

/// The unit vocabulary a fit expression is read and written in
/// (`configuration_fit_units`): VALUE, AXIS_0 and AXIS_1 quantities.
#[derive(Clone)]
pub struct FitUnits {
    pub registry: Option<Arc<UnitRegistry>>,
    pub value: Option<FitQuantity>,
    pub axis_0: Option<FitQuantity>,
    pub axis_1: Option<FitQuantity>,
    pub histogram: bool,
    pub rolling_primary: bool,
}

/// One resolved request: its model and the canonical target it encodes to
/// (`fit_request_target`), plus the fit layer's own typed request.
#[derive(Clone)]
pub struct FitRequest {
    pub model: Arc<FitModelInfo>,
    pub target: ValueMap,
    pub native: Option<Arc<dyn Any + Send + Sync>>,
}

impl fmt::Debug for FitRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FitRequest").field("model", &self.model.model_id).field("target", &self.target).finish()
    }
}

/// The question `resolve_fit_target` answers.
pub struct FitTargetQuery<'a> {
    pub target: &'a ValueMap,
    /// `isinstance(spec, FacetGridPlot)`.
    pub facet_grid: bool,
    pub units: &'a FitUnits,
    pub live: bool,
    pub previous_request: Option<&'a FitRequest>,
    pub previous_expression_failure: Option<&'a (String, String)>,
}

/// `resolve_fit_target`'s answer: the request (none for no model) and the
/// `(expression, reason)` an unparsable expression left.
#[derive(Clone, Debug, Default)]
pub struct FitResolution {
    pub request: Option<FitRequest>,
    pub expression_failure: Option<(String, String)>,
}

/// The fit layer, as configuration sees it.
pub trait FitLayer: Send + Sync {
    /// `FitEngine.registry.models_for(FitTarget(target))`: compatible models,
    /// the target's authored default first.
    fn models_for(&self, target: FitTargetKind) -> Vec<Arc<FitModelInfo>>;
    /// `resolve_fit_target(...)` with `fit_request_target` applied to its
    /// request.  Errors are the Python exception classes it raises.
    fn resolve_target(&self, query: &FitTargetQuery<'_>) -> Result<FitResolution>;
    /// `configuration_fit_units(...).fit_expression_text(model, target)`.
    fn expression_text(&self, units: &FitUnits, model: &FitModelInfo, target: &ValueMap) -> Result<String>;
}

impl fmt::Debug for FitUnits {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FitUnits").field("value", &self.value).field("axis_0", &self.axis_0).field("axis_1", &self.axis_1)
            .field("histogram", &self.histogram).field("rolling_primary", &self.rolling_primary).finish()
    }
}

/// A fit layer with an empty model catalogue (an empty `FitModelRegistry`):
/// no model is offered, a target without a model resolves to no request, and
/// naming a model is the registry's ValueError.  For bring-up and tests.
pub struct EmptyFitLayer;

impl FitLayer for EmptyFitLayer {
    fn models_for(&self, _target: FitTargetKind) -> Vec<Arc<FitModelInfo>> { Vec::new() }
    fn resolve_target(&self, query: &FitTargetQuery<'_>) -> Result<FitResolution> {
        let mut unknown: Vec<&str> = query.target.keys().filter(|key| !FIT_TARGET_FIELDS.contains(key)).collect();
        if !unknown.is_empty() {
            unknown.sort_unstable();
            return Err(super::error::PlotError::type_error(format!("unknown fit target fields: {}", unknown.join(", "))));
        }
        match query.target.get("model") {
            None | Some(super::value::Value::None) => Ok(FitResolution::default()),
            Some(model) if model.py_str().trim().is_empty() => Ok(FitResolution::default()),
            Some(model) => Err(super::error::PlotError::value(format!("unknown fit model: {}", model.py_repr()))),
        }
    }
    fn expression_text(&self, _units: &FitUnits, _model: &FitModelInfo, _target: &ValueMap) -> Result<String> { Ok(String::new()) }
}
