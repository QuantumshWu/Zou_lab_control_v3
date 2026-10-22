//! Authored fit targets: the wire mapping a panel's configuration carries,
//! decoded into a solver request or an expression failure
//! (`_fit_prepare.resolve_fit_target` / `fit_request_target` /
//! `prepare_fit_request`), and the model catalogue a plot offers
//! (`models_for` filtered by `fit_model_units_compatible`).

use std::sync::Arc;

use serde_json::{json, Map, Value};

use super::engine::FitEngine;
use super::num::str_repr;
use super::param_units::FitParameterUnits;
use super::result::{Bounds, FitError, FitOptions, ParamValues};
use super::shared::{authored_seed, solver_bounds};
use super::spec::{FitModelSpec, FitTarget, UnitRelation, FIT_TARGET_FIELDS};
use super::DECISIVE_BIC_GAIN;
use crate::units::Unit;

/// `SelectorKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SelectorKind {
    XRange,
    Area,
    Crosshair,
    Threshold,
}

impl SelectorKind {
    pub fn value(self) -> &'static str {
        match self {
            SelectorKind::XRange => "x_range",
            SelectorKind::Area => "area",
            SelectorKind::Crosshair => "crosshair",
            SelectorKind::Threshold => "threshold",
        }
    }
    pub fn parse(text: &str) -> Result<Self, FitError> {
        Ok(match text {
            "x_range" => SelectorKind::XRange,
            "area" => SelectorKind::Area,
            "crosshair" => SelectorKind::Crosshair,
            "threshold" => SelectorKind::Threshold,
            other => return Err(FitError::value(format!("{} is not a valid SelectorKind", str_repr(other)))),
        })
    }
}

/// `_LiveFitRequest`: one resolved authored request.
#[derive(Debug, Clone)]
pub struct LiveFitRequest {
    pub model: Arc<FitModelSpec>,
    pub selector_kind: Option<SelectorKind>,
    pub initial: Option<ParamValues>,
    pub bounds: Option<Bounds>,
    pub options: Option<FitOptions>,
    pub all_facets: bool,
}

/// `(raw expression, refusal text)`.
pub type ExpressionFailure = (String, String);

/// What target resolution needs to know about the plot spec.
#[derive(Debug, Clone, Copy, Default)]
pub struct TargetFacts {
    /// `isinstance(spec, FacetGridPlot)`.
    pub facet_grid: bool,
}

/// Python's `float(value)` for a JSON value.
fn py_float(value: &Value) -> Result<f64, FitError> {
    match value {
        Value::Number(n) => Ok(n.as_f64().unwrap_or(f64::NAN)),
        Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
        Value::String(s) => crate::pystr::float(s).ok_or_else(|| FitError::value(format!("could not convert string to float: {}", str_repr(s)))),
        Value::Null => Err(FitError::type_error("float() argument must be a string or a real number, not 'NoneType'")),
        Value::Array(_) => Err(FitError::type_error("float() argument must be a string or a real number, not 'list'")),
        Value::Object(_) => Err(FitError::type_error("float() argument must be a string or a real number, not 'dict'")),
    }
}

/// zlc_data's `integer(value, field)` for a JSON value.
fn py_integer(value: &Value, field: &str) -> Result<i64, FitError> {
    match value {
        Value::Number(n) if n.is_i64() || n.is_u64() => n.as_i64().ok_or_else(|| FitError::value(format!("{field} must be an integer"))),
        _ => Err(FitError::type_error(format!("{field} must be an integer"))),
    }
}

/// `FitOptions(**mapping)`.
pub fn options_from_mapping(mapping: &Map<String, Value>) -> Result<FitOptions, FitError> {
    let mut options = FitOptions::default();
    for (key, value) in mapping {
        match key.as_str() {
            "loss" => {
                options.loss = value.as_str().ok_or_else(|| FitError::type_error("fit loss must be text"))?.to_string();
            }
            "max_nfev" => options.max_nfev = py_integer(value, "max_nfev")?,
            "max_point_evaluations" => options.max_point_evaluations = if value.is_null() { None } else { Some(py_integer(value, "max_point_evaluations")?) },
            "deadline_seconds" => {
                options.deadline_seconds = match value {
                    Value::Null => None,
                    Value::Number(n) => Some(n.as_f64().unwrap_or(f64::NAN)),
                    _ => return Err(FitError::type_error("deadline_seconds must be a real number")),
                }
            }
            "max_exact_points" => options.max_exact_points = if value.is_null() { None } else { Some(py_integer(value, "max_exact_points")?) },
            "min_bic_gain" => {
                options.min_bic_gain = match value {
                    Value::Null => None,
                    Value::Number(n) => Some(n.as_f64().unwrap_or(f64::NAN)),
                    _ => return Err(FitError::type_error("min_bic_gain must be a real number or None")),
                }
            }
            other => return Err(FitError::type_error(format!("FitOptions.__init__() got an unexpected keyword argument {}", str_repr(other)))),
        }
    }
    options.validated()
}

fn initial_from_value(value: &Value) -> Result<ParamValues, FitError> {
    match value {
        Value::Object(map) => Ok(ParamValues::Named(map.iter().map(|(k, v)| py_float(v).map(|f| (k.clone(), f))).collect::<Result<_, _>>()?)),
        Value::Array(items) => Ok(ParamValues::Positional(items.iter().map(py_float).collect::<Result<_, _>>()?)),
        _ => Err(FitError::type_error("initial must be a parameter mapping or numeric sequence")),
    }
}

fn bounds_from_value(value: &Value) -> Result<Bounds, FitError> {
    let Value::Object(map) = value else { return Err(FitError::type_error("bounds must be a mapping or None")) };
    let mut bounds = Bounds::default();
    for (name, pair) in map {
        let Value::Array(items) = pair else { return Err(FitError::type_error("each fit bound must contain low and high")) };
        if items.len() != 2 {
            return Err(FitError::type_error("each fit bound must contain low and high"));
        }
        let side = |v: &Value| if v.is_null() { Ok(None) } else { py_float(v).map(Some) };
        bounds.set(name, (side(&items[0])?, side(&items[1])?));
    }
    Ok(bounds)
}

/// `prepare_fit_request(model, selector_kind=, initial=, bounds=, options=)`.
pub fn prepare_fit_request(model: &Arc<FitModelSpec>, selector_kind: Option<SelectorKind>, initial: Option<ParamValues>, bounds: Option<Bounds>, options: Option<FitOptions>) -> Result<LiveFitRequest, FitError> {
    if let Some(initial) = &initial {
        authored_seed(model, initial)?;
    }
    if let Some(bounds) = &bounds {
        if !bounds.is_empty() {
            solver_bounds(model, None, Some(bounds))?;
        }
    }
    Ok(LiveFitRequest { model: model.clone(), selector_kind, initial, bounds, options, all_facets: false })
}

/// `resolve_fit_target(target, spec=, units=, engine=, live=, previous_request=, previous_expression_failure=)`.
#[allow(clippy::too_many_arguments)]
pub fn resolve_fit_target(
    engine: &FitEngine,
    target: &Value,
    facts: TargetFacts,
    units: &FitParameterUnits,
    live: bool,
    previous_request: Option<&LiveFitRequest>,
    previous_expression_failure: Option<&ExpressionFailure>,
) -> Result<(Option<LiveFitRequest>, Option<ExpressionFailure>), FitError> {
    let Value::Object(values) = target else { return Err(FitError::type_error("fit target must be a mapping")) };
    let mut unknown: Vec<&str> = values.keys().map(String::as_str).filter(|k| !FIT_TARGET_FIELDS.contains(k)).collect();
    if !unknown.is_empty() {
        unknown.sort();
        return Err(FitError::type_error(format!("unknown fit target fields: {}", unknown.join(", "))));
    }
    let model = values.get("model").filter(|m| !m.is_null());
    let model_text = match model {
        None => return Ok((None, None)),
        Some(Value::String(text)) if crate::pystr::strip(text).is_empty() => return Ok((None, None)),
        Some(Value::String(text)) => text.clone(),
        Some(_) => return Err(FitError::type_error("model must be a registered model id or FitModelSpec")),
    };
    let model_spec = engine.registry.get(&model_text)?;
    let raw = |key: &str| values.get(key).filter(|v| !v.is_null()).cloned();
    let mut fixed_raw: Option<Value> = raw("fixed");
    let mut initial_raw: Option<Value> = raw("initial");
    let mut bounds_raw: Option<Value> = raw("bounds");
    let mut options_raw: Option<Value> = raw("options");
    let fit_all_facets = match values.get("fit_all_facets") {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Number(n)) => n.as_f64().is_some_and(|v| v != 0.0),
        Some(Value::String(s)) => !s.is_empty(),
        Some(Value::Array(a)) => !a.is_empty(),
        Some(Value::Object(o)) => !o.is_empty(),
    };
    // Options as a mapping are validated when the request is prepared; a
    // threshold beside no options replaces the defaults' at once.
    let mut options_ready: Option<FitOptions> = None;
    if let Some(gain) = values.get("min_bic_gain") {
        if model_spec.reduction.is_none() {
            return Err(FitError::type_error(format!("{} fits one population; it has no min_bic_gain", model_spec.model_id)));
        }
        let gain = if gain.is_null() { Value::Null } else { json!(py_float(gain)?) };
        match &mut options_raw {
            Some(Value::Object(map)) => {
                map.insert("min_bic_gain".into(), gain);
            }
            None => {
                let mut base = FitOptions::default();
                base.min_bic_gain = gain.as_f64();
                options_ready = Some(base.validated()?);
            }
            Some(_) => return Err(FitError::type_error("options must be FitOptions, a mapping, or None")),
        }
    }
    let selector_kind = match values.get("selector_kind") {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => Some(SelectorKind::parse(s)?),
        Some(other) => Some(SelectorKind::parse(&other.to_string())?),
    };
    let previous_model = previous_request.map(|r| r.model.model_id.clone());
    let mut expression_failure = previous_expression_failure.cloned();
    let mut parsed_initial: Option<ParamValues> = None;
    if let Some(expression) = values.get("expression") {
        let text = match expression {
            Value::String(s) => s.clone(),
            Value::Null => "None".to_string(),
            other => other.to_string(),
        };
        match units.fit_expression_target(&model_spec, &text) {
            Err(error) => {
                expression_failure = Some((text, error.to_string()));
                fixed_raw = None;
                initial_raw = None;
                bounds_raw = None;
            }
            Ok(parsed) => {
                fixed_raw = (!parsed.fixed.is_empty()).then(|| Value::Object(parsed.fixed.iter().map(|(k, v)| (k.clone(), json!(v))).collect()));
                parsed_initial = (!parsed.initial.is_empty()).then_some(ParamValues::Named(parsed.initial));
                initial_raw = None;
                bounds_raw = None;
                expression_failure = None;
            }
        }
    } else if previous_model.as_deref() != Some(model_spec.model_id.as_str()) || fixed_raw.is_some() || initial_raw.is_some() || bounds_raw.is_some() {
        expression_failure = None;
    }
    let mut bounds_value = bounds_raw;
    if let Some(fixed) = fixed_raw.filter(|f| f.as_object().is_none_or(|m| !m.is_empty())) {
        let Value::Object(fixed) = fixed else { return Err(FitError::type_error("fixed must be a mapping")) };
        let mut merged = match bounds_value {
            None => Map::new(),
            Some(Value::Object(map)) => map,
            Some(_) => return Err(FitError::type_error("bounds must be a mapping or None")),
        };
        for (name, value) in fixed {
            merged.insert(name, json!([value, value]));
        }
        bounds_value = Some(Value::Object(merged));
    }
    let options = match (options_ready, options_raw) {
        (Some(ready), _) => Some(ready),
        (None, None) => None,
        (None, Some(Value::Object(map))) => Some(options_from_mapping(&map)?),
        (None, Some(_)) => return Err(FitError::type_error("options must be FitOptions, a mapping, or None")),
    };
    let initial = match (parsed_initial, initial_raw) {
        (Some(parsed), _) => Some(parsed),
        (None, None) => None,
        (None, Some(Value::String(_))) => return Err(FitError::type_error("initial must be a parameter mapping or numeric sequence")),
        (None, Some(value)) => Some(initial_from_value(&value)?),
    };
    let bounds = match bounds_value {
        None => None,
        Some(value) => Some(bounds_from_value(&value)?),
    };
    let mut prepared = prepare_fit_request(&model_spec, selector_kind, initial, bounds, options)?;
    prepared.all_facets = fit_all_facets || (live && facts.facet_grid);
    Ok((Some(prepared), expression_failure))
}

/// `fit_request_target(request)`: the canonical wire target of a request.
pub fn fit_request_target(request: Option<&LiveFitRequest>) -> Value {
    let mut fit = Map::new();
    let Some(request) = request else { return Value::Object(fit) };
    fit.insert("model".into(), json!(request.model.model_id));
    if let Some(kind) = request.selector_kind {
        fit.insert("selector_kind".into(), json!(kind.value()));
    }
    if let Some(initial) = &request.initial {
        let mut map = Map::new();
        match initial {
            ParamValues::Named(items) => {
                for (k, v) in items {
                    map.insert(k.clone(), json!(v));
                }
            }
            ParamValues::Positional(values) => {
                for (name, v) in request.model.parameter_names().iter().zip(values) {
                    map.insert(name.to_string(), json!(v));
                }
            }
        }
        fit.insert("initial".into(), Value::Object(map));
    }
    if let Some(bounds) = &request.bounds {
        let mut remaining = Map::new();
        let mut fixed = Map::new();
        for (name, (lo, hi)) in &bounds.0 {
            if lo.is_some() && lo == hi {
                fixed.insert(name.clone(), json!(lo.expect("fixed")));
            } else {
                remaining.insert(name.clone(), json!([lo, hi]));
            }
        }
        if !remaining.is_empty() {
            fit.insert("bounds".into(), Value::Object(remaining));
        }
        if !fixed.is_empty() {
            fit.insert("fixed".into(), Value::Object(fixed));
        }
    }
    if let Some(options) = &request.options {
        let defaults = FitOptions::default();
        let mut map = Map::new();
        if options.loss != defaults.loss {
            map.insert("loss".into(), json!(options.loss));
        }
        if options.max_nfev != defaults.max_nfev {
            map.insert("max_nfev".into(), json!(options.max_nfev));
        }
        if options.max_point_evaluations != defaults.max_point_evaluations {
            map.insert("max_point_evaluations".into(), json!(options.max_point_evaluations));
        }
        if options.deadline_seconds != defaults.deadline_seconds {
            map.insert("deadline_seconds".into(), json!(options.deadline_seconds));
        }
        if options.max_exact_points != defaults.max_exact_points {
            map.insert("max_exact_points".into(), json!(options.max_exact_points));
        }
        if !map.is_empty() {
            fit.insert("options".into(), Value::Object(map));
        }
    }
    if request.model.reduction.is_some() {
        let gain = match &request.options {
            None => Some(DECISIVE_BIC_GAIN),
            Some(options) => options.min_bic_gain,
        };
        fit.insert("min_bic_gain".into(), json!(gain));
    }
    if request.all_facets {
        fit.insert("fit_all_facets".into(), json!(true));
    }
    Value::Object(fit)
}

/// `fit_model_units_compatible(model, axis_0=, axis_1=, value=)`.
pub fn fit_model_units_compatible(model: &FitModelSpec, axis_0: &Unit, axis_1: Option<&Unit>, value: &Unit) -> bool {
    let sources: Vec<Option<&Unit>> = if model.independent_arity == 1 { vec![Some(axis_0)] } else { vec![Some(axis_0), axis_1] };
    if sources.len() != model.coordinate_relations.len() {
        return false;
    }
    sources.iter().zip(&model.coordinate_relations).all(|(source, relation)| {
        let target = match relation {
            UnitRelation::Axis0 => Some(axis_0),
            UnitRelation::Axis1 => axis_1,
            UnitRelation::Value => Some(value),
            _ => None,
        };
        matches!((source, target), (Some(s), Some(t)) if s.compatible_with(t))
    })
}

/// The models a plot offers: its target's catalogue, the default first,
/// filtered to the ones its axes' units can enter.
pub fn models_for_units(engine: &FitEngine, target: FitTarget, axis_0: &Unit, axis_1: Option<&Unit>, value: &Unit) -> Vec<Arc<FitModelSpec>> {
    engine.registry.models_for(target).into_iter().filter(|m| fit_model_units_compatible(m, axis_0, axis_1, value)).collect()
}

/// The small surface the configuration layer resolves fit targets through.
pub trait FitTargets {
    fn models_for(&self, target: FitTarget) -> Vec<Arc<FitModelSpec>>;
    #[allow(clippy::too_many_arguments)]
    fn resolve_fit_target(&self, target: &Value, facts: TargetFacts, units: &FitParameterUnits, live: bool, previous: Option<&LiveFitRequest>, previous_failure: Option<&ExpressionFailure>) -> Result<(Option<LiveFitRequest>, Option<ExpressionFailure>), FitError>;
    fn fit_request_target(&self, request: Option<&LiveFitRequest>) -> Value;
    fn fit_expression_text(&self, model: &FitModelSpec, request: &LiveFitRequest, units: &FitParameterUnits) -> Result<String, FitError>;
}

impl FitTargets for FitEngine {
    fn models_for(&self, target: FitTarget) -> Vec<Arc<FitModelSpec>> { self.registry.models_for(target) }

    fn resolve_fit_target(&self, target: &Value, facts: TargetFacts, units: &FitParameterUnits, live: bool, previous: Option<&LiveFitRequest>, previous_failure: Option<&ExpressionFailure>) -> Result<(Option<LiveFitRequest>, Option<ExpressionFailure>), FitError> {
        resolve_fit_target(self, target, facts, units, live, previous, previous_failure)
    }

    fn fit_request_target(&self, request: Option<&LiveFitRequest>) -> Value { fit_request_target(request) }

    fn fit_expression_text(&self, model: &FitModelSpec, request: &LiveFitRequest, units: &FitParameterUnits) -> Result<String, FitError> {
        let target = fit_request_target(Some(request));
        let pairs = |key: &str| -> Vec<(String, f64)> {
            target.get(key).and_then(Value::as_object).map(|m| m.iter().filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f))).collect()).unwrap_or_default()
        };
        units.fit_expression_text(model, &pairs("fixed"), &pairs("initial"))
    }
}
