//! Units of fit parameters and the one-line expression an operator types
//! (`_fit_units.FitParameterUnits`).

use std::sync::Arc;

use super::num::str_repr;
use super::result::FitError;
use super::spec::{FitModelSpec, FitParameterSpec, UnitRelation};
use crate::units::{Unit, UnitRegistry, DEFAULT_UNITS};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Crossing {
    Point,
    Span,
    Inverse,
}

impl Crossing {
    pub fn value(self) -> &'static str {
        match self {
            Crossing::Point => "point",
            Crossing::Span => "span",
            Crossing::Inverse => "inverse",
        }
    }
}

/// `FitQuantityUnits`: the units one relation's quantity is solved and shown in.
#[derive(Debug, Clone)]
pub struct FitQuantityUnits {
    pub canonical_unit: Arc<Unit>,
    pub display_unit: Arc<Unit>,
}

/// `_FitParameterConversion`.
#[derive(Debug, Clone)]
pub struct FitParameterConversion {
    pub parameter: String,
    pub canonical_unit: Option<Arc<Unit>>,
    pub display_unit: Option<Arc<Unit>>,
    pub symbol: String,
    pub crossing: Crossing,
    pub canonical_symbol: String,
}

impl FitParameterConversion {
    pub fn to_display(&self, value: f64) -> Result<f64, FitError> { self.convert(value, self.canonical_unit.as_deref(), self.display_unit.as_deref()) }
    pub fn to_canonical(&self, value: f64) -> Result<f64, FitError> { self.convert(value, self.display_unit.as_deref(), self.canonical_unit.as_deref()) }

    fn convert(&self, value: f64, source: Option<&Unit>, target: Option<&Unit>) -> Result<f64, FitError> {
        let (Some(source), Some(target)) = (source, target) else { return Ok(value) };
        if source == target {
            return Ok(value);
        }
        if self.crossing == Crossing::Point {
            return source.convert_value_to(value, target).map_err(|e| FitError::value(e.0));
        }
        let (Some(source_scale), Some(target_scale)) = (source.coordinate_scale(), target.coordinate_scale()) else {
            return Err(self.no_value(source, target));
        };
        if source_scale.0 != target_scale.0 {
            return Err(self.no_value(source, target));
        }
        let ratio = source_scale.1 / target_scale.1;
        Ok(value * if self.crossing == Crossing::Span { ratio } else { 1.0 / ratio })
    }

    fn no_value(&self, source: &Unit, target: &Unit) -> FitError {
        FitError::value(format!(
            "fit parameter {} is a {} and has no value between {} and {}; only a position crosses a logarithmic unit",
            str_repr(&self.parameter),
            self.crossing.value(),
            str_repr(&source.symbol),
            str_repr(&target.symbol)
        ))
    }
}

/// The quantity a plot paints for each relation.
pub type Quantities<'a> = dyn Fn(UnitRelation) -> Option<FitQuantityUnits> + 'a;

/// `FitParameterUnits`.
pub struct FitParameterUnits<'a> {
    pub registry: Option<&'a UnitRegistry>,
    pub quantities: Box<Quantities<'a>>,
    pub histogram: bool,
    pub rolling_primary: bool,
}

/// A parsed expression: `{"model", "fixed", "initial"}`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitExpressionTarget {
    pub model: String,
    pub fixed: Vec<(String, f64)>,
    pub initial: Vec<(String, f64)>,
}


impl<'a> FitParameterUnits<'a> {
    fn registry(&self) -> &UnitRegistry { self.registry.unwrap_or(&DEFAULT_UNITS) }

    /// `canonical_parameter_unit(spec)`.
    pub fn canonical_parameter_unit(&self, spec: &FitParameterSpec) -> Result<String, FitError> { Ok(self.parameter_conversion(spec, false, None)?.canonical_symbol) }

    /// `product_unit(spec, display=)`.
    pub fn product_unit(&self, spec: &FitParameterSpec, display: bool) -> Result<Arc<Unit>, FitError> {
        let mut symbols = Vec::new();
        for relation in [UnitRelation::Value, UnitRelation::Axis0] {
            let factor = FitParameterSpec { unit_relation: relation, solver_unit_relation: relation, ..spec.clone() };
            let symbol = if display { self.parameter_conversion(&factor, false, None)?.symbol } else { self.canonical_parameter_unit(&factor)? };
            if !symbol.is_empty() {
                symbols.push(symbol);
            }
        }
        let joined = if symbols.is_empty() { "1".to_string() } else { symbols.join("*") };
        self.registry().resolve(&joined).map_err(|e| FitError::value(e.0))
    }

    /// `parameter_conversion(spec, difference=, display_relation=)`.
    pub fn parameter_conversion(&self, spec: &FitParameterSpec, difference: bool, display_relation: Option<UnitRelation>) -> Result<FitParameterConversion, FitError> {
        let relation = display_relation.unwrap_or(spec.unit_relation);
        let solver_relation = spec.solver_unit_relation;
        let name = spec.name.clone();
        let simple = |symbol: &str, crossing: Crossing| FitParameterConversion { parameter: name.clone(), canonical_unit: None, display_unit: None, symbol: symbol.into(), crossing, canonical_symbol: symbol.into() };
        if relation == UnitRelation::ValueTimesAxis0 {
            if solver_relation != relation {
                return Err(FitError::value("product parameters cannot cross unit relations"));
            }
            let canonical = self.product_unit(spec, false)?;
            let display = self.product_unit(spec, true)?;
            if !canonical.compatible_with(&display) {
                return Err(FitError::value("fit product parameter units require compatible linear scales"));
            }
            return Ok(FitParameterConversion { parameter: name, symbol: display.symbol.clone(), canonical_symbol: canonical.symbol.clone(), canonical_unit: Some(canonical), display_unit: Some(display), crossing: Crossing::Span });
        }
        if matches!(relation, UnitRelation::Dimensionless | UnitRelation::Radian) {
            if solver_relation != relation {
                return Err(FitError::value("unit-free fit parameters cannot cross unit relations"));
            }
            return Ok(simple(if relation == UnitRelation::Radian { "rad" } else { "" }, Crossing::Point));
        }
        if relation == UnitRelation::Value && self.histogram {
            if solver_relation != UnitRelation::Value {
                return Err(FitError::value("histogram count parameters require value solver units"));
            }
            return Ok(simple("count", Crossing::Point));
        }
        if self.rolling_primary && matches!(relation, UnitRelation::Axis0 | UnitRelation::InverseAxis0) {
            if solver_relation != relation {
                return Err(FitError::value("rolling fit parameters cannot cross unit relations"));
            }
            return Ok(simple(if relation == UnitRelation::InverseAxis0 { "1/point" } else { "point" }, Crossing::Point));
        }
        if relation == UnitRelation::InverseAxis0 {
            if solver_relation != UnitRelation::InverseAxis0 {
                return Err(FitError::value("inverse-axis parameters cannot cross unit relations"));
            }
            let Some(quantity) = (self.quantities)(UnitRelation::Axis0) else { return Ok(simple("", Crossing::Inverse)) };
            let inverted = |unit: &Unit| -> String {
                match self.registry().inverse_for(unit) {
                    Some(found) => found.symbol.clone(),
                    None => {
                        if unit.symbol == "1" {
                            String::new()
                        } else {
                            format!("1/{}", unit.symbol)
                        }
                    }
                }
            };
            return Ok(FitParameterConversion {
                parameter: name,
                symbol: inverted(&quantity.display_unit),
                canonical_symbol: inverted(&quantity.canonical_unit),
                canonical_unit: Some(quantity.canonical_unit),
                display_unit: Some(quantity.display_unit),
                crossing: Crossing::Inverse,
            });
        }
        let source = (self.quantities)(solver_relation);
        let target = (self.quantities)(relation);
        let plain = |unit: &Unit| if unit.symbol == "1" { String::new() } else { unit.symbol.clone() };
        let (Some(source), Some(target)) = (source.clone(), target) else {
            let canonical_symbol = source.map(|s| plain(&s.canonical_unit)).unwrap_or_default();
            return Ok(FitParameterConversion { parameter: name, canonical_unit: None, display_unit: None, symbol: String::new(), crossing: Crossing::Point, canonical_symbol });
        };
        let canonical = source.canonical_unit;
        let display = target.display_unit;
        if !canonical.compatible_with(&display) {
            return Err(FitError::value("fit parameter solver and display units are incompatible"));
        }
        Ok(FitParameterConversion {
            parameter: name,
            symbol: plain(&display),
            canonical_symbol: plain(&canonical),
            crossing: if spec.affine_point && !difference { Crossing::Point } else { Crossing::Span },
            canonical_unit: Some(canonical),
            display_unit: Some(display),
        })
    }

    /// `display_parameter_value(spec, value, difference=, display_relation=)`.
    pub fn display_parameter_value(&self, spec: &FitParameterSpec, value: f64, difference: bool, display_relation: Option<UnitRelation>) -> Result<(f64, String), FitError> {
        let conversion = self.parameter_conversion(spec, difference, display_relation)?;
        Ok((conversion.to_display(value)?, conversion.symbol))
    }

    /// `canonical_parameter_value(spec, displayed)`.
    pub fn canonical_parameter_value(&self, spec: &FitParameterSpec, displayed: f64) -> Result<f64, FitError> { self.parameter_conversion(spec, false, None)?.to_canonical(displayed) }

    /// `fit_expression_target(model, expression)`.
    pub fn fit_expression_target(&self, model: &FitModelSpec, expression: &str) -> Result<FitExpressionTarget, FitError> {
        if expression.contains('\n') || expression.contains('\r') {
            return Err(FitError::value("fit expression must be one line of text"));
        }
        let mut target = FitExpressionTarget { model: model.model_id.clone(), ..Default::default() };
        let assignments: Vec<&str> = expression.split(',').map(crate::pystr::strip).collect();
        if !crate::pystr::strip(expression).is_empty() && !assignments.iter().all(|a| !a.is_empty()) {
            return Err(FitError::value("use comma-separated name=value assignments"));
        }
        for assignment in assignments.into_iter().filter(|a| !a.is_empty()) {
            if assignment.matches('=').count() != 1 {
                return Err(FitError::value("use name=value or name=guess(value)"));
            }
            let (symbol, raw) = assignment.split_once('=').expect("one equals sign");
            let (symbol, raw) = (crate::pystr::strip(symbol), crate::pystr::strip(raw));
            let Some(parameter) = model.parameter_for_symbol(symbol) else {
                return Err(FitError::value(format!("{} is not a parameter of this model; it takes {}", str_repr(symbol), model.symbols().join(", "))));
            };
            let name = parameter.name.clone();
            if target.fixed.iter().any(|(n, _)| *n == name) || target.initial.iter().any(|(n, _)| *n == name) {
                return Err(FitError::value(format!("repeated fit parameter {}", str_repr(symbol))));
            }
            let guessed = raw.starts_with("guess(") && raw.ends_with(')') && raw.len() >= 7;
            let number_text = if guessed { &raw[6..raw.len() - 1] } else { raw };
            let Some(displayed) = crate::pystr::float(number_text) else { return Err(FitError::value("use name=value or name=guess(value)")) };
            let converted = self.canonical_parameter_value(parameter, displayed)?;
            let (lower, upper) = parameter.bounds();
            if !converted.is_finite() || !(lower <= converted && converted <= upper) {
                return Err(FitError::value(format!("fit parameter {} is outside its domain", str_repr(symbol))));
            }
            if guessed { target.initial.push((name, converted)) } else { target.fixed.push((name, converted)) }
        }
        Ok(target)
    }

    /// `fit_expression_text(model, target)`: in the symbols the formula prints.
    pub fn fit_expression_text(&self, model: &FitModelSpec, fixed: &[(String, f64)], initial: &[(String, f64)]) -> Result<String, FitError> {
        let mut terms = Vec::new();
        for parameter in &model.parameters {
            let in_fixed = fixed.iter().find(|(n, _)| *n == parameter.name);
            let (value, is_fixed) = match in_fixed {
                Some((_, v)) => (*v, true),
                None => match initial.iter().find(|(n, _)| *n == parameter.name) {
                    Some((_, v)) => (*v, false),
                    None => continue,
                },
            };
            let shown = self.display_parameter_value(parameter, value, false, None)?.0;
            let literal = if shown == 0.0 { "0".to_string() } else { crate::pyfmt::repr(shown) };
            terms.push(if is_fixed { format!("{}={literal}", parameter.symbol) } else { format!("{}=guess({literal})", parameter.symbol) });
        }
        Ok(terms.join(", "))
    }
}
