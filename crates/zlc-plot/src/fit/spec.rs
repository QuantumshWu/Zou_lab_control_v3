//! Model declarations: parameter specs, presentation, reductions, the model
//! spec and its registry, and the thirteen built-in models
//! (`zlc_plot.fit.FitModelSpec` / `FitModelRegistry` /
//! `builtin_fit_models`).

use std::collections::BTreeSet;
use std::sync::{Arc, LazyLock, RwLock};

use super::models::Kernel;
use super::num::{nextafter, str_list_repr, str_repr, PI};

pub const DOMAIN_ANCHORED: &str = "domain_anchored";
pub const REGULAR_IMAGE_RADIAL: &str = "regular_image_radial";
pub const REGULAR_IMAGE_SEPARABLE: &str = "regular_image_separable";

/// The authored fit target fields (`FIT_TARGET_FIELDS`).
pub const FIT_TARGET_FIELDS: [&str; 9] = ["model", "expression", "selector_kind", "fixed", "initial", "bounds", "options", "fit_all_facets", "min_bic_gain"];

/// A refused declaration, in the oracle's words.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SpecError(pub String);

impl std::fmt::Display for SpecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.0) }
}

impl std::error::Error for SpecError {}

fn refuse<T>(message: impl Into<String>) -> Result<T, SpecError> { Err(SpecError(message.into())) }

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParameterDomain {
    Real,
    Positive,
    Nonnegative,
    PhaseRadians,
    UnitInterval,
}

impl ParameterDomain {
    pub fn value(self) -> &'static str {
        match self {
            ParameterDomain::Real => "real",
            ParameterDomain::Positive => "positive",
            ParameterDomain::Nonnegative => "nonnegative",
            ParameterDomain::PhaseRadians => "phase_radians",
            ParameterDomain::UnitInterval => "unit_interval",
        }
    }

    /// `FitParameterSpec.bounds`.
    pub fn bounds(self) -> (f64, f64) {
        match self {
            ParameterDomain::Real => (f64::NEG_INFINITY, f64::INFINITY),
            ParameterDomain::Positive => (nextafter(0.0, f64::INFINITY), f64::INFINITY),
            ParameterDomain::Nonnegative => (0.0, f64::INFINITY),
            ParameterDomain::PhaseRadians => (-PI, nextafter(PI, f64::NEG_INFINITY)),
            ParameterDomain::UnitInterval => (0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UnitRelation {
    Dimensionless,
    Value,
    ValueTimesAxis0,
    Axis0,
    Axis1,
    InverseAxis0,
    Radian,
}

impl UnitRelation {
    pub fn value(self) -> &'static str {
        match self {
            UnitRelation::Dimensionless => "dimensionless",
            UnitRelation::Value => "value",
            UnitRelation::ValueTimesAxis0 => "value_times_axis_0",
            UnitRelation::Axis0 => "axis_0",
            UnitRelation::Axis1 => "axis_1",
            UnitRelation::InverseAxis0 => "inverse_axis_0",
            UnitRelation::Radian => "radian",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum FitTarget {
    Series,
    Histogram,
    Image,
}

impl FitTarget {
    pub fn value(self) -> &'static str {
        match self {
            FitTarget::Series => "series",
            FitTarget::Histogram => "histogram",
            FitTarget::Image => "image",
        }
    }
}

/// `_SYMBOL_PATTERN.fullmatch`.
pub fn is_symbol(text: &str) -> bool {
    let bytes = text.as_bytes();
    if bytes.is_empty() || !bytes[0].is_ascii_alphabetic() {
        return false;
    }
    match_symbol(bytes, 0) == bytes.len()
}

/// The end of the longest `[A-Za-z][A-Za-z0-9]*(?:_[A-Za-z0-9]+)*` match
/// starting at `start` (which must be an ASCII letter).
fn match_symbol(bytes: &[u8], start: usize) -> usize {
    let mut end = start + 1;
    while end < bytes.len() && bytes[end].is_ascii_alphanumeric() {
        end += 1;
    }
    loop {
        if end + 1 < bytes.len() && bytes[end] == b'_' && bytes[end + 1].is_ascii_alphanumeric() {
            end += 2;
            while end < bytes.len() && bytes[end].is_ascii_alphanumeric() {
                end += 1;
            }
        } else {
            return end;
        }
    }
}

/// `formula_symbols(formula)`: every word a rendered formula shows.
pub fn formula_symbols(formula: &str) -> BTreeSet<String> {
    let text = formula.replace('$', " ");
    // re.sub(r"\\mathrm\s*\{([^{}]*)\}", r" \1 ", text)
    let mut stage = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && chars[i + 1..].iter().take(6).collect::<String>() == "mathrm" {
            let mut j = i + 7;
            while j < chars.len() && crate::pystr::isspace(chars[j]) {
                j += 1;
            }
            if j < chars.len() && chars[j] == '{' {
                let mut k = j + 1;
                while k < chars.len() && chars[k] != '{' && chars[k] != '}' {
                    k += 1;
                }
                if k < chars.len() && chars[k] == '}' {
                    stage.push(' ');
                    stage.extend(&chars[j + 1..k]);
                    stage.push(' ');
                    i = k + 1;
                    continue;
                }
            }
        }
        stage.push(chars[i]);
        i += 1;
    }
    // re.sub(r"\\([A-Za-z]+)", r" \1", text)
    let chars: Vec<char> = stage.chars().collect();
    let mut stage2 = String::with_capacity(stage.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\\' && i + 1 < chars.len() && chars[i + 1].is_ascii_alphabetic() {
            stage2.push(' ');
            i += 1;
            while i < chars.len() && chars[i].is_ascii_alphabetic() {
                stage2.push(chars[i]);
                i += 1;
            }
            continue;
        }
        stage2.push(chars[i]);
        i += 1;
    }
    let stage3: String = stage2.chars().map(|c| if c == '{' || c == '}' { ' ' } else { c }).collect();
    let bytes = stage3.as_bytes();
    let mut found = BTreeSet::new();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i].is_ascii_alphabetic() {
            let end = match_symbol(bytes, i);
            found.insert(stage3[i..end].to_string());
            i = end;
        } else {
            i += 1;
        }
    }
    found
}

/// `_text(value, field)`: stripped, non-empty.
fn text(value: &str, field: &str) -> Result<String, SpecError> {
    let stripped = crate::pystr::strip(value);
    if stripped.is_empty() {
        return refuse(format!("{field} must be non-empty"));
    }
    Ok(stripped.to_string())
}

/// One solver parameter and its display semantics.
#[derive(Debug, Clone, PartialEq)]
pub struct FitParameterSpec {
    pub name: String,
    pub unit_relation: UnitRelation,
    pub domain: ParameterDomain,
    pub display_label: Option<String>,
    pub affine_point: bool,
    pub solver_unit_relation: UnitRelation,
    pub symbol: String,
}

impl FitParameterSpec {
    pub fn new(
        name: &str,
        unit_relation: UnitRelation,
        domain: ParameterDomain,
        display_label: Option<&str>,
        affine_point: bool,
        solver_unit_relation: Option<UnitRelation>,
        symbol: Option<&str>,
    ) -> Result<Self, SpecError> {
        let name = text(name, "fit parameter name")?;
        let display_label = match display_label {
            Some(label) => Some(text(label, "fit parameter display_label")?),
            None => None,
        };
        let solver_unit_relation = solver_unit_relation.unwrap_or(unit_relation);
        let mut symbol = symbol.map(str::to_string);
        if symbol.is_none() && display_label.is_none() {
            symbol = Some(name.clone());
        }
        if symbol.is_none() {
            let label = display_label.as_ref().expect("label");
            let found = formula_symbols(label);
            if found.len() != 1 {
                let listed: Vec<&String> = found.iter().collect();
                return refuse(format!(
                    "fit parameter {} has display label {}, which reads as {} -- a label is one symbol, so say which one with symbol=",
                    str_repr(&name),
                    str_repr(label),
                    str_list_repr(&listed)
                ));
            }
            symbol = found.into_iter().next();
        }
        let symbol = text(&symbol.unwrap_or_default(), "fit parameter symbol")?;
        if !is_symbol(&symbol) {
            return refuse(format!("fit parameter symbol {} must be a bare word an operator can type: letters, digits and underscores", str_repr(&symbol)));
        }
        Ok(FitParameterSpec { name, unit_relation, domain, display_label, affine_point, solver_unit_relation, symbol })
    }

    pub fn bounds(&self) -> (f64, f64) { self.domain.bounds() }
}

/// How a component of a built-in presentation evaluates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentKind {
    /// `amplitude*(1-ratio)` population of a two-population model.
    LowerPopulation,
    /// `amplitude*ratio` population of a two-population model.
    UpperPopulation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FitComponentSpec {
    pub component_id: String,
    pub kind: ComponentKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FitEllipseGlyphSpec {
    pub center_parameters: [String; 2],
    pub radius_parameters: [String; 2],
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitPresentationSpec {
    pub components: Vec<FitComponentSpec>,
    pub ellipse_glyph: Option<FitEllipseGlyphSpec>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pin {
    Value(f64),
    Shared(String),
}

/// `FitReductionSpec`: the nested one-population model.
#[derive(Debug, Clone, PartialEq)]
pub struct FitReductionSpec {
    pub nested_model_id: String,
    pub shared: Vec<(String, String)>,
    pub pinned: Vec<(String, Pin)>,
}

impl FitReductionSpec {
    pub fn wide_names(&self) -> BTreeSet<String> {
        self.shared.iter().map(|(wide, _)| wide.clone()).chain(self.pinned.iter().map(|(name, _)| name.clone())).collect()
    }
    pub fn narrow_for(&self, wide: &str) -> Option<&str> { self.shared.iter().find(|(w, _)| w == wide).map(|(_, n)| n.as_str()) }
    pub fn pin(&self, wide: &str) -> Option<&Pin> { self.pinned.iter().find(|(w, _)| w == wide).map(|(_, p)| p) }
}

/// A model the caller supplies (the oracle's custom evaluator callables):
/// it is solved by the fallback lane.
pub trait CustomModel: Send + Sync + std::fmt::Debug {
    /// Values at the points (`coords[axis][point]`).
    fn evaluate(&self, coords: &[&[f64]], params: &[f64]) -> Vec<f64>;
    /// Row-major `points x parameters`, if analytic.
    fn jacobian(&self, _coords: &[&[f64]], _params: &[f64]) -> Option<Vec<f64>> { None }
    /// `initializer(coordinates, observations)`.
    fn initializer(&self, coords: &[&[f64]], observations: &[f64]) -> Vec<f64>;
    /// `candidate_initializer`, if any.
    fn candidates(&self, _coords: &[&[f64]], _observations: &[f64]) -> Option<Vec<Vec<f64>>> { None }
    /// `bounds_initializer`, if any: `(name, (low, high))`.
    fn bounds(&self, _coords: &[&[f64]], _observations: &[f64]) -> Option<Vec<(String, (Option<f64>, Option<f64>))>> { None }
}

#[derive(Debug, Clone)]
pub enum Evaluator {
    Builtin(Kernel),
    Custom(Arc<dyn CustomModel>),
}

/// One fit model.
#[derive(Debug, Clone)]
pub struct FitModelSpec {
    pub model_id: String,
    pub display_name: String,
    pub independent_arity: usize,
    pub parameters: Vec<FitParameterSpec>,
    pub headline: String,
    pub evaluator: Evaluator,
    pub targets: Vec<FitTarget>,
    pub formula: Option<String>,
    pub presentation: FitPresentationSpec,
    pub coordinate_relations: Vec<UnitRelation>,
    pub default_for: Vec<FitTarget>,
    pub capabilities: BTreeSet<String>,
    pub reduction: Option<FitReductionSpec>,
    /// Whether a compiled descriptor exists (built-ins only).
    pub compiled: bool,
    /// Whether the fallback lane reads data-derived bounds from the
    /// compiled preparation (`_compiled_bounds`; built-ins not confined
    /// to a histogram).
    pub compiled_bounds: bool,
    /// The origins `anchored_at` applied, innermost first.
    pub origins: Vec<f64>,
}

impl FitModelSpec {
    /// The validated constructor (`FitModelSpec.__post_init__`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model_id: &str,
        display_name: &str,
        independent_arity: usize,
        parameters: Vec<FitParameterSpec>,
        headline: &str,
        evaluator: Evaluator,
        targets: Vec<FitTarget>,
        formula: Option<&str>,
        presentation: FitPresentationSpec,
        coordinate_relations: Option<Vec<UnitRelation>>,
        default_for: Vec<FitTarget>,
        capabilities: &[&str],
        reduction: Option<FitReductionSpec>,
    ) -> Result<Self, SpecError> {
        let model_id = text(model_id, "fit model id")?;
        let display_name = text(display_name, "fit model display name")?;
        if !(1..=2).contains(&independent_arity) {
            return refuse("fit models support one or two independent axes");
        }
        if parameters.is_empty() {
            return refuse("fit model requires parameter specifications");
        }
        let names: Vec<&str> = parameters.iter().map(|p| p.name.as_str()).collect();
        if names.iter().collect::<BTreeSet<_>>().len() != names.len() {
            return refuse("fit parameter names must be unique");
        }
        let headline = text(headline, "fit model headline")?;
        if !names.contains(&headline.as_str()) {
            return refuse(format!("fit model headline must name a parameter: {}", str_repr(&headline)));
        }
        if targets.is_empty() || targets.iter().collect::<BTreeSet<_>>().len() != targets.len() {
            return refuse("fit model targets must be unique FitTarget values");
        }
        if default_for.iter().collect::<BTreeSet<_>>().len() != default_for.len() || default_for.iter().any(|t| !targets.contains(t)) {
            return refuse("default_for must be unique members of targets");
        }
        let symbols: Vec<&str> = parameters.iter().map(|p| p.symbol.as_str()).collect();
        if symbols.iter().collect::<BTreeSet<_>>().len() != symbols.len() {
            return refuse(format!("fit parameter symbols must be unique within a model: {}", str_list_repr(&symbols)));
        }
        if let Some(formula) = formula {
            let printed = formula_symbols(formula);
            let absent: Vec<&str> = symbols.iter().copied().filter(|s| !printed.contains(*s)).collect();
            if !absent.is_empty() {
                return refuse(format!("fit model {} asks the operator for {}, which its own formula never writes", str_repr(&model_id), str_list_repr(&absent)));
            }
        }
        let mut capability_set = BTreeSet::new();
        for capability in capabilities {
            let value = crate::pystr::strip(capability);
            if value.is_empty() {
                return refuse("fit capabilities must be non-empty text");
            }
            capability_set.insert(value.to_string());
        }
        let ids: Vec<&str> = presentation.components.iter().map(|c| c.component_id.as_str()).collect();
        if ids.iter().collect::<BTreeSet<_>>().len() != ids.len() {
            return refuse("fit presentation component ids must be unique");
        }
        if presentation.ellipse_glyph.is_some() && !presentation.components.is_empty() {
            return refuse("ellipse glyph presentation cannot also define components");
        }
        let coordinate_relations = coordinate_relations.unwrap_or_else(|| {
            if independent_arity == 1 { vec![UnitRelation::Axis0] } else { vec![UnitRelation::Axis0, UnitRelation::Axis1] }
        });
        if coordinate_relations.len() != independent_arity
            || coordinate_relations.iter().any(|r| !matches!(r, UnitRelation::Axis0 | UnitRelation::Axis1))
        {
            return refuse("coordinate_relations must identify one plot axis per independent axis");
        }
        if let Some(glyph) = &presentation.ellipse_glyph {
            if independent_arity != 2 {
                return refuse("ellipse glyph presentation requires a two-axis model");
            }
            let mut unknown: Vec<&str> = glyph.center_parameters.iter().chain(glyph.radius_parameters.iter()).map(String::as_str).filter(|n| !names.contains(n)).collect();
            unknown.sort();
            unknown.dedup();
            if !unknown.is_empty() {
                return refuse(format!("ellipse glyph names unknown parameters: {}", str_list_repr(&unknown)));
            }
        }
        if let Some(reduction) = &reduction {
            let name_set: BTreeSet<String> = names.iter().map(|s| s.to_string()).collect();
            if reduction.wide_names() != name_set {
                return refuse(format!("fit model {}: its reduction must account for every parameter, shared or pinned", str_repr(&model_id)));
            }
            if targets != [FitTarget::Histogram] {
                return refuse(format!("fit model {}: a reduction is weighed by the Poisson deviance of counts; only a histogram model has one", str_repr(&model_id)));
            }
        }
        let formula = match formula {
            Some(f) => Some(text(f, "fit model formula")?),
            None => None,
        };
        let compiled = matches!(evaluator, Evaluator::Builtin(_));
        Ok(FitModelSpec {
            model_id,
            display_name,
            independent_arity,
            parameters,
            headline,
            evaluator,
            targets,
            formula,
            presentation,
            coordinate_relations,
            default_for,
            capabilities: capability_set,
            reduction,
            compiled,
            compiled_bounds: false,
            origins: Vec::new(),
        })
    }

    pub fn parameter_names(&self) -> Vec<&str> { self.parameters.iter().map(|p| p.name.as_str()).collect() }
    pub fn symbols(&self) -> Vec<&str> { self.parameters.iter().map(|p| p.symbol.as_str()).collect() }
    pub fn parameter_for_symbol(&self, symbol: &str) -> Option<&FitParameterSpec> { self.parameters.iter().find(|p| p.symbol == symbol) }

    pub fn parameter_index(&self, name: &str) -> Result<usize, SpecError> {
        let wanted = crate::pystr::strip(name);
        self.parameters.iter().position(|p| p.name == wanted).ok_or_else(|| SpecError(format!("unknown fit parameter: {}", str_repr(name))))
    }

    pub fn kernel(&self) -> Option<Kernel> {
        match &self.evaluator {
            Evaluator::Builtin(kernel) => Some(*kernel),
            Evaluator::Custom(_) => None,
        }
    }

    /// `targets == (FitTarget.HISTOGRAM,)`.
    pub fn counted(&self) -> bool { self.targets == [FitTarget::Histogram] }

    /// `anchored_at(origin)`: the same model read from `origin`.
    pub fn anchored_at(self: &Arc<Self>, origin: f64) -> Result<Arc<Self>, SpecError> {
        if self.independent_arity != 1 {
            return refuse("only single-axis fit models can be anchored");
        }
        if !origin.is_finite() {
            return refuse("fit anchor must be finite");
        }
        if origin == 0.0 {
            return Ok(self.clone());
        }
        let mut anchored = (**self).clone();
        anchored.origins.push(origin);
        anchored.capabilities.remove(DOMAIN_ANCHORED);
        Ok(Arc::new(anchored))
    }

    /// The first coordinate axis as the evaluator reads it: every applied
    /// origin subtracted in turn.
    pub fn anchored_axis(&self, axis: &[f64]) -> Vec<f64> {
        let mut values = axis.to_vec();
        for origin in &self.origins {
            for v in values.iter_mut() {
                *v -= origin;
            }
        }
        values
    }
}

/// The model registry (`FitModelRegistry`).
#[derive(Debug, Default)]
pub struct FitModelRegistry {
    models: RwLock<Vec<Arc<FitModelSpec>>>,
}

impl FitModelRegistry {
    pub fn new(models: impl IntoIterator<Item = Arc<FitModelSpec>>) -> Result<Self, SpecError> {
        let registry = FitModelRegistry::default();
        for model in models {
            registry.register(model, false)?;
        }
        Ok(registry)
    }

    pub fn register(&self, model: Arc<FitModelSpec>, replace: bool) -> Result<(), SpecError> {
        if model.compiled_bounds && model.counted() {
            return refuse(format!(
                "fit model {} declares data-derived bounds and is confined to a histogram; the solver has no rule for combining those two",
                str_repr(&model.model_id)
            ));
        }
        let mut models = self.models.write().expect("registry lock");
        let exists = models.iter().position(|m| m.model_id == model.model_id);
        if exists.is_some() && !replace {
            return refuse(format!("fit model already registered: {}", model.model_id));
        }
        for target in &model.default_for {
            if let Some(conflict) = models.iter().find(|m| m.model_id != model.model_id && m.default_for.contains(target)) {
                return refuse(format!("fit target {} already has default model {}", str_repr(target.value()), str_repr(&conflict.model_id)));
            }
        }
        match exists {
            Some(index) => models[index] = model,
            None => models.push(model),
        }
        Ok(())
    }

    pub fn get(&self, model_id: &str) -> Result<Arc<FitModelSpec>, SpecError> {
        let selected = text(model_id, "fit model id")?;
        let models = self.models.read().expect("registry lock");
        models.iter().find(|m| m.model_id == selected).cloned().ok_or_else(|| SpecError(format!("unknown fit model: {}", str_repr(model_id))))
    }

    pub fn models(&self) -> Vec<Arc<FitModelSpec>> { self.models.read().expect("registry lock").clone() }

    /// Compatible models with the target's authored default first.
    pub fn models_for(&self, target: FitTarget) -> Vec<Arc<FitModelSpec>> {
        let mut models: Vec<Arc<FitModelSpec>> = self.models().into_iter().filter(|m| m.targets.contains(&target)).collect();
        models.sort_by_key(|m| !m.default_for.contains(&target));
        models
    }
}

fn parameter(name: &str, relation: UnitRelation, domain: ParameterDomain, label: &str, affine: bool) -> FitParameterSpec {
    FitParameterSpec::new(name, relation, domain, Some(label), affine, None, None).expect("built-in parameter")
}

fn reduction(nested: &str, shared: &[(&str, &str)], pinned: Vec<(&str, Pin)>) -> FitReductionSpec {
    FitReductionSpec {
        nested_model_id: nested.into(),
        shared: shared.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect(),
        pinned: pinned.into_iter().map(|(n, p)| (n.to_string(), p)).collect(),
    }
}

fn two_components() -> FitPresentationSpec {
    FitPresentationSpec {
        components: vec![
            FitComponentSpec { component_id: "A".into(), kind: ComponentKind::LowerPopulation },
            FitComponentSpec { component_id: "B".into(), kind: ComponentKind::UpperPopulation },
        ],
        ellipse_glyph: None,
    }
}

/// `_builtin_fit_models()`, wrapped as `builtin_fit_models()` wraps it.
fn build_builtins() -> Vec<Arc<FitModelSpec>> {
    use ParameterDomain::*;
    use UnitRelation::*;
    let series = vec![FitTarget::Series];
    let histogram = vec![FitTarget::Histogram];
    let image = vec![FitTarget::Image];
    let mut models = vec![
        FitModelSpec::new(
            "lorentzian",
            "Lorentzian",
            1,
            vec![
                parameter("center", Axis0, Real, r"$x_0$", true),
                parameter("fwhm", Axis0, Positive, r"$\mathrm{FWHM}$", false),
                parameter("amplitude", Value, Real, r"$H$", false),
                parameter("offset", Value, Real, r"$B$", true),
            ],
            "center",
            Evaluator::Builtin(Kernel::Lorentzian),
            series.clone(),
            Some(r"$f(x)=H\frac{(\mathrm{FWHM}/2)^2}{(x-x_0)^2+(\mathrm{FWHM}/2)^2}+B$"),
            FitPresentationSpec::default(),
            None,
            vec![FitTarget::Series],
            &[],
            None,
        ),
        FitModelSpec::new(
            "gaussian_offset",
            "Gaussian with offset",
            1,
            vec![
                parameter("amplitude", Value, Real, r"$A$", false),
                parameter("offset", Value, Real, r"$B$", true),
                parameter("sigma", Axis0, Positive, r"$\sigma$", false),
                parameter("center", Axis0, Real, r"$x_0$", true),
            ],
            "center",
            Evaluator::Builtin(Kernel::Gaussian),
            series.clone(),
            Some(r"$f(x)=A e^{-\frac{1}{2}((x-x_0)/\sigma)^2}+B$"),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[],
            None,
        ),
        FitModelSpec::new(
            "histogram_gaussian",
            "Single Gaussian",
            1,
            vec![
                parameter("amplitude", ValueTimesAxis0, Nonnegative, r"$Nw$", false),
                parameter("center", Axis0, Real, r"$x_0$", true),
                parameter("sigma", Axis0, Positive, r"$\sigma$", false),
            ],
            "center",
            Evaluator::Builtin(Kernel::Histogram),
            histogram.clone(),
            Some(r"$f(x)=\frac{Nw}{\sigma\sqrt{2\pi}}e^{-\frac{1}{2}((x-x_0)/\sigma)^2}$"),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[],
            None,
        ),
        FitModelSpec::new(
            "bimodal_gaussian",
            "Bimodal Gaussian",
            1,
            vec![
                parameter("amplitude", ValueTimesAxis0, Nonnegative, r"$Nw$", false),
                parameter("center", Axis0, Real, r"$x_0$", true),
                parameter("sigma", Axis0, Positive, r"$\sigma$", false),
                parameter("delta_center", Axis0, Nonnegative, r"$\delta$", false),
                parameter("sigma_B", Axis0, Positive, r"$\sigma_B$", false),
                parameter("ratio", Dimensionless, UnitInterval, r"$r$", false),
            ],
            "ratio",
            Evaluator::Builtin(Kernel::Bimodal),
            histogram.clone(),
            Some(concat!(
                r"$f(x)=Nw\left[\frac{1-r}{\sigma\sqrt{2\pi}}",
                r"e^{-\frac{1}{2}((x-x_0)/\sigma)^2}",
                r"+\frac{r}{\sigma_B\sqrt{2\pi}}",
                r"e^{-\frac{1}{2}((x-x_0-\delta)/\sigma_B)^2}\right]$"
            )),
            two_components(),
            None,
            vec![FitTarget::Histogram],
            &[],
            Some(reduction(
                "histogram_gaussian",
                &[("amplitude", "amplitude"), ("center", "center"), ("sigma", "sigma")],
                vec![("delta_center", Pin::Value(0.0)), ("sigma_B", Pin::Shared("sigma".into())), ("ratio", Pin::Value(0.5))],
            )),
        ),
        FitModelSpec::new(
            "histogram_poisson_gaussian",
            "Poisson-Gaussian",
            1,
            vec![
                parameter("amplitude", ValueTimesAxis0, Nonnegative, r"$Nw$", false),
                parameter("rate", Axis0, Nonnegative, r"$\lambda$", true),
                parameter("sigma", Axis0, Positive, r"$\sigma$", false),
            ],
            "rate",
            Evaluator::Builtin(Kernel::Poisson),
            histogram.clone(),
            Some(concat!(
                r"$f(x)=Nw\,P(x;\lambda,\sigma),\ ",
                r"P(x;\lambda,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}\int_0^{\infty}",
                r"\frac{\lambda^u e^{-\lambda}}{\Gamma(u+1)}",
                r"\,e^{-\frac{1}{2}((x-u)/\sigma)^2}\,du$"
            )),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[],
            None,
        ),
        FitModelSpec::new(
            "bimodal_poisson_gaussian",
            "Bimodal Poisson-Gaussian",
            1,
            vec![
                parameter("amplitude", ValueTimesAxis0, Nonnegative, r"$Nw$", false),
                parameter("rate", Axis0, Nonnegative, r"$\lambda$", true),
                parameter("sigma", Axis0, Positive, r"$\sigma$", false),
                parameter("delta_rate", Axis0, Nonnegative, r"$\delta$", false),
                parameter("sigma_B", Axis0, Positive, r"$\sigma_B$", false),
                parameter("ratio", Dimensionless, UnitInterval, r"$r$", false),
            ],
            "ratio",
            Evaluator::Builtin(Kernel::PoissonBimodal),
            histogram.clone(),
            Some(concat!(
                r"$f(x)=Nw\left[(1-r)\,P(x;\lambda,\sigma)",
                r"+r\,P(x;\lambda+\delta,\sigma_B)\right],\ ",
                r"P(x;\lambda,\sigma)=\frac{1}{\sigma\sqrt{2\pi}}\int_0^{\infty}",
                r"\frac{\lambda^u e^{-\lambda}}{\Gamma(u+1)}",
                r"\,e^{-\frac{1}{2}((x-u)/\sigma)^2}\,du$"
            )),
            two_components(),
            None,
            vec![],
            &[],
            Some(reduction(
                "histogram_poisson_gaussian",
                &[("amplitude", "amplitude"), ("rate", "rate"), ("sigma", "sigma")],
                vec![("delta_rate", Pin::Value(0.0)), ("sigma_B", Pin::Shared("sigma".into())), ("ratio", Pin::Value(0.5))],
            )),
        ),
        FitModelSpec::new(
            "symmetric_lorentzian_doublet",
            "Symmetric Lorentzian doublet",
            1,
            vec![
                parameter("center", Axis0, Real, r"$x_0$", true),
                parameter("common_fwhm", Axis0, Positive, r"$\mathrm{FWHM}$", false),
                parameter("component_amplitude", Value, Real, r"$H$", false),
                parameter("offset", Value, Real, r"$B$", true),
                parameter("center_splitting", Axis0, Nonnegative, r"$\delta$", false),
            ],
            "center",
            Evaluator::Builtin(Kernel::Doublet),
            series.clone(),
            Some(r"$f(x)=H[L(x;x_0-\delta/2,\mathrm{FWHM})+L(x;x_0+\delta/2,\mathrm{FWHM})]+B$"),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[],
            None,
        ),
        FitModelSpec::new(
            "damped_sine",
            "Damped sine",
            1,
            vec![
                parameter("amplitude", Value, Nonnegative, r"$A$", false),
                parameter("offset", Value, Real, r"$B$", true),
                parameter("baseband_frequency", InverseAxis0, Positive, r"$f$", false),
                parameter("decay_time", Axis0, Positive, r"$\tau$", false),
                parameter("phase", Radian, PhaseRadians, r"$\phi$", false),
            ],
            "decay_time",
            Evaluator::Builtin(Kernel::Damped),
            series.clone(),
            Some(r"$f(t)=A\sin(2\pi f (t-t_0)+\phi)e^{-(t-t_0)/\tau}+B$"),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[DOMAIN_ANCHORED],
            None,
        ),
        FitModelSpec::new(
            "exponential_decay",
            "Exponential decay",
            1,
            vec![
                parameter("amplitude", Value, Real, r"$A$", false),
                parameter("offset", Value, Real, r"$B$", true),
                parameter("decay_time", Axis0, Positive, r"$\tau$", false),
            ],
            "decay_time",
            Evaluator::Builtin(Kernel::Exponential),
            series.clone(),
            Some(r"$f(t)=A e^{-(t-t_0)/\tau}+B$"),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[DOMAIN_ANCHORED],
            None,
        ),
        FitModelSpec::new(
            "saturation",
            "Saturation",
            1,
            vec![
                parameter("asymptote", Value, Real, r"$A$", true),
                parameter("numerator", ValueTimesAxis0, Real, r"$B$", false),
                parameter("shift", Axis0, Real, r"$C$", false),
            ],
            "asymptote",
            Evaluator::Builtin(Kernel::Saturation),
            series.clone(),
            Some(r"$f(x)=(A x+B)/(x+C)$"),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[],
            None,
        ),
        FitModelSpec::new(
            "release_recapture",
            "Release\u{2013}recapture",
            1,
            vec![
                parameter("amplitude", Value, Nonnegative, r"$A$", false),
                parameter("offset", Value, Real, r"$B$", true),
                parameter("eta", Dimensionless, Positive, r"$\eta$", false),
                parameter("frequency", InverseAxis0, Positive, r"$f$", false),
            ],
            "eta",
            Evaluator::Builtin(Kernel::ReleaseRecapture),
            series.clone(),
            Some(concat!(r"$P(t)=A\,[1-e^{-\eta s}]/[1-e^{-\eta}]+B$", "\n", r"$s=e^{-W_0((2\pi f t)^2)}$")),
            FitPresentationSpec::default(),
            None,
            vec![],
            &[],
            None,
        ),
        FitModelSpec::new(
            "anisotropic_gaussian_center",
            "Anisotropic Gaussian center",
            2,
            vec![
                parameter("amplitude", Value, Real, r"$A$", false),
                parameter("offset", Value, Real, r"$B$", true),
                parameter("radius_x", Axis0, Positive, r"$R_x$", false),
                parameter("radius_y", Axis1, Positive, r"$R_y$", false),
                parameter("center_x", Axis0, Real, r"$x_0$", true),
                parameter("center_y", Axis1, Real, r"$y_0$", true),
            ],
            "center_x",
            Evaluator::Builtin(Kernel::Anisotropic),
            image.clone(),
            Some(r"$f(x,y)=A e^{-((x-x_0)^2/R_x^2+(y-y_0)^2/R_y^2)}+B$"),
            FitPresentationSpec {
                components: vec![],
                ellipse_glyph: Some(FitEllipseGlyphSpec {
                    center_parameters: ["center_x".into(), "center_y".into()],
                    radius_parameters: ["radius_x".into(), "radius_y".into()],
                }),
            },
            Some(vec![Axis0, Axis1]),
            vec![],
            &[REGULAR_IMAGE_SEPARABLE],
            None,
        ),
        FitModelSpec::new(
            "radial_gaussian_center",
            "Radial Gaussian center",
            2,
            vec![
                parameter("amplitude", Value, Real, r"$A$", false),
                parameter("offset", Value, Real, r"$B$", true),
                parameter("one_over_e_radius", Axis0, Positive, r"$R$", false),
                parameter("center_x", Axis0, Real, r"$x_0$", true),
                FitParameterSpec::new("center_y", Axis1, Real, Some(r"$y_0$"), true, Some(Axis0), None).expect("built-in parameter"),
            ],
            "center_x",
            Evaluator::Builtin(Kernel::Radial),
            image.clone(),
            Some(r"$f(x,y)=A e^{-((x-x_0)^2+(y-y_0)^2)/R^2}+B$"),
            FitPresentationSpec {
                components: vec![],
                ellipse_glyph: Some(FitEllipseGlyphSpec {
                    center_parameters: ["center_x".into(), "center_y".into()],
                    radius_parameters: ["one_over_e_radius".into(), "one_over_e_radius".into()],
                }),
            },
            Some(vec![Axis0, Axis0]),
            vec![FitTarget::Image],
            &[REGULAR_IMAGE_RADIAL],
            None,
        ),
    ];
    for model in models.iter_mut() {
        let model = model.as_mut().expect("built-in model");
        model.compiled_bounds = !model.counted();
    }
    models.into_iter().map(|m| Arc::new(m.expect("built-in model"))).collect()
}

static BUILTINS: LazyLock<Vec<Arc<FitModelSpec>>> = LazyLock::new(build_builtins);

/// `builtin_fit_models()`: the catalogue, built once.
pub fn builtin_fit_models() -> Vec<Arc<FitModelSpec>> { BUILTINS.clone() }

/// `default_fit_registry()`.
pub fn default_fit_registry() -> FitModelRegistry { FitModelRegistry::new(builtin_fit_models()).expect("built-in registry") }
