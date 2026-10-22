//! Options, errors, seeds/bounds and the immutable result of a fit
//! (`FitOptions`, `FitResult`, `FacetFitBatchResult` numerics).

use std::sync::{Arc, OnceLock};

use super::num::str_repr;
use super::spec::FitModelSpec;

/// What a fit can raise, by the oracle's exception class; `Display` is
/// `str(error)`.
#[derive(Debug, Clone, PartialEq)]
pub enum FitError {
    Value(String),
    Type(String),
    Runtime(String),
    FloatingPoint(String),
    ZeroDivision,
    /// A kernel table past `num::ALLOCATION_LIMIT` (`MemoryError`).
    Memory,
    LinAlg(String),
    Cancelled,
    Deadline,
}

impl FitError {
    pub fn value(message: impl Into<String>) -> Self { FitError::Value(message.into()) }
    pub fn type_error(message: impl Into<String>) -> Self { FitError::Type(message.into()) }
    pub fn runtime(message: impl Into<String>) -> Self { FitError::Runtime(message.into()) }

    /// `str(error) or type(error).__name__`: what a failed batch cell reports.
    pub fn failure_text(&self) -> String {
        let text = self.to_string();
        if !text.is_empty() {
            return text;
        }
        match self {
            FitError::Value(_) => "ValueError",
            FitError::Type(_) => "TypeError",
            FitError::Runtime(_) => "RuntimeError",
            FitError::FloatingPoint(_) => "FloatingPointError",
            FitError::ZeroDivision => "ZeroDivisionError",
            FitError::Memory => "MemoryError",
            FitError::LinAlg(_) => "LinAlgError",
            FitError::Cancelled => "FitCancelled",
            FitError::Deadline => "FitDeadlineExceeded",
        }
        .to_string()
    }

}

impl std::fmt::Display for FitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FitError::Value(m) | FitError::Type(m) | FitError::Runtime(m) | FitError::FloatingPoint(m) | FitError::LinAlg(m) => f.write_str(m),
            FitError::ZeroDivision => f.write_str("division by zero"),
            FitError::Memory => write!(f, "{}", super::num::KernelError::Allocation),
            FitError::Cancelled => f.write_str("fit cancelled"),
            FitError::Deadline => f.write_str("fit deadline exceeded"),
        }
    }
}

impl std::error::Error for FitError {}

impl From<super::spec::SpecError> for FitError {
    fn from(error: super::spec::SpecError) -> Self { FitError::Value(error.0) }
}

impl From<super::num::KernelError> for FitError {
    fn from(error: super::num::KernelError) -> Self {
        match error { super::num::KernelError::ZeroDivision => FitError::ZeroDivision, super::num::KernelError::Allocation => FitError::Memory }
    }
}

/// Parameter values as authored: by name (a mapping, in its order) or by
/// position (a sequence).
#[derive(Debug, Clone, PartialEq)]
pub enum ParamValues {
    Named(Vec<(String, f64)>),
    Positional(Vec<f64>),
}

impl ParamValues {
    pub fn get(&self, name: &str) -> Option<f64> {
        match self {
            ParamValues::Named(items) => items.iter().find(|(n, _)| n == name).map(|(_, v)| *v),
            ParamValues::Positional(_) => None,
        }
    }
    pub fn len(&self) -> usize {
        match self {
            ParamValues::Named(items) => items.len(),
            ParamValues::Positional(values) => values.len(),
        }
    }
    pub fn is_empty(&self) -> bool { self.len() == 0 }
}

pub type Bound = (Option<f64>, Option<f64>);

/// Authored bounds by parameter name, in authored order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Bounds(pub Vec<(String, Bound)>);

impl Bounds {
    pub fn get(&self, name: &str) -> Option<Bound> { self.0.iter().find(|(n, _)| n == name).map(|(_, b)| *b) }
    pub fn contains(&self, name: &str) -> bool { self.0.iter().any(|(n, _)| n == name) }
    pub fn names(&self) -> impl Iterator<Item = &str> { self.0.iter().map(|(n, _)| n.as_str()) }
    pub fn is_empty(&self) -> bool { self.0.is_empty() }
    /// `dict.update`/`__setitem__`: replace in place or append.
    pub fn set(&mut self, name: &str, bound: Bound) {
        if let Some(entry) = self.0.iter_mut().find(|(n, _)| n == name) {
            entry.1 = bound;
        } else {
            self.0.push((name.to_string(), bound));
        }
    }
}

/// `FitOptions`.
#[derive(Debug, Clone, PartialEq)]
pub struct FitOptions {
    pub loss: String,
    pub max_nfev: i64,
    pub max_point_evaluations: Option<i64>,
    pub deadline_seconds: Option<f64>,
    pub max_exact_points: Option<i64>,
    pub min_bic_gain: Option<f64>,
}

impl Default for FitOptions {
    fn default() -> Self {
        FitOptions {
            loss: "linear".into(),
            max_nfev: 5000,
            max_point_evaluations: Some(1_000_000_000),
            deadline_seconds: None,
            max_exact_points: Some(4096),
            min_bic_gain: Some(super::DECISIVE_BIC_GAIN),
        }
    }
}

impl FitOptions {
    /// `FitOptions.__post_init__`.
    pub fn validated(self) -> Result<Self, FitError> {
        let loss = crate::pystr::strip(&self.loss).to_string();
        if loss.is_empty() {
            return Err(FitError::value("fit loss must be non-empty"));
        }
        if !matches!(loss.as_str(), "linear" | "soft_l1" | "huber" | "cauchy" | "arctan") {
            return Err(FitError::value("unsupported scipy least-squares loss"));
        }
        if self.max_nfev <= 0 {
            return Err(FitError::value("max_nfev must be a positive integer"));
        }
        if let Some(budget) = self.max_point_evaluations {
            if budget <= 0 {
                return Err(FitError::value("max_point_evaluations must be positive"));
            }
        }
        if let Some(deadline) = self.deadline_seconds {
            if !deadline.is_finite() {
                return Err(FitError::value("deadline_seconds must be finite"));
            }
            if deadline <= 0.0 {
                return Err(FitError::value("deadline_seconds must be positive"));
            }
        }
        if let Some(points) = self.max_exact_points {
            if points <= 0 {
                return Err(FitError::value("max_exact_points must be a positive integer"));
            }
        }
        if let Some(gain) = self.min_bic_gain {
            if gain.is_nan() {
                return Err(FitError::value("min_bic_gain cannot be NaN"));
            }
        }
        Ok(FitOptions { loss, ..self })
    }

    /// `evaluation_budget(points)`.
    pub fn evaluation_budget(&self, points: usize) -> i64 {
        match self.max_point_evaluations {
            None => self.max_nfev,
            Some(_) if points == 0 => self.max_nfev,
            Some(budget) => 1.max(self.max_nfev.min(budget.div_euclid(points as i64))),
        }
    }
}

/// The solver's own account of a compiled solve (not part of the oracle's
/// `FitResult`; recorded for diagnostics and the golden tests).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SolveDiagnostics {
    pub status: i32,
    pub nfev: i32,
    pub njev: i32,
    pub iterations: i32,
    pub winner_seed: i32,
    pub cost: f64,
    pub raw_rss: f64,
}

/// The per-observation arrays of a result, possibly deferred (a regular
/// image retains only its input and parameters).
#[derive(Clone)]
pub enum ObservationArrays {
    Eager { fitted: Arc<[f64]>, residuals: Arc<[f64]>, indices: Arc<[i64]> },
    Deferred(Arc<DeferredArrays>),
}

pub type ArraysLoader = Box<dyn Fn() -> (Vec<f64>, Vec<f64>, Vec<i64>) + Send + Sync>;

/// `_DeferredFitData`: computed on first access, then cached.
pub struct DeferredArrays {
    loader: ArraysLoader,
    cell: OnceLock<(Arc<[f64]>, Arc<[f64]>, Arc<[i64]>)>,
}

impl DeferredArrays {
    pub fn new(loader: ArraysLoader) -> Self { DeferredArrays { loader, cell: OnceLock::new() } }
    pub fn arrays(&self) -> &(Arc<[f64]>, Arc<[f64]>, Arc<[i64]>) {
        self.cell.get_or_init(|| {
            let (fitted, residuals, indices) = (self.loader)();
            (fitted.into(), residuals.into(), indices.into())
        })
    }
    pub fn is_materialized(&self) -> bool { self.cell.get().is_some() }
}

impl std::fmt::Debug for DeferredArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeferredArrays").field("materialized", &self.is_materialized()).finish()
    }
}

impl std::fmt::Debug for ObservationArrays {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObservationArrays::Eager { fitted, .. } => write!(f, "Eager({} points)", fitted.len()),
            ObservationArrays::Deferred(d) => write!(f, "{d:?}"),
        }
    }
}

/// An immutable fit output (`FitResult`).
#[derive(Debug, Clone)]
pub struct FitResult {
    pub model: Arc<FitModelSpec>,
    pub parameter_values: Vec<f64>,
    pub standard_errors: Vec<f64>,
    /// Row-major `parameters x parameters`.
    pub covariance: Vec<f64>,
    pub arrays: ObservationArrays,
    pub source_revision: u64,
    pub success: bool,
    pub message: String,
    pub reduced_chi_square: f64,
    pub covariance_valid: bool,
    pub parameter_units: Vec<String>,
    pub batch_revision: u64,
    pub fixed_parameter_names: Vec<String>,
    pub reduced: bool,
    pub evidence: f64,
    pub diagnostics: Option<SolveDiagnostics>,
}

fn allclose(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a == b {
        return true;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

impl FitResult {
    /// The validating constructor (`FitResult.__post_init__`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        model: Arc<FitModelSpec>,
        parameter_values: Vec<f64>,
        standard_errors: Vec<f64>,
        covariance: Vec<f64>,
        arrays: ObservationArrays,
        source_revision: u64,
        success: bool,
        message: &str,
        reduced_chi_square: f64,
        covariance_valid: bool,
        fixed_parameter_names: Vec<String>,
    ) -> Result<Self, FitError> {
        let count = model.parameters.len();
        if parameter_values.len() != count || standard_errors.len() != count {
            return Err(FitError::value("fit parameter/error shapes disagree with model"));
        }
        if covariance.len() != count * count {
            return Err(FitError::value("fit covariance shape disagrees with model"));
        }
        let mut fixed_indices = Vec::new();
        for name in &fixed_parameter_names {
            fixed_indices.push(model.parameter_index(name)?);
        }
        let mut covariance = covariance;
        let mut errors = standard_errors;
        if covariance_valid {
            if covariance.iter().any(|v| !v.is_finite()) || errors.iter().any(|v| !v.is_finite()) {
                return Err(FitError::value("valid fit covariance and standard errors must be finite"));
            }
            for i in 0..count {
                for j in 0..count {
                    if !allclose(covariance[i * count + j], covariance[j * count + i], 1e-12, 1e-15) {
                        return Err(FitError::value("valid fit covariance must be symmetric"));
                    }
                }
            }
            if (0..count).any(|i| covariance[i * count + i] < 0.0) || errors.iter().any(|&e| e < 0.0) {
                return Err(FitError::value("valid fit variances and standard errors cannot be negative"));
            }
            if (0..count).any(|i| !allclose(errors[i] * errors[i], covariance[i * count + i], 1e-10, 1e-15)) {
                return Err(FitError::value("standard errors must agree with fit covariance"));
            }
        } else {
            covariance = vec![f64::NAN; count * count];
            errors = vec![f64::NAN; count];
            for &f in &fixed_indices {
                for k in 0..count {
                    covariance[f * count + k] = 0.0;
                    covariance[k * count + f] = 0.0;
                }
                errors[f] = 0.0;
            }
        }
        if let ObservationArrays::Eager { fitted, residuals, indices } = &arrays {
            if fitted.len() != residuals.len() {
                return Err(FitError::value("fitted values and residuals must have equal shape"));
            }
            if indices.len() != fitted.len() {
                return Err(FitError::value("selected indices must identify every fitted observation"));
            }
        }
        if !reduced_chi_square.is_finite() {
            return Err(FitError::value("reduced_chi_square must be finite"));
        }
        Ok(FitResult {
            parameter_units: vec![String::new(); count],
            model,
            parameter_values,
            standard_errors: errors,
            covariance,
            arrays,
            source_revision,
            success,
            message: crate::pystr::strip(message).to_string(),
            reduced_chi_square,
            covariance_valid,
            batch_revision: 0,
            fixed_parameter_names,
            reduced: false,
            evidence: f64::NAN,
            diagnostics: None,
        })
    }

    pub fn eager(fitted: Vec<f64>, residuals: Vec<f64>, indices: Vec<i64>) -> ObservationArrays {
        ObservationArrays::Eager { fitted: fitted.into(), residuals: residuals.into(), indices: indices.into() }
    }

    pub fn fitted_values(&self) -> Arc<[f64]> {
        match &self.arrays {
            ObservationArrays::Eager { fitted, .. } => fitted.clone(),
            ObservationArrays::Deferred(d) => d.arrays().0.clone(),
        }
    }
    pub fn residuals(&self) -> Arc<[f64]> {
        match &self.arrays {
            ObservationArrays::Eager { residuals, .. } => residuals.clone(),
            ObservationArrays::Deferred(d) => d.arrays().1.clone(),
        }
    }
    pub fn selected_indices(&self) -> Arc<[i64]> {
        match &self.arrays {
            ObservationArrays::Eager { indices, .. } => indices.clone(),
            ObservationArrays::Deferred(d) => d.arrays().2.clone(),
        }
    }

    pub fn parameter_names(&self) -> Vec<&str> { self.model.parameter_names() }

    /// `parameters[name]`.
    pub fn parameter(&self, name: &str) -> Option<f64> { self.model.parameters.iter().position(|p| p.name == name).map(|i| self.parameter_values[i]) }
    pub fn error(&self, name: &str) -> Option<f64> { self.model.parameters.iter().position(|p| p.name == name).map(|i| self.standard_errors[i]) }

    /// `parameter_error_validity`.
    pub fn parameter_error_validity(&self) -> Vec<(String, bool)> {
        let valid = self.success && self.covariance_valid;
        self.model.parameters.iter().map(|p| (p.name.clone(), valid && !self.fixed_parameter_names.contains(&p.name))).collect()
    }

    /// `with_parameter_units(units)`.
    pub fn with_parameter_units(&self, units: &[(String, String)]) -> Result<Self, FitError> {
        let names = self.model.parameter_names();
        let mut unknown: Vec<&str> = units.iter().map(|(n, _)| n.as_str()).filter(|n| !names.contains(n)).collect();
        if !unknown.is_empty() {
            unknown.sort();
            return Err(FitError::value(format!("fit result units name unknown parameters: {}", super::num::str_list_repr(&unknown))));
        }
        let mut clone = self.clone();
        clone.parameter_units = names.iter().map(|n| units.iter().find(|(u, _)| u == n).map(|(_, s)| crate::pystr::strip(s).to_string()).unwrap_or_default()).collect();
        Ok(clone)
    }

    pub fn with_batch_revision(&self, batch_revision: u64) -> Self {
        let mut clone = self.clone();
        clone.batch_revision = batch_revision;
        clone
    }

    pub fn with_evidence(&self, evidence: f64) -> Self {
        let mut clone = self.clone();
        clone.evidence = evidence;
        clone
    }
}

/// `_FitBatchNumerics`: the numerical column view of a batch.
pub fn batch_parameter_columns(results: &[Option<FitResult>], model: &FitModelSpec) -> Vec<(String, Vec<f64>)> {
    model
        .parameters
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let values = results
                .iter()
                .map(|r| match r {
                    Some(fit) if fit.success => fit.parameter_values[index],
                    _ => f64::NAN,
                })
                .collect();
            (p.name.clone(), values)
        })
        .collect()
}

pub fn batch_error_columns(results: &[Option<FitResult>], model: &FitModelSpec) -> Vec<(String, Vec<f64>)> {
    model
        .parameters
        .iter()
        .enumerate()
        .map(|(index, p)| {
            let values = results
                .iter()
                .map(|r| match r {
                    Some(fit) if fit.success && fit.covariance_valid && !fit.fixed_parameter_names.contains(&p.name) => fit.standard_errors[index],
                    _ => f64::NAN,
                })
                .collect();
            (p.name.clone(), values)
        })
        .collect()
}

/// `_FitBatchNumerics.success`.
pub fn batch_success(results: &[Option<FitResult>]) -> Vec<bool> { results.iter().map(|r| r.as_ref().is_some_and(|fit| fit.success)).collect() }

/// `_FitBatchNumerics.parameter_error_validity`: success and a finite error.
pub fn batch_error_validity(results: &[Option<FitResult>], model: &FitModelSpec) -> Vec<(String, Vec<bool>)> {
    let success = batch_success(results);
    batch_error_columns(results, model).into_iter().map(|(name, errors)| (name, errors.iter().zip(&success).map(|(e, &ok)| ok && e.is_finite()).collect())).collect()
}

/// `FitParameterDisplay`: one fitted parameter in painted units.
#[derive(Debug, Clone, PartialEq)]
pub struct FitParameterDisplay {
    pub name: String,
    pub label: String,
    pub value: f64,
    pub standard_error: Option<f64>,
    pub unit: String,
}

impl FitParameterDisplay {
    pub fn new(name: &str, label: &str, value: f64, standard_error: Option<f64>, unit: &str) -> Result<Self, FitError> {
        let name_text = crate::pystr::strip(name);
        if name_text.is_empty() {
            return Err(FitError::value("fit parameter name must be non-empty"));
        }
        let label_text = crate::pystr::strip(label);
        if label_text.is_empty() {
            return Err(FitError::value("fit parameter label must be non-empty"));
        }
        if !value.is_finite() {
            return Err(FitError::value("fit parameter value must be finite"));
        }
        if let Some(error) = standard_error {
            if !error.is_finite() {
                return Err(FitError::value("fit parameter standard_error must be finite"));
            }
            if error < 0.0 {
                return Err(FitError::value("fit parameter standard_error must be non-negative"));
            }
        }
        Ok(FitParameterDisplay { name: name_text.into(), label: label_text.into(), value, standard_error, unit: crate::pystr::strip(unit).into() })
    }
}

/// Text of a refusal that names a parameter.
pub fn repr(text: &str) -> String { str_repr(text) }
