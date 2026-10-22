//! `FitEngine`: one numerical owner for every fit (`fit`, `fit_batch`,
//! `_fit_compiled_batch`, `_nested_population_fit`, `_settle_population`).
//!
//! Cells are solved independently (in parallel); the oracle's buckets are
//! reproduced only where they change an answer: an exception raised while a
//! bucket is prepared or finalised fails every cell of that bucket.

use std::collections::HashMap;
use std::sync::Arc;

use rayon::prelude::*;

use super::models::{loss_code, Axes, Kernel, ObjectiveData};
use super::num::{numpy_sum, pmax, Zr};
use super::prepare::Seeds;
use super::radial::{self, RegularImageFitInput};
use super::result::{Bounds, FitError, FitOptions, FitResult, ObservationArrays, ParamValues, SolveDiagnostics};
use super::shared::{
    authored_seed, binned_curve_statistics, fixed_parameter_partition, histogram_bounds, observation_count, poisson_deviance_residual, sigma_weights, solver_bounds,
    too_few_points,
};
use super::spec::{default_fit_registry, FitModelRegistry, FitModelSpec, Pin, DOMAIN_ANCHORED, REGULAR_IMAGE_RADIAL, REGULAR_IMAGE_SEPARABLE};
use super::trf::{self, CellObjective, FinalizeError, STATUS_INVALID, STATUS_MAX_NFEV, STATUS_NO_CANDIDATE};

/// The input of one cell.
#[derive(Debug, Clone)]
pub enum CellData {
    /// Coordinate arrays (one per independent axis, flattened) and the
    /// observations beside them.
    Points { coordinates: Vec<Arc<[f64]>>, observations: Option<Arc<[f64]>> },
    /// A regular Cartesian image.
    Image(Arc<RegularImageFitInput>),
}

/// One cell of a batch (or the one cell of a single fit).
#[derive(Debug, Clone)]
pub struct FitCell {
    pub data: CellData,
    pub sigma: Option<Arc<[f64]>>,
    pub selected_indices: Option<Arc<[i64]>>,
    pub data_revision: i64,
    /// A previous solution, one value per parameter.
    pub warm_start: Option<Vec<f64>>,
}

impl FitCell {
    pub fn points(coordinates: Vec<Vec<f64>>, observations: Vec<f64>) -> Self {
        FitCell {
            data: CellData::Points { coordinates: coordinates.into_iter().map(Into::into).collect(), observations: Some(observations.into()) },
            sigma: None,
            selected_indices: None,
            data_revision: 0,
            warm_start: None,
        }
    }
    pub fn image(input: RegularImageFitInput) -> Self {
        FitCell { data: CellData::Image(Arc::new(input)), sigma: None, selected_indices: None, data_revision: 0, warm_start: None }
    }
}

/// What the authored request adds to every cell.
#[derive(Clone, Default)]
pub struct FitRequest<'a> {
    pub initial: Option<ParamValues>,
    pub bounds: Option<Bounds>,
    pub options: Option<FitOptions>,
    pub cancelled: Option<&'a (dyn Fn() -> bool + Sync)>,
}

impl FitRequest<'_> {
    fn options(&self) -> FitOptions { self.options.clone().unwrap_or_default() }
}

/// The deadline/cancellation check the oracle's `check()` is.
pub struct Check<'a> {
    started: std::time::Instant,
    deadline: Option<f64>,
    cancelled: Option<&'a (dyn Fn() -> bool + Sync)>,
}

impl<'a> Check<'a> {
    pub fn new(deadline: Option<f64>, cancelled: Option<&'a (dyn Fn() -> bool + Sync)>) -> Self { Check { started: std::time::Instant::now(), deadline, cancelled } }
    pub fn check(&self) -> Result<(), FitError> {
        if let Some(cancelled) = self.cancelled {
            if cancelled() {
                return Err(FitError::Cancelled);
            }
        }
        if let Some(deadline) = self.deadline {
            if self.started.elapsed().as_secs_f64() > deadline {
                return Err(FitError::Deadline);
            }
        }
        Ok(())
    }
}

/// One cell's outcome: a result, or its failure message.
pub type CellOutcome = Result<FitResult, String>;

/// The engine.
pub struct FitEngine {
    pub registry: Arc<FitModelRegistry>,
}

impl Default for FitEngine {
    fn default() -> Self { FitEngine { registry: Arc::new(default_fit_registry()) } }
}

struct Prepared {
    coords: Vec<Vec<f64>>,
    values: Vec<f64>,
    indices: Vec<i64>,
    compiled_coords: Vec<Vec<f64>>,
    solver_values: Vec<f64>,
    weights: Option<Vec<f64>>,
    binned: bool,
    revision: u64,
    origin: f64,
    model: Arc<FitModelSpec>,
    authored: Option<Vec<f64>>,
    warm: Option<Vec<f64>>,
    requested_lower: Vec<f64>,
    requested_upper: Vec<f64>,
    requested_mask: Vec<bool>,
    fixed_names: Vec<String>,
    free_indices: Vec<usize>,
}

/// One cell's compiled solve, before the bucket's verdict.
struct Solved {
    parameters: Vec<f64>,
    status: i32,
    diagnostics: SolveDiagnostics,
    fitted: Vec<f64>,
    residuals: Vec<f64>,
    covariance: Vec<f64>,
    errors: Vec<f64>,
    reduced: f64,
    covariance_valid: bool,
}

struct PointObjective<'a> {
    kernel: Kernel,
    coords: Axes<'a>,
    data: ObjectiveData<'a>,
    selected: usize,
}

impl CellObjective for PointObjective<'_> {
    fn evaluate(&self, params: &[f64], free: &[usize], gradient: &mut [f64], information: &mut [f64], jacobian_row: &mut [f64], derivatives: bool) -> Zr<(f64, f64, bool)> {
        self.kernel.objective(&self.coords, &self.data, params, free, gradient, information, jacobian_row, derivatives)
    }
    fn selected(&self) -> usize { self.selected }
}

fn bucket_failure(error: &FinalizeError) -> String { error.to_string() }

impl FitEngine {
    pub fn new(registry: Arc<FitModelRegistry>) -> Self { FitEngine { registry } }

    /// `_compiled_descriptor(model)`: only this registry's exact built-in.
    fn compiled_kernel(&self, model: &Arc<FitModelSpec>) -> Option<Kernel> {
        let kernel = model.kernel()?;
        if !model.compiled {
            return None;
        }
        let registered = self.registry.get(&model.model_id).ok()?;
        if !Arc::ptr_eq(&registered, model) {
            return None;
        }
        Some(kernel)
    }

    /// `fit(model, coordinates, observations, ...)`.
    pub fn fit(&self, model: &Arc<FitModelSpec>, cell: &FitCell, request: &FitRequest) -> Result<FitResult, FitError> {
        let options = request.options();
        let kernel = self.compiled_kernel(model);
        if let CellData::Image(image) = &cell.data {
            if cell.sigma.is_some() {
                return Err(FitError::type_error("regular-image fitting has no per-point sigma channel"));
            }
            if cell.selected_indices.is_some() {
                return Err(FitError::type_error("regular-image selected_indices belong inside RegularImageFitInput"));
            }
            if !(model.capabilities.contains(REGULAR_IMAGE_RADIAL) || model.capabilities.contains(REGULAR_IMAGE_SEPARABLE)) {
                return Err(FitError::value("this model does not declare a regular-image capability"));
            }
            let outcome = radial::fit_regular_separable_images(model, &[image.clone()], &[cell.data_revision], request, &[cell.warm_start.clone()], &options)?;
            return outcome.into_iter().next().unwrap_or_else(|| Err("regular-image fit failed".into())).map_err(FitError::Value);
        }
        if let CellData::Points { observations: None, .. } = &cell.data {
            return Err(FitError::type_error("observations are required for coordinate-array fitting"));
        }
        match kernel {
            Some(kernel) => {
                let outcomes = self.fit_compiled_batch(model, kernel, std::slice::from_ref(cell), request, &options)?;
                let result = outcomes.into_iter().next().expect("one cell").map_err(|m| FitError::value(if m.is_empty() { "compiled fit failed".to_string() } else { m }))?;
                let nested = self.nested_population_fit(model, std::slice::from_ref(cell), request.bounds.as_ref(), &options, request.cancelled)?;
                self.settle_population(model, result, nested.into_iter().next().flatten().as_ref(), &options)
            }
            None => super::lsq::fit_fallback(self, model, cell, request, &options),
        }
    }

    /// `fit_batch(model, coordinates, observations, ...)`: one outcome per cell.
    pub fn fit_batch(&self, model: &Arc<FitModelSpec>, cells: &[FitCell], request: &FitRequest) -> Result<Vec<CellOutcome>, FitError> {
        if cells.is_empty() {
            return Err(FitError::value("fit batch cannot be empty"));
        }
        let options = request.options();
        let kernel = self.compiled_kernel(model);
        let Some(kernel) = kernel else {
            let mut outcomes = Vec::with_capacity(cells.len());
            for cell in cells {
                match self.fit(model, cell, request) {
                    Ok(result) => outcomes.push(Ok(result)),
                    Err(FitError::Cancelled) => return Err(FitError::Cancelled),
                    Err(error) => outcomes.push(Err(error.failure_text())),
                }
            }
            return Ok(outcomes);
        };
        let image_capable = model.capabilities.contains(REGULAR_IMAGE_RADIAL) || model.capabilities.contains(REGULAR_IMAGE_SEPARABLE);
        if image_capable && cells.iter().all(|c| matches!(c.data, CellData::Image(_))) {
            if cells.iter().any(|c| c.sigma.is_some()) {
                return Err(FitError::type_error("regular-image fitting has no per-point sigma channel"));
            }
            if cells.iter().any(|c| c.selected_indices.is_some()) {
                return Err(FitError::type_error("regular-image selected_indices belong inside RegularImageFitInput"));
            }
            let images: Vec<Arc<RegularImageFitInput>> = cells.iter().map(|c| if let CellData::Image(i) = &c.data { i.clone() } else { unreachable!() }).collect();
            let revisions: Vec<i64> = cells.iter().map(|c| c.data_revision).collect();
            let warm: Vec<Option<Vec<f64>>> = cells.iter().map(|c| c.warm_start.clone()).collect();
            return radial::fit_regular_separable_images(model, &images, &revisions, request, &warm, &options);
        }
        let mut outcomes: Vec<Option<CellOutcome>> = vec![None; cells.len()];
        let mut compiled_cells = Vec::new();
        for (index, cell) in cells.iter().enumerate() {
            match &cell.data {
                CellData::Points { .. } => compiled_cells.push(index),
                CellData::Image(image) => {
                    let outcome = if cell.sigma.is_some() {
                        Err("regular-image fitting has no per-point sigma channel".to_string())
                    } else if cell.selected_indices.is_some() {
                        Err("regular-image selected_indices belong inside RegularImageFitInput".to_string())
                    } else if !image_capable {
                        Err("this model does not declare a regular-image capability".to_string())
                    } else {
                        match radial::fit_regular_separable_images(model, &[image.clone()], &[cell.data_revision], request, &[cell.warm_start.clone()], &options) {
                            Ok(mut v) => v.remove(0),
                            Err(FitError::Cancelled) => return Err(FitError::Cancelled),
                            Err(error) => Err(error.failure_text()),
                        }
                    };
                    outcomes[index] = Some(outcome);
                }
            }
        }
        if compiled_cells.is_empty() {
            return Ok(outcomes.into_iter().map(|o| o.expect("outcome")).collect());
        }
        let compiled: Vec<FitCell> = compiled_cells.iter().map(|&i| cells[i].clone()).collect();
        let solved = self.fit_compiled_batch(model, kernel, &compiled, request, &options)?;
        let mut nested_by_local: HashMap<usize, Option<FitResult>> = HashMap::new();
        if model.reduction.is_some() {
            let wanted: Vec<usize> = (0..compiled.len()).filter(|&l| solved[l].is_ok()).collect();
            if !wanted.is_empty() {
                let wanted_cells: Vec<FitCell> = wanted.iter().map(|&l| FitCell { warm_start: None, ..compiled[l].clone() }).collect();
                let nested = self.nested_population_fit(model, &wanted_cells, request.bounds.as_ref(), &options, request.cancelled)?;
                for (index, &local) in wanted.iter().enumerate() {
                    nested_by_local.insert(local, nested[index].clone());
                }
            }
        }
        for (local, outcome) in solved.into_iter().enumerate() {
            let cell = compiled_cells[local];
            let outcome = match outcome {
                Ok(result) if model.reduction.is_some() => {
                    let nested = nested_by_local.get(&local).cloned().flatten();
                    match self.settle_population(model, result, nested.as_ref(), &options) {
                        Ok(settled) => Ok(settled),
                        Err(FitError::Cancelled) => return Err(FitError::Cancelled),
                        Err(error) => Err(error.failure_text()),
                    }
                }
                other => other,
            };
            outcomes[cell] = Some(outcome);
        }
        Ok(outcomes.into_iter().map(|o| o.expect("outcome")).collect())
    }

    /// `_nested_population_fit`: the one-population answers, in one batch.
    pub(crate) fn nested_population_fit(
        &self,
        spec: &Arc<FitModelSpec>,
        cells: &[FitCell],
        bounds: Option<&Bounds>,
        options: &FitOptions,
        cancelled: Option<&(dyn Fn() -> bool + Sync)>,
    ) -> Result<Vec<Option<FitResult>>, FitError> {
        let none = vec![None; cells.len()];
        let Some(reduction) = &spec.reduction else { return Ok(none) };
        if options.min_bic_gain.is_none() {
            return Ok(none);
        }
        let (fixed, _) = fixed_parameter_partition(spec, bounds);
        if reduction.pinned.iter().any(|(name, _)| fixed.contains(name)) {
            return Ok(none);
        }
        let nested_spec = self.registry.get(&reduction.nested_model_id)?;
        let nested_bounds = bounds.map(|b| Bounds(b.0.iter().filter_map(|(name, pair)| reduction.narrow_for(name).map(|narrow| (narrow.to_string(), *pair))).collect()));
        let nested_bounds = nested_bounds.filter(|b| !b.is_empty());
        let request = FitRequest { initial: None, bounds: nested_bounds, options: Some(options.clone()), cancelled };
        let cells: Vec<FitCell> = cells.iter().map(|c| FitCell { warm_start: None, ..c.clone() }).collect();
        let solved = self.fit_batch(&nested_spec, &cells, &request)?;
        Ok(solved.into_iter().map(Result::ok).collect())
    }

    /// `_settle_population(spec, result, nested, options)`.
    pub fn settle_population(&self, spec: &Arc<FitModelSpec>, result: FitResult, nested: Option<&FitResult>, options: &FitOptions) -> Result<FitResult, FitError> {
        let (Some(reduction), Some(threshold)) = (&spec.reduction, options.min_bic_gain) else { return Ok(result) };
        if !result.success {
            return Ok(result);
        }
        let Some(nested) = nested.filter(|n| n.success) else { return Ok(result) };
        let nested_spec = self.registry.get(&reduction.nested_model_id)?;
        let wide_fitted = result.fitted_values();
        let wide_residuals = result.residuals();
        let wide_observed: Vec<f64> = wide_fitted.iter().zip(wide_residuals.iter()).map(|(f, r)| f + r).collect();
        let narrow_fitted = nested.fitted_values();
        let narrow_residuals = nested.residuals();
        let narrow_observed: Vec<f64> = narrow_fitted.iter().zip(narrow_residuals.iter()).map(|(f, r)| f + r).collect();
        let wide_root = poisson_deviance_residual(&wide_fitted, &wide_observed);
        let narrow_root = poisson_deviance_residual(&narrow_fitted, &narrow_observed);
        let deviance_wide = super::blas::ddot(&wide_root, &wide_root);
        let deviance_narrow = super::blas::ddot(&narrow_root, &narrow_root);
        let clipped: Vec<f64> = wide_observed.iter().map(|&v| if v < 0.0 { 0.0 } else { v }).collect();
        let shots = numpy_sum(&clipped);
        let extra = (spec.parameters.len() as i64 - result.fixed_parameter_names.len() as i64) - (nested_spec.parameters.len() as i64 - nested.fixed_parameter_names.len() as i64);
        let evidence = (deviance_narrow - deviance_wide) - extra as f64 * pmax(shots, 1.0).ln();
        if !evidence.is_finite() {
            return Ok(result);
        }
        if super::decisive(evidence, threshold) {
            return Ok(result.with_evidence(evidence));
        }
        let count = spec.parameters.len();
        let mut values = vec![0.0; count];
        for (index, parameter) in spec.parameters.iter().enumerate() {
            values[index] = match reduction.narrow_for(&parameter.name) {
                Some(narrow) => nested.parameter(narrow).unwrap_or(f64::NAN),
                None => match reduction.pin(&parameter.name) {
                    Some(Pin::Shared(source)) => nested.parameter(reduction.narrow_for(source).unwrap_or(source)).unwrap_or(f64::NAN),
                    Some(Pin::Value(v)) => *v,
                    None => f64::NAN,
                },
            };
        }
        let pinned: Vec<&str> = reduction.pinned.iter().map(|(n, _)| n.as_str()).collect();
        let fixed_names: Vec<String> = spec
            .parameter_names()
            .into_iter()
            .filter(|name| result.fixed_parameter_names.iter().any(|f| f == name) || pinned.contains(name))
            .map(String::from)
            .collect();
        let mut covariance = vec![0.0; count * count];
        let mut errors = vec![0.0; count];
        if nested.covariance_valid {
            let narrow_count = nested_spec.parameters.len();
            let wide: Vec<usize> = reduction.shared.iter().map(|(w, _)| spec.parameter_index(w).expect("shared")).collect();
            let narrow: Vec<usize> = reduction.shared.iter().map(|(_, n)| nested_spec.parameter_index(n).expect("shared")).collect();
            for (a, &wi) in wide.iter().enumerate() {
                for (b, &wj) in wide.iter().enumerate() {
                    covariance[wi * count + wj] = nested.covariance[narrow[a] * narrow_count + narrow[b]];
                }
            }
            for i in 0..count {
                errors[i] = pmax_np(covariance[i * count + i], 0.0).sqrt();
            }
        } else {
            covariance.fill(f64::NAN);
            errors.fill(f64::NAN);
            for name in &fixed_names {
                let f = spec.parameter_index(name)?;
                for k in 0..count {
                    covariance[f * count + k] = 0.0;
                    covariance[k * count + f] = 0.0;
                }
                errors[f] = 0.0;
            }
        }
        Ok(FitResult {
            model: spec.clone(),
            parameter_values: values,
            standard_errors: errors,
            covariance,
            arrays: nested.arrays.clone(),
            source_revision: result.source_revision,
            success: true,
            message: format!("one population: BIC gain {} does not exceed {}", super::format_fixed(evidence, 1), crate::pyfmt::format_g(threshold, 6)),
            reduced_chi_square: nested.reduced_chi_square,
            covariance_valid: nested.covariance_valid,
            parameter_units: result.parameter_units.clone(),
            batch_revision: 0,
            fixed_parameter_names: fixed_names,
            reduced: true,
            evidence,
            diagnostics: nested.diagnostics,
        })
    }

    /// `_fit_compiled_batch`.
    fn fit_compiled_batch(&self, model: &Arc<FitModelSpec>, kernel: Kernel, cells: &[FitCell], request: &FitRequest, opts: &FitOptions) -> Result<Vec<CellOutcome>, FitError> {
        let check = Check::new(opts.deadline_seconds, request.cancelled);
        check.check()?;
        let parameter_count = model.parameters.len();
        let base_lower: Vec<f64> = model.parameters.iter().map(|p| p.bounds().0).collect();
        let base_upper: Vec<f64> = model.parameters.iter().map(|p| p.bounds().1).collect();
        let counted = model.counted();
        let anchored_origin = model.capabilities.contains(DOMAIN_ANCHORED);
        let authored_seed_row = match &request.initial {
            Some(initial) => Some(authored_seed(model, initial)?),
            None => None,
        };
        let mut outcomes: Vec<Option<CellOutcome>> = vec![None; cells.len()];
        let mut prepared: Vec<Option<Prepared>> = (0..cells.len()).map(|_| None).collect();
        let mut histogram_cache: HashMap<Vec<u64>, Option<Bounds>> = HashMap::new();
        let mut anchored_cache: Vec<(f64, Arc<FitModelSpec>)> = Vec::new();
        for (index, cell) in cells.iter().enumerate() {
            check.check()?;
            let attempt = (|| -> Result<Prepared, FitError> {
                let CellData::Points { coordinates, observations } = &cell.data else {
                    return Err(FitError::type_error("regular-image inputs belong to the separable fit route"));
                };
                let Some(observations) = observations else {
                    return Err(FitError::type_error("observations are required for coordinate fitting"));
                };
                if coordinates.len() != model.independent_arity {
                    return Err(FitError::value("coordinate arity does not match fit model"));
                }
                if coordinates.iter().any(|a| a.len() != coordinates[0].len()) {
                    return Err(FitError::value("coordinate arrays must have equal shape"));
                }
                let values: &[f64] = observations;
                if coordinates.iter().any(|a| a.len() != values.len()) {
                    return Err(FitError::value("coordinates and observations must have equal flattened shape"));
                }
                if let Some(sigma) = &cell.sigma {
                    if sigma.len() != values.len() {
                        return Err(FitError::value("observation_sigma must match observations"));
                    }
                    if counted {
                        return Err(FitError::value("histogram targets weight themselves by counts; an external sigma has no meaning there"));
                    }
                }
                if let Some(indices) = &cell.selected_indices {
                    if indices.len() != values.len() {
                        return Err(FitError::value("selected_indices must match observations"));
                    }
                }
                let finite: Vec<bool> = (0..values.len()).map(|i| values[i].is_finite() && coordinates.iter().all(|a| a[i].is_finite())).collect();
                let keep = |source: &[f64]| -> Vec<f64> { source.iter().zip(&finite).filter(|(_, f)| **f).map(|(&v, _)| v).collect() };
                let coords: Vec<Vec<f64>> = coordinates.iter().map(|a| keep(a)).collect();
                let values_kept = keep(values);
                let indices: Vec<i64> = match &cell.selected_indices {
                    Some(indices) => indices.iter().zip(&finite).filter(|(_, f)| **f).map(|(&v, _)| v).collect(),
                    None => (0..values.len() as i64).zip(&finite).filter(|(_, f)| **f).map(|(v, _)| v).collect(),
                };
                let sigma_kept = cell.sigma.as_ref().map(|s| keep(s));
                let cell_bounds = if counted {
                    let key: Vec<u64> = coords[0].iter().map(|v| v.to_bits()).collect();
                    match histogram_cache.get(&key) {
                        Some(b) => b.clone(),
                        None => {
                            let b = histogram_bounds(model, &coords[0], request.bounds.as_ref())?;
                            histogram_cache.insert(key, b.clone());
                            b
                        }
                    }
                } else {
                    histogram_bounds(model, &coords[0], request.bounds.as_ref())?
                };
                let (requested_lower, requested_upper) = solver_bounds(model, None, cell_bounds.as_ref())?;
                let requested_mask: Vec<bool> = model.parameters.iter().map(|p| cell_bounds.as_ref().is_some_and(|b| b.contains(&p.name))).collect();
                let (fixed_names, free_indices) = fixed_parameter_partition(model, cell_bounds.as_ref());
                let points = observation_count(&values_kept, counted);
                if points <= free_indices.len() as i64 {
                    return Err(too_few_points(model, free_indices.len(), points));
                }
                if cell.data_revision < 0 {
                    return Err(FitError::value("data_revision must be non-negative"));
                }
                let mut origin = 0.0;
                let mut effective = model.clone();
                if anchored_origin {
                    let (low, _) = super::shared::min_max(&coords[0]);
                    origin = low;
                    effective = match anchored_cache.iter().find(|(o, _)| *o == origin) {
                        Some((_, m)) => m.clone(),
                        None => {
                            let m = model.anchored_at(origin)?;
                            anchored_cache.push((origin, m.clone()));
                            m
                        }
                    };
                }
                let mut solver_coords = coords.clone();
                let mut solver_values = values_kept.clone();
                let mut weights = None;
                let mut binned = false;
                if let Some(sigma) = &sigma_kept {
                    weights = sigma_weights(sigma);
                } else if !counted && !free_indices.is_empty() && model.independent_arity == 1 && opts.loss == "linear" {
                    if let Some(max_exact) = opts.max_exact_points {
                        if values_kept.len() as i64 > 2 * max_exact {
                            if let Some((x, y, w)) = binned_curve_statistics(&coords[0], &values_kept, max_exact) {
                                solver_coords = vec![x];
                                solver_values = y;
                                weights = Some(w);
                                binned = true;
                            }
                        }
                    }
                }
                let mut compiled_coords = solver_coords;
                if anchored_origin {
                    for v in compiled_coords[0].iter_mut() {
                        *v -= origin;
                    }
                }
                Ok(Prepared {
                    coords,
                    values: values_kept,
                    indices,
                    compiled_coords,
                    solver_values,
                    weights,
                    binned,
                    revision: cell.data_revision as u64,
                    origin,
                    model: effective,
                    authored: None,
                    warm: None,
                    requested_lower,
                    requested_upper,
                    requested_mask,
                    fixed_names,
                    free_indices,
                })
            })();
            match attempt {
                Ok(mut item) => {
                    if !item.free_indices.is_empty() {
                        item.authored = authored_seed_row.clone();
                        if let Some(warm) = &cell.warm_start {
                            item.warm = Some(super::shared::warm_seed(&item.model, warm)?);
                        }
                    }
                    prepared[index] = Some(item);
                }
                Err(FitError::Cancelled) => return Err(FitError::Cancelled),
                Err(error) => outcomes[index] = Some(Err(error.failure_text())),
            }
        }

        // Cells with every parameter fixed: one evaluation.
        for (index, slot) in prepared.iter_mut().enumerate() {
            let Some(item) = slot else { continue };
            if !item.free_indices.is_empty() {
                continue;
            }
            check.check()?;
            let item = slot.take().expect("prepared");
            let fixed_values = item.requested_lower.clone();
            let outcome = (|| -> Result<FitResult, FitError> {
                let fitted = evaluate_model(&item.model, &item.coords, &fixed_values)?;
                if fitted.len() != item.values.len() || fitted.iter().any(|v| !v.is_finite()) {
                    return Err(FitError::runtime("fixed fit evaluation is non-finite"));
                }
                let residuals: Vec<f64> = item.values.iter().zip(&fitted).map(|(v, f)| v - f).collect();
                let quality: Vec<f64> = if counted {
                    poisson_deviance_residual(&fitted, &item.values)
                } else {
                    match &item.weights {
                        None => residuals.clone(),
                        Some(w) => residuals.iter().zip(w).map(|(r, w)| r * w).collect(),
                    }
                };
                let reduced = super::blas::ddot(&quality, &quality) / item.values.len() as f64;
                let count = parameter_count;
                Ok(FitResult {
                    model: item.model.clone(),
                    parameter_values: fixed_values.clone(),
                    standard_errors: vec![0.0; count],
                    covariance: vec![0.0; count * count],
                    arrays: FitResult::eager(fitted, residuals, item.indices.clone()),
                    source_revision: item.revision,
                    success: true,
                    message: "all parameters fixed".into(),
                    reduced_chi_square: reduced,
                    covariance_valid: true,
                    parameter_units: vec![String::new(); count],
                    batch_revision: 0,
                    fixed_parameter_names: item.fixed_names.clone(),
                    reduced: false,
                    evidence: f64::NAN,
                    diagnostics: None,
                })
            })();
            outcomes[index] = Some(match outcome {
                Ok(result) => Ok(result),
                Err(error) => Err(error.failure_text()),
            });
        }

        // Buckets, as the oracle groups them.
        let active: Vec<usize> = (0..cells.len()).filter(|&i| prepared[i].is_some()).collect();
        let buckets = form_buckets(&prepared, &active, counted, opts);
        let loss = loss_code(&opts.loss).ok_or_else(|| FitError::value(format!("unsupported compiled fit loss: {}", super::num::str_repr(&opts.loss))))?;
        // Bucket after bucket, checked before each solve and after each
        // that succeeds (a failed bucket skips that check); a bucket's cells
        // solve in parallel.
        for bucket in &buckets {
            check.check()?;
            match solve_bucket(kernel, &prepared, bucket, &base_lower, &base_upper, counted, loss, parameter_count) {
                Ok(list) => {
                    check.check()?;
                    for (cell, outcome) in list {
                        outcomes[cell] = Some(outcome);
                    }
                }
                Err(message) => {
                    for &cell in &bucket.cells {
                        outcomes[cell] = Some(Err(message.clone()));
                    }
                }
            }
        }
        Ok(outcomes.into_iter().map(|o| o.unwrap_or_else(|| Err("fit failed".into()))).collect())
    }
}

#[inline]
fn pmax_np(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else if a >= b {
        a
    } else {
        b
    }
}

/// `model.evaluate(coordinates, values)` (anchored axis included).
pub fn evaluate_model(model: &FitModelSpec, coords: &[Vec<f64>], params: &[f64]) -> Result<Vec<f64>, FitError> {
    match &model.evaluator {
        super::spec::Evaluator::Builtin(kernel) => {
            let first = model.anchored_axis(&coords[0]);
            let axes = if coords.len() == 1 { Axes::one(&first) } else { Axes::two(&first, &coords[1]) };
            Ok(kernel.value_jacobian(&axes, params, false)?.0)
        }
        super::spec::Evaluator::Custom(custom) => {
            let first = model.anchored_axis(&coords[0]);
            let mut axes: Vec<&[f64]> = vec![&first];
            for axis in &coords[1..] {
                axes.push(axis);
            }
            Ok(custom.evaluate(&axes, params))
        }
    }
}

/// The bucket keys of `_fit_compiled_batch`, with masked-grid merges.
enum Layout {
    Shared,
    Masked { grid: Vec<f64>, positions: Vec<Vec<usize>>, budget: i64 },
}

struct Bucket {
    cells: Vec<usize>,
    layout: Layout,
    budget: i64,
}

fn form_buckets(prepared: &[Option<Prepared>], active: &[usize], counted: bool, opts: &FitOptions) -> Vec<Bucket> {
    // key: (size, coordinate content, free_index) [+ group]
    type Key = (usize, Vec<Vec<u64>>, Vec<usize>, Option<(Vec<usize>, u64, i64, bool)>);
    let mut order: Vec<Key> = Vec::new();
    let mut members: HashMap<Key, Vec<usize>> = HashMap::new();
    let mut groups_order: Vec<(Vec<usize>, u64, i64, bool)> = Vec::new();
    let mut groups: HashMap<(Vec<usize>, u64, i64, bool), Vec<usize>> = HashMap::new();
    let mut cell_key: HashMap<usize, Key> = HashMap::new();
    for &cell in active {
        let item = prepared[cell].as_ref().expect("active");
        let content: Vec<Vec<u64>> = item.compiled_coords.iter().map(|a| a.iter().map(|v| v.to_bits()).collect()).collect();
        let mut key: Key = (item.solver_values.len(), content, item.free_indices.clone(), None);
        let axes = &item.compiled_coords;
        if !counted && !item.binned && axes.len() == 1 && axes[0].windows(2).all(|w| w[1] - w[0] > 0.0) {
            let group = (item.free_indices.clone(), item.origin.to_bits(), opts.evaluation_budget(item.solver_values.len()), item.weights.is_some());
            key.3 = Some(group.clone());
            if !groups.contains_key(&group) {
                groups_order.push(group.clone());
            }
            groups.entry(group).or_default().push(cell);
        }
        if !members.contains_key(&key) {
            order.push(key.clone());
        }
        members.entry(key.clone()).or_default().push(cell);
        cell_key.insert(cell, key);
    }
    let mut buckets: Vec<(Option<Key>, Bucket)> = order
        .iter()
        .map(|k| {
            let first = prepared[members[k][0]].as_ref().expect("member");
            (Some(k.clone()), Bucket { cells: members[k].clone(), layout: Layout::Shared, budget: opts.evaluation_budget(first.solver_values.len()) })
        })
        .collect();
    for group in groups_order {
        let cells = &groups[&group];
        let keys: Vec<&Key> = {
            let mut seen: Vec<&Key> = Vec::new();
            for c in cells {
                let k = &cell_key[c];
                if !seen.contains(&k) {
                    seen.push(k);
                }
            }
            seen
        };
        if keys.len() == 1 {
            continue;
        }
        let axes: Vec<&Vec<f64>> = cells.iter().map(|&c| &prepared[c].as_ref().expect("cell").compiled_coords[0]).collect();
        let grid = super::shared::unique_sorted(&axes.iter().flat_map(|a| a.iter().copied()).collect::<Vec<f64>>());
        if cells.len() * grid.len() > 2 * axes.iter().map(|a| a.len()).sum::<usize>() {
            continue;
        }
        let remove: Vec<Key> = keys.into_iter().cloned().collect();
        buckets.retain(|(k, _)| k.as_ref().is_none_or(|k| !remove.contains(k)));
        let positions: Vec<Vec<usize>> = axes.iter().map(|a| a.iter().map(|v| grid.partition_point(|g| g < v)).collect()).collect();
        buckets.push((None, Bucket { cells: cells.clone(), layout: Layout::Masked { grid, positions, budget: group.2 }, budget: group.2 }));
    }
    buckets.into_iter().map(|(_, b)| b).collect()
}

#[allow(clippy::too_many_arguments)]
fn solve_bucket(kernel: Kernel, prepared: &[Option<Prepared>], bucket: &Bucket, base_lower: &[f64], base_upper: &[f64], counted: bool, loss: i64, parameter_count: usize) -> Result<Vec<(usize, CellOutcome)>, String> {
    let use_weights = bucket.cells.iter().any(|&c| prepared[c].as_ref().expect("cell").weights.is_some());
    let max_nfev = match &bucket.layout {
        Layout::Shared => bucket.budget,
        Layout::Masked { budget, .. } => *budget,
    };
    let solved: Vec<(usize, Result<Solved, String>)> = bucket
        .cells
        .par_iter()
        .enumerate()
        .map(|(local, &cell)| {
            let item = prepared[cell].as_ref().expect("cell");
            (cell, solve_one(kernel, item, &bucket.layout, local, base_lower, base_upper, counted, loss, use_weights, max_nfev))
        })
        .collect();
    // A prepare- or finalize-time exception fails the whole bucket.
    if let Some((_, Err(message))) = solved.iter().find(|(_, s)| s.is_err()) {
        return Err(message.clone());
    }
    let mut outcomes = Vec::with_capacity(bucket.cells.len());
    for (cell, solved) in solved {
        let item = prepared[cell].as_ref().expect("cell");
        let solved = solved.expect("checked");
        outcomes.push((cell, finish_cell(item, solved, parameter_count)));
    }
    Ok(outcomes)
}

#[allow(clippy::too_many_arguments)]
fn solve_one(kernel: Kernel, item: &Prepared, layout: &Layout, local: usize, base_lower: &[f64], base_upper: &[f64], counted: bool, loss: i64, use_weights: bool, max_nfev: i64) -> Result<Solved, String> {
    // The coordinates and observations the kernel sees: the cell's own, or
    // the union grid with this cell's points marked valid.
    let (grid_coords, grid_values, grid_valid, grid_weights): (Vec<Vec<f64>>, Vec<f64>, Vec<bool>, Vec<f64>) = match layout {
        Layout::Shared => (
            item.compiled_coords.clone(),
            item.solver_values.clone(),
            vec![true; item.solver_values.len()],
            item.weights.clone().unwrap_or_else(|| vec![1.0; item.solver_values.len()]),
        ),
        Layout::Masked { grid, positions, .. } => {
            let mut values = vec![0.0; grid.len()];
            let mut valid = vec![false; grid.len()];
            let mut weights = vec![0.0; grid.len()];
            for (k, &p) in positions[local].iter().enumerate() {
                values[p] = item.solver_values[k];
                valid[p] = true;
                weights[p] = item.weights.as_ref().map_or(1.0, |w| w[k]);
            }
            if item.weights.is_none() && !use_weights {
                weights.clear();
            }
            (vec![grid.clone()], values, valid, weights)
        }
    };
    let axes = if grid_coords.len() == 1 { Axes::one(&grid_coords[0]) } else { Axes::two(&grid_coords[0], &grid_coords[1]) };
    let free_mask: Vec<bool> = (0..base_lower.len()).map(|i| item.free_indices.contains(&i)).collect();
    let max_candidates = kernel.max_candidates();
    let seed_capacity = 1 + max_candidates.max(1);
    let authored_rows = item.authored.as_ref().map(|row| vec![row.clone()]);
    let mut prepare = |block: &mut [f64], rows: usize, lower: &mut [f64], upper: &mut [f64]| -> Zr<i64> {
        let mut seeds = Seeds { data: block, rows, cols: base_lower.len() };
        kernel.prepare(&axes, &grid_values, &grid_valid, &mut seeds, lower, upper)
    };
    let prepared = trf::prepare_one(
        &mut prepare,
        base_lower,
        base_upper,
        &item.requested_lower,
        &item.requested_upper,
        &item.requested_mask,
        item.warm.as_deref(),
        authored_rows.as_deref(),
        &free_mask,
        max_candidates,
        seed_capacity,
    )
    .map_err(|e| e.to_string())?;
    let weights_slice: &[f64] = if use_weights { &grid_weights } else { &[] };
    let objective = PointObjective {
        kernel,
        coords: axes,
        data: ObjectiveData { observations: &grid_values, valid: &grid_valid, weights: weights_slice, use_weights, poisson: counted, loss_code: loss },
        selected: grid_valid.iter().filter(|&&v| v).count(),
    };
    let solution = trf::solve_cell(&objective, &prepared.seeds, &prepared.lower, &prepared.upper, &item.free_indices, max_nfev, 1.0e-8, 1.0e-8, 1.0e-8, item.warm.is_some(), prepared.status);
    // Finalize on the cell's selected points.
    let selected_coords: Vec<Vec<f64>> = grid_coords.iter().map(|axis| axis.iter().zip(&grid_valid).filter(|(_, v)| **v).map(|(&x, _)| x).collect()).collect();
    let selected_axes = if selected_coords.len() == 1 { Axes::one(&selected_coords[0]) } else { Axes::two(&selected_coords[0], &selected_coords[1]) };
    let value_jacobian = |with_jacobian: bool| kernel.value_jacobian(&selected_axes, &solution.parameters, with_jacobian);
    let finalized = trf::finalize_one(&value_jacobian, &grid_values, &grid_valid, &solution.parameters, &item.free_indices, weights_slice, use_weights, counted, loss).map_err(|e| bucket_failure(&e))?;
    let covariance_valid = finalized.covariance_valid && solution.status >= STATUS_MAX_NFEV;
    // Back from the grid to the cell's own points.
    let (fitted, residuals) = match layout {
        Layout::Shared => (finalized.fitted, finalized.residuals),
        Layout::Masked { positions, .. } => (positions[local].iter().map(|&p| finalized.fitted[p]).collect(), positions[local].iter().map(|&p| finalized.residuals[p]).collect()),
    };
    Ok(Solved {
        diagnostics: SolveDiagnostics {
            status: solution.status,
            nfev: solution.nfev,
            njev: solution.njev,
            iterations: solution.iterations,
            winner_seed: solution.winner_seed,
            cost: solution.cost,
            raw_rss: solution.raw_rss,
        },
        parameters: solution.parameters,
        status: solution.status,
        fitted,
        residuals,
        covariance: finalized.covariance,
        errors: finalized.errors,
        reduced: finalized.reduced,
        covariance_valid,
    })
}

fn isclose(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    (a - b).abs() <= atol + rtol * b.abs()
}

fn finish_cell(item: &Prepared, mut solved: Solved, parameter_count: usize) -> CellOutcome {
    let status = solved.status;
    if status == STATUS_INVALID || status == STATUS_NO_CANDIDATE {
        return Err(trf::termination_message(status));
    }
    let free_count = item.free_indices.len();
    let mut covariance_valid = solved.covariance_valid;
    let mut final_error: Option<&str> = None;
    let mut final_arrays: Option<(Vec<f64>, Vec<f64>)> = None;
    if item.binned {
        let fitted = match evaluate_model(&item.model, &item.coords, &solved.parameters) {
            Ok(f) => f,
            Err(error) => return Err(error.failure_text()),
        };
        let residuals: Vec<f64> = item.values.iter().zip(&fitted).map(|(v, f)| v - f).collect();
        let reduced = super::blas::ddot(&residuals, &residuals) / (item.values.len() as i64 - free_count as i64).max(1) as f64;
        let compiled_reduced = solved.reduced;
        solved.reduced = reduced;
        if fitted.len() != item.values.len() || fitted.iter().any(|v| !v.is_finite()) || residuals.iter().any(|v| !v.is_finite()) {
            final_error = Some("compiled fit returned invalid full-data arrays");
        }
        if covariance_valid {
            if compiled_reduced > 0.0 {
                let ratio = reduced / compiled_reduced;
                for v in solved.covariance.iter_mut() {
                    *v *= ratio;
                }
                let root = ratio.sqrt();
                for v in solved.errors.iter_mut() {
                    *v *= root;
                }
            } else if reduced > 0.0 {
                covariance_valid = false;
            }
        }
        final_arrays = Some((fitted, residuals));
    }
    if !covariance_valid {
        solved.covariance.fill(f64::NAN);
        solved.errors.fill(f64::NAN);
        for f in (0..parameter_count).filter(|i| !item.free_indices.contains(i)) {
            for k in 0..parameter_count {
                solved.covariance[f * parameter_count + k] = 0.0;
                solved.covariance[k * parameter_count + f] = 0.0;
            }
            solved.errors[f] = 0.0;
        }
    }
    // The batch's validation of the compiled output arrays (for a binned
    // cell, the compiled iteration's own arrays, not the full data's).
    let invalid = solved.parameters.iter().any(|v| !v.is_finite())
        || solved.fitted.iter().any(|v| !v.is_finite())
        || solved.residuals.iter().any(|v| !v.is_finite())
        || !solved.reduced.is_finite()
        || solved.reduced < 0.0;
    if invalid {
        final_error = Some("compiled fit returned invalid arrays");
    }
    if covariance_valid {
        let n = parameter_count;
        let c = &solved.covariance;
        let e = &solved.errors;
        let bad = c.iter().any(|v| !v.is_finite())
            || e.iter().any(|v| !v.is_finite())
            || (0..n).any(|i| (0..n).any(|j| !isclose(c[i * n + j], c[j * n + i], 1e-12, 1e-15)))
            || (0..n).any(|i| c[i * n + i] < 0.0)
            || e.iter().any(|&v| v < 0.0)
            || (0..n).any(|i| !isclose(e[i] * e[i], c[i * n + i], 1e-10, 1e-15));
        if bad {
            final_error = Some("compiled fit returned invalid covariance");
        }
    }
    if let Some(error) = final_error {
        return Err(error.to_string());
    }
    let (fitted, residuals) = final_arrays.unwrap_or((solved.fitted, solved.residuals));
    if item.indices.len() != fitted.len() {
        return Err("selected indices must identify every fitted observation".into());
    }
    Ok(FitResult {
        model: item.model.clone(),
        parameter_values: solved.parameters,
        standard_errors: solved.errors,
        covariance: solved.covariance,
        arrays: ObservationArrays::Eager { fitted: fitted.into(), residuals: residuals.into(), indices: item.indices.clone().into() },
        source_revision: item.revision,
        success: status > STATUS_MAX_NFEV,
        message: trf::termination_message(status),
        reduced_chi_square: solved.reduced,
        covariance_valid,
        parameter_units: vec![String::new(); parameter_count],
        batch_revision: 0,
        fixed_parameter_names: item.fixed_names.clone(),
        reduced: false,
        evidence: f64::NAN,
        diagnostics: Some(solved.diagnostics),
    })
}
