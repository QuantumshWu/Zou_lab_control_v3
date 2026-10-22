//! `fit_evaluation.evaluate_fit`: one solve over cells a projection prepared.
//!
//! The projection side (`_fit_projection`: targets, selections, sample axes,
//! canonical units) lives elsewhere; it hands each target over as a
//! [`TargetCell`] and keeps the evaluation's container.  What is fixed here is
//! the numerical contract: which failures stay per target, which propagate,
//! and which text a target without a result carries.

use super::engine::{FitCell, FitEngine, FitRequest};
use super::result::{FitError, FitResult};
use super::spec::FitModelSpec;
use std::sync::Arc;

/// One fit target as the projection prepared it.
pub enum TargetCell {
    /// A selection to solve (its warm start inside the cell).
    Ready(FitCell),
    /// The selection failed, with `str(error) or type(error).__name__`
    /// (every target carries the same text when the selector itself failed).
    Failed(String),
    /// Outside the requested `distribution_indices`.
    NotRequested,
}

/// Per-target outcomes, in target order: a result with its canonical units,
/// or the failure text.
#[derive(Clone)]
pub struct EvaluatedTargets {
    pub results: Vec<Option<FitResult>>,
    pub failures: Vec<Option<String>>,
}

/// A single fit: every engine error propagates to the host.
pub fn evaluate_single(engine: &FitEngine, model: &Arc<FitModelSpec>, cell: &FitCell, request: &FitRequest, units: &[(String, String)]) -> Result<FitResult, FitError> {
    let result = engine.fit(model, cell, request)?;
    // An engine-bound coordinate origin may copy the model; its id remains.
    if result.model.model_id != model.model_id {
        return Err(FitError::runtime("FitEngine returned a result for another model"));
    }
    if result.source_revision as i64 != cell.data_revision {
        return Err(FitError::runtime("FitEngine returned a result for another data revision"));
    }
    result.with_parameter_units(units)
}

/// A batch: selection and solver failures stay with their targets; only a
/// cancellation (not a deadline, which fails the solved cells) propagates.
pub fn evaluate_batch(engine: &FitEngine, model: &Arc<FitModelSpec>, targets: Vec<TargetCell>, request: &FitRequest, units: &[(String, String)]) -> Result<EvaluatedTargets, FitError> {
    let count = targets.len();
    let mut results: Vec<Option<FitResult>> = vec![None; count];
    let mut failures: Vec<Option<String>> = vec![None; count];
    let mut cells = Vec::new();
    let mut slots = Vec::new();
    for (index, target) in targets.into_iter().enumerate() {
        match target {
            TargetCell::Ready(cell) => {
                cells.push(cell);
                slots.push(index);
            }
            TargetCell::Failed(message) => failures[index] = Some(message),
            TargetCell::NotRequested => failures[index] = Some("fit not requested".into()),
        }
    }
    if !cells.is_empty() {
        let outcomes = match engine.fit_batch(model, &cells, request) {
            Ok(outcomes) => outcomes,
            Err(FitError::Cancelled) => return Err(FitError::Cancelled),
            Err(error) => vec![Err(error.failure_text()); cells.len()],
        };
        for (outcome, &index) in outcomes.into_iter().zip(&slots) {
            match outcome {
                Ok(result) => results[index] = Some(result.with_parameter_units(units)?),
                Err(message) => failures[index] = Some(message),
            }
        }
    }
    for index in 0..count {
        if request.cancelled.is_some_and(|cancelled| cancelled()) {
            return Err(FitError::Cancelled);
        }
        if results[index].is_none() && failures[index].as_deref().is_none_or(str::is_empty) {
            failures[index] = Some("fit failed".into());
        }
    }
    Ok(EvaluatedTargets { results, failures })
}
