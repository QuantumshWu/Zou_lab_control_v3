//! The fit engine: model catalogue, the compiled lane's trust-region solver
//! and its LAPACK, regular-image fits, units of fitted parameters and the
//! decoding of authored fit targets -- a port of `zlc_plot.fit` and its
//! private modules held to `TOLERANCES.md`.

pub mod blas;
pub mod classifier;
pub mod engine;
pub mod evaluation;
pub mod linalg;
pub mod lsq;
pub mod models;
pub mod num;
pub mod param_units;
pub mod plane;
pub mod prepare;
pub mod pylane;
pub mod radial;
pub mod result;
pub mod scratch;
pub mod shared;
pub mod spec;
pub mod target;
pub mod trf;

pub use engine::{CellData, CellOutcome, FitCell, FitEngine, FitRequest};
pub use evaluation::{evaluate_batch, evaluate_single, EvaluatedTargets, TargetCell};
pub use radial::{RegularImageFitInput, Samples};
pub use result::{Bound, Bounds, FitError, FitOptions, FitParameterDisplay, FitResult, ParamValues, SolveDiagnostics};
pub use spec::{builtin_fit_models, default_fit_registry, FitModelRegistry, FitModelSpec, FitParameterSpec, FitTarget, ParameterDomain, UnitRelation};

/// `evidence.DECISIVE_BIC_GAIN`: Kass and Raftery's "very strong".
pub const DECISIVE_BIC_GAIN: f64 = 10.0;

/// `evidence.decisive(gain, threshold)`: the gain must EXCEED it; NaN never does.
pub fn decisive(gain: f64, threshold: f64) -> bool { gain > threshold }

/// Python's `format(x, f".{precision}f")`.
pub fn format_fixed(x: f64, precision: usize) -> String {
    if x.is_nan() {
        return "nan".into();
    }
    if x.is_infinite() {
        return if x < 0.0 { "-inf".into() } else { "inf".into() };
    }
    format!("{x:.precision$}")
}
