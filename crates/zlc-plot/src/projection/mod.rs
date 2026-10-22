//! Native data projection: one immutable dataset revision (or history
//! window) to what a plot draws. A port of `zlc_plot.data_view`,
//! `zlc_plot.preparation`/`data_contract`, the `zlc_data` projection helpers
//! and `zlc_plot._fit_projection`; see `AGREEMENT.md` for the exactness
//! contract the golden tests hold it to.

pub mod nd;
pub mod kernels;
pub mod model;
pub mod history;
pub mod output;
pub mod view;
pub mod reduce;
pub mod strided;
pub mod lattice;
pub mod curve;
pub mod rolling;
pub mod histogram;
pub mod scope;
pub mod plan;
pub mod fit_selection;
pub mod golden;
pub mod samples;

use std::fmt;

pub use model::{SchemaCache, Snapshot, Source, SourceData};
pub use output::*;
pub use plan::{HistogramProjection, Projection, ProjectionPlan};
pub use view::DataView;

/// The Python exception class a refusal corresponds to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ErrorKind { DataView, AxisResolution, Value, Type, Index, EmptySelection, Key, NotImplemented, Runtime }

impl ErrorKind {
    pub fn python_name(self) -> &'static str {
        match self {
            ErrorKind::DataView => "DataViewError", ErrorKind::AxisResolution => "AxisResolutionError",
            ErrorKind::Value => "ValueError", ErrorKind::Type => "TypeError", ErrorKind::Index => "IndexError",
            ErrorKind::EmptySelection => "EmptySelection", ErrorKind::Key => "KeyError", ErrorKind::NotImplemented => "NotImplementedError",
            ErrorKind::Runtime => "RuntimeError",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProjectionError { pub kind: ErrorKind, pub message: String }

impl ProjectionError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self { ProjectionError { kind, message: message.into() } }
    pub fn view(message: impl Into<String>) -> Self { Self::new(ErrorKind::DataView, message) }
    pub fn value(message: impl Into<String>) -> Self { Self::new(ErrorKind::Value, message) }
    pub fn type_error(message: impl Into<String>) -> Self { Self::new(ErrorKind::Type, message) }
}

impl fmt::Display for ProjectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.kind.python_name(), self.message) }
}

impl std::error::Error for ProjectionError {}

impl From<crate::units::UnitError> for ProjectionError {
    fn from(error: crate::units::UnitError) -> Self { ProjectionError::new(ErrorKind::Value, error.0) }
}

pub type Result<T> = std::result::Result<T, ProjectionError>;
