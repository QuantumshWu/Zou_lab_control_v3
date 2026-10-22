//! One error type whose kind is the Python exception class the scientific
//! package raises at the same step, so callers (and the golden tests) can
//! branch exactly where Python's `except` clauses did.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum ErrorKind {
    Type,
    Value,
    Key,
    Index,
    Runtime,
    Overflow,
    /// `zlc_data.units.UnitError`, a `ValueError`.
    Unit,
    /// `semantics.SemanticVacancy`, a `ValueError`: a required role holds no axis.
    Vacancy,
    /// `ZeroDivisionError` (an `ArithmeticError`), raised by the interaction model.
    ZeroDivision,
    /// `FitCancelled` (a `RuntimeError`): a preparation that is no longer current.
    FitCancelled,
    /// Another Python class, by name: a file or figure-archive refusal.
    Other(&'static str),
}

impl ErrorKind {
    /// The Python class name.
    pub fn python_name(self) -> &'static str {
        match self {
            ErrorKind::Type => "TypeError", ErrorKind::Value => "ValueError", ErrorKind::Key => "KeyError",
            ErrorKind::Index => "IndexError", ErrorKind::Runtime => "RuntimeError", ErrorKind::Overflow => "OverflowError",
            ErrorKind::Unit => "UnitError", ErrorKind::Vacancy => "SemanticVacancy",
            ErrorKind::ZeroDivision => "ZeroDivisionError", ErrorKind::FitCancelled => "FitCancelled", ErrorKind::Other(name) => name,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlotError {
    pub kind: ErrorKind,
    /// `str()` of the Python exception: a KeyError's is its argument's repr.
    pub message: String,
    /// The vacant role of a [`ErrorKind::Vacancy`].
    pub role: Option<String>,
}

pub type Result<T> = std::result::Result<T, PlotError>;

impl PlotError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self { Self { kind, message: message.into(), role: None } }
    pub fn type_error(message: impl Into<String>) -> Self { Self::new(ErrorKind::Type, message) }
    pub fn value(message: impl Into<String>) -> Self { Self::new(ErrorKind::Value, message) }
    pub fn key(message: impl Into<String>) -> Self { Self::new(ErrorKind::Key, message) }
    pub fn index(message: impl Into<String>) -> Self { Self::new(ErrorKind::Index, message) }
    pub fn runtime(message: impl Into<String>) -> Self { Self::new(ErrorKind::Runtime, message) }
    pub fn overflow_error(field: &str) -> Self { Self::new(ErrorKind::Overflow, format!("{field} is too large")) }
    pub fn vacancy(role: &str, kind: &str) -> Self {
        Self { kind: ErrorKind::Vacancy, role: Some(role.to_owned()), message: format!(
            "no axis holds '{role}': give an axis the '{role}' fate to draw this {kind}") }
    }
    /// `except ValueError` also catches its subclasses.
    pub fn is_value_error(&self) -> bool { matches!(self.kind, ErrorKind::Value | ErrorKind::Unit | ErrorKind::Vacancy) }
    /// `except (TypeError, ValueError, KeyError)`.
    pub fn is_type_value_or_key(&self) -> bool { self.kind == ErrorKind::Type || self.kind == ErrorKind::Key || self.is_value_error() }
    pub fn python_name(&self) -> &'static str { self.kind.python_name() }
}

impl fmt::Display for PlotError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str(&self.message) }
}

impl std::error::Error for PlotError {}

impl From<crate::units::UnitError> for PlotError {
    fn from(error: crate::units::UnitError) -> Self { PlotError::new(ErrorKind::Unit, error.0) }
}
