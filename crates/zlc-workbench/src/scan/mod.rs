//! Scan authoring and application execution. Immutable geometry lives in
//! zlc_core::scan; this module owns authoring policy and run orchestration.
pub mod authoring;
pub mod wire;
pub mod execution;

pub use authoring::dispatch;
