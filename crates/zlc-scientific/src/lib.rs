//! Shared scientific array semantics. No UI, plotting, device, or interpreter dependencies.
//! Derive arithmetic accepts bool, integer and real floating arrays, as the
//! Python Operand contract does. Pure selection also preserves complex storage.
pub mod numeric;
pub mod statistics;
mod tensor;
mod service;
mod reduce;
mod groups;
mod binary;
mod geometry;
pub use tensor::Tensor;
pub use service::Service;
