//! Typed scientific operations over immutable storage and schemas. This layer
//! has no transport, message dispatch, archive codec, session handles or UI.
pub mod numeric;
pub mod statistics;
mod tensor;
pub mod reduce;
pub mod groups;
pub mod binary;
pub mod geometry;
pub mod transform;
pub use tensor::Tensor;
