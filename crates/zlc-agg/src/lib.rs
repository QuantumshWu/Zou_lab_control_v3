//! Byte-exact port of Matplotlib 3.10's Agg renderer (`_backend_agg`) and
//! the agg24 parts it uses. Arithmetic keeps C++ operand order, integer
//! wrapping and conversion semantics; no stage is reformulated.
pub mod basics;
pub mod color;
pub mod curves;
pub mod path;
pub mod pixfmt;
pub mod rasterizer;
pub mod renderer;
pub mod scanline;
pub mod span;
pub mod trans_affine;
pub mod vcgen;

pub use renderer::{AggError, BufferRegion, ClipPath, Collection, Dashes, Gc, RendererAgg, SketchParams};
