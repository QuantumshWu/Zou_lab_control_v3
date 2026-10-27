//! The Rb87 full-density population core: the folded Fourier lattice
//! (`lattice`) on the request's own frame (`preparation`). The model is
//! borrowed as supplied; no physical constant or coupling is introduced or
//! removed here.
mod lapack;
pub use lapack::{Complex64, Lapack};
use lapack::Complex64 as C;
mod parallel;
pub mod preparation;
mod dense_expm;
pub mod lattice;
