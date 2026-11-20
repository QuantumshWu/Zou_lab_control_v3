//! The static checkpoint's solver (fr.solve with vlat/frx.py's three zero-norm root fixes, RUSTPLAN §1.1): the spectral
//! split by imaginary-part sweep with Gershgorin bounds (`split`), the preconditioner -- fast-fast block diagonal, the
//! slow Schur complement's harmonic diagonal blocks, each by exact sparse LU (`precond`) --, right-preconditioned MGS-GMRES
//! on the row-equilibrated residual to its floor (`gmres`), the certificate ζ ≤ 1 and the fallbacks exact S, direct LU
//! (`solve`).

pub mod split;
pub mod precond;
pub mod gmres;
pub mod solve;

pub use solve::{solve, Route, Solved};
