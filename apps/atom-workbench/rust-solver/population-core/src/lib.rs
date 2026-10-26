//! Fixed-manifold full-density population engine and complete E/F kernels.
//! No physical model is rebuilt; all coefficient entries and mode edges are
//! copied once from the current prepared plan. No trajectory is retained.
mod lapack;
pub use lapack::{Complex64, Lapack};
use lapack::Complex64 as C;
mod parallel;
pub mod preparation;
mod pencil;
pub use pencil::{StageProfile, STAGE_COUNT, STAGE_NAMES};
pub mod owned_prepared;
mod dense_expm;

/// Sparse row-major (CSR) complex matrix.
pub(crate) struct Coefficient {
    pub(crate) rows: usize,
    pub(crate) cols: usize,
    pub(crate) row: Vec<usize>,
    pub(crate) col: Vec<usize>,
    pub(crate) value: Vec<C>,
}

/// Finite check for complex entries, used by every module.
pub(crate) trait Finite {
    fn finite(self) -> bool;
}

impl Finite for C {
    #[inline]
    fn finite(self) -> bool {
        self.re.is_finite() && self.im.is_finite()
    }
}

pub(crate) fn product(values: &[usize]) -> Result<usize, String> {
    values.iter().try_fold(1usize, |a, &b| a.checked_mul(b).ok_or_else(|| "Shape multiplication overflow".into()))
}
