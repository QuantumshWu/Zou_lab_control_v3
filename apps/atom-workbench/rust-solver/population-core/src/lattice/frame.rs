//! The Schrieffer-Wolff / Floquet frame of a request, built once and shared
//! by the error estimate (`estimator`): the first-order kernels X~_s of the
//! explicit near steps in the eigen-coordinates of L0'
//! (`estimator::kernels`; elements above KERNEL_CAP are near resonant and
//! stay couplings) and one eigensystem of the second-order block
//! D' = L0' + sum_s K_s over every step (`estimator::blocks`), the same for
//! every charge.
//!
//! The preconditioner's Floquet kernels (`floquet`) are a different object:
//! separable n x n left/right frames without the recycling term, cheap to
//! apply on every block; the frame here is the superoperator (d x d) one
//! with the recycling inside L0' (the Raman rates).
use super::*;
use super::estimator::{KERNEL_CAP,KERNEL_ENTRY,KERNEL_HORIZON_RATE};
use super::estimator::blocks::BlockSystem;
use super::estimator::kernels::{KernelRule,SwKernels};

pub struct SwFrame{
    pub kernels:SwKernels,
    /// The second-order block over every step.
    pub class:BlockSystem,
    /// Build times (ms): kernels, block eigensystem.
    pub milliseconds:[f64;2],
}
impl SwFrame{
    /// The frame for outputs up to `horizon` (us; the kernels keep elements
    /// whose second-order rate matters over it).
    pub fn new(prepared:&LatticePrepared,horizon:f64)->Result<Self,String>{
        let basis=prepared.eigenbasis()?;let n=prepared.generator.n;
        let clock=Instant::now();
        let rule=KernelRule{entry:KERNEL_ENTRY,horizon_rate:KERNEL_HORIZON_RATE,horizon,cap:KERNEL_CAP};
        let kernels=SwKernels::new(prepared,basis,rule)?;
        let kernel_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
        let class=BlockSystem::new(&prepared.lapack,basis,&kernels,(0..kernels.steps.len()).collect(),n)?;
        Ok(Self{kernels,class,milliseconds:[kernel_ms,clock.elapsed().as_secs_f64()*1e3]})
    }
}
