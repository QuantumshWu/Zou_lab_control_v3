//! The Schrieffer-Wolff / Floquet frame of a request, built once for the
//! error estimate's fast part (`estimator`): the first-order kernels X~_s of
//! the explicit near steps in the eigen-coordinates of L0'
//! (`estimator::kernels`; elements above KERNEL_CAP are near resonant and
//! stay couplings) and one eigensystem D' = V diag(mu) V^-1 of the
//! second-order block L0' + sum_s K_s over every step (`estimator::blocks`),
//! the same for every charge. Its block's eigenvalues are also the classes
//! the slowest relaxation is checked against (`pipeline`).
//!
//! The preconditioner's Floquet kernels (`floquet`) are a different object:
//! separable n x n left/right frames without the recycling term, cheap to
//! apply on every block; the frame here is the superoperator (d x d) one
//! with the recycling inside L0' (the Raman rates).
use super::*;
use super::estimator::{KERNEL_CAP,KERNEL_ENTRY,KERNEL_HORIZON_RATE};
use super::estimator::blocks::BlockSystem;
use super::estimator::kernels::{KernelRule,SwKernels};
use std::sync::OnceLock;

/// The frame of a request, each part built on its first use (a part's
/// failure is kept and returned by every later use).
pub struct SwFrame<'p>{
    prepared:&'p LatticePrepared,rule:KernelRule,
    /// The longest output time it is built for (us).
    pub horizon:f64,
    /// The explicit steps' charges in plan order: the kernels' steps.
    pub steps:Vec<Charge>,
    /// The kernels and the second-order block over every step, each with
    /// its build time (ms).
    kernels:OnceLock<Result<(SwKernels,f64),String>>,
    class:OnceLock<Result<(BlockSystem,f64),String>>,
}
impl<'p> SwFrame<'p>{
    /// The frame for outputs up to `horizon` (us; the kernels keep elements
    /// whose second-order rate matters over it).
    pub fn new(prepared:&'p LatticePrepared,horizon:f64)->Self{
        let rule=KernelRule{entry:KERNEL_ENTRY,horizon_rate:KERNEL_HORIZON_RATE,horizon,cap:KERNEL_CAP};
        let plan=&prepared.plan;let steps=plan.explicit().into_iter().map(|i|plan.steps[i].charge.clone()).collect();
        Self{prepared,rule,horizon,steps,kernels:OnceLock::new(),class:OnceLock::new()}
    }
    pub fn kernels(&self)->Result<&SwKernels,String>{
        self.kernels.get_or_init(||{let clock=Instant::now();let basis=self.prepared.eigenbasis()?;
            Ok((SwKernels::new(self.prepared,basis,self.rule)?,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|&x.0).map_err(|e|e.clone())
    }
    /// The second-order block over every step.
    pub fn class(&self)->Result<&BlockSystem,String>{
        self.class.get_or_init(||{let kernels=self.kernels()?;let clock=Instant::now();let basis=self.prepared.eigenbasis()?;
            let class=BlockSystem::new(&self.prepared.lapack,basis,kernels,(0..kernels.steps.len()).collect(),self.prepared.generator.n,self.horizon)?;
            Ok((class,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|&x.0).map_err(|e|e.clone())
    }
    /// Build times (ms) of the parts built so far: kernels, block.
    pub fn milliseconds(&self)->[f64;2]{
        let ms=|x:Option<f64>|x.unwrap_or(0.);
        [ms(self.kernels.get().and_then(|r|r.as_ref().ok()).map(|x|x.1)),ms(self.class.get().and_then(|r|r.as_ref().ok()).map(|x|x.1))]
    }
}
