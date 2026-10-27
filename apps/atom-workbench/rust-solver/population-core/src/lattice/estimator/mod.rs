//! The error estimate of the unified pipeline (R2, unified/R2-estimate.md
//! sections 10-11): one error equation for every output f(A) x0 of the
//! reduced model.
//!
//! Error equation (exact): with R = A W X - W X G the residual of the
//! reduced model and u(t) its trajectory, the error of x(t) ~ W X u(t)
//! obeys E' = A E + R u(t), E(0) = 0, on the full lattice. Per Ritz mode
//! (b_m = X s_m, theta_m, a_m as the outputs use them) the source is
//! F_m a_m e^{theta_m t}, F_m = (A W - theta_m W) b_m, plus the outflow onto
//! the omitted ring R = ring(P u Q).
//!
//! - Slow Ritz modes (|theta| < `slow::SLOW`, the kernel included) are exact
//!   (`slow`): e_m(t) = a_m (e^{theta_m t} Y_m - e^{tA} Y_m) with
//!   Y_m = (theta_m - A)^-1 F_m on one shared GCRO pool, the ring slaved at
//!   z = theta_m and fed back. The kernel mode's Y is the steady error.
//! - Fast Ritz modes: second-order Schrieffer-Wolff averaging on the
//!   centre and its one-step neighbours (`kernels`, `blocks`, `lattice`,
//!   `propagate`), sources transferred in from their neighbours at first
//!   order, no ring.
//! - Horizon rule: the slow part runs for an output time T only when
//!   T lambda_slow >= 1 (lambda_slow the slowest decay rate of the reduced
//!   model); before that the slow modes' far sources have not grown and the
//!   two large terms of the exact form cancel badly: every Ritz mode goes
//!   through the SW part.
//! Outputs: the trajectory on the grid (laboratory readout), each horizon T
//! (centre readout: the time average) and the steady state. The fold
//! remainders (`ledger`) are generator perturbations of size delta: they
//! move a state by at most delta min(t, tau), tau the reduced model's
//! slowest relaxation time. A gate (`gate`) skips the estimate while the
//! outputs still move by more than 4 eps between checkpoints. Errors are
//! measured with `ledger::metric`.
use super::*;
use super::estimate_api::{EstimateSetup,Estimator,ErrorEstimate,EstimateWork,ReducedView,SteadySolves};
use super::ledger::{self,metric};

pub mod kernels;
pub mod blocks;
pub mod lattice;
pub mod propagate;
pub mod slow;
pub mod gate;
pub use kernels::KernelRule;
pub use lattice::{Classes,ErrorLattice};
pub use propagate::{Readout,RitzSources};
pub use slow::{SlowPool,SlowSolution};
pub use gate::OutputGate;

/// Kernel support: the first-order entry floor (R2: 5e-3 keeps 0.3% of the
/// beat kernel's entries and reproduces the full kernels; 5e-2 does not)
/// and the horizon floor on an entry's second-order rate times the longest
/// output time.
pub const KERNEL_ENTRY:f64=5e-3;
pub const KERNEL_HORIZON_RATE:f64=1e-2;
/// Near-resonant entries above this stay couplings (none by default).
pub const KERNEL_CAP:f64=f64::INFINITY;
/// Every near mode's block carries every step's second-order term: one real
/// eigensystem. Measured against the per-neighbour classes of R2 (9-19
/// complex eigensystems, 10-18 s on three/four lasers): the same estimate to
/// 0.5% on every two/three/four-laser checkpoint, the build 1.0-1.2 s.
pub const CLASSES:Classes=Classes::Full;
/// A mode decays when |Re theta| is above this (1/us).
const DECAYING:f64=1e-10;

/// The longest output time of a request.
pub fn longest_time(times:&[f64],horizons:&[f64])->f64{times.iter().chain(horizons).fold(0_f64,|a,&b|a.max(b.abs()))}

/// The reduced model of a checkpoint in the estimator's terms: the basis
/// and images, X (k x r) and S (r x r) column-major, the values and
/// coefficients the outputs use.
pub struct Checkpoint<'c>{pub basis:&'c[Vec<C>],pub images:&'c[Vec<C>],pub x:&'c[C],pub s:&'c[C],pub theta:&'c[C],pub coefficients:&'c[C]}

/// Population errors of one checkpoint's outputs (before the metric and the
/// fold charges), and what ran.
#[derive(Clone,Debug,Default)]
pub struct Errors{
    pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Option<Vec<f64>>,
    /// Slow modes solved, whether the trajectory used them, lambda_slow.
    pub slow_modes:usize,pub slow_trajectory:bool,pub lambda_slow:f64,
}

pub struct UnifiedEstimate<'a>{
    setup:EstimateSetup<'a>,
    pub lattice:ErrorLattice<'a>,
    pub pool:Option<SlowPool<'a>>,
    pub gate:OutputGate,
    /// Whether the pool's first kernel direction is the pipeline's bordered
    /// solve (else the Floquet preconditioner's). Measured on three and four
    /// lasers at 10 ms: the solve's 12-13 inner steps save about 5 GCRO
    /// steps, a net loss; off by default.
    pub seed_with_pipeline:bool,
}

/// The estimator of a request (`estimate_api::Construct`).
pub const CONSTRUCT:super::estimate_api::Construct=construct;
pub fn construct(setup:EstimateSetup<'_>)->Result<Box<dyn Estimator+'_>,String>{Ok(Box::new(UnifiedEstimate::new(setup)?))}

/// The reduced model's slowest relaxation time: 1 / |theta| over the modes
/// outside the kernel that carry weight.
pub fn relaxation_time(values:&[C],coefficients:&[C],kernel:&[bool])->f64{
    values.iter().zip(coefficients).zip(kernel).filter(|((_,a),&k)|!k&&nonzero(**a)).map(|((z,_),_)|1./magnitude(*z)).fold(0.,f64::max)
}

impl<'a> UnifiedEstimate<'a>{
    pub fn new(setup:EstimateSetup<'a>)->Result<Self,String>{
        let modes=setup.lattice.modes();
        let horizon=longest_time(setup.outputs.times,setup.outputs.horizons);
        let rule=KernelRule{entry:KERNEL_ENTRY,horizon_rate:KERNEL_HORIZON_RATE,horizon,cap:KERNEL_CAP};
        let lattice=ErrorLattice::new(setup.prepared,&modes,rule,CLASSES)?;
        let gate=OutputGate::new(lattice.n,lattice.nu.clone(),lattice.centre);
        Ok(Self{setup,lattice,pool:None,gate,seed_with_pipeline:false})
    }
    /// The population errors of a checkpoint's outputs.
    pub fn errors(&mut self,cp:&Checkpoint<'_>,steady:&mut dyn SteadySolves)->Result<Errors,String>{
        let(k,r)=(cp.basis.len(),cp.theta.len());let outputs=self.setup.outputs;
        self.lattice.add_columns(cp.basis,cp.images)?;
        let mut b=vec![ZERO;k*r];self.setup.prepared.lapack.complex_gemm().multiply(k,r,r,ONE,cp.x,k,cp.s,r,ZERO,&mut b,k)?;
        // The slowest decay rate, the slow modes and which outputs use them.
        let lambda_slow=cp.theta.iter().zip(cp.coefficients).filter(|(z,a)|z.re.abs()>DECAYING&&nonzero(**a)).map(|(z,_)|z.re.abs()).fold(f64::INFINITY,f64::min);
        let lambda_slow=if lambda_slow.is_finite(){lambda_slow}else{0.};
        let uses=|t:f64|t*lambda_slow>=1.;
        let slow:Vec<usize>=(0..r).filter(|&m|magnitude(cp.theta[m])<slow::SLOW&&nonzero(cp.coefficients[m])).collect();
        let grid=outputs.times.last().is_some_and(|&t|uses(t));
        let horizon_slow:Vec<bool>=outputs.horizons.iter().map(|&t|uses(t)).collect();
        let kernel:Vec<usize>=slow.iter().copied().filter(|&m|magnitude(cp.theta[m])<1e-12).collect();
        // The kernel modes first: the pool's first direction is the pipeline's solve of a kernel source.
        let needed:Vec<usize>=if grid||horizon_slow.iter().any(|&h|h){kernel.iter().chain(slow.iter().filter(|m|!kernel.contains(m))).copied().collect()}
            else if outputs.steady{kernel.clone()}else{Vec::new()};
        let mut solutions=Vec::new();
        if !needed.is_empty(){
            if self.pool.is_none(){let mut pool=SlowPool::new(self.setup.prepared,&self.lattice)?;pool.seed_with_pipeline=self.seed_with_pipeline;self.pool=Some(pool);}
            let pool=self.pool.as_mut().unwrap();pool.add_columns(cp.basis,cp.images)?;
            let length=self.lattice.dimension();
            for &m in &needed{
                // x = W b_m, F = (A W - theta W) b_m + the ring's feedback at theta.
                let(mut x,mut f)=(vec![ZERO;length],vec![ZERO;length]);let theta=cp.theta[m];
                for j in 0..k{let c=b[j+k*m];if !nonzero(c){continue;}let ct=mul(c,theta);
                    for((xv,fv),(&w,&aw))in x.iter_mut().zip(f.iter_mut()).zip(cp.basis[j].iter().zip(&cp.images[j])){mul_add(xv,c,w);mul_add(fv,c,aw);let v=mul(ct,w);fv.re-=v.re;fv.im-=v.im;}}
                let g=self.lattice.ring_feedback(&x,theta)?;for(a,v)in f.iter_mut().zip(g){add(a,v);}
                solutions.push(pool.solve(cp.basis,&f,theta,cp.coefficients[m],steady)?);
            }
        }
        let fast=|lattice:&ErrorLattice<'_>,skip:bool,times:&[f64],readout:Readout|->Result<Vec<Vec<f64>>,String>{
            let coefficients:Vec<C>=(0..r).map(|m|if skip&&slow.contains(&m){ZERO}else{cp.coefficients[m]}).collect();
            lattice.populations(&RitzSources{vectors:&b,values:cp.theta,coefficients:&coefficients},times,readout)};
        let sum=|a:Vec<Vec<f64>>,c:Vec<Vec<f64>>|->Vec<Vec<f64>>{a.into_iter().zip(c).map(|(x,y)|x.iter().zip(&y).map(|(p,q)|p+q).collect()).collect()};
        // Without slow modes the slow part is zero and every mode is fast.
        let grid=grid&&!solutions.is_empty();
        let trajectory=if grid{
            let s=self.pool.as_mut().unwrap().populations(&self.lattice,cp.basis,cp.x,cp.s,cp.theta,&solutions,outputs.times,Readout::Laboratory)?;
            sum(fast(&self.lattice,true,outputs.times,Readout::Laboratory)?,s)
        }else{fast(&self.lattice,false,outputs.times,Readout::Laboratory)?};
        let mut horizons=Vec::with_capacity(outputs.horizons.len());
        for(&t,&with)in outputs.horizons.iter().zip(&horizon_slow){
            let row=if with&&!solutions.is_empty(){
                let s=self.pool.as_mut().unwrap().populations(&self.lattice,cp.basis,cp.x,cp.s,cp.theta,&solutions,&[t],Readout::Centre)?;
                sum(fast(&self.lattice,true,&[t],Readout::Centre)?,s)
            }else{fast(&self.lattice,false,&[t],Readout::Centre)?};
            horizons.push(row.into_iter().next().unwrap());
        }
        let steady=if outputs.steady{
            let pool=self.pool.as_ref();let n=self.lattice.n;let mut e=vec![0.;n];
            for sol in solutions.iter().filter(|s|magnitude(s.theta)<1e-12){for(a,v)in e.iter_mut().zip(pool.unwrap().steady_populations(cp.basis,sol)){*a+=v;}}
            Some(e)
        }else{None};
        Ok(Errors{trajectory,horizons,steady,slow_modes:solutions.len(),slow_trajectory:grid,lambda_slow})
    }
}

impl Estimator for UnifiedEstimate<'_>{
    fn estimate(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves)->Result<ErrorEstimate,String>{
        let clock=Instant::now();
        let(k,r)=(reduced.basis.len(),reduced.r);
        let groups=self.setup.outputs.groups;let outputs=self.setup.outputs;
        // X and S column-major and complex; B = X S for the gate; the reduced steady state over the basis.
        let x:Vec<C>=(0..k*r).map(|i|C{re:reduced.x[(i%k)*r+i/k],im:0.}).collect();
        let s:Vec<C>=(0..r*r).map(|i|reduced.vectors[(i%r)*r+i/r]).collect();
        let mut b=vec![ZERO;k*r];self.setup.prepared.lapack.complex_gemm().multiply(k,r,r,ONE,&x,k,&s,r,ZERO,&mut b,k)?;
        let steady_coordinates:Option<Vec<C>>=reduced.steady.map(|u|(0..k).map(|i|C{re:(0..r).map(|l|reduced.x[i*r+l]*u[l]).sum(),im:0.}).collect());
        // The gate: the outputs' change since the previous checkpoint.
        self.gate.add_columns(reduced.basis);
        let current=self.gate.outputs(&self.setup.prepared.lapack,&b,reduced.used_theta,reduced.used_coefficients,outputs.times,outputs.horizons,
            if outputs.steady{steady_coordinates.as_deref()}else{None})?;
        let change=self.gate.change(current,groups);
        let limit=gate::GATE_FACTOR*ledger::POPULATION_TOLERANCE;
        // Closed at the first checkpoint (nothing to compare with: reported as
        // not estimated) and while the outputs move by more than the limit.
        let change=change.unwrap_or(f64::INFINITY);
        if change>limit&&self.gate.last_estimate.is_none_or(|e|e>limit){
            return Ok(ErrorEstimate{trajectory:vec![change;outputs.times.len()],horizons:vec![change;outputs.horizons.len()],
                steady:outputs.steady.then_some(change),work:EstimateWork{milliseconds:clock.elapsed().as_secs_f64()*1e3,..Default::default()}});
        }
        let before=(self.lattice.work.clone(),self.pool.as_ref().map(|p|p.work.clone()).unwrap_or_default());
        let cp=Checkpoint{basis:reduced.basis,images:reduced.images,x:&x,s:&s,theta:reduced.used_theta,coefficients:reduced.used_coefficients};
        let errors=self.errors(&cp,steady)?;
        let zero=vec![0.;self.lattice.n];
        let delta=self.setup.ledger.total();
        let tau=relaxation_time(reduced.used_theta,reduced.used_coefficients,reduced.kernel);
        let fold=|t:f64|delta*t.min(tau);
        let trajectory=errors.trajectory.iter().zip(outputs.times).map(|(p,&t)|metric(p,&zero,groups)+fold(t)).collect();
        let horizons=errors.horizons.iter().zip(outputs.horizons).map(|(p,&t)|metric(p,&zero,groups)+fold(t)).collect();
        let steady_error=errors.steady.as_ref().map(|p|metric(p,&zero,groups)+delta*tau);
        let(w,p)=(&self.lattice.work,self.pool.as_ref().map(|p|p.work.clone()).unwrap_or_default());
        let blocks=self.lattice.modes.len();
        let work=EstimateWork{
            block_operations:w.block_operations-before.0.block_operations+w.frame_blocks-before.0.frame_blocks+p.frame_blocks-before.1.frame_blocks
                +(p.applications-before.1.applications+p.preconditions-before.1.preconditions)*blocks,
            block_solves:w.block_solves-before.0.block_solves,inner_steps:p.inner_steps-before.1.inner_steps,milliseconds:clock.elapsed().as_secs_f64()*1e3};
        let estimate=ErrorEstimate{trajectory,horizons,steady:steady_error,work};
        self.gate.last_estimate=Some(estimate.largest());
        Ok(estimate)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::estimate_api::{LatticeSets,Outputs};
    use super::super::ledger::RemainderLedger;
    use super::super::operator::LatticeOperator;
    use super::super::tests::{fixture,prepared};
    struct NoSolves;
    impl SteadySolves for NoSolves{fn solve(&mut self,_:&[C],_:f64)->Result<(Vec<C>,usize),String>{Err("not used".into())}}
    #[test]
    fn estimator_runs_through_the_pipeline_interface(){
        // A two-column space (x0 and its image's new direction) on the
        // fixture's first ball, through `construct` and `estimate`: one
        // bound per output, finite and positive after t = 0.
        let f=fixture();let p=prepared(&f);let modes=p.modes(1).unwrap();
        let mut op=LatticeOperator::new(&p,&modes).unwrap();let(n,d)=(4,16);let size=op.dimension();
        let dot=|a:&[C],b:&[C]|a.iter().zip(b).map(|(x,y)|x.re*y.re+x.im*y.im).sum::<f64>();
        let mut w0=vec![ZERO;size];w0[op.centre*d]=ONE;
        let mut a0=vec![ZERO;size];op.apply(&w0,&mut a0).unwrap();
        let h=dot(&w0,&a0);let mut w1:Vec<C>=a0.iter().zip(&w0).map(|(a,b)|C{re:a.re-h*b.re,im:a.im-h*b.im}).collect();
        let s=dot(&w1,&w1).sqrt();for v in &mut w1{v.re/=s;v.im/=s;}
        let mut a1=vec![ZERO;size];op.apply(&w1,&mut a1).unwrap();
        let(basis,images)=(vec![w0,w1],vec![a0,a1]);
        let g:Vec<f64>=(0..2).flat_map(|i|(0..2).map(|j|dot(&basis[i],&images[j])).collect::<Vec<_>>()).collect();
        let(theta,vectors)=p.lapack.complex_eigen_right(&[C{re:g[0],im:0.},C{re:g[2],im:0.},C{re:g[1],im:0.},C{re:g[3],im:0.}],2).unwrap();
        // Row-major S and a = S^-1 c0, c0 = (1, 0).
        let sr=[vectors[0],vectors[2],vectors[1],vectors[3]];
        let det=C{re:mul(sr[0],sr[3]).re-mul(sr[1],sr[2]).re,im:mul(sr[0],sr[3]).im-mul(sr[1],sr[2]).im};
        let inv=|z:C|{let m=z.re*z.re+z.im*z.im;C{re:z.re/m,im:-z.im/m}};
        let a=[mul(sr[3],inv(det)),mul(C{re:-sr[2].re,im:-sr[2].im},inv(det))];
        let mut initial=vec![ZERO;n*n];initial[0]=ONE;
        let lattice=LatticeSets{explicit:modes.clone(),ring:Vec::new()};
        let ledger=RemainderLedger::new(&p.plan,op.nu_max(),op.dropped()).unwrap();
        let times=[0.,0.5,1.];let horizons=[10.];let groups=vec![vec![0,1]];
        let setup=EstimateSetup{prepared:&p,lattice:&lattice,outputs:Outputs{times:&times,horizons:&horizons,steady:false,groups:&groups},ledger:&ledger,initial:&initial};
        let mut estimator=CONSTRUCT(setup).unwrap();
        let x=[1.,0.,0.,1.];let c0=[1.,0.];let kernel=[false,false];
        let reduced=ReducedView{basis:&basis,images:&images,r:2,x:&x,g:&g,c0:&c0,theta:&theta,vectors:&sr,coefficients:&a,
            used_theta:&theta,used_coefficients:&a,kernel:&kernel,growth_charge:0.,steady:None};
        // The first checkpoint is not estimated; the second (unchanged outputs) is.
        let first=estimator.estimate(&reduced,&mut NoSolves).unwrap();
        assert!(first.largest().is_infinite());
        let e=estimator.estimate(&reduced,&mut NoSolves).unwrap();
        assert_eq!((e.trajectory.len(),e.horizons.len()),(3,1));
        assert!(e.trajectory[0].abs()<1e-12&&e.trajectory[1..].iter().chain(&e.horizons).all(|v|v.is_finite()&&*v>0.),"{e:?}");
        assert!(e.largest().is_finite()&&e.work.block_operations>0);
    }
}
