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
//!   Y_m = (theta_m - A)^-1 F_m, the space's part of F_m solved in closed
//!   form and the ring (slaved at z = theta_m and fed back) by GCRO on one
//!   shared pool seeded with the pipeline's steady solve, the near-singular
//!   slow Ritz directions deflated (their response in closed form). The
//!   kernel mode's Y is the steady error.
//! - Fast Ritz modes: second-order Schrieffer-Wolff averaging on the near
//!   modes, the centre, its one-step neighbours and the explicit set P
//!   (`kernels`, `blocks`, `lattice`, `propagate`), sources transferred in
//!   from their neighbours at first order, no ring; the neighbours' own
//!   errors by the dressed blocks of L0' alone (five lasers, 20 us: without
//!   them the estimate is 0.50-0.54 of the actual error and accepts a
//!   checkpoint at 5.5e-3). Only the canonical member of each mirror pair
//!   is transformed and propagated (the flip symmetry of the fields).
//! - Horizon rule: the slow part runs for an output time T only when
//!   T lambda_slow >= 1 (lambda_slow the slowest decay rate of the reduced
//!   model); before that the slow modes' far sources have not grown and the
//!   two large terms of the exact form cancel badly: every Ritz mode goes
//!   through the SW part.
//! Outputs: the trajectory on the grid (laboratory readout), each horizon T
//! (centre readout: the time average) and the steady state. The fold
//! remainders (`ledger`) the lattice holds are charged by their computed
//! first-order effect on the outputs, the others by the stored fold's
//! sensitivity times their size (`fold_charge`): the bound of an output is
//! metric(Krylov error) + metric(built effect) + unbuilt charge, charged
//! where the Krylov error alone does not reject the checkpoint. A gate
//! (`gate`) skips the estimate while the outputs still move by more than
//! 4 eps between checkpoints. Errors are measured with `ledger::metric`.
use super::*;
use super::estimate_api::{ACCEPTANCE_FACTOR,EstimateSetup,Estimator,ErrorEstimate,EstimateWork,FoldCharges,FrameColumns,ReducedView,SteadySolves};
use super::ledger::{self,metric};

pub mod kernels;
pub mod blocks;
pub mod lattice;
pub mod propagate;
pub mod slow;
pub mod gate;
pub mod fold_charge;
pub use kernels::KernelRule;
pub use lattice::{Classes,ErrorLattice};
pub use propagate::{FieldCoordinates,FieldSources,Readout,RitzSources};
pub use slow::{SlowPool,SlowSolution};
pub use gate::OutputGate;
pub use fold_charge::FoldCharge;

/// Kernel support: the first-order entry floor and the horizon floor on an
/// entry's second-order rate times the longest output time. R2: 5e-3 keeps
/// 0.3% of the beat kernel's entries and reproduces the full kernels for
/// the estimate (5e-2 does not); the reduced model's frame (`frame`) needs
/// more: five lasers + tweezer at 20 us grew 2-19 pseudo-modes from the
/// 12th solve at 5e-3 (accepted at 40 solves), none at 1e-3 (0.6-2% of
/// the entries; accepted at 12), 1e-4 or 1e-5.
pub const KERNEL_ENTRY:f64=1e-3;
pub const KERNEL_HORIZON_RATE:f64=1e-2;
/// Near-resonant entries above this stay couplings: the first-order
/// kernel is no small parameter there (N1).
pub const KERNEL_CAP:f64=1.;
/// Every near mode's block carries every step's second-order term: one real
/// eigensystem. Measured against the per-neighbour classes of R2 (9-19
/// complex eigensystems, 10-18 s on three/four lasers): the same estimate to
/// 0.5% on every two/three/four-laser checkpoint, the build 1.0-1.2 s.
pub const CLASSES:Classes=Classes::Full;
/// Whether the explicit set joins the near modes of the fast part.
pub const NEAR_EXPLICIT:bool=true;
/// A mode decays when |Re theta| is above this (1/us).
const DECAYING:f64=1e-10;
/// A kernel mode whose source |A W b + ring feedback| is below this
/// fraction of |A| |W b| (|A| the largest image of a basis column) is
/// invariant to roundoff: its error is zero. Every further conserved
/// quantity of a drive-free lattice is such a mode, and the bordered system
/// (one trace row) is singular on it.
const INVARIANT:f64=1e-12;

/// The longest output time of a request.
pub fn longest_time(times:&[f64],horizons:&[f64])->f64{times.iter().chain(horizons).fold(0_f64,|a,&b|a.max(b.abs()))}

/// The reduced model of a checkpoint in the estimator's terms: the basis
/// and images, X (k x r) and S (r x r) column-major, the values and
/// coefficients the outputs use.
pub struct Checkpoint<'c>{pub basis:&'c[Vec<C>],pub images:&'c[Vec<C>],pub x:&'c[C],pub s:&'c[C],pub theta:&'c[C],pub coefficients:&'c[C],
    /// The projection's frame coordinates of new basis columns, if any.
    pub frame:Option<FrameColumns<'c>>}

/// Population errors of one checkpoint's outputs (before the metric and the
/// fold charges), and what ran.
#[derive(Clone,Debug,Default)]
pub struct Errors{
    pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Option<Vec<f64>>,
    /// Slow modes solved, whether the trajectory used them, lambda_slow.
    pub slow_modes:usize,pub slow_trajectory:bool,pub lambda_slow:f64,
    /// Time (ms): the fast part's columns, the pool's columns, the slow
    /// solves (sources, ring, GCRO), the fast propagation, the slow
    /// propagation.
    pub milliseconds:[f64;5],
}

pub struct UnifiedEstimate<'a>{
    setup:EstimateSetup<'a>,
    pub lattice:ErrorLattice<'a>,
    pub pool:Option<SlowPool>,
    pub gate:OutputGate,
    /// The fold remainders' charge (None: left out, for validation).
    pub fold:Option<FoldCharge<'a>>,
    /// The pipeline's steady-solve directions taken so far.
    seeded:usize,
}

/// The estimator of a request (`estimate_api::Construct`).
pub const CONSTRUCT:super::estimate_api::Construct=construct;
pub fn construct(setup:EstimateSetup<'_>)->Result<Box<dyn Estimator+'_>,String>{Ok(Box::new(UnifiedEstimate::new(setup)?))}

/// The reduced model's left functionals of the Ritz modes `modes` on W^T y:
/// rows of S^-1 (X^H X)^-1 X^H (X k x r, S r x r, column-major), so that
/// l_j(W X s_i) = delta_ij.
fn left_functionals(lapack:&Lapack,x:&[C],s:&[C],k:usize,r:usize,modes:&[usize])->Result<Vec<Vec<C>>,String>{
    let mut xhx=vec![ZERO;r*r];
    for a in 0..r{for b in 0..r{let mut v=ZERO;for i in 0..k{mul_add(&mut v,conj(x[i+k*a]),x[i+k*b]);}xhx[a+r*b]=v;}}
    let xh:Vec<C>=(0..k).flat_map(|i|(0..r).map(move|a|conj(x[i+k*a]))).collect();
    let(lu,piv)=lapack.complex_factor(&xhx,r)?;let n=lapack.complex_solve_matrix_factored(&lu,&piv,&xh,r)?;
    let(lu,piv)=lapack.complex_factor(s,r)?;let m=lapack.complex_solve_matrix_factored(&lu,&piv,&n,r)?;
    Ok(modes.iter().map(|&j|(0..k).map(|i|m[j+r*i]).collect()).collect())
}

/// B = X S (column-major k x r) of a checkpoint.
fn ritz_vectors(lapack:&Lapack,cp:&Checkpoint<'_>)->Result<Vec<C>,String>{
    let(k,r)=(cp.basis.len(),cp.theta.len());let mut b=vec![ZERO;k*r];lapack.complex_gemm().multiply(k,r,r,ONE,cp.x,k,cp.s,r,ZERO,&mut b,k)?;Ok(b)
}

impl<'a> UnifiedEstimate<'a>{
    pub fn new(setup:EstimateSetup<'a>)->Result<Self,String>{Self::with_near(setup,NEAR_EXPLICIT,true)}
    /// Without the fold charge: the Krylov part alone (validation).
    pub fn without_fold(setup:EstimateSetup<'a>)->Result<Self,String>{Self::with_near(setup,NEAR_EXPLICIT,false)}
    /// With or without the explicit set among the near modes, and the fold charge.
    pub fn with_near(setup:EstimateSetup<'a>,near_explicit:bool,fold:bool)->Result<Self,String>{
        let modes=setup.lattice.modes();
        let explicit=if near_explicit{setup.lattice.explicit.clone()}else{Vec::new()};
        let lattice=ErrorLattice::new(setup.prepared,&modes,&explicit,setup.frame,CLASSES)?;
        let fold=if fold{Some(FoldCharge::new(&setup)?)}else{None};
        Ok(Self{setup,lattice,pool:None,gate:OutputGate::default(),fold,seeded:0})
    }
    fn pool(&mut self)->Result<&mut SlowPool,String>{
        if self.pool.is_none(){self.pool=Some(SlowPool::new(self.setup.prepared,&self.lattice,longest_time(self.setup.outputs.times,self.setup.outputs.horizons))?);}
        Ok(self.pool.as_mut().unwrap())
    }
    /// Seeds the slow modes' pool with directions z and their images A z on
    /// P u Q (A unbordered): the inner directions of the pipeline's steady
    /// solve, whose images its Arnoldi relation gives for free.
    pub fn seed(&mut self,directions:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{self.pool()?.seed(directions,images)}
    /// The population errors of a checkpoint's outputs; `slow_responses`:
    /// the pool is deflated by every slow Ritz mode (the fold charge's slow
    /// responses) even when no output here takes them.
    pub fn errors(&mut self,cp:&Checkpoint<'_>,slow_responses:bool,steady:&mut dyn SteadySolves)->Result<Errors,String>{
        let(k,r)=(cp.basis.len(),cp.theta.len());let outputs=self.setup.outputs;
        let mut ms=[0.;5];let clock=Instant::now();
        // The fast part's columns (none when no output propagates).
        if !(outputs.times.is_empty()&&outputs.horizons.is_empty()){self.lattice.add_columns(cp.basis,cp.images,cp.frame.as_ref())?;}
        ms[0]=clock.elapsed().as_secs_f64()*1e3;
        let b=ritz_vectors(&self.setup.prepared.lapack,cp)?;
        // The slowest decay rate, the slow modes and which outputs use them.
        let lambda_slow=cp.theta.iter().zip(cp.coefficients).filter(|(z,a)|z.re.abs()>DECAYING&&nonzero(**a)).map(|(z,_)|z.re.abs()).fold(f64::INFINITY,f64::min);
        let lambda_slow=if lambda_slow.is_finite(){lambda_slow}else{0.};
        let uses=|t:f64|t*lambda_slow>=1.;
        let slow:Vec<usize>=(0..r).filter(|&m|magnitude(cp.theta[m])<slow::SLOW&&nonzero(cp.coefficients[m])).collect();
        let grid=outputs.times.last().is_some_and(|&t|uses(t));
        let horizon_slow:Vec<bool>=outputs.horizons.iter().map(|&t|uses(t)).collect();
        let kernel:Vec<usize>=slow.iter().copied().filter(|&m|magnitude(cp.theta[m])<1e-12).collect();
        // The kernel modes first; every slow mode deflates the pool once an
        // output takes one.
        let all:Vec<usize>=kernel.iter().chain(slow.iter().filter(|m|!kernel.contains(m))).copied().collect();
        let long=grid||horizon_slow.iter().any(|&h|h);
        let needed:Vec<usize>=if long{all.clone()}else if outputs.steady{kernel.clone()}else{Vec::new()};
        let deflated:Vec<usize>=if long||slow_responses{all}else{needed.clone()};
        let mut solutions=Vec::new();
        let scale=cp.images.iter().map(|a|a.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()).fold(0_f64,f64::max);
        let size=|v:&[C]|v.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        if !deflated.is_empty(){
            let clock=Instant::now();
            self.pool()?.add_columns(cp.basis)?;ms[1]=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
            let length=self.lattice.dimension();
            // x = W b_m and A x per slow mode. Away from the kernel the solves
            // deflate every slow Ritz mode (`SlowPool::mode`).
            let fields:Vec<(Vec<C>,Vec<C>)>=deflated.iter().map(|&m|{let(mut x,mut ax)=(vec![ZERO;length],vec![ZERO;length]);
                for(j,&c)in b[k*m..k*(m+1)].iter().enumerate(){if !nonzero(c){continue;}
                    for((xv,av),(&w,&aw))in x.iter_mut().zip(ax.iter_mut()).zip(cp.basis[j].iter().zip(&cp.images[j])){mul_add(xv,c,w);mul_add(av,c,aw);}}
                (x,ax)}).collect();
            if deflated.iter().any(|&m|magnitude(cp.theta[m])>=1e-12){
                let left=left_functionals(&self.setup.prepared.lapack,cp.x,cp.s,k,r,&deflated)?;
                let(xs,axs):(Vec<Vec<C>>,Vec<Vec<C>>)=fields.iter().cloned().unzip();
                let thetas:Vec<C>=deflated.iter().map(|&m|cp.theta[m]).collect();let bs:Vec<Vec<C>>=deflated.iter().map(|&m|b[k*m..k*(m+1)].to_vec()).collect();
                self.pool.as_mut().unwrap().deflate(&deflated,&thetas,&bs,&left,&xs,&axs)?;
            }
            for(&m,(x,ax))in deflated.iter().zip(&fields).filter(|(m,_)|needed.contains(m)){
                // The ring's feedback of x at theta.
                let(theta,bm)=(cp.theta[m],&b[k*m..k*(m+1)]);
                let g=self.lattice.ring_feedback(x,theta)?;
                if kernel.contains(&m){let f:Vec<C>=ax.iter().zip(&g).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();if size(&f)<=INVARIANT*scale*size(x){continue;}}
                solutions.push(self.pool.as_mut().unwrap().mode(bm,ax,&g,theta,cp.coefficients[m],steady)?);
            }
            ms[2]=clock.elapsed().as_secs_f64()*1e3;
        }
        let fast_ms=std::cell::Cell::new(0.);let slow_ms=std::cell::Cell::new(0.);
        let fast=|lattice:&ErrorLattice<'_>,skip:bool,times:&[f64],readout:Readout|->Result<Vec<Vec<f64>>,String>{
            let clock=Instant::now();
            let coefficients:Vec<C>=(0..r).map(|m|if skip&&slow.contains(&m){ZERO}else{cp.coefficients[m]}).collect();
            let out=lattice.populations(&RitzSources{vectors:&b,values:cp.theta,coefficients:&coefficients},times,readout);
            fast_ms.set(fast_ms.get()+clock.elapsed().as_secs_f64()*1e3);out};
        let sum=|a:Vec<Vec<f64>>,c:Vec<Vec<f64>>|->Vec<Vec<f64>>{a.into_iter().zip(c).map(|(x,y)|x.iter().zip(&y).map(|(p,q)|p+q).collect()).collect()};
        // Without slow modes the slow part is zero and every mode is fast.
        let grid=grid&&!solutions.is_empty();
        let trajectory=if grid{
            let clock=Instant::now();
            let s=self.pool.as_mut().unwrap().populations(&self.lattice,cp.basis,cp.x,cp.s,cp.theta,&solutions,outputs.times,Readout::Laboratory)?;
            slow_ms.set(slow_ms.get()+clock.elapsed().as_secs_f64()*1e3);
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
        ms[3]=fast_ms.get();ms[4]=slow_ms.get();
        Ok(Errors{trajectory,horizons,steady,slow_modes:solutions.len(),slow_trajectory:grid,lambda_slow,milliseconds:ms})
    }
}

impl UnifiedEstimate<'_>{
    /// The work counters so far: the fast part's, the pool's, the fold charge's.
    fn counters(&self)->(lattice::ErrorWork,slow::PoolWork,fold_charge::FoldWork){
        (self.lattice.work.clone(),self.pool.as_ref().map(|p|p.work.clone()).unwrap_or_default(),self.fold.as_ref().map(|f|f.work.clone()).unwrap_or_default())
    }
    /// The work since `before` and `clock`: block operations (the pool's
    /// products with Q in block units, 4 real multiply-adds per complex one,
    /// d^2 per block; one per block for an application of the fold or of
    /// the built remainders), ring solves, the fold charge's time.
    fn work_since(&self,before:&(lattice::ErrorWork,slow::PoolWork,fold_charge::FoldWork),clock:Instant)->EstimateWork{
        let(w,p,f)=self.counters();
        let blocks=self.lattice.modes.len();
        let qr=((p.qr_macs-before.1.qr_macs)/(4.*(self.lattice.d*self.lattice.d)as f64)).round()as usize;
        let mut fold_ms=f.milliseconds;for(a,b)in fold_ms.iter_mut().zip(before.2.milliseconds){*a-=b;}
        EstimateWork{
            block_operations:w.block_operations-before.0.block_operations+w.frame_blocks-before.0.frame_blocks+p.frame_blocks-before.1.frame_blocks
                +(p.applications-before.1.applications+p.preconditions-before.1.preconditions)*blocks+qr
                +(f.remainders-before.2.remainders+f.folds-before.2.folds)*blocks,
            block_solves:w.block_solves-before.0.block_solves,inner_steps:0,fold_ms,milliseconds:clock.elapsed().as_secs_f64()*1e3}
    }
}

impl Estimator for UnifiedEstimate<'_>{
    fn estimate(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves)->Result<ErrorEstimate,String>{
        let clock=Instant::now();
        let(k,r)=(reduced.basis.len(),reduced.r);
        let groups=self.setup.outputs.groups;let outputs=self.setup.outputs;
        // The new columns enter now, with the projection's frame coordinates
        // (handed over once per checkpoint); their work counts here.
        let before=self.counters();
        if !(outputs.times.is_empty()&&outputs.horizons.is_empty()){self.lattice.add_columns(reduced.basis,reduced.images,reduced.frame_columns.as_ref())?;}
        // X and S column-major and complex.
        let x:Vec<C>=(0..k*r).map(|i|C{re:reduced.x[(i%k)*r+i/k],im:0.}).collect();
        let s:Vec<C>=(0..r*r).map(|i|reduced.vectors[(i%r)*r+i/r]).collect();
        // The gate: the change of the pipeline's outputs since the previous
        // checkpoint; at the first one a steady-only request compares its
        // steady state with the pipeline's own steady solve.
        let current=gate::Outputs{trajectory:reduced.outputs.trajectory.to_vec(),horizons:reduced.outputs.horizons.to_vec(),
            steady:outputs.steady.then(||reduced.outputs.steady.to_vec())};
        let propagates=!outputs.times.is_empty()||!outputs.horizons.is_empty();
        let baseline=reduced.steady_baseline.filter(|_|!propagates).map(|b|metric(reduced.outputs.steady,b,groups));
        let change=self.gate.change(current,groups).or(baseline);
        let limit=gate::GATE_FACTOR*ledger::POPULATION_TOLERANCE;
        // Closed at the first checkpoint without a baseline (reported as not
        // estimated) and while the outputs move by more than the limit.
        let change=change.unwrap_or(f64::INFINITY);
        if change>limit&&self.gate.last_estimate.is_none_or(|e|e>limit){
            return Ok(ErrorEstimate{trajectory:vec![change;outputs.times.len()],horizons:vec![change;outputs.horizons.len()],
                steady:outputs.steady.then_some(change),fold:FoldCharges::default(),work:self.work_since(&before,clock)});
        }
        // The pipeline's steady-solve directions not taken yet seed the pool.
        let(directions,images)=steady.directions();
        if directions.len()>self.seeded{self.seed(&directions[self.seeded..],&images[self.seeded..])?;self.seeded=directions.len();}
        let cp=Checkpoint{basis:reduced.basis,images:reduced.images,x:&x,s:&s,theta:reduced.used_theta,coefficients:reduced.used_coefficients,frame:None};
        let errors=self.errors(&cp,self.fold.is_some()&&!outputs.horizons.is_empty(),steady)?;
        // The fold remainders: the built ones' effect, the unbuilt ones'
        // charge; a checkpoint the Krylov error alone rejects needs neither.
        let zero=vec![0.;self.lattice.n];let size=|p:&[f64]|metric(p,&zero,groups);
        let krylov=errors.trajectory.iter().chain(&errors.horizons).chain(errors.steady.iter()).map(|p|size(p)).fold(0.,f64::max);
        let fold=if self.fold.is_some()&&ACCEPTANCE_FACTOR*krylov<=ledger::POPULATION_TOLERANCE{
            let b=ritz_vectors(&self.setup.prepared.lapack,&cp)?;self.pool()?;
            let Self{fold,lattice,pool,setup,..}=self;
            Some(fold.as_mut().unwrap().effect(&cp,&b,errors.lambda_slow,lattice,pool.as_mut().unwrap(),steady,outputs,setup.ledger)?)
        }else{None};
        let built=|o:Option<&Vec<f64>>|o.map_or(0.,|p|size(p));
        let charges=FoldCharges{built:fold.as_ref().map_or(0.,|f|f.trajectory.iter().chain(&f.horizons).chain(f.steady.iter()).map(|p|size(p)).fold(0.,f64::max)),
            unbuilt:fold.as_ref().map_or(0.,|f|f.unbuilt)};
        let trajectory=errors.trajectory.iter().enumerate().map(|(i,p)|size(p)+built(fold.as_ref().map(|f|&f.trajectory[i]))+charges.unbuilt).collect();
        let horizons=errors.horizons.iter().enumerate().map(|(i,p)|size(p)+built(fold.as_ref().map(|f|&f.horizons[i]))+charges.unbuilt).collect();
        let steady_error=errors.steady.as_ref().map(|p|size(p)+built(fold.as_ref().and_then(|f|f.steady.as_ref()))+charges.unbuilt);
        let estimate=ErrorEstimate{trajectory,horizons,steady:steady_error,fold:charges,work:self.work_since(&before,clock)};
        self.gate.last_estimate=Some(estimate.largest());
        Ok(estimate)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::estimate_api::{LatticeSets,ModelOutputs,Outputs};
    use super::super::operator::LatticeOperator;
    use super::super::tests::{fixture,prepared};
    use super::super::frame::SwFrame;
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
        let ledger=op.ledger().unwrap();
        let times=[0.,0.5,1.];let horizons=[10.];let groups=vec![vec![0,1]];
        let frame=SwFrame::new(&p,longest_time(&times,&horizons));
        let setup=EstimateSetup{prepared:&p,lattice:&lattice,outputs:Outputs{times:&times,horizons:&horizons,steady:false,groups:&groups},frame:&frame,ledger:&ledger,initial:&initial};
        let mut estimator=CONSTRUCT(setup).unwrap();
        let x=[1.,0.,0.,1.];let c0=[1.,0.];let kernel=[false,false];
        let(trajectory,horizon_rows,steady_row)=(vec![vec![0.;n];3],vec![vec![0.;n]],vec![0.;n]);
        let reduced=ReducedView{basis:&basis,images:&images,r:2,x:&x,g:&g,c0:&c0,theta:&theta,vectors:&sr,coefficients:&a,
            used_theta:&theta,used_coefficients:&a,kernel:&kernel,growth_charge:0.,steady:None,
            outputs:ModelOutputs{trajectory:&trajectory,horizons:&horizon_rows,steady:&steady_row},steady_baseline:None,frame_columns:None};
        // The first checkpoint is not estimated; the second (unchanged outputs) is.
        let mut solver=super::super::poles::Lattice::new(&p,lattice.clone()).unwrap();
        let first=estimator.estimate(&reduced,&mut solver).unwrap();
        assert!(first.largest().is_infinite());
        let e=estimator.estimate(&reduced,&mut solver).unwrap();
        assert_eq!((e.trajectory.len(),e.horizons.len()),(3,1));
        assert!(e.trajectory[0].abs()<1e-12&&e.trajectory[1..].iter().chain(&e.horizons).all(|v|v.is_finite()&&*v>0.),"{e:?}");
        assert!(e.largest().is_finite()&&first.work.block_operations+e.work.block_operations>0);
    }
}
