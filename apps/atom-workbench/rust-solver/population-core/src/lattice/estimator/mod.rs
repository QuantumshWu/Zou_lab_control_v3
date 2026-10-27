//! The error estimate of the unified pipeline (R2, unified/R2-estimate.md
//! section 10): one error equation for every output f(A) x0 of the reduced
//! model.
//!
//! Error equation (exact): with R = A W X - W X G the residual of the
//! reduced model and u(t) its trajectory, the error of x(t) ~ W X u(t)
//! obeys E' = A E + R u(t), E(0) = 0, on the full lattice. Embedded in the
//! lattice P u Q u R (R = ring(P u Q), the omitted ring), its rows on P u Q
//! are the Galerkin residual (inexact inner solves, the trace correction
//! and the growing-mode handling included) and its rows on R the outflow
//! C W u(t) (the truncation, signed). Per Ritz mode (b_m = X s_m, theta_m,
//! a_m as the outputs use them) the source is (A W b_m - W b_m theta_m) a_m
//! e^{theta_m t}.
//!
//! The ring is slaved at z = 0: y = -D_R^-1 C W u, exact block solves, its
//! own populations read out and its feedback B y a source on P u Q for all
//! t (a source two steps from the slow centre that the first-order
//! transform below could not carry to it). The error on P u Q is
//! propagated by second-order Schrieffer-Wolff averaging (`kernels`,
//! `blocks`): the source through I - S, every mode's block exactly in its
//! eigensystem (`propagate`), the readout through I + S.
//!
//! Outputs: the trajectory on the grid (laboratory readout), each horizon T
//! (the same equation at t = T, centre readout: the time average), and the
//! steady state by GCRO recycling the space (`steady`); with several
//! conserved quantities the steady error is the horizon's at
//! SEVERAL_KERNELS_HORIZON. The fold remainders (`ledger`) are generator
//! perturbations of size delta: they move a state by at most
//! delta min(t, tau), tau the reduced model's slowest relaxation time.
//! Errors are measured with `ledger::metric`.
//!
//! Work: every basis column once (`lattice::ErrorLattice::add_columns`:
//! the ring's couplings and block solves, the coordinate changes); per
//! checkpoint the reduced products with the Ritz data, the time GEMMs and
//! the steady GCRO's few steps. A gate (`gate`) skips the estimate while
//! the outputs still move by more than 4 eps between checkpoints.
use super::*;
use super::estimate_api::{EstimateSetup,Estimator,ErrorEstimate,EstimateWork,ReducedView,SteadySolves};
use super::ledger::{self,metric};

pub mod kernels;
pub mod blocks;
pub mod lattice;
pub mod propagate;
pub mod steady;
pub mod gate;
pub use kernels::KernelRule;
pub use lattice::{Classes,ErrorLattice};
pub use propagate::{Readout,RitzSources};
pub use steady::SteadyRecycler;
pub use gate::OutputGate;

/// Kernel support: the first-order entry floor (R2: 5e-3 keeps 0.3% of the
/// beat kernel's entries and reproduces the full kernels; 5e-2 does not)
/// and the horizon floor on an entry's second-order rate times the longest
/// output time.
pub const KERNEL_ENTRY:f64=5e-3;
pub const KERNEL_HORIZON_RATE:f64=1e-2;
/// Near-resonant entries above this stay couplings (none by default: R2's
/// cap 0.5 is under evaluation for the no-tweezer four/five-laser lattices).
pub const KERNEL_CAP:f64=f64::INFINITY;
/// Every mode's block carries every step's second-order term: one real
/// eigensystem for the lattice. Measured against the per-neighbour classes
/// of R2 (9-19 complex eigensystems, 10-18 s on three/four lasers): the
/// same estimate to 0.5% on every two/three/four-laser checkpoint, the
/// build 1.0-1.2 s.
pub const CLASSES:Classes=Classes::Full;
/// With several conserved quantities the steady error is the error
/// equation's at this time (us).
pub const SEVERAL_KERNELS_HORIZON:f64=1e9;

/// The longest output time of a request.
pub fn longest_time(times:&[f64],horizons:&[f64])->f64{times.iter().chain(horizons).fold(0_f64,|a,&b|a.max(b.abs()))}

pub struct UnifiedEstimate<'a>{
    setup:EstimateSetup<'a>,
    pub lattice:ErrorLattice<'a>,
    pub steady:Option<SteadyRecycler<'a>>,
    pub gate:OutputGate,
}

/// The estimator of a request (`estimate_api::Construct`).
pub const CONSTRUCT:super::estimate_api::Construct=construct;
pub fn construct(setup:EstimateSetup<'_>)->Result<Box<dyn Estimator+'_>,String>{
    let modes=setup.lattice.modes();
    let horizon=longest_time(setup.outputs.times,setup.outputs.horizons);
    let rule=KernelRule{entry:KERNEL_ENTRY,horizon_rate:KERNEL_HORIZON_RATE,horizon,cap:KERNEL_CAP};
    let lattice=ErrorLattice::new(setup.prepared,&modes,rule,CLASSES)?;
    let steady=if setup.outputs.steady{Some(SteadyRecycler::new(setup.prepared,&modes)?)}else{None};
    let gate=OutputGate::new(lattice.n,lattice.nu.clone(),lattice.centre);
    Ok(Box::new(UnifiedEstimate{setup,lattice,steady,gate}))
}

/// The reduced model's slowest relaxation time: 1 / |theta| over the modes
/// outside the kernel that carry weight.
pub fn relaxation_time(values:&[C],coefficients:&[C],kernel:&[bool])->f64{
    values.iter().zip(coefficients).zip(kernel).filter(|((_,a),&k)|!k&&nonzero(**a)).map(|((z,_),_)|1./magnitude(*z)).fold(0.,f64::max)
}

impl Estimator for UnifiedEstimate<'_>{
    fn estimate(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves)->Result<ErrorEstimate,String>{
        let clock=Instant::now();
        let(k,r)=(reduced.basis.len(),reduced.r);
        let groups=self.setup.outputs.groups;let outputs=self.setup.outputs;
        // Ritz vectors over the basis: B = X S (column-major k x r), and the
        // reduced steady state's coordinates c = X u_inf, h = X G u_inf.
        let mut vectors=vec![ZERO;k*r];
        for m in 0..r{for i in 0..k{let mut v=ZERO;for l in 0..r{let x=reduced.x[i*r+l];let s=reduced.vectors[l*r+m];v.re+=x*s.re;v.im+=x*s.im;}vectors[i+k*m]=v;}}
        let coordinates=reduced.steady.map(|u|{
            let gu:Vec<f64>=(0..r).map(|i|(0..r).map(|l|reduced.g[i*r+l]*u[l]).sum()).collect();
            let over=|v:&[f64]|->Vec<C>{(0..k).map(|i|C{re:(0..r).map(|l|reduced.x[i*r+l]*v[l]).sum(),im:0.}).collect()};
            (over(u),over(&gu))});
        // The gate: the outputs' change since the previous checkpoint.
        self.gate.add_columns(reduced.basis);
        let current=self.gate.outputs(&self.setup.prepared.lapack,&vectors,reduced.used_theta,reduced.used_coefficients,outputs.times,outputs.horizons,
            if outputs.steady{coordinates.as_ref().map(|c|&c.0[..])}else{None})?;
        let change=self.gate.change(current,groups);
        let limit=gate::GATE_FACTOR*ledger::POPULATION_TOLERANCE;
        if let Some(change)=change.filter(|&c|c>limit&&self.gate.last_estimate.is_none_or(|e|e>limit)){
            return Ok(ErrorEstimate{trajectory:vec![change;outputs.times.len()],horizons:vec![change;outputs.horizons.len()],
                steady:outputs.steady.then_some(change),work:EstimateWork{milliseconds:clock.elapsed().as_secs_f64()*1e3,..Default::default()}});
        }
        let before=(self.lattice.work.clone(),self.steady.as_ref().map(|x|x.work.clone()).unwrap_or_default());
        self.lattice.add_columns(reduced.basis,reduced.images)?;
        let ritz=RitzSources{vectors:&vectors,values:reduced.used_theta,coefficients:reduced.used_coefficients};
        let zero=vec![0.;self.lattice.n];
        let delta=self.setup.ledger.total();
        let tau=relaxation_time(reduced.used_theta,reduced.used_coefficients,reduced.kernel);
        let fold=|t:f64|delta*t.min(tau);
        let trajectory=self.lattice.populations(&ritz,outputs.times,Readout::Laboratory)?.iter().zip(outputs.times)
            .map(|(p,&t)|metric(p,&zero,groups)+fold(t)).collect();
        let horizons=self.lattice.populations(&ritz,outputs.horizons,Readout::Centre)?.iter().zip(outputs.horizons)
            .map(|(p,&t)|metric(p,&zero,groups)+fold(t)).collect();
        let steady_error=if !outputs.steady{None}else if let(Some((c,h)),Some(recycler))=(coordinates.as_ref(),self.steady.as_mut()){
            let f=steady_source(&mut self.lattice,reduced.basis,reduced.images,c,h)?;
            let e=recycler.error(reduced.basis,reduced.images,&f,steady)?;
            Some(metric(&e,&zero,groups)+delta*tau)
        }else{
            let p=self.lattice.populations(&ritz,&[SEVERAL_KERNELS_HORIZON],Readout::Centre)?;
            Some(metric(&p[0],&zero,groups)+delta*tau)
        };
        let(w,s)=(&self.lattice.work,self.steady.as_ref().map(|x|x.work.clone()).unwrap_or_default());
        let blocks=self.lattice.modes.len();
        let work=EstimateWork{
            block_operations:w.block_operations-before.0.block_operations+w.frame_blocks-before.0.frame_blocks
                +(s.applications-before.1.applications+s.preconditions-before.1.preconditions)*blocks,
            block_solves:w.block_solves-before.0.block_solves,inner_steps:s.inner_steps-before.1.inner_steps,milliseconds:clock.elapsed().as_secs_f64()*1e3};
        let estimate=ErrorEstimate{trajectory,horizons,steady:steady_error,work};
        self.gate.last_estimate=Some(estimate.largest());
        Ok(estimate)
    }
}

/// The steady source f = A W c - W h + B y(W c) on P u Q (its
/// Hermitian-symmetric part), for the reduced steady state's coordinates
/// c = X u_inf and h = X G u_inf.
pub fn steady_source(lattice:&mut ErrorLattice<'_>,basis:&[Vec<C>],images:&[Vec<C>],c:&[C],h:&[C])->Result<Vec<C>,String>{
    let length=lattice.dimension();
    let mut x=vec![ZERO;length];let mut f=vec![ZERO;length];
    for(j,(w,aw))in basis.iter().zip(images).enumerate().take(c.len()){
        for((xv,fv),(&a,&b))in x.iter_mut().zip(f.iter_mut()).zip(w.iter().zip(aw)){mul_add(xv,c[j],a);mul_add(fv,c[j],b);let v=mul(h[j],a);fv.re-=v.re;fv.im-=v.im;}
    }
    let x=lattice.hermitian_part(&x);
    let y=lattice.slave(&x)?;let g=lattice.feedback(&y)?;
    for(a,b)in f.iter_mut().zip(g){add(a,b);}
    Ok(lattice.hermitian_part(&f))
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
        let setup=EstimateSetup{prepared:&p,lattice:&lattice,outputs:Outputs{times:&times,horizons:&horizons,steady:true,groups:&groups},ledger:&ledger,initial:&initial};
        let mut estimator=CONSTRUCT(setup).unwrap();
        let x=[1.,0.,0.,1.];let c0=[1.,0.];let kernel=[false,false];
        let reduced=ReducedView{basis:&basis,images:&images,r:2,x:&x,g:&g,c0:&c0,theta:&theta,vectors:&sr,coefficients:&a,
            used_theta:&theta,used_coefficients:&a,kernel:&kernel,growth_charge:0.,steady:None};
        let e=estimator.estimate(&reduced,&mut NoSolves).unwrap();
        assert_eq!((e.trajectory.len(),e.horizons.len()),(3,1));
        assert!(e.trajectory[0].abs()<1e-12&&e.trajectory[1..].iter().chain(&e.horizons).all(|v|v.is_finite()&&*v>0.)&&e.steady.is_some_and(|v|v.is_finite()),"{e:?}");
        assert!(e.largest().is_finite()&&e.work.block_operations>0);
    }
}
