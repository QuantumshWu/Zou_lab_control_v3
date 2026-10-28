//! The error estimate of the unified pipeline: one error equation for every
//! output f(A) x0 of the reduced model, with three sources.
//!
//! Error equation (exact): with u(t) the reduced model's trajectory, the
//! error of x(t) ~ W u(t) obeys E' = A' E + (A' W - W G) u(t), E(0) = 0, on
//! the full lattice, A' the true operator. Per Ritz mode (b_m = s_m,
//! theta_m, a_m as the outputs use them) the source is a_m e^{theta_m t}
//! times
//! - F_m = (A W - theta_m W) b_m, the Krylov residual;
//! - g_m, the omitted ring R = ring(P u Q) slaved at z = theta_m and fed back
//!   (`ErrorLattice::ring_feedback`);
//! - dA W b_m, the K^(1) rests the lattice operator leaves out
//!   (`LatticeOperator::left_out`). The other fold remainders are reported
//!   only (`ledger`: their measured action is at most 6e-6).
//!
//! - Slow Ritz modes (|theta| < `slow::SLOW`, the kernel included, a != 0)
//!   are exact (`slow`): e_m(t) = a_m (e^{theta_m t} Y_m - e^{tA} Y_m) with
//!   Y_m = (theta_m - A)^-1 (F_m + g_m + dA W b_m), the space's part of F_m in
//!   closed form and the rest by GCRO on one shared pool seeded with the
//!   pipeline's steady solve, the slow Ritz directions deflated. The kernel
//!   mode's Y is the steady error; before it is propagated its projection
//!   on the space loses its kernel coordinate (Y_0 is trace free, so the
//!   true e^{tA} Y_0 decays; the projection's trace would stay forever).
//! - Fast Ritz modes: second-order Schrieffer-Wolff averaging on the near
//!   modes, the centre, its one-step neighbours and the explicit set P
//!   (`kernels`, `blocks`, `lattice`, `propagate`), sources transferred in
//!   from their neighbours at first order; the neighbours' own errors by
//!   the dressed blocks of L0' alone (five lasers, 20 us: without them the
//!   estimate is 0.50-0.54 of the actual error). Krylov residual alone. Only
//!   the canonical member of each mirror pair is transformed and propagated.
//! - Time rule for the trajectory: the slow part runs when T lambda_slow >= 1
//!   at the grid's last time T (lambda_slow the slowest decay rate of the
//!   reduced model); before that the slow modes' far sources have not grown
//!   and the two large terms of the exact form cancel badly: every Ritz mode
//!   goes through the SW part (always taking the slow part: per-time ratio
//!   0.03) but the kernel modes, which always take it. A horizon
//!   always takes the slow part for the slow modes and the fast part for the
//!   others (the complete form): the fast modes' residuals reach the
//!   slow classes of the frame's block, which carry them to T. Without the
//!   fast part a slow mode the space lacks was invisible: the default scene
//!   (probe + tweezer) at T = 100 ms was off by 7.8e-3 with a bound of
//!   7e-13, its slow classes at 5.6e-7 and 8.7e-7 /us outside the space;
//!   with it the bound is 4.8e-4, still short (the pipeline resolves such a
//!   class: `pipeline::missing_classes`). The steady state takes the slow
//!   part alone (the kernel's error).
//!
//! Outputs: the trajectory on the grid (laboratory readout), each horizon T
//! (centre readout: the time average) and the steady state, every one
//! estimated at every checkpoint. Errors are measured with `ledger::metric`.
use super::*;
use super::estimate_api::{EstimateSetup,ErrorEstimate,EstimateWork,ReducedView,SteadySolves};
use super::ledger::{self,metric};
use super::operator::LatticeOperator;

pub mod kernels;
pub mod blocks;
pub mod lattice;
pub mod propagate;
pub mod slow;
pub use kernels::KernelRule;
pub use lattice::ErrorLattice;
pub use propagate::{Readout,RitzSources};
pub use slow::{SlowPool,SlowSolution};

/// Kernel support: the first-order entry floor and the horizon floor on an
/// entry's second-order rate times the longest output time. Measured: 5e-3
/// keeps 0.3% of the beat kernel's entries and reproduces the full kernels for
/// the estimate (5e-2 does not); the reduced model's metric (`frame`) needs
/// more: five lasers + tweezer at 20 us grew 2-19 pseudo-modes from the
/// 12th solve at 5e-3 (accepted at 40 solves), none at 1e-3 (0.6-2% of the
/// entries; accepted at 12), 1e-4 or 1e-5.
pub const KERNEL_ENTRY:f64=1e-3;
pub const KERNEL_HORIZON_RATE:f64=1e-2;
/// Near-resonant entries above this stay couplings: the first-order
/// kernel is no small parameter there. Derived: |X~| = 1 is where the
/// expansion stops being small (measured without the trap: the 1675 rad/us
/// step's kernel has 212 entries above 1, the largest 24, near an optical
/// resonance).
pub const KERNEL_CAP:f64=1.;
/// A mode decays when |Re theta| is above this (1/us). Derived: a slower
/// mode relaxes over more than 1e4 s, 1e4 times the 1 s horizon.
const DECAYING:f64=1e-10;
/// A Ritz value below this is the kernel (theta = 0 exactly in the outputs).
const KERNEL:f64=1e-12;
/// A kernel mode whose source |A W b + ring feedback + dA W b| is below
/// this fraction of |A| |W b| (|A| the largest image of a basis column) is
/// invariant to roundoff: its error is zero. Every further conserved
/// quantity of a drive-free lattice is such a mode, and the bordered system
/// (one trace row) is singular on it. A deliberate margin of about four
/// orders over double-precision roundoff; measured on the drive-free model:
/// solving that mode gave a steady estimate of 1.4e7, skipping it 6e-13.
const INVARIANT:f64=1e-12;

/// The longest output time of a request.
pub fn longest_time(times:&[f64],horizons:&[f64])->f64{times.iter().chain(horizons).fold(0_f64,|a,&b|a.max(b.abs()))}

/// The reduced model of a checkpoint in the estimator's terms: the basis
/// and images, the Ritz vectors S (r x r, column-major, r = basis.len()),
/// the values and coefficients the outputs use.
pub struct Checkpoint<'c>{pub basis:&'c[Vec<C>],pub images:&'c[Vec<C>],pub s:&'c[C],pub theta:&'c[C],pub coefficients:&'c[C]}

/// Population errors of one checkpoint's outputs (before the metric), and
/// what ran.
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
    /// The lattice operator on P u Q when it leaves the K^(1) rests out
    /// (their application, `LatticeOperator::left_out`).
    rests:Option<LatticeOperator<'a>>,
    /// The pipeline's steady-solve directions taken so far.
    seeded:usize,
}

/// The reduced model's left functionals of the Ritz modes `modes` on W^T y:
/// rows of S^-1 (S column-major r x r), so that l_j(W s_i) = delta_ij.
fn left_functionals(lapack:&Lapack,s:&[C],r:usize,modes:&[usize])->Result<Vec<Vec<C>>,String>{
    let mut identity=vec![ZERO;r*r];for i in 0..r{identity[i+r*i]=ONE;}
    let(lu,piv)=lapack.complex_factor(s,r)?;let m=lapack.complex_solve_matrix_factored(&lu,&piv,&identity,r)?;
    Ok(modes.iter().map(|&j|(0..r).map(|i|m[j+r*i]).collect()).collect())
}

impl<'a> UnifiedEstimate<'a>{
    pub fn new(setup:EstimateSetup<'a>)->Result<Self,String>{
        let modes=setup.lattice.modes();
        let lattice=ErrorLattice::new(setup.prepared,&modes,&setup.lattice.explicit,setup.frame)?;
        let rests=if setup.ledger.k1_rest>0.{Some(LatticeOperator::new(setup.prepared,&modes)?)}else{None};
        Ok(Self{setup,lattice,pool:None,rests,seeded:0})
    }
    fn pool(&mut self)->Result<&mut SlowPool,String>{
        if self.pool.is_none(){self.pool=Some(SlowPool::new(self.setup.prepared,&self.lattice,longest_time(self.setup.outputs.times,self.setup.outputs.horizons))?);}
        Ok(self.pool.as_mut().unwrap())
    }
    /// The ring's feedback and the K^(1) rests left out on a field x at the
    /// shift theta (x complex: its Hermitian-symmetric parts).
    fn sources(&mut self,x:&[C],theta:C)->Result<Vec<C>,String>{
        let mut g=self.lattice.ring_feedback(x,theta)?;
        if let Some(op)=&self.rests{
            let(xa,xb)=lattice::split(x,&self.lattice.mirror,self.lattice.n);
            let rest=lattice::join(&op.left_out(&xa)?,&op.left_out(&xb)?);
            for(a,b)in g.iter_mut().zip(rest){add(a,b);}
        }
        Ok(g)
    }
    /// The population errors of a checkpoint's outputs.
    pub fn errors(&mut self,cp:&Checkpoint<'_>,steady:&mut dyn SteadySolves)->Result<Errors,String>{
        let(k,r)=(cp.basis.len(),cp.theta.len());let outputs=self.setup.outputs;
        let mut ms=[0.;5];let clock=Instant::now();
        // The fast part's columns (the trajectory and the horizons read the
        // fast part).
        if !(outputs.times.is_empty()&&outputs.horizons.is_empty()){self.lattice.add_columns(cp.basis,cp.images)?;}
        ms[0]=clock.elapsed().as_secs_f64()*1e3;
        let b=cp.s;
        // The slow modes and which outputs use them. A projected growing mode
        // (theta = a = 0) is never among them, the kernel included. At an
        // output time T a decaying mode slower than 1/T with a coefficient
        // below the tolerance stays in the SW part and holds no other back
        // (`set`); the slowest decay rate of the others decides whether the
        // slow part runs for the trajectory.
        let slow:Vec<usize>=(0..r).filter(|&m|magnitude(cp.theta[m])<slow::SLOW&&nonzero(cp.coefficients[m])).collect();
        let kernel:Vec<usize>=slow.iter().copied().filter(|&m|magnitude(cp.theta[m])<KERNEL).collect();
        let negligible=|m:usize,t:f64|t*cp.theta[m].re.abs()<1.&&magnitude(cp.coefficients[m])<ledger::POPULATION_TOLERANCE&&!kernel.contains(&m);
        let rate=|t:f64|{let l=(0..r).filter(|&m|cp.theta[m].re.abs()>DECAYING&&nonzero(cp.coefficients[m])&&!negligible(m,t)).map(|m|cp.theta[m].re.abs()).fold(f64::INFINITY,f64::min);
            if l.is_finite(){l}else{0.}};
        let set=|t:f64|->Vec<usize>{slow.iter().copied().filter(|&m|!negligible(m,t)).collect()};
        let last=outputs.times.last().copied().unwrap_or(0.);
        let lambda_slow=rate(last);
        let grid=outputs.times.last().is_some_and(|&t|t*rate(t)>=1.);
        // The kernel modes first, then every slow mode an output takes: every
        // horizon, the grid when it runs the slow part.
        let mut taken:Vec<usize>=Vec::new();
        for t in outputs.horizons.iter().copied().chain(grid.then_some(last)){for m in set(t){if !taken.contains(&m){taken.push(m);}}}
        // Those an output may leave out last, slowest last: every output time
        // takes a prefix of the solutions.
        taken.sort_by(|&p,&q|{let key=|m:usize|(magnitude(cp.coefficients[m])<ledger::POPULATION_TOLERANCE,-cp.theta[m].re.abs());key(p).partial_cmp(&key(q)).unwrap()});
        let needed:Vec<usize>=if !taken.is_empty(){kernel.iter().chain(taken.iter().filter(|m|!kernel.contains(m))).copied().collect()}
            else if outputs.steady||!outputs.times.is_empty(){kernel.clone()}else{Vec::new()};
        // Every slow mode (the kernel's first) deflates the pool once an
        // output takes one.
        let deflated:Vec<usize>=if !taken.is_empty(){kernel.iter().chain(slow.iter().filter(|m|!kernel.contains(m))).copied().collect()}else{needed.clone()};
        let(mut solutions,mut solved,mut invariant)=(Vec::new(),Vec::new(),Vec::new());
        let scale=cp.images.iter().map(|a|lattice::norm(a)).fold(0_f64,f64::max);
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
            if deflated.iter().any(|&m|magnitude(cp.theta[m])>=KERNEL){
                let left=left_functionals(&self.setup.prepared.lapack,cp.s,r,&deflated)?;
                let(xs,axs):(Vec<Vec<C>>,Vec<Vec<C>>)=fields.iter().cloned().unzip();
                let thetas:Vec<C>=deflated.iter().map(|&m|cp.theta[m]).collect();let bs:Vec<Vec<C>>=deflated.iter().map(|&m|b[k*m..k*(m+1)].to_vec()).collect();
                self.pool.as_mut().unwrap().deflate(&deflated,&thetas,&bs,&left,&xs,&axs)?;
            }
            // In the order of `needed` (an output time takes a prefix).
            for &m in &needed{let(x,ax)=&fields[deflated.iter().position(|&d|d==m).ok_or("Estimator: a needed slow mode outside the deflated ones")?];
                let theta=cp.theta[m];
                let g=self.sources(x,theta)?;
                if kernel.contains(&m){let f:Vec<C>=ax.iter().zip(&g).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
                    if lattice::norm(&f)<=INVARIANT*scale*lattice::norm(x){invariant.push(m);continue;}}
                solutions.push(self.pool.as_mut().unwrap().mode(&b[k*m..k*(m+1)],ax,&g,theta,cp.coefficients[m],steady)?);solved.push(m);
            }
            ms[2]=clock.elapsed().as_secs_f64()*1e3;
        }
        let sum=|a:Vec<Vec<f64>>,c:Vec<Vec<f64>>|->Vec<Vec<f64>>{a.into_iter().zip(c).map(|(x,y)|x.iter().zip(&y).map(|(p,q)|p+q).collect()).collect()};
        // The solved slow modes an output time takes (a prefix).
        let taking=|t:f64|->usize{let s=set(t);solved.iter().take_while(|m|s.contains(m)).count()};
        let n=self.lattice.n;
        // The trajectory: the fast part without the modes the slow part takes
        // (and the invariant kernel modes).
        // The kernel modes (first among the solutions) always take the slow
        // part: their error a_0 (Y_0 - e^{tA} Y_0) has no large cancelling
        // terms, and it grows to the steady error, the omitted ring's, which
        // the fast part cannot see (a spurious slow Ritz value, or T just
        // below 1 / lambda_slow, otherwise leaves it out: per-time ratios
        // 0.05 on four lasers at 10 ms, 0.29 on four lasers without the trap
        // at 20 us).
        let slow_grid=grid&&!solutions.is_empty();
        let kernels=solved.iter().take_while(|m|kernel.contains(m)).count();
        let trajectory=if outputs.times.is_empty(){Vec::new()}else{
            let count=if slow_grid{taking(last).max(kernels)}else{kernels};
            let clock=Instant::now();
            let skip:Vec<usize>=solved[..count].iter().chain(&invariant).copied().collect();
            let coefficients:Vec<C>=(0..r).map(|m|if skip.contains(&m){ZERO}else{cp.coefficients[m]}).collect();
            let fast=self.lattice.populations(&RitzSources{vectors:b,values:cp.theta,coefficients:&coefficients},outputs.times,Readout::Laboratory)?;
            ms[3]=clock.elapsed().as_secs_f64()*1e3;
            if count>0{
                let clock=Instant::now();
                let s=self.pool.as_mut().unwrap().populations(&self.lattice,cp.basis,cp.s,cp.theta,&kernel,&solutions[..count],outputs.times,Readout::Laboratory)?;
                ms[4]+=clock.elapsed().as_secs_f64()*1e3;sum(fast,s)
            }else{fast}
        };
        let clock=Instant::now();
        let mut horizons=Vec::with_capacity(outputs.horizons.len());
        for &t in outputs.horizons{
            let count=taking(t);
            let skip:Vec<usize>=solved[..count].iter().chain(&invariant).copied().collect();
            let coefficients:Vec<C>=(0..r).map(|m|if skip.contains(&m){ZERO}else{cp.coefficients[m]}).collect();
            let fast=self.lattice.populations(&RitzSources{vectors:b,values:cp.theta,coefficients:&coefficients},&[t],Readout::Centre)?.remove(0);
            horizons.push(if count==0{fast}else{
                let slow=self.pool.as_mut().unwrap().populations(&self.lattice,cp.basis,cp.s,cp.theta,&kernel,&solutions[..count],&[t],Readout::Centre)?.remove(0);
                fast.iter().zip(&slow).map(|(a,b)|a+b).collect()});
        }
        ms[4]+=clock.elapsed().as_secs_f64()*1e3;
        let steady=outputs.steady.then(||{
            let mut e=vec![0.;n];
            if let Some(pool)=self.pool.as_ref(){for sol in solutions.iter().filter(|s|magnitude(s.theta)<KERNEL){for(a,v)in e.iter_mut().zip(pool.steady_populations(cp.basis,sol)){*a+=v;}}}
            e});
        Ok(Errors{trajectory,horizons,steady,slow_modes:solutions.len(),slow_trajectory:slow_grid,lambda_slow,milliseconds:ms})
    }
    /// The work counters so far: the fast part's and the pool's.
    fn counters(&self)->(lattice::ErrorWork,slow::PoolWork){
        (self.lattice.work.clone(),self.pool.as_ref().map(|p|p.work.clone()).unwrap_or_default())
    }
    /// The work since `before`: block operations (the pool's products with Q
    /// in block units, 4 real multiply-adds per complex one, d^2 per block),
    /// ring solves.
    fn work_since(&self,before:&(lattice::ErrorWork,slow::PoolWork))->(usize,usize){
        let(w,p)=self.counters();
        let blocks=self.lattice.modes.len();
        let qr=((p.qr_macs-before.1.qr_macs)/(4.*(self.lattice.d*self.lattice.d)as f64)).round()as usize;
        (w.block_operations-before.0.block_operations+w.frame_blocks-before.0.frame_blocks+p.frame_blocks-before.1.frame_blocks
            +(p.applications-before.1.applications+p.preconditions-before.1.preconditions)*blocks+qr,
         w.block_solves-before.0.block_solves)
    }
    /// The estimated errors of a checkpoint's outputs; `steady` is the
    /// pipeline's own lattice (its operator, steady preconditioner and
    /// steady-solve directions).
    pub fn estimate(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves)->Result<ErrorEstimate,String>{
        let clock=Instant::now();let r=reduced.r;
        let groups=self.setup.outputs.groups;
        let before=self.counters();
        // S column-major.
        let s:Vec<C>=(0..r*r).map(|i|reduced.vectors[(i%r)*r+i/r]).collect();
        // The pipeline's steady-solve directions not taken yet seed the pool.
        let(directions,images)=steady.directions();
        let seeded=self.seeded;
        if directions.len()>seeded{self.pool()?.seed(&directions[seeded..],&images[seeded..])?;self.seeded=directions.len();}
        let cp=Checkpoint{basis:reduced.basis,images:reduced.images,s:&s,theta:reduced.used_theta,coefficients:reduced.used_coefficients};
        let errors=self.errors(&cp,steady)?;
        let zero=vec![0.;self.lattice.n];let size=|p:&Vec<f64>|metric(p,&zero,groups);
        let(block_operations,block_solves)=self.work_since(&before);
        Ok(ErrorEstimate{trajectory:errors.trajectory.iter().map(size).collect(),horizons:errors.horizons.iter().map(size).collect(),
            steady:errors.steady.as_ref().map(size),
            work:EstimateWork{block_operations,block_solves,parts_ms:errors.milliseconds,milliseconds:clock.elapsed().as_secs_f64()*1e3,
                slow_modes:errors.slow_modes,slow_trajectory:errors.slow_trajectory,lambda_slow:errors.lambda_slow}})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::estimate_api::{LatticeSets,Outputs};
    use super::super::operator::LatticeOperator;
    use super::super::tests::{fixture,prepared};
    use super::super::frame::SwFrame;
    #[test]
    fn estimator_runs_through_the_pipeline_interface(){
        // A two-column space (x0 and its image's new direction) on the
        // fixture's first ball: one estimate per output, zero at t = 0,
        // finite and positive after it.
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
        let mut estimator=UnifiedEstimate::new(setup).unwrap();
        let kernel=[false,false];
        let reduced=ReducedView{basis:&basis,images:&images,r:2,vectors:&sr,used_theta:&theta,used_coefficients:&a,kernel:&kernel};
        let mut solver=super::super::poles::Lattice::new(&p,lattice.clone()).unwrap();
        let e=estimator.estimate(&reduced,&mut solver).unwrap();
        assert_eq!((e.trajectory.len(),e.horizons.len()),(3,1));
        assert!(e.trajectory[0].abs()<1e-12&&e.trajectory[1..].iter().all(|v|v.is_finite()&&*v>0.)&&e.horizons.iter().all(|v|v.is_finite()),"{e:?}");
        assert!(e.largest().is_finite()&&e.work.block_operations>0);
    }
}
