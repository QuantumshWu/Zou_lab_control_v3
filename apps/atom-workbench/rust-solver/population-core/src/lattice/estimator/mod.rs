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
//! - dA W b_m, what the lattice operator leaves out of the second-order far
//!   shell (`LatticeOperator::left_out`): the exact far shell less the stored
//!   fold, i.e. the static expansion's remainder in nu, the series
//!   truncation, the entries below the storage floor, the non-structural
//!   rests, the K^(1) rests it drops and the cross folds below the floor.
//!   The higher-order far-shell paths are reported only (`ledger`: their
//!   measured action is at most 6e-6).
//!
//! - Slow Ritz modes (|theta| < `slow::SLOW`, the kernel included, not
//!   projected out) are exact (`slow`): e_m(t) = a_m (e^{theta_m t} Y_m - e^{tA} Y_m) with
//!   Y_m = (theta_m - A)^-1 (F_m + g_m + dA W b_m), the space's part of F_m in
//!   closed form and the rest by GCRO on one shared pool seeded with the
//!   pipeline's steady solve, the slow Ritz directions deflated. The kernel
//!   mode's Y is the steady error, trace free like its projection on the
//!   space (the basis carries the trace in its first column alone, `space`).
//!   Every kernel mode is solved, whatever the size of its source: a skip of
//!   the modes whose source fell below 1e-12 of the largest image (a scale
//!   the optical-frame frequencies set) zeroed the steady estimate on the B
//!   basis (ex_dark, ex_open, weak_probe and fb_far_only at infinity: bounds
//!   of 4e-16 to 7e-16 against errors of 2.2e-9 to 8.7e-7); with the trace
//!   carried by construction the further conserved quantities of a
//!   drive-free lattice solve without the blow-up that once called for it
//!   (eight kernel modes: the same estimate, 3.9e-6, with and without it).
//! - A slow mode's own residual, charged apart: for a Ritz pair (theta, x)
//!   with r = (A - theta) x, e^{tA} x - e^{theta t} x = int_0^t
//!   e^{(t-s)A} r e^{theta s} ds exactly, and the slow part's in-space -W b,
//!   propagated by the reduced model, cancels that term exactly: the slow
//!   part sees the ring's and the K^(1) rests' sources only, an inaccurate
//!   slow pair not at all. Every non-kernel slow mode the slow part takes
//!   therefore also goes through the SW part (`propagate`), as a metric of
//!   its own added to each output's (summed into the same signed error it
//!   partly cancelled: four lasers at 10 ms, per-time ratio 0.47 against
//!   0.77 without the charge, 1.85 apart). A source of exactly zero and one
//!   at roundoff now give the same estimate: a complete lattice (its
//!   truncated charges promoted, the K^(1) rest kept) left the slow part a
//!   ring source of 1e-26, and its trajectory estimate fell to 7e-27
//!   against an error of 1.6e-3.
//! - Fast Ritz modes: second-order Schrieffer-Wolff averaging on the near
//!   modes, the centre, its one-step neighbours and the explicit set P
//!   (`kernels`, `blocks`, `lattice`, `propagate`), sources transferred in
//!   from their neighbours at first order; the neighbours' own errors by
//!   the dressed blocks of L0' alone (five lasers, 20 us: without them the
//!   estimate is 0.50-0.54 of the actual error). Krylov residual alone. Only
//!   the canonical member of each mirror pair is transformed and propagated.
//! - Time rule for the trajectory: the slow part runs when T lambda_slow >= 1
//!   at the grid's last time T (lambda_slow the slowest decay rate of the
//!   reduced model); before that every Ritz mode goes through the SW part but
//!   the kernel modes, which always take the slow part. Measured without the
//!   rule on the B basis: the 4- and 8-solve checkpoints of four and five
//!   lasers (with and without the trap) at 20 us improve (worst per-time ratio
//!   0.29-0.72 -> 0.93-1.29, the accepted checkpoints unchanged), but the
//!   three-laser D2 scenes' trajectories with p(1 s) are lost: ex_three's
//!   bound 2.7e-3 -> 7.6e-2 (error 4.1e-4, a partial result), b1mG_three
//!   3.8e-3 -> 1.5e-2, b0_three and both_tweezer at 44 and 24 solves instead
//!   of 16 and 8. The cause is the slow modes' own residuals charged apart
//!   (above): split from the other modes', the Krylov residual loses its
//!   cancellation across modes. The slow part's propagation of Y is not it:
//!   with Y's part outside the space on the frame's second-order block the
//!   four scenes stay lost. Summed signed with the other modes' instead,
//!   ex_three's 4- and 8-solve bounds fall from 0.16 and 8.1e-2 to 1.7e-2
//!   and 3.2e-3 and it converges at 8 solves, both_tweezer at 12; but a slow
//!   mode's own residual responds through (theta - A)^-1 F = -W b, a
//!   near-resonance the SW part does not hold late (five lasers at 10 ms,
//!   4 solves: worst per-time ratio 1.31 -> 0.80, 14 samples under near
//!   0.45 ms), and b0_three and b1mG_three stop on the plateau at 8.4e-3 and
//!   7.6e-3, their slow parts 6.9e-3 and 6.1e-3 against errors of 1.7e-3 and
//!   1.9e-3. The rule stays until a slow mode's own residual has one model
//!   that holds early and late.
//!   On the trace-corrected basis it had a second reason: the corrected
//!   kernel vector and slow pairs were no Galerkin vectors, and always taking
//!   the slow part gave per-time ratios down to 0.03. A horizon always takes
//!   the slow part for the slow modes and the fast part for the others (the
//!   complete form): the fast modes' residuals reach the slow classes of the
//!   frame's block, which carry them to T. Without the
//!   fast part a slow mode the space lacks was invisible: the default scene
//!   (probe + tweezer) at T = 100 ms was off by 7.8e-3 with a bound of
//!   7e-13, its slow classes at 5.6e-7 and 8.7e-7 /us outside the space;
//!   with it the bound is 4.8e-4, still short; the slow modes' own
//!   residuals (above) close the gap (the default scene at 10 ms: rejected
//!   at 4 solves, accepted at 8 with a bound of 2.1e-5 against an error of
//!   1.8e-5). The steady state takes the slow part alone (the kernel's
//!   error).
//!
//! Outputs: the trajectory on the grid (laboratory readout), each horizon T
//! (centre readout: the time average) and the steady state, every one
//! estimated at every checkpoint. Errors are measured with `ledger::metric`.
use super::*;
use super::estimate_api::{EstimateSetup,ErrorEstimate,EstimateWork,Outputs,ReducedView,SteadySolves};
use super::ledger::{self,metric};

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
/// A source of the slow part taken alone (`UnifiedEstimate::estimate_alone`,
/// the completion's): what the lattice operator leaves out of the
/// second-order far shell (`LatticeOperator::left_out`, the K^(1) rests it
/// drops among it), or the omitted ring's feedback.
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Source{K1,Ring}
/// Rows of the grid per task of the time GEMMs for `per_time` multiply-adds
/// per time: at most four tasks per reference worker, at least
/// MINIMUM_WORK_PER_TASK each (fixed: the chunking does not depend on the
/// thread count).
pub fn time_chunk(times:usize,per_time:usize)->usize{
    times.div_ceil((4*crate::parallel::REFERENCE_WORKERS).min(times).min(per_time*times/crate::parallel::MINIMUM_WORK_PER_TASK).max(1))
}

/// The reduced model of a checkpoint in the estimator's terms: the basis
/// and images, the Ritz vectors S (r x r, column-major, r = basis.len()),
/// the values and coefficients the outputs use, and per mode the reduced
/// model's kernel and projected-growing flags.
pub struct RitzModel<'c>{pub basis:&'c[Vec<C>],pub images:&'c[Vec<C>],pub s:&'c[C],pub theta:&'c[C],pub coefficients:&'c[C],
    pub kernel:&'c[bool],pub projected:&'c[bool]}
impl<'c> RitzModel<'c>{
    /// A checkpoint's reduced model, its Ritz vectors `s` column-major.
    fn new(reduced:&ReducedView<'c>,s:&'c[C])->Self{
        Self{basis:reduced.basis,images:reduced.images,s,theta:reduced.used_theta,coefficients:reduced.used_coefficients,
            kernel:reduced.kernel,projected:reduced.projected}
    }
}
/// The Ritz vectors S of a reduced model, column-major.
fn column_major(reduced:&ReducedView<'_>)->Vec<C>{let r=reduced.r;(0..r*r).map(|i|reduced.vectors[(i%r)*r+i/r]).collect()}
/// The Ritz vector x = W b_m of mode m and its image A x (from the
/// pipeline's images), on fields of `length`.
fn ritz_field(cp:&RitzModel<'_>,m:usize,length:usize)->(Vec<C>,Vec<C>){
    let k=cp.basis.len();let(mut x,mut ax)=(vec![ZERO;length],vec![ZERO;length]);
    for(j,&c)in cp.s[k*m..k*(m+1)].iter().enumerate(){if !nonzero(c){continue;}
        for((xv,av),(&w,&aw))in x.iter_mut().zip(ax.iter_mut()).zip(cp.basis[j].iter().zip(&cp.images[j])){mul_add(xv,c,w);mul_add(av,c,aw);}}
    (x,ax)
}

/// The slow modes of a checkpoint and which outputs take them. A projected
/// growing mode is never among them; the kernel is. At an output time T a
/// decaying mode slower than 1/T with a coefficient below the tolerance
/// stays in the SW part and holds no other back (`at`); the slowest decay
/// rate of the others decides whether the slow part runs for the
/// trajectory (`grid`).
struct SlowModes{
    /// The slow modes, and the kernel's among them.
    slow:Vec<usize>,kernel:Vec<usize>,
    /// The modes solved, the kernel's first and then in the order an output
    /// time takes a prefix of them, and the modes deflating the pool.
    needed:Vec<usize>,deflated:Vec<usize>,
    /// The grid's last time and whether the trajectory runs the slow part
    /// there (the time rule).
    last:f64,grid:bool,
}
impl SlowModes{
    fn new(cp:&RitzModel<'_>,outputs:&Outputs<'_>)->Self{
        let r=cp.theta.len();
        let slow:Vec<usize>=(0..r).filter(|&m|magnitude(cp.theta[m])<slow::SLOW&&!cp.projected[m]).collect();
        let kernel:Vec<usize>=slow.iter().copied().filter(|&m|cp.kernel[m]).collect();
        let last=outputs.times.last().copied().unwrap_or(0.);
        let modes=Self{slow,kernel,needed:Vec::new(),deflated:Vec::new(),last,grid:false};
        let rate=|t:f64|{let l=(0..r).filter(|&m|cp.theta[m].re.abs()>=super::reduced::KERNEL&&!cp.projected[m]&&!modes.negligible(cp,m,t)).map(|m|cp.theta[m].re.abs()).fold(f64::INFINITY,f64::min);
            if l.is_finite(){l}else{0.}};
        let grid=outputs.times.last().is_some_and(|&t|t*rate(t)>=1.);
        // The kernel modes first, then every slow mode an output takes: every
        // horizon, the grid when it runs the slow part.
        let mut taken:Vec<usize>=Vec::new();
        for t in outputs.horizons.iter().copied().chain(grid.then_some(last)){for m in modes.at(cp,t){if !taken.contains(&m){taken.push(m);}}}
        // Those an output may leave out last, slowest last: every output time
        // takes a prefix of the solutions.
        let key=|m:usize|(magnitude(cp.coefficients[m])<ledger::POPULATION_TOLERANCE,-cp.theta[m].re.abs());
        taken.sort_by(|&p,&q|{let(a,b)=(key(p),key(q));a.0.cmp(&b.0).then(a.1.total_cmp(&b.1))});
        let kernel=&modes.kernel;
        let needed:Vec<usize>=if !taken.is_empty(){kernel.iter().chain(taken.iter().filter(|m|!kernel.contains(m))).copied().collect()}
            else if !outputs.times.is_empty(){kernel.clone()}else{Vec::new()};
        // Every slow mode (the kernel's first) deflates the pool once an
        // output takes one.
        let deflated:Vec<usize>=if !taken.is_empty(){kernel.iter().chain(modes.slow.iter().filter(|m|!kernel.contains(m))).copied().collect()}else{needed.clone()};
        Self{needed,deflated,grid,..modes}
    }
    /// A decaying mode slower than 1/t with a coefficient below the
    /// tolerance, outside the kernel.
    fn negligible(&self,cp:&RitzModel<'_>,m:usize,t:f64)->bool{
        t*cp.theta[m].re.abs()<1.&&magnitude(cp.coefficients[m])<ledger::POPULATION_TOLERANCE&&!self.kernel.contains(&m)
    }
    /// The slow modes an output time t takes.
    fn at(&self,cp:&RitzModel<'_>,t:f64)->Vec<usize>{self.slow.iter().copied().filter(|&m|!self.negligible(cp,m,t)).collect()}
}

/// Population errors of one checkpoint's outputs (before the metric), and
/// what ran (the slow modes' part of the work).
#[derive(Clone,Debug,Default)]
struct Errors{
    trajectory:Vec<Vec<f64>>,horizons:Vec<Vec<f64>>,
    /// The SW part of the non-kernel slow modes the slow part takes (their
    /// own residuals), per trajectory row and horizon: charged apart; and the
    /// slow part alone (rows per grid time and per horizon; empty when it did
    /// not run), what a completed lattice could remove (`pipeline`).
    own_trajectory:Vec<Vec<f64>>,own_horizons:Vec<Vec<f64>>,slow_trajectory:Vec<Vec<f64>>,slow_horizons:Vec<Vec<f64>>,
    work:EstimateWork,
}

pub struct UnifiedEstimate<'a>{
    setup:EstimateSetup<'a>,
    lattice:ErrorLattice<'a>,
    pub pool:Option<SlowPool>,
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
        let lattice=ErrorLattice::new(setup.prepared,&modes,&setup.lattice.explicit,setup.frame,setup.keep_k1)?;
        Ok(Self{setup,lattice,pool:None,seeded:0})
    }
    fn pool(&mut self)->Result<&mut SlowPool,String>{
        if self.pool.is_none(){self.pool=Some(SlowPool::new(self.setup.prepared,&self.lattice,self.setup.frame.horizon)?);}
        Ok(self.pool.as_mut().unwrap())
    }
    /// The slow modes' sources on a field x at the shift theta (x complex:
    /// its Hermitian-symmetric parts): the ring's feedback and what the
    /// pipeline's operator leaves out of the second-order far shell (when it
    /// leaves anything out), or one of them `alone`.
    fn sources(&mut self,x:&[C],theta:C,steady:&dyn SteadySolves,alone:Option<Source>)->Result<Vec<C>,String>{
        let mut g=if alone==Some(Source::K1){vec![ZERO;x.len()]}else{self.lattice.ring_feedback(x,theta)?};
        if alone==Some(Source::Ring){return Ok(g);}
        let(xa,xb)=lattice::split(x,&self.lattice.mirror,self.lattice.n);
        if let(Some(a),Some(b))=(steady.left_out(&xa)?,steady.left_out(&xb)?){
            let rest=lattice::join(&a,&b);
            for(a,b)in g.iter_mut().zip(rest){add(a,b);}
        }
        Ok(g)
    }
    /// The population errors of a checkpoint's outputs; with a source
    /// `alone` those of it alone, through the slow modes (no Krylov residual,
    /// no fast part).
    fn errors(&mut self,cp:&RitzModel<'_>,steady:&mut dyn SteadySolves,alone:Option<Source>)->Result<Errors,String>{
        let(k,r)=(cp.basis.len(),cp.theta.len());let outputs=self.setup.outputs;let single=alone.is_some();
        // The fast part's columns.
        self.lattice.add_columns(cp.basis,cp.images)?;
        let b=cp.s;
        let modes=SlowModes::new(cp,&outputs);let(kernel,deflated)=(&modes.kernel,&modes.deflated);
        let(mut solutions,mut solved)=(Vec::new(),Vec::new());
        if !deflated.is_empty(){
            self.pool()?.add_columns(cp.basis)?;
            let length=self.lattice.dimension();
            // x = W b_m and A x per slow mode. Away from the kernel the solves
            // deflate every slow Ritz mode (`SlowPool::mode`).
            let fields:Vec<(Vec<C>,Vec<C>)>=deflated.iter().map(|&m|ritz_field(cp,m,length)).collect();
            if deflated.iter().any(|&m|!cp.kernel[m]){
                let left=left_functionals(&self.setup.prepared.lapack,cp.s,r,deflated)?;
                let(xs,axs):(Vec<Vec<C>>,Vec<Vec<C>>)=fields.iter().cloned().unzip();
                let thetas:Vec<C>=deflated.iter().map(|&m|cp.theta[m]).collect();let bs:Vec<Vec<C>>=deflated.iter().map(|&m|b[k*m..k*(m+1)].to_vec()).collect();
                self.pool.as_mut().unwrap().deflate(deflated,&thetas,&bs,&left,&xs,&axs)?;
            }
            // In the order of `needed` (an output time takes a prefix).
            for &m in &modes.needed{let(x,ax)=&fields[deflated.iter().position(|&d|d==m).ok_or("Estimator: a needed slow mode outside the deflated ones")?];
                let theta=cp.theta[m];
                let g=self.sources(x,theta,steady,alone)?;
                // One source alone: the mode's own Krylov residual is none.
                let(zb,zax)=(vec![ZERO;k],vec![ZERO;ax.len()]);
                let(bm,axm):(&[C],&[C])=if single{(&zb,&zax)}else{(&b[k*m..k*(m+1)],ax)};
                solutions.push(self.pool.as_mut().unwrap().mode(bm,axm,&g,theta,cp.coefficients[m],kernel.contains(&m),steady)?);solved.push(m);
            }
        }
        let sum=|a:Vec<Vec<f64>>,c:Vec<Vec<f64>>|->Vec<Vec<f64>>{a.into_iter().zip(c).map(|(x,y)|x.iter().zip(&y).map(|(p,q)|p+q).collect()).collect()};
        // The solved slow modes an output time takes (a prefix).
        let taking=|t:f64|->usize{let s=modes.at(cp,t);solved.iter().take_while(|m|s.contains(m)).count()};
        let n=self.lattice.n;
        // The trajectory: the fast part without the modes the slow part takes;
        // none with one source alone. The kernel modes come first among the
        // solutions: their error a_0 (Y_0 - e^{tA} Y_0) grows to the steady
        // error, the omitted ring's, which the fast part cannot see.
        let kernels=solved.iter().take_while(|m|kernel.contains(m)).count();
        // The non-kernel slow modes among the first `count` solutions, for
        // their own residuals' SW part (charged apart); none with one source
        // alone.
        let own=|count:usize|->Vec<C>{(0..r).map(|m|if solved[kernels.min(count)..count].contains(&m){cp.coefficients[m]}else{ZERO}).collect()};
        let(mut own_trajectory,mut slow_trajectory,mut slow_horizons)=(Vec::new(),Vec::new(),Vec::new());
        let trajectory=if outputs.times.is_empty(){Vec::new()}else{
            let count=if modes.grid&&!solutions.is_empty(){taking(modes.last).max(kernels)}else{kernels};
            let fast=if single{vec![vec![0.;n];outputs.times.len()]}else{
                let coefficients:Vec<C>=(0..r).map(|m|if solved[..count].contains(&m){ZERO}else{cp.coefficients[m]}).collect();
                self.lattice.populations(&RitzSources{vectors:b,values:cp.theta,coefficients:&coefficients},outputs.times,Readout::Laboratory)?};
            if count>kernels&&!single{own_trajectory=self.lattice.populations(&RitzSources{vectors:b,values:cp.theta,coefficients:&own(count)},outputs.times,Readout::Laboratory)?;}
            if count>0{
                let s=self.pool.as_mut().unwrap().populations(&self.lattice,cp.basis,cp.s,cp.theta,&solutions[..count],outputs.times,Readout::Laboratory)?;
                slow_trajectory=s.clone();
                sum(fast,s)
            }else{fast}
        };
        let mut horizons=Vec::with_capacity(outputs.horizons.len());let mut own_horizons=Vec::with_capacity(outputs.horizons.len());
        for &t in outputs.horizons{
            let count=taking(t);
            let fast=if single{vec![0.;n]}else{
                let coefficients:Vec<C>=(0..r).map(|m|if solved[..count].contains(&m){ZERO}else{cp.coefficients[m]}).collect();
                self.lattice.populations(&RitzSources{vectors:b,values:cp.theta,coefficients:&coefficients},&[t],Readout::Centre)?.remove(0)};
            own_horizons.push(if count>kernels&&!single{self.lattice.populations(&RitzSources{vectors:b,values:cp.theta,coefficients:&own(count)},&[t],Readout::Centre)?.remove(0)}else{vec![0.;n]});
            horizons.push(if count==0{slow_horizons.push(vec![0.;n]);fast}else{
                let slow=self.pool.as_mut().unwrap().populations(&self.lattice,cp.basis,cp.s,cp.theta,&solutions[..count],&[t],Readout::Centre)?.remove(0);
                let total=fast.iter().zip(&slow).map(|(a,b)|a+b).collect();slow_horizons.push(slow);total});
        }
        Ok(Errors{trajectory,horizons,own_trajectory,own_horizons,slow_trajectory,slow_horizons,work:EstimateWork{slow_modes:solutions.len(),..Default::default()}})
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
    pub fn estimate(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves)->Result<ErrorEstimate,String>{self.estimate_with(reduced,steady,None)}
    /// The estimate with one source of the slow part alone, through the slow
    /// modes (no Krylov residual, no fast part): what completing that source
    /// could remove (`pipeline`'s completion). The pool keeps the directions
    /// these solves add.
    pub fn estimate_alone(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves,source:Source)->Result<ErrorEstimate,String>{self.estimate_with(reduced,steady,Some(source))}
    fn estimate_with(&mut self,reduced:&ReducedView<'_>,steady:&mut dyn SteadySolves,alone:Option<Source>)->Result<ErrorEstimate,String>{
        let clock=Instant::now();
        let groups=self.setup.outputs.groups;
        let before=self.counters();
        let s=column_major(reduced);
        // The pipeline's steady-solve directions not taken yet seed the pool.
        let(directions,images)=steady.directions();
        let seeded=self.seeded;
        if directions.len()>seeded{self.pool()?.seed(&directions[seeded..],&images[seeded..])?;self.seeded=directions.len();}
        let cp=RitzModel::new(reduced,&s);
        let mut errors=self.errors(&cp,steady,alone)?;
        let zero=vec![0.;self.lattice.n];let size=|p:&Vec<f64>|metric(p,&zero,groups);
        (errors.work.block_operations,errors.work.block_solves)=self.work_since(&before);errors.work.milliseconds=clock.elapsed().as_secs_f64()*1e3;
        // Each output's error, and its slow modes' own residuals charged apart.
        let own=|rows:&[Vec<f64>],i:usize|rows.get(i).map_or(0.,|p|size(p));
        Ok(ErrorEstimate{trajectory:errors.trajectory.iter().enumerate().map(|(i,p)|size(p)+own(&errors.own_trajectory,i)).collect(),
            horizons:errors.horizons.iter().enumerate().map(|(i,p)|size(p)+own(&errors.own_horizons,i)).collect(),
            slow_trajectory:if errors.slow_trajectory.is_empty(){vec![0.;errors.trajectory.len()]}else{errors.slow_trajectory.iter().map(size).collect()},
            slow_horizons:errors.slow_horizons.iter().map(size).collect(),work:errors.work})
    }
    /// The ring attribution of a checkpoint (`pipeline`'s completion): per
    /// ring charge (the error lattice's ring order), the sum over the slow
    /// modes the estimate solves (the kernel's included) of |a_m| times the
    /// feedback of the charge's slaved amplitude, |B y_r|
    /// (`ErrorLattice::ring_attribution`): the ring's feedback alone, no solve.
    pub fn ring_attribution(&mut self,reduced:&ReducedView<'_>)->Result<Vec<(Charge,f64)>,String>{
        let s=column_major(reduced);let cp=RitzModel::new(reduced,&s);
        let modes=SlowModes::new(&cp,&self.setup.outputs);let length=self.lattice.dimension();
        let mut shares=vec![0.;self.lattice.ring.len()];
        for &m in &modes.needed{
            let(x,_)=ritz_field(&cp,m,length);let a=magnitude(cp.coefficients[m]);
            for(t,f)in shares.iter_mut().zip(self.lattice.ring_attribution(&x,cp.theta[m])?){*t+=a*f;}
        }
        Ok(self.lattice.ring.iter().cloned().zip(shares).collect())
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
        let mut op=LatticeOperator::new(&p,&modes).unwrap();let d=16;let size=op.dimension();
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
        let lattice=LatticeSets{explicit:modes.clone(),ring:Vec::new()};
        let ledger=op.ledger().unwrap();
        let times=[0.,0.5,1.];let horizons=[10.];let groups=vec![vec![0,1]];
        let frame=SwFrame::new(&p,10.);
        let setup=EstimateSetup{prepared:&p,lattice:&lattice,outputs:Outputs{times:&times,horizons:&horizons,groups:&groups},frame:&frame,ledger:&ledger,keep_k1:false};
        let mut estimator=UnifiedEstimate::new(setup).unwrap();
        let flags=[false,false];
        let reduced=ReducedView{basis:&basis,images:&images,r:2,vectors:&sr,used_theta:&theta,used_coefficients:&a,kernel:&flags,projected:&flags};
        let mut solver=super::super::poles::Lattice::new(&p,lattice.clone()).unwrap();
        let e=estimator.estimate(&reduced,&mut solver).unwrap();
        assert_eq!((e.trajectory.len(),e.horizons.len()),(3,1));
        assert!(e.trajectory[0].abs()<1e-12&&e.trajectory[1..].iter().all(|v|v.is_finite()&&*v>0.)&&e.horizons.iter().all(|v|v.is_finite()),"{e:?}");
        assert!(e.largest().is_finite()&&e.work.block_operations>0);
    }
}
