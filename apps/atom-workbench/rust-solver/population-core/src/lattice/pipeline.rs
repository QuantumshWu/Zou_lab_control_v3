//! The population pipeline: every output of a request is a matrix function
//! of one lattice operator applied to x0, read from one reduced model.
//!
//! 1. The explicit set P and its ring Q (`charge_set`); the lattice P u Q
//!    and its solves at the poles (`poles`). The request's frame (`frame`,
//!    the projection's inner product) is built on its own thread meanwhile.
//! 2. One rational Krylov space from x0 (`space`): the bordered steady
//!    column first, the translations through the gate active from the start,
//!    then the ladder's poles by their expected gain, re-ranked at every
//!    checkpoint: sigma |p_err(xi)| (the error transform) times the share of its
//!    own transform a pole's last solve removed (a pole whose solve did not
//!    reduce it loses its turn; one never solved keeps the whole), per real
//!    solve (a complex pole costs two); a pole that adds no direction leaves
//!    the ladder. The slowest relaxation is an output too: while it is
//!    neither resolved (residual within SLOW_ACCURACY of the rate) nor, for
//!    a request without a horizon, bounded far beyond every output time,
//!    the real poles near its rate gain priority: the best population gain
//!    (at least the tolerance) times its relative need, by their log
//!    distance, and the share of the need their last solve removed.
//! 3. Every CHECKPOINT solves (a request that propagates nothing: also
//!    right after the steady column): the reduced model (`reduced`), its outputs (trajectory
//!    on the grid, horizons, steady state, slowest relaxation, conserved
//!    count) and the error estimate (`estimate_api`); accepted when
//!    ACCEPTANCE_FACTOR x the largest bound is within the tolerance and, for
//!    a horizon, the slowest relaxation is resolved (unless a budget is
//!    spent; otherwise it is reported as the range its residual gives). The
//!    ladder exhausted, the solve budget or the basis budget (BASIS_BYTES)
//!    spent without acceptance is an error.
//!
//! The inner product matters only for the outputs that propagate the
//! reduced model, exp(t G) with t > 0 (the grid, the horizons): the
//! kernel of G, the steady state, does not depend on it up to the solve
//! residual (the steady column is in the space). The frame is built only
//! for a request that propagates; otherwise the model is the Euclidean
//! Galerkin one.
//!
//! The conserved count is the reduced model's kernel (the conserved
//! quantities x0 reaches), and the folded centre block's when the lattice is
//! that one block (a static model: its kernel is known exactly).
use super::*;
use super::charge_set::{self,ExplicitSet};
use super::estimate_api::{ACCEPTANCE_FACTOR,Construct,ErrorEstimate,EstimateSetup,ModelOutputs,Outputs,ReducedView};
use super::ledger::{self,EXPANSION_LIMIT,POPULATION_TOLERANCE,RemainderLedger};
use super::reduced::{Model,Projection,SLOW_ACCURACY,SlowRate,TestSpace};
use super::poles::{InnerWork,Lattice,Pole};
use super::space::{self,Column,SLOW_FLOOR,Space};
use super::frame::{SwFrame,SwMetric};
use super::estimator;

/// The error estimate in use.
pub const ESTIMATOR:Construct=super::estimator::CONSTRUCT;
/// Slow Ritz values checked for the slowest relaxation.
const SLOW_CANDIDATES:usize=6;
/// Solves between checkpoints; the ladder is re-ranked at every checkpoint
/// (from its reduced model) and whenever its queue runs out.
const CHECKPOINT:usize=4;
/// Most solves of a request, and the basis memory (basis, images and their
/// test-space columns).
const MAXIMUM_SOLVES:usize=48;
const BASIS_BYTES:usize=1<<30;
/// Grid samples assumed for the ladder's top when a request has no grid,
/// and the longest time of a request for the limit alone (us).
const LADDER_SAMPLES:f64=200.;
const LIMIT_REACH:f64=1e6;
/// Stack of the frame's thread (LAPACK on it).
const FRAME_STACK:usize=1<<26;

/// What a request asks for.
pub struct Request<'a>{
    /// The uniform grid from 0 (empty: no trajectory).
    pub times:&'a[f64],
    /// Horizons T (us) of the laboratory time average.
    pub horizons:&'a[f64],
    /// Whether the steady state (the t -> infinity mean) is reported.
    pub steady:bool,
    pub groups:&'a[Vec<usize>],
}

/// One checkpoint, as reported to an observer and in the diagnostics.
#[derive(Clone,Debug)]
pub struct Checkpoint{
    pub solves:usize,pub columns:usize,pub dimension:usize,
    pub estimate:ErrorEstimate,pub accepted:bool,
    pub growing:usize,pub clipped:usize,pub projected:usize,pub conserved:usize,pub slowest:Option<SlowRate>,
    /// Raw outputs: trajectory rows, horizon rows, steady row.
    pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Vec<f64>,
    /// The solves so far (inner steps, block operations, ms) and the run's
    /// timing so far.
    pub inner:InnerWork,pub timing:Timing,
    pub milliseconds:f64,
}

#[derive(Clone,Debug,Default)]
pub struct Timing{pub eigenbasis_ms:f64,pub explicit_set_ms:f64,pub setup_ms:f64,pub estimator_build_ms:f64,pub outputs_ms:f64,pub frame_ms:[f64;3],pub frame_wait_ms:f64,pub solves_ms:f64,pub images_ms:f64,pub projection_ms:f64,pub model_ms:f64,pub ordering_ms:f64,pub estimate_ms:f64,pub total_ms:f64}

pub struct Solution{
    pub explicit:ExplicitSet,pub modes:Vec<Charge>,pub translations:Vec<Charge>,
    /// Trajectory rows projected onto the probability simplex (row 0 is
    /// rho0's) and the largest projection change.
    pub trajectory:Vec<Vec<f64>>,pub projection:f64,
    /// Horizon and steady populations (laboratory time averages, raw).
    pub horizons:Vec<Vec<f64>>,pub steady:Vec<f64>,
    /// Their fields on `modes` (Hermitian-symmetric).
    pub horizon_fields:Vec<Vec<C>>,pub steady_field:Vec<C>,
    /// |A x| / |x| of the steady field (rad/us).
    pub steady_residual:f64,
    pub estimate:ErrorEstimate,pub conserved:usize,pub slowest:Option<SlowRate>,
    pub columns:Vec<Column>,pub checkpoints:Vec<Checkpoint>,
    pub dimension:usize,pub work:InnerWork,pub images:usize,
    pub ledger:RemainderLedger,pub timing:Timing,
}

/// Euclidean projection of real values onto the probability simplex.
pub fn simplex(values:&[f64])->Vec<f64>{
    let mut sorted=values.to_vec();sorted.sort_by(|a,b|b.total_cmp(a));
    let(mut sum,mut threshold)=(0.,0.);
    for(index,&value)in sorted.iter().enumerate(){sum+=value;let candidate=(sum-1.)/(index+1)as f64;if value>candidate{threshold=candidate;}else{break;}}
    values.iter().map(|x|(x-threshold).max(0.)).collect()
}

/// The slowest relaxation of a model: among the slowest decaying Ritz pairs
/// (theta, W y) of the reduced model, and with `euclidean` of the space's
/// Euclidean Rayleigh-Ritz too (whose residual is orthogonal to the space:
/// the frame's Galerkin vectors keep a fast part with a large image), the
/// least rate located by its residual |A W y - theta W y| / |W y| (at most
/// the rate; from the images, which keep its accuracy at slow rates), the
/// smallest residual among the pairs within that residual of it; else the
/// best located one. The reported relaxation takes both sets; the poles'
/// ranking the reduced model's alone (the Euclidean pairs' early
/// candidates steered five lasers + tweezer at 20 us into a space whose
/// Galerkin model grew 2-20 modes, not accepted in 39 solves against 12).
fn slowest(model:&Model,galerkin:&Projection,lapack:&Lapack,space:&Space,vectors:&super::krylov::Vectors,euclidean:bool)->Result<Option<SlowRate>,String>{
    let mut candidates:Vec<(C,Vec<C>)>=model.slow_candidates(SLOW_CANDIDATES).into_iter().map(|(j,_)|(model.theta[j],model.ritz_coordinates(j))).collect();
    if euclidean{candidates.extend(galerkin.euclidean_slow_pairs(lapack,SLOW_CANDIDATES)?);}
    if candidates.is_empty(){return Ok(None);}
    let parts:Vec<Vec<f64>>=candidates.iter().flat_map(|(_,y)|[y.iter().map(|z|z.re).collect(),y.iter().map(|z|z.im).collect()]).collect();
    let basis:Vec<&[C]>=space.basis.iter().map(|v|v.as_slice()).collect();let images:Vec<&[C]>=space.images.iter().map(|v|v.as_slice()).collect();
    let(w,aw)=(vectors.combinations(&basis,&parts)?,vectors.combinations(&images,&parts)?);
    let mut found:Vec<SlowRate>=candidates.iter().enumerate().map(|(c,(theta,_))|{
        let(theta,rate)=(*theta,-theta.re);let(wr,wi,ar,ai)=(&w[2*c],&w[2*c+1],&aw[2*c],&aw[2*c+1]);
        // A W y - theta W y, real and imaginary parts.
        let re:Vec<C>=(0..wr.len()).map(|e|C{re:ar[e].re-theta.re*wr[e].re+theta.im*wi[e].re,im:ar[e].im-theta.re*wr[e].im+theta.im*wi[e].im}).collect();
        let im:Vec<C>=(0..wr.len()).map(|e|C{re:ai[e].re-theta.re*wi[e].re-theta.im*wr[e].re,im:ai[e].im-theta.re*wi[e].im-theta.im*wr[e].im}).collect();
        let residual=(vectors.dot(&re,&re)+vectors.dot(&im,&im)).max(0.).sqrt()/(vectors.dot(wr,wr)+vectors.dot(wi,wi)).max(f64::MIN_POSITIVE).sqrt();
        SlowRate{rate,residual}}).collect();
    found.sort_by(|a,b|a.rate.total_cmp(&b.rate));
    let Some(first)=found.iter().copied().find(|s|s.residual<=s.rate)else{
        return Ok(found.iter().copied().min_by(|a,b|(a.residual/a.rate).total_cmp(&(b.residual/b.rate))))};
    Ok(found.iter().copied().filter(|s|(s.rate-first.rate).abs()<=first.residual).min_by(|a,b|a.residual.total_cmp(&b.residual)))
}

/// The reduced view of a model and its outputs for the estimator.
fn view<'v>(space:&'v Space,model:&'v Model,outputs:ModelOutputs<'v>,galerkin:&'v Projection)->ReducedView<'v>{
    ReducedView{basis:&space.basis,images:&space.images,r:model.r,x:&model.x,g:&model.g,c0:&model.c0,
        theta:&model.theta,vectors:&model.vectors,coefficients:&model.coefficients,used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,
        kernel:&model.kernel,growth_charge:model.growth_charge,steady:model.simple_kernel.then_some(model.steady.as_slice()),
        outputs,steady_baseline:space.steady_baseline.as_deref(),frame_columns:galerkin.frame_columns()}
}

/// Every output of a request from rho0 (row-major n x n) with the error
/// estimate `construct` (`ESTIMATOR`); `observer` sees every checkpoint.
pub fn solve(prepared:&LatticePrepared,initial:&[C],request:&Request<'_>,construct:Construct,mut observer:Option<&mut dyn FnMut(&Checkpoint)>)->Result<Solution,String>{
    let started=Instant::now();let tolerance=POPULATION_TOLERANCE;let n=prepared.generator.n;
    if initial.len()!=n*n{return Err("Initial density dimension differs".into());}
    let times=request.times;
    if !times.is_empty(){
        if times.len()<2||times[0]!=0.{return Err("A trajectory grid starts at zero with at least two times".into());}
        let(step,end)=(times[1],times[times.len()-1]);
        if !(end>0.)||times.iter().enumerate().any(|(i,&t)|(t-step*i as f64).abs()>1e-12*end.max(1.)){return Err("A trajectory grid must be uniform".into());}
    }
    if request.horizons.iter().any(|&t|!(t.is_finite()&&t>0.)){return Err("A horizon must be a finite positive time".into());}
    let longest=times.last().copied().into_iter().chain(request.horizons.iter().copied()).fold(0_f64,f64::max);
    let longest=if longest>0.{longest}else{LIMIT_REACH};
    let t1=if times.len()>1{times[1]}else{longest/LADDER_SAMPLES};
    let mut timing=Timing::default();
    // A horizon needs the slowest relaxation resolved. The frame (the inner
    // product) matters only when an output propagates the reduced model
    // (exp(t G), t > 0); without one, every output is the model's kernel,
    // read from the Euclidean Galerkin model and checked right after the
    // steady column.
    let resolve=!request.horizons.is_empty();
    let propagates=!times.is_empty()||!request.horizons.is_empty();
    // 1. The lattice, its space's steady column; the request's frame
    // (kernels and the second-order block's eigensystem, the projection's
    // inner product) on its own thread meanwhile.
    // The eigenbasis: exact for a request that propagates (the frame, its
    // metric and the pole ordering read it), clustered otherwise.
    let clock=Instant::now();let eigen=if propagates{prepared.eigenbasis()?}else{prepared.screening_eigenbasis()?};timing.eigenbasis_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
    let frame=SwFrame::new(prepared,estimator::longest_time(times,request.horizons));
    std::thread::scope(|scope|->Result<Solution,String>{
    if propagates{std::thread::Builder::new().stack_size(FRAME_STACK).spawn_scoped(scope,||frame.build()).map_err(|e|e.to_string())?;}
    let explicit=charge_set::explicit_set(prepared,initial,tolerance)?;timing.explicit_set_ms=clock.elapsed().as_secs_f64()*1e3;
    let clock=Instant::now();
    let mut lattice=Lattice::new(prepared,explicit.sets.clone())?;
    let ledger=RemainderLedger::new(&prepared.plan,lattice.operator().nu_max(),lattice.operator().dropped())?;
    if ledger.expansion_parameter>EXPANSION_LIMIT{return Err(format!("Static fold expansion parameter {:e} above {EXPANSION_LIMIT:e}",ledger.expansion_parameter));}
    // 2. The space: translations, the steady column.
    // Diagnostics (a reference run): ATOM_DIAG_MAX_SOLVES and
    // ATOM_DIAG_BASIS_GB replace the budgets, ATOM_DIAG_NO_ESTIMATE=1 skips
    // the error estimate.
    let maximum_solves=super::diagnostics::perturbation("MAX_SOLVES").map_or(MAXIMUM_SOLVES,|v|v as usize);
    let basis_bytes=super::diagnostics::perturbation("BASIS_GB").map_or(BASIS_BYTES,|v|(v*(1u64<<30)as f64)as usize);
    let no_estimate=super::diagnostics::perturbation("NO_ESTIMATE").is_some();
    let capacity={let op=lattice.operator();let d=op.n*op.n;basis_bytes/(std::mem::size_of::<C>()*2*d*(op.modes.len()+op.canonical_count()))};
    let mut space=Space::new(&mut lattice,initial,capacity)?;
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&mut lattice,&s)?;}}
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    space.solve(&mut lattice,Pole::Zero)?;
    let modes=lattice.modes.clone();
    let outputs=Outputs{times,horizons:request.horizons,steady:request.steady,groups:request.groups};
    let clock=Instant::now();
    let mut estimator=construct(EstimateSetup{prepared,lattice:&explicit.sets,outputs,frame:&frame,ledger:&ledger,initial})?;
    timing.estimator_build_ms=clock.elapsed().as_secs_f64()*1e3;
    let(points,slow_points)=space::ladder(eigen,initial,t1,longest,tolerance)?;let mut alive=vec![true;points.len()];
    // The frame's metric (waiting for the frame's thread), none when no
    // output propagates.
    let clock=Instant::now();
    // Diagnostics: ATOM_DIAG_EUCLIDEAN takes the Euclidean Galerkin model for
    // a request that propagates too (the frame still serves the estimate).
    let euclidean=super::diagnostics::perturbation("EUCLIDEAN").is_some();
    let metric=if propagates&&!euclidean{Some(SwMetric::new(&frame,eigen,lattice.operator())?)}else{None};
    timing.frame_wait_ms=clock.elapsed().as_secs_f64()*1e3;
    let test=metric.as_ref().map(|m|m as &dyn TestSpace);
    let mut galerkin=Projection::new(lattice.operator())?;
    let mut queue:Vec<usize>=Vec::new();
    let mut checkpoints:Vec<Checkpoint>=Vec::new();
    let lapack=&prepared.lapack;let ops=super::krylov::Vectors::new(lattice.operator(),1)?;
    let mut checkpoint=|space:&Space,galerkin:&mut Projection,lattice:&mut Lattice<'_>,timing:&mut Timing,checkpoints:&mut Vec<Checkpoint>,exhausted:bool|->Result<(Model,Checkpoint,Option<SlowRate>),String>{
        let clock=Instant::now();
        galerkin.ingest(test,&space.basis,&space.images)?;
        let model=galerkin.model(lapack,space.initial_norm,longest,tolerance)?;timing.model_ms+=clock.elapsed().as_secs_f64()*1e3;
        timing.projection_ms=galerkin.milliseconds;timing.images_ms=space.images_ms;
        let clock3=Instant::now();
        let trajectory=model.trajectory(galerkin,times);
        let horizons:Vec<Vec<f64>>=request.horizons.iter().map(|&t|model.horizon(galerkin,t)).collect();
        let steady=model.steady_populations(galerkin);
        timing.outputs_ms+=clock3.elapsed().as_secs_f64()*1e3;
        let clock2=Instant::now();
        let mut estimate=if no_estimate{ErrorEstimate{trajectory:vec![f64::INFINITY;trajectory.len()],horizons:vec![f64::INFINITY;horizons.len()],steady:request.steady.then_some(f64::INFINITY),work:Default::default()}}
            else{estimator.estimate(&view(space,&model,ModelOutputs{trajectory:&trajectory,horizons:&horizons,steady:&steady},galerkin),lattice)?};
        galerkin.clear_frame();
        timing.estimate_ms+=clock2.elapsed().as_secs_f64()*1e3;
        for v in estimate.trajectory.iter_mut().chain(estimate.horizons.iter_mut()).chain(estimate.steady.iter_mut()){*v+=model.growth_charge;}
        // A mean also needs the slowest relaxation resolved (the solve or
        // basis budget spent: reported as its range).
        let slow=slowest(&model,galerkin,lapack,space,&ops,true)?;let ranking=slowest(&model,galerkin,lapack,space,&ops,false)?;
        let accepted=ACCEPTANCE_FACTOR*estimate.largest()<=tolerance&&(!resolve||exhausted||slow.is_none_or(|s|s.resolved()))
            // Diagnostics: acceptance held back until ATOM_DIAG_ACCEPT_AFTER solves.
            &&super::diagnostics::perturbation("ACCEPT_AFTER").is_none_or(|after|space.solves()as f64>=after);
        let point=Checkpoint{solves:space.solves(),columns:space.basis.len(),dimension:model.r,estimate,accepted,
            growing:model.growing,clipped:model.clipped,projected:model.projected,conserved:model.conserved(),slowest:slow,
            trajectory,horizons,steady,inner:lattice.work().clone(),timing:timing.clone(),milliseconds:clock.elapsed().as_secs_f64()*1e3};
        checkpoints.push(point.clone());
        Ok((model,point,ranking))
    };
    let mut accepted:Option<Model>=None;
    // The last checkpoint's model, reused by the ranking at the same solve.
    let mut current:Option<(usize,Model)>=None;
    // Per point: its last score, the score it had when last solved, and the
    // share of its transform left after that solve; likewise the slowest
    // relaxation's need when it was last solved and the share left.
    let(mut score,mut solved_at,mut remaining)=(vec![0.;points.len()],vec![None::<f64>;points.len()],vec![0.;points.len()]);
    let(mut slow_at,mut slow_remaining)=(vec![None::<f64>;points.len()],vec![0.;points.len()]);
    // The slowest relaxation while it needs poles.
    let mut slow_need:Option<SlowRate>=None;
    loop{
        let live:Vec<usize>=(0..points.len()).filter(|&i|alive[i]).collect();
        let exhausted=live.is_empty()||space.solves()>=maximum_solves||space.full();
        if exhausted||space.solves()%CHECKPOINT==0||(!propagates&&space.solves()==1){
            let(model,point,ranking)=checkpoint(&space,&mut galerkin,&mut lattice,&mut timing,&mut checkpoints,exhausted)?;
            if let Some(o)=observer.as_mut(){o(&point);}
            if point.accepted{accepted=Some(model);break;}
            if exhausted{break;}
            slow_need=ranking.filter(|s|!(s.resolved()||(!resolve&&s.beyond(longest))));current=Some((space.solves(),model));queue.clear();
        }
        // The next pole by its expected gain.
        if queue.is_empty(){
            let clock=Instant::now();
            let model=match current.take(){Some((solves,model))if solves==space.solves()=>model,_=>{
                galerkin.ingest(test,&space.basis,&space.images)?;galerkin.model(lapack,space.initial_norm,longest,tolerance)?}};
            let xis:Vec<C>=live.iter().map(|&i|points[i].sigma()).collect();
            let mut transform=model.error_transform(&galerkin,&space.basis,&space.images,eigen,lapack,&xis)?;
            // Diagnostics: the error transform perturbed.
            if let Some(eps)=super::diagnostics::perturbation("TRANSFORM"){for(a,p)in transform.iter_mut().enumerate(){for(b,z)in p.iter_mut().enumerate(){*z=scaled(*z,1.+eps*super::diagnostics::noise(5,(a*1000+b)as u64));}}}
            for(&i,p)in live.iter().zip(&transform){if slow_points[i]{continue;}
                score[i]=points[i].sigma().re*p.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
                if let Some(before)=solved_at[i].take(){remaining[i]=if before>0.{(score[i]/before).min(1.)}else{1.};}
            }
            // Before the first checkpoint, the slowest relaxation of this model.
            if checkpoints.is_empty(){slow_need=slowest(&model,&galerkin,lapack,&space,&ops,false)?.filter(|s|!(s.resolved()||(!resolve&&s.beyond(longest))));}
            let slow=slow_need;
            // The slow relaxation's need and target: its rate once its
            // residual locates it, else the slowest reach (the slow points
            // reveal it).
            let need=slow.map_or(0.,|s|(s.residual/(SLOW_ACCURACY*s.rate)).min(1.));
            let lowest=(0..points.len()).filter(|&i|slow_points[i]).map(|i|points[i].sigma().re).fold(SLOW_FLOOR.max(1./longest),f64::min);
            let target=slow.map(|s|if s.residual<=s.rate{s.rate}else{lowest});
            for &i in &live{if let Some(before)=slow_at[i].take(){slow_remaining[i]=if before>0.{(need/before).min(1.)}else{1.};}}
            // The slowest relaxation weighs as much as the best population
            // gain (at least the tolerance) times its need.
            let gain=|i:usize|->f64{if slow_points[i]{0.}else{score[i]*(1.-remaining[i])/points[i].copies()as f64}};
            let weight=live.iter().map(|&i|gain(i)).fold(tolerance,f64::max);
            let priority=|i:usize|->f64{
                let bonus=match(target,points[i]){(Some(target),Pole::Real(sigma))=>{
                    let distance=(sigma/target).ln();weight*need*(1.-slow_remaining[i])/(1.+distance*distance)}
                    _=>0.};
                gain(i)+bonus};
            let mut ranked:Vec<(usize,f64)>=live.iter().map(|&i|(i,priority(i))).collect();
            ranked.sort_by(|a,b|b.1.total_cmp(&a.1));
            if super::diagnostics::reporting(){
                let top:Vec<String>=ranked.iter().take(5).map(|&(i,p)|format!("{}={:.4e}",points[i],p)).collect();
                let margin=if ranked.len()>1&&ranked[0].1>0.{(ranked[0].1-ranked[1].1)/ranked[0].1}else{f64::NAN};
                eprintln!("DBG rank solves {} margin {margin:.3e} top {}",space.solves(),top.join(" "));
            }
            queue=ranked.into_iter().map(|x|x.0).collect();
            timing.ordering_ms+=clock.elapsed().as_secs_f64()*1e3;
        }
        let i=queue.remove(0);if !alive[i]{continue;}
        solved_at[i]=Some(score[i]);if slow_need.is_some(){slow_at[i]=Some(slow_need.map_or(0.,|s|(s.residual/(SLOW_ACCURACY*s.rate)).min(1.)));}
        if space.solve(&mut lattice,points[i])?==0{alive[i]=false;}
    }
    let Some(model)=accepted else{
        let last=checkpoints.last().map_or(f64::NAN,|c|c.estimate.largest());
        return Err(format!("Populations not accepted after {} solves (estimated error {last:e} against {tolerance:e} / {ACCEPTANCE_FACTOR})",space.solves()));
    };
    let last=checkpoints.last().cloned().ok_or("An accepted run has its checkpoint")?;
    // Outputs.
    let clock=Instant::now();
    let mut trajectory=last.trajectory.clone();
    if let Some(first)=trajectory.first_mut(){*first=(0..n).map(|i|initial[i*n+i].re).collect();}
    let projected:Vec<Vec<f64>>=trajectory.iter().map(|p|simplex(p)).collect();
    let projection=trajectory.iter().zip(&projected).map(|(a,b)|ledger::metric(a,b,request.groups)).fold(0.,f64::max);
    let field=|u:&[f64],images:bool|->Result<Vec<C>,String>{
        let coefficients=model.coordinates(u);let source:Vec<&[C]>=if images{space.images.iter().map(|v|v.as_slice()).collect()}else{space.basis.iter().map(|v|v.as_slice()).collect()};
        Ok(ops.combinations(&source,&[coefficients])?.remove(0))};
    let steady_field=field(&model.steady,false)?;let steady_image=field(&model.steady,true)?;
    let steady_residual=ops.norm(&steady_image)/ops.norm(&steady_field).max(f64::MIN_POSITIVE);
    let horizon_fields=request.horizons.iter().map(|&t|field(&model.state(t),false)).collect::<Result<Vec<_>,_>>()?;
    timing.outputs_ms+=clock.elapsed().as_secs_f64()*1e3;
    let work=lattice.work().clone();
    let conserved=if modes.len()==1{eigen.kernel_count(super::reduced::KERNEL).max(1)}else{last.conserved};
    timing.solves_ms=work.milliseconds;timing.images_ms=space.images_ms;timing.projection_ms=galerkin.milliseconds;timing.total_ms=started.elapsed().as_secs_f64()*1e3;
    timing.frame_ms=frame.milliseconds();
    Ok(Solution{modes,translations:space.translations(),trajectory:projected,projection,horizons:last.horizons.clone(),steady:last.steady.clone(),
        horizon_fields,steady_field,steady_residual,estimate:last.estimate.clone(),conserved,slowest:last.slowest,
        columns:space.columns.clone(),dimension:model.r,images:space.basis.len(),work,ledger:ledger.clone(),timing,checkpoints,explicit:explicit.clone()})
    })
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared,dense_steady};
    use super::super::operator::LatticeOperator;
    #[test]
    fn lattice_pipeline_follows_the_exact_lattice_dynamics(){
        // The fixture from |0><0|: the trajectory (2 us), the state at a
        // horizon and the steady state against the same lattice P u Q
        // propagated densely (RK4, real form) and its dense null vector. The
        // far drives at 2% of the fixture's: the estimate charges the fold's
        // remainders over the slowest relaxation time (about 50 us here),
        // which the fixture's own far drives (0.02/us) never let through.
        let mut f=fixture();for v in f.far.iter_mut().chain(f.far2.iter_mut()){*v*=0.02;}let p=prepared(&f);let n=4;let d=16;let groups=vec![vec![0,1],vec![2,3]];
        let mut initial=vec![ZERO;d];initial[0]=ONE;
        let times:Vec<f64>=(0..21).map(|i|0.1*i as f64).collect();let horizons=[3.];
        let s=solve(&p,&initial,&Request{times:&times,horizons:&horizons,steady:true,groups:&groups},ESTIMATOR,None).unwrap();
        let op=LatticeOperator::new(&p,&s.modes).unwrap();let a=op.dense();let size=op.dimension();
        let rate=|x:&[C]|->Vec<C>{(0..size).map(|r|{let mut v=ZERO;for c in 0..size{mul_add(&mut v,a[r*size+c],x[c]);}v}).collect()};
        let mut x=vec![ZERO;size];x[op.centre*d]=ONE;
        let readout=|x:&[C],t:f64|->Vec<f64>{(0..n).map(|i|(0..s.modes.len()).map(|q|mul(C{re:(op.nu[q]*t).cos(),im:(op.nu[q]*t).sin()},x[q*d+i*n+i]).re).sum()).collect()};
        let steps=400;let h=0.1/steps as f64;let mut worst=0_f64;let mut t=0.;
        let axpy=|x:&[C],k:&[C],s:f64|->Vec<C>{x.iter().zip(k).map(|(a,b)|C{re:a.re+s*b.re,im:a.im+s*b.im}).collect()};
        for(i,row)in s.trajectory.iter().enumerate(){
            if i>0{for _ in 0..steps{let k1=rate(&x);let k2=rate(&axpy(&x,&k1,h/2.));let k3=rate(&axpy(&x,&k2,h/2.));let k4=rate(&axpy(&x,&k3,h));
                for e in 0..size{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}t+=h;}}
            worst=worst.max(ledger::metric(&readout(&x,t),row,&groups));
        }
        // The estimate accepts within the tolerance.
        assert!(worst<=POPULATION_TOLERANCE,"trajectory {worst:e}");
        for _ in 0..10*steps{let k1=rate(&x);let k2=rate(&axpy(&x,&k1,h/2.));let k3=rate(&axpy(&x,&k2,h/2.));let k4=rate(&axpy(&x,&k3,h));
            for e in 0..size{x[e].re+=h/6.*(k1[e].re+2.*k2[e].re+2.*k3[e].re+k4[e].re);x[e].im+=h/6.*(k1[e].im+2.*k2[e].im+2.*k3[e].im+k4[e].im);}}
        let centre:Vec<f64>=(0..n).map(|i|x[op.centre*d+i*n+i].re).collect();
        let horizon=ledger::metric(&centre,&s.horizons[0],&groups);assert!(horizon<=POPULATION_TOLERANCE,"horizon {horizon:e}");
        let exact=dense_steady(&p,&a,op.centre,n);
        // The bordered column at a tight tolerance is the lattice's null
        // vector; the reported steady state carries the algebraic error of
        // the bordered solve at STEADY_TOLERANCE.
        let mut lattice=Lattice::new(&p,s.explicit.sets.clone()).unwrap();let mut u=vec![ZERO;size];for i in 0..n{u[op.centre*d+i*n+i]=C{re:0.25,im:0.};}
        let(w,_,_)=lattice.solve(Pole::Zero,&[u],1e-13).unwrap();let trace:f64=(0..n).map(|i|w[0][op.centre*d+i*n+i].re).sum();
        let column:Vec<f64>=(0..n).map(|i|w[0][op.centre*d+i*n+i].re/trace).collect();
        let tight=ledger::metric(&exact,&column,&groups);assert!(tight<1e-10,"bordered column {tight:e}");
        let steady=ledger::metric(&exact,&s.steady,&groups);assert!(steady<1e-4,"steady {steady:e}");
        assert!(s.conserved==1&&s.slowest.is_some());
        // Projected rows: probabilities, the first one rho0's.
        assert!(s.trajectory.iter().all(|r|r.iter().all(|&v|v>=0.)&&(r.iter().sum::<f64>()-1.).abs()<1e-12)&&s.trajectory[0][0]==1.);
    }
}
