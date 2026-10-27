//! The population pipeline: every output of a request is a matrix function
//! of one lattice operator applied to x0, read from one reduced model.
//!
//! 1. The explicit set P and its ring Q (`charge_set`); the lattice P u Q
//!    and its solves at the poles (`poles`).
//! 2. One rational Krylov space from x0 (`space`): the bordered steady
//!    column first, the translations through the gate active from the start,
//!    then the ladder's poles in the order of the error transform (re-ranked
//!    every RERANK solves; a pole that adds no direction leaves the ladder).
//! 3. Every CHECKPOINT solves: the reduced model (`reduced`), its outputs
//!    (trajectory on the grid, horizons, steady state, slowest relaxation,
//!    conserved count) and the error estimate (`estimate_api`); accepted
//!    when ACCEPTANCE_FACTOR x the largest bound is within the tolerance.
//!    The ladder exhausted without acceptance is an error.
use super::*;
use super::charge_set::{self,ExplicitSet};
use super::estimate_api::{ACCEPTANCE_FACTOR,Construct,ErrorEstimate,EstimateSetup,Outputs,ReducedView};
use super::ledger::{self,EXPANSION_LIMIT,POPULATION_TOLERANCE,RemainderLedger};
use super::reduced::{Dressed,Model};
use super::poles::{InnerWork,Lattice,Pole};
use super::space::{self,Column,Space};

/// The error estimate in use.
pub const ESTIMATOR:Construct=super::estimate_placeholder::construct;
/// Solves between checkpoints, and between re-rankings of the ladder.
const CHECKPOINT:usize=4;
const RERANK:usize=2;
/// Most solves of a request.
const MAXIMUM_SOLVES:usize=64;
/// Grid samples assumed for the ladder's top when a request has no grid,
/// and the longest time of a request for the limit alone (us).
const LADDER_SAMPLES:f64=200.;
const LIMIT_REACH:f64=1e6;

/// What a request asks for.
pub struct Request<'a>{
    /// The uniform grid from 0 (empty: no trajectory).
    pub times:&'a[f64],
    /// Horizons T (us) of the laboratory time average.
    pub horizons:&'a[f64],
    pub groups:&'a[Vec<usize>],
}

/// One checkpoint, as reported to an observer and in the diagnostics.
#[derive(Clone,Debug)]
pub struct Checkpoint{
    pub solves:usize,pub columns:usize,pub dimension:usize,
    pub estimate:ErrorEstimate,pub accepted:bool,
    pub growing:usize,pub reflected:usize,pub projected:usize,pub conserved:usize,pub slowest_rate:Option<f64>,
    /// Raw outputs: trajectory rows, horizon rows, steady row.
    pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Vec<f64>,
    pub milliseconds:f64,
}

#[derive(Clone,Debug,Default)]
pub struct Timing{pub explicit_set_ms:f64,pub setup_ms:f64,pub solves_ms:f64,pub images_ms:f64,pub dressed_ms:f64,pub model_ms:f64,pub ordering_ms:f64,pub estimate_ms:f64,pub total_ms:f64}

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
    pub estimate:ErrorEstimate,pub conserved:usize,pub slowest_rate:Option<f64>,
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

/// The reduced view of a model for the estimator.
fn view<'v>(space:&'v Space,model:&'v Model)->ReducedView<'v>{
    ReducedView{basis:&space.basis,images:&space.images,r:model.r,x:&model.x,g:&model.g,c0:&model.c0,
        theta:&model.theta,vectors:&model.vectors,coefficients:&model.coefficients,used_theta:&model.used_theta,used_coefficients:&model.used_coefficients,
        kernel:&model.kernel,growth_charge:model.growth_charge,steady:model.simple_kernel.then_some(model.steady.as_slice())}
}

/// Every output of a request from rho0 (row-major n x n); `observer` sees
/// every checkpoint.
pub fn solve(prepared:&LatticePrepared,initial:&[C],request:&Request<'_>,mut observer:Option<&mut dyn FnMut(&Checkpoint)>)->Result<Solution,String>{
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
    // 1. The lattice.
    let clock=Instant::now();let eigen=prepared.eigenbasis()?;
    let explicit=charge_set::explicit_set(prepared,initial,tolerance)?;timing.explicit_set_ms=clock.elapsed().as_secs_f64()*1e3;
    let clock=Instant::now();
    let mut lattice=Lattice::new(prepared,explicit.sets.clone())?;
    let ledger=RemainderLedger::new(&prepared.plan,lattice.operator().nu_max(),lattice.operator().dropped())?;
    if ledger.expansion_parameter>EXPANSION_LIMIT{return Err(format!("Static fold expansion parameter {:e} above {EXPANSION_LIMIT:e}",ledger.expansion_parameter));}
    let modes=lattice.modes.clone();
    // 2. The space: translations, the steady column.
    let mut space=Space::new(&mut lattice,initial)?;
    for s in charge_set::gated_steps(prepared,tolerance){if lattice.operator().index.contains_key(&s){space.activate(&mut lattice,&s)?;}}
    let outputs=Outputs{times,horizons:request.horizons,steady:true,groups:request.groups};
    let mut estimator=ESTIMATOR(EstimateSetup{prepared,lattice:&explicit.sets,outputs,ledger:&ledger,initial})?;
    let points=space::ladder(eigen,initial,t1,longest,tolerance)?;let mut alive=vec![true;points.len()];
    let mut dressed=Dressed::new(lattice.operator());
    timing.setup_ms=clock.elapsed().as_secs_f64()*1e3;
    space.solve(&mut lattice,Pole::Zero)?;
    let(mut queue,mut age):(Vec<usize>,usize)=(Vec::new(),RERANK);
    let mut checkpoints:Vec<Checkpoint>=Vec::new();
    let lapack=&prepared.lapack;
    let mut checkpoint=|space:&Space,dressed:&mut Dressed,lattice:&mut Lattice<'_>,timing:&mut Timing,checkpoints:&mut Vec<Checkpoint>|->Result<(Model,Checkpoint),String>{
        let clock=Instant::now();
        dressed.ingest(lapack,eigen,&space.basis,&space.images)?;
        let model=dressed.model(lapack,space.initial_norm,longest,tolerance)?;timing.model_ms+=clock.elapsed().as_secs_f64()*1e3;
        let trajectory=model.trajectory(dressed,times);
        let horizons:Vec<Vec<f64>>=request.horizons.iter().map(|&t|model.horizon(dressed,t)).collect();
        let steady=model.steady_populations(dressed);
        let clock2=Instant::now();
        let mut estimate=estimator.estimate(&view(space,&model),lattice)?;
        timing.estimate_ms+=clock2.elapsed().as_secs_f64()*1e3;
        for v in estimate.trajectory.iter_mut().chain(estimate.horizons.iter_mut()).chain(estimate.steady.iter_mut()){*v+=model.growth_charge;}
        let accepted=ACCEPTANCE_FACTOR*estimate.largest()<=tolerance;
        let point=Checkpoint{solves:space.solves(),columns:space.basis.len(),dimension:model.r,estimate,accepted,
            growing:model.growing,reflected:model.reflected,projected:model.projected,conserved:model.conserved(),slowest_rate:model.slowest_rate(),
            trajectory,horizons,steady,milliseconds:clock.elapsed().as_secs_f64()*1e3};
        checkpoints.push(point.clone());
        Ok((model,point))
    };
    let mut accepted:Option<Model>=None;
    loop{
        let live:Vec<usize>=(0..points.len()).filter(|&i|alive[i]).collect();
        let exhausted=live.is_empty()||space.solves()>=MAXIMUM_SOLVES;
        if exhausted||space.solves()%CHECKPOINT==0{
            let(model,point)=checkpoint(&space,&mut dressed,&mut lattice,&mut timing,&mut checkpoints)?;
            if let Some(o)=observer.as_mut(){o(&point);}
            if point.accepted{accepted=Some(model);break;}
            if exhausted{break;}
        }
        // The next pole: the live points ranked by sigma |p_err(xi)|.
        if queue.is_empty()||age>=RERANK{
            let clock=Instant::now();
            dressed.ingest(lapack,eigen,&space.basis,&space.images)?;
            let model=dressed.model(lapack,space.initial_norm,longest,tolerance)?;
            let xis:Vec<C>=live.iter().map(|&i|points[i].sigma()).collect();
            let transform=model.error_transform(&dressed,eigen,lapack,&xis)?;
            let mut scored:Vec<(usize,f64)>=live.iter().zip(&transform).map(|(&i,p)|(i,points[i].sigma().re*p.iter().map(|z|magnitude(*z)).fold(0.,f64::max))).collect();
            scored.sort_by(|a,b|b.1.total_cmp(&a.1));queue=scored.into_iter().map(|x|x.0).collect();age=0;
            timing.ordering_ms+=clock.elapsed().as_secs_f64()*1e3;
        }
        let i=queue.remove(0);age+=1;if !alive[i]{continue;}
        if space.solve(&mut lattice,points[i])?==0{alive[i]=false;}
    }
    let Some(model)=accepted else{
        let last=checkpoints.last().map_or(f64::NAN,|c|c.estimate.largest());
        return Err(format!("Populations not accepted after {} solves (estimated error {last:e} against {tolerance:e} / {ACCEPTANCE_FACTOR})",space.solves()));
    };
    let last=checkpoints.last().cloned().ok_or("An accepted run has its checkpoint")?;
    // Outputs.
    let mut trajectory=last.trajectory.clone();
    if let Some(first)=trajectory.first_mut(){*first=(0..n).map(|i|initial[i*n+i].re).collect();}
    let projected:Vec<Vec<f64>>=trajectory.iter().map(|p|simplex(p)).collect();
    let projection=trajectory.iter().zip(&projected).map(|(a,b)|ledger::metric(a,b,request.groups)).fold(0.,f64::max);
    let ops=super::krylov::Vectors::new(lattice.operator(),1)?;
    let field=|u:&[f64],images:bool|->Result<Vec<C>,String>{
        let coefficients=model.coordinates(u);let source:Vec<&[C]>=if images{space.images.iter().map(|v|v.as_slice()).collect()}else{space.basis.iter().map(|v|v.as_slice()).collect()};
        Ok(ops.combinations(&source,&[coefficients])?.remove(0))};
    let steady_field=field(&model.steady,false)?;let steady_image=field(&model.steady,true)?;
    let steady_residual=ops.norm(&steady_image)/ops.norm(&steady_field).max(f64::MIN_POSITIVE);
    let horizon_fields=request.horizons.iter().map(|&t|field(&model.state(t),false)).collect::<Result<Vec<_>,_>>()?;
    let work=lattice.work().clone();
    timing.solves_ms=work.milliseconds;timing.images_ms=space.images_ms;timing.dressed_ms=dressed.milliseconds;timing.total_ms=started.elapsed().as_secs_f64()*1e3;
    Ok(Solution{modes,translations:space.translations(),trajectory:projected,projection,horizons:last.horizons.clone(),steady:last.steady.clone(),
        horizon_fields,steady_field,steady_residual,estimate:last.estimate.clone(),conserved:last.conserved,slowest_rate:last.slowest_rate,
        columns:space.columns.clone(),dimension:model.r,images:space.basis.len(),work,ledger:ledger.clone(),timing,checkpoints,explicit:explicit.clone()})
}
