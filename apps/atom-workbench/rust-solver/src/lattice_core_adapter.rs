//! The protocol bridge to the population core, the folded Fourier lattice:
//! every rb87-full/2 request. The TypeScript model is the only owner of
//! atomic physics; this adapter borrows its H/C, exact phase coordinates and
//! state metadata, and serves the finite trajectory with the mean p̄(T) at
//! the standard horizons: the mean from a run of the pipeline
//! (`lattice::pipeline`) for the mean alone, the trajectory from a run for
//! the grid alone (see the result contract). The population tolerance is the
//! fixed `accuracy::POPULATION_TOLERANCE`; a request may only restate it.
//!
//! # Result contract
//! The one result shape, read by `normalizeDynamicsResult` in
//! src/solver-client.ts; main.rs adds `protocol` and `modelSchema`. A run
//! stops when every output has converged within the tolerance, when its
//! error bound stops improving, or when its budget is spent, and returns its
//! best checkpoint: a computed result is never withheld. The mean of every
//! request is the run of the mean alone (the same for a full request and a
//! mean-only one, `meanOnly`): one joint run for the two standard horizons,
//! 100 ms and 1 s, `meanHorizonUs` one of them (1e5 or 1e6; there is no t ->
//! infinity mean); a full request runs it on its own thread beside the run
//! of its trajectory, whose space spans the grid's times only, and one with
//! `includeMean` false runs no mean. Every checkpoint of the request's own
//! run (a full request's trajectory, once its mean is there) is streamed
//! first (`progress`: true) in the same shape.
//! - `states` (32 ids), `timesUs`, `populations` (one row per time). A mean
//!   result has the single t = 0 row, the initial populations; a finite result
//!   (at most 10 s, 2 to 2000 samples: a longer or denser grid is refused
//!   with the reason) has >= 2 projected rows and `populationAccuracy`: `status`
//!   (`estimated-converged` when every sample's bound is within the
//!   tolerance, else `estimated-partial` with its `reason`), `sampleErrors`
//!   (per sample, an absolute population error: the sample's convergence
//!   estimate over the last four checkpoints, plus the growth charge and the
//!   change of its projection onto the probability simplex),
//!   `estimatedMaximumError` (their largest), `requestedTolerance`, `scope`.
//! - `meanHorizons`, the mean, whenever one was asked for (computed or not):
//!   the two standard horizons, `[{horizonUs (1e5, 1e6), populations,
//!   densityReal, densityImag, estimatedError, withinTolerance, status,
//!   reason}]`. A computed entry is the mean at T, the laboratory time
//!   average at T (the state at T with its fast oscillations averaged out)
//!   projected onto the density matrices, with `estimatedError` (its
//!   convergence estimate plus the growth charge and the density
//!   projection's change), `withinTolerance` (that bound within the
//!   tolerance), `status` `time-average` and its `reason`. An entry is
//!   `not-converged`, populations and densities null, when its bound is 1 or
//!   more; when the run computed nothing usable, every entry is, with the
//!   run's reason and its error null. The request's `meanHorizonUs` names the
//!   entry the app reads as its mean; the mean's diagnostics below are that
//!   entry's.
//! - `diagnostics`. The reader gates the rows' `traceError` and
//!   `initialStateError`, `elapsedMs`, a finite result's `method`, and a mean's
//!   `meanMethod`, `meanTraceError` and `meanPositivityError`, and shows
//!   `slowestRelaxationUs`; every other key is profiling (the lattice's under
//!   `diagnostics.lattice`, on the final result).
use crate::{Model,Options,harmonics};
use serde_json::{json,Value};
use std::collections::BTreeMap;
use std::time::Instant;
use atom_population_core::{Complex64,Lapack};
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts};
use atom_population_core::lattice::LatticePrepared;
use atom_population_core::lattice::classification::Treatment;
use atom_population_core::lattice::charge_set::ExplicitSet;
use atom_population_core::lattice::accuracy::{self,POPULATION_TOLERANCE};
use atom_population_core::lattice::pipeline::{self,Answer,Checkpoint,LONGEST_TIME_US,MOST_SAMPLES,Request,Solution};

const ENGINE:&str="rust-population-core-lattice";
const METHOD:&str="Full32 density-matrix Lindblad dynamics on the folded Fourier lattice (explicit charges by mixing, seed amplitude, ring condition and the initial state's a-priori transfers, and their ring; far harmonics folded statically by a truncation rule): one block rational Krylov space from the initial state and its translations, every output read from one Petrov-Galerkin reduced model in the Schrieffer-Wolff/Floquet frame's metric, each output's error estimated from its convergence over the last four checkpoints";
/// A mean whose bound reaches this carries no information (a population
/// error of 1 bounds nothing): it is reported not converged, unshown.
const UNINFORMATIVE_BOUND:f64=1.;
/// The sentence added to a mean's reason when its bound is above the
/// tolerance.
const OUTSIDE:&str=" Not converged: its error bound (estimatedError) is above the tolerance, so the populations are an estimate.";
const MEAN_FRAME:&str="laboratory time average in the background eigenbasis";
/// Stack of the mean's thread (LAPACK on it).
const MEAN_STACK:usize=1<<26;

/// The worker start-up check: loads LAPACK and verifies one Hermitian
/// eigensystem, [[2,1-i],[1+i,3]] -> {1,4}.
pub fn warmup()->Result<(),String>{
    let c=|re,im|Complex64{re,im};
    let(values,_)=Lapack::load()?.hermitian_eigen(&[c(2.,0.),c(1.,1.),c(1.,-1.),c(3.,0.)],2)?;
    if (values[0]-1.).abs()>1e-12||(values[1]-4.).abs()>1e-12{return Err("LAPACK numerical self-check failed".into());}
    Ok(())
}

/// A requested population group: the eight F manifolds and all excited states.
struct Group{name:String,indices:Vec<usize>}
/// The ground flags and the requested groups, from the 32 state labels
/// `manifold-F-m`.
fn groups(model:&Model)->Result<(Vec<bool>,Vec<Group>),String>{
    let mut grouped=BTreeMap::<String,Vec<usize>>::new();let mut excited=Vec::new();let mut ground=Vec::new();
    for(i,label)in model.states.iter().enumerate(){
        let mut fields=label.splitn(3,'-');let manifold=fields.next().ok_or("Missing atomic manifold label")?;
        let f=fields.next().ok_or("Missing atomic F label")?;
        if !matches!(manifold,"ground"|"D1"|"D2")||fields.next().is_none(){return Err("The population core needs the 32-state manifold metadata".into());}
        ground.push(manifold=="ground");if manifold!="ground"{excited.push(i);}
        grouped.entry(format!("F:{manifold}:{f}")).or_default().push(i);
    }
    if ground.iter().filter(|&&x|x).count()!=8||grouped.len()!=8||excited.len()!=24{
        return Err("Rb87 population metadata must contain 8 ground and 24 excited states in all 8 F manifolds".into());
    }
    grouped.insert("F:excited".into(),excited);
    Ok((ground,grouped.into_iter().map(|(name,indices)|Group{name,indices}).collect()))
}

/// The row gates every result reports: the largest |row sum - 1| and the
/// first row's largest deviation from the initial populations.
fn row_errors(rows:&[Vec<f64>],initial:&[f64])->Result<(f64,f64),String>{
    let trace=rows.iter().map(|p|(p.iter().sum::<f64>()-1.).abs()).fold(0_f64,f64::max);
    let initial_error=rows[0].iter().zip(initial).map(|(a,b)|(a-b).abs()).fold(0_f64,f64::max);
    if trace>1e-7||initial_error>1e-10{return Err("Returned population probabilities failed fixed trace/initial checks".into());}
    Ok((trace,initial_error))
}

/// rho0 (row-major): `initialReal/initialImag` when given, else the state
/// `initialIndex`. A given density must be Hermitian with trace one and
/// positive semidefinite (to 1e-10).
fn initial_density(model:&Model,lapack:&Lapack)->Result<Vec<Complex64>,String>{
    let n=model.dimension;let d=n*n;
    let density:Vec<Complex64>=match(&model.initial_real,&model.initial_imag){
        (Some(re),Some(im))=>{
            if re.len()!=d||im.len()!=d{return Err("initialReal/initialImag must be n*n row-major arrays".into());}
            if re.iter().chain(im).any(|x|!x.is_finite()){return Err("Nonfinite initial density".into());}
            re.iter().zip(im).map(|(&re,&im)|Complex64{re,im}).collect()
        }
        (None,None)=>{
            if model.initial_index>=n{return Err("initialIndex is outside the state space".into());}
            return Ok((0..d).map(|k|Complex64{re:if k==model.initial_index*(n+1){1.}else{0.},im:0.}).collect());
        }
        _=>return Err("initialReal and initialImag must be provided together".into()),
    };
    let trace:f64=(0..n).map(|i|density[i*n+i].re).sum();let trace_imaginary:f64=(0..n).map(|i|density[i*n+i].im).sum();
    let hermiticity=(0..d).map(|k|{let(a,b)=(density[k],density[(k%n)*n+k/n]);(a.re-b.re).hypot(a.im+b.im)}).fold(0_f64,f64::max);
    if (trace-1.).hypot(trace_imaginary)>1e-10||hermiticity>1e-10{return Err("Initial density fails the trace/Hermitian checks".into());}
    // Read column-major the matrix is its conjugate, which has the same spectrum.
    if lapack.hermitian_eigen(&density,n)?.0[0]< -1e-10{return Err("Initial density is not positive semidefinite".into());}
    Ok(density)
}

/// The model's lattice preparation in the request frame, with its phase axes.
fn prepare(model:&Model,ground:&[bool])->Result<(LatticePrepared,harmonics::PhaseAxes),String>{
    let axes=harmonics::model_phase_axes(model)?;
    let parts:Vec<_>=axes.parts.iter().map(|&(hi,lo)|FrequencyParts{hi,lo}).collect();
    let phases=PhaseBasis::from_parts(&parts,&axes.drive_indices)?;
    let collapse:Vec<_>=model.collapse.iter().map(|x|MatrixView{real:&x.real,imag:&x.imag}).collect();
    let drives:Vec<_>=model.drives.iter().map(|x|DriveView{omega:x.omega,
        cosine:MatrixView{real:&x.cos.real,imag:&x.cos.imag},sine:MatrixView{real:&x.sin.real,imag:&x.sin.imag}}).collect();
    let view=ModelView{dimension:model.dimension,hamiltonian:MatrixView{real:&model.h_real,imag:&model.h_imag},
        collapse:&collapse,drives:&drives,ground};
    let lattice=LatticePrepared::prepare(&view,phases)?;
    // A re-framed excited manifold (`Source::frame_shift` g): its states'
    // laboratory frame indices f - g.
    let mut axes=axes;
    for(f,g)in axes.frame_indices.iter_mut().zip(&lattice.source.frame_shift){for(a,b)in f.iter_mut().zip(g){*a-=b;}}
    Ok((lattice,axes))
}

/// One request's lattice preparation, shared by its mean and its trajectory.
struct Prepared{
    lattice:LatticePrepared,axes:harmonics::PhaseAxes,
    /// Requested groups (indices and names).
    groups:Vec<Vec<usize>>,names:Vec<String>,
    /// rho0 (row-major) and its populations.
    density:Vec<Complex64>,initial:Vec<f64>,
    /// The explicit set of rho0, built by the first run that needs it.
    explicit:std::sync::OnceLock<Result<ExplicitSet,String>>,
    /// The runs that execute at once: 2 for a full request (its trajectory
    /// and mean), else 1.
    runs:usize,
}
fn prepare_request(model:&Model,options:&Options)->Result<Prepared,String>{
    if options.population_tolerance!=POPULATION_TOLERANCE{
        return Err(format!("populationTolerance is fixed at {POPULATION_TOLERANCE} (requested {})",options.population_tolerance));}
    if !STANDARD_HORIZONS.contains(&options.mean_horizon_us){return Err("meanHorizonUs must be 1e5 or 1e6 us (100 ms or 1 s)".into());}
    if model.dimension!=32||model.states.len()!=32{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    let(ground,requested)=groups(model)?;
    let(lattice,axes)=prepare(model,&ground)?;
    let n=32;
    let density=initial_density(model,&lattice.lapack)?;
    let initial=(0..n).map(|i|density[i*n+i].re).collect();
    Ok(Prepared{lattice,axes,groups:requested.iter().map(|g|g.indices.clone()).collect(),names:requested.iter().map(|g|g.name.clone()).collect(),density,initial,explicit:std::sync::OnceLock::new(),runs:if options.mean_only||!options.include_mean{1}else{2}})
}

/// Laboratory time average of a lattice field (row-major density): the
/// same-sector pairs from the mode of their frame charge f_i - f_j; the
/// cross-sector pairs average out (the largest discarded is returned).
fn lab_density(field:&[Complex64],modes:&[Vec<i32>],axes:&harmonics::PhaseAxes)->(Vec<Complex64>,f64){
    let n=32;let d=n*n;let mode=|q:&Vec<i32>|modes.iter().position(|m|m==q);
    let mut density=vec![Complex64::default();d];let mut discarded=0_f64;
    for i in 0..n{for j in 0..n{
        let charge:Vec<i32>=axes.frame_indices[i].iter().zip(&axes.frame_indices[j]).map(|(a,b)|a-b).collect();
        let Some(m)=mode(&charge)else{continue};let v=field[m*d+i*n+j];
        if axes.frame_sectors[i]==axes.frame_sectors[j]{density[i*n+j]=v;}else if charge.iter().all(|&k|k==0){discarded=discarded.max(v.re.hypot(v.im));}
    }}
    for i in 0..n{for j in i..n{let(a,b)=(density[i*n+j],density[j*n+i]);
        let m=Complex64{re:0.5*(a.re+b.re),im:0.5*(a.im-b.im)};density[i*n+j]=m;density[j*n+i]=Complex64{re:m.re,im:-m.im};}}
    (density,discarded)
}
/// The Frobenius-nearest density matrix (Hermitian, positive, trace one) to
/// a Hermitian one (row-major): its eigenvalues projected onto the
/// probability simplex.
fn project_density(lapack:&atom_population_core::Lapack,density:&[Complex64])->Result<Vec<Complex64>,String>{
    let n=32;let mut column=vec![Complex64::default();n*n];for i in 0..n{for j in 0..n{column[i+n*j]=density[i*n+j];}}
    let(values,vectors)=lapack.hermitian_eigen(&column,n)?;let weights=pipeline::simplex(&values);
    let mut out=vec![Complex64::default();n*n];
    for(k,&w)in weights.iter().enumerate(){if w==0.{continue;}for i in 0..n{let a=vectors[i+n*k];for j in 0..n{let b=vectors[j+n*k];
        out[i*n+j].re+=w*(a.re*b.re+a.im*b.im);out[i*n+j].im+=w*(a.im*b.re-a.re*b.im);}}}
    Ok(out)
}

/// The lattice profiling: plan, fold, Floquet kernels and the run.
fn lattice_json(prepared:&LatticePrepared,run:Option<&Solution>)->Value{
    let plan=&prepared.plan;let f=&prepared.fold.diagnostics;
    let targets:Vec<Value>=prepared.fold.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,
        "restStored":t.rest.iter().map(|r|r.stored()).collect::<Vec<_>>(),"denseSectors":t.rest.iter().map(|r|r.blocks.len()).collect::<Vec<_>>(),
        "whole":t.whole,"norms":t.norms,"largestEntry":t.largest_entry})).collect();
    let k=&prepared.floquet.diagnostics;
    let floquet=json!({"milliseconds":k.milliseconds,"iterations":k.iterations,"change":k.change,
        "pairs":k.pairs,"guarded":k.guarded,"smallParameter":k.small_parameter,"frameCondition":k.condition,
        "support":k.support.iter().map(|(q,size)|json!({"charge":q,"kernel":size})).collect::<Vec<_>>(),"belowFloor":k.below});
    let run=run.map(|s|{let e=&s.explicit;let w=&s.work;let t=&s.timing;let f=&s.fold;let dropped=&f.dropped;
        json!({"explicit":e.sets.explicit,"ring":s.lattice.ring,"completed":s.completed,"gated":e.gated,"promoted":e.promoted,"covered":e.covered,"leak":e.leak,
            "ringInternalEdges":e.internal.len(),"explicitSetMs":e.milliseconds,"translations":s.translations,"block":s.block,
            "columns":s.columns.iter().map(|c|json!({"pole":c.pole.to_string(),"innerSteps":c.steps,"residual":c.residual,"added":c.added,"ms":c.milliseconds,"blocks":c.blocks})).collect::<Vec<_>>(),
            "basis":s.basis,"dimension":s.dimension,
            "inner":{"solves":w.solves,"steps":w.steps,"appliedBlocks":w.applied_blocks,"preconditionedBlocks":w.preconditioned_blocks,
                "blockOperations":w.block_operations(),"batches":w.batches,"batchMs":w.batch_ms,
                "applyMs":w.timing.apply_ms,"preconditionMs":w.timing.precondition_ms,"orthogonalizeMs":w.timing.orthogonalize_ms},
            "checkpoints":s.checkpoints.iter().map(|c|json!({"solves":c.solves,"columns":c.columns,"dimension":c.dimension,"accepted":c.accepted,"bound":c.largest,
                "trajectory":c.estimate.trajectory.iter().fold(0_f64,|a,&b|a.max(b)),"horizons":c.estimate.horizons,
                "growing":c.growing,"clipped":c.clipped,"projected":c.projected,"slowestRate":c.slowest.map(|s|s.rate),"ms":c.milliseconds})).collect::<Vec<_>>(),
            "fold":{"horizonUs":f.rule.map(|r|r.horizon),"nuMax":f.rule.map(|r|r.nu_max),"budget":f.rule.map(|r|r.budget),"orders":f.orders,"targets":f.targets,
                "rings":f.rings,"ringAmplitudes":f.ring_amplitudes,"ringOrders":f.ring_orders,
                "dropped":{"entries":dropped.entries,"rests":dropped.rests,"orders":dropped.orders,"targets":dropped.targets,"expansion":dropped.expansion,"total":dropped.total(),
                    "pieces":dropped.pieces,"keptTargets":dropped.kept_targets,"candidateTargets":dropped.candidate_targets},"buildMs":f.build_ms,"depthMs":f.depth_ms},
            "timingMs":{"eigenbasis":t.eigenbasis_ms,"explicitSet":t.explicit_set_ms,"apriori":t.apriori_ms,"setup":t.setup_ms,"outputs":t.outputs_ms,"frame":t.frame_ms.iter().sum::<f64>(),"frameParts":t.frame_ms,"frameWait":t.frame_wait_ms,"solves":t.solves_ms,"projection":t.projection_ms,
                "model":t.model_ms,"estimate":t.estimate_ms,"total":t.total_ms}})});
    json!({
        "harmonicPlan":{"explicit":plan.count(&Treatment::Explicit{structural:true}),"nonStructural":plan.count(&Treatment::Explicit{structural:false}),
            "folded":plan.count(&Treatment::Fold),"gap":plan.gap,"cut":plan.cut,"l0Bound":plan.l0_bound,"structuralFloor":plan.structural_floor,
            "steps":plan.steps.iter().map(|s|json!({"charge":s.charge,"nu":s.nu,"size":s.size,"treatment":format!("{:?}",s.treatment)})).collect::<Vec<_>>(),
            "foldTargets":plan.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>(),
            "omittedTargets":plan.omitted.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>()},
        "fold":{"farSources":f.far_sources,"columns":f.columns,"restBound":f.rest_bound,"lambda0":f.lambda0,"orders":f.orders,
            "series":f.series.iter().map(|(q,rho,terms)|json!({"charge":q,"ratio":rho,"terms":terms})).collect::<Vec<_>>(),"milliseconds":f.milliseconds,
            "phaseMs":{"columns":f.phase_ms[0],"storage":f.phase_ms[1]},"targets":targets},
        "floquet":floquet,"eigenbasis":prepared.eigenbasis_built().map(|e|json!({"milliseconds":e.milliseconds,"clusters":e.clusters,"largestCluster":e.largest_cluster,"passes":e.passes,"condition":e.condition})),"run":run})
}

/// The mean of a request alone: one joint run for the standard horizons
/// (100 ms and 1 s); `observer` sees every checkpoint.
fn mean_run(request:&Prepared,observer:&mut dyn FnMut(&Checkpoint,&Answer))->Result<Solution,String>{
    pipeline::solve(&request.lattice,&request.density,&Request{times:&[],horizons:&STANDARD_HORIZONS,groups:&request.groups,explicit:&request.explicit,runs:request.runs},Some(observer))
}
/// The trajectory of a request alone on its grid.
fn trajectory_run(request:&Prepared,times:&[f64],observer:&mut dyn FnMut(&Checkpoint,&Answer))->Result<Solution,String>{
    pipeline::solve(&request.lattice,&request.density,&Request{times,horizons:&[],groups:&request.groups,explicit:&request.explicit,runs:request.runs},Some(observer))
}

/// The standard horizons (us) every mean run returns, the only ones a
/// request may ask for: 100 ms and 1 s.
const STANDARD_HORIZONS:[f64;2]=[1e5,1e6];

/// One mean of an answer, at its horizon T (us): the laboratory time average
/// projected onto the density matrices (the change charged), its bound and
/// what it is.
struct Mean{
    horizon:f64,populations:Vec<f64>,density:Vec<Complex64>,estimated:f64,within:bool,
    reason:String,discarded:f64,positivity:f64,trace_error:f64,
}
impl Mean{
    /// A bound of 1 or more bounds nothing: the mean is not converged and its
    /// populations are not shown.
    fn informative(&self)->bool{self.estimated<UNINFORMATIVE_BOUND}
    /// The status and reason as reported.
    fn reported(&self)->(&'static str,String){
        if self.informative(){("time-average",self.reason.clone())}
        else{("not-converged",format!("The mean's error bound is {:.2e}, at least 1: it carries no information, so no populations are shown",self.estimated))}
    }
    /// An entry of `meanHorizons`.
    fn entry(&self)->Value{
        let(status,reason)=self.reported();let shown=self.informative();
        json!({"horizonUs":self.horizon,"status":status,"reason":reason,"estimatedError":self.estimated,"withinTolerance":self.within&&shown,
            "populations":shown.then(||self.populations.clone()),
            "densityReal":shown.then(||self.density.iter().map(|z|z.re).collect::<Vec<_>>()),
            "densityImag":shown.then(||self.density.iter().map(|z|z.im).collect::<Vec<_>>())})
    }
}
/// The entry of a mean whose run computed nothing usable.
fn failed(horizon:f64,reason:&str)->Value{
    json!({"horizonUs":horizon,"status":"not-converged","reason":reason,"estimatedError":null,"withinTolerance":false,
        "populations":null,"densityReal":null,"densityImag":null})
}
fn mean_at(model:&Model,request:&Prepared,answer:&Answer,horizon:f64)->Result<Mean,String>{
    let(prepared,axes)=(&request.lattice,&request.axes);let n=32;
    let index=STANDARD_HORIZONS.iter().position(|&h|h==horizon).ok_or("A mean horizon outside the run")?;
    let(field,error)=(&answer.horizon_fields[index],answer.horizon_errors[index]);
    let(raw,discarded)=lab_density(field,&answer.modes,axes);
    let diagonal=|m:&[Complex64]|(0..n).map(|i|m[i*n+i].re).collect::<Vec<f64>>();
    let density=project_density(&prepared.lapack,&raw)?;let projection=accuracy::metric(&diagonal(&raw),&diagonal(&density),&request.groups);
    let estimated=error+projection;
    let populations=diagonal(&density);
    let positivity=(-prepared.lapack.hermitian_eigen(&density,n)?.0[0]).max(0.);
    let trace_error=(populations.iter().sum::<f64>()-1.).abs();
    // Within the tolerance: its bound, the projection's change included.
    let within=estimated<=POPULATION_TOLERANCE;
    // What it is: the state at T of the specified initial density, its fast
    // oscillations averaged out.
    let reason=if model.drives.is_empty(){"State at the horizon (horizonUs) of the undriven Lindblad equation from the specified initial density, its optical coherences averaged out in the laboratory frame."}
        else{"Phase average at the horizon (horizonUs) of the full driven Lindblad equation from the specified initial density: the state at T with only its fast oscillations (beats, optical micromotion) averaged out, not a time-independent density."};
    let reason=if within{reason.to_string()}else{format!("{reason}{OUTSIDE}")};
    Ok(Mean{horizon,populations,density,estimated,within,reason,discarded,positivity,trace_error})
}

/// The mean of an answer: its `meanHorizons` (every standard horizon),
/// and the diagnostics of the requested one (`meanHorizonUs`).
fn mean_fields(model:&Model,options:&Options,request:&Prepared,answer:&Answer)->Result<(Value,serde_json::Map<String,Value>),String>{
    let means=STANDARD_HORIZONS.iter().map(|&h|mean_at(model,request,answer,h)).collect::<Result<Vec<_>,_>>()?;
    let requested=STANDARD_HORIZONS.iter().position(|&h|h==options.mean_horizon_us).ok_or("meanHorizonUs outside the standard horizons")?;
    let mean=&means[requested];
    let entries=json!(means.iter().map(Mean::entry).collect::<Vec<_>>());
    let slow=answer.slowest.as_ref();let resolved=slow.filter(|s|s.resolved());
    let diagnostics=json!({"meanMethod":METHOD,"meanDensityFrame":MEAN_FRAME,
        "meanTraceError":mean.trace_error,"meanPositivityError":mean.positivity,
        "meanAccuracy":{"requestedTolerance":POPULATION_TOLERANCE,"metric":"max over states and requested F groups"},
        "slowestRelaxationUs":resolved.map(|s|1./s.rate),
        "slowestRelaxation":slow.map(|s|json!({"ratePerUs":s.rate,"residualPerUs":s.residual,"classesMatched":s.matched,"resolved":s.resolved(),
            "relaxationRangeUs":[1./(s.rate+s.residual),if s.matched&&s.rate>s.residual{Some(1./(s.rate-s.residual))}else{None}],
            "rule":"the slowest decaying Ritz pair of population weight at least 1% of the tolerance; resolved (slowestRelaxationUs) when its residual is within a tenth of its rate and every slower class of the initial state (the second-order block's eigenvalues) has a pair within 40%, else only the range: at least 1 / (rate + residual)"})),
        "discardedOpticalCoherence":mean.discarded});
    match diagnostics{Value::Object(d)=>Ok((entries,d)),_=>Err("Mean diagnostics are not an object".into())}
}

/// The request's uniform grid (us).
fn grid(options:&Options)->Vec<f64>{(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect()}

/// The mean of a request as computed: its answer and run, or why nothing
/// usable was computed.
type MeanRun<'a>=Result<(&'a Answer,Option<&'a Solution>),String>;

/// The result in the one contract: the trajectory with its per-sample
/// bounds (`trajectory`; none for a mean alone) and the mean when one was
/// asked for; `solutions` (the final result) add the runs' diagnostics, their
/// absence marks a streamed checkpoint (`progress`).
fn result(model:&Model,options:&Options,request:&Prepared,trajectory:Option<(&Answer,Option<&Solution>)>,mean:Option<MeanRun<'_>>,last:bool,started:Instant)->Result<Value,String>{
    let prepared=&request.lattice;
    let rows=match trajectory{Some((answer,_))=>answer.trajectory.clone(),None=>vec![request.initial.clone()]};
    let(row_trace,initial_error)=row_errors(&rows,&request.initial)?;
    let times:Vec<f64>=if trajectory.is_some(){grid(options)}else{vec![0.]};
    let solves=trajectory.map(|t|t.0.solves).or_else(||mean.as_ref().and_then(|m|m.as_ref().ok().map(|m|m.0.solves))).unwrap_or(0);
    let mut result=json!({"states":model.states,"timesUs":times,"populations":rows,
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":request.axes.source_to_basis,"solves":solves,
            "preparationMs":{"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2],"floquetKernels":prepared.timing[3]}}});
    if let Some((answer,solution))=trajectory{
        let estimated=answer.sample_errors.iter().fold(0_f64,|a,&b|a.max(b));
        // Converged: every sample's bound within the tolerance.
        let converged=estimated<=POPULATION_TOLERANCE;
        result["populationAccuracy"]=json!({"status":if converged{"estimated-converged"}else{"estimated-partial"},"requestedTolerance":POPULATION_TOLERANCE,
            "sampleErrors":answer.sample_errors,"estimatedMaximumError":estimated,"scope":{"states":32,"samples":options.samples,"groups":request.names}});
        if !converged{result["populationAccuracy"]["reason"]=json!(solution.and_then(|s|s.reason.clone())
            .unwrap_or_else(||format!("Checkpoint at {} solves: the error bound is {estimated:.2e} against the tolerance {POPULATION_TOLERANCE:e}; the run continues",answer.solves)));}
        result["diagnostics"]["method"]=json!(METHOD);
        if let Some(s)=solution{result["diagnostics"]["lattice"]=lattice_json(prepared,Some(s));}
    }
    match mean{
        Some(Ok((answer,solution)))=>{
            let(entries,diagnostics)=mean_fields(model,options,request,answer)?;
            result["meanHorizons"]=entries;
            if let Value::Object(target)=&mut result["diagnostics"]{for(key,value)in diagnostics{target.insert(key,value);}}
            if let Some(s)=solution{result["diagnostics"][if trajectory.is_some(){"meanLattice"}else{"lattice"}]=lattice_json(prepared,Some(s));}
        }
        // Nothing usable was computed for the mean.
        Some(Err(e))=>{result["meanHorizons"]=json!(STANDARD_HORIZONS.iter().map(|&h|failed(h,&e)).collect::<Vec<_>>());
            result["diagnostics"]["meanMethod"]=json!(METHOD);}
        None=>{}
    }
    if !last{result["progress"]=json!(true);}
    if result["diagnostics"].get("lattice").is_none(){result["diagnostics"]["lattice"]=lattice_json(prepared,None);}
    Ok(result)
}

/// A request through the lattice core: the mean alone (meanOnly), or the
/// mean on its own thread beside the trajectory; every checkpoint of the request's own run
/// (the mean's for a mean alone, the trajectory's with the mean for a full
/// request) is published as it is made, the best one returned.
pub fn solve(model:&Model,options:&Options,publish:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();
    if !options.mean_only{
        if !(options.duration_us.is_finite()&&options.duration_us>0.){return Err("The trajectory duration must be a positive finite time".into());}
        if options.duration_us>LONGEST_TIME_US{return Err(format!("The trajectory duration is limited to 10 s (requested {} s)",options.duration_us/1e6));}
        if !(2..=MOST_SAMPLES).contains(&options.samples){return Err(format!("A trajectory has 2 to {MOST_SAMPLES} samples (requested {})",options.samples));}
    }
    let request=prepare_request(model,options)?;
    let mut failure:Option<String>=None;
    if options.mean_only{
        let mut observe=|_:&Checkpoint,answer:&Answer|{if failure.is_some(){return;}
            match result(model,options,&request,None,Some(Ok((answer,None))),false,started){Ok(value)=>publish(&value),Err(e)=>failure=Some(e)}};
        let mean=mean_run(&request,&mut observe);
        if let Some(e)=failure{return Err(e);}
        return result(model,options,&request,None,Some(mean.as_ref().map(|s|(&s.answer,Some(s))).map_err(|e|e.clone())),true,started);
    }
    let times=grid(options);
    // The mean runs on its own thread while the trajectory runs: a
    // trajectory checkpoint is published once the mean is there.
    let mean:std::sync::OnceLock<Result<Solution,String>>=std::sync::OnceLock::new();
    let mean_view=||mean.get().map(|m|m.as_ref().map(|s|(&s.answer,Some(s))).map_err(|e|e.clone()));
    let trajectory=std::thread::scope(|scope|->Result<Solution,String>{
        let worker=if options.include_mean{Some(std::thread::Builder::new().stack_size(MEAN_STACK).spawn_scoped(scope,||{
            let _=mean.set(mean_run(&request,&mut |_:&Checkpoint,_:&Answer|{}));}).map_err(|e|e.to_string())?)}else{None};
        let mut observe=|_:&Checkpoint,answer:&Answer|{if failure.is_some()||(options.include_mean&&mean.get().is_none()){return;}
            match result(model,options,&request,Some((answer,None)),mean_view(),false,started){Ok(value)=>publish(&value),Err(e)=>failure=Some(e)}};
        let trajectory=trajectory_run(&request,&times,&mut observe);
        if let Some(worker)=worker{worker.join().map_err(|_|"The mean's thread failed".to_string())?;}
        trajectory})?;
    if let Some(e)=failure{return Err(e);}
    result(model,options,&request,Some((&trajectory.answer,Some(&trajectory))),mean_view(),true,started)
}

#[cfg(test)]
mod tests{
    use super::*;
    /// A full request's p(1 s) is the mean-only request's, within the sum of
    /// their bounds, and both are within 1e-4 of the release solver's
    /// (tests/fixtures/probe_only_short.json: the candidate-EXE smoke
    /// fixture, a probe alone with a 0.02 us trajectory, where the full
    /// request's inline mean was once off by 3.4e-2 with a bound of 3e-12).
    #[test]
    fn a_short_trajectory_full_request_has_the_mean_only_mean(){
        let fixture:Value=serde_json::from_str(include_str!("../tests/fixtures/probe_only_short.json")).unwrap();
        let model:Model=serde_json::from_value(fixture["model"].clone()).unwrap();
        let options=|mean_only:bool|->Options{let mut o=fixture["options"].clone();o["meanOnly"]=json!(mean_only);serde_json::from_value(o).unwrap()};
        let full=solve(&model,&options(false),&mut |_|{}).unwrap();
        let mean=solve(&model,&options(true),&mut |_|{}).unwrap();
        let read=|v:&Value|->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()};
        let second=json!(1e6);let(full_mean,only_mean)=(entry(&full,&second),entry(&mean,&second));
        let(a,b,reference)=(read(&full_mean["populations"]),read(&only_mean["populations"]),read(&fixture["referenceMean"]));
        let bounds=full_mean["estimatedError"].as_f64().unwrap()+only_mean["estimatedError"].as_f64().unwrap();
        let difference=a.iter().zip(&b).map(|(x,y)|(x-y).abs()).fold(0.,f64::max);
        assert!(difference<=bounds,"full against mean-only {difference:e}, bounds {bounds:e}");
        for(name,p)in[("full",&a),("mean-only",&b)]{
            let off=p.iter().zip(&reference).map(|(x,y)|(x-y).abs()).fold(0.,f64::max);
            assert!(off<=1e-4,"{name} p(1 s) {off:e} from the release solver's");
        }
        assert_eq!(full["populationAccuracy"]["status"],json!("estimated-converged"));
        assert_eq!(full_mean["withinTolerance"],json!(true));
    }
    /// The default scene's mean at T = 100 ms and 1 s against exact
    /// references (tests/fixtures/default_scene_horizons.json), each within
    /// `allowed` (the release returned the steady state for 100 ms, off by
    /// 7.8e-3 with a bound of 7e-13), and p(T) moving
    /// toward the exact steady state as T grows (exact: 7.8e-3 at 100 ms,
    /// 4.0e-3 at 1 s; the scene's two slow classes decay monotonically).
    /// What a mean may differ from the default scene's references: its bound,
    /// half of what its run's fold dropped by the truncation rule (induced
    /// trace norm x T: the populations move by at most half of it), and the
    /// references' own accuracy, REFERENCE (their two cross-check methods
    /// differ by 3e-5 and more; they hold the far shell at second order,
    /// whose fourth-order ring the rule keeps: 1.2e-5 at 1 s with every fold
    /// piece kept).
    const REFERENCE:f64=3e-5;
    fn allowed(out:&Value,bound:f64)->f64{bound+0.5*out["diagnostics"]["lattice"]["run"]["fold"]["dropped"]["total"].as_f64().unwrap()+REFERENCE}
    #[test]
    fn the_default_scene_means_follow_their_horizons(){
        let fixture:Value=serde_json::from_str(include_str!("../tests/fixtures/default_scene_horizons.json")).unwrap();
        let model:Model=serde_json::from_value(fixture["model"].clone()).unwrap();
        let groups:Vec<Vec<usize>>=serde_json::from_value(fixture["groups"].clone()).unwrap();
        let read=|v:&Value|->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()};
        // The populations' and the groups' largest difference.
        let metric=|a:&[f64],b:&[f64]|->f64{let d:Vec<f64>=a.iter().zip(b).map(|(x,y)|x-y).collect();
            groups.iter().map(|g|g.iter().map(|&i|d[i]).sum::<f64>().abs()).chain(d.iter().map(|x|x.abs())).fold(0.,f64::max)};
        let mut means=Vec::new();
        for(horizon,key)in[(json!(1e5),"100000"),(json!(1e6),"1000000")]{
            let mut o=fixture["options"].clone();o["meanOnly"]=json!(true);o["meanHorizonUs"]=horizon.clone();
            let out=solve(&model,&serde_json::from_value(o).unwrap(),&mut |_|{}).unwrap();
            let mean=entry(&out,&horizon);
            let(p,bound)=(read(&mean["populations"]),mean["estimatedError"].as_f64().unwrap());
            let off=metric(&p,&read(&fixture["exact"][key]));
            assert_eq!(mean["withinTolerance"],json!(true),"{key}");
            assert!(off<=allowed(&out,bound),"p({key}) {off:e} from the exact, bound {bound:e}, allowed {:e}",allowed(&out,bound));
            means.push((p,bound));
        }
        let steady=read(&fixture["exact"]["steady"]);let to_steady=|i:usize|metric(&means[i].0,&steady);
        let bounds:f64=means.iter().map(|m|m.1).sum();
        assert!(to_steady(1)<=to_steady(0)+bounds,"p(1 s) {:e} from the steady state, p(100 ms) {:e}",to_steady(1),to_steady(0));
    }
    /// One mean request returns both standard horizons
    /// (`meanHorizons`: 100 ms and 1 s), each within `allowed` of the
    /// default scene's exact references; another horizon, and the t ->
    /// infinity mean (null), are refused; a full request without the mean
    /// (includeMean false) runs no mean and returns none.
    #[test]
    fn one_mean_request_returns_the_two_standard_horizons(){
        let fixture:Value=serde_json::from_str(include_str!("../tests/fixtures/default_scene_horizons.json")).unwrap();
        let model:Model=serde_json::from_value(fixture["model"].clone()).unwrap();
        let groups:Vec<Vec<usize>>=serde_json::from_value(fixture["groups"].clone()).unwrap();
        let read=|v:&Value|->Vec<f64>{v.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()};
        let metric=|a:&[f64],b:&[f64]|->f64{let d:Vec<f64>=a.iter().zip(b).map(|(x,y)|x-y).collect();
            groups.iter().map(|g|g.iter().map(|&i|d[i]).sum::<f64>().abs()).chain(d.iter().map(|x|x.abs())).fold(0.,f64::max)};
        let mut o=fixture["options"].clone();o["meanOnly"]=json!(true);o["meanHorizonUs"]=json!(1e6);
        let out=solve(&model,&serde_json::from_value(o.clone()).unwrap(),&mut |_|{}).unwrap();
        let entries=out["meanHorizons"].as_array().unwrap();
        assert_eq!(entries.iter().map(|e|e["horizonUs"].clone()).collect::<Vec<_>>(),vec![json!(1e5),json!(1e6)]);
        for(entry,key)in entries.iter().zip(["100000","1000000"]){
            assert_eq!(entry["status"],json!("time-average"),"{key}");
            let(p,bound)=(read(&entry["populations"]),entry["estimatedError"].as_f64().unwrap());
            let off=metric(&p,&read(&fixture["exact"][key]));
            assert_eq!(entry["withinTolerance"],json!(true),"{key}");
            assert!(off<=allowed(&out,bound),"p({key}) {off:e} from the exact, bound {bound:e}, allowed {:e}",allowed(&out,bound));
        }
        // A horizon outside the standard ones is refused, and so is t -> infinity.
        let mut other=o.clone();other["meanHorizonUs"]=json!(3e6);assert!(solve(&model,&serde_json::from_value(other).unwrap(),&mut |_|{}).is_err());
        let mut infinity=o.clone();infinity["meanHorizonUs"]=Value::Null;assert!(serde_json::from_value::<Options>(infinity).is_err());
        // The trajectory alone: no mean run, no mean fields.
        o["meanOnly"]=json!(false);o["includeMean"]=json!(false);o["durationUs"]=json!(1.);o["samples"]=json!(11);
        let alone=solve(&model,&serde_json::from_value(o).unwrap(),&mut |_|{}).unwrap();
        assert!(alone.get("meanHorizons").is_none()&&alone["diagnostics"].get("meanLattice").is_none());
        assert_eq!(alone["populationAccuracy"]["status"],json!("estimated-converged"));
    }
    /// The `meanHorizons` entry of a result at a horizon (us).
    fn entry<'v>(out:&'v Value,horizon:&Value)->&'v Value{
        out["meanHorizons"].as_array().unwrap().iter().find(|e|&e["horizonUs"]==horizon).unwrap()
    }
    /// A mean whose run computed nothing still returns its two standard
    /// horizons, each not converged with the run's reason and its
    /// horizonUs.
    #[test]
    fn a_failed_mean_returns_its_two_horizons_not_converged(){
        let fixture:Value=serde_json::from_str(include_str!("../tests/fixtures/default_scene_horizons.json")).unwrap();
        let model:Model=serde_json::from_value(fixture["model"].clone()).unwrap();
        for horizon in[json!(1e5),json!(1e6)]{
            let mut o=fixture["options"].clone();o["meanOnly"]=json!(true);o["meanHorizonUs"]=horizon.clone();
            let options:Options=serde_json::from_value(o).unwrap();
            let request=prepare_request(&model,&options).unwrap();
            let out=result(&model,&options,&request,None,Some(Err("no usable mean".into())),true,Instant::now()).unwrap();
            let entries=out["meanHorizons"].as_array().unwrap();
            assert_eq!(entries.iter().map(|e|e["horizonUs"].clone()).collect::<Vec<_>>(),vec![json!(1e5),json!(1e6)]);
            assert!(entries.iter().all(|e|e["status"]=="not-converged"&&e["reason"]=="no usable mean"&&e["withinTolerance"]==false&&e["populations"].is_null()));
        }
    }
    /// End-to-end validation of the pipeline against a reference (ignored):
    /// ATOM_VALIDATE_REQUEST=<request.json>, ATOM_VALIDATE_GRID="T:samples"
    /// (absent: no trajectory) and ATOM_VALIDATE_HORIZONS="1e5,1e6" (absent:
    /// none; a request has a trajectory or a horizon),
    /// ATOM_VALIDATE_REFERENCE=<json with times/pops, horizons{T}>,
    /// ATOM_VALIDATE_DUMP=<path> (every checkpoint's outputs and bounds
    /// appended as a JSON line). One JSON line per checkpoint (its bounds and
    /// actual errors: the raw model's against its estimate, the returned
    /// rows' against their per-sample bounds) and a summary line.
    #[test]
    #[ignore]
    fn pipeline_against_a_reference(){
        let var=|k:&str|std::env::var(k).ok().filter(|v|!v.is_empty());
        let request:Value=serde_json::from_str(&std::fs::read_to_string(var("ATOM_VALIDATE_REQUEST").expect("ATOM_VALIDATE_REQUEST")).unwrap()).unwrap();
        let model:Model=serde_json::from_value(request["model"].clone()).unwrap();
        let reference:Value=var("ATOM_VALIDATE_REFERENCE").map_or(Value::Null,|p|serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap());
        let times:Vec<f64>=var("ATOM_VALIDATE_GRID").map_or(vec![],|g|{let(t,s)=g.split_once(':').unwrap();let(t,s):(f64,usize)=(t.parse().unwrap(),s.parse().unwrap());
            (0..s).map(|i|t*i as f64/(s-1)as f64).collect()});
        let horizons:Vec<f64>=var("ATOM_VALIDATE_HORIZONS").map_or(vec![],|h|h.split(',').map(|x|x.parse().unwrap()).collect());
        assert!(!times.is_empty()||!horizons.is_empty(),"ATOM_VALIDATE_GRID or ATOM_VALIDATE_HORIZONS: a request has a trajectory or a horizon");
        let started=Instant::now();
        let options=Options::default();let prepared=prepare_request(&model,&options).unwrap();let prepare_ms=started.elapsed().as_secs_f64()*1e3;
        println!("{}",json!({"preparation":lattice_json(&prepared.lattice,None)}));
        let rows=|v:&Value|->Option<Vec<Vec<f64>>>{v.as_array().map(|a|a.iter().map(|r|r.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()).collect())};
        let reference_pops=rows(&reference["pops"]).filter(|p|p.len()==times.len());
        let reference_horizons:Vec<Option<Vec<f64>>>=horizons.iter().map(|t|{let key=format!("{:.1}",t);
            reference["horizons"].get(&key).and_then(|v|v.as_array()).map(|a|a.iter().map(|x|x.as_f64().unwrap()).collect())}).collect();
        let groups=prepared.groups.clone();let zero=vec![0.;32];
        let dump=var("ATOM_VALIDATE_DUMP");
        let ratios=|bounds:&[f64],errors:&[f64]|->(f64,usize,f64){
            // The smallest bound / actual over the samples whose actual error
            // is at least 2% of the tolerance (below it the references' own
            // accuracy enters), the samples whose actual error exceeds their
            // bound, and the smallest bound / actual over every sample.
            let worst=bounds.iter().zip(errors).filter(|(_,&e)|e>=0.02*POPULATION_TOLERANCE).map(|(b,e)|b/e).fold(f64::INFINITY,f64::min);
            let over=bounds.iter().zip(errors).filter(|(b,e)|e>b).count();
            let all=bounds.iter().zip(errors).filter(|(_,&e)|e>0.).map(|(b,e)|b/e).fold(f64::INFINITY,f64::min);
            (worst,over,all)};
        let mut observe=|c:&Checkpoint,a:&Answer|{
            if let Some(path)=&dump{use std::io::Write;
                let line=json!({"solves":c.solves,"trajectory":c.trajectory,"horizons":c.horizons,
                    "estTrajectory":c.estimate.trajectory,"estHorizons":c.estimate.horizons,"sampleErrors":a.sample_errors,"returned":a.trajectory,
                    "horizonErrors":a.horizon_errors,"accepted":c.accepted});
                let mut f=std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();writeln!(f,"{line}").unwrap();}
            // The raw model's errors (what the estimate bounds) and the returned rows'.
            let raw:Option<Vec<f64>>=reference_pops.as_ref().map(|r|c.trajectory.iter().zip(r).map(|(a,b)|accuracy::metric(a,b,&groups)).collect());
            let returned:Option<Vec<f64>>=reference_pops.as_ref().map(|r|a.trajectory.iter().zip(r).map(|(a,b)|accuracy::metric(a,b,&groups)).collect());
            let at=raw.as_ref().map(|e|{let(i,m)=e.iter().enumerate().fold((0,0_f64),|(bi,bm),(i,&v)|if v>bm{(i,v)}else{(bi,bm)});(m,times[i])});
            let error_h:Vec<Option<f64>>=c.horizons.iter().zip(&reference_horizons).map(|(a,b)|b.as_ref().map(|b|accuracy::metric(a,b,&groups))).collect();
            let est_traj=c.estimate.trajectory.iter().fold(0_f64,|a,&b|a.max(b));
            let quantiles=raw.as_ref().map(|e|{let mut r:Vec<f64>=e.iter().zip(&c.estimate.trajectory).filter(|(a,_)|**a>1e-4).map(|(a,b)|b/a).collect();
                r.sort_by(f64::total_cmp);if r.is_empty(){vec![]}else{vec![r[0],r[r.len()/20],r[r.len()/2],r[(r.len()*19)/20]]}});
            let coverage=returned.as_ref().map(|e|ratios(&a.sample_errors,e));
            let line=json!({"solves":c.solves,"columns":c.columns,"dimension":c.dimension,"accepted":c.accepted,"largest":c.largest,
                "estTraj":est_traj,"estH":c.estimate.horizons,
                "boundTraj":a.sample_errors.iter().fold(0_f64,|x,&y|x.max(y)),"boundH":a.horizon_errors,
                "errTraj":at.map(|e|e.0),"errTrajAt":at.map(|e|e.1),"errH":error_h,"estOverErrQuantiles":quantiles,
                "perTimeMin":coverage.map(|c|c.0),"perTimeViolations":coverage.map(|c|c.1),"perTimeMinAll":coverage.map(|c|c.2),
                "returnedErrTraj":returned.as_ref().map(|e|e.iter().fold(0_f64,|x,&y|x.max(y))),
                "growing":c.growing,"clipped":c.clipped,"projected":c.projected,
                "slowestRate":c.slowest.map(|s|s.rate),"slowestResidual":c.slowest.map(|s|s.residual),"slowestMatched":c.slowest.map(|s|s.matched),"ms":c.milliseconds});
            let work=json!({"operatorWork":c.inner.operator_work,"innerSteps":c.inner.steps,"innerMs":c.inner.milliseconds,"applyMs":c.inner.timing.apply_ms,"preconditionMs":c.inner.timing.precondition_ms,"orthogonalizeMs":c.inner.timing.orthogonalize_ms,"batchMs":c.inner.batch_ms,"blockOperations":c.inner.block_operations(),
                "timingMs":{"explicitSet":c.timing.explicit_set_ms,"apriori":c.timing.apriori_ms,"setup":c.timing.setup_ms,"frame":c.timing.frame_ms.iter().sum::<f64>(),"frameWait":c.timing.frame_wait_ms,"projection":c.timing.projection_ms,"model":c.timing.model_ms,"estimate":c.timing.estimate_ms,"outputs":c.timing.outputs_ms},
                "elapsedMs":started.elapsed().as_secs_f64()*1e3,"trajectoryNorm":c.trajectory.iter().map(|p|accuracy::metric(p,&zero,&groups)).fold(0_f64,f64::max)});
            let mut line=line;if let(Value::Object(a),Value::Object(b))=(&mut line,work){a.extend(b);}
            println!("{line}");
        };
        let result=pipeline::solve(&prepared.lattice,&prepared.density,&Request{times:&times,horizons:&horizons,groups:&prepared.groups,explicit:&std::sync::OnceLock::new(),runs:1},Some(&mut observe));
        let total_ms=started.elapsed().as_secs_f64()*1e3;
        match result{
            Ok(s)=>{let a=&s.answer;
                let returned:Option<Vec<f64>>=reference_pops.as_ref().map(|r|a.trajectory.iter().zip(r).map(|(x,y)|accuracy::metric(x,y,&groups)).collect());
                let coverage=returned.as_ref().map(|e|ratios(&a.sample_errors,e));
                println!("{}",json!({"summary":true,"prepareMs":prepare_ms,"totalMs":total_ms,"lattice":lattice_json(&prepared.lattice,Some(&s))["run"],
                    "converged":s.converged,"reason":s.reason,"solves":a.solves,"bound":a.largest(),
                    "perTimeMin":coverage.map(|c|c.0),"perTimeViolations":coverage.map(|c|c.1),"perTimeMinAll":coverage.map(|c|c.2),
                    "slowestRate":a.slowest.map(|x|x.rate),"slowestResidual":a.slowest.map(|x|x.residual),"slowestResolved":a.slowest.map(|x|x.resolved())}))},
            Err(e)=>println!("{}",json!({"summary":true,"error":e,"totalMs":total_ms})),
        }
    }
}
