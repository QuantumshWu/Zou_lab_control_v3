//! The protocol bridge to the population core, the folded Fourier lattice:
//! every rb87-full/2 request. The TypeScript model is the only owner of
//! atomic physics; this adapter borrows its H/C, exact phase coordinates and
//! state metadata, and serves the finite trajectory with the t -> infinity
//! mean or the mean at the horizon `meanHorizonUs`: every output from one
//! run of the pipeline (`lattice::pipeline`). The population tolerance is
//! the fixed `ledger::POPULATION_TOLERANCE`; a request may only restate it.
//!
//! # Result contract
//! The one result shape, read by `normalizeDynamicsResult` in
//! src/solver-client.ts; main.rs adds `protocol` and `modelSchema`.
//! - `states` (32 ids), `timesUs`, `populations` (one row per time). A mean
//!   result has the single t = 0 row, the initial populations; a finite result
//!   has >= 2 projected rows and `populationAccuracy`, their estimated error.
//! - `horizonUs`: the echo of the request's `meanHorizonUs` (us, or null for
//!   the t -> infinity mean), on every result that carries a mean.
//! - `asymptoticStatus`. A converged mean (`stationary`, `population-stationary`
//!   and `initial-dependent` for static models; `time-average` and
//!   `initial-dependent-time-average` for driven ones) comes with
//!   `asymptoticReason`, `asymptoticPopulations` and `asymptoticDensityReal/Imag`;
//!   otherwise the status is `not-converged` (with its `asymptoticReason`) or
//!   `not-requested`.
//! - `diagnostics`. The reader gates the rows' `traceError` and
//!   `initialStateError`, `elapsedMs`, a finite result's `method`, and a mean's
//!   `asymptoticMethod`, `asymptoticTraceError`, `asymptoticPositivityError` and
//!   `residual`, and shows `slowestRelaxationUs` and `conservedQuantities`;
//!   every other key is profiling (the lattice's under `diagnostics.lattice`).
//!   A finite result carries the mean's diagnostics under the same names.
use crate::{Model,Options,harmonics};
use serde_json::{json,Value};
use std::collections::BTreeMap;
use std::time::Instant;
use atom_population_core::{Complex64,Lapack};
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts};
use atom_population_core::lattice::LatticePrepared;
use atom_population_core::lattice::classification::Treatment;
use atom_population_core::lattice::estimate_api::ACCEPTANCE_FACTOR;
use atom_population_core::lattice::ledger::{self,POPULATION_TOLERANCE};
use atom_population_core::lattice::pipeline::{self,Request,Solution};

const ENGINE:&str="rust-population-core-lattice";
const METHOD:&str="Full32 density-matrix Lindblad dynamics on the folded Fourier lattice (explicit charges by mixing, seed amplitude and ring condition; their ring eliminated exactly at every pole; far harmonics folded statically, higher orders charged): one rational Krylov space from the initial state, every output read from one reduced model in the dressed coordinates, its error estimated from the embedded residual";
const MEAN_FRAME:&str="laboratory time average in the background eigenbasis";

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
    Ok((LatticePrepared::prepare(&view,phases)?,axes))
}

/// One request's lattice preparation, shared by its mean and its trajectory.
struct Prepared{
    lattice:LatticePrepared,axes:harmonics::PhaseAxes,
    /// Requested groups (indices and names).
    groups:Vec<Vec<usize>>,names:Vec<String>,
    /// rho0 (row-major) and its populations.
    density:Vec<Complex64>,initial:Vec<f64>,
}
fn prepare_request(model:&Model,options:&Options)->Result<Prepared,String>{
    if options.population_tolerance!=POPULATION_TOLERANCE{
        return Err(format!("populationTolerance is fixed at {POPULATION_TOLERANCE} (requested {})",options.population_tolerance));}
    if options.mean_horizon_us.is_some_and(|h|!(h.is_finite()&&h>0.)){return Err("meanHorizonUs must be a finite positive time in us, or null".into());}
    if model.dimension!=32||model.states.len()!=32{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    let(ground,requested)=groups(model)?;
    let(lattice,axes)=prepare(model,&ground)?;
    let n=32;
    let density=initial_density(model,&lattice.lapack)?;
    let initial=(0..n).map(|i|density[i*n+i].re).collect();
    Ok(Prepared{lattice,axes,groups:requested.iter().map(|g|g.indices.clone()).collect(),names:requested.iter().map(|g|g.name.clone()).collect(),density,initial})
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
        "restStored":[t.rest[0].stored(),t.rest[1].stored()],"denseSectors":[t.rest[0].blocks.len(),t.rest[1].blocks.len()],
        "norms":t.norms,"dropped":t.dropped,"largestEntry":t.largest_entry})).collect();
    let k=&prepared.floquet.diagnostics;
    let floquet=json!({"milliseconds":k.milliseconds,"iterations":k.iterations,"change":k.change,
        "pairs":k.pairs,"guarded":k.guarded,"smallParameter":k.small_parameter,"frameCondition":k.condition,
        "support":k.support.iter().map(|(q,size)|json!({"charge":q,"kernel":size})).collect::<Vec<_>>(),"belowFloor":k.below,"precessingRate":k.precessing_rate});
    let run=run.map(|s|{let e=&s.explicit;let w=&s.work;let t=&s.timing;let l=&s.ledger;
        json!({"explicit":e.sets.explicit,"ring":e.sets.ring,"gated":e.gated,"promoted":e.promoted,"covered":e.covered,"leak":e.leak,
            "ringInternalEdges":e.internal.len(),"explicitSetMs":e.milliseconds,"translations":s.translations,
            "columns":s.columns.iter().map(|c|json!({"pole":c.pole.to_string(),"innerSteps":c.steps,"residual":c.residual,"added":c.added,"ms":c.milliseconds})).collect::<Vec<_>>(),
            "basis":s.images,"dimension":s.dimension,
            "inner":{"solves":w.solves,"steps":w.steps,"appliedBlocks":w.applied_blocks,"preconditionedBlocks":w.preconditioned_blocks,
                "blockOperations":w.block_operations(),"imageBlocks":s.images*s.modes.len(),"batches":w.batches,"batchMs":w.batch_ms,
                "applyMs":w.timing.apply_ms,"preconditionMs":w.timing.precondition_ms,"orthogonalizeMs":w.timing.orthogonalize_ms},
            "checkpoints":s.checkpoints.iter().map(|c|json!({"solves":c.solves,"columns":c.columns,"dimension":c.dimension,"accepted":c.accepted,
                "trajectory":c.estimate.trajectory.iter().fold(0_f64,|a,&b|a.max(b)),"horizons":c.estimate.horizons,"steady":c.estimate.steady,
                "growing":c.growing,"reflected":c.reflected,"projected":c.projected,"conserved":c.conserved,"slowestRate":c.slowest.map(|s|s.rate),
                "estimateMs":c.estimate.work.milliseconds,"ms":c.milliseconds})).collect::<Vec<_>>(),
            "ledger":{"nuFreeze":l.nu_freeze,"thirdOrder":l.third_order,"farShell":l.far_shell,"tail":l.tail,"omittedCross":l.omitted_cross,
                "nonStructural":l.non_structural,"droppedEntries":l.dropped_entries,"total":l.total(),"fold":l.fold,"expansionParameter":l.expansion_parameter},
            "timingMs":{"explicitSet":t.explicit_set_ms,"setup":t.setup_ms,"solves":t.solves_ms,"images":t.images_ms,"dressed":t.dressed_ms,
                "model":t.model_ms,"ordering":t.ordering_ms,"estimate":t.estimate_ms,"total":t.total_ms}})});
    json!({
        "harmonicPlan":{"explicit":plan.count(&Treatment::Explicit{structural:true}),"nonStructural":plan.count(&Treatment::Explicit{structural:false}),
            "folded":plan.count(&Treatment::Fold),"gap":plan.gap,"cut":plan.cut,"l0Bound":plan.l0_bound,"structuralFloor":plan.structural_floor,
            "steps":plan.steps.iter().map(|s|json!({"charge":s.charge,"nu":s.nu,"size":s.size,"treatment":format!("{:?}",s.treatment)})).collect::<Vec<_>>(),
            "foldTargets":plan.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>(),
            "omittedTargets":plan.omitted.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>()},
        "fold":{"farSources":f.far_sources,"columns":f.columns,"restBound":f.rest_bound,"chargedTruncation":f.truncation,
            "series":f.series.iter().map(|(q,rho,order)|json!({"charge":q,"ratio":rho,"orders":order})).collect::<Vec<_>>(),"milliseconds":f.milliseconds,
            "phaseMs":{"columns":f.phase_ms[0],"storage":f.phase_ms[1]},"targets":targets},
        "floquet":floquet,"eigenbasis":prepared.eigenbasis().ok().map(|e|json!({"milliseconds":e.milliseconds,"blocks":e.blocks,"largestBlock":e.largest_block,"condition":e.condition})),"run":run})
}

/// One run of the pipeline for a request: the grid (empty for a mean
/// alone) and the mean's horizon when one is asked for.
fn run(request:&Prepared,times:&[f64],horizon:Option<f64>)->Result<Solution,String>{
    let horizons:Vec<f64>=horizon.into_iter().collect();
    pipeline::solve(&request.lattice,&request.density,&Request{times,horizons:&horizons,groups:&request.groups},None)
}

/// The mean of a run: the laboratory time average of the steady state, or
/// of the state at the horizon, projected onto the density matrices (the
/// change charged); `not-converged` with its reason when the run failed or
/// its estimate leaves the tolerance.
fn mean_value(model:&Model,options:&Options,request:&Prepared,run:Result<&Solution,&String>,started:Instant)->Result<Value,String>{
    let(prepared,axes)=(&request.lattice,&request.axes);let n=32;
    let rows=vec![request.initial.clone()];let(row_trace,initial_error)=row_errors(&rows,&request.initial)?;
    let preparation=json!({"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2],"floquetKernels":prepared.timing[3]});
    let not_converged=|reason:String,solution:Option<&Solution>|json!({"timesUs":[0.],"states":model.states,"populations":rows,
        "horizonUs":options.mean_horizon_us,"asymptoticStatus":"not-converged","asymptoticReason":reason,
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,"elapsedMs":started.elapsed().as_secs_f64()*1000.,
            "carrierSourceToBasis":axes.source_to_basis,"asymptoticMethod":METHOD,"preparationMs":preparation,
            "lattice":lattice_json(prepared,solution)}});
    let solution=match run{Ok(s)=>s,Err(e)=>return Ok(not_converged(e.clone(),None))};
    let horizon=options.mean_horizon_us;
    let(field,error)=match horizon{
        Some(_)=>(&solution.horizon_fields[0],solution.estimate.horizons[0]),
        None=>(&solution.steady_field,solution.estimate.steady.unwrap_or(f64::INFINITY))};
    let(raw,discarded)=lab_density(field,&solution.modes,axes);
    let diagonal=|m:&[Complex64]|(0..n).map(|i|m[i*n+i].re).collect::<Vec<f64>>();
    let density=project_density(&prepared.lapack,&raw)?;let projection=ledger::metric(&diagonal(&raw),&diagonal(&density),&request.groups);
    let estimated=error+projection;
    if ACCEPTANCE_FACTOR*error+projection>POPULATION_TOLERANCE{
        return Ok(not_converged(format!("The mean is estimated at {estimated:e}; the tolerance was not relaxed"),Some(solution)));}
    let populations=diagonal(&density);
    let positivity=(-prepared.lapack.hermitian_eigen(&density,n)?.0[0]).max(0.);
    let trace_error=(populations.iter().sum::<f64>()-1.).abs();
    // At a horizon, populations still relaxing by more than a twentieth of
    // the tolerance (or several conserved quantities) keep the mean
    // initial-state dependent.
    let relaxing=horizon.map(|_|ledger::metric(&solution.horizons[0],&solution.steady,&request.groups));
    let dependent=solution.conserved>1||relaxing.is_some_and(|r|r>0.05*POPULATION_TOLERANCE);
    let(status,reason)=match(model.drives.is_empty(),dependent){
        (false,true)=>("initial-dependent-time-average","Long-time phase average at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent."),
        (true,true)=>("initial-dependent","Lindblad mean of the specified initial density: slow modes still relaxing at the horizon (horizonUs), or conserved quantities, keep it initial-state dependent."),
        (false,false)=>("time-average","Converged long-time phase average of the full driven Lindblad equation (unique limit: only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time); populations are a time average, not a time-independent density."),
        (true,false)if discarded>1e-8=>("population-stationary","Populations converge in the background basis; exported density is the laboratory-frame time average."),
        (true,false)=>("stationary","Unique full-density Lindblad stationary solution (only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time)."),
    };
    Ok(json!({"timesUs":[0.],"states":model.states,"populations":rows,"horizonUs":horizon,
        "asymptoticStatus":status,"asymptoticReason":reason,"asymptoticPopulations":populations,
        "asymptoticDensityReal":density.iter().map(|z|z.re).collect::<Vec<_>>(),
        "asymptoticDensityImag":density.iter().map(|z|z.im).collect::<Vec<_>>(),
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":axes.source_to_basis,
            "asymptoticMethod":METHOD,"meanDensityFrame":MEAN_FRAME,
            "uniquenessScope":if solution.conserved>1{"several conserved quantities: mean of the specified initial density"}else{"unique limit: every mode but the trace relaxes, however slowly (slowestRelaxationUs)"},
            "asymptoticTraceError":trace_error,"asymptoticPositivityError":positivity,"residual":solution.steady_residual,
            "asymptoticAccuracy":{"requestedTolerance":POPULATION_TOLERANCE,"estimatedMaximumError":estimated,"acceptanceFactor":ACCEPTANCE_FACTOR,
                "projectionCharge":projection,"metric":"max over states and requested F groups"},
            "zeroModeCount":solution.conserved,"conservedQuantities":solution.conserved,
            "slowestRatePerUs":solution.slowest.map(|s|s.rate),"slowestRelaxationUs":solution.slowest.map(|s|1./s.rate),
            "slowestRelaxation":solution.slowest.map(|s|json!({"ratePerUs":s.rate,"residualPerUs":s.residual,"resolved":s.resolved(),
                "relaxationRangeUs":[1./(s.rate+s.residual),if s.rate>s.residual{Some(1./(s.rate-s.residual))}else{None}],
                "rule":"the decaying Ritz value of least rate outside the kernel; resolved when its residual is within a tenth of the rate, else the range from the residual"})),
            "discardedOpticalCoherence":discarded,"horizonRelaxingBound":relaxing,
            "preparationMs":preparation,"lattice":lattice_json(prepared,Some(solution))}}))
}

/// The lattice core's mean result: the t -> infinity mean, or the mean at
/// the horizon `meanHorizonUs`.
pub fn mean_result(model:&Model,options:&Options,started:Instant)->Result<Value,String>{
    let request=prepare_request(model,options)?;
    let solution=run(&request,&[],options.mean_horizon_us);
    mean_value(model,options,&request,solution.as_ref(),started)
}

/// A request through the lattice core: the mean alone (steadyOnly), or the
/// trajectory with the mean (published once ready) merged into the finite
/// result, both from one run.
pub fn solve(model:&Model,options:&Options,publish_mean:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();
    if options.steady_only{return mean_result(model,options,started);}
    if !options.duration_us.is_finite()||options.duration_us<=0.||!(2..=2001).contains(&options.samples){return Err("Invalid finite population grid".into());}
    let request=prepare_request(model,options)?;
    let times:Vec<f64>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    let solution=run(&request,&times,options.mean_horizon_us.filter(|_|options.include_asymptotic))?;
    let mean=if options.include_asymptotic{let mean=mean_value(model,options,&request,Ok(&solution),started)?;publish_mean(&mean);Some(mean)}else{None};
    let(row_trace,initial_error)=row_errors(&solution.trajectory,&request.initial)?;
    let estimated=solution.estimate.trajectory.iter().fold(0_f64,|a,&b|a.max(b))+solution.projection;
    if ACCEPTANCE_FACTOR*(estimated-solution.projection)+solution.projection>POPULATION_TOLERANCE{
        return Err(format!("Finite-time populations estimated at {estimated:e}; the tolerance was not relaxed"));}
    let prepared=&request.lattice;
    let mut result=json!({"states":model.states,"timesUs":times,"populations":solution.trajectory,"asymptoticStatus":"not-requested",
        "populationAccuracy":{"status":"estimated-converged","requestedTolerance":POPULATION_TOLERANCE,
            "estimatedMaximumError":estimated,"projectionCharge":solution.projection,
            "includesProjectionCharge":true,"scope":{"states":32,"samples":options.samples,"groups":request.names}},
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":request.axes.source_to_basis,"method":METHOD,
            "preparationMs":{"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2],"floquetKernels":prepared.timing[3]},
            "lattice":lattice_json(prepared,Some(&solution))}});
    if let Some(Value::Object(mean))=mean{
        for(key,value)in mean{
            if key=="diagnostics"{if let(Value::Object(target),Value::Object(mean))=(&mut result["diagnostics"],value){for(key,value)in mean{target.entry(key).or_insert(value);}}}
            else if key.starts_with("asymptotic")||key=="horizonUs"{result[key.as_str()]=value;}
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests{
    use super::*;
    /// Validation dump for the Python comparison (ignored; needs
    /// ATOM_LATTICE_DUMP_REQUEST=<request.json> and ATOM_LATTICE_DUMP_DIR).
    /// Writes K_0^(0) (unthresholded), every stored K_t^(o) as row-major
    /// little-endian complex128 d x d, the folded background Hamiltonian H_2
    /// (n x n) and `index.json` with charges, frequencies and diagnostics.
    #[test]
    #[ignore]
    fn lattice_dump_fold_for_python_comparison(){
        let path=std::env::var("ATOM_LATTICE_DUMP_REQUEST").expect("ATOM_LATTICE_DUMP_REQUEST");
        let dir=std::path::PathBuf::from(std::env::var("ATOM_LATTICE_DUMP_DIR").expect("ATOM_LATTICE_DUMP_DIR"));
        std::fs::create_dir_all(&dir).unwrap();
        let request:Value=serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let model:Model=serde_json::from_value(request["model"].clone()).unwrap();
        let(ground,_)=groups(&model).unwrap();
        let clock=Instant::now();let(p,_)=prepare(&model,&ground).unwrap();let ms=clock.elapsed().as_secs_f64()*1e3;
        let write=|name:&str,m:&[Complex64]|{
            let bytes:Vec<u8>=m.iter().flat_map(|z|z.re.to_le_bytes().into_iter().chain(z.im.to_le_bytes())).collect();
            std::fs::write(dir.join(name),bytes).unwrap();};
        let(n,d)=(32,1024);
        let targets:Vec<Value>=p.fold.targets.iter().enumerate().map(|(t,data)|{
            let files=[format!("common_t{t}_o0.bin"),format!("common_t{t}_o1.bin")];
            for(o,file)in files.iter().enumerate(){
                let m=if t==0&&o==0{p.fold.zero_dense.clone().unwrap()}else{let mut m=vec![Complex64::default();d*d];data.dense_into(n,o,Complex64{re:1.,im:0.},&mut m);m};
                write(file,&m);}
            json!({"charge":data.charge,"nu":data.nu,"files":files})}).collect();
        let(h2,rest)=p.folded_background_hamiltonian().unwrap();write("common_h2.bin",&h2);
        let far:Vec<Value>=p.plan.steps.iter().filter(|s|s.treatment==Treatment::Fold).map(|s|json!({"charge":s.charge,"nu":s.nu,"size":s.size})).collect();
        let index=json!({"common":{"targets":targets,"corrections":[],"far":far,"h2":"common_h2.bin","h2RestShare":rest,
            "foldMs":p.timing[2],"prepareMs":ms,"diagnostics":{"restBound":p.fold.diagnostics.rest_bound,"phaseMs":p.fold.diagnostics.phase_ms,
            "series":p.fold.diagnostics.series.iter().map(|(q,rho,order)|json!({"charge":q,"ratio":rho,"order":order})).collect::<Vec<_>>()}}});
        std::fs::write(dir.join("index.json"),serde_json::to_string_pretty(&index).unwrap()).unwrap();
    }
    /// The lattice operator on a request against a per-component reference
    /// (every mode computed, fold targets as dense matrices); ignored,
    /// `validation` feature, ATOM_LATTICE_REQUEST=<request.json>.
    #[cfg(feature="validation")]
    #[test]
    #[ignore]
    fn lattice_operator_against_its_components_on_a_request(){
        use atom_population_core::lattice::operator::LatticeOperator;
        let path=std::env::var("ATOM_LATTICE_REQUEST").expect("ATOM_LATTICE_REQUEST");
        let request:Value=serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let model:Model=serde_json::from_value(request["model"].clone()).unwrap();
        let(ground,_)=groups(&model).unwrap();let(p,_)=prepare(&model,&ground).unwrap();
        let modes=p.modes(2).unwrap();let mut op=LatticeOperator::new(&p,&modes).unwrap();
        let(n,d)=(32,1024);let c=|re:f64,im:f64|Complex64{re,im};
        let mut x:Vec<Complex64>=(0..modes.len()*d).map(|k|c((((k*7+3)%29) as f64/29.-0.5)*1e-3,(((k*11+5)%31) as f64/31.-0.5)*1e-3)).collect();
        for q in 0..modes.len(){let m=op.mirror[q];if q>m{continue;}
            for i in 0..n{for j in 0..n{if q==m&&j<i{continue;}let v=x[q*d+i*n+j];let w=if q==m&&i==j{c(v.re,0.)}else{v};x[q*d+i*n+j]=w;x[m*d+j*n+i]=c(w.re,-w.im);}}}
        let mut y=vec![c(0.,0.);x.len()];op.apply(&x,&mut y).unwrap();
        let mut reference=vec![c(0.,0.);x.len()];let mut small=Vec::new();
        let add=|a:&mut Complex64,b:Complex64|{a.re+=b.re;a.im+=b.im;};
        for(q,mode)in modes.iter().enumerate(){
            p.generator.zero_component().op.apply_into(n,c(1.,0.),&x[q*d..(q+1)*d],&mut reference[q*d..(q+1)*d],&mut small);
            for k in 0..d{let v=x[q*d+k];add(&mut reference[q*d+k],c(op.nu[q]*v.im,-op.nu[q]*v.re));}
            for s in p.plan.explicit(){let t=&p.plan.steps[s].charge;
                let source:Vec<i32>=mode.iter().zip(t).map(|(a,b)|a-b).collect();let Some(&s0)=op.index.get(&source)else{continue};
                p.generator.components[t].op.apply_into(n,c(1.,0.),&x[s0*d..(s0+1)*d],&mut reference[q*d..(q+1)*d],&mut small);}
        }
        let mut dense=vec![c(0.,0.);d*d];
        for t in &p.fold.targets{for order in 0..2{
            dense.iter_mut().for_each(|v|*v=c(0.,0.));t.dense_into(n,order,c(1.,0.),&mut dense);
            for(q,mode)in modes.iter().enumerate(){
                let source:Vec<i32>=mode.iter().zip(&t.charge).map(|(a,b)|a-b).collect();let Some(&s0)=op.index.get(&source)else{continue};
                let scale=if order==0{c(1.,0.)}else{c(0.,op.nu[s0])};
                for r in 0..d{let mut acc=c(0.,0.);for k in 0..d{let(a,b)=(dense[r*d+k],x[s0*d+k]);acc.re+=a.re*b.re-a.im*b.im;acc.im+=a.re*b.im+a.im*b.re;}
                    add(&mut reference[q*d+r],c(scale.re*acc.re-scale.im*acc.im,scale.re*acc.im+scale.im*acc.re));}
            }}}
        // Linearity and repeatability of the operator and the preconditioner.
        let x2:Vec<Complex64>=(0..x.len()).map(|k|{let m=k/d;let r=k%d;let(i,j)=(r/n,r%n);let src=op.mirror[m]*d+j*n+i;
            let v=x[if op.canonical[m]{k}else{src}];let w=c(v.im*0.7+v.re*0.1,v.re*0.3-v.im*0.2);if op.canonical[m]{w}else{c(w.re,-w.im)}}).collect();
        let mut x2=x2;for q in 0..modes.len(){let m=op.mirror[q];if q!=m{continue;}for i in 0..n{for j in 0..=i{let v=x2[q*d+j*n+i];let w=if i==j{c(v.re,0.)}else{v};x2[q*d+j*n+i]=w;x2[q*d+i*n+j]=c(w.re,-w.im);}}}
        for q in 0..modes.len(){if op.canonical[q]{continue;}let m=op.mirror[q];for i in 0..n{for j in 0..n{let v=x2[m*d+j*n+i];x2[q*d+i*n+j]=c(v.re,-v.im);}}}
        let sum:Vec<Complex64>=x.iter().zip(&x2).map(|(a,b)|c(a.re+b.re,a.im+b.im)).collect();
        let linear=|f:&mut dyn FnMut(&[Complex64],&mut [Complex64])|{
            let(mut a,mut b,mut s,mut a2)=(vec![c(0.,0.);x.len()],vec![c(0.,0.);x.len()],vec![c(0.,0.);x.len()],vec![c(0.,0.);x.len()]);
            f(&x,&mut a);f(&x2,&mut b);f(&sum,&mut s);f(&x,&mut a2);
            let scale=s.iter().map(|v|v.re.hypot(v.im)).fold(0.,f64::max);
            let nonlinear=s.iter().zip(a.iter().zip(&b)).map(|(s,(a,b))|(s.re-a.re-b.re).hypot(s.im-a.im-b.im)).fold(0.,f64::max)/scale;
            let repeat=a.iter().zip(&a2).map(|(p,q)|(p.re-q.re).hypot(p.im-q.im)).fold(0.,f64::max)/scale;(nonlinear,repeat)};
        let operator_linearity=linear(&mut |u,v|op.apply(u,v).unwrap());
        use atom_population_core::lattice::preconditioner::{Preconditioner,Preconditioning,Shift};
        let pre=Preconditioning::new(&p).unwrap();
        let mut batch=pre.batch(&op,Shift::Steady,Some(atom_population_core::lattice::floquet::Reach::All)).unwrap();
        let preconditioner_linearity=linear(&mut |u,v|batch.apply(u,v).unwrap());
        println!("{}",json!({"operatorLinearity":operator_linearity,"preconditionerLinearity":preconditioner_linearity}));
        let size=reference.iter().map(|v|v.re.hypot(v.im)).fold(0.,f64::max);
        let mut worst=(0.,0,0);
        for(k,(a,b))in y.iter().zip(&reference).enumerate(){let e=(a.re-b.re).hypot(a.im-b.im);if e>worst.0{worst=(e,k/d,k%d);}}
        println!("{}",json!({"request":path,"modes":modes.len(),"largest":size,"worstError":worst.0,"worstMode":worst.1,"worstEntry":worst.2,
            "canonical":op.canonical,"charges":modes}));
    }
    /// End-to-end validation of the pipeline against a reference (ignored):
    /// ATOM_VALIDATE_REQUEST=<request.json>, ATOM_VALIDATE_GRID="T:samples"
    /// (absent: no trajectory), ATOM_VALIDATE_HORIZONS="1e5,1e6",
    /// ATOM_VALIDATE_REFERENCE=<json with times/pops, steady, horizons{T}>.
    /// One JSON line per checkpoint (its
    /// estimate and actual errors) and a summary line.
    #[test]
    #[ignore]
    fn pipeline_against_a_reference(){
        use atom_population_core::lattice::pipeline::Checkpoint;
        let var=|k:&str|std::env::var(k).ok().filter(|v|!v.is_empty());
        let request:Value=serde_json::from_str(&std::fs::read_to_string(var("ATOM_VALIDATE_REQUEST").expect("ATOM_VALIDATE_REQUEST")).unwrap()).unwrap();
        let model:Model=serde_json::from_value(request["model"].clone()).unwrap();
        let reference:Value=var("ATOM_VALIDATE_REFERENCE").map_or(Value::Null,|p|serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap());
        let times:Vec<f64>=var("ATOM_VALIDATE_GRID").map_or(vec![],|g|{let(t,s)=g.split_once(':').unwrap();let(t,s):(f64,usize)=(t.parse().unwrap(),s.parse().unwrap());
            (0..s).map(|i|t*i as f64/(s-1)as f64).collect()});
        let horizons:Vec<f64>=var("ATOM_VALIDATE_HORIZONS").map_or(vec![],|h|h.split(',').map(|x|x.parse().unwrap()).collect());
        let started=Instant::now();
        let options=Options::default();let prepared=prepare_request(&model,&options).unwrap();let prepare_ms=started.elapsed().as_secs_f64()*1e3;
        println!("{}",json!({"preparation":lattice_json(&prepared.lattice,None)}));
        let rows=|v:&Value|->Option<Vec<Vec<f64>>>{v.as_array().map(|a|a.iter().map(|r|r.as_array().unwrap().iter().map(|x|x.as_f64().unwrap()).collect()).collect())};
        let reference_pops=rows(&reference["pops"]).filter(|p|p.len()==times.len());
        let reference_steady:Option<Vec<f64>>=reference["steady"].as_array().map(|a|a.iter().map(|x|x.as_f64().unwrap()).collect());
        let reference_horizons:Vec<Option<Vec<f64>>>=horizons.iter().map(|t|{let key=format!("{:.1}",t);
            reference["horizons"].get(&key).and_then(|v|v.as_array()).map(|a|a.iter().map(|x|x.as_f64().unwrap()).collect())}).collect();
        let groups=prepared.groups.clone();let zero=vec![0.;32];
        let mut lines=Vec::new();
        let mut observe=|c:&Checkpoint|{
            let error_traj=reference_pops.as_ref().map(|r|{let e:Vec<f64>=c.trajectory.iter().zip(r).map(|(a,b)|ledger::metric(a,b,&groups)).collect();
                let(i,m)=e.iter().enumerate().fold((0,0_f64),|(bi,bm),(i,&v)|if v>bm{(i,v)}else{(bi,bm)});(m,times[i],e)});
            let error_h:Vec<Option<f64>>=c.horizons.iter().zip(&reference_horizons).map(|(a,b)|b.as_ref().map(|b|ledger::metric(a,b,&groups))).collect();
            let error_s=reference_steady.as_ref().map(|r|ledger::metric(&c.steady,r,&groups));
            let est_traj=c.estimate.trajectory.iter().fold(0_f64,|a,&b|a.max(b));
            let ratio=error_traj.as_ref().map(|(_,_,e)|{let mut r:Vec<f64>=e.iter().zip(&c.estimate.trajectory).filter(|(a,_)|**a>1e-4).map(|(a,b)|b/a).collect();
                r.sort_by(f64::total_cmp);if r.is_empty(){vec![]}else{vec![r[r.len()/20],r[r.len()/2],r[(r.len()*19)/20]]}});
            let line=json!({"solves":c.solves,"columns":c.columns,"dimension":c.dimension,"accepted":c.accepted,"estTraj":est_traj,"estH":c.estimate.horizons,"estSteady":c.estimate.steady,
                "errTraj":error_traj.as_ref().map(|e|e.0),"errTrajAt":error_traj.as_ref().map(|e|e.1),"errH":error_h,"errSteady":error_s,"estOverErrQuantiles":ratio,
                "growing":c.growing,"reflected":c.reflected,"projected":c.projected,"conserved":c.conserved,"slowestRate":c.slowest.map(|s|s.rate),"slowestResidual":c.slowest.map(|s|s.residual),"ms":c.milliseconds,
                "operatorWork":c.inner.operator_work,"innerSteps":c.inner.steps,"innerMs":c.inner.milliseconds,"applyMs":c.inner.timing.apply_ms,"preconditionMs":c.inner.timing.precondition_ms,"orthogonalizeMs":c.inner.timing.orthogonalize_ms,"batchMs":c.inner.batch_ms,"blockOperations":c.inner.block_operations(),
                "timingMs":{"explicitSet":c.timing.explicit_set_ms,"setup":c.timing.setup_ms,"dressed":c.timing.dressed_ms,"model":c.timing.model_ms,"ordering":c.timing.ordering_ms,"estimate":c.timing.estimate_ms},
                "elapsedMs":started.elapsed().as_secs_f64()*1e3,"trajectoryNorm":c.trajectory.iter().map(|p|ledger::metric(p,&zero,&groups)).fold(0_f64,f64::max)});
            println!("{line}");lines.push(line);
        };
        let result=pipeline::solve(&prepared.lattice,&prepared.density,&Request{times:&times,horizons:&horizons,groups:&prepared.groups},Some(&mut observe));
        let total_ms=started.elapsed().as_secs_f64()*1e3;
        match result{
            Ok(s)=>println!("{}",json!({"summary":true,"prepareMs":prepare_ms,"totalMs":total_ms,"lattice":lattice_json(&prepared.lattice,Some(&s))["run"],
                "conserved":s.conserved,"slowestRate":s.slowest.map(|x|x.rate),"slowestResidual":s.slowest.map(|x|x.residual),"steadyResidual":s.steady_residual})),
            Err(e)=>println!("{}",json!({"summary":true,"error":e,"totalMs":total_ms})),
        }
    }
    /// The error estimator (`lattice::estimator`) on R2's dumped spaces (N1's
    /// pipeline, `research/population/unified/r2/n2_dump.py` on branch
    /// claude/unified-estimate), exported by
    /// `research/population/unified/est/est_export.py` (summary:
    /// `est_summary.py` on the output): estimate against the actual error
    /// and against R2's Python estimate. Ignored; ATOM_ESTIMATOR_FIXTURES =
    /// the export root, ATOM_ESTIMATOR_CASES = comma-separated case names,
    /// optional: ATOM_ESTIMATOR_SEED (0: no seed; else the steady solve's
    /// directions seed the slow pool, as the pipeline's hook will),
    /// ATOM_ESTIMATOR_GATED (1: a closed gate skips the estimate, so the work
    /// is the estimator's own), ATOM_ESTIMATOR_RUST_IMAGES (1: the images by
    /// the Rust operator, which differs from the dumped ones by
    /// `operatorDeviation`), ATOM_ESTIMATOR_RANK (1: singular values of the
    /// weighted Ritz sources), ATOM_ESTIMATOR_NEAR (one-step: the explicit
    /// set not among the near modes; all: every mode near),
    /// ATOM_ESTIMATOR_FASTRING, ATOM_ESTIMATOR_OUTER (variants of the fast
    /// part). One JSON line per checkpoint; the fold charges (added by the
    /// Estimator) are left out.
    #[test]
    #[ignore]
    fn lattice_estimator_against_r2_fixtures(){
        use atom_population_core::lattice::estimator::{self,Checkpoint,OutputGate,UnifiedEstimate};
        use atom_population_core::lattice::estimate_api::{EstimateSetup,LatticeSets,Outputs};
        use atom_population_core::lattice::ledger::RemainderLedger;
        use atom_population_core::lattice::operator::LatticeOperator;
        use atom_population_core::lattice::preconditioner::{Preconditioning,Shift};
        use atom_population_core::lattice::floquet::Reach;
        use atom_population_core::lattice::krylov::{self,KrylovOptions,KrylovTiming,LinearSystem,System,Vectors};
        type C=Complex64;
        // The pipeline's steady solve as the pool's seed (R2 section 12): its
        // preconditioned directions z_j recorded (canonical modes, mirrored
        // here), their images A z_j computed here (the pipeline has them from
        // its Arnoldi relation; not counted).
        struct Recording<'r,'s,'a>{system:&'r mut System<'s,'a>,directions:Vec<Vec<C>>}
        impl LinearSystem for Recording<'_,'_,'_>{
            fn length(&self)->usize{self.system.length()}
            fn vectors(&self)->&Vectors{self.system.vectors()}
            fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{self.system.apply(x,y)}
            fn precondition(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
                self.system.precondition(x,y)?;let mut z=y.to_vec();self.system.vectors().mirror(&mut z);self.directions.push(z);Ok(())}
            fn timing(&mut self)->&mut KrylovTiming{self.system.timing()}
        }
        let root=std::path::PathBuf::from(std::env::var("ATOM_ESTIMATOR_FIXTURES").expect("ATOM_ESTIMATOR_FIXTURES"));
        let cases=std::env::var("ATOM_ESTIMATOR_CASES").expect("ATOM_ESTIMATOR_CASES");
        let seed=std::env::var("ATOM_ESTIMATOR_SEED").map_or(true,|v|v!="0");
        let gated=std::env::var("ATOM_ESTIMATOR_GATED").is_ok_and(|v|v=="1");
        let complex=|bytes:&[u8]|->Vec<C>{bytes.chunks_exact(16).map(|b|C{re:f64::from_le_bytes(b[..8].try_into().unwrap()),im:f64::from_le_bytes(b[8..].try_into().unwrap())}).collect()};
        let charges=|v:&Value|->Vec<Vec<i32>>{v.as_array().unwrap().iter().map(|q|q.as_array().unwrap().iter().map(|k|k.as_i64().unwrap() as i32).collect()).collect()};
        let floats=|v:&Value|->Vec<f64>{v.as_array().map(|a|a.iter().map(|x|x.as_f64().unwrap_or(f64::NAN)).collect()).unwrap_or_default()};
        for case in cases.split(','){
            let dir=root.join(case);
            let index:Value=serde_json::from_str(&std::fs::read_to_string(dir.join("index.json")).unwrap()).unwrap();
            let request:Value=serde_json::from_str(&std::fs::read_to_string(index["request"].as_str().unwrap()).unwrap()).unwrap();
            let model:Model=serde_json::from_value(request["model"].clone()).unwrap();
            let(ground,_)=groups(&model).unwrap();let(p,_)=prepare(&model,&ground).unwrap();
            let(explicit,ring)=(charges(&index["P"]),charges(&index["Q"]));
            let modes:Vec<Vec<i32>>=explicit.iter().chain(&ring).cloned().collect();
            let fixture_groups:Vec<Vec<usize>>=index["groups"].as_array().unwrap().iter().map(|g|g.as_array().unwrap().iter().map(|i|i.as_u64().unwrap() as usize).collect()).collect();
            let(times,horizons)=(floats(&index["times"]),floats(&index["horizons"]));
            let dim=index["dim"].as_u64().unwrap() as usize;
            let w=complex(&std::fs::read(dir.join("W.bin")).unwrap());let aw=complex(&std::fs::read(dir.join("AW.bin")).unwrap());
            let basis:Vec<Vec<C>>=w.chunks_exact(dim).map(|c|c.to_vec()).collect();let images:Vec<Vec<C>>=aw.chunks_exact(dim).map(|c|c.to_vec()).collect();
            drop(w);drop(aw);
            // The Rust operator against the dumped images (same lattice, same model).
            let mut op=LatticeOperator::new(&p,&modes).unwrap();let mut y=vec![C::default();dim];let mut operator=0_f64;
            for j in [0,basis.len()/2,basis.len()-1]{op.apply(&basis[j],&mut y).unwrap();
                let e=y.iter().zip(&images[j]).map(|(a,b)|(a.re-b.re).hypot(a.im-b.im)).fold(0.,f64::max);
                operator=operator.max(e/images[j].iter().map(|v|v.re.hypot(v.im)).fold(0.,f64::max));}
            let ledger=RemainderLedger::new(&p.plan,op.nu_max(),op.dropped()).unwrap();
            // The images by the Rust operator (the fixtures' are Python's, which differ by `operatorDeviation`).
            let images:Vec<Vec<C>>=if std::env::var("ATOM_ESTIMATOR_RUST_IMAGES").is_ok_and(|v|v=="1"){basis.iter().map(|w|{let mut y=vec![C::default();dim];op.apply(w,&mut y).unwrap();y}).collect()}else{images};
            drop(op);
            let sets=LatticeSets{explicit,ring};let initial=vec![C::default();p.generator.n*p.generator.n];
            let setup=EstimateSetup{prepared:&p,lattice:&sets,outputs:Outputs{times:&times,horizons:&horizons,steady:true,groups:&fixture_groups},ledger:&ledger,initial:&initial};
            let clock=Instant::now();
            let near=std::env::var("ATOM_ESTIMATOR_NEAR").unwrap_or_default();
            let all_sets=LatticeSets{explicit:modes.clone(),ring:Vec::new()};
            let mut estimate=if near=="all"{
                // Every mode near: the explicit set given as the whole lattice (the setup's lattice is only read for its modes and P).
                UnifiedEstimate::with_near(EstimateSetup{lattice:&all_sets,..setup},true).unwrap()
            }else{UnifiedEstimate::with_near(setup,near!="one-step").unwrap()};
            estimate.lattice.ring_in_fast=std::env::var("ATOM_ESTIMATOR_FASTRING").is_ok_and(|v|v=="1");
            estimate.lattice.outer_bare=std::env::var("ATOM_ESTIMATOR_OUTER").map_or(true,|v|v!="0");
            let build_ms=clock.elapsed().as_secs_f64()*1e3;
            let n_modes=modes.len();
            // The steady solve (u = I/n on the centre, R2's tolerance 1e-9, no restart) and its directions.
            let(mut seed_count,mut seed_steps)=(0,0);
            if seed{
                let mut op=LatticeOperator::new(&p,&modes).unwrap();
                let mut pre=Preconditioning::new(&p).unwrap().batch(&op,Shift::Steady,Some(Reach::All)).unwrap();
                let size=op.dimension();let n=p.generator.n;let centre=sets.centre().unwrap();
                let mut u=vec![C::default();size];for i in 0..n{u[centre*n*n+i*n+i]=C{re:1./n as f64,im:0.};}
                let directions={let mut system=System::new(&mut op,&mut pre,Shift::Steady).unwrap();
                    let mut recording=Recording{system:&mut system,directions:Vec::new()};
                    let mut options=KrylovOptions::new(1e-9);options.restart=200;options.maximum_steps=200;
                    let solve=krylov::gmres(&mut recording,&u,None,&options).unwrap();seed_steps=solve.steps;
                    // The last preconditioning forms the iterate (x = P^-1 V y), in the span of the others.
                    let mut directions=recording.directions;directions.pop();directions};
                let images:Vec<Vec<C>>=directions.iter().map(|z|{let mut y=vec![C::default();size];op.apply(z,&mut y).unwrap();y}).collect();
                seed_count=directions.len();
                estimate.seed(&directions,&images).unwrap();
            }
            // The pipeline's real poles in the slow region (its ladder: 10 / t1 down to
            // 1 / T, two per decade, those below slow::SLOW), solved on the first basis
            // column at its inner tolerance, as further seeds.
            let mut pole_seeds=0;
            if std::env::var("ATOM_ESTIMATOR_POLE_SEEDS").is_ok_and(|v|v=="1"){
                let t1=times.iter().copied().find(|&t|t>0.).unwrap_or(1.);let longest=times.iter().chain(&horizons).fold(0_f64,|a,&b|a.max(b));
                let mut sigma=10./t1;let mut poles=Vec::new();
                while sigma>=(1./longest)*(1.-1e-9){if sigma<=estimator::slow::SLOW{poles.push(sigma);}sigma/=10_f64.sqrt();}
                if let Some(top)=std::env::var("ATOM_ESTIMATOR_POLE_MAX").ok().and_then(|v|v.parse::<f64>().ok()){poles.retain(|&s|s<=top);}
                if let Some(keep)=std::env::var("ATOM_ESTIMATOR_POLE_KEEP").ok().and_then(|v|v.parse::<usize>().ok()){poles.truncate(keep);}
                for sigma in poles{
                    let mut op=LatticeOperator::new(&p,&modes).unwrap();
                    let mut pre=Preconditioning::new(&p).unwrap().batch(&op,Shift::Real(sigma),Some(Reach::All)).unwrap();
                    let size=op.dimension();
                    let directions={let mut system=System::new(&mut op,&mut pre,Shift::Real(sigma)).unwrap();
                        let mut recording=Recording{system:&mut system,directions:Vec::new()};
                        let mut options=KrylovOptions::new(1e-7);options.restart=200;options.maximum_steps=200;
                        krylov::gmres(&mut recording,&basis[0],None,&options).unwrap();
                        let mut directions=recording.directions;directions.pop();directions};
                    let images:Vec<Vec<C>>=directions.iter().map(|z|{let mut y=vec![C::default();size];op.apply(z,&mut y).unwrap();y}).collect();
                    pole_seeds+=directions.len();estimate.seed(&directions,&images).unwrap();
                }
            }
            let seed_macs=estimate.pool.as_ref().map_or(0.,|p|p.work.qr_macs);
            println!("{}",json!({"case":case,"modes":n_modes,"near":estimate.lattice.near.len(),"ring":estimate.lattice.ring.len(),"operatorDeviation":operator,"buildMs":build_ms,
                "kernelMs":estimate.lattice.build_ms[0],"classMs":estimate.lattice.build_ms[1],"classes":estimate.lattice.classes.len(),"largestCluster":estimate.lattice.classes.iter().map(|c|c.largest_cluster).max(),"eigenbasisMs":p.eigenbasis().unwrap().milliseconds,"eigenBlocks":p.eigenbasis().unwrap().blocks,"largestBlock":p.eigenbasis().unwrap().largest_block,
                "seeds":seed_count,"seedSteps":seed_steps,"poleSeeds":pole_seeds,"seedMacs":seed_macs,"d":p.generator.n*p.generator.n,"kernelEntries":estimate.lattice.kernels.diagnostics.entries,"smallParameter":estimate.lattice.kernels.diagnostics.largest,"ledger":ledger.total()}));
            let mut gate=OutputGate::new(p.generator.n,estimate.lattice.nu.clone(),estimate.lattice.centre);let limit=estimator::gate::GATE_FACTOR*POPULATION_TOLERANCE;
            let zero=vec![0.;p.generator.n];
            for cp in index["checkpoints"].as_array().unwrap(){
                let(k,r,solves)=(cp["k"].as_u64().unwrap() as usize,cp["r"].as_u64().unwrap() as usize,cp["solves"].as_u64().unwrap());
                let data=complex(&std::fs::read(dir.join(format!("cp{solves:03}.bin"))).unwrap());
                let mut at=0;let mut take=|len:usize|{let s=data[at..at+len].to_vec();at+=len;s};
                let(x,s,b,_g)=(take(k*r),take(r*r),take(k*r),take(r*r));let(_c0,_th,th_e,_a,a_e,_ker,us)=(take(r),take(r),take(r),take(r),take(r),take(r),take(r));
                // Diagnostics: singular values of the weighted Ritz sources a_m (A W - theta_m W) b_m (all modes, fast modes).
                let rank=if std::env::var("ATOM_ESTIMATOR_RANK").is_ok_and(|v|v=="1"){
                    let gemm=p.lapack.complex_gemm();let(one,zero_c)=(C{re:1.,im:0.},C::default());let cm=|a:C,b:C|C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re};
                    let mut bm=vec![zero_c;k*r];gemm.multiply(k,r,r,one,&x,k,&s,r,zero_c,&mut bm,k).unwrap();
                    let(wcat,acat):(Vec<C>,Vec<C>)=(basis[..k].concat(),images[..k].concat());
                    let spectrum=|fast:bool|->Value{
                        let(mut ba,mut bta)=(vec![zero_c;k*r],vec![zero_c;k*r]);
                        for m in 0..r{let a=if fast&&th_e[m].re.hypot(th_e[m].im)<estimator::slow::SLOW{zero_c}else{a_e[m]};let ta=cm(th_e[m],a);
                            for i in 0..k{ba[i+k*m]=cm(bm[i+k*m],a);bta[i+k*m]=cm(bm[i+k*m],ta);}}
                        let mut f=vec![zero_c;dim*r];gemm.multiply(dim,r,k,one,&acat,dim,&ba,k,zero_c,&mut f,dim).unwrap();
                        gemm.multiply(dim,r,k,C{re:-1.,im:0.},&wcat,dim,&bta,k,one,&mut f,dim).unwrap();
                        let mut fh=vec![zero_c;dim*r];for m in 0..r{for i in 0..dim{fh[m+r*i]=C{re:f[i+dim*m].re,im:-f[i+dim*m].im};}}
                        let mut g=vec![zero_c;r*r];gemm.multiply(r,r,dim,one,&fh,r,&f,dim,zero_c,&mut g,r).unwrap();
                        let(vals,_)=p.lapack.hermitian_eigen(&g,r).unwrap();
                        let mut sv:Vec<f64>=vals.iter().map(|v|v.max(0.).sqrt()).collect();sv.sort_by(|a,b|b.total_cmp(a));
                        let top=sv[0].max(1e-300);
                        let counts:Vec<usize>=[1e-1,1e-2,1e-3,1e-4,1e-6].iter().map(|&t|sv.iter().filter(|&&v|v>t*top).count()).collect();
                        json!({"top":sv[0],"counts":counts,
                            "nonzero":(0..r).filter(|&m|!(fast&&th_e[m].re.hypot(th_e[m].im)<estimator::slow::SLOW)&&a_e[m].re.hypot(a_e[m].im)>0.).count()})};
                    json!({"r":r,"all":spectrum(false),"fast":spectrum(true)})
                }else{Value::Null};
                // The gate as the estimator runs it; in the gated mode a closed
                // gate skips the estimate (its columns are processed at the next
                // open checkpoint), so the work is the estimator's own.
                gate.add_columns(&basis[..k]);
                let mul=|a:C,b:C|C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re};
                let c:Vec<C>=(0..k).map(|i|us.iter().enumerate().fold(C::default(),|v,(l,&u)|{let t=mul(x[i+k*l],u);C{re:v.re+t.re,im:v.im+t.im}})).collect();
                let outputs=gate.outputs(&p.lapack,&b,&th_e,&a_e,&times,&horizons,Some(&c)).unwrap();
                let change=gate.change(outputs,&fixture_groups);
                let open=change.is_some_and(|v|v<=limit)||gate.last_estimate.is_some_and(|v|v<=limit);
                let skipped=gated&&!open;
                let before=(estimate.lattice.work.clone(),estimate.pool.as_ref().map(|p|p.work.clone()).unwrap_or_default());
                let clock=Instant::now();
                let errors=if skipped{estimator::Errors::default()}else{estimate.errors(&Checkpoint{basis:&basis[..k],images:&images[..k],x:&x,s:&s,theta:&th_e,coefficients:&a_e}).unwrap()};
                let ms=clock.elapsed().as_secs_f64()*1e3;
                let series:Vec<f64>=errors.trajectory.iter().map(|p|ledger::metric(p,&zero,&fixture_groups)).collect();
                let hz:Vec<f64>=errors.horizons.iter().map(|p|ledger::metric(p,&zero,&fixture_groups)).collect();
                let steady_est=errors.steady.as_ref().map(|p|ledger::metric(p,&zero,&fixture_groups)).unwrap_or(0.);
                let largest=series.iter().fold(steady_est,|a,&b|a.max(b));if open{gate.last_estimate=Some(largest);}
                let(lw,pw)=(&estimate.lattice.work,estimate.pool.as_ref().map(|p|p.work.clone()).unwrap_or_default());
                let mut row=json!({"case":case,"solves":solves,"k":k,"ms":ms,"estMax":series.iter().fold(0_f64,|a,&b|a.max(b)),"steadyEst":steady_est,"horizons":hz,
                    "slowModes":errors.slow_modes,"partsMs":errors.milliseconds,"slowTrajectory":errors.slow_trajectory,"lambdaSlow":errors.lambda_slow,
                    "gate":{"change":change,"open":open},"skipped":skipped,
                    "work":{"columns":lw.columns-before.0.columns,"columnMs":lw.column_ms-before.0.column_ms,"frameBlocks":lw.frame_blocks-before.0.frame_blocks,
                        "ringOperations":lw.block_operations-before.0.block_operations,"ringSolves":lw.block_solves-before.0.block_solves,
                        "poolApplications":pw.applications-before.1.applications,"poolPreconditions":pw.preconditions-before.1.preconditions,"poolSteps":pw.steps-before.1.steps,
                        "poolSolves":pw.solves-before.1.solves,"poolFrames":pw.frame_blocks-before.1.frame_blocks,"poolMacs":pw.qr_macs-before.1.qr_macs,"pipelineSolves":0,
                        "pipelineSteps":0,"pipelineMs":0,
                        "extendMs":pw.extend_ms-before.1.extend_ms,"stepMs":pw.step_ms-before.1.step_ms,"solveMs":pw.solve_ms-before.1.solve_ms},
                    "count":cp["count"].clone(),"modes":n_modes,"rank":rank});
                if let Some(actual)=cp["actual"].as_array().filter(|_|!skipped){
                    let actual:Vec<f64>=actual.iter().map(|v|v.as_f64().unwrap()).collect();
                    let select:Vec<bool>=cp["select"].as_array().unwrap().iter().map(|v|v.as_u64()==Some(1)).collect();
                    let chosen:Vec<usize>=(0..actual.len()).filter(|&i|select[i]).collect();
                    let amax=actual.iter().fold(0_f64,|a,&b|a.max(b));
                    let ratio=if chosen.is_empty(){f64::NAN}else{chosen.iter().map(|&i|series[i]).fold(0.,f64::max)/chosen.iter().map(|&i|actual[i]).fold(0.,f64::max)};
                    let mut ratios:Vec<f64>=chosen.iter().map(|&i|series[i]/actual[i]).collect();ratios.sort_by(f64::total_cmp);
                    let quantile=|q:f64|if ratios.is_empty(){f64::NAN}else{let pos=q*(ratios.len()-1) as f64;let(lo,hi)=(pos.floor() as usize,pos.ceil() as usize);ratios[lo]+(ratios[hi]-ratios[lo])*(pos-lo as f64)};
                    row["actualMax"]=json!(amax);row["ratio"]=json!(ratio);row["quantiles"]=json!([quantile(0.05),quantile(0.5),quantile(0.95),ratios.first().copied().unwrap_or(f64::NAN)]);
                    row["steadyActual"]=cp["err_steady"].clone();row["horizonActual"]=cp["err_h"].clone();
                }
                if let Some(py)=cp.get("python").and_then(|v|v.as_object()).filter(|_|!skipped){
                    for(variant,v)in py{let s=floats(&v["series"]);if s.len()!=series.len(){continue;}
                        let size=s.iter().fold(0_f64,|a,&b|a.max(b));
                        let dev=s.iter().zip(&series).map(|(a,b)|(a-b).abs()).fold(0.,f64::max)/size.max(1e-300);
                        row["python"][variant]=json!({"ratio":v["ratio"],"deviation":dev,"max":size});}
                    if let Some(st)=cp.get("python_steady"){row["pythonSteady"]=st.clone();}
                }
                println!("{row}");
            }
        }
    }
}
