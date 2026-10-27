//! The protocol bridge to the population core, the folded Fourier lattice:
//! every rb87-full/2 request. The TypeScript model is the only owner of
//! atomic physics; this adapter borrows its H/C, exact phase coordinates and
//! state metadata, and serves the t -> infinity mean or the mean at the
//! horizon `meanHorizonUs`, and the finite trajectory (with its mean
//! alongside; beyond `finite::LONG_HORIZON` by the slow-mode expansion).
//! The population tolerance is the fixed `ledger::POPULATION_TOLERANCE`; a
//! request may only restate it.
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
//!   every other key is profiling (the lattice's under `diagnostics.lattice`
//!   and `diagnostics.finite`). A finite result carries the mean's
//!   diagnostics under the same names; keys both report keep the finite value.
use crate::{Model,Options,harmonics};
use serde_json::{json,Value};
use std::collections::BTreeMap;
use std::time::Instant;
use atom_population_core::{Complex64,Lapack};
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts};
use atom_population_core::lattice::LatticePrepared;
use atom_population_core::lattice::classification::Treatment;
use atom_population_core::lattice::ledger::{self,KAPPA,POPULATION_TOLERANCE};
use atom_population_core::lattice::steady::{self,Acceptance,Attempt,SteadySolution};
use atom_population_core::lattice::finite::{self,Declined};
use atom_population_core::lattice::slow::Purpose;

const ENGINE:&str="rust-population-core-lattice";
const MEAN_METHOD:&str="Full32 time average of the folded Fourier lattice (near charges explicit, far harmonics folded statically at second order, higher orders charged): Floquet-frame preconditioned null vector, its truncation and fold errors estimated in the solver's Krylov space";
const MEAN_FRAME:&str="laboratory time average in the background eigenbasis";
const STATIC_METHOD:&str="Full32 static Lindblad generator: spectral projectors of the imaginary-axis eigenvalues of each connected block applied to the initial density";
const FINITE_METHOD:&str="Full32 density-matrix Lindblad evolution on the folded Fourier lattice: rational Krylov with lattice-translation augmentation, its basis, near-shell and fold errors estimated by propagating their sources through the dressed blocks; beyond 1 ms the slow eigenmodes' expansion from the Krylov state at a switch time";

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

fn attempt_json(a:&Attempt)->Value{
    let charge=a.fold.as_ref().map(|c|{let p=&c.charges;json!({"source":c.source,"response":c.response,"sourceBound":c.source_bound,
        "sensitivity":p.sensitivity,"total":p.total(),"nuFreeze":p.nu_freeze,"thirdOrder":p.third_order,"farShell":p.far_shell,"tail":p.tail,
        "omittedCross":p.omitted_cross,"nonStructural":p.non_structural,"droppedEntries":p.dropped_entries})});
    let(t,k,l)=(&a.timing,&a.timing.krylov,&a.ledger);
    json!({"order":a.order,"modes":a.modes,"canonicalModes":a.canonical,"nuMax":a.nu_max,"steps":a.steps,"restarts":a.restarts,"stop":a.stop,
        "primalResidual":a.residual,"populations":a.populations,"truncationEstimate":a.truncation,"truncationResidual":a.truncation_residual,
        "augmentationSteps":a.augmentation,"equations":a.equations.iter().map(|(l,k,v)|json!({"label":l,"steps":k,"estimate":v})).collect::<Vec<_>>(),"shellApplications":a.shell_applications,"shellModes":a.shell_modes,
        "corrections":a.corrections,"contraction":a.contraction,"claimed":a.claimed,"foldCharge":charge,
        "applications":a.applications,"preconditions":a.preconditions,
        "operatorMacPerApplication":a.operator_work,"preconditionerMacPerApplication":a.precondition_work,
        "regularizedDenominators":a.regularized,"edges":{"explicit":a.edges[0],"crossFold":a.edges[1]},
        "timingMs":{"setup":t.setup_ms,"primal":t.primal_ms,"estimate":t.estimate_ms,"correction":t.correction_ms,
            "estimateParts":{"shell":t.estimate_parts[0],"truncationSource":t.estimate_parts[1],"foldSource":t.estimate_parts[2],
                "augmentation":t.estimate_parts[3],"checkedSolution":t.estimate_parts[4]},
            "apply":k.apply_ms,"precondition":k.precondition_ms,"orthogonalize":k.orthogonalize_ms},
        "ledger":{"nuFreeze":l.nu_freeze,"thirdOrder":l.third_order,"farShell":l.far_shell,"tail":l.tail,
            "omittedCross":l.omitted_cross,"nonStructural":l.non_structural,"droppedEntries":l.dropped_entries,"total":l.total(),
            "fold":l.fold,"expansionParameter":l.expansion_parameter}})
}

/// The lattice profiling: plan, fold, preconditioner, slow modes and attempts.
fn lattice_json(prepared:&LatticePrepared,solution:Option<&SteadySolution>)->Value{
    let plan=&prepared.plan;let f=&prepared.fold.diagnostics;
    let targets:Vec<Value>=prepared.fold.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,
        "restStored":[t.rest[0].stored(),t.rest[1].stored()],"denseSectors":[t.rest[0].blocks.len(),t.rest[1].blocks.len()],
        "norms":t.norms,"dropped":t.dropped,"largestEntry":t.largest_entry})).collect();
    let k=&prepared.floquet.diagnostics;
    let floquet=json!({"milliseconds":k.milliseconds,"iterations":k.iterations,"change":k.change,
        "pairs":k.pairs,"guarded":k.guarded,"smallParameter":k.small_parameter,"frameCondition":k.condition,
        "support":k.support.iter().map(|(q,size)|json!({"charge":q,"kernel":size})).collect::<Vec<_>>(),"belowFloor":k.below,"precessingRate":k.precessing_rate});
    json!({
        "harmonicPlan":{"explicit":plan.count(&Treatment::Explicit{structural:true}),"nonStructural":plan.count(&Treatment::Explicit{structural:false}),
            "folded":plan.count(&Treatment::Fold),"gap":plan.gap,"cut":plan.cut,"l0Bound":plan.l0_bound,"structuralFloor":plan.structural_floor,
            "steps":plan.steps.iter().map(|s|json!({"charge":s.charge,"nu":s.nu,"size":s.size,"treatment":format!("{:?}",s.treatment)})).collect::<Vec<_>>(),
            "foldTargets":plan.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>(),
            "omittedTargets":plan.omitted.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>()},
        "fold":{"farSources":f.far_sources,"columns":f.columns,"restBound":f.rest_bound,
            "series":f.series.iter().map(|(q,rho,order)|json!({"charge":q,"ratio":rho,"order":order})).collect::<Vec<_>>(),"milliseconds":f.milliseconds,
            "phaseMs":{"columns":f.phase_ms[0],"storage":f.phase_ms[1]},"targets":targets},
        "floquet":floquet,
        "preconditioner":solution.map(|s|json!({"kind":"floquet","woodburyMs":s.preconditioner.milliseconds,"woodburyRank":s.preconditioner.woodbury_rank})),
        "slowModes":solution.map(|s|{let m=&s.slow;json!({"conserved":m.conserved,"slowestRate":m.slowest_rate,"artifacts":m.artifacts,
            "artifactGrowth":m.artifact_growth,"solveSteps":m.solve_steps,"milliseconds":m.milliseconds,
            "kept":m.modes.iter().map(|x|json!({"lambda":[x.lambda.re,x.lambda.im],"residual":x.residual})).collect::<Vec<_>>(),
            "ritz":m.ritz.iter().map(|r|json!({"lambda":[r.lambda.re,r.lambda.im],"change":r.change,"residual":r.residual})).collect::<Vec<_>>()})}),
        "attempts":solution.map(|s|s.attempts.iter().map(attempt_json).collect::<Vec<_>>())})
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
    let(values,vectors)=lapack.hermitian_eigen(&column,n)?;let weights=finite::simplex(&values);
    let mut out=vec![Complex64::default();n*n];
    for(k,&w)in weights.iter().enumerate(){if w==0.{continue;}for i in 0..n{let a=vectors[i+n*k];for j in 0..n{let b=vectors[j+n*k];
        out[i*n+j].re+=w*(a.re*b.re+a.im*b.im);out[i*n+j].im+=w*(a.im*b.re-a.re*b.im);}}}
    Ok(out)
}

/// The steady solution a request needs: for a horizon mean or a long
/// trajectory the long-time one (raised order instead of the shell
/// correction, slow modes up to the gap), else the report's.
fn steady_solution(request:&Prepared,long:bool)->Result<SteadySolution,String>{
    steady::solve(&request.lattice,&request.groups,if long{Purpose::LongTime}else{Purpose::Report})
}

/// A horizon T with T times the tail rate at least TAIL_EXPONENT has the
/// limit as its mean: the deviation from the limit (at most 2 in trace
/// norm) has decayed by e^-30 = 1e-13, so an eigenbasis condition up to 1e8
/// leaves under 1e-2 of the tolerance.
const TAIL_EXPONENT:f64=30.;
/// The rate bounding what still relaxes: the reported slowest relaxation
/// (`SteadySolution::slowest_rate`).
fn tail_rate(steady:&SteadySolution)->Option<f64>{steady.slowest_rate()}
fn relaxed_by(steady:&SteadySolution,horizon:f64)->bool{tail_rate(steady).is_some_and(|r|r*horizon>=TAIL_EXPONENT)}

/// The state at the horizon T: its field (on the steady solution's modes),
/// the estimated error of its populations, the populations still relaxing
/// at T and how it was reached.
struct HorizonState{field:Vec<Complex64>,error:f64,relaxing:f64,details:Value}
/// Relaxed by T (`relaxed_by`): the limit, its tail charged. Else beyond
/// LONG_HORIZON the long-time expansion (a trajectory's, else its own); up
/// to it, or when the expansion misses slow modes, the Krylov propagation
/// to T.
fn horizon_state(request:&Prepared,steady:&SteadySolution,horizon:f64,long:Option<Result<&finite::LongTime,&Declined>>)->Result<HorizonState,String>{
    let groups=&request.groups;
    if let Some(rate)=tail_rate(steady).filter(|_|relaxed_by(steady,horizon)){
        let tail=(-rate*horizon).exp();
        return Ok(HorizonState{field:steady.field.clone(),error:steady.estimated_error+tail,relaxing:tail,
            details:json!({"method":"the limit, relaxed by the horizon","tailRatePerUs":rate,"tailBound":tail,"steadyEstimate":steady.estimated_error})});
    }
    if horizon<=finite::LONG_HORIZON{return propagated_state(request,steady,horizon,None);}
    let own;
    let long=match long{Some(l)=>l,None=>{own=finite::long_time(&request.lattice,&request.density,groups,steady,None);own.as_ref()}};
    let long=match long{Ok(l)if horizon>=l.switch=>l,Ok(l)=>return propagated_state(request,steady,horizon,Some(format!("meanHorizonUs lies before the switch time {} us",l.switch))),
        Err(Declined::MissedModes(e))=>return propagated_state(request,steady,horizon,Some(e.clone())),Err(Declined::Failed(e))=>return Err(e.clone())};
    Ok(HorizonState{field:long.field(horizon),error:long.error(horizon),relaxing:long.relaxing(horizon,groups),
        details:json!({"method":"slow expansion of the Krylov state at the switch time","switchUs":long.switch,"krylovEstimate":long.krylov.estimated_error,
            "outsideSlowModes":long.fast,"fitDrift":long.drift,"eigenResiduals":long.residuals,"artifactGrowth":long.artifact_growth,
            "steadyEstimate":long.steady_error,"krylovColumns":long.krylov.columns,"krylovOrder":long.krylov.order,"reachUs":long.times.last(),"rejectedSwitches":long.trail})})
}
/// The state at T by the Krylov propagation to T, carried to the steady
/// modes (a part left out charged; `declined`: why the long-time expansion
/// was not used).
fn propagated_state(request:&Prepared,steady:&SteadySolution,horizon:f64,declined:Option<String>)->Result<HorizonState,String>{
    let groups=&request.groups;
    let times:Vec<f64>=(0..finite::SWITCH_SAMPLES).map(|i|horizon*i as f64/(finite::SWITCH_SAMPLES-1)as f64).collect();
    let mut krylov=finite::propagate(&request.lattice,&request.density,&times,groups,None,&[finite::SWITCH_SAMPLES-1])
        .map_err(|e|match &declined{Some(d)=>format!("{d}; the propagation to the horizon: {e}"),None=>e})?;
    let(field,_,left)=finite::transfer(&request.lattice,&krylov.fields.pop().ok_or("The propagation keeps the field at the horizon")?,
        &request.lattice.modes(krylov.order)?,&steady.modes,horizon,groups)?;
    let readout=finite::Readout::new(&request.lattice,&steady.modes)?;
    let relative:Vec<Complex64>=field.iter().zip(&steady.field).map(|(a,b)|Complex64{re:a.re-b.re,im:a.im-b.im}).collect();
    let relaxing=ledger::metric(&readout.populations(&relative,horizon),&[0.;32],groups);
    Ok(HorizonState{error:krylov.estimated_error+left,relaxing,
        details:json!({"method":"Krylov propagation to the horizon","krylovEstimate":krylov.estimated_error,"krylovColumns":krylov.columns,"krylovOrder":krylov.order,"expansionDeclined":declined}),
        field})
}

/// The lattice core's mean result: the t -> infinity mean, or the mean at
/// the horizon `meanHorizonUs`.
pub fn mean_result(model:&Model,options:&Options,started:Instant)->Result<Value,String>{
    let request=prepare_request(model,options)?;mean_of(model,options,&request,started)
}
/// The mean of a static model (`steady::static_mean`): exact, any number of
/// conserved quantities (the initial state's projection onto them,
/// `initial-dependent` when there are several). None when it does not serve
/// the request (a block too large for the dense decomposition, still
/// relaxing at the horizon, or a kernel too close to a slow relaxation for
/// its error bound): the steady pipeline then does.
fn static_value(model:&Model,options:&Options,request:&Prepared,started:Instant)->Option<Value>{
    let(prepared,axes)=(&request.lattice,&request.axes);let n=32;
    let mean=steady::static_mean(prepared,&request.density,options.mean_horizon_us).ok()?;
    (||->Result<Value,String>{
        let rows=vec![request.initial.clone()];let(row_trace,initial_error)=row_errors(&rows,&request.initial)?;
        let(raw,discarded)=lab_density(&mean.density,&[vec![0;axes.frame_indices[0].len()]],axes);
        let density=project_density(&prepared.lapack,&raw)?;
        let diagonal=|m:&[Complex64]|(0..n).map(|i|m[i*n+i].re).collect::<Vec<f64>>();
        let projection=ledger::metric(&diagonal(&raw),&diagonal(&density),&request.groups);
        let estimated=KAPPA*mean.error+mean.tail+projection;
        let populations=diagonal(&density);
        let positivity=(-prepared.lapack.hermitian_eigen(&density,n)?.0[0]).max(0.);
        let horizon=options.mean_horizon_us.is_some();
        let(status,reason)=match(mean.conserved>1,horizon,discarded>1e-8){
            (true,true,_)=>("initial-dependent","Lindblad mean at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent."),
            (true,false,_)=>("initial-dependent","Full Lindblad stationary projection from the specified initial density."),
            (false,_,true)=>("population-stationary","Populations converge in the background basis; exported density is the laboratory-frame time average."),
            (false,_,false)=>("stationary","Unique full-density Lindblad stationary solution (only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time)."),
        };
        if estimated>POPULATION_TOLERANCE{return Err(format!("The static mean is estimated at {estimated:e}; the tolerance was not relaxed"));}
        Ok(json!({"timesUs":[0.],"states":model.states,"populations":rows,"horizonUs":options.mean_horizon_us,
            "asymptoticStatus":status,"asymptoticReason":reason,"asymptoticPopulations":populations,
            "asymptoticDensityReal":density.iter().map(|z|z.re).collect::<Vec<_>>(),
            "asymptoticDensityImag":density.iter().map(|z|z.im).collect::<Vec<_>>(),
            "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
                "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":axes.source_to_basis,
                "asymptoticMethod":STATIC_METHOD,"meanDensityFrame":MEAN_FRAME,
                "uniquenessScope":if mean.conserved>1{"several conserved quantities: mean of the specified initial density"}else{"unique limit: every mode but the trace relaxes, however slowly (slowestRelaxationUs)"},
                "asymptoticTraceError":(populations.iter().sum::<f64>()-1.).abs(),"asymptoticPositivityError":positivity,"residual":mean.residual,
                "asymptoticAccuracy":{"requestedTolerance":POPULATION_TOLERANCE,"estimatedMaximumError":estimated,"projectionCharge":projection,
                    "kernelError":mean.error,"tail":mean.tail,"safetyFactor":KAPPA,"metric":"max over states and requested F groups"},
                "zeroModeCount":mean.conserved,"conservedQuantities":mean.conserved,
                "slowestRatePerUs":mean.slowest_rate,"slowestRelaxationUs":mean.slowest_rate.map(|r|1./r),
                "discardedOpticalCoherence":discarded,"horizonRelaxingBound":horizon.then_some(mean.tail),
                "horizonMethod":horizon.then(||json!({"method":"spectral projectors of the static generator's imaginary-axis eigenvalues"})),
                "staticMean":{"blocks":mean.blocks,"largestBlock":mean.largest_block,"milliseconds":mean.milliseconds}}}))
    })().ok()
}

/// The mean with its own steady solution: the report's, unless a horizon
/// the found relaxation has not passed needs the long-time one.
fn mean_of(model:&Model,options:&Options,request:&Prepared,started:Instant)->Result<Value,String>{
    if model.drives.is_empty(){if let Some(value)=static_value(model,options,request,started){return Ok(value);}}
    let report=steady_solution(request,false);
    let limit=match(options.mean_horizon_us,&report){(None,_)=>true,(Some(t),Ok(s))=>relaxed_by(s,t),(Some(_),Err(_))=>false};
    if limit{return mean_value(model,options,request,report.as_ref(),None,started);}
    let steady=steady_solution(request,true);
    mean_value(model,options,request,steady.as_ref(),None,started)
}

/// The mean: one t = 0 row of initial populations and the `asymptotic*`
/// fields (`not-converged` with its reason when no mean is accepted). At a
/// horizon T it is the laboratory time average of the state at T, projected
/// onto the density matrices (the change charged); `long` is a trajectory's
/// long-time state (or why it failed) to reuse.
fn mean_value(model:&Model,options:&Options,request:&Prepared,steady:Result<&SteadySolution,&String>,long:Option<Result<&finite::LongTime,&Declined>>,started:Instant)->Result<Value,String>{
    let(prepared,axes)=(&request.lattice,&request.axes);
    let n=32;
    let rows=vec![request.initial.clone()];let(row_trace,initial_error)=row_errors(&rows,&request.initial)?;
    let preparation=json!({"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2]});
    let not_converged=|reason:String,solution:Option<&SteadySolution>|json!({"timesUs":[0.],"states":model.states,"populations":rows,
        "horizonUs":options.mean_horizon_us,"asymptoticStatus":"not-converged","asymptoticReason":reason,
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,"elapsedMs":started.elapsed().as_secs_f64()*1000.,
            "carrierSourceToBasis":axes.source_to_basis,"asymptoticMethod":MEAN_METHOD,"preparationMs":preparation,
            "lattice":lattice_json(prepared,solution)}});
    let solution=match steady{Ok(s)=>s,Err(e)=>return Ok(not_converged(e.clone(),None))};
    let spectrum=&solution.slow;
    if spectrum.conserved>1{
        return Ok(not_converged(format!("{} conserved quantities: the mean depends on the initial state, which the lattice core does not serve",spectrum.conserved),Some(solution)));}
    // The state: the steady field, or the state at the horizon.
    let state=match options.mean_horizon_us{
        Some(horizon)=>match horizon_state(request,solution,horizon,long){Ok(s)=>Some(s),Err(e)=>return Ok(not_converged(e,Some(solution)))},
        None=>None};
    let(raw,discarded)=lab_density(state.as_ref().map_or(&solution.field,|s|&s.field),&solution.modes,axes);
    let diagonal=|m:&[Complex64]|(0..n).map(|i|m[i*n+i].re).collect::<Vec<f64>>();
    let(density,projection)=match &state{
        Some(_)=>{let density=project_density(&prepared.lapack,&raw)?;let charge=ledger::metric(&diagonal(&raw),&diagonal(&density),&request.groups);(density,charge)}
        None=>(raw,0.)};
    let estimated=state.as_ref().map_or(solution.estimated_error,|s|s.error+projection);
    if estimated>POPULATION_TOLERANCE{
        return Ok(not_converged(format!("The mean at the horizon is estimated at {estimated:e}; the tolerance was not relaxed"),Some(solution)));}
    let populations=diagonal(&density);
    let positivity=(-prepared.lapack.hermitian_eigen(&density,n)?.0[0]).max(0.);
    let trace_error=(populations.iter().sum::<f64>()-1.).abs();
    let accepted=solution.attempts.last().ok_or("An accepted lattice solution has its attempt")?;
    let solve_ms:f64=solution.attempts.iter().map(|a|a.timing.setup_ms+a.timing.primal_ms+a.timing.estimate_ms+a.timing.correction_ms).sum::<f64>()+solution.preconditioner.milliseconds;
    let control_ms:f64=solution.attempts.iter().map(|a|a.timing.estimate_ms+a.timing.correction_ms).sum();
    // At a horizon, populations still relaxing by more than a twentieth of
    // the tolerance keep the mean initial-state dependent.
    let dependent=state.as_ref().is_some_and(|s|s.relaxing>0.05*POPULATION_TOLERANCE);
    let(status,reason)=match(model.drives.is_empty(),dependent){
        (false,true)=>("initial-dependent-time-average","Long-time phase average at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent."),
        (true,true)=>("initial-dependent","Lindblad mean at the observation horizon (horizonUs) of the specified initial density: slow modes still relaxing at the horizon (or conserved quantities) keep it initial-state dependent."),
        (false,false)=>("time-average","Converged long-time phase average of the full driven Lindblad equation (unique limit: only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time); populations are a time average, not a time-independent density."),
        (true,false)if discarded>1e-8=>("population-stationary","Populations converge in the background basis; exported density is the laboratory-frame time average."),
        (true,false)=>("stationary","Unique full-density Lindblad stationary solution (only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time)."),
    };
    let fold=accepted.fold.as_ref().map(|f|f.charges.total());
    Ok(json!({"timesUs":[0.],"states":model.states,"populations":rows,"horizonUs":options.mean_horizon_us,
        "asymptoticStatus":status,"asymptoticReason":reason,"asymptoticPopulations":populations,
        "asymptoticDensityReal":density.iter().map(|z|z.re).collect::<Vec<_>>(),
        "asymptoticDensityImag":density.iter().map(|z|z.im).collect::<Vec<_>>(),
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":axes.source_to_basis,
            "asymptoticMethod":MEAN_METHOD,"meanDensityFrame":MEAN_FRAME,
            "uniquenessScope":"unique limit: every mode but the trace relaxes, however slowly (slowestRelaxationUs)",
            "asymptoticTraceError":trace_error,"asymptoticPositivityError":positivity,"residual":solution.null_residual,
            "asymptoticAccuracy":{"requestedTolerance":POPULATION_TOLERANCE,"estimatedMaximumError":estimated,
                "acceptance":match solution.acceptance{Acceptance::Estimate=>"estimate",Acceptance::Corrected=>"shell-corrected"},
                "truncationEstimate":accepted.truncation,"foldCharge":fold,"safetyFactor":KAPPA,"projectionCharge":projection,
                "scope":"near-shell truncation and the primal's algebraic error (an error equation in the Krylov space) plus the fold remainders charged through the fold response; at a horizon the propagation's estimate instead (beyond the switch time also the part outside the slow modes, the fit's drift and the slow modes' residuals)",
                "metric":"max over states and requested F groups"},
            "zeroModeCount":spectrum.conserved,"conservedQuantities":spectrum.conserved,
            "slowestRatePerUs":solution.slowest_rate(),"slowestRelaxationUs":solution.slowest_rate().map(|r|1./r),
            "slowestRelaxationBasis":{"shiftRatePerUs":spectrum.slowest_rate,"precessingRatePerUs":solution.precessing_rate,
                "rule":"min of the zero shift's slowest rate and half the dressed frames' slowest precessing decay"},
            "latticeOrder":solution.order,"latticeModes":solution.modes.len(),"discardedOpticalCoherence":discarded,
            "horizonMethod":state.as_ref().map(|s|s.details.clone()),"horizonRelaxingBound":state.as_ref().map(|s|s.relaxing),
            "horizonPropagationEstimate":state.as_ref().map(|s|s.error),"horizonProjectionCharge":state.as_ref().map(|_|projection),
            "preparationMs":preparation,"slowModesMs":spectrum.milliseconds,
            "meanTimingMs":{"solve":solve_ms,"errorControl":control_ms,"preconditioner":solution.preconditioner.milliseconds},
            "lattice":lattice_json(prepared,Some(solution))}}))
}

/// A trajectory: the Krylov propagation, or beyond LONG_HORIZON the
/// long-time state (Krylov rows to t_s, the slow expansion after).
enum Run{Short(finite::FiniteSolution),Long(finite::LongTime)}
struct Trajectory{run:Run,rows:Vec<Vec<f64>>,estimated:f64,projection:f64}
impl Trajectory{
    fn short(solution:finite::FiniteSolution)->Self{
        Self{rows:solution.populations.clone(),estimated:solution.estimated_error,projection:solution.projection_charge,run:Run::Short(solution)}
    }
    /// The output grid `times` from the long-time state: the Krylov rows up
    /// to its reach, then the expansion's rows projected onto the simplex
    /// (each with its error and projection charge).
    fn long(request:&Prepared,long:finite::LongTime,times:&[f64])->Self{
        let step=long.times[1];let(mut estimated,mut projection)=(long.krylov.estimated_error,long.krylov.projection_charge);
        let mut rows=Vec::with_capacity(times.len());
        for &t in times{
            let k=(t/step).round()as usize;
            if k<long.times.len()&&(long.times[k]-t).abs()<=1e-9*t.max(1.){rows.push(long.krylov.populations[k].clone());continue;}
            let raw=long.populations(t);let projected=finite::simplex(&raw);let charge=ledger::metric(&raw,&projected,&request.groups);
            projection=projection.max(charge);estimated=estimated.max(long.error(t)+charge);rows.push(projected);
        }
        Self{run:Run::Long(long),rows,estimated,projection}
    }
    fn krylov(&self)->&finite::FiniteSolution{match &self.run{Run::Short(s)=>s,Run::Long(l)=>&l.krylov}}
    fn long_time(&self)->Option<&finite::LongTime>{match &self.run{Run::Long(l)=>Some(l),Run::Short(_)=>None}}
}

/// A request through the lattice core: the mean alone (steadyOnly), or the
/// trajectory with the mean (published as soon as it is ready) merged into
/// the finite result. Up to LONG_HORIZON the trajectory runs on its own
/// thread beside the mean; beyond it one long-time steady solution serves
/// the trajectory and then the mean (its horizon state too).
pub fn solve(model:&Model,options:&Options,publish_mean:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();
    if options.steady_only{return mean_result(model,options,started);}
    if !options.duration_us.is_finite()||options.duration_us<=0.||!(2..=2001).contains(&options.samples){return Err("Invalid finite population grid".into());}
    let request=prepare_request(model,options)?;
    let times:Vec<f64>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    let mut publish=|mean:Result<Value,String>|{if let Ok(value)=&mean{publish_mean(value);}mean};
    // A static model's lattice is one block: its trajectory is propagated
    // directly at any length.
    let(mean,trajectory)=if options.duration_us>finite::LONG_HORIZON&&!model.drives.is_empty(){
        // The long-time expansion, else (slow modes it misses) the Krylov
        // propagation over the whole grid.
        let steady=steady_solution(&request,true);
        let long=match &steady{Ok(s)=>finite::long_time(&request.lattice,&request.density,&request.groups,s,Some(times[1])),Err(e)=>Err(Declined::Failed(e.clone()))};
        let mean=options.include_asymptotic.then(||publish(mean_value(model,options,&request,steady.as_ref(),Some(long.as_ref()),started)));
        let trajectory=match long{Ok(l)=>Ok(Trajectory::long(&request,l,&times)),
            Err(Declined::MissedModes(e))=>finite::propagate(&request.lattice,&request.density,&times,&request.groups,None,&[]).map(Trajectory::short)
                .map_err(|f|format!("{e}; the propagation over the grid: {f}")),
            Err(Declined::Failed(e))=>Err(e)};
        (mean,trajectory)
    }else{
        std::thread::scope(|scope|{
            let trajectory=scope.spawn(||finite::propagate(&request.lattice,&request.density,&times,&request.groups,None,&[]).map(Trajectory::short));
            let mean=options.include_asymptotic.then(||publish(mean_of(model,options,&request,started)));
            (mean,trajectory.join().unwrap_or_else(|panic|std::panic::resume_unwind(panic)))})
    };
    let trajectory=trajectory?;
    let(row_trace,initial_error)=row_errors(&trajectory.rows,&request.initial)?;
    let solution=trajectory.krylov();let t=&solution.timing;let prepared=&request.lattice;
    let checkpoints:Vec<Value>=solution.checkpoints.iter().map(|c|json!({"order":c.order,"columns":c.columns,"dimension":c.dimension,
        "reflected":c.reflected,"gramCondition":c.gram_condition,"growth":c.growth,"basis":c.basis,"shell":c.shell,"fold":c.fold,"projection":c.projection,"total":c.total,"action":c.action,"milliseconds":c.milliseconds})).collect();
    let control_ms=t.basis_ms+t.shell_ms+t.fold_ms;
    let solve_ms=t.preconditioner_ms+t.columns_ms+t.images_ms+t.gram_ms+t.reduced_ms+control_ms;
    let long=trajectory.long_time().map(|l|json!({"switchUs":l.switch,"krylovEstimate":l.krylov.estimated_error,"outsideSlowModes":l.fast,"fitDrift":l.drift,
        "eigenResiduals":l.residuals,"artifactGrowth":l.artifact_growth,"steadyEstimate":l.steady_error,"reachUs":l.times.last(),"rejectedSwitches":l.trail}));
    if trajectory.estimated>POPULATION_TOLERANCE{return Err(format!("Finite-time populations estimated at {:e}; the tolerance was not relaxed",trajectory.estimated));}
    let mut result=json!({"states":model.states,"timesUs":times,"populations":trajectory.rows,"asymptoticStatus":"not-requested",
        "populationAccuracy":{"status":"estimated-converged","requestedTolerance":POPULATION_TOLERANCE,
            "estimatedMaximumError":trajectory.estimated,"projectionCharge":trajectory.projection,
            "includesProjectionCharge":true,"scope":{"states":32,"samples":options.samples,"groups":request.names}},
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":request.axes.source_to_basis,"method":FINITE_METHOD,
            "preparationMs":{"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2],"floquetKernels":prepared.timing[3]},
            "finiteTimingMs":{"solve":solve_ms,"errorControl":control_ms,"eigenbasis":t.eigenbasis_ms,"preconditioner":t.preconditioner_ms,
                "columns":t.columns_ms,"innerApply":t.inner.apply_ms,"innerPrecondition":t.inner.precondition_ms,"innerOrthogonalize":t.inner.orthogonalize_ms,"images":t.images_ms,"gram":t.gram_ms,"reduced":t.reduced_ms,"basis":t.basis_ms,"shell":t.shell_ms,"fold":t.fold_ms},
            "finite":{"order":solution.order,"modes":solution.modes,"columns":solution.columns,"dimension":solution.dimension,
                "translations":solution.translations,"innerSteps":solution.inner_steps,"innerStepsPerColumn":solution.inner_steps as f64/solution.columns.max(1) as f64,
                "applications":solution.applications,"preconditions":solution.preconditions,"reaches":solution.reaches.iter().map(|r|format!("{r:?}")).collect::<Vec<_>>(),
                "eigenbasisCondition":solution.eigenbasis_condition,"ritzCondition":solution.ritz_condition,
                "basisEstimate":solution.basis,"shellEstimate":solution.shell,"foldEstimate":solution.fold,"safetyFactor":KAPPA,
                "longTime":long,"checkpoints":checkpoints}}});
    match mean{
        None=>{}
        Some(Ok(Value::Object(mean)))=>for(key,value)in mean{
            if key=="diagnostics"{if let(Value::Object(target),Value::Object(mean))=(&mut result["diagnostics"],value){for(key,value)in mean{target.entry(key).or_insert(value);}}}
            else if key.starts_with("asymptotic")||key=="horizonUs"{result[key.as_str()]=value;}
        },
        Some(Ok(_))=>return Err("The lattice mean is not an object".into()),
        Some(Err(reason))=>{result["asymptoticStatus"]=json!("not-converged");result["asymptoticReason"]=json!(reason);result["horizonUs"]=json!(options.mean_horizon_us);}
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
        let modes=p.modes(steady::START_ORDER).unwrap();let mut op=LatticeOperator::new(&p,&modes).unwrap();
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
}
