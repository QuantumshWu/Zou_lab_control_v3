//! The folded-lattice population core, selected by ATOM_POPULATION_CORE=lattice
//! or the hidden request option `solverCore`: the t -> infinity mean and the
//! finite trajectory (with its mean alongside). It runs beside the shear core
//! until the cut-over; without the selection every request is served by the
//! shear core unchanged.
//!
//! The results have the shapes of the shear core's: a mean result (one t = 0
//! row of initial populations, the `asymptotic*` fields, the
//! conserved-quantity count and the slowest relaxation) and a finite result
//! (projected rows, `populationAccuracy`, the mean merged in), with the
//! lattice profiling under `diagnostics.lattice` and `diagnostics.finite`. The population tolerance is the fixed
//! `ledger::POPULATION_TOLERANCE`; a request may only restate it.
use crate::{Model,Options,harmonics};
use crate::population_core_adapter::{groups,row_errors};
use serde_json::{json,Value};
use std::time::Instant;
use atom_population_core::Complex64;
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts};
use atom_population_core::lattice::LatticePrepared;
use atom_population_core::lattice::classification::Treatment;
use atom_population_core::lattice::ledger::{KAPPA,POPULATION_TOLERANCE};
use atom_population_core::lattice::spectrum::{self,SlowSpectrum};
use atom_population_core::lattice::steady::{self,Acceptance,Attempt,SteadySolution};
use atom_population_core::lattice::finite;

const ENGINE:&str="rust-population-core-lattice";
const MEAN_METHOD:&str="Full32 time average of the folded Fourier lattice (near charges explicit, far harmonics folded statically at second order, higher orders charged): dressed-block preconditioned null vector, its truncation and fold errors estimated in the solver's Krylov space";
const MEAN_FRAME:&str="laboratory time average in the background eigenbasis";
const FINITE_METHOD:&str="Full32 density-matrix Lindblad evolution on the folded Fourier lattice: rational Krylov with lattice-translation augmentation, its basis, near-shell and fold errors estimated by propagating their sources through the dressed blocks";

/// Whether the lattice core serves this request.
pub fn selected(options:&Options)->Result<bool,String>{
    let choice=options.solver_core.clone().or_else(||std::env::var("ATOM_POPULATION_CORE").ok());
    match choice.as_deref(){
        None|Some("")|Some("shear")=>Ok(false),
        Some("lattice")=>Ok(true),
        Some(other)=>Err(format!("Unknown population core {other:?}; expected shear or lattice")),
    }
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
        collapse:&collapse,drives:&drives,ground,initial_index:model.initial_index};
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

/// The lattice profiling: plan, fold, preconditioner, spectrum and attempts.
fn lattice_json(prepared:&LatticePrepared,spectrum:Option<&SlowSpectrum>,solution:Option<&SteadySolution>)->Value{
    let plan=&prepared.plan;let f=&prepared.fold.diagnostics;
    let targets:Vec<Value>=prepared.fold.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,
        "restStored":[t.rest[0].stored(),t.rest[1].stored()],"denseSectors":[t.rest[0].blocks.len(),t.rest[1].blocks.len()],
        "norms":t.norms,"dropped":t.dropped,"largestEntry":t.largest_entry})).collect();
    json!({
        "harmonicPlan":{"explicit":plan.count(&Treatment::Explicit{structural:true}),"nonStructural":plan.count(&Treatment::Explicit{structural:false}),
            "folded":plan.count(&Treatment::Fold),"gap":plan.gap,"cut":plan.cut,"l0Bound":plan.l0_bound,"structuralFloor":plan.structural_floor,
            "steps":plan.steps.iter().map(|s|json!({"charge":s.charge,"nu":s.nu,"size":s.size,"treatment":format!("{:?}",s.treatment)})).collect::<Vec<_>>(),
            "foldTargets":plan.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>(),
            "omittedTargets":plan.omitted.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>()},
        "fold":{"backend":f.backend,"frameCondition":f.frame_condition,"woodburyRank":f.woodbury_rank,"farSources":f.far_sources,"columns":f.columns,
            "solveResidual":f.solve_residual,"refinedSolves":f.refined,"refinedResidual":f.refined_residual,"milliseconds":f.milliseconds,
            "phaseMs":{"solves":f.phase_ms[0],"storage":f.phase_ms[1]},"targets":targets},
        "preconditioner":solution.map(|s|json!({"kind":"structured","milliseconds":s.dressed_ms,"frameCondition":s.dressed.condition,
            "frameReconstruction":s.dressed.reconstruction,"woodburyRank":s.dressed.woodbury_rank})),
        "spectrum":spectrum.map(|s|json!({"conserved":s.conserved,"slowestRate":s.slowest_rate,"floor":s.floor,"milliseconds":s.milliseconds,
            "slowest":s.slowest.iter().map(|z|[z.re,z.im]).collect::<Vec<_>>()})),
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
    if options.mean_horizon_us.is_some(){return Err("The lattice core serves the t -> infinity mean only (meanHorizonUs must be null)".into());}
    if model.dimension!=32||model.states.len()!=32{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    let(ground,requested)=groups(model)?;
    let(lattice,axes)=prepare(model,&ground)?;
    let n=32;
    let density:Vec<Complex64>=match &model.initial_real{
        Some(re)=>{let im=model.initial_imag.as_deref();(0..n*n).map(|k|Complex64{re:re[k],im:im.map_or(0.,|v|v[k])}).collect()}
        None=>(0..n*n).map(|k|Complex64{re:if k==model.initial_index*(n+1){1.}else{0.},im:0.}).collect(),
    };
    let initial=(0..n).map(|i|density[i*n+i].re).collect();
    Ok(Prepared{lattice,axes,groups:requested.iter().map(|g|g.indices.clone()).collect(),names:requested.iter().map(|g|g.name.clone()).collect(),density,initial})
}

/// The lattice core's steady (t -> infinity mean) result.
pub fn mean_result(model:&Model,options:&Options,started:Instant)->Result<Value,String>{
    let request=prepare_request(model,options)?;mean_value(model,options,&request,started)
}

/// The mean: one t = 0 row of initial populations and the `asymptotic*`
/// fields (`not-converged` with its reason when no mean is accepted).
fn mean_value(model:&Model,options:&Options,request:&Prepared,started:Instant)->Result<Value,String>{
    let(prepared,axes,group_indices)=(&request.lattice,&request.axes,&request.groups);
    let n=32;let d=n*n;
    let rows=vec![request.initial.clone()];let(row_trace,initial_error)=row_errors(&rows,&request.initial)?;
    let preparation=json!({"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2]});
    let not_converged=|reason:String,spectrum:Option<&SlowSpectrum>|json!({"timesUs":[0.],"states":model.states,"populations":rows,
        "horizonUs":options.mean_horizon_us,"asymptoticStatus":"not-converged","asymptoticReason":reason,
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,"elapsedMs":started.elapsed().as_secs_f64()*1000.,
            "carrierSourceToBasis":axes.source_to_basis,"asymptoticMethod":MEAN_METHOD,"preparationMs":preparation,
            "lattice":lattice_json(prepared,spectrum,None)}});
    // The slow spectrum (one dense LU) runs beside the steady solve.
    let(spectrum,solution)=std::thread::scope(|scope|{
        let slow=scope.spawn(||spectrum::slow_spectrum(prepared));
        let solution=steady::solve(prepared,group_indices);
        (slow.join().unwrap_or_else(|_|Err("Slow spectrum thread panicked".into())),solution)});
    let spectrum=match spectrum{Ok(s)=>s,
        Err(e)=>return Ok(not_converged(format!("Slow spectrum of the folded centre generator unavailable: {e}"),None))};
    if spectrum.conserved>1{
        return Ok(not_converged(format!("{} conserved quantities: the mean depends on the initial state, which the lattice core does not serve",spectrum.conserved),Some(&spectrum)));}
    let solution=match solution{Ok(s)=>s,Err(e)=>return Ok(not_converged(e,Some(&spectrum)))};
    // Laboratory time average: same-sector pairs from the mode of their frame
    // charge f_i - f_j; cross-sector pairs average out.
    let mode=|q:&Vec<i32>|solution.modes.iter().position(|m|m==q);
    let mut density=vec![Complex64::default();d];let mut discarded=0_f64;
    for i in 0..n{for j in 0..n{
        let charge:Vec<i32>=axes.frame_indices[i].iter().zip(&axes.frame_indices[j]).map(|(a,b)|a-b).collect();
        let Some(m)=mode(&charge)else{continue};let v=solution.field[m*d+i*n+j];
        if axes.frame_sectors[i]==axes.frame_sectors[j]{density[i*n+j]=v;}else if charge.iter().all(|&k|k==0){discarded=discarded.max(v.re.hypot(v.im));}
    }}
    for i in 0..n{for j in i..n{let(a,b)=(density[i*n+j],density[j*n+i]);
        let m=Complex64{re:0.5*(a.re+b.re),im:0.5*(a.im-b.im)};density[i*n+j]=m;density[j*n+i]=Complex64{re:m.re,im:-m.im};}}
    let populations:Vec<f64>=(0..n).map(|i|density[i*n+i].re).collect();
    let positivity=(-prepared.lapack.hermitian_eigen(&density,n)?.0[0]).max(0.);
    let trace_error=(populations.iter().sum::<f64>()-1.).abs();
    let accepted=solution.attempts.last().ok_or("An accepted lattice solution has its attempt")?;
    let solve_ms:f64=solution.attempts.iter().map(|a|a.timing.setup_ms+a.timing.primal_ms+a.timing.estimate_ms+a.timing.correction_ms).sum::<f64>()+solution.dressed_ms;
    let control_ms:f64=solution.attempts.iter().map(|a|a.timing.estimate_ms+a.timing.correction_ms).sum();
    let(status,reason)=match(model.drives.is_empty(),discarded>1e-8){
        (false,_)=>("time-average","Converged long-time phase average of the full driven Lindblad equation (unique limit: only the trace is conserved; diagnostics.slowestRelaxationUs is the slowest relaxation time); populations are a time average, not a time-independent density."),
        (true,true)=>("population-stationary","Populations converge in the background basis; exported density is the laboratory-frame time average."),
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
            "asymptoticAccuracy":{"requestedTolerance":POPULATION_TOLERANCE,"estimatedMaximumError":solution.estimated_error,
                "acceptance":match solution.acceptance{Acceptance::Estimate=>"estimate",Acceptance::Corrected=>"shell-corrected"},
                "truncationEstimate":accepted.truncation,"foldCharge":fold,"safetyFactor":KAPPA,
                "scope":"near-shell truncation and the primal's algebraic error (an error equation in the Krylov space) plus the fold remainders charged through the fold response",
                "metric":"max over states and requested F groups"},
            "zeroModeCount":spectrum.conserved,"conservedQuantities":spectrum.conserved,
            "slowestRatePerUs":spectrum.slowest_rate,"slowestRelaxationUs":spectrum.slowest_rate.map(|r|1./r),
            "latticeOrder":solution.order,"latticeModes":solution.modes.len(),"discardedOpticalCoherence":discarded,
            "preparationMs":preparation,"spectrumMs":spectrum.milliseconds,
            "meanTimingMs":{"solve":solve_ms,"errorControl":control_ms,"dressed":solution.dressed_ms},
            "lattice":lattice_json(prepared,Some(&spectrum),Some(&solution))}}))
}


/// A request through the lattice core: the mean alone (steadyOnly), or the
/// trajectory on its own thread beside the mean, which is published as soon
/// as it is ready and merged into the finite result.
pub fn solve(model:&Model,options:&Options,publish_mean:&mut dyn FnMut(&Value),started:Instant)->Result<Value,String>{
    if options.steady_only{return mean_result(model,options,started);}
    let request=prepare_request(model,options)?;
    let times:Vec<f64>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    let(mean,trajectory)=std::thread::scope(|scope|{
        let trajectory=scope.spawn(||finite::propagate(&request.lattice,&request.density,&times,&request.groups));
        let mean=options.include_asymptotic.then(||{let mean=mean_value(model,options,&request,started);if let Ok(value)=&mean{publish_mean(value);}mean});
        (mean,trajectory.join().unwrap_or_else(|panic|std::panic::resume_unwind(panic)))});
    let solution=trajectory?;
    let rows=solution.populations.clone();
    let(row_trace,initial_error)=row_errors(&rows,&request.initial)?;
    let t=&solution.timing;let prepared=&request.lattice;
    let checkpoints:Vec<Value>=solution.checkpoints.iter().map(|c|json!({"order":c.order,"columns":c.columns,"dimension":c.dimension,
        "reflected":c.reflected,"gramCondition":c.gram_condition,"growth":c.growth,"basis":c.basis,"shell":c.shell,"fold":c.fold,"projection":c.projection,"total":c.total,"action":c.action,"milliseconds":c.milliseconds})).collect();
    let control_ms=t.basis_ms+t.shell_ms+t.fold_ms;
    let solve_ms=t.dressed_ms+t.columns_ms+t.images_ms+t.gram_ms+t.reduced_ms+control_ms;
    let mut result=json!({"states":model.states,"timesUs":times,"populations":rows,"asymptoticStatus":"not-requested",
        "populationAccuracy":{"status":"estimated-converged","requestedTolerance":POPULATION_TOLERANCE,
            "estimatedMaximumError":solution.estimated_error,"projectionCharge":solution.projection_charge,
            "includesProjectionCharge":true,"scope":{"states":32,"samples":options.samples,"groups":request.names}},
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":request.axes.source_to_basis,"method":FINITE_METHOD,
            "preparationMs":{"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2]},
            "finiteTimingMs":{"solve":solve_ms,"errorControl":control_ms,"eigenbasis":t.eigenbasis_ms,"dressed":t.dressed_ms,
                "columns":t.columns_ms,"innerApply":t.inner.apply_ms,"innerPrecondition":t.inner.precondition_ms,"innerOrthogonalize":t.inner.orthogonalize_ms,"images":t.images_ms,"gram":t.gram_ms,"reduced":t.reduced_ms,"basis":t.basis_ms,"shell":t.shell_ms,"fold":t.fold_ms},
            "finite":{"order":solution.order,"modes":solution.modes,"columns":solution.columns,"dimension":solution.dimension,
                "translations":solution.translations,"innerSteps":solution.inner_steps,"innerStepsPerColumn":solution.inner_steps as f64/solution.columns.max(1) as f64,
                "applications":solution.applications,"preconditions":solution.preconditions,
                "eigenbasisCondition":solution.eigenbasis_condition,"ritzCondition":solution.ritz_condition,
                "basisEstimate":solution.basis,"shellEstimate":solution.shell,"foldEstimate":solution.fold,"safetyFactor":KAPPA,
                "checkpoints":checkpoints}}});
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
            "foldMs":p.timing[2],"prepareMs":ms,"diagnostics":{"backend":p.fold.diagnostics.backend,"condition":p.fold.diagnostics.frame_condition,
            "residual":p.fold.diagnostics.solve_residual,"refined":p.fold.diagnostics.refined,"refinedResidual":p.fold.diagnostics.refined_residual}}});
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
        let structured=atom_population_core::lattice::preconditioner::StructuredDressed::new(&p).unwrap();
        let mut batch=structured.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true).unwrap();
        use atom_population_core::lattice::preconditioner::Preconditioner;
        let preconditioner_linearity=linear(&mut |u,v|batch.apply(u,v).unwrap());
        println!("{}",json!({"operatorLinearity":operator_linearity,"preconditionerLinearity":preconditioner_linearity}));
        let size=reference.iter().map(|v|v.re.hypot(v.im)).fold(0.,f64::max);
        let mut worst=(0.,0,0);
        for(k,(a,b))in y.iter().zip(&reference).enumerate(){let e=(a.re-b.re).hypot(a.im-b.im);if e>worst.0{worst=(e,k/d,k%d);}}
        println!("{}",json!({"request":path,"modes":modes.len(),"largest":size,"worstError":worst.0,"worstMode":worst.1,"worstEntry":worst.2,
            "canonical":op.canonical,"charges":modes}));
    }
    /// Step counts of the primal at order 2 with the structured and the exact
    /// dense dressed preconditioner (ignored, `validation` feature; needs
    /// ATOM_LATTICE_REQUEST=<request.json>; prints one JSON line).
    #[cfg(feature="validation")]
    #[test]
    #[ignore]
    fn lattice_preconditioner_steps_against_the_dense_dressed_block(){
        use atom_population_core::lattice::krylov::{self,KrylovOptions,System};
        use atom_population_core::lattice::operator::LatticeOperator;
        use atom_population_core::lattice::preconditioner::StructuredDressed;
        use atom_population_core::lattice::dense_dressed::DenseDressed;
        let path=std::env::var("ATOM_LATTICE_REQUEST").expect("ATOM_LATTICE_REQUEST");
        let request:Value=serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let model:Model=serde_json::from_value(request["model"].clone()).unwrap();
        let(ground,requested)=groups(&model).unwrap();let(p,_)=prepare(&model,&ground).unwrap();
        let group_indices:Vec<Vec<usize>>=requested.iter().map(|g|g.indices.clone()).collect();
        let modes=p.modes(steady::START_ORDER).unwrap();let mut options=KrylovOptions::default();
        if let Ok(v)=std::env::var("ATOM_LATTICE_RESTART"){options.restart=v.parse().unwrap();}
        if let Ok(v)=std::env::var("ATOM_LATTICE_GUARD"){options.residual_guard=v.parse().unwrap();}
        let run=|pre:&mut dyn atom_population_core::lattice::preconditioner::Preconditioner|{
            let op=LatticeOperator::new(&p,&modes).unwrap();let mut system=System::new(op,pre).unwrap();let b=system.source();
            let clock=Instant::now();let k=krylov::gmres(&mut system,&b,None,&group_indices,&options).unwrap();
            let t:f64=system.watch.iter().map(|&i|k.x[i].re).sum();
            (k.steps,clock.elapsed().as_secs_f64()*1e3,system.watch.iter().map(|&i|k.x[i].re/t).collect::<Vec<f64>>(),k.stop,k.residual)};
        let clock=Instant::now();let structured=StructuredDressed::new(&p).unwrap();let structured_ms=clock.elapsed().as_secs_f64()*1e3;
        let op=LatticeOperator::new(&p,&modes).unwrap();
        let mut batch=structured.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true).unwrap();let a=run(&mut batch);
        let clock=Instant::now();let dense=DenseDressed::new(&p).unwrap();let dense_ms=clock.elapsed().as_secs_f64()*1e3;
        let mut batch=dense.batch(&op.nu,op.centre,Some((op.canonical.clone(),op.mirror.clone()))).unwrap();let b=run(&mut batch);
        let difference=a.2.iter().zip(&b.2).map(|(x,y)|(x-y).abs()).fold(0.,f64::max);
        println!("{}",json!({"request":path,"modes":modes.len(),"structured":{"steps":a.0,"primalMs":a.1,"buildMs":structured_ms,"stop":a.3,"residual":a.4,"populations":a.2},
            "dense":{"steps":b.0,"primalMs":b.1,"buildMs":dense_ms,"stop":b.3,"residual":b.4,"populations":b.2},"populationDifference":difference}));
    }
}
