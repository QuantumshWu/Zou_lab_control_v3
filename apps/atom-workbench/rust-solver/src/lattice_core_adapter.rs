//! The folded-lattice population core, selected by ATOM_POPULATION_CORE=lattice
//! or the hidden request option `solverCore`, for steadyOnly requests (the
//! t -> infinity mean). It runs beside the shear core until the cut-over;
//! every other request is served by the shear core unchanged.
//!
//! The result has the shape of the shear core's mean result (one t = 0 row
//! of initial populations, the `asymptotic*` fields) with the lattice
//! profiling under `diagnostics.lattice`. The population tolerance is the
//! fixed `ledger::POPULATION_TOLERANCE`; a request may only restate it.
//!
//! Validation scaffolding: the hidden request object `lattice` (or the
//! environment) sets `orders` (ATOM_LATTICE_ORDERS="2,3": solve exactly these
//! near orders instead of the nested check), `algebraic` / `residualGuard`
//! (the GMRES stop) and `history` (per-step populations).
use crate::{Model,Options,harmonics};
use crate::population_core_adapter::{groups,row_errors};
use serde_json::{json,Value};
use std::time::Instant;
use atom_population_core::Complex64;
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts};
use atom_population_core::lattice::LatticePrepared;
use atom_population_core::lattice::classification::Treatment;
use atom_population_core::lattice::ledger::POPULATION_TOLERANCE;
use atom_population_core::lattice::nested;
use atom_population_core::lattice::steady::SteadyOptions;

const ENGINE:&str="rust-population-core-lattice";
const MEAN_METHOD:&str="Full32 time average of the folded Fourier lattice (near charges explicit, far harmonics folded statically at second order, higher orders charged): dressed-block preconditioned null vector";
const MEAN_FRAME:&str="laboratory time average in the background eigenbasis";

/// Whether the lattice core serves this request.
pub fn selected(options:&Options)->Result<bool,String>{
    let choice=options.solver_core.clone().or_else(||std::env::var("ATOM_POPULATION_CORE").ok());
    match choice.as_deref(){
        None|Some("")|Some("shear")=>Ok(false),
        Some("lattice")=>Ok(options.steady_only),
        Some(other)=>Err(format!("Unknown population core {other:?}; expected shear or lattice")),
    }
}

/// Validation settings (scaffolding).
struct Settings{orders:Option<Vec<usize>>,steady:SteadyOptions}
fn setting(options:&Options,key:&str,variable:&str)->Option<Value>{
    options.lattice.as_ref().and_then(|v|v.get(key).cloned()).or_else(||std::env::var(variable).ok().map(|text|
        serde_json::from_str::<Value>(&text).unwrap_or(Value::String(text))))
}
fn number(value:Option<Value>,label:&str)->Result<Option<f64>,String>{
    match value{None=>Ok(None),Some(v)=>v.as_f64().or_else(||v.as_str().and_then(|s|s.parse().ok())).map(Some).ok_or_else(||format!("Lattice setting {label} must be a number"))}
}
impl Settings{
    fn read(options:&Options)->Result<Self,String>{
        let mut steady=SteadyOptions::default();
        let orders=match setting(options,"orders","ATOM_LATTICE_ORDERS"){None=>None,Some(v)=>Some(match &v{
            Value::Array(a)=>a.iter().map(|x|x.as_u64().map(|k|k as usize).ok_or("Lattice orders must be integers")).collect::<Result<Vec<_>,_>>()?,
            Value::String(s)=>s.split(',').map(|x|x.trim().parse::<usize>().map_err(|_|"Lattice orders must be integers")).collect::<Result<_,_>>()?,
            Value::Number(n)=>vec![n.as_u64().ok_or("Lattice orders must be integers")? as usize],
            _=>return Err("Lattice orders must be a list".into()),
        })};
        if let Some(v)=number(setting(options,"algebraic","ATOM_LATTICE_ALGEBRAIC"),"algebraic")?{steady.algebraic=v;}
        if let Some(v)=number(setting(options,"residualGuard","ATOM_LATTICE_RESIDUAL_GUARD"),"residualGuard")?{steady.residual_guard=v;}
        if let Some(v)=setting(options,"history","ATOM_LATTICE_HISTORY"){steady.history=v.as_bool().or_else(||v.as_str().map(|s|s=="1"||s=="true")).unwrap_or(false);}
        Ok(Self{orders,steady})
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

/// The lattice core's steady (t -> infinity mean) result.
pub fn mean_result(model:&Model,options:&Options,started:Instant)->Result<Value,String>{
    if options.population_tolerance!=POPULATION_TOLERANCE{
        return Err(format!("populationTolerance is fixed at {POPULATION_TOLERANCE} (requested {})",options.population_tolerance));}
    if options.mean_horizon_us.is_some(){return Err("The lattice core serves the t -> infinity mean only (meanHorizonUs must be null)".into());}
    if model.dimension!=32||model.states.len()!=32{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    let settings=Settings::read(options)?;
    let(ground,requested)=groups(model)?;
    let(prepared,axes)=prepare(model,&ground)?;
    let group_indices:Vec<Vec<usize>>=requested.iter().map(|g|g.indices.clone()).collect();
    let solution=nested::solve(&prepared,&group_indices,&settings.steady,settings.orders.as_deref())?;
    let accepted=&solution.solves[solution.accepted];
    let n=32;let d=n*n;
    // Laboratory time average: same-sector pairs from the mode of their frame
    // charge f_i - f_j; cross-sector pairs average out.
    let mode=|q:&Vec<i32>|accepted.modes.iter().position(|m|m==q);
    let mut density=vec![Complex64::default();d];let mut discarded=0_f64;
    for i in 0..n{for j in 0..n{
        let charge:Vec<i32>=axes.frame_indices[i].iter().zip(&axes.frame_indices[j]).map(|(a,b)|a-b).collect();
        let Some(m)=mode(&charge)else{continue};let v=accepted.field[m*d+i*n+j];
        if axes.frame_sectors[i]==axes.frame_sectors[j]{density[i*n+j]=v;}else if charge.iter().all(|&k|k==0){discarded=discarded.max(v.re.hypot(v.im));}
    }}
    for i in 0..n{for j in i..n{let(a,b)=(density[i*n+j],density[j*n+i]);
        let m=Complex64{re:0.5*(a.re+b.re),im:0.5*(a.im-b.im)};density[i*n+j]=m;density[j*n+i]=Complex64{re:m.re,im:-m.im};}}
    let populations:Vec<f64>=(0..n).map(|i|density[i*n+i].re).collect();
    let positivity=(-prepared.lapack.hermitian_eigen(&density,n)?.0[0]).max(0.);
    let trace_error=(populations.iter().sum::<f64>()-1.).abs();
    let initial:Vec<f64>=match &model.initial_real{Some(re)=>(0..n).map(|i|re[i*n+i]).collect(),None=>(0..n).map(|i|if i==model.initial_index{1.}else{0.}).collect()};
    let rows=vec![initial.clone()];let(row_trace,initial_error)=row_errors(&rows,&initial)?;
    let plan=&prepared.plan;
    let solves:Vec<Value>=solution.solves.iter().map(|s|{
        let per_step=s.timing.total_ms/(s.steps.max(1) as f64);let l=&s.ledger;
        let charge=s.charge.as_ref().map(|c|{let p=&c.charges;json!({"source":c.source,"response":c.response,"sourceBound":c.source_bound,
            "sensitivity":p.sensitivity,"steps":c.steps,"milliseconds":c.milliseconds,"total":p.total(),
            "nuFreeze":p.nu_freeze,"thirdOrder":p.third_order,"farShell":p.far_shell,"tail":p.tail,
            "omittedCross":p.omitted_cross,"nonStructural":p.non_structural,"droppedEntries":p.dropped_entries})});
        let mut v=json!({"order":s.order,"modes":s.modes.len(),"nuMax":s.nu_max,"steps":s.steps,"restarts":s.restarts,"restartLength":s.restart,"stop":s.stop,
            "residual":s.residual,"nullResidual":s.null_residual,"trace":s.trace,"stepsToTightResidual":s.steps_to_tight_residual,
            "populations":s.populations,"applications":s.applications,"preconditions":s.preconditions,
            "operatorMacPerApplication":s.operator_work,"preconditionerMacPerApplication":s.precondition_work,
            "regularizedDenominators":s.regularized,"edges":{"explicit":s.edges[0],"crossFold":s.edges[1]},
            "timingMs":{"setup":s.timing.setup_ms,"apply":s.timing.apply_ms,"precondition":s.timing.precondition_ms,
                "orthogonalize":s.timing.orthogonalize_ms,"total":s.timing.total_ms},
            "msPerStep":per_step,"msPerApplication":s.timing.apply_ms/(s.applications.max(1) as f64),
            "msPerPrecondition":s.timing.precondition_ms/(s.preconditions.max(1) as f64),
            "ledger":{"nuFreeze":l.nu_freeze,"thirdOrder":l.third_order,"farShell":l.far_shell,"tail":l.tail,
                "omittedCross":l.omitted_cross,"nonStructural":l.non_structural,"droppedEntries":l.dropped_entries,"total":l.total(),
                "fold":l.fold,"expansionParameter":l.expansion_parameter},
            "foldCharge":charge});
        if settings.steady.history{v["history"]=json!(s.history.iter().map(|h|json!({"residual":h.residual,"change":h.change,"populations":h.populations})).collect::<Vec<_>>());}
        v
    }).collect();
    let f=&prepared.fold.diagnostics;
    let targets:Vec<Value>=prepared.fold.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,
        "restNonzeros":[t.rest[0].nonzeros(),t.rest[1].nonzeros()],"norms":t.norms,"dropped":t.dropped,"largestEntry":t.largest_entry})).collect();
    let dr=&solution.dressed;
    let lattice=json!({
        "harmonicPlan":{"explicit":plan.count(&Treatment::Explicit{structural:true}),"nonStructural":plan.count(&Treatment::Explicit{structural:false}),
            "folded":plan.count(&Treatment::Fold),"gap":plan.gap,"cut":plan.cut,"l0Bound":plan.l0_bound,"structuralFloor":plan.structural_floor,
            "steps":plan.steps.iter().map(|s|json!({"charge":s.charge,"nu":s.nu,"size":s.size,"treatment":format!("{:?}",s.treatment)})).collect::<Vec<_>>(),
            "foldTargets":plan.targets.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>(),
            "omittedTargets":plan.omitted.iter().map(|t|json!({"charge":t.charge,"nu":t.nu,"pairs":t.pairs.len(),"size":t.size})).collect::<Vec<_>>()},
        "fold":{"backend":f.backend,"frameCondition":f.frame_condition,"woodburyRank":f.woodbury_rank,"farSources":f.far_sources,"columns":f.columns,
            "solveResidual":f.solve_residual,"refinedSolves":f.refined,"refinedResidual":f.refined_residual,"milliseconds":f.milliseconds,"targets":targets},
        "preconditioner":{"kind":"dense-exact (validation)","milliseconds":solution.dressed_ms,"conditionBound":dr.condition_bound,"reconstruction":dr.reconstruction,
            "tiny":dr.tiny,"sigma":dr.sigma,"slowestRate":dr.slowest_rate,"central":dr.central,"centralRank":dr.central_rank},
        "solves":solves,"accepted":solution.accepted,"nestedDifference":solution.difference,
        "algebraicTolerance":settings.steady.algebraic,"residualGuard":settings.steady.residual_guard});
    let solve_ms:f64=solution.solves.iter().map(|s|s.timing.total_ms+s.timing.setup_ms).sum();
    let status=if model.drives.is_empty(){"stationary"}else{"time-average"};
    let reason=if model.drives.is_empty(){"Full Lindblad stationary solution of the folded lattice (unique: the trace-row system is nonsingular)."}
        else{"Converged long-time phase average of the driven Lindblad equation on the folded Fourier lattice; populations are a time average, not a time-independent density."};
    Ok(json!({"timesUs":[0.],"states":model.states,"populations":rows,"horizonUs":options.mean_horizon_us,
        "asymptoticStatus":status,"asymptoticReason":reason,"asymptoticPopulations":populations,
        "asymptoticDensityReal":density.iter().map(|z|z.re).collect::<Vec<_>>(),
        "asymptoticDensityImag":density.iter().map(|z|z.im).collect::<Vec<_>>(),
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"carrierSourceToBasis":axes.source_to_basis,
            "asymptoticMethod":MEAN_METHOD,"meanDensityFrame":MEAN_FRAME,
            "asymptoticTraceError":trace_error,"asymptoticPositivityError":positivity,"residual":accepted.null_residual,
            "asymptoticAccuracy":{"requestedTolerance":POPULATION_TOLERANCE,"estimatedMaximumError":solution.estimate,
                "truncationEstimate":solution.difference,"foldCharge":accepted.charge.as_ref().map(|c|c.charges.total()),
                "scope":"near-order check k vs k+1 plus the fold remainders charged through the steady sensitivity",
                "metric":"max over states and requested F groups"},
            "latticeModes":accepted.modes.len(),"discardedOpticalCoherence":discarded,
            "preparationMs":{"source":prepared.timing[0],"plan":prepared.timing[1],"fold":prepared.timing[2],"dressed":solution.dressed_ms},
            "meanTimingMs":{"solve":solve_ms},"lattice":lattice}}))
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
}
