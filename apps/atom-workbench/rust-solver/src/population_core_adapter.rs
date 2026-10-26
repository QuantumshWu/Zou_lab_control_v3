//! Production protocol bridge to the single population-core source tree.
//! The TS Model remains the only owner of atomic physics. This adapter only
//! borrows its numerical H/C, exact phase coordinates and state metadata.
use crate::{Model,Options,Solver,full_model_solve,publish_mean,harmonics};
use serde_json::{json,Value};
use std::collections::BTreeMap;
use std::time::Instant;
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts,GaugePolicy};
use atom_population_core::owned_prepared::{KernelPolicy,coordinates::InitialView,b0_inverse::P0Policy};
use atom_population_core::owned_prepared::runtime_core::RuntimeCore;
use atom_population_core::owned_prepared::engine::{self,FiniteOptions,RequestedGroup};

pub fn supports(model:&Model,options:&Options)->bool {
    !options.steady_only&&model.dimension==32
        &&model.states.len()==32&&model.frequency_coordinates.is_some()
}

fn groups(model:&Model)->Result<(Vec<bool>,Vec<RequestedGroup>),String>{
    let mut grouped=BTreeMap::<String,Vec<usize>>::new();let mut excited=Vec::new();let mut ground=Vec::new();
    for(i,label)in model.states.iter().enumerate(){
        let mut fields=label.splitn(3,'-');let manifold=fields.next().ok_or("Missing atomic manifold label")?;
        let f=fields.next().ok_or("Missing atomic F label")?;
        if !matches!(manifold,"ground"|"D1"|"D2")||fields.next().is_none(){return Err("Population core needs original32-state manifold metadata".into());}
        ground.push(manifold=="ground");if manifold!="ground"{excited.push(i);}
        grouped.entry(format!("F:{manifold}:{f}")).or_default().push(i);
    }
    if ground.iter().filter(|&&x|x).count()!=8||grouped.len()!=8||excited.len()!=24{
        return Err("Rb87 population metadata must contain8 ground and24 excited states in all8 F manifolds".into());
    }
    grouped.insert("F:excited".into(),excited);
    Ok((ground,grouped.into_iter().map(|(name,indices)|RequestedGroup{name,indices}).collect()))
}

pub fn solve(solver:&mut Solver,model:&Model,options:&Options)->Result<Value,String>{
    let result=solve_inner(solver,model,options);
    // Worker requests are serial; the core's single-threaded small BLAS
    // must not change the preferred policy of a later periodic/steady job.
    solver.lap.restore_configured_threads()?;
    result
}

fn solve_inner(solver:&mut Solver,model:&Model,options:&Options)->Result<Value,String>{
    let started=Instant::now();options.validate_population_tolerance()?;
    if !options.duration_us.is_finite()||options.duration_us<=0.||!(2..=2001).contains(&options.samples){return Err("Invalid finite population grid".into());}
    let(ground,requested_groups)=groups(model)?;
    let names:Vec<_>=requested_groups.iter().map(|g|g.name.clone()).collect();
    let declared=model.frequency_coordinates.as_ref().ok_or("Population core requires declared phase coordinates")?;
    let axes=declared.frequencies.len();
    let identity:Vec<_>=(0..axes).map(|i|(0..axes).map(|j|if i==j{1}else{0}).collect::<Vec<i32>>()).collect();
    let parts=harmonics::model_phase_frequency_parts(model,&identity)?;
    let parts:Vec<_>=parts.into_iter().map(|(hi,lo)|FrequencyParts{hi,lo}).collect();
    let phases=PhaseBasis::from_parts(&parts,&declared.drive_indices)?;
    let collapse:Vec<_>=model.collapse.iter().map(|x|MatrixView{real:&x.real,imag:&x.imag}).collect();
    let drives:Vec<_>=model.drives.iter().map(|x|DriveView{omega:x.omega,
        cosine:MatrixView{real:&x.cos.real,imag:&x.cos.imag},sine:MatrixView{real:&x.sin.real,imag:&x.sin.imag}}).collect();
    let view=ModelView{dimension:model.dimension,hamiltonian:MatrixView{real:&model.h_real,imag:&model.h_imag},
        collapse:&collapse,drives:&drives,ground:&ground,initial_index:model.initial_index};
    let core=RuntimeCore::prepare(&view,InitialView{index:model.initial_index,
        real:model.initial_real.as_deref(),imag:model.initial_imag.as_deref()},phases,
        GaugePolicy::UnsafeComponentAligned,KernelPolicy::HybridBlas,P0Policy::EigenPreferred)?;
    // Core initial validation has already checked shape, finite entries,
    // Hermiticity and trace before any Hermitian eigensolver is invoked.
    if model.initial_real.is_some(){
        let initial:Vec<_>=core.initial().original().iter().map(|z|crate::C{re:z.re,im:z.im}).collect();
        if crate::positivity(&solver.lap,&initial,32)?>1e-10{return Err("Initial density is not positive semidefinite".into());}
    }
    let preparation_ms=started.elapsed().as_secs_f64()*1000.;
    let initial_populations:Vec<_>=(0..32).map(|i|core.initial().original()[i*32+i].re).collect();
    let times:Vec<_>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    // TEMPORARY development knobs (removed before delivery).
    let knob=|name:&str|std::env::var(name).ok().and_then(|v|v.parse::<f64>().ok());
    let mut finite=FiniteOptions{population_tolerance:options.population_tolerance,..Default::default()};
    if let Some(c)=knob("ATOM_EXP_SLOW_POLE"){finite.slow_pole_factor=c;}
    if let Some(c)=knob("ATOM_EXP_FAST_POLE"){finite.fast_pole=c;}
    if let Some(c)=knob("ATOM_EXP_MAX_POLES"){finite.maximum_poles=c as usize;}
    if let Ok(v)=std::env::var("ATOM_EXP_POLES"){finite.poles=Some(v.split(',').filter_map(|x|x.parse::<f64>().ok()).collect());}
    if let Some(s)=knob("ATOM_EXP_INNER_STEPS"){finite.inner.maximum_steps=s as usize;}
    if let Some(t)=knob("ATOM_EXP_INNER_TOL"){finite.inner.relative_residual=t;}
    if let Ok(v)=std::env::var("ATOM_EXP_FIXED"){let mut it=v.split(':').map(|x|x.parse::<usize>().unwrap_or(0));finite.fixed=Some((it.next().unwrap_or(1),it.next().unwrap_or(16)));}
    let solution=engine::solve(core,&times,requested_groups,finite)?;
    let d=&solution.diagnostics;
    let populations:Vec<_>=solution.populations.chunks_exact(32).map(|p|p.to_vec()).collect();
    let trace=populations.iter().map(|p|(p.iter().sum::<f64>()-1.).abs()).fold(0_f64,f64::max);
    let initial_error=populations[0].iter().zip(initial_populations).map(|(a,b)|(a-b).abs()).fold(0_f64,f64::max);
    if trace>1e-7||initial_error>1e-10{return Err("Returned population probabilities failed fixed trace/initial checks".into());}
    let checkpoints:Vec<_>=d.checkpoints.iter().map(|c|json!({
        "modes":c.modes,"basis":c.columns,"action":c.action,
        "basisEstimate":c.basis_estimate,"haloEstimate":c.halo_estimate,"elapsedMs":c.elapsed_ms
    })).collect();
    let t=&d.timing;
    let mut result=json!({"states":model.states,"timesUs":times,"populations":populations,
        "asymptoticPopulations":null,"asymptoticStatus":"not-requested",
        "populationAccuracy":{"status":"estimated-converged","requestedTolerance":options.population_tolerance,
            "estimatedMaximumError":d.estimated_error,"projectionCharge":d.projection_charge,
            "includesProjectionCharge":true,"scope":{"states":32,"samples":options.samples,"groups":names}},
        "diagnostics":{"engine":"rust-population-core","method":"Full32 density-matrix Lindblad evolution with adaptive population refinement",
            "validationKind":"population","traceError":trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"constructionMs":preparation_ms,
            "propagateMs":d.elapsed_ms,"populationTolerance":options.population_tolerance,
            "populationConvergenceEstimate":d.estimated_error,"errorBoundCertified":false,
            "populationProjectionCharge":d.projection_charge,"originalHamiltonianAndCollapseRetained":true,
            "errorControl":d.error_control,"rawPopulationEstimate":d.raw_estimate,"finalModes":d.final_modes,"finalBasis":d.final_columns,
            "imageColumns":d.image_columns,"innerOperatorApplications":d.operator_applications,
            "peakTrialBytes":d.peak_trial_bytes,"timingMs":{"directions":t.directions,"orthogonalize":t.orthogonalize,"images":t.images,"projections":t.projections,"readoutAppend":t.readout_append,"transfer":t.transfer,"reduced":t.reduced,"readoutEvaluate":t.readout_evaluate,"dualPrepare":t.dual_prepare,"basisEstimate":t.basis_estimate,"haloEstimate":t.halo_estimate},"maximumImaginaryPopulation":d.maximum_imaginary_population,
            "historicalAnswerUsed":false,"cached":false,
            "frequencyCoordinateSource":"declared-carrier-coordinates",
            "carrierSourceToBasis":identity,"finiteCarrierSourceToBasis":identity,"populationCheckpoints":checkpoints,
            "asymptoticStatus":"not-requested"}});
    // Existing stationary/periodic/quasiperiodic qualification remains owned
    // by the mature steady solver. The normal UI dispatches this in its
    // independent native lane; this path serves direct combined JSON calls.
    if options.include_asymptotic {
        solver.lap.restore_configured_threads()?;
        let mean_options=Options{steady_only:true,..*options};
        match full_model_solve(solver,model,&mean_options){
            Ok(mean)=>{
                publish_mean(mean.clone());
                if let Some(fields)=mean.as_object(){
                    for(key,value)in fields{if key.starts_with("asymptotic"){result[key]=value.clone();}}
                }
                let md=&mean["diagnostics"];
                result["diagnostics"]["stationaryDiagnostics"]=md.clone();
                for key in ["asymptoticStatus","asymptoticReason","steadyMethod","populationResidual","residual",
                    "carrierSourceToBasis","frequencyCoordinateSource","frequencyReconstructionErrors","meanDensityFrame","densityLimitExists"]{
                    if let Some(value)=md.get(key){result["diagnostics"][key]=value.clone();}
                }
                result["diagnostics"]["asymptoticTraceError"]=md["traceError"].clone();
                result["diagnostics"]["asymptoticPositivityError"]=md["positivityError"].clone();
            }
            Err(reason)=>{
                result["asymptoticStatus"]=json!("not-converged");result["asymptoticReason"]=json!(reason);
                result["diagnostics"]["asymptoticStatus"]=json!("not-converged");
                result["diagnostics"]["asymptoticReason"]=result["asymptoticReason"].clone();
            }
        }
        result["diagnostics"]["elapsedMs"]=json!(started.elapsed().as_secs_f64()*1000.);
    }
    Ok(result)
}
