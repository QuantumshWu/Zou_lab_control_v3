//! Production protocol bridge to the single population-core source tree.
//! The TS Model remains the only owner of atomic physics. This adapter only
//! borrows its numerical H/C, exact phase coordinates and state metadata.
//!
//! One prepared core serves both outputs of a request: the long-time mean is
//! computed FIRST (and published as `asymptotic` progress), then the core is
//! handed to the finite-time engine. A steadyOnly request runs only the mean,
//! through the same function, so inline and steadyOnly means are identical.
use crate::{Model,Options,Solver,publish_mean,harmonics};
use serde_json::{json,Value};
use std::collections::BTreeMap;
use std::time::Instant;
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts,GaugePolicy};
use atom_population_core::owned_prepared::{KernelPolicy,coordinates::InitialView,b0_inverse::P0Policy};
use atom_population_core::owned_prepared::runtime_core::RuntimeCore;
use atom_population_core::owned_prepared::engine::{self,FiniteOptions,RequestedGroup};
use atom_population_core::owned_prepared::asymptotic::{self,LabFrame,MeanKind,MeanOptions,MeanSolution};

const MEAN_METHOD:&str="Full32 Cesàro mean of the gauge pencil: defect-corrected null vector with a near-zero pole, halo-condensed Fourier truncation";
const MEAN_FRAME:&str="laboratory time average in the background eigenbasis";

pub fn supports(model:&Model,_options:&Options)->bool {
    model.dimension==32&&model.states.len()==32&&model.frequency_coordinates.is_some()
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

/// Everything a request needs from the Model, prepared once.
struct Request{
    core:RuntimeCore,
    groups:Vec<RequestedGroup>,
    axes:harmonics::PhaseAxes,
    initial_populations:Vec<f64>,
    preparation_ms:f64,
}

fn prepare(solver:&Solver,model:&Model)->Result<Request,String>{
    let started=Instant::now();
    let(ground,groups)=groups(model)?;
    // One phase-axis construction for the mean AND the finite engine: the
    // declared axes, or the exact reduced basis when integer relations exist.
    let axes=harmonics::model_phase_axes(model)?;
    let parts:Vec<_>=axes.parts.iter().map(|&(hi,lo)|FrequencyParts{hi,lo}).collect();
    let phases=PhaseBasis::from_parts(&parts,&axes.drive_indices)?;
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
    let initial_populations=(0..32).map(|i|core.initial().original()[i*32+i].re).collect();
    Ok(Request{core,groups,axes,initial_populations,preparation_ms:started.elapsed().as_secs_f64()*1000.})
}

/// The long-time mean in the steadyOnly result shape. Inline (combined) and
/// steadyOnly requests both call exactly this function.
fn mean_result(solver:&Solver,model:&Model,options:&Options,request:&mut Request,started:Instant)->Result<Value,String>{
    let lab=LabFrame{frame_indices:request.axes.frame_indices.clone(),
        sectors:request.axes.frame_sectors.iter().map(|&s|if s==0{0}else{1}).collect()};
    let mean=asymptotic::solve(&mut request.core,&MeanOptions::new(options.population_tolerance,request.groups.clone(),lab))?;
    let density:Vec<_>=mean.density.iter().map(|z|crate::C{re:z.re,im:z.im}).collect();
    let positivity=crate::positivity(&solver.lap,&density,32)?;
    let trace_error=(mean.populations.iter().sum::<f64>()-1.).abs();
    let(status,reason)=classify(model,&mean);
    let d=&mean.diagnostics;
    let checkpoints:Vec<_>=d.checkpoints.iter().map(|c|json!({"basis":c.columns,"modes":c.modes,"pole":c.pole,
        "basisEstimate":c.basis_estimate,"haloEstimate":c.halo_estimate,"haloModes":c.halo_modes.0,"transportHaloModes":c.halo_modes.1,
        "innerResidual":c.inner_residual,"action":c.action})).collect();
    Ok(json!({"timesUs":[0.],"states":model.states,"populations":[request.initial_populations],
        "asymptoticPopulations":mean.populations,
        "asymptoticDensityReal":mean.density.iter().map(|z|z.re).collect::<Vec<_>>(),
        "asymptoticDensityImag":mean.density.iter().map(|z|z.im).collect::<Vec<_>>(),
        "asymptoticStatus":status,"asymptoticReason":reason,
        "diagnostics":{"engine":"rust-population-core","method":MEAN_METHOD,"steadyMethod":MEAN_METHOD,
            "asymptoticStatus":status,"asymptoticReason":reason,"meanDensityFrame":MEAN_FRAME,
            "densityLimitExists":model.drives.is_empty()&&mean.kind==MeanKind::Unique&&mean.discarded_optical_coherence<=1e-8,
            "initialDependent":!model.drives.is_empty()||mean.kind!=MeanKind::Unique,
            "uniquenessScope":match mean.kind{MeanKind::Unique=>"one-dimensional null space within the resolved rates",
                MeanKind::InitialDependent{..}=>"several conserved quantities: mean of the specified initial density"},
            "traceError":trace_error,"positivityError":positivity,"hermiticityError":0.,
            "residual":d.relative_residual,
            "asymptoticAccuracy":{"requestedTolerance":options.population_tolerance,"estimatedMaximumError":d.estimated_error,
                "basisEstimate":d.basis_estimate,"haloEstimate":d.halo_estimate,"projectionCharge":d.projection_charge,
                "minimumEigenvalueBeforeProjection":d.minimum_eigenvalue_before_projection,
                "traceErrorBeforeProjection":d.trace_error_before_projection,"hermiticityErrorBeforeProjection":d.hermiticity_error,
                "metric":"max over states, requested F groups and the spectral norm of the density error"},
            "zeroModeCount":d.zero_modes,"slowestRatePerUs":d.slowest_rate,"nearZeroPolePerUs":d.pole,
            "meanBasis":d.columns,"meanModes":d.modes,"innerOperatorApplications":d.operator_applications,
            "meanCheckpoints":checkpoints,"discardedOpticalCoherence":mean.discarded_optical_coherence,
            "meanTimingMs":{"operators":d.timing[0],"corrections":d.timing[1],"halo":d.timing[2],"degeneracyProbe":d.timing[3]},
            "uncorrectedPopulations":d.uncorrected_populations,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"constructionMs":request.preparation_ms,
            "meanMs":d.elapsed_ms,"stationaryOnly":options.steady_only,
            "frequencyCoordinateSource":"declared-carrier-coordinates","carrierSourceToBasis":request.axes.source_to_basis}}))
}

/// Status names consumed by the UI. A driven model keeps the long-time
/// phase-average label; a static model distinguishes a stationary density,
/// converging populations with a persistent optical coherence (the lab-frame
/// time average then drops that coherence) and initial-state dependence.
fn classify(model:&Model,mean:&MeanSolution)->(&'static str,&'static str){
    if !model.drives.is_empty(){
        return("initial-dependent-mean",match mean.kind{
            MeanKind::Unique=>"Converged long-time phase average of the full driven Lindblad equation (one-dimensional null space); populations are a time average, not a time-independent density.",
            MeanKind::InitialDependent{..}=>"Converged long-time phase average of the specified initial density; several conserved quantities make it initial-state dependent.",
        });
    }
    match mean.kind{
        MeanKind::Unique if mean.discarded_optical_coherence>1e-8=>("population-stationary","Populations converge in the background basis; exported density is the laboratory-frame time average."),
        MeanKind::Unique=>("stationary","Unique full-density Lindblad stationary solution within the resolved rates."),
        MeanKind::InitialDependent{..}=>("nonunique-initial-dependent","Full Lindblad stationary projection from the specified initial density."),
    }
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
    if !options.steady_only&&(!options.duration_us.is_finite()||options.duration_us<=0.||!(2..=2001).contains(&options.samples)){return Err("Invalid finite population grid".into());}
    let mut request=prepare(solver,model)?;
    if options.steady_only{return mean_result(solver,model,options,&mut request,started);}
    let mean=if options.include_asymptotic{
        let value=mean_result(solver,model,options,&mut request,started);
        if let Ok(value)=&value{publish_mean(value.clone());}
        Some(value)
    }else{None};
    let finite_started=Instant::now();
    let names:Vec<_>=request.groups.iter().map(|g|g.name.clone()).collect();
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
    let solution=engine::solve(request.core,&times,request.groups,finite)?;
    let d=&solution.diagnostics;
    let populations:Vec<_>=solution.populations.chunks_exact(32).map(|p|p.to_vec()).collect();
    let trace=populations.iter().map(|p|(p.iter().sum::<f64>()-1.).abs()).fold(0_f64,f64::max);
    let initial_error=populations[0].iter().zip(&request.initial_populations).map(|(a,b)|(a-b).abs()).fold(0_f64,f64::max);
    if trace>1e-7||initial_error>1e-10{return Err("Returned population probabilities failed fixed trace/initial checks".into());}
    let checkpoints:Vec<_>=d.checkpoints.iter().map(|c|json!({
        "order":c.order,"modes":c.modes,"basis":c.columns,"action":c.action,
        "basisChange":c.basis_change,"domainChange":c.domain_change,"elapsedMs":c.elapsed_ms
    })).collect();
    let mut result=json!({"states":model.states,"timesUs":times,"populations":populations,
        "asymptoticPopulations":null,"asymptoticStatus":"not-requested",
        "populationAccuracy":{"status":"estimated-converged","requestedTolerance":options.population_tolerance,
            "estimatedMaximumError":d.estimated_error,"projectionCharge":d.projection_charge,
            "includesProjectionCharge":true,"scope":{"states":32,"samples":options.samples,"groups":names}},
        "diagnostics":{"engine":"rust-population-core","method":"Full32 density-matrix Lindblad evolution with adaptive population refinement",
            "validationKind":"population","traceError":trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"constructionMs":request.preparation_ms,
            "propagateMs":d.elapsed_ms,"finiteMs":finite_started.elapsed().as_secs_f64()*1000.,"populationTolerance":options.population_tolerance,
            "populationConvergenceEstimate":d.estimated_error,"errorBoundCertified":false,
            "populationProjectionCharge":d.projection_charge,"originalHamiltonianAndCollapseRetained":true,
            "finalDomainOrder":d.final_order,"finalModes":d.final_modes,"finalBasis":d.final_columns,
            "imageColumns":d.image_columns,"innerOperatorApplications":d.operator_applications,
            "peakTrialBytes":d.peak_trial_bytes,"timingMs":{"directions":d.timing[0],"orthogonalize":d.timing[1],"images":d.timing[2],"projections":d.timing[3],"readoutAppend":d.timing[4],"transfer":d.timing[5],"reduced":d.timing[6],"readoutEvaluate":d.timing[7]},"maximumImaginaryPopulation":d.maximum_imaginary_population,
            "historicalAnswerUsed":false,"cached":false,
            "frequencyCoordinateSource":"declared-carrier-coordinates",
            "carrierSourceToBasis":request.axes.source_to_basis,"finiteCarrierSourceToBasis":request.axes.source_to_basis,"populationCheckpoints":checkpoints,
            "asymptoticStatus":"not-requested"}});
    match mean{
        None=>{}
        Some(Ok(mean))=>{
            if let Some(fields)=mean.as_object(){
                for(key,value)in fields{if key.starts_with("asymptotic"){result[key]=value.clone();}}
            }
            let md=&mean["diagnostics"];
            result["diagnostics"]["stationaryDiagnostics"]=md.clone();
            for key in ["asymptoticStatus","asymptoticReason","steadyMethod","residual","asymptoticAccuracy",
                "meanDensityFrame","densityLimitExists","initialDependent","uniquenessScope"]{
                if let Some(value)=md.get(key){result["diagnostics"][key]=value.clone();}
            }
            result["diagnostics"]["asymptoticTraceError"]=md["traceError"].clone();
            result["diagnostics"]["asymptoticPositivityError"]=md["positivityError"].clone();
            result["diagnostics"]["stationarySolveMs"]=md["meanMs"].clone();
        }
        Some(Err(reason))=>{
            result["asymptoticStatus"]=json!("not-converged");result["asymptoticReason"]=json!(reason);
            result["diagnostics"]["asymptoticStatus"]=json!("not-converged");
            result["diagnostics"]["asymptoticReason"]=result["asymptoticReason"].clone();
        }
    }
    Ok(result)
}
