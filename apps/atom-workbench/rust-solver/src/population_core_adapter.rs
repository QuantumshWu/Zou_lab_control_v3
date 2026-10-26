//! Production protocol bridge to the single population-core source tree.
//! The TS Model remains the only owner of atomic physics. This adapter only
//! borrows its numerical H/C, exact phase coordinates and state metadata.
//!
//! One prepared core serves both outputs of a request: the long-time mean is
//! computed FIRST (and published as `asymptotic` progress), then the core is
//! handed to the finite-time engine. A steadyOnly request runs only the mean,
//! through the same function, so inline and steadyOnly means are identical.
//!
//! # Result contract
//! The one result shape, read by `normalizeDynamicsResult` in
//! src/solver-client.ts; main.rs adds `protocol` and `modelSchema`.
//! - `states` (32 ids), `timesUs`, `populations` (one row per time). A mean
//!   result has the single t = 0 row, the initial populations; a finite result
//!   has >= 2 rows and `populationAccuracy`, their estimated error.
//! - `asymptoticStatus`. A converged mean (`stationary`, `population-stationary`
//!   and `initial-dependent` for static models; `time-average` and
//!   `initial-dependent-time-average` for driven ones) comes with
//!   `asymptoticReason`, `asymptoticPopulations` and `asymptoticDensityReal/Imag`;
//!   otherwise the status is `not-converged` (with its `asymptoticReason`) or
//!   `not-requested`.
//! - `diagnostics`. The reader gates the rows' `traceError` and
//!   `initialStateError`, `elapsedMs`, a finite result's `method`, and a mean's
//!   `asymptoticMethod`, `asymptoticTraceError`, `asymptoticPositivityError` and
//!   `residual` (relative `|F x|/|E x|` of the mean's null vector); every
//!   other key is profiling. A finite result carries the mean's diagnostics
//!   under the same names; keys both report (engine, row gates, timings,
//!   operator counts) keep the finite value.
use crate::{Model,Options,harmonics};
use serde_json::{json,Value};
use std::collections::BTreeMap;
use std::time::Instant;
use atom_population_core::preparation::{ModelView,MatrixView,DriveView,PhaseBasis,FrequencyParts,GaugePolicy};
use atom_population_core::Complex64;
use atom_population_core::owned_prepared::{self,KernelPolicy,coordinates::InitialView,b0_inverse::P0Policy};
use atom_population_core::owned_prepared::runtime_core::RuntimeCore;
use atom_population_core::owned_prepared::engine::{self,FiniteOptions,RequestedGroup};
use atom_population_core::owned_prepared::asymptotic::{self,LabFrame,MeanKind,MeanOptions,MeanSolution};

const ENGINE:&str="rust-population-core";
const FINITE_METHOD:&str="Full32 density-matrix Lindblad evolution with adaptive population refinement";
const MEAN_METHOD:&str="Full32 Cesàro mean of the gauge pencil: defect-corrected null vector with a near-zero pole, halo-condensed Fourier truncation";
const MEAN_FRAME:&str="laboratory time average in the background eigenbasis";

/// The worker start-up check: the LAPACK runtime every solve uses.
pub fn warmup()->Result<(),String>{owned_prepared::self_check()}

/// Most negative eigenvalue (as a nonnegative violation) of a row-major
/// Hermitian 32x32 density. Read column-major it is the complex conjugate,
/// which has the same spectrum.
fn positivity(core:&RuntimeCore,density:&[Complex64])->Result<f64,String>{
    Ok((-core.lapack().hermitian_eigen(density,32)?.0[0]).max(0.))
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

fn prepare(model:&Model)->Result<Request,String>{
    let started=Instant::now();
    if model.dimension!=32||model.states.len()!=32{return Err("rb87-full/2 requires the complete 32-state Rb87 model".into());}
    if model.frequency_coordinates.is_none(){return Err("rb87-full/2 requires authoritative frequencyCoordinates metadata".into());}
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
    if model.initial_real.is_some()&&positivity(&core,core.initial().original())?>1e-10{
        return Err("Initial density is not positive semidefinite".into());
    }
    let initial_populations=(0..32).map(|i|core.initial().original()[i*32+i].re).collect();
    Ok(Request{core,groups,axes,initial_populations,preparation_ms:started.elapsed().as_secs_f64()*1000.})
}

/// The row gates every result reports: the largest |row sum - 1| and the
/// first row's largest deviation from the prepared initial populations.
fn row_errors(rows:&[Vec<f64>],initial:&[f64])->Result<(f64,f64),String>{
    let trace=rows.iter().map(|p|(p.iter().sum::<f64>()-1.).abs()).fold(0_f64,f64::max);
    let initial_error=rows[0].iter().zip(initial).map(|(a,b)|(a-b).abs()).fold(0_f64,f64::max);
    if trace>1e-7||initial_error>1e-10{return Err("Returned population probabilities failed fixed trace/initial checks".into());}
    Ok((trace,initial_error))
}

/// The long-time mean result. Inline (combined) and steadyOnly requests both
/// call exactly this function.
fn mean_result(model:&Model,options:&Options,request:&mut Request,started:Instant)->Result<Value,String>{
    let lab=LabFrame{frame_indices:request.axes.frame_indices.clone(),
        sectors:request.axes.frame_sectors.iter().map(|&s|if s==0{0}else{1}).collect()};
    let mean=asymptotic::solve(&mut request.core,&MeanOptions::new(options.population_tolerance,request.groups.clone(),lab))?;
    let positivity=positivity(&request.core,&mean.density)?;
    let trace_error=(mean.populations.iter().sum::<f64>()-1.).abs();
    let(status,reason)=classify(model,&mean);
    let rows=vec![request.initial_populations.clone()];
    let(row_trace,initial_error)=row_errors(&rows,&request.initial_populations)?;
    let d=&mean.diagnostics;
    let checkpoints:Vec<_>=d.checkpoints.iter().map(|c|json!({"basis":c.columns,"modes":c.modes,"pole":c.pole,
        "basisEstimate":c.basis_estimate,"haloEstimate":c.halo_estimate,"haloModes":c.halo_modes.0,"transportHaloModes":c.halo_modes.1,
        "innerResidual":c.inner_residual,"action":c.action})).collect();
    Ok(json!({"timesUs":[0.],"states":model.states,"populations":rows,
        "asymptoticStatus":status,"asymptoticReason":reason,"asymptoticPopulations":mean.populations,
        "asymptoticDensityReal":mean.density.iter().map(|z|z.re).collect::<Vec<_>>(),
        "asymptoticDensityImag":mean.density.iter().map(|z|z.im).collect::<Vec<_>>(),
        "diagnostics":{"engine":ENGINE,"traceError":row_trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"constructionMs":request.preparation_ms,
            "carrierSourceToBasis":request.axes.source_to_basis,"innerOperatorApplications":d.operator_applications,
            "asymptoticMethod":MEAN_METHOD,"meanDensityFrame":MEAN_FRAME,
            "uniquenessScope":match mean.kind{MeanKind::Unique=>"one-dimensional null space within the resolved rates",
                MeanKind::InitialDependent{..}=>"several conserved quantities: mean of the specified initial density"},
            "asymptoticTraceError":trace_error,"asymptoticPositivityError":positivity,
            "residual":d.relative_residual,
            "asymptoticAccuracy":{"requestedTolerance":options.population_tolerance,"estimatedMaximumError":d.estimated_error,
                "basisEstimate":d.basis_estimate,"haloEstimate":d.halo_estimate,"projectionCharge":d.projection_charge,
                "minimumEigenvalueBeforeProjection":d.minimum_eigenvalue_before_projection,
                "traceErrorBeforeProjection":d.trace_error_before_projection,"hermiticityErrorBeforeProjection":d.hermiticity_error,
                "metric":"max over states, requested F groups and the spectral norm of the density error"},
            "zeroModeCount":d.zero_modes,"slowestRatePerUs":d.slowest_rate,"nearZeroPolePerUs":d.pole,
            "meanBasis":d.columns,"meanModes":d.modes,"meanCheckpoints":checkpoints,
            "discardedOpticalCoherence":mean.discarded_optical_coherence,
            "meanTimingMs":{"operators":d.timing[0],"corrections":d.timing[1],"halo":d.timing[2],"degeneracyProbe":d.timing[3]},
            "meanMs":d.elapsed_ms,"uncorrectedPopulations":d.uncorrected_populations}}))
}

/// A finite result carries its mean: the `asymptotic*` fields, and every mean
/// diagnostic the finite result does not report itself.
fn merge_mean(result:&mut Value,mean:Value){
    let Value::Object(mean)=mean else{unreachable!("mean_result builds an object")};
    for(key,value)in mean{
        if key=="diagnostics"{
            if let(Value::Object(finite),Value::Object(mean))=(&mut result["diagnostics"],value){
                for(key,value)in mean{finite.entry(key).or_insert(value);}
            }
        }else if key.starts_with("asymptotic"){result[key.as_str()]=value;}
    }
}

/// Status names consumed by the UI. A driven model keeps the long-time
/// phase-average label; a static model distinguishes a stationary density,
/// converging populations with a persistent optical coherence (the lab-frame
/// time average then drops that coherence) and initial-state dependence.
fn classify(model:&Model,mean:&MeanSolution)->(&'static str,&'static str){
    match(model.drives.is_empty(),&mean.kind){
        (false,MeanKind::Unique)=>("time-average","Converged long-time phase average of the full driven Lindblad equation (one-dimensional null space); populations are a time average, not a time-independent density."),
        (false,MeanKind::InitialDependent{..})=>("initial-dependent-time-average","Converged long-time phase average of the specified initial density; several conserved quantities make it initial-state dependent."),
        (true,MeanKind::Unique) if mean.discarded_optical_coherence>1e-8=>("population-stationary","Populations converge in the background basis; exported density is the laboratory-frame time average."),
        (true,MeanKind::Unique)=>("stationary","Unique full-density Lindblad stationary solution within the resolved rates."),
        (true,MeanKind::InitialDependent{..})=>("initial-dependent","Full Lindblad stationary projection from the specified initial density."),
    }
}

/// One rb87-full/2 request. With `includeAsymptotic` the mean is solved
/// first and handed to `publish_mean` before the finite-time engine runs.
pub fn solve(model:&Model,options:&Options,publish_mean:&mut dyn FnMut(&Value))->Result<Value,String>{
    let started=Instant::now();options.validate_population_tolerance()?;
    if !options.steady_only&&(!options.duration_us.is_finite()||options.duration_us<=0.||!(2..=2001).contains(&options.samples)){return Err("Invalid finite population grid".into());}
    let mut request=prepare(model)?;
    if options.steady_only{return mean_result(model,options,&mut request,started);}
    let mean=if options.include_asymptotic{
        let value=mean_result(model,options,&mut request,started);
        if let Ok(value)=&value{publish_mean(value);}
        Some(value)
    }else{None};
    let names:Vec<_>=request.groups.iter().map(|g|g.name.clone()).collect();
    let times:Vec<_>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    let finite=FiniteOptions{population_tolerance:options.population_tolerance,..Default::default()};
    let preparation=request.core.preparation_timing();
    let solution=engine::solve(request.core,&times,request.groups,finite)?;
    let d=&solution.diagnostics;
    let populations:Vec<_>=solution.populations.chunks_exact(32).map(|p|p.to_vec()).collect();
    let(trace,initial_error)=row_errors(&populations,&request.initial_populations)?;
    let checkpoints:Vec<_>=d.checkpoints.iter().map(|c|json!({
        "modes":c.modes,"basis":c.columns,"action":c.action,
        "basisEstimate":c.basis_estimate,"haloEstimate":c.halo_estimate,"elapsedMs":c.elapsed_ms
    })).collect();
    let t=&d.timing;
    let mut result=json!({"states":model.states,"timesUs":times,"populations":populations,
        "asymptoticStatus":"not-requested",
        "populationAccuracy":{"status":"estimated-converged","requestedTolerance":options.population_tolerance,
            "estimatedMaximumError":d.estimated_error,"projectionCharge":d.projection_charge,
            "includesProjectionCharge":true,"scope":{"states":32,"samples":options.samples,"groups":names}},
        "diagnostics":{"engine":ENGINE,"traceError":trace,"initialStateError":initial_error,
            "elapsedMs":started.elapsed().as_secs_f64()*1000.,"constructionMs":request.preparation_ms,
            "carrierSourceToBasis":request.axes.source_to_basis,"innerOperatorApplications":d.operator_applications,
            "method":FINITE_METHOD,"propagateMs":d.elapsed_ms,
            "errorControl":d.error_control,"rawPopulationEstimate":d.raw_estimate,
            "estimateMargin":d.estimate_margin,"boundedResonantComponents":d.bounded_resonant_components,"finalModes":d.final_modes,"finalBasis":d.final_columns,
            "imageColumns":d.image_columns,"peakTrialBytes":d.peak_trial_bytes,
            "preparationMs":{"gauge":preparation[0],"primitives":preparation[1],"library":preparation[2],"initial":preparation[3],"readoutDuals":preparation[4],"p0":preparation[5]},
            "timingMs":{"directions":t.directions,"orthogonalize":t.orthogonalize,"images":t.images,"projections":t.projections,"readoutAppend":t.readout_append,"transfer":t.transfer,"reduced":t.reduced,"readoutEvaluate":t.readout_evaluate,"dualPrepare":t.dual_prepare,"basisEstimate":t.basis_estimate,"haloEstimate":t.halo_estimate},
            "maximumImaginaryPopulation":d.maximum_imaginary_population,"populationCheckpoints":checkpoints}});
    match mean{
        None=>{}
        Some(Ok(mean))=>merge_mean(&mut result,mean),
        Some(Err(reason))=>{result["asymptoticStatus"]=json!("not-converged");result["asymptoticReason"]=json!(reason);}
    }
    Ok(result)
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn a_finite_result_carries_its_mean_and_keeps_its_own_diagnostics(){
        let mut finite=json!({"timesUs":[0.,1.],"populations":[[1.],[1.]],"asymptoticStatus":"not-requested",
            "diagnostics":{"engine":ENGINE,"traceError":1e-12,"elapsedMs":9.,"innerOperatorApplications":5,"method":FINITE_METHOD}});
        let mean=json!({"timesUs":[0.],"populations":[[0.5]],"asymptoticStatus":"stationary","asymptoticReason":"unique",
            "asymptoticPopulations":[1.],"asymptoticDensityReal":[1.],"asymptoticDensityImag":[0.],
            "diagnostics":{"engine":ENGINE,"traceError":0.,"elapsedMs":2.,"innerOperatorApplications":3,
                "asymptoticMethod":MEAN_METHOD,"residual":1e-9,"meanMs":1.}});
        merge_mean(&mut finite,mean);
        assert_eq!(finite["asymptoticStatus"],"stationary");
        assert_eq!(finite["asymptoticReason"],"unique");
        assert_eq!((&finite["asymptoticPopulations"],&finite["asymptoticDensityReal"],&finite["asymptoticDensityImag"]),(&json!([1.]),&json!([1.]),&json!([0.])));
        assert_eq!((&finite["timesUs"],&finite["populations"]),(&json!([0.,1.]),&json!([[1.],[1.]])));
        let d=&finite["diagnostics"];
        assert_eq!((d["traceError"].as_f64(),d["elapsedMs"].as_f64(),d["innerOperatorApplications"].as_i64(),d["method"].as_str()),
            (Some(1e-12),Some(9.),Some(5),Some(FINITE_METHOD)));
        assert_eq!((d["asymptoticMethod"].as_str(),d["residual"].as_f64(),d["meanMs"].as_f64()),(Some(MEAN_METHOD),Some(1e-9),Some(1.)));
    }
}
