//! Experimental complete period matrix and exact integer matrix powers.
//!
//! This path retains every Hermitian density coordinate. It uses no Arnoldi
//! projection, eigendecomposition, frequency averaging, or mode removal.
//! Each period matrix is a full numerical time-step composition; independent
//! M and 3M+1 grids are compared at every actual requested output time.
use crate::{build_generator,expand_density,positivity,Lapack,Model,Options};
use crate::cptp_period::Action;
use crate::fixed_etd::Report;
use serde_json::json;
use std::{collections::BTreeMap,time::Instant};

#[path="period_clock.rs"]
mod period_clock;
use period_clock::Clock;

const POPULATION_TOLERANCE:f64=2e-4;
/// Experimental allocation within the requested 1e-3 population budget.
/// Trace/PSD and Krylov checks are independent and are never relaxed here.
pub(crate) fn population_tolerance()->f64{
    std::env::var("ATOM_PERIOD_POPULATION_TOLERANCE").ok().and_then(|s|s.parse::<f64>().ok())
        .filter(|v|v.is_finite()&&*v>=1e-7&&*v<=5e-4).unwrap_or(POPULATION_TOLERANCE)
}
fn debug()->bool{std::env::var_os("ATOM_SOLVER_DEBUG").is_some()||std::env::var_os("ATOM_FULL_PERIOD_PROFILE").is_some()}
pub fn supports(model:&Model)->bool{!model.drives.is_empty()&&Clock::new(model).is_ok()}

/// H=I-2vv^T sends e0 to I/sqrt(n) in the population block. The other
/// n²-n coordinates already have the orthonormal Hermitian sqrt(2) weights.
struct TraceBasis {n:usize,d:usize,v:Vec<f64>}
impl TraceBasis {
    fn new(n:usize)->Self{
        let mut v=vec![-1./(n as f64).sqrt();n];v[0]+=1.;
        let length=v.iter().map(|value|value*value).sum::<f64>().sqrt();
        if length>0.{for value in &mut v{*value/=length;}}
        Self{n,d:n*n,v}
    }
    fn reflect(&self,x:&mut[f64]){
        let coefficient=2.*self.v.iter().zip(&x[..self.n]).map(|(a,b)|a*b).sum::<f64>();
        for (value,&v) in x[..self.n].iter_mut().zip(&self.v){*value-=coefficient*v;}
    }
    fn forward(&self,x:&[f64])->Vec<f64>{
        let mut y=x.to_vec();for value in &mut y[self.n..]{*value*=2_f64.sqrt();}self.reflect(&mut y);y
    }
    fn backward(&self,x:&[f64])->Vec<f64>{
        let mut y=x.to_vec();self.reflect(&mut y);for value in &mut y[self.n..]{*value/=2_f64.sqrt();}y
    }
    fn matrix(&self,raw:Vec<f64>)->Result<(Vec<f64>,f64),String>{
        if raw.len()!=self.d*self.d||raw.iter().any(|v|!v.is_finite()){return Err("Invalid complete period matrix".into());}
        let mut result=raw;
        for j in 0..self.d{let input_weight=if j<self.n{1.}else{2_f64.sqrt()};for i in 0..self.d{let output_weight=if i<self.n{1.}else{2_f64.sqrt()};result[i+j*self.d]*=output_weight/input_weight;}}
        // Left H: only the population rows are touched.
        for column in result.chunks_exact_mut(self.d){self.reflect(column);}
        // Right H: only the population columns are touched.
        for i in 0..self.d{
            let coefficient=2.*self.v.iter().enumerate().map(|(j,&v)|result[i+j*self.d]*v).sum::<f64>();
            for (j,&v) in self.v.iter().enumerate(){result[i+j*self.d]-=coefficient*v;}
        }
        let trace_error=trace_row(&result,self.d);
        if trace_error>1e-10{return Err(format!("Full period trace-row defect requires refinement: {trace_error:e}"));}
        set_trace_row(&mut result,self.d);Ok((result,trace_error))
    }
}
fn trace_row(matrix:&[f64],d:usize)->f64{(0..d).map(|j|(matrix[j*d]-if j==0{1.}else{0.}).abs()).fold(0.,f64::max)}
fn set_trace_row(matrix:&mut[f64],d:usize){for j in 0..d{matrix[j*d]=if j==0{1.}else{0.};}}

/// Stream binary powers through all sample RHS columns. At most one full
/// power plus its square is live, never a table of 24 complete matrices.
pub(crate) fn integer_actions(lap:&Lapack,mut power:Vec<f64>,initial:&[f64],exponents:&[u64])->Result<(Vec<f64>,usize),String>{
    let d=initial.len();let samples=exponents.len();let maximum=exponents.iter().copied().max().unwrap_or(0);
    let bits=(64-maximum.leading_zeros())as usize;
    let mut rhs:Vec<_>=(0..samples).flat_map(|_|initial.iter().copied()).collect();let mut squares=0;
    for bit in 0..bits{
        let selected:Vec<_>=exponents.iter().enumerate().filter_map(|(i,&count)|((count>>bit)&1==1).then_some(i)).collect();
        if !selected.is_empty(){
            let columns:Vec<_>=selected.iter().flat_map(|&i|rhs[i*d..(i+1)*d].iter().copied()).collect();
            let advanced=lap.multiply(&power,&columns,d,selected.len(),d);
            if advanced.iter().any(|x|!x.is_finite()){return Err("Non-finite full-period matrix power action".into());}
            for (j,&i) in selected.iter().enumerate(){rhs[i*d..(i+1)*d].copy_from_slice(&advanced[j*d..(j+1)*d]);}
        }
        if bit+1<bits{
            power=lap.multiply(&power,&power,d,d,d);squares+=1;
            if power.iter().any(|x|!x.is_finite()){return Err("Non-finite squared full-period matrix".into());}
            let defect=trace_row(&power,d);if defect>1e-10{return Err(format!("Squared complete period matrix lost trace identity: {defect:e}"));}
            // Preserve ONLY the exact TP row. The nonunital first column,
            // every coherence, and every other entry remain untouched.
            set_trace_row(&mut power,d);
        }
    }
    Ok((rhs,squares))
}

struct Grid {states:Vec<Vec<f64>>,scheme:String,is_cptp:bool,build_profile:Option<crate::cptp_period::BuildProfile>,build_ms:f64,powers_ms:f64,partial_ms:f64,squares:usize,trace_defect:f64}
fn grid(lap:&Lapack,model:&Model,initial:&[f64],times:&[f64],clock:&Clock,steps:usize)->Result<Grid,String>{
    let period=clock.period();
    if std::env::var("ATOM_PERIOD_EXTRAPOLATE").is_ok_and(|value|value=="1"){
        return extrapolated_grid(lap,model,initial,times,clock,steps);
    }
    let trace=debug();let build_started=Instant::now();
    if trace{eprintln!("Full period build begin steps={steps} dimension={}",model.dimension*model.dimension);}
    let action=if std::env::var("ATOM_PERIOD_ORDER").is_ok_and(|value|value=="4"){Action::new_symmetric_fourth_order(lap,model,period,steps)?}else{Action::new(lap,model,period,steps)?};let basis=TraceBasis::new(model.dimension);
    if action.dimension()!=basis.d{return Err("Full period action dimension mismatch".into());}
    let frame_powers=std::env::var("ATOM_PERIOD_FRAME_POWERS").is_ok_and(|value|value=="1")&&action.has_bath_frame();
    // A unitary basis change preserves trace and the Hilbert--Schmidt metric.
    // Keep this one basis through all integer powers; only the initial state
    // and final sampled states need conjugation, not 1024 basis columns twice.
    let raw=if frame_powers{action.full_matrix_in_bath_frame(lap)?}else{action.full_matrix(lap)?};
    let (matrix,trace_defect)=basis.matrix(raw)?;
    let build_ms=build_started.elapsed().as_secs_f64()*1000.;
    if trace{eprintln!("Full period build done steps={steps} buildMs={build_ms:.3} traceDefect={trace_defect:e}");}
    let parts:Vec<_>=times.iter().map(|&time|clock.split(time)).collect::<Result<_,_>>()?;
    let exponents:Vec<_>=parts.iter().map(|&(count,_)|count).collect();
    let power_initial=if frame_powers{action.state_into_bath_frame(lap,initial)?}else{initial.to_vec()};
    let weighted=basis.forward(&power_initial);
    let powers_started=Instant::now();
    let (values,squares)=integer_actions(lap,matrix,&weighted,&exponents)?;
    let powers_ms=powers_started.elapsed().as_secs_f64()*1000.;
    if trace{eprintln!("Full period powers done steps={steps} squares={squares} powersMs={powers_ms:.3}");}
    let partial_started=Instant::now();
    let mut states=if frame_powers{
        let boundaries:Vec<_>=values.chunks_exact(basis.d).map(|column|basis.backward(column)).collect();
        let phases:Vec<_>=parts.iter().map(|&(_,phase)|phase).collect();
        action.partial_many_from_bath_frame(lap,&boundaries,&phases)?
    }else if std::env::var("ATOM_PERIOD_BATCH_PARTIAL").is_ok_and(|value|value=="1"){
        let boundaries:Vec<_>=values.chunks_exact(basis.d).map(|column|basis.backward(column)).collect();
        let phases:Vec<_>=parts.iter().map(|&(_,phase)|phase).collect();
        action.partial_many(lap,&boundaries,&phases)?
    }else{
        let mut result=Vec::with_capacity(times.len());
        for (sample,&(_,phase)) in parts.iter().enumerate(){
            let boundary=basis.backward(&values[sample*basis.d..(sample+1)*basis.d]);
            result.push(action.partial(lap,&boundary,phase)?);
        }
        result
    };
    if let Some(first)=states.first_mut(){*first=initial.to_vec();}
    if states.iter().flatten().any(|value|!value.is_finite()){return Err("Non-finite full-period trajectory".into());}
    let partial_ms=partial_started.elapsed().as_secs_f64()*1000.;
    if trace{eprintln!("Full period partials done steps={steps} partialMs={partial_ms:.3}");}
    Ok(Grid{states,scheme:action.name().into(),is_cptp:action.is_cptp(),build_profile:action.full_matrix_profile(),build_ms,powers_ms,partial_ms,squares,trace_defect})
}
/// Numerical multi-product extrapolation of the COMPLETE map, not a change
/// to H or C. (4 P_{2M} - P_M)/3 cancels the leading symmetric-splitting
/// error. Negative combination weights mean finite-grid CP is NOT claimed.
/// Both maps use original physical coordinates, so no assumption about
/// independent eigensolver choices in their bath frames enters the sum.
fn extrapolated_grid(lap:&Lapack,model:&Model,initial:&[f64],times:&[f64],clock:&Clock,steps:usize)->Result<Grid,String>{
    let period=clock.period();
    let started=Instant::now();let basis=TraceBasis::new(model.dimension);
    let coarse=Action::new(lap,model,period,steps)?;
    let fine=Action::new(lap,model,period,steps.checked_mul(2).ok_or("Extrapolation grid overflow")?)?;
    let a=coarse.full_matrix(lap)?;let mut raw=fine.full_matrix(lap)?;
    for (b,a) in raw.iter_mut().zip(a){*b=(4.* *b-a)/3.;}
    let(matrix,trace_defect)=basis.matrix(raw)?;let build_ms=started.elapsed().as_secs_f64()*1000.;
    let parts:Vec<_>=times.iter().map(|&time|clock.split(time)).collect::<Result<_,_>>()?;
    let exponents:Vec<_>=parts.iter().map(|&(count,_)|count).collect();
    let started=Instant::now();let(values,squares)=integer_actions(lap,matrix,&basis.forward(initial),&exponents)?;
    let powers_ms=started.elapsed().as_secs_f64()*1000.;let started=Instant::now();
    let boundaries:Vec<_>=values.chunks_exact(basis.d).map(|column|basis.backward(column)).collect();
    let phases:Vec<_>=parts.iter().map(|&(_,phase)|phase).collect();
    let a=coarse.partial_many(lap,&boundaries,&phases)?;let mut states=fine.partial_many(lap,&boundaries,&phases)?;
    for (b,a) in states.iter_mut().zip(a){for(b,a)in b.iter_mut().zip(a){*b=(4.* *b-a)/3.;}}
    if let Some(first)=states.first_mut(){if times.first()==Some(&0.){*first=initial.to_vec();}}
    if states.iter().flatten().any(|v|!v.is_finite()){return Err("Non-finite extrapolated full-period trajectory".into());}
    let partial_ms=started.elapsed().as_secs_f64()*1000.;
    if debug(){eprintln!("Extrapolated full period M={steps}/{} buildMs={build_ms:.3} powersMs={powers_ms:.3} partialMs={partial_ms:.3}",2*steps);}
    Ok(Grid{states,scheme:"Full-map Richardson extrapolation of M/2M Strang grids; not finite-grid CPTP".into(),is_cptp:false,
        build_profile:None,build_ms,powers_ms,partial_ms,squares,trace_defect})
}
fn groups(model:&Model)->Vec<Vec<usize>>{
    let mut groups=BTreeMap::<String,Vec<usize>>::new();
    for (i,name) in model.states.iter().take(model.dimension).enumerate(){groups.entry(name.split('-').take(2).collect::<Vec<_>>().join("-")).or_default().push(i);}
    groups.into_values().collect()
}
pub fn solve(lap:&Lapack,model:&Model,options:&Options)->Result<Report,String>{
    let started=Instant::now();let clock=Clock::new(model)?;let period=clock.period();
    if options.samples<2||options.samples>100000||!options.duration_us.is_finite()||options.duration_us<=0.{return Err("Invalid full-period output grid".into());}
    let generator=build_generator(model)?;let groups=groups(model);
    let times:Vec<_>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    let mut steps=std::env::var("ATOM_FULL_PERIOD_STEPS").ok().and_then(|s|s.parse::<usize>().ok()).unwrap_or(24).clamp(2,2048);
    let mut attempts=Vec::new();let mut reason=String::new();
    let nested=std::env::var("ATOM_PERIOD_NESTED_REFINEMENT").is_ok_and(|value|value=="1");
    // Request-local continuation only: a completed fine trajectory becomes
    // the next coarse trajectory. No result survives this solve call.
    let mut retained:Option<Grid>=None;let mut total_stage_ms=[0.;3];let mut total_charged_ms=0.;
    for _ in 0..6{
        let fine_steps=if nested{2*steps+1}else{3*steps+1};let coarse_reused=retained.is_some();
        let mut coarse_charged_ms=0.;let mut fine_charged_ms=0.;let mut failed_stage="coarse";
        let pair=(||{
            let coarse=if let Some(previous)=retained.take(){
                if debug(){eprintln!("Full period reuse previous fine as coarse steps={steps} chargedMs=0");}previous
            }else{
                let before=Instant::now();let result=grid(lap,model,&generator.initial,&times,&clock,steps);coarse_charged_ms=before.elapsed().as_secs_f64()*1000.;total_charged_ms+=coarse_charged_ms;
                if let Ok(value)=&result{total_stage_ms[0]+=value.build_ms;total_stage_ms[1]+=value.powers_ms;total_stage_ms[2]+=value.partial_ms;}result?
            };
            failed_stage="fine";let before=Instant::now();let result=grid(lap,model,&generator.initial,&times,&clock,fine_steps);fine_charged_ms=before.elapsed().as_secs_f64()*1000.;total_charged_ms+=fine_charged_ms;
            if let Ok(value)=&result{total_stage_ms[0]+=value.build_ms;total_stage_ms[1]+=value.powers_ms;total_stage_ms[2]+=value.partial_ms;}
            Ok::<_,String>((coarse,result?))
        })();
        let (coarse,fine)=match pair{Ok(pair)=>pair,Err(error)=>{
            if debug(){eprintln!("Full period grid failure steps={steps} stage={failed_stage}: {error}");}
            attempts.push(json!({"coarseSteps":steps,"independentSteps":fine_steps,"failureStage":failed_stage,"coarseReusedWithinRefinement":coarse_reused,"coarseChargedMs":coarse_charged_ms,"fineChargedMs":fine_charged_ms,"chargedMs":coarse_charged_ms+fine_charged_ms,"error":error}));
            reason=error;steps=if nested{if failed_stage=="coarse"{fine_steps}else{2*fine_steps+1}}else{2*steps};continue;
        }};
        let mut population_difference=0_f64;let mut group_difference=0_f64;let mut trace_error=0_f64;let mut positivity_error=0_f64;
        for (a,b) in coarse.states.iter().zip(&fine.states){
            for i in 0..model.dimension{population_difference=population_difference.max((a[i]-b[i]).abs());}
            for group in &groups{group_difference=group_difference.max(group.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}
            trace_error=trace_error.max((b[..model.dimension].iter().sum::<f64>()-1.).abs());
            positivity_error=positivity_error.max(positivity(lap,&expand_density(&generator.coords,b,model.dimension),model.dimension)?);
        }
        if debug(){eprintln!("Full period independent comparison M={steps} fine={fine_steps} reusedCoarse={coarse_reused} population={population_difference:e} Fgroup={group_difference:e} trace={trace_error:e} positivity={positivity_error:e}");}
        attempts.push(json!({"coarseSteps":steps,"independentSteps":fine_steps,"coarseReusedWithinRefinement":coarse_reused,"coarseChargedMs":coarse_charged_ms,"fineChargedMs":fine_charged_ms,"chargedMs":coarse_charged_ms+fine_charged_ms,
            "coarseBuildProfile":if coarse_reused{None}else{coarse.build_profile.as_ref()},"fineBuildProfile":fine.build_profile,
            "coarseBuildMs":if coarse_reused{0.}else{coarse.build_ms},"fineBuildMs":fine.build_ms,"coarsePowersMs":if coarse_reused{0.}else{coarse.powers_ms},"finePowersMs":fine.powers_ms,"coarsePartialMs":if coarse_reused{0.}else{coarse.partial_ms},"finePartialMs":fine.partial_ms,
            "populationDifference":population_difference,"groupDifference":group_difference,"traceError":trace_error,"positivityError":positivity_error}));
        if population_difference.max(group_difference)<=population_tolerance()&&trace_error<=1e-7&&positivity_error<=1e-5{
            let diagnostics=json!({"engine":"rust-full-period-experiment","method":"Complete full-density period matrix / streamed integer matrix powers","periodAction":fine.scheme,"finiteGridCptp":fine.is_cptp,
                "elapsedMs":started.elapsed().as_secs_f64()*1000.,"periodUs":period,"maximumPeriods":clock.split(options.duration_us)?.0,"periodClock":clock.source(),
                "physicalHilbertDimension":model.dimension,"fullLiouvilleDimension":model.dimension*model.dimension,"periodSteps":fine_steps,"squares":fine.squares,
                "buildMs":total_stage_ms[0],"powersMs":total_stage_ms[1],"partialMs":total_stage_ms[2],"chargedMs":total_charged_ms,"nestedRefinement":nested,
                "populationRefinementTolerance":population_tolerance(),"requestedPopulationAccuracy":1e-3,"populationRefinementError":population_difference,"groupPopulationRefinementError":group_difference,"traceError":trace_error,"positivityError":positivity_error,
                "periodTraceDefect":coarse.trace_defect.max(fine.trace_defect),"attempts":attempts,
                "refinementGrids":if nested{"Consecutive coprime M,2M+1 grids; each completed grid evaluated once within this request"}else{"Independent M,3M+1 grids"},
                "errorScope":"Complete density coordinates and all requested times; independent period/phase discretization comparison, exact TP-row preservation, trace and PSD checks. No eigenvalue clipping, physical averaging, or inferred infinite-time mixture."});
            return Ok(Report{times,states:fine.states,diagnostics,trace_error,positivity_error});
        }
        reason=format!("population={population_difference:e}, Fgroup={group_difference:e}, trace={trace_error:e}, positivity={positivity_error:e}");
        if nested{retained=Some(fine);steps=fine_steps;}else{steps*=2;}
    }
    Err(format!("Complete period-matrix independent-grid comparison did not converge: {reason}"))
}

#[cfg(test)]
mod tests{
    use super::*;use crate::{Drive,Operator};
    #[test]
    fn extrapolated_period_matches_noncommuting_hamiltonian_decay_and_fractional_times(){
        let zero=Operator{real:vec![0.;4],imag:vec![0.;4]};
        let model=Model{frequency_coordinates:None,dimension:2,states:vec!["ground-1-0".into(),"D2-1-0".into()],
            h_real:vec![0.2,0.8,0.8,-0.2],h_imag:vec![0.;4],
            collapse:vec![Operator{real:vec![0.,0.7_f64.sqrt(),0.,0.],imag:vec![0.;4]}],
            initial_index:1,initial_real:None,initial_imag:None,time_dependent:true,
            drives:vec![Drive{omega:std::f64::consts::TAU/0.4,cos:zero.clone(),sin:zero}]};
        let lap=Lapack::load().unwrap();let g=build_generator(&model).unwrap();
        let times=vec![0.,0.033,0.4,0.913,4.,20.];
        let actual=extrapolated_grid(&lap,&model,&g.initial,&times,&Clock::new(&model).unwrap(),8).unwrap();
        assert!(!actual.is_cptp);
        let mut dense=vec![0.;16];for(j,column)in g.matrix.iter().enumerate(){for &(i,value)in column{dense[i+j*4]+=value;}}
        let exact=crate::dense_expm::action(&lap,&dense,&g.initial,&times).unwrap();
        let error=actual.states.iter().zip(exact).flat_map(|(a,b)|a.iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max);
        assert!(error<2e-6,"Complete extrapolated density error {error:e}");
        for state in &actual.states{assert!((state[0]+state[1]-1.).abs()<1e-10);}
    }
    #[test]
    fn full_period_constant_decay_retains_ten_million_period_slow_population(){
        let rate=1e-5_f64;let zero=Operator{real:vec![0.;4],imag:vec![0.;4]};
        let model=Model{ frequency_coordinates:None,dimension:2,states:vec![],h_real:vec![0.;4],h_imag:vec![0.;4],collapse:vec![Operator{real:vec![0.,rate.sqrt(),0.,0.],imag:vec![0.;4]}],initial_index:1,initial_real:None,initial_imag:None,time_dependent:true,drives:vec![Drive{omega:std::f64::consts::TAU/0.01,cos:zero.clone(),sin:zero}]};
        let lap=Lapack::load().unwrap();let options=Options{duration_us:100000.,samples:5,include_asymptotic:false,..Options::default()};
        let actual=solve(&lap,&model,&options).unwrap();
        for (&time,state) in actual.times.iter().zip(actual.states){let expected=(-rate*time).exp();assert!((state[1]-expected).abs()<2e-7);assert!((state[0]-(1.-expected)).abs()<2e-7);}
    }
    #[test]
    fn full_period_commuting_drive_matches_complete_matrix_exponential(){
        let n=2;let omega=3.7;let unit=vec![0.,1.,1.,0.];
        let model=Model{ frequency_coordinates:None,dimension:n,states:vec![],h_real:unit.iter().map(|v|0.3*v).collect(),h_imag:vec![0.;4],collapse:vec![],initial_index:0,initial_real:None,initial_imag:None,time_dependent:true,
            drives:vec![Drive{omega,cos:Operator{real:unit.iter().map(|v|0.2*v).collect(),imag:vec![0.;4]},sin:Operator{real:unit.iter().map(|v|-0.1*v).collect(),imag:vec![0.;4]}}]};
        let lap=Lapack::load().unwrap();let options=Options{duration_us:10.,samples:17,include_asymptotic:false,..Options::default()};
        let actual=solve(&lap,&model,&options).unwrap();
        let mut constant=model.clone();constant.h_real=unit;constant.drives.clear();constant.time_dependent=false;
        let g=build_generator(&constant).unwrap();let mut dense=vec![0.;16];for(j,col)in g.matrix.iter().enumerate(){for &(i,value)in col{dense[i+j*4]+=value;}}
        let angles:Vec<_>=actual.times.iter().map(|&time|0.3*time+0.2*(omega*time).sin()/omega-0.1*(1.-(omega*time).cos())/omega).collect();
        let expected=crate::dense_expm::action(&lap,&dense,&g.initial,&angles).unwrap();
        let error=actual.states.iter().zip(expected).flat_map(|(a,b)|a.iter().zip(b).map(|(a,b)|(a-b).abs())).fold(0.,f64::max);
        assert!(error<3e-4,"Full period complete-density discrepancy={error:e}");
    }
}
