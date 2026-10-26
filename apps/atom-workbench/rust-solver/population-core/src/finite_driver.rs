//! Complete request-local finite evolution using the established empirical
//! population controller. No dual-estimator research path is used here.
//! Every trial column retains original E/F; domain expansion reuses only
//! current-request operator images. No historical trajectory is an input.
use super::*;
use super::runtime_core::{RuntimeCore,Candidate,AdaptiveConstruction};
use super::runtime_core::workspace::DirectionPolicy;
use super::adaptive_direction::CheckpointPolicy;
use super::domain_transfer::TransferOptions;
use super::residual_proposal::ProposalOptions;
use super::population_control::{PopulationControl,ControlOptions,RequestedGroup,ControlFrame,
    FrameContext,OutputDelta,OutputCheck,WorkDecision,WorkEvidence,Action};

#[derive(Clone,Debug)]
pub struct FiniteOptions {
    pub population_tolerance:f64,
    pub first_slow_columns:usize,
    pub basis_block:usize,
    pub maximum_basis:usize,
    pub maximum_domain_order:usize,
    pub maximum_correction_rounds:usize,
    pub wall_seconds:f64,
    pub fast_pair:bool,
    /// Population-only output may be projected with its measured change
    /// charged to the SAME user budget. Raw candidate/density are retained.
    pub constrain_population_output:bool,
}
impl Default for FiniteOptions {
    fn default()->Self {Self{population_tolerance:0.005,first_slow_columns:16,basis_block:8,
        maximum_basis:256,maximum_domain_order:5,maximum_correction_rounds:24,
        wall_seconds:120.,fast_pair:true,constrain_population_output:true}}
}
#[derive(Clone,Debug)]
pub struct FiniteCheckpoint {
    pub domain_order:usize,pub modes:usize,pub columns:usize,pub generation:u64,
    pub zero_lift_only:bool,pub work_ms:f64,pub check_ms:f64,pub elapsed_ms:f64,
    pub decision:WorkDecision,pub output:OutputCheck,
}
#[derive(Clone,Debug)]
pub struct FiniteDiagnostics {
    pub requested_tolerance:f64,pub estimated_error:f64,pub projection_charge:f64,
    pub accepted:bool,pub empirical_only:bool,pub rigorous_error_bound:bool,
    pub exact_zero_fourier:bool,pub final_domain_order:usize,pub final_basis:usize,
    pub fast_pair_columns:usize,pub original_physics_changed:bool,
    pub historical_answer_used:bool,pub elapsed_ms:f64,
    pub checkpoints:Vec<FiniteCheckpoint>,
}
pub struct FiniteSolution {
    pub core:RuntimeCore,pub candidate:Candidate,
    /// Returned probability populations, row-major [time,state].
    pub populations:Vec<f64>,
    pub diagnostics:FiniteDiagnostics,
}

/// Signed graph ball of the actual declared drive charges. No beam-count,
/// state-selection or geometry-specific domain is inserted.
pub fn support_domain(axes:usize,steps:&[Charge],order:usize)->Result<Vec<Charge>,String>{
    if steps.iter().any(|q|q.len()!=axes){return Err("Drive-charge axes differ".into());}
    let mut result:BTreeSet<Charge>=BTreeSet::from([vec![0i32;axes]]);let mut directions:BTreeSet<Charge>=BTreeSet::new();
    for step in steps {
        directions.insert(step.clone());
        directions.insert(step.iter().map(|x|x.checked_neg().ok_or("Drive-charge negation overflow")).collect::<Result<Charge,_>>()?);
    }
    for _ in 0..order {
        let old:Vec<_>=result.iter().cloned().collect();
        for q in old {for step in &directions {
            result.insert(q.iter().zip(step).map(|(a,b)|a.checked_add(*b).ok_or("Fourier graph-ball charge overflow")).collect::<Result<Charge,_>>()?);
        }}
    }
    Ok(result.into_iter().collect())
}

/// Numerical probability display constraint only, not a density repair.
/// Its state/group changes are charged before a trajectory can be returned.
pub fn constrain_populations(raw:&[C],n:usize,groups:&[RequestedGroup])->Result<(Vec<f64>,f64),String>{
    if n==0||raw.len()%n!=0||raw.iter().any(|x|!x.finite()){
        return Err("Complete finite population rows are required".into());
    }
    let mut output=Vec::with_capacity(raw.len());let mut maximum=0_f64;
    for row in raw.chunks_exact(n) {
        let top=row.iter().map(|x|x.re).fold(f64::NEG_INFINITY,f64::max);
        let mut ordered:Vec<_>=row.iter().map(|x|x.re-top).collect();
        ordered.sort_by(|a,b|b.total_cmp(a));
        let(mut sum,mut threshold)=(ordered[0],ordered[0]-1.);
        for index in 1..n {
            let next=sum+ordered[index];let proposed=(next-1.)/(index+1)as f64;
            if ordered[index]<=proposed{break;}
            sum=next;threshold=proposed;
        }
        let projected:Vec<_>=row.iter().map(|x|(x.re-top-threshold).max(0.)).collect();
        if projected.iter().any(|x|!x.is_finite())||(projected.iter().sum::<f64>()-1.).abs()>64.*f64::EPSILON*n as f64 {
            return Err("Probability output projection is numerically unresolved".into());
        }
        for(a,b)in projected.iter().zip(row){maximum=maximum.max((a-b.re).abs());}
        for group in groups {
            if group.indices.iter().any(|&i|i>=n){return Err("Population output group outside state range".into());}
            maximum=maximum.max(group.indices.iter().map(|&i|projected[i]-row[i].re).sum::<f64>().abs());
        }
        output.extend(projected);
    }
    Ok((output,maximum))
}

fn check_deadline(start:Instant,seconds:f64,order:usize,columns:usize)->Result<(),String>{
    if start.elapsed().as_secs_f64()>seconds {
        Err(format!("Population work limit reached after {:.3}s at Fourier order{order}, basis{columns}; requested tolerance was not relaxed",start.elapsed().as_secs_f64()))
    }else{Ok(())}
}

pub fn solve(mut core:RuntimeCore,times:&[f64],groups:Vec<RequestedGroup>,options:FiniteOptions)->Result<FiniteSolution,String>{
    let start=Instant::now();
    if times.len()<2||times[0]!=0.||times.iter().any(|x|!x.is_finite()||*x<0.)
        ||times.windows(2).any(|x|x[0]>=x[1])||!options.wall_seconds.is_finite()||options.wall_seconds<=0.
        ||options.first_slow_columns==0||options.basis_block==0||options.maximum_basis<options.first_slow_columns
        ||options.maximum_domain_order==0||options.maximum_correction_rounds<2 {
        return Err("Invalid finite population time grid or work budget".into());
    }
    let n=core.prepared().primitives().hilbert.source.n;
    let initial:Vec<_>=(0..n).map(|i|core.initial().original()[i*n+i].re).collect();
    let control=PopulationControl::new(core.prepared(),ControlOptions{epsilon:options.population_tolerance,..Default::default()},times,&initial,groups)?;
    let source=&core.prepared().primitives().hilbert.source;
    let axes=source.phases.axes.len();let steps=source.phases.drive_indices.clone();
    // EXPERIMENT (temporary): pole independent of output sampling.
    let pole_factor=std::env::var("ATOM_EXP_POLE_C").ok().and_then(|v|v.parse::<f64>().ok()).unwrap_or((times.len()-1)as f64);
    let pole=pole_factor/times[times.len()-1];
    let pole_times=[0.,times[times.len()-1]/pole_factor];
    let policy=DirectionPolicy::Adaptive(AdaptiveConstruction{construction_budget_scale:options.population_tolerance,
        maximum_inner_steps:12,checkpoint_policy:CheckpointPolicy::Cost});
    let modes=support_domain(axes,&steps,1)?;
    let mut workspace=core.cold_workspace(&modes,times,pole,policy)?;
    let mut fast_columns=0;
    if options.fast_pair {
        let selection=core.select_fast_pair(&pole_times,pole)?;
        if selection.pole.is_some() {
            let old=workspace.columns();core.add_fast_pair(&mut workspace,&selection)?;
            fast_columns=workspace.columns()-old;
        }
    }
    let mut target=options.first_slow_columns.saturating_add(fast_columns).min(options.maximum_basis);
    let mut began=Instant::now();workspace.extend_to(target)?;
    if !workspace.statistics().complete_frontier_generation{workspace.finish_frontier_generation()?;}
    let mut work_ms=began.elapsed().as_secs_f64()*1000.;
    let mut current=workspace.candidate()?;
    let mut order=1;let mut generation=0u64;let mut round=0;
    let mut frames=Vec::<ControlFrame>::new();let mut changes=Vec::<OutputDelta>::new();
    let mut last_population:Option<Vec<C>>=None;
    let mut previous_domain:Option<(ControlFrame,Vec<C>)>=None;
    let mut zero_lift=false;let mut history=Vec::new();
    let mut last_correction_cost:Option<f64>=None;let mut last_transfer_cost=0.;
    loop {
        check_deadline(start,options.wall_seconds,order,current.dimension)?;
        if !current.diagnostics.complete_frontier_generation {
            began=Instant::now();let before=workspace.columns();workspace.finish_frontier_generation()?;
            if workspace.columns()==before{return Err("Incomplete numerical frontier did not advance; no plateau was accepted".into());}
            work_ms+=began.elapsed().as_secs_f64()*1000.;current=workspace.candidate()?;
            continue;
        }
        let checking=Instant::now();generation+=1;
        let output=control.assess_candidate(core.prepared(),&current)?;
        if !output.numerical_invariants_passed{return Err("Native candidate failed fixed trace/initial/REAL invariants".into());}
        let new_columns=frames.last().map_or(current.dimension,|old|current.dimension.saturating_sub(old.basis_columns));
        if !frames.is_empty()&&new_columns==0{return Err("No genuinely expanded trial space is available; population accuracy remains unresolved".into());}
        let frame=control.frame_from_candidate(core.prepared(),&current,FrameContext{
            generation,domain_order:order,new_directions:if zero_lift{0}else{new_columns},zero_lift_only:zero_lift,
            retains_previous_trial_space:true,all_current_residual_source_columns:order>1})?;
        if let Some(previous)=&last_population{changes.push(control.difference(previous,&current.populations)?);}
        frames.push(frame.clone());
        let domain_change=previous_domain.as_ref().map(|(_,p)|control.difference(p,&current.populations)).transpose()?;
        let proof=control.prove_exact_zero_fourier(core.prepared(),&current.modes)?;
        let next_ratio=if order<options.maximum_domain_order {
            support_domain(axes,&steps,order+1)?.len()as f64/current.modes.len()as f64
        }else{1.};
        let expansion_cost=last_correction_cost.map(|cost|last_transfer_cost*next_ratio+2.*cost);
        let mut decision=control.decide_mature(WorkEvidence{frames:&frames,changes:&changes,current_output:&output,
            domain_change:domain_change.as_ref(),previous_domain:previous_domain.as_ref().map(|(f,_)|f),
            previous_domain_basis_passed:previous_domain.is_some(),exact_zero:if proof.proven{Some(&proof)}else{None},
            basis_cost_ms:last_correction_cost,expansion_cost_ms:expansion_cost});
        let mut final_values=None;let mut projection_charge=0.;
        if decision.action==Action::EstimateAccepted {
            let(raw,charge)=if options.constrain_population_output {
                constrain_populations(&current.populations,n,control.groups())?
            }else{(current.populations.iter().map(|x|x.re).collect(),0.)};
            projection_charge=charge;
            let error=decision.joint_indicator.or(decision.basis_indicator).ok_or("Accepted population screen has no explicit numerical indicator")?;
            if error+charge<=options.population_tolerance {final_values=Some(raw);}
            else {decision.action=Action::RefineBasis;decision.reason="Output probability correction consumes the unchanged total population budget";}
        }
        history.push(FiniteCheckpoint{domain_order:order,modes:current.modes.len(),columns:current.dimension,generation,
            zero_lift_only:zero_lift,work_ms,check_ms:checking.elapsed().as_secs_f64()*1000.,
            elapsed_ms:start.elapsed().as_secs_f64()*1000.,decision:decision.clone(),output});
        #[cfg(test)]
        whole_fixture::record(history.last().unwrap());
        if let Some(populations)=final_values {
            let diagnostics=FiniteDiagnostics{requested_tolerance:options.population_tolerance,
                estimated_error:decision.joint_indicator.or(decision.basis_indicator).unwrap()+projection_charge,
                projection_charge,accepted:true,empirical_only:true,rigorous_error_bound:false,
                exact_zero_fourier:proof.proven,final_domain_order:order,final_basis:current.dimension,
                fast_pair_columns:fast_columns,original_physics_changed:false,historical_answer_used:false,
                elapsed_ms:start.elapsed().as_secs_f64()*1000.,checkpoints:history};
            return Ok(FiniteSolution{core,candidate:current,populations,diagnostics});
        }
        match decision.action {
            Action::RefineBasis=>{
                last_population=Some(current.populations.clone());zero_lift=false;
                began=Instant::now();
                if order==1 {
                    target=workspace.columns().saturating_add(options.basis_block).min(options.maximum_basis);
                    if target<=workspace.columns(){return Err(format!("Population basis work budget exhausted at{} columns; tolerance remains{}",workspace.columns(),options.population_tolerance));}
                    let old=workspace.columns();workspace.extend_to(target)?;
                    if !workspace.statistics().complete_frontier_generation{workspace.finish_frontier_generation()?;}
                    if workspace.columns()<=old{return Err("Cold trial has no new independent direction; no same-space convergence claim".into());}
                    current=workspace.candidate()?;
                }else{
                    if round>=options.maximum_correction_rounds{return Err(format!("Residual correction work budget exhausted in Fourier order{order}"));}
                    let available=options.maximum_basis.saturating_sub(workspace.columns());
                    if available==0{return Err("Population basis budget exhausted before a qualified result".into());}
                    let correction=core.correct_workspace(&mut workspace,&current,ProposalOptions{
                        pole,duration:times[times.len()-1],maximum_rhs:4,minimum_directions:4,
                        relative_source_tail:0.05,absolute_source_tail:options.population_tolerance*times[times.len()-1].sqrt()/20.,
                        check_gram_scaling:false},available)?;
                    if correction.deferred_for_work_budget{return Err("Complete residual proposal exceeds the explicit work budget; no directions were silently dropped".into());}
                    current=correction.candidate.ok_or("Complete residual proposal supplied no resolved new direction")?;
                    round+=1;
                }
                work_ms=began.elapsed().as_secs_f64()*1000.;last_correction_cost=Some(work_ms);
            }
            Action::ExpandDomain=>{
                if order>=options.maximum_domain_order{return Err("Fourier work budget exhausted before requested population accuracy; original32 states were retained".into());}
                let new_modes=support_domain(axes,&steps,order+1)?;
                if new_modes.len()<=current.modes.len(){return Err("Domain expansion adds no actual Fourier modes and lacks an exact-zero proof".into());}
                began=Instant::now();
                let transferred=core.prepared().transfer_candidate_images(&current,&new_modes,times,pole,TransferOptions::default())?;
                last_transfer_cost=transferred.diagnostics.elapsed_ms;
                previous_domain=Some((frame,current.populations.clone()));
                workspace=core.workspace_from_transfer(&current,transferred,policy)?;
                current=workspace.candidate()?;order+=1;round=0;
                frames.clear();changes.clear();last_population=None;zero_lift=true;
                work_ms=began.elapsed().as_secs_f64()*1000.;last_correction_cost=None;
            }
            Action::RepairNumerics|Action::Unresolved=>return Err(format!("Native population control unresolved at order{order}, basis{}: {}",current.dimension,decision.reason)),
            Action::EstimateAccepted=>return Err("Accepted population evidence did not produce a valid output".into()),
        }
    }
}

/// Opt-in original-Model whole runner. Entire module and checkpoint hook are
/// absent from production; no stopping, numerical or physics rule is changed.
#[cfg(test)]
mod whole_fixture{
    use super::*;
    use super::super::coordinates::InitialView;
    use super::super::b0_inverse::P0Policy;
    use std::{cell::RefCell,path::PathBuf,panic::{catch_unwind,AssertUnwindSafe}};
    struct Capture{start:Instant,path:PathBuf,epsilon:f64,model_sha:String,stage:&'static str,
        model_ms:Option<f64>,runtime_ms:Option<f64>,checkpoints:Vec<FiniteCheckpoint>}
    thread_local!{static CAPTURE:RefCell<Option<Capture>>=const{RefCell::new(None)};}
    fn quoted(value:&str)->String{use std::fmt::Write;let mut out=String::from("\"");for c in value.chars(){match c{
        '"'=>out.push_str("\\\""),'\\'=>out.push_str("\\\\"),'\n'=>out.push_str("\\n"),'\r'=>out.push_str("\\r"),'\t'=>out.push_str("\\t"),
        c if c<'\u{20}'=>{let _=write!(out,"\\u{:04x}",c as u32);},c=>out.push(c),}}out.push('"');out}
    fn number(value:f64)->String{if value.is_finite(){value.to_string()}else{"null".into()}}
    fn optional(value:Option<f64>)->String{value.map(number).unwrap_or_else(||"null".into())}
    fn checkpoints(values:&[FiniteCheckpoint])->String{
        let rows:Vec<_>=values.iter().map(|c|format!("{{\"order\":{},\"modes\":{},\"basis\":{},\"generation\":{},\"zeroLiftOnly\":{},\"workMs\":{},\"checkMs\":{},\"elapsedMs\":{},\"action\":{},\"reason\":{},\"basisIndicator\":{},\"domainIndicator\":{},\"jointIndicator\":{},\"remainingBudget\":{},\"basisComparisons\":{},\"traceError\":{},\"imaginaryError\":{},\"initialPopulationError\":{},\"minimumPopulation\":{},\"maximumPopulation\":{},\"numericalInvariantsPassed\":{},\"rangePassed\":{}}}",
            c.domain_order,c.modes,c.columns,c.generation,c.zero_lift_only,number(c.work_ms),number(c.check_ms),number(c.elapsed_ms),quoted(c.decision.action.as_str()),quoted(c.decision.reason),
            optional(c.decision.basis_indicator),optional(c.decision.domain_indicator),optional(c.decision.joint_indicator),number(c.decision.remaining_budget),c.decision.genuine_basis_comparisons,
            number(c.output.trace_error),number(c.output.imaginary_error),number(c.output.initial_population_error),number(c.output.minimum_population),number(c.output.maximum_population),
            c.output.numerical_invariants_passed,c.output.probability_ranges_passed)).collect();format!("[{}]",rows.join(","))
    }
    fn base(c:&Capture,status:&str,error:Option<&str>)->String{
        format!("{{\"kind\":\"fresh-original-Model-native-finite\",\"status\":{},\"stage\":{},\"modelSha256\":{},\"populationTolerance\":{},\"durationUs\":20,\"samples\":200,\"physicalDimension\":32,\"elapsedTotalMs\":{},\"modelPreparationMs\":{},\"runtimePreparationMs\":{},\"error\":{},\"checkpoints\":{},\"referenceRead\":false,\"historicalAnswerOrExpectedFactorsSeeded\":false,\"nonfiniteDiagnosticNumbersRepresentedAsNull\":true}}",
            quoted(status),quoted(c.stage),quoted(&c.model_sha),number(c.epsilon),number(c.start.elapsed().as_secs_f64()*1000.),optional(c.model_ms),optional(c.runtime_ms),
            error.map(quoted).unwrap_or_else(||"null".into()),checkpoints(&c.checkpoints))
    }
    pub(super) fn record(checkpoint:&FiniteCheckpoint){CAPTURE.with(|capture|{let mut value=capture.borrow_mut();if let Some(c)=value.as_mut(){c.stage="finite-checkpoint";c.checkpoints.push(checkpoint.clone());let _=std::fs::write(&c.path,base(c,"running",None));}});}
    fn stage(name:&'static str,model_ms:Option<f64>,runtime_ms:Option<f64>){CAPTURE.with(|capture|{if let Some(c)=capture.borrow_mut().as_mut(){c.stage=name;if model_ms.is_some(){c.model_ms=model_ms;}if runtime_ms.is_some(){c.runtime_ms=runtime_ms;}let _=std::fs::write(&c.path,base(c,"running",None));}});}
    fn fixture_model_sha()->Result<String,String>{
        let path=PathBuf::from(std::env::var("ATOM_PREPARATION_FIXTURE").map_err(|_|"ATOM_PREPARATION_FIXTURE is required")?).with_extension("json");
        let metadata=std::fs::read_to_string(path).map_err(|e|e.to_string())?;
        let rest=metadata.split_once("\"modelSha256\"").ok_or("Fixture sidecar model SHA missing")?.1;
        let rest=rest.split_once(':').ok_or("Fixture sidecar SHA separator missing")?.1.trim_start();
        let sha=rest.strip_prefix('"').and_then(|s|s.split_once('"').map(|v|v.0)).ok_or("Malformed fixture model SHA")?;
        if sha.len()!=64||!sha.bytes().all(|c|c.is_ascii_hexdigit()){return Err("Invalid fixture model SHA".into());}Ok(sha.to_owned())
    }
    fn fixture_groups()->Vec<RequestedGroup>{
        // This named test fixture has the original TS32 state order. These
        // are readout partitions only, not a model or physical selection.
        [("ground-F1",0,3),("ground-F2",3,8),("D1-F1",8,11),("D1-F2",11,16),
         ("D2-F0",16,17),("D2-F1",17,20),("D2-F2",20,25),("D2-F3",25,32),("excited",8,32)]
            .into_iter().map(|(name,start,end)|RequestedGroup{name:name.into(),indices:(start..end).collect()}).collect()
    }
    #[test]
    #[ignore="One coordinated fresh full32 complete solve; no reference or old basis input"]
    fn preparation_finite_actual_full32_fresh_model_runner()->Result<(),String>{
        let path=PathBuf::from(std::env::var("ATOM_NATIVE_FINITE_REPORT").map_err(|_|"ATOM_NATIVE_FINITE_REPORT is required")?);
        let epsilon=std::env::var("ATOM_NATIVE_FINITE_EPSILON").ok().map(|s|s.parse::<f64>()).transpose().map_err(|e|e.to_string())?.unwrap_or(0.005);
        if !epsilon.is_finite()||epsilon<=0.||epsilon>0.02{return Err("Native finite runner tolerance must be in(0,.02]".into());}
        let wall=std::env::var("ATOM_NATIVE_FINITE_WALL_SECONDS").ok().map(|s|s.parse::<f64>()).transpose().map_err(|e|e.to_string())?.unwrap_or(120.);
        let capture=Capture{start:Instant::now(),path,epsilon,model_sha:fixture_model_sha()?,stage:"fresh-original-model",model_ms:None,runtime_ms:None,checkpoints:vec![]};
        std::fs::write(&capture.path,base(&capture,"running",None)).map_err(|e|e.to_string())?;CAPTURE.with(|c|*c.borrow_mut()=Some(capture));
        let times:Vec<_>=(0..200).map(|i|20.*i as f64/199.).collect();
        let result=catch_unwind(AssertUnwindSafe(||->Result<FiniteSolution,String>{
            let began=Instant::now();let(owner,initial,_)=super::super::model_fixture::fresh_owned_for_slice();stage("runtime-preparation",Some(began.elapsed().as_secs_f64()*1000.),None);
            let began=Instant::now();let core=RuntimeCore::from_owned(owner,InitialView{index:initial,real:None,imag:None},P0Policy::EigenPreferred)?;
            stage("complete-finite-solve",None,Some(began.elapsed().as_secs_f64()*1000.));
            solve(core,&times,fixture_groups(),FiniteOptions{population_tolerance:epsilon,wall_seconds:wall,..Default::default()})
        }));
        match result{
            Ok(Ok(solution))=>{
                let d=&solution.diagnostics;let last=&solution.candidate;let raw:Vec<_>=last.populations.iter().map(|z|z.re).collect();
                let mut trace=0_f64;let mut minimum=f64::INFINITY;let mut maximum=f64::NEG_INFINITY;
                for row in solution.populations.chunks_exact(32){trace=trace.max((row.iter().sum::<f64>()-1.).abs());for &v in row{minimum=minimum.min(v);maximum=maximum.max(v);}}
                let groups=fixture_groups();let grouped:Vec<Vec<f64>>=groups.iter().map(|g|solution.populations.chunks_exact(32).map(|row|g.indices.iter().map(|&i|row[i]).sum()).collect()).collect();
                CAPTURE.with(|c|{let mut state=c.borrow_mut();let record=state.as_mut().unwrap();record.stage="complete";record.checkpoints=d.checkpoints.clone();
                    let mut output=base(record,"accepted-empirical",None);output.pop();
                    output.push_str(&format!(",\"accepted\":{},\"empiricalOnly\":{},\"rigorousErrorBound\":{},\"solverElapsedMs\":{},\"estimatedError\":{},\"projectionCharge\":{},\"finalOrder\":{},\"finalBasis\":{},\"fastPairColumns\":{},\"returnedTraceError\":{},\"returnedMinimum\":{},\"returnedMaximum\":{},\"rawTraceError\":{},\"rawImaginaryError\":{},\"rawInitialProjectionError\":{},\"times\":{:?},\"populationShape\":[200,32],\"populations\":{:?},\"rawPopulations\":{:?},\"groupNames\":[{}],\"groupPopulations\":{:?}}}",
                        d.accepted,d.empirical_only,d.rigorous_error_bound,number(d.elapsed_ms),number(d.estimated_error),number(d.projection_charge),d.final_domain_order,d.final_basis,d.fast_pair_columns,
                        number(trace),number(minimum),number(maximum),number(last.diagnostics.population_trace_error),number(last.diagnostics.population_imaginary_error),number(last.diagnostics.original_initial_projection_error_hs),times,solution.populations,raw,
                        groups.iter().map(|g|quoted(&g.name)).collect::<Vec<_>>().join(","),grouped));
                    let write=std::fs::write(&record.path,output).map_err(|e|e.to_string());*state=None;write
                })?;Ok(())
            },
            other=>{let reason=match other{Ok(Err(e))=>e,Err(p)=>p.downcast_ref::<String>().cloned().or_else(||p.downcast_ref::<&str>().map(|s|s.to_string())).unwrap_or_else(||"Native finite runner panicked".into()),_=>unreachable!()};
                CAPTURE.with(|c|{let mut state=c.borrow_mut();if let Some(record)=state.as_mut(){record.stage="failed";let _=std::fs::write(&record.path,base(record,"unresolved",Some(&reason)));}*state=None;});Err(reason)}
        }
    }
}

#[cfg(test)]mod tests{
    use super::*;
    #[test]fn preparation_finite_support_is_parameter_graph_not_laser_count(){
        assert_eq!(support_domain(0,&[],3).unwrap(),vec![vec![]]);
        let s=[vec![1,0],vec![0,1],vec![1,1]];let a=support_domain(2,&s,1).unwrap();let b=support_domain(2,&s,2).unwrap();
        assert_eq!(a.len(),7);assert!(a.iter().all(|q|b.contains(q)));assert!(b.len()>a.len());
        for q in &b{assert!(b.contains(&q.iter().map(|x|-x).collect::<Vec<_>>()));}
    }
    #[test]fn preparation_finite_probability_change_is_charged_for_overlapping_groups(){
        let raw=[C{re:-0.001,im:0.},C{re:0.4,im:0.},C{re:0.601,im:0.}];
        let groups=[RequestedGroup{name:"first two".into(),indices:vec![0,1]},RequestedGroup{name:"all".into(),indices:vec![0,1,2]}];
        let(p,charge)=constrain_populations(&raw,3,&groups).unwrap();
        assert!(p.iter().all(|x|*x>=0.));assert!((p.iter().sum::<f64>()-1.).abs()<1e-14);assert!((charge-0.001).abs()<1e-14);
        assert_eq!(raw[0].re,-0.001);
    }
}
