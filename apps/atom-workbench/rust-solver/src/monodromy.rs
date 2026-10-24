//! Experimental full-density, single-frequency period-map propagation.
//!
//! No Fourier truncation and no averaging are used: t=k*T+s is evaluated as
//! Phi(s,0) P^k rho(0), where P is the fixed, linear full-density period action.
//! Period discretization and Krylov projection are numerical approximations;
//! both are compared at EVERY ACTUAL requested output time. This module does
//! not infer an infinite-time mixture from an unresolved near-unit mode.
use crate::{build_generator, expand_density, positivity, Coordinate, Lapack, Model, Options};
use crate::fixed_etd::{PeriodAction,Report};
use serde_json::json;
use std::{collections::{BTreeMap,VecDeque},time::Instant};

const KRYLOV_DIFFERENCE:f64=5e-6;

fn norm(x:&[f64])->f64{x.iter().map(|v|v*v).sum::<f64>().sqrt()}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn profiling()->bool{std::env::var_os("ATOM_MONODROMY_PROFILE").is_some()||std::env::var_os("ATOM_SOLVER_DEBUG").is_some()}

/// Both period actions retain the full original Hamiltonian and collapse
/// operators. This dispatch changes only the numerical time-step method.
enum Action {Etd(PeriodAction),Cptp(crate::cptp_period::Action)}
impl Action {
    fn new(lap:&Lapack,model:&Model,period:f64,steps:usize)->Result<Self,String>{
        if std::env::var("ATOM_PERIOD_CPTP").is_ok_and(|value|value=="1"){
            crate::cptp_period::Action::new(lap,model,period,steps).map(Self::Cptp)
        }else{PeriodAction::new(model,period,steps).map(Self::Etd)}
    }
    fn dimension(&self)->usize{match self{Self::Etd(action)=>action.dimension(),Self::Cptp(action)=>action.dimension()}}
    fn apply(&self,lap:&Lapack,x:&[f64])->Result<Vec<f64>,String>{match self{Self::Etd(action)=>action.apply(x),Self::Cptp(action)=>action.apply(lap,x)}}
    fn partial(&self,lap:&Lapack,x:&[f64],phase:f64)->Result<Vec<f64>,String>{match self{Self::Etd(action)=>action.partial(x,phase),Self::Cptp(action)=>action.partial(lap,x,phase)}}
    fn name(&self)->&'static str{match self{Self::Etd(_)=>"ETD-AB4",Self::Cptp(_)=>"CPTP-Strang"}}
    fn is_cptp(&self)->bool{matches!(self,Self::Cptp(_))}
}

enum GridIssue {Refine(String),BasisLimit(String),Fatal(String)}
impl From<String> for GridIssue {fn from(reason:String)->Self{Self::Fatal(reason)}}
impl From<&str> for GridIssue {fn from(reason:&str)->Self{Self::Fatal(reason.into())}}

fn period(model:&Model)->Result<f64,String>{
    let basis=crate::harmonics::model_frequency_basis(model,16)?;
    if !basis.exact||basis.rounded_relations!=0||basis.frequencies.len()!=1||basis.frequencies[0]<=0.{return Err("Monodromy requires one exact commensurate drive frequency; no carrier was rounded".into());}
    Ok(std::f64::consts::TAU/basis.frequencies[0])
}
pub fn supports(model:&Model)->bool{!model.drives.is_empty()&&period(model).is_ok()}

struct Coordinates {indices:Vec<usize>,weights:Vec<f64>,populations:Vec<usize>,dimension:usize}
impl Coordinates {
    fn new(model:&Model,initial:&[f64])->Result<Self,String>{
        let generator=build_generator(model)?;let dimension=initial.len();
        let mut edges:Vec<Vec<usize>>=generator.matrix.iter().map(|column|column.iter().filter_map(|&(i,v)|(v!=0.).then_some(i)).collect()).collect();
        for drive in &model.drives {for operator in [&drive.cos,&drive.sin]{
            let mut component=model.clone();component.h_real=operator.real.clone();component.h_imag=operator.imag.clone();
            component.collapse.clear();component.drives.clear();component.time_dependent=false;
            let matrix=build_generator(&component)?.matrix;
            for (column,entries) in edges.iter_mut().zip(matrix){column.extend(entries.into_iter().filter_map(|(i,v)|(v!=0.).then_some(i)));}
        }}
        let mut seen:Vec<_>=initial.iter().map(|&v|v!=0.).collect();
        let mut work:VecDeque<_>=seen.iter().enumerate().filter_map(|(i,&yes)|yes.then_some(i)).collect();
        while let Some(j)=work.pop_front(){for &i in &edges[j]{if !seen[i]{seen[i]=true;work.push_back(i);}}}
        let indices:Vec<_>=(0..dimension).filter(|&i|seen[i]).collect();
        let weights=indices.iter().map(|&i|if matches!(generator.coords[i],Coordinate::Population(_)){1.}else{2_f64.sqrt()}).collect();
        let populations=indices.iter().enumerate().filter_map(|(local,&global)|matches!(generator.coords[global],Coordinate::Population(_)).then_some(local)).collect();
        Ok(Self{indices,weights,populations,dimension})
    }
    fn lift(&self,x:&[f64])->Vec<f64>{let mut result=vec![0.;self.dimension];for ((&global,&weight),&value) in self.indices.iter().zip(&self.weights).zip(x){result[global]=value/weight;}result}
    fn restrict(&self,x:&[f64])->Vec<f64>{self.indices.iter().zip(&self.weights).map(|(&global,&weight)|x[global]*weight).collect()}
    fn apply(&self,lap:&Lapack,action:&Action,x:&[f64])->Result<Vec<f64>,String>{
        let full=action.apply(lap,&self.lift(x))?;
        let projected=self.restrict(&full);
        let restored=self.lift(&projected);
        let leakage=full.iter().zip(restored).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);
        if leakage>1e-10||norm(&projected)>10.*(action.dimension()as f64).sqrt()*norm(x).max(1.){
            return Err(format!("Period grid is unstable or violates exact graph support (leakage={leakage:e})"));
        }
        Ok(projected)
    }
}

fn split_time(time:f64,period:f64)->Result<(u64,f64),String>{
    let quotient=(time/period).floor();
    if !quotient.is_finite()||quotient<0.||quotient>4503599627370495.{return Err("Period count exceeds exact integer range".into());}
    let mut count=quotient as u64;
    let mut phase=(-(count as f64)).mul_add(period,time);
    if phase<0.&&count>0{count-=1;phase+=period;}
    if phase>=period{count+=1;phase-=period;}
    if phase<0.||phase>period{return Err("Invalid period remainder".into());}
    Ok((count,phase))
}

#[cfg(test)]
fn powers(lap:&Lapack,matrix:&[f64],n:usize,maximum:u64)->Vec<Vec<f64>>{
    let bits=(64-maximum.leading_zeros())as usize;
    if bits==0{return Vec::new();}
    let mut result=vec![matrix.to_vec()];
    for bit in 1..bits {result.push(lap.multiply(&result[bit-1],&result[bit-1],n,n,n));}
    result
}
#[cfg(test)]
fn power_action(lap:&Lapack,powers:&[Vec<f64>],initial:&[f64],mut exponent:u64)->Vec<f64>{
    let mut result=initial.to_vec();let mut bit=0;let n=initial.len();
    while exponent>0{if exponent&1==1{result=lap.multiply(&powers[bit],&result,n,1,n);}exponent>>=1;bit+=1;}
    result
}

fn groups(model:&Model)->Vec<Vec<usize>>{
    let mut groups=BTreeMap::<String,Vec<usize>>::new();
    for (i,name) in model.states.iter().enumerate(){if i<model.dimension{let key=name.split('-').take(2).collect::<Vec<_>>().join("-");groups.entry(key).or_default().push(i);}}
    groups.into_values().collect()
}
fn difference(a:&[Vec<f64>],b:&[Vec<f64>],n:usize,groups:&[Vec<usize>])->(f64,f64){
    let mut individual=0_f64;let mut grouped=0_f64;
    for (a,b) in a.iter().zip(b){for i in 0..n{individual=individual.max((a[i]-b[i]).abs());}for group in groups{grouped=grouped.max(group.iter().map(|&i|a[i]-b[i]).sum::<f64>().abs());}}
    (individual,grouped)
}

fn boundary_difference(a:&[Vec<f64>],b:&[Vec<f64>],dimension:usize,coordinates:&Coordinates,cptp:bool)->f64{
    a.iter().zip(b).map(|(a,b)|{
        let difference:Vec<_>=a.iter().zip(b).map(|(a,b)|a-b).collect();
        let hs=norm(&difference);
        if cptp{
            // Every state population and F aggregate is a projector observable.
            // CPTP evolution contracts Hermitian trace distance. HS weights
            // include ALL coherences; the extra trace term covers roundoff.
            let trace=coordinates.populations.iter().map(|&i|difference[i]).sum::<f64>().abs();
            0.5*((dimension as f64).sqrt()*hs+trace)
        }else{hs}
    }).fold(0.,f64::max)
}
fn apply_partials(lap:&Lapack,action:&Action,coordinates:&Coordinates,boundaries:&[Vec<f64>],parts:&[(u64,f64)])->Result<Vec<Vec<f64>>,String>{
    boundaries.iter().zip(parts).map(|(weighted,&(_,phase))|{
        let state=action.partial(lap,&coordinates.lift(weighted),phase)?;
        if state.iter().any(|value|!value.is_finite()){return Err("Non-finite projected period trajectory".into());}
        Ok(state)
    }).collect()
}

struct GridResult {states:Vec<Vec<f64>>,order:usize,applications:usize,krylov_difference:f64,period_residual:f64,trace_defect:f64,elapsed_ms:f64,
    period_apply_ms:f64,projection_ms:f64,partial_ms:f64,partial_applications:usize,projection_checks:usize,check_interval:usize,basis_limit:usize}
#[derive(Default)]
struct GridWork {period_apply_ms:f64,projection_ms:f64,partial_ms:f64,partial_applications:usize,projection_checks:usize}
impl GridWork {fn charge(&mut self,grid:&GridResult){
    self.period_apply_ms+=grid.period_apply_ms;self.projection_ms+=grid.projection_ms;self.partial_ms+=grid.partial_ms;
    self.partial_applications+=grid.partial_applications;self.projection_checks+=grid.projection_checks;
}}

fn grid(lap:&Lapack,model:&Model,action:&Action,coordinates:&Coordinates,initial:&[f64],times:&[f64],period:f64,groups:&[Vec<usize>])->Result<GridResult,GridIssue>{
    let started=Instant::now();let n=coordinates.indices.len();
    let debug=profiling();if debug{eprintln!("Monodromy grid begin method={} reachable={n} samples={} duration={:e}",action.name(),times.len(),times.last().copied().unwrap_or(0.));}
    let parts:Vec<_>=times.iter().map(|&time|split_time(time,period)).collect::<Result<_,_>>()?;
    let exponents:Vec<_>=parts.iter().map(|&(count,_)|count).collect();
    let maximum=parts.iter().map(|(count,_)|*count).max().unwrap_or(0);
    let partial_count=parts.iter().filter(|(_,phase)|*phase>0.).count();
    if maximum==0{
        let partial_started=Instant::now();
        let states=parts.iter().map(|&(_,phase)|action.partial(lap,initial,phase)).collect::<Result<_,_>>().map_err(GridIssue::Refine)?;
        let partial_ms=partial_started.elapsed().as_secs_f64()*1000.;
        if debug{eprintln!("Monodromy grid partial-only done ms={:.3}",started.elapsed().as_secs_f64()*1000.);}
        return Ok(GridResult{states,order:0,applications:0,krylov_difference:0.,period_residual:0.,trace_defect:0.,elapsed_ms:started.elapsed().as_secs_f64()*1000.,
            period_apply_ms:0.,projection_ms:0.,partial_ms,partial_applications:partial_count,projection_checks:0,check_interval:0,basis_limit:0});
    }
    let initial=coordinates.restrict(initial);
    let count=coordinates.populations.len();if count==0{return Err("Reachable density has no trace coordinate".into());}
    let mut anchor=vec![0.;n];for &i in &coordinates.populations{anchor[i]=1./(count as f64).sqrt();}
    let mut basis=vec![anchor];
    // A small per-time direction can accumulate over a long requested horizon.
    // Do not use a fixed dependence cutoff to declare such a chain closed.
    let independent=(0.1*KRYLOV_DIFFERENCE/times.last().copied().unwrap_or(period).max(period)).min(1e-12);
    let orthogonalize=|value:&mut Vec<f64>,basis:&[Vec<f64>]|{
        for _ in 0..2{for column in basis{let coefficient=dot(column,value);for (v,&q) in value.iter_mut().zip(column){*v-=coefficient*q;}}}
        let length=norm(value);if length>independent{for value in value{*value/=length;}true}else{false}
    };
    let mut initial_direction=initial.clone();if orthogonalize(&mut initial_direction,&basis){basis.push(initial_direction);}
    let seeds=basis.len();
    let limit=std::env::var("ATOM_MONODROMY_BASIS").ok().and_then(|s|s.parse::<usize>().ok()).unwrap_or(if action.is_cptp(){512}else{256}).clamp(seeds,n);
    let check_interval=std::env::var("ATOM_MONODROMY_CHECK_INTERVAL").ok().and_then(|s|s.parse::<usize>().ok()).filter(|&value|value>0).unwrap_or(if action.is_cptp(){64}else{16}).min(limit);
    if debug{eprintln!("Monodromy Krylov schedule basisLimit={limit} checkInterval={check_interval} stableComparisonsRequired=2");}
    let mut products:Vec<Vec<f64>>=Vec::new();let mut previous:Option<Vec<Vec<f64>>>=None;
    let mut stable=0;let mut trace_defect=0_f64;let mut last_reason=String::from("Krylov space not yet checked");
    let mut checked=0;let mut projection_checks=0;let mut latest_unstable=false;let mut next_check_interval=check_interval;
    let mut period_apply_ms=0.;let mut projection_ms=0.;let mut partial_ms=0.;let mut partial_applications=0;
    while products.len()<basis.len()&&products.len()<limit{
        let index=products.len();
        if debug&&index%16==0{eprintln!("Monodromy apply begin index={index} basis={} elapsedMs={:.3}",basis.len(),started.elapsed().as_secs_f64()*1000.);}
        let apply_started=Instant::now();
        let value=coordinates.apply(lap,action,&basis[index]).map_err(GridIssue::Refine)?;
        period_apply_ms+=apply_started.elapsed().as_secs_f64()*1000.;
        if debug&&index%16==0{eprintln!("Monodromy apply done index={index} elapsedMs={:.3}",started.elapsed().as_secs_f64()*1000.);}
        let basis_started=Instant::now();
        let expected_trace: f64=coordinates.populations.iter().map(|&i|basis[index][i]).sum();
        let actual_trace: f64=coordinates.populations.iter().map(|&i|value[i]).sum();
        trace_defect=trace_defect.max((actual_trace-expected_trace).abs());
        // Enforce the known exact trace row only for roundoff-sized defects;
        // larger integrator defects trigger independent period-grid refinement.
        if trace_defect>1e-10{return Err(GridIssue::Refine(format!("Period-map trace defect requires grid refinement: {trace_defect:e}")));}
        let mut candidate:Vec<_>=value.iter().zip(&basis[index]).map(|(a,b)|(a-b)/period).collect();
        let added=orthogonalize(&mut candidate,&basis);
        products.push(value);
        if added&&basis.len()<limit{basis.push(candidate);}
        let size=products.len();let closed=!added&&size==basis.len();
        projection_ms+=basis_started.elapsed().as_secs_f64()*1000.;
        // Skip expensive spectra/powers between numerical subspace checks.
        // Closed spaces and the requested basis limit are ALWAYS checked;
        // the same two successful comparisons and tolerances still apply.
        if size<seeds||(!closed&&size!=limit&&size<checked+next_check_interval){continue;}
        checked=size;projection_checks+=1;
        let projection_started=Instant::now();
        if debug{eprintln!("Monodromy check begin order={size} closed={closed} elapsedMs={:.3}",started.elapsed().as_secs_f64()*1000.);}
        let columns:Vec<_>=basis[..size].iter().flatten().copied().collect();
        let mut transpose=vec![0.;n*size];for (j,column) in basis[..size].iter().enumerate(){for (i,&v) in column.iter().enumerate(){transpose[j+i*size]=v;}}
        let product_columns:Vec<_>=products.iter().flatten().copied().collect();
        let mut projected=lap.multiply(&transpose,&product_columns,size,size,n);
        for j in 0..size{projected[j*size]=if j==0{1.}else{0.};}
        let (real,imag,_)=lap.eigen(&projected,size)?;
        if real.iter().chain(&imag).any(|value|!value.is_finite()){return Err(GridIssue::Refine("Non-finite projected period spectrum".into()));}
        let growth=real.iter().zip(imag).map(|(re,im)|re.hypot(im).ln().max(0.)*maximum as f64).fold(0.,f64::max);
        if debug{eprintln!("Monodromy spectrum done order={size} horizonLogGrowth={growth:e} elapsedMs={:.3}",started.elapsed().as_secs_f64()*1000.);}
        if growth>1e-6{
            latest_unstable=true;stable=0;previous=None;last_reason=format!("Unstable projected period map at order {size}, horizon log-growth {growth:e}");
            next_check_interval=check_interval;
            projection_ms+=projection_started.elapsed().as_secs_f64()*1000.;
            if closed{return Err(if action.is_cptp(){GridIssue::BasisLimit(format!("CPTP period projection remains unstable in the closed candidate: {last_reason}"))}else{GridIssue::Refine(last_reason)});}continue;
        }
        latest_unstable=false;
        let initial_coefficients:Vec<_>=basis[..size].iter().map(|q|dot(q,&initial)).collect();
        let projected_products=lap.multiply(&columns,&projected,n,size,size);
        let defect:Vec<_>=product_columns.iter().zip(projected_products).map(|(a,b)|a-b).collect();
        // Advance every requested exponent together: one streamed power
        // sequence, then two batched GEMMs for all boundaries and residuals.
        // This is the same discrete H^k action, with no output/coherence drop.
        let(coefficients,_)=crate::full_period::integer_actions(lap,projected,&initial_coefficients,&exponents)?;
        let weighted_all=lap.multiply(&columns,&coefficients,n,times.len(),size);
        let residual_all=lap.multiply(&defect,&coefficients,n,times.len(),size);
        let mut period_residual=0_f64;let mut boundaries=Vec::with_capacity(times.len());
        for sample in 0..times.len(){
            if sample==0{boundaries.push(initial.clone());continue;}
            let weighted=weighted_all[sample*n..(sample+1)*n].to_vec();
            period_residual=period_residual.max(norm(&residual_all[sample*n..(sample+1)*n]));
            if weighted.iter().any(|value|!value.is_finite()){return Err(GridIssue::Refine("Non-finite period-boundary trajectory".into()));}
            boundaries.push(weighted);
        }
        // The expensive phase segment does not participate in intermediate
        // Krylov checks. For CPTP it cannot increase the projector bound;
        // ETD uses the HS difference only as a trigger for final verification.
        let delta=previous.as_ref().map(|old|boundary_difference(old,&boundaries,model.dimension,coordinates,action.is_cptp())).unwrap_or(f64::INFINITY);
        if delta<KRYLOV_DIFFERENCE{stable+=1;}else{stable=0;}
        // Once a candidate is close, use shorter numerical-space increments
        // to obtain both unchanged confirmation checks before the basis cap.
        next_check_interval=if delta<1e-3{check_interval.min(16)}else{check_interval};
        let closure=defect.iter().map(|value|value.abs()).fold(0.,f64::max);
        projection_ms+=projection_started.elapsed().as_secs_f64()*1000.;
        if debug{eprintln!("Monodromy boundary check done order={size} delta={delta:e} cptpBound={} fullPeriodResidual={period_residual:e} closure={closure:e} stable={stable} nextInterval={next_check_interval} elapsedMs={:.3}",action.is_cptp(),started.elapsed().as_secs_f64()*1000.);}
        let closed_accepted=closed&&closure<1e-10;
        if (stable>=2||closed_accepted)&&period_residual<1e-8{
            if debug{eprintln!("Monodromy final partial maps begin order={size} samples={}",parts.len());}
            let partial_started=Instant::now();
            let states=apply_partials(lap,action,coordinates,&boundaries,&parts).map_err(GridIssue::Refine)?;
            partial_applications+=partial_count;
            let verified_delta=if !action.is_cptp()&&!closed_accepted{
                let old_states=apply_partials(lap,action,coordinates,previous.as_ref().ok_or("Missing previous period-boundary candidate")?,&parts).map_err(GridIssue::Refine)?;
                partial_applications+=partial_count;
                let(individual,grouped)=difference(&old_states,&states,model.dimension,groups);individual.max(grouped)
            }else if delta.is_finite(){delta}else{0.};
            partial_ms+=partial_started.elapsed().as_secs_f64()*1000.;
            if debug{eprintln!("Monodromy final partial maps done order={size} verifiedDelta={verified_delta:e} partialMs={partial_ms:.3} partialApplications={partial_applications}");}
            if !closed_accepted&&verified_delta>=KRYLOV_DIFFERENCE{
                stable=0;last_reason=format!("order={size}, final phase-segment difference={verified_delta:e}, full-period residual={period_residual:e}");
                previous=Some(boundaries);continue;
            }
            if debug{eprintln!("Monodromy grid converged order={size} elapsedMs={:.3}",started.elapsed().as_secs_f64()*1000.);}
            return Ok(GridResult{states,order:size,applications:products.len(),krylov_difference:verified_delta,period_residual,trace_defect,elapsed_ms:started.elapsed().as_secs_f64()*1000.,
                period_apply_ms,projection_ms,partial_ms,partial_applications,projection_checks,check_interval,basis_limit:limit});
        }
        last_reason=format!("order={size}, period-boundary difference/bound={delta:e}, full-period residual={period_residual:e}, closure={closure:e}");
        previous=Some(boundaries);
    }
    if latest_unstable{
        if action.is_cptp(){
            if debug{eprintln!("Monodromy CPTP projection unstable at basis limit; period-grid refinement cannot repair a missing Krylov direction: {last_reason}");}
            Err(GridIssue::BasisLimit(format!("CPTP period projection remains unstable at the Krylov basis limit: {last_reason}")))
        }else{
            if debug{eprintln!("Monodromy grid unstable at basis limit; refine period mesh: {last_reason}");}
            Err(GridIssue::Refine(last_reason))
        }
    }else{
        if debug{eprintln!("Monodromy stable-map basis limit: {last_reason}");}
        Err(GridIssue::BasisLimit(format!("Period-map Krylov convergence unresolved: {last_reason}")))
    }
}

pub fn solve(lap:&Lapack,model:&Model,options:&Options)->Result<Report,String>{
    let started=Instant::now();let period=period(model)?;
    if options.samples<2||options.samples>100000||!options.duration_us.is_finite()||options.duration_us<=0.{return Err("Invalid monodromy output grid".into());}
    let generator=build_generator(model)?;let coordinates=Coordinates::new(model,&generator.initial)?;
    let times:Vec<_>=(0..options.samples).map(|i|options.duration_us*i as f64/(options.samples-1)as f64).collect();
    let groups=groups(model);let mut subdivisions=std::env::var("ATOM_MONODROMY_STEPS").ok().and_then(|s|s.parse::<usize>().ok()).unwrap_or(24).clamp(4,4096);
    let mut attempts=Vec::new();let mut last_reason=String::new();
    let debug=profiling();let action_name=if std::env::var("ATOM_PERIOD_CPTP").is_ok_and(|value|value=="1"){"CPTP-Strang"}else{"ETD-AB4"};
    let nested=std::env::var("ATOM_PERIOD_NESTED_REFINEMENT").is_ok_and(|value|value=="1");
    let mut retained:Option<GridResult>=None;let mut work=GridWork::default();let mut total_charged_ms=0.;
    for _ in 0..5{
        let fine_steps=if nested{2*subdivisions+1}else{3*subdivisions+1};let coarse_reused=retained.is_some();
        let mut coarse_charged_ms=0.;let mut fine_charged_ms=0.;let mut failed_stage="coarse";
        let evaluate=|steps|->Result<GridResult,GridIssue>{
            if debug{eprintln!("Monodromy prepare begin method={action_name} substeps={steps}");}
            let action=Action::new(lap,model,period,steps).map_err(GridIssue::Refine)?;
            if debug{eprintln!("Monodromy prepare done substeps={steps}");}
            grid(lap,model,&action,&coordinates,&generator.initial,&times,period,&groups)
        };
        let pair=(||{
            let coarse=if let Some(previous)=retained.take(){
                if debug{eprintln!("Monodromy reuse previous fine as coarse substeps={subdivisions} chargedMs=0");}previous
            }else{
                let before=Instant::now();let result=evaluate(subdivisions);coarse_charged_ms=before.elapsed().as_secs_f64()*1000.;total_charged_ms+=coarse_charged_ms;
                if let Ok(value)=&result{work.charge(value);}result?
            };
            failed_stage="fine";let before=Instant::now();let result=evaluate(fine_steps);fine_charged_ms=before.elapsed().as_secs_f64()*1000.;total_charged_ms+=fine_charged_ms;
            if let Ok(value)=&result{work.charge(value);}
            Ok::<_,GridIssue>((coarse,result?))
        })();
        let (coarse,fine)=match pair{
            Ok(pair)=>pair,
            Err(GridIssue::BasisLimit(reason)|GridIssue::Fatal(reason))=>return Err(reason),
            Err(GridIssue::Refine(reason))=>{
                let next=if nested{if failed_stage=="coarse"{fine_steps}else{2*fine_steps+1}}else{2*subdivisions};
                if debug{eprintln!("Monodromy mesh rejected substeps={subdivisions} stage={failed_stage}; retry {next}: {reason}");}
                attempts.push(json!({"substeps":subdivisions,"independentSubsteps":fine_steps,"failureKind":"period-grid","failureStage":failed_stage,"coarseReusedWithinRefinement":coarse_reused,"coarseChargedMs":coarse_charged_ms,"fineChargedMs":fine_charged_ms,"chargedMs":coarse_charged_ms+fine_charged_ms,"rejected":reason}));
                last_reason=reason;subdivisions=next;continue;
            }
        };
        let (individual,grouped)=difference(&coarse.states,&fine.states,model.dimension,&groups);
        let mut trace_error=0_f64;let mut positivity_error=0_f64;
        for state in &fine.states{trace_error=trace_error.max((state[..model.dimension].iter().sum::<f64>()-1.).abs());positivity_error=positivity_error.max(positivity(lap,&expand_density(&generator.coords,state,model.dimension),model.dimension)?);}
        if debug{eprintln!("Monodromy independent comparison M={subdivisions} fine={fine_steps} reusedCoarse={coarse_reused} population={individual:e} Fgroup={grouped:e} trace={trace_error:e} positivity={positivity_error:e}");}
        attempts.push(json!({"coarseSubsteps":subdivisions,"independentSubsteps":fine_steps,"coarseOrder":coarse.order,"fineOrder":fine.order,"coarseMs":if coarse_reused{0.}else{coarse.elapsed_ms},"fineMs":fine.elapsed_ms,
            "coarseReusedWithinRefinement":coarse_reused,"coarseChargedMs":coarse_charged_ms,"fineChargedMs":fine_charged_ms,"chargedMs":coarse_charged_ms+fine_charged_ms,
            "coarseProjectionChecks":if coarse_reused{0}else{coarse.projection_checks},"fineProjectionChecks":fine.projection_checks,"checkInterval":fine.check_interval,"basisLimit":fine.basis_limit,
            "coarsePeriodApplyMs":if coarse_reused{0.}else{coarse.period_apply_ms},"coarseProjectionMs":if coarse_reused{0.}else{coarse.projection_ms},"coarsePartialMs":if coarse_reused{0.}else{coarse.partial_ms},"coarsePartialApplications":if coarse_reused{0}else{coarse.partial_applications},
            "finePeriodApplyMs":fine.period_apply_ms,"fineProjectionMs":fine.projection_ms,"finePartialMs":fine.partial_ms,"finePartialApplications":fine.partial_applications,
            "populationDifference":individual,"groupDifference":grouped,"traceError":trace_error,"positivityError":positivity_error}));
        if individual.max(grouped)<=crate::full_period::population_tolerance()&&trace_error<=1e-7&&positivity_error<=1e-5{
            let diagnostics=json!({"engine":"rust-monodromy-experiment","method":"Full-density period map / matrix-free Krylov / discrete matrix powers",
                "elapsedMs":started.elapsed().as_secs_f64()*1000.,"periodAction":action_name,"periodUs":period,"maximumPeriods":split_time(options.duration_us,period)?.0,
                "physicalHilbertDimension":model.dimension,"fullLiouvilleDimension":model.dimension*model.dimension,"exactReachableDimension":coordinates.indices.len(),
                "basisOrder":fine.order,"periodApplications":fine.applications,"substeps":fine_steps,"populationRefinementTolerance":crate::full_period::population_tolerance(),"requestedPopulationAccuracy":1e-3,"populationRefinementError":individual,"groupPopulationRefinementError":grouped,
                "projectionChecks":work.projection_checks,"fineProjectionChecks":fine.projection_checks,"checkInterval":fine.check_interval,"basisLimit":fine.basis_limit,
                "checkStrategy":"Configured spacing until full-density boundary difference is below1e-3; then at most16 new directions between the same two required confirmation checks",
                "krylovActualTimeDifference":fine.krylov_difference,"fullPeriodResidual":fine.period_residual,"periodTraceDefect":fine.trace_defect,
                "krylovDifferenceKind":if action_name=="CPTP-Strang"{"projector bound from full-HS period-boundary difference"}else{"verified final phase population/F-group difference or invariant closure"},
                "periodApplyMs":work.period_apply_ms,"projectionMs":work.projection_ms,"partialMs":work.partial_ms,"partialApplications":work.partial_applications,"chargedMs":total_charged_ms,"nestedRefinement":nested,
                "traceError":trace_error,"positivityError":positivity_error,"attempts":attempts,
                "refinementGrids":if nested{"Consecutive coprime M,2M+1 grids; each completed grid evaluated once within this request"}else{"Independent M,3M+1 grids"},
                "errorScope":"Observed independent period grids and nested Krylov spaces at every actual requested time; full period-map residual and trace/PSD checks. No certified continuous-time bound or inferred infinite-time mixture."});
            return Ok(Report{times,states:fine.states,diagnostics,trace_error,positivity_error});
        }
        last_reason=format!("population={individual:e}, Fgroup={grouped:e}, trace={trace_error:e}, positivity={positivity_error:e}");
        if nested{retained=Some(fine);subdivisions=fine_steps;}else{subdivisions*=2;}
    }
    Err(format!("Full-duration monodromy refinement did not converge: {last_reason}"))
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn discrete_powers_retain_ten_million_period_slow_decay(){
        let lap=Lapack::load().unwrap();let p=vec![1.,0.,2e-8,1.-2e-8];let initial=[0.,1.];let periods=10_000_000;
        let actual=power_action(&lap,&powers(&lap,&p,2,periods),&initial,periods);
        let survival=((1_f64-2e-8).ln()*periods as f64).exp();
        assert!((actual[1]-survival).abs()<1e-9);assert!((actual[0]+actual[1]-1.).abs()<1e-8);
    }
    #[test]
    fn compensated_period_split_reconstructs_requested_long_times(){
        let period=0.01;
        for time in [0.,0.0137,20.,100000.,100000.0037]{let(n,phase)=split_time(time,period).unwrap();assert!(phase>=0.&&phase<period);assert!(((n as f64).mul_add(period,phase)-time).abs()<2e-11);}
    }
}
