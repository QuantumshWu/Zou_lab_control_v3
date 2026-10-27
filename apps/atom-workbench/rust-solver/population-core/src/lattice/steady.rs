//! The steady pipeline: the t -> infinity time average of the populations.
//!
//! 1. Near order k = START_ORDER: the primal solve of M x = u (`krylov`).
//! 2. The error estimate in the primal's Krylov space (`estimate`): the
//!    one-step source of the order-(k+1) shell, which carries the primal's
//!    algebraic error too, and the third-order fold source, whose response
//!    turns the fold ledger into population charges.
//! 3. Accept when KAPPA (truncation + fold) <= POPULATION_TOLERANCE.
//!    Otherwise the shell Gauss-Seidel correction when its contraction
//!    q_2 = d_2 / d_1 < 1: GAUSS_SEIDEL_STEPS steps, remainder
//!    q / (1 - q) d_last plus the order-(k+2) shell extrapolated as
//!    Delta_k+1^2 / Delta_k (Delta_k = |p_k - p_k-1|, calibrated:
//!    conservative 40-1000 times on the research cases). Else the next
//!    order, warm-started from the corrected field.
//!
//! The populations reported are those of the accepted field: the order-k
//! primal, or its Gauss-Seidel correction.
use super::*;
use super::krylov::{self,KrylovOptions,KrylovTiming,System};
use super::estimate::{Augmented,Shell};
use super::operator::LatticeOperator;
use super::preconditioner::{StructuredDiagnostics,StructuredDressed};
use super::ledger::{self,KAPPA,POPULATION_TOLERANCE,PopulationCharges,RemainderLedger};

pub const START_ORDER:usize=2;
pub const MAXIMUM_ORDER:usize=4;
/// Shell Gauss-Seidel steps when the correction is used.
const GAUSS_SEIDEL_STEPS:usize=3;

/// The fold remainders as population charges.
#[derive(Clone,Debug)]
pub struct FoldCharge{
    /// "third-order" or "fold": the source whose response is measured.
    pub source:&'static str,
    /// |p(delta)| of that source and its generator-norm bound.
    pub response:f64,pub source_bound:f64,
    pub charges:PopulationCharges,
}
#[derive(Clone,Debug,Default)]
pub struct AttemptTiming{pub setup_ms:f64,pub primal_ms:f64,pub estimate_ms:f64,pub correction_ms:f64,
    /// The estimate's parts: shell operator and preconditioner, truncation
    /// source, fold source, augmentation, checked solution.
    pub estimate_parts:[f64;5],
    pub krylov:KrylovTiming}
/// One near order.
#[derive(Clone,Debug)]
pub struct Attempt{
    pub order:usize,pub modes:usize,pub canonical:usize,pub nu_max:f64,
    pub steps:usize,pub restarts:usize,pub stop:&'static str,pub residual:f64,
    pub populations:Vec<f64>,
    /// |delta p| of the truncation equation (one-step shell source with the
    /// primal's algebraic error).
    pub truncation:f64,
    pub fold:Option<FoldCharge>,
    /// Augmentation steps, shell applications, true residual of the
    /// truncation equation.
    pub augmentation:usize,pub shell_applications:usize,pub shell_modes:usize,pub truncation_residual:Option<f64>,
    /// Per error equation: label, augmentation steps to R2, |delta p|.
    pub equations:Vec<(&'static str,usize,f64)>,
    /// Gauss-Seidel population steps d_i and the contraction q_2.
    pub corrections:Vec<f64>,pub contraction:Option<f64>,
    /// KAPPA (estimates); the accepted attempt's is the reported error.
    pub claimed:f64,
    pub applications:usize,pub preconditions:usize,
    pub operator_work:usize,pub precondition_work:usize,pub regularized:usize,pub edges:[usize;2],
    pub ledger:RemainderLedger,
    pub timing:AttemptTiming,
}
#[derive(Clone,Copy,Debug,PartialEq)]
pub enum Acceptance{
    /// The primal at its order, error estimated.
    Estimate,
    /// The shell Gauss-Seidel correction of the primal.
    Corrected,
}
pub struct SteadySolution{
    pub order:usize,pub modes:Vec<Charge>,
    /// Normalized field (central trace 1) and its populations.
    pub field:Vec<C>,pub populations:Vec<f64>,
    pub acceptance:Acceptance,
    /// KAPPA (algebraic + truncation + fold): the reported population error.
    pub estimated_error:f64,
    /// |A x| / |x| of the returned field (rad/us); for a corrected field, of
    /// the field with its shell on the order-(k+1) lattice.
    pub null_residual:f64,
    pub attempts:Vec<Attempt>,
    pub dressed:StructuredDiagnostics,pub dressed_ms:f64,
}

fn size(p:&[f64],groups:&[Vec<usize>])->f64{ledger::metric(p,&vec![0.;p.len()],groups)}
fn shifted(p:&[f64],delta:&[f64])->Vec<f64>{p.iter().zip(delta).map(|(a,b)|a+b).collect()}

/// The field on another mode set (zero on modes it does not have).
fn embed(field:&[C],from:&[Charge],to:&[Charge],d:usize)->Vec<C>{
    let index:BTreeMap<&Charge,usize>=to.iter().enumerate().map(|(i,q)|(q,i)).collect();
    let mut out=vec![ZERO;to.len()*d];
    for(k,q)in from.iter().enumerate(){if let Some(&t)=index.get(q){out[t*d..(t+1)*d].copy_from_slice(&field[k*d..(k+1)*d]);}}
    out
}

/// The steady state with its estimated error.
pub fn solve(prepared:&LatticePrepared,groups:&[Vec<usize>])->Result<SteadySolution,String>{
    let clock=Instant::now();
    let dressed=StructuredDressed::new(prepared)?;let dressed_ms=clock.elapsed().as_secs_f64()*1e3;
    let n=prepared.generator.n;let d=n*n;let options=KrylovOptions::default();
    let mut attempts:Vec<Attempt>=Vec::new();
    let mut warm:Option<(Vec<Charge>,Vec<C>)>=None;
    for order in START_ORDER..=MAXIMUM_ORDER{
        let start=Instant::now();
        let modes=prepared.modes(order)?;
        let op=LatticeOperator::new(prepared,&modes)?;
        let ledger=RemainderLedger::new(&prepared.plan,op.nu_max(),prepared.fold.dropped())?;
        if ledger.expansion_parameter>ledger::EXPANSION_LIMIT{
            return Err(format!("Static fold expansion parameter {:e} above {:e} at near order {order}",ledger.expansion_parameter,ledger::EXPANSION_LIMIT));}
        let(nu_max,canonical,operator_work,edges)=(op.nu_max(),op.canonical_count(),op.work,op.edge_counts);
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre))?;
        let(precondition_work,regularized)=(batch.work,batch.regularized);
        let mut system=System::new(op,&mut batch)?;
        let mut timing=AttemptTiming{setup_ms:start.elapsed().as_secs_f64()*1e3,..Default::default()};
        // Primal.
        let start=Instant::now();
        let b=system.source();let x0=warm.take().map(|(from,field)|embed(&field,&from,&modes,d));
        let primal=krylov::gmres(&mut system,&b,x0,groups,&options)?;
        let trace:f64=system.watch.iter().map(|&k|primal.x[k].re).sum();
        if !(trace.is_finite()&&trace.abs()>0.){return Err("Lattice steady solution has no central trace".into());}
        let field:Vec<C>=primal.x.iter().map(|v|scaled(*v,1./trace)).collect();
        let populations=system.populations(&field);
        timing.primal_ms=start.elapsed().as_secs_f64()*1e3;
        // Estimate.
        let start=Instant::now();
        let mut augmented=Augmented::new(primal.arnoldi);
        let mut parts=[0.;5];let mut lap=Instant::now();
        let mut shell=Shell::new(prepared,&dressed,&system.op,order)?;
        let mut split=|k:usize,parts:&mut[f64;5]|{parts[k]=lap.elapsed().as_secs_f64()*1e3;lap=Instant::now();};
        split(0,&mut parts);
        let(source,mut shell_field)=match &mut shell{
            Some(s)=>s.source(&field)?,
            None=>{let mut w=vec![ZERO;system.length()];system.op.apply(&field,&mut w)?;system.applications+=1;
                (w.into_iter().map(|v|C{re:-v.re,im:-v.im}).collect(),Vec::new())}};
        let mut truncation=augmented.equation(&system,"truncation",source)?;
        split(1,&mut parts);
        let fold_source=if prepared.plan.far().is_empty(){None}else{
            let third=prepared.fold.third_order_source(&system.op,&field)?;
            Some(if ledger.third_order>0.&&third.iter().any(|z|nonzero(*z)){(third,"third-order",ledger.third_order)}
                else{(system.op.apply_fold(&field)?,"fold",ledger.fold)})};
        let mut fold=match fold_source{Some((s,label,bound))=>Some((augmented.equation(&system,label,s)?,bound)),None=>None};
        split(2,&mut parts);
        match &mut fold{
            Some((eq,_))=>augmented.solve(&mut system,&mut[&mut truncation,eq],groups)?,
            None=>augmented.solve(&mut system,&mut[&mut truncation],groups)?,
        }
        split(3,&mut parts);
        let delta=augmented.checked_solution(&mut system,&mut truncation,groups)?;
        split(4,&mut parts);
        let mut equations=vec![(truncation.label,truncation.stable_at.unwrap_or(truncation.steps()),size(truncation.estimate(),groups))];
        if let Some((eq,_))=&fold{equations.push((eq.label,eq.stable_at.unwrap_or(eq.steps()),size(eq.estimate(),groups)));}
        let estimate=truncation.estimate().to_vec();
        let fold=fold.map(|(eq,bound)|{let response=size(eq.estimate(),groups);let sensitivity=if bound>0.{response/bound}else{0.};
            FoldCharge{source:eq.label,response,source_bound:bound,charges:PopulationCharges::new(&ledger,sensitivity)}});
        let fold_total=fold.as_ref().map_or(0.,|f|f.charges.total());
        let truncation_size=size(&estimate,groups);
        timing.estimate_ms=start.elapsed().as_secs_f64()*1e3;timing.estimate_parts=parts;
        let mut attempt=Attempt{order,modes:modes.len(),canonical,nu_max,steps:primal.steps,restarts:primal.restarts,stop:primal.stop,residual:primal.residual,
            populations:populations.clone(),truncation:truncation_size,fold,augmentation:augmented.steps,
            shell_applications:shell.as_ref().map_or(0,|s|s.applications),shell_modes:shell.as_ref().map_or(0,|s|s.modes.len()),
            truncation_residual:truncation.true_residual,equations,corrections:vec![],contraction:None,claimed:KAPPA*(truncation_size+fold_total),
            applications:0,preconditions:0,operator_work,precondition_work,regularized,edges,ledger,timing};
        if attempt.claimed<=POPULATION_TOLERANCE{
            let residual=null_residual(&mut system,&field)?;finish(&mut attempt,&system);let estimated_error=attempt.claimed;attempts.push(attempt);
            return Ok(SteadySolution{order,modes,field,populations,acceptance:Acceptance::Estimate,estimated_error,null_residual:residual,attempts,dressed:dressed.diagnostics.clone(),dressed_ms});
        }
        // Shell Gauss-Seidel correction.
        let mut corrected:Vec<C>=field.iter().zip(&delta).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
        if let Some(shell)=&mut shell{
            let start=Instant::now();
            let mut previous=shifted(&populations,&estimate);let mut steps=vec![ledger::metric(&previous,&populations,groups)];
            let mut converging=true;
            for _ in 1..GAUSS_SEIDEL_STEPS{
                let(source,next)=shell.gauss_seidel(&field,&corrected,&shell_field)?;shell_field=next;
                let mut eq=augmented.equation(&system,"correction",source)?;augmented.solve(&mut system,&mut[&mut eq],groups)?;
                let delta=augmented.checked_solution(&mut system,&mut eq,groups)?;
                corrected=field.iter().zip(&delta).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
                let current=shifted(&populations,eq.estimate());
                steps.push(ledger::metric(&current,&previous,groups));previous=current;
                if steps.len()==2{let q=steps[1]/steps[0].max(f64::MIN_POSITIVE);attempt.contraction=Some(q);if !(q<1.){converging=false;break;}}
            }
            attempt.corrections=steps.clone();attempt.augmentation=augmented.steps;attempt.shell_applications=shell.applications;
            attempt.timing.correction_ms=start.elapsed().as_secs_f64()*1e3;
            if converging{
                let q=steps.windows(2).map(|w|w[1]/w[0].max(f64::MIN_POSITIVE)).fold(0.,f64::max);
                let remainder=q/(1.-q)*steps[steps.len()-1];
                let next_shell=ledger::metric(&previous,&populations,groups);
                // Delta_k from the previous order's primal (order k-1 solved here when k is the start).
                let lower=match attempts.last(){Some(a)=>a.populations.clone(),None=>lower_order(prepared,&dressed,order-1,groups,&options)?};
                let shell_k=ledger::metric(&populations,&lower,groups);
                let extrapolated=if shell_k>0.{next_shell*next_shell/shell_k}else{f64::INFINITY};
                let claimed=KAPPA*(remainder+extrapolated+fold_total);
                if claimed<=POPULATION_TOLERANCE{
                    let trace:f64=system.watch.iter().map(|&k|corrected[k].re).sum();
                    let field:Vec<C>=corrected.iter().map(|v|scaled(*v,1./trace)).collect();
                    let populations=system.populations(&field);
                    let shell_scaled:Vec<C>=shell_field.iter().map(|v|scaled(*v,1./trace)).collect();let residual=shell.residual(&field,&shell_scaled)?;
                    attempt.claimed=claimed;finish(&mut attempt,&system);attempts.push(attempt);
                    return Ok(SteadySolution{order,modes,field,populations,acceptance:Acceptance::Corrected,estimated_error:claimed,null_residual:residual,attempts,dressed:dressed.diagnostics.clone(),dressed_ms});
                }
            }
            // The next order starts from the corrected field and its shell.
            let next=prepared.modes(order+1)?;
            let mut start_field=embed(&corrected,&modes,&next,d);
            for(k,&s)in shell.modes.iter().enumerate(){start_field[s*d..(s+1)*d].copy_from_slice(&shell_field[k*d..(k+1)*d]);}
            let trace:f64=(0..n).map(|i|start_field[shell.op.centre*d+i*n+i].re).sum();
            warm=Some((next,start_field.iter().map(|v|scaled(*v,1./trace)).collect()));
        }else{
            finish(&mut attempt,&system);attempts.push(attempt);
            return Err(format!("Lattice steady state not accepted: estimated error {:e} above {POPULATION_TOLERANCE} with no near shell to add",KAPPA*(truncation_size+fold_total)));
        }
        finish(&mut attempt,&system);attempts.push(attempt);
    }
    let last=attempts.last().map_or(f64::NAN,|a|a.claimed);
    Err(format!("Lattice steady state not accepted up to near order {MAXIMUM_ORDER} (estimated error {last:e})"))
}

fn finish(attempt:&mut Attempt,system:&System<'_,'_>){
    attempt.applications=system.applications;attempt.preconditions=system.preconditions;attempt.timing.krylov=system.timing.clone();
}
/// |A x| / |x|.
fn null_residual(system:&mut System<'_,'_>,x:&[C])->Result<f64,String>{
    let mut w=vec![ZERO;x.len()];system.op.apply(x,&mut w)?;Ok(system.vectors.norm(&w)/system.vectors.norm(x).max(f64::MIN_POSITIVE))
}
/// Populations of the order-k primal alone (for the shell ratio).
fn lower_order(prepared:&LatticePrepared,dressed:&StructuredDressed,order:usize,groups:&[Vec<usize>],options:&KrylovOptions)->Result<Vec<f64>,String>{
    let modes=prepared.modes(order)?;let op=LatticeOperator::new(prepared,&modes)?;
    let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre))?;
    let mut system=System::new(op,&mut batch)?;let b=system.source();
    let primal=krylov::gmres(&mut system,&b,None,groups,options)?;
    let p=system.populations(&primal.x);let t:f64=p.iter().sum();Ok(p.iter().map(|v|v/t).collect())
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared,explicit_matrix,dense_steady};
    fn tight()->KrylovOptions{KrylovOptions{algebraic:1e-13,residual_guard:1e-13,..KrylovOptions::default()}}
    fn primal(p:&LatticePrepared,order:usize)->(Vec<f64>,Vec<C>,Vec<Charge>){
        let dressed=StructuredDressed::new(p).unwrap();let modes=p.modes(order).unwrap();
        let op=LatticeOperator::new(p,&modes).unwrap();
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre)).unwrap();
        let mut system=System::new(op,&mut batch).unwrap();let b=system.source();
        let k=krylov::gmres(&mut system,&b,None,&[vec![0,1],vec![2,3]],&tight()).unwrap();
        let t:f64=system.watch.iter().map(|&i|k.x[i].re).sum();let x:Vec<C>=k.x.iter().map(|v|scaled(*v,1./t)).collect();
        (system.populations(&x),x,modes)
    }
    #[test]
    fn lattice_primal_is_the_null_vector_of_the_dense_lattice(){
        // Against a dense LU of the same truncated, folded lattice with the
        // trace row: the algebra only.
        let f=fixture();let p=prepared(&f);let groups=vec![vec![0,1],vec![2,3]];
        let(populations,_,modes)=primal(&p,2);
        let op=LatticeOperator::new(&p,&modes).unwrap();
        let exact=dense_steady(&p,&op.dense(),op.centre,4);
        let error=ledger::metric(&exact,&populations,&groups);
        assert!(error<1e-10,"{error:e}");
        assert!((populations.iter().sum::<f64>()-1.).abs()<1e-12);
    }
    #[test]
    fn lattice_truncation_estimate_follows_the_next_order(){
        // The GCRO estimate of the order-1 -> 2 shell against the tight
        // order-2 solve (the nested k vs k+1 comparison, test-only).
        let f=fixture();let p=prepared(&f);let groups=vec![vec![0,1],vec![2,3]];
        let dressed=StructuredDressed::new(&p).unwrap();let modes=p.modes(1).unwrap();
        let op=LatticeOperator::new(&p,&modes).unwrap();
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre)).unwrap();
        let mut system=System::new(op,&mut batch).unwrap();let b=system.source();
        let k=krylov::gmres(&mut system,&b,None,&groups,&tight()).unwrap();
        let t:f64=system.watch.iter().map(|&i|k.x[i].re).sum();let x:Vec<C>=k.x.iter().map(|v|scaled(*v,1./t)).collect();
        let pk=system.populations(&x);
        let mut augmented=Augmented::new(k.arnoldi);
        let mut shell=Shell::new(&p,&dressed,&system.op,1).unwrap().unwrap();
        let(source,_)=shell.source(&x).unwrap();
        let mut eq=augmented.equation(&system,"truncation",source).unwrap();
        augmented.solve(&mut system,&mut[&mut eq],&groups).unwrap();
        let estimate=size(eq.estimate(),&groups);
        let(next,_,_)=primal(&p,2);let actual=ledger::metric(&next,&pk,&groups);
        assert!(augmented.steps>=2);
        assert!(estimate/actual>0.5&&estimate/actual<4.,"estimate {estimate:e}, actual {actual:e}");
    }
    #[test]
    fn lattice_fold_charge_covers_the_explicit_first_far_shell(){
        // The folded steady state against an explicit first far shell (near
        // couplings between far modes included): the difference is the fold
        // remainder, and the charged bound must cover it.
        let f=fixture();let p=prepared(&f);let n=4;let groups=vec![vec![0,1],vec![2,3]];
        let s=solve(&p,&groups).unwrap();
        let(folded,_,_)=primal(&p,s.order);
        let charges:Vec<Charge>=(-1..=1).flat_map(|b:i32|{let reach=s.order as i32+if b==0{0}else{1};(-reach..=reach).map(move|a|vec![a,b])}).collect();
        let(matrix,centre)=explicit_matrix(&p,&charges);
        let explicit=dense_steady(&p,&matrix,centre,n);
        let actual=ledger::metric(&explicit,&folded,&groups);
        let fold=s.attempts.last().unwrap().fold.clone().unwrap();
        assert_eq!(fold.source,"third-order");
        assert!(actual<=fold.charges.total(),"actual fold remainder {actual:e} above its charge {:e}",fold.charges.total());
    }
}
