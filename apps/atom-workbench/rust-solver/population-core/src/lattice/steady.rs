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
use super::slow::{FLOOR,Purpose,SlowModes,slow_modes};

pub const START_ORDER:usize=2;
/// Share of the tolerance a long-time steady solution may take (the
/// propagation to the switch time and the expansion need the rest), unless
/// its truncation part is already below TRUNCATION_FLOOR of the tolerance
/// (the fold charges dominate and no order reduces them).
pub const LONG_TIME_SHARE:f64=0.5;
const TRUNCATION_FLOOR:f64=0.1;
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
    /// The slow eigenpairs of the accepted order's operator.
    pub slow:SlowModes,
}

impl SteadySolution{
    /// The slowest relaxation rate (1/us): the slowest found at the zero
    /// shift and half the dressed frames' slowest precessing decay (out of
    /// the shift's reach; the half covers the recycling's change of its
    /// rate), so the slowest relaxation is never understated. None when the
    /// shift found no relaxation.
    pub fn slowest_rate(&self)->Option<f64>{
        let precessing=self.dressed.precessing_rate.map_or(f64::INFINITY,|r|0.5*r);
        self.slow.slowest_rate.map(|r|r.min(precessing))
    }
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
/// `purpose`: the slow modes it carries (`slow`). For the long-time
/// machinery the order is the one accepted without the shell correction:
/// the order-k dynamics and the steady state must be the same lattice's.
pub fn solve(prepared:&LatticePrepared,groups:&[Vec<usize>],purpose:Purpose)->Result<SteadySolution,String>{
    let clock=Instant::now();
    let dressed=StructuredDressed::new(prepared)?;let dressed_ms=clock.elapsed().as_secs_f64()*1e3;
    let frames=super::floquet::from_env(prepared)?;
    let n=prepared.generator.n;let d=n*n;let options=KrylovOptions::default();
    let mut attempts:Vec<Attempt>=Vec::new();
    let mut warm:Option<(Vec<Charge>,Vec<C>)>=None;
    for order in START_ORDER..=MAXIMUM_ORDER{
        let start=Instant::now();
        let modes=prepared.modes(order)?;
        let op=LatticeOperator::new(prepared,&modes)?;
        let ledger=RemainderLedger::new(&prepared.plan,op.nu_max(),op.dropped())?;
        if ledger.expansion_parameter>ledger::EXPANSION_LIMIT{
            return Err(format!("Static fold expansion parameter {:e} above {:e} at near order {order}",ledger.expansion_parameter,ledger::EXPANSION_LIMIT));}
        let(nu_max,canonical,operator_work,edges)=(op.nu_max(),op.canonical_count(),op.work,op.edge_counts);
        let mut sbatch=None;let mut fbatch=None;
        let(precondition_work,regularized)=match &frames{
            Some(f)=>{let b=f.batch_with(&op,0.,true,true)?;let w=(b.work,b.moved);fbatch=Some(b);w}
            None=>{let b=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true)?;let w=(b.work,b.regularized);sbatch=Some(b);w}};
        let pre:&mut dyn super::preconditioner::Preconditioner=match(&mut fbatch,&mut sbatch){(Some(f),_)=>f,(_,Some(s))=>s,_=>unreachable!()};
        let mut system=System::new(op,pre)?;
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
        let mut shell=Shell::new(prepared,&dressed,&system.op,order,true)?;
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
        // The long-time machinery keeps part of the tolerance for its own
        // terms: it raises the order past LONG_TIME_SHARE while the
        // truncation is what a higher order can still reduce.
        let accept=match purpose{Purpose::Report=>attempt.claimed<=POPULATION_TOLERANCE,
            Purpose::LongTime=>attempt.claimed<=LONG_TIME_SHARE*POPULATION_TOLERANCE
                ||(attempt.claimed<=POPULATION_TOLERANCE&&KAPPA*truncation_size<=TRUNCATION_FLOOR*POPULATION_TOLERANCE)};
        if accept{
            let residual=null_residual(&mut system,&field)?;let slow=slow_modes(&mut system,purpose)?;
            finish(&mut attempt,&system);let estimated_error=attempt.claimed;attempts.push(attempt);
            return Ok(SteadySolution{order,modes,field,populations,acceptance:Acceptance::Estimate,estimated_error,null_residual:residual,attempts,dressed:dressed.diagnostics.clone(),dressed_ms,slow});
        }
        // Shell Gauss-Seidel correction.
        let mut corrected:Vec<C>=field.iter().zip(&delta).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
        if let Some(shell)=&mut shell{
            // Only the report takes the shell correction: the long-time
            // machinery needs the steady state of the lattice it propagates.
            if purpose==Purpose::Report{
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
                        let slow=slow_modes(&mut system,purpose)?;
                        attempt.claimed=claimed;finish(&mut attempt,&system);attempts.push(attempt);
                        return Ok(SteadySolution{order,modes,field,populations,acceptance:Acceptance::Corrected,estimated_error:claimed,null_residual:residual,attempts,dressed:dressed.diagnostics.clone(),dressed_ms,slow});
                    }
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
    let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true)?;
    let mut system=System::new(op,&mut batch)?;let b=system.source();
    let primal=krylov::gmres(&mut system,&b,None,groups,options)?;
    let p=system.populations(&primal.x);let t:f64=p.iter().sum();Ok(p.iter().map(|v|v/t).collect())
}

/// Largest connected block of a static generator decomposed densely.
pub const STATIC_BLOCK:usize=256;
/// The mean of a static model (no drives: the lattice is the centre block
/// L0 alone), exact: L0 splits into the blocks its entries connect (the
/// populations with the coherences the decay feeds, and the coherences of
/// each Bohr frequency). On the imaginary axis every eigenvalue of a
/// Lindbladian is semisimple, so each one's spectral projector is
/// R (L^H R)^-1 L^H from its right and left eigenvectors (defective decaying
/// modes elsewhere do not matter). The t -> infinity mean is rho0's kernel
/// part; at a horizon T the imaginary-axis parts evolve with e^(lambda T)
/// once the decaying ones are below e^-TAIL_EXPONENT. Any number of
/// conserved quantities is served: the mean is then rho0's projection onto
/// them.
pub struct StaticMean{
    /// Row-major n x n density.
    pub density:Vec<C>,
    pub conserved:usize,pub slowest_rate:Option<f64>,
    /// e^(-slowest rate T) at a horizon (zero for the limit).
    pub tail:f64,
    /// |L0 x_0| / (|L0| |x_0|) of the kernel part, and its population error
    /// bound |L0 x_0| / (the block's smallest nonzero |lambda|).
    pub residual:f64,pub error:f64,
    pub blocks:usize,pub largest_block:usize,pub milliseconds:f64,
}
/// A horizon T with T times the slowest decay rate at least TAIL_EXPONENT is
/// relaxed: the deviation (at most 2 in trace norm) has decayed by e^-30 =
/// 1e-13, so an eigenbasis condition up to 1e8 leaves under 1e-2 of the
/// tolerance.
pub const TAIL_EXPONENT:f64=30.;
pub fn static_mean(prepared:&LatticePrepared,initial:&[C],horizon:Option<f64>)->Result<StaticMean,String>{
    let clock=Instant::now();let lapack=&prepared.lapack;let n=prepared.generator.n;let d=n*n;
    if !prepared.plan.explicit().is_empty()||!prepared.fold.targets.is_empty(){return Err("A static mean needs a model without drives".into());}
    let mut a=vec![ZERO;d*d];prepared.generator.zero_component().op.dense_into(n,ONE,&mut a);
    // Connected blocks: union-find over the nonzero entries.
    fn root(parent:&mut[usize],mut i:usize)->usize{while parent[i]!=i{parent[i]=parent[parent[i]];i=parent[i];}i}
    let mut parent:Vec<usize>=(0..d).collect();
    for r in 0..d{for c in 0..d{if magnitude(a[r*d+c])>0.{let(x,y)=(root(&mut parent,r),root(&mut parent,c));if x!=y{parent[x]=y;}}}}
    let mut blocks:BTreeMap<usize,Vec<usize>>=BTreeMap::new();
    for k in 0..d{let r=root(&mut parent,k);blocks.entry(r).or_default().push(k);}
    let largest_block=blocks.values().map(|b|b.len()).max().unwrap_or(0);
    if largest_block>STATIC_BLOCK{return Err(format!("A static block of {largest_block} elements is above {STATIC_BLOCK}"));}
    let column=|members:&[usize],adjoint:bool|{let m=members.len();let mut b=vec![ZERO;m*m];
        for(j,&c)in members.iter().enumerate(){for(i,&r)in members.iter().enumerate(){let z=a[r*d+c];if adjoint{b[j+m*i]=C{re:z.re,im:-z.im};}else{b[i+m*j]=z;}}}b};
    let cexp=|z:C,t:f64|{let m=(z.re*t).exp();C{re:m*(z.im*t).cos(),im:m*(z.im*t).sin()}};
    let(mut mean,mut kernel)=(vec![ZERO;d],vec![ZERO;d]);
    let(mut conserved,mut slowest,mut error)=(0,None::<f64>,0_f64);
    let mut spectra=Vec::with_capacity(blocks.len());
    for members in blocks.values(){
        let m=members.len();let(lambda,right)=lapack.complex_eigen_right(&column(members,false),m)?;
        for l in &lambda{if magnitude(*l)<FLOOR{conserved+=1;}else if -l.re>=FLOOR{slowest=Some(slowest.map_or(-l.re,|s:f64|s.min(-l.re)));}}
        spectra.push((lambda,right));
    }
    let tail=match(horizon,slowest){(Some(t),Some(g))=>{if g*t<TAIL_EXPONENT{
            return Err(format!("A static model still relaxing at the horizon (slowest rate {g:e}/us)"));}(-g*t).exp()}
        _=>0.};
    for(members,(lambda,right))in blocks.values().zip(&spectra){
        let m=members.len();let rho:Vec<C>=members.iter().map(|&k|initial[k]).collect();
        if rho.iter().all(|z|magnitude(*z)==0.){continue;}
        // Imaginary-axis eigenvalues kept: the kernel for the limit, all of
        // them at a horizon; clustered within FLOOR.
        let on_axis:Vec<usize>=(0..m).filter(|&k|lambda[k].re.abs()<FLOOR&&(horizon.is_some()||magnitude(lambda[k])<FLOOR)).collect();
        if on_axis.is_empty(){continue;}
        let(mu,left)=lapack.complex_eigen_right(&column(members,true),m)?;
        let gap=lambda.iter().map(|l|magnitude(*l)).filter(|&v|v>=FLOOR).fold(f64::INFINITY,f64::min);
        let mut done=vec![false;m];
        for &k in &on_axis{
            if done[k]{continue;}
            let centre=lambda[k];let near=|z:C|magnitude(C{re:z.re-centre.re,im:z.im-centre.im})<FLOOR;
            let r:Vec<usize>=on_axis.iter().copied().filter(|&j|near(lambda[j])).collect();for &j in &r{done[j]=true;}
            let l:Vec<usize>=(0..m).filter(|&j|near(C{re:mu[j].re,im:-mu[j].im})).collect();
            if l.len()!=r.len(){return Err(format!("Eigenvalue {centre:?} has {} right and {} left eigenvectors",r.len(),l.len()));}
            let q=r.len();
            // G = L^H R (q x q, column-major) and L^H rho.
            let dot=|x:usize,y:&dyn Fn(usize)->C|(0..m).fold(ZERO,|s,i|{let u=left[i+m*x];let v=y(i);C{re:s.re+u.re*v.re+u.im*v.im,im:s.im+u.re*v.im-u.im*v.re}});
            let mut g=vec![ZERO;q*q];for(a1,&x)in l.iter().enumerate(){for(b1,&y)in r.iter().enumerate(){g[a1+q*b1]=dot(x,&|i|right[i+m*y]);}}
            let h:Vec<C>=l.iter().map(|&x|dot(x,&|i|rho[i])).collect();
            let(lu,pivots)=lapack.complex_factor(&g,q)?;let c=lapack.complex_solve_matrix_factored(&lu,&pivots,&h,q)?;
            let phase=match horizon{Some(t)=>cexp(centre,t),None=>ONE};let zero=magnitude(centre)<FLOOR;
            for(b1,&y)in r.iter().enumerate(){let w=mul(c[b1],phase);
                for(i,&e)in members.iter().enumerate(){add(&mut mean[e],mul(right[i+m*y],w));if zero{add(&mut kernel[e],mul(right[i+m*y],c[b1]));}}}
        }
        // The kernel part's residual against the block's gap.
        let block=column(members,false);let defect=(0..m).map(|i|magnitude((0..m).fold(ZERO,|s,j|{let mut s=s;add(&mut s,mul(block[i+m*j],kernel[members[j]]));s}))).fold(0_f64,f64::max);
        if gap.is_finite(){error=error.max(defect/gap);}
    }
    // The Hermitian part (roundoff).
    for i in 0..n{for j in i..n{let(x,y)=(mean[i*n+j],mean[j*n+i]);let h=C{re:0.5*(x.re+y.re),im:0.5*(x.im-y.im)};mean[i*n+j]=h;mean[j*n+i]=C{re:h.re,im:-h.im};}}
    let norm=|x:&[C]|x.iter().fold(0_f64,|s,z|s.hypot(magnitude(*z)));
    let mut image=vec![ZERO;d];for r in 0..d{for c in 0..d{let z=a[r*d+c];if magnitude(z)>0.{add(&mut image[r],mul(z,kernel[c]));}}}
    let scale=a.iter().map(|z|magnitude(*z)).fold(0_f64,f64::max)*norm(&kernel);
    Ok(StaticMean{density:mean,conserved:conserved.max(1),slowest_rate:slowest,tail,residual:norm(&image)/scale.max(f64::MIN_POSITIVE),error,
        blocks:blocks.len(),largest_block,milliseconds:clock.elapsed().as_secs_f64()*1e3})
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared,explicit_matrix,dense_steady};
    fn tight()->KrylovOptions{KrylovOptions{algebraic:1e-13,residual_guard:1e-13,..KrylovOptions::default()}}
    fn primal(p:&LatticePrepared,order:usize)->(Vec<f64>,Vec<C>,Vec<Charge>){
        let dressed=StructuredDressed::new(p).unwrap();let modes=p.modes(order).unwrap();
        let op=LatticeOperator::new(p,&modes).unwrap();
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true).unwrap();
        let mut system=System::new(op,&mut batch).unwrap();let b=system.source();
        let k=krylov::gmres(&mut system,&b,None,&[vec![0,1],vec![2,3]],&tight()).unwrap();
        let t:f64=system.watch.iter().map(|&i|k.x[i].re).sum();let x:Vec<C>=k.x.iter().map(|v|scaled(*v,1./t)).collect();
        (system.populations(&x),x,modes)
    }
    #[test]
    fn lattice_slowest_relaxation_is_not_understated(){
        // The fixture's reported slowest rate against the dense spectrum of
        // the accepted order's lattice within half the smallest near beat
        // (the lattice repeats every mode at mu - i nu_s, and its edge copies
        // carry the truncation): no decaying mode there, precessing ones
        // included, relaxes more slowly than reported.
        let f=fixture();let p=prepared(&f);let groups=vec![vec![0,1],vec![2,3]];
        let s=solve(&p,&groups,Purpose::LongTime).unwrap();
        let op=LatticeOperator::new(&p,&s.modes).unwrap();let size=op.dimension();let a=op.dense();let real=2*size;
        let mut big=vec![0.;real*real];
        for r in 0..size{for c in 0..size{let v=a[r*size+c];
            big[r+real*c]=v.re;big[r+real*(c+size)]=-v.im;big[r+size+real*c]=v.im;big[r+size+real*(c+size)]=v.re;}}
        let(re,im,_)=p.lapack.eigen(&big,real).unwrap();
        let beat=p.plan.explicit().into_iter().map(|e|p.plan.steps[e].nu.abs()).fold(f64::INFINITY,f64::min);
        let slowest=re.iter().zip(&im).filter(|(_,b)|b.abs()<=0.5*beat).map(|(a,_)|-a).filter(|&g|g>1e-9).fold(f64::INFINITY,f64::min);
        let reported=s.slowest_rate().unwrap();
        assert!(reported<=slowest*(1.+1e-6),"reported {reported:e}, dense slowest decay {slowest:e}");
        assert!(reported<=s.slow.slowest_rate.unwrap()&&reported<=0.5*s.dressed.precessing_rate.unwrap());
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
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true).unwrap();
        let mut system=System::new(op,&mut batch).unwrap();let b=system.source();
        let k=krylov::gmres(&mut system,&b,None,&groups,&tight()).unwrap();
        let t:f64=system.watch.iter().map(|&i|k.x[i].re).sum();let x:Vec<C>=k.x.iter().map(|v|scaled(*v,1./t)).collect();
        let pk=system.populations(&x);
        let mut augmented=Augmented::new(k.arnoldi);
        let mut shell=Shell::new(&p,&dressed,&system.op,1,true).unwrap().unwrap();
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
        let s=solve(&p,&groups,Purpose::Report).unwrap();
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
