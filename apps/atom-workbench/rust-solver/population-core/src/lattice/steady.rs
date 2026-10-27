//! Steady state on the folded lattice at one near order. The time-averaged
//! populations are the diagonal of the central block of the null vector of
//! A; the central trace row M = A + u v^T (u = I/n on the centre, v its
//! trace) makes it the solution of M x = u (A x = 0, v^T x = 1 when it is
//! unique).
//!
//! Right-preconditioned restarted GMRES with the dressed block
//! preconditioner (fixed, so only the populations of P v_j are kept per step
//! and the iterate is formed once per cycle). It stops when the populations
//! and the requested groups of the Krylov iterate have stabilised (their
//! change stays below the algebraic share of the tolerance for STABLE_STEPS
//! consecutive steps) and the true relative residual, recomputed from the
//! iterate, is below a guard. Both are needed: the population change alone
//! can plateau for a few steps in slow directions while still 0.1 away from
//! the answer, and the recursive GMRES residual can fall eight orders below
//! the true one (the operator spans 1e-6 to 5e7 rad/us).
//!
//! Fold charge: the ledger's generator-norm bounds become population charges
//! through one sensitivity solve, M delta = s, on the largest remainder's own
//! source s (the third-order term when the lattice has near couplings, else
//! the fold itself): sensitivity = |p(delta)| / bound(s).
use super::*;
use super::operator::LatticeOperator;
use super::preconditioner::{DenseDressed,DressedBatch};
use super::ledger::{self,PopulationCharges,RemainderLedger};
use crate::parallel::Execution;

const STABLE_STEPS:usize=3;
/// Blocks per reduction chunk: fixed, so sums do not depend on threads.
const VECTOR_CHUNK:usize=16;
/// True relative residual at which step counts are compared with the
/// reference runs (rtol 1e-10).
const TIGHT_RESIDUAL:f64=1e-10;
/// Sensitivity solve: populations stable to this fraction of the response,
/// true relative residual below SENSITIVITY_GUARD.
const SENSITIVITY_SHARE:f64=0.05;
const SENSITIVITY_GUARD:f64=1e-6;

#[derive(Clone,Debug)]
pub struct SteadyOptions{
    /// Population-stability tolerance of the algebraic stop.
    pub algebraic:f64,
    /// Relative true-residual guard of the algebraic stop. Measured: a true
    /// residual of 3e-7 left 1.8e-5 in the populations (two lasers +
    /// tweezer, a 1e-6 /us Raman manifold); at 1e-8 every case is within
    /// 1e-7 of its reference.
    pub residual_guard:f64,
    pub restart:usize,pub maximum_steps:usize,
    /// Keep every step's residual and populations.
    pub history:bool,
    /// Krylov basis memory budget (bytes); the restart length shrinks to fit.
    pub basis_bytes:usize,
}
impl Default for SteadyOptions{
    fn default()->Self{
        Self{algebraic:ledger::POPULATION_TOLERANCE*ledger::ALGEBRAIC_SHARE,residual_guard:1e-8,restart:80,maximum_steps:800,history:false,basis_bytes:1<<30}
    }
}
#[derive(Clone,Debug)]
pub struct StepRecord{pub residual:f64,pub change:f64,pub populations:Vec<f64>}
#[derive(Clone,Debug,Default)]
pub struct SolveTiming{pub setup_ms:f64,pub apply_ms:f64,pub precondition_ms:f64,pub orthogonalize_ms:f64,pub total_ms:f64}
/// The populations' response to the largest remainder and the charges it
/// implies.
#[derive(Clone,Debug)]
pub struct FoldCharge{
    /// "third-order" or "fold".
    pub source:&'static str,
    /// |p(delta)| and the generator-norm bound of its source.
    pub response:f64,pub source_bound:f64,
    pub charges:PopulationCharges,
    pub steps:usize,pub milliseconds:f64,
}
pub struct OrderSolve{
    pub order:usize,pub modes:Vec<Charge>,pub nu_max:f64,
    /// Normalized populations and field (central trace 1).
    pub populations:Vec<f64>,pub field:Vec<C>,
    pub steps:usize,pub restarts:usize,pub restart:usize,pub stop:&'static str,
    /// |u - M x| / |u|, |A x| / |x| and the central trace before normalization.
    pub residual:f64,pub null_residual:f64,pub trace:f64,
    pub applications:usize,pub preconditions:usize,
    pub operator_work:usize,pub precondition_work:usize,pub regularized:usize,pub edges:[usize;2],
    pub timing:SolveTiming,
    pub ledger:RemainderLedger,
    pub charge:Option<FoldCharge>,
    pub history:Vec<StepRecord>,
    /// Step at which the GMRES residual first fell below TIGHT_RESIDUAL.
    pub steps_to_tight_residual:Option<usize>,
}

/// Deterministic parallel vector kernels (fixed reduction chunks).
struct Vectors{execution:Execution,d:usize}
impl Vectors{
    fn ranges(&self,length:usize)->Vec<(usize,usize)>{let w=VECTOR_CHUNK*self.d;(0..length.div_ceil(w)).map(|c|(c*w,((c+1)*w).min(length))).collect()}
    fn dots(&self,basis:&[Vec<C>],w:&[C])->Vec<C>{
        let parts=self.execution.map(self.ranges(w.len()),|(a,b)|basis.iter().map(|v|{let mut s=ZERO;for k in a..b{mul_add(&mut s,conj(v[k]),w[k]);}s}).collect::<Vec<C>>());
        let mut out=vec![ZERO;basis.len()];for p in parts{for(o,v)in out.iter_mut().zip(p){add(o,v);}}out
    }
    fn norm(&self,w:&[C])->f64{
        self.execution.map(self.ranges(w.len()),|(a,b)|w[a..b].iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>()).into_iter().sum::<f64>().sqrt()
    }
    /// w -= sum_i h_i basis_i.
    fn subtract(&self,basis:&[Vec<C>],h:&[C],w:&mut[C])->Result<(),String>{
        let d=self.d;let work=basis.len()*w.len();
        self.execution.for_mode_chunks(w,d,work,crate::parallel::MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
            for(v,&c)in basis.iter().zip(h){let c=C{re:-c.re,im:-c.im};for(t,&x)in chunk.iter_mut().zip(&v[first*d..]){mul_add(t,c,x);}}
            Ok(())
        })
    }
}

/// y = M x = A x + u (v^T x) on the centre.
fn apply_m(op:&mut LatticeOperator<'_>,watch:&[usize],x:&[C],y:&mut[C],clock:&mut f64)->Result<(),String>{
    let start=Instant::now();op.apply(x,y)?;
    let n=watch.len();let t=watch.iter().fold(ZERO,|s,&k|C{re:s.re+x[k].re,im:s.im+x[k].im});
    for &k in watch{add(&mut y[k],scaled(t,1./n as f64));}
    *clock+=start.elapsed().as_secs_f64()*1e3;Ok(())
}
fn precondition(pre:&mut DressedBatch<'_>,x:&[C],y:&mut[C],clock:&mut f64)->Result<(),String>{
    let start=Instant::now();pre.apply(x,y)?;*clock+=start.elapsed().as_secs_f64()*1e3;Ok(())
}

/// When the Krylov iterate is accepted.
struct Stop{
    /// Populations normalized by the central trace (primal) or raw (a
    /// response to a traceless source).
    normalize:bool,
    /// Stability tolerance: absolute, or relative to the populations' size.
    tolerance:f64,relative:bool,
    residual_guard:f64,
}
struct Krylov{x:Vec<C>,steps:usize,restarts:usize,restart:usize,stop:&'static str,history:Vec<StepRecord>,tight:Option<usize>,
    timing:SolveTiming,applications:usize,preconditions:usize}

fn populations(diagonal:&[C],normalize:bool)->Vec<f64>{
    let t:f64=if normalize{diagonal.iter().map(|z|z.re).sum()}else{1.};diagonal.iter().map(|z|z.re/t).collect()
}

fn gmres(op:&mut LatticeOperator<'_>,pre:&mut DressedBatch<'_>,b:&[C],stop:&Stop,groups:&[Vec<usize>],options:&SteadyOptions,vectors:&Vectors)->Result<Krylov,String>{
    let n=op.n;let d=n*n;let length=op.dimension();let c0=op.centre*d;
    let watch:Vec<usize>=(0..n).map(|i|c0+i*n+i).collect();
    let bnorm=vectors.norm(b);
    let fit=(options.basis_bytes/(length*std::mem::size_of::<C>())).saturating_sub(2);
    let restart=options.restart.min(fit).max(8);
    let mut timing=SolveTiming::default();let(mut applications,mut preconditions)=(0,0);
    let mut x=vec![ZERO;length];let(mut z,mut w)=(vec![ZERO;length],vec![ZERO;length]);
    let mut history=Vec::new();let mut previous:Option<Vec<f64>>=None;let mut changes:Vec<f64>=Vec::new();
    let(mut steps,mut restarts)=(0,0);let mut stopped:Option<&'static str>=None;let mut tight=None;
    if bnorm==0.{return Ok(Krylov{x,steps,restarts,restart,stop:"zero-source",history,tight,timing,applications,preconditions});}
    // r = b - M x, recomputed from x after every cycle: the recursive GMRES
    // residual drifts from the true one (M spans 1e-6 to 5e7 rad/us), so a
    // stop is accepted only on the true residual.
    let mut r=b.to_vec();let mut last_true=1_f64;
    loop{
        let beta=vectors.norm(&r);
        if beta<=1e-15*bnorm{stopped=Some("exact");break;}
        let mut basis=vec![r.iter().map(|v|scaled(*v,1./beta)).collect::<Vec<C>>()];let mut rotated:Vec<Vec<C>>=Vec::new();
        let(mut cs,mut sn)=(Vec::<C>::new(),Vec::<C>::new());let mut g=vec![C{re:beta,im:0.}];
        let mut zdiag:Vec<Vec<C>>=Vec::new();let xdiag:Vec<C>=watch.iter().map(|&k|x[k]).collect();
        let mut y=Vec::new();
        for j in 0..restart{
            precondition(pre,&basis[j],&mut z,&mut timing.precondition_ms)?;preconditions+=1;
            zdiag.push(watch.iter().map(|&k|z[k]).collect());
            apply_m(op,&watch,&z,&mut w,&mut timing.apply_ms)?;applications+=1;
            let start=Instant::now();
            let h1=vectors.dots(&basis,&w);vectors.subtract(&basis,&h1,&mut w)?;
            let h2=vectors.dots(&basis,&w);vectors.subtract(&basis,&h2,&mut w)?;
            let mut column:Vec<C>=h1.iter().zip(&h2).map(|(a,b)|C{re:a.re+b.re,im:a.im+b.im}).collect();
            let hn=vectors.norm(&w);column.push(C{re:hn,im:0.});
            timing.orthogonalize_ms+=start.elapsed().as_secs_f64()*1e3;
            for i in 0..j{let(a,c)=(column[i],column[i+1]);
                column[i]=ZERO;mul_add(&mut column[i],conj(cs[i]),a);mul_add(&mut column[i],conj(sn[i]),c);
                let mut lower=ZERO;mul_add(&mut lower,C{re:-sn[i].re,im:-sn[i].im},a);mul_add(&mut lower,cs[i],c);column[i+1]=lower;}
            let size=magnitude(column[j]).hypot(magnitude(column[j+1]));
            let(c,s)=if size>0.{(scaled(column[j],1./size),scaled(column[j+1],1./size))}else{(ONE,ZERO)};
            cs.push(c);sn.push(s);column[j]=C{re:size,im:0.};column.truncate(j+1);
            let gj=g[j];g.push(mul(C{re:-s.re,im:-s.im},gj));g[j]=mul(conj(c),gj);
            rotated.push(column);steps+=1;
            let residual=magnitude(g[j+1])/bnorm;
            if tight.is_none()&&residual<=TIGHT_RESIDUAL{tight=Some(steps);}
            // y = R^-1 g, R upper triangular by columns.
            y=vec![ZERO;j+1];
            for i in(0..=j).rev(){let mut s=g[i];for l in i+1..=j{let v=mul(rotated[l][i],y[l]);s.re-=v.re;s.im-=v.im;}
                let rii=rotated[i][i];let q=rii.re*rii.re+rii.im*rii.im;y[i]=C{re:(s.re*rii.re+s.im*rii.im)/q,im:(s.im*rii.re-s.re*rii.im)/q};}
            let mut diagonal=xdiag.clone();for(k,zk)in zdiag.iter().enumerate(){for(t,&v)in diagonal.iter_mut().zip(zk){mul_add(t,y[k],v);}}
            let current=populations(&diagonal,stop.normalize);
            let change=previous.as_ref().map_or(f64::INFINITY,|p|ledger::metric(p,&current,groups));changes.push(change);
            let scale=if stop.relative{current.iter().map(|v|v.abs()).fold(0.,f64::max)}else{1.};
            if options.history{history.push(StepRecord{residual,change,populations:current.clone()});}
            previous=Some(current);
            let stable=changes.len()>=STABLE_STEPS&&changes[changes.len()-STABLE_STEPS..].iter().all(|&c|c<stop.tolerance*scale);
            if stable&&residual<=stop.residual_guard{stopped=Some("populations-stable");}
            else if hn==0.{stopped=Some("exact");}
            else if residual<=1e-15{stopped=Some("residual-at-roundoff");}
            if stopped.is_some()||steps>=options.maximum_steps{break;}
            if j+1<restart{let next:Vec<C>=w.iter().map(|v|scaled(*v,1./hn)).collect();basis.push(next);}
        }
        // x += P (V y).
        let mut combination=vec![ZERO;length];
        for(v,&c)in basis.iter().zip(&y){for(t,&a)in combination.iter_mut().zip(v){mul_add(t,c,a);}}
        precondition(pre,&combination,&mut z,&mut timing.precondition_ms)?;preconditions+=1;
        for(a,c)in x.iter_mut().zip(&z){add(a,*c);}
        apply_m(op,&watch,&x,&mut w,&mut timing.apply_ms)?;applications+=1;
        for((a,&bb),c)in r.iter_mut().zip(b).zip(&w){*a=C{re:bb.re-c.re,im:bb.im-c.im};}
        let true_residual=vectors.norm(&r)/bnorm;
        if stopped.is_some(){
            if true_residual<=stop.residual_guard{break;}
            // Restarting no longer reduces the true residual: its floor.
            if true_residual>0.5*last_true{stopped=Some("true-residual-floor");break;}
            stopped=None;
        }
        if steps>=options.maximum_steps{return Err(format!("Lattice steady GMRES did not stabilise in {steps} steps (true residual {true_residual:e})"));}
        last_true=true_residual;restarts+=1;
    }
    Ok(Krylov{x,steps,restarts,restart,stop:stopped.unwrap_or("maximum"),history,tight,timing,applications,preconditions})
}

/// The steady state on the near ball of `order`, with its fold charge when
/// `charge` is set.
pub fn solve_order(prepared:&LatticePrepared,dressed:&DenseDressed,order:usize,groups:&[Vec<usize>],options:&SteadyOptions,charge:bool)->Result<OrderSolve,String>{
    let clock=Instant::now();
    let n=prepared.generator.n;let d=n*n;
    let vectors=Vectors{execution:Execution::new()?,d};
    let modes=prepared.modes(order)?;
    let mut op=LatticeOperator::new(prepared,&modes)?;
    let ledger=RemainderLedger::new(&prepared.plan,op.nu_max(),prepared.fold.dropped())?;
    if ledger.expansion_parameter>ledger::EXPANSION_LIMIT{
        return Err(format!("Static fold expansion parameter {:e} above {:e} at order {order}",ledger.expansion_parameter,ledger::EXPANSION_LIMIT));}
    let mut batch=dressed.batch(&op.nu,op.centre)?;
    let setup_ms=clock.elapsed().as_secs_f64()*1e3;
    let length=op.dimension();let c0=op.centre*d;let watch:Vec<usize>=(0..n).map(|i|c0+i*n+i).collect();
    let mut b=vec![ZERO;length];for &k in &watch{b[k]=C{re:1./n as f64,im:0.};}
    let primal=Stop{normalize:true,tolerance:options.algebraic,relative:false,residual_guard:options.residual_guard};
    let krylov=gmres(&mut op,&mut batch,&b,&primal,groups,options,&vectors)?;
    let mut timing=krylov.timing;timing.setup_ms=setup_ms;
    let(mut applications,mut preconditions)=(krylov.applications,krylov.preconditions);
    let x=krylov.x;let mut w=vec![ZERO;length];
    apply_m(&mut op,&watch,&x,&mut w,&mut timing.apply_ms)?;applications+=1;
    let residual=b.iter().zip(&w).fold(0_f64,|s,(a,c)|s.hypot(magnitude(C{re:a.re-c.re,im:a.im-c.im})))/vectors.norm(&b);
    let start=Instant::now();op.apply(&x,&mut w)?;timing.apply_ms+=start.elapsed().as_secs_f64()*1e3;applications+=1;
    let null_residual=vectors.norm(&w)/vectors.norm(&x).max(f64::MIN_POSITIVE);
    let trace:f64=watch.iter().map(|&k|x[k].re).sum();
    if !(trace.is_finite()&&trace.abs()>0.){return Err("Lattice steady solution has no central trace".into());}
    let field:Vec<C>=x.iter().map(|v|scaled(*v,1./trace)).collect();
    let populations:Vec<f64>=watch.iter().map(|&k|field[k].re).collect();
    timing.total_ms=clock.elapsed().as_secs_f64()*1e3-setup_ms;
    let fold_charge=if charge&&!prepared.plan.far().is_empty(){
        let start=Instant::now();
        let third=prepared.fold.third_order_source(&op,&field)?;
        let(source,label,bound)=if ledger.third_order>0.&&third.iter().any(|z|nonzero(*z)){(third,"third-order",ledger.third_order)}
            else{(op.apply_fold(&field)?,"fold",ledger.fold)};
        let stop=Stop{normalize:false,tolerance:SENSITIVITY_SHARE,relative:true,residual_guard:SENSITIVITY_GUARD};
        let response=gmres(&mut op,&mut batch,&source,&stop,groups,options,&vectors)?;
        applications+=response.applications;preconditions+=response.preconditions;
        let delta:Vec<f64>=watch.iter().map(|&k|response.x[k].re).collect();
        let size=ledger::metric(&delta,&vec![0.;n],groups);
        let sensitivity=if bound>0.{size/bound}else{0.};
        Some(FoldCharge{source:label,response:size,source_bound:bound,charges:PopulationCharges::new(&ledger,sensitivity),
            steps:response.steps,milliseconds:start.elapsed().as_secs_f64()*1e3})
    }else{None};
    Ok(OrderSolve{order,modes,nu_max:op.nu_max(),populations,field,steps:krylov.steps,restarts:krylov.restarts,restart:krylov.restart,stop:krylov.stop,
        residual,null_residual,trace,applications,preconditions,operator_work:op.work,precondition_work:batch.work,regularized:batch.regularized,
        edges:op.edge_counts,timing,ledger,charge:fold_charge,history:krylov.history,steps_to_tight_residual:krylov.tight})
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_steady_state_is_the_null_vector_of_the_dense_lattice(){
        // Against a dense LU of the same truncated, folded lattice with the
        // trace row: the algebra only (truncation and fold are the same).
        let f=fixture();let p=prepared(&f);let groups=vec![vec![0,1],vec![2,3]];
        let dressed=DenseDressed::new(&p).unwrap();
        let options=SteadyOptions{algebraic:1e-12,residual_guard:1e-13,..SteadyOptions::default()};
        let s=solve_order(&p,&dressed,2,&groups,&options,true).unwrap();
        let op=LatticeOperator::new(&p,&s.modes).unwrap();
        let exact=super::super::tests::dense_steady(&p,&op.dense(),op.centre,4);
        let error=ledger::metric(&exact,&s.populations,&groups);
        assert!(error<1e-10,"{error:e}, stop {}",s.stop);
        assert!((s.populations.iter().sum::<f64>()-1.).abs()<1e-12);
        let charge=s.charge.unwrap();assert_eq!(charge.source,"third-order");
        assert!(charge.charges.sensitivity>0.&&charge.charges.total().is_finite());
    }
    #[test]
    fn lattice_fold_charge_covers_the_explicit_first_far_shell(){
        // The folded steady state against an explicit first far shell (near
        // couplings between far modes included): the difference is the fold
        // remainder, and the charged bound must cover it.
        let f=fixture();let p=prepared(&f);let n=4;let groups=vec![vec![0,1],vec![2,3]];
        let dressed=DenseDressed::new(&p).unwrap();
        let options=SteadyOptions{algebraic:1e-12,residual_guard:1e-13,..SteadyOptions::default()};
        let s=solve_order(&p,&dressed,2,&groups,&options,true).unwrap();
        // Every first-shell far charge the ball reaches (the 2003 drive moves the
        // near index by one), near couplings between them included.
        let charges:Vec<Charge>=(-1..=1).flat_map(|b:i32|{let reach=if b==0{2}else{3};(-reach..=reach).map(move|a|vec![a,b])}).collect();
        let(matrix,centre)=super::super::tests::explicit_matrix(&p,&charges);
        let explicit=super::super::tests::dense_steady(&p,&matrix,centre,n);
        let actual=ledger::metric(&explicit,&s.populations,&groups);
        let charged=s.charge.unwrap().charges.total();
        assert!(actual<=charged,"actual fold remainder {actual:e} above its charge {charged:e}");
    }
}
