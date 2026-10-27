//! The slow eigenpairs of the folded lattice operator A at one near order
//! (beat-mediated paths between modes included): the uniqueness and
//! slowest-relaxation report, and the long-time machinery (`finite`).
//!
//! Shift-invert Arnoldi with M^-1 on the trace-free fields, M = A + u v^T the
//! steady trace-row system: M maps trace-free fields to trace-free fields
//! and equals A there, so the slowest modes have the largest Ritz values
//! theta = 1/lambda while the steady state (lambda = 0) is excluded. Every
//! M^-1 q is a preconditioned GMRES solve from zero (starting from the
//! projection on the steady solve's Krylov space saved no steps, and
//! Rayleigh-Ritz on that space alone left eigen-residuals of 1e2 to 1e5).
//!
//! Two purposes, one mechanism: the steady report (uniqueness, the slowest
//! relaxation) needs eigenvalues only; the long-time machinery (`finite`)
//! also needs eigenvectors, so it takes more steps at a tighter tolerance.
//! An eigenvalue counts (conserved quantities, the slowest relaxation,
//! artifacts) once stable between the last two Arnoldi steps or resolved:
//! its eigenvector's residual |A r - lambda r| / (|Re lambda| |r|) is at most
//! RESIDUAL (it bounds the relative error of a mode's propagated amplitude
//! over its lifetime,
//! charged by the caller). Only resolved modes enter the expansion.
//!
//! The shift at zero orders the modes by |lambda|: a slowly decaying mode
//! that precesses fast (a ground-state coherence at the Larmor frequency) is
//! out of its reach. The long-time machinery detects such modes by an
//! expansion error that does not decay with the switch time.
//!
//! Real coordinates: a complex pair lambda, conj(lambda) is carried by two
//! real fields a, b with A a = l_r a - l_i b and A b = l_i a + l_r b, the
//! coefficients of x = alpha a + beta b evolving by a damped rotation.
//!
//! An accepted mode that grows is a truncation artifact of the lattice: it
//! is excluded and reported (its growth enters the caller's charge).
//! |lambda| below FLOOR is a conserved quantity.
use super::*;
use super::krylov::{self,KrylovOptions,System};

/// What the modes are for.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Purpose{Report,LongTime}
/// Arnoldi steps and solve tolerances per purpose. Measured on four lasers
/// + tweezer (order 3): 12 long-time steps resolve the three slowest modes
/// (rates to 8e-3/us, eigen-residuals 1e-7), and the expansion then needs a
/// switch time of 40 us (a second, longer Krylov propagation); 24 steps add
/// the modes to rates of about 1/us, and it switches at 20 us. Solves to
/// 1e-6 leave eigen-residuals of 4e-7 there at two thirds of the cost of
/// 1e-8; 18 steps leave 3e-5, 8 steps 5e-3.
const REPORT_STEPS:usize=4;
const LONG_TIME_STEPS:usize=24;
const REPORT_RTOL:f64=1e-6;
const LONG_TIME_RTOL:f64=1e-6;
/// Relative change of a reported eigenvalue over the last Arnoldi step.
const STABLE:f64=1e-3;
/// Relative eigenpair residual of a mode kept for the long-time machinery
/// (at 1e-2 a precessing mode resolved to 8e-3 put 1e-3 of misfit into the
/// expansion on the test fixture).
pub const RESIDUAL:f64=1e-3;
/// Rates below this (1/us, relaxation slower than 1000 s) are conserved.
pub const FLOOR:f64=1e-9;

/// A mode of the expansion (resolved: its eigenvector's residual is below
/// RESIDUAL).
#[derive(Clone,Debug)]
pub struct SlowMode{
    pub lambda:C,
    /// The real field (lambda real), or the real and imaginary parts of the
    /// pair's eigenvector.
    pub a:Vec<C>,pub b:Option<Vec<C>>,
    /// Relative residual |A r - lambda r| / (|Re lambda| |r|).
    pub residual:f64,
}
/// A Ritz value with its relative change over the last Arnoldi step and,
/// for the long-time purpose, its eigenvector's relative residual.
#[derive(Clone,Debug)]
pub struct Ritz{pub lambda:C,pub change:f64,pub residual:Option<f64>}
#[derive(Clone,Debug,Default)]
pub struct SlowModes{
    /// Resolved decaying modes, slowest first (long-time purpose).
    pub modes:Vec<SlowMode>,
    /// Conserved quantities, the trace included.
    pub conserved:usize,
    /// Slowest relaxation rate (1/us) of the accepted decaying modes.
    pub slowest_rate:Option<f64>,
    /// Accepted growing modes (truncation artifacts) and their largest rate.
    pub artifacts:usize,pub artifact_growth:f64,
    /// Every Ritz value with its acceptance measure (diagnostics).
    pub ritz:Vec<Ritz>,
    pub solve_steps:usize,pub milliseconds:f64,
}

/// A trace-free Hermitian-symmetric field (fixed pseudo-random entries).
fn start_field(system:&System<'_,'_>)->Vec<C>{
    let op=&system.op;let n=op.n;let d=n*n;let modes=op.modes.len();
    let mut state=0x2545f4914f6cdd1d_u64;let mut random=||{state^=state<<13;state^=state>>7;state^=state<<17;(state>>11)as f64/(1u64<<53)as f64-0.5};
    let mut x=vec![ZERO;modes*d];
    for q in 0..modes{if !op.canonical[q]{continue;}for k in 0..d{x[q*d+k]=C{re:random(),im:random()};}}
    mirror_field(&mut x,n,&op.canonical,&op.mirror);
    trace_free(&mut x,system);x
}
fn trace_free(x:&mut[C],system:&System<'_,'_>){
    let n=system.watch.len();let t:f64=system.watch.iter().map(|&k|x[k].re).sum::<f64>()/n as f64;
    for &k in &system.watch{x[k].re-=t;}
}
/// Eigenvalues lambda = 1/theta of the leading m x m block of the Arnoldi
/// Hessenberg (column-major), with the real eigenvectors.
fn ritz(lapack:&Lapack,h:&[Vec<f64>],m:usize)->Result<(Vec<C>,Vec<bool>,Vec<f64>),String>{
    let mut hm=vec![0.;m*m];for(j,col)in h.iter().take(m).enumerate(){for(i,&v)in col.iter().enumerate().take(m){hm[i+m*j]=v;}}
    let(re,im,vectors)=lapack.eigen(&hm,m)?;
    let mut lambda=vec![ZERO;m];let mut pair=vec![false;m];
    for j in 0..m{let size=re[j]*re[j]+im[j]*im[j];if size>0.{lambda[j]=C{re:re[j]/size,im:-im[j]/size};}pair[j]=im[j]>0.&&j+1<m;}
    Ok((lambda,pair,vectors))
}

/// The slow modes of the steady trace-row system's operator.
pub fn slow_modes(system:&mut System<'_,'_>,purpose:Purpose)->Result<SlowModes,String>{
    let clock=Instant::now();let lapack=Arc::clone(&system.op.prepared.lapack);
    let(steps,rtol)=match purpose{Purpose::Report=>(REPORT_STEPS,REPORT_RTOL),Purpose::LongTime=>(LONG_TIME_STEPS,LONG_TIME_RTOL)};
    let options=KrylovOptions{populations:false,residual_guard:rtol,..KrylovOptions::default()};
    let length=system.length();let mut solve_steps=0;
    let mut q:Vec<Vec<C>>=Vec::new();let mut h:Vec<Vec<f64>>=Vec::new();
    let mut v=start_field(system);let norm=system.vectors.norm(&v);for z in v.iter_mut(){*z=scaled(*z,1./norm);}q.push(v);
    for j in 0..steps{
        let solve=krylov::gmres(system,&q[j],None,&[],&options)?;solve_steps+=solve.steps;let mut w=solve.x;
        trace_free(&mut w,system);
        let refs:Vec<&[C]>=q.iter().map(|x|x.as_slice()).collect();
        let mut column=system.vectors.dots(&refs,&w);system.vectors.subtract(&refs,&column,&mut w)?;
        let again=system.vectors.dots(&refs,&w);system.vectors.subtract(&refs,&again,&mut w)?;
        for(a,b)in column.iter_mut().zip(&again){*a+=b;}
        let hn=system.vectors.norm(&w);column.push(hn);h.push(column);
        if !(hn>1e-14){break;}
        q.push(w.iter().map(|z|scaled(*z,1./hn)).collect());
    }
    let m=h.len();let(lambda,pair,vectors)=ritz(&lapack,&h,m)?;
    let previous=if m>1{ritz(&lapack,&h,m-1)?.0}else{vec![]};
    // The long-time machinery's eigenvectors (Ritz vectors over q).
    let fields=match purpose{
        Purpose::Report=>vec![],
        Purpose::LongTime=>{let coefficients:Vec<Vec<f64>>=(0..m).map(|c|(0..m).map(|i|vectors[i+m*c]).collect()).collect();
            let basis:Vec<&[C]>=q[..m].iter().map(|x|x.as_slice()).collect();system.vectors.combinations(&basis,&coefficients)?}};
    let mut result=SlowModes{conserved:1,solve_steps,..Default::default()};
    let mut j=0;
    while j<m{
        let(l,p,at)=(lambda[j],pair[j],j);j+=if p{2}else{1};
        if magnitude(l)==0.{continue;}
        let change=previous.iter().map(|x|magnitude(C{re:x.re-l.re,im:x.im-l.im})).fold(f64::INFINITY,f64::min)/magnitude(l);
        // Relative to the decay rate: the amplitude's error over the mode's
        // lifetime (a precessing mode's |lambda| would hide it).
        let rate=l.re.abs().max(FLOOR);
        let residual=if fields.is_empty(){None}else{
            let a=&fields[at];let mut aa=vec![ZERO;length];system.op.apply(a,&mut aa)?;
            Some(if p{let b=&fields[at+1];let mut ab=vec![ZERO;length];system.op.apply(b,&mut ab)?;
                // A a - l_r a + l_i b and A b - l_i a - l_r b.
                let ra:Vec<C>=aa.iter().zip(a).zip(b).map(|((x,y),w)|C{re:x.re-l.re*y.re+l.im*w.re,im:x.im-l.re*y.im+l.im*w.im}).collect();
                let rb:Vec<C>=ab.iter().zip(b).zip(a).map(|((x,y),w)|C{re:x.re-l.re*y.re-l.im*w.re,im:x.im-l.re*y.im-l.im*w.im}).collect();
                (system.vectors.norm(&ra).hypot(system.vectors.norm(&rb)))/(rate*system.vectors.norm(a).hypot(system.vectors.norm(b))).max(f64::MIN_POSITIVE)}
            else{let ra:Vec<C>=aa.iter().zip(a).map(|(x,y)|C{re:x.re-l.re*y.re,im:x.im-l.re*y.im}).collect();
                system.vectors.norm(&ra)/(rate*system.vectors.norm(a)).max(f64::MIN_POSITIVE)})};
        result.ritz.push(Ritz{lambda:l,change,residual});
        // An eigenvalue counts once stable over the last Arnoldi step or
        // resolved (its eigenvector's residual); a mode enters the
        // expansion only resolved.
        let resolved=residual.is_some_and(|r|r<=RESIDUAL);
        if !(change<=STABLE||resolved){continue;}
        if magnitude(l)<FLOOR{result.conserved+=1;}
        else if l.re>0.{result.artifacts+=1;result.artifact_growth=result.artifact_growth.max(l.re);}
        else{
            result.slowest_rate=Some(result.slowest_rate.map_or(-l.re,|r:f64|r.min(-l.re)));
            if let(true,Some(r))=(resolved,residual){result.modes.push(SlowMode{lambda:l,a:fields[at].clone(),b:p.then(||fields[at+1].clone()),residual:r});}
        }
    }
    result.modes.sort_by(|x,y|(-x.lambda.re).total_cmp(&-y.lambda.re));
    result.milliseconds=clock.elapsed().as_secs_f64()*1e3;
    Ok(result)
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::operator::LatticeOperator;
    use super::super::preconditioner::StructuredDressed;
    /// Eigenvalues of a dense complex matrix (row-major) through its real
    /// 2N x 2N form (each eigenvalue appears with its conjugate).
    fn dense_eigenvalues(p:&LatticePrepared,a:&[C],size:usize)->Vec<C>{
        let real=2*size;let mut big=vec![0.;real*real];
        for r in 0..size{for c in 0..size{let v=a[r*size+c];
            big[r+real*c]=v.re;big[r+real*(c+size)]=-v.im;big[r+size+real*c]=v.im;big[r+size+real*(c+size)]=v.re;}}
        let(re,im,_)=p.lapack.eigen(&big,real).unwrap();
        re.iter().zip(&im).map(|(&re,&im)|C{re,im}).collect()
    }
    #[test]
    fn lattice_slow_modes_are_eigenpairs_of_the_dense_lattice(){
        // The fixture at order 2: every resolved mode is an eigenvalue of the
        // dense lattice matrix, the slowest relaxation is its smallest
        // nonzero |lambda|, and only the trace is conserved.
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let dressed=StructuredDressed::new(&p).unwrap();let op=LatticeOperator::new(&p,&modes).unwrap();
        let exact=dense_eigenvalues(&p,&op.dense(),op.dimension());
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre),0.,true).unwrap();
        let mut system=System::new(op,&mut batch).unwrap();
        let slow=slow_modes(&mut system,Purpose::LongTime).unwrap();
        assert_eq!(slow.conserved,1);assert_eq!(slow.artifacts,0);
        assert!(!slow.modes.is_empty());
        for m in &slow.modes{
            let distance=exact.iter().map(|z|magnitude(C{re:z.re-m.lambda.re,im:z.im-m.lambda.im})).fold(f64::INFINITY,f64::min);
            assert!(distance<=1e-6*magnitude(m.lambda),"{:?} is {distance:e} from the dense spectrum",m.lambda);
            assert!(m.residual<=RESIDUAL);
        }
        let smallest=exact.iter().filter(|z|magnitude(**z)>1e-9).min_by(|a,b|magnitude(**a).total_cmp(&magnitude(**b))).unwrap();
        let rate=slow.slowest_rate.unwrap();
        assert!((rate+smallest.re).abs()<=1e-6*rate,"slowest {rate:e} against {smallest:?}");
    }
}
