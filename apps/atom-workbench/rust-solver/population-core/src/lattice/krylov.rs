//! The primal Krylov solve of the steady trace-row system M x = u, with
//! M = A + u v^T (u = I/n on the centre, v its trace): the time-averaged
//! populations are the central diagonal of x (A x = 0, v^T x = 1 when the
//! null vector is unique).
//!
//! Real coordinates: the fields are Hermitian-symmetric (x_-q = x_q^dagger),
//! a real vector space that M and the dressed preconditioner preserve, with
//! the real inner product Re <a, b>. Inner products are summed over the
//! canonical modes only (weight 2 for a pair, 1 for the centre) and the
//! Hessenberg matrix, rotations and coefficients are real.
//!
//! Right-preconditioned restarted GMRES with a fixed preconditioner, so only
//! the populations of P^-1 v_j are kept per step and the iterate is formed
//! once per cycle. It stops when the populations and the requested groups
//! have stabilised (changes below the algebraic share of the tolerance for
//! STABLE_STEPS consecutive steps) and the true relative residual, recomputed
//! from the iterate after the cycle, is below a guard: the population change
//! alone can plateau in slow directions, and the recursive GMRES residual can
//! fall eight orders below the true one (the operator spans 1e-6 to 5e7
//! rad/us). The longest cycle's Arnoldi relation M P^-1 V_m = V_m+1 Hbar_m
//! is kept for the error estimate (`estimate`): the preconditioner is fixed,
//! so every cycle's relation stays valid, and a final cycle of a few steps
//! (a restart for the true-residual guard) holds too little of the space.
use super::*;
use super::operator::LatticeOperator;
use super::preconditioner::Preconditioner;
use super::ledger;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

const STABLE_STEPS:usize=3;
/// Blocks per reduction chunk: fixed, so sums do not depend on threads.
const VECTOR_CHUNK:usize=16;

#[derive(Clone,Debug)]
pub struct KrylovOptions{
    /// Population-stability tolerance.
    pub algebraic:f64,
    /// Relative true-residual guard. Measured: a true residual of 3e-7 left
    /// 1.8e-5 in the populations (two lasers + tweezer, a 1e-6 /us Raman
    /// manifold); at 1e-8 every case is within 1e-7 of its reference.
    pub residual_guard:f64,
    pub restart:usize,pub maximum_steps:usize,
    /// Krylov basis memory budget (bytes); the restart length shrinks to fit.
    pub basis_bytes:usize,
}
impl Default for KrylovOptions{
    fn default()->Self{
        Self{algebraic:ledger::POPULATION_TOLERANCE*ledger::ALGEBRAIC_SHARE,residual_guard:1e-8,restart:80,maximum_steps:800,basis_bytes:1<<30}
    }
}
#[derive(Clone,Debug,Default)]
pub struct KrylovTiming{pub apply_ms:f64,pub precondition_ms:f64,pub orthogonalize_ms:f64}

/// Deterministic parallel kernels on Hermitian-symmetric fields.
pub struct Vectors{execution:Execution,d:usize,
    /// Real inner-product weight per mode: 2 for a canonical pair member, 1
    /// for the centre, 0 for a mirrored mode.
    weights:Vec<f64>}
impl Vectors{
    pub fn new(op:&LatticeOperator<'_>)->Result<Self,String>{
        let weights=(0..op.modes.len()).map(|q|if !op.canonical[q]{0.}else if op.mirror[q]==q{1.}else{2.}).collect();
        Ok(Self{execution:Execution::new()?,d:op.n*op.n,weights})
    }
    fn ranges(&self,length:usize)->Vec<(usize,usize)>{
        let w=VECTOR_CHUNK*self.d;(0..length.div_ceil(w)).map(|c|(c*w,((c+1)*w).min(length))).collect()
    }
    fn partial(&self,a:&[C],b:&[C],start:usize,end:usize)->f64{
        let d=self.d;let mut s=0.;let mut k=start;
        while k<end{let q=k/d;let stop=((q+1)*d).min(end);let w=self.weights[q];
            if w>0.{let mut t=0.;for i in k..stop{t+=a[i].re*b[i].re+a[i].im*b[i].im;}s+=w*t;}k=stop;}
        s
    }
    /// Re <basis_i, w> for every basis vector.
    pub fn dots(&self,basis:&[&[C]],w:&[C])->Vec<f64>{
        let parts=self.execution.map(self.ranges(w.len()),|(a,b)|basis.iter().map(|v|self.partial(v,w,a,b)).collect::<Vec<f64>>());
        let mut out=vec![0.;basis.len()];for p in parts{for(o,v)in out.iter_mut().zip(p){*o+=v;}}out
    }
    pub fn dot(&self,a:&[C],b:&[C])->f64{self.dots(&[a],b)[0]}
    pub fn norm(&self,w:&[C])->f64{self.dot(w,w).max(0.).sqrt()}
    /// w -= sum_i h_i basis_i.
    pub fn subtract(&self,basis:&[&[C]],h:&[f64],w:&mut[C])->Result<(),String>{
        let d=self.d;let work=basis.len()*w.len();
        self.execution.for_mode_chunks(w,d,work,MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
            for(v,&c)in basis.iter().zip(h){for(t,&x)in chunk.iter_mut().zip(&v[first*d..]){t.re-=c*x.re;t.im-=c*x.im;}}
            Ok(())
        })
    }
}
/// sum_i c_i basis_i.
pub fn combine(basis:&[&[C]],c:&[f64],length:usize)->Vec<C>{
    let mut out=vec![ZERO;length];
    for(v,&a)in basis.iter().zip(c){if a==0.{continue;}for(t,&x)in out.iter_mut().zip(v.iter()){t.re+=a*x.re;t.im+=a*x.im;}}
    out
}

/// The trace-row system on one mode set: M = A + u v^T, its preconditioner
/// and the vector kernels.
pub struct System<'a,'p>{
    pub op:LatticeOperator<'a>,
    pub pre:&'p mut dyn Preconditioner,
    pub vectors:Vectors,
    /// Field indices of the central populations.
    pub watch:Vec<usize>,
    pub timing:KrylovTiming,pub applications:usize,pub preconditions:usize,
}
impl<'a,'p> System<'a,'p>{
    pub fn new(op:LatticeOperator<'a>,pre:&'p mut dyn Preconditioner)->Result<Self,String>{
        let n=op.n;let c0=op.centre*n*n;let watch=(0..n).map(|i|c0+i*n+i).collect();let vectors=Vectors::new(&op)?;
        Ok(Self{op,pre,vectors,watch,timing:KrylovTiming::default(),applications:0,preconditions:0})
    }
    pub fn length(&self)->usize{self.op.dimension()}
    /// y = M x.
    pub fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let start=Instant::now();self.op.apply(x,y)?;
        let n=self.watch.len();let t=self.watch.iter().fold(ZERO,|s,&k|C{re:s.re+x[k].re,im:s.im+x[k].im});
        for &k in &self.watch{add(&mut y[k],scaled(t,1./n as f64));}
        self.timing.apply_ms+=start.elapsed().as_secs_f64()*1e3;self.applications+=1;Ok(())
    }
    /// y = P^-1 x.
    pub fn precondition(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let start=Instant::now();self.pre.apply(x,y)?;
        self.timing.precondition_ms+=start.elapsed().as_secs_f64()*1e3;self.preconditions+=1;Ok(())
    }
    /// The central populations of a field.
    pub fn populations(&self,x:&[C])->Vec<f64>{self.watch.iter().map(|&k|x[k].re).collect()}
    /// The right-hand side u.
    pub fn source(&self)->Vec<C>{
        let n=self.watch.len();let mut b=vec![ZERO;self.length()];for &k in &self.watch{b[k]=C{re:1./n as f64,im:0.};}b
    }
}

/// The last GMRES cycle's Arnoldi relation M P^-1 V_m = V_m+1 Hbar_m in
/// factored form: Hbar_m = Q_H [R; 0] with Q_H the product of the Givens
/// rotations.
pub struct Arnoldi{
    /// v_1 .. v_m+1 (v_m+1 is zero when the relation closed exactly).
    pub basis:Vec<Vec<C>>,
    /// Givens rotations (c, s): rotation j acts on rows j, j+1.
    pub rotations:Vec<(f64,f64)>,
    /// R by columns (column j has j+1 entries).
    pub r:Vec<Vec<f64>>,
    /// Central populations of P^-1 v_j, j < m.
    pub diagonal:Vec<Vec<f64>>,
}
impl Arnoldi{
    pub fn steps(&self)->usize{self.r.len()}
    /// Q_H^T t (t has m + 1 entries).
    pub fn rotate(&self,t:&mut[f64]){
        for(j,&(c,s))in self.rotations.iter().enumerate(){let(a,b)=(t[j],t[j+1]);t[j]=c*a+s*b;t[j+1]=-s*a+c*b;}
    }
    /// Q_H t.
    pub fn unrotate(&self,t:&mut[f64]){
        for(j,&(c,s))in self.rotations.iter().enumerate().rev(){let(a,b)=(t[j],t[j+1]);t[j]=c*a-s*b;t[j+1]=s*a+c*b;}
    }
    /// R^-1 g (g has m entries).
    pub fn back_substitute(&self,g:&[f64])->Vec<f64>{
        let m=self.r.len();let mut y=vec![0.;m];
        for i in(0..m).rev(){let mut s=g[i];for l in i+1..m{s-=self.r[l][i]*y[l];}y[i]=s/self.r[i][i];}
        y
    }
}

pub struct Krylov{
    pub x:Vec<C>,pub steps:usize,pub restarts:usize,pub restart:usize,pub stop:&'static str,
    /// The true relative residual |u - M x| / |u| of the returned iterate.
    pub residual:f64,
    /// The longest cycle's Arnoldi relation.
    pub arnoldi:Arnoldi,
}

/// Solves M x = b from x0 (zero when None).
pub fn gmres(system:&mut System<'_,'_>,b:&[C],x0:Option<Vec<C>>,groups:&[Vec<usize>],options:&KrylovOptions)->Result<Krylov,String>{
    let length=system.length();
    let bnorm=system.vectors.norm(b);
    let fit=(options.basis_bytes/(length*std::mem::size_of::<C>())).saturating_sub(2);
    let restart=options.restart.min(fit).max(8);
    let mut x=x0.unwrap_or_else(||vec![ZERO;length]);let(mut z,mut w)=(vec![ZERO;length],vec![ZERO;length]);
    let mut previous:Option<Vec<f64>>=None;let mut changes:Vec<f64>=Vec::new();
    let(mut steps,mut restarts)=(0,0);
    if bnorm==0.{return Err("Lattice steady source is zero".into());}
    // r = b - M x, recomputed from x after every cycle (see the module
    // documentation).
    let mut r=b.to_vec();
    if x.iter().any(|v|nonzero(*v)){system.apply(&x,&mut w)?;for((a,&bb),c)in r.iter_mut().zip(b).zip(&w){*a=C{re:bb.re-c.re,im:bb.im-c.im};}}
    let mut last_true=system.vectors.norm(&r)/bnorm;
    let mut longest:Option<Arnoldi>=None;
    loop{
        let beta=system.vectors.norm(&r);
        let mut arnoldi=Arnoldi{basis:vec![r.iter().map(|v|scaled(*v,1./beta)).collect()],rotations:vec![],r:vec![],diagonal:vec![]};
        let mut g=vec![beta];
        let xdiag=system.populations(&x);
        let mut y=Vec::new();let mut stopped:Option<&'static str>=None;
        for j in 0..restart{
            system.precondition(&arnoldi.basis[j],&mut z)?;
            arnoldi.diagonal.push(system.populations(&z));
            system.apply(&z,&mut w)?;
            let start=Instant::now();
            let column={let basis:Vec<&[C]>=arnoldi.basis.iter().map(|v|v.as_slice()).collect();
                let h1=system.vectors.dots(&basis,&w);system.vectors.subtract(&basis,&h1,&mut w)?;
                let h2=system.vectors.dots(&basis,&w);system.vectors.subtract(&basis,&h2,&mut w)?;
                h1.iter().zip(&h2).map(|(a,b)|a+b).collect::<Vec<f64>>()};
            let hn=system.vectors.norm(&w);
            system.timing.orthogonalize_ms+=start.elapsed().as_secs_f64()*1e3;
            let mut column=column;column.push(hn);
            arnoldi.rotate(&mut column);
            let size=column[j].hypot(column[j+1]);
            let(c,s)=if size>0.{(column[j]/size,column[j+1]/size)}else{(1.,0.)};
            arnoldi.rotations.push((c,s));column[j]=size;column.truncate(j+1);
            let gj=g[j];g.push(-s*gj);g[j]=c*gj;
            arnoldi.r.push(column);steps+=1;
            let residual=g[j+1].abs()/bnorm;
            y=arnoldi.back_substitute(&g[..=j]);
            let mut current=xdiag.clone();for(k,zk)in arnoldi.diagonal.iter().enumerate(){for(t,&v)in current.iter_mut().zip(zk){*t+=y[k]*v;}}
            let trace:f64=current.iter().sum();let current:Vec<f64>=current.iter().map(|v|v/trace).collect();
            let change=previous.as_ref().map_or(f64::INFINITY,|p|ledger::metric(p,&current,groups));changes.push(change);
            previous=Some(current);
            let stable=changes.len()>=STABLE_STEPS&&changes[changes.len()-STABLE_STEPS..].iter().all(|&c|c<options.algebraic);
            if stable&&residual<=options.residual_guard{stopped=Some("populations-stable");}
            else if hn==0.{stopped=Some("exact");}
            else if residual<=1e-15{stopped=Some("residual-at-roundoff");}
            let last=stopped.is_some()||steps>=options.maximum_steps||j+1==restart;
            arnoldi.basis.push(if hn>0.{w.iter().map(|v|scaled(*v,1./hn)).collect()}else{vec![ZERO;length]});
            if last{break;}
        }
        // x += P^-1 (V y).
        let combination={let basis:Vec<&[C]>=arnoldi.basis[..y.len()].iter().map(|v|v.as_slice()).collect();combine(&basis,&y,length)};
        system.precondition(&combination,&mut z)?;
        for(a,c)in x.iter_mut().zip(&z){add(a,*c);}
        system.apply(&x,&mut w)?;
        for((a,&bb),c)in r.iter_mut().zip(b).zip(&w){*a=C{re:bb.re-c.re,im:bb.im-c.im};}
        let true_residual=system.vectors.norm(&r)/bnorm;
        let arnoldi=match longest.take(){Some(l)if l.steps()>arnoldi.steps()=>l,_=>arnoldi};
        if let Some(stop)=stopped{
            if true_residual<=options.residual_guard{return Ok(Krylov{x,steps,restarts,restart,stop,residual:true_residual,arnoldi});}
            // Restarting no longer reduces the true residual: its floor.
            if true_residual>0.5*last_true{return Ok(Krylov{x,steps,restarts,restart,stop:"true-residual-floor",residual:true_residual,arnoldi});}
        }
        longest=Some(arnoldi);
        if steps>=options.maximum_steps{return Err(format!("Lattice steady GMRES did not stabilise in {steps} steps (true residual {true_residual:e})"));}
        last_true=true_residual;restarts+=1;
    }
}

/// x_q <- (x_q + x_-q^dagger) / 2: removes the roundoff asymmetry of a
/// source assembled on every mode.
pub fn symmetrize(op:&LatticeOperator<'_>,x:&mut[C]){
    let n=op.n;let d=n*n;
    for q in 0..op.modes.len(){let p=op.mirror[q];if p<q{continue;}
        for i in 0..n{for j in 0..n{let(a,b)=(q*d+i*n+j,p*d+j*n+i);if p==q&&j<i{continue;}
            let m=C{re:0.5*(x[a].re+x[b].re),im:0.5*(x[a].im-x[b].im)};x[a]=m;x[b]=conj(m);}}}
}
