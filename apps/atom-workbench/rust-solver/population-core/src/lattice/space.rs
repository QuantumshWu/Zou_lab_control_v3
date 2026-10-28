//! The one rational Krylov space of a request, built from x0 on the lattice
//! P u Q, and its poles.
//!
//! Columns: w = (xi - A)^-1 v with v the last vector of the Krylov sequence
//! (orthonormalized within the sequence only, so the continuation stays in
//! the rational Krylov space of x0: a continuation carrying translated
//! copies or the steady vector would solve for other poles). Every column
//! is a Hermitian-symmetric (J-real, (J x)_q = x_-q^dagger) field: a real
//! pole's solve stays in that real space (its J-odd part, inner-solve error,
//! is never formed); a complex pole's solve is the real form w_1 + i w_2,
//! whose parts are both J-real, so the conjugate pole comes free. A
//! direction below NOISE x the inner tolerance of its column is noise and
//! is not added (a nearly repeated pole would add only roundoff); the same
//! floor applies to its part outside the whole basis, and only a Krylov
//! vector that adds a direction is translated. The basis is capped by a
//! memory budget (`Space::new`).
//!
//! Pole 0 is the residue at 0: the bordered solve (A + u tau0^T) w = u gives
//! Pi0 u = Pi0 x0 when the trace is the only left kernel vector; it is done
//! first and joins the basis only. When the bordered system is singular
//! (more conserved quantities) the residue is taken at 0+ on x0
//! (ZERO_PLUS), where the kernel dominates the column.
//!
//! Translations (A S_s = S_s (A - i nu_s) up to the fold's i nu_s K^(1) and
//! the lattice boundary): S_s K(A, x0; xi) = K(A, S_s x0; xi - i nu_s), the
//! space of the translated initial state at the poles xi - i nu_s, with no
//! solve. Every explicit step through the translation gate is active from
//! the start (`charge_set::gated_steps`); its copies S_s v + S_-s v and
//! i (S_s v - S_-s v) of every Krylov vector join the basis (J-real like v).
//!
//! The ladder: sigma log-spaced from LADDER_TOP / t1 down to 1 / T_max
//! (T_max the longest output time), PER_DECADE per decade; complex poles
//! sigma + i omega only at the frequencies omega > sigma of dressed centre
//! modes slower than sigma (|Re lambda| < sigma) that carry at least
//! COMPLEX_WEIGHT x tolerance of x0's population weight (merged within
//! sigma). Below the ladder, the slow points (one per decade down to
//! SLOW_FLOOR) serve the slowest relaxation alone, a rate the outputs'
//! times do not reach. The pipeline orders the live points.
use super::*;
use super::poles::{Lattice,Pole};
use super::krylov::Vectors;
use super::eigenbasis::BlockEigenbasis;

/// Relative residual of the shifted solves, and of the bordered steady
/// solve (its algebraic error is the steady output's).
pub const INNER_TOLERANCE:f64=1e-7;
pub const STEADY_TOLERANCE:f64=1e-9;
/// The bordered solve counts as singular above this true residual.
const BORDERED_SINGULAR:f64=1e-6;
/// The pole 0+ (1/us) of a singular bordered system.
const ZERO_PLUS:f64=1e-9;
/// A Krylov direction below NOISE x INNER_TOLERANCE of its column is noise.
const NOISE:f64=100.;
/// A basis vector whose part outside the basis is below this fraction of
/// its norm is dependent.
const APPEND_FLOOR:f64=1e-11;
const PER_DECADE:f64=2.;
/// The slow points reach relaxation rates down to SLOW_FLOOR (1/us).
pub const SLOW_FLOOR:f64=1e-7;
const LADDER_TOP:f64=10.;
const COMPLEX_WEIGHT:f64=0.1;

/// One solve of the space.
#[derive(Clone,Debug)]
/// One solve: its pole, inner steps, true relative residual, directions
/// added, time (ms) and lattice block operations (operator and
/// preconditioner applications per block).
pub struct Column{pub pole:Pole,pub steps:usize,pub residual:f64,pub added:usize,pub milliseconds:f64,pub blocks:usize}

pub struct Space{
    /// Orthonormal basis (real inner product) and its images A W.
    pub basis:Vec<Vec<C>>,pub images:Vec<Vec<C>>,
    krylov:Vec<Vec<C>>,
    /// Active translations: the step and, per mode, the source modes of
    /// q - s and of q + s.
    translations:Vec<(Charge,Vec<Option<usize>>,Vec<Option<usize>>)>,
    vectors:Vectors,noise_floor:f64,
    /// Most basis columns (the memory budget).
    capacity:usize,
    pub initial_norm:f64,
    pub columns:Vec<Column>,
    pub images_ms:f64,
}
impl Space{
    /// The space of x0 = rho0 on the centre, at most `capacity` columns.
    pub fn new(lattice:&mut Lattice<'_>,initial:&[C],capacity:usize)->Result<Self,String>{
        let op=lattice.operator();let d=op.n*op.n;let mut x0=vec![ZERO;op.dimension()];
        x0[op.centre*d..(op.centre+1)*d].copy_from_slice(initial);
        let vectors=Vectors::new(op,1)?;let initial_norm=vectors.norm(&x0);
        if !(initial_norm>0.){return Err("Zero initial density".into());}
        let mut space=Self{basis:vec![],images:vec![],krylov:vec![],translations:vec![],vectors,noise_floor:0.,capacity:capacity.max(1),initial_norm,columns:vec![],images_ms:0.};
        space.add_krylov(lattice,x0)?;Ok(space)
    }
    pub fn translations(&self)->Vec<Charge>{self.translations.iter().map(|t|t.0.clone()).collect()}
    /// Whether the basis has reached its capacity.
    pub fn full(&self)->bool{self.basis.len()>=self.capacity}
    /// The floor of a new direction (relative to its norm).
    fn floor(&self)->f64{self.noise_floor.max(APPEND_FLOOR)}
    /// Activates the translation along s: the copies of every Krylov vector.
    pub fn activate(&mut self,lattice:&mut Lattice<'_>,step:&Charge)->Result<usize,String>{
        let op=lattice.operator();
        let source=|sign:i32|->Vec<Option<usize>>{op.modes.iter().map(|q|{
            let shifted:Option<Charge>=q.iter().zip(step).map(|(&a,&b)|a.checked_sub(sign*b)).collect();
            shifted.and_then(|c|op.index.get(&c).copied())}).collect()};
        self.translations.push((step.clone(),source(1),source(-1)));
        let t=self.translations.len()-1;let mut added=0;
        let floor=self.floor();
        for v in self.krylov.clone(){for copy in self.translates(&v,t){added+=usize::from(self.append(lattice,copy,floor)?);}}
        Ok(added)
    }
    /// S_s v + S_-s v and i (S_s v - S_-s v) of translation t.
    fn translates(&self,v:&[C],t:usize)->[Vec<C>;2]{
        let(_,minus,plus)=&self.translations[t];let d=v.len()/minus.len();
        let(mut sum,mut difference)=(vec![ZERO;v.len()],vec![ZERO;v.len()]);
        for q in 0..minus.len(){
            let a=minus[q].map(|p|&v[p*d..(p+1)*d]);let b=plus[q].map(|p|&v[p*d..(p+1)*d]);
            for e in 0..d{let x=a.map_or(ZERO,|a|a[e]);let y=b.map_or(ZERO,|b|b[e]);
                sum[q*d+e]=C{re:x.re+y.re,im:x.im+y.im};difference[q*d+e]=C{re:-(x.im-y.im),im:x.re-y.re};}
        }
        [sum,difference]
    }
    /// Appends a vector orthonormalized against the basis, with its image,
    /// when its part outside the basis is above `floor` of its norm and the
    /// basis has room.
    fn append(&mut self,lattice:&mut Lattice<'_>,mut w:Vec<C>,floor:f64)->Result<bool,String>{
        let size=self.vectors.norm(&w);if !(size>0.)||self.full(){return Ok(false);}
        for _ in 0..2{let refs:Vec<&[C]>=self.basis.iter().map(|v|v.as_slice()).collect();let h=self.vectors.dots(&refs,&w);self.vectors.subtract(&refs,&h,&mut w)?;}
        let rest=self.vectors.norm(&w);if !(rest>floor*size){return Ok(false);}
        for v in w.iter_mut(){*v=scaled(*v,1./rest);}
        let clock=Instant::now();let mut image=vec![ZERO;w.len()];lattice.apply(&w,&mut image)?;self.images_ms+=clock.elapsed().as_secs_f64()*1e3;
        self.basis.push(w);self.images.push(image);Ok(true)
    }
    /// A Krylov vector: orthonormalized within the sequence (dropped below
    /// the noise floor), appended to the basis with its translated copies.
    fn add_krylov(&mut self,lattice:&mut Lattice<'_>,p:Vec<C>)->Result<usize,String>{
        let size=self.vectors.norm(&p);let mut v=p;
        for _ in 0..2{let refs:Vec<&[C]>=self.krylov.iter().map(|x|x.as_slice()).collect();let h=self.vectors.dots(&refs,&v);self.vectors.subtract(&refs,&h,&mut v)?;}
        let rest=self.vectors.norm(&v);if !(rest>self.noise_floor*size)||!(rest>0.){return Ok(0);}
        for x in v.iter_mut(){*x=scaled(*x,1./rest);}
        self.krylov.push(v.clone());
        let floor=self.floor();
        if !self.append(lattice,v.clone(),floor)?{return Ok(0);}
        let mut added=1;
        for t in 0..self.translations.len(){for copy in self.translates(&v,t){added+=usize::from(self.append(lattice,copy,floor)?);}}
        Ok(added)
    }
    /// One column at a pole; returns the directions it added.
    pub fn solve(&mut self,lattice:&mut Lattice<'_>,pole:Pole)->Result<usize,String>{
        let clock=Instant::now();let length=self.basis[0].len();let before=lattice.work().block_operations();
        let(inner,steady)=(INNER_TOLERANCE,STEADY_TOLERANCE);
        let(added,steps,residual)=match pole{
            Pole::Zero=>{
                let op=lattice.operator();let n=op.n;let c0=op.centre*n*n;let mut u=vec![ZERO;length];for i in 0..n{u[c0+i*n+i]=C{re:1./n as f64,im:0.};}
                let bordered=lattice.solve(Pole::Zero,&[u],steady);
                let(w,steps,residual)=match bordered{Ok((w,steps,residual))if residual<=BORDERED_SINGULAR=>(w.into_iter().next().unwrap_or_default(),steps,residual),
                    _=>{let x0:Vec<C>=self.krylov[0].iter().map(|v|C{re:-v.re,im:-v.im}).collect();
                        let(w,steps,residual)=lattice.solve(Pole::Real(ZERO_PLUS),&[x0],inner)?;(w.into_iter().next().unwrap_or_default(),steps,residual)}};
                let floor=self.floor();(usize::from(self.append(lattice,w,floor)?),steps,residual)
            }
            Pole::Real(_)=>{
                let f:Vec<C>=self.krylov.last().ok_or("Empty Krylov sequence")?.iter().map(|v|C{re:-v.re,im:-v.im}).collect();
                let(w,steps,residual)=lattice.solve(pole,&[f],inner)?;
                let mut added=0;for column in w{added+=self.add_krylov(lattice,column)?;}
                (added,steps,residual)
            }
            Pole::Complex(_)=>{
                let f:Vec<C>=self.krylov.last().ok_or("Empty Krylov sequence")?.iter().map(|v|C{re:-v.re,im:-v.im}).collect();
                let(w,steps,residual)=lattice.solve(pole,&[f,vec![ZERO;length]],inner)?;
                let mut added=0;for column in w{added+=self.add_krylov(lattice,column)?;}
                (added,steps,residual)
            }
        };
        self.noise_floor=NOISE*inner;
        self.columns.push(Column{pole,steps,residual,added,milliseconds:clock.elapsed().as_secs_f64()*1e3,blocks:lattice.work().block_operations()-before});
        Ok(added)
    }
    pub fn solves(&self)->usize{self.columns.len()}
}

/// The ladder of a request: real poles from LADDER_TOP / t1 down to
/// 1 / T_max, and the complex poles at the frequencies of x0's slow dressed
/// centre modes; then the slow points below it.
pub fn ladder(eigen:&BlockEigenbasis,initial:&[C],t1:f64,longest:f64,tolerance:f64)->Result<(Vec<Pole>,Vec<bool>),String>{
    let(high,low)=(LADDER_TOP/t1,1./longest);
    let count=((high/low).log10()*PER_DECADE).ceil().max(1.)as usize+1;
    let sigmas:Vec<f64>=(0..count).map(|i|high*(low/high).powf(i as f64/(count-1)as f64)).collect();
    let weights=eigen.mode_weights(initial)?;
    let mut poles:Vec<Pole>=sigmas.iter().map(|&s|Pole::Real(s)).collect();
    for &s in &sigmas{
        let mut frequencies:Vec<f64>=eigen.lambda.iter().zip(&weights).filter(|(l,&w)|l.re.abs()<s&&l.im>s&&w>=COMPLEX_WEIGHT*tolerance).map(|(l,_)|l.im).collect();
        frequencies.sort_by(f64::total_cmp);
        let mut chosen=vec![0.];
        for w in frequencies{if chosen.iter().all(|&v|(w-v).abs()>s){chosen.push(w);}}
        poles.extend(chosen.into_iter().skip(1).map(|w|Pole::Complex(C{re:s,im:w})));
    }
    let mut slow=vec![false;poles.len()];
    let mut sigma=low/10.;
    while sigma>=SLOW_FLOOR*(1.-1e-9){poles.push(Pole::Real(sigma));slow.push(true);sigma/=10.;}
    Ok((poles,slow))
}
