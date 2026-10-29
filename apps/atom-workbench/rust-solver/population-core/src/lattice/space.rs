//! The one rational Krylov space of a request, built from x0 on the lattice
//! P u Q, and its poles.
//!
//! Trace by construction: every Fourier component of the generator, and so
//! every coupling and fold term of the lattice operator, annihilates the
//! trace, so each block trace tau_q(x) = tr x_q is an exact left eigenvector,
//! tau_q A = -i nu_q tau_q, whatever P u Q is (measured to 1e-15 of the
//! operator's entries). The solution stays in u + K, u = I / n on the centre
//! (tr u = 1 = tr x0) and K the fields of zero block traces, and A maps
//! u + K into K. The basis is [u / |u|, then K]: its first column is the
//! trace carrier, every later direction is projected into K (its block
//! traces removed along I / n: the orthogonal projection, so the basis stays
//! orthonormal) before it is orthogonalized. The reduced model then holds
//! the carrier's coefficient fixed and conserves every block trace exactly
//! (`reduced`).
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
//! (`reduced::KERNEL`, the rate below which a mode is conserved), where the
//! kernel dominates the column.
//!
//! Translations (A S_s = S_s (A - i nu_s) up to the fold's i nu_s K^(1) and
//! the lattice boundary): S_s K(A, x0; xi) = K(A, S_s x0; xi - i nu_s), the
//! space of the translated initial state at the poles xi - i nu_s, with no
//! solve. Every explicit step through the translation gate is active from
//! the start (`charge_set::gated_steps`); its copies S_s v + S_-s v and
//! i (S_s v - S_-s v) of every Krylov vector join the basis (J-real like v).
//!
//! The ladder: sigma log-spaced from LADDER_TOP / t1 down to 1 / T_max
//! (T_max the grid's last time, else the longest horizon: `pipeline`),
//! PER_DECADE per decade; complex poles
//! sigma + i omega only at the frequencies omega > sigma of dressed centre
//! modes slower than sigma (|Re lambda| < sigma) that carry at least
//! COMPLEX_WEIGHT x tolerance of x0's population weight (merged within
//! sigma). The pipeline orders the live points.
use super::*;
use super::poles::{Lattice,Pole};
use super::krylov::Vectors;
use super::eigenbasis::BlockEigenbasis;

/// Relative residual of the shifted solves, and of the bordered steady
/// solve (its algebraic error is the steady output's).
pub const INNER_TOLERANCE:f64=1e-7;
pub const STEADY_TOLERANCE:f64=1e-9;
/// The bordered solve counts as singular above this true residual: a
/// deliberate margin of 1000x over STEADY_TOLERANCE, which a regular
/// bordered system reaches and a singular one (more conserved quantities)
/// stalls far above.
const BORDERED_SINGULAR:f64=1e-6;
/// A Krylov direction below NOISE x INNER_TOLERANCE of its column is noise.
/// Measured: without the floor a repeated pole near 1 / T added inner-solve
/// noise and a spurious mode (Re theta +4.6e-2 /us) on two lasers at 10 ms.
const NOISE:f64=100.;
/// A basis vector whose part outside the basis is below this fraction of
/// its norm is dependent.
const APPEND_FLOOR:f64=1e-11;
/// Ladder points per decade: a deliberate choice, not measured (the
/// reference runs used two to three); the ranking chooses among them.
const PER_DECADE:f64=2.;
/// The ladder's top, LADDER_TOP / t1: the shift-and-invert rule at the
/// earliest output time, t1 the first sample time (without a grid T_max /
/// LADDER_SAMPLES, `pipeline`). Van den Eshof and Hochbruck's choice for
/// exp(t A) v by (I - gamma A)^-1 is gamma = t / 10, a pole at 1 / gamma =
/// 10 / t1: a rule, not a measured constant. Measured against it: three
/// lasers at 20 us stalled at 3.8-5.8e-3 over 48 solves with 1 / t1, reached
/// 1.9e-3 at 24 solves with 10 / t1 (half the inner steps per solve); 30 / t1
/// gave the same errors on five lasers at 10 us; twice and four times the
/// top, points above it and tops from x0's dressed spectrum were none better
/// across the board (the necessity review).
const LADDER_TOP:f64=10.;
/// A deliberate margin: a dressed mode carrying less than a tenth of the
/// tolerance of x0's population weight gets no complex pole.
const COMPLEX_WEIGHT:f64=0.1;

#[derive(Clone,Debug)]
/// One solve of the space: its pole, inner steps, true relative residual, directions
/// added, time (ms) and lattice block operations (operator and
/// preconditioner applications per block).
pub struct Column{pub pole:Pole,pub steps:usize,pub residual:f64,pub added:usize,pub milliseconds:f64,pub blocks:usize}

pub struct Space{
    /// Orthonormal basis (real inner product) and its images A W.
    pub basis:Vec<Vec<C>>,pub images:Vec<Vec<C>>,
    krylov:Vec<Vec<C>>,
    /// (research, S3) Per Krylov vector whether translations copy it, and
    /// whether the next solve's vectors are copied.
    copyable:Vec<bool>,pub untranslated_next:bool,
    /// Active translations: the step and, per mode, the source modes of
    /// q - s and of q + s.
    translations:Vec<(Charge,Vec<Option<usize>>,Vec<Option<usize>>)>,
    vectors:Vectors,noise_floor:f64,
    /// Most basis columns (the memory budget).
    capacity:usize,
    /// x0's coordinates in the basis: the carrier's |u| and its K part's |x0 - u|
    /// (the further columns are orthogonal to both).
    pub initial:Vec<f64>,
    pub columns:Vec<Column>,
    pub images_ms:f64,
}
impl Space{
    /// The space of x0 = rho0 on the centre, at most `capacity` columns.
    pub fn new(lattice:&mut Lattice<'_>,initial:&[C],capacity:usize)->Result<Self,String>{
        let op=lattice.operator();let d=op.n*op.n;let mut x0=vec![ZERO;op.dimension()];
        x0[op.centre*d..(op.centre+1)*d].copy_from_slice(initial);
        let vectors=Vectors::new(op,1)?;
        if !(vectors.norm(&x0)>0.){return Err("Zero initial density".into());}
        // The trace carrier u = I / n on the centre (tr u = tr x0 = 1).
        let n=op.n;let mut u=vec![ZERO;op.dimension()];for i in 0..n{u[op.centre*d+i*n+i]=C{re:1./n as f64,im:0.};}
        let carrier=vectors.norm(&u);
        let mut space=Self{basis:vec![],images:vec![],krylov:vec![],copyable:vec![],untranslated_next:false,translations:vec![],vectors,noise_floor:0.,capacity:capacity.max(2),initial:vec![carrier],columns:vec![],images_ms:0.};
        space.append(lattice,u,0.)?;
        // x0's K part (zero when rho0 is I / n).
        let mut k0=x0.clone();project_traces(&mut k0,n);let rest=space.vectors.norm(&k0);
        space.add_krylov(lattice,x0)?;
        if space.basis.len()>1{space.initial.push(rest);}
        Ok(space)
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
        for(v,ok)in self.krylov.clone().into_iter().zip(self.copyable.clone()){if !ok{continue;}for copy in self.translates(&v,t){added+=usize::from(self.append(lattice,copy,floor)?);}}
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
    /// basis has room. Every column but the first (the trace carrier) is
    /// projected into K first, and again after the orthogonalization (which
    /// leaves K only at roundoff).
    fn append(&mut self,lattice:&mut Lattice<'_>,mut w:Vec<C>,floor:f64)->Result<bool,String>{
        let n=lattice.operator().n;let carrier=self.basis.is_empty();
        let size=self.vectors.norm(&w);if !(size>0.)||self.full(){return Ok(false);}
        if !carrier{project_traces(&mut w,n);}
        for _ in 0..2{let refs:Vec<&[C]>=self.basis.iter().map(|v|v.as_slice()).collect();let h=self.vectors.dots(&refs,&w);self.vectors.subtract(&refs,&h,&mut w)?;}
        if !carrier{project_traces(&mut w,n);}
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
        self.krylov.push(v.clone());self.copyable.push(!self.untranslated_next);
        let floor=self.floor();
        if !self.append(lattice,v.clone(),floor)?{return Ok(0);}
        let mut added=1;
        if !self.untranslated_next{for t in 0..self.translations.len(){for copy in self.translates(&v,t){added+=usize::from(self.append(lattice,copy,floor)?);}}}
        Ok(added)
    }
    /// (research, S3) A column at the pole at infinity: A v of the last
    /// Krylov vector (one operator application, no solve).
    pub fn infinite(&mut self,lattice:&mut Lattice<'_>)->Result<usize,String>{
        let v=self.krylov.last().ok_or("Empty Krylov sequence")?.clone();let mut w=vec![ZERO;v.len()];lattice.apply(&v,&mut w)?;
        self.add_krylov(lattice,w)
    }
    /// One column at a pole; returns the directions it added.
    pub fn solve(&mut self,lattice:&mut Lattice<'_>,pole:Pole)->Result<usize,String>{
        let clock=Instant::now();let length=self.basis[0].len();let before=lattice.work().block_operations();
        // (research, S2) ATOM_INNER_TOL: the shifted solves' relative residual.
        let inner=std::env::var("ATOM_INNER_TOL").ok().and_then(|v|v.parse::<f64>().ok()).unwrap_or(INNER_TOLERANCE);
        let steady=STEADY_TOLERANCE;
        let(added,steps,residual)=match pole{
            Pole::Zero=>{
                let op=lattice.operator();let n=op.n;let c0=op.centre*n*n;let mut u=vec![ZERO;length];for i in 0..n{u[c0+i*n+i]=C{re:1./n as f64,im:0.};}
                let bordered=lattice.solve(Pole::Zero,&[u],steady);
                let(w,steps,residual)=match bordered{Ok((w,steps,residual))if residual<=BORDERED_SINGULAR=>(w.into_iter().next().unwrap_or_default(),steps,residual),
                    _=>{let x0:Vec<C>=self.krylov[0].iter().map(|v|C{re:-v.re,im:-v.im}).collect();
                        let(w,steps,residual)=lattice.solve(Pole::Real(super::reduced::KERNEL),&[x0],inner)?;(w.into_iter().next().unwrap_or_default(),steps,residual)}};
                let floor=self.floor();(usize::from(self.append(lattice,w,floor)?),steps,residual)
            }
            Pole::Real(_)|Pole::Complex(_)=>{
                // A complex pole's real form: the second right-hand side zero.
                let f:Vec<C>=self.krylov.last().ok_or("Empty Krylov sequence")?.iter().map(|v|C{re:-v.re,im:-v.im}).collect();
                let rhs=if pole.copies()==2{vec![f,vec![ZERO;length]]}else{vec![f]};
                let(w,steps,residual)=lattice.solve(pole,&rhs,inner)?;
                let mut added=0;for column in w{added+=self.add_krylov(lattice,column)?;}
                (added,steps,residual)
            }
        };
        self.noise_floor=NOISE*inner;
        self.columns.push(Column{pole,steps,residual,added,milliseconds:clock.elapsed().as_secs_f64()*1e3,blocks:lattice.work().block_operations()-before});
        Ok(added)
    }
    pub fn solves(&self)->usize{self.columns.len()}
    pub fn krylov_count(&self)->usize{self.krylov.len()}
}

/// Projects a Hermitian-symmetric field into K: every block x_q loses its
/// trace along I / n (x_q - tr(x_q) I / n; a mirror block's trace is the
/// conjugate, so the field stays Hermitian-symmetric).
pub fn project_traces(x:&mut[C],n:usize){
    let d=n*n;
    for block in x.chunks_exact_mut(d){
        let t=(0..n).fold(ZERO,|s,i|C{re:s.re+block[i*n+i].re,im:s.im+block[i*n+i].im});
        for i in 0..n{block[i*n+i].re-=t.re/n as f64;block[i*n+i].im-=t.im/n as f64;}
    }
}

/// The ladder of a request: real poles from LADDER_TOP / t1 down to
/// 1 / T_max, and the complex poles at the frequencies of x0's slow dressed
/// centre modes.
pub fn ladder(eigen:&BlockEigenbasis,initial:&[C],nu:&[f64],t1:f64,longest:f64,tolerance:f64)->Result<Vec<Pole>,String>{
    let(high,low)=(LADDER_TOP/t1,1./longest);
    let count=((high/low).log10()*PER_DECADE).ceil().max(1.)as usize+1;
    let sigmas:Vec<f64>=(0..count).map(|i|high*(low/high).powf(i as f64/(count-1)as f64)).collect();
    let weights=eigen.mode_weights(initial)?;
    let mut poles:Vec<Pole>=sigmas.iter().map(|&s|Pole::Real(s)).collect();
    // (research) ATOM_POLES_ALL: every x0-weighted dressed centre frequency is
    // a candidate, the fast modes' too (no slowness filter |Re lambda| < s).
    // (research) ATOM_POLES_LATTICE: the lattice modes' frequencies by the same
    // rule, every weighted dressed centre frequency shifted by every lattice
    // mode's nu_q (the lattice's dressed modes are lambda_j - i nu_q).
    let lattice_modes=std::env::var("ATOM_POLES_LATTICE").is_ok();
    let every=lattice_modes||std::env::var("ATOM_POLES_ALL").is_ok();
    let shifts:Vec<f64>=if lattice_modes{nu.to_vec()}else{vec![0.]};
    for &s in &sigmas{
        let mut frequencies:Vec<f64>=eigen.lambda.iter().zip(&weights).filter(|(l,&w)|(every||l.re.abs()<s)&&w>=COMPLEX_WEIGHT*tolerance)
            .flat_map(|(l,_)|shifts.iter().map(move|&v|l.im-v)).filter(|&w|w>s).collect();
        frequencies.sort_by(f64::total_cmp);
        let mut chosen=vec![0.];
        for w in frequencies{if chosen.iter().all(|&v|(w-v).abs()>s){chosen.push(w);}}
        poles.extend(chosen.into_iter().skip(1).map(|w|Pole::Complex(C{re:s,im:w})));
    }
    Ok(poles)
}
