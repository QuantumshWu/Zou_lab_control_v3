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
//! The Krylov sequence starts from x0; a solve at a pole takes its last
//! vector as the right-hand side. Columns: w = (xi - A)^-1 v with v the last
//! vector of the Krylov sequence (orthonormalized within the sequence only,
//! so the continuation stays in the rational Krylov space of x0: a
//! continuation carrying translated copies or the steady vector would solve
//! for other poles). Every column is a
//! Hermitian-symmetric (J-real, (J x)_q = x_-q^dagger) field: a real pole's
//! solve stays in that real space. A direction below NOISE x the inner
//! tolerance of its column is noise and is not added (a nearly repeated pole
//! would add only roundoff); the same floor applies to its part outside the
//! whole basis, and only a Krylov vector that adds a direction is
//! translated. The basis and its images (`reduced::Galerkin`) are capped by
//! a memory budget (`Space::new`).
//!
//! Pole 0 is the residue at 0: the bordered solve (A + u tau0^T) w = u gives
//! Pi0 u = Pi0 x0 when the trace is the only left kernel vector; it is done
//! first and joins the basis only. When the bordered system is singular
//! (more conserved quantities) the residue is taken at 0+ on x0 (the rate
//! below which a mode is conserved over the run), where the kernel
//! dominates the column.
//!
//! Translations (A S_s = S_s (A - i nu_s) up to the fold's orders in nu and
//! the lattice boundary): S_s K(A, x0; xi) = K(A, S_s x0; xi - i nu_s), the
//! space of the translated initial state at the poles xi - i nu_s, with no
//! solve. The copies S_s v + S_-s v and i (S_s v - S_-s v) of every Krylov
//! vector join the basis (J-real like v).
//!
//! The ladder: real sigma log-spaced from the run's top down to 1 / T
//! (`pipeline`), PER_DECADE per decade, solved in a fixed bit-reversed cycle.
use super::*;
use super::poles::{Lattice,Pole};
use super::krylov::Vectors;

/// Relative residual of the shifted solves, and of the bordered steady
/// solve (the steady column's algebraic error).
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
/// Ladder points per decade. Measured against one: one per decade took
/// 22-29 solves instead of 40 on three, four and five lasers at 20 us but
/// stalled or ended at the tolerance's edge (four lasers 3.1e-3 against
/// 7.7e-4; five without the trap, 1000 samples and 10 us / 2000 partial).
const PER_DECADE:f64=2.;

#[derive(Clone,Debug)]
/// One solve of the space: its pole, inner steps (over its right-hand
/// sides), largest true relative residual, directions added, time (ms) and
/// lattice block operations (operator and preconditioner applications per
/// block).
pub struct Column{pub pole:Pole,pub steps:usize,pub residual:f64,pub added:usize,pub milliseconds:f64,pub blocks:usize}

pub struct Space{
    /// Orthonormal basis (real inner product).
    pub basis:Vec<Vec<C>>,
    krylov:Vec<Vec<C>>,
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
}
impl Space{
    /// The space of x0 = rho0 on the centre, at most `capacity` columns.
    pub fn new(lattice:&mut Lattice<'_>,initial:&[C],capacity:usize)->Result<Self,String>{
        let op=lattice.operator();let d=op.n*op.n;let mut x0=vec![ZERO;op.dimension()];
        x0[op.centre*d..(op.centre+1)*d].copy_from_slice(initial);
        let vectors=Vectors::new(op)?;
        if !(vectors.norm(&x0)>0.){return Err("Zero initial density".into());}
        // The trace carrier u = I / n on the centre (tr u = tr x0 = 1).
        let n=op.n;let mut u=vec![ZERO;op.dimension()];for i in 0..n{u[op.centre*d+i*n+i]=C{re:1./n as f64,im:0.};}
        let carrier=vectors.norm(&u);
        let mut space=Self{basis:vec![],krylov:vec![],translations:vec![],vectors,noise_floor:0.,capacity:capacity.max(2),initial:vec![carrier],columns:vec![]};
        space.append(lattice,u,0.)?;
        // x0's K part (zero when rho0 is I / n).
        let mut k0=x0.clone();project_traces(&mut k0,n);let rest=space.vectors.norm(&k0);
        space.add_krylov(lattice,vec![x0])?;
        if space.basis.len()>1{space.initial.push(rest);}
        Ok(space)
    }
    pub fn translations(&self)->Vec<Charge>{self.translations.iter().map(|t|t.0.clone()).collect()}
    /// Whether the basis has reached its capacity.
    pub fn full(&self)->bool{self.basis.len()>=self.capacity}
    /// The floor of a new direction (relative to its norm).
    fn floor(&self)->f64{self.noise_floor.max(APPEND_FLOOR)}
    /// Activates the translation along s: the copies of every Krylov vector.
    pub fn activate(&mut self,lattice:&Lattice<'_>,step:&Charge)->Result<usize,String>{
        let op=lattice.operator();
        let source=|sign:i32|->Vec<Option<usize>>{op.modes.iter().map(|q|{
            let shifted:Option<Charge>=q.iter().zip(step).map(|(&a,&b)|a.checked_sub(sign*b)).collect();
            shifted.and_then(|c|op.index.get(&c).copied())}).collect()};
        self.translations.push((step.clone(),source(1),source(-1)));
        let t=self.translations.len()-1;let mut added=0;
        let floor=self.floor();let n=op.n;
        for v in self.krylov.clone(){added+=self.append_block(n,self.translates(&v,t).into(),floor)?;}
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
    /// Appends a vector orthonormalized against the basis when its part
    /// outside the basis is above `floor` of its norm and the basis has
    /// room; the first column (the trace carrier) as it is, every other one
    /// projected into K first, and again after the orthogonalization (which
    /// leaves K only at roundoff).
    fn append(&mut self,lattice:&Lattice<'_>,w:Vec<C>,floor:f64)->Result<bool,String>{
        let n=lattice.operator().n;
        if self.basis.is_empty(){
            let size=self.vectors.norm(&w);if !(size>0.){return Ok(false);}
            self.basis.push(w.iter().map(|v|scaled(*v,1./size)).collect());return Ok(true);
        }
        self.append_projected(n,w,floor)
    }
    fn append_projected(&mut self,n:usize,w:Vec<C>,floor:f64)->Result<bool,String>{Ok(self.append_block(n,vec![w],floor)?>0)}
    /// Appends a block of vectors in order, each as `append_projected`
    /// would: block classical Gram-Schmidt against the basis (one Gram and
    /// one combination per pass, a GEMM per canonical block: one sweep of
    /// the basis for the whole block), the Kahan-Parlett test per vector (a
    /// second pass for the vectors whose first removed more than 1 -
    /// 1/sqrt(2) of the norm), then each against the block's vectors taken
    /// before it, the floor test and the normalization. Returns the
    /// directions added.
    fn append_block(&mut self,n:usize,block:Vec<Vec<C>>,floor:f64)->Result<usize,String>{self.append_groups(n,block.into_iter().map(|w|vec![w]).collect(),floor)}
    /// `append_block` over groups [v, its translated copies]: a group's
    /// copies are taken only when its first vector adds a direction.
    fn append_groups(&mut self,n:usize,groups:Vec<Vec<Vec<C>>>,floor:f64)->Result<usize,String>{
        let block:Vec<(Vec<C>,usize,bool)>=groups.into_iter().enumerate().flat_map(|(g,group)|group.into_iter().enumerate().map(move|(k,w)|(w,g,k==0))).collect();
        let sizes:Vec<f64>=block.iter().map(|w|self.vectors.norm(&w.0)).collect();
        let mut block:Vec<(Vec<C>,f64,usize,bool)>=block.into_iter().zip(sizes).filter(|(w,size)|*size>0.||w.2).map(|((w,g,lead),size)|(w,size,g,lead)).collect();
        if block.is_empty()||self.full(){return Ok(0);}
        for(w,..)in &mut block{project_traces(w,n);}
        let before:Vec<f64>=block.iter().map(|(w,..)|self.vectors.norm(w)).collect();
        let mut pending:Vec<usize>=(0..block.len()).collect();
        for pass in 0..2{
            if pending.is_empty()||self.basis.is_empty(){break;}
            let refs:Vec<&[C]>=self.basis.iter().map(|v|v.as_slice()).collect();
            let rows:Vec<&[C]>=pending.iter().map(|&i|block[i].0.as_slice()).collect();
            let r=refs.len();let h=self.vectors.gram(&rows,&refs)?;drop(rows);
            let coefficients:Vec<Vec<f64>>=(0..pending.len()).map(|k|h[k*r..(k+1)*r].to_vec()).collect();
            let parts=self.vectors.combinations(&refs,&coefficients)?;drop(refs);
            for(&i,part)in pending.iter().zip(&parts){for(t,x)in block[i].0.iter_mut().zip(part){t.re-=x.re;t.im-=x.im;}}
            if pass==0{pending.retain(|&i|!(self.vectors.norm(&block[i].0)>std::f64::consts::FRAC_1_SQRT_2*before[i]));}
        }
        let mut added=0;let start=self.basis.len();let mut skipped=None;
        for(mut w,size,group,lead)in block{
            if self.full(){break;}
            if skipped==Some(group){continue;}
            if lead&&!(size>0.){skipped=Some(group);continue;}
            // Against the block's vectors appended before it (twice).
            for _ in 0..2{let refs:Vec<&[C]>=self.basis[start..].iter().map(|v|v.as_slice()).collect();if refs.is_empty(){break;}
                let h=self.vectors.dots(&refs,&w);self.vectors.subtract(&refs,&h,&mut w)?;}
            project_traces(&mut w,n);
            let rest=self.vectors.norm(&w);if !(rest>floor*size){if lead{skipped=Some(group);}continue;}
            for v in w.iter_mut(){*v=scaled(*v,1./rest);}
            self.basis.push(w);added+=1;
        }
        Ok(added)
    }
    /// Krylov vectors: each orthonormalized within the sequence (dropped
    /// below the noise floor), then appended to the basis with its
    /// translated copies, all as one block (`append_groups`).
    fn add_krylov(&mut self,lattice:&Lattice<'_>,vectors:Vec<Vec<C>>)->Result<usize,String>{
        let mut accepted=Vec::new();
        for p in vectors{
            let size=self.vectors.norm(&p);let mut v=p;
            for _ in 0..2{let refs:Vec<&[C]>=self.krylov.iter().map(|x|x.as_slice()).collect();let h=self.vectors.dots(&refs,&v);self.vectors.subtract(&refs,&h,&mut v)?;}
            let rest=self.vectors.norm(&v);if !(rest>self.noise_floor*size)||!(rest>0.){continue;}
            for x in v.iter_mut(){*x=scaled(*x,1./rest);}
            self.krylov.push(v.clone());accepted.push(v);
        }
        // The carrier is the basis' first column (`new`): every Krylov vector
        // is projected into K.
        let groups:Vec<Vec<Vec<C>>>=accepted.into_iter().map(|v|{let copies:Vec<Vec<C>>=(0..self.translations.len()).flat_map(|t|self.translates(&v,t)).collect();
            std::iter::once(v).chain(copies).collect()}).collect();
        let(floor,n)=(self.floor(),lattice.operator().n);
        self.append_groups(n,groups,floor)
    }
    /// Fields joining the basis as they are (J-real, no translation): one
    /// block; returns the directions added.
    pub fn add(&mut self,lattice:&Lattice<'_>,fields:Vec<Vec<C>>)->Result<usize,String>{
        let(floor,n)=(self.floor(),lattice.operator().n);self.append_block(n,fields,floor)
    }
    /// One column at a pole; returns the directions it added.
    /// `kernel`: the rate below which a mode is conserved over the run, the
    /// 0+ pole when the bordered system is singular.
    pub fn solve(&mut self,lattice:&mut Lattice<'_>,pole:Pole,kernel:f64)->Result<usize,String>{
        let clock=Instant::now();let length=self.basis[0].len();let before=lattice.work().block_operations();
        let inner=INNER_TOLERANCE;
        let(added,steps,residual)=match pole{
            Pole::Zero=>{
                let op=lattice.operator();let n=op.n;let c0=op.centre*n*n;let mut u=vec![ZERO;length];for i in 0..n{u[c0+i*n+i]=C{re:1./n as f64,im:0.};}
                let bordered=lattice.solve(Pole::Zero,&u,STEADY_TOLERANCE);
                let(w,steps,residual)=match bordered{Ok((w,steps,residual))if residual<=BORDERED_SINGULAR=>(w,steps,residual),
                    _=>{let x0:Vec<C>=self.krylov[0].iter().map(|v|C{re:-v.re,im:-v.im}).collect();lattice.solve(Pole::Real(kernel),&x0,inner)?}};
                let floor=self.floor();(usize::from(self.append(lattice,w,floor)?),steps,residual)
            }
            Pole::Real(_)=>{
                let rhs:Vec<C>=self.krylov.last().ok_or("A space without its start")?.iter().map(|v|C{re:-v.re,im:-v.im}).collect();
                let(x,steps,residual)=lattice.solve(pole,&rhs,inner)?;
                (self.add_krylov(lattice,vec![x])?,steps,residual)
            }
        };
        self.noise_floor=NOISE*inner;
        self.columns.push(Column{pole,steps,residual,added,milliseconds:clock.elapsed().as_secs_f64()*1e3,blocks:lattice.work().block_operations()-before});
        Ok(added)
    }
    pub fn solves(&self)->usize{self.columns.len()}
}
/// Whether a column's solve reached the noise floor of its directions
/// (NOISE x INNER_TOLERANCE): below it the column is the pole's.
pub fn resolved(column:&Column)->bool{column.residual<=NOISE*INNER_TOLERANCE}

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

/// The ladder of a run: real poles from `high` down to `low`, PER_DECADE per
/// decade.
pub fn ladder(high:f64,low:f64)->Vec<Pole>{
    let high=high.max(low);let intervals=((high/low).log10()*PER_DECADE).ceil()as usize;
    (0..=intervals).map(|i|Pole::Real(if intervals==0{high}else{high*(low/high).powf(i as f64/intervals as f64)})).collect()
}
