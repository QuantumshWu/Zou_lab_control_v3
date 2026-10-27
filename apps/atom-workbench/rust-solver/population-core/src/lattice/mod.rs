//! The folded Fourier lattice: the population core on the Fourier envelope of
//! the request's own frame (its per-F rotating frames). Every near-resonant
//! charge step is an explicit lattice coupling; every far step (the tweezer
//! and every counter-rotating harmonic) is folded statically into
//! K_t^(0) + i nu K_t^(1), built with the exact (L0 - i nu_s)^-1, so far
//! charges never enter the lattice; what the fold leaves out is a charged
//! bound. Decisions are made by contribution size; exact zeros are exploited
//! for speed only. The population tolerance is the fixed
//! `ledger::POPULATION_TOLERANCE`.
//!
//! Pipeline, by module:
//! - `generator`: the frame's Fourier components L_t;
//! - `classification`: explicit / fold for every nonzero charge step, and the
//!   far + far pairs that land on near charges;
//! - `fold`: the structured (L0 - z)^-1, the folded couplings K_t and the
//!   third-order fold source;
//! - `operator`: the lattice operator on a set of near charges;
//! - `preconditioner`: the dressed block preconditioner (L0' - i nu_q)^-1;
//! - `krylov`: the primal solve of the trace-row system;
//! - `estimate`: error equations in the primal's Krylov space (truncation,
//!   fold) and the shell correction;
//! - `steady`: the steady pipeline (order, estimate, acceptance);
//! - `slow`: the slow eigenpairs of the lattice operator (conserved
//!   quantities, the slowest relaxation, the long-time machinery);
//! - `ledger`: the tolerance, the error metric and every remainder bound.
//!
//! Validation only (tests and the `validation` feature): `dense_dressed`, the
//! exact dense block inverse the structured preconditioner is measured
//! against.
//!
//! Layout: an n x n block is row-major (entry (i, j) at i*n+j); a field on N
//! modes is N consecutive blocks; a superoperator as a d x d matrix (d = n^2)
//! is row-major with row (i, j) and column (k, l).
use std::collections::{BTreeMap,BTreeSet};
use std::sync::Arc;
use std::time::Instant;
use crate::{C,Lapack};
use crate::preparation::{Charge,ModelView,PhaseBasis,Source};

pub mod generator;
pub mod classification;
pub mod fold;
pub mod operator;
pub mod preconditioner;
pub mod krylov;
pub mod estimate;
pub mod steady;
pub mod slow;
pub mod eigenbasis;
pub mod finite;
pub mod ledger;
#[cfg(any(test,feature="validation"))]
pub mod dense_dressed;
pub use generator::{Component,Generator,Sparse,Superop};

pub(crate) const ZERO:C=C{re:0.,im:0.};
pub(crate) const ONE:C=C{re:1.,im:0.};
#[inline]pub(crate) fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
#[inline]pub(crate) fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
#[inline]pub(crate) fn mul_add(acc:&mut C,a:C,b:C){acc.re+=a.re*b.re-a.im*b.im;acc.im+=a.re*b.im+a.im*b.re;}
#[inline]pub(crate) fn conj(a:C)->C{C{re:a.re,im:-a.im}}
#[inline]pub(crate) fn scaled(a:C,s:f64)->C{C{re:a.re*s,im:a.im*s}}
#[inline]pub(crate) fn magnitude(a:C)->f64{a.re.hypot(a.im)}
#[inline]pub(crate) fn nonzero(a:C)->bool{a.re!=0.||a.im!=0.}
pub(crate) fn grow(buffer:&mut Vec<C>,length:usize){if buffer.len()<length{buffer.resize(length,ZERO);}}
/// A Hermitian-symmetric field (x_-q = x_q^dagger) from its canonical modes:
/// the mirrored modes copied, the self-mirrored centre replaced by its
/// Hermitian part. The anti-Hermitian roundoff of the centre is otherwise
/// amplified by the slow modes and leaves the real space the solver works in
/// (the true residual then stalls near 1e-7).
pub(crate) fn mirror_field(x:&mut[C],n:usize,canonical:&[bool],mirror:&[usize]){
    let d=n*n;
    for q in 0..canonical.len(){let p=mirror[q];
        if p==q{let b=&mut x[q*d..(q+1)*d];
            for i in 0..n{b[i*n+i].im=0.;for j in 0..i{let(u,v)=(b[i*n+j],b[j*n+i]);let m=C{re:0.5*(u.re+v.re),im:0.5*(u.im-v.im)};b[i*n+j]=m;b[j*n+i]=conj(m);}}}
        else if !canonical[q]{
            if !canonical[p]{x[q*d..(q+1)*d].fill(ZERO);continue;}
            for i in 0..n{for j in 0..n{x[q*d+i*n+j]=conj(x[p*d+j*n+i]);}}}
    }
}
pub(crate) fn charge_plus(a:&[i32],b:&[i32])->Result<Charge,String>{
    a.iter().zip(b).map(|(&x,&y)|x.checked_add(y).ok_or_else(||"Lattice charge overflow".into())).collect()
}
pub(crate) fn charge_minus(a:&[i32],b:&[i32])->Result<Charge,String>{
    a.iter().zip(b).map(|(&x,&y)|x.checked_sub(y).ok_or_else(||"Lattice charge overflow".into())).collect()
}
pub(crate) fn opposite(a:&[i32])->Charge{a.iter().map(|&k|-k).collect()}

/// Request-local lattice preparation: generator, classification and fold. It
/// depends on the model only, never on a lattice mode set.
pub struct LatticePrepared{
    /// The model in the request's frame.
    pub source:Source,
    pub generator:Generator,
    pub plan:classification::HarmonicPlan,
    pub fold:fold::FoldLibrary,
    pub lapack:Arc<Lapack>,
    /// Milliseconds: source, generator and classification, fold.
    pub timing:[f64;3],
}
impl LatticePrepared{
    pub fn prepare(model:&ModelView<'_>,phases:PhaseBasis)->Result<Self,String>{
        let lapack=Arc::new(Lapack::load()?);
        let clock=Instant::now();
        let source=Source::prepare(model,phases)?;
        let source_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
        let generator=Generator::new(&source)?;
        let plan=classification::HarmonicPlan::new(&generator)?;
        let plan_ms=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
        let fold=fold::FoldLibrary::build(&generator,&plan,&lapack)?;
        let fold_ms=clock.elapsed().as_secs_f64()*1e3;
        Ok(Self{source,generator,plan,fold,lapack,timing:[source_ms,plan_ms,fold_ms]})
    }
    /// The near ball of order k: every charge reachable from 0 with at most k
    /// structural steps (explicit near steps above the structural floor and
    /// the folded far + far targets t != 0), sorted by (|q|_1, q). Closed
    /// under q -> -q because the step set is.
    pub fn modes(&self,order:usize)->Result<Vec<Charge>,String>{
        let steps=self.plan.structural_steps();
        let mut all=BTreeSet::from([self.generator.zero.clone()]);let mut frontier=vec![self.generator.zero.clone()];
        for _ in 0..order{
            let mut next=Vec::new();
            for q in &frontier{for s in &steps{let t=charge_plus(q,s)?;if all.insert(t.clone()){next.push(t);}}}
            frontier=next;
        }
        let mut modes:Vec<Charge>=all.into_iter().collect();
        modes.sort_by_key(|q|(q.iter().map(|&k|k.unsigned_abs()as u64).sum::<u64>(),q.clone()));
        Ok(modes)
    }
    pub fn nu(&self,q:&[i32])->Result<f64,String>{let f=self.source.phases.frequency(q)?;Ok(f.hi+f.lo)}
    /// The folded centre generator L0' = L0 + K_0^(0) as a row-major d x d
    /// matrix.
    pub fn centre_generator(&self)->Vec<C>{
        let n=self.generator.n;let d=n*n;
        let mut l=vec![ZERO;d*d];self.generator.zero_component().op.dense_into(n,ONE,&mut l);
        if let Some(t)=self.fold.targets.first().filter(|t|t.charge==self.generator.zero){t.dense_into(n,0,ONE,&mut l);}
        l
    }
    /// Test hook for the static-shift consistency check (the all-order static
    /// shifts are compared with this second-order fold): the Hermitian
    /// background Hamiltonian H_2 of the diagonal fold at q = 0. The
    /// absorbable part of K_0^(0) is A X + X B with A = -i H_2 - G/2 and
    /// B = i H_2 - G/2 up to a multiple of the identity, so H_2 is the
    /// Hermitian part of i (A - B)/2 with its trace removed (row-major n x n,
    /// rad/us). Also returned: |K - (A X + X B)|_F / |K|_F, the share of the
    /// fold that is not of that form (scattering and higher orders in 1/nu).
    pub fn folded_background_hamiltonian(&self)->Option<(Vec<C>,f64)>{
        let k=self.fold.zero_dense.as_ref()?;let n=self.generator.n;let d=n*n;
        let(a,b)=fold::absorbable(k,n);
        let m:Vec<C>=(0..d).map(|i|{let x=C{re:a[i].re-b[i].re,im:a[i].im-b[i].im};C{re:-0.5*x.im,im:0.5*x.re}}).collect();
        let mut h:Vec<C>=(0..d).map(|i|{let(r,c)=(i/n,i%n);let y=conj(m[c*n+r]);C{re:0.5*(m[i].re+y.re),im:0.5*(m[i].im+y.im)}}).collect();
        let trace=(0..n).map(|i|h[i*n+i].re).sum::<f64>()/n as f64;for i in 0..n{h[i*n+i].re-=trace;}
        let op=Superop{left:Sparse::from_dense(&a,n),right:Sparse::from_dense(&b,n),sandwich:vec![]};
        let mut rest=k.clone();op.dense_into(n,C{re:-1.,im:0.},&mut rest);
        let norm=|v:&[C]|v.iter().fold(0_f64,|s,z|s.hypot(magnitude(*z)));
        Some((h,norm(&rest)/norm(k).max(f64::MIN_POSITIVE)))
    }
}

#[cfg(test)]
pub(crate) mod tests{
    use super::*;
    use crate::preparation::{DriveView,MatrixView};
    /// A 4-level fixture (ground 0, 1; excited 2, 3) with a near-resonant
    /// static coupling in H0, one near drive (beat 3 rad/us), two far drives
    /// (2000 and 2003 rad/us, so far + far pairs land on the near charge too)
    /// and a non-secular decay channel.
    pub(crate) struct Fixture{pub h:Vec<f64>,pub zero:Vec<f64>,pub near:Vec<f64>,pub far:Vec<f64>,pub far2:Vec<f64>,pub c:Vec<f64>}
    pub(crate) fn fixture()->Fixture{
        let n=4;let mut h=vec![0.;16];for(i,v)in[0.,0.7,1.3,-0.4].iter().enumerate(){h[i*n+i]=*v;}
        for(i,j,v)in[(0,2,0.6),(1,3,0.25)]{h[i*n+j]=v;h[j*n+i]=v;}
        let mut near=vec![0.;16];for(i,j,v)in[(1,2,0.4),(0,3,0.15)]{near[i*n+j]=v;near[j*n+i]=v;}
        let mut far=vec![0.;16];for(i,j,v)in[(0,2,40.),(1,3,25.),(0,3,10.)]{far[i*n+j]=v;far[j*n+i]=v;}
        let mut far2=vec![0.;16];for(i,j,v)in[(0,2,30.),(1,2,12.)]{far2[i*n+j]=v;far2[j*n+i]=v;}
        let mut c=vec![0.;16];c[2]=0.8;c[3]=0.3;c[n+2]=0.2;c[n+3]=0.9;
        Fixture{h,zero:vec![0.;16],near,far,far2,c}
    }
    pub(crate) fn prepared(f:&Fixture)->LatticePrepared{
        let h=MatrixView{real:&f.h,imag:&f.zero};let channels=[MatrixView{real:&f.c,imag:&f.zero}];
        let drives=[DriveView{omega:3.,cosine:MatrixView{real:&f.near,imag:&f.zero},sine:MatrixView{real:&f.zero,imag:&f.zero}},
            DriveView{omega:2000.,cosine:MatrixView{real:&f.far,imag:&f.zero},sine:MatrixView{real:&f.zero,imag:&f.zero}},
            DriveView{omega:2003.,cosine:MatrixView{real:&f.far2,imag:&f.zero},sine:MatrixView{real:&f.zero,imag:&f.zero}}];
        let model=ModelView{dimension:4,hamiltonian:h,collapse:&channels,drives:&drives,ground:&[true,true,false,false]};
        let phases=PhaseBasis::from_declared(&[3.,2000.],&[vec![1,0],vec![0,1],vec![1,1]]).unwrap();
        LatticePrepared::prepare(&model,phases).unwrap()
    }
    /// The dense lattice generator on any charge set with every component
    /// explicit (nothing folded), row-major, and the centre index.
    pub(crate) fn explicit_matrix(p:&LatticePrepared,charges:&[Charge])->(Vec<C>,usize){
        let n=p.generator.n;let d=n*n;let size=charges.len()*d;let mut out=vec![ZERO;size*size];let mut block=vec![ZERO;d*d];
        for(qi,q)in charges.iter().enumerate(){for(pi,source)in charges.iter().enumerate(){
            let t=charge_minus(q,source).unwrap();let Some(c)=p.generator.components.get(&t)else{continue};
            block.fill(ZERO);c.op.dense_into(n,ONE,&mut block);
            if qi==pi{let nu=p.nu(q).unwrap();for r in 0..d{block[r*d+r].im-=nu;}}
            for r in 0..d{for k in 0..d{add(&mut out[(qi*d+r)*size+pi*d+k],block[r*d+k]);}}
        }}
        (out,charges.iter().position(|q|q.iter().all(|&k|k==0)).unwrap())
    }
    /// Populations of the null vector of a dense lattice matrix (trace row
    /// on the centre), by LU.
    pub(crate) fn dense_steady(p:&LatticePrepared,matrix:&[C],centre:usize,n:usize)->Vec<f64>{
        let d=n*n;let size=(matrix.len() as f64).sqrt().round() as usize;let watch:Vec<usize>=(0..n).map(|i|centre*d+i*n+i).collect();
        let mut col=vec![ZERO;size*size];for r in 0..size{for c in 0..size{col[r+size*c]=matrix[r*size+c];}}
        for &r in &watch{for &c in &watch{col[r+size*c].re+=1./n as f64;}}
        let(lu,piv)=p.lapack.complex_factor(&col,size).unwrap();let mut rhs=vec![ZERO;size];for &k in &watch{rhs[k].re=1./n as f64;}
        let x=p.lapack.complex_solve_matrix_factored(&lu,&piv,&rhs,size).unwrap();
        let trace:f64=watch.iter().map(|&k|x[k].re).sum();watch.iter().map(|&k|x[k].re/trace).collect()
    }
    #[test]
    fn lattice_folded_background_hamiltonian_is_the_light_shift(){
        // A two-level atom (g, e at 1 rad/us) driven by a rotating far coupling
        // v (|e><g| e^{i nu t} + h.c.), nu = 2000, no decay: the diagonal fold is
        // the commutator with H_2, whose splitting is the light shift
        // 2 v^2 / nu up to O(1/nu) relative.
        let(v,nu)=(5.,2000.);let zero=vec![0.;4];let h=vec![0.,0.,0.,1.];
        let(cos,sin)=(vec![0.,v,v,0.],vec![0.,-v,v,0.]);
        let drives=[DriveView{omega:nu,cosine:MatrixView{real:&cos,imag:&zero},sine:MatrixView{real:&zero,imag:&sin}}];
        let model=ModelView{dimension:2,hamiltonian:MatrixView{real:&h,imag:&zero},collapse:&[],drives:&drives,ground:&[true,false]};
        let p=LatticePrepared::prepare(&model,PhaseBasis::from_declared(&[nu],&[vec![1]]).unwrap()).unwrap();
        let(h2,rest)=p.folded_background_hamiltonian().unwrap();
        assert!((h2[1].re-h2[2].re).abs()<1e-15&&(h2[1].im+h2[2].im).abs()<1e-15&&(h2[0].re+h2[3].re).abs()<1e-15);
        let splitting=h2[3].re-h2[0].re;let expected=2.*v*v/nu;
        assert!((splitting.abs()/expected-1.).abs()<2e-3&&magnitude(h2[1])<1e-9*expected,"splitting {splitting:e}, 2 v^2/nu {expected:e}");
        assert!(rest<1e-2,"rest share {rest:e}");
    }
}
