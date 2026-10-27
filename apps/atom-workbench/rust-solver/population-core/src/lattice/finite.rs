//! Finite-time populations p(t) on the folded lattice: x' = A x,
//! x(0) = delta_q0 rho0, read out as the local time average (the near
//! charges' diagonals with their phases e^{i nu_q t}; far charges never
//! appear).
//!
//! Rational Krylov: v_j+1 is (sigma_j - A)^-1 v_j orthonormalized, with the
//! pole POLE (1/us) and, for horizons above POLE_HORIZON, POLE_SCALE / T
//! alternating with it. Every shifted solve is right-preconditioned GMRES
//! with the dressed blocks shifted by the pole, to INNER_RTOL.
//!
//! Translation augmentation: the lattice is redundant (every Floquet mode mu
//! has copies mu - i nu_s), x(0) is not in the span of the centred copies,
//! and the difference evolves along the slow modes' translated copies, which
//! a real pole cannot reach (the 2e-3 stall of b1-finite.md). For every
//! explicit near step s with |V_s| / |nu_s| > TRANSLATION_MIXING the
//! charge-shifted copies of the basis join the Galerkin space (no new shifted
//! solves, one image each): in real coordinates S_s v + S_-s v and
//! i (S_s v - S_-s v), Hermitian-symmetric like v.
//!
//! Reduced model: Galerkin on the span Z of the basis and its copies,
//! orthonormalized through the Gram matrix (dependent directions dropped);
//! every mode's trace evolves exactly (tr x_q' = -i nu_q tr x_q: the centre
//! trace is conserved, the others stay zero), and reduced eigenvalues that
//! grow over the horizon are reflected.
//!
//! Error ledger, accepted when KAPPA (basis + shell + fold) + projection
//! <= POPULATION_TOLERANCE:
//! - basis: the Galerkin residual r(t) = (A W - W G) u(t), exact (it includes
//!   the inexact shifted solves) and nearly rank one; its leading singular
//!   components (a randomized range finder) are propagated through every
//!   mode's dressed block (`eigenbasis`);
//! - shell: the next order's slaved source B y, y = -P_s^-1 C x(t), on a
//!   low-rank compression of the trajectory, propagated the same way, plus
//!   the shell's own populations;
//! - fold: the third-order fold source on the same compression, propagated,
//!   with the other fold bounds scaled by its response per unit bound (as for
//!   the steady state);
//! - projection: the change of the rows onto the probability simplex.
//!
//! Long times (`long_time`): one Krylov propagation to a short reach, then
//! x(t) = x_s + sum_j (slow mode j)(t) for t >= t_s, the steady field and
//! the slow modes those of the steady solution (`slow`), the coefficients
//! fitted to x(t_s) - x_s (the propagated field carried to the steady
//! solution's modes). The
//! switch time t_s is the earliest candidate whose part outside the slow
//! modes plus the drift of the fit (from t_s / 2) is below FAST_SHARE of the
//! tolerance; the error at t is the Krylov estimate + the steady estimate +
//! KAPPA (that part, grown by any excluded artifact, + the slow modes'
//! eigen-residuals), and the propagation is accepted at the share of the
//! tolerance the other terms leave.
use super::*;
use super::krylov::{self,KrylovOptions,System,Vectors};
use super::operator::LatticeOperator;
use super::preconditioner::{StructuredBatch,StructuredDressed};
use super::eigenbasis::BlockEigenbasis;
use super::estimate::Shell;
use super::ledger::{self,KAPPA,POPULATION_TOLERANCE,PopulationCharges,RemainderLedger};
use super::steady::{MAXIMUM_ORDER,SteadySolution};
use super::slow::SlowMode;
use std::sync::OnceLock;

const POLE:f64=20.;
const POLE_HORIZON:f64=40.;
const POLE_SCALE:f64=200.;
/// The first near order (order 1 suffices for three lasers + tweezer, 1.9
/// s, but four and five lasers, old and research, need order 2, and finding
/// that out costs about 1 s: the old models are the common case).
const START_ORDER:usize=2;
/// Relative residual of the shifted solves; 1e-3 left 2-9e-3 on three
/// lasers + tweezer (b1-finite.md section 6.2).
const INNER_RTOL:f64=1e-6;
const TRANSLATION_MIXING:f64=3e-3;
/// At most this many translation pairs, the most strongly mixing: with all
/// three on five lasers the copies made the space larger and less accurate
/// (old five: 80 columns and 3.2e-3 against 60 and 2.2e-3 with two); with one
/// the estimate missed the stall (0.57 of the actual).
const MAXIMUM_TRANSLATIONS:usize=2;
/// Basis columns at the first checkpoint, added per checkpoint, at most.
const FIRST_COLUMNS:usize=40;
const MORE_COLUMNS:usize=20;
const MAXIMUM_COLUMNS:usize=200;
/// A generator whose part outside the span of the generators before it is
/// below DEPENDENT of its norm (squared) is dropped: the translated copies
/// grow nearly dependent (Gram condition 4e7 at 120 columns on old five
/// lasers, where keeping them made the estimate erratic).
const DEPENDENT:f64=1e-6;
/// Randomized range of the residual per Galerkin generator of a basis
/// vector (plus 8 oversampling), and its components kept down to this
/// fraction of the largest (the estimate scaled by the dropped norm).
const RESIDUAL_RANGE:usize=4;
const RESIDUAL_COMPONENT:f64=1e-3;
/// Trajectory components kept for the shell and fold sources (their
/// estimates are scaled by the norm of the dropped rest).
const TRAJECTORY_COMPONENT:f64=1e-4;
const TRAJECTORY_RANGE:usize=12;
/// Growth Re(beta) T above which a reduced eigenvalue is reflected.
const GROWTH:f64=1e-9;
/// Trace functionals below this fraction of the largest singular value are
/// left unconstrained.
const TRACE_RANK:f64=1e-7;

#[derive(Clone,Debug,Default)]
pub struct FiniteTiming{
    pub eigenbasis_ms:f64,pub dressed_ms:f64,pub columns_ms:f64,pub images_ms:f64,
    pub gram_ms:f64,pub reduced_ms:f64,pub basis_ms:f64,pub shell_ms:f64,pub fold_ms:f64,
    /// The shifted solves' operator, preconditioner and orthogonalization time.
    pub inner:krylov::KrylovTiming,
}
#[derive(Clone,Debug)]
pub struct Checkpoint{
    pub order:usize,pub columns:usize,pub dimension:usize,pub reflected:usize,
    /// Largest / smallest kept Gram eigenvalue, and max Re(mu) T after the
    /// reflection.
    pub gram_condition:f64,pub growth:f64,
    pub basis:f64,pub shell:Option<f64>,pub fold:Option<f64>,pub projection:f64,
    /// KAPPA (basis + shell + fold) + projection, once all are known.
    pub total:Option<f64>,
    pub action:&'static str,pub milliseconds:f64,
}
pub struct FiniteSolution{
    /// Rows projected onto the probability simplex (row 0 is rho0's).
    pub populations:Vec<Vec<f64>>,
    pub estimated_error:f64,pub projection_charge:f64,
    pub basis:f64,pub shell:f64,pub fold:f64,
    pub order:usize,pub modes:usize,pub columns:usize,pub dimension:usize,pub translations:usize,
    pub inner_steps:usize,pub applications:usize,pub preconditions:usize,
    pub eigenbasis_condition:f64,pub ritz_condition:f64,
    pub checkpoints:Vec<Checkpoint>,
    pub timing:FiniteTiming,
    /// The fields at the kept time indices, on the modes of `order`.
    pub fields:Vec<Vec<C>>,
}

/// Real symmetric eigendecomposition (ascending; column-major vectors).
fn symmetric_eigen(lapack:&Lapack,a:&[f64],m:usize)->Result<(Vec<f64>,Vec<f64>),String>{
    let complex:Vec<C>=a.iter().map(|&x|C{re:x,im:0.}).collect();
    let(values,vectors)=lapack.hermitian_eigen(&complex,m)?;
    // A real symmetric input has real eigenvectors up to a phase per column.
    let mut real=vec![0.;m*m];
    for j in 0..m{let col=&vectors[j*m..(j+1)*m];
        let pivot=col.iter().copied().max_by(|a,b|magnitude(*a).total_cmp(&magnitude(*b))).unwrap_or(ONE);
        let s=magnitude(pivot).max(f64::MIN_POSITIVE);let phase=C{re:pivot.re/s,im:-pivot.im/s};
        for i in 0..m{real[i+m*j]=mul(col[i],phase).re;}}
    Ok((values,real))
}
/// Row-major (r x c) = (r x k) (k x c), all row-major: the column-major
/// product C^T = B^T A^T.
fn product(lapack:&Lapack,a:&[f64],b:&[f64],r:usize,k:usize,c:usize)->Vec<f64>{
    let mut out=vec![0.;r*c];
    if r*k*c>0{lapack.gemm(false,false,c,r,k,1.,b,c,a,k,0.,&mut out,c).expect("product dimensions are consistent");}
    out
}
fn transpose_rows(a:&[f64],r:usize,c:usize)->Vec<f64>{let mut t=vec![0.;r*c];for i in 0..r{for j in 0..c{t[j*r+i]=a[i*c+j];}}t}

/// Reflects reduced eigenvalues with Re(beta) T > GROWTH to -Re(beta)
/// (row-major real generator): G' = G - 2 V_u diag(Re beta_u) (V^-1)_u over
/// the growing columns, kept only when G' V = V L' holds to roundoff.
fn reflect_growing(lapack:&Lapack,g:&mut[f64],m:usize,horizon:f64)->Result<usize,String>{
    let column=transpose_rows(g,m,m);
    let(re,im,vectors)=lapack.eigen(&column,m)?;
    let growing:Vec<usize>=(0..m).filter(|&j|re[j]*horizon>GROWTH).collect();
    if growing.is_empty()||vectors.iter().any(|x|!x.is_finite()){return Ok(0);}
    let(lu,pivots)=lapack.factor(&vectors,m)?;
    let mut identity=vec![0.;m*m];for j in 0..m{identity[j+j*m]=1.;}
    let inverse=lapack.solve_matrix_factored(&lu,&pivots,&identity,m)?;
    let mut reflected=g.to_vec();
    for &u in &growing{let a=re[u];for c in 0..m{let row=inverse[u+c*m];if row==0.{continue;}
        for r in 0..m{reflected[r*m+c]-=2.*a*vectors[r+u*m]*row;}}}
    if reflected.iter().any(|x|!x.is_finite()){return Ok(0);}
    // Check G' V = V L'.
    let scale=g.iter().fold(0_f64,|s,x|s.max(x.abs())).max(f64::MIN_POSITIVE);
    let mut defect=0_f64;let mut j=0;
    // G' V, column-major.
    let mut gv=vec![0.;m*m];lapack.gemm(false,false,m,m,m,1.,&transpose_rows(&reflected,m,m),m,&vectors,m,0.,&mut gv,m)?;
    let apply=|col:usize|->Vec<f64>{gv[col*m..(col+1)*m].to_vec()};
    while j<m{
        let a=if re[j]*horizon>GROWTH{-re[j]}else{re[j]};
        let norm=|c:usize|vectors[c*m..(c+1)*m].iter().map(|x|x*x).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        if im[j]!=0.&&j+1<m{let b=im[j];let(p,q)=(apply(j),apply(j+1));let s=norm(j).hypot(norm(j+1));
            for r in 0..m{let(vr,vi)=(vectors[r+j*m],vectors[r+(j+1)*m]);
                defect=defect.max((p[r]-(a*vr-b*vi)).abs()/s).max((q[r]-(b*vr+a*vi)).abs()/s);}j+=2;}
        else{let p=apply(j);for r in 0..m{defect=defect.max((p[r]-a*vectors[r+j*m]).abs()/norm(j));}j+=1;}
    }
    if defect>1e-8*scale{return Ok(0);}
    g.copy_from_slice(&reflected);Ok(growing.len())
}

/// The Galerkin span: basis vectors with their translated copies (the
/// generators), their images, and the Gram matrices.
struct Space{
    /// The rational Krylov basis (indices of the generators).
    basis:Vec<usize>,
    z:Vec<Vec<C>>,az:Vec<Vec<C>>,
    /// Columns covered by the Gram matrices.
    covered:usize,
    /// K = Z^T Z and H = Z^T A Z, row-major over `covered` columns.
    k:Vec<f64>,h:Vec<f64>,
    /// Per generator: the trace functionals and the canonical populations.
    traces:Vec<Vec<f64>>,readout:Vec<Vec<C>>,
}
/// The reduced model of a checkpoint.
struct Reduced{
    /// Generators -> orthonormal coordinates (dimension x r, row-major).
    t:Vec<f64>,r:usize,
    g:Vec<f64>,reflected:usize,gram_condition:f64,growth:f64,
    /// Trajectory u(t_i), row-major [time][r].
    u:Vec<Vec<f64>>,
    /// Ritz values, eigenvectors (row-major r x r, complex) and the
    /// coefficients of u0 on them.
    mu:Vec<C>,s:Vec<C>,a:Vec<C>,condition:f64,
    /// LU of the real eigenvector matrix and the pair layout, for
    /// coordinates on the eigenvectors.
    lu:Vec<f64>,pivots:Vec<i32>,pair:Vec<u8>,
}
impl Reduced{
    /// psi(t) = b^T u(t) as an exponential sum: the coefficients of e^{mu_k t}.
    fn exponential(&self,b:&[f64])->Vec<C>{
        let r=self.r;(0..r).map(|k|{let mut s=ZERO;for i in 0..r{let v=self.s[i*r+k];s.re+=b[i]*v.re;s.im+=b[i]*v.im;}mul(s,self.a[k])}).collect()
    }
    /// Coordinates of a real reduced vector on the (complex) eigenvectors.
    fn ritz(&self,lapack:&Lapack,w:&[f64])->Result<Vec<C>,String>{
        let r=self.r;let y=lapack.solve_matrix_factored(&self.lu,&self.pivots,w,r)?;let mut c=vec![ZERO;r];let mut j=0;
        while j<r{if self.pair[j]==1{c[j]=C{re:0.5*y[j],im:-0.5*y[j+1]};c[j+1]=C{re:0.5*y[j],im:0.5*y[j+1]};j+=2;}else{c[j]=C{re:y[j],im:0.};j+=1;}}
        Ok(c)
    }
}

struct Lattice<'a,'p>{
    prepared:&'a LatticePrepared,
    systems:Vec<System<'a,'p>>,poles:Vec<f64>,
    vectors:Vectors,
    /// Per translation pair: the source mode of q - s and of q + s.
    translations:Vec<(Vec<Option<usize>>,Vec<Option<usize>>)>,
    /// Canonical modes with their weights and frequencies.
    canonical:Vec<(usize,f64,f64)>,
    centre:usize,n:usize,d:usize,
    inner_steps:usize,applications:usize,
    timing:FiniteTiming,
}
impl Lattice<'_,'_>{
    fn op(&self)->&LatticeOperator<'_>{&self.systems[0].op}
    fn length(&self)->usize{self.op().dimension()}
    fn image(&mut self,x:&[C])->Result<Vec<C>,String>{
        let mut y=vec![ZERO;x.len()];self.systems[0].op.apply(x,&mut y)?;self.applications+=1;Ok(y)
    }
    fn traces(&self,x:&[C])->Vec<f64>{
        let(n,d)=(self.n,self.d);let trace=|q:usize|(0..n).fold(ZERO,|s,i|C{re:s.re+x[q*d+i*n+i].re,im:s.im+x[q*d+i*n+i].im});
        let mut out=vec![trace(self.centre).re];
        for &(q,_,_)in &self.canonical{if q==self.centre{continue;}let t=trace(q);out.push(t.re);out.push(t.im);}
        out
    }
    fn readout(&self,x:&[C])->Vec<C>{
        let(n,d)=(self.n,self.d);self.canonical.iter().flat_map(|&(q,_,_)|(0..n).map(move|i|x[q*d+i*n+i])).collect()
    }
    /// The basis vector and its translated copies, with their images.
    fn add_generators(&mut self,space:&mut Space,v:Vec<C>)->Result<(),String>{
        let start=Instant::now();
        let mut new=Vec::with_capacity(1+2*self.translations.len());
        for(minus,plus)in &self.translations{
            let(mut sum,mut diff)=(vec![ZERO;v.len()],vec![ZERO;v.len()]);let d=self.d;
            for q in 0..minus.len(){
                let a=minus[q].map(|p|&v[p*d..(p+1)*d]);let b=plus[q].map(|p|&v[p*d..(p+1)*d]);
                for e in 0..d{let x=a.map_or(ZERO,|a|a[e]);let y=b.map_or(ZERO,|b|b[e]);
                    sum[q*d+e]=C{re:x.re+y.re,im:x.im+y.im};
                    // i (x - y).
                    diff[q*d+e]=C{re:-(x.im-y.im),im:x.re-y.re};}
            }
            new.push(sum);new.push(diff);
        }
        new.insert(0,v);
        space.basis.push(space.z.len());
        for z in new{let image=self.image(&z)?;space.traces.push(self.traces(&z));space.readout.push(self.readout(&z));space.z.push(z);space.az.push(image);}
        self.timing.images_ms+=start.elapsed().as_secs_f64()*1e3;Ok(())
    }
    /// One rational Krylov column.
    fn extend(&mut self,space:&mut Space)->Result<bool,String>{
        let start=Instant::now();
        let j=space.basis.len()-1;let p=j%self.poles.len();
        let b:Vec<C>=space.z[space.basis[j]].iter().map(|v|C{re:-v.re,im:-v.im}).collect();
        let options=KrylovOptions{populations:false,residual_guard:INNER_RTOL,restart:60,maximum_steps:600,..KrylovOptions::default()};
        let solve=krylov::gmres(&mut self.systems[p],&b,None,&[],&options)?;self.inner_steps+=solve.steps;
        let mut w=solve.x;
        let refs:Vec<&[C]>=space.basis.iter().map(|&g|space.z[g].as_slice()).collect();
        for _ in 0..2{let h=self.vectors.dots(&refs,&w);self.vectors.subtract(&refs,&h,&mut w)?;}
        let norm=self.vectors.norm(&w);
        self.timing.columns_ms+=start.elapsed().as_secs_f64()*1e3;
        if !(norm>1e-12){return Ok(false);}
        for v in w.iter_mut(){*v=scaled(*v,1./norm);}
        self.add_generators(space,w)?;Ok(true)
    }
    /// Gram matrices over every generator.
    fn cover(&mut self,space:&mut Space)->Result<(),String>{
        let start=Instant::now();let(old,new)=(space.covered,space.z.len());if new==old{return Ok(());}
        let z:Vec<&[C]>=space.z.iter().map(|v|v.as_slice()).collect();let az:Vec<&[C]>=space.az.iter().map(|v|v.as_slice()).collect();
        let kz=self.vectors.gram(&z,&z[old..])?;let hz=self.vectors.gram(&z,&az[old..])?;let hr=self.vectors.gram(&z[old..],&az[..old])?;
        let width=new-old;let(mut k,mut h)=(vec![0.;new*new],vec![0.;new*new]);
        for i in 0..old{for j in 0..old{k[i*new+j]=space.k[i*old+j];h[i*new+j]=space.h[i*old+j];}}
        for i in 0..new{for c in 0..width{let j=old+c;k[i*new+j]=kz[i*width+c];k[j*new+i]=kz[i*width+c];h[i*new+j]=hz[i*width+c];}}
        for c in 0..width{for j in 0..old{h[(old+c)*new+j]=hr[c*old+j];}}
        space.k=k;space.h=h;space.covered=new;
        self.timing.gram_ms+=start.elapsed().as_secs_f64()*1e3;Ok(())
    }
    fn reduce(&mut self,space:&Space,initial_norm:f64,times:&[f64])->Result<Reduced,String>{
        let start=Instant::now();let lapack=&self.prepared.lapack;let m=space.covered;let horizon=*times.last().unwrap_or(&0.);
        // Orthonormal coordinates W = Z T (the first generator, x(0), is
        // always kept).
        let(t,r,gram_condition)=krylov::orthonormal_coordinates(&space.k,m,DEPENDENT);
        let tt=transpose_rows(&t,m,r);
        let g0=product(lapack,&product(lapack,&tt,&space.h,r,m,m),&t,r,m,r);
        let column0:Vec<f64>=(0..m).map(|i|initial_norm*space.k[i*m]).collect();
        let u0=product(lapack,&tt,&column0,r,m,1);
        // Trace evolution: T_W G = Omega T_W, by the least-norm correction.
        let tz=transpose_rows(&space.traces.concat(),m,space.traces[0].len());let kt=space.traces[0].len();
        let tw=product(lapack,&tz,&t,kt,m,r);
        let mut omega=vec![0.;kt*kt];
        for(c,&(q,_,nu))in self.canonical.iter().filter(|c|c.0!=self.centre).enumerate(){let _=q;
            let(a,b)=(1+2*c,2+2*c);omega[a*kt+b]=nu;omega[b*kt+a]=-nu;}
        let defect={let mut x=product(lapack,&tw,&g0,kt,r,r);let y=product(lapack,&omega,&tw,kt,kt,r);for(a,b)in x.iter_mut().zip(&y){*a-=b;}x};
        let gram=product(lapack,&tw,&transpose_rows(&tw,kt,r),kt,r,kt);
        let(gv,ge)=symmetric_eigen(lapack,&gram,kt)?;let top=gv.iter().fold(0_f64,|a,&v|a.max(v));
        let mut pseudo=vec![0.;kt*kt];
        for j in 0..kt{if gv[j]>TRACE_RANK*TRACE_RANK*top{for a in 0..kt{for b in 0..kt{pseudo[a*kt+b]+=ge[a+kt*j]*ge[b+kt*j]/gv[j];}}}}
        let correction=product(lapack,&product(lapack,&transpose_rows(&tw,kt,r),&pseudo,r,kt,kt),&defect,r,kt,r);
        let mut g:Vec<f64>=g0.iter().zip(&correction).map(|(a,b)|a-b).collect();
        let reflected=if horizon>0.{reflect_growing(lapack,&mut g,r,horizon)?}else{0};
        // Trajectory on the uniform grid.
        let step=if times.len()>1{times[1]-times[0]}else{0.};
        let gcol=transpose_rows(&g,r,r);
        let propagator=crate::dense_expm::exponential(lapack,&gcol,r,step)?;
        let mut u=vec![u0.clone()];
        for _ in 1..times.len(){let last=u.last().unwrap();let next:Vec<f64>=(0..r).map(|i|(0..r).map(|c|propagator[i+r*c]*last[c]).sum()).collect();u.push(next);}
        if u.iter().flatten().any(|x|!x.is_finite()){return Err("Nonfinite reduced trajectory".into());}
        // Ritz decomposition for the error propagation.
        let(re,im,ev)=lapack.eigen(&gcol,r)?;
        let mut mu=vec![ZERO;r];let mut s=vec![ZERO;r*r];let mut pair=vec![0_u8;r];let mut j=0;
        while j<r{if im[j]!=0.&&j+1<r{mu[j]=C{re:re[j],im:im[j]};mu[j+1]=C{re:re[j],im:-im[j]};pair[j]=1;pair[j+1]=2;
                for i in 0..r{let(a,b)=(ev[i+r*j],ev[i+r*(j+1)]);s[i*r+j]=C{re:a,im:b};s[i*r+j+1]=C{re:a,im:-b};}j+=2;}
            else{mu[j]=C{re:re[j],im:0.};for i in 0..r{s[i*r+j]=C{re:ev[i+r*j],im:0.};}j+=1;}}
        let(lu,pivots)=lapack.factor(&ev,r)?;let y=lapack.solve_matrix_factored(&lu,&pivots,&u0,r)?;
        let mut a=vec![ZERO;r];let mut j=0;
        while j<r{if pair[j]==1{a[j]=C{re:0.5*y[j],im:-0.5*y[j+1]};a[j+1]=C{re:0.5*y[j],im:0.5*y[j+1]};j+=2;}else{a[j]=C{re:y[j],im:0.};j+=1;}}
        let mut identity=vec![0.;r*r];for i in 0..r{identity[i+r*i]=1.;}
        let inverse=lapack.solve_matrix_factored(&lu,&pivots,&identity,r)?;
        let one=|x:&[f64]|(0..r).map(|c|x[c*r..(c+1)*r].iter().map(|v|v.abs()).sum::<f64>()).fold(0.,f64::max);
        let condition=one(&ev)*one(&inverse);
        self.timing.reduced_ms+=start.elapsed().as_secs_f64()*1e3;
        let growth=mu.iter().map(|z|z.re*horizon).fold(f64::NEG_INFINITY,f64::max);
        Ok(Reduced{t,r,g,reflected,gram_condition,growth,u,mu,s,a,condition,lu,pivots,pair})
    }
    /// Raw populations of the reduced trajectory.
    fn populations(&self,space:&Space,reduced:&Reduced,times:&[f64])->Vec<Vec<f64>>{self.readout_rows(space,reduced,times,&reduced.u)}
    /// Populations of reduced vectors (one per time) in the orthonormal coordinates.
    fn readout_rows(&self,space:&Space,reduced:&Reduced,times:&[f64],vectors:&[Vec<f64>])->Vec<Vec<f64>>{
        let(n,m,r)=(self.n,space.covered,reduced.r);let lapack=&self.prepared.lapack;
        // Per canonical mode: populations of the orthonormal coordinates
        // (real parts in rows 0..n, imaginary in n..2n) and along the trajectory.
        let mut trajectory=vec![0.;r*times.len()];for(k,u)in vectors.iter().enumerate(){for i in 0..r{trajectory[i*times.len()+k]=u[i];}}
        let mut rows=vec![vec![0.;n];times.len()];
        for(c,&(_,w,nu))in self.canonical.iter().enumerate(){
            let mut y=vec![0.;2*n*m];
            for g in 0..m{for i in 0..n{let v=space.readout[g][c*n+i];y[i*m+g]=v.re;y[(n+i)*m+g]=v.im;}}
            let along=product(lapack,&product(lapack,&y,&reduced.t,2*n,m,r),&trajectory,2*n,r,times.len());
            for(k,&time)in times.iter().enumerate(){let(cs,sn)=((nu*time).cos(),(nu*time).sin());
                for i in 0..n{let(re,im)=(along[i*times.len()+k],along[(n+i)*times.len()+k]);rows[k][i]+=w*(cs*re-sn*im);}}
        }
        rows
    }
    /// Lattice fields Z T b for coefficient vectors b over the orthonormal coordinates.
    fn fields(&self,space:&Space,reduced:&Reduced,coefficients:&[Vec<f64>],images:bool)->Result<Vec<Vec<C>>,String>{
        let(m,r)=(space.covered,reduced.r);
        let p=coefficients.len();let mut b=vec![0.;r*p];for(j,c)in coefficients.iter().enumerate(){for i in 0..r{b[i*p+j]=c[i];}}
        let tb=product(&self.prepared.lapack,&reduced.t,&b,m,r,p);
        let over:Vec<Vec<f64>>=(0..p).map(|j|(0..m).map(|g|tb[g*p+j]).collect()).collect();
        let source:Vec<&[C]>=if images{space.az.iter().map(|v|v.as_slice()).collect()}else{space.z.iter().map(|v|v.as_slice()).collect()};
        self.vectors.combinations(&source[..m],&over)
    }
    /// The basis estimate: the residual's leading singular components,
    /// propagated.
    /// An orthonormal basis Q of the range of R Omega (R = A W - W G) and
    /// B = Q^T R (rows of length r).
    fn residual_range(&self,space:&Space,reduced:&Reduced,omega:&[Vec<f64>])->Result<(Vec<Vec<C>>,Vec<f64>),String>{
        let(r,m)=(reduced.r,space.covered);let lapack=&self.prepared.lapack;
        let g_omega:Vec<Vec<f64>>=omega.iter().map(|o|(0..r).map(|i|(0..r).map(|c|reduced.g[i*r+c]*o[c]).sum()).collect()).collect();
        let first=self.fields(space,reduced,omega,true)?;let second=self.fields(space,reduced,&g_omega,false)?;
        let mut q:Vec<Vec<C>>=Vec::new();
        for(a,b)in first.into_iter().zip(second){
            let mut v:Vec<C>=a.iter().zip(&b).map(|(x,y)|C{re:x.re-y.re,im:x.im-y.im}).collect();
            for _ in 0..2{if q.is_empty(){break;}let refs:Vec<&[C]>=q.iter().map(|x|x.as_slice()).collect();
                let h=self.vectors.dots(&refs,&v);self.vectors.subtract(&refs,&h,&mut v)?;}
            let norm=self.vectors.norm(&v);if norm>1e-300{q.push(v.iter().map(|x|scaled(*x,1./norm)).collect());}
        }
        let rows=q.len();if rows==0{return Ok((q,Vec::new()));}
        let qr:Vec<&[C]>=q.iter().map(|v|v.as_slice()).collect();
        let za:Vec<&[C]>=space.az[..m].iter().map(|v|v.as_slice()).collect();let zz:Vec<&[C]>=space.z[..m].iter().map(|v|v.as_slice()).collect();
        let(qa,qz)=(self.vectors.gram(&qr,&za)?,self.vectors.gram(&qr,&zz)?);
        let(ta,tz)=(product(lapack,&qa,&reduced.t,rows,m,r),product(lapack,&qz,&reduced.t,rows,m,r));
        let tg=product(lapack,&tz,&reduced.g,rows,r,r);
        Ok((q,ta.iter().zip(&tg).map(|(x,y)|x-y).collect()))
    }
    /// The basis estimate: the residual's leading singular components (a
    /// randomized range finder with one power iteration), propagated.
    fn basis_estimate(&mut self,space:&Space,reduced:&Reduced,eigenbasis:&BlockEigenbasis,times:&[f64],groups:&[Vec<usize>])->Result<f64,String>{
        let start=Instant::now();let r=reduced.r;let lapack=&self.prepared.lapack;
        // Translated copies multiply the residual's rank.
        let p=(RESIDUAL_RANGE*(1+2*self.translations.len())+8).min(r);
        let mut state=0x9e3779b97f4a7c15_u64;let mut random=||{state^=state<<13;state^=state>>7;state^=state<<17;(state>>11)as f64/(1u64<<53)as f64-0.5};
        let omega:Vec<Vec<f64>>=(0..p).map(|_|(0..r).map(|_|random()).collect()).collect();
        let(_,b)=self.residual_range(space,reduced,&omega)?;
        let rows=b.len()/r.max(1);if rows==0{return Ok(0.);}
        // Power iteration: Omega' = R^T Q = B^T.
        let omega:Vec<Vec<f64>>=(0..rows).map(|i|b[i*r..(i+1)*r].to_vec()).collect();
        let(q,b)=self.residual_range(space,reduced,&omega)?;
        let rows=q.len();if rows==0{return Ok(0.);}
        // SVD of B through B B^T: R ~ sum_j sigma_j (Q l_j) b_j^T.
        let bbt=product(lapack,&b,&transpose_rows(&b,rows,r),rows,r,rows);
        let(values,vectors)=symmetric_eigen(lapack,&bbt,rows)?;
        let top=values.iter().fold(0_f64,|a,&v|a.max(v)).max(0.);
        let total:f64=values.iter().map(|v|v.max(0.)).sum();let mut kept=0.;
        let mut lefts=Vec::new();let mut alpha=Vec::new();
        for j in(0..rows).rev(){
            if !(values[j]>RESIDUAL_COMPONENT*RESIDUAL_COMPONENT*top){continue;}
            kept+=values[j];let sigma=values[j].sqrt();let left:Vec<f64>=(0..rows).map(|i|vectors[i+rows*j]).collect();
            let right:Vec<f64>=(0..r).map(|c|(0..rows).map(|i|b[i*r+c]*left[i]).sum::<f64>()/sigma).collect();
            lefts.push(left);alpha.push(reduced.exponential(&right).into_iter().map(|a|scaled(a,sigma)).collect::<Vec<C>>());
        }
        let qr:Vec<&[C]>=q.iter().map(|v|v.as_slice()).collect();
        let fields=self.vectors.combinations(&qr,&lefts)?;
        let rows_t=eigenbasis.propagate(self.op(),&fields,&alpha,&reduced.mu,times)?;
        self.timing.basis_ms+=start.elapsed().as_secs_f64()*1e3;
        let scale=(total/kept.max(f64::MIN_POSITIVE)).sqrt();
        Ok(scale*rows_t.iter().map(|p|ledger::metric(p,&vec![0.;self.n],groups)).fold(0.,f64::max))
    }
    /// Population response to sources sum_i s_i psi_i(t) (psi_i exponential
    /// sums over the Ritz values): the Galerkin projection of the sources is
    /// solved in the reduced model, which carries the couplings between
    /// modes (a source on the ball's edge reaches the centre populations
    /// only through them); the remainder is propagated through every mode's
    /// dressed block (`eigenbasis`).
    fn source_response(&self,space:&Space,reduced:&Reduced,eigen:&BlockEigenbasis,sources:&[Vec<C>],alpha:&[Vec<C>],times:&[f64])->Result<Vec<Vec<f64>>,String>{
        let(m,r)=(space.covered,reduced.r);let lapack=&self.prepared.lapack;let count=sources.len();
        if count==0{return Ok(vec![vec![0.;self.n];times.len()]);}
        let z:Vec<&[C]>=space.z[..m].iter().map(|v|v.as_slice()).collect();let src:Vec<&[C]>=sources.iter().map(|v|v.as_slice()).collect();
        // W^T s_i = T^T Z^T s_i (r x count).
        let zs=self.vectors.gram(&z,&src)?;let ws=product(lapack,&transpose_rows(&reduced.t,m,r),&zs,r,m,count);
        let columns:Vec<Vec<f64>>=(0..count).map(|i|(0..r).map(|k|ws[k*count+i]).collect()).collect();
        let projected=self.fields(space,reduced,&columns,false)?;
        let remainder:Vec<Vec<C>>=sources.iter().zip(&projected).map(|(a,b)|a.iter().zip(b).map(|(x,y)|C{re:x.re-y.re,im:x.im-y.im}).collect()).collect();
        let mut rows=eigen.propagate(self.op(),&remainder,alpha,&reduced.mu,times)?;
        // eps' = G eps + sum_i (W^T s_i) psi_i(t): in the eigenbasis,
        // eps_k(t) = sum_l beta_kl (e^{mu_l t} - e^{mu_k t}) / (mu_l - mu_k).
        let(mu,t_count)=(&reduced.mu,times.len());let horizon=times.iter().fold(0_f64,|a,&t|a.max(t)).max(1e-300);
        let mut beta=vec![ZERO;r*r];
        for(i,column)in columns.iter().enumerate(){let c=reduced.ritz(lapack,column)?;for k in 0..r{if !nonzero(c[k]){continue;}for l in 0..r{mul_add(&mut beta[k*r+l],c[k],alpha[i][l]);}}}
        let cexp=|z:C,t:f64|{let a=(z.re*t).exp();C{re:a*(z.im*t).cos(),im:a*(z.im*t).sin()}};
        // Column-major: Gamma (r x r), e^{mu t} (r x T), S (r x r).
        let exps:Vec<C>=times.iter().flat_map(|&t|mu.iter().map(move|&z|cexp(z,t))).collect();
        let mut gamma=vec![ZERO;r*r];let mut diagonal=vec![ZERO;r];let mut linear=vec![ZERO;r];
        for k in 0..r{for l in 0..r{let b=beta[k*r+l];if !nonzero(b){continue;}let den=C{re:mu[l].re-mu[k].re,im:mu[l].im-mu[k].im};
            if magnitude(den)*horizon<1e-8{add(&mut linear[k],b);continue;}
            let q=den.re*den.re+den.im*den.im;let v=mul(b,C{re:den.re/q,im:-den.im/q});gamma[k+r*l]=v;add(&mut diagonal[k],v);}}
        let gemm=lapack.complex_gemm();
        let mut hat=vec![ZERO;r*t_count];gemm.multiply(r,t_count,r,ONE,&gamma,r,&exps,r,ZERO,&mut hat,r)?;
        for(ti,&t)in times.iter().enumerate(){for k in 0..r{let e=exps[k+r*ti];let w=mul(e,diagonal[k]);let h=&mut hat[k+r*ti];h.re-=w.re;h.im-=w.im;mul_add(h,scaled(linear[k],t),e);}}
        let mut s_cm=vec![ZERO;r*r];for i in 0..r{for k in 0..r{s_cm[i+r*k]=reduced.s[i*r+k];}}
        let mut full=vec![ZERO;r*t_count];gemm.multiply(r,t_count,r,ONE,&s_cm,r,&hat,r,ZERO,&mut full,r)?;
        let eps:Vec<Vec<f64>>=(0..t_count).map(|ti|(0..r).map(|i|full[i+r*ti].re).collect()).collect();
        for(row,galerkin)in rows.iter_mut().zip(self.readout_rows(space,reduced,times,&eps)){for(a,b)in row.iter_mut().zip(galerkin){*a+=b;}}
        Ok(rows)
    }
    /// The next shell's estimate (slaved shell, its source on the ball
    /// propagated, plus the shell's own populations) and, when `fold`, the
    /// fold charges, on the trajectory's compression.
    #[allow(clippy::too_many_arguments)]
    fn shell_and_fold(&mut self,dressed:&StructuredDressed,ledger:&RemainderLedger,order:usize,space:&Space,reduced:&Reduced,eigen:&BlockEigenbasis,
        times:&[f64],groups:&[Vec<usize>],fold:bool)->Result<(f64,f64),String>{
        let(n,d)=(self.n,self.d);let prepared=self.prepared;
        let(xs,ys,scale)=self.compression(space,reduced)?;
        let phi:Vec<Vec<C>>=ys.iter().map(|y|reduced.exponential(y)).collect();
        let start=Instant::now();
        let shell=match Shell::new(prepared,dressed,self.op(),order,false)?{
            None=>0.,
            Some(mut shell)=>{
                let mut sources=Vec::new();let mut own=vec![vec![0.;n];times.len()];
                for(x,y)in xs.iter().zip(&ys){
                    let(source,field)=shell.slaved(x)?;sources.push(source);
                    // The shell's own populations, phi_i(t) = y_i^T u(t).
                    for(ti,(&time,u))in times.iter().zip(&reduced.u).enumerate(){
                        let phi_t:f64=y.iter().zip(u).map(|(a,b)|a*b).sum();if phi_t==0.{continue;}
                        for(k,&s)in shell.modes.iter().enumerate(){if !shell.op.canonical[s]{continue;}
                            let w=if shell.op.mirror[s]==s{1.}else{2.};let nu=shell.op.nu[s];let phase=C{re:(nu*time).cos(),im:(nu*time).sin()};
                            for i in 0..n{own[ti][i]+=w*phi_t*mul(phase,field[k*d+i*n+i]).re;}}
                    }
                }
                let propagated=self.source_response(space,reduced,eigen,&sources,&phi,times)?;
                scale*propagated.iter().zip(&own).map(|(a,b)|{let s:Vec<f64>=a.iter().zip(b).map(|(x,y)|x+y).collect();ledger::metric(&s,&vec![0.;n],groups)}).fold(0.,f64::max)
            }
        };
        self.timing.shell_ms+=start.elapsed().as_secs_f64()*1e3;let start=Instant::now();
        let charges=if !fold||prepared.plan.far().is_empty(){0.}else{
            let mut sources=Vec::new();let mut third=false;
            for x in &xs{let s=prepared.fold.third_order_source(self.op(),x)?;third|=s.iter().any(|z|nonzero(*z));sources.push(s);}
            let(bound,sources)=if ledger.third_order>0.&&third{(ledger.third_order,sources)}
                else{let mut fold=Vec::new();for x in &xs{fold.push(self.systems[0].op.apply_fold(x)?);}(ledger.fold,fold)};
            let propagated=self.source_response(space,reduced,eigen,&sources,&phi,times)?;
            let response=scale*propagated.iter().map(|p|ledger::metric(p,&vec![0.;n],groups)).fold(0.,f64::max);
            PopulationCharges::new(ledger,if bound>0.{response/bound}else{0.}).total()
        };
        self.timing.fold_ms+=start.elapsed().as_secs_f64()*1e3;
        Ok((shell,charges))
    }
    /// Low-rank compression of the trajectory: fields x_i and coefficient
    /// vectors y_i with x(t) ~ sum_i x_i (y_i^T u(t)), and the factor
    /// |U|_F / |U_kept|_F by which estimates on it are scaled.
    fn compression(&self,space:&Space,reduced:&Reduced)->Result<(Vec<Vec<C>>,Vec<Vec<f64>>,f64),String>{
        let r=reduced.r;let mut uut=vec![0.;r*r];
        for u in &reduced.u{for i in 0..r{for j in 0..r{uut[i*r+j]+=u[i]*u[j];}}}
        let(values,vectors)=symmetric_eigen(&self.prepared.lapack,&uut,r)?;
        let top=values.iter().fold(0_f64,|a,&v|a.max(v)).max(f64::MIN_POSITIVE);
        let chosen:Vec<usize>=(0..r).rev().filter(|&j|values[j]>TRAJECTORY_COMPONENT*TRAJECTORY_COMPONENT*top).take(TRAJECTORY_RANGE).collect();
        let total:f64=values.iter().map(|v|v.max(0.)).sum();let kept:f64=chosen.iter().map(|&j|values[j]).sum();
        let ys:Vec<Vec<f64>>=chosen.iter().map(|&j|(0..r).map(|i|vectors[i+r*j]).collect()).collect();
        Ok((self.fields(space,reduced,&ys,false)?,ys,(total/kept.max(f64::MIN_POSITIVE)).sqrt()))
    }
}

/// Order-k propagation with its checkpoints; None when the shell asks for
/// the next order.
#[allow(clippy::too_many_arguments)]
fn attempt(prepared:&LatticePrepared,dressed:&StructuredDressed,eigenbasis:&OnceLock<Result<BlockEigenbasis,String>>,
    order:usize,initial:&[C],times:&[f64],keep:&[usize],groups:&[Vec<usize>],tolerance:f64,checkpoints:&mut Vec<Checkpoint>)->Result<Option<FiniteSolution>,String>{
    let n=prepared.generator.n;let d=n*n;let horizon=*times.last().unwrap_or(&0.);
    let modes=prepared.modes(order)?;
    let first=LatticeOperator::new(prepared,&modes)?;
    let ledger=RemainderLedger::new(&prepared.plan,first.nu_max(),first.dropped())?;
    if ledger.expansion_parameter>ledger::EXPANSION_LIMIT{
        return Err(format!("Static fold expansion parameter {:e} above {:e} at near order {order}",ledger.expansion_parameter,ledger::EXPANSION_LIMIT));}
    let poles=if horizon>POLE_HORIZON{vec![POLE,POLE_SCALE/horizon]}else{vec![POLE]};
    let clock=Instant::now();
    // The pole moves the slow population block off zero, and the Woodbury
    // term no longer helps: measured on old four lasers and three lasers +
    // tweezer, 715 and 996 shifted-solve steps without it against 706 and
    // 972 with it, at a sixth of the cost.
    let mut batches:Vec<StructuredBatch<'_>>=poles.iter().map(|&s|dressed.batch(&first.nu,&first.canonical,&first.mirror,None,s,false)).collect::<Result<_,_>>()?;
    let dressed_ms=clock.elapsed().as_secs_f64()*1e3;
    let vectors=Vectors::new(&first)?;
    let centre=first.centre;
    let canonical:Vec<(usize,f64,f64)>=(0..modes.len()).filter(|&q|first.canonical[q]).map(|q|(q,if first.mirror[q]==q{1.}else{2.},first.nu[q])).collect();
    // Translation pairs: explicit near steps mixing above TRANSLATION_MIXING.
    let mut strong:Vec<(f64,usize,&Charge)>=prepared.plan.explicit().into_iter().map(|s|&prepared.plan.steps[s])
        .filter(|st|st.size>TRANSLATION_MIXING*st.nu.abs()&&st.charge>=opposite(&st.charge)).enumerate().map(|(i,st)|(st.size/st.nu.abs(),i,&st.charge)).collect();
    // The strongest, kept in the plan's order.
    strong.sort_by(|a,b|b.0.total_cmp(&a.0));strong.truncate(MAXIMUM_TRANSLATIONS);strong.sort_by_key(|x|x.1);
    let mut translations=Vec::new();
    for(_,_,charge)in strong{
        let source=|sign:i32|->Vec<Option<usize>>{modes.iter().map(|q|{
            let shifted:Option<Charge>=q.iter().zip(charge).map(|(&a,&b)|a.checked_sub(sign*b)).collect();
            shifted.and_then(|c|first.index.get(&c).copied())}).collect()};
        translations.push((source(1),source(-1)));
    }
    let mut systems=Vec::new();let mut first=Some(first);
    for(batch,&sigma)in batches.iter_mut().zip(&poles){
        let op=match first.take(){Some(op)=>op,None=>LatticeOperator::new(prepared,&modes)?};
        systems.push(System::shifted(op,batch,sigma)?);}
    let mut lattice=Lattice{prepared,systems,poles,vectors,translations,canonical,centre,n,d,inner_steps:0,applications:0,
        timing:FiniteTiming{dressed_ms,..Default::default()}};
    // x(0) = delta_q0 rho0.
    let mut x0=vec![ZERO;lattice.length()];x0[centre*d..(centre+1)*d].copy_from_slice(initial);
    let initial_norm=lattice.vectors.norm(&x0);if !(initial_norm>0.){return Err("Zero initial density".into());}
    let mut space=Space{basis:vec![],z:vec![],az:vec![],covered:0,k:vec![],h:vec![],traces:vec![],readout:vec![]};
    lattice.add_generators(&mut space,x0.iter().map(|v|scaled(*v,1./initial_norm)).collect())?;
    let initial_rows:Vec<f64>=(0..n).map(|i|initial[i*n+i].re).collect();
    let mut target=FIRST_COLUMNS;
    loop{
        let clock=Instant::now();
        let mut exhausted=false;
        while space.basis.len()<target{if !lattice.extend(&mut space)?{exhausted=true;break;}}
        lattice.cover(&mut space)?;
        let reduced=lattice.reduce(&space,initial_norm,times)?;
        let mut raw=lattice.populations(&space,&reduced,times);raw[0]=initial_rows.clone();
        let projected:Vec<Vec<f64>>=raw.iter().map(|p|simplex(p)).collect();
        let projection=raw.iter().zip(&projected).map(|(a,b)|ledger::metric(a,b,groups)).fold(0.,f64::max);
        let eigen=eigenbasis.get_or_init(||BlockEigenbasis::new(prepared)).as_ref().map_err(|e|e.clone())?;
        let basis=lattice.basis_estimate(&space,&reduced,eigen,times,groups)?;
        let columns=space.basis.len();
        let mut checkpoint=Checkpoint{order,columns,dimension:reduced.r,reflected:reduced.reflected,gram_condition:reduced.gram_condition,growth:reduced.growth,basis,shell:None,fold:None,projection,total:None,action:"",milliseconds:0.};
        let more=|exhausted:bool,columns:usize|!exhausted&&columns<MAXIMUM_COLUMNS;
        if KAPPA*basis+projection>tolerance{
            checkpoint.action=if more(exhausted,columns){"add-columns"}else{"not-converged"};
            checkpoint.milliseconds=clock.elapsed().as_secs_f64()*1e3;checkpoints.push(checkpoint);
            if !more(exhausted,columns){
                let trail:Vec<String>=checkpoints.iter().map(|c|format!("{} columns {:.2e} (dimension {}, Gram condition {:.1e}, growth {:.1e}, reflected {})",
                    c.columns,c.basis,c.dimension,c.gram_condition,c.growth,c.reflected)).collect();
                return Err(format!("Finite-time basis estimate not below the tolerance: {}",trail.join(", ")));}
            target+=MORE_COLUMNS;continue;
        }
        let(shell,fold)=lattice.shell_and_fold(dressed,&ledger,order,&space,&reduced,eigen,times,groups,true)?;
        let total=KAPPA*(basis+shell+fold)+projection;
        checkpoint.shell=Some(shell);checkpoint.fold=Some(fold);checkpoint.total=Some(total);
        if total<=tolerance{
            checkpoint.action="accept";checkpoint.milliseconds=clock.elapsed().as_secs_f64()*1e3;checkpoints.push(checkpoint);
            let preconditions=lattice.systems.iter().map(|s|s.preconditions).sum();
            for s in &lattice.systems{let t=&mut lattice.timing.inner;t.apply_ms+=s.timing.apply_ms;t.precondition_ms+=s.timing.precondition_ms;t.orthogonalize_ms+=s.timing.orthogonalize_ms;}
            let applications=lattice.applications+lattice.systems.iter().map(|s|s.applications).sum::<usize>();
            let picks:Vec<Vec<f64>>=keep.iter().map(|&k|reduced.u[k].clone()).collect();
            let fields=if picks.is_empty(){vec![]}else{lattice.fields(&space,&reduced,&picks,false)?};
            return Ok(Some(FiniteSolution{fields,populations:projected,estimated_error:total,projection_charge:projection,basis,shell,fold,
                order,modes:modes.len(),columns,dimension:reduced.r,translations:lattice.translations.len(),
                inner_steps:lattice.inner_steps,applications,preconditions,eigenbasis_condition:eigen.condition,ritz_condition:reduced.condition,
                checkpoints:std::mem::take(checkpoints),timing:lattice.timing}));
        }
        if KAPPA*shell>0.5*tolerance&&order<MAXIMUM_ORDER{
            checkpoint.action="next-order";checkpoint.milliseconds=clock.elapsed().as_secs_f64()*1e3;checkpoints.push(checkpoint);return Ok(None);}
        checkpoint.action=if more(exhausted,columns){"add-columns"}else{"not-converged"};
        checkpoint.milliseconds=clock.elapsed().as_secs_f64()*1e3;checkpoints.push(checkpoint);
        if !more(exhausted,columns){return Err(format!("Finite-time estimate {total:e} not below the tolerance with {columns} columns"));}
        target+=MORE_COLUMNS;
    }
}

/// Euclidean projection of real values onto the probability simplex.
pub fn simplex(values:&[f64])->Vec<f64>{
    let mut sorted=values.to_vec();sorted.sort_by(|a,b|b.total_cmp(a));
    let(mut sum,mut threshold)=(0.,0.);
    for(index,&value)in sorted.iter().enumerate(){sum+=value;let candidate=(sum-1.)/(index+1)as f64;if value>candidate{threshold=candidate;}else{break;}}
    values.iter().map(|x|(x-threshold).max(0.)).collect()
}

/// The populations of rho0 (row-major n x n, the request frame at t = 0)
/// on the uniform grid `times` (starting at zero), with their estimated
/// error; the near order rises from `start` (else START_ORDER) as the shell
/// estimate asks, and the fields at the time indices `keep` are returned.
pub fn propagate(prepared:&LatticePrepared,initial:&[C],times:&[f64],groups:&[Vec<usize>],start:Option<usize>,keep:&[usize])->Result<FiniteSolution,String>{
    propagate_within(prepared,initial,times,groups,start,keep,POPULATION_TOLERANCE)
}
/// `propagate` accepted at `tolerance` (a share of POPULATION_TOLERANCE).
fn propagate_within(prepared:&LatticePrepared,initial:&[C],times:&[f64],groups:&[Vec<usize>],start:Option<usize>,keep:&[usize],tolerance:f64)->Result<FiniteSolution,String>{
    let n=prepared.generator.n;
    if initial.len()!=n*n{return Err("Initial density dimension differs".into());}
    if times.len()<2||times[0]!=0.{return Err("Finite-time grid must start at zero with at least two times".into());}
    let step=times[1];let horizon=times[times.len()-1];
    if !(horizon>0.)||times.iter().enumerate().any(|(i,&t)|(t-step*i as f64).abs()>1e-12*horizon.max(1.)){return Err("Finite-time grid must be uniform".into());}
    if keep.iter().any(|&k|k>=times.len()){return Err("A kept field lies beyond the grid".into());}
    // The centre eigenbasis is prepared beside the first columns.
    let eigenbasis=OnceLock::new();
    std::thread::scope(|scope|{
        scope.spawn(||{eigenbasis.get_or_init(||BlockEigenbasis::new(prepared));});
        let clock=Instant::now();
        let dressed=StructuredDressed::new(prepared)?;let dressed_ms=clock.elapsed().as_secs_f64()*1e3;
        let mut checkpoints=Vec::new();
        for order in start.unwrap_or(START_ORDER)..=MAXIMUM_ORDER{
            if let Some(mut solution)=attempt(prepared,&dressed,&eigenbasis,order,initial,times,keep,groups,tolerance,&mut checkpoints)?{
                solution.timing.dressed_ms+=dressed_ms;
                solution.timing.eigenbasis_ms=eigenbasis.get().and_then(|e|e.as_ref().ok()).map_or(0.,|e|e.milliseconds);
                return Ok(solution);
            }
        }
        Err(format!("Finite-time populations not accepted up to near order {MAXIMUM_ORDER}"))
    })
}

/// Horizons above this (us) use the long-time machinery.
pub const LONG_HORIZON:f64=1000.;
/// The Krylov propagation's first reach (us; quadrupled while no switch
/// time is accepted) and its largest; the earliest switch time. With the
/// slow modes up to rates of about 1/us (`slow`), four lasers + tweezer
/// switch at 20 us; five lasers + tweezer resolve a 20 us reach at order 2
/// in 60 columns but needed 180 at 40 us and none converged at 80 us.
const REACH_START:f64=20.;
const REACH_LIMIT:f64=1280.;
const SWITCH_START:f64=20.;
/// Fewest points of the Krylov grid to its reach.
pub const SWITCH_SAMPLES:usize=65;
/// Share of the tolerance the expansion's error at t_s may take (it is
/// charged for every later time).
const FAST_SHARE:f64=0.1;

/// Populations of a field at time t: the near charges' diagonals with their
/// phases (canonical modes, weights 2 for a pair).
pub struct Readout{n:usize,modes:Vec<(usize,f64,f64)>}
impl Readout{
    pub fn new(prepared:&LatticePrepared,modes:&[Charge])->Result<Self,String>{
        let index:BTreeMap<&Charge,usize>=modes.iter().enumerate().map(|(i,q)|(q,i)).collect();
        let mut out=Vec::new();
        for(q,c)in modes.iter().enumerate(){let p=*index.get(&opposite(c)).ok_or("Mode set is not closed under q -> -q")?;
            if c<&modes[p]{continue;}out.push((q,if p==q{1.}else{2.},prepared.nu(c)?));}
        Ok(Self{n:prepared.generator.n,modes:out})
    }
    pub fn populations(&self,x:&[C],t:f64)->Vec<f64>{
        let(n,d)=(self.n,self.n*self.n);let mut p=vec![0.;n];
        for &(q,w,nu)in &self.modes{let(c,s)=((nu*t).cos(),(nu*t).sin());for i in 0..n{let v=x[q*d+i*n+i];p[i]+=w*(c*v.re-s*v.im);}}
        p
    }
    /// The populations' size at t (the ledger metric against zero).
    fn size(&self,x:&[C],t:f64,groups:&[Vec<usize>])->f64{ledger::metric(&self.populations(x,t),&vec![0.;self.n],groups)}
}

/// A field on the modes `from` carried to the modes `to` (another near
/// order): which of `to` it covers (the others zero, unknown to it), and
/// the populations' size at t of the modes only in `from` (left out).
pub fn transfer(prepared:&LatticePrepared,x:&[C],from:&[Charge],to:&[Charge],t:f64,groups:&[Vec<usize>])->Result<(Vec<C>,Vec<bool>,f64),String>{
    let d=prepared.generator.n.pow(2);let index:BTreeMap<&Charge,usize>=from.iter().enumerate().map(|(i,q)|(q,i)).collect();
    let mut kept=vec![ZERO;to.len()*d];let mut covered=vec![false;to.len()];let mut rest=x.to_vec();
    for(i,q)in to.iter().enumerate(){let Some(&j)=index.get(q)else{continue};covered[i]=true;
        kept[i*d..(i+1)*d].copy_from_slice(&x[j*d..(j+1)*d]);rest[j*d..(j+1)*d].fill(ZERO);}
    Ok((kept,covered,Readout::new(prepared,from)?.size(&rest,t,groups)))
}
/// x on the covered modes only (block size d).
fn masked(x:&[C],covered:&[bool],d:usize)->Vec<C>{
    let mut y=x.to_vec();for(q,&c)in covered.iter().enumerate(){if !c{y[q*d..(q+1)*d].fill(ZERO);}}y
}

/// Least-squares coefficients (alpha, beta) per slow mode of y on the
/// modes' real fields over the modes y covers (Hermitian-symmetric fields:
/// the real inner product), and the part of y outside their span there.
fn fit(lapack:&Lapack,modes:&[SlowMode],y:&[C],covered:&[bool])->Result<(Vec<(f64,f64)>,Vec<C>),String>{
    let d=y.len()/covered.len().max(1);
    let fields:Vec<Vec<C>>=modes.iter().flat_map(|m|std::iter::once(&m.a).chain(m.b.as_ref())).map(|f|masked(f,covered,d)).collect();
    let dot=|a:&[C],b:&[C]|a.iter().zip(b).map(|(x,y)|x.re*y.re+x.im*y.im).sum::<f64>();
    let r=fields.len();let mut gram=vec![0.;r*r];let mut rhs=vec![0.;r];
    for i in 0..r{rhs[i]=dot(&fields[i],y);for j in 0..r{gram[i+r*j]=dot(&fields[i],&fields[j]);}}
    let solution=if r>0{lapack.solve(&gram,&rhs,r)?}else{vec![]};
    let mut outside=masked(y,covered,d);for(f,&c)in fields.iter().zip(&solution){for(t,v)in outside.iter_mut().zip(f.iter()){t.re-=c*v.re;t.im-=c*v.im;}}
    let mut at=0;let coefficients=modes.iter().map(|m|{let alpha=solution[at];at+=1;
        let beta=if m.b.is_some(){at+=1;solution[at-1]}else{0.};(alpha,beta)}).collect();
    Ok((coefficients,outside))
}
/// Adds the slow modes from coefficients (alpha, beta), evolved over tau, to x.
fn evolve(modes:&[SlowMode],coefficients:&[(f64,f64)],tau:f64,x:&mut[C]){
    for(mode,&(alpha,beta))in modes.iter().zip(coefficients){
        let(lr,li)=(mode.lambda.re,mode.lambda.im);let decay=(lr*tau).exp();
        // (alpha, beta)' = [[l_r, l_i], [-l_i, l_r]] (alpha, beta).
        let(c,s)=((li*tau).cos(),(li*tau).sin());let(a,b)=(decay*(c*alpha+s*beta),decay*(-s*alpha+c*beta));
        for(t,v)in x.iter_mut().zip(&mode.a){t.re+=a*v.re;t.im+=a*v.im;}
        if let Some(bv)=&mode.b{for(t,v)in x.iter_mut().zip(bv){t.re+=b*v.re;t.im+=b*v.im;}}
    }
}
/// The Krylov grid to a reach t: uniform, at least SWITCH_SAMPLES points,
/// holding the output grid's multiples of `step` (t then rounded up to one
/// of them when step <= t).
fn switch_grid(t:f64,step:Option<f64>)->Vec<f64>{
    let finest=t/(SWITCH_SAMPLES-1)as f64;
    match step{
        Some(step)if step<=t=>{let per=(step/finest).ceil().max(1.)as usize;let count=(t/step-1e-9).ceil()as usize*per;
            (0..=count).map(|i|step*i as f64/per as f64).collect()}
        _=>(0..SWITCH_SAMPLES).map(|i|finest*i as f64).collect(),
    }
}

/// The long-time state: x(t) = x_s + sum_j (slow mode j)(t) for t >= t_s,
/// the coefficients fitted (least squares) to x(t_s) - x_s from the Krylov
/// propagation up to t_s, at the steady solution's order.
pub struct LongTime{
    pub switch:f64,
    /// The Krylov propagation on its grid `times` (to its reach, t_s or
    /// beyond).
    pub krylov:FiniteSolution,pub times:Vec<f64>,
    steady:Vec<C>,modes:Vec<SlowMode>,coefficients:Vec<(f64,f64)>,
    readout:Readout,
    /// Populations of the part of x(t_s) - x_s outside the slow modes (near
    /// shells beyond the steady order included), the drift of the fit (the
    /// fit at t_s / 2 evolved to t_s against the propagation: the fast part
    /// the fit attributes to the slow modes, which only shrinks as it
    /// decays), the eigen residuals' charge
    /// (relative residual times each mode's size) and the largest growth rate
    /// of excluded truncation artifacts.
    pub fast:f64,pub drift:f64,pub residuals:f64,pub artifact_growth:f64,
    pub steady_error:f64,
    /// The rejected switch times with their expansion errors.
    pub trail:Vec<(f64,f64)>,
}
impl LongTime{
    /// The field at t >= t_s.
    pub fn field(&self,t:f64)->Vec<C>{
        let mut x=self.steady.clone();evolve(&self.modes,&self.coefficients,t-self.switch,&mut x);x
    }
    pub fn populations(&self,t:f64)->Vec<f64>{self.readout.populations(&self.field(t),t)}
    /// Populations still relaxing at t (the slow modes' part).
    pub fn relaxing(&self,t:f64,groups:&[Vec<usize>])->f64{
        let mut x=vec![ZERO;self.steady.len()];evolve(&self.modes,&self.coefficients,t-self.switch,&mut x);self.readout.size(&x,t,groups)
    }
    /// The estimated error of the populations at t >= t_s.
    pub fn error(&self,t:f64)->f64{
        let growth=(self.artifact_growth*(t-self.switch)).exp();
        self.krylov.estimated_error+self.steady_error+KAPPA*((self.fast+self.drift)*growth+self.residuals)
    }
}

/// Why the long-time machinery declined: slow modes outside the
/// shift-invert set (a direct propagation may serve instead), or a failure.
#[derive(Clone,Debug)]
pub enum Declined{MissedModes(String),Failed(String)}
impl std::fmt::Display for Declined{
    fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result{match self{Declined::MissedModes(m)|Declined::Failed(m)=>f.write_str(m)}}
}

/// The long-time machinery from rho0: one Krylov propagation (on
/// `switch_grid`, holding the output grid of step `step` if given) to its
/// reach, the switch time t_s the earliest of reach/8, reach/4, reach/2 and
/// the reach whose expansion is accepted, then the slow expansion of the
/// steady solution (its slow modes for the long-time purpose). The
/// propagation takes its own order; its fields are carried to the steady
/// modes, a part left out counted as outside the slow modes, and the fit
/// uses the modes they cover (the slow modes supply the others).
pub fn long_time(prepared:&LatticePrepared,initial:&[C],groups:&[Vec<usize>],steady:&SteadySolution,step:Option<f64>)->Result<LongTime,Declined>{
    let failed=Declined::Failed;
    let readout=Readout::new(prepared,&steady.modes).map_err(failed)?;let modes=&steady.slow.modes;let lapack=&prepared.lapack;
    let relative=|x:&[C]|->Vec<C>{x.iter().zip(&steady.field).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect()};
    // The propagation's share: what the steady estimate and the expansion's
    // own terms (FAST_SHARE, KAPPA-weighted, and a tenth for the residuals
    // and the projection) leave.
    let budget=0.9*(POPULATION_TOLERANCE-steady.estimated_error-KAPPA*FAST_SHARE*POPULATION_TOLERANCE);
    if !(budget>0.1*POPULATION_TOLERANCE){return Err(failed(format!("The steady estimate {:e} leaves the propagation no room",steady.estimated_error)));}
    let mut reach=REACH_START;let mut trail=Vec::new();
    loop{
        let times=switch_grid(reach,step);let(last,h)=(times.len()-1,times[1]);
        let mut candidates:Vec<usize>=[8.,4.,2.,1.].iter().map(|f|((times[last]/f/h).round()as usize).min(last))
            .filter(|&k|k>=2&&times[k]>=SWITCH_START*(1.-1e-9)).collect();
        candidates.dedup();
        let mut keep:Vec<usize>=candidates.iter().flat_map(|&k|[k/2,k]).collect();keep.sort_unstable();keep.dedup();
        let krylov=propagate_within(prepared,initial,&times,groups,None,&keep,budget).map_err(failed)?;
        let from=prepared.modes(krylov.order).map_err(failed)?;
        let fields:Vec<(Vec<C>,Vec<bool>,f64)>=keep.iter().zip(&krylov.fields).map(|(&k,x)|transfer(prepared,x,&from,&steady.modes,times[k],groups))
            .collect::<Result<_,_>>().map_err(failed)?;
        let field=|k:usize|&fields[keep.binary_search(&k).unwrap_or(0)];let d=prepared.generator.n.pow(2);
        for &k in &candidates{
            let switch=times[k];let(at,covered,left)=field(k);let y=relative(at);let(coefficients,outside)=fit(lapack,modes,&y,covered).map_err(failed)?;
            // The fit at t_s / 2 evolved to t_s against the propagation.
            let(early,_)=fit(lapack,modes,&relative(&field(k/2).0),covered).map_err(failed)?;
            let mut drift=masked(&y,covered,d);let mut predicted=vec![ZERO;y.len()];evolve(modes,&early,switch-times[k/2],&mut predicted);
            for(a,b)in drift.iter_mut().zip(&masked(&predicted,covered,d)){a.re-=b.re;a.im-=b.im;}
            let(fast,drift)=(readout.size(&outside,switch,groups)+left,readout.size(&drift,switch,groups));
            if fast+drift<=FAST_SHARE*POPULATION_TOLERANCE{
                let residuals=modes.iter().zip(&coefficients).map(|(m,c)|{let mut part=vec![ZERO;y.len()];
                    evolve(std::slice::from_ref(m),std::slice::from_ref(c),0.,&mut part);m.residual*readout.size(&part,switch,groups)}).sum();
                return Ok(LongTime{switch,krylov,times,steady:steady.field.clone(),modes:modes.clone(),coefficients,readout,
                    fast,drift,residuals,artifact_growth:steady.slow.artifact_growth,steady_error:steady.estimated_error,trail});
            }
            trail.push((switch,fast+drift));
        }
        let trail_text=||trail.iter().map(|(t,e)|format!("{t:.0} us: {e:.1e}")).collect::<Vec<_>>().join(", ");
        // An error that does not shrink with t_s (to half over the first
        // reaches) is no decaying fast part: slow modes the shift-invert set
        // misses (precessing ones, |lambda| large), which a longer reach
        // cannot fix.
        if reach<=4.*REACH_START&&trail.len()>2&&trail[trail.len()-1].1>0.5*trail[0].1{
            return Err(Declined::MissedModes(format!("The slow expansion's error does not decay with the switch time ({}): slow modes outside the shift-invert set",trail_text())));}
        if times[last]>=REACH_LIMIT{
            return Err(failed(format!("The slow expansion stays off by more than {:e} up to {:.0} us ({})",FAST_SHARE*POPULATION_TOLERANCE,times[last],trail_text())));}
        reach=4.*times[last];
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_finite_populations_follow_the_exact_lattice_propagation(){
        // The fixture from |0><0| over 2 us against the exact exponential of
        // the same order-2 lattice (dense, real 2N x 2N form): the actual
        // error stays below the claimed one.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let groups=vec![vec![0,1],vec![2,3]];
        let mut initial=vec![ZERO;d];initial[0]=ONE;
        let times:Vec<f64>=(0..21).map(|i|0.1*i as f64).collect();
        let s=propagate(&p,&initial,&times,&groups,None,&[]).unwrap();
        let modes=p.modes(s.order).unwrap();let op=LatticeOperator::new(&p,&modes).unwrap();
        let a=op.dense();let size=op.dimension();let real=2*size;
        let mut big=vec![0.;real*real];
        for r in 0..size{for c in 0..size{let v=a[r*size+c];
            big[r+real*c]=v.re;big[r+real*(c+size)]=-v.im;big[r+size+real*c]=v.im;big[r+size+real*(c+size)]=v.re;}}
        let step=crate::dense_expm::exponential(&p.lapack,&big,real,0.1).unwrap();
        let mut x=vec![0.;real];x[op.centre*d]=1.;
        let mut worst=0_f64;
        for(i,&t)in times.iter().enumerate(){
            if i>0{x=(0..real).map(|r|(0..real).map(|c|step[r+real*c]*x[c]).sum()).collect();}
            let pops:Vec<f64>=(0..n).map(|k|(0..modes.len()).map(|q|{let e=q*d+k*n+k;
                let v=C{re:x[e],im:x[e+size]};mul(C{re:(op.nu[q]*t).cos(),im:(op.nu[q]*t).sin()},v).re}).sum()).collect();
            worst=worst.max(ledger::metric(&pops,&s.populations[i],&groups));
        }
        assert!(worst<=s.estimated_error,"actual {worst:e} above the estimate {:e} ({:?})",s.estimated_error,s.checkpoints);
        assert!(s.estimated_error<=POPULATION_TOLERANCE);
    }
    #[test]
    fn lattice_long_time_expansion_follows_the_exact_lattice_propagation(){
        // The fixture from |0><0|: the slow expansion after the switch time
        // against the exact exponential of the steady solution's lattice,
        // within its claimed error, and its state at 1 s the steady field.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let groups=vec![vec![0,1],vec![2,3]];
        let mut initial=vec![ZERO;d];initial[0]=ONE;
        let mut steady=super::super::steady::solve(&p,&groups,super::super::slow::Purpose::LongTime).unwrap();
        // The reference is the same lattice, whose null vector the steady
        // field is to solver precision: its fold charges (a bound against
        // the unfolded model, 2.9e-3 here) would only crowd the budget.
        steady.estimated_error=0.;
        let long=long_time(&p,&initial,&groups,&steady,None).unwrap();
        let op=LatticeOperator::new(&p,&steady.modes).unwrap();
        let a=op.dense();let size=op.dimension();let real=2*size;
        let mut big=vec![0.;real*real];
        for r in 0..size{for c in 0..size{let v=a[r*size+c];
            big[r+real*c]=v.re;big[r+real*(c+size)]=-v.im;big[r+size+real*c]=v.im;big[r+size+real*(c+size)]=v.re;}}
        let step=long.switch/4.;let propagator=crate::dense_expm::exponential(&p.lapack,&big,real,step).unwrap();
        let mut x=vec![0.;real];x[op.centre*d]=1.;
        let mut compared=0;
        for k in 1..=20{
            x=(0..real).map(|r|(0..real).map(|c|propagator[r+real*c]*x[c]).sum()).collect();
            let t=step*k as f64;if t<long.switch*(1.-1e-12){continue;}
            let pops:Vec<f64>=(0..n).map(|i|(0..steady.modes.len()).map(|q|{let e=q*d+i*n+i;
                let v=C{re:x[e],im:x[e+size]};mul(C{re:(op.nu[q]*t).cos(),im:(op.nu[q]*t).sin()},v).re}).sum()).collect();
            let error=ledger::metric(&pops,&long.populations(t),&groups);
            assert!(error<=long.error(t),"t {t}: actual {error:e} above the estimate {:e}",long.error(t));
            compared+=1;
        }
        assert!(compared>=10&&!long.modes.is_empty());
        let away=long.field(1e6).iter().zip(&steady.field).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0_f64,f64::max);
        assert!(away<1e-12,"{away:e}");
        assert!(long.relaxing(1e6,&groups)<1e-12);
    }
}
