//! The Schrieffer-Wolff / Floquet frame of a request, built once and shared
//! by the reduced model's inner product (`SwMetric`), the a-priori amplitudes
//! (`apriori`) and the slowest relaxation's classes (`pipeline`): the
//! first-order kernels X~_s of the explicit near steps in the
//! eigen-coordinates of L0' (`kernels`; elements above KERNEL_CAP are near
//! resonant and stay couplings) and one eigensystem D' = V diag(mu) V^-1 of
//! the second-order block L0' + sum_s K_s over every step (`blocks`), the
//! same for every charge.
//!
//! The preconditioner's Floquet kernels (`floquet`) are a different object:
//! separable n x n left/right frames without the recycling term, cheap to
//! apply on every block; the frame here is the superoperator (d x d) one
//! with the recycling inside L0' (the Raman rates).
//!
//! The metric: T = V^-1 (I - S~) E^-1 on a lattice field,
//! E^-1 the eigen-coordinates of L0' per block, S~ the kernels on every
//! lattice edge q - s -> q; the inner product Re <T x, T y> summed over
//! every mode, M = T^H T. The first-order couplings between the modes are
//! removed in this frame and its blocks are dissipative, so the Galerkin
//! model's numerical range is pushed right only at second order (and by
//! the near-resonant elements): no strict guarantee. The only provable
//! bound is the Hilbert-Schmidt one, Re theta <= Gamma/2 (the couplings are
//! skew there, the decay of populations is not normal). Why not the
//! Euclidean model: it grew spurious near-zero slow Ritz pairs after the
//! ladder's poles near 1 / T of a 1 s horizon (rates 1e-8 to 1e-5,
//! residuals 0.2-3), and four and five lasers + tweezer at 10 ms ended
//! partial (bound about 1e-2, the p(1 s) error itself 9e-3 to 1.2e-2 at
//! those checkpoints), where this metric accepted them at 16 and 12 solves.
//!
//! Hermitian symmetry: for x_-q = x_q^dagger the eigen-coordinates satisfy
//! c_-q = P conj(c_q) (P the conjugate partner of each coordinate), the
//! kernels X~_-s = P conj(X~_s) P and the real class matrix keeps it, so
//! T x and M x are Hermitian-symmetric as well: M x is computed on the
//! canonical modes, the mirrors' coordinates taken by the rule.
use super::*;
use super::eigenbasis::BlockEigenbasis;
use super::operator::LatticeOperator;
use super::blocks::BlockSystem;
use super::kernels::{KERNEL_CAP,KERNEL_ENTRY,KERNEL_HORIZON_RATE,KernelRule,SwKernels};
use crate::parallel::Execution;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicUsize,Ordering};

/// Fewest blocks per task of the metric's coordinate changes (each GEMM
/// reads its d x d matrix once per task).
const METRIC_BLOCKS:usize=16;
/// Columns of N per task.
const NORMAL_PANEL:usize=64;
/// The metric forms N = V^-H V^-1 (a d x d x d product) once it has
/// applied d / NORMAL_SHARE blocks with V^-1 and V^-H separately: at most
/// that share of N's cost goes to the second product, and a short request
/// never forms N.
const NORMAL_SHARE:usize=4;

/// The frame of a request, each part built on its first use (a part's
/// failure is kept and returned by every later use).
pub struct SwFrame<'p>{
    prepared:&'p LatticePrepared,rule:KernelRule,
    /// The longest output time it is built for (us).
    pub horizon:f64,
    /// The kernels; the second-order block over every step; N = V^-H V^-1
    /// (column-major d x d, eigen-coordinates of L0'); each with its build
    /// time (ms).
    kernels:OnceLock<Result<(SwKernels,f64),String>>,
    class:OnceLock<Result<(BlockSystem,f64),String>>,
    normal:OnceLock<Result<(Vec<C>,f64),String>>,
}
impl<'p> SwFrame<'p>{
    /// The frame for outputs up to `horizon` (us; the kernels keep elements
    /// whose second-order rate matters over it).
    pub fn new(prepared:&'p LatticePrepared,horizon:f64)->Self{
        let rule=KernelRule{entry:KERNEL_ENTRY,horizon_rate:KERNEL_HORIZON_RATE,horizon,cap:KERNEL_CAP};
        Self{prepared,rule,horizon,kernels:OnceLock::new(),class:OnceLock::new(),normal:OnceLock::new()}
    }
    pub fn kernels(&self)->Result<&SwKernels,String>{
        self.kernels.get_or_init(||{let clock=Instant::now();let basis=self.prepared.eigenbasis()?;
            Ok((SwKernels::new(self.prepared,basis,self.rule)?,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|&x.0).map_err(|e|e.clone())
    }
    /// The second-order block over every step.
    pub fn class(&self)->Result<&BlockSystem,String>{
        self.class.get_or_init(||{let kernels=self.kernels()?;let clock=Instant::now();let basis=self.prepared.eigenbasis()?;
            let class=BlockSystem::new(&self.prepared.lapack,basis,kernels,self.prepared.generator.n,self.horizon)?;
            Ok((class,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|&x.0).map_err(|e|e.clone())
    }
    /// N = V^-H V^-1 by column panels over the workers (`SwMetric`
    /// forms it once it pays).
    pub fn normal(&self)->Result<&[C],String>{
        self.normal.get_or_init(||{let class=self.class()?;let clock=Instant::now();
            let(d,inverse,lapack)=(self.prepared.generator.n*self.prepared.generator.n,&class.inverse,&self.prepared.lapack);
            let parts=Execution::new()?.map((0..d).step_by(NORMAL_PANEL).collect(),|start|->Result<Vec<C>,String>{
                let width=NORMAL_PANEL.min(d-start);let mut out=vec![ZERO;d*width];
                lapack.complex_gemm().multiply_adjoint(d,width,d,ONE,inverse,d,&inverse[start*d..],d,ZERO,&mut out,d)?;Ok(out)});
            let mut normal=Vec::with_capacity(d*d);for part in parts{normal.extend(part?);}
            Ok((normal,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|x.0.as_slice()).map_err(|e|e.clone())
    }
    /// Build times (ms) of the parts built so far: kernels, block, N.
    pub fn milliseconds(&self)->[f64;3]{
        let ms=|x:Option<f64>|x.unwrap_or(0.);
        [ms(self.kernels.get().and_then(|r|r.as_ref().ok()).map(|x|x.1)),ms(self.class.get().and_then(|r|r.as_ref().ok()).map(|x|x.1)),
            ms(self.normal.get().and_then(|r|r.as_ref().ok()).map(|x|x.1))]
    }
}

/// The frame's metric M = T^H T on a lattice's fields.
pub struct SwMetric<'f>{
    /// The frame, its kernels, V^-1 of the second-order block (column-major
    /// d x d) and the eigenbasis of L0'.
    frame:&'f SwFrame<'f>,kernels:&'f SwKernels,inverse:&'f [C],basis:&'f BlockEigenbasis,lapack:Arc<Lapack>,
    /// Blocks applied so far (`NORMAL_SHARE`).
    applied:AtomicUsize,
    d:usize,
    /// Per mode: its canonical slot (or that of its mirror) and whether it
    /// is the mirror.
    slot:Vec<(usize,bool)>,
    /// Canonical modes and their weights (2 for a pair, 1 for the centre).
    pub canonical:Vec<usize>,pub weights:Vec<f64>,
    /// Per canonical mode: the kernel edges into it (source mode, step) and
    /// out of it (target mode, step).
    into:Vec<Vec<(usize,usize)>>,out_of:Vec<Vec<(usize,usize)>>,
    partner:Vec<usize>,
}
impl<'f> SwMetric<'f>{
    pub fn new(frame:&'f SwFrame<'f>,basis:&'f BlockEigenbasis,op:&LatticeOperator<'_>)->Result<Self,String>{
        let(kernels,inverse)=(frame.kernels()?,frame.class()?.inverse.as_slice());
        let d=op.n*op.n;let modes=op.modes.len();
        let canonical:Vec<usize>=(0..modes).filter(|&q|op.canonical[q]).collect();
        let weights=canonical.iter().map(|&q|if op.mirror[q]==q{1.}else{2.}).collect();
        let mut slot=vec![(usize::MAX,false);modes];
        for(c,&q)in canonical.iter().enumerate(){slot[q]=(c,false);if op.mirror[q]!=q{slot[op.mirror[q]]=(c,true);}}
        let(mut into,mut out_of)=(vec![Vec::new();canonical.len()],vec![Vec::new();canonical.len()]);
        for(c,&q)in canonical.iter().enumerate(){for(s,step)in kernels.steps.iter().enumerate(){
            if let Some(&b)=op.index.get(&charge_minus(&op.modes[q],&step.charge)?){into[c].push((b,s));}
            if let Some(&a)=op.index.get(&charge_plus(&op.modes[q],&step.charge)?){out_of[c].push((a,s));}
        }}
        Ok(Self{frame,kernels,inverse,basis,applied:AtomicUsize::new(0),lapack:Arc::clone(&op.prepared.lapack),d,slot,canonical,weights,into,out_of,partner:(0..d).map(|j|basis.partner(j)).collect()})
    }
    /// A mode's block of per-canonical coordinates (the mirror's by the
    /// partner rule).
    fn block(&self,canonical:&[C],q:usize)->Vec<C>{
        let d=self.d;let(c,mirror)=self.slot[q];let b=&canonical[c*d..(c+1)*d];
        if mirror{(0..d).map(|j|conj(b[self.partner[j]])).collect()}else{b.to_vec()}
    }
    /// M x on the canonical modes (row-major blocks in `canonical` order,
    /// each times its weight, one field after the other) for
    /// Hermitian-symmetric fields: Re <y, M x> summed over every mode is
    /// the plain sum of Re <y_q, (weighted M x)_q> over the canonical ones.
    /// The coordinate changes and N (or V^-1 and V^-H before N is formed,
    /// `NORMAL_SHARE`) as GEMMs over chunks of blocks, a few per worker and
    /// at least METRIC_BLOCKS each.
    pub fn apply(&self,fields:&[&[C]])->Result<Vec<C>,String>{
        let(d,cc,m)=(self.d,self.canonical.len(),fields.len());
        if m==0{return Ok(Vec::new());}
        let execution=Execution::new()?;let kernels=self.kernels;let basis=self.basis;
        let blocks=m*cc;let width=blocks.div_ceil(execution.tasks(blocks,blocks,METRIC_BLOCKS));
        let chunks:Vec<(usize,usize)>=(0..blocks).step_by(width).map(|s|(s,(s+width).min(blocks))).collect();
        let gather=|parts:Vec<Result<Vec<C>,String>>|->Result<Vec<C>,String>{let mut out=Vec::with_capacity(d*m*cc);for p in parts{out.extend(p?);}Ok(out)};
        let coefficients=gather(execution.map(chunks.clone(),|(s,e)|{
            let blocks:Vec<&[C]>=(s..e).map(|i|{let(f,c)=(i/cc,i%cc);let q=self.canonical[c];&fields[f][q*d..(q+1)*d]}).collect();basis.coefficients(&blocks)}))?;
        let items:Vec<(usize,usize)>=(0..m).flat_map(|f|(0..cc).map(move|c|(f,c))).collect();
        // t = (I - S~) c on the canonical modes.
        let t:Vec<Vec<C>>=execution.map(items.clone(),|(f,c)|{let field=&coefficients[f*cc*d..(f+1)*cc*d];
            let mut t=field[c*d..(c+1)*d].to_vec();for &(b,s)in &self.into[c]{kernels.subtract_kernel(s,&self.block(field,b),&mut t);}t});
        let t:Vec<C>=t.concat();
        // v = N t.
        let normal=if self.applied.fetch_add(blocks,Ordering::Relaxed)>=d/NORMAL_SHARE{Some(self.frame.normal()?)}else{None};
        let(inverse,lapack,t_ref)=(self.inverse,&self.lapack,&t);
        let v=gather(execution.map(chunks.clone(),|(s,e)|{let(gemm,width,t)=(lapack.complex_gemm(),e-s,&t_ref[s*d..e*d]);let mut out=vec![ZERO;d*width];
            match normal{
                Some(normal)=>gemm.multiply(d,width,d,ONE,normal,d,t,d,ZERO,&mut out,d)?,
                None=>{let mut u=vec![ZERO;d*width];gemm.multiply(d,width,d,ONE,inverse,d,t,d,ZERO,&mut u,d)?;
                    gemm.multiply_adjoint(d,width,d,ONE,inverse,d,&u,d,ZERO,&mut out,d)?;}
            }
            Ok(out)}))?;
        // w = (I - S~)^H v on the canonical modes.
        let w:Vec<Vec<C>>=execution.map(items,|(f,c)|{let field=&v[f*cc*d..(f+1)*cc*d];
            let mut w=field[c*d..(c+1)*d].to_vec();for &(a,s)in &self.out_of[c]{kernels.subtract_kernel_adjoint(s,&self.block(field,a),&mut w);}w});
        let w:Vec<C>=w.concat();let w_ref=&w;
        let mut out=gather(execution.map(chunks,|(s,e)|basis.coefficients_adjoint(&w_ref[s*d..e*d],e-s)))?;
        for(e,block)in out.chunks_exact_mut(d).enumerate(){let s=self.weights[e%cc];for z in block{*z=scaled(*z,s);}}
        Ok(out)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::operator::symmetric_field;
    #[test]
    fn frame_metric_is_t_adjoint_t_over_every_mode(){
        // Re <y, M x> from the canonical modes against |T| computed on every
        // mode directly (no mirror rule): T x_q = V^-1 (c_q - sum X~_s c_{q-s}).
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let op=LatticeOperator::new(&p,&p.fold,&modes).unwrap();let d=16;
        let frame=SwFrame::new(&p,10.);let basis=p.eigenbasis().unwrap();let(kernels,class)=(frame.kernels().unwrap(),frame.class().unwrap());
        let metric=SwMetric::new(&frame,basis,&op).unwrap();
        let(x,y)=(symmetric_field(&op,1),symmetric_field(&op,2));
        let t=|z:&[C]|->Vec<C>{let blocks:Vec<&[C]>=z.chunks_exact(d).collect();let c=basis.coefficients(&blocks).unwrap();
            let mut out=Vec::new();
            for(q,charge)in modes.iter().enumerate(){let mut t=c[q*d..(q+1)*d].to_vec();
                for(s,step)in kernels.steps.iter().enumerate(){if let Some(&b)=op.index.get(&charge_minus(charge,&step.charge).unwrap()){kernels.subtract_kernel(s,&c[b*d..(b+1)*d],&mut t);}}
                let mut z=vec![ZERO;d];p.lapack.complex_gemm().multiply(d,1,d,ONE,&class.inverse,d,&t,d,ZERO,&mut z,d).unwrap();out.extend(z);}
            out};
        let(tx,ty)=(t(&x),t(&y));let direct:f64=tx.iter().zip(&ty).map(|(a,b)|a.re*b.re+a.im*b.im).sum();
        let mx=metric.apply(&[&x]).unwrap();
        let reduced:f64=metric.canonical.iter().enumerate().map(|(c,&q)|(0..d).map(|k|{let(a,b)=(y[q*d+k],mx[c*d+k]);a.re*b.re+a.im*b.im}).sum::<f64>()).sum();
        let scale=tx.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()*ty.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        assert!(kernels.steps.iter().any(|s|!s.entries.is_empty()),"the fixture's kernels are empty");
        assert!((direct-reduced).abs()<=1e-11*scale,"{direct:e} against {reduced:e} (scale {scale:e})");
        // Positive: Re <x, M x> = |T x|^2.
        let xx:f64=metric.canonical.iter().enumerate().map(|(c,&q)|(0..d).map(|k|{let(a,b)=(x[q*d+k],mx[c*d+k]);a.re*b.re+a.im*b.im}).sum::<f64>()).sum();
        let norm:f64=tx.iter().map(|z|z.re*z.re+z.im*z.im).sum();
        assert!((xx-norm).abs()<=1e-11*norm,"{xx:e} against {norm:e}");
        // N once d / NORMAL_SHARE blocks are applied: the same metric.
        metric.apply(&vec![x.as_slice();d/NORMAL_SHARE]).unwrap();
        let again=metric.apply(&[&x]).unwrap();
        assert!(frame.normal.get().is_some(),"N not formed");
        let size=mx.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        assert!(mx.iter().zip(&again).all(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})<=1e-12*size),"N against V^-1 and V^-H");
    }
}
