//! The Schrieffer-Wolff / Floquet frame of a request, built once and shared
//! by the reduced model's inner product (`SwMetric`) and the error estimate
//! (`estimator`): the first-order kernels X~_s of the explicit near steps in
//! the eigen-coordinates of L0' (`estimator::kernels`; elements above
//! KERNEL_CAP are near resonant and stay couplings) and one eigensystem
//! D' = V diag(mu) V^-1 of the second-order block L0' + sum_s K_s over every
//! step (`estimator::blocks`), the same for every charge.
//!
//! The preconditioner's Floquet kernels (`floquet`) are a different object:
//! separable n x n left/right frames without the recycling term, cheap to
//! apply on every block; the frame here is the superoperator (d x d) one
//! with the recycling inside L0' (the Raman rates).
//!
//! The metric (N1, round four): T = V^-1 (I - S~) E^-1 on a lattice field,
//! E^-1 the eigen-coordinates of L0' per block, S~ the kernels on every
//! lattice edge q - s -> q; the inner product Re <T x, T y> summed over
//! every mode, M = T^H T. The first-order couplings between the modes are
//! removed in this frame and its blocks are dissipative, so the Galerkin
//! model's numerical range is pushed right only at second order (and by
//! the near-resonant elements): no strict guarantee. The only provable
//! bound is the Hilbert-Schmidt one, Re theta <= Gamma/2 (the couplings are
//! skew there, the decay of populations is not normal).
//!
//! Hermitian symmetry: for x_-q = x_q^dagger the eigen-coordinates satisfy
//! c_-q = P conj(c_q) (P the conjugate partner of each coordinate), the
//! kernels X~_-s = P conj(X~_s) P and the real class matrix keeps it, so
//! T x and M x are Hermitian-symmetric as well: M x is computed on the
//! canonical modes, the mirrors' coordinates taken by the rule.
use super::*;
use super::eigenbasis::BlockEigenbasis;
use super::operator::LatticeOperator;
use super::estimator::{KERNEL_CAP,KERNEL_ENTRY,KERNEL_HORIZON_RATE};
use super::estimator::blocks::BlockSystem;
use super::estimator::kernels::{KernelRule,SwKernels};
use crate::parallel::Execution;
use std::sync::OnceLock;

/// Columns of N per task.
const NORMAL_PANEL:usize=64;
/// Blocks per task of the metric's coordinate changes.
const METRIC_BLOCKS:usize=16;

/// A batch of fields in the frame: M x on the canonical modes (each block
/// times its mode's weight), the eigen-coordinates E^-1 x and their
/// transfer (I - S~) E^-1 x on the canonical modes (field after field,
/// blocks in `SwMetric::canonical` order).
pub struct FrameFields{pub metric:Vec<C>,pub eigen:Vec<C>,pub transferred:Vec<C>}

/// The frame of a request, each part built on its first use (a part's
/// failure is kept and returned by every later use); a request that
/// propagates nothing (`pipeline`)
/// reads nothing of it.
pub struct SwFrame<'p>{
    prepared:&'p LatticePrepared,rule:KernelRule,
    /// The longest output time it is built for (us).
    pub horizon:f64,
    /// The explicit steps' charges in plan order: the kernels' steps.
    pub steps:Vec<Charge>,
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
        let plan=&prepared.plan;let steps=plan.explicit().into_iter().map(|i|plan.steps[i].charge.clone()).collect();
        Self{prepared,rule,horizon,steps,kernels:OnceLock::new(),class:OnceLock::new(),normal:OnceLock::new()}
    }
    pub fn kernels(&self)->Result<&SwKernels,String>{
        self.kernels.get_or_init(||{let clock=Instant::now();let basis=self.prepared.eigenbasis()?;
            Ok((SwKernels::new(self.prepared,basis,self.rule)?,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|&x.0).map_err(|e|e.clone())
    }
    /// The second-order block over every step.
    pub fn class(&self)->Result<&BlockSystem,String>{
        self.class.get_or_init(||{let kernels=self.kernels()?;let clock=Instant::now();let basis=self.prepared.eigenbasis()?;
            let class=BlockSystem::new(&self.prepared.lapack,basis,kernels,(0..kernels.steps.len()).collect(),self.prepared.generator.n,self.horizon)?;
            Ok((class,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|&x.0).map_err(|e|e.clone())
    }
    /// N = V^-H V^-1 by column panels over the workers.
    pub fn normal(&self)->Result<&[C],String>{
        self.normal.get_or_init(||{let class=self.class()?;let clock=Instant::now();
            let d=self.prepared.generator.n*self.prepared.generator.n;
            let inv=&class.inverse;let mut adjoint=vec![ZERO;d*d];for j in 0..d{for i in 0..d{adjoint[j+d*i]=conj(inv[i+d*j]);}}
            let panels:Vec<usize>=(0..d).step_by(NORMAL_PANEL).collect();let(adjoint,lapack)=(&adjoint,&self.prepared.lapack);
            let parts=Execution::new()?.map(panels,|start|->Result<Vec<C>,String>{let width=NORMAL_PANEL.min(d-start);let mut out=vec![ZERO;d*width];
                lapack.complex_gemm().multiply(d,width,d,ONE,adjoint,d,&inv[start*d..],d,ZERO,&mut out,d)?;Ok(out)});
            let mut normal=Vec::with_capacity(d*d);for part in parts{normal.extend(part?);}
            Ok((normal,clock.elapsed().as_secs_f64()*1e3))}).as_ref().map(|x|x.0.as_slice()).map_err(|e|e.clone())
    }
    /// Every part (the projection's metric reads all three).
    pub fn build(&self)->Result<(),String>{self.normal().map(|_|())}
    /// Build times (ms) of the parts built so far: kernels, block, N.
    pub fn milliseconds(&self)->[f64;3]{
        let ms=|x:Option<f64>|x.unwrap_or(0.);
        [ms(self.kernels.get().and_then(|r|r.as_ref().ok()).map(|x|x.1)),ms(self.class.get().and_then(|r|r.as_ref().ok()).map(|x|x.1)),ms(self.normal.get().and_then(|r|r.as_ref().ok()).map(|x|x.1))]
    }
}

/// The frame's metric M = T^H T on a lattice's fields.
pub struct SwMetric<'f>{
    kernels:&'f SwKernels,normal:&'f [C],basis:&'f BlockEigenbasis,lapack:Arc<Lapack>,
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
    pub fn new(frame:&'f SwFrame<'_>,basis:&'f BlockEigenbasis,op:&LatticeOperator<'_>)->Result<Self,String>{
        let(kernels,normal)=(frame.kernels()?,frame.normal()?);
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
        Ok(Self{kernels,normal,basis,lapack:Arc::clone(&op.prepared.lapack),d,slot,canonical,weights,into,out_of,partner:(0..d).map(|j|basis.partner(j)).collect()})
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
    /// The coordinate changes and N as GEMMs over chunks of blocks on the
    /// workers.
    pub fn apply(&self,fields:&[&[C]])->Result<FrameFields,String>{
        let(d,cc,m)=(self.d,self.canonical.len(),fields.len());
        if m==0{return Ok(FrameFields{metric:Vec::new(),eigen:Vec::new(),transferred:Vec::new()});}
        let execution=Execution::new()?;let kernels=self.kernels;let basis=self.basis;
        let chunks:Vec<(usize,usize)>=(0..m*cc).step_by(METRIC_BLOCKS).map(|s|(s,(s+METRIC_BLOCKS).min(m*cc))).collect();
        let gather=|parts:Vec<Result<Vec<C>,String>>|->Result<Vec<C>,String>{let mut out=Vec::with_capacity(d*m*cc);for p in parts{out.extend(p?);}Ok(out)};
        let coefficients=gather(execution.map(chunks.clone(),|(s,e)|{
            let blocks:Vec<&[C]>=(s..e).map(|i|{let(f,c)=(i/cc,i%cc);let q=self.canonical[c];&fields[f][q*d..(q+1)*d]}).collect();basis.coefficients(&blocks)}))?;
        let items:Vec<(usize,usize)>=(0..m).flat_map(|f|(0..cc).map(move|c|(f,c))).collect();
        // t = (I - S~) c on the canonical modes.
        let t:Vec<Vec<C>>=execution.map(items.clone(),|(f,c)|{let field=&coefficients[f*cc*d..(f+1)*cc*d];
            let mut t=field[c*d..(c+1)*d].to_vec();for &(b,s)in &self.into[c]{kernels.subtract_kernel(s,&self.block(field,b),&mut t);}t});
        let t:Vec<C>=t.concat();
        let(normal,lapack,t_ref)=(self.normal,&self.lapack,&t);
        let v=gather(execution.map(chunks.clone(),|(s,e)|{let mut out=vec![ZERO;d*(e-s)];
            lapack.complex_gemm().multiply(d,e-s,d,ONE,normal,d,&t_ref[s*d..e*d],d,ZERO,&mut out,d)?;Ok(out)}))?;
        // w = (I - S~)^H v on the canonical modes.
        let w:Vec<Vec<C>>=execution.map(items,|(f,c)|{let field=&v[f*cc*d..(f+1)*cc*d];
            let mut w=field[c*d..(c+1)*d].to_vec();for &(a,s)in &self.out_of[c]{kernels.subtract_kernel_adjoint(s,&self.block(field,a),&mut w);}w});
        let w:Vec<C>=w.concat();let w_ref=&w;
        let mut out=gather(execution.map(chunks,|(s,e)|basis.coefficients_adjoint(&w_ref[s*d..e*d],e-s)))?;
        for(e,block)in out.chunks_exact_mut(d).enumerate(){let s=self.weights[e%cc];for z in block{*z=scaled(*z,s);}}
        Ok(FrameFields{metric:out,eigen:coefficients,transferred:t})
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
        let op=LatticeOperator::new(&p,&modes).unwrap();let d=16;
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
        let mx=metric.apply(&[&x]).unwrap().metric;
        let reduced:f64=metric.canonical.iter().enumerate().map(|(c,&q)|(0..d).map(|k|{let(a,b)=(y[q*d+k],mx[c*d+k]);a.re*b.re+a.im*b.im}).sum::<f64>()).sum();
        let scale=tx.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()*ty.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        assert!(kernels.steps.iter().any(|s|!s.entries.is_empty()),"the fixture's kernels are empty");
        assert!((direct-reduced).abs()<=1e-11*scale,"{direct:e} against {reduced:e} (scale {scale:e})");
        // Positive: Re <x, M x> = |T x|^2.
        let xx:f64=metric.canonical.iter().enumerate().map(|(c,&q)|(0..d).map(|k|{let(a,b)=(x[q*d+k],mx[c*d+k]);a.re*b.re+a.im*b.im}).sum::<f64>()).sum();
        let norm:f64=tx.iter().map(|z|z.re*z.re+z.im*z.im).sum();
        assert!((xx-norm).abs()<=1e-11*norm,"{xx:e} against {norm:e}");
    }
}
