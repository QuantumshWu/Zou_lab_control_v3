//! The dressed block preconditioner: per mode q an inverse of the dressed
//! block L0' - i nu_q, L0' = L0 + K_0^(0) (the light-shifted, scattering-
//! dressed generator; without the dressing the tweezer's second-order shifts
//! move the bare block spectrum far beyond the optical decay and a block
//! preconditioner does not converge).
//!
//! Structured: L0' X = NL' X + X NR' + R0(X) + Krest(X), with NL', NR' the
//! n x n factors carrying L0 and the fold's absorbable part, R0 the
//! recycling and Krest the fold rest. The separable part is inverted in the
//! eigenframes of NL' and NR' (elementwise in frame coordinates); R0 and the
//! rest's ground-ground rows and columns form a Woodbury term of rank 128
//! (the scattering into and out of the ground populations, where the rest is
//! largest). The rest's excited-excited and optical blocks (about 20 and 10
//! rad/us) are left to the Krylov solver. The centre carries the steady trace
//! row exactly (one more Woodbury column), and frame denominators at roundoff
//! (undamped ground populations at q = 0) are moved to -sigma and corrected
//! exactly by further Woodbury columns, so the centre block is inverted
//! exactly for the operator it represents.
//!
//! Hermitian half: the preconditioner preserves x_-q = x_q^dagger, so only the
//! canonical modes are computed and the rest mirrored.
use super::*;
use super::fold::{eigen_frame,inverse,sandwich_batch,transpose};
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Frame denominators below this fraction of the frame scale are at roundoff.
const TINY_DENOMINATOR:f64=1e-13;
/// Regularized denominators are moved to -SIGMA_FRACTION times the largest
/// frame damping.
const SIGMA_FRACTION:f64=1e-3;

/// A block preconditioner on a lattice field.
pub trait Preconditioner{
    /// y = P x for a Hermitian-symmetric field x.
    fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>;
    /// Complex multiply-adds of one application.
    fn work(&self)->usize;
}

#[derive(Clone,Debug,Default)]
pub struct StructuredDiagnostics{
    pub milliseconds:f64,
    /// Condition bounds and relative reconstruction errors of the frames.
    pub condition:[f64;2],pub reconstruction:[f64;2],
    pub woodbury_rank:usize,
}
/// The request-level part: frames and Woodbury factors in frame coordinates.
pub struct StructuredDressed{
    n:usize,d:usize,lapack:Arc<Lapack>,
    /// Row-major frame transforms: hat(Y) = left_in Y right_in,
    /// X = left_out hat(X) right_out.
    left_in:Vec<C>,right_in:Vec<C>,left_out:Vec<C>,right_out:Vec<C>,
    lambda:Vec<C>,mu:Vec<C>,
    /// Woodbury U and V in frame coordinates: d x rank and rank x d,
    /// column-major.
    uhat:Vec<C>,vhat:Vec<C>,rank:usize,
    /// The trace row u v^T (u = I/n, v = trace) in frame coordinates.
    trace_u:Vec<C>,trace_v:Vec<C>,
    tiny:f64,sigma:f64,
    pub diagnostics:StructuredDiagnostics,
}
/// One mode's shift-dependent factors.
struct Factor{
    reciprocal:Vec<C>,
    /// Extra Woodbury columns (frame U, d x e) and rows (frame V, e x d).
    extra_u:Vec<C>,extra_v:Vec<C>,extra:usize,
    /// (I + V' diag(reciprocal) U')^-1, column-major.
    capacitance:Vec<C>,
}
fn frame_transform(lapack:&Lapack,n:usize,a:&[C],b:&[C],blocks:&mut[C],m:usize)->Result<(),String>{
    let mut stack=Vec::new();sandwich_batch(&lapack.complex_gemm(),n,a,b,blocks,m,&mut stack)
}
impl StructuredDressed{
    pub fn new(prepared:&LatticePrepared)->Result<Self,String>{
        let clock=Instant::now();let lapack=&prepared.lapack;let generator=&prepared.generator;
        let n=generator.n;let d=n*n;let zero=generator.zero_component();
        let diagonal=prepared.fold.targets.first().filter(|t|t.charge==generator.zero);
        let(mut nl,mut nr)=(zero.op.left.dense(n),zero.op.right.dense(n));
        if let Some(t)=diagonal{for k in 0..d{add(&mut nl[k],t.left[0][k]);add(&mut nr[k],t.right[0][k]);}}
        // W = R0 + Krest (row-major d x d).
        let mut w=vec![ZERO;d*d];
        Superop{left:Sparse::default(),right:Sparse::default(),sandwich:zero.op.sandwich.clone()}.dense_into(n,ONE,&mut w);
        if let Some(t)=diagonal{t.rest[0].dense_into(d,ONE,&mut w);}
        let positions:Vec<usize>=(0..d).filter(|&p|generator.ground[p/n]&&generator.ground[p%n]).collect();
        let r=positions.len();let rank=2*r;let ground_pair=|p:usize|generator.ground[p/n]&&generator.ground[p%n];
        // U = [injection at the ground pairs | W's ground-pair columns off those rows],
        // V = [W's ground-pair rows ; selection of the ground pairs].
        let mut u=vec![ZERO;d*rank];let mut v_rows=vec![ZERO;rank*d];
        for(k,&p)in positions.iter().enumerate(){
            u[k*d+p]=ONE;
            for row in 0..d{if !ground_pair(row){u[(r+k)*d+row]=w[row*d+p];}}
            v_rows[k*d..(k+1)*d].copy_from_slice(&w[p*d..(p+1)*d]);
            v_rows[(r+k)*d+p]=ONE;
        }
        drop(w);
        let(lambda,pl,plinv,ca,ra)=eigen_frame(lapack,&nl,n)?;let(mu,q,qinv,cb,rb)=eigen_frame(lapack,&nr,n)?;
        let(left_in,right_in,left_out,right_out)=(plinv,q,pl,qinv);
        let mut uhat=u;frame_transform(lapack,n,&left_in,&right_in,&mut uhat,rank)?;
        // V vec(T_out(X)) = sum_ij V_ij (P X Q^-1)_ij = vec(P^T V Q^-T) . vec(X).
        let(at,bt)=(transpose(&left_out,n),transpose(&right_out,n));
        frame_transform(lapack,n,&at,&bt,&mut v_rows,rank)?;
        let mut vhat=vec![ZERO;rank*d];for k in 0..rank{for c in 0..d{vhat[k+rank*c]=v_rows[k*d+c];}}
        let mut trace_u=vec![ZERO;d];let mut trace_v=vec![ZERO;d];for i in 0..n{trace_u[i*n+i]=C{re:1./n as f64,im:0.};trace_v[i*n+i]=ONE;}
        frame_transform(lapack,n,&left_in,&right_in,&mut trace_u,1)?;frame_transform(lapack,n,&at,&bt,&mut trace_v,1)?;
        let scale=lambda.iter().map(|z|magnitude(*z)).fold(0.,f64::max)+mu.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let damping=lambda.iter().map(|z|z.re.abs()).fold(0.,f64::max)+mu.iter().map(|z|z.re.abs()).fold(0.,f64::max);
        let diagnostics=StructuredDiagnostics{milliseconds:clock.elapsed().as_secs_f64()*1e3,condition:[ca,cb],reconstruction:[ra,rb],woodbury_rank:rank};
        Ok(Self{n,d,lapack:Arc::clone(lapack),left_in,right_in,left_out,right_out,lambda,mu,uhat,vhat,rank,trace_u,trace_v,
            tiny:TINY_DENOMINATOR*scale,sigma:SIGMA_FRACTION*damping,diagnostics})
    }
    /// Factors of mode shift i nu; the centre adds the trace row.
    fn factor(&self,nu:f64,centre:bool)->Result<(Factor,usize),String>{
        let(n,d,rank)=(self.n,self.d,self.rank);
        let mut reciprocal=vec![ZERO;d];let mut moved=Vec::new();
        for a in 0..n{for b in 0..n{let z=C{re:self.lambda[a].re+self.mu[b].re,im:self.lambda[a].im+self.mu[b].im-nu};
            let(z,shift)=if magnitude(z)<self.tiny{(C{re:-self.sigma,im:0.},Some(z))}else{(z,None)};
            if let Some(original)=shift{moved.push((a*n+b,C{re:original.re+self.sigma,im:original.im}));}
            let s=z.re*z.re+z.im*z.im;reciprocal[a*n+b]=C{re:z.re/s,im:-z.im/s};}}
        let extra=usize::from(centre)+moved.len();
        let mut extra_u=vec![ZERO;d*extra];let mut extra_v=vec![ZERO;extra*d];let mut e=0;
        if centre{extra_u[..d].copy_from_slice(&self.trace_u);for c in 0..d{extra_v[e+extra*c]=self.trace_v[c];}e+=1;}
        for &(p,delta)in &moved{extra_u[e*d+p]=ONE;extra_v[e+extra*p]=delta;e+=1;}
        // G = I + V' diag(reciprocal) U'.
        let total=rank+extra;let gemm=self.lapack.complex_gemm();
        let mut scaled=vec![ZERO;d*total];
        for k in 0..rank{for c in 0..d{scaled[k*d+c]=mul(self.uhat[k*d+c],reciprocal[c]);}}
        for k in 0..extra{for c in 0..d{scaled[(rank+k)*d+c]=mul(extra_u[k*d+c],reciprocal[c]);}}
        let mut g=vec![ZERO;total*total];for k in 0..total{g[k+total*k]=ONE;}
        let mut top=vec![ZERO;rank*total];gemm.multiply(rank,total,d,ONE,&self.vhat,rank,&scaled,d,ZERO,&mut top,rank)?;
        for j in 0..total{for i in 0..rank{add(&mut g[i+total*j],top[i+rank*j]);}}
        if extra>0{let mut bottom=vec![ZERO;extra*total];gemm.multiply(extra,total,d,ONE,&extra_v,extra,&scaled,d,ZERO,&mut bottom,extra)?;
            for j in 0..total{for i in 0..extra{add(&mut g[rank+i+total*j],bottom[i+extra*j]);}}}
        let capacitance=inverse(&self.lapack,&g,total)?;
        Ok((Factor{reciprocal,extra_u,extra_v,extra,capacitance},moved.len()))
    }
    /// The preconditioner of one mode set: frequencies, the computed modes
    /// (the canonical half, or a subset of it), the Hermitian pairing and the
    /// centre (None for a set without the zero charge).
    pub fn batch<'a>(&'a self,nu:&[f64],canonical:&[bool],mirror:&[usize],centre:Option<usize>)->Result<StructuredBatch<'a>,String>{
        let execution=Execution::new()?;
        let computed:Vec<usize>=(0..nu.len()).filter(|&q|canonical[q]).collect();
        let factors=execution.map(computed.clone(),|q|self.factor(nu[q],Some(q)==centre)).into_iter().collect::<Result<Vec<_>,_>>()?;
        let regularized=factors.iter().map(|f|f.1).sum();let factors:Vec<Factor>=factors.into_iter().map(|f|f.0).collect();
        let(n,d)=(self.n,self.d);
        let work=factors.iter().map(|f|4*n*n*n+2*(self.rank+f.extra)*d+(self.rank+f.extra).pow(2)).sum();
        Ok(StructuredBatch{dressed:self,computed,factors,canonical:canonical.to_vec(),mirror:mirror.to_vec(),
            buffer:Vec::new(),execution,scratch:Vec::new(),regularized,work})
    }
}

#[derive(Default)]
pub struct BatchScratch{w:Vec<C>,c:Vec<C>,t:Vec<C>}
pub struct StructuredBatch<'a>{
    dressed:&'a StructuredDressed,
    /// Canonical modes and their factors.
    computed:Vec<usize>,factors:Vec<Factor>,
    canonical:Vec<bool>,mirror:Vec<usize>,
    buffer:Vec<C>,execution:Execution,scratch:Vec<BatchScratch>,
    /// Frame denominators at roundoff moved to -sigma (corrected exactly).
    pub regularized:usize,
    pub work:usize,
}
impl Preconditioner for StructuredBatch<'_>{
    fn work(&self)->usize{self.work}
    fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let dressed=self.dressed;let(n,d,rank)=(dressed.n,dressed.d,dressed.rank);let modes=self.canonical.len();
        if x.len()!=modes*d||y.len()!=x.len(){return Err("Dressed preconditioner field dimensions differ".into());}
        let m=self.computed.len();grow(&mut self.buffer,m*d);
        for(k,&q)in self.computed.iter().enumerate(){self.buffer[k*d..(k+1)*d].copy_from_slice(&x[q*d..(q+1)*d]);}
        let gemm=dressed.lapack.complex_gemm();let factors=&self.factors;let lapack=&dressed.lapack;
        self.execution.for_mode_chunks(&mut self.buffer[..m*d],d,self.work,MINIMUM_WORK_PER_TASK,&mut self.scratch,|first,chunk,scratch|{
            let count=chunk.len()/d;
            frame_transform(lapack,n,&dressed.left_in,&dressed.right_in,chunk,count)?;
            for(k,block)in chunk.chunks_exact_mut(d).enumerate(){for(v,&r)in block.iter_mut().zip(&factors[first+k].reciprocal){*v=mul(*v,r);}}
            grow(&mut scratch.w,rank*count);grow(&mut scratch.t,d*count);
            gemm.multiply(rank,count,d,ONE,&dressed.vhat,rank,chunk,d,ZERO,&mut scratch.w,rank)?;
            let total_max=factors[first..first+count].iter().map(|f|rank+f.extra).max().unwrap_or(rank);grow(&mut scratch.c,total_max*count);
            for k in 0..count{let f=&factors[first+k];let total=rank+f.extra;let block=&chunk[k*d..(k+1)*d];
                let mut w=scratch.w[k*rank..(k+1)*rank].to_vec();
                for e in 0..f.extra{let mut s=ZERO;for c in 0..d{mul_add(&mut s,f.extra_v[e+f.extra*c],block[c]);}w.push(s);}
                let c=&mut scratch.c[k*total_max..k*total_max+total];c.fill(ZERO);
                for j in 0..total{for i in 0..total{mul_add(&mut c[i],f.capacitance[i+total*j],w[j]);}}}
            // t = U c (base columns as one GEMM, extra columns per mode).
            let mut base=vec![ZERO;rank*count];for k in 0..count{base[k*rank..(k+1)*rank].copy_from_slice(&scratch.c[k*total_max..k*total_max+rank]);}
            gemm.multiply(d,count,rank,ONE,&dressed.uhat,d,&base,rank,ZERO,&mut scratch.t,d)?;
            for k in 0..count{let f=&factors[first+k];let t=&mut scratch.t[k*d..(k+1)*d];
                for e in 0..f.extra{let ce=scratch.c[k*total_max+rank+e];for c in 0..d{mul_add(&mut t[c],f.extra_u[e*d+c],ce);}}
                for(v,(&tv,&r))in chunk[k*d..(k+1)*d].iter_mut().zip(t.iter().zip(&f.reciprocal)){let s=mul(tv,r);v.re-=s.re;v.im-=s.im;}}
            frame_transform(lapack,n,&dressed.left_out,&dressed.right_out,chunk,count)
        })?;
        for(k,&q)in self.computed.iter().enumerate(){y[q*d..(q+1)*d].copy_from_slice(&self.buffer[k*d..(k+1)*d]);}
        // Mirrored modes (zero when the partner is outside the batch's mode
        // subset), the centre Hermitian.
        mirror_field(y,n,&self.canonical,&self.mirror);
        if y.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()){return Err("Nonfinite dressed preconditioner action".into());}
        Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::operator::{LatticeOperator,symmetric_field};
    #[test]
    fn lattice_structured_preconditioner_inverts_its_block_model_exactly(){
        // Every block of the preconditioner is the exact inverse of
        // NL' X + X NR' + R0 + (rest on ground-pair rows and columns) - i nu
        // (+ the trace row at the centre), assembled densely.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;
        let dressed=StructuredDressed::new(&p).unwrap();
        let modes=p.modes(2).unwrap();let op=LatticeOperator::new(&p,&modes).unwrap();
        let mut batch=dressed.batch(&op.nu,&op.canonical,&op.mirror,Some(op.centre)).unwrap();
        let x=symmetric_field(&op,3);let mut y=vec![ZERO;x.len()];batch.apply(&x,&mut y).unwrap();
        let zero=p.generator.zero_component();let t=&p.fold.targets[0];
        let mut base=vec![ZERO;d*d];zero.op.dense_into(n,ONE,&mut base);
        let absorbed=Superop{left:Sparse::from_dense(&t.left[0],n),right:Sparse::from_dense(&t.right[0],n),sandwich:vec![]};absorbed.dense_into(n,ONE,&mut base);
        let mut rest=vec![ZERO;d*d];t.rest[0].dense_into(d,ONE,&mut rest);
        let ground=|q:usize|p.generator.ground[q/n]&&p.generator.ground[q%n];
        for r in 0..d{for c in 0..d{if ground(r)||ground(c){add(&mut base[r*d+c],rest[r*d+c]);}}}
        for(q,&w)in op.nu.iter().enumerate(){
            let yq=&y[q*d..(q+1)*d];let mut back=vec![ZERO;d];
            for r in 0..d{for c in 0..d{mul_add(&mut back[r],base[r*d+c],yq[c]);}mul_add(&mut back[r],C{re:0.,im:-w},yq[r]);}
            if q==op.centre{let tr=(0..n).fold(ZERO,|s,i|C{re:s.re+yq[i*n+i].re,im:s.im+yq[i*n+i].im});for i in 0..n{add(&mut back[i*n+i],scaled(tr,1./n as f64));}}
            let scale=x[q*d..(q+1)*d].iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            let error=back.iter().zip(&x[q*d..(q+1)*d]).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
            assert!(error<1e-10*scale,"mode {q}: {error:e}");
        }
    }
}
