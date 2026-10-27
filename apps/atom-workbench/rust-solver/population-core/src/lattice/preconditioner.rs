//! The dressed block preconditioner (L0' - i nu_q)^-1 with L0' = L0 + K_0^(0),
//! the light-shifted, scattering-dressed block of every mode. The dressing is
//! what makes the block preconditioner work: the tweezer's second-order
//! shifts move the bare block spectrum far beyond the optical decay.
//!
//! Validation scaffolding: `DenseDressed` is the exact dense version, one
//! eigendecomposition L0' = W diag(lambda) W^-1 of the d x d block, applied
//! as two row-split GEMMs per application. The structured version (n x n
//! frames plus a Woodbury term, as `fold::ShiftedL0`) replaces it.
//!
//! The centre carries the steady trace row exactly: (L0' + u v^T)^-1 is a
//! Woodbury correction in the eigenframe of rank one plus the eigenvalues at
//! roundoff (the trace mode), which are first moved to -sigma. Any other
//! mode's shifted eigenvalue at roundoff is moved to -sigma (counted); the
//! lattice operator itself is never changed, only its preconditioner.
use super::*;
use super::fold::inverse;
use crate::parallel::Execution;

/// Eigenvalues below this fraction of the largest are at roundoff level.
const TINY_EIGENVALUE:f64=1e-13;
/// Regularized eigenvalues are moved to -SIGMA_FRACTION * max |Re lambda|.
const SIGMA_FRACTION:f64=1e-3;
/// Condition bound of the central Woodbury matrix above which the centre
/// keeps only the rank-one trace term.
const CENTRAL_CONDITION_LIMIT:f64=1e12;

#[derive(Clone,Debug,Default)]
pub struct DressedDiagnostics{
    pub milliseconds:f64,
    /// ||W||_1 ||W^-1||_1 and max |L0' W - W Lambda| / max |L0'|.
    pub condition_bound:f64,pub reconstruction:f64,
    pub tiny:f64,pub sigma:f64,
    /// Smallest |lambda| above roundoff: the slowest dressed-block rate.
    pub slowest_rate:f64,
    pub central_rank:usize,pub central:&'static str,
}
pub struct DenseDressed{
    d:usize,lapack:Arc<Lapack>,
    /// Column-major W and W^-1.
    w:Vec<C>,w_inverse:Vec<C>,
    pub lambda:Vec<C>,
    lambda_regular:Vec<C>,
    /// W^-1 u and W^T v for the trace row (u = I/n, v = trace).
    a:Vec<C>,b:Vec<C>,
    /// Lambda_reg^-1 U and the inverse of I + V^T Lambda_reg^-1 U, with
    /// U = [a, e_k], V = [b, (lambda_k - lambda_reg_k) e_k] over `moved`.
    moved:Vec<usize>,central_u:Vec<C>,central_inverse:Vec<C>,
    pub diagnostics:DressedDiagnostics,
}
impl DenseDressed{
    pub fn new(prepared:&LatticePrepared)->Result<Self,String>{
        let clock=Instant::now();let lapack=&prepared.lapack;
        let n=prepared.generator.n;let d=n*n;
        let mut block=vec![ZERO;d*d];prepared.generator.zero_component().op.dense_into(n,ONE,&mut block);
        if let Some(k0)=&prepared.fold.zero_dense{for(a,&b)in block.iter_mut().zip(k0){add(a,b);}}
        // Column-major data of the row-major block.
        let mut column=vec![ZERO;d*d];for r in 0..d{for c in 0..d{column[r+d*c]=block[r*d+c];}}drop(block);
        let(lambda,w)=lapack.complex_eigen_right(&column,d)?;
        let w_inverse=inverse(lapack,&w,d)?;
        let one=|m:&[C]|(0..d).map(|c|m[c*d..(c+1)*d].iter().map(|z|magnitude(*z)).sum::<f64>()).fold(0.,f64::max);
        let condition_bound=one(&w)*one(&w_inverse);
        let product=lapack.multiply_complex(&column,&w,d,d,d)?;
        let scale=column.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let mut reconstruction=0_f64;
        for k in 0..d{for r in 0..d{let v=product[r+d*k];let e=mul(w[r+d*k],lambda[k]);reconstruction=reconstruction.max(magnitude(C{re:v.re-e.re,im:v.im-e.im}));}}
        reconstruction/=scale.max(f64::MIN_POSITIVE);
        let largest=lambda.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let tiny=TINY_EIGENVALUE*largest;let sigma=SIGMA_FRACTION*lambda.iter().map(|z|z.re.abs()).fold(0.,f64::max);
        let moved:Vec<usize>=(0..d).filter(|&k|magnitude(lambda[k])<tiny).collect();
        let slowest_rate=lambda.iter().map(|z|magnitude(*z)).filter(|&m|m>=tiny).fold(f64::INFINITY,f64::min);
        let mut lambda_regular=lambda.clone();for &k in &moved{lambda_regular[k]=C{re:-sigma,im:0.};}
        let mut u=vec![ZERO;d];for i in 0..n{u[i*n+i]=C{re:1./n as f64,im:0.};}
        let a=lapack.multiply_complex(&w_inverse,&u,d,1,d)?;
        let b:Vec<C>=(0..d).map(|k|{let mut s=ZERO;for i in 0..n{add(&mut s,w[i*n+i+d*k]);}s}).collect();
        let mut result=Self{d,lapack:Arc::clone(lapack),w,w_inverse,lambda,lambda_regular,a,b,moved:vec![],central_u:vec![],central_inverse:vec![],
            diagnostics:DressedDiagnostics{condition_bound,reconstruction,tiny,sigma,slowest_rate,..Default::default()}};
        // Exact centre (rank one plus moved eigenvalues), else rank one.
        for(candidate,label)in[(moved.clone(),"exact"),(vec![],"rank-one trace row")]{
            let rank=1+candidate.len();let mut cu=vec![ZERO;d*rank];
            for k in 0..d{cu[k]=div(result.a[k],result.lambda_regular[k]);}
            for(j,&k)in candidate.iter().enumerate(){cu[d*(j+1)+k]=div(ONE,result.lambda_regular[k]);}
            let mut small=vec![ZERO;rank*rank];for j in 0..rank{small[j+rank*j]=ONE;}
            for j in 0..rank{let column=&cu[d*j..d*(j+1)];
                let mut s=ZERO;for k in 0..d{mul_add(&mut s,result.b[k],column[k]);}add(&mut small[rank*j],s);
                for(i,&k)in candidate.iter().enumerate(){let shift=C{re:result.lambda[k].re-result.lambda_regular[k].re,im:result.lambda[k].im-result.lambda_regular[k].im};
                    add(&mut small[i+1+rank*j],mul(shift,column[k]));}}
            let Ok(small_inverse)=inverse(lapack,&small,rank)else{continue};
            let norm=|m:&[C]|(0..rank).map(|c|(0..rank).map(|r|magnitude(m[r+rank*c])).sum::<f64>()).fold(0.,f64::max);
            let condition=norm(&small)*norm(&small_inverse);
            if condition.is_finite()&&condition<=CENTRAL_CONDITION_LIMIT{
                result.moved=candidate;result.central_u=cu;result.central_inverse=small_inverse;
                result.diagnostics.central=label;result.diagnostics.central_rank=rank;break;
            }
        }
        if result.central_u.is_empty(){return Err("Dressed central block with the trace row is singular".into());}
        result.diagnostics.milliseconds=clock.elapsed().as_secs_f64()*1e3;
        Ok(result)
    }
    /// The preconditioner of one mode set (frequencies nu, centre index).
    pub fn batch<'a>(&'a self,nu:&[f64],centre:usize)->Result<DressedBatch<'a>,String>{
        let tiny=self.diagnostics.tiny;let sigma=self.diagnostics.sigma;let mut regularized=0;
        let mut reciprocal=vec![ZERO;nu.len()*self.d];
        for(m,&w)in nu.iter().enumerate(){if m==centre{continue;}
            for k in 0..self.d{let mut z=C{re:self.lambda[k].re,im:self.lambda[k].im-w};
                if magnitude(z)<tiny{z=C{re:-sigma,im:0.};regularized+=1;}reciprocal[m*self.d+k]=div(ONE,z);}}
        Ok(DressedBatch{dressed:self,reciprocal,centre,regularized,execution:Execution::new()?,z:Vec::new(),
            work:nu.len()*2*self.d*self.d})
    }
    /// The centre's (L0' + u v^T)^-1 in the eigenframe, in place on one vector.
    fn central(&self,z:&mut[C]){
        let d=self.d;let rank=1+self.moved.len();
        for k in 0..d{z[k]=div(z[k],self.lambda_regular[k]);}
        let mut t=vec![ZERO;rank];
        for k in 0..d{mul_add(&mut t[0],self.b[k],z[k]);}
        for(i,&k)in self.moved.iter().enumerate(){t[i+1]=mul(C{re:self.lambda[k].re-self.lambda_regular[k].re,im:self.lambda[k].im-self.lambda_regular[k].im},z[k]);}
        let mut c=vec![ZERO;rank];for i in 0..rank{for j in 0..rank{mul_add(&mut c[i],self.central_inverse[i+rank*j],t[j]);}}
        for j in 0..rank{let column=&self.central_u[d*j..d*(j+1)];for k in 0..d{let v=mul(column[k],c[j]);z[k].re-=v.re;z[k].im-=v.im;}}
    }
}
fn div(a:C,b:C)->C{let s=b.re*b.re+b.im*b.im;C{re:(a.re*b.re+a.im*b.im)/s,im:(a.im*b.re-a.re*b.im)/s}}

pub struct DressedBatch<'a>{
    dressed:&'a DenseDressed,reciprocal:Vec<C>,centre:usize,
    /// Non-central (mode, eigenvalue) denominators moved to -sigma.
    pub regularized:usize,
    execution:Execution,z:Vec<C>,
    /// Complex multiply-adds of one application (all modes).
    pub work:usize,
}
/// Rows of W (and W^-1) per task: fixed, so results do not depend on the
/// thread count.
const ROWS_PER_TASK:usize=64;
/// out = A B for the column-major d x d A and d x m B, split by rows of A:
/// A is read once per application, whatever the number of modes.
fn rows_gemm(execution:&Execution,lapack:&Lapack,a:&[C],b:&[C],out:&mut[C],d:usize,m:usize)->Result<(),String>{
    let gemm=lapack.complex_gemm();
    let ranges:Vec<(usize,usize)>=(0..d.div_ceil(ROWS_PER_TASK)).map(|i|(i*ROWS_PER_TASK,((i+1)*ROWS_PER_TASK).min(d))).collect();
    let parts=execution.map(ranges.clone(),|(r0,r1)|->Result<Vec<C>,String>{
        let h=r1-r0;let mut part=vec![ZERO;h*m];gemm.multiply(h,m,d,ONE,&a[r0..],d,b,d,ZERO,&mut part,h)?;Ok(part)});
    for(&(r0,r1),part)in ranges.iter().zip(parts){let part=part?;let h=r1-r0;
        for c in 0..m{out[c*d+r0..c*d+r1].copy_from_slice(&part[c*h..(c+1)*h]);}}
    Ok(())
}
impl DressedBatch<'_>{
    /// y = P x: every mode's block inverse, the centre with the trace row.
    pub fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let dressed=self.dressed;let d=dressed.d;let modes=self.reciprocal.len()/d;
        if x.len()!=y.len()||x.len()!=self.reciprocal.len(){return Err("Dressed preconditioner field dimensions differ".into());}
        grow(&mut self.z,d*modes);let z=&mut self.z[..d*modes];
        rows_gemm(&self.execution,&dressed.lapack,&dressed.w_inverse,x,z,d,modes)?;
        for(mode,column)in z.chunks_exact_mut(d).enumerate(){
            if mode==self.centre{dressed.central(column);}
            else{for(v,&r)in column.iter_mut().zip(&self.reciprocal[mode*d..(mode+1)*d]){*v=mul(*v,r);}}
        }
        rows_gemm(&self.execution,&dressed.lapack,&dressed.w,z,y,d,modes)?;
        if y.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()){return Err("Nonfinite dressed preconditioner action".into());}
        Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_dense_dressed_inverts_every_block_and_the_trace_row(){
        let f=fixture();let p=prepared(&f);let n=4;let d=16;
        let dressed=DenseDressed::new(&p).unwrap();assert_eq!(dressed.diagnostics.central,"exact");
        let mut block=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut block);
        for(a,&b)in block.iter_mut().zip(p.fold.zero_dense.as_ref().unwrap()){add(a,b);}
        let nu=[0.,3.,-6.];let mut batch=dressed.batch(&nu,0).unwrap();
        let x:Vec<C>=(0..3*d).map(|k|C{re:((k*7+3)%29)as f64/29.-0.5,im:((k*11+5)%31)as f64/31.-0.5}).collect();
        let mut y=vec![ZERO;3*d];batch.apply(&x,&mut y).unwrap();
        for(m,&w)in nu.iter().enumerate(){
            let ym=&y[m*d..(m+1)*d];let mut back=vec![ZERO;d];
            for r in 0..d{for c in 0..d{mul_add(&mut back[r],block[r*d+c],ym[c]);}mul_add(&mut back[r],C{re:0.,im:-w},ym[r]);}
            if m==0{let trace=(0..n).fold(ZERO,|s,i|C{re:s.re+ym[i*n+i].re,im:s.im+ym[i*n+i].im});for i in 0..n{add(&mut back[i*n+i],scaled(trace,1./n as f64));}}
            let error=back.iter().zip(&x[m*d..(m+1)*d]).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
            assert!(error<1e-11,"mode {m}: {error:e}");
        }
    }
}
