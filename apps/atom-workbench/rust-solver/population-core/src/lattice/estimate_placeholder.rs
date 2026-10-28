//! TEMPORARY error estimate, until the unified estimate (R2, its own module
//! behind `estimate_api`) lands: the residual-based bound of the reduced
//! model, propagated through the dressed blocks only.
//!
//! The Galerkin error e = x - W X u obeys e' = A e + R u(t), e(0) = 0, with
//! R = A W X - W X G. Here A is replaced by its block-diagonal dressed part
//! (`eigenbasis::propagate`, closed form over the Ritz exponentials), R by
//! its leading singular components (R^T R from the Gram matrices of W and
//! A W, cached per column; the estimate scaled by the norm left out). The
//! steady error is the centre block's -L0'^+ R u_inf. Left out, and covered
//! by the unified estimate: the couplings between blocks in the propagation,
//! the truncation outflow beyond P u Q and the fold charges.
use super::*;
use super::estimate_api::{ErrorEstimate,EstimateSetup,EstimateWork,Estimator,ReducedView,SteadySolves};
use super::krylov::Vectors;
use super::operator::LatticeOperator;
use super::reduced::{product,transpose_rows};

/// Residual components kept: down to this fraction of the largest singular
/// value, at most COMPONENTS.
const COMPONENT:f64=1e-3;
const COMPONENTS:usize=24;
/// |lambda| below this is the dressed centre's kernel.
const KERNEL:f64=1e-9;

pub struct Placeholder<'a>{
    setup:EstimateSetup<'a>,op:LatticeOperator<'a>,vectors:Vectors,
    /// E1 = W^T A W and E2 = (A W)^T A W over the covered columns.
    e1:Vec<f64>,e2:Vec<f64>,covered:usize,
}
pub fn construct<'a>(setup:EstimateSetup<'a>)->Result<Box<dyn Estimator+'a>,String>{
    let op=LatticeOperator::new(setup.prepared,&setup.lattice.modes())?;let vectors=Vectors::new(&op,1)?;
    Ok(Box::new(Placeholder{setup,op,vectors,e1:vec![],e2:vec![],covered:0}))
}
impl Placeholder<'_>{
    fn cover(&mut self,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let(old,new)=(self.covered,basis.len());if new==old{return Ok(());}
        let w:Vec<&[C]>=basis.iter().map(|v|v.as_slice()).collect();let a:Vec<&[C]>=images.iter().map(|v|v.as_slice()).collect();
        let e1c=self.vectors.gram(&w,&a[old..])?;let e1r=self.vectors.gram(&w[old..],&a[..old])?;let e2c=self.vectors.gram(&a,&a[old..])?;
        let width=new-old;let(mut e1,mut e2)=(vec![0.;new*new],vec![0.;new*new]);
        for i in 0..old{for j in 0..old{e1[i*new+j]=self.e1[i*old+j];e2[i*new+j]=self.e2[i*old+j];}}
        for i in 0..new{for c in 0..width{let j=old+c;e1[i*new+j]=e1c[i*width+c];e2[i*new+j]=e2c[i*width+c];e2[j*new+i]=e2c[i*width+c];}}
        for c in 0..width{for j in 0..old{e1[(old+c)*new+j]=e1r[c*old+j];}}
        self.e1=e1;self.e2=e2;self.covered=new;Ok(())
    }
}
impl Estimator for Placeholder<'_>{
    fn estimate(&mut self,view:&ReducedView<'_>,_steady:&mut dyn SteadySolves)->Result<ErrorEstimate,String>{
        let clock=Instant::now();
        self.cover(view.basis,view.images)?;
        let prepared=self.setup.prepared;let lapack=&prepared.lapack;let eigen=prepared.eigenbasis()?;
        let(k,r)=(view.basis.len(),view.r);let n=prepared.generator.n;let groups=self.setup.outputs.groups;
        // R^T R = X^T E2 X - X^T E1^T X G - G^T X^T E1 X + G^T X^T X G.
        let(x,g)=(view.x,view.g);let xt=transpose_rows(x,k,r);
        let ax=product(lapack,&xt,&product(lapack,&self.e2,x,k,k,r)?,r,k,r)?;
        let b=product(lapack,&xt,&product(lapack,&self.e1,x,k,k,r)?,r,k,r)?;   // X^T E1 X
        let bg=product(lapack,&b,g,r,r,r)?;let xx=product(lapack,&xt,x,r,k,r)?;
        let gt=transpose_rows(g,r,r);let gxxg=product(lapack,&gt,&product(lapack,&xx,g,r,r,r)?,r,r,r)?;
        let btg=transpose_rows(&bg,r,r);
        let rr:Vec<f64>=(0..r*r).map(|e|ax[e]-btg[e]-bg[e]+gxxg[e]).collect();
        let complex:Vec<C>=rr.iter().map(|&v|C{re:v,im:0.}).collect();
        let(values,vectors)=lapack.hermitian_eigen(&complex,r)?;
        let top=values.iter().fold(0_f64,|a,&v|a.max(v)).max(0.);let total:f64=values.iter().map(|v|v.max(0.)).sum();
        let mut order:Vec<usize>=(0..r).collect();order.sort_by(|&a,&b|values[b].total_cmp(&values[a]));
        let chosen:Vec<usize>=order.into_iter().filter(|&j|values[j]>COMPONENT*COMPONENT*top).take(COMPONENTS).collect();
        let kept:f64=chosen.iter().map(|&j|values[j]).sum();
        let scale=if kept>0.{(total/kept).sqrt()}else{1.};
        // Fields R v_j = A W (X v_j) - W (X G v_j); psi_j(t) = sum_k (v_j^T S)_k a_k e^{theta_k t}.
        // Real eigenvectors (a real symmetric matrix's, up to a phase per column).
        let real=|j:usize|->Vec<f64>{let col=&vectors[r*j..r*(j+1)];let pivot=col.iter().copied().max_by(|a,b|magnitude(*a).total_cmp(&magnitude(*b))).unwrap_or(ONE);
            let s=magnitude(pivot).max(f64::MIN_POSITIVE);let phase=C{re:pivot.re/s,im:-pivot.im/s};col.iter().map(|z|mul(*z,phase).re).collect()};
        let mut fields=Vec::new();let mut alpha=Vec::new();
        let w:Vec<&[C]>=view.basis.iter().map(|v|v.as_slice()).collect();let a:Vec<&[C]>=view.images.iter().map(|v|v.as_slice()).collect();
        for &j in &chosen{
            let mut v=real(j);let size=v.iter().map(|x|x*x).sum::<f64>().sqrt();for x in v.iter_mut(){*x/=size.max(f64::MIN_POSITIVE);}
            let xv=product(lapack,x,&v,k,r,1)?;let gv=product(lapack,g,&v,r,r,1)?;let xgv=product(lapack,x,&gv,k,r,1)?;
            let mut f=self.vectors.combinations(&a,&[xv])?.remove(0);let back=self.vectors.combinations(&w,&[xgv])?.remove(0);
            for(t,u)in f.iter_mut().zip(&back){t.re-=u.re;t.im-=u.im;}
            fields.push(f);
            alpha.push((0..r).map(|kk|{let mut s=ZERO;for i in 0..r{mul_add(&mut s,C{re:v[i],im:0.},view.vectors[i*r+kk]);}mul(s,view.used_coefficients[kk])}).collect::<Vec<C>>());
        }
        let outputs=&self.setup.outputs;
        let times:Vec<f64>=outputs.times.iter().chain(outputs.horizons).copied().collect();
        let rows=if fields.is_empty()||times.is_empty(){vec![vec![0.;n];times.len()]}else{eigen.propagate(&self.op,&fields,&alpha,view.used_theta,&times)?};
        let(trajectory_rows,horizon_rows)=rows.split_at(outputs.times.len());
        let zero=vec![0.;n];
        let mut trajectory:Vec<f64>=trajectory_rows.iter().map(|p|scale*ledger::metric(p,&zero,groups)).collect();
        if let Some(first)=trajectory.first_mut(){*first=0.;}
        let horizons:Vec<f64>=horizon_rows.iter().map(|p|scale*ledger::metric(p,&zero,groups)).collect();
        // Steady: -L0'^+ (R u_inf) on the centre block, u_inf the reduced
        // bordered solution, else the kernel's spectral projection.
        let kernel_part:Vec<f64>=(0..r).map(|i|(0..r).filter(|&j|view.kernel[j]).fold(0.,|acc,j|acc+mul(view.vectors[i*r+j],view.coefficients[j]).re)).collect();
        let u=view.steady.unwrap_or(&kernel_part);
        let xu=product(lapack,x,u,k,r,1)?;let gu=product(lapack,g,u,r,r,1)?;let xgu=product(lapack,x,&gu,k,r,1)?;
        let mut f=self.vectors.combinations(&a,&[xu])?.remove(0);let back=self.vectors.combinations(&w,&[xgu])?.remove(0);
        for(t,v)in f.iter_mut().zip(&back){t.re-=v.re;t.im-=v.im;}
        let d=n*n;let c=self.op.centre;let y=eigen.dressed(&f[c*d..(c+1)*d])?;let z=eigen.complex_coefficients(&y);
        let mut p=vec![0.;n];
        for(jj,&zj)in z.iter().enumerate(){let l=eigen.lambda[jj];if magnitude(l)<KERNEL{continue;}let s2=l.re*l.re+l.im*l.im;
            // -z / lambda.
            let q=mul(zj,C{re:-l.re/s2,im:l.im/s2});for(i,pi)in p.iter_mut().enumerate(){*pi+=mul(eigen.readout[i+n*jj],q).re;}}
        let steady=Some(ledger::metric(&p,&zero,groups));
        Ok(ErrorEstimate{trajectory,horizons,steady,work:EstimateWork{milliseconds:clock.elapsed().as_secs_f64()*1e3,..Default::default()}})
    }
}
