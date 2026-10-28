//! The lattice preconditioner: the Floquet frames (`floquet`) with a centre
//! Woodbury term, for every shift class.
//!
//! (S - sigma)^-1 on the separable part is P D^-1 P~ with the transforms of
//! `floquet::Transform` and the elementwise D = lambda_a + mu_b - i nu_m -
//! sigma; the kernels' truncation and every non-separable part are left to
//! the Krylov solver but at the centre:
//! - centre Woodbury: the recycling and the diagonal fold rest on the
//!   ground-pair rows and columns (rank 2 r: the scattering into and out of
//!   the ground populations, where the rest is largest), in the central
//!   frame, on the centre mode; the steady trace row joins it, and centre
//!   denominators at roundoff are moved and corrected exactly. Off the
//!   centre the population denominators carry -i nu_q (|nu_q| >> Gamma) and
//!   the recycling is a small perturbation.
//! - the kernels act on every mode.
//!
//! Shifts (`poles::Pole`): the steady trace row and real sigma act on
//! Hermitian-symmetric fields (only canonical modes computed, the rest
//! mirrored); a complex sigma acts on the real form of a complex field, a
//! pair of Hermitian-symmetric fields (x_1, x_2) for x_1 + i x_2, through the
//! complex inverse on every mode.

use super::*;
use super::floquet::{FloquetKernels,Transform,adjoint,inverse,mirror_computed,sandwich_batch,transpose};
use super::operator::LatticeOperator;
use super::poles::Pole;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Centre denominators below this fraction of the frame scale are at
/// roundoff (a deliberate margin of about 500x over double precision);
/// they are moved to -MOVED_FRACTION times the largest frame damping and
/// corrected exactly.
const TINY_DENOMINATOR:f64=1e-13;
/// A deliberate margin: ten orders above TINY_DENOMINATOR; the exact
/// rank-one correction (Woodbury) removes the move up to roundoff.
const MOVED_FRACTION:f64=1e-3;
/// Column ranges of the centre factor's z = G^-1 V^T c0: fixed, so that the
/// partial sums do not depend on the thread count; one per worker of the
/// desktop app's eight.
const CENTRE_RANGES:usize=8;


/// A lattice preconditioner as the Krylov solver sees it.
pub trait Preconditioner{
    /// y = P x.
    fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>;
}

/// The request-level preconditioner: the shared Floquet kernels and the
/// centre Woodbury factors in the central frame.
pub struct Preconditioning<'p>{
    kernels:&'p FloquetKernels,lapack:Arc<Lapack>,
    /// U-hat (rank blocks, column-major d x rank), V-hat (rank x d,
    /// column-major); the trace row's u-hat and v-hat.
    uhat:Vec<C>,vhat:Vec<C>,rank:usize,trace_u:Vec<C>,trace_v:Vec<C>,
    tiny:f64,moved:f64,
}

/// The ground-pair rows and columns of W = R0 + Krest_0 (recycling and the
/// diagonal fold rest) as the rank-2r factors U (2r blocks) and V (2r row
/// blocks): U = [e_p | W[:, p] off the ground pairs], V = [W[p, :]; e_p].
fn woodbury_factors(prepared:&LatticePrepared)->(Vec<C>,Vec<C>,usize){
    let generator=&prepared.generator;let n=generator.n;let d=n*n;
    let ground_pair=|p:usize|generator.ground[p/n]&&generator.ground[p%n];
    let positions:Vec<usize>=(0..d).filter(|&p|ground_pair(p)).collect();let r=positions.len();
    let mut slot=vec![usize::MAX;d];for(k,&p)in positions.iter().enumerate(){slot[p]=k;}
    let(mut columns,mut rows)=(vec![ZERO;r*d],vec![ZERO;r*d]);
    let mut put=|row:usize,col:usize,v:C|{
        if slot[col]!=usize::MAX{add(&mut columns[slot[col]*d+row],v);}
        if slot[row]!=usize::MAX{add(&mut rows[slot[row]*d+col],v);}};
    for(p,q)in &generator.zero_component().op.sandwich{for &(i,k,a)in &p.entries{for &(l,j,b)in &q.entries{put(i*n+j,k*n+l,mul(a,b));}}}
    if let Some(t)=prepared.fold.targets.first().filter(|t|t.charge==generator.zero){
        let rest=&t.rest[0];
        for block in &rest.blocks{let h=block.rows.len();
            for(c,&col)in block.cols.iter().enumerate(){for(i,&row)in block.rows.iter().enumerate(){put(row as usize,col as usize,block.values[i+h*c]);}}}
        let s=&rest.sparse;
        for row in 0..s.row.len().saturating_sub(1){for at in s.row[row]as usize..s.row[row+1]as usize{put(row,s.col[at]as usize,s.value[at]);}}
    }
    let rank=2*r;let(mut u,mut v)=(vec![ZERO;rank*d],vec![ZERO;rank*d]);
    for(k,&p)in positions.iter().enumerate(){
        u[k*d+p]=ONE;
        for row in 0..d{if !ground_pair(row){u[(r+k)*d+row]=columns[k*d+row];}}
        v[k*d..(k+1)*d].copy_from_slice(&rows[k*d..(k+1)*d]);v[(r+k)*d+p]=ONE;
    }
    (u,v,rank)
}

impl<'p> Preconditioning<'p>{
    pub fn new(prepared:&'p LatticePrepared)->Result<Self,String>{
        let kernels=&prepared.floquet;let lapack=&prepared.lapack;let gemm=lapack.complex_gemm();
        let n=prepared.generator.n;let d=n*n;
        // hat(Y) = P0^-1 Y Q0^-1 and V-hat = P0^T V Q0^T (so that
        // V . vec(P0 c Q0) = V-hat . vec(c)), Q0 = P0^dagger.
        let(p0,p0inv)=kernels.central();let(q0,q0inv)=(adjoint(p0,n),adjoint(p0inv,n));let(pt,qt)=(transpose(p0,n),transpose(&q0,n));
        let(mut u,mut v,rank)=woodbury_factors(prepared);let mut stack=Vec::new();
        sandwich_batch(&gemm,n,p0inv,&q0inv,&mut u,rank,&mut stack)?;sandwich_batch(&gemm,n,&pt,&qt,&mut v,rank,&mut stack)?;
        let mut vhat=vec![ZERO;rank*d];for k in 0..rank{for c in 0..d{vhat[k+rank*c]=v[k*d+c];}}
        let(mut trace_u,mut trace_v)=(vec![ZERO;d],vec![ZERO;d]);for i in 0..n{trace_u[i*n+i]=C{re:1./n as f64,im:0.};trace_v[i*n+i]=ONE;}
        sandwich_batch(&gemm,n,p0inv,&q0inv,&mut trace_u,1,&mut stack)?;sandwich_batch(&gemm,n,&pt,&qt,&mut trace_v,1,&mut stack)?;
        let(lambda,mu)=kernels.quasi_energies();
        let scale=lambda.iter().chain(mu).map(|z|magnitude(*z)).fold(0.,f64::max);
        let damping=lambda.iter().map(|z|z.re.abs()).fold(0.,f64::max)+mu.iter().map(|z|z.re.abs()).fold(0.,f64::max);
        Ok(Self{kernels,lapack:Arc::clone(lapack),uhat:u,vhat,rank,trace_u,trace_v,tiny:TINY_DENOMINATOR*scale,moved:MOVED_FRACTION*damping})
    }
    /// The preconditioner of an operator's modes at a shift: the kernels on
    /// every mode, the centre Woodbury and, for the steady shift, the trace
    /// row.
    pub fn batch(&self,op:&LatticeOperator<'_>,shift:Pole)->Result<FloquetBatch<'p>,String>{
        let computed=op.canonical.clone();
        let transform=self.kernels.transform(op,&computed)?;
        FloquetBatch::new(self.kernels,self,op,shift,computed,transform)
    }
}

/// (D + U V^T)^-1 = D^-1 - D^-1 U G^-1 V^T D^-1 on the centre block: D^-1 U
/// (d x total) and G^-1 V^T (total x d), column-major.
struct CentreFactor{total:usize,du:Vec<C>,gv:Vec<C>}
pub struct FloquetBatch<'p>{
    n:usize,lapack:Arc<Lapack>,
    shift:Pole,centre:usize,
    /// The modes the transforms compute (canonical or a subset of them),
    /// and those divided (every mode for a complex shift).
    computed:Vec<bool>,divided:Vec<bool>,mirror:Vec<usize>,
    transform:Transform<'p>,
    /// 1 / (lambda_a + mu_b - i nu_q - sigma) per divided mode.
    reciprocal:Vec<C>,
    factor:Option<CentreFactor>,
    /// Centre denominators at roundoff, moved and corrected.
    pub moved:usize,
    execution:Execution,buffers:[Vec<C>;3],
}
impl<'p> FloquetBatch<'p>{
    fn new(kernels:&'p FloquetKernels,woodbury:&Preconditioning<'p>,op:&LatticeOperator<'_>,shift:Pole,computed:Vec<bool>,transform:Transform<'p>)->Result<Self,String>{
        let n=op.n;let d=n*n;let modes=op.modes.len();let sigma=shift.sigma();let centre=op.centre;
        let(lambda,mu)=kernels.quasi_energies();
        let divided=if shift.copies()==2{vec![true;modes]}else{computed.clone()};
        let mut reciprocal=vec![ZERO;modes*d];let mut moved=Vec::new();
        for q in(0..modes).filter(|&q|divided[q]){for a in 0..n{for b in 0..n{
            let z=C{re:lambda[a].re+mu[b].re-sigma.re,im:lambda[a].im+mu[b].im-op.nu[q]-sigma.im};
            let z=if q==centre&&magnitude(z)<woodbury.tiny{moved.push((a*n+b,z));C{re:-woodbury.moved,im:0.}}else{z};
            let s=z.re*z.re+z.im*z.im;if !(s>0.&&s.is_finite()){return Err("Unresolved Floquet frame denominator".into());}
            reciprocal[q*d+a*n+b]=C{re:z.re/s,im:-z.im/s};}}}
        // Centre factor: the Woodbury columns, the trace row, the moved
        // denominators (true z for -moved: add (z + moved) e_k e_k^T).
        let factor=if divided[centre]{let w=woodbury;
                let rank=w.rank;let trace=usize::from(shift.steady());let total=rank+trace+moved.len();
                let mut u=vec![ZERO;d*total];let mut v=vec![ZERO;total*d];
                u[..d*rank].copy_from_slice(&w.uhat);for c in 0..d{for k in 0..rank{v[k+total*c]=w.vhat[k+rank*c];}}
                let mut e=rank;
                if trace>0{u[e*d..(e+1)*d].copy_from_slice(&w.trace_u);for c in 0..d{v[e+total*c]=w.trace_v[c];}e+=1;}
                for &(k,z)in &moved{u[e*d+k]=ONE;v[e+total*k]=C{re:z.re+w.moved,im:z.im};e+=1;}
                let rec=&reciprocal[centre*d..(centre+1)*d];
                let du:Vec<C>=(0..d*total).map(|k|mul(rec[k%d],u[k])).collect();
                let mut g=vec![ZERO;total*total];for k in 0..total{g[k+total*k]=ONE;}
                w.lapack.complex_gemm().multiply(total,total,d,ONE,&v,total,&du,d,ONE,&mut g,total)?;
                let ginv=inverse(&w.lapack,&g,total)?;let mut gv=vec![ZERO;total*d];
                w.lapack.complex_gemm().multiply(total,d,total,ONE,&ginv,total,&v,total,ZERO,&mut gv,total)?;
                Some(CentreFactor{total,du,gv})}
            else{None};
        Ok(Self{n,lapack:Arc::clone(&op.prepared.lapack),shift,centre,computed,divided,mirror:op.mirror.clone(),transform,reciprocal,factor,
            moved:moved.len(),execution:Execution::new()?,buffers:[vec![],vec![],vec![]]})
    }
    /// Mirrored blocks from the computed ones (zero where neither a block nor
    /// its mirror is computed), the centre Hermitian.
    fn mirror(&self,x:&mut[C]){mirror_computed(x,self.n,&self.computed,&self.mirror);}
    /// Divides by the denominators on the divided modes and applies the
    /// centre factor.
    fn divide(&self,c:&mut[C])->Result<(),String>{
        let d=self.n*self.n;
        for q in(0..self.divided.len()).filter(|&q|self.divided[q]){for(v,&r)in c[q*d..(q+1)*d].iter_mut().zip(&self.reciprocal[q*d..(q+1)*d]){*v=mul(*v,r);}}
        if let Some(f)=&self.factor{
            // Two memory-bound products, split over the workers: z = G^-1 V^T c0
            // by CENTRE_RANGES column ranges (partial sums added in range
            // order, independent of the thread count), then c0 -= D^-1 U z by
            // row ranges.
            let c0=&mut c[self.centre*d..(self.centre+1)*d];let t=f.total;let lapack=&self.lapack;
            let width=d.div_ceil(CENTRE_RANGES);let starts:Vec<usize>=(0..d).step_by(width).collect();let view:&[C]=c0;
            let parts=self.execution.map(starts,|start|->Result<Vec<C>,String>{let end=(start+width).min(d);let mut z=vec![ZERO;t];
                lapack.complex_gemm().multiply(t,1,end-start,ONE,&f.gv[start*t..],t,&view[start..end],end-start,ZERO,&mut z,t)?;Ok(z)});
            let mut z=vec![ZERO;t];for part in parts{for(a,b)in z.iter_mut().zip(part?){add(a,b);}}
            let z=&z;
            self.execution.for_mode_chunks(c0,1,2*t*d,MINIMUM_WORK_PER_TASK/8,&mut Vec::<()>::new(),|start,chunk,_|{
                lapack.complex_gemm().multiply(chunk.len(),1,t,C{re:-1.,im:0.},&f.du[start..],d,z,t,ONE,chunk,chunk.len())})?;
        }
        Ok(())
    }
    fn solve(&mut self,x:&[C],y:&mut[C],buffers:&mut[Vec<C>;3])->Result<(),String>{
        let(n,d)=(self.n,self.n*self.n);let modes=self.computed.len();let size=modes*d;
        let[a,b,c]=buffers;let(a,b,c)=(&mut a[..size],&mut b[..size],&mut c[..size]);
        if self.shift.copies()==2{
            // The transforms are complex-linear and keep Hermitian symmetry:
            // c = backward(x_1) + i backward(x_2), divided on every mode, then
            // forward of its Hermitian-symmetric parts (c + c^dagger)/2 and
            // (c - c^dagger)/2i, (c^dagger)_q = (c_-q)^dagger.
            let(x1,x2)=x.split_at(size);let(y1,y2)=y.split_at_mut(size);
            self.transform.backward(x1,a)?;self.mirror(a);
            self.transform.backward(x2,b)?;self.mirror(b);
            for(u,v)in a.iter_mut().zip(b.iter()){*u=C{re:u.re-v.im,im:u.im+v.re};}
            self.divide(a)?;
            for q in 0..modes{let p=self.mirror[q];
                for i in 0..n{for j in 0..n{let(z,m)=(a[q*d+i*n+j],conj(a[p*d+j*n+i]));
                    b[q*d+i*n+j]=C{re:0.5*(z.re+m.re),im:0.5*(z.im+m.im)};c[q*d+i*n+j]=C{re:0.5*(z.im-m.im),im:-0.5*(z.re-m.re)};}}}
            self.transform.forward(b,y1)?;self.mirror(y1);
            self.transform.forward(c,y2)?;self.mirror(y2);
        }else{
            self.transform.backward(x,b)?;self.divide(b)?;
            self.transform.forward(b,y)?;self.mirror(y);
        }
        Ok(())
    }
}
impl Preconditioner for FloquetBatch<'_>{
    fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let size=self.computed.len()*self.n*self.n;
        if x.len()!=size*self.shift.copies()||y.len()!=x.len(){return Err("Floquet preconditioner field dimensions differ".into());}
        let mut buffers=std::mem::take(&mut self.buffers);for b in &mut buffers{grow(b,size);}
        let result=self.solve(x,y,&mut buffers);
        self.buffers=buffers;result?;
        if y.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()){return Err("Nonfinite Floquet preconditioner action".into());}
        Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{Fixture,fixture,prepared};
    use super::super::operator::symmetric_field;
    use super::super::krylov::{self,KrylovOptions,System};
    use crate::preparation::{DriveView,MatrixView,ModelView,PhaseBasis};
    /// The fixture with its far drives only: no explicit step, so no kernel
    /// and the central frame is the block frame.
    fn far_only(f:&Fixture)->LatticePrepared{
        let z=&f.zero;let h=MatrixView{real:&f.h,imag:z};let channels=[MatrixView{real:&f.c,imag:z}];
        let drives=[DriveView{omega:2000.,cosine:MatrixView{real:&f.far,imag:z},sine:MatrixView{real:z,imag:z}},
            DriveView{omega:2003.,cosine:MatrixView{real:&f.far2,imag:z},sine:MatrixView{real:z,imag:z}}];
        let model=ModelView{dimension:4,hamiltonian:h,collapse:&channels,drives:&drives,ground:&[true,true,false,false]};
        LatticePrepared::prepare(&model,PhaseBasis::from_declared(&[3.,2000.],&[vec![0,1],vec![1,1]]).unwrap()).unwrap()
    }
    /// The dense block model of one mode: NL' X + X NR' - i nu - sigma, with
    /// R0 and the rest on ground-pair rows and columns when `woodbury`.
    fn block_model(p:&LatticePrepared,nu:f64,sigma:C,woodbury:bool)->Vec<C>{
        let n=p.generator.n;let d=n*n;let zero=p.generator.zero_component();let t=&p.fold.targets[0];
        let mut nl=zero.op.left.dense(n);let mut nr=zero.op.right.dense(n);for k in 0..d{add(&mut nl[k],t.left[0][k]);add(&mut nr[k],t.right[0][k]);}
        let mut m=vec![ZERO;d*d];Superop{left:Sparse::from_dense(&nl,n),right:Sparse::from_dense(&nr,n),sandwich:vec![]}.dense_into(n,ONE,&mut m);
        for r in 0..d{m[r*d+r].im-=nu;m[r*d+r].re-=sigma.re;m[r*d+r].im-=sigma.im;}
        if woodbury{
            let mut w=vec![ZERO;d*d];Superop{left:Sparse::default(),right:Sparse::default(),sandwich:zero.op.sandwich.clone()}.dense_into(n,ONE,&mut w);
            t.rest[0].dense_into(d,ONE,&mut w);let ground=|q:usize|p.generator.ground[q/n]&&p.generator.ground[q%n];
            for r in 0..d{for c in 0..d{if ground(r)||ground(c){add(&mut m[r*d+c],w[r*d+c]);}}}
        }
        m
    }
    fn deviation(a:&[C],b:&[C])->f64{a.iter().zip(b).map(|(x,y)|magnitude(C{re:x.re-y.re,im:x.im-y.im})).fold(0.,f64::max)}
    #[test]
    fn lattice_centre_factor_inverts_its_woodbury_model_for_every_shift(){
        // Without kernels (far drives only) on the centre alone: the batch is
        // the exact inverse of the block with the recycling and rest on the
        // ground pairs (+ the trace row for the steady shift), for the steady,
        // a real and a complex shift (real form).
        let f=fixture();let p=far_only(&f);let n=4;let d=16;
        assert!(p.floquet.diagnostics.support.is_empty()&&p.fold.targets.len()>1);
        let pre=Preconditioning::new(&p).unwrap();let op=LatticeOperator::new(&p,&[p.generator.zero.clone()]).unwrap();
        for shift in[Pole::Zero,Pole::Real(0.3),Pole::Complex(C{re:0.5,im:7.})]{
            let mut batch=pre.batch(&op,shift).unwrap();
            let mut x=symmetric_field(&op,5);if shift.copies()==2{x.extend(symmetric_field(&op,9));}
            let mut y=vec![ZERO;x.len()];batch.apply(&x,&mut y).unwrap();
            let join=|v:&[C]|->Vec<C>{if shift.copies()==2{(0..d).map(|k|C{re:v[k].re-v[d+k].im,im:v[k].im+v[d+k].re}).collect()}else{v.to_vec()}};
            let(yc,xc)=(join(&y),join(&x));
            let m=block_model(&p,0.,shift.sigma(),true);let mut back=vec![ZERO;d];for r in 0..d{for c in 0..d{mul_add(&mut back[r],m[r*d+c],yc[c]);}}
            if shift.steady(){let tr=(0..n).fold(ZERO,|s,i|C{re:s.re+yc[i*n+i].re,im:s.im+yc[i*n+i].im});for i in 0..n{add(&mut back[i*n+i],scaled(tr,1./n as f64));}}
            let scale=xc.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            assert!(deviation(&back,&xc)<1e-10*scale,"{shift:?}: {:e}",deviation(&back,&xc));
        }
    }
    /// The relative error of a tight solve of one shift class on the order-2
    /// lattice against a dense LU solve.
    fn shifted_solve(p:&LatticePrepared,shift:Pole)->f64{
        let pre=Preconditioning::new(p).unwrap();
        let modes=p.modes(2).unwrap();let mut op=LatticeOperator::new(p,&modes).unwrap();let size=op.dimension();let d=16;
        let dense=op.dense();let centre=op.centre;
        let mut batch=pre.batch(&op,shift).unwrap();
        let x=symmetric_field(&op,2);
        let mut b:Vec<C>=x.iter().map(|v|C{re:-v.re,im:-v.im}).collect();if shift.copies()==2{b.extend(vec![ZERO;size]);}
        let mut system=System::new(&mut op,&mut batch,shift).unwrap();
        let solve=krylov::gmres(&mut system,&b,&KrylovOptions::new(1e-11)).unwrap();
        // Dense: (A - sigma) w = b, the steady system with the trace row u v^T.
        let sigma=shift.sigma();let mut m=dense;for r in 0..size{m[r*size+r].re-=sigma.re;m[r*size+r].im-=sigma.im;}
        if shift.steady(){let c=centre*d;for i in 0..4{for j in 0..4{m[(c+i*4+i)*size+c+j*4+j].re+=0.25;}}}
        let mut column=vec![ZERO;size*size];for r in 0..size{for c in 0..size{column[r+size*c]=m[r*size+c];}}
        let(lu,piv)=p.lapack.complex_factor(&column,size).unwrap();
        let exact=p.lapack.complex_solve_matrix_factored(&lu,&piv,&b[..size],size).unwrap();
        let w:Vec<C>=if shift.copies()==2{(0..size).map(|k|C{re:solve.x[k].re-solve.x[size+k].im,im:solve.x[k].im+solve.x[size+k].re}).collect()}else{solve.x};
        let scale=exact.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        deviation(&w,&exact)/scale
    }
    #[test]
    fn lattice_floquet_preconditioner_solves_every_shift_class(){
        // The steady trace row, small and large real poles, a complex pole and
        // a translated pole sigma + i nu_s: GMRES with the Floquet frames
        // reaches the dense solution.
        let f=fixture();let p=prepared(&f);
        assert!(!p.floquet.diagnostics.support.is_empty(),"the fixture's near pair enters the support");
        let nu=p.plan.steps[p.plan.explicit()[0]].nu.abs();
        for shift in[Pole::Zero,Pole::Real(1e-3),Pole::Real(20.),Pole::Complex(C{re:0.5,im:13.}),Pole::Complex(C{re:2.,im:nu})]{
            let error=shifted_solve(&p,shift);
            assert!(error<1e-8,"{shift:?}: error {error:e}");
        }
    }
}
