//! The steady error by GCRO, recycling the space.
//!
//! At t -> infinity the error is e = -A^# f with f = R u_inf + B y_inf, the
//! residual of the reduced steady state plus the slaved ring's feedback
//! (R2: exact, A (W X u_inf) = R u_inf), read as the centre populations of
//! the bordered solution M y = f, M = A + u tau0^T (f has no centre trace).
//! GCRO minimizes |f - M D x| over the directions D whose images M D are
//! known:
//! - every basis column w_j, M w_j = A w_j + u tau0(w_j) from the images;
//! - every earlier augmentation direction with its image, kept across
//!   checkpoints: the first one is the pipeline's own bordered solve of the
//!   first checkpoint's f (`SteadySolves`), the later ones preconditioned
//!   residuals (the shared Floquet preconditioner of the steady shift).
//! The images are orthonormalized as they arrive (M D = Q R): the new basis
//! columns of a checkpoint as one block (block CGS2 against Q, then a
//! pivoted Cholesky QR twice inside the block), Q stored in row panels so that the projections are GEMMs spread
//! over the workers (the real inner product of Hermitian-symmetric fields is
//! the dot product of their real and imaginary parts). A checkpoint then
//! costs the projections of f and, usually, a few steps. It stops when the
//! residual is below STEADY_RESIDUAL |f| and the readout has settled (two
//! consecutive changes below STABLE of its size).
use super::super::*;
use super::super::estimate_api::SteadySolves;
use super::super::operator::LatticeOperator;
use super::super::preconditioner::{FloquetBatch,Preconditioner,Preconditioning,Shift};
use super::super::floquet::Reach;
use crate::parallel::Execution;

const STEADY_RESIDUAL:f64=1e-4;
const STABLE:f64=0.1;
const MAXIMUM_STEPS:usize=60;
/// Real rows per panel of Q.
const PANEL_ROWS:usize=1<<15;
/// A block image whose part outside the block's earlier pivots is below
/// this fraction of its size is dropped (a Gram matrix resolves singular
/// values down to about the square root of the roundoff).
const CHOLESKY_DEPENDENT:f64=1e-7;

#[derive(Clone,Debug,Default)]
pub struct SteadyWork{pub applications:usize,pub preconditions:usize,pub pipeline_solves:usize,pub inner_steps:usize,pub steps:usize,
    /// Time (ms) orthonormalizing images, and in the steps (preconditioner, operator).
    pub push_ms:f64,pub step_ms:f64}

/// Real columns (a field's real and imaginary parts interleaved) stored by
/// row panels, each panel column-major.
struct Panels{length:usize,starts:Vec<usize>,q:Vec<Vec<f64>>,count:usize}
impl Panels{
    fn new(length:usize)->Self{
        let starts:Vec<usize>=(0..length).step_by(PANEL_ROWS).collect();let panels=starts.len();
        Self{length,starts,q:vec![Vec::new();panels],count:0}
    }
    fn rows(&self,p:usize)->(usize,usize){(self.starts[p],self.starts.get(p+1).copied().unwrap_or(self.length))}
    fn push(&mut self,column:&[f64]){for p in 0..self.starts.len(){let(s,e)=self.rows(p);self.q[p].extend_from_slice(&column[s..e]);}self.count+=1;}
    /// Q^T g (count x m, column-major) for g (length x m, column-major).
    fn dots(&self,lapack:&Lapack,execution:&Execution,g:&[f64],m:usize)->Result<Vec<f64>,String>{
        let count=self.count;if count==0||m==0{return Ok(vec![0.;count*m]);}
        let parts=execution.map((0..self.starts.len()).collect(),|p|->Result<Vec<f64>,String>{
            let(s,e)=self.rows(p);let mut out=vec![0.;count*m];
            lapack.gemm(true,false,count,m,e-s,1.,&self.q[p],e-s,&g[s..],self.length,0.,&mut out,count)?;Ok(out)});
        let mut h=vec![0.;count*m];for part in parts{for(a,b)in h.iter_mut().zip(part?){*a+=b;}}
        Ok(h)
    }
    /// g -= Q h.
    fn subtract(&self,lapack:&Lapack,execution:&Execution,g:&mut[f64],h:&[f64],m:usize)->Result<(),String>{
        let count=self.count;if count==0||m==0{return Ok(());}
        let parts=execution.map((0..self.starts.len()).collect(),|p|->Result<Vec<f64>,String>{
            let(s,e)=self.rows(p);let mut out=vec![0.;(e-s)*m];
            lapack.gemm(false,false,e-s,m,count,1.,&self.q[p],e-s,h,count,0.,&mut out,e-s)?;Ok(out)});
        for(p,part)in parts.into_iter().enumerate(){let part=part?;let(s,e)=self.rows(p);let rows=e-s;
            for c in 0..m{for(t,v)in g[s+self.length*c..e+self.length*c].iter_mut().zip(&part[rows*c..rows*(c+1)]){*t-=v;}}}
        Ok(())
    }
}

fn reals(x:&[C],out:&mut Vec<f64>){for v in x{out.push(v.re);out.push(v.im);}}
/// Rows per task of the block products.
const BLOCK_ROWS:usize=1<<14;
/// G^T G (m x m, column-major) of a block G (length x m), by row chunks.
fn block_gram(lapack:&Lapack,execution:&Execution,g:&[f64],length:usize,m:usize)->Result<Vec<f64>,String>{
    let parts=execution.map((0..length).step_by(BLOCK_ROWS).collect(),|s:usize|->Result<Vec<f64>,String>{
        let rows=(s+BLOCK_ROWS).min(length)-s;let mut out=vec![0.;m*m];
        lapack.gemm(true,false,m,m,rows,1.,&g[s..],length,&g[s..],length,0.,&mut out,m)?;Ok(out)});
    let mut gram=vec![0.;m*m];for part in parts{for(a,b)in gram.iter_mut().zip(part?){*a+=b;}}
    Ok(gram)
}
/// G T (length x k) for a block G (length x k) and T (k x k), by row chunks.
fn block_times(lapack:&Lapack,execution:&Execution,g:&[f64],length:usize,k:usize,t:&[f64])->Result<Vec<f64>,String>{
    let starts:Vec<usize>=(0..length).step_by(BLOCK_ROWS).collect();
    let parts=execution.map(starts.clone(),|s|->Result<Vec<f64>,String>{
        let rows=(s+BLOCK_ROWS).min(length)-s;let mut out=vec![0.;rows*k];
        lapack.gemm(false,false,rows,k,k,1.,&g[s..],length,t,k,0.,&mut out,rows)?;Ok(out)});
    let mut out=vec![0.;length*k];
    for(&s,part)in starts.iter().zip(parts){let part=part?;let rows=(s+BLOCK_ROWS).min(length)-s;
        for c in 0..k{out[s+length*c..s+rows+length*c].copy_from_slice(&part[rows*c..rows*(c+1)]);}}
    Ok(out)
}
/// Cholesky factor (column-major k x k, upper, in pivot order) of a Gram
/// matrix, with diagonal pivoting or in the given order: a column whose
/// remaining norm is below CHOLESKY_DEPENDENT of its size is dropped.
/// Returns the order.
fn cholesky(gram:&[f64],m:usize,sizes:&[f64],pivoting:bool)->(Vec<usize>,Vec<f64>){
    let mut a=gram.to_vec();let mut left:Vec<usize>=(0..m).collect();let mut order=Vec::new();let mut rows:Vec<Vec<f64>>=Vec::new();
    loop{
        let next=if pivoting{left.iter().enumerate().max_by(|x,y|a[*x.1+m*x.1].total_cmp(&a[*y.1+m*y.1]))}else{left.iter().enumerate().next()};
        let Some((pos,&c))=next else{break};
        let d=a[c+m*c];if !(d>(CHOLESKY_DEPENDENT*sizes[c]).powi(2)){
            left.retain(|&j|a[j+m*j]>(CHOLESKY_DEPENDENT*sizes[j]).powi(2));if left.is_empty(){break;}continue;}
        let pivot=d.sqrt();left.remove(pos);
        // Row of R: r[c] = pivot, r[j] = a[c, j] / pivot for the columns left.
        let mut row=vec![0.;m];row[c]=pivot;for &j in &left{row[j]=a[c+m*j]/pivot;}
        for &i in &left{for &j in &left{a[i+m*j]-=row[i]*row[j];}}
        order.push(c);rows.push(row);
    }
    let k=order.len();let mut r=vec![0.;k*k];
    for(i,row)in rows.iter().enumerate(){for(j,&c)in order.iter().enumerate().skip(i){r[i+k*j]=row[c];}}
    (order,r)
}
/// The inverse of an upper triangular k x k matrix (column-major).
fn upper_inverse(r:&[f64],k:usize)->Vec<f64>{
    let mut inv=vec![0.;k*k];
    for j in 0..k{inv[j+k*j]=1./r[j+k*j];for i in(0..j).rev(){let s:f64=(i+1..=j).map(|l|r[i+k*l]*inv[l+k*j]).sum();inv[i+k*j]=-s/r[i+k*i];}}
    inv
}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}

pub struct SteadyRecycler<'p>{
    op:LatticeOperator<'p>,batch:FloquetBatch<'p>,lapack:Arc<Lapack>,execution:Execution,
    n:usize,watch:Vec<usize>,
    /// Orthonormal images and the triangular factor (column c has c + 1
    /// entries), in the order the directions arrived.
    q:Panels,r:Vec<Vec<f64>>,
    /// Augmentation directions so far.
    augmentations:usize,
    /// Centre populations of every direction.
    populations:Vec<Vec<f64>>,
    columns:usize,
    pub work:SteadyWork,
}

impl<'p> SteadyRecycler<'p>{
    pub fn new(prepared:&'p LatticePrepared,modes:&[Charge])->Result<Self,String>{
        let op=LatticeOperator::new(prepared,modes)?;
        let batch=Preconditioning::new(prepared)?.batch(&op,Shift::Steady,Some(Reach::All))?;
        let n=op.n;let c0=op.centre*n*n;let watch=(0..n).map(|i|c0+i*n+i).collect();let length=2*op.dimension();
        Ok(Self{op,batch,lapack:Arc::clone(&prepared.lapack),execution:Execution::new()?,n,watch,q:Panels::new(length),r:Vec::new(),
            augmentations:0,populations:Vec::new(),columns:0,work:SteadyWork::default()})
    }
    fn trace_term(&self,x:&[C],y:&mut[C]){
        let t=self.watch.iter().fold(ZERO,|s,&k|C{re:s.re+x[k].re,im:s.im+x[k].im});
        for &k in &self.watch{add(&mut y[k],scaled(t,1./self.n as f64));}
    }
    fn apply(&mut self,x:&[C])->Result<Vec<C>,String>{
        let mut y=vec![ZERO;x.len()];self.op.apply(x,&mut y)?;self.trace_term(x,&mut y);self.work.applications+=1;Ok(y)
    }
    fn centre_populations(&self,x:&[C])->Vec<f64>{self.watch.iter().map(|&k|x[k].re).collect()}
    /// Adds directions with their images (real columns, length x m): the
    /// images' parts outside Q join it, in the order of a pivoted Cholesky
    /// QR (twice) of the block, dependent images dropped. Returns the new
    /// orthonormal columns.
    fn push(&mut self,populations:Vec<Vec<f64>>,mut images:Vec<f64>)->Result<Vec<Vec<f64>>,String>{
        let clock=Instant::now();
        let(length,m)=(self.q.length,populations.len());
        let sizes:Vec<f64>=(0..m).map(|c|{let g=&images[c*length..(c+1)*length];dot(g,g).sqrt()}).collect();
        let mut h=vec![0.;self.q.count*m];
        for _ in 0..2{let a=self.q.dots(&self.lapack,&self.execution,&images,m)?;self.q.subtract(&self.lapack,&self.execution,&mut images,&a,m)?;
            for(x,y)in h.iter_mut().zip(a){*x+=y;}}
        // First pass, pivoted: the block's Gram matrix, its Cholesky factor
        // in pivot order with the dependent columns dropped.
        let gram=block_gram(&self.lapack,&self.execution,&images,length,m)?;
        let(order,r1)=cholesky(&gram,m,&sizes,true);
        let k=order.len();
        let mut selected=vec![0.;length*k];for(i,&c)in order.iter().enumerate(){selected[i*length..(i+1)*length].copy_from_slice(&images[c*length..(c+1)*length]);}
        drop(images);
        let q1=block_times(&self.lapack,&self.execution,&selected,length,k,&upper_inverse(&r1,k))?;drop(selected);
        // Second pass: the orthogonality CholQR leaves at the first.
        let gram=block_gram(&self.lapack,&self.execution,&q1,length,k)?;
        let(identity,r2)=cholesky(&gram,k,&vec![1.;k],false);
        if identity.len()!=k||identity.iter().enumerate().any(|(i,&j)|i!=j){return Err("Steady GCRO: the second Cholesky QR pass lost a column".into());}
        let q=block_times(&self.lapack,&self.execution,&q1,length,k,&upper_inverse(&r2,k))?;drop(q1);
        // R of the block: r2 r1 (upper triangular, pivot order).
        let base=self.q.count;let mut added=Vec::with_capacity(k);let mut populations=populations;
        for(i,&c)in order.iter().enumerate(){
            let mut column:Vec<f64>=(0..base).map(|j|h[j+base*c]).collect();
            column.extend((0..=i).map(|row|(row..=i).map(|l|r2[row+k*l]*r1[l+k*i]).sum::<f64>()));
            let g=q[i*length..(i+1)*length].to_vec();self.q.push(&g);self.r.push(column);
            self.populations.push(std::mem::take(&mut populations[c]));added.push(g);
        }
        self.work.push_ms+=clock.elapsed().as_secs_f64()*1e3;
        Ok(added)
    }
    /// The solution's centre populations for the coordinates c of f on Q.
    fn readout(&self,c:&[f64])->Vec<f64>{
        let m=c.len();let mut x=c.to_vec();
        for j in(0..m).rev(){x[j]/=self.r[j][j];for i in 0..j{x[i]-=self.r[j][i]*x[j];}}
        let mut out=vec![0.;self.n];for(j,&a)in x.iter().enumerate(){for(o,&p)in out.iter_mut().zip(&self.populations[j]){*o+=a*p;}}
        out
    }
    /// One augmentation direction z with its image: joins Q, and f's
    /// coordinates and residual follow.
    fn augment(&mut self,z:&[C],c:&mut Vec<f64>,res:&mut[f64])->Result<(),String>{
        let image=self.apply(z)?;self.augmentations+=1;
        let mut g=Vec::with_capacity(self.q.length);reals(&image,&mut g);
        for q in self.push(vec![self.centre_populations(z)],g)?{let a=dot(&q,res);c.push(a);for(t,v)in res.iter_mut().zip(&q){*t-=a*v;}}
        Ok(())
    }
    /// The centre populations of the steady error -A^# f (f Hermitian-symmetric).
    pub fn error(&mut self,basis:&[Vec<C>],images:&[Vec<C>],f:&[C],steady:&mut dyn SteadySolves)->Result<Vec<f64>,String>{
        if basis.len()>self.columns{
            let mut block=Vec::with_capacity(self.q.length*(basis.len()-self.columns));let mut pops=Vec::new();
            for j in self.columns..basis.len(){let mut image=images[j].clone();self.trace_term(&basis[j],&mut image);reals(&image,&mut block);pops.push(self.centre_populations(&basis[j]));}
            self.push(pops,block)?;
        }
        self.columns=basis.len();
        let mut res=Vec::with_capacity(self.q.length);reals(f,&mut res);
        let fsize=dot(&res,&res).sqrt();if fsize==0.{return Ok(vec![0.;self.n]);}
        let mut c=self.q.dots(&self.lapack,&self.execution,&res,1)?;self.q.subtract(&self.lapack,&self.execution,&mut res,&c,1)?;
        if self.augmentations==0{
            let(y,steps)=steady.solve(f,STEADY_RESIDUAL)?;self.work.pipeline_solves+=1;self.work.inner_steps+=steps;
            self.augment(&y,&mut c,&mut res)?;
        }
        let mut history=vec![self.readout(&c)];
        let mut steps=0;
        loop{
            let rn=dot(&res,&res).sqrt();
            if rn<=1e-13*fsize{break;}
            if history.len()>=3&&rn<=STEADY_RESIDUAL*fsize{
                let h=&history;let last=&h[h.len()-1];let size=last.iter().fold(0_f64,|a,&b|a.max(b.abs()));
                let change=|a:&[f64],b:&[f64]|a.iter().zip(b).fold(0_f64,|m,(x,y)|m.max((x-y).abs()));
                if change(last,&h[h.len()-2])<=STABLE*size&&change(&h[h.len()-2],&h[h.len()-3])<=STABLE*size{break;}
            }
            if steps>=MAXIMUM_STEPS{break;}
            let scaled:Vec<C>=res.chunks_exact(2).map(|v|C{re:v[0]/rn,im:v[1]/rn}).collect();
            let clock=Instant::now();
            let mut z=vec![ZERO;scaled.len()];self.batch.apply(&scaled,&mut z)?;self.work.preconditions+=1;
            self.augment(&z,&mut c,&mut res)?;self.work.step_ms+=clock.elapsed().as_secs_f64()*1e3;
            steps+=1;history.push(self.readout(&c));
        }
        self.work.steps+=steps;
        Ok(history.last().unwrap().iter().map(|v|-v).collect())
    }
    /// Directions with independent images (basis columns and augmentations).
    pub fn directions(&self)->usize{self.q.count}
}
