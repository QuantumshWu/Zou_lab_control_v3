//! The slow Ritz modes, exact (R2 section 11.2).
//!
//! For a Ritz mode k with |theta_k| < SLOW (the kernel included) the error
//! its source F_k drives is, exactly,
//!   e_k(t) = a_k (e^{theta_k t} Y_k - e^{tA} Y_k),  Y_k = (theta_k - A)^-1 F_k,
//! with F_k = (A W - theta_k W) b_k plus the omitted ring slaved at
//! z = theta_k and fed back (`ErrorLattice::ring_feedback`). The kernel
//! mode's Y (from the bordered system (A + u tau0^T) y = F, Y = -y) is the
//! steady error itself.
//!
//! One shared pool answers every shift: the basis columns W with A W, the
//! trace-row vector u, and every augmentation direction z with A z, their
//! span kept orthonormal (Q, row panels, extended block by block) with every
//! direction's and image's coordinates. For any shift the images of the pool
//! are (theta U - A U) in those coordinates, so GCRO for (theta_k, F_k)
//! costs the projection of F_k, a small least-squares problem, and, while
//! the residual or the readout has not settled, preconditioned augmentation
//! steps that join the pool for every later mode and checkpoint. The
//! preconditioner is the solver's own: the shared Floquet frames at the
//! steady shift, and the first direction is the pipeline's bordered solve
//! (`SteadySolves`) of the first kernel source.
//!
//! e^{tA} Y_k: Y's part in the space, W X r0 (r0 its least-squares
//! coordinates), is propagated by the reduced model, W X S e^{theta t} S^-1 r0;
//! the rest decays at the fast rates and is propagated by the dressed blocks
//! of L0' on the near modes only (its readout elsewhere is micromotion).
use super::super::*;
use super::super::estimate_api::SteadySolves;
use super::super::operator::LatticeOperator;
use super::super::preconditioner::{FloquetBatch,Preconditioner,Preconditioning,Shift};
use super::super::floquet::Reach;
use super::lattice::{ErrorLattice,split};
use super::propagate::Readout;
use crate::parallel::Execution;

/// Ritz modes with |theta| below this (1/us) are answered exactly.
pub const SLOW:f64=0.05;
/// GCRO stops when the residual is below RESIDUAL |F| and the readout has
/// settled (two consecutive changes below STABLE of its size).
const RESIDUAL:f64=1e-4;
const STABLE:f64=0.1;
const MAXIMUM_STEPS:usize=60;
/// A Ritz value below this is the kernel (theta = 0 exactly in the outputs).
const KERNEL:f64=1e-12;
/// Real rows per panel of Q and per task of the block products.
const PANEL_ROWS:usize=1<<15;
/// A block column whose part outside the block's earlier pivots is below
/// this fraction of its size adds no direction (a Gram matrix resolves
/// singular values down to about the square root of the roundoff).
const CHOLESKY_DEPENDENT:f64=1e-7;
/// A column of the small least-squares problem below this fraction of its
/// size outside the earlier ones is left out.
const SMALL_DEPENDENT:f64=1e-12;

fn reals(x:&[C])->Vec<f64>{let mut out=Vec::with_capacity(2*x.len());for v in x{out.push(v.re);out.push(v.im);}out}
fn complex(x:&[f64])->Vec<C>{x.chunks_exact(2).map(|v|C{re:v[0],im:v[1]}).collect()}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn cexp(z:C,t:f64)->C{let m=(z.re*t).exp();if m==0.{return ZERO;}C{re:m*(z.im*t).cos(),im:m*(z.im*t).sin()}}

/// Orthonormal real columns stored by row panels (each column-major).
struct Panels{length:usize,starts:Vec<usize>,q:Vec<Vec<f64>>,count:usize}
impl Panels{
    fn new(length:usize)->Self{let starts:Vec<usize>=(0..length).step_by(PANEL_ROWS).collect();let panels=starts.len();Self{length,starts,q:vec![Vec::new();panels],count:0}}
    fn rows(&self,p:usize)->(usize,usize){(self.starts[p],self.starts.get(p+1).copied().unwrap_or(self.length))}
    fn push(&mut self,column:&[f64]){for p in 0..self.starts.len(){let(s,e)=self.rows(p);self.q[p].extend_from_slice(&column[s..e]);}self.count+=1;}
    /// Q^T g (count x m) for g (length x m), column-major.
    fn dots(&self,lapack:&Lapack,execution:&Execution,g:&[f64],m:usize)->Result<Vec<f64>,String>{
        let count=self.count;if count==0||m==0{return Ok(vec![0.;count*m]);}
        let parts=execution.map((0..self.starts.len()).collect(),|p|->Result<Vec<f64>,String>{
            let(s,e)=self.rows(p);let mut out=vec![0.;count*m];
            lapack.gemm(true,false,count,m,e-s,1.,&self.q[p],e-s,&g[s..],self.length,0.,&mut out,count)?;Ok(out)});
        let mut h=vec![0.;count*m];for part in parts{for(a,b)in h.iter_mut().zip(part?){*a+=b;}}
        Ok(h)
    }
    /// g += alpha Q h.
    fn combine(&self,lapack:&Lapack,execution:&Execution,g:&mut[f64],h:&[f64],m:usize,alpha:f64)->Result<(),String>{
        let count=self.count;if count==0||m==0{return Ok(());}
        let parts=execution.map((0..self.starts.len()).collect(),|p|->Result<Vec<f64>,String>{
            let(s,e)=self.rows(p);let mut out=vec![0.;(e-s)*m];
            lapack.gemm(false,false,e-s,m,count,1.,&self.q[p],e-s,h,count,0.,&mut out,e-s)?;Ok(out)});
        for(p,part)in parts.into_iter().enumerate(){let part=part?;let(s,e)=self.rows(p);let rows=e-s;
            for c in 0..m{for(t,v)in g[s+self.length*c..e+self.length*c].iter_mut().zip(&part[rows*c..rows*(c+1)]){*t+=alpha*v;}}}
        Ok(())
    }
    /// Extends Q by the span of the columns g (length x m): block CGS2
    /// against Q, a pivoted Cholesky QR twice inside the block. Returns the
    /// coordinates of every column on the extended Q and the new columns.
    fn extend(&mut self,lapack:&Lapack,execution:&Execution,mut g:Vec<f64>,m:usize)->Result<(Vec<Vec<f64>>,Vec<Vec<f64>>),String>{
        let length=self.length;let base=self.count;
        let sizes:Vec<f64>=(0..m).map(|c|{let x=&g[c*length..(c+1)*length];dot(x,x).sqrt()}).collect();
        let mut h=vec![0.;base*m];
        for _ in 0..2{let a=self.dots(lapack,execution,&g,m)?;self.combine(lapack,execution,&mut g,&a,m,-1.)?;for(x,y)in h.iter_mut().zip(a){*x+=y;}}
        let gram=block_cross(lapack,execution,&g,m,&g,m,length)?;
        let(order,r1)=cholesky(&gram,m,&sizes,true);let k=order.len();
        let mut fresh:Vec<Vec<f64>>=Vec::new();let mut t=vec![0.;k*m];
        if k>0{
            let mut selected=vec![0.;length*k];for(i,&c)in order.iter().enumerate(){selected[i*length..(i+1)*length].copy_from_slice(&g[c*length..(c+1)*length]);}
            let q1=block_times(lapack,execution,&selected,length,k,&upper_inverse(&r1,k))?;drop(selected);
            let gram=block_cross(lapack,execution,&q1,k,&q1,k,length)?;
            let(kept,r2)=cholesky(&gram,k,&vec![1.;k],false);
            if kept.len()!=k{return Err("Slow pool: the second Cholesky QR pass lost a column".into());}
            let q=block_times(lapack,execution,&q1,length,k,&upper_inverse(&r2,k))?;drop(q1);
            t=block_cross(lapack,execution,&q,k,&g,m,length)?;
            for i in 0..k{let column=q[i*length..(i+1)*length].to_vec();self.push(&column);fresh.push(column);}
        }
        let coordinates=(0..m).map(|c|{let mut v:Vec<f64>=(0..base).map(|i|h[i+base*c]).collect();v.extend((0..k).map(|i|t[i+k*c]));v}).collect();
        Ok((coordinates,fresh))
    }
}
/// A^T B (ka x mb, column-major) of blocks A (length x ka) and B (length x mb), by row chunks.
fn block_cross(lapack:&Lapack,execution:&Execution,a:&[f64],ka:usize,b:&[f64],mb:usize,length:usize)->Result<Vec<f64>,String>{
    if ka==0||mb==0{return Ok(vec![0.;ka*mb]);}
    let parts=execution.map((0..length).step_by(PANEL_ROWS).collect(),|s:usize|->Result<Vec<f64>,String>{
        let rows=(s+PANEL_ROWS).min(length)-s;let mut out=vec![0.;ka*mb];
        lapack.gemm(true,false,ka,mb,rows,1.,&a[s..],length,&b[s..],length,0.,&mut out,ka)?;Ok(out)});
    let mut out=vec![0.;ka*mb];for part in parts{for(x,y)in out.iter_mut().zip(part?){*x+=y;}}
    Ok(out)
}
/// G T (length x k) for G (length x k) and T (k x k), by row chunks.
fn block_times(lapack:&Lapack,execution:&Execution,g:&[f64],length:usize,k:usize,t:&[f64])->Result<Vec<f64>,String>{
    let starts:Vec<usize>=(0..length).step_by(PANEL_ROWS).collect();
    let parts=execution.map(starts.clone(),|s|->Result<Vec<f64>,String>{
        let rows=(s+PANEL_ROWS).min(length)-s;let mut out=vec![0.;rows*k];
        lapack.gemm(false,false,rows,k,k,1.,&g[s..],length,t,k,0.,&mut out,rows)?;Ok(out)});
    let mut out=vec![0.;length*k];
    for(&s,part)in starts.iter().zip(parts){let part=part?;let rows=(s+PANEL_ROWS).min(length)-s;
        for c in 0..k{out[s+length*c..s+rows+length*c].copy_from_slice(&part[rows*c..rows*(c+1)]);}}
    Ok(out)
}
/// Upper Cholesky factor (column-major, in the returned order) of a Gram
/// matrix, with diagonal pivoting or in the given order; a column whose
/// remaining norm is below CHOLESKY_DEPENDENT of its size is dropped.
fn cholesky(gram:&[f64],m:usize,sizes:&[f64],pivoting:bool)->(Vec<usize>,Vec<f64>){
    let mut a=gram.to_vec();let mut left:Vec<usize>=(0..m).collect();let mut order=Vec::new();let mut rows:Vec<Vec<f64>>=Vec::new();
    let small=|a:&[f64],j:usize|!(a[j+m*j]>(CHOLESKY_DEPENDENT*sizes[j]).powi(2));
    loop{
        left.retain(|&j|!small(&a,j));
        let next=if pivoting{left.iter().enumerate().max_by(|x,y|a[*x.1+m*x.1].total_cmp(&a[*y.1+m*y.1]))}else{left.iter().enumerate().next()};
        let Some((pos,&c))=next else{break};
        let pivot=a[c+m*c].sqrt();left.remove(pos);
        let mut row=vec![0.;m];row[c]=pivot;for &j in &left{row[j]=a[c+m*j]/pivot;}
        for &i in &left{for &j in &left{a[i+m*j]-=row[i]*row[j];}}
        order.push(c);rows.push(row);
    }
    let k=order.len();let mut r=vec![0.;k*k];
    for(i,row)in rows.iter().enumerate(){for(j,&c)in order.iter().enumerate().skip(i){r[i+k*j]=row[c];}}
    (order,r)
}
fn upper_inverse(r:&[f64],k:usize)->Vec<f64>{
    let mut inv=vec![0.;k*k];
    for j in 0..k{inv[j+k*j]=1./r[j+k*j];for i in(0..j).rev(){let s:f64=(i+1..=j).map(|l|r[i+k*l]*inv[l+k*j]).sum();inv[i+k*j]=-s/r[i+k*i];}}
    inv
}
/// min |b - M x| for a complex M (rows x cols, column-major) by CGS2:
/// x (zero on dependent columns) and the residual b - M x.
fn least_squares(m:&[C],rows:usize,cols:usize,b:&[C])->(Vec<C>,Vec<C>){
    let mut q:Vec<Vec<C>>=Vec::new();let mut r:Vec<(usize,Vec<C>)>=Vec::new();
    let cdot=|a:&[C],b:&[C]|a.iter().zip(b).fold(ZERO,|s,(x,y)|{let v=mul(conj(*x),*y);C{re:s.re+v.re,im:s.im+v.im}});
    for j in 0..cols{
        let mut v=m[j*rows..(j+1)*rows].to_vec();let size=v.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        let mut h=vec![ZERO;q.len()];
        for _ in 0..2{for(i,qi)in q.iter().enumerate(){let a=cdot(qi,&v);add(&mut h[i],a);for(t,w)in v.iter_mut().zip(qi){let x=mul(a,*w);t.re-=x.re;t.im-=x.im;}}}
        let eta=v.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        if !(eta>SMALL_DEPENDENT*size){continue;}
        for t in &mut v{t.re/=eta;t.im/=eta;}
        h.push(C{re:eta,im:0.});q.push(v);r.push((j,h));
    }
    let c:Vec<C>=q.iter().map(|qi|cdot(qi,b)).collect();
    let mut residual=b.to_vec();for(qi,&a)in q.iter().zip(&c){for(t,w)in residual.iter_mut().zip(qi){let x=mul(a,*w);t.re-=x.re;t.im-=x.im;}}
    let k=q.len();let mut y=c.clone();
    for i in(0..k).rev(){
        for l in i+1..k{let v=mul(r[l].1[i],y[l]);y[i].re-=v.re;y[i].im-=v.im;}
        let d=r[i].1[i];let s=d.re*d.re+d.im*d.im;y[i]=mul(y[i],C{re:d.re/s,im:-d.im/s});
    }
    let mut x=vec![ZERO;cols];for(i,(j,_))in r.iter().enumerate(){x[*j]=y[i];}
    (x,residual)
}

#[derive(Clone,Copy,Debug)]
enum Direction{Basis(usize),Augmented(usize)}

#[derive(Clone,Debug,Default)]
pub struct PoolWork{
    /// Operator and preconditioner applications on P u Q, the pipeline's
    /// bordered solves and their inner steps, GCRO steps and solves (one
    /// per slow mode and checkpoint), ring block solves inside the ring
    /// feedback are counted by the lattice.
    pub applications:usize,pub preconditions:usize,pub pipeline_solves:usize,pub inner_steps:usize,pub steps:usize,pub solves:usize,
    /// Near-mode coordinate changes of the slow modes' remainders.
    pub frame_blocks:usize,
    /// Time (ms): extending Q, GCRO steps, the rest of the solves.
    pub extend_ms:f64,pub step_ms:f64,pub solve_ms:f64,
}

/// One slow mode's solution: Y = sign * U x over the pool's directions.
pub struct SlowSolution{pub theta:C,pub coefficient:C,x:Vec<C>,sign:f64}

pub struct SlowPool<'p>{
    op:LatticeOperator<'p>,batch:FloquetBatch<'p>,lapack:Arc<Lapack>,execution:Execution,
    n:usize,d:usize,watch:Vec<usize>,mirror:Vec<usize>,near:Vec<usize>,
    q:Panels,
    /// Pool directions and the coordinates (on Q at insertion) of each and
    /// of its image, its centre trace, the trace-row vector's coordinates.
    directions:Vec<Direction>,cu:Vec<Vec<f64>>,cau:Vec<Vec<f64>>,trace:Vec<f64>,u:Vec<f64>,
    /// Near-mode parts of the augmentation directions (near.len() blocks each).
    augmented:Vec<Vec<C>>,
    /// Whether the first kernel direction comes from the pipeline's solve.
    pub seed_with_pipeline:bool,
    columns:usize,
    pub work:PoolWork,
}

impl<'p> SlowPool<'p>{
    pub fn new(prepared:&'p LatticePrepared,lattice:&ErrorLattice<'_>)->Result<Self,String>{
        let op=LatticeOperator::new(prepared,&lattice.modes)?;
        let batch=Preconditioning::new(prepared)?.batch(&op,Shift::Steady,Some(Reach::All))?;
        let(n,d)=(lattice.n,lattice.d);let c0=lattice.centre*d;let watch:Vec<usize>=(0..n).map(|i|c0+i*n+i).collect();
        let length=2*op.dimension();
        let mut pool=Self{op,batch,lapack:Arc::clone(&prepared.lapack),execution:Execution::new()?,n,d,watch,mirror:lattice.mirror.clone(),near:lattice.near.clone(),
            q:Panels::new(length),directions:Vec::new(),cu:Vec::new(),cau:Vec::new(),trace:Vec::new(),u:Vec::new(),augmented:Vec::new(),
            seed_with_pipeline:true,columns:0,work:PoolWork::default()};
        let mut u=vec![ZERO;length/2];for &k in &pool.watch{u[k]=C{re:1./n as f64,im:0.};}
        let(mut coordinates,_)=pool.q.extend(&pool.lapack,&pool.execution,reals(&u),1)?;pool.u=coordinates.remove(0);
        Ok(pool)
    }
    fn centre_trace(&self,x:&[C])->f64{self.watch.iter().map(|&k|x[k].re).sum()}
    fn apply(&mut self,x:&[C])->Result<Vec<C>,String>{let mut y=vec![ZERO;x.len()];self.op.apply(x,&mut y)?;self.work.applications+=1;Ok(y)}
    /// Adds directions with their images: Q extended by both, the
    /// coordinates kept. Returns Q's new columns.
    fn add(&mut self,directions:Vec<(Direction,Vec<C>,Vec<C>)>)->Result<Vec<Vec<f64>>,String>{
        let clock=Instant::now();let m=directions.len();let length=self.q.length;
        let mut block=Vec::with_capacity(2*m*length);for(_,x,ax)in &directions{block.extend(reals(x));block.extend(reals(ax));}
        let(coordinates,fresh)=self.q.extend(&self.lapack,&self.execution,block,2*m)?;
        for(i,(kind,x,_))in directions.into_iter().enumerate(){
            self.trace.push(self.centre_trace(&x));self.cu.push(coordinates[2*i].clone());self.cau.push(coordinates[2*i+1].clone());self.directions.push(kind);
            if let Direction::Augmented(_)=kind{let d=self.d;self.augmented.push(self.near.iter().flat_map(|&q|x[q*d..(q+1)*d].iter().copied()).collect());}
        }
        self.work.extend_ms+=clock.elapsed().as_secs_f64()*1e3;
        Ok(fresh)
    }
    /// The basis columns not in the pool yet.
    pub fn add_columns(&mut self,basis:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        if basis.len()>self.columns{
            let new:Vec<(Direction,Vec<C>,Vec<C>)>=(self.columns..basis.len()).map(|j|(Direction::Basis(j),basis[j].clone(),images[j].clone())).collect();
            self.add(new)?;self.columns=basis.len();
        }
        Ok(())
    }
    /// Pool image coordinates of direction j at the shift: theta u_j - A u_j,
    /// or at the kernel the bordered A u_j + u tau0(u_j).
    fn image(&self,j:usize,theta:C,kernel:bool,count:usize)->Vec<C>{
        let mut v=vec![ZERO;count];
        if kernel{for(i,&a)in self.cau[j].iter().enumerate(){v[i].re+=a;}for(i,&a)in self.u.iter().enumerate(){v[i].re+=a*self.trace[j];}}
        else{for(i,&a)in self.cu[j].iter().enumerate(){v[i]=scaled(theta,a);}for(i,&a)in self.cau[j].iter().enumerate(){v[i].re-=a;}}
        v
    }
    /// Centre populations (real parts) of U x.
    fn readout(&self,basis:&[Vec<C>],x:&[C])->Vec<f64>{
        let(n,d)=(self.n,self.d);let centre_slot=self.near.iter().position(|&q|q==self.watch[0]/d).unwrap_or(0);
        let mut out=vec![0.;n];
        for(j,kind)in self.directions.iter().enumerate().take(x.len()){let c=x[j];if !nonzero(c){continue;}
            for(i,o)in out.iter_mut().enumerate(){let v=match kind{Direction::Basis(b)=>basis[*b][self.watch[i]],Direction::Augmented(a)=>self.augmented[*a][centre_slot*d+i*n+i]};*o+=mul(c,v).re;}}
        out
    }
    /// Y = (theta - A)^-1 f (at the kernel the trace-free solution -y of
    /// (A + u tau0^T) y = f) by GCRO on the pool.
    pub fn solve(&mut self,basis:&[Vec<C>],f:&[C],theta:C,coefficient:C,steady:&mut dyn SteadySolves)->Result<SlowSolution,String>{
        let clock=Instant::now();self.work.solves+=1;
        let kernel=magnitude(theta)<KERNEL;let length=self.q.length;
        // f = e^{i phi} (g1 + i g2) with g1, g2 Hermitian-symmetric and
        // orthogonal, g1 the larger: a real problem when theta is real.
        let(f1,f2)=split(f,&self.mirror,self.n);let(a,b)=(reals(&f1),reals(&f2));
        let(aa,bb,ab)=(dot(&a,&a),dot(&b,&b),dot(&a,&b));let phi=0.5*(2.*ab).atan2(aa-bb);let(cs,sn)=(phi.cos(),phi.sin());
        let g1:Vec<f64>=a.iter().zip(&b).map(|(x,y)|cs*x+sn*y).collect();let g2:Vec<f64>=a.iter().zip(&b).map(|(x,y)|-sn*x+cs*y).collect();
        let fsize=(dot(&g1,&g1)+dot(&g2,&g2)).sqrt();
        if fsize==0.{return Ok(SlowSolution{theta,coefficient,x:vec![ZERO;self.directions.len()],sign:1.});}
        // Coordinates on Q and the parts outside it.
        let mut parts=g1.clone();parts.extend(&g2);
        let h=self.q.dots(&self.lapack,&self.execution,&parts,2)?;self.q.combine(&self.lapack,&self.execution,&mut parts,&h,2,-1.)?;
        let count=self.q.count;let mut phi_c:Vec<C>=(0..count).map(|i|C{re:h[i],im:h[i+count]}).collect();
        let mut history:Vec<Vec<f64>>=Vec::new();let mut steps=0;
        let solution=loop{
            let(count,p)=(self.q.count,self.directions.len());
            let mut m=Vec::with_capacity(count*p);for j in 0..p{m.extend(self.image(j,theta,kernel,count));}
            let(x,rho)=least_squares(&m,count,p,&phi_c);
            let outside=dot(&parts,&parts);let rn=(rho.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>()+outside).sqrt();
            history.push(self.readout(basis,&x));
            let settled=history.len()>=3&&rn<=RESIDUAL*fsize&&{
                let h=&history;let last=&h[h.len()-1];let size=last.iter().fold(0_f64,|a,&b|a.max(b.abs()));
                let change=|a:&[f64],b:&[f64]|a.iter().zip(b).fold(0_f64,|m,(x,y)|m.max((x-y).abs()));
                change(last,&h[h.len()-2])<=STABLE*size&&change(&h[h.len()-2],&h[h.len()-3])<=STABLE*size};
            if settled||rn<=1e-13*fsize||steps>=MAXIMUM_STEPS{break x;}
            // The residual field r = Q rho + (the parts outside Q), split in two real fields.
            let mut r=parts.clone();
            let rho_parts:Vec<f64>=rho.iter().map(|z|z.re).chain(rho.iter().map(|z|z.im)).collect();
            self.q.combine(&self.lapack,&self.execution,&mut r,&rho_parts,2,1.)?;
            let step=Instant::now();
            let mut new=Vec::new();
            for part in r.chunks_exact(length){
                let size=dot(part,part).sqrt();if !(size>1e-3*rn){continue;}
                let field:Vec<C>=complex(part).iter().map(|v|C{re:v.re/size,im:v.im/size}).collect();
                let z=if self.seed_with_pipeline&&kernel&&self.augmented.is_empty()&&new.is_empty(){
                    let(y,inner)=steady.solve(&field,RESIDUAL)?;self.work.pipeline_solves+=1;self.work.inner_steps+=inner;y
                }else{let mut z=vec![ZERO;field.len()];self.batch.apply(&field,&mut z)?;self.work.preconditions+=1;z};
                let az=self.apply(&z)?;new.push((Direction::Augmented(self.augmented.len()+new.len()),z,az));
            }
            if new.is_empty(){break x;}
            let fresh=self.add(new)?;
            // f's coordinates on the new columns, and its parts outside Q.
            for q in &fresh{let(c1,c2)=(dot(q,&parts[..length]),dot(q,&parts[length..]));phi_c.push(C{re:c1,im:c2});
                for(t,v)in parts[..length].iter_mut().zip(q){*t-=c1*v;}for(t,v)in parts[length..].iter_mut().zip(q){*t-=c2*v;}}
            steps+=1;self.work.steps+=1;self.work.step_ms+=step.elapsed().as_secs_f64()*1e3;
        };
        // Back from the rotation: Y = sign e^{i phi} U x.
        let rotation=C{re:cs,im:sn};
        let x=solution.iter().map(|&v|mul(rotation,v)).collect();
        self.work.solve_ms+=clock.elapsed().as_secs_f64()*1e3;
        Ok(SlowSolution{theta,coefficient,x,sign:if kernel{-1.}else{1.}})
    }
    /// The Gram matrix W^T U of the basis against every pool direction
    /// (column-major k x p), from the pool coordinates.
    fn basis_gram(&self,k:usize)->Vec<f64>{
        let p=self.directions.len();let mut g=vec![0.;k*p];
        for(i,kind)in self.directions.iter().enumerate(){let Direction::Basis(b)=*kind else{continue};if b>=k{continue;}
            for j in 0..p{let l=self.cu[i].len().min(self.cu[j].len());g[b+k*j]=dot(&self.cu[i][..l],&self.cu[j][..l]);}}
        g
    }
    /// W^H Y (from the Gram matrix) and Y on the near modes.
    fn parts(&self,basis:&[Vec<C>],gram:&[f64],s:&SlowSolution)->(Vec<C>,Vec<C>){
        // Directions added after this solution carry no coefficient.
        let d=self.d;let k=basis.len();let p=s.x.len();
        let coordinates:Vec<C>=(0..k).map(|i|scaled((0..p).fold(ZERO,|v,j|{let w=scaled(s.x[j],gram[i+k*j]);C{re:v.re+w.re,im:v.im+w.im}}),s.sign)).collect();
        let mut near=vec![ZERO;self.near.len()*d];
        for(j,kind)in self.directions.iter().enumerate().take(p){let c=scaled(s.x[j],s.sign);if !nonzero(c){continue;}
            for(slot,&q)in self.near.iter().enumerate(){let block:&[C]=match kind{Direction::Basis(b)=>&basis[*b][q*d..(q+1)*d],Direction::Augmented(a)=>&self.augmented[*a][slot*d..(slot+1)*d]};
                for(t,&v)in near[slot*d..(slot+1)*d].iter_mut().zip(block){mul_add(t,c,v);}}}
        (coordinates,near)
    }
    /// The centre populations of a_k Y_k (a kernel mode's: its steady error).
    pub fn steady_populations(&self,basis:&[Vec<C>],s:&SlowSolution)->Vec<f64>{
        let c=mul(s.coefficient,C{re:s.sign,im:0.});
        let y=self.readout(basis,&s.x.iter().map(|&v|mul(c,v)).collect::<Vec<_>>());
        y
    }
    /// Population errors (one row per time) of the slow modes,
    /// sum_k a_k (e^{theta_k t} Y_k - e^{tA} Y_k), for the reduced model's
    /// coordinates X (k x r), eigenvectors S (r x r, column-major) and the
    /// values theta (r) the outputs use.
    #[allow(clippy::too_many_arguments)]
    pub fn populations(&mut self,lattice:&ErrorLattice<'_>,basis:&[Vec<C>],x:&[C],s:&[C],theta:&[C],solutions:&[SlowSolution],times:&[f64],readout:Readout)->Result<Vec<Vec<f64>>,String>{
        let(n,d,k,r)=(self.n,self.d,basis.len(),theta.len());let t_count=times.len();
        if solutions.is_empty()||t_count==0{return Ok(vec![vec![0.;n];t_count]);}
        let lapack=Arc::clone(&self.lapack);let gemm=lapack.complex_gemm();
        // (X^H X) and S factored once; B = X S.
        let mut xhx=vec![ZERO;r*r];
        for a in 0..r{for b in 0..r{let mut v=ZERO;for i in 0..k{mul_add(&mut v,conj(x[i+k*a]),x[i+k*b]);}xhx[a+r*b]=v;}}
        let(lu_x,piv_x)=lapack.complex_factor(&xhx,r)?;let(lu_s,piv_s)=lapack.complex_factor(s,r)?;
        let mut b=vec![ZERO;k*r];gemm.multiply(k,r,r,ONE,x,k,s,r,ZERO,&mut b,k)?;
        let gram=self.basis_gram(k);
        let modes:Vec<usize>=match readout{Readout::Laboratory=>(0..lattice.modes.len()).collect(),Readout::Centre=>vec![lattice.centre]};
        let phase=|q:usize,t:f64|match readout{Readout::Laboratory=>cexp(C{re:0.,im:lattice.nu[q]},t),Readout::Centre=>ONE};
        let mut total=vec![ZERO;n*t_count];
        // Per slow mode: X r0 (Y's part in the space) and the remainder on the near modes.
        let mut ritz_weights=vec![ZERO;r];let mut in_space:Vec<Vec<C>>=Vec::new();let mut remainders:Vec<(usize,C,C,Vec<C>)>=Vec::new();
        for sol in solutions{
            let(wy,near)=self.parts(basis,&gram,sol);
            let xh:Vec<C>=(0..r).map(|a|(0..k).fold(ZERO,|v,i|{let w=mul(conj(x[i+k*a]),wy[i]);C{re:v.re+w.re,im:v.im+w.im}})).collect();
            let r0=lapack.complex_solve_matrix_factored(&lu_x,&piv_x,&xh,r)?;let cs=lapack.complex_solve_matrix_factored(&lu_s,&piv_s,&r0,r)?;
            for(w,c)in ritz_weights.iter_mut().zip(&cs){mul_add(w,sol.coefficient,*c);}
            let xr:Vec<C>=(0..k).map(|i|(0..r).fold(ZERO,|v,a|{let w=mul(x[i+k*a],r0[a]);C{re:v.re+w.re,im:v.im+w.im}})).collect();
            for(slot,&q)in self.near.iter().enumerate(){
                if readout==Readout::Centre&&q!=lattice.centre{continue;}
                let mut block=near[slot*d..(slot+1)*d].to_vec();
                for(j,&c)in xr.iter().enumerate(){if !nonzero(c){continue;}for(t,&v)in block.iter_mut().zip(&basis[j][q*d..(q+1)*d]){let w=mul(c,v);t.re-=w.re;t.im-=w.im;}}
                remainders.push((q,sol.coefficient,sol.theta,block));
            }
            in_space.push(xr);
        }
        // In the space, per mode: sum_k a_k e^{theta_k t} P_q(W X r0_k) - P_q(B (sum_k a_k cS_k) e^{theta t}).
        let ritz_exp:Vec<C>=times.iter().flat_map(|&t|theta.iter().zip(&ritz_weights).map(move|(&z,&w)|mul(w,cexp(z,t)))).collect();
        for &q in &modes{
            let mut rows=vec![ZERO;n*k];for j in 0..k{for i in 0..n{rows[i+n*j]=basis[j][q*d+i*n+i];}}
            let mut rb=vec![ZERO;n*r];gemm.multiply(n,r,k,ONE,&rows,n,&b,k,ZERO,&mut rb,n)?;
            let mut reduced=vec![ZERO;n*t_count];gemm.multiply(n,t_count,r,ONE,&rb,n,&ritz_exp,r,ZERO,&mut reduced,n)?;
            let mut constants=Vec::with_capacity(solutions.len());
            for(sol,xr)in solutions.iter().zip(&in_space){let mut v=vec![ZERO;n];gemm.multiply(n,1,k,ONE,&rows,n,xr,k,ZERO,&mut v,n)?;constants.push((sol.coefficient,sol.theta,v));}
            for(ti,&t)in times.iter().enumerate(){let ph=phase(q,t);
                for i in 0..n{let mut v=C{re:-reduced[i+n*ti].re,im:-reduced[i+n*ti].im};
                    for(a,th,c)in &constants{mul_add(&mut v,mul(*a,cexp(*th,t)),c[i]);}
                    mul_add(&mut total[i+n*ti],ph,v);}}
        }
        // The remainders on the near modes: e^{i nu_q t} e^{theta_k t} Y_perp - its dressed-block propagation.
        if !remainders.is_empty(){
            let lambda:Vec<C>=lattice.basis.lambda.iter().map(|z|C{re:z.re.min(0.),im:z.im}).collect();
            let block:Vec<C>=times.iter().flat_map(|&t|lambda.iter().map(move|&z|cexp(z,t))).collect();
            let readout_rows=lattice.basis.population_rows();
            let blocks:Vec<&[C]>=remainders.iter().map(|x|&x.3[..]).collect();
            let coefficients=lattice.basis.coefficients(&blocks)?;self.work.frame_blocks+=remainders.len();
            for(i,(q,a,th,y))in remainders.iter().enumerate(){
                let mut h=readout_rows.to_vec();for j in 0..d{let c=mul(*a,coefficients[j+d*i]);for row in 0..n{h[row+n*j]=mul(h[row+n*j],c);}}
                let mut dec=vec![ZERO;n*t_count];gemm.multiply(n,t_count,d,ONE,&h,n,&block,d,ZERO,&mut dec,n)?;
                for(ti,&t)in times.iter().enumerate(){let ph=mul(phase(*q,t),mul(*a,cexp(*th,t)));
                    for row in 0..n{let v=mul(ph,y[row*n+row]);total[row+n*ti].re+=v.re-dec[row+n*ti].re;total[row+n*ti].im+=v.im-dec[row+n*ti].im;}}
            }
        }
        Ok((0..t_count).map(|ti|(0..n).map(|i|total[i+n*ti].re).collect()).collect())
    }
}
