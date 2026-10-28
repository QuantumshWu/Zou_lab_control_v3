//! The slow Ritz modes, exact.
//!
//! For a Ritz mode k with |theta_k| < SLOW (the kernel included) the error
//! its source F_k drives is, exactly,
//!   e_k(t) = a_k (e^{theta_k t} Y_k - e^{tA} Y_k),  Y_k = (theta_k - A)^-1 F_k,
//! with F_k = (A W - theta_k W) b_k plus g_k, the omitted ring slaved at
//! z = theta_k and fed back (`ErrorLattice::ring_feedback`), and the K^(1)
//! rests the lattice operator leaves out on W b_k (one source g_k). The kernel
//! mode's Y (from the bordered system (A + u tau0^T) y = F, Y = -y) is the
//! steady error itself.
//!
//! The space's part of the source is solved exactly away from the kernel:
//! (theta - A)^-1 of (A - theta) W b is -W b (`mode`), and only the ring's
//! feedback g_k needs a solve. At the kernel the whole (small) source is
//! solved. Why the closed form: a pool of seeds and augmentations alone
//! failed on two of sixteen solves of the whole source. Why W is not in the
//! pool: for g_k alone it adds little (as directions it saved 4 of 44 steps
//! at the first estimate of three and four lasers at 10 ms, for 250-940
//! GEMV of products with Q); the basis columns enter only through W^T U.
//!
//! One shared pool answers every shift: the trace-row vector u, the seed
//! directions (the inner directions of the pipeline's own steady solve,
//! whose images its Arnoldi relation gives for free) and every augmentation
//! direction z with A z, their span kept orthonormal (Q, row panels of the
//! fields' canonical half, extended block by block: no Gram matrix of the
//! pool is ever rebuilt) with every direction's and image's coordinates.
//! For any shift the images of the pool are (theta U - A U) in those
//! coordinates, so GCRO for (theta_k, g_k) costs the projection of g_k, a
//! small least-squares problem, and, while the residual is above
//! RESIDUAL |g_k|, preconditioned augmentation steps that join the pool for
//! every later mode and checkpoint. The preconditioner is the solver's own
//! first-order-kernel one: the shared Floquet frames at the steady shift.
//! A solve stops when the mode's contribution to the outputs has settled
//! (`SETTLE`), or at the residual RESIDUAL.
//!
//! Deflation (away from the kernel): a slow Ritz value lies close to an
//! eigenvalue of A, so (theta - A) is nearly singular along the slow Ritz
//! vectors x_j = W s_j, and Y carried large components there whose
//! response cancelled to the error. The solve is bordered by them and by
//! the reduced model's left functionals l_j: (theta - A) y + sum_j lambda_j x_j = g
//! with l_j(y) = 0, and the lambda_j respond in closed form (`mode`).
//! Measured on three, four and five lasers at 10 ms: the first estimate's
//! GCRO steps at the same residual 42-50 -> 21-26 (1e-4), and the
//! residual-only stop reliable at 1e-5 (undeflated: 10% at 1e-4, a factor
//! two at 1e-3).
//!
//! e^{tA} Y_k: Y's part in the space, W r0 (r0 = W^T Y), is propagated by
//! the reduced model, W S e^{theta t} S^-1 r0 (a kernel mode's without its
//! kernel coordinates, `populations`);
//! the rest decays at the fast rates and is propagated by the dressed blocks
//! of L0' on the near modes only (its readout elsewhere is micromotion).
use super::super::*;
use super::super::estimate_api::SteadySolves;
use super::lattice::{ErrorLattice,split};
use super::propagate::Readout;
use crate::parallel::Execution;

/// Ritz modes with |theta| below this (1/us) are answered exactly: a
/// deliberate margin over the research cases' slowest relaxation rates
/// (2e-3 to 8e-3 /us on three lasers, 3.4e-2 on four), whose far sources'
/// transport the SW part cannot follow (it is no small parameter there).
pub const SLOW:f64=0.05;
/// GCRO stops when the residual is below RESIDUAL |F|.
/// With the deflation, measured against solves to 1e-7 on two to five lasers
/// at 10 ms (first estimates): within 1.2% (3e-5: 2.7%; 1e-4: 5.4% there, but
/// 42% at a later checkpoint of four lasers; 1e-3: 8%). The residual bounds
/// the readout loosely: the tolerance keeps a margin.
const RESIDUAL:f64=1e-5;
/// A solve also stops once its contribution to the outputs has settled:
/// two consecutive changes below SETTLE of its size (away from the kernel
/// at least of the steady error), with the residual below SETTLED_RESIDUAL;
/// the response is read at SETTLE_TIMES times halving from the horizon.
/// Measured against solves to 1e-7 on two to five lasers at 10 ms: within
/// 2.1%, the first estimate's steps 27-32 -> 17-29 (one settled change, or
/// 3e-2: 30% once, 3.6%).
const SETTLE:f64=1e-2;
const SETTLED_RESIDUAL:f64=1e-3;
const SETTLE_TIMES:usize=6;
const MAXIMUM_STEPS:usize=60;
/// Real rows per panel of Q and per task of the block products.
const PANEL_ROWS:usize=1<<12;
/// A block column whose part outside the block's earlier pivots is below
/// this fraction of its size adds no direction (a Gram matrix resolves
/// singular values down to about the square root of the roundoff).
const CHOLESKY_DEPENDENT:f64=1e-7;
/// A column of the small least-squares problem below this fraction of its
/// size outside the earlier ones is left out.
const SMALL_DEPENDENT:f64=1e-12;
/// A real or imaginary part of the residual field below this fraction of
/// the residual's norm adds no augmentation direction: a deliberate margin,
/// the part left is at most a thousandth of the residual.
const AUGMENT_PART:f64=1e-3;

/// Hermitian-symmetric fields as real vectors: the canonical modes only
/// (the others are their mirrors' adjoints), real and imaginary parts, a
/// pair's block scaled by sqrt 2 so that the dot product is the real inner
/// product over every mode. Half the length of the whole field.
#[derive(Clone)]
struct Packing{d:usize,n:usize,canonical:Vec<usize>,mirror:Vec<usize>,modes:usize}
impl Packing{
    fn new(lattice:&ErrorLattice<'_>)->Self{
        let canonical:Vec<usize>=(0..lattice.modes.len()).filter(|&q|lattice.canonical[q]).collect();
        Self{d:lattice.d,n:lattice.n,canonical,mirror:lattice.mirror.clone(),modes:lattice.modes.len()}
    }
    fn length(&self)->usize{2*self.d*self.canonical.len()}
    fn pack(&self,x:&[C])->Vec<f64>{
        let d=self.d;let mut out=Vec::with_capacity(self.length());
        for &q in &self.canonical{let w=if self.mirror[q]==q{1.}else{std::f64::consts::SQRT_2};for v in &x[q*d..(q+1)*d]{out.push(w*v.re);out.push(w*v.im);}}
        out
    }
    fn unpack(&self,v:&[f64])->Vec<C>{
        let(n,d)=(self.n,self.d);let mut x=vec![ZERO;self.modes*d];
        for(i,&q)in self.canonical.iter().enumerate(){let w=if self.mirror[q]==q{1.}else{std::f64::consts::FRAC_1_SQRT_2};
            for k in 0..d{x[q*d+k]=C{re:w*v[2*(i*d+k)],im:w*v[2*(i*d+k)+1]};}
            let m=self.mirror[q];if m!=q{for a in 0..n{for b in 0..n{x[m*d+a*n+b]=conj(x[q*d+b*n+a]);}}}}
        x
    }
}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
/// (e^{a t} - e^{b t}) / (a - b), and its limit t e^{b t} (1 + d t / 2 + (d t)^2 / 6)
/// for d = a - b with |d t| small.
fn secular(a:C,b:C,t:f64)->C{
    let d=C{re:a.re-b.re,im:a.im-b.im};let dt=magnitude(d)*t;
    if dt<1e-3{let e=scaled(cexp(b,t),t);let(x,y)=(scaled(d,t/2.),scaled(mul(d,d),t*t/6.));return mul(e,C{re:1.+x.re+y.re,im:x.im+y.im});}
    let n=C{re:cexp(a,t).re-cexp(b,t).re,im:cexp(a,t).im-cexp(b,t).im};let s=d.re*d.re+d.im*d.im;mul(n,C{re:d.re/s,im:-d.im/s})
}

/// Orthonormal real columns stored by row panels (each column-major), with
/// the real multiply-adds of every product with them.
struct Panels{length:usize,starts:Vec<usize>,q:Vec<Vec<f64>>,count:usize,macs:std::sync::atomic::AtomicU64}
impl Panels{
    fn new(length:usize)->Self{let starts:Vec<usize>=(0..length).step_by(PANEL_ROWS).collect();let panels=starts.len();Self{length,starts,q:vec![Vec::new();panels],count:0,macs:std::sync::atomic::AtomicU64::new(0)}}
    fn count_macs(&self,v:f64){self.macs.fetch_add(v as u64,std::sync::atomic::Ordering::Relaxed);}
    fn macs(&self)->f64{self.macs.load(std::sync::atomic::Ordering::Relaxed)as f64}
    fn rows(&self,p:usize)->(usize,usize){(self.starts[p],self.starts.get(p+1).copied().unwrap_or(self.length))}
    fn push(&mut self,column:&[f64]){for p in 0..self.starts.len(){let(s,e)=self.rows(p);self.q[p].extend_from_slice(&column[s..e]);}self.count+=1;}
    /// Q^T g (count x m) for g (length x m), column-major.
    fn dots(&self,lapack:&Lapack,execution:&Execution,g:&[f64],m:usize)->Result<Vec<f64>,String>{
        let count=self.count;if count==0||m==0{return Ok(vec![0.;count*m]);}
        self.count_macs((count*m*self.length)as f64);
        let parts=execution.map((0..self.starts.len()).collect(),|p|->Result<Vec<f64>,String>{
            let(s,e)=self.rows(p);let mut out=vec![0.;count*m];
            lapack.gemm(true,false,count,m,e-s,1.,&self.q[p],e-s,&g[s..],self.length,0.,&mut out,count)?;Ok(out)});
        let mut h=vec![0.;count*m];for part in parts{for(a,b)in h.iter_mut().zip(part?){*a+=b;}}
        Ok(h)
    }
    /// g += alpha Q h.
    fn combine(&self,lapack:&Lapack,execution:&Execution,g:&mut[f64],h:&[f64],m:usize,alpha:f64)->Result<(),String>{
        let count=self.count;if count==0||m==0{return Ok(());}
        self.count_macs((count*m*self.length)as f64);
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
        // Every column at the unit scale of its size: the block's Gram matrix
        // then resolves each column's own part (seed directions and images
        // span six or more orders of magnitude).
        for(c,&size)in sizes.iter().enumerate(){let scale=if size>0.{1./size}else{0.};for v in &mut g[c*length..(c+1)*length]{*v*=scale;}}
        let gram=block_cross(lapack,execution,&g,m,&g,m,length)?;
        let(order,r1)=cholesky(&gram,m,&vec![1.;m],true);let mut k=order.len();
        let mut fresh:Vec<Vec<f64>>=Vec::new();let mut t=Vec::new();
        if k>0{
            let mut selected=vec![0.;length*k];for(i,&c)in order.iter().enumerate(){selected[i*length..(i+1)*length].copy_from_slice(&g[c*length..(c+1)*length]);}
            let mut q1=block_times(lapack,execution,&selected,length,k,&upper_inverse(&r1,k),k)?;drop(selected);
            // A column the second pass loses was dependent within the first
            // pass's accuracy (the Gram squares its condition): it is left out.
            let r2=loop{
                let gram=block_cross(lapack,execution,&q1,k,&q1,k,length)?;
                let(kept,r2)=cholesky(&gram,k,&vec![1.;k],false);
                if kept.len()==k{break r2;}
                let mut columns=vec![0.;length*kept.len()];for(i,&c)in kept.iter().enumerate(){columns[i*length..(i+1)*length].copy_from_slice(&q1[c*length..(c+1)*length]);}
                q1=columns;k=kept.len();
            };
            if k>0{
                let q=block_times(lapack,execution,&q1,length,k,&upper_inverse(&r2,k),k)?;drop(q1);
                t=block_cross(lapack,execution,&q,k,&g,m,length)?;
                for(c,&size)in sizes.iter().enumerate(){for v in &mut t[c*k..(c+1)*k]{*v*=size;}}
                for i in 0..k{let column=q[i*length..(i+1)*length].to_vec();self.push(&column);fresh.push(column);}
            }
        }
        // The Gram matrices, the two triangular products and the coordinates.
        self.count_macs(((m*m+2*k*k+k*k+k*m)*length)as f64);
        let coordinates=(0..m).map(|c|{let mut v:Vec<f64>=(0..base).map(|i|h[i+base*c]).collect();v.extend((0..k).map(|i|t[i+k*c]));v}).collect();
        Ok((coordinates,fresh))
    }
}
/// Tasks of the block products: fixed, so that their partial sums (and the
/// results) do not depend on the thread count; two per worker of the
/// desktop app's eight.
const BLOCK_TASKS:usize=16;
/// Rows per task of the block products: BLOCK_TASKS tasks, at most a panel.
fn block_rows(length:usize)->usize{length.div_ceil(BLOCK_TASKS).clamp(1<<12,PANEL_ROWS)}
/// A^T B (ka x mb, column-major) of blocks A (length x ka) and B (length x mb), by row chunks.
fn block_cross(lapack:&Lapack,execution:&Execution,a:&[f64],ka:usize,b:&[f64],mb:usize,length:usize)->Result<Vec<f64>,String>{
    if ka==0||mb==0{return Ok(vec![0.;ka*mb]);}
    let chunk=block_rows(length);
    let parts=execution.map((0..length).step_by(chunk).collect(),|s:usize|->Result<Vec<f64>,String>{
        let rows=(s+chunk).min(length)-s;let mut out=vec![0.;ka*mb];
        lapack.gemm(true,false,ka,mb,rows,1.,&a[s..],length,&b[s..],length,0.,&mut out,ka)?;Ok(out)});
    let mut out=vec![0.;ka*mb];for part in parts{for(x,y)in out.iter_mut().zip(part?){*x+=y;}}
    Ok(out)
}
/// G T (length x m) for G (length x k) and T (k x m), by row chunks.
fn block_times(lapack:&Lapack,execution:&Execution,g:&[f64],length:usize,k:usize,t:&[f64],m:usize)->Result<Vec<f64>,String>{
    if k==0||m==0{return Ok(vec![0.;length*m]);}
    let chunk=block_rows(length);let starts:Vec<usize>=(0..length).step_by(chunk).collect();
    let parts=execution.map(starts.clone(),|s|->Result<Vec<f64>,String>{
        let rows=(s+chunk).min(length)-s;let mut out=vec![0.;rows*m];
        lapack.gemm(false,false,rows,m,k,1.,&g[s..],length,t,k,0.,&mut out,rows)?;Ok(out)});
    let mut out=vec![0.;length*m];
    for(&s,part)in starts.iter().zip(parts){let part=part?;let rows=(s+chunk).min(length)-s;
        for c in 0..m{out[s+length*c..s+rows+length*c].copy_from_slice(&part[rows*c..rows*(c+1)]);}}
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

#[derive(Clone,Debug,Default)]
pub struct PoolWork{
    /// Operator and preconditioner applications on P u Q, GCRO steps and
    /// solves (one per slow mode and checkpoint); ring block solves inside
    /// the ring feedback are counted by the lattice.
    pub applications:usize,pub preconditions:usize,pub steps:usize,pub solves:usize,
    /// Seed directions taken from the pipeline.
    pub seeds:usize,
    /// Near-mode coordinate changes of the slow modes' remainders.
    pub frame_blocks:usize,
    /// Real multiply-adds of the products with Q and with the basis
    /// (extensions, projections, W^T Q).
    pub qr_macs:f64,
    /// Time (ms): extending Q and the basis products, GCRO steps, the rest
    /// of the solves.
    pub extend_ms:f64,pub step_ms:f64,pub solve_ms:f64,
}

/// One slow mode's error: Y = W c + U x (the basis part c, the pool part x
/// over the pool's directions) and the deflated amplitudes lambda_j of the
/// slow Ritz vectors x_j (Ritz index, value, lambda), whose response is in
/// closed form.
pub struct SlowSolution{pub theta:C,pub coefficient:C,pub kernel:bool,basis:Vec<C>,x:Vec<C>,deflated:Vec<(usize,C,C)>}

/// The reduced model's slow invariant subspace at one checkpoint: per slow
/// Ritz mode j its Ritz index and value, b_j = s_j, its left functional
/// on W^T y (row j of S^-1: l_j(x_i) = delta_ij), and the
/// coordinates on Q of x_j = W b_j and of A x_j.
struct Deflation{index:Vec<usize>,theta:Vec<C>,b:Vec<Vec<C>>,left:Vec<Vec<C>>,cx:Vec<Vec<C>>,cax:Vec<Vec<C>>,centre:Vec<Vec<C>>}

/// A solve's coefficients: the pool part, the basis part, and the deflated
/// modes (positions in the deflation) with their amplitudes.
struct Solved{x:Vec<C>,basis:Vec<C>,deflated:Vec<(usize,C)>}

pub struct SlowPool{
    dimension:usize,lapack:Arc<Lapack>,execution:Execution,
    n:usize,d:usize,watch:Vec<usize>,mirror:Vec<usize>,near:Vec<usize>,packing:Packing,
    /// The centre's slot among the near modes.
    centre_slot:usize,
    q:Panels,
    /// Per direction: its coordinates and its image's (on Q at insertion),
    /// its centre trace and its near-mode blocks (near.len() each); the
    /// trace-row vector's coordinates.
    cu:Vec<Vec<f64>>,cau:Vec<Vec<f64>>,trace:Vec<f64>,near_parts:Vec<Vec<C>>,u:Vec<f64>,
    /// The basis columns' canonical halves (column-major, length x k) and
    /// per column of Q its products with them, W^T q (k each).
    packed_basis:Vec<f64>,wq:Vec<Vec<f64>>,columns:usize,
    /// The checkpoint's slow invariant subspace (`deflate`).
    deflation:Option<Deflation>,
    /// The longest output time (us), and the latest kernel mode's steady
    /// error (population units), the floor of the other modes' settling
    /// scale (`solve`).
    horizon:f64,steady_scale:f64,
    pub work:PoolWork,
}

impl SlowPool{
    /// `horizon`: the longest output time (us).
    pub fn new(prepared:&LatticePrepared,lattice:&ErrorLattice<'_>,horizon:f64)->Result<Self,String>{
        let(n,d)=(lattice.n,lattice.d);let c0=lattice.centre*d;let watch:Vec<usize>=(0..n).map(|i|c0+i*n+i).collect();
        let packing=Packing::new(lattice);let length=packing.length();
        let mut pool=Self{dimension:lattice.dimension(),lapack:Arc::clone(&prepared.lapack),execution:Execution::new()?,n,d,watch,mirror:lattice.mirror.clone(),near:lattice.near.clone(),packing,centre_slot:lattice.near.iter().position(|&q|q==lattice.centre).unwrap_or(0),
            q:Panels::new(length),cu:Vec::new(),cau:Vec::new(),trace:Vec::new(),near_parts:Vec::new(),u:Vec::new(),
            packed_basis:Vec::new(),wq:Vec::new(),columns:0,deflation:None,horizon,steady_scale:0.,work:PoolWork::default()};
        let packed=pool.packing.pack(&pool.trace_row());let(mut coordinates,_)=pool.extend(packed,1)?;pool.u=coordinates.remove(0);
        Ok(pool)
    }
    /// u = I/n on the centre.
    fn trace_row(&self)->Vec<C>{let mut u=vec![ZERO;self.dimension];for &k in &self.watch{u[k]=C{re:1./self.n as f64,im:0.};}u}
    fn centre_trace(&self,x:&[C])->f64{self.watch.iter().map(|&k|x[k].re).sum()}
    fn apply(&mut self,steady:&mut dyn SteadySolves,x:&[C])->Result<Vec<C>,String>{let mut y=vec![ZERO;x.len()];steady.apply(x,&mut y)?;self.work.applications+=1;Ok(y)}
    /// Extends Q by the span of m packed columns and every new column's
    /// products with the basis. Returns the columns' coordinates and Q's new
    /// columns.
    fn extend(&mut self,block:Vec<f64>,m:usize)->Result<(Vec<Vec<f64>>,Vec<Vec<f64>>),String>{
        let(coordinates,fresh)=self.q.extend(&self.lapack,&self.execution,block,m)?;
        let(k,length,count)=(self.columns,self.q.length,fresh.len());
        if count>0{
            let products=if k>0{let flat=fresh.concat();block_cross(&self.lapack,&self.execution,&self.packed_basis,k,&flat,count,length)?}else{Vec::new()};
            self.q.count_macs((k*count*length)as f64);
            for c in 0..count{self.wq.push(products.get(c*k..(c+1)*k).map_or(Vec::new(),|v|v.to_vec()));}
        }
        Ok((coordinates,fresh))
    }
    /// Adds directions with their images: Q extended by both, the
    /// coordinates kept. Returns Q's new columns.
    fn add(&mut self,directions:Vec<(Vec<C>,Vec<C>)>)->Result<Vec<Vec<f64>>,String>{
        let clock=Instant::now();let m=directions.len();let length=self.q.length;
        let mut block=Vec::with_capacity(2*m*length);for(x,ax)in &directions{block.extend(self.packing.pack(x));block.extend(self.packing.pack(ax));}
        let(coordinates,fresh)=self.extend(block,2*m)?;
        let d=self.d;
        for(i,(x,_))in directions.into_iter().enumerate(){
            self.trace.push(self.centre_trace(&x));self.cu.push(coordinates[2*i].clone());self.cau.push(coordinates[2*i+1].clone());
            self.near_parts.push(self.near.iter().flat_map(|&q|x[q*d..(q+1)*d].iter().copied()).collect());
        }
        self.work.extend_ms+=clock.elapsed().as_secs_f64()*1e3;self.work.qr_macs=self.q.macs();
        Ok(fresh)
    }
    /// Seed directions z with their images A z (A on P u Q, unbordered):
    /// Hermitian-symmetric fields, such as the inner directions of the
    /// pipeline's steady solve.
    pub fn seed(&mut self,directions:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        if directions.len()!=images.len(){return Err("Slow pool: seed directions and images differ in number".into());}
        self.work.seeds+=directions.len();
        if !directions.is_empty(){self.add(directions.iter().cloned().zip(images.iter().cloned()).collect())?;}
        Ok(())
    }
    /// The basis columns not seen yet: their products with Q (they are not
    /// pool directions; the slow modes' sources in their span are solved
    /// exactly, `mode`).
    pub fn add_columns(&mut self,basis:&[Vec<C>])->Result<(),String>{
        if basis.len()<=self.columns{return Ok(());}
        let clock=Instant::now();let(first,length)=(self.columns,self.q.length);let m=basis.len()-first;
        let mut block=Vec::with_capacity(m*length);for w in &basis[first..]{block.extend(self.packing.pack(w));}
        let products=self.q.dots(&self.lapack,&self.execution,&block,m)?;let count=self.q.count;
        for(i,w)in self.wq.iter_mut().enumerate(){for c in 0..m{w.push(products[i+count*c]);}}
        self.packed_basis.extend(block);self.columns=basis.len();
        self.work.extend_ms+=clock.elapsed().as_secs_f64()*1e3;self.work.qr_macs=self.q.macs();
        Ok(())
    }
    /// The checkpoint's slow Ritz modes to deflate: Ritz indices, values,
    /// b_j = s_j, the left functionals (k each), x_j = W b_j and A x_j
    /// (the pipeline's images). x_j and A x_j join Q (their two
    /// Hermitian-symmetric parts), not the directions.
    pub fn deflate(&mut self,index:&[usize],theta:&[C],b:&[Vec<C>],left:&[Vec<C>],fields:&[Vec<C>],images:&[Vec<C>])->Result<(),String>{
        let clock=Instant::now();let length=self.q.length;let m=index.len();
        let mut block=Vec::with_capacity(4*m*length);
        for(x,ax)in fields.iter().zip(images){
            let((xa,xb),(aa,ab))=(split(x,&self.mirror,self.n),split(ax,&self.mirror,self.n));
            for part in[&xa,&xb,&aa,&ab]{block.extend(self.packing.pack(part));}
        }
        let(coordinates,_)=self.extend(block,4*m)?;
        let complex=|a:&[f64],b:&[f64]|->Vec<C>{(0..a.len().max(b.len())).map(|i|C{re:a.get(i).copied().unwrap_or(0.),im:b.get(i).copied().unwrap_or(0.)}).collect()};
        let(cx,cax)=((0..m).map(|i|complex(&coordinates[4*i],&coordinates[4*i+1])).collect(),(0..m).map(|i|complex(&coordinates[4*i+2],&coordinates[4*i+3])).collect());
        let centre=fields.iter().map(|x|self.watch.iter().map(|&k|x[k]).collect()).collect();
        self.deflation=Some(Deflation{index:index.to_vec(),theta:theta.to_vec(),b:b.to_vec(),left:left.to_vec(),cx,cax,centre});
        self.work.extend_ms+=clock.elapsed().as_secs_f64()*1e3;self.work.qr_macs=self.q.macs();
        Ok(())
    }
    /// A slow mode's contribution measure (centre populations, n values):
    /// |a| (Y_perp + sum_j lambda_j Phi_j x_j) for the pool coefficients c,
    /// the deflated amplitudes lambda and the directions' left functionals
    /// l_j(u_k); without the deflation the centre populations of U c.
    fn contribution(&self,c:&[C],lambda:&[C],ell:&[Vec<C>],deflation:Option<&Deflation>,phi:&[f64],weight:f64)->Vec<C>{
        let(n,d,centre_slot)=(self.n,self.d,self.centre_slot);
        let mut out=vec![ZERO;n];
        for(k,&v)in c.iter().enumerate(){if !nonzero(v){continue;}for(i,o)in out.iter_mut().enumerate(){mul_add(o,v,self.near_parts[k][centre_slot*d+i*n+i]);}}
        if let Some(d)=deflation{for(i,centre)in d.centre.iter().enumerate(){
            // - l_i(U c) x_i + lambda_i Phi_i x_i
            let l=c.iter().enumerate().fold(ZERO,|s,(k,&v)|{let w=mul(ell[k][i],v);C{re:s.re+w.re,im:s.im+w.im}});
            let a=C{re:lambda[i].re*phi[i]-l.re,im:lambda[i].im*phi[i]-l.im};
            for(o,&v)in out.iter_mut().zip(centre){mul_add(o,a,v);}}}
        for o in &mut out{o.re*=weight;o.im*=weight;}
        out
    }
    /// W^T u_j of direction j (k values), from its coordinates and W^T Q.
    fn basis_products(&self,j:usize)->Vec<f64>{
        let k=self.columns;let mut out=vec![0.;k];
        for(i,&c)in self.cu[j].iter().enumerate(){if c==0.{continue;}for(t,&w)in out.iter_mut().zip(&self.wq[i][..k]){*t+=c*w;}}
        out
    }
    /// Pool image coordinates of direction j at the shift: theta u_j - A u_j,
    /// or at the kernel the bordered A u_j + u tau0(u_j).
    fn image(&self,j:usize,theta:C,kernel:bool,count:usize)->Vec<C>{
        let mut v=vec![ZERO;count];
        if kernel{for(i,&a)in self.cau[j].iter().enumerate(){v[i].re+=a;}for(i,&a)in self.u.iter().enumerate(){v[i].re+=a*self.trace[j];}}
        else{for(i,&a)in self.cu[j].iter().enumerate(){v[i]=scaled(theta,a);}for(i,&a)in self.cau[j].iter().enumerate(){v[i].re-=a;}}
        v
    }
    /// The centre populations of U x (complex: x carries both real parts).
    fn centre_populations(&self,x:&[C])->Vec<C>{
        let(n,d,centre_slot)=(self.n,self.d,self.centre_slot);
        let mut out=vec![ZERO;n];
        for(j,&c)in x.iter().enumerate(){if !nonzero(c){continue;}for(i,o)in out.iter_mut().enumerate(){mul_add(o,c,self.near_parts[j][centre_slot*d+i*n+i]);}}
        out
    }
    /// Centre populations (real parts) of scale Y.
    fn readout(&self,basis:&[Vec<C>],s:&SlowSolution,scale:C)->Vec<f64>{
        let(n,d,centre_slot)=(self.n,self.d,self.centre_slot);
        let mut out=vec![0.;n];
        for(j,&c)in s.x.iter().enumerate(){if !nonzero(c){continue;}let c=mul(scale,c);
            for(i,o)in out.iter_mut().enumerate(){*o+=mul(c,self.near_parts[j][centre_slot*d+i*n+i]).re;}}
        for(j,&c)in s.basis.iter().enumerate(){if !nonzero(c){continue;}let c=mul(scale,c);
            for(i,o)in out.iter_mut().enumerate(){*o+=mul(c,basis[j][self.watch[i]]).re;}}
        out
    }
    /// The coefficients of (theta - A)^-1 f (at the kernel of the trace-free
    /// -y, (A + u tau0^T) y = f) by GCRO on the pool. With the deflation (away
    /// from the kernel) the solve is for (theta - A) y + sum_j lambda_j x_j = f
    /// with l_j(y) = 0: y = P U c, P = 1 - sum_j x_j l_j (its x_j parts go to
    /// the basis part), and the lambda_j. The kernel solve is not deflated:
    /// its steady limit, -lambda_j x_j / theta_j, would weigh the Ritz
    /// vectors' residuals by 1 / theta_j (measured: the steady estimate moved
    /// by up to a factor 2.7).
    fn solve(&mut self,f:&[C],theta:C,coefficient:C,kernel:bool,deflate:bool,steady:&mut dyn SteadySolves)->Result<Solved,String>{
        let clock=Instant::now();self.work.solves+=1;
        let length=self.q.length;
        let deflation=if deflate&&!kernel{self.deflation.take()}else{None};
        let slow=deflation.as_ref().map_or(0,|d|d.theta.len());
        // f = e^{i phi} (g1 + i g2) with g1, g2 Hermitian-symmetric and
        // orthogonal, g1 the larger: a real problem when theta is real.
        let(f1,f2)=split(f,&self.mirror,self.n);let(a,b)=(self.packing.pack(&f1),self.packing.pack(&f2));
        let(aa,bb,ab)=(dot(&a,&a),dot(&b,&b),dot(&a,&b));let phi=0.5*(2.*ab).atan2(aa-bb);let(cs,sn)=(phi.cos(),phi.sin());
        let g1:Vec<f64>=a.iter().zip(&b).map(|(x,y)|cs*x+sn*y).collect();let g2:Vec<f64>=a.iter().zip(&b).map(|(x,y)|-sn*x+cs*y).collect();
        let fsize=(dot(&g1,&g1)+dot(&g2,&g2)).sqrt();
        if fsize==0.{if deflation.is_some(){self.deflation=deflation;}return Ok(Solved{x:vec![ZERO;self.cu.len()],basis:Vec::new(),deflated:Vec::new()});}
        // Coordinates on Q and the parts outside it.
        let mut parts=g1.clone();parts.extend(&g2);
        let h=self.q.dots(&self.lapack,&self.execution,&parts,2)?;self.q.combine(&self.lapack,&self.execution,&mut parts,&h,2,-1.)?;
        self.q.count_macs((4*length)as f64);
        let count=self.q.count;let mut phi_c:Vec<C>=(0..count).map(|i|C{re:h[i],im:h[i+count]}).collect();
        // The deflated modes' left functionals of every direction, l_j(u_k).
        let mut ell:Vec<Vec<C>>=Vec::new();
        let functionals=|this:&Self,d:&Deflation,k:usize|->Vec<C>{let w=this.basis_products(k);
            d.left.iter().map(|l|l.iter().zip(&w).fold(ZERO,|s,(a,&v)|C{re:s.re+a.re*v,im:s.im+a.im*v})).collect()};
        if let Some(d)=&deflation{for k in 0..self.cu.len(){ell.push(functionals(self,d,k));}}
        // The settling measure: the mode's contribution to the outputs read at
        // the centre, a (Y_perp + sum_j lambda_j Phi_j x_j), Phi_j the largest
        // closed-form response of x_j over the outputs' times (`secular`).
        let times:Vec<f64>=(0..SETTLE_TIMES).map(|i|self.horizon/f64::from(1<<i)).collect();
        let phi:Vec<f64>=deflation.as_ref().map_or(Vec::new(),|d|d.theta.iter().map(|&z|times.iter().map(|&t|magnitude(secular(theta,z,t))).fold(0.,f64::max)).collect());
        let weight=magnitude(coefficient);
        let(mut previous,mut settled):(Option<Vec<C>>,usize)=(None,0);
        let floor=if kernel{0.}else{self.steady_scale};
        let mut steps=0;
        let solution=loop{
            let(count,p)=(self.q.count,self.cu.len());
            let mut m=Vec::with_capacity(count*(p+slow));
            for j in 0..p{
                let mut v=self.image(j,theta,kernel,count);
                if let Some(d)=&deflation{for(i,&l)in ell[j].iter().enumerate(){if !nonzero(l){continue;}
                    // - l_i(u) (theta x_i - A x_i)
                    for(t,(&x,&ax))in v.iter_mut().zip(d.cx[i].iter().zip(&d.cax[i])){let w=C{re:theta.re*x.re-theta.im*x.im-ax.re,im:theta.re*x.im+theta.im*x.re-ax.im};let y=mul(l,w);t.re-=y.re;t.im-=y.im;}}}
                m.extend(v);
            }
            if let Some(d)=&deflation{for cx in &d.cx{let mut v=cx.clone();v.resize(count,ZERO);m.extend(v);}}
            let(x,rho)=least_squares(&m,count,p+slow,&phi_c);
            let outside=dot(&parts,&parts);let rn=(rho.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>()+outside).sqrt();
            if rn<=RESIDUAL*fsize||steps>=MAXIMUM_STEPS{break x;}
            // Settled: two consecutive changes of the contribution below SETTLE
            // of its size (or of the steady error, the larger), the residual
            // below SETTLED_RESIDUAL.
            let contribution=self.contribution(&x[..p],&x[p..],&ell,deflation.as_ref(),&phi,weight);
            if let Some(last)=&previous{
                let largest=|v:&[C]|v.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
                let change=contribution.iter().zip(last).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
                if change<=SETTLE*largest(&contribution).max(floor){settled+=1;}else{settled=0;}
                if settled>=2&&rn<=SETTLED_RESIDUAL*fsize{break x;}
            }
            previous=Some(contribution);
            // The residual field r = Q rho + (the parts outside Q), split in two real fields.
            let mut r=parts.clone();
            let rho_parts:Vec<f64>=rho.iter().map(|z|z.re).chain(rho.iter().map(|z|z.im)).collect();
            self.q.combine(&self.lapack,&self.execution,&mut r,&rho_parts,2,1.)?;
            let step=Instant::now();
            let mut new=Vec::new();
            for part in r.chunks_exact(length){
                let size=dot(part,part).sqrt();if !(size>AUGMENT_PART*rn){continue;}
                let field:Vec<C>=self.packing.unpack(part).iter().map(|v|C{re:v.re/size,im:v.im/size}).collect();
                let mut z=vec![ZERO;field.len()];steady.precondition(&field,&mut z)?;self.work.preconditions+=1;
                let az=self.apply(steady,&z)?;new.push((z,az));
            }
            if new.is_empty(){break x;}
            let first=self.cu.len();let fresh=self.add(new)?;
            if let Some(d)=&deflation{for k in first..self.cu.len(){ell.push(functionals(self,d,k));}}
            // f's coordinates on the new columns, and its parts outside Q.
            for q in &fresh{let(c1,c2)=(dot(q,&parts[..length]),dot(q,&parts[length..]));phi_c.push(C{re:c1,im:c2});
                for(t,v)in parts[..length].iter_mut().zip(q){*t-=c1*v;}for(t,v)in parts[length..].iter_mut().zip(q){*t-=c2*v;}}
            steps+=1;self.work.steps+=1;self.work.step_ms+=step.elapsed().as_secs_f64()*1e3;
        };
        // Back from the rotation: e^{i phi} x, negated at the kernel (Y = -y);
        // the deflated directions' parts of P U c to the basis part.
        let rotation=if kernel{C{re:-cs,im:-sn}}else{C{re:cs,im:sn}};
        let p=solution.len()-slow;
        let x:Vec<C>=solution[..p].iter().map(|&v|mul(rotation,v)).collect();
        let deflated:Vec<(usize,C)>=solution[p..].iter().enumerate().map(|(i,&v)|(i,mul(rotation,v))).collect();
        let mut basis=Vec::new();
        if let Some(d)=&deflation{
            basis=vec![ZERO;self.columns];
            for(i,b)in d.b.iter().enumerate(){let l=(0..p).fold(ZERO,|s,k|{let v=mul(ell[k][i],x[k]);C{re:s.re+v.re,im:s.im+v.im}});
                for(t,&v)in basis.iter_mut().zip(b){let y=mul(l,v);t.re-=y.re;t.im-=y.im;}}
        }
        if deflation.is_some(){self.deflation=deflation;}
        self.work.solve_ms+=clock.elapsed().as_secs_f64()*1e3;self.work.qr_macs=self.q.macs();
        Ok(Solved{x,basis,deflated})
    }
    /// Y of a slow mode from its Ritz vector x = W b, A x = A W b (the
    /// pipeline's images) and the ring's feedback g of x at theta. Away from
    /// the kernel the source (A - theta) x + g gives, exactly,
    /// Y = -W b + (theta - A)^-1 g: the first part lies in the space and needs
    /// no solve. The slow Ritz values lie close to eigenvalues of A, so
    /// (theta - A) is nearly singular along their Ritz vectors: those
    /// (`deflate`) take the amplitudes lambda_j of g and respond in closed form,
    /// a_m lambda_j (e^{theta_m t} - e^{theta_j t}) / (theta_m - theta_j) x_j
    /// (t e^{theta_m t} for j = m: the ring's shift of the Ritz value), and
    /// the solve is for the rest. At the kernel the same split would read
    /// Y = -x + tau0(x) x_ss - (A + u tau0^T)^-1 g, whose first two terms (the
    /// reduced and the true steady state) cancel to the steady error itself,
    /// so there the small source A x + g is solved whole.
    pub fn mode(&mut self,b:&[C],ax:&[C],g:&[C],theta:C,coefficient:C,kernel:bool,steady:&mut dyn SteadySolves)->Result<SlowSolution,String>{
        if kernel{
            let f:Vec<C>=ax.iter().zip(g).map(|(a,v)|C{re:a.re+v.re,im:a.im+v.im}).collect();
            let s=self.solve(&f,theta,coefficient,true,false,steady)?;
            let solution=SlowSolution{theta,coefficient,kernel,basis:Vec::new(),x:s.x,deflated:Vec::new()};
            self.steady_scale=self.centre_populations(&solution.x).iter().map(|z|magnitude(*z)).fold(0.,f64::max)*magnitude(coefficient);
            return Ok(solution);
        }
        let s=self.solve(g,theta,coefficient,false,true,steady)?;
        let mut basis=s.basis;basis.resize(b.len(),ZERO);for(t,v)in basis.iter_mut().zip(b){t.re-=v.re;t.im-=v.im;}
        let deflated=self.deflation.as_ref().map_or(Vec::new(),|d|s.deflated.iter().map(|&(i,l)|(d.index[i],d.theta[i],l)).collect());
        Ok(SlowSolution{theta,coefficient,kernel,basis,x:s.x,deflated})
    }
    /// The Gram matrix W^T U of the basis against every pool direction
    /// (column-major k x p), from the directions' coordinates and W^T Q.
    fn basis_gram(&self,k:usize)->Vec<f64>{
        let p=self.cu.len();let mut g=vec![0.;k*p];
        for(j,cu)in self.cu.iter().enumerate(){for(i,&c)in cu.iter().enumerate(){if c==0.{continue;}
            for(t,&w)in g[k*j..k*(j+1)].iter_mut().zip(&self.wq[i][..k]){*t+=c*w;}}}
        g
    }
    /// W^T Y (from the Gram matrix) and Y on the near modes.
    fn parts(&self,basis:&[Vec<C>],gram:&[f64],s:&SlowSolution)->(Vec<C>,Vec<C>){
        // Directions added after this solution carry no coefficient.
        let d=self.d;let k=basis.len();let p=s.x.len();
        let coordinates:Vec<C>=(0..k).map(|i|{let mut v=s.basis.get(i).copied().unwrap_or(ZERO);for j in 0..p{add(&mut v,scaled(s.x[j],gram[i+k*j]));}v}).collect();
        let mut near=vec![ZERO;self.near.len()*d];
        for(j,&c)in s.x.iter().enumerate(){if !nonzero(c){continue;}for(t,&v)in near.iter_mut().zip(&self.near_parts[j]){mul_add(t,c,v);}}
        for(i,&c)in s.basis.iter().enumerate(){if !nonzero(c){continue;}
            for(slot,&q)in self.near.iter().enumerate(){for(t,&v)in near[slot*d..(slot+1)*d].iter_mut().zip(&basis[i][q*d..(q+1)*d]){mul_add(t,c,v);}}}
        (coordinates,near)
    }
    /// The centre populations of a_k Y_k (a kernel mode's: its steady error).
    pub fn steady_populations(&self,basis:&[Vec<C>],s:&SlowSolution)->Vec<f64>{self.readout(basis,s,s.coefficient)}
    /// Population errors (one row per time) of the slow modes,
    /// sum_k a_k (e^{theta_k t} Y_k - e^{tA} Y_k), for the reduced model's
    /// eigenvectors S (r x r, column-major, r = basis.len()), the values theta
    /// (r) the outputs use and its kernel modes `kernel`. A kernel mode's Y
    /// (the steady error) is trace free: the true e^{tA} Y decays, so its
    /// projection on the space is propagated without its kernel coordinates
    /// (which the reduced model would keep forever; measured: early
    /// checkpoints of three and four lasers 0.88 and 2.3 of the actual error
    /// without, 1.00-1.01 with).
    #[allow(clippy::too_many_arguments)]
    pub fn populations(&mut self,lattice:&ErrorLattice<'_>,basis:&[Vec<C>],s:&[C],theta:&[C],kernel:&[usize],solutions:&[SlowSolution],times:&[f64],readout:Readout)->Result<Vec<Vec<f64>>,String>{
        let(n,d,k,r)=(self.n,self.d,basis.len(),theta.len());let t_count=times.len();
        if solutions.is_empty()||t_count==0{return Ok(vec![vec![0.;n];t_count]);}
        let lapack=Arc::clone(&self.lapack);
        // S factored once; the Ritz vectors B = S over the basis.
        let(lu_s,piv_s)=lapack.complex_factor(s,r)?;let b=s;
        let gram=self.basis_gram(k);
        let modes:Vec<usize>=match readout{Readout::Laboratory=>(0..lattice.modes.len()).collect(),Readout::Centre=>vec![lattice.centre]};
        let phase=|q:usize,t:f64|match readout{Readout::Laboratory=>cexp(C{re:0.,im:lattice.nu[q]},t),Readout::Centre=>ONE};
        // Per slow mode: r0 = W^T Y (Y's part in the space) and the remainder on the near modes.
        let mut ritz_weights=vec![ZERO;r];let mut in_space:Vec<Vec<C>>=Vec::new();let mut remainders:Vec<(usize,C,C,Vec<C>)>=Vec::new();
        for sol in solutions{
            let(r0,near)=self.parts(basis,&gram,sol);
            let mut cs=lapack.complex_solve_matrix_factored(&lu_s,&piv_s,&r0,r)?;
            if sol.kernel{for &j in kernel{cs[j]=ZERO;}}
            for(w,c)in ritz_weights.iter_mut().zip(&cs){mul_add(w,sol.coefficient,*c);}
            let xr=r0;
            for(slot,&q)in self.near.iter().enumerate(){
                if readout==Readout::Centre&&q!=lattice.centre{continue;}
                let mut block=near[slot*d..(slot+1)*d].to_vec();
                for(j,&c)in xr.iter().enumerate(){if !nonzero(c){continue;}for(t,&v)in block.iter_mut().zip(&basis[j][q*d..(q+1)*d]){let w=mul(c,v);t.re-=w.re;t.im-=w.im;}}
                remainders.push((q,sol.coefficient,sol.theta,block));
            }
            in_space.push(xr);
        }
        // In the space, per mode: sum_k a_k e^{theta_k t} P_q(W r0_k) - P_q(B (sum_k a_k cS_k) e^{theta t}),
        // and the deflated amplitudes' closed-form response, P_q(B h(t)) with
        // h_j(t) = sum_k a_k lambda_kj (e^{theta_k t} - e^{theta_j t}) / (theta_k - theta_j).
        // Per mode: P_q(B) and the constants P_q(W r0_k) (time-independent).
        let per_mode=self.execution.map(modes.clone(),|q|->Result<(usize,Vec<C>,Vec<(C,C,Vec<C>)>),String>{
            let gemm=lapack.complex_gemm();
            let mut rows=vec![ZERO;n*k];for j in 0..k{for i in 0..n{rows[i+n*j]=basis[j][q*d+i*n+i];}}
            let mut rb=vec![ZERO;n*r];gemm.multiply(n,r,k,ONE,&rows,n,b,k,ZERO,&mut rb,n)?;
            let mut constants=Vec::with_capacity(solutions.len());
            for(sol,xr)in solutions.iter().zip(&in_space){let mut v=vec![ZERO;n];gemm.multiply(n,1,k,ONE,&rows,n,xr,k,ZERO,&mut v,n)?;constants.push((sol.coefficient,sol.theta,v));}
            Ok((q,rb,constants))}).into_iter().collect::<Result<Vec<_>,_>>()?;
        // The remainders on the near modes: e^{i nu_q t} e^{theta_k t} Y_perp
        // - its dressed-block propagation R diag(a c) e^{lambda t}, which
        // carries no phase of its own: every remainder's decay is one
        // H = R diag(sum a c) for all of them.
        let lambda:Vec<C>=lattice.basis()?.lambda.iter().map(|z|C{re:z.re.min(0.),im:z.im}).collect();
        let mut rises:Vec<(usize,C,C,Vec<C>)>=Vec::with_capacity(remainders.len());let mut decay:Option<Vec<C>>=None;
        if !remainders.is_empty(){
            let readout_rows=lattice.basis()?.population_rows();
            let blocks:Vec<&[C]>=remainders.iter().map(|x|&x.3[..]).collect();
            let coefficients=lattice.basis()?.coefficients(&blocks)?;self.work.frame_blocks+=remainders.len();
            let mut weights=vec![ZERO;d];
            for(i,(q,a,th,y))in remainders.iter().enumerate(){
                for(j,w)in weights.iter_mut().enumerate(){mul_add(w,*a,coefficients[j+d*i]);}
                rises.push((*q,*a,*th,(0..n).map(|row|y[row*n+row]).collect()));
            }
            let mut h=readout_rows.to_vec();for j in 0..d{for row in 0..n{h[row+n*j]=mul(h[row+n*j],weights[j]);}}
            decay=Some(h);
        }
        // The time GEMMs by chunks of the grid over the workers: every
        // output entry is accumulated by one task in the serial order.
        let per_time=per_mode.len()*n*r+usize::from(decay.is_some())*n*d+rises.len()*n;
        let width=super::time_chunk(t_count,per_time);
        let chunks:Vec<(usize,usize)>=(0..t_count).step_by(width).map(|a|(a,(a+width).min(t_count))).collect();
        let(ritz_weights,per_mode,rises,decay,lambda)=(&ritz_weights,&per_mode,&rises,&decay,&lambda);
        let parts=self.execution.map(chunks,|(a0,b0)|->Result<Vec<C>,String>{
            let(gemm,ts,tc)=(lapack.complex_gemm(),&times[a0..b0],b0-a0);
            let mut total=vec![ZERO;n*tc];
            let mut ritz_exp:Vec<C>=ts.iter().flat_map(|&t|theta.iter().zip(ritz_weights).map(move|(&z,&w)|mul(w,cexp(z,t)))).collect();
            for sol in solutions{for &(j,zj,l)in &sol.deflated{let a=mul(sol.coefficient,l);if !nonzero(a){continue;}
                for(ti,&t)in ts.iter().enumerate(){let v=mul(a,secular(sol.theta,zj,t));let e=&mut ritz_exp[j+r*ti];e.re-=v.re;e.im-=v.im;}}}
            for(q,rb,constants)in per_mode{
                let mut reduced=vec![ZERO;n*tc];gemm.multiply(n,tc,r,ONE,rb,n,&ritz_exp,r,ZERO,&mut reduced,n)?;
                for(ti,&t)in ts.iter().enumerate(){let ph=phase(*q,t);
                    for i in 0..n{let mut v=C{re:-reduced[i+n*ti].re,im:-reduced[i+n*ti].im};
                        for(a,th,c)in constants{mul_add(&mut v,mul(*a,cexp(*th,t)),c[i]);}
                        mul_add(&mut total[i+n*ti],ph,v);}}
            }
            for(q,a,th,y)in rises{
                for(ti,&t)in ts.iter().enumerate(){let ph=mul(phase(*q,t),mul(*a,cexp(*th,t)));
                    for row in 0..n{mul_add(&mut total[row+n*ti],ph,y[row]);}}
            }
            if let Some(h)=decay{
                let block:Vec<C>=ts.iter().flat_map(|&t|lambda.iter().map(move|&z|cexp(z,t))).collect();
                let mut dec=vec![ZERO;n*tc];gemm.multiply(n,tc,d,ONE,h,n,&block,d,ZERO,&mut dec,n)?;
                for(t,v)in total.iter_mut().zip(&dec){t.re-=v.re;t.im-=v.im;}
            }
            Ok(total)});
        let mut total=Vec::with_capacity(n*t_count);for part in parts{total.extend(part?);}
        Ok((0..t_count).map(|ti|(0..n).map(|i|total[i+n*ti].re).collect()).collect())
    }
}
