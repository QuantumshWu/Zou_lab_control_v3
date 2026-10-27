//! Krylov solves on the lattice: the steady trace-row system M x = u, with
//! M = A + u v^T (u = I/n on the centre, v its trace: the time-averaged
//! populations are the central diagonal of x, A x = 0 and v^T x = 1 when the
//! null vector is unique), and the shifted systems (A - sigma) w = b of the
//! rational Krylov propagation (`finite`), sigma real or complex.
//!
//! Real coordinates: the fields are Hermitian-symmetric (x_-q = x_q^dagger),
//! a real vector space that M and the preconditioner preserve, with the real
//! inner product Re <a, b>. Inner products are summed over the canonical
//! modes only (weight 2 for a pair, 1 for the centre) and the Hessenberg
//! matrix, rotations and coefficients are real. A complex shift acts on the
//! real form: a pair (w_1, w_2) of such fields for w_1 + i w_2, with
//! (A - a) w_1 + b w_2 and (A - a) w_2 - b w_1 for sigma = a + i b.
//!
//! Right-preconditioned restarted GMRES with a fixed preconditioner, so only
//! the populations of P^-1 v_j are kept per step and the iterate is formed
//! once per cycle. Orthogonalization is DCGS2 (classical Gram-Schmidt with
//! the reorthogonalization delayed by one step): one pass over the basis
//! takes the new image's projections together with the previous candidate's
//! reorthogonalization coefficients, a second pass finalizes the candidate
//! and orthogonalizes the image once, the Hessenberg column of the candidate
//! corrected by the Arnoldi relation; two basis passes per step against four
//! for CGS2, on the canonical modes only (the rest mirrored). The column of
//! the latest step is provisional (once orthogonalized) for the stopping
//! tests, and finalized with one more reorthogonalization when the cycle
//! ends. The steady solve stops when the populations and the
//! requested groups have stabilised (changes below the algebraic share of the
//! tolerance for STABLE_STEPS consecutive steps) and the true relative
//! residual, recomputed from the iterate after the cycle, is below a guard; a
//! shifted solve stops on the true relative residual alone. The population change
//! alone can plateau in slow directions, and the recursive GMRES residual can
//! fall eight orders below the true one (the operator spans 1e-6 to 5e7
//! rad/us). The longest cycle's Arnoldi relation M P^-1 V_m = V_m+1 Hbar_m
//! is kept for the error estimate (`estimate`): the preconditioner is fixed,
//! so every cycle's relation stays valid, and a final cycle of a few steps
//! (a restart for the true-residual guard) holds too little of the space.
use super::*;
use super::operator::LatticeOperator;
use super::preconditioner::{Preconditioner,Shift};
use super::ledger;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

const STABLE_STEPS:usize=3;
/// Blocks per reduction chunk: fixed, so sums do not depend on threads.
const VECTOR_CHUNK:usize=16;

#[derive(Clone,Debug)]
pub struct KrylovOptions{
    /// Stop only once the populations have stabilised as well (the steady
    /// solve), and their stability tolerance.
    pub populations:bool,pub algebraic:f64,
    /// Relative true-residual guard. Measured: a true residual of 3e-7 left
    /// 1.8e-5 in the populations (two lasers + tweezer, a 1e-6 /us Raman
    /// manifold); at 1e-8 every case is within 1e-7 of its reference.
    pub residual_guard:f64,
    pub restart:usize,pub maximum_steps:usize,
    /// Krylov basis memory budget (bytes); the restart length shrinks to fit.
    pub basis_bytes:usize,
}
impl Default for KrylovOptions{
    fn default()->Self{
        Self{populations:true,algebraic:ledger::POPULATION_TOLERANCE*ledger::ALGEBRAIC_SHARE,residual_guard:1e-8,restart:80,maximum_steps:800,basis_bytes:1<<30}
    }
}
#[derive(Clone,Debug,Default)]
pub struct KrylovTiming{pub apply_ms:f64,pub precondition_ms:f64,pub orthogonalize_ms:f64}

/// Deterministic parallel kernels on fields of `copies` Hermitian-symmetric
/// components (one block per mode and copy).
pub struct Vectors{execution:Execution,n:usize,d:usize,
    /// Real inner-product weight per block: 2 for a canonical pair member, 1
    /// for the centre, 0 for a mirrored mode.
    weights:Vec<f64>,
    canonical:Vec<bool>,mirror:Vec<usize>,lapack:Arc<Lapack>}
impl Vectors{
    pub fn new(op:&LatticeOperator<'_>,copies:usize)->Result<Self,String>{
        let modes=op.modes.len();
        let weights=(0..copies*modes).map(|k|{let q=k%modes;if !op.canonical[q]{0.}else if op.mirror[q]==q{1.}else{2.}}).collect();
        let canonical=(0..copies*modes).map(|k|op.canonical[k%modes]).collect();
        let mirror=(0..copies*modes).map(|k|k-k%modes+op.mirror[k%modes]).collect();
        Ok(Self{execution:Execution::new()?,n:op.n,d:op.n*op.n,weights,canonical,mirror,lapack:Arc::clone(&op.prepared.lapack)})
    }
    /// Mirrored blocks from their canonical partners, the centre Hermitian.
    pub fn mirror(&self,x:&mut[C]){mirror_field(x,self.n,&self.canonical,&self.mirror);}
    /// Elements of the canonical blocks (what a pass over a vector reads).
    pub fn canonical_length(&self)->usize{self.weights.iter().filter(|&&w|w>0.).count()*self.d}
    /// One pass: Re <v_i, u> and Re <v_i, w> over the basis, Re <u, u> and
    /// Re <u, w>.
    pub fn fused_dots(&self,basis:&[&[C]],u:&[C],w:&[C])->(Vec<f64>,Vec<f64>,f64,f64){
        let m=basis.len();
        let parts=self.execution.map(self.ranges(w.len()),|(a,b)|{
            let mut out=Vec::with_capacity(2*m+2);
            for v in basis{out.push(self.partial(v,u,a,b));out.push(self.partial(v,w,a,b));}
            out.push(self.partial(u,u,a,b));out.push(self.partial(u,w,a,b));out});
        let mut sums=vec![0.;2*m+2];for p in parts{for(o,v)in sums.iter_mut().zip(p){*o+=v;}}
        let(vu,vw)=((0..m).map(|i|sums[2*i]).collect(),(0..m).map(|i|sums[2*i+1]).collect());
        (vu,vw,sums[2*m],sums[2*m+1])
    }
    /// One pass on the canonical blocks: u <- (u - sum_i a_i v_i) / alpha,
    /// then w <- (w - sum_i c_i v_i - c_m u) / alpha (c has one more entry
    /// than the basis, for the new u); both mirrored. Returns |w|^2.
    pub fn fused_update(&self,basis:&[&[C]],u:&mut[C],a:&[f64],alpha:f64,w:&mut[C],c:&[f64])->Result<f64,String>{
        let(d,m)=(self.d,basis.len());let blocks=w.len()/d;let inverse=1./alpha;
        let tasks=self.execution.tasks(blocks,(2*m+2)*w.len(),MINIMUM_WORK_PER_TASK);let per=blocks.div_ceil(tasks);
        let mut items=Vec::with_capacity(tasks);let(mut ur,mut wr)=(&mut u[..],&mut w[..]);let mut first=0;
        while first<blocks{let count=per.min(blocks-first);let(uc,ut)=ur.split_at_mut(count*d);let(wc,wt)=wr.split_at_mut(count*d);
            items.push((first,uc,wc));ur=ut;wr=wt;first+=count;}
        let parts=self.execution.map(items,|(first,uc,wc)|{
            let mut norm=0.;
            for(k,(ub,wb))in uc.chunks_exact_mut(d).zip(wc.chunks_exact_mut(d)).enumerate(){
                let q=first+k;let weight=self.weights[q];if weight==0.{continue;}
                for(v,&ai)in basis.iter().zip(a){if ai==0.{continue;}for(t,x)in ub.iter_mut().zip(&v[q*d..(q+1)*d]){t.re-=ai*x.re;t.im-=ai*x.im;}}
                for t in ub.iter_mut(){*t=scaled(*t,inverse);}
                for(v,&ci)in basis.iter().zip(c){for(t,x)in wb.iter_mut().zip(&v[q*d..(q+1)*d]){t.re-=ci*x.re;t.im-=ci*x.im;}}
                let cm=c[m];let mut s=0.;
                for(t,x)in wb.iter_mut().zip(ub.iter()){t.re=(t.re-cm*x.re)*inverse;t.im=(t.im-cm*x.im)*inverse;s+=t.re*t.re+t.im*t.im;}
                norm+=weight*s;
            }
            norm});
        self.mirror(u);self.mirror(w);
        Ok(parts.into_iter().sum())
    }
    /// Columns of a canonical block as a column-major (2 d) x count real
    /// matrix (re and im interleaved: the real inner product is its dot).
    fn block(&self,vectors:&[&[C]],q:usize)->Vec<f64>{
        let d=self.d;let mut out=Vec::with_capacity(2*d*vectors.len());
        for v in vectors{for z in &v[q*d..(q+1)*d]{out.push(z.re);out.push(z.im);}}
        out
    }
    /// Re <rows_i, cols_j> for every pair (row-major |rows| x |cols|): one
    /// real GEMM per canonical block, summed in block order.
    pub fn gram(&self,rows:&[&[C]],cols:&[&[C]])->Result<Vec<f64>,String>{
        let(r,c)=(rows.len(),cols.len());let d2=2*self.d;
        if r==0||c==0{return Ok(vec![0.;r*c]);}
        let blocks:Vec<usize>=(0..self.weights.len()).filter(|&q|self.weights[q]>0.).collect();
        let parts=self.execution.map(blocks,|q|->Result<Vec<f64>,String>{
            let(a,b)=(self.block(rows,q),self.block(cols,q));let mut out=vec![0.;r*c];
            self.lapack.gemm(true,false,r,c,d2,self.weights[q],&a,d2,&b,d2,0.,&mut out,r)?;Ok(out)});
        let mut gram=vec![0.;r*c];
        for part in parts{let part=part?;for i in 0..r{for j in 0..c{gram[i*c+j]+=part[i+r*j];}}}
        Ok(gram)
    }
    /// sum_i coefficients[k][i] basis_i for every k (Hermitian-symmetric
    /// fields: canonical blocks by one real GEMM each, the rest mirrored).
    pub fn combinations(&self,basis:&[&[C]],coefficients:&[Vec<f64>])->Result<Vec<Vec<C>>,String>{
        let(m,k)=(basis.len(),coefficients.len());let(n,d)=(self.n,self.d);let d2=2*d;let modes=self.weights.len();
        let mut out=vec![vec![ZERO;modes*d];k];
        if m==0||k==0{return Ok(out);}
        let mut coefficient=vec![0.;m*k];for(j,c)in coefficients.iter().enumerate(){coefficient[j*m..(j+1)*m].copy_from_slice(&c[..m]);}
        let blocks:Vec<usize>=(0..modes).filter(|&q|self.canonical[q]).collect();
        let parts=self.execution.map(blocks.clone(),|q|->Result<Vec<f64>,String>{
            let a=self.block(basis,q);let mut product=vec![0.;d2*k];
            self.lapack.gemm(false,false,d2,k,m,1.,&a,d2,&coefficient,m,0.,&mut product,d2)?;Ok(product)});
        for(&q,part)in blocks.iter().zip(parts){let part=part?;
            for(j,field)in out.iter_mut().enumerate(){for e in 0..d{field[q*d+e]=C{re:part[j*d2+2*e],im:part[j*d2+2*e+1]};}}}
        for field in &mut out{mirror_field(field,n,&self.canonical,&self.mirror);}
        Ok(out)
    }
    fn ranges(&self,length:usize)->Vec<(usize,usize)>{
        let w=VECTOR_CHUNK*self.d;(0..length.div_ceil(w)).map(|c|(c*w,((c+1)*w).min(length))).collect()
    }
    fn partial(&self,a:&[C],b:&[C],start:usize,end:usize)->f64{
        let d=self.d;let mut s=0.;let mut k=start;
        while k<end{let q=k/d;let stop=((q+1)*d).min(end);let w=self.weights[q];
            if w>0.{let mut t=0.;for i in k..stop{t+=a[i].re*b[i].re+a[i].im*b[i].im;}s+=w*t;}k=stop;}
        s
    }
    /// Re <basis_i, w> for every basis vector.
    pub fn dots(&self,basis:&[&[C]],w:&[C])->Vec<f64>{
        let parts=self.execution.map(self.ranges(w.len()),|(a,b)|basis.iter().map(|v|self.partial(v,w,a,b)).collect::<Vec<f64>>());
        let mut out=vec![0.;basis.len()];for p in parts{for(o,v)in out.iter_mut().zip(p){*o+=v;}}out
    }
    pub fn dot(&self,a:&[C],b:&[C])->f64{self.dots(&[a],b)[0]}
    pub fn norm(&self,w:&[C])->f64{self.dot(w,w).max(0.).sqrt()}
    /// w -= sum_i h_i basis_i.
    pub fn subtract(&self,basis:&[&[C]],h:&[f64],w:&mut[C])->Result<(),String>{
        let d=self.d;let work=basis.len()*w.len();
        self.execution.for_mode_chunks(w,d,work,MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
            for(v,&c)in basis.iter().zip(h){for(t,&x)in chunk.iter_mut().zip(&v[first*d..]){t.re-=c*x.re;t.im-=c*x.im;}}
            Ok(())
        })
    }
}
/// Orthonormal coordinates of a span from its Gram matrix K (row-major m x
/// m): Gram-Schmidt with dropping in the given order (ordered Cholesky
/// K_SS = L L^T over the kept vectors S, a vector dropped when its part
/// outside the span of those before it is below `dependent` of its norm,
/// squared). Returns T (row-major m x r) with W = Z T orthonormal, and
/// (largest / smallest diagonal of L)^2.
pub fn orthonormal_coordinates(k:&[f64],m:usize,dependent:f64)->(Vec<f64>,usize,f64){
    let mut kept:Vec<usize>=Vec::new();let mut lower:Vec<Vec<f64>>=Vec::new();
    for g in 0..m{
        let mut l=Vec::with_capacity(kept.len()+1);
        for(a,&ka)in kept.iter().enumerate(){let mut s=k[ka*m+g];for b in 0..a{s-=lower[a][b]*l[b];}l.push(s/lower[a][a]);}
        let residual=k[g*m+g]-l.iter().map(|x|x*x).sum::<f64>();
        if residual>dependent*k[g*m+g]{l.push(residual.sqrt());lower.push(l);kept.push(g);}
    }
    let r=kept.len();
    // L^-1 by forward substitution; T has the rows of L^-T on the kept vectors.
    let mut inverse=vec![0.;r*r];
    for c in 0..r{for i in c..r{let mut s=if i==c{1.}else{0.};for b in c..i{s-=lower[i][b]*inverse[b*r+c];}inverse[i*r+c]=s/lower[i][i];}}
    let mut t=vec![0.;m*r];
    for(a,&g)in kept.iter().enumerate(){for c in 0..r{t[g*r+c]=inverse[c*r+a];}}
    let(largest,smallest)=lower.iter().enumerate().map(|(a,row)|row[a]).fold((0_f64,f64::INFINITY),|(x,y),v|(x.max(v),y.min(v)));
    (t,r,(largest/smallest).powi(2))
}
/// sum_i c_i basis_i.
pub fn combine(basis:&[&[C]],c:&[f64],length:usize)->Vec<C>{
    let mut out=vec![ZERO;length];
    for(v,&a)in basis.iter().zip(c){if a==0.{continue;}for(t,&x)in out.iter_mut().zip(v.iter()){t.re+=a*x.re;t.im+=a*x.im;}}
    out
}

/// A system on one mode set at a shift (`Shift`: A + u v^T, the steady
/// trace row, or A - sigma), with its preconditioner and the vector kernels.
pub struct System<'a,'p>{
    pub op:LatticeOperator<'a>,
    pub pre:&'p mut dyn Preconditioner,
    pub shift:Shift,
    pub vectors:Vectors,
    /// Field indices of the central populations (the first copy).
    pub watch:Vec<usize>,
    pub timing:KrylovTiming,
    /// Operator applications (a complex shift's real form takes two per
    /// system application) and preconditioner applications.
    pub applications:usize,pub preconditions:usize,
}
impl<'a,'p> System<'a,'p>{
    pub fn new(op:LatticeOperator<'a>,pre:&'p mut dyn Preconditioner,shift:Shift)->Result<Self,String>{
        let n=op.n;let c0=op.centre*n*n;let watch=(0..n).map(|i|c0+i*n+i).collect();let vectors=Vectors::new(&op,shift.copies())?;
        Ok(Self{op,pre,shift,vectors,watch,timing:KrylovTiming::default(),applications:0,preconditions:0})
    }
    pub fn length(&self)->usize{self.op.dimension()*self.shift.copies()}
    /// y = M x (the trace row, or the shift).
    pub fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let start=Instant::now();
        match self.shift{
            Shift::Steady=>{self.op.apply(x,y)?;let n=self.watch.len();let t=self.watch.iter().fold(ZERO,|s,&k|C{re:s.re+x[k].re,im:s.im+x[k].im});
                for &k in &self.watch{add(&mut y[k],scaled(t,1./n as f64));}self.applications+=1;}
            Shift::Real(sigma)=>{self.op.apply(x,y)?;for(v,a)in y.iter_mut().zip(x){v.re-=sigma*a.re;v.im-=sigma*a.im;}self.applications+=1;}
            Shift::Complex(sigma)=>{
                let size=self.op.dimension();let(x1,x2)=x.split_at(size);let(y1,y2)=y.split_at_mut(size);
                self.op.apply(x1,y1)?;self.op.apply(x2,y2)?;self.applications+=2;
                for k in 0..size{let(a,b)=(x1[k],x2[k]);
                    y1[k].re+=-sigma.re*a.re+sigma.im*b.re;y1[k].im+=-sigma.re*a.im+sigma.im*b.im;
                    y2[k].re+=-sigma.re*b.re-sigma.im*a.re;y2[k].im+=-sigma.re*b.im-sigma.im*a.im;}
            }
        }
        self.timing.apply_ms+=start.elapsed().as_secs_f64()*1e3;Ok(())
    }
    /// y = P^-1 x.
    pub fn precondition(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let start=Instant::now();self.pre.apply(x,y)?;
        self.timing.precondition_ms+=start.elapsed().as_secs_f64()*1e3;self.preconditions+=1;Ok(())
    }
    /// The central populations of a field.
    pub fn populations(&self,x:&[C])->Vec<f64>{self.watch.iter().map(|&k|x[k].re).collect()}
    /// The right-hand side u.
    pub fn source(&self)->Vec<C>{
        let n=self.watch.len();let mut b=vec![ZERO;self.length()];for &k in &self.watch{b[k]=C{re:1./n as f64,im:0.};}b
    }
}

/// The last GMRES cycle's Arnoldi relation M P^-1 V_m = V_m+1 Hbar_m in
/// factored form: Hbar_m = Q_H [R; 0] with Q_H the product of the Givens
/// rotations.
pub struct Arnoldi{
    /// v_1 .. v_m+1 (v_m+1 is zero when the relation closed exactly).
    pub basis:Vec<Vec<C>>,
    /// Givens rotations (c, s): rotation j acts on rows j, j+1.
    pub rotations:Vec<(f64,f64)>,
    /// R by columns (column j has j+1 entries).
    pub r:Vec<Vec<f64>>,
    /// Central populations of P^-1 v_j, j < m.
    pub diagonal:Vec<Vec<f64>>,
}
impl Arnoldi{
    pub fn steps(&self)->usize{self.r.len()}
    /// Q_H^T t (t has m + 1 entries).
    pub fn rotate(&self,t:&mut[f64]){
        for(j,&(c,s))in self.rotations.iter().enumerate(){let(a,b)=(t[j],t[j+1]);t[j]=c*a+s*b;t[j+1]=-s*a+c*b;}
    }
    /// Q_H t.
    pub fn unrotate(&self,t:&mut[f64]){
        for(j,&(c,s))in self.rotations.iter().enumerate().rev(){let(a,b)=(t[j],t[j+1]);t[j]=c*a-s*b;t[j+1]=s*a+c*b;}
    }
    /// R^-1 g (g has m entries).
    pub fn back_substitute(&self,g:&[f64])->Vec<f64>{
        let m=self.r.len();let mut y=vec![0.;m];
        for i in(0..m).rev(){let mut s=g[i];for l in i+1..m{s-=self.r[l][i]*y[l];}y[i]=s/self.r[i][i];}
        y
    }
}

pub struct Krylov{
    pub x:Vec<C>,pub steps:usize,pub restarts:usize,pub restart:usize,pub stop:&'static str,
    /// The true relative residual |u - M x| / |u| of the returned iterate.
    pub residual:f64,
    /// The longest cycle's Arnoldi relation.
    pub arnoldi:Arnoldi,
}

/// Applies the rotations to a new Hessenberg column (j + 2 entries) and
/// returns the R column (j + 1 entries) with the new rotation.
fn givens(rotations:&[(f64,f64)],column:&[f64])->(Vec<f64>,(f64,f64)){
    let mut column=column.to_vec();
    for(i,&(c,s))in rotations.iter().enumerate(){let(a,b)=(column[i],column[i+1]);column[i]=c*a+s*b;column[i+1]=-s*a+c*b;}
    let j=rotations.len();let size=column[j].hypot(column[j+1]);
    let rotation=if size>0.{(column[j]/size,column[j+1]/size)}else{(1.,0.)};
    column[j]=size;column.truncate(j+1);(column,rotation)
}

/// Solves M x = b from x0 (zero when None).
pub fn gmres(system:&mut System<'_,'_>,b:&[C],x0:Option<Vec<C>>,groups:&[Vec<usize>],options:&KrylovOptions)->Result<Krylov,String>{
    let length=system.length();
    let bnorm=system.vectors.norm(b);
    let fit=(options.basis_bytes/(length*std::mem::size_of::<C>())).saturating_sub(2);
    let restart=options.restart.min(fit).max(8);
    let mut x=x0.unwrap_or_else(||vec![ZERO;length]);let(mut z,mut w)=(vec![ZERO;length],vec![ZERO;length]);
    let mut previous:Option<Vec<f64>>=None;let mut changes:Vec<f64>=Vec::new();
    let(mut steps,mut restarts)=(0,0);
    if bnorm==0.{return Err("Lattice steady source is zero".into());}
    // r = b - M x, recomputed from x after every cycle (see the module
    // documentation).
    let mut r=b.to_vec();
    if x.iter().any(|v|nonzero(*v)){system.apply(&x,&mut w)?;for((a,&bb),c)in r.iter_mut().zip(b).zip(&w){*a=C{re:bb.re-c.re,im:bb.im-c.im};}}
    let mut last_true=system.vectors.norm(&r)/bnorm;
    let mut longest:Option<Arnoldi>=None;
    loop{
        let beta=system.vectors.norm(&r);
        let mut arnoldi=Arnoldi{basis:vec![r.iter().map(|v|scaled(*v,1./beta)).collect()],rotations:vec![],r:vec![],diagonal:vec![]};
        let mut g=vec![beta];
        // Final (unrotated) Hessenberg columns, and the latest column's
        // once-orthogonalized coefficients: the last basis vector is then
        // the unnormalized candidate.
        let mut columns:Vec<Vec<f64>>=Vec::new();let mut provisional:Option<Vec<f64>>=None;
        let xdiag=system.populations(&x);
        let mut y=Vec::new();let mut stopped:Option<&'static str>=None;
        for j in 0..restart{
            system.precondition(&arnoldi.basis[j],&mut z)?;let zpop=system.populations(&z);
            system.apply(&z,&mut w)?;
            let start=Instant::now();
            let(done,rest)=arnoldi.basis.split_at_mut(j);let u=&mut rest[0];
            let refs:Vec<&[C]>=done.iter().map(|v|v.as_slice()).collect();
            // The candidate's reorthogonalization coefficients a and the
            // image's projections, one pass; alpha by Pythagoras unless the
            // candidate was mostly inside the span (then explicitly).
            let(mut a,projections,mut uu,mut uw)=system.vectors.fused_dots(&refs,u,&w);
            let inside=a.iter().map(|v|v*v).sum::<f64>();let explicit=!(uu-inside>0.25*uu);
            if explicit{
                system.vectors.subtract(&refs,&a,u)?;let again=system.vectors.dots(&refs,u);system.vectors.subtract(&refs,&again,u)?;
                for(x,y)in a.iter_mut().zip(again){*x+=y;}uu=system.vectors.dot(u,u);uw=system.vectors.dot(u,&w);
            }
            let alpha=(if explicit{uu}else{uu-inside}).sqrt();
            if !(alpha>0.&&alpha.is_finite()){return Err("Lattice GMRES basis broke down".into());}
            if j==0{g[0]*=alpha;}
            // Column j - 1 final: its coefficients plus a, and alpha.
            if let Some(mut column)=provisional.take(){
                for(x,y)in column.iter_mut().zip(&a){*x+=y;}column.push(alpha);
                let(rcol,(c,s))=givens(&arnoldi.rotations,&column);
                arnoldi.rotations.push((c,s));arnoldi.r.push(rcol);let gj=g[j-1];g[j-1]=c*gj;g.push(-s*gj);columns.push(column);
            }
            let mut pop=zpop;for(d,&ai)in arnoldi.diagonal.iter().zip(&a){for(t,&v)in pop.iter_mut().zip(d){*t-=ai*v;}}
            arnoldi.diagonal.push(pop.into_iter().map(|v|v/alpha).collect());
            // A P^-1 v_j = (w - V e) / alpha with e = Hbar a; its projections
            // h on v_0..v_j, assuming the basis orthonormal.
            let mut e=vec![0.;j+1];for(column,&ai)in columns.iter().zip(&a){for(t,&v)in e.iter_mut().zip(column){*t+=v*ai;}}
            let along=if explicit{uw/alpha}else{(uw-a.iter().zip(&projections).map(|(x,y)|x*y).sum::<f64>())/alpha};
            let mut h:Vec<f64>=(0..j).map(|i|(projections[i]-e[i])/alpha).collect();h.push((along-e[j])/alpha);
            let coefficients:Vec<f64>=e.iter().zip(&h).map(|(x,y)|x+alpha*y).collect();
            let reorthogonalize=if explicit{vec![0.;j]}else{a};
            let hn=system.vectors.fused_update(&refs,u,&reorthogonalize,alpha,&mut w,&coefficients)?.sqrt();
            system.timing.orthogonalize_ms+=start.elapsed().as_secs_f64()*1e3;
            steps+=1;
            // Tentative rotation of the provisional column j.
            let mut column=h.clone();column.push(hn);
            let(rcol,(c,s))=givens(&arnoldi.rotations,&column);
            let residual=(s*g[j]).abs()/bnorm;
            arnoldi.r.push(rcol);let mut gt=g.clone();gt[j]=c*g[j];y=arnoldi.back_substitute(&gt[..=j]);arnoldi.r.pop();
            let stable=if options.populations{
                let mut current=xdiag.clone();for(k,zk)in arnoldi.diagonal.iter().enumerate(){for(t,&v)in current.iter_mut().zip(zk){*t+=y[k]*v;}}
                let trace:f64=current.iter().sum();let current:Vec<f64>=current.iter().map(|v|v/trace).collect();
                let change=previous.as_ref().map_or(f64::INFINITY,|p|ledger::metric(p,&current,groups));changes.push(change);
                previous=Some(current);
                changes.len()>=STABLE_STEPS&&changes[changes.len()-STABLE_STEPS..].iter().all(|&c|c<options.algebraic)
            }else{true};
            if stable&&residual<=options.residual_guard{stopped=Some("populations-stable");}
            else if hn==0.{stopped=Some("exact");}
            else if residual<=1e-15{stopped=Some("residual-at-roundoff");}
            let last=stopped.is_some()||steps>=options.maximum_steps||j+1==restart;
            if !last{arnoldi.basis.push(w.clone());provisional=Some(h);continue;}
            // The cycle ends: reorthogonalize the image once more, close the
            // relation M P^-1 V_m = V_m+1 Hbar_m with column j final.
            let start=Instant::now();
            let mut column=h;
            let next=if hn>0.{
                let refs:Vec<&[C]>=arnoldi.basis.iter().map(|v|v.as_slice()).collect();
                let a=system.vectors.dots(&refs,&w);system.vectors.subtract(&refs,&a,&mut w)?;
                let size=system.vectors.norm(&w);for(x,y)in column.iter_mut().zip(&a){*x+=y;}column.push(size);
                if size>0.{w.iter().map(|v|scaled(*v,1./size)).collect()}else{vec![ZERO;length]}
            }else{column.push(0.);vec![ZERO;length]};
            let(rcol,(c,s))=givens(&arnoldi.rotations,&column);
            arnoldi.rotations.push((c,s));arnoldi.r.push(rcol);let gj=g[j];g[j]=c*gj;g.push(-s*gj);
            y=arnoldi.back_substitute(&g[..=j]);arnoldi.basis.push(next);
            system.timing.orthogonalize_ms+=start.elapsed().as_secs_f64()*1e3;
            break;
        }
        // x += P^-1 (V y).
        let combination={let basis:Vec<&[C]>=arnoldi.basis[..y.len()].iter().map(|v|v.as_slice()).collect();combine(&basis,&y,length)};
        system.precondition(&combination,&mut z)?;
        for(a,c)in x.iter_mut().zip(&z){add(a,*c);}
        system.apply(&x,&mut w)?;
        for((a,&bb),c)in r.iter_mut().zip(b).zip(&w){*a=C{re:bb.re-c.re,im:bb.im-c.im};}
        let true_residual=system.vectors.norm(&r)/bnorm;
        let arnoldi=match longest.take(){Some(l)if l.steps()>arnoldi.steps()=>l,_=>arnoldi};
        if let Some(stop)=stopped{
            if true_residual<=options.residual_guard{return Ok(Krylov{x,steps,restarts,restart,stop,residual:true_residual,arnoldi});}
            // Restarting no longer reduces the true residual: its floor.
            if true_residual>0.5*last_true{return Ok(Krylov{x,steps,restarts,restart,stop:"true-residual-floor",residual:true_residual,arnoldi});}
        }
        longest=Some(arnoldi);
        if steps>=options.maximum_steps{return Err(format!("Lattice GMRES did not converge in {steps} steps (true residual {true_residual:e})"));}
        last_true=true_residual;restarts+=1;
    }
}

/// x_q <- (x_q + x_-q^dagger) / 2: removes the roundoff asymmetry of a
/// source assembled on every mode.
pub fn symmetrize(op:&LatticeOperator<'_>,x:&mut[C]){
    let n=op.n;let d=n*n;
    for q in 0..op.modes.len(){let p=op.mirror[q];if p<q{continue;}
        for i in 0..n{for j in 0..n{let(a,b)=(q*d+i*n+j,p*d+j*n+i);if p==q&&j<i{continue;}
            let m=C{re:0.5*(x[a].re+x[b].re),im:0.5*(x[a].im-x[b].im)};x[a]=m;x[b]=conj(m);}}}
}
