//! R3 prototype (validation only): Floquet frames. The separable part of the
//! lattice operator (every left/right factor: L0's and the absorbable fold's
//! with -i nu_q, and the left/right parts of the explicit near steps) is a
//! Kronecker sum of a left and a right Floquet operator,
//!   (A f)_k = sum_t A_t f_{k-t} - i nu_k f_k,  (g B)_k = sum_t g_{k-t} B_t - i nu_k g_k,
//! and on the infinite lattice
//!   S = P Q (Lambda . + . M - i nu_q) Q^-1 P^-1,
//! with P, Q the charge convolutions by the micromotion kernels (P x)_q =
//! sum_k P_k x_{q-k}, (x Q)_q = sum_l x_{q-l} Q_l: the current eigenframes are
//! the kernel support {0}. Here the kernels carry the strong near steps to
//! first order (Brillouin-Wigner self-consistent central frames), the inverse
//! kernels the Neumann series of P(theta)^-1 to second order, all truncated
//! to the support K = {0} + strong steps. Applied as four charge convolutions
//! and one elementwise division; the truncation is left to GMRES.
use super::*;
use super::fold::{eigen_frame,transpose};
use super::operator::LatticeOperator;
use super::preconditioner::Preconditioner;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

#[derive(Clone,Debug,Default)]
pub struct FloquetDiagnostics{
    pub milliseconds:f64,pub iterations:[usize;2],pub change:[f64;2],
    /// Frame pairs left out of the first-order kernels (|mixing| above the guard).
    pub guarded:[usize;2],
    /// |P_s|_F / |P_0|_F per strong step (left).
    pub kernel_size:Vec<f64>,
    pub condition:[f64;2],
}
/// One side's Floquet frame: central quasi-energies, the forward kernels
/// P_k and the inverse kernels on the support (row-major n x n).
struct Side{values:Vec<C>,forward:Vec<Vec<C>>,backward:Vec<Vec<C>>}
pub struct FloquetFrames{
    n:usize,
    /// Centre Woodbury factors in the central Floquet frame (recycling and
    /// the diagonal fold rest on ground-pair rows and columns, as the dressed
    /// preconditioner's): U-hat d x r and V-hat r x d, column-major; the trace
    /// row's u-hat and v-hat.
    uhat:Vec<C>,vhat:Vec<C>,rank:usize,trace_u:Vec<C>,trace_v:Vec<C>,scale:f64,
    /// The kernel support: 0 first, then the strong steps (charges).
    pub support:Vec<Charge>,
    left:Side,
    /// Right side stored as its transposes' frame (Q_l = R_l^T).
    right:Side,
    lapack:Arc<Lapack>,
    pub diagnostics:FloquetDiagnostics,
}
fn mat_mul(a:&[C],b:&[C],n:usize)->Vec<C>{let mut o=vec![ZERO;n*n];for i in 0..n{for k in 0..n{let x=a[i*n+k];if !nonzero(x){continue;}for j in 0..n{mul_add(&mut o[i*n+j],x,b[k*n+j]);}}}o}
fn mat_add(a:&mut[C],b:&[C],s:C){for(x,&y)in a.iter_mut().zip(b){mul_add(x,s,y);}}
fn frob(a:&[C])->f64{a.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()}

/// First-order Floquet frame of one side: a0 the central matrix, steps
/// (charge, a_s, nu_s), all row-major. Returns the side, iterations, last
/// change, guarded pairs and kernel sizes.
fn side(lapack:&Lapack,n:usize,a0:&[C],steps:&[(Charge,Vec<C>,f64)],guard:f64)->Result<(Side,usize,f64,usize,Vec<f64>,f64),String>{
    let(l0,v,vinv,_,_)=eigen_frame(lapack,a0,n)?;
    let mut eff=a0.to_vec();let mut values:Vec<C>=Vec::new();let mut iterations=0;let mut change=f64::INFINITY;let mut guarded=0;
    let mut p0=Vec::new();let mut p0inv=Vec::new();let mut ps:Vec<Vec<C>>=Vec::new();let mut condition=0.;
    let scale=l0.iter().map(|z|magnitude(*z)).fold(0.,f64::max).max(1.);
    for it in 0..12{
        let(lam,p,pinv,cond,_)=eigen_frame(lapack,&eff,n)?;condition=cond;
        // Match the new values to the old order (nearest), to measure the change.
        if !values.is_empty(){
            change=lam.iter().map(|z|values.iter().map(|w:&C|magnitude(C{re:z.re-w.re,im:z.im-w.im})).fold(f64::INFINITY,f64::min)).fold(0.,f64::max)/scale;}
        values=lam;p0=p;p0inv=pinv;iterations=it+1;
        // P_s = V X, X[a,alpha] = -(V^-1 A_s P0)[a,alpha] / (l0_a - lambda_alpha - i nu_s).
        ps.clear();guarded=0;
        for(_,a_s,nu)in steps{
            let r=mat_mul(&vinv,&mat_mul(a_s,&p0,n),n);let mut x=vec![ZERO;n*n];
            for a in 0..n{for al in 0..n{let den=C{re:l0[a].re-values[al].re,im:l0[a].im-values[al].im-nu};
                let num=r[a*n+al];if magnitude(num)>guard*magnitude(den){guarded+=1;continue;}
                let s=den.re*den.re+den.im*den.im;x[a*n+al]=C{re:-(num.re*den.re+num.im*den.im)/s,im:-(num.im*den.re-num.re*den.im)/s};}}
            ps.push(mat_mul(&v,&x,n));
        }
        if change<1e-10{break;}
        // eff = a0 + sum_s A_-s P_s P0^-1.
        let mut next=a0.to_vec();
        for(i,(charge,_,_))in steps.iter().enumerate(){
            let back=opposite(charge);let Some(j)=steps.iter().position(|s|s.0==back)else{continue};
            let t=mat_mul(&steps[j].1,&mat_mul(&ps[i],&p0inv,n),n);mat_add(&mut next,&t,ONE);}
        eff=next;
    }
    // Inverse kernels: the Neumann series of (P0 + dP(theta))^-1 to second order on the support.
    let mut backward=vec![p0inv.clone()];
    let mut centre=p0inv.clone();
    for(i,(charge,_,_))in steps.iter().enumerate(){
        let t=mat_mul(&p0inv,&mat_mul(&ps[i],&p0inv,n),n);
        backward.push(t.iter().map(|z|C{re:-z.re,im:-z.im}).collect());
        let back=opposite(charge);
        if let Some(j)=steps.iter().position(|s|s.0==back){let w=mat_mul(&t,&mat_mul(&ps[j],&p0inv,n),n);mat_add(&mut centre,&w,ONE);}
    }
    backward[0]=centre;
    let size0=frob(&p0);let sizes=ps.iter().map(|p|frob(p)/size0).collect();
    let mut forward=vec![p0];forward.extend(ps);
    Ok((Side{values,forward,backward},iterations,change,guarded,sizes,condition))
}

/// R3 prototype switch: Floquet frames when ATOM_R3_PRECOND=floquet, with the
/// kernel floor ATOM_R3_KAPPA (default 0.005).
pub fn from_env(prepared:&LatticePrepared)->Result<Option<FloquetFrames>,String>{
    if std::env::var("ATOM_R3_PRECOND").ok().as_deref()!=Some("floquet"){return Ok(None);}
    let kappa=std::env::var("ATOM_R3_KAPPA").ok().and_then(|v|v.parse().ok()).unwrap_or(0.005);
    FloquetFrames::by_kernel(prepared,kappa,0.5).map(Some)
}
/// R3 prototype: the shifted solves' relative residual (ATOM_R3_INNER_RTOL).
pub fn inner_rtol(default:f64)->f64{std::env::var("ATOM_R3_INNER_RTOL").ok().and_then(|v|v.parse().ok()).unwrap_or(default)}
impl FloquetFrames{
    /// R3 diagnostic: singular values (descending) of each micromotion kernel
    /// in frame coordinates, E_s = P_0^-1 P_s.
    pub fn kernel_singular_values(&self)->Result<Vec<Vec<f64>>,String>{
        let n=self.n;let p0inv=super::fold::inverse(&self.lapack,&self.left.forward[0],n)?;
        self.left.forward[1..].iter().map(|ps|{let e=mat_mul(&p0inv,ps,n);
            // E^H E (row-major), Hermitian; its eigenvalues are sigma^2.
            let mut g=vec![ZERO;n*n];for i in 0..n{for j in 0..n{let mut acc=ZERO;for k in 0..n{mul_add(&mut acc,conj(e[k*n+i]),e[k*n+j]);}g[i*n+j]=acc;}}
            let(values,_)=self.lapack.hermitian_eigen(&g,n)?;let mut sv:Vec<f64>=values.iter().map(|v|v.max(0.).sqrt()).collect();sv.sort_by(|a,b|b.total_cmp(a));Ok(sv)}).collect()
    }
    /// Frames with the explicit near steps whose mixing size / |nu| is at
    /// least `strong` (both signs of each), the first-order element guard.
    pub fn new(prepared:&LatticePrepared,strong:f64,guard:f64)->Result<Self,String>{Self::with(prepared,strong,guard,usize::MAX)}
    /// As `new`, keeping at most `pairs` step pairs (the most strongly mixing).
    /// The right side is the conjugate of the left (NR'^T = conj NL' + c I and
    /// B_t^T = conj A_-t for the Lindblad form; checked): R_l = conj P_-l,
    /// mu = conj lambda + c.
    /// The strength rule: every explicit pair whose first-order kernel
    /// |P_s|_F / |P_0|_F is at least `floor` (kernels computed with all pairs
    /// mixing above 1e-4, then the frames rebuilt with the kept ones).
    pub fn by_kernel(prepared:&LatticePrepared,floor:f64,guard:f64)->Result<Self,String>{
        let all=Self::with(prepared,1e-4,guard,usize::MAX)?;
        let kept:Vec<Charge>=all.support[1..].iter().zip(&all.diagnostics.kernel_size).filter(|(_,&k)|k>=floor).map(|(q,_)|q.clone()).collect();
        let mut out=Self::select(prepared,guard,&|q:&Charge|kept.contains(q)||kept.contains(&opposite(q)))?;
        out.diagnostics.milliseconds+=all.diagnostics.milliseconds;Ok(out)
    }
    pub fn with(prepared:&LatticePrepared,strong:f64,guard:f64,pairs:usize)->Result<Self,String>{
        let mut candidates:Vec<(f64,usize)>=prepared.plan.explicit().into_iter().map(|s|{let st=&prepared.plan.steps[s];(st.size/st.nu.abs(),s)})
            .filter(|&(m,s)|m>=strong&&prepared.plan.steps[s].charge>=opposite(&prepared.plan.steps[s].charge)).collect();
        candidates.sort_by(|a,b|b.0.total_cmp(&a.0));candidates.truncate(pairs);
        let keep:Vec<Charge>=candidates.iter().map(|&(_,s)|prepared.plan.steps[s].charge.clone()).collect();
        Self::select(prepared,guard,&|q:&Charge|keep.contains(q)||keep.contains(&opposite(q)))
    }
    fn select(prepared:&LatticePrepared,guard:f64,keep:&dyn Fn(&Charge)->bool)->Result<Self,String>{
        let clock=Instant::now();let lapack=&prepared.lapack;let generator=&prepared.generator;let n=generator.n;let d=n*n;
        let zero=generator.zero_component();
        let diagonal=prepared.fold.targets.first().filter(|t|t.charge==generator.zero);
        let(mut nl,mut nr)=(zero.op.left.dense(n),zero.op.right.dense(n));
        if let Some(t)=diagonal{for k in 0..d{add(&mut nl[k],t.left[0][k]);add(&mut nr[k],t.right[0][k]);}}
        let mut chosen=Vec::new();
        for s in prepared.plan.explicit(){let st=&prepared.plan.steps[s];if !keep(&st.charge){continue;}
            let c=&generator.components[&st.charge];chosen.push((st.charge.clone(),c.op.left.dense(n),st.nu));}
        let(left,il,cl,gl,sizes,condl)=side(lapack,n,&nl,&chosen,guard)?;
        // Conjugate check and the scalar c.
        let shift=C{re:nr[0].re-nl[0].re,im:nr[0].im+nl[0].im};let scale=nl.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let mut defect=0_f64;
        for i in 0..n{for j in 0..n{let a=nr[j*n+i];let b=nl[i*n+j];let mut e=C{re:a.re-b.re,im:a.im+b.im};if i==j{e.re-=shift.re;e.im-=shift.im;}defect=defect.max(magnitude(e));}}
        for(q,_,_)in &chosen{let back=opposite(q);let b=transpose(&generator.components[q].op.right.dense(n),n);let a=generator.components[&back].op.left.dense(n);
            for k in 0..d{defect=defect.max(magnitude(C{re:b[k].re-a[k].re,im:b[k].im+a[k].im}));}}
        if defect>1e-12*scale{return Err(format!("Floquet frames: right data are not the conjugate of the left ({:e})",defect/scale));}
        let mut support=vec![generator.zero.clone()];support.extend(chosen.iter().map(|c|c.0.clone()));
        // Right: R_l = conj(P_-l) on the support (index of -l), mu = conj(lambda) + c.
        let index_of=|q:&Charge|support.iter().position(|k|k==q).unwrap();
        let conj_all=|m:&Vec<C>|m.iter().map(|z|conj(*z)).collect::<Vec<C>>();
        let forward:Vec<Vec<C>>=support.iter().map(|k|conj_all(&left.forward[index_of(&opposite(k))])).collect();
        let backward:Vec<Vec<C>>=support.iter().map(|k|conj_all(&left.backward[index_of(&opposite(k))])).collect();
        let values:Vec<C>=left.values.iter().map(|z|C{re:z.re+shift.re,im:-z.im+shift.im}).collect();
        let right=Side{values,forward,backward};
        let diagnostics=FloquetDiagnostics{milliseconds:clock.elapsed().as_secs_f64()*1e3,iterations:[il,il],change:[cl,cl],guarded:[gl,gl],kernel_size:sizes,condition:[condl,condl]};
        // Centre Woodbury in the central frame: hat(Y) = Pb0 Y Qb0, X = Pf0 hat(X) Qf0.
        let(pb,qb,pf,qf)=(&left.backward[0],transpose(&right.backward[0],n),&left.forward[0],transpose(&right.forward[0],n));
        let mut w=vec![ZERO;d*d];
        Superop{left:Sparse::default(),right:Sparse::default(),sandwich:zero.op.sandwich.clone()}.dense_into(n,ONE,&mut w);
        if let Some(t)=diagonal{t.rest[0].dense_into(d,ONE,&mut w);}
        let ground_pair=|p:usize|generator.ground[p/n]&&generator.ground[p%n];
        let positions:Vec<usize>=(0..d).filter(|&p|ground_pair(p)).collect();let r=positions.len();let rank=2*r;
        let mut uhat=vec![ZERO;d*rank];let mut vrows=vec![ZERO;rank*d];
        for(k,&p)in positions.iter().enumerate(){
            uhat[k*d+p]=ONE;for row in 0..d{if !ground_pair(row){uhat[(r+k)*d+row]=w[row*d+p];}}
            vrows[k*d..(k+1)*d].copy_from_slice(&w[p*d..(p+1)*d]);vrows[(r+k)*d+p]=ONE;}
        drop(w);
        let sandwich=|a:&[C],x:&[C],b:&[C]|mat_mul(&mat_mul(a,x,n),b,n);
        let(pft,qft)=(transpose(pf,n),transpose(&qf,n));
        for k in 0..rank{let u=sandwich(pb,&uhat[k*d..(k+1)*d],&qb);uhat[k*d..(k+1)*d].copy_from_slice(&u);
            let v=sandwich(&pft,&vrows[k*d..(k+1)*d],&qft);vrows[k*d..(k+1)*d].copy_from_slice(&v);}
        let mut vhat=vec![ZERO;rank*d];for k in 0..rank{for c in 0..d{vhat[k+rank*c]=vrows[k*d+c];}}
        let mut tu=vec![ZERO;d];let mut tv=vec![ZERO;d];for i in 0..n{tu[i*n+i]=C{re:1./n as f64,im:0.};tv[i*n+i]=ONE;}
        let trace_u=sandwich(pb,&tu,&qb);let trace_v=sandwich(&pft,&tv,&qft);
        let scale=left.values.iter().map(|z|magnitude(*z)).fold(0.,f64::max)+right.values.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        Ok(Self{n,support,left,right,lapack:Arc::clone(lapack),diagnostics,uhat,vhat,rank,trace_u,trace_v,scale})
    }
    /// The preconditioner on an operator's modes at the real shift sigma:
    /// the Hermitian half (c_-q = c_q^dagger in frame coordinates), each
    /// two-sided transform's inner one-sided stage on the modes the canonical
    /// outputs need, its outer stage on the canonical modes.
    pub fn batch<'a>(&'a self,op:&LatticeOperator<'_>,sigma:f64)->Result<FloquetBatch<'a>,String>{self.batch_with(op,sigma,false,false)}
    /// With the centre Woodbury (`woodbury`) and the steady trace row
    /// (`trace`); centre denominators at roundoff are moved and corrected.
    pub fn batch_with<'a>(&'a self,op:&LatticeOperator<'_>,sigma:f64,woodbury:bool,trace:bool)->Result<FloquetBatch<'a>,String>{
        let shell=std::env::var("ATOM_R3_SHELL").ok().and_then(|v|v.parse::<usize>().ok());
        let active=shell.map(|k|{let ball:BTreeSet<Charge>=self.ball(k);op.modes.iter().map(|q|ball.contains(q)).collect::<Vec<bool>>()});
        self.batch_masked(op,sigma,woodbury,trace,active.as_deref())
    }
    /// Charges within k kernel steps of 0.
    fn ball(&self,k:usize)->BTreeSet<Charge>{
        let mut all=BTreeSet::from([self.support[0].clone()]);let mut frontier=vec![self.support[0].clone()];
        for _ in 0..k{let mut next=Vec::new();for q in &frontier{for s in &self.support[1..]{if let Ok(t)=charge_plus(q,s){if all.insert(t.clone()){next.push(t);}}}}frontier=next;}
        all
    }
    /// As `batch_with`; the micromotion terms act only on the output modes
    /// flagged `active` (the others keep the central frame alone).
    pub fn batch_masked<'a>(&'a self,op:&LatticeOperator<'_>,sigma:f64,woodbury:bool,trace:bool,active:Option<&[bool]>)->Result<FloquetBatch<'a>,String>{
        self.general(op,C{re:sigma,im:0.},woodbury,trace,active,false)
    }
    /// A complex shift: fields are not Hermitian-symmetric, so every stage
    /// runs on every mode (no mirror).
    pub fn batch_complex<'a>(&'a self,op:&LatticeOperator<'_>,sigma:C,woodbury:bool)->Result<FloquetBatch<'a>,String>{
        self.general(op,sigma,woodbury,false,None,true)
    }
    fn general<'a>(&'a self,op:&LatticeOperator<'_>,sigma:C,woodbury:bool,trace:bool,active:Option<&[bool]>,full:bool)->Result<FloquetBatch<'a>,String>{
        let(n,d)=(self.n,self.n*self.n);let modes=op.modes.len();
        let neighbours:Vec<Vec<Option<usize>>>=op.modes.iter().enumerate().map(|(m,q)|self.support.iter().enumerate().map(|(i,k)|
            if i>0&&active.is_some_and(|a|!a[m]){None}else{charge_minus(q,k).ok().and_then(|c|op.index.get(&c).copied())}).collect()).collect();
        let mut needed=op.canonical.clone();
        for q in 0..modes{if op.canonical[q]{for p in neighbours[q].iter().flatten(){needed[*p]=true;}}}
        let mut denominators=vec![ZERO;modes*d];
        for(q,&w)in op.nu.iter().enumerate(){for a in 0..n{for b in 0..n{
            let z=C{re:self.left.values[a].re+self.right.values[b].re-sigma.re,im:self.left.values[a].im+self.right.values[b].im-w-sigma.im};
            let s=z.re*z.re+z.im*z.im;denominators[q*d+a*n+b]=C{re:z.re/s,im:-z.im/s};}}}
        if full{needed=vec![true;modes];}
        let count=|set:&[bool]|->usize{neighbours.iter().enumerate().filter(|(q,_)|set[*q]).map(|(_,v)|v.iter().filter(|x|x.is_some()).count()).sum()};
        let work=2*(count(&needed)+count(&op.canonical))*n*n*n;
        let qf:Vec<Vec<C>>=self.right.forward.iter().map(|m|transpose(m,n)).collect();
        let qb:Vec<Vec<C>>=self.right.backward.iter().map(|m|transpose(m,n)).collect();
        let half_work=4*count(&op.canonical)*n*n*n;
        // Centre factor: rank columns (Woodbury), the trace row, moved denominators.
        let centre=op.centre;let tiny=1e-13*self.scale;let regular=1e-3*self.scale.max(1e-300);
        let mut moved=Vec::new();
        for k in 0..d{let(a,b)=(k/n,k%n);let z=C{re:self.left.values[a].re+self.right.values[b].re-sigma.re,im:self.left.values[a].im+self.right.values[b].im-op.nu[centre]-sigma.im};
            if magnitude(z)<tiny{moved.push((k,z));denominators[centre*d+k]=C{re:-1./regular,im:0.};}}
        let rank=if woodbury{self.rank}else{0};let extra=usize::from(trace)+moved.len();let total=rank+extra;
        let mut factor=None;
        if total>0{
            let mut u=vec![ZERO;d*total];let mut v=vec![ZERO;total*d];
            if rank>0{u[..d*rank].copy_from_slice(&self.uhat[..d*rank]);for k in 0..rank{for c in 0..d{v[k+total*c]=self.vhat[k+rank*c];}}}
            let mut e=rank;
            if trace{u[e*d..(e+1)*d].copy_from_slice(&self.trace_u);for c in 0..d{v[e+total*c]=self.trace_v[c];}e+=1;}
            // Moved: the true denominator z replaced by -regular: add (z + regular) e_k e_k^T.
            for &(k,z)in &moved{u[e*d+k]=ONE;v[e+total*k]=C{re:z.re+regular,im:z.im};e+=1;}
            let rec=&denominators[centre*d..(centre+1)*d];
            let mut g=vec![ZERO;total*total];for k in 0..total{g[k+total*k]=ONE;}
            for j in 0..total{for i in 0..total{let mut acc=ZERO;for c in 0..d{mul_add(&mut acc,v[i+total*c],mul(rec[c],u[j*d+c]));}add(&mut g[i+total*j],acc);}}
            let ginv=super::fold::inverse(&self.lapack,&g,total)?;
            let du:Vec<C>=(0..d*total).map(|k|mul(rec[k%d],u[k])).collect();
            factor=Some(CentreFactor{total,du,v,ginv});
        }
        let work=if full{4*count(&vec![true;modes])*n*n*n}else{work}+factor.as_ref().map_or(0,|f|2*f.total*d+f.total*f.total);
        Ok(FloquetBatch{frames:self,neighbours,denominators,canonical:op.canonical.clone(),mirror:op.mirror.clone(),needed,qf,qb,
            execution:Execution::new()?,buffers:[vec![],vec![]],work,half_work,centre,factor,moved:moved.len(),full})
    }
}
/// (D + U V^T)^-1 = D^-1 - D^-1 U G^-1 V^T D^-1 at the centre: D^-1 U
/// (d x total), V (total x d) and G^-1, column-major.
struct CentreFactor{total:usize,du:Vec<C>,v:Vec<C>,ginv:Vec<C>}
pub struct FloquetBatch<'a>{
    centre:usize,factor:Option<CentreFactor>,pub moved:usize,full:bool,
    frames:&'a FloquetFrames,neighbours:Vec<Vec<Option<usize>>>,denominators:Vec<C>,
    canonical:Vec<bool>,mirror:Vec<usize>,needed:Vec<bool>,qf:Vec<Vec<C>>,qb:Vec<Vec<C>>,
    execution:Execution,buffers:[Vec<C>;2],
    /// Multiply-adds of one application, and of the ideal Hermitian half
    /// (every stage on canonical modes only).
    pub work:usize,pub half_work:usize,
}
/// out_q = sum_k (left ? K_k in_{q-k} : in_{q-k} K_k) over the support.
fn convolve(execution:&Execution,lapack:&Lapack,n:usize,neighbours:&[Vec<Option<usize>>],kernels:&[Vec<C>],left:bool,input:&[C],out:&mut[C],only:Option<&[bool]>)->Result<(),String>{
    let d=n*n;let work=neighbours.len()*kernels.len()*n*n*n;
    execution.for_mode_chunks(out,d,work,MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
        let gemm=lapack.complex_gemm();
        for(m,o)in chunk.chunks_exact_mut(d).enumerate(){let q=first+m;o.fill(ZERO);
            if let Some(c)=only{if !c[q]{continue;}}
            for(k,nb)in neighbours[q].iter().enumerate(){let Some(p)=nb else{continue};let x=&input[p*d..(p+1)*d];
                // Row-major y = K x is column-major y^T = x^T K^T: multiply(x, K); y = x K: multiply(K, x).
                if left{gemm.multiply(n,n,n,ONE,x,n,&kernels[k],n,ONE,o,n)?;}else{gemm.multiply(n,n,n,ONE,&kernels[k],n,x,n,ONE,o,n)?;}}
        }
        Ok(())
    })
}
impl Preconditioner for FloquetBatch<'_>{
    fn work(&self)->usize{self.work}
    fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let f=self.frames;let n=f.n;let lapack=&f.lapack;
        let[a,b]=&mut self.buffers;grow(a,x.len());grow(b,x.len());let(a,b)=(&mut a[..x.len()],&mut b[..x.len()]);
        let outer=if self.full{None}else{Some(self.canonical.as_slice())};
        convolve(&self.execution,lapack,n,&self.neighbours,&self.qb,false,x,a,Some(&self.needed))?;
        convolve(&self.execution,lapack,n,&self.neighbours,&f.left.backward,true,a,b,outer)?;
        if !self.full{mirror_field(b,n,&self.canonical,&self.mirror);}
        for(v,r)in b.iter_mut().zip(&self.denominators){*v=mul(*v,*r);}
        if let Some(fc)=&self.factor{let d=n*n;let c0=&mut b[self.centre*d..(self.centre+1)*d];let t=fc.total;
            let mut w=vec![ZERO;t];for c in 0..d{let x=c0[c];for i in 0..t{mul_add(&mut w[i],fc.v[i+t*c],x);}}
            let mut z=vec![ZERO;t];for j in 0..t{for i in 0..t{mul_add(&mut z[i],fc.ginv[i+t*j],w[j]);}}
            for j in 0..t{let zj=z[j];for c in 0..d{let s=mul(fc.du[j*d+c],zj);c0[c].re-=s.re;c0[c].im-=s.im;}}}
        convolve(&self.execution,lapack,n,&self.neighbours,&self.qf,false,b,a,Some(&self.needed))?;
        convolve(&self.execution,lapack,n,&self.neighbours,&f.left.forward,true,a,y,outer)?;
        if !self.full{mirror_field(y,n,&self.canonical,&self.mirror);}
        if y.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()){return Err("Nonfinite Floquet preconditioner action".into());}
        Ok(())
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::operator::symmetric_field;
    use super::super::preconditioner::StructuredDressed;
    #[test]
    fn lattice_floquet_frames_with_the_central_support_are_the_eigenframes(){
        // Support {0}: the Floquet frames reduce to the separable dressed
        // preconditioner (eigenframes of NL', NR' at -i nu_q - sigma).
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();let op=LatticeOperator::new(&p,&modes).unwrap();
        let frames=FloquetFrames::by_kernel(&p,1e9,0.5).unwrap();assert_eq!(frames.support.len(),1);
        let dressed=StructuredDressed::new(&p).unwrap();
        for sigma in [0.5,20.]{
            let mut a=frames.batch(&op,sigma).unwrap();let mut b=dressed.batch(&op.nu,&op.canonical,&op.mirror,None,sigma,false).unwrap();
            let x=symmetric_field(&op,1);let(mut ya,mut yb)=(vec![ZERO;x.len()],vec![ZERO;x.len()]);a.apply(&x,&mut ya).unwrap();b.apply(&x,&mut yb).unwrap();
            let scale=yb.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            let error=ya.iter().zip(&yb).map(|(u,v)|magnitude(C{re:u.re-v.re,im:u.im-v.im})).fold(0.,f64::max);
            assert!(error<=1e-10*scale,"sigma {sigma}: {error:e} of {scale:e}");
        }
    }
}
