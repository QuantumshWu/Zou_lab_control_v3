//! The fold remainders charged by their action on the computed state (R2,
//! unified/R2-estimate.md section 13).
//!
//! The lattice operator A leaves remainders out of the folded far shell
//! (`ledger`). Their generator-norm bounds times a relaxation time are
//! 6-8 orders of magnitude above what they do: the K^(1) rests act on the
//! coherences of the fast Raman blocks only, and a coherence source
//! reaches the populations through the near couplings. So the remainders
//! the lattice holds are built and their first-order effect on the outputs
//! computed, signed:
//! - the K^(1) rests the operator leaves out but for their population rows
//!   (`operator::K1_REST_SHARE`): i nu_p (K^(1) rest - its population rows)
//!   on every fold edge p -> q;
//! - the memory of the fold: the far blocks eliminated at the Laplace
//!   frequency z give K^(0) + (i nu_p + z) K^(1), and z K^(1) x is K^(1) A x
//!   on every fold edge;
//! - the third-order returns through the explicit near couplings
//!   (`FoldLibrary::third_order_source`).
//! Together dA_b. With x(t) = W B e^{Theta t} a, to first order
//!   dx(t) = int_0^t e^{(t-s)A} dA_b x(s) ds,
//! read from a Galerkin model that holds both the trajectory and its
//! response: on M = [x0, Z, V_s, Y] (Z = W U_r for the leading left singular
//! vectors of B diag(a), V_s the slow Ritz vectors, Y_m = (theta_m - A)^-1
//! dA_b v_m their responses solved on the shared pool, the kernel's
//! bordered) G = Q^T A Q and dG = Q^T dA_b Q (Q orthonormal over M, both
//! with the centre trace kept), dx(t) = Q S c(t) with
//!   c_k(t) = sum_l Phi_kl(t) (S^-1 dG S)_kl a_l,
//!   Phi_kl(t) = (e^{theta_k t} - e^{theta_l t}) / (theta_k - theta_l)
//! (t e^{theta t} at coincidence: the shift of a slow Ritz value). Only
//! A x0, A W and the pool's images enter: dA_b acts on 1 + r + n_s (+ Y)
//! fields. The steady state's response is the kernel's Y itself,
//! a_0 Y_0 = -a_0 A^# dA_b v_0 (the model's long-time mean on the kernel's
//! augmentation reproduces it). Signed estimate/actual 0.99-1.11 on three
//! to five lasers at 10 ms, the steady state and the mean over 1 s (R2 13.3).
//!
//! Time rule (as the slow part's): an output at T with T lambda_slow >= 1,
//! every horizon and the steady state take the slow responses. A shorter
//! trajectory is read from the pipeline's own model, dG = B^+ W^T D B with
//! D = dA_b W in the low-rank form L C (exact on the slow Ritz vectors),
//! plus the response of D's part outside the space, (I - W W^T) L, through
//! the fast part's SW propagation (`ErrorLattice::field_populations`).
//!
//! The remainders the lattice does not hold (`RemainderLedger::unbuilt`)
//! are higher products of the same far couplings and far resolvents as the
//! fold K itself, smaller by the ledger's ratio: they are charged
//! SAFETY S_K U, S_K the steady response to the stored fold K (one kernel
//! solve) per unit of its size. Not a strict bound (measured margin 12-29);
//! the third order changes the near couplings themselves and must be built
//! (S_K underestimates it 30 times).
use super::super::*;
use super::super::estimate_api::{EstimateSetup,Outputs,SteadySolves};
use super::super::ledger::{RemainderLedger,metric};
use super::super::operator::LatticeOperator;
use super::lattice::ErrorLattice;
use super::propagate::{FieldSources,Readout};
use super::slow::{self,Packing,SlowPool,block_cross,block_times,cexp,cholesky,upper_inverse};
use super::Checkpoint;
use crate::parallel::Execution;

/// The source's rank: singular values of B diag(a) above this fraction of
/// the largest (R2: 1e-2 is unstable, 14-20x).
const SOURCE_RANK:f64=1e-3;
/// The unbuilt remainders' charge SAFETY S_K U (without it the measured
/// margin is 1.2-2.9).
const SAFETY:f64=10.;
/// The compact model's kernel: its Ritz value of least modulus below this.
const MODEL_KERNEL:f64=1e-6;
/// A Ritz value below this is the kernel (theta = 0 exactly).
const KERNEL:f64=1e-12;
/// Two Ritz values within this (relative) are a conjugate pair.
const PAIRED:f64=1e-8;
/// Source columns per task of a dense K^(1) rest sector's GEMM.
const DENSE_COLUMNS:usize=64;
/// The slow responses' pool solves stop at this residual (R2: only their
/// directions outside the space enter the model).
const RESPONSE_RESIDUAL:f64=1e-4;

/// The built remainders' effect per output (population differences) and
/// the unbuilt ones' charge.
#[derive(Clone,Debug,Default)]
pub struct FoldEffect{pub trajectory:Vec<Vec<f64>>,pub horizons:Vec<Vec<f64>>,pub steady:Option<Vec<f64>>,pub unbuilt:f64}

#[derive(Clone,Debug,Default)]
pub struct FoldWork{
    /// dA_b applications, applications of the stored fold, pool solves.
    pub remainders:usize,pub folds:usize,pub solves:usize,
    /// Time (ms): the K^(1) rests and the memory, the third order, the
    /// slow responses, the models and the fast part.
    pub milliseconds:[f64;4],
}

pub struct FoldCharge<'a>{
    prepared:&'a LatticePrepared,
    /// The pipeline's lattice operator on P u Q (whether it leaves the
    /// K^(1) rests out; its fold part, the calibration's perturbation).
    op:LatticeOperator<'a>,
    /// Fold edges (target, source mode, target mode) on P u Q, the diagonal
    /// target's included.
    edges:Vec<(usize,usize,usize)>,
    /// x0 on the lattice and |x0|.
    x0:Vec<C>,x0_norm:f64,
    pub work:FoldWork,
}

/// A reduced model's first-order response: values, coefficients and
/// dG^ = S^-1 dG S (column-major r x r) in its Ritz coordinates.
struct Response{theta:Vec<C>,a:Vec<C>,dg:Vec<C>}
impl Response{
    /// c(t) (`Phi` divided differences).
    fn at(&self,t:f64)->Vec<C>{
        let r=self.theta.len();
        (0..r).map(|k|(0..r).fold(ZERO,|s,l|{let v=mul(mul(phi(self.theta[k],self.theta[l],t),self.dg[k+r*l]),self.a[l]);C{re:s.re+v.re,im:s.im+v.im}})).collect()
    }
}
/// (e^{a t} - e^{b t}) / (a - b) = e^{hi t} t phi_1((lo - hi) t), hi the
/// value with the larger real part, phi_1(x) = expm1(x) / x.
fn phi(a:C,b:C,t:f64)->C{
    let(hi,lo)=if a.re>b.re{(a,b)}else{(b,a)};
    let x=C{re:(lo.re-hi.re)*t,im:(lo.im-hi.im)*t};
    let h=if magnitude(x)<1e-8{C{re:1.+x.re/2.,im:x.im/2.}}else{
        // expm1(x) = expm1(Re x) cos(Im x) - 2 sin^2(Im x / 2) + i e^{Re x} sin(Im x).
        let e=C{re:x.re.exp_m1()*x.im.cos()-2.*(x.im/2.).sin().powi(2),im:x.re.exp()*x.im.sin()};div(e,x)};
    mul(scaled(cexp(hi,t),t),h)
}
fn div(a:C,b:C)->C{let s=b.re*b.re+b.im*b.im;C{re:(a.re*b.re+a.im*b.im)/s,im:(a.im*b.re-a.re*b.im)/s}}
/// x -= tau^T (tau x) / |tau|^2 for a column-major m x m matrix: the
/// centre trace kept exactly (tau the centre traces of the model's
/// columns; A and dA_b conserve it).
fn keep_trace(x:&mut[f64],tau:&[f64]){
    let m=tau.len();let size:f64=tau.iter().map(|t|t*t).sum();if !(size>0.){return;}
    for j in 0..m{let s:f64=(0..m).map(|i|tau[i]*x[i+m*j]).sum::<f64>()/size;for i in 0..m{x[i+m*j]-=tau[i]*s;}}
}
/// Population readout of a model: per mode, the diagonals of its fields
/// times the Ritz vectors (n x r, column-major).
struct Rows{rows:Vec<Vec<C>>,n:usize}
impl Rows{
    /// The pipeline's model: per mode the basis' diagonals times B (k x r).
    fn new(lapack:&Lapack,execution:&Execution,basis:&[Vec<C>],b:&[C],r:usize,n:usize)->Result<Self,String>{
        let(k,d)=(basis.len(),n*n);let modes=basis[0].len()/d;
        let rows=execution.map((0..modes).collect(),|q|->Result<Vec<C>,String>{
            let diagonal:Vec<C>=(0..k).flat_map(|j|(0..n).map(move|i|basis[j][q*d+i*n+i])).collect();
            let mut out=vec![ZERO;n*r];lapack.complex_gemm().multiply(n,r,k,ONE,&diagonal,n,b,k,ZERO,&mut out,n)?;Ok(out)})
            .into_iter().collect::<Result<Vec<_>,_>>()?;
        Ok(Self{rows,n})
    }
    /// The laboratory readout of a response on the grid (one row of n per
    /// time): c(t) for every time, then per mode its rows times them.
    fn trajectory(&self,lapack:&Lapack,response:&Response,times:&[f64],nu:&[f64])->Result<Vec<Vec<f64>>,String>{
        let(n,m,count)=(self.n,response.theta.len(),times.len());let execution=Execution::new()?;
        let c:Vec<C>=execution.map(times.to_vec(),|t|response.at(t)).concat();
        let parts=execution.map((0..self.rows.len()).collect(),|q|->Result<Vec<f64>,String>{
            let mut p=vec![ZERO;n*count];lapack.complex_gemm().multiply(n,count,m,ONE,&self.rows[q],n,&c,m,ZERO,&mut p,n)?;
            Ok((0..n*count).map(|e|mul(cexp(C{re:0.,im:nu[q]},times[e/n]),p[e]).re).collect())});
        let mut out=vec![vec![0.;n];count];
        for part in parts{for(e,v)in part?.into_iter().enumerate(){out[e/n][e%n]+=v;}}
        Ok(out)
    }
    fn centre(&self,c:&[C],centre:usize)->Vec<f64>{
        let(n,r)=(self.n,c.len());let rows=&self.rows[centre];
        (0..n).map(|i|(0..r).fold(ZERO,|s,m|{let w=mul(rows[i+n*m],c[m]);C{re:s.re+w.re,im:s.im+w.im}}).re).collect()
    }
}

impl<'a> FoldCharge<'a>{
    pub fn new(setup:&EstimateSetup<'a>)->Result<Self,String>{
        let prepared=setup.prepared;let op=LatticeOperator::new(prepared,&setup.lattice.modes())?;
        let mut edges=Vec::new();
        for(t,target)in prepared.fold.targets.iter().enumerate(){for(q,mode)in op.modes.iter().enumerate(){
            let Ok(source)=charge_minus(mode,&target.charge)else{continue};if let Some(&p)=op.index.get(&source){edges.push((t,p,q));}}}
        let(n,d)=(op.n,op.n*op.n);let mut x0=vec![ZERO;op.dimension()];x0[op.centre*d..(op.centre+1)*d].copy_from_slice(&setup.initial[..n*n]);
        let x0_norm=x0.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt();
        Ok(Self{prepared,op,edges,x0,x0_norm,work:FoldWork::default()})
    }
    /// dA_b on Hermitian-symmetric fields x with their images A x.
    /// The K^(1) rest on every fold edge acts once on i nu_p x_p + (A x)_p
    /// (i nu_p x_p only where the operator leaves the rest out): its dense
    /// sector blocks by GEMM over every (field, edge) side by side. The
    /// third order acts on the fields marked `third` (`effect`).
    fn remainder(&mut self,fields:&[(&[C],&[C])],third:&[bool])->Result<Vec<Vec<C>>,String>{
        let clock=Instant::now();self.work.remainders+=fields.len();
        let(n,d,count)=(self.op.n,self.op.n*self.op.n,fields.len());let(fold,nu,edges)=(&self.prepared.fold,&self.op.nu,&self.edges);
        let(canonical,mirror)=(&self.op.canonical,&self.op.mirror);
        let dropped=self.op.dropped_k1_rest>0.;let lapack=&self.prepared.lapack;let execution=Execution::new()?;
        // The rest's sources, field after field.
        let sources:Vec<Vec<C>>=execution.map(fields.to_vec(),|(x,ax)|{let mut s=ax.to_vec();
            if dropped{for(p,w)in nu.iter().enumerate(){for(o,v)in s[p*d..(p+1)*d].iter_mut().zip(&x[p*d..(p+1)*d]){o.re-=w*v.im;o.im+=w*v.re;}}}s});
        // The absorbable parts on (A x)_p, the sparse remainders and the population rows, per field.
        let edges:Vec<(usize,usize,usize)>=edges.iter().copied().filter(|e|canonical[e.2]).collect();
        let parts=execution.map((0..count).collect(),|f|->Result<Vec<C>,String>{
            let(gemm,(x,ax),source)=(lapack.complex_gemm(),fields[f],&sources[f]);let mut y=vec![ZERO;x.len()];
            for &(t,p,q)in &edges{let target=&fold.targets[t];let yq=&mut y[q*d..(q+1)*d];
                gemm.multiply(n,n,n,ONE,&ax[p*d..(p+1)*d],n,&target.left[1],n,ONE,yq,n)?;gemm.multiply(n,n,n,ONE,&target.right[1],n,&ax[p*d..(p+1)*d],n,ONE,yq,n)?;
                target.rest[1].sparse.apply_into(ONE,&source[p*d..(p+1)*d],yq);
                if dropped{target.population_rest[1].apply_into(C{re:0.,im:-nu[p]},&x[p*d..(p+1)*d],yq);}}
            Ok(y)});
        let mut out=parts.into_iter().collect::<Result<Vec<_>,_>>()?;
        // The dense sector blocks: per target and block, every (field, edge)'s
        // source columns side by side, GEMMs over chunks of them.
        let mut by_target:BTreeMap<usize,Vec<(usize,usize)>>=BTreeMap::new();for &(t,p,q)in &edges{by_target.entry(t).or_default().push((p,q));}
        for(t,list)in &by_target{for block in &fold.targets[*t].rest[1].blocks{
            let(h,w)=(block.rows.len(),block.cols.len());let items:Vec<(usize,usize)>=(0..count).flat_map(|f|(0..list.len()).map(move|e|(f,e))).collect();
            let chunks:Vec<Vec<(usize,usize)>>=items.chunks(DENSE_COLUMNS).map(|c|c.to_vec()).collect();
            let products=execution.map(chunks.clone(),|chunk|->Result<Vec<C>,String>{
                let mut x=vec![ZERO;w*chunk.len()];
                for(c,&(f,e))in chunk.iter().enumerate(){let p=list[e].0;for(i,&col)in block.cols.iter().enumerate(){x[i+w*c]=sources[f][p*d+col as usize];}}
                let mut y=vec![ZERO;h*chunk.len()];lapack.complex_gemm().multiply(h,chunk.len(),w,ONE,&block.values,h,&x,w,ZERO,&mut y,h)?;Ok(y)});
            for(chunk,y)in chunks.iter().zip(products){let y=y?;
                for(c,&(f,e))in chunk.iter().enumerate(){let q=list[e].1;for(i,&row)in block.rows.iter().enumerate(){add(&mut out[f][q*d+row as usize],y[i+h*c]);}}}
        }}
        for y in &mut out{mirror_field(y,n,canonical,mirror);}
        self.work.milliseconds[0]+=clock.elapsed().as_secs_f64()*1e3;let clock=Instant::now();
        let marked:Vec<usize>=(0..count).filter(|&f|third[f]).collect();
        let refs:Vec<&[C]>=marked.iter().map(|&f|fields[f].0).collect();
        for(&f,y)in marked.iter().zip(fold.third_order_source(&self.op,&refs)?){for(a,b)in out[f].iter_mut().zip(y){add(a,b);}}
        self.work.milliseconds[1]+=clock.elapsed().as_secs_f64()*1e3;
        Ok(out)
    }
    /// The stored fold K (the operator's fold part) on Hermitian-symmetric fields.
    fn fold(&mut self,fields:&[&[C]])->Result<Vec<Vec<C>>,String>{
        self.work.folds+=fields.len();fields.iter().map(|x|self.op.apply_fold(x)).collect()
    }
    /// The built remainders' effect on a checkpoint's outputs and the
    /// unbuilt ones' charge. `b`: the Ritz vectors over the basis (B = X S,
    /// column-major k x r); `lambda_slow` the slowest decay rate; the pool
    /// deflated by the slow Ritz modes when an output is long.
    #[allow(clippy::too_many_arguments)]
    pub fn effect(&mut self,cp:&Checkpoint<'_>,b:&[C],lambda_slow:f64,lattice:&mut ErrorLattice<'_>,pool:&mut SlowPool,steady:&mut dyn SteadySolves,
        outputs:Outputs<'_>,ledger:&RemainderLedger)->Result<FoldEffect,String>{
        let(basis,images,theta,a)=(cp.basis,cp.images,cp.theta,cp.coefficients);
        let(k,r)=(basis.len(),theta.len());let n=self.op.n;
        let lapack=Arc::clone(&self.prepared.lapack);let execution=Execution::new()?;let packing=Packing::new(lattice);
        let length=packing.length();
        let slow:Vec<usize>=(0..r).filter(|&m|magnitude(theta[m])<slow::SLOW&&nonzero(a[m])).collect();
        let kernel=slow.iter().copied().find(|&m|magnitude(theta[m])<KERNEL);
        let long_trajectory=outputs.times.last().is_some_and(|&t|t*lambda_slow>=1.);
        let long=long_trajectory||!outputs.horizons.is_empty();
        // The trajectory's and the horizons' outputs read the low-rank
        // source and every slow Ritz vector; a steady state alone, the
        // kernel's.
        let propagates=!outputs.times.is_empty()||long;
        let mut effect=FoldEffect{trajectory:vec![vec![0.;n];outputs.times.len()],horizons:vec![vec![0.;n];outputs.horizons.len()],steady:outputs.steady.then(||vec![0.;n]),unbuilt:0.};
        // The real bases over W: U_r of B diag(a) (its columns are closed
        // under conjugation, so its Gram matrix is real) and R_s spanning
        // the slow Ritz vectors' real and imaginary parts, the kernel's first.
        let mut source:Vec<Vec<f64>>=Vec::new();
        if propagates{
            let mut gram=vec![0.;k*k];
            for m in 0..r{let v:Vec<C>=(0..k).map(|i|mul(b[i+k*m],a[m])).collect();for i in 0..k{for j in 0..k{gram[i+k*j]+=v[i].re*v[j].re+v[i].im*v[j].im;}}}
            let(values,_,vectors)=lapack.eigen(&gram,k)?;
            let largest=values.iter().fold(0_f64,|s,&v|s.max(v));
            source=(0..k).filter(|&j|values[j]>SOURCE_RANK*SOURCE_RANK*largest).map(|j|vectors[k*j..k*(j+1)].to_vec()).collect();
            orthonormalize(&mut source);
        }
        let rank=source.len();
        let mut ordered:Vec<usize>=if propagates{slow.clone()}else{kernel.into_iter().collect()};ordered.sort_by_key(|&m|Some(m)!=kernel);
        let mut spanned:Vec<Vec<f64>>=ordered.iter().flat_map(|&m|[(0..k).map(|i|b[i+k*m].re).collect(),(0..k).map(|i|b[i+k*m].im).collect()]).collect();
        orthonormalize(&mut spanned);let ns=spanned.len();
        let columns:Vec<Vec<f64>>=source.into_iter().chain(spanned).collect();
        // The fields x0, Z = W U_r, V_s = W R_s, their images and dA_b. The
        // third order acts on the slow Ritz vectors (the kernel's alone
        // without a long output): on x0 and Z its effect is a short
        // trajectory's, below 2e-6 (R2 13.6); on the responses Y it is of
        // second order.
        let(w,aw):(Vec<f64>,Vec<f64>)=(basis.iter().flat_map(|x|packing.pack(x)).collect(),images.iter().flat_map(|x|packing.pack(x)).collect());
        let combine=|packed:&[f64],c:&[Vec<f64>]|->Result<Vec<Vec<C>>,String>{let flat:Vec<f64>=c.concat();
            Ok(block_times(&lapack,&execution,packed,length,k,&flat,c.len())?.chunks_exact(length).map(|v|packing.unpack(v)).collect())};
        let mut model:Vec<[Vec<C>;2]>=vec![[self.x0.clone(),images[0].iter().map(|z|scaled(*z,self.x0_norm)).collect()]];
        for(x,ax)in combine(&w,&columns)?.into_iter().zip(combine(&aw,&columns)?){model.push([x,ax]);}
        let third:Vec<bool>=(0..model.len()).map(|j|j>rank&&(long||(j==rank+1&&kernel.is_some()))).collect();
        let pairs:Vec<(&[C],&[C])>=model.iter().map(|[x,ax]|(&x[..],&ax[..])).collect();
        let mut perturbed=self.remainder(&pairs,&third)?;
        // The slow responses Y_m to dA_b v_m = dA_b V_s (R_s^T b_m): the
        // kernel's for the steady state, every slow mode's (one of each
        // conjugate pair: the other's Y is the flip) for a long output.
        let clock=Instant::now();
        let partner=|m:usize|slow.iter().any(|&l|magnitude(C{re:theta[l].re-theta[m].re,im:theta[l].im+theta[m].im})<PAIRED*(1.+magnitude(theta[m])));
        let respond:Vec<usize>=ordered.iter().copied().filter(|&m|if long{theta[m].im>=0.||!partner(m)}else{outputs.steady&&Some(m)==kernel}).collect();
        if !respond.is_empty(){pool.add_columns(basis)?;}
        let mut responses=Vec::new();
        for &m in &respond{
            let coordinates:Vec<C>=columns[rank..].iter().map(|c|(0..k).fold(ZERO,|s,i|C{re:s.re+c[i]*b[i+k*m].re,im:s.im+c[i]*b[i+k*m].im})).collect();
            let mut f=vec![ZERO;self.x0.len()];
            for(c,dv)in coordinates.iter().zip(&perturbed[1+rank..1+rank+ns]){for(o,v)in f.iter_mut().zip(dv){mul_add(o,*c,*v);}}
            let solution=pool.response(&f,theta[m],a[m],RESPONSE_RESIDUAL,steady)?;self.work.solves+=1;
            // The steady state: the kernel's response itself, a_0 Y_0.
            if Some(m)==kernel&&outputs.steady{effect.steady=Some(pool.steady_populations(basis,&solution));}
            if long{let[ya,yb,aa,ab]=pool.fields(basis,images,&solution)?;for(y,ay)in[(ya,aa),(yb,ab)]{if y.iter().any(|z|nonzero(*z)){responses.push([y,ay]);}}}
        }
        self.work.milliseconds[2]+=clock.elapsed().as_secs_f64()*1e3;
        // A long output: the compact model on [x0, Z, V_s, Y].
        if long{
            let pairs:Vec<(&[C],&[C])>=responses.iter().map(|[y,ay]|(&y[..],&ay[..])).collect();
            perturbed.extend(self.remainder(&pairs,&vec![false;pairs.len()])?);model.extend(responses);
            let clock=Instant::now();
            let(response,rows)=Compact::new(&lapack,&execution,&packing,&model,&perturbed,&self.x0,lattice.centre,n)?.response(&lapack)?;
            if long_trajectory{effect.trajectory=rows.trajectory(&lapack,&response,outputs.times,&lattice.nu)?;}
            effect.horizons=outputs.horizons.iter().map(|&t|rows.centre(&response.at(t),lattice.centre)).collect();
            self.work.milliseconds[3]+=clock.elapsed().as_secs_f64()*1e3;
        }
        // A short trajectory: the pipeline's own model and the fast response
        // of D's part outside the space.
        if !long_trajectory&&!outputs.times.is_empty(){
            let clock=Instant::now();
            let p=rank+ns;
            // C = [U_r^T (I - R_s P_s); P_s], P_s = R_s^T B diag(1_s) B^+.
            let bplus=pseudo_inverse(&lapack,b,k,r)?;
            let mut projector=vec![0.;k*k];
            for &m in &slow{for i in 0..k{for j in 0..k{projector[i+k*j]+=mul(b[i+k*m],bplus[m+r*j]).re;}}}
            let ps:Vec<Vec<f64>>=columns[rank..].iter().map(|c|(0..k).map(|j|(0..k).map(|i|c[i]*projector[i+k*j]).sum()).collect()).collect();
            let mut cmat=vec![0.;p*k];
            for(i,u)in columns[..rank].iter().enumerate(){for j in 0..k{
                let mut v=u[j];for(s,rs)in columns[rank..].iter().enumerate(){let urs:f64=(0..k).map(|l|u[l]*rs[l]).sum();v-=urs*ps[s][j];}cmat[i+p*j]=v;}}
            for(s,row)in ps.iter().enumerate(){for j in 0..k{cmat[rank+s+p*j]=row[j];}}
            // W^T L and the out-of-space part (I - W W^T) L of L = dA_b [Z, V_s].
            let l:Vec<f64>=perturbed[1..1+p].iter().flat_map(|x|packing.pack(x)).collect();
            let wl=block_cross(&lapack,&execution,&w,k,&l,p,length)?;
            let mut mw=vec![0.;k*k];for i in 0..k{for j in 0..k{mw[i+k*j]=(0..p).map(|s|wl[i+k*s]*cmat[s+p*j]).sum();}}
            let tau:Vec<f64>=basis.iter().map(|x|centre_trace(x,lattice.centre,n)).collect();keep_trace(&mut mw,&tau);
            // dG^ = B^+ M_w B.
            let mwb:Vec<C>=(0..k*r).map(|e|{let(i,m)=(e%k,e/k);(0..k).fold(ZERO,|s,j|{let v=scaled(b[j+k*m],mw[i+k*j]);C{re:s.re+v.re,im:s.im+v.im}})}).collect();
            let mut dg=vec![ZERO;r*r];
            for x in 0..r{for y in 0..r{dg[x+r*y]=(0..k).fold(ZERO,|s,i|{let v=mul(bplus[x+r*i],mwb[i+k*y]);C{re:s.re+v.re,im:s.im+v.im}});}}
            let response=Response{theta:theta.to_vec(),a:a.to_vec(),dg};
            let rows=Rows::new(&lapack,&execution,basis,b,r,n)?;
            let inside=rows.trajectory(&lapack,&response,outputs.times,&lattice.nu)?;
            let mut outside=l;let minus:Vec<f64>=wl.iter().map(|v|-v).collect();
            for(o,v)in outside.iter_mut().zip(block_times(&lapack,&execution,&w,length,k,&minus,p)?){*o+=v;}
            let fields:Vec<Vec<C>>=outside.chunks_exact(length).map(|v|packing.unpack(v)).collect();
            let refs:Vec<&[C]>=fields.iter().map(|f|&f[..]).collect();
            let coordinates=lattice.field_coordinates(&refs)?;
            // h = C B diag(a) (p x r).
            let weights:Vec<C>=(0..p*r).map(|e|{let(s,m)=(e%p,e/p);let v=(0..k).fold(ZERO,|z,j|{let w=scaled(b[j+k*m],cmat[s+p*j]);C{re:z.re+w.re,im:z.im+w.im}});mul(v,a[m])}).collect();
            let fast=lattice.field_populations(&FieldSources{fields:&coordinates,weights:&weights,values:theta},outputs.times,Readout::Laboratory)?;
            effect.trajectory=inside.iter().zip(&fast).map(|(x,y)|x.iter().zip(y).map(|(u,v)|u+v).collect()).collect();
            self.work.milliseconds[3]+=clock.elapsed().as_secs_f64()*1e3;
        }
        // The unbuilt remainders: S_K from the steady response to the
        // stored fold on x_inf = W b_0 a_0 (one kernel solve).
        if let Some(m)=kernel.filter(|_|ledger.fold>0.&&ledger.unbuilt()>0.){
            let clock=Instant::now();
            let c:Vec<f64>=(0..k).map(|i|mul(b[i+k*m],a[m]).re).collect();
            let x=combine(&w,std::slice::from_ref(&c))?.remove(0);
            let f=self.fold(&[&x])?.remove(0);
            let solution=pool.response(&f,ZERO,ONE,RESPONSE_RESIDUAL,steady)?;self.work.solves+=1;
            let sensitivity=metric(&pool.steady_populations(basis,&solution),&vec![0.;n],outputs.groups)/ledger.fold;
            effect.unbuilt=SAFETY*sensitivity*ledger.unbuilt();
            self.work.milliseconds[2]+=clock.elapsed().as_secs_f64()*1e3;
        }
        Ok(effect)
    }
}

/// Modified Gram-Schmidt (twice) of real vectors; a vector whose part
/// outside the earlier ones is below 1e-10 of its size is dropped.
fn orthonormalize(v:&mut Vec<Vec<f64>>){
    let mut out:Vec<Vec<f64>>=Vec::new();
    for x in std::mem::take(v){
        let size=x.iter().map(|y|y*y).sum::<f64>().sqrt();let mut x=x;
        for _ in 0..2{for q in &out{let h:f64=q.iter().zip(&x).map(|(a,b)|a*b).sum();for(y,&w)in x.iter_mut().zip(q){*y-=h*w;}}}
        let eta=x.iter().map(|y|y*y).sum::<f64>().sqrt();
        if eta>1e-10*size{for y in &mut x{*y/=eta;}out.push(x);}
    }
    *v=out;
}
/// B^+ = (B^H B)^-1 B^H (r x k, column-major) for B (k x r).
fn pseudo_inverse(lapack:&Lapack,b:&[C],k:usize,r:usize)->Result<Vec<C>,String>{
    let mut bhb=vec![ZERO;r*r];for x in 0..r{for y in 0..r{bhb[x+r*y]=(0..k).fold(ZERO,|s,i|{let v=mul(conj(b[i+k*x]),b[i+k*y]);C{re:s.re+v.re,im:s.im+v.im}});}}
    let bh:Vec<C>=(0..k).flat_map(|i|(0..r).map(move|x|conj(b[i+k*x]))).collect();
    let(lu,piv)=lapack.complex_factor(&bhb,r)?;lapack.complex_solve_matrix_factored(&lu,&piv,&bh,r)
}
fn centre_trace(x:&[C],centre:usize,n:usize)->f64{let d=n*n;(0..n).map(|i|x[centre*d+i*n+i].re).sum()}

/// The Galerkin model on the span of Hermitian-symmetric fields M (in
/// order, nearly dependent ones left out): Q = M T orthonormal (Cholesky
/// QR twice), G = Q^T A Q and dG = Q^T dA Q, the coordinates of the
/// initial state, the centre traces and the population rows of Q.
struct Compact{g:Vec<f64>,dg:Vec<f64>,initial:Vec<f64>,tau:Vec<f64>,rows:Vec<Vec<C>>,n:usize}
impl Compact{
    #[allow(clippy::too_many_arguments)]
    fn new(lapack:&Lapack,execution:&Execution,packing:&Packing,model:&[[Vec<C>;2]],perturbed:&[Vec<C>],initial:&[C],centre:usize,n:usize)->Result<Self,String>{
        let(length,m,d)=(packing.length(),model.len(),n*n);
        let sizes:Vec<f64>=model.iter().map(|[x,_]|x.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()).collect();
        let scale=|j:usize|if sizes[j]>0.{1./sizes[j]}else{0.};
        let pack=|j:usize,x:&[C]|->Vec<f64>{let s=scale(j);packing.pack(x).into_iter().map(|v|v*s).collect()};
        let fields:Vec<f64>=(0..m).flat_map(|j|pack(j,&model[j][0])).collect();
        let gram=block_cross(lapack,execution,&fields,m,&fields,m,length)?;
        let(order,r1)=cholesky(&gram,m,&vec![1.;m],false);
        let mut kept=order.clone();let mut t=upper_inverse(&r1,kept.len());
        let select=|all:&[f64],kept:&[usize]|->Vec<f64>{kept.iter().flat_map(|&j|all[j*length..(j+1)*length].iter().copied()).collect()};
        // The second pass (a column it loses was dependent within the first's accuracy).
        let q=loop{
            let selected=select(&fields,&kept);
            let q1=block_times(lapack,execution,&selected,length,kept.len(),&t,kept.len())?;
            let gram=block_cross(lapack,execution,&q1,kept.len(),&q1,kept.len(),length)?;
            let(again,r2)=cholesky(&gram,kept.len(),&vec![1.;kept.len()],false);
            if again.len()==kept.len(){let inverse=upper_inverse(&r2,kept.len());t=upper_times(&t,&inverse,kept.len());break block_times(lapack,execution,&selected,length,kept.len(),&t,kept.len())?;}
            kept=again.iter().map(|&i|kept[i]).collect();let selected=select(&fields,&kept);
            let gram=block_cross(lapack,execution,&selected,kept.len(),&selected,kept.len(),length)?;
            let(_,r)=cholesky(&gram,kept.len(),&vec![1.;kept.len()],false);t=upper_inverse(&r,kept.len());
        };
        let c=kept.len();
        let images:Vec<f64>=kept.iter().flat_map(|&j|pack(j,&model[j][1])).collect();
        let changes:Vec<f64>=kept.iter().flat_map(|&j|pack(j,&perturbed[j])).collect();
        let(aq,dq)=(block_times(lapack,execution,&images,length,c,&t,c)?,block_times(lapack,execution,&changes,length,c,&t,c)?);
        let g=block_cross(lapack,execution,&q,c,&aq,c,length)?;let dg=block_cross(lapack,execution,&q,c,&dq,c,length)?;
        let initial=block_cross(lapack,execution,&q,c,&packing.pack(initial),1,length)?;
        // The centre traces and population rows of Q = M T (complex rows).
        let column=|j:usize,e:usize|->C{scaled(model[j][0][e],scale(j))};
        let traced:Vec<f64>=kept.iter().map(|&j|(0..n).map(|i|column(j,centre*d+i*n+i).re).sum()).collect();
        let tau:Vec<f64>=(0..c).map(|y|(0..c).map(|x|traced[x]*t[x+c*y]).sum()).collect();
        let modes=model[0][0].len()/d;let tc:Vec<C>=t.iter().map(|&v|C{re:v,im:0.}).collect();
        let rows=execution.map((0..modes).collect(),|q|->Result<Vec<C>,String>{
            let diagonal:Vec<C>=kept.iter().flat_map(|&j|(0..n).map(move|i|column(j,q*d+i*n+i))).collect();
            let mut out=vec![ZERO;n*c];lapack.complex_gemm().multiply(n,c,c,ONE,&diagonal,n,&tc,c,ZERO,&mut out,n)?;Ok(out)})
            .into_iter().collect::<Result<Vec<_>,_>>()?;
        Ok(Self{g,dg,initial,tau,rows,n})
    }
    /// The model's first-order response (both matrices with the centre
    /// trace kept; growing Ritz values reflected, the least one the kernel
    /// when below MODEL_KERNEL) and its rows times the Ritz vectors.
    fn response(&self,lapack:&Lapack)->Result<(Response,Rows),String>{
        let(c,n)=(self.tau.len(),self.n);
        let(mut g,mut dg)=(self.g.clone(),self.dg.clone());keep_trace(&mut g,&self.tau);keep_trace(&mut dg,&self.tau);
        let complex=|x:&[f64]|->Vec<C>{x.iter().map(|&v|C{re:v,im:0.}).collect()};
        let(mut theta,s)=lapack.complex_eigen_right(&complex(&g),c)?;
        if let Some(j)=(0..c).min_by(|&x,&y|magnitude(theta[x]).total_cmp(&magnitude(theta[y]))).filter(|&j|magnitude(theta[j])<MODEL_KERNEL){theta[j]=ZERO;}
        for z in &mut theta{if z.re>0.{z.re=-z.re;}}
        let(lu,piv)=lapack.complex_factor(&s,c)?;
        let a=lapack.complex_solve_matrix_factored(&lu,&piv,&complex(&self.initial),c)?;
        let gemm=lapack.complex_gemm();let mut product=vec![ZERO;c*c];gemm.multiply(c,c,c,ONE,&complex(&dg),c,&s,c,ZERO,&mut product,c)?;
        let dg=lapack.complex_solve_matrix_factored(&lu,&piv,&product,c)?;
        let rows=self.rows.iter().map(|q|->Result<Vec<C>,String>{let mut out=vec![ZERO;n*c];gemm.multiply(n,c,c,ONE,q,n,&s,c,ZERO,&mut out,n)?;Ok(out)})
            .collect::<Result<Vec<_>,_>>()?;
        Ok((Response{theta,a,dg},Rows{rows,n}))
    }
}
/// The product of two upper triangular k x k matrices (column-major).
fn upper_times(a:&[f64],b:&[f64],k:usize)->Vec<f64>{
    let mut out=vec![0.;k*k];for j in 0..k{for i in 0..=j{out[i+k*j]=(i..=j).map(|l|a[i+k*l]*b[l+k*j]).sum();}}out
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::super::tests::{fixture,prepared};
    use super::super::super::operator::symmetric_field;
    use super::super::super::estimate_api::LatticeSets;
    use super::super::super::frame::SwFrame;
    #[test]
    fn fold_remainder_is_the_left_out_rests_the_memory_and_the_third_order(){
        // dA_b x against its dense assembly on the fixture's second ball: on
        // every fold edge p -> q, K^(1) (A x)_p and, where the operator leaves
        // the K^(1) rests out, i nu_p (rest - population rows) x_p; plus the
        // third-order source. Hermitian-symmetric, the centre trace kept.
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let mut op=LatticeOperator::new(&p,&modes).unwrap();let(n,d)=(4,16);let size=op.dimension();
        let sets=LatticeSets{explicit:modes.clone(),ring:vec![]};let ledger=op.ledger().unwrap();
        let(times,groups)=([0.,1.],vec![vec![0]]);let frame=SwFrame::new(&p,1.);let mut initial=vec![ZERO;d];initial[0]=ONE;
        let setup=EstimateSetup{prepared:&p,lattice:&sets,outputs:Outputs{times:&times,horizons:&[],steady:true,groups:&groups},frame:&frame,ledger:&ledger,initial:&initial};
        let mut charge=FoldCharge::new(&setup).unwrap();
        let x=symmetric_field(&op,5);let mut ax=vec![ZERO;size];op.apply(&x,&mut ax).unwrap();
        let y=charge.remainder(&[(&x,&ax)],&[true]).unwrap().remove(0);
        let mut expected=p.fold.third_order_source(&op,&[&x]).unwrap().remove(0);
        let dropped=op.dropped_k1_rest>0.;
        for target in &p.fold.targets{
            let(mut k1,mut rest,mut population)=(vec![ZERO;d*d],vec![ZERO;d*d],vec![ZERO;d*d]);
            target.dense_into(n,1,ONE,&mut k1);target.rest[1].dense_into(d,ONE,&mut rest);target.population_rest[1].dense_into(d,ONE,&mut population);
            for(q,mode)in modes.iter().enumerate(){
                let Ok(source)=charge_minus(mode,&target.charge)else{continue};let Some(&s)=op.index.get(&source)else{continue};
                for r in 0..d{for c in 0..d{
                    mul_add(&mut expected[q*d+r],k1[r*d+c],ax[s*d+c]);
                    if dropped{let v=C{re:rest[r*d+c].re-population[r*d+c].re,im:rest[r*d+c].im-population[r*d+c].im};mul_add(&mut expected[q*d+r],mul(C{re:0.,im:op.nu[s]},v),x[s*d+c]);}
                }}
            }
        }
        let scale=expected.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let error=expected.iter().zip(&y).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
        assert!(scale>0.&&error<=1e-12*scale,"dA_b x {error:e} of {scale:e}");
        let trace=(0..n).fold(ZERO,|s,i|{let v=y[op.centre*d+i*n+i];C{re:s.re+v.re,im:s.im+v.im}});
        assert!(magnitude(trace)<=1e-12*scale,"centre trace {trace:?}");
        for q in 0..modes.len(){let m=op.mirror[q];for i in 0..n{for j in 0..n{
            let(a,b)=(y[m*d+j*n+i],conj(y[q*d+i*n+j]));assert!(magnitude(C{re:a.re-b.re,im:a.im-b.im})<=1e-12*scale);}}}
    }
    #[test]
    fn fold_divided_differences_are_stable_at_coincidence(){
        // (e^{a t} - e^{b t}) / (a - b), its limit t e^{a t}, and close values.
        let(a,b,t)=(C{re:-0.3,im:2.},C{re:-1.1,im:-0.5},1.7);
        let direct=div(C{re:cexp(a,t).re-cexp(b,t).re,im:cexp(a,t).im-cexp(b,t).im},C{re:a.re-b.re,im:a.im-b.im});
        let(x,y)=(phi(a,b,t),phi(b,a,t));
        for v in[x,y]{assert!(magnitude(C{re:v.re-direct.re,im:v.im-direct.im})<=1e-14*magnitude(direct));}
        let limit=scaled(cexp(a,t),t);assert!(magnitude(C{re:phi(a,a,t).re-limit.re,im:phi(a,a,t).im-limit.im})<=1e-15*magnitude(limit));
        let close=C{re:a.re-1e-9,im:a.im};let v=phi(a,close,t);assert!(magnitude(C{re:v.re-limit.re,im:v.im-limit.im})<=1e-8*magnitude(limit));
    }
}
