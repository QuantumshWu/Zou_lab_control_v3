//! Floquet frames of the lattice: the micromotion kernels, shared by the
//! preconditioner (`preconditioner`) and the error estimate.
//!
//! The separable part of the lattice operator (every left/right factor: L0's
//! and the diagonal fold's absorbable part with -i nu_q, and the left/right
//! parts of the explicit near steps) is a Kronecker sum of a left and a right
//! Floquet operator,
//!   (A f)_k = sum_t A_t f_{k-t} - i nu_k f_k,  (g B)_k = sum_t g_{k-t} B_t - i nu_k g_k.
//! Its central eigen-sequences, stacked as the kernels P_k and Q_l
//! (sum_t A_t P_{k-t} - i nu_k P_k = P_k Lambda, likewise Q with M),
//! diagonalize it on the infinite lattice: x_q = sum_{k,l} P_k c_{q-k-l} Q_l
//! turns it into the elementwise (lambda_a + mu_b - i nu_m) c_m.
//! - First order around Brillouin-Wigner self-consistent central frames: for
//!   every explicit step pair mixing (|V_s| / |nu_s|) above MIXING_FLOOR,
//!   P_s = V X_s, X_s[a, alpha] = -(V^-1 A_s P_0)[a, alpha] /
//!   (l0_a - lambda_alpha - i nu_s) (V, l0 the eigenframe of A_0 = NL'), and
//!   the effective central generator A_eff = A_0 + sum_s A_-s P_s P_0^-1 with
//!   eigenvectors P_0 and quasi-energies Lambda. Elements of X_s closer to
//!   resonance than RESONANCE_GUARD are left out; the largest kept |X_s| is
//!   the small parameter.
//! - The inverse kernels are the Neumann series of P(theta)^-1 to second
//!   order: P~_s = -P_0^-1 P_s P_0^-1, P~_0 = P_0^-1 + sum_s P_0^-1 P_s P_0^-1 P_-s P_0^-1.
//! - The support (the strength rule): a pair's kernels enter the transforms
//!   when |P_s|_F / |P_0|_F is at least KERNEL_FLOOR; the self-energy of
//!   every pair above the mixing floor stays in A_eff.
//! - For the Lindblad form the right data are the conjugates of the left
//!   (NR'^T = conj NL' + c I, B_t^T = conj A_-t; checked), so
//!   Q_l = P_-l^dagger and mu = conj(lambda) + c.
//! Without kernels (`Frames::Block`) the frames are the eigenframes of the
//! dressed block NL' X + X NR'.
//!
//! A `Transform` applies x -> sum P_k x Q_l (forward) or the inverse kernels
//! (backward) on the Hermitian-symmetric fields of a mode set: only the
//! canonical modes are computed (the caller mirrors), and every output
//! entry is accumulated by one task in a fixed order, so results do not
//! depend on the thread count.
use super::*;
use super::operator::LatticeOperator;
use crate::lapack::ComplexGemm;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Explicit step pairs mixing (|V_s| / |nu_s|) above this enter the
/// self-consistent central frames.
const MIXING_FLOOR:f64=1e-4;
/// Kernels |P_s|_F / |P_0|_F below this stay out of the transforms
/// (measured: two tiers, 0.08-0.11 for the main beat pair and 0.008-0.011
/// for the others; both pay at small and complex shifts and at the steady
/// state).
const KERNEL_FLOOR:f64=5e-3;
/// Kernel elements with |numerator / denominator| above this are near
/// resonant and left out.
const RESONANCE_GUARD:f64=0.5;
/// Relative quasi-energy change at which the self-consistency stops (it
/// stalls near 1e-9 on eigenvalue pairing noise), and its iteration cap.
const SELF_CONSISTENT:f64=1e-8;
const ITERATIONS:usize=12;
/// Frame pairs whose frequency exceeds this fraction of the frame scale
/// precess (below it: populations and degenerate coherences).
const PRECESSING:f64=1e-9;

/// Row-major n x n helpers.
pub(crate) fn transpose(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|k|a[(k%n)*n+k/n]).collect()}
pub(crate) fn adjoint(a:&[C],n:usize)->Vec<C>{transpose(a,n).into_iter().map(conj).collect()}
fn frobenius(a:&[C])->f64{a.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()}
fn one_norm(a:&[C],n:usize)->f64{(0..n).map(|j|(0..n).map(|i|magnitude(a[i*n+j])).sum::<f64>()).fold(0.,f64::max)}
/// a b (row-major n x n): the column-major (a b)^T = b^T a^T.
pub(crate) fn product(gemm:&ComplexGemm<'_>,a:&[C],b:&[C],n:usize)->Result<Vec<C>,String>{
    let mut c=vec![ZERO;n*n];gemm.multiply(n,n,n,ONE,b,n,a,n,ZERO,&mut c,n)?;Ok(c)
}
/// Inverse of a square complex matrix (any consistent major order).
pub(crate) fn inverse(lapack:&Lapack,a:&[C],n:usize)->Result<Vec<C>,String>{
    let(lu,pivots)=lapack.complex_factor(a,n)?;let mut identity=vec![ZERO;n*n];for i in 0..n{identity[i*n+i]=ONE;}
    lapack.complex_solve_matrix_factored(&lu,&pivots,&identity,n)
}
/// Z_k = A Y_k B for m consecutive row-major n x n blocks (in place); A, B
/// row-major. Row-major data are column-major transposes, so
/// Z_k^T = B^T Y_k^T A^T: one GEMM with B^T on the blocks side by side, one
/// with A^T on the blocks stacked.
pub(crate) fn sandwich_batch(gemm:&ComplexGemm<'_>,n:usize,a:&[C],b:&[C],blocks:&mut[C],m:usize,stack:&mut Vec<C>)->Result<(),String>{
    let d=n*n;grow(stack,d*m);let stack=&mut stack[..d*m];
    gemm.multiply(n,n*m,n,ONE,b,n,blocks,n,ZERO,stack,n)?;
    for k in 0..m{for j in 0..n{for i in 0..n{blocks[(i+n*k)+n*m*j]=stack[i+n*(j+n*k)];}}}
    gemm.multiply(n*m,n,n,ONE,blocks,n*m,a,n,ZERO,stack,n*m)?;
    for k in 0..m{for j in 0..n{for i in 0..n{blocks[i+n*j+d*k]=stack[(i+n*k)+n*m*j];}}}
    Ok(())
}

/// An eigenframe of a row-major n x n matrix, m = p diag(values) p^-1, with
/// the condition bound |p|_1 |p^-1|_1.
#[derive(Clone)]
struct Frame{values:Vec<C>,p:Vec<C>,pinv:Vec<C>,condition:f64}
fn eigenframe(lapack:&Lapack,m:&[C],n:usize)->Result<Frame,String>{
    // Column-major data of m^T is m's row-major data.
    let(values,vectors)=lapack.complex_eigen_right(&transpose(m,n),n)?;
    let p=transpose(&vectors,n);let pinv=inverse(lapack,&p,n)?;
    if p.iter().chain(&pinv).any(|z|!z.re.is_finite()||!z.im.is_finite()){return Err("Nonfinite Floquet frame".into());}
    let condition=one_norm(&p,n)*one_norm(&pinv,n);
    Ok(Frame{values,p,pinv,condition})
}

/// Which frames a transform uses.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Frames{
    /// The eigenframes of the dressed block (no kernels).
    Block,
    /// The self-consistent central frames with the kernels of the support.
    Floquet,
}
/// Where the kernels act: on every mode, or on the centre and its kernel
/// neighbours only (elsewhere the central frame alone).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Reach{All,Centre}

/// One frame set as the tables of the four convolution stages: the support
/// charges (0 first), the quasi-energies and the row-major kernels. Index k
/// of `forward` is support charge k; `backward` holds the plain inverse
/// P_0^-1 first (modes the kernels do not reach), then P~_k.
struct Tables{
    charges:Vec<Charge>,
    lambda:Vec<C>,mu:Vec<C>,
    /// [left, right]: P_k and Q_k = P_-k^dagger.
    forward:[Vec<Vec<C>>;2],
    /// [left, right]: P_0^-1, P~_0, P~_k and their right counterparts.
    backward:[Vec<Vec<C>>;2],
}
impl Tables{
    /// Left kernels (forward P_k and backward P~_k on `charges`) to the
    /// two-sided tables: Q_l = P_-l^dagger, mu = conj(lambda) + c.
    fn new(n:usize,charges:Vec<Charge>,lambda:Vec<C>,shift:C,forward:Vec<Vec<C>>,plain:Vec<C>,backward:Vec<Vec<C>>)->Result<Self,String>{
        let index=|q:&Charge|charges.iter().position(|k|k==q).ok_or("Floquet support is not closed under q -> -q");
        let right=|table:&[Vec<C>]|->Result<Vec<Vec<C>>,String>{charges.iter().map(|k|Ok(adjoint(&table[index(&opposite(k))?],n))).collect()};
        let(forward_right,backward_right)=(right(&forward)?,right(&backward)?);
        let mu=lambda.iter().map(|z|C{re:z.re+shift.re,im:-z.im+shift.im}).collect();
        let mut left=vec![plain.clone()];left.extend(backward);
        let mut back_right=vec![adjoint(&plain,n)];back_right.extend(backward_right);
        Ok(Self{charges,lambda,mu,forward:[forward,forward_right],backward:[left,back_right]})
    }
    fn scale(&self)->f64{self.lambda.iter().chain(&self.mu).map(|z|magnitude(*z)).fold(0.,f64::max)}
}

#[derive(Clone,Debug,Default)]
pub struct KernelDiagnostics{
    /// Build time (ms): frames, kernels and self-consistency.
    pub milliseconds:f64,
    /// Self-consistency iterations and the last relative quasi-energy change.
    pub iterations:usize,pub change:f64,
    /// Pairs in the central frames, kernel elements left out as near
    /// resonant, and the small parameter: the largest kept |X_s[a, alpha]|.
    pub pairs:usize,pub guarded:usize,pub small_parameter:f64,
    /// Condition bounds of the block and the central frames.
    pub condition:[f64;2],
    /// The support: (charge, |P_s|_F / |P_0|_F) per kept kernel; the pairs
    /// below the floor.
    pub support:Vec<(Charge,f64)>,pub below:usize,
    /// The slowest decay rate (1/us) of the dressed block's precessing pairs
    /// lambda_a + mu_b: the coherences a shift at zero cannot reach.
    pub precessing_rate:Option<f64>,
}

/// The request-level Floquet frames: the block frames, the central frames
/// with the kernels of every pair above the mixing floor (the support
/// tables use those above the kernel floor), and the effective central
/// generator.
pub struct FloquetKernels{
    n:usize,lapack:Arc<Lapack>,
    block:Tables,floquet:Tables,
    /// A_eff = P_0 Lambda P_0^-1 (row-major); B_eff^T = conj A_eff + c I.
    effective:Vec<C>,shift:C,
    /// Every pair's step charge and kernel P_s (above the mixing floor).
    steps:Vec<(Charge,Vec<C>)>,
    pub diagnostics:KernelDiagnostics,
}
impl FloquetKernels{
    pub fn new(generator:&Generator,plan:&classification::HarmonicPlan,fold:&fold::FoldLibrary,lapack:&Arc<Lapack>)->Result<Self,String>{
        let clock=Instant::now();let gemm=lapack.complex_gemm();
        let n=generator.n;let d=n*n;let zero=generator.zero_component();
        let diagonal=fold.targets.first().filter(|t|t.charge==generator.zero);
        let(mut nl,mut nr)=(zero.op.left.dense(n),zero.op.right.dense(n));
        if let Some(t)=diagonal{for k in 0..d{add(&mut nl[k],t.left[0][k]);add(&mut nr[k],t.right[0][k]);}}
        let scale=nl.iter().map(|z|magnitude(*z)).fold(0.,f64::max).max(f64::MIN_POSITIVE);
        // NR'^T = conj NL' + c I and B_s^T = conj A_-s.
        let shift=C{re:nr[0].re-nl[0].re,im:nr[0].im+nl[0].im};
        let mut defect=0_f64;
        for i in 0..n{for j in 0..n{let(a,b)=(nr[j*n+i],nl[i*n+j]);let mut e=C{re:a.re-b.re,im:a.im+b.im};if i==j{e.re-=shift.re;e.im-=shift.im;}defect=defect.max(magnitude(e));}}
        let steps:Vec<(Charge,Vec<C>,f64)>=plan.explicit().into_iter().map(|s|&plan.steps[s])
            .filter(|st|st.size>=MIXING_FLOOR*st.nu.abs()).map(|st|(st.charge.clone(),generator.components[&st.charge].op.left.dense(n),st.nu)).collect();
        for(q,_,_)in &steps{
            let back=generator.components.get(&opposite(q)).ok_or("Floquet frames: an explicit step without its opposite")?;
            let b=transpose(&generator.components[q].op.right.dense(n),n);let a=back.op.left.dense(n);
            for k in 0..d{defect=defect.max(magnitude(C{re:b[k].re-a[k].re,im:b[k].im+a[k].im}));}
        }
        if defect>1e-12*scale{return Err(format!("Floquet frames: the right data are not the conjugate of the left ({:e})",defect/scale));}
        let index=|q:&Charge|steps.iter().position(|s|&s.0==q);
        let block=eigenframe(lapack,&nl,n)?;
        // Kernels around a central frame, with the guard count and the
        // largest kept |X|.
        let kernels=|central:&Frame|->Result<(Vec<Vec<C>>,usize,f64),String>{
            let(mut guarded,mut largest)=(0,0_f64);let mut out=Vec::with_capacity(steps.len());
            for(_,a_s,nu)in &steps{
                let r=product(&gemm,&block.pinv,&product(&gemm,a_s,&central.p,n)?,n)?;let mut x=vec![ZERO;d];
                for a in 0..n{for al in 0..n{
                    let den=C{re:block.values[a].re-central.values[al].re,im:block.values[a].im-central.values[al].im-nu};let num=r[a*n+al];
                    if magnitude(num)>RESONANCE_GUARD*magnitude(den){guarded+=1;continue;}
                    let s=den.re*den.re+den.im*den.im;let v=C{re:-(num.re*den.re+num.im*den.im)/s,im:-(num.im*den.re-num.re*den.im)/s};
                    largest=largest.max(magnitude(v));x[a*n+al]=v;}}
                out.push(product(&gemm,&block.p,&x,n)?);
            }
            Ok((out,guarded,largest))
        };
        // A_eff = A_0 + sum_s A_-s P_s P_0^-1.
        let effective=|forward:&[Vec<C>],central:&Frame|->Result<Vec<C>,String>{
            let mut e=nl.clone();
            for(i,(q,_,_))in steps.iter().enumerate(){let Some(j)=index(&opposite(q))else{continue};
                let t=product(&gemm,&steps[j].1,&product(&gemm,&forward[i],&central.pinv,n)?,n)?;for(a,v)in e.iter_mut().zip(t){add(a,v);}}
            Ok(e)
        };
        // The central frame diagonalizes `diagonalized` (A_eff of the
        // previous kernels); the kernels are those of the central frame.
        let(mut central,mut diagonalized)=(block.clone(),nl.clone());let(mut forward,mut guarded,mut small)=kernels(&central)?;
        let(mut iterations,mut change)=(0,0.);
        let energy=block.values.iter().map(|z|magnitude(*z)).fold(0.,f64::max).max(1.);
        while !steps.is_empty()&&iterations<ITERATIONS{
            diagonalized=effective(&forward,&central)?;let next=eigenframe(lapack,&diagonalized,n)?;
            change=next.values.iter().map(|z|central.values.iter().map(|w|magnitude(C{re:z.re-w.re,im:z.im-w.im})).fold(f64::INFINITY,f64::min)).fold(0.,f64::max)/energy;
            central=next;iterations+=1;(forward,guarded,small)=kernels(&central)?;
            if change<SELF_CONSISTENT{break;}
        }
        // The support and the inverse kernels.
        let size0=frobenius(&central.p);let sizes:Vec<f64>=forward.iter().map(|p|frobenius(p)/size0).collect();
        let kept:Vec<usize>=(0..steps.len()).filter(|&i|sizes[i]>=KERNEL_FLOOR||index(&opposite(&steps[i].0)).is_some_and(|j|sizes[j]>=KERNEL_FLOOR)).collect();
        let pinv=&central.pinv;
        let mut charges=vec![generator.zero.clone()];let mut left=vec![central.p.clone()];let mut back=Vec::with_capacity(kept.len());
        let mut centre=pinv.clone();
        for &i in &kept{
            let t=product(&gemm,pinv,&product(&gemm,&forward[i],pinv,n)?,n)?;
            if let Some(j)=index(&opposite(&steps[i].0)).filter(|j|kept.contains(j)){
                let w=product(&gemm,&t,&product(&gemm,&forward[j],pinv,n)?,n)?;for(c,v)in centre.iter_mut().zip(w){add(c,v);}}
            back.push(t.into_iter().map(|z|C{re:-z.re,im:-z.im}).collect::<Vec<C>>());
            charges.push(steps[i].0.clone());left.push(forward[i].clone());
        }
        let mut backward=vec![centre];backward.extend(back);
        let floquet=Tables::new(n,charges,central.values.clone(),shift,left,pinv.clone(),backward)?;
        let block_tables=Tables::new(n,vec![generator.zero.clone()],block.values.clone(),shift,vec![block.p.clone()],block.pinv.clone(),vec![block.pinv.clone()])?;
        let block_scale=block_tables.scale();
        let precessing_rate=block_tables.lambda.iter().flat_map(|a|block_tables.mu.iter().map(move|b|C{re:a.re+b.re,im:a.im+b.im}))
            .filter(|z|z.im.abs()>PRECESSING*block_scale).map(|z|(-z.re).max(0.)).reduce(f64::min);
        let diagnostics=KernelDiagnostics{milliseconds:clock.elapsed().as_secs_f64()*1e3,iterations,change,pairs:steps.len(),guarded,small_parameter:small,
            condition:[block.condition,central.condition],support:kept.iter().map(|&i|(steps[i].0.clone(),sizes[i])).collect(),below:steps.len()-kept.len(),precessing_rate};
        let steps=steps.into_iter().zip(forward).map(|((q,_,_),p)|(q,p)).collect();
        Ok(Self{n,lapack:Arc::clone(lapack),block:block_tables,floquet,effective:diagonalized,shift,steps,diagnostics})
    }
    fn tables(&self,frames:Frames)->&Tables{match frames{Frames::Block=>&self.block,Frames::Floquet=>&self.floquet}}
    /// The quasi-energies (lambda, mu) of a frame set.
    pub fn quasi_energies(&self,frames:Frames)->(&[C],&[C]){let t=self.tables(frames);(&t.lambda,&t.mu)}
    /// The central kernel P_0 and its inverse (row-major) of a frame set.
    pub fn central(&self,frames:Frames)->(&[C],&[C]){let t=self.tables(frames);(&t.forward[0][0],&t.backward[0][0])}
    /// The effective central generators (A_eff, B_eff) of
    /// X -> A_eff X + X B_eff (row-major), B_eff^T = conj A_eff + c I.
    pub fn effective_generators(&self)->[Vec<C>;2]{
        let n=self.n;let mut bt:Vec<C>=self.effective.iter().map(|z|conj(*z)).collect();
        for i in 0..n{add(&mut bt[i*n+i],self.shift);}
        [self.effective.clone(),transpose(&bt,n)]
    }
    /// Every pair's first-order kernel P_s (row-major), above the mixing floor.
    pub fn step_kernels(&self)->&[(Charge,Vec<C>)]{&self.steps}
    /// The two-sided transform on an operator's Hermitian-symmetric fields:
    /// the outer stage on `computed` (the canonical modes or a subset), the
    /// kernels where `reach` says.
    pub fn transform<'k>(&'k self,op:&LatticeOperator<'_>,frames:Frames,computed:&[bool],reach:Reach)->Result<Transform<'k>,String>{
        let tables=self.tables(frames);let modes=op.modes.len();let slots=1+tables.charges.len();
        let near:BTreeSet<&Charge>=tables.charges.iter().collect();
        let active:Vec<bool>=op.modes.iter().map(|q|reach==Reach::All||near.contains(q)).collect();
        // Slot 0: the central frame alone (modes the kernels do not reach);
        // slot 1 + k: support charge k.
        let mut sources=vec![usize::MAX;modes*slots];
        for q in 0..modes{
            if !active[q]{sources[q*slots]=q;continue;}
            for(k,c)in tables.charges.iter().enumerate(){
                if let Some(&p)=charge_minus(&op.modes[q],c).ok().as_ref().and_then(|p|op.index.get(p)){sources[q*slots+1+k]=p;}}
        }
        let mut needed=vec![false;modes];
        for q in(0..modes).filter(|&q|computed[q]){for &p in &sources[q*slots..(q+1)*slots]{if p!=usize::MAX{needed[p]=true;}}}
        let inner:Vec<bool>=(0..modes).map(|q|needed[op.mirror[q]]).collect();
        let n=self.n;
        let count=|set:&[bool]|(0..modes).filter(|&q|set[q]).map(|q|sources[q*slots..(q+1)*slots].iter().filter(|&&p|p!=usize::MAX).count()).sum::<usize>()*n*n*n;
        Ok(Transform{lapack:&self.lapack,tables,n,reach,slots,work:[count(&inner),count(computed)],sources,needed,inner,computed:computed.to_vec(),
            mirror:op.mirror.clone(),execution:Execution::new()?,scratch:Vec::new(),middle:Vec::new(),flipped:Vec::new()})
    }
}

#[derive(Default)]
struct StageScratch{a:Vec<C>,b:Vec<C>}
/// The two-sided Floquet transform on one mode set's Hermitian-symmetric
/// fields. On such fields the left convolution is the mirrored right one,
/// sum_k P_k x_{p-k} = (sum_l x_{-p-l} Q_l)^dagger with Q_l = P_-l^dagger,
/// so both stages are right multiplications x_p K: per kernel one GEMM over
/// every gathered block of a task, (x K)^T = K^T x^T on the row-major
/// blocks side by side.
pub struct Transform<'k>{
    lapack:&'k Lapack,tables:&'k Tables,n:usize,
    pub reach:Reach,
    /// Per output mode and slot (0: the central frame alone, 1 + k: support
    /// charge k) the source mode, usize::MAX for none.
    slots:usize,sources:Vec<usize>,
    /// The modes the outer stage reads, the inner stage computes (their
    /// mirrors) and the outer stage computes.
    needed:Vec<bool>,inner:Vec<bool>,computed:Vec<bool>,mirror:Vec<usize>,
    /// Complex multiply-adds of the inner and the outer stage.
    work:[usize;2],
    execution:Execution,scratch:Vec<StageScratch>,middle:Vec<C>,flipped:Vec<C>,
}
impl Transform<'_>{
    /// Complex multiply-adds of one transform.
    pub fn work(&self)->usize{self.work[0]+self.work[1]}
    /// Whether two transforms apply the same kernels on the same modes.
    pub fn same(&self,other:&Transform<'_>)->bool{self.sources==other.sources}
    /// c = sum P~_k x Q~_l on the computed modes (frame coordinates; the
    /// caller mirrors).
    pub fn backward(&mut self,x:&[C],c:&mut[C])->Result<(),String>{self.run(true,x,c)}
    /// x = sum P_k c Q_l on the computed modes (the caller mirrors).
    pub fn forward(&mut self,c:&[C],x:&mut[C])->Result<(),String>{self.run(false,c,x)}
    fn run(&mut self,backward:bool,input:&[C],out:&mut[C])->Result<(),String>{
        let(n,d)=(self.n,self.n*self.n);let t=self.tables;let length=input.len();
        // Backward: [Q0^-1 plain, Q~_0, Q~_k]; forward: [Q_0, Q_0, Q_k].
        let table:Vec<&[C]>=(0..self.slots).map(|s|if backward{t.backward[1][s].as_slice()}else{t.forward[1][s.saturating_sub(1)].as_slice()}).collect();
        grow(&mut self.middle,length);grow(&mut self.flipped,length);
        stage(&self.execution,self.lapack,&mut self.scratch,n,&self.sources,self.slots,&table,input,&mut self.middle[..length],&self.inner,self.work[0])?;
        // The left stage from the right one: u_p = (v_-p)^dagger.
        let(middle,needed,mirror)=(&self.middle,&self.needed,&self.mirror);
        self.execution.for_mode_chunks(&mut self.flipped[..length],d,needed.len()*d,MINIMUM_WORK_PER_TASK/8,&mut Vec::<()>::new(),|first,chunk,_|{
            for(k,block)in chunk.chunks_exact_mut(d).enumerate(){let p=first+k;if !needed[p]{continue;}let m=mirror[p];
                for i in 0..n{for j in 0..n{block[i*n+j]=conj(middle[m*d+j*n+i]);}}}
            Ok(())})?;
        stage(&self.execution,self.lapack,&mut self.scratch,n,&self.sources,self.slots,&table,&self.flipped[..length],out,&self.computed,self.work[1])
    }
}
/// out_q = sum over q's slots s of in_{source(q, s)} K_s on the modes
/// `only` (zeroed first): per task and slot one GEMM over the gathered
/// blocks, tasks of about one worker's share. Each output block is
/// accumulated by one task in slot order.
#[allow(clippy::too_many_arguments)]
fn stage(execution:&Execution,lapack:&Lapack,scratch:&mut Vec<StageScratch>,n:usize,sources:&[usize],slots:usize,table:&[&[C]],input:&[C],out:&mut[C],only:&[bool],work:usize)->Result<(),String>{
    let d=n*n;let minimum=work.div_ceil(execution.threads()).max(MINIMUM_WORK_PER_TASK/8);
    execution.for_mode_chunks(out,d,work,minimum,scratch,|first,chunk,scratch|{
        let gemm=lapack.complex_gemm();let count=chunk.len()/d;
        grow(&mut scratch.a,count*d);grow(&mut scratch.b,count*d);
        for(k,block)in chunk.chunks_exact_mut(d).enumerate(){if only[first+k]{block.fill(ZERO);}}
        let mut targets=Vec::with_capacity(count);
        for s in 0..slots{
            targets.clear();
            for k in 0..count{let q=first+k;if !only[q]{continue;}let p=sources[q*slots+s];if p==usize::MAX{continue;}
                let i=targets.len();scratch.a[i*d..(i+1)*d].copy_from_slice(&input[p*d..(p+1)*d]);targets.push(k);}
            let m=targets.len();if m==0{continue;}
            gemm.multiply(n,n*m,n,ONE,table[s],n,&scratch.a[..m*d],n,ZERO,&mut scratch.b[..m*d],n)?;
            for(i,&k)in targets.iter().enumerate(){for(o,v)in chunk[k*d..(k+1)*d].iter_mut().zip(&scratch.b[i*d..(i+1)*d]){add(o,*v);}}
        }
        Ok(())
    })
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::operator::symmetric_field;
    #[test]
    fn lattice_floquet_kernels_solve_their_frame_equations(){
        // The fixture's near pair: P_0 diagonalizes A_eff = A_0 + sum A_-s P_s
        // P_0^-1, the right side is the conjugate of the left, the kernels
        // solve the first-order equations, and the block transforms are exact
        // inverses of each other.
        let f=fixture();let p=prepared(&f);let k=&p.floquet;let n=4;let gemm=p.lapack.complex_gemm();
        assert!(!k.step_kernels().is_empty()&&!k.diagnostics.support.is_empty());
        let(lambda,_)=k.quasi_energies(Frames::Floquet);let(p0,p0inv)=k.central(Frames::Floquet);
        let[a_eff,b_eff]=k.effective_generators();
        // P_0 diagonalizes A_eff, and P_0^-1 is its inverse.
        let ap=product(&gemm,&a_eff,p0,n).unwrap();let scale=a_eff.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        for i in 0..n{for a in 0..n{let e=C{re:ap[i*n+a].re-(p0[i*n+a].re*lambda[a].re-p0[i*n+a].im*lambda[a].im),im:ap[i*n+a].im-(p0[i*n+a].re*lambda[a].im+p0[i*n+a].im*lambda[a].re)};
            assert!(magnitude(e)<1e-10*scale);}}
        let id=product(&gemm,p0,p0inv,n).unwrap();for i in 0..n{for j in 0..n{assert!(magnitude(C{re:id[i*n+j].re-f64::from(u8::from(i==j)),im:id[i*n+j].im})<1e-12);}}
        // B_eff^T = conj A_eff + c I.
        let bt=transpose(&b_eff,n);let c=C{re:bt[0].re-a_eff[0].re,im:bt[0].im+a_eff[0].im};
        for i in 0..n{for j in 0..n{let mut e=C{re:bt[i*n+j].re-a_eff[i*n+j].re,im:bt[i*n+j].im+a_eff[i*n+j].im};if i==j{e.re-=c.re;e.im-=c.im;}assert!(magnitude(e)<1e-12*scale);}}
        // First order: (A_0 - i nu_s) P_s - P_s Lambda = -A_s P_0 exactly (A_0
        // = NL', the block frame's generator), unless elements were guarded.
        let n0=p.generator.zero_component().op.left.dense(n);let t=&p.fold.targets[0];
        let a0:Vec<C>=n0.iter().zip(&t.left[0]).map(|(x,y)|C{re:x.re+y.re,im:x.im+y.im}).collect();
        if k.diagnostics.guarded==0{for(q,ps)in k.step_kernels(){
            let st=p.plan.steps.iter().find(|s|&s.charge==q).unwrap();let a_s=p.generator.components[q].op.left.dense(n);
            let lhs=product(&gemm,&a0,ps,n).unwrap();let ap0=product(&gemm,&a_s,p0,n).unwrap();
            let size=ap0.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for i in 0..n{for a in 0..n{let v=ps[i*n+a];let mut e=lhs[i*n+a];e.re+=st.nu*v.im;e.im-=st.nu*v.re;
                e.re-=v.re*lambda[a].re-v.im*lambda[a].im;e.im-=v.re*lambda[a].im+v.im*lambda[a].re;add(&mut e,ap0[i*n+a]);
                assert!(magnitude(e)<=1e-10*size,"step {q:?}: {e:?}");}}
        }}
        assert!(k.diagnostics.small_parameter>0.&&k.diagnostics.small_parameter<RESONANCE_GUARD);
        let modes=p.modes(2).unwrap();let op=LatticeOperator::new(&p,&modes).unwrap();
        let x=symmetric_field(&op,4);let(mut c,mut y)=(vec![ZERO;x.len()],vec![ZERO;x.len()]);
        // The block frames: an exact round trip.
        let mut b=k.transform(&op,Frames::Block,&op.canonical,Reach::All).unwrap();
        b.backward(&x,&mut c).unwrap();mirror_field(&mut c,n,&op.canonical,&op.mirror);b.forward(&c,&mut y).unwrap();mirror_field(&mut y,n,&op.canonical,&op.mirror);
        let e=x.iter().zip(&y).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);assert!(e<1e-12,"block round trip {e:e}");
    }
}
