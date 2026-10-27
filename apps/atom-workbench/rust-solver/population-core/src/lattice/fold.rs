//! The static second-order fold. Every far step s is eliminated with the
//! exact inverse D_s^-1 = (L0 - i nu_s)^-1 of its far block:
//!   K_t^(0) = -sum_{s + s' = t} C_s' D_s^-1 C_s,  K_t^(1) = -sum C_s' D_s^-2 C_s,
//! so the far shell seen from a near source q acts as K_t^(0) + i nu_q K_t^(1)
//! on the edge q -> q + t: the expansion of (D_s - i nu_q)^-1 in nu_q, whose
//! remainder is `ledger::RemainderLedger::nu_freeze`. t = 0 is the diagonal
//! dressing (light shifts and scattering); t != 0 are the cross folds of far
//! steps whose sum is a near charge.
//!
//! D_s^-1 is structured, never a d x d factorization: L0 X = NL X + X NR +
//! R0(X) with NL, NR the n x n left/right factors, inverted in their
//! eigenframes (Schur frames when the eigenvectors are ill-conditioned), and
//! R0 the charge-0 recycling, a Woodbury term of rank (recycled output rows).
//! Each shift is checked against the sparse L0 and refined once when its
//! residual is above roundoff. The shift factors stay with the library: the
//! third-order source (the near coupling inside the far shell, charged, not
//! folded) applies them on a lattice field.
//!
//! Storage: each K_t^(o) is its absorbable part A X + X B (n x n left/right
//! factors, the least-squares projection) plus the rest, stored by sectors
//! (GG, GE, EG, EE of the output and input elements): a sector pair that is
//! filled is a dense block applied by GEMM, the scattered entries of the
//! others a CSR remainder above ENTRY_FLOOR times the largest entry.
//! Population rows are never dropped, so the stored fold stays exactly trace
//! preserving; the dropped entries' largest row sum is charged.
use super::*;
use super::classification::HarmonicPlan;
use super::operator::LatticeOperator;
use crate::lapack::{ComplexGemm,SylvesterKernel};
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Eigenframes whose condition product exceeds this use Schur frames.
const EIGEN_CONDITION_LIMIT:f64=1e4;
/// Relative residual of a structured solve above which it is refined once.
const REFINE_RESIDUAL:f64=1e-14;
/// Sparse-remainder entries below this fraction of a target's largest entry
/// are dropped (and charged). The dark states amplify dropped coherence
/// entries: at 1e-9 probe + tweezer populations move by 1e-4, at 1e-12 by
/// 3e-8. Dense sector blocks drop nothing.
const ENTRY_FLOOR:f64=1e-12;
/// A sector pair with at least this share of its entries above the floor is
/// stored as a dense block.
const DENSE_FILL:f64=0.2;
const MINUS_ONE:C=C{re:-1.,im:0.};

/// Row-major sparse d x d matrix.
#[derive(Clone,Debug,Default)]
pub struct Csr{pub row:Vec<u32>,pub col:Vec<u32>,pub value:Vec<C>}
impl Csr{
    pub fn nonzeros(&self)->usize{self.value.len()}
    /// y += s A x.
    pub fn apply_into(&self,s:C,x:&[C],y:&mut[C]){
        for r in 0..self.row.len().saturating_sub(1){
            let mut acc=ZERO;
            for at in self.row[r]as usize..self.row[r+1]as usize{mul_add(&mut acc,self.value[at],x[self.col[at]as usize]);}
            mul_add(&mut y[r],s,acc);
        }
    }
    pub fn dense_into(&self,d:usize,s:C,out:&mut[C]){
        for r in 0..d{for at in self.row[r]as usize..self.row[r+1]as usize{add(&mut out[r*d+self.col[at]as usize],mul(s,self.value[at]));}}
    }
}

/// A dense sector block: rows x cols of the d x d matrix (column-major).
pub struct DenseSector{pub rows:Vec<u32>,pub cols:Vec<u32>,pub values:Vec<C>}
/// A fold rest: dense sector blocks plus a sparse remainder.
#[derive(Default)]
pub struct Rest{pub blocks:Vec<DenseSector>,pub sparse:Csr}
impl Rest{
    /// Stored entries (multiply-adds per source block).
    pub fn stored(&self)->usize{self.blocks.iter().map(|b|b.values.len()).sum::<usize>()+self.sparse.nonzeros()}
    pub fn dense_into(&self,d:usize,s:C,out:&mut[C]){
        for b in &self.blocks{for(c,&col)in b.cols.iter().enumerate(){for(r,&row)in b.rows.iter().enumerate(){
            add(&mut out[row as usize*d+col as usize],mul(s,b.values[r+b.rows.len()*c]));}}}
        self.sparse.dense_into(d,s,out);
    }
}
/// One stored fold target: K_t^(o) X = left[o] X + X right[o] + rest[o] X.
pub struct StoredFold{
    pub charge:Charge,pub nu:f64,
    pub left:[Vec<C>;2],pub right:[Vec<C>;2],pub rest:[Rest;2],
    /// The rest's population rows alone (sparse): what keeps a rest exactly
    /// trace preserving when the rest is otherwise left out.
    pub population_rest:[Rest;2],
    /// Largest row sum of the dropped rest entries.
    pub dropped:[f64;2],
    /// Frobenius norms: whole K, absorbed part, rest.
    pub norms:[[f64;3];2],
    pub largest_entry:[f64;2],
    /// Largest absolute row sum of the stored rest (a generator-norm bound).
    pub rest_row_sum:[f64;2],
}
impl StoredFold{
    /// out += s K^(o) as a row-major d x d matrix.
    pub fn dense_into(&self,n:usize,order:usize,s:C,out:&mut[C]){
        let op=Superop{left:Sparse::from_dense(&self.left[order],n),right:Sparse::from_dense(&self.right[order],n),sandwich:vec![]};
        op.dense_into(n,s,out);self.rest[order].dense_into(n*n,s,out);
    }
}
#[derive(Clone,Debug,Default)]
pub struct FoldDiagnostics{
    pub far_sources:usize,pub columns:usize,pub backend:&'static str,
    pub frame_condition:f64,pub woodbury_rank:usize,
    /// Largest relative residual of a structured solve before / after refinement.
    pub solve_residual:f64,pub refined_residual:f64,pub refined:usize,
    pub milliseconds:f64,
    /// Milliseconds: the far solves (factors and columns), the storage.
    pub phase_ms:[f64;2],
}
pub struct FoldLibrary{
    pub targets:Vec<StoredFold>,
    /// K_0^(0) as a row-major d x d matrix, for the dense dressed block and
    /// the static-shift test hook.
    pub zero_dense:Option<Vec<C>>,
    /// The structured (L0 - z)^-1 and the shift factor of every far step.
    solver:Option<ShiftedL0>,factors:BTreeMap<usize,ShiftFactor>,
    pub diagnostics:FoldDiagnostics,
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
pub(crate) fn transpose(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|k|a[(k%n)*n+k/n]).collect()}
fn adjoint(a:&[C],n:usize)->Vec<C>{transpose(a,n).into_iter().map(conj).collect()}
fn frobenius(a:&[C])->f64{a.iter().map(|z|z.re*z.re+z.im*z.im).sum::<f64>().sqrt()}
#[inline]fn squared(z:C)->f64{z.re*z.re+z.im*z.im}
fn one_norm(a:&[C],n:usize)->f64{(0..n).map(|j|(0..n).map(|i|magnitude(a[i*n+j])).sum::<f64>()).fold(0.,f64::max)}
/// Inverse of a square complex matrix (any consistent major order).
pub(crate) fn inverse(lapack:&Lapack,a:&[C],n:usize)->Result<Vec<C>,String>{
    let(lu,pivots)=lapack.complex_factor(a,n)?;let mut identity=vec![ZERO;n*n];for i in 0..n{identity[i*n+i]=ONE;}
    lapack.complex_solve_matrix_factored(&lu,&pivots,&identity,n)
}

/// Eigenframe of a row-major n x n matrix m = P diag(values) P^-1: the
/// values, row-major P and P^-1, a condition bound |P|_1 |P^-1|_1 and the
/// relative reconstruction error.
pub(crate) fn eigen_frame(lapack:&Lapack,m:&[C],n:usize)->Result<(Vec<C>,Vec<C>,Vec<C>,f64,f64),String>{
    // Column-major data of m^T is m's row-major data; we need m itself.
    let(values,vectors)=lapack.complex_eigen_right(&transpose(m,n),n)?;
    let p=transpose(&vectors,n);let pinv=inverse(lapack,&p,n)?;
    let condition=one_norm(&p,n)*one_norm(&pinv,n);
    let mut recon=vec![ZERO;n*n];
    for i in 0..n{for j in 0..n{let mut s=ZERO;for k in 0..n{mul_add(&mut s,mul(p[i*n+k],values[k]),pinv[k*n+j]);}recon[i*n+j]=C{re:s.re-m[i*n+j].re,im:s.im-m[i*n+j].im};}}
    Ok((values,p,pinv,condition,frobenius(&recon)/frobenius(m).max(f64::MIN_POSITIVE)))
}
enum Reduced{Eigen{lambda:Vec<C>,mu:Vec<C>},Schur{ta:Vec<C>,tb:Vec<C>}}
/// The structured inverse of L0 - z (see the module documentation).
pub struct ShiftedL0{
    n:usize,
    /// Row-major frame transforms: hat(Y) = left_in Y right_in,
    /// X = left_out hat(X) right_out.
    left_in:Vec<C>,right_in:Vec<C>,left_out:Vec<C>,right_out:Vec<C>,
    reduced:Reduced,
    l0:Superop,
    /// Output positions of R0 and its rows there (rank x d, column-major).
    positions:Vec<usize>,v:Vec<C>,
    pub backend:&'static str,pub condition:f64,
}
enum ReducedShift{Eigen(Vec<C>),Schur(Vec<C>)}
pub struct ShiftFactor{shift:C,reduced:ReducedShift,xk:Vec<C>,capacitance:Vec<C>,pub residual:f64,pub refine:bool}
#[derive(Default)]
pub struct SolveScratch{stack:Vec<C>,column:Vec<C>,small:Vec<C>,w:Vec<C>,c:Vec<C>}

impl ShiftedL0{
    pub fn new(generator:&Generator,lapack:&Lapack)->Result<Self,String>{
        let n=generator.n;let d=n*n;let zero=generator.zero_component();
        let nl=zero.op.left.dense(n);let nr=zero.op.right.dense(n);
        let r0=Superop{left:Sparse::default(),right:Sparse::default(),sandwich:zero.op.sandwich.clone()};
        let rows:BTreeSet<usize>=r0.sandwich.iter().flat_map(|(p,_)|p.entries.iter().map(|e|e.0)).collect();
        let cols:BTreeSet<usize>=r0.sandwich.iter().flat_map(|(_,q)|q.entries.iter().map(|e|e.1)).collect();
        let positions:Vec<usize>=rows.iter().flat_map(|&i|cols.iter().map(move|&j|i*n+j)).collect();
        let rank=positions.len();
        let mut dense=vec![ZERO;d*d];r0.dense_into(n,ONE,&mut dense);
        let mut v=vec![ZERO;rank*d];for(k,&p)in positions.iter().enumerate(){for c in 0..d{v[k+rank*c]=dense[p*d+c];}}
        let eigen=|m:&[C]|eigen_frame(lapack,m,n);
        let frames=match(eigen(&nl),eigen(&nr)){
            (Ok((lambda,p,pinv,ca,ra)),Ok((mu,q,qinv,cb,rb)))if ca*cb<=EIGEN_CONDITION_LIMIT&&ra.max(rb)<=1e-12=>
                Some((pinv,q,p,qinv,Reduced::Eigen{lambda,mu},ca*cb)),
            _=>None,
        };
        let(left_in,right_in,left_out,right_out,reduced,condition,backend)=match frames{
            Some((a,b,c,e,r,k))=>(a,b,c,e,r,k,"eigen"),
            None=>{
                // NL = Qa Ta Qa^H; NR^H = Qb Tb Qb^H so that NR = Qb Tb^H Qb^H.
                let(ta,qa)=lapack.complex_schur(&transpose(&nl,n),n)?;let(tb,qb)=lapack.complex_schur(&transpose(&adjoint(&nr,n),n),n)?;
                let qa=transpose(&qa,n);let qb=transpose(&qb,n);
                (adjoint(&qa,n),qb.clone(),qa,adjoint(&qb,n),Reduced::Schur{ta,tb},1.,"schur")
            }
        };
        Ok(Self{n,left_in,right_in,left_out,right_out,reduced,l0:zero.op.clone(),positions,v,backend,condition})
    }
    pub fn rank(&self)->usize{self.positions.len()}
    /// Shift-dependent factors of (L0 - shift)^-1, checked on two blocks.
    pub fn factor(&self,shift:C,lapack:&Lapack)->Result<ShiftFactor,String>{
        let n=self.n;let d=n*n;let rank=self.rank();
        let reduced=match &self.reduced{
            Reduced::Eigen{lambda,mu}=>{let mut r=vec![ZERO;d];
                for a in 0..n{for b in 0..n{let z=C{re:lambda[a].re+mu[b].re-shift.re,im:lambda[a].im+mu[b].im-shift.im};let s=z.re*z.re+z.im*z.im;
                    if !(s>0.&&s.is_finite()){return Err("Unresolved shifted L0 denominator".into());}r[a*n+b]=C{re:z.re/s,im:-z.im/s};}}
                ReducedShift::Eigen(r)},
            Reduced::Schur{ta,..}=>{let mut t=ta.clone();for i in 0..n{t[i+n*i].re-=shift.re;t[i+n*i].im-=shift.im;}ReducedShift::Schur(t)},
        };
        let mut factor=ShiftFactor{shift,reduced,xk:vec![],capacitance:vec![],residual:0.,refine:false};
        let mut scratch=SolveScratch::default();let gemm=lapack.complex_gemm();let sylvester=lapack.sylvester_kernel();
        if rank>0{
            let mut xk=vec![ZERO;d*rank];for(k,&p)in self.positions.iter().enumerate(){xk[k*d+p]=ONE;}
            self.solve_plain(&factor,&gemm,&sylvester,&mut xk,rank,&mut scratch)?;
            let mut g=vec![ZERO;rank*rank];for k in 0..rank{g[k+rank*k]=ONE;}
            gemm.multiply(rank,rank,d,ONE,&self.v,rank,&xk,d,ONE,&mut g,rank)?;
            factor.capacitance=inverse(lapack,&g,rank)?;factor.xk=xk;
        }
        let y:Vec<C>=(0..2*d).map(|k|C{re:((k*37+11)%97)as f64/97.-0.5,im:((k*53+7)%89)as f64/89.-0.5}).collect();
        let mut x=y.clone();self.solve_with(&factor,&gemm,&sylvester,&mut x,2,&mut scratch,false)?;
        factor.residual=self.relative_residual(&factor,&x,&y,2,&mut scratch);
        factor.refine=factor.residual>REFINE_RESIDUAL;
        if !factor.residual.is_finite()||factor.residual>1e-8{return Err(format!("Structured (L0 - z)^-1 unresolved: relative residual {:e}",factor.residual));}
        Ok(factor)
    }
    /// |Y - (L0 - shift) X| / |Y| over m blocks.
    fn relative_residual(&self,factor:&ShiftFactor,x:&[C],y:&[C],m:usize,scratch:&mut SolveScratch)->f64{
        let d=self.n*self.n;let mut r=y.to_vec();
        for k in 0..m{let(xs,rs)=(&x[k*d..(k+1)*d],&mut r[k*d..(k+1)*d]);self.l0.apply_into(self.n,MINUS_ONE,xs,rs,&mut scratch.small);
            for(a,&b)in rs.iter_mut().zip(xs){mul_add(a,factor.shift,b);}}
        frobenius(&r)/frobenius(y).max(f64::MIN_POSITIVE)
    }
    fn solve_plain(&self,factor:&ShiftFactor,gemm:&ComplexGemm<'_>,sylvester:&SylvesterKernel<'_>,blocks:&mut[C],m:usize,scratch:&mut SolveScratch)->Result<(),String>{
        let n=self.n;let d=n*n;
        sandwich_batch(gemm,n,&self.left_in,&self.right_in,blocks,m,&mut scratch.stack)?;
        match &factor.reduced{
            ReducedShift::Eigen(r)=>for k in 0..m{for(x,&s)in blocks[k*d..(k+1)*d].iter_mut().zip(r){*x=mul(*x,s);}},
            ReducedShift::Schur(t)=>{let Reduced::Schur{tb,..}=&self.reduced else{return Err("Schur shift factor on an eigen frame".into())};
                let minus:Vec<C>=tb.iter().map(|z|C{re:-z.re,im:-z.im}).collect();grow(&mut scratch.column,d);
                for k in 0..m{let block=&mut blocks[k*d..(k+1)*d];for i in 0..n{for j in 0..n{scratch.column[i+n*j]=block[i*n+j];}}
                    let solved=sylvester.complex_sylvester(t,&minus,&scratch.column[..d],n)?;
                    for i in 0..n{for j in 0..n{block[i*n+j]=solved[i+n*j];}}}}
        }
        sandwich_batch(gemm,n,&self.left_out,&self.right_out,blocks,m,&mut scratch.stack)
    }
    fn solve_with(&self,factor:&ShiftFactor,gemm:&ComplexGemm<'_>,sylvester:&SylvesterKernel<'_>,blocks:&mut[C],m:usize,scratch:&mut SolveScratch,refine:bool)->Result<(),String>{
        let d=self.n*self.n;let rank=self.rank();
        let original=if refine{Some(blocks.to_vec())}else{None};
        self.solve_plain(factor,gemm,sylvester,blocks,m,scratch)?;
        if rank>0{
            grow(&mut scratch.w,rank*m);grow(&mut scratch.c,rank*m);
            gemm.multiply(rank,m,d,ONE,&self.v,rank,blocks,d,ZERO,&mut scratch.w,rank)?;
            gemm.multiply(rank,m,rank,ONE,&factor.capacitance,rank,&scratch.w,rank,ZERO,&mut scratch.c,rank)?;
            gemm.multiply(d,m,rank,MINUS_ONE,&factor.xk,d,&scratch.c,rank,ONE,blocks,d)?;
        }
        if let Some(y)=original{
            // One step of iterative refinement: X += D^-1 (Y - D X).
            let mut r=y;
            for k in 0..m{let(xs,rs)=(&blocks[k*d..(k+1)*d],&mut r[k*d..(k+1)*d]);self.l0.apply_into(self.n,MINUS_ONE,xs,rs,&mut scratch.small);
                for(a,&b)in rs.iter_mut().zip(xs){mul_add(a,factor.shift,b);}}
            self.solve_with(factor,gemm,sylvester,&mut r,m,scratch,false)?;
            for(x,c)in blocks.iter_mut().zip(r){add(x,c);}
        }
        if blocks.iter().any(|z|!z.re.is_finite()||!z.im.is_finite()){return Err("Nonfinite structured (L0 - z)^-1 action".into());}
        Ok(())
    }
    /// (L0 - shift)^-1 on m consecutive row-major blocks, in place.
    pub fn solve(&self,factor:&ShiftFactor,gemm:&ComplexGemm<'_>,sylvester:&SylvesterKernel<'_>,blocks:&mut[C],m:usize,scratch:&mut SolveScratch)->Result<(),String>{
        self.solve_with(factor,gemm,sylvester,blocks,m,scratch,factor.refine)
    }
    /// Relative residual of the (refined) solve on two test blocks.
    pub fn refined_residual(&self,factor:&ShiftFactor,lapack:&Lapack)->Result<f64,String>{
        let d=self.n*self.n;let mut scratch=SolveScratch::default();
        let y:Vec<C>=(0..2*d).map(|k|C{re:((k*31+5)%83)as f64/83.-0.5,im:((k*19+3)%79)as f64/79.-0.5}).collect();
        let mut x=y.clone();self.solve(factor,&lapack.complex_gemm(),&lapack.sylvester_kernel(),&mut x,2,&mut scratch)?;
        Ok(self.relative_residual(factor,&x,&y,2,&mut scratch))
    }
}

/// Whether input coordinate c = (k, l) has a nonzero image under `op`
/// (structurally; an exactly cancelling image just folds to zero).
fn column_active(op:&Superop,n:usize,c:usize)->bool{
    let(k,l)=(c/n,c%n);
    op.left.entries.iter().any(|e|e.1==k)||op.right.entries.iter().any(|e|e.0==l)
        ||op.sandwich.iter().any(|(p,q)|p.entries.iter().any(|e|e.1==k)&&q.entries.iter().any(|e|e.0==l))
}

#[derive(Default)]
struct BuildScratch{first:Vec<C>,second:Vec<C>,unit:Vec<C>,small:Vec<C>,solve:SolveScratch}

/// The absorbable part of a row-major d x d K: the least-squares projection
/// onto X -> A X + X B (row-major n x n A, B; the identity split between them
/// by the trace convention A_ii -= tr K / n^2).
pub(crate) fn absorbable(k:&[C],n:usize)->(Vec<C>,Vec<C>){
    let d=n*n;let scale=1./n as f64;let(mut a,mut b)=(vec![ZERO;n*n],vec![ZERO;n*n]);let mut trace=ZERO;
    for i in 0..n{for l in 0..n{add(&mut trace,k[(i*n+l)*d+i*n+l]);}}
    for i in 0..n{for kk in 0..n{let mut s=ZERO;for l in 0..n{add(&mut s,k[(i*n+l)*d+kk*n+l]);}a[i*n+kk]=scaled(s,scale);}}
    for i in 0..n{a[i*n+i]=C{re:a[i*n+i].re-trace.re*scale*scale,im:a[i*n+i].im-trace.im*scale*scale};}
    for l in 0..n{for j in 0..n{let mut s=ZERO;for i in 0..n{add(&mut s,k[(i*n+j)*d+i*n+l]);}b[l*n+j]=scaled(s,scale);}}
    (a,b)
}

/// Splits a row-major d x d K into its absorbable part and the rest, stored
/// by sectors (see the module documentation). Population rows (i, i) are
/// never dropped, so the stored K keeps the fold's exact trace preservation
/// (v^T K = 0); the largest dropped row sum is returned.
fn store(k:&mut[C],n:usize,ground:&[bool])->(Vec<C>,Vec<C>,Rest,f64,[f64;3],f64,f64,Rest){
    let d=n*n;let largest=k.iter().map(|z|squared(*z)).fold(0.,f64::max).sqrt();let total=frobenius(k);
    let(a,b)=absorbable(k,n);
    let op=Superop{left:Sparse::from_dense(&a,n),right:Sparse::from_dense(&b,n),sandwich:vec![]};
    let mut absorbed=vec![ZERO;d*d];op.dense_into(n,ONE,&mut absorbed);
    let norms=[total,frobenius(&absorbed),{for(x,y)in k.iter_mut().zip(&absorbed){x.re-=y.re;x.im-=y.im;}frobenius(k)}];
    let row_sum=(0..d).map(|r|k[r*d..(r+1)*d].iter().map(|z|squared(*z).sqrt()).sum::<f64>()).fold(0.,f64::max);
    let mut populations=Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]};populations.row.push(0);
    for r in 0..d{if r/n==r%n{for c in 0..d{let v=k[r*d+c];if nonzero(v){populations.col.push(c as u32);populations.value.push(v);}}}
        populations.row.push(populations.value.len()as u32);}
    let floor=ENTRY_FLOOR*largest;let floor2=floor*floor;
    let sector=|p:usize|2*usize::from(!ground[p/n])+usize::from(!ground[p%n]);
    let members:Vec<Vec<usize>>=(0..4).map(|s|(0..d).filter(|&p|sector(p)==s).collect()).collect();
    let mut dense=[[false;4];4];let mut blocks=Vec::new();
    for so in 0..4{for si in 0..4{
        let(rows,cols)=(&members[so],&members[si]);if rows.is_empty()||cols.is_empty(){continue;}
        let filled=rows.iter().map(|&r|cols.iter().filter(|&&c|squared(k[r*d+c])>=floor2&&floor>0.).count()).sum::<usize>();
        if (filled as f64)<DENSE_FILL*(rows.len()*cols.len())as f64{continue;}
        dense[so][si]=true;
        let mut values=vec![ZERO;rows.len()*cols.len()];
        for(c,&col)in cols.iter().enumerate(){for(r,&row)in rows.iter().enumerate(){values[r+rows.len()*c]=k[row*d+col];}}
        blocks.push(DenseSector{rows:rows.iter().map(|&r|r as u32).collect(),cols:cols.iter().map(|&c|c as u32).collect(),values});
    }}
    let mut dropped=0_f64;
    let mut csr=Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]};csr.row.push(0);
    for r in 0..d{let population=r/n==r%n;let mut lost=0.;
        for c in 0..d{if dense[sector(r)][sector(c)]{continue;}let v=k[r*d+c];let m=squared(v);if m==0.{continue;}
            if population||m>=floor2{csr.col.push(c as u32);csr.value.push(v);}else{lost+=m.sqrt();}}
        dropped=dropped.max(lost);csr.row.push(csr.value.len()as u32);}
    (a,b,Rest{blocks,sparse:csr},dropped,norms,largest,row_sum,Rest{blocks:vec![],sparse:populations})
}

impl FoldLibrary{
    pub fn build(generator:&Generator,plan:&HarmonicPlan,lapack:&Lapack)->Result<Self,String>{
        let clock=Instant::now();let n=generator.n;let d=n*n;
        if plan.targets.is_empty(){return Ok(Self{targets:vec![],zero_dense:None,solver:None,factors:BTreeMap::new(),
            diagnostics:FoldDiagnostics{backend:"none",..Default::default()}});}
        let solver=ShiftedL0::new(generator,lapack)?;
        let mut diagnostics=FoldDiagnostics{backend:solver.backend,frame_condition:solver.condition,woodbury_rank:solver.rank(),..Default::default()};
        let targets=plan.targets.len();let width=2*targets*d;
        // Column c of every K_t^(o), as [c][t][o][row].
        let mut build=vec![ZERO;d*width];
        let mut sources=BTreeMap::<usize,Vec<(usize,usize)>>::new();
        for(t,target)in plan.targets.iter().enumerate(){for pair in &target.pairs{sources.entry(pair.source).or_default().push((t,pair.back));}}
        let execution=Execution::new()?;let gemm=lapack.complex_gemm();let sylvester=lapack.sylvester_kernel();
        let mut scratch:Vec<BuildScratch>=Vec::new();let mut factors=BTreeMap::new();
        for s in plan.far(){
            let step=&plan.steps[s];let op=&generator.components[&step.charge].op;
            let factor=solver.factor(C{re:0.,im:step.nu},lapack)?;
            diagnostics.solve_residual=diagnostics.solve_residual.max(factor.residual);
            if factor.refine{diagnostics.refined+=1;diagnostics.refined_residual=diagnostics.refined_residual.max(solver.refined_residual(&factor,lapack)?);}
            if let Some(partners)=sources.get(&s){
                let active:Vec<bool>=(0..d).map(|c|column_active(op,n,c)).collect();
                let count=active.iter().filter(|&&a|a).count();diagnostics.columns+=count;diagnostics.far_sources+=1;
                let backs:Vec<&Superop>=partners.iter().map(|&(_,b)|&generator.components[&plan.steps[b].charge].op).collect();
                let work=count*(if factor.refine{2}else{1})*700_000;
                let(solver,factor,active,backs)=(&solver,&factor,&active,&backs);
                execution.for_mode_chunks(&mut build,width,work,MINIMUM_WORK_PER_TASK,&mut scratch,|first,chunk,scratch|{
                    let columns:Vec<usize>=(first..first+chunk.len()/width).filter(|&c|active[c]).collect();
                    if columns.is_empty(){return Ok(());}
                    let m=columns.len();let BuildScratch{first:one,second:two,unit,small,solve}=scratch;
                    grow(one,d*m);grow(two,d*m);grow(unit,d);
                    let y=&mut one[..d*m];y.fill(ZERO);
                    for(j,&c)in columns.iter().enumerate(){let e=&mut unit[..d];e.fill(ZERO);e[c]=ONE;op.apply_into(n,ONE,e,&mut y[j*d..(j+1)*d],small);}
                    // M1 = D_s^-1 C_s and M2 = D_s^-1 M1 on these columns.
                    solver.solve(factor,&gemm,&sylvester,y,m,solve)?;
                    two[..d*m].copy_from_slice(y);
                    solver.solve(factor,&gemm,&sylvester,&mut two[..d*m],m,solve)?;
                    for(j,&c)in columns.iter().enumerate(){
                        let column=&mut chunk[(c-first)*width..(c-first+1)*width];
                        for(p,&(t,_))in partners.iter().enumerate(){
                            let base=2*t*d;
                            backs[p].apply_into(n,MINUS_ONE,&one[j*d..(j+1)*d],&mut column[base..base+d],small);
                            backs[p].apply_into(n,MINUS_ONE,&two[j*d..(j+1)*d],&mut column[base+d..base+2*d],small);
                        }
                    }
                    Ok(())
                })?;
            }
            factors.insert(s,factor);
        }
        diagnostics.phase_ms[0]=clock.elapsed().as_secs_f64()*1e3;
        let mut stored=Vec::with_capacity(targets);let mut zero_dense=None;let mut k=vec![ZERO;d*d];
        for(t,target)in plan.targets.iter().enumerate(){
            let mut left:[Vec<C>;2]=[vec![],vec![]];let mut right:[Vec<C>;2]=[vec![],vec![]];
            let mut rest:[Rest;2]=[Rest::default(),Rest::default()];let mut dropped=[0.;2];let mut norms=[[0.;3];2];let mut largest=[0.;2];let mut rows=[0.;2];
            let mut population_rest:[Rest;2]=[Rest::default(),Rest::default()];
            for order in 0..2{
                // Row-major K (row (i, j), column (k, l)).
                for c in 0..d{let base=c*width+2*t*d+order*d;for r in 0..d{k[r*d+c]=build[base+r];}}
                if t==0&&order==0&&target.charge==generator.zero{zero_dense=Some(k.clone());}
                let(a,b,stored_rest,lost,norm,big,row,population)=store(&mut k,n,&generator.ground);
                left[order]=a;right[order]=b;rest[order]=stored_rest;dropped[order]=lost;norms[order]=norm;largest[order]=big;rows[order]=row;population_rest[order]=population;
            }
            stored.push(StoredFold{charge:target.charge.clone(),nu:target.nu,left,right,rest,population_rest,dropped,norms,largest_entry:largest,rest_row_sum:rows});
        }
        diagnostics.milliseconds=clock.elapsed().as_secs_f64()*1e3;diagnostics.phase_ms[1]=diagnostics.milliseconds-diagnostics.phase_ms[0];
        Ok(Self{targets:stored,zero_dense,solver:Some(solver),factors,diagnostics})
    }
    /// Largest dropped row sums of K^(0) and K^(1), summed over the targets.
    pub fn dropped(&self)->[f64;2]{
        let mut out=[0.;2];for t in &self.targets{for o in 0..2{out[o]+=t.dropped[o];}}out
    }
    /// The third-order fold source on a lattice field x: for every far step s,
    /// explicit near coupling e and far step b returning below the cut,
    ///   y[q + s + e + b] += C_b D_s^-1 C_e D_s^-1 C_s x_q   (both in the lattice),
    /// the far-shell paths q -> q+s -> q+s+e -> q+s+e+b that the second-order
    /// fold leaves out, with both far blocks at the static frequency of s.
    /// Applied in factored form, no d x d storage: one block solve per (s, q)
    /// and one per far mode r + s reached in the middle (r = q + e: the paths
    /// through the same far mode share it), whose returns land in the lattice.
    pub fn third_order_source(&self,op:&LatticeOperator<'_>,x:&[C])->Result<Vec<C>,String>{
        let prepared=op.prepared;let(plan,generator)=(&prepared.plan,&prepared.generator);
        let n=op.n;let d=n*n;let modes=op.modes.len();let mut out=vec![ZERO;modes*d];
        let Some(solver)=&self.solver else{return Ok(out)};
        let(gemm,sylvester)=(prepared.lapack.complex_gemm(),prepared.lapack.sylvester_kernel());
        let explicit=plan.explicit();
        let far=plan.far();
        // One task per far step; the parts are summed in step order.
        let steps:Vec<(&usize,&ShiftFactor)>=self.factors.iter().collect();
        let parts=Execution::new()?.map(steps,|(&s,factor)|->Result<Option<Vec<C>>,String>{
            let mut scratch=SolveScratch::default();let mut small=Vec::new();let mut out=vec![ZERO;modes*d];
            let charge_s=&plan.steps[s].charge;
            // Middle charges r (far mode r + s) with their returns (b, target).
            let mut middles:BTreeMap<Charge,Vec<(usize,usize)>>=BTreeMap::new();
            let mut hops:Vec<(usize,usize,Charge)>=Vec::new();
            for &e in &explicit{
                if (plan.steps[s].nu+plan.steps[e].nu).abs()<plan.cut{continue;}
                for q in 0..modes{
                    let r=charge_plus(&op.modes[q],&plan.steps[e].charge)?;
                    if !middles.contains_key(&r){
                        let returns:Vec<(usize,usize)>=far.iter().filter_map(|&b|{
                            let t=charge_plus(&charge_plus(&r,charge_s).ok()?,&plan.steps[b].charge).ok()?;op.index.get(&t).map(|&t|(b,t))}).collect();
                        middles.insert(r.clone(),returns);
                    }
                    if !middles[&r].is_empty(){hops.push((q,e,r));}
                }
            }
            if hops.is_empty(){return Ok(None);}
            let cs=&generator.components[charge_s].op;
            let mut y=vec![ZERO;modes*d];
            for q in 0..modes{cs.apply_into(n,ONE,&x[q*d..(q+1)*d],&mut y[q*d..(q+1)*d],&mut small);}
            solver.solve(factor,&gemm,&sylvester,&mut y,modes,&mut scratch)?;
            let slots:Vec<&Charge>=middles.iter().filter(|(_,r)|!r.is_empty()).map(|(c,_)|c).collect();
            let slot=|c:&Charge|slots.binary_search(&c).map_err(|_|"Third-order source middle without its slot".to_string());
            let mut z=vec![ZERO;slots.len()*d];
            for(q,e,r)in &hops{let k=slot(r)?;
                generator.components[&plan.steps[*e].charge].op.apply_into(n,ONE,&y[q*d..(q+1)*d],&mut z[k*d..(k+1)*d],&mut small);}
            solver.solve(factor,&gemm,&sylvester,&mut z,slots.len(),&mut scratch)?;
            for(k,r)in slots.iter().enumerate(){for &(b,t)in &middles[*r]{
                generator.components[&plan.steps[b].charge].op.apply_into(n,ONE,&z[k*d..(k+1)*d],&mut out[t*d..(t+1)*d],&mut small);}}
            Ok(Some(out))
        });
        for part in parts{if let Some(part)=part?{for(o,v)in out.iter_mut().zip(part){add(o,v);}}}
        Ok(out)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    fn dense_l0(p:&LatticePrepared)->Vec<C>{let n=p.generator.n;let d=n*n;let mut m=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut m);m}
    #[test]
    fn lattice_structured_l0_inverse_is_exact_including_recycling(){
        let f=fixture();let p=prepared(&f);let lap=&p.lapack;
        let solver=ShiftedL0::new(&p.generator,lap).unwrap();assert!(solver.rank()>0,"recycling must be a Woodbury term");
        for shift in[C{re:0.,im:2000.},C{re:0.3,im:-7.},C{re:0.,im:3.}]{
            let factor=solver.factor(shift,lap).unwrap();assert!(factor.residual<1e-13,"{:e}",factor.residual);
        }
        // The Schur frames give the same solves.
        let mut schur=ShiftedL0::new(&p.generator,lap).unwrap();let n=4;
        let nl=p.generator.zero_component().op.left.dense(n);let nr=p.generator.zero_component().op.right.dense(n);
        let(ta,qa)=lap.complex_schur(&transpose(&nl,n),n).unwrap();let(tb,qb)=lap.complex_schur(&transpose(&adjoint(&nr,n),n),n).unwrap();
        let(qa,qb)=(transpose(&qa,n),transpose(&qb,n));
        schur.left_in=adjoint(&qa,n);schur.right_in=qb.clone();schur.left_out=qa;schur.right_out=adjoint(&qb,n);schur.reduced=Reduced::Schur{ta,tb};
        let factor=schur.factor(C{re:0.,im:2000.},lap).unwrap();assert!(factor.residual<1e-13,"{:e}",factor.residual);
    }
    #[test]
    fn lattice_fold_matches_the_dense_schur_complement(){
        // Every target's K^(0) + i nu K^(1) against the exact elimination of
        // the far modes at source frequency nu, from dense d x d LU solves; the
        // difference is the second-order nu-freeze remainder.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let lap=&p.lapack;
        let l0=dense_l0(&p);
        let dense=|q:&[i32]|{let mut m=vec![ZERO;d*d];p.generator.components[q].op.dense_into(n,ONE,&mut m);m};
        for nu in[0_f64,3.,-6.]{
            let ledger=super::super::ledger::RemainderLedger::new(&p.plan,nu.abs(),[0.,0.]).unwrap();
            for(target,stored)in p.plan.targets.iter().zip(&p.fold.targets){
                let mut exact=vec![ZERO;d*d];
                for pair in &target.pairs{
                    let(s,b)=(&p.plan.steps[pair.source],&p.plan.steps[pair.back]);
                    let mut block=l0.clone();for i in 0..d{block[i*d+i].im-=s.nu+nu;}
                    let(lu,piv)=lap.complex_factor(&transpose(&block,d),d).unwrap();
                    let solved=transpose(&lap.complex_solve_matrix_factored(&lu,&piv,&transpose(&dense(&s.charge),d),d).unwrap(),d);
                    let cb=dense(&b.charge);
                    for r in 0..d{for c in 0..d{let mut v=ZERO;for k in 0..d{mul_add(&mut v,cb[r*d+k],solved[k*d+c]);}exact[r*d+c].re-=v.re;exact[r*d+c].im-=v.im;}}
                }
                let mut folded=vec![ZERO;d*d];stored.dense_into(n,0,ONE,&mut folded);stored.dense_into(n,1,C{re:0.,im:nu},&mut folded);
                // The +-s terms cancel to (V/nu)^2: roundoff scales with the pair bounds.
                let scale=target.size;
                let error=exact.iter().zip(&folded).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
                assert!(error<=ledger.nu_freeze+1e-13*scale,"t {:?} nu {nu}: |exact - folded| {error:e} > bound {:e}",target.charge,ledger.nu_freeze);
                if nu==0.{assert!(error<1e-13*scale,"static fold must be exact at nu = 0: {error:e}");}
                // Trace preservation: the output is traceless for every input.
                for c in 0..d{let mut t=ZERO;for i in 0..n{add(&mut t,folded[(i*n+i)*d+c]);}assert!(magnitude(t)<1e-13*scale);}
            }
        }
    }
    #[test]
    fn lattice_third_order_source_is_the_near_coupling_inside_the_far_shell(){
        // The factored source equals + C_-s D_s^-1 C_e D_s^-1 C_s x from dense
        // solves, and with it in the lattice the folded steady state moves
        // toward the explicit first far shell (near couplings between far
        // modes included): what is left is beyond third order.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let lap=&p.lapack;
        let l0=dense_l0(&p);
        let dense=|q:&[i32]|{let mut m=vec![ZERO;d*d];p.generator.components[q].op.dense_into(n,ONE,&mut m);m};
        let solve=|nu:f64,rhs:&[C]|{let mut b=l0.clone();for i in 0..d{b[i*d+i].im-=nu;}
            let(lu,piv)=lap.complex_factor(&transpose(&b,d),d).unwrap();transpose(&lap.complex_solve_matrix_factored(&lu,&piv,&transpose(rhs,d),d).unwrap(),d)};
        let mm=|a:&[C],b:&[C]|{let mut o=vec![ZERO;d*d];for r in 0..d{for k in 0..d{let x=a[r*d+k];if !nonzero(x){continue;}for c in 0..d{mul_add(&mut o[r*d+c],x,b[k*d+c]);}}}o};
        let modes=p.modes(2).unwrap();let op=LatticeOperator::new(&p,&modes).unwrap();let size=op.dimension();
        let x:Vec<C>=(0..size).map(|k|C{re:((k*7+3)%29)as f64/29.-0.5,im:((k*11+5)%31)as f64/31.-0.5}).collect();
        let source=p.fold.third_order_source(&op,&x).unwrap();
        // Dense: the sum over (s, e, b) of the third-order operators on the lattice edges.
        let mut third=vec![ZERO;size*size];
        for s in p.plan.far(){let st=&p.plan.steps[s];
            for e in p.plan.explicit(){let et=&p.plan.steps[e];
                let inner=solve(st.nu,&mm(&dense(&et.charge),&solve(st.nu,&dense(&st.charge))));
                for b in p.plan.far(){let bt=&p.plan.steps[b];if (st.nu+et.nu+bt.nu).abs()>=p.plan.cut{continue;}
                    let term=mm(&dense(&bt.charge),&inner);
                    let shift=charge_plus(&charge_plus(&st.charge,&et.charge).unwrap(),&bt.charge).unwrap();
                    for(qi,q)in modes.iter().enumerate(){let t=charge_plus(q,&shift).unwrap();let Some(ti)=modes.iter().position(|m|*m==t)else{continue};
                        for r in 0..d{for c in 0..d{add(&mut third[(ti*d+r)*size+qi*d+c],term[r*d+c]);}}}
                }
            }}
        let mut expected=vec![ZERO;size];for r in 0..size{for c in 0..size{mul_add(&mut expected[r],third[r*size+c],x[c]);}}
        let scale=expected.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        let error=expected.iter().zip(&source).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
        assert!(scale>0.&&error<=1e-10*scale,"third-order source {error:e} of {scale:e}");
        // Every first-shell far charge the ball reaches (the 2003 drive moves the
        // near index by one), near couplings between them included.
        let charges:Vec<Charge>=(-1..=1).flat_map(|b:i32|{let reach=if b==0{2}else{3};(-reach..=reach).map(move|a|vec![a,b])}).collect();
        let(matrix,centre)=super::super::tests::explicit_matrix(&p,&charges);
        let explicit=super::super::tests::dense_steady(&p,&matrix,centre,n);
        let mut lattice=op.dense();let second=super::super::tests::dense_steady(&p,&lattice,op.centre,n);
        for(a,b)in lattice.iter_mut().zip(&third){add(a,*b);}
        let corrected=super::super::tests::dense_steady(&p,&lattice,op.centre,n);
        let error=|a:&[f64]|a.iter().zip(&explicit).map(|(x,y)|(x-y).abs()).fold(0.,f64::max);
        assert!(error(&second)>1e-9&&error(&corrected)<0.05*error(&second),"fold-2 {:e}, fold-2 + third order {:e}",error(&second),error(&corrected));
    }
}
