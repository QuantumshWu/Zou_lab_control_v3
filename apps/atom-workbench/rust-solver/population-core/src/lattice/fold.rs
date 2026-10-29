//! The static second-order fold. Every far step s is eliminated with the
//! exact inverse D_s^-1 = (L0 - i nu_s)^-1 of its far block:
//!   K_t^(0) = -sum_{s + s' = t} C_s' D_s^-1 C_s,  K_t^(1) = -sum C_s' D_s^-2 C_s,
//! so the far shell seen from a near source q acts as K_t^(0) + i nu_q K_t^(1)
//! on the edge q -> q + t: the expansion of (D_s - i nu_q)^-1 in nu_q, whose
//! remainder is `ledger::RemainderLedger::nu_freeze`. t = 0 is the diagonal
//! dressing (light shifts and scattering); t != 0 are the cross folds of far
//! steps whose sum is a near charge.
//!
//! Closed form: L0 = L_c + L_r with L_c the diagonal of L0 (the left and
//! right factors' diagonals) and L_r the rest (their off-diagonal entries and
//! the recycling). G_s = (L_c - i nu_s)^-1 is elementwise, and
//!   D_s^-1 = sum_{k <= m} (-G_s L_r)^k G_s + R,
//!   |R| <= |G_s| rho_s^(m+1) / (1 - rho_s),  rho_s = |L_r| |G_s|
//! (largest row sums). Far blocks sit beyond 2 |L0| (classification), so
//! rho_s is small (3e-6 for the tweezer, 1e-7 for counter-rotating terms).
//! Each step's orders come from its own bounds: m0 for D_s^-1 C_s with the
//! pairs' remainders below K0_FLOOR of their targets' sizes, m1 for the
//! second D_s^-1 of K^(1) (which enters times nu_q) below K1_FLOOR; both
//! remainders are charged with the dropped entries. A K column c is
//! Y = D_s^-1 C_s e_c, whose first L_r acts on the coupling's few elements
//! of e_c (scattered), Z = D_s^-1 Y, and -C_s' Y, -C_s' Z; the further L_r
//! act by GEMM over batches of columns (L_r's factors are nearly dense in
//! the request frame), columns spread over the workers.
//!
//! Hermitian pairing: C_-s(X) = C_s(X^dagger)^dagger and D_-s likewise, so a
//! pair (-s, -s') gives flip(T_{s,s'}), flip(K)[(i,j),(k,l)] =
//! conj K[(j,i),(l,k)]: only the pairs with canonical s are solved for the
//! diagonal target (K_0 = H + flip(H)), only canonical targets t != 0 are
//! built (K_-t = flip(K_t)).
//!
//! Storage: each K_t^(o) is its absorbable part A X + X B (n x n left/right
//! factors, the least-squares projection) plus the rest, stored by sectors
//! (GG, GE, EG, EE of the output and input elements): a sector pair that is
//! filled is a dense block applied by GEMM, the scattered entries of the
//! others a CSR remainder above ENTRY_FLOOR times the largest entry. A rest
//! whose row sums stay below the structural coupling floor (K^(1)'s times
//! the cut frequency, the largest |nu_q| it meets) is a non-structural
//! coupling: the operator applies it but for its population rows (without a
//! tweezer the diagonal fold's rest is about 1e-7 rad/us in 0.57M entries),
//! and the rest is stored whole, so that the error estimate charges what
//! the operator leaves out by its action (`LatticeOperator::left_out`).
//! Population rows are never dropped, so the stored fold stays exactly trace
//! preserving; the dropped entries' (below ENTRY_FLOOR) largest row sum is
//! charged in the ledger.
//!
//! What the stored fold leaves out of the second-order far shell: from a
//! near mode p the far shell acts exactly as -C_b (D_s - i nu_p)^-1 C_s on
//! every far + far pair, of which the operator applies K^(0) + i nu_p K^(1)
//! with the series truncated, the entries below ENTRY_FLOOR dropped, a
//! non-structural rest but its population rows, and the kept targets only.
//! `far_responses` gives the exact action on a field (the series summed to
//! convergence, the shifted block's G elementwise from G_s), and the error
//! estimate charges the difference (`LatticeOperator::left_out`): the
//! expansion's remainder in nu_p, the truncation, the dropped entries, the
//! rests and the omitted cross folds, by their action.
use super::*;
use super::classification::HarmonicPlan;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Sparse-remainder entries below this fraction of a target's largest entry
/// are dropped (and charged). The dark states amplify dropped coherence
/// entries: at 1e-9 probe + tweezer populations move by 1e-4, at 1e-12 by
/// 3e-8. Dense sector blocks drop nothing.
const ENTRY_FLOOR:f64=1e-12;
/// A sector pair with at least this share of its entries above the floor is
/// stored as a dense block.
const DENSE_FILL:f64=0.2;
/// Each far block's Neumann remainders, relative to its targets' sizes: for
/// K^(0) far below every remainder the ledger carries, for K^(1), which
/// enters times nu_q, about the accuracy it needs (what is left the error
/// estimate charges by its action, `far_responses`).
const K0_FLOOR:f64=1e-10;
const K1_FLOOR:f64=1e-8;
/// Columns per batch of the far-block series.
const BATCH:usize=32;
/// Terms of a far block's series summed to convergence: every far block has
/// rho_s < SERIES_LIMIT, so 64 terms take any remainder below 2^-64.
const CONVERGED_TERMS:usize=64;
/// The series ratio rho_s up to which a far block is folded.
const SERIES_LIMIT:f64=0.5;
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
    /// y += s R x on one block (row-major d-vectors).
    pub fn apply_into(&self,s:C,x:&[C],y:&mut[C]){
        for b in &self.blocks{let h=b.rows.len();
            for(c,&col)in b.cols.iter().enumerate(){let v=mul(s,x[col as usize]);if !nonzero(v){continue;}
                for(r,&row)in b.rows.iter().enumerate(){mul_add(&mut y[row as usize],b.values[r+h*c],v);}}}
        self.sparse.apply_into(s,x,y);
    }
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
    /// Whether the rest is structural (its row sum times its frequency scale
    /// at least the structural floor): the operator applies it whole, else
    /// but for its population rows.
    pub kept:[bool;2],
    /// Largest row sum of the dropped rest entries.
    pub dropped:[f64;2],
    /// Frobenius norms: whole K, absorbed part, rest.
    pub norms:[[f64;3];2],
    pub largest_entry:[f64;2],
    /// Largest absolute row sum of the rest (a generator-norm bound).
    pub rest_row_sum:[f64;2],
}
impl StoredFold{
    /// The rest of order o the structural rule applies: whole, or but for
    /// its population rows (`kept`).
    pub fn applied(&self,order:usize)->&Rest{if self.kept[order]{&self.rest[order]}else{&self.population_rest[order]}}
    /// out += s K^(o) as the structural rule applies it (`applied`), a
    /// row-major d x d matrix.
    pub fn dense_into(&self,n:usize,order:usize,s:C,out:&mut[C]){
        let op=Superop{left:Sparse::from_dense(&self.left[order],n),right:Sparse::from_dense(&self.right[order],n),sandwich:vec![]};
        op.dense_into(n,s,out);self.applied(order).dense_into(n*n,s,out);
    }
}
#[derive(Clone,Debug,Default)]
pub struct FoldDiagnostics{
    /// Far steps solved (the rest by pairing) and their active columns.
    pub far_sources:usize,pub columns:usize,
    /// |L_r| (largest row sum, rad/us) and per far step (charge, rho_s, the
    /// Neumann orders of D_s^-1 for K^(0) and of the second D_s^-1 for K^(1)).
    pub rest_bound:f64,pub series:Vec<(Charge,f64,[usize;2])>,
    /// The charged Neumann remainders of K^(0) and K^(1) (row sums, summed
    /// over the targets).
    pub truncation:[f64;2],
    pub milliseconds:f64,
    /// Milliseconds: the far columns, the storage.
    pub phase_ms:[f64;2],
}
/// One far block's inverse (L0 - i nu_s)^-1 as truncated Neumann series:
/// the orders for D_s^-1 C_s (K^(0)) and for the second D_s^-1 (K^(1)).
struct FarBlock{order:[usize;2],
    /// G_s = (L_c - i nu_s)^-1, elementwise on a row-major block.
    g:Vec<C>}
/// A superoperator's factors indexed for single elements: X -> A X + X B +
/// sum P X Q at the element (k, l) is A's column k into column l, B's row l
/// into row k, and P's column k times Q's row l.
struct Scatter{left:Vec<Vec<(usize,C)>>,right:Vec<Vec<(usize,C)>>,sandwich:Vec<(Vec<Vec<(usize,C)>>,Vec<Vec<(usize,C)>>)>}
impl Scatter{
    fn new(op:&Superop,n:usize)->Self{
        let columns=|s:&Sparse|{let mut out=vec![Vec::new();n];for &(i,k,a)in &s.entries{out[k].push((i,a));}out};
        let rows=|s:&Sparse|{let mut out=vec![Vec::new();n];for &(l,j,b)in &s.entries{out[l].push((j,b));}out};
        Self{left:columns(&op.left),right:rows(&op.right),sandwich:op.sandwich.iter().map(|(p,q)|(columns(p),rows(q))).collect()}
    }
    /// out += op(v E_kl).
    fn add(&self,n:usize,k:usize,l:usize,v:C,out:&mut[C]){
        for &(i,a)in &self.left[k]{mul_add(&mut out[i*n+l],a,v);}
        for &(j,b)in &self.right[l]{mul_add(&mut out[k*n+j],v,b);}
        for(p,q)in &self.sandwich{for &(i,a)in &p[k]{let av=mul(a,v);for &(j,b)in &q[l]{mul_add(&mut out[i*n+j],av,b);}}}
    }
}
/// L_r = (NL - diag) X + X (NR - diag) + sum P X Q (the recycling): the
/// off-diagonal factors dense (nearly full in the request frame), applied
/// by GEMM over a batch of blocks; single elements scattered.
struct RestOperator{left:Vec<C>,right:Vec<C>,recycling:Superop,scatter:Scatter,lapack:Arc<Lapack>}
impl RestOperator{
    /// t -= L_r w on m consecutive row-major blocks: the right factor one
    /// GEMM on the blocks side by side ((w NR)^T = NR^T w^T), the left one
    /// one GEMM on the blocks stacked ((NL w)^T = w^T NL^T).
    fn subtract(&self,n:usize,w:&[C],t:&mut[C],m:usize,stack:&mut Vec<C>,small:&mut Vec<C>)->Result<(),String>{
        let d=n*n;let gemm=self.lapack.complex_gemm();
        gemm.multiply(n,n*m,n,MINUS_ONE,&self.right,n,w,n,ONE,t,n)?;
        grow(stack,2*d*m);let(stacked,product)=stack.split_at_mut(d*m);
        for k in 0..m{for j in 0..n{for i in 0..n{stacked[(i+n*k)+n*m*j]=w[i+n*(j+n*k)];}}}
        gemm.multiply(n*m,n,n,ONE,stacked,n*m,&self.left,n,ZERO,&mut product[..d*m],n*m)?;
        for k in 0..m{for j in 0..n{for i in 0..n{let v=product[(i+n*k)+n*m*j];let o=&mut t[i+n*(j+n*k)];o.re-=v.re;o.im-=v.im;}}}
        for k in 0..m{self.recycling.apply_into(n,MINUS_ONE,&w[k*d..(k+1)*d],&mut t[k*d..(k+1)*d],small);}
        Ok(())
    }
    /// t -= L_r w on one block with few nonzero elements.
    fn subtract_sparse(&self,n:usize,w:&[C],t:&mut[C]){
        for(e,&v)in w.iter().enumerate(){if nonzero(v){self.scatter.add(n,e/n,e%n,C{re:-v.re,im:-v.im},t);}}
    }
}
/// The largest row sum of a superoperator as a d x d matrix (A's rows, B's
/// columns, and their products for the sandwich terms).
fn row_bound(op:&Superop,n:usize)->f64{
    let rows=|s:&Sparse|{let mut r=vec![0_f64;n];for &(i,_,a)in &s.entries{r[i]+=magnitude(a);}r.into_iter().fold(0.,f64::max)};
    let columns=|s:&Sparse|{let mut c=vec![0_f64;n];for &(_,j,b)in &s.entries{c[j]+=magnitude(b);}c.into_iter().fold(0.,f64::max)};
    rows(&op.left)+columns(&op.right)+op.sandwich.iter().map(|(p,q)|rows(p)*columns(q)).sum::<f64>()
}
pub struct FoldLibrary{
    pub targets:Vec<StoredFold>,
    /// K_0^(0) as a row-major d x d matrix (the static-shift test hook).
    #[cfg(test)]
    pub zero_dense:Option<Vec<C>>,
    /// L_r and every far step's block (the expansion remainder's series).
    rest:RestOperator,blocks:BTreeMap<usize,FarBlock>,
    pub diagnostics:FoldDiagnostics,
}
#[inline]fn squared(z:C)->f64{z.re*z.re+z.im*z.im}

/// Whether input coordinate c = (k, l) has a nonzero image under `op`
/// (structurally; an exactly cancelling image just folds to zero).
fn column_active(op:&Superop,n:usize,c:usize)->bool{
    let(k,l)=(c/n,c%n);
    op.left.entries.iter().any(|e|e.1==k)||op.right.entries.iter().any(|e|e.0==l)
        ||op.sandwich.iter().any(|(p,q)|p.entries.iter().any(|e|e.1==k)&&q.entries.iter().any(|e|e.0==l))
}
/// The superoperator index of the transposed element: (i, j) -> (j, i).
#[inline]fn swapped(p:usize,n:usize)->usize{(p%n)*n+p/n}
#[derive(Default)]
struct BuildScratch{y:Vec<C>,z:Vec<C>,t:Vec<C>,w:Vec<C>,stack:Vec<C>,small:Vec<C>}

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
/// by sectors (see the module documentation), and whether the rest is
/// structural (its largest row sum times `scale` at least `rest_floor`);
/// the population rows are also stored alone. Population rows (i, i) are
/// never dropped, so the stored K keeps the fold's exact trace preservation
/// (v^T K = 0); the largest dropped row sum is returned.
/// Three passes: the scale, the rest's measures after the absorbable band is
/// taken out, the storage.
fn store(k:&mut[C],n:usize,ground:&[bool],scale:f64,rest_floor:f64)->StoredOrder{
    let d=n*n;
    let(largest,total)={let(mut l,mut t)=(0_f64,0_f64);for z in k.iter(){let m=squared(*z);l=l.max(m);t+=m;}(l.sqrt(),t.sqrt())};
    // A X + X B lives on the band ((i,j),(k,j)) <- A[i,k], ((i,j),(i,l)) <- B[l,j].
    let(a,b)=absorbable(k,n);let mut absorbed=0_f64;
    for i in 0..n{for j in 0..n{let row=&mut k[(i*n+j)*d..(i*n+j+1)*d];
        for kk in 0..n{let mut v=a[i*n+kk];if kk==i{add(&mut v,b[j*n+j]);}let e=&mut row[kk*n+j];e.re-=v.re;e.im-=v.im;absorbed+=squared(v);}
        for l in(0..n).filter(|&l|l!=j){let v=b[l*n+j];let e=&mut row[i*n+l];e.re-=v.re;e.im-=v.im;absorbed+=squared(v);}}}
    let floor=ENTRY_FLOOR*largest;let floor2=floor*floor;
    let sector:Vec<usize>=(0..d).map(|p|2*usize::from(!ground[p/n])+usize::from(!ground[p%n])).collect();
    let mut members:Vec<Vec<usize>>=vec![Vec::new();4];let mut position=vec![0;d];
    for p in 0..d{position[p]=members[sector[p]].len();members[sector[p]].push(p);}
    let(mut rest,mut row_sum,mut filled)=(0_f64,0_f64,[[0_usize;4];4]);
    for r in 0..d{let(row,so)=(&k[r*d..(r+1)*d],sector[r]);let mut sum=0.;
        for(c,z)in row.iter().enumerate(){let m=squared(*z);rest+=m;sum+=m.sqrt();if m>=floor2&&floor>0.{filled[so][sector[c]]+=1;}}
        row_sum=row_sum.max(sum);}
    let kept=scale*row_sum>=rest_floor;
    let mut dense=[[None;4];4];let mut blocks=Vec::new();
    for so in 0..4{for si in 0..4{
        let(rows,cols)=(&members[so],&members[si]);if rows.is_empty()||cols.is_empty(){continue;}
        if (filled[so][si] as f64)<DENSE_FILL*(rows.len()*cols.len())as f64{continue;}
        dense[so][si]=Some(blocks.len());
        blocks.push(DenseSector{rows:rows.iter().map(|&r|r as u32).collect(),cols:cols.iter().map(|&c|c as u32).collect(),values:vec![ZERO;rows.len()*cols.len()]});
    }}
    let(mut dropped,mut populations)=(0_f64,Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]});populations.row.push(0);
    let mut csr=Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]};csr.row.push(0);
    for r in 0..d{let(population,so)=(r/n==r%n,sector[r]);let mut lost=0.;
        for(c,&v)in k[r*d..(r+1)*d].iter().enumerate(){let m=squared(v);if m==0.{continue;}
            if population{populations.col.push(c as u32);populations.value.push(v);}
            if let Some(blk)=dense[so][sector[c]]{let h=members[so].len();blocks[blk].values[position[r]+h*position[c]]=v;continue;}
            if population||m>=floor2{csr.col.push(c as u32);csr.value.push(v);}else{lost+=m.sqrt();}}
        dropped=dropped.max(lost);csr.row.push(csr.value.len()as u32);populations.row.push(populations.value.len()as u32);}
    (a,b,Rest{blocks,sparse:csr},dropped,[total,absorbed.sqrt(),rest.sqrt()],largest,row_sum,Rest{blocks:vec![],sparse:populations},kept)
}

/// x <- sum_{k <= order} (-G_s L_r)^k G_s x on m consecutive blocks (term,
/// t scratch of the same length); with `scattered` the first L_r acts on
/// single elements (each block a coupling's image of a unit element).
#[allow(clippy::too_many_arguments)]
fn series(rest:&RestOperator,g:&[C],order:usize,n:usize,x:&mut[C],term:&mut[C],t:&mut[C],m:usize,scattered:bool,stack:&mut Vec<C>,small:&mut Vec<C>)->Result<(),String>{
    let d=n*n;
    for xb in x.chunks_exact_mut(d){for(v,&gv)in xb.iter_mut().zip(g){*v=mul(*v,gv);}}
    term.copy_from_slice(x);
    for k in 0..order{
        t.fill(ZERO);
        if k==0&&scattered{for(tb,wb)in t.chunks_exact_mut(d).zip(term.chunks_exact(d)){rest.subtract_sparse(n,wb,tb);}}
        else{rest.subtract(n,term,t,m,stack,small)?;}
        for(wb,tb)in term.chunks_exact_mut(d).zip(t.chunks_exact(d)){for((v,&gv),&u)in wb.iter_mut().zip(g).zip(tb){*v=mul(u,gv);}}
        for(a,&b)in x.iter_mut().zip(term.iter()){add(a,b);}
    }
    Ok(())
}
/// x <- (L0 - i nu_s - i nu)^-1 x on one block (G the shifted block's
/// elementwise inverse), the series summed until a term is below the sum's
/// roundoff.
fn converged(rest:&RestOperator,g:&[C],n:usize,x:&mut[C],term:&mut[C],t:&mut[C],stack:&mut Vec<C>,small:&mut Vec<C>)->Result<(),String>{
    for(v,&gv)in x.iter_mut().zip(g){*v=mul(*v,gv);}
    term.copy_from_slice(x);
    let size=|v:&[C]|v.iter().map(|z|squared(*z)).sum::<f64>();
    for _ in 0..CONVERGED_TERMS{
        t.fill(ZERO);rest.subtract(n,term,t,1,stack,small)?;
        for((v,&gv),&u)in term.iter_mut().zip(g).zip(t.iter()){*v=mul(u,gv);}
        for(a,&b)in x.iter_mut().zip(term.iter()){add(a,b);}
        if size(term)<=f64::EPSILON*f64::EPSILON*size(x){break;}
    }
    Ok(())
}
/// One stored order of a target, as `store` returns it: the absorbable
/// factors, the rest, the dropped row sum, the norms, the largest entry, the
/// rest's row sum, its population rows and whether it is structural.
type StoredOrder=(Vec<C>,Vec<C>,Rest,f64,[f64;3],f64,f64,Rest,bool);

impl FoldLibrary{
    pub fn build(generator:&Generator,plan:&HarmonicPlan,lapack:&Arc<Lapack>)->Result<Self,String>{
        let clock=Instant::now();let n=generator.n;let d=n*n;let zero=generator.zero_component();
        // L_c (the factors' diagonals) and L_r.
        let(mut left,mut right)=(vec![ZERO;n],vec![ZERO;n]);let(mut left_rest,mut right_rest)=(Vec::new(),Vec::new());
        for &(i,k,a)in &zero.op.left.entries{if i==k{add(&mut left[i],a);}else{left_rest.push((i,k,a));}}
        for &(k,j,b)in &zero.op.right.entries{if k==j{add(&mut right[j],b);}else{right_rest.push((k,j,b));}}
        let off=Superop{left:Sparse{entries:left_rest},right:Sparse{entries:right_rest},sandwich:zero.op.sandwich.clone()};
        let bound=row_bound(&off,n);
        let rest=RestOperator{left:off.left.dense(n),right:off.right.dense(n),recycling:Superop{left:Sparse::default(),right:Sparse::default(),sandwich:off.sandwich.clone()},
            scatter:Scatter::new(&off,n),lapack:Arc::clone(lapack)};
        let mut diagnostics=FoldDiagnostics{rest_bound:bound,..Default::default()};
        if plan.targets.is_empty(){return Ok(Self{targets:vec![],#[cfg(test)]zero_dense:None,rest,blocks:BTreeMap::new(),diagnostics});}
        // Every far block's G_s and rho_s, the couplings' row sums and the
        // targets' sizes in the same norms.
        let couplings:BTreeMap<usize,f64>=plan.steps.iter().enumerate().filter(|(_,s)|s.treatment==classification::Treatment::Fold)
            .map(|(i,s)|(i,row_bound(&generator.components[&s.charge].op,n))).collect();
        let mut blocks=BTreeMap::new();let mut scale=BTreeMap::new();
        for s in plan.far(){
            let step=&plan.steps[s];
            let g:Vec<C>=(0..d).map(|p|{let z=C{re:left[p/n].re+right[p%n].re,im:left[p/n].im+right[p%n].im-step.nu};let m=squared(z);C{re:z.re/m,im:-z.im/m}}).collect();
            let largest=g.iter().map(|z|magnitude(*z)).fold(0.,f64::max);let rho=bound*largest;
            if !(rho<SERIES_LIMIT){return Err(format!("Far step {:?} is too close to L0's spectrum to fold (series ratio {rho:e})",step.charge));}
            scale.insert(s,(largest,rho));blocks.insert(s,FarBlock{order:[0,0],g});
        }
        let pair=|p:&classification::FoldPair|->(f64,f64,f64){let(g,rho)=scale[&p.source];(couplings[&p.back]*couplings[&p.source],g,rho)};
        let sizes:Vec<[f64;2]>=plan.targets.iter().map(|t|t.pairs.iter().fold([0.;2],|z,p|{let(c,g,_)=pair(p);[z[0]+c*g,z[1]+c*g*g]})).collect();
        // The orders: c_s' c_s |G| rho^(m0+1) / (1 - rho) within K0_FLOOR,
        // c_s' c_s |G|^2 (rho^(m0+1) + rho^(m1+1)) / (1 - rho)^2 within
        // K1_FLOOR of the target's size, for every pair of the step.
        let remainder=|c:f64,g:f64,rho:f64,[m0,m1]:[usize;2]|[c*g*rho.powi(m0 as i32+1)/(1.-rho),c*g*g*(rho.powi(m0 as i32+1)+rho.powi(m1 as i32+1))/(1.-rho).powi(2)];
        for(&s,block)in blocks.iter_mut(){
            let(_,rho)=scale[&s];
            let pairs:Vec<(f64,f64,usize)>=plan.targets.iter().enumerate().flat_map(|(t,target)|target.pairs.iter().filter(|p|p.source==s).map(move|p|(p,t)))
                .map(|(p,t)|{let(c,g,_)=pair(p);(c,g,t)}).collect();
            let fits=|order:[usize;2],o:usize|pairs.iter().all(|&(c,g,t)|remainder(c,g,rho,order)[o]<=[K0_FLOOR,K1_FLOOR][o]*sizes[t][o]);
            let mut order=[0,0];
            while !fits(order,0)&&order[0]<64{order[0]+=1;}
            while !fits(order,1)&&order[1]<64{order[1]+=1;}
            block.order=order;diagnostics.series.push((plan.steps[s].charge.clone(),rho,order));
        }
        let charged:Vec<[f64;2]>=plan.targets.iter().map(|t|t.pairs.iter().fold([0.;2],|z,p|{let(c,g,rho)=pair(p);let r=remainder(c,g,rho,blocks[&p.source].order);[z[0]+r[0],z[1]+r[1]]})).collect();
        for c in &charged{diagnostics.truncation[0]+=c[0];diagnostics.truncation[1]+=c[1];}
        // The targets built here (the diagonal one and canonical t != 0) and
        // the far steps solved for them (for the diagonal target, the pairs
        // with a canonical source only).
        let built:Vec<usize>=(0..plan.targets.len()).filter(|&t|{let q=&plan.targets[t].charge;*q>=opposite(q)}).collect();
        let mut partners=BTreeMap::<usize,Vec<(usize,usize)>>::new();
        for(slot,&t)in built.iter().enumerate(){let target=&plan.targets[t];let diagonal=target.charge==generator.zero;
            for pair in &target.pairs{let s=&plan.steps[pair.source].charge;if diagonal&&*s<opposite(s){continue;}
                partners.entry(pair.source).or_default().push((slot,pair.back));}}
        let width=2*built.len()*d;
        // Column c of every built K^(o), as [c][slot][o][row].
        let mut build=vec![ZERO;d*width];
        let execution=Execution::new()?;let mut scratch:Vec<BuildScratch>=Vec::new();
        for(&s,partners)in &partners{
            let op=&generator.components[&plan.steps[s].charge].op;let block=&blocks[&s];let coupling=Scatter::new(op,n);
            let active:Vec<bool>=(0..d).map(|c|column_active(op,n,c)).collect();
            let count=active.iter().filter(|&&a|a).count();diagnostics.columns+=count;diagnostics.far_sources+=1;
            let backs:Vec<&Superop>=partners.iter().map(|&(_,b)|&generator.components[&plan.steps[b].charge].op).collect();
            let work=count*((block.order[0]+block.order[1])*(2*n*n*n+rest.recycling.work(n))+backs.iter().map(|b|2*b.work(n)).sum::<usize>());
            let(rest,active,backs,coupling)=(&rest,&active,&backs,&coupling);
            execution.for_mode_chunks(&mut build,width,work,MINIMUM_WORK_PER_TASK,&mut scratch,|first,chunk,scratch|{
                let BuildScratch{y,z,t,w,stack,small}=scratch;
                let columns:Vec<usize>=(first..first+chunk.len()/width).filter(|&c|active[c]).collect();
                for group in columns.chunks(BATCH){
                    let m=group.len();for v in[&mut *y,&mut *z,&mut *t,&mut *w]{grow(v,m*d);}
                    let(y,z,t,w)=(&mut y[..m*d],&mut z[..m*d],&mut t[..m*d],&mut w[..m*d]);
                    // Y = D_s^-1 C_s e_c and Z = D_s^-1 Y.
                    y.fill(ZERO);for(k,&c)in group.iter().enumerate(){coupling.add(n,c/n,c%n,ONE,&mut y[k*d..(k+1)*d]);}
                    series(rest,&block.g,block.order[0],n,y,w,t,m,true,stack,small)?;
                    z.copy_from_slice(y);series(rest,&block.g,block.order[1],n,z,w,t,m,false,stack,small)?;
                    for(k,&c)in group.iter().enumerate(){let column=&mut chunk[(c-first)*width..(c-first+1)*width];
                        for(&(slot,_),back)in partners.iter().zip(backs){let base=2*slot*d;
                            back.apply_into(n,MINUS_ONE,&y[k*d..(k+1)*d],&mut column[base..base+d],small);
                            back.apply_into(n,MINUS_ONE,&z[k*d..(k+1)*d],&mut column[base+d..base+2*d],small);}}
                }
                Ok(())
            })?;
        }
        diagnostics.phase_ms[0]=clock.elapsed().as_secs_f64()*1e3;
        // Row-major K_t^(o): the built ones from their columns (the diagonal
        // one completed by its flip), the others the flips of their partners;
        // stored in parallel.
        let items:Vec<(usize,usize)>=(0..plan.targets.len()).flat_map(|t|[(t,0),(t,1)]).collect();
        let(build,built)=(&build,&built);
        let stored=execution.map(items,|(t,order)|->Result<(StoredOrder,Option<Vec<C>>),String>{
            let target=&plan.targets[t];
            let(slot,flip)=match built.iter().position(|&b|b==t){Some(slot)=>(slot,false),
                None=>(built.iter().position(|&b|plan.targets[b].charge==opposite(&target.charge)).ok_or("A fold target without its built partner")?,true)};
            // K[r, c] = H[r, c] (+ conj H[r~, c~] for the diagonal target), or
            // conj K_-t[r~, c~] for a flipped one, H's column c at build[c][slot][order].
            let offset=(2*slot+order)*d;let h=|r:usize,c:usize|build[c*width+offset+r];
            let diagonal=!flip&&target.charge==generator.zero;
            let mut k=vec![ZERO;d*d];
            for cb in(0..d).step_by(64){for rb in(0..d).step_by(64){for c in cb..(cb+64).min(d){let cs=swapped(c,n);for r in rb..(rb+64).min(d){
                k[r*d+c]=if flip{conj(h(swapped(r,n),cs))}else if diagonal{let(u,v)=(h(r,c),h(swapped(r,n),cs));C{re:u.re+v.re,im:u.im-v.im}}else{h(r,c)};}}}}
            let zero=(cfg!(test)&&order==0&&target.charge==generator.zero).then(||k.clone());
            Ok((store(&mut k,n,&generator.ground,if order==0{1.}else{plan.cut.min(f64::MAX)},plan.structural_floor),zero))
        });
        let mut parts=stored.into_iter();let mut targets=Vec::with_capacity(plan.targets.len());
        #[cfg(test)]let mut zero_dense=None;
        for(t,target)in plan.targets.iter().enumerate(){
            let mut fold=StoredFold{charge:target.charge.clone(),nu:target.nu,left:[vec![],vec![]],right:[vec![],vec![]],rest:[Rest::default(),Rest::default()],
                population_rest:[Rest::default(),Rest::default()],kept:[true;2],dropped:[0.;2],norms:[[0.;3];2],largest_entry:[0.;2],rest_row_sum:[0.;2]};
            for order in 0..2{
                let((a,b,rest,lost,norms,largest,row,population,kept),_zero)=parts.next().ok_or("Fold storage lost a target")??;
                #[cfg(test)]if _zero.is_some(){zero_dense=_zero;}
                fold.left[order]=a;fold.right[order]=b;fold.rest[order]=rest;fold.dropped[order]=lost+charged[t][order];fold.norms[order]=norms;
                fold.largest_entry[order]=largest;fold.rest_row_sum[order]=row;fold.population_rest[order]=population;fold.kept[order]=kept;
            }
            targets.push(fold);
        }
        diagnostics.milliseconds=clock.elapsed().as_secs_f64()*1e3;diagnostics.phase_ms[1]=diagnostics.milliseconds-diagnostics.phase_ms[0];
        Ok(Self{targets,#[cfg(test)]zero_dense,rest,blocks,diagnostics})
    }
    /// The far shell's exact second-order action from a near mode of
    /// frequency nu on its block x (row-major d): per far step s that is the
    /// source of a far + far pair (of a kept or an omitted target),
    /// y_s = -(D_s - i nu)^-1 C_s x, the series summed to convergence; the
    /// pairs' C_b complete it (`LatticeOperator::left_out`).
    pub fn far_responses(&self,generator:&Generator,plan:&HarmonicPlan,nu:f64,x:&[C])->Result<BTreeMap<usize,Vec<C>>,String>{
        let n=generator.n;let d=n*n;
        let sources:BTreeSet<usize>=plan.targets.iter().chain(&plan.omitted).flat_map(|t|t.pairs.iter().map(|p|p.source)).collect();
        let(mut term,mut t,mut stack,mut small)=(vec![ZERO;d],vec![ZERO;d],Vec::new(),Vec::new());
        let mut out=BTreeMap::new();
        for s in sources{
            let block=self.blocks.get(&s).ok_or("A fold pair's source without its far block")?;
            let mut y=vec![ZERO;d];generator.components[&plan.steps[s].charge].op.apply_into(n,MINUS_ONE,x,&mut y,&mut small);
            // (D_s - i nu)^-1: G elementwise 1 / (1 / G_s - i nu).
            let shifted:Vec<C>=block.g.iter().map(|&g|{let m=squared(g);let z=C{re:g.re/m,im:-g.im/m-nu};let w=squared(z);C{re:z.re/w,im:-z.im/w}}).collect();
            converged(&self.rest,&shifted,n,&mut y,&mut term,&mut t,&mut stack,&mut small)?;
            out.insert(s,y);
        }
        Ok(out)
    }
    /// D_s^-1 (order m0) on m consecutive row-major blocks, in place.
    #[cfg(test)]
    fn far_solve(&self,s:usize,blocks:&mut[C],m:usize)->Result<(),String>{
        let block=self.blocks.get(&s).ok_or("A far step without its block")?;let n=(((blocks.len()/m.max(1))as f64).sqrt().round())as usize;
        let(mut w,mut t)=(vec![ZERO;blocks.len()],vec![ZERO;blocks.len()]);
        series(&self.rest,&block.g,block.order[0],n,blocks,&mut w,&mut t,m,false,&mut Vec::new(),&mut Vec::new())
    }
    /// Largest dropped row sums of K^(0) and K^(1), summed over the targets.
    pub fn dropped(&self)->[f64;2]{
        let mut out=[0.;2];for t in &self.targets{for o in 0..2{out[o]+=t.dropped[o];}}out
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::floquet::transpose;
    fn dense_l0(p:&LatticePrepared)->Vec<C>{let n=p.generator.n;let d=n*n;let mut m=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut m);m}
    #[test]
    fn lattice_far_block_series_is_the_exact_inverse(){
        // Every far block's truncated Neumann series solves (L0 - i nu_s) X = Y
        // to roundoff, recycling included.
        let f=fixture();let p=prepared(&f);let d=16;let l0=dense_l0(&p);
        assert!(!p.fold.blocks.is_empty()&&p.fold.blocks.values().any(|b|b.order[0]>0));
        for(&s,block)in &p.fold.blocks{
            let nu=p.plan.steps[s].nu;let mut m=l0.clone();for i in 0..d{m[i*d+i].im-=nu;}
            let y:Vec<C>=(0..2*d).map(|k|C{re:((k*37+11)%97)as f64/97.-0.5,im:((k*53+7)%89)as f64/89.-0.5}).collect();
            let mut x=y.clone();p.fold.far_solve(s,&mut x,2).unwrap();
            let scale=y.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for b in 0..2{let mut r=y[b*d..(b+1)*d].to_vec();for i in 0..d{for k in 0..d{let v=mul(m[i*d+k],x[b*d+k]);r[i].re-=v.re;r[i].im-=v.im;}}
                // (L0 - i nu) S_m = 1 - (-L_r G)^(m+1): the residual is within rho^(m+1).
                let error=r.iter().map(|z|magnitude(*z)).fold(0.,f64::max);let rho=p.fold.diagnostics.rest_bound*block.g.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
                assert!(error<=(rho.powi(block.order[0] as i32+1)+1e-13)*scale,"step {s}: residual {error:e} (order {:?})",block.order);}
        }
    }
    #[test]
    fn lattice_weak_folds_are_omitted_or_reduced_and_charged(){
        // Far drives weak enough that their cross folds and the diagonal
        // fold's rest fall below the structural floor (a tweezer-free model's
        // counter-rotating terms): the cross targets are omitted and charged,
        // the operator applies the diagonal's absorbable part and its rest's
        // population rows.
        let mut f=fixture();for v in f.far.iter_mut().chain(f.far2.iter_mut()){*v*=1e-3;}
        let p=prepared(&f);let n=4;
        assert_eq!(p.plan.targets.len(),1);assert!(!p.plan.omitted.is_empty()&&p.plan.omitted.iter().all(|t|t.size<p.plan.structural_floor));
        // The rest is stored whole (the estimate charges what the operator
        // leaves out), its population rows apart.
        let t=&p.fold.targets[0];let s=&t.population_rest[0].sparse;
        assert!(!t.kept[0]&&t.rest[0].stored()>t.population_rest[0].stored(),"the non-structural rest stored whole");
        assert!(t.population_rest[0].blocks.is_empty()&&(0..n*n).all(|r|r/n==r%n||s.row[r+1]==s.row[r]),"population rows alone");
        assert!(std::ptr::eq(t.applied(0),&t.population_rest[0])&&t.dropped[0]<p.plan.structural_floor);
        let ledger=super::super::ledger::RemainderLedger::new(&p.plan,3.,p.fold.dropped(),0.).unwrap();
        assert!(ledger.omitted_cross>0.&&ledger.dropped_entries>=t.dropped[0]);
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
            let ledger=super::super::ledger::RemainderLedger::new(&p.plan,nu.abs(),[0.,0.],0.).unwrap();
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
                // Within nu-freeze and the charged Neumann remainders (and dropped entries).
                let charged=stored.dropped[0]+nu.abs()*stored.dropped[1];
                assert!(error<=ledger.nu_freeze+charged+1e-13*scale,"t {:?} nu {nu}: |exact - folded| {error:e} > bound {:e}",target.charge,ledger.nu_freeze+charged);
                if nu==0.{assert!(error<=stored.dropped[0]+1e-13*scale,"static fold beyond its charged remainder at nu = 0: {error:e}");}
                // The far responses (`far_responses`, completed by the pairs'
                // C_b) are the exact elimination, to roundoff.
                let(mut responses,mut small)=(vec![ZERO;d*d],Vec::new());
                for c in 0..d{let mut e=vec![ZERO;d];e[c]=ONE;let parts=p.fold.far_responses(&p.generator,&p.plan,nu,&e).unwrap();
                    let mut column=vec![ZERO;d];for pair in &target.pairs{p.generator.components[&p.plan.steps[pair.back].charge].op.apply_into(n,ONE,&parts[&pair.source],&mut column,&mut small);}
                    for r in 0..d{responses[r*d+c]=column[r];}}
                let rest=exact.iter().zip(&responses).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
                assert!(rest<=1e-12*scale,"t {:?} nu {nu}: the far responses {rest:e} from the exact elimination",target.charge);
                // Trace preservation: the output is traceless for every input.
                for c in 0..d{let mut t=ZERO;for i in 0..n{add(&mut t,folded[(i*n+i)*d+c]);}assert!(magnitude(t)<1e-13*scale);}
            }
        }
    }
}
