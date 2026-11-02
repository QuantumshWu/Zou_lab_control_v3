//! The fold: the far shell eliminated statically. Every far step s is
//! eliminated with the exact inverse D_s^-1 = (L0 - i nu_s)^-1 of its far
//! block:
//!   K_t^(o) = -sum_{s + s' = t} C_s' D_s^-(o+1) C_s,
//! so the far shell seen from a near source q acts as sum_o (i nu_q)^o K_t^(o)
//! on the edge q -> q + t: the expansion of (D_s - i nu_q)^-1 in nu_q. t = 0
//! is the diagonal dressing (light shifts and scattering); t != 0 are the
//! cross folds of far steps whose sum is a near charge. One fold per request
//! (`LatticePrepared::fold`): every target of the plan, K^(0) and K^(1).
//!
//! What a run applies (`RunFold::screened`): every target's absorbable part
//! (it rides in the operator's block factors and edge GEMMs at no extra
//! cost), and every stored rest but those whose action over the run is
//! negligible: a piece P is left out, smallest first, while the sum of
//! |P|_1->1 nu_max^o T over the pieces left out stays within the run's share
//! of the tolerance (the trace-norm contraction of the propagator: a static
//! piece moves the state by at most int_0^T |P x(s)|_1 ds), whatever the
//! initial state. The far shell's rings beyond the second order are the
//! run's ring targets (`ring_targets`, `pipeline`).
//!
//! Closed form: L0 = L_c + L_r with L_c the diagonal of L0 (the left and
//! right factors' diagonals) and L_r the rest (their off-diagonal entries and
//! the recycling). G_s = (L_c - i nu_s)^-1 is elementwise, and
//!   D_s^-1 = sum_k (-G_s L_r)^k G_s,  rho_s = |L_r| |G_s|
//! (largest row sums): the series is summed to the roundoff of its sum (rho_s
//! is small: 3e-6 for the tweezer, 1e-7 for counter-rotating terms); a far
//! block with rho_s >= SERIES_LIMIT is refused (such a step is not far from
//! L0's spectrum).
//! A K column c is Y_0 = D_s^-1 C_s e_c, whose first L_r acts on the
//! coupling's few elements of e_c (scattered), Y_o = D_s^-1 Y_(o-1), and
//! -C_s' Y_o; the further L_r act by GEMM over batches of columns (L_r's
//! factors are nearly dense in the request frame), columns spread over the
//! workers.
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
//! others a CSR remainder. A rest's population rows are stored apart as well
//! and are never left out, so every applied piece stays exactly trace
//! preserving.
use super::*;
use super::classification::HarmonicPlan;
use super::eigenbasis::BlockEigenbasis;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// A sector pair with at least this share of its entries stored is a dense
/// block.
const DENSE_FILL:f64=0.2;
/// Columns per batch of the far-block series.
const BATCH:usize=32;
/// Terms of a far block's series summed at most: every far block has
/// rho_s < SERIES_LIMIT, so 64 terms take any remainder below 2^-64.
const CONVERGED_TERMS:usize=64;
/// The series ratio rho_s up to which a far block is folded (the Neumann
/// series' convergence: a step whose block does not contract by half is no
/// far step).
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
/// One stored fold target: the operator applies sum_o (i nu_q)^o K^(o) with
/// K^(o) X = left[o] X + X right[o] + rest[o] X.
pub struct StoredFold{
    pub charge:Charge,pub nu:f64,
    pub left:Vec<Vec<C>>,pub right:Vec<Vec<C>>,pub rest:Vec<Rest>,
    /// The rest's population rows alone (sparse): what a run applies when
    /// it leaves the rest's other rows out (exactly trace preserving).
    pub population_rest:Vec<Rest>,
    /// Per order: induced trace-norm bounds |.|_1->1 of K^(o) whole and of
    /// its rest outside the population rows (rad/us).
    pub bounds:Vec<[f64;2]>,
}
impl StoredFold{
    /// The orders in nu_q stored.
    pub fn orders(&self)->usize{self.left.len()}
    /// out += s (A X + X B + rest X) of order o as a row-major d x d matrix,
    /// with the rest `rest` (the whole one or the population rows).
    pub fn dense_into(&self,n:usize,order:usize,rest:&Rest,s:C,out:&mut[C]){
        let op=Superop{left:Sparse::from_dense(&self.left[order],n),right:Sparse::from_dense(&self.right[order],n),sandwich:vec![]};
        op.dense_into(n,s,out);rest.dense_into(n*n,s,out);
    }
}
#[derive(Clone,Debug,Default)]
pub struct FoldDiagnostics{
    /// Far steps solved (the rest by pairing) and their active columns.
    pub far_sources:usize,pub columns:usize,
    /// |L_r| (largest row sum, rad/us), lambda0 (L0's trace-norm bound) and per
    /// far step (charge, rho_s, the series' terms).
    pub rest_bound:f64,pub lambda0:f64,pub series:Vec<(Charge,f64,usize)>,
    pub orders:usize,pub milliseconds:f64,
    /// Milliseconds: the far columns, the storage.
    pub phase_ms:[f64;2],
}

/// One target as a run applies it: the stored target and, per order, the
/// rest applied (the whole one or its population rows).
pub struct RunTarget<'a>{pub fold:&'a StoredFold,pub rests:Vec<&'a Rest>}
impl RunTarget<'_>{
    /// out += s K^(o) as applied, a row-major d x d matrix.
    pub fn dense_into(&self,n:usize,order:usize,s:C,out:&mut[C]){self.fold.dense_into(n,order,self.rests[order],s,out);}
}
/// What a run's screen left out (`RunFold::screened`): the pieces and their
/// summed bound |P|_1->1 nu_max^o T against the budget, whole targets among
/// them.
#[derive(Clone,Debug,Default)]
pub struct Screened{pub pieces:usize,pub targets:usize,pub spent:f64,pub budget:f64}
/// The fold a run's operator applies.
pub struct RunFold<'a>{pub targets:Vec<RunTarget<'a>>}
impl<'a> RunFold<'a>{
    /// Every target with every rest whole: the request's own operator (the
    /// eigenbasis, the Floquet frames, the explicit set).
    pub fn whole(fold:&'a FoldLibrary)->Self{
        Self{targets:fold.targets.iter().map(|t|RunTarget{fold:t,rests:t.rest.iter().collect()}).collect()}
    }
    /// A run's: the request's fold with its ring targets `rings` (applied
    /// whole), the pieces whose action over the run is negligible left out
    /// smallest first while their summed bound stays within `budget`: a
    /// rest's rows outside the population rows (bound |R^(o)|_1->1 nu^o T),
    /// a whole cross fold t != 0 (sum_o |K^(o)|_1->1 nu^o T; what its rests
    /// already took counted once). nu is the lattice's largest |nu_q|, T the
    /// run's longest output time. The diagonal target's absorbable part is
    /// always applied: it is part of the block factors.
    pub fn screened(fold:&'a FoldLibrary,rings:&'a[StoredFold],zero:&Charge,nu:f64,horizon:f64,budget:f64)->(Self,Screened){
        let factor=|o:usize|nu.powi(o as i32)*horizon;
        // Pieces: (cost, target, order or None for the whole target).
        let mut pieces:Vec<(f64,usize,Option<usize>)>=Vec::new();
        for(t,target)in fold.targets.iter().enumerate(){
            for(o,b)in target.bounds.iter().enumerate(){if b[1]>0.{pieces.push((b[1]*factor(o),t,Some(o)));}}
            if &target.charge!=zero{pieces.push((target.bounds.iter().enumerate().map(|(o,b)|b[0]*factor(o)).sum(),t,None));}
        }
        pieces.sort_by(|a,b|a.0.total_cmp(&b.0).then((a.1,a.2).cmp(&(b.1,b.2))));
        let mut kept=vec![true;fold.targets.len()];let mut whole:Vec<Vec<bool>>=fold.targets.iter().map(|t|vec![true;t.orders()]).collect();
        let mut taken=vec![0_f64;fold.targets.len()];let mut report=Screened{budget,..Screened::default()};
        for(cost,t,order)in pieces{
            if !kept[t]{continue;}
            let extra=match order{Some(_)=>cost,None=>(cost-taken[t]).max(0.)};
            if report.spent+extra>budget{continue;}
            match order{Some(o)=>{whole[t][o]=false;taken[t]+=cost;}None=>{kept[t]=false;report.targets+=1;}}
            report.spent+=extra;report.pieces+=1;
        }
        let mut targets:Vec<RunTarget<'a>>=fold.targets.iter().enumerate().filter(|&(t,_)|kept[t]).map(|(t,target)|RunTarget{fold:target,
            rests:(0..target.orders()).map(|o|if whole[t][o]{&target.rest[o]}else{&target.population_rest[o]}).collect()}).collect();
        targets.extend(rings.iter().map(|t|RunTarget{fold:t,rests:t.rest.iter().collect()}));
        (Self{targets},report)
    }
}
/// One far block's inverse (L0 - i nu_s)^-1: the series' terms to its
/// roundoff and G_s = (L_c - i nu_s)^-1, elementwise on a row-major block.
struct FarBlock{terms:usize,g:Vec<C>}
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
struct BuildScratch{y:Vec<C>,t:Vec<C>,w:Vec<C>,stack:Vec<C>,small:Vec<C>}

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

/// Upper bound of the spectral norm of a row-major n x n matrix:
/// sqrt(|A|_1 |A|_inf).
fn spectral_bound(a:&[C],n:usize)->f64{
    let(mut rows,mut cols)=(vec![0_f64;n],vec![0_f64;n]);
    for i in 0..n{for j in 0..n{let m=magnitude(a[i*n+j]);rows[i]+=m;cols[j]+=m;}}
    (rows.into_iter().fold(0.,f64::max)*cols.into_iter().fold(0.,f64::max)).sqrt()
}
fn sparse_spectral_bound(s:&Sparse,n:usize)->f64{
    let(mut rows,mut cols)=(vec![0_f64;n],vec![0_f64;n]);
    for &(i,j,a)in &s.entries{let m=magnitude(a);rows[i]+=m;cols[j]+=m;}
    (rows.into_iter().fold(0.,f64::max)*cols.into_iter().fold(0.,f64::max)).sqrt()
}
/// The induced trace norm of X -> A X + X B + sum P X Q, bounded by
/// |A|_2 + |B|_2 + sum |P|_2 |Q|_2.
pub(crate) fn trace_norm_bound(op:&Superop,n:usize)->f64{
    sparse_spectral_bound(&op.left,n)+sparse_spectral_bound(&op.right,n)
        +op.sandwich.iter().map(|(p,q)|sparse_spectral_bound(p,n)*sparse_spectral_bound(q,n)).sum::<f64>()
}
/// The induced trace-norm bound |P|_1->1 <= sqrt(n) |P|_2->2 <= sqrt(n)
/// sqrt(largest row sum x largest column sum) of a d x d piece from its row
/// and column sums of moduli (Schur's test).
fn schur_bound(rows:&[f64],cols:&[f64],n:usize)->f64{
    ((n as f64)*rows.iter().copied().fold(0.,f64::max)*cols.iter().copied().fold(0.,f64::max)).sqrt()
}
/// K with its absorbable part taken out (in place), the factors returned.
fn take_absorbable(k:&mut[C],n:usize)->(Vec<C>,Vec<C>){
    let d=n*n;
    // A X + X B lives on the band ((i,j),(k,j)) <- A[i,k], ((i,j),(i,l)) <- B[l,j].
    let(a,b)=absorbable(k,n);
    for i in 0..n{for j in 0..n{let row=&mut k[(i*n+j)*d..(i*n+j+1)*d];
        for kk in 0..n{let mut v=a[i*n+kk];if kk==i{add(&mut v,b[j*n+j]);}let e=&mut row[kk*n+j];e.re-=v.re;e.im-=v.im;}
        for l in(0..n).filter(|&l|l!=j){let v=b[l*n+j];let e=&mut row[i*n+l];e.re-=v.re;e.im-=v.im;}}}
    (a,b)
}
/// One stored order: the absorbable factors, the rest (dense sector blocks
/// and a sparse remainder), its population rows alone, and the bounds of
/// the whole K and of the rest outside the population rows.
type StoredOrder=(Vec<C>,Vec<C>,Rest,Rest,[f64;2]);
fn store_order(k:&mut[C],n:usize,ground:&[bool])->StoredOrder{
    let d=n*n;
    let(mut rows,mut cols)=(vec![0_f64;d],vec![0_f64;d]);
    for r in 0..d{for(c,z)in k[r*d..(r+1)*d].iter().enumerate(){let m=magnitude(*z);rows[r]+=m;cols[c]+=m;}}
    let whole=schur_bound(&rows,&cols,n);
    let(a,b)=take_absorbable(k,n);
    let sector:Vec<usize>=(0..d).map(|p|2*usize::from(!ground[p/n])+usize::from(!ground[p%n])).collect();
    let mut members:Vec<Vec<usize>>=vec![Vec::new();4];let mut position=vec![0;d];
    for p in 0..d{position[p]=members[sector[p]].len();members[sector[p]].push(p);}
    rows.fill(0.);cols.fill(0.);let mut filled=[[0_usize;4];4];
    for r in 0..d{let so=sector[r];let population=r/n==r%n;
        for(c,z)in k[r*d..(r+1)*d].iter().enumerate(){let m=magnitude(*z);if m==0.{continue;}filled[so][sector[c]]+=1;
            if !population{rows[r]+=m;cols[c]+=m;}}}
    let outside=schur_bound(&rows,&cols,n);
    let mut dense=[[None;4];4];let mut blocks=Vec::new();
    for so in 0..4{for si in 0..4{
        let(rows,cols)=(&members[so],&members[si]);if rows.is_empty()||cols.is_empty(){continue;}
        if (filled[so][si] as f64)<DENSE_FILL*(rows.len()*cols.len())as f64{continue;}
        dense[so][si]=Some(blocks.len());
        blocks.push(DenseSector{rows:rows.iter().map(|&r|r as u32).collect(),cols:cols.iter().map(|&c|c as u32).collect(),values:vec![ZERO;rows.len()*cols.len()]});
    }}
    let mut populations=Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]};populations.row.push(0);
    let mut csr=Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]};csr.row.push(0);
    for r in 0..d{let(population,so)=(r/n==r%n,sector[r]);
        for(c,&v)in k[r*d..(r+1)*d].iter().enumerate(){if !nonzero(v){continue;}
            if population{populations.col.push(c as u32);populations.value.push(v);}
            if let Some(blk)=dense[so][sector[c]]{let h=members[so].len();blocks[blk].values[position[r]+h*position[c]]=v;continue;}
            csr.col.push(c as u32);csr.value.push(v);}
        csr.row.push(csr.value.len()as u32);populations.row.push(populations.value.len()as u32);}
    (a,b,Rest{blocks,sparse:csr},Rest{blocks:vec![],sparse:populations},[whole,outside])
}

/// x <- sum_{k <= terms} (-G L_r)^k G x on m consecutive blocks, block i
/// with its own G (gs[i]): the far shell's blocks of one relative charge.
#[allow(clippy::too_many_arguments)]
fn series_blocks(rest:&RestOperator,gs:&[Vec<C>],terms:usize,n:usize,x:&mut[C],term:&mut[C],t:&mut[C],m:usize,stack:&mut Vec<C>,small:&mut Vec<C>)->Result<(),String>{
    let d=n*n;
    for(xb,g)in x.chunks_exact_mut(d).zip(gs){for(v,&gv)in xb.iter_mut().zip(g){*v=mul(*v,gv);}}
    term.copy_from_slice(x);
    for _ in 0..terms{
        t.fill(ZERO);rest.subtract(n,term,t,m,stack,small)?;
        for((wb,tb),g)in term.chunks_exact_mut(d).zip(t.chunks_exact(d)).zip(gs){for((v,&gv),&u)in wb.iter_mut().zip(g).zip(tb){*v=mul(u,gv);}}
        for(a,&b)in x.iter_mut().zip(term.iter()){add(a,b);}
    }
    Ok(())
}
/// x <- sum_{k <= terms} (-G L_r)^k G x on m consecutive blocks (term, t
/// scratch of the same length); with `scattered` the first L_r acts on
/// single elements (each block a coupling's image of a unit element).
#[allow(clippy::too_many_arguments)]
fn series(rest:&RestOperator,g:&[C],terms:usize,n:usize,x:&mut[C],term:&mut[C],t:&mut[C],m:usize,scattered:bool,stack:&mut Vec<C>,small:&mut Vec<C>)->Result<(),String>{
    let d=n*n;
    for xb in x.chunks_exact_mut(d){for(v,&gv)in xb.iter_mut().zip(g){*v=mul(*v,gv);}}
    term.copy_from_slice(x);
    for k in 0..terms{
        t.fill(ZERO);
        if k==0&&scattered{for(tb,wb)in t.chunks_exact_mut(d).zip(term.chunks_exact(d)){rest.subtract_sparse(n,wb,tb);}}
        else{rest.subtract(n,term,t,m,stack,small)?;}
        for(wb,tb)in term.chunks_exact_mut(d).zip(t.chunks_exact(d)){for((v,&gv),&u)in wb.iter_mut().zip(g).zip(tb){*v=mul(u,gv);}}
        for(a,&b)in x.iter_mut().zip(term.iter()){add(a,b);}
    }
    Ok(())
}
/// The terms that sum a series of ratio rho to its roundoff:
/// rho^(m+1) / (1 - rho) <= 2^-52.
fn converged_terms(rho:f64)->usize{terms_to(rho,f64::EPSILON)}
/// The terms that leave a series of ratio rho a remainder within `share`
/// of its sum: rho^(m+1) / (1 - rho) <= share.
fn terms_to(rho:f64,share:f64)->usize{
    if !(rho>0.){return 0;}
    let mut m=0;while m<CONVERGED_TERMS&&rho.powi(m as i32+1)/(1.-rho)>share{m+=1;}m
}
/// G of a far charge at frequency nu: (L_c - i nu)^-1
/// elementwise, with its largest entry.
fn far_g(diagonal:&[C],nu:f64)->(Vec<C>,f64){
    let g:Vec<C>=diagonal.iter().map(|&l|{let w=C{re:l.re,im:l.im-nu};let m=squared(w);C{re:w.re/m,im:-w.im/m}}).collect();
    let largest=g.iter().map(|z|magnitude(*z)).fold(0_f64,f64::max);(g,largest)
}
pub struct FoldLibrary{
    pub targets:Vec<StoredFold>,
    /// K_0^(0) as a row-major d x d matrix (the static-shift test hook).
    #[cfg(test)]
    pub zero_dense:Option<Vec<C>>,
    /// L_r and L_c elementwise (every far block's series).
    rest:RestOperator,diagonal:Vec<C>,
    pub diagnostics:FoldDiagnostics,
}

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
        let diagonal:Vec<C>=(0..d).map(|p|C{re:left[p/n].re+right[p%n].re,im:left[p/n].im+right[p%n].im}).collect();
        // lambda0: L0's trace-norm bound with its Hamiltonian's diagonal centred
        // (A - i mu, B + i mu leave A X + X B unchanged).
        let(mut high,mut low)=(f64::NEG_INFINITY,f64::INFINITY);for i in 0..n{high=high.max(left[i].im);low=low.min(left[i].im);}
        let mu=if high.is_finite(){0.5*(high+low)}else{0.};
        let(mut a0,mut b0)=(zero.op.left.dense(n),zero.op.right.dense(n));for i in 0..n{a0[i*n+i].im-=mu;b0[i*n+i].im+=mu;}
        let lambda0=spectral_bound(&a0,n)+spectral_bound(&b0,n)+zero.op.sandwich.iter().map(|(p,q)|sparse_spectral_bound(p,n)*sparse_spectral_bound(q,n)).sum::<f64>();
        let mut diagnostics=FoldDiagnostics{rest_bound:bound,lambda0,..Default::default()};
        let candidates:Vec<&classification::FoldTarget>=plan.targets.iter().collect();
        if candidates.is_empty(){return Ok(Self{targets:vec![],#[cfg(test)]zero_dense:None,rest,diagonal,diagnostics});}
        // Every far block's G_s, its series' terms and gamma_s.
        let mut blocks=BTreeMap::new();let mut gamma=BTreeMap::new();
        for s in plan.far(){
            let step=&plan.steps[s];let(g,largest)=far_g(&diagonal,step.nu);let rho=bound*largest;
            if !(rho<SERIES_LIMIT){return Err(format!("Far step {:?} is too close to L0's spectrum to fold (series ratio {rho:e})",step.charge));}
            let room=step.nu.abs()-lambda0;
            if !(room>0.){return Err(format!("Far step {:?} at {:e} rad/us does not clear L0's trace-norm bound {lambda0:e}: no far block",step.charge,step.nu));}
            gamma.insert(s,1./room);
            let terms=converged_terms(rho);blocks.insert(s,FarBlock{terms,g});diagnostics.series.push((step.charge.clone(),rho,terms));
        }
        // The orders in nu_q: K^(0) and K^(1) (their remainder's bound, sum
        // c' c gamma (nu gamma)^2 / (1 - nu gamma), within the explicit
        // lattice's frequencies is what the classification's gap leaves).
        let orders=2;
        diagnostics.orders=orders;
        // The targets built here (the diagonal one and canonical t != 0) and
        // the far steps solved for them (for the diagonal target, the pairs
        // with a canonical source only).
        let built:Vec<usize>=(0..candidates.len()).filter(|&t|{let q=&candidates[t].charge;*q>=opposite(q)}).collect();
        let mut partners=BTreeMap::<usize,Vec<(usize,usize)>>::new();
        for(slot,&t)in built.iter().enumerate(){let target=candidates[t];let diagonal=target.charge==generator.zero;
            for pair in &target.pairs{let s=&plan.steps[pair.source].charge;if diagonal&&*s<opposite(s){continue;}
                partners.entry(pair.source).or_default().push((slot,pair.back));}}
        let width=orders*built.len()*d;
        // Column c of every built K^(o), as [c][slot][o][row].
        let mut build=vec![ZERO;d*width];
        let execution=Execution::new()?;let mut scratch:Vec<BuildScratch>=Vec::new();
        for(&s,partners)in &partners{
            let op=&generator.components[&plan.steps[s].charge].op;let block=blocks.get(&s).ok_or("A fold pair's source without its far block")?;let coupling=Scatter::new(op,n);
            let active:Vec<bool>=(0..d).map(|c|column_active(op,n,c)).collect();
            let count=active.iter().filter(|&&a|a).count();diagnostics.columns+=count;diagnostics.far_sources+=1;
            let backs:Vec<&Superop>=partners.iter().map(|&(_,b)|&generator.components[&plan.steps[b].charge].op).collect();
            let work=count*(orders*block.terms.max(1)*(2*n*n*n+rest.recycling.work(n))+orders*backs.iter().map(|b|b.work(n)).sum::<usize>());
            let(rest,active,backs,coupling)=(&rest,&active,&backs,&coupling);
            execution.for_mode_chunks(&mut build,width,work,MINIMUM_WORK_PER_TASK,&mut scratch,|first,chunk,scratch|{
                let BuildScratch{y,t,w,stack,small}=scratch;
                let columns:Vec<usize>=(first..first+chunk.len()/width).filter(|&c|active[c]).collect();
                for group in columns.chunks(BATCH){
                    let m=group.len();for v in[&mut *y,&mut *t,&mut *w]{grow(v,m*d);}
                    let(y,t,w)=(&mut y[..m*d],&mut t[..m*d],&mut w[..m*d]);
                    // Y_0 = D_s^-1 C_s e_c, then Y_o = D_s^-1 Y_(o-1).
                    y.fill(ZERO);for(k,&c)in group.iter().enumerate(){coupling.add(n,c/n,c%n,ONE,&mut y[k*d..(k+1)*d]);}
                    for o in 0..orders{
                        series(rest,&block.g,block.terms,n,y,w,t,m,o==0,stack,small)?;
                        for(k,&c)in group.iter().enumerate(){let column=&mut chunk[(c-first)*width..(c-first+1)*width];
                            for(&(slot,_),back)in partners.iter().zip(backs){let base=(orders*slot+o)*d;
                                back.apply_into(n,MINUS_ONE,&y[k*d..(k+1)*d],&mut column[base..base+d],small);}}
                    }
                }
                Ok(())
            })?;
        }
        diagnostics.phase_ms[0]=clock.elapsed().as_secs_f64()*1e3;
        // Row-major K_t^(o): the built ones from their columns (the diagonal
        // one completed by its flip), the others the flips of their partners.
        let items:Vec<(usize,usize)>=(0..candidates.len()).flat_map(|t|(0..orders).map(move|o|(t,o))).collect();
        let(build,built)=(&build,&built);
        let assemble=|t:usize,order:usize|->Result<Vec<C>,String>{
            let target=candidates[t];
            let(slot,flip)=match built.iter().position(|&b|b==t){Some(slot)=>(slot,false),
                None=>(built.iter().position(|&b|candidates[b].charge==opposite(&target.charge)).ok_or("A fold target without its built partner")?,true)};
            // K[r, c] = H[r, c] (+ conj H[r~, c~] for the diagonal target), or
            // conj K_-t[r~, c~] for a flipped one, H's column c at build[c][slot][order].
            let offset=(orders*slot+order)*d;let h=|r:usize,c:usize|build[c*width+offset+r];
            let diagonal=!flip&&target.charge==generator.zero;
            let mut k=vec![ZERO;d*d];
            for cb in(0..d).step_by(64){for rb in(0..d).step_by(64){for c in cb..(cb+64).min(d){let cs=swapped(c,n);for r in rb..(rb+64).min(d){
                k[r*d+c]=if flip{conj(h(swapped(r,n),cs))}else if diagonal{let(u,v)=(h(r,c),h(swapped(r,n),cs));C{re:u.re+v.re,im:u.im-v.im}}else{h(r,c)};}}}}
            Ok(k)};
        // The storage.
        let stored=execution.map(items.clone(),|(t,o)|->Result<(StoredOrder,Option<Vec<C>>),String>{
            let mut k=assemble(t,o)?;
            let zero=(cfg!(test)&&o==0&&candidates[t].charge==generator.zero).then(||k.clone());
            Ok((store_order(&mut k,n,&generator.ground),zero))
        });
        let mut parts=stored.into_iter();let mut targets=Vec::new();
        #[cfg(test)]let mut zero_dense=None;
        for target in &candidates{
            let mut fold=StoredFold{charge:target.charge.clone(),nu:target.nu,left:vec![],right:vec![],rest:vec![],population_rest:vec![],bounds:vec![]};
            for _ in 0..orders{
                let((a,b,rest,population,bounds),_zero)=parts.next().ok_or("Fold storage lost a target")??;
                #[cfg(test)]if _zero.is_some(){zero_dense=_zero;}
                fold.left.push(a);fold.right.push(b);fold.rest.push(rest);fold.population_rest.push(population);fold.bounds.push(bounds);
            }
            targets.push(fold);
        }
        diagnostics.milliseconds=clock.elapsed().as_secs_f64()*1e3;diagnostics.phase_ms[1]=diagnostics.milliseconds-diagnostics.phase_ms[0];
        Ok(Self{targets,#[cfg(test)]zero_dense,rest,diagonal,diagnostics})
    }

    /// The far shell's action on a field x on the lattice modes `modes`, ring
    /// by ring to all orders. A charge outside the lattice
    /// at |nu| >= cut that a step reaches from a lattice mode p is a far mode
    /// of p; its amplitude solves (L0 - i nu_r) y_r = -(the steps into r
    /// from the lattice and from the other far modes), exactly at its own
    /// frequency (the series to RING_SERIES of its sum), the cross couplings
    /// kept: ring k holds the amplitudes k + 1 steps out (the Neumann series in
    /// the far-far couplings), and its landing back on the lattice is the far
    /// shell's part of order k + 2. Translation covariance: every lattice mode
    /// p meets the same relative far charges w, so ring k is solved per w on
    /// the blocks of a group of FAR_GROUP source modes at once (the series'
    /// GEMMs over the blocks, each block its own G at nu_p + nu_w), the
    /// groups spread over the workers. What a far block b and its descendants
    /// can land is c gamma |b|_1 / (1 - rho) in trace norm (c the steps'
    /// summed trace-norm bounds, gamma = 1 / (|nu_r| - lambda0) the block
    /// inverse's, rho = c gamma_max over the far modes met, |b|_1 <= sqrt(n)
    /// |b|_F), times its source's weight; the sum over every block left out
    /// stays within `tolerance` (each group its sources' share): the rings
    /// stop when the next one's sum is within what is left, and before that
    /// the next ring's smallest blocks are left out while their sum stays
    /// within half of it. Landings on near charges outside the lattice are
    /// left out (the lattice's own truncation). Returns per ring the landing
    /// on every mode (field blocks weigh one).
    #[allow(clippy::too_many_arguments)]
    pub fn far_shell(&self,generator:&Generator,plan:&HarmonicPlan,nu_of:&(dyn Fn(&[i32])->Result<f64,String>+Sync),modes:&[Charge],x:&[C],tolerance:f64)->Result<FarShell,String>{
        let clock=Instant::now();let d=generator.n*generator.n;let count=modes.len();
        let sources:Vec<(usize,&[C],f64)>=(0..count).map(|p|(p,&x[p*d..(p+1)*d],1.)).filter(|(_,b,_)|b.iter().any(|z|nonzero(*z))).collect();
        let(rings,contraction,terms)=self.far_shell_sources(generator,plan,nu_of,modes,&sources,tolerance,f64::INFINITY)?;
        let mut shell=FarShell{contraction,terms,..FarShell::default()};
        for ring in rings{let mut landing=vec![ZERO;count*d];
            for((_,q),block)in ring{for(a,&b)in landing[q*d..(q+1)*d].iter_mut().zip(&block){add(a,b);}}
            shell.landings.push(landing);}
        shell.milliseconds=clock.elapsed().as_secs_f64()*1e3;
        Ok(shell)
    }
    /// The far shell of separate source blocks (`far_shell`): each source a
    /// block on a lattice mode with its weight, landed apart; per ring the
    /// landings keyed by (source, mode), the largest contraction met and the
    /// series' terms. A ring whose far blocks would take more than `bytes`
    /// (the groups together) stops the request with the reason.
    #[allow(clippy::too_many_arguments)]
    pub fn far_shell_sources(&self,generator:&Generator,plan:&HarmonicPlan,nu_of:&(dyn Fn(&[i32])->Result<f64,String>+Sync),modes:&[Charge],sources:&[(usize,&[C],f64)],tolerance:f64,bytes:f64)->Result<(Vec<BTreeMap<(usize,usize),Vec<C>>>,f64,usize),String>{
        let n=generator.n;let d=n*n;
        let index:BTreeMap<&[i32],usize>=modes.iter().enumerate().map(|(i,q)|(q.as_slice(),i)).collect();
        let nu_modes:Vec<f64>=modes.iter().map(|q|nu_of(q)).collect::<Result<_,_>>()?;
        let steps:Vec<(&Charge,f64,&Superop)>=plan.steps.iter().map(|s|(&s.charge,s.nu,&generator.components[&s.charge].op)).collect();
        let c_all:f64=plan.steps.iter().map(|s|trace_norm_bound(&generator.components[&s.charge].op,n)).sum();
        let(bound,lambda0)=(self.diagnostics.rest_bound,self.diagnostics.lambda0);
        let trace_bound=|v:&[C]|(n as f64).sqrt()*v.iter().map(|z|squared(*z)).sum::<f64>().sqrt();
        // The sources in groups; each group's far shell on its own.
        let groups:Vec<Vec<usize>>=(0..sources.len()).collect::<Vec<_>>().chunks(FAR_GROUP).map(|g|g.to_vec()).collect();
        let weight:f64=sources.iter().map(|s|s.2).sum();let groups_count=groups.len();
        type Ring=BTreeMap<Charge,(f64,BTreeMap<usize,Vec<C>>)>;
        type Group=(Vec<BTreeMap<(usize,usize),Vec<C>>>,f64,usize);
        let execution=Execution::new()?;
        let parts=execution.map(groups,|group|->Result<Group,String>{
            let(mut landings,mut contraction,mut terms)=(Vec::new(),0_f64,0);
            let(mut small,mut stack)=(Vec::new(),Vec::new());
            // The first ring's sources: every step from the group's blocks to a far charge.
            let mut ring:Ring=BTreeMap::new();
            for &(c,nu,op)in &steps{for &s in &group{let(p,x,_)=sources[s];
                let r=charge_plus(&modes[p],c)?;if index.contains_key(r.as_slice())||(nu_modes[p]+nu).abs()<plan.cut{continue;}
                let entry=ring.entry(c.clone()).or_insert_with(||(nu,BTreeMap::new()));
                op.apply_into(n,ONE,x,entry.1.entry(s).or_insert_with(||vec![ZERO;d]),&mut small);}}
            // The group's share of the tolerance, and what is left of it.
            let mut left=if weight>0.{tolerance*group.iter().map(|&s|sources[s].2).sum::<f64>()/weight}else{0.};
            let mut gamma_far=0_f64;
            for _ in 0..FAR_RINGS{
                if ring.is_empty(){break;}
                let(mut landing,mut next):(BTreeMap<(usize,usize),Vec<C>>,Ring)=(BTreeMap::new(),BTreeMap::new());
                for(w,(nu,blocks))in std::mem::take(&mut ring){
                    // y = -(L0 - i nu_r)^-1 b on the blocks of w.
                    let active:Vec<usize>=blocks.keys().copied().collect();let m=active.len();
                    let mut y:Vec<C>=Vec::with_capacity(m*d);for s in &active{y.extend(blocks[s].iter().map(|z|C{re:-z.re,im:-z.im}));}
                    let mut gs:Vec<Vec<C>>=Vec::with_capacity(m);let mut rho=0_f64;
                    for &s in &active{let p=sources[s].0;let(g,gm)=far_g(&self.diagonal,nu_modes[p]+nu);let r=bound*gm;
                        if !(r<SERIES_LIMIT){return Err(format!("Far shell: the relative charge {w:?} from {:?} is too close to L0's spectrum (series ratio {r:e})",modes[p]));}
                        rho=rho.max(r);gs.push(g);}
                    let(mut term,mut t)=(vec![ZERO;m*d],vec![ZERO;m*d]);
                    let k=terms_to(rho,RING_SERIES);terms+=k*m;
                    series_blocks(&self.rest,&gs,k,n,&mut y,&mut term,&mut t,m,&mut stack,&mut small)?;
                    for(i,&s)in active.iter().enumerate(){let p=sources[s].0;let ys=&y[i*d..(i+1)*d];
                        for &(c,step_nu,op)in &steps{
                            let w2=charge_plus(&w,c)?;let r2=charge_plus(&modes[p],&w2)?;
                            if let Some(&q)=index.get(r2.as_slice()){op.apply_into(n,ONE,ys,landing.entry((s,q)).or_insert_with(||vec![ZERO;d]),&mut small);continue;}
                            if (nu_modes[p]+nu+step_nu).abs()<plan.cut{continue;}
                            let entry=next.entry(w2).or_insert_with(||(nu+step_nu,BTreeMap::new()));
                            op.apply_into(n,ONE,ys,entry.1.entry(s).or_insert_with(||vec![ZERO;d]),&mut small);
                        }}
                }
                landings.push(landing);
                // What the next ring's blocks and their descendants can land: each
                // block's gamma, rho over every far mode met.
                let mut potentials:Vec<(Charge,usize,f64)>=Vec::new();
                for(w,(nu,blocks))in &next{for(&s,b)in blocks{
                    // A far mode within L0's spectrum: the far shell is no
                    // shell (a resonant path through far steps).
                    let at=nu_modes[sources[s].0]+nu;let room=at.abs()-lambda0;
                    if !(room>0.){return Err(format!("Far shell at {at:e} rad/us is not separated from L0's spectrum (|L0| <= {lambda0:e} rad/us): not solved"));}
                    let gamma=1./room;
                    gamma_far=gamma_far.max(gamma);potentials.push((w.clone(),s,sources[s].2*gamma*trace_bound(b)));}}
                let rho=c_all*gamma_far;contraction=contraction.max(rho);
                let scale=if rho<1.{c_all/(1.-rho)}else{f64::INFINITY};
                let total:f64=potentials.iter().map(|x|scale*x.2).sum();
                if total<=left{break;}
                // The smallest blocks left out within half of what is left.
                potentials.sort_by(|a,b|a.2.total_cmp(&b.2));let mut pruned=0.;
                for(w,s,v)in &potentials{if pruned+scale*v>0.5*left{break;}pruned+=scale*v;if let Some(e)=next.get_mut(w){e.1.remove(s);}}
                left-=pruned;
                next.retain(|_,v|!v.1.is_empty());
                let held=(next.values().map(|v|v.1.len()).sum::<usize>()*d*std::mem::size_of::<C>()*groups_count)as f64;
                if held>bytes{return Err(format!("The far shell of this request would need about {:.1} GB, above its {:.1} GB share of the memory budget: not solved",held/1e9,bytes/1e9));}
                ring=next;
            }
            Ok((landings,contraction,terms))
        });
        let(mut rings,mut contraction,mut terms):(Vec<BTreeMap<(usize,usize),Vec<C>>>,f64,usize)=(Vec::new(),0.,0);
        for part in parts{let(landings,c,t)=part?;contraction=contraction.max(c);terms+=t;
            for(k,landing)in landings.into_iter().enumerate(){if k==rings.len(){rings.push(BTreeMap::new());}rings[k].extend(landing);}}
        Ok((rings,contraction,terms))
    }
    /// The far shell's static targets of rings 1 to `rings` (paths of three to
    /// rings + 2 steps; ring 0, the second order, is K^(0)) with `orders`
    /// orders in nu_q, one per landing charge in `offsets` (the lattice's edges
    /// q - p, closed under q -> -q), for a run whose responses keep them
    /// (`pipeline`): per column c the far shell from the unit element e_c,
    /// every far mode solved exactly, the columns in batches spread over the
    /// workers. The orders in nu_q are a far mode's Taylor coefficients,
    /// (D - i nu) y = -b giving D y^(o) = y^(o-1) - b^(o). A far mode of ring
    /// k + 1 whose source is below prune[k] / m of its ring's largest (m the
    /// ring's far modes) is left out: the modes left out of a ring carry at
    /// most prune[k] of its response.
    /// Hermitian pairing halves the columns: the far shell of E_c^dagger is
    /// the adjoint of E_c's with the landings mirrored, so column c~ of K_t^(o)
    /// is conj K_-t^(o)[r~, c] and only the columns c <= c~ are solved.
    #[allow(clippy::too_many_arguments)]
    pub fn ring_targets(&self,generator:&Generator,plan:&HarmonicPlan,rings:usize,orders:usize,prune:&[f64],offsets:&BTreeSet<Charge>)->Result<Vec<StoredFold>,String>{
        let n=generator.n;let d=n*n;let orders=orders.max(1);
        let landings:Vec<Charge>=offsets.iter().cloned().collect();
        if rings==0||landings.is_empty(){return Ok(Vec::new());}
        let slot:BTreeMap<&Charge,usize>=landings.iter().enumerate().map(|(i,q)|(q,i)).collect();
        let mirror:Vec<usize>=landings.iter().map(|q|slot.get(&opposite(q)).copied().ok_or("Ring targets: the landings are not closed under q -> -q")).collect::<Result<_,_>>()?;
        let steps:Vec<(&Charge,f64,&Superop)>=plan.steps.iter().map(|s|(&s.charge,s.nu,&generator.components[&s.charge].op)).collect();
        let width=landings.len()*orders*d;let mut out=vec![ZERO;d*width];
        let execution=Execution::new()?;let mut scratch:Vec<BuildScratch>=Vec::new();
        let(rest,steps,slot,diagonal,bound)=(&self.rest,&steps,&slot,&self.diagonal,self.diagnostics.rest_bound);
        let couplings:Vec<Scatter>=steps.iter().map(|&(_,_,op)|Scatter::new(op,n)).collect();let couplings=&couplings;
        let work=d*rings*orders*plan.steps.len()*plan.steps.len()*(2*n*n*n);
        execution.for_mode_chunks(&mut out,width,work,MINIMUM_WORK_PER_TASK,&mut scratch,|first,chunk,scratch|{
            let BuildScratch{t,w,stack,small,..}=scratch;
            let columns:Vec<usize>=(first..first+chunk.len()/width).filter(|&c|c<=swapped(c,n)).collect();
            for group in columns.chunks(BATCH){
                let m=group.len();for v in[&mut *t,&mut *w]{grow(v,m*d);}let(t,w)=(&mut t[..m*d],&mut w[..m*d]);
                // Ring 0: every far step from the unit elements; per far mode the
                // sources' Taylor coefficients in nu_q (order 0 alone here).
                let mut ring:BTreeMap<Charge,(f64,Vec<Vec<C>>)>=BTreeMap::new();
                for(e,&(c,nu,_))in steps.iter().enumerate(){if nu.abs()<plan.cut{continue;}
                    let mut b=vec![vec![ZERO;m*d];orders];
                    for(k,&col)in group.iter().enumerate(){couplings[e].add(n,col/n,col%n,ONE,&mut b[0][k*d..(k+1)*d]);}
                    ring.insert(c.clone(),(nu,b));}
                for k in 0..=rings{
                    let mut next:BTreeMap<Charge,(f64,Vec<Vec<C>>)>=BTreeMap::new();
                    for(r,(nu,b))in std::mem::take(&mut ring){
                        // y^(0) = -D^-1 b^(0), y^(o) = D^-1 (y^(o-1) - b^(o)), the
                        // series to RING_SERIES.
                        let(g,gm)=far_g(diagonal,nu);let terms=terms_to(bound*gm,RING_SERIES);
                        let mut y:Vec<Vec<C>>=Vec::with_capacity(orders);
                        for o in 0..orders{
                            let mut v:Vec<C>=if o==0{b[0].iter().map(|z|C{re:-z.re,im:-z.im}).collect()}else{y[o-1].iter().zip(&b[o]).map(|(a,c)|C{re:a.re-c.re,im:a.im-c.im}).collect()};
                            series(rest,&g,terms,n,&mut v,w,t,m,false,stack,small)?;y.push(v);}
                        for &(c,step_nu,op)in steps.iter(){
                            let r2=charge_plus(&r,c)?;let nu2=nu+step_nu;
                            if nu2.abs()<plan.cut{
                                // A landing: ring k >= 1 is this run's (ring 0 is K^(0)).
                                if k==0{continue;}let Some(&s)=slot.get(&r2)else{continue};
                                for(o,yo)in y.iter().enumerate(){for(i,&col)in group.iter().enumerate(){let base=(col-first)*width+(s*orders+o)*d;
                                    op.apply_into(n,ONE,&yo[i*d..(i+1)*d],&mut chunk[base..base+d],small);}}
                                continue;}
                            if k==rings{continue;}
                            let entry=next.entry(r2).or_insert_with(||(nu2,vec![vec![ZERO;m*d];orders]));
                            for(o,yo)in y.iter().enumerate(){for i in 0..m{op.apply_into(n,ONE,&yo[i*d..(i+1)*d],&mut entry.1[o][i*d..(i+1)*d],small);}}
                        }
                    }
                    // The next ring's far modes whose sources are below their share
                    // of the ring's largest are left out.
                    let top=next.values().map(|v|v.1[0].iter().map(|z|magnitude(*z)).fold(0_f64,f64::max)).fold(0_f64,f64::max);
                    let floor=prune.get(k).copied().unwrap_or(0.)*top/next.len().max(1)as f64;
                    next.retain(|_,v|v.1[0].iter().any(|z|magnitude(*z)>floor));
                    ring=next;
                }
            }
            Ok(())
        })?;
        // Row-major K^(o) per landing: the solved columns, and the others from
        // the mirror landing's.
        let(out,mirror)=(&out,&mirror);
        let assembled=execution.map((0..landings.len()).collect(),|s|->StoredFold{
            let charge=landings[s].clone();
            let nu=plan.targets.iter().chain(&plan.omitted).find(|t|t.charge==charge).map_or(0.,|t|t.nu);
            let mut fold=StoredFold{charge,nu,left:vec![],right:vec![],rest:vec![],population_rest:vec![],bounds:vec![]};
            for o in 0..orders{
                let column=|slot:usize,r:usize,c:usize|out[c*width+(slot*orders+o)*d+r];
                let mut k=vec![ZERO;d*d];
                for r in 0..d{for c in 0..d{let cs=swapped(c,n);k[r*d+c]=if c<=cs{column(s,r,c)}else{conj(column(mirror[s],swapped(r,n),cs))};}}
                let(a,b,rest,population,bounds)=store_order(&mut k,n,&generator.ground);
                fold.left.push(a);fold.right.push(b);fold.rest.push(rest);fold.population_rest.push(population);fold.bounds.push(bounds);
            }
            fold});
        Ok(assembled)
    }
}
/// At most FAR_RINGS rings of the far shell (`FoldLibrary::far_shell`).
/// Source modes per group of the far shell's action: the blocks one series
/// solves at once (GEMMs over them), a group's rings held in memory at a
/// time; the groups spread over the workers.
pub(crate) const FAR_GROUP:usize=8;
const FAR_RINGS:usize=16;
/// A ring target's far blocks are summed until the series' remainder is
/// within this share of the sum.
const RING_SERIES:f64=1e-12;
/// One far-shell action (`FoldLibrary::far_shell`): per ring k its landing on
/// the lattice (a field on the modes: the far shell's part of order k + 2),
/// the largest contraction c gamma met, the series' terms (per block) and
/// the time.
#[derive(Clone,Debug,Default)]
pub struct FarShell{pub landings:Vec<Vec<C>>,pub contraction:f64,pub terms:usize,pub milliseconds:f64}

/// x0's eigen-components below this share of the largest (|c_j| times the
/// eigenvector's largest entry) are not sources of the Duhamel response: the
/// responses are compared with a budget, not summed to roundoff. Measured on
/// three lasers at 20 us: the dropped pieces' responses 1.236e-5 with this
/// floor (17 sources) against 1.249e-5 at 1e-6 (182 sources).
const SOURCE_FLOOR:f64=1e-3;
/// The first-order Duhamel response of x0's evolution under L0' to a static
/// piece D, the far shell's rings beyond the second order (`pipeline`):
/// dx(t) = int_0^t e^{(t - s) L0'} D e^{s L0'} x0 ds, closed form in L0''s
/// eigenbasis, (e^{mu_i t} - e^{lambda_j t}) / (mu_i - lambda_j) (t e^{mu t}
/// when they coincide), mu_i = lambda_i - i nu_q on the target mode q; read as
/// populations at the run's sample times: x0's significant eigen-components
/// (their indices, coefficients and eigenvector blocks) and the sample times.
pub struct Duhamel<'e>{eigen:&'e BlockEigenbasis,columns:Vec<usize>,c:Vec<C>,blocks:Vec<C>,samples:Vec<f64>}
impl<'e> Duhamel<'e>{
    pub fn new(eigen:&'e BlockEigenbasis,initial:&[C],samples:&[f64])->Result<Self,String>{
        let d=eigen.dimension();let c0=eigen.coefficients(&[initial])?;let all=eigen.eigenvector_blocks();
        let size:Vec<f64>=(0..d).map(|j|magnitude(c0[j])*all[j*d..(j+1)*d].iter().map(|z|magnitude(*z)).fold(0_f64,f64::max)).collect();
        let top=size.iter().copied().fold(0_f64,f64::max);
        let columns:Vec<usize>=(0..d).filter(|&j|size[j]>SOURCE_FLOOR*top).collect();
        let blocks=columns.iter().flat_map(|&j|all[j*d..(j+1)*d].iter().copied()).collect();
        Ok(Self{eigen,c:columns.iter().map(|&j|c0[j]).collect(),columns,blocks,samples:samples.to_vec()})
    }
    /// The source blocks: the eigenvectors E_j of x0's significant components.
    pub fn sources(&self)->std::slice::ChunksExact<'_,C>{self.blocks.chunks_exact(self.eigen.dimension())}
    /// The sources' weights |c_j|.
    pub fn weights(&self)->impl Iterator<Item=f64>+'_{self.c.iter().map(|z|magnitude(*z))}
    pub fn count(&self)->usize{self.columns.len()}
    /// The response of a piece on a target mode of frequency nu (a mirror
    /// pair's `weight` 2): `images` holds D E_j on the target block for every
    /// source (`sources`, in order); the largest population over the sample
    /// times and states, weight x |rows (sum_j E^-1 D E_j c_j phi_ij(t))|.
    pub fn response(&self,nu:f64,weight:f64,images:&[C])->Result<f64,String>{
        let d=self.eigen.dimension();let n=(d as f64).sqrt().round()as usize;let count=self.columns.len();
        if count==0||images.iter().all(|z|!nonzero(*z)){return Ok(0.);}
        let blocks:Vec<&[C]>=images.chunks_exact(d).collect();let m=self.eigen.coefficients(&blocks)?;
        let(lambda,rows)=(&self.eigen.lambda,self.eigen.population_rows());
        let mu:Vec<C>=lambda.iter().map(|l|C{re:l.re,im:l.im-nu}).collect();
        let mut amplitude=0_f64;
        for &t in &self.samples{
            let ea:Vec<C>=mu.iter().map(|&z|cexp(z,t)).collect();
            let eb:Vec<C>=self.columns.iter().map(|&j|cexp(lambda[j],t)).collect();
            let delta:Vec<C>=(0..d).map(|i|{let mut v=ZERO;
                for jj in 0..count{let a=m[i+d*jj];if !nonzero(a){continue;}
                    let k=lambda[self.columns[jj]];let gap=C{re:mu[i].re-k.re,im:mu[i].im-k.im};
                    let phi=if magnitude(gap)*t<1e-6{let mid=C{re:0.5*(mu[i].re+k.re),im:0.5*(mu[i].im+k.im)};scaled(cexp(mid,t),t)}
                        else{let num=C{re:ea[i].re-eb[jj].re,im:ea[i].im-eb[jj].im};let g2=gap.re*gap.re+gap.im*gap.im;mul(num,C{re:gap.re/g2,im:-gap.im/g2})};
                    mul_add(&mut v,mul(a,self.c[jj]),phi);}
                v}).collect();
            for i in 0..n{let mut v=ZERO;for j in 0..d{mul_add(&mut v,rows[i+n*j],delta[j]);}amplitude=amplitude.max(weight*magnitude(v));}
        }
        Ok(amplitude)
    }
}
/// A trajectory's sample times for the responses: grid index 1, then each
/// index GEOMETRIC x the last (at least one more), to the grid's end; a
/// mean's, its horizons.
const GEOMETRIC:f64=1.5;
pub fn sample_times(times:&[f64],horizons:&[f64])->Vec<f64>{
    if times.is_empty(){return horizons.to_vec();}
    let mut out=Vec::new();let mut i=1usize;
    while i<times.len(){out.push(times[i]);i=((i as f64*GEOMETRIC)as usize).max(i+1);}
    out
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    use super::super::floquet::transpose;
    fn dense_l0(p:&LatticePrepared)->Vec<C>{let n=p.generator.n;let d=n*n;let mut m=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut m);m}
    #[test]
    fn lattice_far_block_series_is_the_exact_inverse(){
        // Every far block's series, summed to its roundoff, solves
        // (L0 - i nu_s) X = Y to roundoff, recycling included.
        let f=fixture();let p=prepared(&f);let d=16;let l0=dense_l0(&p);let fold=&p.fold;
        assert!(!p.plan.far().is_empty());
        for s in p.plan.far(){
            let nu=p.plan.steps[s].nu;let mut m=l0.clone();for i in 0..d{m[i*d+i].im-=nu;}
            let(g,largest)=far_g(&fold.diagonal,nu);let terms=converged_terms(fold.diagnostics.rest_bound*largest);
            let y:Vec<C>=(0..2*d).map(|k|C{re:((k*37+11)%97)as f64/97.-0.5,im:((k*53+7)%89)as f64/89.-0.5}).collect();
            let(mut x,mut w,mut t)=(y.clone(),vec![ZERO;2*d],vec![ZERO;2*d]);
            series(&fold.rest,&g,terms,4,&mut x,&mut w,&mut t,2,false,&mut Vec::new(),&mut Vec::new()).unwrap();
            let scale=y.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for b in 0..2{let mut r=y[b*d..(b+1)*d].to_vec();for i in 0..d{for k in 0..d{let v=mul(m[i*d+k],x[b*d+k]);r[i].re-=v.re;r[i].im-=v.im;}}
                let error=r.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
                assert!(error<=1e-13*scale.max(1.),"step {s}: residual {error:e} ({terms} terms)");}
        }
    }
    #[test]
    fn lattice_fold_matches_the_dense_schur_complement(){
        // The frame's K^(0) + i nu K^(1) against the exact elimination of the
        // far modes at source frequency nu, from dense d x d LU solves: within
        // the never-built orders' trace-norm bound (which bounds every entry)
        // and roundoff; every stored order trace preserving.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let lap=&p.lapack;let fold=&p.fold;
        let l0=dense_l0(&p);
        let dense=|q:&[i32]|{let mut m=vec![ZERO;d*d];p.generator.components[q].op.dense_into(n,ONE,&mut m);m};
        let gamma=|s:usize|1./(p.plan.steps[s].nu.abs()-fold.diagnostics.lambda0);
        let coupling=|s:usize|trace_norm_bound(&p.generator.components[&p.plan.steps[s].charge].op,n);
        for nu in[0_f64,3.,-6.]{
            for(target,stored)in p.plan.targets.iter().zip(&fold.targets){
                let mut exact=vec![ZERO;d*d];let mut bound=0.;
                for pair in &target.pairs{
                    let(s,b)=(&p.plan.steps[pair.source],&p.plan.steps[pair.back]);
                    let mut block=l0.clone();for i in 0..d{block[i*d+i].im-=s.nu+nu;}
                    let(lu,piv)=lap.complex_factor(&transpose(&block,d),d).unwrap();
                    let solved=transpose(&lap.complex_solve_matrix_factored(&lu,&piv,&transpose(&dense(&s.charge),d),d).unwrap(),d);
                    let cb=dense(&b.charge);
                    for r in 0..d{for c in 0..d{let mut v=ZERO;for k in 0..d{mul_add(&mut v,cb[r*d+k],solved[k*d+c]);}exact[r*d+c].re-=v.re;exact[r*d+c].im-=v.im;}}
                    let(g,q)=(gamma(pair.source),nu.abs()*gamma(pair.source));bound+=coupling(pair.back)*coupling(pair.source)*g*q*q/(1.-q);
                }
                let mut folded=vec![ZERO;d*d];stored.dense_into(n,0,&stored.rest[0],ONE,&mut folded);stored.dense_into(n,1,&stored.rest[1],C{re:0.,im:nu},&mut folded);
                let scale=target.size;
                let error=exact.iter().zip(&folded).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
                assert!(error<=bound+1e-13*scale,"t {:?} nu {nu}: |exact - folded| {error:e} > bound {bound:e}",target.charge);
                for c in 0..d{let mut t=ZERO;for i in 0..n{add(&mut t,folded[(i*n+i)*d+c]);}assert!(magnitude(t)<1e-13*scale);}
            }
        }
        // The far shell's first ring on a one-mode lattice is K_0 exactly, to roundoff.
        let zero=vec![p.generator.zero.clone()];let nu_of=|q:&[i32]|p.nu(q);
        let mut k0=vec![ZERO;d*d];fold.targets[0].dense_into(n,0,&fold.targets[0].rest[0],ONE,&mut k0);
        let scale=p.plan.targets[0].size;
        for c in 0..d{let mut e=vec![ZERO;d];e[c]=ONE;let shell=fold.far_shell(&p.generator,&p.plan,&nu_of,&zero,&e,0.).unwrap();
            for r in 0..d{let gap=magnitude(C{re:shell.landings[0][r].re-k0[r*d+c].re,im:shell.landings[0][r].im-k0[r*d+c].im});
                assert!(gap<=1e-12*scale,"K_0[{r},{c}]: the far shell's first ring {gap:e} off");}}
    }
    #[test]
    fn lattice_screen_leaves_out_pieces_within_its_budget_and_fewer_for_longer_times(){
        // The screen's left-out pieces cost at most the budget; a longer run
        // leaves out no more; the diagonal target's absorbable part and its
        // population rows are always applied; the rings are appended whole.
        let f=fixture();let p=prepared(&f);let fold=&p.fold;let zero=p.generator.zero.clone();
        let budget=5e-4;let mut left=usize::MAX;
        for horizon in[1e-3,1.,1e3,1e9]{
            let(run,report)=RunFold::screened(fold,&[],&zero,6.,horizon,budget);
            assert!(report.spent<=budget*(1.+1e-12),"T {horizon}: {} > {budget}",report.spent);
            assert!(report.pieces<=left,"T {horizon}: more pieces left out");left=report.pieces;
            assert!(run.targets.iter().any(|t|t.fold.charge==zero),"the diagonal target is applied");
            for t in &run.targets{for(o,rest)in t.rests.iter().enumerate(){
                let whole=std::ptr::eq(*rest,&t.fold.rest[o]);let population=std::ptr::eq(*rest,&t.fold.population_rest[o]);
                assert!(whole||population);}}
        }
        let(run,report)=RunFold::screened(fold,&[],&zero,6.,1e-12,budget);
        assert!(report.pieces>0&&run.targets.len()<fold.targets.len(),"a very short run leaves out every rest and the cross folds");
    }
    #[test]
    fn lattice_ring_targets_are_the_far_shells_higher_rings(){
        // On a one-mode lattice the ring targets (rings 1 and 2) act on every
        // unit element as the far shell's rings 1 and 2 at nu_q = 0.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let fold=&p.fold;
        let zero=vec![p.generator.zero.clone()];let nu_of=|q:&[i32]|p.nu(q);
        let targets=fold.ring_targets(&p.generator,&p.plan,2,1,&[0.,0.,0.],&zero.iter().cloned().collect()).unwrap();
        assert_eq!(targets.len(),1);let mut k=vec![ZERO;d*d];targets[0].dense_into(n,0,&targets[0].rest[0],ONE,&mut k);
        let scale=k.iter().map(|z|magnitude(*z)).fold(1e-300,f64::max);
        for c in 0..d{let mut e=vec![ZERO;d];e[c]=ONE;let shell=fold.far_shell(&p.generator,&p.plan,&nu_of,&zero,&e,0.).unwrap();
            for r in 0..d{let want=shell.landings.iter().skip(1).take(2).fold(ZERO,|a,l|C{re:a.re+l[r].re,im:a.im+l[r].im});
                // The diagonal ring target is made Hermitian: on unit elements the
                // far shell's own landing is Hermitian to roundoff as well.
                let gap=magnitude(C{re:want.re-k[r*d+c].re,im:want.im-k[r*d+c].im});
                assert!(gap<=1e-10*scale,"ring target [{r},{c}] {gap:e} off ({scale:e})");}}
    }
}
