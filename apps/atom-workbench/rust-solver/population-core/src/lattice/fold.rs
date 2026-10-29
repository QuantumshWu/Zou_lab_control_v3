//! The fold: the far shell eliminated statically, every piece of it kept or
//! dropped by one truncation rule. Every far step s is eliminated with the
//! exact inverse D_s^-1 = (L0 - i nu_s)^-1 of its far block:
//!   K_t^(o) = -sum_{s + s' = t} C_s' D_s^-(o+1) C_s,
//! so the far shell seen from a near source q acts as sum_o (i nu_q)^o K_t^(o)
//! on the edge q -> q + t: the expansion of (D_s - i nu_q)^-1 in nu_q. t = 0
//! is the diagonal dressing (light shifts and scattering); t != 0 are the
//! cross folds of far steps whose sum is a near charge.
//!
//! The truncation rule. The true propagator is CPTP, a trace-norm contraction
//! phase by phase on the lattice (every set of laser phases is a physical
//! evolution), so a static piece D the operator leaves out moves the state by
//! at most int_0^T ||D x(s)||_1 ds and the populations by half of that. A
//! run's fold therefore keeps every piece but those it drops smallest first
//! (across every kind: the storage of small rest entries, a rest but for its
//! population rows, an order in nu_q, a cross fold) while the sum of the
//! dropped pieces' induced trace norms times T, the run's longest output
//! time, stays within BUILD_SHARE x FOLD_SHARE x tolerance; the far shell's
//! depth (`ring_targets`) takes the rest of the share at the first checkpoint,
//! on the reduced solution (`pipeline`). The norms are rigorous: a
//! superoperator X -> A X + X B + sum P X Q is bounded by |A|_2 + |B|_2 +
//! sum |P|_2 |Q|_2, a stored d x d piece by sqrt(n) sqrt(r c) (r, c its
//! largest row and column sums: ||Y||_1 <= sqrt(n) ||Y||_F), a far block's
//! inverse by gamma_s = 1 / (|nu_s| - lambda0) (lambda0 the trace-norm bound
//! of L0 with its Hamiltonian centred), and the orders in nu_q beyond the
//! built ones by sum c' c gamma (nu_max gamma)^M / (1 - nu_max gamma). Whatever
//! is dropped is computed again, by its action on the reduced solution, at
//! the end of a run (`LatticeOperator::left_out_exact`, `pipeline`).
//!
//! Closed form: L0 = L_c + L_r with L_c the diagonal of L0 (the left and
//! right factors' diagonals) and L_r the rest (their off-diagonal entries and
//! the recycling). G_s = (L_c - i nu_s)^-1 is elementwise, and
//!   D_s^-1 = sum_k (-G_s L_r)^k G_s,  rho_s = |L_r| |G_s|
//! (largest row sums): the series is summed to the roundoff of its sum (rho_s
//! is small: 3e-6 for the tweezer, 1e-7 for counter-rotating terms); a far
//! block with rho_s >= SERIES_LIMIT, or an expansion in nu_q whose ratio
//! nu_max gamma_s reaches it, is refused (such a step is not far from the
//! lattice's modes).
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
//! others a CSR remainder. Population rows are never dropped, so every stored
//! piece stays exactly trace preserving.
use super::*;
use super::classification::HarmonicPlan;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// The truncation rule's share eta of the tolerance: the dropped pieces'
/// trace-norm actions times T sum to at most FOLD_SHARE x tolerance, so they
/// move the populations by at most half of it, 5% of the tolerance (the
/// a-posteriori term at the end measures it again on the solution). A
/// deliberate margin: the reduced model's estimate keeps the other 95%.
pub const FOLD_SHARE:f64=0.1;
/// The share of the rule's budget a run's build may spend; the far shell's
/// depth, decided at the first checkpoint on the reduced solution, takes the
/// rest. The build cannot know the depth's share (the rings' actions on the
/// solution), so it may not spend the whole budget: half, so that the depth
/// needs no restart of the build's choices.
pub const BUILD_SHARE:f64=0.5;
/// The never-built orders in nu_q may take at most this share of the build's
/// budget (the orders are chosen so).
const EXPANSION_SHARE:f64=1e-2;
/// Orders in nu_q a fold builds at most.
const ORDERS_MAX:usize=8;
/// Storage bands: a rest's CSR entries (outside its population rows) by
/// magnitude, one band per decade below 1e-4 of the target's largest entry
/// down to 1e-16; each band is one piece of the rule.
const BANDS:usize=13;
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
    /// The rest's population rows alone (sparse): what keeps a rest exactly
    /// trace preserving when the rule drops the rest but for them.
    pub population_rest:Vec<Rest>,
    /// Per order: whether the operator applies the rest whole (else but for
    /// its population rows).
    pub whole:Vec<bool>,
    /// Per order: Frobenius norms (whole K, absorbed part, rest) and the
    /// largest entry.
    pub norms:Vec<[f64;3]>,pub largest_entry:Vec<f64>,
}
impl StoredFold{
    /// The orders in nu_q the operator applies.
    pub fn orders(&self)->usize{self.left.len()}
    /// The rest of order o as the rule applies it: whole, or but for its
    /// population rows.
    pub fn applied(&self,order:usize)->&Rest{if self.whole[order]{&self.rest[order]}else{&self.population_rest[order]}}
    /// out += s K^(o) as the rule applies it (`applied`), a row-major d x d
    /// matrix.
    pub fn dense_into(&self,n:usize,order:usize,s:C,out:&mut[C]){
        let op=Superop{left:Sparse::from_dense(&self.left[order],n),right:Sparse::from_dense(&self.right[order],n),sandwich:vec![]};
        op.dense_into(n,s,out);self.applied(order).dense_into(n*n,s,out);
    }
}
/// The pieces a run's fold dropped by the truncation rule, as their bounds
/// (induced trace norm x T, the rule's units): storage bands, rests but for
/// their population rows, orders in nu_q, cross folds, and the never-built
/// orders' remainder; the pieces counted and the targets kept of the
/// candidates.
#[derive(Clone,Debug,Default)]
pub struct DroppedPieces{pub entries:f64,pub rests:f64,pub orders:f64,pub targets:f64,pub expansion:f64,pub pieces:usize,pub kept_targets:usize,pub candidate_targets:usize}
impl DroppedPieces{
    /// The dropped pieces' bound (induced trace norm x T).
    pub fn total(&self)->f64{self.entries+self.rests+self.orders+self.targets+self.expansion}
}
#[derive(Clone,Debug,Default)]
pub struct FoldDiagnostics{
    /// Far steps solved (the rest by pairing) and their active columns.
    pub far_sources:usize,pub columns:usize,
    /// |L_r| (largest row sum, rad/us), lambda0 (L0's trace-norm bound) and per
    /// far step (charge, rho_s, the series' terms).
    pub rest_bound:f64,pub lambda0:f64,pub series:Vec<(Charge,f64,usize)>,
    /// The rule (None: the frame's fold, every piece kept): T (us), the
    /// lattice's largest |nu_q|, the build's budget (trace norm x T); the
    /// orders built; what was dropped.
    pub rule:Option<FoldRule>,pub orders:usize,pub dropped:DroppedPieces,
    pub milliseconds:f64,
    /// Milliseconds: the far columns, the storage.
    pub phase_ms:[f64;2],
}
/// A run's truncation rule: its longest output time T (us), the largest
/// |nu_q| of its lattice and the budget of the pieces its fold may drop
/// (their induced trace norms times T summed).
#[derive(Clone,Copy,Debug)]
pub struct FoldRule{pub horizon:f64,pub nu_max:f64,pub budget:f64}
impl FoldRule{
    /// The rule of a run: T, nu_max and the build's share of the budget.
    pub fn new(horizon:f64,nu_max:f64)->Self{Self{horizon,nu_max,budget:BUILD_SHARE*FOLD_SHARE*super::ledger::POPULATION_TOLERANCE}}
}
/// What a fold library stores: the frame's (every target of the plan, K^(0)
/// and K^(1), every piece; the eigenbasis, the Floquet frames, the
/// preconditioner and the explicit set read it), or a run's operator fold by
/// its rule (every candidate target, the omitted cross folds among them).
#[derive(Clone,Copy,Debug)]
pub enum Storage{Frame,Rule(FoldRule)}
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
/// A stored d x d piece's induced trace norm from its largest row and column
/// sums: ||E(X)||_1 <= sqrt(n) ||E(X)||_F <= sqrt(n) sqrt(r c) ||X||_1.
fn piece_bound(row:f64,col:f64,n:usize)->f64{(n as f64).sqrt()*(row*col).sqrt()}
/// The trace norm of a row-major n x n block (its singular values summed,
/// from the eigenvalues of X^dagger X).
pub(crate) fn trace_norm(lapack:&Lapack,x:&[C],n:usize)->Result<f64,String>{
    if x.iter().all(|z|!nonzero(*z)){return Ok(0.);}
    // (X^dagger X) column-major: entry (i, j) at i + n j.
    let mut g=vec![ZERO;n*n];
    for i in 0..n{for j in 0..n{let mut v=ZERO;for k in 0..n{mul_add(&mut v,conj(x[k*n+i]),x[k*n+j]);}g[i+n*j]=v;}}
    Ok(lapack.hermitian_eigen(&g,n)?.0.iter().map(|&l|l.max(0.).sqrt()).sum())
}
/// The trace norm of a lattice field: its blocks' trace norms summed (N_1).
pub(crate) fn field_trace_norm(lapack:&Lapack,x:&[C],n:usize)->Result<f64,String>{
    x.chunks_exact(n*n).map(|b|trace_norm(lapack,b,n)).sum()
}

/// What the rule reads of one order of a target before storing it: the
/// trace-norm bounds of its absorbable part, of its whole
/// rest and of the rest's part outside the population rows, and per storage
/// band (`BANDS`) that part's bound.
#[derive(Clone,Debug)]
struct OrderStats{absorbable:f64,rest:f64,outside:f64,bands:[f64;BANDS]}
/// The band of an entry of magnitude m below the target's largest entry
/// (BANDS: at or above the top threshold).
fn band(m:f64,largest:f64)->usize{
    if !(m>0.){return 0;}
    let decades=(m/largest).log10()+16.;
    if decades<0.{0}else{(decades.floor()as usize+1).min(BANDS)}
}
/// The threshold below which the first `bands` bands lie (absolute).
fn threshold(bands:usize,largest:f64)->f64{if bands==0{0.}else{largest*10_f64.powi(bands as i32-17)}}
/// K with its absorbable part taken out (in place), the factors returned.
fn take_absorbable(k:&mut[C],n:usize)->(Vec<C>,Vec<C>,f64){
    let d=n*n;
    let largest=k.iter().map(|z|magnitude(*z)).fold(0_f64,f64::max);
    // A X + X B lives on the band ((i,j),(k,j)) <- A[i,k], ((i,j),(i,l)) <- B[l,j].
    let(a,b)=absorbable(k,n);
    for i in 0..n{for j in 0..n{let row=&mut k[(i*n+j)*d..(i*n+j+1)*d];
        for kk in 0..n{let mut v=a[i*n+kk];if kk==i{add(&mut v,b[j*n+j]);}let e=&mut row[kk*n+j];e.re-=v.re;e.im-=v.im;}
        for l in(0..n).filter(|&l|l!=j){let v=b[l*n+j];let e=&mut row[i*n+l];e.re-=v.re;e.im-=v.im;}}}
    (a,b,largest)
}
/// The rule's measures of one order (`OrderStats`); K is consumed.
fn order_stats(k:&mut[C],n:usize)->OrderStats{
    let d=n*n;let(a,b,largest)=take_absorbable(k,n);
    let(mut row_all,mut col_all)=(vec![0_f64;d],vec![0_f64;d]);
    let(mut row_out,mut col_out)=(vec![0_f64;d],vec![0_f64;d]);
    let(mut row_band,mut col_band)=(vec![[0_f64;BANDS];d],vec![[0_f64;BANDS];d]);
    for r in 0..d{let population=r/n==r%n;
        for(c,z)in k[r*d..(r+1)*d].iter().enumerate(){let m=magnitude(*z);if m==0.{continue;}
            row_all[r]+=m;col_all[c]+=m;if population{continue;}
            row_out[r]+=m;col_out[c]+=m;let j=band(m,largest);if j<BANDS{row_band[r][j]+=m;col_band[c][j]+=m;}}}
    let top=|v:&[f64]|v.iter().copied().fold(0_f64,f64::max);
    let mut bands=[0.;BANDS];
    for(j,value)in bands.iter_mut().enumerate(){
        let(r,c)=(row_band.iter().map(|x|x[j]).fold(0_f64,f64::max),col_band.iter().map(|x|x[j]).fold(0_f64,f64::max));
        *value=piece_bound(r,c,n);}
    OrderStats{absorbable:spectral_bound(&a,n)+spectral_bound(&b,n),rest:piece_bound(top(&row_all),top(&col_all),n),
        outside:piece_bound(top(&row_out),top(&col_out),n),bands}
}
/// One stored order: the absorbable factors, the rest (whole, its entries
/// outside the population rows below `floor` dropped; with `whole` false
/// only the population rows) and its population rows alone, the norms
/// (Frobenius: whole, absorbed, rest) and the largest entry. Dense sector
/// blocks are stored whole.
type StoredOrder=(Vec<C>,Vec<C>,Rest,Rest,[f64;3],f64);
fn store_order(k:&mut[C],n:usize,ground:&[bool],floor:f64,whole:bool)->StoredOrder{
    let d=n*n;
    let total=k.iter().map(|z|squared(*z)).sum::<f64>().sqrt();
    let(a,b,largest)=take_absorbable(k,n);
    let absorbed=(a.iter().chain(&b).map(|z|squared(*z)).sum::<f64>()*n as f64).sqrt();
    let sector:Vec<usize>=(0..d).map(|p|2*usize::from(!ground[p/n])+usize::from(!ground[p%n])).collect();
    let mut members:Vec<Vec<usize>>=vec![Vec::new();4];let mut position=vec![0;d];
    for p in 0..d{position[p]=members[sector[p]].len();members[sector[p]].push(p);}
    let kept=|r:usize,m:f64|m>0.&&(r/n==r%n||(whole&&m>=floor));
    let(mut rest,mut filled)=(0_f64,[[0_usize;4];4]);
    for r in 0..d{let so=sector[r];
        for(c,z)in k[r*d..(r+1)*d].iter().enumerate(){let m=squared(*z);rest+=m;if kept(r,m.sqrt()){filled[so][sector[c]]+=1;}}}
    let mut dense=[[None;4];4];let mut blocks=Vec::new();
    if whole{for so in 0..4{for si in 0..4{
        let(rows,cols)=(&members[so],&members[si]);if rows.is_empty()||cols.is_empty(){continue;}
        if (filled[so][si] as f64)<DENSE_FILL*(rows.len()*cols.len())as f64{continue;}
        dense[so][si]=Some(blocks.len());
        blocks.push(DenseSector{rows:rows.iter().map(|&r|r as u32).collect(),cols:cols.iter().map(|&c|c as u32).collect(),values:vec![ZERO;rows.len()*cols.len()]});
    }}}
    let mut populations=Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]};populations.row.push(0);
    let mut csr=Csr{row:Vec::with_capacity(d+1),col:vec![],value:vec![]};csr.row.push(0);
    for r in 0..d{let(population,so)=(r/n==r%n,sector[r]);
        for(c,&v)in k[r*d..(r+1)*d].iter().enumerate(){let m=magnitude(v);if m==0.{continue;}
            if population{populations.col.push(c as u32);populations.value.push(v);}
            if let Some(blk)=dense[so][sector[c]]{let h=members[so].len();blocks[blk].values[position[r]+h*position[c]]=v;continue;}
            if kept(r,m){csr.col.push(c as u32);csr.value.push(v);}}
        csr.row.push(csr.value.len()as u32);populations.row.push(populations.value.len()as u32);}
    let population_rest=Rest{blocks:vec![],sparse:populations};
    let rest_stored=if whole{Rest{blocks,sparse:csr}}else{Rest::default()};
    (a,b,rest_stored,population_rest,[total,absorbed,rest.sqrt()],largest)
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
/// G of a far charge at frequency nu and shift z: (L_c - i nu - z)^-1
/// elementwise, with its largest entry.
fn far_g(diagonal:&[C],nu:f64,z:C)->(Vec<C>,f64){
    let g:Vec<C>=diagonal.iter().map(|&l|{let w=C{re:l.re-z.re,im:l.im-nu-z.im};let m=squared(w);C{re:w.re/m,im:-w.im/m}}).collect();
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
    pub fn build(generator:&Generator,plan:&HarmonicPlan,lapack:&Arc<Lapack>,storage:Storage)->Result<Self,String>{
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
        let couplings:Vec<f64>=plan.steps.iter().map(|s|trace_norm_bound(&generator.components[&s.charge].op,n)).collect();
        let rule=match storage{Storage::Rule(r)=>Some(r),Storage::Frame=>None};
        let mut diagnostics=FoldDiagnostics{rest_bound:bound,lambda0,rule,..Default::default()};
        // The candidate targets: the plan's, and for a run's fold its omitted
        // cross folds as well.
        let candidates:Vec<&classification::FoldTarget>=match storage{Storage::Frame=>plan.targets.iter().collect(),Storage::Rule(_)=>plan.targets.iter().chain(&plan.omitted).collect()};
        if candidates.is_empty(){return Ok(Self{targets:vec![],#[cfg(test)]zero_dense:None,rest,diagonal,diagnostics});}
        // Every far block's G_s, its series' terms and gamma_s.
        let mut blocks=BTreeMap::new();let mut gamma=BTreeMap::new();
        for s in plan.far(){
            let step=&plan.steps[s];let(g,largest)=far_g(&diagonal,step.nu,ZERO);let rho=bound*largest;
            if !(rho<SERIES_LIMIT){return Err(format!("Far step {:?} is too close to L0's spectrum to fold (series ratio {rho:e})",step.charge));}
            let room=step.nu.abs()-lambda0;
            if !(room>0.){return Err(format!("Far step {:?} at {:e} rad/us does not clear L0's trace-norm bound {lambda0:e}: no far block",step.charge,step.nu));}
            gamma.insert(s,1./room);
            let terms=converged_terms(rho);blocks.insert(s,FarBlock{terms,g});diagnostics.series.push((step.charge.clone(),rho,terms));
        }
        // The orders in nu_q: the frame's two; a run's until the never-built
        // orders' remainder, sum c' c gamma (nu_max gamma)^M / (1 - nu_max gamma),
        // times T is within EXPANSION_SHARE of the budget.
        let remainder=|target:&classification::FoldTarget,nu_max:f64,orders:usize|->Result<f64,String>{
            let mut r=0.;for pair in &target.pairs{let g=gamma[&pair.source];let p=nu_max*g;
                if !(p<SERIES_LIMIT){return Err(format!("The static expansion in nu_q does not converge on this lattice (|nu_q| gamma = {p:e} for the far step {:?}): the step is not far from the lattice's modes",plan.steps[pair.source].charge));}
                r+=couplings[pair.back]*couplings[pair.source]*g*p.powi(orders as i32)/(1.-p);}
            Ok(r)};
        let orders=match rule{None=>2,Some(rule)=>{
            let mut m=1;
            if rule.nu_max>0.{while m<ORDERS_MAX&&candidates.iter().map(|t|remainder(t,rule.nu_max,m)).sum::<Result<f64,String>>()?*rule.horizon>EXPANSION_SHARE*rule.budget{m+=1;}}
            m}};
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
        // The rule: per target and order, its pieces and their bounds; dropped
        // smallest first while the sum stays within the budget.
        let(mut floors,mut whole,mut kept_orders,mut kept_target)=(vec![vec![0_usize;orders];candidates.len()],vec![vec![true;orders];candidates.len()],vec![orders;candidates.len()],vec![true;candidates.len()]);
        if let Some(rule)=rule{
            let stats=execution.map(items.clone(),|(t,o)|->Result<OrderStats,String>{let mut k=assemble(t,o)?;Ok(order_stats(&mut k,n))});
            let mut table=vec![vec![None;orders];candidates.len()];
            for(&(t,o),s)in items.iter().zip(stats){table[t][o]=Some(s?);}
            let table:Vec<Vec<OrderStats>>=table.into_iter().map(|v|v.into_iter().map(|s|s.ok_or("Fold storage lost an order")).collect::<Result<_,_>>()).collect::<Result<_,_>>()?;
            let scale=|o:usize|rule.horizon*rule.nu_max.powi(o as i32);
            let whole_order=|t:usize,o:usize|(table[t][o].absorbable+table[t][o].rest)*scale(o);
            let expansion:Vec<f64>=candidates.iter().map(|t|Ok(remainder(t,rule.nu_max,orders)?*rule.horizon)).collect::<Result<_,String>>()?;
            // Pieces: (cost, kind, target, order, band): 0 a storage band, 1 a
            // rest but its population rows, 2 an order, 3 a target.
            let mut pieces:Vec<(f64,u8,usize,usize,usize)>=Vec::new();
            for t in 0..candidates.len(){
                for o in 0..orders{let s=&table[t][o];
                    for(j,&b)in s.bands.iter().enumerate(){pieces.push((b*scale(o),0,t,o,j));}
                    if s.outside>0.{pieces.push((s.outside*scale(o),1,t,o,0));}
                    if o>0{pieces.push((whole_order(t,o),2,t,o,0));}}
                pieces.push(((0..orders).map(|o|whole_order(t,o)).sum::<f64>()+expansion[t],3,t,0,0));
            }
            pieces.sort_by(|a,b|a.0.total_cmp(&b.0).then((a.1,a.2,a.3,a.4).cmp(&(b.1,b.2,b.3,b.4))));
            // What a dropped piece already contains of another dropped one.
            let(mut spent,mut contained)=(0_f64,vec![vec![0_f64;orders];candidates.len()]);
            let mut dropped=DroppedPieces{candidate_targets:candidates.len(),..Default::default()};
            let budget=rule.budget-expansion.iter().sum::<f64>();
            for &(cost,kind,t,o,j)in &pieces{
                if !kept_target[t]{continue;}
                match kind{
                    0=>{if o>=kept_orders[t]||!whole[t][o]||floors[t][o]!=j||spent+cost>budget{continue;}
                        floors[t][o]=j+1;contained[t][o]+=cost;spent+=cost;dropped.entries+=cost;}
                    1=>{if o>=kept_orders[t]||!whole[t][o]{continue;}let extra=(cost-contained[t][o]).max(0.);if spent+extra>budget{continue;}
                        whole[t][o]=false;dropped.entries-=contained[t][o];dropped.rests+=cost;spent+=extra;contained[t][o]=cost;}
                    2=>{if kept_orders[t]!=o+1{continue;}let extra=(cost-contained[t][o]).max(0.);if spent+extra>budget{continue;}
                        kept_orders[t]=o;spent+=extra;dropped.orders+=cost;
                        if whole[t][o]{dropped.entries-=contained[t][o];}else{dropped.rests-=contained[t][o];}contained[t][o]=cost;}
                    _=>{let inside:f64=contained[t].iter().sum();let extra=(cost-inside).max(0.);if spent+extra>budget{continue;}
                        kept_target[t]=false;spent+=extra;dropped.targets+=cost;
                        for o in 0..orders{if o>=kept_orders[t]{dropped.orders-=contained[t][o];}else if whole[t][o]{dropped.entries-=contained[t][o];}else{dropped.rests-=contained[t][o];}}}
                }
                dropped.pieces+=1;
            }
            dropped.expansion=(0..candidates.len()).filter(|&t|kept_target[t]).map(|t|expansion[t]).sum();
            dropped.kept_targets=kept_target.iter().filter(|&&k|k).count();
            diagnostics.dropped=dropped;
        }
        // The storage of what the rule keeps.
        let kept_items:Vec<(usize,usize)>=items.iter().copied().filter(|&(t,o)|kept_target[t]&&o<kept_orders[t]).collect();
        let(floors,whole)=(&floors,&whole);
        let stored=execution.map(kept_items.clone(),|(t,o)|->Result<(StoredOrder,Option<Vec<C>>),String>{
            let mut k=assemble(t,o)?;
            let zero=(cfg!(test)&&o==0&&candidates[t].charge==generator.zero).then(||k.clone());
            let largest=k.iter().map(|z|magnitude(*z)).fold(0_f64,f64::max);
            Ok((store_order(&mut k,n,&generator.ground,threshold(floors[t][o],largest),whole[t][o]),zero))
        });
        let mut parts=stored.into_iter();let mut targets=Vec::new();
        #[cfg(test)]let mut zero_dense=None;
        for(t,target)in candidates.iter().enumerate(){
            if !kept_target[t]{continue;}
            let mut fold=StoredFold{charge:target.charge.clone(),nu:target.nu,left:vec![],right:vec![],rest:vec![],population_rest:vec![],whole:vec![],norms:vec![],largest_entry:vec![]};
            for o in 0..kept_orders[t]{
                let((a,b,rest,population,norms,largest),_zero)=parts.next().ok_or("Fold storage lost a target")??;
                #[cfg(test)]if _zero.is_some(){zero_dense=_zero;}
                fold.left.push(a);fold.right.push(b);fold.rest.push(rest);fold.population_rest.push(population);fold.whole.push(whole[t][o]);
                fold.norms.push(norms);fold.largest_entry.push(largest);
            }
            targets.push(fold);
        }
        diagnostics.milliseconds=clock.elapsed().as_secs_f64()*1e3;diagnostics.phase_ms[1]=diagnostics.milliseconds-diagnostics.phase_ms[0];
        Ok(Self{targets,#[cfg(test)]zero_dense,rest,diagonal,diagnostics})
    }

    /// The far shell's action on a field x on the lattice modes `modes` at
    /// the shift z, ring by ring to all orders. A charge outside the lattice
    /// at |nu| >= cut that a step reaches from a lattice mode p is a far mode
    /// of p; its amplitude solves (L0 - i nu_r - z) y_r = -(the steps into r
    /// from the lattice and from the other far modes), exactly at its own
    /// frequency (the series to RING_SERIES of its sum), the cross couplings
    /// kept: ring k holds the amplitudes k + 1 steps out (the Neumann series in
    /// the far-far couplings), and its landing back on the lattice is the far
    /// shell's part of order k + 2. Translation covariance: every lattice mode
    /// p meets the same relative far charges w, so ring k is solved per w on
    /// the blocks of a group of FAR_GROUP source modes at once (the series'
    /// GEMMs over the blocks, each block its own G at nu_p + nu_w + z), the
    /// groups spread over the workers. The rings stop when what the next one
    /// and its descendants can land, c gamma |b|_1 / (1 - rho) in trace norm
    /// (c the steps' summed trace-norm bounds, gamma = 1 / (|nu_r| - lambda0 -
    /// |z|) the block inverse's, rho = c gamma_max over the far modes met,
    /// |b|_1 <= sqrt(n) |b|_F), is within `tolerance` (N_1, summed over the
    /// blocks); a block whose own bound is below FAR_PRUNE x tolerance per
    /// block of its ring is left out; both go to `tail`. Landings on near charges outside
    /// the lattice are the near ring's and are left out; the far amplitudes'
    /// trace norms per ring are the dressing. Returns per ring the landing on
    /// every mode (not mirror-completed: at a complex z the response of a
    /// Hermitian-symmetric field is not Hermitian-symmetric).
    #[allow(clippy::too_many_arguments)]
    pub fn far_shell(&self,generator:&Generator,plan:&HarmonicPlan,nu_of:&(dyn Fn(&[i32])->Result<f64,String>+Sync),modes:&[Charge],x:&[C],z:C,tolerance:f64)->Result<FarShell,String>{
        let clock=Instant::now();let n=generator.n;let d=n*n;let count=modes.len();
        let index:BTreeMap<&[i32],usize>=modes.iter().enumerate().map(|(i,q)|(q.as_slice(),i)).collect();
        let nu_modes:Vec<f64>=modes.iter().map(|q|nu_of(q)).collect::<Result<_,_>>()?;
        let steps:Vec<(&Charge,f64,&Superop)>=plan.steps.iter().map(|s|(&s.charge,s.nu,&generator.components[&s.charge].op)).collect();
        let c_all:f64=plan.steps.iter().map(|s|trace_norm_bound(&generator.components[&s.charge].op,n)).sum();
        let(bound,lambda0)=(self.diagnostics.rest_bound,self.diagnostics.lambda0);let lapack=&*self.rest.lapack;
        let largest=|v:&[C]|v.iter().map(|z|magnitude(*z)).fold(0_f64,f64::max);
        let trace_bound=|v:&[C]|(n as f64).sqrt()*v.iter().map(|z|squared(*z)).sum::<f64>().sqrt();
        // The source modes in groups; each group's far shell on its own.
        let groups:Vec<Vec<usize>>=(0..count).filter(|&p|largest(&x[p*d..(p+1)*d])>0.).collect::<Vec<_>>().chunks(FAR_GROUP).map(|g|g.to_vec()).collect();
        type Ring=BTreeMap<Charge,(f64,BTreeMap<usize,Vec<C>>)>;
        type Group=(Vec<BTreeMap<usize,Vec<C>>>,Vec<f64>,f64,f64,usize);
        let execution=Execution::new()?;
        let parts=execution.map(groups,|group|->Result<Group,String>{
            let(mut landings,mut dressing,mut tail,mut contraction,mut terms)=(Vec::new(),Vec::new(),0.,0_f64,0);
            let(mut small,mut stack)=(Vec::new(),Vec::new());
            // The first ring's sources: every step from the group's modes to a far charge.
            let mut sources:Ring=BTreeMap::new();
            for &(c,nu,op)in &steps{for &p in &group{
                let r=charge_plus(&modes[p],c)?;if index.contains_key(r.as_slice())||(nu_modes[p]+nu).abs()<plan.cut{continue;}
                let entry=sources.entry(c.clone()).or_insert_with(||(nu,BTreeMap::new()));
                op.apply_into(n,ONE,&x[p*d..(p+1)*d],entry.1.entry(p).or_insert_with(||vec![ZERO;d]),&mut small);}}
            let mut gamma_far=0_f64;
            for ring in 0..FAR_RINGS{
                if sources.is_empty(){break;}
                let(mut landing,mut dressed,mut next):(BTreeMap<usize,Vec<C>>,f64,Ring)=(BTreeMap::new(),0.,BTreeMap::new());
                for(w,(nu,blocks))in std::mem::take(&mut sources){
                    // y = -(L0 - i nu_r - z)^-1 b on the blocks of w.
                    let active:Vec<usize>=blocks.keys().copied().collect();let m=active.len();
                    let mut y:Vec<C>=Vec::with_capacity(m*d);for p in &active{y.extend(blocks[p].iter().map(|z|C{re:-z.re,im:-z.im}));}
                    let mut gs:Vec<Vec<C>>=Vec::with_capacity(m);let mut rho=0_f64;
                    for &p in &active{let(g,gm)=far_g(&self.diagonal,nu_modes[p]+nu,z);let r=bound*gm;
                        if !(r<SERIES_LIMIT){return Err(format!("Far shell: the relative charge {w:?} from {:?} is too close to L0's spectrum (series ratio {r:e})",modes[p]));}
                        rho=rho.max(r);gs.push(g);}
                    let(mut term,mut t)=(vec![ZERO;m*d],vec![ZERO;m*d]);
                    let k=terms_to(rho,RING_SERIES);terms+=k*m;
                    series_blocks(&self.rest,&gs,k,n,&mut y,&mut term,&mut t,m,&mut stack,&mut small)?;
                    for(i,&p)in active.iter().enumerate(){let yp=&y[i*d..(i+1)*d];dressed+=trace_norm(lapack,yp,n)?;
                        for &(c,step_nu,op)in &steps{
                            let w2=charge_plus(&w,c)?;let r2=charge_plus(&modes[p],&w2)?;
                            if let Some(&q)=index.get(r2.as_slice()){op.apply_into(n,ONE,yp,landing.entry(q).or_insert_with(||vec![ZERO;d]),&mut small);continue;}
                            if (nu_modes[p]+nu+step_nu).abs()<plan.cut{continue;}
                            let entry=next.entry(w2).or_insert_with(||(nu+step_nu,BTreeMap::new()));
                            op.apply_into(n,ONE,yp,entry.1.entry(p).or_insert_with(||vec![ZERO;d]),&mut small);
                        }}
                }
                landings.push(landing);dressing.push(dressed);
                // What the next ring's blocks and their descendants can land: each
                // block's gamma, rho over every far mode met.
                let mut potentials:Vec<(Charge,usize,f64)>=Vec::new();
                for(w,(nu,blocks))in &next{for(&p,b)in blocks{
                    let room=(nu_modes[p]+nu).abs()-lambda0-magnitude(z);let gamma=if room>0.{1./room}else{f64::INFINITY};
                    gamma_far=gamma_far.max(gamma);potentials.push((w.clone(),p,gamma*trace_bound(b)));}}
                let rho=c_all*gamma_far;contraction=contraction.max(rho);
                let scale=if rho<1.{c_all/(1.-rho)}else{f64::INFINITY};
                let total:f64=potentials.iter().map(|x|scale*x.2).sum();
                if total<=tolerance{tail+=total;break;}
                let floor=FAR_PRUNE*tolerance/(potentials.len()as f64);
                for(w,p,v)in &potentials{if scale*v<=floor{tail+=scale*v;if let Some(e)=next.get_mut(w){e.1.remove(p);}}}
                next.retain(|_,v|!v.1.is_empty());
                sources=next;
                if ring+1==FAR_RINGS{tail+=potentials.iter().filter(|x|sources.get(&x.0).is_some_and(|e|e.1.contains_key(&x.1))).map(|x|scale*x.2).sum::<f64>();}
            }
            Ok((landings,dressing,tail,contraction,terms))
        });
        let mut shell=FarShell::default();
        for part in parts{let(landings,dressing,tail,contraction,terms)=part?;
            for(k,landing)in landings.into_iter().enumerate(){
                if k==shell.landings.len(){shell.landings.push(vec![ZERO;count*d]);shell.dressing.push(0.);}
                for(q,block)in landing{for(a,&b)in shell.landings[k][q*d..(q+1)*d].iter_mut().zip(&block){add(a,b);}}
                shell.dressing[k]+=dressing[k];}
            shell.tail+=tail;shell.contraction=shell.contraction.max(contraction);shell.terms+=terms;}
        shell.milliseconds=clock.elapsed().as_secs_f64()*1e3;
        Ok(shell)
    }
    /// The far shell's static targets of rings 1 to `rings` (paths of three to
    /// rings + 2 steps; ring 0, the second order, is K^(0)) with `orders`
    /// orders in nu_q, one per landing charge in `offsets` (the lattice's edges
    /// q - p, closed under q -> -q), for a run whose first checkpoint found
    /// them above its share of the rule: per column c the far shell from the
    /// unit element e_c, every far mode solved exactly, the columns in batches
    /// spread over the workers. The orders in nu_q are a far mode's Taylor
    /// coefficients, (D - i nu) y = -b giving D y^(o) = y^(o-1) - b^(o). A
    /// far mode of ring k + 1 whose source is below prune[k] of its ring's
    /// largest is left out (the run's a-posteriori term measures it).
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
                        let(g,gm)=far_g(diagonal,nu,ZERO);let terms=terms_to(bound*gm,RING_SERIES);
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
                    // The next ring's far modes whose sources are below its share of
                    // the ring's largest are left out.
                    let top=next.values().map(|v|v.1[0].iter().map(|z|magnitude(*z)).fold(0_f64,f64::max)).fold(0_f64,f64::max);
                    let floor=prune.get(k).copied().unwrap_or(0.)*top;
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
            let mut fold=StoredFold{charge,nu,left:vec![],right:vec![],rest:vec![],population_rest:vec![],whole:vec![],norms:vec![],largest_entry:vec![]};
            for o in 0..orders{
                let column=|slot:usize,r:usize,c:usize|out[c*width+(slot*orders+o)*d+r];
                let mut k=vec![ZERO;d*d];
                for r in 0..d{for c in 0..d{let cs=swapped(c,n);k[r*d+c]=if c<=cs{column(s,r,c)}else{conj(column(mirror[s],swapped(r,n),cs))};}}
                let(a,b,rest,population,norms,largest)=store_order(&mut k,n,&generator.ground,0.,true);
                fold.left.push(a);fold.right.push(b);fold.rest.push(rest);fold.population_rest.push(population);fold.whole.push(true);fold.norms.push(norms);fold.largest_entry.push(largest);
            }
            fold});
        Ok(assembled)
    }
}
/// The far shell's rings stop when the next one can land within the
/// caller's tolerance; a far mode whose bound is below FAR_PRUNE of it per
/// mode of its ring is left out (to the tail); at most FAR_RINGS rings.
const FAR_PRUNE:f64=1e-2;
/// Source modes per group of the far shell's action: the blocks one series
/// solves at once (GEMMs over them), a group's rings held in memory at a
/// time; the groups spread over the workers.
pub(crate) const FAR_GROUP:usize=8;
const FAR_RINGS:usize=16;
/// A ring target's far blocks are summed until the series' remainder is
/// within this share of the sum (the a-posteriori term measures the rest).
const RING_SERIES:f64=1e-12;
/// One far-shell action (`FoldLibrary::far_shell`): per ring k its landing on
/// the lattice (a field on the modes: the far shell's part of order k + 2)
/// and its far amplitudes' trace norms summed (the dressing); the tail bound
/// past the last ring (N_1 of what the rings left out can land), the largest
/// contraction c gamma met, the series' terms (per block) and the time.
#[derive(Clone,Debug,Default)]
pub struct FarShell{pub landings:Vec<Vec<C>>,pub dressing:Vec<f64>,pub tail:f64,pub contraction:f64,pub terms:usize,pub milliseconds:f64}

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
            let(g,largest)=far_g(&fold.diagonal,nu,ZERO);let terms=converged_terms(fold.diagnostics.rest_bound*largest);
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
                let mut folded=vec![ZERO;d*d];stored.dense_into(n,0,ONE,&mut folded);stored.dense_into(n,1,C{re:0.,im:nu},&mut folded);
                let scale=target.size;
                let error=exact.iter().zip(&folded).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
                assert!(error<=bound+1e-13*scale,"t {:?} nu {nu}: |exact - folded| {error:e} > bound {bound:e}",target.charge);
                for c in 0..d{let mut t=ZERO;for i in 0..n{add(&mut t,folded[(i*n+i)*d+c]);}assert!(magnitude(t)<1e-13*scale);}
            }
        }
        // The far shell's first ring on a one-mode lattice is K_0 exactly, to roundoff.
        let zero=vec![p.generator.zero.clone()];let nu_of=|q:&[i32]|p.nu(q);
        let mut k0=vec![ZERO;d*d];fold.targets[0].dense_into(n,0,ONE,&mut k0);
        let scale=p.plan.targets[0].size;
        for c in 0..d{let mut e=vec![ZERO;d];e[c]=ONE;let shell=fold.far_shell(&p.generator,&p.plan,&nu_of,&zero,&e,ZERO,0.).unwrap();
            for r in 0..d{let gap=magnitude(C{re:shell.landings[0][r].re-k0[r*d+c].re,im:shell.landings[0][r].im-k0[r*d+c].im});
                assert!(gap<=1e-12*scale,"K_0[{r},{c}]: the far shell's first ring {gap:e} off");}}
    }
    #[test]
    fn lattice_rule_drops_within_its_budget_and_less_for_longer_times(){
        // A run's fold drops pieces only within its budget; at a longer horizon
        // the dropped pieces' norm is no larger (every piece costs more), and
        // every candidate target is kept at a long enough horizon.
        let mut f=fixture();for v in f.far.iter_mut().chain(f.far2.iter_mut()){*v*=1e-3;}
        let p=prepared(&f);
        let mut previous=f64::INFINITY;let mut kept=0;
        for horizon in[1e-3,1.,1e3,1e9]{
            let rule=FoldRule::new(horizon,6.);
            let fold=FoldLibrary::build(&p.generator,&p.plan,&p.lapack,Storage::Rule(rule)).unwrap();
            let dropped=&fold.diagnostics.dropped;
            assert!(dropped.total()<=rule.budget*(1.+1e-12),"T {horizon}: dropped {:e} > budget {:e}",dropped.total(),rule.budget);
            assert!(dropped.kept_targets>=kept,"T {horizon}: fewer targets kept");kept=dropped.kept_targets;
            let norm=dropped.total()/horizon;assert!(norm<=previous*(1.+1e-12),"T {horizon}: the dropped norm grew ({norm:e} > {previous:e})");previous=norm;
            assert_eq!(dropped.candidate_targets,p.plan.targets.len()+p.plan.omitted.len());
        }
        assert_eq!(kept,p.plan.targets.len()+p.plan.omitted.len(),"a long horizon keeps every cross fold");
    }
    #[test]
    fn lattice_ring_targets_are_the_far_shells_higher_rings(){
        // On a one-mode lattice the ring targets (rings 1 and 2) act on every
        // unit element as the far shell's rings 1 and 2 at nu_q = 0.
        let f=fixture();let p=prepared(&f);let n=4;let d=16;let fold=&p.fold;
        let zero=vec![p.generator.zero.clone()];let nu_of=|q:&[i32]|p.nu(q);
        let targets=fold.ring_targets(&p.generator,&p.plan,2,1,&[0.,0.,0.],&zero.iter().cloned().collect()).unwrap();
        assert_eq!(targets.len(),1);let mut k=vec![ZERO;d*d];targets[0].dense_into(n,0,ONE,&mut k);
        let scale=k.iter().map(|z|magnitude(*z)).fold(1e-300,f64::max);
        for c in 0..d{let mut e=vec![ZERO;d];e[c]=ONE;let shell=fold.far_shell(&p.generator,&p.plan,&nu_of,&zero,&e,ZERO,0.).unwrap();
            for r in 0..d{let want=shell.landings.iter().skip(1).take(2).fold(ZERO,|a,l|C{re:a.re+l[r].re,im:a.im+l[r].im});
                // The diagonal ring target is made Hermitian: on unit elements the
                // far shell's own landing is Hermitian to roundoff as well.
                let gap=magnitude(C{re:want.re-k[r*d+c].re,im:want.im-k[r*d+c].im});
                assert!(gap<=1e-10*scale,"ring target [{r},{c}] {gap:e} off ({scale:e})");}}
    }
}
