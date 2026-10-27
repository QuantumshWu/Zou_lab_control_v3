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
//!   D_s^-1 = sum_{k <= m_s} (-G_s L_r)^k G_s + R_s,
//!   |R_s| <= |G_s| rho_s^(m_s+1) / (1 - rho_s),  rho_s = |L_r| |G_s|
//! (Schur-test norms). Far blocks sit beyond 2 |L0| (classification), so
//! rho_s is small (about 1e-6 for the tweezer, 1e-7 for counter-rotating
//! terms) and each step's order m_s comes from its own bound: its pairs'
//! remainders below SERIES_FLOOR of their targets' sizes, D_s^-2 included,
//! far below the storage floor. Each K column c is then a short chain of
//! sparse factor applications on one block, C_s e_c, G, L_r, ..., C_s',
//! columns spread over the workers.
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
//! others a CSR remainder above ENTRY_FLOOR times the largest entry.
//! Population rows are never dropped, so the stored fold stays exactly trace
//! preserving; the dropped entries' largest row sum is charged.
use super::*;
use super::classification::HarmonicPlan;
use super::operator::LatticeOperator;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Sparse-remainder entries below this fraction of a target's largest entry
/// are dropped (and charged). The dark states amplify dropped coherence
/// entries: at 1e-9 probe + tweezer populations move by 1e-4, at 1e-12 by
/// 3e-8. Dense sector blocks drop nothing.
const ENTRY_FLOOR:f64=1e-12;
/// A sector pair with at least this share of its entries above the floor is
/// stored as a dense block.
const DENSE_FILL:f64=0.2;
/// Each far block's Neumann remainder, relative to its targets' sizes.
const SERIES_FLOOR:f64=1e-16;
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
    /// Far steps solved (the rest by pairing) and their active columns.
    pub far_sources:usize,pub columns:usize,
    /// |L_r| (Schur-test bound, rad/us) and per far step (charge, rho_s, m_s).
    pub rest_bound:f64,pub series:Vec<(Charge,f64,usize)>,
    pub milliseconds:f64,
    /// Milliseconds: the far columns, the storage.
    pub phase_ms:[f64;2],
}
/// One far block's inverse (L0 - i nu_s)^-1 as its truncated Neumann series.
struct FarBlock{order:usize,
    /// G_s = (L_c - i nu_s)^-1, elementwise on a row-major block.
    g:Vec<C>}
pub struct FoldLibrary{
    pub targets:Vec<StoredFold>,
    /// K_0^(0) as a row-major d x d matrix (the static-shift test hook).
    pub zero_dense:Option<Vec<C>>,
    /// L_r and every far step's block.
    rest:Superop,blocks:BTreeMap<usize,FarBlock>,
    pub diagnostics:FoldDiagnostics,
}
#[inline]fn squared(z:C)->f64{z.re*z.re+z.im*z.im}
fn frobenius(a:&[C])->f64{a.iter().map(|z|squared(*z)).sum::<f64>().sqrt()}

/// Whether input coordinate c = (k, l) has a nonzero image under `op`
/// (structurally; an exactly cancelling image just folds to zero).
fn column_active(op:&Superop,n:usize,c:usize)->bool{
    let(k,l)=(c/n,c%n);
    op.left.entries.iter().any(|e|e.1==k)||op.right.entries.iter().any(|e|e.0==l)
        ||op.sandwich.iter().any(|(p,q)|p.entries.iter().any(|e|e.1==k)&&q.entries.iter().any(|e|e.0==l))
}
/// The superoperator index of the transposed element: (i, j) -> (j, i).
#[inline]fn swapped(p:usize,n:usize)->usize{(p%n)*n+p/n}
/// K + flip(K) in place (row-major d x d).
fn add_flip(k:&mut[C],n:usize){
    let d=n*n;
    for r in 0..d{for c in 0..d{let(a,b)=(r*d+c,swapped(r,n)*d+swapped(c,n));
        if a<b{let v=C{re:k[a].re+k[b].re,im:k[a].im-k[b].im};k[a]=v;k[b]=conj(v);}else if a==b{k[a]=C{re:2.*k[a].re,im:0.};}}}
}
/// flip(K) (row-major d x d).
fn flipped(k:&[C],n:usize)->Vec<C>{
    let d=n*n;(0..d*d).map(|e|conj(k[swapped(e/d,n)*d+swapped(e%d,n)])).collect()
}

#[derive(Default)]
struct BuildScratch{y:Vec<C>,z:Vec<C>,t:Vec<C>,w:Vec<C>,small:Vec<C>}

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

/// x <- sum_{k <= m_s} (-G_s L_r)^k G_s x on one block (w, t scratch).
fn series(rest:&Superop,block:&FarBlock,n:usize,x:&mut[C],w:&mut[C],t:&mut[C],small:&mut Vec<C>){
    for(v,&g)in x.iter_mut().zip(&block.g){*v=mul(*v,g);}
    w.copy_from_slice(x);
    for _ in 0..block.order{
        t.fill(ZERO);rest.apply_into(n,MINUS_ONE,w,t,small);
        for((v,&g),&u)in w.iter_mut().zip(&block.g).zip(t.iter()){*v=mul(u,g);}
        for(a,&b)in x.iter_mut().zip(w.iter()){add(a,b);}
    }
}
/// One stored order of a target, as `store` returns it.
type StoredOrder=(Vec<C>,Vec<C>,Rest,f64,[f64;3],f64,f64,Rest);

impl FoldLibrary{
    pub fn build(generator:&Generator,plan:&HarmonicPlan)->Result<Self,String>{
        let clock=Instant::now();let n=generator.n;let d=n*n;let zero=generator.zero_component();
        // L_c (the factors' diagonals) and L_r.
        let(mut left,mut right)=(vec![ZERO;n],vec![ZERO;n]);let(mut left_rest,mut right_rest)=(Vec::new(),Vec::new());
        for &(i,k,a)in &zero.op.left.entries{if i==k{add(&mut left[i],a);}else{left_rest.push((i,k,a));}}
        for &(k,j,b)in &zero.op.right.entries{if k==j{add(&mut right[j],b);}else{right_rest.push((k,j,b));}}
        let rest=Superop{left:Sparse{entries:left_rest},right:Sparse{entries:right_rest},sandwich:zero.op.sandwich.clone()};
        let rest_bound=rest.norm_bound(n);
        let mut diagnostics=FoldDiagnostics{rest_bound,..Default::default()};
        if plan.targets.is_empty(){return Ok(Self{targets:vec![],zero_dense:None,rest,blocks:BTreeMap::new(),diagnostics});}
        // Every far block: G_s, rho_s and the order from its pairs' bounds
        // (c_s c_s' |G_s| rho^(m+1) against SERIES_FLOOR of the target's size,
        // with 2 / (1 - rho) for D_s^-2).
        let mut blocks=BTreeMap::new();
        for s in plan.far(){
            let step=&plan.steps[s];
            let g:Vec<C>=(0..d).map(|p|{let z=C{re:left[p/n].re+right[p%n].re,im:left[p/n].im+right[p%n].im-step.nu};let m=squared(z);C{re:z.re/m,im:-z.im/m}}).collect();
            let largest=g.iter().map(|z|magnitude(*z)).fold(0.,f64::max);let rho=rest_bound*largest;
            if !(rho<SERIES_LIMIT){return Err(format!("Far step {:?} is too close to L0's spectrum to fold (series ratio {rho:e})",step.charge));}
            let need=plan.targets.iter().flat_map(|t|t.pairs.iter().filter(|p|p.source==s).map(move|p|plan.steps[p.back].size/t.size.max(f64::MIN_POSITIVE))).fold(0.,f64::max);
            let scale=step.size*largest*need*2./(1.-rho).powi(2);
            let mut order=0;while scale*rho.powi(order as i32+1)>SERIES_FLOOR&&order<64{order+=1;}
            diagnostics.series.push((step.charge.clone(),rho,order));
            blocks.insert(s,FarBlock{order,g});
        }
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
            let op=&generator.components[&plan.steps[s].charge].op;let block=&blocks[&s];
            let active:Vec<bool>=(0..d).map(|c|column_active(op,n,c)).collect();
            let count=active.iter().filter(|&&a|a).count();diagnostics.columns+=count;diagnostics.far_sources+=1;
            let backs:Vec<&Superop>=partners.iter().map(|&(_,b)|&generator.components[&plan.steps[b].charge].op).collect();
            let work=count*((2*block.order+1)*rest.work(n)+backs.iter().map(|b|2*b.work(n)).sum::<usize>()+op.work(n));
            let(rest,active,backs)=(&rest,&active,&backs);
            execution.for_mode_chunks(&mut build,width,work,MINIMUM_WORK_PER_TASK,&mut scratch,|first,chunk,scratch|{
                let BuildScratch{y,z,t,w,small}=scratch;for v in[&mut *y,&mut *z,&mut *t,&mut *w]{grow(v,d);}
                let(y,z,t,w)=(&mut y[..d],&mut z[..d],&mut t[..d],&mut w[..d]);
                for(k,column)in chunk.chunks_exact_mut(width).enumerate(){let c=first+k;if !active[c]{continue;}
                    // Y = D_s^-1 C_s e_c and Z = D_s^-1 Y.
                    t.fill(ZERO);t[c]=ONE;y.fill(ZERO);op.apply_into(n,ONE,t,y,small);
                    series(rest,block,n,y,w,t,small);
                    z.copy_from_slice(y);series(rest,block,n,z,w,t,small);
                    for(&(slot,_),back)in partners.iter().zip(backs){let base=2*slot*d;
                        back.apply_into(n,MINUS_ONE,y,&mut column[base..base+d],small);
                        back.apply_into(n,MINUS_ONE,z,&mut column[base+d..base+2*d],small);}
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
            let mut k=vec![ZERO;d*d];let offset=(2*slot+order)*d;
            for cb in(0..d).step_by(64){for rb in(0..d).step_by(64){for c in cb..(cb+64).min(d){let column=&build[c*width+offset..c*width+offset+d];
                for r in rb..(rb+64).min(d){k[r*d+c]=column[r];}}}}
            if flip{k=flipped(&k,n);}else if target.charge==generator.zero{add_flip(&mut k,n);}
            let zero=(order==0&&target.charge==generator.zero).then(||k.clone());
            Ok((store(&mut k,n,&generator.ground),zero))
        });
        let mut parts=stored.into_iter();let mut targets=Vec::with_capacity(plan.targets.len());let mut zero_dense=None;
        for target in &plan.targets{
            let mut fold=StoredFold{charge:target.charge.clone(),nu:target.nu,left:[vec![],vec![]],right:[vec![],vec![]],rest:[Rest::default(),Rest::default()],
                population_rest:[Rest::default(),Rest::default()],dropped:[0.;2],norms:[[0.;3];2],largest_entry:[0.;2],rest_row_sum:[0.;2]};
            for order in 0..2{
                let((a,b,rest,lost,norms,largest,row,population),zero)=parts.next().ok_or("Fold storage lost a target")??;
                if zero.is_some(){zero_dense=zero;}
                fold.left[order]=a;fold.right[order]=b;fold.rest[order]=rest;fold.dropped[order]=lost;fold.norms[order]=norms;
                fold.largest_entry[order]=largest;fold.rest_row_sum[order]=row;fold.population_rest[order]=population;
            }
            targets.push(fold);
        }
        diagnostics.milliseconds=clock.elapsed().as_secs_f64()*1e3;diagnostics.phase_ms[1]=diagnostics.milliseconds-diagnostics.phase_ms[0];
        Ok(Self{targets,zero_dense,rest,blocks,diagnostics})
    }
    /// D_s^-1 on m consecutive row-major blocks, in place.
    fn far_solve(&self,s:usize,blocks:&mut[C],m:usize)->Result<(),String>{
        let block=self.blocks.get(&s).ok_or("A far step without its block")?;let d=blocks.len()/m.max(1);let n=(d as f64).sqrt().round() as usize;
        let(mut w,mut t,mut small)=(vec![ZERO;d],vec![ZERO;d],Vec::new());
        for x in blocks.chunks_exact_mut(d){series(&self.rest,block,n,x,&mut w,&mut t,&mut small);}
        Ok(())
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
        if self.blocks.is_empty(){return Ok(out);}
        let explicit=plan.explicit();
        let far=plan.far();
        // One task per far step; the parts are summed in step order.
        let steps:Vec<usize>=self.blocks.keys().copied().collect();
        let parts=Execution::new()?.map(steps,|s|->Result<Option<Vec<C>>,String>{
            let mut small=Vec::new();let mut out=vec![ZERO;modes*d];
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
            self.far_solve(s,&mut y,modes)?;
            let slots:Vec<&Charge>=middles.iter().filter(|(_,r)|!r.is_empty()).map(|(c,_)|c).collect();
            let slot=|c:&Charge|slots.binary_search(&c).map_err(|_|"Third-order source middle without its slot".to_string());
            let mut z=vec![ZERO;slots.len()*d];
            for(q,e,r)in &hops{let k=slot(r)?;
                generator.components[&plan.steps[*e].charge].op.apply_into(n,ONE,&y[q*d..(q+1)*d],&mut z[k*d..(k+1)*d],&mut small);}
            self.far_solve(s,&mut z,slots.len())?;
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
    use super::super::floquet::transpose;
    fn dense_l0(p:&LatticePrepared)->Vec<C>{let n=p.generator.n;let d=n*n;let mut m=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut m);m}
    #[test]
    fn lattice_far_block_series_is_the_exact_inverse(){
        // Every far block's truncated Neumann series solves (L0 - i nu_s) X = Y
        // to roundoff, recycling included.
        let f=fixture();let p=prepared(&f);let d=16;let l0=dense_l0(&p);
        assert!(!p.fold.blocks.is_empty()&&p.fold.blocks.values().any(|b|b.order>0));
        for(&s,block)in &p.fold.blocks{
            let nu=p.plan.steps[s].nu;let mut m=l0.clone();for i in 0..d{m[i*d+i].im-=nu;}
            let y:Vec<C>=(0..2*d).map(|k|C{re:((k*37+11)%97)as f64/97.-0.5,im:((k*53+7)%89)as f64/89.-0.5}).collect();
            let mut x=y.clone();p.fold.far_solve(s,&mut x,2).unwrap();
            let scale=y.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            for b in 0..2{let mut r=y[b*d..(b+1)*d].to_vec();for i in 0..d{for k in 0..d{let v=mul(m[i*d+k],x[b*d+k]);r[i].re-=v.re;r[i].im-=v.im;}}
                let error=r.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
                assert!(error<1e-13*scale,"step {s}: residual {error:e} (order {})",block.order);}
        }
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
