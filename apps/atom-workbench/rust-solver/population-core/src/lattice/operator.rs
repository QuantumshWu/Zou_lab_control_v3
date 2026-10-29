//! The lattice operator on a set of near charges (modes). For every target
//! mode q,
//!   y_q = (L0 - i nu_q + K_0^(0) + i nu_q K_0^(1)) x_q
//!       + sum_{explicit t} L_t x_{q-t}
//!       + sum_{fold targets t != 0} (K_t^(0) + i nu_{q-t} K_t^(1)) x_{q-t},
//! with sources restricted to the set (the Fourier truncation). The block
//! factor NL'_q X + X NR'_q carries L0's left/right factors, -i nu_q and the
//! diagonal fold's absorbable part as two dense n x n GEMMs; recycling and
//! explicit steps act through their sparse factors. Every fold rest is
//! applied once per application on all of its edges: its dense sector blocks
//! as GEMMs over the gathered sources, its sparse remainder by rows.
//!
//! Masks: an application may read some modes only and compute some targets
//! only (`Mask`): the pieces A_PP, A_QP, A_PQ, A_QQ of a lattice split into
//! its explicit charges and their ring, or one layer of the seed sweep.
//!
//! Hermitian half: the set is closed under q -> -q and the fields the solver
//! applies it to satisfy x_-q = x_q^dagger (the density is Hermitian, and
//! every component of the generator preserves that), so only the canonical
//! half of the targets (the centre and one of each pair) is computed, the
//! rest mirrored and the centre replaced by its Hermitian part
//! (`mirror_field`). Every output entry is accumulated by one task in a fixed
//! order, so results do not depend on the thread count.
use super::*;
use super::fold::{Rest,StoredFold};
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Rows of a dense sector block per task.
const SECTOR_ROWS:usize=64;
/// The K^(1) rests are left out, but for their population rows (which keep
/// the fold exactly trace preserving), and charged as dropped fold entries
/// when nu_max times their summed row sums is below this fraction of the
/// fold's size: about half of a research model's operator work.
const K1_REST_SHARE:f64=1e-3;

#[derive(Clone,Copy,Debug)]
enum Edge{Explicit(usize),Fold(usize)}
/// Which parts of the operator an application includes.
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Parts{
    /// The whole lattice operator.
    Full,
    /// The mode-diagonal blocks only (no edge between modes).
    Diagonal,
}
/// A restriction of an application: which modes are read (the others are
/// taken as zero) and which are computed (the others come out zero).
#[derive(Clone,Copy,Debug)]
pub struct Mask<'m>{pub targets:&'m[bool],pub sources:&'m[bool]}
#[derive(Default)]
pub struct OperatorScratch{small:Vec<C>,left:Vec<C>,right:Vec<C>}
/// One stored fold rest on all of its lattice edges (source, target);
/// sources are scaled by i nu_source for K^(1).
struct Term<'a>{rest:&'a Rest,edges:Vec<(usize,usize)>,frequency:bool,diagonal:bool}
/// A rest term on the edges one application selects, with its offset in the
/// gathered sources.
struct Selected<'a>{rest:&'a Rest,edges:Vec<(usize,usize)>,frequency:bool,offset:usize}
pub struct LatticeOperator<'a>{
    pub prepared:&'a LatticePrepared,
    pub n:usize,pub modes:Vec<Charge>,pub nu:Vec<f64>,pub centre:usize,
    pub index:BTreeMap<Charge,usize>,
    /// The mode of -q, and whether q is computed (the others are mirrored).
    pub mirror:Vec<usize>,pub canonical:Vec<bool>,
    /// Per mode: row-major L0's left/right factors with -i nu_q and the
    /// diagonal fold's absorbable part.
    left:Vec<Vec<C>>,right:Vec<Vec<C>>,
    recycling:Superop,
    explicit:Vec<&'a Superop>,
    /// Per canonical target: (source mode, edge kind).
    edges:Vec<Vec<(usize,Edge)>>,
    terms:Vec<Term<'a>>,
    /// Gathered sparse-term sources ([c][edge] per term) and the rest output
    /// interleaved by mode ([row][mode]).
    gathered:Vec<C>,interleaved:Vec<C>,
    execution:Execution,
    scratch:Vec<OperatorScratch>,
    /// Complex multiply-adds of one application: per-block part, rests.
    block_work:usize,rest_work:usize,
    pub work:usize,
    /// Whether the K^(1) rests are applied but for their population rows
    /// (the rest is left out: `left_out`), and their summed row sums; whether
    /// anything of the folded generator is left out (`left_out`).
    pub drops_k1_rest:bool,k1_rest:f64,leaves_out:bool,
}
impl<'a> LatticeOperator<'a>{
    pub fn new(prepared:&'a LatticePrepared,modes:&[Charge])->Result<Self,String>{Self::with_k1_rests(prepared,modes,false)}
    /// The operator with every K^(1) rest kept when `keep_k1` (a completed
    /// lattice: `pipeline`'s completion).
    pub fn with_k1_rests(prepared:&'a LatticePrepared,modes:&[Charge],keep_k1:bool)->Result<Self,String>{
        let n=prepared.generator.n;let d=n*n;
        let index:BTreeMap<Charge,usize>=modes.iter().enumerate().map(|(i,q)|(q.clone(),i)).collect();
        if index.len()!=modes.len(){return Err("Lattice modes must be distinct".into());}
        let centre=*index.get(&prepared.generator.zero).ok_or("Lattice modes must contain the zero charge")?;
        let mirror=modes.iter().map(|q|index.get(&opposite(q)).copied().ok_or("Lattice modes must be closed under q -> -q"))
            .collect::<Result<Vec<_>,_>>()?;
        let canonical:Vec<bool>=(0..modes.len()).map(|q|modes[q]>=modes[mirror[q]]).collect();
        let nu=modes.iter().map(|q|prepared.nu(q)).collect::<Result<Vec<_>,_>>()?;
        let zero=prepared.generator.zero_component();
        let(nl,nr)=(zero.op.left.dense(n),zero.op.right.dense(n));
        let diagonal=prepared.fold.targets.first().filter(|t|t.charge==prepared.generator.zero);
        let(mut left,mut right)=(vec![],vec![]);
        for &w in &nu{
            let(mut a,mut b)=(vec![ZERO;d],vec![ZERO;d]);
            if let Some(t)=diagonal{for k in 0..d{
                a[k]=t.left[0][k];mul_add(&mut a[k],C{re:0.,im:w},t.left[1][k]);
                b[k]=t.right[0][k];mul_add(&mut b[k],C{re:0.,im:w},t.right[1][k]);}}
            let mut l:Vec<C>=nl.iter().zip(&a).map(|(x,y)|C{re:x.re+y.re,im:x.im+y.im}).collect();
            let r:Vec<C>=nr.iter().zip(&b).map(|(x,y)|C{re:x.re+y.re,im:x.im+y.im}).collect();
            for i in 0..n{l[i*n+i].im-=w;}
            left.push(l);right.push(r);
        }
        let recycling=Superop{left:Sparse::default(),right:Sparse::default(),sandwich:zero.op.sandwich.clone()};
        let explicit_steps:Vec<&classification::HarmonicStep>=prepared.plan.explicit().into_iter().map(|i|&prepared.plan.steps[i]).collect();
        let explicit:Vec<&Superop>=explicit_steps.iter().map(|s|&prepared.generator.components[&s.charge].op).collect();
        let targets:Vec<usize>=(0..modes.len()).filter(|&q|canonical[q]).collect();
        let mut edges=vec![Vec::new();modes.len()];
        let mut block_work=targets.len()*(2*n*n*n+recycling.work(n));
        let nu_max=nu.iter().map(|w|w.abs()).fold(0.,f64::max);
        let k1_rest:f64=prepared.fold.targets.iter().filter(|t|t.kept[1]).map(|t|t.rest_row_sum[1]).sum();
        let fold_size:f64=prepared.fold.targets.iter().map(|t|t.norms[0][0]).sum();
        let drops_k1_rest=k1_rest>0.&&nu_max*k1_rest<=K1_REST_SHARE*fold_size&&!keep_k1;
        // A non-structural rest is applied but for its population rows.
        let k0=|t:&'a StoredFold|t.applied(0);
        let k1=|t:&'a StoredFold|if drops_k1_rest{&t.population_rest[1]}else{t.applied(1)};
        let leaves_out=!(prepared.fold.targets.is_empty()&&prepared.plan.omitted.is_empty());
        let mut terms=Vec::new();
        if let Some(t)=diagonal{let own:Vec<(usize,usize)>=targets.iter().map(|&q|(q,q)).collect();
            terms.push(Term{rest:k0(t),edges:own.clone(),frequency:false,diagonal:true});
            terms.push(Term{rest:k1(t),edges:own,frequency:true,diagonal:true});}
        let mut fold_edges=vec![Vec::new();prepared.fold.targets.len()];
        for &target in &targets{let q=&modes[target];
            for(e,s)in explicit_steps.iter().enumerate(){
                if let Some(&p)=index.get(&charge_minus(q,&s.charge)?){edges[target].push((p,Edge::Explicit(e)));block_work+=explicit[e].work(n);}
            }
            for(t,data)in prepared.fold.targets.iter().enumerate(){
                if data.charge==prepared.generator.zero{continue;}
                if let Some(&p)=index.get(&charge_minus(q,&data.charge)?){edges[target].push((p,Edge::Fold(t)));
                    block_work+=4*n*n*n;fold_edges[t].push((p,target));}
            }
        }
        for(t,list)in fold_edges.into_iter().enumerate(){if list.is_empty(){continue;}let data=&prepared.fold.targets[t];
            terms.push(Term{rest:k0(data),edges:list.clone(),frequency:false,diagonal:false});
            terms.push(Term{rest:k1(data),edges:list,frequency:true,diagonal:false});}
        terms.retain(|t|t.rest.stored()>0);
        let offset:usize=terms.iter().map(|t|d*t.edges.len()).sum();
        let rest_work:usize=terms.iter().map(|t|t.rest.stored()*t.edges.len()).sum();
        Ok(Self{prepared,n,modes:modes.to_vec(),nu,centre,index,mirror,canonical,left,right,recycling,explicit,edges,terms,
            gathered:vec![ZERO;offset],interleaved:vec![ZERO;d*modes.len()],execution:Execution::new()?,scratch:Vec::new(),
            block_work,rest_work,work:block_work+rest_work,drops_k1_rest,k1_rest,leaves_out})
    }
    pub fn dimension(&self)->usize{self.modes.len()*self.n*self.n}
    pub fn nu_max(&self)->f64{self.nu.iter().map(|w|w.abs()).fold(0.,f64::max)}
    /// Complex multiply-adds of one application: the blocks and edges, the
    /// fold rests.
    pub fn work_split(&self)->[usize;2]{[self.block_work,self.rest_work]}
    /// The fold remainders of this lattice as generator-norm bounds.
    pub fn ledger(&self)->Result<super::ledger::RemainderLedger,String>{
        super::ledger::RemainderLedger::new(&self.prepared.plan,self.nu_max(),self.prepared.fold.dropped(),if self.drops_k1_rest{self.k1_rest}else{0.})
    }
    /// y = A x for a Hermitian-symmetric field x (x_-q = x_q^dagger).
    pub fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{self.apply_parts(x,y,Parts::Full)}
    /// y = (selected parts of A) x for a Hermitian-symmetric field x.
    pub fn apply_parts(&mut self,x:&[C],y:&mut[C],parts:Parts)->Result<(),String>{self.apply_masked(x,y,parts,None)}
    /// y = (selected parts of A) x restricted by a mask: the sources outside
    /// `mask.sources` are read as zero and skipped, only the targets in
    /// `mask.targets` (with their mirrors) are computed, the others are zero.
    pub fn apply_masked(&mut self,x:&[C],y:&mut[C],parts:Parts,mask:Option<Mask<'_>>)->Result<(),String>{
        let n=self.n;let d=n*n;let modes=self.modes.len();
        if x.len()!=self.dimension()||y.len()!=x.len(){return Err("Lattice operator field dimensions differ".into());}
        if mask.is_some_and(|m|m.targets.len()!=modes||m.sources.len()!=modes){return Err("Lattice operator mask dimensions differ".into());}
        let Self{prepared,left,right,recycling,explicit,edges,terms,gathered,interleaved,execution,scratch,block_work,rest_work,nu,canonical,mirror,..}=self;
        let gemm=prepared.lapack.complex_gemm();let fold=&prepared.fold.targets;let canonical=&*canonical;let mirror=&*mirror;
        let computed=|q:usize|canonical[q]&&mask.is_none_or(|m|m.targets[q]||m.targets[mirror[q]]);
        let reads=|p:usize|mask.is_none_or(|m|m.sources[p]);
        execution.for_mode_chunks(y,d,*block_work,MINIMUM_WORK_PER_TASK,scratch,|first,chunk,scratch|{
            for(m,yq)in chunk.chunks_exact_mut(d).enumerate(){
                let q=first+m;if !canonical[q]{continue;}
                if !computed(q){yq.fill(ZERO);continue;}
                if reads(q){
                    let xq=&x[q*d..(q+1)*d];
                    // Column-major views are transposes: y^T = X^T NL^T + NR^T X^T.
                    gemm.multiply(n,n,n,ONE,xq,n,&left[q],n,ZERO,yq,n)?;
                    gemm.multiply(n,n,n,ONE,&right[q],n,xq,n,ONE,yq,n)?;
                    recycling.apply_into(n,ONE,xq,yq,&mut scratch.small);
                }else{yq.fill(ZERO);}
                if parts==Parts::Diagonal{continue;}
                for &(p,edge)in &edges[q]{
                    if !reads(p){continue;}
                    let xp=&x[p*d..(p+1)*d];
                    match edge{
                        Edge::Explicit(e)=>explicit[e].apply_into(n,ONE,xp,yq,&mut scratch.small),
                        Edge::Fold(t)=>{
                            let data=&fold[t];let w=C{re:0.,im:nu[p]};
                            grow(&mut scratch.left,d);grow(&mut scratch.right,d);
                            for k in 0..d{scratch.left[k]=data.left[0][k];mul_add(&mut scratch.left[k],w,data.left[1][k]);
                                scratch.right[k]=data.right[0][k];mul_add(&mut scratch.right[k],w,data.right[1][k]);}
                            gemm.multiply(n,n,n,ONE,xp,n,&scratch.left[..d],n,ONE,yq,n)?;
                            gemm.multiply(n,n,n,ONE,&scratch.right[..d],n,xp,n,ONE,yq,n)?;
                        }
                    }
                }
            }
            Ok(())
        })?;
        // The rest terms on the edges the application selects, gathered at
        // local offsets.
        let mut active:Vec<Selected<'_>>=Vec::new();let mut offset=0;
        for t in terms.iter().filter(|t|parts!=Parts::Diagonal||t.diagonal){
            let chosen:Vec<(usize,usize)>=t.edges.iter().copied().filter(|&(p,q)|reads(p)&&computed(q)).collect();
            if chosen.is_empty(){continue;}
            let count=chosen.len();active.push(Selected{rest:t.rest,edges:chosen,frequency:t.frequency,offset});offset+=d*count;
        }
        if !active.is_empty(){
            interleaved.fill(ZERO);grow(gathered,offset);
            // Dense sector blocks: gather each block's sources, GEMM by row chunks.
            let mut items=Vec::new();let mut sources=Vec::new();
            for(k,term)in active.iter().enumerate(){let count=term.edges.len();
                for(b,block)in term.rest.blocks.iter().enumerate(){
                    let mut xs=vec![ZERO;block.cols.len()*count];
                    for(e,&(p,_))in term.edges.iter().enumerate(){let s=if term.frequency{C{re:0.,im:nu[p]}}else{ONE};
                        for(c,&col)in block.cols.iter().enumerate(){xs[c+block.cols.len()*e]=mul(s,x[p*d+col as usize]);}}
                    sources.push(xs);
                    for start in(0..block.rows.len()).step_by(SECTOR_ROWS){items.push((k,b,sources.len()-1,start,(start+SECTOR_ROWS).min(block.rows.len())));}
                }
            }
            let parts_out=execution.map(items.clone(),|(k,b,s,start,end)|->Result<Vec<C>,String>{
                let(term,block)=(&active[k],&active[k].rest.blocks[b]);let count=term.edges.len();let(h,w)=(end-start,block.cols.len());
                let mut out=vec![ZERO;h*count];
                gemm.multiply(h,count,w,ONE,&block.values[start..],block.rows.len(),&sources[s],w,ZERO,&mut out,h)?;Ok(out)});
            for(&(k,b,_,start,end),out)in items.iter().zip(parts_out){let out=out?;let(term,block)=(&active[k],&active[k].rest.blocks[b]);let h=end-start;
                for(e,&(_,q))in term.edges.iter().enumerate(){for i in 0..h{add(&mut interleaved[block.rows[start+i]as usize*modes+q],out[i+h*e]);}}}
            // Sparse remainders, by rows.
            for term in &active{let count=term.edges.len();
                for(e,&(p,_))in term.edges.iter().enumerate(){let s=if term.frequency{C{re:0.,im:nu[p]}}else{ONE};
                    for c in 0..d{gathered[term.offset+c*count+e]=mul(s,x[p*d+c]);}}}
            let gathered=&*gathered;let active=&active;
            execution.for_mode_chunks(interleaved,modes,*rest_work,MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
                for term in active{let count=term.edges.len();let g=&gathered[term.offset..term.offset+d*count];let m=&term.rest.sparse;
                    for(i,out)in chunk.chunks_exact_mut(modes).enumerate(){let r=first+i;
                        for at in m.row[r]as usize..m.row[r+1]as usize{let v=m.value[at];let xs=&g[m.col[at]as usize*count..(m.col[at]as usize+1)*count];
                            for(&(_,q),&xv)in term.edges.iter().zip(xs){mul_add(&mut out[q],v,xv);}}
                    }
                }
                Ok(())
            })?;
            for r in 0..d{for q in 0..modes{if computed(q){add(&mut y[q*d+r],interleaved[r*modes+q]);}}}
        }
        // y_-q = y_q^dagger, the centre Hermitian.
        mirror_field(y,n,canonical,mirror);
        Ok(())
    }
    /// Whether the operator leaves K^(1) rests out (the estimate charges them).
    pub fn drops_k1_rests(&self)->bool{self.drops_k1_rest}
    /// What the operator leaves out of the second-order far shell on a
    /// Hermitian-symmetric field x, on every fold edge p -> q of the lattice
    /// (kept and omitted targets): the exact action, sum over the target's
    /// pairs of -C_b (D_s - i nu_p)^-1 C_s x_p (`FoldLibrary::far_responses`),
    /// less the fold the operator applies there, (K^(0) + i nu_p K^(1)) x_p as
    /// stored (the series truncated, entries below the storage floor dropped,
    /// a non-structural rest but its population rows, the K^(1) rests but
    /// their population rows when dropped). None without a far shell.
    pub fn left_out(&self,x:&[C])->Result<Option<Vec<C>>,String>{
        if !self.leaves_out{return Ok(None);}
        let(n,d)=(self.n,self.n*self.n);let mut y=vec![ZERO;self.dimension()];
        let prepared=self.prepared;let(library,plan,generator)=(&prepared.fold,&prepared.plan,&prepared.generator);
        // The far shell's exact parts per source mode.
        let responses:Vec<BTreeMap<usize,Vec<C>>>=self.execution.map((0..self.modes.len()).collect(),
            |p|library.far_responses(generator,plan,self.nu[p],&x[p*d..(p+1)*d])).into_iter().collect::<Result<_,_>>()?;
        let targets:Vec<usize>=(0..self.modes.len()).filter(|&q|self.canonical[q]).collect();
        let(fold,modes,index,nu,drops)=(&library.targets,&self.modes,&self.index,&self.nu,self.drops_k1_rest);
        let responses=&responses;
        let blocks=self.execution.map(targets.clone(),|q|->Result<Vec<C>,String>{
            let mut out=vec![ZERO;d];let mut small=Vec::new();
            for target in plan.targets.iter().chain(&plan.omitted){
                let Some(&p)=index.get(&charge_minus(&modes[q],&target.charge)?)else{continue};
                for pair in &target.pairs{
                    let part=responses[p].get(&pair.source).ok_or("Lattice operator: a fold pair without its far response")?;
                    generator.components[&plan.steps[pair.back].charge].op.apply_into(n,ONE,part,&mut out,&mut small);}
            }
            for data in fold{
                let Some(&p)=index.get(&charge_minus(&modes[q],&data.charge)?)else{continue};
                let xp=&x[p*d..(p+1)*d];
                for(order,s)in[(0,C{re:-1.,im:0.}),(1,C{re:0.,im:-nu[p]})]{
                    absorbed_into(&data.left[order],&data.right[order],n,s,xp,&mut out);
                    let rest=if order==1&&drops{&data.population_rest[1]}else{data.applied(order)};rest.apply_into(s,xp,&mut out);
                }
            }
            Ok(out)});
        for(&q,block)in targets.iter().zip(blocks){y[q*d..(q+1)*d].copy_from_slice(&block?);}
        mirror_field(&mut y,n,&self.canonical,&self.mirror);
        Ok(Some(y))
    }
    /// The operator as a dense row-major matrix (tests).
    #[cfg(test)]
    pub fn dense(&self)->Vec<C>{
        let n=self.n;let d=n*n;let size=self.dimension();let mut out=vec![ZERO;size*size];
        let generator=&self.prepared.generator;let fold=&self.prepared.fold.targets;
        let mut block=vec![ZERO;d*d];
        let place=|out:&mut Vec<C>,block:&[C],q:usize,p:usize|{for r in 0..d{for c in 0..d{add(&mut out[(q*d+r)*size+p*d+c],block[r*d+c]);}}};
        // K^(0) + i w K^(1) as the operator applies them.
        let dense_fold=|t:&StoredFold,w:f64,block:&mut Vec<C>|{t.dense_into(n,0,ONE,block);
            if self.drops_k1_rest{let s=C{re:0.,im:w};Superop{left:Sparse::from_dense(&t.left[1],n),right:Sparse::from_dense(&t.right[1],n),sandwich:vec![]}.dense_into(n,s,block);t.population_rest[1].dense_into(d,s,block);}
            else{t.dense_into(n,1,C{re:0.,im:w},block);}};
        for q in 0..self.modes.len(){
            block.fill(ZERO);generator.zero_component().op.dense_into(n,ONE,&mut block);for r in 0..d{block[r*d+r].im-=self.nu[q];}
            if let Some(t)=fold.first().filter(|t|t.charge==generator.zero){dense_fold(t,self.nu[q],&mut block);}
            place(&mut out,&block,q,q);
            for(t,data)in fold.iter().enumerate(){if data.charge==generator.zero{continue;}
                let Ok(source)=charge_minus(&self.modes[q],&data.charge)else{continue};let Some(&p)=self.index.get(&source)else{continue};
                block.fill(ZERO);dense_fold(&fold[t],self.nu[p],&mut block);place(&mut out,&block,q,p);}
            for(e,s)in self.prepared.plan.explicit().into_iter().enumerate(){let step=&self.prepared.plan.steps[s];
                let Ok(source)=charge_minus(&self.modes[q],&step.charge)else{continue};let Some(&p)=self.index.get(&source)else{continue};
                block.fill(ZERO);self.explicit[e].dense_into(n,ONE,&mut block);place(&mut out,&block,q,p);}
        }
        out
    }
}

/// out += s (A X + X B) for row-major n x n A, B and the block X.
fn absorbed_into(a:&[C],b:&[C],n:usize,s:C,x:&[C],out:&mut[C]){
    for i in 0..n{for j in 0..n{let mut v=ZERO;
        for k in 0..n{mul_add(&mut v,a[i*n+k],x[k*n+j]);mul_add(&mut v,x[i*n+k],b[k*n+j]);}
        mul_add(&mut out[i*n+j],s,v);}}
}

/// A Hermitian-symmetric test field: x_-q = x_q^dagger, the centre Hermitian.
#[cfg(test)]
pub(crate) fn symmetric_field(op:&LatticeOperator<'_>,seed:usize)->Vec<C>{
    let n=op.n;let d=n*n;let mut x:Vec<C>=(0..op.dimension()).map(|k|C{re:(((k+seed)*7+3)%29)as f64/29.-0.5,im:(((k+seed)*11+5)%31)as f64/31.-0.5}).collect();
    for q in 0..op.modes.len(){if op.canonical[q]&&q!=op.centre{continue;}let m=op.mirror[q];
        for i in 0..n{for j in 0..n{let v=conj(x[m*d+j*n+i]);if q==op.centre&&i>j{continue;}x[q*d+i*n+j]=v;}}}
    let c=op.centre*d;for i in 0..n{x[c+i*n+i].im=0.;for j in 0..i{x[c+i*n+j]=conj(x[c+j*n+i]);}}
    x
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_operator_matches_its_dense_assembly_on_hermitian_fields(){
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let mut op=LatticeOperator::new(&p,&modes).unwrap();let size=op.dimension();let d=16;
        assert!(op.edges.iter().flatten().any(|e|matches!(e.1,Edge::Explicit(_)))&&op.edges.iter().flatten().any(|e|matches!(e.1,Edge::Fold(_))),"explicit and cross-fold edges");
        assert_eq!(op.canonical.iter().filter(|&&c|c).count(),3,"centre and one of each pair");
        let x=symmetric_field(&op,0);
        let mut y=vec![ZERO;size];op.apply(&x,&mut y).unwrap();
        let dense=op.dense();let mut scale=0_f64;let mut error=0_f64;
        for r in 0..size{let mut v=ZERO;for c in 0..size{mul_add(&mut v,dense[r*size+c],x[c]);}
            scale=scale.max(magnitude(v));error=error.max(magnitude(C{re:v.re-y[r].re,im:v.im-y[r].im}));}
        assert!(error<=1e-13*scale,"{error:e} of {scale:e}");
        let mut y2=vec![ZERO;size];op.apply(&x,&mut y2).unwrap();
        assert!(y.iter().zip(&y2).all(|(a,b)|a.re.to_bits()==b.re.to_bits()&&a.im.to_bits()==b.im.to_bits()),"repeatable");
        // The diagonal blocks against the dense assembly.
        let mut diagonal=vec![ZERO;size];op.apply_parts(&x,&mut diagonal,Parts::Diagonal).unwrap();
        let mut blocks=vec![ZERO;size];
        for q in 0..modes.len(){let mut z=vec![ZERO;size];z[q*d..(q+1)*d].copy_from_slice(&x[q*d..(q+1)*d]);
            for r in 0..d{let mut v=ZERO;for c in 0..d{mul_add(&mut v,dense[(q*d+r)*size+q*d+c],z[q*d+c]);}blocks[q*d+r]=v;}}
        let error=diagonal.iter().zip(&blocks).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
        assert!(error<=1e-13*scale,"diagonal part {error:e} of {scale:e}");
    }
    #[test]
    fn lattice_left_out_is_the_exact_far_shell_less_the_applied_fold(){
        // On every fold edge p -> q: the exact second-order elimination of the
        // far modes at the source's frequency, sum over the pairs of
        // -C_b (L0 - i (nu_s + nu_p))^-1 C_s from dense LU solves, less the fold
        // the operator applies (its dense assembly), is what `left_out` returns.
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let op=LatticeOperator::new(&p,&modes).unwrap();let(n,d)=(op.n,op.n*op.n);let size=op.dimension();
        let x=symmetric_field(&op,5);let left=op.left_out(&x).unwrap().expect("a far shell");
        let dense=|q:&[i32]|{let mut m=vec![ZERO;d*d];p.generator.components[q].op.dense_into(n,ONE,&mut m);m};
        let mut l0=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut l0);
        let lap=&p.lapack;let mut expected=vec![ZERO;size];
        for q in 0..op.modes.len(){if !op.canonical[q]{continue;}
            for target in p.plan.targets.iter().chain(&p.plan.omitted){
                let Ok(source)=charge_minus(&op.modes[q],&target.charge)else{continue};let Some(&s)=op.index.get(&source)else{continue};
                for pair in &target.pairs{let step=&p.plan.steps[pair.source];
                    let mut block=l0.clone();for i in 0..d{block[i*d+i].im-=step.nu+op.nu[s];}
                    let c=dense(&step.charge);let cx:Vec<C>=(0..d).map(|r|(0..d).fold(ZERO,|v,k|{let mut v=v;mul_add(&mut v,c[r*d+k],x[s*d+k]);v})).collect();
                    let t=|m:&[C]|(0..d*d).map(|e|m[(e%d)*d+e/d]).collect::<Vec<C>>();
                    let(lu,piv)=lap.complex_factor(&t(&block),d).unwrap();let y=lap.complex_solve_matrix_factored(&lu,&piv,&cx,d).unwrap();
                    let b=dense(&p.plan.steps[pair.back].charge);
                    for r in 0..d{let mut v=ZERO;for k in 0..d{mul_add(&mut v,b[r*d+k],y[k]);}expected[q*d+r].re-=v.re;expected[q*d+r].im-=v.im;}}
            }
            for data in &p.fold.targets{
                let Ok(source)=charge_minus(&op.modes[q],&data.charge)else{continue};let Some(&s)=op.index.get(&source)else{continue};
                let mut folded=vec![ZERO;d*d];data.dense_into(n,0,ONE,&mut folded);data.dense_into(n,1,C{re:0.,im:op.nu[s]},&mut folded);
                for r in 0..d{let mut v=ZERO;for k in 0..d{mul_add(&mut v,folded[r*d+k],x[s*d+k]);}expected[q*d+r].re-=v.re;expected[q*d+r].im-=v.im;}
            }
        }
        let scale=x.iter().map(|z|magnitude(*z)).fold(0.,f64::max)*p.fold.targets.iter().map(|t|t.norms[0][0]).fold(0.,f64::max);
        let error=(0..op.modes.len()).filter(|&q|op.canonical[q]).flat_map(|q|(0..d).map(move|r|q*d+r)).map(|e|magnitude(C{re:left[e].re-expected[e].re,im:left[e].im-expected[e].im})).fold(0.,f64::max);
        assert!(error<=1e-12*scale,"left out {error:e} from the exact far shell less the fold (scale {scale:e})");
    }
    #[test]
    fn lattice_block_traces_are_exact_left_eigenvectors(){
        // B's foundation (`space`): every coupling is a commutator and every
        // fold term keeps its population rows, so tau_q(A x) = -i nu_q
        // tau_q(x) on every block, for any field, on the lattice and with what
        // the operator leaves out added.
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let mut op=LatticeOperator::new(&p,&modes).unwrap();let(n,d)=(op.n,op.n*op.n);
        let x=symmetric_field(&op,3);let mut y=vec![ZERO;op.dimension()];op.apply(&x,&mut y).unwrap();
        if let Some(rest)=op.left_out(&x).unwrap(){for(a,b)in y.iter_mut().zip(rest){add(a,b);}}
        let trace=|v:&[C],q:usize|(0..n).fold(ZERO,|s,i|C{re:s.re+v[q*d+i*n+i].re,im:s.im+v[q*d+i*n+i].im});
        let scale=y.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
        for q in 0..op.modes.len(){let(ty,tx)=(trace(&y,q),trace(&x,q));let defect=C{re:ty.re-op.nu[q]*tx.im,im:ty.im+op.nu[q]*tx.re};
            assert!(magnitude(defect)<=1e-13*scale,"block {q}: {:e} of {scale:e}",magnitude(defect));}
    }
    #[test]
    fn lattice_operator_masks_restrict_sources_and_targets(){
        // A masked application equals the full one on the field restricted to
        // the sources, read on the targets (both closed under q -> -q).
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let mut op=LatticeOperator::new(&p,&modes).unwrap();let size=op.dimension();let d=16;
        let x=symmetric_field(&op,3);let count=modes.len();
        let inner:Vec<bool>=modes.iter().map(|q|q.iter().map(|k|k.unsigned_abs()).sum::<u32>()<=1).collect();
        let outer:Vec<bool>=inner.iter().map(|&b|!b).collect();
        for parts in[Parts::Full,Parts::Diagonal]{for(targets,sources)in[(&outer,&inner),(&inner,&outer),(&outer,&outer)]{
            let mut restricted=x.clone();for q in 0..count{if !sources[q]{restricted[q*d..(q+1)*d].fill(ZERO);}}
            let mut full=vec![ZERO;size];op.apply_parts(&restricted,&mut full,parts).unwrap();
            for q in 0..count{if !targets[q]{full[q*d..(q+1)*d].fill(ZERO);}}
            let mut masked=vec![ZERO;size];op.apply_masked(&x,&mut masked,parts,Some(Mask{targets,sources})).unwrap();
            let scale=full.iter().map(|z|magnitude(*z)).fold(1e-300,f64::max);
            let error=full.iter().zip(&masked).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
            assert!(error<=1e-13*scale,"{parts:?}: {error:e} of {scale:e}");
        }}
    }
}
