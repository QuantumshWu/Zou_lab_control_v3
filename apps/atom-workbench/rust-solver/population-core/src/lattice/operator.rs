//! The lattice operator on a set of near charges (modes) and a run's fold
//! (`fold::FoldLibrary`). For every target mode q,
//!   y_q = (L0 - i nu_q + sum_o (i nu_q)^o K_0^(o)) x_q
//!       + sum_{explicit t} L_t x_{q-t}
//!       + sum_{fold targets t != 0} sum_o (i nu_{q-t})^o K_t^(o) x_{q-t},
//! with sources restricted to the set (the Fourier truncation); the far
//! shell's ring targets (`FoldLibrary::ring_targets`) are fold targets of
//! order 0 like the others. The block
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
use super::fold::{FoldLibrary,Rest,StoredFold};
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

/// Rows of a dense sector block per task.
const SECTOR_ROWS:usize=64;

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
/// sources are scaled by (i nu_source)^power for K^(power).
struct Term<'a>{rest:&'a Rest,edges:Vec<(usize,usize)>,power:usize,diagonal:bool}
/// A rest term on the edges one application selects, with its offset in the
/// gathered sources.
struct Selected<'a>{rest:&'a Rest,edges:Vec<(usize,usize)>,power:usize,offset:usize}
/// (i nu)^power.
fn frequency_power(nu:f64,power:usize)->C{let mut s=ONE;for _ in 0..power{s=C{re:-s.im*nu,im:s.re*nu};}s}
pub struct LatticeOperator<'a>{
    pub prepared:&'a LatticePrepared,
    /// The run's fold.
    pub fold:&'a FoldLibrary,
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
}
impl<'a> LatticeOperator<'a>{
    /// The operator on `modes` with the fold `fold` (a run's, or the frame's).
    pub fn new(prepared:&'a LatticePrepared,fold:&'a FoldLibrary,modes:&[Charge])->Result<Self,String>{
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
        let diagonals:Vec<&StoredFold>=fold.targets.iter().filter(|t|t.charge==prepared.generator.zero).collect();
        let(mut left,mut right)=(vec![],vec![]);
        for &w in &nu{
            let(mut a,mut b)=(vec![ZERO;d],vec![ZERO;d]);
            for t in &diagonals{for o in 0..t.orders(){let s=frequency_power(w,o);for k in 0..d{
                mul_add(&mut a[k],s,t.left[o][k]);mul_add(&mut b[k],s,t.right[o][k]);}}}
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
        let mut terms=Vec::new();
        for t in &diagonals{let own:Vec<(usize,usize)>=targets.iter().map(|&q|(q,q)).collect();
            for o in 0..t.orders(){terms.push(Term{rest:t.applied(o),edges:own.clone(),power:o,diagonal:true});}}
        let mut fold_edges=vec![Vec::new();fold.targets.len()];
        for &target in &targets{let q=&modes[target];
            for(e,s)in explicit_steps.iter().enumerate(){
                if let Some(&p)=index.get(&charge_minus(q,&s.charge)?){edges[target].push((p,Edge::Explicit(e)));block_work+=explicit[e].work(n);}
            }
            for(t,data)in fold.targets.iter().enumerate(){
                if data.charge==prepared.generator.zero{continue;}
                if let Some(&p)=index.get(&charge_minus(q,&data.charge)?){edges[target].push((p,Edge::Fold(t)));
                    block_work+=2*(data.orders()+1)*n*n*n;fold_edges[t].push((p,target));}
            }
        }
        for(t,list)in fold_edges.into_iter().enumerate(){if list.is_empty(){continue;}let data=&fold.targets[t];
            for o in 0..data.orders(){terms.push(Term{rest:data.applied(o),edges:list.clone(),power:o,diagonal:false});}}
        terms.retain(|t|t.rest.stored()>0);
        let offset:usize=terms.iter().map(|t|d*t.edges.len()).sum();
        let rest_work:usize=terms.iter().map(|t|t.rest.stored()*t.edges.len()).sum();
        Ok(Self{prepared,fold,n,modes:modes.to_vec(),nu,centre,index,mirror,canonical,left,right,recycling,explicit,edges,terms,
            gathered:vec![ZERO;offset],interleaved:vec![ZERO;d*modes.len()],execution:Execution::new()?,scratch:Vec::new(),
            block_work,rest_work,work:block_work+rest_work})
    }
    pub fn dimension(&self)->usize{self.modes.len()*self.n*self.n}
    pub fn nu_max(&self)->f64{self.nu.iter().map(|w|w.abs()).fold(0.,f64::max)}
    /// Complex multiply-adds of one application: the blocks and edges, the
    /// fold rests.
    pub fn work_split(&self)->[usize;2]{[self.block_work,self.rest_work]}
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
        let Self{prepared,fold,left,right,recycling,explicit,edges,terms,gathered,interleaved,execution,scratch,block_work,rest_work,nu,canonical,mirror,..}=self;
        let gemm=prepared.lapack.complex_gemm();let fold=&fold.targets;let canonical=&*canonical;let mirror=&*mirror;
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
                            let data=&fold[t];
                            grow(&mut scratch.left,d);grow(&mut scratch.right,d);
                            scratch.left[..d].fill(ZERO);scratch.right[..d].fill(ZERO);
                            for o in 0..data.orders(){let w=frequency_power(nu[p],o);for k in 0..d{
                                mul_add(&mut scratch.left[k],w,data.left[o][k]);mul_add(&mut scratch.right[k],w,data.right[o][k]);}}
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
            let count=chosen.len();active.push(Selected{rest:t.rest,edges:chosen,power:t.power,offset});offset+=d*count;
        }
        if !active.is_empty(){
            interleaved.fill(ZERO);grow(gathered,offset);
            // Dense sector blocks: gather each block's sources, GEMM by row chunks.
            let mut items=Vec::new();let mut sources=Vec::new();
            for(k,term)in active.iter().enumerate(){let count=term.edges.len();
                for(b,block)in term.rest.blocks.iter().enumerate(){
                    let mut xs=vec![ZERO;block.cols.len()*count];
                    for(e,&(p,_))in term.edges.iter().enumerate(){let s=frequency_power(nu[p],term.power);
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
                for(e,&(p,_))in term.edges.iter().enumerate(){let s=frequency_power(nu[p],term.power);
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
    /// The adjoint in the real inner product Re <x, y> of Hermitian-symmetric
    /// fields (every mode, weight one): per canonical target q (its mirror's
    /// share folded in, weight 2) every block, edge and rest of the forward map
    /// conjugate-transposed and accumulated on the sources, then the
    /// Hermitian-symmetric part (x + J x) / 2. Serial.
    pub fn apply_adjoint(&self,y:&[C])->Result<Vec<C>,String>{
        let n=self.n;let d=n*n;let modes=self.modes.len();
        if y.len()!=self.dimension(){return Err("Lattice operator field dimensions differ".into());}
        let gemm=self.prepared.lapack.complex_gemm();let fold=&self.fold.targets;
        let ct=|m:&[C]|->Vec<C>{let mut o=vec![ZERO;d];for i in 0..n{for j in 0..n{o[i+n*j]=conj(m[j+n*i]);}}o};
        let adj=|s:&Sparse|->Sparse{let mut e:Vec<(usize,usize,C)>=s.entries.iter().map(|&(i,j,v)|(j,i,conj(v))).collect();e.sort_by_key(|x|(x.0,x.1));Sparse{entries:e}};
        let adjop=|o:&Superop|->Superop{Superop{left:adj(&o.left),right:adj(&o.right),sandwich:o.sandwich.iter().map(|(p,q)|(adj(p),adj(q))).collect()}};
        let recycling=adjop(&self.recycling);let explicit:Vec<Superop>=self.explicit.iter().map(|o|adjop(o)).collect();
        let mut z=vec![ZERO;y.len()];let mut scratch=Vec::new();
        let weight=|q:usize|if self.mirror[q]==q{1.}else{2.};
        for q in 0..modes{if !self.canonical[q]{continue;}
            let yq:Vec<C>=y[q*d..(q+1)*d].iter().map(|v|scaled(*v,weight(q))).collect();
            // Column-major view Z: forward Z L + R Z, adjoint Z L^H + R^H Z.
            let(la,ra)=(ct(&self.left[q]),ct(&self.right[q]));
            let mut t=vec![ZERO;d];gemm.multiply(n,n,n,ONE,&yq,n,&la,n,ZERO,&mut t,n)?;gemm.multiply(n,n,n,ONE,&ra,n,&yq,n,ONE,&mut t,n)?;
            recycling.apply_into(n,ONE,&yq,&mut t,&mut scratch);
            for(a,b)in z[q*d..(q+1)*d].iter_mut().zip(&t){add(a,*b);}
            for &(p,edge)in &self.edges[q]{
                let mut t=vec![ZERO;d];
                match edge{
                    Edge::Explicit(e)=>explicit[e].apply_into(n,ONE,&yq,&mut t,&mut scratch),
                    Edge::Fold(f)=>{let data=&fold[f];
                        let(mut l,mut r)=(vec![ZERO;d],vec![ZERO;d]);
                        for o in 0..data.orders(){let w=frequency_power(self.nu[p],o);
                            for k in 0..d{mul_add(&mut l[k],w,data.left[o][k]);mul_add(&mut r[k],w,data.right[o][k]);}}
                        let(la,ra)=(ct(&l),ct(&r));
                        gemm.multiply(n,n,n,ONE,&yq,n,&la,n,ONE,&mut t,n)?;gemm.multiply(n,n,n,ONE,&ra,n,&yq,n,ONE,&mut t,n)?;}
                }
                for(a,b)in z[p*d..(p+1)*d].iter_mut().zip(&t){add(a,*b);}
            }
        }
        for term in &self.terms{
            for &(p,q)in &term.edges{
                let s=conj(frequency_power(self.nu[p],term.power));let wq=weight(q);
                for b in &term.rest.blocks{let h=b.rows.len();
                    for(c,&col)in b.cols.iter().enumerate(){let mut acc=ZERO;
                        for(r,&row)in b.rows.iter().enumerate(){mul_add(&mut acc,conj(b.values[r+h*c]),y[q*d+row as usize]);}
                        add(&mut z[p*d+col as usize],scaled(mul(s,acc),wq));}}
                let m=&term.rest.sparse;
                for r in 0..d{for at in m.row[r]as usize..m.row[r+1]as usize{
                    add(&mut z[p*d+m.col[at]as usize],scaled(mul(s,mul(conj(m.value[at]),y[q*d+r])),wq));}}
            }
        }
        // The Hermitian-symmetric part.
        let mut out=vec![ZERO;z.len()];
        for q in 0..modes{let m=self.mirror[q];for i in 0..n{for j in 0..n{let(u,v)=(z[q*d+i*n+j],conj(z[m*d+j*n+i]));out[q*d+i*n+j]=C{re:0.5*(u.re+v.re),im:0.5*(u.im+v.im)};}}}
        Ok(out)
    }
    /// What the operator leaves out of the far shell on a field x at the
    /// shift z, to all orders: the far shell's exact landing on every mode
    /// (`FoldLibrary::far_shell`, rings until the next one can land within
    /// `tolerance`, largest entry) less the fold the operator applies there
    /// (every order and rest as the rule keeps it), not mirror-completed (at a
    /// complex z the response of a Hermitian-symmetric field is not
    /// Hermitian-symmetric); with the far shell's rings.
    pub fn left_out_exact(&self,x:&[C],z:C,tolerance:f64)->Result<(Vec<C>,super::fold::FarShell),String>{
        let(n,d)=(self.n,self.n*self.n);let prepared=self.prepared;
        let nu_of=|q:&[i32]|prepared.nu(q);
        let shell=self.fold.far_shell(&prepared.generator,&prepared.plan,&nu_of,&self.modes,x,z,tolerance)?;
        let mut y=vec![ZERO;self.dimension()];for ring in &shell.landings{for(a,&b)in y.iter_mut().zip(ring){add(a,b);}}
        for q in 0..self.modes.len(){
            let out=&mut y[q*d..(q+1)*d];
            for data in &self.fold.targets{
                let Ok(source)=charge_minus(&self.modes[q],&data.charge)else{continue};
                let Some(&p)=self.index.get(&source)else{continue};
                let xp=&x[p*d..(p+1)*d];
                for o in 0..data.orders(){let w=frequency_power(self.nu[p],o);let s=C{re:-w.re,im:-w.im};
                    absorbed_into(&data.left[o],&data.right[o],n,s,xp,out);data.applied(o).apply_into(s,xp,out);}
            }
        }
        Ok((y,shell))
    }
    /// The operator as a dense row-major matrix (tests).
    #[cfg(test)]
    pub fn dense(&self)->Vec<C>{
        let n=self.n;let d=n*n;let size=self.dimension();let mut out=vec![ZERO;size*size];
        let generator=&self.prepared.generator;let fold=&self.fold.targets;
        let mut block=vec![ZERO;d*d];
        let place=|out:&mut Vec<C>,block:&[C],q:usize,p:usize|{for r in 0..d{for c in 0..d{add(&mut out[(q*d+r)*size+p*d+c],block[r*d+c]);}}};
        // sum_o (i w)^o K^(o) as the operator applies them.
        let dense_fold=|t:&StoredFold,w:f64,block:&mut Vec<C>|{for o in 0..t.orders(){t.dense_into(n,o,frequency_power(w,o),block);}};
        for q in 0..self.modes.len(){
            block.fill(ZERO);generator.zero_component().op.dense_into(n,ONE,&mut block);for r in 0..d{block[r*d+r].im-=self.nu[q];}
            for t in fold.iter().filter(|t|t.charge==generator.zero){dense_fold(t,self.nu[q],&mut block);}
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
        let mut op=LatticeOperator::new(&p,&p.fold,&modes).unwrap();let size=op.dimension();let d=16;
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
    fn lattice_operator_adjoint_matches_the_forward_map_and_the_dense_adjoint(){
        // Re <y, A x> = Re <A^H y, x> on Hermitian-symmetric fields, to
        // roundoff; and A^H y is the Hermitian-symmetric part of the dense
        // conjugate transpose applied to y (the adjoint on the J-real fields).
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let mut op=LatticeOperator::new(&p,&p.fold,&modes).unwrap();let(n,d,size)=(op.n,16,op.dimension());
        assert!(p.fold.targets.iter().any(|t|t.orders()>1)&&op.terms.iter().any(|t|t.power>0)&&op.terms.iter().any(|t|!t.diagonal),"orders in nu_q and cross-fold rests");
        let dense=op.dense();
        for(a,b)in[(1,2),(4,7),(9,3)]{
            let(x,y)=(symmetric_field(&op,a),symmetric_field(&op,b));
            let mut ax=vec![ZERO;size];op.apply(&x,&mut ax).unwrap();let ay=op.apply_adjoint(&y).unwrap();
            let dot=|u:&[C],v:&[C]|u.iter().zip(v).map(|(s,t)|s.re*t.re+s.im*t.im).sum::<f64>();
            let(l,r)=(dot(&y,&ax),dot(&ay,&x));let scale=dot(&ax,&ax).sqrt()*dot(&y,&y).sqrt();
            assert!((l-r).abs()<=1e-13*scale,"<y,Ax> {l:e} against <A^H y,x> {r:e} (scale {scale:e})");
            let mut z=vec![ZERO;size];for c in 0..size{for r in 0..size{mul_add(&mut z[c],conj(dense[r*size+c]),y[r]);}}
            let mut sym=vec![ZERO;size];
            for q in 0..op.modes.len(){let m=op.mirror[q];for i in 0..n{for j in 0..n{let(u,v)=(z[q*d+i*n+j],conj(z[m*d+j*n+i]));sym[q*d+i*n+j]=C{re:0.5*(u.re+v.re),im:0.5*(u.im+v.im)};}}}
            let top=sym.iter().map(|z|magnitude(*z)).fold(0.,f64::max);
            let gap=sym.iter().zip(&ay).map(|(u,v)|magnitude(C{re:u.re-v.re,im:u.im-v.im})).fold(0.,f64::max);
            assert!(gap<=1e-13*top,"against the dense adjoint {gap:e} of {top:e}");
        }
    }
    #[test]
    fn lattice_far_shell_is_the_exact_elimination_and_left_out_its_rest(){
        // The far shell's first ring on every fold edge p -> q is the exact
        // second-order elimination of the far modes at the source's frequency,
        // sum over the pairs of -C_b (L0 - i (nu_s + nu_p))^-1 C_s from dense LU
        // solves; `left_out_exact` is every ring less the fold the operator
        // applies (its dense assembly).
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let op=LatticeOperator::new(&p,&p.fold,&modes).unwrap();let(n,d)=(op.n,op.n*op.n);let size=op.dimension();
        let x=symmetric_field(&op,5);let(left,shell)=op.left_out_exact(&x,ZERO,0.).unwrap();
        let dense=|q:&[i32]|{let mut m=vec![ZERO;d*d];p.generator.components[q].op.dense_into(n,ONE,&mut m);m};
        let mut l0=vec![ZERO;d*d];p.generator.zero_component().op.dense_into(n,ONE,&mut l0);
        let lap=&p.lapack;let(mut second,mut applied)=(vec![ZERO;size],vec![ZERO;size]);
        for q in 0..op.modes.len(){
            for target in p.plan.targets.iter().chain(&p.plan.omitted){
                let Ok(source)=charge_minus(&op.modes[q],&target.charge)else{continue};let Some(&s)=op.index.get(&source)else{continue};
                for pair in &target.pairs{let step=&p.plan.steps[pair.source];
                    let mut block=l0.clone();for i in 0..d{block[i*d+i].im-=step.nu+op.nu[s];}
                    let c=dense(&step.charge);let cx:Vec<C>=(0..d).map(|r|(0..d).fold(ZERO,|v,k|{let mut v=v;mul_add(&mut v,c[r*d+k],x[s*d+k]);v})).collect();
                    let t=|m:&[C]|(0..d*d).map(|e|m[(e%d)*d+e/d]).collect::<Vec<C>>();
                    let(lu,piv)=lap.complex_factor(&t(&block),d).unwrap();let y=lap.complex_solve_matrix_factored(&lu,&piv,&cx,d).unwrap();
                    let b=dense(&p.plan.steps[pair.back].charge);
                    for r in 0..d{let mut v=ZERO;for k in 0..d{mul_add(&mut v,b[r*d+k],y[k]);}second[q*d+r].re-=v.re;second[q*d+r].im-=v.im;}}
            }
            for data in &p.fold.targets{
                let Ok(source)=charge_minus(&op.modes[q],&data.charge)else{continue};let Some(&s)=op.index.get(&source)else{continue};
                let mut folded=vec![ZERO;d*d];for o in 0..data.orders(){data.dense_into(n,o,frequency_power(op.nu[s],o),&mut folded);}
                for r in 0..d{let mut v=ZERO;for k in 0..d{mul_add(&mut v,folded[r*d+k],x[s*d+k]);}applied[q*d+r]=C{re:applied[q*d+r].re+v.re,im:applied[q*d+r].im+v.im};}
            }
        }
        let scale=x.iter().map(|z|magnitude(*z)).fold(0.,f64::max)*p.fold.targets.iter().map(|t|t.norms[0][0]).fold(0.,f64::max);
        let gap=|a:&[C],b:&[C]|a.iter().zip(b).map(|(u,v)|magnitude(C{re:u.re-v.re,im:u.im-v.im})).fold(0.,f64::max);
        assert!(gap(&shell.landings[0],&second)<=1e-12*scale,"the first ring {:e} from the exact second order (scale {scale:e})",gap(&shell.landings[0],&second));
        let mut rings=vec![ZERO;size];for ring in &shell.landings{for(a,&b)in rings.iter_mut().zip(ring){add(a,b);}}
        let expected:Vec<C>=rings.iter().zip(&applied).map(|(a,b)|C{re:a.re-b.re,im:a.im-b.im}).collect();
        assert!(gap(&left,&expected)<=1e-12*scale,"left out {:e} from every ring less the fold (scale {scale:e})",gap(&left,&expected));
    }
    #[test]
    fn lattice_block_traces_are_exact_left_eigenvectors(){
        // B's foundation (`space`): every coupling is a commutator and every
        // fold term keeps its population rows, so tau_q(A x) = -i nu_q
        // tau_q(x) on every block, for any field, on the lattice and with what
        // the operator leaves out added.
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let mut op=LatticeOperator::new(&p,&p.fold,&modes).unwrap();let(n,d)=(op.n,op.n*op.n);
        let x=symmetric_field(&op,3);let mut y=vec![ZERO;op.dimension()];op.apply(&x,&mut y).unwrap();
        let(rest,_)=op.left_out_exact(&x,ZERO,0.).unwrap();for(a,b)in y.iter_mut().zip(rest){add(a,b);}
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
        let mut op=LatticeOperator::new(&p,&p.fold,&modes).unwrap();let size=op.dimension();let d=16;
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
