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
    /// The fold alone: sum_t (K_t^(0) + i nu_p K_t^(1)) x_p on every fold edge.
    Fold,
}
#[derive(Default)]
pub struct OperatorScratch{small:Vec<C>,left:Vec<C>,right:Vec<C>}
/// One stored fold rest on all of its lattice edges (source, target);
/// sources are scaled by i nu_source for K^(1).
struct Term<'a>{rest:&'a Rest,edges:Vec<(usize,usize)>,frequency:bool,diagonal:bool,offset:usize}
pub struct LatticeOperator<'a>{
    pub prepared:&'a LatticePrepared,
    pub n:usize,pub modes:Vec<Charge>,pub nu:Vec<f64>,pub centre:usize,
    pub index:BTreeMap<Charge,usize>,
    /// The mode of -q, and whether q is computed (the others are mirrored).
    pub mirror:Vec<usize>,pub canonical:Vec<bool>,
    /// Per mode: row-major L0's left/right factors with -i nu_q and the
    /// diagonal fold's absorbable part, and the absorbable part alone.
    left:Vec<Vec<C>>,right:Vec<Vec<C>>,fold_left:Vec<Vec<C>>,fold_right:Vec<Vec<C>>,
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
    /// Edges per kind (canonical targets): explicit steps, cross folds.
    pub edge_counts:[usize;2],
    /// Summed row sums of the K^(1) rests left out (0 when they are applied).
    pub dropped_k1_rest:f64,
}
impl<'a> LatticeOperator<'a>{
    pub fn new(prepared:&'a LatticePrepared,modes:&[Charge])->Result<Self,String>{
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
        let(mut left,mut right,mut fold_left,mut fold_right)=(vec![],vec![],vec![],vec![]);
        for &w in &nu{
            let(mut a,mut b)=(vec![ZERO;d],vec![ZERO;d]);
            if let Some(t)=diagonal{for k in 0..d{
                a[k]=t.left[0][k];mul_add(&mut a[k],C{re:0.,im:w},t.left[1][k]);
                b[k]=t.right[0][k];mul_add(&mut b[k],C{re:0.,im:w},t.right[1][k]);}}
            let mut l:Vec<C>=nl.iter().zip(&a).map(|(x,y)|C{re:x.re+y.re,im:x.im+y.im}).collect();
            let r:Vec<C>=nr.iter().zip(&b).map(|(x,y)|C{re:x.re+y.re,im:x.im+y.im}).collect();
            for i in 0..n{l[i*n+i].im-=w;}
            left.push(l);right.push(r);fold_left.push(a);fold_right.push(b);
        }
        let recycling=Superop{left:Sparse::default(),right:Sparse::default(),sandwich:zero.op.sandwich.clone()};
        let explicit_steps:Vec<&classification::HarmonicStep>=prepared.plan.explicit().into_iter().map(|i|&prepared.plan.steps[i]).collect();
        let explicit:Vec<&Superop>=explicit_steps.iter().map(|s|&prepared.generator.components[&s.charge].op).collect();
        let targets:Vec<usize>=(0..modes.len()).filter(|&q|canonical[q]).collect();
        let mut edges=vec![Vec::new();modes.len()];let mut edge_counts=[0;2];
        let mut block_work=targets.len()*(2*n*n*n+recycling.work(n));
        let nu_max=nu.iter().map(|w|w.abs()).fold(0.,f64::max);
        let k1_rest:f64=prepared.fold.targets.iter().map(|t|t.rest_row_sum[1]).sum();
        let fold_size:f64=prepared.fold.targets.iter().map(|t|t.norms[0][0]).sum();
        let dropped_k1_rest=if nu_max*k1_rest<=K1_REST_SHARE*fold_size{k1_rest}else{0.};
        let k1=|t:&'a StoredFold|if dropped_k1_rest>0.{&t.population_rest[1]}else{&t.rest[1]};
        let mut terms=Vec::new();
        if let Some(t)=diagonal{let own:Vec<(usize,usize)>=targets.iter().map(|&q|(q,q)).collect();
            terms.push(Term{rest:&t.rest[0],edges:own.clone(),frequency:false,diagonal:true,offset:0});
            terms.push(Term{rest:k1(t),edges:own,frequency:true,diagonal:true,offset:0});}
        let mut fold_edges=vec![Vec::new();prepared.fold.targets.len()];
        for &target in &targets{let q=&modes[target];
            for(e,s)in explicit_steps.iter().enumerate(){
                if let Some(&p)=index.get(&charge_minus(q,&s.charge)?){edges[target].push((p,Edge::Explicit(e)));edge_counts[0]+=1;block_work+=explicit[e].work(n);}
            }
            for(t,data)in prepared.fold.targets.iter().enumerate(){
                if data.charge==prepared.generator.zero{continue;}
                if let Some(&p)=index.get(&charge_minus(q,&data.charge)?){edges[target].push((p,Edge::Fold(t)));edge_counts[1]+=1;
                    block_work+=4*n*n*n;fold_edges[t].push((p,target));}
            }
        }
        for(t,list)in fold_edges.into_iter().enumerate(){if list.is_empty(){continue;}let data=&prepared.fold.targets[t];
            terms.push(Term{rest:&data.rest[0],edges:list.clone(),frequency:false,diagonal:false,offset:0});
            terms.push(Term{rest:k1(data),edges:list,frequency:true,diagonal:false,offset:0});}
        terms.retain(|t|t.rest.stored()>0);
        let mut offset=0;for t in &mut terms{t.offset=offset;offset+=d*t.edges.len();}
        let rest_work:usize=terms.iter().map(|t|t.rest.stored()*t.edges.len()).sum();
        Ok(Self{prepared,n,modes:modes.to_vec(),nu,centre,index,mirror,canonical,left,right,fold_left,fold_right,recycling,explicit,edges,terms,
            gathered:vec![ZERO;offset],interleaved:vec![ZERO;d*modes.len()],execution:Execution::new()?,scratch:Vec::new(),
            block_work,rest_work,work:block_work+rest_work,edge_counts,dropped_k1_rest})
    }
    pub fn dimension(&self)->usize{self.modes.len()*self.n*self.n}
    pub fn nu_max(&self)->f64{self.nu.iter().map(|w|w.abs()).fold(0.,f64::max)}
    pub fn canonical_count(&self)->usize{self.canonical.iter().filter(|&&c|c).count()}
    /// Dropped fold entries (largest row sums of K^(0) and K^(1)), the
    /// K^(1) rests left out included.
    pub fn dropped(&self)->[f64;2]{let[a,b]=self.prepared.fold.dropped();[a,b+self.dropped_k1_rest]}
    /// y = A x for a Hermitian-symmetric field x (x_-q = x_q^dagger).
    pub fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{self.apply_parts(x,y,Parts::Full)}
    /// y = (selected parts of A) x for a Hermitian-symmetric field x.
    pub fn apply_parts(&mut self,x:&[C],y:&mut[C],parts:Parts)->Result<(),String>{
        let n=self.n;let d=n*n;let modes=self.modes.len();
        if x.len()!=self.dimension()||y.len()!=x.len(){return Err("Lattice operator field dimensions differ".into());}
        let Self{prepared,left,right,fold_left,fold_right,recycling,explicit,edges,terms,gathered,interleaved,execution,scratch,block_work,rest_work,nu,canonical,mirror,..}=self;
        let gemm=prepared.lapack.complex_gemm();let fold=&prepared.fold.targets;let canonical=&*canonical;
        execution.for_mode_chunks(y,d,*block_work,MINIMUM_WORK_PER_TASK,scratch,|first,chunk,scratch|{
            for(m,yq)in chunk.chunks_exact_mut(d).enumerate(){
                let q=first+m;if !canonical[q]{continue;}let xq=&x[q*d..(q+1)*d];
                // Column-major views are transposes: y^T = X^T NL^T + NR^T X^T.
                let(l,r)=if parts==Parts::Fold{(&fold_left[q],&fold_right[q])}else{(&left[q],&right[q])};
                gemm.multiply(n,n,n,ONE,xq,n,l,n,ZERO,yq,n)?;
                gemm.multiply(n,n,n,ONE,r,n,xq,n,ONE,yq,n)?;
                if parts!=Parts::Fold{recycling.apply_into(n,ONE,xq,yq,&mut scratch.small);}
                if parts==Parts::Diagonal{continue;}
                for &(p,edge)in &edges[q]{
                    let xp=&x[p*d..(p+1)*d];
                    match edge{
                        Edge::Explicit(e)=>if parts==Parts::Full{explicit[e].apply_into(n,ONE,xp,yq,&mut scratch.small)},
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
        let active:Vec<&Term>=terms.iter().filter(|t|parts!=Parts::Diagonal||t.diagonal).collect();
        if !active.is_empty(){
            interleaved.fill(ZERO);
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
                let(term,block)=(active[k],&active[k].rest.blocks[b]);let count=term.edges.len();let(h,w)=(end-start,block.cols.len());
                let mut out=vec![ZERO;h*count];
                gemm.multiply(h,count,w,ONE,&block.values[start..],block.rows.len(),&sources[s],w,ZERO,&mut out,h)?;Ok(out)});
            for(&(k,b,_,start,end),out)in items.iter().zip(parts_out){let out=out?;let(term,block)=(active[k],&active[k].rest.blocks[b]);let h=end-start;
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
            for r in 0..d{for q in 0..modes{if canonical[q]{add(&mut y[q*d+r],interleaved[r*modes+q]);}}}
        }
        // y_-q = y_q^dagger, the centre Hermitian.
        mirror_field(y,n,canonical,mirror);
        Ok(())
    }
    /// The fold part alone on a Hermitian-symmetric field.
    pub fn apply_fold(&mut self,x:&[C])->Result<Vec<C>,String>{let mut y=vec![ZERO;self.dimension()];self.apply_parts(x,&mut y,Parts::Fold)?;Ok(y)}
    /// The operator as a dense row-major matrix (validation only).
    pub fn dense(&self)->Vec<C>{
        let n=self.n;let d=n*n;let size=self.dimension();let mut out=vec![ZERO;size*size];
        let generator=&self.prepared.generator;let fold=&self.prepared.fold.targets;
        let mut block=vec![ZERO;d*d];
        let place=|out:&mut Vec<C>,block:&[C],q:usize,p:usize|{for r in 0..d{for c in 0..d{add(&mut out[(q*d+r)*size+p*d+c],block[r*d+c]);}}};
        let dense_fold=|t:&StoredFold,w:f64,block:&mut Vec<C>|{t.dense_into(n,0,ONE,block);t.dense_into(n,1,C{re:0.,im:w},block);};
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
        assert!(op.edge_counts[0]>0&&op.edge_counts[1]>0,"explicit and cross-fold edges");
        assert_eq!(op.canonical_count(),3,"centre and one of each pair");
        let x=symmetric_field(&op,0);
        let mut y=vec![ZERO;size];op.apply(&x,&mut y).unwrap();
        let dense=op.dense();let mut scale=0_f64;let mut error=0_f64;
        for r in 0..size{let mut v=ZERO;for c in 0..size{mul_add(&mut v,dense[r*size+c],x[c]);}
            scale=scale.max(magnitude(v));error=error.max(magnitude(C{re:v.re-y[r].re,im:v.im-y[r].im}));}
        assert!(error<=1e-13*scale,"{error:e} of {scale:e}");
        let mut y2=vec![ZERO;size];op.apply(&x,&mut y2).unwrap();
        assert!(y.iter().zip(&y2).all(|(a,b)|a.re.to_bits()==b.re.to_bits()&&a.im.to_bits()==b.im.to_bits()),"repeatable");
        // The diagonal blocks and the fold alone against the explicit components.
        let(mut diagonal,mut small)=(vec![ZERO;size],Vec::new());op.apply_parts(&x,&mut diagonal,Parts::Diagonal).unwrap();
        let fold=op.apply_fold(&x).unwrap();let mut explicit=vec![ZERO;size];
        for(q,mode)in modes.iter().enumerate(){for(t,c)in &p.generator.components{
            if p.plan.steps.iter().any(|s|&s.charge==t&&s.treatment==classification::Treatment::Fold){continue;}
            let Ok(source)=charge_minus(mode,t)else{continue};let Some(&s)=op.index.get(&source)else{continue};
            c.op.apply_into(4,ONE,&x[s*d..(s+1)*d],&mut explicit[q*d..(q+1)*d],&mut small);}
            for k in 0..d{explicit[q*d+k].im-=op.nu[q]*x[q*d+k].re;explicit[q*d+k].re+=op.nu[q]*x[q*d+k].im;}}
        let error=y.iter().zip(&explicit).zip(&fold).map(|((a,b),c)|magnitude(C{re:a.re-b.re-c.re,im:a.im-b.im-c.im})).fold(0.,f64::max);
        assert!(error<=1e-13*scale,"fold part {error:e} of {scale:e}");
        let mut blocks=vec![ZERO;size];
        for q in 0..modes.len(){let mut z=vec![ZERO;size];z[q*d..(q+1)*d].copy_from_slice(&x[q*d..(q+1)*d]);
            for r in 0..d{let mut v=ZERO;for c in 0..d{mul_add(&mut v,dense[(q*d+r)*size+q*d+c],z[q*d+c]);}blocks[q*d+r]=v;}}
        let error=diagonal.iter().zip(&blocks).map(|(a,b)|magnitude(C{re:a.re-b.re,im:a.im-b.im})).fold(0.,f64::max);
        assert!(error<=1e-13*scale,"diagonal part {error:e} of {scale:e}");
    }
}
