//! The lattice operator on a set of near charges (modes). For every target
//! mode q,
//!   y_q = (L0 - i nu_q + K_0^(0) + i nu_q K_0^(1)) x_q
//!       + sum_{explicit t} L_t x_{q-t}
//!       + sum_{fold targets t != 0} (K_t^(0) + i nu_{q-t} K_t^(1)) x_{q-t},
//! with sources restricted to the set (the Fourier truncation). The block
//! factor NL'_q X + X NR'_q carries L0's left/right factors, -i nu_q and the
//! diagonal fold's absorbable part as two dense n x n GEMMs; recycling and
//! explicit steps act through their sparse factors, and every fold rest (CSR)
//! is applied once per application on all of its edges, so each stored
//! matrix is read once. Block work is split over the worker pool by target
//! modes, CSR work by rows; every output entry is accumulated by one task in
//! a fixed order, so results do not depend on the thread count. Every mode
//! of the set is evaluated (the Hermitian symmetry x_-q = x_q^dagger is not
//! used to halve the work).
use super::*;
use crate::parallel::{Execution,MINIMUM_WORK_PER_TASK};

#[derive(Clone,Copy,Debug)]
enum Edge{Explicit(usize),Fold(usize)}
#[derive(Default)]
pub struct OperatorScratch{small:Vec<C>,left:Vec<C>,right:Vec<C>}
/// One stored CSR matrix (a fold rest) on all of its
/// lattice edges (source, target), sources scaled by i nu_source for K^(1).
struct Term<'a>{matrix:&'a fold::Csr,edges:Vec<(usize,usize)>,frequency:bool,offset:usize}
pub struct LatticeOperator<'a>{
    pub prepared:&'a LatticePrepared,
    pub n:usize,pub modes:Vec<Charge>,pub nu:Vec<f64>,pub centre:usize,
    pub index:BTreeMap<Charge,usize>,
    /// Per mode: row-major NL'_q and NR'_q.
    left:Vec<Vec<C>>,right:Vec<Vec<C>>,
    recycling:Superop,
    explicit:Vec<&'a Superop>,
    edges:Vec<Vec<(usize,Edge)>>,
    terms:Vec<Term<'a>>,
    /// Gathered term sources ([c][edge] per term) and the CSR output
    /// interleaved by mode ([row][mode]).
    gathered:Vec<C>,interleaved:Vec<C>,
    execution:Execution,
    scratch:Vec<OperatorScratch>,
    /// Complex multiply-adds of one application: per-block part, CSR part.
    block_work:usize,term_work:usize,
    pub work:usize,
    /// Edges per kind: explicit steps, cross folds.
    pub edge_counts:[usize;2],
    /// Stored CSR entries applied per application (with their edges).
    pub csr_work:usize,
}
impl<'a> LatticeOperator<'a>{
    pub fn new(prepared:&'a LatticePrepared,modes:&[Charge])->Result<Self,String>{
        let n=prepared.generator.n;let d=n*n;
        let index:BTreeMap<Charge,usize>=modes.iter().enumerate().map(|(i,q)|(q.clone(),i)).collect();
        if index.len()!=modes.len(){return Err("Lattice modes must be distinct".into());}
        let centre=*index.get(&prepared.generator.zero).ok_or("Lattice modes must contain the zero charge")?;
        let nu=modes.iter().map(|q|prepared.nu(q)).collect::<Result<Vec<_>,_>>()?;
        let zero=prepared.generator.zero_component();
        let(nl,nr)=(zero.op.left.dense(n),zero.op.right.dense(n));
        let diagonal=prepared.fold.targets.first().filter(|t|t.charge==prepared.generator.zero);
        let mut left=Vec::with_capacity(modes.len());let mut right=Vec::with_capacity(modes.len());
        for &w in &nu{
            let(mut l,mut r)=(nl.clone(),nr.clone());
            if let Some(t)=diagonal{for k in 0..d{
                add(&mut l[k],t.left[0][k]);add(&mut l[k],mul(C{re:0.,im:w},t.left[1][k]));
                add(&mut r[k],t.right[0][k]);add(&mut r[k],mul(C{re:0.,im:w},t.right[1][k]));}}
            for i in 0..n{l[i*n+i].im-=w;}
            left.push(l);right.push(r);
        }
        let recycling=Superop{left:Sparse::default(),right:Sparse::default(),sandwich:zero.op.sandwich.clone()};
        let explicit_steps:Vec<&classification::HarmonicStep>=prepared.plan.explicit().into_iter().map(|i|&prepared.plan.steps[i]).collect();
        let explicit:Vec<&Superop>=explicit_steps.iter().map(|s|&prepared.generator.components[&s.charge].op).collect();
        let mut edges=vec![Vec::new();modes.len()];let mut edge_counts=[0;2];
        let mut block_work=modes.len()*(2*n*n*n+recycling.work(n));
        let mut terms=Vec::new();
        if let Some(t)=diagonal{let all:Vec<(usize,usize)>=(0..modes.len()).map(|q|(q,q)).collect();
            terms.push(Term{matrix:&t.rest[0],edges:all.clone(),frequency:false,offset:0});
            terms.push(Term{matrix:&t.rest[1],edges:all,frequency:true,offset:0});}
        let mut fold_edges=vec![Vec::new();prepared.fold.targets.len()];
        for(target,q)in modes.iter().enumerate(){
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
            terms.push(Term{matrix:&data.rest[0],edges:list.clone(),frequency:false,offset:0});
            terms.push(Term{matrix:&data.rest[1],edges:list,frequency:true,offset:0});}
        terms.retain(|t|t.matrix.nonzeros()>0);
        let mut offset=0;for t in &mut terms{t.offset=offset;offset+=d*t.edges.len();}
        let term_work:usize=terms.iter().map(|t|t.matrix.nonzeros()*t.edges.len()).sum();
        Ok(Self{prepared,n,modes:modes.to_vec(),nu,centre,index,left,right,recycling,explicit,edges,terms,
            gathered:vec![ZERO;offset],interleaved:vec![ZERO;d*modes.len()],execution:Execution::new()?,scratch:Vec::new(),
            block_work,term_work,work:block_work+term_work,edge_counts,csr_work:term_work})
    }
    pub fn dimension(&self)->usize{self.modes.len()*self.n*self.n}
    pub fn nu_max(&self)->f64{self.nu.iter().map(|w|w.abs()).fold(0.,f64::max)}
    /// y = A x on the mode set: the per-block part by target modes, then
    /// every stored CSR on all of its edges at once (each matrix is read
    /// once per application), rows split over the workers.
    pub fn apply(&mut self,x:&[C],y:&mut[C])->Result<(),String>{
        let n=self.n;let d=n*n;let modes=self.modes.len();
        if x.len()!=self.dimension()||y.len()!=x.len(){return Err("Lattice operator field dimensions differ".into());}
        let Self{prepared,left,right,recycling,explicit,edges,terms,gathered,interleaved,execution,scratch,block_work,term_work,nu,..}=self;
        let gemm=prepared.lapack.complex_gemm();let fold=&prepared.fold.targets;
        execution.for_mode_chunks(y,d,*block_work,MINIMUM_WORK_PER_TASK,scratch,|first,chunk,scratch|{
            for(m,yq)in chunk.chunks_exact_mut(d).enumerate(){
                let q=first+m;let xq=&x[q*d..(q+1)*d];
                // Column-major views are transposes: y^T = X^T NL^T + NR^T X^T.
                gemm.multiply(n,n,n,ONE,xq,n,&left[q],n,ZERO,yq,n)?;
                gemm.multiply(n,n,n,ONE,&right[q],n,xq,n,ONE,yq,n)?;
                recycling.apply_into(n,ONE,xq,yq,&mut scratch.small);
                for &(p,edge)in &edges[q]{
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
        if terms.is_empty(){return Ok(());}
        for term in terms.iter(){let count=term.edges.len();
            for(e,&(p,_))in term.edges.iter().enumerate(){let s=if term.frequency{C{re:0.,im:nu[p]}}else{ONE};
                for c in 0..d{gathered[term.offset+c*count+e]=mul(s,x[p*d+c]);}}}
        interleaved.fill(ZERO);
        let(terms,gathered)=(&*terms,&*gathered);
        execution.for_mode_chunks(interleaved,modes,*term_work,MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
            for term in terms{let count=term.edges.len();let g=&gathered[term.offset..term.offset+d*count];let m=term.matrix;
                for(i,out)in chunk.chunks_exact_mut(modes).enumerate(){let r=first+i;
                    for at in m.row[r]as usize..m.row[r+1]as usize{let v=m.value[at];let xs=&g[m.col[at]as usize*count..(m.col[at]as usize+1)*count];
                        for(&(_,q),&xv)in term.edges.iter().zip(xs){mul_add(&mut out[q],v,xv);}}
                }
            }
            Ok(())
        })?;
        for r in 0..d{for q in 0..modes{add(&mut y[q*d+r],interleaved[r*modes+q]);}}
        Ok(())
    }
    /// The fold part of the operator alone, sum_t (K_t^(0) + i nu_p K_t^(1))
    /// x_p on every fold edge (diagonal and cross).
    pub fn apply_fold(&self,x:&[C])->Result<Vec<C>,String>{
        let n=self.n;let d=n*n;let mut y=vec![ZERO;self.dimension()];
        let fold=&self.prepared.fold.targets;let zero=&self.prepared.generator.zero;
        let mut apply=|stored:&fold::StoredFold,p:usize,q:usize|{let w=C{re:0.,im:self.nu[p]};
            let(xp,yq)=(&x[p*d..(p+1)*d],&mut y[q*d..(q+1)*d]);
            let op=Superop{left:Sparse::from_dense(&stored.left[0],n),right:Sparse::from_dense(&stored.right[0],n),sandwich:vec![]};
            let op1=Superop{left:Sparse::from_dense(&stored.left[1],n),right:Sparse::from_dense(&stored.right[1],n),sandwich:vec![]};
            let mut small=Vec::new();op.apply_into(n,ONE,xp,yq,&mut small);op1.apply_into(n,w,xp,yq,&mut small);
            stored.rest[0].apply_into(ONE,xp,yq);stored.rest[1].apply_into(w,xp,yq);};
        for q in 0..self.modes.len(){
            if let Some(t)=fold.first().filter(|t|t.charge==*zero){apply(t,q,q);}
            for &(p,edge)in &self.edges[q]{if let Edge::Fold(t)=edge{apply(&fold[t],p,q);}}
        }
        Ok(y)
    }
    /// The operator as a dense row-major matrix (validation only).
    pub fn dense(&self)->Vec<C>{
        let n=self.n;let d=n*n;let size=self.dimension();let mut out=vec![ZERO;size*size];
        let generator=&self.prepared.generator;let fold=&self.prepared.fold.targets;
        let mut block=vec![ZERO;d*d];
        let place=|out:&mut Vec<C>,block:&[C],q:usize,p:usize|{for r in 0..d{for c in 0..d{add(&mut out[(q*d+r)*size+p*d+c],block[r*d+c]);}}};
        for q in 0..self.modes.len(){
            block.fill(ZERO);generator.zero_component().op.dense_into(n,ONE,&mut block);for r in 0..d{block[r*d+r].im-=self.nu[q];}
            if let Some(t)=fold.first().filter(|t|t.charge==generator.zero){t.dense_into(n,0,ONE,&mut block);t.dense_into(n,1,C{re:0.,im:self.nu[q]},&mut block);}
            place(&mut out,&block,q,q);
            for &(p,edge)in &self.edges[q]{
                block.fill(ZERO);
                match edge{Edge::Explicit(e)=>self.explicit[e].dense_into(n,ONE,&mut block),
                    Edge::Fold(t)=>{fold[t].dense_into(n,0,ONE,&mut block);fold[t].dense_into(n,1,C{re:0.,im:self.nu[p]},&mut block);}}
                place(&mut out,&block,q,p);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::tests::{fixture,prepared};
    #[test]
    fn lattice_operator_matches_its_dense_assembly(){
        let f=fixture();let p=prepared(&f);let modes=p.modes(2).unwrap();
        let mut op=LatticeOperator::new(&p,&modes).unwrap();let size=op.dimension();let d=16;
        assert!(op.edge_counts[0]>0&&op.edge_counts[1]>0,"explicit and cross-fold edges");
        let x:Vec<C>=(0..size).map(|k|C{re:((k*7+3)%29)as f64/29.-0.5,im:((k*11+5)%31)as f64/31.-0.5}).collect();
        let mut y=vec![ZERO;size];op.apply(&x,&mut y).unwrap();
        let dense=op.dense();let mut scale=0_f64;let mut error=0_f64;
        for r in 0..size{let mut v=ZERO;for c in 0..size{mul_add(&mut v,dense[r*size+c],x[c]);}
            scale=scale.max(magnitude(v));error=error.max(magnitude(C{re:v.re-y[r].re,im:v.im-y[r].im}));}
        assert!(error<=1e-13*scale,"{error:e} of {scale:e}");
        let mut y2=vec![ZERO;size];op.apply(&x,&mut y2).unwrap();
        assert!(y.iter().zip(&y2).all(|(a,b)|a.re.to_bits()==b.re.to_bits()&&a.im.to_bits()==b.im.to_bits()),"repeatable");
        // The fold part alone: the operator minus its explicit components.
        let fold=op.apply_fold(&x).unwrap();let mut rest=vec![ZERO;size];let mut small=Vec::new();
        for(q,mode)in modes.iter().enumerate(){for(t,c)in &p.generator.components{
            if c.size>0.&&p.plan.steps.iter().any(|s|&s.charge==t&&s.treatment==classification::Treatment::Fold){continue;}
            let Ok(source)=charge_minus(mode,t)else{continue};let Some(&s)=op.index.get(&source)else{continue};
            c.op.apply_into(4,ONE,&x[s*d..(s+1)*d],&mut rest[q*d..(q+1)*d],&mut small);}
            for k in 0..d{rest[q*d+k].im-=op.nu[q]*x[q*d+k].re;rest[q*d+k].re+=op.nu[q]*x[q*d+k].im;}}
        let error=y.iter().zip(&rest).zip(&fold).map(|((a,b),c)|magnitude(C{re:a.re-b.re-c.re,im:a.im-b.im-c.im})).fold(0.,f64::max);
        assert!(error<=1e-13*scale,"fold part {error:e} of {scale:e}");
    }
}
