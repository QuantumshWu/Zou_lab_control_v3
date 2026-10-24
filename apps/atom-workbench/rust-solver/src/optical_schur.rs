//! Experimental EXACT block elimination inside an original shifted solve.
//!
//! The caller supplies its complete HS-weighted, active-coordinate matrix
//! M=shift*I-L and marks optical ge/eg coordinates Q. Nothing in M is dropped.
//! Exact nonzero connected components of M_QQ are inverted; GMRES acts on
//! S=M_PP-M_PQ M_QQ^-1 M_QP without constructing its potentially filled matrix.
//! Every returned solution reconstructs Q, including the original RHS_Q, and
//! passes the ORIGINAL full matrix's balanced residual gate. This is not an
//! effective master equation or a reduced physical density representation.
//!
//! Working directly with the supplied real matrix also retains the folded
//! Fourier +/-frequency signs and every Hilbert--Schmidt weight. In particular
//! a noncentral Fourier coefficient is NOT assumed to be Hermitian.
use crate::{lapack::Lapack,native_superlu};
use std::{cell::RefCell,collections::BTreeMap,time::Instant};

type Sparse=Vec<Vec<(usize,f64)>>;
type RawProduct=Box<dyn Fn(&[f64])->Vec<f64>>;
const MAX_Q_BLOCK:usize=128;
const DENSE_SEED_LIMIT:usize=256;

fn norm(x:&[f64])->f64{x.iter().map(|v|v*v).sum::<f64>().sqrt()}
fn elapsed(t:Instant)->f64{t.elapsed().as_secs_f64()*1000.}
fn product(a:&Sparse,x:&[f64],rows:usize)->Vec<f64>{
    let mut out=vec![0.;rows];
    for(j,col)in a.iter().enumerate(){for &(i,v)in col{out[i]+=v*x[j];}}out
}
fn checked_vector(x:&[f64],n:usize,name:&str)->Result<(),String>{
    if x.len()!=n||x.iter().any(|v|!v.is_finite()){return Err(format!("Invalid optical Schur {name}"));}Ok(())
}
fn canonicalize(matrix:&mut Sparse){
    for col in matrix{
        col.sort_by_key(|&(row,_)|row);
        let mut out=Vec::<(usize,f64)>::with_capacity(col.len());
        for &(row,value)in col.iter(){
            if let Some(last)=out.last_mut(){if last.0==row{last.1+=value;continue;}}
            out.push((row,value));
        }
        out.retain(|&(_,v)|v!=0.);*col=out;
    }
}
struct QBlock{indices:Vec<usize>,inverse:Vec<f64>}
enum PFactor{
    Dense{lu:Vec<f64>,pivots:Vec<i32>},
    Incomplete(native_superlu::Factor),
}
struct PSeed{scales:Vec<f64>,factor:PFactor}
impl PSeed{
    fn new(lap:&Lapack,pp:&Sparse)->Result<Self,String>{
        let n=pp.len();let mut scales=vec![0_f64;n];
        for col in pp{for &(i,v)in col{scales[i]=scales[i].max(v.abs());}}
        if scales.iter().any(|v|*v==0.||!v.is_finite()){
            return Err("Optical Schur seed has a zero/non-finite row; retain the original solver".into());
        }
        let balanced:Sparse=pp.iter().map(|col|col.iter().map(|&(i,v)|(i,v/scales[i])).collect()).collect();
        let factor=if n<=DENSE_SEED_LIMIT{
            let mut dense=vec![0.;n*n];for(j,col)in balanced.iter().enumerate(){for &(i,v)in col{dense[i+j*n]+=v;}}
            let(lu,pivots)=lap.factor(&dense,n)?;PFactor::Dense{lu,pivots}
        }else{
            let drop=std::env::var("ATOM_ILUTP_DROP").ok().and_then(|v|v.parse::<f64>().ok())
                .filter(|v|v.is_finite()&&*v>0.&&*v<1.).unwrap_or(1e-3);
            // Only the numerical seed is incomplete; S and its original
            // residual retain every entry, including weak optical feedback.
            PFactor::Incomplete(native_superlu::Factor::new(&balanced,drop,4.)?)
        };
        Ok(Self{scales,factor})
    }
    fn apply(&self,lap:&Lapack,rhs:&[f64])->Result<Vec<f64>,String>{
        let rhs:Vec<_>=rhs.iter().zip(&self.scales).map(|(b,s)|b/s).collect();
        let result=match &self.factor{
            PFactor::Dense{lu,pivots}=>lap.solve_factored(lu,pivots,&rhs,rhs.len())?,
            PFactor::Incomplete(factor)=>factor.solve(&rhs,false)?,
        };
        checked_vector(&result,self.scales.len(),"numerical seed result")?;Ok(result)
    }
}

#[derive(Clone,Debug,Default,serde::Serialize)]
#[serde(rename_all="camelCase")]
pub struct Diagnostics{
    pub dimension:usize,pub retained_dimension:usize,pub optical_dimension:usize,
    pub original_nonzeros:usize,pub pp_nonzeros:usize,pub pq_nonzeros:usize,pub qp_nonzeros:usize,
    pub q_blocks:usize,pub largest_q_block:usize,pub q_inverse_entries:usize,
    pub prepare_ms:f64,pub q_prepare_ms:f64,pub p_prepare_ms:f64,
    pub seed_nonzeros:usize,pub seed_groups:usize,pub seed_feedback_terms:usize,pub seed_feedback_ms:f64,
    pub seed_dense_workspace_bytes:usize,
    pub same_group_feedback:bool,pub structured_product:bool,
    pub solves:usize,pub failures:usize,pub iterations:usize,pub original_checks:usize,
    pub solve_ms:f64,pub product_ms:f64,pub precondition_ms:f64,pub orthogonal_ms:f64,pub check_ms:f64,
    pub last_residual:f64,pub last_tolerance:f64,
}
#[derive(Clone,Copy,Debug)]
pub struct Residual{pub norm:f64,pub tolerance:f64,pub accepted:bool}
pub struct Prepared{
    original:Sparse,scales:Vec<f64>,balanced_one_norm:f64,
    p:Vec<usize>,q:Vec<usize>,pp:Sparse,pq:Sparse,qp:Sparse,
    q_blocks:Vec<QBlock>,seed:PSeed,raw_product:Option<RawProduct>,profile:RefCell<Diagnostics>,
}

/// Add only zero-net-group optical loops to the NUMERICAL seed. Grouping
/// does not change the exact Schur product, Q inverse, or original equation.
/// For a real folded harmonic, its real and imaginary parts share one group.
/// The cross operators are joined per exact Q connected block and per P
/// group BEFORE products are formed, never by materializing the full Schur
/// complement and subsequently discarding its off-harmonic entries.
fn feedback_seed(pp:&Sparse,pq:&Sparse,qp:&Sparse,blocks:&[QBlock],groups:&[usize])->Result<(Sparse,usize),String>{
    let np=pp.len();let nq=pq.len();
    if groups.len()!=np{return Err("Optical Schur seed group dimensions differ".into());}
    let mut rows=vec![Vec::<(usize,f64)>::new();nq];
    for(j,col)in qp.iter().enumerate(){for &(i,value)in col{rows[i].push((j,value));}}
    let mut members=BTreeMap::<usize,Vec<usize>>::new();
    for(i,&group)in groups.iter().enumerate(){members.entry(group).or_default().push(i);}
    let members:Vec<_>=members.into_values().collect();let ng=members.len();
    let mut group_index=vec![usize::MAX;np];let mut local_index=vec![usize::MAX;np];
    for(group,indices)in members.iter().enumerate(){for(local,&i)in indices.iter().enumerate(){group_index[i]=group;local_index[i]=local;}}
    // Incidence only: do not retain expanded dense C columns for every Q
    // block/group. That would consume much more memory than the accumulator.
    let mut jobs=vec![Vec::new();ng];let mut input_mark=vec![usize::MAX;ng];let mut output_mark=vec![usize::MAX;ng];
    for(which,block)in blocks.iter().enumerate(){
        let mut touched=Vec::new();
        for &q in &block.indices{
            for &(p,_)in &rows[q]{let group=group_index[p];if input_mark[group]!=which{input_mark[group]=which;touched.push(group);}}
            for &(p,_)in &pq[q]{output_mark[group_index[p]]=which;}
        }
        for group in touched{if output_mark[group]==which{jobs[group].push(which);}}
    }
    let largest=members.iter().map(Vec::len).max().unwrap_or(0);
    let capacity=largest.checked_mul(largest).ok_or("Optical seed group size overflow")?;
    let bytes=capacity.checked_mul(std::mem::size_of::<f64>()).ok_or("Optical seed group allocation overflow")?;
    if bytes>64*1024*1024{return Err("Optical seed harmonic group exceeds the bounded dense workspace; retain original solver".into());}
    let mut result=vec![Vec::new();np];let mut terms=0usize;let mut scratch=vec![0.;capacity];
    for(group,indices)in members.iter().enumerate(){
        let width=indices.len();let count=width.checked_mul(width).ok_or("Optical seed group size overflow")?;
        let accumulator=&mut scratch[..count];accumulator.fill(0.);
        for(column,&global)in indices.iter().enumerate(){for &(row,value)in &pp[global]{
            if group_index[row]==group{accumulator[local_index[row]+column*width]=value;}
        }}
        // Reordering independent groups does not reorder any individual
        // matrix entry's sum: Q blocks, P columns and B entries retain the
        // exact traversal order of feedback_seed_tree below.
        for &which in &jobs[group]{
            let block=&blocks[which];let m=block.indices.len();
            let mut input=BTreeMap::<usize,Vec<f64>>::new();let mut output=Vec::new();
            for(local,&q)in block.indices.iter().enumerate(){
                for &(column,value)in &rows[q]{if group_index[column]==group{
                    input.entry(column).or_insert_with(||vec![0.;m])[local]+=value;
                }}
                for &(row,value)in &pq[q]{if group_index[row]==group{output.push((local,local_index[row],value));}}
            }
            let mut solved=vec![0.;m];
            for(column,c)in input{
                solved.fill(0.);for j in 0..m{for i in 0..m{solved[i]+=block.inverse[i+j*m]*c[j];}}
                let offset=local_index[column]*width;
                for &(local,row,b)in &output{
                    let value=b*solved[local];if value!=0.{
                        accumulator[row+offset]-=value;terms=terms.checked_add(1).ok_or("Optical Schur feedback count overflow")?;
                    }
                }
            }
        }
        for(column,&global)in indices.iter().enumerate(){
            let values=&accumulator[column*width..(column+1)*width];
            if values.iter().any(|v|!v.is_finite()){return Err("Non-finite optical Schur feedback seed".into());}
            let outside=pp[global].iter().filter(|&&(row,_)|group_index[row]!=group).count();
            let mut entries=Vec::with_capacity(outside+values.iter().filter(|&&v|v!=0.).count());
            // APP entries between different groups are retained exactly;
            // only the added optical feedback was restricted by group.
            entries.extend(pp[global].iter().copied().filter(|&(row,_)|group_index[row]!=group));
            entries.extend(indices.iter().copied().zip(values.iter().copied()).filter(|&(_,v)|v!=0.));
            entries.sort_unstable_by_key(|&(row,_)|row);result[global]=entries;
        }
    }
    Ok((result,terms))
}

// Small independent construction oracle. The production path above avoids
// one tree lookup/allocation on every B D^-1 C accumulation term.
#[cfg(test)]
fn feedback_seed_tree(pp:&Sparse,pq:&Sparse,qp:&Sparse,blocks:&[QBlock],groups:&[usize])->Result<(Sparse,usize),String>{
    let np=pp.len();let nq=pq.len();
    if groups.len()!=np{return Err("Optical Schur seed group dimensions differ".into());}
    let mut rows=vec![Vec::<(usize,f64)>::new();nq];
    for(j,col)in qp.iter().enumerate(){for &(i,value)in col{rows[i].push((j,value));}}
    let mut seed:Vec<BTreeMap<usize,f64>>=pp.iter().map(|col|col.iter().copied().collect()).collect();
    let mut terms=0usize;
    for block in blocks{
        let m=block.indices.len();
        let mut input=BTreeMap::<usize,BTreeMap<usize,Vec<f64>>>::new();
        let mut output=BTreeMap::<usize,Vec<(usize,usize,f64)>>::new();
        for(local,&global)in block.indices.iter().enumerate(){
            for &(column,value)in &rows[global]{
                input.entry(groups[column]).or_default().entry(column).or_insert_with(||vec![0.;m])[local]+=value;
            }
            for &(row,value)in &pq[global]{output.entry(groups[row]).or_default().push((local,row,value));}
        }
        for(group,columns)in input{
            let Some(entries)=output.get(&group)else{continue;};
            for(column,c)in columns{
                let mut solved=vec![0.;m];
                for j in 0..m{for i in 0..m{solved[i]+=block.inverse[i+j*m]*c[j];}}
                // Several Q coordinates may contribute to the same P pair.
                // Accumulate every term, including small off-diagonal G
                // contributions. Only exactly-zero terms are skipped.
                for &(local,row,b)in entries{
                    let value=b*solved[local];if value!=0.{
                        *seed[column].entry(row).or_default()-=value;terms=terms.checked_add(1).ok_or("Optical Schur feedback count overflow")?;
                    }
                }
            }
        }
    }
    let result:Sparse=seed.into_iter().map(|col|col.into_iter().filter(|&(_,v)|v!=0.).collect()).collect();
    if result.iter().flatten().any(|(_,v)|!v.is_finite()){return Err("Non-finite optical Schur feedback seed".into());}
    Ok((result,terms))
}
impl Prepared{
    /// The supplied matrix is already in the caller's desired physical/HS
    /// coordinates, but has NOT been row-equilibrated by block_solve.
    /// qmask marks the FULL active real ge/eg coordinate set. No harmonic or
    /// electronic manifold assumptions are used to delete matrix entries.
    pub fn new(lap:&Lapack,original:&Sparse,qmask:&[bool])->Result<Self,String>{
        Self::prepare(lap,original,qmask,None)
    }
    /// Include optical feedback with equal input/output harmonic group in
    /// the preconditioner. `groups` is in ORIGINAL active coordinates; real
    /// and imaginary parts of each folded harmonic must share its group.
    pub fn new_with_groups(lap:&Lapack,original:&Sparse,qmask:&[bool],groups:&[usize])->Result<Self,String>{
        if groups.len()!=original.len(){return Err("Optical Schur harmonic group dimensions differ".into());}
        Self::prepare(lap,original,qmask,Some(groups))
    }
    fn prepare(lap:&Lapack,original:&Sparse,qmask:&[bool],groups:Option<&[usize]>)->Result<Self,String>{
        let started=Instant::now();let n=original.len();
        if n==0||qmask.len()!=n||!qmask.iter().any(|v|*v)||!qmask.iter().any(|v|!*v){
            return Err("Optical Schur requires nonempty P and Q coordinate sets".into());
        }
        if original.iter().flatten().any(|&(i,v)|i>=n||!v.is_finite()){
            return Err("Invalid original optical Schur sparse matrix".into());
        }
        let p:Vec<_>=(0..n).filter(|&i|!qmask[i]).collect();let q:Vec<_>=(0..n).filter(|&i|qmask[i]).collect();
        let mut pi=vec![usize::MAX;n];let mut qi=vec![usize::MAX;n];
        for(local,&global)in p.iter().enumerate(){pi[global]=local;}
        for(local,&global)in q.iter().enumerate(){qi[global]=local;}
        let mut pp=vec![vec![];p.len()];let mut pq=vec![vec![];q.len()];
        let mut qp=vec![vec![];p.len()];let mut qq=vec![vec![];q.len()];let mut scales=vec![0_f64;n];
        for(j,col)in original.iter().enumerate(){for &(i,v)in col{
            scales[i]=scales[i].max(v.abs());
            match(qmask[i],qmask[j]){
                (false,false)=>pp[pi[j]].push((pi[i],v)),
                (false,true)=>pq[qi[j]].push((pi[i],v)),
                (true,false)=>qp[pi[j]].push((qi[i],v)),
                (true,true)=>qq[qi[j]].push((qi[i],v)),
            }
        }}
        if scales.iter().any(|s|*s==0.){return Err("Original optical Schur matrix has a zero row".into());}
        canonicalize(&mut pp);canonicalize(&mut pq);canonicalize(&mut qp);canonicalize(&mut qq);
        if [&pp,&pq,&qp,&qq].iter().any(|matrix|matrix.iter().flatten().any(|(_,v)|!v.is_finite())){
            return Err("Non-finite combined optical Schur matrix entry".into());
        }
        let balanced_one_norm=original.iter().map(|col|col.iter().map(|&(i,v)|(v/scales[i]).abs()).sum::<f64>()).fold(0.,f64::max);
        let mut profile=Diagnostics{dimension:n,retained_dimension:p.len(),optical_dimension:q.len(),
            original_nonzeros:original.iter().map(Vec::len).sum(),pp_nonzeros:pp.iter().map(Vec::len).sum(),
            pq_nonzeros:pq.iter().map(Vec::len).sum(),qp_nonzeros:qp.iter().map(Vec::len).sum(),..Diagnostics::default()};
        let q_started=Instant::now();let mut parent:Vec<_>=(0..q.len()).collect();
        fn root(parent:&mut[usize],mut i:usize)->usize{while parent[i]!=i{parent[i]=parent[parent[i]];i=parent[i];}i}
        for(j,col)in qq.iter().enumerate(){for &(i,v)in col{if i!=j&&v!=0.{
            let a=root(&mut parent,i);let b=root(&mut parent,j);parent[a]=b;
        }}}
        let mut components=BTreeMap::<usize,Vec<usize>>::new();
        for i in 0..q.len(){let key=root(&mut parent,i);components.entry(key).or_default().push(i);}
        let mut local=vec![usize::MAX;q.len()];let mut q_blocks=Vec::with_capacity(components.len());
        for indices in components.into_values(){
            let m=indices.len();
            if m>MAX_Q_BLOCK{return Err(format!("Exact optical Q component has {m} coordinates; retain original solver instead of truncating it"));}
            for(j,&i)in indices.iter().enumerate(){local[i]=j;}
            let mut dense=vec![0.;m*m];
            for(j,&global)in indices.iter().enumerate(){for &(i,v)in &qq[global]{
                if local[i]==usize::MAX{return Err("Optical Q connectivity partition lost an exact entry".into());}
                dense[local[i]+j*m]+=v;
            }}
            let(lu,pivots)=lap.factor(&dense,m)?;let mut identity=vec![0.;m*m];for i in 0..m{identity[i+i*m]=1.;}
            let inverse=lap.solve_matrix_factored(&lu,&pivots,&identity,m)?;
            if inverse.iter().any(|v|!v.is_finite()){return Err("Non-finite exact optical Q inverse".into());}
            profile.largest_q_block=profile.largest_q_block.max(m);profile.q_inverse_entries+=m*m;
            for &i in &indices{local[i]=usize::MAX;}q_blocks.push(QBlock{indices,inverse});
        }
        profile.q_blocks=q_blocks.len();profile.q_prepare_ms=elapsed(q_started);
        let seed_matrix=if let Some(groups)=groups{
            let start=Instant::now();let p_groups:Vec<_>=p.iter().map(|&i|groups[i]).collect();
            let mut counts=BTreeMap::<usize,usize>::new();for &group in &p_groups{*counts.entry(group).or_default()+=1;}
            profile.seed_groups=counts.len();let width=counts.values().copied().max().unwrap_or(0);
            profile.seed_dense_workspace_bytes=width.checked_mul(width).and_then(|n|n.checked_mul(std::mem::size_of::<f64>())).ok_or("Optical seed workspace size overflow")?;
            let(matrix,terms)=feedback_seed(&pp,&pq,&qp,&q_blocks,&p_groups)?;
            profile.same_group_feedback=true;profile.seed_feedback_terms=terms;profile.seed_feedback_ms=elapsed(start);Some(matrix)
        }else{None};
        let matrix=seed_matrix.as_ref().unwrap_or(&pp);profile.seed_nonzeros=matrix.iter().map(Vec::len).sum();
        let p_started=Instant::now();let seed=PSeed::new(lap,matrix)?;profile.p_prepare_ms=elapsed(p_started);profile.prepare_ms=elapsed(started);
        Ok(Self{original:original.clone(),scales,balanced_one_norm,p,q,pp,pq,qp,q_blocks,seed,raw_product:None,profile:RefCell::new(profile)})
    }
    /// Optional exact ORIGINAL full M product. It only accelerates proposed
    /// Schur products; the unchanged sparse M remains the acceptance oracle.
    pub fn with_raw_product(mut self,product:RawProduct)->Self{
        self.raw_product=Some(product);self.profile.get_mut().structured_product=true;self
    }
    pub fn dimension(&self)->usize{self.original.len()}
    pub fn retained_dimension(&self)->usize{self.p.len()}
    pub fn diagnostics(&self)->Diagnostics{self.profile.borrow().clone()}
    fn inverse_q(&self,rhs:&[f64])->Vec<f64>{
        let mut out=vec![0.;self.q.len()];
        for block in &self.q_blocks{let m=block.indices.len();for(j,&global)in block.indices.iter().enumerate(){
            let x=rhs[global];for(i,&target)in block.indices.iter().enumerate(){out[target]+=block.inverse[i+j*m]*x;}
        }}out
    }
    /// Exact matrix-free S action; no incomplete factor is used here.
    pub fn schur_product(&self,p:&[f64])->Result<Vec<f64>,String>{
        checked_vector(p,self.p.len(),"Schur product input")?;
        if let Some(product)=&self.raw_product{
            let mut lifted=vec![0.;self.dimension()];for(&i,&v)in self.p.iter().zip(p){lifted[i]=v;}
            let first=product(&lifted);checked_vector(&first,self.dimension(),"first full product")?;
            let q_rhs:Vec<_>=self.q.iter().map(|&i|first[i]).collect();let eliminated=self.inverse_q(&q_rhs);
            lifted.fill(0.);for(&i,v)in self.q.iter().zip(eliminated){lifted[i]=v;}
            let second=product(&lifted);checked_vector(&second,self.dimension(),"second full product")?;
            let result:Vec<_>=self.p.iter().map(|&i|first[i]-second[i]).collect();
            checked_vector(&result,self.p.len(),"structured Schur result")?;return Ok(result);
        }
        let mut result=product(&self.pp,p,self.p.len());
        let eliminated=self.inverse_q(&product(&self.qp,p,self.q.len()));
        let feedback=product(&self.pq,&eliminated,self.p.len());
        for(value,other)in result.iter_mut().zip(feedback){*value-=other;}
        checked_vector(&result,self.p.len(),"Schur product result")?;Ok(result)
    }
    pub fn reduced_rhs(&self,rhs:&[f64])->Result<Vec<f64>,String>{
        checked_vector(rhs,self.dimension(),"full RHS")?;
        let q:Vec<_>=self.q.iter().map(|&i|rhs[i]).collect();let eliminated=self.inverse_q(&q);
        let feedback=product(&self.pq,&eliminated,self.p.len());
        Ok(self.p.iter().zip(feedback).map(|(&i,v)|rhs[i]-v).collect())
    }
    pub fn reconstruct(&self,rhs:&[f64],p:&[f64])->Result<Vec<f64>,String>{
        checked_vector(rhs,self.dimension(),"reconstruction RHS")?;checked_vector(p,self.p.len(),"reconstruction P")?;
        let cross=product(&self.qp,p,self.q.len());
        let q_rhs:Vec<_>=self.q.iter().zip(cross).map(|(&i,v)|rhs[i]-v).collect();let q=self.inverse_q(&q_rhs);
        let mut result=vec![0.;self.dimension()];
        for(&i,&value)in self.p.iter().zip(p){result[i]=value;}
        for(&i,value)in self.q.iter().zip(q){result[i]=value;}
        checked_vector(&result,self.dimension(),"full reconstructed solution")?;Ok(result)
    }
    /// Same row equilibrium, product traversal, relative target and rounding
    /// allowance as block_solve. S, APP and a preconditioned residual cannot
    /// certify a solution in place of this full ORIGINAL-matrix check.
    pub fn original_residual(&self,rhs:&[f64],x:&[f64])->Result<Residual,String>{
        checked_vector(rhs,self.dimension(),"residual RHS")?;checked_vector(x,self.dimension(),"residual solution")?;
        let b:Vec<_>=rhs.iter().zip(&self.scales).map(|(b,s)|b/s).collect();let mut ax=vec![0.;self.dimension()];
        for(j,col)in self.original.iter().enumerate(){for &(i,v)in col{ax[i]+=(v/self.scales[i])*x[j];}}
        let residual=norm(&b.iter().zip(ax).map(|(b,a)|b-a).collect::<Vec<_>>());
        let tolerance=(1e-11*norm(&b).max(1e-30)).max(32.*f64::EPSILON*(self.balanced_one_norm*x.iter().map(|v|v.abs()).sum::<f64>()+b.iter().map(|v|v.abs()).sum::<f64>()));
        Ok(Residual{norm:residual,tolerance,accepted:residual.is_finite()&&residual<tolerance})
    }
    fn checked_candidate(&self,rhs:&[f64],p:&[f64],profile:&mut Diagnostics)->Result<Option<Vec<f64>>,String>{
        let start=Instant::now();let full=self.reconstruct(rhs,p)?;let check=self.original_residual(rhs,&full)?;
        profile.original_checks+=1;profile.check_ms+=elapsed(start);profile.last_residual=check.norm;profile.last_tolerance=check.tolerance;
        Ok(check.accepted.then_some(full))
    }
    /// Returns a FULL restored solution, never just a preconditioner proposal.
    /// Failure means the caller must retain/restart its original full solve.
    pub fn solve(&self,lap:&Lapack,rhs:&[f64])->Result<Vec<f64>,String>{
        let start=Instant::now();let mut profile=Diagnostics::default();
        let result=self.solve_inner(lap,rhs,&mut profile);profile.solve_ms=elapsed(start);
        let mut total=self.profile.borrow_mut();total.solves+=1;total.failures+=usize::from(result.is_err());total.iterations+=profile.iterations;
        total.original_checks+=profile.original_checks;total.solve_ms+=profile.solve_ms;total.product_ms+=profile.product_ms;
        total.precondition_ms+=profile.precondition_ms;total.orthogonal_ms+=profile.orthogonal_ms;total.check_ms+=profile.check_ms;
        total.last_residual=profile.last_residual;total.last_tolerance=profile.last_tolerance;
        if std::env::var_os("ATOM_GMRES_PROFILE").is_some(){eprintln!("Optical Schur solve n={} p={} q={} steps={} ms={:.3} productMs={:.3} preconditionMs={:.3} orthogonalMs={:.3} originalCheckMs={:.3} residual={:e} tolerance={:e} success={}",
            self.dimension(),self.p.len(),self.q.len(),profile.iterations,profile.solve_ms,profile.product_ms,profile.precondition_ms,profile.orthogonal_ms,profile.check_ms,profile.last_residual,profile.last_tolerance,result.is_ok());}
        result
    }
    fn solve_inner(&self,lap:&Lapack,rhs:&[f64],profile:&mut Diagnostics)->Result<Vec<f64>,String>{
        let reduced=self.reduced_rhs(rhs)?;let n=self.p.len();let start=Instant::now();let mut x=self.seed.apply(lap,&reduced)?;
        profile.precondition_ms+=elapsed(start);
        for _restart in 0..4{
            if let Some(full)=self.checked_candidate(rhs,&x,profile)?{return Ok(full);}
            let start=Instant::now();let ax=self.schur_product(&x)?;profile.product_ms+=elapsed(start);
            let residual:Vec<_>=reduced.iter().zip(ax).map(|(b,a)|b-a).collect();
            let start=Instant::now();let residual=self.seed.apply(lap,&residual)?;profile.precondition_ms+=elapsed(start);
            let beta=norm(&residual);if !beta.is_finite()||beta==0.{return Err("Optical Schur stagnated before its original residual passed".into());}
            let m=32.min(n);let stride=m+1;let mut basis=vec![residual.iter().map(|v|v/beta).collect::<Vec<_>>()];
            let mut h=vec![0.;stride*m];let mut cs=vec![0.;m];let mut sn=vec![0.;m];let mut g=vec![0.;stride];g[0]=beta;
            let mut used=0;let mut last_candidate=None;
            for j in 0..m{
                let start=Instant::now();let a=self.schur_product(&basis[j])?;profile.product_ms+=elapsed(start);
                let start=Instant::now();let mut w=self.seed.apply(lap,&a)?;profile.precondition_ms+=elapsed(start);profile.iterations+=1;
                let start=Instant::now();
                for _ in 0..2{for i in 0..=j{
                    let dot=w.iter().zip(&basis[i]).map(|(a,b)|a*b).sum::<f64>();h[i+j*stride]+=dot;
                    for(value,&q)in w.iter_mut().zip(&basis[i]){*value-=dot*q;}
                }}
                let length=norm(&w);if !length.is_finite(){return Err("Non-finite optical Schur Krylov direction".into());}
                h[j+1+j*stride]=length;if length>1e-25{basis.push(w.iter().map(|v|v/length).collect());}
                for i in 0..j{let a=h[i+j*stride];let b=h[i+1+j*stride];h[i+j*stride]=cs[i]*a+sn[i]*b;h[i+1+j*stride]=-sn[i]*a+cs[i]*b;}
                let diagonal=h[j+j*stride];let sub=h[j+1+j*stride];let magnitude=diagonal.hypot(sub);
                if magnitude==0.||!magnitude.is_finite(){return Err("Optical Schur GMRES breakdown".into());}
                cs[j]=diagonal/magnitude;sn[j]=sub/magnitude;h[j+j*stride]=magnitude;g[j+1]=-sn[j]*g[j];g[j]*=cs[j];used=j+1;
                let near=g[j+1].abs()<1e-12*beta||length<=1e-25;
                profile.orthogonal_ms+=elapsed(start);
                if near||used%8==0{
                    let start=Instant::now();let candidate=gmres_candidate(&x,&basis,&h,&g,used,stride)?;profile.orthogonal_ms+=elapsed(start);
                    if let Some(full)=self.checked_candidate(rhs,&candidate,profile)?{return Ok(full);}
                    last_candidate=Some((used,candidate));
                }
                if near{break;}
            }
            let start=Instant::now();x=match last_candidate{Some((count,candidate))if count==used=>candidate,_=>gmres_candidate(&x,&basis,&h,&g,used,stride)?};profile.orthogonal_ms+=elapsed(start);
        }
        if let Some(full)=self.checked_candidate(rhs,&x,profile)?{return Ok(full);}
        Err("Optical Schur did not meet the ORIGINAL full-system residual; retain the original solver".into())
    }
}
fn gmres_candidate(x:&[f64],basis:&[Vec<f64>],h:&[f64],g:&[f64],used:usize,stride:usize)->Result<Vec<f64>,String>{
    let mut coefficients=g[..used].to_vec();
    for i in (0..used).rev(){for j in i+1..used{coefficients[i]-=h[i+j*stride]*coefficients[j];}coefficients[i]/=h[i+i*stride];}
    let mut out=x.to_vec();for j in 0..used{for(value,&q)in out.iter_mut().zip(&basis[j]){*value+=coefficients[j]*q;}}
    checked_vector(&out,x.len(),"GMRES candidate")?;Ok(out)
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::{build_generator,coordinates,Coordinate,Model,Operator};
    fn sparse(dense:&[f64],n:usize)->Sparse{(0..n).map(|j|(0..n).filter_map(|i|(dense[i+j*n]!=0.).then_some((i,dense[i+j*n]))).collect()).collect()}
    fn dense(a:&Sparse)->Vec<f64>{let n=a.len();let mut out=vec![0.;n*n];for(j,col)in a.iter().enumerate(){for &(i,v)in col{out[i+j*n]+=v;}}out}
    fn close(a:&[f64],b:&[f64],tolerance:f64){assert_eq!(a.len(),b.len());let error=a.iter().zip(b).map(|(a,b)|(a-b).abs()).fold(0.,f64::max);assert!(error<tolerance,"error {error:e}");}
    fn example()->(Sparse,Vec<bool>){
        let n=7;let q=vec![false,true,false,true,true,false,true];let mut a=vec![0.;n*n];
        for i in 0..n{a[i+i*n]=2.+i as f64;}
        for &(i,j,v)in &[(1,3,4.),(3,1,-3.),(4,6,-7.),(6,4,5.),(0,1,0.4),(1,0,-0.2),(2,3,0.3),(3,2,0.7),(5,4,-0.8),(6,5,0.6),(0,2,-0.1),(5,0,0.3),(2,6,0.2),(4,2,-0.4)]{a[i+j*n]=v;}
        let weights=[1.,2.,0.7,3.,0.4,1.3,2.5];for j in 0..n{for i in 0..n{a[i+j*n]*=weights[i]/weights[j];}}
        (sparse(&a,n),q)
    }
    #[test]
    fn optical_schur_exact_weighted_permuted_blocks_and_nonzero_optical_rhs(){
        let lap=Lapack::load().unwrap();let(a,q)=example();let prepared=Prepared::new(&lap,&a,&q).unwrap();let n=a.len();
        assert_eq!(prepared.diagnostics().q_blocks,2);assert_eq!(prepared.diagnostics().largest_q_block,2);
        for column in 0..=n{
            let rhs:Vec<_>=(0..n).map(|i|if column<n{if i==column{1.}else{0.}}else{(0.3+i as f64).sin()}).collect();
            let expected=lap.solve(&dense(&a),&rhs,n).unwrap();let actual=prepared.solve(&lap,&rhs).unwrap();close(&actual,&expected,1e-12);
            let p:Vec<_>=prepared.p.iter().map(|&i|expected[i]).collect();close(&prepared.schur_product(&p).unwrap(),&prepared.reduced_rhs(&rhs).unwrap(),2e-12);
            close(&prepared.reconstruct(&rhs,&p).unwrap(),&expected,2e-12);assert!(prepared.original_residual(&rhs,&actual).unwrap().accepted);
        }
        let zero=prepared.solve(&lap,&vec![0.;n]).unwrap();assert!(zero.iter().all(|v|*v==0.));
        let rhs=vec![1.;n];let mut bad=prepared.solve(&lap,&rhs).unwrap();bad[prepared.q[0]]+=1e-5;
        assert!(!prepared.original_residual(&rhs,&bad).unwrap().accepted,"a correct reduced P cannot hide an incorrect restored Q");
    }
    #[test]
    fn optical_schur_folded_fourier_keeps_both_quadratures_and_complete_complex_decay(){
        let lap=Lapack::load().unwrap();let n=3;let d=n*n;
        let base=Model{dimension:n,states:vec![],h_real:vec![0.,0.3,-0.2,0.3,5.,0.,-0.2,0.,8.],
            h_imag:vec![0.,0.1,0.4,-0.1,0.,0.,-0.4,0.,0.],
            collapse:vec![Operator{real:vec![0.,1.,0.3,0.,0.,0.,0.,0.,0.],imag:vec![0.,0.2,-0.1,0.,0.,0.,0.,0.,0.]}],
            initial_index:0,initial_real:None,initial_imag:None,time_dependent:false,drives:vec![],frequency_coordinates:None};
        let l0=build_generator(&base).unwrap().matrix;let mut cos=base.clone();cos.collapse.clear();
        cos.h_real=vec![0.,0.7,0.2,0.7,0.,0.,0.2,0.,0.];cos.h_imag=vec![0.;d];let ac=build_generator(&cos).unwrap().matrix;
        let mut sin=cos.clone();sin.h_real=vec![0.;d];sin.h_imag=vec![0.,0.2,-0.3,-0.2,0.,0.,0.3,0.,0.];let as_=build_generator(&sin).unwrap().matrix;
        let size=3*d;let mut a=vec![0.;size*size];let shift=0.17;let omega=6.3;
        for part in 0..3{for j in 0..d{for &(i,v)in &l0[j]{a[part*d+i+(part*d+j)*size]-=v;}a[part*d+j+(part*d+j)*size]+=shift;}}
        for j in 0..d{
            a[d+j+(2*d+j)*size]-=omega;a[2*d+j+(d+j)*size]+=omega;
            for &(i,v)in &ac[j]{a[i+(d+j)*size]-=v;a[d+i+j*size]-=0.5*v;}
            for &(i,v)in &as_[j]{a[i+(2*d+j)*size]+=v;a[2*d+i+j*size]+=0.5*v;}
        }
        let physical_q:Vec<_>=coordinates(n).iter().map(|c|matches!(c,Coordinate::Real(0,_)|Coordinate::Imag(0,_))).collect();
        let q:Vec<_>=(0..size).map(|i|physical_q[i%d]).collect();
        let weights:Vec<_>=(0..size).map(|i|((if i%d<n{1_f64}else{2.})*(if i<d{1.}else{2.})).sqrt()).collect();
        for j in 0..size{for i in 0..size{a[i+j*size]*=weights[i]/weights[j];}}
        let prepared=Prepared::new(&lap,&sparse(&a,size),&q).unwrap();
        assert_eq!(prepared.retained_dimension(),15);assert!(prepared.diagnostics().largest_q_block>=4);
        for column in [0,3,4,12,13,21,22,size]{
            let rhs:Vec<_>=(0..size).map(|i|if column<size{if i==column{1.}else{0.}}else{(0.17+i as f64*0.7).cos()}).collect();
            close(&prepared.solve(&lap,&rhs).unwrap(),&lap.solve(&a,&rhs,size).unwrap(),2e-10);
        }
    }
    #[test]
    fn optical_schur_exact_connectivity_retains_tiny_links_and_full_residual_rejects_corruption(){
        let lap=Lapack::load().unwrap();let(a,q)=example();let mut a=a;
        a[3].push((4,1e-16));let prepared=Prepared::new(&lap,&a,&q).unwrap();
        assert_eq!(prepared.diagnostics().q_blocks,1);assert_eq!(prepared.diagnostics().largest_q_block,4);
        let rhs:Vec<_>=(0..a.len()).map(|i|i as f64+0.5).collect();let p=prepared.reduced_rhs(&rhs).unwrap();
        let fake=prepared.reconstruct(&rhs,&p).unwrap();assert!(!prepared.original_residual(&rhs,&fake).unwrap().accepted);
        close(&prepared.solve(&lap,&rhs).unwrap(),&lap.solve(&dense(&a),&rhs,a.len()).unwrap(),2e-12);
    }
    #[test]
    fn optical_schur_feedback_seed_contains_only_equal_harmonic_group_loops(){
        let lap=Lapack::load().unwrap();let(a,q)=example();let prepared=Prepared::new(&lap,&a,&q).unwrap();
        let groups=[0,0,1,0,1,1,1];let p_groups:Vec<_>=prepared.p.iter().map(|&i|groups[i]).collect();
        let(seed,terms)=feedback_seed(&prepared.pp,&prepared.pq,&prepared.qp,&prepared.q_blocks,&p_groups).unwrap();
        assert!(terms>0);let seed=dense(&seed);let pp=dense(&prepared.pp);let n=prepared.retained_dimension();
        let mut excluded_feedback=0_f64;
        for j in 0..n{
            let mut basis=vec![0.;n];basis[j]=1.;let s=prepared.schur_product(&basis).unwrap();
            for i in 0..n{
                let expected=if p_groups[i]==p_groups[j]{s[i]}else{excluded_feedback=excluded_feedback.max((s[i]-pp[i+j*n]).abs());pp[i+j*n]};
                assert!((seed[i+j*n]-expected).abs()<2e-14);
            }
        }
        assert!(excluded_feedback>1e-5,"the fixture must actually distinguish the seed from full S");
        let grouped=Prepared::new_with_groups(&lap,&a,&q,&groups).unwrap();let rhs=vec![1.;a.len()];
        close(&grouped.solve(&lap,&rhs).unwrap(),&lap.solve(&dense(&a),&rhs,a.len()).unwrap(),2e-12);
        assert!(grouped.diagnostics().same_group_feedback);assert_eq!(grouped.diagnostics().seed_groups,2);
        // One group produces the entire Schur matrix in this tiny algebra
        // test only. Its dense numerical seed is exact and needs no GMRES.
        let all=Prepared::new_with_groups(&lap,&a,&q,&vec![0;a.len()]).unwrap();
        close(&all.solve(&lap,&rhs).unwrap(),&lap.solve(&dense(&a),&rhs,a.len()).unwrap(),2e-12);
        assert_eq!(all.diagnostics().iterations,0);
        assert!(Prepared::new_with_groups(&lap,&a,&q,&[0]).is_err());
    }
    #[test]
    fn optical_schur_dense_group_seed_matches_tree_accumulation_bit_for_bit(){
        let lap=Lapack::load().unwrap();let n=19;let qmask:Vec<_>=(0..n).map(|i|i%3==1).collect();
        let p:Vec<_>=(0..n).filter(|&i|!qmask[i]).collect();let q:Vec<_>=(0..n).filter(|&i|qmask[i]).collect();
        let mut a=vec![0.;n*n];for i in 0..n{a[i+i*n]=3.+0.1*i as f64;}
        for(i,&row)in q.iter().enumerate(){for(j,&column)in q.iter().enumerate(){if i!=j&&i/3==j/3{a[row+column*n]=0.23*(1.+i as f64-j as f64);}}}
        for &column in &p{for &row in &q{
            a[row+column*n]=(0.17+0.19*row as f64+0.23*column as f64).sin();
            a[column+row*n]=(0.29+0.11*row as f64-0.31*column as f64).cos();
        }}
        for(i,&row)in p.iter().enumerate(){for(j,&column)in p.iter().enumerate(){if i!=j&&(i+j)%3==0{a[row+column*n]=0.012*(i as f64-j as f64);}}}
        let prepared=Prepared::new(&lap,&sparse(&a,n),&qmask).unwrap();
        for groups in [vec![5;p.len()],(0..p.len()).map(|i|7*(i%3)+11).collect(),(0..p.len()).map(|i|if i<2{90}else if i<7{3}else{77}).collect()]{
            let (actual,terms)=feedback_seed(&prepared.pp,&prepared.pq,&prepared.qp,&prepared.q_blocks,&groups).unwrap();
            let (expected,expected_terms)=feedback_seed_tree(&prepared.pp,&prepared.pq,&prepared.qp,&prepared.q_blocks,&groups).unwrap();
            assert_eq!(terms,expected_terms);assert!(terms>0);
            assert_eq!(actual.len(),expected.len());
            for(a,b)in actual.iter().zip(&expected){assert_eq!(a.len(),b.len());for(&(row,x),&(other,y))in a.iter().zip(b){assert_eq!(row,other);assert_eq!(x.to_bits(),y.to_bits());}}
        }
    }
    #[test]
    fn optical_schur_full_structured_product_matches_sparse_but_cannot_certify_itself(){
        use std::{cell::Cell,rc::Rc};
        let lap=Lapack::load().unwrap();let(a,q)=example();let reference=Prepared::new(&lap,&a,&q).unwrap();
        let count=Rc::new(Cell::new(0));let seen=Rc::clone(&count);let copied=a.clone();
        let structured=Prepared::new(&lap,&a,&q).unwrap().with_raw_product(Box::new(move|x|{seen.set(seen.get()+1);product(&copied,x,x.len())}));
        let p:Vec<_>=(0..reference.retained_dimension()).map(|i|(i as f64+0.3).sin()).collect();
        close(&structured.schur_product(&p).unwrap(),&reference.schur_product(&p).unwrap(),2e-14);assert_eq!(count.get(),2);
        let rhs=vec![1.;a.len()];close(&structured.solve(&lap,&rhs).unwrap(),&reference.solve(&lap,&rhs).unwrap(),2e-12);
        assert!(structured.diagnostics().structured_product);assert!(structured.diagnostics().original_checks>0);
        let wrong=Prepared::new(&lap,&a,&q).unwrap().with_raw_product(Box::new(|x|vec![0.;x.len()]));
        assert!(wrong.solve(&lap,&rhs).is_err(),"a zero proposed operator must not pass the original sparse residual");
        let malformed=Prepared::new(&lap,&a,&q).unwrap().with_raw_product(Box::new(|_|vec![0.]));
        assert!(malformed.schur_product(&p).is_err());
    }
    #[test]
    fn optical_schur_unsupported_or_singular_structure_fails_without_deleting_entries(){
        let lap=Lapack::load().unwrap();let(a,q)=example();
        assert!(Prepared::new(&lap,&a,&q[..q.len()-1]).is_err());
        assert!(Prepared::new(&lap,&a,&vec![false;q.len()]).is_err());
        // Full matrix can be invertible while APP alone is singular. This
        // prototype must fall back; it must not alter APP or the trace row.
        assert!(Prepared::new(&lap,&vec![vec![(1,1.)],vec![(0,1.),(1,1.)]],&[false,true]).is_err());
        assert!(Prepared::new(&lap,&vec![vec![(0,1.),(1,1.)],vec![(0,1.)]],&[false,true]).is_err());
        let mut bad=a.clone();bad[0].push((0,f64::NAN));assert!(Prepared::new(&lap,&bad,&q).is_err());
        let prepared=Prepared::new(&lap,&a,&q).unwrap();assert!(prepared.solve(&lap,&[1.]).is_err());
        assert!(prepared.schur_product(&vec![f64::NAN;prepared.retained_dimension()]).is_err());
    }
}
