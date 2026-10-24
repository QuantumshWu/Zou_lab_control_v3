//! A block preconditioner for the ORIGINAL sparse system. No matrix entry is
//! removed from residual products. Blocks are a numerical device only.
use super::{lapack::Lapack, norm_one};
use std::{cell::{Cell,RefCell},rc::Rc};
use faer::{
    linalg::solvers::Solve,
    sparse::{SparseColMat, Triplet},
    Mat,
};
type Sparse = Vec<Vec<(usize, f64)>>;
pub(super) type CustomApply=Box<dyn Fn(&Lapack,&[f64])->Result<Vec<f64>,String>>;
pub(super) type RawProduct=Box<dyn Fn(&[f64])->Vec<f64>>;
fn operator_norms(matrix:&Sparse)->(f64,f64){
    // Preserve the exact traversal/summation expressions previously used in
    // each solve. These are immutable operator constants, never solved data.
    let one=matrix.iter().map(|col|col.iter().map(|(_,v)|v.abs()).sum::<f64>()).fold(0.,f64::max);
    let mut rows=vec![0.;matrix.len()];for col in matrix{for &(i,v)in col{rows[i]+=v.abs();}}
    (one,rows.into_iter().fold(0.,f64::max))
}
/// A fixed linear preconditioner in the ORIGINAL (not row-scaled) coordinates.
/// It never calls GMRES and never stores a solved RHS or a physical result.
pub(super) struct RawPreconditioner{
    pub dimension:usize,
    apply:Rc<dyn Fn(&Lapack,&[f64],bool)->Result<Vec<f64>,String>>,
}
impl RawPreconditioner{
    fn apply(&self,lap:&Lapack,rhs:&[f64])->Result<Vec<f64>,String>{
        self.apply_side(lap,rhs,false)
    }
    fn apply_transpose(&self,lap:&Lapack,rhs:&[f64])->Result<Vec<f64>,String>{
        self.apply_side(lap,rhs,true)
    }
    fn apply_side(&self,lap:&Lapack,rhs:&[f64],transpose:bool)->Result<Vec<f64>,String>{
        if rhs.len()!=self.dimension{return Err("Raw preconditioner dimensions differ".into());}
        let result=(self.apply)(lap,rhs,transpose)?;
        if result.len()!=self.dimension{return Err("Raw preconditioner returned a different dimension".into());}Ok(result)
    }
    fn from_prepared(prepared:Prepared)->Rc<Self>{
        Self::from_shared(Rc::new(prepared))
    }
    pub fn from_shared(prepared:Rc<Prepared>)->Rc<Self>{
        let dimension=prepared.matrix.len();
        Rc::new(Self{dimension,apply:Rc::new(move|lap,rhs,transpose|{
            if transpose{
                // P_raw=P_eq*S^-1, hence P_raw^T=S^-1*P_eq^T.
                let mut result=prepared.precondition(lap,rhs,true)?;
                for(value,scale)in result.iter_mut().zip(&prepared.scales){*value/=scale;}return Ok(result);
            }
            let scaled:Vec<_>=rhs.iter().zip(&prepared.scales).map(|(b,s)|b/s).collect();
            prepared.precondition(lap,&scaled,false)
        })})
    }
}
fn native_factor(matrix:&Sparse)->Result<Rc<crate::native_superlu::Factor>,String>{
    let drop_tolerance=std::env::var("ATOM_ILUTP_DROP").ok().and_then(|s|s.parse::<f64>().ok())
        .filter(|v|v.is_finite()&&*v>0.&&*v<1.).unwrap_or(1e-3);
    let seed_cutoff=std::env::var("ATOM_ILUTP_SEED_DROP").ok().and_then(|s|s.parse::<f64>().ok())
        .filter(|v|v.is_finite()&&*v>0.&&*v<1.).unwrap_or(0.);
    let seed=(seed_cutoff>0.).then(||ilu_seed(matrix,seed_cutoff));
    crate::native_superlu::Factor::new(seed.as_ref().unwrap_or(matrix),drop_tolerance,4.).map(Rc::new)
}
/// Sparsify only the matrix supplied to the incomplete factorization. The
/// original balanced operator below remains unchanged for every solve gate.
fn ilu_seed(matrix:&Sparse,cutoff:f64)->Sparse{
    let n=matrix.len();let mut strongest=vec![(usize::MAX,0_f64);n];
    for (column,entries) in matrix.iter().enumerate(){for &(row,value) in entries{
        if row!=column&&value.abs()>strongest[row].1{strongest[row]=(column,value.abs());}
    }}
    matrix.iter().enumerate().map(|(column,entries)|entries.iter().copied()
        .filter(|&(row,value)|row==column||value.abs()>=cutoff||strongest[row].0==column).collect()).collect()
}
/// Adapt only the preconditioner graph. The original operator and residual
/// products retain every entry. Rejecting an overgrown block must not force a
/// full-system factorization before trying a bounded grouping.
fn groups_at(a:&Sparse,central:&[usize],threshold:f64)->Vec<Vec<usize>>{
    let n=a.len();let mut parent:Vec<_>=(0..n).collect();let mut size=vec![1usize;n];
    fn root(p:&mut[usize],mut i:usize)->usize{while p[i]!=i{p[i]=p[p[i]];i=p[i];}i}
    fn union(p:&mut[usize],size:&mut[usize],i:usize,j:usize){let(mut x,mut y)=(root(p,i),root(p,j));if x==y{return;}if size[x]<size[y]{std::mem::swap(&mut x,&mut y);}p[y]=x;size[x]+=size[y];}
    let mut is_central=vec![false;n];for &i in central{is_central[i]=true;if let Some(&first)=central.first(){union(&mut parent,&mut size,first,i);}}
    for(j,col)in a.iter().enumerate(){
        // Undamped harmonic quadratures have zero diagonal. Their companion
        // coordinate must stay in the block even when its frequency is below
        // the optional graph cutoff, otherwise a valid operator gets a
        // singular one-coordinate preconditioner.
        if !is_central[j]&&col.iter().filter(|(i,_)|*i==j).map(|(_,v)|v.abs()).sum::<f64>()==0.{
            if let Some(&(i,_))=col.iter().filter(|(i,_)|*i!=j&&!is_central[*i]).max_by(|a,b|a.1.abs().total_cmp(&b.1.abs())){union(&mut parent,&mut size,i,j);}
        }
        for &(i,v)in col{if i!=j&&!is_central[i]&&!is_central[j]&&v.abs()>threshold{union(&mut parent,&mut size,i,j);}}
    }
    let mut groups=std::collections::BTreeMap::<usize,Vec<usize>>::new();for i in 0..n{groups.entry(root(&mut parent,i)).or_default().push(i);}groups.into_values().collect()
}
fn bounded_groups(a:&Sparse,central:&[usize],threshold:f64,maximum:usize,adaptive:bool)->Result<(Vec<Vec<usize>>,f64),String>{
    if central.len()>maximum{return Err("Central preconditioner block is too large".into());}
    let mut cutoff=threshold;let mut groups=groups_at(a,central,cutoff);
    if adaptive{let ceiling=a.iter().flatten().map(|(_,v)|v.abs()).fold(0.,f64::max);while groups.iter().any(|g|g.len()>maximum)&&cutoff<ceiling{cutoff=(cutoff*2.).max(1e-12).min(ceiling);groups=groups_at(a,central,cutoff);}}
    if groups.iter().any(|g|g.len()>maximum){return Err("Preconditioner block is too large".into());}Ok((groups,cutoff))
}
/// Keep slow central coordinates together; fast coherences use their own
/// blocks. This partitions ONLY the preconditioner, never the equation.
pub(super) fn slow_central(a: &Sparse, central: &[usize], omega: f64) -> Vec<usize> {
    let cutoff = (0.25 * omega).max(100.);
    central.iter().copied().filter(|&j| a[j].iter().all(|(_,v)|v.abs()<=cutoff)).collect()
}
struct Block {
    indices: Vec<usize>,
    lu: Vec<f64>,
    pivots: Vec<i32>,
    inverse: Option<Vec<f64>>,
    sparse_lu: Option<faer::sparse::linalg::solvers::Lu<usize, f64>>,
}
#[derive(Default)]
struct GmresProfile {
    enabled:bool,steps:usize,original_checks:usize,early_checks:usize,accepted_early:bool,
    precondition_ms:f64,product_ms:f64,orthogonal_ms:f64,true_check_ms:f64,
    recycle_vectors:usize,recycle_used:bool,recycle_rejected:bool,recycle_stored:bool,recycle_products:usize,
    recycle_apply_ms:f64,recycle_update_ms:f64,recycle_product_ms:f64,
}
struct CorrectionDirection { w:Vec<f64>,u:Vec<f64> }
#[derive(Default)]
struct CorrectionSpaces { forward:Vec<CorrectionDirection>,transpose:Vec<CorrectionDirection> }
impl CorrectionSpaces {
    fn side(&self,transpose:bool)->&Vec<CorrectionDirection>{if transpose{&self.transpose}else{&self.forward}}
    fn side_mut(&mut self,transpose:bool)->&mut Vec<CorrectionDirection>{if transpose{&mut self.transpose}else{&mut self.forward}}
}
fn vector_norm(v:&[f64])->f64{v.iter().map(|v|v*v).sum::<f64>().sqrt()}
fn gmres_stamp(enabled:bool)->Option<std::time::Instant>{enabled.then(std::time::Instant::now)}
fn gmres_elapsed(start:Option<std::time::Instant>)->f64{start.map_or(0.,|t|t.elapsed().as_secs_f64()*1000.)}
fn gmres_candidate(x:&[f64],basis:&[Vec<f64>],h:&[f64],g:&[f64],used:usize,stride:usize)->Vec<f64>{
    let mut y=g[..used].to_vec();
    for i in (0..used).rev(){for j in i+1..used{y[i]-=h[i+j*stride]*y[j];}y[i]/=h[i+i*stride];}
    let mut candidate=x.to_vec();
    for j in 0..used{for i in 0..candidate.len(){candidate[i]+=basis[j][i]*y[j];}}
    candidate
}
pub(super) struct Prepared {
    pub matrix: Sparse,
    pub scales: Vec<f64>,
    norm_one:f64,
    norm_infty:f64,
    blocks: Vec<Block>,
    schur:Option<crate::schur_shift::Partition>,
    sweep:bool,
    membership:Vec<usize>,
    transpose_matrix:Sparse,
    custom:Option<CustomApply>,
    custom_transpose:Option<CustomApply>,
    native_ilu:Option<Rc<crate::native_superlu::Factor>>,
    extension:Option<Rc<RawPreconditioner>>,
    extension_retired:Cell<bool>,
    fresh_native:RefCell<Option<Rc<crate::native_superlu::Factor>>>,
    raw_product:Option<RawProduct>,
    raw_transpose:Option<RawProduct>,
    probe_identity:u64,
    // This object owns one fixed balanced A. Neither directions nor solved
    // densities cross Prepared/request boundaries; transpose has its own pool.
    corrections:RefCell<CorrectionSpaces>,
}
impl Prepared {
    pub fn new(
        lap: &Lapack,
        a: &Sparse,
        central: &[usize],
        threshold: f64,
    ) -> Result<Self, String> {
        let n = a.len();
        let threshold=threshold*std::env::var("ATOM_BLOCK_THRESHOLD_SCALE").ok().and_then(|s|s.parse::<f64>().ok()).filter(|v|v.is_finite()&&*v>0.).unwrap_or(1.);
        let maximum_block=std::env::var("ATOM_BLOCK_MAX").ok().and_then(|s|s.parse::<usize>().ok()).filter(|v|*v>=128&&*v<=4096).unwrap_or(1024);
        let mut scales = vec![0_f64; n];
        for col in a {
            for &(i, v) in col {
                scales[i] = scales[i].max(v.abs());
            }
        }
        if scales.iter().any(|v| *v == 0.) {
            return Err("Zero preconditioner row".into());
        }
        let matrix: Sparse = a
            .iter()
            .map(|col| col.iter().map(|&(i, v)| (i, v / scales[i])).collect())
            .collect();
        let(cached_one,cached_infty)=operator_norms(&matrix);
        if n>=1024&&std::env::var("ATOM_NATIVE_ILUTP").is_ok_and(|value|value=="1"){
            match native_factor(&matrix){
                Ok(factor)=>{
                    if std::env::var_os("ATOM_GMRES_PROFILE").is_some(){eprintln!("Native ILUTP prepared {}",serde_json::to_string(&factor.diagnostics()).unwrap_or_default());}
                    return Ok(Self{matrix,scales,norm_one:cached_one,norm_infty:cached_infty,blocks:Vec::new(),schur:None,sweep:false,membership:Vec::new(),transpose_matrix:Vec::new(),custom:None,custom_transpose:None,
                        native_ilu:Some(factor),extension:None,extension_retired:Cell::new(false),fresh_native:RefCell::new(None),raw_product:None,raw_transpose:None,probe_identity:crate::matrix_probe::new_identity(),corrections:RefCell::new(CorrectionSpaces::default())});
                },
                Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Native ILUTP unavailable; original block preconditioner retained: {error}");}},
            }
        }
        let (groups,selected_threshold)=bounded_groups(a,central,threshold,maximum_block,std::env::var_os("ATOM_BLOCK_LEGACY_GROUPS").is_none())?;
        if std::env::var_os("ATOM_GMRES_PROFILE").is_some(){eprintln!("Block grouping n={n} threshold={selected_threshold:e} initial={threshold:e} blocks={} largest={}",groups.len(),groups.iter().map(|g|g.len()).max().unwrap_or(0));}
        let mut blocks = Vec::new();
        let mut local = vec![usize::MAX; n];
        for indices in groups {
            let m = indices.len();
            if m > maximum_block {
                return Err("Preconditioner block is too large".into());
            }
            for (i, &g) in indices.iter().enumerate() {
                local[g] = i;
            }
            // Broad central blocks become nearly dense after factorization.
            // LAPACK's blocked LU avoids sparse bookkeeping on every RHS.
            if m > 8 && m < 128 {
                let mut triplets = Vec::new();
                for (j, &g) in indices.iter().enumerate() {
                    for &(row, v) in &matrix[g] {
                        if local[row] != usize::MAX {
                            triplets.push(Triplet::new(local[row], j, v));
                        }
                    }
                }
                let sparse = SparseColMat::<usize, f64>::try_new_from_triplets(m, m, &triplets)
                    .map_err(|e| format!("Preconditioner sparse matrix: {e:?}"))?;
                let sparse_lu = sparse
                    .sp_lu()
                    .map_err(|e| format!("Preconditioner sparse LU: {e:?}"))?;
                for &g in &indices {
                    local[g] = usize::MAX;
                }
                blocks.push(Block {
                    indices,
                    lu: Vec::new(),
                    pivots: Vec::new(),
                    inverse: None,
                    sparse_lu: Some(sparse_lu),
                });
            } else {
                let mut dense = vec![0.; m * m];
                for (j, &g) in indices.iter().enumerate() {
                    for &(row, v) in &matrix[g] {
                        if local[row] != usize::MAX {
                            dense[local[row] + j * m] += v;
                        }
                    }
                }
                let (lu, pivots) = lap.factor(&dense, m)?;
                let condition = lap.condition(&lu, m, norm_one(&dense, m))?;
                if !condition.is_finite() || condition > 1e13 {
                    return Err("Ill-conditioned preconditioner block".into());
                }
                let inverse = if m<=8 {
                    let mut identity = vec![0.; m * m];
                    for i in 0..m {identity[i + i * m] = 1.;}
                    Some(lap.solve_matrix_factored(&lu, &pivots, &identity, m)?)
                } else {None};
                for &g in &indices {
                    local[g] = usize::MAX;
                }
                blocks.push(Block {
                    indices,
                    lu,
                    pivots,
                    inverse,
                    sparse_lu: None,
                });
            }
        }
        let sweep=std::env::var_os("ATOM_BLOCK_SWEEP").is_some();
        let mut membership=vec![0;n];for(i,block)in blocks.iter().enumerate(){for &j in &block.indices{membership[j]=i;}}
        let mut transpose_matrix=if sweep{vec![Vec::new();n]}else{Vec::new()};
        if sweep{for(j,col)in matrix.iter().enumerate(){for &(i,v)in col{transpose_matrix[i].push((j,v));}}}
        Ok(Self {
            matrix,
            scales,
            norm_one:cached_one,norm_infty:cached_infty,
            blocks,
            schur:None,
            sweep,membership,transpose_matrix,custom:None,custom_transpose:None,native_ilu:None,extension:None,extension_retired:Cell::new(false),fresh_native:RefCell::new(None),raw_product:None,raw_transpose:None,probe_identity:crate::matrix_probe::new_identity(),
            corrections:RefCell::new(CorrectionSpaces::default()),
        })
    }
    pub fn new_schur(lap:&Lapack,a:&Sparse,shared:std::rc::Rc<crate::schur_shift::Shared>,omega:f64,shift:f64,weighted:bool)->Result<Self,String>{
        let schur=crate::schur_shift::Partition::new(lap,a,shared,omega,shift,weighted)?;
        let mut scales=vec![0_f64;a.len()];for col in a{for &(i,v)in col{scales[i]=scales[i].max(v.abs());}}
        if scales.iter().any(|s|*s==0.){return Err("Zero Schur preconditioner row".into());}
        let matrix=a.iter().map(|col|col.iter().map(|&(i,v)|(i,v/scales[i])).collect()).collect();
        let(cached_one,cached_infty)=operator_norms(&matrix);
        Ok(Self{matrix,scales,norm_one:cached_one,norm_infty:cached_infty,blocks:Vec::new(),schur:Some(schur),sweep:false,membership:Vec::new(),transpose_matrix:Vec::new(),custom:None,custom_transpose:None,native_ilu:None,extension:None,extension_retired:Cell::new(false),fresh_native:RefCell::new(None),raw_product:None,raw_transpose:None,probe_identity:crate::matrix_probe::new_identity(),corrections:RefCell::new(CorrectionSpaces::default())})
    }
    pub fn with_custom(a:&Sparse,custom:CustomApply)->Result<Self,String>{
        let mut scales=vec![0_f64;a.len()];for col in a{for &(i,v)in col{scales[i]=scales[i].max(v.abs());}}
        if scales.iter().any(|s|*s==0.){return Err("Zero custom-preconditioner row".into());}
        let matrix=a.iter().map(|col|col.iter().map(|&(i,v)|(i,v/scales[i])).collect()).collect();
        let(cached_one,cached_infty)=operator_norms(&matrix);
        Ok(Self{matrix,scales,norm_one:cached_one,norm_infty:cached_infty,blocks:Vec::new(),schur:None,sweep:false,membership:Vec::new(),transpose_matrix:Vec::new(),custom:Some(custom),custom_transpose:None,native_ilu:None,extension:None,extension_retired:Cell::new(false),fresh_native:RefCell::new(None),raw_product:None,raw_transpose:None,probe_identity:crate::matrix_probe::new_identity(),corrections:RefCell::new(CorrectionSpaces::default())})
    }
    /// Transpose of an ORIGINAL-coordinate custom inverse. For A_bal=S^-1 A,
    /// the corresponding inverse transpose is S P_raw^T; scaling is applied
    /// after the callback, not to its input.
    pub fn with_custom_transpose(mut self,apply:CustomApply)->Self{self.custom_transpose=Some(apply);self}
    pub fn operator_norm(&self,transpose:bool)->f64{if transpose{self.norm_infty}else{self.norm_one}}
    pub fn matches_original(&self,a:&Sparse)->bool{
        a.len()==self.matrix.len()&&a.iter().zip(&self.matrix).all(|(raw,scaled)|raw.len()==scaled.len()
            &&raw.iter().zip(scaled).all(|(&(i,value),&(j,balanced))|i==j&&i<self.scales.len()&&value/self.scales[i]==balanced))
    }
    pub fn raw_preconditioner(&self)->Option<Rc<RawPreconditioner>>{
        let factor=self.fresh_native.borrow().clone().or_else(||self.native_ilu.clone());
        if let Some(factor)=factor{
            let scales=self.scales.clone();let dimension=scales.len();
            return Some(Rc::new(RawPreconditioner{dimension,apply:Rc::new(move|_,rhs,transpose|{
                if transpose{
                    let mut result=factor.solve(rhs,true)?;
                    for(value,scale)in result.iter_mut().zip(&scales){*value/=scale;}return Ok(result);
                }
                let scaled:Vec<_>=rhs.iter().zip(&scales).map(|(b,s)|b/s).collect();factor.solve(&scaled,false)
            })}));
        }
        if self.extension_retired.get(){None}else{self.extension.clone()}
    }
    pub fn has_fresh_native(&self)->bool{self.fresh_native.borrow().is_some()}
    pub fn is_harmonic_extension(&self)->bool{self.extension.is_some()}
    /// Restart numerical refinement with a full-order factor. The operator,
    /// row scales, structured products and original residual gate are unchanged.
    pub fn refresh_native_for_refinement(&self)->Result<(),String>{
        let factor=native_factor(&self.matrix)?;
        *self.fresh_native.borrow_mut()=Some(factor);self.extension_retired.set(true);Ok(())
    }
    pub fn extend_raw(lap:&Lapack,a:&Sparse,old_matrix:&Sparse,old:Rc<RawPreconditioner>,old_to_new:&[usize],threshold:f64)->Result<Self,String>{
        let n=a.len();let m=old_matrix.len();
        if m==0||m>=n||old.dimension!=m||old_to_new.len()!=m{return Err("Invalid harmonic preconditioner extension dimensions".into());}
        if a.iter().flatten().any(|&(row,value)|row>=n||!value.is_finite())||old_matrix.iter().flatten().any(|&(row,value)|row>=m||!value.is_finite()){
            return Err("Invalid original harmonic extension matrix".into());
        }
        let mut reverse=vec![usize::MAX;n];
        for(local,&global)in old_to_new.iter().enumerate(){
            if global>=n||reverse[global]!=usize::MAX{return Err("Harmonic preconditioner embedding is not injective".into());}
            reverse[global]=local;
        }
        for(local,&global)in old_to_new.iter().enumerate(){
            let mut principal:Vec<_>=a[global].iter().filter_map(|&(row,value)|{
                (row<n&&reverse[row]!=usize::MAX).then(||(reverse[row],value))
            }).collect();principal.sort_by_key(|entry|entry.0);
            let mut expected=old_matrix[local].clone();expected.sort_by_key(|entry|entry.0);
            if principal!=expected{return Err("The embedded original shifted operator changed; prepare a fresh factor".into());}
        }
        let shell:Vec<_>=(0..n).filter(|&i|reverse[i]==usize::MAX).collect();let mut shell_position=vec![usize::MAX;n];
        for(local,&global)in shell.iter().enumerate(){shell_position[global]=local;}
        let mut lower=vec![vec![];m];
        for(local,&global)in old_to_new.iter().enumerate(){for &(row,value)in &a[global]{
            if row>=n||!value.is_finite(){return Err("Invalid extended shifted matrix entry".into());}
            if shell_position[row]!=usize::MAX{lower[local].push((shell_position[row],value));}
        }}
        let shell_matrix:Sparse=shell.iter().map(|&global|a[global].iter().filter_map(|&(row,value)|
            (row<n&&shell_position[row]!=usize::MAX).then(||(shell_position[row],value))).collect()).collect();
        let shell_factor=RawPreconditioner::from_prepared(Self::new(lap,&shell_matrix,&[],threshold)?);
        let old_to_new=old_to_new.to_vec();
        let raw=Rc::new(RawPreconditioner{dimension:n,apply:Rc::new(move|lap,rhs,transpose|{
            let mut old_rhs:Vec<_>=old_to_new.iter().map(|&i|rhs[i]).collect();
            let mut shell_rhs:Vec<_>=shell.iter().map(|&i|rhs[i]).collect();
            let(old_value,shell_value)=if transpose{
                // [[B,0],[C,D]]^-T: solve the NEW shell first, then pass
                // C^T back to the old coordinates. Every factor uses its
                // own raw transpose, including its original row scales.
                let shell_value=shell_factor.apply_transpose(lap,&shell_rhs)?;
                for(j,column)in lower.iter().enumerate(){for &(i,value)in column{old_rhs[j]-=value*shell_value[i];}}
                (old.apply_transpose(lap,&old_rhs)?,shell_value)
            }else{
                let old_value=old.apply(lap,&old_rhs)?;
                for(j,column)in lower.iter().enumerate(){for &(i,value)in column{shell_rhs[i]-=value*old_value[j];}}
                (old_value,shell_factor.apply(lap,&shell_rhs)?)
            };
            let mut result=vec![0.;n];
            for(&i,&value)in old_to_new.iter().zip(&old_value){result[i]=value;}
            for(&i,&value)in shell.iter().zip(&shell_value){result[i]=value;}
            Ok(result)
        })});
        let shared=Rc::clone(&raw);let mut prepared=Self::with_custom(a,Box::new(move|lap,rhs|shared.apply(lap,rhs)))?;
        prepared.extension=Some(raw);Ok(prepared)
    }
    fn precondition(&self, lap: &Lapack, b: &[f64],transpose:bool) -> Result<Vec<f64>, String> {
        if let Some(factor)=self.fresh_native.borrow().as_ref(){return factor.solve(b,transpose);}
        if self.extension_retired.get(){return Err("Retired harmonic extension requires the full original fallback".into());}
        if let Some(factor)=&self.native_ilu{return factor.solve(b,transpose);}
        if transpose{if let Some(extension)=&self.extension{
            // The outer equation is S_new^-1*A. Its transpose inverse is
            // S_new*A^-T; this scaling belongs AFTER the raw solve.
            let mut result=extension.apply_transpose(lap,b)?;
            for(value,scale)in result.iter_mut().zip(&self.scales){*value*=scale;}return Ok(result);
        }}
        if let Some(custom)=&self.custom{
            if transpose{
                let apply=self.custom_transpose.as_ref().ok_or("Custom transpose preconditioner unavailable")?;
                let mut result=apply(lap,b)?;
                if result.len()!=self.scales.len(){return Err("Custom transpose preconditioner dimension mismatch".into());}
                for(value,&scale)in result.iter_mut().zip(&self.scales){*value*=scale;}
                return Ok(result);
            }
            let rhs:Vec<_>=b.iter().zip(&self.scales).map(|(b,s)|b*s).collect();return custom(lap,&rhs);
        }
        if let Some(schur)=&self.schur{return schur.apply(lap,b,&self.scales,transpose);}
        let mut x = vec![0.; b.len()];
        let mut working=if self.sweep{b.to_vec()}else{Vec::new()};
        for ordinal in 0..self.blocks.len() {
            let bi=if transpose&&self.sweep{self.blocks.len()-1-ordinal}else{ordinal};let block=&self.blocks[bi];
            let rhs=if self.sweep{&working[..]}else{b};
            let m = block.indices.len();
            if let Some(inverse) = &block.inverse {
                for (i, &g) in block.indices.iter().enumerate() {
                    x[g] = block
                        .indices
                        .iter()
                        .enumerate()
                        .map(|(j, &h)| inverse[if transpose{j+i*m}else{i+j*m}] * rhs[h])
                        .sum();
                }
            } else if let Some(lu) = &block.sparse_lu {
                let rhs = Mat::<f64>::from_fn(m, 1, |i, _| rhs[block.indices[i]]);
                let y = if transpose{lu.solve_transpose(&rhs)}else{lu.solve(&rhs)};
                for (i, &g) in block.indices.iter().enumerate() {
                    x[g] = y[(i, 0)];
                }
            } else {
                let rhs: Vec<_> = block.indices.iter().map(|&i| rhs[i]).collect();
                let y = if transpose{lap.solve_factored_transpose(&block.lu,&block.pivots,&rhs,m)?}else{lap.solve_factored(&block.lu, &block.pivots, &rhs, m)?};
                for (&i, v) in block.indices.iter().zip(y) {
                    x[i] = v;
                }
            }
            // A triangular block preconditioner incorporates cross-block
            // feedback in its sweep; GMRES still tests the complete matrix.
            if self.sweep{let entries=if transpose{&self.transpose_matrix}else{&self.matrix};for &j in &block.indices{for &(i,v)in &entries[j]{if if transpose{self.membership[i]<bi}else{self.membership[i]>bi}{working[i]-=v*x[j];}}}}
        }
        Ok(x)
    }

    pub fn with_raw_product(mut self,product:RawProduct)->Self{self.raw_product=Some(product);self}
    pub fn with_raw_transpose(mut self,product:RawProduct)->Self{self.raw_transpose=Some(product);self}
    fn product(&self,x:&[f64],transpose:bool)->Vec<f64>{
        if transpose{if let Some(product)=&self.raw_transpose{
            let input:Vec<_>=x.iter().zip(&self.scales).map(|(x,s)|x/s).collect();
            let result=product(&input);
            return if result.len()==self.matrix.len(){result}else{vec![f64::NAN;self.matrix.len()]};
        }}
        if !transpose{if let Some(product)=&self.raw_product{
            let mut result=product(x);
            if result.len()!=self.matrix.len(){return vec![f64::NAN;self.matrix.len()];}
            for(value,scale)in result.iter_mut().zip(&self.scales){*value/=scale;}
            return result;
        }}
        self.sparse_product(x,transpose)
    }
    fn sparse_product(&self, x: &[f64],transpose:bool) -> Vec<f64> {
        let mut y = vec![0.; x.len()];
        for (j, col) in self.matrix.iter().enumerate() {
            for &(i, v) in col {
                if transpose {y[j]+=v*x[i];}else{y[i] += v * x[j];}
            }
        }
        y
    }
    /// A structured product can propose convergence, but never certify itself.
    /// The unchanged explicit matrix remains the final residual oracle.
    fn sparse_confirmation(&self,x:&[f64],b:&[f64],transpose:bool,tolerance:f64)->bool{
        if (transpose&&self.raw_transpose.is_none())||(!transpose&&self.raw_product.is_none()){return true;}
        let residual=self.sparse_product(x,transpose);
        let norm=b.iter().zip(residual).map(|(b,a)|(b-a)*(b-a)).sum::<f64>().sqrt();
        norm<tolerance
    }
    pub fn solve(&self, lap: &Lapack, rhs: &[f64]) -> Result<Vec<f64>, String> {
        let capture=crate::matrix_probe::begin(self.probe_identity,&self.matrix,&self.scales)?;
        let b: Vec<_> = rhs.iter().zip(&self.scales).map(|(b, s)| b / s).collect();
        let result=self.solve_scaled_with_fallback(lap,&b,false);
        if capture{if let Ok(solution)=&result{crate::matrix_probe::record(self.probe_identity,&b,rhs,solution)?;}}
        result
    }
    fn solve_scaled_with_fallback(&self,lap:&Lapack,b:&[f64],transpose:bool)->Result<Vec<f64>,String>{
        let mut result=self.solve_scaled(lap,b,transpose);
        if result.is_err()&&self.extension.is_some()&&!self.extension_retired.replace(true){
            if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Harmonic extension stalled; restart with a fresh full-order ILUTP");}
            match native_factor(&self.matrix){
                Ok(factor)=>{*self.fresh_native.borrow_mut()=Some(factor);result=self.solve_scaled(lap,b,transpose);},
                Err(error)=>{if std::env::var_os("ATOM_SOLVER_DEBUG").is_some(){eprintln!("Fresh full-order ILUTP unavailable: {error}");}},
            }
        }
        result
    }
    /// Transpose of the equilibrated matrix; reuse the very same block factors.
    pub fn solve_scaled_transpose(&self,lap:&Lapack,rhs:&[f64])->Result<Vec<f64>,String>{self.solve_scaled_with_fallback(lap,rhs,true)}
    fn correction_product(&self,x:&[f64],transpose:bool,profile:&mut GmresProfile)->Vec<f64>{
        let time=gmres_stamp(profile.enabled);
        // Always the explicit ORIGINAL balanced matrix, including all drive
        // entries. A structured callback cannot certify a recycled direction.
        let result=self.sparse_product(x,transpose);
        profile.recycle_product_ms+=gmres_elapsed(time);profile.recycle_products+=1;result
    }
    /// Initial-guess projection only, NOT full LGMRES/GCRO Arnoldi augmentation.
    /// Related stored-(v,Av) interface: SciPy's primary LGMRES documentation,
    /// https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lgmres.html
    fn correction_guess(&self,b:&[f64],base:&[f64],transpose:bool,profile:&mut GmresProfile)->Option<Vec<f64>>{
        let time=gmres_stamp(profile.enabled);let product_before=profile.recycle_product_ms;
        let attempt=(||{
            let spaces=self.corrections.borrow();let pool=spaces.side(transpose);profile.recycle_vectors=pool.len();
            if pool.is_empty(){return None;}
            if pool.iter().any(|d|d.w.len()!=base.len()||d.u.len()!=base.len()||d.w.iter().chain(&d.u).any(|v|!v.is_finite())){
                profile.recycle_rejected=true;return None;
            }
            let mut residual:Vec<_>=b.iter().zip(self.correction_product(base,transpose,profile)).map(|(b,a)|b-a).collect();
            let before=vector_norm(&residual);if !before.is_finite(){profile.recycle_rejected=true;return None;}if before==0.{return None;}
            let mut candidate=base.to_vec();
            // Two passes retain the u-space projection despite accumulated
            // orthogonalization roundoff. Update x and residual consistently.
            for _ in 0..2{for direction in pool{
                let amount=direction.u.iter().zip(&residual).map(|(u,r)|u*r).sum::<f64>();
                for i in 0..candidate.len(){candidate[i]+=amount*direction.w[i];residual[i]-=amount*direction.u[i];}
            }}
            if candidate.iter().any(|v|!v.is_finite()){profile.recycle_rejected=true;return None;}
            let actual:Vec<_>=b.iter().zip(self.correction_product(&candidate,transpose,profile)).map(|(b,a)|b-a).collect();
            let after=vector_norm(&actual);
            // A bad/stale pair may predict zero residual. Judge its actual
            // effect using A, never the projected residual or a relaxed gate.
            let mismatch=actual.iter().zip(&residual).map(|(a,p)|(a-p)*(a-p)).sum::<f64>().sqrt();
            let bound=128.*f64::EPSILON*(b.iter().map(|v|v.abs()).sum::<f64>()+
                self.operator_norm(transpose)*candidate.iter().chain(base).map(|v|v.abs()).sum::<f64>());
            if !after.is_finite()||!mismatch.is_finite()||mismatch>bound||after>before*(1.+64.*f64::EPSILON){profile.recycle_rejected=true;return None;}
            profile.recycle_used=true;Some(candidate)
        })();
        if profile.recycle_rejected{self.corrections.borrow_mut().side_mut(transpose).clear();}
        profile.recycle_apply_ms+=(gmres_elapsed(time)-(profile.recycle_product_ms-product_before)).max(0.);
        attempt
    }
    fn remember_correction(&self,solution:&[f64],base:&[f64],transpose:bool,profile:&mut GmresProfile){
        let time=gmres_stamp(profile.enabled);let product_before=profile.recycle_product_ms;
        let direction=(||{
            let mut w:Vec<_>=solution.iter().zip(base).map(|(x,p)|x-p).collect();
            if w.iter().any(|v|!v.is_finite())||w.iter().all(|v|*v==0.){return None;}
            let mut u=self.correction_product(&w,transpose,profile);let original=vector_norm(&u);
            if !original.is_finite()||original==0.{return None;}
            let spaces=self.corrections.borrow();let pool=spaces.side(transpose);
            for _ in 0..2{for previous in pool{
                let dot=u.iter().zip(&previous.u).map(|(u,v)|u*v).sum::<f64>();
                for i in 0..u.len(){u[i]-=dot*previous.u[i];w[i]-=dot*previous.w[i];}
            }}
            let length=vector_norm(&u);
            // Numerical dependence only; dropping an augmentation direction
            // does not alter A, its physical coordinates, or its solve gate.
            if !length.is_finite()||length<=1e-12*original{return None;}
            for i in 0..u.len(){u[i]/=length;w[i]/=length;}
            if u.iter().chain(&w).any(|v|!v.is_finite()){return None;}
            Some(CorrectionDirection{w,u})
        })();
        if let Some(direction)=direction{
            let mut spaces=self.corrections.borrow_mut();let pool=spaces.side_mut(transpose);
            if pool.len()==8{pool.remove(0);}pool.push(direction);profile.recycle_stored=true;
        }
        profile.recycle_update_ms+=(gmres_elapsed(time)-(profile.recycle_product_ms-product_before)).max(0.);
    }
    fn solve_scaled(&self,lap:&Lapack,b:&[f64],transpose:bool)->Result<Vec<f64>,String>{
        let started=std::time::Instant::now();
        let mut profile=GmresProfile{enabled:std::env::var_os("ATOM_GMRES_PROFILE").is_some(),..GmresProfile::default()};
        let interval=if std::env::var("ATOM_GMRES_TRUE_CHECK").is_ok_and(|value|value=="1"){4}else{0};
        let recycle=std::env::var("ATOM_GMRES_RECYCLE").is_ok_and(|value|value=="1");
        let result=self.solve_scaled_inner(lap,b,transpose,interval,recycle,&mut profile);
        if profile.enabled{eprintln!("GMRES n={} transpose={transpose} steps={} blocks={} largest={} ms={:.3} preconditionMs={:.3} productMs={:.3} orthogonalMs={:.3} trueCheckMs={:.3} originalChecks={} extraChecks={} acceptedEarly={} success={} recycleVectors={} recycleUsed={} recycleRejected={} recycleStored={} recycleProjectionMs={:.3} recycleUpdateMs={:.3} recycleProductMs={:.3} recycleProducts={}",
            b.len(),profile.steps,self.blocks.len(),self.blocks.iter().map(|b|b.indices.len()).max().unwrap_or(0),started.elapsed().as_secs_f64()*1000.,
            profile.precondition_ms,profile.product_ms,profile.orthogonal_ms,profile.true_check_ms,profile.original_checks,profile.early_checks,profile.accepted_early,result.is_ok(),
            profile.recycle_vectors,profile.recycle_used,profile.recycle_rejected,profile.recycle_stored,profile.recycle_apply_ms,profile.recycle_update_ms,profile.recycle_product_ms,profile.recycle_products);}
        result
    }
    // Explicit switches avoid environment races in the algebra regressions.
    // Recycling changes only the initial guess; the original GMRES iterations,
    // operator confirmations and accepted tolerance below remain unchanged.
    fn solve_scaled_inner(&self,lap:&Lapack,b:&[f64],transpose:bool,true_check_every:usize,recycle:bool,profile:&mut GmresProfile)->Result<Vec<f64>,String>{
        let time=gmres_stamp(profile.enabled);let first=self.precondition(lap,b,transpose);profile.precondition_ms+=gmres_elapsed(time);let base=first?;
        if !recycle{return self.solve_scaled_iterations(lap,b,transpose,true_check_every,base,profile);}
        let initial=self.correction_guess(b,&base,transpose,profile);
        let augmented=initial.is_some();
        let mut result=self.solve_scaled_iterations(lap,b,transpose,true_check_every,initial.unwrap_or_else(||base.clone()),profile);
        if augmented&&result.is_err(){
            self.corrections.borrow_mut().side_mut(transpose).clear();profile.recycle_rejected=true;
            result=self.solve_scaled_iterations(lap,b,transpose,true_check_every,base.clone(),profile);
        }
        if let Ok(solution)=&result{self.remember_correction(solution,&base,transpose,profile);}
        else{self.corrections.borrow_mut().side_mut(transpose).clear();}
        result
    }
    fn solve_scaled_iterations(&self,lap:&Lapack,b:&[f64],transpose:bool,true_check_every:usize,mut x:Vec<f64>,profile:&mut GmresProfile)->Result<Vec<f64>,String>{
        let n=b.len();let norm=vector_norm;
        let scale=norm(b).max(1e-30);let rhs_one=b.iter().map(|v|v.abs()).sum::<f64>();
        let operator_norm=self.operator_norm(transpose);
        let tolerance=|candidate:&[f64]|(1e-11*scale).max(32.*f64::EPSILON*(operator_norm*candidate.iter().map(|v|v.abs()).sum::<f64>()+rhs_one));
        for _restart in 0..4{
            let time=gmres_stamp(profile.enabled);
            let residual:Vec<_>=b.iter().zip(self.product(&x,transpose)).map(|(b,a)|b-a).collect();
            let accepted=norm(&residual)<tolerance(&x)&&self.sparse_confirmation(&x,b,transpose,tolerance(&x));profile.original_checks+=1;profile.true_check_ms+=gmres_elapsed(time);
            if accepted{return Ok(x);}
            let time=gmres_stamp(profile.enabled);let preconditioned=self.precondition(lap,&residual,transpose);profile.precondition_ms+=gmres_elapsed(time);let r=preconditioned?;
            let beta=norm(&r);if !beta.is_finite()||beta==0.{return Err("Invalid preconditioned residual".into());}
            let mut basis=vec![r.iter().map(|v|v/beta).collect::<Vec<_>>()];let m=32.min(n);
            let mut h=vec![0.;(m+1)*m];let mut cs=vec![0.;m];let mut sn=vec![0.;m];let mut g=vec![0.;m+1];g[0]=beta;
            let mut used=0;let mut last_candidate:Option<(usize,Vec<f64>)>=None;
            for j in 0..m{
                profile.steps+=1;
                let time=gmres_stamp(profile.enabled);let product=self.product(&basis[j],transpose);profile.product_ms+=gmres_elapsed(time);
                let time=gmres_stamp(profile.enabled);let preconditioned=self.precondition(lap,&product,transpose);profile.precondition_ms+=gmres_elapsed(time);let mut w=preconditioned?;
                let time=gmres_stamp(profile.enabled);
                for _ in 0..2{for i in 0..=j{
                    let dot=w.iter().zip(&basis[i]).map(|(a,b)|a*b).sum::<f64>();h[i+j*(m+1)]+=dot;
                    for(k,v)in w.iter_mut().enumerate(){*v-=dot*basis[i][k];}
                }}
                let length=norm(&w);h[j+1+j*(m+1)]=length;if length>1e-25{basis.push(w.iter().map(|v|v/length).collect());}
                for i in 0..j{let a=h[i+j*(m+1)];let b=h[i+1+j*(m+1)];h[i+j*(m+1)]=cs[i]*a+sn[i]*b;h[i+1+j*(m+1)]=-sn[i]*a+cs[i]*b;}
                let diagonal=h[j+j*(m+1)];let subdiagonal=h[j+1+j*(m+1)];let d=diagonal.hypot(subdiagonal);
                if d==0.||!d.is_finite(){profile.orthogonal_ms+=gmres_elapsed(time);return Err("GMRES breakdown".into());}
                cs[j]=diagonal/d;sn[j]=subdiagonal/d;h[j+j*(m+1)]=d;g[j+1]=-sn[j]*g[j];g[j]*=cs[j];used=j+1;
                profile.orthogonal_ms+=gmres_elapsed(time);
                if true_check_every>0&&used%true_check_every==0{
                    let time=gmres_stamp(profile.enabled);
                    let candidate=gmres_candidate(&x,&basis,&h,&g,used,m+1);
                    let residual:Vec<_>=b.iter().zip(self.product(&candidate,transpose)).map(|(b,a)|b-a).collect();
                    // EXACTLY the same original balanced-system gate used at
                    // restart boundaries. The cheap preconditioned estimate
                    // does not decide acceptance here.
                    let accepted=norm(&residual)<tolerance(&candidate)&&self.sparse_confirmation(&candidate,b,transpose,tolerance(&candidate));
                    profile.original_checks+=1;profile.early_checks+=1;profile.true_check_ms+=gmres_elapsed(time);
                    if accepted{profile.accepted_early=true;return Ok(candidate);}
                    last_candidate=Some((used,candidate));
                }
                if g[j+1].abs()<1e-12*beta||length<=1e-25{break;}
            }
            let time=gmres_stamp(profile.enabled);
            x=match last_candidate{Some((size,candidate))if size==used=>candidate,_=>gmres_candidate(&x,&basis,&h,&g,used,m+1)};
            profile.orthogonal_ms+=gmres_elapsed(time);
        }
        let time=gmres_stamp(profile.enabled);
        let residual:Vec<_>=b.iter().zip(self.product(&x,transpose)).map(|(b,a)|b-a).collect();
        let accepted=norm(&residual)<tolerance(&x)&&self.sparse_confirmation(&x,b,transpose,tolerance(&x));profile.original_checks+=1;profile.true_check_ms+=gmres_elapsed(time);
        if accepted{return Ok(x);}
        Err("Block-preconditioned solve did not meet the original-system residual".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn correction_fixture()->(Prepared,Vec<f64>){
        let n=12;let mut dense=vec![0.;n*n];
        for i in 0..n{
            dense[i+i*n]=2.+0.1*i as f64;
            if i+1<n{dense[i+(i+1)*n]=0.7;dense[i+1+i*n]=-0.4;}
            if i+3<n{dense[i+(i+3)*n]=0.13;}
        }
        let matrix:Sparse=(0..n).map(|j|(0..n).filter_map(|i|(dense[i+j*n]!=0.).then_some((i,dense[i+j*n]))).collect()).collect();
        let prepared=Prepared::with_custom(&matrix,Box::new(|_,b|Ok(b.to_vec()))).unwrap()
            .with_custom_transpose(Box::new(|_,b|Ok(b.to_vec())));
        for j in 0..n{for i in 0..n{dense[i+j*n]/=prepared.scales[i];}}
        (prepared,dense)
    }
    #[test]
    fn correction_space_recycling_multiple_rhs_matches_dense_forward_and_transpose(){
        let lap=Lapack::load().unwrap();let(prepared,dense)=correction_fixture();let n=prepared.matrix.len();
        let(lu,pivots)=lap.factor(&dense,n).unwrap();
        for transpose in [false,true]{
            for sample in 0..11{
                let rhs:Vec<_>=(0..n).map(|i|((i+1)as f64*(sample as f64+0.3)).sin()+if i==sample%n{0.2}else{0.}).collect();
                let mut profile=GmresProfile::default();
                let result=prepared.solve_scaled_inner(&lap,&rhs,transpose,4,true,&mut profile).unwrap();
                let expected=if transpose{lap.solve_factored_transpose(&lu,&pivots,&rhs,n)}else{lap.solve_factored(&lu,&pivots,&rhs,n)}.unwrap();
                assert!(result.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));
                let pool=prepared.corrections.borrow();let pool=pool.side(transpose);assert!(!pool.is_empty()&&pool.len()<=8);
                for(direction,index)in pool.iter().zip(0..){
                    let product=prepared.sparse_product(&direction.w,transpose);
                    assert!(product.iter().zip(&direction.u).all(|(a,b)|(a-b).abs()<1e-10));
                    for(other,j)in pool.iter().zip(0..){
                        let dot=direction.u.iter().zip(&other.u).map(|(a,b)|a*b).sum::<f64>();
                        assert!((dot-if index==j{1.}else{0.}).abs()<1e-10);
                    }
                }
            }
        }
        assert!(!prepared.corrections.borrow().forward.is_empty());assert!(!prepared.corrections.borrow().transpose.is_empty());
        let(new_matrix,_)=correction_fixture();assert!(new_matrix.corrections.borrow().forward.is_empty());assert!(new_matrix.corrections.borrow().transpose.is_empty());
    }
    #[test]
    fn correction_projection_accelerates_a_related_rhs_without_a_result_lookup(){
        let lap=Lapack::load().unwrap();
        for transpose in [false,true]{
            let(prepared,_)=correction_fixture();let rhs:Vec<_>=(0..12).map(|i|(i as f64+0.2).cos()).collect();
            let mut first=GmresProfile::default();let reference=prepared.solve_scaled_inner(&lap,&rhs,transpose,4,true,&mut first).unwrap();
            assert!(first.steps>0&&first.recycle_stored);
            let next:Vec<_>=rhs.iter().map(|b|1.7*b).collect();let mut second=GmresProfile::default();
            let result=prepared.solve_scaled_inner(&lap,&next,transpose,4,true,&mut second).unwrap();
            assert!(second.recycle_used);assert_eq!(second.steps,0);assert!(second.recycle_products>=2);
            assert!(result.iter().zip(reference).all(|(a,b)|(a-1.7*b).abs()<1e-10));
        }
    }
    #[test]
    fn bad_correction_pairs_are_discarded_and_cannot_bypass_the_original_gate(){
        let lap=Lapack::load().unwrap();
        for transpose in [false,true]{for nonfinite in [false,true]{
            let(prepared,dense)=correction_fixture();let n=12;let mut u=vec![0.;n];u[0]=1.;
            // The finite pair predicts a correction but w=0 cannot supply it.
            let mut w=vec![0.;n];if nonfinite{w[1]=f64::NAN;}
            prepared.corrections.borrow_mut().side_mut(transpose).push(CorrectionDirection{w,u});
            let rhs:Vec<_>=(0..n).map(|i|(i as f64+0.7).cos()).collect();let mut profile=GmresProfile::default();
            let result=prepared.solve_scaled_inner(&lap,&rhs,transpose,4,true,&mut profile).unwrap();
            assert!(profile.recycle_rejected);assert!(!profile.recycle_used);
            let(lu,pivots)=lap.factor(&dense,n).unwrap();
            let expected=if transpose{lap.solve_factored_transpose(&lu,&pivots,&rhs,n)}else{lap.solve_factored(&lu,&pivots,&rhs,n)}.unwrap();
            assert!(result.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));
            let mut disabled=GmresProfile::default();prepared.solve_scaled_inner(&lap,&rhs,transpose,4,false,&mut disabled).unwrap();
            assert_eq!(disabled.recycle_products,0);assert!(!disabled.recycle_used);
        }}
        let(prepared,_)=correction_fixture();let n=12;
        let wrong=prepared.with_raw_product(Box::new(|x|x.to_vec()));
        let rhs=vec![0.7;n];let mut profile=GmresProfile::default();
        // Even when a different product pretends the candidate is solved,
        // sparse_confirmation still checks the untouched original A.
        assert!(wrong.solve_scaled_inner(&lap,&rhs,false,4,true,&mut profile).is_err());
        assert!(wrong.corrections.borrow().forward.is_empty());
    }
    #[test]
    fn custom_raw_transpose_applies_current_row_scaling_on_output(){
        let lap=Lapack::load().unwrap();
        let a=vec![vec![(0,2.),(1,0.25)],vec![(0,1.),(1,40000.)]];
        let m=vec![1.,-0.2,3.,2.];let copy=m.clone();
        let prepared=Prepared::with_custom(&a,Box::new(move|lap,b|lap.solve(&copy,b,2))).unwrap()
            .with_custom_transpose(Box::new(|lap,b|lap.solve(&[1.,3.,-0.2,2.],b,2)));
        let b=[0.7,-0.4];let expected=lap.solve(&[1.,3.,-0.2,2.],&b,2).unwrap();
        let actual=prepared.precondition(&lap,&b,true).unwrap();
        assert!(actual.iter().zip(expected).zip(&prepared.scales).all(|((&a,b),s)|(a-b*s).abs()<1e-10));
        let known=[0.2,-0.3];let rhs=prepared.sparse_product(&known,true);
        let solution=prepared.solve_scaled_transpose(&lap,&rhs).unwrap();
        assert!(solution.iter().zip(known).all(|(a,b)|(a-b).abs()<1e-10));
    }
    #[test]
    fn immutable_operator_norms_match_previous_sums_for_forward_and_transpose_solves(){
        let lap=Lapack::load().unwrap();let matrix=vec![vec![(0,2.),(1,-0.3)],vec![(0,0.7),(1,4.)]];
        let prepared=Prepared::new(&lap,&matrix,&[0,1],0.).unwrap();let saved=prepared.matrix.clone();
        let expected_one=saved.iter().map(|col|col.iter().map(|(_,v)|v.abs()).sum::<f64>()).fold(0.,f64::max);
        let mut rows=vec![0.;2];for col in &saved{for &(i,v)in col{rows[i]+=v.abs();}}let expected_infty=rows.into_iter().fold(0.,f64::max);
        for x in [[0.2,-0.6],[1.1,0.4]]{
            let rhs=prepared.sparse_product(&x,true);let actual=prepared.solve_scaled_transpose(&lap,&rhs).unwrap();
            assert!(actual.iter().zip(x).all(|(a,b)|(a-b).abs()<1e-12));
            prepared.solve(&lap,&x).unwrap();
            assert_eq!(prepared.operator_norm(false).to_bits(),expected_one.to_bits());
            assert_eq!(prepared.operator_norm(true).to_bits(),expected_infty.to_bits());assert_eq!(prepared.matrix,saved);
        }
    }
    #[test]
    fn harmonic_extension_reuses_raw_scales_and_retains_full_equation_residual(){
        let lap=Lapack::load().unwrap();
        let old=vec![vec![(0,2.),(1,0.4)],vec![(0,0.3),(1,7.)]];
        // Old coordinates move to [2,0]; shell coordinates are interleaved.
        // Upper-right coupling changes row 2's scale from 2 to 8.
        let matrix=vec![vec![(0,7.),(1,-2.),(2,0.3),(3,0.7)],
            vec![(0,-1.),(1,5.),(2,8.),(3,-0.1)],
            vec![(0,0.4),(1,1.2),(2,2.),(3,0.9)],
            vec![(0,1.4),(1,0.2),(2,-0.6),(3,4.)]];
        let factor=RawPreconditioner::from_prepared(Prepared::new(&lap,&old,&[0,1],0.).unwrap());
        let prepared=Prepared::extend_raw(&lap,&matrix,&old,Rc::clone(&factor),&[2,0],0.).unwrap();
        assert_eq!(prepared.scales[2],8.);
        let rhs=vec![0.7,-0.4,1.2,0.1];let p=prepared.raw_preconditioner().unwrap().apply(&lap,&rhs).unwrap();
        assert!((2.*p[2]+0.3*p[0]-rhs[2]).abs()<1e-12);
        assert!((0.4*p[2]+7.*p[0]-rhs[0]).abs()<1e-12);
        assert!((1.2*p[2]-2.*p[0]+5.*p[1]+0.2*p[3]-rhs[1]).abs()<1e-12);
        assert!((0.9*p[2]+0.7*p[0]-0.1*p[1]+4.*p[3]-rhs[3]).abs()<1e-12);
        let expected=[0.3,-0.2,0.5,0.1];let mut right=vec![0.;4];
        for(j,column)in matrix.iter().enumerate(){for &(i,value)in column{right[i]+=value*expected[j];}}
        let actual=prepared.solve(&lap,&right).unwrap();
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));
        // Independent dense oracle for the transpose of the same LOWER
        // preconditioner. It includes old/new scale differences and relocation.
        let mut lower=vec![0.;16];for(j,col)in matrix.iter().enumerate(){for &(i,value)in col{
            if !([0,2].contains(&i)&&[1,3].contains(&j)){lower[i+j*4]=value;}
        }}
        let transposed:Vec<_>=(0..16).map(|k|lower[k/4+(k%4)*4]).collect();
        let oracle=lap.solve(&transposed,&rhs,4).unwrap();
        let raw_transpose=prepared.raw_preconditioner().unwrap().apply_transpose(&lap,&rhs).unwrap();
        assert!(raw_transpose.iter().zip(oracle).all(|(a,b)|(a-b).abs()<1e-12));
        let transpose_rhs=prepared.sparse_product(&expected,true);
        let transpose_solution=prepared.solve_scaled_transpose(&lap,&transpose_rhs).unwrap();
        assert!(transpose_solution.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-10));
        let mut changed=matrix.clone();changed[2][2].1+=1e-12;
        assert!(Prepared::extend_raw(&lap,&changed,&old,Rc::clone(&factor),&[2,0],0.).is_err());
        assert!(Prepared::extend_raw(&lap,&matrix,&old,factor,&[2,2],0.).is_err());
    }
    #[test]
    fn harmonic_extension_structured_callback_cannot_certify_an_incorrect_original_solution(){
        let lap=Lapack::load().unwrap();let old=vec![vec![(0,2.)]];
        let matrix=vec![vec![(0,2.),(1,0.3)],vec![(0,0.2),(1,3.)]];
        let factor=RawPreconditioner::from_prepared(Prepared::new(&lap,&old,&[0],0.).unwrap());
        let wrong=Prepared::extend_raw(&lap,&matrix,&old,factor,&[0],0.).unwrap()
            .with_raw_product(Box::new(|x|x.to_vec()));
        assert!(wrong.solve(&lap,&[0.7,-0.3]).is_err());
    }
    #[test]
    fn harmonic_extension_transpose_requires_the_original_scaled_residual(){
        let lap=Lapack::load().unwrap();let old=vec![vec![(0,2.)]];
        let matrix=vec![vec![(0,2.),(1,0.3)],vec![(0,0.2),(1,3.)]];
        let factor=RawPreconditioner::from_prepared(Prepared::new(&lap,&old,&[0],0.).unwrap());
        let wrong=Prepared::extend_raw(&lap,&matrix,&old,factor,&[0],0.).unwrap()
            .with_raw_transpose(Box::new(|x|x.to_vec()));
        assert!(wrong.solve_scaled_transpose(&lap,&[0.7,-0.3]).is_err());
    }
    #[test]
    fn normalized_stationary_extension_matches_dense_forward_and_all_transpose_rhs(){
        let lap=Lapack::load().unwrap();
        let old=vec![vec![(0,1.),(1,0.4)],vec![(0,1.),(1,-0.7)]];
        // The same trace row gains one NEW population coefficient. It is
        // present in the original equation even though the lower numerical
        // preconditioner omits that upper-right entry.
        let matrix=vec![vec![(0,1.),(1,0.4),(2,0.1)],vec![(0,1.),(1,-0.7),(2,0.3)],vec![(0,1.),(1,0.2),(2,-0.8)]];
        let factor=RawPreconditioner::from_prepared(Prepared::new(&lap,&old,&[0,1],0.).unwrap());
        let prepared=Prepared::extend_raw(&lap,&matrix,&old,factor,&[0,1],0.).unwrap();
        let mut dense=vec![0.;9];for(j,col)in matrix.iter().enumerate(){for &(i,value)in col{dense[i+j*3]=value;}}
        let oracle=lap.solve(&dense,&[1.,0.,0.],3).unwrap();let actual=prepared.solve(&lap,&[1.,0.,0.]).unwrap();
        assert!(actual.iter().zip(oracle).all(|(a,b)|(a-b).abs()<1e-11));assert!((actual.iter().sum::<f64>()-1.).abs()<1e-12);
        let mut scaled_transpose=vec![0.;9];for(j,col)in prepared.matrix.iter().enumerate(){for &(i,value)in col{scaled_transpose[j+i*3]=value;}}
        for column in 0..3{let mut rhs=vec![0.;3];rhs[column]=1.;
            let oracle=lap.solve(&scaled_transpose,&rhs,3).unwrap();let actual=prepared.solve_scaled_transpose(&lap,&rhs).unwrap();
            assert!(actual.iter().zip(oracle).all(|(a,b)|(a-b).abs()<1e-11));
        }
    }
    #[test]
    #[ignore="Requires the freshly built native SuperLU bridge and OpenBLAS"]
    fn stalled_harmonic_extension_restarts_with_fresh_full_order_native_factor(){
        let lap=Lapack::load().unwrap();let old=vec![vec![(0,2.)]];
        let matrix=vec![vec![(0,2.),(1,0.3)],vec![(0,0.2),(1,3.)]];
        let failed=Rc::new(RawPreconditioner{dimension:1,apply:Rc::new(|_,_,_|Err("Deliberately failed inherited preconditioner".into()))});
        let prepared=Prepared::extend_raw(&lap,&matrix,&old,failed,&[0],0.).unwrap();
        let expected=[0.2,-0.7];let rhs=prepared.sparse_product(&expected,true);
        let actual=prepared.solve_scaled_transpose(&lap,&rhs).unwrap();
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-11));
        assert!(prepared.has_fresh_native()); // fallback happened in Hager's transpose direction
        for expected in [[0.4,-0.2],[-0.1,0.8]]{
            let rhs=[2.*expected[0]+0.2*expected[1],0.3*expected[0]+3.*expected[1]];
            let actual=prepared.solve(&lap,&rhs).unwrap();
            assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-11));
            assert!(prepared.has_fresh_native());assert!(prepared.extension_retired.get());
        }
    }
    #[test]
    fn ilu_seed_retains_diagonals_and_strongest_companions_without_mutating_operator(){
        let matrix=vec![vec![(0,1.),(1,1e-8),(2,1e-9)],vec![(0,1e-9),(1,1e-12),(2,0.4)],vec![(0,1e-8),(1,1e-10),(2,0.)]];
        let original=matrix.clone();let seed=ilu_seed(&matrix,1e-6);
        assert_eq!(seed,vec![vec![(0,1.),(1,1e-8)],vec![(1,1e-12),(2,0.4)],vec![(0,1e-8),(2,0.)]]);
        assert_eq!(matrix,original);
    }
    #[test]
    fn structured_product_requires_independent_original_sparse_confirmation(){
        let lap=Lapack::load().unwrap();let matrix=vec![vec![(0,2.)],vec![(1,3.)]];let rhs=[2.,6.];
        let correct=Prepared::with_custom(&matrix,Box::new(|_,b|Ok(b.to_vec()))).unwrap()
            .with_raw_product(Box::new(|x|vec![2.*x[0],3.*x[1]]));
        let result=correct.solve(&lap,&rhs).unwrap();
        assert!((result[0]-1.).abs()<1e-12&&(result[1]-2.).abs()<1e-12);
        // The wrong callback declares the initial preconditioned vector an
        // exact solution. It MUST fail the independent explicit-matrix gate.
        let wrong=Prepared::with_custom(&matrix,Box::new(|_,b|Ok(b.to_vec()))).unwrap()
            .with_raw_product(Box::new(|x|x.to_vec()));
        assert!(wrong.solve(&lap,&rhs).is_err());
        let matrix=vec![vec![(0,2.),(1,-0.4)],vec![(0,0.7),(1,3.)]];
        let transpose=Prepared::new(&lap,&matrix,&[0,1],100.).unwrap()
            .with_raw_transpose(Box::new(|x|vec![2.*x[0]-0.4*x[1],0.7*x[0]+3.*x[1]]));
        let expected=[0.7,-0.2];
        let rhs=transpose.sparse_product(&expected,true);
        let actual=transpose.solve_scaled_transpose(&lap,&rhs).unwrap();
        assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-12));
    }
    #[test]
    fn original_residual_inner_check_accepts_nonnormal_system_without_relaxing_the_gate(){
        let lap=Lapack::load().unwrap();let n=4;
        // Strongly asymmetric upper coupling makes this nonnormal. The
        // custom device is deliberately not the exact inverse.
        let dense=vec![2.,0.03,0.,0., 5.,3.,-0.02,0., 0.,-4.,4.,0.01, 0.,0.,3.,5.];
        let matrix:Sparse=(0..n).map(|j|(0..n).filter_map(|i|(dense[i+j*n]!=0.).then_some((i,dense[i+j*n]))).collect()).collect();
        let prepared=Prepared::with_custom(&matrix,Box::new(|_,rhs|Ok(rhs.to_vec()))).unwrap();
        let known:Vec<_>=(0..n).map(|i|(i as f64+0.2).sin()).collect();
        let rhs:Vec<_>=(0..n).map(|i|(0..n).map(|j|dense[i+j*n]*known[j]).sum()).collect();
        let balanced:Vec<_>=rhs.iter().zip(&prepared.scales).map(|(b,s)|b/s).collect();
        let mut early=GmresProfile::default();let actual=prepared.solve_scaled_inner(&lap,&balanced,false,4,false,&mut early).unwrap();
        let mut legacy=GmresProfile::default();let unchanged=prepared.solve_scaled_inner(&lap,&balanced,false,0,false,&mut legacy).unwrap();
        let oracle=lap.solve(&dense,&rhs,n).unwrap();
        assert!(early.accepted_early&&early.early_checks>0);assert!(early.steps<=legacy.steps);
        assert!(actual.iter().zip(&oracle).all(|(a,b)|(a-b).abs()<1e-11));
        assert!(actual.iter().zip(unchanged).all(|(a,b)|(a-b).abs()<1e-11));
        let residual:Vec<_>=balanced.iter().zip(prepared.product(&actual,false)).map(|(b,a)|b-a).collect();
        let norm=|x:&[f64]|x.iter().map(|v|v*v).sum::<f64>().sqrt();
        let op=prepared.matrix.iter().map(|col|col.iter().map(|(_,v)|v.abs()).sum::<f64>()).fold(0.,f64::max);
        let backward=32.*f64::EPSILON*(op*actual.iter().map(|v|v.abs()).sum::<f64>()+balanced.iter().map(|v|v.abs()).sum::<f64>());
        assert!(norm(&residual)<(1e-11*norm(&balanced).max(1e-30)).max(backward));
    }
    #[test]
    fn adaptive_grouping_bounds_weakly_linked_blocks_without_changing_operator(){
        let n=18;let mut a:Sparse=(0..n).map(|i|vec![(i,10.)]).collect();
        for i in 2..n-1{let v=if i%4==1{0.3}else{4.};a[i].push((i+1,v));a[i+1].push((i,v));}
        let original=a.clone();assert!(bounded_groups(&a,&[0,1],0.1,4,false).is_err());
        let(groups,cutoff)=bounded_groups(&a,&[0,1],0.1,4,true).unwrap();
        assert!(cutoff>0.1);assert!(groups.iter().all(|g|g.len()<=4));assert!(groups.iter().any(|g|g==&vec![0,1]));
        let mut covered:Vec<_>=groups.into_iter().flatten().collect();covered.sort();assert_eq!(covered,(0..n).collect::<Vec<_>>());assert_eq!(a,original);
    }
    #[test]
    fn transpose_reuses_factors_and_solves_the_equilibrated_equation() {
        let lap=Lapack::load().unwrap();
        for n in [12,128] {
            let matrix:Sparse=(0..n).map(|j|(0..n).map(|i|(i,if i==j{2.+i as f64/n as f64}else{0.01*((i+1)*(j+2))as f64/(n*n)as f64})).collect()).collect();
            let prepared=Prepared::new(&lap,&matrix,&(0..n).collect::<Vec<_>>(),100.).unwrap();
            let expected:Vec<_>=(0..n).map(|i|(i as f64+0.3).sin()).collect();
            let rhs:Vec<_>=matrix.iter().map(|col|col.iter().map(|&(i,v)|v/prepared.scales[i]*expected[i]).sum()).collect();
            let actual=prepared.solve_scaled_transpose(&lap,&rhs).unwrap();
            assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<1e-11));
        }
    }
    #[test]
    fn preconditioner_keeps_all_cross_block_entries_in_the_solved_equation() {
        let lap = Lapack::load().unwrap();
        let n = 12;
        let mut dense = vec![0.; n * n];
        for i in 0..n {
            dense[i + i * n] = if i < 4 { 5. } else { 1e4 };
            if i + 1 < n {
                dense[i + (i + 1) * n] = 0.3;
                dense[i + 1 + i * n] = -0.7;
            }
        }
        dense[8 + 3 * n] = 2.;
        dense[3 + 8 * n] = 4.;
        let sparse: Sparse = (0..n)
            .map(|j| {
                (0..n)
                    .filter_map(|i| (dense[i + j * n] != 0.).then_some((i, dense[i + j * n])))
                    .collect()
            })
            .collect();
        let prepared = Prepared::new(&lap, &sparse, &[0, 1, 2, 3], 100.).unwrap();
        let rhs: Vec<_> = (0..n).map(|i| (i + 1) as f64).collect();
        let actual = prepared.solve(&lap, &rhs).unwrap();
        let expected = lap.solve(&dense, &rhs, n).unwrap();
        assert!(actual
            .iter()
            .zip(expected)
            .all(|(a, b)| (a - b).abs() < 1e-10));
    }
}
