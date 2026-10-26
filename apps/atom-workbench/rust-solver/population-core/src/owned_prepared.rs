//! Current-request Rust preparation -> the existing complete native E/F kernel.
//!
//! No C descriptors, JSON, density history, or Python coefficient copy is used
//! here. One immutable coefficient library is shared across all domain plans;
//! each mutable plan owns its own scratch. Input/output mode order is explicit,
//! and reverse dependencies retain every path that returns to the output.
use std::{collections::{BTreeMap,BTreeSet},sync::Arc,time::Instant};
use crate::{AwbStats,C,Coefficient,Execution,Lapack,Link,Map,Number,Plan,Workspace,execute,ns,product};
use crate::preparation::{Charge,FourierMatrices,GaugedSource,GaugePolicy,ModelView,PhaseBasis};
use crate::preparation::primitives::{self,PrimitivePreparation};

type Modes=BTreeSet<Charge>;
type Family=BTreeMap<Charge,usize>;
fn population_thread_policy(value:Option<&str>)->Result<usize,String>{
    let threads=value.map(str::parse::<usize>).transpose().map_err(|_|"ATOM_POPULATION_BLAS_THREADS must be an integer in1..32")?.unwrap_or(1);
    if !(1..=32).contains(&threads){return Err("ATOM_POPULATION_BLAS_THREADS must be in1..32".into());}Ok(threads)
}
#[path="physical_coordinates.rs"]
pub mod coordinates;
#[path="b0_factor.rs"]
pub mod b0_factor;
#[path="b0_inverse.rs"]
pub mod b0_inverse;
#[path="real_fourier.rs"]
pub mod real_fourier;
#[path="reduced_system.rs"]
pub mod reduced_system;
#[path="runtime_core.rs"]
pub mod runtime_core;
#[path="adaptive_direction.rs"]
pub mod adaptive_direction;
#[path="domain_transfer.rs"]
pub mod domain_transfer;
#[path="continuous_source_gram.rs"]
pub mod continuous_source_gram;
#[path="residual_proposal.rs"]
pub mod residual_proposal;
#[path="population_control.rs"]
pub mod population_control;
#[path="finite_driver.rs"]
pub mod finite_driver;

#[derive(Clone,Copy,Debug)]
pub enum KernelPolicy { Csr, HybridBlas }

/// Owns complete current Model coordinates and one compiled primitive library.
/// No default gauge policy is chosen by this API: the caller must name it.
pub struct OwnedPrepared {
    primitives:Arc<PrimitivePreparation>,
    coefficients:Arc<Vec<Coefficient>>,
    families:[Family;7], // S,F,G,K,R,LGG,LOG. S is shared by both commutators.
    lapack:Option<Arc<Lapack>>,
    preparation_lapack:Arc<Lapack>,
    prepared_reality:Arc<runtime_core::PreparedRealityBudget>,
    coefficient_bytes:usize,
}

#[derive(Clone,Debug)]
pub struct DomainDiagnostics {
    pub input_modes:usize,pub output_modes:usize,
    pub first_intermediate_modes:usize,pub j_intermediate_modes:usize,pub z_intermediate_modes:usize,
    /// Shared allocation size, not an additional allocation for this domain.
    pub shared_coefficient_bytes:usize,
    pub domain_index_bytes:usize,
    pub complete_reverse_dependencies:bool,
    pub arbitrary_complex_action:bool,
    pub roundoff_terms:usize,
    pub prepared_e_conjugacy_defect_bound:f64,pub prepared_f_conjugacy_defect_bound:f64,
    pub preparation_budget_is_population_error_bound:bool,
}

pub struct OwnedPencil {
    // Keeps the coordinate metadata alive even if its factory is dropped.
    primitives:Arc<PrimitivePreparation>,
    input_modes:Vec<Charge>,output_modes:Vec<Charge>,
    plan:Plan,complex:Workspace<C>,absolute:Workspace<f64>,stats:AwbStats,
    diagnostics:DomainDiagnostics,
}

#[cfg(test)]
#[path="preparation_model_fixture.rs"]
mod model_fixture;

fn add_charge(a:&[i32],b:&[i32],subtract:bool)->Result<Charge,String>{
    if a.len()!=b.len(){return Err("Native mode dimensions differ".into());}
    a.iter().zip(b).map(|(&a,&b)|if subtract{a.checked_sub(b)}else{a.checked_add(b)}
        .ok_or_else(||"Native intermediate mode overflow".into())).collect()
}
fn combinations(a:&Modes,b:&Modes,subtract:bool)->Result<Modes,String>{
    a.iter().flat_map(|q|b.iter().map(move|p|add_charge(q,p,subtract))).collect()
}
fn mode_set(values:&[Charge],axes:usize)->Result<Modes,String>{
    if values.is_empty()||values.iter().any(|q|q.len()!=axes){return Err("Input/output domain must be nonempty and use the complete declared phase basis".into());}
    let result:Modes=values.iter().cloned().collect();
    if result.len()!=values.len(){return Err("Repeated input/output mode".into());}Ok(result)
}
fn map(input:&[Charge],output:&[Charge])->Map{
    let index:BTreeMap<_,_>=output.iter().enumerate().map(|(i,q)|(q,i)).collect();
    let mut result=Map{source:vec![],target:vec![]};
    for(i,q)in input.iter().enumerate(){if let Some(&j)=index.get(q){result.source.push(i);result.target.push(j);}}result
}
fn links(family:&Family,input:&[Charge],output:&[Charge])->Result<Vec<Link>,String>{
    let index:BTreeMap<_,_>=output.iter().enumerate().map(|(i,q)|(q,i)).collect();let mut result=Vec::new();
    for(charge,&coefficient)in family{let mut link=Link{coefficient,source:vec![],target:vec![]};
        for(i,q)in input.iter().enumerate(){let target=add_charge(q,charge,false)?;
            if let Some(&j)=index.get(&target){link.source.push(i);link.target.push(j);}}
        if !link.source.is_empty(){result.push(link);}
    }Ok(result)
}
fn coefficient(matrix:&[C],rows:usize,cols:usize,policy:KernelPolicy)->Result<Coefficient,String>{
    if matrix.len()!=product(&[rows,cols])?||matrix.iter().any(|z|!z.re.is_finite()||!z.im.is_finite()){
        return Err("Invalid owned complete primitive matrix".into());}
    let mut row=vec![0];let mut col=Vec::new();let mut value=Vec::new();
    for i in 0..rows{for j in 0..cols{let z=matrix[i*cols+j];if z.re!=0.||z.im!=0.{col.push(j);value.push(z);}}row.push(value.len());}
    let absolute:Vec<_>=value.iter().map(|z|z.re.hypot(z.im)).collect();let mut dense=vec![];let mut dense_abs=vec![];
    // Exactly the existing C-ABI hybrid eligibility; no amplitude threshold.
    if matches!(policy,KernelPolicy::HybridBlas)&&rows.min(cols)>=8&&(value.len()as f64)>=0.15*(matrix.len()as f64){
        dense=vec![C::default();matrix.len()];dense_abs=vec![0.;matrix.len()];
        for i in 0..rows{for at in row[i]..row[i+1]{dense[i+col[at]*rows]=value[at];dense_abs[i+col[at]*rows]=absolute[at];}}
    }
    Ok(Coefficient{rows,cols,row,col,value,absolute,dense,dense_abs})
}
fn coefficient_bytes(values:&[Coefficient])->usize{
    values.iter().map(|a|(a.row.len()+a.col.len())*std::mem::size_of::<usize>()
        +(a.value.len()+a.dense.len())*std::mem::size_of::<C>()+(a.absolute.len()+a.dense_abs.len())*8).sum()
}

impl OwnedPrepared {
    /// The only public preparation constructor starts from original supplied
    /// Model matrices. Intermediate mutable primitive structs cannot silently
    /// inject another model or a stale incomplete radiation family.
    pub fn prepare(model:&ModelView<'_>,phases:PhaseBasis,gauge:GaugePolicy<'_>,policy:KernelPolicy)->Result<Self,String>{
        // Population kernels were validated with one BLAS thread and a
        // separate output-mode pool. Do not inherit the legacy period's8.
        let setting=std::env::var("ATOM_POPULATION_BLAS_THREADS").ok();
        let lap=Arc::new(Lapack::load_with_threads(population_thread_policy(setting.as_deref())?)?);
        let source=GaugedSource::prepare_with_lapack(model,phases,gauge,&lap)?;
        let primitives=primitives::prepare(&lap,source)?;
        Self::from_primitives(primitives,policy,lap)
    }
    fn from_primitives(primitives:PrimitivePreparation,policy:KernelPolicy,lap:Arc<Lapack>)->Result<Self,String>{
        let prepared_reality=Arc::new(runtime_core::PreparedRealityBudget::prepare(&primitives)?);
        let n=primitives.hilbert.source.n;let g=primitives.hilbert.source.ground.len();let e=primitives.hilbert.source.excited.len();
        let rank=primitives.gg_rows.len();let optical=primitives.optical_rows.len();let axes=primitives.hilbert.source.phases.axes.len();
        let shapes=[(n,n),(n,n),(n,n),(rank,e*e),(rank,e*e),(rank,rank),(optical,rank)];
        let inputs:[&FourierMatrices;7]=[&primitives.s,&primitives.no_jump_left,&primitives.no_jump_right,
            &primitives.selected_k,&primitives.selected_recycling,&primitives.gg_hamiltonian,&primitives.optical_hamiltonian];
        if n==0||g==0||e==0||g+e!=n{return Err("Invalid owned Hilbert partition".into());}
        let mut coefficients=Vec::new();let mut families:[Family;7]=std::array::from_fn(|_|Family::new());
        for(i,family)in inputs.into_iter().enumerate(){for(q,matrix)in family{
            if q.len()!=axes{return Err("Primitive coefficient has another phase basis".into());}
            // Exact zero matrices may remain in a prepared family. Keeping
            // their charge is harmless and agrees with the old plan library.
            let id=coefficients.len();coefficients.push(coefficient(matrix,shapes[i].0,shapes[i].1,policy)?);
            families[i].insert(q.clone(),id);
        }}
        let bytes=coefficient_bytes(&coefficients);
        Ok(Self{primitives:Arc::new(primitives),coefficients:Arc::new(coefficients),families,
            lapack:if matches!(policy,KernelPolicy::HybridBlas){Some(Arc::clone(&lap))}else{None},
            preparation_lapack:lap,prepared_reality,coefficient_bytes:bytes})
    }
    pub fn primitives(&self)->&PrimitivePreparation{&self.primitives}
    pub fn same_request(&self,other:&Self)->bool{Arc::ptr_eq(&self.primitives,&other.primitives)}
    pub fn shared_coefficient_bytes(&self)->usize{self.coefficient_bytes}
    pub fn configured_population_blas_threads(&self)->usize{self.preparation_lapack.configured_blas_threads()}
    pub fn physical_coordinates(&self)->coordinates::PhysicalCoordinates{
        coordinates::PhysicalCoordinates::new(Arc::clone(&self.primitives),Arc::clone(&self.preparation_lapack))
    }

    pub fn pencil(&self,input_modes:&[Charge],output_modes:&[Charge],max_rhs:usize)->Result<OwnedPencil,String>{
        let began=Instant::now();if max_rhs==0{return Err("max_rhs must be positive".into());}
        let p=&self.primitives;let source=&p.hilbert.source;let axes=source.phases.axes.len();
        let input=mode_set(input_modes,axes)?;let output=mode_set(output_modes,axes)?;
        let families:Vec<Modes>=self.families.iter().map(|f|f.keys().cloned().collect()).collect();
        let(s,f,g,k,r,lgg,log)=(&families[0],&families[1],&families[2],&families[3],&families[4],&families[5],&families[6]);
        let mut j_support=Modes::from([vec![0;axes]]);j_support.extend(s.iter().cloned());j_support.extend(combinations(s,s,false)?);
        let forward_j=combinations(&input,&j_support,false)?;
        let mut z_needed=output.clone();z_needed.extend(combinations(&output,lgg,true)?);z_needed.extend(combinations(&output,log,true)?);
        let z_modes:Vec<_>=combinations(&forward_j,k,false)?.intersection(&z_needed).cloned().collect();
        let n_support:Modes=f.union(g).cloned().collect();
        let mut j_needed=output.clone();j_needed.extend(combinations(&output,&n_support,true)?);
        j_needed.extend(combinations(&output,r,true)?);j_needed.extend(combinations(&z_modes.iter().cloned().collect(),k,true)?);
        let j_set:Modes=forward_j.intersection(&j_needed).cloned().collect();let j_modes:Vec<_>=j_set.iter().cloned().collect();
        let mut d_needed=j_set.clone();d_needed.extend(combinations(&j_set,s,true)?);
        let d_modes:Vec<_>=combinations(&input,s,false)?.intersection(&d_needed).cloned().collect();
        let stages=[links(&self.families[0],input_modes,&d_modes)?,links(&self.families[0],&d_modes,&j_modes)?,
            links(&self.families[1],&j_modes,output_modes)?,links(&self.families[2],&j_modes,output_modes)?,
            links(&self.families[3],&j_modes,&z_modes)?,links(&self.families[4],&j_modes,output_modes)?,
            links(&self.families[5],&z_modes,output_modes)?,links(&self.families[6],&z_modes,output_modes)?];
        let maps=[map(input_modes,&j_modes),map(&d_modes,&j_modes),map(&j_modes,output_modes),map(&z_modes,output_modes)];
        let n=source.n;let g=source.ground.len();let e=source.excited.len();let d=product(&[n,n])?;
        let frequency=output_modes.iter().map(|q|source.phases.frequency(q).map(|v|v.hi+v.lo)).collect::<Result<_,_>>()?;
        let plan=Plan{n,g,e,d,rank:p.gg_rows.len(),ni:input_modes.len(),no:output_modes.len(),nd:d_modes.len(),nj:j_modes.len(),nz:z_modes.len(),max_rhs,
            coefficients:Arc::clone(&self.coefficients),stages,maps,gg:p.gg_rows.iter().map(|&row|(row/g)*n+row%g).collect(),
            optical:p.optical_rows.clone(),frequency,lapack:self.lapack.clone(),execution:Execution::new()?};
        // Unlike the public C descriptor constructor, a disjoint rectangular
        // operator may have no intermediate modes; its exact action is zero.
        let bytes=plan.bytes();let mut roundoff_terms=0;
        // Same scalar operation count as the validated Python Links guard:
        // sum each coefficient's maximum row/column length per output mode.
        for stage in &plan.stages{let mut left=BTreeMap::<usize,usize>::new();let mut right=BTreeMap::<usize,usize>::new();
            for link in stage{let a=&plan.coefficients[link.coefficient];let row_max=a.row.windows(2).map(|r|r[1]-r[0]).max().unwrap_or(0);
                let mut columns=vec![0usize;a.cols];for &col in &a.col{columns[col]+=1;}let column_max=columns.into_iter().max().unwrap_or(0);
                for &target in &link.target{*left.entry(target).or_default()+=row_max;*right.entry(target).or_default()+=column_max;}}
            roundoff_terms=roundoff_terms.max(left.values().chain(right.values()).copied().max().unwrap_or(0));
        }
        roundoff_terms+=n+16;
        let maximum_frequency=plan.frequency.iter().map(|w|w.abs()).fold(0_f64,f64::max);let mut frequency_defect=0_f64;
        for(q,&w)in output_modes.iter().zip(&plan.frequency){let opposite=q.iter().map(|&k|k.checked_neg().ok_or("REAL transport partner overflow")).collect::<Result<Charge,_>>()?;
            let other=source.phases.frequency(&opposite)?;frequency_defect=frequency_defect.max((w+(other.hi+other.lo)).abs());}
        let e_defect=self.prepared_reality.e_defect;
        let f_defect=self.prepared_reality.f_base_defect+maximum_frequency*e_defect+frequency_defect*self.prepared_reality.e_norm;
        if !f_defect.is_finite(){return Err("Prepared finite-domain REAL-defect budget overflow".into());}
        let diagnostics=DomainDiagnostics{input_modes:plan.ni,output_modes:plan.no,
            first_intermediate_modes:plan.nd,j_intermediate_modes:plan.nj,z_intermediate_modes:plan.nz,
            shared_coefficient_bytes:self.coefficient_bytes,domain_index_bytes:bytes-self.coefficient_bytes,
            complete_reverse_dependencies:true,arbitrary_complex_action:true,roundoff_terms,
            prepared_e_conjugacy_defect_bound:e_defect,prepared_f_conjugacy_defect_bound:f_defect,preparation_budget_is_population_error_bound:false};
        Ok(OwnedPencil{primitives:Arc::clone(&self.primitives),input_modes:input_modes.to_vec(),output_modes:output_modes.to_vec(),plan,
            complex:Workspace::default(),absolute:Workspace::default(),stats:AwbStats{prepared_bytes:bytes as u64,preparation_ns:ns(began),..AwbStats::default()},diagnostics})
    }
}

impl OwnedPencil {
    pub fn input_modes(&self)->&[Charge]{&self.input_modes}
    pub fn output_modes(&self)->&[Charge]{&self.output_modes}
    pub fn input_dimension(&self)->usize{self.plan.ni*self.plan.d}
    pub fn output_dimension(&self)->usize{self.plan.no*self.plan.d}
    pub fn diagnostics(&self)->&DomainDiagnostics{&self.diagnostics}
    pub(super) fn real_preparation_defects(&self)->(f64,f64){(self.diagnostics.prepared_e_conjugacy_defect_bound,self.diagnostics.prepared_f_conjugacy_defect_bound)}
    pub fn primitives(&self)->&PrimitivePreparation{&self.primitives}
    pub fn workspace_bytes(&self)->usize{self.complex.bytes()+self.absolute.bytes()}
    fn validate(&self,input:&[C],rhs:usize,lengths:&[usize])->Result<(),String>{
        if rhs==0||rhs>self.plan.max_rhs{return Err("RHS count exceeds this owned plan; caller must chunk complete columns".into());}
        let ni=product(&[self.input_dimension(),rhs])?;let no=product(&[self.output_dimension(),rhs])?;
        if input.len()!=ni||lengths.iter().any(|&v|v!=no){return Err("Owned pencil buffers must be exact row-major [coordinate,rhs] shapes".into());}
        if input.iter().any(|z|!z.finite()){return Err("Nonfinite owned pencil input".into());}Ok(())
    }
    pub fn apply_into(&mut self,input:&[C],rhs:usize,mass:&mut[C],pencil:&mut[C])->Result<(),String>{
        self.validate(input,rhs,&[mass.len(),pencil.len()])?;let began=Instant::now();
        self.complex.prepare(&self.plan,rhs)?;self.complex.x.copy_from_slice(input);
        execute(&self.plan,&mut self.complex,rhs,&mut self.stats)?;
        mass.copy_from_slice(&self.complex.mass);pencil.copy_from_slice(&self.complex.pencil);
        self.stats.apply_calls+=1;self.stats.columns+=rhs as u64;self.stats.apply_ns+=ns(began);Ok(())
    }
    /// Same arithmetic envelope as ABI apply_abs, not an interval certificate.
    pub fn action_magnitudes_into(&mut self,input:&[C],rhs:usize,mass:&mut[f64],pencil:&mut[f64])->Result<(),String>{
        self.validate(input,rhs,&[mass.len(),pencil.len()])?;let began=Instant::now();
        self.absolute.prepare(&self.plan,rhs)?;for(x,z)in self.absolute.x.iter_mut().zip(input){*x=z.re.hypot(z.im);}
        execute(&self.plan,&mut self.absolute,rhs,&mut self.stats)?;
        mass.copy_from_slice(&self.absolute.mass);pencil.copy_from_slice(&self.absolute.pencil);
        self.stats.absolute_calls+=1;self.stats.absolute_ns+=ns(began);Ok(())
    }
    pub fn apply_shifted_into(&mut self,input:&[C],rhs:usize,shift:C,output:&mut[C])->Result<(),String>{
        self.validate(input,rhs,&[output.len()])?;if !shift.finite(){return Err("Nonfinite owned pencil shift".into());}
        let began=Instant::now();self.complex.prepare(&self.plan,rhs)?;self.complex.x.copy_from_slice(input);
        execute(&self.plan,&mut self.complex,rhs,&mut self.stats)?;
        for(f,e)in self.complex.pencil.iter_mut().zip(&self.complex.mass){
            *f=C{re:shift.re*e.re-shift.im*e.im-f.re,im:shift.re*e.im+shift.im*e.re-f.im};
            if !f.finite(){return Err("Nonfinite owned complete shifted action".into());}
        }
        output.copy_from_slice(&self.complex.pencil);self.stats.apply_calls+=1;self.stats.columns+=rhs as u64;self.stats.apply_ns+=ns(began);Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn preparation_thread_policy_population_is_explicit_and_legacy_restore_is_stable(){
        // OpenBLAS settings are process-global. Run the mutating portion in
        // an isolated test process so this remains safe in a parallel suite.
        if std::env::var("ATOM_THREAD_POLICY_TEST_CHILD").as_deref()!=Ok("1"){
            let mut command=std::process::Command::new(std::env::current_exe().unwrap());
            command.args(["--exact","owned_prepared::tests::preparation_thread_policy_population_is_explicit_and_legacy_restore_is_stable","--nocapture"])
                .env("ATOM_THREAD_POLICY_TEST_CHILD","1").env("ATOM_BLAS_THREADS","8").env_remove("ATOM_POPULATION_BLAS_THREADS");
            #[cfg(windows)]{use std::os::windows::process::CommandExt;command.creation_flags(0x08000000);}
            assert!(command.status().unwrap().success());return;
        }
        assert_eq!(population_thread_policy(None).unwrap(),1);
        assert_eq!(population_thread_policy(Some("4")).unwrap(),4);
        for bad in["0","33","no"]{assert!(population_thread_policy(Some(bad)).is_err());}
        let parent=Lapack::load().unwrap();assert_eq!(parent.configured_blas_threads(),8);
        let core=fixture();assert_eq!(core.configured_population_blas_threads(),1);
        // Explicit process switching does not overwrite either owner's
        // configured preference; parent can restore its original8 afterwards.
        core.preparation_lapack.set_process_blas_threads(2).unwrap();
        assert_eq!(core.configured_population_blas_threads(),1);assert_eq!(parent.configured_blas_threads(),8);
        parent.restore_configured_threads().unwrap();
        assert!(parent.set_process_blas_threads(0).is_err());assert!(Lapack::load_with_threads(33).is_err());
        std::env::set_var("ATOM_POPULATION_BLAS_THREADS","4");let overridden=fixture();assert_eq!(overridden.configured_population_blas_threads(),4);
        parent.restore_configured_threads().unwrap();
        assert_eq!(parent.multiply(&[2.],&[3.],1,1,1),vec![6.]);
    }
    use crate::preparation::MatrixView;
    fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
    fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
    fn scale(a:C,s:f64)->C{C{re:a.re*s,im:a.im*s}}
    fn mm(a:&[C],b:&[C],n:usize)->Vec<C>{let mut x=vec![C::default();n*n];for i in 0..n{for j in 0..n{for k in 0..n{add(&mut x[i*n+j],mul(a[i*n+k],b[k*n+j]));}}}x}
    fn add_field(out:&mut FourierMatrices,q:Charge,a:&[C],s:f64){let value=out.entry(q).or_insert_with(||vec![C::default();a.len()]);for(x,&y)in value.iter_mut().zip(a){add(x,scale(y,s));}}
    fn comm(family:&FourierMatrices,x:&FourierMatrices,n:usize)->FourierMatrices{
        let mut out=FourierMatrices::new();for(q,a)in family{for(p,b)in x{let key=add_charge(q,p,false).unwrap();
            add_field(&mut out,key.clone(),&mm(a,b,n),1.);add_field(&mut out,key,&mm(b,a,n),-1.);}}out
    }
    pub(super) fn fixture()->OwnedPrepared{
        let n=4;let mut h=vec![0.;16];let zero=vec![0.;16];for(i,v)in[0.,0.6,3.,4.].iter().enumerate(){h[i*n+i]=*v;}
        for(i,j,v)in[(0,2,0.1),(1,3,0.2),(0,1,0.02)]{h[i*n+j]=v;h[j*n+i]=v;}
        let mut c=vec![0.;16];c[2]=0.7;c[3]=0.1;c[6]=0.2;c[7]=0.8;
        let channels=[MatrixView{real:&c,imag:&zero}];let model=ModelView{dimension:n,hamiltonian:MatrixView{real:&h,imag:&zero},
            collapse:&channels,drives:&[],ground:&[true,true,false,false],initial_index:0};
        let phases=PhaseBasis::from_declared(&[7.],None,&[]).unwrap();let gauge=[vec![0],vec![1],vec![0],vec![0]];
        OwnedPrepared::prepare(&model,phases,GaugePolicy::Explicit(&gauge),KernelPolicy::Csr).unwrap()
    }
    /// Full convolution independent of the staged return graph and compact
    /// K/R/LGG/LOG: form T x, then use original transformed H/C directly.
    fn original_oracle(p:&PrimitivePreparation,x:&FourierMatrices)->(FourierMatrices,FourierMatrices){
        let n=p.hilbert.source.n;let g=p.hilbert.source.ground.len();let e=n-g;
        let dx=comm(&p.s,x,n);let ddx=comm(&p.s,&dx,n);let mut j=x.clone();
        for(q,a)in dx{add_field(&mut j,q,&a,1.);}for(q,a)in ddx{add_field(&mut j,q,&a,0.5);}
        let mut mass=j.clone();for(q,k)in &p.k_hat{for(h,a)in &j{let mut z=vec![C::default();n*n];
            for ab in 0..g*g{for ij in 0..e*e{add(&mut z[(ab/g)*n+ab%g],mul(k[ab*e*e+ij],a[(g+ij/e)*n+g+ij%e]));}}
            add_field(&mut mass,add_charge(q,h,false).unwrap(),&z,1.);
        }}
        let mut out=comm(&p.hilbert.source.hamiltonian,&mass,n);for a in out.values_mut(){for z in a{*z=C{re:z.im,im:-z.re};}}
        // All original same-channel charge pairs, not compiled R/Gamma.
        for channel in &p.hilbert.source.collapse{for(a,ca)in channel{for(b,cb)in channel{
            let cbh:Vec<_>=(0..n*n).map(|i|C{re:cb[(i%n)*n+i/n].re,im:-cb[(i%n)*n+i/n].im}).collect();
            let gamma=mm(&cbh,ca,n);let charge=add_charge(a,b,true).unwrap();
            for(h,rho)in &mass{let q=add_charge(&charge,h,false).unwrap();
                add_field(&mut out,q.clone(),&mm(&mm(ca,rho,n),&cbh,n),1.);
                add_field(&mut out,q.clone(),&mm(&gamma,rho,n),-0.5);add_field(&mut out,q,&mm(rho,&gamma,n),-0.5);
            }
        }}}
        for(q,a)in &mass{let parts=p.hilbert.source.phases.frequency(q).unwrap();let w=parts.hi+parts.lo;
            let derivative:Vec<_>=a.iter().map(|z|C{re:w*z.im,im:-w*z.re}).collect();add_field(&mut out,q.clone(),&derivative,1.);
        }(mass,out)
    }
    #[test]
    fn preparation_owned_complete_rectangular_action_and_shared_lifetime(){
        let owned=fixture();let input_modes=[vec![2],vec![-1],vec![0]];let n=4;let d=n*n;let rhs=2;
        let x:Vec<_>=(0..input_modes.len()*d*rhs).map(|i|C{re:((i*17+3)%41)as f64/47.,im:((i*11+7)%29)as f64/-31.}).collect();
        let mut oracles=Vec::new();let mut targets=Modes::new();
        for b in 0..rhs{let input:FourierMatrices=input_modes.iter().enumerate().map(|(i,q)|(q.clone(),(0..d).map(|k|x[(i*d+k)*rhs+b]).collect())).collect();
            let oracle=original_oracle(owned.primitives(),&input);targets.extend(oracle.0.keys().chain(oracle.1.keys()).cloned());oracles.push(oracle);}
        let mut outputs:Vec<_>=targets.into_iter().collect();outputs.reverse();let mut full=owned.pencil(&input_modes,&outputs,rhs).unwrap();
        let rectangle=[vec![0],vec![2]];let mut part=owned.pencil(&input_modes,&rectangle,rhs).unwrap();
        assert!(Arc::ptr_eq(&full.plan.coefficients,&part.plan.coefficients));assert!(full.diagnostics.shared_coefficient_bytes>0);
        assert!(full.diagnostics.j_intermediate_modes>input_modes.len());
        let count=outputs.len()*d*rhs;let(mut ef,mut ff)=(vec![C::default();count],vec![C::default();count]);full.apply_into(&x,rhs,&mut ef,&mut ff).unwrap();
        for(m,q)in outputs.iter().enumerate(){for k in 0..d{for b in 0..rhs{let at=(m*d+k)*rhs+b;
            for(actual,family)in[(ef[at],&oracles[b].0),(ff[at],&oracles[b].1)]{
                let expected=family.get(q).map_or(C::default(),|a|a[k]);assert!((actual.re-expected.re).hypot(actual.im-expected.im)<2e-11*(1.+expected.re.hypot(expected.im)));
            }
        }}}
        let(mut ea,mut fa)=(vec![0.;count],vec![0.;count]);full.action_magnitudes_into(&x,rhs,&mut ea,&mut fa).unwrap();
        for i in 0..count{assert!(ef[i].re.hypot(ef[i].im)<=ea[i]*(1.+1e-14));assert!(ff[i].re.hypot(ff[i].im)<=fa[i]*(1.+1e-14));}
        // Safe plan retains both coefficient and coordinate owners after factory drop.
        drop(owned);drop(full);let count=rectangle.len()*d*rhs;let(mut ep,mut fp)=(vec![C::default();count],vec![C::default();count]);
        part.apply_into(&x,rhs,&mut ep,&mut fp).unwrap();
        for(m,q)in rectangle.iter().enumerate(){let old=outputs.iter().position(|p|p==q).unwrap();for k in 0..d*rhs{
            assert_eq!(ep[m*d*rhs+k].re.to_bits(),ef[old*d*rhs+k].re.to_bits());assert_eq!(fp[m*d*rhs+k].im.to_bits(),ff[old*d*rhs+k].im.to_bits());}}
        let shift=C{re:8.,im:-2.};let mut shifted=vec![C::default();count];part.apply_shifted_into(&x,rhs,shift,&mut shifted).unwrap();
        for i in 0..count{let expected=mul(shift,ep[i]);assert_eq!(shifted[i].re,expected.re-fp[i].re);assert_eq!(shifted[i].im,expected.im-fp[i].im);}
        let previous=fp.clone();part.apply_into(&vec![C::default();x.len()],rhs,&mut ep,&mut fp).unwrap();assert!(fp.iter().chain(&ep).all(|z|z.re==0.&&z.im==0.));assert!(previous.iter().any(|z|z.re!=0.));
    }
    #[test]
    fn preparation_owned_disjoint_domains_and_shape_failures(){
        let owned=fixture();let mut empty=owned.pencil(&[vec![0]],&[vec![100]],1).unwrap();
        assert_eq!(empty.diagnostics.j_intermediate_modes,0);let x=vec![C{re:1.,im:0.3};16];let(mut e,mut f)=(vec![C{re:4.,im:5.};16],vec![C{re:8.,im:9.};16]);
        empty.apply_into(&x,1,&mut e,&mut f).unwrap();assert!(e.iter().chain(&f).all(|z|z.re==0.&&z.im==0.));
        assert!(owned.pencil(&[vec![0],vec![0]],&[vec![0]],1).is_err());assert!(owned.pencil(&[vec![0,0]],&[vec![0]],1).is_err());
        assert!(empty.apply_into(&x,2,&mut e,&mut f).is_err());let mut bad=x.clone();bad[2].re=f64::NAN;
        assert!(empty.apply_into(&bad,1,&mut e,&mut f).is_err());assert!(empty.apply_into(&x[..15],1,&mut e,&mut f).is_err());
    }
}
