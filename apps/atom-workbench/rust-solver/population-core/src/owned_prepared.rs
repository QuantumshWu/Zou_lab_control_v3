//! Current-request Rust preparation -> the complete native E/F kernel.
//!
//! No C descriptors, JSON, density history, or Python coefficient copy is used
//! here. One immutable coefficient library is shared across all domain plans;
//! each mutable plan owns its own scratch. Input/output mode order is explicit,
//! and reverse dependencies retain every path that returns to the output.
use std::{collections::{BTreeMap,BTreeSet},sync::Arc,time::Instant};
use crate::{C,Coefficient,Finite,Lapack,StageProfile,product};
use crate::pencil::{self,Family,HilbertCoefficient,Library,Plan,PlanInput,StageClock,ThinCoefficient,Workspace};
use crate::preparation::{Charge,FourierMatrices,GaugedSource,GaugePolicy,ModelView,PhaseBasis};
use crate::preparation::primitives::{self,PrimitivePreparation};

type Modes=BTreeSet<Charge>;
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
#[path="runtime_core.rs"]
pub mod runtime_core;
#[path="trial.rs"]
pub(crate) mod trial;
#[path="readout.rs"]
pub(crate) mod readout;
#[path="reduced.rs"]
pub(crate) mod reduced;
#[path="directions.rs"]
pub mod directions;
#[path="engine.rs"]
pub mod engine;
#[path="slow_dual.rs"]
pub mod slow_dual;
#[path="error_estimate.rs"]
pub mod error_estimate;
#[path="asymptotic.rs"]
pub mod asymptotic;

/// Arithmetic backend of the pencil kernels. `HybridBlas` multiplies the
/// coefficient blocks with OpenBLAS ZGEMM; `Csr` runs the same blocked
/// algorithm with a plain in-crate multiply (small validation fixtures).
#[derive(Clone,Copy,Debug)]
pub enum KernelPolicy { Csr, HybridBlas }

/// Owns complete current Model coordinates and one compiled primitive library.
/// No default gauge policy is chosen by this API: the caller must name it.
pub struct OwnedPrepared {
    primitives:Arc<PrimitivePreparation>,
    library:Arc<Library>,
    /// S, NL, NR, K, R, L_GG, L_OG. S is shared by both commutators.
    families:[Family;7],
    lapack:Option<Arc<Lapack>>,
    preparation_lapack:Arc<Lapack>,
    prepared_reality:Arc<runtime_core::PreparedRealityBudget>,
    coefficient_bytes:usize,
    /// Milliseconds: gauge/source, S/K primitives, operator library.
    pub timing:[f64;3],
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
    pub prepared_e_conjugacy_defect_bound:f64,pub prepared_f_conjugacy_defect_bound:f64,
    pub preparation_budget_is_population_error_bound:bool,
    /// Exactly Hermitian-symmetric inputs are evaluated on the canonical half
    /// of every mode set and mirrored; this agrees with a full evaluation to
    /// roundoff plus the prepared conjugacy defect bounds above.
    pub hermitian_half_capable:bool,
}

pub struct OwnedPencil {
    // Keeps the coordinate metadata alive even if its factory is dropped.
    primitives:Arc<PrimitivePreparation>,
    input_modes:Vec<Charge>,output_modes:Vec<Charge>,
    plan:Plan,
    workspace:Workspace,
    profile:StageProfile,
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
/// Sparse CSR copy of a row-major matrix. Storage does not depend on the
/// kernel policy; the argument is kept for existing callers.
fn coefficient(matrix:&[C],rows:usize,cols:usize,_policy:KernelPolicy)->Result<Coefficient,String>{
    if matrix.len()!=product(&[rows,cols])?||matrix.iter().any(|z|!z.finite()){
        return Err("Invalid owned complete primitive matrix".into());}
    let mut row=vec![0];let mut col=Vec::new();let mut value=Vec::new();
    for i in 0..rows{for j in 0..cols{let z=matrix[i*cols+j];if z.re!=0.||z.im!=0.{col.push(j);value.push(z);}}row.push(value.len());}
    Ok(Coefficient{rows,cols,row,col,value})
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
        let clock=Instant::now();
        let source=GaugedSource::prepare_with_lapack(model,phases,gauge,&lap)?;
        let gauge_ms=clock.elapsed().as_secs_f64()*1000.;let clock=Instant::now();
        let primitives=primitives::prepare(&lap,source)?;
        let primitives_ms=clock.elapsed().as_secs_f64()*1000.;let clock=Instant::now();
        let mut prepared=Self::from_primitives(primitives,policy,lap)?;
        prepared.timing=[gauge_ms,primitives_ms,clock.elapsed().as_secs_f64()*1000.];
        Ok(prepared)
    }
    fn from_primitives(primitives:PrimitivePreparation,policy:KernelPolicy,lap:Arc<Lapack>)->Result<Self,String>{
        let prepared_reality=Arc::new(runtime_core::PreparedRealityBudget::prepare(&primitives)?);
        let n=primitives.hilbert.source.n;let g=primitives.hilbert.source.ground.len();let e=primitives.hilbert.source.excited.len();
        let rank=primitives.gg_rows.len();let optical=primitives.optical_rows.len();let axes=primitives.hilbert.source.phases.axes.len();
        if n==0||g==0||e==0||g+e!=n{return Err("Invalid owned Hilbert partition".into());}
        // K and R act on vec(J_EE), which the kernel stores column-major.
        let excited_order:Vec<usize>=(0..e*e).map(|c|c/e+e*(c%e)).collect();
        let hilbert_families=[&primitives.s,&primitives.no_jump_left,&primitives.no_jump_right];
        let thin_families:[(&FourierMatrices,usize,usize,Option<&[usize]>);4]=[
            (&primitives.selected_k,rank,e*e,Some(&excited_order)),(&primitives.selected_recycling,rank,e*e,Some(&excited_order)),
            (&primitives.gg_hamiltonian,rank,rank,None),(&primitives.optical_hamiltonian,optical,rank,None)];
        let mut library=Library::default();let mut families:[Family;7]=std::array::from_fn(|_|Family::new());
        let valid=|q:&Charge,matrix:&[C],length:usize|q.len()==axes&&matrix.len()==length&&matrix.iter().all(|z|z.finite());
        // Exact zero matrices may remain in a prepared family. Keeping their
        // charge is harmless; their block pattern is empty.
        for(i,family)in hilbert_families.into_iter().enumerate(){for(q,matrix)in family{
            if !valid(q,matrix,n*n){return Err("Invalid owned complete primitive matrix".into());}
            families[i].insert(q.clone(),library.hilbert.len());library.hilbert.push(HilbertCoefficient::new(matrix,n,g));
        }}
        for(i,(family,rows,cols,order))in thin_families.into_iter().enumerate(){for(q,matrix)in family{
            if !valid(q,matrix,rows*cols){return Err("Invalid owned complete primitive matrix".into());}
            families[3+i].insert(q.clone(),library.thin.len());library.thin.push(ThinCoefficient::new(matrix,rows,cols,order));
        }}
        let bytes=library.bytes();
        Ok(Self{primitives:Arc::new(primitives),library:Arc::new(library),families,
            lapack:if matches!(policy,KernelPolicy::HybridBlas){Some(Arc::clone(&lap))}else{None},
            preparation_lapack:lap,prepared_reality,coefficient_bytes:bytes,timing:[0.;3]})
    }
    pub fn primitives(&self)->&PrimitivePreparation{&self.primitives}
    pub fn same_request(&self,other:&Self)->bool{Arc::ptr_eq(&self.primitives,&other.primitives)}
    pub fn shared_coefficient_bytes(&self)->usize{self.coefficient_bytes}
    pub fn configured_population_blas_threads(&self)->usize{self.preparation_lapack.configured_blas_threads()}
    pub fn physical_coordinates(&self)->coordinates::PhysicalCoordinates{
        coordinates::PhysicalCoordinates::new(Arc::clone(&self.primitives),Arc::clone(&self.preparation_lapack))
    }

    pub fn pencil(&self,input_modes:&[Charge],output_modes:&[Charge],max_rhs:usize)->Result<OwnedPencil,String>{
        if max_rhs==0{return Err("max_rhs must be positive".into());}
        let p=&self.primitives;let source=&p.hilbert.source;let axes=source.phases.axes.len();
        let input=mode_set(input_modes,axes)?;let output=mode_set(output_modes,axes)?;
        // Forward reachability from the input, intersected with every path
        // that returns to the output (reverse dependencies).
        let keys:Vec<Modes>=self.families.iter().map(|f|f.keys().cloned().collect()).collect();
        let(s,left,right,k,r,lgg,log)=(&keys[0],&keys[1],&keys[2],&keys[3],&keys[4],&keys[5],&keys[6]);
        let mut j_support=Modes::from([vec![0;axes]]);j_support.extend(s.iter().cloned());j_support.extend(combinations(s,s,false)?);
        let forward_j=combinations(&input,&j_support,false)?;
        let mut z_needed=output.clone();z_needed.extend(combinations(&output,lgg,true)?);z_needed.extend(combinations(&output,log,true)?);
        let z_modes:Vec<_>=combinations(&forward_j,k,false)?.intersection(&z_needed).cloned().collect();
        let no_jump:Modes=left.union(right).cloned().collect();
        let mut j_needed=output.clone();j_needed.extend(combinations(&output,&no_jump,true)?);
        j_needed.extend(combinations(&output,r,true)?);j_needed.extend(combinations(&z_modes.iter().cloned().collect(),k,true)?);
        let j_set:Modes=forward_j.intersection(&j_needed).cloned().collect();let j_modes:Vec<_>=j_set.iter().cloned().collect();
        let mut d_needed=j_set.clone();d_needed.extend(combinations(&j_set,s,true)?);
        let d_modes:Vec<_>=combinations(&input,s,false)?.intersection(&d_needed).cloned().collect();
        let n=source.n;let g=source.ground.len();
        let frequency:Vec<f64>=output_modes.iter().map(|q|source.phases.frequency(q).map(|v|v.hi+v.lo)).collect::<Result<_,_>>()?;
        let plan=Plan::new(PlanInput{n,ground:g,library:Arc::clone(&self.library),
            commutator:&self.families[0],left:&self.families[1],right:&self.families[2],selected_k:&self.families[3],
            recycling:&self.families[4],ground_hamiltonian:&self.families[5],optical_hamiltonian:&self.families[6],
            input:input_modes,first:&d_modes,j:&j_modes,z:&z_modes,output:output_modes,
            gg:p.gg_rows.iter().map(|&row|(row/g,row%g)).collect(),optical:p.optical_rows.iter().map(|&at|(at/n,at%n)).collect(),
            frequency:frequency.clone(),max_rhs,blas:self.lapack.clone()})?;
        // A disjoint rectangular operator may have no intermediate modes; its
        // exact action is zero.
        let maximum_frequency=frequency.iter().map(|w|w.abs()).fold(0_f64,f64::max);let mut frequency_defect=0_f64;
        for(q,&w)in output_modes.iter().zip(&frequency){let opposite=q.iter().map(|&k|k.checked_neg().ok_or("REAL transport partner overflow")).collect::<Result<Charge,_>>()?;
            let other=source.phases.frequency(&opposite)?;frequency_defect=frequency_defect.max((w+(other.hi+other.lo)).abs());}
        let e_defect=self.prepared_reality.e_defect;
        let f_defect=self.prepared_reality.f_base_defect+maximum_frequency*e_defect+frequency_defect*self.prepared_reality.e_norm;
        if !f_defect.is_finite(){return Err("Prepared finite-domain REAL-defect budget overflow".into());}
        let(first,j,z)=plan.intermediate_modes();
        let diagnostics=DomainDiagnostics{input_modes:plan.input_modes(),output_modes:plan.output_modes(),
            first_intermediate_modes:first,j_intermediate_modes:j,z_intermediate_modes:z,
            shared_coefficient_bytes:self.coefficient_bytes,domain_index_bytes:plan.index_bytes(),
            complete_reverse_dependencies:true,arbitrary_complex_action:true,
            prepared_e_conjugacy_defect_bound:e_defect,prepared_f_conjugacy_defect_bound:f_defect,preparation_budget_is_population_error_bound:false,
            hermitian_half_capable:plan.hermitian_half_capable()};
        Ok(OwnedPencil{primitives:Arc::clone(&self.primitives),input_modes:input_modes.to_vec(),output_modes:output_modes.to_vec(),plan,
            workspace:Workspace::default(),profile:StageProfile::default(),diagnostics})
    }
}

impl OwnedPencil {
    pub fn input_modes(&self)->&[Charge]{&self.input_modes}
    pub fn output_modes(&self)->&[Charge]{&self.output_modes}
    pub fn input_dimension(&self)->usize{self.input_modes.len()*self.dimension()}
    pub fn output_dimension(&self)->usize{self.output_modes.len()*self.dimension()}
    fn dimension(&self)->usize{let n=self.primitives.hilbert.source.n;n*n}
    pub fn diagnostics(&self)->&DomainDiagnostics{&self.diagnostics}
    pub(super) fn real_preparation_defects(&self)->(f64,f64){(self.diagnostics.prepared_e_conjugacy_defect_bound,self.diagnostics.prepared_f_conjugacy_defect_bound)}
    pub fn primitives(&self)->&PrimitivePreparation{&self.primitives}
    pub fn workspace_bytes(&self)->usize{self.workspace.bytes()}
    /// Upper bound of the scratch bytes one right-hand side needs.
    pub fn workspace_bytes_per_rhs(&self)->usize{self.plan.workspace_bytes_per_rhs()}
    /// Per-stage wall time of apply_into/apply_shifted_into (diagnostic).
    pub fn stage_profile(&self)->&StageProfile{&self.profile}
    pub fn reset_stage_profile(&mut self){self.profile=StageProfile::default();}
    fn validate(&self,input:&[C],rhs:usize,lengths:&[usize])->Result<(),String>{
        if rhs==0||rhs>self.plan.max_rhs(){return Err("RHS count exceeds this owned plan; caller must chunk complete columns".into());}
        let ni=product(&[self.input_dimension(),rhs])?;let no=product(&[self.output_dimension(),rhs])?;
        if input.len()!=ni||lengths.iter().any(|&v|v!=no){return Err("Owned pencil buffers must be exact row-major [coordinate,rhs] shapes".into());}
        Ok(())
    }
    /// E x and F x of a signed field with layout [mode][(i*n+j)*rhs+b].
    pub fn apply_into(&mut self,input:&[C],rhs:usize,mass:&mut[C],pencil:&mut[C])->Result<(),String>{
        self.validate(input,rhs,&[mass.len(),pencil.len()])?;
        let half=self.plan.apply(&mut self.workspace,input,rhs,&mut self.profile)?;
        let mut clock=StageClock::start();
        self.plan.store(&self.workspace,half,rhs,mass,pencil)?;
        clock.mark(&mut self.profile,pencil::STAGE_COUNT-1);self.profile.applications+=1;Ok(())
    }
    /// shift * E x - F x.
    pub fn apply_shifted_into(&mut self,input:&[C],rhs:usize,shift:C,output:&mut[C])->Result<(),String>{
        self.validate(input,rhs,&[output.len()])?;if !shift.finite(){return Err("Nonfinite owned pencil shift".into());}
        let half=self.plan.apply(&mut self.workspace,input,rhs,&mut self.profile)?;
        let mut clock=StageClock::start();
        self.plan.store_shifted(&self.workspace,shift,half,rhs,output)?;
        clock.mark(&mut self.profile,pencil::STAGE_COUNT-1);self.profile.applications+=1;Ok(())
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
    pub(super) fn fixture()->OwnedPrepared{fixture_with(KernelPolicy::Csr)}
    fn fixture_with(policy:KernelPolicy)->OwnedPrepared{
        let n=4;let mut h=vec![0.;16];let zero=vec![0.;16];for(i,v)in[0.,0.6,3.,4.].iter().enumerate(){h[i*n+i]=*v;}
        for(i,j,v)in[(0,2,0.1),(1,3,0.2),(0,1,0.02)]{h[i*n+j]=v;h[j*n+i]=v;}
        let mut c=vec![0.;16];c[2]=0.7;c[3]=0.1;c[6]=0.2;c[7]=0.8;
        let channels=[MatrixView{real:&c,imag:&zero}];let model=ModelView{dimension:n,hamiltonian:MatrixView{real:&h,imag:&zero},
            collapse:&channels,drives:&[],ground:&[true,true,false,false],initial_index:0};
        let phases=PhaseBasis::from_declared(&[7.],None,&[]).unwrap();let gauge=[vec![0],vec![1],vec![0],vec![0]];
        OwnedPrepared::prepare(&model,phases,GaugePolicy::Explicit(&gauge),policy).unwrap()
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
        assert!(Arc::ptr_eq(full.plan.library(),part.plan.library()));assert!(full.diagnostics.shared_coefficient_bytes>0);
        assert!(full.diagnostics.j_intermediate_modes>input_modes.len());
        let count=outputs.len()*d*rhs;let(mut ef,mut ff)=(vec![C::default();count],vec![C::default();count]);full.apply_into(&x,rhs,&mut ef,&mut ff).unwrap();
        for(m,q)in outputs.iter().enumerate(){for k in 0..d{for b in 0..rhs{let at=(m*d+k)*rhs+b;
            for(actual,family)in[(ef[at],&oracles[b].0),(ff[at],&oracles[b].1)]{
                let expected=family.get(q).map_or(C::default(),|a|a[k]);assert!((actual.re-expected.re).hypot(actual.im-expected.im)<2e-11*(1.+expected.re.hypot(expected.im)));
            }
        }}}
        // Safe plan retains both coefficient and coordinate owners after factory drop.
        drop(owned);drop(full);let count=rectangle.len()*d*rhs;let(mut ep,mut fp)=(vec![C::default();count],vec![C::default();count]);
        part.apply_into(&x,rhs,&mut ep,&mut fp).unwrap();
        for(m,q)in rectangle.iter().enumerate(){let old=outputs.iter().position(|p|p==q).unwrap();for k in 0..d*rhs{
            assert_eq!(ep[m*d*rhs+k].re.to_bits(),ef[old*d*rhs+k].re.to_bits());assert_eq!(fp[m*d*rhs+k].im.to_bits(),ff[old*d*rhs+k].im.to_bits());}}
        let shift=C{re:8.,im:-2.};let mut shifted=vec![C::default();count];part.apply_shifted_into(&x,rhs,shift,&mut shifted).unwrap();
        for i in 0..count{let expected=mul(shift,ep[i]);assert_eq!(shifted[i].re,expected.re-fp[i].re);assert_eq!(shifted[i].im,expected.im-fp[i].im);}
        let previous=fp.clone();part.apply_into(&vec![C::default();x.len()],rhs,&mut ep,&mut fp).unwrap();assert!(fp.iter().chain(&ep).all(|z|z.re==0.&&z.im==0.));assert!(previous.iter().any(|z|z.re!=0.));
    }
    /// Hermitian-symmetric input on a symmetric domain uses the canonical
    /// half and mirrors it; a perturbed input takes the full path. Both
    /// kernel backends must match the oracle in both cases.
    #[test]
    fn preparation_owned_hermitian_half_and_blas_match_oracle(){
        let modes=[vec![-2],vec![-1],vec![0],vec![1],vec![2]];let n=4;let d=n*n;let rhs=3;
        for policy in[KernelPolicy::Csr,KernelPolicy::HybridBlas]{
            let owned=fixture_with(policy);let map=owned.real_fourier_map(&modes).unwrap();
            let real:Vec<_>=(0..map.dimension()*rhs).map(|i|((i*29+11)%53)as f64/57.-0.4).collect();
            let mut symmetric=vec![C::default();real.len()];map.push_into(&real,rhs,&mut symmetric).unwrap();
            let mut skew=symmetric.clone();skew[5].im+=0.25;
            let mut plan=owned.pencil(&modes,&modes,rhs).unwrap();assert!(plan.diagnostics.hermitian_half_capable);
            for x in[&symmetric,&skew]{
                let(mut e,mut f)=(vec![C::default();x.len()],vec![C::default();x.len()]);plan.apply_into(x,rhs,&mut e,&mut f).unwrap();
                for b in 0..rhs{
                    let input:FourierMatrices=modes.iter().enumerate().map(|(i,q)|(q.clone(),(0..d).map(|k|x[(i*d+k)*rhs+b]).collect())).collect();
                    let(mass,pencil)=original_oracle(owned.primitives(),&input);
                    for(m,q)in modes.iter().enumerate(){for k in 0..d{let at=(m*d+k)*rhs+b;
                        for(actual,family)in[(e[at],&mass),(f[at],&pencil)]{
                            let expected=family.get(q).map_or(C::default(),|a|a[k]);
                            assert!((actual.re-expected.re).hypot(actual.im-expected.im)<2e-11*(1.+expected.re.hypot(expected.im)));
                        }
                    }}
                }
                let shift=C{re:3.,im:1.5};let mut shifted=vec![C::default();x.len()];plan.apply_shifted_into(x,rhs,shift,&mut shifted).unwrap();
                for i in 0..x.len(){let expected=mul(shift,e[i]);
                    assert!((shifted[i].re-(expected.re-f[i].re)).abs()<=1e-14*(1.+expected.re.abs()+f[i].re.abs()));
                    assert!((shifted[i].im-(expected.im-f[i].im)).abs()<=1e-14*(1.+expected.im.abs()+f[i].im.abs()));}
            }
        }
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
