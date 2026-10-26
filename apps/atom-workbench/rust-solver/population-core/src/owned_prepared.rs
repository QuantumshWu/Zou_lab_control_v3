//! Current-request Rust preparation -> the complete native E/F kernel.
//!
//! No C descriptors, JSON, density history, or Python coefficient copy is used
//! here. One immutable coefficient library is shared across all domain plans;
//! each mutable plan owns its own scratch. Input/output mode order is explicit,
//! and reverse dependencies retain every path that returns to the output.
use std::{collections::{BTreeMap,BTreeSet},sync::Arc,time::Instant};
use crate::{C,Coefficient,Finite,Lapack,product};
use crate::pencil::{Family,HilbertCoefficient,LayerInput,Library,Plan,PlanInput,ThinCoefficient,Workspace};
use crate::preparation::{Charge,FourierMatrices,GaugedSource,GaugePolicy,ModelView,PhaseBasis};
use crate::preparation::primitives::{self,PrimitivePreparation};

type Modes=BTreeSet<Charge>;
/// Worker start-up check: loads LAPACK and verifies one Hermitian
/// eigensystem, [[2,1-i],[1+i,3]] -> {1,4}.
pub fn self_check()->Result<(),String>{
    let a=[C{re:2.,im:0.},C{re:1.,im:1.},C{re:1.,im:-1.},C{re:3.,im:0.}];
    let(values,_)=Lapack::load()?.hermitian_eigen(&a,2)?;
    if (values[0]-1.).abs()>1e-12||(values[1]-4.).abs()>1e-12{return Err("LAPACK numerical self-check failed".into());}Ok(())
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
    /// S (the union of all shear layers), NL, NR, K, R, L_GG, L_OG.
    families:[Family;7],
    /// The shear layers, inner first, each a list of classes; the pencil
    /// applies the layers in order and sums the classes of one layer.
    layers:Vec<Vec<Family>>,
    lapack:Option<Arc<Lapack>>,
    preparation_lapack:Arc<Lapack>,
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
    /// Exactly Hermitian-symmetric inputs are evaluated on the canonical half
    /// of every mode set and mirrored.
    pub hermitian_half_capable:bool,
}

pub struct OwnedPencil {
    // Keeps the coordinate metadata alive even if its factory is dropped.
    primitives:Arc<PrimitivePreparation>,
    input_modes:Vec<Charge>,output_modes:Vec<Charge>,
    plan:Plan,
    workspace:Workspace,
    diagnostics:DomainDiagnostics,
}


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
    Ok(Coefficient{rows,row,col,value})
}

impl OwnedPrepared {
    /// The only public preparation constructor starts from original supplied
    /// Model matrices. Intermediate mutable primitive structs cannot silently
    /// inject another model or a stale incomplete radiation family.
    pub fn prepare(model:&ModelView<'_>,phases:PhaseBasis,gauge:GaugePolicy<'_>,policy:KernelPolicy)->Result<Self,String>{
        let lap=Arc::new(Lapack::load()?);
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
        let n=primitives.hilbert.source.n;let g=primitives.hilbert.source.ground.len();let e=primitives.hilbert.source.excited.len();
        let rank=primitives.gg_rows.len();let optical=primitives.optical_rows.len();let axes=primitives.hilbert.source.phases.axes.len();
        if n==0||g==0||e==0||g+e!=n{return Err("Invalid owned Hilbert partition".into());}
        // K and R act on vec(J_EE), which the kernel stores column-major.
        let excited_order:Vec<usize>=(0..e*e).map(|c|c/e+e*(c%e)).collect();
        let hilbert_families=[&primitives.s,&primitives.no_jump_left,&primitives.no_jump_right];
        let thin_families:[(&FourierMatrices,usize,usize,Option<&[usize]>);4]=[
            (&primitives.selected_k,rank,e*e,Some(&excited_order)),(&primitives.selected_recycling,rank,e*e,Some(&excited_order)),
            (&primitives.gg_hamiltonian,rank,rank,None),(&primitives.optical_hamiltonian,optical,rank,None)];
        let valid=|q:&Charge,matrix:&[C],length:usize|q.len()==axes&&matrix.len()==length&&matrix.iter().all(|z|z.finite());
        let mut library=Library::default();let mut families:[Family;7]=std::array::from_fn(|_|Family::new());
        let mut layers=Vec::with_capacity(primitives.layers.len());
        for layer in &primitives.layers{let mut classes=Vec::with_capacity(layer.len());for class in layer{let mut family=Family::new();
            for(q,matrix)in class{
                if !valid(q,matrix,n*n){return Err("Invalid owned complete primitive matrix".into());}
                family.insert(q.clone(),library.hilbert.len());library.hilbert.push(HilbertCoefficient::new(matrix,n,g));
            }classes.push(family);}layers.push(classes);}
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
        Ok(Self{primitives:Arc::new(primitives),library:Arc::new(library),families,layers,
            lapack:if matches!(policy,KernelPolicy::HybridBlas){Some(Arc::clone(&lap))}else{None},
            preparation_lapack:lap,coefficient_bytes:bytes,timing:[0.;3]})
    }
    pub fn primitives(&self)->&PrimitivePreparation{&self.primitives}
    /// The generator of the sheared coordinates at `charges`
    /// (`preparation::effective`).
    pub fn effective_generator(&self,charges:&[Charge])->Result<crate::preparation::effective::EffectiveGenerator,String>{
        crate::preparation::effective::EffectiveGenerator::new(&self.primitives,&self.preparation_lapack,charges)
    }
    /// Forward modes of every shear layer from `input`: ([S_c, .] modes per
    /// class, output modes) per layer, and the final J modes.
    fn forward_layers(&self,input:&Modes)->Result<(Vec<(Vec<Modes>,Modes)>,Modes),String>{
        let axes=self.primitives.hilbert.source.phases.axes.len();let mut current=input.clone();let mut out=Vec::with_capacity(self.layers.len());
        for classes in &self.layers{let mut firsts=Vec::with_capacity(classes.len());let mut support=Modes::from([vec![0;axes]]);
            for family in classes{let keys:Modes=family.keys().cloned().collect();
                firsts.push(combinations(&current,&keys,false)?);
                support.extend(keys.iter().cloned());support.extend(combinations(&keys,&keys,false)?);}
            let next=combinations(&current,&support,false)?;out.push((firsts,next.clone()));current=next;}
        Ok((out,current))
    }
    /// Every output mode one pencil application reaches from `input`. The
    /// estimate of the Fourier truncation must evaluate exactly these.
    pub fn reachable(&self,input:&[Charge])->Result<Vec<Charge>,String>{
        let keys:Vec<Modes>=self.families.iter().map(|f|f.keys().cloned().collect()).collect();
        let axes=self.primitives.hilbert.source.phases.axes.len();let input=mode_set(input,axes)?;
        let(_,forward_j)=self.forward_layers(&input)?;let z=combinations(&forward_j,&keys[3],false)?;
        let mut output=forward_j.clone();output.extend(z.iter().cloned());
        for family in[&keys[1],&keys[2],&keys[4]]{output.extend(combinations(&forward_j,family,false)?);}
        for family in[&keys[5],&keys[6]]{output.extend(combinations(&z,family,false)?);}
        Ok(output.into_iter().collect())
    }
    pub fn shared_coefficient_bytes(&self)->usize{self.coefficient_bytes}
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
        let(left,right,k,r,lgg,log)=(&keys[1],&keys[2],&keys[3],&keys[4],&keys[5],&keys[6]);
        let(forward,forward_j)=self.forward_layers(&input)?;
        let mut z_needed=output.clone();z_needed.extend(combinations(&output,lgg,true)?);z_needed.extend(combinations(&output,log,true)?);
        let z_modes:Vec<_>=combinations(&forward_j,k,false)?.intersection(&z_needed).cloned().collect();
        let no_jump:Modes=left.union(right).cloned().collect();
        let mut j_needed=output.clone();j_needed.extend(combinations(&output,&no_jump,true)?);
        j_needed.extend(combinations(&output,r,true)?);j_needed.extend(combinations(&z_modes.iter().cloned().collect(),k,true)?);
        let j_set:Modes=forward_j.intersection(&j_needed).cloned().collect();let j_modes:Vec<_>=j_set.iter().cloned().collect();
        // Backward through the layers: an output mode needs its input mode
        // (identity) and first mode (identity and [S, first]); a first mode
        // needs the input modes [S, .] maps onto it.
        let mut sets:Vec<(Vec<Vec<Charge>>,Vec<Charge>)>=vec![(Vec::new(),Vec::new());self.layers.len()];let mut needed=j_set.clone();
        for(l,classes)in self.layers.iter().enumerate().rev(){
            let(first_forward,out_forward)=&forward[l];
            let out:Modes=needed.intersection(out_forward).cloned().collect();let mut input_needed=out.clone();let mut firsts=Vec::with_capacity(classes.len());
            for(family,forward_first)in classes.iter().zip(first_forward){let keys:Modes=family.keys().cloned().collect();
                let mut first_needed=out.clone();first_needed.extend(combinations(&out,&keys,true)?);
                let first:Modes=first_needed.intersection(forward_first).cloned().collect();
                input_needed.extend(combinations(&first,&keys,true)?);firsts.push(first.into_iter().collect());}
            sets[l]=(firsts,out.into_iter().collect());needed=input_needed;
        }
        let layers:Vec<LayerInput<'_>>=self.layers.iter().zip(&sets).map(|(classes,(firsts,output))|LayerInput{
            classes:classes.iter().zip(firsts).map(|(family,first)|(family,first.as_slice())).collect(),output}).collect();
        let n=source.n;let g=source.ground.len();
        let frequency:Vec<f64>=output_modes.iter().map(|q|source.phases.frequency(q).map(|v|v.hi+v.lo)).collect::<Result<_,_>>()?;
        let plan=Plan::new(PlanInput{n,ground:g,library:Arc::clone(&self.library),
            layers,left:&self.families[1],right:&self.families[2],selected_k:&self.families[3],
            recycling:&self.families[4],ground_hamiltonian:&self.families[5],optical_hamiltonian:&self.families[6],
            input:input_modes,j:&j_modes,z:&z_modes,output:output_modes,
            gg:p.gg_rows.iter().map(|&row|(row/g,row%g)).collect(),optical:p.optical_rows.iter().map(|&at|(at/n,at%n)).collect(),
            frequency:frequency.clone(),max_rhs,blas:self.lapack.clone()})?;
        // A disjoint rectangular operator may have no intermediate modes; its
        // exact action is zero.
        let(first,j,z)=plan.intermediate_modes();
        let diagnostics=DomainDiagnostics{input_modes:plan.input_modes(),output_modes:plan.output_modes(),
            first_intermediate_modes:first,j_intermediate_modes:j,z_intermediate_modes:z,
            shared_coefficient_bytes:self.coefficient_bytes,domain_index_bytes:plan.index_bytes(),
            hermitian_half_capable:plan.hermitian_half_capable()};
        Ok(OwnedPencil{primitives:Arc::clone(&self.primitives),input_modes:input_modes.to_vec(),output_modes:output_modes.to_vec(),plan,
            workspace:Workspace::default(),diagnostics})
    }
}

impl OwnedPencil {
    pub fn input_modes(&self)->&[Charge]{&self.input_modes}
    pub fn output_modes(&self)->&[Charge]{&self.output_modes}
    pub fn input_dimension(&self)->usize{self.input_modes.len()*self.dimension()}
    pub fn output_dimension(&self)->usize{self.output_modes.len()*self.dimension()}
    fn dimension(&self)->usize{let n=self.primitives.hilbert.source.n;n*n}
    pub fn diagnostics(&self)->&DomainDiagnostics{&self.diagnostics}
    pub fn primitives(&self)->&PrimitivePreparation{&self.primitives}
    /// Upper bound of the scratch bytes one right-hand side needs.
    pub fn workspace_bytes_per_rhs(&self)->usize{self.plan.workspace_bytes_per_rhs()}
    fn validate(&self,input:&[C],rhs:usize,lengths:&[usize])->Result<(),String>{
        if rhs==0||rhs>self.plan.max_rhs(){return Err("RHS count exceeds this owned plan; caller must chunk complete columns".into());}
        let ni=product(&[self.input_dimension(),rhs])?;let no=product(&[self.output_dimension(),rhs])?;
        if input.len()!=ni||lengths.iter().any(|&v|v!=no){return Err("Owned pencil buffers must be exact row-major [coordinate,rhs] shapes".into());}
        Ok(())
    }
    /// E x and F x of a signed field with layout [mode][(i*n+j)*rhs+b].
    pub fn apply_into(&mut self,input:&[C],rhs:usize,mass:&mut[C],pencil:&mut[C])->Result<(),String>{
        self.validate(input,rhs,&[mass.len(),pencil.len()])?;
        let half=self.plan.apply(&mut self.workspace,input,rhs)?;
        self.plan.store(&self.workspace,half,rhs,mass,pencil)
    }
    /// shift * E x - F x.
    pub fn apply_shifted_into(&mut self,input:&[C],rhs:usize,shift:C,output:&mut[C])->Result<(),String>{
        self.validate(input,rhs,&[output.len()])?;if !shift.finite(){return Err("Nonfinite owned pencil shift".into());}
        let half=self.plan.apply(&mut self.workspace,input,rhs)?;
        self.plan.store_shifted(&self.workspace,shift,half,rhs,output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preparation::{DriveView,MatrixView};
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
    /// A near drive (detuning 0.1) and a strong far drive (detuning ~17)
    /// whose cross couplings are significant: the shear has two layers.
    fn two_layer_fixture()->OwnedPrepared{
        let n=4;let mut h=vec![0.;16];let zero=vec![0.;16];for(i,v)in[0.,0.6,3.,4.].iter().enumerate(){h[i*n+i]=*v;}
        let mut near=vec![0.;16];near[2]=0.01;near[2*n]=0.01;
        let mut far=vec![0.;16];for(i,j,v)in[(0,2,2.),(1,3,1.5),(0,3,0.7)]{far[i*n+j]=v;far[j*n+i]=v;}
        let drives=[DriveView{omega:2.9,cosine:MatrixView{real:&near,imag:&zero},sine:MatrixView{real:&zero,imag:&zero}},
            DriveView{omega:20.,cosine:MatrixView{real:&far,imag:&zero},sine:MatrixView{real:&zero,imag:&zero}}];
        let mut c=vec![0.;16];c[2]=0.7;c[3]=0.1;c[6]=0.2;c[7]=0.8;
        let channels=[MatrixView{real:&c,imag:&zero}];let model=ModelView{dimension:n,hamiltonian:MatrixView{real:&h,imag:&zero},
            collapse:&channels,drives:&drives,ground:&[true,true,false,false],initial_index:0};
        let phases=PhaseBasis::from_declared(&[2.9,20.],None,&[vec![1,0],vec![0,1]]).unwrap();
        OwnedPrepared::prepare(&model,phases,GaugePolicy::MaximumScoreGapForest,KernelPolicy::Csr).unwrap()
    }
    /// Full convolution independent of the staged return graph and compact
    /// K/R/LGG/LOG: form T x, then use original transformed H/C directly.
    fn original_oracle(p:&PrimitivePreparation,x:&FourierMatrices)->(FourierMatrices,FourierMatrices){
        let n=p.hilbert.source.n;let g=p.hilbert.source.ground.len();let e=n-g;
        let mut j=x.clone();
        for layer in &p.layers{let mut next=j.clone();for class in layer{let dx=comm(class,&j,n);let ddx=comm(class,&dx,n);
            for(q,a)in dx{add_field(&mut next,q,&a,1.);}for(q,a)in ddx{add_field(&mut next,q,&a,0.5);}}j=next;}
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
    /// Layered shears: the pencil applies T_far(T_near(x)) exactly, and
    /// `reachable` names every mode it reaches.
    #[test]
    fn preparation_owned_layered_shears_match_oracle(){
        let owned=two_layer_fixture();let p=owned.primitives();
        assert!(p.layers.len()>=2,"expected separate near and far layers, got {}",p.layers.len());
        let input_modes=[vec![0,0],vec![1,0],vec![0,-1]];let n=4;let d=n*n;let rhs=2;
        let x:Vec<_>=(0..input_modes.len()*d*rhs).map(|i|C{re:((i*17+3)%41)as f64/47.,im:((i*11+7)%29)as f64/-31.}).collect();
        let mut oracles=Vec::new();let mut targets=Modes::new();
        for b in 0..rhs{let input:FourierMatrices=input_modes.iter().enumerate().map(|(i,q)|(q.clone(),(0..d).map(|k|x[(i*d+k)*rhs+b]).collect())).collect();
            let oracle=original_oracle(p,&input);targets.extend(oracle.0.keys().chain(oracle.1.keys()).cloned());oracles.push(oracle);}
        let reach:Modes=owned.reachable(&input_modes).unwrap().into_iter().collect();
        assert!(targets.iter().all(|q|reach.contains(q)),"reachable misses an oracle output mode");
        let outputs:Vec<_>=targets.into_iter().collect();let mut plan=owned.pencil(&input_modes,&outputs,rhs).unwrap();
        let count=outputs.len()*d*rhs;let(mut e,mut f)=(vec![C::default();count],vec![C::default();count]);plan.apply_into(&x,rhs,&mut e,&mut f).unwrap();
        let scale=oracles.iter().flat_map(|(a,b)|a.values().chain(b.values())).flatten().map(|z|z.re.hypot(z.im)).fold(0_f64,f64::max);
        for(m,q)in outputs.iter().enumerate(){for k in 0..d{for b in 0..rhs{let at=(m*d+k)*rhs+b;
            for(actual,family)in[(e[at],&oracles[b].0),(f[at],&oracles[b].1)]{
                let expected=family.get(q).map_or(C::default(),|a|a[k]);
                assert!((actual.re-expected.re).hypot(actual.im-expected.im)<1e-12*scale,"mode {q:?} entry {k}");
            }
        }}}
    }
    /// The effective generator L'_0 is the charge-0 block of the pencil's
    /// own generator T^{-1} F (T^{-1} by a Neumann series on a window of
    /// modes) within its stated bounds, far closer than the bare L_0, and it
    /// conserves the trace and the Hermitian pairing exactly.
    #[test]
    fn preparation_effective_generator_matches_the_pencil_frame(){
        use crate::preparation::effective::EffectiveGenerator;
        let owned=two_layer_fixture();let p=owned.primitives();let lap=&owned.preparation_lapack;let n=4;let d=n*n;
        assert!(p.k_hat.is_empty(),"the fixture's frame must be the shear alone");
        let zero=vec![0,0];let near=vec![1,0];let far_near=vec![-1,1];
        let generator=EffectiveGenerator::new(p,lap,&[zero.clone(),near.clone(),vec![-1,0],far_near.clone()]).unwrap();
        // Hermitian pairing and exact trace conservation.
        let(i,j)=(generator.position(&near).unwrap(),generator.position(&[-1,0]).unwrap());
        for a in 0..n{for b in 0..n{for f in[&generator.hamiltonian,&generator.gamma]{
            let(x,y)=(f[i][a*n+b],f[j][b*n+a]);assert!((x.re-y.re).abs()+(x.im+y.im).abs()<1e-12*(1.+x.re.hypot(x.im)));}}}
        let x:Vec<C>=(0..d).map(|k|C{re:((k*7+3)%11)as f64/9.-0.5,im:((k*5+1)%13)as f64/17.-0.3}).collect();
        for q in [&zero,&near,&far_near]{let k=generator.position(q).unwrap();
            let r=generator.recycle(k,&x,lap).unwrap();let gm=&generator.gamma[k];
            let(mut tr,mut tg)=(C::default(),C::default());for a in 0..n{add(&mut tr,r[a*n+a]);for b in 0..n{add(&mut tg,mul(gm[a*n+b],x[b*n+a]));}}
            assert!((tr.re-tg.re).hypot(tr.im-tg.im)<1e-12,"trace at {q:?}");}
        // The pencil frame: z = T^{-1} (F x) at the central mode.
        let mut window:Modes=owned.reachable(&[zero.clone()]).unwrap().into_iter().collect();
        for _ in 0..2{let current:Vec<Charge>=window.iter().cloned().collect();window.extend(owned.reachable(&current).unwrap());}
        let window:Vec<Charge>=window.into_iter().collect();let m=window.len();let center=window.iter().position(|q|*q==zero).unwrap();
        let mut once=owned.pencil(&[zero.clone()],&window,1).unwrap();let(mut ex,mut fx)=(vec![C::default();m*d],vec![C::default();m*d]);
        once.apply_into(&x,1,&mut ex,&mut fx).unwrap();
        let mut full=owned.pencil(&window,&window,1).unwrap();let mut z=fx.clone();
        for _ in 0..200{let(mut tz,mut unused)=(vec![C::default();m*d],vec![C::default();m*d]);full.apply_into(&z,1,&mut tz,&mut unused).unwrap();
            let mut change=0_f64;for k in 0..m*d{let next=C{re:fx[k].re-(tz[k].re-z[k].re),im:fx[k].im-(tz[k].im-z[k].im)};
                change=change.max((next.re-z[k].re).hypot(next.im-z[k].im));z[k]=next;}
            if change<1e-14{break;}}
        let exact=&z[center*d..(center+1)*d];let effective=generator.apply(generator.position(&zero).unwrap(),&x,lap).unwrap();
        // The bare frame L_0 for comparison.
        let src=&p.hilbert.source;let(g,e)=(src.ground.len(),n-src.ground.len());let h=&src.hamiltonian[&zero];
        let mut gm=vec![C::default();d];if let Some(block)=p.hilbert.radiation.gamma.get(&zero){for a in 0..e{for b in 0..e{gm[(g+a)*n+g+b]=block[a*e+b];}}}
        let mut bare=vec![C::default();d];
        for a in 0..n{for b in 0..n{let mut v=C::default();for c in 0..n{
            let commutator=C{re:mul(h[a*n+c],x[c*n+b]).re-mul(x[a*n+c],h[c*n+b]).re,im:mul(h[a*n+c],x[c*n+b]).im-mul(x[a*n+c],h[c*n+b]).im};
            let anti=C{re:mul(gm[a*n+c],x[c*n+b]).re+mul(x[a*n+c],gm[c*n+b]).re,im:mul(gm[a*n+c],x[c*n+b]).im+mul(x[a*n+c],gm[c*n+b]).im};
            v.re+=commutator.im-0.5*anti.re;v.im+=-commutator.re-0.5*anti.im;}bare[a*n+b]=v;}}
        if let Some(r)=p.hilbert.radiation.recycling.get(&zero){for a in 0..g{for b in 0..g{for i in 0..e{for j in 0..e{
            add(&mut bare[a*n+b],mul(r[(a*g+b)*e*e+i*e+j],x[(g+i)*n+g+j]));}}}}}
        let distance=|a:&[C],b:&[C]|a.iter().zip(b).fold(0_f64,|s,(x,y)|s.hypot((x.re-y.re).hypot(x.im-y.im)));
        let size=x.iter().fold(0_f64,|s,z|s.hypot(z.re.hypot(z.im)));
        let(error,error_bare)=(distance(exact,&effective),distance(exact,&bare));
        eprintln!("effective error {error:.3e}, bare {error_bare:.3e}, bound {:.3e} + frame {:.3e} (|x| {size:.2})",generator.remainder,generator.frame_difference);
        assert!(error<=(generator.remainder+generator.frame_difference)*size,"outside the stated bounds");
        assert!(error<0.2*error_bare,"not closer than the bare generator");
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
