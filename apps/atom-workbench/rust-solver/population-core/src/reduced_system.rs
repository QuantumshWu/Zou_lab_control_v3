//! Complete reduced mass/pencil and actual-time action. The matrix exponential
//! is the existing production dense_expm.rs; only factor scheduling/time-grid
//! bookkeeping is supplied here. No eigenmode/forcing/constant component is
//! deleted or repaired, and no error-control/physical acceptance is implied.
use super::*;
pub use crate::dense_expm::StepDiagnostics;

#[derive(Clone,Debug)]
pub struct ReducedPreparationDiagnostics{
    pub dimension:usize,pub mass_condition:f64,pub mass_solve_backward_error:f64,
    pub complete_mass_and_pencil_retained:bool,pub trace_test_modified_here:bool,
}
#[derive(Clone,Debug)]
pub struct ReducedActionDiagnostics{
    pub method:&'static str,pub dimension:usize,pub samples:usize,
    pub production_pade_steps:Vec<StepDiagnostics>,pub distinct_recurrence_factors:usize,
    pub nominal_step:f64,pub maximum_applied_time_correction:f64,
    pub maximum_local_taylor_remainder:f64,pub time_correction_goal:f64,
    pub actual_requested_times_preserved:bool,pub trace_or_initial_repair:bool,
    pub local_remainder_is_full_accuracy_certificate:bool,
}
pub struct ReducedAction{pub states:Vec<f64>,pub diagnostics:ReducedActionDiagnostics}
pub struct ReducedSystem{lapack:Arc<Lapack>,generator:Vec<f64>,diagnostics:ReducedPreparationDiagnostics}

fn finite(a:&[f64],label:&str)->Result<(),String>{if a.iter().any(|x|!x.is_finite()){Err(format!("Nonfinite reduced {label}"))}else{Ok(())}}
fn one_norm(a:&[f64],n:usize)->f64{a.chunks_exact(n).map(|c|c.iter().map(|v|v.abs()).sum::<f64>()).fold(0_f64,f64::max)}
fn inf_norm(a:&[f64],n:usize)->f64{(0..n).map(|i|(0..n).map(|j|a[i+j*n].abs()).sum::<f64>()).fold(0_f64,f64::max)}
fn maximum(a:&[f64])->f64{a.iter().map(|x|x.abs()).fold(0_f64,f64::max)}
fn identity(n:usize)->Vec<f64>{let mut a=vec![0.;n*n];for i in 0..n{a[i+i*n]=1.;}a}
fn two_sum(a:f64,b:f64)->(f64,f64){let high=a+b;let other=high-a;(high,(a-(high-other))+(b-other))}
fn compensated(values:&[f64])->f64{let mut hi=0.;let mut lo=0.;for &value in values{let(s,e)=two_sum(hi,value);hi=s;lo+=e;}hi+lo}

impl ReducedSystem{
    pub(super) fn prepare(lapack:Arc<Lapack>,mass:&[f64],pencil:&[f64],n:usize)->Result<Self,String>{
        if n==0||mass.len()!=product(&[n,n])?||pencil.len()!=mass.len(){return Err("Reduced E/F must be complete equal column-major squares".into());}
        finite(mass,"mass")?;finite(pencil,"pencil")?;let(lu,pivots)=lapack.factor(mass,n)?;
        let condition=lapack.condition(&lu,n,one_norm(mass,n))?;
        if !condition.is_finite(){return Err("Reduced mass condition is not finite".into());}
        // One original E factorization, complete multiple-RHS solve for F.
        let generator=lapack.solve_matrix_factored(&lu,&pivots,pencil,n)?;finite(&generator,"generator")?;
        let represented=lapack.multiply(mass,&generator,n,n,n);let difference:Vec<_>=represented.iter().zip(pencil).map(|(a,b)|a-b).collect();
        let denominator=one_norm(mass,n)*one_norm(&generator,n)+one_norm(pencil,n);
        let residual=if denominator==0.{one_norm(&difference,n)}else{one_norm(&difference,n)/denominator};
        if !residual.is_finite(){return Err("Reduced original mass solve residual is not finite".into());}
        Ok(Self{lapack,generator,diagnostics:ReducedPreparationDiagnostics{dimension:n,mass_condition:condition,mass_solve_backward_error:residual,
            complete_mass_and_pencil_retained:true,trace_test_modified_here:false}})
    }
    pub fn generator(&self)->&[f64]{&self.generator}
    pub fn diagnostics(&self)->&ReducedPreparationDiagnostics{&self.diagnostics}
    fn exponential(&self,time:f64,steps:&mut Vec<StepDiagnostics>)->Result<Vec<f64>,String>{
        let(result,diagnostic)=crate::dense_expm::exponential(&self.lapack,&self.generator,self.diagnostics.dimension,time)?;
        steps.push(diagnostic);Ok(result)
    }
    fn tiny_correction(&self,state:&[f64],delta:f64,norm_g:f64,goal:f64,steps:&mut Vec<StepDiagnostics>)->Result<(Vec<f64>,f64),String>{
        if delta==0.{return Ok((state.to_vec(),0.));}let n=state.len();let alpha=delta.abs()*norm_g;
        if alpha.is_finite()&&alpha<=1.{let bound=alpha.exp()*alpha.powi(3)*maximum(state)/6.;
            if bound<=goal{let scaled:Vec<_>=self.generator.iter().map(|v|v*delta).collect();
                let first=self.lapack.multiply(&scaled,state,n,1,n);let second=self.lapack.multiply(&scaled,&first,n,1,n);
                let corrected=state.iter().zip(first).zip(second).map(|((&a,b),c)|a+b+0.5*c).collect();return Ok((corrected,bound));}
        }
        let matrix=self.exponential(delta,steps)?;Ok((self.lapack.multiply(&matrix,state,n,1,n),0.))
    }
    /// initial contains every reduced coordinate, including any explicit
    /// affine constant. It is never reset to one after an exponential step.
    pub fn evolve(&self,initial:&[f64],times:&[f64])->Result<ReducedAction,String>{
        let n=self.diagnostics.dimension;if initial.len()!=n||times.is_empty()||times.iter().any(|t|!t.is_finite()||*t<0.)||times.windows(2).any(|a|a[0]>a[1]){
            return Err("Reduced initial/time grid must be complete, finite, nonnegative and ordered".into());}
        finite(initial,"initial")?;let norm_g=inf_norm(&self.generator,n);
        if !norm_g.is_finite(){return Err("Reduced generator infinity norm overflow".into());}
        let goal=1e-12*maximum(initial).max(1.);let mut steps=Vec::new();let t0=times[0];
        let start=if t0==0.{initial.to_vec()}else{let matrix=self.exponential(t0,&mut steps)?;self.lapack.multiply(&matrix,initial,n,1,n)};
        let mut states=vec![0.;product(&[times.len(),n])?];let mut max_correction=0_f64;let mut max_remainder=0_f64;
        let mut nominal=0.;let method;let factors;
        if times.len()==1{states.copy_from_slice(&start);method="single existing production Pade action";factors=0;}
        else{
            nominal=(times[times.len()-1]-t0)/(times.len()-1)as f64;
            let offsets:Vec<_>=times.iter().enumerate().map(|(i,&t)|{let high=(i as f64)*nominal;let low=(i as f64).mul_add(nominal,-high);compensated(&[t,-t0,-high,-low])}).collect();
            let largest=maximum(&offsets);let eligibility=norm_g*largest;
            if eligibility.is_finite()&&eligibility<=1e-3{
                method="one production Pade factor plus actual-time corrected recurrence";
                let matrix=if nominal==0.{identity(n)}else{self.exponential(nominal,&mut steps)?};factors=usize::from(nominal!=0.);
                let mut raw=start;for(i,&delta)in offsets.iter().enumerate(){if i>0{raw=self.lapack.multiply(&matrix,&raw,n,1,n);}
                    let(value,bound)=self.tiny_correction(&raw,delta,norm_g,goal,&mut steps)?;
                    states[i*n..(i+1)*n].copy_from_slice(&value);max_correction=max_correction.max(delta.abs());max_remainder=max_remainder.max(bound);
                }
            }else{
                method="existing production Pade factors on actual nonuniform increments";
                let mut cache=BTreeMap::<u64,Vec<f64>>::new();let mut raw=start;states[..n].copy_from_slice(&raw);
                for i in 1..times.len(){let(high,low)=two_sum(times[i],-times[i-1]);let key=high.to_bits();
                    if !cache.contains_key(&key){cache.insert(key,if high==0.{identity(n)}else{self.exponential(high,&mut steps)?});}
                    raw=self.lapack.multiply(&cache[&key],&raw,n,1,n);let(value,bound)=self.tiny_correction(&raw,low,norm_g,goal,&mut steps)?;raw=value;
                    states[i*n..(i+1)*n].copy_from_slice(&raw);max_correction=max_correction.max(low.abs());max_remainder=max_remainder.max(bound);
                }factors=cache.len();
            }
        }
        finite(&states,"complete action")?;
        Ok(ReducedAction{states,diagnostics:ReducedActionDiagnostics{method,dimension:n,samples:times.len(),production_pade_steps:steps,
            distinct_recurrence_factors:factors,nominal_step:nominal,maximum_applied_time_correction:max_correction,
            maximum_local_taylor_remainder:max_remainder,time_correction_goal:goal,actual_requested_times_preserved:true,
            trace_or_initial_repair:false,local_remainder_is_full_accuracy_certificate:false}})
    }
}
impl OwnedPrepared{
    /// Caller supplies the FINAL original E/F projection, including its
    /// separately verified trace-test rows if used. No row is changed here.
    pub fn reduced_system(&self,mass:&[f64],pencil:&[f64],dimension:usize)->Result<ReducedSystem,String>{
        ReducedSystem::prepare(Arc::clone(&self.preparation_lapack),mass,pencil,dimension)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn preparation_reduced_system_nonsymmetric_mass_affine_jordan_and_grid(){
        let owner=super::super::tests::fixture();let lap=&owner.preparation_lapack;
        // G represents z1'=z2, z2'=constant, constant'=0, not a reset rule.
        let g=[0.,0.,0.,1.,0.,0.,0.,1.,0.];let mass=[1.,0.,0.,0.2,1.3,0.,-0.1,0.4,0.8];let pencil=lap.multiply(&mass,&g,3,3,3);
        let system=owner.reduced_system(&mass,&pencil,3).unwrap();assert!(system.diagnostics.mass_solve_backward_error<1e-15);
        for times in[vec![0.,0.1,0.2,0.30000000000000004],vec![0.,0.03,0.03,0.37,0.91]]{
            let result=system.evolve(&[0.,0.,1.],&times).unwrap();for(i,&t)in times.iter().enumerate(){let x=&result.states[i*3..(i+1)*3];
                assert!((x[0]-0.5*t*t).abs()<5e-14);assert!((x[1]-t).abs()<5e-14);assert!((x[2]-1.).abs()<5e-14);}
            assert!(!result.diagnostics.trace_or_initial_repair);assert!(!result.diagnostics.local_remainder_is_full_accuracy_certificate);
        }
        let constant=owner.reduced_system(&identity(2),&[0.;4],2).unwrap().evolve(&[0.4,-0.2],&[0.,0.1,0.3]).unwrap();
        assert_eq!(constant.states,vec![0.4,-0.2,0.4,-0.2,0.4,-0.2]);
    }
    #[test]
    fn preparation_reduced_system_preserves_floating_actual_times_and_rejects_bad_mass(){
        let owner=super::super::tests::fixture();let omega=1e5;let generator=[0.,omega,-omega,0.];let system=owner.reduced_system(&identity(2),&generator,2).unwrap();
        let times:Vec<_>=(0..11).map(|i|i as f64*0.0000003).collect();let result=system.evolve(&[1.,0.],&times).unwrap();
        assert_eq!(result.diagnostics.distinct_recurrence_factors,1);assert_eq!(result.diagnostics.production_pade_steps.len(),1);
        for(i,&t)in times.iter().enumerate(){assert!((result.states[2*i]-(omega*t).cos()).abs()<2e-13);assert!((result.states[2*i+1]-(omega*t).sin()).abs()<2e-13);}
        assert!(owner.reduced_system(&[0.;4],&generator,2).is_err());assert!(system.evolve(&[1.,0.],&[0.2,0.1]).is_err());assert!(system.evolve(&[f64::NAN,0.],&[0.]).is_err());
    }
}
