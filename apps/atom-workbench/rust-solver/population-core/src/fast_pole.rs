//! Validated SINGLE population-weighted fast-pole augmentation. The spectrum
//! chooses a numerical pole only. Every solve and closure uses original E/F;
//! no level, frequency, physical coefficient or emission channel is removed.
use super::*;
use super::workspace::{TrialWorkspace,AppendReport};

pub struct FastPoleSelection{owner:Arc<PrimitivePreparation>,pub pole:Option<C>,pub report:FastSelectionReport}
#[derive(Clone,Debug)]pub struct FastSelectionReport{
    pub selected:bool,pub reason:String,pub first_positive_time:Option<f64>,pub candidate_pairs:usize,
    pub eigen_condition:f64,pub relative_eigen_residual:f64,pub eigenvalue_uncertainty:f64,
    pub chosen_indices:Option<(usize,usize)>,pub population_weight:f64,pub hilbert_weight:f64,
    pub reused_current_left_frame:bool,pub maximum_pairs:usize,pub physical_terms_removed:bool,
}
#[derive(Clone,Debug)]pub struct FastPairInstall{
    pub added:usize,pub selected_pole:Option<C>,pub strict_iterations:usize,pub complete_m_products:usize,
    pub strict_original_relative_residual:f64,pub full_pair_closure_relative:f64,
    pub transported_image_columns:usize,pub recomputed_image_columns:usize,
    pub original_pair_images_reused:bool,pub auxiliary_real_pole_successors_not_scheduled:bool,
    pub accepted_trajectory:bool,pub elapsed_ms:f64,pub reason:String,
}
fn add(a:&mut C,b:C){a.re+=b.re;a.im+=b.im;}
fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
fn conj(a:C)->C{C{re:a.re,im:-a.im}}
fn sub(a:C,b:C)->C{C{re:a.re-b.re,im:a.im-b.im}}
fn scale(a:C,s:f64)->C{C{re:a.re*s,im:a.im*s}}
fn magnitude(a:C)->f64{a.re.hypot(a.im)}
fn cnorm(a:&[C])->f64{a.iter().fold(0_f64,|v,&z|v.hypot(magnitude(z)))}
fn cdot(a:&[C],b:&[C])->C{let mut sum=C::default();for(&a,&b)in a.iter().zip(b){add(&mut sum,mul(conj(a),b));}sum}
fn divide(a:C,b:C)->Result<C,String>{let length=magnitude(b);if !length.is_finite()||length==0.{return Err("Unresolved complex QR diagonal".into());}
    let answer=scale(mul(a,conj(scale(b,1./length))),1./length);if !answer.finite(){return Err("Complex QR division overflow".into());}Ok(answer)}
fn col(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|i|a[(i%n)*n+i/n]).collect()}
fn adj(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|i|conj(a[(i%n)*n+i/n])).collect()}
fn spec(lap:&Lapack,a:&[C],n:usize)->Result<f64,String>{let gram=lap.multiply_complex(&adj(a,n),a,n,n,n)?;let(values,_)=lap.hermitian_eigen(&gram,n)?;Ok(values.into_iter().fold(0_f64,f64::max).sqrt())}
fn inverse(lap:&Lapack,a:&[C],n:usize)->Result<Vec<C>,String>{let(lu,piv)=lap.complex_factor(a,n)?;let mut out=vec![C::default();n*n];
    for j in 0..n{let mut unit=vec![C::default();n];unit[j].re=1.;out[j*n..(j+1)*n].copy_from_slice(&lap.complex_solve_factored(&lu,&piv,&unit,n)?);}Ok(out)}

impl RuntimeCore{
    pub fn select_fast_pair(&self,times:&[f64],base_shift:f64)->Result<FastPoleSelection,String>{
        if !base_shift.is_finite()||base_shift<=0.||times.iter().any(|t|!t.is_finite()||*t<0.){return Err("Invalid fast-pole shift/time grid".into());}
        let mut report=FastSelectionReport{selected:false,reason:String::new(),first_positive_time:times.iter().copied().filter(|t|*t>0.).reduce(f64::min),candidate_pairs:0,
            eigen_condition:0.,relative_eigen_residual:0.,eigenvalue_uncertainty:0.,chosen_indices:None,population_weight:0.,hilbert_weight:0.,reused_current_left_frame:false,maximum_pairs:1,physical_terms_removed:false};
        let Some(tmin)=report.first_positive_time else{report.reason="No positive requested time; retain real-pole chain".into();return Ok(FastPoleSelection{owner:Arc::clone(&self.owned.primitives),pole:None,report});};
        let n=self.owned.primitives.hilbert.source.n;let lap=&self.owned.preparation_lapack;let a=col(&self.p0.factor().left,n);
        let(eigen,v,w)=if let Some(values)=self.p0.left_selector_frame(){report.reused_current_left_frame=true;values}else{let(e,v)=lap.complex_eigen_right(&a,n)?;let w=inverse(lap,&v,n)?;(e,v,w)};
        let gram=lap.multiply_complex(&adj(&v,n),&v,n,n,n)?;let(squared,_)=lap.hermitian_eigen(&gram,n)?;
        let largest=squared.iter().copied().fold(0_f64,f64::max);let smallest=squared.iter().copied().fold(f64::INFINITY,f64::min);
        if smallest<=0.||!largest.is_finite(){report.reason="Unresolved selector eigenframe; retain real-pole chain".into();return Ok(FastPoleSelection{owner:Arc::clone(&self.owned.primitives),pole:None,report});}
        let condition=(largest/smallest).sqrt();let av=lap.multiply_complex(&a,&v,n,n,n)?;
        let residual:Vec<_>=av.iter().enumerate().map(|(i,&value)|sub(value,mul(v[i],eigen[i/n]))).collect();
        let relative=cnorm(&residual)/(cnorm(&a).max(f64::MIN_POSITIVE)*cnorm(&v));
        let uncertainty=128.*f64::EPSILON*spec(lap,&a,n)?*condition+spec(lap,&residual,n)?/smallest.sqrt();
        report.eigen_condition=condition;report.relative_eigen_residual=relative;report.eigenvalue_uncertainty=uncertainty;
        if !condition.is_finite()||condition>1e6||!relative.is_finite()||relative>1e-11||!uncertainty.is_finite(){
            report.reason="Selector conditioning/backward error fails; retain real-pole chain".into();return Ok(FastPoleSelection{owner:Arc::clone(&self.owned.primitives),pole:None,report});}
        let sigma=col(self.initial.point(),n);let modal=lap.multiply_complex(&lap.multiply_complex(&w,&sigma,n,n,n)?,&adj(&w,n),n,n,n)?;
        let physical=lap.multiply_complex(&self.owned.primitives.hilbert.basis_column_major,&v,n,n,n)?;
        let nearest:Vec<_>=(0..n).map(|i|(0..n).filter(|&j|i!=j).map(|j|magnitude(sub(eigen[i],eigen[j]))).fold(f64::INFINITY,f64::min)).collect();
        let mut chosen=None;for i in 0..n{for j in i+1..n{
            let lambda=C{re:eigen[i].re+eigen[j].re,im:eigen[i].im-eigen[j].im};let frequency=lambda.im.abs();
            if frequency*tmin<1.||frequency<=4.*uncertainty||nearest[i].min(nearest[j])<=4.*uncertainty{continue;}
            let common=2.*magnitude(modal[i+j*n])*(lambda.re*tmin).exp();
            let score=(0..n).map(|k|common*magnitude(mul(physical[k+i*n],conj(physical[k+j*n])))).fold(0_f64,f64::max);
            if !score.is_finite()||score<=0.{continue;}report.candidate_pairs+=1;
            let hilbert=common*cnorm(&v[i*n..(i+1)*n])*cnorm(&v[j*n..(j+1)*n]);
            if chosen.as_ref().is_none_or(|&(old_score,old_frequency,_,_,_)|score>old_score||(score==old_score&&frequency<old_frequency)){chosen=Some((score,frequency,i,j,hilbert));}
        }}
        let pole=chosen.map(|(score,frequency,i,j,hilbert)|{report.selected=true;report.chosen_indices=Some((i,j));report.population_weight=score;report.hilbert_weight=hilbert;C{re:base_shift,im:frequency}});
        report.reason=if pole.is_some(){"Sole maximum population-amplitude pair; actual inverse remains original E/F"}else{"No resolved weighted fast pair; retain real-pole chain"}.into();
        Ok(FastPoleSelection{owner:Arc::clone(&self.owned.primitives),pole,report})
    }
}

struct FastOperator<'a>{domain:&'a mut Domain,preconditioner:P0Batch,pole:C,rhs_signed:Vec<C>,rhs_native:Vec<C>,
    last_values:Vec<f64>,last_e:Vec<f64>,last_f:Vec<f64>,last_closure:f64,products:usize}
impl FastOperator<'_>{
    fn precondition(&mut self,x:&[C])->Result<Vec<C>,String>{self.domain.prepare_buffers(1)?;self.domain.map.push_complex_into(x,1,&mut self.domain.signed)?;
        self.preconditioner.apply_into(&self.domain.signed,1,&mut self.domain.e)?;self.domain.map.pull_complex_into(&self.domain.e,1,&mut self.domain.native)?;Ok(self.domain.native.clone())}
    fn product(&mut self,x:&[C])->Result<Vec<C>,String>{self.domain.prepare_buffers(1)?;self.domain.map.push_complex_into(x,1,&mut self.domain.signed)?;
        self.domain.plan.apply_shifted_into(&self.domain.signed,1,self.pole,&mut self.domain.e)?;self.domain.map.pull_complex_into(&self.domain.e,1,&mut self.domain.native)?;self.products+=1;Ok(self.domain.native.clone())}
    /// Batch2 ORIGINAL images are retained for pair closure and later append.
    fn verify(&mut self,x:&[C])->Result<(f64,Vec<C>),String>{let n=x.len();let mut pair=vec![0.;2*n];for i in 0..n{pair[2*i]=x[i].re;pair[2*i+1]=x[i].im;}
        let(e,f)=self.domain.entered_images_columns(&pair,2)?;let mut residual=vec![C::default();n];let mut ru=Vec::with_capacity(n);let mut rv=Vec::with_capacity(n);
        for i in 0..n{let eu=self.domain.e[2*i];let ev=self.domain.e[2*i+1];let fu=self.domain.f[2*i];let fv=self.domain.f[2*i+1];
            let mass=C{re:eu.re-ev.im,im:eu.im+ev.re};let pencil=C{re:fu.re-fv.im,im:fu.im+fv.re};residual[i]=sub(self.rhs_signed[i],sub(mul(self.pole,mass),pencil));
            ru.push(sub(fu,sub(sub(scale(eu,self.pole.re),scale(ev,self.pole.im)),self.rhs_signed[i])));
            rv.push(sub(fv,C{re:self.pole.im*eu.re+self.pole.re*ev.re,im:self.pole.im*eu.im+self.pole.re*ev.im}));
        }
        let relative=cnorm(&residual)/cnorm(&self.rhs_signed).max(f64::MIN_POSITIVE);self.last_closure=cnorm(&ru).max(cnorm(&rv))/cnorm(&self.rhs_signed).max(f64::MIN_POSITIVE);
        self.last_values=vec![0.;2*n];self.last_e=vec![0.;2*n];self.last_f=vec![0.;2*n];
        for i in 0..n{for j in 0..2{self.last_values[j*n+i]=pair[2*i+j];self.last_e[j*n+i]=e[2*i+j];self.last_f[j*n+i]=f[2*i+j];}}
        let mut native=vec![C::default();n];self.domain.map.pull_complex_into(&residual,1,&mut native)?;self.products+=1;Ok((relative,native))
    }
}
fn complex_gmres(operator:&mut FastOperator<'_>)->Result<(Vec<C>,usize,f64),String>{
    let rhs=operator.rhs_native.clone();let n=rhs.len();let beta0=cnorm(&rhs);if beta0==0.{return Err("Fast-pole initial mass is unresolved".into());}
    let mut x=vec![C::default();n];let mut residual=rhs;let mut iterations=0;
    for _ in 0..4{let beta=cnorm(&residual);if beta==0.{return Err("Fast-pole restart lost its unresolved complex residual".into());}
        let cap=32.min(n);let stride=cap+1;let mut v=vec![residual.iter().map(|&z|scale(z,1./beta)).collect::<Vec<_>>()];let mut z=Vec::new();
        let mut h=vec![C::default();stride*cap];let mut cs=vec![0.;cap];let mut sn=vec![C::default();cap];let mut g=vec![C::default();stride];g[0].re=beta;
        let base=x.clone();let mut restarted=false;
        for j in 0..cap{let direction=operator.precondition(&v[j])?;let mut work=operator.product(&direction)?;let scale_work=cnorm(&work);z.push(direction);iterations+=1;
            for _ in 0..2{let coefficients:Vec<_>=v.iter().map(|basis|cdot(basis,&work)).collect();for i in 0..=j{add(&mut h[i+j*stride],coefficients[i]);}
                for k in 0..n{let mut correction=C::default();for i in 0..=j{add(&mut correction,mul(v[i][k],coefficients[i]));}work[k]=sub(work[k],correction);}}
            let tail=cnorm(&work);let breakdown=tail<=64.*f64::EPSILON*(n+j+1)as f64*scale_work.max(f64::MIN_POSITIVE);
            if !breakdown{h[j+1+j*stride].re=tail;v.push(work.iter().map(|&a|scale(a,1./tail)).collect());}
            for i in 0..j{let a=h[i+j*stride];let b=h[i+1+j*stride];h[i+j*stride]=C{re:cs[i]*a.re+mul(sn[i],b).re,im:cs[i]*a.im+mul(sn[i],b).im};h[i+1+j*stride]=sub(scale(b,cs[i]),mul(conj(sn[i]),a));}
            let a=h[j+j*stride];let b=h[j+1+j*stride];let aa=magnitude(a);let bb=magnitude(b);let length=aa.hypot(bb);
            if length==0.||!length.is_finite(){return Err("Fast-pole complex QR breakdown".into());}
            let phase=if aa==0.{C{re:1.,im:0.}}else{scale(a,1./aa)};cs[j]=aa/length;sn[j]=scale(mul(phase,conj(b)),1./length);h[j+j*stride]=scale(phase,length);h[j+1+j*stride]=C::default();
            g[j+1]=scale(mul(conj(sn[j]),g[j]),-1.);g[j]=scale(g[j],cs[j]);let k=j+1;
            if magnitude(g[k])>(1e-9*beta0).max(1e-13)&&!breakdown&&k<cap{continue;}
            let mut y=g[..k].to_vec();for i in(0..k).rev(){for l in i+1..k{y[i]=sub(y[i],mul(h[i+l*stride],y[l]));}y[i]=divide(y[i],h[i+i*stride])?;}
            let mut candidate=base.clone();for l in 0..k{for i in 0..n{add(&mut candidate[i],mul(z[l][i],y[l]));}}
            let(relative,restart)=operator.verify(&candidate)?;
            if relative.is_finite()&&relative<=3e-9&&operator.last_closure<=5e-9{return Ok((candidate,iterations,relative));}
            if breakdown||k==cap{x=candidate;residual=restart;restarted=true;break;}
        }
        if !restarted{return Err("Fast-pole solve ended without original residual state".into());}
    }Err("Fast-pole original complex M residual did not pass four32-step cycles".into())
}

impl RuntimeCore{
    pub fn add_fast_pair(&mut self,space:&mut TrialWorkspace,selection:&FastPoleSelection)->Result<FastPairInstall,String>{
        let began=Instant::now();if !space.belongs_to(&self.owned)||!Arc::ptr_eq(&selection.owner,&self.owned.primitives){return Err("Fast pair belongs to another current Model".into());}
        let mut report=FastPairInstall{added:0,selected_pole:selection.pole,strict_iterations:0,complete_m_products:0,strict_original_relative_residual:0.,full_pair_closure_relative:0.,
            transported_image_columns:0,recomputed_image_columns:0,original_pair_images_reused:false,auxiliary_real_pole_successors_not_scheduled:false,accepted_trajectory:false,elapsed_ms:0.,reason:selection.report.reason.clone()};
        let Some(pole)=selection.pole else{return Ok(report);};
        if space.columns()!=1||pole.re.to_bits()!=space.pole().to_bits()||pole.im<=0.{return Err("Install the sole fast pair at the cold initial seed with matching real pole".into());}
        let source=space.initial_native().to_vec();let initial_scale=norm(&source);let modes=space.modes().to_vec();let phases=&self.owned.primitives.hilbert.source.phases;
        let shifts=modes.iter().map(|q|phases.frequency(q).map(|f|C{re:pole.re,im:pole.im+f.hi+f.lo})).collect::<Result<Vec<_>,_>>()?;
        let preconditioner=self.p0.prepare_batch(&shifts,1)?;let rows=source.len();
        // Initial column's actual signed E image is still in the workspace;
        // reuse it instead of another seed action. Scaling is linear only.
        let domain=space.native_domain();if domain.entered!=1||domain.m_products!=0||domain.p0_calls!=0{return Err("Install the fast pair before advancing the cold frontier".into());}
        domain.prepare_buffers(1)?;let rhs_signed:Vec<_>=domain.e.iter().map(|&z|scale(z,initial_scale)).collect();
        let mut rhs_native=vec![C::default();rows];domain.map.pull_complex_into(&rhs_signed,1,&mut rhs_native)?;
        let mut operator=FastOperator{domain,preconditioner,pole,rhs_signed,rhs_native,last_values:vec![],last_e:vec![],last_f:vec![],last_closure:0.,products:0};
        let solved=complex_gmres(&mut operator);let values=std::mem::take(&mut operator.last_values);let e=std::mem::take(&mut operator.last_e);let f=std::mem::take(&mut operator.last_f);
        report.complete_m_products=operator.products;report.full_pair_closure_relative=operator.last_closure;
        drop(operator);
        match solved{
            Ok((_,steps,residual))=>{report.strict_iterations=steps;report.strict_original_relative_residual=residual;
                let before=space.columns();let AppendReport{retained_columns,transported_image_columns,recomputed_image_columns,..}=space.append_verified_pair(&values,&e,&f)?;
                report.added=retained_columns;report.transported_image_columns=transported_image_columns;report.recomputed_image_columns=recomputed_image_columns;
                space.register_verified_fast_auxiliaries(before,retained_columns);report.original_pair_images_reused=transported_image_columns>0;
                report.auxiliary_real_pole_successors_not_scheduled=true;report.reason="Sole strict complex pair installed; original E/F closure verified; mature slow chain retained".into();
            },
            Err(reason)=>{report.reason=format!("Fast numerical augmentation unavailable; retain complete real-pole chain: {reason}");},
        }
        report.elapsed_ms=began.elapsed().as_secs_f64()*1000.;Ok(report)
    }
}

#[cfg(test)]mod tests{
    use super::*;use super::super::workspace::DirectionPolicy;
    #[test]fn preparation_fast_single_population_pair_uses_original_complex_inverse_and_images(){
        let owned=super::super::super::tests::fixture();let mut runtime=RuntimeCore::from_owned(owned,InitialView{index:0,real:None,imag:None},P0Policy::EigenPreferred).unwrap();
        let times=[0.,0.5,1.];let selection=runtime.select_fast_pair(&times,8.).unwrap();assert!(selection.pole.is_some());assert_eq!(selection.report.maximum_pairs,1);assert!(!selection.report.physical_terms_removed);
        let mut space=runtime.cold_workspace(&[vec![0],vec![1],vec![-1]],&times,8.,DirectionPolicy::Fixed{steps:1}).unwrap();let report=runtime.add_fast_pair(&mut space,&selection).unwrap();
        assert!(report.added>0, "{}",report.reason);assert!(report.strict_original_relative_residual<=3e-9);assert!(report.full_pair_closure_relative<=5e-9);
        assert_eq!(space.statistics().frontier_pending,1);let candidate=space.candidate().unwrap();assert!(!candidate.diagnostics.accepted);
        let map=runtime.prepared().real_fourier_map(&candidate.modes).unwrap();let mut original=runtime.prepared().pencil(&candidate.modes,&candidate.modes,1).unwrap();
        for column in 0..candidate.dimension{let mut signed=vec![C::default();candidate.rows];let mut e=signed.clone();let mut f=e.clone();let mut native=e.clone();
            map.push_into(&candidate.q[column*candidate.rows..(column+1)*candidate.rows],1,&mut signed).unwrap();original.apply_into(&signed,1,&mut e,&mut f).unwrap();
            for(image,stored)in[(&e,&candidate.eq),(&f,&candidate.fq)]{map.pull_complex_into(image,1,&mut native).unwrap();for i in 0..candidate.rows{let value=stored[column*candidate.rows+i];
                assert!((native[i].re-value).hypot(native[i].im)<3e-11*(1.+value.abs()));}}
        }
        for t in 0..times.len(){let rho=runtime.reconstruct_candidate_at(&candidate,t).unwrap();for i in 0..4{assert!((rho[i*4+i].re-candidate.populations[t*4+i].re).abs()<1e-11);}}
    }
    #[test]fn preparation_fast_selector_has_no_reference_or_nonpositive_time_pair(){
        let owned=super::super::super::tests::fixture();let runtime=RuntimeCore::from_owned(owned,InitialView{index:0,real:None,imag:None},P0Policy::EigenPreferred).unwrap();
        assert!(runtime.select_fast_pair(&[0.],8.).unwrap().pole.is_none());assert!(runtime.select_fast_pair(&[0.,0.1],0.).is_err());
    }
}
