//! Continuous REAL source Gram for trial-space importance only.
//!
//! W(T)=integral exp(Gt)c0 c0^T exp(G^T t) dt. A small-step Lyapunov
//! Taylor integral and W(2h)=W(h)+E(h)W(h)E(h)^T avoid exp(-G^T T),
//! spectral division, unresolved time sampling and special zero-mode rules.
//! E(h) comes ONLY from the existing production Pade implementation.
use super::*;

#[derive(Clone,Debug)]
pub struct LocalGramDiagnostics{
    pub scaling_depth:u32,pub local_step:f64,pub local_lyapunov_argument:f64,
    pub local_taylor_terms:usize,pub local_exact_arithmetic_tail:f64,
    pub maximum_arithmetic_skew_hs:f64,pub maximum_propagator_norm_upper:f64,
    pub production_pade:crate::dense_expm::StepDiagnostics,
}
#[derive(Clone,Debug)]
pub struct SourceGramDiagnostics{
    pub dimension:usize,pub duration:f64,pub exact_zero_source_or_horizon:bool,
    pub primary:Option<LocalGramDiagnostics>,pub alternate:Option<LocalGramDiagnostics>,
    pub scaling_comparison_hs:Option<f64>,pub scaling_comparison_relative:Option<f64>,
    pub finite_lyapunov_identity_hs:f64,pub finite_lyapunov_identity_relative:f64,
    pub minimum_eigenvalue:f64,pub negative_eigenvalue_mass:f64,pub trace:f64,
    pub time_sampling_used:bool,pub negative_eigenvalues_clipped:bool,
    pub spectral_truncation:bool,pub exponential_of_minus_adjoint_used:bool,
    pub proposal_selection_only:bool,pub physical_error_certificate:bool,pub elapsed_ms:f64,
}
pub struct ContinuousSourceGram{
    /// Complete symmetric column-major [dimension,dimension] Gram.
    pub matrix:Vec<f64>,pub diagnostics:SourceGramDiagnostics,
}
fn hs(a:&[f64])->f64{a.iter().fold(0_f64,|sum,&x|sum.hypot(x))}
fn one(a:&[f64],n:usize)->f64{a.chunks_exact(n).map(|c|c.iter().map(|x|x.abs()).sum::<f64>()).fold(0_f64,f64::max)}
fn inf(a:&[f64],n:usize)->f64{(0..n).map(|i|(0..n).map(|j|a[i+j*n].abs()).sum::<f64>()).fold(0_f64,f64::max)}
fn finite(a:&[f64],label:&str)->Result<(),String>{if a.iter().any(|x|!x.is_finite()){Err(format!("Nonfinite source Gram {label}"))}else{Ok(())}}
fn transpose(a:&[f64],n:usize)->Vec<f64>{(0..n*n).map(|k|a[(k%n)*n+k/n]).collect()}
fn symmetric(a:&mut[f64],n:usize)->f64{
    let mut skew=0_f64;for i in 0..n{for j in i+1..n{let x=a[i+j*n];let y=a[j+i*n];skew=skew.hypot(x-y).hypot(y-x);
        let mean=0.5*x+0.5*y;a[i+j*n]=mean;a[j+i*n]=mean;}}skew
}
fn product_matrix(lap:&Lapack,a:&[f64],b:&[f64],n:usize,label:&str)->Result<Vec<f64>,String>{
    let out=lap.multiply(a,b,n,n,n);finite(&out,label)?;Ok(out)
}

fn at_depth(lap:&Lapack,g:&[f64],source:&[f64],n:usize,duration:f64,extra:u32)->Result<(Vec<f64>,Vec<f64>,LocalGramDiagnostics),String>{
    let norm_g=one(g,n)+inf(g,n);let argument=duration*norm_g;
    if !argument.is_finite(){return Err("Source Gram integration scale overflow".into());}
    let base=if argument>0.25{(argument.log2()+2.).ceil().max(0.) as u32}else{0};
    let depth=base.checked_add(extra).ok_or("Source Gram scale-depth overflow")?;
    let mut h=duration;for _ in 0..depth{h*=0.5;}
    if !h.is_finite()||h<=0.{return Err("Source Gram local-step underflow".into());}
    let b:Vec<_>=g.iter().map(|x|x*h).collect();let bt=transpose(&b,n);let alpha=h*norm_g;
    let(mut e,pade)=crate::dense_expm::exponential(lap,g,n,h)?;
    let mut gram:Vec<_>=source.iter().map(|x|x*h).collect();let mut term=gram.clone();
    let source_norm=hs(source);if !source_norm.is_finite(){return Err("Source Gram source norm overflow".into());}
    finite(&gram,"initial local integral")?;
    let mut coefficient=1.;let mut tail=h*source_norm*alpha.exp()*alpha/2.;
    let target=0.125*f64::EPSILON*h*source_norm;let mut terms=0;let mut converged=false;
    for degree in 1..=64{
        if tail<=target{converged=true;break;}
        let left=product_matrix(lap,&b,&term,n,"local left product")?;
        let right=product_matrix(lap,&term,&bt,n,"local right product")?;
        for k in 0..term.len(){term[k]=(left[k]+right[k])/(degree+1)as f64;gram[k]+=term[k];}
        finite(&gram,"local Taylor accumulation")?;
        coefficient*=alpha/(degree+1)as f64;
        tail=h*source_norm*alpha.exp()*coefficient*alpha/(degree+2)as f64;terms=degree;
    }
    if !converged&&tail>target{return Err("Local continuous Gram series did not meet its numerical target".into());}
    let mut skew=symmetric(&mut gram,n);let mut e_norm=one(&e,n).sqrt()*inf(&e,n).sqrt();
    for _ in 0..depth{
        let left=product_matrix(lap,&e,&gram,n,"doubled left product")?;
        let correction=product_matrix(lap,&left,&transpose(&e,n),n,"doubled right product")?;
        for(a,b)in gram.iter_mut().zip(correction){*a+=b;}finite(&gram,"doubling accumulation")?;
        skew=skew.max(symmetric(&mut gram,n));e=product_matrix(lap,&e,&e,n,"propagator square")?;
        e_norm=e_norm.max(one(&e,n).sqrt()*inf(&e,n).sqrt());
    }
    Ok((gram,e,LocalGramDiagnostics{scaling_depth:depth,local_step:h,local_lyapunov_argument:alpha,
        local_taylor_terms:terms,local_exact_arithmetic_tail:tail,maximum_arithmetic_skew_hs:skew,
        maximum_propagator_norm_upper:e_norm,production_pade:pade}))
}

pub(super) fn prepare(lap:&Lapack,g:&[f64],initial:&[f64],duration:f64,check_scaling:bool)->Result<ContinuousSourceGram,String>{
    let began=Instant::now();let n=initial.len();
    if n==0||g.len()!=product(&[n,n])?||!duration.is_finite()||duration<0.{return Err("Complete finite G/c0 and nonnegative horizon required".into());}
    finite(g,"generator")?;finite(initial,"initial vector")?;
    let mut diagnostic=SourceGramDiagnostics{dimension:n,duration,exact_zero_source_or_horizon:false,
        primary:None,alternate:None,scaling_comparison_hs:None,scaling_comparison_relative:None,
        finite_lyapunov_identity_hs:0.,finite_lyapunov_identity_relative:0.,minimum_eigenvalue:0.,negative_eigenvalue_mass:0.,trace:0.,
        time_sampling_used:false,negative_eigenvalues_clipped:false,spectral_truncation:false,
        exponential_of_minus_adjoint_used:false,proposal_selection_only:true,physical_error_certificate:false,elapsed_ms:0.};
    if duration==0.||initial.iter().all(|&x|x==0.){
        diagnostic.exact_zero_source_or_horizon=true;diagnostic.elapsed_ms=began.elapsed().as_secs_f64()*1000.;
        return Ok(ContinuousSourceGram{matrix:vec![0.;g.len()],diagnostics:diagnostic});
    }
    let mut source=vec![0.;g.len()];
    for j in 0..n{for i in 0..n{source[i+j*n]=initial[i]*initial[j];
        if initial[i]!=0.&&initial[j]!=0.&&source[i+j*n]==0.{return Err("Nonzero source outer-product entry underflowed".into());}
    }}finite(&source,"source outer product")?;
    let(matrix,endpoint,local)=at_depth(lap,g,&source,n,duration,0)?;diagnostic.primary=Some(local);
    if check_scaling{
        let(other,_,detail)=at_depth(lap,g,&source,n,duration,1)?;
        let difference=hs(&matrix.iter().zip(other).map(|(a,b)|a-b).collect::<Vec<_>>());
        diagnostic.scaling_comparison_hs=Some(difference);diagnostic.scaling_comparison_relative=Some(difference/hs(&matrix).max(f64::MIN_POSITIVE));
        diagnostic.alternate=Some(detail);
    }
    let state=lap.multiply(&endpoint,initial,n,1,n);finite(&state,"endpoint state")?;
    let left=product_matrix(lap,g,&matrix,n,"Lyapunov left")?;
    let right=product_matrix(lap,&matrix,&transpose(g,n),n,"Lyapunov right")?;
    let endpoint_source:Vec<_>=(0..n*n).map(|i|state[i%n]*state[i/n]-source[i]).collect();
    let defect:Vec<_>=(0..n*n).map(|i|left[i]+right[i]-endpoint_source[i]).collect();
    finite(&endpoint_source,"endpoint outer product")?;finite(&defect,"Lyapunov diagnostic")?;
    diagnostic.finite_lyapunov_identity_hs=hs(&defect);
    diagnostic.finite_lyapunov_identity_relative=hs(&defect)/(hs(&left)+hs(&right)+hs(&endpoint_source)).max(f64::MIN_POSITIVE);
    let complex:Vec<_>=matrix.iter().map(|&re|C{re,im:0.}).collect();let(eigen,_)=lap.hermitian_eigen(&complex,n)?;
    diagnostic.minimum_eigenvalue=eigen[0];diagnostic.negative_eigenvalue_mass=eigen.iter().map(|x|(-x).max(0.)).sum();
    diagnostic.trace=(0..n).map(|i|matrix[i+i*n]).sum();diagnostic.elapsed_ms=began.elapsed().as_secs_f64()*1000.;
    Ok(ContinuousSourceGram{matrix,diagnostics:diagnostic})
}

impl OwnedPrepared{
    pub fn continuous_source_gram(&self,generator:&[f64],initial:&[f64],duration:f64,check_scaling:bool)->Result<ContinuousSourceGram,String>{
        prepare(&self.preparation_lapack,generator,initial,duration,check_scaling)
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn close(a:&[f64],b:&[f64],tolerance:f64){for(a,b)in a.iter().zip(b){assert!((a-b).abs()<=tolerance*(1.+b.abs()),"{a} != {b}");}}
    #[test]
    fn preparation_source_gram_zero_and_repeated_jordan_modes(){
        let owner=super::super::tests::fixture();let t=0.7;
        let zero=owner.continuous_source_gram(&[0.;4],&[0.3,-0.4],t,true).unwrap();
        close(&zero.matrix,&[0.09*t,-0.12*t,-0.12*t,0.16*t],2e-14);
        let jordan=owner.continuous_source_gram(&[0.,0.,1.,0.],&[0.,1.],t,true).unwrap();
        close(&jordan.matrix,&[t*t*t/3.,t*t/2.,t*t/2.,t],3e-13);
        assert!(!jordan.diagnostics.physical_error_certificate);assert!(!jordan.diagnostics.spectral_truncation);
    }
    #[test]
    fn preparation_source_gram_resolves_fast_rotation_without_sampling(){
        let owner=super::super::tests::fixture();let w:f64=1e5;let t:f64=0.0314159;
        let gram=owner.continuous_source_gram(&[0.,w,-w,0.],&[1.,0.],t,true).unwrap();
        let diagonal=(2.*w*t).sin()/(4.*w);let cross=(1.-(2.*w*t).cos())/(4.*w);
        close(&gram.matrix,&[t/2.+diagonal,cross,cross,t/2.-diagonal],2e-10);
        assert!(!gram.diagnostics.time_sampling_used);assert!(gram.diagnostics.primary.as_ref().unwrap().scaling_depth>10);
    }
    #[test]
    fn preparation_source_gram_ten_ms_decay_and_exact_zero_horizon(){
        let owner=super::super::tests::fixture();let t=10000.;
        let gram=owner.continuous_source_gram(&[0.,0.,0.,-2.],&[0.3,0.7],t,true).unwrap();
        close(&gram.matrix,&[0.09*t,0.105,0.105,0.1225],4e-10);
        assert!(!gram.diagnostics.exponential_of_minus_adjoint_used);assert!(gram.matrix.iter().all(|x|x.is_finite()));
        let zero=owner.continuous_source_gram(&[0.,0.,0.,-2.],&[0.3,0.7],0.,true).unwrap();assert_eq!(zero.matrix,vec![0.;4]);
    }
}
