//! Fixed-P right Arnoldi direction proposals. The projected correction is an
//! empirical construction indicator, never a direction/population certificate.
//! All accepted outer columns still require original E/F. Unresolved proposals
//! go to restarted right GMRES and a COMPLETE complex original-M residual gate.
use std::time::Instant;

pub struct FullResidual{pub norm:f64,pub real_residual:Vec<f64>}
pub trait DirectionOperator{
    /// These two maps must remain fixed and linear for the whole call.
    fn apply_m_real(&mut self,x:&[f64])->Result<Vec<f64>,String>;
    fn precondition(&mut self,x:&[f64])->Result<Vec<f64>,String>;
    /// Must include the complete original complex M residual, including its
    /// imaginary component. real_residual is b-Re(Mx), reused at restarts.
    fn full_residual(&mut self,x:&[f64],rhs:&[f64])->Result<FullResidual,String>;
}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub enum CheckpointPolicy{Pairs,Cost}
#[derive(Clone,Copy,Debug)]pub struct AdaptiveOptions{pub direction_budget:f64,pub maximum_steps:usize,pub checkpoint_policy:CheckpointPolicy}
#[derive(Clone,Debug)]pub struct Checkpoint{
    pub steps:usize,pub full_m_relative_residual:f64,pub coefficient_relative_residual:f64,
    pub residual_identity_error:f64,pub residual_roundoff_allowance:f64,
    pub solution_norm:f64,pub projected_direction_norm:f64,pub orthogonal_roundoff_floor:f64,
    pub projected_correction_indicator:Option<f64>,pub normwise_correction_indicator:Option<f64>,
    pub indicator_decreasing:bool,pub residual_contracting:bool,pub arnoldi_breakdown:bool,
    pub normalized_direction_change:Option<f64>,pub checkpoint_ms:f64,pub lookahead_p_ms:f64,
}
#[derive(Clone,Debug)]pub struct StepTiming{
    pub step:usize,pub m_ms:f64,pub actual_p_ms:f64,pub reused_lookahead:bool,
    pub checkpoint_performed:bool,pub checkpoint_reason:&'static str,
    pub recent_m_plus_p_ms:f64,pub previous_complete_checkpoint_ms:Option<f64>,
}
#[derive(Clone,Debug)]pub struct AdaptiveReport{
    pub status:&'static str,pub reason:String,pub direction_budget:f64,pub maximum_steps:usize,
    pub inner_steps:usize,pub full_m_products:usize,pub preconditioner_applications:usize,
    pub checkpoint_preconditioner_additions:usize,pub reused_checkpoint_preconditioner_columns:usize,
    pub odd_cost_checks:usize,pub odd_cost_skips:usize,pub measured_m_ms:f64,pub measured_p_ms:f64,
    pub measured_checkpoint_ms:f64,pub elapsed_ms:f64,pub proposal_passed:bool,pub requires_strict_fallback:bool,
    pub linear_solution_confirmed:bool,pub direction_error_certified:bool,pub population_error_certified:bool,
    pub checkpoints:Vec<Checkpoint>,pub step_timing:Vec<StepTiming>,
}
#[derive(Clone,Debug)]pub struct StrictReport{
    pub iterations:usize,pub restart_cycles:usize,pub products:usize,pub preconditioner_applications:usize,
    pub complete_complex_checks:usize,pub relative_full_m_residual:f64,pub strict_residual_passed:bool,
}
#[derive(Clone,Debug)]pub struct DirectionReport{pub adaptive:AdaptiveReport,pub strict_fallback:Option<StrictReport>}
pub struct DirectionResult{pub vector:Vec<f64>,pub report:DirectionReport}
pub struct TrialResult{pub vector:Vec<f64>,pub report:AdaptiveReport}

fn ms(t:Instant)->f64{t.elapsed().as_secs_f64()*1000.}
fn norm(a:&[f64])->f64{a.iter().fold(0_f64,|v,x|v.hypot(*x))}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn checked(value:Vec<f64>,n:usize,label:&str)->Result<Vec<f64>,String>{if value.len()!=n||value.iter().any(|x|!x.is_finite()){Err(format!("Invalid {label} direction vector"))}else{Ok(value)}}
fn combination(columns:&[Vec<f64>],coefficients:&[f64],n:usize)->Vec<f64>{
    let mut result=vec![0.;n];for j in 0..coefficients.len(){for i in 0..n{result[i]+=columns[j][i]*coefficients[j];}}result
}
fn project(value:&[f64],q:&[&[f64]])->Vec<f64>{let mut out=value.to_vec();
    for _ in 0..2{let coefficients:Vec<_>=q.iter().map(|column|dot(column,&out)).collect();
        for i in 0..out.len(){let correction=q.iter().zip(&coefficients).map(|(column,&c)|column[i]*c).sum::<f64>();out[i]-=correction;}}
    out
}
struct Arnoldi{
    n:usize,cap:usize,stride:usize,beta:f64,v:Vec<Vec<f64>>,z:Vec<Vec<f64>>,mz:Vec<Vec<f64>>,
    have_z:Vec<bool>,lookahead:Vec<bool>,h:Vec<f64>,qr:Vec<f64>,cs:Vec<f64>,sn:Vec<f64>,g:Vec<f64>,
}
impl Arnoldi{
    fn new(rhs:&[f64],cap:usize)->Result<Self,String>{let n=rhs.len();let beta=norm(rhs);if beta==0.||!beta.is_finite(){return Err("Unresolved Arnoldi initial residual".into());}
        let cap=cap.min(n);let stride=cap+1;let mut v=vec![vec![0.;n];cap+1];for i in 0..n{v[0][i]=rhs[i]/beta;}let mut g=vec![0.;stride];g[0]=beta;
        Ok(Self{n,cap,stride,beta,v,z:vec![vec![0.;n];cap+1],mz:vec![vec![0.;n];cap],have_z:vec![false;cap+1],lookahead:vec![false;cap+1],
            h:vec![0.;stride*cap],qr:vec![0.;stride*cap],cs:vec![0.;cap],sn:vec![0.;cap],g})
    }
    fn prepare<S:DirectionOperator>(&mut self,system:&mut S,index:usize,checkpoint:bool)->Result<(bool,bool,f64),String>{
        if !self.have_z[index]{let began=Instant::now();self.z[index]=checked(system.precondition(&self.v[index])?,self.n,"fixed preconditioner")?;
            let elapsed=ms(began);self.have_z[index]=true;self.lookahead[index]=checkpoint;Ok((true,false,elapsed))}
        else{let reused=self.lookahead[index]&&!checkpoint;if reused{self.lookahead[index]=false;}Ok((false,reused,0.))}
    }
    fn advance<S:DirectionOperator>(&mut self,system:&mut S,column:usize)->Result<(bool,Vec<f64>,Vec<f64>,f64),String>{
        let started=Instant::now();self.mz[column]=checked(system.apply_m_real(&self.z[column])?,self.n,"complete M")?;let m_ms=ms(started);
        let mut work=self.mz[column].clone();let original=norm(&work);
        // Two COMPLETE CGS passes, not a one-pass Gram approximation.
        for _ in 0..2{let coefficients:Vec<_>=self.v[..=column].iter().map(|v|dot(v,&work)).collect();
            for i in 0..=column{self.h[i+column*self.stride]+=coefficients[i];}
            for row in 0..self.n{work[row]-=(0..=column).map(|i|self.v[i][row]*coefficients[i]).sum::<f64>();}}
        let tail=norm(&work);if !tail.is_finite(){return Err("Nonfinite right-Arnoldi tail".into());}
        let breakdown=tail<=64.*f64::EPSILON*(self.n+column+1)as f64*original.max(f64::MIN_POSITIVE);
        self.h[column+1+column*self.stride]=if breakdown{0.}else{tail};
        if !breakdown{for row in 0..self.n{self.v[column+1][row]=work[row]/tail;}}
        for i in 0..=column+1{self.qr[i+column*self.stride]=self.h[i+column*self.stride];}
        for i in 0..column{let a=self.qr[i+column*self.stride];let b=self.qr[i+1+column*self.stride];
            self.qr[i+column*self.stride]=self.cs[i]*a+self.sn[i]*b;self.qr[i+1+column*self.stride]=-self.sn[i]*a+self.cs[i]*b;}
        let a=self.qr[column+column*self.stride];let b=self.qr[column+1+column*self.stride];let length=a.hypot(b);
        if length==0.||!length.is_finite(){return Err("Unresolved right-Arnoldi least-squares rank; require original strict path".into());}
        self.cs[column]=a/length;self.sn[column]=b/length;self.qr[column+column*self.stride]=length;self.qr[column+1+column*self.stride]=0.;
        self.g[column+1]=-self.sn[column]*self.g[column];self.g[column]*=self.cs[column];let k=column+1;let mut y=self.g[..k].to_vec();
        let scale=(0..k).flat_map(|j|(0..=j).map(move|i|(i,j))).map(|(i,j)|self.qr[i+j*self.stride].abs()).fold(0_f64,f64::max);
        for i in(0..k).rev(){let diagonal=self.qr[i+i*self.stride];
            if diagonal.abs()<=32.*f64::EPSILON*(k+1)as f64*scale{return Err("Unresolved small QR diagonal; no noisy direction normalization".into());}
            for j in i+1..k{y[i]-=self.qr[i+j*self.stride]*y[j];}y[i]/=diagonal;}
        if y.iter().any(|x|!x.is_finite()){return Err("Nonfinite right-Arnoldi coefficients".into());}
        let mut residual=vec![0.;k+1];residual[0]=self.beta;for j in 0..k{for i in 0..=k{residual[i]-=self.h[i+j*self.stride]*y[j];}}
        Ok((breakdown,y,residual,m_ms))
    }
}

pub fn propose_direction<S:DirectionOperator>(system:&mut S,rhs:&[f64],outer:&[Vec<f64>],options:AdaptiveOptions)->Result<TrialResult,String>{
    let columns:Vec<_>=outer.iter().map(Vec::as_slice).collect();propose_refs(system,rhs,&columns,options)
}
fn propose_refs<S:DirectionOperator>(system:&mut S,rhs:&[f64],outer:&[&[f64]],options:AdaptiveOptions)->Result<TrialResult,String>{
    let started=Instant::now();let n=rhs.len();if n==0||options.maximum_steps==0||!options.direction_budget.is_finite()||options.direction_budget<=0.
        ||rhs.iter().any(|x|!x.is_finite())||outer.iter().any(|q|q.len()!=n||q.iter().any(|x|!x.is_finite())){return Err("Invalid adaptive direction inputs/budget/outer shape".into());}
    let mut report=AdaptiveReport{status:"strict-fallback-requested",reason:String::new(),direction_budget:options.direction_budget,maximum_steps:options.maximum_steps,
        inner_steps:0,full_m_products:0,preconditioner_applications:0,checkpoint_preconditioner_additions:0,reused_checkpoint_preconditioner_columns:0,
        odd_cost_checks:0,odd_cost_skips:0,measured_m_ms:0.,measured_p_ms:0.,measured_checkpoint_ms:0.,elapsed_ms:0.,proposal_passed:false,requires_strict_fallback:true,
        linear_solution_confirmed:false,direction_error_certified:false,population_error_certified:false,checkpoints:vec![],step_timing:vec![]};
    let mut latest=vec![0.;n];
    let result=(||->Result<(),String>{
        if norm(rhs)==0.{return Err("Zero RHS supplies no independent outer direction".into());}
        let mut a=Arnoldi::new(rhs,options.maximum_steps)?;let beta=a.beta;
        let(mut previous_indicator,mut previous_unit):(Option<f64>,Option<Vec<f64>>)=(None,None);
        let(mut previous_checkpoint_residual,mut previous_step_residual)=(beta,beta);
        let(mut last_p_ms,mut last_checkpoint_ms):(Option<f64>,Option<f64>)=(None,None);
        for column in 0..a.cap{
            let(added,reused,p_ms)=a.prepare(system,column,false)?;
            if added{report.preconditioner_applications+=1;report.measured_p_ms+=p_ms;last_p_ms=Some(p_ms);}
            if reused{report.reused_checkpoint_preconditioner_columns+=1;}
            report.full_m_products+=1;let(breakdown,y,residual_coefficients,m_ms)=a.advance(system,column)?;report.measured_m_ms+=m_ms;
            let k=column+1;report.inner_steps=k;let coefficient_norm=norm(&residual_coefficients);
            let contracting=coefficient_norm<=previous_step_residual*(1.+128.*f64::EPSILON*(n+k)as f64);previous_step_residual=coefficient_norm;
            latest=combination(&a.z,&y,n);let recent=m_ms+last_p_ms.unwrap_or(0.);
            let(screen,reason)=if k==a.cap||breakdown{(true,"maximum step or breakdown requires existing screen")}
                else if k%2==0{(true,"normal even-step screen")}
                else if options.checkpoint_policy==CheckpointPolicy::Pairs{(false,"paired checkpoint policy")}
                else if k<3||last_checkpoint_ms.is_none()||last_p_ms.is_none(){(false,"first collect normal k2 cost")}
                else{let selected=recent.is_finite()&&recent>0.&&last_checkpoint_ms.unwrap()<=recent;
                    if selected{report.odd_cost_checks+=1;}else{report.odd_cost_skips+=1;}(selected,if selected{"last complete checkpoint costs no more than recent M+P"}else{"checkpoint costs more than recent M+P"})};
            report.step_timing.push(StepTiming{step:k,m_ms,actual_p_ms:p_ms,reused_lookahead:reused,checkpoint_performed:screen,checkpoint_reason:reason,
                recent_m_plus_p_ms:recent,previous_complete_checkpoint_ms:last_checkpoint_ms});
            if !screen{continue;}
            let began=Instant::now();let represented_image=combination(&a.mz,&y,n);let residual:Vec<_>=rhs.iter().zip(represented_image).map(|(b,m)|b-m).collect();
            let residual_norm=norm(&residual);let represented=combination(&a.v,&residual_coefficients,n);
            let identity_error=norm(&residual.iter().zip(represented).map(|(a,b)|a-b).collect::<Vec<_>>());
            let mz_norm=a.mz[..k].iter().fold(0_f64,|v,x|v.hypot(norm(x)));let allowance=256.*f64::EPSILON*(n+k+1)as f64*(beta+mz_norm*norm(&y));
            if identity_error>allowance{return Err("Stored original M residual and Arnoldi relation disagree beyond roundoff".into());}
            let mut lookahead_ms=0.;if !breakdown{let(added,_,elapsed)=a.prepare(system,k,true)?;lookahead_ms=elapsed;
                if added{report.preconditioner_applications+=1;report.checkpoint_preconditioner_additions+=1;report.measured_p_ms+=elapsed;last_p_ms=Some(elapsed);}}
            let correction=combination(&a.z,&residual_coefficients[..if breakdown{k}else{k+1}],n);
            let direction=project(&latest,outer);let projected_correction=project(&correction,outer);let xnorm=norm(&latest);let dnorm=norm(&direction);
            let floor=128.*f64::EPSILON*(n+outer.len()+k+1)as f64*xnorm.max(f64::MIN_POSITIVE);let resolved=dnorm>floor&&xnorm>0.;
            let indicator=resolved.then(||norm(&projected_correction)/dnorm);let normwise=resolved.then(||norm(&correction)/dnorm);
            let slack=128.*f64::EPSILON*(n+k+outer.len()+1)as f64;
            let decreasing=previous_indicator.is_none()||indicator.is_some_and(|value|value<=previous_indicator.unwrap()*(1.+slack));
            let residual_contracting=contracting&&residual_norm<=previous_checkpoint_residual*(1.+slack);
            let unit=resolved.then(||direction.iter().map(|x|x/dnorm).collect::<Vec<_>>());
            let change=unit.as_ref().zip(previous_unit.as_ref()).map(|(a,b)|{
                norm(&a.iter().zip(b).map(|(x,y)|x-y).collect::<Vec<_>>()).min(norm(&a.iter().zip(b).map(|(x,y)|x+y).collect::<Vec<_>>()))});
            let elapsed=ms(began);last_checkpoint_ms=Some(elapsed);report.measured_checkpoint_ms+=elapsed;
            report.checkpoints.push(Checkpoint{steps:k,full_m_relative_residual:residual_norm/beta,coefficient_relative_residual:coefficient_norm/beta,
                residual_identity_error:identity_error,residual_roundoff_allowance:allowance,solution_norm:xnorm,projected_direction_norm:dnorm,
                orthogonal_roundoff_floor:floor,projected_correction_indicator:indicator,normwise_correction_indicator:normwise,
                indicator_decreasing:decreasing,residual_contracting,arnoldi_breakdown:breakdown,normalized_direction_change:change,checkpoint_ms:elapsed,lookahead_p_ms:lookahead_ms});
            if indicator.is_some_and(|v|v.is_finite()&&v<=options.direction_budget)&&residual_contracting&&decreasing{
                report.proposal_passed=true;report.requires_strict_fallback=false;report.status="trial-direction-proposed";
                report.reason="Projected correction screen passed; original outer E/F and population/domain checks remain required".into();return Ok(());}
            if breakdown{return Err("Arnoldi breakdown without a passing resolved-direction screen".into());}
            previous_indicator=indicator;previous_checkpoint_residual=residual_norm;previous_unit=unit;
        }
        Err("Maximum proposal steps exhausted before the direction screen passed".into())
    })();
    if let Err(reason)=result{report.reason=reason;report.requires_strict_fallback=true;report.proposal_passed=false;report.status="strict-fallback-requested";}
    report.elapsed_ms=ms(started);Ok(TrialResult{vector:latest,report})
}

pub fn strict_direction<S:DirectionOperator>(system:&mut S,rhs:&[f64])->Result<(Vec<f64>,StrictReport),String>{
    let n=rhs.len();let beta0=norm(rhs);if n==0||rhs.iter().any(|x|!x.is_finite()){return Err("Invalid strict original-M RHS".into());}
    let mut report=StrictReport{iterations:0,restart_cycles:0,products:0,preconditioner_applications:0,complete_complex_checks:0,relative_full_m_residual:f64::INFINITY,strict_residual_passed:false};
    let mut x=vec![0.;n];let mut residual=rhs.to_vec();
    if beta0==0.{let checked=system.full_residual(&x,rhs)?;report.complete_complex_checks+=1;
        if checked.norm==0.{report.relative_full_m_residual=0.;report.strict_residual_passed=true;return Ok((x,report));}return Err("Zero RHS failed original complex M check".into());}
    let target=(1e-9*beta0).max(1e-13);
    for _ in 0..4{
        report.restart_cycles+=1;let mut a=Arnoldi::new(&residual,24)?;let base=x.clone();let mut completed=false;
        for column in 0..a.cap{let(added,_,_)=a.prepare(system,column,false)?;if added{report.preconditioner_applications+=1;}
            report.products+=1;report.iterations+=1;let(breakdown,y,r,_) =a.advance(system,column)?;let k=column+1;
            let check=norm(&r)<=target||breakdown||k==a.cap;if !check{continue;}
            let correction=combination(&a.z,&y,n);let candidate:Vec<_>=base.iter().zip(correction).map(|(a,b)|a+b).collect();
            let full=system.full_residual(&candidate,rhs)?;report.complete_complex_checks+=1;
            if !full.norm.is_finite()||full.norm<0.{return Err("Nonfinite original complex M residual gate".into());}
            if full.real_residual.len()!=n||full.real_residual.iter().any(|x|!x.is_finite()){return Err("Invalid original complex-gate real residual vector".into());}
            report.relative_full_m_residual=full.norm/beta0;
            if report.relative_full_m_residual<=3e-9{report.strict_residual_passed=true;return Ok((candidate,report));}
            if breakdown||k==a.cap{x=candidate;residual=checked(full.real_residual,n,"original restart residual")?;completed=true;break;}
        }
        if !completed{return Err("Strict right GMRES ended without an original residual/restart state".into());}
    }
    Err(format!("Original complete complex M inverse unresolved after four24-step cycles: {}",report.relative_full_m_residual))
}
pub fn solve_direction<S:DirectionOperator>(system:&mut S,rhs:&[f64],outer:&[Vec<f64>],options:AdaptiveOptions)->Result<DirectionResult,String>{
    let trial=propose_direction(system,rhs,outer,options)?;
    finish_direction(system,rhs,trial)
}
/// Borrow append-only outer columns in place; do not clone all old Q for each
/// direction. Only a small array of slice views is built.
pub fn solve_direction_flat<S:DirectionOperator>(system:&mut S,rhs:&[f64],outer:&[f64],columns:usize,options:AdaptiveOptions)->Result<DirectionResult,String>{
    let n=rhs.len();if n==0||n.checked_mul(columns)!=Some(outer.len()){return Err("Invalid borrowed column-major outer basis".into());}
    let views:Vec<_>=outer.chunks_exact(n).collect();let trial=propose_refs(system,rhs,&views,options)?;finish_direction(system,rhs,trial)
}
fn finish_direction<S:DirectionOperator>(system:&mut S,rhs:&[f64],trial:TrialResult)->Result<DirectionResult,String>{
    if trial.report.requires_strict_fallback{let(vector,strict)=strict_direction(system,rhs)?;
        Ok(DirectionResult{vector,report:DirectionReport{adaptive:trial.report,strict_fallback:Some(strict)}})}
    else{Ok(DirectionResult{vector:trial.vector,report:DirectionReport{adaptive:trial.report,strict_fallback:None}})}
}

#[cfg(test)]
mod tests{
    use super::*;
    struct Diagonal{values:Vec<f64>,p_calls:usize,m_calls:usize,extra_imaginary:f64}
    impl DirectionOperator for Diagonal{
        fn apply_m_real(&mut self,x:&[f64])->Result<Vec<f64>,String>{self.m_calls+=1;Ok(self.values.iter().zip(x).map(|(a,b)|a*b).collect())}
        fn precondition(&mut self,x:&[f64])->Result<Vec<f64>,String>{self.p_calls+=1;Ok(x.to_vec())}
        fn full_residual(&mut self,x:&[f64],rhs:&[f64])->Result<FullResidual,String>{let image=self.apply_m_real(x)?;let real:Vec<_>=rhs.iter().zip(image).map(|(a,b)|a-b).collect();Ok(FullResidual{norm:norm(&real).hypot(self.extra_imaginary),real_residual:real})}
    }
    #[test]
    fn preparation_adaptive_direction_reuses_checkpoint_p_and_never_certifies_trial(){
        let mut system=Diagonal{values:vec![0.7,1.1,1.9,2.7,4.3,6.1],p_calls:0,m_calls:0,extra_imaginary:0.};let rhs=vec![1.;6];
        let trial=propose_direction(&mut system,&rhs,&[],AdaptiveOptions{direction_budget:1e-16,maximum_steps:4,checkpoint_policy:CheckpointPolicy::Pairs}).unwrap();
        assert!(trial.report.requires_strict_fallback);assert_eq!(trial.report.full_m_products,4);assert_eq!(system.m_calls,4);
        assert_eq!(trial.report.checkpoint_preconditioner_additions,2);assert_eq!(trial.report.reused_checkpoint_preconditioner_columns,1);
        assert_eq!(system.p_calls,5);assert!(!trial.report.linear_solution_confirmed&&!trial.report.direction_error_certified&&!trial.report.population_error_certified);
        for check in &trial.report.checkpoints{assert!(check.residual_identity_error<=check.residual_roundoff_allowance);assert!(check.projected_correction_indicator.is_some());}
    }
    #[test]
    fn preparation_adaptive_direction_unresolved_outer_component_uses_full_strict_gate(){
        let mut system=Diagonal{values:vec![1.;4],p_calls:0,m_calls:0,extra_imaginary:0.};let rhs=vec![1.,0.,0.,0.];let outer=vec![rhs.clone()];
        let result=solve_direction(&mut system,&rhs,&outer,AdaptiveOptions{direction_budget:1e-3,maximum_steps:4,checkpoint_policy:CheckpointPolicy::Cost}).unwrap();
        assert!(result.report.adaptive.requires_strict_fallback);assert!(!result.report.adaptive.proposal_passed);
        assert!(result.report.strict_fallback.as_ref().unwrap().strict_residual_passed);assert_eq!(result.vector,rhs);
        let mut wrong=Diagonal{values:vec![1.;4],p_calls:0,m_calls:0,extra_imaginary:1e-4};assert!(strict_direction(&mut wrong,&rhs).is_err());
    }
    #[test]
    fn preparation_adaptive_direction_cost_schedule_preserves_indicators_and_complete_sources(){
        let rhs=vec![1.;6];let options=AdaptiveOptions{direction_budget:1e-7,maximum_steps:6,checkpoint_policy:CheckpointPolicy::Cost};
        let mut system=Diagonal{values:vec![0.9,1.05,1.2,1.5,1.9,2.4],p_calls:0,m_calls:0,extra_imaginary:0.};
        let result=solve_direction(&mut system,&rhs,&[],options).unwrap();
        if result.report.adaptive.proposal_passed{assert!(result.report.strict_fallback.is_none());assert!(!result.report.adaptive.linear_solution_confirmed);}
        else{assert!(result.report.strict_fallback.as_ref().unwrap().strict_residual_passed);}
        assert!(result.report.adaptive.step_timing.iter().any(|s|s.step==2&&s.checkpoint_performed));
        assert!(result.vector.iter().all(|x|x.is_finite()));
    }
}
