//! Request-local population WORK/empirical stopping, never a physical bound.
//! Default: the established two-expansion + adjacent-domain controller.
//! Optional: pointwise basis/outer output probe, explicitly a separate mode.
//! No physical coefficient, epsilon, raw population or prior answer is changed.
use super::*;
use super::runtime_core::Candidate;

pub const DEFAULT_POPULATION_TOLERANCE:f64=0.005;
pub const MAX_POPULATION_TOLERANCE:f64=0.02;
#[derive(Clone,Debug)]pub struct RequestedGroup{pub name:String,pub indices:Vec<usize>}
#[derive(Clone,Copy,Debug)]pub struct ControlOptions{
    pub epsilon:f64,pub reserve_fraction:f64,pub maximum_no_improvement:usize,pub enable_pointwise_outer:bool,
}
impl Default for ControlOptions{fn default()->Self{Self{epsilon:DEFAULT_POPULATION_TOLERANCE,reserve_fraction:0.1,maximum_no_improvement:3,enable_pointwise_outer:false}}}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub enum Action{RefineBasis,ExpandDomain,EstimateAccepted,RepairNumerics,Unresolved}
impl Action{pub fn as_str(self)->&'static str{match self{Self::RefineBasis=>"refineBasis",Self::ExpandDomain=>"expandDomain",Self::EstimateAccepted=>"estimateAccepted",Self::RepairNumerics=>"repairNumerics",Self::Unresolved=>"unresolved"}}}
#[derive(Clone,Copy,Debug,PartialEq,Eq)]pub enum ControlMode{MatureRefinement,PointwiseOuter}
#[derive(Clone,Debug)]pub struct WorkDecision{
    pub action:Action,pub mode:ControlMode,pub epsilon:f64,pub effective_stop_budget:f64,pub reserve:f64,
    pub basis_indicator:Option<f64>,pub domain_indicator:Option<f64>,pub joint_indicator:Option<f64>,
    pub remaining_budget:f64,pub local_basis_screened:bool,pub genuine_basis_comparisons:usize,
    pub nonmonotone:bool,pub no_improvement_streak:usize,pub exact_fourier_truncation_zero:bool,
    pub reason:&'static str,pub empirical_only:bool,pub rigorous_error_bound:bool,pub epsilon_changed:bool,
}
#[derive(Clone,Debug)]pub struct OutputCheck{
    scope:Arc<()>,pub finite:bool,pub numerical_invariants_passed:bool,pub probability_ranges_passed:bool,
    pub trace_error:f64,pub imaginary_error:f64,pub initial_population_error:f64,
    pub minimum_population:f64,pub maximum_population:f64,pub group_ranges:Vec<(f64,f64)>,
    pub range_budget:f64,pub density_positivity_validated:bool,
}
impl OutputCheck{pub fn eligible(&self)->bool{self.finite&&self.numerical_invariants_passed&&self.probability_ranges_passed}}
#[derive(Clone,Debug)]pub struct OutputDelta{scope:Arc<()>,pub state:f64,pub groups:Vec<f64>,pub maximum:f64}
#[derive(Clone,Copy,Debug)]pub struct FrameContext{
    pub generation:u64,pub domain_order:usize,pub new_directions:usize,pub zero_lift_only:bool,
    pub retains_previous_trial_space:bool,pub all_current_residual_source_columns:bool,
}
#[derive(Clone,Debug)]pub struct ControlFrame{
    scope:Arc<()>,pub generation:u64,pub domain_order:usize,pub domain_modes:Vec<Charge>,pub basis_columns:usize,
    pub workspace_identity:usize,pub basis_revision:usize,pub construction_epoch:usize,
    pub new_directions:usize,pub zero_lift_only:bool,pub retains_previous_trial_space:bool,
    pub complete_generation:bool,pub original_ef_complete:bool,pub initial_state_retained:bool,
    pub all_current_residual_source_columns:bool,
}
#[derive(Clone,Debug)]pub struct ExactZeroProof{scope:Arc<()>,domain:Vec<Charge>,pub proven:bool,pub reason:&'static str}
pub struct WorkEvidence<'a>{
    pub frames:&'a[ControlFrame],pub changes:&'a[OutputDelta],pub current_output:&'a OutputCheck,
    pub domain_change:Option<&'a OutputDelta>,pub previous_domain:Option<&'a ControlFrame>,
    pub previous_domain_basis_passed:bool,pub exact_zero:Option<&'a ExactZeroProof>,
    pub basis_cost_ms:Option<f64>,pub expansion_cost_ms:Option<f64>,
}
pub struct PointwiseDecision{
    pub decision:WorkDecision,pub state_indicator:Vec<f64>,pub group_indicators:Vec<Vec<f64>>,
    pub signed_basis_change:Vec<f64>,pub signed_outer_change:Vec<f64>,
    pub sum_of_separate_global_maxima:f64,pub all_halo_error_control:bool,
}
pub struct PopulationControl{
    owner:Arc<PrimitivePreparation>,scope:Arc<()>,options:ControlOptions,times:Vec<f64>,initial:Vec<f64>,groups:Vec<RequestedGroup>,n:usize,
}
fn nonnegative(x:f64)->bool{x.is_finite()&&x>=0.}
fn same_times(a:&[f64],b:&[f64])->bool{a.len()==b.len()&&a.iter().zip(b).all(|(a,b)|a.to_bits()==b.to_bits())}
fn ulp(x:f64)->f64{if x==0.{f64::from_bits(1)}else if x==f64::MAX{x-f64::from_bits(x.to_bits()-1)}else{f64::from_bits(x.to_bits()+1)-x}}
fn maximum(a:&[f64])->f64{a.iter().map(|x|x.abs()).fold(0_f64,f64::max)}
fn hs(a:&[f64])->f64{a.iter().fold(0_f64,|v,&x|v.hypot(x))}

impl PopulationControl{
    pub fn new(prepared:&OwnedPrepared,options:ControlOptions,times:&[f64],initial_populations:&[f64],groups:Vec<RequestedGroup>)->Result<Self,String>{
        if !options.epsilon.is_finite()||options.epsilon<=0.||options.epsilon>MAX_POPULATION_TOLERANCE
            ||!options.reserve_fraction.is_finite()||options.reserve_fraction<=0.||options.reserve_fraction>=1.||options.maximum_no_improvement<2{
            return Err("Population epsilon must satisfy0<epsilon<=0.02; invalid work-control options".into());}
        let n=prepared.primitives.hilbert.source.n;
        if n>32||times.is_empty()||times[0]!=0.||times.iter().any(|x|!x.is_finite()||*x<0.)||times.windows(2).any(|t|t[0]>t[1])
            ||initial_populations.len()!=n||initial_populations.iter().any(|x|!x.is_finite()){
            return Err("Complete finite state outputs, initial population and exact grid from zero required".into());}
        let mut names=BTreeSet::new();for group in &groups{let ids:BTreeSet<_>=group.indices.iter().copied().collect();
            if group.name.is_empty()||!names.insert(group.name.clone())||ids.is_empty()||ids.len()!=group.indices.len()||ids.iter().any(|&i|i>=n){return Err("Requested groups need unique names and valid distinct indices; overlap BETWEEN groups is allowed".into());}}
        Ok(Self{owner:Arc::clone(&prepared.primitives),scope:Arc::new(()),options,times:times.to_vec(),initial:initial_populations.to_vec(),groups,n})
    }
    pub fn epsilon(&self)->f64{self.options.epsilon}
    pub fn groups(&self)->&[RequestedGroup]{&self.groups}
    pub fn times(&self)->&[f64]{&self.times}
    fn owner(&self,prepared:&OwnedPrepared)->Result<(),String>{if Arc::ptr_eq(&self.owner,&prepared.primitives){Ok(())}else{Err("Population control cannot cross current-request Model owners".into())}}
    fn scope(&self,scope:&Arc<()>)->bool{Arc::ptr_eq(&self.scope,scope)}
    fn decision(&self,mode:ControlMode)->WorkDecision{let budget=(1.-self.options.reserve_fraction)*self.options.epsilon;
        WorkDecision{action:Action::Unresolved,mode,epsilon:self.options.epsilon,effective_stop_budget:budget,reserve:self.options.reserve_fraction*self.options.epsilon,
            basis_indicator:None,domain_indicator:None,joint_indicator:None,remaining_budget:budget,local_basis_screened:false,genuine_basis_comparisons:0,
            nonmonotone:false,no_improvement_streak:0,exact_fourier_truncation_zero:false,reason:"No eligible evidence",empirical_only:true,rigorous_error_bound:false,epsilon_changed:false}}
    pub fn assess_output(&self,populations:&[C],external_numerical_invariants:bool)->Result<OutputCheck,String>{
        if populations.len()!=product(&[self.n,self.times.len()])?{return Err("Population output shape changed".into());}
        let finite=populations.iter().all(|z|z.finite());let mut trace=0_f64;let mut imaginary=0_f64;let mut initial=0_f64;
        let mut low=f64::INFINITY;let mut high=f64::NEG_INFINITY;let mut ranges=vec![(f64::INFINITY,f64::NEG_INFINITY);self.groups.len()];
        for(t,row)in populations.chunks_exact(self.n).enumerate(){trace=trace.max((row.iter().map(|x|x.re).sum::<f64>()-1.).abs());
            for(i,z)in row.iter().enumerate(){imaginary=imaginary.max(z.im.abs());low=low.min(z.re);high=high.max(z.re);if t==0{initial=initial.max((z.re-self.initial[i]).abs());}}
            for(j,group)in self.groups.iter().enumerate(){let sum=group.indices.iter().map(|&i|row[i].re).sum::<f64>();ranges[j].0=ranges[j].0.min(sum);ranges[j].1=ranges[j].1.max(sum);}
        }
        let budget=self.options.epsilon/4.;let numerical=finite&&external_numerical_invariants&&trace<=1e-7&&imaginary<=1e-9&&initial<=1e-10;
        let probability=finite&&low>=-budget&&high<=1.+budget&&ranges.iter().all(|&(a,b)|a>=-budget&&b<=1.+budget);
        Ok(OutputCheck{scope:Arc::clone(&self.scope),finite,numerical_invariants_passed:numerical,probability_ranges_passed:probability,
            trace_error:trace,imaginary_error:imaginary,initial_population_error:initial,minimum_population:low,maximum_population:high,group_ranges:ranges,range_budget:budget,density_positivity_validated:false})
    }
    pub fn assess_candidate(&self,prepared:&OwnedPrepared,candidate:&Candidate)->Result<OutputCheck,String>{
        self.owner(prepared)?;if !candidate.belongs_to(prepared)||!same_times(&candidate.times,&self.times){return Err("Candidate owner/time grid differs from population control".into());}
        let d=&candidate.diagnostics;let external=d.orthogonality_error_hs<=1e-10&&d.original_initial_projection_error_hs<=1e-10
            &&d.maximum_entered_reality_ratio<=1.&&d.mass_trace_identity_error<=1e-11&&d.noncentral_trace_error<=1e-10
            &&d.pencil_trace_identity_error<=4096.*f64::EPSILON*hs(&candidate.fq).max(1.);
        self.assess_output(&candidate.populations,external)
    }
    pub fn difference(&self,before:&[C],after:&[C])->Result<OutputDelta,String>{
        if before.len()!=product(&[self.n,self.times.len()])?||after.len()!=before.len()||before.iter().chain(after).any(|z|!z.finite()){
            return Err("Finite same-grid complete populations required for output comparison".into());}
        let mut state=0_f64;let mut groups=vec![0_f64;self.groups.len()];
        for(a,b)in before.chunks_exact(self.n).zip(after.chunks_exact(self.n)){
            for(i,j)in a.iter().zip(b){state=state.max((j.re-i.re).abs());}
            for(k,group)in self.groups.iter().enumerate(){let signed=group.indices.iter().map(|&i|b[i].re-a[i].re).sum::<f64>();groups[k]=groups[k].max(signed.abs());}
        }let maximum=state.max(groups.iter().copied().fold(0_f64,f64::max));
        if !maximum.is_finite(){return Err("Population comparison overflow".into());}
        Ok(OutputDelta{scope:Arc::clone(&self.scope),state,groups,maximum})
    }
    pub fn frame_from_candidate(&self,prepared:&OwnedPrepared,candidate:&Candidate,context:FrameContext)->Result<ControlFrame,String>{
        self.owner(prepared)?;if !candidate.belongs_to(prepared)||!same_times(&candidate.times,&self.times){return Err("Frame is not from the current Model/grid".into());}
        if context.generation==0||context.domain_order==0||context.new_directions>candidate.dimension{return Err("Invalid actual generation metadata".into());}
        let count=product(&[candidate.rows,candidate.dimension])?;
        let complete=candidate.dimension>0&&candidate.q.len()==count&&candidate.eq.len()==count&&candidate.fq.len()==count&&candidate.diagnostics.entered_columns==candidate.dimension;
        Ok(ControlFrame{scope:Arc::clone(&self.scope),generation:context.generation,domain_order:context.domain_order,domain_modes:candidate.modes.clone(),basis_columns:candidate.dimension,
            workspace_identity:candidate.workspace_identity_key(),basis_revision:candidate.basis_revision(),construction_epoch:candidate.diagnostics.frontier_epoch,
            new_directions:context.new_directions,zero_lift_only:context.zero_lift_only,retains_previous_trial_space:context.retains_previous_trial_space,
            complete_generation:candidate.diagnostics.complete_frontier_generation,original_ef_complete:complete,
            initial_state_retained:candidate.diagnostics.original_initial_projection_error_hs<=1e-10,
            all_current_residual_source_columns:context.all_current_residual_source_columns})
    }
    pub fn prove_exact_zero_fourier(&self,prepared:&OwnedPrepared,domain:&[Charge])->Result<ExactZeroProof,String>{
        self.owner(prepared)?;let p=&prepared.primitives;let source=&p.hilbert.source;
        let zero=vec![0;source.phases.axes.len()];let constant=|family:&FourierMatrices|family.keys().all(|q|*q==zero);
        let proven=source.phases.axes.is_empty()&&source.phases.drive_indices.is_empty()&&domain==[zero.clone()]
            &&constant(&source.hamiltonian)&&source.collapse.iter().all(constant)&&constant(&p.s)&&constant(&p.k_hat)
            &&constant(&p.hilbert.radiation.gamma)&&constant(&p.hilbert.radiation.recycling);
        Ok(ExactZeroProof{scope:Arc::clone(&self.scope),domain:domain.to_vec(),proven,
            reason:if proven{"Complete original coefficients/coordinates/initial extension have only zero Fourier charge; basis/time/output checks remain"}else{"No exact zero-Fourier support proof"}})
    }
    fn valid_frame(&self,f:&ControlFrame)->bool{self.scope(&f.scope)&&f.generation>0&&f.domain_order>0&&f.basis_columns>0&&!f.domain_modes.is_empty()}
    fn expanded_pair(&self,a:&ControlFrame,b:&ControlFrame)->bool{
        self.valid_frame(a)&&self.valid_frame(b)&&b.generation>a.generation&&b.basis_columns>a.basis_columns
            &&b.new_directions==b.basis_columns-a.basis_columns&&!b.zero_lift_only&&b.retains_previous_trial_space
    }
    pub fn decide_mature(&self,input:WorkEvidence<'_>)->WorkDecision{
        let mut result=self.decision(ControlMode::MatureRefinement);
        macro_rules! done{($action:ident,$why:expr)=>{{result.action=Action::$action;result.reason=$why;return result;}}}
        if !self.scope(&input.current_output.scope)||input.changes.iter().any(|d|!self.scope(&d.scope))||input.domain_change.is_some_and(|d|!self.scope(&d.scope)){
            done!(Unresolved,"Evidence belongs to another request, group/grid scope or tolerance");}
        if input.changes.iter().any(|d|!nonnegative(d.maximum))||input.domain_change.is_some_and(|d|!nonnegative(d.maximum)){
            done!(RepairNumerics,"Population changes must be finite nonnegative values");}
        if !input.current_output.numerical_invariants_passed{done!(RepairNumerics,"Fixed numerical invariants failed; epsilon cannot relax them");}
        if input.frames.len()!=input.changes.len()+1||input.frames.is_empty(){done!(Unresolved,"Round zero plus every genuine expansion must be present");}
        if input.frames.iter().any(|f|!self.valid_frame(f)){done!(Unresolved,"Invalid or foreign generation metadata");}
        if input.frames.iter().any(|f|!f.complete_generation||!f.original_ef_complete||!f.initial_state_retained){done!(RefineBasis,"Complete original images/initial/generation before comparing outputs");}
        let current=input.frames.last().unwrap();
        if input.frames.iter().any(|f|f.domain_order!=current.domain_order||f.domain_modes!=current.domain_modes||f.construction_epoch!=current.construction_epoch){
            done!(Unresolved,"Basis comparison window crossed a domain or construction epoch");}
        if input.frames.windows(2).any(|f|!self.expanded_pair(&f[0],&f[1])||f[0].workspace_identity!=f[1].workspace_identity||f[1].basis_revision<=f[0].basis_revision){
            done!(Unresolved,"Same-space, zero-lift, disconnected or smaller basis is not new evidence");}
        let exact=if let Some(proof)=input.exact_zero{
            if !self.scope(&proof.scope)||proof.domain!=current.domain_modes{done!(Unresolved,"Exact-domain proof belongs to another request/domain");}proof.proven
        }else{false};result.exact_fourier_truncation_zero=exact;
        let mut domain=input.domain_change.map(|d|d.maximum);
        if exact{if domain.is_some_and(|d|d!=0.){done!(Unresolved,"Exact Fourier proof conflicts with nonzero domain indicator");}domain=Some(0.);}
        result.domain_indicator=domain;result.genuine_basis_comparisons=input.changes.len();
        if input.changes.len()<2{done!(RefineBasis,"Two genuine consecutive basis expansions required; round zero is not evidence");}
        let changes:Vec<_>=input.changes.iter().map(|d|d.maximum).collect();let len=changes.len();let basis=2.*changes[len-2].max(changes[len-1]);
        let budget=result.effective_stop_budget;result.basis_indicator=Some(basis);result.remaining_budget=budget-basis;result.local_basis_screened=basis<=budget;
        result.nonmonotone=changes[len-1]>changes[len-2]+32.*ulp(changes[len-1].max(changes[len-2]));
        for pair in changes.windows(2).rev(){if pair[1]>=pair[0]-32.*ulp(pair[0].max(pair[1])){result.no_improvement_streak+=1;}else{break;}}
        let stalled=result.no_improvement_streak>=self.options.maximum_no_improvement;
        if !input.current_output.probability_ranges_passed{if stalled{done!(Unresolved,"Probability-range compatibility failed after stalled work");}done!(RefineBasis,"Output probability compatibility is independent of small refinement differences");}
        let domain=match domain{Some(v)=>v,None=>{if basis<=budget{done!(ExpandDomain,"First-domain basis locally screened; obtain an adjacent-domain comparison");}
            if stalled{done!(Unresolved,"First-domain indicators stagnate above budget");}done!(RefineBasis,"Resolve the basis before paying for another domain");}};
        if !exact{
            if !input.previous_domain_basis_passed{done!(Unresolved,"Previous-domain curve lacks mature basis qualification");}
            let previous=match input.previous_domain{Some(v)=>v,None=>done!(Unresolved,"Adjacent previous-domain frame is required")};
            if !self.valid_frame(previous)||previous.domain_order+1!=current.domain_order{done!(Unresolved,"Domain comparison must connect adjacent current-request domains");}
            let old:Modes=previous.domain_modes.iter().cloned().collect();let new:Modes=current.domain_modes.iter().cloned().collect();
            if !old.is_subset(&new)||old.len()>=new.len(){done!(Unresolved,"A same-domain lift cannot establish domain accuracy");}
        }
        let joint=basis+domain;result.joint_indicator=Some(joint);result.remaining_budget=budget-joint;
        if joint<=budget{done!(EstimateAccepted,"Two genuine expansions plus adjacent-domain change fit the empirical total budget");}
        if stalled{if basis<=budget&&domain>0.{done!(ExpandDomain,"Basis refinement stalled; inspect a new domain without accepting a plateau");}
            done!(Unresolved,"Basis indicators stagnated above the remaining budget");}
        if basis>budget{done!(RefineBasis,"Basis indicator alone exceeds the total budget");}
        let bc=input.basis_cost_ms.filter(|&x|nonnegative(x));let dc=input.expansion_cost_ms.filter(|&x|nonnegative(x));
        if domain>=budget{
            if basis>0.&&domain-budget<basis&&bc.zip(dc).is_some_and(|(b,d)|b<d){done!(RefineBasis,"Ambiguous domain excess; a measured cheaper basis block can clarify it");}
            done!(ExpandDomain,"Domain change consumes the total budget");}
        if bc.is_some_and(|b|dc.is_none_or(|d|b<=d)){done!(RefineBasis,"Another measured basis block is cheaper");}
        if bc.zip(dc).is_some_and(|(b,d)|d<b){done!(ExpandDomain,"Measured domain expansion is cheaper");}
        done!(RefineBasis,"Refine basis without extrapolating an unobserved convergence rate");
    }
    pub fn decide_pointwise(&self,frames:[&ControlFrame;3],populations:[&[C];3],checks:[&OutputCheck;3])->Result<PointwiseDecision,String>{
        let mut decision=self.decision(ControlMode::PointwiseOuter);
        if !self.options.enable_pointwise_outer{return Err("Pointwise outer path is optional and disabled by default; use mature controller".into());}
        if frames.iter().any(|f|!self.valid_frame(f)||f.zero_lift_only||!f.complete_generation||!f.original_ef_complete||!f.initial_state_retained)
            ||checks.iter().any(|c|!self.scope(&c.scope)||!c.numerical_invariants_passed){return Err("Incomplete, zero-lift or foreign pointwise evidence".into());}
        if !self.expanded_pair(frames[0],frames[1])||!self.expanded_pair(frames[1],frames[2])
            ||frames[0].domain_order!=frames[1].domain_order||frames[0].domain_modes!=frames[1].domain_modes
            ||frames[0].workspace_identity!=frames[1].workspace_identity||frames[0].construction_epoch!=frames[1].construction_epoch
            ||!frames[2].all_current_residual_source_columns||frames[2].domain_order<=frames[1].domain_order{
            return Err("Pointwise probe requires a genuine basis expansion then a full-source new-domain correction".into());}
        let a:Modes=frames[1].domain_modes.iter().cloned().collect();let b:Modes=frames[2].domain_modes.iter().cloned().collect();
        if !a.is_subset(&b)||a.len()>=b.len(){return Err("Outer pointwise probe did not add Fourier modes".into());}
        let first=self.difference(populations[0],populations[1])?;let second=self.difference(populations[1],populations[2])?;
        let d1:Vec<_>=populations[1].iter().zip(populations[0]).map(|(a,b)|a.re-b.re).collect();
        let d2:Vec<_>=populations[2].iter().zip(populations[1]).map(|(a,b)|a.re-b.re).collect();
        let state:Vec<_>=d1.iter().zip(&d2).map(|(a,b)|a.abs()+b.abs()).collect();let mut groups=vec![vec![0.;self.times.len()];self.groups.len()];
        for(t,(a,b))in d1.chunks_exact(self.n).zip(d2.chunks_exact(self.n)).enumerate(){for(j,group)in self.groups.iter().enumerate(){
            groups[j][t]=group.indices.iter().map(|&i|a[i]).sum::<f64>().abs()+group.indices.iter().map(|&i|b[i]).sum::<f64>().abs();}}
        let joint=maximum(&state).max(groups.iter().map(|g|maximum(g)).fold(0_f64,f64::max));
        decision.basis_indicator=Some(first.maximum);decision.domain_indicator=Some(second.maximum);decision.joint_indicator=Some(joint);decision.remaining_budget=decision.effective_stop_budget-joint;
        decision.genuine_basis_comparisons=2;decision.local_basis_screened=first.maximum<=decision.effective_stop_budget;
        if !checks[1].probability_ranges_passed||!checks[2].probability_ranges_passed{decision.action=Action::RefineBasis;decision.reason="Fixed numerical checks passed but current/probe probability compatibility did not";}
        else if joint<=decision.effective_stop_budget{decision.action=Action::EstimateAccepted;decision.reason="Pointwise two-source empirical estimate fits budget; infinite halo remains unverified";}
        else{decision.action=Action::RefineBasis;decision.reason="Optional pointwise estimate exceeds budget; continue genuine work or mature path";}
        Ok(PointwiseDecision{decision,state_indicator:state,group_indicators:groups,signed_basis_change:d1,signed_outer_change:d2,
            sum_of_separate_global_maxima:first.maximum+second.maximum,all_halo_error_control:false})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn groups()->Vec<RequestedGroup>{vec![RequestedGroup{name:"F0".into(),indices:vec![0,1]},RequestedGroup{name:"F1".into(),indices:vec![2,3]},RequestedGroup{name:"all".into(),indices:vec![0,1,2,3]}]}
    fn control(owner:&OwnedPrepared,epsilon:f64,early:bool)->PopulationControl{PopulationControl::new(owner,ControlOptions{epsilon,enable_pointwise_outer:early,..Default::default()},&[0.,1.],&[0.25;4],groups()).unwrap()}
    fn frame(c:&PopulationControl,generation:u64,basis:usize,order:usize,modes:Vec<Charge>,new:usize)->ControlFrame{
        ControlFrame{scope:Arc::clone(&c.scope),generation,domain_order:order,domain_modes:modes,basis_columns:basis,workspace_identity:order,basis_revision:basis,construction_epoch:0,
            new_directions:new,zero_lift_only:false,retains_previous_trial_space:true,complete_generation:true,original_ef_complete:true,initial_state_retained:true,all_current_residual_source_columns:true}}
    fn delta(c:&PopulationControl,value:f64)->OutputDelta{OutputDelta{scope:Arc::clone(&c.scope),state:value,groups:vec![0.;c.groups.len()],maximum:value}}
    fn good(c:&PopulationControl)->OutputCheck{c.assess_output(&vec![C{re:0.25,im:0.};8],true).unwrap()}
    #[test]
    fn preparation_population_control_mature_matches_python_three_tolerances_and_zero_lift_rejection(){
        let owner=super::super::tests::fixture();
        for(epsilon,expected)in[(0.02,Action::EstimateAccepted),(0.005,Action::RefineBasis),(0.001,Action::RefineBasis)]{
            let c=control(&owner,epsilon,false);let modes=vec![vec![-1],vec![0],vec![1]];
            let frames=[frame(&c,1,2,2,modes.clone(),2),frame(&c,2,3,2,modes.clone(),1),frame(&c,3,4,2,modes.clone(),1)];
            let changes=[delta(&c,0.0036),delta(&c,0.0009)];let domain=delta(&c,0.0032);let previous=frame(&c,1,2,1,vec![vec![0]],2);let output=good(&c);
            let result=c.decide_mature(WorkEvidence{frames:&frames,changes:&changes,current_output:&output,domain_change:Some(&domain),previous_domain:Some(&previous),previous_domain_basis_passed:true,exact_zero:None,basis_cost_ms:Some(1.),expansion_cost_ms:Some(2.)});
            assert_eq!(result.action,expected);assert_eq!(result.epsilon,epsilon);assert!(!result.rigorous_error_bound);assert!((result.basis_indicator.unwrap()-0.0072).abs()<1e-16);
            let mut invalid=frames.clone();invalid[2].basis_columns=3;invalid[2].new_directions=0;invalid[2].zero_lift_only=true;
            let denied=c.decide_mature(WorkEvidence{frames:&invalid,changes:&changes,current_output:&output,domain_change:Some(&domain),previous_domain:Some(&previous),previous_domain_basis_passed:true,exact_zero:None,basis_cost_ms:None,expansion_cost_ms:None});assert_eq!(denied.action,Action::Unresolved);
        }
        assert_eq!(ControlOptions::default().epsilon,0.005);
        assert!(PopulationControl::new(&owner,ControlOptions{epsilon:0.,..Default::default()},&[0.],&[0.25;4],groups()).is_err());
    }
    #[test]
    fn preparation_population_control_pointwise_overlapping_groups_and_separate_invariants(){
        let owner=super::super::tests::fixture();let c=control(&owner,0.02,true);let modes=vec![vec![-1],vec![0],vec![1]];
        let frames=[frame(&c,1,2,1,modes.clone(),2),frame(&c,2,3,1,modes.clone(),1),frame(&c,3,4,2,vec![vec![-2],vec![-1],vec![0],vec![1],vec![2]],1)];
        let a=vec![C{re:0.25,im:0.};8];let mut b=a.clone();b[4].re+=0.006;b[5].re-=0.006;let mut d=b.clone();d[6].re+=0.006;d[7].re-=0.006;
        let checks=[c.assess_output(&a,true).unwrap(),c.assess_output(&b,true).unwrap(),c.assess_output(&d,true).unwrap()];
        let result=c.decide_pointwise([&frames[0],&frames[1],&frames[2]],[&a,&b,&d],[&checks[0],&checks[1],&checks[2]]).unwrap();
        assert!((result.decision.joint_indicator.unwrap()-0.006).abs()<1e-15);assert!((result.sum_of_separate_global_maxima-0.012).abs()<1e-15);assert_eq!(result.decision.action,Action::EstimateAccepted);
        assert!(result.group_indicators.iter().flatten().all(|x|x.abs()<1e-15));assert!(!result.all_halo_error_control);
        let mut bad=a.clone();bad[4].re=-0.006;bad[5].re=0.506;let check=c.assess_output(&bad,true).unwrap();assert!(check.numerical_invariants_passed);assert!(!check.probability_ranges_passed);
        bad[0].im=1e-5;assert!(!c.assess_output(&bad,true).unwrap().numerical_invariants_passed);
        let other=control(&owner,0.02,true);assert!(other.decide_pointwise([&frames[0],&frames[1],&frames[2]],[&a,&b,&d],[&checks[0],&checks[1],&checks[2]]).is_err());
    }
    #[test]
    fn preparation_population_control_exact_zero_domain_only_replaces_domain_comparison(){
        use crate::preparation::MatrixView;
        let h=[0.,0.2,0.2,3.];let zero=[0.;4];let cc=[0.,0.5,0.,0.];let collapse=[MatrixView{real:&cc,imag:&zero}];
        let model=ModelView{dimension:2,hamiltonian:MatrixView{real:&h,imag:&zero},collapse:&collapse,drives:&[],ground:&[true,false],initial_index:0};
        let owner=OwnedPrepared::prepare(&model,PhaseBasis::from_declared(&[],None,&[]).unwrap(),GaugePolicy::Identity,KernelPolicy::Csr).unwrap();
        let c=PopulationControl::new(&owner,ControlOptions::default(),&[0.,1.],&[1.,0.],vec![]).unwrap();let domain=vec![vec![]];let proof=c.prove_exact_zero_fourier(&owner,&domain).unwrap();assert!(proof.proven);
        let frames=[frame(&c,1,2,1,domain.clone(),2),frame(&c,2,3,1,domain.clone(),1),frame(&c,3,4,1,domain.clone(),1)];let changes=[delta(&c,1e-6),delta(&c,2e-6)];
        let output=c.assess_output(&[C{re:1.,im:0.},C::default(),C{re:1.,im:0.},C::default()],true).unwrap();
        let result=c.decide_mature(WorkEvidence{frames:&frames,changes:&changes,current_output:&output,domain_change:None,previous_domain:None,previous_domain_basis_passed:false,exact_zero:Some(&proof),basis_cost_ms:None,expansion_cost_ms:None});
        assert_eq!(result.action,Action::EstimateAccepted);assert!(result.exact_fourier_truncation_zero);assert!(!result.rigorous_error_bound);
        let incomplete=c.decide_mature(WorkEvidence{frames:&frames[..2],changes:&changes[..1],current_output:&output,domain_change:None,previous_domain:None,previous_domain_basis_passed:false,exact_zero:Some(&proof),basis_cost_ms:None,expansion_cost_ms:None});assert_eq!(incomplete.action,Action::RefineBasis);
    }
}
