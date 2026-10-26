//! Append-only current-request trial workspace. Checkpoint evaluation never
//! regenerates old directions, E/F images, projection blocks or observer rows.
//! External corrections/fast-pole pairs/moments enter through one raw-vector
//! interface and receive original images AFTER actual orthogonalization.
use super::*;
use super::super::coordinates::{PopulationRowBlocks,PopulationPhasePlan};
use super::super::residual_proposal::{ProposalOptions,ProposalDiagnostics,ResidualInput,ResidualProposal};
use super::super::domain_transfer::TransferredImages;
use std::collections::VecDeque;

#[derive(Clone,Copy,Debug)]
pub enum DirectionPolicy{Fixed{steps:usize},Adaptive(AdaptiveConstruction),Strict}
#[derive(Clone,Debug)]
pub enum TrialSource{Initial,Frontier{column:usize,epoch:usize},FastPolePair,InitialMoment{order:usize},ResidualCorrection,ImportedDomain,External}
#[derive(Clone,Copy)]struct Frontier{column:usize,generation:usize}
#[derive(Clone,Debug)]pub struct AppendReport{pub supplied_columns:usize,pub retained_columns:usize,pub deflated_columns:usize,pub prior_columns:usize,pub current_columns:usize,pub source:TrialSource,
    pub transported_image_columns:usize,pub recomputed_image_columns:usize}
#[derive(Clone,Debug)]pub struct ExtensionReport{pub requested_columns:usize,pub prior_columns:usize,pub current_columns:usize,pub consumed_frontier_sources:usize,pub complete_generation:bool,pub breakdown_unqualified:bool}
#[derive(Clone,Debug)]pub struct WorkspaceStatistics{
    pub columns:usize,pub original_image_columns_evaluated:usize,pub observation_columns_projected:usize,
    pub raw_mass_entries_computed:usize,pub raw_pencil_entries_computed:usize,
    pub old_images_recomputed:bool,pub candidates_evaluated:usize,pub frontier_pending:usize,
    pub complete_frontier_generation:bool,pub frontier_epoch:usize,
    pub observation_phase_evaluations:usize,
}
pub struct RawTrialMatrices<'a>{pub rows:usize,pub columns:usize,pub q:&'a[f64],pub eq:&'a[f64],pub fq:&'a[f64],pub projected_mass:&'a[f64],pub projected_pencil:&'a[f64]}

pub struct TrialWorkspace{
    owner:Arc<PrimitivePreparation>,identity:Arc<()>,lapack:Arc<Lapack>,observer:Arc<PopulationObservation>,phase_plan:PopulationPhasePlan,phase_prepare_ms:f64,domain:Domain,
    modes:Vec<Charge>,times:Vec<f64>,initial:Vec<f64>,initial_imaginary:f64,policy:DirectionPolicy,
    q:Vec<f64>,eq:Vec<f64>,fq:Vec<f64>,mass:Vec<f64>,pencil:Vec<f64>,gram:Vec<f64>,
    initial_coefficients:Vec<f64>,initial_reconstruction:Vec<f64>,trace_q:Vec<f64>,trace_e:Vec<f64>,trace_f:Vec<f64>,
    fq_norm:f64,noncentral_trace:f64,observation:Option<PopulationRowBlocks>,
    columns:usize,requested:usize,frontier:VecDeque<Frontier>,epoch:usize,breakdown:bool,
    origins:Vec<TrialSource>,directions:Vec<DirectionReport>,runtime_profile:RuntimePreparationProfile,p0_profile:RequestPreparationProfile,
    projection_entries:usize,observed_columns:usize,candidates:usize,last_candidate_columns:Option<usize>,
}
fn valid_policy(policy:DirectionPolicy)->Result<(),String>{
    match policy{DirectionPolicy::Fixed{steps}if steps==0||steps>16=>Err("Fixed proposal steps must be1..16".into()),
        DirectionPolicy::Adaptive(a)if !a.construction_budget_scale.is_finite()||a.construction_budget_scale<=0.||a.maximum_inner_steps==0=>Err("Invalid adaptive construction policy".into()),_=>Ok(())}
}
/// Build exactly three new blocks, preserving old-old values bitwise. E/F
/// are not symmetric: top-right and bottom-left are independently evaluated.
fn enlarged_projection(lap:&Lapack,old:&[f64],q:&[f64],image:&[f64],new_q:&[f64],new_image:&[f64],rows:usize,m:usize,k:usize)->Result<Vec<f64>,String>{
    if old.len()!=product(&[m,m])?||q.len()!=product(&[rows,m])?||image.len()!=q.len()||new_q.len()!=product(&[rows,k])?||new_image.len()!=new_q.len(){
        return Err("Append-only projected-image shapes are inconsistent".into());}
    let size=m.checked_add(k).ok_or("Projected dimension overflow")?;let mut out=vec![0.;product(&[size,size])?];
    for j in 0..m{out[j*size..j*size+m].copy_from_slice(&old[j*m..(j+1)*m]);}
    if m>0&&k>0{
        let mut top=vec![0.;m*k];let mut bottom=vec![0.;k*m];
        lap.multiply_transpose_left_into(q,new_image,&mut top,rows,m,k,rows,rows,m)?;
        lap.multiply_transpose_left_into(new_q,image,&mut bottom,rows,k,m,rows,rows,k)?;
        for j in 0..k{out[(m+j)*size..(m+j)*size+m].copy_from_slice(&top[j*m..(j+1)*m]);}
        for j in 0..m{out[j*size+m..j*size+m+k].copy_from_slice(&bottom[j*k..(j+1)*k]);}
    }
    if k>0{let mut diagonal=vec![0.;k*k];lap.multiply_transpose_left_into(new_q,new_image,&mut diagonal,rows,k,k,rows,rows,k)?;
        for j in 0..k{out[(m+j)*size+m..(m+j)*size+m+k].copy_from_slice(&diagonal[j*k..(j+1)*k]);}}
    finite(&out,"incremental projected block")?;Ok(out)
}

impl RuntimeCore{
    pub fn cold_workspace(&mut self,modes:&[Charge],times:&[f64],pole:f64,policy:DirectionPolicy)->Result<TrialWorkspace,String>{
        valid_policy(policy)?;if !pole.is_finite()||pole<=0.||times.is_empty()||times.iter().any(|t|!t.is_finite()||*t<0.)||times.windows(2).any(|w|w[0]>w[1]){
            return Err("Cold workspace needs a positive pole and ordered finite actual times".into());}
        let mut domain=self.domain(modes,pole,4)?;let rows=domain.map.dimension();let n=self.owned.primitives.hilbert.source.n;let d=n*n;
        let central=modes.iter().position(|q|q.iter().all(|&x|x==0)).ok_or("Missing central initial mode")?;
        domain.signed[central*d..(central+1)*d].copy_from_slice(self.initial.point());domain.map.pull_complex_into(&domain.signed,1,&mut domain.native)?;
        let imaginary=domain.native.iter().fold(0_f64,|v,z|v.hypot(z.im));let allowance=512.*f64::EPSILON*n as f64*complex_norm(&domain.signed);
        if imaginary>allowance{return Err("Initial REAL pull exceeds original isometry roundoff guard".into());}
        let initial:Vec<_>=domain.native.iter().map(|z|z.re).collect();if norm(&initial)==0.{return Err("Unresolved workspace initial seed".into());}
        let phase_begin=Instant::now();let phase_plan=self.observer.prepare_phase_plan(modes,times)?;let phase_prepare_ms=phase_begin.elapsed().as_secs_f64()*1000.;
        let mut workspace=TrialWorkspace{owner:Arc::clone(&self.owned.primitives),identity:Arc::new(()),lapack:Arc::clone(&self.owned.preparation_lapack),observer:Arc::clone(&self.observer),phase_plan,phase_prepare_ms,domain,
            modes:modes.to_vec(),times:times.to_vec(),initial:initial.clone(),initial_imaginary:imaginary,policy,q:vec![],eq:vec![],fq:vec![],mass:vec![],pencil:vec![],gram:vec![],
            initial_coefficients:vec![],initial_reconstruction:vec![0.;rows],trace_q:vec![],trace_e:vec![],trace_f:vec![],fq_norm:0.,noncentral_trace:0.,observation:None,
            columns:0,requested:1,frontier:VecDeque::new(),epoch:0,breakdown:false,origins:vec![],directions:vec![],
            runtime_profile:self.preparation_profile.clone(),p0_profile:self.p0.preparation_profile().clone(),projection_entries:0,observed_columns:0,candidates:0,last_candidate_columns:None};
        workspace.append_internal(&initial,1,&[0.],TrialSource::Initial,0)?;Ok(workspace)
    }
    /// Consume an exact expanded-domain image transfer. Old basis/images and
    /// observer factors are imported, not recalculated. No 50-column rational
    /// frontier is silently started: expanded domains use complete residual
    /// correction blocks unless the caller explicitly restarts that frontier.
    pub fn workspace_from_transfer(&mut self,previous:&Candidate,transferred:TransferredImages,policy:DirectionPolicy)->Result<TrialWorkspace,String>{
        valid_policy(policy)?;
        if !previous.belongs_to(&self.owned)||!transferred.belongs_to(&self.owned)||previous.dimension!=transferred.columns
            ||previous.pole.to_bits()!=transferred.pole.to_bits()
            ||previous.observation_rows.coefficients().as_ptr()!=transferred.observation_rows.coefficients().as_ptr(){
            return Err("Expanded workspace needs its exact same-request source candidate/transfer".into());}
        let m=transferred.columns;let rows=transferred.rows;let n=self.owned.primitives.hilbert.source.n;let d=n*n;
        if previous.projected_gram.len()!=m*m||previous.projected_initial.len()!=m{return Err("Missing original incremental Gram/initial projection".into());}
        let modes=transferred.modes;let times=transferred.observation_rows.times().to_vec();let mut domain=self.domain(&modes,transferred.pole,4)?;
        if domain.map.dimension()!=rows||transferred.q.len()!=rows*m||transferred.eq.len()!=rows*m||transferred.fq.len()!=rows*m{return Err("Transferred workspace shapes differ".into());}
        let central=modes.iter().position(|q|q.iter().all(|&k|k==0)).ok_or("Transferred domain lost central mode")?;
        domain.signed[central*d..(central+1)*d].copy_from_slice(self.initial.point());domain.map.pull_complex_into(&domain.signed,1,&mut domain.native)?;
        let imaginary=domain.native.iter().fold(0_f64,|v,z|v.hypot(z.im));let allowance=512.*f64::EPSILON*n as f64*complex_norm(&domain.signed);
        if imaginary>allowance{return Err("Expanded-domain initial REAL pull is unresolved".into());}
        let initial:Vec<_>=domain.native.iter().map(|z|z.re).collect();let initial_coefficients=previous.projected_initial.clone();
        let initial_reconstruction=self.owned.preparation_lapack.multiply(&transferred.q,&initial_coefficients,rows,1,m);
        let mut noncentral=0_f64;for block in domain.map.harmonics(){if block.has_imaginary_block{for offset in[block.offset,block.offset+d]{for j in 0..m{
            noncentral=noncentral.max(transferred.q[j*rows+offset..j*rows+offset+n].iter().sum::<f64>().abs());}}}}
        if noncentral>1e-10{return Err("Transferred trial has invalid noncentral trace".into());}
        let trace_q=(0..m).map(|j|transferred.q[j*rows..j*rows+n].iter().sum()).collect();
        let trace_e=(0..m).map(|j|transferred.eq[j*rows..j*rows+n].iter().sum()).collect();
        let trace_f=(0..m).map(|j|transferred.fq[j*rows..j*rows+n].iter().sum()).collect();let fq_norm=norm(&transferred.fq);
        let before=Instant::now();let phase_plan=self.observer.prepare_phase_plan(&modes,&times)?;let phase_prepare_ms=before.elapsed().as_secs_f64()*1000.;
        Ok(TrialWorkspace{owner:Arc::clone(&self.owned.primitives),identity:Arc::new(()),lapack:Arc::clone(&self.owned.preparation_lapack),observer:Arc::clone(&self.observer),phase_plan,phase_prepare_ms,domain,
            modes,times,initial,initial_imaginary:imaginary,policy,q:transferred.q,eq:transferred.eq,fq:transferred.fq,mass:transferred.projected_mass,pencil:transferred.projected_pencil,
            gram:previous.projected_gram.clone(),initial_coefficients,initial_reconstruction,trace_q,trace_e,trace_f,fq_norm,noncentral_trace:noncentral,
            observation:Some(PopulationRowBlocks::new(transferred.observation_rows)),columns:m,requested:m,frontier:VecDeque::new(),epoch:0,breakdown:false,
            origins:vec![TrialSource::ImportedDomain;m],directions:vec![],runtime_profile:self.preparation_profile.clone(),p0_profile:self.p0.preparation_profile().clone(),
            projection_entries:0,observed_columns:0,candidates:0,last_candidate_columns:None})
    }
}

impl TrialWorkspace{
    pub fn belongs_to(&self,prepared:&OwnedPrepared)->bool{Arc::ptr_eq(&self.owner,&prepared.primitives)}
    pub fn is_current_candidate(&self,candidate:&Candidate)->bool{
        Arc::ptr_eq(&self.identity,&candidate.workspace_identity)&&candidate.basis_revision==self.columns&&candidate.dimension==self.columns
            &&candidate.modes==self.modes&&candidate.pole.to_bits()==self.domain.pole.to_bits()
            &&candidate.times.len()==self.times.len()&&candidate.times.iter().zip(&self.times).all(|(a,b)|a.to_bits()==b.to_bits())
    }
    pub fn columns(&self)->usize{self.columns}
    pub fn modes(&self)->&[Charge]{&self.modes}
    pub fn pole(&self)->f64{self.domain.pole}
    pub(super) fn initial_native(&self)->&[f64]{&self.initial}
    pub(super) fn native_domain(&mut self)->&mut Domain{&mut self.domain}
    pub fn times(&self)->&[f64]{&self.times}
    pub fn raw_matrices(&self)->RawTrialMatrices<'_>{RawTrialMatrices{rows:self.domain.map.dimension(),columns:self.columns,q:&self.q,eq:&self.eq,fq:&self.fq,projected_mass:&self.mass,projected_pencil:&self.pencil}}
    pub fn origins(&self)->&[TrialSource]{&self.origins}
    fn complete_generation(&self)->bool{self.frontier.front().is_none_or(|first|self.frontier.iter().all(|f|f.generation==first.generation))}
    pub fn statistics(&self)->WorkspaceStatistics{WorkspaceStatistics{columns:self.columns,original_image_columns_evaluated:self.domain.entered,
        observation_columns_projected:self.observed_columns,raw_mass_entries_computed:self.projection_entries,raw_pencil_entries_computed:self.projection_entries,
        old_images_recomputed:false,candidates_evaluated:self.candidates,frontier_pending:self.frontier.len(),complete_frontier_generation:self.complete_generation(),frontier_epoch:self.epoch,
        observation_phase_evaluations:self.phase_plan.phase_evaluations()}}
    /// Changing inner policy explicitly restarts the complete current frontier;
    /// old true Q/EQ/FQ/observer factors remain unchanged and are not recomputed.
    pub fn restart_frontier(&mut self,policy:DirectionPolicy)->Result<(),String>{valid_policy(policy)?;self.policy=policy;self.epoch+=1;self.breakdown=false;
        self.frontier=(0..self.columns).map(|column|Frontier{column,generation:0}).collect();Ok(())}
    pub fn extend_to(&mut self,target:usize)->Result<ExtensionReport,String>{
        let before=self.columns;if target<before||target>self.domain.map.dimension(){return Err("Workspace checkpoint must extend within the complete domain dimension".into());}
        self.requested=target;let mut consumed=0;while self.columns<target&&!self.frontier.is_empty(){self.advance_one()?;consumed+=1;}
        if self.frontier.is_empty()&&self.columns<target{self.breakdown=true;}
        Ok(ExtensionReport{requested_columns:target,prior_columns:before,current_columns:self.columns,consumed_frontier_sources:consumed,
            complete_generation:self.complete_generation(),breakdown_unqualified:self.breakdown})
    }
    /// Optional block-frontier completion. No member is skipped merely because
    /// another seed failed to add a direction. This is not a convergence gate.
    pub fn finish_frontier_generation(&mut self)->Result<ExtensionReport,String>{
        let before=self.columns;let generation=self.frontier.front().map(|f|f.generation);let mut consumed=0;
        while self.frontier.front().is_some_and(|f|Some(f.generation)==generation){self.advance_one()?;consumed+=1;}
        if self.frontier.is_empty(){self.breakdown=true;}
        Ok(ExtensionReport{requested_columns:before,prior_columns:before,current_columns:self.columns,consumed_frontier_sources:consumed,
            complete_generation:self.complete_generation(),breakdown_unqualified:self.breakdown})
    }
    fn advance_one(&mut self)->Result<(),String>{
        let source=*self.frontier.front().ok_or("No frontier source")?;let rows=self.domain.map.dimension();let rhs=&self.eq[source.column*rows..(source.column+1)*rows];
        let(mut report,mut strict_report)=(None,None);
        let vector=match self.policy{
            DirectionPolicy::Fixed{steps}=>self.domain.direction(rhs,steps)?,
            DirectionPolicy::Adaptive(policy)=>{let result=adaptive_direction::solve_direction_flat(&mut self.domain,rhs,&self.q,self.columns,AdaptiveOptions{
                direction_budget:policy.construction_budget_scale/(16.max(self.columns)as f64),maximum_steps:policy.maximum_inner_steps,checkpoint_policy:policy.checkpoint_policy})?;
                report=Some(result.report);result.vector},
            DirectionPolicy::Strict=>{let(vector,strict)=adaptive_direction::strict_direction(&mut self.domain,rhs)?;strict_report=Some(strict);vector},
        };
        // Do not consume the source if a new original image/guard fails.
        self.append_internal(&vector,1,&[0.],TrialSource::Frontier{column:source.column,epoch:self.epoch},source.generation+1)?;
        let popped=self.frontier.pop_front().ok_or("Frontier unexpectedly disappeared")?;
        if popped.column!=source.column||popped.generation!=source.generation{return Err("Frontier source order changed during append".into());}
        if let Some(report)=report{self.directions.push(report);}
        // Strict-only work is accounted by original Domain counters. It does
        // not masquerade as a passed adaptive screen in DirectionReport.
        let _=strict_report;Ok(())
    }
    /// Complete raw REAL block in column-major [rows,columns]. `resolution`
    /// contains caller-provided absolute uncertainty floors, not population
    /// tolerances. Zero is appropriate for an exactly supplied test vector;
    /// residual proposals must preserve their source-formation qualification.
    pub fn append_trial_columns(&mut self,columns:&[f64],count:usize,resolution:&[f64],source:TrialSource)->Result<AppendReport,String>{
        let generation=self.frontier.front().map_or(0,|f|f.generation);
        self.append_internal(columns,count,resolution,source,generation)
    }
    /// Split BOTH components of a current signed complex-pole direction using
    /// the complex extension of the actual REAL isometry. A complex-pole
    /// solution is not itself a Hermitian field: no REAL-only pull is valid.
    /// This does not select a pole or assert its successor/solve qualification.
    pub fn append_complex_trial_pair(&mut self,signed:&[C],resolution:&[f64;2])->Result<AppendReport,String>{
        let rows=self.domain.map.dimension();let mut native=vec![C::default();rows];self.domain.map.pull_complex_into(signed,1,&mut native)?;
        let mut values=vec![0.;2*rows];for i in 0..rows{values[i]=native[i].re;values[rows+i]=native[i].im;}
        self.append_trial_columns(&values,2,resolution,TrialSource::FastPolePair)
    }
    fn append_internal(&mut self,input:&[f64],count:usize,resolution:&[f64],source:TrialSource,generation:usize)->Result<AppendReport,String>{
        self.append_impl(input,count,resolution,source,generation,None)
    }
    /// Only the strict fast-pole module supplies these original raw images.
    /// Transport uses the identical orthogonalization coefficients as Q;
    /// cancellation/trace failures trigger actual normalized E/F evaluation.
    pub(super) fn append_verified_pair(&mut self,input:&[f64],e:&[f64],f:&[f64])->Result<AppendReport,String>{
        let generation=self.frontier.front().map_or(0,|v|v.generation);
        self.append_impl(input,2,&[0.,0.],TrialSource::FastPolePair,generation,Some((e,f)))
    }
    /// The mature single complex-pole pair is an auxiliary augmentation, not
    /// two new real-pole chains. Its original full E/F closure is checked by
    /// the only caller before this is used; all slow-chain sources remain.
    pub(super) fn register_verified_fast_auxiliaries(&mut self,first:usize,count:usize){
        self.frontier.retain(|f|f.column<first||f.column>=first+count);
    }
    fn transported_trace_passes(&self,q:&[f64],e:&[f64],f:&[f64])->Result<bool,String>{
        let n=self.owner.hilbert.source.n;let d=n*n;let fgoal=4096.*f64::EPSILON*norm(f).max(1.);
        for block in self.domain.map.harmonics(){let offsets=if block.has_imaginary_block{vec![block.offset,block.offset+d]}else{vec![block.offset]};
            let tr:Vec<_>=offsets.iter().map(|&o|q[o..o+n].iter().sum::<f64>()).collect();let frequency=self.owner.hilbert.source.phases.frequency(&block.index)?;let w=frequency.hi+frequency.lo;
            for(j,&o)in offsets.iter().enumerate(){let et=e[o..o+n].iter().sum::<f64>();let ft=f[o..o+n].iter().sum::<f64>();
                let expected=if !block.has_imaginary_block{0.}else if j==0{w*tr[1]}else{-w*tr[0]};
                if (et-tr[j]).abs()>1e-11||(ft-expected).abs()>fgoal{return Ok(false);}}
        }Ok(true)
    }
    fn append_impl(&mut self,input:&[f64],count:usize,resolution:&[f64],source:TrialSource,generation:usize,provided:Option<(&[f64],&[f64])>)->Result<AppendReport,String>{
        let rows=self.domain.map.dimension();let old=self.columns;
        if count==0||input.len()!=product(&[rows,count])?||resolution.len()!=count||resolution.iter().any(|x|!x.is_finite()||*x<0.){
            return Err("Invalid raw trial block or resolution floors".into());}finite(input,"raw trial block")?;
        if let Some((e,f))=provided{if e.len()!=input.len()||f.len()!=input.len(){return Err("Verified trial-image shape differs".into());}finite(e,"verified E")?;finite(f,"verified F")?;}
        let mut new_q=Vec::new();let mut deflated=0;let mut saved_e=Vec::new();let mut saved_f=Vec::new();let(mut transported,mut recomputed)=(0,0);
        for j in 0..count{let mut value=input[j*rows..(j+1)*rows].to_vec();let original=norm(&value);
            let mut ev=provided.map(|(e,_)|e[j*rows..(j+1)*rows].to_vec());let mut fv=provided.map(|(_,f)|f[j*rows..(j+1)*rows].to_vec());
            for _ in 0..2{for(index,q)in self.q.chunks_exact(rows).chain(new_q.chunks_exact(rows)).enumerate(){let coefficient=dot(q,&value);for i in 0..rows{value[i]-=coefficient*q[i];}
                if let(Some(e),Some(f))=(&mut ev,&mut fv){let(eq,fq)=if index<old{(&self.eq[index*rows..(index+1)*rows],&self.fq[index*rows..(index+1)*rows])}
                    else{let k=index-old;(&saved_e[k*rows..(k+1)*rows],&saved_f[k*rows..(k+1)*rows])};
                    for i in 0..rows{e[i]-=coefficient*eq[i];f[i]-=coefficient*fq[i];}}
            }}
            let length=norm(&value);if !length.is_finite(){return Err("Nonfinite trial orthogonalization".into());}
            if length==0.||length<=resolution[j].max(1e-12*original){deflated+=1;continue;}
            if old+new_q.len()/rows>=rows{return Err("Resolved trial block would exceed complete REAL dimension".into());}
            for x in &mut value{*x/=length;}
            if let(Some(mut e),Some(mut f))=(ev,fv){for x in &mut e{*x/=length;}for x in &mut f{*x/=length;}
                if length<=1e-9*original||!self.transported_trace_passes(&value,&e,&f)?{let actual=self.domain.entered_images(&value)?;e=actual.0;f=actual.1;recomputed+=1;}else{transported+=1;}
                saved_e.extend(e);saved_f.extend(f);
            }
            new_q.extend(value);
        }
        let k=new_q.len()/rows;
        if k==0{return Ok(AppendReport{supplied_columns:count,retained_columns:0,deflated_columns:deflated,prior_columns:old,current_columns:old,source,transported_image_columns:0,recomputed_image_columns:0});}
        // Prepare all new E/F and observer blocks before committing the basis.
        // A failed block leaves the old workspace matrices/rows immutable.
        let mut new_e=if provided.is_some(){saved_e}else{vec![0.;rows*k]};let mut new_f=if provided.is_some(){saved_f}else{vec![0.;rows*k]};let mut observer=self.observation.clone();
        for start in(0..k).step_by(4){let width=4.min(k-start);let mut packed=vec![0.;rows*width];
            for i in 0..rows{for j in 0..width{packed[i*width+j]=new_q[(start+j)*rows+i];}}
            if provided.is_none(){let(e,f)=self.domain.entered_images_columns(&packed,width)?;
                for i in 0..rows{for j in 0..width{new_e[(start+j)*rows+i]=e[i*width+j];new_f[(start+j)*rows+i]=f[i*width+j];}}recomputed+=width;
            }else{self.domain.prepare_buffers(width)?;self.domain.map.push_into(&packed,width,&mut self.domain.signed)?;}
            // entered_images already left this exact new block in signed form.
            let projected=self.observer.project(&self.modes,&self.domain.signed,width)?.at_phase_plan(&self.phase_plan)?;
            if let Some(store)=observer.as_mut(){store.append(projected)?;}else{observer=Some(PopulationRowBlocks::new(projected));}
        }
        let mass=enlarged_projection(&self.lapack,&self.mass,&self.q,&self.eq,&new_q,&new_e,rows,old,k)?;
        let pencil=enlarged_projection(&self.lapack,&self.pencil,&self.q,&self.fq,&new_q,&new_f,rows,old,k)?;
        let gram=enlarged_projection(&self.lapack,&self.gram,&self.q,&self.q,&new_q,&new_q,rows,old,k)?;
        let size=old+k;let mut defect=gram.clone();for i in 0..size{defect[i+i*size]-=1.;}
        if norm(&defect)>1e-10{return Err("New trial block loses complete REAL orthogonality".into());}
        let n=self.owner.hilbert.source.n;let d=n*n;let mut noncentral=self.noncentral_trace;
        for block in self.domain.map.harmonics(){if block.has_imaginary_block{for offset in[block.offset,block.offset+d]{for j in 0..k{
            noncentral=noncentral.max(new_q[j*rows+offset..j*rows+offset+n].iter().sum::<f64>().abs());
        }}}}
        if noncentral>1e-10{return Err("New raw trial block violates fixed noncentral trace invariance".into());}
        let mut initial_coefficients=vec![0.;k];self.lapack.multiply_transpose_left_into(&new_q,&self.initial,&mut initial_coefficients,rows,k,1,rows,rows,k)?;
        let increment=self.lapack.multiply(&new_q,&initial_coefficients,rows,1,k);
        for i in 0..rows{self.initial_reconstruction[i]+=increment[i];}
        for j in 0..k{
            self.trace_q.push(new_q[j*rows..j*rows+n].iter().sum());self.trace_e.push(new_e[j*rows..j*rows+n].iter().sum());self.trace_f.push(new_f[j*rows..j*rows+n].iter().sum());
            self.origins.push(source.clone());self.frontier.push_back(Frontier{column:old+j,generation});
        }
        self.frontier.make_contiguous().sort_by_key(|f|f.generation);
        self.fq_norm=self.fq_norm.hypot(norm(&new_f));self.initial_coefficients.extend(initial_coefficients);
        self.q.extend(new_q);self.eq.extend(new_e);self.fq.extend(new_f);self.mass=mass;self.pencil=pencil;self.gram=gram;self.observation=observer;
        self.columns=size;self.requested=self.requested.max(size);self.observed_columns+=k;self.projection_entries+=2*old*k+k*k;self.noncentral_trace=noncentral;self.breakdown=false;
        Ok(AppendReport{supplied_columns:count,retained_columns:k,deflated_columns:deflated,prior_columns:old,current_columns:size,source,transported_image_columns:transported,recomputed_image_columns:recomputed})
    }
    /// Immutable publication snapshot. Large Q/EQ/FQ arrays are copied only
    /// for this caller-owned candidate, not reapplied or used as cached answers.
    pub fn candidate(&mut self)->Result<Candidate,String>{
        let m=self.columns;let rows=self.domain.map.dimension();let n=self.owner.hilbert.source.n;let lap=&self.lapack;
        if m==0{return Err("Workspace has no initialized basis".into());}
        let overlap=norm(&self.trace_q);if overlap<1e-8{return Err("No resolved physical trace direction".into());}
        let mut v:Vec<_>=self.trace_q.iter().map(|x|x/overlap).collect();v[0]+=if v[0]>=0.{1.}else{-1.};let length=norm(&v);for x in &mut v{*x/=length;}
        let mut rotation=vec![0.;m*m];for j in 0..m{for i in 0..m{rotation[i+j*m]=-2.*v[i]*v[j];}rotation[j+j*m]+=1.;}
        let rt=transpose(&rotation,m,m);let gamma=dot(&self.trace_q,&rotation[..m]);
        let mut em=lap.multiply(&lap.multiply(&rt,&self.mass,m,m,m),&rotation,m,m,m);
        let mut fm=lap.multiply(&lap.multiply(&rt,&self.pencil,m,m,m),&rotation,m,m,m);
        let mut mass_error=0_f64;let mut pencil_error=0_f64;
        for j in 0..m{mass_error=mass_error.max((dot(&self.trace_e,&rotation[j*m..(j+1)*m])/gamma-if j==0{1.}else{0.}).abs());
            pencil_error=pencil_error.max((dot(&self.trace_f,&rotation[j*m..(j+1)*m])/gamma).abs());}
        if mass_error>1e-11||pencil_error>4096.*f64::EPSILON*self.fq_norm.max(1.){return Err("Original trace-test identity failed before local Petrov test replacement".into());}
        for j in 0..m{em[j*m]=if j==0{1.}else{0.};fm[j*m]=0.;}
        let initial_error=norm(&self.initial_reconstruction.iter().zip(&self.initial).map(|(a,b)|a-b).collect::<Vec<_>>());
        if initial_error>1e-10{return Err("Workspace lost its complete original initial vector".into());}
        let c0=lap.multiply(&rt,&self.initial_coefficients,m,1,m);
        let system=super::super::reduced_system::ReducedSystem::prepare(Arc::clone(lap),&em,&fm,m)?;
        let action=system.evolve(&c0,&self.times)?;let unrotated=lap.multiply(&rotation,&action.states,m,self.times.len(),m);let coefficients=transpose(&unrotated,m,self.times.len());
        let observation_rows=self.observation.as_ref().ok_or("Missing append-only observation rows")?.snapshot()?;
        let complex_coefficients:Vec<_>=coefficients.iter().map(|&re|C{re,im:0.}).collect();let mut populations=vec![C::default();self.times.len()*n];observation_rows.evaluate_into(&complex_coefficients,&mut populations)?;
        let population_imaginary=populations.iter().map(|z|z.im.abs()).fold(0_f64,f64::max);
        if population_imaginary>1e-9{return Err("Exact population rows lost fixed real invariance".into());}
        let population_trace=populations.chunks_exact(n).map(|row|(row.iter().map(|z|z.re).sum::<f64>()-1.).abs()).fold(0_f64,f64::max);
        let mut gram=self.gram.clone();for i in 0..m{gram[i+i*m]-=1.;}let orthogonality=norm(&gram);
        let prior=self.last_candidate_columns;self.last_candidate_columns=Some(m);self.candidates+=1;
        let diagnostics=CandidateDiagnostics{accepted:false,construction_only:true,proposal_policy:match self.policy{DirectionPolicy::Fixed{..}=>"fixed-count full-M/P0 construction-only workspace",DirectionPolicy::Adaptive(_)=>"adaptive fixed-P right-Arnoldi direction workspace with original complex-M strict fallback",DirectionPolicy::Strict=>"strict original-M directions; outer population candidate remains unqualified"},
            requested_basis:self.requested,actual_basis:m,breakdown_unqualified:self.breakdown,orthogonality_error_hs:orthogonality,original_initial_projection_error_hs:initial_error,
            initial_real_pull_imaginary_hs:self.initial_imaginary,maximum_entered_reality_ratio:self.domain.maximum_reality_ratio,maximum_proposal_pull_imaginary_hs:self.domain.maximum_proposal_imaginary,
            maximum_entered_multiplication_only_ratio:self.domain.maximum_arithmetic_ratio,prepared_e_real_defect_bound:self.domain.plan.real_preparation_defects().0,
            prepared_f_real_defect_bound:self.domain.plan.real_preparation_defects().1,preparation_real_budget_is_population_error_bound:false,
            entered_columns:m,original_inner_m_products:self.domain.m_products,preconditioner_calls:self.domain.p0_calls,original_complex_residual_checks:self.domain.strict_checks,directions:self.directions.clone(),
            runtime_preparation:self.runtime_profile.clone(),p0_preparation:self.p0_profile.clone(),domain_setup_ms:self.domain.setup_ms,domain_shift_preparation_ms:self.domain.shift_preparation_ms,
            trace_test_overlap:overlap,mass_trace_identity_error:mass_error,pencil_trace_identity_error:pencil_error,noncentral_trace_error:self.noncentral_trace,trace_petrov_test_row_replaced:true,
            population_imaginary_error:population_imaginary,population_trace_error:population_trace,small_preparation:system.diagnostics().clone(),small_action:action.diagnostics,
            workspace_original_image_evaluations:self.domain.entered,workspace_observation_columns:self.observed_columns,workspace_raw_projection_entries:self.projection_entries,
            workspace_phase_evaluations:self.phase_plan.phase_evaluations(),workspace_phase_preparation_ms:self.phase_prepare_ms,
            new_columns_since_prior_candidate:prior.map(|p|m-p),complete_frontier_generation:self.complete_generation(),frontier_pending:self.frontier.len(),frontier_epoch:self.epoch};
        Ok(Candidate{owner:Arc::clone(&self.owner),workspace_identity:Arc::clone(&self.identity),basis_revision:m,q:self.q.clone(),eq:self.eq.clone(),fq:self.fq.clone(),rows,dimension:m,modes:self.modes.clone(),pole:self.domain.pole,
            projected_mass:self.mass.clone(),projected_pencil:self.pencil.clone(),projected_gram:self.gram.clone(),projected_initial:self.initial_coefficients.clone(),rotation,generator:system.generator().to_vec(),c0,coefficients,times:self.times.clone(),populations,observation_rows,diagnostics})
    }
}

pub struct CorrectionResult{
    pub proposal:ProposalDiagnostics,pub proposed_columns:usize,pub deferred_for_work_budget:bool,
    pub pending:Option<PendingCorrection>,pub append:Option<AppendReport>,pub candidate:Option<Candidate>,pub accepted:bool,
}
/// Current residual trial data, not propagated outputs. Keeping this on a
/// work-budget deferral prevents recomputing R/P0/Gram/SVD just to raise a cap.
pub struct PendingCorrection{owner:Arc<PrimitivePreparation>,workspace_identity:Arc<()>,basis_revision:usize,proposal:ResidualProposal}
impl PendingCorrection{
    pub fn columns(&self)->usize{self.proposal.columns}
    pub fn directions(&self)->&[f64]{&self.proposal.directions}
    pub fn diagnostics(&self)->&ProposalDiagnostics{&self.proposal.diagnostics}
    pub fn is_current(&self,space:&TrialWorkspace)->bool{Arc::ptr_eq(&self.owner,&space.owner)&&Arc::ptr_eq(&self.workspace_identity,&space.identity)&&self.basis_revision==space.columns}
}
impl RuntimeCore{
    /// Complete current residual proposal -> unified append -> NEW candidate.
    /// No old simulation or a proposal tail/Gram number is an acceptance gate.
    /// An over-budget proposal is reported intact, never silently truncated.
    pub fn correct_workspace(&mut self,space:&mut TrialWorkspace,current:&Candidate,options:ProposalOptions,maximum_new_columns:usize)->Result<CorrectionResult,String>{
        let pending=self.prepare_workspace_correction(space,current,options)?;
        self.apply_pending_correction(space,pending,maximum_new_columns)
    }
    pub fn prepare_workspace_correction(&mut self,space:&TrialWorkspace,current:&Candidate,options:ProposalOptions)->Result<PendingCorrection,String>{
        if !space.belongs_to(&self.owned)||!space.is_current_candidate(current){
            return Err("Correction requires the same current workspace snapshot".into());}
        if options.pole.to_bits()!=space.pole().to_bits(){return Err("Correction pole differs from workspace".into());}
        let source=self.owned.unrotated_candidate_source(current)?;let raw=space.raw_matrices();
        let proposal=self.owned.residual_proposal(&mut self.p0,ResidualInput{modes:space.modes(),columns:raw.columns,q:raw.q,eq:raw.eq,fq:raw.fq,source:&source},options)?;
        Ok(PendingCorrection{owner:Arc::clone(&space.owner),workspace_identity:Arc::clone(&space.identity),basis_revision:space.columns,proposal})
    }
    pub fn apply_pending_correction(&mut self,space:&mut TrialWorkspace,pending:PendingCorrection,maximum_new_columns:usize)->Result<CorrectionResult,String>{
        if !space.belongs_to(&self.owned)||!pending.is_current(space){return Err("Prepared residual proposal is stale or belongs to another workspace".into());}
        let proposed=pending.proposal.columns;
        if proposed>maximum_new_columns{return Ok(CorrectionResult{proposal:pending.proposal.diagnostics.clone(),proposed_columns:proposed,
            deferred_for_work_budget:true,pending:Some(pending),append:None,candidate:None,accepted:false});}
        let proposal=pending.proposal;
        if proposed==0{return Ok(CorrectionResult{proposal:proposal.diagnostics,proposed_columns:0,
            deferred_for_work_budget:false,pending:None,append:None,candidate:None,accepted:false});}
        // The proposal module has already propagated source-formation and
        // weighted-range resolution qualifications. Actual columns are still
        // reorthogonalized here and get complete original normalized E/F.
        let append=space.append_trial_columns(&proposal.directions,proposed,&vec![0.;proposed],TrialSource::ResidualCorrection)?;
        space.p0_profile=self.p0.preparation_profile().clone();
        let candidate=if append.retained_columns>0{Some(space.candidate()?)}else{None};
        Ok(CorrectionResult{proposal:proposal.diagnostics,proposed_columns:proposed,deferred_for_work_budget:false,pending:None,append:Some(append),candidate,accepted:false})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    fn runtime()->RuntimeCore{RuntimeCore::from_owned(super::super::super::tests::fixture(),InitialView{index:0,real:None,imag:None},P0Policy::EigenPreferred).unwrap()}
    #[test]
    fn preparation_workspace_extend8_to12_reuses_images_projection_and_observation(){
        let mut runtime=runtime();let modes=[vec![0],vec![1],vec![-1],vec![2],vec![-2]];let times=[0.,0.03,0.1];
        let mut split=runtime.cold_workspace(&modes,&times,8.,DirectionPolicy::Fixed{steps:1}).unwrap();split.extend_to(8).unwrap();let old=split.candidate().unwrap();assert_eq!(old.dimension,8);
        let saved=(old.q.clone(),old.eq.clone(),old.fq.clone(),old.projected_mass.clone(),old.projected_pencil.clone());let before=split.statistics();assert_eq!(before.original_image_columns_evaluated,8);assert_eq!(before.observation_columns_projected,8);
        split.extend_to(12).unwrap();let result=split.candidate().unwrap();assert_eq!(result.dimension,12);
        let mut direct=runtime.cold_workspace(&modes,&times,8.,DirectionPolicy::Fixed{steps:1}).unwrap();direct.extend_to(12).unwrap();let expected=direct.candidate().unwrap();
        for(a,b)in[(&result.q,&expected.q),(&result.eq,&expected.eq),(&result.fq,&expected.fq),(&result.projected_mass,&expected.projected_mass),(&result.projected_pencil,&expected.projected_pencil)]{assert_eq!(a,b);}
        for(a,b)in result.observation_rows.coefficients().iter().zip(expected.observation_rows.coefficients()){
            assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}
        assert_eq!(&result.eq[..8*old.rows],saved.1.as_slice());assert_eq!(&result.fq[..8*old.rows],saved.2.as_slice());assert_eq!(old.q,saved.0);
        for j in 0..8{for i in 0..8{assert_eq!(result.projected_mass[i+j*12].to_bits(),saved.3[i+j*8].to_bits());assert_eq!(result.projected_pencil[i+j*12].to_bits(),saved.4[i+j*8].to_bits());}}
        let after=split.statistics();assert_eq!(after.original_image_columns_evaluated,12);assert_eq!(after.observation_columns_projected,12);assert_eq!(after.raw_mass_entries_computed,144);
        assert_eq!(before.observation_phase_evaluations,after.observation_phase_evaluations);
        split.extend_to(12).unwrap();assert_eq!(split.statistics().original_image_columns_evaluated,12);let duplicate=split.candidate().unwrap();assert_eq!(duplicate.diagnostics.new_columns_since_prior_candidate,Some(0));
        assert_eq!(result.diagnostics.new_columns_since_prior_candidate,Some(4));assert!(!result.diagnostics.accepted);
    }
    #[test]
    fn preparation_workspace_extra_blocks_have_complete_frontiers_and_deflate_unresolved(){
        let mut runtime=runtime();let modes=[vec![0],vec![1],vec![-1]];let mut work=runtime.cold_workspace(&modes,&[0.,0.02],8.,DirectionPolicy::Fixed{steps:1}).unwrap();
        let rows=work.raw_matrices().rows;let mut extra=vec![0.;rows*2];extra[4]=1.;extra[rows+5]=1.;
        let native:Vec<_>=(0..rows).map(|i|C{re:extra[i],im:extra[rows+i]}).collect();let mut signed=vec![C::default();rows];work.domain.map.push_complex_into(&native,1,&mut signed).unwrap();
        let report=work.append_complex_trial_pair(&signed,&[0.,0.]).unwrap();assert_eq!(report.retained_columns,2);assert_eq!(work.columns(),3);
        let zero=work.append_trial_columns(&vec![0.;rows],1,&[0.],TrialSource::InitialMoment{order:1}).unwrap();assert_eq!(zero.retained_columns,0);assert_eq!(work.statistics().original_image_columns_evaluated,3);
        work.extend_to(4).unwrap();assert!(!work.statistics().complete_frontier_generation);work.finish_frontier_generation().unwrap();assert!(work.statistics().complete_frontier_generation);
        let count=work.columns();let images=work.statistics().original_image_columns_evaluated;work.restart_frontier(DirectionPolicy::Strict).unwrap();assert_eq!(work.statistics().frontier_pending,count);assert_eq!(work.statistics().original_image_columns_evaluated,images);
        assert!(work.append_trial_columns(&extra,2,&[f64::NAN,0.],TrialSource::External).is_err());assert_eq!(work.columns(),count);
    }
    #[test]
    fn preparation_workspace_residual_correction_joins_current_snapshot_without_acceptance(){
        let mut runtime=runtime();let modes=[vec![0],vec![1],vec![-1]];let times=[0.,0.03,0.1];
        let mut space=runtime.cold_workspace(&modes,&times,8.,DirectionPolicy::Fixed{steps:1}).unwrap();space.extend_to(3).unwrap();let initial=space.candidate().unwrap();
        let saved=(initial.q.clone(),initial.eq.clone(),initial.fq.clone());let options=ProposalOptions{pole:8.,duration:0.1,maximum_rhs:2,minimum_directions:1,
            relative_source_tail:0.05,absolute_source_tail:1e-4,check_gram_scaling:true};
        let prepared=runtime.prepare_workspace_correction(&space,&initial,options).unwrap();assert!(prepared.columns()>0);
        let p0_hits=runtime.p0.preparation_profile().shift_cache_hits;let deferred=runtime.apply_pending_correction(&mut space,prepared,0).unwrap();
        assert!(deferred.deferred_for_work_budget);assert_eq!(space.columns(),initial.dimension);assert!(deferred.candidate.is_none());
        let pending=deferred.pending.unwrap();assert!(pending.is_current(&space));
        let result=runtime.apply_pending_correction(&mut space,pending,16).unwrap();assert_eq!(runtime.p0.preparation_profile().shift_cache_hits,p0_hits);
        assert!(!result.accepted&&!result.proposal.accepted);assert!(!result.deferred_for_work_budget);
        let append=result.append.unwrap();assert!(append.retained_columns>0);let corrected=result.candidate.unwrap();
        assert_eq!(corrected.dimension,initial.dimension+append.retained_columns);assert!(!corrected.diagnostics.accepted);
        assert_eq!(&corrected.q[..saved.0.len()],saved.0.as_slice());assert_eq!(&corrected.eq[..saved.1.len()],saved.1.as_slice());assert_eq!(&corrected.fq[..saved.2.len()],saved.2.as_slice());
        assert_eq!(space.statistics().original_image_columns_evaluated,corrected.dimension);
        let count=space.columns();assert!(runtime.correct_workspace(&mut space,&initial,options,16).is_err());assert_eq!(space.columns(),count);
        let mut foreign=runtime.cold_workspace(&modes,&times,8.,DirectionPolicy::Fixed{steps:1}).unwrap();foreign.extend_to(count).unwrap();let same_size=foreign.candidate().unwrap();
        assert!(runtime.correct_workspace(&mut space,&same_size,options,16).is_err());
        for t in 0..times.len(){let rho=runtime.reconstruct_candidate_at(&corrected,t).unwrap();for i in 0..4{assert!((rho[i*4+i].re-corrected.populations[t*4+i].re).abs()<1e-12);}}
    }
    #[test]
    #[ignore="Requires an explicitly supplied original full32 Model fixture"]
    fn preparation_workspace_actual_full32_adaptive_incremental_slice(){
        let started=Instant::now();let(owner,initial,modes)=super::super::super::model_fixture::fresh_owned_for_slice();let model_ms=started.elapsed().as_secs_f64()*1000.;
        let before=Instant::now();let mut runtime=RuntimeCore::from_owned(owner,InitialView{index:initial,real:None,imag:None},P0Policy::EigenPreferred).unwrap();
        let runtime_ms=before.elapsed().as_secs_f64()*1000.;let times=[0.,0.0137,0.125];let before=Instant::now();
        let mut space=runtime.cold_workspace(&modes,&times,8.,DirectionPolicy::Adaptive(AdaptiveConstruction{
            construction_budget_scale:0.005,maximum_inner_steps:12,checkpoint_policy:CheckpointPolicy::Cost})).unwrap();
        let workspace_ms=before.elapsed().as_secs_f64()*1000.;let before=Instant::now();space.extend_to(8).unwrap();let extend8_ms=before.elapsed().as_secs_f64()*1000.;
        let before=Instant::now();let first=space.candidate().unwrap();let check8_ms=before.elapsed().as_secs_f64()*1000.;let old=space.statistics();assert_eq!(first.dimension,8);
        let before=Instant::now();space.extend_to(12).unwrap();let extend4_ms=before.elapsed().as_secs_f64()*1000.;let before=Instant::now();let last=space.candidate().unwrap();
        let check12_ms=before.elapsed().as_secs_f64()*1000.;let now=space.statistics();assert_eq!(last.dimension,12);
        assert_eq!(now.original_image_columns_evaluated-old.original_image_columns_evaluated,4);assert_eq!(now.observation_columns_projected-old.observation_columns_projected,4);
        assert_eq!(now.raw_mass_entries_computed-old.raw_mass_entries_computed,80);
        assert_eq!(&last.q[..first.q.len()],first.q.as_slice());assert_eq!(&last.eq[..first.eq.len()],first.eq.as_slice());assert_eq!(&last.fq[..first.fq.len()],first.fq.as_slice());
        for j in 0..8{for i in 0..8{assert_eq!(last.projected_mass[i+j*12].to_bits(),first.projected_mass[i+j*8].to_bits());assert_eq!(last.projected_pencil[i+j*12].to_bits(),first.projected_pencil[i+j*8].to_bits());}}
        for row in 0..times.len()*32{for j in 0..8{let(a,b)=(first.observation_rows.coefficients()[row*8+j],last.observation_rows.coefficients()[row*12+j]);assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}}
        let mut agreement=0_f64;let mut hermiticity=0_f64;let mut initial_error=0_f64;
        for t in 0..times.len(){let rho=runtime.reconstruct_candidate_at(&last,t).unwrap();for i in 0..32{for j in 0..32{
            hermiticity=hermiticity.max((rho[i*32+j].re-rho[j*32+i].re).hypot(rho[i*32+j].im+rho[j*32+i].im));
            if t==0{let b=runtime.initial().original()[i*32+j];initial_error=initial_error.max((rho[i*32+j].re-b.re).hypot(rho[i*32+j].im-b.im));}
        }let p=last.populations[t*32+i];agreement=agreement.max((p.re-rho[i*32+i].re).hypot(p.im-rho[i*32+i].im));}}
        assert!(agreement<1e-9&&hermiticity<1e-9&&initial_error<1e-9);assert!(!last.diagnostics.accepted);
        let d=&last.diagnostics;let steps=d.directions.iter().map(|r|r.adaptive.inner_steps.to_string()).collect::<Vec<_>>().join(",");
        let strict=d.directions.iter().filter(|r|r.strict_fallback.is_some()).count();let lookahead=d.directions.iter().map(|r|r.adaptive.reused_checkpoint_preconditioner_columns).sum::<usize>();
        let minimum=last.populations.iter().map(|z|z.re).fold(f64::INFINITY,f64::min);let maximum=last.populations.iter().map(|z|z.re).fold(f64::NEG_INFINITY,f64::max);
        let report=format!("{{\"status\":\"construction-only-unqualified\",\"accepted\":false,\"sameWorkspace\":true,\"physicalDimension\":32,\"signedModes\":{},\"checkpoints\":[8,12],\"times\":[0,0.0137,0.125],\"modelPreparationMs\":{model_ms},\"runtimePreparationMs\":{runtime_ms},\"workspacePreparationMs\":{workspace_ms},\"extendTo8Ms\":{extend8_ms},\"check8Ms\":{check8_ms},\"extendNew4Ms\":{extend4_ms},\"check12Ms\":{check12_ms},\"totalNumericMs\":{},\"initialCoordinateMs\":{},\"observationPreparationMs\":{},\"b0FactorMs\":{},\"b0FrameMs\":{},\"shiftFactorMs\":{},\"feedbackColumns\":{},\"originalEFColumns\":{},\"newEFColumnsAfter8\":4,\"observationProjectedColumns\":{},\"newObservationColumnsAfter8\":4,\"rawProjectionEntriesPerMatrix\":{},\"newProjectionEntriesPerMatrixAfter8\":80,\"oldRawBlocksBitwise\":true,\"oldObservationColumnsBitwise\":true,\"fullDensityObservationDifference\":{agreement},\"fullDensityHermiticityError\":{hermiticity},\"fullDensityInitialError\":{initial_error},\"orthogonalityHs\":{},\"enteredRealityRatio\":{},\"populationTraceError\":{},\"populationMinimum\":{minimum},\"populationMaximum\":{maximum},\"adaptiveSteps\":[{steps}],\"strictFallbacks\":{strict},\"reusedLookaheadP\":{lookahead},\"preconditionerCalls\":{},\"originalMProducts\":{},\"errorAcceptancePerformed\":false,\"refinedAnswerOrHistoricalSeedUsed\":false}}",
            modes.len(),started.elapsed().as_secs_f64()*1000.,d.runtime_preparation.initial_ms,d.runtime_preparation.observation_ms,d.p0_preparation.factor_ms,d.p0_preparation.initial_frame_ms,
            d.p0_preparation.shift_factors_ms,d.p0_preparation.feedback_columns,now.original_image_columns_evaluated,now.observation_columns_projected,now.raw_mass_entries_computed,
            d.orthogonality_error_hs,d.maximum_entered_reality_ratio,d.population_trace_error,d.preconditioner_calls,d.original_inner_m_products);
        if let Ok(path)=std::env::var("ATOM_NATIVE_SLICE_REPORT"){std::fs::write(path,&report).unwrap();}println!("{report}");
    }
}
