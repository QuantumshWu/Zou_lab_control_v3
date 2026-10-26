//! Current-request candidate construction with complete original E/F images.
//! Trial directions are numerical proposals; population acceptance is owned
//! by finite_driver and population_control. Trace-Petrov projection, the
//! small action and observable readout use these same original images.
use super::*;
use super::b0_inverse::{P0Batch,P0Policy,RequestP0,RequestPreparationProfile};
use super::coordinates::{InitialCoordinates,InitialView,PhysicalCoordinates,PopulationObservation,PopulationRows};
use super::real_fourier::RealFourierMap;
use super::reduced_system::{ReducedActionDiagnostics,ReducedPreparationDiagnostics};
use super::adaptive_direction::{self,AdaptiveOptions,CheckpointPolicy,DirectionOperator,DirectionReport,FullResidual};
#[path="runtime_workspace.rs"]
pub mod workspace;
#[path="fast_pole.rs"]
pub mod fast_pole;

/// Defect of the REPRESENTED prepared coefficients under Hermitian-Fourier
/// conjugacy. These triangle bounds concern numerical REAL arithmetic only;
/// they are neither a forward model error nor a population-error certificate.
#[derive(Clone,Debug)]
pub(super) struct PreparedRealityBudget{pub e_norm:f64,pub e_defect:f64,pub f_base_defect:f64,
    pub s_defect:f64,pub k_defect:f64,pub nojump_defect:f64,pub recycling_defect:f64,pub gg_action_defect:f64}
fn complex_family_norm(f:&FourierMatrices)->f64{f.values().map(|a|complex_norm(a)).sum()}
fn hilbert_pair_defect(a:&FourierMatrices,b:&FourierMatrices,n:usize,sign:f64)->f64{
    let keys:Modes=a.keys().cloned().chain(b.keys().map(|q|q.iter().map(|x|-x).collect())).collect();let mut total=0.;
    for q in keys{let other:Charge=q.iter().map(|x|-x).collect();let x=a.get(&q);let y=b.get(&other);let mut hs=0_f64;
        for i in 0..n{for j in 0..n{let x=x.map_or(C::default(),|v|v[i*n+j]);let y=y.map_or(C::default(),|v|v[j*n+i]);hs=hs.hypot((x.re-sign*y.re).hypot(x.im+sign*y.im));}}total+=hs;
    }total
}
fn recycling_pair_defect(a:&FourierMatrices,g:usize,e:usize)->f64{
    let keys:Modes=a.keys().cloned().chain(a.keys().map(|q|q.iter().map(|x|-x).collect())).collect();let mut total=0.;
    for q in keys{let other:Charge=q.iter().map(|x|-x).collect();let x=a.get(&q);let y=a.get(&other);let mut hs=0_f64;
        for ga in 0..g{for gb in 0..g{for i in 0..e{for j in 0..e{let x=x.map_or(C::default(),|v|v[(ga*g+gb)*e*e+i*e+j]);let y=y.map_or(C::default(),|v|v[(gb*g+ga)*e*e+j*e+i]);hs=hs.hypot((x.re-y.re).hypot(x.im+y.im));}}}}total+=hs;
    }total
}
impl PreparedRealityBudget{
    pub(super) fn prepare(p:&PrimitivePreparation)->Result<Self,String>{
        let n=p.hilbert.source.n;let g=p.hilbert.source.ground.len();let e=n-g;
        // ||ad S|| <=2 sum||S_q||F. Partner S is -S_-q^dagger.
        let s=2.*complex_family_norm(&p.s);let ds=2.*hilbert_pair_defect(&p.s,&p.s,n,-1.);
        let k=complex_family_norm(&p.k_hat);let dk=recycling_pair_defect(&p.k_hat,g,e);
        let r=complex_family_norm(&p.hilbert.radiation.recycling);let dr=recycling_pair_defect(&p.hilbert.radiation.recycling,g,e);
        let nj=complex_family_norm(&p.no_jump_left)+complex_family_norm(&p.no_jump_right);
        // Conjugacy swaps the executed left/right no-jump factors.
        let dn=2.*hilbert_pair_defect(&p.no_jump_left,&p.no_jump_right,n,1.);
        // L on a GG-supported matrix uses only H[:,G] and H[G,:], not the
        // irrelevant large EE carrier. This covers complete LGG plus LOG.
        let h=&p.hilbert.source.hamiltonian;let keys:Modes=h.keys().cloned().chain(h.keys().map(|q|q.iter().map(|x|-x).collect())).collect();let(mut l,mut dl)=(0.,0.);
        for q in keys{let other:Charge=q.iter().map(|x|-x).collect();let a=h.get(&q);let b=h.get(&other);let(mut columns,mut rows,mut dc,mut dr0)=(0_f64,0_f64,0_f64,0_f64);
            for i in 0..n{for j in 0..n{let x=a.map_or(C::default(),|v|v[i*n+j]);let y=b.map_or(C::default(),|v|v[j*n+i]);let delta=(x.re-y.re).hypot(x.im+y.im);
                if j<g{columns=columns.hypot(x.re.hypot(x.im));dc=dc.hypot(delta);}if i<g{rows=rows.hypot(x.re.hypot(x.im));dr0=dr0.hypot(delta);}
            }}l+=columns+rows;dl+=dc+dr0;
        }
        let j=1.+s+0.5*s*s;let dj=ds*(1.+s);
        let e_norm=(1.+k)*j;let e_defect=(1.+k)*dj+dk*j;
        let f_base_defect=(dn+dr)*j+(nj+r)*dj+dl*k*j+l*(k*dj+dk*j);
        if [s,ds,k,dk,r,dr,nj,dn,l,dl,e_norm,e_defect,f_base_defect].iter().any(|x|!x.is_finite()||*x<0.){return Err("Prepared coefficient REAL-defect budget overflow".into());}
        Ok(Self{e_norm,e_defect,f_base_defect,s_defect:ds,k_defect:dk,nojump_defect:dn,recycling_defect:dr,gg_action_defect:dl})
    }
}
#[derive(Clone,Debug)]
pub(super) struct RealGuardReport{pub passed:bool,pub maximum_ratio:f64,pub maximum_arithmetic_ratio:f64,pub failed_index:usize,pub excess_hs:f64,pub preparation_allowance_hs:f64}
/// The distance from Imag(output) to the ORIGINAL componentwise arithmetic
/// box must fit one preparation HS ball per RHS. A full HS budget is not
/// independently spent on every component. No coefficient is altered.
pub(super) fn check_real_image(value:&[C],envelope:&[f64],rhs:usize,input_hs:&[f64],terms:f64,defect:f64)->Result<RealGuardReport,String>{
    if rhs==0||value.len()!=envelope.len()||value.len()%rhs!=0||input_hs.len()!=rhs||!defect.is_finite()||defect<0.||!terms.is_finite()||terms<0.
        ||value.iter().any(|z|!z.finite())||envelope.iter().any(|x|!x.is_finite()||*x<0.)||input_hs.iter().any(|x|!x.is_finite()||*x<0.){
        return Err("Invalid complete REAL image/preparation budget".into());}
    let rows=value.len()/rhs;let mut report=RealGuardReport{passed:true,maximum_ratio:0.,maximum_arithmetic_ratio:0.,failed_index:0,excess_hs:0.,preparation_allowance_hs:0.};
    for column in 0..rhs{let preparation=0.5*defect*input_hs[column];let mut excess_hs=0_f64;let mut largest=0_f64;let mut at=column;
        if !preparation.is_finite(){return Err("Prepared REAL image allowance overflow".into());}
        for i in 0..rows{let index=i*rhs+column;let imaginary=value[index].im.abs();let arithmetic=256.*f64::EPSILON*terms*envelope[index];let excess=(imaginary-arithmetic).max(0.);
            excess_hs=excess_hs.hypot(excess);if excess>largest{largest=excess;at=index;}
            let raw=if arithmetic==0.{if imaginary==0.{0.}else{f64::INFINITY}}else{imaginary/arithmetic};report.maximum_arithmetic_ratio=report.maximum_arithmetic_ratio.max(raw);
            let allowance=arithmetic+preparation;let ratio=if allowance==0.{if imaginary==0.{0.}else{f64::INFINITY}}else{imaginary/allowance};report.maximum_ratio=report.maximum_ratio.max(ratio);
        }
        let ratio=if preparation==0.{if excess_hs==0.{0.}else{f64::INFINITY}}else{excess_hs/preparation};report.maximum_ratio=report.maximum_ratio.max(ratio);
        if excess_hs>preparation{report.passed=false;report.failed_index=at;report.excess_hs=excess_hs;report.preparation_allowance_hs=preparation;return Ok(report);}
    }Ok(report)
}

pub struct RuntimeCore{
    owned:OwnedPrepared,physical:PhysicalCoordinates,initial:InitialCoordinates,
    observer:Arc<PopulationObservation>,p0:RequestP0,preparation_profile:RuntimePreparationProfile,
}
#[derive(Clone,Debug)]pub struct RuntimePreparationProfile{pub initial_ms:f64,pub observation_ms:f64,pub p0_initial_ms:f64}
#[derive(Clone,Debug)]
pub struct ColdOptions{pub pole:f64,pub maximum_basis:usize,pub proposal_steps:usize}
#[derive(Clone,Copy,Debug)]
pub struct AdaptiveConstruction{pub construction_budget_scale:f64,pub maximum_inner_steps:usize,pub checkpoint_policy:CheckpointPolicy}
#[derive(Clone,Debug)]
pub struct CandidateDiagnostics{
    pub accepted:bool,pub construction_only:bool,pub proposal_policy:&'static str,
    pub requested_basis:usize,pub actual_basis:usize,pub breakdown_unqualified:bool,
    pub orthogonality_error_hs:f64,pub original_initial_projection_error_hs:f64,
    pub initial_real_pull_imaginary_hs:f64,pub maximum_entered_reality_ratio:f64,
    pub maximum_entered_multiplication_only_ratio:f64,pub prepared_e_real_defect_bound:f64,pub prepared_f_real_defect_bound:f64,
    pub preparation_real_budget_is_population_error_bound:bool,
    pub maximum_proposal_pull_imaginary_hs:f64,pub entered_columns:usize,
    pub original_inner_m_products:usize,pub preconditioner_calls:usize,
    pub original_complex_residual_checks:usize,pub directions:Vec<DirectionReport>,
    pub runtime_preparation:RuntimePreparationProfile,pub p0_preparation:RequestPreparationProfile,
    pub domain_setup_ms:f64,pub domain_shift_preparation_ms:f64,
    pub workspace_original_image_evaluations:usize,pub workspace_observation_columns:usize,pub workspace_raw_projection_entries:usize,
    pub workspace_phase_evaluations:usize,pub workspace_phase_preparation_ms:f64,
    pub new_columns_since_prior_candidate:Option<usize>,pub complete_frontier_generation:bool,pub frontier_pending:usize,pub frontier_epoch:usize,
    pub trace_test_overlap:f64,pub mass_trace_identity_error:f64,pub pencil_trace_identity_error:f64,
    pub noncentral_trace_error:f64,pub trace_petrov_test_row_replaced:bool,
    pub population_imaginary_error:f64,pub population_trace_error:f64,
    pub small_preparation:ReducedPreparationDiagnostics,pub small_action:ReducedActionDiagnostics,
}
pub struct Candidate{
    owner:Arc<PrimitivePreparation>,
    workspace_identity:Arc<()>,basis_revision:usize,
    /// Complete REAL basis and original images, each column-major [rows,m].
    pub q:Vec<f64>,pub eq:Vec<f64>,pub fq:Vec<f64>,
    pub rows:usize,pub dimension:usize,pub modes:Vec<Charge>,pub pole:f64,
    /// Unmodified raw projections before trace rotation/test-row replacement.
    pub projected_mass:Vec<f64>,pub projected_pencil:Vec<f64>,
    pub projected_gram:Vec<f64>,pub projected_initial:Vec<f64>,
    pub rotation:Vec<f64>,pub generator:Vec<f64>,pub c0:Vec<f64>,
    /// Unrotated Q coefficients, row-major [column,time].
    pub coefficients:Vec<f64>,pub times:Vec<f64>,pub populations:Vec<C>,
    pub observation_rows:PopulationRows,pub diagnostics:CandidateDiagnostics,
}
struct Domain{
    map:RealFourierMap,plan:OwnedPencil,p0:P0Batch,pole:f64,
    signed:Vec<C>,e:Vec<C>,f:Vec<C>,native:Vec<C>,
    eb:Vec<f64>,fb:Vec<f64>,bound:Vec<f64>,
    maximum_reality_ratio:f64,maximum_arithmetic_ratio:f64,maximum_proposal_imaginary:f64,entered:usize,m_products:usize,p0_calls:usize,strict_checks:usize,
    setup_ms:f64,shift_preparation_ms:f64,
}

impl DirectionOperator for Domain{
    fn apply_m_real(&mut self,x:&[f64])->Result<Vec<f64>,String>{self.original_m_proposal(x)}
    fn precondition(&mut self,x:&[f64])->Result<Vec<f64>,String>{self.precondition_proposal(x)}
    fn full_residual(&mut self,x:&[f64],rhs:&[f64])->Result<FullResidual,String>{
        self.prepare_buffers(1)?;
        self.map.push_into(x,1,&mut self.signed)?;
        // Independent original E/F verification path, not only the real
        // Arnoldi combination or its least-squares residual estimate.
        self.plan.apply_into(&self.signed,1,&mut self.e,&mut self.f)?;
        self.map.push_into(rhs,1,&mut self.signed)?;
        for i in 0..self.e.len(){self.e[i]=C{re:self.signed[i].re-self.pole*self.e[i].re+self.f[i].re,
            im:self.signed[i].im-self.pole*self.e[i].im+self.f[i].im};}
        self.map.pull_complex_into(&self.e,1,&mut self.native)?;self.m_products+=1;self.strict_checks+=1;
        Ok(FullResidual{norm:complex_norm(&self.e),real_residual:self.native.iter().map(|z|z.re).collect()})
    }
}
fn norm(a:&[f64])->f64{a.iter().fold(0_f64,|x,y|x.hypot(*y))}
fn complex_norm(a:&[C])->f64{a.iter().fold(0_f64,|x,z|x.hypot(z.re.hypot(z.im)))}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(x,y)|x*y).sum()}
fn transpose(a:&[f64],rows:usize,cols:usize)->Vec<f64>{(0..rows*cols).map(|i|a[(i%cols)*rows+i/cols]).collect()}
fn maximum(a:&[f64])->f64{a.iter().map(|x|x.abs()).fold(0_f64,f64::max)}
fn finite(a:&[f64],label:&str)->Result<(),String>{if a.iter().any(|x|!x.is_finite()){Err(format!("Nonfinite native candidate {label}"))}else{Ok(())}}

impl Domain{
    fn real_coordinate_label(&self,row:usize)->String{
        let n=self.map.physical_dimension();let d=n*n;
        for block in self.map.harmonics(){let width=if block.has_imaginary_block{2*d}else{d};
            if row<block.offset||row>=block.offset+width{continue;}
            let local=row-block.offset;let component=local%d;let quadrature=if !block.has_imaginary_block{"central"}else if local<d{"real-harmonic"}else{"imag-harmonic"};
            let mut label=if component<n{format!("population({component})")}else{String::new()};let mut at=n;
            for i in 0..n{for j in i+1..n{if component==at{label=format!("real({i},{j})");}if component==at+1{label=format!("imag({i},{j})");}at+=2;}}
            return format!("mode={:?},quadrature={quadrature},coordinate={label}",block.index);
        }format!("unmapped-row={row}")
    }
    fn preparation_reality_summary(&self)->String{
        let p=self.plan.primitives();let n=p.hilbert.source.n;let g=p.hilbert.source.ground.len();let e=n-g;
        let hilbert_defect=|family:&FourierMatrices,anti:bool|{let(mut largest,mut hs)=(0_f64,0_f64);
            for(q,a)in family{let opposite:Charge=q.iter().map(|&x|-x).collect();let partner=family.get(&opposite);
                for i in 0..n{for j in 0..n{let b=partner.map_or(C::default(),|v|v[j*n+i]);let sign=if anti{-1.}else{1.};
                    let error=(a[i*n+j].re-sign*b.re).hypot(a[i*n+j].im+sign*b.im);largest=largest.max(error);hs=hs.hypot(error);}}}
            (largest,hs)};
        let super_defect=|family:&FourierMatrices|{let(mut largest,mut hs)=(0_f64,0_f64);
            for(q,a)in family{let opposite:Charge=q.iter().map(|&x|-x).collect();let partner=family.get(&opposite);
                for ga in 0..g{for gb in 0..g{for i in 0..e{for j in 0..e{let at=(ga*g+gb)*e*e+i*e+j;
                    let b=partner.map_or(C::default(),|v|v[(gb*g+ga)*e*e+j*e+i]);let error=(a[at].re-b.re).hypot(a[at].im+b.im);
                    largest=largest.max(error);hs=hs.hypot(error);
                }}}}}
            (largest,hs)};
        let(smax,shs)=hilbert_defect(&p.s,true);let(hmax,hhs)=hilbert_defect(&p.hilbert.source.hamiltonian,false);
        let(kmax,khs)=super_defect(&p.k_hat);let(rmax,rhs)=super_defect(&p.hilbert.radiation.recycling);
        format!("prep[S_pair_max={smax:e},S_pair_hs={shs:e},K_pair_max={kmax:e},K_pair_hs={khs:e},H_pair_max={hmax:e},H_pair_hs={hhs:e},R_pair_max={rmax:e},R_pair_hs={rhs:e},B_unitary_hs={:e},K_equation_hs={:e},K_trace_hs={:e},static={}]",
            p.hilbert.diagnostics.basis_unitarity_defect_hs,p.shear_diagnostics.unprojected_homological_residual_hs,p.shear_diagnostics.final_trace_functional_hs,cfg!(feature="static-numerics"))
    }
    fn prepare_buffers(&mut self,rhs:usize)->Result<(),String>{
        let length=product(&[self.map.dimension(),rhs])?;
        for value in[&mut self.signed,&mut self.e,&mut self.f,&mut self.native]{value.resize(length,C::default());}
        for value in[&mut self.eb,&mut self.fb,&mut self.bound]{value.resize(length,0.);}Ok(())
    }
    fn entered_images(&mut self,x:&[f64])->Result<(Vec<f64>,Vec<f64>),String>{
        self.entered_images_columns(x,1)
    }
    fn entered_images_columns(&mut self,x:&[f64],rhs:usize)->Result<(Vec<f64>,Vec<f64>),String>{
        self.prepare_buffers(rhs)?;self.map.push_into(x,rhs,&mut self.signed)?;self.plan.apply_into(&self.signed,rhs,&mut self.e,&mut self.f)?;
        let mut outputs=Vec::with_capacity(2);
        for value in[&self.e,&self.f]{
            self.map.pull_complex_into(value,rhs,&mut self.native)?;
            outputs.push(self.native.iter().map(|z|z.re).collect());
        }
        self.entered+=rhs;Ok((outputs.remove(0),outputs.remove(0)))
    }
    fn original_m_proposal(&mut self,x:&[f64])->Result<Vec<f64>,String>{
        self.prepare_buffers(1)?;
        self.map.push_into(x,1,&mut self.signed)?;self.plan.apply_shifted_into(&self.signed,1,C{re:self.pole,im:0.},&mut self.e)?;
        self.map.pull_complex_into(&self.e,1,&mut self.native)?;
        let imaginary=self.native.iter().fold(0_f64,|v,z|v.hypot(z.im));self.maximum_proposal_imaginary=self.maximum_proposal_imaginary.max(imaginary);self.m_products+=1;
        Ok(self.native.iter().map(|z|z.re).collect())
    }
    fn precondition_proposal(&mut self,x:&[f64])->Result<Vec<f64>,String>{
        self.prepare_buffers(1)?;
        self.map.push_into(x,1,&mut self.signed)?;self.p0.apply_into(&self.signed,1,&mut self.e)?;
        let mut value=vec![0.;x.len()];let diagnostic=self.map.pull_real_proposal_into(&self.e,1,&mut value)?;
        self.maximum_proposal_imaginary=self.maximum_proposal_imaginary.max(diagnostic.imaginary_hs_by_column[0]);self.p0_calls+=1;Ok(value)
    }
    /// A fixed linear numerical proposal, not a completed/accepted solve.
    fn direction(&mut self,rhs:&[f64],steps:usize)->Result<Vec<f64>,String>{
        let mut value=vec![0.;rhs.len()];let mut residual=rhs.to_vec();
        for step in 0..steps{if step>0{let image=self.original_m_proposal(&value)?;for i in 0..rhs.len(){residual[i]=rhs[i]-image[i];}}
            let correction=self.precondition_proposal(&residual)?;for i in 0..rhs.len(){value[i]+=correction[i];}finite(&value,"fixed-step proposal")?;
        }Ok(value)
    }
}

impl RuntimeCore{
    pub fn from_owned(owned:OwnedPrepared,initial:InitialView<'_>,p0_policy:P0Policy)->Result<Self,String>{
        let physical=owned.physical_coordinates();let began=Instant::now();let initial=physical.initial(initial)?;let initial_ms=began.elapsed().as_secs_f64()*1000.;
        let began=Instant::now();let observer=Arc::new(physical.population_observation()?);let observation_ms=began.elapsed().as_secs_f64()*1000.;
        let began=Instant::now();let p0=owned.prepare_p0(p0_policy)?;let p0_initial_ms=began.elapsed().as_secs_f64()*1000.;
        Ok(Self{owned,physical,initial,observer,p0,preparation_profile:RuntimePreparationProfile{initial_ms,observation_ms,p0_initial_ms}})
    }
    pub fn prepare(model:&ModelView<'_>,initial:InitialView<'_>,phases:PhaseBasis,gauge:GaugePolicy<'_>,kernel:KernelPolicy,p0:P0Policy)->Result<Self,String>{
        Self::from_owned(OwnedPrepared::prepare(model,phases,gauge,kernel)?,initial,p0)
    }
    pub fn prepared(&self)->&OwnedPrepared{&self.owned}
    pub fn initial(&self)->&InitialCoordinates{&self.initial}
    pub fn prepared_shift_count(&self)->usize{self.p0.prepared_shift_count()}
    pub fn preparation_profile(&self)->&RuntimePreparationProfile{&self.preparation_profile}
    /// Same-request operator context for independent domain/proposal modules.
    /// Shift preparation may grow this cache; existing P0Batch factors remain
    /// immutable and no previously solved vector is stored here.
    pub fn operator_context(&mut self)->(&OwnedPrepared,&mut RequestP0){(&self.owned,&mut self.p0)}
    fn domain(&mut self,modes:&[Charge],pole:f64,max_rhs:usize)->Result<Domain,String>{
        let began=Instant::now();
        let map=self.owned.real_fourier_map(modes)?;let plan=self.owned.pencil(modes,modes,max_rhs)?;
        let phases=&self.owned.primitives.hilbert.source.phases;
        let shifts=modes.iter().map(|q|phases.frequency(q).map(|f|C{re:pole,im:f.hi+f.lo})).collect::<Result<Vec<_>,_>>()?;
        let before=Instant::now();let p0=self.p0.prepare_batch(&shifts,max_rhs)?;let shift_preparation_ms=before.elapsed().as_secs_f64()*1000.;let rows=map.dimension();
        Ok(Domain{map,plan,p0,pole,signed:vec![C::default();rows],e:vec![C::default();rows],f:vec![C::default();rows],native:vec![C::default();rows],
            eb:vec![0.;rows],fb:vec![0.;rows],bound:vec![0.;rows],maximum_reality_ratio:0.,maximum_arithmetic_ratio:0.,maximum_proposal_imaginary:0.,entered:0,m_products:0,p0_calls:0,strict_checks:0,
            setup_ms:began.elapsed().as_secs_f64()*1000.,shift_preparation_ms})
    }
    /// No epsilon or accept action is accepted by this constructor. The result
    /// is a candidate for an external error/work policy, never a full solution.
    pub fn cold_candidate(&mut self,modes:&[Charge],times:&[f64],options:ColdOptions)->Result<Candidate,String>{
        self.construct_candidate(modes,times,options,None)
    }
    /// One-shot convenience for diagnostics. A final adaptive checkpoint loop
    /// must keep cold_workspace() and call extend_to(), not call this method
    /// repeatedly. The returned candidate is still UNQUALIFIED.
    pub fn adaptive_candidate(&mut self,modes:&[Charge],times:&[f64],options:ColdOptions,adaptive:AdaptiveConstruction)->Result<Candidate,String>{
        if !adaptive.construction_budget_scale.is_finite()||adaptive.construction_budget_scale<=0.||adaptive.maximum_inner_steps==0{
            return Err("Invalid adaptive construction work budget".into());}
        self.construct_candidate(modes,times,options,Some(adaptive))
    }
    fn construct_candidate(&mut self,modes:&[Charge],times:&[f64],options:ColdOptions,adaptive:Option<AdaptiveConstruction>)->Result<Candidate,String>{
        let n=self.owned.primitives.hilbert.source.n;
        if options.maximum_basis==0||options.maximum_basis>product(&[modes.len(),n,n])?{return Err("Invalid native basis checkpoint".into());}
        let policy=if let Some(value)=adaptive{workspace::DirectionPolicy::Adaptive(value)}else{workspace::DirectionPolicy::Fixed{steps:options.proposal_steps}};
        let mut work=self.cold_workspace(modes,times,options.pole,policy)?;
        work.extend_to(options.maximum_basis)?;work.candidate()
    }
    pub fn reconstruct_candidate_at(&self,candidate:&Candidate,time_index:usize)->Result<Vec<C>,String>{
        if !Arc::ptr_eq(&self.owned.primitives,&candidate.owner)||time_index>=candidate.times.len(){return Err("Candidate belongs to another current Model or time grid".into());}
        let m=candidate.dimension;let nt=candidate.times.len();let mut native=vec![0.;candidate.rows];
        for j in 0..m{let coefficient=candidate.coefficients[j*nt+time_index];for i in 0..candidate.rows{native[i]+=candidate.q[i+j*candidate.rows]*coefficient;}}
        let map=self.owned.real_fourier_map(&candidate.modes)?;let mut signed=vec![C::default();candidate.rows];map.push_into(&native,1,&mut signed)?;
        self.physical.field_at(&candidate.modes,&signed,candidate.times[time_index])
    }
}

impl Candidate{
    pub fn belongs_to(&self,prepared:&OwnedPrepared)->bool{Arc::ptr_eq(&self.owner,&prepared.primitives)}
    pub fn workspace_identity_key(&self)->usize{Arc::as_ptr(&self.workspace_identity)as usize}
    pub fn basis_revision(&self)->usize{self.basis_revision}
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    fn preparation_real_budget_covers_known_coefficient_defect_without_changing_physics(){
        let owner=super::super::tests::fixture();let lap=Arc::clone(&owner.preparation_lapack);
        let mut primitive=match Arc::try_unwrap(owner.primitives){Ok(p)=>p,Err(_)=>panic!("fixture owner was unexpectedly shared")};
        let h=primitive.hilbert.source.hamiltonian.clone();let channels=primitive.hilbert.source.collapse.clone();
        let g=primitive.hilbert.source.ground.len();let e=primitive.hilbert.source.excited.len();let n=g+e;
        let key=primitive.k_hat.keys().find(|q|q.iter().copied().find(|&x|x!=0).is_some_and(|x|x>0)).unwrap().clone();
        // Known numerical preparation perturbation in ONE signed auxiliary
        // coefficient. Keep GG trace and every original physical H/C unchanged.
        let delta=1e-11;for(row,amount)in[(0,delta),(g*g-1,-delta)]{
            primitive.k_hat.get_mut(&key).unwrap()[row*e*e].re+=amount;
            let selected=primitive.gg_rows.iter().position(|&r|r==row).unwrap();primitive.selected_k.get_mut(&key).unwrap()[selected*e*e].re+=amount;
        }
        let prepared=OwnedPrepared::from_primitives(primitive,KernelPolicy::Csr,lap).unwrap();
        for(q,a)in &h{for(x,y)in a.iter().zip(&prepared.primitives().hilbert.source.hamiltonian[q]){assert_eq!(x.re.to_bits(),y.re.to_bits());assert_eq!(x.im.to_bits(),y.im.to_bits());}}
        for(a,b)in channels.iter().zip(&prepared.primitives().hilbert.source.collapse){for(q,values)in a{for(x,y)in values.iter().zip(&b[q]){assert_eq!(x.re.to_bits(),y.re.to_bits());assert_eq!(x.im.to_bits(),y.im.to_bits());}}}
        let opposite=key.iter().map(|&k|-k).collect();let modes=[vec![0;key.len()],key,opposite];let map=prepared.real_fourier_map(&modes).unwrap();let rows=map.dimension();
        let mut plan=prepared.pencil(&modes,&modes,1).unwrap();let mut input=vec![0.;rows];input[g]=1.;
        let mut signed=vec![C::default();rows];map.push_into(&input,1,&mut signed).unwrap();let mut mass=signed.clone();let mut pencil=signed.clone();plan.apply_into(&signed,1,&mut mass,&mut pencil).unwrap();
        let(mut eb,mut fb,mut bound)=(vec![0.;rows],vec![0.;rows],vec![0.;rows]);plan.action_magnitudes_into(&signed,1,&mut eb,&mut fb).unwrap();map.pull_envelope_into(&eb,1,&mut bound).unwrap();
        let mut actual=vec![C::default();rows];map.pull_complex_into(&mass,1,&mut actual).unwrap();let terms=plan.diagnostics().roundoff_terms as f64;let input_hs=[complex_norm(&signed)];
        let old=check_real_image(&actual,&bound,1,&input_hs,terms,0.).unwrap();assert!(!old.passed);
        let budget=plan.real_preparation_defects().0;assert!(budget>0.);let qualified=check_real_image(&actual,&bound,1,&input_hs,terms,budget).unwrap();
        assert!(qualified.passed);assert!(qualified.maximum_ratio<=1.);assert!(qualified.maximum_arithmetic_ratio>1.);
        actual[old.failed_index].im+=1e-4;assert!(!check_real_image(&actual,&bound,1,&input_hs,terms,budget).unwrap().passed);
    }
    #[test]
    fn preparation_real_budget_is_one_hs_budget_and_rejects_extra_error(){
        let value=[C{re:0.,im:0.8},C{re:0.,im:0.8}];
        // Each component individually fits1, but their HS excess does not:
        // the same preparation budget must not be spent twice.
        let report=check_real_image(&value,&[0.,0.],1,&[1.],32.,2.).unwrap();assert!(!report.passed);assert!(report.excess_hs>report.preparation_allowance_hs);
        assert!(!check_real_image(&value,&[0.,0.],1,&[0.],32.,2.).unwrap().passed);
        let good=[C{re:1.,im:0.}];assert!(check_real_image(&good,&[1.],1,&[1.],32.,0.).unwrap().passed);
    }
    #[test]
    fn preparation_runtime_cold_vertical_slice_keeps_original_images_and_unqualified_status(){
        let owned=super::super::tests::fixture();let mut runtime=RuntimeCore::from_owned(owned,InitialView{index:0,real:None,imag:None},P0Policy::EigenPreferred).unwrap();
        let modes=[vec![0],vec![1],vec![-1]];let times=[0.,0.03,0.1];let candidate=runtime.cold_candidate(&modes,&times,ColdOptions{pole:8.,maximum_basis:4,proposal_steps:2}).unwrap();
        assert!(!candidate.diagnostics.accepted&&candidate.diagnostics.construction_only);assert_eq!(candidate.diagnostics.entered_columns,candidate.dimension);
        assert_eq!(candidate.q.len(),candidate.rows*candidate.dimension);assert!(candidate.diagnostics.orthogonality_error_hs<1e-12);assert!(candidate.diagnostics.original_initial_projection_error_hs<1e-12);
        let map=runtime.owned.real_fourier_map(&modes).unwrap();let mut plan=runtime.owned.pencil(&modes,&modes,1).unwrap();
        for j in 0..candidate.dimension{let q=&candidate.q[j*candidate.rows..(j+1)*candidate.rows];let mut signed=vec![C::default();candidate.rows];let mut e=signed.clone();let mut f=e.clone();let mut native=e.clone();
            map.push_into(q,1,&mut signed).unwrap();plan.apply_into(&signed,1,&mut e,&mut f).unwrap();
            for(value,expected)in[(&e,&candidate.eq),(&f,&candidate.fq)]{map.pull_complex_into(value,1,&mut native).unwrap();for i in 0..candidate.rows{assert_eq!(native[i].re.to_bits(),expected[i+j*candidate.rows].to_bits());}}
        }
        for t in 0..times.len(){let rho=runtime.reconstruct_candidate_at(&candidate,t).unwrap();for i in 0..4{let p=candidate.populations[t*4+i];let expected=rho[i*4+i];assert!((p.re-expected.re).hypot(p.im-expected.im)<1e-12);}}
        let rho0=runtime.reconstruct_candidate_at(&candidate,0).unwrap();assert!((rho0[0].re-1.).abs()<1e-12);
        let shifts=runtime.prepared_shift_count();let again=runtime.cold_candidate(&modes,&times,ColdOptions{pole:8.,maximum_basis:3,proposal_steps:1}).unwrap();assert_eq!(runtime.prepared_shift_count(),shifts);assert!(!again.diagnostics.accepted);
    }
    #[test]
    fn preparation_runtime_adaptive_direction_keeps_true_images_and_separate_qualification(){
        let owned=super::super::tests::fixture();let mut runtime=RuntimeCore::from_owned(owned,InitialView{index:0,real:None,imag:None},P0Policy::EigenPreferred).unwrap();
        let candidate=runtime.adaptive_candidate(&[vec![0],vec![1],vec![-1]],&[0.,0.03,0.1],ColdOptions{pole:8.,maximum_basis:4,proposal_steps:2},
            AdaptiveConstruction{construction_budget_scale:0.005,maximum_inner_steps:6,checkpoint_policy:CheckpointPolicy::Cost}).unwrap();
        assert!(!candidate.diagnostics.accepted);assert_eq!(candidate.diagnostics.entered_columns,candidate.dimension);
        assert_eq!(candidate.diagnostics.directions.len(),candidate.dimension-1);
        for report in &candidate.diagnostics.directions{if let Some(strict)=&report.strict_fallback{assert!(strict.strict_residual_passed);assert!(strict.relative_full_m_residual<=3e-9);}
            else{assert!(report.adaptive.proposal_passed);assert!(!report.adaptive.linear_solution_confirmed);}
            assert!(!report.adaptive.direction_error_certified&&!report.adaptive.population_error_certified);}
        for t in 0..candidate.times.len(){let physical=runtime.reconstruct_candidate_at(&candidate,t).unwrap();for i in 0..4{assert!((physical[i*4+i].re-candidate.populations[t*4+i].re).abs()<1e-12);}}
    }
    #[test]
    #[ignore="Requires an explicitly supplied original full32 Model fixture"]
    fn preparation_runtime_actual_full32_cold_construction_only(){
        actual_full32_slice(false);
    }
    #[test]
    #[ignore="Requires an explicitly supplied original full32 Model fixture"]
    fn preparation_runtime_full32_adaptive_slice_construction_only(){
        actual_full32_slice(true);
    }
    fn actual_full32_slice(adaptive:bool){
        let began=Instant::now();let(owner,initial,modes)=super::super::model_fixture::fresh_owned_for_slice();let preparation_ms=began.elapsed().as_secs_f64()*1000.;
        let runtime_begin=Instant::now();let mut runtime=RuntimeCore::from_owned(owner,InitialView{index:initial,real:None,imag:None},P0Policy::EigenPreferred).unwrap();
        let output_p0_ms=runtime_begin.elapsed().as_secs_f64()*1000.;let construction=Instant::now();
        let times=[0.,0.0137,0.125];let options=ColdOptions{pole:8.,maximum_basis:8,proposal_steps:2};
        let candidate=if adaptive{runtime.adaptive_candidate(&modes,&times,options,AdaptiveConstruction{construction_budget_scale:0.005,maximum_inner_steps:12,checkpoint_policy:CheckpointPolicy::Cost}).unwrap()}
            else{runtime.cold_candidate(&modes,&times,options).unwrap()};
        let construction_ms=construction.elapsed().as_secs_f64()*1000.;let n=runtime.owned.primitives.hilbert.source.n;
        let mut agreement=0_f64;let mut hermitian=0_f64;let mut initial_error=0_f64;
        for t in 0..times.len(){let rho=runtime.reconstruct_candidate_at(&candidate,t).unwrap();for i in 0..n{for j in 0..n{
            hermitian=hermitian.max((rho[i*n+j].re-rho[j*n+i].re).hypot(rho[i*n+j].im+rho[j*n+i].im));
            if t==0{let target=runtime.initial.original()[i*n+j];initial_error=initial_error.max((rho[i*n+j].re-target.re).hypot(rho[i*n+j].im-target.im));}
        }let expected=rho[i*n+i];let actual=candidate.populations[t*n+i];agreement=agreement.max((actual.re-expected.re).hypot(actual.im-expected.im));}}
        assert!(!candidate.diagnostics.accepted&&candidate.diagnostics.construction_only);assert_eq!(candidate.dimension,8);
        assert!(agreement<1e-9&&hermitian<1e-9&&initial_error<1e-9);
        let minimum=candidate.populations.iter().map(|z|z.re).fold(f64::INFINITY,f64::min);let maximum=candidate.populations.iter().map(|z|z.re).fold(f64::NEG_INFINITY,f64::max);
        let d=&candidate.diagnostics;
        let fixed_steps=if adaptive{"null"}else{"2"};
        let mut report=format!("{{\"status\":\"construction-only-unqualified\",\"physicalDimension\":{n},\"signedModes\":{},\"basis\":{},\"fixedProposalSteps\":{fixed_steps},\"adaptiveDirections\":{adaptive},\"times\":[0,0.0137,0.125],\"accepted\":false,\"modelPreparationMs\":{preparation_ms},\"outputAndP0PreparationMs\":{output_p0_ms},\"constructionMs\":{construction_ms},\"totalNumericMs\":{},\"orthogonalityHs\":{},\"enteredRealityRatio\":{},\"originalInitialProjectionHs\":{},\"traceMassError\":{},\"tracePencilError\":{},\"noncentralTraceError\":{},\"populationTraceError\":{},\"populationImaginaryError\":{},\"populationMinimum\":{minimum},\"populationMaximum\":{maximum},\"fullDensityObservationDifference\":{agreement},\"fullDensityHermiticityError\":{hermitian},\"fullDensityInitialError\":{initial_error},\"preconditionerCalls\":{},\"innerOriginalMProducts\":{},\"enteredColumns\":{},\"preparedUniqueShifts\":{},\"refinedAnswerOrHistoricalSeedUsed\":false}}",
            modes.len(),candidate.dimension,began.elapsed().as_secs_f64()*1000.,d.orthogonality_error_hs,d.maximum_entered_reality_ratio,d.original_initial_projection_error_hs,
            d.mass_trace_identity_error,d.pencil_trace_identity_error,d.noncentral_trace_error,d.population_trace_error,d.population_imaginary_error,d.preconditioner_calls,d.original_inner_m_products,d.entered_columns,runtime.prepared_shift_count());
        let inner_steps=d.directions.iter().map(|r|r.adaptive.inner_steps.to_string()).collect::<Vec<_>>().join(",");
        let strict=d.directions.iter().filter(|r|r.strict_fallback.is_some()).count();let reused=d.directions.iter().map(|r|r.adaptive.reused_checkpoint_preconditioner_columns).sum::<usize>();
        report.pop();report.push_str(&format!(",\"initialCoordinateMs\":{},\"observationPreparationMs\":{},\"b0FactorMs\":{},\"b0InitialFrameMs\":{},\"b0LazySchurFrameMs\":{},\"shiftFactorsMs\":{},\"feedbackColumns\":{},\"domainSetupMs\":{},\"domainShiftPreparationMs\":{},\"adaptiveStepCounts\":[{inner_steps}],\"strictFallbacks\":{strict},\"reusedLookaheadP\":{reused},\"strictOriginalComplexChecks\":{},\"errorAcceptancePerformed\":false}}",
            d.runtime_preparation.initial_ms,d.runtime_preparation.observation_ms,d.p0_preparation.factor_ms,d.p0_preparation.initial_frame_ms,
            d.p0_preparation.lazy_schur_frame_ms,d.p0_preparation.shift_factors_ms,d.p0_preparation.feedback_columns,d.domain_setup_ms,d.domain_shift_preparation_ms,d.original_complex_residual_checks));
        if let Ok(path)=std::env::var("ATOM_NATIVE_SLICE_REPORT"){std::fs::write(path,&report).unwrap();}println!("{report}");
    }
}
