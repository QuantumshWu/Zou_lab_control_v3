//! Exact current-request transfer of Q/EQ/FQ to a strictly larger domain.
//!
//! Old REAL harmonic blocks are copied bitwise. Only original-kernel images
//! with input=old domain and output=new shell are evaluated. The shell need
//! not contain zero: its complex images/envelopes are embedded into the full
//! new signed domain before the complete REAL map is applied. No intermediate
//! return path, source column, physical coefficient or Fourier partner is cut.
use super::*;
use super::runtime_core::Candidate;
use super::real_fourier::{RealFourierMap,HarmonicBlock};
use super::coordinates::PopulationRows;

#[derive(Clone,Copy,Debug)]
pub struct TransferOptions{
    pub maximum_rhs:usize,
    /// Chunk scratch only: excludes caller inputs, returned Q/E/F, shared
    /// coefficient/observation owners and opaque vendor-BLAS workspace.
    pub scratch_bytes:usize,
}
impl Default for TransferOptions{
    fn default()->Self{Self{maximum_rhs:4,scratch_bytes:256*1024*1024}}
}

#[derive(Clone,Debug)]
pub struct TransferDiagnostics{
    pub old_modes:usize,pub new_modes:usize,pub shell_modes:usize,pub columns:usize,
    pub old_image_rows_recomputed:bool,pub zero_padded_input_products_computed:bool,
    pub complete_intermediate_return_paths:bool,pub old_real_blocks_bitwise_retained:bool,
    pub original_observation_rows_shared:bool,pub population_answers_reused:bool,
    pub same_request_owner:bool,pub same_pole:bool,pub same_time_grid:bool,
    pub zero_lift_is_new_convergence_evidence:bool,
    pub rhs_chunk_size:usize,pub chunks:usize,pub scratch_budget_bytes:usize,
    pub scratch_upper_bytes_per_rhs:usize,pub peak_measured_scratch_bytes:usize,
    pub returned_image_bytes:usize,pub maximum_reality_allowance_ratio:f64,
    pub maximum_multiplication_only_ratio:f64,pub prepared_e_conjugacy_defect_bound:f64,pub prepared_f_conjugacy_defect_bound:f64,
    pub preparation_budget_is_population_error_bound:bool,
    pub native_shell_action_ms:f64,pub envelope_and_pull_ms:f64,pub elapsed_ms:f64,
}

/// This is a trial/image seed, not a newly propagated or accepted candidate.
pub struct TransferredImages{
    owner:Arc<PrimitivePreparation>,
    /// Complete column-major REAL [rows,columns] arrays.
    pub q:Vec<f64>,pub eq:Vec<f64>,pub fq:Vec<f64>,
    pub rows:usize,pub columns:usize,pub modes:Vec<Charge>,pub pole:f64,
    /// Old-old raw projections are mathematically unchanged by zero lift.
    pub projected_mass:Vec<f64>,pub projected_pencil:Vec<f64>,
    /// Immutable same-Q/time observation coefficients, not saved populations.
    pub observation_rows:PopulationRows,
    pub diagnostics:TransferDiagnostics,
}
impl TransferredImages{
    pub fn belongs_to(&self,prepared:&OwnedPrepared)->bool{Arc::ptr_eq(&self.owner,&prepared.primitives)}
}

fn sum_size(values:&[usize])->Result<usize,String>{
    values.iter().try_fold(0usize,|sum,&value|sum.checked_add(value).ok_or_else(||"Transfer allocation overflow".into()))
}
fn same_times(a:&[f64],b:&[f64])->bool{
    !a.is_empty()&&a.len()==b.len()&&a.iter().zip(b).all(|(a,b)|a.is_finite()&&a.to_bits()==b.to_bits())
}
fn block_width(block:&HarmonicBlock,d:usize)->Result<usize,String>{product(&[d,if block.has_imaginary_block{2}else{1}])}
fn lift_real(old:&[f64],old_map:&RealFourierMap,new_map:&RealFourierMap,columns:usize)->Result<Vec<f64>,String>{
    let old_rows=old_map.dimension();let new_rows=new_map.dimension();let d=product(&[old_map.physical_dimension(),old_map.physical_dimension()])?;
    if old.len()!=product(&[old_rows,columns])?||old.iter().any(|x|!x.is_finite()){return Err("Invalid complete old REAL image columns".into());}
    let mut output=vec![0.;product(&[new_rows,columns])?];
    for block in old_map.harmonics(){
        let target=new_map.harmonics().iter().find(|h|h.index==block.index).ok_or("Old REAL harmonic absent from expanded domain")?;
        if target.has_imaginary_block!=block.has_imaginary_block{return Err("REAL harmonic pairing changed during transfer".into());}
        let width=block_width(block,d)?;
        for j in 0..columns{
            output[j*new_rows+target.offset..j*new_rows+target.offset+width]
                .copy_from_slice(&old[j*old_rows+block.offset..j*old_rows+block.offset+width]);
        }
    }Ok(output)
}

fn scratch_upper_per_rhs(shell:&OwnedPencil,old_rows:usize,new_rows:usize,shell_rows:usize)->Result<usize,String>{
    let p=&shell.plan;
    // Exact core workspace element counts; packed BLAS buffers can retain
    // at most their growth slack (two times each largest request plus8).
    let base=sum_size(&[product(&[p.ni,p.d])?,product(&[p.nd,p.d])?,product(&[p.nj,p.d])?,
        product(&[p.nj,p.e,p.e])?,product(&[p.nz,p.rank])?,product(&[2,p.no,p.d])?])?;
    let(mut packed,mut result)=(0,0);
    if p.lapack.is_some(){for stage in &p.stages[4..]{for link in stage{
        let coefficient=&p.coefficients[link.coefficient];
        if !coefficient.dense.is_empty(){packed=packed.max(product(&[coefficient.cols,link.source.len()])?);
            result=result.max(product(&[coefficient.rows,link.source.len()])?);}
    }}}
    let internal=product(&[sum_size(&[base,product(&[2,packed])?,product(&[2,result])?,16])?,24])?;
    let external=sum_size(&[product(&[old_rows,24])?,product(&[shell_rows,48])?,product(&[new_rows,48])?])?;
    sum_size(&[internal,external])
}

impl OwnedPrepared{
    pub fn transfer_candidate_images(&self,old:&Candidate,new_modes:&[Charge],times:&[f64],pole:f64,
                                     options:TransferOptions)->Result<TransferredImages,String>{
        let began=Instant::now();
        if !old.belongs_to(self){return Err("Domain transfer requires the SAME live current-request coefficient owner, not merely an equal Model".into());}
        if !pole.is_finite()||pole<=0.||old.pole.to_bits()!=pole.to_bits(){return Err("Domain transfer pole changed".into());}
        if !same_times(&old.times,times)||!same_times(old.observation_rows.times(),times){return Err("Domain transfer requested time grid changed".into());}
        if options.maximum_rhs==0||options.scratch_bytes==0||old.dimension==0{return Err("Positive complete column/chunk budgets required".into());}
        let old_map=self.real_fourier_map(&old.modes)?;let new_map=self.real_fourier_map(new_modes)?;
        let axes=self.primitives.hilbert.source.phases.axes.len();let old_set=mode_set(&old.modes,axes)?;let new_set=mode_set(new_modes,axes)?;
        if !old_set.is_subset(&new_set)||old_set.len()>=new_set.len(){return Err("Image transfer requires strict nested Fourier domain inclusion".into());}
        let n=self.primitives.hilbert.source.n;let d=product(&[n,n])?;let columns=old.dimension;
        let old_rows=old_map.dimension();let new_rows=new_map.dimension();
        if old.rows!=old_rows||old.observation_rows.columns()!=columns||old.observation_rows.coefficients().len()!=product(&[times.len(),n,columns])?
            ||old.diagnostics.entered_columns!=columns||old.projected_mass.len()!=product(&[columns,columns])?||old.projected_pencil.len()!=product(&[columns,columns])?{
            return Err("Old candidate does not contain all current trial/image/observation columns".into());}
        let q=lift_real(&old.q,&old_map,&new_map,columns)?;
        let mut eq=lift_real(&old.eq,&old_map,&new_map,columns)?;let mut fq=lift_real(&old.fq,&old_map,&new_map,columns)?;
        let shell_modes:Vec<_>=new_modes.iter().filter(|q|!old_set.contains(*q)).cloned().collect();
        let shell_set:Modes=shell_modes.iter().cloned().collect();
        for q in &shell_set{let negative=q.iter().map(|&k|k.checked_neg().ok_or("Shell partner overflow")).collect::<Result<Charge,_>>()?;
            if !shell_set.contains(&negative){return Err("New shell omitted a signed Fourier partner".into());}}
        let shell_rows=product(&[shell_modes.len(),d])?;
        let mut shell=self.pencil(&old.modes,&shell_modes,options.maximum_rhs)?;
        let per=scratch_upper_per_rhs(&shell,old_rows,new_rows,shell_rows)?;
        if per>options.scratch_bytes{return Err(format!("Transfer scratch budget cannot hold one complete RHS: {per} bytes"));}
        let chunk=columns.min(options.maximum_rhs).min(options.scratch_bytes/per).max(1);
        let signed_at:BTreeMap<_,_>=new_modes.iter().enumerate().map(|(i,q)|(q,i)).collect();
        let targets:Vec<_>=shell_modes.iter().map(|q|signed_at[q]).collect();
        let shell_blocks:Vec<_>=new_map.harmonics().iter().filter(|h|shell_set.contains(&h.index)).collect();
        let mut packed=vec![0.;product(&[old_rows,chunk])?];let mut signed=vec![C::default();packed.len()];
        let mut e=vec![C::default();product(&[shell_rows,chunk])?];let mut f=e.clone();
        let mut eb=vec![0.;e.len()];let mut fb=eb.clone();
        let mut assembled=vec![C::default();product(&[new_rows,chunk])?];let mut native=assembled.clone();
        let mut envelope=vec![0.;assembled.len()];let mut bound=envelope.clone();
        let external_bytes=sum_size(&[product(&[packed.capacity(),8])?,product(&[signed.capacity(),16])?,
            product(&[e.capacity()+f.capacity(),16])?,product(&[eb.capacity()+fb.capacity(),8])?,
            product(&[assembled.capacity()+native.capacity(),16])?,product(&[envelope.capacity()+bound.capacity(),8])?])?;
        let mut maximum_ratio=0_f64;let mut maximum_arithmetic_ratio=0_f64;let defects=shell.real_preparation_defects();let mut action_ms=0.;let mut guard_ms=0.;let mut peak=external_bytes;let mut chunks=0;
        for start in (0..columns).step_by(chunk){let width=chunk.min(columns-start);let ni=old_rows*width;let ns=shell_rows*width;let no=new_rows*width;
            for row in 0..old_rows{for j in 0..width{packed[row*width+j]=old.q[(start+j)*old_rows+row];}}
            old_map.push_into(&packed[..ni],width,&mut signed[..ni])?;
            let input_hs:Vec<_>=(0..width).map(|j|(0..old_rows).fold(0_f64,|v,i|v.hypot(signed[i*width+j].re.hypot(signed[i*width+j].im)))).collect();
            let before=Instant::now();shell.apply_into(&signed[..ni],width,&mut e[..ns],&mut f[..ns])?;
            action_ms+=before.elapsed().as_secs_f64()*1000.;
            let before=Instant::now();shell.action_magnitudes_into(&signed[..ni],width,&mut eb[..ns],&mut fb[..ns])?;
            let terms=shell.diagnostics().roundoff_terms as f64;
            for(output,values,magnitudes,label)in[(&mut eq,&e,&eb,"E"),(&mut fq,&f,&fb,"F")]{
                assembled[..no].fill(C::default());envelope[..no].fill(0.);
                for(source,&target)in targets.iter().enumerate(){
                    assembled[target*d*width..(target+1)*d*width].copy_from_slice(&values[source*d*width..(source+1)*d*width]);
                    envelope[target*d*width..(target+1)*d*width].copy_from_slice(&magnitudes[source*d*width..(source+1)*d*width]);
                }
                new_map.pull_complex_into(&assembled[..no],width,&mut native[..no])?;
                new_map.pull_envelope_into(&envelope[..no],width,&mut bound[..no])?;
                let defect=if label=="E"{defects.0}else{defects.1};
                let guard=super::runtime_core::check_real_image(&native[..no],&bound[..no],width,&input_hs,terms,defect)?;
                maximum_ratio=maximum_ratio.max(guard.maximum_ratio);maximum_arithmetic_ratio=maximum_arithmetic_ratio.max(guard.maximum_arithmetic_ratio);
                if !guard.passed{return Err(format!("Transferred original {label} shell loses REAL invariance: excess_hs={:e},prepared_allowance_hs={:e},prepared_defect={defect:e}",guard.excess_hs,guard.preparation_allowance_hs));}
                // Write ONLY new REAL blocks. Old Q/EQ/FQ never go through a
                // push/pull round trip and remain bitwise original.
                for block in &shell_blocks{let count=block_width(block,d)?;
                    for row in block.offset..block.offset+count{for j in 0..width{output[(start+j)*new_rows+row]=native[row*width+j].re;}}
                }
            }
            guard_ms+=before.elapsed().as_secs_f64()*1000.;chunks+=1;
            peak=peak.max(sum_size(&[external_bytes,shell.workspace_bytes()])?);
            if peak>options.scratch_bytes{return Err("Measured transfer scratch exceeded its explicit byte budget".into());}
        }
        let returned_bytes=product(&[3,new_rows,columns,8])?;
        let diagnostics=TransferDiagnostics{old_modes:old.modes.len(),new_modes:new_modes.len(),shell_modes:shell_modes.len(),columns,
            old_image_rows_recomputed:false,zero_padded_input_products_computed:false,complete_intermediate_return_paths:true,
            old_real_blocks_bitwise_retained:true,original_observation_rows_shared:true,population_answers_reused:false,
            same_request_owner:true,same_pole:true,same_time_grid:true,zero_lift_is_new_convergence_evidence:false,
            rhs_chunk_size:chunk,chunks,scratch_budget_bytes:options.scratch_bytes,scratch_upper_bytes_per_rhs:per,
            peak_measured_scratch_bytes:peak,returned_image_bytes:returned_bytes,maximum_reality_allowance_ratio:maximum_ratio,
            maximum_multiplication_only_ratio:maximum_arithmetic_ratio,prepared_e_conjugacy_defect_bound:defects.0,prepared_f_conjugacy_defect_bound:defects.1,preparation_budget_is_population_error_bound:false,
            native_shell_action_ms:action_ms,envelope_and_pull_ms:guard_ms,elapsed_ms:began.elapsed().as_secs_f64()*1000.};
        Ok(TransferredImages{owner:Arc::clone(&self.primitives),q,eq,fq,rows:new_rows,columns,modes:new_modes.to_vec(),pole,
            projected_mass:old.projected_mass.clone(),projected_pencil:old.projected_pencil.clone(),
            observation_rows:old.observation_rows.clone(),diagnostics})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::runtime_core::{RuntimeCore,ColdOptions};
    use super::super::coordinates::InitialView;
    use super::super::b0_inverse::P0Policy;
    fn runtime()->RuntimeCore{
        RuntimeCore::from_owned(super::super::tests::fixture(),InitialView{index:0,real:None,imag:None},P0Policy::EigenPreferred).unwrap()
    }
    fn packed(column:&[f64],rows:usize,m:usize)->Vec<f64>{(0..rows*m).map(|i|column[(i%m)*rows+i/m]).collect()}
    #[test]
    fn preparation_domain_transfer_complete_images_permuted_real_blocks_and_shared_rows(){
        let mut runtime=runtime();let old_modes=[vec![2],vec![-1],vec![0],vec![-2],vec![1]];
        let new_modes=[vec![-3],vec![1],vec![-2],vec![3],vec![0],vec![2],vec![-1]];let times=[0.,0.03,0.1];
        let old=runtime.cold_candidate(&old_modes,&times,ColdOptions{pole:8.,maximum_basis:3,proposal_steps:1}).unwrap();
        let saved=(old.q.clone(),old.eq.clone(),old.fq.clone());let prepared=runtime.prepared();
        let lifted=prepared.transfer_candidate_images(&old,&new_modes,&times,8.,TransferOptions{maximum_rhs:2,scratch_bytes:64*1024*1024}).unwrap();
        assert!(lifted.belongs_to(prepared));assert_eq!(lifted.columns,old.dimension);assert_eq!(lifted.diagnostics.shell_modes,2);
        assert!(!lifted.diagnostics.old_image_rows_recomputed);assert!(!lifted.diagnostics.zero_lift_is_new_convergence_evidence);
        assert!(lifted.diagnostics.peak_measured_scratch_bytes<=lifted.diagnostics.scratch_budget_bytes);
        assert_eq!(old.observation_rows.coefficients().as_ptr(),lifted.observation_rows.coefficients().as_ptr());
        let old_map=prepared.real_fourier_map(&old_modes).unwrap();let new_map=prepared.real_fourier_map(&new_modes).unwrap();let d=16;
        for block in old_map.harmonics(){let target=new_map.harmonics().iter().find(|h|h.index==block.index).unwrap();let width=block_width(block,d).unwrap();
            for j in 0..old.dimension{for i in 0..width{for(a,b)in[(&old.q,&lifted.q),(&old.eq,&lifted.eq),(&old.fq,&lifted.fq)]{
                assert_eq!(a[j*old.rows+block.offset+i].to_bits(),b[j*lifted.rows+target.offset+i].to_bits());
            }}}
        }
        assert_eq!(old.q,saved.0);assert_eq!(old.eq,saved.1);assert_eq!(old.fq,saved.2);
        let m=lifted.columns;let mut full=prepared.pencil(&new_modes,&new_modes,m).unwrap();
        let mut signed=vec![C::default();lifted.rows*m];new_map.push_into(&packed(&lifted.q,lifted.rows,m),m,&mut signed).unwrap();
        let mut e=signed.clone();let mut f=signed.clone();let mut actual=signed.clone();full.apply_into(&signed,m,&mut e,&mut f).unwrap();
        for(values,wanted)in[(&e,&lifted.eq),(&f,&lifted.fq)]{new_map.pull_complex_into(values,m,&mut actual).unwrap();
            for i in 0..lifted.rows{for j in 0..m{let z=actual[i*m+j];let expected=wanted[j*lifted.rows+i];assert!((z.re-expected).hypot(z.im)<3e-11*(1.+expected.abs()));}}
        }
        // Source and observation owners remain valid after dropping the factory.
        drop(runtime);let small:Vec<_>=old.coefficients.iter().map(|&re|C{re,im:0.}).collect();let mut pop=vec![C::default();times.len()*4];
        lifted.observation_rows.evaluate_into(&small,&mut pop).unwrap();for(a,b)in pop.iter().zip(&old.populations){assert_eq!(a.re.to_bits(),b.re.to_bits());assert_eq!(a.im.to_bits(),b.im.to_bits());}
    }
    #[test]
    fn preparation_domain_transfer_rejects_equal_model_new_owner_shift_time_and_incomplete_shell(){
        let mut first=runtime();let old_modes=[vec![0],vec![1],vec![-1]];let times=[0.,0.1];
        let old=first.cold_candidate(&old_modes,&times,ColdOptions{pole:8.,maximum_basis:2,proposal_steps:1}).unwrap();
        let new_modes=[vec![2],vec![0],vec![-1],vec![-2],vec![1]];let second=runtime();let options=TransferOptions::default();
        assert!(!first.prepared().same_request(second.prepared()));
        assert!(second.prepared().transfer_candidate_images(&old,&new_modes,&times,8.,options).is_err());
        assert!(first.prepared().transfer_candidate_images(&old,&new_modes,&times,9.,options).is_err());
        assert!(first.prepared().transfer_candidate_images(&old,&new_modes,&[0.,0.2],8.,options).is_err());
        assert!(first.prepared().transfer_candidate_images(&old,&old_modes,&times,8.,options).is_err());
        assert!(first.prepared().transfer_candidate_images(&old,&[vec![0],vec![1],vec![-1],vec![2]],&times,8.,options).is_err());
        assert!(first.prepared().transfer_candidate_images(&old,&new_modes,&times,8.,TransferOptions{maximum_rhs:1,scratch_bytes:1}).is_err());
    }
}
