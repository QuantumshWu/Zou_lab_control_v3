//! Complete current residual -> P0 -> REAL tangent -> continuous-weighted range.
//!
//! Output is numerical trial directions ONLY. No linear solution, original
//! operator, population accuracy, discarded-tail bound or accept decision is
//! asserted. Every new actual normalized trial column still needs full E/F.
use super::*;
use super::b0_inverse::RequestP0;
use super::runtime_core::Candidate;
use super::continuous_source_gram::{ContinuousSourceGram,SourceGramDiagnostics};

pub struct UnrotatedSource{pub generator:Vec<f64>,pub initial:Vec<f64>}
pub struct ResidualInput<'a>{
    pub modes:&'a[Charge],pub columns:usize,
    /// Column-major complete REAL arrays [domain_dimension,columns].
    pub q:&'a[f64],pub eq:&'a[f64],pub fq:&'a[f64],
    /// Explicitly UNROTATED, in the supplied Q coordinates.
    pub source:&'a UnrotatedSource,
}
#[derive(Clone,Copy,Debug)]
pub struct ProposalOptions{
    pub pole:f64,pub duration:f64,pub maximum_rhs:usize,pub minimum_directions:usize,
    /// Selection target=min(relative*weighted-HS,absolute). These are source
    /// importance controls, never population tolerances or error bounds.
    pub relative_source_tail:f64,pub absolute_source_tail:f64,pub check_gram_scaling:bool,
}
#[derive(Clone,Debug)]
pub struct RangeDiagnostics{
    pub method:&'static str,pub input_columns:usize,pub qr_resolved_rank:usize,pub svd_resolved_rank:usize,
    pub selected_directions:usize,pub qr_roundoff_threshold:f64,pub singular_values:Vec<f64>,
    pub discarded_source_hs:f64,pub target_source_hs:f64,pub target_met_numerically:bool,
    pub selected_orthogonality_hs:f64,pub population_error_bound:bool,
}
#[derive(Clone,Debug)]
pub struct ProposalDiagnostics{
    pub original_residual_columns:usize,pub all_columns_preconditioned:usize,pub p0_chunks:usize,
    pub original_residual_norms:Vec<f64>,pub source_formation_floors:Vec<f64>,pub source_resolved:Vec<bool>,
    pub raw_preconditioned_norms:Vec<f64>,pub trace_tangent_change_hs:Vec<f64>,
    pub weighted_column_norms:Vec<f64>,pub pre_normalization_floors:Vec<f64>,pub weighted_resolved_columns:usize,
    pub maximum_proposal_imaginary_hs:f64,pub gram:SourceGramDiagnostics,
    pub gram_negative_roundoff_clipped_for_importance:f64,pub selection:RangeDiagnostics,
    pub final_old_new_overlap_hs:f64,pub final_noncentral_trace_max:f64,pub full_original_images_required:bool,
    pub physical_coefficients_deleted:bool,pub accepted:bool,pub proposal_only:bool,
    pub residual_ms:f64,pub precondition_ms:f64,pub projection_and_gram_ms:f64,pub range_ms:f64,pub elapsed_ms:f64,
}
pub struct ResidualProposal{pub directions:Vec<f64>,pub rows:usize,pub columns:usize,pub diagnostics:ProposalDiagnostics}

fn norm(a:&[f64])->f64{let scale=a.iter().map(|x|x.abs()).fold(0_f64,f64::max);
    if scale==0.{0.}else{a.iter().map(|x|{let y=x/scale;y*y}).sum::<f64>().sqrt()*scale}}
fn finite(a:&[f64],name:&str)->Result<(),String>{if a.iter().any(|x|!x.is_finite()){Err(format!("Nonfinite residual proposal {name}"))}else{Ok(())}}
fn dot(a:&[f64],b:&[f64])->f64{a.iter().zip(b).map(|(a,b)|a*b).sum()}
fn transpose(a:&[f64],n:usize)->Vec<f64>{(0..n*n).map(|k|a[(k%n)*n+k/n]).collect()}
fn empty_range(m:usize,target:f64)->RangeDiagnostics{RangeDiagnostics{method:"pivoted double-pass MGS-QR plus complete small Hermitian-dilation SVD",
    input_columns:m,qr_resolved_rank:0,svd_resolved_rank:0,selected_directions:0,qr_roundoff_threshold:0.,singular_values:vec![],
    discarded_source_hs:0.,target_source_hs:target,target_met_numerically:true,selected_orthogonality_hs:0.,population_error_bound:false}}

/// Reliable reference path: pivoted two-pass QR on actual columns, then the
/// SVD of its small R from eigenpairs of [0 R; R^T 0]. This uses existing
/// Hermitian LAPACK without squaring singular values in a long-matrix Gram.
fn select_range(lap:&Lapack,mut work:Vec<f64>,rows:usize,m:usize,target:f64,minimum:usize)->Result<(Vec<f64>,RangeDiagnostics),String>{
    let mut report=empty_range(m,target);if m==0{return Ok((vec![],report));}
    let mut largest=0_f64;for column in work.chunks_exact(rows){largest=largest.max(norm(column));}
    if largest==0.{return Ok((vec![],report));}
    let cutoff=64.*f64::EPSILON*rows.max(m)as f64*largest.max(f64::MIN_POSITIVE);
    report.qr_roundoff_threshold=cutoff;
    let mut q=Vec::<f64>::new();let mut r=vec![0.;product(&[m,m])?];let mut rank=0;
    for k in 0..m.min(rows){
        let(mut pivot,mut pivot_norm)=(k,0.);for j in k..m{let length=norm(&work[j*rows..(j+1)*rows]);if length>pivot_norm{pivot=j;pivot_norm=length;}}
        if pivot_norm<=cutoff{break;}
        if pivot!=k{for i in 0..rows{work.swap(k*rows+i,pivot*rows+i);}for i in 0..k{r.swap(i+k*m,i+pivot*m);}}
        let mut next=work[k*rows..(k+1)*rows].to_vec();
        for _ in 0..2{for i in 0..rank{let basis=&q[i*rows..(i+1)*rows];let c=dot(basis,&next);r[i+k*m]+=c;for j in 0..rows{next[j]-=c*basis[j];}}}
        let length=norm(&next);if !length.is_finite(){return Err("QR proposal norm is nonfinite".into());}
        if length<=cutoff{break;}
        for x in &mut next{*x/=length;}r[k+k*m]=length;
        for j in k+1..m{for _ in 0..2{let c=dot(&next,&work[j*rows..(j+1)*rows]);r[k+j*m]+=c;
            for i in 0..rows{work[j*rows+i]-=c*next[i];}}}
        work[k*rows..(k+1)*rows].fill(0.);q.extend(next);rank+=1;
    }
    report.qr_resolved_rank=rank;let unresolved_qr=norm(&work);
    if rank==0{report.discarded_source_hs=unresolved_qr;report.target_met_numerically=unresolved_qr<=target;return Ok((vec![],report));}
    let side=rank.checked_add(m).ok_or("Dilation dimension overflow")?;let mut dilation=vec![C::default();product(&[side,side])?];
    for j in 0..m{for i in 0..rank{let value=r[i+j*m];dilation[i+(rank+j)*side].re=value;dilation[rank+j+i*side].re=value;}}
    let(values,vectors)=lap.hermitian_eigen(&dilation,side)?;
    let positive:Vec<_>=(0..side).rev().filter(|&i|values[i]>0.).collect();
    let resolved=positive.iter().filter(|&&i|values[i]>cutoff).count().min(rank);
    report.svd_resolved_rank=resolved;report.singular_values=positive.iter().map(|&i|values[i]).collect();
    let mut tails=vec![unresolved_qr;positive.len()+1];
    for j in (0..positive.len()).rev(){tails[j]=tails[j+1].hypot(values[positive[j]]);}
    let needed=(1..=resolved).find(|&r|tails[r]<=target).unwrap_or(resolved);
    let chosen=resolved.min(needed.max(minimum.min(resolved)));report.selected_directions=chosen;
    report.discarded_source_hs=tails[chosen];report.target_met_numerically=report.discarded_source_hs<=target;
    let mut small=vec![0.;product(&[rank,chosen])?];
    for j in 0..chosen{let source=positive[j];let mut length=0_f64;let mut imaginary=0_f64;
        for i in 0..rank{let z=vectors[i+source*side];length=length.hypot(z.re);imaginary=imaginary.hypot(z.im);}
        if imaginary>256.*f64::EPSILON*side as f64||(length-std::f64::consts::FRAC_1_SQRT_2).abs()>1e-8{
            return Err("Small SVD dilation eigenvector is not a resolved REAL left direction".into());}
        for i in 0..rank{small[i+j*rank]=vectors[i+source*side].re/length;}
    }
    let selected=lap.multiply(&q,&small,rows,chosen,rank);finite(&selected,"selected SVD range")?;
    let mut gram=vec![0.;product(&[chosen,chosen])?];
    if chosen>0{lap.multiply_transpose_left_into(&selected,&selected,&mut gram,rows,chosen,chosen,rows,rows,chosen)?;
        for i in 0..chosen{gram[i+i*chosen]-=1.;}}
    report.selected_orthogonality_hs=norm(&gram);
    if report.selected_orthogonality_hs>1e-10{return Err("Selected trial SVD range lost numerical orthogonality".into());}
    Ok((selected,report))
}

impl OwnedPrepared{
    pub fn unrotated_candidate_source(&self,candidate:&Candidate)->Result<UnrotatedSource,String>{
        if !candidate.belongs_to(self){return Err("Small source belongs to another current request".into());}
        let m=candidate.dimension;let r=&candidate.rotation;let lap=&self.preparation_lapack;
        if r.len()!=product(&[m,m])?||candidate.generator.len()!=r.len()||candidate.c0.len()!=m{return Err("Incomplete rotated small source".into());}
        let generator=lap.multiply(&lap.multiply(r,&candidate.generator,m,m,m),&transpose(r,m),m,m,m);
        let initial=lap.multiply(r,&candidate.c0,m,1,m);finite(&generator,"unrotated generator")?;finite(&initial,"unrotated initial")?;
        Ok(UnrotatedSource{generator,initial})
    }

    pub fn residual_proposal(&self,p0:&mut RequestP0,input:ResidualInput<'_>,options:ProposalOptions)->Result<ResidualProposal,String>{
        let began=Instant::now();let lap=&self.preparation_lapack;
        if !p0.belongs_to(self){return Err("Residual proposal P0 belongs to another current request".into());}
        if input.columns==0||options.maximum_rhs==0||options.minimum_directions==0||!options.pole.is_finite()||options.pole<=0.
            ||!options.duration.is_finite()||options.duration<0.||!options.relative_source_tail.is_finite()||options.relative_source_tail<0.
            ||!options.absolute_source_tail.is_finite()||options.absolute_source_tail<0.{return Err("Invalid numerical proposal controls".into());}
        let map=self.real_fourier_map(input.modes)?;let rows=map.dimension();let m=input.columns;let size=product(&[rows,m])?;
        if input.q.len()!=size||input.eq.len()!=size||input.fq.len()!=size||input.source.generator.len()!=product(&[m,m])?||input.source.initial.len()!=m{
            return Err("Complete Q/EQ/FQ and matching UNROTATED G/c0 required".into());}
        for(a,name)in[(input.q,"Q"),(input.eq,"EQ"),(input.fq,"FQ"),(&input.source.generator,"G"),(&input.source.initial,"c0")]{finite(a,name)?;}
        let source_begin=Instant::now();let represented=lap.multiply(input.eq,&input.source.generator,rows,m,m);
        let residual:Vec<_>=input.fq.iter().zip(represented).map(|(a,b)|a-b).collect();finite(&residual,"full FQ-EQG")?;
        let e_norms:Vec<_>=input.eq.chunks_exact(rows).map(norm).collect();let f_norms:Vec<_>=input.fq.chunks_exact(rows).map(norm).collect();
        let residual_norms:Vec<_>=residual.chunks_exact(rows).map(norm).collect();
        let source_floors:Vec<_>=(0..m).map(|j|128.*f64::EPSILON*(m+1)as f64*(f_norms[j]+(0..m).map(|i|e_norms[i]*input.source.generator[i+j*m].abs()).sum::<f64>())).collect();
        let source_resolved:Vec<_>=residual_norms.iter().zip(&source_floors).map(|(a,b)|a>b).collect();
        let residual_ms=source_begin.elapsed().as_secs_f64()*1000.;
        let p_begin=Instant::now();let phases=&self.primitives.hilbert.source.phases;
        let shifts=input.modes.iter().map(|q|phases.frequency(q).map(|f|C{re:options.pole,im:f.hi+f.lo})).collect::<Result<Vec<_>,_>>()?;
        let chunk=options.maximum_rhs.min(m);let mut batch=p0.prepare_batch(&shifts,chunk)?;
        let mut proposal=vec![0.;size];let mut packed=vec![0.;product(&[rows,chunk])?];let mut signed=vec![C::default();packed.len()];let mut answer=signed.clone();let mut pulled=packed.clone();
        let mut chunks=0;let mut imaginary=0_f64;
        for start in (0..m).step_by(chunk){let width=chunk.min(m-start);let count=rows*width;
            for i in 0..rows{for j in 0..width{packed[i*width+j]=residual[(start+j)*rows+i];}}
            map.push_into(&packed[..count],width,&mut signed[..count])?;batch.apply_into(&signed[..count],width,&mut answer[..count])?;
            let detail=map.pull_real_proposal_into(&answer[..count],width,&mut pulled[..count])?;
            for v in detail.imaginary_hs_by_column{imaginary=imaginary.max(v);}
            for i in 0..rows{for j in 0..width{proposal[(start+j)*rows+i]=pulled[i*width+j];}}chunks+=1;
        }
        let precondition_ms=p_begin.elapsed().as_secs_f64()*1000.;drop(residual);drop(packed);drop(signed);drop(answer);drop(pulled);
        let projection_begin=Instant::now();let n=self.primitives.hilbert.source.n;let d=n*n;
        let raw_norms:Vec<_>=proposal.chunks_exact(rows).map(norm).collect();let mut work=raw_norms.clone();let mut tangent=vec![0_f64;m];
        for block in map.harmonics(){if block.has_imaginary_block{for offset in[block.offset,block.offset+d]{for j in 0..m{
            let trace: f64=proposal[j*rows+offset..j*rows+offset+n].iter().sum();tangent[j]=tangent[j].hypot(trace/(n as f64).sqrt());
            for i in 0..n{proposal[j*rows+offset+i]-=trace/n as f64;}
        }}}}
        for j in 0..m{work[j]+=tangent[j];}
        for _ in 0..2{let mut coefficients=vec![0.;m*m];lap.multiply_transpose_left_into(input.q,&proposal,&mut coefficients,rows,m,m,rows,rows,m)?;
            for j in 0..m{work[j]+=norm(&coefficients[j*m..(j+1)*m]);}
            let removed=lap.multiply(input.q,&coefficients,rows,m,m);for(a,b)in proposal.iter_mut().zip(removed){*a-=b;}
        }
        // Unresolved source subtraction cannot become a qualified direction
        // merely because P0 amplifies it. Original R itself was never changed.
        for j in 0..m{if !source_resolved[j]{proposal[j*rows..(j+1)*rows].fill(0.);}}
        let ContinuousSourceGram{matrix:gram,diagnostics:gram_diagnostics}=self.continuous_source_gram(&input.source.generator,&input.source.initial,options.duration,options.check_gram_scaling)?;
        let complex:Vec<_>=gram.iter().map(|&re|C{re,im:0.}).collect();let(values,vectors)=lap.hermitian_eigen(&complex,m)?;
        let scale=values.iter().map(|x|x.abs()).fold(0_f64,f64::max).max(f64::MIN_POSITIVE);
        if values[0]< -1e-10*scale{return Err("Continuous source Gram has unresolved numerical indefiniteness".into());}
        let mut root=vec![0.;m*m];for j in 0..m{for i in 0..m{let z=vectors[i+j*m];if z.im.abs()>256.*f64::EPSILON*m as f64{return Err("REAL Gram eigenvector is unresolved".into());}
            root[i+j*m]=z.re*values[j].max(0.).sqrt();}}
        let weighted=lap.multiply(&proposal,&root,rows,m,m);drop(proposal);finite(&weighted,"complete weighted source")?;
        let norms:Vec<_>=weighted.chunks_exact(rows).map(norm).collect();let floors:Vec<_>=(0..m).map(|j|128.*f64::EPSILON*(rows+m+32)as f64*
            (0..m).map(|i|work[i]*root[i+j*m].abs()).sum::<f64>()).collect();
        let resolved:Vec<_>=(0..m).filter(|&j|norms[j]>floors[j]).collect();let mut compact=Vec::with_capacity(product(&[rows,resolved.len()])?);
        for &j in &resolved{compact.extend_from_slice(&weighted[j*rows..(j+1)*rows]);}let weighted_norm=norm(&weighted);drop(weighted);
        let target=(options.relative_source_tail*weighted_norm).min(options.absolute_source_tail);
        if !weighted_norm.is_finite()||!target.is_finite(){return Err("Weighted source/importance target norm overflow".into());}
        let projection_and_gram_ms=projection_begin.elapsed().as_secs_f64()*1000.;let range_begin=Instant::now();
        let(mut directions,mut selection)=select_range(lap,compact,rows,resolved.len(),target,options.minimum_directions)?;
        let chosen=selection.selected_directions;
        // Resolve the final actual selected vectors before the caller asks
        // for original E/F. This is trial-coordinate work, not rho repair.
        for j in 0..chosen{let mut column=directions[j*rows..(j+1)*rows].to_vec();let original=norm(&column);
            for block in map.harmonics(){if block.has_imaginary_block{for offset in[block.offset,block.offset+d]{
                let trace: f64=column[offset..offset+n].iter().sum();for i in 0..n{column[offset+i]-=trace/n as f64;}
            }}}
            for _ in 0..2{for i in 0..m{let q=&input.q[i*rows..(i+1)*rows];let c=dot(q,&column);for k in 0..rows{column[k]-=c*q[k];}}
                for i in 0..j{let q=&directions[i*rows..(i+1)*rows];let c=dot(q,&column);for k in 0..rows{column[k]-=c*q[k];}}}
            let length=norm(&column);if length<=128.*f64::EPSILON*(rows+m+chosen)as f64*original{return Err("Selected weighted trial direction is unresolved before normalization".into());}
            for x in &mut column{*x/=length;}directions[j*rows..(j+1)*rows].copy_from_slice(&column);
        }
        let mut overlap=vec![0.;product(&[m,chosen])?];let mut new_gram=vec![0.;product(&[chosen,chosen])?];
        if chosen>0{lap.multiply_transpose_left_into(input.q,&directions,&mut overlap,rows,m,chosen,rows,rows,m)?;
            lap.multiply_transpose_left_into(&directions,&directions,&mut new_gram,rows,chosen,chosen,rows,rows,chosen)?;for i in 0..chosen{new_gram[i+i*chosen]-=1.;}}
        selection.selected_orthogonality_hs=norm(&new_gram);let old_new=norm(&overlap);
        if old_new>1e-10||selection.selected_orthogonality_hs>1e-10{return Err("Final weighted trial block lost REAL orthogonality".into());}
        let mut trace_max=0_f64;for block in map.harmonics(){if block.has_imaginary_block{for offset in[block.offset,block.offset+d]{for j in 0..chosen{
            trace_max=trace_max.max(directions[j*rows+offset..j*rows+offset+n].iter().sum::<f64>().abs());
        }}}}
        if trace_max>1e-10{return Err("Final weighted trial block lost the fixed noncentral trace constraint".into());}
        let range_ms=range_begin.elapsed().as_secs_f64()*1000.;
        Ok(ResidualProposal{directions,rows,columns:chosen,diagnostics:ProposalDiagnostics{
            original_residual_columns:m,all_columns_preconditioned:m,p0_chunks:chunks,original_residual_norms:residual_norms,
            source_formation_floors:source_floors,source_resolved,raw_preconditioned_norms:raw_norms,trace_tangent_change_hs:tangent,
            weighted_column_norms:norms,pre_normalization_floors:floors,weighted_resolved_columns:resolved.len(),maximum_proposal_imaginary_hs:imaginary,
            gram:gram_diagnostics,gram_negative_roundoff_clipped_for_importance:(-values[0]).max(0.),selection,
            final_old_new_overlap_hs:old_new,final_noncentral_trace_max:trace_max,full_original_images_required:true,physical_coefficients_deleted:false,accepted:false,proposal_only:true,
            residual_ms,precondition_ms,projection_and_gram_ms,range_ms,elapsed_ms:began.elapsed().as_secs_f64()*1000.}})
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    use super::super::runtime_core::{RuntimeCore,ColdOptions};
    use super::super::coordinates::InitialView;
    use super::super::b0_inverse::P0Policy;
    #[test]
    fn preparation_residual_range_small_dilation_keeps_dominant_singular_direction(){
        let owner=super::super::tests::fixture();let columns=vec![3.,0.,0.,0.,0.,1.,0.,0.,0.,0.,0.,0.];
        let(q,report)=select_range(&owner.preparation_lapack,columns,4,3,1.1,1).unwrap();
        assert_eq!(report.qr_resolved_rank,2);assert_eq!(report.selected_directions,1);
        assert!((q[0].abs()-1.).abs()<1e-13);assert!((report.discarded_source_hs-1.).abs()<1e-12);assert!(!report.population_error_bound);
    }
    #[test]
    fn preparation_residual_proposal_full_columns_current_owner_unrotated_source_and_original_images(){
        let mut runtime=RuntimeCore::from_owned(super::super::tests::fixture(),InitialView{index:0,real:None,imag:None},P0Policy::EigenPreferred).unwrap();
        let modes=[vec![1],vec![0],vec![-1]];let candidate=runtime.cold_candidate(&modes,&[0.,0.03,0.1],ColdOptions{pole:8.,maximum_basis:3,proposal_steps:1}).unwrap();
        let owner=runtime.prepared();let source=owner.unrotated_candidate_source(&candidate).unwrap();let saved=(candidate.q.clone(),candidate.eq.clone(),candidate.fq.clone());
        let input=||ResidualInput{modes:&modes,columns:candidate.dimension,q:&candidate.q,eq:&candidate.eq,fq:&candidate.fq,source:&source};
        let options=ProposalOptions{pole:8.,duration:0.1,maximum_rhs:2,minimum_directions:1,relative_source_tail:0.05,absolute_source_tail:1e-4,check_gram_scaling:true};
        let mut p0=owner.prepare_p0(P0Policy::EigenPreferred).unwrap();let result=owner.residual_proposal(&mut p0,input(),options).unwrap();
        assert_eq!(result.diagnostics.original_residual_columns,candidate.dimension);assert_eq!(result.diagnostics.all_columns_preconditioned,candidate.dimension);
        assert!(result.columns>0);assert!(result.diagnostics.final_old_new_overlap_hs<1e-10);assert!(!result.diagnostics.accepted);
        assert!(result.diagnostics.full_original_images_required);assert_eq!(candidate.q,saved.0);assert_eq!(candidate.eq,saved.1);assert_eq!(candidate.fq,saved.2);
        let other=super::super::tests::fixture();let mut wrong=other.prepare_p0(P0Policy::EigenPreferred).unwrap();assert!(owner.residual_proposal(&mut wrong,input(),options).is_err());
    }
}
