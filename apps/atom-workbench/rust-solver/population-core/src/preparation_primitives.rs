//! Complete Khat/S/J2 primitive preparation after the Model/gauge layer.
//! This defines numerical coordinates only. Unsafe S/K choices leave the
//! original H/C and full recycling in the source, never a rate equation.
use std::collections::BTreeSet;
use super::{Complex64 as C,FourierMatrices,GaugedSource,Sum,add,mul,conj,nonzero,opposite,keep_exact_nonzero,largest_coupling,STRUCTURAL_COUPLING_FLOOR};
use super::hilbert::{HilbertPrepared,transform_matrix};
use crate::lapack::Lapack;

pub struct OpticalCoordinateDiagnostics{pub selected_entries:usize,pub unsafe_optical_entries:usize,pub charges:usize}
pub struct ShearDiagnostics{
    pub enabled:bool,pub fallback_reason:Option<String>,pub sylvester_rows:usize,
    pub unprojected_homological_residual_hs:f64,pub trace_projection_change_hs:f64,
    pub final_trace_functional_hs:f64,pub uniform_frobenius_upper:f64,
}
pub struct PrimitivePreparation{
    pub hilbert:HilbertPrepared,
    pub s:FourierMatrices,
    /// Complete trace-preserving auxiliary Khat: all GG by all EE.
    pub k_hat:FourierMatrices,
    pub no_jump_left:FourierMatrices,pub no_jump_right:FourierMatrices,
    pub gg_rows:Vec<usize>,pub optical_rows:Vec<usize>,
    pub selected_k:FourierMatrices,pub selected_recycling:FourierMatrices,
    pub gg_hamiltonian:FourierMatrices,pub optical_hamiltonian:FourierMatrices,
    pub optical_diagnostics:OpticalCoordinateDiagnostics,pub shear_diagnostics:ShearDiagnostics,
}
fn norm(a:&[C])->f64{a.iter().fold(0_f64,|sum,z|sum.hypot(z.re.hypot(z.im)))}
fn row_to_column(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|k|a[(k%n)*n+k/n]).collect()}
fn column_to_row(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|k|a[k%n*n+k/n]).collect()}
fn minus_i(a:C)->C{C{re:a.im,im:-a.re}}
fn plus_i(a:C)->C{C{re:-a.im,im:a.re}}
fn scaled(a:C,s:f64)->C{C{re:a.re*s,im:a.im*s}}

pub(super) fn spectral_norm(lap:&Lapack,a:&[C],n:usize)->Result<f64,String>{
    let column=row_to_column(a,n);let adjoint:Vec<_>=(0..n*n).map(|k|conj(column[(k%n)*n+k/n])).collect();
    let gram=lap.multiply_complex(&adjoint,&column,n,n,n)?;let(values,_)=lap.hermitian_eigen(&gram,n)?;
    let largest=values.iter().copied().fold(0_f64,f64::max);
    if !largest.is_finite(){return Err("Coordinate source norm overflow".into());}Ok(largest.sqrt())
}
/// Selection is made in the ORIGINAL gauged basis before Ug/UE, matching
/// the current diagonal-optical-coordinate algorithm. No physical element
/// is removed when its homological denominator is unsafe.
pub fn select_optical_s(lap:&Lapack,source:&GaugedSource)->Result<(FourierMatrices,OpticalCoordinateDiagnostics),String>{
    let n=source.n;let zero=vec![0;source.phases.axes.len()];let h0=&source.hamiltonian[&zero];
    let ground:BTreeSet<_>=source.ground.iter().copied().collect();let max_energy=(0..n).map(|i|h0[i*n+i].re.abs()).fold(0_f64,f64::max);
    let mut output=FourierMatrices::new();let(mut selected,mut unsafe_entries)=(0,0);
    // A charge enters S with ALL its optical entries when its largest one is
    // significant; otherwise its couplings stay (exactly) in H only.
    let significant=STRUCTURAL_COUPLING_FLOOR*largest_coupling(&source.hamiltonian,n);
    for(q,h)in &source.hamiltonian{
        if q.iter().copied().find(|&v|v!=0).is_none_or(|v|v<0){continue;}
        let w=source.phases.frequency(q)?;let floor=(4.*spectral_norm(lap,h,n)?).max(256.*f64::EPSILON*(2.*max_energy+(w.hi+w.lo).abs()));
        let mut s=vec![C::default();n*n];
        for i in 0..n{for j in 0..n{if ground.contains(&i)==ground.contains(&j)||!nonzero(&h[i*n+j]){continue;}
            let mut denominator=Sum::default();for value in[h0[i*n+i].re,-h0[j*n+j].re,w.hi,w.lo]{denominator.add(value);}let d=denominator.value();
            if d.abs()>floor{let value=scaled(h[i*n+j],-1./d);if !value.re.is_finite()||!value.im.is_finite(){return Err("Nonfinite optical auxiliary coordinate".into());}s[i*n+j]=value;selected+=1;}else{unsafe_entries+=1;}
        }}
        let strongest=(0..n*n).filter(|&k|nonzero(&s[k])).map(|k|h[k].re.hypot(h[k].im)).fold(0_f64,f64::max);
        if strongest>=significant{let partner=(0..n*n).map(|k|scaled(conj(s[(k%n)*n+k/n]),-1.)).collect();output.insert(q.clone(),s);output.insert(opposite(q)?,partner);}
    }
    let diagnostics=OpticalCoordinateDiagnostics{selected_entries:selected,unsafe_optical_entries:unsafe_entries,charges:output.len()};Ok((output,diagnostics))
}

fn prepare_k(lap:&Lapack,h:&HilbertPrepared)->Result<(FourierMatrices,ShearDiagnostics),String>{
    let(g,e)=(h.source.ground.len(),h.source.excited.len());let e2=e*e;
    let mut diagnostic=ShearDiagnostics{enabled:false,fallback_reason:None,sylvester_rows:0,
        unprojected_homological_residual_hs:0.,trace_projection_change_hs:0.,final_trace_functional_hs:0.,uniform_frobenius_upper:0.};
    if h.diagnostics.minimum_mean_radiative_eigenvalue<=0.{diagnostic.fallback_reason=Some("No resolved positive EE mean damping; K coordinate disabled, complete R remains".into());return Ok((FourierMatrices::new(),diagnostic));}
    let execution=crate::parallel::Execution::new()?;
    let t=&h.excited_nojump_schur;let tr=column_to_row(t,e);let mut output=FourierMatrices::new();let mut residual_squared=0.;let mut trace_change_squared=0.;let mut trace_squared=0.;
    // Khat is formed only for recycling charges that are significant relative
    // to the largest one; smaller recycling stays (exactly) in R only.
    let significant=STRUCTURAL_COUPLING_FLOOR*h.radiation.recycling.values().map(|r|norm(r)).fold(0_f64,f64::max);
    for(q,r)in &h.radiation.recycling{if q.iter().all(|&v|v==0)||norm(r)<significant{continue;}
        let w=h.source.phases.frequency(q)?;let mut k=vec![C::default();g*g*e2];
        // The g*g rows (a,b) are independent Sylvester solves, run on the
        // worker pool; each returns its K row and residual sum of squares.
        let kernel=lap.sylvester_kernel();let mut rows=vec![(Vec::new(),0_f64);g*g];
        let solved=execution.for_mode_chunks(&mut rows,1,g*g*4*e*e2,crate::parallel::MINIMUM_WORK_PER_TASK,&mut Vec::<()>::new(),|first,chunk,_|{
            for(offset_row,(row_values,residual))in chunk.iter_mut().enumerate(){let row=first+offset_row;let(a,b)=(row/g,row%g);
                let mut frequency=Sum::default();for v in[h.ground_reference_energies[a],-h.ground_reference_energies[b],w.hi,w.lo]{frequency.add(v);}
                let mu=C{re:0.,im:-frequency.value()};
                // A^H Y-Y T=-Rrow^T with A=conj(mu)I-T. A/T remain
                // triangular; existing ztrsyl C/N scaling handling is reused.
                let mut aa:Vec<_>=t.iter().map(|&z|scaled(z,-1.)).collect();for i in 0..e{add(&mut aa[i+i*e],conj(mu));}
                let offset=row*e2;
                // Row-major vec(Rrow), interpreted column-major, is Rrow^T.
                // Likewise vec_F(Y) equals vec_row(Y^T): no extra transpose.
                let rhs:Vec<_>=r[offset..offset+e2].iter().map(|&z|scaled(z,-1.)).collect();
                let solved=kernel.complex_sylvester_rect_adjoint(&aa,t,&rhs,e,e)?;
                // Independent row-functional residual mu*K - T^T*K - K*conj(T)+R.
                let mut squared=0.;
                for i in 0..e{for j in 0..e{let mut value=mul(mu,solved[i*e+j]);add(&mut value,r[offset+i*e+j]);
                    for l in 0..e{add(&mut value,scaled(mul(tr[l*e+i],solved[l*e+j]),-1.));add(&mut value,scaled(mul(solved[i*e+l],conj(tr[l*e+j])),-1.));}
                    squared+=value.re*value.re+value.im*value.im;
                }}
                *row_values=solved;*residual=squared;
            }Ok(())
        });
        if let Err(reason)=solved{diagnostic.fallback_reason=Some(format!("K coordinate disabled after Sylvester failure; original R retained: {reason}"));return Ok((FourierMatrices::new(),diagnostic));}
        for(row,(values,residual))in rows.into_iter().enumerate(){k[row*e2..(row+1)*e2].copy_from_slice(&values);residual_squared+=residual;diagnostic.sylvester_rows+=1;}
        // Khat is a NEW numerical coordinate. Its trace projection is not
        // applied to the original R/H/C and is never called an exact solve
        // of the unprojected homological equation.
        for col in 0..e2{let mut trace=C::default();for a in 0..g{add(&mut trace,k[(a*g+a)*e2+col]);}
            let correction=scaled(trace,1./g as f64);trace_change_squared+=(trace.re*trace.re+trace.im*trace.im)/g as f64;
            for a in 0..g{add(&mut k[(a*g+a)*e2+col],scaled(correction,-1.));}
            let mut after=C::default();for a in 0..g{add(&mut after,k[(a*g+a)*e2+col]);}trace_squared+=after.re*after.re+after.im*after.im;
        }
        diagnostic.uniform_frobenius_upper+=norm(&k);output.insert(q.clone(),k);
    }
    diagnostic.enabled=true;diagnostic.unprojected_homological_residual_hs=residual_squared.sqrt();
    diagnostic.trace_projection_change_hs=trace_change_squared.sqrt();diagnostic.final_trace_functional_hs=trace_squared.sqrt();
    Ok((output,diagnostic))
}

pub fn prepare(lap:&Lapack,source:GaugedSource)->Result<PrimitivePreparation,String>{
    let(original_s,optical_diagnostics)=select_optical_s(lap,&source)?;
    let hilbert=super::hilbert::prepare(lap,source)?;let n=hilbert.source.n;let g=hilbert.source.ground.len();let e=n-g;
    let mut s=FourierMatrices::new();for(q,value)in original_s{s.insert(q,transform_matrix(lap,&hilbert.basis_column_major,&value,n)?);}
    let(k_hat,shear_diagnostics)=prepare_k(lap,&hilbert)?;
    let mut left=FourierMatrices::new();let mut right=FourierMatrices::new();
    let charges:BTreeSet<_>=hilbert.source.hamiltonian.keys().chain(hilbert.radiation.gamma.keys()).cloned().collect();
    for q in charges{let mut f=vec![C::default();n*n];let mut r=f.clone();
        if let Some(h)=hilbert.source.hamiltonian.get(&q){for i in 0..n*n{f[i]=minus_i(h[i]);r[i]=plus_i(h[i]);}}
        if let Some(gamma)=hilbert.radiation.gamma.get(&q){for i in 0..e{for j in 0..e{let value=scaled(gamma[i*e+j],-0.5);add(&mut f[(g+i)*n+g+j],value);add(&mut r[(g+i)*n+g+j],value);}}}
        left.insert(q.clone(),f);right.insert(q,r);
    }keep_exact_nonzero(&mut left);keep_exact_nonzero(&mut right);
    let mut support=BTreeSet::new();for value in hilbert.radiation.recycling.values().chain(k_hat.values()){
        for row in 0..g*g{if value[row*e*e..(row+1)*e*e].iter().any(nonzero){support.insert(row);}}
    }
    loop{let before=support.len();let old:Vec<_>=support.iter().copied().collect();
        for h in hilbert.source.hamiltonian.values(){for &row in &old{let(a,b)=(row/g,row%g);
            for i in 0..g{if nonzero(&h[i*n+a]){support.insert(i*g+b);}}
            for j in 0..g{if nonzero(&h[b*n+j]){support.insert(a*g+j);}}
        }}if support.len()==before{break;}}
    let gg_rows:Vec<_>=support.into_iter().collect();let rank=gg_rows.len();
    let optical_rows:Vec<_>=(0..n*n).filter(|&i|(i/n<g)!=(i%n<g)).collect();
    let select=|family:&FourierMatrices|->FourierMatrices{family.iter().map(|(q,a)|{
        (q.clone(),gg_rows.iter().flat_map(|&row|a[row*e*e..(row+1)*e*e].iter().copied()).collect())}).collect()};
    let selected_k=select(&k_hat);let selected_recycling=select(&hilbert.radiation.recycling);
    let mut gg=FourierMatrices::new();let mut optical=FourierMatrices::new();
    for(q,h)in &hilbert.source.hamiltonian{
        let mut ground=vec![C::default();rank*rank];let mut other=vec![C::default();optical_rows.len()*rank];
        for(column,&ab)in gg_rows.iter().enumerate(){let(a,b)=(ab/g,ab%g);
            for(row,&ij)in gg_rows.iter().enumerate(){let(i,j)=(ij/g,ij%g);let mut value=C::default();
                if b==j{add(&mut value,minus_i(h[i*n+a]));}if i==a{add(&mut value,plus_i(h[b*n+j]));}ground[row*rank+column]=value;}
            for(row,&ij)in optical_rows.iter().enumerate(){let(i,j)=(ij/n,ij%n);let mut value=C::default();
                if b==j{add(&mut value,minus_i(h[i*n+a]));}if i==a{add(&mut value,plus_i(h[b*n+j]));}other[row*rank+column]=value;}
        }
        if ground.iter().any(nonzero){gg.insert(q.clone(),ground);}if other.iter().any(nonzero){optical.insert(q.clone(),other);}
    }
    Ok(PrimitivePreparation{hilbert,s,k_hat,no_jump_left:left,no_jump_right:right,gg_rows,optical_rows,
        selected_k,selected_recycling,gg_hamiltonian:gg,optical_hamiltonian:optical,optical_diagnostics,shear_diagnostics})
}

#[cfg(test)]
mod tests{
    use super::*;use crate::preparation::{MatrixView,ModelView,PhaseBasis,GaugePolicy};
    #[test]
    fn preparation_primitives_keep_recycling_and_trace_preserving_auxiliary(){
        let n=4;let mut h=vec![0.;16];let zero=vec![0.;16];for(i,v)in[0.,0.6,3.,4.].iter().enumerate(){h[i*n+i]=*v;}
        for(i,j,v)in[(0,2,0.1),(1,3,0.2)]{h[i*n+j]=v;h[j*n+i]=v;}
        let mut c=vec![0.;16];c[2]=0.7;c[3]=0.1;c[6]=0.2;c[7]=0.8;
        let channels=[MatrixView{real:&c,imag:&zero}];let model=ModelView{dimension:n,hamiltonian:MatrixView{real:&h,imag:&zero},
            collapse:&channels,drives:&[],ground:&[true,true,false,false],initial_index:0};
        let phases=PhaseBasis::from_declared(&[7.],None,&[]).unwrap();let gauge=[vec![0],vec![1],vec![0],vec![0]];
        let source=GaugedSource::prepare(&model,phases,GaugePolicy::Explicit(&gauge)).unwrap();
        let lap=Lapack::load().unwrap();let prepared=prepare(&lap,source).unwrap();
        assert!(prepared.shear_diagnostics.enabled);assert!(prepared.shear_diagnostics.sylvester_rows>0);
        assert!(prepared.shear_diagnostics.unprojected_homological_residual_hs<1e-11);
        assert!(prepared.shear_diagnostics.final_trace_functional_hs<1e-13);
        assert_eq!(prepared.hilbert.source.collapse.len(),1);assert_eq!(prepared.hilbert.radiation.coefficient_pairs,4);
        assert!(prepared.selected_recycling.contains_key(&vec![0]));assert!(prepared.selected_recycling.contains_key(&vec![1]));
        for(q,s)in &prepared.s{let other=&prepared.s[&opposite(q).unwrap()];for i in 0..n{for j in 0..n{
            assert!((s[i*n+j].re+other[j*n+i].re).abs()<1e-13);assert!((s[i*n+j].im-other[j*n+i].im).abs()<1e-13);
        }}}
        assert!(prepared.gg_rows.len()<=4);assert_eq!(prepared.optical_rows.len(),8);
    }
}
