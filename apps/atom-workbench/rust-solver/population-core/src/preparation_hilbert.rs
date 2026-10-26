//! Next dependency block: complete block-diagonal Hilbert coordinate change.
//! Only G/E membership is used. Every H_q and every original C_q is rotated;
//! Gamma/R are reconstructed from all charge pairs afterwards. Existing
//! LAPACK zheev/zgees are reused; no1024 eigensolve or alternate physics model.
use super::{Complex64 as C,FourierMatrices,GaugedSource,RadiationCoefficients,nonzero,conj};
use crate::lapack::Lapack;

pub struct HilbertDiagnostics{
    pub ground_dimension:usize,pub excited_dimension:usize,
    pub minimum_mean_radiative_eigenvalue:f64,
    pub basis_unitarity_defect_hs:f64,
    pub ground_selector_hermitian_difference_hs:f64,
    pub transformed_hamiltonian_charges:usize,
    pub original_channels:usize,pub transformed_channel_charge_blocks:usize,
    pub coordinate_only:bool,
}
pub struct HilbertPrepared{
    pub source:GaugedSource,
    /// Column-major B: original gauged-Hilbert matrix = B X B^dagger.
    pub basis_column_major:Vec<C>,
    pub original_ground_indices:Vec<usize>,pub original_excited_indices:Vec<usize>,
    pub ground_reference_energies:Vec<f64>,
    /// Complete EE no-jump Schur T, column-major, reused by K Sylvester rows.
    pub excited_nojump_schur:Vec<C>,
    pub radiation:RadiationCoefficients,
    pub diagnostics:HilbertDiagnostics,
}
fn norm(values:&[C])->f64{values.iter().fold(0_f64,|sum,z|sum.hypot(z.re.hypot(z.im)))}
fn column_matrix(row:&[C],n:usize)->Vec<C>{(0..n*n).map(|k|row[(k%n)*n+k/n]).collect()}
fn row_matrix(column:&[C],n:usize)->Vec<C>{(0..n*n).map(|k|column[k%n*n+k/n]).collect()}
fn adjoint_column(a:&[C],n:usize)->Vec<C>{(0..n*n).map(|k|conj(a[(k%n)*n+k/n])).collect()}
fn block(matrix:&[C],n:usize,indices:&[usize])->Vec<C>{
    indices.iter().flat_map(|&i|indices.iter().map(move|&j|matrix[i*n+j])).collect()
}
/// Select the same lower-triangle Hermitian reference used by the Python
/// eigh call. This selects a numerical B only; original H remains untouched.
fn lower_hermitian(matrix:&[C],n:usize)->(Vec<C>,f64){
    let mut result=matrix.to_vec();let mut differences=Vec::with_capacity(n*n);
    for i in 0..n{for j in 0..n{
        result[i*n+j]=if i==j{C{re:matrix[i*n+i].re,im:0.}}else if i>j{matrix[i*n+j]}else{conj(matrix[j*n+i])};
        differences.push(C{re:result[i*n+j].re-matrix[i*n+j].re,im:result[i*n+j].im-matrix[i*n+j].im});
    }}(result,norm(&differences))
}

pub fn transform_matrix(lap:&Lapack,basis_column_major:&[C],matrix_row_major:&[C],n:usize)->Result<Vec<C>,String>{
    if basis_column_major.len()!=n*n||matrix_row_major.len()!=n*n{return Err("Complete Hilbert transform dimensions differ".into());}
    let adjoint=adjoint_column(basis_column_major,n);let column=column_matrix(matrix_row_major,n);
    let temp=lap.multiply_complex(&adjoint,&column,n,n,n)?;
    let transformed=lap.multiply_complex(&temp,basis_column_major,n,n,n)?;
    Ok(row_matrix(&transformed,n))
}
fn transform_family(lap:&Lapack,basis:&[C],family:&FourierMatrices,n:usize)->Result<FourierMatrices,String>{
    let mut result=FourierMatrices::new();
    for(q,a)in family{let transformed=transform_matrix(lap,basis,a,n)?;
        // Exact storage-zero cleanup only. The zero H block is retained below.
        if transformed.iter().any(nonzero){result.insert(q.clone(),transformed);}}
    Ok(result)
}

pub fn prepare(lap:&Lapack,mut source:GaugedSource)->Result<HilbertPrepared,String>{
    let n=source.n;let g=source.ground.len();let e=source.excited.len();let zero=vec![0;source.phases.axes.len()];
    let h0=source.hamiltonian.get(&zero).ok_or("Complete source lacks zero Hamiltonian")?;
    let original_radiation=source.radiation()?;
    let mean_gamma=original_radiation.gamma.get(&zero).cloned().unwrap_or_else(||vec![C::default();e*e]);
    let(hg,hg_defect)=lower_hermitian(&block(h0,n,&source.ground),g);
    let he=block(h0,n,&source.excited);
    let gamma_reference=lower_hermitian(&mean_gamma,e).0;
    let(rates,_)=lap.hermitian_eigen(&column_matrix(&gamma_reference,e),e)?;
    let(ground_energies,ug)=lap.hermitian_eigen(&column_matrix(&hg,g),g)?;
    let nojump:Vec<_>=he.iter().zip(&mean_gamma).map(|(h,gamma)|C{re:h.im-0.5*gamma.re,im:-h.re-0.5*gamma.im}).collect();
    let(t,ue)=lap.complex_schur(&column_matrix(&nojump,e),e)?;
    let mut basis=vec![C::default();n*n];
    for a in 0..g{for b in 0..g{basis[source.ground[a]+b*n]=ug[a+b*g];}}
    for a in 0..e{for b in 0..e{basis[source.excited[a]+(g+b)*n]=ue[a+b*e];}}
    let mut gram=lap.multiply_complex(&adjoint_column(&basis,n),&basis,n,n,n)?;
    for i in 0..n{gram[i+i*n].re-=1.;}
    let unitary=norm(&gram);
    if !unitary.is_finite()||unitary>1024.*f64::EPSILON*n as f64{return Err("Block Hilbert basis is not resolved as unitary".into());}
    let old_ground=source.ground.clone();let old_excited=source.excited.clone();
    source.hamiltonian=transform_family(lap,&basis,&source.hamiltonian,n)?;
    source.hamiltonian.entry(zero).or_insert_with(||vec![C::default();n*n]);
    source.collapse=source.collapse.iter().map(|channel|transform_family(lap,&basis,channel,n)).collect::<Result<_,_>>()?;
    source.ground=(0..g).collect();source.excited=(g..n).collect();
    // Original state gauges/frequency parts still label the OUTER original
    // state basis: restoration is D_g B X B^dagger D_g^dagger. Do not rotate
    // integer gauges or pretend they label the columns of B.
    let radiation=source.radiation()?;
    let diagnostics=HilbertDiagnostics{ground_dimension:g,excited_dimension:e,
        minimum_mean_radiative_eigenvalue:rates.iter().copied().fold(f64::INFINITY,f64::min),
        basis_unitarity_defect_hs:unitary,ground_selector_hermitian_difference_hs:hg_defect,
        transformed_hamiltonian_charges:source.hamiltonian.len(),original_channels:source.collapse.len(),
        transformed_channel_charge_blocks:source.collapse.iter().map(|c|c.len()).sum(),coordinate_only:true};
    Ok(HilbertPrepared{source,basis_column_major:basis,original_ground_indices:old_ground,original_excited_indices:old_excited,
        ground_reference_energies:ground_energies,excited_nojump_schur:t,radiation,diagnostics})
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::preparation::{MatrixView,ModelView,PhaseBasis,GaugePolicy};
    #[test]
    fn preparation_hilbert_keeps_complete_noncontiguous_sectors_and_channels(){
        let n=4;let mut hr=vec![0.;16];let hi=vec![0.;16];
        for(i,v)in[3.,0.,4.,1.].iter().enumerate(){hr[i*n+i]=*v;}
        for(i,j,v)in[(0,2,0.3),(1,3,0.2),(0,1,0.4),(2,3,0.5)]{hr[i*n+j]=v;hr[j*n+i]=v;}
        let mut cr=vec![0.;16];let mut ci=vec![0.;16];cr[4]=0.7;cr[6]=0.1;ci[12]=0.2;cr[14]=0.6;
        let channels=[MatrixView{real:&cr,imag:&ci}];let ground=[false,true,false,true];
        let model=ModelView{dimension:n,hamiltonian:MatrixView{real:&hr,imag:&hi},collapse:&channels,drives:&[],ground:&ground,initial_index:1};
        let source=GaugedSource::prepare(&model,PhaseBasis::from_declared(&[],None,&[]).unwrap(),GaugePolicy::Identity).unwrap();
        let h_old=source.hamiltonian[&vec![]].clone();let c_old=source.collapse[0][&vec![]].clone();
        let lap=Lapack::load().unwrap();let prepared=prepare(&lap,source).unwrap();
        assert_eq!(prepared.original_ground_indices,vec![1,3]);assert_eq!(prepared.original_excited_indices,vec![0,2]);
        assert_eq!(prepared.source.ground,vec![0,1]);assert_eq!(prepared.source.excited,vec![2,3]);
        assert!(prepared.diagnostics.basis_unitarity_defect_hs<1e-12);
        assert_eq!(prepared.source.collapse.len(),1);assert_eq!(prepared.radiation.coefficient_pairs,1);
        let b=&prepared.basis_column_major;
        for(actual,original)in[(&prepared.source.hamiltonian[&vec![]],h_old),(&prepared.source.collapse[0][&vec![]],c_old)]{
            let column=column_matrix(actual,n);let temp=lap.multiply_complex(b,&column,n,n,n).unwrap();
            let back=lap.multiply_complex(&temp,&adjoint_column(b,n),n,n,n).unwrap();let back=row_matrix(&back,n);
            let error:Vec<_>=back.iter().zip(original).map(|(x,y)|C{re:x.re-y.re,im:x.im-y.im}).collect();assert!(norm(&error)<1e-12);
        }
        assert!(prepared.diagnostics.minimum_mean_radiative_eigenvalue>0.);
    }
}
