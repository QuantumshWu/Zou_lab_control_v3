//! Complete real-coordinate representation of rho -> U rho U^H.
//!
//! Input U and output K are column-major. K uses the ORIGINAL Hermitian
//! coordinate order from coordinates(n): n populations, then interleaved
//! Re/Im pairs (0,1),(0,2),... . No state, coherence or small entry is removed.
//! This is an isolated construction candidate; it performs no propagation and
//! does not replace the caller's validation of its prepared unitary matrix.
use crate::lapack::Complex64;

#[inline]
fn product_conjugate(a:Complex64,b:Complex64)->(f64,f64){
    (a.re*b.re+a.im*b.im,a.im*b.re-a.re*b.im)
}

/// K x is the Hermitian-coordinate vector of U rho(x) U^H. Its entries are
/// formed directly from U; no n²-dimensional eigensystem or basis sampling is
/// used. The same algebra works in the original or a unitarily changed basis.
pub fn matrix(unitary:&[Complex64],n:usize)->Result<Vec<f64>,String>{
    let d=n.checked_mul(n).ok_or("Unitary superoperator dimension overflow")?;
    let size=d.checked_mul(d).ok_or("Unitary superoperator allocation overflow")?;
    if n==0||unitary.len()!=d||unitary.iter().any(|v|!v.re.is_finite()||!v.im.is_finite()){
        return Err("Invalid unitary superoperator input".into());
    }
    let mut result=vec![0.;size];
    // A population basis matrix |a><a| produces one outer product.
    for a in 0..n{
        let column=&mut result[a*d..(a+1)*d];
        for k in 0..n{let u=unitary[k+a*n];column[k]=u.re*u.re+u.im*u.im;}
        let mut row=n;
        for k in 0..n{for l in k+1..n{
            let(re,im)=product_conjugate(unitary[k+a*n],unitary[l+a*n]);
            column[row]=re;column[row+1]=im;row+=2;
        }}
    }
    // The two off-diagonal basis matrices share the same products:
    // Re(a,b): A+B; Im(a,b): i(A-B), A=U_ka conj(U_lb).
    let mut input=n;
    for a in 0..n{for b in a+1..n{
        let (before,after)=result.split_at_mut((input+1)*d);
        let real=&mut before[input*d..(input+1)*d];let imag=&mut after[..d];
        for k in 0..n{
            let(re,im)=product_conjugate(unitary[k+a*n],unitary[k+b*n]);
            real[k]=2.*re;imag[k]=-2.*im;
        }
        let mut row=n;
        for k in 0..n{for l in k+1..n{
            let(ar,ai)=product_conjugate(unitary[k+a*n],unitary[l+b*n]);
            let(br,bi)=product_conjugate(unitary[k+b*n],unitary[l+a*n]);
            real[row]=ar+br;real[row+1]=ai+bi;
            imag[row]=bi-ai;imag[row+1]=ar-br;row+=2;
        }}
        input+=2;
    }}
    Ok(result)
}

#[cfg(test)]
mod tests{
    use super::*;
    use crate::{coordinates,Coordinate};
    fn multiply(a:&[Complex64],b:&[Complex64],n:usize)->Vec<Complex64>{
        let mut out=vec![Complex64::default();n*n];
        for j in 0..n{for i in 0..n{for k in 0..n{
            let x=a[i+k*n];let y=b[k+j*n];
            out[i+j*n].re+=x.re*y.re-x.im*y.im;
            out[i+j*n].im+=x.re*y.im+x.im*y.re;
        }}}
        out
    }
    fn nonreal_unitary(n:usize)->Vec<Complex64>{
        // DFT with distinct left/right phase factors: all rows and columns
        // mix complex amplitudes, exercising the signs of both Im coordinates.
        (0..n*n).map(|index|{
            let row=index%n;let column=index/n;
            let phase=std::f64::consts::TAU*(row*column)as f64/n as f64+0.31*row as f64-0.2*column as f64;
            Complex64{re:phase.cos()/(n as f64).sqrt(),im:phase.sin()/(n as f64).sqrt()}
        }).collect()
    }
    #[test]
    fn every_real_coordinate_column_matches_a_nonreal_complex_sandwich(){
        let n=3;let d=n*n;let u=nonreal_unitary(n);let k=matrix(&u,n).unwrap();let coords=coordinates(n);
        let adjoint:Vec<_>=(0..d).map(|i|{let z=u[i/n+(i%n)*n];Complex64{re:z.re,im:-z.im}}).collect();
        for(column,&coordinate)in coords.iter().enumerate(){
            let mut rho=vec![Complex64::default();d];
            match coordinate{
                Coordinate::Population(a)=>rho[a+a*n].re=1.,
                Coordinate::Real(a,b)=>{rho[a+b*n].re=1.;rho[b+a*n].re=1.;},
                Coordinate::Imag(a,b)=>{rho[a+b*n].im=1.;rho[b+a*n].im=-1.;},
            }
            let expected=multiply(&multiply(&u,&rho,n),&adjoint,n);
            for(row,&output)in coords.iter().enumerate(){
                let value=match output{Coordinate::Population(i)=>expected[i+i*n].re,
                    Coordinate::Real(i,j)=>expected[i+j*n].re,Coordinate::Imag(i,j)=>expected[i+j*n].im};
                assert!((k[row+column*d]-value).abs()<2e-14,"row={row}, column={column}");
            }
        }
    }
    #[test]
    fn full_superoperator_preserves_the_complete_hilbert_schmidt_metric_and_trace(){
        let n=3;let d=n*n;let k=matrix(&nonreal_unitary(n),n).unwrap();
        let weight=|i:usize|if i<n{1.}else{2.};
        for a in 0..d{for b in 0..d{
            let gram=(0..d).map(|i|weight(i)*k[i+a*d]*k[i+b*d]).sum::<f64>();
            assert!((gram-if a==b{weight(a)}else{0.}).abs()<3e-14);
        }}
        for j in 0..d{let trace=(0..n).map(|i|k[i+j*d]).sum::<f64>();assert!((trace-if j<n{1.}else{0.}).abs()<2e-14);}
    }
    #[test]
    fn identity_and_invalid_dimensions_are_explicit(){
        let n=3;let d=n*n;let mut identity=vec![Complex64::default();d];for i in 0..n{identity[i+i*n].re=1.;}
        let k=matrix(&identity,n).unwrap();
        for j in 0..d{for i in 0..d{assert_eq!(k[i+j*d],if i==j{1.}else{0.});}}
        assert!(matrix(&[],0).is_err());assert!(matrix(&identity[..8],3).is_err());
        identity[0].im=f64::NAN;assert!(matrix(&identity,3).is_err());
    }
}
