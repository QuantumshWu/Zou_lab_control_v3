//! Diagnostics of the pipeline's sensitivity, behind environment switches
//! (all off by default; nothing here changes a result unless set):
//! - ATOM_DIAG=1: per checkpoint the projected operator's numerical abscissa
//!   in the metric (`abscissa`), the Gram's condition and the Ritz values
//!   nearest the axis; per ranking the leading poles and their margin
//!   (stderr, lines starting with DBG).
//! - ATOM_DIAG_<PART>=eps: a deterministic relative perturbation of size eps
//!   of one part alone (`perturbation`, `noise`): EIGEN (the exact
//!   eigenvectors, their inverse recomputed), ROTATE (eigenvectors rotated
//!   within eigenspaces whose values agree to eps), INVERSE (the inverse
//!   alone), VALUES (the eigenvalues), KERNELS (the SW kernels'
//!   entries), METRIC (the metric's columns M w), MODEL (the reduced G),
//!   TRANSFORM (the ranking's error transform). ATOM_DIAG_ACCEPT_AFTER=k
//!   holds acceptance back until k solves (the run continues past it).
use super::*;

/// The relative size of the perturbation ATOM_DIAG_<part> (None unset).
pub(crate) fn perturbation(part:&str)->Option<f64>{std::env::var(format!("ATOM_DIAG_{part}")).ok().and_then(|v|v.parse::<f64>().ok())}
/// Whether the reports are on (ATOM_DIAG).
pub(crate) fn reporting()->bool{std::env::var("ATOM_DIAG").is_ok()}
/// A deterministic value in [-1, 1) for stream s and index i (splitmix64).
pub(crate) fn noise(s:u64,i:u64)->f64{
    let mut x=i.wrapping_mul(0x9E37_79B9_7F4A_7C15)^s.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    x^=x>>31;x=x.wrapping_mul(0xBF58_476D_1CE4_E5B9);x^=x>>27;x=x.wrapping_mul(0x94D0_49BB_1331_11EB);x^=x>>31;
    (x>>11)as f64/(1_u64<<52)as f64-1.
}
/// The numerical abscissa of the operator whose K-weighted form is s (real,
/// row-major k x k, s = K G): the largest eigenvalue of K^-1/2 sym(s) K^-1/2,
/// the largest growth rate any vector of the space sees in the K-norm.
pub(crate) fn abscissa(lapack:&Lapack,kmat:&[f64],s:&[f64],k:usize)->Result<f64,String>{
    let(values,vectors)=lapack.hermitian_eigen(&kmat.iter().map(|&v|C{re:v,im:0.}).collect::<Vec<_>>(),k)?;
    // K^-1/2 = V diag(1/sqrt) V^T (column-major vectors).
    let mut root=vec![0.;k*k];
    for j in 0..k{let w=1./values[j].max(f64::MIN_POSITIVE).sqrt();for a in 0..k{for b in 0..k{root[a*k+b]+=vectors[a+k*j].re*vectors[b+k*j].re*w;}}}
    let sym:Vec<f64>=(0..k*k).map(|e|0.5*(s[e]+s[(e%k)*k+e/k])).collect();
    let product=|a:&[f64],b:&[f64]|->Vec<f64>{let mut c=vec![0.;k*k];for i in 0..k{for l in 0..k{let v=a[i*k+l];if v==0.{continue;}for j in 0..k{c[i*k+j]+=v*b[l*k+j];}}}c};
    let c=product(&product(&root,&sym),&root);
    let(values,_)=lapack.hermitian_eigen(&c.iter().map(|&v|C{re:v,im:0.}).collect::<Vec<_>>(),k)?;
    Ok(values.iter().fold(f64::NEG_INFINITY,|a,&v|a.max(v)))
}
