//! Test-only binary reader. Production takes borrowed ModelView, not this file.
use super::*;
use crate::preparation::{DriveView,FrequencyParts,MatrixView};
use crate::preparation::hilbert::transform_matrix;
use std::path::PathBuf;

struct Reader{bytes:Vec<u8>,at:usize}
impl Reader{
    fn take<const N:usize>(&mut self)->[u8;N]{let end=self.at.checked_add(N).unwrap();let out=self.bytes[self.at..end].try_into().unwrap();self.at=end;out}
    fn u(&mut self)->usize{u32::from_le_bytes(self.take())as usize}
    fn real(&mut self)->f64{f64::from_le_bytes(self.take())}
    fn ints(&mut self,n:usize)->Vec<i32>{(0..n).map(|_|i32::from_le_bytes(self.take())).collect()}
    fn complex(&mut self,n:usize)->Vec<C>{(0..n).map(|_|C{re:self.real(),im:self.real()}).collect()}
    fn family(&mut self,axes:usize,width:usize)->FourierMatrices{let count=self.u();(0..count).map(|_|(self.ints(axes),self.complex(width))).collect()}
}
struct Matrix{real:Vec<f64>,imag:Vec<f64>}
impl Matrix{fn from(a:Vec<C>)->Self{Self{real:a.iter().map(|z|z.re).collect(),imag:a.iter().map(|z|z.im).collect()}}fn view(&self)->MatrixView<'_>{MatrixView{real:&self.real,imag:&self.imag}}}
fn compare_family(actual:&FourierMatrices,expected:&FourierMatrices,label:&str)->(f64,f64){
    let keys:Modes=actual.keys().chain(expected.keys()).cloned().collect();let(mut difference,mut scale)=(0_f64,0_f64);
    for q in keys{let width=actual.get(&q).or_else(||expected.get(&q)).unwrap().len();
        for i in 0..width{let a=actual.get(&q).map_or(C::default(),|x|x[i]);let b=expected.get(&q).map_or(C::default(),|x|x[i]);
            difference=difference.hypot((a.re-b.re).hypot(a.im-b.im));scale=scale.hypot(b.re.hypot(b.im));
        }
    }
    let relative=difference/scale.max(f64::MIN_POSITIVE);println!("native_model_parity {label}: absolute_hs={difference:e}, relative_hs={relative:e}");(difference,relative)
}
fn row_to_column(a:&[C],rows:usize,cols:usize)->Vec<C>{(0..rows*cols).map(|i|a[(i%rows)*cols+i/rows]).collect()}
fn row_product(lap:&Lapack,a:&[C],b:&[C],m:usize,n:usize,k:usize)->Vec<C>{
    let out=lap.multiply_complex(&row_to_column(a,m,k),&row_to_column(b,k,n),m,n,k).unwrap();
    (0..m*n).map(|i|out[i%n*m+i/n]).collect()
}
fn mul(a:C,b:C)->C{C{re:a.re*b.re-a.im*b.im,im:a.re*b.im+a.im*b.re}}
fn conj(a:C)->C{C{re:a.re,im:-a.im}}

/// Test-only fresh preparation from the ORIGINAL Model prefix, not expected
/// factors or candidate arrays. Current exported four-near fixture uses its
/// global index initial; production full initial data is a separate borrowed
/// InitialView and is covered by the coordinate tests.
pub(super) fn fresh_owned_for_slice()->(OwnedPrepared,usize,Vec<Charge>){
    let path=PathBuf::from(std::env::var("ATOM_PREPARATION_FIXTURE").expect("Set current original Model fixture"));
    let mut r=Reader{bytes:std::fs::read(path).unwrap(),at:0};assert_eq!(&r.take::<8>(),b"AWBPREP1");
    let(n,axes,nd,nc,initial)=(r.u(),r.u(),r.u(),r.u(),r.u());let d=n*n;
    let ground:Vec<_>=r.ints(n).iter().map(|&x|x==1).collect();let parts:Vec<_>=(0..axes).map(|_|FrequencyParts{hi:r.real(),lo:r.real()}).collect();let h=Matrix::from(r.complex(d));
    let mut drive_data=Vec::new();let mut indices=Vec::new();for _ in 0..nd{let omega=r.real();indices.push(r.ints(axes));drive_data.push((omega,Matrix::from(r.complex(d)),Matrix::from(r.complex(d))));}
    let channels:Vec<_>=(0..nc).map(|_|Matrix::from(r.complex(d))).collect();let collapse:Vec<_>=channels.iter().map(Matrix::view).collect();
    let drives:Vec<_>=drive_data.iter().map(|(omega,c,s)|DriveView{omega:*omega,cosine:c.view(),sine:s.view()}).collect();
    let model=ModelView{dimension:n,hamiltonian:h.view(),collapse:&collapse,drives:&drives,ground:&ground,initial_index:initial};
    let mut modes=Modes::from([vec![0;axes]]);for q in &indices{modes.insert(q.clone());modes.insert(q.iter().map(|&i|-i).collect());}
    let owner=OwnedPrepared::prepare(&model,PhaseBasis::from_parts(&parts,&indices).unwrap(),GaugePolicy::UnsafeComponentAligned,KernelPolicy::HybridBlas).unwrap();
    (owner,initial,modes.into_iter().collect())
}

#[test]
#[ignore="Requires an explicitly supplied original full32 Model fixture and expected coordinate factors"]
fn preparation_model_actual_current_full32_parity(){
    // Explicit opt-in fixture prevents an absent real Model from looking like
    // a passed numerical test. It is not part of the earlier tiny filters.
    let path=PathBuf::from(std::env::var("ATOM_PREPARATION_FIXTURE").expect("Set an explicitly generated current-Model fixture"));
    let mut r=Reader{bytes:std::fs::read(path).unwrap(),at:0};assert_eq!(&r.take::<8>(),b"AWBPREP1");
    let(n,axes,nd,nc,initial)=(r.u(),r.u(),r.u(),r.u(),r.u());assert_eq!(n,32);let d=n*n;
    let ground:Vec<_>=r.ints(n).iter().map(|&x|x==1).collect();let g=ground.iter().filter(|&&x|x).count();let e=n-g;
    let parts:Vec<_>=(0..axes).map(|_|FrequencyParts{hi:r.real(),lo:r.real()}).collect();let h=Matrix::from(r.complex(d));
    let mut drive_data=Vec::new();let mut indices=Vec::new();for _ in 0..nd{let omega=r.real();indices.push(r.ints(axes));drive_data.push((omega,Matrix::from(r.complex(d)),Matrix::from(r.complex(d))));}
    let collapse_data:Vec<_>=(0..nc).map(|_|Matrix::from(r.complex(d))).collect();
    let gauges:Vec<_>=(0..n).map(|_|r.ints(axes)).collect();let expected_h=r.family(axes,d);
    let expected_c:Vec<_>=(0..nc).map(|_|r.family(axes,d)).collect();let expected_s=r.family(axes,d);let expected_k=r.family(axes,g*g*e*e);assert_eq!(r.at,r.bytes.len());
    let drives:Vec<_>=drive_data.iter().map(|(omega,cosine,sine)|DriveView{omega:*omega,cosine:cosine.view(),sine:sine.view()}).collect();
    let collapse:Vec<_>=collapse_data.iter().map(Matrix::view).collect();let model=ModelView{dimension:n,hamiltonian:h.view(),collapse:&collapse,drives:&drives,ground:&ground,initial_index:initial};
    let phases=PhaseBasis::from_parts(&parts,&indices).unwrap();let lap=Arc::new(Lapack::load().unwrap());let began=Instant::now();
    let source=GaugedSource::prepare_with_lapack(&model,phases,GaugePolicy::UnsafeComponentAligned,&lap).unwrap();
    assert_eq!(source.gauges,gauges,"Gauge selection must match current request policy before any Hilbert comparison");
    let(diff,_)=compare_family(&source.hamiltonian,&expected_h,"original_gauged_H");assert_eq!(diff,0.);
    assert_eq!(source.collapse.len(),expected_c.len());for(i,(a,b))in source.collapse.iter().zip(&expected_c).enumerate(){let(diff,_)=compare_family(a,b,&format!("original_channel_{i}"));assert_eq!(diff,0.);}
    let(s,_)=primitives::select_optical_s(&lap,&source).unwrap();let(_,srel)=compare_family(&s,&expected_s,"bare_diagonal_S");assert!(srel<2e-13);
    let source_ms=began.elapsed().as_secs_f64()*1000.;let primitive_start=Instant::now();let prepared=primitives::prepare(&lap,source).unwrap();
    let primitive_ms=primitive_start.elapsed().as_secs_f64()*1000.;let b=&prepared.hilbert.basis_column_major;
    // Degenerate ground eigenspaces and Schur eigenvector phases are free.
    // Compare complete coefficients after restoring the original Hilbert basis.
    let bh:Vec<_>=(0..d).map(|i|conj(b[(i%n)*n+i/n])).collect();
    let restore=|family:&FourierMatrices|family.iter().map(|(q,a)|(q.clone(),transform_matrix(&lap,&bh,a,n).unwrap())).collect();
    let(_,hrel)=compare_family(&restore(&prepared.hilbert.source.hamiltonian),&expected_h,"restored_H");assert!(hrel<2e-13);
    for(a,b)in prepared.hilbert.source.collapse.iter().zip(&expected_c){let(_,relative)=compare_family(&restore(a),b,"restored_channel");assert!(relative<2e-13);}
    let(_,sback)=compare_family(&restore(&prepared.s),&expected_s,"restored_S");assert!(sback<5e-13);
    let gi=&prepared.hilbert.original_ground_indices;let ei=&prepared.hilbert.original_excited_indices;
    let mut left=vec![C::default();g*g*g*g];let mut right=vec![C::default();e*e*e*e];
    for a in 0..g{for bb in 0..g{for i in 0..g{for j in 0..g{left[(a*g+bb)*g*g+i*g+j]=mul(b[gi[a]+i*n],conj(b[gi[bb]+j*n]));}}}}
    for i in 0..e{for j in 0..e{for a in 0..e{for bb in 0..e{right[(i*e+j)*e*e+a*e+bb]=mul(conj(b[ei[a]+(g+i)*n]),b[ei[bb]+(g+j)*n]);}}}}
    let restored_k=prepared.k_hat.iter().map(|(q,k)|{let temp=row_product(&lap,&left,k,g*g,e*e,g*g);(q.clone(),row_product(&lap,&temp,&right,g*g,e*e,e*e))}).collect();
    let(_,krel)=compare_family(&restored_k,&expected_k,"restored_full_Khat");assert!(krel<2e-9);
    assert!(prepared.shear_diagnostics.enabled,"Current Model comparison requires the same complete K preparation, not a silent auxiliary fallback");
    let owned=OwnedPrepared::from_primitives(prepared,KernelPolicy::Csr,lap).unwrap();
    let zero=vec![0;axes];let plan=owned.pencil(&[zero.clone()],&[zero],4).unwrap();assert!(plan.diagnostics.complete_reverse_dependencies);
    println!("native_model_parity setup_ms={{\"gaugeAndS\":{source_ms},\"primitives\":{primitive_ms},\"totalIncludingParity\":{}}}",began.elapsed().as_secs_f64()*1000.);
}
