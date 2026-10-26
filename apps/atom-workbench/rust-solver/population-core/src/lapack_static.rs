//! Native LP64 static OpenBLAS ABI. Algorithms remain in lapack.rs.
use super::*;
extern "C" {
    fn dgeev_(a0: *const c_char, a1: *const c_char, a2: *const i32, a3: *mut f64, a4: *const i32, a5: *mut f64, a6: *mut f64, a7: *mut f64, a8: *const i32, a9: *mut f64, a10: *const i32, a11: *mut f64, a12: *const i32, a13: *mut i32);
    fn dgesv_(a0: *const i32, a1: *const i32, a2: *mut f64, a3: *const i32, a4: *mut i32, a5: *mut f64, a6: *const i32, a7: *mut i32) -> i32;
    fn dgetrf_(a0: *const i32, a1: *const i32, a2: *mut f64, a3: *const i32, a4: *mut i32, a5: *mut i32) -> i32;
    fn dgetrs_(a0: *const c_char, a1: *const i32, a2: *const i32, a3: *const f64, a4: *const i32, a5: *const i32, a6: *mut f64, a7: *const i32, a8: *mut i32) -> i32;
    fn zheev_(a0: *const c_char, a1: *const c_char, a2: *const i32, a3: *mut Complex64, a4: *const i32, a5: *mut f64, a6: *mut Complex64, a7: *const i32, a8: *mut f64, a9: *mut i32);
    fn dgemm_(a0: *const c_char, a1: *const c_char, a2: *const i32, a3: *const i32, a4: *const i32, a5: *const f64, a6: *const f64, a7: *const i32, a8: *const f64, a9: *const i32, a10: *const f64, a11: *mut f64, a12: *const i32);
    fn zgemm_(a0: *const c_char, a1: *const c_char, a2: *const i32, a3: *const i32, a4: *const i32, a5: *const Complex64, a6: *const Complex64, a7: *const i32, a8: *const Complex64, a9: *const i32, a10: *const Complex64, a11: *mut Complex64, a12: *const i32);
    fn zgeev_(jobvl:*const c_char,jobvr:*const c_char,n:*const i32,a:*mut Complex64,lda:*const i32,w:*mut Complex64,vl:*mut Complex64,ldvl:*const i32,vr:*mut Complex64,ldvr:*const i32,work:*mut Complex64,lwork:*const i32,rwork:*mut f64,info:*mut i32);
    fn zgees_(a0: *const c_char, a1: *const c_char, a2: Option<unsafe extern "C" fn(*const Complex64) -> i32>, a3: *const i32, a4: *mut Complex64, a5: *const i32, a6: *mut i32, a7: *mut Complex64, a8: *mut Complex64, a9: *const i32, a10: *mut Complex64, a11: *const i32, a12: *mut f64, a13: *mut i32, a14: *mut i32);
    fn ztrsyl_(a0: *const c_char, a1: *const c_char, a2: *const i32, a3: *const i32, a4: *const i32, a5: *const Complex64, a6: *const i32, a7: *const Complex64, a8: *const i32, a9: *mut Complex64, a10: *const i32, a11: *mut f64, a12: *mut i32);
    fn zgetrf_(a0: *const i32, a1: *const i32, a2: *mut Complex64, a3: *const i32, a4: *mut i32, a5: *mut i32) -> i32;
    fn zgetrs_(a0: *const c_char, a1: *const i32, a2: *const i32, a3: *const Complex64, a4: *const i32, a5: *const i32, a6: *mut Complex64, a7: *const i32, a8: *mut i32) -> i32;
}
unsafe extern "C" fn atom_static_dgesv_(a0: *const i32, a1: *const i32, a2: *mut f64, a3: *const i32, a4: *mut i32, a5: *mut f64, a6: *const i32, a7: *mut i32) { let _ = dgesv_(a0, a1, a2, a3, a4, a5, a6, a7); }
unsafe extern "C" fn atom_static_dgetrf_(a0: *const i32, a1: *const i32, a2: *mut f64, a3: *const i32, a4: *mut i32, a5: *mut i32) { let _ = dgetrf_(a0, a1, a2, a3, a4, a5); }
unsafe extern "C" fn atom_static_dgetrs_(a0: *const c_char, a1: *const i32, a2: *const i32, a3: *const f64, a4: *const i32, a5: *const i32, a6: *mut f64, a7: *const i32, a8: *mut i32) { let _ = dgetrs_(a0, a1, a2, a3, a4, a5, a6, a7, a8); }
unsafe extern "C" fn atom_static_zgetrf_(a0: *const i32, a1: *const i32, a2: *mut Complex64, a3: *const i32, a4: *mut i32, a5: *mut i32) { let _ = zgetrf_(a0, a1, a2, a3, a4, a5); }
unsafe extern "C" fn atom_static_zgetrs_(a0: *const c_char, a1: *const i32, a2: *const i32, a3: *const Complex64, a4: *const i32, a5: *const i32, a6: *mut Complex64, a7: *const i32, a8: *mut i32) { let _ = zgetrs_(a0, a1, a2, a3, a4, a5, a6, a7, a8); }
extern "C" {fn gotoblas_init();fn openblas_set_num_threads(threads:i32);}
pub(super) fn load()->Result<Lapack,String>{
 unsafe{gotoblas_init();openblas_set_num_threads(1);}
 Ok(Lapack{
 geev:dgeev_,gesv:atom_static_dgesv_,getrf:atom_static_dgetrf_,getrs:atom_static_dgetrs_,heev:zheev_,gemm:dgemm_,zgemm:zgemm_,zgees:zgees_,zgeev:zgeev_,ztrsyl:ztrsyl_,zgetrf:atom_static_zgetrf_,zgetrs:atom_static_zgetrs_,})
}
