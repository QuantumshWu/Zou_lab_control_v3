use std::ffi::c_char;
#[cfg(not(feature="static-numerics"))]
use std::ffi::c_void;
#[cfg(not(feature="static-numerics"))]
use std::path::PathBuf;
#[cfg(feature="static-numerics")]
#[path="lapack_static.rs"]
mod static_api;

type GeEv = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const i32,
    *mut f64,
    *const i32,
    *mut f64,
    *mut f64,
    *mut f64,
    *const i32,
    *mut f64,
    *const i32,
    *mut f64,
    *const i32,
    *mut i32,
);
type GeSv = unsafe extern "C" fn(
    *const i32,
    *const i32,
    *mut f64,
    *const i32,
    *mut i32,
    *mut f64,
    *const i32,
    *mut i32,
);
type GetRf = unsafe extern "C" fn(*const i32, *const i32, *mut f64, *const i32, *mut i32, *mut i32);
type GetRs = unsafe extern "C" fn(
    *const c_char,
    *const i32,
    *const i32,
    *const f64,
    *const i32,
    *const i32,
    *mut f64,
    *const i32,
    *mut i32,
);
type Gemm = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const i32,
    *const i32,
    *const i32,
    *const f64,
    *const f64,
    *const i32,
    *const f64,
    *const i32,
    *const f64,
    *mut f64,
    *const i32,
);
#[cfg(not(feature="static-numerics"))]
type SetThreads = unsafe extern "C" fn(i32);
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct Complex64 {
    pub re: f64,
    pub im: f64,
}
type HeEv = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const i32,
    *mut Complex64,
    *const i32,
    *mut f64,
    *mut Complex64,
    *const i32,
    *mut f64,
    *mut i32,
);
type ZGemm = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const i32,
    *const i32,
    *const i32,
    *const Complex64,
    *const Complex64,
    *const i32,
    *const Complex64,
    *const i32,
    *const Complex64,
    *mut Complex64,
    *const i32,
);
type ZGeEv = unsafe extern "C" fn(
    *const c_char,*const c_char,*const i32,*mut Complex64,*const i32,*mut Complex64,
    *mut Complex64,*const i32,*mut Complex64,*const i32,*mut Complex64,*const i32,*mut f64,*mut i32,
);
type ZGetRf = unsafe extern "C" fn(*const i32, *const i32, *mut Complex64, *const i32, *mut i32, *mut i32);
type ZGetRs = unsafe extern "C" fn(
    *const c_char, *const i32, *const i32, *const Complex64, *const i32,
    *const i32, *mut Complex64, *const i32, *mut i32,
);

#[cfg(not(feature="static-numerics"))]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryW(name: *const u16) -> *mut c_void;
    fn GetProcAddress(module: *mut c_void, name: *const c_char) -> *mut c_void;
}
/// Resolved LAPACK/BLAS entry points. The library stays loaded for the
/// process, so only function pointers are kept and the owner is Send + Sync.
pub struct Lapack {
    geev: GeEv,
    gesv: GeSv,
    getrf: GetRf,
    getrs: GetRs,
    heev: HeEv,
    gemm: Gemm,
    zgemm: ZGemm,
    zgeev: ZGeEv,
    zgetrf: ZGetRf,
    zgetrs: ZGetRs,
}
/// Strided column-major complex GEMM on sub-matrices. It copies only the
/// immutable BLAS entry point, so it is Send + Sync and may
/// be called from worker threads while OpenBLAS itself runs single-threaded.
#[derive(Clone, Copy)]
pub struct ComplexGemm<'a> {
    zgemm: ZGemm,
    _owner: std::marker::PhantomData<&'a ()>,
}
impl ComplexGemm<'_> {
    /// C = alpha * A * B + beta * C, where A is m x k (leading dimension lda),
    /// B is k x n (ldb) and C is m x n (ldc), all column-major and untransposed.
    /// Only shapes are validated; values are never scanned.
    #[allow(clippy::too_many_arguments)]
    pub fn multiply(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        beta: Complex64,
        c: &mut [Complex64],
        ldc: usize,
    ) -> Result<(), String> {
        self.call(false, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    /// C = alpha * A^H * B + beta * C, where A is k x m (leading dimension
    /// lda); otherwise as `multiply`.
    #[allow(clippy::too_many_arguments)]
    pub fn multiply_adjoint(
        &self,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        beta: Complex64,
        c: &mut [Complex64],
        ldc: usize,
    ) -> Result<(), String> {
        self.call(true, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
    }
    #[allow(clippy::too_many_arguments)]
    fn call(
        &self,
        adjoint: bool,
        m: usize,
        n: usize,
        k: usize,
        alpha: Complex64,
        a: &[Complex64],
        lda: usize,
        b: &[Complex64],
        ldb: usize,
        beta: Complex64,
        c: &mut [Complex64],
        ldc: usize,
    ) -> Result<(), String> {
        fn span(rows: usize, cols: usize, ld: usize) -> Option<usize> {
            if rows == 0 || cols == 0 { Some(0) } else { ld.checked_mul(cols - 1)?.checked_add(rows) }
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let (a_rows, a_cols) = if adjoint { (k, m) } else { (m, k) };
        let fits = |v: usize| v <= i32::MAX as usize;
        let valid = lda >= a_rows.max(1) && ldb >= k.max(1) && ldc >= m
            && [m, n, k, lda, ldb, ldc].iter().all(|&v| fits(v))
            && span(a_rows, a_cols, lda).is_some_and(|v| v <= a.len())
            && span(k, n, ldb).is_some_and(|v| v <= b.len())
            && span(m, n, ldc).is_some_and(|v| v <= c.len());
        if !valid {
            return Err("Invalid strided complex GEMM dimensions".into());
        }
        let (transa, no) = ((if adjoint { b'C' } else { b'N' }) as c_char, b'N' as c_char);
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let (la, lb, lc) = (lda as i32, ldb as i32, ldc as i32);
        unsafe {
            (self.zgemm)(&transa, &no, &mi, &ni, &ki, &alpha, a.as_ptr(), &la, b.as_ptr(), &lb, &beta, c.as_mut_ptr(), &lc);
        }
        Ok(())
    }
}
impl Lapack {
    /// Thread-shareable strided complex GEMM entry (see `ComplexGemm`).
    pub fn complex_gemm(&self) -> ComplexGemm<'_> {
        ComplexGemm { zgemm: self.zgemm, _owner: std::marker::PhantomData }
    }

    /// BLAS always runs single-threaded: the lattice kernels parallelize
    /// over output modes (`ATOM_SOLVER_THREADS`), and threaded level-3 OpenBLAS
    /// both oversubscribes that pool and keeps per-call job tables on the
    /// calling thread's stack.
    #[cfg(feature="static-numerics")]
    pub fn load()->Result<Self,String>{static_api::load()}
    #[cfg(not(feature="static-numerics"))]
    pub fn load() -> Result<Self, String> {
        let path = std::env::var_os("ATOM_OPENBLAS_DLL")
            .map(PathBuf::from)
            .or_else(|| {
                std::env::current_exe()
                    .ok()
                    .map(|p| p.with_file_name("libscipy_openblas.dll"))
            })
            .ok_or("Set ATOM_OPENBLAS_DLL to a LAPACK/OpenBLAS DLL")?;
        let wide: Vec<u16> = path
            .to_string_lossy()
            .encode_utf16()
            .chain(Some(0))
            .collect();
        let library = unsafe { LoadLibraryW(wide.as_ptr()) };
        if library.is_null() {
            return Err(format!(
                "Cannot load {}. Set ATOM_OPENBLAS_DLL to a SciPy-prefixed LP64 OpenBLAS DLL.",
                path.display()
            ));
        }
        unsafe fn symbol<T: Copy>(lib: *mut c_void, name: &[u8]) -> Result<T, String> {
            let ptr = GetProcAddress(lib, name.as_ptr().cast());
            if ptr.is_null() {
                return Err(format!(
                    "Missing LAPACK symbol {}",
                    String::from_utf8_lossy(name)
                ));
            }
            Ok(std::mem::transmute_copy(&ptr))
        }
        unsafe {
            let set_threads: SetThreads = symbol(library, b"scipy_openblas_set_num_threads\0")?;
            set_threads(1);
            Ok(Self {
                geev: symbol(library, b"scipy_dgeev_\0")?,
                gesv: symbol(library, b"scipy_dgesv_\0")?,
                getrf: symbol(library, b"scipy_dgetrf_\0")?,
                getrs: symbol(library, b"scipy_dgetrs_\0")?,
                heev: symbol(library, b"scipy_zheev_\0")?,
                gemm: symbol(library, b"scipy_dgemm_\0")?,
                zgemm: symbol(library, b"scipy_zgemm_\0")?,
                zgeev: symbol(library, b"scipy_zgeev_\0")?,
                zgetrf: symbol(library, b"scipy_zgetrf_\0")?,
                zgetrs: symbol(library, b"scipy_zgetrs_\0")?,
            })
        }
    }
    /// General complex right eigenvectors, COLUMN-MAJOR. Callers validate
    /// conditioning/backward error themselves.
    pub fn complex_eigen_right(&self,a:&[Complex64],n:usize)->Result<(Vec<Complex64>,Vec<Complex64>),String>{
        if n==0||n>i32::MAX as usize/2||n.checked_mul(n)!=Some(a.len())||a.iter().any(|z|!z.re.is_finite()||!z.im.is_finite()){
            return Err("Invalid general complex eigensystem input".into());
        }
        let function=self.zgeev;
        let ni=n as i32;let no=b'N' as c_char;let yes=b'V' as c_char;let mut input=a.to_vec();
        let mut eigen=vec![Complex64::default();n];let mut right=vec![Complex64::default();n*n];let mut left=[Complex64::default()];
        let mut real_work=vec![0.;2*n];let mut query=[Complex64::default()];let mut info=0;
        unsafe{function(&no,&yes,&ni,input.as_mut_ptr(),&ni,eigen.as_mut_ptr(),left.as_mut_ptr(),&1,right.as_mut_ptr(),&ni,
            query.as_mut_ptr(),&-1,real_work.as_mut_ptr(),&mut info);}
        if info!=0||!query[0].re.is_finite()||query[0].re>i32::MAX as f64{return Err(format!("LAPACK zgeev workspace query failed: {info}"));}
        let length=(query[0].re as i32).max(2*ni);let mut work=vec![Complex64::default();length as usize];input.copy_from_slice(a);
        unsafe{function(&no,&yes,&ni,input.as_mut_ptr(),&ni,eigen.as_mut_ptr(),left.as_mut_ptr(),&1,right.as_mut_ptr(),&ni,
            work.as_mut_ptr(),&length,real_work.as_mut_ptr(),&mut info);}
        if info!=0||eigen.iter().chain(&right).any(|z|!z.re.is_finite()||!z.im.is_finite()){return Err(format!("LAPACK zgeev unresolved: {info}"));}
        Ok((eigen,right))
    }
    pub fn complex_factor(&self, a: &[Complex64], n: usize) -> Result<(Vec<Complex64>, Vec<i32>), String> {
        if n == 0 || n > i32::MAX as usize || a.len() != n*n { return Err("Invalid complex LU dimensions".into()); }
        let ni = n as i32;
        let mut lu = a.to_vec();
        let mut pivots = vec![0; n];
        let mut info = 0;
        unsafe { (self.zgetrf)(&ni, &ni, lu.as_mut_ptr(), &ni, pivots.as_mut_ptr(), &mut info); }
        if info != 0 { return Err(format!("LAPACK zgetrf failed: {info}")); }
        Ok((lu, pivots))
    }
    /// Solve every column-major RHS column with one complex LU factorization.
    pub fn complex_solve_matrix_factored(&self, lu: &[Complex64], pivots: &[i32], rhs: &[Complex64], n: usize) -> Result<Vec<Complex64>, String> {
        if n == 0 || n > i32::MAX as usize || lu.len() != n*n || pivots.len() != n || rhs.len() % n != 0 || rhs.len() / n > i32::MAX as usize {
            return Err("Invalid complex multiple-RHS LU solve dimensions".into());
        }
        let mut x = rhs.to_vec();
        if x.is_empty() { return Ok(x); }
        let (ni, columns) = (n as i32, (rhs.len() / n) as i32);
        let mut info = 0;
        unsafe { (self.zgetrs)(&(b'N' as c_char), &ni, &columns, lu.as_ptr(), &ni, pivots.as_ptr(), x.as_mut_ptr(), &ni, &mut info); }
        if info != 0 || x.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) { return Err(format!("LAPACK zgetrs multiple-RHS failed: {info}")); }
        Ok(x)
    }
    pub fn eigen(&self, a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>), String> {
        let ni = n as i32;
        let mut input = a.to_vec();
        let mut wr = vec![0.; n];
        let mut wi = vec![0.; n];
        let mut vl = [0.];
        let mut vr = vec![0.; n * n];
        let mut query = [0.];
        let mut info = 0;
        let no = b'N' as c_char;
        let yes = b'V' as c_char;
        unsafe {
            (self.geev)(
                &no,
                &yes,
                &ni,
                input.as_mut_ptr(),
                &ni,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &1,
                vr.as_mut_ptr(),
                &ni,
                query.as_mut_ptr(),
                &-1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgeev workspace query failed: {info}"));
        }
        let len = (query[0] as i32).max(4 * ni);
        let mut work = vec![0.; len as usize];
        unsafe {
            (self.geev)(
                &no,
                &yes,
                &ni,
                input.as_mut_ptr(),
                &ni,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                vl.as_mut_ptr(),
                &1,
                vr.as_mut_ptr(),
                &ni,
                work.as_mut_ptr(),
                &len,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgeev failed: {info}"));
        }
        Ok((wr, wi, vr))
    }
    pub fn solve(&self, a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>, String> {
        let ni = n as i32;
        let mut matrix = a.to_vec();
        let mut rhs = b.to_vec();
        let mut piv = vec![0; n];
        let mut info = 0;
        unsafe {
            (self.gesv)(
                &ni,
                &1,
                matrix.as_mut_ptr(),
                &ni,
                piv.as_mut_ptr(),
                rhs.as_mut_ptr(),
                &ni,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgesv failed: {info}"));
        }
        Ok(rhs)
    }
    /// Hermitian complex eigensystem (zheev), column-major input/output.
    pub fn hermitian_eigen(
        &self,
        a: &[Complex64],
        n: usize,
    ) -> Result<(Vec<f64>, Vec<Complex64>), String> {
        if n == 0
            || n > i32::MAX as usize
            || n.checked_mul(n) != Some(a.len())
            || a.iter().any(|x| !x.re.is_finite() || !x.im.is_finite())
        {
            return Err("Invalid complex Hermitian eigensystem input".into());
        }
        let ni = n as i32;
        let mut matrix = a.to_vec();
        let mut values = vec![0.; n];
        let mut query = [Complex64::default()];
        let mut real_work = vec![0.; (3 * n).saturating_sub(2).max(1)];
        let mut info = 0;
        let yes = b'V' as c_char;
        let upper = b'U' as c_char;
        unsafe {
            (self.heev)(
                &yes,
                &upper,
                &ni,
                matrix.as_mut_ptr(),
                &ni,
                values.as_mut_ptr(),
                query.as_mut_ptr(),
                &-1,
                real_work.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK zheev workspace query failed: {info}"));
        }
        let length = (query[0].re as i32).max(2 * ni - 1);
        let mut work = vec![Complex64::default(); length as usize];
        unsafe {
            (self.heev)(
                &yes,
                &upper,
                &ni,
                matrix.as_mut_ptr(),
                &ni,
                values.as_mut_ptr(),
                work.as_mut_ptr(),
                &length,
                real_work.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK zheev failed: {info}"));
        }
        Ok((values, matrix))
    }
    pub fn factor(&self, a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<i32>), String> {
        let ni = n as i32;
        let mut matrix = a.to_vec();
        let mut piv = vec![0; n];
        let mut info = 0;
        unsafe {
            (self.getrf)(
                &ni,
                &ni,
                matrix.as_mut_ptr(),
                &ni,
                piv.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgetrf failed: {info}"));
        }
        Ok((matrix, piv))
    }
    /// Solve all column-major RHS columns with one factorization/call.
    pub fn solve_matrix_factored(
        &self,
        a: &[f64],
        pivots: &[i32],
        b: &[f64],
        n: usize,
    ) -> Result<Vec<f64>, String> {
        if n == 0
            || n > i32::MAX as usize
            || n.checked_mul(n) != Some(a.len())
            || pivots.len() != n
            || b.len() % n != 0
            || b.len() / n > i32::MAX as usize
        {
            return Err("Invalid multiple-RHS factor solve dimensions".into());
        }
        if b.is_empty() {
            return Ok(vec![]);
        }
        let ni = n as i32;
        let columns = (b.len() / n) as i32;
        let mut rhs = b.to_vec();
        let mut info = 0;
        let no = b'N' as c_char;
        unsafe {
            (self.getrs)(
                &no,
                &ni,
                &columns,
                a.as_ptr(),
                &ni,
                pivots.as_ptr(),
                rhs.as_mut_ptr(),
                &ni,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgetrs multiple-RHS failed: {info}"));
        }
        Ok(rhs)
    }
    pub fn multiply(&self, a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
        let mut out = vec![0.; m * n];
        let no = b'N' as c_char;
        let mi = m as i32;
        let ni = n as i32;
        let ki = k as i32;
        unsafe {
            (self.gemm)(
                &no,
                &no,
                &mi,
                &ni,
                &ki,
                &1.,
                a.as_ptr(),
                &mi,
                b.as_ptr(),
                &ki,
                &0.,
                out.as_mut_ptr(),
                &mi,
            );
        }
        out
    }
    /// General column-major `C = alpha op(A) op(B) + beta C` with explicit
    /// leading dimensions; `op(X)` is `X^T` when the flag is set. Shapes and
    /// spans are validated in O(1); data finiteness is the caller's concern.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &self, transpose_a: bool, transpose_b: bool, m: usize, n: usize, k: usize,
        alpha: f64, a: &[f64], lda: usize, b: &[f64], ldb: usize,
        beta: f64, c: &mut [f64], ldc: usize,
    ) -> Result<(), String> {
        let span = |rows: usize, cols: usize, ld: usize| -> Option<usize> {
            if rows == 0 || cols == 0 { Some(0) } else { (cols - 1).checked_mul(ld)?.checked_add(rows) }
        };
        let (a_rows, a_cols) = if transpose_a { (k, m) } else { (m, k) };
        let (b_rows, b_cols) = if transpose_b { (n, k) } else { (k, n) };
        if [m, n, k, lda, ldb, ldc].iter().any(|&v| v > i32::MAX as usize)
            || lda < a_rows.max(1) || ldb < b_rows.max(1) || ldc < m.max(1)
            || span(a_rows, a_cols, lda).is_none_or(|s| s > a.len())
            || span(b_rows, b_cols, ldb).is_none_or(|s| s > b.len())
            || span(m, n, ldc).is_none_or(|s| s > c.len())
        {
            return Err("Invalid general real BLAS matrix dimensions".into());
        }
        if m == 0 || n == 0 { return Ok(()); }
        let ta = if transpose_a { b'T' } else { b'N' } as c_char;
        let tb = if transpose_b { b'T' } else { b'N' } as c_char;
        let (mi, ni, ki, la, lb, lc) = (m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32);
        unsafe {
            (self.gemm)(&ta, &tb, &mi, &ni, &ki, &alpha, a.as_ptr(), &la, b.as_ptr(), &lb, &beta, c.as_mut_ptr(), &lc);
        }
        Ok(())
    }
    /// Complex C = A B, column-major, with finite inputs and output.
    #[cfg(any(test, feature = "validation"))]
    pub fn multiply_complex(&self, a: &[Complex64], b: &[Complex64], m: usize, n: usize, k: usize) -> Result<Vec<Complex64>, String> {
        if m > i32::MAX as usize || n > i32::MAX as usize || k > i32::MAX as usize
            || m.checked_mul(k) != Some(a.len()) || k.checked_mul(n) != Some(b.len())
        {
            return Err("Invalid complex BLAS matrix dimensions".into());
        }
        if a.iter().chain(b).any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err("Complex BLAS input is not finite".into());
        }
        let mut out = vec![Complex64::default(); m.checked_mul(n).ok_or("Complex BLAS allocation overflow")?];
        if m == 0 || n == 0 || k == 0 {
            return Ok(out);
        }
        let no = b'N' as c_char;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let (one, zero) = (Complex64 { re: 1., im: 0. }, Complex64::default());
        unsafe {
            (self.zgemm)(&no, &no, &mi, &ni, &ki, &one, a.as_ptr(), &mi, b.as_ptr(), &ki, &zero, out.as_mut_ptr(), &mi);
        }
        if out.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err("Complex BLAS output is not finite".into());
        }
        Ok(out)
    }
}

#[cfg(test)]
mod added_wrapper_tests {
    use super::*;
    #[test]
    fn complex_hermitian_vectors_and_values_are_column_major() {
        let lap = Lapack::load().unwrap();
        let a = [
            Complex64 { re: 1., im: 0. },
            Complex64 { re: 0., im: -1. },
            Complex64 { re: 0., im: 1. },
            Complex64 { re: 2., im: 0. },
        ];
        let (values, vectors) = lap.hermitian_eigen(&a, 2).unwrap();
        assert!((values[0] - (3. - 5_f64.sqrt()) / 2.).abs() < 1e-14);
        for j in 0..2 {
            let mut norm = 0.;
            for i in 0..2 {
                let mut r = 0.;
                let mut m = 0.;
                for k in 0..2 {
                    let av = a[i + k * 2];
                    let v = vectors[k + j * 2];
                    r += av.re * v.re - av.im * v.im;
                    m += av.re * v.im + av.im * v.re;
                }
                assert!(
                    (r - values[j] * vectors[i + j * 2].re)
                        .hypot(m - values[j] * vectors[i + j * 2].im)
                        < 1e-14
                );
                norm += vectors[i + j * 2].re.powi(2) + vectors[i + j * 2].im.powi(2);
            }
            assert!((norm - 1.).abs() < 1e-14);
        }
    }
    #[test]
    fn factored_solve_solves_all_right_hand_side_columns() {
        let lap = Lapack::load().unwrap();
        let (lu, pivots) = lap.factor(&[2., 0., 1., 3.], 2).unwrap();
        let inverse = lap
            .solve_matrix_factored(&lu, &pivots, &[1., 0., 0., 1.], 2)
            .unwrap();
        for (a, b) in inverse.iter().zip([0.5, 0., -1. / 6., 1. / 3.]) {
            assert!((a - b).abs() < 1e-15);
        }
    }
    #[test]
    fn complex_gemm_preserves_column_major_phase() {
        let lap = Lapack::load().unwrap();
        let a = [Complex64 { re: 1., im: 2. }, Complex64 { re: 3., im: -1. }];
        let b = [Complex64 { re: 2., im: 1. }];
        let product = lap.multiply_complex(&a, &b, 2, 1, 1).unwrap();
        assert!((product[0].re - 0.).abs() < 1e-15);
        assert!((product[0].im - 5.).abs() < 1e-15);
        assert!((product[1].re - 7.).abs() < 1e-15);
        assert!((product[1].im - 1.).abs() < 1e-15);
    }
}
