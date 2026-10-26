use std::ffi::{c_char, c_void};
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
type GeCon = unsafe extern "C" fn(
    *const c_char,
    *const i32,
    *const f64,
    *const i32,
    *const f64,
    *mut f64,
    *mut f64,
    *mut i32,
    *mut i32,
);
type SyEv = unsafe extern "C" fn(
    *const c_char,
    *const c_char,
    *const i32,
    *mut f64,
    *const i32,
    *mut f64,
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
type SetThreads = unsafe extern "C" fn(i32);
type GeEs = unsafe extern "C" fn(*const c_char,*const c_char,Option<unsafe extern "C" fn(*const f64,*const f64)->i32>,*const i32,*mut f64,*const i32,*mut i32,*mut f64,*mut f64,*mut f64,*const i32,*mut f64,*const i32,*mut i32,*mut i32);
type LaQtr = unsafe extern "C" fn(*const i32,*const i32,*const i32,*const f64,*const i32,*const f64,*const f64,*mut f64,*mut f64,*mut f64,*mut i32);
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
type ZGeEs = unsafe extern "C" fn(
    *const c_char, *const c_char, Option<unsafe extern "C" fn(*const Complex64) -> i32>,
    *const i32, *mut Complex64, *const i32, *mut i32, *mut Complex64,
    *mut Complex64, *const i32, *mut Complex64, *const i32, *mut f64, *mut i32, *mut i32,
);
type ZGeEv = unsafe extern "C" fn(
    *const c_char,*const c_char,*const i32,*mut Complex64,*const i32,*mut Complex64,
    *mut Complex64,*const i32,*mut Complex64,*const i32,*mut Complex64,*const i32,*mut f64,*mut i32,
);
#[cfg(feature="static-numerics")]
extern "C"{fn zgeev_(jobvl:*const c_char,jobvr:*const c_char,n:*const i32,a:*mut Complex64,lda:*const i32,
    w:*mut Complex64,vl:*mut Complex64,ldvl:*const i32,vr:*mut Complex64,ldvr:*const i32,work:*mut Complex64,lwork:*const i32,rwork:*mut f64,info:*mut i32);}
type ZTrSyl = unsafe extern "C" fn(
    *const c_char, *const c_char, *const i32, *const i32, *const i32,
    *const Complex64, *const i32, *const Complex64, *const i32,
    *mut Complex64, *const i32, *mut f64, *mut i32,
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
pub struct Lapack {
    _library: *mut c_void,
    blas_threads: usize,
    set_threads: SetThreads,
    geev: GeEv,
    gesv: GeSv,
    getrf: GetRf,
    getrs: GetRs,
    gecon: GeCon,
    syev: SyEv,
    heev: HeEv,
    gemm: Gemm,
    zgemm: ZGemm,
    gees: GeEs,
    laqtr: LaQtr,
    zgees: ZGeEs,
    ztrsyl: ZTrSyl,
    zgetrf: ZGetRf,
    zgetrs: ZGetRs,
}
/// Narrow shared kernel for independent Sylvester applications. It copies only
/// immutable BLAS/LAPACK entry points; no raw library handle, global setters,
/// mutable factor, or workspace is shared. Its lifetime is tied to the loaded
/// Lapack owner; every FFI output buffer below is allocated inside the call.
/// Function pointers and the lifetime marker are Send + Sync without an unsafe
/// implementation. The caller keeps OpenBLAS single-threaded for outer workers.
#[derive(Clone, Copy)]
pub struct SylvesterKernel<'a> {
    zgemm: ZGemm,
    ztrsyl: ZTrSyl,
    zgetrs: ZGetRs,
    _owner: std::marker::PhantomData<&'a ()>,
}
/// Strided column-major complex GEMM on sub-matrices. Like `SylvesterKernel`
/// it copies only the immutable BLAS entry point, so it is Send + Sync and may
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
        fn span(rows: usize, cols: usize, ld: usize) -> Option<usize> {
            if rows == 0 || cols == 0 { Some(0) } else { ld.checked_mul(cols - 1)?.checked_add(rows) }
        }
        if m == 0 || n == 0 {
            return Ok(());
        }
        let fits = |v: usize| v <= i32::MAX as usize;
        let valid = lda >= m.max(1) && ldb >= k.max(1) && ldc >= m
            && [m, n, k, lda, ldb, ldc].iter().all(|&v| fits(v))
            && span(m, k, lda).is_some_and(|v| v <= a.len())
            && span(k, n, ldb).is_some_and(|v| v <= b.len())
            && span(m, n, ldc).is_some_and(|v| v <= c.len());
        if !valid {
            return Err("Invalid strided complex GEMM dimensions".into());
        }
        let no = b'N' as c_char;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let (la, lb, lc) = (lda as i32, ldb as i32, ldc as i32);
        unsafe {
            (self.zgemm)(&no, &no, &mi, &ni, &ki, &alpha, a.as_ptr(), &la, b.as_ptr(), &lb, &beta, c.as_mut_ptr(), &lc);
        }
        Ok(())
    }
}
impl SylvesterKernel<'_> {
    pub fn complex_sylvester(&self, a: &[Complex64], b: &[Complex64], rhs: &[Complex64], n: usize) -> Result<Vec<Complex64>, String> {
        self.complex_sylvester_rect(a,b,rhs,n,n)
    }
    /// Rectangular A X - X B^H = rhs. The same mature ztrsyl kernel and scale
    /// handling as the square entry; no padding or second decomposition.
    pub fn complex_sylvester_rect(&self,a:&[Complex64],b:&[Complex64],rhs:&[Complex64],m:usize,n:usize)->Result<Vec<Complex64>,String>{
        self.complex_sylvester_rect_side(a,b,rhs,m,n,false)
    }
    /// Rectangular A^H X - X B = rhs, using the SAME upper-triangular A/B.
    /// This is the HS adjoint of the forward Sylvester operator; do not also
    /// conjugate a frequency already contained in A before calling it.
    pub fn complex_sylvester_rect_adjoint(&self,a:&[Complex64],b:&[Complex64],rhs:&[Complex64],m:usize,n:usize)->Result<Vec<Complex64>,String>{
        self.complex_sylvester_rect_side(a,b,rhs,m,n,true)
    }
    fn complex_sylvester_rect_side(&self,a:&[Complex64],b:&[Complex64],rhs:&[Complex64],m:usize,n:usize,adjoint:bool)->Result<Vec<Complex64>,String>{
        if m == 0 || n == 0 || m > i32::MAX as usize || n > i32::MAX as usize || a.len() != m*m || b.len() != n*n || rhs.len() != m*n {
            return Err("Invalid complex Sylvester dimensions".into());
        }
        let mi = m as i32;
        let ni = n as i32;
        let mut x = rhs.to_vec();
        let mut scale = 1.;
        let mut info = 0;
        let ta=(if adjoint{b'C'}else{b'N'}) as c_char;let tb=(if adjoint{b'N'}else{b'C'}) as c_char;
        unsafe { (self.ztrsyl)(&ta, &tb, &-1,
            &mi, &ni, a.as_ptr(), &mi, b.as_ptr(), &ni, x.as_mut_ptr(), &mi, &mut scale, &mut info); }
        if info != 0 || !scale.is_finite() || scale <= 0. {
            return Err(format!("LAPACK ztrsyl unresolved: info={info}, scale={scale}"));
        }
        for z in &mut x { z.re /= scale; z.im /= scale; }
        if x.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) { return Err("Non-finite Sylvester solution".into()); }
        Ok(x)
    }
    pub fn complex_solve_factored(&self, lu: &[Complex64], pivots: &[i32], rhs: &[Complex64], n: usize) -> Result<Vec<Complex64>, String> {
        if n == 0 || lu.len() != n*n || pivots.len() != n || rhs.len() != n { return Err("Invalid complex LU solve dimensions".into()); }
        let ni = n as i32;
        let mut x = rhs.to_vec();
        let mut info = 0;
        unsafe { (self.zgetrs)(&(b'N' as c_char), &ni, &1, lu.as_ptr(), &ni, pivots.as_ptr(), x.as_mut_ptr(), &ni, &mut info); }
        if info != 0 || x.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) { return Err(format!("LAPACK zgetrs failed: {info}")); }
        Ok(x)
    }
    pub fn multiply_complex(
        &self,
        a: &[Complex64],
        b: &[Complex64],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<Complex64>, String> {
        if m > i32::MAX as usize
            || n > i32::MAX as usize
            || k > i32::MAX as usize
            || m.checked_mul(k) != Some(a.len())
            || k.checked_mul(n) != Some(b.len())
        {
            return Err("Invalid complex BLAS matrix dimensions".into());
        }
        if a.iter()
            .chain(b)
            .any(|v| !v.re.is_finite() || !v.im.is_finite())
        {
            return Err("Complex BLAS input is not finite".into());
        }
        let mut out =
            vec![Complex64::default(); m.checked_mul(n).ok_or("Complex BLAS allocation overflow")?];
        if m == 0 || n == 0 || k == 0 {
            return Ok(out);
        }
        let no = b'N' as c_char;
        let (mi, ni, ki) = (m as i32, n as i32, k as i32);
        let one = Complex64 { re: 1., im: 0. };
        let zero = Complex64::default();
        unsafe {
            (self.zgemm)(
                &no,
                &no,
                &mi,
                &ni,
                &ki,
                &one,
                a.as_ptr(),
                &mi,
                b.as_ptr(),
                &ki,
                &zero,
                out.as_mut_ptr(),
                &mi,
            );
        }
        if out.iter().any(|v| !v.re.is_finite() || !v.im.is_finite()) {
            return Err("Complex BLAS output is not finite".into());
        }
        Ok(out)
    }
}
impl Lapack {
    /// Load-time preference for this owner, not a query of OpenBLAS's shared
    /// process setting. Another request-local owner may temporarily change it.
    pub fn configured_blas_threads(&self) -> usize { self.blas_threads }
    fn checked_thread_count(threads:usize)->Result<i32,String>{
        if !(1..=32).contains(&threads){return Err("Explicit OpenBLAS thread count must be in1..32".into());}Ok(threads as i32)
    }
    /// Change the process setting through THIS loaded library's same entry.
    /// Call only between numerical operations, not concurrently with BLAS work.
    /// This intentionally leaves the owner's load-time preference unchanged.
    pub fn set_process_blas_threads(&self,threads:usize)->Result<(),String>{
        let threads=Self::checked_thread_count(threads)?;unsafe{(self.set_threads)(threads);}Ok(())
    }
    pub fn restore_configured_threads(&self)->Result<(),String>{self.set_process_blas_threads(self.blas_threads)}
    pub fn sylvester_kernel(&self) -> SylvesterKernel<'_> {
        SylvesterKernel { zgemm:self.zgemm, ztrsyl:self.ztrsyl, zgetrs:self.zgetrs,
            _owner:std::marker::PhantomData }
    }
    /// Thread-shareable strided complex GEMM entry (see `ComplexGemm`).
    pub fn complex_gemm(&self) -> ComplexGemm<'_> {
        ComplexGemm { zgemm: self.zgemm, _owner: std::marker::PhantomData }
    }

    pub fn load() -> Result<Self, String> {
        // Preserve the legacy environment/default/clamping semantics exactly.
        let threads=std::env::var("ATOM_BLAS_THREADS").ok().and_then(|s|s.parse::<i32>().ok()).unwrap_or(1).clamp(1,32);
        Self::load_with_threads(threads as usize)
    }
    #[cfg(feature="static-numerics")]
    pub fn load_with_threads(threads:usize)->Result<Self,String>{
        Self::checked_thread_count(threads)?;static_api::load(threads)
    }
    #[cfg(not(feature="static-numerics"))]
    pub fn load_with_threads(threads:usize)->Result<Self,String>{
        let threads=Self::checked_thread_count(threads)?;
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
            set_threads(threads);
            Ok(Self {
                _library: library,
                blas_threads: threads as usize,
                set_threads,
                geev: symbol(library, b"scipy_dgeev_\0")?,
                gesv: symbol(library, b"scipy_dgesv_\0")?,
                getrf: symbol(library, b"scipy_dgetrf_\0")?,
                getrs: symbol(library, b"scipy_dgetrs_\0")?,
                gecon: symbol(library, b"scipy_dgecon_\0")?,
                syev: symbol(library, b"scipy_dsyev_\0")?,
                heev: symbol(library, b"scipy_zheev_\0")?,
                gemm: symbol(library, b"scipy_dgemm_\0")?,
                zgemm: symbol(library, b"scipy_zgemm_\0")?,
                gees: symbol(library, b"scipy_dgees_\0")?,
                laqtr: symbol(library, b"scipy_dlaqtr_\0")?,
                zgees: symbol(library, b"scipy_zgees_\0")?,
                ztrsyl: symbol(library, b"scipy_ztrsyl_\0")?,
                zgetrf: symbol(library, b"scipy_zgetrf_\0")?,
                zgetrs: symbol(library, b"scipy_zgetrs_\0")?,
            })
        }
    }
    /// Unitary complex Schur decomposition, column-major input/output.
    pub fn complex_schur(&self, a: &[Complex64], n: usize) -> Result<(Vec<Complex64>, Vec<Complex64>), String> {
        if n == 0 || n > i32::MAX as usize || n.checked_mul(n) != Some(a.len())
            || a.iter().any(|z| !z.re.is_finite() || !z.im.is_finite()) {
            return Err("Invalid complex Schur input".into());
        }
        let ni = n as i32;
        let mut t = a.to_vec();
        let mut q = vec![Complex64::default(); n*n];
        let mut eigen = vec![Complex64::default(); n];
        let mut real_work = vec![0.; n];
        let mut bool_work = vec![0; n];
        let mut selected = 0;
        let mut info = 0;
        let mut query = [Complex64::default()];
        let yes = b'V' as c_char;
        let no = b'N' as c_char;
        unsafe { (self.zgees)(&yes, &no, None, &ni, t.as_mut_ptr(), &ni,
            &mut selected, eigen.as_mut_ptr(), q.as_mut_ptr(), &ni, query.as_mut_ptr(), &-1,
            real_work.as_mut_ptr(), bool_work.as_mut_ptr(), &mut info); }
        if info != 0 { return Err(format!("LAPACK zgees workspace query failed: {info}")); }
        let length = (query[0].re as i32).max(2*ni);
        let mut work = vec![Complex64::default(); length as usize];
        unsafe { (self.zgees)(&yes, &no, None, &ni, t.as_mut_ptr(), &ni,
            &mut selected, eigen.as_mut_ptr(), q.as_mut_ptr(), &ni, work.as_mut_ptr(), &length,
            real_work.as_mut_ptr(), bool_work.as_mut_ptr(), &mut info); }
        if info != 0 { return Err(format!("LAPACK zgees failed: {info}")); }
        Ok((t, q))
    }
    /// General complex right eigenvectors, COLUMN-MAJOR. Optional research
    /// callers must validate conditioning/backward error; this never replaces
    /// the mature Schur API. Resolve the extra symbol only when requested.
    pub fn complex_eigen_right(&self,a:&[Complex64],n:usize)->Result<(Vec<Complex64>,Vec<Complex64>),String>{
        if n==0||n>i32::MAX as usize/2||n.checked_mul(n)!=Some(a.len())||a.iter().any(|z|!z.re.is_finite()||!z.im.is_finite()){
            return Err("Invalid general complex eigensystem input".into());
        }
        #[cfg(feature="static-numerics")]
        let function:ZGeEv=zgeev_;
        #[cfg(not(feature="static-numerics"))]
        let function:ZGeEv=unsafe{
            let address=GetProcAddress(self._library,b"scipy_zgeev_\0".as_ptr().cast());
            if address.is_null(){return Err("Optional complex eigenframe requires LAPACK scipy_zgeev_".into());}
            std::mem::transmute_copy(&address)
        };
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
    /// Solve A X - X B^H = rhs for upper-triangular A and B.
    /// LAPACK's overflow scale is undone, never silently left in the solution.
    pub fn complex_sylvester(&self, a: &[Complex64], b: &[Complex64], rhs: &[Complex64], n: usize) -> Result<Vec<Complex64>, String> {
        self.sylvester_kernel().complex_sylvester(a, b, rhs, n)
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
    pub fn complex_solve_factored(&self, lu: &[Complex64], pivots: &[i32], rhs: &[Complex64], n: usize) -> Result<Vec<Complex64>, String> {
        self.sylvester_kernel().complex_solve_factored(lu, pivots, rhs, n)
    }
    pub fn real_schur(&self,a:&[f64],n:usize)->Result<(Vec<f64>,Vec<f64>),String>{
        if a.len()!=n*n||n==0{return Err("Invalid Schur dimensions".into());}
        let ni=n as i32;let mut t=a.to_vec();let mut q=vec![0.;n*n];let mut wr=vec![0.;n];let mut wi=vec![0.;n];let mut selected=0;let mut info=0;let mut bwork=vec![0;n];let mut query=[0.];let yes=b'V' as c_char;let no=b'N' as c_char;
        unsafe{(self.gees)(&yes,&no,None,&ni,t.as_mut_ptr(),&ni,&mut selected,wr.as_mut_ptr(),wi.as_mut_ptr(),q.as_mut_ptr(),&ni,query.as_mut_ptr(),&-1,bwork.as_mut_ptr(),&mut info);}
        if info!=0{return Err(format!("Schur workspace query {info}"));}
        let lwork=(query[0] as i32).max(3*ni);let mut work=vec![0.;lwork as usize];
        unsafe{(self.gees)(&yes,&no,None,&ni,t.as_mut_ptr(),&ni,&mut selected,wr.as_mut_ptr(),wi.as_mut_ptr(),q.as_mut_ptr(),&ni,work.as_mut_ptr(),&lwork,bwork.as_mut_ptr(),&mut info);}
        if info!=0{return Err(format!("Schur decomposition {info}"));}Ok((t,q))
    }
    /// Padded real Schur matrix, with an isolated leading 1x1 dummy block as
    /// required by DLAQTR's complex system. transpose means conjugate transpose.
    pub fn schur_shifted_solve(&self,t:&[f64],rhs:&[f64],n:usize,omega:f64,transpose:bool)->Result<Vec<f64>,String>{
        let ni=n as i32;let mut b=vec![0.;n];b[0]=omega;let mut x=rhs.to_vec();let mut work=vec![0.;n];let mut scale=1.;let mut info=0;
        if t.len()!=n*n||rhs.len()!=2*n{return Err("Invalid Schur shifted solve".into());}
        unsafe{(self.laqtr)(&(transpose as i32),&0,&ni,t.as_ptr(),&ni,b.as_ptr(),&omega,&mut scale,x.as_mut_ptr(),work.as_mut_ptr(),&mut info);}
        if info!=0||!scale.is_finite()||scale<=0.{return Err(format!("Schur triangular solve info={info}, scale={scale}"));}
        for value in &mut x{*value/=scale;}Ok(x)
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
    /// Real nonsymmetric eigensystem with both left and right columns, using
    /// LAPACK's adjacent Re/Im encoding for each positive-imaginary pair.
    pub fn eigen_both(
        &self,
        a: &[f64],
        n: usize,
    ) -> Result<(Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>), String> {
        if n == 0
            || n > i32::MAX as usize
            || n.checked_mul(n) != Some(a.len())
            || a.iter().any(|x| !x.is_finite())
        {
            return Err("Invalid real eigensystem input".into());
        }
        let ni = n as i32;
        let mut input = a.to_vec();
        let mut wr = vec![0.; n];
        let mut wi = vec![0.; n];
        let mut left = vec![0.; n * n];
        let mut right = vec![0.; n * n];
        let mut query = [0.];
        let mut info = 0;
        let yes = b'V' as c_char;
        unsafe {
            (self.geev)(
                &yes,
                &yes,
                &ni,
                input.as_mut_ptr(),
                &ni,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                left.as_mut_ptr(),
                &ni,
                right.as_mut_ptr(),
                &ni,
                query.as_mut_ptr(),
                &-1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgeev left/right query failed: {info}"));
        }
        let length = (query[0] as i32).max(4 * ni);
        let mut work = vec![0.; length as usize];
        unsafe {
            (self.geev)(
                &yes,
                &yes,
                &ni,
                input.as_mut_ptr(),
                &ni,
                wr.as_mut_ptr(),
                wi.as_mut_ptr(),
                left.as_mut_ptr(),
                &ni,
                right.as_mut_ptr(),
                &ni,
                work.as_mut_ptr(),
                &length,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgeev left/right failed: {info}"));
        }
        Ok((wr, wi, left, right))
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
    pub fn solve_factored(
        &self,
        a: &[f64],
        pivots: &[i32],
        b: &[f64],
        n: usize,
    ) -> Result<Vec<f64>, String> {
        self.solve_factored_op(a,pivots,b,n,false)
    }
    pub fn solve_factored_transpose(&self,a:&[f64],pivots:&[i32],b:&[f64],n:usize)->Result<Vec<f64>,String>{
        self.solve_factored_op(a,pivots,b,n,true)
    }
    fn solve_factored_op(&self,a:&[f64],pivots:&[i32],b:&[f64],n:usize,transpose:bool)->Result<Vec<f64>,String>{
        let ni = n as i32;
        let mut rhs = b.to_vec();
        let mut info = 0;
        let no = if transpose{b'T'}else{b'N'} as c_char;
        unsafe {
            (self.getrs)(
                &no,
                &ni,
                &1,
                a.as_ptr(),
                &ni,
                pivots.as_ptr(),
                rhs.as_mut_ptr(),
                &ni,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgetrs failed: {info}"));
        }
        Ok(rhs)
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
    pub fn condition(&self, a: &[f64], n: usize, norm: f64) -> Result<f64, String> {
        let ni = n as i32;
        let mut reciprocal = 0.;
        let mut work = vec![0.; 4 * n];
        let mut iw = vec![0; n];
        let mut info = 0;
        let one = b'1' as c_char;
        unsafe {
            (self.gecon)(
                &one,
                &ni,
                a.as_ptr(),
                &ni,
                &norm,
                &mut reciprocal,
                work.as_mut_ptr(),
                iw.as_mut_ptr(),
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dgecon failed: {info}"));
        }
        Ok(if reciprocal > 0. {
            1. / reciprocal
        } else {
            f64::INFINITY
        })
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
    /// Allocation-free column-major C = A B for trusted internal dense loops.
    /// Shapes and integer/length overflow are checked in O(1). No input/output
    /// finiteness scan is performed: the caller must validate finite data at
    /// its numerical-operation boundary before exposing a result. Existing C
    /// is overwritten (BLAS beta=0), including when its old entries are NaN.
    pub fn multiply_into(
        &self,a:&[f64],b:&[f64],out:&mut[f64],m:usize,n:usize,k:usize,
    )->Result<(),String>{
        if m>i32::MAX as usize||n>i32::MAX as usize||k>i32::MAX as usize||
            m.checked_mul(k)!=Some(a.len())||k.checked_mul(n)!=Some(b.len())||m.checked_mul(n)!=Some(out.len()){
            return Err("Invalid trusted real BLAS matrix dimensions".into());
        }
        if m==0||n==0{return Ok(());}
        if k==0{out.fill(0.);return Ok(());}
        let no=b'N' as c_char;let(mi,ni,ki)=(m as i32,n as i32,k as i32);
        unsafe{(self.gemm)(&no,&no,&mi,&ni,&ki,&1.,a.as_ptr(),&mi,b.as_ptr(),&ki,&0.,out.as_mut_ptr(),&mi);}
        Ok(())
    }
    /// Column-major C = A^T B, without materializing a transposed A.
    /// A/B have `rows` physical rows and respectively `left_columns` and
    /// `right_columns` columns. Each slice must cover EXACTLY the addressed
    /// span: (columns-1)*leading_dimension + logical_rows, or zero for an
    /// empty matrix. This permits column slices and strided output subblocks.
    /// Padding is untouched. Validation is O(1); the caller owns finite-data
    /// validation. beta=0 overwrites every logical output entry.
    pub fn multiply_transpose_left_into(
        &self, a:&[f64], b:&[f64], out:&mut[f64],
        rows:usize, left_columns:usize, right_columns:usize,
        lda:usize, ldb:usize, ldc:usize,
    )->Result<(),String>{
        let span=|r:usize,c:usize,ld:usize|->Option<usize>{
            if r==0||c==0{Some(0)}else{(c-1).checked_mul(ld)?.checked_add(r)}
        };
        if [rows,left_columns,right_columns,lda,ldb,ldc].iter().any(|&v|v>i32::MAX as usize)
            ||lda<rows.max(1)||ldb<rows.max(1)||ldc<left_columns.max(1)
            ||span(rows,left_columns,lda)!=Some(a.len())
            ||span(rows,right_columns,ldb)!=Some(b.len())
            ||span(left_columns,right_columns,ldc)!=Some(out.len()){
            return Err("Invalid transpose-left real BLAS matrix dimensions".into());
        }
        if left_columns==0||right_columns==0{return Ok(());}
        if rows==0{for j in 0..right_columns{out[j*ldc..j*ldc+left_columns].fill(0.);}return Ok(());}
        let trans=b'T' as c_char;let no=b'N' as c_char;
        let(m,n,k,la,lb,lc)=(left_columns as i32,right_columns as i32,rows as i32,lda as i32,ldb as i32,ldc as i32);
        unsafe{(self.gemm)(&trans,&no,&m,&n,&k,&1.,a.as_ptr(),&la,b.as_ptr(),&lb,&0.,out.as_mut_ptr(),&lc);}
        Ok(())
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
    pub fn multiply_complex(
        &self,
        a: &[Complex64],
        b: &[Complex64],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<Vec<Complex64>, String> {
        self.sylvester_kernel().multiply_complex(a, b, m, n, k)
    }
    /// Allocation-free complex C = A B, column-major and BLAS beta=0.
    /// This validates only dimensions/overflow, not floating-point contents.
    /// Callers using reusable scratch buffers must check finiteness at their
    /// numerical-operation boundary. No allocation or full-array scan occurs.
    pub fn multiply_complex_into(
        &self,a:&[Complex64],b:&[Complex64],out:&mut[Complex64],m:usize,n:usize,k:usize,
    )->Result<(),String>{
        if m>i32::MAX as usize||n>i32::MAX as usize||k>i32::MAX as usize||
            m.checked_mul(k)!=Some(a.len())||k.checked_mul(n)!=Some(b.len())||m.checked_mul(n)!=Some(out.len()){
            return Err("Invalid trusted complex BLAS matrix dimensions".into());
        }
        if m==0||n==0{return Ok(());}
        if k==0{out.fill(Complex64::default());return Ok(());}
        let no=b'N' as c_char;let(mi,ni,ki)=(m as i32,n as i32,k as i32);
        let one=Complex64{re:1.,im:0.};let zero=Complex64::default();
        unsafe{(self.zgemm)(&no,&no,&mi,&ni,&ki,&one,a.as_ptr(),&mi,b.as_ptr(),&ki,&zero,out.as_mut_ptr(),&mi);}
        Ok(())
    }
    pub fn symmetric_eigenvalues(&self, a: &[f64], n: usize) -> Result<Vec<f64>, String> {
        let ni = n as i32;
        let mut matrix = a.to_vec();
        let mut vals = vec![0.; n];
        let mut query = [0.];
        let mut info = 0;
        let no = b'N' as c_char;
        let upper = b'U' as c_char;
        unsafe {
            (self.syev)(
                &no,
                &upper,
                &ni,
                matrix.as_mut_ptr(),
                &ni,
                vals.as_mut_ptr(),
                query.as_mut_ptr(),
                &-1,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dsyev workspace query failed: {info}"));
        }
        let len = (query[0] as i32).max(3 * ni);
        let mut work = vec![0.; len as usize];
        unsafe {
            (self.syev)(
                &no,
                &upper,
                &ni,
                matrix.as_mut_ptr(),
                &ni,
                vals.as_mut_ptr(),
                work.as_mut_ptr(),
                &len,
                &mut info,
            );
        }
        if info != 0 {
            return Err(format!("LAPACK dsyev failed: {info}"));
        }
        Ok(vals)
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
    fn left_right_real_eigenvector_convention_is_verified() {
        let lap = Lapack::load().unwrap();
        let a = [-1., -3., 2., -4.];
        let (wr, wi, left, right) = lap.eigen_both(&a, 2).unwrap();
        assert!(wi[0] > 0.);
        for row in 0..2 {
            let ar = a[row] * right[0] + a[row + 2] * right[1];
            let ai = a[row] * right[2] + a[row + 2] * right[3];
            assert!((ar - wr[0] * right[row] + wi[0] * right[row + 2]).abs() < 1e-13);
            assert!((ai - wr[0] * right[row + 2] - wi[0] * right[row]).abs() < 1e-13);
            let ar = a[row * 2] * left[0] + a[1 + row * 2] * left[1];
            let ai = a[row * 2] * left[2] + a[1 + row * 2] * left[3];
            assert!((ar - wr[0] * left[row] - wi[0] * left[row + 2]).abs() < 1e-13);
            assert!((ai - wr[0] * left[row + 2] + wi[0] * left[row]).abs() < 1e-13);
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
        let mut reused=[Complex64{re:f64::NAN,im:f64::NAN};2];
        lap.multiply_complex_into(&a,&b,&mut reused,2,1,1).unwrap();
        for(actual,expected)in reused.iter().zip(product){assert_eq!(actual.re,expected.re);assert_eq!(actual.im,expected.im);}
    }
    #[test]
    fn trusted_gemm_into_reuses_buffers_and_checks_rectangular_zero_and_invalid_shapes(){
        let lap=Lapack::load().unwrap();
        let a=[1.,2.,3.,4.,5.,6.];let b=[7.,8.,9.,10.,11.,12.];let mut reused=[f64::NAN;4];
        lap.multiply_into(&a,&b,&mut reused,2,2,3).unwrap();
        assert_eq!(reused,[76.,100.,103.,136.]);assert_eq!(reused.to_vec(),lap.multiply(&a,&b,2,2,3));
        lap.multiply_into(&[],&[],&mut reused,2,2,0).unwrap();assert_eq!(reused,[0.;4]);
        let mut complex=[Complex64{re:f64::NAN,im:f64::NAN};4];
        lap.multiply_complex_into(&[],&[],&mut complex,2,2,0).unwrap();
        assert!(complex.iter().all(|z|z.re==0.&&z.im==0.));
        lap.multiply_into(&[],&[],&mut[],0,3,0).unwrap();
        lap.multiply_complex_into(&[],&[],&mut[],0,3,0).unwrap();
        assert!(lap.multiply_into(&a,&b,&mut reused,2,2,2).is_err());
        assert!(lap.multiply_complex_into(&[],&[],&mut complex,2,2,1).is_err());
        assert!(lap.multiply_into(&[],&[],&mut[],usize::MAX,2,0).is_err());
        assert!(lap.multiply_complex_into(&[],&[],&mut[],usize::MAX,2,0).is_err());
    }
}
