//! Python-free SuperLU ILUTP PRECONDITIONER for an already row-equilibrated
//! real sparse operator. This adapter performs no further scaling, truncation,
//! or iterative refinement. The caller retains its original operator and must
//! validate every accepted linear solve with that operator's residual.
use std::{cell::Cell,ffi::{c_char,c_void},path::{Path,PathBuf},ptr::NonNull};

const ABI_VERSION:i32=1;
const ERROR_CAPACITY:usize=2048;
const PIVOT_THRESHOLD:f64=0.1; // Same as the measured SciPy spilu pilot.
type Create=unsafe extern "C" fn(i32,i32,*const i32,*const i32,*const f64,f64,f64,f64,*mut *mut c_void,*mut c_char,usize)->i32;
type Solve=unsafe extern "C" fn(*mut c_void,i32,i32,*mut f64,i32,*mut c_char,usize)->i32;
type Destroy=unsafe extern "C" fn(*mut c_void);
type AbiVersion=unsafe extern "C" fn()->i32;

#[cfg(windows)]
#[link(name="kernel32")]
extern "system"{
    fn LoadLibraryW(name:*const u16)->*mut c_void;
    fn FreeLibrary(module:*mut c_void)->i32;
    fn GetProcAddress(module:*mut c_void,name:*const c_char)->*mut c_void;
    fn GetLastError()->u32;
}

/// Owning module handle. It is deliberately neither Send nor Sync; in
/// particular no Rust caller can use one factor concurrently across threads.
struct Library{
    handle:NonNull<c_void>,
    _single_thread:std::marker::PhantomData<std::rc::Rc<()>>,
}
impl Library{
    #[cfg(windows)]
    fn load(path:&Path)->Result<Self,String>{
        use std::os::windows::ffi::OsStrExt;
        let mut wide:Vec<u16>=path.as_os_str().encode_wide().collect();
        if wide.contains(&0){return Err("SuperLU DLL path contains a NUL character".into());}
        wide.push(0);
        let handle=unsafe{LoadLibraryW(wide.as_ptr())};
        let handle=NonNull::new(handle).ok_or_else(||format!("Cannot load SuperLU DLL {} (Windows error {}). Set ATOM_SUPERLU_DLL to the native bridge DLL.",path.display(),unsafe{GetLastError()}))?;
        Ok(Self{handle,_single_thread:std::marker::PhantomData})
    }
    #[cfg(not(windows))]
    fn load(_path:&Path)->Result<Self,String>{Err("The native SuperLU DLL adapter currently requires Windows".into())}

    /// Only fixed C function-pointer types from the versioned bridge ABI are
    /// requested below. The library owner outlives every returned entry point.
    #[cfg(windows)]
    unsafe fn symbol<T:Copy>(&self,name:&'static [u8])->Result<T,String>{
        if name.last()!=Some(&0)||name[..name.len()-1].contains(&0)||std::mem::size_of::<T>()!=std::mem::size_of::<*mut c_void>(){
            return Err("Invalid native SuperLU entry-point declaration".into());
        }
        let address=GetProcAddress(self.handle.as_ptr(),name.as_ptr().cast());
        if address.is_null(){return Err(format!("Missing native SuperLU symbol {}",String::from_utf8_lossy(&name[..name.len()-1])));}
        Ok(std::mem::transmute_copy(&address))
    }
    #[cfg(not(windows))]
    unsafe fn symbol<T:Copy>(&self,_name:&'static [u8])->Result<T,String>{Err("Native SuperLU symbols are unavailable on this platform".into())}
}
impl Drop for Library{
    fn drop(&mut self){
        #[cfg(windows)]
        unsafe{FreeLibrary(self.handle.as_ptr());}
    }
}
#[cfg(feature="static-numerics")]
extern "C"{
    fn atom_superlu_factor(n:i32,nnz:i32,colptr:*const i32,rowind:*const i32,values:*const f64,drop:f64,fill:f64,pivot:f64,out:*mut *mut c_void,error:*mut c_char,capacity:usize)->i32;
    fn atom_superlu_solve(handle:*mut c_void,transpose:i32,nrhs:i32,b:*mut f64,ld:i32,error:*mut c_char,capacity:usize)->i32;
    fn atom_superlu_destroy(handle:*mut c_void);
    fn atom_superlu_abi_version()->i32;
}
struct Api{_library:Option<Library>,create:Create,solve:Solve,destroy:Destroy}
impl Api{
    #[cfg(feature="static-numerics")]
    fn load(_path:&Path)->Result<Self,String>{
        let actual=unsafe{atom_superlu_abi_version()};
        if actual!=ABI_VERSION{return Err(format!("Static SuperLU ABI mismatch: expected {ABI_VERSION}, received {actual}"));}
        Ok(Self{_library:None,create:atom_superlu_factor,solve:atom_superlu_solve,destroy:atom_superlu_destroy})
    }
    #[cfg(not(feature="static-numerics"))]
    fn load(path:&Path)->Result<Self,String>{
        let library=Library::load(path)?;
        unsafe{
            let version:AbiVersion=library.symbol(b"atom_superlu_abi_version\0")?;
            let actual=version();
            if actual!=ABI_VERSION{return Err(format!("Native SuperLU ABI mismatch: expected {ABI_VERSION}, received {actual}"));}
            let create=library.symbol(b"atom_superlu_factor\0")?;
            let solve=library.symbol(b"atom_superlu_solve\0")?;
            let destroy=library.symbol(b"atom_superlu_destroy\0")?;
            Ok(Self{_library:Some(library),create,solve,destroy})
        }
    }
}
struct Csc{n:i32,nnz:i32,pointers:Vec<i32>,rows:Vec<i32>,values:Vec<f64>}
fn reserve<T>(size:usize)->Result<Vec<T>,String>{
    let mut result=Vec::new();result.try_reserve_exact(size).map_err(|e|format!("Native SuperLU buffer allocation failed: {e}"))?;Ok(result)
}
fn csc(matrix:&[Vec<(usize,f64)>])->Result<Csc,String>{
    let n=i32::try_from(matrix.len()).map_err(|_|"Native SuperLU dimension exceeds the int32 ABI")?;
    if n==0{return Err("Native SuperLU requires a nonempty square matrix".into());}
    let count=matrix.iter().try_fold(0usize,|total,column|total.checked_add(column.len()).ok_or("Native SuperLU nonzero count overflow"))?;
    let nnz=i32::try_from(count).map_err(|_|"Native SuperLU nonzero count exceeds the int32 ABI")?;
    if nnz==0{return Err("Native SuperLU requires a nonzero sparse matrix".into());}
    let pointer_count=matrix.len().checked_add(1).ok_or("Native SuperLU column-pointer count overflow")?;
    let mut pointers=reserve(pointer_count)?;let mut rows=reserve(count)?;let mut values=reserve(count)?;
    pointers.push(0);
    for(column,entries)in matrix.iter().enumerate(){
        let mut previous=None;
        for &(row,value)in entries{
            if row>=matrix.len()||previous.is_some_and(|last|row<=last){return Err(format!("Native SuperLU column {column} must have in-range, sorted, unique row indices"));}
            if !value.is_finite(){return Err(format!("Native SuperLU column {column} contains a non-finite value"));}
            // row < n <= i32::MAX, and every prefix count <= validated nnz.
            rows.push(row as i32);values.push(value);previous=Some(row);
        }
        pointers.push(i32::try_from(rows.len()).map_err(|_|"Native SuperLU column pointer exceeds int32")?);
    }
    Ok(Csc{n,nnz,pointers,rows,values})
}
fn bridge_error(operation:&str,status:i32,buffer:&[u8])->String{
    let length=buffer.iter().position(|&v|v==0).unwrap_or(buffer.len());
    let detail=String::from_utf8_lossy(&buffer[..length]);
    if detail.is_empty(){format!("Native SuperLU {operation} failed with status {status}")}else{format!("Native SuperLU {operation} failed with status {status}: {detail}")}
}

pub struct Factor{
    api:Api,handle:NonNull<c_void>,n:usize,valid:Cell<bool>,
}
impl Factor{
    /// Input is the caller's ALREADY row-equilibrated real CSC matrix. All
    /// supplied entries are passed unchanged; this wrapper neither rebalances
    /// the matrix nor scales RHS/solutions a second time.
    pub fn new(matrix:&[Vec<(usize,f64)>],drop_tolerance:f64,fill_factor:f64)->Result<Self,String>{
        if !drop_tolerance.is_finite()||drop_tolerance<0.||!fill_factor.is_finite()||fill_factor<1.{return Err("Invalid native SuperLU drop tolerance or fill factor".into());}
        let sparse=csc(matrix)?;
        #[cfg(feature="static-numerics")]
        let path=PathBuf::from("static-linked-superlu");
        #[cfg(not(feature="static-numerics"))]
        let path=match std::env::var_os("ATOM_SUPERLU_DLL"){
            Some(path)if !path.is_empty()=>PathBuf::from(path),
            Some(_)=>return Err("ATOM_SUPERLU_DLL must not be empty".into()),
            None=>std::env::current_exe().map_err(|e|format!("Cannot locate native SuperLU beside executable: {e}"))?.with_file_name("atom_superlu.dll"),
        };
        let api=Api::load(&path)?;
        let mut handle=std::ptr::null_mut();let mut error=[0u8;ERROR_CAPACITY];
        let status=unsafe{(api.create)(sparse.n,sparse.nnz,sparse.pointers.as_ptr(),sparse.rows.as_ptr(),sparse.values.as_ptr(),drop_tolerance,fill_factor,PIVOT_THRESHOLD,&mut handle,error.as_mut_ptr().cast(),error.len())};
        if status!=0{
            // ABI promises NULL on failure; release a defensive non-NULL
            // result too, while the owning DLL is still loaded.
            if !handle.is_null(){unsafe{(api.destroy)(handle);}}
            return Err(bridge_error("factorization",status,&error));
        }
        let handle=NonNull::new(handle).ok_or("Native SuperLU returned success with a null factor")?;
        Ok(Self{api,handle,n:matrix.len(),valid:Cell::new(true)})
    }
    /// Apply the ILUTP factors or their exact transpose. This is only a
    /// preconditioner application; accuracy is decided by the outer solver's
    /// original-matrix residual, never by this factor's approximate matrix.
    pub fn solve(&self,rhs:&[f64],transpose:bool)->Result<Vec<f64>,String>{
        if !self.valid.get(){return Err("Native SuperLU factor was invalidated by an earlier native solve failure".into());}
        if rhs.len()!=self.n||rhs.iter().any(|v|!v.is_finite()){return Err("Invalid native SuperLU RHS dimension or value".into());}
        let mut result=reserve(self.n)?;result.extend_from_slice(rhs);
        let mut error=[0u8;ERROR_CAPACITY];
        let status=unsafe{(self.api.solve)(self.handle.as_ptr(),i32::from(transpose),1,result.as_mut_ptr(),self.n as i32,error.as_mut_ptr().cast(),error.len())};
        if status!=0||result.iter().any(|v|!v.is_finite()){
            self.valid.set(false);
            return Err(if status!=0{bridge_error("transpose/forward triangular solve",status,&error)}else{"Native SuperLU solve produced a non-finite result".into()});
        }
        Ok(result)
    }
}
impl Drop for Factor{
    fn drop(&mut self){
        // Rust drops fields AFTER this body, so api._library remains loaded
        // for destruction, including factors invalidated by a native error.
        unsafe{(self.api.destroy)(self.handle.as_ptr());}
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    #[test]
    #[ignore="Requires the freshly built native SuperLU bridge and OpenBLAS"]
    fn native_adapter_forward_and_transpose_match_dense_oracle(){
        let lap=crate::Lapack::load().unwrap();
        let dense=vec![5.,0.1,-0.3, 1.3,4.,0.4, -0.2,0.7,3.];
        let matrix:Vec<Vec<_>>=(0..3).map(|j|(0..3).map(|i|(i,dense[i+j*3])).collect()).collect();
        for _ in 0..2{
            let factor=Factor::new(&matrix,1e-12,4.).unwrap();
            for rhs in [[1.,-2.,0.3],[0.1,3.,-1.]]{for transposed in [false,true]{
                let actual=factor.solve(&rhs,transposed).unwrap();
                let operator=if transposed{(0..9).map(|i|dense[i/3+(i%3)*3]).collect()}else{dense.clone()};
                let expected=lap.solve(&operator,&rhs,3).unwrap();
                assert!(actual.iter().zip(expected).all(|(a,b)|(a-b).abs()<2e-12));
            }}
        }
    }
    #[test]
    fn csc_encoding_preserves_every_balanced_entry(){
        let matrix=vec![vec![(0,1.),(2,-0.125)],vec![(1,0.5)],vec![(0,0.25),(2,-2.)]];
        let data=csc(&matrix).unwrap();assert_eq!((data.n,data.nnz),(3,5));
        assert_eq!(data.pointers,[0,2,3,5]);assert_eq!(data.rows,[0,2,1,0,2]);assert_eq!(data.values,[1.,-0.125,0.5,0.25,-2.]);
    }
    #[test]
    fn malformed_csc_and_parameters_fail_before_loading_a_dll(){
        assert!(csc(&[]).is_err());assert!(csc(&[vec![]]).is_err());
        for bad in [vec![vec![(1,1.)]],vec![vec![(0,1.),(0,2.)]],vec![vec![(0,f64::NAN)]],vec![vec![(1,1.),(0,2.)],vec![(1,1.)]]]{assert!(csc(&bad).is_err());}
        let valid=vec![vec![(0,1.)]];
        assert!(Factor::new(&valid,-1.,4.).is_err());assert!(Factor::new(&valid,1e-3,0.5).is_err());
        assert!(Factor::new(&valid,f64::NAN,4.).is_err());
    }
    #[test]
    fn native_error_buffer_is_bounded_even_without_terminator(){
        assert_eq!(bridge_error("solve",-3,b"abc"),"Native SuperLU solve failed with status -3: abc");
        assert_eq!(bridge_error("solve",7,b"bad\0ignored"),"Native SuperLU solve failed with status 7: bad");
    }
}
