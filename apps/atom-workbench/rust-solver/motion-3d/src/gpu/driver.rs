//! The CUDA driver API, loaded at run time from `nvcuda.dll` (the NVIDIA display driver): the EXE
//! needs no CUDA toolkit. Only the entry points the engine uses are bound; every call's result is
//! checked and reported with the driver's error name.
//!
//! The primary context of the chosen device is retained with blocking synchronization, so a
//! thread waiting for the GPU sleeps instead of spinning a core. `ATOM_HIDE_NVCUDA` (any value)
//! hides the driver, as on a machine without one.
use std::ffi::{c_char,c_void,CStr};
use std::sync::OnceLock;

type CuResult=i32;
pub type DevicePtr=u64;
type Handle=*mut c_void;

#[link(name="kernel32")]
extern "system"{
    fn LoadLibraryW(name:*const u16)->Handle;
    fn GetProcAddress(module:Handle,name:*const c_char)->Handle;
}

macro_rules! api{
    ($($name:ident:fn($($arg:ty),*);)*)=>{
        #[allow(non_snake_case)]
        struct Api{$($name:unsafe extern "system" fn($($arg),*)->CuResult,)*}
        impl Api{
            /// Every entry point of `nvcuda.dll`, or the first one it lacks.
            unsafe fn load(library:Handle)->Result<Api,String>{
                Ok(Api{$($name:{
                    let symbol=GetProcAddress(library,concat!(stringify!($name),"\0").as_ptr() as *const c_char);
                    if symbol.is_null(){return Err(format!("NVIDIA 驱动太旧（nvcuda.dll 缺少 {}）",stringify!($name)));}
                    std::mem::transmute::<Handle,unsafe extern "system" fn($($arg),*)->CuResult>(symbol)
                },)*})
            }
        }
    };
}
api!{
    cuInit:fn(u32);
    cuDriverGetVersion:fn(*mut i32);
    cuGetErrorName:fn(CuResult,*mut *const c_char);
    cuDeviceGetCount:fn(*mut i32);
    cuDeviceGet:fn(*mut i32,i32);
    cuDeviceGetName:fn(*mut c_char,i32,i32);
    cuDeviceGetAttribute:fn(*mut i32,i32,i32);
    cuDeviceTotalMem_v2:fn(*mut usize,i32);
    cuDevicePrimaryCtxSetFlags_v2:fn(i32,u32);
    cuDevicePrimaryCtxRetain:fn(*mut Handle,i32);
    cuDevicePrimaryCtxRelease_v2:fn(i32);
    cuCtxSetCurrent:fn(Handle);
    cuMemGetInfo_v2:fn(*mut usize,*mut usize);
    cuModuleLoadDataEx:fn(*mut Handle,*const c_void,u32,*mut i32,*mut *mut c_void);
    cuModuleUnload:fn(Handle);
    cuModuleGetFunction:fn(*mut Handle,Handle,*const c_char);
    cuModuleGetGlobal_v2:fn(*mut DevicePtr,*mut usize,Handle,*const c_char);
    cuFuncSetAttribute:fn(Handle,i32,i32);
    cuFuncGetAttribute:fn(*mut i32,i32,Handle);
    cuOccupancyMaxActiveBlocksPerMultiprocessor:fn(*mut i32,Handle,i32,usize);
    cuMemAlloc_v2:fn(*mut DevicePtr,usize);
    cuMemFree_v2:fn(DevicePtr);
    cuMemAllocHost_v2:fn(*mut *mut c_void,usize);
    cuMemFreeHost:fn(*mut c_void);
    cuMemcpyHtoDAsync_v2:fn(DevicePtr,*const c_void,usize,Handle);
    cuMemcpyDtoHAsync_v2:fn(*mut c_void,DevicePtr,usize,Handle);
    cuMemsetD8Async:fn(DevicePtr,u8,usize,Handle);
    cuLaunchKernel:fn(Handle,u32,u32,u32,u32,u32,u32,u32,Handle,*mut *mut c_void,*mut *mut c_void);
    cuStreamCreate:fn(*mut Handle,u32);
    cuStreamDestroy_v2:fn(Handle);
    cuStreamSynchronize:fn(Handle);
    cuEventCreate:fn(*mut Handle,u32);
    cuEventRecord:fn(Handle,Handle);
    cuEventSynchronize:fn(Handle);
    cuEventElapsedTime:fn(*mut f32,Handle,Handle);
    cuEventDestroy_v2:fn(Handle);
}

/// The loaded driver: `nvcuda.dll` bound and initialized, once per process.
struct Driver{api:Api,version:i32}
unsafe impl Send for Driver{}
unsafe impl Sync for Driver{}

/// Why the GPU engine cannot run on a machine without the NVIDIA driver.
pub const NO_DRIVER:&str="找不到 NVIDIA 驱动（nvcuda.dll）";

fn driver()->Result<&'static Driver,String>{
    static DRIVER:OnceLock<Result<Driver,String>>=OnceLock::new();
    DRIVER.get_or_init(||unsafe{
        if std::env::var_os("ATOM_HIDE_NVCUDA").is_some(){return Err(NO_DRIVER.into());}
        let name:Vec<u16>="nvcuda.dll\0".encode_utf16().collect();
        let library=LoadLibraryW(name.as_ptr());
        if library.is_null(){return Err(NO_DRIVER.into());}
        let api=Api::load(library)?;
        let result=(api.cuInit)(0);
        if result!=0{return Err(format!("CUDA 初始化失败（{}）：没有可用的 NVIDIA 显卡",error_name(&api,result)));}
        let mut version=0;
        if (api.cuDriverGetVersion)(&mut version)!=0{return Err("cuDriverGetVersion failed".into());}
        Ok(Driver{api,version})
    }).as_ref().map_err(|e|e.clone())
}

fn error_name(api:&Api,result:CuResult)->String{
    let mut name:*const c_char=std::ptr::null();
    unsafe{if (api.cuGetErrorName)(result,&mut name)==0&&!name.is_null(){return CStr::from_ptr(name).to_string_lossy().into_owned();}}
    format!("CUDA error {result}")
}
/// A driver call's result as Ok or the call named with the driver's error.
fn check(what:&str,result:CuResult)->Result<(),String>{
    if result==0{Ok(())}else{Err(format!("{what} failed: {}",driver().map(|d|error_name(&d.api,result)).unwrap_or_else(|_|result.to_string())))}
}
macro_rules! call{($what:expr,$name:ident($($arg:expr),*))=>{check($what,unsafe{(driver()?.api.$name)($($arg),*)})};}

/// The driver's CUDA version, e.g. 13020 for 13.2.
pub fn driver_version()->Result<i32,String>{Ok(driver()?.version)}

/// A device and the properties the engine plans with.
#[derive(Clone,Debug)]
pub struct DeviceInfo{
    pub ordinal:i32,
    pub name:String,
    pub compute_capability:(i32,i32),
    pub multiprocessors:i32,
    pub total_memory:usize,
    /// Whether the display watchdog bounds every kernel's run time (WDDM).
    pub watchdog:bool,
}

/// Every CUDA device the driver lists.
pub fn devices()->Result<Vec<DeviceInfo>,String>{
    let mut count=0;call!("cuDeviceGetCount",cuDeviceGetCount(&mut count))?;
    (0..count).map(|ordinal|{
        let mut device=0;call!("cuDeviceGet",cuDeviceGet(&mut device,ordinal))?;
        let mut name=[0 as c_char;256];call!("cuDeviceGetName",cuDeviceGetName(name.as_mut_ptr(),256,device))?;
        let attribute=|a:i32|->Result<i32,String>{let mut v=0;call!("cuDeviceGetAttribute",cuDeviceGetAttribute(&mut v,a,device))?;Ok(v)};
        let mut total=0;call!("cuDeviceTotalMem",cuDeviceTotalMem_v2(&mut total,device))?;
        // CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR/MINOR, MULTIPROCESSOR_COUNT, KERNEL_EXEC_TIMEOUT.
        Ok(DeviceInfo{ordinal,name:unsafe{CStr::from_ptr(name.as_ptr())}.to_string_lossy().into_owned(),
            compute_capability:(attribute(75)?,attribute(76)?),multiprocessors:attribute(16)?,total_memory:total,watchdog:attribute(17)?!=0})
    }).collect()
}

/// The primary context of a device; released on drop.
pub struct Context{device:i32,context:Handle,pub info:DeviceInfo}
unsafe impl Send for Context{}
unsafe impl Sync for Context{}
impl Context{
    pub fn new(info:&DeviceInfo)->Result<Context,String>{
        let mut device=0;call!("cuDeviceGet",cuDeviceGet(&mut device,info.ordinal))?;
        // CU_CTX_SCHED_BLOCKING_SYNC: a synchronizing host thread sleeps.
        call!("cuDevicePrimaryCtxSetFlags",cuDevicePrimaryCtxSetFlags_v2(device,4))?;
        let mut context=std::ptr::null_mut();call!("cuDevicePrimaryCtxRetain",cuDevicePrimaryCtxRetain(&mut context,device))?;
        let out=Context{device,context,info:info.clone()};out.make_current()?;Ok(out)
    }
    /// Makes the context current on the calling thread (every thread that uses it first).
    pub fn make_current(&self)->Result<(),String>{call!("cuCtxSetCurrent",cuCtxSetCurrent(self.context))}
    /// Free and total device memory (bytes).
    pub fn memory(&self)->Result<(usize,usize),String>{let(mut free,mut total)=(0,0);call!("cuMemGetInfo",cuMemGetInfo_v2(&mut free,&mut total))?;Ok((free,total))}
    /// A PTX module, compiled for this device by the driver's JIT (cached by the driver).
    pub fn load(&self,ptx:&str)->Result<Module,String>{
        let mut image=ptx.as_bytes().to_vec();image.push(0);
        let mut log=vec![0u8;16384];
        // CU_JIT_ERROR_LOG_BUFFER and its size.
        let mut options=[5i32,6];let mut values=[log.as_mut_ptr() as *mut c_void,log.len() as *mut c_void];
        let mut module=std::ptr::null_mut();
        let result=unsafe{(driver()?.api.cuModuleLoadDataEx)(&mut module,image.as_ptr() as *const c_void,2,options.as_mut_ptr(),values.as_mut_ptr())};
        if result!=0{
            let end=log.iter().position(|&b|b==0).unwrap_or(log.len());
            return Err(format!("{}: {}",error_name(&driver()?.api,result),String::from_utf8_lossy(&log[..end]).trim()));
        }
        Ok(Module{module})
    }
}
impl Drop for Context{fn drop(&mut self){if let Ok(d)=driver(){unsafe{(d.api.cuDevicePrimaryCtxRelease_v2)(self.device);}}}}

pub struct Module{module:Handle}
unsafe impl Send for Module{}
unsafe impl Sync for Module{}
impl Module{
    pub fn function(&self,name:&str)->Result<Function,String>{
        let c=format!("{name}\0");let mut f=std::ptr::null_mut();
        check(&format!("cuModuleGetFunction({name})"),unsafe{(driver()?.api.cuModuleGetFunction)(&mut f,self.module,c.as_ptr() as *const c_char)})?;
        Ok(Function{function:f,name:name.to_string()})
    }
    /// A module variable's device address.
    pub fn global(&self,name:&str)->Result<DevicePtr,String>{
        let c=format!("{name}\0");let(mut pointer,mut bytes)=(0,0);
        check(&format!("cuModuleGetGlobal({name})"),unsafe{(driver()?.api.cuModuleGetGlobal_v2)(&mut pointer,&mut bytes,self.module,c.as_ptr() as *const c_char)})?;Ok(pointer)
    }
}
impl Drop for Module{fn drop(&mut self){if let Ok(d)=driver(){unsafe{(d.api.cuModuleUnload)(self.module);}}}}

#[derive(Clone)]
pub struct Function{function:Handle,pub name:String}
unsafe impl Send for Function{}
unsafe impl Sync for Function{}
impl Function{
    /// Prefers the largest shared memory carveout (CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT).
    pub fn prefer_shared(&self)->Result<(),String>{call!("cuFuncSetAttribute",cuFuncSetAttribute(self.function,9,100))}
    /// Registers per thread and local (spill) bytes per thread.
    pub fn resources(&self)->Result<(i32,i32),String>{
        let(mut registers,mut local)=(0,0);
        call!("cuFuncGetAttribute",cuFuncGetAttribute(&mut registers,4,self.function))?;call!("cuFuncGetAttribute",cuFuncGetAttribute(&mut local,3,self.function))?;
        Ok((registers,local))
    }
    /// The blocks of `threads` threads one multiprocessor holds at once.
    pub fn occupancy(&self,threads:i32)->Result<i32,String>{
        let mut blocks=0;call!("cuOccupancyMaxActiveBlocksPerMultiprocessor",cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut blocks,self.function,threads,0))?;Ok(blocks)
    }
}

/// A kernel argument: a value whose address the launch passes.
pub trait Arg{fn pointer(&self)->*mut c_void;}
macro_rules! arg{($($t:ty),*)=>{$(impl Arg for $t{fn pointer(&self)->*mut c_void{self as *const $t as *mut c_void}})*};}
arg!(u32,i32,u64,f32,f64,super::program::Run);

/// A stream: launches, copies and events in order.
pub struct Stream{stream:Handle}
unsafe impl Send for Stream{}
unsafe impl Sync for Stream{}
impl Stream{
    pub fn new()->Result<Stream,String>{let mut s=std::ptr::null_mut();call!("cuStreamCreate",cuStreamCreate(&mut s,1))?;Ok(Stream{stream:s})}
    pub fn launch(&self,f:&Function,blocks:u32,threads:u32,args:&[&dyn Arg])->Result<(),String>{
        let mut pointers:Vec<*mut c_void>=args.iter().map(|a|a.pointer()).collect();
        check(&format!("launch {}",f.name),unsafe{(driver()?.api.cuLaunchKernel)(f.function,blocks,1,1,threads,1,1,0,self.stream,pointers.as_mut_ptr(),std::ptr::null_mut())})
    }
    /// Copies a value to a device address (from pageable memory: done on return).
    pub fn put<T:Copy>(&self,to:DevicePtr,data:&T)->Result<(),String>{
        call!("cuMemcpyHtoDAsync",cuMemcpyHtoDAsync_v2(to,data as *const T as *const c_void,std::mem::size_of::<T>(),self.stream))?;
        self.synchronize()
    }
    /// Copies `data` into the buffer at byte offset `at` (from pageable memory: done on return).
    pub fn upload_at<T:Copy>(&self,to:&Buffer,at:usize,data:&[T])->Result<(),String>{
        let bytes=std::mem::size_of_val(data);assert!(at+bytes<=to.bytes,"upload beyond the buffer");
        if bytes==0{return Ok(());}
        call!("cuMemcpyHtoDAsync",cuMemcpyHtoDAsync_v2(to.pointer+at as u64,data.as_ptr() as *const c_void,bytes,self.stream))?;
        self.synchronize()
    }
    pub fn upload<T:Copy>(&self,to:&Buffer,data:&[T])->Result<(),String>{self.upload_at(to,0,data)}
    pub fn download<T:Copy>(&self,from:&Buffer,data:&mut [T])->Result<(),String>{
        let bytes=std::mem::size_of_val(data);assert!(bytes<=from.bytes,"download beyond the buffer");
        if bytes==0{return Ok(());}
        call!("cuMemcpyDtoHAsync",cuMemcpyDtoHAsync_v2(data.as_mut_ptr() as *mut c_void,from.pointer,bytes,self.stream))?;
        self.synchronize()
    }
    pub fn zero(&self,buffer:&Buffer)->Result<(),String>{call!("cuMemsetD8Async",cuMemsetD8Async(buffer.pointer,0,buffer.bytes.max(1),self.stream))}
    pub fn synchronize(&self)->Result<(),String>{call!("cuStreamSynchronize",cuStreamSynchronize(self.stream))}
    pub fn record(&self,event:&Event)->Result<(),String>{call!("cuEventRecord",cuEventRecord(event.event,self.stream))}
}
impl Drop for Stream{fn drop(&mut self){if let Ok(d)=driver(){unsafe{(d.api.cuStreamDestroy_v2)(self.stream);}}}}

/// A timing event.
pub struct Event{event:Handle}
impl Event{
    /// CU_EVENT_BLOCKING_SYNC: waiting on it sleeps.
    pub fn new()->Result<Event,String>{let mut e=std::ptr::null_mut();call!("cuEventCreate",cuEventCreate(&mut e,1))?;Ok(Event{event:e})}
    pub fn synchronize(&self)->Result<(),String>{call!("cuEventSynchronize",cuEventSynchronize(self.event))}
    /// Milliseconds from `start` to this event.
    pub fn since(&self,start:&Event)->Result<f64,String>{let mut ms=0f32;call!("cuEventElapsedTime",cuEventElapsedTime(&mut ms,start.event,self.event))?;Ok(ms as f64)}
}
impl Drop for Event{fn drop(&mut self){if let Ok(d)=driver(){unsafe{(d.api.cuEventDestroy_v2)(self.event);}}}}

/// Device memory; freed on drop.
pub struct Buffer{pub pointer:DevicePtr,pub bytes:usize}
impl Buffer{
    pub fn new(bytes:usize)->Result<Buffer,String>{
        let mut pointer=0;check(&format!("cuMemAlloc({bytes} bytes)"),unsafe{(driver()?.api.cuMemAlloc_v2)(&mut pointer,bytes.max(1))})?;Ok(Buffer{pointer,bytes})
    }
    /// A buffer holding `data`.
    pub fn with<T:Copy>(stream:&Stream,data:&[T])->Result<Buffer,String>{let b=Buffer::new(std::mem::size_of_val(data))?;stream.upload(&b,data)?;Ok(b)}
}
impl Drop for Buffer{fn drop(&mut self){if let Ok(d)=driver(){unsafe{(d.api.cuMemFree_v2)(self.pointer);}}}}

/// Words in host memory the device writes (a launch's report, read without a copy once an event says it is done).
pub struct HostWords{pointer:*mut u64,len:usize}
impl HostWords{
    pub fn new(data:&[u64])->Result<HostWords,String>{
        let mut pointer=std::ptr::null_mut();check(&format!("cuMemAllocHost({} words)",data.len()),unsafe{(driver()?.api.cuMemAllocHost_v2)(&mut pointer,8*data.len().max(1))})?;
        let out=HostWords{pointer:pointer as *mut u64,len:data.len()};unsafe{std::ptr::copy_nonoverlapping(data.as_ptr(),out.pointer,data.len());}Ok(out)
    }
    /// The device's address of the words.
    pub fn address(&self)->u64{self.pointer as u64}
    /// The words: to be read only when no launch that writes them is running.
    pub fn read(&self,out:&mut [u64]){unsafe{std::ptr::copy_nonoverlapping(self.pointer as *const u64,out.as_mut_ptr(),self.len.min(out.len()));}}
}
impl Drop for HostWords{fn drop(&mut self){if let Ok(d)=driver(){unsafe{(d.api.cuMemFreeHost)(self.pointer as *mut c_void);}}}}
