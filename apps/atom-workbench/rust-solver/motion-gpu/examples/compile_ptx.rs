//! Compiles kernels/motion.cu to kernels/motion.ptx with the pinned NVRTC (scripts/ptx.ps1 fetches
//! it and runs this): `cargo run --example compile_ptx -- <nvrtc64_120_0.dll>`.
use atom_motion_gpu::ptx::{header,NVRTC_OPTIONS,NVRTC_VERSION,SOURCE};
use std::ffi::{c_char,c_void,CStr,CString};

type Handle=*mut c_void;
#[link(name="kernel32")]
extern "system"{fn LoadLibraryW(name:*const u16)->Handle;fn GetProcAddress(module:Handle,name:*const c_char)->Handle;}

fn main(){
    let dll=std::env::args().nth(1).expect("usage: compile_ptx <path to nvrtc64_120_0.dll>");
    let wide:Vec<u16>=dll.encode_utf16().chain([0]).collect();
    let library=unsafe{LoadLibraryW(wide.as_ptr())};assert!(!library.is_null(),"cannot load {dll}");
    macro_rules! entry{($name:literal,$t:ty)=>{unsafe{let p=GetProcAddress(library,concat!($name,"\0").as_ptr() as *const c_char);assert!(!p.is_null(),$name);std::mem::transmute::<Handle,$t>(p)}};}
    let version=entry!("nvrtcVersion",unsafe extern "C" fn(*mut i32,*mut i32)->i32);
    let create=entry!("nvrtcCreateProgram",unsafe extern "C" fn(*mut Handle,*const c_char,*const c_char,i32,*const *const c_char,*const *const c_char)->i32);
    let compile=entry!("nvrtcCompileProgram",unsafe extern "C" fn(Handle,i32,*const *const c_char)->i32);
    let log_size=entry!("nvrtcGetProgramLogSize",unsafe extern "C" fn(Handle,*mut usize)->i32);
    let get_log=entry!("nvrtcGetProgramLog",unsafe extern "C" fn(Handle,*mut c_char)->i32);
    let ptx_size=entry!("nvrtcGetPTXSize",unsafe extern "C" fn(Handle,*mut usize)->i32);
    let get_ptx=entry!("nvrtcGetPTX",unsafe extern "C" fn(Handle,*mut c_char)->i32);
    let(mut major,mut minor)=(0,0);unsafe{version(&mut major,&mut minor);}
    assert!(NVRTC_VERSION.starts_with(&format!("{major}.{minor}.")),"NVRTC {major}.{minor} is not the pinned {NVRTC_VERSION}");
    let source=CString::new(SOURCE).unwrap();let name=CString::new("motion.cu").unwrap();
    let options:Vec<CString>=NVRTC_OPTIONS.iter().map(|o|CString::new(*o).unwrap()).collect();
    let pointers:Vec<*const c_char>=options.iter().map(|o|o.as_ptr()).collect();
    let mut program=std::ptr::null_mut();
    unsafe{
        assert_eq!(create(&mut program,source.as_ptr(),name.as_ptr(),0,std::ptr::null(),std::ptr::null()),0,"nvrtcCreateProgram");
        let result=compile(program,pointers.len() as i32,pointers.as_ptr());
        let mut size=0;log_size(program,&mut size);let mut log=vec![0 as c_char;size.max(1)];get_log(program,log.as_mut_ptr());
        let log=CStr::from_ptr(log.as_ptr()).to_string_lossy().into_owned();
        if !log.trim().is_empty(){eprintln!("{log}");}
        assert_eq!(result,0,"NVRTC compilation failed");
        let mut size=0;ptx_size(program,&mut size);let mut ptx=vec![0 as c_char;size];get_ptx(program,ptx.as_mut_ptr());
        let ptx=CStr::from_ptr(ptx.as_ptr()).to_string_lossy().into_owned();
        let out=concat!(env!("CARGO_MANIFEST_DIR"),"/kernels/motion.ptx");
        std::fs::write(out,format!("{}{}",header(SOURCE),ptx.replace("\r\n","\n"))).unwrap();
        println!("wrote {out}: {} bytes of PTX",ptx.len());
    }
}
