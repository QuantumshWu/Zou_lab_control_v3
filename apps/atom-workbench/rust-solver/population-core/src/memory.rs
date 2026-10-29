//! The machine's installed physical memory, which the pipeline's memory
//! guard takes its budget from (`lattice::pipeline`).

/// MEMORYSTATUSEX.
#[repr(C)]
struct MemoryStatus {
    length: u32,
    load: u32,
    total_physical: u64,
    available_physical: u64,
    total_page_file: u64,
    available_page_file: u64,
    total_virtual: u64,
    available_virtual: u64,
    available_extended_virtual: u64,
}
#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(buffer: *mut MemoryStatus) -> i32;
}

/// The installed physical memory (bytes).
pub fn physical_bytes() -> Result<f64, String> {
    let mut status = MemoryStatus {
        length: std::mem::size_of::<MemoryStatus>() as u32,
        load: 0,
        total_physical: 0,
        available_physical: 0,
        total_page_file: 0,
        available_page_file: 0,
        total_virtual: 0,
        available_virtual: 0,
        available_extended_virtual: 0,
    };
    if unsafe { GlobalMemoryStatusEx(&mut status) } == 0 {
        return Err("The physical memory could not be read".into());
    }
    Ok(status.total_physical as f64)
}

/// (research, S3) PROCESS_MEMORY_COUNTERS_EX.
#[repr(C)]
struct Counters{cb:u32,faults:u32,peak_ws:usize,ws:usize,a:usize,b:usize,c:usize,e:usize,pagefile:usize,peak_pagefile:usize,private:usize}
#[link(name = "kernel32")]
extern "system" {fn GetCurrentProcess()->isize;fn K32GetProcessMemoryInfo(p:isize,c:*mut Counters,cb:u32)->i32;}
/// (research, S3) The process's private bytes and their peak.
pub fn private_bytes()->(f64,f64){let mut c=Counters{cb:std::mem::size_of::<Counters>() as u32,faults:0,peak_ws:0,ws:0,a:0,b:0,c:0,e:0,pagefile:0,peak_pagefile:0,private:0};
    if unsafe{K32GetProcessMemoryInfo(GetCurrentProcess(),&mut c,c.cb)}==0{return(0.,0.);}(c.private as f64,c.peak_pagefile as f64)}
