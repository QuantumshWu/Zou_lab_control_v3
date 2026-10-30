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
