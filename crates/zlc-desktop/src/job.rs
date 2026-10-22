//! A child process ended with its holder, optionally held to a memory limit: a
//! Windows job object (the operator's scan program runs in one, and so does
//! each step installation.exe runs).
use std::os::windows::io::AsRawHandle;

use windows_sys::Win32::Foundation::{CloseHandle, HANDLE};
use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, CreateJobObjectW, JobObjectExtendedLimitInformation,
    JOB_OBJECT_LIMIT_JOB_MEMORY, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, SetInformationJobObject};

/// The job; dropping it ends whatever it still holds.
pub struct Limited(HANDLE);

impl Drop for Limited {
    fn drop(&mut self) { unsafe { CloseHandle(self.0) }; }
}

/// Put `child` in a job of at most `memory_mib` (None: no limit); None when Windows refuses (the child still runs).
pub fn limit(child: &std::process::Child, memory_mib: Option<u64>) -> Option<Limited> {
    let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
    if handle.is_null() { return None; }
    let job = Limited(handle);
    let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
    if let Some(memory) = memory_mib {
        info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
        info.JobMemoryLimit = (memory as usize) << 20;
    }
    let set = unsafe { SetInformationJobObject(handle, JobObjectExtendedLimitInformation,
        (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) };
    if set == 0 { return None; }
    if unsafe { AssignProcessToJobObject(handle, child.as_raw_handle() as HANDLE) } == 0 { return None; }
    Some(job)
}
