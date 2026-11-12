//! The processes this application starts, and the ones on this computer that hold
//! what it cannot open.
//!
//! Every long-lived child runs in a [`Job`] that ends it with its holder, from its
//! first instruction: it is created suspended, placed in the job, and only then
//! resumed. Nothing it starts (a venv launcher's interpreter, Vivado's helpers) can
//! begin outside the job, and a process the job cannot take never runs.
//!
//! [`holders`] answers, when a device's open failed because the device is held, which
//! processes on this computer have the device vendor's transport library loaded.
use std::collections::HashMap;
use std::io;
use std::os::windows::io::{AsRawHandle, FromRawHandle, OwnedHandle};
use std::os::windows::process::CommandExt;
use std::process::{Child, Command};

use serde_json::{Value, json};
use windows_sys::Wdk::System::Threading::{NtQueryInformationProcess, ProcessCommandLineInformation};
use windows_sys::Win32::Foundation::{CloseHandle, FILETIME, HANDLE, INVALID_HANDLE_VALUE, SYSTEMTIME, UNICODE_STRING};
use windows_sys::Win32::System::Diagnostics::ToolHelp::{CreateToolhelp32Snapshot, MODULEENTRY32W, Module32FirstW, Module32NextW,
    PROCESSENTRY32W, Process32FirstW, Process32NextW, TH32CS_SNAPMODULE, TH32CS_SNAPMODULE32, TH32CS_SNAPPROCESS, TH32CS_SNAPTHREAD,
    THREADENTRY32, Thread32First, Thread32Next};
use windows_sys::Win32::System::JobObjects::{AssignProcessToJobObject, CreateJobObjectW, IsProcessInJob, JOB_OBJECT_LIMIT_JOB_MEMORY,
    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE, JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation, SetInformationJobObject,
    TerminateJobObject};
use windows_sys::Win32::System::Threading::{CREATE_SUSPENDED, GetProcessTimes, OpenProcess, OpenThread, PROCESS_NAME_WIN32,
    PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW, ResumeThread, THREAD_SUSPEND_RESUME};
use windows_sys::Win32::System::Time::{FileTimeToSystemTime, SystemTimeToTzSpecificLocalTime};

/// A job object whose processes end when it closes (with the application, at the
/// latest), optionally held together to a memory limit.
pub struct Job(usize);

impl Job {
    pub fn new(memory_limit_mib: Option<u64>) -> io::Result<Self> {
        let handle = unsafe { CreateJobObjectW(std::ptr::null(), std::ptr::null()) };
        if handle.is_null() { return Err(io::Error::last_os_error()); }
        let job = Self(handle as usize);
        let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        if let Some(limit) = memory_limit_mib {
            info.BasicLimitInformation.LimitFlags |= JOB_OBJECT_LIMIT_JOB_MEMORY;
            info.JobMemoryLimit = (limit as usize) << 20;
        }
        let set = unsafe { SetInformationJobObject(handle, JobObjectExtendedLimitInformation,
            (&info as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(), size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) };
        if set == 0 { return Err(io::Error::last_os_error()); }
        Ok(job)
    }

    fn handle(&self) -> HANDLE { self.0 as HANDLE }

    /// Start `command` inside this job. `flags` are its creation flags (they replace any
    /// the command was given; `CREATE_SUSPENDED` is added): it runs only once it is in
    /// the job, and one the job cannot take is ended before its first instruction.
    pub fn spawn(&self, command: &mut Command, flags: u32) -> io::Result<Child> {
        let mut child = command.creation_flags(flags | CREATE_SUSPENDED).spawn()?;
        let placed = if unsafe { AssignProcessToJobObject(self.handle(), child.as_raw_handle() as HANDLE) } == 0 {
            Err(io::Error::other(format!("the process could not be placed in its job: {}", io::Error::last_os_error())))
        } else { resume(child.id()) };
        if let Err(error) = placed {
            let _ = child.kill();
            let _ = child.wait();
            return Err(error);
        }
        Ok(child)
    }

    /// Whether `process` (a child, or a handle to one of its descendants) is in this job.
    pub fn holds(&self, process: &impl AsRawHandle) -> bool {
        let mut inside = 0;
        unsafe { IsProcessInJob(process.as_raw_handle() as HANDLE, self.handle(), &mut inside) != 0 && inside != 0 }
    }

    /// End every process in the job now.
    pub fn terminate(&self) -> io::Result<()> {
        if unsafe { TerminateJobObject(self.handle(), 1) } == 0 { return Err(io::Error::last_os_error()); }
        Ok(())
    }
}

impl Drop for Job {
    fn drop(&mut self) { unsafe { CloseHandle(self.handle()) }; }
}

fn owned(handle: HANDLE) -> Option<OwnedHandle> {
    (!handle.is_null() && handle != INVALID_HANDLE_VALUE).then(|| unsafe { OwnedHandle::from_raw_handle(handle as _) })
}

/// Resume the (one) thread of a process created suspended.
fn resume(pid: u32) -> io::Result<()> {
    let snapshot = owned(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0) }).ok_or_else(io::Error::last_os_error)?;
    let mut entry: THREADENTRY32 = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<THREADENTRY32>() as u32;
    let mut resumed = 0;
    let mut more = unsafe { Thread32First(snapshot.as_raw_handle() as HANDLE, &mut entry) } != 0;
    while more {
        if entry.th32OwnerProcessID == pid {
            let thread = owned(unsafe { OpenThread(THREAD_SUSPEND_RESUME, 0, entry.th32ThreadID) }).ok_or_else(io::Error::last_os_error)?;
            if unsafe { ResumeThread(thread.as_raw_handle() as HANDLE) } == u32::MAX { return Err(io::Error::last_os_error()); }
            resumed += 1;
        }
        more = unsafe { Thread32Next(snapshot.as_raw_handle() as HANDLE, &mut entry) } != 0;
    }
    if resumed == 0 { return Err(io::Error::other(format!("process {pid} has no thread to resume"))); }
    Ok(())
}

// ------------------------------------------------------------------ holders

/// A process on this computer with a device vendor's transport library loaded.
#[derive(Clone, Debug, PartialEq)]
pub struct Holder {
    pub pid: u32,
    pub exe: String,
    /// None when this user may not read it.
    pub command_line: Option<String>,
    /// Microseconds since the Unix epoch, as the journal's times.
    pub started_us: Option<u64>,
    /// A descendant of this application (or the application itself).
    pub ours: bool,
    /// The loaded modules that matched.
    pub modules: Vec<String>,
}

impl Holder {
    pub fn json(&self) -> Value {
        json!({"pid": self.pid, "exe": self.exe, "command_line": self.command_line, "started_us": self.started_us,
               "started": self.started_us.map(local_time), "ours": self.ours, "modules": self.modules})
    }

    /// One line for an operator: who it is, since when, and whether it is this application's.
    pub fn describe(&self) -> String {
        let started = self.started_us.map(|us| format!(", started {}", local_time(us))).unwrap_or_default();
        let whose = if self.ours { "this application's" } else { "not this application's" };
        let line = self.command_line.as_deref().map(|line| format!(": {line}")).unwrap_or_default();
        format!("pid {} {} ({whose}{started}; {}){line}", self.pid, self.exe, self.modules.join(", "))
    }
}

/// What [`holders`] found: the processes, and how many it could not look into (the
/// system's, other users', elevated ones).
#[derive(Clone, Debug, Default)]
pub struct Holders { pub processes: Vec<Holder>, pub uninspected: usize }

impl Holders {
    pub fn json(&self) -> Value {
        json!({"processes": self.processes.iter().map(Holder::json).collect::<Vec<_>>(), "uninspected": self.uninspected})
    }

    /// For an error message: each holder, and what could not be looked at.
    pub fn describe(&self, library: &str) -> String {
        let unseen = if self.uninspected > 0 { format!(" ({} processes could not be inspected: the system's, other users' or elevated ones)", self.uninspected) } else { String::new() };
        if self.processes.is_empty() { return format!("No process on this computer that could be inspected has {library} loaded{unseen}."); }
        let list: Vec<String> = self.processes.iter().map(Holder::describe).collect();
        format!("Processes on this computer with {library} loaded{unseen}: {}", list.join("; "))
    }
}

/// The processes on this computer that have a module loaded whose file name matches
/// one of `patterns` (`*` any run, `?` one character, case ignored). Asked when an
/// open failed because the device is held: it looks once, at every process.
pub fn holders(patterns: &[String]) -> Holders {
    let mut found = Holders::default();
    if patterns.is_empty() { return found; }
    let parents = processes();
    for &pid in parents.keys() {
        if pid == 0 { continue; }
        let Some(modules) = modules(pid) else { found.uninspected += 1; continue };
        let mut matched: Vec<String> = modules.into_iter().filter(|name| patterns.iter().any(|pattern| matches(pattern, name))).collect();
        if matched.is_empty() { continue; }
        matched.sort();
        matched.dedup();
        let process = owned(unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) });
        let exe = process.as_ref().and_then(image).unwrap_or_else(|| parents[&pid].1.clone());
        found.processes.push(Holder { pid, exe, command_line: process.as_ref().and_then(command_line), started_us: started_us(pid),
            ours: descends_from_this_process(pid, &parents), modules: matched });
    }
    found.processes.sort_by_key(|holder| holder.pid);
    found
}

/// Every process now: pid -> (parent pid, executable file name).
fn processes() -> HashMap<u32, (u32, String)> {
    let mut all = HashMap::new();
    let Some(snapshot) = owned(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0) }) else { return all };
    let mut entry: PROCESSENTRY32W = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<PROCESSENTRY32W>() as u32;
    let mut more = unsafe { Process32FirstW(snapshot.as_raw_handle() as HANDLE, &mut entry) } != 0;
    while more {
        all.insert(entry.th32ProcessID, (entry.th32ParentProcessID, wide(&entry.szExeFile)));
        more = unsafe { Process32NextW(snapshot.as_raw_handle() as HANDLE, &mut entry) } != 0;
    }
    all
}

/// The file names of the modules `pid` has loaded; None when it cannot be looked into.
fn modules(pid: u32) -> Option<Vec<String>> {
    let snapshot = owned(unsafe { CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid) })?;
    let mut entry: MODULEENTRY32W = unsafe { std::mem::zeroed() };
    entry.dwSize = size_of::<MODULEENTRY32W>() as u32;
    let mut names = Vec::new();
    let mut more = unsafe { Module32FirstW(snapshot.as_raw_handle() as HANDLE, &mut entry) } != 0;
    while more {
        names.push(wide(&entry.szModule));
        more = unsafe { Module32NextW(snapshot.as_raw_handle() as HANDLE, &mut entry) } != 0;
    }
    Some(names)
}

fn wide(text: &[u16]) -> String { String::from_utf16_lossy(&text[..text.iter().position(|&unit| unit == 0).unwrap_or(text.len())]) }

fn image(process: &OwnedHandle) -> Option<String> {
    let mut buffer = vec![0u16; 32768];
    let mut size = buffer.len() as u32;
    let ok = unsafe { QueryFullProcessImageNameW(process.as_raw_handle() as HANDLE, PROCESS_NAME_WIN32, buffer.as_mut_ptr(), &mut size) };
    (ok != 0).then(|| String::from_utf16_lossy(&buffer[..size as usize]))
}

fn command_line(process: &OwnedHandle) -> Option<String> {
    let mut needed = 0u32;
    unsafe { NtQueryInformationProcess(process.as_raw_handle() as HANDLE, ProcessCommandLineInformation, std::ptr::null_mut(), 0, &mut needed) };
    if needed < size_of::<UNICODE_STRING>() as u32 { return None; }
    // u64 words: the answer starts with a UNICODE_STRING (pointer-aligned) whose text follows it.
    let mut buffer = vec![0u64; (needed as usize).div_ceil(8)];
    let status = unsafe { NtQueryInformationProcess(process.as_raw_handle() as HANDLE, ProcessCommandLineInformation,
        buffer.as_mut_ptr().cast(), (buffer.len() * 8) as u32, &mut needed) };
    if status < 0 { return None; }
    let text = unsafe { &*buffer.as_ptr().cast::<UNICODE_STRING>() };
    if text.Buffer.is_null() { return None; }
    let units = unsafe { std::slice::from_raw_parts(text.Buffer, usize::from(text.Length) / 2) };
    Some(String::from_utf16_lossy(units))
}

/// When `pid` started, in microseconds since the Unix epoch.
fn started_us(pid: u32) -> Option<u64> {
    let process = owned(unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid) })?;
    let zero = FILETIME { dwLowDateTime: 0, dwHighDateTime: 0 };
    let (mut created, mut exited, mut kernel, mut user) = (zero, zero, zero, zero);
    if unsafe { GetProcessTimes(process.as_raw_handle() as HANDLE, &mut created, &mut exited, &mut kernel, &mut user) } == 0 { return None; }
    let ticks = (u64::from(created.dwHighDateTime) << 32) | u64::from(created.dwLowDateTime);
    // FILETIME counts 100 ns from 1601-01-01, 11 644 473 600 s before the Unix epoch.
    Some((ticks / 10).saturating_sub(11_644_473_600_000_000))
}

/// Whether `pid` is this process or a descendant of it. A parent link counts only
/// when the parent started before the child: a parent pid Windows has since given
/// to a newer process is not the parent.
fn descends_from_this_process(pid: u32, parents: &HashMap<u32, (u32, String)>) -> bool {
    let me = std::process::id();
    let (mut child, mut child_started) = (pid, started_us(pid));
    for _ in 0..=parents.len() {
        if child == me { return true; }
        let Some(&(parent, _)) = parents.get(&child) else { return false };
        let parent_started = started_us(parent);
        match (parent_started, child_started) {
            (Some(parent_at), Some(child_at)) if parent_at <= child_at => {}
            _ => return false,
        }
        (child, child_started) = (parent, parent_started);
    }
    false
}

/// `name` matched against `pattern`: `*` any run, `?` one character, case ignored.
pub fn matches(pattern: &str, name: &str) -> bool {
    let pattern: Vec<char> = pattern.to_lowercase().chars().collect();
    let name: Vec<char> = name.to_lowercase().chars().collect();
    let (mut p, mut n, mut star, mut resume_at) = (0, 0, None, 0);
    while n < name.len() {
        if p < pattern.len() && (pattern[p] == '?' || pattern[p] == name[n]) { p += 1; n += 1; }
        else if p < pattern.len() && pattern[p] == '*' { star = Some(p); resume_at = n; p += 1; }
        else if let Some(at) = star { p = at + 1; resume_at += 1; n = resume_at; }
        else { return false; }
    }
    pattern[p..].iter().all(|&char| char == '*')
}

/// A journal time as this computer's local clock reads it: `2026-11-12 10:03:11.250`.
pub fn local_time(us: u64) -> String {
    let ticks = (us + 11_644_473_600_000_000) * 10;
    let file = FILETIME { dwLowDateTime: ticks as u32, dwHighDateTime: (ticks >> 32) as u32 };
    let (mut utc, mut local): (SYSTEMTIME, SYSTEMTIME) = unsafe { (std::mem::zeroed(), std::mem::zeroed()) };
    if unsafe { FileTimeToSystemTime(&file, &mut utc) } == 0 || unsafe { SystemTimeToTzSpecificLocalTime(std::ptr::null(), &utc, &mut local) } == 0 {
        return format!("{us} us");
    }
    format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:03}", local.wYear, local.wMonth, local.wDay, local.wHour, local.wMinute, local.wSecond, local.wMilliseconds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn module_patterns_match_as_file_globs_ignoring_case() {
        assert!(matches("Pylon*", "PylonBase_v9.dll") && matches("pylon*", "PYLONUSB_V9.DLL"));
        assert!(matches("GCBase*Basler*", "GCBase_MD_VC141_v3_1_Basler_pylon_v3.dll"));
        assert!(!matches("GCBase*Basler*", "GCBase_MD_VC141_v3_1.dll"));
        assert!(matches("dcamapi?.dll", "dcamapi4.dll") && !matches("dcamapi?.dll", "dcamapi.dll"));
        assert!(matches("*", "") && !matches("a", ""));
    }

    #[test]
    fn a_spawned_process_is_in_its_job_before_it_runs() {
        let job = Job::new(None).unwrap();
        // `exit 7` would be over long before an assignment after the start could be checked.
        let mut child = job.spawn(Command::new("cmd.exe").args(["/c", "exit 7"]), 0x0800_0000).unwrap();
        assert!(job.holds(&child), "in the job the moment spawn returns");
        assert_eq!(child.wait().unwrap().code(), Some(7), "and it then ran");
    }

    #[test]
    fn a_process_its_job_cannot_take_never_runs() {
        use windows_sys::Win32::System::JobObjects::JOB_OBJECT_LIMIT_ACTIVE_PROCESS;
        let job = Job::new(None).unwrap();
        // A job full at one process refuses a second.
        let mut limit: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = unsafe { std::mem::zeroed() };
        limit.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_ACTIVE_PROCESS | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        limit.BasicLimitInformation.ActiveProcessLimit = 1;
        assert_ne!(unsafe { SetInformationJobObject(job.handle(), JobObjectExtendedLimitInformation, (&limit as *const JOBOBJECT_EXTENDED_LIMIT_INFORMATION).cast(),
            size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32) }, 0);
        let mut first = job.spawn(Command::new("cmd.exe").args(["/c", "ping -n 30 127.0.0.1 >nul"]), 0x0800_0000).unwrap();
        let mark = std::env::temp_dir().join(format!("zlc-job-refused-{}.txt", std::process::id()));
        let _ = std::fs::remove_file(&mark);
        let refused = job.spawn(Command::new("cmd.exe").arg("/c").arg(format!("echo ran> \"{}\"", mark.display())), 0x0800_0000);
        assert!(refused.is_err(), "the job could not take it");
        assert!(!mark.exists(), "it never ran a single instruction of its own");
        let _ = first.kill();
        let _ = first.wait();
    }

    #[test]
    fn this_process_is_its_own_descendant_and_a_holder_is_found_by_its_loaded_module() {
        let parents = processes();
        assert!(descends_from_this_process(std::process::id(), &parents));
        let me = std::env::current_exe().unwrap().file_name().unwrap().to_string_lossy().into_owned();
        let found = holders(&[me.clone()]);
        let holder = found.processes.iter().find(|holder| holder.pid == std::process::id()).expect("this test process holds its own image");
        assert!(holder.ours && holder.modules == vec![me.clone()] && holder.started_us.is_some());
        assert!(holder.command_line.as_deref().is_some_and(|line| line.contains(&me)), "{holder:?}");
    }
}
