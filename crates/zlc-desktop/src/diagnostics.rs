//! Where the desktop's diagnostics go. zlc-desktop is a Windows GUI program with
//! no console. Its JSON lines and errors (stdout, stderr) stay where whoever
//! started it redirected them (a harness's file or pipe); a stream nobody
//! redirected goes to this run's log, `<checkout>\.local\logs\<name>-<date>-
//! <time>-<pid>.log` (zlc-desktop; installation for installation.exe and its
//! steps), which the children that inherit stderr (the local pulse server)
//! write to as well.
use std::path::PathBuf;

use windows_sys::Win32::Foundation::{HANDLE, INVALID_HANDLE_VALUE};
use windows_sys::Win32::Storage::FileSystem::{FILE_TYPE_DISK, FILE_TYPE_PIPE, GetFileType};
use windows_sys::Win32::System::Console::{GetStdHandle, STD_ERROR_HANDLE, STD_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle};

/// The runs whose logs are kept; older ones are removed when a run starts.
const KEPT: usize = 20;

pub struct Routed {
    /// This run's log, when a stream goes to it.
    pub log: Option<PathBuf>,
    /// Whether a caller reads stderr (it was redirected to a file or pipe).
    errors_read: bool,
}

/// Whether `stream` was handed to this process as a file or a pipe.
fn redirected(stream: STD_HANDLE) -> bool {
    let handle: HANDLE = unsafe { GetStdHandle(stream) };
    !handle.is_null() && handle != INVALID_HANDLE_VALUE && matches!(unsafe { GetFileType(handle) }, FILE_TYPE_DISK | FILE_TYPE_PIPE)
}

/// Send the streams nobody redirected to this run's log, `<name>-…` (Rust's stdout
/// and stderr look the standard handles up on every write).
pub fn route(name: &str) -> Routed {
    let streams: Vec<STD_HANDLE> = [STD_OUTPUT_HANDLE, STD_ERROR_HANDLE].into_iter().filter(|stream| !redirected(*stream)).collect();
    let errors_read = !streams.contains(&STD_ERROR_HANDLE);
    if streams.is_empty() { return Routed { log: None, errors_read }; }
    let folder = zlc_core::checkout::local("logs");
    if std::fs::create_dir_all(&folder).is_err() { return Routed { log: None, errors_read }; }
    // The names sort by start time: all but the newest KEPT - 1 of this name go.
    let prefix = format!("{name}-");
    let mut old: Vec<PathBuf> = std::fs::read_dir(&folder).into_iter().flatten().filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.file_name().and_then(|name| name.to_str()).is_some_and(|name| name.starts_with(&prefix) && name.ends_with(".log")))
        .collect();
    old.sort();
    for path in old.iter().rev().skip(KEPT - 1) { let _ = std::fs::remove_file(path); }
    let mut time = unsafe { std::mem::zeroed::<windows_sys::Win32::Foundation::SYSTEMTIME>() };
    unsafe { windows_sys::Win32::System::SystemInformation::GetLocalTime(&mut time) };
    let path = folder.join(format!("{prefix}{:04}{:02}{:02}-{:02}{:02}{:02}-{}.log", time.wYear, time.wMonth, time.wDay,
                                   time.wHour, time.wMinute, time.wSecond, std::process::id()));
    let Ok(file) = std::fs::OpenOptions::new().append(true).create(true).open(&path) else { return Routed { log: None, errors_read } };
    // The handle lives as long as the process.
    let handle = std::os::windows::io::IntoRawHandle::into_raw_handle(file) as HANDLE;
    for stream in streams { unsafe { SetStdHandle(stream, handle) }; }
    Routed { log: Some(path), errors_read }
}

impl Routed {
    /// A failure that ends the program, said where the operator sees it: a message
    /// box when no caller reads stderr and the windows are on screen.
    pub fn fatal(&self, error: &str) {
        eprintln!("{}", serde_json::json!({"type": "desktop-error", "error": error}));
        if self.errors_read || std::env::var_os("ZLC_V3_WINDOW_AT").is_some() { return; }
        let log = self.log.as_ref().map(|path| format!("\n\nLog: {}", path.display())).unwrap_or_default();
        let text: Vec<u16> = format!("Zou lab control stopped: {error}{log}").encode_utf16().chain([0]).collect();
        let title: Vec<u16> = "Zou lab control".encode_utf16().chain([0]).collect();
        use windows_sys::Win32::UI::WindowsAndMessaging::{MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW};
        unsafe { MessageBoxW(std::ptr::null_mut(), text.as_ptr(), title.as_ptr(), MB_OK | MB_ICONERROR | MB_SETFOREGROUND) };
    }
}
