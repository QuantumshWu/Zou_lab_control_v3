//! Refusals carrying the Python exception class and message the scientific
//! archive code raises for the same input.

use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ErrorKind {
    Value, Type, Key, Index, Runtime, NotImplemented, Overflow, EmptySelection,
    /// `zlc_data.io.NPZFormatError` (a ValueError).
    NpzFormat,
    /// `zipfile.BadZipFile`.
    BadZipFile,
    Eof, FileNotFound, Permission, Os, BlockingIo, BrokenPipe, ChildProcess,
    /// `zlib.error`.
    Zlib,
    /// `tokenize.TokenError` (an NPY header numpy cannot tokenize).
    Token,
    NotADirectory, UnicodeDecode, Recursion, Memory, FileExists, Syntax, Indentation, Tab, System,
    /// `zlc_durable.DirectoryDurabilityError` (a RuntimeError).
    DirectoryDurability,
}

impl ErrorKind {
    pub fn python_name(self) -> &'static str {
        match self {
            Self::Value => "ValueError", Self::Type => "TypeError", Self::Key => "KeyError", Self::Index => "IndexError",
            Self::Runtime => "RuntimeError", Self::NotImplemented => "NotImplementedError", Self::Overflow => "OverflowError",
            Self::EmptySelection => "EmptySelection", Self::NpzFormat => "NPZFormatError", Self::BadZipFile => "BadZipFile",
            Self::Eof => "EOFError", Self::FileNotFound => "FileNotFoundError", Self::Permission => "PermissionError", Self::Os => "OSError",
            Self::BlockingIo => "BlockingIOError", Self::BrokenPipe => "BrokenPipeError", Self::ChildProcess => "ChildProcessError",
            Self::Zlib => "error", Self::Token => "TokenError", Self::NotADirectory => "NotADirectoryError", Self::UnicodeDecode => "UnicodeDecodeError",
            Self::Recursion => "RecursionError", Self::Memory => "MemoryError", Self::FileExists => "FileExistsError", Self::Syntax => "SyntaxError",
            Self::Indentation => "IndentationError", Self::Tab => "TabError", Self::System => "SystemError",
            Self::DirectoryDurability => "DirectoryDurabilityError",
        }
    }
}

/// `str(exception)` is `message`; `Display` is `Class: message`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArchiveError { pub kind: ErrorKind, pub message: String }

impl ArchiveError {
    pub fn new(kind: ErrorKind, message: impl Into<String>) -> Self { Self { kind, message: message.into() } }
    pub fn value(message: impl Into<String>) -> Self { Self::new(ErrorKind::Value, message) }
    pub fn type_error(message: impl Into<String>) -> Self { Self::new(ErrorKind::Type, message) }
    pub fn npz(message: impl Into<String>) -> Self { Self::new(ErrorKind::NpzFormat, message) }
    pub fn zip(message: impl Into<String>) -> Self { Self::new(ErrorKind::BadZipFile, message) }
    /// A C runtime call's failure on `path` (`open`), as Python words it:
    /// `[Errno n] text: 'path'`.
    pub fn io(error: &std::io::Error, path: &str) -> Self { Self::os(error, Some(path)) }

    /// A C runtime call's failure (`open`, `read`, `fsync`): Windows codes
    /// become the CRT's errno (`_dosmaperr`), worded by its `strerror`.
    pub fn os(error: &std::io::Error, path: Option<&str>) -> Self {
        let errno = match error.raw_os_error() {
            Some(code) if cfg!(windows) => crt_errno(code as u32),
            Some(code) => code,
            None => return Self::new(ErrorKind::Os, error.to_string()),
        };
        let text = strerror(errno);
        let message = match path { Some(path) => format!("[Errno {errno}] {text}: {}", crate::json::py_repr_str(path)), None => format!("[Errno {errno}] {text}") };
        Self::new(errno_kind(errno), message)
    }

    /// A Win32 call's failure (`os.replace`, `os.unlink`): `[WinError n]
    /// text: 'a' -> 'b'`, its class from `winerror_to_errno`.
    #[cfg(windows)]
    pub fn windows(code: u32, paths: &[&str]) -> Self {
        let mut message = format!("[WinError {code}] {}", windows_message(code));
        for (index, path) in paths.iter().enumerate() {
            message.push_str(if index == 0 { ": " } else { " -> " });
            message.push_str(&crate::json::py_repr_str(path));
        }
        Self::new(errno_kind(winerror_to_errno(code)), message)
    }
}

/// The `OSError` subclass CPython raises for an errno.
fn errno_kind(errno: i32) -> ErrorKind {
    match errno {
        1 | 13 => ErrorKind::Permission, 2 => ErrorKind::FileNotFound, 17 => ErrorKind::FileExists, 20 => ErrorKind::NotADirectory,
        10 => ErrorKind::ChildProcess, 11 => ErrorKind::BlockingIo, 32 => ErrorKind::BrokenPipe, _ => ErrorKind::Os,
    }
}

/// The C runtime's `strerror` (the UCRT's table).
fn strerror(errno: i32) -> &'static str {
    const TEXT: [&str; 43] = [
        "No error", "Operation not permitted", "No such file or directory", "No such process", "Interrupted function call", "Input/output error",
        "No such device or address", "Arg list too long", "Exec format error", "Bad file descriptor", "No child processes",
        "Resource temporarily unavailable", "Not enough space", "Permission denied", "Bad address", "Unknown error", "Resource device", "File exists",
        "Improper link", "No such device", "Not a directory", "Is a directory", "Invalid argument", "Too many open files in system",
        "Too many open files", "Inappropriate I/O control operation", "Unknown error", "File too large", "No space left on device", "Invalid seek",
        "Read-only file system", "Too many links", "Broken pipe", "Domain error", "Result too large", "Unknown error", "Resource deadlock avoided",
        "Unknown error", "Filename too long", "No locks available", "Function not implemented", "Directory not empty", "Illegal byte sequence",
    ];
    usize::try_from(errno).ok().and_then(|index| TEXT.get(index)).copied().unwrap_or("Unknown error")
}

/// The UCRT's `_dosmaperr`: a Win32 code as the errno a CRT call sets.
fn crt_errno(code: u32) -> i32 {
    match code {
        2 | 3 | 15 | 18 | 53 | 67 | 161 | 206 => 2,
        4 => 24, 5 | 16 | 33 | 65 | 82 | 83 | 108 | 132 | 158 | 167 | 19..=36 => 13,
        6 | 114 | 130 => 9, 7 | 8 | 9 | 1816 => 12, 10 => 7, 11 | 188..=202 => 8, 17 => 18, 80 | 183 => 17,
        89 | 164 | 215 => 11, 109 => 32, 112 => 28, 128 | 129 => 10, 145 => 41,
        _ => 22,
    }
}

/// CPython's `winerror_to_errno` (PC/errmap.h).
#[cfg(windows)]
fn winerror_to_errno(code: u32) -> i32 {
    let code = if code & 0xFFFF_0000 == 0x8007_0000 { code & 0xFFFF } else { code };
    match code {
        10004 | 10009 | 10013 | 10014 | 10022 | 10024 => code as i32 - 10000,
        10000..=11999 => code as i32,
        267 => 20, 232 => 32, 1113 => 42,
        _ => crt_errno(code),
    }
}

/// `FormatMessageW` of a system error, trailing whitespace and periods
/// removed (CPython's `PyErr_SetExcFromWindowsErr`).
#[cfg(windows)]
fn windows_message(code: u32) -> String {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn FormatMessageW(flags: u32, source: *const std::ffi::c_void, message: u32, language: u32, buffer: *mut u16, size: u32,
                          arguments: *const std::ffi::c_void) -> u32;
    }
    const FROM_SYSTEM: u32 = 0x1000;
    const IGNORE_INSERTS: u32 = 0x200;
    let mut buffer = vec![0u16; 4096];
    // SAFETY: the buffer holds `size` UTF-16 units; no inserts are read.
    let length = unsafe { FormatMessageW(FROM_SYSTEM | IGNORE_INSERTS, std::ptr::null(), code, 0x400, buffer.as_mut_ptr(), buffer.len() as u32, std::ptr::null()) };
    if length == 0 { return format!("Windows Error 0x{code:x}"); }
    let mut units = &buffer[..length as usize];
    while let [rest @ .., last] = units { if *last <= u16::from(b' ') || *last == u16::from(b'.') { units = rest } else { break } }
    String::from_utf16_lossy(units)
}

impl fmt::Display for ArchiveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}: {}", self.kind.python_name(), self.message) }
}

impl std::error::Error for ArchiveError {}

impl From<zlc_core::restrict::RestrictError> for ArchiveError {
    fn from(error: zlc_core::restrict::RestrictError) -> Self {
        use zlc_core::restrict::ErrorKind as Kind;
        let kind = match error.kind { Kind::Value => ErrorKind::Value, Kind::Type => ErrorKind::Type, Kind::Index => ErrorKind::Index,
                                      Kind::EmptySelection => ErrorKind::EmptySelection };
        Self::new(kind, error.message)
    }
}

pub type Result<T> = std::result::Result<T, ArchiveError>;
