//! `zlc_durable`'s publication of files: a same-directory temporary synced and
//! renamed into place, and directories created and flushed level by level, so
//! a published name is crash-durable or the error says it is not.

use std::ffi::OsString;
use std::fs::File;
use std::path::{Component, Path, PathBuf};

use crate::error::{ArchiveError, ErrorKind, Result};

/// `Path.expanduser()`: `~` and, on Windows, `~user` as a sibling profile.
fn expand_user(path: &Path) -> Result<PathBuf> {
    let text = path.to_string_lossy();
    if !text.starts_with('~') { return Ok(path.to_path_buf()); }
    let unknown = || ArchiveError::new(ErrorKind::Runtime, "Could not determine home directory.");
    let end = text.find(if cfg!(windows) { &['/', '\\'][..] } else { &['/'][..] }).unwrap_or(text.len());
    let home = if cfg!(windows) {
        std::env::var_os("USERPROFILE").map(PathBuf::from).or_else(|| {
            let drive = PathBuf::from(std::env::var_os("HOMEDRIVE").unwrap_or_default());
            std::env::var_os("HOMEPATH").map(|path| drive.join(path))
        })
    } else {
        std::env::var_os("HOME").map(PathBuf::from)
    };
    let mut home = home.ok_or_else(unknown)?;
    if end != 1 {
        let user = &text[1..end];
        let current = std::env::var("USERNAME").ok();
        if !cfg!(windows) { return Err(unknown()); }
        if current.as_deref() != Some(user) {
            if current.as_deref() != home.file_name().map(|name| name.to_string_lossy()).as_deref() { return Err(unknown()); }
            home = home.with_file_name(user);
        }
    }
    let mut joined = OsString::from(home);
    joined.push(&text[end..]);
    Ok(PathBuf::from(joined))
}

/// A canonical path without the `\\?\` prefix Windows adds.
fn plain(path: PathBuf) -> PathBuf {
    let text = path.to_string_lossy();
    if let Some(rest) = text.strip_prefix(r"\\?\UNC\") { return PathBuf::from(format!(r"\\{rest}")); }
    if let Some(rest) = text.strip_prefix(r"\\?\") { return PathBuf::from(rest); }
    path
}

/// `Path(path).expanduser().resolve()`: absolute and normalized, with the
/// longest existing prefix resolved to its final on-disk name.
pub fn resolve(path: &Path) -> Result<PathBuf> {
    let expanded = expand_user(path)?;
    let absolute = if expanded.is_absolute() { expanded } else { std::env::current_dir().unwrap_or_default().join(expanded) };
    let mut normal = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => { normal.pop(); }
            other => normal.push(other),
        }
    }
    let mut existing = normal.clone();
    let mut tail: Vec<OsString> = Vec::new();
    loop {
        if let Ok(real) = std::fs::canonicalize(&existing) {
            let mut real = plain(real);
            for part in tail.iter().rev() { real.push(part); }
            return Ok(real);
        }
        match (existing.file_name().map(ToOwned::to_owned), existing.parent().map(Path::to_path_buf)) {
            (Some(name), Some(parent)) => { tail.push(name); existing = parent; }
            _ => return Ok(normal),
        }
    }
}

fn not_a_directory(path: &Path) -> ArchiveError { ArchiveError::new(ErrorKind::NotADirectory, path.display().to_string()) }

/// A failed Win32 call (`os.mkdir`, `os.replace`, `os.unlink`) as Python words it.
fn win32(error: &std::io::Error, paths: &[&str]) -> ArchiveError {
    #[cfg(windows)]
    if let Some(code) = error.raw_os_error() { return ArchiveError::windows(code as u32, paths); }
    ArchiveError::os(error, paths.first().copied())
}

/// `os.replace`: `MoveFileExW(source, target, MOVEFILE_REPLACE_EXISTING)`,
/// which refuses a target another handle holds open.
#[cfg(windows)]
fn replace(source: &Path, target: &Path) -> std::io::Result<()> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" { fn MoveFileExW(existing: *const u16, new: *const u16, flags: u32) -> i32; }
    // As std names a file to Win32: a long absolute path in its verbatim form.
    let wide = |path: &Path| {
        let text = path.to_string_lossy();
        let name = if text.len() < 248 || text.starts_with(r"\\?\") { text.into_owned() }
            else if let Some(share) = text.strip_prefix(r"\\") { format!(r"\\?\UNC\{share}") } else { format!(r"\\?\{text}") };
        std::ffi::OsStr::new(&name).encode_wide().chain(Some(0)).collect::<Vec<u16>>()
    };
    let (source, target) = (wide(source), wide(target));
    // SAFETY: both names are NUL-terminated UTF-16 buffers alive across the call.
    let moved = unsafe { MoveFileExW(source.as_ptr(), target.as_ptr(), 1) };
    if moved == 0 { Err(std::io::Error::last_os_error()) } else { Ok(()) }
}

#[cfg(not(windows))]
fn replace(source: &Path, target: &Path) -> std::io::Result<()> { std::fs::rename(source, target) }

/// `tempfile.mkstemp(prefix, suffix, dir)`: eight random characters from
/// tempfile's alphabet, created exclusively; a name taken (or, on Windows,
/// refused as a directory would be) draws another, up to `TMP_MAX` names.
fn mkstemp(directory: &Path, prefix: &str, suffix: &str) -> Result<(PathBuf, File)> {
    use std::hash::{BuildHasher, Hasher};
    const CHARACTERS: &[u8] = b"abcdefghijklmnopqrstuvwxyz0123456789_";
    const TMP_MAX: usize = 10_000;
    let mut state = std::collections::hash_map::RandomState::new().build_hasher().finish() | 1;
    for _ in 0..TMP_MAX {
        let name: String = (0..8).map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            CHARACTERS[(state % CHARACTERS.len() as u64) as usize] as char
        }).collect();
        let path = directory.join(format!("{prefix}{name}{suffix}"));
        let mut options = std::fs::OpenOptions::new();
        options.read(true).write(true).create_new(true);
        #[cfg(windows)]
        std::os::windows::fs::OpenOptionsExt::share_mode(&mut options, 0x3);
        match options.open(&path) {
            Ok(file) => return Ok((path, file)),
            Err(error) => {
                let error = ArchiveError::io(&error, &path.to_string_lossy());
                let retry = error.kind == ErrorKind::FileExists || (cfg!(windows) && error.kind == ErrorKind::Permission && directory.is_dir());
                if !retry { return Err(error); }
            }
        }
    }
    Err(ArchiveError::new(ErrorKind::FileExists, "[Errno 17] No usable temporary file name found"))
}

#[cfg(windows)]
fn flush_handle(directory: &Path) -> std::io::Result<()> {
    use std::os::windows::fs::OpenOptionsExt;
    // GENERIC_WRITE (FlushFileBuffers needs it), shared read/write/delete,
    // FILE_FLAG_BACKUP_SEMANTICS for a directory handle.
    std::fs::OpenOptions::new().access_mode(0x4000_0000).share_mode(0x7).custom_flags(0x0200_0000).open(directory)?.sync_all()
}

#[cfg(not(windows))]
fn flush_handle(directory: &Path) -> std::io::Result<()> { File::open(directory)?.sync_all() }

/// `flush_directory`: durably flush one existing directory or fail explicitly.
pub fn flush_directory(directory: &Path) -> Result<()> {
    let path = resolve(directory)?;
    if !path.is_dir() { return Err(not_a_directory(&path)); }
    flush_handle(&path).map_err(|_| ArchiveError::new(ErrorKind::DirectoryDurability, format!("directory durability flush failed for {}", path.display())))
}

/// `_flush_published`: the flush acknowledging an already visible artifact.
fn flush_published(directory: &Path, published: &Path) -> Result<()> {
    flush_directory(directory).map_err(|error| match error.kind {
        ErrorKind::DirectoryDurability => ArchiveError::new(ErrorKind::DirectoryDurability, format!(
            "{} is published and visible, but its directory entry could not be made crash-durable: {}", published.display(), error.message)),
        _ => error,
    })
}

/// `durable_mkdir`: confirm a directory and its parent entry, creating it when missing.
pub fn durable_mkdir(directory: &Path) -> Result<PathBuf> {
    let target = resolve(directory)?;
    let parent = target.parent().map_or_else(|| target.clone(), Path::to_path_buf);
    if !parent.exists() {
        return Err(ArchiveError::new(ErrorKind::FileNotFound, format!("parent directory does not exist for durable mkdir: {}", parent.display())));
    }
    if !parent.is_dir() { return Err(not_a_directory(&parent)); }
    if let Err(error) = std::fs::create_dir(&target) {
        if error.kind() != std::io::ErrorKind::AlreadyExists { return Err(win32(&error, &[&target.to_string_lossy()])); }
        if !target.is_dir() { return Err(not_a_directory(&target)); }
    }
    flush_published(&target, &target)?;
    if parent != target { flush_published(&parent, &target)?; }
    Ok(target)
}

/// `durable_makedirs`: confirm the nearest existing anchor, then create each missing level.
pub fn durable_makedirs(directory: &Path) -> Result<PathBuf> {
    let target = resolve(directory)?;
    let mut missing = Vec::new();
    let mut anchor = target.clone();
    while !anchor.exists() {
        missing.push(anchor.clone());
        match anchor.parent() { Some(parent) => anchor = parent.to_path_buf(), None => break }
    }
    if !anchor.is_dir() { return Err(not_a_directory(&anchor)); }
    durable_mkdir(&anchor)?;
    for level in missing.iter().rev() { durable_mkdir(level)?; }
    Ok(target)
}

/// `atomic_write_file(target, writer)`: `write` fills a same-directory
/// temporary, which is synced, renamed over `target` and acknowledged.
pub fn atomic_write_file(target: &Path, write: impl FnOnce(File) -> Result<File>) -> Result<PathBuf> {
    let destination = resolve(target)?;
    let parent = destination.parent().map_or_else(|| destination.clone(), Path::to_path_buf);
    if !parent.is_dir() {
        return Err(ArchiveError::new(ErrorKind::FileNotFound, format!("target parent directory does not exist: {}", parent.display())));
    }
    let name = destination.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
    let (temporary, file) = mkstemp(&parent, &format!(".{name}."), ".tmp")?;
    let display = temporary.to_string_lossy().into_owned();
    let result = write(file).and_then(|file| {
        file.sync_all().map_err(|error| ArchiveError::os(&error, None))?;
        drop(file);
        replace(&temporary, &destination).map_err(|error| win32(&error, &[&display, &destination.to_string_lossy()]))?;
        flush_published(&parent, &destination)
    });
    if let Err(original) = result {
        // `temporary.unlink()` in the handler: a missing file is fine, any other failure replaces the error.
        return Err(match std::fs::remove_file(&temporary) {
            Err(error) if error.kind() != std::io::ErrorKind::NotFound => win32(&error, &[&display]),
            _ => original,
        });
    }
    Ok(destination)
}
