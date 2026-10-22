//! The catalogue discovery found, kept per checkout in `.local\plugins\catalog.cache`
//! under a key of everything it was found from: every file of the plugin sources
//! (its path, size and modification time), the interpreter, its venv's
//! configuration and installed distributions (the base's too when the venv uses
//! them), the packages asked for and the processes' environment (PATH, where a
//! vendor runtime announces itself). A start whose key matches reads the file
//! instead of running Python; any change runs discovery again, once.
use std::ffi::OsString;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};

use crate::HostConfig;

fn file() -> PathBuf { zlc_core::checkout::local("plugins").join("catalog.cache") }

/// The key of the catalogue `config` discovers with `environment`.
pub(crate) fn key(config: &HostConfig, environment: &[(&str, OsString)]) -> String {
    let mut hasher = DefaultHasher::new();
    ("zlc plugin catalogue 1", crate::PROTOCOL, &config.python, &config.packages).hash(&mut hasher);
    for root in &config.python_path {
        root.hash(&mut hasher);
        tree(root, root, &mut hasher);
    }
    // A venv: its configuration and what is installed in it (each distribution's folder names its version).
    if let Some(venv) = config.python.parent().and_then(Path::parent) {
        let settings = std::fs::read_to_string(venv.join("pyvenv.cfg")).unwrap_or_default();
        settings.hash(&mut hasher);
        listing(&venv.join("Lib").join("site-packages"), &mut hasher);
        let inherits = settings.lines().any(|line| line.split_once('=').is_some_and(|(key, value)|
            key.trim() == "include-system-site-packages" && value.trim().eq_ignore_ascii_case("true")));
        if inherits { listing(&crate::base_home(&config.python).join("Lib").join("site-packages"), &mut hasher); }
    }
    environment.hash(&mut hasher);
    ["PATH", "PYTHONPATH"].map(std::env::var_os).hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

/// Every file under `folder` but bytecode: its path below `root`, size and modification time.
fn tree(root: &Path, folder: &Path, hasher: &mut DefaultHasher) {
    let Ok(entries) = std::fs::read_dir(folder) else { return };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        if entry.file_name() == "__pycache__" { continue; }
        let Ok(meta) = entry.metadata() else { continue };
        let path = entry.path();
        if meta.is_dir() { tree(root, &path, hasher); continue; }
        (path.strip_prefix(root).unwrap_or(&path), meta.len(), meta.modified().ok()).hash(hasher);
    }
}

/// The names in `folder` (a site-packages: one folder per installed distribution and version).
fn listing(folder: &Path, hasher: &mut DefaultHasher) {
    let mut names: Vec<OsString> = std::fs::read_dir(folder).into_iter().flatten().flatten().map(|entry| entry.file_name())
        .filter(|name| name != "__pycache__").collect();
    names.sort();
    names.hash(hasher);
}

/// The kept describe output when it was found under `key`.
pub(crate) fn read(key: &str) -> Option<Vec<u8>> {
    let text = std::fs::read(file()).ok()?;
    let at = text.iter().position(|byte| *byte == b'\n')?;
    (text[..at] == *format!("key {key}").as_bytes()).then(|| text[at + 1..].to_vec())
}

/// Keep `describe` (discovery's stdout) under `key`; a failure to write only costs the next start a discovery.
pub(crate) fn write(key: &str, describe: &[u8]) {
    let path = file();
    if let Some(folder) = path.parent() { let _ = std::fs::create_dir_all(folder); }
    let mut text = format!("key {key}\n").into_bytes();
    text.extend_from_slice(describe);
    let _ = std::fs::write(path, text);
}

#[cfg(test)]
mod tests {
    use crate::HostConfig;

    #[test]
    fn a_source_edit_changes_the_key_and_bytecode_does_not() {
        let root = std::env::temp_dir().join(format!("zlc-catalogue-key-{}", std::process::id()));
        let (bytecode, source) = (root.join("package").join("__pycache__"), root.join("package").join("types.py"));
        std::fs::create_dir_all(&bytecode).unwrap();
        std::fs::write(&source, "A = 1\n").unwrap();
        let config = HostConfig { python: root.join("python.exe"), python_path: vec![root.clone()], packages: vec!["package".into()], memory_limit_mib: None };
        let first = super::key(&config, &[]);
        std::fs::write(bytecode.join("types.pyc"), "compiled").unwrap();
        assert_eq!(super::key(&config, &[]), first, "bytecode is not a source");
        std::fs::write(&source, "A = 12\n").unwrap();
        assert_ne!(super::key(&config, &[]), first, "an edited source is");
        for file in [bytecode.join("types.pyc"), source] { std::fs::remove_file(file).unwrap(); }
        for folder in [bytecode, root.join("package"), root] { std::fs::remove_dir(folder).unwrap(); }
    }
}
