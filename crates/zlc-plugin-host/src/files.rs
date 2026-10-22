//! A device command's file is on the computer of the core that asks for it
//! (PLUGIN_PROTOCOL §8 `file`): the core reads the file and the plugin gets its
//! bytes, or the plugin gives the bytes and the core writes the file. A device
//! another core serves (zlc-fabric) thus reads and writes the operator's files,
//! as a device of this core does (v2: the editor's files were the operator's).
use std::path::{Path, PathBuf};
use std::sync::Arc;

use base64::Engine;
use serde_json::{Value, json};
use zlc_core::ForeignBytes;

use crate::blocks::{Lend, Registry};
use crate::descriptor::FileParam;
use crate::wire::INLINE_BYTES;
use crate::{CallError, ErrorKind, Reply};

fn failed(message: String) -> CallError { CallError::new(ErrorKind::Failed, message) }

/// The file a command's params name.
pub fn path(params: &Value, file: &FileParam) -> Result<PathBuf, CallError> {
    params["params"][&file.param].as_str().map(str::trim).filter(|path| !path.is_empty()).map(PathBuf::from)
        .ok_or_else(|| CallError::new(ErrorKind::Invalid, "Name the file"))
}

/// The file as the plugin is handed it: its name, and for a read its bytes
/// (inline when small, else a block of this core); the blocks it lends.
pub fn hand(params: &mut Value, file: &FileParam, path: &Path, registry: &Arc<Registry>) -> Result<Vec<Arc<Lend>>, CallError> {
    let mut given = json!({"name": path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default()});
    let mut lends = Vec::new();
    if file.direction == "read" {
        let bytes = std::fs::read(path).map_err(|error| failed(format!("Could not read {}: {error}", path.display())))?;
        given["bytes"] = if bytes.len() < INLINE_BYTES {
            json!({"$inline": {"b64": base64::engine::general_purpose::STANDARD.encode(&bytes), "dtype": "|u1", "shape": [bytes.len()]}})
        } else {
            let lend = registry.export(&bytes).map_err(failed)?;
            let reference = json!({"$array": {"block": lend.name(), "size": lend.bytes().len(), "offset": 0, "dtype": "|u1", "shape": [bytes.len()]}});
            lends.push(Arc::new(lend));
            reference
        };
    }
    params["params"][&file.param] = given;
    Ok(lends)
}

/// A write's answer: the bytes the plugin gave (`file`) replace the file at
/// `path` whole or not at all; the caller gets the command's own `value`.
pub fn write(reply: Reply, path: &Path) -> Result<Reply, CallError> {
    let array = reply.array(&reply.value["file"]).map_err(|error| failed(format!("The command gave no file: {error}")))?;
    let bytes = array.values.as_bytes();
    let saved = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) { std::fs::create_dir_all(parent)?; }
        let name = path.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let partial = path.with_file_name(format!(".{name}.{}.partial", std::process::id()));
        std::fs::write(&partial, bytes)?;
        std::fs::rename(&partial, path).inspect_err(|_| { let _ = std::fs::remove_file(&partial); })
    })();
    saved.map_err(|error| failed(format!("Could not write {}: {error}", path.display())))?;
    Ok(Reply::new(reply.value["value"].clone(), Vec::new()))
}
