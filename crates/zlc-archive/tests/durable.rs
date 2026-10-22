//! Durable publication against `zlc_durable` on Windows: every case of
//! `gen/durable_cases.py` (replacement, read-only, open and directory
//! targets, missing parents, the longest name whose temporary still fits, a
//! failing writer, makedirs through a file) ends with Python's outcome,
//! class and message, the same target bytes and no temporary left behind.
//!
//! Committed: `tests/data/durable.json` (`gen/durable_cases.py OUT WORKDIR`).
#![cfg(windows)]

use std::fs::File;
use std::os::windows::fs::OpenOptionsExt;
use std::path::Path;

use zlc_archive::durable::{atomic_write_file, durable_makedirs, resolve};
use zlc_archive::json::py_repr_str;
use zlc_archive::{ArchiveError, ErrorKind};

/// The message with the work directory as `{root}` and a temporary's eight
/// random characters as `********`.
fn normalized(message: &str, root: &Path) -> String {
    let root = root.to_string_lossy();
    let repr = py_repr_str(&root);
    let mut text = message.replace(&repr[1..repr.len() - 1], "{root}").replace(&*root, "{root}");
    let mut from = 0;
    while let Some(found) = text[from..].find(".tmp") {
        let end = from + found;
        if end >= 9 && text.as_bytes()[end - 9] == b'.' && text[end - 8..end].bytes().all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'_') {
            text.replace_range(end - 8..end, "********");
        }
        from = end + 4;
    }
    text
}

fn leftovers(directory: &Path) -> usize {
    std::fs::read_dir(directory).map_or(0, |entries| entries.flatten().filter(|entry| {
        let name = entry.file_name().to_string_lossy().into_owned();
        name.starts_with('.') && name.ends_with(".tmp")
    }).count())
}

#[test]
fn durable_writes_end_like_zlc_durable() {
    let golden: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/durable.json")).unwrap()).unwrap();
    let base = Path::new(env!("CARGO_TARGET_TMPDIR")).join("durable");
    std::fs::create_dir_all(&base).unwrap();
    let root = resolve(&base.join(format!("run-{}", std::process::id()))).unwrap();
    std::fs::create_dir(&root).unwrap();
    let cases = golden.as_array().unwrap();
    let mut failures = Vec::new();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let directory = root.join(name);
        std::fs::create_dir(&directory).unwrap();
        let relative = match name {
            "missing-parent" => "missing/fig.npz".to_owned(),
            "parent-is-file" => "file/fig.npz".to_owned(),
            "name-241" => format!("{}.npz", "n".repeat(237)),
            "name-242" => format!("{}.npz", "n".repeat(238)),
            "makedirs-through-file" => "file/sub/deeper".to_owned(),
            _ => "fig.npz".to_owned(),
        };
        let target = directory.join(&relative);
        let mut held: Option<File> = None;
        match name {
            "replace-existing" | "writer-fails" | "readonly-target" | "held-without-share-delete" | "held-with-share-delete" => std::fs::write(&target, b"OLD").unwrap(),
            "target-is-directory" => std::fs::create_dir(&target).unwrap(),
            "parent-is-file" => std::fs::write(target.parent().unwrap(), b"x").unwrap(),
            "makedirs-through-file" => std::fs::write(target.parent().unwrap().parent().unwrap(), b"x").unwrap(),
            _ => {}
        }
        if name == "readonly-target" {
            let mut permissions = std::fs::metadata(&target).unwrap().permissions();
            permissions.set_readonly(true);
            std::fs::set_permissions(&target, permissions).unwrap();
        }
        if let Some(share) = match name { "held-without-share-delete" => Some(0x3), "held-with-share-delete" => Some(0x7), _ => None } {
            held = Some(std::fs::OpenOptions::new().read(true).share_mode(share).open(&target).unwrap());
        }
        let before = std::fs::read(&target).ok().filter(|_| target.is_file());
        let result = match name {
            "makedirs-through-file" => durable_makedirs(&target).map(|_| ()),
            "writer-fails" => atomic_write_file(&target, |mut file| {
                std::io::Write::write_all(&mut file, b"partial").unwrap();
                Err(ArchiveError::new(ErrorKind::Runtime, "writer failed"))
            }).map(|_| ()),
            _ => atomic_write_file(&target, |mut file| {
                std::io::Write::write_all(&mut file, b"NEW").unwrap();
                Ok(file)
            }).map(|_| ()),
        };
        drop(held);
        if name == "readonly-target" {
            let mut permissions = std::fs::metadata(&target).unwrap().permissions();
            #[allow(clippy::permissions_set_readonly_false)]
            permissions.set_readonly(false);
            std::fs::set_permissions(&target, permissions).unwrap();
        }
        let after = std::fs::read(&target).ok().filter(|_| target.is_file());
        let outcome = match result {
            Ok(()) => serde_json::json!("ok"),
            Err(error) => serde_json::json!([error.kind.python_name(), normalized(&error.message, &root)]),
        };
        let intact = before.as_ref().map(|before| Some(before) == after.as_ref());
        let actual = serde_json::json!({"name": name, "outcome": outcome, "intact": intact,
                                        "written": after.as_deref() == Some(&b"NEW"[..]), "leftovers": leftovers(&directory)});
        if actual != *case { failures.push(format!("{actual}\n  vs {case}")); }
    }
    assert!(failures.is_empty(), "{} of {} durable cases differ:\n{}", failures.len(), cases.len(), failures.join("\n"));
    assert_eq!(cases.len(), 12);
}
