//! The reference scenes are complete requests. `refs/scenes/*.json` are the native requests the app builds for the
//! references R1–R9 (refs/scenes.mjs runs the app's own model builder, src/full-master-model.ts via
//! scripts/native-protocol.mjs); each is read by the crate's one JSON → Request mapping, `Request::from_native`, which
//! ends with `Request::validate` (contract §2.4's request-only checks).
use atom_motion::request::Request;
use serde_json::Value;
use std::path::PathBuf;

/// Every reference scene (R1–R9, R4's TW1/TW2, R8) is a request the crate accepts, its trap beam focused; the set is
/// exactly the scenes refs/scenes.mjs wrote (its manifest).
#[test]
fn every_reference_scene_is_a_valid_request() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes");
    let mut seen = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_str().unwrap().to_owned();
        if !name.ends_with(".json") || name.ends_with(".scene.json") || name == "manifest.json" {
            continue;
        }
        let line: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let r = Request::from_native(&line).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(r.beams[r.trap_beam].waist.is_some(), "{name}");
        seen += 1;
    }
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(seen, manifest["scenes"].as_object().unwrap().len(), "{}", dir.display());
}
