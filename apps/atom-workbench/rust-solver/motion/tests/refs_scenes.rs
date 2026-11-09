//! The reference scenes are complete requests. `refs/scenes/*.json` are the native requests the app builds for the
//! references R1–R9 (refs/scenes.mjs runs the app's own model builder, src/full-master-model.ts via
//! scripts/native-protocol.mjs); each maps onto `request::Request` field by field (refs/lib/contract.mjs lists the map)
//! and passes `Request::validate`, contract §2.4's request-only checks.
//!
//! The JSON → Request map below is this test's own until the native adapter's parser exists (the solve agent's); the
//! test then reads the scenes through that parser and this map goes.
use atom_motion::linalg::{C64, Mat};
use atom_motion::request::{Beam, OutsideForm, Request, STATES};
use serde_json::Value;
use std::path::PathBuf;

fn matrix(v: &Value) -> Mat {
    let part = |key: &str| -> Vec<f64> { v[key].as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect() };
    let (re, im) = (part("real"), part("imag"));
    assert_eq!((re.len(), im.len()), (STATES * STATES, STATES * STATES));
    Mat::from_fn(STATES, STATES, |i, j| C64::new(re[i * STATES + j], im[i * STATES + j]))
}
fn numbers(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}
fn indices(v: &Value) -> Vec<i32> {
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap() as i32).collect()
}

/// The request a native line describes (contract §1.3–1.4): the times are the app's grid, linspace(0, durationUs,
/// samples).
fn request(line: &Value) -> Request {
    let (model, motion) = (&line["model"], &line["model"]["motion"]);
    let beams: Vec<Beam> = motion["beams"]
        .as_array()
        .unwrap()
        .iter()
        .map(|b| Beam {
            name: b["name"].as_str().unwrap().into(),
            direction: std::array::from_fn(|a| b["direction"][a].as_f64().unwrap()),
            wave_number: b["waveNumberPerUm"].as_f64().unwrap(),
            waist: b["waistUm"].as_f64(),
            frequency_index: indices(&b["frequencyIndex"]),
            coupling: matrix(&b["coupling"]),
            counter_coupling: matrix(&b["counterCoupling"]),
            field: std::array::from_fn(|q| C64::new(b["electricField"][q]["re"].as_f64().unwrap(), b["electricField"][q]["im"].as_f64().unwrap())),
        })
        .collect();
    let trap = motion["trapBeamId"].as_str().unwrap();
    let ids: Vec<&str> = motion["beams"].as_array().unwrap().iter().map(|b| b["id"].as_str().unwrap()).collect();
    let (duration, samples) = (line["options"]["durationUs"].as_f64().unwrap(), line["options"]["samples"].as_u64().unwrap() as usize);
    Request {
        states: model["states"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().into()).collect(),
        hamiltonian: matrix(&motion["hamiltonian"]),
        frame_frequencies_mhz: numbers(&motion["frameFrequenciesMHz"]),
        frequencies_mhz: numbers(&model["frequencyCoordinates"]["frequenciesMHz"]),
        beams,
        outside_forms: motion["outsideForms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|f| OutsideForm { frequency_index: indices(&f["frequencyIndex"]), matrices: std::array::from_fn(|k| matrix(&f["matrices"][k])) })
            .collect(),
        trap_beam: ids.iter().position(|&id| id == trap).expect("trapBeamId names a beam"),
        emission: std::array::from_fn(|a| matrix(&motion["emission"][a])),
        mass_kg: motion["massKg"].as_f64().unwrap(),
        temperature_uk: motion["temperatureUK"].as_f64().unwrap(),
        initial_density: matrix(&serde_json::json!({"real": model["initialReal"], "imag": model["initialImag"]})),
        times_us: (0..samples).map(|i| duration * i as f64 / (samples - 1) as f64).collect(),
    }
}

/// Every reference scene (R1–R9, R4's TW1/TW2, R8) is a request the crate accepts: all fields present with the
/// contract's shapes, `Request::validate` passes, and the scene keeps what its reference reads (one trap beam with a
/// waist; the stated initial state).
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
        let r = request(&line);
        assert_eq!(r.validate(), Ok(()), "{name}");
        assert!(r.beams[r.trap_beam].waist.is_some(), "{name}");
        seen += 1;
    }
    // Every scene refs/scenes.mjs wrote (its manifest) and none other.
    let manifest: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("manifest.json")).unwrap()).unwrap();
    assert_eq!(seen, manifest["scenes"].as_object().unwrap().len(), "{}", dir.display());
}
