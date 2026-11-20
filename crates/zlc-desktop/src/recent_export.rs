//! The data the Runtime keeps of recent publications (`Runtime::keep_recent`), as a diagnostics archive
//! carries it, and the budget it is kept within.
//!
//! One `.npy` per signal and run: every kept event of it stacked on a leading axis, in the order it was
//! committed. `data/index.json` says, for each file, the signal, its run (as the journal names it), each
//! event's ordinal and commit time (the journal's clock), its validity and the dataset's schema -- so a
//! line of the journal that names a publication is found here by signal, run and ordinal.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use serde_json::{Value, json};
use zlc_archive::npy::{NpyArray, numeric_from_bytes};
use zlc_core::DType;
use zlc_core::dataset::Validity;
use zlc_core::runtime::{Publication, RecentData, Runtime};

/// The budget a workspace starts with until its operator sets another (MiB).
pub const DEFAULT_BUDGET_MIB: u64 = 256;

fn settings() -> PathBuf { zlc_core::checkout::local("diagnostics.json") }

/// The budget this checkout was last given, else the default.
pub fn saved_budget_mib() -> u64 {
    std::fs::read_to_string(settings()).ok().and_then(|text| serde_json::from_str::<Value>(&text).ok())
        .and_then(|saved| saved["budget_mib"].as_u64()).unwrap_or(DEFAULT_BUDGET_MIB)
}

/// Keep recent data within `mib` from now on (0: none), and remember it for the next start.
pub fn set_budget_mib(runtime: &Runtime, mib: u64) {
    runtime.keep_recent((mib as usize) << 20);
    let path = settings();
    if let Some(folder) = path.parent() { let _ = std::fs::create_dir_all(folder); }
    let _ = std::fs::write(path, json!({"budget_mib": mib}).to_string());
}

/// The `diagnostics` state section: the budget, and how far back the kept data goes (whole minutes; it
/// changes at most once a minute, so a page is not told it at every shot).
pub fn state(runtime: &Runtime) -> Value {
    let (budget, bytes, span) = runtime.recent_held();
    json!({"budget_mib": budget >> 20, "held_minutes": span.map(|seconds| (seconds / 60.0).floor() as u64),
           "full": budget > 0 && bytes.saturating_mul(10) >= budget.saturating_mul(9)})
}

fn file_name(index: usize, signal: &str) -> String {
    let safe: String = signal.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    format!("data/{index:03}-{}.npy", safe.trim_matches('_'))
}

fn npy(bytes: &[u8], dtype: DType, shape: Vec<usize>) -> Vec<u8> {
    let array = NpyArray::numeric(numeric_from_bytes(bytes, dtype, false), shape);
    let mut out = array.header();
    out.extend_from_slice(&array.data_bytes());
    out
}

/// The archive members of everything `recent` kept (taken with `Runtime::recent`): `data/index.json` and the files it names. Empty while
/// nothing is kept.
pub fn members(recent: RecentData) -> Vec<(String, Vec<u8>)> {
    // Per signal and run, in commit order.
    let mut groups: BTreeMap<(String, String), Vec<(u64, Arc<Publication>)>> = BTreeMap::new();
    for (us, publication) in recent.publications {
        if publication.data().is_none() { continue; }
        let key = publication.key();
        groups.entry((key.signal.to_string(), key.generation.as_uuid().simple().to_string())).or_default().push((us, publication));
    }
    let (mut out, mut index) = (Vec::new(), Vec::new());
    for ((signal, generation), events) in groups {
        let first = events[0].1.data().expect("kept with data");
        let (schema, dtype) = (first.schema().clone(), first.values().dtype());
        // One run of a signal has one schema; an event of another shape (it cannot be) is left out and said.
        let same: Vec<&(u64, Arc<Publication>)> = events.iter().filter(|(_, publication)| publication.data().is_some_and(|data| data.schema() == &schema)).collect();
        let mut values = Vec::with_capacity(same.len() * first.values().byte_len());
        for (_, publication) in &same { values.extend_from_slice(publication.data().expect("kept with data").values().as_bytes()); }
        let mut shape = vec![same.len()];
        shape.extend_from_slice(schema.physical_shape());
        let name = file_name(index.len() + 1, &signal);
        out.push((name.clone(), npy(&values, dtype, shape.clone())));
        // Validity: one flag per event while every event is all valid or all invalid, else a mask beside the values.
        let masked = same.iter().any(|(_, publication)| matches!(publication.data().expect("kept").validity(), Validity::Mask(_)));
        let valid = if masked {
            let cells: usize = schema.physical_shape().iter().product();
            let mut mask = Vec::with_capacity(same.len() * cells);
            for (_, publication) in &same {
                match publication.data().expect("kept").validity() {
                    Validity::Uniform(flag) => mask.extend(std::iter::repeat(u8::from(*flag)).take(cells)),
                    Validity::Mask(bits) => mask.extend(bits.as_slice().iter().map(|bit| u8::from(*bit))),
                }
            }
            let file = name.replace(".npy", ".valid.npy");
            out.push((file.clone(), npy(&mask, DType::Bool, shape.clone())));
            json!({"mask": file})
        } else {
            json!(same.iter().map(|(_, publication)| matches!(publication.data().expect("kept").validity(), Validity::Uniform(true))).collect::<Vec<_>>())
        };
        index.push(json!({
            "file": name, "signal": signal, "owner": same[0].1.run().owner.as_ref(), "run": &generation[..8], "generation": generation,
            "events": same.len(), "left_out": events.len() - same.len(), "shape": shape, "valid": valid,
            "ordinals": same.iter().map(|(_, publication)| publication.key().ordinal).collect::<Vec<_>>(),
            "us": same.iter().map(|(us, _)| *us).collect::<Vec<_>>(),
            "schema": zlc_plugin_host::wire::schema_tree(&schema),
        }));
    }
    if index.is_empty() { return out; }
    let about = "Each file holds every kept event of one signal of one run, stacked on its first axis in commit order. `us` is when each \
                 was committed (the journal's clock), `ordinals` the publication ordinals the journal's `runtime commit` lines name, `run` \
                 the run as those lines name it. `valid` is one flag per event, or the file of a mask shaped as the values.";
    out.insert(0, ("data/index.json".into(), serde_json::to_vec_pretty(&json!({"about": about, "budget_mib": recent.budget >> 20, "files": index})).unwrap_or_default()));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use zlc_core::OwnedBuffer;
    use zlc_core::dataset::DataBlock;
    use zlc_core::runtime::{Bundle, Declaration, Input, Storage};
    use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};

    #[test]
    fn kept_events_are_one_stacked_file_per_signal_named_by_run_and_ordinal() {
        let runtime = Runtime::new();
        let scalar = Domain::new(vec![1], vec![], vec![]).unwrap();
        let axis = |name: &str, size: usize| Axis::new(name.into(), name.into(), "coordinate".into(), Coordinates::Index { origin: 0, len: size }, None, None, None, None).unwrap();
        let cells = Domain::dense(vec![2, 3], vec![axis("y", 2), axis("x", 3)]).unwrap();
        let schema = DatasetSchema::new(scalar.clone(), scalar, cells, ValueSchema::new(DType::U16, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap();
        let generation = runtime.begin("camera", vec![Declaration { signal: "@logic/camera/frames".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
        assert!(members(runtime.recent()).is_empty(), "nothing kept, nothing carried");
        runtime.keep_recent(1 << 20);
        for ordinal in 0..3u16 {
            let data = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(vec![ordinal; 6]).into(), Validity::Uniform(ordinal != 1), None).unwrap();
            runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "@logic/camera/frames".into(), ordinal: u64::from(ordinal),
                data, origin: None, shot_time: None }], metadata: json!({}), parents: vec![] }).unwrap();
        }
        let carried = members(runtime.recent());
        let names: Vec<&str> = carried.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["data/index.json", "data/001-logic_camera_frames.npy"]);
        let index: Value = serde_json::from_slice(&carried[0].1).unwrap();
        let file = &index["files"][0];
        assert_eq!((&file["signal"], &file["owner"], &file["events"], &file["ordinals"], &file["valid"]),
                   (&json!("@logic/camera/frames"), &json!("camera"), &json!(3), &json!([0, 1, 2]), &json!([true, false, true])));
        assert_eq!(file["shape"], json!([3, 1, 1, 2, 3]));
        assert_eq!(file["run"].as_str().unwrap(), &generation.id().as_uuid().simple().to_string()[..8], "the run as the journal names it");
        // The file is numpy's: its header, then eighteen little-endian u16 in commit order.
        let npy = &carried[1].1;
        assert!(npy.starts_with(b"\x93NUMPY") && String::from_utf8_lossy(&npy[..120]).contains("'shape': (3, 1, 1, 2, 3)"));
        let values: Vec<u16> = npy[npy.len() - 36..].chunks(2).map(|pair| u16::from_le_bytes([pair[0], pair[1]])).collect();
        assert_eq!(values, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]);
        assert_eq!(state(&runtime)["budget_mib"], 1);
    }
}
