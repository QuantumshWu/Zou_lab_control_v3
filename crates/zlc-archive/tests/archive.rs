//! Figure archives against the pinned scientific code: every archive
//! `write_figure_archive` wrote reads back to the same info, members and
//! native datasets `NativeDataset.from_science` adopts, a reopen-and-save
//! writes the bytes (or the refusal) Python's writes, and every malformed
//! file is refused with Python's class and message.
//!
//! Committed: `tests/data/archive` (`gen/archive_cases.py OUTDIR`). The large
//! set (`gen/archive_cases.py OUTDIR --large`) is checked when
//! `ZLC_ARCHIVE_GOLDEN` names its directory (several separated by `;`).

use std::io::Cursor;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};
use zlc_archive::json::{dumps, py_repr_str, Style};
use zlc_archive::pystr;
use zlc_archive::zip::ZipReader;
use zlc_archive::{read_archive, write_figure_archive, ArchiveDataset, Entry, FigureArchive, Json, NpyData};
use zlc_core::dataset::Validity;

fn load(directory: &Path) -> serde_json::Value {
    let bytes = std::fs::read(directory.join("golden.json.gz")).unwrap_or_else(|error| panic!("{}: {error}", directory.display()));
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn digest(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

fn directories() -> Vec<PathBuf> {
    let mut directories = vec![Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("archive")];
    if let Ok(paths) = std::env::var("ZLC_ARCHIVE_GOLDEN") { directories.extend(paths.split(';').filter(|path| !path.is_empty()).map(PathBuf::from)); }
    directories
}

fn codes(text: &str) -> Vec<u64> { pystr::code_points(text).map(u64::from).collect() }

fn golden_codes(value: &serde_json::Value) -> Vec<u64> { value.as_array().unwrap().iter().map(|code| code.as_u64().unwrap()).collect() }

/// What `read_archive` returned, against the golden description.
fn check_valid(path: &Path, case: &serde_json::Value) -> Vec<String> {
    let mut failures = Vec::new();
    let file = case["file"].as_str().unwrap();
    let archive = match read_archive(path) { Ok(archive) => archive, Err(error) => return vec![format!("{file}: {error}")] };
    let mut fail = |what: String| failures.push(format!("{file}: {what}"));
    if codes(&dumps(&archive.info, Style::Figure).unwrap()) != golden_codes(&case["info_codes"]) { fail("info differs".into()); }
    let expected = case["arrays"].as_array().unwrap();
    if expected.len() != archive.arrays.len() { fail(format!("{} members, expected {}", archive.arrays.len(), expected.len())); }
    for ((name, array), entry) in archive.arrays.iter().zip(expected) {
        if name != entry["name"].as_str().unwrap() { fail(format!("member {name} where {} expected", entry["name"])); continue; }
        if array.dtype.str() != entry["dtype"].as_str().unwrap() { fail(format!("{name} dtype {}", array.dtype.str())); }
        let shape: Vec<usize> = entry["shape"].as_array().unwrap().iter().map(|size| size.as_u64().unwrap() as usize).collect();
        if array.shape != shape { fail(format!("{name} shape {:?}", array.shape)); }
        if entry.get("empty").is_some() {
            if array.dtype.itemsize != 0 { fail(format!("{name} items are not empty")); }
        } else if let (Some(texts), NpyData::Text(units)) = (entry["text_codes"].as_array(), &array.data) {
            let actual: Vec<Vec<u64>> = units.chunks(array.dtype.itemsize / 4).map(|item| {
                let end = item.iter().rposition(|unit| *unit != 0).map_or(0, |last| last + 1);
                item[..end].iter().map(|unit| u64::from(*unit)).collect()
            }).collect();
            if actual != texts.iter().map(golden_codes).collect::<Vec<_>>() { fail(format!("{name} text differs")); }
        } else if let Some(items) = entry["bytes"].as_array() {
            let hex: Vec<String> = (0..array.size()).map(|index| array.bytes_item(index).iter().map(|byte| format!("{byte:02x}")).collect()).collect();
            if hex != items.iter().map(|item| item.as_str().unwrap().to_owned()).collect::<Vec<_>>() { fail(format!("{name} bytes differ")); }
        } else if digest(&array.little_endian_bytes()) != entry["sha256"].as_str().unwrap() {
            fail(format!("{name} values differ"));
        }
    }
    let expected = case["datasets"].as_array().unwrap();
    if expected.len() != archive.datasets.len() { fail(format!("{} datasets, expected {}", archive.datasets.len(), expected.len())); }
    for ((name, dataset), entry) in archive.datasets.iter().zip(expected) {
        if name != entry["name"].as_str().unwrap() { fail(format!("dataset {name}")); continue; }
        let reference = &entry["ref"];
        if dataset.reference.block_id != reference["block_id"].as_str().unwrap() || dataset.reference.stream_generation != reference["stream_generation"].as_str().unwrap()
            || dataset.reference.revision != reference["revision"].as_u64().unwrap() { fail(format!("{name} reference {:?}", dataset.reference)); }
        let tree = |schema: &zlc_core::schema::DatasetSchema| dumps(&zlc_archive::tree::schema_tree(schema), Style::Figure).unwrap();
        if tree(dataset.snapshot.schema()) != entry["schema"].as_str().unwrap() { fail(format!("{name} schema differs")); }
        if dataset.snapshot.window().is_some() { fail(format!("{name} has a window")); }
        let segments: Vec<_> = dataset.snapshot.segments().collect();
        let expected = entry["segments"].as_array().unwrap();
        if segments.len() != expected.len() { fail(format!("{name}: {} segments, expected {}", segments.len(), expected.len())); continue; }
        for (index, (segment, want)) in segments.iter().zip(expected).enumerate() {
            let pair = |value: &serde_json::Value| [value[0].as_u64().unwrap() as usize, value[1].as_u64().unwrap() as usize];
            if segment.origin() != pair(&want["origin"]) || segment.extent() != pair(&want["extent"]) { fail(format!("{name}[{index}] placement")); }
            let data = segment.data();
            if tree(data.schema()) != want["schema"].as_str().unwrap() { fail(format!("{name}[{index}] chunk schema differs")); }
            if digest(data.values().as_bytes()) != want["values"].as_str().unwrap() { fail(format!("{name}[{index}] values differ")); }
            match (data.validity(), &want["validity"]) {
                (Validity::Uniform(flag), serde_json::Value::Bool(expected)) if flag == expected => {}
                (Validity::Mask(mask), serde_json::Value::Object(expected)) if digest(mask.as_bytes()) == expected["sha256"].as_str().unwrap() => {}
                (actual, expected) => fail(format!("{name}[{index}] validity {} vs {expected}", match actual { Validity::Uniform(flag) => flag.to_string(), Validity::Mask(_) => "mask".into() })),
            }
            match (data.sigma(), want["sigma"].as_str()) {
                (None, None) => {}
                (Some(sigma), Some(expected)) if digest(sigma.as_bytes()) == expected => {}
                _ => fail(format!("{name}[{index}] sigma differs")),
            }
        }
    }
    let rewritten = match rewrite(path, &archive) {
        Ok(bytes) => serde_json::json!(digest(&bytes)),
        Err(error) => serde_json::json!([error.kind.python_name(), error.message]),
    };
    if rewritten != case["resave"] { fail(format!("rewrite {rewritten} vs Python's resave {}", case["resave"])); }
    failures
}

/// What a reopen-and-save writes natively: the read datasets as their
/// blocks are (dense or placed) and the plain members, in member order.
fn rewrite(path: &Path, archive: &FigureArchive) -> zlc_archive::Result<Vec<u8>> {
    let section = archive.info.get("sections").unwrap().get("dataset").filter(|value| !value.is_null());
    let manifest = |name: &str| section.and_then(|section| section.get(name)).unwrap();
    let owned: Vec<String> = archive.datasets.iter().flat_map(|(name, _)| zlc_archive::manifest::manifest_array_keys(manifest(name))).collect();
    let order: Vec<String> = ZipReader::open(path)?.names().into_iter().map(str::to_owned).collect();
    let mut entries = Vec::new();
    for member in &order {
        let key = member.strip_suffix(".npy").unwrap_or(member);
        if key == "info" { continue; }
        if let Some((name, dataset)) = archive.datasets.iter().find(|(name, _)| name == key) {
            entries.push((name.clone(), Entry::Dataset(ArchiveDataset { reference: dataset.reference.clone(), source: dataset.source() })));
        } else if !owned.iter().any(|name| name == key) {
            entries.push((key.to_owned(), Entry::Array(archive.array(key).unwrap().clone())));
        }
    }
    let mut sections = archive.info.get("sections").unwrap().clone();
    sections.remove("dataset");
    let name = archive.info.get("name").and_then(Json::as_str).unwrap();
    Ok(write_figure_archive(Cursor::new(Vec::new()), name, &entries, &sections)?.into_inner())
}

/// One golden entry: the description, or the refusal (the missing file's
/// message names this checkout's path).
fn check(directory: &Path, folder: &str, case: &serde_json::Value) -> Vec<String> {
    let file = case["file"].as_str().unwrap();
    let path = directory.join(folder).join(file);
    if let Some(description) = case.get("ok") { return check_valid(&path, description); }
    let mut expected = case["error"].clone();
    if file == "does-not-exist.npz" {
        expected[1] = Json::Str(format!("[Errno 2] No such file or directory: {}", py_repr_str(&path.to_string_lossy()))).to_serde();
    }
    let actual = match read_archive(&path) { Ok(_) => serde_json::Value::Null, Err(error) => serde_json::json!([error.kind.python_name(), error.message]) };
    if actual != expected { vec![format!("{file}: {actual} vs {expected}")] } else { Vec::new() }
}

fn check_all(folder: &str, minimum: usize) {
    let mut failures = Vec::new();
    let mut checked = 0;
    for directory in directories() {
        let golden = load(&directory);
        for case in golden[folder].as_array().unwrap() {
            failures.extend(check(&directory, folder, case));
            checked += 1;
        }
    }
    assert!(failures.is_empty(), "{} of {checked} {folder} archives differ:\n{}", failures.len(), failures.join("\n"));
    assert!(checked >= minimum, "{checked} {folder} archives checked");
}

#[test]
fn python_archives_read_back_and_resave_like_python() { check_all("valid", 33); }

/// Members Python inflates with bz2 or lzma. The native reader has no such
/// decoder (none builds offline, and no zlc writer stores one), so it
/// refuses them as an interpreter built without those modules does.
const UNSUPPORTED: [(&str, &str); 2] = [
    ("targeted/method-bzip2.npz", "Compression requires the (missing) bz2 module"),
    ("targeted/method-lzma.npz", "Compression requires the (missing) lzma module"),
];

/// The hostile and odd archives review found (`tests/data/regression`,
/// `gen/regression_cases.py`): each refused, or read and re-saved, as Python
/// does (`UNSUPPORTED` apart).
#[test]
fn regression_archives_end_like_python() {
    let directory = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests").join("data").join("regression");
    let golden = load(&directory);
    let cases = golden["cases"].as_array().unwrap();
    let failures: Vec<String> = cases.iter().flat_map(|case| {
        let file = case["file"].as_str().unwrap();
        match UNSUPPORTED.iter().find(|(name, _)| *name == file) {
            Some((_, message)) => check(&directory, "", &serde_json::json!({"file": file, "error": ["RuntimeError", message]})),
            None => check(&directory, "", case),
        }
    }).collect();
    assert!(failures.is_empty(), "{} of {} regression archives differ:
{}", failures.len(), cases.len(), failures.join("
"));
    assert!(cases.len() >= 137);
}

#[test]
fn malformed_archives_are_refused_like_python() { check_all("invalid", 150); }

fn apparatus_document(path: &Path) -> serde_json::Value {
    let mut zip = ZipReader::open(path).unwrap();
    let array = zlc_archive::npy::read_array(&mut zip.open_member("info.npy").unwrap()).unwrap();
    serde_json::from_str(&array.text(0).unwrap()).unwrap()
}

fn write_apparatus_document(path: &Path, info: &serde_json::Value) {
    let info = zlc_archive::NpyArray::unicode(vec![info.to_string()], Vec::new());
    let data = zlc_archive::NpyArray::unicode(vec!["sample".into()], vec![1]);
    let mut zip = zlc_archive::zip::ZipWriter::new(std::fs::File::create(path).unwrap()).unwrap();
    for (name, array) in [("info.npy", info), ("data.npy", data)] {
        zip.write_member(name, true, &[&array.header(), &array.data_bytes()]).unwrap();
    }
    zip.finish().unwrap();
}

#[test]
fn shared_apparatus_is_stored_once_and_public_reader_restores_every_record() {
    let path = std::env::temp_dir().join(format!("zlc-shared-apparatus-{}.npz", std::process::id()));
    let shared = serde_json::json!({"context": "capture-a", "taken_at": "2026-10-29T12:03:23Z",
        "devices": {"rf": {"values": {"frequency": 80000000.0}, "units": {"frequency": "Hz"}}}});
    let mut other = shared.clone(); other["context"] = serde_json::json!("capture-b");
    let sections = serde_json::json!({"lineage": {"nodes": [
        {"record": {"apparatus": shared}, "event_record": {"apparatus": shared,
            "acquisition": {"runs": [{"stream":"camera", "generation":"capture-a", "record":{"apparatus":shared}}], "events":[], "ranges":[]}}},
        {"record": {"apparatus": other}},
        {"record": {"apparatus": {"$apparatus": 0}}}], "custom": {"apparatus": {"$apparatus": 99}}},
        "source": {"run_record": {"apparatus": shared}},
        "custom": {"record": {"apparatus": {"$apparatus": 100}}}});
    let original = Json::from_serde(&sections);
    let entries = [("data".to_owned(), Entry::Array(zlc_archive::NpyArray::unicode(vec!["sample".into()], vec![1])))];
    let bytes = write_figure_archive(Cursor::new(Vec::new()), "shared", &entries, &original).unwrap().into_inner();
    std::fs::write(&path, &bytes).unwrap();
    let stored = apparatus_document(&path);
    assert_eq!(stored["apparatus_table"]["schema"], "zlc.figure.apparatus-table/1");
    assert_eq!(stored["apparatus_table"]["records"].as_array().unwrap().len(), 3);
    assert_eq!(stored["sections"]["lineage"]["nodes"][0]["record"]["apparatus"], serde_json::json!({"$apparatus": 0}));
    assert_eq!(stored["sections"]["lineage"]["nodes"][1]["record"]["apparatus"], serde_json::json!({"$apparatus": 1}));
    assert_eq!(stored["sections"]["source"]["run_record"]["apparatus"], serde_json::json!({"$apparatus": 0}));
    assert_eq!(stored["sections"]["lineage"]["nodes"][0]["event_record"]["acquisition"]["runs"][0]["record"]["apparatus"], serde_json::json!({"$apparatus":0}));
    assert_eq!(stored["sections"]["custom"], sections["custom"], "unrelated metadata is literal");
    let decoded = read_archive(&path).unwrap();
    assert!(decoded.info.get("apparatus_table").is_none());
    assert_eq!(dumps(decoded.info.get("sections").unwrap(), Style::Figure).unwrap(), dumps(&original, Style::Figure).unwrap());
    assert_eq!(rewrite(&path, &decoded).unwrap(), bytes, "restored sections encode deterministically");
    // An uncompressed historical file remains literal, including reference-shaped user values.
    write_apparatus_document(&path, &decoded.info.to_serde());
    assert_eq!(read_archive(&path).unwrap().info, decoded.info);
    std::fs::remove_file(path).unwrap();
}

#[test]
fn shared_apparatus_refuses_malformed_tables_and_references() {
    let path = std::env::temp_dir().join(format!("zlc-shared-apparatus-invalid-{}.npz", std::process::id()));
    let document = serde_json::json!({"schema":"zlc.figure", "name":"shared", "members":{"data":{"dtype":"<U6","shape":[1]}},
        "sections":{"source":{"run_record":{"apparatus":{"$apparatus":0}}}},
        "apparatus_table":{"schema":"zlc.figure.apparatus-table/1","records":[{"context":"one"}]}});
    let mut cases = Vec::new();
    for reference in [serde_json::json!({"$apparatus": -1}), serde_json::json!({"$apparatus": 1}),
        serde_json::json!({"$apparatus": true}), serde_json::json!({"$apparatus": 0.0}), serde_json::json!({"$apparatus": "0"})] {
        let mut bad = document.clone(); bad["sections"]["source"]["run_record"]["apparatus"] = reference;
        cases.push((bad, "figure apparatus reference is out of range"));
    }
    let mut bad = document.clone(); bad["sections"]["source"]["run_record"]["apparatus"]["extra"] = true.into();
    cases.push((bad, "figure apparatus reference must contain only $apparatus"));
    let mut bad = document.clone(); bad["apparatus_table"]["schema"] = "unknown".into();
    cases.push((bad, "unsupported figure apparatus table schema"));
    let mut bad = document.clone(); bad["apparatus_table"]["records"] = serde_json::json!([]);
    cases.push((bad, "figure apparatus table records must be a non-empty array"));
    let mut bad = document.clone(); bad["apparatus_table"]["records"] = serde_json::json!([{}, {}]);
    cases.push((bad, "figure apparatus table contains unused records"));
    for (info, expected) in cases {
        write_apparatus_document(&path, &info);
        let error = read_archive(&path).err().expect("malformed apparatus storage was accepted");
        assert_eq!(error.message, expected);
    }
    std::fs::remove_file(path).unwrap();
}

#[test]
fn finite_acquisition_provenance_survives_figure_archive_without_extra_logic_nodes() {
    let path = std::env::temp_dir().join(format!("zlc-acquisition-ranges-{}.npz",std::process::id()));
    let acquired = serde_json::json!({"runs":[{"stream":"scan","generation":"g","record":{"node":"scan"}}],
        "events":[
            {"run":0,"signal":"scan","revision":1,"ordinal":0,"record":{"records":{"scan":{"actual_devices":{"rf":{"frequency":80}}}}},"parents":[]},
            {"run":0,"signal":"scan","revision":2,"ordinal":1,"record":{"records":{"scan":{"actual_devices":{"rf":{"frequency":81}}}}},"parents":[]}],
        "ranges":[{"event":0,"origin":[0,0],"extent":[1,1]},{"event":1,"origin":[0,1],"extent":[1,1]}]});
    let native = Json::from_serde(&serde_json::json!({"publication":{"stream":"scan","generation":"g","signal":"scan","revision":2,"ordinal":1},
        "run":{"node":"scan"},"event":{"records":{"scan":{"actual_devices":{"rf":{"frequency":81}}}}},"parents":[],"acquisition":acquired}));
    let (lineage,source) = zlc_archive::provenance::capture_figure_provenance(Some(&native)).unwrap();
    assert_eq!(lineage.get("nodes").unwrap().as_array().unwrap().len(),1);
    let sections = Json::object([("lineage",lineage),("source",source)]);
    let entries = [("data".to_owned(),Entry::Array(zlc_archive::NpyArray::unicode(vec!["a".into(),"b".into()],vec![2])))];
    std::fs::write(&path,write_figure_archive(Cursor::new(Vec::new()),"scan",&entries,&sections).unwrap().into_inner()).unwrap();
    let decoded = read_archive(&path).unwrap().info.to_serde();
    assert_eq!(decoded["sections"]["lineage"]["nodes"][0]["event_record"]["acquisition"],acquired);
    std::fs::remove_file(path).unwrap();
}
