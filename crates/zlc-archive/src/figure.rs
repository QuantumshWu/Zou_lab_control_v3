//! `zlc_data.figure_archive`: a figure's arrays and typed datasets as NPZ
//! members beside one strict JSON metadata tree, written and read exactly as
//! the scientific code does (member order, Deflate choice, refusals).

use std::collections::{HashMap, HashSet};
use std::io::{Read, Seek, Write};
use std::path::Path;

use crate::deflate::zlib_compressed_len;
use crate::error::{ArchiveError, ErrorKind, Result};
use crate::json::{dumps, parse_metadata, py_repr_names, py_repr_str, Json, Style};
use crate::manifest::{exact_keys, manifest_array_keys, materialize, members_of, snapshot_from_manifest, snapshot_manifest, ArchiveDataset, ReadDataset};
use crate::npy::{bytes_scalar, read_array, NpyArray, NpySource, MAGIC};
use crate::tree::stripped;
use crate::zip::{ZipReader, ZipWriter};

pub const FIGURE_SCHEMA: &str = "zlc.figure";
const INFO_KEY: &str = "info";
const PROBE_BYTES: usize = 1 << 20;
const MIN_SAVINGS: f64 = 0.20;
const WRITE_BUFFER: usize = 16 * 1024 * 1024;
/// How deep `_jsonable` recurses, from the sections root (1), before the
/// interpreter's recursion limit stops it (`write_figure_archive` from a script).
const MAX_METADATA_DEPTH: usize = 996;

/// One named figure entry: a plain array or a typed dataset.
pub enum Entry { Array(NpyArray), Dataset(ArchiveDataset) }

/// `_jsonable(sections)`: finite numbers only, and no deeper than Python recurses.
fn check_metadata(value: &Json) -> Result<()> {
    let mut pending: Vec<(&Json, String, usize)> = vec![(value, "metadata".to_owned(), 1)];
    while let Some((value, path, depth)) = pending.pop() {
        if depth > MAX_METADATA_DEPTH { return Err(ArchiveError::new(ErrorKind::Recursion, "maximum recursion depth exceeded")); }
        match value {
            Json::Float(real) if !real.is_finite() => return Err(ArchiveError::type_error(format!("{path} contains a non-finite metadata number"))),
            Json::Array(items) => pending.extend(items.iter().enumerate().rev().map(|(index, item)| (item, format!("{path}[{index}]"), depth + 1))),
            Json::Object(pairs) => pending.extend(pairs.iter().rev().map(|(key, item)| (item, format!("{path}.{key}"), depth + 1))),
            _ => {}
        }
    }
    Ok(())
}

/// `_compression_sample(array)` of the member's stored bytes.
fn compression_sample(data: &[u8]) -> Vec<u8> {
    let wanted = data.len().min(PROBE_BYTES);
    if data.len() <= wanted { return data.to_vec(); }
    let chunk = (wanted / 3).max(1);
    let last = data.len() - chunk;
    [0, last / 2, last].iter().flat_map(|start| data[*start..*start + chunk].iter().copied()).collect()
}

/// `_member_compression(array)`: whole-member Deflate when it earns it.
fn deflates(data: &[u8]) -> bool {
    if data.len() <= PROBE_BYTES { return true; }
    let sample = compression_sample(data);
    sample.is_empty() || zlib_compressed_len(&sample) as f64 <= sample.len() as f64 * (1.0 - MIN_SAVINGS)
}

/// One member's `write` calls: the header, then numpy's 16 MiB data chunks.
fn write_member<W: Write + Seek>(zip: &mut ZipWriter<W>, name: &str, array: &NpyArray) -> std::io::Result<()> {
    let header = array.header();
    let data = array.data_bytes();
    let itemsize = array.dtype.itemsize;
    let step = if itemsize == 0 { usize::MAX } else { (WRITE_BUFFER / itemsize).max(1) * itemsize };
    let mut parts: Vec<&[u8]> = vec![&header];
    if itemsize > 0 { parts.extend(data.chunks(step)); }
    zip.write_member(&format!("{name}.npy"), deflates(&data), &parts)
}

/// `write_figure_archive(stream, name, arrays=..., sections=...)`.
pub fn write_figure_archive<W: Write + Seek>(stream: W, name: &str, arrays: &[(String, Entry)], sections: &Json) -> Result<W> {
    if name.is_empty() || stripped(name) { return Err(ArchiveError::value("figure name must be canonical non-empty text")); }
    if arrays.is_empty() { return Err(ArchiveError::value("a figure archive must carry at least one array")); }
    if sections.as_object().is_none() { return Err(ArchiveError::type_error("figure sections must be a metadata dict")); }
    let materialized: Vec<Option<_>> = arrays.iter().map(|(_, entry)| match entry { Entry::Dataset(dataset) => Some(materialize(&dataset.source)), Entry::Array(_) => None }).collect();
    let mut owners: HashMap<String, &str> = HashMap::new();
    for ((key, _), planes) in arrays.iter().zip(&materialized) {
        if key.is_empty() { return Err(ArchiveError::type_error("figure array names must be non-empty text")); }
        if key == INFO_KEY { return Err(ArchiveError::value(format!("{} is reserved for the metadata document", py_repr_str(INFO_KEY)))); }
        let claimed = match planes { Some(planes) => members_of(key, planes), None => vec![key.clone()] };
        for member in claimed {
            let previous = *owners.entry(member.clone()).or_insert(key);
            if previous != key {
                return Err(ArchiveError::value(format!("figure member name collision for {}: claimed by {} and {}", py_repr_str(&member), py_repr_str(previous), py_repr_str(key))));
            }
        }
    }
    let mut stored: Vec<(String, NpyArray)> = Vec::new();
    let mut datasets: Vec<(String, Json)> = Vec::new();
    for ((key, entry), planes) in arrays.iter().zip(materialized) {
        match (entry, planes) {
            (Entry::Dataset(dataset), Some(planes)) => datasets.push((key.clone(), snapshot_manifest(key, &dataset.reference, planes, &mut stored))),
            (Entry::Array(array), _) => stored.push((key.clone(), array.clone())),
            _ => unreachable!(),
        }
    }
    check_metadata(sections)?;
    let mut sections = sections.clone();
    if !datasets.is_empty() {
        if sections.get("dataset").is_some() { return Err(ArchiveError::value("the dataset section is written from the snapshots themselves")); }
        sections.insert("dataset", Json::Object(datasets));
    }
    let members = Json::Object(stored.iter().map(|(key, array)| (key.clone(), Json::object([
        ("dtype", Json::Str(array.dtype.str())),
        ("shape", Json::Array(array.shape.iter().map(|size| Json::Int(*size as i128)).collect())),
    ]))).collect());
    let info = dumps(&Json::object([("schema", Json::str(FIGURE_SCHEMA)), ("name", Json::str(name)), ("members", members), ("sections", sections)]), Style::Figure)?;
    let io = |error: std::io::Error| ArchiveError::os(&error, None);
    let mut zip = ZipWriter::new(stream).map_err(io)?;
    write_member(&mut zip, INFO_KEY, &NpyArray::unicode(vec![info], Vec::new())).map_err(io)?;
    for (key, array) in &stored { write_member(&mut zip, key, array).map_err(io)?; }
    zip.finish().map_err(io)
}

/// `atomic_write_file(path, lambda stream: write_figure_archive(stream, ...))`:
/// the archive at its resolved path, durably published.
pub fn write_figure_file(path: &Path, name: &str, arrays: &[(String, Entry)], sections: &Json) -> Result<std::path::PathBuf> {
    crate::durable::atomic_write_file(path, |file| {
        write_figure_archive(std::io::BufWriter::with_capacity(1 << 20, file), name, arrays, sections)?
            .into_inner().map_err(|error| ArchiveError::os(error.error(), None))
    })
}

// ------------------------------------------------------------- reading

/// An archive's members by name, in metadata order.
pub struct Members { entries: Vec<(String, NpyArray)>, index: HashMap<String, usize> }

impl Members {
    pub fn new(entries: Vec<(String, NpyArray)>) -> Self {
        let index = entries.iter().enumerate().map(|(position, (name, _))| (name.clone(), position)).collect();
        Self { entries, index }
    }
    pub fn get(&self, name: &str) -> Option<&NpyArray> { self.index.get(name).map(|position| &self.entries[*position].1) }
    pub fn iter(&self) -> std::slice::Iter<'_, (String, NpyArray)> { self.entries.iter() }
    pub fn len(&self) -> usize { self.entries.len() }
    pub fn is_empty(&self) -> bool { self.entries.is_empty() }
}

/// What `read_archive` returns: the metadata, every member, and the typed datasets.
pub struct FigureArchive { pub info: Json, pub arrays: Members, pub datasets: Vec<(String, ReadDataset)> }

impl FigureArchive {
    pub fn array(&self, name: &str) -> Option<&NpyArray> { self.arrays.get(name) }
    pub fn dataset(&self, name: &str) -> Option<&ReadDataset> { self.datasets.iter().find(|(key, _)| key == name).map(|(_, dataset)| dataset) }
}

/// `NpzFile[name]`: a peek at the magic, then the NPY member, or its bytes.
fn load_member(zip: &mut ZipReader, physical: &str) -> Result<NpyArray> {
    let magic = zip.open_member(physical)?.read(MAGIC.len())?;
    if magic.as_slice() != MAGIC { return Ok(bytes_scalar(zip.read(physical)?)); }
    let mut member = zip.open_member(physical)?;
    read_array(&mut member)
}

/// `_parse_info(array)` then `_validate_current_info(info)`.
fn parse_info(array: &NpyArray) -> Result<Json> {
    if !array.shape.is_empty() || array.dtype.kind != crate::npdtype::Kind::Unicode {
        return Err(ArchiveError::value("figure info must be one scalar Unicode JSON document"));
    }
    let info = parse_metadata(&array.text(0)?)?;
    if info.as_object().is_none() { return Err(ArchiveError::value("figure metadata root must be an object")); }
    exact_keys(&info, &["schema", "name", "members", "sections"], "figure metadata")?;
    let schema = info.get("schema").unwrap();
    if schema.as_str() != Some(FIGURE_SCHEMA) { return Err(ArchiveError::value(format!("unsupported figure format {}", crate::json::py_repr(schema)))); }
    match info.get("name").unwrap() { Json::Str(name) if !name.is_empty() && !stripped(name) => {} _ => return Err(ArchiveError::value("figure metadata name must be non-empty text")) }
    if info.get("sections").unwrap().as_object().is_none() { return Err(ArchiveError::value("figure metadata sections must be an object")); }
    let members = match info.get("members").unwrap() { Json::Object(pairs) if !pairs.is_empty() => pairs, _ => return Err(ArchiveError::value("figure metadata members must be a non-empty object")) };
    for (key, descriptor) in members {
        if key.is_empty() || key == INFO_KEY { return Err(ArchiveError::value("figure member names must be non-empty, non-reserved text")); }
        if descriptor.as_object().is_none() { return Err(ArchiveError::value(format!("figure member {} descriptor must be an object", py_repr_str(key)))); }
        exact_keys(descriptor, &["dtype", "shape"], &format!("figure member {}", py_repr_str(key)))?;
        let Json::Str(dtype) = descriptor.get("dtype").unwrap() else { return Err(ArchiveError::value(format!("figure member {} dtype must be text", py_repr_str(key)))) };
        let parsed = crate::npdtype::from_str(dtype).map_err(|error| match error.kind {
            ErrorKind::Type => ArchiveError::value(format!("figure member {} dtype is invalid", py_repr_str(key))),
            _ => error,
        })?;
        if parsed.has_object() || parsed.has_fields() || parsed.str() != *dtype {
            return Err(ArchiveError::value(format!("figure member {} dtype is not canonical", py_repr_str(key))));
        }
        let valid_shape = matches!(descriptor.get("shape").unwrap(), Json::Array(sizes)
            if sizes.iter().all(|size| matches!(size, Json::Int(value) if *value >= 0) || matches!(size, Json::BigInt(text) if !text.starts_with('-'))));
        if !valid_shape { return Err(ArchiveError::value(format!("figure member {} shape must be nonnegative integers", py_repr_str(key)))); }
    }
    Ok(info)
}

/// A shape tuple's `repr` from its Python ints.
fn tuple_repr(sizes: &[String]) -> String {
    match sizes.len() { 1 => format!("({},)", sizes[0]), _ => format!("({})", sizes.join(", ")) }
}

/// A plain `.npy` file `np.load` reads through `numpy.fromfile`.
struct FileReader { file: std::fs::File }

impl NpySource for FileReader {
    fn read(&mut self, n: usize) -> Result<Vec<u8>> {
        let mut out = Vec::new();
        (&mut self.file).take(n as u64).read_to_end(&mut out).map_err(|error| ArchiveError::os(&error, None))?;
        Ok(out)
    }
    fn is_file(&self) -> bool { true }
}

/// `read_archive(path)`: read once, returning metadata, members and the typed datasets.
pub fn read_archive(path: &Path) -> Result<FigureArchive> {
    let display = path.to_string_lossy().into_owned();
    let mut file = std::fs::File::open(path).map_err(|error| ArchiveError::io(&error, &display))?;
    let mut magic = Vec::with_capacity(6);
    (&mut file).take(6).read_to_end(&mut magic).map_err(|error| ArchiveError::os(&error, None))?;
    if magic.is_empty() { return Err(ArchiveError::new(ErrorKind::Eof, "No data left in file")); }
    file.seek(std::io::SeekFrom::Start(0)).map_err(|error| ArchiveError::os(&error, None))?;
    if !(magic.starts_with(b"PK\x03\x04") || magic.starts_with(b"PK\x05\x06")) {
        if magic == MAGIC {
            read_array(&mut FileReader { file })?;
            return Err(ArchiveError::type_error("'numpy.ndarray' object does not support the context manager protocol"));
        }
        return Err(ArchiveError::value("This file contains pickled (object) data. If you trust the file you can load it unsafely using the `allow_pickle=` keyword argument or `pickle.load()`."));
    }
    let mut zip = ZipReader::from_file(file)?;
    let zip_names: Vec<String> = zip.names().into_iter().map(str::to_owned).collect();
    if zip_names.iter().any(|name| !name.ends_with(".npy")) { return Err(ArchiveError::value("figure NPZ may contain only NPY members")); }
    let names: Vec<&str> = zip_names.iter().map(|name| &name[..name.len() - 4]).collect();
    let mut counts: HashMap<&str, usize> = HashMap::with_capacity(names.len());
    for name in &names { *counts.entry(name).or_insert(0) += 1; }
    let duplicates: Vec<&str> = counts.iter().filter(|(_, count)| **count > 1).map(|(name, _)| *name).collect();
    if !duplicates.is_empty() { return Err(ArchiveError::value(format!("figure contains duplicate NPZ members {}", py_repr_names(&duplicates)))); }
    if !counts.contains_key(INFO_KEY) { return Err(ArchiveError::value(format!("figure carries no {INFO_KEY} document"))); }
    let info = parse_info(&load_member(&mut zip, &format!("{INFO_KEY}.npy"))?)?;
    let members = info.get("members").unwrap().as_object().unwrap();
    let declared: HashSet<&str> = members.iter().map(|(key, _)| key.as_str()).collect();
    let missing: Vec<&str> = declared.iter().copied().filter(|key| !counts.contains_key(key)).collect();
    let extra: Vec<&str> = names.iter().copied().filter(|name| *name != INFO_KEY && !declared.contains(name)).collect();
    if !missing.is_empty() || !extra.is_empty() {
        return Err(ArchiveError::value(format!("figure NPZ members mismatch; missing={}, extra={}", py_repr_names(&missing), py_repr_names(&extra))));
    }
    let mut arrays = Vec::with_capacity(members.len());
    for (key, _) in members { arrays.push((key.clone(), load_member(&mut zip, &format!("{key}.npy"))?)); }
    for ((key, descriptor), (_, array)) in members.iter().zip(&arrays) {
        let expected: Vec<String> = descriptor.get("shape").unwrap().as_array().unwrap().iter().map(crate::json::py_repr).collect();
        let actual: Vec<String> = array.shape.iter().map(usize::to_string).collect();
        if actual != expected {
            return Err(ArchiveError::value(format!("figure member {} shape {} does not match metadata {}", py_repr_str(key), tuple_repr(&actual), tuple_repr(&expected))));
        }
        let dtype = descriptor.get("dtype").unwrap().as_str().unwrap();
        if array.dtype.str() != dtype {
            return Err(ArchiveError::value(format!("figure member {} dtype {} does not match metadata {}", py_repr_str(key), py_repr_str(&array.dtype.str()), py_repr_str(dtype))));
        }
    }
    let mut arrays = Members::new(arrays);
    let datasets = read_datasets(&info, &arrays)?;
    // A dataset's members cross into C-ordered buffers (`np.frombuffer(array.tobytes())`).
    for (name, _) in &datasets {
        let manifest = info.get("sections").unwrap().get("dataset").unwrap().get(name).unwrap();
        for member in manifest_array_keys(manifest) {
            if let Some(position) = arrays.index.get(&member) { arrays.entries[*position].1.fortran = false; }
        }
    }
    Ok(FigureArchive { info, arrays, datasets })
}

/// `_read_datasets(info, arrays)`.
fn read_datasets(info: &Json, arrays: &Members) -> Result<Vec<(String, ReadDataset)>> {
    let Some(raw) = info.get("sections").unwrap().get("dataset") else { return Ok(Vec::new()) };
    if raw.is_null() { return Ok(Vec::new()); }
    let Json::Object(entries) = raw else { return Err(ArchiveError::value("figure dataset section must be an object")) };
    let mut referenced: HashSet<String> = HashSet::new();
    let mut datasets = Vec::with_capacity(entries.len());
    for (name, manifest) in entries {
        if name.is_empty() { return Err(ArchiveError::value("figure dataset names must be non-empty text")); }
        if manifest.as_object().is_none() || manifest.get("values_key").and_then(Json::as_str) != Some(name.as_str()) {
            return Err(ArchiveError::value(format!("figure dataset {} must own its same-named array", py_repr_str(name))));
        }
        let keys = manifest_array_keys(manifest);
        if !keys.contains(name) || keys.iter().any(|member| member != name && !member.starts_with(&format!("{name}."))) {
            return Err(ArchiveError::value(format!("figure dataset {} uses non-canonical member names", py_repr_str(name))));
        }
        let duplicate: HashSet<&str> = keys.iter().filter(|key| referenced.contains(*key)).map(String::as_str).collect();
        if !duplicate.is_empty() {
            let duplicate: Vec<&str> = duplicate.into_iter().collect();
            return Err(ArchiveError::value(format!("figure datasets contain duplicate array reference {}", py_repr_names(&duplicate))));
        }
        referenced.extend(keys.iter().cloned());
        if let Some(member) = keys.iter().find(|member| arrays.get(member).is_none()) {
            return Err(ArchiveError::npz(format!("missing array {}", py_repr_str(member))));
        }
        datasets.push((name.clone(), snapshot_from_manifest(manifest, arrays)?));
    }
    Ok(datasets)
}
