//! What remote clients read without the window's turn: the workspace's files
//! (confined to the workspace, v2 `lab_files`/`lab_artifact`), a published
//! signal's numbers (v2 `lab_signal`) and what plots can do (v2 `lab_catalog`'s
//! plot section). Each answer comes from the file system, the Runtime or the
//! plot crate's own tables; nothing here changes the instrument.
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Value, json};
use zlc_bus::mcp::ToolOutput;
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::restrict::Source;
use zlc_core::runtime::{Read, Runtime, Storage};
use zlc_core::schema::{Coordinate, DatasetSchema, DomainRole, ValidityContract};
use zlc_core::AnyBuffer;

fn output(json: Value) -> ToolOutput { ToolOutput { json, png: None } }

/// `path` inside the workspace `root` (relative paths are read against it).
fn inside(root: &Path, path: &str) -> Result<PathBuf, String> {
    let root = root.canonicalize().map_err(|error| format!("The workspace {} cannot be read: {error}", root.display()))?;
    let chosen = Path::new(path);
    let joined = if chosen.is_absolute() { chosen.to_path_buf() } else { root.join(chosen) };
    let resolved = joined.canonicalize().map_err(|_| format!("No file or folder {path:?} in the workspace"))?;
    if !resolved.starts_with(&root) { return Err("Remote file access is limited to this experiment's workspace".into()); }
    Ok(resolved)
}

fn relative(root: &Path, path: &Path) -> String {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    path.strip_prefix(&root).unwrap_or(path).to_string_lossy().replace('\\', "/")
}

fn page(value: &Value, name: &str, default: u64, range: std::ops::RangeInclusive<u64>) -> Result<usize, String> {
    let chosen = value[name].as_u64().unwrap_or(default);
    if !range.contains(&chosen) { return Err(format!("{name} must be from {} to {}", range.start(), range.end())); }
    Ok(chosen as usize)
}

pub fn list_files(root: &Path, params: &Value) -> Result<ToolOutput, String> {
    let directory = params["directory"].as_str().unwrap_or(".");
    let offset = params["offset"].as_u64().unwrap_or(0) as usize;
    let limit = page(params, "limit", 100, 1..=200)?;
    let folder = inside(root, directory)?;
    if !folder.is_dir() { return Err(format!("{directory:?} is a file; read it with read_file")); }
    let mut entries: Vec<_> = std::fs::read_dir(&folder).map_err(|error| error.to_string())?.flatten().collect();
    entries.sort_by_key(|entry| entry.file_name().to_string_lossy().to_lowercase());
    let total = entries.len();
    let rows: Vec<Value> = entries.iter().skip(offset).take(limit).filter_map(|entry| {
        let metadata = entry.metadata().ok()?;
        let modified = metadata.modified().ok().and_then(|time| time.duration_since(std::time::UNIX_EPOCH).ok()).map(|age| age.as_secs());
        Some(json!({"path": relative(root, &entry.path()), "directory": metadata.is_dir(), "bytes": metadata.len(), "modified_unix_seconds": modified}))
    }).collect();
    Ok(output(json!({"directory": relative(root, &folder), "entries": rows, "offset": offset,
        "next_offset": (offset + limit < total).then_some(offset + limit), "total": total})))
}

/// The longest prefix of `bytes` (at most `limit`) that ends on a whole UTF-8 character.
fn utf8_page(bytes: &[u8], limit: usize) -> Result<&str, String> {
    let mut end = bytes.len().min(limit);
    while end > 0 && std::str::from_utf8(&bytes[..end]).is_err() { end -= 1; }
    if end == 0 && !bytes.is_empty() { return Err("limit is too small for the next UTF-8 character".into()); }
    Ok(std::str::from_utf8(&bytes[..end]).expect("checked"))
}

/// Large metadata kept readable: deep objects as their keys, long lists as their count and first rows.
fn compact(value: &Value, depth: usize) -> Value {
    match value {
        Value::Object(object) if depth >= 8 => json!({"keys": object.keys().collect::<Vec<_>>()}),
        Value::Object(object) => Value::Object(object.iter().map(|(key, item)| (key.clone(), compact(item, depth + 1))).collect()),
        Value::Array(items) if items.len() > 40 => json!({"count": items.len(), "first": items.iter().take(8).map(|item| compact(item, depth + 1)).collect::<Vec<_>>()}),
        Value::Array(items) => Value::Array(items.iter().map(|item| compact(item, depth + 1)).collect()),
        other => other.clone(),
    }
}

const TEXT: &[&str] = &["json", "txt", "md", "log", "csv", "tsv"];

pub fn read_file(root: &Path, params: &Value) -> Result<ToolOutput, String> {
    let path = params["path"].as_str().ok_or("path is required")?;
    let offset = params["offset"].as_u64().unwrap_or(0) as usize;
    let limit = page(params, "limit", 16000, 1..=64000)?;
    let file = inside(root, path)?;
    if file.is_dir() { return Err(format!("{path:?} is a folder; list it with list_files")); }
    let size = file.metadata().map_err(|error| error.to_string())?.len();
    let shown = relative(root, &file);
    let suffix = file.extension().map(|suffix| suffix.to_string_lossy().to_lowercase()).unwrap_or_default();
    match suffix.as_str() {
        "npz" => npz(&file, &shown, size, offset, limit, params["section"].as_str()),
        "png" | "jpg" | "jpeg" => {
            if size > 16 << 20 { return Err("The picture exceeds 16 MiB".into()); }
            let bytes = std::fs::read(&file).map_err(|error| error.to_string())?;
            Ok(ToolOutput { json: json!({"path": shown, "bytes": size}), png: Some(bytes) })
        }
        _ if TEXT.contains(&suffix.as_str()) => {
            use std::io::{Read as _, Seek, SeekFrom};
            let mut stream = std::fs::File::open(&file).map_err(|error| error.to_string())?;
            stream.seek(SeekFrom::Start(offset as u64)).map_err(|error| error.to_string())?;
            let mut raw = Vec::with_capacity(limit);
            stream.take(limit as u64).read_to_end(&mut raw).map_err(|error| error.to_string())?;
            let text = utf8_page(&raw, limit)?;
            let end = offset + text.len();
            Ok(output(json!({"path": shown, "offset": offset, "next_offset": ((end as u64) < size).then_some(end),
                "total_bytes": size, "text": text})))
        }
        _ => Ok(output(json!({"path": shown, "bytes": size, "note": "A binary file; read_file reads text, pictures and .npz archives"}))),
    }
}

fn npz(file: &Path, shown: &str, size: u64, offset: usize, limit: usize, section: Option<&str>) -> Result<ToolOutput, String> {
    let mut summary = match zlc_archive::read_archive(file) {
        Ok(archive) => {
            let info = archive.info.to_serde();
            let names: Vec<String> = info["sections"].as_object().map(|sections| sections.keys().cloned().collect()).unwrap_or_default();
            let sections = match section {
                Some(name) => json!({name: info["sections"].get(name).ok_or_else(|| format!("No section {name:?}; sections: {}", names.join(", ")))?}),
                None => info["sections"].clone(),
            };
            let datasets: Map<String, Value> = archive.datasets.iter().map(|(key, dataset)| {
                let schema = dataset.snapshot.schema();
                (key.clone(), json!({"shape": schema.physical_shape(), "dtype": dtype(schema), "acquired_cells": dataset.snapshot.written_cells(),
                    "total_cells": dataset.snapshot.total_cells(), "schema": compact(&zlc_archive::tree::schema_tree(schema).to_serde(), 0),
                    "ref": {"block_id": dataset.reference.block_id, "stream_generation": dataset.reference.stream_generation, "revision": dataset.reference.revision}}))
            }).collect();
            json!({"path": shown, "format": info["schema"], "name": info["name"], "section_names": names, "sections": compact(&sections, 0),
                   "datasets": datasets, "array_members": members(archive.arrays.iter().map(|(name, array)| (name.as_str(), array)))})
        }
        // Another NumPy archive (a Science Context, a calibration's arrays): its members.
        Err(figure) => {
            if size > 256 << 20 { return Err(format!("Not a Figure archive ({figure}), and too large to list")); }
            let mut zip = zlc_archive::zip::ZipReader::open(file).map_err(|error| format!("Not a NumPy archive: {error}"))?;
            let names: Vec<String> = zip.names().into_iter().map(str::to_owned).collect();
            let mut arrays = Vec::new();
            for name in names {
                let mut member = zip.open_member(&name).map_err(|error| error.to_string())?;
                let array = zlc_archive::npy::read_array(&mut member).map_err(|error| format!("{name}: {error}"))?;
                arrays.push((name.trim_end_matches(".npy").to_owned(), array));
            }
            json!({"path": shown, "format": "npz", "note": format!("Not a Figure archive: {figure}"),
                   "array_members": members(arrays.iter().map(|(name, array)| (name.as_str(), array)))})
        }
    };
    let encoded = serde_json::to_string(&summary).map_err(|error| error.to_string())?;
    if offset > 0 || encoded.len() > limit {
        let bytes = encoded.as_bytes();
        let start = offset.min(bytes.len());
        let text = utf8_page(&bytes[start..], limit)?;
        let end = start + text.len();
        summary = json!({"path": shown, "format": summary["format"], "name": summary["name"], "section_names": summary["section_names"],
            "offset": start, "next_offset": (end < bytes.len()).then_some(end), "total_bytes": bytes.len(), "text": text,
            "note": "This summary is longer than limit: its JSON text a page at a time (or ask for one section)."});
    }
    Ok(output(summary))
}

fn members<'a>(arrays: impl Iterator<Item = (&'a str, &'a zlc_archive::NpyArray)>) -> Value {
    Value::Object(arrays.map(|(name, array)| {
        let mut row = json!({"shape": array.shape, "dtype": array.dtype.str()});
        // A short text member (metadata) is shown as written.
        if let (true, Ok(text)) = (array.shape.is_empty(), array.text(0)) {
            if text.len() <= 2000 { row["text"] = json!(text); }
        }
        (name.to_owned(), row)
    }).collect())
}

fn dtype(schema: &DatasetSchema) -> String { format!("{:?}", schema.dtype()).to_lowercase() }

fn half(bits: u16) -> f64 {
    let (sign, exponent, fraction) = (if bits >> 15 == 1 { -1.0 } else { 1.0 }, (bits >> 10) & 0x1f, f64::from(bits & 0x3ff));
    sign * match exponent {
        0 => fraction * 2f64.powi(-24),
        31 => if fraction == 0.0 { f64::INFINITY } else { f64::NAN },
        _ => (1.0 + fraction / 1024.0) * 2f64.powi(i32::from(exponent) - 15),
    }
}

/// One stored value as JSON: non-finite numbers are null, complex is [real, imaginary].
fn value_at(buffer: &AnyBuffer, index: usize) -> Value {
    let number = |value: f64| if value.is_finite() { json!(value) } else { Value::Null };
    match buffer {
        AnyBuffer::Bool(v) => json!(v.as_slice()[index]),
        AnyBuffer::U8(v) => json!(v.as_slice()[index]), AnyBuffer::I8(v) => json!(v.as_slice()[index]),
        AnyBuffer::U16(v) => json!(v.as_slice()[index]), AnyBuffer::I16(v) => json!(v.as_slice()[index]),
        AnyBuffer::U32(v) => json!(v.as_slice()[index]), AnyBuffer::I32(v) => json!(v.as_slice()[index]),
        AnyBuffer::U64(v) => json!(v.as_slice()[index]), AnyBuffer::I64(v) => json!(v.as_slice()[index]),
        AnyBuffer::F16(v) => number(half(v.as_slice()[index].to_bits())),
        AnyBuffer::F32(v) => number(f64::from(v.as_slice()[index])), AnyBuffer::F64(v) => number(v.as_slice()[index]),
        AnyBuffer::C64(v) => { let c = v.as_slice()[index]; json!([number(f64::from(c.re)), number(f64::from(c.im))]) }
        AnyBuffer::C128(v) => { let c = v.as_slice()[index]; json!([number(c.re), number(c.im)]) }
    }
}

fn coordinate(value: Option<Coordinate>) -> Value {
    match value {
        Some(Coordinate::Integer(value)) => i64::try_from(value).map_or_else(|_| json!(value.to_string()), |value| json!(value)),
        Some(Coordinate::Real(value)) if value.is_finite() => json!(value),
        Some(Coordinate::Text(text)) => json!(text.as_ref()),
        _ => Value::Null,
    }
}

/// The validity of one value of `block` at its own flat index (`rp` its Repeat × Point row, `cell` its Cell index).
fn valid_at(block: &DataBlock, local: usize, rp: usize, cell: usize) -> bool {
    match block.validity() {
        Validity::Uniform(flag) => *flag,
        Validity::Mask(mask) => match block.schema().value().validity() {
            ValidityContract::Value => mask.as_slice()[local],
            ValidityContract::Components(ids) => {
                let domain = block.schema().domain(DomainRole::Cell);
                let (mut index, mut stride) = (0, 1);
                for id in ids.iter().rev() {
                    let length = domain.axis(id).map_or(1, |axis| axis.len());
                    index += domain.mapping(id).and_then(|mapping| mapping.code(cell)).unwrap_or(0) * stride;
                    stride *= length;
                }
                mask.as_slice()[rp * stride + index]
            }
        },
    }
}

pub fn read_signal(runtime: &Runtime, params: &Value) -> Result<ToolOutput, String> {
    let name = params["signal"].as_str().ok_or("signal is required")?;
    let offset = params["offset"].as_u64().unwrap_or(0) as usize;
    let limit = page(params, "limit", 256, 1..=4096)?;
    let dataset = params["dataset"] == true;
    // The live run's signal, else the newest that published.
    let candidates: Vec<_> = runtime.signals().into_iter().filter(|signal| signal.signal.as_ref() == name && signal.published).collect();
    let chosen = match params["generation"].as_str() {
        Some(generation) => candidates.into_iter().find(|signal| signal.generation.id().as_uuid().to_string() == generation)
            .ok_or("This signal's run is no longer that generation; read the current one before using this result")?,
        None => candidates.iter().position(|signal| !signal.sealed).or(candidates.len().checked_sub(1))
            .map(|index| candidates.into_iter().nth(index).expect("index"))
            .ok_or_else(|| format!("{name} has not published (list_signals names the signals)"))?,
    };
    let latest = runtime.read_latest(&chosen.generation, name).map_err(|error| format!("{name}: {error}"))?;
    let read = if !dataset { Read::from_event(latest.clone()).map_err(|error| format!("{name}: {error}"))? }
        else if latest.canonical().is_some() { Read::from_canonical(latest.clone()) }
        else if chosen.retained_events > 1 {
            Read::from_window(runtime.read_window(&chosen.generation, name, chosen.retained_events).map_err(|error| format!("{name}: {error}"))?)
        } else if latest.storage() == Storage::Latest {
            return Err(format!("{name} keeps no history: a panel reading a window of it retains one; read its latest publication without dataset"));
        } else { Read::from_event(latest.clone()).map_err(|error| format!("{name}: {error}"))? };
    let source = Source::of_read(&read)?;
    let schema = source.schema().clone();
    let (repeats, points, cells) = (schema.domain(DomainRole::Repeat).size(), schema.domain(DomainRole::Point).size(), schema.domain(DomainRole::Cell).size());
    let total = repeats * points * cells;
    let end = (offset + limit).min(total);
    let flat: Vec<usize> = (offset.min(total)..end).collect();
    // Where each Repeat × Point row of this page is stored: (block, its origin, its points).
    let segments: Vec<([usize; 2], Arc<DataBlock>)> = match &source {
        Source::Dense(block) => vec![([0, 0], block.clone())],
        Source::Placed { segments, .. } => segments.clone(),
    };
    let rows: std::collections::BTreeSet<usize> = flat.iter().map(|index| index / cells).collect();
    let mut stored: HashMap<usize, usize> = HashMap::new();
    for (position, (origin, block)) in segments.iter().enumerate() {
        let extent = [block.schema().domain(DomainRole::Repeat).size(), block.schema().domain(DomainRole::Point).size()];
        for row in rows.range(origin[0] * points..(origin[0] + extent[0]) * points) {
            let p = row % points;
            if p >= origin[1] && p < origin[1] + extent[1] { stored.insert(*row, position); }
        }
    }
    let (mut values, mut valid, mut sigma, mut holes) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let mut any_sigma = false;
    for &index in &flat {
        let (row, cell) = (index / cells, index % cells);
        match stored.get(&row) {
            Some(&position) => {
                let (origin, block) = &segments[position];
                let block_points = block.schema().domain(DomainRole::Point).size();
                let local_row = (row / points - origin[0]) * block_points + (row % points - origin[1]);
                let local = local_row * cells + cell;
                values.push(value_at(block.values(), local));
                valid.push(valid_at(block, local, local_row, cell));
                sigma.push(block.sigma().map_or(Value::Null, |errors| { any_sigma = true; let s = errors.as_slice()[local]; if s.is_finite() { json!(s) } else { Value::Null } }));
            }
            None => { values.push(Value::Null); valid.push(false); sigma.push(Value::Null); holes.push(index); }
        }
    }
    let physical = schema.physical_shape();
    let unravel = |mut index: usize| -> Vec<usize> {
        let mut at = vec![0; physical.len()];
        for (slot, size) in physical.iter().enumerate().rev() { at[slot] = index % size.max(&1); index /= size.max(&1); }
        at
    };
    let mut described = Map::new();
    for (key, role) in [("repeat", DomainRole::Repeat), ("point", DomainRole::Point), ("cell", DomainRole::Cell)] {
        let domain = schema.domain(role);
        let within = |index: usize| match role { DomainRole::Repeat => index / cells / points, DomainRole::Point => index / cells % points, DomainRole::Cell => index % cells };
        let axes: Vec<Value> = domain.axes().iter().map(|axis| {
            let codes: Vec<Option<usize>> = flat.iter().map(|index| domain.mapping(axis.id()).and_then(|mapping| mapping.code(within(*index)))).collect();
            json!({"axis_id": axis.id(), "name": axis.name(), "role": axis.role(), "unit": axis.unit(), "size": axis.len(),
                   "coordinate_frame": axis.coordinate_frame(), "coordinate_of": axis.coordinate_of(), "sample_codes": codes,
                   "sample_coordinates": codes.iter().map(|code| coordinate(code.and_then(|code| axis.coordinate(code)))).collect::<Vec<_>>(),
                   "sample_labels": axis.labels().map(|labels| codes.iter().map(|code| code.map(|code| labels[code].as_ref())).collect::<Vec<_>>())})
        }).collect();
        described.insert(key.into(), json!({"shape": domain.physical_shape(), "axes": axes}));
    }
    let value = schema.value();
    described.insert("value".into(), json!({"dtype": dtype(&schema), "unit": value.unit(), "name": value.name(),
        "validity": match value.validity() { ValidityContract::Value => json!("value"), ValidityContract::Components(ids) => json!({"components": ids.iter().map(|id| id.as_ref()).collect::<Vec<_>>()}) }}));
    let key = read.key();
    Ok(output(json!({"signal": name, "generation": key.generation.as_uuid().to_string(), "revision": key.revision, "ordinal": key.ordinal,
        "scope": if dataset { "dataset" } else { "publication" }, "sealed": chosen.sealed, "shape": physical, "schema": described,
        "offset": offset, "returned": flat.len(), "total": total, "next_offset": (end < total).then_some(end),
        "flat_indices": flat, "physical_indices": flat.iter().map(|index| unravel(*index)).collect::<Vec<_>>(),
        "values": values, "valid": valid, "sigma": if any_sigma { Value::Array(sigma) } else { Value::Null },
        "holes": holes, "nonfinite_encoding": "null"})))
}

pub fn plot_catalog(params: &Value) -> Result<ToolOutput, String> {
    use zlc_plot::configuration::{semantics, spec, HANDLERS};
    let details = params["details"] == true;
    let kinds: Vec<Value> = HANDLERS.iter().filter(|handler| handler.kind.value() != "pulse_timeline").map(|handler| json!({
        "kind": handler.kind.value(), "label": handler.display_name, "semantic_fields": handler.semantic_fields,
        "fit_target": handler.fit_target.map(|target| target.value()), "record_input": handler.record_input})).collect();
    let models: Vec<Value> = zlc_plot::fit::builtin_fit_models().iter().map(|model| {
        let mut row = json!({"model_id": model.model_id, "label": model.display_name, "targets": model.targets.iter().map(|target| target.value()).collect::<Vec<_>>(),
                             "headline": model.headline, "parameters": model.parameters.iter().map(|parameter| parameter.name.clone()).collect::<Vec<_>>()});
        if details {
            row["formula"] = json!(model.formula);
            row["parameters"] = Value::Array(model.parameters.iter().map(|parameter| json!({"name": parameter.name, "symbol": parameter.symbol,
                "label": parameter.display_label, "domain": format!("{:?}", parameter.domain), "unit": format!("{:?}", parameter.unit_relation)})).collect());
        }
        row
    }).collect();
    Ok(output(json!({"kinds": kinds, "facet_cell_kinds": spec::GRID_CELL_KINDS.iter().map(|kind| kind.value()).collect::<Vec<_>>(),
        "axis_roles": spec::Role::ALL.iter().map(|role| role.name()).collect::<Vec<_>>(),
        "other_fates": [semantics::FATE_REDUCE, semantics::FATE_POOL, "a scope pin (the semantic control's choices)"],
        "reductions": zlc_plot::kinds::Reduction::ALL.iter().map(|reduction| reduction.value()).collect::<Vec<_>>(),
        "sizes": zlc_plot::layout::PANEL_PRESETS.iter().map(|preset| preset.0).collect::<Vec<_>>(), "fit_models": models,
        "bound": "A panel's own controls (get_state {panel}) list the axes, choices and compatible fit models of its signal."})))
}
