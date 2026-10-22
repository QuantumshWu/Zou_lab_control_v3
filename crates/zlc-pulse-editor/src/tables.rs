//! Scan tables as files and as text: the Scan page's view of a table
//! (`_scan_table_text`), Open array (`.npy`, `.csv`, `.txt`, `_read_text_table`)
//! and Save array (a named `.npy`, one field per column, as v2 writes it).
use std::path::{Path, PathBuf};

use serde_json::Value;
use zlc_archive::pyliteral::{Obj, literal_eval};
use zlc_pulse::error::repr;
use zlc_pulse::scan::{ScanColumnSpec, ScanTable};
use zlc_pulse::{Error, Kind, Result};

use crate::args::py_float;
use crate::text::counted;

fn value_error(message: impl Into<String>) -> Error { Error::new(Kind::Value, message) }

fn right(text: &str, width: usize) -> String {
    let length = text.chars().count();
    format!("{}{text}", " ".repeat(width.saturating_sub(length)))
}

/// `_scan_table_text`: a header, the units, then the first 40 rows.
pub fn table_text(rows: &[Vec<f64>], columns: &[ScanColumnSpec]) -> String {
    let mut lines = Vec::new();
    if !columns.is_empty() {
        lines.push(columns.iter().map(|c| right(if c.label.is_empty() { &c.name } else { &c.label }, 14)).collect::<Vec<_>>().join("  "));
        lines.push(columns.iter().map(|c| right(&c.unit, 14)).collect::<Vec<_>>().join("  "));
    }
    for row in rows.iter().take(40) {
        lines.push(row.iter().zip(columns).map(|(value, spec)| {
            if spec.is_dac { right(&(value.round_ties_even() as i128).to_string(), 14) } else { right(&zlc_core::pyfmt::format_g(*value, 6), 14) }
        }).collect::<Vec<_>>().join("  "));
    }
    if rows.len() > 40 { lines.push(format!("... {}", counted(rows.len() - 40, "more point"))); }
    lines.join("\n")
}

/// `_read_text_table`: comma-separated rows, `#` lines and blanks skipped,
/// the first other line naming the columns when it is not numbers.
pub fn read_text_table(path: &Path) -> Result<ScanTable> {
    let text = std::fs::read_to_string(path).map_err(|error| Error::new(Kind::Io, format!("{}: {error}", path.display())))?;
    let text = text.strip_prefix('\u{feff}').unwrap_or(&text);
    let mut lines: Vec<&str> = text.lines().filter(|line| !line.trim().is_empty() && !line.trim_start().starts_with('#')).collect();
    let mut names: Vec<String> = lines.first().map(|line| line.split(',').map(|cell| cell.trim().to_owned()).collect()).unwrap_or_default();
    if names.iter().all(|cell| py_float(&Value::String(cell.clone())).is_ok()) { names.clear(); } else { lines.remove(0); }
    if lines.is_empty() { return Err(value_error("the file holds no scan points")); }
    let mut data: Vec<Vec<f64>> = Vec::new();
    for (row, line) in lines.iter().enumerate() {
        let values = line.split(',').enumerate().map(|(column, cell)| py_float(&Value::String(cell.trim().to_owned()))
            .map_err(|_| value_error(format!("could not convert string {} to float64 at row {row}, column {}.", repr(cell.trim()), column + 1))))
            .collect::<Result<Vec<f64>>>()?;
        if let Some(first) = data.first() {
            if first.len() != values.len() {
                return Err(value_error(format!("the number of columns changed from {} to {} at row {}; use `usecols` to select a subset and avoid this error",
                                               first.len(), values.len(), row + 1)));
            }
        }
        data.push(values);
    }
    if names.is_empty() { return Ok(ScanTable::Rows(data)); }
    for (position, name) in names.iter().enumerate() {
        if name.is_empty() { return Err(value_error(format!("column {} of the header has no name", position + 1))); }
        if names.iter().filter(|other| *other == name).count() > 1 { return Err(value_error(format!("the header names {} more than once", repr(name)))); }
    }
    let width = data[0].len();
    if names.len() != width { return Err(value_error(format!("the header names {} but each row holds {width}", counted(names.len(), "column")))); }
    Ok(ScanTable::Columns(names.into_iter().enumerate().map(|(index, name)| (name, data.iter().map(|row| row[index]).collect())).collect()))
}

fn scalar(bytes: &[u8], descr: &str) -> Result<f64> {
    let little = !descr.starts_with('>');
    let code = descr.trim_start_matches(['<', '>', '=', '|']);
    let take = |n: usize| -> [u8; 8] {
        let mut raw = [0u8; 8];
        let mut part = bytes[..n].to_vec();
        if !little { part.reverse(); }
        raw[..n].copy_from_slice(&part);
        raw
    };
    Ok(match code {
        "f8" => f64::from_le_bytes(take(8)),
        "f4" => f32::from_le_bytes(take(4)[..4].try_into().expect("4")) as f64,
        "i8" => i64::from_le_bytes(take(8)) as f64,
        "i4" => i32::from_le_bytes(take(4)[..4].try_into().expect("4")) as f64,
        "i2" => i16::from_le_bytes(take(2)[..2].try_into().expect("2")) as f64,
        "i1" => bytes[0] as i8 as f64,
        "u8" => u64::from_le_bytes(take(8)) as f64,
        "u4" => u32::from_le_bytes(take(4)[..4].try_into().expect("4")) as f64,
        "u2" => u16::from_le_bytes(take(2)[..2].try_into().expect("2")) as f64,
        "u1" | "b1" => bytes[0] as f64,
        other => return Err(value_error(format!("a scan array holds numbers, not {}", repr(other)))),
    })
}

fn size_of(descr: &str) -> Result<usize> {
    descr.trim_start_matches(['<', '>', '=', '|']).get(1..).and_then(|digits| digits.parse().ok())
        .ok_or_else(|| value_error(format!("unsupported scan array dtype {}", repr(descr))))
}

/// A `.npy` scan table: a named (structured) array gives each column its
/// field; a plain one is rows, a 1-D one a single field's points.
pub fn read_npy_table(path: &Path, slots: usize) -> Result<ScanTable> {
    let bytes = std::fs::read(path).map_err(|error| Error::new(Kind::Io, format!("{}: {error}", path.display())))?;
    if bytes.len() < 10 || &bytes[..6] != b"\x93NUMPY" { return Err(value_error("not a NumPy .npy file")); }
    let (length, start) = match bytes[6] {
        1 => (u16::from_le_bytes([bytes[8], bytes[9]]) as usize, 10),
        2 | 3 => (u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize, 12),
        version => return Err(value_error(format!("unsupported .npy version {version}"))),
    };
    let header = std::str::from_utf8(bytes.get(start..start + length).ok_or_else(|| value_error("truncated .npy header"))?)
        .map_err(|_| value_error("the .npy header is not text"))?;
    let Obj::Dict(items) = literal_eval(header.trim()).map_err(|error| value_error(format!("bad .npy header: {error:?}")))? else {
        return Err(value_error("bad .npy header"));
    };
    let get = |key: &str| items.iter().find(|(name, _)| matches!(name, Obj::Str(text) if text == key)).map(|(_, value)| value);
    if matches!(get("fortran_order"), Some(Obj::Bool(true))) { return Err(value_error("a Fortran-ordered scan array is not supported; save it in C order")); }
    let shape: Vec<usize> = match get("shape") {
        Some(Obj::Tuple(dims)) => dims.iter().map(|d| match d { Obj::Int(n) => n.as_i128().map(|v| v as usize).ok_or_else(|| value_error("bad shape")), _ => Err(value_error("bad shape")) })
            .collect::<Result<_>>()?,
        _ => return Err(value_error("bad .npy header: no shape")),
    };
    let data = &bytes[start + length..];
    match get("descr") {
        Some(Obj::Str(descr)) => {
            let size = size_of(descr)?;
            let count: usize = shape.iter().product();
            if data.len() < count * size { return Err(value_error("the .npy file is not fully written")); }
            let values = (0..count).map(|i| scalar(&data[i * size..], descr)).collect::<Result<Vec<f64>>>()?;
            match shape.as_slice() {
                [rows, columns] => Ok(ScanTable::Rows(values.chunks(*columns).take(*rows).map(<[f64]>::to_vec).collect())),
                [points] if slots == 1 => Ok(ScanTable::Rows(values.iter().take(*points).map(|v| vec![*v]).collect())),
                _ => Err(value_error("a scan array has one row per point and one column per Scan field")),
            }
        }
        Some(Obj::List(fields)) => {
            let mut layout = Vec::new();
            for field in fields {
                let Obj::Tuple(parts) = field else { return Err(value_error("bad structured .npy dtype")) };
                match parts.as_slice() {
                    [Obj::Str(name), Obj::Str(descr), ..] => layout.push((name.clone(), descr.clone(), size_of(descr)?)),
                    _ => return Err(value_error("bad structured .npy dtype")),
                }
            }
            let record: usize = layout.iter().map(|(_, _, size)| size).sum();
            let count: usize = shape.iter().product();
            if data.len() < count * record { return Err(value_error("the .npy file is not fully written")); }
            let mut columns: Vec<(String, Vec<f64>)> = layout.iter().map(|(name, _, _)| (name.clone(), Vec::with_capacity(count))).collect();
            for index in 0..count {
                let mut offset = index * record;
                for (column, (_, descr, size)) in layout.iter().enumerate() {
                    columns[column].1.push(scalar(&data[offset..], descr)?);
                    offset += size;
                }
            }
            Ok(ScanTable::Columns(columns))
        }
        _ => Err(value_error("bad .npy header: no descr")),
    }
}

pub fn read_table(path: &Path, slots: usize) -> Result<ScanTable> {
    if path.extension().is_some_and(|e| e.eq_ignore_ascii_case("npy")) { read_npy_table(path, slots) } else { read_text_table(path) }
}

/// Save array: one float64 field per column the table drives (`array[field_id]`).
pub fn write_named_npy(path: &Path, fields: &[(String, String)], rows: &[Vec<f64>]) -> Result<()> {
    let descr = fields.iter().map(|(field, _)| format!("({}, '<f8')", repr(field))).collect::<Vec<_>>().join(", ");
    let mut header = format!("{{'descr': [{descr}], 'fortran_order': False, 'shape': ({},), }}", rows.len());
    let (prefix, version) = if header.len() + 11 < 65536 { (10, 1u8) } else { (12, 2u8) };
    let total = prefix + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut bytes = b"\x93NUMPY".to_vec();
    bytes.extend([version, 0]);
    if version == 1 { bytes.extend((header.len() as u16).to_le_bytes()); } else { bytes.extend((header.len() as u32).to_le_bytes()); }
    bytes.extend(header.as_bytes());
    for row in rows { for value in row { bytes.extend(value.to_le_bytes()); } }
    if let Some(parent) = path.parent() { let _ = std::fs::create_dir_all(parent); }
    zlc_pulse::codec::atomic_write(path, &bytes)
}

/// `unique_path(folder, stem, suffix)`'s name: the stem made safe, then
/// `stem.suffix`, `stem-2.suffix`, ... the first that does not exist.
pub fn unique_path(folder: &Path, stem: &str, suffix: &str) -> PathBuf {
    let mut safe = String::new();
    let mut previous_unsafe = false;
    for character in stem.chars() {
        let unsafe_char = (character as u32) < 0x20 || character as u32 == 0x7f || "<>:\"/\\|?*".contains(character) || character.is_whitespace();
        if unsafe_char { if !previous_unsafe { safe.push('-'); } previous_unsafe = true; } else { safe.push(character); previous_unsafe = false; }
    }
    let mut safe = safe.trim_matches([' ', '.', '-']).to_owned();
    if safe.is_empty() { safe = "untitled".into(); }
    let upper = safe.to_ascii_uppercase();
    let base = upper.split('.').next().unwrap_or("");
    if ["CON", "PRN", "AUX", "NUL"].contains(&base) || ((base.starts_with("COM") || base.starts_with("LPT")) && base.len() == 4
        && base.as_bytes()[3].is_ascii_digit() && base.as_bytes()[3] != b'0') { safe = format!("_{safe}"); }
    let mut ordinal = 1;
    loop {
        let name = if ordinal == 1 { format!("{safe}{suffix}") } else { format!("{safe}-{ordinal}{suffix}") };
        let candidate = folder.join(name);
        if !candidate.exists() { return candidate; }
        ordinal += 1;
    }
}
