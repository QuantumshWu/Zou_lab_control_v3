//! The scan model: one `(points x slots)` table whose columns advance together.
//! Each column knows its unit, its legal range and how it reaches the wire
//! (`wire = value * scale + offset`), so the one conversion has one description.
use std::collections::BTreeMap;

use zlc_core::pyfmt::format_g;

use crate::binding::{field_label, normalize_binding_values, pulse_field_value, replace_pulse_field, resolve_subpulse_sources, ticks_per};
use crate::error::{Error, Kind, Result, value_error};
use crate::model::{FieldKind, FieldRef, MAXIMUM_REPEAT_COUNT, Sequence, Source};
use crate::num::Num;
use crate::wire::StreamerParams;

/// One column: its variable name, a starting range that suits its kind, the
/// limits the board accepts and its wire conversion.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanColumnSpec {
    pub name: String,
    pub lo: f64,
    pub hi: f64,
    pub is_dac: bool,
    pub unit: String,
    pub limit_lo: f64,
    pub limit_hi: f64,
    pub wire_scale: f64,
    pub wire_offset: f64,
    pub label: String,
}

impl ScanColumnSpec {
    #[allow(clippy::too_many_arguments)]
    fn new(name: &str, lo: f64, hi: f64, is_dac: bool, unit: &str, limit_lo: f64, limit_hi: f64, wire_scale: f64, wire_offset: f64,
           label: &str) -> Result<ScanColumnSpec> {
        if name.trim().is_empty() { return value_error("scan column name must be non-empty"); }
        if label.trim().is_empty() { return value_error("scan column label must be non-empty text"); }
        if hi <= lo { return value_error(format!("scan column {name} needs hi > lo")); }
        Ok(ScanColumnSpec { name: name.into(), lo, hi, is_dac, unit: unit.into(), limit_lo, limit_hi, wire_scale, wire_offset, label: label.into() })
    }
}

fn py_max(a: f64, b: f64) -> f64 { if b > a { b } else { a } }
fn py_min(a: f64, b: f64) -> f64 { if b < a { b } else { a } }

fn column_for_field(sequence: &Sequence, name: &str, reference: &FieldRef, unit: &str, geometry: &StreamerParams, api: bool) -> Result<ScanColumnSpec> {
    let label = field_label(sequence, reference)?;
    match reference.kind {
        FieldKind::BracketCount => {
            let nominal = pulse_field_value(sequence, reference, unit)?;
            let maximum = MAXIMUM_REPEAT_COUNT as f64;
            ScanColumnSpec::new(name, py_max(1.0, nominal - 1.0), py_min(maximum, nominal + 1.0), false, "count", 1.0, maximum, 1.0, 0.0, &label)
        }
        FieldKind::Dac => {
            let (low, high) = sequence.target.port(reference.port.as_deref().unwrap_or("")).and_then(|port| port.signed_range()).expect("a DAC");
            ScanColumnSpec::new(name, low as f64, high as f64, true, "DAC code (0 = 0 V)", low as f64, high as f64, 1.0, -(low as f64), &label)
        }
        _ => {
            let per = ticks_per(sequence, unit)?;
            let quantum = 1.0 / per;
            if api && reference.kind == FieldKind::Delay {
                let nominal = pulse_field_value(sequence, reference, unit)?;
                let longest = geometry.ttl_delay_max_ticks as f64 / per;
                let span = py_max(quantum, nominal.abs() * 0.5);
                return ScanColumnSpec::new(name, py_max(-longest, nominal - span), py_min(longest, nominal + span), false, unit, -longest, longest,
                                           per, 0.0, &label);
            }
            let nominal = pulse_field_value(sequence, reference, unit)?.abs();
            let longest = ((1u128 << geometry.tick_width) - 1) as f64 / per;
            ScanColumnSpec::new(name, py_max(quantum, nominal * 0.5), py_min(longest, py_max(2.0 * quantum, nominal * 1.5)), false, unit,
                                quantum, longest, per, 0.0, &label)
        }
    }
}

/// The hardware-scan columns declared by this sequence, in slot order.
pub fn scan_columns_for(sequence: &Sequence, params: &StreamerParams) -> Result<Vec<ScanColumnSpec>> {
    let sequence = resolve_subpulse_sources(sequence, None)?;
    sequence.scan_bindings().iter().map(|slot| column_for_field(&sequence, &slot.field_id(), &slot.field_ref, &slot.unit, params, false)).collect()
}

/// The API-value columns declared by this sequence, in parameter order.
pub fn api_parameter_columns_for(sequence: &Sequence) -> Result<Vec<ScanColumnSpec>> {
    let sequence = resolve_subpulse_sources(sequence, None)?;
    let geometry = StreamerParams::default();
    sequence.api_bindings().iter().map(|parameter| column_for_field(&sequence, &parameter.field_id(), &parameter.field_ref, &parameter.unit, &geometry, true)).collect()
}

/// `resolve_scan_point(sequence, values)`: one scan row baked into the fields
/// its slots point at, with no slots left; `None` resolves the authored values.
pub fn resolve_scan_point(sequence: &Sequence, values: Option<&[f64]>) -> Result<Sequence> {
    let sequence = resolve_subpulse_sources(sequence, None)?;
    let slots: Vec<_> = sequence.scan_bindings().into_iter().cloned().collect();
    let row: Vec<f64> = match values {
        None => slots.iter().map(|slot| pulse_field_value(&sequence, &slot.field_ref, &slot.unit)).collect::<Result<_>>()?,
        Some(values) => values.to_vec(),
    };
    if row.len() != slots.len() {
        return value_error(format!("a scan point has one value per slot: {} slot(s), {} value(s)", slots.len(), row.len()));
    }
    if row.is_empty() { return Ok(sequence); }
    let mut result = sequence;
    for (slot, value) in slots.iter().zip(&row) {
        result = replace_pulse_field(&result, &slot.field_ref, crate::num::number_for(*value), &slot.unit, &slot.field_id())?;
    }
    let explicit = values.is_some();
    result.with(|parts| for binding in &mut parts.bindings {
        if binding.scan {
            binding.scan = false;
            if explicit { binding.source = Source::Default; binding.config_key.clear(); }
        }
    })
}

/// Python `int(round(x))`: half to even.
fn round_int(value: f64) -> i128 { value.round_ties_even() as i128 }

/// `scan_rows_to_wire(rows, columns)`: author's numbers to what each slot holds.
pub fn scan_rows_to_wire(rows: &[Vec<f64>], columns: &[ScanColumnSpec]) -> Result<Vec<Vec<i128>>> {
    let converted: Vec<Vec<i128>> = rows.iter().map(|row| row.iter().zip(columns).map(|(value, spec)| round_int(value * spec.wire_scale + spec.wire_offset)).collect()).collect();
    for row in &converted {
        for (value, spec) in row.iter().zip(columns) {
            let low = round_int(spec.limit_lo * spec.wire_scale + spec.wire_offset);
            let high = round_int(spec.limit_hi * spec.wire_scale + spec.wire_offset);
            if !(low <= *value && *value <= high) {
                return value_error(format!("{}: this scan needs wire value {value}, outside the board's range [{low} .. {high}]", spec.name));
            }
        }
    }
    Ok(converted)
}

/// `scan_rows_from_wire(rows, columns)`: what a board holds, in the author's units.
pub fn scan_rows_from_wire(rows: &[Vec<i128>], columns: &[ScanColumnSpec]) -> Vec<Vec<f64>> {
    rows.iter().map(|row| row.iter().zip(columns).map(|(value, spec)| (*value as f64 - spec.wire_offset) / spec.wire_scale).collect()).collect()
}

fn next_up(value: f64) -> f64 { if value.is_nan() || value == f64::INFINITY { value } else if value == 0.0 { f64::from_bits(1) } else if value > 0.0 { f64::from_bits(value.to_bits() + 1) } else { f64::from_bits(value.to_bits() - 1) } }

/// `validate_scan_table(rows, columns)`: one table in the author's units, or a
/// refusal. DAC columns come back as whole codes, durations keep their fraction.
pub fn validate_scan_table(rows: &[Vec<f64>], columns: &[ScanColumnSpec]) -> Result<Vec<Vec<f64>>> {
    if columns.is_empty() { return value_error("a scan table requires at least one bound field"); }
    let width = rows.first().map_or(0, Vec::len);
    if rows.iter().any(|row| row.len() != width) { return value_error("a scan table is two-dimensional: one row per point"); }
    if width != columns.len() {
        return value_error(format!("scan table has {width} column(s) but {} field(s) are bound", columns.len()));
    }
    if rows.is_empty() { return value_error("a scan table needs at least one point"); }
    if rows.iter().flatten().any(|value| !value.is_finite()) { return value_error("a scan table cannot hold a non-finite value"); }
    for (index, spec) in columns.iter().enumerate() {
        let tolerance = if spec.is_dac { 0.0 } else { next_up(0.5 / spec.wire_scale.abs()) };
        let (lower, upper) = (spec.limit_lo - tolerance, spec.limit_hi + tolerance);
        let column: Vec<f64> = rows.iter().map(|row| row[index]).collect();
        if column.iter().any(|value| *value < lower || *value > upper) {
            let least = column.iter().copied().fold(f64::INFINITY, f64::min);
            let most = column.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            return value_error(format!("{}: {} must be within [{} .. {}], and this table asks for {} .. {}", spec.name, spec.unit,
                                       format_g(spec.limit_lo, 6), format_g(spec.limit_hi, 6), format_g(least, 6), format_g(most, 6)));
        }
    }
    Ok(rows.iter().map(|row| row.iter().zip(columns).map(|(value, spec)| if spec.is_dac { value.round_ties_even() } else { *value }).collect()).collect())
}

/// `prepare_scan_application(sequence, rows, params=)`: the table as the board
/// will play it (`effective`, author's units after the wire's rounding) and
/// as the wire holds it.
pub fn prepare_scan_application(sequence: &Sequence, rows: &[Vec<f64>], params: &StreamerParams) -> Result<(Vec<Vec<f64>>, Vec<Vec<i128>>)> {
    let columns = scan_columns_for(sequence, params)?;
    let authored = validate_scan_table(rows, &columns)?;
    let wire = scan_rows_to_wire(&authored, &columns)?;
    let effective = scan_rows_from_wire(&wire, &columns);
    validate_scan_table(&effective, &columns)?;
    let mut played: BTreeMap<Vec<u64>, &Vec<f64>> = BTreeMap::new();
    for (requested, row) in authored.iter().zip(&effective) {
        let key: Vec<u64> = row.iter().map(|value| (value + 0.0).to_bits()).collect();
        let previous = played.entry(key).or_insert(requested);
        if *previous != requested {
            return value_error("two distinct scan points become identical at the required hardware time resolution; reduce the point \
                                count or narrow the duration range");
        }
    }
    Ok((effective, wire))
}

/// A scan table as its author gives it: one column per Scan field in slot
/// order, or the columns of the Scan fields it drives, by id or label.
pub enum ScanTable { Rows(Vec<Vec<f64>>), Columns(Vec<(String, Vec<f64>)>) }

/// `scan_table_for(sequence, table)`: the pulse with Scan on the fields the
/// table drives only, and the validated rows.
pub fn scan_table_for(sequence: &Sequence, table: &ScanTable) -> Result<(Sequence, Vec<Vec<f64>>)> {
    let sequence = resolve_subpulse_sources(sequence, None)?;
    let (sequence, rows) = match table {
        ScanTable::Rows(rows) => (sequence, rows.clone()),
        ScanTable::Columns(columns) => {
            let named = normalize_binding_values(&sequence, columns, "scan")?;
            if named.is_empty() { return value_error("the scan table names no Scan field"); }
            let driven = sequence.with(|parts| for binding in &mut parts.bindings {
                if binding.scan && !named.iter().any(|(key, _)| *key == binding.field_id()) { binding.scan = false; }
            })?;
            let mut data = Vec::new();
            for binding in driven.scan_bindings() {
                data.push(named.iter().find(|(key, _)| *key == binding.field_id()).map(|(_, column)| column.clone()).expect("named"));
            }
            if data.iter().map(Vec::len).collect::<std::collections::BTreeSet<_>>().len() != 1 {
                return value_error("every column of the scan table must have the same number of points");
            }
            let rows = (0..data[0].len()).map(|row| data.iter().map(|column| column[row]).collect()).collect();
            (driven, rows)
        }
    };
    let columns = scan_columns_for(&sequence, &StreamerParams::default())?;
    let rows = validate_scan_table(&rows, &columns)?;
    Ok((sequence, rows))
}

/// `scan_table_template(kind, columns)`: starter Python that builds one table
/// (`columns` per field, or a `grid` of every combination).
pub fn scan_table_template(kind: &str, columns: &[ScanColumnSpec]) -> Result<String> {
    if columns.is_empty() { return value_error("a scan template needs at least one bound field; bind one first"); }
    let g = |value: f64| format_g(value, 6);
    let sweep = |spec: &ScanColumnSpec, size: &str| {
        let base = format!("np.linspace({}, {}, {size})", g(spec.lo), g(spec.hi));
        if spec.is_dac { format!("{base}.round().astype(int)") } else { base }
    };
    let note = |spec: &ScanColumnSpec| {
        let legal = format!("{} .. {}", g(spec.limit_lo), g(spec.limit_hi));
        if spec.is_dac { format!("{}: {}, {legal}", spec.label, spec.unit) } else { format!("{}: duration in the binding's {} ({legal})", spec.label, spec.unit) }
    };
    let table = |values: &[String]| {
        let mut lines = vec!["scan_table = {".to_owned()];
        lines.extend(columns.iter().zip(values).map(|(spec, value)| format!("    \"{}\": {value},", spec.name)));
        lines.push("}".into());
        lines
    };
    let count = columns.len();
    if kind == "grid" {
        let sizes: Vec<usize> = [5, 4, 3].into_iter().chain(std::iter::repeat(2)).take(count.max(3)).collect();
        let mut lines: Vec<String> = ["import numpy as np", "", "# Grid scan: every combination of the per-field axes.  scan_table maps",
            "# each Scan field to its column; delete a field's axis and its entry and", "# it plays its own value."].map(String::from).to_vec();
        for (index, spec) in columns.iter().enumerate() {
            lines.push(String::new());
            lines.push(format!("# {}", note(spec)));
            lines.push(format!("a{index} = {}", sweep(spec, &sizes[index].to_string())));
        }
        let join = |prefix: &str| (0..count).map(|index| format!("{prefix}{index}")).collect::<Vec<_>>().join(", ");
        lines.push(String::new());
        lines.push(format!("{}, = np.meshgrid({}, indexing=\"ij\")", join("A"), join("a")));
        lines.extend(table(&(0..count).map(|index| format!("A{index}.ravel()")).collect::<Vec<_>>()));
        let shape = (0..count).map(|index| format!("len(a{index})")).collect::<Vec<_>>().join(", ");
        lines.push(if count == 1 { format!("scan_shape = ({shape},)") } else { format!("scan_shape = ({shape})") });
        return Ok(lines.join("\n") + "\n");
    }
    let mut lines: Vec<String> = ["import numpy as np", "", "# scan_table maps each Scan field this scan drives to its column: one",
        "# value per point, in the unit the editor shows for the field.  The", "# columns advance together.  Delete a field's entry and it plays its",
        "# own value.", "", "# number of scan points", "N = 21"].map(String::from).to_vec();
    for (index, spec) in columns.iter().enumerate() {
        lines.push(String::new());
        lines.push(format!("# {}", note(spec)));
        lines.push(format!("a{index} = {}", sweep(spec, "N")));
    }
    lines.push(String::new());
    lines.extend(table(&(0..count).map(|index| format!("a{index}")).collect::<Vec<_>>()));
    Ok(lines.join("\n") + "\n")
}

/// The wire rows as the device takes them (slot values are unsigned).
pub fn wire_rows(rows: &[Vec<i128>]) -> Result<Vec<Vec<u64>>> {
    rows.iter().map(|row| row.iter().map(|value| u64::try_from(*value)
        .map_err(|_| Error::new(Kind::Value, format!("scan wire value {value} is negative")))).collect()).collect()
}

/// A number from a scan row in a binding's unit.
pub fn scan_number(value: f64) -> Num { crate::num::number_for(value) }
