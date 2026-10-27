//! What a run asks of the board (line D's `nodes/_framework/pulse.py`): a
//! document's description with the facts a scan reads, a FIRE with API values
//! and a scan table whose receipt says what the program plays when, a FIRE's
//! report, and the Pulse ticks a scan plan schedules (the board's half of
//! v2's `estimate_scan_time`). Every rule is v2's own function, ported.
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::binding::{api_bindings_in_period_order, apply_config_values, bracket_label, convert_time, field_label, pulse_field_value,
                     resolve_api_parameters, resolve_subpulse_sources};
use crate::board::{Board, archive_snapshot};
use crate::codec::{ConfigValues, read_pulse_document};
use crate::compile::{CompiledProgram, compile_sequence};
use crate::device::{AppliedState, DoneReport};
use crate::error::{Error, Kind, Result, repr, value_error};
use crate::json::Json;
use crate::model::{FieldKind, FieldRef, Sequence};
use crate::num::Num;
use crate::scan::{ScanColumnSpec, api_parameter_columns_for, prepare_scan_application, scan_columns_for, wire_rows};
use crate::schedule::trigger_edge_ticks;
use crate::wire::{STATUS_ERROR, STATUS_UNDERFLOW, StreamerParams};

fn int(value: impl Into<i128>) -> Json { Json::Int(value.into()) }

/// Every loop body is replayed at most twice in a receipt's timing: a period
/// inside a loop shows two starts without its loop being expanded.
fn twice(_span: u128) -> u64 { 2 }

/// A pulse document, its Subpulse files resolved beside it.
pub fn read_document(path: &Path) -> Result<(PathBuf, Sequence)> {
    let path = crate::codec::absolute(path);
    let (sequence, _editor) = read_pulse_document(&path)?;
    let resolved = resolve_subpulse_sources(&sequence, path.parent())?;
    Ok((path, resolved))
}

// ------------------------------------------------------------ describe
fn facts(column: &ScanColumnSpec) -> [(&'static str, Json); 3] {
    [("is_dac", Json::Bool(column.is_dac)), ("limit_lo", Json::Float(column.limit_lo)), ("limit_hi", Json::Float(column.limit_hi))]
}

/// What a node reads of a pulse: its API fields in the order it plays them
/// (own unit, authored value, column facts), its scan slots in slot order
/// (with the value each holds in the pulse, where a plan's default sweep is
/// centred), the API fields it offers no plan (an empty bracket's count) and
/// its periods.
pub fn describe_sequence(sequence: &Sequence) -> Result<Json> {
    let columns = api_parameter_columns_for(sequence)?;
    let mut api_fields = Vec::new();
    for binding in api_bindings_in_period_order(sequence) {
        let column = columns.iter().find(|column| column.name == binding.field_id()).expect("one column per API field");
        let mut items = vec![
            ("field_id", Json::str(binding.field_id())), ("label", Json::str(field_label(sequence, &binding.field_ref)?)),
            ("kind", Json::str(binding.kind().name())), ("unit", Json::str(&binding.unit)),
            ("value", Json::Float(pulse_field_value(sequence, &binding.field_ref, &binding.unit)?)),
        ];
        items.extend(facts(column));
        api_fields.push(Json::object(items));
    }
    let mut scan_fields = Vec::new();
    // A slot's value as the board plays it unscanned (resolve_scan_point's authored row).
    let resolved = resolve_subpulse_sources(sequence, None)?;
    let slots = resolved.scan_bindings();
    for column in scan_columns_for(sequence, &StreamerParams::default())? {
        let kind = sequence.bindings.iter().find(|binding| binding.field_id() == column.name).map_or("duration", |binding| binding.kind().name());
        let slot = slots.iter().find(|slot| slot.field_id() == column.name).expect("one slot per scan column");
        let mut items = vec![("field_id", Json::str(&column.name)), ("label", Json::str(&column.label)), ("kind", Json::str(kind)),
                             ("unit", Json::str(&column.unit)), ("value", Json::Float(pulse_field_value(&resolved, &slot.field_ref, &slot.unit)?))];
        items.extend(facts(&column));
        scan_fields.push(Json::object(items));
    }
    let api: Vec<String> = sequence.api_bindings().iter().map(|binding| binding.field_id()).collect();
    let mut unscannable = Vec::new();
    for (bracket, (start, end)) in sequence.brackets.iter().zip(sequence.bracket_bounds()) {
        let field = FieldRef::bracket_count(&bracket.bracket_id).key();
        if start == end && api.contains(&field) {
            let label = bracket_label(sequence, &bracket.bracket_id)?;
            unscannable.push((field, Json::str(format!("{label} is empty, so scanning its repeat count changes nothing"))));
        }
    }
    let periods = sequence.periods.iter().map(|period| Json::object(vec![
        ("period_id", Json::str(&period.period_id)), ("name", Json::str(&period.name)), ("label", Json::str(sequence.period_label(&period.period_id))),
    ])).collect();
    Ok(Json::object(vec![("api_fields", Json::Array(api_fields)), ("scan_fields", Json::Array(scan_fields)),
                         ("unscannable", Json::Object(unscannable)), ("periods", Json::Array(periods))]))
}

/// `pulse_describe`: the document's path and name over [`describe_sequence`].
pub fn describe_document(path: &Path) -> Result<Json> {
    let (path, sequence) = read_document(path)?;
    let Json::Object(mut items) = describe_sequence(&sequence)? else { unreachable!("an object") };
    items.insert(0, ("name".into(), Json::str(path.file_stem().map_or_else(String::new, |stem| stem.to_string_lossy().into_owned()))));
    items.insert(0, ("path".into(), Json::str(path.display().to_string())));
    Ok(Json::Object(items))
}

// ------------------------------------------------------------ a run's plan
/// API values as a caller gives them: field id -> `(number, unit)`.
pub type ApiValues = [(String, (f64, String))];

/// A scan table to play: the named scan slots, one row per point in their
/// own units and in that field order.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanRows { pub fields: Vec<String>, pub rows: Vec<Vec<f64>> }

/// What a FIRE of one document plays: the executable pulse, the API values
/// in their fields' units, the wire rows and the table as played.
pub struct RunPlan { pub resolved: Arc<Sequence>, pub values: Vec<(String, f64)>, pub rows: Vec<Vec<u64>>, pub played: Option<ScanRows> }

/// Each API value in its field's own unit: a time converted exactly (v2
/// `convert_time`); any other field takes its own unit only.
fn api_inputs(sequence: &Sequence, api_values: &ApiValues) -> Result<Vec<(String, f64)>> {
    let declared = sequence.api_bindings();
    api_values.iter().map(|(key, (value, unit))| {
        let Some(binding) = declared.iter().find(|binding| binding.field_id() == *key) else {
            return value_error(format!("{} is not an API field of this pulse", repr(key)));
        };
        let value = if *unit == binding.unit { *value } else { convert_time(*value, unit, &binding.unit)? };
        Ok((key.clone(), value))
    }).collect()
}

fn nums(values: &[(String, f64)]) -> Result<Vec<(String, Num)>> {
    values.iter().map(|(key, value)| Ok((key.clone(), Num::new(*value, key)?))).collect()
}

/// The pulse with Scan on exactly these fields (each a field of the pulse,
/// named once).
fn scanning(sequence: &Sequence, planned: &[String]) -> Result<Sequence> {
    for (index, field) in planned.iter().enumerate() {
        if !sequence.bindings.iter().any(|binding| binding.field_id() == *field) {
            return value_error(format!("{} is not a field of this pulse", repr(field)));
        }
        if planned[..index].contains(field) { return value_error(format!("the scan names {} twice", repr(field))); }
    }
    sequence.with(|parts| for binding in &mut parts.bindings { binding.scan = planned.contains(&binding.field_id()); })
}

/// The plan of one FIRE (line D's stand-in, v2's functions): API values
/// resolved; with a table, its fields are the scan slots (no other slot
/// streams), quantized to what the board plays; without one no slot streams.
pub fn plan_run(sequence: &Sequence, api_values: &ApiValues, scan: Option<&ScanRows>, geometry: &StreamerParams) -> Result<RunPlan> {
    let values = api_inputs(sequence, api_values)?;
    let planned: &[String] = scan.map_or(&[], |table| table.fields.as_slice());
    let resolved = resolve_api_parameters(&scanning(sequence, planned)?, Some(&nums(&values)?))?;
    let (mut rows, mut played) = (Vec::new(), None);
    if let Some(table) = scan {
        let slots = resolved.scan_bindings().len();
        if slots as u64 > geometry.num_slots {
            return value_error(format!("the board advances at most {} slots per cycle; this plan scans {slots} slots", geometry.num_slots));
        }
        let names: Vec<String> = scan_columns_for(&resolved, geometry)?.into_iter().map(|column| column.name).collect();
        let order: Vec<usize> = names.iter().map(|name| planned.iter().position(|field| field == name).expect("a planned slot")).collect();
        let authored = table.rows.iter().map(|row| {
            if row.len() != planned.len() {
                return value_error(format!("each scan row has one value per field: {} field(s), {} value(s)", planned.len(), row.len()));
            }
            Ok(order.iter().map(|index| row[*index]).collect())
        }).collect::<Result<Vec<Vec<f64>>>>()?;
        let (effective, wire) = prepare_scan_application(&resolved, &authored, geometry)?;
        rows = wire_rows(&wire)?;
        played = Some(ScanRows {
            fields: planned.to_vec(),
            rows: effective.iter().map(|row| planned.iter().map(|field| row[names.iter().position(|name| name == field).expect("a slot")]).collect()).collect(),
        });
    }
    Ok(RunPlan { resolved: Arc::new(resolved), values, rows, played })
}

/// The receipt's `timing`: the program as a FIRE plays it, over one run (of
/// the table's first row), every loop body replayed at most twice -- when each
/// period starts and how long it lasts, and the rising edges of every digital
/// lane that is not a clock output.
pub fn timing_json(source: &Sequence, program: &CompiledProgram, point: &[u64]) -> Result<Json> {
    let visits = program.frame_visits(point, Some(&twice))?;
    let durations = program.resolved_durations(point)?;
    let channels: Vec<&str> = program.channels.iter().enumerate().filter(|(bit, _)| program.clk_enable & (1u128 << bit) == 0)
        .map(|(_, channel)| channel.as_str()).collect();
    let table = (!point.is_empty()).then(|| vec![point.to_vec()]);
    let streams = trigger_edge_ticks(program, &channels, table.as_deref(), 1, 1, Some(&twice))?;
    let periods = source.periods.iter().enumerate().map(|(row, period)| Json::object(vec![
        ("period_id", Json::str(&period.period_id)), ("name", Json::str(&period.name)), ("label", Json::str(source.period_label(&period.period_id))),
        ("starts", Json::Array(visits.iter().filter(|(index, _)| *index == row).map(|(_, tick)| int(*tick as i128)).collect())),
        ("duration_ticks", int(durations[row])),
    ])).collect();
    let earliest = program.channel_delays.iter().map(|delay| *delay as i128).min().unwrap_or(0).min(0);
    Ok(Json::object(vec![
        ("clock_hz", Json::Float(program.clock_hz)), ("digest", Json::str(program.digest())),
        ("duration_ticks", int(program.frame_ticks(point)? as i128)), ("earliest_lane_delay_ticks", int(earliest)),
        ("periods", Json::Array(periods)),
        ("rising_edges", Json::Object(streams.into_iter().map(|(channel, edges)| {
            (channel, Json::Array(edges.iter().step_by(2).map(|tick| int(*tick as i128)).collect()))
        }).collect())),
    ]))
}

fn played_json(table: &ScanRows) -> Json {
    Json::object(vec![("fields", Json::strings(&table.fields)),
                      ("rows", Json::Array(table.rows.iter().map(|row| Json::Array(row.iter().map(|v| Json::Float(*v)).collect())).collect()))])
}

/// Plan, ABI check, compile and LOAD one read document.
fn load_plan(board: &Board, sequence: &Sequence, api_values: &ApiValues, scan: Option<&ScanRows>) -> Result<RunPlan> {
    let description = board.description();
    let plan = plan_run(sequence, api_values, scan, &description.geometry)?;
    if plan.resolved.target.abi_fingerprint() != description.target.abi_fingerprint() {
        return value_error("pulse target ABI is incompatible with the connected board");
    }
    let program = Arc::new(compile_sequence(&plan.resolved, &description.geometry, description.clock_hz)?);
    board.load(program, Some(plan.resolved.clone()), plan.rows.clone())?;
    Ok(plan)
}

/// What the board holds or plays as a receipt: the command id of a FIRE, the
/// archive snapshot, `timing` of the program (Config filled) and, from the
/// plan, the API values (fields' units) and the rows as played.
fn receipt(board: &Board, applied: &AppliedState, command_id: Option<u32>, plan: Option<&RunPlan>) -> Result<Json> {
    let source = applied.source.clone().or_else(|| plan.map(|plan| plan.resolved.clone()))
        .ok_or_else(|| Error::new(Kind::Runtime, "the board holds a program without its pulse"))?;
    let timing = timing_json(&source, &applied.program, applied.rows.first().map_or(&[], Vec::as_slice))?;
    let mut receipt = Vec::new();
    if let Some(command_id) = command_id { receipt.push(("command_id", int(command_id))); }
    if let Some(plan) = plan {
        receipt.push(("api_values", Json::Object(plan.values.iter().map(|(key, value)| (key.clone(), Json::Float(*value))).collect())));
    }
    receipt.extend([("applied", archive_snapshot(board, applied)?), ("timing", timing)]);
    if let Some(played) = plan.and_then(|plan| plan.played.as_ref()) { receipt.push(("scan", played_json(played))); }
    Ok(Json::object(receipt))
}

/// `pulse_run`: plan, ABI check, compile, load and fire one document; the
/// receipt is the FIRE's command id, the API values played (fields' units),
/// the archive snapshot, `timing` of the program fired (Config filled) and,
/// with a table, the rows as played.
pub fn run_document(board: &Board, path: &Path, api_values: &ApiValues, scan: Option<&ScanRows>, run_repeats: u64, scan_repeats: u64) -> Result<Json> {
    let (_path, sequence) = read_document(path)?;
    run_sequence(board, &sequence, api_values, scan, run_repeats, scan_repeats)
}

/// [`run_document`] of a document already read (a run's pinned pulse).
pub fn run_sequence(board: &Board, sequence: &Sequence, api_values: &ApiValues, scan: Option<&ScanRows>, run_repeats: u64, scan_repeats: u64)
    -> Result<Json> {
    let plan = load_plan(board, sequence, api_values, scan)?;
    let (applied, command_id) = board.fire(run_repeats, scan_repeats)?;
    receipt(board, &applied, Some(command_id), Some(&plan))
}

/// `pulse_run` stage `load` (v2 `arm_sequencer`): the document LOADed and not
/// fired; the receipt has no command id, `applied` is the LOAD's.
pub fn load_sequence(board: &Board, sequence: &Sequence, api_values: &ApiValues, scan: Option<&ScanRows>) -> Result<Json> {
    let plan = load_plan(board, sequence, api_values, scan)?;
    let applied = board.applied()?.ok_or_else(|| Error::new(Kind::Runtime, "the board reports no LOAD"))?;
    receipt(board, &applied, None, Some(&plan))
}

/// `pulse_run` stage `fire`: FIRE what a LOAD put on the board, only while the
/// board still holds that program (`loaded`: its digest).
pub fn fire_loaded(board: &Board, loaded: &str, run_repeats: u64, scan_repeats: u64) -> Result<Json> {
    let held = board.applied()?.map(|applied| applied.program.digest());
    if held.as_deref() != Some(loaded) {
        return Err(Error::new(Kind::Runtime, "the board no longer holds the pulse this run loaded; load it again"));
    }
    let (applied, command_id) = board.fire(run_repeats, scan_repeats)?;
    receipt(board, &applied, Some(command_id), None)
}

/// `pulse_wait_done`'s answer: a report's words, or not done yet.
pub fn done_json(report: Option<&DoneReport>) -> Json {
    let Some(report) = report else { return Json::object(vec![("done", Json::Bool(false))]) };
    let status = if report.status & STATUS_UNDERFLOW != 0 || report.underflow { "underflow" }
                 else if report.status & STATUS_ERROR != 0 || !report.fault().is_empty() { "error" } else { "done" };
    Json::object(vec![
        ("done", Json::Bool(true)), ("status", Json::str(status)), ("cursor", report.cursor.map_or(Json::Null, |cursor| int(cursor))),
        ("elapsed_seconds", Json::Float(report.elapsed_seconds)), ("command_id", int(report.command_id)), ("fault", Json::str(report.fault())),
    ])
}

// ------------------------------------------------------------ estimate
/// A cartesian scan plan's board axes: each named slot's values, in its own unit.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanAxes { pub fields: Vec<String>, pub columns: Vec<Vec<f64>> }

/// At most this many API timing combinations are compiled for one estimate (v2's preview budget).
pub const ESTIMATE_PROGRAMS: usize = 256;

/// The Pulse ticks one API point of a scan plan schedules (v2
/// `estimate_scan_time`'s compile loop): the board table is each axis's
/// values quantized as the board plays them, its rows' ticks summed the way
/// v2 sums them (duration slots add; other slots play the first row's), the
/// first row's ticks and the per-FIRE output-delay bound. `config` fills the
/// pulse as a FIRE would; `None` leaves it as authored.
pub fn estimate_point(sequence: &Sequence, api_values: &ApiValues, axes: Option<&ScanAxes>, geometry: &StreamerParams, clock_hz: f64,
                      config: Option<&ConfigValues>) -> Result<Json> {
    let planned: &[String] = axes.map_or(&[], |axes| axes.fields.as_slice());
    let marked = scanning(sequence, planned)?;
    let mut values_by_slot: Vec<Vec<u64>> = Vec::new();
    for column in scan_columns_for(&marked, geometry)? {
        let axes = axes.expect("slots are planned fields");
        let index = axes.fields.iter().position(|field| *field == column.name).expect("a planned slot");
        let Some(values) = axes.columns.get(index).filter(|values| !values.is_empty()) else {
            return value_error(format!("the scan axis {} has no values", repr(&column.name)));
        };
        let one = marked.with(|parts| for binding in &mut parts.bindings { binding.scan = binding.field_id() == column.name; })?;
        let rows: Vec<Vec<f64>> = values.iter().map(|value| vec![*value]).collect();
        let (_effective, wire) = prepare_scan_application(&one, &rows, geometry)?;
        values_by_slot.push(wire_rows(&wire)?.into_iter().map(|row| row[0]).collect());
    }
    let first: Vec<u64> = values_by_slot.iter().map(|values| values[0]).collect();
    let count = values_by_slot.iter().try_fold(1u128, |product, values| product.checked_mul(values.len() as u128))
        .ok_or_else(|| Error::new(Kind::Value, "the scan table has too many rows to count"))?;
    let values = api_inputs(&marked, api_values)?;
    let streamed = resolve_api_parameters(&marked, Some(&nums(&values)?))?;
    let filled = match config { Some(entries) => apply_config_values(&resolve_subpulse_sources(&streamed, None)?, entries, None)?.0, None => streamed };
    let program = compile_sequence(&filled, geometry, clock_hz)?;
    let base = program.frame_ticks(&first)? as i128;
    let overflow = || Error::new(Kind::Value, "the scan table's ticks overflow");
    let mut table = base.checked_mul(count as i128).ok_or_else(overflow)?;
    for (index, values) in values_by_slot.iter().enumerate() {
        if program.slot_kinds[index] != FieldKind::Duration { continue; }
        let mut delta: i128 = 0;
        for value in values {
            let mut point = first.clone();
            point[index] = *value;
            delta += program.frame_ticks(&point)? as i128 - base;
        }
        table = table.checked_add(delta.checked_mul((count / values.len() as u128) as i128).ok_or_else(overflow)?).ok_or_else(overflow)?;
    }
    let delay = program.channel_delays.iter().copied().chain(program.bus_delays.iter().map(|item| item.delay_ticks)).max().unwrap_or(0);
    Ok(Json::object(vec![
        ("api_values", Json::Object(values.iter().map(|(key, value)| (key.clone(), Json::Float(*value))).collect())),
        ("board_rows", int(count as i128)), ("first_row_ticks", int(base)), ("table_ticks", int(table)), ("delay_ticks", int(delay)),
    ]))
}

/// `pulse_estimate`: [`estimate_point`] for each API timing combination of a
/// plan (at most [`ESTIMATE_PROGRAMS`]), compiled for this board with the
/// Config values it holds; the sums say what one sweep of the plan's host
/// points schedules.
pub fn estimate_document(board: &Board, path: &Path, api_points: &[Vec<(String, (f64, String))>], axes: Option<&ScanAxes>) -> Result<Json> {
    if api_points.len() > ESTIMATE_PROGRAMS {
        return value_error(format!("{} API timing combinations exceed the {ESTIMATE_PROGRAMS}-program preview budget.", api_points.len()));
    }
    let (_path, sequence) = read_document(path)?;
    let description = board.description();
    let config = board.config().values();
    let none = [Vec::new()];
    let api_points = if api_points.is_empty() { &none[..] } else { api_points };
    let points = api_points.iter().map(|values| estimate_point(&sequence, values, axes, &description.geometry, description.clock_hz, Some(&config)))
        .collect::<Result<Vec<_>>>()?;
    let sum = |key: &str| points.iter().map(|point| point.get(key).and_then(Json::as_int).unwrap_or(0)).sum::<i128>();
    Ok(Json::object(vec![("clock_hz", Json::Float(description.clock_hz)), ("table_ticks", int(sum("table_ticks"))),
                         ("delay_ticks", int(sum("delay_ticks"))), ("points", Json::Array(points))]))
}
