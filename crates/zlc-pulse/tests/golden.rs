//! Every stage of the pulse stack against v2's own answers
//! (`research/pulse-golden/make_golden.py`): decode refusals, the saved file's
//! bytes, played time, labels, resolved executable pulses, the compiled program
//! and its digest, the packed board image, scan wire rows and banks, analog
//! levels, schedule windows and the delay-FIFO capacity verdicts.
//!
//! Env-gated: `ZLC_PULSE_GOLDEN` = the generator's output directory.
use std::collections::BTreeMap;

use zlc_pulse::binding::{api_bindings_in_period_order, apply_config_values, field_label, resolve_api_parameters};
use zlc_pulse::codec::{sequence_from_tree, sequence_to_tree};
use zlc_pulse::compile::{analog_levels, compile_sequence, nominal_slot_values};
use zlc_pulse::json::{self, Json};
use zlc_pulse::model::Sequence;
use zlc_pulse::num::Num;
use zlc_pulse::scan::{prepare_scan_application, scan_columns_for};
use zlc_pulse::schedule::{bus_action_ticks, run_duration_seconds, trigger_windows_by_channel, validate_delay_capacity};
use zlc_pulse::wire::{FROZEN_CLOCK_HZ, StreamerParams, pack_program, pack_scan_rows};
use zlc_pulse::{Error, Result};

#[derive(Default)]
struct Tally { compared: BTreeMap<&'static str, usize>, failed: BTreeMap<&'static str, usize>, shown: usize }

impl Tally {
    fn check(&mut self, stage: &'static str, case: &str, ours: &Json, golden: &Json) {
        *self.compared.entry(stage).or_default() += 1;
        if ours.canonical() != golden.canonical() {
            *self.failed.entry(stage).or_default() += 1;
            if self.shown < 40 {
                self.shown += 1;
                let (a, b) = (ours.canonical(), golden.canonical());
                let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
                let from = at.saturating_sub(120);
                eprintln!("--- {case} {stage}\n  ours:   ...{}\n  golden: ...{}", &a[from..(at + 200).min(a.len())], &b[from..(at + 200).min(b.len())]);
            }
        }
    }
}

fn answer<T>(result: Result<T>, ok: impl FnOnce(T) -> Json) -> Json {
    match result {
        Ok(value) => Json::object(vec![("ok", ok(value))]),
        Err(error) => error_json(&error),
    }
}

fn error_json(error: &Error) -> Json {
    Json::object(vec![("error", Json::Array(vec![Json::str(error.kind.python_name()), Json::str(&error.message)]))])
}

/// The golden's answer with only the members a stage compares.
fn only(golden: &Json, keys: &[&str]) -> Json {
    match golden {
        Json::Object(items) => Json::Object(items.iter().filter(|(key, _)| keys.contains(&key.as_str())).cloned().collect()),
        other => other.clone(),
    }
}

fn geometry(case: &Json) -> StreamerParams {
    let mut params = StreamerParams::default();
    if let Some(Json::Object(items)) = case.get("geometry") {
        for (key, value) in items { assert!(params.set(key, value.as_int().unwrap() as u64), "geometry field {key}"); }
    }
    params
}

fn floats(rows: &Json) -> Vec<Vec<f64>> {
    rows.as_array().unwrap().iter().map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()).collect()
}

fn executable(sequence: &Sequence, case: &Json) -> Result<(Sequence, Option<Json>)> {
    let values: Option<Vec<(String, Num)>> = case.get("api_values").and_then(Json::as_object)
        .map(|items| items.iter().map(|(key, value)| (key.clone(), Num::from_json(value, key).unwrap())).collect());
    let resolved = resolve_api_parameters(sequence, values.as_deref())?;
    let Some(Json::Object(entries)) = case.get("config_values") else { return Ok((resolved, None)) };
    let entries: Vec<(String, (f64, String))> = entries.iter().map(|(key, pair)| {
        let pair = pair.as_array().unwrap();
        (key.clone(), (pair[0].as_f64().unwrap(), pair[1].as_str().unwrap().to_owned()))
    }).collect();
    let (filled, applied, unknown) = apply_config_values(&resolved, &entries, None)?;
    Ok((filled, Some(Json::object(vec![("applied", Json::strings(&applied)), ("unknown", Json::strings(&unknown))]))))
}

fn run_case(case: &Json, tally: &mut Tally) {
    let name = case.get("name").and_then(Json::as_str).unwrap_or("?").to_owned();
    let decoded = sequence_from_tree(case.get("tree").unwrap());
    let golden_decode = case.get("decode").unwrap();
    let ours = match &decoded { Ok(_) => Json::object(vec![("ok", Json::Bool(true))]), Err(error) => error_json(error) };
    tally.check("decode", &name, &ours, golden_decode);
    let Ok(sequence) = decoded else { return };
    if golden_decode.get("ok").is_none() { return; }
    let params = geometry(case);
    if let Some(golden) = case.get("roundtrip") {
        tally.check("roundtrip", &name, &Json::object(vec![("ok", Json::str(sequence_to_tree(&sequence).readable()))]), golden);
    }
    if let Some(golden) = case.get("played_ns") {
        tally.check("played_ns", &name, &Json::object(vec![("ok", Json::Float(sequence.played_nanoseconds()))]), golden);
    }
    if let Some(golden) = case.get("labels") {
        let labels = (|| -> Result<Json> {
            let bindings = sequence.bindings.iter().map(|binding| Ok(Json::Array(vec![Json::str(binding.field_id()),
                Json::str(field_label(&sequence, &binding.field_ref)?)]))).collect::<Result<Vec<_>>>()?;
            Ok(Json::object(vec![
                ("bindings", Json::Array(bindings)),
                ("api_order", Json::Array(api_bindings_in_period_order(&sequence).iter().map(|b| Json::str(b.field_id())).collect())),
                ("periods", Json::Array(sequence.periods.iter().map(|p| Json::str(sequence.period_label(&p.period_id))).collect())),
                ("bracket_bounds", Json::Array(sequence.bracket_bounds().iter().map(|(a, b)| Json::ints(&[*a as i64, *b as i64])).collect())),
                ("loops", Json::Array(sequence.loops().iter().map(|(a, b, c)| Json::ints(&[*a as i64, *b as i64, *c as i64])).collect())),
                ("abi", Json::str(sequence.target.abi_fingerprint())),
            ]))
        })();
        tally.check("labels", &name, &answer(labels, |x| x), golden);
    }
    if let Some(golden) = case.get("levels") {
        let levels = analog_levels(&sequence).map(|levels| Json::Object(levels.into_iter().map(|(key, points)| (key,
            Json::Array(points.into_iter().map(|(tick, value)| Json::Array(vec![Json::Int(tick), Json::Int(value.into())])).collect()))).collect()));
        tally.check("levels", &name, &answer(levels, |x| x), golden);
    }
    let Some(golden_resolved) = case.get("resolved") else { return };
    let (resolved, config_answer) = match executable(&sequence, case) {
        Ok(value) => value,
        Err(error) => { tally.check("resolved", &name, &error_json(&error), golden_resolved); return; }
    };
    let ours = Json::object(vec![("ok", sequence_to_tree(&resolved)), ("config", config_answer.unwrap_or(Json::Null))]);
    tally.check("resolved", &name, &ours, golden_resolved);
    let Some(golden_program) = case.get("program") else { return };
    let program = match compile_sequence(&resolved, &params, FROZEN_CLOCK_HZ) {
        Ok(program) => program,
        Err(error) => { tally.check("program", &name, &error_json(&error), golden_program); return; }
    };
    let nominal = nominal_slot_values(&resolved).unwrap();
    let ours = Json::object(vec![("ok", program.canonical_tree()), ("digest", Json::str(program.digest())),
                                 ("nominal", Json::ints(&nominal.iter().map(|v| *v as i64).collect::<Vec<_>>()))]);
    tally.check("program", &name, &ours, &only(golden_program, &["ok", "digest", "nominal"]));
    tally.check("frame_ticks", &name, &answer(program.frame_ticks(&nominal), |t| Json::Int(t as i128)), golden_program.get("frame_ticks").unwrap());
    if let Some(golden) = case.get("image") {
        let image = pack_program(&program, &params, &resolved.target)
            .map(|words| Json::Array(words.into_iter().map(|(a, v)| Json::ints(&[i64::from(a), i64::from(v)])).collect()));
        tally.check("image", &name, &answer(image, |x| x), golden);
    }
    let mut rows: Option<Vec<Vec<u64>>> = None;
    if let (Some(table), Some(golden)) = (case.get("scan_rows"), case.get("scan")) {
        let scanned = (|| -> Result<(Json, Vec<Vec<u64>>)> {
            let (effective, wire) = prepare_scan_application(&resolved, &floats(table), &params)?;
            let wire_u = zlc_pulse::scan::wire_rows(&wire)?;
            let columns = scan_columns_for(&resolved, &params)?;
            let mut banks = Vec::new();
            for chunk in 0..3u64.min((wire.len() as u64).div_ceil(params.bank_size)) {
                let words = pack_scan_rows(&wire_u, &params, chunk & 1, chunk)?;
                banks.push((chunk.to_string(), Json::Array(words.into_iter().map(|(a, v)| Json::ints(&[i64::from(a), i64::from(v)])).collect())));
            }
            let json = Json::object(vec![
                ("effective", Json::Array(effective.iter().map(|row| Json::Array(row.iter().map(|v| Json::Float(*v)).collect())).collect())),
                ("wire", Json::Array(wire.iter().map(|row| Json::Array(row.iter().map(|v| Json::Int(*v)).collect())).collect())),
                ("columns", Json::Array(columns.iter().map(|c| Json::object(vec![
                    ("name", Json::str(&c.name)), ("lo", Json::Float(c.lo)), ("hi", Json::Float(c.hi)), ("is_dac", Json::Bool(c.is_dac)),
                    ("unit", Json::str(&c.unit)), ("limit_lo", Json::Float(c.limit_lo)), ("limit_hi", Json::Float(c.limit_hi)),
                    ("wire_scale", Json::Float(c.wire_scale)), ("wire_offset", Json::Float(c.wire_offset)), ("label", Json::str(&c.label)),
                ])).collect())),
                ("banks", Json::Object(banks)),
            ]);
            Ok((json, wire_u))
        })();
        match scanned {
            Ok((json, wire)) => { tally.check("scan", &name, &Json::object(vec![("ok", json)]), &normalize_floats(golden)); rows = Some(wire); }
            Err(error) => tally.check("scan", &name, &error_json(&error), golden),
        }
    } else if program.slot_count() > 0 {
        rows = Some(vec![nominal.clone()]);
    }
    if let Some(golden) = case.get("schedule").filter(|golden| golden.get("skipped").is_none()) {
        let digital: Vec<&str> = resolved.target.ports.iter().filter(|p| p.kind == zlc_pulse::model::PortKind::Digital).map(|p| p.key.as_str()).collect();
        let table = rows.as_deref();
        let schedule = (|| -> Result<Json> {
            let windows = trigger_windows_by_channel(&program, &digital, table, 2, 1)?;
            let bus = bus_action_ticks(&program, table, 2, 1, None)?;
            Ok(Json::object(vec![
                ("windows", Json::Object(windows.into_iter().map(|(key, w)| (key, Json::Array(w.into_iter()
                    .map(|(a, b)| Json::Array(vec![Json::Int(a as i128), Json::Int(b as i128)])).collect()))).collect())),
                ("bus", Json::Object(bus.into_iter().enumerate().map(|(index, ticks)| (index.to_string(),
                    Json::Array(ticks.into_iter().map(|t| Json::Int(t as i128)).collect()))).collect())),
                ("seconds", Json::Float(run_duration_seconds(&program, table, 2, 1)?)),
            ]))
        })();
        tally.check("schedule", &name, &answer(schedule, |x| x), golden);
    }
    if let Some(Json::Object(golden)) = case.get("capacity") {
        for (key, expected) in golden {
            let (run, scan) = key.split_once(',').unwrap();
            let verdict = validate_delay_capacity(&program, rows.as_deref().unwrap_or(&[]), &params, run.parse().unwrap(), scan.parse().unwrap());
            tally.check("capacity", &name, &answer(verdict, |()| Json::str("ok")), expected);
        }
    }
}

/// The golden writes a DAC's effective value as a float and its wire value as
/// an int, as we do; only an integral float written `512` (an int) needs a nudge.
fn normalize_floats(golden: &Json) -> Json {
    let mut golden = golden.clone();
    if let Json::Object(items) = &mut golden {
        for (key, value) in items.iter_mut() {
            if key != "ok" { continue; }
            if let Json::Object(fields) = value {
                for (field, rows) in fields.iter_mut() {
                    if field == "effective" {
                        if let Json::Array(rows) = rows { for row in rows { if let Json::Array(values) = row {
                            for v in values { if let Json::Int(i) = v { *v = Json::Float(*i as f64); } }
                        } } }
                    }
                }
            }
        }
    }
    golden
}

#[test]
fn pulse_stack_matches_the_v2_goldens() {
    let Ok(root) = std::env::var("ZLC_PULSE_GOLDEN") else { return };
    let mut tally = Tally::default();
    let mut cases = 0;
    for group in ["examples", "diff", "synthetic", "fuzz_deployed", "fuzz_small"] {
        let path = std::path::Path::new(&root).join(format!("{group}.json"));
        let Ok(text) = std::fs::read_to_string(&path) else { continue };
        let document = json::parse(&text, "golden").unwrap();
        for case in document.get("cases").unwrap().as_array().unwrap() { run_case(case, &mut tally); cases += 1; }
    }
    for (stage, count) in &tally.compared {
        eprintln!("{stage:>12}: {count} compared, {} different", tally.failed.get(stage).copied().unwrap_or(0));
    }
    eprintln!("{cases} cases");
    assert!(tally.failed.is_empty(), "golden mismatches");
}

#[test]
fn geometry_manifest_and_config_values_match_v2() {
    let Ok(root) = std::env::var("ZLC_PULSE_GOLDEN") else { return };
    let text = std::fs::read_to_string(std::path::Path::new(&root).join("extras.json")).unwrap();
    let extras = json::parse(&text, "extras").unwrap();
    let mut tally = Tally::default();
    let config = zlc_pulse::wire::load_streamer_config(None).unwrap();
    let golden = extras.get("config").unwrap();
    tally.check("config", "deployed", &Json::object(vec![
        ("clock_hz", Json::Float(config.clock_hz)), ("uart_baud", Json::Int(config.uart_baud.into())),
        ("fpga_part", Json::str(&config.fpga_part)), ("target_pct", Json::Float(config.target_pct)),
        ("warnings", Json::strings(&config.warnings)),
    ]), &only(golden, &["clock_hz", "uart_baud", "fpga_part", "target_pct", "warnings"]));
    for (index, case) in extras.get("geometries").unwrap().as_array().unwrap().iter().enumerate() {
        let mut params = StreamerParams::default();
        for (key, value) in case.get("params").unwrap().as_object().unwrap() { params.set(key, value.as_int().unwrap() as u64); }
        let bases = zlc_pulse::wire::region_bases(&params);
        let ours = Json::object(vec![
            ("fingerprint", Json::Int(zlc_pulse::wire::build_fingerprint(&params).unwrap().into())),
            ("bases", Json::object(vec![("ctrl", Json::Int(bases.ctrl.into())), ("rows", Json::Int(bases.rows.into())), ("scan", Json::Int(bases.scan.into())),
                                        ("loop", Json::Int(bases.r#loop.into())), ("delay", Json::Int(bases.delay.into())), ("total", Json::Int(bases.total.into()))])),
            ("ip", Json::Object(zlc_pulse::wire::build_ip_sizes(&params).into_iter().map(|(k, v)| (k.to_owned(), Json::Int(v.into()))).collect())),
            ("row_bits", Json::Int(params.row_bits().into())), ("row_words", Json::Int(params.row_words().into())),
            ("slot_sel_width", Json::Int(params.slot_sel_width().into())), ("num_delay_ch", Json::Int(params.num_delay_ch().into())),
            ("bus_action_bits", Json::Int(params.bus_action_bits().into())),
        ]);
        tally.check("geometry", &format!("geometry {index}"), &ours, &only(case, &["fingerprint", "bases", "ip", "row_bits", "row_words",
                                                                                    "slot_sel_width", "num_delay_ch", "bus_action_bits"]));
    }
    let target = zlc_pulse::manifest::pulse_target_from_xdc(None, None, None).unwrap();
    let golden = extras.get("target").unwrap();
    tally.check("target", "deployed", &Json::object(vec![("tree", zlc_pulse::codec::target_to_tree(&target)), ("abi", Json::str(target.abi_fingerprint()))]), golden);
    for (index, case) in extras.get("config_values").unwrap().as_array().unwrap().iter().enumerate() {
        let tree = case.get("tree").unwrap();
        let decoded = zlc_pulse::codec::config_values_from_tree(tree);
        let ours = answer(decoded.clone(), |values| Json::Object(values.into_iter()
            .map(|(key, (value, unit))| (key, Json::Array(vec![Json::Float(value), Json::str(unit)]))).collect()));
        tally.check("config_values", &format!("config {index}"), &ours, case.get("decoded").unwrap());
        let text = decoded.and_then(|values| zlc_pulse::codec::config_values_to_tree(&values)).map(|tree| tree.readable());
        tally.check("config_text", &format!("config {index}"), &answer(text, Json::str), case.get("text").unwrap());
    }
    for (stage, count) in &tally.compared {
        eprintln!("{stage:>14}: {count} compared, {} different", tally.failed.get(stage).copied().unwrap_or(0));
    }
    assert!(tally.failed.is_empty(), "golden mismatches");
}
