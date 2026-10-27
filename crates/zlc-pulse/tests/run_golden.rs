//! What a run asks of the board against v2's own answers
//! (`research/pulse-golden/make_run_golden.py`): the description's scan facts,
//! a FIRE's API values, scan table as played and receipt `timing`, and the
//! Pulse ticks of an estimate, over make_golden.py's cases and line D's
//! documents. A table's fields given in another order play the same rows.
//!
//! Env-gated: `ZLC_PULSE_RUN_GOLDEN` = the generator's output directory.
use std::collections::BTreeMap;

use zlc_pulse::binding::{apply_config_values, resolve_subpulse_sources};
use zlc_pulse::codec::sequence_from_tree;
use zlc_pulse::compile::compile_sequence;
use zlc_pulse::json::{self, Json};
use zlc_pulse::model::Sequence;
use zlc_pulse::run::{ScanAxes, ScanRows, describe_document, describe_sequence, estimate_point, plan_run, timing_json};
use zlc_pulse::wire::{FROZEN_CLOCK_HZ, StreamerParams};
use zlc_pulse::{Error, Result};

#[derive(Default)]
struct Tally { compared: BTreeMap<&'static str, usize>, failed: BTreeMap<&'static str, usize>, shown: usize }

impl Tally {
    fn check(&mut self, stage: &'static str, case: &str, ours: &Json, golden: &Json) {
        *self.compared.entry(stage).or_default() += 1;
        if ours.canonical() != golden.canonical() {
            *self.failed.entry(stage).or_default() += 1;
            if self.shown < 30 {
                self.shown += 1;
                let (a, b) = (ours.canonical(), golden.canonical());
                let at = a.bytes().zip(b.bytes()).position(|(x, y)| x != y).unwrap_or(a.len().min(b.len()));
                let from = at.saturating_sub(120);
                eprintln!("--- {case} {stage}\n  ours:   ...{}\n  golden: ...{}", &a[from..(at + 200).min(a.len())], &b[from..(at + 200).min(b.len())]);
            }
        }
    }
}

fn answer(result: Result<Json>) -> Json {
    match result {
        Ok(value) => Json::object(vec![("ok", value)]),
        Err(error) => error_json(&error),
    }
}

fn error_json(error: &Error) -> Json {
    Json::object(vec![("error", Json::Array(vec![Json::str(error.kind.python_name()), Json::str(&error.message)]))])
}

/// A description as both sides are compared: v3 states each scan slot's `value`
/// (where a plan's default sweep is centred) and seeds no sweep, so neither
/// that nor v2's column `lo`/`hi` is compared; the rest is v2's to the digit.
fn comparable(answer: &Json) -> Json {
    let Some(Json::Object(items)) = answer.get("ok") else { return answer.clone() };
    let strip = |fields: &Json, keys: &[&str]| Json::Array(fields.as_array().unwrap().iter().map(|field| Json::Object(
        field.as_object().unwrap().iter().filter(|(key, _)| !keys.contains(&key.as_str())).cloned().collect())).collect());
    let items = items.iter().map(|(key, value)| (key.clone(), match key.as_str() {
        "api_fields" => strip(value, &["lo", "hi"]),
        "scan_fields" => strip(value, &["lo", "hi", "value"]),
        _ => value.clone(),
    })).collect();
    Json::object(vec![("ok", Json::Object(items))])
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

fn strings(items: &Json) -> Vec<String> { items.as_array().unwrap().iter().map(|item| item.as_str().unwrap().to_owned()).collect() }

fn api_values(request: &Json) -> Vec<(String, (f64, String))> {
    request.get("api_values").and_then(Json::as_object).unwrap_or(&[]).iter().map(|(key, pair)| {
        let pair = pair.as_array().unwrap();
        (key.clone(), (pair[0].as_f64().unwrap(), pair[1].as_str().unwrap().to_owned()))
    }).collect()
}

fn config(case: &Json) -> Option<Vec<(String, (f64, String))>> {
    case.get("config_values").and_then(Json::as_object).map(|entries| entries.iter().map(|(key, pair)| {
        let pair = pair.as_array().unwrap();
        (key.clone(), (pair[0].as_f64().unwrap(), pair[1].as_str().unwrap().to_owned()))
    }).collect())
}

/// One FIRE's receipt facts as the board computes them, with the FIRE's Config fill.
fn receipt(sequence: &Sequence, api: &[(String, (f64, String))], scan: Option<&ScanRows>, params: &StreamerParams,
           config: Option<&Vec<(String, (f64, String))>>) -> Result<Json> {
    let plan = plan_run(sequence, api, scan, params)?;
    let source = match config {
        Some(entries) => apply_config_values(&resolve_subpulse_sources(&plan.resolved, None)?, entries, None)?.0,
        None => (*plan.resolved).clone(),
    };
    let program = compile_sequence(&source, params, FROZEN_CLOCK_HZ)?;
    let mut items = vec![
        ("api_values", Json::Object(plan.values.iter().map(|(key, value)| (key.clone(), Json::Float(*value))).collect())),
        ("timing", timing_json(&source, &program, plan.rows.first().map_or(&[], Vec::as_slice))?),
    ];
    if let Some(played) = &plan.played {
        items.push(("scan", Json::object(vec![("fields", Json::strings(&played.fields)),
            ("rows", Json::Array(played.rows.iter().map(|row| Json::Array(row.iter().map(|v| Json::Float(*v)).collect())).collect()))])));
    }
    Ok(Json::object(items))
}

fn run_case(case: &Json, tally: &mut Tally) {
    let name = case.get("name").and_then(Json::as_str).unwrap_or("?").to_owned();
    let Ok(sequence) = sequence_from_tree(case.get("tree").unwrap()) else { return };
    let params = geometry(case);
    tally.check("describe", &name, &comparable(&answer(describe_sequence(&sequence))), &comparable(case.get("describe").unwrap()));
    let Some(request) = case.get("request") else { return };
    let api = api_values(request);
    let config = config(case);
    let scan = request.get("scan").map(|scan| ScanRows { fields: strings(scan.get("fields").unwrap()), rows: floats(scan.get("rows").unwrap()) });
    let golden_run = case.get("run").unwrap();
    tally.check("run", &name, &answer(receipt(&sequence, &api, scan.as_ref(), &params, config.as_ref())), golden_run);
    tally.check("unscanned", &name, &answer(receipt(&sequence, &api, None, &params, config.as_ref())), case.get("unscanned").unwrap());
    // The same table with its fields in reverse order plays the same rows, said in that order.
    if let (Some(table), Some(played)) = (&scan, golden_run.get("ok").and_then(|ok| ok.get("scan"))) {
        if table.fields.len() > 1 {
            let reversed = ScanRows {
                fields: table.fields.iter().rev().cloned().collect(),
                rows: table.rows.iter().map(|row| row.iter().rev().copied().collect()).collect(),
            };
            let expected = Json::Array(played.get("rows").unwrap().as_array().unwrap().iter()
                .map(|row| Json::Array(row.as_array().unwrap().iter().rev().cloned().collect())).collect());
            let ours = plan_run(&sequence, &api, Some(&reversed), &params).map(|plan| Json::Array(plan.played.unwrap().rows.iter()
                .map(|row| Json::Array(row.iter().map(|v| Json::Float(*v)).collect())).collect()));
            tally.check("reordered", &name, &answer(ours), &Json::object(vec![("ok", expected)]));
        }
    }
    let estimate_request = case.get("estimate_request").unwrap();
    let axes = estimate_request.get("scan").map(|scan| ScanAxes { fields: strings(scan.get("fields").unwrap()), columns: floats(scan.get("columns").unwrap()) });
    let ours = estimate_point(&sequence, &api_values(estimate_request), axes.as_ref(), &params, FROZEN_CLOCK_HZ, config.as_ref()).map(|point| {
        let Json::Object(items) = point else { unreachable!() };
        Json::Object(items.into_iter().filter(|(key, _)| key != "api_values").collect())
    });
    tally.check("estimate", &name, &answer(ours), case.get("estimate").unwrap());
}

#[test]
fn the_run_surface_matches_the_v2_goldens() {
    let Ok(root) = std::env::var("ZLC_PULSE_RUN_GOLDEN") else { return };
    let root = std::path::Path::new(&root);
    let mut tally = Tally::default();
    let mut cases = 0;
    for group in ["examples", "diff", "synthetic", "fuzz_deployed", "fuzz_small"] {
        let Ok(text) = std::fs::read_to_string(root.join(format!("{group}.json"))) else { continue };
        let document = json::parse(&text, "golden").unwrap();
        for case in document.get("cases").unwrap().as_array().unwrap() { run_case(case, &mut tally); cases += 1; }
    }
    // Line D's documents, read from their files: description and an unscanned FIRE's timing.
    let documents = json::parse(&std::fs::read_to_string(root.join("documents.json")).unwrap(), "documents").unwrap();
    for document in documents.get("documents").unwrap().as_array().unwrap() {
        let name = document.get("name").and_then(Json::as_str).unwrap().to_owned();
        let path = std::path::PathBuf::from(document.get("document").and_then(Json::as_str).unwrap());
        let described = describe_document(&path).map(|description| {
            let Json::Object(items) = description else { unreachable!() };
            Json::Object(items.into_iter().filter(|(key, _)| key != "path" && key != "name").collect())
        });
        tally.check("document describe", &name, &comparable(&answer(described)), &comparable(document.get("describe").unwrap()));
        let (_path, sequence) = zlc_pulse::run::read_document(&path).unwrap();
        tally.check("document timing", &name, &answer(receipt(&sequence, &[], None, &StreamerParams::default(), None)), document.get("unscanned").unwrap());
        cases += 1;
    }
    for (stage, count) in &tally.compared {
        eprintln!("{stage:>17}: {count} compared, {} different", tally.failed.get(stage).copied().unwrap_or(0));
    }
    eprintln!("{cases} cases");
    assert!(tally.failed.is_empty(), "golden mismatches");
}
