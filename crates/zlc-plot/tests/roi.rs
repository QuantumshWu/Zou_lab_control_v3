//! The coordinate ROI against `zlc_science.roi`: every refusal by class and
//! message, every planned output schema, and every applied snapshot (segment
//! placements, chunk schemas, planes, validity, sigma, window) as
//! `ROIPlan.apply` then `NativeDataset.from_science` produced it.
//!
//! Committed: `tests/data/roi.json.gz` (`gen/roi_cases.py OUT`). The camera
//! frame case (`gen/roi_cases.py OUT --large`) and every ROI effect of the
//! recorded corpus sessions (`gen/roi_cases.py OUT --corpus CORPUS_DIR`,
//! planned from the document's own effect metadata) are checked when
//! `ZLC_ROI_GOLDEN` names them (several separated by `;`).

use std::sync::Arc;

use base64::Engine;
use serde_json::Value as Json;
use sha2::{Digest, Sha256};
use zlc_archive::json::{dumps, parse_metadata, Style};
use zlc_archive::tree::{schema_from_tree, schema_tree};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::restrict::Source;
use zlc_core::schema::DatasetSchema;
use zlc_core::OwnedBuffer;
use zlc_plot::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange, SelectorState, SelectorValue};
use zlc_plot::kinds::{AxisDomain, AxisRef, PlotKind};
use zlc_plot::projection::nd::Scalar;
use zlc_plot::projection::output::SelectionSubject;
use zlc_plot::roi::{plan_from_metadata, RoiPlan};

fn load(path: &std::path::Path) -> Json {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn schema(text: &Json) -> Arc<DatasetSchema> { schema_from_tree(&parse_metadata(text.as_str().unwrap()).unwrap()).unwrap() }
fn tree(schema: &DatasetSchema) -> String { dumps(&schema_tree(schema), Style::Figure).unwrap() }
fn bytes(text: &Json) -> Vec<u8> { base64::engine::general_purpose::STANDARD.decode(text.as_str().unwrap()).unwrap() }
fn floats(text: &Json) -> Vec<f64> { bytes(text).chunks_exact(8).map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap())).collect() }
fn digest(data: &[u8]) -> String { format!("{:x}", Sha256::digest(data)) }

fn source(golden: &Json) -> Source {
    let full = schema(&golden["schema"]);
    let blocks: Vec<([usize; 2], Arc<DataBlock>)> = golden["segments"].as_array().unwrap().iter().map(|segment| {
        let chunk = if golden["kind"] == "dense" { full.clone() } else { schema(&segment["schema"]) };
        let validity = match &segment["validity"] {
            Json::Bool(flag) => Validity::Uniform(*flag),
            text => Validity::Mask(OwnedBuffer::from_vec(bytes(text).into_iter().map(|byte| byte != 0).collect())),
        };
        let sigma = (!segment["sigma"].is_null()).then(|| OwnedBuffer::from_vec(floats(&segment["sigma"])));
        let origin = [segment["origin"][0].as_u64().unwrap() as usize, segment["origin"][1].as_u64().unwrap() as usize];
        let values = zlc_archive::npy::numeric_from_bytes(&bytes(&segment["values"]), chunk.dtype(), false);
        (origin, DataBlock::new(chunk, values, validity, sigma).unwrap())
    }).collect();
    if golden["kind"] == "dense" { return Source::Dense(blocks[0].1.clone()); }
    let window = golden["window"].as_array().map(|pair| (i128::from(pair[0].as_i64().unwrap()), pair[1].as_u64().unwrap()));
    Source::Placed { schema: full, segments: blocks, window }
}

fn axis(value: &Json) -> Option<AxisRef> {
    value.as_array().map(|pair| AxisRef::new(AxisDomain::parse(pair[0].as_str().unwrap()).unwrap(), pair[1].as_str().unwrap()).unwrap())
}

fn subject(golden: &Json) -> SelectionSubject {
    let text = |value: &Json| value.as_str().map(Arc::<str>::from);
    SelectionSubject {
        plot_kind: PlotKind::parse(golden["plot_kind"].as_str().unwrap()).unwrap(),
        x: axis(&golden["x"]), y: axis(&golden["y"]),
        x_coordinate_frame: text(&golden["x_coordinate_frame"]), y_coordinate_frame: text(&golden["y_coordinate_frame"]),
        scope: golden["scope"].as_array().unwrap().iter().map(|term| {
            let value = &term[1];
            let scalar = if let Some(text) = value.get("text") { Scalar::Text(text.as_str().unwrap().into()) }
                else if let Some(integer) = value.get("int") { Scalar::Int(integer.as_str().unwrap().parse().unwrap()) }
                else if let Some(bits) = value.get("float") { Scalar::Float(f64::from_bits(bits.as_u64().unwrap())) } else { Scalar::None };
            (axis(&term[0]).unwrap(), scalar)
        }).collect(),
        source_window: golden["source_window"].as_u64().map(|window| window as usize),
    }
}

fn selector(golden: &Json) -> SelectorState {
    let real = |value: &Json| f64::from_bits(value.as_u64().unwrap());
    let range = |value: &Json| NumericRange::new(real(&value[0]), real(&value[1])).unwrap();
    let value = &golden["value"];
    let selected = match golden["kind"].as_str().unwrap() {
        "x_range" => SelectorValue::XRange(range(value)),
        "area" => SelectorValue::Area(RectangleRange { x: range(&value["x"]), y: range(&value["y"]) }),
        "crosshair" => SelectorValue::Crosshair(CrosshairPoint::new(real(&value[0]), real(&value[1])).unwrap()),
        _ => SelectorValue::Threshold(real(value)),
    };
    SelectorState::new(selected, golden["revision"].as_u64().unwrap(), golden["facet_index"].as_u64()).unwrap()
}

fn check(golden: &Json) -> (usize, Vec<String>) {
    let mut failures = Vec::new();
    let cases = golden["cases"].as_array().unwrap();
    for (index, case) in cases.iter().enumerate() {
        let input = &golden["sources"][case["source"].as_str().unwrap()];
        let source_schema = schema(&input["schema"]);
        // Planned from the subject and selector, and from the effect
        // metadata text the host is handed for them.
        let mut plans = Vec::new();
        if case.get("subject").is_some() { plans.push(("", RoiPlan::create(&source_schema, &subject(&case["subject"]), &selector(&case["selector"])))); }
        if let Some(metadata) = case.get("metadata") { plans.push((" from metadata", plan_from_metadata(&source_schema, metadata.as_str().unwrap()))); }
        for (route, plan) in plans {
            let label = format!("case {index} ({}){route}", case["source"].as_str().unwrap());
            let plan = match (plan, case.get("error")) {
                (Err(error), Some(expected)) => {
                    if serde_json::json!([error.kind.python_name(), error.message]) != *expected { failures.push(format!("{label}: {error:?} vs {expected}")); }
                    continue;
                }
                (Err(error), None) => { failures.push(format!("{label}: refused {error:?}")); continue; }
                (Ok(_), Some(expected)) => { failures.push(format!("{label}: planned, expected {expected}")); continue; }
                (Ok(plan), None) => plan,
            };
            if tree(plan.schema()) != case["schema"].as_str().unwrap() { failures.push(format!("{label}: planned schema differs")); }
            let output = match plan.apply_source(&source(input)) { Ok(output) => output, Err(error) => { failures.push(format!("{label}: apply {error:?}")); continue; } };
            if !Arc::ptr_eq(output.schema(), plan.schema()) { failures.push(format!("{label}: output schema is not the plan's")); }
            let window = case["output"]["window"].as_array().map(|pair| (i128::from(pair[0].as_i64().unwrap()), pair[1].as_u64().unwrap()));
            if output.window() != window { failures.push(format!("{label}: window {:?}", output.window())); }
            let expected = case["output"]["segments"].as_array().unwrap();
            let segments: Vec<_> = output.segments().collect();
            if segments.len() != expected.len() { failures.push(format!("{label}: {} segments, expected {}", segments.len(), expected.len())); continue; }
            for (position, (segment, want)) in segments.iter().zip(expected).enumerate() {
                let pair = |value: &Json| [value[0].as_u64().unwrap() as usize, value[1].as_u64().unwrap() as usize];
                let data = segment.data();
                if segment.origin() != pair(&want["origin"]) || segment.extent() != pair(&want["extent"]) { failures.push(format!("{label}[{position}] placement")); }
                if tree(data.schema()) != want["schema"].as_str().unwrap() { failures.push(format!("{label}[{position}] chunk schema differs")); }
                if digest(data.values().as_bytes()) != want["values"].as_str().unwrap() { failures.push(format!("{label}[{position}] values differ")); }
                let validity_ok = match (data.validity(), &want["validity"]) {
                    (Validity::Uniform(flag), Json::Bool(expected)) => flag == expected,
                    (Validity::Mask(mask), Json::Object(expected)) => digest(mask.as_bytes()) == expected["sha256"].as_str().unwrap(),
                    _ => false,
                };
                if !validity_ok { failures.push(format!("{label}[{position}] validity differs")); }
                let sigma_ok = match (data.sigma(), want["sigma"].as_str()) { (None, None) => true, (Some(sigma), Some(expected)) => digest(sigma.as_bytes()) == expected, _ => false };
                if !sigma_ok { failures.push(format!("{label}[{position}] sigma differs")); }
            }
        }
    }
    (cases.len(), failures)
}

#[test]
fn coordinate_rois_plan_and_cut_like_python() {
    let mut golden = vec![load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/roi.json.gz"))];
    if let Ok(paths) = std::env::var("ZLC_ROI_GOLDEN") {
        golden.extend(paths.split(';').filter(|path| !path.is_empty()).map(|path| load(std::path::Path::new(path))));
    }
    let mut total = 0;
    let mut failures = Vec::new();
    for golden in &golden {
        let (count, mut found) = check(golden);
        total += count;
        failures.append(&mut found);
    }
    assert!(failures.is_empty(), "{} of {total} ROI cases differ:\n{}", failures.len(), failures.join("\n"));
    assert!(total >= 43);
}
