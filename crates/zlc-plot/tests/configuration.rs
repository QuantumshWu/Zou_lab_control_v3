//! The configuration layer against zlc_plot: parameter schemas and their
//! probes, transitions, the default table, history layouts, semantic
//! descriptions and composition, rebasing, the recipe codec, thresholds,
//! interaction identities, and whole resolve_configuration sequences --
//! every observable field, floats bit for bit, errors by exception class.
//!
//! Committed: `tests/data/config.json.gz` (`gen/config_cases.py OUT.json.gz`)
//! and `tests/data/config_replay.json.gz` (`gen/config_replay.py`, every
//! resolve call of the recorded sessions) and `tests/data/config_fuzz.json.gz`
//! (`gen/config_fuzz.py OUT.json.gz 30 7`, mutated wire targets).  The full synthetic set
//! (`config_cases.py OUT --full`) is checked when `ZLC_CONFIG_GOLDEN` names
//! it (several paths separated by `;`).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use serde_json::{json, Map, Value as Json};
use zlc_core::schema::{Axis, CodeBase, CodeMapping, Coordinate, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::{DType, OwnedBuffer};
use zlc_plot::configuration::{self as cfg, *};
use zlc_plot::configuration::data_contract::{indexed_history_layout as layout_of, schemas_continue};
use zlc_plot::configuration::recipe::{decode_selectors, decode_spec, encode_spec, selectors_document};
use zlc_plot::configuration::selectors::thresholds_wire;
use zlc_plot::configuration::semantics::CycleChoices;
use zlc_plot::configuration::state::{InteractionUpdate, PresentationUpdate};

// ------------------------------------------------------------------ codec
fn dec(value: &Json) -> Value {
    match value {
        Json::Null => Value::None,
        Json::Bool(flag) => Value::Bool(*flag),
        Json::Number(number) => match number.as_i64() {
            Some(integer) => Value::Int(i128::from(integer)),
            None => Value::Int(i128::from(number.as_u64().expect("integer"))),
        },
        Json::String(text) => Value::text(text.as_str()),
        Json::Array(items) => Value::list(items.iter().map(dec)),
        Json::Object(map) => {
            if let (1, Some(bits)) = (map.len(), map.get("$f")) { return Value::Float(f64::from_bits(bits.as_u64().unwrap())); }
            if let (1, Some(text)) = (map.len(), map.get("$i")) { return Value::Int(text.as_str().unwrap().parse().unwrap()); }
            Value::Map(map.iter().map(|(key, value)| (key.as_str(), dec(value))).collect())
        }
    }
}

fn dec_map(value: &Json) -> ValueMap { dec(value).as_map().cloned().unwrap_or_else(|| panic!("mapping expected: {value}")) }

fn enc(value: &Value) -> Json {
    match value {
        Value::None => Json::Null,
        Value::Bool(flag) => Json::Bool(*flag),
        Value::Int(integer) => match i64::try_from(*integer) { Ok(v) => json!(v), Err(_) => match u64::try_from(*integer) { Ok(v) => json!(v), Err(_) => json!({"$i": integer.to_string()}) } },
        Value::Float(real) => json!({"$f": real.to_bits()}),
        Value::Text(text) => Json::String(text.to_string()),
        Value::List(items) => Json::Array(items.iter().map(enc).collect()),
        Value::Map(map) => Json::Object(map.iter().map(|(key, value)| (key.to_owned(), enc(value))).collect()),
    }
}

/// Plain JSON (as `json.loads` gives it) in the golden encoding.
fn enc_plain(value: &Json) -> Json {
    match value {
        Json::Number(number) if number.is_f64() => json!({"$f": number.as_f64().unwrap().to_bits()}),
        Json::Array(items) => Json::Array(items.iter().map(enc_plain).collect()),
        Json::Object(map) => Json::Object(map.iter().map(|(key, value)| (key.clone(), enc_plain(value))).collect()),
        other => other.clone(),
    }
}

fn bits(value: &Json) -> f64 { f64::from_bits(value.as_u64().expect("float bits")) }
fn error_name(error: &PlotError) -> Json { json!({"error": error.python_name()}) }
fn outcome(result: Result<Json>) -> Json { match result { Ok(value) => json!({"ok": value}), Err(error) => error_name(&error) } }

// ---------------------------------------------------------------- schemas
fn coordinates(doc: &Json) -> Coordinates {
    let map = doc.as_object().unwrap();
    if let Some(origin) = map.get("index") {
        return Coordinates::Index { origin: i128::from(origin.as_i64().unwrap()), len: map["len"].as_u64().unwrap() as usize };
    }
    if let Some(origins) = map.get("origins") {
        let floats = |key: &str| OwnedBuffer::from_vec(map[key].as_array().unwrap().iter().map(bits).collect::<Vec<f64>>());
        let _ = origins;
        return Coordinates::OriginsAndOffsets { origins: floats("origins"), offsets: floats("offsets") };
    }
    if let Some(values) = map.get("f64") { return Coordinates::Numeric(OwnedBuffer::from_vec(values.as_array().unwrap().iter().map(bits).collect::<Vec<f64>>()).into()); }
    if let Some(values) = map.get("i64") { return Coordinates::Numeric(OwnedBuffer::from_vec(values.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect::<Vec<i64>>()).into()); }
    if let Some(values) = map.get("u64") { return Coordinates::Numeric(OwnedBuffer::from_vec(values.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect::<Vec<u64>>()).into()); }
    Coordinates::Values(map["values"].as_array().unwrap().iter().map(|value| match dec(value) {
        Value::Int(integer) => Coordinate::Integer(integer), Value::Float(real) => Coordinate::Real(real),
        Value::Text(text) => Coordinate::Text(text), _ => Coordinate::Missing,
    }).collect::<Vec<_>>().into())
}

fn text_option(value: &Json) -> Option<Arc<str>> { value.as_str().map(Arc::from) }

fn axis(doc: &Json) -> Arc<Axis> {
    Axis::new(Arc::from(doc["id"].as_str().unwrap()), Arc::from(doc["name"].as_str().unwrap()), Arc::from(doc["role"].as_str().unwrap()),
              coordinates(&doc["coordinates"]), text_option(&doc["unit"]), text_option(&doc["frame"]),
              doc["labels"].as_array().map(|labels| labels.iter().map(|label| Arc::from(label.as_str().unwrap())).collect()),
              text_option(&doc["coordinate_of"])).unwrap()
}

fn domain(doc: &Json) -> Arc<Domain> {
    let shape: Vec<usize> = doc["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let axes: Vec<Arc<Axis>> = doc["axes"].as_array().unwrap().iter().map(axis).collect();
    match doc["codes"].as_array() {
        None => Domain::dense(shape, axes).unwrap(),
        Some(codes) => {
            let mappings = codes.iter().map(|code| {
                let (inner, outer) = (code["inner"].as_u64().unwrap() as usize, code["outer"].as_u64().unwrap() as usize);
                let base = match code.get("range") {
                    Some(range) => CodeBase::Range { start: range[0].as_u64().unwrap(), step: range[1].as_i64().unwrap(), len: range[2].as_u64().unwrap() as usize },
                    None => CodeBase::Explicit(OwnedBuffer::from_vec(code["explicit"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect())),
                };
                CodeMapping::new(base, inner, outer).unwrap()
            }).collect();
            Domain::new(shape, axes, mappings).unwrap()
        }
    }
}

fn dtype(name: &str) -> DType {
    match name {
        "bool" => DType::Bool, "uint8" => DType::U8, "int8" => DType::I8, "uint16" => DType::U16, "int16" => DType::I16,
        "uint32" => DType::U32, "int32" => DType::I32, "uint64" => DType::U64, "int64" => DType::I64, "float16" => DType::F16,
        "float32" => DType::F32, "float64" => DType::F64, "complex64" => DType::C64, "complex128" => DType::C128,
        other => panic!("dtype {other}"),
    }
}

fn schema(doc: &Json) -> Arc<PlotSchema> {
    let value = &doc["value"];
    let validity = match value["validity"].as_array() {
        None => ValidityContract::Value,
        Some(ids) => ValidityContract::Components(ids.iter().map(|id| Arc::from(id.as_str().unwrap())).collect()),
    };
    let value = ValueSchema::new(dtype(value["dtype"].as_str().unwrap()), text_option(&value["unit"]), text_option(&value["name"]), validity).unwrap();
    PlotSchema::new(DatasetSchema::new(domain(&doc["repeat"]), domain(&doc["point"]), domain(&doc["cell"]), value).unwrap())
}

// ------------------------------------------------------------------ specs
fn spec_from(doc: &Json) -> PlotSpec {
    if doc["kind"] == "pulse_timeline" {
        let labels = dec_map(&doc["labels"]);
        let slot = |name: &str| labels.get(name).and_then(Value::as_str).map(Arc::from);
        return PlotSpec::PulseTimeline(PulseTimelinePlot { labels: PlotLabels::new(slot("title"), slot("x"), slot("y"), slot("value")).unwrap() });
    }
    decode_spec(&dec(doc)).unwrap_or_else(|error| panic!("spec {doc}: {error}"))
}

fn spec_doc(spec: &PlotSpec) -> Json {
    match spec {
        PlotSpec::PulseTimeline(pulse) => {
            let text = |label: &Option<Arc<str>>| label.as_ref().map_or(Json::Null, |t| Json::String(t.to_string()));
            json!({"kind": "pulse_timeline", "labels": {"title": text(&pulse.labels.title), "x": text(&pulse.labels.x),
                   "y": text(&pulse.labels.y), "value": text(&pulse.labels.value)}})
        }
        other => enc(&Value::Map(encode_spec(other).unwrap())),
    }
}

fn axis_doc(axis: &AxisRef) -> Json { json!({"domain": axis.domain.value(), "axis_id": axis.axis_id}) }
fn axis_option(axis: &Option<AxisRef>) -> Json { axis.as_ref().map_or(Json::Null, axis_doc) }

fn cycle_doc(cycle: &CycleChoices, value: &Value) -> Json {
    let count = cycle.len();
    let position = cycle.position_of(value);
    let mut samples: Vec<usize> = [Some(0), Some(1), Some(count / 2), count.checked_sub(1), position].into_iter().flatten().filter(|i| *i < count).collect();
    samples.sort_unstable();
    samples.dedup();
    json!({"len": count, "position": position, "samples": samples.iter().map(|i| {
        let (fate, label) = cycle.get(*i).unwrap();
        json!([i, enc(&fate), label])
    }).collect::<Vec<_>>()})
}

fn describe_doc(description: &SemanticDescription) -> Json {
    json!({
        "kind": description.kind.value(),
        "kind_choices": description.kind_choices.iter().map(|kind| kind.value()).collect::<Vec<_>>(),
        "fields": description.fields.iter().map(|field| json!({
            "name": field.name, "label": field.label, "value": enc(&field.value),
            "choices": field.choices.iter().map(|(value, label)| json!([enc(value), label])).collect::<Vec<_>>(),
            "required": field.required, "cycle": field.cycle_choices.as_ref().map_or(Json::Null, |cycle| cycle_doc(cycle, &field.value)),
        })).collect::<Vec<_>>(),
        "axis_choices": description.axis_choices.iter().map(axis_doc).collect::<Vec<_>>(),
        "x": axis_option(&description.x), "y": axis_option(&description.y), "group": axis_option(&description.group),
        "reduction": description.reduction.map(|r| r.value()), "facet": axis_option(&description.facet),
        "facet_max_cells": description.facet_max_cells,
        "fate_rows": description.fate_rows.iter().map(|(axis, name)| json!([axis_doc(axis), name])).collect::<Vec<_>>(),
    })
}

fn parameter_schema_doc(schema: &ParameterSchema) -> Json {
    Json::Array(schema.specs().iter().map(|spec| json!({
        "name": spec.name,
        "type": match spec.value_type { ValueType::Bool => "bool", ValueType::Number => "number", ValueType::Int => "int", ValueType::Text => "text" },
        "effects": spec.effects.bits(), "default": enc(&spec.default), "allow_none": spec.allow_none, "label": spec.label,
        "choices": spec.choices.iter().map(|(value, label)| json!([enc(value), label])).collect::<Vec<_>>(),
        "minimum": spec.minimum.map_or(Json::Null, |v| enc(&Value::Float(v))), "maximum": spec.maximum.map_or(Json::Null, |v| enc(&Value::Float(v))),
        "step": spec.step.map_or(Json::Null, |v| enc(&Value::Float(v))), "portable": spec.portable, "segmented": spec.segmented,
    })).collect())
}

fn viewport_doc(viewport: &Option<Viewport>) -> Json {
    match viewport {
        None => Json::Null,
        Some((x, y)) => Json::Array([x, y].iter().map(|range| range.map_or(Json::Null, |r| json!([enc(&Value::Float(r.low)), enc(&Value::Float(r.high))]))).collect()),
    }
}

fn viewport_from(doc: &Json) -> Option<Viewport> {
    let items = doc.as_array()?;
    let range = |slot: &Json| slot.as_array().map(|pair| NumericRange::new(bits(&pair[0]["$f"]), bits(&pair[1]["$f"])).unwrap());
    Some((range(&items[0]), range(&items[1])))
}

// --------------------------------------------------------------- fit layer
struct RecordedFit {
    catalogue: HashMap<String, Vec<Arc<FitModelInfo>>>,
    targets: Vec<(Json, Json)>,
    expressions: Vec<(Json, Json)>,
    missing: Mutex<Vec<Json>>,
}

fn relation(name: &str) -> UnitRelation { if name == "axis_1" { UnitRelation::Axis1 } else { UnitRelation::Axis0 } }

impl RecordedFit {
    fn new(data: &Json) -> Self {
        let mut catalogue = HashMap::new();
        for (target, models) in data["catalogue"].as_object().unwrap() {
            catalogue.insert(target.clone(), models.as_array().unwrap().iter().map(|model| Arc::new(FitModelInfo {
                model_id: model["model_id"].as_str().unwrap().into(), display_name: model["display_name"].as_str().unwrap().into(),
                independent_arity: model["independent_arity"].as_u64().unwrap() as usize,
                coordinate_relations: model["coordinate_relations"].as_array().unwrap().iter().map(|r| relation(r.as_str().unwrap())).collect(),
                parameter_names: model["parameter_names"].as_array().unwrap().iter().map(|n| n.as_str().unwrap().into()).collect(),
                parameter_symbols: Vec::new(),
            })).collect());
        }
        let pairs = |key: &str| data["fit"][key].as_array().unwrap().iter().map(|entry| (entry["query"].clone(), entry["answer"].clone())).collect();
        Self { catalogue, targets: pairs("targets"), expressions: pairs("expressions"), missing: Mutex::new(Vec::new()) }
    }
    fn model(&self, id: &str) -> Arc<FitModelInfo> {
        self.catalogue.values().flatten().find(|model| model.model_id == id).cloned().unwrap_or_else(|| panic!("model {id}"))
    }
    fn answer<'a>(&'a self, records: &'a [(Json, Json)], query: &Json) -> Result<&'a Json> {
        match records.iter().find(|(recorded, _)| recorded == query) {
            Some((_, answer)) => {
                if let Some(name) = answer.get("error") { return Err(error_of(name.as_str().unwrap())); }
                Ok(answer)
            }
            None => {
                self.missing.lock().unwrap().push(query.clone());
                Err(PlotError::runtime("unrecorded fit query"))
            }
        }
    }
}

fn error_of(name: &str) -> PlotError {
    let kind = match name {
        "TypeError" => ErrorKind::Type, "KeyError" => ErrorKind::Key, "IndexError" => ErrorKind::Index, "UnitError" => ErrorKind::Unit,
        "OverflowError" => ErrorKind::Overflow, "RuntimeError" => ErrorKind::Runtime, _ => ErrorKind::Value,
    };
    PlotError::new(kind, name)
}

fn units_doc(units: &FitUnits) -> Json {
    let quantity = |q: &Option<FitQuantity>| q.as_ref().map_or(Json::Null, |q| json!([q.canonical.symbol, q.display.symbol]));
    json!({"value": quantity(&units.value), "axis_0": quantity(&units.axis_0), "axis_1": quantity(&units.axis_1),
           "histogram": units.histogram, "rolling_primary": units.rolling_primary})
}

impl FitLayer for RecordedFit {
    fn models_for(&self, target: FitTargetKind) -> Vec<Arc<FitModelInfo>> { self.catalogue[target.value()].clone() }
    fn resolve_target(&self, query: &FitTargetQuery<'_>) -> Result<FitResolution> {
        let doc = json!({
            "target": enc(&Value::Map(query.target.clone())), "facet_grid": query.facet_grid, "units": units_doc(query.units),
            "live": query.live, "previous_model": query.previous_request.map(|request| request.model.model_id.clone()),
            "previous_failure": query.previous_expression_failure.map(|(a, b)| json!([a, b])),
        });
        let answer = self.answer(&self.targets, &doc)?;
        Ok(FitResolution {
            request: answer["request"].as_object().map(|request| FitRequest {
                model: self.model(request["model"].as_str().unwrap()), target: dec_map(&request["target"]), native: None,
            }),
            expression_failure: answer["failure"].as_array().map(|pair| (pair[0].as_str().unwrap().into(), pair[1].as_str().unwrap().into())),
        })
    }
    fn expression_text(&self, units: &FitUnits, model: &FitModelInfo, target: &ValueMap) -> Result<String> {
        let doc = json!({"units": units_doc(units), "model": model.model_id, "target": enc(&Value::Map(target.clone()))});
        Ok(self.answer(&self.expressions, &doc)?["text"].as_str().unwrap().to_owned())
    }
}

// ---------------------------------------------------------- configuration
fn configuration_doc(configuration: &ResolvedConfiguration, signatures: &mut Vec<Signature>, live_window: bool) -> Result<Json> {
    let index = signatures.iter().position(|signature| *signature == configuration.signature).unwrap_or(signatures.len());
    signatures.push(configuration.signature.clone());
    let state = &configuration.display_state;
    let metadata = &configuration.metadata;
    let mut describe = describe_panel(configuration, live_window)?;
    // A model's parameters with their symbols are the console's (v2 workbench panel_state), not the scientific package's describe.
    for model in describe["fit"]["models"].as_array_mut().into_iter().flatten() { model.as_object_mut().map(|model| model.remove("parameters")); }
    // A live panel's Window is the console's binding row, held where the spec declares it; the scientific package leaves it out.
    if let Some(controls) = describe["controls"].as_array_mut() { controls.retain(|control| control.get("bound").is_none()); }
    Ok(json!({
        "spec": spec_doc(&configuration.spec),
        "semantics": describe_doc(&configuration.semantics),
        "parameters": configuration.parameter_schema.names().collect::<Vec<_>>(),
        "display": {"revision": state.revision, "values": enc(&Value::Map(state.values.clone())),
                    "changed": state.changed_names.iter().collect::<Vec<_>>(), "effects": state.effects.bits(), "interaction": enc(&state.interaction.wire())},
        "size": configuration.size,
        "selectors": enc(&selectors_document(&configuration.selectors)),
        "viewport": viewport_doc(&configuration.viewport),
        "facet_focus": configuration.facet_focus.map(|v| v as i64), "focused_facet_index": configuration.focused_facet_index.map(|v| v as i64),
        "thresholds": enc(&thresholds_wire(&configuration.classifier_thresholds)),
        "fit_target": enc(&Value::Map(configuration.fit_target.clone())), "fit_live": configuration.fit_live,
        "presentation": enc(&configuration.presentation.wire()),
        "ignored": configuration.ignored_interaction, "same_source": configuration.same_source,
        "discard_unmatched_classifier": configuration.discard_unmatched_classifier,
        "expression_failure": configuration.expression_failure.as_ref().map(|(a, b)| json!([a, b])),
        "has_numerical_work": configuration.has_numerical_work(), "science_changed": configuration.science_changed,
        "metadata": {
            "fit_models": metadata.fit_models.iter().map(|model| model.model_id.clone()).collect::<Vec<_>>(),
            "parameter_choices": Json::Object(metadata.parameter_choices.iter().map(|(name, values)| (name.to_string(), enc(&Value::list(values.iter().cloned())))).collect()),
            "automatic_values": Json::Object(metadata.automatic_values.iter().map(|(name, value)| (name.to_string(), enc(value))).collect()),
        },
        "fit_expression": configuration.fit_expression()?, "fit_expression_error": configuration.fit_expression_error(),
        "target": enc(&Value::Map(configuration.target()?)),
        "signature_class": index,
        "image_overlay": configuration.image_overlay.is_some(),
        "describe": enc_plain(&describe),
    }))
}

fn selector_updates(doc: &Json) -> Vec<(SelectorKind, Option<SelectorState>)> {
    doc.as_array().unwrap().iter().map(|pair| {
        let kind = SelectorKind::parse(&dec(&pair[0])).unwrap();
        let state = (!pair[1].is_null()).then(|| decode_selectors(&Value::list([dec(&pair[1])])).unwrap()[0]);
        (kind, state)
    }).collect()
}

fn wrapped<T>(doc: &Map<String, Json>, key: &str, read: impl Fn(&Json) -> T) -> Option<T> { doc.get(key).map(|wrapper| read(&wrapper["value"])) }

fn typed_updates(doc: &Json, overlays: &[OverlayHandle]) -> ConfigurationUpdates {
    let doc = doc.as_object().unwrap();
    ConfigurationUpdates {
        semantic: wrapped(doc, "semantic", |value| dec(value).as_map().cloned().unwrap_or_default()),
        parameters: doc.get("parameters").map(dec_map),
        parameter_updates: doc.get("parameter_updates").map(dec_map),
        size: doc.get("size").map(|size| size.as_str().unwrap().to_owned()),
        selectors: doc.get("selectors").map(|states| decode_selectors(&dec(states)).unwrap()),
        selector_updates: doc.get("selector_updates").map(selector_updates),
        viewport: wrapped(doc, "viewport", viewport_from),
        facet_focus: wrapped(doc, "facet_focus", |value| value.as_i64().map(i128::from)),
        focused_facet_index: wrapped(doc, "focused_facet_index", |value| value.as_i64().map(i128::from)),
        classifier_thresholds: doc.get("classifier_thresholds").map(dec),
        fit: wrapped(doc, "fit", |value| dec(value).as_map().cloned().unwrap_or_default()),
        fit_live: doc.get("fit_live").map(|live| live.as_bool().unwrap()),
        interaction: doc.get("interaction").map(|value| InteractionUpdate::from_value(&dec(value)).unwrap()),
        presentation: doc.get("presentation").map(|value| PresentationUpdate::from_value(&dec(value)).unwrap()),
        image_overlay: wrapped(doc, "image_overlay", |value| value.as_u64().map(|token| overlays[token as usize].clone())),
        overlay_follows_kind: false,
    }
}

/// Collected disagreements; the test fails listing the first few.
#[derive(Default)]
struct Report { checked: usize, failures: Vec<String> }

impl Report {
    fn compare(&mut self, section: &str, context: &dyn Fn() -> String, got: &Json, want: &Json) {
        self.checked += 1;
        if got != want {
            let detail = first_difference(got, want, String::new()).unwrap_or_default();
            self.failures.push(format!("{section}: {}\n    at {detail}", context()));
        }
    }
    fn finish(self, name: &str) {
        eprintln!("{name}: {} checks, {} mismatches", self.checked, self.failures.len());
        let mut sections: Vec<(String, usize)> = Vec::new();
        for failure in &self.failures {
            let section = failure.split(':').next().unwrap_or("").to_owned();
            match sections.iter_mut().find(|(held, _)| *held == section) { Some(slot) => slot.1 += 1, None => sections.push((section, 1)) }
        }
        for (section, count) in &sections { eprintln!("  {section}: {count}"); }
        let limit: usize = std::env::var("ZLC_CONFIG_SHOW").ok().and_then(|v| v.parse().ok()).unwrap_or(12);
        let only = std::env::var("ZLC_CONFIG_ONLY").ok();
        if !self.failures.is_empty() {
            for failure in self.failures.iter().filter(|f| only.as_ref().is_none_or(|o| f.starts_with(o.as_str()))).take(limit) { eprintln!("  {failure}"); }
            panic!("{name}: {} of {} disagree", self.failures.len(), self.checked);
        }
    }
}

fn first_difference(got: &Json, want: &Json, path: String) -> Option<String> {
    if got == want { return None; }
    match (got, want) {
        (Json::Object(a), Json::Object(b)) => {
            for key in a.keys().chain(b.keys()) {
                let (x, y) = (a.get(key).unwrap_or(&Json::Null), b.get(key).unwrap_or(&Json::Null));
                if a.contains_key(key) != b.contains_key(key) { return Some(format!("{path}.{key}: present here {} / in Python {}", a.contains_key(key), b.contains_key(key))); }
                if let Some(found) = first_difference(x, y, format!("{path}.{key}")) { return Some(found); }
            }
            None
        }
        (Json::Array(a), Json::Array(b)) if a.len() == b.len() => {
            a.iter().zip(b).enumerate().find_map(|(index, (x, y))| first_difference(x, y, format!("{path}[{index}]")))
        }
        _ => {
            let clip = |value: &Json| { let text = value.to_string(); if text.len() > 300 { format!("{}...", &text[..300]) } else { text } };
            Some(format!("{path}: here {} / Python {}", clip(got), clip(want)))
        }
    }
}

struct Golden { data: Json, schemas: Vec<Arc<PlotSchema>>, fit: Arc<RecordedFit> }

impl Golden {
    fn load(path: &std::path::Path) -> Self {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let text = if path.extension().is_some_and(|ext| ext == "gz") {
            let mut text = String::new();
            std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text).unwrap();
            text
        } else { String::from_utf8(bytes).unwrap() };
        let data: Json = serde_json::from_str(&text).unwrap();
        let schemas = data["schemas"].as_array().unwrap().iter().map(schema).collect();
        let fit = Arc::new(RecordedFit::new(&data));
        Golden { data, schemas, fit }
    }
    fn schema(&self, id: &Json) -> Option<Arc<PlotSchema>> { id.as_u64().map(|id| self.schemas[id as usize].clone()) }
    fn cases(&self, key: &str) -> &[Json] { self.data[key].as_array().map_or(&[], |v| v.as_slice()) }
}

fn kind_cell(case: &Json) -> (PlotKind, Option<PlotKind>) {
    (PlotKind::parse(case["kind"].as_str().unwrap()).unwrap(), case["cell"].as_str().map(|c| PlotKind::parse(c).unwrap()))
}

fn check_parameters(golden: &Golden, report: &mut Report) {
    let style = zlc_plot::style::PlotStyle::default();
    let probes: Vec<Value> = golden.cases("probe_values").iter().map(dec).collect();
    for case in golden.cases("parameter_schemas") {
        let (kind, cell) = kind_cell(case);
        let schema = parameter_schema_for_kind(kind, &style, cell).unwrap();
        report.compare("parameter schema", &|| format!("{kind:?}/{cell:?}"), &parameter_schema_doc(&schema), &case["schema"]);
        for probe in case["probes"].as_array().unwrap() {
            let name = probe[0].as_str().unwrap();
            let spec = schema.get(name).unwrap();
            for (value, want) in probes.iter().zip(probe[1].as_array().unwrap()) {
                let got = outcome(spec.prepare(value).map(|v| enc(&v)));
                report.compare("probe", &|| format!("{kind:?}/{cell:?} {name} <- {}", value.py_repr()), &got, want);
            }
        }
    }
    for case in golden.cases("kind_errors") {
        let kind = case["kind"].as_str().and_then(PlotKind::parse);
        let cell = case["cell"].as_str().map(|c| PlotKind::parse(c).unwrap());
        let got = match kind {
            None => error_name(&PlotError::value("not a kind")),
            Some(kind) => outcome(parameter_schema_for_kind(kind, &style, cell).map(|schema| json!(schema.names().collect::<Vec<_>>()))),
        };
        report.compare("kind error", &|| case.to_string(), &got, &case["result"]);
    }
    if !golden.data["plot_kinds"].is_null() {
        report.compare("plot kinds", &|| String::new(), &plot_kinds(), &golden.data["plot_kinds"]);
    }
    if !golden.data["non_portable"].is_null() {
        let names: Vec<&str> = non_portable_display_names().into_iter().collect();
        report.compare("non portable", &|| String::new(), &json!(names), &golden.data["non_portable"]);
    }
    for case in golden.cases("transitions") {
        let (kind, cell) = kind_cell(case);
        let schema = parameter_schema_for_kind(kind, &style, cell).unwrap();
        let got = outcome(schema.prepare_transition(&dec_map(&case["current"]), &dec_map(&case["updates"])).map(|v| enc(&Value::Map(v))));
        report.compare("transition", &|| format!("{kind:?}/{cell:?} {}", case["updates"]), &got, &case["result"]);
    }
    for case in golden.cases("retained") {
        let (kind, cell) = kind_cell(case);
        let schema = parameter_schema_for_kind(kind, &style, cell).unwrap();
        let got = outcome(cfg::metadata::retained_parameters(&dec_map(&case["old"]), &schema).map(|v| enc(&Value::Map(v))));
        report.compare("retained", &|| format!("{kind:?}/{cell:?}"), &got, &case["result"]);
    }
    for case in golden.cases("authored_display") {
        let (kind, cell) = kind_cell(case);
        let got = outcome(validate_authored_display(kind, &dec_map(&case["values"]), &style, cell).map(|_| Json::Null));
        report.compare("authored display", &|| case.to_string(), &got, &case["result"]);
    }
}

fn check_specs(golden: &Golden, report: &mut Report) {
    for case in golden.cases("spec_cases") {
        let schema = golden.schema(&case["schema"]);
        let got = match case["call"].as_str().unwrap() {
            "fitting" => {
                let kind = case["kind"].as_str().map(|k| PlotKind::parse(k).unwrap());
                let cell = case["cell"].as_str().map(|c| PlotKind::parse(c).unwrap());
                outcome(fitting_spec(schema.as_deref(), kind, cell).map(|spec| spec.as_ref().map_or(Json::Null, spec_doc)))
            }
            _ => outcome(chosen_spec(schema.as_deref(), &spec_from(&case["current"])).map(|spec| spec.as_ref().map_or(Json::Null, spec_doc))),
        };
        report.compare("spec", &|| format!("schema {} {} {:?}/{:?}", case["schema"], case["call"], case["kind"], case["cell"]), &got, &case["result"]);
    }
    for case in golden.cases("history") {
        let schema = golden.schema(&case["schema"]).unwrap();
        let got = outcome(layout_of(schema.dataset()).map(|layout| layout.map_or(Json::Null, |l| json!({
            "shots": l.shot_count(), "inner_count": l.inner_count, "primary": &*l.primary_axis_id, "time": l.time_axis_id.as_deref(),
            "cells": l.cells, "event_size": l.event.event_size, "times": l.times.as_ref().map(|t| t.iter().map(|v| v.to_bits()).collect::<Vec<_>>()),
            "windows": ([0i128, 1, 2, 5, 100]).iter().map(|w| { let r = l.window_rows(*w); json!([w, r.start, r.end]) }).collect::<Vec<_>>(),
            "codes": (0..l.row_count().min(200)).map(|row| l.code(row).unwrap()).collect::<Vec<_>>(),
        }))));
        report.compare("history", &|| format!("schema {}", case["schema"]), &got, &case["result"]);
    }
    for case in golden.cases("continue") {
        let (a, b) = (golden.schema(&case[0]).unwrap(), golden.schema(&case[1]).unwrap());
        report.compare("continue", &|| format!("{} -> {}", case[0], case[1]), &outcome(schemas_continue(&a, &b).map(Json::Bool)), &case[2]);
    }
}

fn check_semantics(golden: &Golden, report: &mut Report) {
    let layout = zlc_plot::layout::LayoutConfig::default();
    for case in golden.cases("describe") {
        let schema = golden.schema(&case["schema"]);
        let spec = spec_from(&case["spec"]);
        let got = outcome(describe_semantics(schema.as_deref(), &spec, &layout).map(|d| describe_doc(&d)));
        report.compare("describe", &|| format!("schema {} spec {}", case["schema"], case["spec"]), &got, &case["result"]);
    }
    for case in golden.cases("compose") {
        let schema = golden.schema(&case["schema"]);
        let spec = spec_from(&case["spec"]);
        let got = outcome(composed_spec(schema.as_deref(), &spec, &dec_map(&case["values"])).map(|s| spec_doc(&s)));
        report.compare("compose", &|| format!("schema {} spec {} values {}", case["schema"], case["spec"], case["values"]), &got, &case["result"]);
    }
    for case in golden.cases("rebase") {
        let schema = golden.schema(&case["schema"]).unwrap();
        let kind = PlotKind::parse(case["kind"].as_str().unwrap()).unwrap();
        let got = outcome(rebased_spec(&schema, kind, &dec_map(&case["values"])).map(|s| spec_doc(&s)));
        report.compare("rebase", &|| format!("schema {} {} {}", case["schema"], case["kind"], case["values"]), &got, &case["result"]);
    }
}

fn check_records(golden: &Golden, report: &mut Report) {
    for case in golden.cases("recipes") {
        let got = match case["call"].as_str().unwrap() {
            "encode" => {
                let spec = spec_from(&case["spec"]);
                let selectors = [SelectorState::new(SelectorValue::XRange(NumericRange::new(0.0, 1.0).unwrap()), 3, None).unwrap(),
                                 SelectorState::new(SelectorValue::Crosshair(CrosshairPoint::new(1.0, 2.0).unwrap()), 0, None).unwrap()];
                let fit = ValueMap::from_pairs([("model", Value::text("gaussian_offset"))]);
                let parts = RecipeParts { viewport: Some((Some(NumericRange::new(2.0, -1.0).unwrap()), None)), fit: Some(&fit), selectors: &selectors, ..RecipeParts::default() };
                outcome(encode_plot_recipe(&spec, &dec_map(&case["parameters"]), " 4X4 ", &parts).map(|r| enc(&Value::Map(r))))
            }
            _ => outcome(decode_plot_recipe(&dec(&case["value"])).map(|d| json!({
                "spec": spec_doc(&d.spec), "parameters": enc(&Value::Map(d.parameters.clone())), "size": &*d.size,
                "viewport": viewport_doc(&d.viewport), "classifier_thresholds": enc(&d.classifier_thresholds), "facet_focus": d.facet_focus.map(|v| v as i64),
                "fit": enc(&Value::Map(d.fit.clone())), "selectors": enc(&selectors_document(&d.selectors)), "overlay": enc(&d.overlay),
                "interaction": enc(&update_doc_interaction(&d.interaction)), "presentation": enc(&update_doc_presentation(&d.presentation)),
            }))),
        };
        report.compare("recipe", &|| format!("{} {}", case["call"], case.get("value").or(case.get("spec")).unwrap()), &got, &case["result"]);
    }
    for case in golden.cases("thresholds") {
        let got = outcome(normalize_classifier_threshold_targets(&dec(&case["value"])).map(|targets| enc(&thresholds_wire(&targets))));
        report.compare("thresholds", &|| case["value"].to_string(), &got, &case["result"]);
    }
    for case in golden.cases("interactions") {
        let got = outcome(InteractionUpdate::from_value(&dec(&case["value"])).map(|update| enc(&update_doc_interaction(&update))));
        report.compare("interaction", &|| case["value"].to_string(), &got, &case["result"]);
    }
    for case in golden.cases("presentations") {
        let got = outcome(PresentationUpdate::from_value(&dec(&case["value"])).map(|update| enc(&update_doc_presentation(&update))));
        report.compare("presentation", &|| case["value"].to_string(), &got, &case["result"]);
    }
}

/// A normalized partial mapping: only the fields it carried.
fn update_doc_interaction(update: &InteractionUpdate) -> Value {
    match &update.series_lock {
        None => Value::Map(ValueMap::new()),
        Some(target) => Interaction { series_lock: target.clone() }.wire(),
    }
}

fn update_doc_presentation(update: &PresentationUpdate) -> Value {
    match &update.series_readout {
        None => Value::Map(ValueMap::new()),
        Some(readout) => Presentation { series_readout: readout.clone() }.wire(),
    }
}

fn check_sequences(golden: &Golden, report: &mut Report) {
    let (mut resolve_time, mut resolve_count) = (std::time::Duration::ZERO, 0usize);
    let overlays: Vec<OverlayHandle> = (0..golden.data["overlays"].as_u64().unwrap_or(0)).map(|_| Arc::new(()) as OverlayHandle).collect();
    for sequence in golden.cases("sequences") {
        let name = sequence["name"].as_str().unwrap();
        let mut configs: Vec<Option<Arc<ResolvedConfiguration>>> = Vec::new();
        let mut signatures = Vec::new();
        for (index, step) in sequence["steps"].as_array().unwrap().iter().enumerate() {
            let reference = |key: &str| step[key].as_u64().map(|i| configs[i as usize].clone().expect("successful step"));
            let (base, presented) = (reference("base"), reference("presented"));
            let target = step["target"].as_object().map(|_| dec_map(&step["target"]));
            let interaction = step["interaction_updates"].as_object().map(|doc| {
                let mut recipe = dec_map(&step["interaction_updates"]);
                recipe.remove("selector_updates");
                recipe.remove("focused_facet_index");
                InteractionUpdates { recipe, selector_updates: doc.get("selector_updates").map(selector_updates),
                                     focused_facet_index: doc.get("focused_facet_index").map(|value| value.as_i64().map(i128::from)) }
            });
            let fixed_limits = step["fixed_limits"].as_object().map(|limits| OrderedMap::from_pairs(limits.iter()
                .map(|(name, pair)| (name.clone(), (bits(&pair[0]), bits(&pair[1])))))).unwrap_or_default();
            let request = ResolveRequest {
                schema: golden.schema(&step["schema"]), kind: step["kind"].as_str().map(|k| PlotKind::parse(k).unwrap()),
                target: target.as_ref(), base: base.as_ref(), presented: presented.as_ref(),
                spec: step["spec"].as_object().map(|_| spec_from(&step["spec"])),
                semantic_updates: step["semantic_updates"].as_object().map(|_| dec_map(&step["semantic_updates"])),
                parameter_updates: step["parameter_updates"].as_object().map(|_| dec_map(&step["parameter_updates"])),
                different_source: !step["same_source"].as_bool().unwrap(), window: step["window"].as_i64().map(i128::from),
                defaults: None, fit: Some(golden.fit.clone()), unit_registry: None, fixed_limits, interaction_updates: interaction,
                configuration_updates: step["configuration_updates"].as_object().map(|_| typed_updates(&step["configuration_updates"], &overlays)),
            };
            let live_window = step["live_window"].as_bool().unwrap_or(false);
            let started = std::time::Instant::now();
            let result = resolve_configuration(request);
            resolve_time += started.elapsed();
            resolve_count += 1;
            let got = match &result {
                Ok(configuration) => outcome(configuration_doc(configuration, &mut signatures, live_window)),
                Err(error) => error_name(error),
            };
            let mut got = got;
            if let (Some(ours), Some(theirs)) = (got.get_mut("ok").and_then(Json::as_object_mut), step["result"]["ok"].as_object()) {
                for key in ["describe", "semantics"] { if !theirs.contains_key(key) { ours.remove(key); } }
            }
            report.compare("sequence", &|| format!("{name} step {index}: {}", step_summary(step)), &got, &step["result"]);
            if let (Ok(configuration), Some(shared)) = (&result, step.get("cancellation_shared").and_then(Json::as_bool)) {
                let ours = base.as_ref().is_some_and(|base| Arc::ptr_eq(&base.cancellation, &configuration.cancellation));
                report.compare("cancellation", &|| format!("{name} step {index}"), &json!(ours), &json!(shared));
            }
            configs.push(result.ok());
        }
    }
    if resolve_count > 0 {
        eprintln!("resolve_configuration: {resolve_count} calls, {:.1} us each", resolve_time.as_secs_f64() * 1e6 / resolve_count as f64);
    }
    let missing = golden.fit.missing.lock().unwrap();
    if !missing.is_empty() { eprintln!("unrecorded fit queries: {} (first: {})", missing.len(), missing[0]); }
}

fn check_extras(golden: &Golden, report: &mut Report) {
    let style = zlc_plot::style::PlotStyle::default();
    for case in golden.cases("projection_scope") {
        let schema = golden.schema(&case["schema"]).unwrap();
        let spec = spec_from(&case["spec"]);
        let domains: Option<Vec<AxisDomain>> = case["domains"].as_array().map(|items| items.iter().map(|d| AxisDomain::parse(d.as_str().unwrap()).unwrap()).collect());
        let got = outcome(cfg::semantics::projection_scope(&schema, &spec, domains.as_deref()).map(|scope| Json::Array(scope.iter().map(|(axis, value)| json!([
            axis_doc(axis), match value { cfg::semantics::ScopeCoordinate::Latest => json!("$latest"),
                                          cfg::semantics::ScopeCoordinate::Value(c) => enc(&cfg::kinds::coordinate_value(c)) }])).collect())));
        report.compare("projection scope", &|| format!("schema {} spec {} {:?}", case["schema"], case["spec"], case["domains"]), &got, &case["result"]);
    }
    for case in golden.cases("admits_scope") {
        let schema = golden.schema(&case["schema"]).unwrap();
        let axis = AxisRef::new(AxisDomain::parse(case["axis"]["domain"].as_str().unwrap()).unwrap(), case["axis"]["axis_id"].as_str().unwrap()).unwrap();
        let got = outcome(cfg::semantics::axis_admits_scope(&schema, &axis, &dec(&case["value"])).map(Json::Bool));
        report.compare("admits scope", &|| case.to_string(), &got, &case["result"]);
    }
    for case in golden.cases("store") {
        let (kind, cell) = kind_cell(case);
        let schema = parameter_schema_for_kind(kind, &style, cell).unwrap();
        let snap = |state: &DisplayState| json!({"revision": state.revision, "values": enc(&Value::Map(state.values.clone())),
            "changed": state.changed_names.iter().collect::<Vec<_>>(), "effects": state.effects.bits(), "interaction": enc(&state.interaction.wire())});
        let mut store = DisplayStateStore::new(schema.clone(), Some(&ValueMap::from_pairs([("title", Value::text("Start"))])), 3,
                                               RenderEffect::LAYOUT, Interaction::default()).unwrap();
        let mut log = vec![json!(["initial", snap(store.state())])];
        for entry in case["log"].as_array().unwrap().iter().skip(1) {
            match entry[0].as_str().unwrap() {
                "commit" => {
                    let previous = store.state().clone();
                    let candidate = schema.prepare_transition(&previous.values, &dec_map(&entry[1])).unwrap();
                    let state = store.commit_prepared(&previous, candidate).unwrap();
                    log.push(json!(["commit", entry[1], snap(&state)]));
                }
                _ => {
                    let state = store.commit_interaction(&InteractionUpdate::from_value(&dec(&entry[1])).unwrap());
                    log.push(json!(["interaction", entry[1], snap(&state)]));
                }
            }
        }
        report.compare("store", &|| format!("{kind:?}/{cell:?}"), &json!(log), &case["log"]);
    }
    for case in golden.cases("controls_for_kind") {
        let (kind, cell) = kind_cell(case);
        let values = case["values"].as_object().map(|_| dec_map(&case["values"]));
        let float = |value: Option<f64>| value.map_or(Json::Null, |v| enc(&Value::Float(v)));
        let got = outcome(parameter_controls_for_kind(kind, values.as_ref(), cell, &style).map(|controls| Json::Array(controls.iter().map(|control| json!({
            "name": control.name, "label": control.label, "kind": control.kind.value(), "value": enc(&control.value), "allow_none": control.allow_none,
            "choices": control.choices.iter().map(|(value, label)| json!([enc(value), label])).collect::<Vec<_>>(),
            "minimum": float(control.minimum), "maximum": float(control.maximum), "step": float(control.step), "effects": control.effects.bits(),
            "automatic": control.automatic, "automatic_value": enc(&control.automatic_value), "unavailable_reason": control.unavailable_reason,
            "segmented": control.segmented,
        })).collect())));
        report.compare("controls for kind", &|| format!("{kind:?}/{cell:?} {}", case["values"]), &got, &case["result"]);
    }
}

fn step_summary(step: &Json) -> String {
    let mut parts = Vec::new();
    for key in ["schema", "base", "presented", "kind", "semantic_updates", "parameter_updates", "configuration_updates", "interaction_updates", "window", "same_source"] {
        if !step[key].is_null() { parts.push(format!("{key}={}", step[key])); }
    }
    if !step["target"].is_null() { parts.push("target=..".into()); }
    let text = parts.join(" ");
    if text.len() > 400 { format!("{}...", &text[..400]) } else { text }
}

fn run(path: &std::path::Path) {
    let golden = Golden::load(path);
    let mut report = Report::default();
    check_parameters(&golden, &mut report);
    check_specs(&golden, &mut report);
    check_semantics(&golden, &mut report);
    check_records(&golden, &mut report);
    check_extras(&golden, &mut report);
    check_sequences(&golden, &mut report);
    report.finish(&path.display().to_string());
}

#[test]
fn configuration_matches_the_scientific_package() {
    let data = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    run(&data.join("config.json.gz"));
    run(&data.join("config_replay.json.gz"));
    run(&data.join("config_fuzz.json.gz"));
}

#[test]
fn configuration_matches_the_full_golden_set() {
    match std::env::var_os("ZLC_CONFIG_GOLDEN") {
        Some(paths) => for path in paths.to_string_lossy().split(';').filter(|p| !p.is_empty()) { run(std::path::Path::new(path)) },
        None => eprintln!("ZLC_CONFIG_GOLDEN is not set; the full set is skipped"),
    }
}
