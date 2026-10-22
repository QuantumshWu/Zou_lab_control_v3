//! Replays the golden corpus (`corpus_gen/export_native.py`) through the
//! native Figure document: every command in order, with the recorded reads,
//! live-fit answers, overlays and acceptance decisions, and compares what the
//! document answers with what the D1/D2-fixed Python oracle answered -- the
//! state, effects, presented/admitted reads, publications, fronts, stale
//! cohorts and errors -- plus every call the sessions make into their
//! renderers.
//!
//! The renderer is a recorded double: each Python renderer's call stream
//! (and its observable state after every call) was captured, and the double
//! answers the session's queries from it while checking that the session
//! makes the same calls with the same frames.  Fronts are compared by the
//! recorded sha256 of the front each call returned.
//!
//! Committed scenarios live in `tests/data/session`; the whole exported
//! corpus runs when `ZLC_SESSION_CORPUS` names its directory (filter with
//! `ZLC_SESSION_FILTER=name[,prefix...]`, summary JSON to
//! `ZLC_SESSION_REPORT`, `ZLC_SESSION_NATIVE=1` tees the native renderer
//! into every Curve/Rolling session, as `committed_scenarios_native_renderer`
//! always does).

use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::{Path, PathBuf};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicIsize, AtomicU32, AtomicU64, Ordering};
use std::time::Instant;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use zlc_core::OwnedBuffer;
use zlc_plot::configuration::error::{PlotError, Result as PlotResult};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::configuration::selectors::{NumericRange, SelectorKind, SelectorState, SelectorValue};
use zlc_plot::configuration::spec::PlotSpec;
use zlc_plot::configuration::state::{Interaction, InteractionUpdate, Presentation, PresentationUpdate};
use zlc_plot::configuration::value::{Value as PlotValue, ValueMap};
use zlc_plot::figure::ArchiveSaver;
use zlc_plot::document::{error_text, DocumentRead, Effect, FigureDocument, FigureSaver, FigureWrite, FitPublisher, SaveRequest};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::engine::{AxesId, CameraAngles};
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::interaction::transform::CanvasSize;
use zlc_plot::layout::SurfacePlan;
use zlc_plot::projection::plan::ProjectionPlan;
use zlc_plot::projection::{Source, SourceData};
use zlc_plot::render::scale::AxisScale;
use zlc_plot::session::{AxesGeometry, EngineFitLayer, FitOverlay, ImagePointOverlay, PanelRenderer, PointStatus, RenderFrame, RendererFactory,
                        RgbaFront, SeriesAction, SessionServices};
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::style::PlotStyle;

use zlc_plot::projection::golden::{self, Suite};

/// Relative agreement of numbers a fit produced (the oracle's own numba
/// dispatch history moves them at ~1e-9; `fit/TOLERANCES.md`).
const FIT_TOLERANCE: f64 = 1e-8;

// ------------------------------------------------------------------ measurement
/// Heap bytes in use, for what a document holds.
struct Counting;

static ALLOCATED: AtomicIsize = AtomicIsize::new(0);

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        ALLOCATED.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATED.fetch_add(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        ALLOCATED.fetch_add(size as isize - layout.size() as isize, Ordering::Relaxed);
        unsafe { System.realloc(pointer, layout, size) }
    }
}

#[global_allocator]
static GLOBAL: Counting = Counting;

thread_local! { static OUTSIDE_DEPTH: Cell<u32> = const { Cell::new(0) }; }
static OUTSIDE_NANOS: AtomicU64 = AtomicU64::new(0);

/// Time inside the doubles (the host's renderer, ROI planner, publisher and
/// saver), kept apart from the native document's own.
struct Outside(Option<Instant>);

fn outside() -> Outside {
    OUTSIDE_DEPTH.with(|depth| { depth.set(depth.get() + 1); Outside((depth.get() == 1).then(Instant::now)) })
}

impl Drop for Outside {
    fn drop(&mut self) {
        OUTSIDE_DEPTH.with(|depth| depth.set(depth.get() - 1));
        if let Some(started) = self.0 { OUTSIDE_NANOS.fetch_add(started.elapsed().as_nanos() as u64, Ordering::Relaxed); }
    }
}

// ------------------------------------------------------------------ reads
#[derive(Clone)]
struct CorpusRead { source: Arc<Source> }

impl DocumentRead for CorpusRead {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String {
        json!({"publication": {"stream": &*self.source.block_id, "generation": &*self.source.generation, "revision": self.source.revision,
                               "signal": "corpus"}, "run": {}, "event": {}, "parents": []}).to_string()
    }
}

/// One command's snapshots, decoded once each (a pickle's shared objects).
struct Reads<'a> { suite: &'a Suite, cache: HashMap<String, Arc<Source>> }

impl Reads<'_> {
    fn get(&mut self, key: &str) -> Arc<Source> {
        let suite = self.suite;
        self.cache.entry(key.to_owned()).or_insert_with(|| Arc::new(golden::source(suite, &suite.document["snapshots"][key]))).clone()
    }
}

// ------------------------------------------------------------------ renderer double
#[derive(Clone, Debug)]
struct Entry { command: u64, call: String, args: Value, result: Value, error: Option<String>, post: usize }

struct Script {
    observations: Vec<Value>,
    /// Each recorded renderer's remaining calls, by serial.
    entries: BTreeMap<u64, VecDeque<Entry>>,
    /// Each panel's renderers in creation order.
    creations: BTreeMap<String, VecDeque<u64>>,
    divergences: Vec<String>,
    frame_mismatches: Vec<String>,
}

#[derive(Clone)]
struct Shared { script: Arc<Mutex<Script>>, command: Arc<AtomicU32> }

impl Shared {
    fn diverge(&self, message: String) {
        let mut script = self.script.lock().unwrap();
        if script.divergences.len() < 40 { script.divergences.push(message); }
    }
}

struct Double {
    shared: Shared,
    serial: u64,
    plan: SurfacePlan,
    spec: PlotSpec,
    state: Value,
}

fn finite(value: &Value) -> f64 {
    match value {
        Value::Number(n) => n.as_f64().unwrap(),
        Value::Object(map) => match map["nf"].as_str().unwrap() { "nan" => f64::NAN, "inf" => f64::INFINITY, _ => f64::NEG_INFINITY },
        other => panic!("not a float: {other}"),
    }
}

fn pair(value: &Value) -> (f64, f64) { (finite(&value[0]), finite(&value[1])) }

fn scale(value: &Value) -> AxisScale {
    match value {
        Value::String(text) if text == "log" => AxisScale::Log,
        Value::String(_) => AxisScale::Linear,
        Value::Array(items) => AxisScale::Samples(items.iter().map(finite).collect::<Vec<_>>().into()),
        other => panic!("scale {other}"),
    }
}

fn geometry(axes: &Value) -> AxesGeometry {
    let bbox = &axes["bbox"];
    AxesGeometry {
        id: AxesId::from_gid(axes["gid"].as_str().unwrap()),
        bbox: (finite(&bbox[0]), finite(&bbox[1]), finite(&bbox[2]), finite(&bbox[3])),
        x_limits: pair(&axes["xlim"]), y_limits: pair(&axes["ylim"]), x_scale: scale(&axes["xscale"]), y_scale: scale(&axes["yscale"]),
    }
}

impl Double {
    /// One mutating call: the next recorded call of this renderer, whose
    /// observable state then becomes current.
    fn step(&mut self, call: &str) -> Option<Entry> {
        let command = self.shared.command.load(Ordering::SeqCst) as u64;
        let mut script = self.shared.script.lock().unwrap();
        let queue = script.entries.get_mut(&self.serial).unwrap();
        match queue.front() {
            Some(entry) if entry.call == call && entry.command == command => {
                let entry = queue.pop_front().unwrap();
                self.state = script.observations[entry.post].clone();
                Some(entry)
            }
            next => {
                let expected = next.map_or("nothing".to_owned(), |entry| format!("{} (#{})", entry.call, entry.command));
                let message = format!("#{command} renderer {}: rust called {call}, python called {expected}", self.serial);
                if script.divergences.len() < 40 { script.divergences.push(message); }
                None
            }
        }
    }

    fn result_bool(entry: Option<Entry>) -> bool { entry.and_then(|entry| entry.result.as_bool()).unwrap_or(false) }

    fn frame_mismatch(&self, field: &str, expected: &Value, actual: &Value) {
        let command = self.shared.command.load(Ordering::SeqCst);
        let mut script = self.shared.script.lock().unwrap();
        if script.frame_mismatches.len() < 40 {
            script.frame_mismatches.push(format!("#{command} renderer {} frame {field}: python {} rust {}", self.serial, truncate(expected), truncate(actual)));
        }
    }

    fn axes_list(&self) -> Vec<&Value> { self.state["axes"].as_array().map(|axes| axes.iter().collect()).unwrap_or_default() }
}

fn truncate(value: &Value) -> String { let text = value.to_string(); if text.len() > 400 { format!("{}...", &text[..400]) } else { text } }

fn close(a: f64, b: f64, tolerance: f64) -> bool { a == b || (a.is_nan() && b.is_nan()) || (a - b).abs() <= tolerance * a.abs().max(b.abs()) }

fn selector_doc(state: &SelectorState) -> Value {
    let value = match state.value {
        SelectorValue::XRange(range) => json!([range.low, range.high]),
        SelectorValue::Area(area) => json!([[area.x.low, area.x.high], [area.y.low, area.y.high]]),
        SelectorValue::Crosshair(point) => json!([point.x, point.y]),
        SelectorValue::Threshold(level) => json!(level),
    };
    json!({"kind": state.kind().value(), "value": value, "revision": state.revision, "facet_index": state.facet_index})
}

fn number_json(value: f64) -> Value { if value.is_finite() { json!(value) } else { json!({"nf": if value.is_nan() { "nan" } else if value > 0.0 { "inf" } else { "-inf" }}) } }

fn overlay_compare(expected: &Value, actual: &FitOverlay, suite: &Suite, out: &mut Vec<String>, path: &str) {
    let floats = |reference: &Value| -> Vec<f64> {
        let (_, _, bytes) = suite.bytes(reference);
        bytes.chunks_exact(8).map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap())).collect()
    };
    let mut check = |field: &str, ok: bool, detail: String| if !ok { out.push(format!("{path}.{field}: {detail}")); };
    check("success", expected["success"] == json!(actual.success), format!("{} vs {}", expected["success"], actual.success));
    check("formula", expected["formula"] == json!(actual.formula), String::new());
    check("diagnostic", expected["diagnostic"] == json!(actual.diagnostic), format!("{} vs {:?}", expected["diagnostic"], actual.diagnostic));
    check("facet_index", expected["facet_index"] == json!(actual.facet_index), String::new());
    check("evidence", expected["evidence"] == json!(actual.evidence), format!("{} vs {:?}", expected["evidence"], actual.evidence));
    let parameters = expected["parameters"].as_array().unwrap();
    check("parameters", parameters.len() == actual.parameter_display.len(), "count".into());
    for (wanted, got) in parameters.iter().zip(&actual.parameter_display) {
        let same = wanted["name"] == json!(got.name) && wanted["label"] == json!(got.label) && wanted["unit"] == json!(got.unit)
            && close(finite(&wanted["value"]), got.value, FIT_TOLERANCE)
            && match (&wanted["standard_error"], got.standard_error) { (Value::Null, None) => true, (value, Some(error)) if !value.is_null() => close(finite(value), error, FIT_TOLERANCE), _ => false };
        check("parameter", same, format!("{wanted} vs {got:?}"));
    }
    let lines = expected["polylines"].as_array().unwrap();
    check("polylines", lines.len() == actual.polylines.len(), format!("{} vs {}", lines.len(), actual.polylines.len()));
    for (wanted, got) in lines.iter().zip(&actual.polylines) {
        let (x, y) = (floats(&wanted["x"]), floats(&wanted["y"]));
        let same = wanted["role"] == json!(got.role) && wanted["component_index"] == json!(got.component_index) && x.len() == got.x.len()
            && x.iter().zip(got.x.iter()).all(|(a, b)| close(*a, *b, FIT_TOLERANCE)) && y.iter().zip(got.y.iter()).all(|(a, b)| close(*a, *b, FIT_TOLERANCE));
        check("polyline", same, format!("role {} len {} vs {} len {}", wanted["role"], x.len(), got.role, got.x.len()));
    }
}

impl PanelRenderer for Double {
    fn plan(&self) -> &SurfacePlan { &self.plan }
    fn spec(&self) -> &PlotSpec { &self.spec }
    fn set_spec(&mut self, spec: PlotSpec) { self.spec = spec; }
    fn relayout(&mut self, plan: SurfacePlan, facet_index: Option<usize>, facet_focus_index: Option<usize>) -> PlotResult<()> {
        let _outside = outside();
        if let Some(entry) = self.step("relayout") {
            let recorded = &entry.args;
            let actual = json!({"preset": plan.preset, "raster_size": [plan.raster_size.0, plan.raster_size.1],
                                "facet_index": facet_index, "facet_focus_index": facet_focus_index});
            let wanted = json!({"preset": recorded["plan"]["preset"], "raster_size": recorded["plan"]["raster_size"],
                                "facet_index": recorded["facet_index"], "facet_focus_index": recorded["facet_focus_index"]});
            if wanted != actual { self.frame_mismatch("relayout", &wanted, &actual); }
            if let Some(error) = entry.error { return Err(PlotError::value(error)); }
        }
        self.plan = plan;
        Ok(())
    }
    fn present(&mut self, frame: &RenderFrame, compose: bool) -> PlotResult<()> {
        let _outside = outside();
        let Some(entry) = self.step("present") else { return Ok(()) };
        let args = &entry.args;
        let check = |double: &Double, field: &str, wanted: &Value, actual: Value| if *wanted != actual { double.frame_mismatch(field, wanted, &actual); };
        check(self, "compose", &args["compose"], json!(compose));
        check(self, "effects", &args["effects"], json!(frame.effects.bits()));
        check(self, "data_revision", &args["data_revision"], json!(frame.data_revision));
        check(self, "state.revision", &args["state"]["revision"], json!(frame.state.revision));
        let values: Value = PlotValue::Map(frame.state.values.clone()).to_json();
        if !json_close(&args["state"]["values"], &values) { self.frame_mismatch("state.values", &args["state"]["values"], &values); }
        check(self, "interaction", &args["state"]["interaction"], frame.state.interaction.wire().to_json());
        check(self, "fit_model_id", &args["fit_model_id"], json!(frame.fit_model_id));
        check(self, "classifier_labels", &args["classifier_labels"], json!(frame.classifier_labels));
        let thresholds: Vec<Value> = frame.classifier_thresholds.iter().map(|value| value.map_or(Value::Null, number_json)).collect();
        if !json_close(&args["classifier_thresholds"], &json!(thresholds)) { self.frame_mismatch("classifier_thresholds", &args["classifier_thresholds"], &json!(thresholds)); }
        check(self, "image_overlay", &args["image_overlay"], json!(frame.image_overlay.as_ref().map(|overlay| overlay.revision)));
        let selectors = json!({"committed": frame.selectors.committed.iter().map(selector_doc).collect::<Vec<_>>(),
                                "candidate": frame.selectors.candidate.as_ref().map(selector_doc)});
        check(self, "selectors", &args["selectors"], selectors);
        check(self, "facet_index", &args["facet_index"], json!(frame.facet_index));
        check(self, "facet_focus_index", &args["facet_focus_index"], json!(frame.facet_focus_index));
        let view = frame.view_limits.map(|(x, y)| json!([x.map(|x| [number_json(x.0), number_json(x.1)]), y.map(|y| [number_json(y.0), number_json(y.1)])]));
        check(self, "view_limits", &args["view_limits"], view.unwrap_or(Value::Null));
        check(self, "presentation", &args["presentation"], frame.presentation.as_ref().map_or(Value::Null, |p| p.wire().to_json()));
        let suite = SUITE.with(|suite| suite.borrow().clone());
        if let Some(suite) = suite {
            let mut out = Vec::new();
            for (name, wanted, actual) in [("fit_overlays", &args["fit_overlays"], &frame.fit_overlays), ("classifier_overlays", &args["classifier_overlays"], &frame.classifier_overlays)] {
                let wanted = wanted.as_array().unwrap();
                if wanted.len() != actual.len() { out.push(format!("{name}: {} vs {}", wanted.len(), actual.len())); continue; }
                for (index, (wanted, actual)) in wanted.iter().zip(actual.iter()).enumerate() { overlay_compare(wanted, actual, &suite, &mut out, &format!("{name}[{index}]")); }
            }
            if !out.is_empty() { self.frame_mismatch("overlays", &json!(out), &Value::Null); }
        }
        if let Some(error) = entry.error { return Err(PlotError::value(error)); }
        Ok(())
    }
    fn draw(&mut self) -> PlotResult<()> { let _outside = outside(); self.step("draw"); Ok(()) }
    fn canvas_size(&self) -> CanvasSize {
        let _outside = outside();
        let canvas = &self.state["canvas"];
        CanvasSize::new(finite(&canvas[0]), finite(&canvas[1])).unwrap_or_else(|_| CanvasSize::new(self.plan.raster_size.0 as f64, self.plan.raster_size.1 as f64).unwrap())
    }
    fn axes(&self, role: &str) -> Vec<AxesId> {
        let _outside = outside();
        self.state["roles"].get(role).and_then(Value::as_array).map(|gids| gids.iter().map(|gid| AxesId::from_gid(gid.as_str().unwrap())).collect()).unwrap_or_default()
    }
    fn visible_axes(&self) -> Vec<AxesGeometry> {
        let _outside = outside();
        self.axes_list().into_iter().filter(|axes| axes["visible"] == json!(true)).map(geometry).collect()
    }
    fn geometry(&self, axes: &AxesId) -> Option<AxesGeometry> {
        let _outside = outside();
        self.axes_list().into_iter().map(geometry).find(|geometry| geometry.id == *axes)
    }
    fn primary_axes(&self) -> AxesId { let _outside = outside(); AxesId::from_gid(self.state["primary"].as_str().unwrap_or("")) }
    fn interactive_axes_at(&self, x: f64, y: f64) -> Option<AxesId> {
        let _outside = outside();
        let visible = self.visible_axes();
        let primary = self.primary_axes();
        let mut ordered: Vec<&AxesGeometry> = visible.iter().filter(|axes| axes.id == primary).collect();
        ordered.extend(visible.iter().rev().filter(|axes| axes.id != primary));
        ordered.into_iter().find(|axes| axes.contains(x, y)).map(|axes| axes.id.clone())
    }
    fn facet_index_for_axes(&self, axes: &AxesId) -> Option<usize> {
        let _outside = outside();
        if !self.spec.is_facet_grid() { return None; }
        let count = self.state["facets"].as_u64().unwrap_or(0) as usize;
        self.axes("facet_cell").iter().position(|candidate| candidate == axes).filter(|index| *index < count)
    }
    fn resolved_color_limits(&self) -> Option<(f64, f64)> { let _outside = outside(); self.state["clim"].as_array().map(|_| pair(&self.state["clim"])) }
    fn begin_selector_gesture(&mut self, kind: SelectorKind, compose: bool) -> PlotResult<()> {
        let _outside = outside();
        if let Some(entry) = self.step("begin_selector_gesture") {
            let wanted = json!({"kind": entry.args["kind"], "compose": entry.args["compose"]});
            let actual = json!({"kind": kind.value(), "compose": compose});
            if wanted != actual { self.frame_mismatch("begin_selector_gesture", &wanted, &actual); }
        }
        Ok(())
    }
    fn begin_color_limit_gesture(&mut self, candidate: NumericRange) -> PlotResult<()> {
        let _outside = outside();
        if let Some(entry) = self.step("begin_color_limit_gesture") {
            let actual = json!([candidate.low, candidate.high]);
            if !json_close(&entry.args["candidate"], &actual) { self.frame_mismatch("color candidate", &entry.args["candidate"], &actual); }
        }
        Ok(())
    }
    fn preview_selector(&mut self, state: &SelectorState) -> PlotResult<bool> {
        let _outside = outside();
        let entry = self.step("preview_selector");
        if let Some(entry) = &entry { if entry.args["state"] != selector_doc(state) { self.frame_mismatch("preview_selector", &entry.args["state"], &selector_doc(state)); } }
        Ok(Self::result_bool(entry))
    }
    fn preview_color_limit_candidate(&mut self, candidate: NumericRange) -> PlotResult<bool> {
        let _outside = outside();
        let entry = self.step("preview_color_limit_candidate");
        if let Some(entry) = &entry {
            let actual = json!([candidate.low, candidate.high]);
            if !json_close(&entry.args["candidate"], &actual) { self.frame_mismatch("color preview", &entry.args["candidate"], &actual); }
        }
        Ok(Self::result_bool(entry))
    }
    fn end_selector_gesture(&mut self) { let _outside = outside(); self.step("end_selector_gesture"); }
    fn set_classifier_labels(&mut self, labels: &[String]) {
        let _outside = outside();
        if let Some(entry) = self.step("set_classifier_labels") {
            if entry.args["labels"] != json!(labels) { self.frame_mismatch("classifier labels", &entry.args["labels"], &json!(labels)); }
        }
    }
    fn set_view_dragging(&mut self, axes: Option<&AxesId>) -> PlotResult<()> { let _outside = outside(); self.step("set_view_dragging"); let _ = axes; Ok(()) }
    fn capture_gesture_background(&mut self) -> PlotResult<()> { let _outside = outside(); self.step("capture_gesture_background"); Ok(()) }
    fn set_height_bars_dragging(&mut self, dragging: bool) -> PlotResult<()> { let _outside = outside(); self.step("set_height_bars_dragging"); let _ = dragging; Ok(()) }
    fn height_bars_scene(&self) -> bool { let _outside = outside(); self.state["scene"] == json!(true) }
    fn height_bars_camera(&self) -> Option<CameraAngles> {
        let _outside = outside();
        let camera = self.state["camera"].as_array()?;
        Some(CameraAngles { azimuth_deg: finite(&camera[0]), elevation_deg: finite(&camera[1]), zoom: finite(&camera[2]) })
    }
    fn height_bars_pick(&self, _x: f64, _y: f64) -> Option<(f64, f64)> {
        let _outside = outside();
        let command = self.shared.command.load(Ordering::SeqCst) as u64;
        let mut script = self.shared.script.lock().unwrap();
        let queue = script.entries.get_mut(&self.serial).unwrap();
        let entry = queue.front().filter(|entry| entry.call == "height_bars_pick" && entry.command == command).cloned()?;
        queue.pop_front();
        entry.result.as_array().map(|_| pair(&entry.result))
    }
    fn series_interaction(&self) -> Interaction {
        let _outside = outside();
        InteractionUpdate::from_value(&PlotValue::from_json(&self.state["interaction"])).unwrap().applied_to(&Interaction::default())
    }
    fn series_presentation(&self) -> Presentation {
        let _outside = outside();
        PresentationUpdate::from_value(&PlotValue::from_json(&self.state["presentation"])).unwrap().applied_to(&Presentation::default())
    }
    fn series_focus(&mut self, action: SeriesAction, _axes: Option<&AxesId>, _x: f64, _y: f64, _hit: f64, _click: f64) -> PlotResult<bool> {
        let _outside = outside();
        let entry = self.step("series_focus");
        if let Some(entry) = &entry { if entry.args["action"] != json!(action.value()) { self.frame_mismatch("series_focus", &entry.args["action"], &json!(action.value())); } }
        Ok(Self::result_bool(entry))
    }
    fn series_focus_scroll(&mut self, _axes: Option<&AxesId>, _step: f64) -> PlotResult<bool> { let _outside = outside(); Ok(Self::result_bool(self.step("series_focus_scroll"))) }
    fn raster_generation(&self) -> u64 { let _outside = outside(); self.state["generation"].as_u64().unwrap_or(0) }
    fn rgba(&mut self) -> PlotResult<RgbaFront> {
        let _outside = outside();
        let entry = self.step("rgba");
        let (height, width, sha) = match &entry {
            Some(entry) => (entry.result["shape"][0].as_u64().unwrap() as u32, entry.result["shape"][1].as_u64().unwrap() as u32,
                            entry.result["sha"].as_str().unwrap().to_owned()),
            None => (self.plan.raster_size.1, self.plan.raster_size.0, "unrecorded".to_owned()),
        };
        Ok(RgbaFront { width, height, pixels: OwnedBuffer::from_vec(sha.into_bytes()) })
    }
    fn save(&mut self, _format: &str, _dpi: f64, _restore_display: bool) -> PlotResult<Vec<u8>> { let _outside = outside(); self.step("save"); Ok(Vec::new()) }
}

thread_local! {
    static SUITE: std::cell::RefCell<Option<Arc<Suite>>> = const { std::cell::RefCell::new(None) };
}

/// Numbers agree within the fit tolerance, everything else exactly.
fn json_close(expected: &Value, actual: &Value) -> bool {
    match (expected, actual) {
        (Value::Number(a), Value::Number(b)) => match (a.as_f64(), b.as_f64()) {
            (Some(x), Some(y)) if a.is_f64() || b.is_f64() => close(x, y, FIT_TOLERANCE) && a.is_f64() == b.is_f64(),
            _ => a == b,
        },
        (Value::Array(a), Value::Array(b)) => a.len() == b.len() && a.iter().zip(b).all(|(x, y)| json_close(x, y)),
        (Value::Object(a), Value::Object(b)) => a.len() == b.len() && a.iter().all(|(key, x)| b.get(key).is_some_and(|y| json_close(x, y))),
        (a, b) => a == b,
    }
}

struct Factory { shared: Shared, native: Option<Arc<Mutex<NativeTally>>> }

impl RendererFactory for Factory {
    fn create(&self, panel: Option<&str>, spec: &PlotSpec, plan: &SurfacePlan, style: &PlotStyle) -> PlotResult<Box<dyn PanelRenderer>> {
        let _outside = outside();
        let panel = panel.unwrap_or("");
        let command = self.shared.command.load(Ordering::SeqCst) as u64;
        let mut script = self.shared.script.lock().unwrap();
        let serial = script.creations.get_mut(panel).and_then(VecDeque::pop_front);
        let Some(serial) = serial else {
            drop(script);
            self.shared.diverge(format!("#{command} rust created a renderer for {panel} that python did not"));
            return Err(PlotError::runtime("unrecorded renderer"));
        };
        let entry = script.entries.get_mut(&serial).and_then(VecDeque::pop_front).expect("create entry");
        let state = script.observations[entry.post].clone();
        drop(script);
        if entry.call != "create" || entry.command != command {
            self.shared.diverge(format!("#{command} renderer {serial}: created at #{} by python", entry.command));
        }
        let double = Box::new(Double { shared: self.shared.clone(), serial, plan: plan.clone(), spec: spec.clone(), state });
        match &self.native {
            Some(tally) if zlc_plot::render::core::kind_for(spec).is_some() => {
                // v2 recorded the fronts with its rail count labels centred on the ends.
                let mut recorded = style.clone();
                recorded.render.image_distribution_counts_inside = false;
                let native = NativeRendererFactory.create(Some(panel), spec, plan, &recorded)?;
                let styled = *style != PlotStyle::default();
                Ok(Box::new(Tee { double, native, tally: tally.clone(), command: self.shared.command.clone(), serial, styled, tainted: false }))
            }
            _ => Ok(double),
        }
    }
}

// ------------------------------------------------------------------ native renderer cross-check
/// What the native renderer answered differently from the recorded one
/// (`ZLC_SESSION_NATIVE=1`).
#[derive(Default, Debug)]
struct NativeTally { fronts: usize, exact: usize, tainted: usize, queries: usize, mismatches: Vec<String> }

/// A renderer of a kind the native renderer draws: the recorded double, teed with
/// the native renderer. Every call reaches both; the session is answered by
/// the double (so the replay keeps to the recorded path), and every answer
/// and front of the native renderer is compared with the double's. Nothing
/// is compared while the spec is of a kind the native renderer does not draw
/// yet, nor a front while a layer it does not draw yet (the image point
/// overlay, a non-default style) is on it.
struct Tee { double: Box<dyn PanelRenderer>, native: Box<dyn PanelRenderer>, tally: Arc<Mutex<NativeTally>>, command: Arc<AtomicU32>,
             serial: u64, styled: bool, tainted: bool }

impl Tee {
    fn drawn_kind(&self) -> bool { zlc_plot::render::core::kind_for(self.double.spec()).is_some() }

    fn check<T: std::fmt::Debug + PartialEq>(&self, what: &str, python: T, native: T) -> T {
        if !self.drawn_kind() { return python; }
        let mut tally = self.tally.lock().unwrap();
        tally.queries += 1;
        if python != native && tally.mismatches.len() < 60 {
            let command = self.command.load(Ordering::SeqCst);
            let text = |value: &T| { let text = format!("{value:?}"); if text.len() > 300 { format!("{}...", &text[..300]) } else { text } };
            tally.mismatches.push(format!("#{command} renderer {} {what}: python {} native {}", self.serial, text(&python), text(&native)));
        }
        python
    }

    fn both<T: std::fmt::Debug + PartialEq>(&mut self, what: &str, call: impl Fn(&mut dyn PanelRenderer) -> PlotResult<T>) -> PlotResult<T> {
        let python = call(&mut *self.double);
        let native = call(&mut *self.native);
        match (python, native) {
            (Ok(python), Ok(native)) => Ok(self.check(what, python, native)),
            (python, native) => {
                let (a, b) = (python.as_ref().err().map(|e| e.to_string()), native.as_ref().err().map(|e| e.to_string()));
                self.check(&format!("{what} error"), a, b);
                python
            }
        }
    }
}

impl PanelRenderer for Tee {
    fn plan(&self) -> &SurfacePlan { self.double.plan() }
    fn spec(&self) -> &PlotSpec { self.double.spec() }
    fn set_spec(&mut self, spec: PlotSpec) { self.native.set_spec(spec.clone()); self.double.set_spec(spec) }
    fn relayout(&mut self, plan: SurfacePlan, facet_index: Option<usize>, facet_focus_index: Option<usize>) -> PlotResult<()> {
        self.both("relayout", |r| r.relayout(plan.clone(), facet_index, facet_focus_index))
    }
    fn present(&mut self, frame: &RenderFrame, compose: bool) -> PlotResult<()> {
        self.tainted = self.styled || frame.image_overlay.is_some() || !self.drawn_kind();
        self.both("present", |r| r.present(frame, compose))
    }
    fn draw(&mut self) -> PlotResult<()> { self.both("draw", |r| r.draw()) }
    fn canvas_size(&self) -> CanvasSize { self.check("canvas_size", self.double.canvas_size(), self.native.canvas_size()) }
    fn axes(&self, role: &str) -> Vec<AxesId> { self.check(&format!("axes({role})"), self.double.axes(role), self.native.axes(role)) }
    fn visible_axes(&self) -> Vec<AxesGeometry> { self.check("visible_axes", self.double.visible_axes(), self.native.visible_axes()) }
    fn geometry(&self, axes: &AxesId) -> Option<AxesGeometry> { self.check("geometry", self.double.geometry(axes), self.native.geometry(axes)) }
    fn primary_axes(&self) -> AxesId { self.check("primary_axes", self.double.primary_axes(), self.native.primary_axes()) }
    fn interactive_axes_at(&self, x: f64, y: f64) -> Option<AxesId> {
        self.check("interactive_axes_at", self.double.interactive_axes_at(x, y), self.native.interactive_axes_at(x, y))
    }
    fn facet_index_for_axes(&self, axes: &AxesId) -> Option<usize> {
        self.check("facet_index_for_axes", self.double.facet_index_for_axes(axes), self.native.facet_index_for_axes(axes))
    }
    fn resolved_color_limits(&self) -> Option<(f64, f64)> {
        self.check("resolved_color_limits", self.double.resolved_color_limits(), self.native.resolved_color_limits())
    }
    fn begin_selector_gesture(&mut self, kind: SelectorKind, compose: bool) -> PlotResult<()> {
        self.both("begin_selector_gesture", |r| r.begin_selector_gesture(kind, compose))
    }
    fn begin_color_limit_gesture(&mut self, candidate: NumericRange) -> PlotResult<()> {
        self.both("begin_color_limit_gesture", |r| r.begin_color_limit_gesture(candidate))
    }
    fn preview_selector(&mut self, state: &SelectorState) -> PlotResult<bool> { self.both("preview_selector", |r| r.preview_selector(state)) }
    fn preview_color_limit_candidate(&mut self, candidate: NumericRange) -> PlotResult<bool> {
        self.both("preview_color_limit_candidate", |r| r.preview_color_limit_candidate(candidate))
    }
    fn end_selector_gesture(&mut self) { self.native.end_selector_gesture(); self.double.end_selector_gesture() }
    fn set_classifier_labels(&mut self, labels: &[String]) { self.native.set_classifier_labels(labels); self.double.set_classifier_labels(labels) }
    fn set_view_dragging(&mut self, axes: Option<&AxesId>) -> PlotResult<()> { self.both("set_view_dragging", |r| r.set_view_dragging(axes)) }
    fn capture_gesture_background(&mut self) -> PlotResult<()> { self.both("capture_gesture_background", |r| r.capture_gesture_background()) }
    fn set_height_bars_dragging(&mut self, dragging: bool) -> PlotResult<()> {
        self.both("set_height_bars_dragging", |r| r.set_height_bars_dragging(dragging))
    }
    fn height_bars_scene(&self) -> bool { self.check("height_bars_scene", self.double.height_bars_scene(), self.native.height_bars_scene()) }
    fn height_bars_camera(&self) -> Option<CameraAngles> { self.double.height_bars_camera() }
    fn height_bars_pick(&self, x: f64, y: f64) -> Option<(f64, f64)> { self.double.height_bars_pick(x, y) }
    fn series_interaction(&self) -> Interaction {
        self.check("series_interaction", self.double.series_interaction(), self.native.series_interaction())
    }
    fn series_presentation(&self) -> Presentation {
        self.check("series_presentation", self.double.series_presentation(), self.native.series_presentation())
    }
    fn series_focus(&mut self, action: SeriesAction, axes: Option<&AxesId>, x: f64, y: f64, hit: f64, click: f64) -> PlotResult<bool> {
        self.both("series_focus", |r| r.series_focus(action, axes, x, y, hit, click))
    }
    fn series_focus_scroll(&mut self, axes: Option<&AxesId>, step: f64) -> PlotResult<bool> {
        self.both("series_focus_scroll", |r| r.series_focus_scroll(axes, step))
    }
    fn raster_generation(&self) -> u64 { self.double.raster_generation() }
    fn rgba(&mut self) -> PlotResult<RgbaFront> {
        let python = self.double.rgba()?;
        let tainted = self.tainted || !self.drawn_kind();
        let native = self.native.rgba();
        let mut tally = self.tally.lock().unwrap();
        tally.fronts += 1;
        if tainted { tally.tainted += 1; } else {
            let wanted = String::from_utf8_lossy(python.pixels.as_slice()).to_string();
            let got = native.as_ref().map(|front| {
                use sha2::Digest;
                sha2::Sha256::digest(front.pixels.as_slice()).iter().map(|b| format!("{b:02x}")).collect::<String>()
            });
            match got {
                Ok(sha) if sha == wanted => tally.exact += 1,
                other => if tally.mismatches.len() < 60 {
                    let command = self.command.load(Ordering::SeqCst);
                    let what = match other { Ok(_) => "a different front".to_owned(), Err(error) => format!("error {error}") };
                    tally.mismatches.push(format!("#{command} renderer {} rgba: native drew {what}", self.serial));
                },
            }
        }
        Ok(python)
    }
    fn save(&mut self, format: &str, dpi: f64, restore_display: bool) -> PlotResult<Vec<u8>> {
        let _ = self.native.save(format, dpi, restore_display);
        self.double.save(format, dpi, restore_display)
    }
}

// ------------------------------------------------------------------ publisher, saver doubles
struct Publisher { fresh: HashMap<(String, usize, bool), Option<CorpusRead>>, published: Mutex<Vec<Value>> }

impl FitPublisher<CorpusRead> for Publisher {
    fn publish(&self, panel: &str, _read: &CorpusRead, values: BTreeMap<String, zlc_core::store::Snapshot>, records: bool, metadata: Value)
               -> std::result::Result<(), String> {
        let _outside = outside();
        self.published.lock().unwrap().push(json!({"panel": panel, "records": records, "metadata": metadata, "values": datasets_json(&values)}));
        Ok(())
    }
    fn read(&self, signal: &str, window: usize, records: bool) -> std::result::Result<Option<CorpusRead>, String> {
        let _outside = outside();
        Ok(self.fresh.get(&(signal.to_owned(), window, records)).cloned().flatten())
    }
}

/// The native archive saver writing into a scratch folder; the answer is
/// the paths Python wrote, and each archive's last bytes are kept to compare
/// with the archive Python left beside them.
struct Saver { saved: Mutex<Value>, archives: Arc<Mutex<BTreeMap<PathBuf, Vec<u8>>>> }

impl FigureSaver<CorpusRead> for Saver {
    fn prepare(&self, request: &SaveRequest<'_, CorpusRead>) -> PlotResult<Box<dyn FigureWrite>> {
        let _outside = outside();
        let python = PathBuf::from(request.path);
        let name = python.file_name().map(|name| name.to_string_lossy().into_owned()).unwrap_or_default();
        let scratch = std::env::temp_dir().join(format!("zlc_session_replay_{}", std::process::id())).join(&name);
        let path = scratch.to_string_lossy().into_owned();
        let inner = FigureSaver::<CorpusRead>::prepare(&ArchiveSaver, &SaveRequest { path: &path, read: request.read, overlay: request.overlay.clone() })?;
        let saved = self.saved.lock().unwrap().clone();
        let paths = saved.as_array().map(|items| items.iter().filter_map(|item| item.as_str().map(str::to_owned)).collect()).unwrap_or_default();
        Ok(Box::new(Written { inner, python, paths, archives: self.archives.clone() }))
    }
}

struct Written { inner: Box<dyn FigureWrite>, python: PathBuf, paths: Vec<String>, archives: Arc<Mutex<BTreeMap<PathBuf, Vec<u8>>>> }

impl FigureWrite for Written {
    fn write(self: Box<Self>, recipe: ValueMap, image: &mut dyn FnMut(&str) -> PlotResult<Vec<u8>>) -> PlotResult<Vec<String>> {
        let _outside = outside();
        for written in self.inner.write(recipe, image)? {
            let written = PathBuf::from(written);
            if written.extension().is_some_and(|extension| extension == "npz") {
                let bytes = std::fs::read(&written).unwrap();
                self.archives.lock().unwrap().insert(self.python.with_extension("npz"), bytes);
            }
            std::fs::remove_file(&written).unwrap();
        }
        Ok(self.paths)
    }
}

// ------------------------------------------------------------------ datasets
fn datasets_json(values: &BTreeMap<String, zlc_core::store::Snapshot>) -> Value {
    Value::Object(values.iter().map(|(name, snapshot)| {
        let schema = snapshot.schema();
        let segments: Vec<Value> = snapshot.segments().map(|placement| {
            let data = placement.data();
            let values = match data.values() { zlc_core::AnyBuffer::F64(values) => values.as_slice().to_vec(), _ => Vec::new() };
            let valid = match data.validity() { zlc_core::dataset::Validity::Uniform(v) => vec![*v; values.len()], zlc_core::dataset::Validity::Mask(m) => m.as_slice().to_vec() };
            let sigma = data.sigma().map(|sigma| sigma.as_slice().to_vec()).unwrap_or_default();
            json!({"origin": placement.origin(), "values": values.iter().map(|v| number_json(*v)).collect::<Vec<_>>(), "valid": valid,
                   "sigma": sigma.iter().map(|v| number_json(*v)).collect::<Vec<_>>()})
        }).collect();
        (name.clone(), json!({"schema": schema_json(schema), "window": snapshot.window().map(|(a, b)| json!([a as i64, b])), "segments": segments}))
    }).collect())
}

fn schema_json(schema: &zlc_core::schema::DatasetSchema) -> Value {
    use zlc_core::schema::{Coordinate, Coordinates, DomainRole};
    let domain = |role: DomainRole| {
        let domain = schema.domain(role);
        json!(domain.axes().iter().map(|axis| {
            let coordinates: Vec<Value> = (0..axis.len()).map(|index| match axis.coordinate(index) {
                Some(Coordinate::Integer(v)) => json!(v as i64), Some(Coordinate::Real(v)) => number_json(v),
                Some(Coordinate::Text(t)) => json!(&*t), _ => Value::Null,
            }).collect();
            let _ = matches!(axis.coordinates(), Coordinates::Index { .. });
            json!({"id": axis.id(), "name": axis.name(), "role": axis.role(), "unit": axis.unit(), "frame": axis.coordinate_frame(),
                   "labels": axis.labels().map(|labels| labels.iter().map(|l| l.to_string()).collect::<Vec<_>>()), "coordinates": coordinates})
        }).collect::<Vec<_>>())
    };
    let value = schema.value();
    json!({"repeat": domain(DomainRole::Repeat), "point": domain(DomainRole::Point), "cell": domain(DomainRole::Cell),
           "value": {"unit": value.unit(), "name": value.name()}})
}

fn source_datasets_json(source: &Source) -> Value {
    let schema = &source.schema;
    let segments: Vec<Value> = match &source.data {
        SourceData::Dense(data) => vec![(([0usize, 0usize]), data.clone())],
        SourceData::Segments(entries) => entries.iter().map(|(origin, data)| (*origin, data.clone())).collect(),
    }.into_iter().map(|(origin, data)| {
        let values: Vec<f64> = match data.values() { zlc_core::AnyBuffer::F64(values) => values.as_slice().to_vec(), _ => Vec::new() };
        let valid = match data.validity() { zlc_core::dataset::Validity::Uniform(v) => vec![*v; values.len()], zlc_core::dataset::Validity::Mask(m) => m.as_slice().to_vec() };
        let sigma = data.sigma().map(|sigma| sigma.as_slice().to_vec()).unwrap_or_default();
        json!({"origin": origin, "values": values.iter().map(|v| number_json(*v)).collect::<Vec<_>>(), "valid": valid,
               "sigma": sigma.iter().map(|v| number_json(*v)).collect::<Vec<_>>()})
    }).collect();
    json!({"schema": schema_json(schema), "window": source.window.map(|(a, b)| json!([a as i64, b as i64])), "segments": segments})
}

// ------------------------------------------------------------------ overlays
fn overlay(suite: &Suite, reads: &mut Reads<'_>, doc: &Value) -> Option<Arc<ImagePointOverlay>> {
    if doc.is_null() { return None; }
    let floats = |reference: &Value| -> (Vec<f64>, Vec<usize>) {
        let (_, shape, bytes) = suite.bytes(reference);
        (bytes.chunks_exact(8).map(|chunk| f64::from_le_bytes(chunk.try_into().unwrap())).collect(), shape)
    };
    let strings = |value: &Value| value.as_array().map(|items| items.iter().map(|item| item.as_str().unwrap().to_owned()).collect::<Vec<_>>());
    let status = doc["status"].as_str().map(|key| reads.get(key));
    let paths = (!doc["paths_xy"].is_null()).then(|| { let (values, shape) = floats(&doc["paths_xy"]); (values.into(), shape[1]) });
    Some(Arc::new(ImagePointOverlay {
        revision: doc["revision"].as_u64().unwrap(), coordinates: floats(&doc["coordinates"]).0.into(), point_ids: strings(&doc["point_ids"]),
        labels: doc["labels"].as_array().map(|items| items.iter().map(|item| item.as_str().map(str::to_owned)).collect()),
        static_statuses: doc["static_statuses"].as_array().map(|items| items.iter().map(|item| match item.as_str().unwrap() {
            "empty" => PointStatus::Empty, "occupied" => PointStatus::Occupied, "invalid" => PointStatus::Invalid, _ => PointStatus::Unknown,
        }).collect()),
        status, paths_xy: paths,
    }))
}

// ------------------------------------------------------------------ replay
#[derive(Default, serde::Serialize)]
struct Outcome {
    commands: usize, identical: usize, mismatches: Vec<(u64, String, String)>, divergences: Vec<String>, frames: Vec<String>,
    /// Per command `{number, action, panels, wall_ms, outside_ms, science_ms}`.
    timing: Vec<Value>,
    /// Saved archives byte-identical to the one Python left at the same path.
    archives_identical: usize,
    /// Differing commands the corpus tags with a known oracle defect.
    defect_tagged: usize,
    /// Heap bytes released when the document is dropped after its last command.
    document_bytes: isize,
    /// The native renderer cross-check: fronts, exact, tainted, queries, then what differed.
    native: Option<(usize, usize, usize, usize, Vec<String>)>,
}

fn replay(directory: &Path, native: bool) -> Outcome {
    replay_measured(directory, false, native)
}

fn replay_measured(directory: &Path, timed: bool, native: bool) -> Outcome {
    let suite = Arc::new(load(&suite_path(directory).expect("native suite")));
    SUITE.with(|cell| *cell.borrow_mut() = Some(suite.clone()));
    let observations: Vec<Value> = suite.document["observations"].as_array().unwrap().clone();
    let cases = suite.document["cases"].as_array().unwrap().clone();
    let mut entries: BTreeMap<u64, VecDeque<Entry>> = BTreeMap::new();
    let mut creations: BTreeMap<String, VecDeque<u64>> = BTreeMap::new();
    for case in &cases {
        let number = case["number"].as_u64().unwrap();
        for entry in case["renderer"].as_array().unwrap() {
            let serial = entry["r"].as_u64().unwrap();
            let call = entry["call"].as_str().unwrap().to_owned();
            if call == "create" { creations.entry(entry["panel"].as_str().unwrap_or("").to_owned()).or_default().push_back(serial); }
            entries.entry(serial).or_default().push_back(Entry { command: number, call, args: entry["args"].clone(), result: entry.get("result").cloned().unwrap_or(Value::Null),
                error: entry["error"].as_str().map(str::to_owned), post: entry["post"].as_u64().unwrap_or(0) as usize });
        }
    }
    let command = Arc::new(AtomicU32::new(0));
    let shared = Shared { script: Arc::new(Mutex::new(Script { observations, entries, creations, divergences: Vec::new(), frame_mismatches: Vec::new() })),
                          command: command.clone() };
    let saver = Arc::new(Saver { saved: Mutex::new(Value::Null), archives: Arc::new(Mutex::new(BTreeMap::new())) });
    let native = native.then(|| Arc::new(Mutex::new(NativeTally::default())));
    let services = SessionServices {
        renderers: Arc::new(Factory { shared: shared.clone(), native: native.clone() }), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default(),
    };
    let mut document: FigureDocument<CorpusRead> = FigureDocument::new(services, Some(saver.clone()));
    let mut outcome = Outcome::default();
    for case in &cases {
        let number = case["number"].as_u64().unwrap();
        command.store(number as u32, Ordering::SeqCst);
        let expected = &case["expected"];
        *saver.saved.lock().unwrap() = expected["state"]["saved"].clone();
        let mut reads = Reads { suite: &suite, cache: HashMap::new() };
        let inputs: BTreeMap<String, CorpusRead> = case["reads"].as_object().unwrap().iter()
            .map(|(panel, key)| (panel.clone(), CorpusRead { source: reads.get(key.as_str().unwrap()) })).collect();
        let fresh = case["fresh"].as_array().unwrap().iter().map(|item| {
            let arguments = &item[0];
            ((arguments[0].as_str().unwrap().to_owned(), arguments[1].as_u64().unwrap() as usize, arguments[2].as_bool().unwrap()),
             item[1].as_str().map(|key| CorpusRead { source: reads.get(key) }))
        }).collect();
        let publisher = Publisher { fresh, published: Mutex::new(Vec::new()) };
        let overlays = case["overlays"].as_object().map(|items| items.iter().map(|(panel, doc)| (panel.clone(), overlay(&suite, &mut reads, doc))).collect());
        let command_text = case["command"].as_str().unwrap();
        OUTSIDE_NANOS.store(0, Ordering::Relaxed);
        let started = Instant::now();
        let answer = document.execute(command_text, inputs, case["publisher"].as_bool().unwrap().then_some(&publisher as &dyn FitPublisher<CorpusRead>), overlays);
        let wall = started.elapsed().as_secs_f64() * 1e3;
        outcome.commands += 1;
        let parsed = serde_json::from_str::<Value>(command_text).unwrap();
        let action = parsed["action"].as_str().unwrap_or("").to_owned();
        if timed {
            let outside = OUTSIDE_NANOS.load(Ordering::Relaxed) as f64 / 1e6;
            outcome.timing.push(json!({"number": number, "action": action, "panels": parsed["inputs"].as_array().map_or(0, Vec::len),
                                       "wall_ms": wall, "outside_ms": outside, "science_ms": wall - outside}));
        }
        let mut mismatches: Vec<String> = Vec::new();
        match (&answer, expected.get("error").and_then(Value::as_str)) {
            (Err(error), Some(wanted)) => if error_text(error) != wanted { mismatches.push(format!("error: rust {:?} python {wanted:?}", error_text(error))); },
            (Err(error), None) => mismatches.push(format!("error: rust raised {:?}", error_text(error))),
            (Ok(_), Some(wanted)) => mismatches.push(format!("error: python raised {wanted:?}")),
            (Ok(answer), None) => compare_answer(&suite, answer, expected, &publisher, &mut mismatches),
        }
        compare_projections(&document, &case["projections"], &mut mismatches);
        if let Some(finish) = case["finish"].as_array() {
            let names = |value: &Value| value.as_array().unwrap().iter().map(|item| item.as_str().unwrap().to_owned()).collect::<BTreeSet<_>>();
            document.finish_presentation(&names(&finish[0]), &names(&finish[1]));
        }
        {
            let mut script = shared.script.lock().unwrap();
            let mut missed = Vec::new();
            for (serial, queue) in script.entries.iter_mut() {
                while queue.front().is_some_and(|entry| entry.command <= number) {
                    let entry = queue.pop_front().unwrap();
                    if entry.call != "height_bars_pick" { missed.push(format!("#{number} renderer {serial}: python called {} that rust did not", entry.call)); }
                }
            }
            for message in missed { if script.divergences.len() < 40 { script.divergences.push(message); } }
        }
        let defects: Vec<&str> = case["corpus"]["known_defects"].as_array().into_iter().flatten().filter_map(Value::as_str).collect();
        if mismatches.is_empty() { outcome.identical += 1; } else if !defects.is_empty() { outcome.defect_tagged += 1; }
        let tag = if defects.is_empty() { String::new() } else { format!(" [corpus defect {}]", defects.join("+")) };
        for mismatch in mismatches { if outcome.mismatches.len() < 200 { outcome.mismatches.push((number, format!("{action}{tag}"), mismatch)); } }
    }
    for (python, native) in saver.archives.lock().unwrap().iter() {
        match std::fs::read(python) {
            Ok(bytes) if bytes == *native => outcome.archives_identical += 1,
            Ok(bytes) => outcome.divergences.push(format!("archive {}: {} python bytes, {} native, not identical", python.display(), bytes.len(), native.len())),
            Err(_) => {}
        }
    }
    std::fs::remove_dir(std::env::temp_dir().join(format!("zlc_session_replay_{}", std::process::id()))).ok();
    let held = ALLOCATED.load(Ordering::SeqCst);
    drop(document);
    outcome.document_bytes = held - ALLOCATED.load(Ordering::SeqCst);
    let script = shared.script.lock().unwrap();
    outcome.divergences = script.divergences.clone();
    outcome.frames = script.frame_mismatches.clone();
    outcome.native = native.map(|tally| {
        let tally = tally.lock().unwrap();
        (tally.fronts, tally.exact, tally.tainted, tally.queries, tally.mismatches.clone())
    });
    outcome
}

fn compare_answer(suite: &Suite, answer: &zlc_plot::document::Execution<CorpusRead>, expected: &Value, publisher: &Publisher, out: &mut Vec<String>) {
    let mut state = answer.state.clone();
    state.as_object_mut().unwrap().remove("scientific_ms");
    // A model's parameters with their symbols, and a live panel's bound Window row, are the console's (v2 workbench
    // panel_state), not the recorded document's.
    for panel in state["panels"].as_array_mut().into_iter().flatten() {
        for model in panel["fit"]["models"].as_array_mut().into_iter().flatten() { model.as_object_mut().map(|model| model.remove("parameters")); }
        if let Some(controls) = panel["controls"].as_array_mut() { controls.retain(|control| control.get("bound").is_none()); }
    }
    let mut differences = Vec::new();
    json_diff(&expected["state"], &state, "$.state", &mut differences);
    out.extend(differences.into_iter().take(12));
    let keys = |map: &BTreeMap<String, CorpusRead>| json!(map.keys().collect::<Vec<_>>());
    if keys(&answer.presented) != expected["presented"] { out.push(format!("presented: {} vs {}", expected["presented"], keys(&answer.presented))); }
    if keys(&answer.admitted) != expected["admitted"] { out.push(format!("admitted: {} vs {}", expected["admitted"], keys(&answer.admitted))); }
    let fronts: Value = answer.fronts.iter().map(|(key, front)| (key.clone(), json!([[front.height, front.width, 4], String::from_utf8_lossy(front.pixels.as_slice())])))
        .collect::<serde_json::Map<_, _>>().into();
    if fronts != expected["fronts"] { out.push(format!("fronts: {} vs {}", truncate(&expected["fronts"]), truncate(&fronts))); }
    let mut cohorts: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for key in answer.admitted.keys() {
        let identity = answer.presented.get(key).map_or("none".to_owned(), |read| format!("{}/{}/{}", read.source.block_id, read.source.generation, read.source.revision));
        cohorts.entry(identity).or_default().push(key.clone());
    }
    let stale: BTreeSet<String> = cohorts.values().filter(|members| members.iter().any(|member| !answer.fronts.contains_key(member))).flatten().cloned().collect();
    if json!(stale) != expected["stale"] { out.push(format!("stale: {} vs {}", expected["stale"], json!(stale))); }
    // A region's document for its producer is the console's (v2 SelectionBridge), not the recorded document's.
    let effects: Vec<Value> = answer.effects.iter().filter_map(|effect| match effect {
        Effect::Remove { panel, output } => Some(json!({"kind": "remove", "panel": panel, "output": output})),
        Effect::Roi { panel, metadata, .. } => Some(json!({"kind": "roi", "panel": panel, "metadata": metadata})),
        Effect::Fit { panel, values, records, metadata, .. } => Some(json!({"kind": "fit", "panel": panel, "metadata": metadata, "records": records, "values": datasets_json(values)})),
        Effect::Selection { .. } => None,
    }).collect();
    compare_effects(suite, &expected["effects"], &effects, "effects", out);
    let mut published = publisher.published.lock().unwrap().clone();
    published.sort_by_key(|item| item["panel"].to_string());
    let mut wanted = expected["published"].as_array().cloned().unwrap_or_default();
    wanted.sort_by_key(|item| item["panel"].to_string());
    compare_effects(suite, &Value::Array(wanted), &published, "published", out);
}

/// The last projection Python built for a panel is the one its session
/// holds: the adapter's plan, and the histogram bins it carries on.
fn compare_projections(document: &FigureDocument<CorpusRead>, records: &Value, out: &mut Vec<String>) {
    let mut last: BTreeMap<&str, &Value> = BTreeMap::new();
    for record in records.as_array().into_iter().flatten().filter(|record| record["error"].is_null()) {
        if let Some(panel) = record["panel"].as_str() { last.insert(panel, record); }
    }
    for (panel, record) in last {
        let Some(handle) = document.session(panel) else { continue };
        let session = handle.lock().unwrap();
        let projection = session.projection();
        let wanted: ProjectionPlan = serde_json::from_value(record["plan"].clone()).expect("recorded projection plan");
        if *projection.plan() != wanted {
            out.push(format!("projection plan of {panel}: {} vs {}", truncate(&record["plan"]), truncate(&serde_json::to_value(projection.plan()).unwrap())));
        }
        let histogram = golden::histogram_state(&record["histogram"]);
        if projection.histogram() != histogram.as_ref() {
            out.push(format!("histogram bins of {panel}: {} vs {}", truncate(&record["histogram"]), truncate(&golden::histogram_json(&projection.histogram().cloned()))));
        }
    }
}

fn compare_effects(suite: &Suite, expected: &Value, actual: &[Value], label: &str, out: &mut Vec<String>) {
    let expected = expected.as_array().cloned().unwrap_or_default();
    if expected.len() != actual.len() {
        out.push(format!("{label}: {} vs {}", truncate(&json!(expected.iter().map(|e| json!([e["kind"], e["panel"]])).collect::<Vec<_>>())),
                         truncate(&json!(actual.iter().map(|e| json!([e["kind"], e["panel"]])).collect::<Vec<_>>()))));
        return;
    }
    for (index, (wanted, got)) in expected.iter().zip(actual).enumerate() {
        for field in ["kind", "panel", "output", "metadata", "records"] {
            if wanted.get(field) != got.get(field) && !(wanted.get(field).is_none() && got.get(field).is_none()) {
                out.push(format!("{label}[{index}].{field}: {} vs {}", truncate(&wanted[field]), truncate(&got[field])));
            }
        }
        if let Some(values) = wanted.get("values").and_then(Value::as_object) {
            for (name, key) in values {
                let source = golden::source(suite, &suite.document["snapshots"][key.as_str().unwrap()]);
                let expected_json = source_datasets_json(&source);
                if !json_close(&expected_json, &got["values"][name]) {
                    let mut differences = Vec::new();
                    json_diff(&expected_json, &got["values"][name], &format!("{label}[{index}].values.{name}"), &mut differences);
                    out.extend(differences.into_iter().take(4));
                }
            }
        }
    }
}

fn json_diff(expected: &Value, actual: &Value, path: &str, out: &mut Vec<String>) {
    if out.len() >= 12 { return; }
    let fit_number = path.contains(".result.") || path.contains("standard_errors") || path.contains("reduced_chi_square");
    match (expected, actual) {
        (Value::Object(a), Value::Object(b)) => {
            for key in a.keys().chain(b.keys().filter(|key| !a.contains_key(*key))) {
                match (a.get(key), b.get(key)) {
                    (Some(x), Some(y)) => json_diff(x, y, &format!("{path}.{key}"), out),
                    (x, _) => out.push(format!("{path}.{key}: {}", if x.is_some() { "missing" } else { "unexpected" })),
                }
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() { out.push(format!("{path}: length {} != {}", a.len(), b.len())); }
            for (index, (x, y)) in a.iter().zip(b).enumerate() { json_diff(x, y, &format!("{path}[{index}]"), out); }
        }
        (a, b) if fit_number && json_close(a, b) => {}
        (a, b) => if a != b { out.push(format!("{path}: {} != {}", truncate(a), truncate(b))); },
    }
}

/// A scenario's suite: `native.json` (exported) or `native.json.gz` (committed).
/// A suite `native.json[.gz]` with its `native.bin[.gz]` blob.
fn load(path: &Path) -> Suite {
    let read = |path: PathBuf| -> Vec<u8> {
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        if path.extension().is_some_and(|x| x == "gz") {
            let mut out = Vec::new();
            std::io::Read::read_to_end(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut out).unwrap();
            out
        } else { bytes }
    };
    let text = path.to_string_lossy();
    let bin = if let Some(stem) = text.strip_suffix(".json.gz") { PathBuf::from(format!("{stem}.bin.gz")) } else { path.with_extension("bin") };
    Suite::from_bytes(&read(path.to_path_buf()), read(bin))
}

fn suite_path(directory: &Path) -> Option<PathBuf> {
    ["native.json", "native.json.gz"].into_iter().map(|name| directory.join(name)).find(|path| path.exists())
}

fn scenarios(root: &Path, filter: Option<&str>) -> Vec<PathBuf> {
    let mut found: Vec<PathBuf> = std::fs::read_dir(root).map(|entries| entries.filter_map(|entry| entry.ok()).map(|entry| entry.path())
        .filter(|path| suite_path(path).is_some()).collect()).unwrap_or_default();
    found.sort();
    if let Some(filter) = filter {
        let names: Vec<&str> = filter.split(',').collect();
        found.retain(|path| { let name = path.file_name().unwrap().to_string_lossy(); names.iter().any(|item| name.starts_with(item)) });
    }
    found
}

/// Replays every scenario under `root`; `native` tees the native renderer
/// in. Returns the commands, the identical ones and the native renderer's
/// fronts, exact fronts, uncompared fronts, queries and differences.
fn run(root: &Path, native: bool) -> (usize, usize, (usize, usize, usize, usize, usize)) {
    let filter = std::env::var("ZLC_SESSION_FILTER").ok();
    let timing = std::env::var("ZLC_SESSION_TIMING").ok();
    let mut rows = String::new();
    let mut report = serde_json::Map::new();
    let (mut commands, mut identical, mut tagged) = (0, 0, 0);
    let mut native_total = (0, 0, 0, 0, 0);
    for directory in scenarios(root, filter.as_deref()) {
        let name = directory.file_name().unwrap().to_string_lossy().to_string();
        let started = std::time::Instant::now();
        if timing.is_some() { replay(&directory, native); }
        let outcome = std::panic::catch_unwind(|| replay_measured(&directory, timing.is_some(), native)).unwrap_or_else(|panic| {
            let text = panic.downcast_ref::<String>().cloned().or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
            Outcome { divergences: vec![format!("panic: {text}")], ..Outcome::default() }
        });
        commands += outcome.commands;
        identical += outcome.identical;
        tagged += outcome.defect_tagged;
        println!("{} {name:42} commands {:4} identical {:4} divergences {:3} frames {:3} {:.1}s",
                 if outcome.identical == outcome.commands && outcome.divergences.is_empty() && outcome.frames.is_empty() { "OK  " } else { "FAIL" },
                 outcome.commands, outcome.identical, outcome.divergences.len(), outcome.frames.len(), started.elapsed().as_secs_f64());
        for (number, action, mismatch) in outcome.mismatches.iter().take(5) { println!("      #{number} {action}: {mismatch}"); }
        for divergence in outcome.divergences.iter().take(3) { println!("      {divergence}"); }
        for frame in outcome.frames.iter().take(3) { println!("      {frame}"); }
        if let Some((fronts, exact, tainted, queries, mismatches)) = &outcome.native {
            if *fronts > 0 || !mismatches.is_empty() {
                println!("      native renderer: fronts {fronts} exact {exact} not compared {tainted}; queries {queries}, {} differ", mismatches.len());
            }
            for mismatch in mismatches.iter().take(6) { println!("        {mismatch}"); }
            native_total = (native_total.0 + fronts, native_total.1 + exact, native_total.2 + tainted, native_total.3 + queries, native_total.4 + mismatches.len());
        }
        for row in &outcome.timing {
            let mut row = row.clone();
            row["scenario"] = json!(name);
            rows.push_str(&format!("{row}\n"));
        }
        report.insert(name, serde_json::to_value(&outcome).unwrap());
    }
    if let Some(path) = timing { std::fs::write(path, rows).unwrap(); }
    if let Ok(path) = std::env::var("ZLC_SESSION_REPORT") { std::fs::write(path, serde_json::to_string_pretty(&report).unwrap()).unwrap(); }
    println!("total: {identical}/{commands} commands identical; {tagged} of the others carry a corpus defect tag");
    if native {
        let (fronts, exact, tainted, queries, differ) = native_total;
        println!("native renderer: fronts {fronts} exact {exact} not compared {tainted}; queries {queries}, {differ} differ");
    }
    (commands, identical, native_total)
}

#[test]
fn committed_scenarios_replay() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/session");
    let (commands, identical, _) = run(&root, false);
    assert!(commands > 0, "no committed session scenarios");
    assert_eq!(identical, commands);
}

/// The committed scenarios with the native renderer teed into every
/// Curve/Rolling session: it answers every query as the recorded renderer
/// did and paints every front it is responsible for byte-identically.
#[test]
fn committed_scenarios_native_renderer() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/session");
    let (_, _, (fronts, exact, uncompared, queries, differ)) = run(&root, true);
    assert!(fronts > 0 && queries > 0);
    assert_eq!(differ, 0);
    assert_eq!(exact + uncompared, fronts);
}

#[test]
fn corpus_replays() {
    let Ok(root) = std::env::var("ZLC_SESSION_CORPUS") else { return };
    run(Path::new(&root), std::env::var("ZLC_SESSION_NATIVE").is_ok());
}
