//! The renderer against the recorded renderer oracle (`tests/gen/render_capture.py`):
//! every Curve/Rolling renderer of the corpus is rebuilt natively, driven by
//! the recorded frames and session calls, and at every front its
//! materialized IR is compared with the IR extracted from matplotlib.
//!
//! Env-gated: `ZLC_RENDER_ORACLE` = the capture output directory,
//! `ZLC_RENDER_BLOBS` = its blob store; `ZLC_RENDER_ONLY` = scenario prefixes
//! (comma-separated); `ZLC_RENDER_VERBOSE` prints every differing front.
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::Value as Json;
use zlc_plot::configuration::recipe::decode_spec;
use zlc_plot::configuration::{DisplayState, Interaction, PlotLabels, PlotSpec, Presentation, PulseTimelinePlot, RenderEffect, Value, ValueMap};
use zlc_plot::kinds::{AxisDomain, AxisRef};
use zlc_plot::layout::{AxesPlan, FacetTopology, FacetTypography, Kind, NormalizedBox, Room, SurfacePlan};
use zlc_plot::projection::nd::{Array, Scalar};
use zlc_plot::projection::output::{AxisValue, CurveData, CurveSeries, FacetCell, FacetData, HistogramData, ImageData, Payload, PulsePayload, QuantityArray};
use zlc_plot::pulse::PulseTimelineData;
use zlc_plot::render::scale::AxisScale;
use zlc_plot::render::core::{kind_for, Renderer};
use zlc_plot::render::frame::RenderFrame;
use zlc_plot::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorState, SelectorValue};
use zlc_plot::fit::FitParameterDisplay;
use zlc_plot::interaction::controller::SelectorSnapshot;
use zlc_plot::session::{FitEllipseGlyph, FitOverlay, FitPolyline};
use zlc_plot::session::overlay::{ImagePointOverlay, OverlayArray, PointStatus};
use zlc_plot::projection::golden::{source, Suite};
use zlc_plot::session::renderer::PanelRenderer;
use zlc_plot::style::PlotStyle;
use zlc_plot::units::resolve_unit;

// ------------------------------------------------------------------ decoding
fn f(value: &Json) -> f64 { f64::from_bits(value.as_u64().unwrap_or_else(|| panic!("float bits expected: {value}"))) }

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

struct Capture {
    document: Json,
    blob: Vec<u8>,
    payloads: Vec<Option<Arc<Payload>>>,
    /// A pulse-ops capture's timelines by data.json key (one object each, as Python's).
    timelines: BTreeMap<String, Arc<PulseTimelineData>>,
    overlays: BTreeMap<usize, Arc<ImagePointOverlay>>,
}

fn hex_f64(text: &str) -> f64 {
    let bytes: Vec<u8> = (0..8).map(|i| u8::from_str_radix(&text[2 * i..2 * i + 2], 16).unwrap()).collect();
    f64::from_le_bytes(bytes.try_into().unwrap())
}

fn scalar(value: &Json) -> Scalar {
    match value {
        Json::Null => Scalar::None,
        Json::Bool(v) => Scalar::Bool(*v),
        Json::Number(n) => Scalar::Int(n.as_i64().map(i128::from).unwrap_or_else(|| i128::from(n.as_u64().unwrap()))),
        Json::String(s) => Scalar::Text(s.as_str().into()),
        Json::Object(map) => {
            if let Some(Json::String(hex)) = map.get("f") { return Scalar::Float(hex_f64(hex)); }
            if let Some(Json::String(i)) = map.get("i") { return Scalar::Int(i.parse().unwrap()); }
            if let Some(Json::Bool(v)) = map.get("v") { return Scalar::Bool(*v); }
            panic!("scalar {value}")
        }
        other => panic!("scalar {other}"),
    }
}

impl Capture {
    fn load(dir: &Path) -> Capture {
        let document: Json = serde_json::from_slice(&std::fs::read(dir.join("capture.json")).unwrap()).unwrap();
        let blob = std::fs::read(dir.join("capture.bin")).unwrap();
        let count = document["payloads"].as_array().map_or(0, Vec::len);
        let timelines = document["data"].as_object().map(|data| data.iter()
            .map(|(key, tree)| (key.clone(), PulseTimelineData::from_json(tree).expect("a pulse timeline"))).collect()).unwrap_or_default();
        Capture { document, blob, payloads: vec![None; count], timelines, overlays: BTreeMap::new() }
    }

    /// A pulse frame's payload: the timeline its data.json key names.
    fn pulse_payload(&self, key: &Json, revision: u64) -> Arc<Payload> {
        Arc::new(Payload::Pulse(PulsePayload { data: self.timelines[key.as_str().expect("a data key")].clone(), revision }))
    }

    fn array(&self, reference: &Json) -> Array {
        let a = &reference["a"];
        let dtype = a[0].as_str().unwrap();
        let shape: Vec<usize> = a[1].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let offset = a[2].as_u64().unwrap() as usize;
        let n: usize = shape.iter().product();
        macro_rules! typed { ($t:ty) => {{
            let size = std::mem::size_of::<$t>();
            let bytes = &self.blob[offset..offset + n * size];
            Array::from_vec(bytes.chunks_exact(size).map(|c| <$t>::from_le_bytes(c.try_into().unwrap())).collect::<Vec<$t>>(), shape)
        }}; }
        match dtype {
            "f64" => typed!(f64),
            "f32" => typed!(f32),
            "i64" => typed!(i64),
            "i32" => typed!(i32),
            "i16" => typed!(i16),
            "i8" => typed!(i8),
            "u32" => typed!(u32),
            "u64" => typed!(u64),
            "u16" => typed!(u16),
            "u8" => typed!(u8),
            "bool" => Array::from_vec(self.blob[offset..offset + n].iter().map(|b| *b != 0).collect::<Vec<bool>>(), shape),
            other => panic!("dtype {other}"),
        }
    }

    fn quantity(&self, doc: &Json) -> QuantityArray {
        let unit = |name: &str| resolve_unit(doc[name].as_str().unwrap(), None).unwrap_or_else(|e| panic!("unit {}: {e:?}", doc[name]));
        QuantityArray::with_display(self.array(&doc["canonical"]), self.array(&doc["display"]), unit("canonical_unit"),
                                    unit("display_unit"), doc["label"].as_str().unwrap().to_owned())
    }

    fn axis_value(&self, doc: &Json) -> AxisValue {
        let r = &doc["ref"];
        AxisValue {
            axis: AxisRef { domain: AxisDomain::parse(r["domain"].as_str().unwrap()).unwrap(), axis_id: r["axis_id"].as_str().unwrap().to_owned() },
            index: doc["index"].as_u64().map(|v| v as usize),
            canonical: scalar(&doc["canonical"]),
            display: scalar(&doc["display"]),
            label: doc["label"].as_str().unwrap().to_owned(),
        }
    }

    fn axis_ref(doc: &Json) -> AxisRef {
        AxisRef { domain: AxisDomain::parse(doc["domain"].as_str().unwrap()).unwrap(), axis_id: doc["axis_id"].as_str().unwrap().to_owned() }
    }

    fn payload_doc(&self, doc: &Json) -> Payload {
        let revision = doc["revision"].as_u64().unwrap();
        let generation: Arc<str> = doc["generation"].as_str().unwrap().into();
        match doc["type"].as_str().unwrap() {
            "CurveData" => Payload::Curve(CurveData {
                revision, generation, x_ref: None, group_by: Vec::new(),
                series: doc["series"].as_array().unwrap().iter().map(|s| CurveSeries {
                    x: self.quantity(&s["x"]),
                    y: self.quantity(&s["y"]),
                    valid: self.array(&s["valid"]),
                    sem: if s["sem"].is_null() { None } else { Some(self.array(&s["sem"])) },
                    x_labels: s["x_labels"].as_array().map(|v| v.iter().map(|t| t.as_str().unwrap().to_owned()).collect()),
                    group_key: s["group_key"].as_array().unwrap().iter().map(|v| self.axis_value(v)).collect(),
                    label: s["label"].as_str().unwrap().to_owned(),
                }).collect(),
                x_role: doc["x_role"].as_str().map(Into::into),
            }),
            "ImageData" => Payload::Image(ImageData {
                revision, generation, x_ref: Self::axis_ref(&doc["x_ref"]), y_ref: Self::axis_ref(&doc["y_ref"]),
                x: self.quantity(&doc["x"]), y: self.quantity(&doc["y"]), z: self.quantity(&doc["z"]), valid: self.array(&doc["valid"]),
            }),
            "HistogramData" => Payload::Histogram(self.histogram_payload(doc)),
            "FacetData" => Payload::Facet(FacetData {
                revision, generation,
                cells: doc["cells"].as_array().unwrap().iter().map(|cell| FacetCell {
                    facet_index: cell["facet_index"].as_u64().map(|v| v as usize),
                    facet_value_canonical: scalar(&cell["facet_value_canonical"]),
                    facet_value_display: scalar(&cell["facet_value_display"]),
                    label: cell["label"].as_str().unwrap().to_owned(),
                    payload: self.payload_doc(&cell["payload"]),
                }).collect(),
            }),
            other => panic!("payload type {other}"),
        }
    }

    /// A recorded `ImagePointOverlay` (its dynamic status a stored snapshot).
    fn overlay(&mut self, index: usize) -> Arc<ImagePointOverlay> {
        if let Some(overlay) = self.overlays.get(&index) { return overlay.clone(); }
        let doc = self.document["payloads"][index].clone();
        let floats = |reference: &Json| -> (Vec<f64>, Vec<usize>) {
            let array = self.array(reference);
            (array.to_f64(), array.shape().to_vec())
        };
        let texts = |v: &Json| v.as_array().map(|items| items.iter().map(|t| t.as_str().map(str::to_owned)).collect::<Vec<_>>());
        let (coordinates, shape) = floats(&doc["coordinates"]);
        let paths = (!doc["paths_xy"].is_null()).then(|| { let (values, shape) = floats(&doc["paths_xy"]); OverlayArray { values: Some(values), shape } });
        let status = doc["status"].as_str().map(|key| {
            let suite = Suite { document: self.document.clone(), blob: self.blob.clone(), compared: Default::default() };
            Arc::new(source(&suite, &self.document["snapshots"][key]))
        });
        let overlay = Arc::new(ImagePointOverlay::new(
            i128::from(doc["revision"].as_u64().unwrap()), OverlayArray { values: Some(coordinates), shape },
            texts(&doc["point_ids"]).map(|ids| ids.into_iter().map(Option::unwrap).collect()), texts(&doc["labels"]),
            doc["static_statuses"].as_array().map(|s| s.iter().map(|v| PointStatus::parse(v.as_str().unwrap()).unwrap()).collect()),
            status, paths,
        ).expect("overlay"));
        self.overlays.insert(index, overlay.clone());
        overlay
    }

    fn payload(&mut self, index: usize) -> Arc<Payload> {
        if let Some(p) = &self.payloads[index] { return p.clone(); }
        let doc = self.document["payloads"][index].clone();
        let payload = Arc::new(self.payload_doc(&doc));
        self.payloads[index] = Some(payload.clone());
        payload
    }
}

impl Capture {
    /// A recorded `HistogramData`.
    fn histogram_payload(&self, doc: &Json) -> HistogramData {
        HistogramData {
            revision: doc["revision"].as_u64().unwrap(),
            generation: doc["generation"].as_str().unwrap().into(),
            edges: self.quantity(&doc["edges"]),
            centers: self.quantity(&doc["centers"]),
            counts: self.array(&doc["counts"]),
            group_keys: doc["group_keys"].as_array().unwrap().iter()
                .map(|key| key.as_array().unwrap().iter().map(|v| self.axis_value(v)).collect()).collect(),
        }
    }
}

/// A Histogram front's recorded scales against the renderer's: how many axes agree, of how many.
fn histogram_scales(renderer: &Renderer, facts: &Json) -> (usize, usize) {
    let scales = facts["scales"].as_array().unwrap();
    let name = |scale: AxisScale| match scale { AxisScale::Log => "log", _ => "linear" };
    let agree = scales.iter().enumerate().filter(|(index, pair)| {
        *index < renderer.fig.axes.len() && pair[0] == name(renderer.axis_scale(*index, true)) && pair[1] == name(renderer.axis_scale(*index, false))
    }).count();
    (agree, scales.len().max(renderer.fig.axes.len()))
}

/// A goldens scalar as a float (`{"f": hex}`) or an integer.
fn num(value: &Json) -> f64 {
    match value {
        Json::Number(n) => n.as_f64().unwrap(),
        Json::Object(map) if map.contains_key("f") => hex_f64(map["f"].as_str().unwrap()),
        Json::Object(map) if map.contains_key("i") => map["i"].as_str().unwrap().parse().unwrap(),
        other => panic!("number {other}"),
    }
}

fn opt_index(value: &Json) -> Option<usize> { if value.is_null() { None } else { Some(num(value) as usize) } }

impl Capture {
    fn floats(&self, reference: &Json) -> Arc<[f64]> {
        self.array(reference).to_f64().into()
    }

    fn parameter(&self, doc: &Json) -> FitParameterDisplay {
        FitParameterDisplay {
            name: doc["name"].as_str().unwrap().to_owned(), label: doc["label"].as_str().unwrap().to_owned(), value: num(&doc["value"]),
            standard_error: if doc["standard_error"].is_null() { None } else { Some(num(&doc["standard_error"])) },
            unit: doc["unit"].as_str().unwrap().to_owned(),
        }
    }

    fn fit_overlay(&self, doc: &Json) -> FitOverlay {
        let role = |text: &str| -> &'static str { match text { "primary" => "primary", "component" => "component", "total" => "total", other => panic!("role {other}") } };
        FitOverlay {
            polylines: doc["polylines"].as_array().unwrap().iter().map(|p| FitPolyline {
                x: self.floats(&p["x"]), y: self.floats(&p["y"]), role: role(p["role"].as_str().unwrap()), component_index: num(&p["component_index"]) as usize,
            }).collect(),
            ellipse_glyph: if doc["ellipse_glyph"].is_null() { None } else {
                let g = &doc["ellipse_glyph"];
                Some(FitEllipseGlyph { center_x: num(&g["center_x"]), center_y: num(&g["center_y"]), radius_x: num(&g["radius_x"]), radius_y: num(&g["radius_y"]) })
            },
            success: doc["success"].as_bool().unwrap(),
            formula: doc["formula"].as_str().unwrap().to_owned(),
            parameter_display: doc["parameter_display"].as_array().unwrap().iter().map(|p| self.parameter(p)).collect(),
            diagnostic: doc["diagnostic"].as_str().unwrap().to_owned(),
            facet_index: opt_index(&doc["facet_index"]),
            group_key: doc["group_key"].as_array().unwrap().iter().map(|v| self.axis_value(v)).collect(),
            headline_parameter: if doc["headline_parameter"].is_null() { None } else { Some(self.parameter(&doc["headline_parameter"])) },
            evidence: doc["evidence"].as_str().unwrap().to_owned(),
        }
    }
}

/// `config_common.selector_doc` back into a state.
fn selector(doc: &Json) -> SelectorState {
    let kind = doc["kind"].as_str().unwrap();
    let v = dec(&doc["value"]);
    let float = |value: &Value| match value { Value::Float(f) => *f, Value::Int(i) => *i as f64, other => panic!("selector number {other:?}") };
    let pair = |value: &Value| { let items = value.as_list().expect("pair"); (float(&items[0]), float(&items[1])) };
    let value = match kind {
        "x_range" => { let (a, b) = pair(&v); SelectorValue::XRange(NumericRange::new(a, b).unwrap()) }
        "area" => {
            let map = v.as_map().expect("area");
            let (x0, x1) = pair(map.get("x").unwrap());
            let (y0, y1) = pair(map.get("y").unwrap());
            SelectorValue::Area(RectangleRange { x: NumericRange::new(x0, x1).unwrap(), y: NumericRange::new(y0, y1).unwrap() })
        }
        "crosshair" => { let (x, y) = pair(&v); SelectorValue::Crosshair(CrosshairPoint::new(x, y).unwrap()) }
        "threshold" => SelectorValue::Threshold(float(&v)),
        other => panic!("selector kind {other}"),
    };
    SelectorState::new(value, doc["revision"].as_u64().unwrap(), doc["facet_index"].as_u64()).unwrap()
}

/// A recorded spec: the recipe codec's form, or a pulse timeline's labels.
fn spec_of(doc: &Json) -> PlotSpec {
    if doc["kind"] == "pulse_timeline" {
        let labels = dec(&doc["labels"]);
        let slot = |name: &str| labels.as_map().and_then(|map| map.get(name)).and_then(Value::as_str).map(Arc::from);
        return PlotSpec::PulseTimeline(PulseTimelinePlot { labels: PlotLabels::new(slot("title"), slot("x"), slot("y"), slot("value")).unwrap() });
    }
    decode_spec(&dec(doc)).expect("spec")
}

fn snapshot(doc: &Json) -> SelectorSnapshot {
    SelectorSnapshot {
        committed: doc["committed"].as_array().unwrap().iter().map(selector).collect(),
        candidate: if doc["candidate"].is_null() { None } else { Some(selector(&doc["candidate"])) },
    }
}

fn plan(doc: &Json) -> SurfacePlan {
    let axes = |value: &Json| -> Option<Vec<AxesPlan>> {
        value.as_array().map(|list| list.iter().map(|a| {
            let b = &a["box"];
            let r = &a["room"];
            AxesPlan {
                role: a["role"].as_str().unwrap().to_owned(),
                r#box: NormalizedBox { left: f(&b[0]), top: f(&b[1]), right: f(&b[2]), bottom: f(&b[3]) },
                cell_index: a["cell_index"].as_u64().map(|v| v as u32),
                room: Room { left: f(&r[0]), right: f(&r[1]), top: f(&r[2]), bottom: f(&r[3]) },
            }
        }).collect())
    };
    let pair = |v: &Json| (v[0].as_u64().unwrap() as u32, v[1].as_u64().unwrap() as u32);
    let preset: &'static str = Box::leak(doc["preset"].as_str().unwrap().to_owned().into_boxed_str());
    let topology = doc.get("facet_topology").filter(|t| !t.is_null()).map(|t| FacetTopology {
        cell_count: t["cell_count"].as_u64().unwrap() as u32,
        cell_height_over_width: t["cell_height_over_width"].as_u64().map(f64::from_bits),
    });
    let typography = doc.get("facet_typography").filter(|t| !t.is_null()).map(|t| FacetTypography {
        tier: Box::leak(t["tier"].as_str().unwrap().to_owned().into_boxed_str()),
        scale: f(&t["scale"]), cell_title_pt: f(&t["cell_title_pt"]), tick_pt: f(&t["tick_pt"]),
        outer_axis_label_pt: f(&t["outer_axis_label_pt"]), cell_title_min_pt: f(&t["cell_title_min_pt"]),
        cell_title_max_width_pt: f(&t["cell_title_max_width_pt"]),
    });
    SurfacePlan {
        preset,
        kind: Kind::parse(doc["kind"].as_str().unwrap()).unwrap(),
        logical_size: pair(&doc["logical_size"]),
        raster_size: pair(&doc["raster_size"]),
        figure_size_inches: (f(&doc["figure_size_inches"][0]), f(&doc["figure_size_inches"][1])),
        logical_dpi: f(&doc["logical_dpi"]),
        dpi: f(&doc["dpi"]),
        device_pixel_ratio: f(&doc["device_pixel_ratio"]),
        export_scale: f(&doc["export_scale"]),
        axes: axes(&doc["axes"]).unwrap(),
        facet_topology: topology,
        facet_shape: doc["facet_shape"].as_array().map(|_| pair(&doc["facet_shape"])),
        facet_typography: typography,
        facet_focus_axes: axes(&doc["facet_focus_axes"]),
        rolling_side_distribution: doc["rolling_side_distribution"].as_bool().unwrap_or(false),
        image_height_over_width: doc["image_height_over_width"].as_u64().map(f64::from_bits),
    }
}

fn interaction(value: &Value) -> Interaction {
    zlc_plot::configuration::InteractionUpdate::from_value(value).expect("interaction").applied_to(&Interaction::default())
}

fn presentation(value: &Value) -> Presentation {
    zlc_plot::configuration::PresentationUpdate::from_value(value).expect("presentation").applied_to(&Presentation::default())
}

fn state(doc: &Json) -> Arc<DisplayState> {
    let values: ValueMap = dec(&doc["values"]).as_map().cloned().unwrap();
    Arc::new(DisplayState {
        revision: doc["revision"].as_u64().unwrap(),
        values,
        changed_names: Default::default(),
        effects: RenderEffect(doc["effects"].as_u64().unwrap() as u32),
        interaction: interaction(&dec(&doc["interaction"])),
    })
}

fn frame(capture: &mut Capture, doc: &Json) -> RenderFrame {
    let views = doc["view_limits"].as_array().map(|v| {
        let one = |x: &Json| x.as_array().map(|p| (f(&p[0]), f(&p[1])));
        (one(&v[0]), one(&v[1]))
    });
    let fits: Vec<FitOverlay> = doc["fit_overlays"].as_array().unwrap().iter().map(|o| capture.fit_overlay(o)).collect();
    let classifiers: Vec<FitOverlay> = doc["classifier_overlays"].as_array().unwrap().iter().map(|o| capture.fit_overlay(o)).collect();
    let data_revision = doc["data_revision"].as_u64().unwrap_or(0);
    RenderFrame {
        payload: match doc["payload"].get("pulse") {
            Some(key) => capture.pulse_payload(key, data_revision),
            None => capture.payload(doc["payload"].as_u64().expect("a payload") as usize),
        },
        state: state(&doc["state"]),
        effects: RenderEffect(doc["effects"].as_u64().unwrap() as u32),
        data_revision,
        fit_overlays: fits.into(),
        fit_model_id: doc["fit_model_id"].as_str().map(str::to_owned),
        classifier_overlays: classifiers.into(),
        classifier_thresholds: doc["classifier_thresholds"].as_array().unwrap().iter().map(|v| v.as_u64().map(f64::from_bits)).collect(),
        classifier_labels: doc["classifier_labels"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect(),
        image_overlay: doc["image_overlay"].as_u64().map(|index| capture.overlay(index as usize)),
        selectors: snapshot(&doc["selectors"]),
        facet_index: doc["facet_index"].as_u64().map(|v| v as usize),
        facet_focus_index: doc["facet_focus_index"].as_u64().map(|v| v as usize),
        view_limits: views,
        presentation: if doc["presentation"].is_null() { None } else { Some(presentation(&dec(&doc["presentation"]))) },
    }
}

// ---------------------------------------------------------------- comparing
const NODE_FIELDS: [&str; 8] = ["transform", "clip_box", "offset_transform", "trans_data", "trans_axes", "bbox", "trans_figure", "figure_bbox"];

/// Replace node references by the node trees they name.
fn resolve(value: &Json, nodes: &[Json], key: Option<&str>) -> Json {
    match value {
        Json::Object(map) => Json::Object(map.iter().filter(|(k, _)| k.as_str() != "nodes")
            .map(|(k, v)| (k.clone(), resolve(v, nodes, Some(k)))).collect()),
        Json::Array(items) => Json::Array(items.iter().map(|v| resolve(v, nodes, None)).collect()),
        Json::Number(n) if key.is_some_and(|k| NODE_FIELDS.contains(&k)) => node(n.as_u64().unwrap() as usize, nodes),
        other => other.clone(),
    }
}

fn node(index: usize, nodes: &[Json]) -> Json {
    let n = &nodes[index];
    let tag = if n.get("t").is_some() { "t" } else { "b" };
    let refs: &[&str] = match n[tag].as_str().unwrap() {
        "composite" => &["a", "b"],
        "blended" => &["x", "y"],
        "bbox_to" | "bbox_from" => &["bbox"],
        "scaled_translation" => &["scale"],
        "transformed" => &["bbox", "transform"],
        _ => &[],
    };
    Json::Object(n.as_object().unwrap().iter().map(|(k, v)| {
        if refs.contains(&k.as_str()) { (k.clone(), node(v.as_u64().unwrap() as usize, nodes)) } else { (k.clone(), v.clone()) }
    }).collect())
}

fn diff(a: &Json, b: &Json, path: &str, out: &mut Vec<String>, ignore: &dyn Fn(&str) -> bool) {
    if out.len() >= 40 || ignore(path) { return; }
    match (a, b) {
        (Json::Object(x), Json::Object(y)) => {
            for (k, v) in x {
                match y.get(k) { Some(w) => diff(v, w, &format!("{path}/{k}"), out, ignore), None => out.push(format!("{path}/{k}: missing in oracle")) }
            }
            for k in y.keys() { if !x.contains_key(k) && !ignore(&format!("{path}/{k}")) { out.push(format!("{path}/{k}: missing in native")); } }
        }
        (Json::Array(x), Json::Array(y)) => {
            if x.len() != y.len() { out.push(format!("{path}: length {} != {}", x.len(), y.len())); }
            for (i, (v, w)) in x.iter().zip(y).enumerate() { diff(v, w, &format!("{path}/{i}"), out, ignore); }
        }
        _ => if a != b {
            let show = |v: &Json| match v { Json::Number(n) if n.as_u64().is_some_and(|b| b > 1 << 52) => format!("{v} ({})", f64::from_bits(n.as_u64().unwrap())), _ => v.to_string() };
            let (sa, sb) = (show(a), show(b));
            out.push(format!("{path}: native {} != oracle {}", &sa[..sa.len().min(200)], &sb[..sb.len().min(200)]));
        },
    }
}

/// Fields that are a matplotlib draw's residue rather than renderer policy.
fn residue(path: &str) -> bool {
    // A spine's path is re-laid from the view limits by every draw
    // (`Spine._adjust_location`), and an axis label and an unpinned offset
    // text are placed from the tick labels by every Axis draw
    // (`_update_label_position`, `_update_offset_text_position`): what the
    // figure holds is whatever the last draw or measurement left.
    let placed = (path.contains("/xaxis/") || path.contains("/yaxis/"))
        && (path.ends_with("/label/x") || path.ends_with("/label/y") || path.ends_with("/offset_text/x") || path.ends_with("/offset_text/y"));
    // Which drawn tick a spine pads by is picked in Python set order (object
    // hashes); every tick of an axis pads alike.
    let spine_pick = path.contains("/spine_ticks/");
    path.ends_with("/update_calls") || (path.contains("/spines/") && path.contains("/path/vertices")) || placed || spine_pick
}

/// Child fields every draw writes before it reads them: a pulse bracket
/// side's vertices (and their recache), an annotation's transform
/// (`update_positions`) and a text badge's box transform
/// (`update_bbox_position_size`).
fn blank_child_residue(figure: &mut Json) {
    let Some(axes) = figure["axes"].as_array_mut() else { return };
    for axes in axes {
        let Some(children) = axes["children"].as_array_mut() else { continue };
        for child in children {
            let Some(child) = child.as_object_mut() else { continue };
            if child.get("bracket").is_some_and(|b| !b.is_null()) {
                for key in ["xy", "subslice", "x_filled", "stroke_simplify", "points_simplify"] { child.remove(key); }
            }
            if child.get("annotation").is_some_and(|a| !a.is_null()) { child.remove("transform"); }
            if let Some(Json::Object(badge)) = child.get_mut("bbox_patch") { badge.remove("transform"); }
        }
        // Ticks whose own templates differ only by which transform objects
        // their labels hold (`tick_params(pad=...)` gives each its own, a
        // new tick shares the first one's) are ticks alike.
        for name in ["xaxis", "yaxis"] {
            let major = &mut axes[name]["major"];
            let Some(per_tick) = major["per_tick"].as_array() else { continue };
            let free = |tick: &Json| {
                let mut tick = tick.clone();
                for key in ["tick1line", "tick2line", "gridline"] { tick[key]["xy"] = Json::Null; }
                for key in ["label1", "label2"] { for field in ["x", "y", "text"] { tick[key][field] = Json::Null; } }
                tick
            };
            if per_tick.iter().all(|tick| free(tick) == free(&per_tick[0])) { major["per_tick"] = Json::Null; }
        }
    }
}

/// An axes no draw lays out keeps the label and offset positions its last
/// visible draw left: residue on both sides, taken out of the comparison.
fn blank_undrawn_positions(native: &mut Json, oracle: &mut Json) {
    let hidden: Vec<usize> = native["axes"].as_array().map(|axes| axes.iter().enumerate()
        .filter(|(_, a)| a["visible"] == Json::Bool(false)).map(|(i, _)| i).collect()).unwrap_or_default();
    for figure in [native, oracle] {
        for &index in &hidden {
            for axis in ["xaxis", "yaxis"] {
                for text in ["label", "offset_text"] {
                    for coordinate in ["x", "y"] { figure["axes"][index][axis][text][coordinate] = Json::Null; }
                }
            }
        }
    }
}

#[derive(Default, Debug)]
struct Tally {
    fronts: usize, compared: usize, exact: usize, skipped: BTreeMap<String, usize>, failed: BTreeMap<String, usize>,
    pixels_compared: usize, pixels_exact: usize, pixels_failed: BTreeMap<String, usize>, pixels_full_draw: usize,
    overlay_compared: usize, overlay_exact: usize,
    exports: usize, exports_exact: usize, exports_overlaid: usize,
    queries: usize, queries_exact: usize,
    scale_queries: usize, scale_queries_exact: usize,
}

/// Remove from the oracle's axes children what the native IR does not have,
/// keeping the native children as an ordered subsequence (another layer's
/// artists interleave with the renderer's own).
fn strip_overlay_children(native: &Json, oracle: &mut Json) {
    let (Some(mine), Some(theirs)) = (native["axes"].as_array(), oracle["axes"].as_array_mut()) else { return };
    for (a, b) in mine.iter().zip(theirs.iter_mut()) {
        let (Some(wanted), Some(children)) = (a["children"].as_array(), b["children"].as_array()) else { continue };
        let mut kept = Vec::new();
        let mut next = 0;
        for child in children {
            if next < wanted.len() && *child == wanted[next] { kept.push(child.clone()); next += 1; }
        }
        if next == wanted.len() { b["children"] = Json::Array(kept); }
    }
}

fn sha256(bytes: &[u8]) -> String {
    use sha2::Digest;
    sha2::Sha256::digest(bytes).iter().map(|b| format!("{b:02x}")).collect()
}

/// The recorded front, decoded from the corpus PNG (when present), for diff reports.
fn front_png(corpus: &Path, scenario: &str, file: &str) -> Option<Vec<u8>> {
    let file = std::fs::File::open(corpus.join(scenario).join(file)).ok()?;
    let mut decoder = png::Decoder::new(std::io::BufReader::new(file));
    decoder.set_transformations(png::Transformations::IDENTITY);
    let mut reader = decoder.read_info().ok()?;
    let mut buffer = vec![0; reader.output_buffer_size()?];
    let info = reader.next_frame(&mut buffer).ok()?;
    buffer.truncate(info.buffer_size());
    Some(buffer)
}

/// The bench's records: per present (series, points, present + compose ms),
/// and per front the renderer time since its renderer's previous front.
#[derive(Default)]
struct Bench { presents: Vec<(usize, usize, f64)>, fronts: Vec<(String, f64)>, pending: BTreeMap<u64, f64> }

/// The style the oracle was drawn in: v2 centred the image rail's count labels on
/// its ends; v3 hangs them inside the rail (style.render.image_distribution_counts_inside).
fn recorded_style() -> PlotStyle {
    let mut style = PlotStyle::default();
    style.render.image_distribution_counts_inside = false;
    style
}

fn run_scenario(dir: &Path, blobs: &Path, tally: &mut Tally, verbose: bool, pixels: bool, mut bench: Option<&mut Bench>) {
    let mut capture = Capture::load(dir);
    let name = dir.file_name().unwrap().to_string_lossy().to_string();
    let events = capture.document["events"].as_array().unwrap().clone();
    let style = Arc::new(recorded_style());
    let mut renderers: BTreeMap<u64, Renderer> = BTreeMap::new();
    let mut tainted: BTreeMap<u64, std::collections::BTreeSet<&'static str>> = BTreeMap::new();
    // Renderers whose spec is another kind: followed again from their next relayout.
    let mut foreign: BTreeMap<u64, bool> = BTreeMap::new();
    let mut shown = 0;
    // `ZLC_RENDER_KIND=image` follows only that kind (a bench of one panel kind).
    let only = std::env::var("ZLC_RENDER_KIND").ok().filter(|k| !k.is_empty());
    let followed = |spec: &PlotSpec| kind_for(spec).is_some() && only.as_ref().is_none_or(|k| format!("{:?}", spec.kind()).to_lowercase() == *k);
    for (position, event) in events.iter().enumerate() {
        // A pulse-ops capture's op markers belong to no renderer.
        let Some(rid) = event["rid"].as_u64() else { continue };
        let op = event["op"].as_str().unwrap();
        if foreign.get(&rid).copied().unwrap_or(false) && !matches!(op, "create" | "relayout" | "present" | "front") { continue; }
        let started = std::time::Instant::now();
        // Time that is not the front's: decoding a recorded payload, the presents bench's compose.

        let mut decoding = 0.0;
        match op {
            "create" => {
                if !event["default_style"].as_bool().unwrap() { tainted.entry(rid).or_default().insert("non-default style"); }
                let spec = spec_of(&event["spec"]);
                foreign.insert(rid, !followed(&spec));
                renderers.insert(rid, Renderer::new(spec, plan(&event["plan"]), style.clone()));
            }
            "relayout" => {
                // The session sets a new spec before it relayouts; the capture
                // writes it with the present that follows.
                let next = events[position + 1..].iter().find(|e| e["rid"].as_u64() == Some(rid) && e["op"] == "present");
                if let Some(spec) = next.and_then(|e| e.get("spec")) {
                    let spec = decode_spec(&dec(spec)).expect("spec");
                    foreign.insert(rid, kind_for(&spec).is_none());
                    renderers.get_mut(&rid).unwrap().set_spec(spec);
                }
                let a = &event["args"];
                let index = |v: &Json| v.as_u64().map(|v| v as usize);
                renderers.get_mut(&rid).unwrap().relayout(plan(&a["plan"]), index(&a["facet_index"]), index(&a["facet_focus_index"]))
                    .expect("relayout");
            }
            "present" => {
                let doc = &event["frame"];
                if let Some(spec) = event.get("spec") {
                    let spec = spec_of(spec);
                    match followed(&spec) {
                        true => {
                            let r = renderers.get_mut(&rid).unwrap();
                            if foreign[&rid] {
                                // The previous kind's presentations were not followed: start afresh.
                                let plan = r.plan.clone();
                                r.set_spec(spec);
                                r.relayout(plan, None, None).expect("relayout");
                            } else {
                                r.set_spec(spec);
                            }
                            foreign.insert(rid, false);
                        }
                        false => { renderers.get_mut(&rid).unwrap().set_spec(spec); foreign.insert(rid, true); }
                    }
                }
                if foreign[&rid] { continue; }
                let decoded = std::time::Instant::now();
                let frame = frame(&mut capture, doc);
                decoding = decoded.elapsed().as_secs_f64() * 1e3;
                let r = renderers.get_mut(&rid).unwrap();
                // `ZLC_RENDER_BENCH_KIND`: time only renderers of that kind (others still present).
                let timed = std::env::var("ZLC_RENDER_BENCH_KIND").map_or(true, |kind| r.spec.kind().value() == kind);
                if let (Some(bench), true) = (bench.as_deref_mut(), timed) {
                    // A grid is sized by its cells and its largest cell's points; a
                    // histogram by its groups and its bins.
                    let curves: Vec<&CurveData> = match &*frame.payload {
                        Payload::Curve(c) => vec![c],
                        Payload::Facet(f) => f.cells.iter().filter_map(|cell| match &cell.payload { Payload::Curve(c) => Some(c), _ => None }).collect(),
                        _ => Vec::new(),
                    };
                    let series = match &*frame.payload {
                        Payload::Facet(f) => f.cells.len(),
                        Payload::Histogram(h) => h.counts.shape()[0],
                        _ => curves.iter().map(|c| c.series.len()).sum(),
                    };
                    let points = match &*frame.payload {
                        Payload::Histogram(h) => h.counts.shape()[1],
                        _ => curves.iter().flat_map(|c| c.series.iter().map(|s| s.x.display.len())).max().unwrap_or(0),
                    };
                    // `ZLC_RENDER_PROFILE_FIRST`: the phases of each pass's first timed frame alone.
                    let first = bench.presents.is_empty() && std::env::var_os("ZLC_RENDER_PROFILE_FIRST").is_some();
                    if first { zlc_plot::render::profile::report(); }
                    let started = std::time::Instant::now();
                    r.present(&frame);
                    let presented = started.elapsed().as_secs_f64() * 1e3;
                    // ZLC_RENDER_BENCH_FRONTS: the fronts only (the present's compose is the front's).
                    if std::env::var_os("ZLC_RENDER_BENCH_FRONTS").is_none() { std::hint::black_box(r.compose().expect("compose")); }
                    let total = started.elapsed().as_secs_f64() * 1e3;
                    if std::env::var("ZLC_RENDER_SPLIT").is_ok() { eprintln!("SPLIT present {presented:.2} compose {:.2}", total - presented); }
                    if first {
                        for (phase, ms, count) in zlc_plot::render::profile::report() { eprintln!("FIRST {name} {phase}: {ms:.3} ms ({count} calls)"); }
                        eprintln!("FIRST {name} total {total:.3} ms (present {presented:.3})");
                    }
                    bench.presents.push((series, points, total));
                    // The front's own compose is timed where it is published.
                    decoding += total - presented;
                } else {
                    r.present(&frame);
                }
            }
            "series_focus" => {
                let a = &event["args"];
                let axes = a["axes"]["axes"].as_u64().map(|v| v as usize);
                let r = renderers.get_mut(&rid).unwrap();
                r.series_focus(a["action"].as_str().unwrap(), axes, f(&a["px"]["f"]), f(&a["py"]["f"]), f(&a["hit_radius"]["f"]),
                               a.get("click_radius").map_or(0.0, |c| f(&c["f"])));
            }
            "series_focus_scroll" => {
                let a = &event["args"];
                let axes = a["axes"]["axes"].as_u64().map(|v| v as usize);
                renderers.get_mut(&rid).unwrap().series_focus_scroll(axes, f(&a["step"]["f"]));
            }
            "restore_series_interaction" => {
                let a = &event["args"];
                let p = if a["presentation"].is_null() { None } else { Some(presentation(&dec(&a["presentation"]))) };
                renderers.get_mut(&rid).unwrap().restore_series_interaction(&interaction(&dec(&a["interaction"])), p.as_ref());
            }
            "begin_selector_gesture" => {
                let a = &event["args"];
                let kind = SelectorKind::parse(&Value::text(a["kind"]["enum"].as_str().unwrap())).unwrap();
                PanelRenderer::begin_selector_gesture(renderers.get_mut(&rid).unwrap(), kind, a.get("compose").and_then(Json::as_bool).unwrap_or(true)).unwrap();
            }
            "preview_selector" => { renderers.get_mut(&rid).unwrap().preview_selector(&selector(&event["args"]["state"]["selector"])); }
            "end_selector_gesture" => PanelRenderer::end_selector_gesture(renderers.get_mut(&rid).unwrap()),
            "set_view_dragging" => {
                let axes = event["args"]["axes"]["axes"].as_u64().map(|v| v as usize);
                renderers.get_mut(&rid).unwrap().set_view_dragging(axes);
            }
            "draw" | "rgba" | "capture_rgba_bytes" | "capture_gesture_background" => {}
            "set_height_bars_dragging" => {
                let dragging = event["args"]["dragging"].as_bool().unwrap();
                PanelRenderer::set_height_bars_dragging(renderers.get_mut(&rid).unwrap(), dragging).unwrap();
            }
            "begin_color_limit_gesture" | "preview_color_limit_candidate" => {
                let limits = &event["args"]["candidate"]["color_limits"];
                let candidate = NumericRange::new(f(&limits[0]), f(&limits[1])).unwrap();
                let r = renderers.get_mut(&rid).unwrap();
                if op == "begin_color_limit_gesture" { PanelRenderer::begin_color_limit_gesture(r, candidate).unwrap(); }
                else { PanelRenderer::preview_color_limit_candidate(r, candidate).unwrap(); }
            }
            "preview_color_limits" => {
                let a = &event["args"];
                renderers.get_mut(&rid).unwrap().preview_color_limits(f(&a["low"]["f"]), f(&a["high"]["f"])).unwrap();
            }
            "set_classifier_labels" => {
                let labels: Vec<String> = event["args"]["labels"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect();
                PanelRenderer::set_classifier_labels(renderers.get_mut(&rid).unwrap(), &labels);
            }
            "save" => {
                // An export renderer: its png's pixels against the saved file.
                let a = &event["args"];
                let r = renderers.get_mut(&rid).unwrap();
                // A saved file shows every layer: one with another layer's artists is not this one's alone.
                let overlaid = tainted.get(&rid).is_some_and(|t| !t.is_empty());
                if a["format"] == "png" && overlaid { tally.exports_overlaid += 1; }
                if a["format"] == "png" && !foreign[&rid] && !overlaid {
                    let corpus = PathBuf::from(std::env::var("ZLC_RENDER_CORPUS").unwrap_or_default());
                    let file = if matches!(r.spec, PlotSpec::Image(_)) { "image.png" } else { "curve.png" };
                    let saved = corpus.join(&name).join("saved").join(file);
                    if saved.exists() {
                        let png = r.export_png(f(&a["dpi"]["f"])).expect("export");
                        let decode = |bytes: &[u8]| {
                            let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes.to_vec()));
                            decoder.set_transformations(png::Transformations::EXPAND);
                            let mut reader = decoder.read_info().unwrap();
                            let mut buffer = vec![0; reader.output_buffer_size().unwrap()];
                            let info = reader.next_frame(&mut buffer).unwrap();
                            buffer.truncate(info.buffer_size());
                            (info.width, info.height, buffer)
                        };
                        let mine = decode(&png);
                        let theirs = decode(&std::fs::read(&saved).unwrap());
                        tally.exports += 1;
                        if mine == theirs { tally.exports_exact += 1; } else {
                            let differ = mine.2.chunks(4).zip(theirs.2.chunks(4)).filter(|(p, q)| p != q).count();
                            eprintln!("=== {name} export {file} {}x{} vs {}x{}: {differ} pixels differ", mine.0, mine.1, theirs.0, theirs.1);
                            if let Ok(dump) = std::env::var("ZLC_RENDER_DUMP") {
                                std::fs::create_dir_all(&dump).unwrap();
                                std::fs::write(PathBuf::from(dump).join(format!("{name}.export.{file}")), &png).unwrap();
                            }
                        }
                    }
                }
                tainted.entry(rid).or_default().insert("save");
            }
            "front" => {
                if let Some(bench) = bench.as_deref_mut() {
                    let timed = std::env::var("ZLC_RENDER_BENCH_KIND").map_or(true, |kind| renderers[&rid].spec.kind().value() == kind);
                    if foreign[&rid] || !timed { continue; }
                    // `PanelRenderer::present(.., compose=true)` then `rgba()`, as the Python bench times them.
                    let r = renderers.get_mut(&rid).unwrap();
                    r.compose().expect("compose");
                    std::hint::black_box(PanelRenderer::rgba(r).unwrap());
                    let ms = bench.pending.remove(&rid).unwrap_or(0.0) + started.elapsed().as_secs_f64() * 1e3;
                    bench.fronts.push((format!("{}|{}", event["command"].as_str().unwrap(), event["panel"].as_str().unwrap()), ms));
                    continue;
                }
                tally.fronts += 1;
                if foreign[&rid] || event.get("other_kind").is_some() { *tally.skipped.entry("other kind".into()).or_default() += 1; continue; }
                // Every front is a composed frame, as the live panel's are
                // (a compose leaves state behind: where axis labels landed).
                let front = renderers.get_mut(&rid).unwrap().compose().expect("compose");
                // `ZLC_RENDER_HASHES=file`: every front's digest appended, for a
                // byte-for-byte comparison of two builds.
                if let Some(path) = std::env::var_os("ZLC_RENDER_HASHES") {
                    use std::io::Write;
                    let mut file = std::fs::OpenOptions::new().create(true).append(true).open(path).unwrap();
                    writeln!(file, "{name} {} {} {}", event["command"], event["panel"], sha256(front.pixels.as_slice())).unwrap();
                }
                // The image queries the session asks: the painted clim, the 3-D scene and its camera.
                if let Some(clim) = event.get("clim") {
                    let r = &renderers[&rid];
                    let expected = clim.as_array().map(|c| (f(&c[0]), f(&c[1])));
                    let camera = event["height_bars"].as_array().map(|c| (f(&c[0]), f(&c[1]), f(&c[2])));
                    let native_camera = PanelRenderer::height_bars_scene(r).then(|| PanelRenderer::height_bars_camera(r)).flatten()
                        .map(|c| (c.azimuth_deg, c.elevation_deg, c.zoom));
                    tally.queries += 1;
                    if PanelRenderer::resolved_color_limits(r) == expected && native_camera == camera { tally.queries_exact += 1; } else if verbose || shown < 3 {
                        shown += 1;
                        eprintln!("??? {name} {} queries: clim {:?} vs {expected:?}, camera {native_camera:?} vs {camera:?}", event["command"], PanelRenderer::resolved_color_limits(r));
                    }
                }
                if let Some(facts) = event.get("histogram") {
                    let (agree, of) = histogram_scales(&renderers[&rid], facts);
                    tally.scale_queries += of;
                    tally.scale_queries_exact += agree;
                }
                // Selector artists are another layer's: the IR is compared with
                // the oracle's extra children taken out (and no pixels).
                let reasons = tainted.get(&rid).cloned().unwrap_or_default();
                let overlay_tolerant = !reasons.is_empty() && reasons.iter().all(|r| *r == "selectors");
                if !reasons.is_empty() && !overlay_tolerant {
                    let reason = reasons.iter().find(|r| **r != "selectors").unwrap();
                    *tally.skipped.entry(reason.to_string()).or_default() += 1;
                    continue;
                }
                let Some(ir_file) = event["ir"].as_str() else { *tally.skipped.entry("unsupported IR".into()).or_default() += 1; continue; };
                zlc_figure::wire::set_blob_root(Some(blobs.to_path_buf()));
                let oracle: zlc_figure::ir::Figure = serde_json::from_slice(&std::fs::read(dir.join(ir_file)).unwrap()).unwrap();
                let renderer = renderers.get_mut(&rid).unwrap();
                if pixels && !overlay_tolerant {
                    tally.pixels_compared += 1;
                    if sha256(front.pixels.as_slice()) == event["sha256"].as_str().unwrap() { tally.pixels_exact += 1; } else {
                        *tally.pixels_failed.entry(name.clone()).or_default() += 1;
                        let corpus = PathBuf::from(std::env::var("ZLC_RENDER_CORPUS").unwrap_or_default());
                        // A pulse-ops front is op NNNNN's PNG; a document front is per panel.
                        let (command, panel) = match event["command"].as_u64() {
                            Some(op) => (format!("{op:05}"), "pulse"),
                            None => (event["command"].as_str().unwrap().to_owned(), event["panel"].as_str().unwrap()),
                        };
                        let png = if event["command"].is_u64() { format!("{command}.png") } else { format!("{command}.{panel}.png") };
                        if let Some(expected) = front_png(&corpus, &name, &png) {
                            let mut count = 0usize;
                            let mut worst = 0u8;
                            let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
                            for (i, (a, b)) in front.pixels.as_slice().chunks(4).zip(expected.chunks(4)).enumerate() {
                                if a != b {
                                    count += 1;
                                    worst = worst.max(a.iter().zip(b).map(|(p, q)| p.abs_diff(*q)).max().unwrap());
                                    let (x, y) = (i % front.width, i / front.width);
                                    x0 = x0.min(x); y0 = y0.min(y); x1 = x1.max(x); y1 = y1.max(y);
                                }
                            }
                            // Where the native front and the recorded one differ, the native
                            // front against matplotlib's full draw of the same (IR-exact)
                            // figure: agreeing on every such pixel means the recorded front
                            // shows what a full draw of Python's own state does not.
                            let full = renderer.full_draw(renderer.fig.dpi).map(|(_, _, full)| {
                                front.pixels.as_slice().chunks(4).zip(expected.chunks(4)).zip(full.chunks(4)).all(|((a, b), c)| a == b || a == c)
                            }).unwrap_or(false);
                            if full { tally.pixels_full_draw += 1; }
                            if verbose || shown < 3 {
                                shown += 1;
                                let note = if full { "native = full draw there" } else { "native != full draw there" };
                                eprintln!("~~~ {name} {command} {panel}: {count} pixels differ (max {worst}) in x {x0}..{x1} y {y0}..{y1}; {note}");
                            }
                            if let Ok(dump) = std::env::var("ZLC_RENDER_DUMP") {
                                let write = |path: PathBuf, pixels: &[u8]| {
                                    let file = std::fs::File::create(path).unwrap();
                                    let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), front.width as u32, front.height as u32);
                                    encoder.set_color(png::ColorType::Rgba);
                                    encoder.set_depth(png::BitDepth::Eight);
                                    encoder.write_header().unwrap().write_image_data(pixels).unwrap();
                                };
                                let dir = PathBuf::from(dump);
                                std::fs::create_dir_all(&dir).unwrap();
                                write(dir.join(format!("{name}.{command}.{panel}.native.png")), front.pixels.as_slice());
                                write(dir.join(format!("{name}.{command}.{panel}.front.png")), &expected);
                                if let Ok((_, _, full)) = renderer.full_draw(renderer.fig.dpi) {
                                    write(dir.join(format!("{name}.{command}.{panel}.full.png")), &full);
                                }
                                let diff: Vec<u8> = front.pixels.as_slice().chunks(4).zip(expected.chunks(4))
                                    .flat_map(|(a, b)| if a == b { [255, 255, 255, 255] } else { [255, 0, 0, 255] }).collect();
                                write(dir.join(format!("{name}.{command}.{panel}.diff.png")), &diff);
                            }
                        }
                    }
                }
                let renderer = renderers.get(&rid).unwrap();
                let native = renderer.materialized_ir();
                let a = serde_json::to_value(&native).unwrap();
                let b = serde_json::to_value(&oracle).unwrap();
                let mut a = resolve(&a, a["nodes"].as_array().unwrap(), None);
                let mut b = resolve(&b, b["nodes"].as_array().unwrap(), None);
                blank_child_residue(&mut a);
                blank_child_residue(&mut b);
                blank_undrawn_positions(&mut a, &mut b);
                if overlay_tolerant { strip_overlay_children(&a, &mut b); }
                let mut out = Vec::new();
                diff(&a, &b, "", &mut out, &residue);
                tally.compared += 1;
                if overlay_tolerant { tally.overlay_compared += 1; }
                if out.is_empty() { tally.exact += 1; if overlay_tolerant { tally.overlay_exact += 1; } } else {
                    *tally.failed.entry(name.clone()).or_default() += 1;
                    if verbose || shown < 1 {
                        shown += 1;
                        eprintln!("--- {name} {} {}: {} differences", event["command"], event["panel"], out.len());
                        for line in out.iter().take(25) { eprintln!("    {line}"); }
                    }
                }
            }
            other => panic!("unknown event {other}"),
        }
        if let Some(bench) = bench.as_deref_mut() {
            if op != "save" { *bench.pending.entry(rid).or_default() += started.elapsed().as_secs_f64() * 1e3 - decoding; }

        }
    }
}

#[test]
fn renderer_matches_the_recorded_oracle() {
    let Ok(root) = std::env::var("ZLC_RENDER_ORACLE") else { return };
    let root = PathBuf::from(root);
    let blobs = std::env::var("ZLC_RENDER_BLOBS").map(PathBuf::from).unwrap_or_else(|_| root.join("blobs"));
    let only: Vec<String> = std::env::var("ZLC_RENDER_ONLY").map(|v| v.split(',').map(str::to_owned).collect()).unwrap_or_default();
    let verbose = std::env::var("ZLC_RENDER_VERBOSE").is_ok();
    let pixels = std::env::var("ZLC_RENDER_PIXELS").is_ok();
    let mut tally = Tally::default();
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(&root).unwrap().map(|e| e.unwrap().path()).filter(|p| p.join("capture.json").exists()).collect();
    dirs.sort();
    for dir in dirs {
        let name = dir.file_name().unwrap().to_string_lossy().to_string();
        if !only.is_empty() && !only.iter().any(|prefix| name.starts_with(prefix.as_str())) { continue; }
        run_scenario(&dir, &blobs, &mut tally, verbose, pixels, None);
    }
    eprintln!("fronts {} compared {} exact {} skipped {:?}", tally.fronts, tally.compared, tally.exact, tally.skipped);
    eprintln!("of which with selector artists stripped: compared {} exact {}", tally.overlay_compared, tally.overlay_exact);
    eprintln!("png exports compared {} pixel-exact {}; skipped with another layer's artists {}", tally.exports, tally.exports_exact, tally.exports_overlaid);
    eprintln!("image queries compared {} exact {}", tally.queries, tally.queries_exact);
    eprintln!("histogram axis scales compared {} exact {}", tally.scale_queries, tally.scale_queries_exact);
    eprintln!("failed per scenario {:?}", tally.failed);
    if pixels {
        eprintln!("pixels compared {} exact {}; of the rest, equal to the full draw where they differ {}", tally.pixels_compared, tally.pixels_exact, tally.pixels_full_draw);
        eprintln!("pixel failures per scenario {:?}", tally.pixels_failed);
    }
    assert_eq!(tally.exact, tally.compared, "IR mismatches");
    assert_eq!(tally.scale_queries_exact, tally.scale_queries, "histogram axis scale mismatches");
}

/// `ZLC_RENDER_BENCH=scenario,...`: time present + compose per frame (the
/// Python side's present composes and publishes), several passes, medians.
#[test]
fn renderer_bench() {
    let Ok(names) = std::env::var("ZLC_RENDER_BENCH") else { return };
    let root = PathBuf::from(std::env::var("ZLC_RENDER_ORACLE").expect("ZLC_RENDER_ORACLE"));
    let blobs = std::env::var("ZLC_RENDER_BLOBS").map(PathBuf::from).unwrap_or_else(|_| root.join("blobs"));
    for name in names.split(',') {
        let passes = std::env::var("ZLC_RENDER_BENCH_PASSES").ok().and_then(|v| v.parse().ok()).unwrap_or(5);
        let mut all: Vec<Vec<(usize, usize, f64)>> = Vec::new();
        let mut fronts: Vec<Vec<(String, f64)>> = Vec::new();
        for _ in 0..passes {
            let mut bench = Bench::default();
            run_scenario(&root.join(name), &blobs, &mut Tally::default(), false, false, Some(&mut bench));
            all.push(bench.presents);
            fronts.push(bench.fronts);
        }
        // Per frame, the best of the passes after the first (warm caches and fonts).
        let frames = all[0].len();
        let best: Vec<(usize, usize, f64)> = (0..frames).map(|i| {
            let ms = all[1..].iter().map(|pass| pass[i].2).fold(f64::INFINITY, f64::min);
            (all[0][i].0, all[0][i].1, ms)
        }).collect();
        let mut sorted: Vec<f64> = best.iter().map(|b| b.2).collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = sorted[sorted.len() / 2];
        eprintln!("BENCH {name} frames {frames} median {median:.2} ms max {:.2} ms", sorted[sorted.len() - 1]);
        for (phase, ms, count) in zlc_plot::render::profile::report() {
            eprintln!("BENCH {name} phase {phase}: {:.3} ms per call ({count} calls)", ms / count as f64);
        }
        eprintln!("BENCH {name} last {:?}", best.iter().rev().take(8).rev().map(|b| (b.0, b.1, (b.2 * 100.0).round() / 100.0)).collect::<Vec<_>>());
        if let Ok(out) = std::env::var("ZLC_RENDER_BENCH_OUT") {
            std::fs::write(PathBuf::from(&out).join(format!("{name}.native.json")), serde_json::to_string(&best).unwrap()).unwrap();
            // Every warm pass of every present, for medians and minima beside Python's runs.
            let runs: Vec<Vec<f64>> = all[1..].iter().map(|pass| pass.iter().map(|t| t.2).collect()).collect();
            std::fs::write(PathBuf::from(&out).join(format!("{name}.native.runs.json")), serde_json::to_string(&runs).unwrap()).unwrap();
            // Every warm pass of every front, for medians and minima.
            let per_front: Vec<(String, Vec<f64>)> = fronts[0].iter().enumerate()
                .map(|(i, (key, _))| (key.clone(), fronts[1..].iter().map(|pass| pass[i].1).collect())).collect();
            std::fs::write(PathBuf::from(out).join(format!("{name}.native.fronts.json")), serde_json::to_string(&per_front).unwrap()).unwrap();
        }
    }
}
