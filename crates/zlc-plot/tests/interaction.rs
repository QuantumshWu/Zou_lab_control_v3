//! The interaction model against the real zlc_plot objects: selector numbers,
//! the bounded range drag, the selector controller replayed through random
//! pointer sequences (every intermediate state), bit for bit.
//! Committed: `tests/data/interaction.json.gz` (`gen/interaction_cases.py OUT.json.gz`).

use serde_json::{json, Value as J};
use zlc_plot::configuration::error::PlotError;
use zlc_plot::configuration::selectors::{CrosshairPoint, NumericRange, RectangleRange, SelectorKind, SelectorState, SelectorValue, Viewport};
use zlc_plot::configuration::value::Value;
use zlc_plot::interaction::*;
use zlc_plot::kinds::{AxisDomain, AxisRef};
use zlc_plot::render::scale::AxisScale;
use zlc_plot::units::DEFAULT_UNITS;

fn load() -> J {
    let bytes = std::fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/interaction.json.gz")).unwrap();
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn f(v: &J) -> f64 { f64::from_bits(v.as_u64().expect("float bits")) }
fn b(x: f64) -> J { json!(x.to_bits()) }
fn range(v: &J) -> Option<NumericRange> { (!v.is_null()).then(|| NumericRange { low: f(&v[0]), high: f(&v[1]) }) }
fn point(v: &J) -> CrosshairPoint { CrosshairPoint { x: f(&v[0]), y: f(&v[1]) } }
fn enc_range(r: Option<NumericRange>) -> J { r.map_or(J::Null, |r| json!([b(r.low), b(r.high)])) }
fn enc_point(p: CrosshairPoint) -> J { json!([b(p.x), b(p.y)]) }
fn err(e: PlotError) -> J { json!({"error": e.python_name(), "message": e.message}) }
fn ok(v: J) -> J { json!({"ok": v}) }
fn outcome<T>(r: Result<T, PlotError>, enc: impl FnOnce(T) -> J) -> J { match r { Ok(v) => ok(enc(v)), Err(e) => err(e) } }

fn kind(v: &J) -> SelectorKind { SelectorKind::ALL.into_iter().find(|k| v.as_str() == Some(k.value())).expect("kind") }
fn handle(v: &J) -> DragHandle { DragHandle::parse(v.as_str().unwrap()).expect("handle") }

fn value(kind: SelectorKind, v: &J) -> SelectorValue {
    match kind {
        SelectorKind::XRange => SelectorValue::XRange(range(v).unwrap()),
        SelectorKind::Area => SelectorValue::Area(RectangleRange { x: range(&v[0]).unwrap(), y: range(&v[1]).unwrap() }),
        SelectorKind::Crosshair => SelectorValue::Crosshair(point(v)),
        SelectorKind::Threshold => SelectorValue::Threshold(f(v)),
    }
}

fn enc_value(v: &SelectorValue) -> J {
    match v {
        SelectorValue::XRange(r) => enc_range(Some(*r)),
        SelectorValue::Area(r) => json!([enc_range(Some(r.x)), enc_range(Some(r.y))]),
        SelectorValue::Crosshair(p) => enc_point(*p),
        SelectorValue::Threshold(t) => b(*t),
    }
}

fn state(v: &J) -> Option<SelectorState> {
    (!v.is_null()).then(|| {
        let k = kind(&v[0]);
        SelectorState { value: value(k, &v[1]), revision: v[2].as_u64().unwrap(), facet_index: v[3].as_u64() }
    })
}

fn st(s: Option<SelectorState>) -> J {
    s.map_or(J::Null, |s| json!([s.kind().value(), enc_value(&s.value), s.revision, s.facet_index]))
}

fn scale(v: &J) -> AxisScale {
    match v {
        J::String(name) if name == "linear" => AxisScale::Linear,
        J::String(name) if name == "log" => AxisScale::Log,
        J::Array(items) if items[0] == "samples" => AxisScale::Samples(items[1].as_array().unwrap().iter().map(f).collect::<Vec<_>>().into()),
        J::Array(items) if items[0] == "converted" => AxisScale::Converted {
            mapping: Box::new(scale(&items[1])),
            canonical: DEFAULT_UNITS.resolve(items[2].as_str().unwrap()).unwrap(),
            display: DEFAULT_UNITS.resolve(items[3].as_str().unwrap()).unwrap(),
        },
        other => panic!("scale {other}"),
    }
}

#[test]
fn selector_numbers_match() {
    let data = load();
    for c in data["precision"].as_array().unwrap() {
        assert_eq!(outcome(selector_precision(f(&c[0])), |p| json!(p)), c[1], "precision {c}");
    }
    for c in data["number"].as_array().unwrap() {
        assert_eq!(selector_number(f(&c[0]), c[1].as_u64().unwrap() as usize), c[2].as_str().unwrap(), "number {c}");
    }
    for c in data["series_identity"].as_array().unwrap() {
        let key: Vec<(AxisRef, Value)> = c[0].as_array().unwrap().iter().map(|item| {
            let domain = AxisDomain::parse(item[0].as_str().unwrap()).unwrap();
            let canonical = &item[2];
            let canonical = match canonical[0].as_str().unwrap() {
                "bool" => Value::Bool(canonical[1].as_bool().unwrap()),
                "int" => Value::Int(canonical[1].as_str().unwrap().parse().unwrap()),
                "float" => Value::Float(f(&canonical[1])),
                _ => Value::text(canonical[1].as_str().unwrap()),
            };
            (AxisRef { domain, axis_id: item[1].as_str().unwrap().to_owned() }, canonical)
        }).collect();
        let identity: Vec<J> = series_identity(&key).into_iter().map(|(d, a, r)| json!([d, a, r])).collect();
        assert_eq!(J::Array(identity), c[1], "identity {c}");
        assert_eq!(series_identity_repr(&key), c[2].as_str().unwrap(), "identity repr {c}");
    }
}

#[test]
fn range_drags_match() {
    let data = load();
    for c in data["drag"].as_array().unwrap() {
        let got = drag_numeric_range(range(&c[0]).unwrap(), handle(&c[1]), f(&c[2]), f(&c[3]), (!c[4].is_null()).then(|| f(&c[4])),
                                     range(&c[5]), &scale(&c[6]));
        assert_eq!(outcome(got, enc_range), c[7], "drag {c}");
    }
}

fn gesture(g: Option<GestureState>) -> J {
    g.map_or(J::Null, |g| json!([g.kind.value(), g.handle.value(), enc_point(g.origin), enc_value(&g.original_value),
                                 enc_value(&g.candidate_value), g.revision]))
}

fn dump(c: &SelectorController) -> J {
    json!({
        "states": c.states().into_iter().map(|s| st(Some(s))).collect::<Vec<_>>(),
        "revisions": SelectorKind::ALL.iter().map(|k| json!(c.revision(*k))).collect::<Vec<_>>(),
        "draft": st(c.draft()),
        "gesture": gesture(c.active_gesture()),
    })
}

struct Frame { x_bounds: Option<NumericRange>, y_bounds: Option<NumericRange>, x_scale: AxisScale, y_scale: AxisScale }

impl Frame {
    fn of(v: &J) -> Self { Frame { x_bounds: range(&v[0]), y_bounds: range(&v[1]), x_scale: scale(&v[2]), y_scale: scale(&v[3]) } }
    fn drag(&self) -> DragFrame<'_> { DragFrame { x_bounds: self.x_bounds, y_bounds: self.y_bounds, x_scale: &self.x_scale, y_scale: &self.y_scale } }
}

#[test]
fn controller_replays_match() {
    let data = load();
    let mut steps = 0;
    for (number, sequence) in data["controller"].as_array().unwrap().iter().enumerate() {
        let mut c = SelectorController::new();
        let mut handles: Vec<HeldSelector> = Vec::new();
        fn register(held: Option<HeldSelector>, handles: &mut Vec<HeldSelector>) -> J {
            match held { None => J::Null, Some(held) => { handles.push(held); json!(handles.len() - 1) } }
        }
        for step in sequence.as_array().unwrap() {
            let (op, want, after) = (&step[0], &step[1], &step[2]);
            let got = match op[0].as_str().unwrap() {
                "install" => match c.install(state(&op[1]).unwrap()) {
                    Ok((previous, stored)) => {
                        let (ip, is) = (register(previous, &mut handles), register(Some(stored), &mut handles));
                        ok(json!([ip, is, st(previous.map(|h| h.state())), st(Some(stored.state()))]))
                    }
                    Err(e) => err(e),
                },
                "commit" => match c.commit_finished(state(&op[1]).unwrap()) {
                    Ok((previous, stored)) => {
                        let (ip, is) = (register(previous, &mut handles), register(Some(stored), &mut handles));
                        ok(json!([ip, is, st(previous.map(|h| h.state())), st(Some(stored.state()))]))
                    }
                    Err(e) => err(e),
                },
                "remove" => match c.remove(kind(&op[1])) {
                    Ok(removed) => { let i = register(Some(removed), &mut handles); ok(json!([i, st(Some(removed.state()))])) }
                    Err(e) => err(e),
                },
                "restore" => outcome(c.restore_removed(handles[op[1].as_u64().unwrap() as usize]), |_| J::Null),
                "rollback" => {
                    let held = handles[op[1].as_u64().unwrap() as usize];
                    let installed = if op[2].as_bool().unwrap() { HeldSelector::new(held.state()) } else { held };
                    let previous = op[3].as_u64().map(|j| handles[j as usize]);
                    outcome(c.rollback_install(&installed, previous), |_| J::Null)
                }
                "grab" => {
                    let k = kind(&op[1]);
                    match c.state(k) {
                        Err(e) => err(e),
                        Ok(_) => { let held = c.held().into_iter().find(|h| h.kind() == k); ok(register(held, &mut handles)) }
                    }
                }
                "down" => {
                    let k = kind(&op[1]);
                    outcome(c.pointer_down(k, f(&op[2]), f(&op[3]), handle(&op[4]), state(&op[5])), |_| J::Null)
                }
                "move" => { let frame = Frame::of(&op[3]); outcome(c.pointer_move(f(&op[1]), f(&op[2]), &frame.drag()), st) }
                "finish" => { let frame = Frame::of(&op[3]); outcome(c.finish_gesture(f(&op[1]), f(&op[2]), &frame.drag()), st) }
                "cancel" => outcome(c.cancel(), st),
                "lost" => outcome(c.lost_pointer_capture(), st),
                "candidate" => outcome(c.candidate_state(), st),
                "snapshot" => {
                    let include = op[1].as_bool().unwrap();
                    outcome(c.snapshot(include), |snapshot| {
                        json!([snapshot.committed.iter().map(|s| st(Some(*s))).collect::<Vec<_>>(), st(snapshot.candidate),
                               snapshot.states().into_iter().map(|s| st(Some(s))).collect::<Vec<_>>()])
                    })
                }
                "fork" => { let forked = c.fork(); if op[1].as_bool().unwrap() { c = forked; } ok(J::Null) }
                "retarget" => {
                    let kinds: Vec<SelectorKind> = op[2].as_array().unwrap().iter().map(kind).collect();
                    match c.retarget_facet(op[1].as_u64().unwrap(), &kinds) { Ok(moved) => { c = moved; ok(J::Null) } Err(e) => err(e) }
                }
                other => panic!("op {other}"),
            };
            assert_eq!(got, *want, "sequence {number} op {op}");
            assert_eq!(dump(&c), *after, "sequence {number} after {op}");
            steps += 1;
        }
    }
    assert!(steps > 10_000, "{steps} steps");
}


fn pair(v: &J) -> (f64, f64) { (f(&v[0]), f(&v[1])) }
fn enc_pair(p: (f64, f64)) -> J { json!([b(p.0), b(p.1)]) }
fn opt_scale(v: &J) -> Option<AxisScale> { (!v.is_null()).then(|| scale(v)) }
fn enc_hit(h: Option<(f64, DragHandle)>) -> J { h.map_or(J::Null, |(score, handle)| json!([b(score), handle.value()])) }

fn transform(v: &J) -> AxisTransform {
    let bounds: Vec<f64> = v[2].as_array().unwrap().iter().map(f).collect();
    AxisTransform {
        role: v[0].as_str().unwrap().to_owned(), cell_index: v[1].as_u64(), bounds: (bounds[0], bounds[1], bounds[2], bounds[3]),
        x_limits: pair(&v[3]), y_limits: pair(&v[4]), canonical_x_limits: pair(&v[5]), canonical_y_limits: pair(&v[6]),
        x_scale: scale(&v[7]), y_scale: scale(&v[8]), canonical_x_scale: opt_scale(&v[9]), canonical_y_scale: opt_scale(&v[10]),
    }
}

fn scale_json(s: &AxisScale) -> J {
    match s {
        AxisScale::Linear => json!("linear"),
        AxisScale::Log => json!("log"),
        AxisScale::Samples(lattice) => json!(["samples", lattice.iter().map(|v| b(*v)).collect::<Vec<_>>()]),
        AxisScale::Converted { mapping, canonical, display } => json!(["converted", scale_json(mapping), canonical.symbol, display.symbol]),
    }
}

#[test]
fn axis_transforms_match() {
    let data = load();
    let mut queries = 0;
    for c in data["transform"].as_array().unwrap() {
        let t = transform(&c[0]);
        let (w, h) = pair(&c[1]);
        let canvas = CanvasSize::new(w, h);
        assert_eq!(json!([scale_json(t.interaction_scale(Axis::X)), scale_json(t.interaction_scale(Axis::Y))]), c[2], "scales {c}");
        assert_eq!(json!([b(t.drawn_span(Axis::X)), b(t.drawn_span(Axis::Y))]), c[3], "spans {c}");
        for q in c[4].as_array().unwrap() {
            let event = (!q[1].is_null()).then(|| pair(&q[1]));
            let (x, y) = event.unwrap_or((0.0, 0.0));
            let got = match q[0].as_str().unwrap() {
                "to_normalized" => ok(enc_pair(t.display_to_normalized(x, y))),
                "canonical_values" => outcome(t.canonical_values(x, y), enc_pair),
                "canonical_point" => outcome(t.canonical_from_normalized(x, y), enc_point),
                "display_point_n" => outcome(t.display_from_normalized(x, y), enc_point),
                "reading" => match event {
                    None => ok(J::Null),
                    Some(_) => outcome(canvas.clone().and_then(|canvas| t.canonical_reading(event, canvas)), |r| r.map_or(J::Null, enc_pair)),
                },
                "display_point" => match event {
                    None => ok(J::Null),
                    Some(_) => outcome(canvas.clone().and_then(|canvas| t.display_point(event, canvas)), |r| r.map_or(J::Null, enc_point)),
                },
                "to_pixel" => outcome(canvas.clone().map(|canvas| t.display_to_pixel(x, y, canvas)), enc_pair),
                other => panic!("query {other}"),
            };
            assert_eq!(got, q[2], "transform {} query {q}", c[0]);
            queries += 1;
        }
    }
    assert!(queries > 8_000);
}

#[test]
fn gesture_engine_matches() {
    let data = load();
    for c in data["endpoint"].as_array().unwrap() {
        let got = range_endpoint_hit(range(&c[0]).unwrap(), f(&c[1]), f(&c[2]), &scale(&c[3]));
        assert_eq!(outcome(got, enc_hit), c[4], "endpoint {c}");
    }
    for c in data["area_handle"].as_array().unwrap() {
        let t = transform(&c[0]);
        let (w, h) = pair(&c[1]);
        let canvas = CanvasSize::new(w, h).unwrap();
        let value = RectangleRange { x: range(&c[2][0]).unwrap(), y: range(&c[2][1]).unwrap() };
        let pixel_point = |(x, y): (f64, f64)| -> Result<(f64, f64), PlotError> {
            Ok(t.display_to_pixel(pymath::placed(&t.x_scale, x, t.x_limits), pymath::placed(&t.y_scale, y, t.y_limits), canvas))
        };
        let got = area_drag_handle(value, pair(&c[3]), pixel_point, f(&c[4]));
        assert_eq!(outcome(got, enc_hit), c[5], "area handle {c}");
    }
    for c in data["pan"].as_array().unwrap() {
        let got = pan_rectangle(point(&c[0]), point(&c[1]), range(&c[2]).unwrap(), range(&c[3]).unwrap(), c[4].as_bool().unwrap(),
                                &scale(&c[5]), &scale(&c[6]));
        assert_eq!(outcome(got, |r| r.map_or(J::Null, |r| enc_value(&SelectorValue::Area(r)))), c[7], "pan {c}");
    }
    for c in data["color_drag"].as_array().unwrap() {
        let value = range(&c[0]).unwrap();
        let mut drag = ColorLimitDrag { original: value, candidate: value, handle: handle(&c[2]), origin: f(&c[3]),
                                        bounds: range(&c[1]).unwrap(), minimum_span: f(&c[4]) };
        for step in c[5].as_array().unwrap() {
            let got = match drag.moved(f(&step[0])) {
                Ok(moved) => { drag = moved; ok(json!([enc_range(Some(drag.candidate)), drag.changed()])) }
                Err(e) => err(e),
            };
            assert_eq!(got, step[1], "colour drag {c}");
        }
    }
}

fn rgba(v: &J) -> Rgba { (f(&v[0]), f(&v[1]), f(&v[2]), f(&v[3])) }
fn enc_rgba(c: Option<Rgba>) -> J { c.map_or(J::Null, |c| json!([b(c.0), b(c.1), b(c.2), b(c.3)])) }
fn target(v: &J) -> SelectorTarget { SelectorTarget { role: v[0].as_str().unwrap().to_owned(), cell_index: v[1].as_u64() } }
fn enc_target(t: &SelectorTarget) -> J { json!([t.role, t.cell_index]) }
fn enc_points(points: &[(f64, f64)]) -> J { J::Array(points.iter().map(|p| enc_pair(*p)).collect()) }

fn scene_kind(v: &J) -> SceneKind {
    if v == "color_limits" { SceneKind::ColorLimits } else { SceneKind::Selector(kind(v)) }
}

fn context(v: &J) -> Option<SelectorItemContext> {
    (!v.is_null()).then(|| SelectorItemContext {
        kind: scene_kind(&v[0]), target: target(&v[1]), label_target: target(&v[2]), x_limits: pair(&v[3]), y_limits: pair(&v[4]),
        selector_rgba: rgba(&v[5]), color_limit_rgba: (rgba(&v[6][0]), rgba(&v[6][1])), threshold_rgba: rgba(&v[7]),
        threshold_label_rgba: rgba(&v[8]), threshold_uses_x: v[9].as_bool().unwrap(), threshold_text: v[10].as_str().unwrap().to_owned(),
        x_scale: scale(&v[11]), y_scale: scale(&v[12]), x_label_factor: f(&v[13]),
        sample_value: (!v[14].is_null()).then(|| if v[14][0] == "number" { SampleValue::Number(f(&v[14][1])) } else { SampleValue::Text(v[14][1].as_str().unwrap().to_owned()) }),
        sample_span: (!v[15].is_null()).then(|| f(&v[15])), sample_target: (!v[16].is_null()).then(|| target(&v[16])),
        sample_x_limits: (!v[17].is_null()).then(|| pair(&v[17])), sample_rgba: (!v[18].is_null()).then(|| rgba(&v[18])),
    })
}

fn enc_primitive(p: &SelectorPrimitive) -> J {
    match p {
        SelectorPrimitive::Line(l) => json!(["line", l.key, enc_target(&l.target), enc_points(&l.points), enc_rgba(Some(l.color)),
                                             b(l.width_pt), b(l.zorder), l.linestyle]),
        SelectorPrimitive::Markers(m) => json!(["markers", m.key, enc_target(&m.target), enc_points(&m.points), m.shape, b(m.size_pt),
                                                enc_rgba(Some(m.facecolor)), enc_rgba(m.edgecolor), b(m.edge_width_pt), b(m.zorder)]),
        SelectorPrimitive::Text(t) => json!(["text", t.key, enc_target(&t.target), t.text, enc_pair(t.position), t.horizontal_alignment,
                                             enc_rgba(Some(t.color)), t.font_family, b(t.font_size_pt), b(t.zorder)]),
    }
}

#[test]
fn selector_scenes_match() {
    let data = load();
    let mut primitives = 0;
    for c in data["scene"].as_array().unwrap() {
        let s = &c[1];
        let style = SelectorSceneStyle {
            line_width_pt: f(&s[0]), line_alpha: f(&s[1]), handle_size_pt: f(&s[2]), handle_edge_width_pt: f(&s[3]),
            crosshair_size_pt: f(&s[4]), annotation_pt: f(&s[5]), side_distribution_width_pt: f(&s[6]), selector_zorder: f(&s[7]),
            font_family: s[8].as_str().unwrap().to_owned(), label_inset_fraction: f(&s[9]), label_line_fraction: f(&s[10]),
            threshold_width_pt: f(&s[11]), threshold_linestyle: s[12].as_str().unwrap().to_owned(),
        };
        let owner = SelectorSceneOwner { contexts: c[0].as_array().unwrap().iter().map(|v| context(v).unwrap()).collect(), style,
                                         color_context: context(&c[2]) };
        let committed: Vec<SelectorState> = c[3].as_array().unwrap().iter().map(|v| state(v).unwrap()).collect();
        let snapshot = SelectorSnapshot::new(&committed, state(&c[4])).unwrap();
        let color = (!c[5].is_null()).then(|| ColorLimitState { value: range(&c[5][0]).unwrap(), candidate: c[5][1].as_bool().unwrap() });
        let got = outcome(owner.build(&snapshot.states(), color.as_ref()), |scene| {
            primitives += scene.groups.iter().map(|(_, items)| items.len()).sum::<usize>();
            J::Array(scene.groups.iter().map(|(kind, items)| json!([kind.value(), items.iter().map(enc_primitive).collect::<Vec<_>>()])).collect())
        });
        assert_eq!(got, c[6], "scene {c}");
    }
    assert!(primitives > 6_000, "{primitives}");
}

fn defaults(v: &J) -> InteractionDefaults {
    InteractionDefaults { double_click_interval_ms: v[0].as_i64().unwrap(), double_click_radius_px: f(&v[1]),
                          selector_hit_radius_fraction: f(&v[2]), selector_handle_radius_px: f(&v[3]), wheel_zoom_factor: f(&v[4]) }
}
fn opt_pair(v: &J) -> Option<(f64, f64)> { (!v.is_null()).then(|| pair(v)) }
fn enc_viewport(v: Option<Viewport>) -> J { v.map_or(J::Null, |(x, y)| json!([enc_range(x), enc_range(y)])) }
fn viewport(v: &J) -> Option<Viewport> { (!v.is_null()).then(|| (range(&v[0]), range(&v[1]))) }
const UNIT: NumericRange = NumericRange { low: 0.0, high: 1.0 };
const ORIGIN: CrosshairPoint = CrosshairPoint { x: 0.0, y: 0.0 };

fn selector_gesture(t: &AxisTransform, kind: SelectorKind, handle: DragHandle, reading: CrosshairPoint, started: bool) -> SelectorGesture {
    let state = SelectorState { value: SelectorValue::Threshold(0.0), revision: 0, facet_index: None };
    SelectorGesture { axes: AxesId::of(t), transform: t.clone(), kind, state, handle, origin: reading, origin_px: (0.0, 0.0), reading,
                      started, moved: false }
}

fn pan_gesture(t: &AxisTransform, axes: AxesId) -> PanGesture {
    PanGesture { axes, transform: t.clone(), origin: ORIGIN, x: UNIT, y: UNIT, candidate: None }
}

#[test]
fn session_pointer_rules_match() {
    let data = load();
    for c in data["hypot"].as_array().unwrap() {
        assert_eq!(pymath::hypot(f(&c[0]), f(&c[1])).to_bits(), c[2].as_u64().unwrap(), "hypot {c}");
    }
    for c in data["pow"].as_array().unwrap() {
        let got = match pymath::pow(f(&c[0]), f(&c[1])) {
            Ok(v) => ok(b(v)),
            Err(e) if e.message.contains("fractional") => json!({"error": "complex"}),
            Err(e) => err(e),
        };
        assert_eq!(got, c[2], "pow {c}");
    }
    let plain = AxisTransform::linear("main", None, (0.0, 0.0, 1.0, 1.0), (0.0, 1.0), (0.0, 1.0));
    for c in data["event_canonical"].as_array().unwrap() {
        let t = (!c[0].is_null()).then(|| transform(&c[0]));
        let (w, h) = pair(&c[1]);
        let canvas = CanvasSize::new(w, h).unwrap();
        let base = t.clone().unwrap_or_else(|| plain.clone());
        let mut gesture = match &c[3] {
            J::Null => None,
            g if g[0] == "pan" => Some(PointerGesture::Pan(pan_gesture(&base, AxesId::of(&base)))),
            g => Some(PointerGesture::Selector(selector_gesture(&base, kind(&g[1]), handle(&g[2]), point(&g[3]), false))),
        };
        let got = event_canonical(t.as_ref(), opt_pair(&c[2]), canvas, gesture.as_mut(), c[4].as_bool().unwrap());
        assert_eq!(outcome(got, |p| p.map_or(J::Null, enc_point)), c[5], "event canonical {c}");
        let after = match &gesture { Some(PointerGesture::Selector(g)) => enc_point(g.reading), _ => J::Null };
        assert_eq!(after, c[6], "reading after {c}");
    }
    for sequence in data["double_click"].as_array().unwrap() {
        let d = defaults(&sequence[0]);
        let mut history = ClickHistory::new();
        for step in sequence[1].as_array().unwrap() {
            let button = step[1].as_u64().unwrap() as u8;
            let got = history.is_double_click(button, step[2].as_bool().unwrap(), step[3].as_i64(), opt_pair(&step[4]), f(&step[0]), &d);
            assert_eq!(got, step[5].as_bool().unwrap(), "double click {step}");
            let held: serde_json::Map<String, J> = [1u8, 2, 3].iter()
                .filter_map(|button| history.last(*button).map(|(t, x, y)| (button.to_string(), json!([b(t), b(x), b(y)])))).collect();
            assert_eq!(J::Object(held), step[6], "history {step}");
        }
    }
    for c in data["hit_selector"].as_array().unwrap() {
        let t = transform(&c[0]);
        let (w, h) = pair(&c[1]);
        let canvas = CanvasSize::new(w, h).unwrap();
        let d = defaults(&c[2]);
        let states: Vec<SelectorState> = c[4].as_array().unwrap().iter().map(|v| state(v).unwrap()).collect();
        let snapshot = SelectorSnapshot::new(&states, None).unwrap();
        let eligible: Vec<SelectorKind> = c[5].as_array().unwrap().iter().map(kind).collect();
        let pixel = opt_pair(&c[8]);
        let radius = d.selector_handle_radius_px * f(&c[7]);
        let got = hit_selector(point(&c[9]), &snapshot.states(), |s| eligible.contains(&s.kind()), c[3].as_u64(), &t,
                               d.selector_hit_radius_fraction, c[6].as_bool().unwrap(), |s| match s.value {
                                   SelectorValue::Area(value) => session_area_hit(value, pixel, &t, canvas, radius),
                                   _ => Ok(None),
                               });
        assert_eq!(outcome(got, |h| h.map_or(J::Null, |(s, handle)| json!([st(Some(s)), handle.value()]))), c[10], "hit {c}");
    }
    for c in data["color_begin"].as_array().unwrap() {
        assert_eq!(c[4], ok(json!(true)), "colour begin consumed {c}");
        let drag = begin_color_limit_drag(range(&c[0]).unwrap(), range(&c[1]).unwrap(), f(&c[2]), f(&c[3])).unwrap();
        let enc = drag.map_or(J::Null, |d| json!([enc_range(Some(d.original)), enc_range(Some(d.candidate)), d.handle.value(), b(d.origin),
                                                  enc_range(Some(d.bounds)), b(d.minimum_span)]));
        assert_eq!(enc, c[5], "colour begin {c}");
        assert_eq!(drag.map_or(J::Null, |d| enc_range(Some(d.candidate))), c[6], "colour begin preview {c}");
    }
    for c in data["scroll"].as_array().unwrap() {
        let (xs, ys) = (scale(&c[2]), scale(&c[3]));
        let d = defaults(&c[4]);
        let ticks = wheel_ticks(f(&c[8]), c[9] == "up");
        let got = if c[5].as_bool().unwrap() {
            outcome(wheel_camera_zoom(f(&c[10]), d.wheel_zoom_factor, ticks), |zoom| json!([["parameter", "camera_zoom", b(zoom)]]))
        } else {
            let (cx, cy) = viewport(&c[7]).unwrap_or((None, None));
            outcome(wheel_viewport(cx, cy, pair(&c[0]), pair(&c[1]), (&xs, &ys), c[6].as_bool().unwrap(), d.wheel_zoom_factor, ticks,
                                   None),
                    |(x, y)| json!([["viewport", enc_range(Some(x)), enc_range(y)]]))
        };
        let want = if c[11].is_array() { ok(c[11].clone()) } else { c[11].clone() };
        let legacy_collapsed = !c[5].as_bool().unwrap() && want["ok"][0][0] == "viewport"
            && d.wheel_zoom_factor.powf(ticks) < 1.0
            && [1, 2].into_iter().any(|axis| {
                let value = &want["ok"][0][axis];
                value.is_array() && value[0] == value[1] && (axis == 1 || c[6] == true)
            });
        if legacy_collapsed {
            // The v2 oracle lets extreme inward travel collapse both ends
            // to one float. Navigation now stops at a representable range;
            // only those degenerate legacy results change their contract.
            assert_eq!(got["ok"][0][0], "viewport", "scroll {c}");
            let base = viewport(&c[7]).unwrap_or((None, None));
            for (index, scale, committed, drawn) in [(1, &xs, base.0, pair(&c[0])), (2, &ys, base.1, pair(&c[1]))] {
                if index == 2 && c[6] != true {
                    assert_eq!(got["ok"][0][index], want["ok"][0][index], "unchanged y: {c}");
                    continue;
                }
                let actual = range(&got["ok"][0][index]).expect("finite zoom range");
                let expected = range(&want["ok"][0][index]).unwrap();
                let original = committed.unwrap_or(NumericRange { low: drawn.0.min(drawn.1), high: drawn.0.max(drawn.1) });
                let repeated_coordinate = matches!(scale, AxisScale::Samples(values) if expected.low == expected.high
                    && values.windows(2).any(|pair| pair[0] == expected.low && pair[1] == expected.low));
                if original.low == original.high {
                    // Inward travel cannot shrink an already collapsed input;
                    // the outward-only one-ULP recovery is checked below.
                    assert_eq!(actual, expected, "already collapsed base (including its scale round trip): {c}");
                    continue;
                }
                assert!(actual.low.is_finite() && actual.high.is_finite() && (actual.low < actual.high || repeated_coordinate), "scroll {c}");
                let edge_roundoff = 8.0 * f64::EPSILON * original.low.abs().max(original.high.abs());
                assert!(actual.low >= original.low - edge_roundoff && actual.high <= original.high + edge_roundoff,
                    "inward travel must stay inside its base (apart from scale round-trip rounding): {c}");
                let centre = |r: NumericRange| (scale.space(r.low) + scale.space(r.high)) / 2.0;
                let expected_centre = centre(expected);
                if repeated_coordinate {
                    // A repeated coordinate has an interval of sample indices
                    // as its inverse image. The clamp must contain that value
                    // and the original drawn centre; a unique inverse midpoint
                    // (or an epsilon-wide value interval) is not defined here.
                    assert!(actual.low <= expected.low && actual.high >= expected.low, "lost repeated coordinate: {c}");
                    let wanted = centre(original);
                    let (a, b) = if actual.low == actual.high {
                        // Synthetic oracle scales include plateaus (real images
                        // require strictly monotonic coordinates). A zero-width
                        // value range then legitimately represents many indices.
                        assert_eq!(actual, expected);
                        let AxisScale::Samples(values) = scale else { unreachable!() };
                        let position = |index: usize| if values[0] > values[values.len() - 1] { (values.len() - 1 - index) as f64 } else { index as f64 };
                        (position(values.iter().position(|value| *value == actual.low).unwrap()),
                            position(values.iter().rposition(|value| *value == actual.high).unwrap()))
                    } else { (scale.space(actual.low), scale.space(actual.high)) };
                    let tolerance = 8.0 * f64::EPSILON * wanted.abs().max(1.0);
                    assert!(a.min(b) <= wanted + tolerance && a.max(b) >= wanted - tolerance,
                        "precision clamp must contain the original drawn centre: {c}");
                } else {
                    assert!((centre(actual) - expected_centre).abs() <= 8.0 * f64::EPSILON * expected_centre.abs().max(1.0),
                        "precision clamp must preserve the centre: {c}");
                }
            }
        } else {
            let mut wanted = want;
            if !c[5].as_bool().unwrap() && d.wheel_zoom_factor.powf(ticks) > 1.0 && wanted["ok"][0][0] == "viewport" {
                let base = viewport(&c[7]).unwrap_or((None, None));
                for (index, committed, drawn) in [(1, base.0, pair(&c[0])), (2, base.1, pair(&c[1]))] {
                    if index == 2 && c[6] != true { continue; }
                    let original = committed.unwrap_or(NumericRange { low: drawn.0.min(drawn.1), high: drawn.0.max(drawn.1) });
                    // A legacy zero-width input used to be trapped forever,
                    // even scrolling outward. It now advances exactly one ULP
                    // in each direction; every nondegenerate result stays exact.
                    if original.low == original.high && range(&wanted["ok"][0][index]) == Some(original)
                        && original.low.next_down().is_finite() && original.high.next_up().is_finite() {
                        wanted["ok"][0][index] = enc_range(Some(NumericRange { low: original.low.next_down(), high: original.high.next_up() }));
                    }
                }
            }
            assert_eq!(got, wanted, "scroll {c}");
        }
    }
    for c in data["zoom_selection"].as_array().unwrap() {
        let states: Vec<SelectorState> = c[1].as_array().unwrap().iter().map(|v| state(v).unwrap()).collect();
        let got = zoom_to_selection(|k| states.iter().find(|s| s.kind() == k).copied(), c[0].as_u64(), |s| Ok(s.value)).unwrap();
        let enc = match got {
            SelectionZoom::Viewport(x, y) => json!([["viewport", enc_range(Some(x)), enc_range(Some(y))]]),
            SelectionZoom::XLimits(low, high) => json!([["x_limits", b(low), b(high)]]),
            SelectionZoom::Reset => json!([["reset"]]),
        };
        assert_eq!(enc, c[2], "zoom to selection {c}");
    }
    for c in data["pan_update"].as_array().unwrap() {
        let t = transform(&c[0]);
        let (w, h) = pair(&c[1]);
        let canvas = CanvasSize::new(w, h).unwrap();
        let (image, committed, own) = (c[2].as_bool().unwrap(), viewport(&c[3]), viewport(&c[4]));
        let mut gesture = PanGesture { origin: point(&c[5]), x: NumericRange::new(t.x_limits.0, t.x_limits.1).unwrap(),
                                       y: NumericRange::new(t.y_limits.0, t.y_limits.1).unwrap(), ..pan_gesture(&t, AxesId::of(&t)) };
        for step in c[6].as_array().unwrap() {
            let got = update_pan(&mut gesture, opt_pair(&step[0]), canvas, image, own.and_then(|v| v.1), committed, Ok, Ok);
            assert_eq!(outcome(got, |moved| json!(moved)), step[1], "pan update {c}");
            assert_eq!(enc_viewport(gesture.candidate), step[2], "pan candidate {c}");
        }
    }
    for c in data["degenerate"].as_array().unwrap() {
        let s = state(&c[0]).unwrap();
        assert_eq!(is_degenerate_selector(&s), c[1].as_bool().unwrap(), "degenerate {c}");
        assert_eq!(require_stable_selector(&s).err().map(|e| e.message), c[2].as_str().map(str::to_owned), "stable {c}");
    }
    for c in data["bounds"].as_array().unwrap() {
        let t = transform(&c[0]);
        let gesture = (!c[1].is_null()).then(|| PointerGesture::Selector(selector_gesture(&t, kind(&c[1]), DragHandle::Body, ORIGIN, false)));
        let (x, y) = selector_bounds(&t, threshold_gesture_on_histogram(gesture.as_ref(), c[2].as_bool().unwrap()));
        assert_eq!(json!([enc_range(drawn_bounds(t.canonical_x_limits)), enc_range(x), enc_range(y)]), json!([c[3], c[4], c[5]]), "bounds {c}");
    }
    for c in data["pointer_latch"].as_array().unwrap() {
        let gid = match c[2].as_u64() { None => c[1].as_str().unwrap().to_owned(), Some(cell) => format!("{}:{cell}", c[1].as_str().unwrap()) };
        let axes = AxesId::from_gid(&gid);
        let camera = CameraAngles { azimuth_deg: 55.0, elevation_deg: 30.0, zoom: 1.0 };
        let gesture = match &c[0] {
            J::Null => None,
            g => Some(match g[0].as_str().unwrap() {
                "selector" => PointerGesture::Selector(SelectorGesture { axes: axes.clone(),
                    ..selector_gesture(&plain, SelectorKind::Area, DragHandle::New, ORIGIN, g[1].as_bool().unwrap()) }),
                "color" => PointerGesture::Color(ColorGesture { axes: axes.clone(), transform: plain.clone(), drag: ColorLimitDrag {
                    original: UNIT, candidate: UNIT, handle: DragHandle::Low, origin: 0.0, bounds: UNIT, minimum_span: 1e-12 } }),
                "pan" => PointerGesture::Pan(pan_gesture(&plain, axes.clone())),
                "orbit" => PointerGesture::Orbit(OrbitGesture { axes: axes.clone(), transform: plain.clone(), origin_px: (0.0, 0.0), start: camera, current: camera }),
                _ => PointerGesture::Pick(PickGesture { axes: axes.clone(), transform: plain.clone(), origin_px: (0.0, 0.0) }),
            }),
        };
        let has_candidate = c[3].as_bool().unwrap() || matches!(gesture, Some(PointerGesture::Color(_)));
        let (latched, active_pan, armed) = pointer_latch(gesture.as_ref(), has_candidate);
        let got = json!([has_candidate, latched.as_ref().map(|a| a.role.clone()), latched.and_then(|a| a.cell_index), active_pan, armed]);
        assert_eq!(got, c[4], "latch {c}");
    }
}

/// A one-panel session composed from the kernels exactly as the mixin
/// composes them (`_on_button_press`, `_on_motion`, `_on_button_release`,
/// `_cancel_gesture`, `_clear_gesture`), recording the host calls the
/// generator's stand-in session records.
struct Pipeline {
    main: AxisTransform,
    rail: Option<AxisTransform>,
    canvas: CanvasSize,
    defaults: InteractionDefaults,
    histogram: bool,
    image: bool,
    dpr: f64,
    domain: NumericRange,
    color_value: NumericRange,
    controller: SelectorController,
    gesture: Option<PointerGesture>,
    clicks: ClickHistory,
    calls: Vec<J>,
}

type R<T> = Result<T, PlotError>;

impl Pipeline {
    fn main_axes(&self) -> AxesId { AxesId::of(&self.main) }
    fn render(&mut self, effects: &[&str]) { self.calls.push(json!(["render", effects])); }
    fn emit(&mut self, change: &str, state: SelectorState) { self.calls.push(json!(["emit", change, st(Some(state))])); }
    fn committed(&self, kind: SelectorKind) -> Option<SelectorState> { self.controller.states().into_iter().find(|s| s.kind() == kind) }

    fn clear_gesture(&mut self) {
        match self.gesture.take() {
            None => {}
            Some(PointerGesture::Pan(_)) => self.calls.push(json!(["view_dragging", false])),
            Some(PointerGesture::Selector(g)) if !g.started => {}
            Some(_) => self.calls.push(json!(["end_selector"])),
        }
    }

    fn cancel_gesture(&mut self) -> R<Option<SelectorState>> {
        let Some(gesture) = self.gesture.clone() else { return Ok(None) };
        let cancelled = match &gesture { PointerGesture::Selector(g) => cancel_selector_gesture(&mut self.controller, g), _ => Ok(None) };
        self.clear_gesture();
        let cancelled = cancelled?;
        if matches!(&gesture, PointerGesture::Pan(g) if g.candidate.is_some()) { self.render(&["AXIS_TRANSFORM", "OVERLAY"]); }
        let color = matches!(gesture, PointerGesture::Color(_));
        if color || cancelled.is_some() { self.render(if color { &["BASE_GEOMETRY"] } else { &["OVERLAY"] }); }
        Ok(cancelled)
    }

    fn install_selector_state(&mut self, state: SelectorState, emit: bool, finished: bool) -> R<SelectorState> {
        require_stable_selector(&state)?;
        if !finished && matches!(self.gesture, Some(PointerGesture::Selector(_))) { self.cancel_gesture()?; }
        let (previous, stored) = if finished { self.controller.commit_finished(state)? } else { self.controller.install(state)? };
        self.render(&["OVERLAY"]);
        if emit { self.emit(if previous.is_none() { "added" } else { "updated" }, stored.state()); }
        Ok(stored.state())
    }

    fn remove_selector(&mut self, kind: SelectorKind) -> R<()> {
        let state = self.controller.state(kind)?;
        if matches!(&self.gesture, Some(PointerGesture::Selector(g)) if g.kind == state.kind()) { self.cancel_gesture()?; }
        self.controller.remove(kind)?;
        self.render(&["OVERLAY"]);
        self.emit("removed", state);
        Ok(())
    }

    fn press(&mut self, button: u8, pixel: (f64, f64), double: bool, on_rail: bool, now: f64) -> R<()> {
        let transform = if on_rail { self.rail.clone().unwrap() } else { self.main.clone() };
        let is_double = self.clicks.is_double_click(button, double, Some(0), Some(pixel), now, &self.defaults);
        self.cancel_gesture()?;
        if button == 1 && is_double { return Ok(()); }
        if button == 2 {
            if on_rail { return Ok(()); }
            if is_double {
                let zoom = zoom_to_selection(|k| self.committed(k), None, |s| Ok(s.value))?;
                self.calls.push(match zoom {
                    SelectionZoom::Viewport(x, y) => json!(["viewport", enc_range(Some(x)), enc_range(Some(y))]),
                    SelectionZoom::XLimits(low, high) => json!(["x_limits", b(low), b(high)]),
                    SelectionZoom::Reset => json!(["reset"]),
                });
                return Ok(());
            }
            let Some(origin) = transform.display_point(Some(pixel), self.canvas)? else { return Ok(()) };
            let (x, y) = (NumericRange::new(transform.x_limits.0, transform.x_limits.1)?, NumericRange::new(transform.y_limits.0, transform.y_limits.1)?);
            self.gesture = Some(PointerGesture::Pan(PanGesture { axes: self.main_axes(), transform, origin, x, y, candidate: None }));
            self.calls.push(json!(["view_dragging", true]));
            return Ok(());
        }
        let Some(point) = event_canonical(Some(&transform), Some(pixel), self.canvas, self.gesture.as_mut(), self.histogram)? else { return Ok(()) };
        if button == 3 {
            if on_rail { return Ok(()); }
            if is_double {
                if self.committed(SelectorKind::Crosshair).is_some() { self.remove_selector(SelectorKind::Crosshair)?; }
                return Ok(());
            }
            self.install_selector_state(SelectorState::new(SelectorValue::Crosshair(point), 0, None)?, true, false)?;
            return Ok(());
        }
        if on_rail {
            if let Some(drag) = begin_color_limit_drag(self.domain, self.color_value, point.x, self.defaults.selector_hit_radius_fraction)? {
                self.gesture = Some(PointerGesture::Color(ColorGesture { axes: AxesId::from_gid("distribution"), transform, drag }));
                self.calls.push(json!(["begin_color", enc_range(Some(drag.candidate))]));
            }
            return Ok(());
        }
        let radius = self.defaults.selector_handle_radius_px * self.dpr;
        let states = self.controller.snapshot(true)?.states();
        let (canvas, histogram) = (self.canvas, self.histogram);
        let hit = hit_selector(point, &states, |_| true, None, &transform, self.defaults.selector_hit_radius_fraction, histogram, |s| match s.value {
            SelectorValue::Area(value) => session_area_hit(value, Some(pixel), &transform, canvas, radius),
            _ => Ok(None),
        })?;
        let axes = self.main_axes();
        self.gesture = selector_press(point, pixel, hit, true, None, &transform, |_| Some(axes))?.map(PointerGesture::Selector);
        Ok(())
    }

    fn motion(&mut self, pixel: (f64, f64)) -> R<()> {
        let (canvas, histogram, image) = (self.canvas, self.histogram, self.image);
        match self.gesture.as_mut() {
            Some(PointerGesture::Pan(g)) => {
                if update_pan(g, Some(pixel), canvas, image, None, None, Ok, Ok)? { self.render(&["AXIS_TRANSFORM", "OVERLAY"]); }
            }
            Some(PointerGesture::Selector(g)) => {
                let kind = g.kind;
                let motion = selector_motion(&mut self.controller, g, pixel, canvas, histogram);
                if motion.started { self.calls.push(json!(["begin_selector", kind.value()])); }
                match motion.step? {
                    SelectorStep::Cancel => { self.cancel_gesture()?; }
                    SelectorStep::Preview(updated) => {
                        if kind == SelectorKind::Threshold { self.calls.push(json!(["classifier_labels"])); }
                        self.calls.push(json!(["preview", st(Some(updated))]));
                        self.emit("updated", updated);
                    }
                    _ => {}
                }
            }
            Some(PointerGesture::Color(g)) => {
                if color_motion(g, pixel, canvas)? { let candidate = g.drag.candidate; self.calls.push(json!(["preview_color", enc_range(Some(candidate))])); }
            }
            _ => {}
        }
        Ok(())
    }

    fn release(&mut self, button: u8, pixel: (f64, f64)) -> R<()> {
        let (canvas, histogram, image) = (self.canvas, self.histogram, self.image);
        let Some(gesture) = self.gesture.as_mut() else { return Ok(()) };
        if button == 2 {
            let PointerGesture::Pan(g) = gesture else { return Ok(()) };
            let moved = update_pan(g, Some(pixel), canvas, image, None, None, Ok, Ok);
            let candidate = g.candidate;
            self.clear_gesture();
            moved?;
            if let Some(candidate) = candidate { self.calls.push(json!(["viewport_state", enc_viewport(Some(candidate))])); }
            return Ok(());
        }
        match gesture {
            PointerGesture::Pan(_) => Ok(()),
            PointerGesture::Color(g) => match color_release(g, pixel, canvas)? {
                ColorRelease::Cancel => self.cancel_gesture().map(|_| ()),
                ColorRelease::Changed(value) => { self.clear_gesture(); self.calls.push(json!(["color_limits", b(value.low), b(value.high), true])); Ok(()) }
                ColorRelease::Unchanged => { self.clear_gesture(); self.render(&["BASE_GEOMETRY"]); Ok(()) }
            },
            PointerGesture::Selector(g) => {
                let kind = g.kind;
                match selector_release(&mut self.controller, g, pixel, canvas, histogram) {
                    Err(e) => { self.cancel_gesture()?; Err(e) }
                    Ok(SelectorRelease::BlankClick) => {
                        self.clear_gesture();
                        if let Some(committed) = self.committed(kind) { self.remove_selector(committed.kind())?; }
                        Ok(())
                    }
                    Ok(SelectorRelease::Abandoned | SelectorRelease::Nothing) => { self.clear_gesture(); Ok(()) }
                    Ok(SelectorRelease::Cancel) => self.cancel_gesture().map(|_| ()),
                    Ok(SelectorRelease::Commit(candidate)) => {
                        self.clear_gesture();
                        let stored = self.install_selector_state(candidate, false, true)?;
                        self.emit("committed", stored);
                        Ok(())
                    }
                    Ok(SelectorRelease::Degenerate(candidate)) => {
                        self.clear_gesture();
                        match self.committed(candidate.kind()) { Some(committed) => self.remove_selector(committed.kind())?, None => self.render(&["OVERLAY"]) }
                        Ok(())
                    }
                }
            }
            _ => Ok(()),
        }
    }

    fn summary(&self) -> J {
        match &self.gesture {
            None => J::Null,
            Some(PointerGesture::Selector(g)) => json!(["selector", g.kind.value(), g.handle.value(), enc_point(g.origin), enc_pair(g.origin_px),
                                                        enc_point(g.reading), g.started, g.moved, st(Some(g.state))]),
            Some(PointerGesture::Color(g)) => {
                let d = g.drag;
                json!(["color", enc_range(Some(d.original)), enc_range(Some(d.candidate)), d.handle.value(), b(d.origin), enc_range(Some(d.bounds)), b(d.minimum_span)])
            }
            Some(PointerGesture::Pan(g)) => json!(["pan", enc_point(g.origin), enc_range(Some(g.x)), enc_range(Some(g.y)), enc_viewport(g.candidate)]),
            Some(_) => panic!("no orbit or pick on this panel"),
        }
    }
}

#[test]
fn gesture_pipeline_matches() {
    let data = load();
    let mut events = 0;
    for (number, c) in data["pipeline"].as_array().unwrap().iter().enumerate() {
        let (w, h) = pair(&c[2]);
        let mut p = Pipeline {
            main: transform(&c[0]), rail: (!c[1].is_null()).then(|| transform(&c[1])), canvas: CanvasSize::new(w, h).unwrap(),
            defaults: defaults(&c[3]), histogram: c[4].as_bool().unwrap(), image: c[5].as_bool().unwrap(), dpr: f(&c[6]),
            domain: range(&c[7]).unwrap(), color_value: range(&c[8]).unwrap(), controller: SelectorController::new(), gesture: None,
            clicks: ClickHistory::new(), calls: Vec::new(),
        };
        for s in c[9].as_array().unwrap() { p.controller.install(state(s).unwrap()).unwrap(); }
        for step in c[10].as_array().unwrap() {
            p.calls.clear();
            let now = f(&step[5]);
            let pixel = opt_pair(&step[2]);
            let got = match step[0].as_str().unwrap() {
                "press" => outcome(p.press(step[1].as_u64().unwrap() as u8, pixel.unwrap(), step[3].as_bool().unwrap(), step[4].as_bool().unwrap(), now), |_| J::Null),
                "move" => outcome(p.motion(pixel.unwrap()), |_| J::Null),
                "release" => outcome(p.release(step[1].as_u64().unwrap() as u8, pixel.unwrap()), |_| J::Null),
                _ => outcome(p.cancel_gesture(), st),
            };
            let event: Vec<String> = step.as_array().unwrap()[..5].iter().map(|v| v.to_string()).collect();
            assert_eq!(got, step[6], "sequence {number} result of {}", event.join(" "));
            assert_eq!(J::Array(p.calls.clone()), step[7], "sequence {number} calls of {}", step[0]);
            assert_eq!(dump(&p.controller), step[8], "sequence {number} controller after {}", step[0]);
            assert_eq!(p.summary(), step[9], "sequence {number} gesture after {}", step[0]);
            let held: serde_json::Map<String, J> = [1u8, 2, 3].iter()
                .filter_map(|button| p.clicks.last(*button).map(|(t, x, y)| (button.to_string(), json!([b(t), b(x), b(y)])))).collect();
            assert_eq!(J::Object(held), step[10], "sequence {number} clicks after {}", step[0]);
            events += 1;
        }
    }
    assert!(events > 5_000, "{events}");
}

#[test]
fn scene_gestures_match() {
    let data = load();
    let plain = AxisTransform::linear("main", None, (0.0, 0.0, 1.0, 1.0), (0.0, 1.0), (0.0, 1.0));
    for c in data["orbit"].as_array().unwrap() {
        let start = CameraAngles { azimuth_deg: f(&c[0][0]), elevation_deg: f(&c[0][1]), zoom: f(&c[0][2]) };
        let mut gesture = OrbitGesture { axes: AxesId::of(&plain), transform: plain.clone(), origin_px: pair(&c[1]), start, current: start };
        for step in c[2].as_array().unwrap() {
            let camera = gesture.follow(pair(&step[0])).unwrap();
            assert_eq!(json!([b(camera.azimuth_deg), b(camera.elevation_deg)]), step[1], "orbit parameters {c}");
            let current = gesture.current;
            assert_eq!(json!([b(current.azimuth_deg), b(current.elevation_deg), b(current.zoom)]), step[2], "orbit camera {c}");
        }
    }
    for c in data["pick"].as_array().unwrap() {
        let defaults = InteractionDefaults { double_click_radius_px: f(&c[2]), ..InteractionDefaults::default() };
        assert_eq!(is_pick_click(pair(&c[0]), pair(&c[1]), &defaults), c[3].as_bool().unwrap(), "pick {c}");
    }
}
