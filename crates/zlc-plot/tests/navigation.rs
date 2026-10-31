//! Continuous image navigation through the real raster-pointer session path.
//! Viewports are independent of source pixel quantization; samples and ROI
//! semantics stay with the projection layer.

use std::sync::Arc;

use serde_json::json;
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::OwnedBuffer;
use zlc_plot::configuration::resolve::{resolve_configuration, ConfigurationUpdates, Defaults, ResolveRequest};
use zlc_plot::configuration::{default_spec, parameter_schema_for, CrosshairPoint, NumericRange, PlotKind, PlotSchema, SelectorState,
                              SelectorValue};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::{begin_color_limit_drag, hit_selector, InteractionDefaults};
use zlc_plot::interaction::transform::AxisTransform;
use zlc_plot::projection::{SchemaCache, Source};
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{DataPreparation, EngineFitLayer, PlotInput, PlotSession, PointerAction, PointerEvent, SessionServices};

fn services() -> SessionServices {
    SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
                      defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() }
}

fn axis(id: &str, role: &str, coordinates: Coordinates, unit: Option<&str>, frame: Option<&str>) -> Arc<Axis> {
    Axis::new(id.into(), role.into(), role.into(), coordinates, unit.map(Into::into), frame.map(Into::into), None, None).unwrap()
}

fn one(id: &str, role: &str) -> Arc<Domain> { Domain::dense(vec![1], vec![axis(id, role, Coordinates::Index { origin: 0, len: 1 }, None, None)]).unwrap() }

/// One event of `cell` (values row-major) committed to a Runtime and read
/// back as the desktop host reads a camera.
fn source(cell: Arc<Domain>, values: Vec<f64>) -> Source {
    let schema = DatasetSchema::new(one("probe.repeat", "repeat"), one("probe.frame", "readout-event"), cell,
                                    ValueSchema::new(zlc_core::DType::F64, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap();
    let runtime = Runtime::new();
    let generation = runtime.begin("probe", vec![Declaration { signal: "frames".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
    let data = DataBlock::new(schema, OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), None).unwrap();
    runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "frames".into(), ordinal: 0, data, origin: None }],
                                         metadata: json!({}), parents: vec![] }).unwrap();
    let read = runtime.read_coherent_views(&[(generation, Arc::from("frames"), None)], &mut Default::default()).pop().unwrap().unwrap();
    Source::from_read(&read).unwrap()
}

/// A camera frame of `rows` x `columns` pixels: on pixel indices, or on the
/// coordinates given (as they are, no unit).
fn frame(rows: usize, columns: usize, coordinates: Option<(Vec<f64>, Vec<f64>)>) -> Source {
    let (y, x, unit) = match coordinates {
        Some((y, x)) => (y, x, None),
        None => ((0..rows).map(|i| i as f64).collect(), (0..columns).map(|i| i as f64).collect(), Some("pixel")),
    };
    let pixels = vec![axis("camera.spatial-y", "spatial-y", Coordinates::Numeric(OwnedBuffer::from_vec(y).into()), unit, Some("sensor_pixel_xy")),
                      axis("camera.spatial-x", "spatial-x", Coordinates::Numeric(OwnedBuffer::from_vec(x).into()), unit, Some("sensor_pixel_xy"))];
    let values = (0..rows * columns).map(|i| ((i * 7919) % 251) as f64).collect();
    source(Domain::dense(vec![rows, columns], pixels).unwrap(), values)
}

fn session(source: Source, kind: PlotKind) -> PlotSession {
    let services = services();
    let schema = PlotSchema::new(source.schema.clone());
    let spec = default_spec(Some(&schema), kind).unwrap().expect("a default spec");
    let parameters = parameter_schema_for(&spec, &services.defaults.style).initial_values(None).unwrap();
    let candidate = resolve_configuration(ResolveRequest {
        schema: Some(schema), spec: Some(spec), defaults: Some(services.defaults.clone()), fit: Some(services.fit.clone()),
        configuration_updates: Some(ConfigurationUpdates { parameters: Some(parameters), ..Default::default() }), ..Default::default()
    }).unwrap();
    let input = PlotInput::Dataset(Arc::new(source));
    let prepared = DataPreparation::new(candidate.clone(), None)
        .prepare(&input, None, &mut SchemaCache::default(), &services.settings(), &services.fit.engine).unwrap();
    let mut session = PlotSession::from_prepared(prepared, candidate, 1.0, "panel-1", services).unwrap();
    session.finish_presentation(true, None);
    session
}

/// The plot's own axes as the frontend last saw them.
fn plot_axes(session: &PlotSession) -> AxisTransform {
    session.raster_axes_snapshot().unwrap().into_iter().find(|axes| axes.role == "image" || axes.role == "main").expect("the plot's axes")
}

/// One pointer message at `(fx, fy)` of the plot's axes box (0..1, top
/// origin), a second after the last: presses are never a timed double click.
fn pointer(session: &mut PlotSession, action: PointerAction, at: (f64, f64), button: Option<u8>, step: f64) {
    static CLOCK: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    let time = CLOCK.fetch_add(1, std::sync::atomic::Ordering::Relaxed) as f64;
    let axes = plot_axes(session);
    let (left, top, right, bottom) = axes.bounds;
    let event = PointerEvent { action, x: left + at.0 * (right - left), y: top + at.1 * (bottom - top), button, double: false, step, key: None,
                               axes: Some(axes), time: Some(time) };
    session.raster_pointer_event(&event).unwrap();
}

fn wheel(session: &mut PlotSession, step: f64) { pointer(session, PointerAction::Scroll, (0.5, 0.5), None, step); }

/// The view on screen: the committed viewport, else the drawn limits.
fn view(session: &PlotSession) -> (NumericRange, NumericRange) {
    let axes = plot_axes(session);
    let sorted = |(a, b): (f64, f64)| NumericRange::new(a.min(b), a.max(b)).unwrap();
    match session.viewport() {
        Some((Some(x), Some(y))) => (x, y),
        _ => (sorted(axes.x_limits), sorted(axes.y_limits)),
    }
}

/// The limits drawn now (a pan's candidate while it is carried).
fn drawn(session: &PlotSession) -> (NumericRange, NumericRange) {
    let axes = plot_axes(session);
    let sorted = |(a, b): (f64, f64)| NumericRange::new(a.min(b), a.max(b)).unwrap();
    (sorted(axes.x_limits), sorted(axes.y_limits))
}

fn centre(range: NumericRange) -> f64 { (range.low + range.high) / 2.0 }

fn close(a: f64, b: f64) { assert!((a - b).abs() <= 1e-10 * a.abs().max(b.abs()).max(1.0), "{a} != {b}"); }
fn same_view(a: (NumericRange, NumericRange), b: (NumericRange, NumericRange)) {
    for (a, b) in [(a.0.low, b.0.low), (a.0.high, b.0.high), (a.1.low, b.1.low), (a.1.high, b.1.high)] { close(a, b); }
}

/// Subpixel movement is real view state, not an invisible wheel remainder.
/// Each touchpad event changes the view, even when only a few cells remain.
#[test]
fn small_wheel_steps_are_continuous_below_one_source_pixel() {
    let mut plot = session(frame(45, 61, None), PlotKind::Image);
    let range = NumericRange::new(29.0, 31.0).unwrap();
    plot.set_x_limits(range.low, range.high).unwrap();
    let start = view(&plot);
    for event in 1..=330 {
        let before = view(&plot);
        wheel(&mut plot, -0.08);
        let after = view(&plot);
        assert!(after.0.span() < before.0.span(), "event {event} stalled");
        close(after.0.span(), 2.0 * 1.1f64.powf(-0.08 * event as f64));
        close(centre(after.0), centre(start.0));
        close(centre(after.1), centre(start.1));
        same_view(after, drawn(&plot));
    }
    assert!(view(&plot).0.span() < 0.2, "a source pixel is not a navigation limit");
    let before = view(&plot);
    wheel(&mut plot, 0.08);
    assert!(view(&plot).0.span() > before.0.span(), "first reverse event must move");
    for _ in 1..330 { wheel(&mut plot, 0.08); }
    same_view(view(&plot), start);
}

/// Even camera dimensions put the view centre on a pixel edge. That edge
/// must not stop zoom at two cells, or cause a visible stair-step at 20 cells.
#[test]
fn camera_zoom_keeps_advancing_through_cell_edges() {
    for (rows, columns) in [(372, 460), (45, 61)] {
        let mut plot = session(frame(rows, columns, None), PlotKind::Image);
        let home = view(&plot);
        for notch in 1..=100 {
            let before = view(&plot);
            wheel(&mut plot, -1.0);
            assert!(view(&plot).0.span() < before.0.span(), "notch {notch}");
            close(view(&plot).0.span(), home.0.span() * 1.1f64.powi(-notch));
        }
        assert!(view(&plot).0.span() < 0.04);
        if rows % 2 == 1 && columns % 2 == 1 {
            // At this magnification the image axes lie inside the central
            // source pixel. Check actual composed pixels, not only limits:
            // fractional cropping must neither go blank nor sample a neighbour.
            let value = (((rows / 2 * columns + columns / 2) * 7919) % 251) as f64;
            let (low, high) = plot.renderer().resolved_color_limits().unwrap();
            let code = (((value - low) / (high - low) * 256.0) as usize).min(255);
            let cmap = plot.display_state().get("colormap").unwrap().as_str().unwrap().to_owned();
            let expected = &zlc_plot::render::image::color_lut(&cmap).unwrap()[code * 4..code * 4 + 4];
            let axes = plot_axes(&plot);
            let front = plot.rgba().unwrap();
            let (left, top, right, bottom) = axes.bounds;
            for (fx, fy) in [(0.23, 0.37), (0.71, 0.29), (0.67, 0.79)] {
                let x = ((left + fx * (right - left)) * f64::from(front.width)) as usize;
                let y = ((top + fy * (bottom - top)) * f64::from(front.height)) as usize;
                let offset = (y * front.width as usize + x) * 4;
                assert_eq!(&front.pixels.as_slice()[offset..offset + 4], expected, "wrong image pixel at ({fx}, {fy})");
            }
        }
        for _ in 0..100 { wheel(&mut plot, 1.0); }
        let restored = view(&plot);
        // Endpoint subtraction amplifies accumulated rounding after 100
        // contractions/expansions. Normalize this round-trip error to the
        // original view width, not the near-zero lower endpoint itself.
        for (actual, expected) in [(restored.0, home.0), (restored.1, home.1)] {
            for (actual, wanted) in [(actual.low, expected.low), (actual.high, expected.high)] {
                assert!((actual - wanted).abs() <= 1e-11 * expected.span(), "round trip: {actual} != {wanted}");
            }
        }
    }
}

/// Batching a wheel burst changes neither the destination nor its centre.
#[test]
fn many_small_wheel_steps_are_one_big_step() {
    for (step, count) in [(-0.04, 25), (-0.08, 250), (-0.25, 40), (-1.0, 30), (0.1, 30), (1.0, 3)] {
        let mut many = session(frame(372, 460, None), PlotKind::Image);
        for _ in 0..count { wheel(&mut many, step); }
        let mut one = session(frame(372, 460, None), PlotKind::Image);
        wheel(&mut one, step * count as f64);
        same_view(view(&many), view(&one));
    }
}

/// The live-data presentation and saved-configuration paths keep fractional
/// navigation; they must not restart zoom from a rounded or earlier frame.
#[test]
fn continuous_zoom_survives_new_frames_and_session_reconstruction() {
    let input = PlotInput::Dataset(Arc::new(frame(45, 61, None)));
    let mut plot = session(frame(45, 61, None), PlotKind::Image);
    let range = NumericRange::new(29.25, 30.75).unwrap();
    plot.set_x_limits(range.low, range.high).unwrap();
    for revision in 1..=12 {
        wheel(&mut plot, -0.08);
        let shown = view(&plot);
        let configuration = plot.resolve_configuration(None, None, None, None, None, true, None, None, None, None).unwrap();
        let prepared = plot.prepare_data(&input, Some(revision), Some(configuration.clone())).unwrap();
        plot.present_prepared(prepared.clone(), Some(configuration.clone()), true).unwrap();
        plot.finish_presentation(true, None);
        assert_eq!(view(&plot), shown, "live frame {revision} changed navigation");
        let restored = PlotSession::from_prepared(prepared, configuration, 1.0, "restored", services()).unwrap();
        assert_eq!(view(&restored), shown, "restored frame {revision} changed navigation");
        close(shown.0.span(), 1.5 * 1.1f64.powf(-0.08 * revision as f64));
    }
}

/// The desktop document may draw gestures on the old frame while new data
/// is pending, then replay them when data arrives. Continuous wheel travel
/// must appear once, including fractional steps and a direction reversal.
#[test]
fn document_waiting_wheel_replay_preserves_the_continuous_view() {
    use std::collections::{BTreeMap, BTreeSet};
    use zlc_plot::document::{axis_wire, DocumentRead, FigureDocument};
    #[derive(Clone)]
    struct CameraRead(Arc<Source>);
    impl DocumentRead for CameraRead {
        fn source(&self) -> &Arc<Source> { &self.0 }
        fn provenance_json(&self) -> String { "null".into() }
    }
    let mut document = FigureDocument::<CameraRead>::new(services(), None);
    document.execute(r#"{"action":"ready","id":1,"dpr":1.0,"targets":{},"target_requests":{}}"#,
        BTreeMap::new(), None, None).unwrap();
    document.finish_presentation(&BTreeSet::new(), &BTreeSet::new());
    let panels = BTreeSet::from(["p".to_owned()]);
    let reads = BTreeMap::from([("p".to_owned(), CameraRead(Arc::new(frame(45, 61, None))))]);
    let mut binding = json!([{"panel":"p","signal":"frames","kind":"image","window":1,"epoch":1,"waiting":false,"reason":null}]);
    let result = document.execute(&json!({"action":"sync_panels","id":2,"reset_document":true,
        "inputs":binding,"targets":{},"target_requests":{"p":2}}).to_string(), reads.clone(), None, None).unwrap();
    document.finish_presentation(&panels, &panels);
    let mut target = result.state["panels"][0]["target"].clone();
    let result = document.execute(&json!({"action":"configure","id":3,"panel":"p","inputs":binding,
        "viewport":{"x":[29.25,30.75],"y":[21.25,22.75]},"targets":{"p":target},"target_requests":{"p":3}}).to_string(),
        reads.clone(), None, None).unwrap();
    document.finish_presentation(&panels, &panels);
    target = result.state["panels"][0]["target"].clone();
    binding[0]["waiting"] = json!(true);
    binding[0]["reason"] = json!("Waiting for data");
    let mut travel = 0.0;
    for (index, steps) in [-0.08, -0.04, 0.013].into_iter().enumerate() {
        let axes = plot_axes(&document.session("p").unwrap().lock().unwrap());
        let (left, top, right, bottom) = axes.bounds;
        let result = document.execute(&json!({"action":"pointer","event":"scroll","id":4+index,"panel":"p",
            "steps":steps,"axes":axis_wire(&axes),"x":(left+right)/2.0,"y":(top+bottom)/2.0,
            "inputs":binding,"targets":{"p":target},"target_requests":{"p":4+index}}).to_string(),
            BTreeMap::new(), None, None).unwrap();
        assert!(result.state["error"].is_null());
        let fronts = result.fronts.keys().cloned().collect();
        document.finish_presentation(&panels, &fronts);
        target = result.state["panels"][0]["target"].clone();
        travel += steps;
        let shown = view(&document.session("p").unwrap().lock().unwrap());
        close(shown.0.span(), 1.5 * 1.1f64.powf(travel));
        close(shown.1.span(), shown.0.span());
    }
    let waiting = view(&document.session("p").unwrap().lock().unwrap());
    binding[0]["waiting"] = json!(false);
    binding[0]["reason"] = json!(null);
    let result = document.execute(&json!({"action":"sync_panels","id":0,"inputs":binding,
        "targets":{"p":target},"target_requests":{"p":6}}).to_string(), reads, None, None).unwrap();
    assert!(result.state["error"].is_null());
    let fronts = result.fronts.keys().cloned().collect();
    document.finish_presentation(&panels, &fronts);
    same_view(view(&document.session("p").unwrap().lock().unwrap()), waiting);
}

/// A finite precision limit must not accumulate hidden inward travel or
/// bounce outward when the axis renderer expands a numerically singular range.
#[test]
fn numerical_zoom_limit_is_finite_and_reverses_immediately() {
    use zlc_plot::interaction::gesture::wheel_viewport;
    use zlc_plot::render::scale::AxisScale;
    let mut x = NumericRange::new(1e6 - 1.0, 1e6 + 1.0).unwrap();
    let mut y = NumericRange::new(2e6 - 1.0, 2e6 + 1.0).unwrap();
    for _ in 0..1000 {
        let before = (x, y);
        (x, y) = {
            let (x, y) = wheel_viewport(Some(x), Some(y), (x.low, x.high), (y.low, y.high),
                (&AxisScale::Linear, &AxisScale::Linear), true, 1.1, -1.0, None).unwrap();
            (x, y.unwrap())
        };
        assert!(x.span() > 0.0 && y.span() > 0.0);
        assert!(x.span() <= before.0.span() && y.span() <= before.1.span());
    }
    for step in [1.0, 0.08, 0.04, 0.013] {
        let (out, _) = wheel_viewport(Some(x), Some(y), (x.low, x.high), (y.low, y.high),
            (&AxisScale::Linear, &AxisScale::Linear), true, 1.1, step, None).unwrap();
        assert!(out.span() > x.span(), "reverse step {step} stalled");
    }
    let initial = NumericRange::new(-0.5, 0.5).unwrap();
    let zoom = |range: NumericRange, steps| wheel_viewport(Some(range), None, (range.low, range.high), (0.0, 1.0),
        (&AxisScale::Linear, &AxisScale::Linear), false, 1.1, steps, None).unwrap().0;
    let one = zoom(initial, -1000.0);
    let mut many = initial;
    for _ in 0..1000 { many = zoom(many, -1.0); }
    assert!(one.span() > 0.0 && one.span() < 1e-40);
    assert!((one.span() - many.span()).abs() <= one.span() * 1e-12, "centred-at-zero bursts must compound identically");
}

#[test]
fn log_zoom_near_one_keeps_distinct_values_at_the_precision_limit() {
    use zlc_plot::interaction::gesture::wheel_viewport;
    use zlc_plot::render::scale::AxisScale;
    let mut range = NumericRange::new(0.5, 2.0).unwrap();
    for _ in 0..1000 {
        let before = range;
        range = wheel_viewport(Some(range), None, (range.low, range.high), (0.0, 1.0),
            (&AxisScale::Log, &AxisScale::Linear), false, 1.1, -1.0, None).unwrap().0;
        assert!(range.low > 0.0 && range.span() > 1e-15);
        assert!(range.span() <= before.span());
    }
    let out = wheel_viewport(Some(range), None, (range.low, range.high), (0.0, 1.0),
        (&AxisScale::Log, &AxisScale::Linear), false, 1.1, 1.0, None).unwrap().0;
    assert!(out.span() > range.span());
}

#[test]
fn zoom_uses_finite_midpoints_for_large_valid_ranges() {
    use zlc_plot::interaction::gesture::{wheel_zoom, wheel_viewport};
    use zlc_plot::render::scale::AxisScale;
    for (range, factor) in [(NumericRange::new(1e308, 1.2e308).unwrap(), 0.5),
        (NumericRange::new(-1e308, 1e308).unwrap(), 0.5),
        (NumericRange::new(-5e307, 5e307).unwrap(), 3.0)] {
        let zoomed = wheel_zoom(range, &AxisScale::Linear, factor).unwrap();
        assert!(zoomed.low.is_finite() && zoomed.high.is_finite() && zoomed.low < zoomed.high);
    }
    let range = NumericRange::new(-1e308, 1e308).unwrap();
    let zoomed = wheel_viewport(Some(range), None, (range.low, range.high), (0.0, 1.0),
        (&AxisScale::Linear, &AxisScale::Linear), false, 1.1, -1.0, None).unwrap().0;
    assert!(zoomed.low > range.low && zoomed.high < range.high, "a finite range must not be clamped by overflow in its span");
}
/// A pan or home moves the view somewhere the wheel did not ask for: the
/// next notch goes on from there, not from the wheel's earlier request.
#[test]
fn a_pan_or_home_restarts_the_wheel() {
    let mut plot = session(frame(372, 460, None), PlotKind::Image);
    for _ in 0..25 { wheel(&mut plot, -1.0); }
    let zoomed = view(&plot);
    // Drag with the middle button a tenth of the box to the right.
    pointer(&mut plot, PointerAction::Press, (0.5, 0.5), Some(2), 0.0);
    pointer(&mut plot, PointerAction::Move, (0.6, 0.5), Some(2), 0.0);
    pointer(&mut plot, PointerAction::Release, (0.6, 0.5), Some(2), 0.0);
    let panned = view(&plot);
    assert!(centre(panned.0) < centre(zoomed.0) - 2.0, "the pan moved the view: {zoomed:?} -> {panned:?}");
    wheel(&mut plot, 1.0);
    let out = view(&plot);
    assert!((centre(out.0) - centre(panned.0)).abs() <= 1.0, "zoomed about the panned view: {panned:?} -> {out:?}");
    assert!(out.0.span() > panned.0.span());
    plot.reset_viewport().unwrap();
    wheel(&mut plot, -1.0);
    let mut fresh = session(frame(372, 460, None), PlotKind::Image);
    wheel(&mut fresh, -1.0);
    assert_eq!(view(&plot), view(&fresh), "from home, as a fresh plot");
}

/// A pan carries the continuous view and keeps its size, pan after
/// pan (a shift short of a pixel used to grow it by one, and the next pan
/// carried the wider view on).
#[test]
fn a_pan_keeps_the_view_size() {
    let mut plot = session(frame(372, 460, None), PlotKind::Image);
    for _ in 0..40 { wheel(&mut plot, -1.0); }
    let zoomed = view(&plot);
    close(zoomed.0.span(), 460.0 * 1.1f64.powi(-40));
    let mut last = zoomed;
    for to in [(0.57, 0.5), (0.61, 0.43), (0.38, 0.55), (0.5, 0.71), (0.45, 0.45)] {
        let axes = plot_axes(&plot);
        let canvas = plot.renderer().canvas_size();
        pointer(&mut plot, PointerAction::Press, (0.5, 0.5), Some(2), 0.0);
        pointer(&mut plot, PointerAction::Move, ((0.5 + to.0) / 2.0, (0.5 + to.1) / 2.0), Some(2), 0.0);
        let dragging = drawn(&plot);
        pointer(&mut plot, PointerAction::Move, to, Some(2), 0.0);
        pointer(&mut plot, PointerAction::Release, to, Some(2), 0.0);
        let panned = view(&plot);
        for span in [dragging.0.span(), panned.0.span(), panned.1.span()] { close(span, zoomed.0.span()); }
        // Pointer coordinates are whole canvas pixels, while the view is
        // continuous in source coordinates. Account only for that rounding.
        let (left, top, right, bottom) = axes.bounds;
        let point = |(x, y): (f64, f64)| (((left + x * (right - left)) * canvas.width).trunc(),
            ((1.0 - (top + y * (bottom - top))) * canvas.height).trunc());
        let (start, end) = (point((0.5, 0.5)), point(to));
        let travel = ((end.0 - start.0) / ((right - left) * canvas.width) * (axes.x_limits.1 - axes.x_limits.0),
            (end.1 - start.1) / ((bottom - top) * canvas.height) * (axes.y_limits.1 - axes.y_limits.0));
        close(centre(last.0) - centre(panned.0), travel.0);
        close(centre(last.1) - centre(panned.1), travel.1);
        last = panned;
    }
}

/// Restating a continuous view on small physical coordinates keeps it exact.
#[test]
fn an_image_view_on_physical_coordinates_round_trips_exactly() {
    let x: Vec<f64> = (0..61).map(|i| 1.0e-3 + i as f64 * 6.5e-6).collect();
    let y: Vec<f64> = (0..45).map(|i| 2.0e-3 + i as f64 * 6.5e-6).collect();
    let mut plot = session(frame(45, 61, Some((y, x))), PlotKind::Image);
    for notch in 0..30 {
        wheel(&mut plot, if notch < 20 { -0.3 } else { 0.7 });
        let shown = plot.viewport();
        let Some((Some(x), _)) = shown else { panic!("a committed view") };
        plot.set_x_limits(x.low, x.high).unwrap();
        assert_eq!(plot.viewport(), shown, "notch {notch}: committing the view it shows keeps it");
    }
}

/// A locked series moves one per whole notch, however the notch arrives;
/// at an end the travel past it is dropped.
#[test]
fn a_locked_series_moves_one_per_whole_notch() {
    let channels = axis("probe.channel", "channel", Coordinates::Index { origin: 0, len: 3 }, None, None);
    let delays = axis("probe.delay", "delay", Coordinates::Numeric(OwnedBuffer::from_vec((0..40).map(|i| i as f64).collect::<Vec<f64>>()).into()),
                      Some("s"), None);
    let values = (0..3).flat_map(|channel| (0..40).map(move |_| channel as f64)).collect();
    let mut plot = session(source(Domain::dense(vec![3, 40], vec![channels, delays]).unwrap(), values), PlotKind::Curve);
    let locked = |plot: &PlotSession| plot.display_state().interaction.series_lock.clone().map(|target| format!("{:?}", target.key));
    // Click the middle series (y = 1) to lock it.
    let axes = plot_axes(&plot);
    let (nx, ny) = axes.display_to_normalized(20.0, 1.0);
    let at = ((nx - axes.bounds.0) / (axes.bounds.2 - axes.bounds.0), (ny - axes.bounds.1) / (axes.bounds.3 - axes.bounds.1));
    pointer(&mut plot, PointerAction::Press, at, Some(1), 0.0);
    pointer(&mut plot, PointerAction::Release, at, Some(1), 0.0);
    let middle = locked(&plot).expect("a click on a series locks it");
    for _ in 0..3 { wheel(&mut plot, -0.3); }
    assert_eq!(locked(&plot).as_ref(), Some(&middle), "0.9 notch: still the middle series");
    wheel(&mut plot, -0.1);
    let below = locked(&plot).unwrap();
    assert_ne!(below, middle, "a whole notch moves one series");
    wheel(&mut plot, 0.5);
    wheel(&mut plot, 0.5);
    assert_eq!(locked(&plot).as_ref(), Some(&middle), "a notch back, in two halves");
    wheel(&mut plot, 3.0);
    let top = locked(&plot).unwrap();
    assert!(top != middle && top != below, "three notches stop at the end");
    wheel(&mut plot, -1.0);
    assert_eq!(locked(&plot).as_ref(), Some(&middle), "the first notch back from the end moves");
}

/// The colour rail of an image of small values (a current in A): the drag
/// starts at an end only, narrows the limits and lets go changed.
#[test]
fn a_small_valued_colour_rail_drags() {
    let bounds = NumericRange::new(0.0, 1.0e-13).unwrap();
    let value = NumericRange::new(2.0e-14, 8.0e-14).unwrap();
    assert!(begin_color_limit_drag(bounds, value, 5.0e-14, 0.035).unwrap().is_none(), "mid-rail holds no end");
    let drag = begin_color_limit_drag(bounds, value, 2.1e-14, 0.035).unwrap().expect("the low end");
    let moved = drag.moved(3.0e-14).unwrap();
    assert_eq!(moved.candidate, NumericRange::new(3.0e-14, 8.0e-14).unwrap());
    assert!(moved.changed(), "a move of a third of the limits is a change");
    assert!(!drag.moved(2.0e-14).unwrap().changed(), "back where it started is no change");
}

/// An axis of small values keeps its selector reach a fraction of the axis.
#[test]
fn a_small_valued_axis_keeps_its_selector_reach() {
    let transform = AxisTransform::linear("main", None, (0.0, 0.0, 1.0, 1.0), (0.0, 1.0e-13), (0.0, 1.0e-13));
    let range = SelectorState::new(SelectorValue::XRange(NumericRange::new(2.0e-14, 8.0e-14).unwrap()), 0, None).unwrap();
    let hit = |x: f64| hit_selector(CrosshairPoint::new(x, 5.0e-14).unwrap(), &[range], |_| true, None, &transform, 0.035, false, |_| Ok(None))
        .unwrap().map(|(_, handle)| handle.value());
    assert_eq!(hit(9.5e-14), None, "beyond the range, a sixth of the axis from its end");
    assert_eq!(hit(8.2e-14), Some("high"));
    assert_eq!(hit(5.0e-14), Some("body"));
}

#[test]
fn input_timestamps_keep_middle_double_clicks_independent_of_render_latency() {
    use zlc_plot::document::pointer_event;
    use zlc_plot::interaction::ClickHistory;
    let mut history=ClickHistory::new();
    let first=pointer_event(&json!({"event":"press","button":"middle","time":1.0}),Some(100.0)).unwrap();
    let second=pointer_event(&json!({"event":"press","button":"middle","time":1.2}),Some(103.0)).unwrap();
    let defaults=InteractionDefaults::default();
    assert!(!history.is_double_click(2,false,None,Some((20.,20.)),first.time.unwrap(),&defaults));
    assert!(history.is_double_click(2,false,None,Some((20.,20.)),second.time.unwrap(),&defaults));
    assert!(!history.is_double_click(2,false,None,Some((20.,20.)),10.,&defaults));
    assert!(!history.is_double_click(2,false,None,Some((20.,20.)),1.,&defaults),"reloaded browser clock must not synthesize a double click");
}
