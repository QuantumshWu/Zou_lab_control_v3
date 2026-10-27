//! Navigation on a real session, through the pointer path the desktop
//! drives: the wheel zooms an image from where it asked the view to be (not
//! from that request put on the pixel grid), with any step size, down to one
//! source pixel and back; a pan or home restarts it; a locked series moves
//! one per whole notch; a small-valued colour rail and axis keep their
//! drags and reach.

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

/// Wheel `step` until the view has stood still for longer than any whole
/// pixel can take at that step (a third of the view: from 3 pixels to 1),
/// at most `limit` events; every event shrinks (or grows) the view or
/// leaves it, never the other way.  The event that last moved it, and the
/// view's widths.
fn wheel_until_still(session: &mut PlotSession, step: f64, limit: usize) -> (usize, Vec<f64>) {
    let settle = (3f64.ln() / (step.abs() * 1.1f64.ln())).ceil() as usize + 2;
    let mut widths = vec![view(session).0.span()];
    let mut moved = 0;
    for event in 1..=limit {
        wheel(session, step);
        let width = view(session).0.span();
        let last = *widths.last().unwrap();
        assert!(if step < 0.0 { width <= last } else { width >= last }, "event {event}: {last} -> {width}");
        if width != last { moved = event; widths.push(width); }
        if event - moved == settle { break; }
    }
    (moved, widths)
}

/// The events a zoom by `ratio` takes at `step` (a notch is 1.1).
fn events_for(ratio: f64, step: f64) -> f64 { ratio.ln() / (step.abs() * 1.1f64.ln()) }

/// Small wheel steps keep zooming an image down to one source pixel (the
/// frame's centre is a pixel's centre), and the first step back out moves
/// it; the same events in and out bring the view back exactly.
#[test]
fn small_wheel_steps_zoom_an_image_to_one_pixel_and_back() {
    let mut plot = session(frame(45, 61, None), PlotKind::Image);
    let home = view(&plot);
    assert_eq!((home.0.span(), home.1.span()), (61.0, 61.0), "home: the whole frame, letterboxed square");
    // Touchpad-sized events (deltaY 8: 0.08 notch) in, then as many out.
    for _ in 0..500 { wheel(&mut plot, -0.08); }
    let zoomed = view(&plot);
    assert!(zoomed.0.span() <= 4.0, "500 touchpad events (40 notches) zoom in: {zoomed:?}");
    for _ in 0..500 { wheel(&mut plot, 0.08); }
    assert_eq!(view(&plot), home, "the same travel out is home again");
    for step in [-0.08, -1.0] {
        let (events, widths) = wheel_until_still(&mut plot, step, 10_000);
        let last = view(&plot);
        assert_eq!((last.0.span(), last.1.span()), (1.0, 1.0), "step {step}: one source pixel fills the view after {events} events: {widths:?}");
        assert!((events as f64 - events_for(61.0, step)).abs() <= 1.0, "step {step}: there when the wheel asked for one pixel ({events} events)");
        assert_eq!((centre(last.0), centre(last.1)), (centre(home.0), centre(home.1)), "about the view's centre");
        wheel(&mut plot, -step);
        assert!(view(&plot).0.span() > 1.0, "step {step}: the first event back out moves the view");
        for _ in 0..60 { wheel(&mut plot, 1.0); }
        assert!(view(&plot).0.span() >= 61.0, "and out again");
        plot.reset_viewport().unwrap();
    }
}

/// The camera frame (an even width: the view's centre is a pixel edge, so
/// the closest view is the two pixels beside it) zooms down to them with
/// notches and with touchpad steps; it used to stop at 20 pixels with
/// notches and at hundreds with touchpad steps.
#[test]
fn the_camera_frame_zooms_to_the_pixels_at_its_centre() {
    for step in [-1.0, -0.08, -0.013] {
        let mut plot = session(frame(372, 460, None), PlotKind::Image);
        let (events, widths) = wheel_until_still(&mut plot, step, 20_000);
        let last = view(&plot);
        assert_eq!((last.0.span(), last.1.span()), (2.0, 2.0), "step {step}, after {events} events: {:?}", &widths[widths.len().saturating_sub(8)..]);
        // No stall on the way: the two pixels show when the wheel asked for two.
        assert!((events as f64 - events_for(230.0, step)).abs() <= 1.0, "step {step}: {events} events");
    }
}

/// Any step size and any number of events land where one equivalent step
/// does (up to the pixel grid, which both are put on).
#[test]
fn many_small_wheel_steps_are_one_big_step() {
    for (step, count) in [(-0.04, 25), (-0.08, 250), (-0.25, 40), (-1.0, 30), (0.1, 30), (1.0, 3)] {
        let mut many = session(frame(372, 460, None), PlotKind::Image);
        for _ in 0..count { wheel(&mut many, step); }
        let mut one = session(frame(372, 460, None), PlotKind::Image);
        wheel(&mut one, step * count as f64);
        assert_eq!(view(&many), view(&one), "{count} x {step}");
    }
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

/// A pan carries the view by whole pixels and keeps its size, pan after
/// pan (a shift short of a pixel used to grow it by one, and the next pan
/// carried the wider view on).
#[test]
fn a_pan_keeps_the_view_size() {
    let mut plot = session(frame(372, 460, None), PlotKind::Image);
    for _ in 0..40 { wheel(&mut plot, -1.0); }
    let zoomed = view(&plot);
    assert_eq!(zoomed.0.span(), 12.0);
    let mut last = zoomed;
    for (index, to) in [(0.57, 0.5), (0.61, 0.43), (0.38, 0.55), (0.5, 0.71), (0.45, 0.45)].into_iter().enumerate() {
        pointer(&mut plot, PointerAction::Press, (0.5, 0.5), Some(2), 0.0);
        pointer(&mut plot, PointerAction::Move, ((0.5 + to.0) / 2.0, (0.5 + to.1) / 2.0), Some(2), 0.0);
        let dragging = drawn(&plot);
        pointer(&mut plot, PointerAction::Move, to, Some(2), 0.0);
        pointer(&mut plot, PointerAction::Release, to, Some(2), 0.0);
        let panned = view(&plot);
        assert_eq!((dragging.0.span(), panned.0.span(), panned.1.span()), (12.0, 12.0, 12.0), "pan {index}: {last:?} -> {panned:?}");
        // The hand's travel, to the nearest pixel (12 pixels across the box).
        let travel = ((to.0 - 0.5) * 12.0, (0.5 - to.1) * 12.0);
        assert!((centre(last.0) - centre(panned.0) - travel.0).abs() <= 0.5, "pan {index}: x moved with the hand");
        assert!((centre(panned.1) - centre(last.1) - travel.1).abs() <= 0.5, "pan {index}: y moved with the hand");
        last = panned;
    }
}

/// The view snapped on coordinates of a small physical pitch is its own
/// snap: the cell edges are taken within binary noise of the coordinates
/// themselves (a slack in the axis' unit was below that noise).
#[test]
fn an_image_view_on_physical_coordinates_is_its_own_snap() {
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
