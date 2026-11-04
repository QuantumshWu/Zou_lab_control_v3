//! A 1 x 1 image cell is square on screen, in both image boxes and after every
//! gesture. Everything goes through the document path the app takes.
use std::collections::BTreeSet;
use std::sync::Arc;

use serde_json::{json, Value as Json};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::store::{DatasetStore, Placement};
use zlc_core::{DType, OwnedBuffer};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::document::{axis_wire, DocumentRead, FigureDocument};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::interaction::transform::AxisTransform;
use zlc_plot::projection::output::Payload;
use zlc_plot::projection::Source;
use zlc_plot::render::geometry::{image_axis_span, image_coordinate_scale};
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, PlotSession, SessionServices};

#[derive(Clone)]
struct Read(Arc<Source>);

impl DocumentRead for Read {
    fn source(&self) -> &Arc<Source> { &self.0 }
    fn provenance_json(&self) -> String { "{}".into() }
}

fn services() -> SessionServices {
    SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
        defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() }
}

#[derive(Clone, Copy, PartialEq)]
enum Lattice { Uniform, Nonuniform, Reversed }

/// A `repeats` x `points` scan: `scan.point` is drawn on `lattice`, every value a checker square.
fn scan(repeats: usize, points: usize, lattice: Lattice) -> Read {
    let positions: Vec<f64> = (0..points).map(|i| match lattice {
        Lattice::Uniform => i as f64,
        Lattice::Nonuniform => 1.15f64.powi(i.min(60) as i32) + i as f64,
        Lattice::Reversed => (points - i) as f64 * 2.0,
    }).collect();
    let index = |id: &str, role: &str, len: usize| Axis::new(id.into(), id.into(), role.into(), Coordinates::Index { origin: 0, len }, None, None, None, None).unwrap();
    let point = Axis::new("scan.point".into(), "scan.point".into(), "scan-point".into(), Coordinates::Numeric(OwnedBuffer::from_vec(positions).into()), Some("s".into()), None, None, None).unwrap();
    let schema = DatasetSchema::new(Domain::dense(vec![repeats], vec![index("scan.repeat", "repeat", repeats)]).unwrap(), Domain::dense(vec![points], vec![point]).unwrap(),
        Domain::dense(vec![2], vec![index("site", "site", 2)]).unwrap(), ValueSchema::new(DType::F64, None, None, ValidityContract::Value).unwrap()).unwrap();
    let values: Vec<f64> = (0..repeats).flat_map(|r| (0..points).flat_map(move |p| [((r + p) % 2) as f64; 2])).collect();
    let block = DataBlock::new(schema.clone(), OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), None).unwrap();
    let snapshot = DatasetStore::new(schema.clone()).commit(vec![Placement::new([0, 0], [repeats, points], block).unwrap()]).unwrap();
    let runtime = Runtime::new();
    let generation = runtime.begin("fixture", vec![Declaration { signal: "scan".into(), schema: snapshot.schema().clone(), storage: Storage::Snapshot }], json!({})).unwrap();
    runtime.commit(&generation, &Bundle { inputs: vec![Input::Dataset { signal: "scan".into(), ordinal: None, snapshot }], metadata: json!({}), parents: vec![] }).unwrap();
    let read = runtime.read_coherent_views(&[(generation, "scan".into(), None)], &mut Default::default()).pop().unwrap().unwrap();
    Read(Arc::new(Source::from_read(&read).unwrap()))
}

/// Which semantic axis the repeat takes.
fn target(repeat_on_x: bool, size: &str, image_box: Option<&str>) -> Json {
    let (repeat, point) = if repeat_on_x { ("x", "y") } else { ("y", "x") };
    let mut target = json!({"semantic": {"kind": "image", "fate:repeat:scan.repeat": repeat, "fate:point:scan.point": point}, "size": size});
    if let Some(image_box) = image_box { target["parameters"] = json!({"image_box": image_box}); }
    target
}

/// Every command carries the panel's read, as the host's do: one without a read waits for data.
struct Panel { document: FigureDocument<Read>, read: Read, request: u64, time: f64, front: Option<zlc_plot::session::RgbaFront> }

const P: &str = "p";

impl Panel {
    fn open(read: Read, target: Json) -> Panel {
        let mut panel = Panel { document: FigureDocument::new(services(), None), read: read.clone(), request: 0, time: 10.0, front: None };
        panel.execute(json!({"action": "ready", "dpr": 1.0}));
        panel.sync(read, target, 1);
        panel
    }
    fn execute(&mut self, mut command: Json) -> Json {
        self.request += 1;
        command["id"] = json!(self.request);
        let reads = [(P.to_owned(), self.read.clone())].into();
        let mut execution = self.document.execute(&command.to_string(), reads, None, None).unwrap();
        let all: BTreeSet<String> = [P.to_owned()].into();
        self.document.finish_presentation(&all, &all);
        assert!(execution.state["error"].is_null(), "{}", execution.state["error"]);
        if let Some(front) = execution.fronts.remove(P) { self.front = Some(front); }
        assert!(execution.state["panels"][0]["error"].is_null(), "{}", execution.state["panels"][0]["error"]);
        execution.state
    }
    fn sync(&mut self, read: Read, target: Json, epoch: u64) -> Json {
        self.read = read;
        self.execute(json!({"action": "sync_panels", "inputs": [{"panel": P, "kind": "image", "window": 1, "signal": "scan", "epoch": epoch}], "targets": {P: target}}))
    }
    fn with_session<T>(&self, read: impl FnOnce(&PlotSession) -> T) -> T {
        let handle = self.document.session(P).expect("a session");
        let session = handle.lock().unwrap();
        read(&session)
    }
    fn axes(&self) -> AxisTransform {
        self.with_session(|session| session.raster_axes_snapshot().unwrap().into_iter().find(|axes| axes.role == "image").expect("image axes"))
    }
    fn configure(&mut self, change: Json) -> Json {
        let mut command = json!({"action": "configure", "panel": P});
        for (name, value) in change.as_object().unwrap() { command[name] = value.clone(); }
        self.execute(command)
    }
    /// One pointer command at box fractions (top origin) of the image axes.
    fn pointer(&mut self, event: &str, button: Option<&str>, at: (f64, f64), steps: f64) {
        let axes = self.axes();
        let (left, top, right, bottom) = axes.bounds;
        self.time += 1.0;
        let mut command = json!({"action": "pointer", "panel": P, "event": event, "x": left + at.0 * (right - left), "y": top + at.1 * (bottom - top),
                                 "axes": axis_wire(&axes), "steps": steps, "time": self.time});
        if let Some(button) = button { command["button"] = json!(button); }
        self.execute(command);
    }
    fn select_area(&mut self, from: (f64, f64), to: (f64, f64)) {
        self.pointer("press", Some("left"), from, 0.0);
        self.pointer("move", Some("left"), to, 0.0);
        self.pointer("release", Some("left"), to, 0.0);
    }
    fn zoom_to_area(&mut self, from: (f64, f64), to: (f64, f64)) {
        self.select_area(from, to);
        self.pointer("double_click", Some("middle"), (0.5, 0.5), 0.0);
    }
    fn wheel(&mut self, steps: f64) { self.pointer("scroll", None, (0.5, 0.5), steps); }
    fn pan(&mut self, from: (f64, f64), to: (f64, f64)) {
        self.pointer("press", Some("middle"), from, 0.0);
        self.pointer("move", Some("middle"), to, 0.0);
        self.pointer("release", Some("middle"), to, 0.0);
    }
    /// Home as a hand gets there: out far enough to click blank space, which
    /// drops a selection, then the middle double click with nothing selected.
    fn reset(&mut self) {
        self.wheel(8.0);
        self.pointer("press", Some("left"), (0.01, 0.01), 0.0);
        self.pointer("release", Some("left"), (0.01, 0.01), 0.0);
        self.pointer("double_click", Some("middle"), (0.5, 0.5), 0.0);
        assert_eq!(self.with_session(|session| session.viewport()), None, "home");
    }
    fn measured(&self) -> Measured { self.with_session(measure) }
}

/// What one frame draws: the box in pixels and the cells its limits show.
#[derive(Debug)]
struct Measured { box_w: f64, box_h: f64, cells_x: f64, cells_y: f64, footprint: (f64, f64) }

impl Measured {
    fn cell_w(&self) -> f64 { self.box_w / self.cells_x }
    fn cell_h(&self) -> f64 { self.box_h / self.cells_y }
    /// Cells square: the box's sides agree in cells to one pixel of the box.
    #[track_caller]
    fn assert_square(&self, what: &str) {
        let off = (self.cell_h() * self.cells_x - self.box_w).abs().max((self.cell_w() * self.cells_y - self.box_h).abs());
        assert!(off <= 1.0 + 1e-9, "{what}: cells {} x {} px in a {} x {} px box: {off} px off, {self:?}", self.cell_w(), self.cell_h(), self.box_w, self.box_h);
    }
}

fn measure(session: &PlotSession) -> Measured {
    let axes = session.raster_axes_snapshot().unwrap().into_iter().find(|axes| axes.role == "image").expect("image axes");
    let plan = session.surface_plan();
    let (width, height) = (plan.raster_size.0 as f64, plan.raster_size.1 as f64);
    let (left, top, right, bottom) = axes.bounds;
    let payload = session.projection().payload().clone();
    let Payload::Image(image) = &*payload else { panic!("an image") };
    let cells = |values: &[f64], limits: (f64, f64)| {
        let scale = image_coordinate_scale(values).unwrap();
        let drawn: Vec<f64> = values.iter().map(|&value| scale.space(value)).collect();
        let pitch = image_axis_span(&drawn).map_or(1.0, |span| span / drawn.len() as f64);
        (scale.space(limits.1) - scale.space(limits.0)).abs() / pitch
    };
    let (x, y) = (image.x.display.to_f64(), image.y.display.to_f64());
    Measured { box_w: ((right - left) * width).round(), box_h: ((bottom - top) * height).round(), cells_x: cells(&x, axes.x_limits), cells_y: cells(&y, axes.y_limits),
               footprint: (x.len() as f64, y.len() as f64) }
}

/// Every gesture a user has, with the invariant after each.
fn check_every_gesture(read: Read, repeat_on_x: bool, size: &str, image_box: Option<&str>) {
    let what = format!("repeat_on_x {repeat_on_x}, {size}, {image_box:?}");
    let mut panel = Panel::open(read, target(repeat_on_x, size, image_box));
    panel.measured().assert_square(&format!("{what}: home"));
    panel.zoom_to_area((0.2, 0.42), (0.7, 0.58));
    panel.measured().assert_square(&format!("{what}: area zoom"));
    panel.wheel(2.0);
    panel.measured().assert_square(&format!("{what}: wheel in"));
    panel.wheel(-3.0);
    panel.measured().assert_square(&format!("{what}: wheel out"));
    panel.pan((0.5, 0.5), (0.6, 0.45));
    panel.measured().assert_square(&format!("{what}: pan"));
    panel.reset();
    panel.measured().assert_square(&format!("{what}: reset"));
}

#[test]
fn cells_stay_square_through_every_gesture() {
    for lattice in [Lattice::Uniform, Lattice::Nonuniform, Lattice::Reversed] {
        for repeat_on_x in [false, true] {
            for image_box in [None, Some("fit")] {
                for size in ["2x2", "2x8"] {
                    check_every_gesture(scan(200, 30, lattice), repeat_on_x, size, image_box);
                }
            }
        }
    }
}

/// Square box: the selection grows to the square that contains it; its centre stays.
#[test]
fn a_square_box_expands_a_selection_to_the_square_containing_it() {
    let mut panel = Panel::open(scan(2048, 2048, Lattice::Uniform), target(false, "2x2", None));
    let home = panel.measured();
    assert_eq!(home.box_w, home.box_h);
    panel.zoom_to_area((0.1, 0.4), (0.4, 0.5));
    let zoomed = panel.measured();
    zoomed.assert_square("selection 3:1");
    assert_eq!((zoomed.box_w, zoomed.box_h), (home.box_w, home.box_h), "the box does not move");
    assert!((zoomed.cells_x - 0.3 * 2048.0).abs() < 8.0, "x is the long side: {zoomed:?}");
    assert!((zoomed.cells_y - zoomed.cells_x).abs() < 1e-6, "a square of cells: {zoomed:?}");
}

/// Fit box: home hugs the data and the box takes the selection's shape.
#[test]
fn a_fit_box_hugs_the_data_and_takes_the_selections_shape() {
    let mut panel = Panel::open(scan(50, 1000, Lattice::Uniform), target(false, "2x8", Some("fit")));
    let home = panel.measured();
    home.assert_square("home");
    assert_eq!(home.footprint, (1000.0, 50.0));
    assert!((home.cells_x - 1000.0).abs() < 1.0 && (home.cells_y - 50.0).abs() < 1.0, "no blank: {home:?}");
    assert!((home.box_w / home.box_h - 20.0).abs() < 0.01, "the box is the data's shape: {home:?}");
    assert!(home.box_w > 1000.0, "the strip takes the panel's width: {home:?}");
    let plan_before = panel.with_session(|session| session.surface_plan().clone());
    panel.wheel(1.0);
    panel.pan((0.5, 0.5), (0.55, 0.5));
    let moved = panel.with_session(|session| session.surface_plan().clone());
    assert_eq!(plan_before, moved, "wheel and pan keep the box");
    panel.reset();

    // A tall selection in a wide strip.
    panel.zoom_to_area((0.1, 0.1), (0.13, 0.9));
    let zoomed = panel.measured();
    zoomed.assert_square("tall selection");
    assert!((zoomed.cells_x - 30.0).abs() < 2.0 && (zoomed.cells_y - 40.0).abs() < 2.0, "the view is the selection: {zoomed:?}");
    assert!((zoomed.box_h / zoomed.box_w - zoomed.cells_y / zoomed.cells_x).abs() < 0.02, "the box is its shape: {zoomed:?}");
    panel.reset();
    let back = panel.measured();
    assert_eq!((back.box_w, back.box_h), (home.box_w, home.box_h), "reset comes home");
}

/// A selection too thin for its box to read: the box stops at one tick label
/// across and the view grows on the thin side, containing the selection.
#[test]
fn a_thin_selection_keeps_a_legible_box_and_square_cells() {
    let mut panel = Panel::open(scan(50, 1000, Lattice::Uniform), target(false, "2x8", Some("fit")));
    panel.zoom_to_area((0.0, 0.48), (1.0, 0.52));
    let zoomed = panel.measured();
    zoomed.assert_square("one row of the strip");
    let tick_px = 6.5f64 * 210.0 / 72.0;
    assert!(zoomed.box_h.min(zoomed.box_w) >= tick_px.round(), "legible: {zoomed:?}");
    assert!(zoomed.cells_y >= 2.0 && zoomed.cells_x >= 999.0, "the selection stays in view: {zoomed:?}");
}

/// A very thin image is the same: its box never collapses.
#[test]
fn an_extreme_image_keeps_a_legible_box() {
    for (repeats, points) in [(2, 2000), (2000, 2)] {
        let panel = Panel::open(scan(repeats, points, Lattice::Uniform), target(false, "1x2", Some("fit")));
        let home = panel.measured();
        home.assert_square("extreme home");
        assert!(home.box_h.min(home.box_w) >= (6.5f64 * 210.0 / 72.0).round(), "legible: {home:?}");
    }
}

/// The picture itself: run lengths of the checker along a row and a column.
fn checker_runs(panel: &Panel) -> (Vec<usize>, Vec<usize>) {
    let axes = panel.axes();
    let front = panel.front.as_ref().expect("a front");
    let (width, height) = (front.width as usize, front.height as usize);
    let (left, top, right, bottom) = axes.bounds;
    let (x0, x1) = ((left * width as f64).round() as usize + 2, (right * width as f64).round() as usize - 2);
    let (y0, y1) = ((top * height as f64).round() as usize + 2, (bottom * height as f64).round() as usize - 2);
    let pixels = front.pixels.as_slice();
    let at = |x: usize, y: usize| { let i = (y * width + x) * 4; (pixels[i], pixels[i + 1], pixels[i + 2]) };
    let (mid_x, mid_y) = ((x0 + x1) / 2 + 1, (y0 + y1) / 2 + 1);
    let across: Vec<_> = (x0..x1).map(|x| at(x, mid_y)).collect();
    // The two commonest colours of the row through the data are the checker;
    // anything else (blank, a selector's line) continues a run.
    let mut counts = std::collections::BTreeMap::new();
    for colour in &across { *counts.entry(*colour).or_insert(0usize) += 1; }
    let mut by_count: Vec<_> = counts.into_iter().collect();
    by_count.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
    let checker = [by_count[0].0, by_count[1].0];
    let runs = |line: Vec<(u8, u8, u8)>| {
        let mut changes = Vec::new();
        let mut current = None;
        for (index, colour) in line.iter().enumerate() {
            if checker.contains(colour) && current != Some(*colour) {
                if current.is_some() { changes.push(index); }
                current = Some(*colour);
            }
        }
        changes.windows(2).map(|pair| pair[1] - pair[0]).collect::<Vec<_>>()
    };
    (runs(across), runs((y0..y1).map(|y| at(mid_x, y)).collect()))
}

#[test]
fn the_drawn_checker_has_square_cells() {
    for (image_box, lattice, repeat_on_x) in [(None, Lattice::Uniform, false), (Some("fit"), Lattice::Uniform, false), (None, Lattice::Nonuniform, false),
                                              (Some("fit"), Lattice::Nonuniform, true), (Some("fit"), Lattice::Reversed, false)] {
        let mut panel = Panel::open(scan(16, 40, lattice), target(repeat_on_x, "2x4", image_box));
        for zoomed in [false, true] {
            if zoomed { panel.zoom_to_area((0.2, 0.2), (0.8, 0.5)); }
            let (across, down) = checker_runs(&panel);
            assert!(!across.is_empty() && !down.is_empty(), "{image_box:?} zoomed {zoomed}: no checker found");
            // The median: a selector's line across a boundary shifts one run.
            let median = |runs: &[usize]| { let mut runs = runs.to_vec(); runs.sort(); runs[runs.len() / 2] };
            assert!(median(&across).abs_diff(median(&down)) <= 1, "{image_box:?} {zoomed}: runs across {across:?}, down {down:?}");
        }
    }
}

/// The default box of a simple image and its picture, as main c16ed8c2 drew them.
#[test]
fn square_home_is_unchanged() {
    let panel = Panel::open(scan(2, 3, Lattice::Uniform), target(false, "2x2", None));
    let axes = panel.axes();
    let (left, top, right, bottom) = axes.bounds;
    assert_eq!([left, top, right, bottom].map(f64::to_bits), [4594829688393515476, 4594113149293138320, 4604222910473459653, 4605769524799273723]);
    assert_eq!((axes.x_limits, axes.y_limits), ((-0.5, 2.5), (2.0, -1.0)));
    let front = panel.front.as_ref().unwrap();
    let hash = front.pixels.as_slice().iter().fold(0xcbf29ce484222325u64, |hash, byte| (hash ^ *byte as u64).wrapping_mul(0x100000001b3));
    assert_eq!(hash, 0x4881a2e7cb40803);
}

/// The option's life: a control, a recipe, a restored figure and an export.
#[test]
fn the_option_is_saved_restored_and_exported() {
    let read = scan(50, 400, Lattice::Nonuniform);
    let mut panel = Panel::open(read.clone(), target(false, "2x8", None));
    let state = panel.configure(json!({}));
    let controls = state["panels"][0]["controls"].as_array().unwrap();
    let control = controls.iter().find(|control| control["name"] == "image_box").expect("a settings control");
    assert_eq!((control["label"].as_str(), control["value"].as_str(), &control["segmented"]), (Some("Image box"), Some("square"), &json!(true)));
    assert_eq!(control["choices"], json!([["square", "Square"], ["fit", "Fit data"]]));
    let recipe = panel.with_session(|session| session.figure_recipe().unwrap());
    let parameters = recipe.get("parameters").unwrap().as_map().unwrap();
    assert!(parameters.get("image_box").is_none(), "a figure that never chose saves as before");

    let square = panel.measured();
    panel.configure(json!({"parameters": {"image_box": "fit"}}));
    let fit = panel.measured();
    fit.assert_square("the control's edit");
    assert!(fit.box_w > square.box_w && fit.box_h < square.box_h, "the box hugs the data: {square:?} -> {fit:?}");
    panel.zoom_to_area((0.2, 0.2), (0.5, 0.8));
    let zoomed = panel.measured();
    zoomed.assert_square("before saving");

    // The saved target restores the box, view included.
    let saved = panel.with_session(|session| session.capture_configuration().unwrap().target().unwrap());
    assert_eq!(saved.get("parameters").unwrap().as_map().unwrap().get("image_box").and_then(|value| value.as_str()), Some("fit"));
    let restored = Panel::open(read.clone(), zlc_plot::configuration::Value::Map(saved).to_json());
    let again = restored.measured();
    again.assert_square("restored");
    assert_eq!((again.box_w, again.box_h), (zoomed.box_w, zoomed.box_h));

    // The export is its own session at 600 dpi on its own whole pixels.
    let export = panel.with_session(|session| session.export_session(&services()).unwrap());
    let exported = measure(&export);
    exported.assert_square("export");
    assert!((exported.box_h / exported.box_w - zoomed.box_h / zoomed.box_w).abs() < 0.01, "{exported:?} / {zoomed:?}");

    // Back to square: the stored selection grows to its square.
    panel.configure(json!({"parameters": {"image_box": "square"}}));
    let back = panel.measured();
    back.assert_square("back to square");
    assert_eq!(back.box_w, back.box_h);
}

/// Resizing and new data follow the box.
#[test]
fn a_fit_box_follows_the_panel_and_the_data() {
    let mut panel = Panel::open(scan(50, 1000, Lattice::Uniform), target(false, "1x2", Some("fit")));
    let small = panel.measured();
    small.assert_square("1x2");
    for size in ["2x8", "1x8", "8x8"] {
        panel.configure(json!({"size": size}));
        panel.measured().assert_square(size);
    }
    let wide = panel.measured();
    assert!(wide.box_w > small.box_w);
    // A camera's ROI changes shape while the panel lives.
    panel.configure(json!({"size": "2x4"}));
    panel.sync(scan(300, 100, Lattice::Uniform), target(false, "2x4", Some("fit")), 2);
    let roi = panel.measured();
    roi.assert_square("new shape");
    assert_eq!(roi.footprint, (100.0, 300.0));
    assert!((roi.box_h / roi.box_w - 3.0).abs() < 0.02, "the box follows: {roi:?}");
}

#[test]
fn the_new_presets_exist_and_grids_keep_their_capacity() {
    let layout = zlc_plot::layout::LayoutConfig::default();
    assert_eq!(layout.validate_preset("1x8"), Ok("1x8"));
    assert_eq!(layout.validate_preset("2X8"), Ok("2x8"));
    assert_eq!(layout.facet_max_cells(), 64);
    assert_eq!(layout.logical_size("2x8", zlc_plot::layout::Kind::Image), Ok((1498, 357)));
}
