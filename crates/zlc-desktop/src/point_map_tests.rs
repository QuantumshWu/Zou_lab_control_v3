//! A derived signal reads the calibration its actual parent used. Exercise the
//! board's scientific read, both presentations and the Figure Viewer's mount.
use std::path::Path;
use std::sync::Arc;

use serde_json::{Value, json};
use zlc_core::{DType, OwnedBuffer};
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Generation, Input, Publication, Read, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinate, Coordinates, DatasetSchema, Domain, ValueSchema, ValidityContract};
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::document::FigureDocument;
use zlc_plot::figure::ArchiveSaver;
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::projection::output::Payload;
use zlc_plot::projection::site_geometry::SAVED_GEOMETRIES;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, PointerAction, PointerEvent, SessionServices};
use crate::{archive::MountedArchive, science::{HostRead, Science}};

const SITE: &str = "calibration.site";
const PARAMETER: &str = "@logic/rabi/frequency";

fn schema(site_ids: &[usize], unit: &str) -> Arc<DatasetSchema> {
    let one = Domain::new(vec![1], vec![], vec![]).unwrap();
    let coordinates = site_ids.iter().map(|id| Coordinate::Integer(*id as i128)).collect();
    let axis = Axis::new(SITE.into(), "Site".into(), "site".into(), Coordinates::Values(coordinates),
        None, None, None, None).unwrap();
    DatasetSchema::new(one.clone(), one, Domain::dense(vec![site_ids.len()], vec![axis]).unwrap(),
        ValueSchema::new(DType::F64, Some(unit.into()), Some(if unit == "count" { "Counts" } else { "Rabi frequency" }.into()),
            ValidityContract::Components(vec![SITE.into()])).unwrap()).unwrap()
}

fn center(id: usize) -> [f64; 2] {
    let (col, row) = (((id - 1) % 7) as f64, ((id - 1) / 7) as f64);
    [55.0 + col * 13.2 - row * 2.0 + (id as f64).sin() * 0.5,
     30.0 + col * 2.3 + row * 13.8 + (id as f64).cos() * 0.7]
}
fn frequency(id: usize) -> f64 { 48.0 + 0.1 * center(id)[0] }
fn publish(runtime: &Runtime, run: &Generation, signal: &str, schema: Arc<DatasetSchema>, ids: &[usize],
           parents: Vec<Arc<Publication>>) -> Arc<Publication> {
    let values = ids.iter().map(|id| frequency(*id)).collect();
    let valid = ids.iter().map(|id| *id != 10).collect();
    let data = DataBlock::new(schema, OwnedBuffer::from_vec(values).into(), Validity::Mask(OwnedBuffer::from_vec(valid)),
        Some(OwnedBuffer::from_vec(vec![0.25; ids.len()]))).unwrap();
    runtime.commit(run, &Bundle { inputs: vec![Input::Event { signal: signal.into(), ordinal: 0, data, origin: None, shot_time: None }],
        metadata: json!({}), parents }).unwrap().publications[0].clone()
}
fn parameter(runtime: &Runtime) -> Read {
    let ids: Vec<usize> = (1..=35).collect();
    let geometry = json!({"coordinate_frame":"sensor_pixel_xy","x_axis_id":"camera.x","y_axis_id":"camera.y",
        "point_ids":ids.iter().map(|id|format!("S{id:02}")).collect::<Vec<_>>(), "status_axis_id":SITE,
        "labels":ids.iter().map(|id|id.to_string()).collect::<Vec<_>>(), "status_coordinates":ids,
        "coordinates_xy":ids.iter().map(|id|center(*id)).collect::<Vec<_>>()});
    let occupancy = "@logic/occupancy/counts";
    let full = schema(&ids, "count");
    let source = runtime.begin("occupancy", vec![Declaration{signal:occupancy.into(), schema:full.clone(), storage:Storage::Latest}],
        json!({"image_point_overlay_geometry":geometry})).unwrap();
    let parent = publish(runtime, &source, occupancy, full, &ids, vec![]);
    // The parameter has a different order and one absent site. Site 10 is
    // present with an invalid fit; it must keep its location and validity.
    let subset: Vec<_> = ids.into_iter().rev().filter(|id| *id != 12).collect();
    let per_site = schema(&subset, "kHz");
    let run = runtime.begin("rabi", vec![Declaration{signal:PARAMETER.into(), schema:per_site.clone(), storage:Storage::Latest}],
        json!({})).unwrap();
    Read::from_event(publish(runtime, &run, PARAMETER, per_site, &subset, vec![parent])).unwrap()
}

struct Panel { document: FigureDocument<HostRead>, read: HostRead, id: u64 }
impl Panel {
    fn new(read: Read, target: Value) -> Self {
        let services = SessionServices {renderers:Arc::new(NativeRendererFactory),fit:Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
            defaults:Arc::new(Defaults::default()),registry:None,interaction:InteractionDefaults::default()};
        let signal = read.key().signal.clone();
        let mut panel = Self { document: FigureDocument::new(services, Some(Arc::new(ArchiveSaver))),
            read: HostRead::new(read).unwrap(), id:0 };
        panel.command(json!({"action":"ready","dpr":1}));
        panel.command(json!({"action":"sync_panels","inputs":[{"panel":"p","kind":"point_map","window":1,"signal":signal.as_ref(),"epoch":1}],
            "targets":{"p":target}}));
        panel
    }
    fn command(&mut self, mut command: Value) {
        self.id += 1;
        command["id"] = json!(self.id);
        let state = self.document.execute(&command.to_string(),[("p".to_owned(), self.read.clone())].into(),None,None).unwrap().state;
        let panels = ["p".to_owned()].into();
        self.document.finish_presentation(&panels, &panels);
        assert!(state["error"].is_null(), "{state}");
        assert!(state["panels"].as_array().into_iter().flatten().all(|panel| panel["error"].is_null()), "{state}");
    }
    fn assert_mapping(&self) {
        let session = self.document.session("p").unwrap();
        let locked = session.lock().unwrap();
        let Payload::PointMap(map) = &**locked.projection().payload() else { panic!("expected Point Map") };
        assert_eq!(map.geometry.points.len(),35);
        assert_eq!(map.indices.len(),34);
        assert_eq!(map.geometry.frame.as_deref(),Some("sensor_pixel_xy"));
        let values = map.values.canonical.to_f64();
        let sigmas = map.sigma.as_ref().unwrap().to_f64();
        let mut coordinates = Vec::new();
        for (i, &index) in map.indices.iter().enumerate() {
            let id = map.coordinates[i].as_f64().unwrap() as usize;
            coordinates.push(id);
            let point = &map.geometry.points[index];
            assert_eq!(point.id,format!("S{id:02}"));
            assert_eq!(point.xy,center(id));
            assert_eq!(map.valid.bools()[i],id != 10);
            if id != 10 {
                assert!((values[i]-frequency(id)).abs() < 1e-10);
                assert!((sigmas[i]-0.25).abs() < 1e-10);
            }
        }
        coordinates.sort_unstable();
        assert_eq!(coordinates,(1..=35).filter(|id| *id != 12).collect::<Vec<_>>());
    }
    fn select_site(&mut self, id: usize) {
        let handle = self.document.session("p").unwrap();
        let mut plot = handle.lock().unwrap();
        let axes = plot.raster_axes_snapshot().unwrap().into_iter().find(|a| a.role == "image").expect("2D image axes");
        let [x,y] = center(id);
        let (nx,ny) = axes.display_to_normalized(x,y);
        for (step, action) in [PointerAction::Move, PointerAction::Press, PointerAction::Release, PointerAction::Leave].into_iter().enumerate() {
            let leave = action == PointerAction::Leave;
            let button = matches!(action, PointerAction::Press | PointerAction::Release).then_some(1);
            plot.raster_pointer_event(&PointerEvent { action, button, x: if leave {0.0} else {nx}, y: if leave {0.0} else {ny},
                double:false, step:0.0, key:None, axes: (!leave).then(|| axes.clone()), time:Some(1.0+step as f64/10.0) }).unwrap();
            if action == PointerAction::Press { continue; }
            let readout = plot.renderer().series_presentation().series_readout.expect("hit a calibrated site");
            assert_eq!(readout.mode.as_ref(), if action == PointerAction::Move {"hover"} else {"locked"});
            assert_eq!(readout.target.key[0][1].as_ref(), SITE);
            assert_eq!(readout.target.key[0][2].as_ref(), format!("S{id:02}"));
            assert!(readout.label.contains(&zlc_plot::pyfmt::format_g(frequency(id),6)) && readout.label.contains("kHz")
                && readout.label.contains("±") && readout.label.contains("0.25"), "{}",readout.label);
        }
    }
    fn save(&mut self, path: &Path) {
        self.command(json!({"action":"save","panel":"p","path":path,"overwrite":true}));
        assert!(path.metadata().unwrap().len() > 1024);
    }
}

#[test]
fn spatial_parameter_preserves_geometry_in_live_and_reopened_2d_and_3d_figures() {
    let runtime = Runtime::new();
    let mut panel = Panel::new(parameter(&runtime), json!({"semantic":{"kind":"point_map"},"size":"2x2"}));
    panel.assert_mapping();
    panel.select_site(20);
    let folder = std::env::temp_dir().join(format!("zlc-point-map-{}",std::process::id()));
    std::fs::create_dir_all(&folder).unwrap();
    let png = folder.join("site-frequency.png");
    panel.save(&png);
    assert!(std::fs::read(&png).unwrap().starts_with(b"\x89PNG"));
    let vector = folder.join("site-frequency.svg");
    panel.save(&vector);
    let svg = std::fs::read_to_string(&vector).unwrap();
    let filled_circles = svg.lines().filter(|path| path.contains("<path d=") && path.contains(" C ")
        && path.contains("fill:#") && path.contains(";stroke:")).count();
    assert_eq!(filled_circles, 33, "each valid site's face and outline must be on the same vector path");
    panel.command(json!({"action":"configure","panel":"p","parameters":{"presentation":"height_bars"}}));
    let bars = folder.join("site-frequency-bars.svg");
    panel.save(&bars);
    assert!(std::fs::read_to_string(&bars).unwrap().contains("<svg"));
    panel.save(&folder.join("site-frequency-bars.png"));
    // Legacy single-dataset Figures have a lineage but no materialized geometry section.
    for (path, presentation, legacy) in [(&png,"heatmap",false),(&bars,"height_bars",false),(&png,"heatmap",true)] {
        let viewer = Science::new(runtime.clone(), Arc::new(|_|{}),Arc::new(||{}),Arc::new(|_|{}));
        let archive_path = path.with_extension("npz");
        let mut archive = viewer.open(archive_path.to_str().unwrap()).unwrap();
        let saved = &archive.info["sections"][SAVED_GEOMETRIES]["data"][SITE];
        assert_eq!(saved["point_ids"].as_array().unwrap().len(),35);
        if legacy { archive.info["sections"].as_object_mut().unwrap().remove(SAVED_GEOMETRIES); }
        let opened = MountedArchive::mount(runtime.clone(),archive,vec!["p".into()]).unwrap();
        let target = opened.targets().remove("p").unwrap();
        assert_eq!(target["parameters"]["presentation"],presentation);
        let read = opened.inputs().remove(0).read.unwrap();
        Panel::new(read, target).assert_mapping();
    }
    println!("Point Map exports: {}", folder.display());
}
