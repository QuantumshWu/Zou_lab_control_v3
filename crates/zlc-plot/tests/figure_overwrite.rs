//! The document's `save` writes over no file it was not told to: with the
//! image, the archive or both already there it writes nothing and names them;
//! `overwrite: true` writes over them, a list of files only those it names
//! (however spelled); with neither there it writes both.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{json, Value as Json};
use zlc_archive::durable::resolve;
use zlc_core::dataset::{DataBlock, Validity};
use zlc_core::runtime::{Bundle, Declaration, Input, Runtime, Storage};
use zlc_core::schema::{Axis, Coordinates, DatasetSchema, Domain, ValidityContract, ValueSchema};
use zlc_core::OwnedBuffer;
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::document::{DocumentRead, FigureDocument};
use zlc_plot::figure::ArchiveSaver;
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::projection::Source;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, SessionServices};

#[derive(Clone)]
struct Frame { source: Arc<Source> }

impl DocumentRead for Frame {
    fn source(&self) -> &Arc<Source> { &self.source }
    fn provenance_json(&self) -> String { "null".into() }
}

fn axis(id: &str, role: &str, coordinates: Coordinates) -> Arc<Axis> {
    Axis::new(id.into(), role.into(), role.into(), coordinates, None, None, None, None).unwrap()
}

/// One 8 x 12 camera frame committed to a Runtime and read back as the desktop host reads it.
fn frame() -> Frame {
    let one = |id: &str, role: &str| Domain::dense(vec![1], vec![axis(id, role, Coordinates::Index { origin: 0, len: 1 })]).unwrap();
    let cell = Domain::dense(vec![8, 12], vec![axis("camera.spatial-y", "spatial-y", Coordinates::Index { origin: 0, len: 8 }),
                                               axis("camera.spatial-x", "spatial-x", Coordinates::Index { origin: 0, len: 12 })]).unwrap();
    let schema = DatasetSchema::new(one("probe.repeat", "repeat"), one("probe.frame", "readout-event"), cell,
                                    ValueSchema::new(zlc_core::DType::F64, Some("count".into()), None, ValidityContract::Value).unwrap()).unwrap();
    let runtime = Runtime::new();
    let generation = runtime.begin("probe", vec![Declaration { signal: "frames".into(), schema: schema.clone(), storage: Storage::Latest }], json!({})).unwrap();
    let values = (0..96).map(|i| ((i * 7919) % 251) as f64).collect();
    let data = DataBlock::new(schema, OwnedBuffer::from_vec(values).into(), Validity::Uniform(true), None).unwrap();
    runtime.commit(&generation, &Bundle { inputs: vec![Input::Event { signal: "frames".into(), ordinal: 0, data, origin: None }],
                                         metadata: json!({}), parents: vec![] }).unwrap();
    let read = runtime.read_coherent_views(&[(generation, Arc::from("frames"), None)], &mut Default::default()).pop().unwrap().unwrap();
    Frame { source: Arc::new(Source::from_read(&read).unwrap()) }
}

#[test]
fn a_save_writes_over_no_file_unless_told() {
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
                                     defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    let mut document = FigureDocument::new(services, Some(Arc::new(ArchiveSaver)));
    let all: std::collections::BTreeSet<String> = ["p".to_owned()].into();
    document.execute(r#"{"id": 0, "action": "ready", "dpr": 1.0}"#, Default::default(), None, None).unwrap();
    document.finish_presentation(&all, &all);
    let presented = json!({"id": 1, "action": "sync_panels", "inputs": [{"panel": "p", "kind": "image", "window": 1, "signal": "frames", "epoch": 1}],
                           "targets": {"p": {"semantic": {"kind": "image"}}}});
    document.execute(&presented.to_string(), [("p".to_owned(), frame())].into(), None, None).unwrap();
    document.finish_presentation(&all, &all);

    let out = Path::new(env!("CARGO_TARGET_TMPDIR")).join("figure_overwrite");
    let _ = std::fs::remove_dir_all(&out);
    std::fs::create_dir_all(&out).unwrap();
    let out = resolve(&out).unwrap();
    let (image, archive) = (out.join("fig.png"), out.join("fig.npz"));
    let named = |paths: &[&PathBuf]| json!(paths.iter().map(|path| path.display().to_string()).collect::<Vec<_>>());
    let mut request = 1;
    let mut save = |overwrite: Option<Json>| -> Json {
        request += 1;
        let mut command = json!({"id": request, "action": "save", "panel": "p", "path": image.display().to_string()});
        if let Some(overwrite) = overwrite { command["overwrite"] = overwrite; }
        let state = document.execute(&command.to_string(), Default::default(), None, None).unwrap().state;
        document.finish_presentation(&all, &all);
        assert!(state["error"].is_null(), "{}", state["error"]);
        state["saved"].clone()
    };
    // What is on disk: each file's bytes and modification time, or nothing.
    let disk = |path: &Path| std::fs::read(path).ok().map(|bytes| (bytes, std::fs::metadata(path).unwrap().modified().unwrap()));
    let plant = |path: &Path, bytes: &[u8]| std::fs::write(path, bytes).unwrap();
    let real = |path: &Path, magic: &[u8]| disk(path).is_some_and(|(bytes, _)| bytes.starts_with(magic));

    assert_eq!(save(None), named(&[&image, &archive]), "neither there: both written");
    assert!(real(&image, b"\x89PNG") && real(&archive, b"PK"), "a picture and an archive");

    let before = (disk(&image), disk(&archive));
    assert_eq!(save(None), json!({"exists": named(&[&image, &archive])}), "both there");
    assert_eq!(save(Some(json!(false))), json!({"exists": named(&[&image, &archive])}), "both there, overwrite false");
    assert!((disk(&image), disk(&archive)) == before, "a refused save wrote");

    std::fs::remove_file(&image).unwrap();
    plant(&archive, b"the archive already there");
    let before = disk(&archive);
    assert_eq!(save(None), json!({"exists": named(&[&archive])}), "the archive there");
    assert!(disk(&image).is_none() && disk(&archive) == before, "a refused save wrote");

    std::fs::remove_file(&archive).unwrap();
    plant(&image, b"the image already there");
    let before = disk(&image);
    assert_eq!(save(None), json!({"exists": named(&[&image])}), "the image there");
    assert!(disk(&image) == before && disk(&archive).is_none(), "a refused save wrote");

    // Agreeing to the archive (not there) leaves the image unagreed.
    assert_eq!(save(Some(json!([archive.display().to_string()]))), json!({"exists": named(&[&image])}), "the image there, the archive agreed");
    assert!(disk(&image) == before && disk(&archive).is_none(), "a refused save wrote");

    // Both there, one agreed: only the other is named, and nothing is written.
    plant(&archive, b"the archive already there");
    let before = (disk(&image), disk(&archive));
    let respelled = out.join("elsewhere").join("..").join("FIG.NPZ").display().to_string();
    assert_eq!(save(Some(json!([respelled]))), json!({"exists": named(&[&image])}), "the archive agreed as FIG.NPZ");
    assert_eq!(save(Some(json!([image.display().to_string()]))), json!({"exists": named(&[&archive])}), "the image agreed");
    assert_eq!(save(Some(json!([]))), json!({"exists": named(&[&image, &archive])}), "nothing agreed");
    assert!((disk(&image), disk(&archive)) == before, "a refused save wrote");

    // Every file there agreed: written.
    std::fs::remove_file(&image).unwrap();
    assert_eq!(save(Some(json!([respelled]))), named(&[&image, &archive]), "the archive alone there and agreed");
    assert!(real(&image, b"\x89PNG") && real(&archive, b"PK"), "both written");
    plant(&image, b"the image already there");
    plant(&archive, b"the archive already there");
    assert_eq!(save(Some(json!([image.display().to_string(), archive.display().to_string()]))), named(&[&image, &archive]), "both agreed");
    assert!(real(&image, b"\x89PNG") && real(&archive, b"PK"), "both written over");

    plant(&image, b"the image already there");
    plant(&archive, b"the archive already there");
    assert_eq!(save(Some(json!(true))), named(&[&image, &archive]), "overwrite writes both");
    assert!(real(&image, b"\x89PNG") && real(&archive, b"PK"), "both written over");
}
