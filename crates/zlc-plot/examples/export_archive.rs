//! A figure archive saved again as PNG, SVG and PDF by the native renderer,
//! as a run's `run.figure` saves it (one fresh export session per image).
//!
//! `export_archive <archive.npz> <out base> [png,svg,pdf]` writes
//! `<out base>.<format>` (and the archive beside them).
use std::path::{Path, PathBuf};
use std::sync::Arc;

use zlc_archive::read_archive;
use zlc_plot::configuration::resolve::Defaults;
use zlc_plot::figure::{read_figure_plot, save_run_figure, RunFigure};
use zlc_plot::fit::FitEngine;
use zlc_plot::interaction::gesture::InteractionDefaults;
use zlc_plot::projection::Source;
use zlc_plot::render::panel::NativeRendererFactory;
use zlc_plot::session::{EngineFitLayer, SessionServices};

fn main() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    let [_, archive, base, rest @ ..] = &args[..] else { return Err("usage: export_archive <archive.npz> <out base> [png,svg,pdf]".into()) };
    let formats = rest.first().map_or("png,svg,pdf", String::as_str);
    let archive = read_archive(Path::new(archive)).map_err(|e| e.message)?;
    let plot = read_figure_plot(&archive, "data").map_err(|e| e.message)?;
    let sections = archive.info.get("sections").ok_or("an archive without sections")?.to_serde();
    let mut recipe = sections["plot"]["data"].clone();
    recipe.as_object_mut().ok_or("a recipe that is not an object")?.remove("overlay");
    let services = SessionServices { renderers: Arc::new(NativeRendererFactory), fit: Arc::new(EngineFitLayer::new(Arc::new(FitEngine::default()))),
                                     defaults: Arc::new(Defaults::default()), registry: None, interaction: InteractionDefaults::default() };
    for format in formats.split(',') {
        let started = std::time::Instant::now();
        let figure = RunFigure { path: PathBuf::from(format!("{base}.{format}")), data: Source::of_archived(&plot.data), overlay: plot.overlay.clone(),
                                 recipe: recipe.clone(), source: sections["source"].clone(), lineage: sections["lineage"].clone(), image: true };
        let (image, _) = save_run_figure(figure, &services).map_err(|e| e.message)?;
        let image = image.ok_or("no image written")?;
        println!("{} {} bytes {:.0} ms", image.display(), std::fs::metadata(&image).map_err(|e| e.to_string())?.len(), started.elapsed().as_secs_f64() * 1e3);
    }
    Ok(())
}
