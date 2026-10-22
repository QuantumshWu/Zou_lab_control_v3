//! Timing of the draw semantics and of the full raster path on dumped fronts
//! (env-gated: ZLC_FIGURE_BENCH = comma-separated .ir.json paths, or
//! `@FILE` naming one path per line; ZLC_FIGURE_BLOBS, ZLC_FIGURE_FONT_DIRS /
//! ZLC_MPL_FONTS as in pixels_dump; ZLC_FIGURE_ROUNDS, default 20;
//! ZLC_FIGURE_BENCH_REPORT = a file for one JSON line per front).
//! Full raster (a fresh canvas, every pixel checked once against matplotlib's
//! full draw), interleaved per round: warm (fonts and the mathtext engine kept
//! between frames) and mathtext uncached (its layouts and rasters cleared
//! before the frame, faces kept: a new string).
mod support;

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use serde_json::{json, Value};

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

#[test]
fn bench_dumped_fronts() {
    let Ok(list) = std::env::var("ZLC_FIGURE_BENCH") else { return };
    let paths: Vec<String> = match list.strip_prefix('@') {
        Some(file) => std::fs::read_to_string(file).unwrap().lines().map(str::trim).filter(|l| !l.is_empty()).map(String::from).collect(),
        None => list.split(',').map(String::from).collect(),
    };
    zlc_figure::wire::set_blob_root(std::env::var("ZLC_FIGURE_BLOBS").ok().map(PathBuf::from));
    let rounds: usize = std::env::var("ZLC_FIGURE_ROUNDS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let mut report = std::env::var("ZLC_FIGURE_BENCH_REPORT").ok().map(|p| std::fs::File::create(p).unwrap());
    let mut text = support::native_fonts();
    let mut math = support::native_math();
    for path in &paths {
        let figure: zlc_figure::ir::Figure = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let record: Value = serde_json::from_str(&std::fs::read_to_string(path.replace(".ir.json", ".l2.json")).unwrap()).unwrap();
        // Semantics only: recording nothing, fonts answered from the record.
        let mut rfonts = support::RecordedFonts::new(&record);
        let mut rmath = support::RecordedMath::new(&record);
        let t0 = Instant::now();
        for _ in 0..rounds {
            let mut null = zlc_figure::query::NullBackend;
            zlc_figure::draw(&figure, &mut null, &mut rfonts, &mut rmath).unwrap();
        }
        let semantics = t0.elapsed().as_secs_f64() * 1e3 / rounds as f64;
        let canvas = &record["canvas"];
        let (sha, _) = support::paint(&figure, canvas, &mut text, &mut math).unwrap();
        assert_eq!(sha, record["full_draw_sha256"].as_str().unwrap(), "{path}");
        let mut frame = |clear: bool| -> f64 {
            if clear { math.clear_cache(); }
            let t = Instant::now();
            let mut agg = zlc_figure::agg::AggBackend::new(canvas[0].as_f64().unwrap(), canvas[1].as_f64().unwrap(), figure.dpi).unwrap();
            zlc_figure::draw(&figure, &mut agg, &mut text, &mut math).unwrap();
            t.elapsed().as_secs_f64() * 1e3
        };
        let (mut warm, mut uncached) = (Vec::with_capacity(rounds), Vec::with_capacity(rounds));
        for _ in 0..rounds {
            warm.push(frame(false));
            uncached.push(frame(true));
        }
        let (warm, uncached) = (median(warm), median(uncached));
        let calls = record["calls"].as_array().unwrap().len();
        eprintln!("{path}: {calls} calls; semantics {semantics:.3} ms/frame (mean of {rounds}); full raster {warm:.3} ms warm, \
                   {uncached:.3} ms mathtext uncached (medians of {rounds})");
        if let Some(file) = report.as_mut() {
            let line = json!({"front": path, "calls": calls, "rounds": rounds, "semantics_ms": semantics, "warm_ms": warm,
                              "uncached_ms": uncached});
            writeln!(file, "{line}").unwrap();
        }
    }
}
