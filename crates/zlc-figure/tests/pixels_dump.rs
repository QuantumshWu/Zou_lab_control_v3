//! End to end: IR -> zlc-figure -> zlc-agg (+ zlc-text, zlc-mathtext) ->
//! pixels == the pixels of matplotlib's own full draw of the materialized
//! figure (env-gated: ZLC_FIGURE_DUMP and/or ZLC_FIGURE_CORPUS,
//! ZLC_FIGURE_BLOBS; fonts from ZLC_FIGURE_FONT_DIRS, `;`-separated; math
//! fonts from ZLC_MPL_FONTS = matplotlib's mpl-data/fonts/ttf;
//! ZLC_FIGURE_PIXELS_REPORT = a file for one JSON line per front;
//! ZLC_FIGURE_PIXELS_SAVE = a directory for the RGBA of differing fronts).
//! One mathtext engine serves every front, as one renderer's does.
mod support;

use std::io::Write;
use std::path::PathBuf;

use serde_json::{json, Value};

#[test]
fn dumped_fronts_paint_matplotlibs_pixels() {
    zlc_figure::wire::set_blob_root(std::env::var("ZLC_FIGURE_BLOBS").ok().map(PathBuf::from));
    let mut dirs: Vec<PathBuf> = Vec::new();
    if let Ok(dir) = std::env::var("ZLC_FIGURE_DUMP") { if !dir.is_empty() { dirs.push(dir.into()); } }
    if let Ok(root) = std::env::var("ZLC_FIGURE_CORPUS") {
        let mut sub: Vec<PathBuf> = std::fs::read_dir(root).unwrap().map(|e| e.unwrap().path()).filter(|p| p.is_dir()).collect();
        sub.sort();
        dirs.extend(sub);
    }
    if dirs.is_empty() { return; }
    let only = std::env::var("ZLC_FIGURE_ONLY").ok();
    let mut report = std::env::var("ZLC_FIGURE_PIXELS_REPORT").ok().map(|p| std::fs::File::create(p).unwrap());
    let save = std::env::var("ZLC_FIGURE_PIXELS_SAVE").ok().map(PathBuf::from);
    let mut math = support::native_math();
    let (mut equal, mut total, mut front_equal, mut front_total) = (0, 0, 0, 0);
    let (mut math_equal, mut math_total) = (0, 0);
    let mut bad = Vec::new();
    for dir in &dirs {
        let mut names: Vec<PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path())
            .filter(|p| p.to_string_lossy().ends_with(".ir.json"))
            .filter(|p| only.as_ref().is_none_or(|o| p.to_string_lossy().contains(o.as_str()))).collect();
        names.sort();
        for ir_path in names {
            let stem = ir_path.to_string_lossy().trim_end_matches(".ir.json").to_owned();
            let record: Value = serde_json::from_str(&std::fs::read_to_string(format!("{stem}.l2.json")).unwrap()).unwrap();
            let has_math = record["fonts"].get("mathtext").is_some_and(|m| !m.as_object().unwrap().is_empty());
            let figure: zlc_figure::ir::Figure = serde_json::from_str(&std::fs::read_to_string(&ir_path).unwrap()).unwrap();
            total += 1;
            math_total += usize::from(has_math);
            let mut fonts = support::native_fonts();
            let outcome = support::paint(&figure, &record["canvas"], &mut fonts, &mut math)
                .map(|(sha, agg)| (sha == record["full_draw_sha256"].as_str().unwrap(), agg));
            if let Some(file) = report.as_mut() {
                let line = json!({"front": stem, "mathtext": has_math, "equal": outcome.as_ref().ok().map(|o| o.0),
                                  "error": outcome.as_ref().err().map(ToString::to_string),
                                  "compose_contract": record["front_sha256"] == record["full_draw_sha256"]});
                writeln!(file, "{line}").unwrap();
            }
            match outcome {
                Err(e) => bad.push(format!("{}: {e}", ir_path.display())),
                Ok((true, _)) => {
                    equal += 1;
                    math_equal += usize::from(has_math);
                    if record["front_sha256"] == record["full_draw_sha256"] { front_total += 1; front_equal += 1; }
                }
                Ok((false, agg)) => {
                    bad.push(format!("{}: pixels differ", ir_path.display()));
                    if record["front_sha256"] == record["full_draw_sha256"] { front_total += 1; }
                    if let Some(dir) = &save {
                        let name = PathBuf::from(&stem);
                        let parent = name.parent().unwrap().file_name().unwrap().to_string_lossy().into_owned();
                        let file = format!("{parent}.{}.rgba", name.file_name().unwrap().to_string_lossy());
                        std::fs::write(dir.join(file), agg.pixels()).unwrap();
                    }
                }
            }
        }
    }
    for b in bad.iter().take(12) { eprintln!("  {b}"); }
    eprintln!("pixels: full-draw identical {equal}/{total} (mathtext fronts {math_equal}/{math_total}); of the compose-contract fronts {front_equal}/{front_total} equal the live front");
    assert_eq!(equal, total);
}
