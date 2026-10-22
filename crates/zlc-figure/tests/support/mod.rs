//! Test support: Fonts and MathText answered from a recorded draw or natively,
//! and the conversion of recorded Rust calls into the L2 stream's JSON form.
#![allow(dead_code)]
use std::collections::HashMap;
use std::path::PathBuf;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use zlc_figure::backend::{Bitmap, Gc, GrayImage};
use zlc_figure::fonts::{FontId, Fonts, MathParse, MathText};
use zlc_figure::ir::{self, Dash};
use zlc_figure::record::{Call, OwnedPath};
use zlc_figure::transform::M3;
use zlc_figure::{Error, Result};

fn canonical(v: &Value) -> String { serde_json::to_string(v).unwrap() }

/// A table of recorded answers: canonical key JSON -> value.
fn table(fonts: &Value, kind: &str) -> HashMap<String, Value> {
    let mut out = HashMap::new();
    if let Some(map) = fonts.get(kind).and_then(Value::as_object) {
        for (key, value) in map {
            let key: Value = serde_json::from_str(key).unwrap();
            out.insert(canonical(&key), value.clone());
        }
    }
    out
}

struct FontState { file: String, size: Option<(f64, f64)>, text: Option<Value>, image: Option<Bitmap> }

/// FT2Font answered from the L2 record's `fonts` section.
pub struct RecordedFonts {
    set_text: HashMap<String, Value>,
    glyphs: HashMap<String, Value>,
    fonts: Vec<FontState>,
    by_files: HashMap<Vec<String>, FontId>,
    pub misses: Vec<String>,
}

impl RecordedFonts {
    pub fn new(record: &Value) -> Self {
        let fonts = &record["fonts"];
        RecordedFonts { set_text: table(fonts, "set_text"), glyphs: table(fonts, "draw_glyphs_to_bitmap"),
                        fonts: Vec::new(), by_files: HashMap::new(), misses: Vec::new() }
    }

    fn text_answer(&mut self, font: FontId) -> Result<Value> {
        let state = &self.fonts[font];
        let key = state.text.clone().ok_or_else(|| Error::Backend("no set_text".into()))?;
        match self.set_text.get(&canonical(&key)) {
            Some(v) => Ok(v.clone()),
            None => {
                self.misses.push(format!("set_text {key}"));
                Err(Error::Backend(format!("unrecorded set_text {key}")))
            }
        }
    }
}

impl Fonts for RecordedFonts {
    fn get_font(&mut self, files: &[String], _hinting_factor: i64, _kerning_factor: i64) -> Result<FontId> {
        if let Some(&id) = self.by_files.get(files) { return Ok(id); }
        let id = self.fonts.len();
        self.fonts.push(FontState { file: files[0].clone(), size: None, text: None, image: None });
        self.by_files.insert(files.to_vec(), id);
        Ok(id)
    }
    fn clear(&mut self, font: FontId) -> Result<()> {
        self.fonts[font].text = None;
        self.fonts[font].image = None;
        Ok(())
    }
    fn set_size(&mut self, font: FontId, ptsize: f64, dpi: f64) -> Result<()> {
        self.fonts[font].size = Some((ptsize, dpi));
        Ok(())
    }
    fn set_text(&mut self, font: FontId, text: &str, angle: f64, flags: i32) -> Result<()> {
        let state = &mut self.fonts[font];
        let (size, dpi) = state.size.ok_or_else(|| Error::Backend("set_text before set_size".into()))?;
        state.text = Some(json!([state.file, size.to_bits(), dpi.to_bits(), text, angle.to_bits(), flags]));
        state.image = None;
        self.text_answer(font).map(|_| ())
    }
    fn get_width_height(&mut self, font: FontId) -> Result<(i64, i64)> {
        let a = self.text_answer(font)?;
        Ok((a["width"].as_i64().unwrap(), a["height"].as_i64().unwrap()))
    }
    fn get_descent(&mut self, font: FontId) -> Result<i64> { Ok(self.text_answer(font)?["descent"].as_i64().unwrap()) }
    fn get_bitmap_offset(&mut self, font: FontId) -> Result<(i64, i64)> {
        let a = self.text_answer(font)?;
        Ok((a["bitmap_offset"][0].as_i64().unwrap(), a["bitmap_offset"][1].as_i64().unwrap()))
    }
    fn draw_glyphs_to_bitmap(&mut self, font: FontId, antialiased: bool) -> Result<()> {
        let state = &mut self.fonts[font];
        let mut key = state.text.clone().ok_or_else(|| Error::Backend("draw before set_text".into()))?;
        key.as_array_mut().unwrap().push(json!(antialiased));
        match self.glyphs.get(&canonical(&key)) {
            Some(v) => {
                let shape = v["shape"].as_array().unwrap();
                state.image = Some(Bitmap { rows: shape[0].as_u64().unwrap() as usize, cols: shape[1].as_u64().unwrap() as usize,
                                            data: Vec::new(), id: Some(v["sha"].as_str().unwrap().to_owned()) });
                Ok(())
            }
            None => {
                self.misses.push(format!("draw_glyphs_to_bitmap {key}"));
                Err(Error::Backend(format!("unrecorded draw_glyphs_to_bitmap {key}")))
            }
        }
    }
    fn image(&mut self, font: FontId) -> Result<&dyn GrayImage> {
        match &self.fonts[font].image {
            Some(b) => Ok(b),
            None => Err(Error::Backend("image before draw_glyphs_to_bitmap".into())),
        }
    }
}

/// MathTextParser.parse answered from the record.
pub struct RecordedMath { answers: HashMap<String, Value>, pub misses: Vec<String> }

impl RecordedMath {
    pub fn new(record: &Value) -> Self { RecordedMath { answers: table(&record["fonts"], "mathtext"), misses: Vec::new() } }
}

impl MathText for RecordedMath {
    fn parse(&mut self, _rc: &ir::Rc, s: &str, dpi: f64, font: &ir::Font, antialiased: Option<bool>) -> Result<MathParse> {
        let key = json!([s, dpi.to_bits(), serde_json::to_value(font).unwrap(), antialiased, "raster"]);
        let Some(v) = self.answers.get(&canonical(&key)) else {
            self.misses.push(format!("mathtext {key}"));
            return Err(Error::Backend(format!("unrecorded mathtext {key}")));
        };
        let f = |k: &str| f64::from_bits(v[k].as_u64().unwrap());
        let shape = v["image"]["shape"].as_array().unwrap();
        Ok(MathParse {
            ox: f("ox"), oy: f("oy"), width: f("width"), height: f("height"), descent: f("descent"),
            image: Box::new(Bitmap { rows: shape[0].as_u64().unwrap() as usize, cols: shape[1].as_u64().unwrap() as usize,
                                     data: Vec::new(), id: Some(v["image"]["sha"].as_str().unwrap().to_owned()) }),
        })
    }
}

// ------------------------------------------------------------------ repo goldens
/// tests/data (made by tests/gen/goldens.py).
pub fn data_dir() -> PathBuf { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("data") }

/// One golden (`tests/data/goldens/<name>`): source, spec, calls, fonts,
/// queries and the IR.
pub fn golden(name: &str) -> Value {
    use std::io::Read;
    let mut text = String::new();
    let file = std::fs::File::open(data_dir().join("goldens").join(name)).unwrap();
    flate2::read::GzDecoder::new(file).read_to_string(&mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

/// The goldens' blob store as the IR reads it: tests/data/blobs, mirrored
/// under the test target directory with every `<sha>.bin.gz` (a blob too
/// large to keep raw) inflated to `<sha>.bin`.
pub fn golden_blob_root() -> PathBuf {
    static ROOT: std::sync::OnceLock<PathBuf> = std::sync::OnceLock::new();
    ROOT.get_or_init(|| {
        use std::io::Read;
        let source = data_dir().join("blobs");
        let mirror = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("golden-blobs");
        for prefix in std::fs::read_dir(&source).unwrap() {
            let prefix = prefix.unwrap().path();
            let target_dir = mirror.join(prefix.file_name().unwrap());
            std::fs::create_dir_all(&target_dir).unwrap();
            for file in std::fs::read_dir(&prefix).unwrap() {
                let file = file.unwrap().path();
                let name = file.file_name().unwrap().to_string_lossy().into_owned();
                let (target, gz) = match name.strip_suffix(".gz") {
                    Some(raw) => (target_dir.join(raw), true),
                    None => (target_dir.join(&name), false),
                };
                if target.exists() { continue; }
                let mut bytes = Vec::new();
                if gz {
                    flate2::read::GzDecoder::new(std::fs::File::open(&file).unwrap()).read_to_end(&mut bytes).unwrap();
                } else {
                    bytes = std::fs::read(&file).unwrap();
                }
                let digest = format!("{:x}", Sha256::digest(&bytes));
                assert_eq!(format!("{digest}.bin"), target.file_name().unwrap().to_string_lossy(), "{}", file.display());
                let tmp = target.with_extension(format!("tmp{}", std::process::id()));
                std::fs::write(&tmp, &bytes).unwrap();
                std::fs::rename(&tmp, &target).unwrap();
            }
        }
        mirror
    }).clone()
}

// ------------------------------------------------------------------ native fonts
/// Matplotlib's mpl-data/fonts/ttf in the oracle's environment.
const MPL_FONTS: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../.tools/science-env/Lib/site-packages/matplotlib/mpl-data/fonts/ttf");

/// The math fonts zlc-mathtext opens (ZLC_MPL_FONTS overrides).
pub fn math_fonts_dir() -> PathBuf { std::env::var("ZLC_MPL_FONTS").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from(MPL_FONTS)) }

/// Where `ir::Font` file names resolve (ZLC_FIGURE_FONT_DIRS, `;`-separated):
/// the repo's zlc_plot assets, matplotlib's fonts, Windows' fonts.
pub fn font_dirs() -> Vec<PathBuf> {
    match std::env::var("ZLC_FIGURE_FONT_DIRS") {
        Ok(dirs) => dirs.split(';').map(PathBuf::from).collect(),
        Err(_) => vec![PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(r"..\..\science\src\zlc_plot\assets"),
                       math_fonts_dir(), PathBuf::from(r"C:\Windows\Fonts")],
    }
}

pub fn resolver(dirs: &[PathBuf]) -> Box<dyn Fn(&str) -> Option<PathBuf>> {
    let dirs = dirs.to_vec();
    Box::new(move |name: &str| dirs.iter().map(|d| d.join(name)).find(|p| p.exists()))
}

pub fn native_fonts() -> zlc_figure::ftfonts::TextFonts { zlc_figure::ftfonts::TextFonts::new(resolver(&font_dirs())) }

pub fn native_math() -> zlc_figure::mathtext::NativeMathText {
    zlc_figure::mathtext::NativeMathText::new(math_fonts_dir(), resolver(&font_dirs()))
}

/// Draws `figure` through zlc-agg on the recorded `canvas` ([width, height]):
/// the sha256 of the RGBA buffer and the canvas.
pub fn paint(figure: &ir::Figure, canvas: &Value, fonts: &mut dyn Fonts, math: &mut dyn MathText)
             -> Result<(String, zlc_figure::agg::AggBackend)> {
    let mut agg = zlc_figure::agg::AggBackend::new(canvas[0].as_f64().unwrap(), canvas[1].as_f64().unwrap(), figure.dpi)?;
    zlc_figure::draw(figure, &mut agg, fonts, math)?;
    Ok((sha(agg.pixels()), agg))
}

// ------------------------------------------------------------------ L2 form
fn bits(v: f64) -> Value { json!(v.to_bits()) }
fn lbits(v: &[f64]) -> Value { Value::Array(v.iter().map(|x| bits(*x)).collect()) }

fn sha(bytes: &[u8]) -> String { format!("{:x}", Sha256::digest(bytes)) }

/// `l2.arr` for f64 data of `shape`.
pub fn arr_f64(shape: &[usize], values: &[f64]) -> Value {
    let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut out = json!({"dtype": "f8", "shape": shape, "sha": sha(&bytes)});
    if values.len() <= 64 { out["data"] = Value::Array(values.iter().map(|v| json!(v.to_bits())).collect()); }
    out
}

pub fn arr_u8(shape: &[usize], values: &[u8]) -> Value {
    let mut out = json!({"dtype": "u1", "shape": shape, "sha": sha(values)});
    if values.len() <= 64 { out["data"] = Value::Array(values.iter().map(|v| json!(*v)).collect()); }
    out
}

fn style_name<T: serde::Serialize>(v: &T) -> Value { serde_json::to_value(v).unwrap() }

pub fn gc_l2(gc: &Gc) -> Value {
    json!({
        "linewidth": bits(gc.linewidth), "alpha": bits(gc.alpha), "forced_alpha": gc.forced_alpha,
        "rgb": lbits(&gc.rgb), "antialiased": i64::from(gc.antialiased),
        "capstyle": style_name(&gc.capstyle), "joinstyle": style_name(&gc.joinstyle),
        "dashes": [gc.dashes.0.map(f64::to_bits), gc.dashes.1.as_ref().map(|d| lbits(d))],
        "cliprect": gc.cliprect.map(|c| lbits(&c)), "clippath": Value::Null,
        "snap": gc.snap, "hatch": gc.hatch, "hatch_color": lbits(&gc.hatch_color),
        "hatch_path": gc.hatch_path.as_ref().map(|h| path_l2(&OwnedPath { vertices: h.vertices.clone(), codes: Some(h.codes.clone()),
            should_simplify: h.should_simplify, simplify_threshold: h.simplify_threshold })),
        "hatch_linewidth": bits(gc.hatch_linewidth),
    })
}

pub fn path_l2(p: &OwnedPath) -> Value {
    let flat: Vec<f64> = p.vertices.iter().flatten().copied().collect();
    json!({
        "vertices": arr_f64(&[p.vertices.len(), 2], &flat),
        "codes": p.codes.as_ref().map(|c| arr_u8(&[c.len()], c)),
        "should_simplify": p.should_simplify, "simplify_threshold": bits(p.simplify_threshold),
    })
}

fn m6(m: &M3) -> Value { lbits(&m.to6()) }

fn rows4(v: &[[f64; 4]]) -> Value { arr_f64(&[v.len(), 4], &v.iter().flatten().copied().collect::<Vec<_>>()) }
fn rows2(v: &[[f64; 2]]) -> Value { arr_f64(&[v.len(), 2], &v.iter().flatten().copied().collect::<Vec<_>>()) }

fn dash_l2(d: &Dash) -> Value { json!([d.0.to_bits(), d.1.as_ref().map(|s| lbits(s))]) }

/// A recorded call in the L2 stream's JSON form.
pub fn call_l2(call: &Call) -> Value {
    let mut v = match call {
        Call::DrawPath { gc, path, trans, face } => json!({
            "gc": gc_l2(gc), "path": path_l2(path), "trans": m6(trans), "face": face.map(|f| lbits(&f)),
        }),
        Call::DrawMarkers { gc, marker_path, marker_trans, path, trans, face } => json!({
            "gc": gc_l2(gc), "marker_path": path_l2(marker_path), "marker_trans": m6(marker_trans),
            "path": path_l2(path), "trans": m6(trans), "face": face.map(|f| lbits(&f)),
        }),
        Call::DrawTextImage { image, x, y, angle, gc } => json!({
            "image": {"shape": [image.rows, image.cols], "sha": image.id},
            "x": x, "y": y, "angle": bits(*angle), "gc": gc_l2(gc),
        }),
        Call::DrawImage { gc, x, y, image } => json!({
            "gc": gc_l2(gc), "x": bits(*x), "y": bits(*y), "image": arr_u8(&[image.rows, image.cols, 4], &image.data),
        }),
        Call::DrawPathCollection { gc, master_transform, paths, transforms, offsets, offset_trans, facecolors,
                                   edgecolors, linewidths, dashes, antialiaseds } => {
            let tflat: Vec<f64> = transforms.iter().flatten().flatten().copied().collect();
            let aa: Vec<u8> = antialiaseds.iter().map(|&b| u8::from(b)).collect();
            json!({
                "gc": gc_l2(gc), "master_transform": m6(master_transform),
                "paths": paths.iter().map(path_l2).collect::<Vec<_>>(),
                "transforms": arr_f64(&[transforms.len(), 3, 3], &tflat), "offsets": rows2(offsets),
                "offset_trans": m6(offset_trans), "facecolors": rows4(facecolors), "edgecolors": rows4(edgecolors),
                "linewidths": arr_f64(&[linewidths.len()], linewidths),
                "dashes": dashes.iter().map(dash_l2).collect::<Vec<_>>(),
                "antialiaseds": arr_u8(&[aa.len()], &aa), "offset_position": "screen",
            })
        }
        Call::DrawQuadMesh { gc, master_transform, mesh_width, mesh_height, coordinates, offsets, offset_trans,
                             facecolors, antialiased, edgecolors } => {
            let flat: Vec<f64> = coordinates.iter().flatten().copied().collect();
            json!({
                "gc": gc_l2(gc), "master_transform": m6(master_transform), "mesh_width": mesh_width,
                "mesh_height": mesh_height, "coordinates": arr_f64(&[mesh_height + 1, mesh_width + 1, 2], &flat),
                "offsets": rows2(offsets), "offset_trans": m6(offset_trans), "facecolors": rows4(facecolors),
                "antialiased": antialiased, "edgecolors": rows4(edgecolors),
            })
        }
    };
    v["call"] = json!(call.name());
    v
}

/// The recorded L2 call with the oracle-only fields dropped.
pub fn expected_l2(call: &Value) -> Value {
    let mut c = call.clone();
    if c["call"] == "draw_text_image" {
        c["image"] = json!({"shape": c["image"]["shape"], "sha": c["image"]["sha"]});
    }
    c
}

/// Paths at which two JSON values differ (at most `limit`).
pub fn diff(a: &Value, b: &Value, path: &str, out: &mut Vec<String>, limit: usize) {
    if out.len() >= limit { return; }
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            let mut keys: Vec<&String> = x.keys().chain(y.keys()).collect();
            keys.sort();
            keys.dedup();
            for k in keys {
                match (x.get(k), y.get(k)) {
                    (Some(p), Some(q)) => diff(p, q, &format!("{path}.{k}"), out, limit),
                    (p, q) => out.push(format!("{path}.{k}: {} vs {}", short(p), short(q))),
                }
            }
        }
        (Value::Array(x), Value::Array(y)) if x.len() == y.len() => {
            for (i, (p, q)) in x.iter().zip(y).enumerate() { diff(p, q, &format!("{path}[{i}]"), out, limit); }
        }
        _ if a != b => out.push(format!("{path}: {} vs {}", short(Some(a)), short(Some(b)))),
        _ => {}
    }
}

fn short(v: Option<&Value>) -> String {
    match v {
        None => "<missing>".into(),
        Some(Value::Number(n)) if n.as_u64().is_some_and(|u| u > (1u64 << 53)) => {
            let f = f64::from_bits(n.as_u64().unwrap());
            format!("{n}({f:e})")
        }
        Some(v) => { let s = v.to_string(); if s.len() > 160 { format!("{}...", &s[..160]) } else { s } }
    }
}

// ------------------------------------------------------------------ runner
pub struct DirResult {
    pub exact: usize,
    pub total: usize,
    pub calls_equal: usize,
    pub calls_total: usize,
    pub failures: std::collections::BTreeMap<String, usize>,
}

/// Draw every `<stem>.ir.json` of `dir` and compare with its `<stem>.l2.json`.
/// Prints details for the first `show` failing fronts.
pub fn run_dir(dir: &std::path::Path, show: usize) -> DirResult {
    let only = std::env::var("ZLC_FIGURE_ONLY").ok();
    let verbose: usize = std::env::var("ZLC_FIGURE_VERBOSE").ok().and_then(|v| v.parse().ok()).unwrap_or(3);
    let mut names: Vec<std::path::PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".ir.json"))
        .filter(|p| only.as_ref().is_none_or(|o| p.to_string_lossy().contains(o.as_str())))
        .collect();
    names.sort();
    let mut r = DirResult { exact: 0, total: 0, calls_equal: 0, calls_total: 0, failures: Default::default() };
    let mut shown = 0usize;
    for ir_path in &names {
        r.total += 1;
        let stem = ir_path.to_string_lossy().trim_end_matches(".ir.json").to_owned();
        let figure: zlc_figure::ir::Figure = match serde_json::from_str(&std::fs::read_to_string(ir_path).unwrap()) {
            Ok(f) => f,
            Err(e) => { *r.failures.entry(format!("IR does not load: {e}")).or_default() += 1; continue; }
        };
        let record: Value = serde_json::from_str(&std::fs::read_to_string(format!("{stem}.l2.json")).unwrap()).unwrap();
        let expected = record["calls"].as_array().unwrap();
        let mut fonts = RecordedFonts::new(&record);
        let mut math = RecordedMath::new(&record);
        let mut rec = zlc_figure::record::Recorder::default();
        let result = zlc_figure::draw(&figure, &mut rec, &mut fonts, &mut math);
        r.calls_total += expected.len();
        let got: Vec<Value> = rec.calls.iter().map(call_l2).collect();
        let mut first_bad = None;
        for (i, e) in expected.iter().enumerate() {
            match got.get(i) {
                Some(g) if *g == expected_l2(e) => r.calls_equal += 1,
                _ => { if first_bad.is_none() { first_bad = Some(i); } }
            }
        }
        if result.is_ok() && got.len() == expected.len() && first_bad.is_none() {
            r.exact += 1;
            continue;
        }
        let reason = match &result {
            Err(e) => format!("error: {e}"),
            Ok(()) => match first_bad {
                Some(i) => format!("call {} differs", expected[i]["call"]),
                None => format!("{} calls vs {}", got.len(), expected.len()),
            },
        };
        *r.failures.entry(reason.chars().take(100).collect()).or_default() += 1;
        if shown < show {
            shown += 1;
            eprintln!("== {} : {reason} (got {} calls, expected {})", ir_path.display(), got.len(), expected.len());
            if let Some(i) = first_bad.or(if got.len() < expected.len() { Some(got.len()) } else { None }) {
                let e = expected_l2(&expected[i]);
                match got.get(i) {
                    Some(g) => {
                        let mut out = Vec::new();
                        diff(g, &e, &format!("call[{i}]"), &mut out, verbose);
                        for line in out { eprintln!("   {line}"); }
                    }
                    None => eprintln!("   call[{i}] missing; expected {}", e["call"]),
                }
            }
            for m in fonts.misses.iter().chain(&math.misses).take(2) { eprintln!("   miss: {}", &m[..m.len().min(300)]); }
        }
    }
    r
}

/// Compare every recorded geometry query of `dir` with the Scene API.
pub fn run_queries_dir(dir: &std::path::Path, show: usize) -> (usize, usize, Vec<String>) {
    let mut names: Vec<std::path::PathBuf> = std::fs::read_dir(dir).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".ir.json")).collect();
    names.sort();
    let (mut equal, mut total) = (0usize, 0usize);
    let mut bad = Vec::new();
    let b4 = |v: &Value| -> [u64; 4] { [0, 1, 2, 3].map(|i| v[i].as_u64().unwrap()) };
    for ir_path in &names {
        let stem = ir_path.to_string_lossy().trim_end_matches(".ir.json").to_owned();
        // An IR that does not load is a failed front, never a skipped one.
        let figure = match serde_json::from_str::<zlc_figure::ir::Figure>(&std::fs::read_to_string(ir_path).unwrap()) {
            Ok(f) => f,
            Err(e) => {
                total += 1;
                bad.push(format!("{}: IR does not load: {e}", ir_path.display()));
                continue;
            }
        };
        let record: Value = serde_json::from_str(&std::fs::read_to_string(format!("{stem}.l2.json")).unwrap()).unwrap();
        let mut fonts = RecordedFonts::new(&record);
        let mut math = RecordedMath::new(&record);
        let mut null = zlc_figure::query::NullBackend;
        let mut scene = match zlc_figure::query::Scene::new(&figure, &mut null, &mut fonts, &mut math) {
            Ok(s) => s,
            Err(e) => {
                total += 1;
                bad.push(format!("{}: scene: {e}", ir_path.display()));
                continue;
            }
        };
        let q = &record["queries"];
        for (i, a) in q["axes"].as_array().unwrap().iter().enumerate() {
            let checks: [(&str, Result<Vec<u64>>); 4] = [
                ("bbox", scene.axes_bbox(i).map(|p| p.map(f64::to_bits).to_vec())),
                ("position", Ok(scene.axes_position(i).map(f64::to_bits).to_vec())),
                ("trans_data", scene.trans_data(i).map(|m| m.to6().map(f64::to_bits).to_vec())),
                ("trans_axes", scene.trans_axes(i).map(|m| m.to6().map(f64::to_bits).to_vec())),
            ];
            for (name, got) in checks {
                total += 1;
                let want: Vec<u64> = a[name].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
                match got {
                    Ok(g) if g == want => equal += 1,
                    other => if bad.len() < show { bad.push(format!("{} axes {i} {name}: {:?} vs {:?}", ir_path.display(),
                        other.map(|g| g.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>()),
                        want.iter().map(|b| f64::from_bits(*b)).collect::<Vec<_>>())) },
                }
            }
        }
        for (path, want) in q["texts"].as_object().unwrap() {
            total += 1;
            let want = b4(want);
            match scene.text_extent_at(path) {
                Ok(p) if p.map(f64::to_bits) == want => equal += 1,
                other => if bad.len() < show { bad.push(format!("{} {path}: {:?} vs {:?}", ir_path.display(), other,
                                                                want.map(f64::from_bits))) },
            }
        }
    }
    (equal, total, bad)
}
