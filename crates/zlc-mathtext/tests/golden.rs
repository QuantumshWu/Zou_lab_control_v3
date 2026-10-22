//! Matplotlib 3.10.8 goldens (`tests/gen/gen_golden.py`): every case's parse box,
//! `ship` glyph list and rectangles as float bits, and the antialiased and
//! monochrome rasters' metrics bits, size and CRC-32 (full bytes for some).
//! `ZLC_MATHTEXT_GOLDEN` names extra golden files (`;`-separated), e.g. the full set
//! under `zlc_v3_latency_runs/golden_mathtext`. `ZLC_MPL_FONTS` overrides the math
//! font directory recorded in the golden header.

use std::path::{Path, PathBuf};

use serde_json::Value;
use zlc_mathtext::{Config, Error, MathText, Raster, Request};

const HELVETICA: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../../science/src/zlc_plot/assets/helvetica-light-587ebe5a59211.ttf");
/// Grammar the port refuses; a case using one may be refused instead of matched.
const UNSUPPORTED: [&str; 14] = ["\\hat", "\\overline", "\\dfrac", "\\text", "\\operatorname", "\\boldsymbol",
    "\\binom", "\\middle", "\\genfrac", "\\overset", "\\underset", "\\substack", "\\widehat", "\\tilde"];

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = !0u32;
    for &byte in bytes {
        crc ^= byte as u32;
        for _ in 0..8 { crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 } }
    }
    !crc
}

fn base64(text: &str) -> Vec<u8> {
    let value = |c: u8| match c {
        b'A'..=b'Z' => c - b'A',
        b'a'..=b'z' => c - b'a' + 26,
        b'0'..=b'9' => c - b'0' + 52,
        b'+' => 62,
        _ => 63,
    };
    let clean: Vec<u8> = text.bytes().filter(|&c| c != b'=').collect();
    let mut out = Vec::with_capacity(clean.len() * 3 / 4);
    for chunk in clean.chunks(4) {
        let mut acc = 0u32;
        for (i, &c) in chunk.iter().enumerate() { acc |= (value(c) as u32) << (18 - 6 * i) }
        for i in 0..chunk.len() - 1 { out.push((acc >> (16 - 8 * i)) as u8) }
    }
    out
}

fn float(value: &Value) -> f64 { f64::from_bits(u64::from_str_radix(value.as_str().unwrap(), 16).unwrap()) }
fn hex(value: f64) -> String { format!("{:016x}", value.to_bits()) }

#[derive(Default, Debug)]
struct Tally { cases: usize, layouts: usize, rasters: usize, images: usize, errors: usize, unsupported: usize, failures: Vec<String> }

fn check_raster(tally: &mut Tally, label: &str, expected: &Value, raster: &Raster) {
    let e = expected.as_array().unwrap();
    let metrics = [raster.ox, raster.oy, raster.width, raster.height, raster.depth];
    let mut problems = Vec::new();
    for (i, name) in ["ox", "oy", "width", "height", "depth"].iter().enumerate() {
        if hex(metrics[i]) != e[i].as_str().unwrap() { problems.push(format!("{name} {} != {}", metrics[i], float(&e[i]))) }
    }
    let (w, h) = (e[5].as_u64().unwrap() as usize, e[6].as_u64().unwrap() as usize);
    if (raster.image_width, raster.image_height) != (w, h) {
        problems.push(format!("image {}x{} != {w}x{h}", raster.image_width, raster.image_height));
    } else if crc32(&raster.image) != e[7].as_u64().unwrap() as u32 {
        let mut detail = String::from("pixels differ");
        if let Some(bytes) = e.get(8).and_then(Value::as_str) {
            let bytes = base64(bytes);
            let differing = bytes.iter().zip(&raster.image).filter(|(a, b)| a != b).count();
            detail = format!("{differing} of {} pixels differ", bytes.len());
        }
        problems.push(detail);
    } else if e.len() > 8 {
        if base64(e[8].as_str().unwrap()) == raster.image { tally.images += 1 } else { problems.push("image bytes differ".into()) }
    }
    if problems.is_empty() { tally.rasters += 1 } else { tally.failures.push(format!("{label}: {}", problems.join("; "))) }
}

fn check_layout(case: &Value, layout: &zlc_mathtext::Layout) -> Vec<String> {
    let mut problems = Vec::new();
    let boxed = case["box"].as_array().unwrap();
    for (i, (name, value)) in [("width", layout.width), ("height", layout.height), ("depth", layout.depth)].iter().enumerate() {
        if hex(*value) != boxed[i].as_str().unwrap() { problems.push(format!("box {name} {value} != {}", float(&boxed[i]))) }
    }
    let glyphs = case["glyphs"].as_array().unwrap();
    if glyphs.len() != layout.glyphs.len() {
        problems.push(format!("{} glyphs != {}", layout.glyphs.len(), glyphs.len()));
    }
    for (i, (expected, glyph)) in glyphs.iter().zip(&layout.glyphs).enumerate() {
        let e = expected.as_array().unwrap();
        let file = glyph.font_file.file_name().unwrap().to_string_lossy().into_owned();
        let ok = hex(glyph.ox) == e[0].as_str().unwrap() && hex(glyph.oy) == e[1].as_str().unwrap()
            && file == e[2].as_str().unwrap() && glyph.num as u64 == e[3].as_u64().unwrap()
            && hex(glyph.fontsize) == e[4].as_str().unwrap();
        if !ok {
            problems.push(format!("glyph {i}: ({}, {}, {file}, {}, {}) != ({}, {}, {}, {}, {})", glyph.ox, glyph.oy,
                glyph.num, glyph.fontsize, float(&e[0]), float(&e[1]), e[2], e[3], float(&e[4])));
            break;
        }
    }
    let expected_rects: Vec<Vec<String>> = case["rects"].as_array().unwrap().iter()
        .map(|r| r.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect()).collect();
    let rects: Vec<Vec<String>> = layout.rects.iter().map(|r| r.iter().map(|&v| hex(v)).collect()).collect();
    if rects != expected_rects { problems.push(format!("rects {:?} != {:?}", layout.rects, expected_rects)) }
    problems
}

fn run(path: &Path, engine: &mut Option<MathText>, tally: &mut Tally, tiny: bool) {
    let text = std::fs::read_to_string(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut lines = text.lines();
    let header: Value = serde_json::from_str(lines.next().unwrap()).unwrap();
    let fonts = std::env::var("ZLC_MPL_FONTS").map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(header["math_fonts"].as_str().unwrap()));
    let engine = engine.get_or_insert_with(|| {
        let mut config = Config::new(fonts);
        if tiny {
            // Evict every few cases and clear the faces' glyphs often.
            config.cache_entries = 2;
            config.glyph_limit = 40;
        }
        MathText::new(config)
    });
    let font_file = Path::new(HELVETICA);
    for line in lines {
        let case: Value = serde_json::from_str(line).unwrap();
        tally.cases += 1;
        let s = case["s"].as_str().unwrap();
        let (size, dpi) = (float(&case["size"]), float(&case["dpi"]));
        let label = format!("{s:?} size {size} dpi {dpi}");
        let request = Request { text: s, font_file, size, dpi };
        let layout = engine.layout(&request);
        if let Some(kind) = case.get("error").and_then(Value::as_str) {
            match (&layout, kind) {
                (Err(error), "ValueError") if error.is_matplotlib_value_error() => tally.errors += 1,
                (Err(Error::MatplotlibRaises(what)), _) if what.starts_with(kind) => tally.errors += 1,
                (Err(Error::Unsupported(_)), _) if UNSUPPORTED.iter().any(|marker| s.contains(marker)) => tally.unsupported += 1,
                (Err(error), _) => tally.failures.push(format!("{label}: Matplotlib raised {kind}, the port {error}")),
                (Ok(_), _) => tally.failures.push(format!("{label}: Matplotlib raised {kind}, the port parsed it")),
            }
            continue;
        }
        let layout = match layout {
            Ok(layout) => layout,
            Err(Error::Unsupported(_)) if UNSUPPORTED.iter().any(|marker| s.contains(marker)) => {
                tally.unsupported += 1;
                continue;
            }
            Err(error) => { tally.failures.push(format!("{label}: {error}")); continue }
        };
        let problems = check_layout(&case, &layout);
        if problems.is_empty() { tally.layouts += 1 } else { tally.failures.push(format!("{label}: {}", problems.join("; "))) }
        for (key, antialiased) in [("aa", true), ("mono", false)] {
            match engine.raster(&request, antialiased) {
                Ok(raster) => check_raster(tally, &format!("{label} {key}"), &case[key], &raster),
                Err(error) => tally.failures.push(format!("{label} {key}: {error}")),
            }
        }
        match engine.metrics(&request) {
            Ok((width, height, depth)) => {
                let e = case["aa"].as_array().unwrap();
                if [width, height, depth].map(hex) != [2, 3, 4].map(|i| e[i].as_str().unwrap().to_string()) {
                    tally.failures.push(format!("{label}: metrics {width} {height} {depth}"));
                }
            }
            Err(error) => tally.failures.push(format!("{label} metrics: {error}")),
        }
    }
}

#[test]
fn matplotlib_goldens() {
    let mut paths = vec![PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/subset.jsonl"))];
    if let Ok(extra) = std::env::var("ZLC_MATHTEXT_GOLDEN") {
        paths.extend(extra.split(';').filter(|p| !p.is_empty()).map(PathBuf::from));
    }
    let mut engine = None;
    let mut failed = 0;
    for path in paths {
        let mut tally = Tally::default();
        run(&path, &mut engine, &mut tally, false);
        println!("{}: {} cases, layouts {} exact, rasters {} exact ({} with full images), {} errors as Matplotlib, {} refused",
            path.display(), tally.cases, tally.layouts, tally.rasters, tally.images, tally.errors, tally.unsupported);
        for failure in tally.failures.iter().take(60) { println!("  FAIL {failure}") }
        failed += tally.failures.len();
    }
    assert_eq!(failed, 0, "golden failures");
}

/// The committed goldens again with caches so small that layouts are evicted and
/// every face's glyphs cleared all the time: no result may depend on what is cached.
#[test]
fn goldens_with_tiny_caches() {
    let path = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/subset.jsonl"));
    let mut tally = Tally::default();
    run(&path, &mut None, &mut tally, true);
    for failure in tally.failures.iter().take(20) { println!("  FAIL {failure}") }
    assert!(tally.failures.is_empty() && tally.rasters == 2 * tally.layouts && tally.layouts > 1000, "{tally:?}");
}
