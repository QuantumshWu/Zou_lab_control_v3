//! Repo goldens (tests/data, made by tests/gen/goldens.py from the oracle's
//! dumps): every front's draw must reproduce matplotlib's call stream call
//! for call (each call compared by the sha256 of its canonical JSON), and
//! the Scene queries matplotlib's geometry. Natively (zlc-agg, zlc-text,
//! zlc-mathtext; fonts as in pixels_dump): every front paints matplotlib's
//! full-draw pixels (tests/data/pixels.json), and every recorded
//! `MathTextParser.parse` answer is zlc-mathtext's, bit for bit.
mod support;

use std::io::Read;
use std::path::PathBuf;

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use support::data_dir;
use zlc_figure::fonts::MathText;

fn canonical_sha(v: &Value) -> String { format!("{:x}", Sha256::digest(serde_json::to_string(v).unwrap().as_bytes())) }

/// Every golden (file name, content), in name order.
fn goldens() -> Vec<(String, Value)> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(data_dir().join("goldens")).unwrap().map(|e| e.unwrap().path())
        .filter(|p| p.to_string_lossy().ends_with(".json.gz")).collect();
    paths.sort();
    paths.iter().map(|path| {
        let mut text = String::new();
        flate2::read::GzDecoder::new(std::fs::File::open(path).unwrap()).read_to_string(&mut text).unwrap();
        (path.file_name().unwrap().to_string_lossy().into_owned(), serde_json::from_str(&text).unwrap())
    }).collect()
}

#[test]
fn goldens_paint_matplotlibs_pixels() {
    zlc_figure::wire::set_blob_root(Some(support::golden_blob_root()));
    let pixels: Value = serde_json::from_str(&std::fs::read_to_string(data_dir().join("pixels.json")).unwrap()).unwrap();
    let mut math = support::native_math();
    let (mut equal, mut total, mut math_equal, mut math_total) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    for (name, golden) in goldens() {
        let figure: zlc_figure::ir::Figure = serde_json::from_value(golden["ir"].clone()).unwrap();
        let want = &pixels[&name];
        let has_math = golden["fonts"].get("mathtext").is_some_and(|m| !m.as_object().unwrap().is_empty());
        total += 1;
        math_total += usize::from(has_math);
        match support::paint(&figure, &want["canvas"], &mut support::native_fonts(), &mut math) {
            Ok((sha, _)) if sha == want["full_draw_sha256"].as_str().unwrap() => {
                equal += 1;
                math_equal += usize::from(has_math);
            }
            Ok(_) => failures.push(format!("{name}: pixels differ")),
            Err(e) => failures.push(format!("{name}: {e}")),
        }
    }
    for f in failures.iter().take(20) { eprintln!("  {f}"); }
    eprintln!("goldens: pixels identical {equal}/{total} (mathtext fronts {math_equal}/{math_total})");
    assert!(math_total > 10);
    assert_eq!(equal, total);
}

#[test]
fn goldens_mathtext_parses_match_matplotlib() {
    let mut math = support::native_math();
    let bits = |v: f64| Value::from(v.to_bits());
    let (mut equal, mut total) = (0, 0);
    let mut failures = Vec::new();
    for (name, golden) in goldens() {
        let rc: zlc_figure::ir::Rc = serde_json::from_value(golden["ir"]["rc"].clone()).unwrap();
        let Some(answers) = golden["fonts"].get("mathtext").and_then(Value::as_object) else { continue };
        for (key, want) in answers {
            total += 1;
            let key: Value = serde_json::from_str(key).unwrap();
            assert_eq!(key[4], "raster", "{name}: {key}");
            let font: zlc_figure::ir::Font = serde_json::from_value(key[2].clone()).unwrap();
            let s = key[0].as_str().unwrap();
            let got = math.parse(&rc, s, f64::from_bits(key[1].as_u64().unwrap()), &font, key[3].as_bool()).map(|p| json!({
                "ox": bits(p.ox), "oy": bits(p.oy), "width": bits(p.width), "height": bits(p.height), "descent": bits(p.descent),
                "image": {"shape": [p.image.rows(), p.image.cols()],
                          "sha": format!("{:x}", Sha256::digest(p.image.pixels()))},
            }));
            match got {
                Ok(got) if got == *want => equal += 1,
                other => failures.push(format!("{name} {s:?} aa {}: {other:?} vs {want}", key[3])),
            }
        }
    }
    for f in failures.iter().take(20) { eprintln!("  {f}"); }
    eprintln!("goldens: mathtext parses identical {equal}/{total}");
    assert!(total > 100);
    assert_eq!(equal, total);
}

#[test]
fn goldens_match_matplotlib() {
    zlc_figure::wire::set_blob_root(Some(support::golden_blob_root()));
    let paths = goldens();
    let (mut exact, mut calls, mut queries, mut queries_equal) = (0, 0, 0, 0);
    let mut failures = Vec::new();
    for (_, golden) in &paths {
        let figure: zlc_figure::ir::Figure = serde_json::from_value(golden["ir"].clone()).unwrap();
        let record = json!({"fonts": golden["fonts"]});
        let mut fonts = support::RecordedFonts::new(&record);
        let mut math = support::RecordedMath::new(&record);
        let mut rec = zlc_figure::record::Recorder::default();
        let result = zlc_figure::draw(&figure, &mut rec, &mut fonts, &mut math);
        let want: Vec<&str> = golden["calls"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
        let got: Vec<String> = rec.calls.iter().map(|c| canonical_sha(&support::call_l2(c))).collect();
        calls += want.len();
        let first_bad = want.iter().zip(&got).position(|(w, g)| *w != g.as_str());
        if result.is_ok() && got.len() == want.len() && first_bad.is_none() {
            exact += 1;
        } else {
            failures.push(format!("{}: {:?}, {} calls vs {}, first differing call {:?} ({})", golden["source"],
                                  result.err(), got.len(), want.len(), first_bad,
                                  first_bad.map(|i| golden["names"][i].to_string()).unwrap_or_default()));
        }
        // Geometry queries.
        let mut null = zlc_figure::query::NullBackend;
        let mut fonts = support::RecordedFonts::new(&record);
        let mut math = support::RecordedMath::new(&record);
        let mut scene = zlc_figure::query::Scene::new(&figure, &mut null, &mut fonts, &mut math).unwrap();
        for (i, a) in golden["queries"]["axes"].as_array().unwrap().iter().enumerate() {
            let bits = |v: &Value| -> Vec<u64> { v.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect() };
            for (name, got) in [
                ("bbox", scene.axes_bbox(i).unwrap().map(f64::to_bits).to_vec()),
                ("position", scene.axes_position(i).map(f64::to_bits).to_vec()),
                ("trans_data", scene.trans_data(i).unwrap().to6().map(f64::to_bits).to_vec()),
                ("trans_axes", scene.trans_axes(i).unwrap().to6().map(f64::to_bits).to_vec()),
            ] {
                queries += 1;
                if got == bits(&a[name]) { queries_equal += 1; } else { failures.push(format!("{} axes {i} {name}", golden["source"])); }
            }
        }
        for (key, want) in golden["queries"]["texts"].as_object().unwrap() {
            queries += 1;
            let want: Vec<u64> = want.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
            match scene.text_extent_at(key) {
                Ok(p) if p.map(f64::to_bits).to_vec() == want => queries_equal += 1,
                other => failures.push(format!("{} {key}: {other:?}", golden["source"])),
            }
        }
    }
    for f in failures.iter().take(20) { eprintln!("  {f}"); }
    eprintln!("goldens: fronts exact {exact}/{}, {calls} calls; queries {queries_equal}/{queries}", paths.len());
    assert!(paths.len() > 100);
    assert_eq!(exact, paths.len());
    assert_eq!(queries_equal, queries);
}
