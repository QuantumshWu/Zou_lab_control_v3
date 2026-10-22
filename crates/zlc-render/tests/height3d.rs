//! The height-bar raster against zlc_plot._height3d_raster on both of its
//! engines: frames, id planes, scene floats, picks, folds, the room chrome
//! and the cage, from tests/data/height3d.json (tests/gen/height3d_cases.py).
//! Set ZLC_HEIGHT3D_GOLDEN to a directory holding full.json (the realistic
//! panel set) to run that too; ZLC_HEIGHT3D_BENCH=1 times a live panel.
use serde_json::Value;
use zlc_render::height3d::chrome::{self, ChromeLayout, DataFrame};
use zlc_render::height3d::pymath::{py_max, py_min};
use zlc_render::height3d::{self, HeightBarCamera, HeightBarRenderer, HeightBarRequest, HeightBarScene, Origin};

fn float(value: &Value) -> f64 { f64::from_bits(value.as_u64().expect("float bits")) }
fn floats(value: &Value) -> Vec<f64> { value.as_array().unwrap().iter().map(float).collect() }
fn uint(value: &Value) -> u64 { value.as_u64().unwrap() }

// ---- mirrors of the generator ------------------------------------------------
fn splitmix(seed: u64, n: usize) -> Vec<u64> {
    (1..=n as u64).map(|i| {
        let mut z = seed.wrapping_add(i.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }).collect()
}

fn uniform(seed: u64, n: usize) -> Vec<f64> {
    splitmix(seed, n).into_iter().map(|z| (z >> 11) as f64 * 2f64.powi(-53)).collect()
}

fn mixhash(raw: &[u8]) -> u64 {
    let mut total = 0u64;
    for (i, chunk) in raw.chunks(8).enumerate() {
        let mut word = [0u8; 8];
        word[..chunk.len()].copy_from_slice(chunk);
        let mut z = u64::from_le_bytes(word) ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        total = total.wrapping_add(z ^ (z >> 31));
    }
    total ^ (raw.len() as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93)
}

fn float_hash(values: &[f64]) -> u64 {
    let bytes: Vec<u8> = values.iter().flat_map(|v| if v.is_nan() { f64::NAN } else { *v }.to_le_bytes()).collect();
    mixhash(&bytes)
}

fn make_grid(g: &Value) -> (Vec<f64>, usize, usize) {
    let shape = g["shape"].as_array().unwrap();
    let (ny, nx) = (shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
    let n = nx * ny;
    let f = |key: &str| float(&g[key]);
    let mut v: Vec<f64> = match g["kind"].as_str().unwrap() {
        "uniform" => {
            let (lo, hi) = (f("lo"), f("hi"));
            uniform(uint(&g["seed"]), n).into_iter().map(|u| lo + u * (hi - lo)).collect()
        }
        "bump" => {
            let u = uniform(uint(&g["seed"]), n);
            let (amp, cx, cy, w, noise) = (f("amp"), f("cx"), f("cy"), f("w"), f("noise"));
            (0..n).map(|index| {
                let (row, column) = (index / nx, index % nx);
                let dx = (column as f64 + 0.5) / nx as f64 - cx;
                let dy = (row as f64 + 0.5) / ny as f64 - cy;
                let d2 = dx * dx + dy * dy;
                amp / (1.0 + d2 / (w * w)) + noise * (u[index] - 0.5)
            }).collect()
        }
        "constant" => vec![f("value"); n],
        "mixed" => {
            let (u, u2) = (uniform(uint(&g["seed"]), n), uniform(uint(&g["seed"]) + 1, n));
            let (small, large, shift) = (f("small"), f("large"), f("shift"));
            (0..n).map(|i| (if u2[i] < 0.5 { u[i] * small } else { u[i] * large }) - shift).collect()
        }
        other => panic!("grid kind {other}"),
    };
    if !g["nan_fraction"].is_null() && f("nan_fraction") > 0.0 {
        let fraction = f("nan_fraction");
        for (value, u) in v.iter_mut().zip(uniform(uint(&g["nan_seed"]), n)) {
            if u < fraction {
                *value = f64::NAN;
            }
        }
    }
    if let Some(specials) = g["specials"].as_array() {
        for special in specials {
            let index = special[0].as_u64().unwrap() as usize % n;
            v[index] = match special[1].as_str().unwrap() {
                "nan" => f64::NAN,
                "inf" => f64::INFINITY,
                _ => f64::NEG_INFINITY,
            };
        }
    }
    (v, ny, nx)
}

fn luts(data: &Value) -> std::collections::HashMap<String, Vec<[u8; 3]>> {
    data["luts"].as_object().unwrap().iter().map(|(name, hex)| {
        let hex = hex.as_str().unwrap();
        let bytes: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
        (name.clone(), bytes.chunks(3).map(|c| [c[0], c[1], c[2]]).collect())
    }).collect()
}

fn triple(value: &Value) -> [f64; 3] { let v = floats(value); [v[0], v[1], v[2]] }
fn pair(value: &Value) -> (f64, f64) { let v = floats(value); (v[0], v[1]) }

#[derive(Default)]
struct Tally { cases: usize, frames: usize, ids: usize, scenes: usize, picks: usize, chrome: usize, cages: usize, failures: Vec<String> }

impl Tally {
    fn check(&mut self, ok: bool, what: impl FnOnce() -> String) -> bool {
        if !ok && self.failures.len() < 40 {
            self.failures.push(what());
        }
        ok
    }
}

fn scene_floats(scene: &HeightBarScene) -> Vec<f64> {
    vec![scene.ca, scene.sa, scene.se, scene.ce, scene.scale, scene.x_low, scene.y_high, scene.z_unit, scene.value_low, scene.value_high]
}

#[allow(clippy::too_many_lines)]
fn run(data: &Value, tally: &mut Tally) {
    let luts = luts(data);
    for (index, case) in data["cases"].as_array().unwrap().iter().enumerate() {
        tally.cases += 1;
        let context = format!("case {index} grid {} {}x{} ss{}", case["grid"]["shape"], case["width"], case["height"], case["supersample"]);
        let (grid, rows, columns) = make_grid(&case["grid"]);
        assert_eq!(float_hash(&grid), uint(&case["grid_hash"]), "grid mirror drifted: {context}");
        let lut = &luts[case["cmap"].as_str().unwrap()];
        let table = height3d::colour_table(lut);
        let expected = &case["expected"];
        let (low, high) = pair(&case["limits"]);
        let zero_rgb = (!expected["zero_rgb"].is_null()).then(|| triple(&expected["zero_rgb"]));
        if let Some(zero) = zero_rgb {
            let ported = height3d::zero_rgb(lut, low, high).unwrap();
            tally.check(ported.map(f64::to_bits) == zero.map(f64::to_bits), || format!("zero_rgb {context}"));
        }
        let camera = floats(&case["camera"]);
        let request = HeightBarRequest {
            camera: HeightBarCamera::new(camera[0], camera[1], camera[2]).unwrap(),
            value_limits: (low, high),
            supersample: case["supersample"].as_u64().unwrap() as usize,
            side_shades: pair(&case["side_shades"]),
            background_rgb: triple(&case["background"]),
            zero_rgb,
            z_fraction: float(&case["z_fraction"]),
            rim_rgb: triple(&case["rim_rgb"]),
            rim_width_px: float(&case["rim_width_px"]),
            origin: if case["origin"] == "upper" { Origin::Upper } else { Origin::Lower },
            inset_px: pair(&case["inset_px"]),
            ..HeightBarRequest::new(&grid, rows, columns, &table, case["width"].as_u64().unwrap() as usize, case["height"].as_u64().unwrap() as usize)
        };
        let frame = height3d::render_height_bars(&request).unwrap();
        let scene = &frame.scene;
        if expected["reference"] != false {
            assert_eq!(expected["rgba"], expected["reference_rgba"], "engines disagree in the golden itself: {context}");
        }
        if tally.check(mixhash(&frame.rgba) == uint(&expected["rgba"]), || format!("rgba {context}")) {
            tally.frames += 1;
        } else if let Some(hex) = expected["rgba_hex"].as_str() {
            let want: Vec<u8> = (0..hex.len() / 2).map(|i| u8::from_str_radix(&hex[2 * i..2 * i + 2], 16).unwrap()).collect();
            let differing = want.iter().zip(frame.rgba.iter()).filter(|(a, b)| a != b).count();
            tally.failures.push(format!("  {differing} bytes differ"));
        }
        let id_bytes: Vec<u8> = scene.id_plane.iter().flat_map(|v| v.to_le_bytes()).collect();
        if tally.check(mixhash(&id_bytes) == uint(&expected["ids"]), || format!("ids {context}")) {
            tally.ids += 1;
        }
        let scene_ok = tally.check(scene.quadrant == expected["quadrant"].as_i64().unwrap()
            && scene.nx as u64 == uint(&expected["nx"]) && scene.ny as u64 == uint(&expected["ny"])
            && scene.flip_rows == expected["flip_rows"].as_bool().unwrap(), || format!("scene ints {context}"))
            & tally.check(scene_floats(scene).iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                == expected["floats"].as_array().unwrap().iter().map(uint).collect::<Vec<_>>(), || format!("scene floats {context}"))
            & tally.check(float_hash(&scene.top_values) == uint(&expected["top"]) && float_hash(&scene.base_values) == uint(&expected["base"]),
                || format!("top/base values {context}"));
        tally.scenes += usize::from(scene_ok);
        let mut picks_ok = true;
        for pick in expected["picks"].as_array().unwrap() {
            let got = scene.pick(float(&pick[0]), float(&pick[1]));
            let want = pick[2].as_array().map(|p| (p[0].as_i64().unwrap(), p[1].as_i64().unwrap()));
            picks_ok &= tally.check(got == want, || format!("pick {pick} got {got:?} {context}"));
        }
        for fold in expected["folds"].as_array().unwrap() {
            let v: Vec<i64> = [&fold[0], &fold[1], &fold[2], &fold[3], &fold[4][0], &fold[4][1]].iter().map(|x| x.as_i64().unwrap()).collect();
            let (a, b) = scene.fold_cell(v[0], v[1]);
            picks_ok &= tally.check((a, b) == (v[2], v[3]) && scene.unfold_cell(a, b) == (v[4], v[5]), || format!("fold {fold} {context}"));
        }
        for projection in expected["projections"].as_array().unwrap() {
            let p = floats(projection);
            let (x, y) = scene.project(p[0], p[1], p[2]);
            picks_ok &= tally.check(x.to_bits() == p[3].to_bits() && y.to_bits() == p[4].to_bits(), || format!("project {context}"));
        }
        tally.picks += usize::from(picks_ok);

        let recorded = &expected["chrome"];
        if recorded.is_null() {
            continue;
        }
        let ticks = |key: &str| -> Vec<(i64, String)> {
            recorded[key].as_array().unwrap().iter().map(|t| (t[0].as_i64().unwrap(), t[1].as_str().unwrap().to_owned())).collect()
        };
        let (x_ticks, y_ticks) = (ticks("x_ticks"), ticks("y_ticks"));
        tally.check(height3d::coordinate_tick_indices(scene.source_nx) == x_ticks.iter().map(|t| t.0).collect::<Vec<_>>()
            && height3d::coordinate_tick_indices(scene.source_ny) == y_ticks.iter().map(|t| t.0).collect::<Vec<_>>(),
            || format!("tick indices {context}"));
        let lowest = float(&recorded["lowest"]);
        let floor_value = chrome::floor_value(lowest, if low.is_finite() { low } else { 0.0 });
        tally.check(floor_value.to_bits() == uint(&recorded["floor_value"]), || format!("floor value {context}"));
        let offsets = recorded["offsets"].as_array().unwrap();
        let layout = ChromeLayout {
            z_ticks: recorded["z_ticks"].as_array().unwrap().iter().map(|t| (float(&t[0]), t[1].as_str().unwrap().to_owned())).collect(),
            x_ticks,
            y_ticks,
            x_offset: offsets[0].as_str().unwrap().to_owned(),
            y_offset: offsets[1].as_str().unwrap().to_owned(),
            tick_length_px: float(&recorded["tick_length_px"]),
            label_gap_px: float(&recorded["label_gap_px"]),
            box_w: recorded["box"][0].as_u64().unwrap() as usize,
            box_h: recorded["box"][1].as_u64().unwrap() as usize,
            floor_value: Some(floor_value),
        };
        let geometry = chrome::chrome(scene, &layout);
        let line_ok = |x: &[f64], y: &[f64], want: &Value| {
            x.len() as u64 == uint(&want[0]) && float_hash(x) == uint(&want[1]) && float_hash(y) == uint(&want[2])
        };
        let mut chrome_ok = tally.check(line_ok(&geometry.grid_x, &geometry.grid_y, &recorded["grid"]),
            || format!("grid polyline len {} want {} {context}", geometry.grid_x.len(), recorded["grid"][0]))
            & tally.check(line_ok(&geometry.line_x, &geometry.line_y, &recorded["line"]),
            || format!("axis polyline len {} want {} {context}", geometry.line_x.len(), recorded["line"][0]));
        let texts: Vec<Value> = geometry.texts.iter().map(|t| serde_json::json!([t.x.to_bits(), t.y.to_bits(), t.text, t.ha, t.va])).collect();
        chrome_ok &= tally.check(texts == *recorded["texts"].as_array().unwrap(), || format!("texts {context}"));
        chrome_ok &= tally.check(geometry.projected_axes.iter().map(|s| Value::from(*s)).collect::<Vec<_>>() == *recorded["projected_axes"].as_array().unwrap(),
            || format!("projected axes {context}"));
        tally.chrome += usize::from(chrome_ok);
        let extent = floats(&case["extent"]);
        let frame_of = DataFrame { extent: (extent[0], extent[1], extent[2], extent[3]), source_nx: scene.source_nx, source_ny: scene.source_ny };
        let mut cages_ok = true;
        for cage in expected["cages"].as_array().unwrap() {
            let cell = chrome::cell_of(&frame_of, float(&cage[0]), float(&cage[1]));
            let want = cage[2].as_array().map(|p| (p[0].as_i64().unwrap(), p[1].as_i64().unwrap()));
            cages_ok &= tally.check(cell == want, || format!("cell_of {cage} got {cell:?} {context}"));
            if let Some(cell) = cell {
                let z_low = [py_min(scene.value_low, 0.0)];
                let z_high = [py_max(scene.value_high, 0.0)];
                let edges = chrome::box_edges(scene, &[cell], &z_low, &z_high);
                let flat: Vec<f64> = edges.iter().flat_map(|e| e.iter().flatten().copied()).collect();
                cages_ok &= tally.check(edges.len() as u64 == uint(&cage[3]) && float_hash(&flat) == uint(&cage[4]), || format!("box edges {context}"));
                let (xs, ys) = chrome::cage(scene, cell);
                cages_ok &= tally.check(xs.len() as u64 == uint(&cage[5]) && float_hash(&xs) == uint(&cage[6]) && float_hash(&ys) == uint(&cage[7]),
                    || format!("cage polyline len {} want {} {context}", xs.len(), cage[5]));
                let z = grid[cell.0 as usize * columns + cell.1 as usize];
                let text = chrome::readout_text(scene, &frame_of, float(&cage[0]), float(&cage[1]), z);
                cages_ok &= tally.check(text.as_deref().ok() == cage[8].as_str(), || format!("readout {text:?} want {} {context}", cage[8]));
            }
        }
        for pick in expected["canvas_picks"].as_array().unwrap() {
            let got = chrome::pick_axis_space(scene, &frame_of, 10.25, 7.5 + scene.height as f64, float(&pick[0]), float(&pick[1]));
            let want = pick[2].as_array().map(|p| (float(&p[0]), float(&p[1])));
            cages_ok &= tally.check(got.map(|(a, b)| (a.to_bits(), b.to_bits())) == want.map(|(a, b)| (a.to_bits(), b.to_bits())),
                || format!("canvas pick {context}"));
        }
        tally.cages += usize::from(cages_ok);
    }
}

fn report(name: &str, tally: &Tally) {
    eprintln!("{name}: {} cases; frames {} ids {} scenes {} picks {} chrome {} cages {}",
        tally.cases, tally.frames, tally.ids, tally.scenes, tally.picks, tally.chrome, tally.cages);
    for failure in &tally.failures {
        eprintln!("  {failure}");
    }
    assert!(tally.failures.is_empty(), "{name}: {} mismatches", tally.failures.len());
}

#[test]
fn height_bars_match_both_python_engines() {
    let data: Value = serde_json::from_str(include_str!("data/height3d.json")).unwrap();
    assert_eq!(data["engine_disagreements"], 0);
    let mut tally = Tally::default();
    run(&data, &mut tally);
    report("committed", &tally);
}

/// `_update_height_bars_artist` end to end on real Matplotlib axes: the
/// limits, zero colour, box size, taps, rim width, floor and tick indices it
/// decides, the frame it gets back, and the chrome it hands the artists.
#[test]
fn the_artist_glue_decides_what_python_decides() {
    let data: Value = serde_json::from_str(include_str!("data/height3d.json")).unwrap();
    let luts = luts(&data);
    let mut tally = Tally::default();
    for (index, case) in data["artists"].as_array().unwrap().iter().enumerate() {
        tally.cases += 1;
        let context = format!("artist case {index}");
        let (mut grid, rows, columns) = make_grid(&case["grid"]);
        let threshold = float(&case["valid_threshold"]);
        for (value, u) in grid.iter_mut().zip(uniform(uint(&case["valid_seed"]), rows * columns)) {
            if !(u >= threshold) {
                *value = f64::NAN;
            }
        }
        tally.check(float_hash(&grid) == uint(&case["heights"]), || format!("heights {context}"));
        let lut = &luts[case["cmap"].as_str().unwrap()];
        let table = height3d::colour_table(lut);
        let data_range = (!case["data_range"].is_null()).then(|| pair(&case["data_range"]));
        let color_limits = (!case["color_limits"].is_null()).then(|| pair(&case["color_limits"]));
        let (low, high) = height3d::bar_limits(color_limits, data_range);
        tally.check((low.to_bits(), high.to_bits()) == { let v = pair(&case["value_limits"]); (v.0.to_bits(), v.1.to_bits()) },
            || format!("limits {context}"));
        let zero = height3d::zero_rgb(lut, low, high).unwrap();
        tally.check(zero.map(f64::to_bits) == triple(&case["zero_rgb"]).map(f64::to_bits), || format!("zero rgb {context}"));
        let bbox = pair(&case["bbox"]);
        let (box_w, box_h) = (height3d::box_pixels(bbox.0), height3d::box_pixels(bbox.1));
        tally.check(box_w as u64 == uint(&case["width"]) && box_h as u64 == uint(&case["height"]), || format!("box {context}"));
        let supersample = height3d::frame_supersample(float(&case["dpr"]), box_w, box_h, rows, columns);
        tally.check(supersample as u64 == uint(&case["supersample"]), || format!("supersample {supersample} {context}"));
        let rim_width = height3d::rim_width_px(0.4, float(&case["dpi"]));
        tally.check(rim_width.to_bits() == uint(&case["rim_width_px"]), || format!("rim width {context}"));
        let floor_value = chrome::floor_value(data_range.map_or(0.0, |r| r.0), low);
        tally.check(floor_value.to_bits() == uint(&case["floor_value"]), || format!("floor {context}"));
        let ticks = |key: &str| -> Vec<(i64, String)> {
            case[key].as_array().unwrap().iter().map(|t| (t[0].as_i64().unwrap(), t[1].as_str().unwrap().to_owned())).collect()
        };
        let (x_ticks, y_ticks) = (ticks("x_ticks"), ticks("y_ticks"));
        tally.check(height3d::coordinate_tick_indices(columns) == x_ticks.iter().map(|t| t.0).collect::<Vec<_>>()
            && height3d::coordinate_tick_indices(rows) == y_ticks.iter().map(|t| t.0).collect::<Vec<_>>(), || format!("tick indices {context}"));
        let extent = floats(&case["extent"]);
        let values = |indices: &[(i64, String)], start: f64, end: f64, count: usize| -> Vec<Value> {
            indices.iter().map(|(i, _)| Value::from(height3d::coordinate_tick_value(*i, start, end, count).to_bits())).collect()
        };
        let labels = case["label_values"].as_array().unwrap();
        tally.check(values(&x_ticks, extent[0], extent[1], columns) == *labels[0].as_array().unwrap()
            && values(&y_ticks, extent[3], extent[2], rows) == *labels[1].as_array().unwrap(), || format!("label values {context}"));
        let camera = floats(&case["camera"]);
        let request = HeightBarRequest {
            camera: HeightBarCamera::new(camera[0], camera[1], camera[2]).unwrap(),
            value_limits: (low, high),
            supersample,
            side_shades: pair(&case["side_shades"]),
            background_rgb: triple(&case["background"]),
            zero_rgb: Some(zero),
            z_fraction: float(&case["z_fraction"]),
            rim_rgb: triple(&case["rim_rgb"]),
            rim_width_px: rim_width,
            origin: if case["origin"] == "upper" { Origin::Upper } else { Origin::Lower },
            inset_px: pair(&case["inset_px"]),
            ..HeightBarRequest::new(&grid, rows, columns, &table, box_w, box_h)
        };
        let frame = height3d::render_height_bars(&request).unwrap();
        let ids: Vec<u8> = frame.scene.id_plane.iter().flat_map(|v| v.to_le_bytes()).collect();
        if tally.check(mixhash(&frame.rgba) == uint(&case["rgba"]) && mixhash(&ids) == uint(&case["ids"]), || format!("frame {context}")) {
            tally.frames += 1;
        }
        let offsets = case["offsets"].as_array().unwrap();
        let z_ticks = height3d::ticks::z_ticks(low, high, false);
        tally.check(z_ticks.iter().map(|(t, s)| serde_json::json!([t.to_bits(), s])).collect::<Vec<_>>() == *case["z_ticks"].as_array().unwrap(),
            || format!("z ticks {context}"));
        let layout = ChromeLayout {
            z_ticks,
            x_ticks,
            y_ticks,
            x_offset: offsets[0].as_str().unwrap().to_owned(),
            y_offset: offsets[1].as_str().unwrap().to_owned(),
            tick_length_px: float(&case["tick_length_px"]),
            label_gap_px: float(&case["label_gap_px"]),
            box_w,
            box_h,
            floor_value: Some(floor_value),
        };
        let geometry = chrome::chrome(&frame.scene, &layout);
        let line_ok = |x: &[f64], y: &[f64], want: &Value| {
            x.len() as u64 == uint(&want[0]) && float_hash(x) == uint(&want[1]) && float_hash(y) == uint(&want[2])
        };
        let texts: Vec<Value> = geometry.texts.iter().map(|t| serde_json::json!([t.x.to_bits(), t.y.to_bits(), t.text, t.ha, t.va])).collect();
        let chrome_ok = tally.check(line_ok(&geometry.grid_x, &geometry.grid_y, &case["chrome_grid"]), || format!("grid {context}"))
            & tally.check(line_ok(&geometry.line_x, &geometry.line_y, &case["chrome_line"]), || format!("axis line {context}"))
            & tally.check(texts == *case["texts"].as_array().unwrap(), || format!("texts {context}"))
            & tally.check(geometry.projected_axes.iter().map(|s| Value::from(*s)).collect::<Vec<_>>() == *case["projected_axes"].as_array().unwrap(),
                || format!("projected axes {context}"));
        tally.chrome += usize::from(chrome_ok);
    }
    report("artist glue", &tally);
    assert_eq!(tally.frames, tally.cases);
}

#[test]
fn selector_numbers_print_as_python_prints_them() {
    let data: Value = serde_json::from_str(include_str!("data/height3d.json")).unwrap();
    let selector = &data["selector"];
    let mut failures = Vec::new();
    for case in selector["precision"].as_array().unwrap() {
        let got = chrome::selector_precision(float(&case[0])).ok();
        if got != case[1].as_i64() {
            failures.push(format!("precision of {} got {got:?} want {}", float(&case[0]), case[1]));
        }
    }
    let numbers = selector["numbers"].as_array().unwrap();
    for case in numbers {
        let got = chrome::selector_number(float(&case[0]), case[1].as_i64().unwrap());
        if got != case[2].as_str().unwrap() {
            failures.push(format!("{:e} at {} got {got} want {}", float(&case[0]), case[1], case[2]));
        }
    }
    eprintln!("selector: {} spans, {} numbers, {} mismatches", selector["precision"].as_array().unwrap().len(), numbers.len(), failures.len());
    for failure in failures.iter().take(30) {
        eprintln!("  {failure}");
    }
    assert!(failures.is_empty());
}

#[test]
fn height_bars_full_panel_set() {
    let Ok(directory) = std::env::var("ZLC_HEIGHT3D_GOLDEN") else { return };
    let text = std::fs::read_to_string(std::path::Path::new(&directory).join("full.json")).unwrap();
    let data: Value = serde_json::from_str(&text).unwrap();
    let mut tally = Tally::default();
    run(&data, &mut tally);
    report("full", &tally);
}

#[test]
fn a_renderer_reuses_its_caches_without_changing_a_pixel() {
    let (grid, rows, columns) = make_grid(&serde_json::json!({"kind": "uniform", "shape": [30, 40], "seed": 5,
        "lo": 0.0f64.to_bits(), "hi": 1.0f64.to_bits()}));
    let table: Vec<[f32; 3]> = (0..256).map(|i| [i as f32 / 255.0, 0.5, 1.0 - i as f32 / 255.0]).collect();
    let mut renderer = HeightBarRenderer::new();
    let mut held = Vec::new();
    for (step, azimuth) in [10.0, 20.0, 20.0, 100.0, 100.0, 20.0, 30.0, 40.0].into_iter().enumerate() {
        let request = HeightBarRequest {
            camera: HeightBarCamera::new(azimuth, 25.0 + (step / 2) as f64 * 10.0, 1.0).unwrap(),
            input_key: Some(7),
            rim_width_px: 1.5,
            supersample: 2,
            ..HeightBarRequest::new(&grid, rows, columns, &table, 90, 70)
        };
        let cached = renderer.render(&request).unwrap();
        let fresh = height3d::render_height_bars(&request).unwrap();
        assert_eq!(cached.rgba, fresh.rgba, "step {step}");
        assert_eq!(cached.scene.id_plane, fresh.scene.id_plane, "step {step}");
        if step % 3 == 0 {
            // a frame still held is never written over by later renders
            let snapshot = (cached.rgba.as_ref().clone(), cached.scene.id_plane.as_ref().clone());
            held.push((cached, snapshot));
        }
    }
    for (frame, (rgba, ids)) in &held {
        assert_eq!(frame.rgba.as_ref(), rgba);
        assert_eq!(frame.scene.id_plane.as_ref(), ids);
    }
}

/// The live-panel timing of tests/gen/height3d_bench.py, natively.
#[test]
fn height_bars_live_panel_timing() {
    if std::env::var("ZLC_HEIGHT3D_BENCH").is_err() {
        return;
    }
    let frames: usize = std::env::var("ZLC_HEIGHT3D_BENCH").unwrap().parse().unwrap_or(30);
    let data: Value = serde_json::from_str(include_str!("data/height3d.json")).unwrap();
    let lut = &luts(&data)["viridis"];
    let table = height3d::colour_table(lut);
    let (width, height, dpr) = (3062usize, 2232usize, 2.5f64);
    for (shape, seed) in [((35usize, 35usize), 501u64), ((200, 200), 502), ((1200, 1920), 503)] {
        let (grid, rows, columns) = make_grid(&serde_json::json!({"kind": "uniform", "shape": [shape.0, shape.1], "seed": seed,
            "lo": 0.0f64.to_bits(), "hi": 1.0f64.to_bits()}));
        let supersample = height3d::frame_supersample(dpr, width, height, rows, columns);
        for mode in ["orbit", "cold"] {
            let mut renderer = HeightBarRenderer::new();
            let mut times = Vec::new();
            for frame in 0..frames + 3 {
                let request = HeightBarRequest {
                    camera: HeightBarCamera::new(55.0 + 0.37 * frame as f64, 30.0, 1.0).unwrap(),
                    supersample,
                    zero_rgb: Some(height3d::zero_rgb(lut, 0.0, 1.0).unwrap()),
                    rim_width_px: height3d::rim_width_px(0.4, 100.0 * dpr),
                    origin: Origin::Upper,
                    inset_px: (31.5 * dpr, 12.25 * dpr),
                    input_key: (mode == "orbit").then_some(1),
                    ..HeightBarRequest::new(&grid, rows, columns, &table, width, height)
                };
                let start = std::time::Instant::now();
                let rendered = renderer.render(&request).unwrap();
                let elapsed = start.elapsed().as_secs_f64() * 1e3;
                drop(rendered);
                if frame >= 3 {
                    times.push(elapsed);
                }
            }
            times.sort_by(f64::total_cmp);
            if mode == "orbit" {
                let request = HeightBarRequest {
                    supersample, rim_width_px: height3d::rim_width_px(0.4, 100.0 * dpr), origin: Origin::Upper,
                    inset_px: (31.5 * dpr, 12.25 * dpr),
                    ..HeightBarRequest::new(&grid, rows, columns, &table, width, height)
                };
                let frame = height3d::render_height_bars(&request).unwrap();
                let mut rgba = frame.rgba.as_ref().clone();
                let mut rim_times = Vec::new();
                for _ in 0..frames {
                    let start = std::time::Instant::now();
                    height3d::rims::stroke_rims(&mut rgba, &frame.scene.id_plane, width, height, [0.0; 3], 1.389);
                    rim_times.push(start.elapsed().as_secs_f64() * 1e3);
                }
                rim_times.sort_by(f64::total_cmp);
                eprintln!("  rims alone {:.3} ms", rim_times[rim_times.len() / 2]);
            }
            eprintln!("{{\"grid\": [{}, {}], \"mode\": \"{mode}\", \"median_ms\": {:.3}, \"min_ms\": {:.3}}}", shape.0, shape.1, times[times.len() / 2], times[0]);
        }
    }
}

/// New data every frame (a live panel): the planes a derive replaces lend
/// their allocations to the next one, which must still draw what a fresh
/// renderer draws -- also across quadrants, which change the folded shape.
#[test]
fn a_new_shot_reuses_planes_without_changing_a_pixel() {
    let shots: Vec<(Vec<f64>, usize, usize)> = (0..4u64).map(|seed| make_grid(&serde_json::json!({"kind": "uniform",
        "shape": [17, 29], "seed": 40 + seed, "lo": (-0.5f64).to_bits(), "hi": 1.5f64.to_bits(), "nan_fraction": 0.1f64.to_bits(),
        "nan_seed": 90 + seed})))
        .collect();
    let table: Vec<[f32; 3]> = (0..256).map(|i| [0.2, i as f32 / 255.0, 0.7]).collect();
    let mut renderer = HeightBarRenderer::new();
    for (step, azimuth) in [30.0, 30.0, 120.0, 200.0, 290.0, 290.0, 10.0, 100.0].into_iter().enumerate() {
        let (grid, rows, columns) = &shots[step % shots.len()];
        let request = HeightBarRequest {
            camera: HeightBarCamera::new(azimuth, 40.0, 1.0).unwrap(),
            value_limits: (-0.5, 1.5),
            zero_rgb: Some([0.5, 0.5, 0.5]),
            input_key: if step % 3 == 0 { None } else { Some(step as u64) },
            rim_width_px: 1.2,
            ..HeightBarRequest::new(grid, *rows, *columns, &table, 77, 59)
        };
        let reused = renderer.render(&request).unwrap();
        let fresh = height3d::render_height_bars(&request).unwrap();
        assert_eq!(reused.rgba, fresh.rgba, "step {step}");
        assert_eq!(reused.scene.id_plane, fresh.scene.id_plane, "step {step}");
        assert_eq!(reused.scene.top_values, fresh.scene.top_values, "step {step}");
    }
}

/// MaxNLocator(nbins=2) over the bar limits, both autolimit modes, and the
/// kept, labelled z ticks the artist hands the chrome.
#[test]
fn z_ticks_are_matplotlibs() {
    let data: Value = serde_json::from_str(include_str!("data/height3d.json")).unwrap();
    let cases = data["z_ticks"].as_array().unwrap();
    let mut failures = Vec::new();
    for case in cases {
        let (round, low, high) = (case[0].as_bool().unwrap(), float(&case[1]), float(&case[2]));
        let raw = height3d::ticks::max_n_locator(low, high, 2, round);
        if raw.iter().map(|v| v.to_bits()).collect::<Vec<_>>() != case[3].as_array().unwrap().iter().map(uint).collect::<Vec<_>>() {
            failures.push(format!("raw ticks {low:e}..{high:e} round {round}: got {raw:?} want {:?}", floats(&case[3])));
        }
        let kept: Vec<Value> = height3d::ticks::z_ticks(low, high, round).into_iter().map(|(t, s)| serde_json::json!([t.to_bits(), s])).collect();
        if kept != *case[4].as_array().unwrap() {
            failures.push(format!("kept ticks {low:e}..{high:e} round {round}: got {kept:?} want {}", case[4]));
        }
    }
    eprintln!("z ticks: {} ranges, {} mismatches", cases.len(), failures.len());
    for failure in failures.iter().take(20) {
        eprintln!("  {failure}");
    }
    assert!(failures.is_empty());
}
