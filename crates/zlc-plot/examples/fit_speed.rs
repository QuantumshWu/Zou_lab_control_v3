//! Native fit timings on `tests/gen/fit_bench.py`'s inputs, the answers
//! checked to the bit against the Python ones recorded there.
//!
//! `cargo run --release --example fit_speed -- BENCH.json (1t|pool) REPEATS`
//! prints one JSON line: per case its samples (ms) and whether every
//! answer is bit-identical.  `tests/gen/fit_interleave.py` alternates
//! rounds of this with the Python timing and tabulates both.
use std::time::Instant;

use serde_json::Value;
use zlc_plot::fit::*;

fn base64(text: &str) -> Vec<u8> {
    let decode = |c: u8| -> u32 {
        match c {
            b'A'..=b'Z' => (c - b'A') as u32,
            b'a'..=b'z' => (c - b'a' + 26) as u32,
            b'0'..=b'9' => (c - b'0' + 52) as u32,
            b'+' => 62,
            b'/' => 63,
            _ => 0,
        }
    };
    let mut out = Vec::new();
    for chunk in text.as_bytes().chunks(4) {
        let n = chunk.iter().take_while(|&&c| c != b'=').count();
        let mut word = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            word |= (if c == b'=' { 0 } else { decode(c) }) << (18 - 6 * i);
        }
        for i in 0..n.saturating_sub(1) {
            out.push((word >> (16 - 8 * i)) as u8);
        }
    }
    out
}

fn floats(value: &Value) -> Vec<f64> { base64(value["f8"].as_str().unwrap()).chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() }

/// An image's samples in their own dtype (`{"u2": ...}`).
fn samples(value: &Value) -> Samples {
    let (code, packed) = value.as_object().and_then(|o| o.iter().next()).expect("one dtype");
    let bytes = base64(packed.as_str().unwrap());
    match code.as_str() {
        "f8" => bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f64>>().into(),
        "f4" => bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>().into(),
        "u1" => bytes.to_vec().into(),
        "u2" => bytes.chunks(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u16>>().into(),
        "i2" => bytes.chunks(2).map(|c| i16::from_le_bytes(c.try_into().unwrap())).collect::<Vec<i16>>().into(),
        other => panic!("image dtype {other}"),
    }
}

fn cell(value: &Value) -> FitCell {
    let warm = (!value["warm"].is_null()).then(|| value["warm"]["positional"].as_array().unwrap().iter().map(|b| f64::from_bits(b.as_u64().unwrap())).collect());
    let mut cell = if let Some(image) = value.get("image") {
        let mask = image["mask"].as_array().map(|m| m.iter().map(|v| v.as_bool().unwrap()).collect());
        FitCell::image(RegularImageFitInput::new(floats(&image["x"]), floats(&image["y"]), samples(&image["observations"]), mask, None).unwrap())
    } else {
        FitCell::points(value["coordinates"].as_array().unwrap().iter().map(floats).collect(), floats(&value["observations"]))
    };
    cell.warm_start = warm;
    cell
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let data: Value = serde_json::from_str(&std::fs::read_to_string(&args[1]).unwrap()).unwrap();
    let pooled = args[2] == "pool";
    let repeats: usize = args[3].parse().unwrap();
    let engine = FitEngine::default();
    let single = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let (mut samples_out, mut same_out) = (serde_json::Map::new(), serde_json::Map::new());
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let model = engine.registry.get(case["model"].as_str().unwrap()).unwrap();
        let cells: Vec<FitCell> = case["cells"].as_array().unwrap().iter().map(cell).collect();
        let request = FitRequest::default();
        let run = || -> Vec<Option<FitResult>> {
            if case["kind"] == "single" {
                vec![engine.fit(&model, &cells[0], &request).ok()]
            } else {
                engine.fit_batch(&model, &cells, &request).unwrap().into_iter().map(Result::ok).collect()
            }
        };
        let go = || if pooled { run() } else { single.install(run) };
        let results = go();
        let same = results.iter().zip(case["parameters"].as_array().unwrap()).all(|(r, want)| match (r, want) {
            (Some(r), Value::Array(w)) => r.parameter_values.len() == w.len() && r.parameter_values.iter().zip(w).all(|(a, b)| a.to_bits() == b.as_u64().unwrap()),
            (None, Value::Null) => true,
            _ => false,
        });
        std::hint::black_box(go());
        let mut times = Vec::with_capacity(repeats);
        for _ in 0..repeats {
            let start = Instant::now();
            std::hint::black_box(go());
            times.push(start.elapsed().as_secs_f64() * 1000.0);
        }
        samples_out.insert(name.into(), times.into());
        same_out.insert(name.into(), same.into());
    }
    println!("{}", serde_json::json!({"samples": samples_out, "same": same_out}));
}
