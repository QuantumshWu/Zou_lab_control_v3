//! numpy reductions: pairwise float64/float32 sums, buffered casting sums and
//! means, bit for bit against numpy 2.4.
use serde_json::Value;
use zlc_plot::numeric::{mean_cast_f64, mean_f64, sum_cast_f64, sum_f32, sum_f64};

#[test]
fn reductions_match_numpy() {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/numeric.json")).unwrap();
    let data: Value = serde_json::from_str(&text).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let f64s: Vec<f64> = case["f64"].as_array().unwrap().iter().map(|v| f64::from_bits(v.as_u64().unwrap())).collect();
        let f32s: Vec<f32> = case["f32"].as_array().unwrap().iter().map(|v| f32::from_bits(v.as_u64().unwrap() as u32)).collect();
        let u8s: Vec<u8> = case["u8"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect();
        let i16s: Vec<i16> = case["i16"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i16).collect();
        let n = f64s.len();
        assert_eq!(sum_f64(&f64s).to_bits(), case["sum_f64"].as_u64().unwrap(), "sum f64 n={n}");
        assert_eq!(sum_f32(&f32s).to_bits() as u64, case["sum_f32"].as_u64().unwrap(), "sum f32 n={n}");
        assert_eq!(sum_cast_f64(&f32s).to_bits(), case["sum_f32_as_f64"].as_u64().unwrap(), "sum f32 as f64 n={n}");
        assert_eq!(sum_cast_f64(&u8s).to_bits(), case["sum_u8"].as_u64().unwrap(), "sum u8 n={n}");
        assert_eq!(sum_cast_f64(&i16s).to_bits(), case["sum_i16"].as_u64().unwrap(), "sum i16 n={n}");
        if n > 0 {
            assert_eq!(mean_f64(&f64s).to_bits(), case["mean_f64"].as_u64().unwrap(), "mean f64 n={n}");
            assert_eq!(mean_cast_f64(&u8s).to_bits(), case["mean_u8"].as_u64().unwrap(), "mean u8 n={n}");
        }
    }
    for (n, want) in [1usize, 7, 8, 20, 200].into_iter().zip(data["negative_zero"].as_array().unwrap()) {
        assert_eq!(sum_f64(&vec![-0.0; n]).to_bits(), want.as_u64().unwrap(), "sum of {n} negative zeros");
    }
}

/// Speed against numpy on one camera frame's worth of samples; written to
/// target/numeric_speed.txt (numpy's own numbers come from the Python side).
#[test]
fn reduction_speed_on_a_frame() {
    let n = 1200 * 1920;
    let f64s: Vec<f64> = (0..n).map(|i| ((i * 7919) % 1000) as f64 * 0.37).collect();
    let u8s: Vec<u8> = (0..n).map(|i| ((i * 7919) % 251) as u8).collect();
    let time = |f: &dyn Fn() -> f64| {
        let mut best = f64::INFINITY;
        let mut sink = 0.0;
        for _ in 0..30 { let start = std::time::Instant::now(); sink += f(); best = best.min(start.elapsed().as_secs_f64()); }
        assert!(sink.is_finite());
        best * 1e3
    };
    let report = format!("sum_f64 {:.3} ms\nsum_cast_f64(u8) {:.3} ms\n", time(&|| sum_f64(&f64s)), time(&|| sum_cast_f64(&u8s)));
    std::fs::write(concat!(env!("CARGO_MANIFEST_DIR"), "/../../target/numeric_speed.txt"), report).unwrap();
}
