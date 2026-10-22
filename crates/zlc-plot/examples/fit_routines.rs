//! The fit port's BLAS/LAPACK reproductions timed against the routines the
//! Python fit calls (`tests/gen/fit_routines.py`), on the same matrices,
//! and checked to the bit.
//!
//! `cargo run --release --example fit_routines -- ROUTINES.json`
use std::time::Instant;

use serde_json::Value;
use zlc_plot::fit::blas::{self, Mat};
use zlc_plot::fit::linalg;

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

fn same(a: &[f64], b: &[f64]) -> bool { a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits()) }

/// Best per-call nanoseconds over seven batches.
fn best(repeats: usize, mut f: impl FnMut()) -> f64 {
    f();
    (0..7)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..repeats {
                f();
            }
            start.elapsed().as_secs_f64() * 1e9 / repeats as f64
        })
        .fold(f64::INFINITY, f64::min)
}

fn main() {
    let path = std::env::args().nth(1).expect("ROUTINES.json");
    let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    println!("{:<34} {:>12} {:>12} {:>8}  answer", "routine (Python call)", "python ns", "rust ns", "x");
    for record in data["routines"].as_array().unwrap() {
        let (inputs, outputs) = (&record["inputs"], &record["outputs"]);
        let usize_of = |key: &str| inputs[key].as_u64().unwrap() as usize;
        let (ns, exact) = match record["routine"].as_str().unwrap() {
            "eigh" => {
                let (n, matrix) = (usize_of("n"), floats(&inputs["matrix"]));
                let (values, vectors) = linalg::eigh_lower(n, &matrix).unwrap();
                let exact = same(&values, &floats(&outputs["values"])) && same(&vectors, &floats(&outputs["vectors"]));
                (best(20000, || drop(std::hint::black_box(linalg::eigh_lower(n, std::hint::black_box(&matrix))))), exact)
            }
            "qr_r" => {
                let (rows, cols, matrix) = (usize_of("rows"), usize_of("cols"), floats(&inputs["matrix"]));
                let r = linalg::qr_r(rows, cols, &matrix).unwrap();
                let exact = same(&r, &floats(&outputs["r"]));
                (best(if rows > 1000 { 2000 } else { 20000 }, || drop(std::hint::black_box(linalg::qr_r(rows, cols, std::hint::black_box(&matrix))))), exact)
            }
            "svd" => {
                let (n, matrix) = (usize_of("n"), floats(&inputs["matrix"]));
                let (values, vt) = linalg::svd_small(n, &matrix).unwrap();
                let exact = same(&values, &floats(&outputs["values"])) && same(&vt, &floats(&outputs["vt"]));
                (best(20000, || drop(std::hint::black_box(linalg::svd_small(n, std::hint::black_box(&matrix))))), exact)
            }
            "svd_full" => {
                // The port's fallback covariance: R of the QR, then its SVD.
                let (rows, cols, matrix) = (usize_of("rows"), usize_of("cols"), floats(&inputs["matrix"]));
                let run = || {
                    let r = linalg::qr_r(rows, cols, &matrix).unwrap();
                    linalg::svd_small(cols, &r).unwrap()
                };
                let (values, vt) = run();
                let exact = same(&values, &floats(&outputs["values"])) && same(&vt, &floats(&outputs["vt"]));
                (best(20000, || drop(std::hint::black_box(run()))), exact)
            }
            "gemv" => {
                let (rows, cols) = (usize_of("rows"), usize_of("cols"));
                let view = Mat { rows: cols, cols: rows, data: floats(&inputs["image"]) };
                let vector = floats(&inputs["vector"]);
                let y = blas::dgemv_t(cols, rows, 1.0, &view, 0, 0, &vector);
                let exact = same(&y, &floats(&outputs["y"]));
                (best(if rows * cols > 100000 { 200 } else { 5000 }, || drop(std::hint::black_box(blas::dgemv_t(cols, rows, 1.0, std::hint::black_box(&view), 0, 0, &vector)))), exact)
            }
            "dot" => {
                // One vector twice when Python dotted one with itself.
                let a = floats(&inputs["a"]);
                let other = (inputs["a"] != inputs["b"]).then(|| floats(&inputs["b"]));
                let b = other.as_deref().unwrap_or(&a);
                let exact = same(&[blas::ddot(&a, b)], &floats(&outputs["dot"]));
                (best(if a.len() > 10000 { 2000 } else { 50000 }, || { std::hint::black_box(blas::ddot(std::hint::black_box(&a), b)); }), exact)
            }
            "gemm_at_b" => {
                let (k, m, n) = (usize_of("k"), usize_of("m"), usize_of("n"));
                let (a, b) = (floats(&inputs["a"]), floats(&inputs["b"]));
                let exact = same(&blas::gemm_at_b(k, m, n, &a, &b), &floats(&outputs["c"]));
                (best(500, || drop(std::hint::black_box(blas::gemm_at_b(k, m, n, std::hint::black_box(&a), &b)))), exact)
            }
            other => panic!("unknown routine {other}"),
        };
        let python = record["python_ns"].as_f64().unwrap();
        println!("{:<34} {python:>12.0} {ns:>12.0} {:>7.1}x  {}", record["name"].as_str().unwrap(), python / ns, if exact { "bit-identical" } else { "DIFFERS" });
    }
}
