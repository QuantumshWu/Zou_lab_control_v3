//! The zlib inflate port against CPython's `zlib.decompressobj(-15)`, fed as
//! `zipfile.ZipExtFile` feeds it: every decompress step's output, tail, eof
//! and every zlib refusal identical (`gen/inflate_cases.py`).

use base64::Engine;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use zlc_archive::inflate::Decompress;

const MIN_READ: u64 = 4096;

fn run(stream: &[u8], declared: u64, file_size: i64, reads: &[u64]) -> Vec<Value> {
    let (mut position, mut compress_left, mut left) = (0usize, declared, file_size);
    let mut decompressor = Decompress::new();
    let mut events = Vec::new();
    let mut step = 0;
    loop {
        let n = reads[step % reads.len()];
        step += 1;
        let mut data = decompressor.unconsumed_tail().to_vec();
        if n > data.len() as u64 && compress_left > 0 {
            let want = (n - data.len() as u64).max(MIN_READ).min(compress_left) as usize;
            let chunk = &stream[position.min(stream.len())..(position + want).min(stream.len())];
            position += chunk.len();
            compress_left -= chunk.len() as u64;
            if chunk.is_empty() { events.push(json!(["EOFError"])); return events; }
            data.extend_from_slice(chunk);
        }
        let mut out = match decompressor.decompress(&data, n.max(MIN_READ) as usize) {
            Ok(out) => out,
            Err(error) => { events.push(json!(["error", error.message])); return events; }
        };
        let mut eof = decompressor.eof() || (compress_left == 0 && decompressor.unconsumed_tail().is_empty());
        if eof { out.extend(decompressor.flush()); }
        out.truncate(left.max(0) as usize);
        left -= out.len() as i64;
        if left <= 0 { eof = true; }
        let digest = format!("{:x}", Sha256::digest(&out));
        events.push(json!([out.len(), &digest[..16], decompressor.unconsumed_tail().len(), decompressor.eof(), decompressor.unused_data().len(), eof]));
        if eof || step > 4000 { return events; }
    }
}

#[test]
fn inflate_steps_like_cpython_zlib() {
    let bytes = std::fs::read(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/inflate.json.gz")).unwrap();
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text).unwrap();
    let golden: Value = serde_json::from_str(&text).unwrap();
    let pool: Vec<Vec<u8>> = golden["pool"].as_array().unwrap().iter()
        .map(|item| base64::engine::general_purpose::STANDARD.decode(item.as_str().unwrap()).unwrap()).collect();
    let cases = golden["cases"].as_array().unwrap();
    let mut failures = Vec::new();
    for (index, case) in cases.iter().enumerate() {
        let mut stream = pool[case["pool"].as_u64().unwrap() as usize].clone();
        for edit in case["edits"].as_array().unwrap() { stream[edit[0].as_u64().unwrap() as usize] = edit[1].as_u64().unwrap() as u8; }
        if let Some(cut) = case["cut"].as_u64() { stream.truncate(cut as usize); }
        let extra = case["extra"].as_str().unwrap();
        stream.extend((0..extra.len()).step_by(2).map(|at| u8::from_str_radix(&extra[at..at + 2], 16).unwrap()));
        let reads: Vec<u64> = case["reads"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect();
        let events = run(&stream, case["declared"].as_u64().unwrap(), case["file_size"].as_i64().unwrap(), &reads);
        if Value::Array(events.clone()) != case["events"] {
            let at = events.iter().zip(case["events"].as_array().unwrap()).position(|(a, b)| a != b);
            failures.push(format!("case {index}: step {at:?}: {:?} vs {:?}", at.map(|i| &events[i]), at.map(|i| &case["events"][i])));
        }
    }
    assert!(failures.is_empty(), "{} of {} inflate cases differ:\n{}", failures.len(), cases.len(), failures.iter().take(30).cloned().collect::<Vec<_>>().join("\n"));
    assert!(cases.len() >= 2500);
}
