//! The level-1 deflate port against CPython's zlib 1.3.1, byte for byte.
//!
//! Committed: `tests/data/deflate.json.gz` (`gen/deflate_cases.py OUT`). The
//! multi-megabyte set (`gen/deflate_cases.py OUT --full`) is checked when
//! `ZLC_DEFLATE_GOLDEN` names it.

use base64::Engine;
use sha2::{Digest, Sha256};
use zlc_archive::deflate::{zlib_compressed_len, Deflater};

fn load(path: &std::path::Path) -> serde_json::Value {
    let bytes = std::fs::read(path).unwrap_or_else(|error| panic!("{}: {error}", path.display()));
    let mut text = String::new();
    std::io::Read::read_to_string(&mut flate2::read::GzDecoder::new(&bytes[..]), &mut text).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn check(golden: &serde_json::Value) -> usize {
    let decode = |text: &serde_json::Value| base64::engine::general_purpose::STANDARD.decode(text.as_str().unwrap()).unwrap();
    let mut checked = 0;
    for case in golden["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let input = decode(&case["input"]);
        let mut deflater = Deflater::new();
        let mut produced = Vec::new();
        let mut offset = 0;
        for size in case["chunks"].as_array().unwrap() {
            let size = size.as_u64().unwrap() as usize;
            deflater.compress(&input[offset..offset + size]);
            produced.extend(deflater.take_output());
            offset += size;
        }
        produced.extend(deflater.finish());
        if let Some(expected) = case.get("output") {
            let expected = decode(expected);
            let first = produced.iter().zip(&expected).position(|(a, b)| a != b);
            assert!(produced == expected, "{name}: {} vs {} bytes, first difference at {first:?}", produced.len(), expected.len());
        }
        assert_eq!(produced.len() as u64, case["output_len"].as_u64().unwrap(), "{name}");
        assert_eq!(format!("{:x}", Sha256::digest(&produced)), case["output_sha256"].as_str().unwrap(), "{name}");
        assert_eq!(zlib_compressed_len(&input) as u64, case["zlib_len"].as_u64().unwrap(), "{name} probe length");
        checked += 1;
    }
    checked
}

#[test]
fn level_one_streams_equal_cpython_zlib() {
    let golden = load(&std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/deflate.json.gz"));
    assert_eq!(golden["zlib"], "1.3.1");
    assert!(check(&golden) >= 39);
    if let Ok(path) = std::env::var("ZLC_DEFLATE_GOLDEN") { assert!(check(&load(std::path::Path::new(&path))) > 0); }
}
