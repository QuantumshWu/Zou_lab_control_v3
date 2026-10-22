//! Python float formatting and the tick module's number spellings.
use serde_json::Value;
use zlc_plot::pyfmt::{format_e, format_g};
use zlc_plot::ticks::{compact_number, count_label};

#[test]
fn spellings_match_python() {
    let cases: Value = serde_json::from_str(include_str!("data/format.json")).unwrap();
    let mut compared = 0;
    for case in cases.as_array().unwrap() {
        let value = f64::from_bits(case["v"].as_u64().unwrap());
        for (precision, expected) in case["g"].as_object().unwrap() {
            assert_eq!(format_g(value, precision.parse().unwrap()), expected.as_str().unwrap(), "{value:e} .{precision}g");
            compared += 1;
        }
        for (precision, expected) in case["e"].as_object().unwrap() {
            assert_eq!(format_e(value, precision.parse().unwrap()), expected.as_str().unwrap(), "{value:e} .{precision}e");
            compared += 1;
        }
        for (length, expected) in case["compact"].as_object().unwrap() {
            assert_eq!(compact_number(value, length.parse().ok()), expected.as_str().unwrap(), "compact {value:e} {length}");
            compared += 1;
        }
        for (length, expected) in case["count"].as_object().unwrap() {
            assert_eq!(count_label(value, length.parse().ok()), expected.as_str().unwrap(), "count {value:e} {length}");
            compared += 1;
        }
    }
    assert!(compared > 100_000, "{compared}");
}
