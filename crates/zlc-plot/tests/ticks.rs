//! The tick ladder against zlc_plot.ticks on real Matplotlib axes: every view of
//! every recorded axis, with the scientific policy's own text measurements.
use std::collections::HashMap;
use serde_json::Value;
use zlc_plot::ticks::{compact_number, count_label, DeclaredLocator, Geometry, Measure, Scale, SmartOffsetLocator};

fn float(value: &Value) -> f64 { f64::from_bits(value.as_u64().expect("float bits")) }

struct Recorded { sizes: HashMap<(String, u64), (f64, f64)>, lines: HashMap<u64, f64> }

impl Measure for Recorded {
    fn size(&self, text: &str, size_pt: f64) -> (f64, f64) {
        *self.sizes.get(&(text.to_owned(), size_pt.to_bits())).unwrap_or_else(|| panic!("unrecorded measurement {text:?} at {size_pt}"))
    }
    fn line(&self, size_pt: f64) -> f64 { *self.lines.get(&size_pt.to_bits()).unwrap_or_else(|| panic!("unrecorded line at {size_pt}")) }
}

fn geometry(value: &Value) -> Option<Geometry> {
    (!value.is_null()).then(|| Geometry { extent: float(&value["extent"]), before: float(&value["before"]), after: float(&value["after"]),
        across: (!value["across"].is_null()).then(|| float(&value["across"])), horizontal: value["horizontal"].as_bool().unwrap(), inward: false })
}

fn texts(value: &Value) -> Vec<String> { value.as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_owned()).collect() }
fn floats(value: &Value) -> Vec<u64> { value.as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect() }

#[test]
fn tick_ladder_matches_the_scientific_policy() {
    let data: Value = serde_json::from_str(include_str!("data/ticks.json")).unwrap();
    let measure = Recorded {
        sizes: data["sizes"].as_array().unwrap().iter().map(|e| ((e[0].as_str().unwrap().to_owned(), e[1].as_u64().unwrap()),
            (float(&e[2][0]), float(&e[2][1])))).collect(),
        lines: data["lines"].as_array().unwrap().iter().map(|e| (e[0].as_u64().unwrap(), float(&e[1]))).collect(),
    };
    let (mut cases, mut steps) = (0, 0);
    for case in data["cases"].as_array().unwrap() {
        let label_pt = float(&case["label_pt"]);
        match case["type"].as_str().unwrap() {
            "smart" => {
                let scale = if case["scale"] == "log" { Scale::Log10 } else { Scale::Linear };
                let horizontal = case["axis"] == "x";
                let mut locator = SmartOffsetLocator::new(label_pt);
                for (index, step) in case["steps"].as_array().unwrap().iter().enumerate() {
                    let ticks = locator.tick_values(float(&step["vmin"]), float(&step["vmax"]), geometry(&step["geometry"]), &scale, 0, &measure);
                    let context = format!("{} step {index}", case["axis"]);
                    assert_eq!(ticks.iter().map(|t| t.to_bits()).collect::<Vec<_>>(), floats(&step["ticks"]), "ticks {context} {step}");
                    assert_eq!(locator.ladder.texts, texts(&step["texts"]), "texts {context}");
                    assert_eq!(locator.ladder.drawn_pt.to_bits(), step["drawn"].as_u64().unwrap(), "drawn {context}");
                    let settled = locator.settled.map(|(s, d)| serde_json::json!([s, d])).unwrap_or(Value::Null);
                    assert_eq!(settled, step["settled"], "settled {context}");
                    assert_eq!((locator.k, locator.m, locator.c_exp, locator.step),
                               (step["k"].as_i64().unwrap(), step["m"].as_i64().unwrap(), step["c_exp"].as_i64().unwrap(), step["step"].as_i64().unwrap()), "operands {context}");
                    assert_eq!(locator.c_int.to_string(), step["c_int"].as_str().unwrap(), "C_int {context}");
                    assert_eq!(locator.c.to_bits(), step["c"].as_u64().unwrap(), "C {context}");
                    assert_eq!(locator.n_array.iter().map(|n| n.to_string()).collect::<Vec<_>>(), texts(&step["n"]), "n {context}");
                    assert_eq!(locator.offset_text(horizontal), step["offset"].as_str().unwrap(), "offset {context}");
                    steps += 1;
                }
            }
            _ => {
                let lengths: Vec<Option<usize>> = case["lengths"].as_array().unwrap().iter().map(|v| v.as_u64().map(|n| n as usize)).collect();
                let count_text = |value: f64, length: Option<usize>| count_label(value, length);
                let compact_text = |value: f64, length: Option<usize>| compact_number(value, length);
                let text: &dyn Fn(f64, Option<usize>) -> String = if case["kind"] == "count" { &count_text } else { &compact_text };
                let span = case["span"].as_array().map(|s| (float(&s[0]), float(&s[1])));
                let mut locator = DeclaredLocator::new(case["count"].as_u64().unwrap() as usize, text, lengths,
                                                       case["zero_optional"].as_bool().unwrap(), label_pt, span);
                for step in case["steps"].as_array().unwrap() {
                    let ticks = locator.tick_values(float(&step["vmin"]), float(&step["vmax"]), geometry(&step["geometry"]), &Scale::Linear, 0, &measure);
                    assert_eq!(ticks.iter().map(|t| t.to_bits()).collect::<Vec<_>>(), floats(&step["ticks"]), "declared ticks {step}");
                    assert_eq!(locator.ladder.texts, texts(&step["texts"]), "declared texts {step}");
                    assert_eq!(locator.ladder.drawn_pt.to_bits(), step["drawn"].as_u64().unwrap(), "declared drawn");
                    steps += 1;
                }
            }
        }
        cases += 1;
    }
    assert!(cases > 400 && steps > 2000, "{cases} cases, {steps} steps");
}
