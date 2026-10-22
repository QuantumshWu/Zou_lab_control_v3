//! Units against zlc_data.units: resolution of every spelling, conversion,
//! the shortest float text, decimal text forms, prefix choice, formatting and
//! parsing — values bit for bit, text character for character.
use serde_json::Value;
use zlc_plot::decimal::Decimal;
use zlc_plot::pyfmt::repr;
use zlc_plot::units::{figure_padded, format_quantity, parse_quantity, prefix_for, Conversion, Number, Unit, DEFAULT_UNITS, PREFIXES};

fn float(value: &Value) -> f64 { f64::from_bits(value.as_u64().expect("float bits")) }
fn text(value: &Value) -> Option<String> { value.as_str().map(str::to_owned) }

fn conversion(c: &Conversion) -> Value {
    match c {
        Conversion::Scaled(f) => serde_json::json!(["scaled", f.to_bits()]),
        Conversion::Offset(o) => serde_json::json!(["offset", o.to_bits()]),
        Conversion::Decibel { reference, factor } => serde_json::json!(["decibel", reference.to_bits(), factor.to_bits()]),
        Conversion::VoltageIntoLoad { load_ohms, divisor } => serde_json::json!(["load", load_ohms.to_bits(), divisor.to_bits()]),
        Conversion::Prefixed { inner, factor } => serde_json::json!(["prefixed", conversion(inner), factor.to_bits()]),
    }
}

fn describe(unit: &Unit) -> Value {
    let (family, prefix) = DEFAULT_UNITS.family_and_prefix(unit).unwrap();
    serde_json::json!({
        "symbol": unit.symbol, "dimension": unit.dimension, "conversion": conversion(&unit.conversion),
        "prefixes": unit.prefixes, "aliases": unit.aliases, "inverse_dimension": unit.inverse_dimension, "decade": unit.decade,
        "ladder": unit.ladder().map(|p| p.symbol).collect::<Vec<_>>(), "is_base": unit.is_base(), "is_linear": unit.is_linear(),
        "coordinate_scale": unit.coordinate_scale().map(|(d, s)| serde_json::json!([d, s.to_bits()])),
        "family": [family.symbol, prefix.symbol], "base": DEFAULT_UNITS.base_for_unit(unit).ok().map(|b| b.symbol.clone()),
        "inverse": DEFAULT_UNITS.inverse_for(unit).map(|u| u.symbol.clone()),
        "choices": DEFAULT_UNITS.display_choices(unit).unwrap(),
    })
}

fn number(value: &Value) -> Number {
    match value[0].as_str().unwrap() {
        "i" => Number::Int(value[1].as_i64().map(i128::from).unwrap_or_else(|| value[1].as_u64().unwrap() as i128)),
        _ => Number::Float(float(&value[1])),
    }
}

fn prefix(symbol: &Value) -> Option<&'static zlc_plot::units::Prefix> {
    symbol.as_str().map(|s| PREFIXES.iter().find(|p| p.symbol == s).unwrap())
}

#[test]
fn units_match_the_data_layer() {
    let data: Value = serde_json::from_str(include_str!("data/units.json")).unwrap();
    for case in data["reprs"].as_array().unwrap() {
        assert_eq!(repr(float(&case[0])), case[1].as_str().unwrap());
    }
    for case in data["decimals"].as_array().unwrap() {
        let decimal = Decimal::parse(case[0].as_str().unwrap()).unwrap();
        assert_eq!(decimal.to_sci_string(), case[1].as_str().unwrap(), "str {case}");
        assert_eq!(decimal.format_fixed(), case[2].as_str().unwrap(), "format f {case}");
        assert_eq!(decimal.normalize().to_sci_string(), case[3].as_str().unwrap(), "normalize {case}");
        if let Some(adjusted) = case[4].as_i64() { assert_eq!(decimal.adjusted(), adjusted, "adjusted {case}"); }
    }
    let mut resolved = 0;
    for case in data["units"].as_array().unwrap() {
        let spelling = case[0].as_str().unwrap();
        match DEFAULT_UNITS.resolve(spelling) {
            Ok(unit) => {
                assert!(!case[1].is_null(), "{spelling:?} resolved here but not in Python");
                assert_eq!(describe(&unit), case[1], "{spelling:?}");
                resolved += 1;
            }
            Err(error) => assert!(case[1].is_null(), "{spelling:?} failed here ({error}) but resolved in Python"),
        }
    }
    assert!(resolved > 80, "{resolved} spellings resolved");
    let samples = [0.0, 1.0, -1.0, 2.5, -0.04, 1e-9, 123456.789, 1e300, -1e-300, 0.001, 273.15, -273.15, 10.0, 0.5];
    let texts = ["1", "0.1", "-2.50", "1e-7", "123456.789", "0"];
    for case in data["conversions"].as_array().unwrap() {
        let (a, b) = (DEFAULT_UNITS.resolve(case[0].as_str().unwrap()).unwrap(), DEFAULT_UNITS.resolve(case[1].as_str().unwrap()).unwrap());
        let got: Vec<u64> = samples.iter().map(|&v| a.convert_value_to(v, &b).unwrap().to_bits()).collect();
        let want: Vec<u64> = case[2].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        for (index, (g, w)) in got.iter().zip(&want).enumerate() {
            let (g, w) = (f64::from_bits(*g), f64::from_bits(*w));
            assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "{} -> {} of {}: {g} vs {w}", a.symbol, b.symbol, samples[index]);
        }
        for (t, want) in texts.iter().zip(case[3].as_array().unwrap()) {
            let got = DEFAULT_UNITS.convert_decimal(&Decimal::parse(t).unwrap(), &a, &b).ok().map(|d| d.to_sci_string());
            assert_eq!(got, text(want), "decimal {t} {} -> {}", a.symbol, b.symbol);
        }
    }
    for case in data["prefix_cases"].as_array().unwrap() {
        let unit = DEFAULT_UNITS.resolve(case[0].as_str().unwrap()).unwrap();
        let values: Vec<Number> = case[1].as_array().unwrap().iter().map(number).collect();
        assert_eq!(prefix_for(&values, &unit).symbol, case[2].as_str().unwrap(), "prefix {case}");
    }
    for case in data["formats"].as_array().unwrap() {
        let unit = DEFAULT_UNITS.resolve(case[0].as_str().unwrap()).unwrap();
        assert_eq!(format_quantity(number(&case[1]), &unit, prefix(&case[2])).ok(), text(&case[3]), "format {case}");
    }
    for case in data["parses"].as_array().unwrap() {
        let unit = DEFAULT_UNITS.resolve(case[0].as_str().unwrap()).unwrap();
        let got = parse_quantity(case[1].as_str().unwrap(), &unit, None).ok();
        match case[2].as_u64() {
            Some(bits) => {
                let got = got.unwrap_or_else(|| panic!("parse {case} failed"));
                assert!(got.to_bits() == bits || (got.is_nan() && f64::from_bits(bits).is_nan()), "parse {case}: {got}");
            }
            None => assert!(got.is_none(), "parse {case} should fail, got {got:?}"),
        }
    }
    for case in data["padded"].as_array().unwrap() {
        assert_eq!(figure_padded(case[0].as_str().unwrap(), case[1].as_str().unwrap()), case[2].as_str().unwrap());
    }
}
