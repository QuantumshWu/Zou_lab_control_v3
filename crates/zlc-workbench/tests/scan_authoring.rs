//! Authoring banks compile through one native authority, independent of UI or Python.
use serde_json::{Value, json};
use zlc_workbench::scan::authoring::{dispatch, formula_values, parse_values, range_values};

fn compile(document: Value, mode: &str) -> Result<Value, String> {
    dispatch("scan.compile", &json!({"document":document,"mode":mode}))
}

/// `facts` with `more`'s fields beside its own.
fn merged(mut facts: Value, more: &Value) -> Value {
    for (key, value) in more.as_object().unwrap() { facts[key] = value.clone(); }
    facts
}

/// A plan compiled with its ports' domains, as the scan's owners state them.
fn on_ports(document: Value, mode: &str, ports: Value) -> Result<Value, String> {
    dispatch("scan.compile", &json!({"document":document,"mode":mode,"ports":ports}))
}

/// A bracket's repeat count (zlc-pulse's description of it: 1..2^32-1, whole, halves to even),
/// a duration on a 20 ns clock (at least one tick) and device frequencies on no grid.
fn bench() -> Value {
    json!([
        {"port":"api:bracket_count:loop","unit":"count","lo":1.0,"hi":4294967295.0,"grid":{"step":1.0,"ties":"even"}},
        {"port":"api:duration:hold","unit":"s","lo":2e-8,"hi":85.89934590,"grid":{"step":2e-8,"ties":"away"}},
        {"port":"device:rf:frequency","unit":"Hz","lo":1e6,"hi":1e9,"grid":null},
        {"port":"device:rf:carrier","unit":"Hz","lo":5e8,"hi":8e9,"grid":null},
        {"port":"device:dds:frequency","unit":"MHz","lo":1.0,"hi":1000.0,"grid":null},
    ])
}

fn values(compiled: &Value, axis: usize) -> Vec<f64> {
    compiled["axes"][axis]["values"].as_array().unwrap().iter().map(|value| value.as_f64().unwrap()).collect()
}

#[test]
fn inactive_banks_and_disabled_links_survive_without_being_evaluated() {
    let document = json!({"axes":[{"port":"pulse:param:A","values":[1,2,3],"unit":"us","mode":"range",
        "value_text":"unfinished,", "link_enabled":false,"links":[{"port":"api:B","values":[7,9],
            "mode":"formula","formula":"unfinished(","value_text":"also unfinished"}]}]});
    let compiled = compile(document.clone(), "compile").unwrap();
    assert_eq!(compiled["axes"][0]["values"], json!([1.0,2.0,3.0]));
    assert_eq!(compiled["rows"][0]["value_text"], "unfinished,");
    assert_eq!(compiled["rows"][0]["links"][0]["formula"], "unfinished(");
    let mut active = document;
    active["axes"][0]["link_enabled"] = json!(true);
    let preview = compile(active, "preview").unwrap();
    assert!(!preview["problem"].as_str().unwrap().is_empty());
    assert_eq!(preview["rows"][0]["links"][0]["formula"], "unfinished(");
}

#[test]
fn compact_ranges_and_values_are_independent_and_keep_endpoint_precision() {
    let mut document = json!({"axes":[{"port":"device:rf:frequency","values":[],"range":{"start":0.1,"stop":0.3,"count":3},
        "unit":"MHz","mode":"range","value_text":" 4, 2, 4, -0.0 "}]});
    let range = compile(document.clone(), "compile").unwrap();
    assert_eq!(range["axes"][0]["values"], json!([0.1,0.2,0.3]));
    assert_eq!(range["rows"][0]["range"], document["axes"][0]["range"]);
    assert_eq!(range["rows"][0]["values"], json!([]));
    assert_eq!(range["axis_facts"][0]["active_bounds"], json!([0.1,0.3]));
    document["axes"][0]["mode"] = json!("values");
    let values = compile(document, "compile").unwrap();
    assert_eq!(values["axes"][0]["values"], json!([4.0,2.0,4.0,-0.0]));
    assert_eq!(values["axes"][0]["values"][3].as_f64().unwrap().to_bits(), (-0.0f64).to_bits());
    assert_eq!(values["rows"][0]["range"]["count"], 3);
    assert_eq!(range_values(3.0,1.0,3).unwrap(), [3.0,2.0,1.0]);
    assert_eq!(range_values(0.1,999.0,1).unwrap(), [0.1]);
    assert!(range_values(0.0,1.0,100_001).is_err());
}

/// A linked parameter is not scanned: it follows its axis by a formula of x, one value
/// per point, a coordinate of that axis. Range and Values links are not offered.
#[test]
fn linked_formulas_follow_their_axis_and_move_the_whole_group_outside() {
    let document = json!({"axes":[
        {"port":"pulse:param:inner","values":[7,8]},
        {"port":"pulse:param:A","values":[1,2,3],"link_enabled":true,"links":[
            {"port":"api:B","mode":"formula","formula":"x + 6.5"},
            {"port":"device:rf:frequency","values":[0],"mode":"formula","formula":"120 - x * 2"}]}]});
    let result = compile(document,"compile").unwrap();
    let axes = result["axes"].as_array().unwrap();
    assert_eq!(axes.iter().map(|a|a["port"].as_str().unwrap()).collect::<Vec<_>>(),
        ["pulse:param:A","api:B","device:rf:frequency","pulse:param:inner"]);
    assert_eq!(axes[1]["values"],json!([7.5,8.5,9.5]));
    assert_eq!(axes[2]["values"],json!([118.0,116.0,114.0]));
    assert_eq!(axes[2]["coordinate_of"],"pulse:param:A");
    // The editor's facts say what each plays, its ends and count, without the whole bank.
    let facts = &result["axis_facts"][0];
    let unmoved = json!({"written":3,"moved":0,"distinct":3});
    assert_eq!(facts["played"], merged(json!({"first":1.0,"last":3.0,"count":3}), &unmoved));
    assert_eq!(facts["links"], json!([merged(json!({"port":"api:B","first":7.5,"last":9.5,"count":3,"problem":""}), &unmoved),
        merged(json!({"port":"device:rf:frequency","first":118.0,"last":114.0,"count":3,"problem":""}), &unmoved)]));
    assert_eq!(result["axis_facts"][1]["links"], json!([]));
    let broken = json!({"axes":[{"port":"pulse:param:A","values":[1,2],"link_enabled":true,"links":[
        {"port":"api:B","mode":"formula","formula":"x +"},{"port":"api:C","mode":"formula","formula":"1 / (x - 1)"}]}]});
    let preview = compile(broken,"preview").unwrap();
    let links = preview["axis_facts"][0]["links"].as_array().unwrap();
    assert!(links[0]["problem"].as_str().unwrap().contains("formula accepts"), "{}", links[0]);
    assert!(links[1]["problem"].as_str().unwrap().contains("division by zero"), "{}", links[1]);
    assert_eq!(links[1]["first"], Value::Null);
    for mode in ["range", "values"] {
        let old = json!({"axes":[{"port":"pulse:param:A","values":[1,2,3],"link_enabled":true,
            "links":[{"port":"api:B","values":[10,20],"mode":mode,"value_text":"3,2,1"}]}]});
        assert!(compile(old.clone(),"compile").unwrap_err().contains("formula of x"), "{mode}");
        let preview = compile(old,"preview").unwrap();
        assert_eq!(preview["rows"][0]["port"], "pulse:param:A", "the plan's rows are still shown");
        assert!(preview["problem"].as_str().unwrap().contains("formula of x"));
    }
    assert_eq!((result["outer_count"].as_u64(),result["inner_count"].as_u64()),(Some(3),Some(2)));
    assert_eq!(result["shape"],json!([3,2]));
    assert_eq!(result["point_count"],6);
}

#[test]
fn formulas_keep_python_arithmetic_precedence_and_bounded_numeric_grammar() {
    for (source, expected) in [("-x**2",-4.0),("(-x)**2",4.0),("2**-x",0.25),("2**3**2",512.0),
        ("+x * 3 - 1 / 2",5.5),("1_000 + 0x10 + 0o10 + 0b10",1026.0),(".5e+2 / x",25.0)] {
        assert_eq!(formula_values(source,&[2.0]).unwrap(),[expected],"{source}");
    }
    for source in ["x[0]","__import__('os')","True","x // 2","x ^ 2","(1,2)","01 + x","1e", "x +"] {
        assert!(formula_values(source,&[2.0]).is_err(),"{source}");
    }
    assert!(formula_values("2**65",&[1.0]).unwrap_err().contains("exponent exceeds"));
    assert!(formula_values("1 / (x-x)",&[1.0]).unwrap_err().contains("division by zero"));
    assert!(formula_values("(-x)**0.5",&[1.0]).is_err());
    assert!(formula_values(&"+".repeat(2049),&[1.0]).unwrap_err().contains("too long"));
    assert!(formula_values(&format!("{}x", "-".repeat(100)),&[1.0]).unwrap_err().contains("too complex"));
}

#[test]
fn values_preview_and_strict_json_reject_bad_inputs_without_losing_drafts() {
    assert_eq!(parse_values("1e-3, 1_000, -0, 3,3").unwrap(),[0.001,1000.0,-0.0,3.0,3.0]);
    for text in ["", "1,", "1,,2", "1__2", "nan", "1e999", "True"] { assert!(parse_values(text).is_err(),"{text}"); }
    let invalid = json!({"axes":[{"port":"pulse:param:A","values":[1,2],"mode":"values","value_text":"1,"}]});
    assert!(compile(invalid.clone(),"compile").is_err());
    let preview=compile(invalid,"preview").unwrap();
    assert_eq!(preview["rows"][0]["value_text"],"1,");
    assert!(preview["problem"].as_str().unwrap().contains("Values item 2"));
    assert!(compile(json!(r#"{"axes":[],"axes":[]}"#),"normalize").unwrap_err().contains("duplicate key"));
    assert!(compile(json!({"axes":[{"port":"x","values":[true]}]}),"normalize").is_err());
    assert!(compile(json!({"axes":[{"port":"manual:   ","values":[1]}]}),"compile").is_err());
}

#[test]
fn preview_reports_custom_legacy_range_without_rewriting_it() {
    let document=json!({"axes":[{"port":"pulse:param:A","values":[1,3,4,9],"value_text":"10,20"}]});
    let preview=compile(document,"preview").unwrap();
    assert_eq!(preview["axis_facts"][0]["range_uniform"],false);
    assert_eq!(preview["rows"][0]["values"],json!([1.0,3.0,4.0,9.0]));
}

#[test]
fn direct_axis_order_and_binding_share_native_policy_without_requiring_external_parents() {
    let axes=json!([
        {"port":"pulse:param:inner","values":[1,2],"unit":"us"},
        {"port":"pulse:param:paired","values":[1,2],"unit":"ms","coordinate_of":"api:outer"},
        {"port":"api:outer","values":[3,4],"unit":"us"},
        {"port":"device:rf:frequency","values":[80,81],"unit":"MHz"}]);
    assert_eq!(dispatch("scan.order",&json!({"axes":axes})).unwrap()["order"],json!([1,2,3,0]));
    // The pulse-only admission view deliberately omits the API parent.
    let axes=json!([{ "port":"pulse:param:paired","values":[1,2],"unit":"ms","coordinate_of":"api:outer"}]);
    assert_eq!(dispatch("scan.order",&json!({"axes":axes})).unwrap()["order"],json!([0]));
    let args=json!({"axes":axes,"ports":[{"port":"pulse:param:paired","unit":"s","lo":0.001,"hi":0.002}],"withheld":{}});
    assert_eq!(dispatch("scan.bind",&args).unwrap()["ports"],json!(["pulse:param:paired"]));
    let mut outside=args.clone(); outside["ports"][0]["hi"]=json!(0.0015);
    assert!(dispatch("scan.bind",&outside).unwrap_err().contains("outside the port's range"));
    let mut absent=args; absent["ports"]=json!([]);
    absent["withheld"]=json!({"pulse:param:paired":"this bracket is empty"});
    assert!(dispatch("scan.bind",&absent).unwrap_err().contains("this bracket is empty"));
}

/// A row the rules refuse (a Range past its point limit) is still previewed as written,
/// beside the others, with the reason: the editor keeps every row it was showing.
#[test]
fn a_preview_keeps_every_row_when_one_breaks_a_rule() {
    let document = json!({"axes":[
        {"port":"pulse:param:A","mode":"range","values":[1.0,2.0],"range":{"start":1.0,"stop":2.0,"count":200000},"unit":"s","value_text":""},
        {"port":"pulse:param:B","mode":"values","values":[3.0,4.0],"unit":"","value_text":"3, 4"}]});
    assert!(compile(document.clone(),"compile").is_err());
    let preview = compile(document,"preview").unwrap();
    let ports: Vec<&str> = preview["rows"].as_array().unwrap().iter().map(|row| row["port"].as_str().unwrap()).collect();
    assert_eq!(ports, ["pulse:param:A","pulse:param:B"]);
    assert_eq!(preview["rows"][0]["range"]["count"], 200000, "as written");
    assert!(preview["problem"].as_str().unwrap().contains("Range points"), "{}", preview["problem"]);
    assert!(compile(json!("not json"),"preview").is_err(), "an unreadable document is still refused");
}

/// A whole-number port plays whole numbers, however its range divides: 1 to 10 in five
/// points is 1, 3.25, 5.5, 7.75, 10 as written and 1, 3, 6, 8, 10 as played.
#[test]
fn a_whole_number_port_plays_its_range_on_whole_numbers() {
    let document = json!({"axes":[{"port":"api:bracket_count:loop","values":[],"range":{"start":1.0,"stop":10.0,"count":5},"unit":"","mode":"range","value_text":""}]});
    let compiled = on_ports(document.clone(), "preview", bench()).unwrap();
    assert_eq!(values(&compiled, 0), [1.0, 3.0, 6.0, 8.0, 10.0]);
    assert_eq!(compiled["axis_facts"][0]["played"], json!({"first":1.0,"last":10.0,"count":5,"written":5,"moved":3,"distinct":5}));
    assert_eq!(compiled["rows"][0]["range"], document["axes"][0]["range"], "the plan as written is kept");
    // Without its port's domain the plan plays what was written.
    assert_eq!(values(&compile(document, "compile").unwrap(), 0), [1.0, 3.25, 5.5, 7.75, 10.0]);
}

/// Durations play on the clock grid, in the unit the axis is written in: 0.03 us is a tick and a
/// half and goes up, as the binding layer aligns it; a value below one tick is the binding's to refuse.
#[test]
fn durations_play_on_the_clock_grid_in_the_axis_unit() {
    let document = |text: &str| json!({"axes":[{"port":"api:duration:hold","values":[],"unit":"us","mode":"values","value_text":text}]});
    let compiled = on_ports(document("0.03, 0.05, 0.029, 1.234567"), "compile", bench()).unwrap();
    assert_eq!(values(&compiled, 0), [0.04, 0.06, 0.02, 1.24]);
    let compiled = on_ports(document("0.01, 0.1"), "compile", bench()).unwrap();
    assert_eq!(values(&compiled, 0), [0.01, 0.1], "0.01 us lies outside the port: kept as written");
    let refused = dispatch("scan.bind", &json!({"axes":compiled["axes"],"ports":bench(),"withheld":{}})).unwrap_err();
    assert!(refused.contains("outside the port's range"), "{refused}");
    // A port that cannot read the axis's unit leaves its values as written too.
    let compiled = on_ports(json!({"axes":[{"port":"api:duration:hold","values":[0.03],"unit":"MHz"}]}), "compile", bench()).unwrap();
    assert_eq!(values(&compiled, 0), [0.03]);
}

/// Values typed by hand play on the port's grid; a device knob that states no grid plays them as typed.
#[test]
fn hand_typed_values_play_on_the_ports_grid() {
    let document = json!({"axes":[
        {"port":"api:bracket_count:loop","values":[],"unit":"count","mode":"values","value_text":"1.2, 2.8, 4, 6.5"},
        {"port":"device:rf:frequency","values":[],"unit":"MHz","mode":"values","value_text":"80.1234567, 81"}]});
    let compiled = on_ports(document, "preview", bench()).unwrap();
    let axis = |port: &str| compiled["axes"].as_array().unwrap().iter().position(|axis| axis["port"] == port).unwrap();
    assert_eq!(values(&compiled, axis("api:bracket_count:loop")), [1.0, 3.0, 4.0, 6.0]);
    assert_eq!(values(&compiled, axis("device:rf:frequency")), [80.1234567, 81.0]);
    assert_eq!(compiled["axis_facts"][0]["played"]["moved"], 3);
    assert_eq!(compiled["axis_facts"][1]["played"]["moved"], 0);
}

/// A linked parameter follows what its axis plays, and plays on its own port's grid.
#[test]
fn a_linked_formula_follows_the_played_axis_onto_its_own_ports_grid() {
    let document = |duration: &str| json!({"axes":[{"port":"api:bracket_count:loop","values":[1.4, 2.6, 4],"unit":"count","link_enabled":true,"links":[
        {"port":"device:rf:frequency","mode":"formula","formula":"x * 10","unit":"MHz"},
        {"port":"api:duration:hold","mode":"formula","formula":duration,"unit":"us"}]}]});
    let compiled = on_ports(document("x / 50 + 0.001"), "preview", bench()).unwrap();
    assert_eq!(compiled["problem"], "");
    assert_eq!(values(&compiled, 0), [1.0, 3.0, 4.0]);
    assert_eq!(values(&compiled, 1), [10.0, 30.0, 40.0], "x is the count played");
    assert_eq!(values(&compiled, 2), [0.02, 0.06, 0.08], "0.021, 0.061, 0.081 us on the 20 ns clock");
    let links = &compiled["axis_facts"][0]["links"];
    assert_eq!((links[0]["moved"].as_u64(), links[1]["moved"].as_u64(), links[1]["first"].as_f64()), (Some(0), Some(3), Some(0.02)));
    // Two counts the clock makes one duration (0.051 and 0.061 us are both 3 ticks): the dataset
    // cannot tell them apart, and the plan says so.
    let compiled = on_ports(document("x / 100 + 0.021"), "preview", bench()).unwrap();
    let problem = compiled["problem"].as_str().unwrap();
    assert!(problem.contains("its knob's grid made them one") && problem.contains("alternative coordinates must be unique"), "{problem}");
    // A count following a device knob (1.0, 1.4, 1.6 MHz) by x: 1.4 is a count of 1 as well.
    let following = json!({"axes":[{"port":"device:dds:frequency","values":[1.0, 1.4, 1.6],"unit":"MHz","link_enabled":true,"links":[
        {"port":"api:bracket_count:loop","mode":"formula","formula":"x","unit":"count"}]}]});
    let problem = on_ports(following, "compile", bench()).unwrap_err();
    assert!(problem.contains("\"api:bracket_count:loop\" plays 1 count at two points of its linked group, written 1 count and 1.4 count"), "{problem}");
    // A repeat as written is the plan's own, refused as before, its axis not called a linked one.
    let written = json!({"axes":[{"port":"pulse:param:A","values":[1, 1]},{"port":"pulse:param:B","values":[5, 6],"coordinate_of":"pulse:param:A"}]});
    let problem = compile(written, "compile").unwrap_err();
    assert!(problem.contains("axis \"pulse:param:A\" of a linked group repeats a coordinate") && !problem.contains("linked axis"), "{problem}");
}

/// Points a port's grid makes one are played once (the user's choice of 2026-11-23: no
/// refusal, the editor says how many were written and how many distinct values play); a
/// point written twice still plays twice, and what is linked to a merged point goes with it.
#[test]
fn points_the_grid_makes_one_are_played_once() {
    let document = json!({"axes":[{"port":"api:bracket_count:loop","values":[],"range":{"start":1.0,"stop":3.0,"count":5},"unit":"",
        "mode":"range","value_text":"","link_enabled":true,"links":[{"port":"device:rf:frequency","mode":"formula","formula":"x * 10","unit":"MHz"}]}]});
    let compiled = on_ports(document, "preview", bench()).unwrap();
    assert_eq!(compiled["problem"], "");
    assert_eq!((values(&compiled, 0), values(&compiled, 1)), (vec![1.0, 2.0, 3.0], vec![10.0, 20.0, 30.0]));
    assert_eq!((compiled["shape"].clone(), compiled["point_count"].clone()), (json!([3]), json!(3)));
    let facts = &compiled["axis_facts"][0];
    assert_eq!(facts["played"], json!({"first":1.0,"last":3.0,"count":3,"written":5,"moved":2,"distinct":3}));
    assert_eq!((facts["links"][0]["count"].as_u64(), facts["links"][0]["written"].as_u64()), (Some(3), Some(5)));
    let twice = json!({"axes":[{"port":"api:bracket_count:loop","values":[],"unit":"","mode":"values","value_text":"2, 2.4, 2, 1.6"}]});
    let compiled = on_ports(twice, "preview", bench()).unwrap();
    assert_eq!(values(&compiled, 0), [2.0, 2.0], "the 2 written twice plays twice; 2.4 and 1.6 were 2 only on the grid");
    assert_eq!(compiled["axis_facts"][0]["played"], json!({"first":2.0,"last":2.0,"count":2,"written":4,"moved":2,"distinct":1}));
}

/// What lies on a port's grid as written, and every value of a knob without one, plays as the
/// decimal written and is not moved, whatever binary noise a Range's arithmetic or a unit's
/// conversion carries: a point is moved only where the decimal written is not on the grid.
#[test]
fn a_value_on_the_grid_or_a_knob_without_one_plays_as_written() {
    let range = |port: &str, unit: &str, start: f64, stop: f64, count: usize| json!({"axes":[{"port":port,"values":[],
        "range":{"start":start,"stop":stop,"count":count},"unit":unit,"mode":"range","value_text":""}]});
    for document in [range("api:duration:hold", "us", 0.1, 1.0, 10), range("api:duration:hold", "us", 0.02, 0.2, 10),
                     range("api:bracket_count:loop", "", 1.0, 9.0, 5), range("device:rf:carrier", "GHz", 6.734, 6.934, 11),
                     range("device:dds:frequency", "kHz", 100.1, 200.3, 37),
                     // The editor's first sweeps on the 20 ns clock (scan_plan.ts firstSweep), 3 ticks apart: of 10 us,
                     // and of 10.01 us (500.5 ticks), centred on 10.02 us.
                     range("api:duration:hold", "us", 9.88, 10.12, 5), range("api:duration:hold", "us", 9.9, 10.14, 5)] {
        let compiled = on_ports(document.clone(), "preview", bench()).unwrap();
        let played = &compiled["axis_facts"][0]["played"];
        assert_eq!((played["moved"].as_u64(), played["count"].clone(), played["written"].clone()), (Some(0), document["axes"][0]["range"]["count"].clone(),
            document["axes"][0]["range"]["count"].clone()), "{document}");
        assert_eq!(values(&compiled, 0), values(&compile(document.clone(), "compile").unwrap(), 0), "as written: {document}");
    }
    // 0.05 us apart, 2.5 ticks, two of five points would move.
    let halfway = on_ports(range("api:duration:hold", "us", 9.9, 10.1, 5), "preview", bench()).unwrap();
    assert_eq!(halfway["axis_facts"][0]["played"]["moved"], 2);
    // Each the float of its decimal, bit for bit: 0.1, 0.2, ... 1 us on the clock, 6.734 ... 6.934 GHz
    // on a gridless knob -- not the Range's arithmetic, which reaches 0.30000000000000004.
    let bits = |values: Vec<f64>| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let tenths: Vec<f64> = (1..=10).map(|tenth| f64::from(tenth) / 10.0).collect();
    assert_ne!(bits(range_values(0.1, 1.0, 10).unwrap()), bits(tenths.clone()), "the Range's own arithmetic is noisy");
    let compiled = on_ports(range("api:duration:hold", "us", 0.1, 1.0, 10), "compile", bench()).unwrap();
    assert_eq!(bits(values(&compiled, 0)), bits(tenths));
    let compiled = on_ports(range("device:rf:carrier", "GHz", 6.734, 6.934, 11), "compile", bench()).unwrap();
    assert_eq!(bits(values(&compiled, 0)), bits((0..=10).map(|step| f64::from(6734 + 20 * step) / 1000.0).collect()));
}

/// A point the grid did not move plays wherever it stands; one it moved onto another's value is
/// dropped -- whether that other comes before it or after -- and a return to a value is kept.
#[test]
fn a_point_the_grid_did_not_move_always_plays() {
    for (text, played) in [("1, 1.4, 1", vec![1.0, 1.0]), ("1.4, 1, 1", vec![1.0, 1.0]), ("1.4, 1, 2, 1", vec![1.0, 2.0, 1.0]),
                           ("1.4, 1.2", vec![1.0]), ("1.4, 1.4", vec![1.0, 1.0])] {
        let document = json!({"axes":[{"port":"api:bracket_count:loop","values":[],"unit":"","mode":"values","value_text":text}]});
        assert_eq!(values(&on_ports(document, "compile", bench()).unwrap(), 0), played, "{text}");
    }
}

#[test]
fn a_ports_domain_must_be_one() {
    let document = json!({"axes":[{"port":"api:bracket_count:loop","values":[1]}]});
    for (port, problem) in [(json!({"port":"api:x","unit":"","lo":2.0,"hi":1.0}), "no usable range"),
                            (json!({"port":"api:x","unit":"","lo":0.0,"hi":1.0,"grid":{"step":0.0}}), "step must be positive"),
                            (json!({"port":"api:x","unit":"","lo":0.0,"hi":1.0,"grid":{"step":1.0,"ties":"up"}}), "away or even")] {
        assert!(on_ports(document.clone(), "compile", json!([port])).unwrap_err().contains(problem), "{problem}");
    }
}

/// A count following a manual knob by a formula -- 0.1 .. 0.5 in five points, x * 10 -- plays the
/// decimals its formula writes, 1, 2, 3, 4, 5 (the floats of the Range reach 3.0000000000000004),
/// and the binding layer takes each as the count it is; 0.57 * 100, 56.99999999999999 in floats,
/// plays 57; a device's int field plays the same whole numbers.
#[test]
fn a_linked_whole_number_plays_the_decimal_its_formula_writes() {
    use zlc_pulse::binding::{pulse_field_value, replace_pulse_field};
    use zlc_pulse::model::{Binding, Bracket, FieldRef};
    use zlc_pulse::num::number_for;
    let document = |port: &str, unit: &str| json!({"axes":[{"port":"manual:power","values":[],
        "range":{"start":0.1,"stop":0.5,"count":5},"unit":"","mode":"range","value_text":"","link_enabled":true,
        "links":[{"port":port,"mode":"formula","formula":"x*10","unit":unit}]}]});
    let bits = |values: Vec<f64>| values.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
    let whole = |last: u32| bits((1..=last).map(f64::from).collect());
    assert!(formula_values("x*10", &range_values(0.1, 0.5, 5).unwrap()).unwrap().contains(&3.0000000000000004), "the floats are noisy");
    assert!(formula_values("x*100", &[0.57]).unwrap() == [56.99999999999999], "and so is 0.57 * 100");
    let compiled = on_ports(document("api:bracket_count:loop", "count"), "preview", bench()).unwrap();
    let counts = values(&compiled, 1);
    assert_eq!((bits(counts.clone()), compiled["axis_facts"][0]["links"][0]["moved"].as_u64()), (whole(5), Some(0)));
    let hundredfold = json!({"axes":[{"port":"manual:power","values":[0.57],"link_enabled":true,
        "links":[{"port":"api:bracket_count:loop","mode":"formula","formula":"x*100","unit":"count"}]}]});
    assert_eq!(bits(values(&on_ports(hundredfold, "compile", bench()).unwrap(), 1)), bits(vec![57.0]));
    // Each count as the binding layer writes it into a pulse whose bracket count is an API parameter.
    let template = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../zlc-pulse/tests/pulses/imaging_template.json");
    let (_path, sequence) = zlc_pulse::run::read_document(&template).unwrap();
    let field = FieldRef::bracket_count("loop");
    let sequence = sequence.with(|parts| {
        parts.brackets.push(Bracket::new("loop", Some("long_before"), Some("long_after"), 2).unwrap());
        parts.bindings.push(Binding::new(field.clone(), "count", false, "api", "").unwrap());
    }).unwrap();
    for count in counts {
        let written = replace_pulse_field(&sequence, &field, number_for(count), "count", "bracket_count:loop").unwrap();
        assert_eq!(pulse_field_value(&written, &field, "count").unwrap().to_bits(), count.to_bits(), "{count}");
    }
    let mut ports = bench();
    ports.as_array_mut().unwrap().push(json!({"port":"device:counter:steps","unit":"Hz","lo":0.0,"hi":1e6,"grid":{"step":1.0}}));
    let compiled = on_ports(document("device:counter:steps", "Hz"), "compile", ports).unwrap();
    assert_eq!(bits(values(&compiled, 1)), whole(5), "a device's int field");
}
