//! Authoring banks compile through one native authority, independent of UI or Python.
use serde_json::{Value, json};
use zlc_scan::authoring::{dispatch, formula_values, parse_values, range_values};

fn compile(document: Value, mode: &str) -> Result<Value, String> {
    dispatch("scan.compile", &json!({"document":document,"mode":mode}))
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
    assert_eq!(compiled["rows"][0]["links"][0]["values"], json!([7.0,9.0]));
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

#[test]
fn linked_modes_compile_at_the_parent_count_and_move_the_whole_group_outside() {
    let document = json!({"axes":[
        {"port":"pulse:param:inner","values":[7,8]},
        {"port":"pulse:param:A","values":[1,2,3],"link_enabled":true,"links":[
            {"port":"api:B","values":[10,20],"mode":"range"},
            {"port":"device:rf:frequency","values":[0],"mode":"formula","formula":"120 - x * 2"},
            {"port":"pulse:param:C","values":[0],"mode":"values","value_text":"3,2,1"}]}]});
    let result = compile(document,"compile").unwrap();
    let axes = result["axes"].as_array().unwrap();
    assert_eq!(axes.iter().map(|a|a["port"].as_str().unwrap()).collect::<Vec<_>>(),
        ["pulse:param:A","api:B","device:rf:frequency","pulse:param:C","pulse:param:inner"]);
    assert_eq!(axes[1]["values"],json!([10.0,15.0,20.0]));
    assert_eq!(axes[2]["values"],json!([118.0,116.0,114.0]));
    assert_eq!(axes[3]["coordinate_of"],"pulse:param:A");
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
