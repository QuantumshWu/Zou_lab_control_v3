//! The fit engine's declarative and textual surface against zlc_plot:
//! catalogue, options, classifier, components, parameter units, the
//! expression grammar and fit-target resolution (golden data:
//! `tests/gen/fit_api_cases.py`); and `evaluate_fit`'s per-target contract.
use std::sync::Arc;

use serde_json::Value;
use zlc_plot::fit::classifier::{bimodal_classifier_metrics, evaluate_component};
use zlc_plot::fit::param_units::{FitParameterUnits, FitQuantityUnits};
use zlc_plot::fit::target::{fit_request_target, resolve_fit_target, ExpressionFailure, TargetFacts};
use zlc_plot::fit::*;
use zlc_plot::units::DEFAULT_UNITS;

fn float(value: &Value) -> f64 { f64::from_bits(value.as_u64().expect("float bits")) }

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

fn floats(value: &Value) -> Vec<f64> {
    if let Some(packed) = value.get("f8") {
        return base64(packed.as_str().unwrap()).chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    }
    value.as_array().unwrap().iter().map(float).collect()
}

fn same(a: f64, b: f64) -> bool { a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) }

fn relation(text: &str) -> UnitRelation {
    match text {
        "dimensionless" => UnitRelation::Dimensionless,
        "value" => UnitRelation::Value,
        "value_times_axis_0" => UnitRelation::ValueTimesAxis0,
        "axis_0" => UnitRelation::Axis0,
        "axis_1" => UnitRelation::Axis1,
        "inverse_axis_0" => UnitRelation::InverseAxis0,
        _ => UnitRelation::Radian,
    }
}

fn data() -> Value { serde_json::from_str(include_str!("data/fit_api.json")).unwrap() }

#[test]
fn catalogue_matches() {
    let data = data();
    let models = builtin_fit_models();
    let want = data["catalogue"].as_array().unwrap();
    assert_eq!(models.len(), want.len());
    for (model, want) in models.iter().zip(want) {
        assert_eq!(model.model_id, want["model_id"].as_str().unwrap());
        assert_eq!(model.display_name, want["display_name"].as_str().unwrap());
        assert_eq!(model.independent_arity as u64, want["arity"].as_u64().unwrap());
        for (p, w) in model.parameters.iter().zip(want["parameters"].as_array().unwrap()) {
            assert_eq!(p.name, w["name"].as_str().unwrap());
            assert_eq!(p.unit_relation.value(), w["relation"].as_str().unwrap(), "{}", p.name);
            assert_eq!(p.domain.value(), w["domain"].as_str().unwrap());
            assert_eq!(p.display_label.as_deref(), w["label"].as_str());
            assert_eq!(p.affine_point, w["affine"].as_bool().unwrap());
            assert_eq!(p.solver_unit_relation.value(), w["solver_relation"].as_str().unwrap());
            assert_eq!(p.symbol, w["symbol"].as_str().unwrap());
            let (lo, hi) = p.bounds();
            assert!(same(lo, float(&w["bounds"][0])) && same(hi, float(&w["bounds"][1])), "{} bounds", p.name);
        }
        assert_eq!(model.headline, want["headline"].as_str().unwrap());
        assert_eq!(model.targets.iter().map(|t| t.value()).collect::<Vec<_>>(), want["targets"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect::<Vec<_>>());
        assert_eq!(model.formula.as_deref(), want["formula"].as_str());
        if let Some(formula) = &model.formula {
            let symbols: Vec<String> = spec::formula_symbols(formula).into_iter().collect();
            let want_symbols: Vec<String> = want["formula_symbols"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
            assert_eq!(symbols, want_symbols, "{}", model.model_id);
        }
        assert_eq!(model.presentation.components.iter().map(|c| c.component_id.as_str()).collect::<Vec<_>>(), want["components"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect::<Vec<_>>());
        match (&model.presentation.ellipse_glyph, &want["ellipse"]) {
            (None, Value::Null) => {}
            (Some(g), w) => {
                assert_eq!(g.center_parameters.to_vec(), w[0].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>());
                assert_eq!(g.radius_parameters.to_vec(), w[1].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>());
            }
            _ => panic!("ellipse glyph differs"),
        }
        assert_eq!(model.coordinate_relations.iter().map(|r| r.value()).collect::<Vec<_>>(), want["coordinate_relations"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect::<Vec<_>>());
        assert_eq!(model.default_for.iter().map(|t| t.value()).collect::<Vec<_>>(), want["default_for"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect::<Vec<_>>());
        assert_eq!(model.capabilities.iter().cloned().collect::<Vec<_>>(), want["capabilities"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect::<Vec<_>>());
        assert_eq!(model.compiled_bounds, want["bounds_initializer"].as_bool().unwrap(), "{}", model.model_id);
        assert_eq!(pylane::has_candidates(model), want["candidates"].as_bool().unwrap(), "{}", model.model_id);
        match (&model.reduction, &want["reduction"]) {
            (None, Value::Null) => {}
            (Some(r), w) => {
                assert_eq!(r.nested_model_id, w["nested"].as_str().unwrap());
                let shared: Vec<(String, String)> = w["shared"].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().into(), p[1].as_str().unwrap().into())).collect();
                assert_eq!(r.shared, shared);
                for ((name, pin), w) in r.pinned.iter().zip(w["pinned"].as_array().unwrap()) {
                    assert_eq!(name, w[0].as_str().unwrap());
                    match pin {
                        spec::Pin::Shared(s) => assert_eq!(s, w[1].as_str().unwrap()),
                        spec::Pin::Value(v) => assert!(same(*v, float(&w[1]))),
                    }
                }
            }
            _ => panic!("reduction differs"),
        }
    }
    let registry = default_fit_registry();
    for (target, name) in [(FitTarget::Series, "series"), (FitTarget::Histogram, "histogram"), (FitTarget::Image, "image")] {
        let got: Vec<String> = registry.models_for(target).iter().map(|m| m.model_id.clone()).collect();
        let want: Vec<String> = data["models_for"][name].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert_eq!(got, want, "{name}");
    }
}

#[test]
fn options_match() {
    let data = data();
    for case in data["options"].as_array().unwrap() {
        let mut options = FitOptions::default();
        for (key, value) in case["kwargs"].as_object().unwrap() {
            match key.as_str() {
                "loss" => options.loss = value.as_str().unwrap().into(),
                "max_nfev" => options.max_nfev = value.as_i64().unwrap(),
                "max_point_evaluations" => options.max_point_evaluations = value.as_i64(),
                "deadline_seconds" => options.deadline_seconds = if value.is_null() { None } else { Some(float(value)) },
                "max_exact_points" => options.max_exact_points = value.as_i64(),
                "min_bic_gain" => options.min_bic_gain = if value.is_null() { None } else { Some(float(value)) },
                _ => unreachable!(),
            }
        }
        match (options.validated(), &case["error"]) {
            (Ok(options), Value::Null) => {
                for pair in case["budgets"].as_array().unwrap() {
                    assert_eq!(options.evaluation_budget(pair[0].as_u64().unwrap() as usize), pair[1].as_i64().unwrap(), "{case}");
                }
            }
            (Err(error), want) if !want.is_null() => assert_eq!(error.to_string(), want[1].as_str().unwrap(), "{case}"),
            (got, want) => panic!("{case}: {got:?} vs {want}"),
        }
    }
}

#[test]
fn classifier_and_components_match() {
    let data = data();
    let engine = FitEngine::default();
    let model = engine.registry.get("bimodal_gaussian").unwrap();
    for case in data["classifier"].as_array().unwrap() {
        let cell_data = &case["cell"];
        let coordinates = vec![floats(&cell_data["coordinates"][0])];
        let observations = floats(&cell_data["observations"]);
        let cell = FitCell::points(coordinates.clone(), observations);
        let result = engine.fit(&model, &cell, &FitRequest::default()).unwrap();
        for metric in case["metrics"].as_array().unwrap() {
            let threshold = if metric["threshold"].is_null() { None } else { Some(float(&metric["threshold"])) };
            match (bimodal_classifier_metrics(&result, threshold), metric.get("ok")) {
                (Ok((t, left, right, fidelity)), Some(want)) => {
                    match (t, &want[0]) {
                        (None, Value::Null) => {}
                        (Some(t), w) if !w.is_null() => assert!(same(t, float(w)), "threshold {t} vs {}", float(w)),
                        other => panic!("threshold presence differs: {other:?}"),
                    }
                    assert!(same(left, float(&want[1])) && same(right, float(&want[2])) && same(fidelity, float(&want[3])), "metrics");
                }
                (Err(error), None) => assert_eq!(error.to_string(), metric["error"][1].as_str().unwrap()),
                (got, want) => panic!("{got:?} vs {want:?}"),
            }
        }
        for (component, want) in ["A", "B"].iter().zip(case["components"].as_array().unwrap()) {
            let got = evaluate_component(&result.model, component, &coordinates, &result.parameter_values).unwrap();
            assert!(got.iter().zip(floats(want)).all(|(a, b)| same(*a, b)), "component {component}");
        }
    }
    let poisson = engine.registry.get("bimodal_poisson_gaussian").unwrap();
    for case in data["poisson_components"].as_array().unwrap() {
        let x = vec![floats(&case["x"])];
        let params = floats(&case["params"]);
        for component in ["A", "B"] {
            let got = evaluate_component(&poisson, component, &x, &params).unwrap();
            assert!(got.iter().zip(floats(&case[component])).all(|(a, b)| same(*a, b)), "poisson component {component}");
        }
    }
}

fn units_for(context: &Value) -> FitParameterUnits<'static> {
    let quantities: Vec<(UnitRelation, FitQuantityUnits)> = context["quantities"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(relation_name, pair)| {
            (
                relation(relation_name),
                FitQuantityUnits { canonical_unit: DEFAULT_UNITS.resolve(pair[0].as_str().unwrap()).unwrap(), display_unit: DEFAULT_UNITS.resolve(pair[1].as_str().unwrap()).unwrap() },
            )
        })
        .collect();
    FitParameterUnits {
        registry: None,
        quantities: Box::new(move |r| quantities.iter().find(|(q, _)| *q == r).map(|(_, u)| u.clone())),
        histogram: context["histogram"].as_bool().unwrap(),
        rolling_primary: context["rolling"].as_bool().unwrap(),
    }
}

#[test]
fn parameter_units_and_expressions_match() {
    let data = data();
    let registry = default_fit_registry();
    let mut failures = Vec::new();
    for case in data["units"].as_array().unwrap() {
        let units = units_for(&case["context"]);
        let context = case["context"]["name"].as_str().unwrap();
        for model_case in case["models"].as_array().unwrap() {
            let model = registry.get(model_case["model"].as_str().unwrap()).unwrap();
            for (parameter, want) in model.parameters.iter().zip(model_case["parameters"].as_array().unwrap()) {
                let tag = format!("{context}/{}/{}", model.model_id, parameter.name);
                match (units.canonical_parameter_unit(parameter), want["canonical"].get("ok")) {
                    (Ok(got), Some(w)) if got == w.as_str().unwrap() => {}
                    (Err(e), None) if e.to_string() == want["canonical"]["error"][1].as_str().unwrap() => {}
                    (got, _) => failures.push(format!("{tag} canonical {got:?} vs {}", want["canonical"])),
                }
                match (units.parameter_conversion(parameter, false, None), want["conversion"].get("symbol")) {
                    (Ok(c), Some(symbol)) => {
                        if c.symbol != symbol.as_str().unwrap() || c.crossing.value() != want["conversion"]["crossing"].as_str().unwrap() || c.canonical_symbol != want["conversion"]["canonical_symbol"].as_str().unwrap() {
                            failures.push(format!("{tag} conversion {c:?} vs {}", want["conversion"]));
                        }
                    }
                    (Err(e), None) if e.to_string() == want["conversion"]["error"][1].as_str().unwrap() => {}
                    (got, _) => failures.push(format!("{tag} conversion {got:?} vs {}", want["conversion"])),
                }
                for display in want["display"].as_array().unwrap() {
                    let value = float(&display["value"]);
                    let difference = display["difference"].as_bool().unwrap();
                    match (units.display_parameter_value(parameter, value, difference, None), display.get("ok")) {
                        (Ok((v, symbol)), Some(w)) if same(v, float(&w[0])) && symbol == w[1].as_str().unwrap() => {}
                        (Err(e), None) if e.to_string() == display["error"][1].as_str().unwrap() => {}
                        (got, _) => failures.push(format!("{tag} display {value} {difference}: {got:?} vs {display}")),
                    }
                }
                for item in want["canonical_value"].as_array().unwrap() {
                    let value = float(&item["value"]);
                    match (units.canonical_parameter_value(parameter, value), item.get("ok")) {
                        (Ok(v), Some(w)) if same(v, float(w)) => {}
                        (Err(e), None) if e.to_string() == item["error"][1].as_str().unwrap() => {}
                        (got, _) => failures.push(format!("{tag} canonical value {value}: {got:?} vs {item}")),
                    }
                }
            }
            for expression in model_case["expressions"].as_array().unwrap() {
                let text = expression["expression"].as_str().unwrap();
                let tag = format!("{context}/{} {text:?}", model.model_id);
                match (units.fit_expression_target(&model, text), expression.get("ok")) {
                    (Ok(target), Some(w)) => {
                        let pairs = |key: &str| -> Vec<(String, u64)> { w[key].as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().to_string(), p[1].as_u64().unwrap())).collect() };
                        let got_fixed: Vec<(String, u64)> = target.fixed.iter().map(|(k, v)| (k.clone(), v.to_bits())).collect();
                        let got_initial: Vec<(String, u64)> = target.initial.iter().map(|(k, v)| (k.clone(), v.to_bits())).collect();
                        if got_fixed != pairs("fixed") || got_initial != pairs("initial") {
                            failures.push(format!("{tag}: {target:?} vs {w}"));
                        }
                        match (units.fit_expression_text(&model, &target.fixed, &target.initial), expression["text"].get("ok")) {
                            (Ok(got), Some(want)) if got == want.as_str().unwrap() => {}
                            (Err(e), None) if e.to_string() == expression["text"]["error"][1].as_str().unwrap() => {}
                            (got, _) => failures.push(format!("{tag} text {got:?} vs {}", expression["text"])),
                        }
                    }
                    (Err(e), None) if e.to_string() == expression["error"][1].as_str().unwrap() => {}
                    (got, _) => failures.push(format!("{tag}: {got:?} vs {expression}")),
                }
            }
        }
    }
    for f in failures.iter().take(40) {
        eprintln!("  {f}");
    }
    assert!(failures.is_empty(), "{} unit/expression differences", failures.len());
}

#[test]
fn target_resolution_matches() {
    let data = data();
    let engine = FitEngine::default();
    let contexts: Vec<&Value> = data["units"].as_array().unwrap().iter().map(|c| &c["context"]).collect();
    let mut failures = Vec::new();
    for case in data["resolution"].as_array().unwrap() {
        let context = contexts.iter().find(|c| c["name"] == case["context"]).unwrap();
        let units = units_for(context);
        let facts = TargetFacts { facet_grid: case["spec"] == "facet" };
        let previous = case["previous_model"].as_str().map(|model| {
            let (request, _) = resolve_fit_target(&engine, &serde_json::json!({"model": model}), TargetFacts { facet_grid: false }, &units, false, None, None).unwrap();
            request.expect("previous request")
        });
        let previous_failure: Option<ExpressionFailure> = case["previous_failure"].as_array().map(|f| (f[0].as_str().unwrap().into(), f[1].as_str().unwrap().into()));
        let got = resolve_fit_target(&engine, &case["target"], facts, &units, case["live"].as_bool().unwrap(), previous.as_ref(), previous_failure.as_ref());
        match (got, case.get("error")) {
            (Ok((request, failure)), None) => {
                let target = fit_request_target(request.as_ref());
                let want_target = if case["request"].is_null() { serde_json::json!({}) } else { case["request"].clone() };
                if target != want_target {
                    failures.push(format!("{}: target {target} vs {want_target}", case["target"]));
                }
                let all_facets = request.as_ref().map(|r| r.all_facets);
                if all_facets != case["all_facets"].as_bool() {
                    failures.push(format!("{}: all_facets {all_facets:?} vs {}", case["target"], case["all_facets"]));
                }
                let want_failure = case["failure"].as_array().map(|f| (f[0].as_str().unwrap().to_string(), f[1].as_str().unwrap().to_string()));
                if failure != want_failure {
                    failures.push(format!("{}: failure {failure:?} vs {want_failure:?}", case["target"]));
                }
            }
            (Err(error), Some(want)) => {
                if error.to_string() != want[1].as_str().unwrap() {
                    failures.push(format!("{}: error {:?} vs {:?}", case["target"], error.to_string(), want[1]));
                }
            }
            (got, want) => failures.push(format!("{}: {:?} vs {want:?}", case["target"], got.map(|(r, f)| (r.map(|r| r.model.model_id.clone()), f)))),
        }
    }
    for f in failures.iter().take(40) {
        eprintln!("  {f}");
    }
    assert!(failures.is_empty(), "{} resolution differences", failures.len());
    let _ = Arc::new(());
}

/// `float(str)` and `repr(str)` as CPython computes them.
#[test]
fn python_str_semantics_match() {
    let data = data();
    let mut failures = Vec::new();
    for case in data["pystr"]["float"].as_array().unwrap() {
        let text = case[0].as_str().unwrap();
        let got = zlc_plot::pystr::float(text).map(f64::to_bits);
        if got != case[1].as_u64() {
            failures.push(format!("float({text:?}): {got:?} vs {}", case[1]));
        }
    }
    for case in data["pystr"]["repr"].as_array().unwrap() {
        let text = case[0].as_str().unwrap();
        if zlc_plot::pystr::repr(text) != case[1].as_str().unwrap() {
            failures.push(format!("repr({text:?}): {} vs {}", zlc_plot::pystr::repr(text), case[1]));
        }
    }
    assert!(failures.is_empty(), "{} differences: {:?}", failures.len(), &failures[..failures.len().min(20)]);
}

/// `evaluate_fit`'s per-target contract (orchestration only: the numbers are
/// the engine's, held to the oracle above and in `tests/fit.rs`).
#[test]
fn evaluation_keeps_failures_per_target() {
    let engine = FitEngine::default();
    let model = engine.registry.get("gaussian_offset").unwrap();
    let x: Vec<f64> = (0..41).map(|i| -2.0 + 0.1 * i as f64).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0 * (-0.5 * v * v).exp() + 0.1 + 0.01 * (v * 7.0).sin()).collect();
    let units: Vec<(String, String)> = model.parameter_names().iter().map(|n| (n.to_string(), "V".to_string())).collect();
    let request = FitRequest::default();
    let targets = vec![
        TargetCell::Ready(FitCell::points(vec![x.clone()], y.clone())),
        TargetCell::Failed("selection is empty".into()),
        TargetCell::NotRequested,
        TargetCell::Ready(FitCell::points(vec![x[..3].to_vec()], y[..3].to_vec())),
    ];
    let evaluated = evaluate_batch(&engine, &model, targets, &request, &units).unwrap();
    let solved = evaluated.results[0].as_ref().expect("solved");
    assert!(solved.parameter_units.iter().all(|u| u == "V"));
    assert_eq!(evaluated.failures[..3], [None, Some("selection is empty".to_string()), Some("fit not requested".to_string())]);
    assert!(evaluated.results[3].is_none());
    assert!(evaluated.failures[3].as_deref().is_some_and(|m| m.contains("needs more points")));
    let single = evaluate_single(&engine, &model, &FitCell::points(vec![x.clone()], y.clone()), &request, &units).unwrap();
    assert_eq!(single.parameter_values, solved.parameter_values);
    let cancel = || true;
    let cancelled = FitRequest { cancelled: Some(&cancel), ..FitRequest::default() };
    assert!(matches!(evaluate_batch(&engine, &model, vec![TargetCell::NotRequested], &cancelled, &units), Err(FitError::Cancelled)));
}
