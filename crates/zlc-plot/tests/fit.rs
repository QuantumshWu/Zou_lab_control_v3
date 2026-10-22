//! The fit engine against zlc_plot.fit (see `src/fit/TOLERANCES.md`): every
//! recorded case refitted natively and compared field by field -- values
//! bit for bit, texts character for character, the compiled lane's own
//! counts exactly.  Golden data: `tests/gen/fit_cases.py`.
use std::sync::Arc;

use serde_json::Value;
use sha2::{Digest, Sha256};
use zlc_plot::fit::linalg;
use zlc_plot::fit::*;
use zlc_plot::ticks::compact_number;

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
    let bytes = text.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() * 3 / 4);
    for chunk in bytes.chunks(4) {
        let n = chunk.iter().take_while(|&&c| c != b'=').count();
        let mut word = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            word |= (if c == b'=' { 0 } else { decode(c) }) << (18 - 6 * i);
        }
        let take = n.saturating_sub(1);
        for i in 0..take {
            out.push((word >> (16 - 8 * i)) as u8);
        }
    }
    out
}

/// A float array: base64 little-endian bytes (`{"f8": ...}`) or a list of bits.
fn floats(value: &Value) -> Vec<f64> {
    if let Some(packed) = value.get("f8") {
        return base64(packed.as_str().unwrap()).chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
    }
    value.as_array().expect("array").iter().map(float).collect()
}

/// An image's samples in their own dtype (`{"u2": ...}`); an older golden's
/// float64 copy of an integer plane (`storage: "integer"`) as the integer
/// type its values fit.
fn samples(value: &Value, storage: &Value) -> Samples {
    let (code, packed) = value.as_object().and_then(|o| o.iter().next()).expect("one dtype");
    let bytes = base64(packed.as_str().unwrap());
    match code.as_str() {
        "f8" if storage == "integer" => {
            let values: Vec<f64> = bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
            if values.iter().all(|&v| (0.0..=65535.0).contains(&v)) {
                values.iter().map(|&v| v as u16).collect::<Vec<u16>>().into()
            } else {
                values.iter().map(|&v| v as i16).collect::<Vec<i16>>().into()
            }
        }
        "f8" => bytes.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f64>>().into(),
        "f4" => bytes.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect::<Vec<f32>>().into(),
        "u1" => bytes.to_vec().into(),
        "u2" => bytes.chunks(2).map(|c| u16::from_le_bytes(c.try_into().unwrap())).collect::<Vec<u16>>().into(),
        "i2" => bytes.chunks(2).map(|c| i16::from_le_bytes(c.try_into().unwrap())).collect::<Vec<i16>>().into(),
        other => panic!("image dtype {other}"),
    }
}

fn ints(value: &Value) -> Vec<i64> {
    if let Some(packed) = value.get("i8") {
        return base64(packed.as_str().unwrap()).chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect();
    }
    value.as_array().expect("array").iter().map(|v| v.as_i64().unwrap()).collect()
}
fn opt_float(value: &Value) -> Option<f64> { if value.is_null() { None } else { Some(float(value)) } }

fn param_values(value: &Value) -> Option<ParamValues> {
    if value.is_null() {
        return None;
    }
    if let Some(named) = value.get("named") {
        return Some(ParamValues::Named(named.as_array().unwrap().iter().map(|p| (p[0].as_str().unwrap().to_string(), float(&p[1]))).collect()));
    }
    Some(ParamValues::Positional(floats(&value["positional"])))
}

fn bounds(value: &Value) -> Option<Bounds> {
    if value.is_null() {
        return None;
    }
    Some(Bounds(value.as_array().unwrap().iter().map(|b| (b[0].as_str().unwrap().to_string(), (opt_float(&b[1]), opt_float(&b[2])))).collect()))
}

fn options(value: &Value) -> Option<FitOptions> {
    if value.is_null() {
        return None;
    }
    Some(
        FitOptions {
            loss: value["loss"].as_str().unwrap().to_string(),
            max_nfev: value["max_nfev"].as_i64().unwrap(),
            max_point_evaluations: value["max_point_evaluations"].as_i64(),
            deadline_seconds: None,
            max_exact_points: value["max_exact_points"].as_i64(),
            min_bic_gain: opt_float(&value["min_bic_gain"]),
        }
        .validated()
        .unwrap(),
    )
}

fn cell(value: &Value) -> FitCell {
    // Warm starts are whole previous solutions.
    let warm = (!value["warm"].is_null()).then(|| floats(&value["warm"]["positional"]));
    let revision = value["revision"].as_i64().unwrap();
    if let Some(image) = value.get("image") {
        let mask = if image["mask"].is_null() { None } else { Some(image["mask"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect()) };
        let indices = if image["indices"].is_null() { None } else { Some(ints(&image["indices"])) };
        let input = RegularImageFitInput::new(floats(&image["x"]), floats(&image["y"]), samples(&image["observations"], &image["storage"]), mask, indices).unwrap();
        let mut cell = FitCell::image(input);
        cell.data_revision = revision;
        cell.warm_start = warm;
        return cell;
    }
    let coordinates: Vec<Vec<f64>> = value["coordinates"].as_array().unwrap().iter().map(floats).collect();
    let mut cell = FitCell::points(coordinates, floats(&value["observations"]));
    if !value["sigma"].is_null() {
        cell.sigma = Some(floats(&value["sigma"]).into());
    }
    if !value["indices"].is_null() {
        cell.selected_indices = Some(ints(&value["indices"]).into());
    }
    cell.data_revision = revision;
    cell.warm_start = warm;
    cell
}

fn digest_f64(values: &[f64]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn digest_i64(values: &[i64]) -> String {
    let mut hasher = Sha256::new();
    for v in values {
        hasher.update(v.to_le_bytes());
    }
    format!("{:x}", hasher.finalize())
}

fn same_bits(a: f64, b: f64) -> bool { a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan()) }

fn array_matches(expected: &Value, got: &[f64]) -> bool {
    if let Some(digest) = expected.get("sha256") {
        return digest.as_str().unwrap() == digest_f64(got);
    }
    let want = floats(expected);
    want.len() == got.len() && want.iter().zip(got).all(|(a, b)| same_bits(*a, *b))
}

fn indices_match(expected: &Value, got: &[i64]) -> bool {
    if let Some(digest) = expected.get("sha256") {
        return digest.as_str().unwrap() == digest_i64(got);
    }
    ints(expected) == got
}

#[derive(Default)]
struct Report {
    checked: usize,
    exact_cells: usize,
    failures: Vec<String>,
    worst_parameter: f64,
}

/// Deviation of a value in units of the larger of its magnitude and its
/// standard error (the TOLERANCES.md scale for iterative outputs).
fn scaled_deviation(want: f64, got: f64, error: f64) -> f64 {
    if same_bits(want, got) {
        return 0.0;
    }
    let scale = want.abs().max(if error.is_finite() { error } else { 0.0 }).max(f64::MIN_POSITIVE);
    (want - got).abs() / scale
}

fn compare_result(name: &str, local: usize, want: &Value, got: &FitResult, report: &mut Report) {
    report.checked += 1;
    let mut problems: Vec<String> = Vec::new();
    if want["model"] != got.model.model_id {
        problems.push(format!("model {} vs {}", want["model"], got.model.model_id));
    }
    let want_parameters = floats(&want["parameters"]);
    let want_errors = floats(&want["errors"]);
    let lengths = [(want_parameters.len(), got.parameter_values.len()), (want_errors.len(), got.standard_errors.len()), (floats(&want["covariance"]).len(), got.covariance.len())];
    if lengths.iter().any(|(w, g)| w != g) {
        problems.push(format!("array lengths {lengths:?}"));
    }
    for (i, (&w, &g)) in want_parameters.iter().zip(&got.parameter_values).enumerate() {
        if !same_bits(w, g) {
            let deviation = scaled_deviation(w, g, want_errors[i]);
            report.worst_parameter = report.worst_parameter.max(deviation);
            problems.push(format!("parameter {i}: {w:e} vs {g:e} ({deviation:.2e})"));
        }
    }
    for (label, key, values) in [("error", "errors", &got.standard_errors), ("covariance", "covariance", &got.covariance)] {
        let want_values = floats(&want[key]);
        let mismatched: Vec<usize> = (0..want_values.len()).filter(|&i| !same_bits(want_values[i], values[i])).collect();
        if !mismatched.is_empty() {
            let i = mismatched[0];
            problems.push(format!("{label} {} entries differ, first [{i}] {:e} vs {:e}", mismatched.len(), want_values[i], values[i]));
        }
    }
    if !array_matches(&want["fitted"], &got.fitted_values()) {
        problems.push("fitted values differ".into());
    }
    if !array_matches(&want["residuals"], &got.residuals()) {
        problems.push("residuals differ".into());
    }
    if !indices_match(&want["indices"], &got.selected_indices()) {
        problems.push("selected indices differ".into());
    }
    if want["revision"].as_u64().unwrap() != got.source_revision {
        problems.push("revision differs".into());
    }
    if want["success"].as_bool().unwrap() != got.success {
        problems.push(format!("success {} vs {}", want["success"], got.success));
    }
    if want["message"].as_str().unwrap() != got.message {
        problems.push(format!("message {:?} vs {:?}", want["message"].as_str().unwrap(), got.message));
    }
    if !same_bits(float(&want["reduced_chi_square"]), got.reduced_chi_square) {
        problems.push(format!("reduced chi-square {:e} vs {:e}", float(&want["reduced_chi_square"]), got.reduced_chi_square));
    }
    if want["covariance_valid"].as_bool().unwrap() != got.covariance_valid {
        problems.push("covariance validity differs".into());
    }
    let fixed: Vec<String> = want["fixed"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
    if fixed != got.fixed_parameter_names {
        problems.push(format!("fixed {fixed:?} vs {:?}", got.fixed_parameter_names));
    }
    if want["reduced"].as_bool().unwrap() != got.reduced {
        problems.push("reduced flag differs".into());
    }
    if !same_bits(float(&want["evidence"]), got.evidence) {
        problems.push(format!("evidence {:e} vs {:e}", float(&want["evidence"]), got.evidence));
    }
    let units: Vec<&str> = want["units"].as_array().unwrap().iter().map(|u| u.as_str().unwrap()).collect();
    if units != got.parameter_units {
        problems.push(format!("units {units:?} vs {:?}", got.parameter_units));
    }
    if want["batch_revision"].as_u64().unwrap() != got.batch_revision {
        problems.push(format!("batch revision {} vs {}", want["batch_revision"], got.batch_revision));
    }
    for (i, texts) in want["text"].as_array().unwrap().iter().enumerate() {
        let value = compact_number(got.parameter_values[i], None);
        let error = if got.standard_errors[i].is_finite() { compact_number(got.standard_errors[i], None) } else { "nan".into() };
        if texts[0].as_str().unwrap() != value || texts[1].as_str().unwrap() != error {
            problems.push(format!("painted text [{i}] {} ± {} vs {value} ± {error}", texts[0], texts[1]));
        }
    }
    // The solver row whose parameters are the result's; none for results no
    // solve produced (all fixed, the fallback lane).  A settled one-population
    // answer is the nested solve's, in the wide model's parameters: no row
    // of either solve carries those, so it is not compared.
    match (&want["diagnostics"], got.diagnostics) {
        (Value::Null, None) => {}
        (Value::Null, Some(_)) if got.reduced => {}
        (Value::Null, Some(_)) | (_, None) => problems.push(format!("diagnostics {} vs {:?}", want["diagnostics"], got.diagnostics)),
        (call, Some(d)) => {
        let counts = [("status", d.status as i64), ("nfev", d.nfev as i64), ("njev", d.njev as i64), ("iterations", d.iterations as i64), ("winner", d.winner_seed as i64)];
        for (key, value) in counts {
            if call[key].as_i64().unwrap() != value {
                problems.push(format!("{key} {} vs {value}", call[key]));
            }
        }
        if !same_bits(float(&call["cost"]), d.cost) || !same_bits(float(&call["raw_rss"]), d.raw_rss) {
            problems.push("solver cost differs".into());
        }
        }
    }
    if problems.is_empty() {
        report.exact_cells += 1;
    } else {
        report.failures.push(format!("{name}[{local}]: {}", problems.join("; ")));
    }
}

fn run_cases(data: &Value) -> Report {
    let engine = FitEngine::default();
    let mut report = Report::default();
    for case in data["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let model = engine.registry.get(case["model"].as_str().unwrap()).unwrap();
        let cells: Vec<FitCell> = case["cells"].as_array().unwrap().iter().map(cell).collect();
        let request = FitRequest { initial: param_values(&case["initial"]), bounds: bounds(&case["bounds"]), options: options(&case["options"]), cancelled: None };
        let single = case["kind"] == "single";
        let outcomes: Result<Vec<CellOutcome>, FitError> = if single { engine.fit(&model, &cells[0], &request).map(|r| vec![Ok(r)]) } else { engine.fit_batch(&model, &cells, &request) };
        match (case.get("error"), outcomes) {
            (Some(error), Err(got)) => {
                report.checked += 1;
                if error[1].as_str().unwrap() == got.to_string() {
                    report.exact_cells += 1;
                } else {
                    report.failures.push(format!("{name}: error {:?} vs {:?}", error[1].as_str().unwrap(), got.to_string()));
                }
            }
            (Some(error), Ok(_)) => report.failures.push(format!("{name}: expected error {error} but fitted")),
            (None, Err(got)) => report.failures.push(format!("{name}: unexpected error {got}")),
            (None, Ok(outcomes)) => {
                let results = case["results"].as_array().unwrap();
                let failures = case["failures"].as_array().unwrap();
                if outcomes.len() != results.len() {
                    report.failures.push(format!("{name}: {} outcomes vs {}", outcomes.len(), results.len()));
                    continue;
                }
                for (local, outcome) in outcomes.iter().enumerate() {
                    match (&results[local], &failures[local], outcome) {
                        (want, Value::Null, Ok(got)) if !want.is_null() => {
                            compare_result(name, local, want, got, &mut report);
                        }
                        (_, Value::String(message), Err(got)) => {
                            report.checked += 1;
                            if message == got {
                                report.exact_cells += 1;
                            } else {
                                report.failures.push(format!("{name}[{local}]: failure {message:?} vs {got:?}"));
                            }
                        }
                        (want, failure, got) => report.failures.push(format!("{name}[{local}]: expected {} / {failure} got {:?}", if want.is_null() { "no result" } else { "a result" }, got.as_ref().map(|r| r.message.clone()))),
                    }
                }
            }
        }
    }
    report
}

#[test]
fn fits_match_the_oracle() {
    let data: Value = serde_json::from_str(include_str!("data/fit.json")).unwrap();
    let report = run_cases(&data);
    eprintln!("fit cells checked {}, bit-exact {}, worst parameter deviation {:.3e}", report.checked, report.exact_cells, report.worst_parameter);
    for failure in report.failures.iter().take(60) {
        eprintln!("  {failure}");
    }
    assert!(report.failures.is_empty(), "{} of {} cells differ", report.failures.len(), report.checked);
}

/// A Poisson table sized past `num::ALLOCATION_LIMIT` by the data (a rate
/// near 2**60, or none at all) is refused as an allocation, not made.
#[test]
fn poisson_tables_past_the_allocation_limit_are_refused() {
    assert_eq!(models::poisson_grid(1.0e18, 2.0).unwrap_err(), num::KernelError::Allocation);
    assert_eq!(models::poisson_grid(f64::INFINITY, 2.0).unwrap_err(), num::KernelError::Allocation);
    assert!(models::poisson_grid(40.0, 2.0).is_ok());
    assert!(models::poisson_grid(3.0, 1.0e-3).is_ok());
}

/// Fits the recorded sessions asked (`fit_capture.py` on the golden corpus
/// and the live recording): a few per model and kind, committed.
#[test]
fn captured_session_fits_match_the_oracle() {
    let data: Value = serde_json::from_str(include_str!("data/fit_captured.json")).unwrap();
    let report = run_cases(&data);
    for failure in report.failures.iter().take(30) {
        eprintln!("  {failure}");
    }
    assert!(report.failures.is_empty(), "{} of {} cells differ", report.failures.len(), report.checked);
}

/// The large sets kept outside the repository (big frames, the unbinned
/// robust fit, every captured session fit): set `ZLC_FIT_GOLDEN_LARGE` to
/// one or more files (`;`-separated) the generators wrote.
#[test]
fn large_fits_match_the_oracle() {
    let Ok(paths) = std::env::var("ZLC_FIT_GOLDEN_LARGE") else { return };
    for path in paths.split(';').filter(|p| !p.is_empty()) {
        let data: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let (report, evaluations) = (run_cases(&data), run_evaluations(&data));
        eprintln!("{path}: cells checked {}, bit-exact {}; evaluation targets checked {}, bit-exact {}", report.checked, report.exact_cells, evaluations.checked, evaluations.exact_cells);
        for failure in report.failures.iter().chain(&evaluations.failures).take(30) {
            eprintln!("  {failure}");
        }
        assert!(report.failures.is_empty() && evaluations.failures.is_empty(), "{path}: {} of {} cells and {} of {} evaluation targets differ", report.failures.len(), report.checked, evaluations.failures.len(), evaluations.checked);
    }
}

/// `evaluate_fit` over the targets a session's projection prepared.
fn run_evaluations(data: &Value) -> Report {
    let engine = FitEngine::default();
    let mut report = Report::default();
    for record in data["evaluations"].as_array().map(Vec::as_slice).unwrap_or(&[]) {
        let name = record["name"].as_str().unwrap();
        let model = engine.registry.get(record["model"].as_str().unwrap()).unwrap();
        let units: Vec<(String, String)> = record["units"].as_array().unwrap().iter().map(|u| (u[0].as_str().unwrap().to_string(), u[1].as_str().unwrap().to_string())).collect();
        let request = FitRequest { initial: param_values(&record["initial"]), bounds: bounds(&record["bounds"]), options: options(&record["options"]), cancelled: None };
        let targets: Vec<TargetCell> = record["targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|t| match (t.get("cell"), t.get("failed")) {
                (Some(c), _) => TargetCell::Ready(cell(c)),
                (None, Some(message)) => TargetCell::Failed(message.as_str().unwrap().to_string()),
                (None, None) => TargetCell::NotRequested,
            })
            .collect();
        let evaluated = if record["is_batch"].as_bool().unwrap() {
            evaluate_batch(&engine, &model, targets, &request, &units)
        } else {
            let TargetCell::Ready(only) = &targets[0] else { panic!("{name}: a single evaluation has one selection") };
            evaluate_single(&engine, &model, only, &request, &units).map(|r| EvaluatedTargets { results: vec![Some(r)], failures: vec![None] })
        };
        let Ok(evaluated) = evaluated else {
            report.failures.push(format!("{name}: {:?}", evaluated.err()));
            continue;
        };
        let (results, failures) = (record["results"].as_array().unwrap(), record["failures"].as_array().unwrap());
        if evaluated.results.len() != results.len() {
            report.failures.push(format!("{name}: {} targets vs {}", evaluated.results.len(), results.len()));
            continue;
        }
        for (local, (got, failure)) in evaluated.results.iter().zip(&evaluated.failures).enumerate() {
            match (got, &results[local]) {
                (Some(got), want) if !want.is_null() => compare_result(name, local, want, got, &mut report),
                (None, Value::Null) => {
                    report.checked += 1;
                    if failure.as_deref() == failures[local].as_str() {
                        report.exact_cells += 1;
                    } else {
                        report.failures.push(format!("{name}[{local}]: failure {failure:?} vs {}", failures[local]));
                    }
                }
                (got, want) => report.failures.push(format!("{name}[{local}]: {} vs {}", got.is_some(), want)),
            }
        }
    }
    report
}

#[test]
fn captured_evaluations_match_the_oracle() {
    let data: Value = serde_json::from_str(include_str!("data/fit_captured.json")).unwrap();
    let report = run_evaluations(&data);
    eprintln!("evaluation targets checked {}, bit-exact {}", report.checked, report.exact_cells);
    for failure in report.failures.iter().take(30) {
        eprintln!("  {failure}");
    }
    assert!(report.failures.is_empty() && report.checked > 0, "{} of {} evaluation targets differ", report.failures.len(), report.checked);
}

#[test]
fn libm_matches_numba() {
    let data: Value = serde_json::from_str(include_str!("data/fit.json")).unwrap();
    let libm = &data["libm"];
    let xs = floats(&libm["x"]);
    let out: Vec<Vec<f64>> = libm["out"].as_array().unwrap().iter().map(floats).collect();
    let erf = floats(&libm["erf"]);
    let mut bad = 0;
    for (i, &v) in xs.iter().enumerate() {
        let got = [v.exp(), (v.abs() + 1e-300).ln(), v.abs().ln_1p(), v.exp_m1(), v.sin(), v.cos(), v.atan(), zlc_plot::fit::num::lgamma(v.abs() + 1.0), v.hypot(0.7)];
        for (k, g) in got.iter().enumerate() {
            if !same_bits(*g, out[k][i]) {
                bad += 1;
            }
        }
        if !same_bits(zlc_plot::fit::num::erf(v), erf[i]) {
            bad += 1;
        }
    }
    assert_eq!(bad, 0, "libm differences");
}

#[test]
fn lapack_matches_numba_and_numpy() {
    let data: Value = serde_json::from_str(include_str!("data/fit.json")).unwrap();
    let mut bad = Vec::new();
    for (index, m) in data["matrices"].as_array().unwrap().iter().enumerate() {
        let n = m["n"].as_u64().unwrap() as usize;
        let (w, v) = linalg::eigh_lower(n, &floats(&m["a"])).unwrap();
        if !(w.iter().zip(floats(&m["w"])).all(|(a, b)| same_bits(*a, b)) && v.iter().zip(floats(&m["v"])).all(|(a, b)| same_bits(*a, b))) {
            bad.push(format!("eigh {index} (n={n})"));
        }
        if !(w.iter().zip(floats(&m["np_w"])).all(|(a, b)| same_bits(*a, b)) && v.iter().zip(floats(&m["np_v"])).all(|(a, b)| same_bits(*a, b))) {
            bad.push(format!("numpy eigh {index} (n={n})"));
        }
        let (s, vt) = linalg::svd_small(n, &floats(&m["t"])).unwrap();
        if !(s.iter().zip(floats(&m["s"])).all(|(a, b)| same_bits(*a, b)) && vt.iter().zip(floats(&m["vt"])).all(|(a, b)| same_bits(*a, b))) {
            bad.push(format!("svd {index} (n={n})"));
        }
    }
    for (index, d) in data["dots"].as_array().unwrap().iter().enumerate() {
        if !same_bits(zlc_plot::fit::blas::ddot(&floats(&d["a"]), &floats(&d["b"])), float(&d["dot"])) {
            bad.push(format!("ddot {index}"));
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// `tests/gen/fit_blas_probes.py`'s exact rational inputs.
fn formula(rows: usize, cols: usize, salt: i64) -> Vec<f64> {
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows as i64 {
        for c in 0..cols as i64 {
            out.push((((r * 7919 + c * 104729 + salt) % 2001) - 1000) as f64 / 997.0);
        }
    }
    out
}

#[test]
fn gemm_matches_openblas_at_its_seams() {
    let data: Value = serde_json::from_str(include_str!("data/fit_blas.json")).unwrap();
    let mut bad = Vec::new();
    for probe in data["gemm"].as_array().unwrap() {
        let (k, side) = (probe["k"].as_u64().unwrap() as usize, probe["side"].as_u64().unwrap() as usize);
        let a = formula(k, side, 1);
        let (b0, w) = (formula(k, side, 2), formula(k, 1, 3));
        let b: Vec<f64> = b0.iter().enumerate().map(|(i, v)| v * w[i / side]).collect();
        let got = zlc_plot::fit::blas::gemm_at_b(k, side, side, &a, &b);
        let want: Vec<u64> = probe["c"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let got: Vec<u64> = got.iter().map(|v| v.to_bits()).collect();
        if got != want {
            bad.push(format!("gemm side {side} k {k}"));
        }
    }
    assert!(bad.is_empty(), "{} probes differ: {bad:?}", bad.len());
}

#[test]
fn dnrm2_matches_openblas_across_the_exponent_range() {
    let data: Value = serde_json::from_str(include_str!("data/fit_blas.json")).unwrap();
    let mut bad = Vec::new();
    for (index, probe) in data["nrm2"].as_array().unwrap().iter().enumerate() {
        let x: Vec<f64> = probe["x"].as_array().unwrap().iter().map(float).collect();
        let got = zlc_plot::fit::blas::dnrm2(&x);
        if got.to_bits() != probe["nrm2"].as_u64().unwrap() {
            bad.push(format!("dnrm2 {index}: {:#018x} vs {:#018x}", got.to_bits(), probe["nrm2"].as_u64().unwrap()));
        }
    }
    assert!(bad.is_empty(), "{} probes differ: {bad:?}", bad.len());
}

/// The image passes that read a plane in its own storage, against the
/// plain definitions they replace (`dgemv_t` on the centred float64
/// plane, held to NumPy above; one residual buffer per stripe and `ddot`;
/// one sequential sum per stripe), across the kernel's chunk and tail seams.
#[test]
fn plane_passes_match_their_definitions() {
    use zlc_plot::fit::blas::{self, Mat};
    use zlc_plot::fit::plane::{centred_context, plane_products, residual_rss};
    for &(height, width) in &[(1, 1), (3, 5), (7, 129), (130, 2049), (9, 2051), (6, 4100), (5, 5000), (66, 16), (65, 17), (129, 192)] {
        let data: Vec<u16> = (0..height * width).map(|i| ((i * 7919 + 13) % 4001) as u16).collect();
        let reference = data[0] as f64;
        let centred: Vec<f64> = data.iter().map(|&v| v as f64 - reference).collect();
        let floats: Vec<f32> = data.iter().enumerate().map(|(i, &v)| if i % 97 == 5 { f32::NAN } else { v as f32 }).collect();
        let vectors: Vec<Vec<f64>> = (0..3).map(|k| (0..width).map(|i| (((i * 31 + k * 7) % 97) as f64 - 48.0) / 97.0).collect()).collect();
        let y_basis: Vec<f64> = (0..height).map(|j| ((j * 13 % 29) as f64 + 1.0) / 29.0).collect();
        let samples: Samples = data.clone().into();
        let refs: Vec<&[f64]> = vectors.iter().map(Vec::as_slice).collect();
        let view = Mat { rows: width, cols: height, data: centred.clone() };
        for (k, got) in plane_products(&samples, reference, width, height, &refs).iter().enumerate() {
            let want = blas::dgemv_t(width, height, 1.0, &view, 0, 0, &vectors[k]);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "products {height}x{width} vector {k}");
        }
        let mut want_rss = 0.0;
        for start in (0..height).step_by(64) {
            let stop = height.min(start + 64);
            let residuals: Vec<f64> = (start * width..stop * width).map(|p| 800.0 * y_basis[p / width] * vectors[0][p % width] + 0.5 - centred[p]).collect();
            want_rss += blas::ddot(&residuals, &residuals);
        }
        assert_eq!(residual_rss(&samples, reference, 800.0, 0.5, &vectors[0], &y_basis).to_bits(), want_rss.to_bits(), "rss {height}x{width}");
        let mask: Vec<bool> = (0..height * width).map(|i| i % 11 != 3).collect();
        for (plane, mask) in [(Samples::from(floats.clone()), None), (Samples::from(floats.clone()), Some(&mask[..])), (samples.clone(), Some(&mask[..]))] {
            let usable = |p: usize| mask.is_none_or(|m| m[p]) && plane.get(p).is_finite();
            let first = (0..height * width).find(|&p| usable(p)).map_or(0.0, |p| plane.get(p));
            let mut want = [first, 0.0, 0.0, 0.0];
            let stripe = 64 * width;
            let mut stripes = Vec::new();
            for start in (0..height * width).step_by(stripe) {
                let (mut total, mut squares, mut count) = (0.0, 0.0, 0.0);
                for p in start..(start + stripe).min(height * width) {
                    if usable(p) {
                        let shifted = plane.get(p) - first;
                        total += shifted;
                        squares += shifted * shifted;
                        count += 1.0;
                    }
                }
                stripes.push([total, squares, count]);
            }
            for m in 0..3 {
                for s in &stripes {
                    want[m + 1] += s[m];
                }
            }
            let got = centred_context(&plane, mask, width);
            assert!(got.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "context {height}x{width}: {got:?} vs {want:?}");
        }
    }
}

#[test]
fn catalogue_is_complete() {
    let models = builtin_fit_models();
    assert_eq!(models.len(), 13);
    let _ = Arc::clone(&models[0]);
}
