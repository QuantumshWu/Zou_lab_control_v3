//! `run.fit` (PLUGIN_PROTOCOL §5): a run's independent fits of many series over one
//! axis, by the engine a panel's fit uses, with `FitRequest`'s defaults; and the
//! catalogue of models it takes (`fit_models`).
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock};

use serde_json::{Value, json};
use zlc_plot::fit::{builtin_fit_models, CellData, FitCell, FitEngine, FitError, FitModelSpec, FitRequest, FitTarget};
use zlc_plot::session::real_values;
use zlc_plugin_host::wire::Array;
use zlc_plugin_host::{CallError, ErrorKind, Lent};

static ENGINE: LazyLock<FitEngine> = LazyLock::new(FitEngine::default);

/// A model fitted over one series axis: what `run.fit` takes.
fn fits_a_series(model: &FitModelSpec) -> bool { model.targets.contains(&FitTarget::Series) && model.independent_arity == 1 }

/// `fit_models`: the models `run.fit` takes.
pub fn fit_models() -> Value {
    let models: Vec<Value> = builtin_fit_models().iter().filter(|model| fits_a_series(model)).map(|model| json!({
        "id": model.model_id, "label": model.display_name,
        "parameters": model.parameters.iter().map(|parameter| json!({"name": parameter.name, "unit_relation": parameter.unit_relation.value()})).collect::<Vec<_>>()
    })).collect();
    json!({"models": models})
}

/// What a batch answers, a row per cell. A cell that could not be fitted has NaN
/// values and errors, `success` false and its reason in `messages`; a parameter the
/// data do not determine has a NaN error in a successful cell.
#[derive(Debug, Clone)]
pub struct FitAnswer {
    pub parameters: Vec<String>,
    pub values: Vec<Vec<f64>>,
    pub errors: Vec<Vec<f64>>,
    pub success: Vec<bool>,
    pub reduced_chi_square: Vec<f64>,
    pub messages: Vec<String>,
}

impl FitAnswer {
    /// The reply; a number JSON cannot write (NaN) is null.
    fn to_json(&self) -> Value {
        json!({"parameters": self.parameters, "values": self.values, "errors": self.errors, "success": self.success,
               "reduced_chi_square": self.reduced_chi_square, "messages": self.messages})
    }
}

fn reals(array: &Array, name: &str) -> Result<Vec<f64>, FitError> {
    real_values(&array.values).ok_or_else(|| FitError::value(format!("run.fit: {name} must be real")))
}

/// The values of a `(cells, columns)` array, row after row.
fn rows(array: &Array, name: &str, cells: usize, columns: usize) -> Result<Vec<f64>, FitError> {
    if array.shape != [cells, columns] {
        return Err(FitError::value(format!("run.fit: {name} has shape {:?}, not {:?}", array.shape, [cells, columns])));
    }
    reals(array, name)
}

/// The batch `y` `(cells, n)` over `x` `(n)`, with `sigma` `(cells, n)` and warm starts
/// `warm` `(cells, parameters)`, a row with a non-finite value being none for its cell.
/// One cell's failure is that cell's: only a bad request or `Cancelled` fails the call.
pub fn fit_answer(engine: &FitEngine, model_id: &str, x: &Array, y: &Array, sigma: Option<&Array>, warm: Option<&Array>,
                  cancelled: Option<&(dyn Fn() -> bool + Sync)>) -> Result<FitAnswer, FitError> {
    let model = engine.registry.get(model_id).map_err(|error| FitError::value(error.0))?;
    if !fits_a_series(&model) { return Err(FitError::value(format!("run.fit: {model_id} does not fit a series"))); }
    let [points] = x.shape[..] else { return Err(FitError::value(format!("run.fit: x has shape {:?}, not one axis", x.shape))) };
    let [cells, _] = y.shape[..] else { return Err(FitError::value(format!("run.fit: y has shape {:?}, not (cells, {points})", y.shape))) };
    let count = model.parameters.len();
    let coordinates: Arc<[f64]> = reals(x, "x")?.into();
    let observations = rows(y, "y", cells, points)?;
    let sigma = sigma.map(|sigma| rows(sigma, "sigma", cells, points)).transpose()?;
    let warm = warm.map(|warm| rows(warm, "warm", cells, count)).transpose()?;
    let row = |values: &[f64], width: usize, cell: usize| values[cell * width..(cell + 1) * width].to_vec();
    let batch: Vec<FitCell> = (0..cells).map(|cell| FitCell {
        data: CellData::Points { coordinates: vec![coordinates.clone()], observations: Some(row(&observations, points, cell).into()) },
        sigma: sigma.as_ref().map(|sigma| row(sigma, points, cell).into()),
        sigma_dof: None, selected_indices: None, data_revision: 0,
        warm_start: warm.as_ref().map(|warm| row(warm, count, cell)).filter(|start| start.iter().all(|value| value.is_finite())),
    }).collect();
    let outcomes = if batch.is_empty() { Vec::new() } else {
        let request = FitRequest { cancelled, ..FitRequest::default() };
        match engine.fit_batch(&model, &batch, &request) {
            Ok(outcomes) => outcomes,
            Err(FitError::Cancelled) => return Err(FitError::Cancelled),
            Err(error) => vec![Err(error.failure_text()); cells],
        }
    };
    let mut answer = FitAnswer { parameters: model.parameters.iter().map(|parameter| parameter.name.clone()).collect(),
        values: Vec::with_capacity(cells), errors: Vec::with_capacity(cells), success: Vec::with_capacity(cells),
        reduced_chi_square: Vec::with_capacity(cells), messages: Vec::with_capacity(cells) };
    for outcome in outcomes {
        match outcome {
            Ok(result) => {
                answer.values.push(result.parameter_values);
                answer.errors.push(result.standard_errors);
                answer.success.push(result.success);
                answer.reduced_chi_square.push(result.reduced_chi_square);
                answer.messages.push(result.message);
            }
            Err(message) => {
                answer.values.push(vec![f64::NAN; count]);
                answer.errors.push(vec![f64::NAN; count]);
                answer.success.push(false);
                answer.reduced_chi_square.push(f64::NAN);
                answer.messages.push(message);
            }
        }
    }
    Ok(answer)
}

/// One `run.fit` request `{model, x, y, sigma, warm}`; a run that stops ends its batch.
pub fn answer(params: &Value, lent: &Lent, stop: &AtomicBool) -> Result<Value, CallError> {
    let refused = |message: String| CallError::new(ErrorKind::Refused, message);
    let optional = |field: &str| (!params[field].is_null()).then(|| lent.array(&params[field])).transpose().map_err(refused);
    let (x, y) = (lent.array(&params["x"]).map_err(refused)?, lent.array(&params["y"]).map_err(refused)?);
    let (sigma, warm) = (optional("sigma")?, optional("warm")?);
    let cancelled = || stop.load(Ordering::Relaxed);
    match fit_answer(&ENGINE, params["model"].as_str().unwrap_or(""), &x, &y, sigma.as_ref(), warm.as_ref(), Some(&cancelled)) {
        Ok(answer) => Ok(answer.to_json()),
        Err(FitError::Cancelled) => Err(CallError::new(ErrorKind::Cancelled, "the run was stopped")),
        Err(error) => Err(refused(error.failure_text())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Instant;
    use zlc_core::{AnyBuffer, OwnedBuffer};

    fn array(values: Vec<f64>, shape: &[usize]) -> Array { Array { values: AnyBuffer::from(OwnedBuffer::from_vec(values)), shape: shape.to_vec() } }

    /// `[amplitude, offset, frequency, decay_time, phase]`, read from x = 0.
    fn damped(x: f64, truth: &[f64; 5]) -> f64 {
        truth[0] * (std::f64::consts::TAU * truth[2] * x + truth[4]).sin() * (-x / truth[3]).exp() + truth[1]
    }

    /// Deterministic noise, about N(0, 1): a sum of twelve uniforms from a small generator.
    fn noise(state: &mut u64) -> f64 {
        (0..12).map(|_| {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (*state >> 40) as f64 / (1u64 << 24) as f64
        }).sum::<f64>() - 6.0
    }

    fn truths(cells: usize) -> Vec<[f64; 5]> {
        (0..cells).map(|cell| { let k = cell as f64 / cells.max(1) as f64; [1.0 + k, 0.5 - k, 0.4 + 0.2 * k, 8.0 + 8.0 * k, 0.3 + k] }).collect()
    }

    fn batch(truths: &[[f64; 5]], points: usize, spread: f64) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..points).map(|index| index as f64 * 0.25).collect();
        let mut state = 7;
        let y = truths.iter().flat_map(|truth| x.iter().map(|&x| damped(x, truth)).collect::<Vec<_>>()).map(|value| value + spread * noise(&mut state)).collect();
        (x, y)
    }

    #[test]
    fn a_batch_recovers_its_parameters_and_a_bad_cell_is_only_that_cell() {
        let engine = FitEngine::default();
        let (cells, points, spread) = (6, 60, 0.02);
        let truth = truths(cells);
        let (x, mut y) = batch(&truth, points, spread);
        // Cell 1 all NaN, cell 3 with five finite points for five parameters, cell 4 with holes the engine drops.
        y[points..2 * points].fill(f64::NAN);
        y[3 * points + 5..4 * points].fill(f64::NAN);
        for hole in [3, 17, 40] { y[4 * points + hole] = f64::NAN; }
        let answer = fit_answer(&engine, "damped_sine", &array(x.clone(), &[points]), &array(y.clone(), &[cells, points]), None, None, None).unwrap();
        assert_eq!(answer.parameters, ["amplitude", "offset", "baseband_frequency", "decay_time", "phase"]);
        assert_eq!(answer.success, [true, false, true, false, true, true], "{:?}", answer.messages);
        for cell in [1, 3] {
            assert!(answer.values[cell].iter().chain(&answer.errors[cell]).all(|value| value.is_nan()) && answer.reduced_chi_square[cell].is_nan());
            assert!(!answer.messages[cell].is_empty());
        }
        for cell in [0, 2, 4, 5] {
            for (index, name) in answer.parameters.iter().enumerate() {
                let (value, error) = (answer.values[cell][index], answer.errors[cell][index]);
                assert!(error.is_finite() && error > 0.0, "cell {cell} {name}: error {error}");
                assert!((value - truth[cell][index]).abs() < 6.0 * error, "cell {cell} {name}: {value} ± {error}, truth {}", truth[cell][index]);
            }
        }

        // The answer is the engine's own, cell for cell.
        let shared: Arc<[f64]> = x.clone().into();
        let direct: Vec<FitCell> = y.chunks(points).map(|row| FitCell { data: CellData::Points { coordinates: vec![shared.clone()], observations: Some(row.to_vec().into()) },
            sigma: None, sigma_dof: None, selected_indices: None, data_revision: 0, warm_start: None }).collect();
        let model = engine.registry.get("damped_sine").unwrap();
        let outcomes = engine.fit_batch(&model, &direct, &FitRequest::default()).unwrap();
        for (cell, outcome) in outcomes.iter().enumerate() {
            match outcome {
                Ok(result) => {
                    assert_eq!(answer.values[cell], result.parameter_values);
                    assert_eq!(answer.errors[cell], result.standard_errors);
                    assert_eq!(answer.reduced_chi_square[cell], result.reduced_chi_square);
                    assert_eq!(answer.success[cell], result.success);
                }
                Err(message) => assert_eq!((&answer.messages[cell], answer.success[cell]), (message, false)),
            }
        }

        let reply = answer.to_json();
        assert_eq!(reply["values"][1][0], Value::Null);
        assert_eq!(reply["success"][0], true);
    }

    #[test]
    fn sigma_and_warm_reach_their_cells_and_a_wrong_shape_is_refused() {
        let engine = FitEngine::default();
        let (cells, points, spread) = (3, 60, 0.02);
        let truth = truths(cells);
        let (x, y) = batch(&truth, points, spread);
        let (x, y) = (array(x, &[points]), array(y, &[cells, points]));
        let sigma = array(vec![spread; cells * points], &[cells, points]);
        // A row with a NaN (a segment before any good one) is no warm start, not a failure.
        let mut starts: Vec<f64> = truth.iter().flatten().copied().collect();
        starts[5] = f64::NAN;
        let warm = array(starts, &[cells, 5]);
        let answer = fit_answer(&engine, "damped_sine", &x, &y, Some(&sigma), Some(&warm), None).unwrap();
        assert_eq!(answer.success, [true, true, true], "{:?}", answer.messages);
        // With the noise's own sigma the reduced chi-square is near one.
        assert!(answer.reduced_chi_square.iter().all(|chi| (0.4..1.8).contains(chi)), "{:?}", answer.reduced_chi_square);

        let short = array(vec![0.0; 3], &[3]);
        for result in [fit_answer(&engine, "damped_sine", &short, &y, None, None, None), fit_answer(&engine, "damped_sine", &x, &y, Some(&short), None, None),
                       fit_answer(&engine, "damped_sine", &x, &y, None, Some(&sigma), None), fit_answer(&engine, "no_such_model", &x, &y, None, None, None)] {
            assert!(result.is_err());
        }
        let empty = fit_answer(&engine, "damped_sine", &x, &array(Vec::new(), &[0, points]), None, None, None).unwrap();
        assert!(empty.values.is_empty() && empty.parameters.len() == 5);
    }

    #[test]
    fn a_stopped_run_ends_its_batch() {
        let engine = FitEngine::default();
        let (x, y) = batch(&truths(2), 60, 0.02);
        let stopped = || true;
        let result = fit_answer(&engine, "damped_sine", &array(x, &[60]), &array(y, &[2, 60]), None, None, Some(&stopped));
        assert!(matches!(result, Err(FitError::Cancelled)));

        // A Stop that comes once the batch's one bucket is solving: its cells not begun never start.
        let (cells, points) = (256, 60);
        let (x, y) = batch(&truths(cells), points, 0.02);
        let asked = std::sync::atomic::AtomicUsize::new(0);
        let begun = || asked.fetch_add(1, Ordering::Relaxed) > cells + cells / 2;
        let result = fit_answer(&engine, "damped_sine", &array(x, &[points]), &array(y, &[cells, points]), None, None, Some(&begun));
        assert!(matches!(result, Err(FitError::Cancelled)));
        assert!(asked.load(Ordering::Relaxed) < 2 * cells, "{asked:?}");
    }

    #[test]
    fn the_catalogue_is_the_series_models() {
        let catalogue = fit_models();
        let models = catalogue["models"].as_array().unwrap();
        let damped = models.iter().find(|row| row["id"] == "damped_sine").unwrap();
        assert_eq!(damped["label"], "Damped sine");
        assert_eq!(damped["parameters"][2], json!({"name": "baseband_frequency", "unit_relation": "inverse_axis_0"}));
        for model in builtin_fit_models() {
            let listed = models.iter().find(|row| row["id"] == model.model_id.as_str());
            assert_eq!(listed.is_some(), fits_a_series(&model), "{}", model.model_id);
            if let Some(row) = listed { assert_eq!(row["parameters"].as_array().unwrap().len(), model.parameters.len()); }
        }
    }

    /// `cargo test ... a_thousand_cells_take -- --ignored --nocapture`.
    #[test]
    #[ignore = "a timing, read by hand"]
    fn a_thousand_cells_take() {
        let engine = FitEngine::default();
        let (cells, points) = (1000, 60);
        let (x, y) = batch(&truths(cells), points, 0.02);
        let (x, y) = (array(x, &[points]), array(y, &[cells, points]));
        for round in 0..3 {
            let started = Instant::now();
            let answer = fit_answer(&engine, "damped_sine", &x, &y, None, None, None).unwrap();
            let good = answer.success.iter().filter(|success| **success).count();
            println!("round {round}: {cells} cells of {points} points, damped_sine: {:.1} ms, {good} succeeded", started.elapsed().as_secs_f64() * 1e3);
        }
    }
}
