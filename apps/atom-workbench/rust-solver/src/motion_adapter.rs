//! The protocol bridge to the 3D motion solver (`atom_motion`): an rb87-full/2 request whose model carries the motion
//! block (src/full-master-model.ts `MotionModel`), mapped by `atom_motion::request` (the one JSON → request mapping and its
//! checks), solved on the CPU, the solution returned in the one result contract (contract §2.2), every checkpoint
//! streamed first.
//!
//! # Device
//! Only the CPU solver exists (SPEC §0, §9). `options.device` is read ("gpu" when absent); every solve runs on the CPU,
//! the result says device "cpu", and a request for the GPU carries `deviceNote` (`GPU_ABSENT`). The native `gpu`
//! operation answers at once with the same words, so the app never waits behind its compile screen.
//!
//! # Result contract
//! `states`, `timesUs`, `populations` (the held atom's, each row projected onto the probability simplex as the population
//! core's are), `populationAccuracy` against the method's own target (`POPULATION_TARGET`: each sample's total estimated
//! error, the solve's `estimated_errors`, grown by its row's projection; `reason` where some sample exceeds it),
//! `diagnostics` (`engine`, `traceError` and `initialStateError` of the rows before the projection, `elapsedMs`, `method`,
//! `device`, `deviceNote`, `approximations`: the ledger -- `totals`, `items`, `flagged`, `note` --, and on a checkpoint
//! `estimatedRemainingMs`), and `motion`, read by `readMotionResult` in src/solver-types.ts: per lab axis T, its standard
//! error (null where fewer than two atoms or groups read it), the mean velocity and n̄; T_total and its standard error; the
//! trap's frequencies; the loss; `classical`; the momentum and position distributions; the trajectories and the run's step.
//! Every output but the loss describes the atom still held.
use crate::{lattice_core_adapter, Device, Model, Options};
use atom_motion::ledger::{percent, Ledger, Total};
use atom_motion::request::{Request, STATES};
use atom_motion::solve::{self as engine, Distributions, Solution, POPULATION_TARGET};
use atom_population_core::lattice::accuracy;
use atom_population_core::lattice::pipeline;
use serde_json::{json, Value};
use std::time::Instant;

const ENGINE: &str = "rust-atom-motion";
const AXES: [&str; 3] = ["x", "y", "z"];
/// Why a request for the GPU runs on the CPU, and the `gpu` operation's answer (SPEC §9).
pub const GPU_ABSENT: &str = "GPU 计算尚未提供：本次在 CPU 上计算";

/// The `gpu` operation: no GPU solver exists, answered at once.
pub fn start_gpu() -> Result<(), String> {
    Err(GPU_ABSENT.into())
}

fn per_axis<T: serde::Serialize + Copy>(values: &[[T; 3]]) -> Value {
    json!(AXES.iter().enumerate().map(|(a, name)| (name.to_string(), json!(values.iter().map(|v| v[a]).collect::<Vec<T>>()))).collect::<serde_json::Map<String, Value>>())
}
fn distributions(d: &[Distributions; 3], grid: &str) -> Value {
    json!(AXES.iter().zip(d).map(|(name, x)| (name.to_string(), json!({grid: x.grid, "samples": x.samples, "probabilities": x.probabilities}))).collect::<serde_json::Map<String, Value>>())
}
/// The ledger in the contract's shape.
fn approximations(ledger: &Ledger) -> Value {
    let total = |t: &Total| json!({"error": t.error, "target": t.target});
    let mut out = json!({"totals": {"population": total(&ledger.totals.population), "temperature": total(&ledger.totals.temperature)},
        "items": ledger.items.iter().map(|i| json!({"label": i.label, "estimate": i.estimate, "action": i.action})).collect::<Vec<_>>()});
    if let Some(reason) = &ledger.flagged {
        out["flagged"] = json!(reason);
    }
    if let Some(note) = &ledger.note {
        out["note"] = json!(note);
    }
    out
}

/// The result in the one contract; `remaining` marks a streamed checkpoint, with the solve's estimate of the wall time left
/// (ms) where it has one; `gpu` that the GPU was asked for.
fn result(request: &Request, initial: &[f64], solution: &Solution, groups: &[lattice_core_adapter::Group], remaining: Option<Option<f64>>, gpu: bool, started: Instant) -> Result<Value, String> {
    let times = &request.times_us;
    let rows: Vec<Vec<f64>> = solution.populations.iter().map(|row| row.to_vec()).collect();
    if rows.len() != times.len() || rows.iter().any(|row| row.iter().any(|x| !x.is_finite())) {
        return Err("算出的布居不是有限数（每个时刻应有 32 个态的布居）".into());
    }
    if solution.estimated_errors.len() != times.len() || solution.estimated_errors.iter().chain(&solution.loss_probability).any(|x| !x.is_finite()) {
        return Err("算出的估计误差或丢失概率不是有限数".into());
    }
    // The trace gate checks the engine's rows as they are: a systematic trace error still fails.
    let (trace, initial_error) = lattice_core_adapter::row_errors(&rows, initial)?;
    // Published: each row projected onto the probability simplex, the projection's change (over the states and the
    // groups) added to that sample's error, as the population core's rows are.
    let indices: Vec<Vec<usize>> = groups.iter().map(|g| g.indices.clone()).collect();
    let mut errors = solution.estimated_errors.clone();
    let populations: Vec<Vec<f64>> = rows
        .iter()
        .zip(&mut errors)
        .map(|(row, error)| {
            let projected = pipeline::simplex(row);
            *error += accuracy::metric(row, &projected, &indices);
            projected
        })
        .collect();
    let estimated = errors.iter().fold(0_f64, |a, &b| a.max(b));
    let converged = estimated <= POPULATION_TARGET;
    let mut result = json!({"states": request.states, "timesUs": times, "populations": populations,
        "populationAccuracy": {"status": if converged {"estimated-converged"} else {"estimated-partial"}, "requestedTolerance": POPULATION_TARGET,
            "sampleErrors": errors, "estimatedMaximumError": estimated,
            "scope": {"states": STATES, "samples": times.len(), "groups": groups.iter().map(|g| &g.name).collect::<Vec<_>>()}},
        "diagnostics": {"engine": ENGINE, "traceError": trace, "initialStateError": initial_error, "elapsedMs": started.elapsed().as_secs_f64() * 1000.,
            "method": solution.method, "device": "cpu", "approximations": approximations(&solution.approximations)},
        "motion": {"axis": "3d", "temperatureUK": per_axis(&solution.temperature_uk), "temperatureErrorUK": per_axis(&solution.temperature_errors_uk),
            "temperatureTotalUK": solution.temperature_total_uk, "temperatureTotalErrorUK": solution.temperature_total_errors_uk,
            "meanVelocityMps": per_axis(&solution.mean_velocity_mps), "meanOccupation": per_axis(&solution.mean_occupation),
            "trapFrequencyKHz": json!({"x": solution.trap_frequency_khz[0], "y": solution.trap_frequency_khz[1], "z": solution.trap_frequency_khz[2]}),
            "lossProbability": solution.loss_probability, "classical": solution.classical.iter().map(|axes| axes.iter().all(|&inside| inside)).collect::<Vec<bool>>(),
            "momentum": distributions(&solution.velocity, "velocitiesMps"), "position": distributions(&solution.position, "positionsUm"),
            "trajectories": solution.trajectories, "timeStepUs": solution.time_step_us}});
    if gpu {
        result["diagnostics"]["deviceNote"] = json!(GPU_ABSENT);
    }
    if !converged {
        result["populationAccuracy"]["reason"] = json!(solution
            .reason
            .clone()
            .unwrap_or_else(|| format!("估计误差 {}，超过目标 {}（含把布居的负值去掉、总和归一所作的改动）", percent(estimated), percent(POPULATION_TARGET))));
    }
    if let Some(remaining) = remaining {
        result["progress"] = json!(true);
        if let Some(ms) = remaining.filter(|ms| ms.is_finite()) {
            result["diagnostics"]["estimatedRemainingMs"] = json!(ms.max(0.));
        }
    }
    Ok(result)
}

/// A motion request -- its whole native line mapped and checked by `atom_motion::request::Request::from_native` (a trajectory
/// alone of the complete 32-state model on a finite grid) -- through the solver: every checkpoint published as it is made,
/// the final solution returned.
pub fn solve(model: &Model, line: &Value, options: &Options, publish: &mut dyn FnMut(&Value)) -> Result<Value, String> {
    let started = Instant::now();
    let request = Request::from_native(line)?;
    let (_, groups) = lattice_core_adapter::groups(model)?;
    let initial: Vec<f64> = (0..STATES).map(|i| request.initial_density[(i, i)].re).collect();
    let gpu = options.device == Device::Gpu;
    let mut failure: Option<String> = None;
    let mut observe = |checkpoint: &Solution, remaining: Option<f64>| {
        if failure.is_some() {
            return;
        }
        match result(&request, &initial, checkpoint, &groups, Some(remaining), gpu, started) {
            Ok(value) => publish(&value),
            Err(e) => failure = Some(e),
        }
    };
    let solution = engine::solve(&request, &mut observe)?;
    if let Some(e) = failure {
        return Err(e);
    }
    result(&request, &initial, &solution, &groups, None, gpu, started)
}
