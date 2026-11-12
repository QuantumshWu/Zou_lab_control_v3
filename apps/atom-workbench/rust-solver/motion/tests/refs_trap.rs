//! The reported trap (SPEC §1.3) of every reference scene against its expectation, without a solve: the trap is a
//! number of the model alone -- the request, the gates' closure, the descent --, so the references' `trapFrequency`
//! series (refs/lib/phys.mjs `runningTrap`, `preparedTrap`) are judged here as refs/check.mjs judges them in a
//! result: |engine − reference| ≤ 2u, u the reference's own stated uncertainty (a quantity without a user's target has
//! no other tolerance; check.mjs, rule 0 for one deterministic output).
use atom_motion::estimate::unseen;
use atom_motion::field::Light;
use atom_motion::model::{Model, Trap, Unit, GROUND};
use atom_motion::request::Request;
use atom_motion::units::MOST_TRAJECTORIES;
use serde_json::Value;
use std::path::PathBuf;

fn refs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs")
}

fn json(path: PathBuf) -> Value {
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

/// The prepared and the reported trap of a request as the solve makes them (solve.rs `attempt`: the reference state,
/// the model closed by its gates over the reach at the request's temperature).
fn traps(request: &Request) -> (Trap, Trap) {
    let light = Light::new(request);
    let chief = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
    let mut reference = [0.0; GROUND];
    reference[chief] = 1.0;
    let mut kept: Vec<Unit> = vec![];
    loop {
        let model = Model::new(request, &light, &kept).unwrap();
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            let reported = model.trap(&reference, &model.reported()).unwrap();
            return (trap, reported);
        }
        kept.extend(failing);
    }
}

/// Every expectation that states trap frequencies: the engine's reported trap of its scene is within twice the
/// reference's own uncertainty of them.
#[test]
fn the_reported_trap_is_the_references() {
    let mut seen = 0;
    for entry in std::fs::read_dir(refs().join("expect")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().and_then(|x| x.to_str()) != Some("json") {
            continue;
        }
        let expectation = json(path);
        let Some(series) = expectation["series"].as_array() else { continue };
        let stated: Vec<&Value> = series.iter().filter(|s| s["quantity"] == "trapFrequency").collect();
        if stated.is_empty() {
            continue;
        }
        let id = expectation["id"].as_str().unwrap();
        let request = Request::from_native(&json(refs().join(expectation["request"].as_str().unwrap()))).unwrap();
        let (_, reported) = traps(&request);
        for s in stated {
            let axis = ["x", "y", "z"].iter().position(|a| s["axis"] == *a).unwrap();
            let (value, u) = (s["value"].as_f64().unwrap(), s["u"].as_f64().unwrap());
            let engine = reported.lab[axis] / std::f64::consts::TAU * 1e3;
            println!("{id} {}: engine {engine:.12e} reference {value:.12e} u {u:.3e} (d/u {:.4})", s["axis"], (engine - value) / u);
            assert!((engine - value).abs() <= 2.0 * u, "{id} {}: {engine} against {value} ± {u}", s["axis"]);
            seen += 1;
        }
    }
    assert!(seen > 0);
}

/// A measurement, not a judge (run by name with --ignored): the reported trap's ratio to the prepared one,
/// (ω_reported/ω_prepared)² − 1, of the scenes in REFS_TRAP_SCENES at the request's mass times each factor of
/// REFS_TRAP_MASSES. The ratio does not hold the mass but through the photon's recoil in the folds' denominators
/// (ħk²/2m ∝ 1/m): its change with the mass is the recoil's part of the reported trap (measured 2026-11-11: linear in
/// 1/m on R2, R3, R4_LIN_1 and R10, where the reference's exact denominator has a second-order part as well).
#[test]
#[ignore]
fn the_reported_trap_against_the_mass() {
    let scenes = std::env::var("REFS_TRAP_SCENES").expect("REFS_TRAP_SCENES: scene ids, comma separated");
    let masses = std::env::var("REFS_TRAP_MASSES").unwrap_or_else(|_| "1".into());
    for id in scenes.split(',') {
        for factor in masses.split(',').map(|x| x.parse::<f64>().unwrap()) {
            let mut request = Request::from_native(&json(refs().join("scenes").join(format!("{id}.json")))).unwrap();
            request.mass_kg *= factor;
            let (prepared, reported) = traps(&request);
            let ratio: Vec<String> = (0..3).map(|a| format!("{:.13e}", (reported.lab[a] / prepared.lab[a]).powi(2) - 1.0)).collect();
            println!("TRAP {id} mass {factor} prepared_kHz {:?} ratio_less_1 {}", prepared.lab.map(|w| w / std::f64::consts::TAU * 1e3), ratio.join(" "));
        }
    }
}
