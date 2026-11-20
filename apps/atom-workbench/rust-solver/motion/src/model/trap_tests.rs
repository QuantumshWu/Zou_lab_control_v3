//! The reported trap (SPEC §1.3) of every reference scene against its expectation, without a solve: the trap is a
//! number of the run alone -- the request, the run's fold, the descent --, so the references' `trapFrequency`
//! series (refs/lib/phys.mjs `runningTrap`, `preparedTrap`) are judged here as refs/check.mjs judges them in a
//! result: |engine − reference| ≤ 2u, u the reference's own stated uncertainty (a quantity without a user's target has
//! no other tolerance; check.mjs, rule 0 for one deterministic output).
use crate::model::{Trap, GROUND};
use crate::request::Request;
use serde_json::Value;
use std::path::PathBuf;

fn refs() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs")
}

fn json(path: PathBuf) -> Value {
    serde_json::from_str(&std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()))).unwrap()
}

/// The reported trap of a request as the solve makes it (solve.rs `reported`): the run at the reference state -- the
/// initial component of the largest population --, its reported potential's trap there; none where the scene refuses the
/// request (a light outside the allowed approximations, scene.rs `Scene::new`), with its reason. 临时入口：P 取全部 32
/// 态（`select::choose` 落地前；合并时改回 `choose`）.
fn reported(request: &Request) -> Result<Trap, String> {
    crate::scene::Scene::new(request)?;
    let reference: [f64; GROUND] = std::array::from_fn(|s| f64::from(u8::from(s == crate::solve::chief(request))));
    let run = crate::model::selected::keeping(request, (0..crate::request::STATES).collect(), None, reference);
    crate::solve::reported(&run, &reference)
}

/// Every expectation that states trap frequencies: the engine's reported trap of its scene is within twice the
/// reference's own uncertainty of them -- every axis read and said, those beyond named; a scene the engine refuses is
/// said with its reason.
#[test]
fn the_reported_trap_is_the_references() {
    let (mut seen, mut beyond) = (0, vec![]);
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
        let reported = match reported(&request) {
            Ok(trap) => trap,
            Err(why) => {
                println!("{id}: refused: {why}");
                continue;
            }
        };
        for s in stated {
            let axis = ["x", "y", "z"].iter().position(|a| s["axis"] == *a).unwrap();
            let (value, u) = (s["value"].as_f64().unwrap(), s["u"].as_f64().unwrap());
            let engine = reported.lab[axis] / std::f64::consts::TAU * 1e3;
            println!("{id} {}: engine {engine:.12e} reference {value:.12e} u {u:.3e} (d/u {:.4})", s["axis"], (engine - value) / u);
            if (engine - value).abs() > 2.0 * u {
                beyond.push(format!("{id} {}", s["axis"]));
            }
            seen += 1;
        }
    }
    assert!(seen > 0 && beyond.is_empty(), "{} of {seen} axes beyond 2u: {beyond:?}", beyond.len());
}
