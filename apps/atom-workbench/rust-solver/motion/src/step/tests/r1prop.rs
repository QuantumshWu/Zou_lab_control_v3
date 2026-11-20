//! R1-PROP's measurements on the mainline model (Scene -> choose -> Selected, the solve's `run_of`): the six scenes'
//! requests (R1PROP_REQ: a folder of tilt0, tilt2, R4_LIN_4, GMD1C_1ms, PGC3D_1ms, TET_1ms .json), R1PROP_SCENE some
//! of them (comma separated). Ignored: each reads the scenes' requests, and is run by name, one at a time:
//! `<test exe> step::tests::r1prop::<name> --ignored --nocapture --test-threads 1`.
//! - `r1prop_bits`: the trajectories' states step by step, each step's hash and whether its propagation went through
//!   the decomposition -- the regression of the propagation's change: the steps the decomposition took before are the
//!   same to the bit.
//! - `r1prop_refused`: the steps the decomposition refuses (`spectral::Refusal`), by why, at a step and at its halves.

use crate::model::selected::Selected;
use crate::request::Request;
use crate::step::{Brownian, Trajectory};

use super::super::{Midway, Scratch};
use super::Everywhere;

pub(super) fn env(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().map(|s| s.parse().unwrap()).unwrap_or(default)
}

pub(super) fn list(key: &str, default: &str) -> Vec<f64> {
    std::env::var(key).unwrap_or_else(|_| default.to_string()).split(',').map(|x| x.trim().parse().unwrap()).collect()
}

/// The six scenes' requests (R1PROP_REQ), R1PROP_SCENE some of them.
pub(super) fn scenes() -> Vec<(&'static str, Request)> {
    let dir = std::path::PathBuf::from(std::env::var("R1PROP_REQ").expect("R1PROP_REQ: the requests' folder"));
    let files = [("PGC3D", "PGC3D_1ms.json"), ("TET", "TET_1ms.json"), ("R4", "R4_LIN_4.json"), ("GMD1C", "GMD1C_1ms.json"), ("tilt0", "tilt0.json"), ("tilt2", "tilt2.json")];
    let only = std::env::var("R1PROP_SCENE").ok();
    files
        .into_iter()
        .filter(|(name, _)| only.as_deref().is_none_or(|o| o.split(',').any(|x| x == *name)))
        .map(|(name, file)| {
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
            (name, Request::from_native(&line).unwrap())
        })
        .collect()
}

/// The run as the solve makes it, and its reference state.
pub(super) fn run_of(request: &Request) -> (Selected, usize) {
    let chief = crate::solve::chief(request);
    let run = crate::solve::run_of(request, std::array::from_fn(|s| f64::from(u8::from(s == chief)))).unwrap();
    (run, chief)
}

/// The trajectory i's start: the moving point of DIAGB, (0.1, 0.2, 0.3) µm, at the thermal speed with the signs of
/// the trajectory's number's bits (as STEPCOST's count).
pub(super) fn start(run: &Selected, request: &Request, chief: usize, i: u64) -> Trajectory {
    use crate::step::Dynamics;
    let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
    let sign = |b: u64| if (i >> b) & 1 == 1 { -1.0 } else { 1.0 };
    Trajectory::new(run, chief, [0.1, 0.2, 0.3], [sign(0) * speed, sign(1) * 0.7 * speed, sign(2) * 1.2 * speed], 7, 200 + i)
}

/// FNV-1a of a trajectory's state: position, momentum, internal state, time, threshold, jumps.
fn hashed(t: &Trajectory) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    let mut eat = |x: u64| {
        for b in x.to_le_bytes() {
            h = (h ^ u64::from(b)).wrapping_mul(0x100000001b3);
        }
    };
    for x in t.r.iter().chain(&t.kappa) {
        eat(x.to_bits());
    }
    for z in t.psi.iter() {
        eat(z.re.to_bits());
        eat(z.im.to_bits());
    }
    eat(t.time.to_bits());
    eat(t.threshold.to_bits());
    eat(u64::from(t.jumps));
    h
}

/// The regression (module documentation): R1PROP_TRAJ trajectories (8) from the moving point, each R1PROP_STEPS steps
/// (40) at each τ of R1PROP_TAUS (0.474, 0.948), on all the threads: per step its hash (every step's propagation
/// through the decomposition; before 2026-11-20 whether the last stretch's went through it); a refused step ends the
/// trajectory's record.
#[test]
#[ignore]
fn r1prop_bits() {
    use rayon::prelude::*;
    let taus = list("R1PROP_TAUS", "0.474,0.948");
    let (count, steps) = (env("R1PROP_TRAJ", 8.0) as u64, env("R1PROP_STEPS", 40.0) as usize);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let each: Vec<Vec<String>> = (0..count)
                .into_par_iter()
                .map(|i| {
                    let mut t = start(&run, &request, chief, i);
                    let mut scratch = Scratch::default();
                    let mut lines = Vec::new();
                    for k in 0..steps {
                        match Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere)) {
                            Ok(step) => {
                                t = step.end;
                                lines.push(format!("R1-BITS {name} tau {tau} traj {i} step {k} decomposed 1 hash {:016x}", hashed(&t)));
                            }
                            Err(e) => {
                                lines.push(format!("R1-BITS {name} tau {tau} traj {i} step {k} refused {e:?}"));
                                break;
                            }
                        }
                    }
                    lines
                })
                .collect();
            for line in each.iter().flatten() {
                println!("{line}");
            }
        }
    }
}

/// A step of τ from the trajectory's state as the advance takes it (adapt.rs: a refused step at its half, down to
/// `depth` halvings): the end state, and per halving level the refusals by why -- [decomposition, rounding, radius,
/// other].
fn halved(run: &Selected, t: &Trajectory, tau: f64, scratch: &mut Scratch, depth: usize, refusals: &mut [[usize; 4]]) -> Option<Trajectory> {
    use crate::spectral::Refusal;
    use crate::step::Refused;
    match Midway::new(run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], scratch, &Everywhere)) {
        Ok(step) => Some(step.end),
        Err(e) => {
            let why = match e {
                Refused::Propagation(Refusal::Decomposition) => 0,
                Refused::Propagation(Refusal::Rounding) => 1,
                Refused::Propagation(Refusal::Radius) => 2,
                _ => 3,
            };
            refusals[0][why] += 1;
            if why == 3 || depth == 0 {
                return None;
            }
            let (rest, deeper) = refusals.split_at_mut(1);
            let _ = rest;
            let middle = halved(run, t, 0.5 * tau, scratch, depth - 1, deeper)?;
            halved(run, &middle, 0.5 * tau, scratch, depth - 1, deeper)
        }
    }
}

/// The steps the decomposition refuses (`spectral::Refusal`), as STEPCOST's count of the steps with no route:
/// R1PROP_TRAJ trajectories (128) from the moving point -- the velocity's signs by the trajectory's number --, each
/// R1PROP_STEPS steps (200) at each τ of R1PROP_TAUS, on all the threads; a refused step taken at its halves as the
/// advance takes it. Per τ the refusals by why at the step's own length and at its halves, the trajectories they fall
/// on, the jumps' count, the wall time.
#[test]
#[ignore]
fn r1prop_refused() {
    use rayon::prelude::*;
    let taus = list("R1PROP_TAUS", "0.474,0.948");
    let (count, steps) = (env("R1PROP_TRAJ", 128.0) as u64, env("R1PROP_STEPS", 200.0) as usize);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let clock = std::time::Instant::now();
            let each: Vec<([[usize; 4]; 4], u32, Option<String>)> = (0..count)
                .into_par_iter()
                .map(|i| {
                    let mut t = start(&run, &request, chief, i);
                    let mut scratch = Scratch::default();
                    let mut refusals = [[0usize; 4]; 4];
                    for k in 0..steps {
                        match halved(&run, &t, tau, &mut scratch, 3, &mut refusals) {
                            Some(next) => t = next,
                            None => return (refusals, t.jumps, Some(format!("trajectory {i} ended at step {k}"))),
                        }
                    }
                    (refusals, t.jumps, None)
                })
                .collect();
            let mut total = [[0usize; 4]; 4];
            for (r, _, _) in &each {
                for (level, row) in r.iter().enumerate() {
                    for (why, x) in row.iter().enumerate() {
                        total[level][why] += x;
                    }
                }
            }
            let on = each.iter().filter(|(r, _, _)| r[0].iter().sum::<usize>() > 0).count();
            let jumps: u32 = each.iter().map(|e| e.1).sum();
            let ended: Vec<&String> = each.iter().filter_map(|e| e.2.as_ref()).collect();
            println!(
                "R1-REFUSED {name} tau {tau}: {count} trajectories x {steps} steps = {}; refused at tau [decomposition, rounding, radius, other] {:?} ({:.3} %), on {on} trajectories; at tau/2 {:?}, tau/4 {:?}, tau/8 {:?}; {jumps} jumps; ended {}: {:?}; wall {:.1} s",
                count as usize * steps,
                total[0],
                100.0 * total[0].iter().sum::<usize>() as f64 / (count as usize * steps) as f64,
                total[1],
                total[2],
                total[3],
                ended.len(),
                ended.iter().take(3).collect::<Vec<_>>(),
                clock.elapsed().as_secs_f64()
            );
        }
    }
}
