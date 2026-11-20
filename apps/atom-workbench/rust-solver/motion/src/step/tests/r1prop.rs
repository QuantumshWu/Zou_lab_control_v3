//! R1-PROP's measurements on the mainline model (Scene -> choose -> Selected, the solve's `run_of`): the six scenes'
//! requests (R1PROP_REQ: a folder of tilt0, tilt2, R4_LIN_4, GMD1C_1ms, PGC3D_1ms, TET_1ms .json), R1PROP_SCENE some
//! of them (comma separated). Ignored: each reads the scenes' requests, and is run by name, one at a time:
//! `<test exe> step::tests::r1prop::<name> --ignored --nocapture --test-threads 1`.
//! - `r1prop_bits`: the trajectories' states step by step, each step's hash and whether its propagation went through
//!   the decomposition -- the regression of the propagation's change: the steps the decomposition took before are the
//!   same to the bit.
//! - `r1prop_refused`: the steps the decomposition refuses (`spectral::Refusal`), by why, at a step and at its halves.
//! - `r1prop_remainder`: the remainder estimate (`Picture::remainder`) against the flow.
//! - `r1prop_solve`: the probe and the pilot as the solve runs them, their messages' times.

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

/// The remainder estimate against the flow (spectral.rs `Picture::remainder`; STEPCOST's `stepcost_flow`): a walk from
/// the moving point (trajectory 3) at τ = R1PROP_WALK (0.474), R1PROP_STEPS of its states (12); at each, one step of
/// each τ of R1PROP_TAUS through the picture as the step prepares it, against RK4 of the picture's own generator
/// (R1PROP_RK steps over 0.948 µs, 200000): the state's error at the step's end over its norm, E, their ratio; and E's
/// time (the least of five) and count.
#[test]
#[ignore]
fn r1prop_remainder() {
    use crate::linalg::{norm_sqr, C64};
    use super::super::{pictured, predicted, Along, STATES, SUMS};
    let walk = env("R1PROP_WALK", 0.474);
    let taus = list("R1PROP_TAUS", "0.474,0.948");
    let (steps, rk) = (env("R1PROP_STEPS", 12.0) as usize, env("R1PROP_RK", 200000.0));
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        let mut t = start(&run, &request, chief, 3);
        let mut scratch = Scratch::default();
        for k in 0..steps {
            for &tau in &taus {
                let h = 0.5 * tau;
                let Ok(midway) = Midway::new(&run, &t, tau) else { continue };
                let n = t.psi.len();
                let mut s = Scratch::default();
                s.fit(n, SUMS);
                let kappa_m = predicted(&t.forms, midway.reached.1, tau, t.time + h);
                let along = Along::new(&midway.generator, kappa_m, kappa_m.map(|k| midway.hbar_over_mass * k));
                let launched = along.turned(&t.psi, -h, true);
                if let Err(e) = pictured(&along, &mut s, tau, Some(&launched[..n])) {
                    println!("R1-REMAINDER {name} state {k} tau {tau}: refused {e:?}");
                    continue;
                }
                let picture = &mut s.picture;
                let generator = |u: f64| -> Vec<C64> {
                    let mut g: Vec<C64> = (0..n * n).map(|q| picture.base.data[q] + picture.ends[0].data[q] * u + picture.ends[1].data[q] * (u * u)).collect();
                    for tone in &picture.beats {
                        for (member, beat) in tone.members.iter().zip([tone.beat, -tone.beat]) {
                            let p = C64::cis(beat * u);
                            for (q, g) in g.iter_mut().enumerate() {
                                *g += p * (member[0].data[q] + member[1].data[q] * u + member[2].data[q] * (u * u));
                            }
                        }
                    }
                    g
                };
                let scalar = (0..n).map(|i| picture.base.data[i * n + i]).sum::<C64>() / n as f64;
                let slope = |u: f64, y: &[C64]| -> Vec<C64> {
                    let g = generator(u);
                    (0..n).map(|i| ((0..n).map(|j| g[i * n + j] * y[j]).sum::<C64>() - scalar * y[i]) * C64::new(0.0, -1.0)).collect()
                };
                let count = ((rk * tau / 0.948).ceil() as usize).max(20000);
                let dt = tau / count as f64;
                let mut y: Vec<C64> = launched[..n].to_vec();
                for i in 0..count {
                    let u = -h + i as f64 * dt;
                    let k1 = slope(u, &y);
                    let y2: Vec<C64> = y.iter().zip(&k1).map(|(a, b)| *a + *b * (0.5 * dt)).collect();
                    let k2 = slope(u + 0.5 * dt, &y2);
                    let y3: Vec<C64> = y.iter().zip(&k2).map(|(a, b)| *a + *b * (0.5 * dt)).collect();
                    let k3 = slope(u + 0.5 * dt, &y3);
                    let y4: Vec<C64> = y.iter().zip(&k3).map(|(a, b)| *a + *b * dt).collect();
                    let k4 = slope(u + dt, &y4);
                    for q in 0..n {
                        y[q] += (k1[q] + k2[q] * 2.0 + k3[q] * 2.0 + k4[q]) * (dt / 6.0);
                    }
                }
                let phase = (C64::new(0.0, -tau) * scalar).exp();
                y.iter_mut().for_each(|z| *z *= phase);
                picture.launch(-h, &launched[..n]);
                let (estimate, work) = picture.remainder();
                let us = (0..5)
                    .map(|_| {
                        let clock = std::time::Instant::now();
                        std::hint::black_box(picture.remainder());
                        clock.elapsed().as_secs_f64() * 1e6
                    })
                    .fold(f64::INFINITY, f64::min);
                let mut x = [C64::ZERO; STATES];
                picture.at(h, &mut x[..n]);
                let error = x[..n].iter().zip(&y).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt() / norm_sqr(&y).sqrt();
                println!("R1-REMAINDER {name} state {k} tau {tau}: n {n} error {error:.3e} E {estimate:.3e} E/error {:.3} | E's count {work:.0} complex MA, {us:.1} us ({:.2} G/s), the step's picture {:.0}", estimate / error, work / us * 1e-3, picture.work);
            }
            match Midway::new(&run, &t, walk).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere)) {
                Ok(step) => t = step.end,
                Err(_) => break,
            }
        }
    }
}

/// The probe and the pilot as the solve runs them (solve.rs `solve`), one scene at a time: every message's time,
/// trajectories and step; ended where the product's first batch is reported (more trajectories than the pilot's) or the
/// solve ends. With the feature `prof` the ladder's checks are noted as they are made (`PROFNOTE ladder ...`).
#[test]
#[ignore]
fn r1prop_solve() {
    let pilot = crate::solve::pilot_size();
    for (name, request) in scenes() {
        let clock = std::time::Instant::now();
        let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            crate::solve::solve(&request, &mut |s, remaining| {
                println!("R1-SOLVE {name} {:.1} s: {} trajectories, tau {} us, remaining {remaining:?}", clock.elapsed().as_secs_f64(), s.trajectories, s.time_step_us);
                if s.trajectories > pilot {
                    panic!("R1-SOLVE the product's first batch");
                }
            })
        }));
        let how = match ended {
            Ok(Ok(s)) => format!("solved: tau {} us, {} trajectories", s.time_step_us, s.trajectories),
            Ok(Err(e)) => format!("refused: {e}"),
            Err(_) => "the product began".to_string(),
        };
        println!("R1-SOLVE {name} ended at {:.1} s: {how}", clock.elapsed().as_secs_f64());
    }
}
