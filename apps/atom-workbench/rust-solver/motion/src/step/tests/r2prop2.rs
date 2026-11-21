//! R2-PROP2's measurements on the mainline model (Scene -> choose -> Selected, the solve's `run_of`): the six scenes'
//! requests (R1PROP_REQ, R1PROP_SCENE as `r1prop`) and the app's default scene (R2PROP2_DEFAULT: its native request's
//! path, named S1; R2PROP2_SCENES the ones run, all where unset). Ignored; run by name, one at a time:
//! `<test exe> step::tests::r2prop2::<name> --ignored --nocapture --test-threads 1`.
//! - `r2prop2_census`: a walk's pictures, class by class (the tones' beats, the classes followed and hopped).
//! - `r2prop2_save`, `r2prop2_single`: walks' states saved by their bits, and one step from each on a fresh scratch by
//!   the bits of its end -- two builds compared on the same states (a change's regression: to the bit, or how far off).
//! - `r2prop2_time`: a step's time on one thread, with the profile's regions under the feature `prof`.

use crate::model::selected::Selected;
use crate::request::Request;
use crate::step::{Brownian, Trajectory};

use super::super::{Midway, Scratch};
use super::r1prop::{env, list, run_of, start};
use super::Everywhere;

/// The scenes: R1PROP_REQ's six (`r1prop::scenes`) and S1 (R2PROP2_DEFAULT), R2PROP2_SCENES the ones run (comma
/// separated; all where unset).
pub(super) fn scenes() -> Vec<(&'static str, Request)> {
    let only = std::env::var("R2PROP2_SCENES").ok();
    let wanted = |name: &str| only.as_deref().is_none_or(|o| o.split(',').any(|x| x == name));
    let mut out: Vec<(&'static str, Request)> = if std::env::var("R1PROP_REQ").is_ok() { super::r1prop::scenes().into_iter().filter(|(n, _)| wanted(n)).collect() } else { vec![] };
    if let (Ok(path), true) = (std::env::var("R2PROP2_DEFAULT"), wanted("S1")) {
        let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        line["options"]["device"] = serde_json::json!("cpu");
        out.push(("S1", Request::from_native(&line).unwrap()));
    }
    out
}

/// A preparation well as the solve's (solve.rs `Prepared`, private there): the tweezer's potential on a basis state's
/// populations, the trap of its potential (`solve::reported`: every light's pattern its scale, where the solve's takes the
/// trap light's -- the same bottom), and the trap light's beams with the potential's peak.
struct Prepared<'a> {
    run: &'a Selected,
    potential: crate::model::selected::Potential<'a>,
    trap: crate::model::Trap,
}
impl crate::ensemble::Well for Prepared<'_> {
    fn potential(&self, r: [f64; 3]) -> (f64, [f64; 3]) {
        let q = self.potential.at(r);
        (q.value, q.gradient)
    }
    fn trap(&self) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
        (self.trap.centre, self.trap.axes, self.trap.frequencies)
    }
    fn envelope(&self) -> (Vec<crate::field::Geometry>, f64) {
        let light = self.run.light();
        (light.frequencies[light.trap].beams.iter().map(|&l| light.beams[l]).collect(), self.potential.peak())
    }
}

/// The trajectories as the solve starts them (solve.rs `run`): the initial ensemble in its components' wells
/// (ensemble.rs `Ensemble::draw` at the solve's seed), trajectory i its draw i.
fn solve_starts(run: &Selected, request: &Request, count: u64) -> Vec<Trajectory> {
    use crate::solve::{alone, components, SEED};
    let hm = crate::units::hbar_over_mass(request.mass_kg);
    let initial = components(request);
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared { run, potential: run.prepared(&alone(s)), trap: crate::solve::reported(run, &alone(s)).unwrap() }).collect();
    let pairs: Vec<(usize, f64, &dyn crate::ensemble::Well)> = initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn crate::ensemble::Well)).collect();
    let ensemble = crate::ensemble::Ensemble::new(&pairs, request.temperature_uk, hm).unwrap();
    (0..count).map(|i| {
        let start = ensemble.draw(SEED, i);
        Trajectory::new(run, start.state, start.r, start.kappa, SEED, i)
    }).collect()
}

/// A step from a state on its scratch: the step (none where refused, with why).
fn stepped(run: &Selected, t: &Trajectory, tau: f64, scratch: &mut Scratch) -> Result<crate::step::Step, crate::step::Refused> {
    Midway::new(run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], scratch, &Everywhere))
}

/// A trajectory's state in words: every field's numbers by their bits (hex), space separated.
fn saved(t: &Trajectory) -> String {
    let mut w: Vec<String> = Vec::new();
    let mut f = |x: f64| w.push(format!("{:016x}", x.to_bits()));
    t.r.iter().chain(&t.kappa).for_each(|x| f(*x));
    f(t.psi.len() as f64);
    t.psi.iter().for_each(|z| {
        f(z.re);
        f(z.im)
    });
    f(t.cluster as f64);
    f(t.time);
    f(t.threshold);
    f(f64::from(t.jumps));
    t.scattered.iter().for_each(|x| f(*x));
    f(t.lost.unwrap_or(f64::NAN));
    f(if t.lost.is_some() { 1.0 } else { 0.0 });
    f(if t.unresolved { 1.0 } else { 0.0 });
    f(t.seed as f64);
    f(t.index as f64);
    f(t.work);
    f(t.jumped);
    t.pending.iter().flatten().for_each(|x| f(*x));
    t.booked.iter().for_each(|x| f(*x));
    f(t.forms.time);
    t.forms.held.iter().for_each(|x| f(*x));
    f(t.forms.tones.len() as f64);
    for (nu, a) in &t.forms.tones {
        f(*nu);
        a.iter().for_each(|z| {
            f(z.re);
            f(z.im)
        });
    }
    w.join(" ")
}

/// A trajectory's state from its words (`saved`).
fn loaded(words: &str) -> Trajectory {
    use crate::linalg::C64;
    let mut it = words.split_whitespace().map(|w| f64::from_bits(u64::from_str_radix(w, 16).unwrap()));
    let mut g = || it.next().expect("a saved state's number");
    let r = [g(), g(), g()];
    let kappa = [g(), g(), g()];
    let len = g() as usize;
    let psi: crate::step::Internal = (0..len).map(|_| C64::new(g(), g())).collect();
    let (cluster, time, threshold, jumps) = (g() as usize, g(), g(), g() as u32);
    let scattered = [g(), g()];
    let (lost, is_lost, unresolved) = (g(), g(), g());
    let (seed, index, work, jumped) = (g() as u64, g() as u64, g(), g());
    let pending = std::array::from_fn(|_| [g(), g(), g()]);
    let booked = [g(), g(), g()];
    let (ftime, held) = (g(), [g(), g(), g()]);
    let tones = (0..g() as usize).map(|_| (g(), std::array::from_fn(|_| C64::new(g(), g())))).collect();
    Trajectory { r, kappa, psi, cluster, time, threshold, jumps, scattered, lost: (is_lost == 1.0).then_some(lost), unresolved: unresolved == 1.0, seed, index, work, jumped, pending, booked, forms: crate::step::Forms { time: ftime, held, tones } }
}

/// The states of R2PROP2_STATES (a folder): the file of a scene and τ.
fn states_file(name: &str, tau: f64) -> std::path::PathBuf {
    std::path::PathBuf::from(std::env::var("R2PROP2_STATES").expect("R2PROP2_STATES: the states' folder")).join(format!("{name}_{tau}.txt"))
}

/// Walks saved for single steps (`r2prop2_single`): R2PROP2_TRAJ trajectories (8) from the moving point -- the solve's
/// starts where R2PROP2_ENSEMBLE is set (`solve_starts`) --, each R2PROP2_STEPS steps (12) at each τ of R2PROP2_TAUS
/// (0.474), on all the threads; every step's start state into the scene's and τ's file (`states_file`); a refused step
/// ends the trajectory's walk.
#[test]
#[ignore]
fn r2prop2_save() {
    use rayon::prelude::*;
    let taus = list("R2PROP2_TAUS", "0.474");
    let (count, steps) = (env("R2PROP2_TRAJ", 8.0) as u64, env("R2PROP2_STEPS", 12.0) as usize);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        let starts: Vec<Trajectory> = if std::env::var("R2PROP2_ENSEMBLE").is_ok() { solve_starts(&run, &request, count) } else { (0..count).map(|i| start(&run, &request, chief, i)).collect() };
        for &tau in &taus {
            let each: Vec<Vec<String>> = (0..count)
                .into_par_iter()
                .map(|i| {
                    let mut t = starts[i as usize].clone();
                    let mut scratch = Scratch::default();
                    let mut lines = Vec::new();
                    for k in 0..steps {
                        lines.push(format!("{i} {k} {}", saved(&t)));
                        match stepped(&run, &t, tau, &mut scratch) {
                            Ok(step) => t = step.end,
                            Err(_) => break,
                        }
                    }
                    lines
                })
                .collect();
            let lines: Vec<String> = each.into_iter().flatten().collect();
            std::fs::write(states_file(name, tau), lines.join("\n") + "\n").unwrap();
            println!("R2-SAVE {name} tau {tau}: {} states", lines.len());
        }
    }
}

/// One step from each saved state (`r2prop2_save`), at each factor of R2PROP2_FACTORS (1, 0.5) times the walk's τ, on a
/// fresh scratch each, on all the threads: per state the end's numbers by their bits -- ψ, κ, r, the threshold, the
/// jumps, the momentum's kick and jumps' parts, and the times of the states made -- with ‖ψ‖² and the kick in words.
#[test]
#[ignore]
fn r2prop2_single() {
    use rayon::prelude::*;
    let taus = list("R2PROP2_TAUS", "0.474");
    let factors = list("R2PROP2_FACTORS", "1,0.5");
    for (name, request) in scenes() {
        let (run, _) = run_of(&request);
        for &tau in &taus {
            let text = std::fs::read_to_string(states_file(name, tau)).unwrap();
            let states: Vec<(String, Trajectory)> = text.lines().map(|l| {
                let mut parts = l.splitn(3, ' ');
                let (i, k) = (parts.next().unwrap(), parts.next().unwrap());
                (format!("{i} {k}"), loaded(parts.next().unwrap()))
            }).collect();
            for &factor in &factors {
                let rows: Vec<String> = states
                    .par_iter()
                    .map(|(at, t)| {
                        let step = tau * factor;
                        match stepped(&run, t, step, &mut Scratch::default()) {
                            Ok(s) => {
                                let m = s.momentum();
                                let bits = |x: &[f64]| x.iter().map(|x| format!("{:016x}", x.to_bits())).collect::<Vec<_>>().join(",");
                                let psi: Vec<f64> = s.end.psi.iter().flat_map(|z| [z.re, z.im]).collect();
                                let made: Vec<f64> = s.made.iter().map(|m| m.state.time).collect();
                                let norm: f64 = s.end.psi.iter().map(|z| z.norm_sqr()).sum();
                                format!("R2-SINGLE {name} tau {tau} factor {factor} at {at} ok norm {norm:.17e} kick {:.17e},{:.17e},{:.17e} jumps {} | psi {} kappa {} r {} threshold {} kick {} jumped {} made {}", m.kick[0], m.kick[1], m.kick[2], s.end.jumps, bits(&psi), bits(&s.end.kappa), bits(&s.end.r), bits(&[s.end.threshold]), bits(&m.kick), bits(&m.jumps), bits(&made))
                            }
                            Err(e) => format!("R2-SINGLE {name} tau {tau} factor {factor} at {at} refused {e:?}"),
                        }
                    })
                    .collect();
                for row in rows {
                    println!("{row}");
                }
            }
        }
    }
}

/// A step's time on one thread (the caller pins it): from the moving point (trajectory R2PROP2_TRAJ, 3) a burn-in of
/// R2PROP2_BURN steps (5) at each τ of R2PROP2_TAUS (0.326), then R2PROP2_STEPS steps (10) timed, run R2PROP2_REPEAT
/// times (5) from the same state on a fresh scratch (the same steps to the bit): the least time a step, and with the
/// feature `prof` the regions of the least run.
#[test]
#[ignore]
fn r2prop2_time() {
    let taus = list("R2PROP2_TAUS", "0.326");
    let (burn, steps, repeats, traj) = (env("R2PROP2_BURN", 5.0) as usize, env("R2PROP2_STEPS", 10.0) as usize, env("R2PROP2_REPEAT", 5.0) as usize, env("R2PROP2_TRAJ", 3.0) as u64);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let mut t0 = start(&run, &request, chief, traj);
            let mut scratch = Scratch::default();
            for _ in 0..burn {
                if let Ok(step) = stepped(&run, &t0, tau, &mut scratch) {
                    t0 = step.end;
                }
            }
            let (mut best, mut report, mut runs) = (f64::INFINITY, String::new(), Vec::new());
            let (mut jumps, mut taken) = (0, 0);
            let _ = crate::prof::report();
            for _ in 0..repeats {
                let (mut t, mut s) = (t0.clone(), Scratch::default());
                let before = crate::prof::report();
                let clock = std::time::Instant::now();
                (jumps, taken) = (0, 0);
                for _ in 0..steps {
                    match stepped(&run, &t, tau, &mut s) {
                        Ok(step) => {
                            jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
                            taken += 1;
                            t = step.end;
                        }
                        Err(e) => {
                            println!("R2-TIME {name} tau {tau}: refused {e:?}");
                            break;
                        }
                    }
                }
                let us = clock.elapsed().as_secs_f64() * 1e6 / taken.max(1) as f64;
                runs.push(format!("{us:.0}"));
                let after = crate::prof::report();
                if us < best {
                    (best, report) = (us, format!("before{before}\nafter{after}"));
                }
            }
            println!("R2-TIME {name} tau {tau}: {best:.1} us a step (the least of {repeats} runs [{}] of {taken} steps, {jumps} jumps){}", runs.join(" "), if report.contains("PROF") { format!("\n{report}") } else { String::new() });
        }
    }
}

/// A walk's pictures class by class: from the moving point (trajectory R2PROP2_TRAJ, 3), R2PROP2_STEPS steps (4) at each
/// τ of R2PROP2_TAUS (0.326), the picture of each step's last stretch (`spectral::tests::census`).
#[test]
#[ignore]
fn r2prop2_census() {
    let taus = list("R2PROP2_TAUS", "0.326");
    let (steps, traj) = (env("R2PROP2_STEPS", 4.0) as usize, env("R2PROP2_TRAJ", 3.0) as u64);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let mut t = start(&run, &request, chief, traj);
            let mut scratch = Scratch::default();
            for k in 0..steps {
                match stepped(&run, &t, tau, &mut scratch) {
                    Ok(step) => {
                        t = step.end;
                        println!("R2-CENSUS {name} tau {tau} step {k}: {}", crate::spectral::tests::census(&scratch.picture));
                    }
                    Err(e) => {
                        println!("R2-CENSUS {name} tau {tau} step {k}: refused {e:?}");
                        break;
                    }
                }
            }
        }
    }
}
