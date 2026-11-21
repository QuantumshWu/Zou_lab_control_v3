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
//! - `r1prop_quadrature`: the kick's one panel at the quotients' bandwidth against brute force, by kind of piece.
//! - `r1prop_jumps`: the jumps' times and the states they act on against the flow of the step's own generator.
//! - `r1prop_step_time`: a step's time on one thread.
//! - `r1prop_integration_time`: the stretch's and the picture's profile regions along a walk (feature `prof`).

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
fn halved(run: &Selected, t: &Trajectory, tau: f64, scratch: &mut Scratch, depth: usize, refusals: &mut [[usize; 5]]) -> Option<Trajectory> {
    use crate::spectral::Refusal;
    use crate::step::Refused;
    match Midway::new(run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], scratch, &Everywhere)) {
        Ok(step) => Some(step.end),
        Err(e) => {
            let why = match e {
                Refused::Propagation(Refusal::Decomposition) => 0,
                Refused::Propagation(Refusal::Rounding) => 1,
                Refused::Propagation(Refusal::Radius) => 2,
                Refused::Propagation(Refusal::Growth) => 3,
                _ => 4,
            };
            refusals[0][why] += 1;
            if why == 4 || depth == 0 {
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
            let each: Vec<([[usize; 5]; 4], u32, Option<String>)> = (0..count)
                .into_par_iter()
                .map(|i| {
                    let mut t = start(&run, &request, chief, i);
                    let mut scratch = Scratch::default();
                    let mut refusals = [[0usize; 5]; 4];
                    for k in 0..steps {
                        match halved(&run, &t, tau, &mut scratch, 3, &mut refusals) {
                            Some(next) => t = next,
                            None => return (refusals, t.jumps, Some(format!("trajectory {i} ended at step {k}"))),
                        }
                    }
                    (refusals, t.jumps, None)
                })
                .collect();
            let mut total = [[0usize; 5]; 4];
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
                "R1-REFUSED {name} tau {tau}: {count} trajectories x {steps} steps = {}; refused at tau [decomposition, rounding, radius, growth, other] {:?} ({:.3} %), on {on} trajectories; at tau/2 {:?}, tau/4 {:?}, tau/8 {:?}; {jumps} jumps; ended {}: {:?}; wall {:.1} s",
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
                    .fold(f64::INFINITY, crate::linalg::smaller);
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

/// The old rule's node count over a width at a bandwidth (until 2026-11-20, step.rs `plan`): rules of 2 to 64 nodes,
/// panels within each rule's phase x_m (Davis & Rabinowitz's bound at the unit round-off), the fewest nodes.
fn old_nodes(width: f64, bandwidth: f64) -> usize {
    let u = crate::units::UNIT_ROUNDOFF;
    let ln_factorial = |n: usize| (1..=n).map(|k| (k as f64).ln()).sum::<f64>();
    let mut least = usize::MAX;
    for m in 2..=64usize {
        let ln = ((2 * m + 1) as f64).ln() + 3.0 * ln_factorial(2 * m) - 4.0 * ln_factorial(m);
        let phase = ((u.ln() + ln) / (2 * m) as f64).exp();
        let panels = ((bandwidth * width.abs() / phase).ceil() as usize).max(1);
        least = least.min(m * panels);
        if panels == 1 {
            break;
        }
    }
    least
}

/// The kick's quadrature on a piece against brute force (NODES/CARRIER's nodesbench): the three nodes' Lagrange sums of
/// the densities x x†/‖x‖² over [a, b] (`Sums` without tones), by the engine's panels at the quotients' bandwidth
/// (`Picture::quotient`; `count`, a jump's root in the piece where `ends`) and by 32-node Gauss panels each 4 rad of it,
/// 8 more (`split` times as many: its self-check). Returns the sums of each, the nodes the engine's took and its panels.
fn piece_sums(picture: &mut crate::spectral::Picture, a: f64, b: f64, h: f64, split: usize, ends: bool) -> ([Vec<crate::linalg::C64>; 3], [Vec<crate::linalg::C64>; 3], usize, usize) {
    let (engine, nodes, panels) = engine_sums(picture, a, b, h, ends);
    (engine, brute_sums(picture, a, b, h, split), nodes, panels)
}

/// The engine's sums of `piece_sums`, its nodes and its panels.
fn engine_sums(picture: &mut crate::spectral::Picture, a: f64, b: f64, h: f64, ends: bool) -> ([Vec<crate::linalg::C64>; 3], usize, usize) {
    use crate::linalg::C64;
    let n = picture.states();
    let mut x = [C64::ZERO; super::super::STATES];
    let mut sums = vec![0.0; 2 * n * super::super::row(n) * super::super::SUMS];
    let mut nodes = 0;
    let panels = super::super::count(b - a, picture.quotient, super::super::FORMS, 2, || ends);
    for (u, w) in super::super::panels(a, b, panels).flat_map(|(_, nodes)| nodes) {
        picture.at(u, &mut x[..n]);
        super::super::Sums { still: false, h, tones: &[] }.gather(u, w, &x[..n], None, &mut sums);
        nodes += 1;
    }
    let engine: [Vec<C64>; 3] = std::array::from_fn(|j| {
        let mut m = vec![C64::ZERO; n * n];
        super::super::whole(&sums, j, n, &mut m);
        m
    });
    (engine, nodes, panels.0)
}

/// The brute force's sums of `piece_sums`.
fn brute_sums(picture: &mut crate::spectral::Picture, a: f64, b: f64, h: f64, split: usize) -> [Vec<crate::linalg::C64>; 3] {
    use crate::linalg::{norm_sqr, C64};
    let n = picture.states();
    let mut x = [C64::ZERO; super::super::STATES];
    let rule = crate::special::gauss_legendre(32);
    let count = split * ((picture.quotient * (b - a).abs() / 4.0).ceil() as usize + 8);
    let width = (b - a) / count as f64;
    let mut brute: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n * n]);
    for p in 0..count {
        for &(t, g) in &rule {
            let u = a + (p as f64 + 0.5 + 0.5 * t) * width;
            picture.at(u, &mut x[..n]);
            let norm = norm_sqr(&x[..n]);
            let s = u / h;
            let lagrange = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)];
            for (j, sum) in brute.iter_mut().enumerate() {
                let f = 0.5 * width * g * lagrange[j] / norm;
                for r in 0..n {
                    let xr = x[r] * f;
                    for c in 0..n {
                        sum[r * n + c] += xr * x[c].conj();
                    }
                }
            }
        }
    }
    brute
}

/// The integration to round-off (R1-PROP item 5; acceptance 2): per scene and τ of R1PROP_TAUS (0.3256, 0.948), a walk
/// from the moving point (trajectory 3) at that τ, R1PROP_STEPS steps (24); on each step's picture three kinds of
/// piece -- the step's start state over the step (plain), the state a jump of the step left over what was left of it
/// (after a jump), and the bare start (the trajectory's first step) --: the sums by the engine's one panel at ν_Q against
/// brute force, the worst per kind of S (per unit width), H = Σ_j tr(Γ_j S_j), K = Σ_j Re tr(Ĝ(s_j) S_j) over its
/// largest component, H(a, u*) to u* = a + 0.618 w on its own panel (the root's span); the brute force against itself
/// at twice its panels; the nodes against the old rule's at the forms' bandwidth; ν_Q/B; ν_Q's time and count.
#[test]
#[ignore]
fn r1prop_quadrature() {
    use crate::linalg::C64;
    use crate::step::Middle;
    use super::super::{pictured, predicted, Along, SUMS};
    let taus = list("R1PROP_TAUS", "0.3256,0.948");
    let steps = env("R1PROP_STEPS", 24.0) as usize;
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let h = 0.5 * tau;
            let mut t = start(&run, &request, chief, 3);
            let mut scratch = Scratch::default();
            // Per kind: pieces, worst [S, H, K, H(u*)], brute's self-check, nodes, old nodes, ν_Q/B sum, ν_Q µs and count;
            // and the root's layout (a jump's root in the piece: `count`): its worst [S, H, K], its panels and its nodes.
            let mut tally = [(0usize, [0.0f64; 4], 0.0f64, 0usize, 0usize, 0.0f64, f64::INFINITY, 0.0f64); 3];
            let mut rooted = [([0.0f64; 3], 0usize, 0usize); 3];
            for k in 0..steps {
                let Ok(midway) = Midway::new(&run, &t, tau) else { break };
                let taken = midway.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere);
                let n = t.psi.len();
                let kappa_m = predicted(&t.forms, midway.reached.1, tau, t.time + h);
                let along = Along::new(&midway.generator, kappa_m, kappa_m.map(|k| midway.hbar_over_mass * k));
                // The pieces: (kind, launch time from the middle, the frame's state there).
                let mut pieces = vec![(if k == 0 { 2 } else { 0 }, -h, along.turned(&t.psi, -h, true)[..n].to_vec())];
                if let Ok(step) = &taken {
                    if let Some(made) = step.made.iter().find(|m| m.kick.is_some() && m.state.cluster == t.cluster) {
                        let u = (made.state.time - t.time) - h;
                        if u < h {
                            pieces.push((1, u, along.turned(&made.state.psi, u, true)[..n].to_vec()));
                        }
                    }
                }
                for (kind, from, x) in pieces {
                    let mut s = Scratch::default();
                    s.fit(n, SUMS);
                    if pictured(&along, &mut s, tau, Some(&x)).is_err() {
                        continue;
                    }
                    let picture = &mut s.picture;
                    picture.launch(from, &x);
                    let (_, counted) = crate::spectral::tests::quotient_of(picture, from);
                    let us = (0..5)
                        .map(|_| {
                            let clock = std::time::Instant::now();
                            std::hint::black_box(crate::spectral::tests::quotient_of(picture, from));
                            clock.elapsed().as_secs_f64() * 1e6
                        })
                        .fold(f64::INFINITY, crate::linalg::smaller);
                    let (engine, brute, nodes, _) = piece_sums(picture, from, h, h, 1, false);
                    let check = brute_sums(picture, from, h, h, 2);
                    let inner = from + 0.618 * (h - from);
                    let (engine_in, brute_in, _, _) = piece_sums(picture, from, inner, h, 1, false);
                    let (engine_root, nodes_root, panels_root) = engine_sums(picture, from, h, h, true);
                    let width = (h - from).abs();
                    let apart = |x: &[Vec<C64>; 3], y: &[Vec<C64>; 3]| x.iter().zip(y).flat_map(|(a, b)| a.iter().zip(b).map(|(p, q)| (*p - *q).abs())).fold(0.0, crate::linalg::larger);
                    let picture = &*picture;
                    // Γ = i(G − G†) of the generator at the node, the picture's parabola of the static group.
                    let hazard = |s: &[Vec<C64>; 3]| {
                        let mut hz = 0.0;
                        for (sj, u) in s.iter().zip([-h, 0.0, h]) {
                            let at = |q: usize| picture.base.data[q] + picture.ends[0].data[q] * u + picture.ends[1].data[q] * (u * u);
                            for a in 0..n {
                                for b in 0..n {
                                    let gamma = C64::I * (at(a * n + b) - at(b * n + a).conj());
                                    hz += (gamma * sj[b * n + a]).re;
                                }
                            }
                        }
                        hz
                    };
                    let kick = |s: &[Vec<C64>; 3]| {
                        let mut f = [0.0; 3];
                        for (sj, u) in s.iter().zip([-h, 0.0, h]) {
                            let g = along.middle.force(&along.path, u, sj);
                            for c in 0..3 {
                                f[c] += g[c];
                            }
                        }
                        f
                    };
                    let (ke, kb) = (kick(&engine), kick(&brute));
                    let scale = crate::linalg::larger(kb.iter().map(|x| x.abs()).fold(0.0, crate::linalg::larger), f64::MIN_POSITIVE);
                    let errors = [apart(&engine, &brute) / width, (hazard(&engine) - hazard(&brute)).abs(), ke.iter().zip(&kb).map(|(x, y)| (x - y).abs()).fold(0.0, crate::linalg::larger) / scale, (hazard(&engine_in) - hazard(&brute_in)).abs()];
                    let entry = &mut tally[kind];
                    entry.0 += 1;
                    for (w, e) in entry.1.iter_mut().zip(errors) {
                        *w = crate::linalg::larger(*w, e);
                    }
                    entry.2 = crate::linalg::larger(entry.2, apart(&brute, &check) / width);
                    entry.3 += nodes;
                    entry.4 += old_nodes(h - from, picture.bandwidth).max(super::super::FORMS);
                    entry.5 += picture.quotient / crate::linalg::larger(picture.bandwidth, f64::MIN_POSITIVE);
                    entry.6 = crate::linalg::smaller(entry.6, us);
                    entry.7 = crate::linalg::larger(entry.7, counted);
                    let (kr, root) = (kick(&engine_root), &mut rooted[kind]);
                    let errors = [apart(&engine_root, &brute) / width, (hazard(&engine_root) - hazard(&brute)).abs(), kr.iter().zip(&kb).map(|(x, y)| (x - y).abs()).fold(0.0, crate::linalg::larger) / scale];
                    for (w, e) in root.0.iter_mut().zip(errors) {
                        *w = crate::linalg::larger(*w, e);
                    }
                    (root.1, root.2) = (root.1 + panels_root, root.2 + nodes_root);
                }
                match taken {
                    Ok(step) => t = step.end,
                    Err(_) => break,
                }
            }
            for (kind, (pieces, worst, check, nodes, old, ratio, us, counted)) in tally.iter().enumerate() {
                if *pieces == 0 {
                    continue;
                }
                let (root, panels, root_nodes) = rooted[kind];
                println!(
                    "R1-QUAD {name} tau {tau} {}: {pieces} pieces | worst S {:.1e} H {:.1e} K {:.1e} H(u*) {:.1e} | brute's check {check:.1e} | nodes {:.1} a piece against the old rule's {:.1} ({:.2}) | nu_Q/B {:.2} | nu_Q {us:.1} us, {counted:.0} counted | as a root's piece: {:.1} panels, {:.1} nodes, worst S {:.1e} H {:.1e} K {:.1e}",
                    ["plain", "after a jump", "start"][kind],
                    worst[0],
                    worst[1],
                    worst[2],
                    worst[3],
                    *nodes as f64 / *pieces as f64,
                    *old as f64 / *pieces as f64,
                    *nodes as f64 / *old as f64,
                    ratio / *pieces as f64,
                    panels as f64 / *pieces as f64,
                    root_nodes as f64 / *pieces as f64,
                    root[0],
                    root[1],
                    root[2]
                );
            }
        }
    }
}

/// The jumps' times and the states they act on (acceptance 3): per scene and τ of R1PROP_TAUS (0.326, 0.163), a walk
/// from the moving point (trajectory 3) at τ, R1PROP_STEPS steps (12); at each, its start state's step with the jump's
/// threshold set where the flow of the step's own generator -- the picture's parabola and tones, RK4 at R1PROP_RK steps
/// over 0.948 µs (400000), its norm the survival -- crosses it at 0.6 of the step: the jump's time as the step makes it
/// against the flow's root, as ln N's error there (the time's error times the hazard), and the state the jump acts on
/// (the propagation's at the step's root, its direction) against the flow's direction there.
#[test]
#[ignore]
fn r1prop_jumps() {
    use crate::linalg::{norm_sqr, C64};
    use super::super::{pictured, predicted, Along, STATES, SUMS};
    let taus = list("R1PROP_TAUS", "0.326,0.163");
    let (steps, rk) = (env("R1PROP_STEPS", 12.0) as usize, env("R1PROP_RK", 400000.0));
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let h = 0.5 * tau;
            let mut t = start(&run, &request, chief, 3);
            let mut scratch = Scratch::default();
            let (mut worst_time, mut worst_state, mut count) = (0.0_f64, 0.0_f64, 0);
            for _ in 0..steps {
                let Ok(midway) = Midway::new(&run, &t, tau) else { break };
                let n = t.psi.len();
                let kappa_m = predicted(&t.forms, midway.reached.1, tau, t.time + h);
                let along = Along::new(&midway.generator, kappa_m, kappa_m.map(|k| midway.hbar_over_mass * k));
                let launched = along.turned(&t.psi, -h, true);
                let mut s = Scratch::default();
                s.fit(n, SUMS);
                if pictured(&along, &mut s, tau, Some(&launched[..n])).is_ok() {
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
                    let slope = |u: f64, y: &[C64]| -> Vec<C64> {
                        let g = generator(u);
                        (0..n).map(|i| (0..n).map(|j| g[i * n + j] * y[j]).sum::<C64>() * C64::new(0.0, -1.0)).collect()
                    };
                    // The flow from the launch, its states kept along: (u, x(u)).
                    let count_rk = ((rk * tau / 0.948).ceil() as usize).max(20000);
                    let dt = tau / count_rk as f64;
                    let mut y: Vec<C64> = launched[..n].to_vec();
                    let mut path = vec![(-h, y.clone())];
                    for i in 0..count_rk {
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
                        path.push((u + dt, y.clone()));
                    }
                    // The threshold where the flow's survival is at 0.6 of the step (its norm from the start's).
                    let first = norm_sqr(&launched[..n]);
                    let mark = (0.6 * count_rk as f64) as usize;
                    let theta = norm_sqr(&path[mark].1) / first * norm_sqr(&t.psi);
                    if theta < norm_sqr(&t.psi) * (1.0 - 1e-9) {
                        let mut from = t.clone();
                        from.threshold = theta;
                        if let Ok(step) = midway.attempt(&from, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere) {
                            if let Some(made) = step.made.iter().find(|m| m.kick.is_some()) {
                                let u = (made.state.time - t.time) - h;
                                // The flow's root: where its survival is θ, between the kept states, linear in ln N.
                                let log = |x: &[C64]| (norm_sqr(x) / first * norm_sqr(&t.psi)).ln();
                                let k = path.iter().position(|(_, x)| log(x) <= theta.ln()).unwrap_or(path.len() - 1).max(1);
                                let (l0, l1) = (log(&path[k - 1].1), log(&path[k].1));
                                let root = path[k - 1].0 + (theta.ln() - l0) / (l1 - l0) * dt;
                                let rate = (l0 - l1) / dt;
                                // The states the jump acts on: the propagation's at its root, the flow's at the flow's root.
                                picture.launch(-h, &launched[..n]);
                                let mut x = [C64::ZERO; STATES];
                                picture.at(u, &mut x[..n]);
                                let j = ((root + h) / dt).round() as usize;
                                let reference = &path[j.min(path.len() - 1)].1;
                                let (nx, nr) = (norm_sqr(&x[..n]).sqrt(), norm_sqr(reference).sqrt());
                                let overlap: C64 = x[..n].iter().zip(reference).map(|(a, b)| a.conj() * *b).sum();
                                let phase = if overlap.abs() > 0.0 { overlap / overlap.abs() } else { C64::ONE };
                                let apart = x[..n].iter().zip(reference).map(|(a, b)| (*a * phase / nx - *b / nr).norm_sqr()).sum::<f64>().sqrt();
                                worst_time = crate::linalg::larger(worst_time, ((u - root) * rate).abs());
                                worst_state = crate::linalg::larger(worst_state, apart);
                                count += 1;
                            }
                        }
                    }
                }
                match midway.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere) {
                    Ok(step) => t = step.end,
                    Err(_) => break,
                }
            }
            println!("R1-JUMP {name} tau {tau}: {count} jumps; the time's error as ln N's, at the most {worst_time:.2e}; the state's direction at the most {worst_state:.2e} (the flow's states {:.1e} µs apart)", tau / (crate::linalg::larger((rk * tau / 0.948).ceil(), 20000.0)));
        }
    }
}

/// A step's time (the brief's tilt0, tilt2; on one thread, the caller pins it): from the moving point (trajectory 3) a
/// burn-in of R1PROP_BURN steps (5) at each τ of R1PROP_TAUS (0.326, 0.948), then R1PROP_STEPS steps (10) timed, run
/// R1PROP_REPEAT times (5) from the same state on a fresh scratch (the same steps to the bit): the least time a step.
#[test]
#[ignore]
fn r1prop_step_time() {
    let taus = list("R1PROP_TAUS", "0.326,0.948");
    let (burn, steps, repeats) = (env("R1PROP_BURN", 5.0) as usize, env("R1PROP_STEPS", 10.0) as usize, env("R1PROP_REPEAT", 5.0) as usize);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let mut t0 = start(&run, &request, chief, 3);
            let mut scratch = Scratch::default();
            for _ in 0..burn {
                if let Ok(step) = Midway::new(&run, &t0, tau).and_then(|m| m.attempt(&t0, &Brownian::STILL, &[], &mut scratch, &Everywhere)) {
                    t0 = step.end;
                }
            }
            let mut best = f64::INFINITY;
            let mut jumps = 0;
            for _ in 0..repeats {
                let (mut t, mut s) = (t0.clone(), Scratch::default());
                let clock = std::time::Instant::now();
                jumps = 0;
                for _ in 0..steps {
                    match Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut s, &Everywhere)) {
                        Ok(step) => {
                            jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
                            t = step.end;
                        }
                        Err(e) => {
                            println!("R1-TIME {name} tau {tau}: refused {e:?}");
                            break;
                        }
                    }
                }
                best = crate::linalg::smaller(best, clock.elapsed().as_secs_f64() * 1e3 / steps as f64);
            }
            println!("R1-TIME {name} tau {tau}: {best:.2} ms a step (the least of {repeats} runs of {steps} steps, {jumps} jumps)");
        }
    }
}

/// The quotients' bandwidth along walks (a diagnosis): R1PROP_TRAJ trajectories (8) from the moving point, each
/// R1PROP_STEPS steps (200) at each τ of R1PROP_TAUS (0.948), on all the threads, a refused step taken at its halves:
/// per trajectory the largest ν_Q/B of a step's last launch and the slowest step's time.
#[test]
#[ignore]
fn r1prop_quotient_scan() {
    use rayon::prelude::*;
    let taus = list("R1PROP_TAUS", "0.948");
    let (count, steps) = (env("R1PROP_TRAJ", 8.0) as u64, env("R1PROP_STEPS", 200.0) as usize);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let rows: Vec<String> = (0..count)
                .into_par_iter()
                .map(|i| {
                    let mut t = start(&run, &request, chief, i);
                    let mut scratch = Scratch::default();
                    let (mut ratio, mut slowest, mut at, mut total) = (0.0_f64, 0.0_f64, 0, 0.0);
                    for k in 0..steps {
                        let clock = std::time::Instant::now();
                        let mut refusals = [[0usize; 5]; 4];
                        let Some(next) = halved(&run, &t, tau, &mut scratch, 3, &mut refusals) else { break };
                        let ms = clock.elapsed().as_secs_f64() * 1e3;
                        total += ms;
                        let p = &scratch.picture;
                        ratio = crate::linalg::larger(ratio, p.quotient / crate::linalg::larger(p.bandwidth, f64::MIN_POSITIVE));
                        if ms > slowest {
                            (slowest, at) = (ms, k);
                        }
                        t = next;
                    }
                    format!("R1-SCAN {name} tau {tau} traj {i}: nu_Q/B at the most {ratio:.2}; the slowest step {slowest:.1} ms (step {at}); {:.2} ms a step on the mean", total / steps as f64)
                })
                .collect();
            for row in rows {
                println!("{row}");
            }
        }
    }
    eprintln!("{}", crate::prof::report());
}

/// The integration's time against the engine before it (R1-PROP's LARGEST root fix: the brief's budget ≤ 2.3 times),
/// with the feature `prof`: per scene and τ of R1PROP_TAUS (0.3256, 0.948), one trajectory from the moving point
/// (index 3) R1PROP_BURN steps (5) on, then R1PROP_STEPS steps (40) R1PROP_REPEAT times (3) from that state with a fresh
/// scratch, on one thread; the profile's report after the burn and after each repeat -- the stretch's and the picture's
/// regions read off between them (the integration: the stretch less its picture's making).
#[test]
#[ignore]
fn r1prop_integration_time() {
    let taus = list("R1PROP_TAUS", "0.3256,0.948");
    let (burn, steps, repeats) = (env("R1PROP_BURN", 5.0) as usize, env("R1PROP_STEPS", 40.0) as usize, env("R1PROP_REPEAT", 3.0) as usize);
    for (name, request) in scenes() {
        let (run, chief) = run_of(&request);
        for &tau in &taus {
            let walk = |count: usize, t: &mut Trajectory, scratch: &mut Scratch| -> (usize, usize, f64) {
                let clock = std::time::Instant::now();
                let (mut taken, mut jumps) = (0, 0);
                for _ in 0..count {
                    match Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], scratch, &Everywhere)) {
                        Ok(step) => {
                            jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
                            *t = step.end;
                            taken += 1;
                        }
                        Err(_) => break,
                    }
                }
                (taken, jumps, clock.elapsed().as_secs_f64())
            };
            let mut t0 = start(&run, &request, chief, 3);
            walk(burn, &mut t0, &mut Scratch::default());
            println!("R1-ITIME {name} {tau} burnt{}", crate::prof::report());
            for r in 0..repeats {
                let mut t = t0.clone();
                let (taken, jumps, secs) = walk(steps, &mut t, &mut Scratch::default());
                println!("R1-ITIME {name} {tau} repeat {r}: {taken} steps, {jumps} jumps, {:.4} s{}", secs, crate::prof::report());
            }
        }
    }
}
