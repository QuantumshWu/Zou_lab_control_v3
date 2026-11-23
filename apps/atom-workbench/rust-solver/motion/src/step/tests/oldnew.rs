//! OLDNEW (BRIEF_OLDNEW, 2026-11-23): the new engine's step and whole solve beside the old engine's (motion-3d at
//! c4e45326^, timed by its own `oldnew_parts`). Test builds only, no effect on any result. Ignored; run by name, one at
//! a time, the caller pinning the thread to one performance core for the times:
//! `<test exe> step::tests::oldnew::<name> --exact --include-ignored --nocapture --test-threads 1`.
//! - `oldnew_jumps`: per scene of OLDNEW (comma separated `name=model@tau`, as STEPCOST) a walk of OLDNEW_STEPS (200)
//!   steps from trajectory OLDNEW_INDEX (3) of `r1prop::start`; per state, each the least of OLDNEW_REPEAT (5) in the
//!   thread's cycles after one untimed call: the step as it falls (its jumps where they come) and the same step with no
//!   jump (the threshold at the least positive number), a node at the step's middle, and an output's readout at the
//!   point. Said: the means, the jumps a step, a jump's cost (the steps with jumps, their time less the same steps'
//!   without, over their jumps).
//! - `oldnew_solve`: a whole solve of OLDNEW_REQ (a request's name as `request_of` takes it) on the caller's threads:
//!   per stage message its wall and process CPU, and the solution's trajectories, τ and T at the end.
use std::time::Instant;

use crate::model::selected::Selected;
use crate::request::{Request, GROUND};
use crate::step::{Brownian, Dynamics};

use super::super::{Midway, Scratch};
use super::r1prop::start;
use super::Everywhere;

extern "system" {
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn QueryThreadCycleTime(thread: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
    fn GetCurrentProcess() -> *mut std::ffi::c_void;
    fn GetProcessTimes(process: *mut std::ffi::c_void, creation: *mut u64, exit: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
}

fn thread_cycles() -> u64 {
    let mut c = 0u64;
    // SAFETY: the current thread's pseudo handle and a u64 to write.
    let ok = unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut c) };
    assert!(ok != 0, "QueryThreadCycleTime");
    c
}

/// The process's CPU time so far (s), its threads' user and kernel time.
fn process_cpu() -> f64 {
    let (mut a, mut b, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
    // SAFETY: the current process's pseudo handle and four u64 to write.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut a, &mut b, &mut k, &mut u) };
    (k + u) as f64 * 1e-7
}

/// The cycles' rate (cycles a ns): the most over 200 busy spins of 2 ms (a short spin is seldom shared with another
/// thread on the core; the old engine's `oldnew_parts` takes the same).
fn rate() -> f64 {
    (0..200)
        .map(|_| {
            let (c0, clock) = (thread_cycles(), Instant::now());
            let mut x = 0u64;
            while clock.elapsed().as_micros() < 2000 {
                x = std::hint::black_box(x.wrapping_add(1));
            }
            (thread_cycles() - c0) as f64 / clock.elapsed().as_nanos() as f64
        })
        .fold(0.0, crate::linalg::larger)
}

/// A request by name: a native request's file in R1PROP_REQ, else a reference scene (as STEPCOST's).
fn request_of(name: &str) -> Request {
    let files = [("PGC3D", "PGC3D_1ms.json"), ("TET", "TET_1ms.json"), ("R4", "R4_LIN_4.json"), ("GMD1C", "GMD1C_1ms.json"), ("tilt0", "tilt0.json"), ("tilt2", "tilt2.json")];
    match files.iter().find(|(n, _)| *n == name) {
        Some((_, file)) => {
            let dir = std::path::PathBuf::from(std::env::var("R1PROP_REQ").expect("R1PROP_REQ: the requests' folder"));
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
            Request::from_native(&line).unwrap()
        }
        None => crate::model::testing::scene(name),
    }
}

const SCENES: &str = "PGC3D=ground@0.4741346189956611,TET=ground@0.4741346189956611,GMD1C=ground@0.9484855308393235,tilt0=ground@0.326,tilt2=ground@0.326,R4=+D2-2-1@0.23696682464454977";

#[test]
#[ignore]
fn oldnew_jumps() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (steps, repeat, index) = (env("OLDNEW_STEPS", 200), env("OLDNEW_REPEAT", 5), env("OLDNEW_INDEX", 3) as u64);
    let ghz = rate();
    eprintln!("OLDNEW-RATE {ghz:.4} cycles a ns");
    let least = |f: &mut dyn FnMut()| -> f64 {
        f();
        (0..repeat)
            .map(|_| {
                let c0 = thread_cycles();
                f();
                (thread_cycles() - c0) as f64 / (ghz * 1e3)
            })
            .fold(f64::INFINITY, crate::linalg::smaller)
    };
    // The core's speed: the most loops a thread-ns of the busy spin `rate` takes (one dependent add through memory) over
    // 200 spins of 2 ms -- a proxy of the core's clock, the old engine's harness the same loop (oldnew_parts.rs), to put
    // both on one clock.
    let speed = || -> f64 {
        (0..200)
            .map(|_| {
                let (c0, clock) = (thread_cycles(), Instant::now());
                let mut x = 0u64;
                while clock.elapsed().as_micros() < 2000 {
                    x = std::hint::black_box(x.wrapping_add(1));
                }
                x as f64 / ((thread_cycles() - c0) as f64 / ghz)
            })
            .fold(0.0, crate::linalg::larger)
    };
    let list = std::env::var("OLDNEW").unwrap_or_else(|_| SCENES.to_string());
    for entry in list.split(',') {
        eprintln!("OLDNEW-SPEED before {:.4} loops a ns", speed());
        let (name, rest) = entry.split_once('=').expect("name=model@tau");
        let (spec, tau) = rest.split_once('@').expect("name=model@tau");
        let tau: f64 = tau.parse().unwrap();
        let request = request_of(name);
        let mut keep: Vec<usize> = (0..GROUND).collect();
        if let Some(labels) = spec.strip_prefix('+') {
            keep.extend(labels.split('+').map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
        }
        keep.sort_unstable();
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let chief = crate::solve::chief(&request);
        let reference = crate::solve::alone(chief);
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap();
        let mut t = start(&run, &request, chief, index);
        let attempt = |t: &crate::step::Trajectory| Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere));
        // [step as it falls, step without a jump, node, readout]; the steps with jumps: their time, their time without, their jumps.
        let mut sum = [0.0f64; 4];
        let (mut taken, mut jumps, mut work) = (0usize, 0usize, 0.0);
        let (mut with, mut without, mut jumped) = (0.0, 0.0, 0usize);
        for _ in 0..steps {
            let Ok(step) = attempt(&t) else { break };
            let k = step.made.iter().filter(|m| m.kick.is_some()).count();
            jumps += k;
            work += step.work;
            let a = least(&mut || {
                std::hint::black_box(attempt(&t).is_ok());
            });
            let mut quiet = t.clone();
            quiet.threshold = f64::MIN_POSITIVE;
            let b = least(&mut || {
                std::hint::black_box(attempt(&quiet).is_ok());
            });
            sum[0] += a;
            sum[1] += b;
            if k > 0 {
                with += a;
                without += b;
                jumped += k;
            }
            let midway = Midway::new(&run, &t, tau).unwrap();
            let at = midway.reached;
            sum[2] += least(&mut || {
                std::hint::black_box(run.node(at.0, at.1, false));
            });
            sum[3] += least(&mut || {
                std::hint::black_box(run.readout(t.cluster, &t.point(), &t.psi));
            });
            taken += 1;
            t = step.end;
        }
        let k = taken.max(1) as f64;
        eprintln!(
            "OLDNEW-STEP {name} {spec} (P{}) tau {tau}: a step {:.1} µs, without a jump {:.1} µs | jumps a step {:.4}, a jump {:.1} µs ({jumped} jumps) | a node {:.1} µs | an output's readout {:.1} µs | counted {:.3e} real MA a step | per µs of atom time {:.1} µs, over {taken} steps",
            keep.len(),
            sum[0] / k,
            sum[1] / k,
            jumps as f64 / k,
            (with - without) / jumped.max(1) as f64,
            sum[2] / k,
            sum[3] / k,
            work / k,
            sum[0] / k / tau,
        );
        eprintln!("OLDNEW-SPEED after {:.4} loops a ns", speed());
    }
}

#[test]
#[ignore]
fn oldnew_solve() {
    let name = std::env::var("OLDNEW_REQ").expect("OLDNEW_REQ");
    let request = request_of(&name);
    // OLDNEW's budget check: the quadrature and the cuts at a tolerance in place of the round-off (step.rs `QUAD_TOL`).
    // And the scheme: a fixed rule of OLDNEW_FIXED_NODES nodes, three nodes a step (step.rs `FIXED_NODES`, `THREE_NODES`).
    if let Ok(x) = std::env::var("OLDNEW_FIXED_NODES") {
        crate::step::FIXED_NODES.store(x.parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
    }
    if std::env::var("OLDNEW_THREE_NODES").is_ok() {
        crate::step::THREE_NODES.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // And K split by its fast parts in every run of ground states alone (model/selected.rs `SPLIT_PARTS`).
    if std::env::var("OLDNEW_SPLIT_PARTS").is_ok() {
        crate::model::selected::SPLIT_PARTS.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    // And the folded blocks' Doppler series truncated (rs/fold.rs `DOPPLER_ORDERS`).
    if let Ok(x) = std::env::var("OLDNEW_DOPPLER_ORDERS") {
        crate::model::rs::fold::DOPPLER_ORDERS.store(x.parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
    }
    // And the runs' streams under another seed (random.rs `RUN_SEED`).
    if let Ok(x) = std::env::var("OLDNEW_SEED") {
        crate::random::RUN_SEED.store(x.parse().unwrap(), std::sync::atomic::Ordering::Relaxed);
    }
    if let Ok(x) = std::env::var("OLDNEW_QUAD_TOL") {
        crate::step::QUAD_TOL.store(x.parse::<f64>().unwrap().to_bits(), std::sync::atomic::Ordering::Relaxed);
    }
    *crate::solve::OUTPUT_LOOKS.lock().unwrap() = (0., 0, 0);
    let (clock, cpu0) = (Instant::now(), process_cpu());
    let mut last = String::new();
    let solution = crate::solve::solve(&request, &mut |m| {
        if let crate::solve::Message::Stage(s) = m {
            let said = format!("{:?} tau {:?} batch {:?}", s.step, s.tau, s.batch.as_ref().map(|b| (b.number, b.trajectories, b.want)));
            if said != last {
                eprintln!("OLDNEW-STAGE {:8.2} s wall, {:8.2} s CPU: {said}", clock.elapsed().as_secs_f64(), process_cpu() - cpu0);
                last = said;
            }
        }
    })
    .unwrap();
    let (wall, cpu) = (clock.elapsed().as_secs_f64(), process_cpu() - cpu0);
    let end = solution.temperature_uk.len() - 1;
    // The readings at the end and their estimated errors: F = 1's and F = 2's populations, the loss, T_total.
    let p = &solution.populations[end];
    let f1: f64 = (0..3).map(|i| p[i]).sum();
    let f2: f64 = (3..8).map(|i| p[i]).sum();
    eprintln!(
        "OLDNEW-READ {name}: F=1 {f1:.5}, F=2 {f2:.5} (estimated error {:.3e}); loss {:.4e}; T_total {:.3} ± {:.3} µK; ledger totals {:?}",
        solution.estimated_errors[end],
        solution.loss_probability[end],
        solution.temperature_total_uk[end],
        solution.temperature_total_errors_uk[end].unwrap_or(f64::NAN),
        solution.approximations.totals
    );
    // The last output's statistics for the seeds check: T per axis and total with their standard errors, the rows as
    // shown (here the held populations, no loss) and the loss with theirs (solve.rs `LAST_ERRORS`), the groups.
    let errors = crate::solve::LAST_ERRORS.lock().unwrap().clone();
    let te = &solution.temperature_errors_uk[end];
    eprintln!(
        "OLDNEW-STAT {name} seed {}: T {:?} ± {:?} total {:.4} ± {:.4}; rows {:?} ± {:?}; loss {:.4e} ± {:.3e}; groups {}; trajectories {}; tau {:.5}",
        std::env::var("OLDNEW_SEED").unwrap_or_else(|_| "0".into()),
        solution.temperature_uk[end].map(|x| (x * 1e4).round() / 1e4),
        te.map(|x| x.map(|y| (y * 1e4).round() / 1e4)),
        solution.temperature_total_uk[end],
        solution.temperature_total_errors_uk[end].unwrap_or(f64::NAN),
        p.iter().take(crate::request::GROUND).map(|x| (x * 1e5).round() / 1e5).collect::<Vec<_>>(),
        errors.iter().take(crate::request::GROUND).map(|x| (x * 1e5).round() / 1e5).collect::<Vec<_>>(),
        solution.loss_probability[end],
        errors.get(crate::request::STATES).copied().unwrap_or(f64::NAN),
        errors.last().copied().unwrap_or(f64::NAN),
        solution.trajectories,
        solution.time_step_us
    );
    // The product's groups' sums at the last output (solve.rs `LAST_GROUPS`), one JSON line a group, to OLDNEW_GROUPS.
    if let Ok(path) = std::env::var("OLDNEW_GROUPS") {
        use std::io::Write;
        let groups = crate::solve::LAST_GROUPS.lock().unwrap().clone();
        let mut f = std::fs::File::create(&path).unwrap();
        for g in &groups {
            let line = serde_json::json!({
                "trajectories": g.trajectories, "failed": g.failed, "held": g.held, "rows": &g.rows[..crate::request::GROUND],
                "kappa": g.kappa, "kappa_squared": g.kappa_squared, "kappa2": g.kappa2,
                "shape": g.shape.iter().map(|x| x.to_vec()).collect::<Vec<_>>(), "shape_squared": g.shape_squared
            });
            writeln!(f, "{line}").unwrap();
        }
        eprintln!("OLDNEW-GROUPS {name}: {} groups to {path}", groups.len());
    }
    // The outputs' looks in the product (solve.rs `OUTPUT_LOOKS`): their wall time, the batches, the looks put beyond.
    let (seconds, batches, beyond) = *crate::solve::OUTPUT_LOOKS.lock().unwrap();
    eprintln!("OLDNEW-OUTPUTS {name}: the outputs' looks {seconds:.3} s wall over {batches} reads of the product, {beyond} looks put beyond in all");
    // The ledger as the result says it: its items, what it flags, its note.
    let ledger = &solution.approximations;
    for item in &ledger.items {
        eprintln!("OLDNEW-ITEM {name}: {} = {:.4e}", item.label, item.estimate);
    }
    eprintln!("OLDNEW-FLAG {name}: {}", ledger.flagged.as_deref().unwrap_or("(none)"));
    eprintln!(
        "OLDNEW-ERRORS {name}: the shown error at the outputs, its largest {:.4e} at output {}, mean {:.4e}; T's error bars at the end {:?}",
        solution.estimated_errors.iter().copied().fold(0.0, f64::max),
        solution.estimated_errors.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map_or(0, |x| x.0),
        solution.estimated_errors.iter().sum::<f64>() / solution.estimated_errors.len() as f64,
        solution.temperature_total_errors_uk[end]
    );
    eprintln!(
        "OLDNEW-SOLVE {name}: wall {wall:.2} s, CPU {cpu:.2} s, threads {}, {} trajectories, tau {:.5} µs; T at the end {:?} µK, total {:.2}; loss {:.3e}",
        rayon::current_num_threads(),
        solution.trajectories,
        solution.time_step_us,
        solution.temperature_uk[end],
        solution.temperature_total_uk[end],
        solution.loss_probability[end]
    );
}
