//! STEP's measurement of a step's cost and its parts (BRIEF_IMPL, line STEP, phase 2: "每步代价，单线程钉核，七个场景修前
//! 修后"; study C's `studyc_step`, 81549036, with the parts the phase's items change). Test builds only, no effect on
//! any result.
//! - `stepcost`: per scene of STEPCOST (comma separated, each `name` or `name=model@tau`; default the seven scenes and
//!   R4's P9) its request (a native request's file in R1PROP_REQ, else a reference scene), its model ("ground", or "+"
//!   and explicit excited states' labels joined by "+"), its step τ (µs); a walk of STEPCOST_STEPS (12) steps from
//!   trajectory STEPCOST_INDEX (3) of `r1prop::start` (the chief state moving at the thermal speed). Per state of the walk, each the least of
//!   STEPCOST_REPEAT (5) in the thread's own cycles (another process's share of the core does not enter them), its first
//!   call not timed: the step on a fresh scratch; a node at the step's middle plain, with the photon forms, and a jump's
//!   photon's alone (`Selected::photon_node`, where the build has it); an
//!   output's readout at the point (`Dynamics::readout`) and on the step's middle (`Middle::readout`, where the model
//!   reads there). Said: their means over the walk, the jumps and the nodes a step makes, the counted work a step.
//!   Ignored; run by name, pinned to one performance core by the caller:
//!   `<test exe> step::tests::stepcost::stepcost --exact --include-ignored --nocapture --test-threads 1`.
use std::time::Instant;

use crate::model::selected::Selected;
use crate::request::{Request, GROUND};
use crate::step::{Brownian, Dynamics, Middle};

use super::super::{Midway, Scratch};
use super::r1prop::start;
use super::Everywhere;

extern "system" {
    fn GetCurrentThread() -> *mut std::ffi::c_void;
    fn QueryThreadCycleTime(thread: *mut std::ffi::c_void, cycles: *mut u64) -> i32;
}

/// This thread's CPU cycles so far (`QueryThreadCycleTime`: counted only while the thread runs).
fn thread_cycles() -> u64 {
    let mut c = 0u64;
    // SAFETY: the current thread's pseudo handle and a u64 to write.
    let ok = unsafe { QueryThreadCycleTime(GetCurrentThread(), &mut c) };
    assert!(ok != 0, "QueryThreadCycleTime");
    c
}

/// A request by name: a native request's file in R1PROP_REQ (the scenes study C and T2STEP took), else a reference
/// scene.
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

/// The default scenes: study C's table (C.report.md §2.3), each its model and step.
const SCENES: &str = "PGC3D=ground@0.4741346189956611,TET=ground@0.4741346189956611,GMD1C=ground@0.9484855308393235,tilt0=ground@0.326,tilt2=ground@0.326,R4=ground@0.11848341232227488,R4=+D2-2-1@0.23696682464454977,R5_PUMP=ground@2.782637082313045";

#[test]
#[ignore]
fn stepcost() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (steps, repeat, index) = (env("STEPCOST_STEPS", 12), env("STEPCOST_REPEAT", 5), env("STEPCOST_INDEX", 3) as u64);
    // The cycles' rate: the most cycles a ns over five busy 50 ms spins.
    let ghz: f64 = (0..5)
        .map(|_| {
            let (c0, clock) = (thread_cycles(), Instant::now());
            let mut x = 0u64;
            while clock.elapsed().as_millis() < 50 {
                x = std::hint::black_box(x.wrapping_add(1));
            }
            (thread_cycles() - c0) as f64 / clock.elapsed().as_nanos() as f64
        })
        .fold(0.0, crate::linalg::larger);
    eprintln!("STEPCOST-RATE {ghz:.4} cycles a ns");
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
    let list = std::env::var("STEPCOST").unwrap_or_else(|_| SCENES.to_string());
    for entry in list.split(',') {
        let (name, spec, tau) = match entry.split_once('=') {
            Some((name, rest)) => {
                let (spec, tau) = rest.split_once('@').expect("name=model@tau");
                (name.to_string(), spec.to_string(), tau.parse::<f64>().unwrap())
            }
            None => {
                let found = SCENES.split(',').find(|e| e.starts_with(&format!("{entry}="))).unwrap_or_else(|| panic!("no default for {entry}"));
                let (spec, tau) = found.split_once('=').unwrap().1.split_once('@').unwrap();
                (entry.to_string(), spec.to_string(), tau.parse().unwrap())
            }
        };
        let request = request_of(&name);
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
        let domain = Everywhere;
        // [step, node, node with the photon forms, readout at the point, readout on the middle, a jump's photon's node]
        let mut sum = [0.0f64; 6];
        let (mut work, mut jumps, mut made, mut taken, mut middle_reads) = (0.0, 0usize, [0usize; 2], 0usize, 0usize);
        for _ in 0..steps {
            let before = crate::model::selected::MADE.with(|m| *m.borrow());
            let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)) else { break };
            let after = crate::model::selected::MADE.with(|m| *m.borrow());
            made = [made[0] + after[0] - before[0], made[1] + after[1] - before[1]];
            jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
            work += step.work;
            sum[0] += least(&mut || {
                std::hint::black_box(Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)).is_ok());
            });
            let midway = Midway::new(&run, &t, tau).unwrap();
            let at = midway.reached;
            sum[1] += least(&mut || {
                std::hint::black_box(run.node(at.0, at.1, false));
            });
            sum[2] += least(&mut || {
                std::hint::black_box(run.node(at.0, at.1, true));
            });
            sum[5] += least(&mut || {
                std::hint::black_box(run.photon_node(at.0, at.1));
            });
            sum[3] += least(&mut || {
                std::hint::black_box(run.readout(t.cluster, &t.point(), &t.psi));
            });
            // On the step's middle, a quarter step after it, the middle's node made before (as the step makes it).
            let generator = &midway.generator;
            let path = generator.path(at.1, [0.0; 3]);
            let mut h = crate::linalg::Mat::zeros(t.psi.len(), t.psi.len());
            generator.generator(&path, 0.0, &mut h);
            if generator.readout(&path, 0.25 * tau, &t.psi).is_some() {
                middle_reads += 1;
                sum[4] += least(&mut || {
                    std::hint::black_box(generator.readout(&path, 0.25 * tau, &t.psi));
                });
            }
            taken += 1;
            t = step.end;
        }
        let k = taken.max(1) as f64;
        eprintln!(
            "STEPCOST {name} {spec} (P{}) tau {tau}: a step {:.1} µs | a node {:.1} µs, with the photon forms {:.1} µs, a jump's photon's {:.1} µs | an output's readout at the point {:.1} µs, on the middle {} | nodes a step {:.2} plain + {:.2} with photons, jumps a step {:.3}, counted {:.3e} real MA a step over {taken} steps",
            keep.len(),
            sum[0] / k,
            sum[1] / k,
            sum[2] / k,
            sum[5] / k,
            sum[3] / k,
            if middle_reads > 0 { format!("{:.2} µs", sum[4] / middle_reads as f64) } else { "none".to_string() },
            made[0] as f64 / k,
            made[1] as f64 / k,
            jumps as f64 / k,
            work / k,
        );
    }
}

/// A jump's cost taken apart (STEP, phase 2 item "跳变那部分的拆账"): per scene of STEPCOST (as `stepcost`) the states
/// of a walk of STEPCOST_STEPS (12) steps, and on them each step STEPCOST_REPEAT (5) times as it is and as many with no
/// jump (the threshold 10⁻³⁰⁰), the same fresh scratch each call: the two phases' thread cycles, the jumps the first
/// made, and the profiling build's regions after each phase (`crate::prof::report`, cumulative: the regions' difference
/// over the jumps is a jump's part of each). Said: "JUMPCOST <scene>" with the step as it is and with no jump (µs a step),
/// the jumps a step, a jump's cost; "JUMPCOST-PROF <scene> <phase>" the report after the phase (built with `prof`).
#[test]
#[ignore]
fn jumpcost() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (steps, repeat, index) = (env("STEPCOST_STEPS", 12), env("STEPCOST_REPEAT", 5), env("STEPCOST_INDEX", 3) as u64);
    let ghz: f64 = (0..5)
        .map(|_| {
            let (c0, clock) = (thread_cycles(), Instant::now());
            let mut x = 0u64;
            while clock.elapsed().as_millis() < 50 {
                x = std::hint::black_box(x.wrapping_add(1));
            }
            (thread_cycles() - c0) as f64 / clock.elapsed().as_nanos() as f64
        })
        .fold(0.0, crate::linalg::larger);
    let list = std::env::var("STEPCOST").unwrap_or_else(|_| SCENES.to_string());
    for entry in list.split(',') {
        let (name, rest) = entry.split_once('=').expect("name=model@tau");
        let (spec, tau) = rest.split_once('@').unwrap();
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
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep, ..selection }, crate::solve::alone(chief)).unwrap();
        // The walk's states.
        let mut t = start(&run, &request, chief, index);
        let mut states = Vec::new();
        for _ in 0..steps {
            let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)) else { break };
            states.push(t.clone());
            t = step.end;
        }
        let k = states.len().max(1) as f64;
        eprintln!("JUMPCOST-PROF {name} walk:{}", crate::prof::report());
        let phase = |label: &str, threshold: Option<f64>| -> (f64, usize) {
            let (mut cycles, mut jumps) = (0u64, 0usize);
            for state in &states {
                let mut s = state.clone();
                if let Some(theta) = threshold {
                    s.threshold = theta;
                }
                for _ in 0..repeat {
                    let c0 = thread_cycles();
                    let step = Midway::new(&run, &s, tau).and_then(|m| m.attempt(&s, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere));
                    cycles += thread_cycles() - c0;
                    if let Ok(step) = step {
                        jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
                    }
                }
            }
            eprintln!("JUMPCOST-PROF {name} {label}:{}", crate::prof::report());
            (cycles as f64 / (ghz * 1e3) / (k * repeat as f64), jumps)
        };
        // STEPCOST_PHASE: one phase alone ("as-is" or "no-jump"), its process's report its own (the reports are
        // cumulative over the process, their seconds to the millisecond).
        let only = std::env::var("STEPCOST_PHASE").ok();
        let (with, jumps) = if only.as_deref() != Some("no-jump") { phase("as-is", None) } else { (f64::NAN, 0) };
        let (without, _) = if only.as_deref() != Some("as-is") { phase("no-jump", Some(1e-300)) } else { (f64::NAN, 0) };
        let per = jumps as f64 / (k * repeat as f64);
        eprintln!("JUMPCOST {name} {spec} tau {tau}: a step as it is {with:.1} µs, with no jump {without:.1} µs; {per:.3} jumps a step; a jump {:.1} µs (over {} states, {repeat} calls each)", if per > 0.0 { (with - without) / per } else { 0.0 }, states.len());
    }
}
