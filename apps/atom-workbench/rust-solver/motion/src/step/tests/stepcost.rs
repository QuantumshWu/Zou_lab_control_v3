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
//!   With STEPCOST_PAIRED=1 (2) also the step and the node on the same run with its direct entries one by one
//!   (`Selected::unspanned`; STEPCOST_PLAIN=sidebands: with K's sideband parts in, `Selected::sidebands_in`; =whole:
//!   both, `Selected::whole`), timed in turn with the run's on the same states (each repeat the spanned one first (the
//!   one by one first), the first call of each not timed: a load that comes and goes weighs on both), "STEPCOST-PAIRED".
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
        let order = std::env::var("STEPCOST_PAIRED").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
        let paired = order > 0;
        let mut plain = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..crate::select::choose(&scene).unwrap() }, reference).unwrap();
        match std::env::var("STEPCOST_PLAIN").as_deref() {
            Ok("sidebands") => {
                plain.sidebands_in();
            }
            Ok("whole") => {
                plain.whole();
            }
            _ => plain.unspanned(),
        }
        // [step spanned, step plain, node spanned, node plain]
        let mut pair = [0.0f64; 4];
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
            if paired {
                let timed = |f: &mut dyn FnMut()| -> f64 {
                    let c0 = thread_cycles();
                    f();
                    (thread_cycles() - c0) as f64 / (ghz * 1e3)
                };
                let mut least4 = [f64::INFINITY; 4];
                let runs = if order == 2 { [(1, &plain), (0, &run)] } else { [(0, &run), (1, &plain)] };
                for &(_, r) in &runs {
                    std::hint::black_box(Midway::new(r, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)).is_ok());
                    std::hint::black_box(r.node(at.0, at.1, false));
                }
                for _ in 0..repeat {
                    for &(i, r) in &runs {
                        least4[i] = least4[i].min(timed(&mut || {
                            std::hint::black_box(Midway::new(r, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)).is_ok());
                        }));
                        least4[2 + i] = least4[2 + i].min(timed(&mut || {
                            std::hint::black_box(r.node(at.0, at.1, false));
                        }));
                    }
                }
                for i in 0..4 {
                    pair[i] += least4[i];
                }
            }
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
        if paired {
            eprintln!("STEPCOST-PAIRED {name} {spec} tau {tau}: a step spanned {:.1} µs, one by one {:.1} µs ({:.3}); a node spanned {:.1} µs, one by one {:.1} µs ({:.3}); over {taken} steps", pair[0] / k, pair[1] / k, pair[0] / pair[1], pair[2] / k, pair[3] / k, pair[2] / pair[3]);
        }
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

/// STEP3 item 1 ("计数重写", BRIEF_STEP3): a step's counted work against its thread time, part by part. Per scene of
/// COUNT (comma separated, `name=model@tau`; the model "ground", "P" the three-point check's choice, or "+" and explicit
/// excited states' labels joined by "+"; the request as `stepcost`'s, or COUNT_REQ_<name> a native request's file), a
/// walk of STEPCOST_STEPS (12) steps from trajectory STEPCOST_INDEX (3) of `r1prop::start`, with its own jumps; the
/// profiling build's ATOM_PROF_SPLIT splits the run's blocks (`Selected::new`). Per state of the walk each part the least
/// of STEPCOST_REPEAT (5) calls in the thread's own cycles, after one call, with the count it adds as the step counts it
/// (`Price`):
/// - the step whole, on the scratch its last step left, as a trajectory takes it (`step::priced`'s second attempt):
///   `Step::work`;
/// - the middle with its nodes (`Midway::new`, its nodes made where the picture first asks its generator): its making
///   (`Middle::made`), its time a fresh middle's and its picture's less the picture's on the nodes made;
/// - a node at the step's middle by its parts (`Selected::node_parts`: the point, the values, the fold, the contraction,
///   the filling), and a jump's photon node (`Selected::photon_node`): `Node::work`;
/// - the picture (`pictured`: the generator at the nodes, the tones, the decay laid, the preparation): the propagation's
///   (`count::propagation` with the picture's own count) and the decay laid (`count::hazard` of no trace);
/// - the launch, the cut's choice, the quadrature's nodes into the sums: what each adds to the picture's and the groups'
///   counts, the quadrature's densities (`count::gathered`) with it;
/// - the forms of the sums (their part of `count::impulse` beside the densities) and the scattered light at the middle
///   (`Middle::scattering`'s count).
/// - the step whole with no jump (its threshold 10⁻³⁰⁰): the jumps' part is the step's less it;
/// - the scattered light again, each call after a step with no jump untimed: what the caches leave of it in a step.
/// The rest is the step's less these (its jumps -- photon nodes, emission, landing, roots --, the survival's traces, the
/// end's forms). Said per part, "COUNT-PART <scene> <part>": the count and the time a step, and their ratio (real
/// multiply-adds a ns); "COUNT-STEP" the step's.
#[test]
#[ignore]
fn count_parts() {
    use super::super::{count, pictured, predicted, toned, Along, Sums, SUMS, TONE};
    use crate::decide::COMPLEX;
    use crate::linalg::C64;
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
    eprintln!("COUNT-RATE {ghz:.4} cycles a ns");
    let mut least = |f: &mut dyn FnMut()| -> f64 {
        f();
        (0..repeat)
            .map(|_| {
                let c0 = thread_cycles();
                f();
                (thread_cycles() - c0) as f64 / (ghz * 1e3)
            })
            .fold(f64::INFINITY, crate::linalg::smaller)
    };
    let split = std::env::var("ATOM_PROF_SPLIT").unwrap_or_else(|_| "none".to_string());
    let list = std::env::var("COUNT").expect("COUNT: name=model@tau, comma separated");
    for entry in list.split(',') {
        let (name, rest) = entry.split_once('=').expect("name=model@tau");
        let (spec, tau) = rest.split_once('@').expect("name=model@tau");
        let tau: f64 = tau.parse().unwrap();
        let request = match std::env::var(format!("COUNT_REQ_{name}")) {
            Ok(file) => {
                let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap();
                line["options"]["device"] = serde_json::json!("cpu");
                Request::from_native(&line).unwrap()
            }
            Err(_) => request_of(name),
        };
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let keep: Vec<usize> = match spec {
            "P" => selection.keep.clone(),
            "ground" => (0..GROUND).collect(),
            labels => {
                let mut keep: Vec<usize> = (0..GROUND).collect();
                keep.extend(labels.strip_prefix('+').expect("ground, P or +labels").split('+').map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
                keep.sort_unstable();
                keep
            }
        };
        let chief = crate::solve::chief(&request);
        // COUNT_TURN: the run made as the product makes it, its parts turned where its judgment says (the tests' runs
        // are made unturned).
        // COUNT_TURN=force: every pair of parts turned whatever its W (`FORCE_TURN`).
        let turning = std::env::var("COUNT_TURN").is_ok();
        let forced = std::env::var("COUNT_TURN").is_ok_and(|v| v == "force");
        crate::model::selected::TURNING.with(|t| t.set(turning));
        crate::model::selected::FORCE_TURN.with(|t| t.set(forced));
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, crate::solve::alone(chief)).unwrap();
        crate::model::selected::TURNING.with(|t| t.set(false));
        crate::model::selected::FORCE_TURN.with(|t| t.set(false));
        let ground = {
            let c = run.cluster(0);
            c.ground == c.states.len()
        };
        let blocks = run.blocks().iter().map(Vec::len).collect::<Vec<_>>();
        const PARTS: [&str; 17] = ["step", "middle", "node", "node.point", "node.values", "node.fold", "node.contract", "node.fill", "photon node", "picture", "launch", "choice", "quadrature", "forms", "scattering", "step with no jump", "scattering after a step"];
        // Per part [count, time] summed over the walk's states.
        let mut sum = [[0.0f64; 2]; PARTS.len()];
        let mut t = start(&run, &request, chief, index);
        let (mut taken, mut made, mut jumps, mut points) = (0usize, 0usize, 0usize, 0usize);
        for _ in 0..steps {
            let before = crate::model::selected::MADE.with(|m| m.borrow()[0]);
            let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)) else { break };
            made += crate::model::selected::MADE.with(|m| m.borrow()[0]) - before;
            jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
            let mut parts = [[0.0f64; 2]; PARTS.len()];
            // The step as a trajectory takes it: on the scratch its last step left (`step::priced`'s second attempt).
            let mut warm = Scratch::default();
            let _ = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut warm, &Everywhere));
            let counted = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut warm, &Everywhere)).map_or(f64::NAN, |s| s.work);
            parts[0] = [counted, least(&mut || {
                std::hint::black_box(Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut warm, &Everywhere)).is_ok());
            })];
            // The same step with no jump (its threshold below any norm): what of the step its jumps are.
            let mut still_t = t.clone();
            still_t.threshold = 1e-300;
            let plain = Midway::new(&run, &still_t, tau).and_then(|m| m.attempt(&still_t, &Brownian::STILL, &[], &mut warm, &Everywhere)).map_or(f64::NAN, |s| s.work);
            parts[15] = [plain, least(&mut || {
                std::hint::black_box(Midway::new(&run, &still_t, tau).and_then(|m| m.attempt(&still_t, &Brownian::STILL, &[], &mut warm, &Everywhere)).is_ok());
            })];
            let midway = Midway::new(&run, &t, tau).unwrap();
            // The middle with the nodes it makes (where the picture first asks its generator): its making's count, the
            // time of a fresh middle and its picture less the picture's on the nodes made (below).
            let made_middle = |s: &mut Scratch| -> f64 {
                let fresh = Midway::new(&run, &t, tau).unwrap();
                let kappa_m = predicted(&t.forms, fresh.reached.1, tau, t.time + 0.5 * tau);
                let along = Along::new(&fresh.generator, kappa_m, kappa_m.map(|k| fresh.hbar_over_mass * k));
                let x = along.turned(&t.psi, -0.5 * tau, true)[..t.psi.len()].to_vec();
                let mut st = along.middle.still();
                let _ = pictured(&along, s, tau, Some(&x), ground, &mut st);
                fresh.generator.made()
            };
            let mut s1 = Scratch::default();
            parts[1] = [made_middle(&mut s1), least(&mut || {
                std::hint::black_box(made_middle(&mut s1));
            })];
            let at = midway.reached;
            parts[2] = [run.node(at.0, at.1, false).work, least(&mut || {
                std::hint::black_box(run.node(at.0, at.1, false));
            })];
            for (k, (count, time)) in run.node_parts(at.0, at.1, &mut least).into_iter().enumerate() {
                parts[3 + k] = [count, time];
            }
            parts[8] = [run.photon_node(at.0, at.1).work, least(&mut || {
                std::hint::black_box(run.photon_node(at.0, at.1));
            })];
            // The step's first stretch's parts on its own picture (as `t2step_parts`).
            let (n, h) = (t.psi.len(), 0.5 * tau);
            let kappa_m = predicted(&t.forms, midway.reached.1, tau, t.time + h);
            let along = Along::new(&midway.generator, kappa_m, kappa_m.map(|k| midway.hbar_over_mass * k));
            let x = along.turned(&t.psi, -h, true)[..n].to_vec();
            let mut s = Scratch::default();
            let made_still = along.middle.still();
            let mut still = made_still;
            if pictured(&along, &mut s, tau, Some(&x), ground, &mut still).is_err() {
                t = step.end;
                continue;
            }
            let tones = s.picture.beats.len();
            let costs = along.middle.work();
            let toned_ = toned(along.middle, &s.picture, n, still, ground);
            let picture_count = count::propagation(costs, still || ground, s.picture.work, toned_[0]) + count::hazard(n, 0, still, tones);
            parts[9] = [picture_count, least(&mut || {
                let mut st = made_still;
                std::hint::black_box(pictured(&along, &mut s, tau, Some(&x), ground, &mut st).is_ok());
            })];
            parts[1][1] -= parts[9][1];
            let mut st = made_still;
            let _ = pictured(&along, &mut s, tau, Some(&x), ground, &mut st);
            let slots = Sums { still, h, tones: &s.picture.beats }.count();
            let mut sums = vec![0.0; slots * 2 * n * super::super::row(n)];
            let w0 = s.picture.work;
            s.picture.launch(-h, &x);
            let launched = s.picture.work - w0;
            parts[10] = [COMPLEX * launched, least(&mut || s.picture.launch(-h, &x))];
            let nodes = {
                let Scratch { picture, groups, .. } = &mut s;
                groups.work = 0.0;
                let w0 = picture.work;
                groups.choose(picture, -h, h, (still, tones), &mut |_: &mut crate::spectral::Picture| false);
                let chosen = COMPLEX * (groups.work + picture.work - w0);
                parts[11] = [chosen, least(&mut || {
                    std::hint::black_box(groups.choose(picture, -h, h, (still, tones), &mut |_: &mut crate::spectral::Picture| false));
                })];
                let grouped = groups.groups > 1;
                groups.work = 0.0;
                let w0 = picture.work;
                let nodes = super::t2step::quadrature(picture, groups, still, h, -h, h, grouped, &mut sums);
                let summed = COMPLEX * (groups.work + picture.work - w0) + count::gathered(n, nodes, still, tones);
                parts[12] = [summed, least(&mut || {
                    super::t2step::quadrature(picture, groups, still, h, -h, h, grouped, &mut sums);
                })];
                nodes
            };
            let size = n * n;
            let mut wholes = vec![C64::ZERO; slots * size];
            for j in 0..slots {
                super::super::whole(&sums, j, n, &mut wholes[j * size..(j + 1) * size]);
            }
            let forms_count = COMPLEX * (if still { costs[1] } else { (2 * n * n) as f64 + 3.0 * costs[1] } + toned_[1]);
            let picture = &s.picture;
            parts[13] = [forms_count, least(&mut || {
                if still {
                    std::hint::black_box(along.middle.force(&along.path, 0.0, &wholes[..size]));
                } else {
                    std::hint::black_box(along.middle.nodes(&along.path, h, [&wholes[..size], &wholes[size..2 * size], &wholes[2 * size..3 * size]], None, None));
                    for (m, tone) in picture.beats.iter().enumerate() {
                        let at: Vec<(f64, &[C64], &[C64])> = (0..3)
                            .map(|j| {
                                let slot = (SUMS + TONE * m + 2 * j) * size;
                                ([-h, 0.0, h][j], &wholes[slot..slot + size], &wholes[slot + size..slot + 2 * size])
                            })
                            .collect();
                        std::hint::black_box(along.middle.tone_force(&along.path, tone.candidate, &at));
                    }
                }
            })];
            parts[14] = [along.middle.scattering(&along.path, &x).1, least(&mut || {
                std::hint::black_box(along.middle.scattering(&along.path, &x));
            })];
            // The same scattered light as the step reads it, after the rest of a step (another middle's step untimed,
            // between): what the caches hold of it then.
            parts[16] = [parts[14][0], {
                let mut fastest = f64::INFINITY;
                for _ in 0..repeat {
                    std::hint::black_box(Midway::new(&run, &still_t, tau).and_then(|m| m.attempt(&still_t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)).is_ok());
                    let c0 = thread_cycles();
                    std::hint::black_box(along.middle.scattering(&along.path, &x));
                    fastest = crate::linalg::smaller(fastest, (thread_cycles() - c0) as f64 / (ghz * 1e3));
                }
                fastest
            }];
            for (s, p) in sum.iter_mut().zip(&parts) {
                s[0] += p[0];
                s[1] += p[1];
            }
            points += nodes;
            taken += 1;
            t = step.end;
        }
        let k = taken.max(1) as f64;
        let m = made as f64 / k;
        let label = format!("{name} {spec}(P{}) split {split}{} blocks {blocks:?} tau {tau}", keep.len(), if turning { " turning" } else { "" });
        // The step's parts as its run takes them: the middle with its nodes, the picture, the first stretch's launch,
        // choice, quadrature, forms, the middle's scattered light; the rest the step's less these.
        let within = [1usize, 9, 10, 11, 12, 13, 14];
        let rest: [f64; 2] = std::array::from_fn(|i| sum[0][i] - within.iter().map(|&p| sum[p][i]).sum::<f64>());
        let say = |part: &str, c: f64, t: f64| eprintln!("COUNT-PART {label} {part}: {:.4e} real MA, {:.2} µs, {:.3} a ns", c / k, t / k, c / (t * 1e3));
        for (p, name) in PARTS.iter().enumerate() {
            say(name, sum[p][0], sum[p][1]);
        }
        say("rest", rest[0], rest[1]);
        let j = jumps as f64 / k;
        say("rest less the jumps' photon nodes", rest[0] - j * sum[8][0], rest[1] - j * sum[8][1]);
        // The jumps: the step's less the step with no jump; the rest of a step with no jump its less the parts.
        say("jumps", sum[0][0] - sum[15][0], sum[0][1] - sum[15][1]);
        let plain_rest: [f64; 2] = std::array::from_fn(|i| sum[15][i] - within.iter().map(|&p| sum[p][i]).sum::<f64>());
        say("rest of a step with no jump", plain_rest[0], plain_rest[1]);
        eprintln!(
            "COUNT-STEP {label}: {:.4e} real MA {:.1} µs {:.3} a ns | nodes a step {m:.2} (node {:.3} a ns) | jumps a step {:.3} | quadrature nodes {:.1} | over {taken} states",
            sum[0][0] / k,
            sum[0][1] / k,
            sum[0][0] / (sum[0][1] * 1e3),
            sum[2][0] / (sum[2][1] * 1e3),
            jumps as f64 / k,
            points as f64 / k
        );
        // COUNT_PROF: the profiling build's regions over the scene's calls (cumulative over the process).
        if std::env::var("COUNT_PROF").is_ok() {
            eprintln!("COUNT-PROF {label}:{}", crate::prof::report());
        }
    }
}

/// STEP4 item 2 ("(b) 接入节点"): a step's parts on the run's tables against the fold's own, paired. Per scene of
/// STEPCOST (names of `SCENES`, their ground states; default PGC3D, TET, tilt0, tilt2) a walk of STEPCOST_STEPS (12)
/// steps from trajectory STEPCOST_INDEX (3) of `r1prop::start` on the fold's own nodes; the run's tables bought on [the
/// walk's least s, 1] (`Selected::buy_tables_now`: their time, memory, pieces and counted work); then per state of the
/// walk, version 0 (the fold's own) and version 1 (the tables) in turn -- each repeat the other first --, each the least
/// of STEPCOST_REPEAT (5) in the thread's own cycles after one call untimed: the step on a fresh scratch with no read and
/// with one read at its middle (the lift made where a step reads, `step::Middle::reading`), a node at the step's middle
/// and a jump's photon node there (each on its version, `step::on_version`), and the counted work of the step and of the
/// node. Said: "TABLED-STEP <scene>", the means over the walk of both versions and their ratios, and how many of the
/// steps' points the tables read. Ignored; run by name, pinned to one performance core by the caller.
#[test]
#[ignore]
fn tabled_step_parts() {
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
    let timed = |f: &mut dyn FnMut()| -> f64 {
        let c0 = thread_cycles();
        f();
        (thread_cycles() - c0) as f64 / (ghz * 1e3)
    };
    let list = std::env::var("STEPCOST").unwrap_or_else(|_| "PGC3D,TET,tilt0,tilt2".to_string());
    for entry in list.split(',') {
        let found = SCENES.split(',').find(|e| e.starts_with(&format!("{entry}="))).unwrap_or_else(|| panic!("no default for {entry}"));
        let (spec, tau) = found.split_once('=').unwrap().1.split_once('@').unwrap();
        let (name, tau): (String, f64) = (entry.to_string(), tau.parse().unwrap());
        assert_eq!(spec, "ground", "the tables' scenes are their ground states");
        let request = request_of(&name);
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let chief = crate::solve::chief(&request);
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: (0..GROUND).collect(), ..selection }, crate::solve::alone(chief)).unwrap();
        let domain = Everywhere;
        // The walk on the fold's own nodes, and its least s.
        let mut t = start(&run, &request, chief, index);
        let mut walk = Vec::new();
        for _ in 0..steps {
            let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)) else { break };
            walk.push(t.clone());
            t = step.end;
        }
        let values = |r: [f64; 3], kappa: [f64; 3]| -> crate::model::rs::fold::Values {
            let mut point = crate::scene::Point::default();
            run.scene.point_into(r, kappa, &mut point);
            run.rs.tweezer().values(&run.scene, &point)
        };
        let low = walk.iter().map(|t| crate::model::rs::fold::tabled::Tables::intensity(run.rs.plan(), &values(t.r, t.kappa))).fold(1.0, crate::linalg::smaller) * 0.999;
        let clock = Instant::now();
        let any = run.buy_tables_now(low);
        let bought = clock.elapsed().as_secs_f64();
        let Some(tables) = run.bought() else {
            eprintln!("TABLED-BUY {name}: none kept ({any}): no kind of node costs less off them");
            continue;
        };
        eprintln!(
            "TABLED-BUY {name}: for the kinds [node, with the lift, photon, with the photon forms] {:?}; {} pieces on [{low:.4}, 1], {:.2} MB, {:.3} s (its own clock {:.3} s), counted {:.3e} real MA",
            run.bought_for().unwrap(),
            tables.pieces.len(),
            tables.bytes as f64 / 1e6,
            bought,
            tables.build_seconds,
            crate::decide::COMPLEX * tables.work
        );
        // [version][step, step with a read, node, photon node, node warm]; counted [version][step, node].
        let (mut sum, mut counted) = ([[0.0f64; 5]; 2], [[0.0f64; 3]; 2]);
        let (mut taken, mut read_off, mut points) = (0usize, 0usize, 0usize);
        let mut deviation = [0.0f64; 2];
        for from in &walk {
            let midway = Midway::new(&run, from, tau).unwrap();
            let at = midway.reached;
            let read = [from.time + 0.5 * tau];
            let mut least = [[f64::INFINITY; 5]; 2];
            let versions = [crate::step::Trajectory { version: 0, ..from.clone() }, crate::step::Trajectory { version: 1, ..from.clone() }];
            for (v, t) in versions.iter().enumerate() {
                let step = Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)).unwrap();
                counted[v][0] += step.work;
                let node = crate::step::on_version(v as u32, || run.node(at.0, at.1, false));
                counted[v][1] += node.work;
                deviation = [deviation[0].max(node.deviation[0]), deviation[1].max(node.deviation[1])];
                counted[v][2] += Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &read, &mut Scratch::default(), &domain)).unwrap().work;
                std::hint::black_box(crate::step::on_version(v as u32, || run.photon_node(at.0, at.1)));
            }
            for k in 0..repeat {
                let order: [usize; 2] = if k % 2 == 0 { [0, 1] } else { [1, 0] };
                for &v in &order {
                    let t = &versions[v];
                    least[v][0] = least[v][0].min(timed(&mut || {
                        std::hint::black_box(Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)).is_ok());
                    }));
                    least[v][1] = least[v][1].min(timed(&mut || {
                        std::hint::black_box(Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &read, &mut Scratch::default(), &domain)).is_ok());
                    }));
                    least[v][2] = least[v][2].min(timed(&mut || {
                        std::hint::black_box(crate::step::on_version(v as u32, || run.node(at.0, at.1, false)));
                    }));
                    least[v][3] = least[v][3].min(timed(&mut || {
                        std::hint::black_box(crate::step::on_version(v as u32, || run.photon_node(at.0, at.1)));
                    }));
                }
            }
            // The node warm: its version's calls one after another, nothing else between them.
            for v in 0..2 {
                for _ in 0..repeat {
                    least[v][4] = least[v][4].min(timed(&mut || {
                        std::hint::black_box(crate::step::on_version(v as u32, || run.node(at.0, at.1, false)));
                    }));
                }
            }
            for v in 0..2 {
                for p in 0..5 {
                    sum[v][p] += least[v][p];
                }
            }
            // Whether the tables read the step's three points.
            for s in [-0.5 * tau, 0.0, 0.5 * tau] {
                let r: [f64; 3] = std::array::from_fn(|a| at.0[a] + run.hbar_over_mass() * at.1[a] * s);
                points += 1;
                read_off += usize::from(run.tables_read(r, at.1));
            }
            taken += 1;
        }
        let k = taken.max(1) as f64;
        let m = |v: usize, p: usize| sum[v][p] / k;
        eprintln!(
            "TABLED-STEP {name} tau {tau}: the fold's / the tables' (ratio) -- a step {:.1} / {:.1} µs ({:.3}), with a read {:.1} / {:.1} ({:.3}), a node {:.1} / {:.1} ({:.3}), a jump's photon node {:.1} / {:.1} ({:.3}), a node warm (its calls one after another) {:.1} / {:.1} ({:.3}); counted a step {:.3e} / {:.3e} ({:.3}), with a read {:.3e} / {:.3e} ({:.3}), a node {:.3e} / {:.3e} ({:.3}); the tables read {read_off} of {points} points; the nodes' deviation at the most {deviation:?}; over {taken} states, each the least of {repeat}",
            m(0, 0),
            m(1, 0),
            m(1, 0) / m(0, 0),
            m(0, 1),
            m(1, 1),
            m(1, 1) / m(0, 1),
            m(0, 2),
            m(1, 2),
            m(1, 2) / m(0, 2),
            m(0, 3),
            m(1, 3),
            m(1, 3) / m(0, 3),
            m(0, 4),
            m(1, 4),
            m(1, 4) / m(0, 4),
            counted[0][0] / k,
            counted[1][0] / k,
            counted[1][0] / counted[0][0],
            counted[0][2] / k,
            counted[1][2] / k,
            counted[1][2] / counted[0][2],
            counted[0][1] / k,
            counted[1][1] / k,
            counted[1][1] / counted[0][1],
        );
    }
}

/// STEP4, the coordinator's question: a step's thread time on the fold's version and on the tables' at one thread and at
/// the machine's concurrency, paired in one process. Per scene of STEPCOST (names of `SCENES`, their ground states;
/// default PGC3D, TET) each of STEPCOST_THREADS threads (every core) its own walk of STEPCOST_STEPS (24) steps
/// (trajectory i of `r1prop::start`) on the fold's nodes; the tables bought on [the walks' least s, 1]; then
/// STEPCOST_ROUNDS (3) rounds, each version in turn (the other first every other round), at one thread (the first
/// walk's) and at all of them at once: each thread its walk's states once untimed and once timed in its own cycles, on
/// one scratch as a trajectory carries it. Said: "TABLED-THREADS <scene>", per version and concurrency the mean thread
/// time a step (µs) at its least over the rounds, and the round's wall time. Ignored; run by name, not pinned.
#[test]
#[ignore]
fn tabled_step_threads() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let cores = std::thread::available_parallelism().map_or(1, |n| n.get());
    let (steps, rounds, threads) = (env("STEPCOST_STEPS", 24), env("STEPCOST_ROUNDS", 3), env("STEPCOST_THREADS", cores));
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
    let list = std::env::var("STEPCOST").unwrap_or_else(|_| "PGC3D,TET".to_string());
    for entry in list.split(',') {
        let found = SCENES.split(',').find(|e| e.starts_with(&format!("{entry}="))).unwrap_or_else(|| panic!("no default for {entry}"));
        let tau: f64 = found.split_once('@').unwrap().1.parse().unwrap();
        let request = request_of(entry);
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let chief = crate::solve::chief(&request);
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: (0..GROUND).collect(), ..selection }, crate::solve::alone(chief)).unwrap();
        let domain = Everywhere;
        let walks: Vec<Vec<crate::step::Trajectory>> = (0..threads)
            .map(|i| {
                let mut t = start(&run, &request, chief, i as u64);
                let mut walk = Vec::new();
                for _ in 0..steps {
                    let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &domain)) else { break };
                    walk.push(t.clone());
                    t = step.end;
                }
                walk
            })
            .collect();
        let low = walks
            .iter()
            .flatten()
            .map(|t| {
                let mut point = crate::scene::Point::default();
                run.scene.point_into(t.r, t.kappa, &mut point);
                crate::model::rs::fold::tabled::Tables::intensity(run.rs.plan(), &run.rs.tweezer().values(&run.scene, &point))
            })
            .fold(1.0, crate::linalg::smaller)
            * 0.999;
        assert!(run.buy_tables_now(low), "{entry}: no tables");
        // One thread's walk on the version v: once untimed, once timed; its thread time a step (µs).
        let walked = |walk: &[crate::step::Trajectory], v: u32| -> f64 {
            let mut scratch = Scratch::default();
            let mut go = || {
                for from in walk {
                    let t = crate::step::Trajectory { version: v, ..from.clone() };
                    std::hint::black_box(Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &domain)).is_ok());
                }
            };
            go();
            let c0 = thread_cycles();
            go();
            (thread_cycles() - c0) as f64 / (ghz * 1e3) / walk.len().max(1) as f64
        };
        // [version][one thread, all threads]: the least over the rounds of the threads' mean; and the walls.
        let (mut least, mut wall) = ([[f64::INFINITY; 2]; 2], [[f64::INFINITY; 2]; 2]);
        for round in 0..rounds {
            let order: [u32; 2] = if round % 2 == 0 { [0, 1] } else { [1, 0] };
            for &v in &order {
                for (c, n) in [1, threads].into_iter().enumerate() {
                    let clock = Instant::now();
                    let per: Vec<f64> = std::thread::scope(|s| {
                        let handles: Vec<_> = walks[..n].iter().map(|walk| s.spawn(move || walked(walk, v))).collect();
                        handles.into_iter().map(|h| h.join().unwrap()).collect()
                    });
                    let mean = per.iter().sum::<f64>() / per.len() as f64;
                    least[v as usize][c] = least[v as usize][c].min(mean);
                    wall[v as usize][c] = wall[v as usize][c].min(clock.elapsed().as_secs_f64());
                }
            }
        }
        eprintln!(
            "TABLED-THREADS {entry} tau {tau}: a step's thread time, the fold's / the tables' (ratio) -- one thread {:.1} / {:.1} µs ({:.3}); {threads} threads at once {:.1} / {:.1} µs ({:.3}); the walls of {threads} threads {:.2} / {:.2} s; {steps} steps a thread, the least of {rounds} rounds",
            least[0][0],
            least[1][0],
            least[1][0] / least[0][0],
            least[0][1],
            least[1][1],
            least[1][1] / least[0][1],
            wall[0][1],
            wall[1][1]
        );
    }
}
