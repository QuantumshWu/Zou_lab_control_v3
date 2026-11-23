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
/// - the step whole, on a fresh scratch: `Step::work`;
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
/// - the step whole with no jump (its threshold 10⁻³⁰⁰): the jumps' part is the step's less it.
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
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, crate::solve::alone(chief)).unwrap();
        let ground = {
            let c = run.cluster(0);
            c.ground == c.states.len()
        };
        let blocks = run.blocks().iter().map(Vec::len).collect::<Vec<_>>();
        const PARTS: [&str; 16] = ["step", "middle", "node", "node.point", "node.values", "node.fold", "node.contract", "node.fill", "photon node", "picture", "launch", "choice", "quadrature", "forms", "scattering", "step with no jump"];
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
            parts[0] = [step.work, least(&mut || {
                std::hint::black_box(Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)).is_ok());
            })];
            // The same step with no jump (its threshold below any norm): what of the step its jumps are.
            let mut still_t = t.clone();
            still_t.threshold = 1e-300;
            let plain = Midway::new(&run, &still_t, tau).and_then(|m| m.attempt(&still_t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)).map_or(f64::NAN, |s| s.work);
            parts[15] = [plain, least(&mut || {
                std::hint::black_box(Midway::new(&run, &still_t, tau).and_then(|m| m.attempt(&still_t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)).is_ok());
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
        let label = format!("{name} {spec}(P{}) split {split} blocks {blocks:?} tau {tau}", keep.len());
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
    }
}
