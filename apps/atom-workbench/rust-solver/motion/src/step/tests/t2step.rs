//! T2STEP's measurements (BRIEF_T2STEP): a step of the cluster of ground states alone (G) taken apart, tilt2's against
//! the other scenes'. The scenes T2S_SCENE (comma separated; default tilt2, PGC3D, tilt0, TET, GMD1C, R4, R5_PUMP): the
//! six by their requests (R1PROP_REQ, as `r1prop`), the reference scenes by name (refs/scenes); each scene's models
//! T2S_<SCENE> (comma separated: "ground" its ground states alone, the default; "P" the three-point check's choice); the
//! step T2S_TAU_<SCENE>, else the pilot's of the new selection (p3select.status.md: tilt2 and tilt0 0.326, PGC3D and
//! TET 0.474, GMD1C 0.948, R4 0.1185, R5_PUMP 2.7826 µs). Ignored; run by name, one at a time, the caller pinning the
//! thread to one performance core for the times:
//! `<test exe> step::tests::t2step::<name> --exact --include-ignored --nocapture --test-threads 1`.
//! - `t2step_census`: per model its states, exact blocks, the fold's blocks, the candidate tones with their keys, a
//!   node's beats of the generator and of the gradient, its photon classes and their elements; per step of a walk
//!   (T2S_STEPS, 12) its jumps, whether it took one node, the picture's tones and classes (followed, hopped), the
//!   quadrature's nodes, its groups, the quotient's bandwidth, the sums' count and the counted work by part.
//! - `t2step_cost`: per model a step's time on the states of a walk (T2S_STEPS, 12), each the least of T2S_REPEAT (5)
//!   on a fresh scratch, its first call not timed; a node's time (`Selected::node`, the least of 30) at the walk's
//!   middles and the nodes a step makes; built with the feature `prof`, the step's regions over every timed step
//!   (`crate::prof::report`, inclusive; the slots 84 to 91 are this measurement's: spectral.rs `prepare`'s classes, its
//!   decomposition, the changes to its eigenbasis, `follow`, the hops' radius with the spans; step.rs `Groups::choose`,
//!   the tones' forms, the stretch's forms).

use crate::model::selected::Selected;
use crate::request::{Request, GROUND};
use crate::step::{Brownian, Dynamics, Middle};

use super::super::{Midway, Scratch, Sums};
use super::r1prop::{env, start};
use super::Everywhere;

/// The scenes (module documentation).
fn scenes() -> Vec<(String, Request)> {
    let names = std::env::var("T2S_SCENE").unwrap_or_else(|_| "tilt2,PGC3D,tilt0,TET,GMD1C,R4,R5_PUMP".to_string());
    let files = [("PGC3D", "PGC3D_1ms.json"), ("TET", "TET_1ms.json"), ("R4", "R4_LIN_4.json"), ("GMD1C", "GMD1C_1ms.json"), ("tilt0", "tilt0.json"), ("tilt2", "tilt2.json")];
    names
        .split(',')
        .map(|name| {
            let request = match files.iter().find(|(n, _)| *n == name) {
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
            };
            (name.to_string(), request)
        })
        .collect()
}

/// A scene's models (module documentation): each its label and its run (`Selected::new` on the three-point check's
/// selection with P replaced, the reference state the chief alone, as the solve makes it).
fn models(name: &str, request: &Request) -> Vec<(String, Selected)> {
    let specs = std::env::var(format!("T2S_{name}")).unwrap_or_else(|_| "ground".to_string());
    let scene = crate::scene::Scene::new(request).unwrap();
    let selection = crate::select::choose(&scene).unwrap();
    let reference = crate::solve::alone(crate::solve::chief(request));
    specs
        .split(',')
        .map(|spec| {
            let keep: Vec<usize> = match spec {
                "P" => selection.keep.clone(),
                "ground" => (0..GROUND).collect(),
                // "+a+b": the ground states and the excited states a, b (labels).
                other if other.starts_with('+') => {
                    let mut keep: Vec<usize> = (0..GROUND).collect();
                    keep.extend(other.split('+').filter(|l| !l.is_empty()).map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
                    keep.sort_unstable();
                    keep
                }
                other => panic!("unknown model {other}"),
            };
            let run = Selected::new(crate::scene::Scene::new(request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection.clone() }, reference).unwrap();
            (format!("{spec}(P{})", keep.len()), run)
        })
        .collect()
}

/// The step of a scene's walk (module documentation).
fn tau_of(name: &str) -> f64 {
    let pilot = match name {
        "tilt2" | "tilt0" => 0.326,
        "GMD1C" => 0.9485,
        "R4" => 0.1185,
        "R5_PUMP" => 2.7826,
        _ => 0.4741,
    };
    std::env::var(format!("T2S_TAU_{name}")).ok().map_or(pilot, |x| x.parse().unwrap())
}

#[test]
#[ignore]
fn t2step_census() {
    let steps = env("T2S_STEPS", 12.0) as usize;
    for (name, request) in scenes() {
        let tau = tau_of(&name);
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let cluster = run.cluster(0);
            let mut t = start(&run, &request, chief, 3);
            let middle = run.middle(0, t.r, t.time);
            let path = middle.path(t.kappa, [0.0; 3]);
            let mut h = crate::linalg::Mat::zeros(middle.states(), middle.states());
            middle.generator(&path, 0.0, &mut h);
            let (mut beats, mut keys) = (vec![], vec![]);
            middle.beats(&path, &mut beats, &mut keys);
            let node = run.node(t.r, t.kappa, false);
            let elements: Vec<usize> = node.classes.iter().map(|c| c.axes.iter().flatten().map(|j| j.elements.len()).sum()).collect();
            let components: usize = node.classes.iter().map(|c| c.axes.iter().map(Vec::len).sum::<usize>()).sum();
            println!(
                "T2S-MODEL {name} {label}: {} states, {} ground, exact blocks {:?}; the fold's blocks (size, Q's) {:?}; candidate tones {} {:?} keys {:?}; a node's generator beats {}, gradient beats {}, photon classes {} of {components} components, elements {elements:?}; counted {:.3e} real MA",
                cluster.states.len(),
                cluster.ground,
                middle.blocks().iter().map(Vec::len).collect::<Vec<_>>(),
                run.rs.plan().blocks(),
                beats.len(),
                beats,
                keys,
                node.generator.len(),
                node.gradient.len(),
                node.classes.len(),
                node.work,
            );
            let mut scratch = Scratch::default();
            for k in 0..steps {
                let step = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere));
                match step {
                    Ok(step) => {
                        let picture = &scratch.picture;
                        let slots = Sums { still: step.prices.last().is_some_and(|p| p.1.still), h: 0.5 * tau, tones: &picture.beats }.count();
                        let parts: Vec<String> = step
                            .prices
                            .iter()
                            .map(|(_, p)| {
                                let w = p.parts(p.costs);
                                format!("still {} tones {} nodes {} traces {} grouped {:.0} nu_Q {:.1} | counted [propagation/kick/hazard/optical/end/model] {:.0}/{:.0}/{:.0}/{:.0}/{:.0}/{:.0}", p.still, p.tones, p.nodes, p.traces, p.grouped, p.bandwidth, w[0], w[1], w[2], w[3], w[4], w[5])
                            })
                            .collect();
                        let groups = &scratch.groups;
                        println!(
                            "T2S-STEP {name} {label} step {k}: jumps {} | sums {slots} | last piece's groups {} order {} labels {} | picture {} | {}",
                            step.made.iter().filter(|m| m.kick.is_some()).count(),
                            groups.groups,
                            groups.cut.map_or(0, |c| c.order),
                            groups.labels.len(),
                            crate::spectral::tests::census(picture),
                            parts.join(" ; ")
                        );
                        t = step.end;
                    }
                    Err(e) => {
                        println!("T2S-STEP {name} {label} step {k}: refused {e:?}");
                        break;
                    }
                }
            }
        }
    }
}

/// f's least time of `times` (µs), after one call.
fn least(f: &mut dyn FnMut(), times: usize) -> f64 {
    f();
    (0..times)
        .map(|_| {
            let clock = std::time::Instant::now();
            f();
            clock.elapsed().as_secs_f64() * 1e6
        })
        .fold(f64::INFINITY, crate::linalg::smaller)
}

#[test]
#[ignore]
fn t2step_cost() {
    let (steps, repeat) = (env("T2S_STEPS", 12.0) as usize, env("T2S_REPEAT", 5.0) as usize);
    for (name, request) in scenes() {
        let tau = tau_of(&name);
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let mut t = start(&run, &request, chief, 3);
            let (mut time, mut node, mut counted, mut made, mut taken) = (0.0, 0.0, 0.0, [0usize; 2], 0usize);
            for _ in 0..steps {
                // The step: the least of `repeat`, each on a fresh scratch (its first call not timed); the nodes it makes.
                let mut best = f64::INFINITY;
                let mut next = None;
                for r in 0..=repeat {
                    let mut s = Scratch::default();
                    let before = crate::model::selected::MADE.with(|m| *m.borrow());
                    let clock = std::time::Instant::now();
                    let step = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut s, &Everywhere));
                    let elapsed = clock.elapsed().as_secs_f64() * 1e6;
                    if std::env::var("T2S_VERBOSE").is_ok() {
                        eprintln!("T2S-EACH {name} {label} state {taken} call {r}: {elapsed:.1} µs");
                    }
                    if r > 0 {
                        best = best.min(elapsed);
                    }
                    let after = crate::model::selected::MADE.with(|m| *m.borrow());
                    if r == 0 {
                        made = [made[0] + after[0] - before[0], made[1] + after[1] - before[1]];
                    }
                    if let (Ok(step), None) = (step, &next) {
                        counted += step.work;
                        next = Some(step.end);
                    }
                }
                let Some(end) = next else { break };
                // A node at the step's middle.
                let midway = Midway::new(&run, &t, tau).unwrap();
                let at = midway.reached;
                node += least(&mut || { std::hint::black_box(run.node(at.0, at.1, false)); }, 30);
                time += best;
                taken += 1;
                t = end;
            }
            let k = taken.max(1) as f64;
            eprintln!(
                "T2S-COST {name} {label} tau {tau}: a step {:.1} µs (counted {:.4e} real MA), a node {:.1} µs; nodes made a step {:.2} plain and {:.2} with the photon forms (mean over {taken} states of a walk, each the least of {repeat})",
                time / k,
                counted / k,
                node / k,
                made[0] as f64 / k,
                made[1] as f64 / k
            );
            #[cfg(feature = "prof")]
            eprintln!("T2S-PROF {name} {label} over {} steps:{}", taken * (repeat + 1), crate::prof::report());
        }
    }
}

/// A piece's sums over [a, b] by the cut already chosen (`Groups::spanned`, where `grouped`) or by the plain rule
/// (`count`), laid as the step lays them (gstep.rs `by_rule`): the nodes taken.
#[allow(clippy::too_many_arguments)]
fn quadrature(picture: &mut crate::spectral::Picture, groups: &mut super::super::Groups, still: bool, h: f64, a: f64, b: f64, grouped: bool, sums: &mut [f64]) -> usize {
    use super::super::{count, panels, FORMS};
    let n = picture.states();
    sums.fill(0.0);
    let mut x = [crate::linalg::C64::ZERO; crate::model::STATES];
    let mut nodes = 0;
    if grouped {
        let laid = groups.spanned(b - a);
        let mut from = a;
        for (end, rule) in panels(a, b, laid) {
            groups.begin(picture.origin(), from, end, laid.1);
            for (u, w, k) in rule {
                groups.parts(picture, u);
                groups.gather(k, u, w, &Sums { still, h, tones: &picture.beats }, sums, None);
                nodes += 1;
            }
            from = end;
        }
    } else {
        for (_, rule) in panels(a, b, count(b - a, picture.quotient, FORMS, 2, || false)) {
            for (u, w, _) in rule {
                picture.at(u, &mut x[..n]);
                Sums { still, h, tones: &picture.beats }.gather(u, w, &x[..n], None, sums);
                nodes += 1;
            }
        }
    }
    nodes
}

/// A step's first stretch taken apart: per state of a walk (T2S_STEPS, 12) at the scene's τ, each piece the least of
/// T2S_REPEAT (5) after one call -- the step whole (a fresh scratch each); the middle (`Midway::new`); a node of the
/// model (`Selected::node`) times the nodes a step makes; the picture with its generator at the nodes made once
/// (`pictured` again on the same middle: the generator's assembly, the tones' separation and the preparation -- of it
/// `Picture::prepare` alone); the launch; the cut's choice (`Groups::choose`); the quadrature's nodes into the sums;
/// the forms of the sums (`Middle::nodes` or `Middle::force`, each tone's `Middle::tone_force`); the scattered light at
/// the middle. The step less these is the rest (jumps and their nodes, roots, the survival's traces).
#[test]
#[ignore]
fn t2step_parts() {
    use super::super::{pictured, predicted, Along, SUMS, TONE};
    use crate::linalg::C64;
    let (steps, repeat) = (env("T2S_STEPS", 12.0) as usize, env("T2S_REPEAT", 5.0) as usize);
    for (name, request) in scenes() {
        let tau = tau_of(&name);
        let h = 0.5 * tau;
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let mut t = start(&run, &request, chief, 3);
            // [step, middle, node, picture, preparation, launch, choice, quadrature, forms, scattering]
            let mut total = [0.0f64; 10];
            let (mut taken, mut made, mut nodes, mut jumps) = (0usize, 0usize, 0usize, 0usize);
            let ground = {
                let c = run.cluster(0);
                c.ground == c.states.len()
            };
            for _ in 0..steps {
                let before = crate::model::selected::MADE.with(|m| m.borrow()[0]);
                let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)) else { break };
                made += crate::model::selected::MADE.with(|m| m.borrow()[0]) - before;
                jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
                let whole = least(&mut || { std::hint::black_box(Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)).is_ok()); }, repeat);
                let middle = least(&mut || { std::hint::black_box(Midway::new(&run, &t, tau).is_ok()); }, repeat);
                let midway = Midway::new(&run, &t, tau).unwrap();
                let node = least(&mut || { std::hint::black_box(run.node(midway.reached.0, midway.reached.1, false)); }, repeat);
                let n = t.psi.len();
                let kappa_m = predicted(&t.forms, midway.reached.1, tau, t.time + h);
                let along = Along::new(&midway.generator, kappa_m, kappa_m.map(|k| midway.hbar_over_mass * k));
                let x = along.turned(&t.psi, -h, true)[..n].to_vec();
                let mut s = Scratch::default();
                let made_still = along.middle.still();
                let mut still = made_still;
                // The generator at the nodes made here, once: the middle keeps its nodes.
                if pictured(&along, &mut s, tau, Some(&x), ground, &mut still).is_err() {
                    t = step.end;
                    continue;
                }
                let picture_time = least(&mut || {
                    let mut st = made_still;
                    std::hint::black_box(pictured(&along, &mut s, tau, Some(&x), ground, &mut st).is_ok());
                }, repeat);
                // The preparation alone: the generator at the nodes and the tones taken out as `pictured` takes them
                // (the middle keeps its nodes), untimed, before each timed `prepare`.
                let mut prepare = f64::INFINITY;
                for _ in 0..=repeat {
                    let picture = &mut s.picture;
                    along.middle.generator(&along.path, 0.0, &mut picture.base);
                    if !still {
                        along.middle.generator(&along.path, -h, &mut picture.ends[0]);
                        along.middle.generator(&along.path, h, &mut picture.ends[1]);
                    }
                    picture.separate(
                        h,
                        still,
                        |beats, keys| {
                            if !made_still {
                                along.middle.beats(&along.path, beats, keys);
                            }
                        },
                        |m, u, plus, minus| along.middle.envelopes(&along.path, m, u, plus, minus),
                    );
                    let static_path = still && picture.beats.is_empty();
                    let clock = std::time::Instant::now();
                    std::hint::black_box(picture.prepare(h, static_path, Some(&x)).is_ok());
                    prepare = prepare.min(clock.elapsed().as_secs_f64() * 1e6);
                }
                let mut st = made_still;
                let _ = pictured(&along, &mut s, tau, Some(&x), ground, &mut st);
                let slots = Sums { still, h, tones: &s.picture.beats }.count();
                let size = slots * 2 * n * super::super::row(n);
                let mut sums = vec![0.0; size];
                let launch = least(&mut || s.picture.launch(-h, &x), repeat);
                let tones = s.picture.beats.len();
                let choice = {
                    let Scratch { picture, groups, .. } = &mut s;
                    least(&mut || { std::hint::black_box(groups.choose(picture, -h, h, (still, tones), &mut |_: &mut crate::spectral::Picture| false)); }, repeat)
                };
                let (quad, count) = {
                    let Scratch { picture, groups, .. } = &mut s;
                    let grouped = groups.groups > 1;
                    let mut count = 0;
                    let time = least(&mut || count = quadrature(picture, groups, still, h, -h, h, grouped, &mut sums), repeat);
                    (time, count)
                };
                nodes += count;
                let forms = {
                    let size = n * n;
                    let mut wholes = vec![C64::ZERO; slots * size];
                    for j in 0..slots {
                        super::super::whole(&sums, j, n, &mut wholes[j * size..(j + 1) * size]);
                    }
                    let picture = &s.picture;
                    least(&mut || {
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
                    }, repeat)
                };
                let scattering = least(&mut || { std::hint::black_box(along.middle.scattering(&along.path, &x)); }, repeat);
                for (a, b) in total.iter_mut().zip([whole, middle, node, picture_time, prepare, launch, choice, quad, forms, scattering]) {
                    *a += b;
                }
                taken += 1;
                t = step.end;
            }
            let k = taken.max(1) as f64;
            let m = made as f64 / k;
            let parts = total[1] + m * total[2] + total[3] + total[5] + total[6] + total[7] + total[8] + total[9];
            eprintln!(
                "T2S-PARTS {name} {label} tau {tau}: a step {:.1} µs | middle {:.1} | nodes {m:.2} × {:.1} = {:.1} | picture (the generator's assembly, the tones, the preparation) {:.1}, of it the preparation {:.1} | launch {:.1} | the cut's choice {:.1} | quadrature {:.1} ({:.1} nodes, {:.2} µs each) | forms {:.1} | scattering {:.1} | the rest (jumps {:.2} a step, their nodes, roots, traces) {:.1} (mean over {taken} states, each the least of {repeat})",
                total[0] / k,
                total[1] / k,
                total[2] / k,
                m * total[2] / k,
                total[3] / k,
                total[4] / k,
                total[5] / k,
                total[6] / k,
                total[7] / k,
                nodes as f64 / k,
                total[7] / nodes.max(1) as f64,
                total[8] / k,
                total[9] / k,
                jumps as f64 / k,
                (total[0] - parts) / k
            );
        }
    }
}

/// A walk's every step said to the bit (R2-NODE's `r2_walk_dump` format, read by p2gstep's `walkdiff.py`): per model a
/// walk of T2S_WALK (24) steps at the scene's τ (halved where a step is refused) from the trajectory 3 of `r1prop::start`
/// -- its internal state the T2S_STATE_<SCENE>-th basis state where one is given --, with its own jumps.
#[test]
#[ignore]
fn t2step_walk() {
    let steps = env("T2S_WALK", 24.0) as usize;
    for (name, request) in scenes() {
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let mut t = start(&run, &request, chief, 3);
            if let Some(s) = std::env::var(format!("T2S_STATE_{name}")).ok().and_then(|x| x.parse::<usize>().ok()) {
                let mut psi = crate::step::Internal::zeros(t.psi.len());
                psi[s] = crate::linalg::C64::ONE;
                t.psi = psi;
            }
            let (mut scratch, mut tau, mut taken) = (Scratch::default(), tau_of(&name), 0);
            while taken < steps && tau > 1e-3 {
                match Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere)) {
                    Ok(step) => {
                        let jumps = step.made.iter().filter(|m| m.kick.is_some()).count();
                        t = step.end;
                        taken += 1;
                        let psi: Vec<String> = t.psi.iter().map(|z| format!("{:?},{:?}", z.re, z.im)).collect();
                        eprintln!("R2WALK {name}-{label} {taken} tau {tau:?} time {:?} r {:?} kappa {:?} jumps {jumps} work {:?} psi {}", t.time, t.r, t.kappa, step.work, psi.join(";"));
                    }
                    Err(e) => {
                        eprintln!("R2WALK {name}-{label} refused at tau {tau:?}: {e:?}");
                        tau *= 0.5;
                    }
                }
            }
        }
    }
}

/// The process's CPU time so far, user and kernel (s): Windows' `GetProcessTimes`.
fn process_seconds() -> f64 {
    #[repr(C)]
    struct FileTime {
        low: u32,
        high: u32,
    }
    extern "system" {
        fn GetCurrentProcess() -> *mut std::ffi::c_void;
        fn GetProcessTimes(process: *mut std::ffi::c_void, creation: *mut FileTime, exit: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
    }
    let mut t: [FileTime; 4] = std::array::from_fn(|_| FileTime { low: 0, high: 0 });
    let [c, e, k, u] = &mut t;
    // SAFETY: the current process's pseudo handle and four FILETIMEs to write.
    let ok = unsafe { GetProcessTimes(GetCurrentProcess(), c, e, k, u) };
    assert!(ok != 0, "GetProcessTimes");
    let seconds = |f: &FileTime| ((u64::from(f.high) << 32) | u64::from(f.low)) as f64 * 1e-7;
    seconds(&t[2]) + seconds(&t[3])
}

/// The run's counts as `adapt::counted` says them: steps, refused, the mean step's time (µs).
fn counts() -> (u64, u64, f64) {
    let line = crate::adapt::counted(false);
    let number = |after: &str, upto: &str| -> f64 {
        let from = line.find(after).map(|i| i + after.len()).unwrap_or(0);
        line[from..].split(upto).next().unwrap_or("0").trim().parse().unwrap_or(f64::NAN)
    };
    (number("步 ", "（") as u64, number("拒 ", "）") as u64, number("每步 ", "µs"))
}

/// The whole solve on a model, stage by stage (the coordinator's 11-22 ask): per stage the solve says (`Message::Stage`:
/// its step and rung) and per checkpoint, the wall time since the solve's start and since the last message, the steps
/// taken and refused in it, the time in steps (adapt.rs STEPPING: each step's wall time on its thread, summed) and
/// the process's CPU time, with the thread time per step and the mean busy threads; the models' making beside. The
/// product's first batch ends it (as `gstep_ladder`).
#[test]
#[ignore]
fn t2step_run() {
    for (name, request) in scenes() {
        let clock = std::time::Instant::now();
        for (label, run) in models(&name, &request) {
            eprintln!("T2S-RUN {name} {label}: the models made by {:.1} s", clock.elapsed().as_secs_f64());
            let start = (std::time::Instant::now(), process_seconds(), counts());
            let mut last = (start.0, start.1, start.2, String::from("solve_on"));
            let pilot = crate::solve::pilot_size();
            let total = std::cell::Cell::new((0u64, 0u64));
            let mark = |what: String, last: &mut (std::time::Instant, f64, (u64, u64, f64), String)| {
                let (now, cpu, (steps, refused, mean)) = (std::time::Instant::now(), process_seconds(), counts());
                // A solve's end tallies the counts to zero (solve.rs, `adapt::counted(true)`): after it, from zero.
                let (steps0, refused0, mean0) = if steps < last.2 .0 { (0, 0, 0.0) } else { last.2 };
                let stepping = (steps as f64 * mean - steps0 as f64 * mean0) * 1e-6;
                let (ds, wall, dcpu) = (steps - steps0, (now - last.0).as_secs_f64(), cpu - last.1);
                total.set((total.get().0 + ds, total.get().1 + refused - refused0));
                eprintln!(
                    "T2S-RUN {name} {label} {:>7.1} s | {:<40} lasted {:>6.1} s: steps {:>7} refused {:>5}, in steps {:>7.1} thread-s ({:>7.1} µs a step), process CPU {:>7.1} s ({:>4.1} threads busy)",
                    (now - start.0).as_secs_f64(),
                    last.3,
                    wall,
                    ds,
                    refused - refused0,
                    stepping,
                    if ds > 0 { stepping / ds as f64 * 1e6 } else { 0.0 },
                    dcpu,
                    dcpu / wall.max(1e-9)
                );
                *last = (now, cpu, (steps, refused, mean), what);
            };
            let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::solve::solve_on(&request, &run, &mut |m| match m {
                    crate::solve::Message::Stage(s) => mark(format!("{:?} tau {:?} batch {:?}", s.step, s.tau, s.batch.as_ref().map(|b| b.trajectories)), &mut last),
                    crate::solve::Message::Checkpoint(s, _) => {
                        mark(format!("checkpoint {} trajectories tau {}", s.trajectories, s.time_step_us), &mut last);
                        if s.trajectories > pilot {
                            panic!("T2S-RUN the product's first batch");
                        }
                    }
                })
            }));
            mark("end".into(), &mut last);
            let how = match ended {
                Ok(Ok(s)) => format!("solved: tau {} us, {} trajectories", s.time_step_us, s.trajectories),
                Ok(Err(e)) => format!("refused: {e}"),
                Err(_) => "the product began".to_string(),
            };
            let (steps, refused) = total.get();
            eprintln!("T2S-RUN {name} {label} ended at {:.1} s ({how}); steps {steps} refused {refused}; process CPU {:.1} s", start.0.elapsed().as_secs_f64(), process_seconds() - start.1);
        }
    }
}

/// This thread's affinity (Windows' `SetThreadAffinityMask`).
fn pin(mask: usize) {
    extern "system" {
        fn GetCurrentThread() -> *mut std::ffi::c_void;
        fn SetThreadAffinityMask(thread: *mut std::ffi::c_void, mask: usize) -> usize;
    }
    // SAFETY: the current thread's pseudo handle and a mask of its process's processors.
    let old = unsafe { SetThreadAffinityMask(GetCurrentThread(), mask) };
    assert!(old != 0, "SetThreadAffinityMask {mask:#x}");
}

/// A step's time on each logical processor, alone and with every processor busy (the coordinator's 11-22 ask: the run's
/// step against the pinned one, by the cores' kinds): T2S_STEPS (6) states of a walk from `r1prop::start` at the scene's
/// τ, each step on a fresh scratch. Alone: each processor in turn, the least of T2S_REPEAT (3) a state, the others idle
/// (as far as the machine is ours). Busy: a thread on every processor at once, each pinned to its own, each stepping
/// the same states round after round for T2S_BUSY seconds (4): per processor its mean and its least over the rounds.
#[test]
#[ignore]
fn t2step_cores() {
    let (steps, repeat, busy) = (env("T2S_STEPS", 6.0) as usize, env("T2S_REPEAT", 3.0) as usize, env("T2S_BUSY", 4.0));
    let processors = std::thread::available_parallelism().map_or(1, |n| n.get());
    for (name, request) in scenes() {
        let tau = tau_of(&name);
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let mut states = vec![start(&run, &request, chief, 3)];
            while states.len() < steps {
                let t = states.last().unwrap().clone();
                let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere)) else { break };
                states.push(step.end);
            }
            let one = |t: &crate::step::Trajectory| -> f64 {
                let clock = std::time::Instant::now();
                let _ = Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], &mut Scratch::default(), &Everywhere));
                clock.elapsed().as_secs_f64() * 1e6
            };
            // Alone, processor by processor.
            let alone: Vec<f64> = (0..processors)
                .map(|p| {
                    let (states, one) = (&states, &one);
                    std::thread::scope(|s| {
                        s.spawn(move || {
                            pin(1 << p);
                            states.iter().map(|t| (0..=repeat).map(|_| one(t)).skip(1).fold(f64::INFINITY, crate::linalg::smaller)).sum::<f64>() / states.len() as f64
                        })
                        .join()
                        .unwrap()
                    })
                })
                .collect();
            // Busy: every processor at once.
            let loaded: Vec<(f64, f64, usize)> = std::thread::scope(|s| {
                let handles: Vec<_> = (0..processors)
                    .map(|p| {
                        let (states, one) = (&states, &one);
                        s.spawn(move || {
                            pin(1 << p);
                            let clock = std::time::Instant::now();
                            let (mut sum, mut least, mut rounds) = (0.0, vec![f64::INFINITY; states.len()], 0usize);
                            while clock.elapsed().as_secs_f64() < busy {
                                for (t, l) in states.iter().zip(least.iter_mut()) {
                                    let x = one(t);
                                    sum += x;
                                    *l = crate::linalg::smaller(*l, x);
                                }
                                rounds += 1;
                            }
                            (sum / (rounds * states.len()).max(1) as f64, least.iter().sum::<f64>() / states.len() as f64, rounds)
                        })
                    })
                    .collect();
                handles.into_iter().map(|h| h.join().unwrap()).collect()
            });
            for p in 0..processors {
                eprintln!("T2S-CORE {name} {label} tau {tau} cpu {p:>2}: alone {:>7.1} µs a step | all busy mean {:>7.1} least {:>7.1} µs ({} rounds of {} states)", alone[p], loaded[p].0, loaded[p].1, loaded[p].2, states.len());
            }
            let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
            let rate = |v: &[f64]| v.iter().map(|x| 1.0 / x).sum::<f64>();
            let loaded_mean: Vec<f64> = loaded.iter().map(|x| x.0).collect();
            let best = alone.iter().copied().fold(f64::INFINITY, crate::linalg::smaller);
            eprintln!(
                "T2S-CORES {name} {label}: alone best {best:.1} µs, mean {:.1}; all busy mean {:.1}; steps a µs, the processors alone summed {:.4} against all busy {:.4}: all busy is {:.2} processors' worth of the best alone",
                mean(&alone),
                mean(&loaded_mean),
                rate(&alone),
                rate(&loaded_mean),
                rate(&loaded_mean) * best
            );
        }
    }
}

/// T2STEP2's local measurement (ignored): what the step does over a window W of a trajectory, against the step. Per
/// start i (T2S2_N of them, `start`'s atoms, each its own jumps' stream) a warm-up of T2S2_WARM µs at the finest step,
/// then from that state the window at every step τ_k = W/2^k (k = 0..T2S2_LEVELS), the same draws: the momentum's
/// change along z, the jumps and the photons. Per step their means over the starts, and each step's against the
/// finest's on the same starts with its standard error -- the step's local error over W, what a run adds up W by W.
/// A step the scheme refuses is taken as two halves (as adapt.rs `halving`). T2S_SCENE's first scene, its first model
/// (T2S_<SCENE>), T2S2_W the window (µs).
#[test]
#[ignore]
fn t2step2_window() {
    let (name, request) = scenes().remove(0);
    let (label, run) = models(&name, &request).remove(0);
    let chief = crate::solve::chief(&request);
    let (n, warm, w, levels) = (env("T2S2_N", 256.0) as u64, env("T2S2_WARM", 3.0), env("T2S2_W", 2.8412272892301633), env("T2S2_LEVELS", 7.0) as usize);
    // T2S2_DIVS: the steps W/d for the divisors d given (comma separated, the finest last) instead of the halvings.
    let taus: Vec<f64> = match std::env::var("T2S2_DIVS") {
        Ok(d) => d.split(',').map(|x| w / x.parse::<f64>().unwrap()).collect(),
        Err(_) => (0..levels).map(|k| w / f64::powi(2., k as i32)).collect(),
    };
    let levels = taus.len();
    let finest = taus[levels - 1];
    fn step(run: &Selected, t: &crate::step::Trajectory, tau: f64, s: &mut Scratch, depth: u32) -> crate::step::Trajectory {
        match Midway::new(run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], s, &Everywhere)) {
            Ok(step) => step.end,
            Err(e) => {
                assert!(depth < 30, "refused down to {tau}: {e:?}");
                let half = step(run, t, 0.5 * tau, s, depth + 1);
                step(run, &half, 0.5 * tau, s, depth + 1)
            }
        }
    }
    let go = |t: &crate::step::Trajectory, tau: f64, length: f64| -> crate::step::Trajectory {
        let mut s = Scratch::default();
        let mut t = t.clone();
        let end = t.time + length;
        while t.time < end - 1e-9 * length && t.lost.is_none() {
            t = step(&run, &t, crate::linalg::smaller(tau, end - t.time), &mut s, 0);
        }
        t
    };
    // Per start and step: Δκ_z, jumps, photons.
    let rows: Vec<Vec<[f64; 3]>> = (0..n)
        .into_par_iter()
        .map(|i| {
            let mut t0 = start(&run, &request, chief, i);
            t0 = go(&t0, finest, warm);
            // T2S2_REST: the window from rest at the trap's centre's axis (the warmed state's internal state kept).
            if std::env::var("T2S2_REST").is_ok() {
                t0.kappa = [0.; 3];
                t0.r = [0.; 3];
            }
            // T2S2_NOJUMP: no jump in the window (the threshold below any norm the window reaches).
            if std::env::var("T2S2_NOJUMP").is_ok() {
                t0.threshold = 1e-280;
            }
            taus.iter()
                .map(|&tau| {
                    let t = go(&t0, tau, w);
                    let norm: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
                    let norm0: f64 = t0.psi.iter().map(|z| z.norm_sqr()).sum();
                    let rest = if std::env::var("T2S2_NOJUMP").is_ok() { (norm / norm0).ln() } else { t.scattered[0] - t0.scattered[0] };
                    [t.kappa[2] - t0.kappa[2], f64::from(t.jumps - t0.jumps), rest]
                })
                .collect()
        })
        .collect();
    use rayon::prelude::*;
    let m = n as f64;
    println!("T2S2-WINDOW {name} {label}: {n} starts, warm-up {warm} µs at {finest:.5}, window {w:.5} µs");
    for (k, &tau) in taus.iter().enumerate() {
        let mean = |q: usize| rows.iter().map(|r| r[k][q]).sum::<f64>() / m;
        let diff = |q: usize| {
            let d: Vec<f64> = rows.iter().map(|r| r[k][q] - r[levels - 1][q]).collect();
            let mu = d.iter().sum::<f64>() / m;
            let sd = (d.iter().map(|x| (x - mu) * (x - mu)).sum::<f64>() / (m - 1.)).sqrt();
            (mu, sd / m.sqrt())
        };
        let ((dk, sk), (dj, sj), (dn, sn)) = (diff(0), diff(1), diff(2));
        println!(
            "T2S2-WINDOW tau {tau:.5}: Δκ_z {:.4} /µm, jumps {:.4}, photons (ln N without jumps) {:.6} | against the finest: Δκ_z {:+.5} ± {:.5} ({:+.3} %), jumps {:+.4} ± {:.4}, photons (ln N) {:+.3e} ± {:.1e}",
            mean(0),
            mean(1),
            mean(2),
            dk,
            sk,
            100. * dk / rows.iter().map(|r| r[levels - 1][0]).sum::<f64>() * m,
            dj,
            sj,
            dn,
            sn
        );
    }
}

/// T2STEP2's single-step measurement (ignored): from T2S2_N states of a trajectory (`t2step2_window`'s warm-up, then at
/// rest at the trap's axis: r = κ = 0), one step of each τ with no jump in it, its survival ln(‖ψ₁‖²/‖ψ₀‖²) against the
/// exact flow's of the generator at rest held over the step, ‖e^{−iĤτ}ψ₀‖² (Ĥ the middle's at κ = v = 0, RK4 at 1e-4 µs):
/// what the propagation alone errs by on a long step. T2S2_TAUS the steps (µs, comma separated).
#[test]
#[ignore]
fn t2step2_one() {
    use crate::linalg::{Mat, C64};
    let (name, request) = scenes().remove(0);
    let (label, run) = models(&name, &request).remove(0);
    let chief = crate::solve::chief(&request);
    let n_starts = env("T2S2_N", 16.0) as u64;
    let taus: Vec<f64> = std::env::var("T2S2_TAUS").unwrap_or_else(|_| "0.35515,0.71031,0.94708,1.42061,2.84123".to_string()).split(',').map(|x| x.parse().unwrap()).collect();
    let go = |t: &crate::step::Trajectory, tau: f64, length: f64| -> crate::step::Trajectory {
        let mut s = Scratch::default();
        let mut t = t.clone();
        let end = t.time + length;
        while t.time < end - 1e-9 * length {
            t = Midway::new(&run, &t, crate::linalg::smaller(tau, end - t.time)).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut s, &Everywhere)).unwrap().end;
        }
        t
    };
    let norm = |x: &[C64]| x.iter().map(|z| z.norm_sqr()).sum::<f64>();
    println!("T2S2-ONE {name} {label}: {n_starts} starts at rest");
    for i in 0..n_starts {
        let mut t0 = go(&start(&run, &request, chief, i), 0.17758, 3.0);
        (t0.r, t0.kappa, t0.threshold) = ([0.; 3], [0.; 3], 1e-280);
        let cells: Vec<String> = taus
            .iter()
            .map(|&tau| {
                let mut s = Scratch::default();
                let midway = Midway::new(&run, &t0, tau).unwrap();
                // κ_m as the step predicts it, κ_h taken for the start's (at rest the reference's half-step pull apart).
                let kappa_m = crate::step::predicted(&t0.forms, t0.kappa, tau, t0.time + 0.5 * tau);
                let step = midway.attempt(&t0, &Brownian::STILL, &[], &mut s, &Everywhere).unwrap();
                let engine = (norm(&step.end.psi) / norm(&t0.psi)).ln();
                if std::env::var("T2S2_PICTURE").is_ok() {
                    // The static group's change along the step's path at κ_m: the largest entry of Ĥ(±h) − Ĥ(0).
                    let hm = run.hbar_over_mass();
                    let mid = run.middle(t0.cluster, [0.; 3], t0.time + 0.5 * tau);
                    let along = mid.path(kappa_m, kappa_m.map(|k| hm * k));
                    let m = mid.states();
                    let [mut a0, mut a1, mut a2] = [0, 1, 2].map(|_| Mat::zeros(m, m));
                    mid.generator(&along, -0.5 * tau, &mut a0);
                    mid.generator(&along, 0.0, &mut a1);
                    mid.generator(&along, 0.5 * tau, &mut a2);
                    let change = |x: &Mat, y: &Mat| x.data.iter().zip(&y.data).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max);
                    println!("T2S2-ONE start {i} tau {tau:.4}: κ_m {kappa_m:?} change {:.3e} {:.3e} | {}", change(&a0, &a1), change(&a2, &a1), crate::spectral::tests::census(&s.picture));
                }
                let middle = run.middle(t0.cluster, [0.; 3], t0.time + 0.5 * tau);
                let path = middle.path([0.; 3], [0.; 3]);
                let n = middle.states();
                let mut h = Mat::zeros(n, n);
                middle.generator(&path, 0.0, &mut h);
                let slope = |y: &[C64]| -> Vec<C64> { h.apply(y).into_iter().map(|z| C64::new(z.im, -z.re)).collect() };
                let add = |y: &[C64], k: &[C64], s: f64| -> Vec<C64> { y.iter().zip(k).map(|(y, k)| *y + *k * s).collect() };
                let steps = (tau / 1e-4).ceil() as usize;
                let dt = tau / steps as f64;
                let mut x: Vec<C64> = t0.psi.iter().copied().collect();
                for _ in 0..steps {
                    let k1 = slope(&x);
                    let k2 = slope(&add(&x, &k1, 0.5 * dt));
                    let k3 = slope(&add(&x, &k2, 0.5 * dt));
                    let k4 = slope(&add(&x, &k3, dt));
                    x = (0..n).map(|s| x[s] + (k1[s] + k2[s] * 2.0 + k3[s] * 2.0 + k4[s]) * (dt / 6.0)).collect();
                }
                let exact = (norm(&x) / norm(&t0.psi)).ln();
                // The end's direction: the excited share and the overlap with the exact flow's.
                let e = &step.end.psi;
                let overlap = e.iter().zip(&x).fold(C64::ZERO, |acc, (a, b)| acc + a.conj() * *b).norm_sqr() / (norm(e) * norm(&x));
                let share = |y: &[C64]| y[n - 1].norm_sqr() / norm(y);
                format!("{tau:.4}: {engine:.6} vs {exact:.6} ({:+.2e}) excited {:.4} vs {:.4} 1−overlap {:.1e}", engine - exact, share(e), share(&x), 1. - overlap)
            })
            .collect();
        let total = norm(&t0.psi);
        let pops: Vec<String> = t0.psi.iter().map(|z| format!("{:.2e}", z.norm_sqr() / total)).collect();
        println!("T2S2-ONE start {i}: jumps {} norm {:.3e} populations [{}] | {}", t0.jumps, total, pops.join(" "), cells.join(" | "));
    }
}

