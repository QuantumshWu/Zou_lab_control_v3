//! P4-TABLE's measurements (BRIEF_P4_TABLE): the folded blocks tabled in the tweezer's intensity (model/rs/fold_table.rs)
//! against the node before it, by golden's measure, and their cost. The scenes P4T_SCENE (comma separated; default
//! tilt0, tilt2, R4, GMD1C, PGC3D, TET, R5_PUMP): the six by their requests (R1PROP_REQ, as `r1prop`), the reference
//! scenes by name (refs/scenes); each scene's models P4T_<SCENE> (comma separated: "P" the solve's choice, "ground" its
//! ground states alone; default "P,ground" for PGC3D and TET, "P" for the others). Ignored; run by name, one at a time,
//! the caller pinning the thread to one performance core for the times:
//! `<test exe> step::tests::p4table::<name> --exact --include-ignored --nocapture --test-threads 1`.
//! - `p4table_dump`: per scene and model, at golden's three node points (the centre and the trap region's edge at rest,
//!   the moving point) and P4T_POINTS (24) points of the reach drawn by a fixed seed, the node with its photon forms
//!   (`Selected::node`: the generator by beats, ∇H, the classes' L, ∇L and K), and at golden's three checkpoint points
//!   the model by components (`Rs::at`: H, ∇H, the channels' L, ∇L, K, the lift) -- written into P4T_OUT
//!   (`<scene>_<model>.txt`, one number a line, its quantity's tag first), read by `NEW\p4table\golden.py`.
//! - `p4table_cost`: per model the node's time (each the least of P4T_REPEAT, 30) at the moving point, the centre and
//!   P4T_TIMED (6) points of the reach, plain and with the photon forms, and its counted work; a step's time (the least
//!   of P4T_REPEAT_STEP, 5) on the states of a walk (P4T_STEPS, 12) from the moving point, with its counted work.

use std::fmt::Write as _;

use crate::linalg::{C64, Mat, V3};
use crate::model::selected::{Node, Selected};
use crate::request::{Request, GROUND};
use crate::step::Brownian;

use super::super::{Midway, Scratch};
use super::r1prop::{env, start};
use super::Everywhere;

/// The scenes (module documentation).
fn scenes() -> Vec<(String, Request)> {
    let names = std::env::var("P4T_SCENE").unwrap_or_else(|_| "tilt0,tilt2,R4,GMD1C,PGC3D,TET,R5_PUMP".to_string());
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

/// A scene's models (module documentation): each its label and its run (`Selected::new` on the selection with P
/// replaced, the reference state the chief alone, as the solve makes it).
fn models(name: &str, request: &Request) -> Vec<(String, Selected)> {
    let default = if ["PGC3D", "TET"].contains(&name) { "P,ground" } else { "P" };
    let specs = std::env::var(format!("P4T_{name}")).unwrap_or_else(|_| default.to_string());
    let scene = crate::scene::Scene::new(request).unwrap();
    let clock = std::time::Instant::now();
    let selection = crate::select::choose(&scene).unwrap();
    eprintln!("P4T {name}: chosen in {:.3} s", clock.elapsed().as_secs_f64());
    let reference = crate::solve::alone(crate::solve::chief(request));
    specs
        .split(',')
        .map(|spec| {
            let keep: Vec<usize> = match spec {
                "P" => selection.keep.clone(),
                "ground" => (0..GROUND).collect(),
                other => panic!("unknown model {other}"),
            };
            let clock = std::time::Instant::now();
            let run = Selected::new(crate::scene::Scene::new(request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection.clone() }, reference).unwrap();
            eprintln!("P4T {name} {spec}(P{}): made in {:.3} s", keep.len(), clock.elapsed().as_secs_f64());
            (format!("{spec}{}", keep.len()), run)
        })
        .collect()
}

/// golden's moving point: r = (0.1, 0.2, 0.3) µm, v = 0.1 µm/µs along U.
const R1: V3 = [0.1, 0.2, 0.3];
const U: V3 = [0.3119, 0.5237, 0.7928];

fn moving(run: &Selected) -> V3 {
    use crate::step::Dynamics;
    let norm = crate::linalg::norm3(U);
    U.map(|u| u / norm * 0.1 / run.hbar_over_mass())
}

/// golden's trap region's edge at rest: across the tweezer from its focus where its intensity is 1/20 of the focus'.
fn edge(run: &Selected) -> V3 {
    let g = &run.scene.geometry[run.scene.tweezer()];
    let w = g.waist.expect("the tweezer's waist");
    let (across, _) = crate::linalg::perpendicular(g.direction);
    let rho = w * (20.0_f64.ln() / 2.0).sqrt();
    across.map(|x| x * rho)
}

/// A small xorshift for the points.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn signed(&mut self) -> f64 {
        2.0 * self.next() - 1.0
    }
}

/// The points: golden's three node points, then `count` points of the reach -- across the tweezer within 1.5 waists,
/// along it within 2 Rayleigh lengths, the speed up to 0.4 µm/µs in a random direction (about the escape speed of the
/// scenes' traps).
fn points(run: &Selected, count: usize, seed: u64) -> Vec<(String, V3, V3)> {
    use crate::step::Dynamics;
    let mut out = vec![("centre".to_string(), [0.0; 3], [0.0; 3]), ("edge".to_string(), edge(run), [0.0; 3]), ("moving".to_string(), R1, moving(run))];
    let g = &run.scene.geometry[run.scene.tweezer()];
    let (w, k) = (g.waist.expect("the tweezer's waist"), g.wave_number);
    let rayleigh = 0.5 * k * w * w;
    let (a, b) = crate::linalg::perpendicular(g.direction);
    let mut rng = Rng(seed | 1);
    for i in 0..count {
        let (x, y, z) = (1.5 * w * rng.signed(), 1.5 * w * rng.signed(), 2.0 * rayleigh * rng.signed());
        let r: V3 = std::array::from_fn(|c| x * a[c] + y * b[c] + z * g.direction[c]);
        let d: V3 = [rng.signed(), rng.signed(), rng.signed()];
        let norm = crate::linalg::norm3(d).max(1e-12);
        let speed = 0.4 * rng.next();
        let kappa = d.map(|v| v / norm * speed / run.hbar_over_mass());
        out.push((format!("reach{i}"), r, kappa));
    }
    out
}

fn put(out: &mut String, tag: &str, z: C64) {
    let _ = writeln!(out, "{tag} {:e} {:e}", z.re, z.im);
}

fn put_mat(out: &mut String, tag: &str, m: &Mat) {
    for z in &m.data {
        put(out, tag, *z);
    }
}

fn put_node(out: &mut String, node: &Node) {
    for (_, m) in &node.generator {
        put_mat(out, "gen", m);
    }
    for (_, g) in &node.gradient {
        for m in g {
            put_mat(out, "dh", m);
        }
    }
    for c in &node.classes {
        for axis in &c.axes {
            for j in axis {
                for e in &j.elements {
                    put(out, "l", e.value);
                    for g in e.gradient {
                        put(out, "dl", g);
                    }
                    put(out, "k", e.kick);
                }
            }
        }
    }
}

fn put_model(out: &mut String, m: &crate::model::rs::ModelAt) {
    for p in &m.effective {
        put_mat(out, "ch", &p.value);
        for g in &p.gradient {
            put_mat(out, "cdh", g);
        }
    }
    for c in &m.jumps {
        for p in &c.parts {
            put_mat(out, "cl", &p.value);
            for g in &p.gradient {
                put_mat(out, "cdl", g);
            }
        }
        for p in &c.kicks {
            put_mat(out, "ck", &p.value);
        }
    }
    for p in &m.lift {
        put_mat(out, "lift", &p.value);
        for g in &p.gradient {
            put_mat(out, "dlift", g);
        }
    }
}

#[test]
#[ignore]
fn p4table_dump() {
    let dir = std::path::PathBuf::from(std::env::var("P4T_OUT").expect("P4T_OUT: the directory to write into"));
    let count = env("P4T_POINTS", 24.0) as usize;
    // P4T_MODE (model/rs/fold_table.rs `MODE`): 1 every block decomposed at the node, 2 its generator moved by a unit round-off.
    crate::model::rs::fold::table::MODE.with(|m| m.set(env("P4T_MODE", 0.0) as u8));
    for (name, request) in scenes() {
        for (label, run) in models(&name, &request) {
            let mut out = String::new();
            for (point, r, kappa) in points(&run, count, 0x9E37_79B9_7F4A_7C15 ^ name.len() as u64) {
                let _ = writeln!(out, "node {point}");
                put_node(&mut out, &run.node(r, kappa, true));
            }
            let zero = vec![0.0; run.scene.request.beams.len()];
            for (point, components) in [("centre", run.scene.components([0.0; 3], &zero, 1.0)), ("edge", run.scene.components([0.0; 3], &zero, 0.0)), ("moving", run.scene.components_at(R1, moving(&run)))] {
                let _ = writeln!(out, "checkpoint {point}");
                put_model(&mut out, &run.rs.at(&run.scene, &components));
            }
            std::fs::write(dir.join(format!("{name}_{label}.txt")), out).unwrap();
            eprintln!("P4T-DUMP {name} {label}: written");
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

/// The step τ of a scene's walk: P4T_TAU_<SCENE>, or 0.474 (GMD1C 0.948), the ladders' (P2-GSTEP).
fn tau_of(name: &str) -> f64 {
    std::env::var(format!("P4T_TAU_{name}")).ok().map_or(if name == "GMD1C" { 0.948 } else { 0.474 }, |x| x.parse().unwrap())
}

#[test]
#[ignore]
fn p4table_cost() {
    let (repeat, timed, steps, repeat_step) = (env("P4T_REPEAT", 30.0) as usize, env("P4T_TIMED", 6.0) as usize, env("P4T_STEPS", 12.0) as usize, env("P4T_REPEAT_STEP", 5.0) as usize);
    for (name, request) in scenes() {
        let tau = tau_of(&name);
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let all = points(&run, timed, 0x2545_F491_4F6C_DD1D ^ name.len() as u64);
            let chosen: Vec<&(String, V3, V3)> = all.iter().filter(|p| p.0 != "edge").collect();
            let (mut plain, mut kicked, mut work) = (0.0, 0.0, 0.0);
            let mut each = String::new();
            for (point, r, kappa) in &chosen {
                let p = least(&mut || { std::hint::black_box(run.node(*r, *kappa, false)); }, repeat);
                let k = least(&mut || { std::hint::black_box(run.node(*r, *kappa, true)); }, repeat);
                let w = run.node(*r, *kappa, false).work;
                let _ = write!(each, " {point} {p:.1}/{k:.1}");
                (plain, kicked, work) = (plain + p, kicked + k, work + w);
            }
            let c = chosen.len() as f64;
            eprintln!("P4T-NODE {name} {label}: node {:.1} µs, with the photon forms {:.1} µs, counted {:.4e} real MA (mean over {} points, each the least of {repeat}); per point{each}", plain / c, kicked / c, work / c, chosen.len());
            // A step's time on a walk's states.
            let mut t = start(&run, &request, chief, 3);
            let (mut time, mut counted, mut taken) = (0.0, 0.0, 0usize);
            for _ in 0..steps {
                let mut best = f64::INFINITY;
                let mut next = None;
                for _ in 0..repeat_step {
                    let mut s = Scratch::default();
                    let _ = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut s, &Everywhere));
                    let clock = std::time::Instant::now();
                    let step = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut s, &Everywhere));
                    best = best.min(clock.elapsed().as_secs_f64() * 1e6);
                    if let Ok(step) = step {
                        if next.is_none() {
                            counted += step.work;
                            next = Some(step.end.clone());
                        }
                    }
                }
                match next {
                    Some(end) => {
                        time += best;
                        taken += 1;
                        t = end;
                    }
                    None => break,
                }
            }
            let k = taken.max(1) as f64;
            eprintln!("P4T-STEP {name} {label} tau {tau}: a step {:.1} µs, counted {:.4e} (mean over {taken} states of a walk, each the least of {repeat_step})", time / k, counted / k);
        }
    }
}

/// Per model its plan's tables (model/rs/fold_table.rs): the blocks, the making's time and memory, why a block is not
/// tabled; per tabled block its clusters, small parts (coefficient, guard), pieces (ends, degree, small degree, the
/// largest tail over its floor) and a node's counted work off the table against made at the node.
#[test]
#[ignore]
fn p4table_census() {
    for (name, request) in scenes() {
        for (label, run) in models(&name, &request) {
            let plan = run.rs.plan();
            let t = plan.table(&crate::model::rs::Values::default());
            eprintln!("P4T-TABLE {name} {label}: blocks (size, Q's) {:?}; made in {:.4} s, {:.3} MB; not tabled {:?}", plan.blocks(), t.seconds, t.bytes() as f64 / 1e6, t.why);
            for (b, bt) in t.blocks.iter().enumerate() {
                if let Some(bt) = bt {
                    let clusters: Vec<usize> = bt.clusters.iter().map(|c| c.states.len()).collect();
                    let smalls: Vec<String> = bt.smalls.iter().map(|s| format!("c{}", s.coefficient)).collect();
                    let pieces: Vec<String> = bt.pieces.iter().map(|p| format!("[{:.4},{:.4}] d{} e{} tail {:.2} (q{}{}) A {:?}", p.a, p.b, p.degree, p.small_degree, p.tail, p.worst, if p.stalled { " plateau" } else { "" }, p.guards.iter().map(|g| format!("{g:.1e}")).collect::<Vec<_>>())).collect();
                    eprintln!("P4T-TABLE {name} {label}: block {b} size {}: clusters {clusters:?}; smalls {smalls:?}; pieces {pieces:?}; guard floor {:.2e}; work off the table {:.0} against made {:.0} (CMA)", bt.size, bt.relative, bt.work.0, bt.work.1);
                }
            }
        }
    }
}

/// The node off the tables against the node decomposing every block (model/rs/fold_table.rs `MODE` 0 and 1), in one
/// process, interleaved: per model, at the moving point, the centre and P4T_TIMED (6) points of the reach, the fold's
/// evaluation (`Rs::evaluate`) and the whole node (plain and with the photon forms), each the least of P4T_REPEAT (30)
/// taken in P4T_ROUNDS (4) rounds alternating the two; their counted work.
#[test]
#[ignore]
fn p4table_ab() {
    let (repeat, timed, rounds) = (env("P4T_REPEAT", 30.0) as usize, env("P4T_TIMED", 6.0) as usize, env("P4T_ROUNDS", 4.0) as usize);
    let mode = &crate::model::rs::fold::table::MODE;
    for (name, request) in scenes() {
        for (label, run) in models(&name, &request) {
            let all = points(&run, timed, 0x2545_F491_4F6C_DD1D ^ name.len() as u64);
            let chosen: Vec<&(String, V3, V3)> = all.iter().filter(|p| p.0 != "edge").collect();
            // [mode][evaluate, node, kicked node]
            let mut best = [[0.0f64; 3]; 2];
            let mut counted = [0.0f64; 2];
            for (_, r, kappa) in &chosen {
                let mut least = [[f64::INFINITY; 3]; 2];
                let mut local = crate::model::rs::fold::Local::default();
                let point = run.scene.point_at(*r, *kappa);
                for _ in 0..rounds {
                    for m in 0..2 {
                        mode.with(|x| x.set(m as u8));
                        let e = super::p4table::least(&mut || run.rs.evaluate(&run.scene, &point, &mut local, crate::model::rs::Making::node(false)), repeat);
                        let n = super::p4table::least(&mut || { std::hint::black_box(run.node(*r, *kappa, false)); }, repeat);
                        let k = super::p4table::least(&mut || { std::hint::black_box(run.node(*r, *kappa, true)); }, repeat);
                        for (x, y) in least[m].iter_mut().zip([e, n, k]) {
                            *x = x.min(y);
                        }
                    }
                }
                for m in 0..2 {
                    mode.with(|x| x.set(m as u8));
                    counted[m] += run.node(*r, *kappa, false).work;
                    for (x, y) in best[m].iter_mut().zip(least[m]) {
                        *x += y;
                    }
                }
            }
            mode.with(|x| x.set(0));
            let c = chosen.len() as f64;
            let show = |m: usize| format!("evaluate {:.1} node {:.1} kicked {:.1} µs, counted {:.4e}", best[m][0] / c, best[m][1] / c, best[m][2] / c, counted[m] / c);
            eprintln!("P4T-AB {name} {label}: tabled: {} | decomposed: {} | ratio node {:.3}, evaluate {:.3} (mean over {} points, each the least of {rounds}×{repeat})", show(0), show(1), best[0][1] / best[1][1], best[0][0] / best[1][0], chosen.len());
        }
    }
}

/// Per model and tabled block, the fill's time against the block made at the node (model/rs/fold_table.rs
/// `Table::timed`), at the centre's s and coefficients and at s = ½ with the moving point's.
#[test]
#[ignore]
fn p4table_fill_time() {
    let repeat = env("P4T_REPEAT", 200.0) as usize;
    for (name, request) in scenes() {
        for (label, run) in models(&name, &request) {
            use crate::step::Dynamics;
            let plan = run.rs.plan();
            let t = plan.table(&crate::model::rs::Values::default());
            let point = run.scene.point_at(R1, moving(&run));
            let values = run.rs.tweezer().values(&run.scene, &point);
            let _ = run.hbar_over_mass();
            for (b, tabled, made) in t.timed(plan, 0.5, &values.coefficients, repeat) {
                eprintln!("P4T-FILL {name} {label}: block {b}: fill {tabled:.2} µs against made {made:.2} µs");
            }
        }
    }
}
