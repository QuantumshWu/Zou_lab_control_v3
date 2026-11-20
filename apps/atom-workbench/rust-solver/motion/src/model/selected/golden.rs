//! R1-MODEL's check of the model against the model as it stood at 80b9782e (BRIEF_R1MODEL, # 验收): the six scenes'
//! own requests (V3C_REQ) on PYSEL's P, at three points each, the checkpoint's model (`Rs::at`: its centre and edge, the
//! tweezer full and dark at rest, and a moving point) and the trajectory's node (`Selected::node` with the photon forms:
//! the centre at rest, a point at the trap region's edge at rest, the moving point) and its lift. `r1_golden_dump`
//! writes them as they are (into the directory R1_GOLDEN); `r1_against_golden` reads them back and says, per quantity,
//! the largest difference in units of ε times the quantity's largest entry.

use super::*;
use crate::model::rs::ModelPart;
use crate::request::Request;
use serde_json::{json, Value};

/// The moving point (lin.status.md 1.1): r = (0.1, 0.2, 0.3) µm, v = 0.1 µm/µs along U, along no beam of the six scenes.
const R1: V3 = [0.1, 0.2, 0.3];
const U: V3 = [0.3119, 0.5237, 0.7928];
const SPEED: f64 = 0.1;

fn moving(run: &Selected) -> V3 {
    let norm = crate::linalg::norm3(U);
    U.map(|u| u / norm * SPEED / run.hbar_over_mass())
}

/// The trap region's edge at rest: across the tweezer from its focus where its intensity is 1/20 of the focus'.
fn edge(run: &Selected) -> V3 {
    let g = &run.scene.geometry[run.scene.tweezer()];
    let w = g.waist.expect("the tweezer's waist");
    let (across, _) = crate::linalg::perpendicular(g.direction);
    let rho = w * (20.0_f64.ln() / 2.0).sqrt();
    across.map(|x| x * rho)
}

fn mat(m: &Mat) -> Value {
    let e: Vec<Value> = (0..m.rows).flat_map(|i| (0..m.cols).map(move |j| (i, j))).filter(|&(i, j)| m[(i, j)] != C64::ZERO).map(|(i, j)| json!([i, j, m[(i, j)].re, m[(i, j)].im])).collect();
    json!({"shape": [m.rows, m.cols], "e": e})
}

fn unmat(v: &Value) -> Mat {
    let shape = v["shape"].as_array().unwrap();
    let mut m = Mat::zeros(shape[0].as_u64().unwrap() as usize, shape[1].as_u64().unwrap() as usize);
    for e in v["e"].as_array().unwrap() {
        let e = e.as_array().unwrap();
        m[(e[0].as_u64().unwrap() as usize, e[1].as_u64().unwrap() as usize)] = C64::new(e[2].as_f64().unwrap(), e[3].as_f64().unwrap());
    }
    m
}

fn part(p: &ModelPart) -> Value {
    json!({"key": p.key, "f": p.frequency, "value": mat(&p.value), "grad": p.gradient.iter().map(mat).collect::<Vec<_>>()})
}

fn model(m: &ModelAt) -> Value {
    let jumps: Vec<Value> = m.jumps.iter().map(|c| json!({"emission": c.emission, "fast": c.fast, "parts": c.parts.iter().map(part).collect::<Vec<_>>(), "kicks": c.kicks.iter().map(part).collect::<Vec<_>>()})).collect();
    json!({"states": m.states, "folded": m.folded, "effective": m.effective.iter().map(part).collect::<Vec<_>>(), "jumps": jumps, "lift": m.lift.iter().map(part).collect::<Vec<_>>()})
}

fn exact(run: &Selected, beat: usize) -> Value {
    run.read(|b| {
        let (n, p) = &b.exact[beat];
        json!({"n": n, "p": p.iter().map(|r| [r.num, r.den]).collect::<Vec<_>>(), "value": b.value[beat]})
    })
}

fn node(run: &Selected, node: &Node) -> Value {
    let generator: Vec<Value> = node.generator.iter().map(|(b, m)| json!({"beat": exact(run, *b), "value": mat(m)})).collect();
    let gradient: Vec<Value> = node.gradient.iter().map(|(b, g)| json!({"beat": exact(run, *b), "grad": g.iter().map(mat).collect::<Vec<_>>()})).collect();
    let classes: Vec<Value> = node
        .classes
        .iter()
        .map(|c| {
            let axes: Vec<Value> = c
                .axes
                .iter()
                .map(|axis| {
                    Value::Array(
                        axis.iter()
                            .map(|j| {
                                let e: Vec<Value> = j.elements.iter().map(|e| json!([e.at, e.value.re, e.value.im, e.gradient.iter().flat_map(|g| [g.re, g.im]).collect::<Vec<_>>(), e.kick.re, e.kick.im])).collect();
                                json!({"beat": exact(run, j.beat), "elements": e})
                            })
                            .collect(),
                    )
                })
                .collect();
            json!({"fast": c.fast, "absorbed": c.absorbed, "axes": axes})
        })
        .collect();
    json!({"generator": generator, "gradient": gradient, "classes": classes})
}

/// The six scenes (V3C_REQ, PYSEL's P) as `super::scenes`.
fn runs() -> Vec<(&'static str, Request, Vec<usize>)> {
    super::scenes()
}

/// The checkpoint's three points: its centre and edge (the tweezer full and dark, at rest, no Doppler shift) and the
/// moving point.
fn checkpoint(run: &Selected) -> [(&'static str, crate::scene::Components); 3] {
    let zero = vec![0.0; run.scene.request.beams.len()];
    [("centre", run.scene.components([0.0; 3], &zero, 1.0)), ("edge", run.scene.components([0.0; 3], &zero, 0.0)), ("moving", run.scene.components_at(R1, moving(run)))]
}

/// The node's three points: the centre at rest, the region's edge at rest, the moving point.
fn nodes(run: &Selected) -> [(&'static str, V3, V3); 3] {
    [("centre", [0.0; 3], [0.0; 3]), ("edge", edge(run), [0.0; 3]), ("moving", R1, moving(run))]
}

#[test]
#[ignore]
fn r1_golden_dump() {
    let dir = std::path::PathBuf::from(std::env::var("R1_GOLDEN").expect("R1_GOLDEN: the directory to write into"));
    for (name, request, keep) in runs() {
        let run = run_of(&request, &keep);
        let mut doc = json!({"scene": name, "keep": run.rs.keep});
        for (point, components) in checkpoint(&run) {
            doc["checkpoint"][point] = model(&run.rs.at(&run.scene, &components));
        }
        for (point, r, kappa) in nodes(&run) {
            let made = run.node(r, kappa, true);
            doc["node"][point] = node(&run, &made);
            doc["node"][point]["r"] = json!(r);
            doc["node"][point]["kappa"] = json!(kappa);
            doc["lift"][point] = model(&run.at(r, kappa, Making { model: false, kicks: false, lift: true }));
        }
        std::fs::write(dir.join(format!("{name}.json")), serde_json::to_string(&doc).unwrap()).unwrap();
        eprintln!("{name}: written");
    }
}

/// The largest |a − b| over two lists of matrices matched by a label, and the largest entry of b (the golden's).
#[derive(Default, Clone, Copy)]
struct Apart {
    d: f64,
    s: f64,
}

impl Apart {
    fn take(&mut self, a: Option<&Mat>, b: Option<&Mat>) {
        let size = a.or(b).map(|m| (m.rows, m.cols)).unwrap_or((0, 0));
        let zero = Mat::zeros(size.0, size.1);
        let (a, b) = (a.unwrap_or(&zero), b.unwrap_or(&zero));
        self.d = self.d.max(a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max));
        self.s = self.s.max(b.max_abs());
    }

    fn ratio(&self) -> f64 {
        self.d / (crate::units::UNIT_ROUNDOFF * 2.0 * self.s.max(f64::MIN_POSITIVE))
    }
}

/// Parts by key: (key, value, three gradients).
fn parts(v: &Value) -> Vec<(Vec<i64>, Mat, [Mat; 3])> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let g = p["grad"].as_array().unwrap();
            (p["key"].as_array().unwrap().iter().map(|k| k.as_i64().unwrap()).collect(), unmat(&p["value"]), [unmat(&g[0]), unmat(&g[1]), unmat(&g[2])])
        })
        .collect()
}

thread_local! {
    /// The key's reduction the comparisons take (`reduced`): the frame's primitive vector, set per scene.
    static REDUCE: std::cell::RefCell<Option<Box<dyn Fn(&[i64]) -> Vec<i64>>>> = const { std::cell::RefCell::new(None) };
}

/// Parts summed by the frame's primitive vector of their keys (the lab's frequency: two keys of one frequency, e.g. a
/// product of two beams of one frequency at e_m − e_l and the static part, are one Floquet component).
fn reduced(list: Vec<(Vec<i64>, Mat, [Mat; 3])>) -> Vec<(Vec<i64>, Mat, [Mat; 3])> {
    REDUCE.with(|r| {
        let r = r.borrow();
        let Some(f) = r.as_ref() else { return list };
        let mut out: Vec<(Vec<i64>, Mat, [Mat; 3])> = Vec::new();
        for (k, v, g) in list {
            let key = f(&k);
            match out.iter_mut().find(|x| x.0 == key) {
                Some(x) => {
                    x.1.add_scaled(C64::ONE, &v);
                    for (a, b) in x.2.iter_mut().zip(&g) {
                        a.add_scaled(C64::ONE, b);
                    }
                }
                None => out.push((key, v, g)),
            }
        }
        out
    })
}

fn set_reduce(run: &Selected) {
    let frame = run.frame.clone();
    REDUCE.with(|r| {
        *r.borrow_mut() = Some(Box::new(move |k: &[i64]| {
            let v: crate::floquet::frame::FrameKey = k.iter().copied().chain(std::iter::once(0)).collect();
            frame.primitive(&v).iter().flat_map(|x| [x.num, x.den]).collect()
        }))
    });
}

/// Two lists of parts by key: value and gradients apart.
fn by_key(a: &[(Vec<i64>, Mat, [Mat; 3])], b: &[(Vec<i64>, Mat, [Mat; 3])], value: &mut Apart, gradient: &mut Apart) {
    let keys: std::collections::BTreeSet<Vec<i64>> = a.iter().chain(b).map(|p| p.0.clone()).collect();
    for k in keys {
        let (x, y) = (a.iter().find(|p| p.0 == k), b.iter().find(|p| p.0 == k));
        value.take(x.map(|p| &p.1), y.map(|p| &p.1));
        for i in 0..3 {
            gradient.take(x.map(|p| &p.2[i]), y.map(|p| &p.2[i]));
        }
    }
}

/// A model (`model`'s form) against the golden's: H, ∇H, L, ∇L, K, M.
fn models(a: &Value, b: &Value) -> [Apart; 6] {
    let mut out = [Apart::default(); 6];
    let (mut h, mut dh) = (Apart::default(), Apart::default());
    by_key(&reduced(parts(&a["effective"])), &reduced(parts(&b["effective"])), &mut h, &mut dh);
    (out[0], out[1]) = (h, dh);
    let channels = |v: &Value| -> Vec<((i64, Vec<i64>), Value, Value)> {
        v["jumps"].as_array().unwrap().iter().map(|c| ((c["emission"].as_i64().unwrap(), c["fast"].as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()), c["parts"].clone(), c["kicks"].clone())).collect()
    };
    let (ca, cb) = (channels(a), channels(b));
    let labels: std::collections::BTreeSet<(i64, Vec<i64>)> = ca.iter().chain(&cb).map(|c| c.0.clone()).collect();
    let none = json!([]);
    for l in labels {
        let (x, y) = (ca.iter().find(|c| c.0 == l), cb.iter().find(|c| c.0 == l));
        let (mut kd, mut kg) = (Apart::default(), Apart::default());
        let (mut l, mut dl) = (out[2], out[3]);
        by_key(&reduced(parts(x.map_or(&none, |c| &c.1))), &reduced(parts(y.map_or(&none, |c| &c.1))), &mut l, &mut dl);
        (out[2], out[3]) = (l, dl);
        by_key(&reduced(parts(x.map_or(&none, |c| &c.2))), &reduced(parts(y.map_or(&none, |c| &c.2))), &mut kd, &mut kg);
        out[4].d = out[4].d.max(kd.d);
        out[4].s = out[4].s.max(kd.s);
    }
    let mut g = Apart::default();
    by_key(&reduced(parts(&a["lift"])), &reduced(parts(&b["lift"])), &mut out[5], &mut g);
    out
}

/// A node (`node`'s form) against the golden's, its beats matched by their exact vectors: the generator, H_eff's
/// gradient, the classes' L, ∇L and K by (fast, axis, beat, place).
fn nodes_apart(a: &Value, b: &Value) -> ([Apart; 5], usize, usize) {
    let mut out = [Apart::default(); 5];
    let key = |v: &Value| -> String { format!("{}|{}", v["beat"]["n"], v["beat"]["p"]) };
    let list = |v: &Value, field: &str| -> Vec<(String, Vec<Mat>)> {
        v[field].as_array().unwrap().iter().map(|x| (key(x), if field == "generator" { vec![unmat(&x["value"])] } else { x["grad"].as_array().unwrap().iter().map(unmat).collect() })).collect()
    };
    for (slot, field) in [(0, "generator"), (1, "gradient")] {
        let (x, y) = (list(a, field), list(b, field));
        let labels: std::collections::BTreeSet<String> = x.iter().chain(&y).map(|p| p.0.clone()).collect();
        for l in labels {
            let (p, q) = (x.iter().find(|p| p.0 == l), y.iter().find(|p| p.0 == l));
            let count = p.or(q).unwrap().1.len();
            for i in 0..count {
                out[slot].take(p.map(|p| &p.1[i]), q.map(|q| &q.1[i]));
            }
        }
    }
    // The classes' elements: (fast, axis, beat, place) → (L, ∇L, K).
    type Elements = std::collections::BTreeMap<String, (C64, [C64; 3], C64)>;
    let elements = |v: &Value| -> Elements {
        let mut out = Elements::new();
        for c in v["classes"].as_array().unwrap() {
            for (axis, jumps) in c["axes"].as_array().unwrap().iter().enumerate() {
                for j in jumps.as_array().unwrap() {
                    for e in j["elements"].as_array().unwrap() {
                        let e = e.as_array().unwrap();
                        let g = e[3].as_array().unwrap();
                        let z = |i: usize| C64::new(g[2 * i].as_f64().unwrap(), g[2 * i + 1].as_f64().unwrap());
                        let entry = out.entry(format!("{}|{axis}|{}|{}", c["fast"], key(j), e[0])).or_insert((C64::ZERO, [C64::ZERO; 3], C64::ZERO));
                        entry.0 += C64::new(e[1].as_f64().unwrap(), e[2].as_f64().unwrap());
                        for i in 0..3 {
                            entry.1[i] += z(i);
                        }
                        entry.2 += C64::new(e[4].as_f64().unwrap(), e[5].as_f64().unwrap());
                    }
                }
            }
        }
        out
    };
    let (x, y) = (elements(a), elements(b));
    let labels: std::collections::BTreeSet<&String> = x.keys().chain(y.keys()).collect();
    let zero = (C64::ZERO, [C64::ZERO; 3], C64::ZERO);
    for l in labels {
        let (p, q) = (x.get(l).unwrap_or(&zero), y.get(l).unwrap_or(&zero));
        out[2].d = out[2].d.max((p.0 - q.0).abs());
        out[2].s = out[2].s.max(q.0.abs());
        for i in 0..3 {
            out[3].d = out[3].d.max((p.1[i] - q.1[i]).abs());
            out[3].s = out[3].s.max(q.1[i].abs());
        }
        out[4].d = out[4].d.max((p.2 - q.2).abs());
        out[4].s = out[4].s.max(q.2.abs());
    }
    (out, a["generator"].as_array().unwrap().len(), b["generator"].as_array().unwrap().len())
}

/// The model now against the golden (R1_GOLDEN): per scene and point, each quantity's largest difference over ε times
/// its largest entry (u = ε/2 the unit roundoff: `Apart::ratio` divides by 2u = ε).
#[test]
#[ignore]
fn r1_against_golden() {
    let dir = std::path::PathBuf::from(std::env::var("R1_GOLDEN").expect("R1_GOLDEN: the golden's directory"));
    for (name, request, keep) in runs() {
        let golden: Value = serde_json::from_str(&std::fs::read_to_string(dir.join(format!("{name}.json"))).unwrap()).unwrap();
        let run = run_of(&request, &keep);
        set_reduce(&run);
        assert_eq!(serde_json::to_value(&run.rs.keep).unwrap(), golden["keep"], "{name}: P");
        for (point, components) in checkpoint(&run) {
            let now = model(&run.rs.at(&run.scene, &components));
            let d = models(&now, &golden["checkpoint"][point]);
            eprintln!("R1GOLD {name} checkpoint {point}: H {:.1} ∇H {:.1} L {:.1} ∇L {:.1} K {:.1} M {:.1} (ε·max; max H {:.3e})", d[0].ratio(), d[1].ratio(), d[2].ratio(), d[3].ratio(), d[4].ratio(), d[5].ratio(), d[0].s);
        }
        for (point, r, kappa) in nodes(&run) {
            let made = run.node(r, kappa, true);
            let (d, beats, golden_beats) = nodes_apart(&node(&run, &made), &golden["node"][point]);
            let lift = models(&model(&run.at(r, kappa, Making { model: false, kicks: false, lift: true })), &golden["lift"][point]);
            eprintln!(
                "R1GOLD {name} node {point}: generator {:.1} ∇H {:.1} L {:.1} ∇L {:.1} K {:.1} lift {:.1} (ε·max); beats {beats} (golden {golden_beats})",
                d[0].ratio(),
                d[1].ratio(),
                d[2].ratio(),
                d[3].ratio(),
                d[4].ratio(),
                lift[5].ratio()
            );
        }
    }
}

/// The lab grouping against the by-key contraction reduced by the frame (one fold, two orders of its sums): on the six
/// scenes at the node's three points, every quantity to rounding.
#[test]
#[ignore]
fn r1_lab_against_key() {
    for (name, request, keep) in runs() {
        let run = run_of(&request, &keep);
        set_reduce(&run);
        for (point, r, kappa) in nodes(&run) {
            let lab = model(&run.at(r, kappa, Making::ALL));
            let key = model(&run.rs.local(&run.scene, &run.scene.point_at(r, kappa)).components());
            let d = models(&lab, &key);
            eprintln!("R1LAB {name} {point}: H {:.1} ∇H {:.1} L {:.1} ∇L {:.1} K {:.1} M {:.1}", d[0].ratio(), d[1].ratio(), d[2].ratio(), d[3].ratio(), d[4].ratio(), d[5].ratio());
        }
    }
}

/// R1-MODEL: one node's parts timed (µs, least of R1_REPEAT runs, one thread): the point and the values, the fold's
/// numbers, the contraction, the node's filling; with the plan's sizes.
#[test]
#[ignore]
fn r1_node_parts() {
    let repeat: usize = std::env::var("R1_REPEAT").ok().map_or(5, |s| s.parse().unwrap());
    for (name, request, keep) in runs() {
        let run = run_of(&request, &keep);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let (r, kappa) = (R1, [speed, 0.0, 0.0]);
        let least = |f: &mut dyn FnMut()| (0..repeat).map(|_| {
            let clock = std::time::Instant::now();
            f();
            clock.elapsed().as_secs_f64() * 1e6
        }).fold(f64::INFINITY, f64::min);
        let mut work = Work::default();
        let mut node = run.template.clone();
        run.scene.point_into(r, kappa, &mut work.point);
        run.rs.evaluate(&run.scene, &work.point, &mut work.local);
        work.out.contract(run.rs.plan(), &run.lab, &work.local, None, Making { model: true, kicks: false, lift: false });
        let point = least(&mut || {
            run.scene.point_into(r, kappa, &mut work.point);
        });
        let values = least(&mut || {
            let mut v = std::mem::take(&mut work.local.values);
            run.rs.tweezer().values_into(&run.scene, &work.point, &mut v);
            work.local.values = v;
        });
        let evaluate = least(&mut || run.rs.evaluate(&run.scene, &work.point, &mut work.local));
        let contract = least(&mut || work.out.contract(run.rs.plan(), &run.lab, &work.local, None, Making { model: true, kicks: false, lift: false }));
        let fill = least(&mut || run.fill(&mut node, &work.out, false, &mut work.jumps, &mut work.product));
        let whole = least(&mut || {
            std::hint::black_box(run.node(r, kappa, false));
        });
        let plan = run.rs.plan();
        eprintln!(
            "R1PARTS {name}: point {point:.1} values {values:.1} evaluate {evaluate:.1} contract {contract:.1} fill {fill:.1}, node {whole:.1} µs; counted {:.3e} + {:.3e} CMA; blocks {:?}, {} pieces, outputs {} H / {} channels, beats {}",
            work.local.work,
            work.out.work,
            plan.blocks(),
            plan.pieces(),
            run.lab.outputs.len(),
            run.lab.channel_outputs.len(),
            node.generator.len()
        );
        eprintln!("R1SIZES {name}: zs ws h_pairs j_pairs h_flops j_flops {:?}; amplitudes {}", run.lab.sizes(plan), plan.amplitudes.len());
    }
}

/// R1-MODEL item 2: every beat of the node's generator and of its gradient with its frequency, its exact vector and its
/// largest element at the centre and at the moving point -- H_eff's part (θ and the decay left out) and the decay's apart
/// --, and the step's candidate tones; per scene (V3E_SCENE one of them).
#[test]
#[ignore]
fn r1_tones() {
    for (name, request, keep) in runs() {
        let run = run_of(&request, &keep);
        let n = keep.len();
        for (point, r, kappa) in [nodes(&run)[0], nodes(&run)[2]] {
            let full = run.node(r, kappa, false);
            // H_eff alone: the node without the decay (its classes' values zeroed before the decay is made).
            let mut bare = run.template.clone();
            let mut work = Work::default();
            run.scene.point_into(r, kappa, &mut work.point);
            run.rs.evaluate(&run.scene, &work.point, &mut work.local);
            work.out.contract(run.rs.plan(), &run.lab, &work.local, None, Making { model: true, kicks: false, lift: false });
            run.fill(&mut bare, &work.out, false, &mut work.jumps, &mut work.product);
            let (values, exact) = run.components();
            for (k, ((beat, g), (_, gb))) in full.generator.iter().zip(&bare.generator).enumerate() {
                let total = g.max_abs();
                let _ = gb;
                let grad = full.gradient.iter().find(|x| x.0 == *beat).map_or(0.0, |x| x.1.iter().map(Mat::max_abs).fold(0.0, f64::max));
                eprintln!("R1TONE {name} {point}: slot {k} beat {beat} Ω {:+.6e} rad/µs exact {:?}: generator max {total:.3e}, gradient max {grad:.3e}", values[*beat], exact[*beat]);
            }
            let middle = run.middle(0, r, 0.0);
            let path = middle.path(kappa, [0.0; 3]);
            let _ = middle.node(&path, 0.0, false);
            let (mut beats, mut keys) = (vec![], vec![]);
            middle.beats(&path, &mut beats, &mut keys);
            eprintln!("R1TONE {name} {point}: {} candidate tones {:?} (n {n})", beats.len(), beats);
        }
    }
}

/// R1-MODEL item 3: the nodes a step makes (STEPCOST finding 1): from the reference state at the trap's centre at the
/// thermal speed along x, R1_STEPS steps at τ = V3E_TAU µs (0.948) with the Brownian draws and the jumps, per step the
/// nodes made without the photon forms and with them; and the gradient's slope taken over the step's own ends.
#[test]
#[ignore]
fn r1_nodes_per_step() {
    use crate::step::{Brownian, Midway, Scratch, Trajectory};
    let tau = super::env("V3E_TAU", 0.948);
    let steps = super::env("R1_STEPS", 40.0) as usize;
    for (name, request, keep) in runs() {
        let run = run_of(&request, &keep);
        let chief = crate::solve::chief(&request);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let mut t = Trajectory::new(&run, chief, R1, [speed, 0.0, 0.0], 11, 0);
        let mut scratch = Scratch::default();
        MADE.with(|m| *m.borrow_mut() = [0; 2]);
        let (mut taken, mut jumps) = (0, 0);
        for _ in 0..steps {
            let Ok(m) = Midway::new(&run, &t, tau) else { break };
            let Ok(step) = m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &super::Anywhere) else { break };
            jumps += step.made.iter().filter(|m| m.kick.is_some()).count();
            t = step.end;
            taken += 1;
        }
        let made = MADE.with(|m| *m.borrow());
        eprintln!("R1NODES {name}: τ {tau}, {taken} steps, {jumps} jumps: {:.3} nodes a step without the photon forms, {:.3} with them", made[0] as f64 / taken as f64, made[1] as f64 / taken as f64);
    }
}
