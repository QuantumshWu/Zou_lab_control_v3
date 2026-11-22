//! P2-GSTEP's measurements (BRIEF_P2_GSTEP): a cluster of ground states alone stepped by one node, and the kick's sums
//! by groups of its eigencomponents (step.rs, # One node a step, # The quadrature by groups). The scenes GSTEP_SCENE
//! (comma separated; default PGC3D, TET, GMD1C, R5_PUMP): the six by their requests (R1PROP_REQ, as `r1prop`), the
//! reference scenes by name (refs/scenes); each scene's models GSTEP_<SCENE> (comma separated: "P" the solve's choice,
//! "ground" its ground states alone; default "ground" for PGC3D and TET, "P" for the others). Ignored; run by name, one
//! at a time, the caller pinning the thread to one performance core:
//! `<test exe> step::tests::gstep::<name> --exact --include-ignored --nocapture --test-threads 1`.
//! - `gstep_census`: per model its states, ground states, exact blocks and candidate tones at its first middle; per step
//!   of a walk whether it took one node, its stretches' nodes and their cut.
//! - `gstep_filon`: per piece of a walk -- the step's start state (plain), a jump's state (after a jump), the bare start
//!   -- the sums by the engine's rule (`Groups::choose`) and by the plain rule (`count`) against brute force: S per unit
//!   width, the hazard H = tr(Γ S), the kick K = Re tr(Ĝ S) over its largest component, the hazard over a span inside
//!   the piece (a root's: `Groups::spanned`); the brute force against itself at twice its panels; nodes, groups, the
//!   series' order and labels, the counted work and the time of each rule.
//! - `gstep_one_node`: one node a step against three (the model with its cluster said to hold an explicit state,
//!   `Three`), the step's end against three nodes' at τ/16 (sixteen steps) and at τ/2 (two steps), no jump; and a
//!   step's time either way, each its least of GSTEP_REPEAT, with its count by part (`Price::parts`).
//! - `gstep_ladder`: the probe and the pilot on the model (solve.rs `solve_on`), the step each checkpoint stands at.

use crate::linalg::{norm_sqr, C64};
use crate::model::selected::Selected;
use crate::model::{Cluster, Point, Quadratic, STATES};
use crate::request::{Request, GROUND};
use crate::step::{Brownian, Dynamics, Middle, Trajectory};

use super::super::{count, panels, pictured, predicted, whole, Along, Groups, Midway, Scratch, Sums, FORMS, SUMS};
use super::r1prop::{env, start};
use super::Everywhere;

/// The scenes (module documentation).
fn scenes() -> Vec<(String, Request)> {
    let names = std::env::var("GSTEP_SCENE").unwrap_or_else(|_| "PGC3D,TET,GMD1C,R5_PUMP".to_string());
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
    let default = if ["PGC3D", "TET"].contains(&name) { "ground" } else { "P" };
    let specs = std::env::var(format!("GSTEP_{name}")).unwrap_or_else(|_| default.to_string());
    let scene = crate::scene::Scene::new(request).unwrap();
    let selection = crate::select::choose(&scene).unwrap();
    let reference = crate::solve::alone(crate::solve::chief(request));
    specs
        .split(',')
        .map(|spec| {
            let keep: Vec<usize> = match spec {
                "P" => selection.keep.clone(),
                "ground" => (0..GROUND).collect(),
                other => panic!("unknown model {other}"),
            };
            let run = Selected::new(crate::scene::Scene::new(request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection.clone() }, reference).unwrap();
            (format!("{spec}(P{})", keep.len()), run)
        })
        .collect()
}

/// The step τ of a scene's walk: GSTEP_TAU_<SCENE>, or the scene's ladder's (0.474; GMD1C 0.948).
fn tau_of(name: &str) -> f64 {
    std::env::var(format!("GSTEP_TAU_{name}")).ok().map_or(if name == "GMD1C" { 0.948 } else { 0.474 }, |x| x.parse().unwrap())
}

/// A model whose cluster is said to hold an explicit excited state (its ground states one fewer), all else the model's:
/// its steps take three nodes (step.rs, # One node a step) -- the reference and the step before this line.
struct Three<'a, D: Dynamics> {
    model: &'a D,
    cluster: Cluster,
}

impl<'a, D: Dynamics> Three<'a, D> {
    fn new(model: &'a D) -> Three<'a, D> {
        let mut cluster = model.cluster(0).clone();
        cluster.ground = cluster.states.len().saturating_sub(1);
        Three { model, cluster }
    }
}

impl<D: Dynamics> Dynamics for Three<'_, D> {
    type Middle<'b>
        = D::Middle<'b>
    where
        Self: 'b;
    fn hbar_over_mass(&self) -> f64 {
        self.model.hbar_over_mass()
    }
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.model.reference(r)
    }
    fn cluster(&self, _: usize) -> &Cluster {
        &self.cluster
    }
    fn cluster_of(&self, ground: usize) -> usize {
        self.model.cluster_of(ground)
    }
    fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> Self::Middle<'_> {
        self.model.middle(cluster, r, t)
    }
    fn emit<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Self::Middle<'m>, &<Self::Middle<'m> as Middle>::Path, f64)>, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<crate::step::Emitted> {
        self.model.emit(cluster, at, on, psi, draw)
    }
    fn landed<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Self::Middle<'m>, &<Self::Middle<'m> as Middle>::Path)>, f: &[C64], threshold: f64) -> crate::step::Landing {
        self.model.landed(cluster, at, on, f, threshold)
    }
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        self.model.energy(cluster, at, psi)
    }
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.model.lost(cluster, at, psi)
    }
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.model.readout(cluster, at, psi)
    }
    fn photon_work(&self) -> f64 {
        self.model.photon_work()
    }
}

/// A step of τ on a model (the engine's: its structure decides its nodes).
fn stepped<D: Dynamics>(model: &D, t: &Trajectory, tau: f64, scratch: &mut Scratch) -> Result<crate::step::Step, crate::step::Refused> {
    Midway::new(model, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], scratch, &Everywhere))
}

/// The least of `times` runs of `f` (s).
fn least(f: &mut dyn FnMut(), times: usize) -> f64 {
    (0..times)
        .map(|_| {
            let clock = std::time::Instant::now();
            f();
            clock.elapsed().as_secs_f64()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
#[ignore]
fn gstep_census() {
    let steps = env("GSTEP_STEPS", 12.0) as usize;
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
            println!("GSTEP-CENSUS {name} {label}: {} states, {} ground, blocks {:?}, candidate tones {} {:?}", cluster.states.len(), cluster.ground, middle.blocks().iter().map(Vec::len).collect::<Vec<_>>(), beats.len(), beats);
            let mut scratch = Scratch::default();
            for k in 0..steps {
                match stepped(&run, &t, tau, &mut scratch) {
                    Ok(step) => {
                        let parts: Vec<String> = step.prices.iter().map(|(_, p)| format!("still {} nodes {} grouped {:.0} bandwidth {:.1}", p.still, p.nodes, p.grouped, p.bandwidth)).collect();
                        println!("GSTEP-CENSUS {name} {label} step {k}: jumps {} | {}", step.made.iter().filter(|m| m.kick.is_some()).count(), parts.join(" ; "));
                        t = step.end;
                    }
                    Err(e) => {
                        println!("GSTEP-CENSUS {name} {label} step {k}: refused {e:?}");
                        break;
                    }
                }
            }
        }
    }
}

/// The step's sums (`Sums`: one where it takes one node, `still`; else the three nodes' and six a tone's) over [a, b] by
/// brute force on a picture launched at or before a, laid as the step lays them (`accumulate`): 32-node Gauss panels
/// each 4 rad of the quotient's bandwidth, 8 more, `split` times as many.
fn brute(picture: &mut crate::spectral::Picture, still: bool, h: f64, a: f64, b: f64, split: usize) -> Vec<f64> {
    use super::super::row;
    let n = picture.states();
    let slots = Sums { still, h, tones: &picture.beats }.count();
    let size = slots * 2 * n * row(n);
    let (mut sums, mut lost, mut node) = (vec![0.0; size], vec![0.0; size], vec![0.0; size]);
    let mut x = [C64::ZERO; STATES];
    let rule = crate::special::gauss_legendre(32);
    let count = split * ((picture.quotient * (b - a).abs() / 4.0).ceil() as usize + 8);
    let width = (b - a) / count as f64;
    for p in 0..count {
        for &(t, g) in &rule {
            let u = a + (p as f64 + 0.5 + 0.5 * t) * width;
            picture.at(u, &mut x[..n]);
            node.fill(0.0);
            Sums { still, h, tones: &picture.beats }.gather(u, 0.5 * width * g, &x[..n], None, &mut node);
            // Neumaier's compensated sum: what each addition rounds off is kept and added back at the end.
            for ((s, c), v) in sums.iter_mut().zip(lost.iter_mut()).zip(&node) {
                let t = *s + v;
                *c += if s.abs() >= v.abs() { (*s - t) + v } else { (v - t) + *s };
                *s = t;
            }
        }
    }
    sums.iter().zip(&lost).map(|(s, c)| s + c).collect()
}

/// The step's sums over [a, b] by the plain rule (`count`, no root) or by the groups' cut already chosen
/// (`Groups::spanned`, where `grouped`), laid as the step lays them: the sums, the nodes, whether every node's series
/// held, the groups' counted work and the picture's.
fn by_rule(picture: &mut crate::spectral::Picture, groups: &mut Groups, still: bool, h: f64, a: f64, b: f64, grouped: bool) -> (Vec<f64>, usize, bool, f64, f64) {
    use super::super::row;
    let n = picture.states();
    let slots = Sums { still, h, tones: &picture.beats }.count();
    let mut sums = vec![0.0; slots * 2 * n * row(n)];
    let mut x = [C64::ZERO; STATES];
    let (work, made) = (groups.work, picture.work);
    let (mut nodes, mut held) = (0, true);
    if grouped {
        let laid = groups.spanned(b - a);
        let mut from = a;
        for (end, rule) in panels(a, b, laid) {
            groups.begin(picture.origin(), from, end, laid.1);
            for (u, w, k) in rule {
                groups.parts(picture, u);
                held &= groups.gather(k, u, w, &Sums { still, h, tones: &picture.beats }, &mut sums, None);
                nodes += 1;
            }
            from = end;
        }
    } else {
        for (_, rule) in panels(a, b, count(b - a, picture.quotient, FORMS, 2, || false)) {
            for (u, w, _) in rule {
                picture.at(u, &mut x[..n]);
                Sums { still, h, tones: &picture.beats }.gather(u, w, &x[..n], None, &mut sums);
                nodes += 1;
            }
        }
    }
    (sums, nodes, held, groups.work - work, picture.work - made)
}

#[test]
#[ignore]
fn gstep_filon() {
    use super::super::{hazard, row};
    let steps = env("GSTEP_STEPS", 24.0) as usize;
    for (name, request) in scenes() {
        let tau = tau_of(&name);
        let h = 0.5 * tau;
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let mut t = start(&run, &request, chief, 3);
            if let Some(s) = std::env::var("GSTEP_STATE").ok().and_then(|x| x.parse::<usize>().ok()) {
                let mut psi = crate::step::Internal::zeros(t.psi.len());
                psi[s] = C64::ONE;
                t = Trajectory::new(&run, 0, t.r, t.kappa, t.seed, t.index);
                t.psi = psi;
            }
            let mut scratch = Scratch::default();
            // Per kind (plain, after a jump, start): pieces; worst [S, H, K, H(u*)] of the engine's rule and of the plain
            // rule; the brute force's self-check; nodes [plain, engine]; groups, series' order, labels; counted work
            // [plain, engine (groups + picture), the choice's]; time µs [plain, the choice, the engine's sums]; pieces by
            // groups; pieces whose series did not hold; bandwidths [ν_Q, ν_env].
            #[derive(Default, Clone)]
            struct Tally {
                pieces: usize,
                engine: [f64; 4],
                plain: [f64; 4],
                check: f64,
                nodes: [usize; 2],
                groups: usize,
                order: usize,
                labels: usize,
                work: [f64; 3],
                time: [f64; 3],
                grouped: usize,
                failed: usize,
                bandwidth: [f64; 2],
            }
            let mut tally = vec![Tally::default(); 3];
            for k in 0..steps {
                let Ok(midway) = Midway::new(&run, &t, tau) else { break };
                let taken = midway.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Everywhere);
                let n = t.psi.len();
                let kappa_m = predicted(&t.forms, midway.reached.1, tau, t.time + h);
                let along = Along::new(&midway.generator, kappa_m, kappa_m.map(|k| midway.hbar_over_mass * k));
                let mut pieces = vec![(if k == 0 { 2 } else { 0 }, -h, along.turned(&t.psi, -h, true)[..n].to_vec())];
                if let Ok(step) = &taken {
                    if let Some(made) = step.made.iter().find(|m| m.kick.is_some() && m.state.cluster == t.cluster) {
                        let u = (made.state.time - t.time) - h;
                        if u < h {
                            pieces.push((1, u, along.turned(&made.state.psi, u, true)[..n].to_vec()));
                        }
                    }
                }
                let ground = {
                    let c = run.cluster(t.cluster);
                    c.ground == c.states.len()
                };
                for (kind, from, x) in pieces {
                    let mut s = Scratch::default();
                    s.fit(n, SUMS);
                    let mut still = along.middle.still();
                    if pictured(&along, &mut s, tau, Some(&x), ground, &mut still).is_err() {
                        continue;
                    }
                    let decay = s.carve().decay.to_vec();
                    let Scratch { picture, groups, .. } = &mut s;
                    picture.launch(from, &x);
                    let made = picture.work;
                    let tones = picture.beats.len();
                    groups.choose(picture, from, h, (still, tones), &mut |_: &mut crate::spectral::Picture| false);
                    let chosen = picture.work - made + groups.work;
                    let grouped = groups.groups > 1;
                    let inner = from + 0.618 * (h - from);
                    let (engine, nodes, held, worked, stops) = by_rule(picture, groups, still, h, from, h, grouped);
                    let (engine_in, _, held_in, _, _) = by_rule(picture, groups, still, h, from, inner, grouped);
                    let (plain, plain_nodes, _, _, plain_stops) = by_rule(picture, groups, still, h, from, h, false);
                    let (plain_in, _, _, _, _) = by_rule(picture, groups, still, h, from, inner, false);
                    let (exact, exact_in, check) = (brute(picture, still, h, from, h, 1), brute(picture, still, h, from, inner, 1), brute(picture, still, h, from, h, 2));
                    let times = [
                        least(&mut || { std::hint::black_box(by_rule(picture, groups, still, h, from, h, false)); }, 5) * 1e6,
                        least(&mut || { std::hint::black_box(groups.choose(picture, from, h, (still, tones), &mut |_: &mut crate::spectral::Picture| false)); }, 5) * 1e6,
                        least(&mut || { std::hint::black_box(by_rule(picture, groups, still, h, from, h, grouped)); }, 5) * 1e6,
                    ];
                    // The choice's parts on the cut taken: its envelopes' bound, its labels, its rule's sizes (µs).
                    if let (Some(cut), true) = (groups.cut, std::env::var("GSTEP_PARTS").is_ok()) {
                        let shift: Vec<f64> = (0..n).map(|a| cut.centres[usize::from(cut.group[a])]).collect();
                        let envelope = least(&mut || { std::hint::black_box(picture.envelope(from, &shift)); }, 5) * 1e6;
                        let labels = least(&mut || { std::hint::black_box(groups.label(cut.groups, cut.order, &cut.linked, &cut.centres, usize::MAX)); }, 5) * 1e6;
                        let sizes = least(&mut || { std::hint::black_box(super::super::arranged(h - from, |w| super::super::resolved(cut.bandwidth, w, FORMS, 2, true), |m| m as f64, || false)); }, 5) * 1e6;
                        let plain = least(&mut || { std::hint::black_box(count(h - from, picture.quotient, FORMS, 2, || false)); }, 5) * 1e6;
                        println!("GSTEP-PARTS {name} {label} step {k} kind {kind}: envelope {envelope:.2} us, labels {labels:.2} us, a layout at the envelopes' bandwidth {sizes:.2} us, the plain layout {plain:.2} us");
                    }
                    let width = (h - from).abs();
                    let blocks = picture.blocks.clone();
                    let block_of: Vec<usize> = (0..n).map(|i| blocks.iter().position(|b| b.contains(&i)).unwrap_or(0)).collect();
                    let slots = exact.len() / (2 * n * row(n));
                    let wholes = |sums: &[f64]| -> Vec<Vec<C64>> {
                        (0..slots)
                            .map(|j| {
                                let mut m = vec![C64::ZERO; n * n];
                                whole(sums, j, n, &mut m);
                                m
                            })
                            .collect()
                    };
                    let apart = |x: &[f64], y: &[f64]| {
                        let (x, y) = (wholes(x), wholes(y));
                        x.iter().zip(&y).flat_map(|(p, q)| (0..n * n).filter(|&e| block_of[e / n] == block_of[e % n]).map(move |e| (p[e] - q[e]).abs())).fold(0.0, crate::linalg::larger)
                    };
                    let kick = |sums: &[f64]| -> [f64; 3] {
                        let w = wholes(sums);
                        if still {
                            return along.middle.force(&along.path, 0.0, &w[0]);
                        }
                        let mut f = [0.0; 3];
                        for (j, s) in [-h, 0.0, h].into_iter().enumerate() {
                            let g = along.middle.force(&along.path, s, &w[j]);
                            for c in 0..3 {
                                f[c] += g[c];
                            }
                        }
                        f
                    };
                    let kb = kick(&exact);
                    let scale = crate::linalg::larger(kb.iter().map(|x| x.abs()).fold(0.0, crate::linalg::larger), f64::MIN_POSITIVE);
                    let errors = |x: &[f64], x_in: &[f64]| {
                        let kx = kick(x);
                        [apart(x, &exact) / width, (hazard(&decay, x, n) - hazard(&decay, &exact, n)).abs(), kx.iter().zip(&kb).map(|(p, q)| (p - q).abs()).fold(0.0, crate::linalg::larger) / scale, (hazard(&decay, x_in, n) - hazard(&decay, &exact_in, n)).abs()]
                    };
                    let worse = |w: &mut [f64; 4], e: [f64; 4]| {
                        for (a, b) in w.iter_mut().zip(e) {
                            *a = crate::linalg::larger(*a, b);
                        }
                    };
                    let entry = &mut tally[kind];
                    entry.pieces += 1;
                    worse(&mut entry.engine, errors(&engine, &engine_in));
                    worse(&mut entry.plain, errors(&plain, &plain_in));
                    entry.check = crate::linalg::larger(entry.check, apart(&exact, &check) / width);
                    entry.nodes[0] += plain_nodes;
                    entry.nodes[1] += nodes;
                    entry.groups += groups.groups;
                    entry.grouped += usize::from(grouped);
                    entry.failed += usize::from(!(held && held_in));
                    if grouped {
                        entry.order = entry.order.max(groups.cut.map_or(0, |c| c.order));
                        entry.labels += groups.labels.len();
                    }
                    let unit = crate::step::count::gathered(n, 1, still, tones) / crate::decide::COMPLEX;
                    let plain_work = plain_stops + plain_nodes as f64 * unit;
                    let engine_work = worked + stops + if grouped { 0.0 } else { nodes as f64 * unit };
                    entry.work[0] += plain_work;
                    entry.work[1] += engine_work;
                    entry.work[2] += chosen;
                    for (t, x) in entry.time.iter_mut().zip(times) {
                        *t += x;
                    }
                    entry.bandwidth[0] += picture.quotient;
                    entry.bandwidth[1] += groups.cut.map_or(picture.quotient, |c| c.bandwidth);
                    if std::env::var("GSTEP_VERBOSE").is_ok() {
                        let (lambda, held, _) = picture.spectrum();
                        let spectrum: Vec<String> = (0..n).filter(|&a| held[a]).map(|a| format!("{:.1}{:+.2}i", lambda[a].re, lambda[a].im)).collect();
                        println!("GSTEP-PIECE {name} {label} step {k} kind {kind} from {from:.4}: still {still} tones {tones} nu_Q {:.1} plain nodes {plain_nodes} engine nodes {nodes} groups {} centres {:?} nu_env {:.1} | held {}", picture.quotient, groups.groups, groups.cut.map(|c| c.centres[..c.groups].to_vec()), groups.cut.map_or(f64::NAN, |c| c.bandwidth), spectrum.join(" "));
                    }
                }
                match taken {
                    Ok(step) => t = step.end,
                    Err(_) => break,
                }
            }
            for (kind, e) in tally.iter().enumerate() {
                if e.pieces == 0 {
                    continue;
                }
                let p = e.pieces as f64;
                let show = |w: &[f64; 4]| format!("S {:.1e} H {:.1e} K {:.1e} H(u*) {:.1e}", w[0], w[1], w[2], w[3]);
                println!(
                    "GSTEP-FILON {name} {label} tau {tau} {}: {} pieces ({} by groups, {} not held) | engine {} | plain {} | brute's check {:.1e} | nodes plain {:.1} engine {:.1} | groups {:.2} order {} labels {:.1} | nu_Q {:.0} nu_env {:.0} | counted (complex MA) plain {:.0} engine {:.0} its choice {:.0} | us plain {:.1} the choice {:.1} the engine's sums {:.1}",
                    ["plain", "after a jump", "start"][kind],
                    e.pieces,
                    e.grouped,
                    e.failed,
                    show(&e.engine),
                    show(&e.plain),
                    e.check,
                    e.nodes[0] as f64 / p,
                    e.nodes[1] as f64 / p,
                    e.groups as f64 / p,
                    e.order,
                    e.labels as f64 / e.grouped.max(1) as f64,
                    e.bandwidth[0] / p,
                    e.bandwidth[1] / p,
                    e.work[0] / p,
                    e.work[1] / p,
                    e.work[2] / p,
                    e.time[0] / p,
                    e.time[1] / p,
                    e.time[2] / p,
                );
            }
        }
    }
}

/// The trajectory after `count` steps of τ on a model (none where one is refused).
fn walked<D: Dynamics>(model: &D, t: &Trajectory, tau: f64, count: usize) -> Option<Trajectory> {
    let mut scratch = Scratch::default();
    let mut t = t.clone();
    for _ in 0..count {
        t = stepped(model, &t, tau, &mut scratch).ok()?.end;
    }
    Some(t)
}

/// Two trajectories apart: momentum (1/µm), position (µm), the normalized internal state's density (largest entry), ln of
/// the survival.
fn differ(a: &Trajectory, b: &Trajectory) -> [f64; 4] {
    let d3 = |x: [f64; 3], y: [f64; 3]| (0..3).map(|i| (x[i] - y[i]).powi(2)).sum::<f64>().sqrt();
    let (na, nb) = (norm_sqr(&a.psi), norm_sqr(&b.psi));
    let mut rho = 0.0f64;
    for i in 0..a.psi.len() {
        for j in 0..a.psi.len() {
            rho = rho.max((a.psi[i] * a.psi[j].conj() / na - b.psi[i] * b.psi[j].conj() / nb).abs());
        }
    }
    [d3(a.kappa, b.kappa), d3(a.r, b.r), rho, (na.ln() - nb.ln()).abs()]
}

#[test]
#[ignore]
fn gstep_one_node() {
    let (burn, count, repeats, fine) = (env("GSTEP_BURN", 5.0) as usize, env("GSTEP_STATES", 12.0) as usize, env("GSTEP_REPEAT", 5.0) as usize, env("GSTEP_FINE", 16.0) as usize);
    for (name, request) in scenes() {
        let tau = tau_of(&name);
        let chief = crate::solve::chief(&request);
        for (label, run) in models(&name, &request) {
            let three = Three::new(&run);
            let mut t = start(&run, &request, chief, 3);
            let mut scratch = Scratch::default();
            for _ in 0..burn {
                if let Ok(step) = stepped(&three, &t, tau, &mut scratch) {
                    t = step.end;
                }
            }
            // Per state: [three at τ, three at τ/2 ×2, one at τ, one at τ/2 ×2] against the reference, and one against
            // three at τ; each of the four measures.
            let mut rows: Vec<[[f64; 4]; 5]> = Vec::new();
            let (mut time, mut counted, mut parts, mut taken, mut nodes) = ([0.0f64; 2], [0.0f64; 2], [[0.0f64; 6]; 2], 0usize, [0usize; 2]);
            for _ in 0..count {
                let mut quiet = t.clone();
                quiet.threshold = f64::MIN_POSITIVE;
                let reference = walked(&three, &quiet, tau / fine as f64, fine);
                let (a, b) = (walked(&three, &quiet, tau, 1), walked(&three, &quiet, 0.5 * tau, 2));
                let (c, d) = (walked(&run, &quiet, tau, 1), walked(&run, &quiet, 0.5 * tau, 2));
                if let (Some(r), Some(a), Some(b), Some(c), Some(d)) = (&reference, &a, &b, &c, &d) {
                    rows.push([differ(a, r), differ(b, r), differ(c, r), differ(d, r), differ(c, a)]);
                }
                // A step's time on the trajectory's own state, either way, each its least; its count by part.
                let mut next = None;
                for way in 0..2 {
                    let mut best = f64::INFINITY;
                    for _ in 0..repeats {
                        let mut s = Scratch::default();
                        let _ = if way == 0 { stepped(&three, &t, tau, &mut s) } else { stepped(&run, &t, tau, &mut s) };
                        let clock = std::time::Instant::now();
                        let step = if way == 0 { stepped(&three, &t, tau, &mut s) } else { stepped(&run, &t, tau, &mut s) };
                        best = best.min(clock.elapsed().as_secs_f64() * 1e6);
                        if let Ok(step) = step {
                            if next.is_none() && way == 0 {
                                next = Some(step.end.clone());
                            }
                            if repeats > 0 {
                                counted[way] += step.work / repeats as f64;
                                for (_, price) in &step.prices {
                                    for (p, x) in parts[way].iter_mut().zip(price.parts(price.costs)) {
                                        *p += x / repeats as f64;
                                    }
                                    nodes[way] += price.nodes;
                                }
                            }
                        }
                    }
                    time[way] += best;
                }
                taken += 1;
                match next {
                    Some(end) => t = end,
                    None => break,
                }
            }
            let k = 2.0 * std::f64::consts::PI / 0.780;
            let stat = |i: usize, m: usize| {
                let mut v: Vec<f64> = rows.iter().map(|r| r[i][m]).collect();
                v.sort_by(f64::total_cmp);
                (v.get(v.len() / 2).copied().unwrap_or(f64::NAN), v.last().copied().unwrap_or(f64::NAN))
            };
            let names = ["three(tau)", "three(tau/2)x2", "one(tau)", "one(tau/2)x2", "one-three(tau)"];
            let mut line = String::new();
            for (i, w) in names.iter().enumerate() {
                let (km, kx) = stat(i, 0);
                let (pm, px) = stat(i, 2);
                let (lm, lx) = stat(i, 3);
                line += &format!(" | {w}: dk {:.2e}/{:.2e} hbar k, drho {:.2e}/{:.2e}, dlnN {:.2e}/{:.2e}", km / k, kx / k, pm, px, lm, lx);
            }
            println!("GSTEP-ONE {name} {label} tau {tau}: {} states (median/max against three nodes at tau/{fine}){line}", rows.len());
            let per = |x: f64| x / taken.max(1) as f64;
            let show = |p: &[f64; 6]| p.iter().map(|x| format!("{:.0}", per(*x))).collect::<Vec<_>>().join("/");
            println!(
                "GSTEP-ONE-TIME {name} {label} tau {tau}: three nodes {:.1} us a step, one node {:.1} us ({:.3}) | counted (real MA) three {:.0} one {:.0} ({:.3}); by part [propagation/kick/hazard/optical/end/model] three {} one {} | kick nodes a step three {:.1} one {:.1} | the sum of {taken} steps' least of {repeats}",
                per(time[0]),
                per(time[1]),
                time[1] / time[0],
                per(counted[0]),
                per(counted[1]),
                counted[1] / counted[0],
                show(&parts[0]),
                show(&parts[1]),
                nodes[0] as f64 / (taken * repeats).max(1) as f64,
                nodes[1] as f64 / (taken * repeats).max(1) as f64,
            );
        }
    }
}

#[test]
#[ignore]
fn gstep_ladder() {
    let pilot = crate::solve::pilot_size();
    for (name, request) in scenes() {
        for (label, run) in models(&name, &request) {
            let clock = std::time::Instant::now();
            let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                crate::solve::solve_on(&request, &run, &mut |m| {
                    let crate::solve::Message::Checkpoint(s, remaining) = m else { return };
                    println!("GSTEP-LADDER {name} {label} {:.1} s: {} trajectories, tau {} us, remaining {remaining:?}", clock.elapsed().as_secs_f64(), s.trajectories, s.time_step_us);
                    if s.trajectories > pilot {
                        panic!("GSTEP-LADDER the product's first batch");
                    }
                })
            }));
            let how = match ended {
                Ok(Ok(s)) => format!("solved: tau {} us, {} trajectories", s.time_step_us, s.trajectories),
                Ok(Err(e)) => format!("refused: {e}"),
                Err(_) => "the product began".to_string(),
            };
            println!("GSTEP-LADDER {name} {label} ended at {:.1} s: {how}", clock.elapsed().as_secs_f64());
        }
    }
}

/// A step's sums by groups are its integral to the round-off (step.rs, # The quadrature by groups): eight states of a
/// tweezer (`Synthetic`: light shifts and couplings of 1 to 30 rad/µs, decays of 0.01 to 5 /µs), the first three raised
/// by 1500 rad/µs -- two manifolds, as a ground state's two hyperfine levels in a frame --, still along the path, from
/// the first state over a step of 0.5 µs, where the plain rule takes hundreds of nodes: the cut is two groups, its
/// nodes a quarter of the plain rule's or fewer (measured 192 against 873: the manifolds' own spread sets them), its sum
/// within 10⁻¹³ of the width of brute force's (1.6·10⁻¹⁵), as the plain rule's (1.4·10⁻¹⁵).
#[test]
fn a_piece_s_sums_by_groups_are_its_integral_to_the_round_off() {
    use super::Synthetic;
    let mut model = Synthetic::new(8, 1000.0, [0.0; 3], &[]);
    for a in 0..3 {
        model.still[(a, a)] += C64::real(1500.0);
    }
    let (tau, from) = (0.5, Trajectory::new(&model, 0, [0.05, 0.0, 0.0], [0.0, 0.0, 3.0], 7, 0));
    let h = 0.5 * tau;
    let midway = Midway::new(&model, &from, tau).unwrap();
    let n = from.psi.len();
    let kappa_m = predicted(&from.forms, midway.reached.1, tau, from.time + h);
    let along = Along::new(&midway.generator, kappa_m, kappa_m.map(|k| midway.hbar_over_mass * k));
    let x = along.turned(&from.psi, -h, true)[..n].to_vec();
    let mut s = Scratch::default();
    s.fit(n, SUMS);
    let mut still = along.middle.still();
    pictured(&along, &mut s, tau, Some(&x), true, &mut still).unwrap();
    assert!(still, "a generator with no beat is still");
    let Scratch { picture, groups, .. } = &mut s;
    picture.launch(-h, &x);
    groups.choose(picture, -h, h, (true, 0), &mut |_: &mut crate::spectral::Picture| false);
    let (grouped, nodes, held, _, _) = by_rule(picture, groups, true, h, -h, h, groups.groups > 1);
    let (plain, plain_nodes, _, _, _) = by_rule(picture, groups, true, h, -h, h, false);
    let exact = brute(picture, true, h, -h, h, 1);
    assert!(held, "every node's series holds");
    let apart = |a: &[f64], b: &[f64]| {
        let (mut x, mut y) = (vec![C64::ZERO; n * n], vec![C64::ZERO; n * n]);
        whole(a, 0, n, &mut x);
        whole(b, 0, n, &mut y);
        x.iter().zip(&y).map(|(p, q)| (*p - *q).abs()).fold(0.0, crate::linalg::larger)
    };
    eprintln!("groups {}, nodes {nodes} against the plain rule's {plain_nodes}; apart from brute force over the width: groups {:.1e}, plain {:.1e}", groups.groups, apart(&grouped, &exact) / tau, apart(&plain, &exact) / tau);
    assert_eq!(groups.groups, 2, "the two manifolds' groups");
    assert!(4 * nodes <= plain_nodes && plain_nodes > 100, "nodes {nodes} against {plain_nodes}");
    assert!(apart(&grouped, &exact) < 1e-13 * tau && apart(&plain, &exact) < 1e-13 * tau);
}

/// A cluster of ground states alone whose model gives no candidate tone takes its generator at the middle alone
/// (step.rs, # One node a step): five ground states in a tweezer beating along the path by nine pairs of plane waves'
/// parts (`Synthetic`), their candidates hidden (`Counted::quiet`), as the selected model gives its pattern by its
/// points: one generator a step, where the same model with its candidates takes three; and its step of the second
/// order -- against three nodes' at τ/16, two steps of τ/2 err four times less than one of τ, as three nodes' do (τ =
/// 0.1 µs: one node's fall 3.9 to 6.2, three nodes' 3.8 to 7.5; its error two to ten times theirs).
#[test]
fn a_ground_cluster_beating_in_no_time_takes_one_node_of_the_second_order() {
    use super::{atoms, step_of, Counted, Synthetic, K};
    let six = [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0], [1.0, 1.0, 0.0], [1.0, -1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0], [0.0, 1.0, -1.0]].map(|q| q.map(|x| x * K));
    let model = Synthetic::new(5, 1000.0, [0.0; 3], &six[..]);
    let starts: Vec<Trajectory> = atoms(&model, 4).into_iter().map(|t| Trajectory { threshold: f64::MIN_POSITIVE, ..t }).collect();
    let tau = 0.1;
    for (quiet, nodes) in [(false, 3), (true, 1)] {
        let counted = Counted { model: &model, calls: std::cell::Cell::new([0; 5]), quiet };
        for s in &starts {
            step_of(&counted, s, tau, &mut Scratch::default()).unwrap();
        }
        assert_eq!(counted.calls.get()[2], nodes * starts.len() as u64, "the generator's calls, quiet {quiet}");
    }
    let (three, one) = (Counted { model: &model, calls: std::cell::Cell::new([0; 5]), quiet: false }, Counted { model: &model, calls: std::cell::Cell::new([0; 5]), quiet: true });
    let walk = |model: &dyn Fn(&Trajectory, f64) -> Trajectory, t: &Trajectory, tau: f64, count: usize| (0..count).fold(t.clone(), |t, _| model(&t, tau));
    let three_step = |t: &Trajectory, tau: f64| step_of(&three, t, tau, &mut Scratch::default()).unwrap().end;
    let one_step = |t: &Trajectory, tau: f64| step_of(&one, t, tau, &mut Scratch::default()).unwrap().end;
    let off = |a: &Trajectory, b: &Trajectory| (0..3).map(|i| (a.kappa[i] - b.kappa[i]).powi(2)).sum::<f64>().sqrt();
    for t in &starts {
        let reference = walk(&three_step, t, tau / 16.0, 16);
        let errors = [walk(&one_step, t, tau, 1), walk(&one_step, t, 0.5 * tau, 2), walk(&three_step, t, tau, 1), walk(&three_step, t, 0.5 * tau, 2)].map(|e| off(&e, &reference));
        eprintln!("momentum off the reference: one node {:.2e} / {:.2e} ({:.1}), three {:.2e} / {:.2e} ({:.1})", errors[0], errors[1], errors[0] / errors[1], errors[2], errors[3], errors[2] / errors[3]);
        assert!(errors[0] / errors[1] > 3.0, "one node's step of the second order: {errors:?}");
    }
}

/// The whole solve on the model (solve.rs `solve_on`), every output said: the ground states' populations, the
/// populations' estimated error, T and its error, the loss; and the step, the trajectories and the wall time.
#[test]
#[ignore]
fn gstep_solve() {
    for (name, request) in scenes() {
        for (label, run) in models(&name, &request) {
            let clock = std::time::Instant::now();
            match crate::solve::solve_on(&request, &run, &mut |_| {}) {
                Ok(s) => {
                    for (k, p) in s.populations.iter().enumerate() {
                        let ground: Vec<String> = p[..GROUND].iter().map(|x| format!("{x:.6e}")).collect();
                        println!("GSTEP-SOLVE-AT {name} {label} {k} pops {} err {:.3e} T {:.4} {:?} loss {:.3e}", ground.join(","), s.estimated_errors.get(k).copied().unwrap_or(f64::NAN), s.temperature_total_uk[k], s.temperature_total_errors_uk[k], s.loss_probability[k]);
                    }
                    println!("GSTEP-SOLVE {name} {label}: tau {} us, {} trajectories, {:.1} s", s.time_step_us, s.trajectories, clock.elapsed().as_secs_f64());
                }
                Err(e) => println!("GSTEP-SOLVE {name} {label}: refused {e} after {:.1} s", clock.elapsed().as_secs_f64()),
            }
        }
    }
}
