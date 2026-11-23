//! OLDNEW item C (BRIEF_OLDNEW; tests alone): the node's contraction off tables of pair products (rs/fold_pairs.rs)
//! against the node's own (`Rs::evaluate` + `Out::contract`), element by element, and the two nodes' costs.
//! - `oldnew_pairs`: per scene of OLDNEW_PAIRS (comma separated `name=model`, model "ground" or "+label+label"; default
//!   PGC3D, TET, tilt0, tilt2 on their ground states and R4 with D2-2-1) the tables built (OLDNEW_ORDERS Doppler terms,
//!   4; Chebyshev degree OLDNEW_DEGREE, 12) with their time and memory; at OLDNEW_POINTS (24) points -- r in the trap's
//!   centre's cloud, κ at one and three times the thermal speed in turning directions -- the contraction both ways with
//!   the photon forms (`Making::node(true)`), every H and channel element's value and gradient compared, the difference
//!   over the element's own rounding scale (`Out::scales`) and over the output's largest value, beside the Doppler tail's
//!   bound at the orders taken; then the filled nodes (`Selected::fill`) compared; and each way's time (the least of
//!   OLDNEW_REPEAT, 7), the tables at 1 to OLDNEW_ORDERS terms. Ignored; run by name pinned to one core:
//!   `<test exe> model::selected::tests::oldnew::oldnew_pairs --exact --include-ignored --nocapture --test-threads 1`.
use super::*;
use crate::model::rs::fold::pairs::Pairs;
use crate::model::rs::fold::Values;
use crate::step::Dynamics;

fn request_of(name: &str) -> Request {
    let files = [("PGC3D", "PGC3D_1ms.json"), ("TET", "TET_1ms.json"), ("R4", "R4_LIN_4.json"), ("GMD1C", "GMD1C_1ms.json"), ("tilt0", "tilt0.json"), ("tilt2", "tilt2.json")];
    let file = files.iter().find(|(n, _)| *n == name).unwrap_or_else(|| panic!("no request {name}")).1;
    let dir = std::path::PathBuf::from(std::env::var("V3C_REQ").expect("V3C_REQ: the requests' folder"));
    let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
    line["options"]["device"] = serde_json::json!("cpu");
    for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
        form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
    }
    Request::from_native(&line).unwrap()
}

/// The largest |a − b| over b's scale and over the output's largest |b|, per kind: [H value, H gradient, channel value,
/// channel gradient].
fn compare(a: &Out, b: &Out) -> [[f64; 2]; 4] {
    if std::env::var("OLDNEW_WORST").is_ok() {
        for (o, (x, y)) in a.channels.iter().zip(&b.channels).enumerate() {
            let top = y.value.iter().map(|z| z.abs()).fold(0.0, f64::max);
            let d = x.value.iter().zip(&y.value).map(|(p, q)| (*p - *q).abs()).fold(0.0, f64::max);
            if d > 1e-3 * top {
                let tt = x.value.iter().map(|z| z.abs()).fold(0.0, f64::max);
                eprintln!("OLDNEW-WORST channel output {o}: largest |exact| {top:.3e}, |table| {tt:.3e}, |difference| {d:.3e}");
            }
        }
    }
    let mut worst = [[0.0f64; 2]; 4];
    for (kind, (xs, ys)) in [(0usize, (&a.h, &b.h)), (2, (&a.channels, &b.channels))] {
        // Over the largest of every output of the kind (an output of 10⁻²⁴ is no scale of its own).
        let top = ys.iter().flat_map(|y| y.value.iter().map(|z| z.abs())).fold(0.0, f64::max);
        let gtop = ys.iter().flat_map(|y| y.gradient.iter().flat_map(|g| g.iter().map(|z| z.abs()))).fold(0.0, f64::max);
        for (x, y) in xs.iter().zip(ys.iter()) {
            for k in 0..y.value.len() {
                let d = (x.value[k] - y.value[k]).abs();
                let scale = y.scale.get(k).copied().unwrap_or(0.0).max(1e-300);
                worst[kind][0] = worst[kind][0].max(d / scale);
                if top > 0.0 {
                    worst[kind][1] = worst[kind][1].max(d / top);
                }
            }
            for a3 in 0..3 {
                for k in 0..y.value.len() {
                    let d = (x.gradient[a3][k] - y.gradient[a3][k]).abs();
                    let scale = y.gradient_scale[a3].get(k).copied().unwrap_or(0.0).max(1e-300);
                    worst[kind + 1][0] = worst[kind + 1][0].max(d / scale);
                    if gtop > 0.0 {
                        worst[kind + 1][1] = worst[kind + 1][1].max(d / gtop);
                    }
                }
            }
        }
    }
    worst
}

/// Two filled nodes' largest difference over the larger's largest entry: generator, gradient, photon classes' elements.
fn node_difference(a: &Node, b: &Node) -> [f64; 2] {
    let (mut d, mut top) = ([0.0f64; 2], [0.0f64; 2]);
    for ((_, x), (_, y)) in a.generator.iter().zip(&b.generator) {
        for (p, q) in x.data.iter().zip(&y.data) {
            d[0] = d[0].max((*p - *q).abs());
            top[0] = top[0].max(q.abs());
        }
    }
    for ((_, x), (_, y)) in a.gradient.iter().zip(&b.gradient) {
        for (gx, gy) in x.iter().zip(y) {
            for (p, q) in gx.data.iter().zip(&gy.data) {
                d[1] = d[1].max((*p - *q).abs());
                top[1] = top[1].max(q.abs());
            }
        }
    }
    [d[0] / top[0].max(1e-300), d[1] / top[1].max(1e-300)]
}

#[test]
#[ignore]
fn oldnew_pairs() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (orders, degree, count, repeat) = (env("OLDNEW_ORDERS", 4), env("OLDNEW_DEGREE", 12), env("OLDNEW_POINTS", 24), env("OLDNEW_REPEAT", 7));
    let list = std::env::var("OLDNEW_PAIRS").unwrap_or_else(|_| "PGC3D=ground,TET=ground,tilt0=ground,tilt2=ground,R4=+D2-2-1".to_string());
    let least = |f: &mut dyn FnMut()| -> f64 {
        f();
        (0..repeat)
            .map(|_| {
                let clock = std::time::Instant::now();
                f();
                clock.elapsed().as_secs_f64() * 1e6
            })
            .fold(f64::INFINITY, f64::min)
    };
    for entry in list.split(',') {
        let (name, spec) = entry.split_once('=').expect("name=model");
        let request = request_of(name);
        let mut keep: Vec<usize> = (0..crate::request::GROUND).collect();
        if let Some(labels) = spec.strip_prefix('+') {
            keep.extend(labels.split('+').map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
        }
        keep.sort_unstable();
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap();
        let plan = run.rs.plan();
        // The values at rest (the members' frequencies at rest).
        let mut work = Work::default();
        run.scene.point_into([0.0; 3], [0.0; 3], &mut work.point);
        let mut rest = Values::default();
        run.rs.tweezer().values_into(&run.scene, &work.point, &mut rest);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let mut reach = Values::default();
        run.scene.point_into([0.0; 3], [3.0 * speed; 3], &mut work.point);
        run.rs.tweezer().values_into(&run.scene, &work.point, &mut reach);
        let low: f64 = std::env::var("OLDNEW_LOW").ok().map_or(0.0, |x| x.parse().unwrap());
        let mut pairs = Pairs::new(plan, &run.lab, &rest, &reach, orders, degree, low);
        let (n_pairs, tables, elements, atoms) = pairs.census();
        let (zr, wr) = pairs.ranks();
        eprintln!(
            "OLDNEW-PAIRS {name} {spec} (P{}): built in {:.3} s, {:.2} MB, {} pieces of s on [{low}, 1] at degree {degree}, the spans' dropped bound {:.1e}; {n_pairs} pairs, {tables} tables ({orders} Doppler terms), {elements} table elements, {atoms} atoms; untabled blocks {:?}; amplitude sums (atoms, rank) {:?}, (atoms, rank, rank of the first coefficient's) {:?}; leg sums {:?}",
            keep.len(),
            pairs.build_seconds,
            pairs.bytes as f64 / 1e6,
            pairs.pieces.len(),
            pairs.dropped,
            pairs.untabled,
            zr,
            pairs.kinds,
            wr
        );
        if std::env::var("OLDNEW_ATOMS").is_ok() {
            for z in 0..zr.len() {
                let (block, cols, atoms) = pairs.atoms_of(plan, &run.lab, z);
                eprintln!("OLDNEW-ATOMS {name} z {z}: block {block} cols {cols:?}");
                for (piece, f, c, size) in atoms {
                    eprintln!("    piece {piece} key {:?} factor {f} = {:?} coefficient {c} (value {:.3e}) |α| {size:.3e}", plan.keys()[piece], rest.factors[f].0, rest.coefficients[c]);
                }
            }
        }
        let speed = if std::env::var("OLDNEW_REST").is_ok() { 0.0 } else { speed };
        let (kinds, cross) = pairs.breakdown(plan, &run.lab);
        eprintln!("OLDNEW-KINDS {name}: (pairs, table elements per order) H on Q {:?}, H on sidebands {:?}, channels on Q {:?}, on sidebands {:?}, photon forms on Q {:?}, on sidebands {:?}; H across K's fast parts {:?}", kinds[0], kinds[1], kinds[2], kinds[3], kinds[4], kinds[5], cross);
        pairs.split = std::env::var("OLDNEW_SPLIT").is_ok();
        pairs.noside = std::env::var("OLDNEW_NOSIDE").is_ok();
        pairs.spanned = std::env::var("OLDNEW_SPANNED").is_ok();
        let (entries, datoms, dspanned) = pairs.direct_census(plan);
        eprintln!("OLDNEW-DIRECT {name}: {entries} direct entries one by one; spanned {datoms} atoms onto {dspanned} basis places");
        // The points: a walk of the whole model where OLDNEW_WALK gives its steps (at OLDNEW_TAU), its states' (r, κ) --
        // the atoms' own intensities --, else the cloud below.
        let walk: Vec<(V3, V3)> = match std::env::var("OLDNEW_WALK").ok().and_then(|x| x.parse::<usize>().ok()) {
            Some(steps) => {
                let tau: f64 = std::env::var("OLDNEW_TAU").expect("OLDNEW_TAU").parse().unwrap();
                let chief = crate::solve::chief(&request);
                let v0 = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
                let mut t = Trajectory::new(&run, chief, [0.1, 0.2, 0.3], [-v0, 0.7 * v0, -1.2 * v0], 7, 203);
                let mut out = Vec::new();
                for _ in 0..steps {
                    let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Anywhere)) else { break };
                    t = step.end;
                    out.push((t.r, t.kappa));
                }
                out
            }
            None => Vec::new(),
        };
        let count = if walk.is_empty() { count } else { walk.len() };
        // The step's middle node is plain (no photon forms); OLDNEW_KICKS a jump's with them.
        let kicked = std::env::var("OLDNEW_KICKS").is_ok();
        let making = Making::node(kicked);
        let mut table_out = Out::default();
        let (mut exact_t, mut table_t) = (0.0, vec![0.0; orders]);
        let mut fill_t = 0.0;
        let mut parts_t = [0.0f64; 3];
        let (mut inside, mut outside) = (0usize, 0usize);
        let mut worst = vec![[[0.0f64; 2]; 4]; orders];
        let mut node_worst = vec![[0.0f64; 2]; orders];
        let (mut rho_max, mut bound_max) = (0.0f64, vec![0.0f64; orders]);
        let mut counted = (0.0, vec![0.0; orders]);
        for i in 0..count {
            // A point: r in the centre's cloud, κ at one or three thermal speeds along a turning direction.
            let u = i as f64;
            let (th, ph) = (0.9 * u + 0.3, 1.7 * u);
            let v = if i % 2 == 0 { 1.0 } else { 3.0 } * speed;
            let (r, kappa) = if walk.is_empty() {
                ([0.3 * (1.3 * u).sin(), 0.3 * (2.1 * u + 0.4).cos(), 1.0 * (0.7 * u + 1.1).sin()], [v * th.sin() * ph.cos(), v * th.sin() * ph.sin(), v * th.cos()])
            } else {
                walk[i]
            };
            run.scene.point_into(r, kappa, &mut work.point);
            run.rs.evaluate(&run.scene, &work.point, &mut work.local, making);
            work.out.contract(plan, &run.lab, &work.local, None, making);
            work.out.scales(plan, &run.lab, &work.local, None);
            let mut exact_node = run.template.clone();
            run.fill(&mut exact_node, &work.out, (kicked, true), &mut work.jumps);
            let values = work.local.values.clone();
            if !pairs.covers(plan, &values) {
                outside += 1;
                continue;
            }
            inside += 1;
            let (rho, _) = pairs.rho(plan, &run.lab, &values);
            rho_max = rho_max.max(rho);
            for o in 1..=orders {
                pairs.contract(plan, &run.lab, &values, &mut table_out, making, o);
                let w = compare(&table_out, &work.out);
                for (a, b) in worst[o - 1].iter_mut().zip(w) {
                    a[0] = a[0].max(b[0]);
                    a[1] = a[1].max(b[1]);
                }
                bound_max[o - 1] = bound_max[o - 1].max(rho.powi(o as i32) / (1.0 - rho));
                let mut table_node = run.template.clone();
                run.fill(&mut table_node, &table_out, (kicked, true), &mut work.jumps);
                let nd = node_difference(&table_node, &exact_node);
                node_worst[o - 1] = [node_worst[o - 1][0].max(nd[0]), node_worst[o - 1][1].max(nd[1])];
                counted.1[o - 1] += pairs.work;
            }
            // Times: the node's evaluate + contract, and the tables' values + contract, at each order count.
            exact_t += least(&mut || {
                run.scene.point_into(r, kappa, &mut work.point);
                run.rs.evaluate(&run.scene, &work.point, &mut work.local, making);
                work.out.contract(plan, &run.lab, &work.local, None, making);
            });
            counted.0 += work.local.work + work.out.work;
            pairs.parts = [0.0; 3];
            for o in 1..=orders {
                let mut vals = Values::default();
                table_t[o - 1] += least(&mut || {
                    run.scene.point_into(r, kappa, &mut work.point);
                    run.rs.tweezer().values_into(&run.scene, &work.point, &mut vals);
                    pairs.contract(plan, &run.lab, &vals, &mut table_out, making, o);
                });
            }
            parts_t = [parts_t[0] + pairs.parts[0], parts_t[1] + pairs.parts[1], parts_t[2] + pairs.parts[2]];
            let mut node = run.template.clone();
            fill_t += least(&mut || run.fill(&mut node, &work.out, (kicked, true), &mut work.jumps));
        }
        let k = inside.max(1) as f64;
        eprintln!("OLDNEW-COVER {name}: {inside} points within the tables' s, {outside} below them (their node the fold's own)");
        let calls = (inside * orders * (repeat + 1)).max(1) as f64;
        eprintln!("OLDNEW-PARTS {name}: per table contraction (mean over every order and call): coordinates {:.1} µs, pairs {:.1} µs, direct entries {:.1} µs", parts_t[0] / calls, parts_t[1] / calls, parts_t[2] / calls);
        eprintln!(
            "OLDNEW-NODE {name}: the node's point + fold + contraction {:.1} µs ({:.3e} CMA counted), fill {:.1} µs; ρ max {rho_max:.3e}",
            exact_t / k,
            counted.0 / k,
            fill_t / k
        );
        for o in 0..orders {
            let w = worst[o];
            eprintln!(
                "OLDNEW-TABLE {name} {} terms: point + values + tables {:.1} µs ({:.3e} CMA counted); tail bound {:.2e}; worst over the element's scale / over the kind's largest: H {:.2e} / {:.2e}, ∇H {:.2e} / {:.2e}, channels {:.2e} / {:.2e}, ∇channels {:.2e} / {:.2e}; filled node: generator {:.2e}, gradient {:.2e} of the largest",
                o + 1,
                table_t[o] / k,
                counted.1[o] / k,
                bound_max[o],
                w[0][0],
                w[0][1],
                w[1][0],
                w[1][1],
                w[2][0],
                w[2][1],
                w[3][0],
                w[3][1],
                node_worst[o][0],
                node_worst[o][1]
            );
        }
    }
}

/// K split into its fast parts (a check of item D: what the step costs and changes when the static F = 1 ↔ F = 2
/// elements are left out): the run's blocks set to K's fast parts (the states of one fast part one block) as `judge`
/// sets judged blocks, every H place across two blocks out of the layout and the decay's products recut.
fn split_by_parts(run: &mut Selected) -> Vec<Vec<usize>> {
    let n = run.rs.keep.len();
    let parts = run.rs.plan().parts().to_vec();
    let mut blocks: Vec<Vec<usize>> = Vec::new();
    let mut seen: Vec<[i64; 2]> = Vec::new();
    for a in 0..n {
        match seen.iter().position(|p| *p == parts[a]) {
            Some(i) => blocks[i].push(a),
            None => {
                seen.push(parts[a]);
                blocks.push(vec![a]);
            }
        }
    }
    run.blocks.clone_from(&blocks);
    for (b, states) in blocks.iter().enumerate() {
        for &s in states {
            run.block_of[s] = b;
        }
    }
    let block_of = run.block_of.clone();
    for list in run.tables.h.iter_mut() {
        for e in list.iter_mut().filter(|e| block_of[e.0 as usize / n] != block_of[e.0 as usize % n]) {
            *e = (e.0, u32::MAX, u32::MAX);
        }
    }
    run.decay_terms();
    let id = run.id;
    POOL.with(|p| p.borrow_mut().retain(|x| x.0 != id));
    blocks
}

/// OLDNEW item D: per scene of OLDNEW_SPLIT_SCENES (`name=model@tau`, default PGC3D, TET and tilt2 on their ground
/// states) a walk of OLDNEW_STEPS (120) steps of the whole model; per state, the step (the least of OLDNEW_REPEAT, 5) on
/// the whole model and on the model split by K's fast parts, the same start; the split's change of a jump-free step's
/// end (κ, ψ over its norm) beside the whole model's own step error (one step against two halves); the cross elements'
/// largest |H_ab| over |H_aa − H_bb| at the walk's middles; the jumps and the counted work both ways.
#[test]
#[ignore]
fn oldnew_split() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (steps, repeat) = (env("OLDNEW_STEPS", 120), env("OLDNEW_REPEAT", 5));
    let list = std::env::var("OLDNEW_SPLIT_SCENES").unwrap_or_else(|_| "PGC3D=ground@0.4741346189956611,TET=ground@0.4741346189956611,tilt2=ground@0.326".to_string());
    let least = |f: &mut dyn FnMut()| -> f64 {
        f();
        (0..repeat)
            .map(|_| {
                let clock = std::time::Instant::now();
                f();
                clock.elapsed().as_secs_f64() * 1e6
            })
            .fold(f64::INFINITY, f64::min)
    };
    for entry in list.split(',') {
        let (name, rest) = entry.split_once('=').unwrap();
        let (_, tau) = rest.split_once('@').unwrap();
        let tau: f64 = tau.parse().unwrap();
        let request = request_of(name);
        let keep: Vec<usize> = (0..crate::request::GROUND).collect();
        let make = || {
            let scene = crate::scene::Scene::new(&request).unwrap();
            let selection = crate::select::choose(&scene).unwrap();
            let reference = crate::solve::alone(crate::solve::chief(&request));
            Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap()
        };
        let whole = make();
        let mut split = make();
        let before = whole.blocks().to_vec();
        let blocks = split_by_parts(&mut split);
        let chief = crate::solve::chief(&request);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / whole.hbar_over_mass()).sqrt();
        let mut t = Trajectory::new(&whole, chief, [0.1, 0.2, 0.3], [-speed, 0.7 * speed, -1.2 * speed], 7, 203);
        let attempt = |run: &Selected, t: &Trajectory, tau: f64| Midway::new(run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], &mut Scratch::default(), &Anywhere));
        let (mut tw, mut ts, mut taken, mut jw, mut js) = (0.0, 0.0, 0usize, 0usize, 0usize);
        let (mut dk, mut dpsi, mut ek, mut epsi) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        let mut cross: f64 = 0.0;
        let (mut work_w, mut work_s) = (0.0, 0.0);
        let norm_psi = |psi: &[C64]| norm_sqr(psi).sqrt();
        for _ in 0..steps {
            let Ok(a) = attempt(&whole, &t, tau) else { break };
            let Ok(b) = attempt(&split, &t, tau) else { break };
            jw += a.made.iter().filter(|m| m.kick.is_some()).count();
            js += b.made.iter().filter(|m| m.kick.is_some()).count();
            work_w += a.work;
            work_s += b.work;
            tw += least(&mut || {
                std::hint::black_box(attempt(&whole, &t, tau).is_ok());
            });
            ts += least(&mut || {
                std::hint::black_box(attempt(&split, &t, tau).is_ok());
            });
            // Where neither jumped: the split's change of the end against the whole model's step error (one step
            // against its two halves).
            if a.made.iter().all(|m| m.kick.is_none()) && b.made.iter().all(|m| m.kick.is_none()) {
                let k = |x: &Trajectory, y: &Trajectory| (0..3).map(|c| (x.kappa[c] - y.kappa[c]).powi(2)).sum::<f64>().sqrt();
                let p = |x: &Trajectory, y: &Trajectory| {
                    let (nx, ny) = (norm_psi(&x.psi), norm_psi(&y.psi));
                    x.psi.iter().zip(y.psi.iter()).map(|(u, v)| (*u * (1.0 / nx) - *v * (1.0 / ny)).norm_sqr()).sum::<f64>().sqrt()
                };
                dk = dk.max(k(&a.end, &b.end));
                dpsi = dpsi.max(p(&a.end, &b.end));
                if let Ok(h1) = attempt(&whole, &t, 0.5 * tau) {
                    if let Ok(h2) = attempt(&whole, &h1.end, 0.5 * tau) {
                        if h1.made.iter().chain(&h2.made).all(|m| m.kick.is_none()) {
                            ek = ek.max(k(&a.end, &h2.end));
                            epsi = epsi.max(p(&a.end, &h2.end));
                        }
                    }
                }
            }
            // The cross elements at the middle's node, against their diagonal gap.
            let node = whole.node(t.r, t.kappa, false);
            let n = whole.rs.keep.len();
            let block_of = &split.block_of;
            for (beat, g) in &node.generator {
                if *beat != 0 {
                    continue;
                }
                for a1 in 0..n {
                    for b1 in 0..n {
                        if block_of[a1] != block_of[b1] {
                            let gap = (g.data[a1 * n + a1] - g.data[b1 * n + b1]).abs();
                            cross = cross.max(g.data[a1 * n + b1].abs() / gap);
                        }
                    }
                }
            }
            taken += 1;
            t = a.end;
        }
        let k = taken.max(1) as f64;
        eprintln!(
            "OLDNEW-SPLIT {name} tau {tau}: blocks {:?} -> {:?}; a step whole {:.1} µs, split {:.1} µs ({:.2}); jumps {jw} / {js}; counted {:.3e} / {:.3e} real MA a step; the split's change of a jump-free step's end: κ {dk:.2e}, ψ {dpsi:.2e} against the whole step's own error κ {ek:.2e}, ψ {epsi:.2e}; cross |H_ab|/|H_aa − H_bb| up to {cross:.2e}; over {taken} steps",
            before.iter().map(Vec::len).collect::<Vec<_>>(),
            blocks.iter().map(Vec::len).collect::<Vec<_>>(),
            tw / k,
            ts / k,
            ts / tw,
            work_w / k,
            work_s / k
        );
    }
}
