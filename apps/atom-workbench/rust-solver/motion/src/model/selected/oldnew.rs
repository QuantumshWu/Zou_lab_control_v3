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
            fill_t += least(&mut || { run.fill(&mut node, &work.out, (kicked, true), &mut work.jumps); });
        }
        // The coordinator's follow-up (OLDNEW_QUAD): what a step reads of the channels off their quadratic forms'
        // tables (rs/fold_pairs.rs `Quad`), against the node's channels' own; and a jump's photon node off the tables.
        if std::env::var("OLDNEW_QUAD").is_ok() {
            let mut quad = pairs.quad(plan, &run.lab);
            eprintln!(
                "OLDNEW-QUAD {name}: built in {:.3} s, {:.2} MB, {} pairs of contributions, {} table elements (4 tables a pair) at degree {}; against the channel pairs' {} table elements a Doppler order (H's {})",
                quad.build_seconds,
                quad.bytes as f64 / 1e6,
                quad.pairs_len(),
                quad.elements,
                2 * degree,
                kinds[2].1 + if pairs.noside { 0 } else { kinds[3].1 },
                kinds[0].1 + if pairs.noside { 0 } else { kinds[1].1 }
            );
            let n = plan.n;
            let mut worst = vec![[0.0f64; 3]; orders];
            let (mut t_channels, mut t_quad, mut t_fill, mut t_h) = (vec![0.0; orders], vec![0.0; orders], 0.0, vec![0.0; orders]);
            let (mut t_photon_exact, mut t_photon_table) = (0.0, vec![0.0; orders]);
            let mut counted = vec![0.0; orders];
            let mut taken = 0usize;
            for i in 0..count {
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
                let values = work.local.values.clone();
                if !pairs.covers(plan, &values) {
                    continue;
                }
                taken += 1;
                let (ed, ea, eb) = crate::model::rs::fold::pairs::quadratic_of(&work.out, &run.lab, &quad.classes, n);
                let top = |x: &[Vec<C64>]| x.iter().flat_map(|m| m.iter().map(|z| z.abs())).fold(0.0, f64::max);
                let (td, ta, tb) = (top(&ed), top(&ea.iter().flatten().cloned().collect::<Vec<_>>()), top(&eb.iter().flatten().cloned().collect::<Vec<_>>()));
                for o in 1..=orders {
                    pairs.channels_off = true;
                    pairs.contract(plan, &run.lab, &values, &mut table_out, making, o);
                    quad.contract(&pairs, plan, &values, o);
                    pairs.channels_off = false;
                    let diff = |x: &[Vec<C64>], y: &[Vec<C64>]| x.iter().zip(y).flat_map(|(p, q)| p.iter().zip(q).map(|(a, b)| (*a - *b).abs())).fold(0.0, f64::max);
                    let fa: Vec<Vec<C64>> = quad.a.iter().flatten().cloned().collect();
                    let fb: Vec<Vec<C64>> = quad.b.iter().flatten().cloned().collect();
                    let w = [diff(&quad.d, &ed) / td.max(1e-300), diff(&fa, &ea.iter().flatten().cloned().collect::<Vec<_>>()) / ta.max(1e-300), diff(&fb, &eb.iter().flatten().cloned().collect::<Vec<_>>()) / tb.max(1e-300)];
                    for (x, y) in worst[o - 1].iter_mut().zip(w) {
                        *x = x.max(y);
                    }
                    counted[o - 1] += quad.work;
                    // Times: H alone, H + the channels (the node's route), H + the quadratic forms.
                    t_h[o - 1] += least(&mut || {
                        let mut vals = Values::default();
                        run.scene.point_into(r, kappa, &mut work.point);
                        run.rs.tweezer().values_into(&run.scene, &work.point, &mut vals);
                        pairs.channels_off = true;
                        pairs.contract(plan, &run.lab, &vals, &mut table_out, making, o);
                        pairs.channels_off = false;
                    });
                    t_quad[o - 1] += least(&mut || {
                        let mut vals = Values::default();
                        run.scene.point_into(r, kappa, &mut work.point);
                        run.rs.tweezer().values_into(&run.scene, &work.point, &mut vals);
                        pairs.channels_off = true;
                        pairs.contract(plan, &run.lab, &vals, &mut table_out, making, o);
                        pairs.channels_off = false;
                        quad.contract(&pairs, plan, &vals, o);
                    });
                    t_channels[o - 1] += least(&mut || {
                        let mut vals = Values::default();
                        run.scene.point_into(r, kappa, &mut work.point);
                        run.rs.tweezer().values_into(&run.scene, &work.point, &mut vals);
                        pairs.contract(plan, &run.lab, &vals, &mut table_out, making, o);
                    });
                    // A jump's photon node off the tables: the channels with their photon forms alone.
                    t_photon_table[o - 1] += least(&mut || {
                        let mut vals = Values::default();
                        run.scene.point_into(r, kappa, &mut work.point);
                        run.rs.tweezer().values_into(&run.scene, &work.point, &mut vals);
                        pairs.contract(plan, &run.lab, &vals, &mut table_out, Making::PHOTONS, o);
                    });
                }
                t_photon_exact += least(&mut || {
                    run.scene.point_into(r, kappa, &mut work.point);
                    run.rs.evaluate(&run.scene, &work.point, &mut work.local, Making::PHOTONS);
                    work.out.contract(plan, &run.lab, &work.local, None, Making::PHOTONS);
                });
                run.rs.evaluate(&run.scene, &work.point, &mut work.local, making);
                work.out.contract(plan, &run.lab, &work.local, None, making);
                let mut node = run.template.clone();
                t_fill += least(&mut || { run.fill(&mut node, &work.out, (kicked, true), &mut work.jumps); });
            }
            let k = taken.max(1) as f64;
            eprintln!("OLDNEW-QUADT {name}: fill (the decay from the channels, the layout) {:.1} µs; a jump's photon node: the fold's {:.1} µs", t_fill / k, t_photon_exact / k);
            for o in 0..orders {
                eprintln!(
                    "OLDNEW-QUADO {name} {} terms: H alone {:.1} µs; H + channels {:.1} µs; H + quadratic forms {:.1} µs ({:.3e} CMA counted); photon node off the tables {:.1} µs; quadratic forms against the node's channels' (over each kind's largest): D {:.2e}, A {:.2e}, B {:.2e}",
                    o + 1,
                    t_h[o] / k,
                    t_channels[o] / k,
                    t_quad[o] / k,
                    counted[o] / k,
                    t_photon_table[o] / k,
                    worst[o][0],
                    worst[o][1],
                    worst[o][2]
                );
            }
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

/// K split into its fast parts within the run's blocks (`Selected::split_by_parts`).
fn split_by_parts(run: &mut Selected) -> Vec<Vec<usize>> {
    run.split_by_parts()
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

/// OLDNEW's budget check (the coordinator's 11-23 follow-up): the step with the quadrature's rule, the quotient's
/// bandwidth and the cuts laid to a tolerance ε in place of the round-off (step.rs `QUAD_TOL`, tests alone). Per scene of
/// OLDNEW_TOL_SCENES (`name=model@tau`) and model (whole, and split by K's fast parts where OLDNEW_TOL_SPLIT), a walk of
/// OLDNEW_STEPS (80) steps at the round-off; per state and ε of OLDNEW_TOLS the step (the least of OLDNEW_REPEAT, 5) and its
/// end against the round-off's: |Δκ|, |Δψ| over its norm, |Δ ln‖ψ‖²| (the survival), beside the step's own error (one step
/// against its two halves, at the round-off).
#[test]
#[ignore]
fn oldnew_tol() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (steps, repeat) = (env("OLDNEW_STEPS", 80), env("OLDNEW_REPEAT", 5));
    let list = std::env::var("OLDNEW_TOL_SCENES").unwrap_or_else(|_| "PGC3D=ground@0.4741346189956611".to_string());
    let tols: Vec<f64> = std::env::var("OLDNEW_TOLS").unwrap_or_else(|_| "1e-12,1e-9,1e-6,1e-4".to_string()).split(',').map(|x| x.parse().unwrap()).collect();
    let set = |x: f64| crate::step::QUAD_TOL.store(x.to_bits(), std::sync::atomic::Ordering::Relaxed);
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
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let mut run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap();
        let split = std::env::var("OLDNEW_TOL_SPLIT").is_ok();
        if split {
            split_by_parts(&mut run);
        }
        let chief = crate::solve::chief(&request);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let attempt = |t: &Trajectory, tau: f64| Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], &mut Scratch::default(), &Anywhere));
        set(0.0);
        let mut states = Vec::new();
        let mut t = Trajectory::new(&run, chief, [0.1, 0.2, 0.3], [-speed, 0.7 * speed, -1.2 * speed], 7, 203);
        for _ in 0..steps {
            states.push(t.clone());
            let Ok(s) = attempt(&t, tau) else { break };
            t = s.end;
        }
        let norm_psi = |psi: &[C64]| norm_sqr(psi).sqrt();
        let dpsi = |x: &Trajectory, y: &Trajectory| {
            let (nx, ny) = (norm_psi(&x.psi), norm_psi(&y.psi));
            x.psi.iter().zip(y.psi.iter()).map(|(u, v)| (*u * (1.0 / nx) - *v * (1.0 / ny)).norm_sqr()).sum::<f64>().sqrt()
        };
        let dk = |x: &Trajectory, y: &Trajectory| (0..3).map(|c| (x.kappa[c] - y.kappa[c]).powi(2)).sum::<f64>().sqrt();
        let dn = |x: &Trajectory, y: &Trajectory| ((norm_sqr(&x.psi)).ln() - (norm_sqr(&y.psi)).ln()).abs();
        // The round-off's: its time, its ends, the step's own error.
        let (mut t0, mut ek, mut ep) = (0.0, 0.0f64, 0.0f64);
        let mut ends = Vec::new();
        for s in &states {
            let a = attempt(s, tau).unwrap();
            t0 += least(&mut || {
                std::hint::black_box(attempt(s, tau).is_ok());
            });
            if a.made.iter().all(|m| m.kick.is_none()) {
                if let Ok(h1) = attempt(s, 0.5 * tau) {
                    if let Ok(h2) = attempt(&h1.end, 0.5 * tau) {
                        if h1.made.iter().chain(&h2.made).all(|m| m.kick.is_none()) {
                            ek = ek.max(dk(&a.end, &h2.end));
                            ep = ep.max(dpsi(&a.end, &h2.end));
                        }
                    }
                }
            }
            ends.push(a);
        }
        let k = states.len().max(1) as f64;
        eprintln!("OLDNEW-TOL {name}{} tau {tau}: at the round-off a step {:.1} µs; the step's own error κ {ek:.2e}, ψ {ep:.2e}; over {} states", if split { " split" } else { "" }, t0 / k, states.len());
        for &eps in &tols {
            set(eps);
            let (mut te, mut mk, mut mp, mut mn, mut moved) = (0.0, 0.0f64, 0.0f64, 0.0f64, 0usize);
            for (s, a0) in states.iter().zip(&ends) {
                let Ok(a) = attempt(s, tau) else { continue };
                te += least(&mut || {
                    std::hint::black_box(attempt(s, tau).is_ok());
                });
                // The same jumps both ways: the ends compare.
                if a.made.iter().filter(|m| m.kick.is_some()).count() == a0.made.iter().filter(|m| m.kick.is_some()).count() {
                    mk = mk.max(dk(&a.end, &a0.end));
                    mp = mp.max(dpsi(&a.end, &a0.end));
                    mn = mn.max(dn(&a.end, &a0.end));
                } else {
                    moved += 1;
                }
            }
            eprintln!("OLDNEW-TOL {name}{} ε {eps:.0e}: a step {:.1} µs ({:.2} of the round-off's); against the round-off's end κ {mk:.2e}, ψ {mp:.2e}, ln N {mn:.2e}; {moved} steps whose jumps moved", if split { " split" } else { "" }, te / k, te / t0);
        }
        set(0.0);
    }
}

/// OLDNEW's scheme check (the coordinator's 11-23 follow-up): on the model split by K's fast parts (OLDNEW_SCHEME_WHOLE
/// for the whole one), per scene of OLDNEW_SCHEME_SCENES (`name=model@tau`) a walk of OLDNEW_STEPS (60) steps; per state
/// and scheme of OLDNEW_SCHEMES (`fixed:m` a fixed rule of m Gauss nodes, `three` the generator at three nodes; the
/// current scheme first) the step (the least of OLDNEW_REPEAT, 5), its end against the current scheme's, and its own error
/// -- one step of τ against two of τ/2, and one of τ/2 against two of τ/4, by the scheme itself --: what the step check
/// would read of it and its order.
#[test]
#[ignore]
fn oldnew_scheme() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (steps, repeat) = (env("OLDNEW_STEPS", 60), env("OLDNEW_REPEAT", 5));
    let list = std::env::var("OLDNEW_SCHEME_SCENES").unwrap_or_else(|_| "PGC3D=ground@0.4741346189956611".to_string());
    let schemes: Vec<String> = std::iter::once("current".to_string()).chain(std::env::var("OLDNEW_SCHEMES").unwrap_or_else(|_| "fixed:1,fixed:2,fixed:3,three".to_string()).split(',').map(String::from)).collect();
    let set = |scheme: &str| {
        let (m, three) = match scheme.split_once(':') {
            Some(("fixed", m)) => (m.parse().unwrap(), false),
            _ => (0usize, scheme == "three"),
        };
        crate::step::FIXED_NODES.store(m, std::sync::atomic::Ordering::Relaxed);
        crate::step::THREE_NODES.store(three, std::sync::atomic::Ordering::Relaxed);
    };
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
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let mut run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap();
        let whole = std::env::var("OLDNEW_SCHEME_WHOLE").is_ok();
        if !whole {
            run.split_by_parts();
        }
        let chief = crate::solve::chief(&request);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let attempt = |t: &Trajectory, tau: f64| Midway::new(&run, t, tau).and_then(|m| m.attempt(t, &Brownian::STILL, &[], &mut Scratch::default(), &Anywhere));
        set("current");
        let mut states = Vec::new();
        let mut t = Trajectory::new(&run, chief, [0.1, 0.2, 0.3], [-speed, 0.7 * speed, -1.2 * speed], 7, 203);
        for _ in 0..steps {
            states.push(t.clone());
            let Ok(s) = attempt(&t, tau) else { break };
            t = s.end;
        }
        let norm_psi = |psi: &[C64]| norm_sqr(psi).sqrt();
        let dpsi = |x: &Trajectory, y: &Trajectory| {
            let (nx, ny) = (norm_psi(&x.psi), norm_psi(&y.psi));
            x.psi.iter().zip(y.psi.iter()).map(|(u, v)| (*u * (1.0 / nx) - *v * (1.0 / ny)).norm_sqr()).sum::<f64>().sqrt()
        };
        let dk = |x: &Trajectory, y: &Trajectory| (0..3).map(|c| (x.kappa[c] - y.kappa[c]).powi(2)).sum::<f64>().sqrt();
        let dn = |x: &Trajectory, y: &Trajectory| ((norm_sqr(&x.psi)).ln() - (norm_sqr(&y.psi)).ln()).abs();
        let jumps = |s: &crate::step::Step| s.made.iter().filter(|m| m.kick.is_some()).count();
        let quiet = |s: &crate::step::Step| jumps(s) == 0;
        // One step of h against two of h/2 by the scheme set, where none jumps: (|Δκ|, |Δψ|).
        let doubling = |s: &Trajectory, h: f64| -> Option<(f64, f64)> {
            let a = attempt(s, h).ok()?;
            let h1 = attempt(s, 0.5 * h).ok()?;
            let h2 = attempt(&h1.end, 0.5 * h).ok()?;
            (quiet(&a) && quiet(&h1) && quiet(&h2)).then(|| (dk(&a.end, &h2.end), dpsi(&a.end, &h2.end)))
        };
        set("current");
        let base: Vec<Option<crate::step::Step>> = states.iter().map(|s| attempt(s, tau).ok()).collect();
        let mut t_current = 0.0;
        for (i, scheme) in schemes.iter().enumerate() {
            set(scheme);
            let (mut te, mut mk, mut mp, mut mn, mut moved) = (0.0, 0.0f64, 0.0f64, 0.0f64, 0usize);
            let (mut e1k, mut e1p, mut e2k, mut e2p, mut ratio_k, mut ratios) = (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0usize);
            let (mut with, mut without, mut jumped) = (0.0, 0.0, 0usize);
            for (s, b) in states.iter().zip(&base) {
                let Ok(a) = attempt(s, tau) else { continue };
                let x = least(&mut || {
                    std::hint::black_box(attempt(s, tau).is_ok());
                });
                te += x;
                if jumps(&a) > 0 {
                    let mut q = s.clone();
                    q.threshold = f64::MIN_POSITIVE;
                    without += least(&mut || {
                        std::hint::black_box(attempt(&q, tau).is_ok());
                    });
                    with += x;
                    jumped += jumps(&a);
                }
                if let Some(b) = b {
                    if jumps(&a) == jumps(b) {
                        mk = mk.max(dk(&a.end, &b.end));
                        mp = mp.max(dpsi(&a.end, &b.end));
                        mn = mn.max(dn(&a.end, &b.end));
                    } else {
                        moved += 1;
                    }
                }
                if let (Some(d1), Some(d2)) = (doubling(s, tau), doubling(s, 0.5 * tau)) {
                    e1k = e1k.max(d1.0);
                    e1p = e1p.max(d1.1);
                    e2k = e2k.max(d2.0);
                    e2p = e2p.max(d2.1);
                    if d2.0 > 0.0 {
                        ratio_k += (d1.0 / d2.0).log2();
                        ratios += 1;
                    }
                }
            }
            let k = states.len().max(1) as f64;
            if i == 0 {
                t_current = te;
            }
            eprintln!(
                "OLDNEW-SCHEME {name}{} tau {tau} {scheme}: a step {:.1} µs ({:.2} of the current scheme's); against the current scheme's end κ {mk:.2e}, ψ {mp:.2e}, ln N {mn:.2e} ({moved} steps' jumps moved); its own doubling error at τ κ {e1k:.2e}, ψ {e1p:.2e}, at τ/2 κ {e2k:.2e}, ψ {e2p:.2e}, the order log₂(τ-error/τ/2-error) {:.2} (mean over {ratios} states); a jump {:.1} µs ({jumped} jumps)",
                if whole { "" } else { " split" },
                te / k,
                te / t_current,
                ratio_k / ratios.max(1) as f64,
                (with - without) / jumped.max(1) as f64
            );
        }
        set("current");
    }
}

/// OLDNEW item 2: the node's filling (`Selected::fill`) taken apart, its parts timed as it makes them (the least of
/// OLDNEW_REPEAT, 7) at OLDNEW_POINTS (24) points of the centre's cloud: the photon classes zeroed and their channel
/// elements put in, H's outputs put in the generator and its gradient, the decay's spherical components and their
/// products; with their counts. And the middle's scattered light (`photons` + `Scattering::of_moments`) on a state.
#[test]
#[ignore]
fn oldnew_fill() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (count, repeat) = (env("OLDNEW_POINTS", 24), env("OLDNEW_REPEAT", 7));
    let list = std::env::var("OLDNEW_FILL").unwrap_or_else(|_| "PGC3D,TET,tilt0,tilt2".to_string());
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
    for name in list.split(',') {
        let request = request_of(name);
        let keep: Vec<usize> = (0..crate::request::GROUND).collect();
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap();
        let plan = run.rs.plan();
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let n = run.rs.keep.len();
        let mut work = Work::default();
        let making = Making::node(false);
        let (mut t_whole, mut t_ch, mut t_h, mut t_decay, mut t_photons, mut t_moments) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let mut node = run.template.clone();
        for i in 0..count {
            let u = i as f64;
            let r = [0.3 * (1.3 * u).sin(), 0.3 * (2.1 * u + 0.4).cos(), 1.0 * (0.7 * u + 1.1).sin()];
            let kappa = [speed * (0.9 * u).sin(), speed * (1.7 * u).cos(), speed];
            run.scene.point_into(r, kappa, &mut work.point);
            run.rs.evaluate(&run.scene, &work.point, &mut work.local, making);
            work.out.contract(plan, &run.lab, &work.local, None, making);
            let out = &work.out;
            let jumps = &mut work.jumps;
            t_whole += least(&mut || { run.fill(&mut node, out, (false, true), jumps); });
            // (a) the classes zeroed and their channel elements put in.
            t_ch += least(&mut || {
                for c in node.classes.iter_mut() {
                    for axis in c.axes.iter_mut() {
                        for j in axis.iter_mut() {
                            for e in j.elements.iter_mut() {
                                (e.value, e.gradient, e.kick) = (C64::ZERO, [C64::ZERO; 3], C64::ZERO);
                            }
                        }
                    }
                }
                for ((list, c), (_, _, which, _, _)) in run.tables.channels.iter().zip(&out.channels).zip(&run.lab.channel_outputs) {
                    if *which == 1 {
                        continue;
                    }
                    for (k, &(_, class, axis, jump, element)) in list.iter().enumerate() {
                        let e = &mut node.classes[class as usize].axes[axis as usize][jump as usize].elements[element as usize];
                        e.value += c.value[k];
                        for (g, d) in e.gradient.iter_mut().zip(&c.gradient) {
                            *g += d[k];
                        }
                    }
                }
            });
            // (b) H's outputs into the generator and its gradient.
            t_h += least(&mut || {
                for (_, m) in node.generator.iter_mut() {
                    m.data.fill(C64::ZERO);
                }
                for (_, g) in node.gradient.iter_mut() {
                    for m in g.iter_mut() {
                        m.data.fill(C64::ZERO);
                    }
                }
                for (list, c) in run.tables.h.iter().zip(&out.h) {
                    for (k, &(x, g, d)) in list.iter().enumerate() {
                        if g == u32::MAX {
                            continue;
                        }
                        let x = x as usize;
                        node.generator[g as usize].1.data[x] += c.value[k];
                        for (m, dg) in node.gradient[d as usize].1.iter_mut().zip(&c.gradient) {
                            m.data[x] += dg[k];
                        }
                    }
                }
            });
            // (c) the decay: the spherical components, then each product.
            t_decay += least(&mut || {
                jumps.clear();
                for j in &run.tables.jumps {
                    let at = jumps.len();
                    jumps.resize(at + GROUND * j.cols.len(), C64::ZERO);
                    let axes = &node.classes[j.class].axes;
                    for (p, sources) in &j.elements {
                        jumps[at + p] = sources.iter().map(|&(a, c, e, u)| u * axes[a][c].elements[e].value).sum();
                    }
                }
                let half = C64::new(0.0, -0.5);
                let mut from = 0;
                for e in &run.tables.entries {
                    let (mut re, mut im) = (0.0_f64, 0.0_f64);
                    for &(l, r) in &run.tables.terms[from..e.end as usize] {
                        let (x, y) = (jumps[l as usize].conj(), jumps[r as usize]);
                        re = x.re.mul_add(y.re, (-x.im).mul_add(y.im, re));
                        im = x.re.mul_add(y.im, x.im.mul_add(y.re, im));
                    }
                    from = e.end as usize;
                    let p = C64::new(re, im);
                    node.generator[e.forward.0 as usize].1.data[e.forward.1 as usize] += half * p;
                    if e.back.0 != u32::MAX {
                        node.generator[e.back.0 as usize].1.data[e.back.1 as usize] += half * p.conj();
                    }
                }
            });
            // The middle's scattered light on a state: the classes' amplitudes and their moments.
            let psi = spread(n);
            let mut list = Vec::new();
            let values: Vec<f64> = run.read(|b| b.value.clone());
            t_photons += least(&mut || {
                std::hint::black_box(photons(&node, &values, 0.0, &psi, &mut list).len());
            });
            let moments: Vec<Moments> = photons(&node, &values, 0.0, &psi, &mut list).iter().map(|p| p.moments).collect();
            t_moments += least(&mut || {
                std::hint::black_box(Scattering::of_moments(&moments));
            });
        }
        let k = count as f64;
        let channel_elements: usize = run.tables.channels.iter().map(Vec::len).sum();
        let h_elements: usize = run.tables.h.iter().map(Vec::len).sum();
        let elements: usize = node.classes.iter().flat_map(|c| c.axes.iter().flatten().map(|j| j.elements.len())).sum();
        eprintln!(
            "OLDNEW-FILL {name}: the filling {:.1} µs = classes and channels {:.1} ({channel_elements} channel places) + H {:.1} ({h_elements} H places, {} beats) + decay {:.1} ({} components, {} products' terms); the middle's photons {:.1} µs ({} classes, {elements} elements) + moments {:.2} µs",
            t_whole / k,
            t_ch / k,
            t_h / k,
            node.generator.len(),
            t_decay / k,
            run.tables.jumps.iter().map(|j| j.elements.len()).sum::<usize>(),
            run.tables.terms.len(),
            t_photons / k,
            node.classes.len(),
            t_moments / k
        );
    }
}

/// OLDNEW implementation round, items 2 and 3 (the coordinator's 11-23): a jump's photon node and the middle's photons
/// taken apart on the run split by K's fast parts (OLDNEW_JN_WHOLE for the whole one), per scene of OLDNEW_JN (PGC3D,
/// TET, tilt0, tilt2) at OLDNEW_POINTS (12) moving points, each the least of OLDNEW_REPEAT (9): the fold (`Rs::evaluate`)
/// and the contraction (`Out::contract`) with the photon forms (`Making::PHOTONS`) and without them, the photon classes
/// filled (`Selected::fill`); per class the channel outputs that feed it; and on a full node at the point every class's
/// photons whole (`photons`), every class's rate alone and one class whole (`class_photons`).
#[test]
#[ignore]
fn oldnew_jumpnode() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (count, repeat) = (env("OLDNEW_POINTS", 12), env("OLDNEW_REPEAT", 9));
    let list = std::env::var("OLDNEW_JN").unwrap_or_else(|_| "PGC3D,TET,tilt0,tilt2".to_string());
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
    for name in list.split(',') {
        let request = request_of(name);
        let keep: Vec<usize> = (0..crate::request::GROUND).collect();
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let mut run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap();
        let whole = std::env::var("OLDNEW_JN_WHOLE").is_ok();
        if !whole {
            run.split_by_parts();
        }
        let plan = run.rs.plan();
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let n = run.rs.keep.len();
        let mut work = Work::default();
        let no_kicks = Making { kicks: false, ..Making::PHOTONS };
        let (mut ev_ph, mut co_ph, mut fill_ph, mut ev_nk, mut co_nk) = (0.0, 0.0, 0.0, 0.0, 0.0);
        let (mut w_ev, mut w_co) = (0.0, 0.0);
        let (mut t_photons, mut t_rates, mut t_one) = (0.0, 0.0, 0.0);
        let mut photon_node = run.template.clone();
        let mut full = run.template.clone();
        for i in 0..count {
            let u = i as f64;
            let r = [0.3 * (1.3 * u).sin(), 0.3 * (2.1 * u + 0.4).cos(), 1.0 * (0.7 * u + 1.1).sin()];
            let kappa = [speed * (0.9 * u).sin(), speed * (1.7 * u).cos(), speed];
            run.scene.point_into(r, kappa, &mut work.point);
            ev_ph += least(&mut || run.rs.evaluate(&run.scene, &work.point, &mut work.local, Making::PHOTONS));
            w_ev += work.local.work;
            co_ph += least(&mut || work.out.contract(plan, &run.lab, &work.local, None, Making::PHOTONS));
            w_co += work.out.work;
            let out = &work.out;
            let jumps = &mut work.jumps;
            fill_ph += least(&mut || { run.fill(&mut photon_node, out, (true, false), jumps); });
            ev_nk += least(&mut || run.rs.evaluate(&run.scene, &work.point, &mut work.local, no_kicks));
            co_nk += least(&mut || work.out.contract(plan, &run.lab, &work.local, None, no_kicks));
            // The middle's photons on a full node at the point.
            run.node_into(&mut full, r, kappa, (false, true), false);
            let psi = spread(n);
            let norm = norm_sqr(&psi);
            let mut list = Vec::new();
            let values: Vec<f64> = run.read(|b| b.value.clone());
            t_photons += least(&mut || {
                std::hint::black_box(photons(&full, &values, 0.37, &psi, &mut list).len());
            });
            let mut rates = Vec::new();
            t_rates += least(&mut || {
                rates.clear();
                for (class, p) in full.classes.iter().zip(list.iter_mut()) {
                    rates.push(class_photons(class, full.kicked, (&values, 0.37), &psi, norm, p, false));
                }
                std::hint::black_box(rates.len());
            });
            let k = full.classes.len() / 2;
            t_one += least(&mut || {
                std::hint::black_box(class_photons(&full.classes[k], full.kicked, (&values, 0.37), &psi, norm, &mut list[k], true));
            });
        }
        let k = count as f64;
        // Per class the channel outputs that feed it, and the jump pairs and direct entries behind each output.
        let classes = photon_node.classes.len();
        let mut feeds = vec![std::collections::BTreeSet::new(); classes];
        for (o, list) in run.tables.channels.iter().enumerate() {
            for &(_, class, _, _, _) in list {
                feeds[class as usize].insert(o);
            }
        }
        let outputs = run.lab.channel_outputs.len();
        let feeds: Vec<usize> = feeds.iter().map(|s| s.len()).collect();
        let elements: Vec<usize> = photon_node.classes.iter().map(|c| c.axes.iter().flatten().map(|j| j.elements.len()).sum()).collect();
        let jumps: Vec<usize> = photon_node.classes.iter().map(|c| c.axes.iter().map(Vec::len).sum()).collect();
        eprintln!(
            "OLDNEW-JN {name}{}: n {n}; a jump's photon node: fold {:.1} µs ({:.3e} CMA counted) + contraction {:.1} µs ({:.3e}) + fill {:.1} µs; without the photon forms fold {:.1} + contraction {:.1} µs; {classes} classes, {outputs} channel outputs, per class outputs {feeds:?}, components {jumps:?}, elements {elements:?}; the middle's photons every class whole {:.2} µs, every class's rate alone {:.2} µs, one class whole {:.2} µs",
            if whole { "" } else { " split" },
            ev_ph / k,
            w_ev / k,
            co_ph / k,
            w_co / k,
            fill_ph / k,
            ev_nk / k,
            co_nk / k,
            t_photons / k,
            t_rates / k,
            t_one / k
        );
    }
}

/// `class_photons` as it was until 2026-11-23 (the kernel check's reference, `oldnew_kernel`): five passes a component,
/// every buffer cleared and grown again.
fn class_photons_before(class: &Class, kicked: bool, (values, t): (&[f64], f64), psi: &[C64], norm: f64, p: &mut Photons, whole: bool) -> f64 {
    let n = psi.len();
    let Photons { u, du, ku, .. } = &mut *p;
    for v in u.iter_mut().chain(ku.iter_mut()).chain(du.iter_mut().flatten()) {
        v.clear();
        v.resize(n, C64::ZERO);
    }
    for (a, axis) in class.axes.iter().enumerate() {
        for jump in axis {
            let f = if jump.beat == 0 { C64::ONE } else { C64::cis(-values[jump.beat] * t) };
            let apply = |m: &dyn Fn(&Element) -> C64, out: &mut Vec<C64>| {
                for row in jump.elements.chunk_by(|x, y| x.at / n == y.at / n) {
                    let z: C64 = row.iter().map(|e| m(e) * psi[e.at % n]).sum();
                    out[row[0].at / n] += f * z;
                }
            };
            apply(&|e| e.value, &mut u[a]);
            if !whole {
                continue;
            }
            if kicked {
                apply(&|e| e.kick, &mut ku[a]);
            }
            for (i, d) in du.iter_mut().enumerate() {
                apply(&|e| e.gradient[i], &mut d[a]);
            }
        }
    }
    if !whole {
        let weight: f64 = (0..3).map(|a| norm_sqr(&u[a])).sum();
        return weight / norm;
    }
    let moments = Moments::of(u, du, class.absorbed);
    let weight = moments.weight;
    p.wave_number = kicked.then(|| if weight != 0.0 { (0..3).map(|a| inner(&u[a], &ku[a]).re).sum::<f64>() / weight } else { 0.0 });
    p.moments = moments.over(norm);
    p.moments.weight
}

/// OLDNEW item 3 (the coordinator's 11-23): the photons' kernel (`class_photons`, one pass a component) against its
/// form before (`class_photons_before`) to the bit -- every class's u_a, ∇u_a, K u_a, moments, rate and wave number, whole
/// and its rate alone -- on the middle's node (no photon forms) and on a jump's photon node (with them), on the run
/// split by K's fast parts (OLDNEW_KERNEL_WHOLE for the whole one), per scene of OLDNEW_KERNEL (PGC3D, TET, tilt0,
/// tilt2) at OLDNEW_POINTS (12) moving points, three states each, at a time where the beats turn; and each way's time
/// (the least of OLDNEW_REPEAT, 9) of the middle's photons (`photons`: every class whole) and of a jump's rates.
#[test]
#[ignore]
fn oldnew_kernel() {
    let env = |k: &str, d: usize| std::env::var(k).ok().and_then(|x| x.parse().ok()).unwrap_or(d);
    let (count, repeat) = (env("OLDNEW_POINTS", 12), env("OLDNEW_REPEAT", 9));
    let list = std::env::var("OLDNEW_KERNEL").unwrap_or_else(|_| "PGC3D,TET,tilt0,tilt2".to_string());
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
    let bits = |v: &[C64]| v.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]).collect::<Vec<u64>>();
    let same = |p: &Photons, q: &Photons, whole: bool| -> bool {
        let u = (0..3).all(|a| bits(&p.u[a]) == bits(&q.u[a]));
        if !whole {
            return u;
        }
        let du = (0..3).all(|i| (0..3).all(|a| bits(&p.du[i][a]) == bits(&q.du[i][a])));
        let ku = (0..3).all(|a| bits(&p.ku[a]) == bits(&q.ku[a]));
        let m = |m: &Moments| (m.weight.to_bits(), m.mean.map(f64::to_bits), m.second.map(|r| r.map(f64::to_bits)), m.share.to_bits(), m.kick.map(f64::to_bits), m.states);
        u && du && ku && m(&p.moments) == m(&q.moments) && p.wave_number.map(f64::to_bits) == q.wave_number.map(f64::to_bits)
    };
    for name in list.split(',') {
        let request = request_of(name);
        let keep: Vec<usize> = (0..crate::request::GROUND).collect();
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let mut run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep: keep.clone(), ..selection }, reference).unwrap();
        let whole_model = std::env::var("OLDNEW_KERNEL_WHOLE").is_ok();
        if !whole_model {
            run.split_by_parts();
        }
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let n = run.rs.keep.len();
        let values: Vec<f64> = run.read(|b| b.value.clone());
        let (mut t_new, mut t_old, mut r_new, mut r_old, mut checked) = (0.0, 0.0, 0.0, 0.0, 0usize);
        let mut node = run.template.clone();
        let mut photon_node = run.template.clone();
        for i in 0..count {
            let u = i as f64;
            let r = [0.3 * (1.3 * u).sin(), 0.3 * (2.1 * u + 0.4).cos(), 1.0 * (0.7 * u + 1.1).sin()];
            let kappa = [speed * (0.9 * u).sin(), speed * (1.7 * u).cos(), speed];
            run.node_into(&mut node, r, kappa, (false, true), false);
            run.node_into(&mut photon_node, r, kappa, (true, false), false);
            for s in 0..3 {
                let psi: Vec<C64> = (0..n).map(|k| C64::new((0.7 * (k + s) as f64 + 0.3 * u).sin(), (1.3 * (k * s) as f64 + 0.11 * u).cos())).collect();
                let norm = norm_sqr(&psi);
                let t = 0.37 + 0.05 * s as f64;
                for nd in [&node, &photon_node] {
                    for class in &nd.classes {
                        for whole in [true, false] {
                            let (mut p, mut q) = (Photons::default(), Photons::default());
                            let a = class_photons(class, nd.kicked, (&values, t), &psi, norm, &mut p, whole);
                            let b = class_photons_before(class, nd.kicked, (&values, t), &psi, norm, &mut q, whole);
                            assert_eq!(a.to_bits(), b.to_bits(), "{name}: a class's rate apart");
                            assert!(same(&p, &q, whole), "{name}: a class's photons apart (whole {whole}, kicked {})", nd.kicked);
                            // And on buffers a class made before: the kernel's zeroing in place.
                            let c = class_photons(class, nd.kicked, (&values, t), &psi, norm, &mut p, whole);
                            assert_eq!(c.to_bits(), b.to_bits(), "{name}: a reused buffer's rate apart");
                            assert!(same(&p, &q, whole), "{name}: a reused buffer's photons apart");
                            checked += 1;
                        }
                    }
                }
            }
            let psi = spread(n);
            let norm = norm_sqr(&psi);
            let mut list: Vec<Photons> = Vec::new();
            list.resize_with(node.classes.len(), Photons::default);
            t_new += least(&mut || {
                for (class, p) in node.classes.iter().zip(list.iter_mut()) {
                    std::hint::black_box(class_photons(class, node.kicked, (&values, 0.37), &psi, norm, p, true));
                }
            });
            t_old += least(&mut || {
                for (class, p) in node.classes.iter().zip(list.iter_mut()) {
                    std::hint::black_box(class_photons_before(class, node.kicked, (&values, 0.37), &psi, norm, p, true));
                }
            });
            r_new += least(&mut || {
                for (class, p) in photon_node.classes.iter().zip(list.iter_mut()) {
                    std::hint::black_box(class_photons(class, true, (&values, 0.37), &psi, norm, p, false));
                }
            });
            r_old += least(&mut || {
                for (class, p) in photon_node.classes.iter().zip(list.iter_mut()) {
                    std::hint::black_box(class_photons_before(class, true, (&values, 0.37), &psi, norm, p, false));
                }
            });
        }
        let k = count as f64;
        let elements: usize = node.classes.iter().flat_map(|c| c.axes.iter().flatten().map(|j| j.elements.len())).sum();
        let components: usize = node.classes.iter().map(|c| c.axes.iter().map(Vec::len).sum::<usize>()).sum();
        let beats: Vec<usize> = node.classes.iter().map(|c| c.axes.iter().flatten().map(|j| j.beat).collect::<std::collections::BTreeSet<_>>().len()).collect();
        eprintln!("OLDNEW-KERNEL {name}: distinct beats per class {beats:?}");
        eprintln!(
            "OLDNEW-KERNEL {name}{}: {checked} class photons the same to the bit; the middle's photons (every class whole, {} classes, {components} components, {elements} elements) {:.2} µs before, {:.2} µs now ({:.2}); a jump's rates {:.2} µs before, {:.2} µs now ({:.2})",
            if whole_model { "" } else { " split" },
            node.classes.len(),
            t_old / k,
            t_new / k,
            t_new / t_old,
            r_old / k,
            r_new / k,
            r_new / r_old
        );
    }
}
