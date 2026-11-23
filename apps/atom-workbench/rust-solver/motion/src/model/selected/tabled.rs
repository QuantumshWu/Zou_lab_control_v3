//! STEP item 4 (a): the node off the tables of pair products (rs/fold_tabled.rs) against the fold's own node
//! (`Rs::evaluate` + `Out::contract`), element by element, within the bounds the reading gives (fold_tabled.rs, # The
//! bounds, # The interpolation's bound): the strict interpolation's on each output beside the others, its size over the
//! output's largest scale and over its largest element at the node (what the run books), and as the drop tier's rate
//! would book it. Ignored (it reads the requests' folder, V3C_REQ); run by name:
//! `<test exe> model::selected::tests::tabled::the_tabled_node_is_the_node_within_its_bounds --exact --include-ignored
//! --nocapture --test-threads 1`.
use super::oldnew::request_of;
use super::*;
use crate::model::rs::fold::tabled::{Interpolated, Reading, Scratch as TableScratch, Tables};
use crate::model::rs::fold::Values;

/// Per kind [H, channels, lift] the largest of the value's and the gradient's difference over their bound's scale:
/// (|a − b| over the element's own scale (`Out::scales`; the lift's: its group's largest), the same over the bound).
#[derive(Clone)]
struct Worst {
    over_scale: [[f64; 2]; 3],
    over_bound: [[f64; 2]; 3],
}

/// The bound of an element's difference at `orders` terms (fold_tabled.rs, # The bounds) as [on the element's own
/// scale, on its output's largest scale] for the value, and for the gradient [on its own gradient scale (the tail's
/// terms each with its own s-derivative, Σ_{J≥O} (J + 1) ρ^J = ρ^O (O + 1 − Oρ)/(1 − ρ)²), on its value's
/// scale (per µm), on its output's largest gradient scale]: the Doppler tail on the element's own moduli (per
/// eigen-term); ΔG's part, the spans' share and the tables' rounding (γ of a table element's terms) on the output's
/// largest -- they are of the pair's sums, not of the element --; a gradient's beside them the series' terms' gradients
/// through δ's, Σ_{J≥O} J ρ^{J−1} = O ρ^{O−1}/(1 − ρ)² (to first order) times |∇ΔE|/g, on the value's scale, and ΔG's
/// gradient's. The interpolation's, strict and absolute on each output (`Interpolated`), is added by `compare`.
fn bounds(tables: &Tables, reading: &Reading, orders: usize) -> ([f64; 2], [f64; 3]) {
    let rho = reading.rho;
    let rounding = {
        let n = (2 * (tables.degree + 1) + 8) as f64;
        n * crate::units::UNIT_ROUNDOFF / (1.0 - n * crate::units::UNIT_ROUNDOFF)
    };
    // The interpolation's part is the strict one, absolute on each output (`Interpolated`, `compare`).
    let whole = reading.moved + tables.dropped + rounding;
    let tail = reading.tail(orders);
    // A gradient's tail: the series' terms' own s-derivatives, (J + 1) R^{J+2} on the J-th, Σ_{J≥O} (J + 1) ρ^J.
    let o = orders as f64;
    let gradient_tail = rho.powi(orders as i32) * (o + 1.0 - o * rho) / ((1.0 - rho) * (1.0 - rho));
    let through = orders as f64 * rho.powi(orders as i32 - 1) / ((1.0 - rho) * (1.0 - rho)) * reading.shift_gradient + reading.moved_gradient;
    ([tail, whole], [gradient_tail, through, whole])
}

/// The strict interpolation bound (absolute, per output `Interpolated`): per kind [H, channels, lift] × [value, gradient]
/// the largest ratio of it to the output's largest scale.
#[derive(Clone)]
struct Strict {
    over_scale: [[f64; 2]; 3],
    /// Over the output's largest |value| (and |gradient|) at the node: what a node can read of itself.
    over_value: [[f64; 2]; 3],
}

fn compare(a: &Out, b: &Out, lift: &[(Mat, Mat)], bound: ([f64; 2], [f64; 3]), making: Making, strict: &Interpolated, s: &mut Strict) -> Worst {
    let ([bv, wv], [bg, through, wg]) = bound;
    let mut w = Worst { over_scale: [[0.0; 2]; 3], over_bound: [[0.0; 2]; 3] };
    for (kind, (xs, ys), interp) in [(0usize, (&a.h, &b.h), &strict.h), (1, (&a.channels, &b.channels), &strict.channels)] {
        for ((x, y), &[iv, ig]) in xs.iter().zip(ys.iter()).zip(interp.iter()) {
            let top = y.scale.iter().fold(0.0, |m: f64, &x| m.max(x));
            let gtop = y.gradient_scale.iter().flat_map(|g| g.iter()).fold(0.0, |m: f64, &x| m.max(x));
            if top > 0.0 {
                s.over_scale[kind][0] = s.over_scale[kind][0].max(iv / top);
            }
            if gtop > 0.0 {
                s.over_scale[kind][1] = s.over_scale[kind][1].max(ig / gtop);
            }
            let most = x.value.iter().fold(0.0, |m: f64, z| m.max(z.abs()));
            let gmost = x.gradient.iter().flat_map(|g| g.iter()).fold(0.0, |m: f64, z| m.max(z.abs()));
            if most > 0.0 {
                s.over_value[kind][0] = s.over_value[kind][0].max(iv / most);
            }
            if gmost > 0.0 {
                s.over_value[kind][1] = s.over_value[kind][1].max(ig / gmost);
            }
            for k in 0..y.value.len() {
                let scale = y.scale[k];
                let d = (x.value[k] - y.value[k]).abs();
                w.over_scale[kind][0] = w.over_scale[kind][0].max(d / scale.max(f64::MIN_POSITIVE));
                w.over_bound[kind][0] = w.over_bound[kind][0].max(d / (bv * scale + wv * top + iv).max(f64::MIN_POSITIVE));
                for a3 in 0..3 {
                    let gscale = y.gradient_scale[a3][k];
                    let d = (x.gradient[a3][k] - y.gradient[a3][k]).abs();
                    w.over_scale[kind][1] = w.over_scale[kind][1].max(d / gscale.max(f64::MIN_POSITIVE));
                    w.over_bound[kind][1] = w.over_bound[kind][1].max(d / (bg * gscale + through * scale + wg * gtop + ig).max(f64::MIN_POSITIVE));
                }
            }
        }
    }
    if making.lift {
        for ((((xv, xg), (yv, yg)), (sv, sg)), &[iv, ig]) in a.lift.iter().zip(&b.lift).zip(lift).zip(&strict.lift) {
            let top = sv.data.iter().fold(0.0, |m: f64, x| m.max(x.re));
            let gtop = sg.data.iter().fold(0.0, |m: f64, x| m.max(x.re));
            if top > 0.0 {
                s.over_scale[2][0] = s.over_scale[2][0].max(iv / top);
            }
            if gtop > 0.0 {
                s.over_scale[2][1] = s.over_scale[2][1].max(ig / gtop);
            }
            for ((p, q), sc) in xv.data.iter().zip(&yv.data).zip(&sv.data) {
                let d = (*p - *q).abs();
                w.over_scale[2][0] = w.over_scale[2][0].max(d / sc.re.max(f64::MIN_POSITIVE));
                w.over_bound[2][0] = w.over_bound[2][0].max(d / (bv * sc.re + wv * top + iv).max(f64::MIN_POSITIVE));
            }
            for (g, h) in xg.iter().zip(yg) {
                for (((p, q), sc), gs) in g.data.iter().zip(&h.data).zip(&sv.data).zip(&sg.data) {
                    let d = (*p - *q).abs();
                    w.over_scale[2][1] = w.over_scale[2][1].max(d / gs.re.max(f64::MIN_POSITIVE));
                    w.over_bound[2][1] = w.over_bound[2][1].max(d / (bg * gs.re + through * sc.re + wg * gtop + ig).max(f64::MIN_POSITIVE));
                }
            }
        }
    }
    w
}

/// On PGC3D, TET, tilt0 and tilt2 (their ground states; TABLED_SCENES to choose) the tables built (3 Doppler terms,
/// degree 4, from the least s of the walk) and their time and memory; along a walk of TABLED_STEPS steps (60) at the
/// scene's step, at every step's end the middle's node both ways (the model with its gradients and the lift), every
/// H, channel and lift element compared at 1–3 terms, the difference over the element's scale and over its bound; and
/// each way's time (the least of 5).
#[test]
#[ignore]
fn the_tabled_node_is_the_node_within_its_bounds() {
    let scenes = std::env::var("TABLED_SCENES").unwrap_or_else(|_| "PGC3D:0.474,TET:0.474,tilt0:0.163,tilt2:0.326".to_string());
    let steps: usize = std::env::var("TABLED_STEPS").ok().and_then(|x| x.parse().ok()).unwrap_or(60);
    let (orders, degree) = (3, 4);
    let least = |f: &mut dyn FnMut()| -> f64 {
        f();
        (0..5)
            .map(|_| {
                let clock = std::time::Instant::now();
                f();
                clock.elapsed().as_secs_f64() * 1e6
            })
            .fold(f64::INFINITY, f64::min)
    };
    let mut failed = Vec::new();
    for entry in scenes.split(',') {
        // `name:tau`, or `name+label+label:tau` with explicit excited states (R4+D2-2-1).
        let (spec, tau) = entry.split_once(':').expect("name:tau");
        let tau: f64 = tau.parse().unwrap();
        let mut labels = spec.split('+');
        let name = labels.next().unwrap();
        let request = request_of(name);
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let reference = crate::solve::alone(crate::solve::chief(&request));
        let mut keep: Vec<usize> = (0..crate::request::GROUND).collect();
        keep.extend(labels.map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
        keep.sort_unstable();
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep, ..selection }, reference).unwrap();
        let plan = run.rs.plan();
        // The walk: the whole model's steps from an off-centre thermal atom.
        let chief = crate::solve::chief(&request);
        let v0 = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let mut t = Trajectory::new(&run, chief, [0.1, 0.2, 0.3], [-v0, 0.7 * v0, -1.2 * v0], 7, 203);
        let mut walk = Vec::new();
        for _ in 0..steps {
            let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Anywhere)) else { break };
            t = step.end;
            walk.push((t.r, t.kappa));
        }
        let mut work = Work::default();
        let values = |r: V3, kappa: V3, work: &mut Work| -> Values {
            run.scene.point_into(r, kappa, &mut work.point);
            let mut v = Values::default();
            run.rs.tweezer().values_into(&run.scene, &work.point, &mut v);
            v
        };
        let rest = values([0.0; 3], [0.0; 3], &mut work);
        let reach = run.reach();
        let low = walk.iter().map(|&(r, k)| Tables::intensity(plan, &values(r, k, &mut work))).fold(1.0, f64::min);
        let Some(layout) = Tables::layout(plan, &run.lab, &rest, &reach, orders, degree, low, None) else {
            eprintln!("TABLED {name}: no tables (a block not tabled)");
            continue;
        };
        eprintln!("TABLED {name}: the layout {:.3e} counted, the fill priced {:.3e}", layout.work, layout.price);
        let layout_work = layout.work;
        let Some(tables) = layout.fill(plan, &run.lab, &rest) else {
            eprintln!("TABLED {name}: no tables (a block not tabled)");
            continue;
        };
        eprintln!("TABLED {name}: the fill counted {:.3e}", tables.work - layout_work);
        eprintln!(
            "TABLED {name}: built in {:.3} s, {:.2} MB, {} pieces of s on [{low:.4}, 1] at degree {degree} (strict ≤ {:.1e}), the spans' dropped share {:.1e}",
            tables.build_seconds,
            tables.bytes as f64 / 1e6,
            tables.pieces.len(),
            tables.pieces.iter().map(|p| p.worst[0]).fold(0.0, f64::max),
            tables.dropped
        );
        // Per piece the strict bound against the entries' scale, per term (the largest over the tables), beside the
        // coefficients' estimate.
        for p in &tables.pieces {
            eprintln!("TABLED {name} piece [{:.4}, {:.4}]: the singularity {:.1} half widths off; worst {:.1e} (values), {:.1e} (slopes)", p.range[0], p.range[1], p.room, p.worst[0], p.worst[1]);
        }
        // The model with its gradients, the photon forms and the lift: every table read.
        let making = Making { lift: true, ..Making::node(true) };
        let (mut scratch, mut table_out) = (TableScratch::default(), Out::default());
        let mut worst = vec![(Worst { over_scale: [[0.0; 2]; 3], over_bound: [[0.0; 2]; 3] }, 0.0f64); orders];
        let mut strict = vec![Strict { over_scale: [[0.0; 2]; 3], over_value: [[0.0; 2]; 3] }; orders];
        let mut rate_most = 0.0f64;
        let mut relative_most = [0.0f64; 2];
        let (mut read_count, mut rho_max, mut moved_max, mut moved_gradient_max) = (0, 0.0f64, 0.0f64, 0.0f64);
        let (mut exact_t, mut table_t) = (0.0, vec![0.0; orders]);
        let plain = Making::node(false);
        let (mut exact_plain, mut table_plain) = (0.0, vec![0.0; orders]);
        let (mut exact_work, mut table_work) = (0.0, vec![0.0; orders]);
        for &(r, kappa) in &walk {
            run.scene.point_into(r, kappa, &mut work.point);
            run.rs.evaluate(&run.scene, &work.point, &mut work.local, making);
            work.out.contract(plan, &run.lab, &work.local, None, making);
            exact_work += work.local.work + work.out.work;
            work.out.scales(plan, &run.lab, &work.local, None);
            let lift_scales = Out::lift_scales(plan, &run.lab, &work.local);
            let v = work.local.values.clone();
            let Some(reading) = tables.read(plan, &run.lab, &v, &mut scratch) else { continue };
            read_count += 1;
            (rho_max, moved_max, moved_gradient_max) = (rho_max.max(reading.rho), moved_max.max(reading.moved), moved_gradient_max.max(reading.moved_gradient));
            for o in 1..=orders {
                table_work[o - 1] += tables.contract(plan, &run.lab, &v, &reading, o, making, &mut table_out, &mut scratch);
                // The interpolation's values as the drop tier books a change of the outputs (`Selected::rate`): each
                // output's per-element bound times the root of its elements (its Frobenius norm), each channel's ‖L‖ the
                // node's own (its value outputs' Frobenius norms summed over its beats) plus the change.
                let i = &scratch.interpolated;
                let frobenius = |x: &[[f64; 2]], lists: &[Vec<u32>]| -> Vec<f64> { x.iter().zip(lists).map(|(b, l)| b[0] * (l.len() as f64).sqrt()).collect() };
                let (h, channels) = (frobenius(&i.h, &run.lab.lists), frobenius(&i.channels, &run.lab.channel_lists));
                let mut whole: std::collections::HashMap<(usize, [i64; 2]), f64> = std::collections::HashMap::new();
                for (o, (q, fast, which, ..)) in run.lab.channel_outputs.iter().enumerate() {
                    if *which == 0 {
                        *whole.entry((*q, *fast)).or_insert(0.0) += table_out.channels[o].value.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() + channels[o];
                    }
                }
                let norms: Vec<f64> = run.lab.channel_outputs.iter().map(|(q, fast, ..)| whole.get(&(*q, *fast)).copied().unwrap_or(0.0)).collect();
                let y = run.rate(&h, &channels, &norms);
                rate_most = rate_most.max(y);
                relative_most = [relative_most[0].max(scratch.interpolated.relative[0]), relative_most[1].max(scratch.interpolated.relative[1])];
                let w = compare(&table_out, &work.out, &lift_scales, bounds(&tables, &reading, o), making, &scratch.interpolated, &mut strict[o - 1]);
                let (acc, tail) = &mut worst[o - 1];
                for k in 0..3 {
                    for g in 0..2 {
                        acc.over_scale[k][g] = acc.over_scale[k][g].max(w.over_scale[k][g]);
                        acc.over_bound[k][g] = acc.over_bound[k][g].max(w.over_bound[k][g]);
                    }
                }
                *tail = tail.max(reading.tail(o));
            }
            exact_t += least(&mut || {
                run.scene.point_into(r, kappa, &mut work.point);
                run.rs.evaluate(&run.scene, &work.point, &mut work.local, making);
                work.out.contract(plan, &run.lab, &work.local, None, making);
            });
            exact_plain += least(&mut || {
                run.scene.point_into(r, kappa, &mut work.point);
                run.rs.evaluate(&run.scene, &work.point, &mut work.local, plain);
                work.out.contract(plan, &run.lab, &work.local, None, plain);
            });
            for o in 1..=orders {
                table_t[o - 1] += least(&mut || {
                    let v = values(r, kappa, &mut work);
                    let reading = tables.read(plan, &run.lab, &v, &mut scratch).expect("read once");
                    tables.contract(plan, &run.lab, &v, &reading, o, making, &mut table_out, &mut scratch);
                });
                table_plain[o - 1] += least(&mut || {
                    let v = values(r, kappa, &mut work);
                    let reading = tables.read(plan, &run.lab, &v, &mut scratch).expect("read once");
                    tables.contract(plan, &run.lab, &v, &reading, o, plain, &mut table_out, &mut scratch);
                });
            }
        }
        let n = read_count.max(1) as f64;
        eprintln!("TABLED {name}: the interpolation's values as a rate Y ≤ {rate_most:.2e} /µs (over 1 ms {:.2e}); the judgment's dropped terms' Y {:.2e} /µs; the node's relative bound ≤ {:.1e} (values), {:.1e} (gradients)", rate_most * 1e3, run.dropped(), relative_most[0], relative_most[1]);
        eprintln!("TABLED {name}: {read_count} of {} walk points read off the tables; ρ ≤ {rho_max:.2e}, ΔG's ε ≤ {moved_max:.1e}, its gradient's ≤ {moved_gradient_max:.1e} /µm; the node {:.1} µs ({:.3e} counted), without the lift {:.1} µs", walk.len(), exact_t / n, COMPLEX * exact_work / n, exact_plain / n);
        for (o, (w, tail)) in worst.iter().enumerate() {
            eprintln!(
                "TABLED {name} {} terms: {:.1} µs ({:.3e} counted), without the lift {:.1} µs, tail ≤ {tail:.1e}; |Δ| over scale [H, channels, lift] × [value, gradient] {:?}; over the bound {:?}",
                o + 1,
                table_t[o] / n,
                COMPLEX * table_work[o] / n,
                table_plain[o] / n,
                w.over_scale.map(|x| x.map(|y| format!("{y:.1e}"))),
                w.over_bound.map(|x| x.map(|y| format!("{y:.2}")))
            );
            eprintln!("TABLED {name} {} terms: the strict interpolation bound over the output's largest scale {:?}", o + 1, strict[o].over_scale.map(|x| x.map(|y| format!("{y:.1e}"))));
            eprintln!("TABLED {name} {} terms: the strict interpolation bound over the output's largest |value| at the node {:?}", o + 1, strict[o].over_value.map(|x| x.map(|y| format!("{y:.1e}"))));
            if o + 1 >= 2 && w.over_bound.iter().flatten().any(|&x| x > 1.0) {
                failed.push(format!("{name} at {} terms", o + 1));
            }
        }
    }
    assert!(failed.is_empty(), "beyond the bounds: {failed:?}");
}

/// Two filled nodes' largest difference over the second's largest entry: [the generator's, its gradient's, the photon
/// classes' elements' (values and photon forms)] (from OLDNEW's `oldnew_pairs`).
fn node_difference(a: &Node, b: &Node) -> [f64; 3] {
    let (mut d, mut top) = ([0.0f64; 3], [0.0f64; 3]);
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
    for (x, y) in a.classes.iter().zip(&b.classes) {
        for (ax, ay) in x.axes.iter().zip(&y.axes) {
            for (jx, jy) in ax.iter().zip(ay) {
                for (p, q) in jx.elements.iter().zip(&jy.elements) {
                    d[2] = d[2].max((p.value - q.value).abs().max((p.kick - q.kick).abs()));
                    top[2] = top[2].max(q.value.abs().max(q.kick.abs()));
                }
            }
        }
    }
    std::array::from_fn(|k| d[k] / top[k].max(f64::MIN_POSITIVE))
}

/// The run's node off the tables it bought (model/selected.rs, # The tables, # The tables' budget) against its fold's
/// own, filled -- what a step reads: on PGC3D, TET, tilt2 and R4 with D2-2-1 (TABLED_SCENES as the test above's), the
/// tables bought on [the walk's least s, 1] (`Selected::buy_tables_now`), at every point of a walk of TABLED_STEPS (60)
/// the node with the photon forms on version 1 and on version 0 (`step::on_version`), their largest difference over
/// the largest entry (`node_difference`: the generator, its gradient, the photon classes), beside the node's own booked
/// deviation (`Node::deviation`); and the kinds the tables are read for and their budget. Ignored, by name.
#[test]
#[ignore]
fn the_run_s_tabled_node_is_its_fold_s() {
    let scenes = std::env::var("TABLED_SCENES").unwrap_or_else(|_| "PGC3D:0.474,TET:0.474,tilt2:0.326,R4+D2-2-1:0.237".to_string());
    let steps: usize = std::env::var("TABLED_STEPS").ok().and_then(|x| x.parse().ok()).unwrap_or(60);
    for entry in scenes.split(',') {
        let (spec, tau) = entry.split_once(':').expect("name:tau");
        let tau: f64 = tau.parse().unwrap();
        let mut labels = spec.split('+');
        let name = labels.next().unwrap();
        let request = request_of(name);
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let chief = crate::solve::chief(&request);
        let mut keep: Vec<usize> = (0..crate::request::GROUND).collect();
        keep.extend(labels.map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
        keep.sort_unstable();
        let run = Selected::new(crate::scene::Scene::new(&request).unwrap(), crate::select::Selection { keep, ..selection }, crate::solve::alone(chief)).unwrap();
        let v0 = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let mut t = Trajectory::new(&run, chief, [0.1, 0.2, 0.3], [-v0, 0.7 * v0, -1.2 * v0], 7, 203);
        let mut walk = Vec::new();
        for _ in 0..steps {
            let Ok(step) = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut Scratch::default(), &Anywhere)) else { break };
            t = step.end;
            walk.push((t.r, t.kappa));
        }
        let mut work = Work::default();
        let low = walk
            .iter()
            .map(|&(r, k)| {
                run.scene.point_into(r, k, &mut work.point);
                let mut v = Values::default();
                run.rs.tweezer().values_into(&run.scene, &work.point, &mut v);
                Tables::intensity(run.rs.plan(), &v)
            })
            .fold(1.0, f64::min);
        if !run.buy_tables_now(low) {
            eprintln!("TABLED-RUN {spec}: no tables kept (no kind of node costs less off them)");
            continue;
        }
        let (uses, budget) = run.bought_for().unwrap();
        let (mut worst, mut deviation, mut off) = ([0.0f64; 3], [0.0f64; 2], 0usize);
        for &(r, kappa) in &walk {
            let (a, b) = (crate::step::on_version(1, || run.node(r, kappa, true)), crate::step::on_version(0, || run.node(r, kappa, true)));
            let d = node_difference(&a, &b);
            worst = std::array::from_fn(|k| worst[k].max(d[k]));
            deviation = [deviation[0].max(a.deviation[0]), deviation[1].max(a.deviation[1])];
            off += usize::from(a.deviation != [0.0; 2]);
        }
        eprintln!(
            "TABLED-RUN {spec}: for the kinds [node, with the lift, photon, with the photon forms] {uses:?}, budget {budget:?}; {off} of {} walk points off the tables; the filled node's largest difference over the largest entry: generator {:.2e}, gradient {:.2e}, photon classes {:.2e}; its booked deviation at the most {:.2e} (rates), {:.2e} (friction)",
            walk.len(),
            worst[0],
            worst[1],
            worst[2],
            deviation[0],
            deviation[1]
        );
    }
}
