// TEMPORARY timing harness (to be replaced by the named timing test of tests/model_tables.rs).
mod model_common;
use atom_motion::field::Light;
use atom_motion::linalg::C64;
use atom_motion::model::{Model, Point, Run, Unit, GROUND};
use atom_motion::random::{Purpose, Stream};
use atom_motion::request::Request;
use atom_motion::step::Dynamics;
use serde_json::Value;
use std::path::PathBuf;
use std::time::Instant;

fn scene(name: &str) -> Request {
    let path = if name.ends_with(".json") { PathBuf::from(name) } else { PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json")) };
    let line: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}
fn closed(request: &Request, light: &Light, p: &[f64; GROUND]) -> (Model, Vec<Unit>) {
    let share = atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64);
    let mut kept: Vec<Unit> = vec![];
    loop {
        let model = Model::new(request, light, &kept).unwrap();
        let trap = model.trap(p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, p, request.temperature_uk, share);
        let units = model.units();
        let gates = model.gates(&units, &reach);
        let failing: Vec<Unit> = units.iter().zip(&gates).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            return (model, kept);
        }
        kept.extend(failing);
    }
}
#[test]
fn closure_timing() {
    let name = std::env::var("SCENE").unwrap();
    let request = scene(&name);
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = {
        let mut p = [0.0; GROUND];
        p[b] = 1.0;
        p
    };
    let s = Instant::now();
    let (model, kept) = closed(&request, &light, &p);
    eprintln!("{name}: closure {:?}; kept {} units; {}", s.elapsed(), kept.len(), model.describe());
    for (i, c) in model.clusters().iter().enumerate() {
        eprintln!("  cluster {i}: states {:?} ground {}", c.states, c.ground);
    }
    let s = Instant::now();
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let reach = model.reach(&trap, &p, request.temperature_uk, atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64));
    let tables = model.tabulate(&reach, 1 << 20);
    eprintln!("{name}: tabulate {:?}, {} tables, work {:.3e} complex MA", s.elapsed(), tables.tables.len(), tables.work);
    for t in &tables.tables {
        eprintln!("  cluster {} lights {:?}: axes {:?} nodes {:?} width {}, weights {:?}", t.layout.cluster, t.layout.lights, &t.layout.coordinates[..t.axes()], t.table.axes().iter().map(|a| a.nodes).collect::<Vec<_>>(), t.table.width(), t.layout.weights.iter().map(|w| format!("{w:.2e}")).collect::<Vec<_>>());
    }
    let exact = Run::new(&model, p);
    let fast = Run::tabulated(&model, &tables, p);
    let hm = model.hbar_over_mass();
    let kt = atom_motion::units::KT_PER_UK * request.temperature_uk;
    let mut stream = Stream::new(11, Purpose::Initial, 0, 0);
    let points: Vec<Point> = (0..16)
        .map(|_| {
            let r: [f64; 3] = std::array::from_fn(|a| {
                let e = trap.axes.iter().zip(&trap.frequencies).map(|(axis, w)| axis[a] * (kt * hm).sqrt() / w).collect::<Vec<f64>>();
                trap.centre[a] + e.iter().map(|x| x * model_common::normal(&mut stream)).sum::<f64>()
            });
            Point { r, kappa: std::array::from_fn(|_| (kt / hm).sqrt() * model_common::normal(&mut stream)), t: 0.0 }
        })
        .collect();
    for cluster in 0..model.clusters().len() {
        let n = model.clusters()[cluster].states.len();
        let psi: Vec<C64> = (0..n).map(|_| C64::new(model_common::normal(&mut stream), model_common::normal(&mut stream))).collect();
        // The least of eight batches: the machine is shared.
        let time = |f: &dyn Fn(&Point), rounds: usize| {
            (0..8)
                .map(|_| {
                    let s = Instant::now();
                    for k in 0..rounds {
                        f(&points[k % points.len()]);
                    }
                    s.elapsed().as_secs_f64() * 1e6 / rounds as f64
                })
                .fold(f64::INFINITY, f64::min)
        };
        let rounds = |run: &Run| if run.tables.is_some() { 400 } else { 1 };
        let g = |run: &Run| time(&|at| { std::hint::black_box(run.generator(cluster, at)); }, rounds(run));
        let ph = |run: &Run| time(&|at| { std::hint::black_box(run.photons(cluster, at, &psi)); }, rounds(run));
        let rd = |run: &Run| time(&|at| { std::hint::black_box(run.readout(cluster, at, &psi)); }, rounds(run));
        let en = |run: &Run| time(&|at| { std::hint::black_box(run.energy(cluster, at, &psi)); }, rounds(run));
        let rf = time(&|at| { std::hint::black_box(exact.reference(at.r)); }, 400);
        let parts = exact.generator(cluster, &points[0]).parts.len();
        let classes = exact.photons(cluster, &points[0], &psi).len();
        eprintln!(
            "{name} cluster {cluster} (n {n}, parts {parts}, classes {classes}): generator {:.1} / {:.2} us, photons {:.1} / {:.2} us, readout {:.1} / {:.2} us, energy {:.1} / {:.2} us (exact / tabulated), reference {rf:.2} us",
            g(&exact), g(&fast), ph(&exact), ph(&fast), rd(&exact), rd(&fast), en(&exact), en(&fast)
        );
    }
}

/// TEMPORARY: the exact model's outputs at fixed points, written for the rewrite's regression (not committed).
#[test]
fn golden_dump() {
    use std::fmt::Write;
    let name = std::env::var("SCENE").unwrap();
    let out_path = std::env::var("GOLDEN").unwrap();
    let request = scene(&name);
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = {
        let mut p = [0.0; GROUND];
        p[b] = 1.0;
        p
    };
    let (model, kept) = closed(&request, &light, &p);
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let exact = Run::new(&model, p);
    let hm = model.hbar_over_mass();
    let kt = atom_motion::units::KT_PER_UK * request.temperature_uk;
    let mut stream = Stream::new(11, Purpose::Initial, 0, 0);
    let points: Vec<Point> = (0..4)
        .map(|i| {
            let r: [f64; 3] = std::array::from_fn(|a| {
                let e = trap.axes.iter().zip(&trap.frequencies).map(|(axis, w)| axis[a] * (kt * hm).sqrt() / w).collect::<Vec<f64>>();
                trap.centre[a] + e.iter().map(|x| x * model_common::normal(&mut stream)).sum::<f64>()
            });
            Point { r, kappa: std::array::from_fn(|_| (kt / hm).sqrt() * model_common::normal(&mut stream)), t: if i == 3 { 0.37 } else { 0.0 } }
        })
        .collect();
    let mut out = String::new();
    writeln!(out, "kept {}", kept.len()).unwrap();
    let c = |z: C64| format!("{:e} {:e}", z.re, z.im);
    for cluster in 0..model.clusters().len() {
        let n = model.clusters()[cluster].states.len();
        let psi: Vec<C64> = (0..n).map(|_| C64::new(model_common::normal(&mut stream), model_common::normal(&mut stream))).collect();
        writeln!(out, "cluster {cluster} states {:?} psi {}", model.clusters()[cluster].states, psi.iter().map(|z| c(*z)).collect::<Vec<_>>().join(" ")).unwrap();
        for (pi, at) in points.iter().enumerate() {
            writeln!(out, "point {pi} r {:e} {:e} {:e} kappa {:e} {:e} {:e} t {:e}", at.r[0], at.r[1], at.r[2], at.kappa[0], at.kappa[1], at.kappa[2], at.t).unwrap();
            let g = exact.generator(cluster, at);
            for part in &g.parts {
                writeln!(out, "part beat {:?} omega {:e} wave {:e} {:e} {:e}", part.beat.0, part.omega, part.wave[0], part.wave[1], part.wave[2]).unwrap();
                writeln!(out, "M {}", part.generator.data.iter().map(|z| c(*z)).collect::<Vec<_>>().join(" ")).unwrap();
                for i in 0..3 {
                    writeln!(out, "G{i} {}", part.force[i].data.iter().map(|z| c(*z)).collect::<Vec<_>>().join(" ")).unwrap();
                }
            }
            for ph in exact.photons(cluster, at, &psi) {
                writeln!(out, "photon landing {} k {:e}", ph.landing, ph.wave_number).unwrap();
                for a in 0..3 {
                    writeln!(out, "u{a} {}", ph.amplitudes[a].iter().map(|z| c(*z)).collect::<Vec<_>>().join(" ")).unwrap();
                }
                for e in [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.48, -0.64, 0.6]] {
                    let r = exact.recoil(&ph, &e);
                    writeln!(out, "recoil {:e} {:e} {:e} {}", r.mean[0], r.mean[1], r.mean[2], r.second.iter().flatten().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" ")).unwrap();
                }
            }
            writeln!(out, "readout {}", exact.readout(cluster, at, &psi).iter().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" ")).unwrap();
            writeln!(out, "energy {:e} lost {}", exact.energy(cluster, at, &psi), exact.lost(cluster, at, &psi)).unwrap();
            let q = exact.reference(at.r);
            writeln!(out, "reference {:e} {:e} {:e} {:e} {}", q.value, q.gradient[0], q.gradient[1], q.gradient[2], q.hessian.iter().flatten().map(|x| format!("{x:e}")).collect::<Vec<_>>().join(" ")).unwrap();
        }
    }
    std::fs::write(out_path, out).unwrap();
}

/// TEMPORARY: the closure's rounds, the failing units and their gates.
#[test]
fn closure_rounds() {
    let name = std::env::var("SCENE").unwrap();
    let request = scene(&name);
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = {
        let mut p = [0.0; GROUND];
        p[b] = 1.0;
        p
    };
    let share = atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64);
    let mut kept: Vec<Unit> = vec![];
    for round in 0..40 {
        let s = Instant::now();
        let model = Model::new(&request, &light, &kept).unwrap();
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, request.temperature_uk, share);
        let units = model.units();
        let gates = model.gates(&units, &reach);
        let failing: Vec<(Unit, atom_motion::model::Gate)> = units.iter().zip(&gates).filter(|(_, g)| !g.exists()).map(|(u, g)| (u.clone(), *g)).collect();
        eprintln!("round {round}: {} units, {} failing, {:?}; clusters {:?}; reach cell {:?} speed {:.4}", units.len(), failing.len(), s.elapsed(), model.clusters().iter().map(|c| (c.ground, c.states.len() - c.ground)).collect::<Vec<_>>(), reach.cell, reach.speed);
        let mut shown = 0;
        for (u, g) in &failing {
            if shown < 12 {
                eprintln!("    {} : S {:.3} a {:.3}", model.label(u), g.saturation, g.sweep);
                shown += 1;
            }
        }
        // The folded drives' gates that hold, the largest.
        let mut holding: Vec<(String, f64, f64)> = units.iter().zip(&gates).filter(|(u, g)| g.exists() && matches!(u, Unit::Drive { .. })).map(|(u, g)| (model.label(u), g.saturation, g.sweep)).collect();
        holding.sort_by(|a, b| b.1.total_cmp(&a.1));
        for h in holding.iter().take(4) {
            eprintln!("    holds: {} : S {:.3} a {:.3}", h.0, h.1, h.2);
        }
        if failing.is_empty() {
            break;
        }
        kept.extend(failing.into_iter().map(|f| f.0));
    }
}

/// TEMPORARY: the model's outputs against the golden dump of the code as found.
#[test]
fn golden_check() {
    let name = std::env::var("SCENE").unwrap();
    let golden = std::fs::read_to_string(std::env::var("GOLDEN").unwrap()).unwrap();
    let request = scene(&name);
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = {
        let mut p = [0.0; GROUND];
        p[b] = 1.0;
        p
    };
    let (model, kept) = closed(&request, &light, &p);
    let exact = Run::new(&model, p);
    let numbers = |line: &str, skip: usize| -> Vec<f64> { line.split_whitespace().skip(skip).filter_map(|x| x.parse::<f64>().ok()).collect() };
    let complexes = |line: &str, skip: usize| -> Vec<C64> { numbers(line, skip).chunks(2).map(|c| C64::new(c[0], c[1])).collect() };
    let mut lines = golden.lines().peekable();
    assert_eq!(lines.next().unwrap(), format!("kept {}", kept.len()));
    let (mut cluster, mut psi, mut at) = (0usize, vec![], Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 });
    // The largest differences, relative: generator parts (static time, later time), forces, photons, wave number, recoil, readout, energy, reference.
    let mut worst = [0.0_f64; 9];
    let mut missing = 0;
    let mut generator = None;
    let mut photons: Vec<atom_motion::model::Photon> = vec![];
    let mut taken: Vec<bool> = vec![];
    let mut scale = (0.0_f64, 0.0_f64);
    let mut old_parts: std::collections::BTreeMap<(Vec<i64>, [i64; 3]), Vec<Vec<C64>>> = Default::default();
    let mut old_count = 0;
    while let Some(line) = lines.next() {
        if line.starts_with("cluster ") {
            cluster = line.split_whitespace().nth(1).unwrap().parse().unwrap();
            let at_psi = line.find(" psi ").unwrap();
            psi = complexes(&line[at_psi..], 1);
        } else if line.starts_with("point ") {
            let w: Vec<f64> = line.split_whitespace().filter_map(|x| x.parse::<f64>().ok()).collect();
            at = Point { r: [w[1], w[2], w[3]], kappa: [w[4], w[5], w[6]], t: w[7] };
            let g = exact.generator(cluster, &at);
            scale = (g.parts.iter().map(|p| p.generator.max_abs()).fold(0.0, f64::max), g.parts.iter().flat_map(|p| p.force.iter().map(|f| f.max_abs())).fold(0.0, f64::max));
            generator = Some(g);
            photons = exact.photons(cluster, &at, &psi);
            taken = vec![false; photons.len()];
        } else if line.starts_with("part ") {
            // Parts of one beat and wave vector's value are one: summed on both sides before they are compared.
            let open = line.find("[").unwrap();
            let close = line.find("]").unwrap();
            let beat: Vec<i64> = line[open + 1..close].split(",").filter(|x| !x.trim().is_empty()).map(|x| x.trim().parse().unwrap()).collect();
            let rest: Vec<f64> = line[close + 1..].split_whitespace().filter_map(|x| x.parse::<f64>().ok()).collect();
            let wave = [rest[1], rest[2], rest[3]];
            let m = complexes(lines.next().unwrap(), 1);
            let gs: Vec<Vec<C64>> = (0..3).map(|_| complexes(lines.next().unwrap(), 1)).collect();
            let key = (beat, wave.map(|x| ((x * 1e9).round() as i64)));
            let entry = old_parts.entry(key).or_insert_with(|| vec![vec![C64::ZERO; m.len()]; 4]);
            for (x, y) in entry[0].iter_mut().zip(&m) {
                *x += *y;
            }
            for i in 0..3 {
                for (x, y) in entry[1 + i].iter_mut().zip(&gs[i]) {
                    *x += *y;
                }
            }
            if !lines.peek().is_some_and(|l| l.starts_with("part ")) {
                let g = generator.as_ref().unwrap();
                let mut new_parts: std::collections::BTreeMap<(Vec<i64>, [i64; 3]), Vec<Vec<C64>>> = Default::default();
                for p in &g.parts {
                    let key = (p.beat.0.clone(), p.wave.map(|x| ((x * 1e9).round() as i64)));
                    let entry = new_parts.entry(key).or_insert_with(|| vec![vec![C64::ZERO; p.generator.data.len()]; 4]);
                    for (x, y) in entry[0].iter_mut().zip(&p.generator.data) {
                        *x += *y;
                    }
                    for i in 0..3 {
                        for (x, y) in entry[1 + i].iter_mut().zip(&p.force[i].data) {
                            *x += *y;
                        }
                    }
                }
                let zero = vec![vec![C64::ZERO; g.parts[0].generator.data.len()]; 4];
                let keys: std::collections::BTreeSet<_> = old_parts.keys().chain(new_parts.keys()).cloned().collect();
                for key in keys {
                    let (a, b) = (old_parts.get(&key).unwrap_or(&zero), new_parts.get(&key).unwrap_or(&zero));
                    let dm = a[0].iter().zip(&b[0]).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max) / scale.0;
                    let dg = (1..4).map(|i| a[i].iter().zip(&b[i]).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max)).fold(0.0, f64::max) / scale.1.max(1e-300);
                    let k = if at.t == 0.0 { 0 } else { 1 };
                    if dm > 1e-9 || dg > 1e-9 {
                        eprintln!("cluster {cluster} t {} part {key:?}: dM {dm:.2e} dG {dg:.2e}", at.t);
                    }
                    worst[k] = worst[k].max(dm);
                    worst[2] = worst[2].max(dg);
                }
                eprintln!("cluster {cluster}: parts old {} new {}, of distinct values {} {}", old_count + 1, g.parts.len(), old_parts.len(), new_parts.len());
                old_parts.clear();
                old_count = 0;
            } else {
                old_count += 1;
            }
        } else if line.starts_with("photon ") {
            let w: Vec<f64> = line.split_whitespace().filter_map(|x| x.parse::<f64>().ok()).collect();
            let (landing, k) = (w[0] as usize, w[1]);
            let u: Vec<Vec<C64>> = (0..3).map(|_| complexes(lines.next().unwrap(), 1)).collect();
            let recoils: Vec<Vec<f64>> = (0..3).map(|_| numbers(lines.next().unwrap(), 1)).collect();
            let total: f64 = u.iter().flatten().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let all: f64 = photons.iter().flat_map(|p| p.amplitudes.iter().flatten()).map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            // The nearest untaken class of the same landing.
            let distance = |p: &atom_motion::model::Photon| (0..3).map(|a| u[a].iter().zip(&p.amplitudes[a]).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>()).sum::<f64>().sqrt();
            let best = (0..photons.len()).filter(|&i| !taken[i] && photons[i].landing == landing).min_by(|&i, &j| distance(&photons[i]).total_cmp(&distance(&photons[j])));
            match best {
                None => {
                    if total > 1e-12 * all {
                        missing += 1;
                        eprintln!("missing photon class landing {landing} of size {total:e} (all {all:e})");
                    }
                }
                Some(i) => {
                    taken[i] = true;
                    let d = distance(&photons[i]) / all;
                    if d > 1e-9 {
                        eprintln!("cluster {cluster} photon landing {landing}: du {d:.2e} (class size {:.2e} of {all:.2e})", total);
                    }
                    worst[3] = worst[3].max(d);
                    worst[4] = worst[4].max((photons[i].wave_number - k).abs() / k);
                    for (e, r) in [[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.48, -0.64, 0.6]].iter().zip(&recoils) {
                        let now = exact.recoil(&photons[i], e);
                        // Weighted by the class's share: a class of no weight has no recoil to compare.
                        let share = (total / all).powi(2);
                        let dm = (0..3).map(|x| (now.mean[x] - r[x]).abs()).fold(0.0, f64::max) / k * share;
                        let ds = (0..9).map(|x| (now.second[x / 3][x % 3] - r[3 + x]).abs()).fold(0.0, f64::max) / (k * k) * share;
                        if (dm > 1e-7 || ds > 1e-7) && r.iter().all(|x| x.is_finite()) {
                            eprintln!("cluster {cluster} photon landing {landing} recoil: mean {dm:.2e} second {ds:.2e} share {share:.2e}");
                        }
                        if r.iter().all(|x| x.is_finite()) && now.mean.iter().all(|x| x.is_finite()) {
                            worst[5] = worst[5].max(dm.max(ds));
                        }
                    }
                }
            }
        } else if line.starts_with("readout ") {
            let v = numbers(line, 1);
            let now = exact.readout(cluster, &at, &psi);
            worst[6] = worst[6].max(v.iter().zip(&now).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
            // Photon classes the new model has and the old one had not.
            let all: f64 = photons.iter().flat_map(|p| p.amplitudes.iter().flatten()).map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            for (i, p) in photons.iter().enumerate() {
                let size = p.amplitudes.iter().flatten().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                if !taken[i] && size > 1e-12 * all {
                    eprintln!("cluster {cluster}: a new photon class landing {} of size {:.2e} of {all:.2e}", p.landing, size);
                    worst[3] = worst[3].max(size / all);
                }
            }
        } else if line.starts_with("energy ") {
            let v = numbers(line, 1);
            let now = exact.energy(cluster, &at, &psi);
            worst[7] = worst[7].max((now - v[0]).abs() / v[0].abs().max(1e-300));
            assert_eq!(line.ends_with("true"), exact.lost(cluster, &at, &psi));
        } else if line.starts_with("reference ") {
            let v = numbers(line, 1);
            let q = exact.reference(at.r);
            let now: Vec<f64> = [q.value].into_iter().chain(q.gradient).chain(q.hessian.into_iter().flatten()).collect();
            let size = v.iter().map(|x| x.abs()).fold(0.0, f64::max);
            worst[8] = worst[8].max(v.iter().zip(&now).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / size);
        }
    }
    eprintln!("{name}: worst relative differences: generator {:.2e} (t = 0), {:.2e} (t > 0), force {:.2e}, photons {:.2e}, wave number {:.2e}, recoil {:.2e}, readout {:.2e}, energy {:.2e}, reference {:.2e}; missing {missing}", worst[0], worst[1], worst[2], worst[3], worst[4], worst[5], worst[6], worst[7], worst[8]);
}

#[test]
#[ignore]
fn dbg_r2_photons() {
    use atom_motion::model::{Component, Unit};
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let beam = request.beams.iter().position(|b| b.id == "probe").unwrap();
    let probe = light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap();
    let near = Component { frequency: probe, counter: false };
    let (b, e) = (7, 31);
    let drive = Unit::Drive { carrier: near, excited: e, ground: b };
    let folded = Model::new(&request, &light, &[Unit::Far { component: near }]).unwrap();
    let kept = Model::new(&request, &light, &[Unit::Far { component: near }, drive]).unwrap();
    let at = Point { r: [0.1, 0.2, -0.3], kappa: [0.3, -0.2, 1.1], t: 0.37 };
    eprintln!("probe k {:e}", request.beams[beam].wave_number);
    for (name, model, psi) in [("folded", &folded, vec![C64::ONE]), ("kept", &kept, vec![C64::new(0.6, 0.0), C64::new(0.0, 0.8)])] {
        let c = model.cluster_of(b);
        for ph in model.photons(c, &at, &psi) {
            let w: f64 = ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum();
            let m = ph.recoil(&[1.0, 0.0, 0.0]).second;
            eprintln!("{name}: landing {} k {:.12e} (k/probe - 1 = {:.3e}) weight {:.4e} explicit {} M00 {:.10e} M22 {:.10e}", ph.landing, ph.wave_number, ph.wave_number / request.beams[beam].wave_number - 1.0, w, ph.explicit.len(), m[0][0], m[2][2]);
        }
    }
}
