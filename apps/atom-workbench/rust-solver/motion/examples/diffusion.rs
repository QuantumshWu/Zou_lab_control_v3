//! The engine's momentum bookkeeping on a prescribed path, and its model's master equation for an exact reference
//! (motion3d_status/rewrite/diffusion.status.md; the reference itself is motion/refs/diffusion/ref.py). A measurement
//! tool beside the crate: it calls the model and the step as the solve does and changes neither.
//!
//! `diffusion info <scene> [waist]`: the clusters and units of the scene's model as the gates close it.
//! `diffusion dump <scene> <waist> <kind> <points.json> <out.json>`: the model's master equation at the points
//!   [[z, κ_z], ...] (x = y = 0): per cluster the generator H, ∂_z of its Hermitian part and the jump operators, these
//!   also at z ∓ ε (their gradient). `kind` "closed": the gates' closure, the engine's own model; "explicit": every
//!   near-resonant drive explicit beside it (no fold of the cooling and repump light: the textbook master equation).
//! `diffusion engine <scene> <waist> <course> <tau> <trajectories> <burn> <span> <every> <out.json>`: trajectories of
//!   the closed model held on a course (`engine`).
//! `diffusion free <scene> <1d|3d|free:axes> <temperature> <tau> <atoms> <span> <every> <out.json>`: the engine's own
//!   dynamics from a thermal start (`free`).
//! `diffusion account <scene> <1d|3d|free:axes> <temperature> <tau> <atoms> <span> <every> <out.json>`: the same, every
//!   change of an axis's kinetic energy booked to the part of the engine that made it (`account`).
//!
//! `waist` (µm; 0: the scene's own) replaces the trap beam's waist: the same light at the focus, uniform over the path.

use atom_motion::field::Light;
use atom_motion::linalg::{C64, Mat};
use atom_motion::model::{Model, Point, Run, Unit, GROUND};
use atom_motion::request::Request;
use atom_motion::step::{Attempt, Brownian, Dynamics, Middle, Midway, Scratch, Trajectory};

/// The distance (µm) over which a jump operator's gradient is read as a central difference: the operators change over
/// 1/k ≈ 0.12 µm, so the difference errs by (kε)²/6 ≈ 10⁻⁷ of the gradient.
const EPSILON: f64 = 1e-4;

fn scene(name: &str, waist: f64) -> Request {
    let path = if name.ends_with(".json") { std::path::PathBuf::from(name) } else { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json")) };
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut request = Request::from_native(&line).unwrap();
    if waist > 0.0 {
        let trap = request.trap_beam;
        request.beams[trap].waist = Some(waist);
    }
    request
}

fn reference(request: &Request) -> (usize, [f64; GROUND]) {
    let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut p = [0.0; GROUND];
    p[state] = 1.0;
    (state, p)
}

/// The units the gates keep over the scene's own reach (the solve's closure without its pilot's keeps).
fn closure(name: &str) -> Vec<Unit> {
    let request = scene(name, 0.0);
    let light = Light::new(&request);
    let (_, p) = reference(&request);
    let mut kept: Vec<Unit> = vec![];
    loop {
        let model = Model::new(&request, &light, &kept).unwrap();
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, request.temperature_uk, atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64));
        let units = model.units();
        let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            return kept;
        }
        kept.extend(failing);
    }
}

fn describe(request: &Request, model: &Model) {
    for (c, cluster) in model.clusters().iter().enumerate() {
        let names: Vec<&str> = cluster.states.iter().map(|&s| request.states[s].as_str()).collect();
        eprintln!("cluster {c} ({}+{}): {names:?}", cluster.ground, cluster.states.len() - cluster.ground);
    }
}

fn info(name: &str, waist: f64) {
    let kept = closure(name);
    eprintln!("{name}: the gates keep {} units", kept.len());
    for u in &kept {
        eprintln!("  kept {u:?}");
    }
    let request = scene(name, waist);
    let light = Light::new(&request);
    let (state, p) = reference(&request);
    let model = Model::new(&request, &light, &kept).unwrap();
    describe(&request, &model);
    let trap = model.trap(&p, &model.prepared()).unwrap();
    eprintln!("reference state {} ; trap centre {:?} frequencies {:?} rad/µs lab {:?}; ħ/m {}", request.states[state], trap.centre, trap.frequencies, trap.lab, model.hbar_over_mass());
    for (f, b) in request.beams.iter().enumerate() {
        eprintln!("beam {f} {} k {} index {:?}", b.id, b.wave_number, b.frequency_index);
    }
    // Where a jump's bare state lands (`Run::landed`: the slow state it continues in, with its norm): the weight that
    // does not continue -- the probability of a prompt photon -- from each basis state.
    let run = Run::new(&model, p);
    for z in [0.0, 0.05, 0.1] {
        let at = Point { r: [0.0, 0.0, z], kappa: [0.0; 3], t: 0.0 };
        for (c, cluster) in model.clusters().iter().enumerate() {
            let prompt: Vec<f64> = (0..cluster.states.len())
                .map(|i| {
                    let f: Vec<C64> = (0..cluster.states.len()).map(|x| if x == i { C64::ONE } else { C64::ZERO }).collect();
                    1.0 - run.landed(c, &at, &f).iter().map(|z| z.norm_sqr()).sum::<f64>()
                })
                .collect();
            eprintln!("z = {z}: cluster {c} lands with a prompt photon at {prompt:?}");
        }
    }
}

/// The model with every near-resonant drive explicit: the gates' closure, and of the frequencies that have a carrier
/// there (the cooling and the repump light) every co-rotating drive from the hyperfine level it is near to the D2 states,
/// kept bare and then kept at the atom.
fn explicit(name: &str, request: &Request, light: &Light) -> Model {
    let mut kept = closure(name);
    let carriers: Vec<(usize, bool)> = kept.iter().filter_map(|u| if let Unit::Bare { component, ground, .. } = u { Some((component.frequency, *ground < 3)) } else { None }).collect();
    let near = |component: &atom_motion::model::Component, excited: usize, ground: usize| !component.counter && excited >= 16 && carriers.contains(&(component.frequency, ground < 3));
    for round in 0..2 {
        let model = Model::new(request, light, &kept).unwrap();
        let more: Vec<Unit> = model
            .units()
            .into_iter()
            .filter(|u| match u {
                Unit::Bare { component, excited, ground } => round == 0 && near(component, *excited, *ground),
                Unit::Drive { component, excited, ground } => round == 1 && near(component, *excited, *ground),
                _ => false,
            })
            .filter(|u| !kept.contains(u))
            .collect();
        kept.extend(more);
    }
    Model::new(request, light, &kept).unwrap()
}

fn matrix_json(m: &[C64]) -> serde_json::Value {
    serde_json::json!(m.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<f64>>())
}

/// A cluster's generator and ∂_z of its Hermitian part at a point, the generator taken at the momentum κ (n × n, row by
/// row). G from its forms Re tr(G ρ): ρ = E_ij + E_ji gives 2 Re G_ji, ρ = i(E_ij − E_ji) gives −2 Im G_ji.
fn generator_at(run: &Run, cluster: usize, at: &Point) -> (Mat, Vec<C64>) {
    let n = run.cluster(cluster).states.len();
    let middle = run.middle(cluster, at.r, at.t);
    let path = middle.path(at.kappa, [0.0; 3]);
    let mut h = Mat::zeros(n, n);
    middle.generator(&path, 0.0, &mut h);
    let mut g = vec![C64::ZERO; n * n];
    let mut rho = vec![C64::ZERO; n * n];
    for i in 0..n {
        for j in i..n {
            rho.fill(C64::ZERO);
            if i == j {
                rho[i * n + i] = C64::ONE;
                g[i * n + i] = C64::real(middle.force(&path, 0.0, &rho)[2]);
                continue;
            }
            rho[i * n + j] = C64::ONE;
            rho[j * n + i] = C64::ONE;
            let re = 0.5 * middle.force(&path, 0.0, &rho)[2];
            rho[i * n + j] = C64::new(0.0, 1.0);
            rho[j * n + i] = C64::new(0.0, -1.0);
            let im = 0.5 * middle.force(&path, 0.0, &rho)[2];
            g[j * n + i] = C64::new(re, -im);
            g[i * n + j] = C64::new(re, im);
        }
    }
    (h, g)
}

/// The jump operators of a cluster at a point: per class (landing cluster, photon wave number) and emitted axis a the
/// matrix J[b', s] on the cluster's states, from the photons of its basis states.
fn jumps_at(run: &Run, cluster: usize, at: &Point) -> Vec<serde_json::Value> {
    let n = run.cluster(cluster).states.len();
    let mut classes: std::collections::BTreeMap<usize, (usize, f64, f64, Vec<Vec<C64>>)> = std::collections::BTreeMap::new();
    for s in 0..n {
        let mut psi = vec![C64::ZERO; n];
        psi[s] = C64::ONE;
        for photon in run.photon_amplitudes(cluster, at, &psi) {
            let m = run.cluster(photon.landing).states.len();
            let entry = classes.entry(photon.class).or_insert_with(|| (photon.landing, photon.wave_number, photon.absorbed, vec![vec![C64::ZERO; m * n]; 3]));
            for a in 0..3 {
                for b in 0..m {
                    entry.3[a][b * n + s] = photon.amplitudes[a][b];
                }
            }
        }
    }
    classes.into_iter().map(|(class, (landing, k, absorbed, j))| serde_json::json!({"class": class, "landing": landing, "k": k, "absorbed": absorbed, "j": j.iter().map(|m| matrix_json(m)).collect::<Vec<_>>()})).collect()
}

fn dump(name: &str, waist: f64, kind: &str, points: &str, out: &str) {
    let request = scene(name, waist);
    let light = Light::new(&request);
    let (_, p) = reference(&request);
    let model = if kind == "explicit" { explicit(name, &request, &light) } else { Model::new(&request, &light, &closure(name)).unwrap() };
    describe(&request, &model);
    let run = Run::new(&model, p);
    let list: Vec<[f64; 2]> = serde_json::from_str(&std::fs::read_to_string(points).unwrap()).unwrap();
    let points: Vec<serde_json::Value> = list
        .iter()
        .map(|&[z, kappa]| {
            let at = |dz: f64| Point { r: [0.0, 0.0, z + dz], kappa: [0.0, 0.0, kappa], t: 0.0 };
            let clusters: Vec<serde_json::Value> = (0..model.clusters().len())
                .map(|c| {
                    let (h, g) = generator_at(&run, c, &at(0.0));
                    serde_json::json!({"h": matrix_json(&h.data), "g": matrix_json(&g), "jumps": jumps_at(&run, c, &at(0.0)), "before": jumps_at(&run, c, &at(-EPSILON)), "after": jumps_at(&run, c, &at(EPSILON))})
                })
                .collect();
            serde_json::json!({"z": z, "kappa": kappa, "clusters": clusters})
        })
        .collect();
    let clusters: Vec<serde_json::Value> = model.clusters().iter().map(|c| serde_json::json!({"states": c.states, "ground": c.ground, "names": c.states.iter().map(|&s| request.states[s].clone()).collect::<Vec<_>>()})).collect();
    std::fs::write(out, serde_json::to_string(&serde_json::json!({"scene": name, "waist": waist, "kind": kind, "epsilon": EPSILON, "hbar_over_mass": model.hbar_over_mass(), "clusters": clusters, "points": points})).unwrap()).unwrap();
}

/// A prescribed path along z (x = y = 0): at rest, at a constant velocity (the position wrapped by the light's period, on
/// which the generator is the same), or the oscillation z_c + A cos(ωt + φ).
#[derive(Clone, Copy, Debug)]
enum Course {
    Rest { z: f64 },
    Constant { z: f64, v: f64, period: f64 },
    Oscillation { centre: f64, amplitude: f64, omega: f64, phase: f64 },
}

impl Course {
    /// (z, v) at the time t.
    fn at(&self, t: f64) -> (f64, f64) {
        match *self {
            Course::Rest { z } => (z, 0.0),
            Course::Constant { z, v, period } => {
                let x = z + v * t;
                (x - period * (x / period).round(), v)
            }
            Course::Oscillation { centre, amplitude, omega, phase } => (centre + amplitude * (omega * t + phase).cos(), -amplitude * omega * (omega * t + phase).sin()),
        }
    }
    /// The two weights a kick at the time t is accumulated with beside 1: an oscillation's quadratures −sin(ωt + φ) (the
    /// velocity's, over Aω) and cos(ωt + φ); 1 and 0 elsewhere.
    fn weights(&self, t: f64) -> [f64; 2] {
        match *self {
            Course::Oscillation { omega, phase, .. } => [-(omega * t + phase).sin(), (omega * t + phase).cos()],
            _ => [1.0, 0.0],
        }
    }
}

/// The width of a sample (`held`).
const WIDTH: usize = 11;

/// One trajectory of the engine held on a course: the model's steps and jumps as the solve takes them (`adapt::advance`'s
/// loop, on a Brownian path that does not move), the position and the momentum put back on the course before every step
/// and every jump. What the engine gives the momentum along z is accumulated apart, from the time `burn` on, and
/// sampled every `every` µs: with the weights w = 1, w₁, w₂ (`Course::weights`, at a step's middle) in turn
/// [Σ w Δκ of the steps (the Ehrenfest impulse and the scattered light's force), Σ w Δκ of the jumps (the emitted recoil
/// and the absorbed mean), ∫ w² 2D_zz dt of the Brownian part], then the jumps' count and the population outside the
/// start's hyperfine level. Beside them the normalized internal state (its cluster and amplitudes) at every sample.
fn held(run: &Run, start: usize, course: Course, tau: f64, burn: f64, span: f64, every: f64, seed: u64, index: u64) -> (Vec<[f64; WIDTH]>, Vec<(usize, Vec<C64>)>) {
    let hm = run.hbar_over_mass();
    let place = |course: &Course, t: &mut Trajectory| {
        let (z, v) = course.at(t.time);
        t.r = [0.0, 0.0, z];
        t.kappa = [0.0, 0.0, v / hm];
    };
    let mut t = Trajectory::new(run, start, [0.0; 3], [0.0; 3], seed, index);
    let mut scratch = Scratch::default();
    // `DPHI=<rad²/µs>`: an oscillation's phase walks at that rate (a normal step of variance DPHI × the step's length
    // before every step, from a stream of the trajectory's own): the same amplitude, a path that loses its phase.
    let wander: f64 = std::env::var("DPHI").map_or(0.0, |x| x.parse().unwrap());
    let mut walk = atom_motion::random::Stream::new(67, atom_motion::random::Purpose::Initial, index, 0);
    let mut course = course;
    let (mut sums, mut out, mut states) = ([0.0; WIDTH], vec![], vec![]);
    let mut next = burn;
    let end = burn + span;
    // The share of a step's length within which a threshold's root is the step's end (adapt.rs takes its own).
    let timing = f64::EPSILON.sqrt();
    while t.time < end {
        let rest = next - t.time;
        let mut most = tau.min(rest);
        let mut bracket = (0.0, f64::INFINITY);
        let norm: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
        let ahead = if t.rate > 0.0 { (norm / t.threshold).ln() / t.rate } else { f64::INFINITY };
        let mut length = most.min(ahead);
        if let (Course::Oscillation { phase, .. }, true) = (&mut course, wander > 0.0) {
            *phase += (wander * most).sqrt() * walk.normals()[0];
        }
        place(&course, &mut t);
        let (taken, jumps) = loop {
            if !(t.time + length > t.time) {
                break (None, true);
            }
            let slack = if bracket.1 - bracket.0 <= timing * length { f64::INFINITY } else { timing * length };
            let reach = length.min(most - length);
            match Midway::new(run, &t, length).and_then(|through| through.attempt(&t, &Brownian::STILL, slack, reach, &mut scratch)) {
                Ok(Attempt::Whole(step)) => break (Some(step), false),
                Ok(Attempt::Reached(step)) => break (Some(step), true),
                Ok(Attempt::Proposes { length: proposed, .. }) => {
                    if proposed < length {
                        bracket.1 = length.min(bracket.1);
                    } else {
                        bracket.0 = length.max(bracket.0);
                    }
                    length = if proposed > bracket.0 && proposed < bracket.1 || !(t.time + proposed > t.time) { proposed } else { 0.5 * (bracket.0 + bracket.1.min(most)) };
                }
                Err(e) => {
                    length *= 0.5;
                    assert!(t.time + length > t.time, "refused at every length: {e:?}");
                    (most, bracket) = (length, (0.0, f64::INFINITY));
                }
            }
        };
        if let Some(step) = taken {
            let s = &step.segment;
            if t.time >= burn {
                let kick = s.kicked.1[2] - s.flow.advance(s.from, 0.5 * s.tau).1[2];
                let spread = 2.0 * step.heating[2] / hm;
                let w = course.weights(t.time + 0.5 * s.tau);
                for (i, w) in [1.0, w[0], w[1]].into_iter().enumerate() {
                    sums[3 * i] += w * kick;
                    sums[3 * i + 2] += w * w * spread;
                }
            }
            let fall = (norm / step.end.psi.iter().map(|z| z.norm_sqr()).sum::<f64>()).ln();
            t.psi = step.end.psi;
            t.time = step.end.time;
            t.rate = fall / s.tau;
            if !jumps && length == rest {
                t.time = next;
            }
        }
        if jumps {
            place(&course, &mut t);
            let before = t.kappa[2];
            atom_motion::step::jump(run, &mut t).expect("a jump at the threshold");
            t.lost = None;
            if t.time >= burn {
                let w = course.weights(t.time);
                for (i, w) in [1.0, w[0], w[1]].into_iter().enumerate() {
                    sums[3 * i + 1] += w * (t.kappa[2] - before);
                }
                sums[9] += 1.0;
            }
        }
        if t.time >= next {
            sums[10] = run.cluster(t.cluster).states.iter().zip(t.psi.iter()).filter(|(&s, _)| (s < 3) != (start < 3)).map(|(_, z)| z.norm_sqr()).sum::<f64>() / t.psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
            out.push(sums);
            let size = t.psi.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            states.push((t.cluster, t.psi.iter().map(|z| *z * (1.0 / size)).collect()));
            next = burn + out.len() as f64 * every;
        }
    }
    (out, states)
}

/// `engine`: trajectories of the gates' closed model held on a course. `course`: "rest:<z>", "const:<v>" (the start
/// uniform over the light's period) or "osc:<amplitude>:<omega>:<centre>" (the phase uniform). The run takes its near
/// tables over the scene's reach, as the solve does (`TABLES=0`: the exact elimination at every point). Writes every
/// trajectory's samples.
fn engine(name: &str, waist: f64, course: &str, tau: f64, count: u64, burn: f64, span: f64, every: f64, out: &str) {
    use rayon::prelude::*;
    let request = scene(name, waist);
    let light = Light::new(&request);
    let (start, p) = reference(&request);
    let model = Model::new(&request, &light, &closure(name)).unwrap();
    let tables = (std::env::var("TABLES").as_deref() != Ok("0")).then(|| {
        // The scene's own reach: of another waist's trap the scene's cell (the course stays on the axis within it) with
        // that trap's own energies.
        let own = scene(name, 0.0);
        let own_light = Light::new(&own);
        let own_model = Model::new(&own, &own_light, &closure(name)).unwrap();
        let share = atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64);
        let cell = own_model.reach(&own_model.trap(&p, &own_model.prepared()).unwrap(), &p, own.temperature_uk, share).cell;
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = if waist == 0.0 { model.reach(&trap, &p, request.temperature_uk, share) } else { atom_motion::model::Reach { cell, ..own_model.reach(&own_model.trap(&p, &own_model.prepared()).unwrap(), &p, own.temperature_uk, share) } };
        let reach = atom_motion::model::Reach { minimum: trap.minimum, ..reach };
        let clock = std::time::Instant::now();
        let tables = model.tabulate(&reach, 1 << 20);
        eprintln!("reach cell {:?} speed {}; {} tables in {:.1?}", reach.cell, reach.speed, tables.tables.len(), clock.elapsed());
        tables
    });
    let run = match &tables {
        Some(tables) => Run::tabulated(&model, tables, p),
        None => Run::new(&model, p),
    };
    let period = std::f64::consts::TAU / request.beams.iter().find(|b| b.id.starts_with("cool")).expect("a cooling beam").wave_number;
    let parts: Vec<&str> = course.split(':').collect();
    let number = |i: usize| -> f64 { parts[i].parse().unwrap() };
    let clock = std::time::Instant::now();
    // `PHASES=<n>`: the start phases on a grid of n over the period (the states' samples then sit on the reference's own
    // points of the path); `STATES=<file>`: every sample's internal state, [trajectory's phase/2π, cluster, re, im, ...].
    let grid: Option<f64> = std::env::var("PHASES").ok().map(|n| n.parse().unwrap());
    let both: Vec<(Vec<[f64; WIDTH]>, Vec<(usize, Vec<C64>)>, f64)> = (0..count)
        .into_par_iter()
        .map(|i| {
            let u = atom_motion::random::Stream::new(41, atom_motion::random::Purpose::Initial, i, 0).uniform();
            let u = grid.map_or(u, |n| (u * n).floor() / n);
            let course = match parts[0] {
                "rest" => Course::Rest { z: number(1) },
                "const" => Course::Constant { z: (u - 0.5) * period, v: number(1), period },
                "osc" => Course::Oscillation { centre: number(3), amplitude: number(1), omega: number(2), phase: std::f64::consts::TAU * u },
                other => panic!("unknown course {other}"),
            };
            let (samples, states) = held(&run, start, course, tau, burn, span, every, 43, i);
            (samples, states, u)
        })
        .collect();
    eprintln!("{name} {course}: {count} trajectories × {} µs at τ {tau} in {:.1?}", burn + span, clock.elapsed());
    // The samples every trajectory has (the last one falls on the span's end or a rounding past it).
    let common = both.iter().map(|(samples, _, _)| samples.len()).min().unwrap();
    let both: Vec<(Vec<[f64; WIDTH]>, Vec<(usize, Vec<C64>)>, f64)> = both.into_iter().map(|(mut samples, mut states, u)| { samples.truncate(common); states.truncate(common); (samples, states, u) }).collect();
    if let Ok(file) = std::env::var("STATES") {
        let rows: Vec<serde_json::Value> = both.iter().map(|(_, states, u)| serde_json::json!({"phase": u, "states": states.iter().map(|(c, psi)| serde_json::json!([c, psi.iter().flat_map(|z| [z.re, z.im]).collect::<Vec<f64>>()])).collect::<Vec<_>>()})).collect();
        std::fs::write(file, serde_json::to_string(&rows).unwrap()).unwrap();
    }
    // `ENDS=1`: the first and the last sample alone (the rates over the whole span need no more).
    let ends = std::env::var("ENDS").as_deref() == Ok("1");
    let all: Vec<Vec<[f64; WIDTH]>> = both.into_iter().map(|(samples, _, _)| if ends { vec![samples[0], samples[samples.len() - 1]] } else { samples }).collect();
    let flat: Vec<f64> = all.iter().flat_map(|t| t.iter().flat_map(|s| s.iter().copied())).collect();
    std::fs::write(out, serde_json::to_string(&serde_json::json!({"scene": name, "waist": waist, "course": course, "tau": tau, "count": count, "burn": burn, "span": span, "every": every, "samples": all[0].len(), "width": WIDTH, "hbar_over_mass": model.hbar_over_mass(), "data": flat})).unwrap()).unwrap();
}

/// `free`: the engine's own dynamics (`adapt::advance` at the constant step τ, its Brownian path with it) of atoms that
/// start in the reference state with the Gaussian positions and momenta of `temperature` µK in the trap's harmonic
/// approximation. `dimension`: "1d", z alone -- x and y and their momenta put back after every step: the 1D problem the
/// quantum reference and the phase-space equation solve --; "3d", the three axes; "free:yz", the axes named, the others
/// held; with ",cold:y" a free axis starts at the centre at rest. The held axes sit at the trap's centre, or at
/// `OFFSET=x,y,z` µm from it, or (`SWING=1`) swing on the trap's own oscillation of that amplitude. `TABLES=0`: the
/// exact elimination at every point. Per window of `every` µs: the mean over the atoms of κ_z² averaged over the
/// window's steps' ends, its standard error, the means of κ_z, z², the population outside the start's hyperfine level,
/// the atoms lost, κ_x² and κ_y².
fn free(name: &str, dimension: &str, temperature: f64, tau: f64, count: u64, span: f64, every: f64, out: &str) {
    use rayon::prelude::*;
    let request = scene(name, 0.0);
    let light = Light::new(&request);
    let (start, p) = reference(&request);
    let model = Model::new(&request, &light, &closure(name)).unwrap();
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let reach = model.reach(&trap, &p, request.temperature_uk, atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64));
    let tables = model.tabulate(&reach, 1 << 20);
    let run = if std::env::var("TABLES").as_deref() == Ok("0") { Run::new(&model, p) } else { Run::tabulated(&model, &tables, p) };
    let (hm, kt) = (run.hbar_over_mass(), atom_motion::units::KT_PER_UK * temperature);
    let (windows, per) = ((span / every).round() as usize, (every / tau).round() as usize);
    // The axes held at the trap's centre: "1d" x and y, "3d" none, "free:<axes>" the lab axes not named (free:yz holds x).
    let held: Vec<usize> = match dimension {
        "1d" => vec![0, 1],
        "3d" => vec![],
        other => (0..3).filter(|&a| !other.split(',').next().unwrap().trim_start_matches("free:").contains(["x", "y", "z"][a])).collect(),
    };
    // The free axes that start at the trap's centre at rest ("free:yz,cold:y"): free, without their thermal motion.
    let cold: Vec<usize> = (0..3).filter(|&a| dimension.split(',').any(|part| part.starts_with("cold:") && part.contains(["x", "y", "z"][a]))).collect();
    // Where the held axes are held, from the trap's centre (µm; `OFFSET=x,y,z`, none: at the centre).
    let offset: [f64; 3] = std::env::var("OFFSET").map_or([0.0; 3], |s| {
        let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
        [v[0], v[1], v[2]]
    });
    let swinging = std::env::var("SWING").as_deref() == Ok("1");
    let clock = std::time::Instant::now();
    let all: Vec<Vec<[f64; 7]>> = (0..count)
        .into_par_iter()
        .map(|i| {
            let mut stream = atom_motion::random::Stream::new(47, atom_motion::random::Purpose::Initial, i, 0);
            let (mut r, mut kappa) = (trap.centre, [0.0; 3]);
            let (places, momenta) = (stream.normals(), stream.normals());
            for axis in 0..3 {
                let (x, k) = (places[axis] * (kt * hm).sqrt() / trap.lab[axis], momenta[axis] * (kt / hm).sqrt());
                if !held.contains(&axis) && !cold.contains(&axis) {
                    r[axis] += x;
                    kappa[axis] += k;
                }
            }
            let mut phases = atom_motion::random::Stream::new(53, atom_motion::random::Purpose::Initial, i, 0);
            let swing: [f64; 3] = std::array::from_fn(|_| std::f64::consts::TAU * phases.uniform());
            for &axis in &held {
                r[axis] += offset[axis] * if swinging { swing[axis].cos() } else { 1.0 };
                kappa[axis] = if swinging { -offset[axis] * trap.lab[axis] * swing[axis].sin() / hm } else { 0.0 };
            }
            let mut t = Trajectory::new(&run, start, r, kappa, 47, i);
            let mut path = atom_motion::adapt::Path::new(47, i, tau, 0);
            let mut scratch = Scratch::default();
            let mut rows = vec![];
            for w in 0..windows {
                let mut sums = [0.0; 7];
                for s in 0..per {
                    let to = (w * per + s + 1) as f64 * tau;
                    atom_motion::adapt::advance(&run, tau, &mut t, &mut scratch, &mut path, to, &[], &mut |_, _| {}, &mut |_, _| {}).unwrap();
                    for &axis in &held {
                        // Held at its offset; or, `SWING=1`, on the trap's own oscillation of that amplitude along the
                        // axis, each atom at its own phase: the axis moves as a trapped atom's does and takes nothing
                        // from the light or from z.
                        let phase = trap.lab[axis] * to + swing[axis];
                        (t.r[axis], t.kappa[axis]) = if swinging { (trap.centre[axis] + offset[axis] * phase.cos(), -offset[axis] * trap.lab[axis] * phase.sin() / hm) } else { (trap.centre[axis] + offset[axis], 0.0) };
                    }
                    let other = run.cluster(t.cluster).states.iter().zip(t.psi.iter()).filter(|(&s, _)| (s < 3) != (start < 3)).map(|(_, z)| z.norm_sqr()).sum::<f64>() / t.psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
                    for (sum, x) in sums.iter_mut().zip([t.kappa[2] * t.kappa[2], t.kappa[2], t.r[2] * t.r[2], other, if t.lost.is_some() { 1.0 } else { 0.0 }, t.kappa[0] * t.kappa[0], t.kappa[1] * t.kappa[1]]) {
                        *sum += x / per as f64;
                    }
                }
                rows.push(sums);
            }
            rows
        })
        .collect();
    eprintln!("{name} free {dimension}: {count} atoms × {span} µs at τ {tau} in {:.1?}", clock.elapsed());
    let n = count as f64;
    let rows: Vec<serde_json::Value> = (0..windows)
        .map(|w| {
            let mean = |c: usize| all.iter().map(|a| a[w][c]).sum::<f64>() / n;
            let m = mean(0);
            let se = (all.iter().map(|a| (a[w][0] - m).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
            serde_json::json!([(w + 1) as f64 * every, m, se, mean(1), mean(2), mean(3), mean(4), mean(5), mean(6)])
        })
        .collect();
    std::fs::write(out, serde_json::to_string(&serde_json::json!({"scene": name, "dimension": dimension, "temperature": temperature, "tau": tau, "count": count, "hbar_over_mass": hm, "omega": trap.lab[2], "rows": rows})).unwrap()).unwrap();
}

/// The parts a kinetic energy's change is accounted by (`account`): the kick's Ehrenfest impulse beyond the reference's
/// force, the scattered light's continuous force, the Brownian part, the jumps' kicks, and the reference's flow.
const PARTS: usize = 5;

/// `account`: `free`'s atoms on the same dynamics, stepped here (the step's middle, its attempt for the jump and the
/// jump, as `adapt::advance`; each step's part of a Brownian path of its own, independent increments) so that every
/// change of an axis's kinetic energy (ħ/m)κ_a²/2 is booked to the part of the engine that made it. A kick Δκ at the
/// middle momentum κ_h changes it by (ħ/m)(κ_h + Δκ/2)·Δκ, shared between the kick's parts in proportion: the Brownian
/// part Λ ΔW (the step's own factor on its increment), the scattered light's force τF_s (of the state at the step's
/// start), and the rest, the Ehrenfest impulse less the reference's force; a jump by its kick; the reference's two half
/// flows by the kinetic energy they turn into potential energy or into another axis's.
/// Per window of `every` µs, means over the atoms: per axis the kinetic energy averaged over the window's steps' ends
/// and the cumulative work of each part (rad/µs); the squared angular momentum about x, (y κ_z − z κ_y)²; the
/// cumulative Σ Δκ_a Δκ_b of the jumps and ∫(Λ Λᵀ)_ab dt for ab = yy, zz, yz; the jumps; the standard errors of the
/// kinetic energies; and per atom its run's mean kinetic energies along y and z with the light's and the flow's work on
/// each, and its start energies along y and z in the trap's harmonic approximation (`atoms`). `TEMPS=x,y,z`: the axes' own start temperatures (µK).
fn account(name: &str, dimension: &str, temperature: f64, tau: f64, count: u64, span: f64, every: f64, out: &str) {
    use rayon::prelude::*;
    let request = scene(name, 0.0);
    let light = Light::new(&request);
    let (start, p) = reference(&request);
    let model = Model::new(&request, &light, &closure(name)).unwrap();
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let reach = model.reach(&trap, &p, request.temperature_uk, atom_motion::estimate::unseen(atom_motion::units::MOST_TRAJECTORIES as f64));
    let tables = model.tabulate(&reach, 1 << 20);
    let run = if std::env::var("TABLES").as_deref() == Ok("0") { Run::new(&model, p) } else { Run::tabulated(&model, &tables, p) };
    let hm = run.hbar_over_mass();
    let temperatures: [f64; 3] = std::env::var("TEMPS").map_or([temperature; 3], |s| {
        let v: Vec<f64> = s.split(',').map(|x| x.parse().unwrap()).collect();
        [v[0], v[1], v[2]]
    });
    let held: Vec<usize> = match dimension {
        "1d" => vec![0, 1],
        "3d" => vec![],
        other => (0..3).filter(|&a| !other.trim_start_matches("free:").contains(["x", "y", "z"][a])).collect(),
    };
    let (windows, per) = ((span / every).round() as usize, (every / tau).round() as usize);
    const WIDE: usize = 3 + 3 * PARTS + 1 + 6 + 1;
    let timing = f64::EPSILON.sqrt();
    let clock = std::time::Instant::now();
    let both: Vec<(Vec<[f64; WIDE]>, [f64; 8])> = (0..count)
        .into_par_iter()
        .map(|i| {
            let mut stream = atom_motion::random::Stream::new(47, atom_motion::random::Purpose::Initial, i, 0);
            let (mut r, mut kappa) = (trap.centre, [0.0; 3]);
            let (places, momenta) = (stream.normals(), stream.normals());
            for axis in 0..3 {
                if !held.contains(&axis) {
                    let kt = atom_motion::units::KT_PER_UK * temperatures[axis];
                    r[axis] += places[axis] * (kt * hm).sqrt() / trap.lab[axis];
                    kappa[axis] += momenta[axis] * (kt / hm).sqrt();
                }
            }
            // The axes' start energies in the trap's harmonic approximation (rad/µs).
            let begun: [f64; 3] = std::array::from_fn(|a| 0.5 * hm * kappa[a] * kappa[a] + 0.5 * trap.lab[a].powi(2) * (r[a] - trap.centre[a]).powi(2) / hm);
            let mut t = Trajectory::new(&run, start, r, kappa, 47, i);
            let mut noise = atom_motion::random::Stream::new(61, atom_motion::random::Purpose::Initial, i, 0);
            let mut scratch = Scratch::default();
            let (mut work, mut moments, mut jumps) = ([[0.0; PARTS]; 3], [0.0; 6], 0.0);
            let mut rows = vec![];
            for w in 0..windows {
                let (mut kinetic, mut turning) = ([0.0; 3], 0.0);
                for s in 0..per {
                    let to = (w * per + s + 1) as f64 * tau;
                    while t.time < to {
                        let rest = to - t.time;
                        let mut most = rest;
                        let mut bracket = (0.0, f64::INFINITY);
                        let norm: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
                        let ahead = if t.rate > 0.0 { (norm / t.threshold).ln() / t.rate } else { f64::INFINITY };
                        let mut length = most.min(ahead);
                        let (taken, jumped) = loop {
                            if !(t.time + length > t.time) {
                                break (None, true);
                            }
                            let slack = if bracket.1 - bracket.0 <= timing * length { f64::INFINITY } else { timing * length };
                            let reach = length.min(most - length);
                            let brownian = Brownian::over(length, &[noise.normals(), noise.normals()]);
                            match Midway::new(&run, &t, length).and_then(|through| through.attempt(&t, &brownian, slack, reach, &mut scratch)) {
                                Ok(Attempt::Whole(step)) => break (Some((step, brownian)), false),
                                Ok(Attempt::Reached(step)) => break (Some((step, brownian)), true),
                                Ok(Attempt::Proposes { length: proposed, .. }) => {
                                    if proposed < length {
                                        bracket.1 = length.min(bracket.1);
                                    } else {
                                        bracket.0 = length.max(bracket.0);
                                    }
                                    length = if proposed > bracket.0 && proposed < bracket.1 || !(t.time + proposed > t.time) { proposed } else { 0.5 * (bracket.0 + bracket.1.min(most)) };
                                }
                                Err(e) => {
                                    length *= 0.5;
                                    assert!(t.time + length > t.time, "refused at every length: {e:?}");
                                    (most, bracket) = (length, (0.0, f64::INFINITY));
                                }
                            }
                        };
                        if let Some((step, brownian)) = taken {
                            let g = &step.segment;
                            let (r_h, k_h) = g.flow.advance(g.from, 0.5 * g.tau);
                            let scattered = run.scattering_at(t.cluster, &Point { r: r_h, kappa: k_h, t: t.time + 0.5 * g.tau }, &t.psi, None).force;
                            for a in 0..3 {
                                let kick = g.kicked.1[a] - k_h[a];
                                let random: f64 = (0..3).map(|c| step.spread[a][c] * brownian.increment[c]).sum();
                                let force = g.tau * scattered[a];
                                let middle = hm * (k_h[a] + 0.5 * kick);
                                work[a][0] += middle * (kick - random - force);
                                work[a][1] += middle * force;
                                work[a][2] += middle * random;
                                work[a][4] += 0.5 * hm * (step.end.kappa[a].powi(2) - g.kicked.1[a].powi(2) + k_h[a].powi(2) - g.from.1[a].powi(2));
                            }
                            for (m, (a, b)) in [(1, 1), (2, 2), (1, 2)].into_iter().enumerate() {
                                moments[3 + m] += g.tau * (0..3).map(|c| step.spread[a][c] * step.spread[b][c]).sum::<f64>();
                            }
                            let fall = (norm / step.end.psi.iter().map(|z| z.norm_sqr()).sum::<f64>()).ln();
                            (t.r, t.kappa, t.psi, t.time) = (step.end.r, step.end.kappa, step.end.psi, step.end.time);
                            t.rate = fall / g.tau;
                            if !jumped && length == rest {
                                t.time = to;
                            }
                        }
                        if jumped {
                            let before = t.kappa;
                            atom_motion::step::jump(&run, &mut t).expect("a jump at the threshold");
                            t.lost = None;
                            for a in 0..3 {
                                work[a][3] += 0.5 * hm * (t.kappa[a].powi(2) - before[a].powi(2));
                            }
                            for (m, (a, b)) in [(1, 1), (2, 2), (1, 2)].into_iter().enumerate() {
                                moments[m] += (t.kappa[a] - before[a]) * (t.kappa[b] - before[b]);
                            }
                            jumps += 1.0;
                        }
                    }
                    for &axis in &held {
                        (t.r[axis], t.kappa[axis]) = (trap.centre[axis], 0.0);
                    }
                    for a in 0..3 {
                        kinetic[a] += 0.5 * hm * t.kappa[a] * t.kappa[a] / per as f64;
                    }
                    turning += (t.r[1] * t.kappa[2] - t.r[2] * t.kappa[1]).powi(2) / per as f64;
                }
                let mut row = [0.0; WIDE];
                row[..3].copy_from_slice(&kinetic);
                for a in 0..3 {
                    row[3 + a * PARTS..3 + (a + 1) * PARTS].copy_from_slice(&work[a]);
                }
                row[3 + 3 * PARTS] = turning;
                row[4 + 3 * PARTS..10 + 3 * PARTS].copy_from_slice(&moments);
                row[10 + 3 * PARTS] = jumps;
                rows.push(row);
            }
            // The atom's own account: its kinetic energies along y and z averaged over the run, and the light's and
            // the flow's work on each of the two.
            let mean = |a: usize| rows.iter().map(|r| r[a]).sum::<f64>() / rows.len() as f64;
            let light = |a: usize| work[a][..4].iter().sum::<f64>();
            let atom = [mean(1), mean(2), light(1), work[1][4], light(2), work[2][4], begun[1], begun[2]];
            (rows, atom)
        })
        .collect();
    eprintln!("{name} account {dimension}: {count} atoms × {span} µs at τ {tau} in {:.1?}", clock.elapsed());
    let atoms: Vec<[f64; 8]> = both.iter().map(|(_, atom)| *atom).collect();
    let all: Vec<Vec<[f64; WIDE]>> = both.into_iter().map(|(rows, _)| rows).collect();
    let n = count as f64;
    let rows: Vec<serde_json::Value> = (0..windows)
        .map(|w| {
            let means: Vec<f64> = (0..WIDE).map(|c| all.iter().map(|a| a[w][c]).sum::<f64>() / n).collect();
            let errors: Vec<f64> = (0..3).map(|c| (all.iter().map(|a| (a[w][c] - means[c]).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt()).collect();
            serde_json::json!({"t": (w + 1) as f64 * every, "means": means, "errors": errors})
        })
        .collect();
    std::fs::write(out, serde_json::to_string(&serde_json::json!({"scene": name, "dimension": dimension, "temperatures": temperatures, "tau": tau, "count": count, "hbar_over_mass": hm, "omega": trap.lab, "parts": PARTS, "rows": rows, "atoms": atoms})).unwrap()).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let number = |i: usize| -> f64 { args[i].parse().unwrap() };
    match args[1].as_str() {
        "info" => info(&args[2], args.get(3).map_or(0.0, |w| w.parse().unwrap())),
        "dump" => dump(&args[2], number(3), &args[4], &args[5], &args[6]),
        "engine" => engine(&args[2], number(3), &args[4], number(5), args[6].parse().unwrap(), number(7), number(8), number(9), &args[10]),
        "free" => free(&args[2], &args[3], number(4), number(5), args[6].parse().unwrap(), number(7), number(8), &args[9]),
        "account" => account(&args[2], &args[3], number(4), number(5), args[6].parse().unwrap(), number(7), number(8), &args[9]),
        other => panic!("unknown mode {other}"),
    }
}
