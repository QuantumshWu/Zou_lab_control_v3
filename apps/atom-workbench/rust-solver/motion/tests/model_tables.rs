//! The near light's tables (model/tabulate.rs, SPEC §2.3) against the exact elimination: across an axis that carries a line through
//! resonance (R3_DOPPLER's cooling pair in its tweezer), to the interpolation's orders (R2_RECOIL's probe), their error reads; and,
//! run by name alone, the report of a scene's tables against its exact elimination over its reach and the timing of the model's
//! calls (`SCENE`: a reference scene's name or a request file's path).
use atom_motion::field::Light;
use atom_motion::linalg::{norm_sqr, C64};
use atom_motion::model::tabulate::{Coordinate, Tables};
use atom_motion::model::{Lights, Lit, Model, Photon, Point, Reach, Run, Sample, Trap, Unit, UnitError, GROUND};
use atom_motion::random::{Purpose, Stream};
use atom_motion::request::Request;
use atom_motion::step::{Dynamics, Middle};
use atom_motion::units::{CONFIDENCE, KT_PER_UK, MOST_TRAJECTORIES};
use std::time::Instant;

mod model_common;
use model_common::{normal, scene};

/// The populations of a ground state alone.
fn alone(state: usize) -> [f64; GROUND] {
    std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 })
}

/// The share the largest ensemble leaves unseen (estimate.rs `unseen`): the reach's.
fn unseen() -> f64 {
    atom_motion::estimate::unseen(MOST_TRAJECTORIES as f64)
}

/// The scene's model as the solve closes it (solve.rs `attempt`): every unit whose fold does not exist over the reach at
/// `temperature` kept, until none fails; with the units kept, the reference state, its trap and that reach.
fn closed(request: &Request, light: &Light, temperature: f64) -> (Model, Vec<Unit>, usize, Trap, Reach) {
    let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = alone(state);
    let mut kept: Vec<Unit> = vec![];
    loop {
        let model = Model::new(request, light, &kept).unwrap();
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, temperature, unseen());
        let units = model.units();
        let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            return (model, kept, state, trap, reach);
        }
        kept.extend(failing);
    }
}

/// The tables' shares of the targets (tabulate.rs, # Refinement): z/(2√N) on what is shown, z√(2/N) on T.
fn shares() -> (f64, f64) {
    let n = MOST_TRAJECTORIES as f64;
    (CONFIDENCE / (2.0 * n.sqrt()), CONFIDENCE * (2.0 / n).sqrt())
}

/// A cluster's photon rate Σ_φ Σ_a ‖u_a‖²/‖ψ‖² of a state at a point.
fn rate(run: &Run, cluster: usize, at: &Point, psi: &[C64]) -> f64 {
    run.photons(cluster, at, psi).iter().flat_map(|p| p.amplitudes.iter()).map(|u| norm_sqr(u)).sum::<f64>() / norm_sqr(psi)
}

/// A cluster's generator at a point as a step meets it (step.rs `Middle`): on the path at the point's own velocity, its
/// matrix at three times about the middle, and there its gradient's forms with the density of a state.
fn nodes(model: &Model, middle: &impl Middle, at: &Point, psi: &[C64]) -> (Vec<C64>, Vec<f64>) {
    let n = psi.len();
    let path = middle.path(at.kappa, at.kappa.map(|k| model.hbar_over_mass() * k));
    let rho: Vec<C64> = (0..n * n).map(|i| psi[i / n] * psi[i % n].conj() / norm_sqr(psi)).collect();
    let (mut matrices, mut forms) = (vec![], vec![]);
    let mut h = atom_motion::linalg::Mat::zeros(n, n);
    for s in [-0.2, 0.0, 0.3] {
        middle.generator(&path, s, &mut h);
        matrices.extend_from_slice(&h.data);
        forms.extend(middle.force(&path, s, &rho));
    }
    (matrices, forms)
}

/// The distances of two generators' nodes, the matrices' (Frobenius) and the forms'.
fn distance(a: &(Vec<C64>, Vec<f64>), b: &(Vec<C64>, Vec<f64>)) -> (f64, f64) {
    (a.0.iter().zip(&b.0).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>().sqrt(), a.1.iter().zip(&b.1).map(|(x, y)| (x - y) * (x - y)).sum::<f64>().sqrt())
}

/// R3_DOPPLER's tweezer carries its cooling line through resonance within reach of atoms that have heated to 100 µK (the line
/// is 3 half-widths red at the bottom and on resonance 381 µK up the trap's wall: a pole of the responses in the tweezer's
/// intensity, the tables' axis). Across the tweezer from its axis to the reach's edge, at 240 places between the nodes, the
/// tabulated model is the exact one within the tables' shares (tabulate.rs, # Refinement): the photon rate of an atom moving
/// along the cooling pair, the error of its generator as the ledger reads it (`Model::table_reads`: δ on what is shown, on T
/// with the potential's), and the generator on the atom's path and its force there to the same size -- the matrices' error of
/// the near light's own share at the place, the forces' of the near light's largest over the sweep (its force crosses zero,
/// where no error is small beside it). The sweep does cross the resonance: the exact rate peaks at several times its value on the
/// axis. (The first grid's seven nodes on the axis, as the tables were first made, missed that rate by −13 % to +77 %.)
#[test]
fn a_table_follows_a_line_through_resonance_across_its_axis() {
    let request = scene("R3_DOPPLER");
    let light = Light::new(&request);
    let (model, kept, state, trap, reach) = closed(&request, &light, 100.0);
    let p = alone(state);
    let tables = model.tabulate(&reach, 1 << 20);
    let (exact, fast) = (Run::new(&model, p), Run::tabulated(&model, &tables, p));
    let cluster = model.cluster_of(state);
    let index = tables.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).expect("a table for the running lights");
    let table = &tables.tables[index];
    assert_eq!(table.axes(), 1, "{:?}", table.layout.coordinates);
    assert!(matches!(table.layout.coordinates[0], Coordinate::Chart { index: 0, .. }), "the tweezer's intensity: {:?}", table.layout.coordinates[0]);
    // The light with a drive folded at the atom off: what the exact generator is compared with for that light's own size.
    let dark = Lights(model.running().0.iter().enumerate().map(|(f, l)| if kept.iter().any(|u| matches!(u, Unit::Bare { component, .. } if component.frequency == f)) { Lit::Off } else { *l }).collect());
    let prepared = model.prepared();
    let edge = (1..).map(|i| 0.01 * i as f64).find(|&rho| model.potential([trap.centre[0] + rho, trap.centre[1], trap.centre[2]], &p, &prepared) > reach.minimum + reach.energy).unwrap();
    let momentum = 0.5 * reach.speed / model.hbar_over_mass();
    let places = 240;
    let samples: Vec<Sample> = (0..places)
        .map(|i| {
            let rho = edge * (i as f64 + 0.37) / places as f64;
            Sample { cluster, at: Point { r: [trap.centre[0] + rho, trap.centre[1], trap.centre[2] + 0.11], kappa: [0.0, 0.0, momentum], t: 0.0 }, psi: vec![C64::ONE] }
        })
        .collect();
    let (shown, temperature) = shares();
    let reads = &model.table_reads(&tables, &samples)[index];
    let (mut worst, mut rates, mut force) = ([0.0_f64; 4], vec![], [0.0_f64; 2]);
    for (sample, read) in samples.iter().zip(&reads.whole) {
        let (at, psi) = (&sample.at, &sample.psi);
        let (r, t) = (rate(&exact, cluster, at, psi), rate(&fast, cluster, at, psi));
        rates.push(r);
        let (e, f) = (nodes(&model, &exact.middle(cluster, at.r, at.t), at, psi), nodes(&model, &fast.middle(cluster, at.r, at.t), at, psi));
        let far = nodes(&model, &model.assemble(cluster, at.r, at.t, &dark), at, psi);
        let (error, near) = (distance(&e, &f), distance(&e, &far));
        worst = [worst[0].max((t - r).abs() / r), worst[1].max(read.population), worst[2].max(read.temperature), worst[3].max(error.0 / near.0)];
        force = [force[0].max(error.1), force[1].max(near.1)];
    }
    let (peak, forces) = (rates.iter().copied().fold(0.0, f64::max), force[0] / force[1]);
    eprintln!("R3_DOPPLER over {edge:.2} µm, {} nodes: rate {:.4} on the axis, {peak:.4} at its peak; largest errors: rate {:.2e}, shown {:.2e}, T {:.2e}, parts {:.2e}, forces {forces:.2e}", table.table.axes()[0].nodes, rates[0], worst[0], worst[1], worst[2], worst[3]);
    assert!(peak > 5.0 * rates[0], "the sweep crosses the resonance: {peak} against {} on the axis", rates[0]);
    assert!(worst[0] <= shown && worst[1] <= shown && worst[2] <= temperature, "rate {:.2e}, shown {:.2e} beyond {shown:.2e}, or T {:.2e} beyond {temperature:.2e}", worst[0], worst[1], worst[2]);
    assert!(worst[3] <= shown && forces <= temperature, "the near light's share of the generator {:.2e} and forces {forces:.2e}", worst[3]);
}

/// Light whose intensity is the same everywhere is no axis, on whatever cells its reach is cut (tabulate.rs, # Axes): the
/// lin-perp-lin scene's lights of two counter-propagating plane waves of orthogonal polarizations, their dressings kept --
/// each on its chart, as the solve's pilot keeps the cooling light's at four times the base intensity --, over a reach grown
/// to 75 µK (its level, 12.89 k_B T, still below the trap's far value: a bounded reach). Each chart's intensity is the same at every point, its weight nothing on the reach's first cell: the tables
/// are made, and no axis of theirs is an intensity but the tweezer's. (Bounded by the field's modulus along the two waves'
/// phases, that weight's proof cut 8 million cells in 43 s and did not end.)
#[test]
fn a_light_of_one_intensity_everywhere_is_no_axis() {
    let request = scene("R4_LIN_4");
    let light = Light::new(&request);
    let (folded, mut kept, state, ..) = closed(&request, &light, request.temperature_uk);
    let dressings: Vec<Unit> = folded.units().into_iter().filter(|u| matches!(u, Unit::Dressing { .. })).collect();
    assert!(!dressings.is_empty(), "the scene's carriers have dressings to keep");
    kept.extend(dressings);
    let model = Model::new(&request, &light, &kept).unwrap();
    let p = alone(state);
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let reach = model.reach(&trap, &p, 75.0, unseen());
    let tables = model.tabulate(&reach, 1 << 16);
    let axes: Vec<Coordinate> = tables.tables.iter().flat_map(|t| t.layout.coordinates[..t.axes()].iter().copied()).collect();
    assert!(axes.iter().any(|c| matches!(c, Coordinate::Chart { index, .. } if *index > 0)), "a kept dressing's light is on its chart: {axes:?}");
    for c in &axes {
        assert!(!matches!(c, Coordinate::Chart { frequency, index: 0 } if *frequency != light.trap) && !matches!(c, Coordinate::Unit { .. }), "an intensity that is not the tweezer's is an axis: {c:?}");
    }
}

/// R2_RECOIL with its probe near, the atom in |2,2⟩, its reach at the request's temperature and its tables.
fn r2() -> (Model, [f64; GROUND], Tables) {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let beam = request.beams.iter().position(|b| b.id == "probe").unwrap();
    let probe = light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap();
    let model = Model::new(&request, &light, &model_common::at_the_atom(&request, &light, &[probe])).unwrap();
    let p = alone(7);
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let reach = model.reach(&trap, &p, request.temperature_uk, 2e-3);
    let tables = model.tabulate(&reach, 1 << 16);
    (model, p, tables)
}

/// Points through the reach's core: along the tweezer and across it, a few at an angle.
fn points() -> Vec<[f64; 3]> {
    let mut out = vec![];
    for i in 0..9 {
        let u = -0.4 + 0.1 * i as f64;
        out.extend([[u, 0.0, 0.0], [0.0, u * 0.6, 0.0], [0.0, 0.0, u * 0.6], [u * 0.5, u * 0.3, -u * 0.2]]);
    }
    out
}

/// The tweezer's intensity is the one axis (its weight the line's move in half-widths over the reach; the probe's plane wave
/// weighs nothing: its intensity is the same everywhere, and the table's build ends), and the tabulated generator -- its matrix
/// on a path and its force through the chain rule -- is the exact one's to the interpolation's orders: halving the axis's spacing divides
/// the largest value error over the points by about 2³ and the force's by about 2² (within a factor two of each: the points meet
/// the cubic's error shape at different places in its intervals).
#[test]
fn a_table_is_the_elimination_to_its_interpolation_s_orders() {
    let (model, p, mut tables) = r2();
    let cluster = model.cluster_of(7);
    let index = tables.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).expect("a table for the running lights");
    let t = &tables.tables[index];
    assert_eq!(t.axes(), 1, "{:?}", t.layout.coordinates);
    assert!(matches!(t.layout.coordinates[0], Coordinate::Chart { index: 0, .. }), "the tweezer's intensity: {:?}", t.layout.coordinates[0]);
    let exact = Run::new(&model, p);
    let errors = |tables: &Tables| -> (f64, f64) {
        let fast = Run::tabulated(&model, tables, p);
        points()
            .iter()
            .map(|&r| {
                let at = Point { r, kappa: [2.0, -1.0, 0.5], t: 0.0 };
                let (a, b) = (nodes(&model, &exact.middle(cluster, r, 0.0), &at, &[C64::ONE]), nodes(&model, &fast.middle(cluster, r, 0.0), &at, &[C64::ONE]));
                let d = distance(&a, &b);
                (d.0 / norm_sqr(&a.0).sqrt(), d.1 / a.1.iter().map(|x| x * x).sum::<f64>().sqrt())
            })
            .fold((0.0, 0.0), |a, b| (a.0.max(b.0), a.1.max(b.1)))
    };
    let mut levels = vec![errors(&tables)];
    for _ in 0..2 {
        tables.tables[index].table = model.halved(&tables.tables[index], 0, usize::MAX).unwrap();
        levels.push(errors(&tables));
    }
    eprintln!("value and force errors at three spacings: {levels:?}");
    let (coarse, fine) = (levels[1], levels[2]);
    let (pv, pf) = ((coarse.0 / fine.0).log2(), (coarse.1 / fine.1).log2());
    assert!((pv - 3.0).abs() <= 1.0, "value order {pv}: {levels:?}");
    assert!((pf - 2.0).abs() <= 1.0, "force order {pf}: {levels:?}");
    assert!(fine.0 < 1e-6, "the refined table's value error {}", fine.0);
}

/// The table's error reads at samples (`Model::table_reads`), at places spread evenly in the axis' coordinate through whole
/// intervals of its grid (across the tweezer, I = e^{−2ρ²/w₀²}): with no coordinate held at the focus that moves, the whole error
/// is the interpolation's, and the axis's read is of its size -- on average over the samples no less than three quarters of it
/// and no more than four times it (crate::table, # The error read of an axis: inside an axis the value's read is 8/7 of its mean
/// error, more in a bounded axis's end intervals; the generator's error takes the forms' through its own weights).
#[test]
fn a_table_s_axis_read_is_its_whole_error_s_size() {
    let (model, _, tables) = r2();
    let request = scene("R2_RECOIL");
    let waist = request.beams[request.trap_beam].waist.unwrap();
    let cluster = model.cluster_of(7);
    let index = tables.tables.iter().position(|t| t.layout.cluster == cluster && t.layout.lights == model.running()).unwrap();
    let axis = tables.tables[index].table.axes()[0];
    // The last four intervals below the focus' intensity 1.
    let (lo, count) = (1.0 - 4.0 * axis.spacing(), 200);
    let samples: Vec<Sample> = (0..count)
        .map(|i| {
            let intensity = lo + (1.0 - lo) * (i as f64 + 0.5) / count as f64;
            let rho = waist * (0.5 * (1.0 / intensity).ln()).sqrt();
            Sample { cluster, at: Point { r: [rho, 0.0, 0.0], kappa: [1.0, 0.0, 0.0], t: 0.0 }, psi: vec![C64::ONE] }
        })
        .collect();
    let reads = model.table_reads(&tables, &samples);
    let read = &reads[index];
    let mean = |v: &[UnitError]| v.iter().map(|e| e.parameter).sum::<f64>() / v.len() as f64;
    let (axis_read, whole) = (mean(&read.axes[0]), mean(&read.whole));
    eprintln!("{} nodes: mean read {axis_read:.3e}, mean whole error {whole:.3e}", axis.nodes);
    assert!(whole > 0.0 && axis_read >= 0.75 * whole && axis_read <= 4.0 * whole, "read {axis_read} against the whole error {whole}");
}

/// Samples as a scene's atoms have them: places and momenta drawn in the trap's harmonic approximation at `temperature`, a third
/// of them scaled out to energies up to the reach's, each with a random state of its cluster.
fn samples(model: &Model, trap: &Trap, reach: &Reach, temperature: f64, count: usize) -> Vec<Sample> {
    let (hm, kt) = (model.hbar_over_mass(), KT_PER_UK * temperature);
    let mut stream = Stream::new(17, Purpose::Initial, 0, 0);
    let mut out = vec![];
    for cluster in 0..model.clusters().len() {
        for i in 0..count {
            let scale = if i % 3 == 2 { (1.0 + stream.uniform() * (reach.energy / kt - 1.0).max(0.0)).sqrt() } else { 1.0 };
            let along: [f64; 3] = std::array::from_fn(|i| scale * (kt * hm).sqrt() / trap.frequencies[i] * normal(&mut stream));
            let r: [f64; 3] = std::array::from_fn(|a| trap.centre[a] + (0..3).map(|i| trap.axes[i][a] * along[i]).sum::<f64>());
            let kappa: [f64; 3] = std::array::from_fn(|_| scale * (kt / hm).sqrt() * normal(&mut stream));
            let psi = (0..model.clusters()[cluster].states.len()).map(|_| C64::new(normal(&mut stream), normal(&mut stream))).collect();
            out.push(Sample { cluster, at: Point { r, kappa, t: 0.0 }, psi });
        }
    }
    out
}

/// REPORT (run by name: `SCENE=<name or path> cargo test ... --test model_tables a_scene_s_tables_against_its_exact_elimination --
/// --ignored --nocapture`): a scene's tables against its exact elimination at samples over its reach -- the generator on the
/// sample's path (of the near light's own share there) and its forces (of the near light's largest over the samples), the photons (their classes,
/// rates and amplitudes; the absorption's two moments of a fixed polarization summed over the classes by their rates, of k and
/// k² times the rate), the loss energy, the readout, and the ledger's reads.
#[test]
#[ignore]
fn a_scene_s_tables_against_its_exact_elimination() {
    let name = std::env::var("SCENE").expect("SCENE: a reference scene's name or a request file's path");
    let request = scene(&name);
    let light = Light::new(&request);
    let started = Instant::now();
    let (model, kept, state, trap, reach) = closed(&request, &light, request.temperature_uk);
    let closure = started.elapsed();
    let p = alone(state);
    let started = Instant::now();
    let tables = model.tabulate(&reach, 1 << 20);
    eprintln!("{name}: closure {closure:.2?}, {} units kept; clusters {:?}; tabulate {:.2?}, {} tables", kept.len(), model.clusters().iter().map(|c| (c.ground, c.states.len() - c.ground)).collect::<Vec<_>>(), started.elapsed(), tables.tables.len());
    for t in &tables.tables {
        eprintln!("  cluster {} lights {:?}: axes {:?} nodes {:?}, {} reals a node", t.layout.cluster, t.layout.lights.0, &t.layout.coordinates[..t.axes()], t.table.axes().iter().map(|a| a.nodes).collect::<Vec<_>>(), t.table.width());
    }
    let (exact, fast) = (Run::new(&model, p), Run::tabulated(&model, &tables, p));
    let dark = Lights(model.running().0.iter().enumerate().map(|(f, l)| if kept.iter().any(|u| matches!(u, Unit::Bare { component, .. } if component.frequency == f)) { Lit::Off } else { *l }).collect());
    let count: usize = std::env::var("SAMPLES").ok().and_then(|x| x.parse().ok()).unwrap_or(12);
    let samples = samples(&model, &trap, &reach, request.temperature_uk, count);
    let reads = model.table_reads(&tables, &samples);
    // The largest: parts, photon rate, classes' weights, amplitudes, the absorption's mean and second moment, loss energy (µK),
    // readout, the reads' shown and T; the forces' largest error and the near light's largest force.
    let (mut worst, mut force) = ([0.0_f64; 11], [0.0_f64; 2]);
    for (k, sample) in samples.iter().enumerate() {
        let (cluster, at, psi) = (sample.cluster, &sample.at, &sample.psi);
        let (e, f) = (nodes(&model, &exact.middle(cluster, at.r, at.t), at, psi), nodes(&model, &fast.middle(cluster, at.r, at.t), at, psi));
        let far = nodes(&model, &model.assemble(cluster, at.r, at.t, &dark), at, psi);
        let (error, near) = (distance(&e, &f), distance(&e, &far));
        if near.0 > 0.0 {
            worst[0] = worst[0].max(error.0 / near.0);
        }
        force = [force[0].max(error.1), force[1].max(near.1)];
        let (pe, pf) = (exact.photons(cluster, at, psi), fast.photons(cluster, at, psi));
        assert_eq!(pe.iter().map(|x| x.landing).collect::<Vec<_>>(), pf.iter().map(|x| x.landing).collect::<Vec<_>>(), "the classes and their order");
        let weight = |x: &Photon| x.amplitudes.iter().map(|u| norm_sqr(u)).sum::<f64>();
        let total: f64 = pe.iter().map(weight).sum();
        if total > 0.0 {
            worst[2] = worst[2].max((pf.iter().map(weight).sum::<f64>() - total).abs() / total);
            // The classes' sums of the absorbed light's mean and second moment per unit time, exact less tabulated.
            let (mut apart, mut rate, mut mean, mut second) = (0.0, 0.0, [0.0; 3], [0.0; 9]);
            for (x, y) in pe.iter().zip(&pf) {
                worst[3] = worst[3].max((weight(x) - weight(y)).abs() / total);
                apart += (0..3).map(|a| x.amplitudes[a].iter().zip(&y.amplitudes[a]).map(|(u, v)| (*u - *v).norm_sqr()).sum::<f64>()).sum::<f64>();
                let (rx, ry) = (x.moments(), y.moments());
                if rx.weight > 0.0 && ry.weight > 0.0 {
                    rate += rx.weight;
                    (0..3).for_each(|i| mean[i] += (rx.mean[i] - ry.mean[i]) / x.wave_number);
                    (0..9).for_each(|i| second[i] += (rx.second[i / 3][i % 3] - ry.second[i / 3][i % 3]) / x.wave_number.powi(2));
                }
            }
            worst[4] = worst[4].max((apart / total).sqrt());
            if rate > 0.0 {
                worst[5] = worst[5].max(mean.iter().map(|x| x.abs()).fold(0.0, f64::max) / rate);
                worst[6] = worst[6].max(second.iter().map(|x| x.abs()).fold(0.0, f64::max) / rate);
            }
        }
        worst[7] = worst[7].max((exact.energy(cluster, at, psi) - fast.energy(cluster, at, psi)).abs() / KT_PER_UK);
        let (re, rf) = (exact.readout(cluster, at, psi), fast.readout(cluster, at, psi));
        worst[8] = worst[8].max(re.iter().zip(&rf).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max));
        for read in &reads {
            worst[9] = worst[9].max(read.whole[k].population);
            worst[10] = worst[10].max(read.whole[k].temperature);
        }
    }
    // The candidate units' errors at the samples, from the tables against the exact elimination's: the largest difference of a
    // unit's error on T.
    let units = model.units();
    let (slow, quick) = (model.errors(&units, &samples), model.errors_with(&units, &samples, Some(&tables)));
    let apart = slow.iter().zip(&quick).flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (x.temperature - y.temperature).abs())).fold(0.0, f64::max);
    let largest = slow.iter().flat_map(|a| a.iter().map(|x| x.temperature)).fold(0.0, f64::max);
    eprintln!("{name}: {} candidate units' errors at the samples: the largest on T {largest:.2e}; tabulated against exact, the largest difference {apart:.1e}", units.len());
    let (shown, temperature) = shares();
    eprintln!(
        "{name}: {} samples; largest differences, tabulated against exact: generator {:.1e}, forces {:.1e} (of the near light's); photon rate {:.1e}, class weights {:.1e}, amplitudes {:.1e}, absorption's mean {:.1e} k, second moment {:.1e} k²; loss energy {:.1e} µK; readout {:.1e}; ledger reads: shown {:.1e} (share {shown:.1e}), T {:.1e} (share {temperature:.1e})",
        samples.len(), worst[0], if force[1] > 0.0 { force[0] / force[1] } else { 0.0 }, worst[2], worst[3], worst[4], worst[5], worst[6], worst[7], worst[8], worst[9], worst[10]
    );
}

/// TIMING (run by name: `SCENE=<name or path> cargo test --release ... --test model_tables the_model_s_calls_timed -- --ignored
/// --nocapture`): per cluster the model's calls on one thread at the scene's samples, the least of eight batches (µs a call),
/// with the exact elimination and with the tables.
#[test]
#[ignore]
fn the_model_s_calls_timed() {
    let name = std::env::var("SCENE").expect("SCENE: a reference scene's name or a request file's path");
    let request = scene(&name);
    let light = Light::new(&request);
    let (model, _, state, trap, reach) = closed(&request, &light, request.temperature_uk);
    let p = alone(state);
    let started = Instant::now();
    let tables = model.tabulate(&reach, 1 << 20);
    eprintln!("{name}: tabulate {:.2?}, {} tables, nodes {:?}", started.elapsed(), tables.tables.len(), tables.tables.iter().map(|t| t.table.axes().iter().map(|a| a.nodes).collect::<Vec<_>>()).collect::<Vec<_>>());
    let (exact, fast) = (Run::new(&model, p), Run::tabulated(&model, &tables, p));
    let all = samples(&model, &trap, &reach, request.temperature_uk, 16);
    for cluster in 0..model.clusters().len() {
        let own: Vec<&Sample> = all.iter().filter(|s| s.cluster == cluster).collect();
        let time = |call: &dyn Fn(&Sample), rounds: usize| {
            (0..8)
                .map(|_| {
                    let started = Instant::now();
                    for k in 0..rounds {
                        call(own[k % own.len()]);
                    }
                    started.elapsed().as_secs_f64() * 1e6 / rounds as f64
                })
                .fold(f64::INFINITY, f64::min)
        };
        let both = |call: &dyn Fn(&Run, &Sample)| (time(&|s| call(&exact, s), 1), time(&|s| call(&fast, s), 256));
        let middle = both(&|run, s| drop(std::hint::black_box(run.middle(cluster, s.at.r, s.at.t))));
        // A node's calls on the tabulated middles: the path, the generator, the force's form, the tones.
        let n = model.clusters()[cluster].states.len();
        let middles: Vec<_> = own.iter().map(|s| fast.middle(cluster, s.at.r, s.at.t)).collect();
        let velocity = |s: &Sample| s.at.kappa.map(|k| model.hbar_over_mass() * k);
        let paths: Vec<_> = own.iter().zip(&middles).map(|(s, m)| m.path(s.at.kappa, velocity(s))).collect();
        let rhos: Vec<Vec<C64>> = own.iter().map(|s| (0..n * n).map(|i| s.psi[i / n] * s.psi[i % n].conj() / norm_sqr(&s.psi)).collect()).collect();
        let each = |call: &mut dyn FnMut(usize)| {
            (0..8)
                .map(|_| {
                    let started = Instant::now();
                    for k in 0..1024 {
                        call(k % own.len());
                    }
                    started.elapsed().as_secs_f64() * 1e6 / 1024.0
                })
                .fold(f64::INFINITY, f64::min)
        };
        let mut h = atom_motion::linalg::Mat::zeros(n, n);
        let path = each(&mut |k| drop(std::hint::black_box(middles[k].path(own[k].at.kappa, velocity(own[k])))));
        let generator = each(&mut |k| middles[k].generator(&paths[k], 0.1, std::hint::black_box(&mut h)));
        let force = each(&mut |k| {
            std::hint::black_box(middles[k].force(&paths[k], 0.1, &rhos[k]));
        });
        let tones = each(&mut |k| middles[k].tones(&paths[k], &rhos[k], &mut |omega, f| {
            std::hint::black_box((omega, f));
        }));
        let photons = both(&|run, s| drop(std::hint::black_box(run.photons(cluster, &s.at, &s.psi))));
        let readout = both(&|run, s| {
            std::hint::black_box(run.readout(cluster, &s.at, &s.psi));
        });
        let energy = both(&|run, s| {
            std::hint::black_box(run.energy(cluster, &s.at, &s.psi));
        });
        let reference = time(&|s| {
            std::hint::black_box(exact.reference(s.at.r));
        }, 256);
        eprintln!(
            "{name} cluster {cluster} (n {n}, {}): middle {:.0} / {:.2} µs (exact / tabulated); a path {path:.3}, a node's generator {generator:.3}, force {force:.3}, the tones {tones:.3} µs; photons {:.0} / {:.2}, readout {:.0} / {:.2}, energy {:.0} / {:.2} (exact / tabulated); reference {reference:.2} µs",
            if middles[0].still() { "still" } else { "beating" },
            middle.0, middle.1, photons.0, photons.1, readout.0, readout.1, energy.0, energy.1
        );
    }
    // The units' errors at a state (units.rs `Model::errors_with`): every candidate unit's, with the exact elimination and with
    // the tables.
    let units = model.units();
    let few = &all[..all.len().min(4)];
    let started = Instant::now();
    std::hint::black_box(model.errors(&units, few));
    let slow = started.elapsed().as_secs_f64() * 1e3 / few.len() as f64;
    let tabulated = |of: &[Unit]| {
        (0..4)
            .map(|_| {
                let started = Instant::now();
                std::hint::black_box(model.errors_with(of, &all, Some(&tables)));
                started.elapsed().as_secs_f64() * 1e3 / all.len() as f64
            })
            .fold(f64::INFINITY, f64::min)
    };
    eprintln!("{name}: the {} candidate units' errors a state: {slow:.2} ms exact, {:.3} ms tabulated", units.len(), tabulated(&units));
    // What a state's cost is made of: what its units share (no unit asked), and each kind's units beyond that.
    let shared = tabulated(&[]);
    let kinds: [(&str, &dyn Fn(&Unit) -> bool); 5] = [
        ("bare folds", &|u| matches!(u, Unit::Bare { .. })),
        ("folds at the atom", &|u| matches!(u, Unit::Drive { .. })),
        ("couplings", &|u| matches!(u, Unit::Term { .. })),
        ("dressings", &|u| matches!(u, Unit::Dressing { .. })),
        ("coherences", &|u| matches!(u, Unit::Coherence { .. })),
    ];
    let parts: Vec<String> = kinds
        .iter()
        .map(|(kind, pick)| {
            let of: Vec<Unit> = units.iter().filter(|u| pick(u)).cloned().collect();
            format!("{} {kind} {:.3} ms", of.len(), tabulated(&of) - shared)
        })
        .collect();
    eprintln!("{name}: of a state's errors, tabulated: shared {shared:.3} ms; {}", parts.join(", "));
}
