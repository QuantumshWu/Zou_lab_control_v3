//! A path's generator, force and tones (assemble.rs `Middle`, `Path`) against the model's definitions written out on dense
//! matrices (`written_out`): none of the plan's blocks, flags, near forms, slots or tones.

use super::*;
use crate::field::Light;
use crate::model::Unit;
use crate::linalg::norm3;
use crate::random::{Purpose, Stream};
use crate::request::Request;
use crate::step::Middle as _;
use crate::units::rounding;

/// δ of a near generator against another on a sample's state (tabulate.rs `Model::form_error`): with N the near light's part
/// of the generator, `far` the one without it, ‖(N′ − N)ψ‖/‖Nψ‖ -- `second` N′, `first` N, each on the state.
fn delta(far: &[C64], first: &[C64], second: &[C64]) -> f64 {
    let moved: f64 = second.iter().zip(first).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt();
    let size: f64 = first.iter().zip(far).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>().sqrt();
    if size == 0.0 { 0.0 } else { moved / size }
}

fn scene(name: &str) -> Request {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// A scene's model with the drives `at_the_atom` takes folded with the levels at the atom, and of the units it then folds those
/// `keep` takes kept.
fn model(name: &str, at_the_atom: &dyn Fn(&Light, &Unit) -> bool, keep: &dyn Fn(&Unit) -> bool) -> Model {
    let request = scene(name);
    let light = Light::new(&request);
    let bare = Model::new(&request, &light, &[]).unwrap();
    let lifted: Vec<Unit> = (0..light.frequencies.len()).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).filter(|u| at_the_atom(&light, u)).collect();
    let units = Model::new(&request, &light, &lifted).unwrap().candidates();
    let kept: Vec<Unit> = lifted.into_iter().chain(units.into_iter().filter(keep)).collect();
    Model::new(&request, &light, &kept).unwrap()
}

/// Every drive of every frequency but the trap's.
fn others(light: &Light, unit: &Unit) -> bool {
    matches!(unit, Unit::Bare { component, .. } if component.frequency != light.trap)
}

/// The models the tests run on. With every drive bare: one beam (still clusters); two frequencies with every coupling of
/// two states kept, their beats' too (tones at rest). With the drives of the light that is not the trap's folded at the
/// atom: the drives of the stretched state kept (explicit states with their shifts, drives and own photons; one carrier,
/// two of two beats, and eight explicit states); every coupling of two states kept -- of one carrier with the trap's bare
/// light (a bare coupling against a response at the atom), of one with the photons' coherences (merged classes), of two
/// carriers of several beams (all eight ground states).
fn models() -> Vec<(&'static str, Model)> {
    let none = |_: &Unit| false;
    let nothing = |_: &Light, _: &Unit| false;
    let couplings = |u: &Unit| matches!(u, Unit::Term { .. });
    let drives = |u: &Unit| matches!(u, Unit::Drive { ground: 7, .. });
    vec![
        ("R1_TWEEZER bare", model("R1_TWEEZER", &nothing, &none)),
        ("R2_RECOIL bare, couplings kept", model("R2_RECOIL", &nothing, &couplings)),
        ("R2_RECOIL at the atom, drives of |2,2> kept", model("R2_RECOIL", &others, &drives)),
        ("R2_RECOIL at the atom, couplings kept", model("R2_RECOIL", &others, &couplings)),
        ("R3_DOPPLER at the atom, couplings and coherences kept", model("R3_DOPPLER", &others, &|u| matches!(u, Unit::Term { .. } | Unit::Coherence { .. }))),
        ("R4_TW1 at the atom, couplings kept", model("R4_TW1", &others, &couplings)),
        ("R5_PUMP at the atom, drives of |2,2> kept", model("R5_PUMP", &others, &drives)),
        ("R6_RAMAN_CO at the atom, drives of |2,2> kept", model("R6_RAMAN_CO", &others, &drives)),
    ]
}

/// A value with its gradient in the middle's position; a product's rule and a sum's.
type Jet = [C64; 4];
const NONE: Jet = [C64::ZERO; 4];

fn times(a: Jet, b: Jet) -> Jet {
    std::array::from_fn(|p| if p == 0 { a[0] * b[0] } else { a[p] * b[0] + a[0] * b[p] })
}
fn plus(a: Jet, b: Jet) -> Jet {
    std::array::from_fn(|p| a[p] + b[p])
}
fn conjugate(a: Jet) -> Jet {
    a.map(|z| z.conj())
}
fn scaled(a: Jet, z: C64) -> Jet {
    a.map(|x| x * z)
}

/// A cluster's generator on a path, written out (`written_out`): its Hermitian terms' entries with their gradients in the
/// middle's position, its decay's entries (both n × n, row by row, in the path's frames), and the most terms an entry sums.
struct Written {
    hermitian: Vec<Jet>,
    decay: Vec<C64>,
    terms: usize,
}

/// A cluster's generator at the time s from a middle (r, t) along the path of velocity v, taken at the momentum κ, from the
/// model's definitions on dense matrices (model.rs, far.rs, eliminate.rs, photon.rs), every light on:
/// - every beam's amplitude the middle's with its gradient, its plane wave turned by e^{ik_l·v s}; a frequency's field
///   component e_k = Σ_l a_l γ_lk, and Doppler-weighted Σ_l w_l a_l γ_lk;
/// - the states with every light off, in their frames; per pair of frequencies their constant form Σ_k'k e'_k'* e_k C_k'k at
///   their beat (far.rs `constant`; a frequency's own its static form), an entry where the frames or a kept unit hold its beat;
/// - per pair of frequencies −½(V'†R̃ + R̃'†V) on the ground places whose beat the frames or a kept unit hold, V = Σ_k e_k U^f_k
///   a frequency's every folded coupling and R̃ = Σ_k (e_k T_k − ẽ_k T2_k) its response at the atom of the elimination at the
///   middle, in the levels its dressings make there (`Model::far_hamiltonian`; none where it is no carrier); the explicit states' van Vleck shift by the drives folded at the atom, each its own
///   denominator at the energy its cluster's ground states read the state at against the other leg's folded coupling; the explicit drives Σ_k e_k U_k and their adjoints;
/// - the decay −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}, a class's jump its feeds' emission rows on their replicas' excited
///   amplitude 𝒯_ρ (photon.rs, # Replicas and classes) -- per channel at a ground place a component's bare drives'
///   e_k R_k (e_k* counter-rotating) and a carrier's R̃, an explicit state's −1 --, each at its relative beat;
/// - every entry at its beat's phase at t + s and in its two states' Doppler frames, with their diag(K_s·v).
/// With `moduli` every factor is its modulus: the same sums with nothing cancelling, the scale of their rounding.
fn written_out(model: &Model, cluster: usize, r: [f64; 3], t: f64, kappa: [f64; 3], velocity: [f64; 3], s: f64, moduli: bool) -> Written {
    let c = &model.clusters[cluster];
    let (n, g) = (c.states.len(), c.ground);
    let c_states = |y: usize| c.states[y];
    let lights = model.running();
    let leaf = move |z: C64| if moduli { C64::real(z.abs()) } else { z };
    let real = move |x: f64| leaf(C64::real(x));
    let constant = |z: C64| -> Jet { [z, C64::ZERO, C64::ZERO, C64::ZERO] };
    let amplitudes = model.light.amplitudes(r);
    let on: Vec<Jet> = model
        .light
        .beams
        .iter()
        .zip(&amplitudes)
        .map(|(beam, a)| {
            let turn = leaf(C64::cis(dot3(beam.wave_vector(), velocity) * s));
            [a.value, a.gradient[0], a.gradient[1], a.gradient[2]].map(|z| leaf(z) * turn)
        })
        .collect();
    let field = |frequency: usize, k: usize, weighted: bool| -> Jet {
        let f = &model.light.frequencies[frequency];
        f.beams.iter().zip(&f.coefficients).fold(NONE, |sum, (&l, gamma)| {
            let weight = if weighted { leaf(gamma[k]) * real(model.doppler_shift(l, kappa)) } else { leaf(gamma[k]) };
            plus(sum, scaled(on[l], weight))
        })
    };
    let zero = model.axes.zero();
    let phase = |beat: &Beat, x: usize, y: usize| leaf(C64::cis(model.omega(&beat.plus(1, &c.frames[x]).plus(-1, &c.frames[y])) * (t + s)));
    let mut hermitian = vec![NONE; n * n];
    let mut counts = vec![0_usize; n * n];
    let mut add = |x: usize, y: usize, term: Jet| {
        hermitian[x * n + y] = plus(hermitian[x * n + y], term);
        counts[x * n + y] += 1;
    };
    // The states with every light off, in their frames; the constant forms of each pair of frequencies.
    for x in 0..n {
        for y in 0..n {
            if model.term_included(cluster, x, y, &zero) {
                add(x, y, constant(leaf(model.base[(c.states[x], c.states[y])]) * phase(&zero, x, y)));
            }
        }
        add(x, x, constant(real(-model.omega(&c.frames[x]))));
    }
    for (to, emitted) in model.tones.iter().enumerate() {
        for (from, absorbed) in model.tones.iter().enumerate() {
            let beat = emitted.beat.plus(-1, &absorbed.beat);
            let forms = if to == from { emitted.forms.clone() } else { crate::model::far::constant(emitted, absorbed) };
            let rank = absorbed.basis[0].len();
            for (at, form) in forms.iter().enumerate() {
                let product = times(conjugate(field(to, at / rank, false)), field(from, at % rank, false));
                for x in 0..n {
                    for y in 0..n {
                        if model.term_included(cluster, x, y, &beat) {
                            add(x, y, scaled(product, leaf(form[(c.states[x], c.states[y])]) * phase(&beat, x, y)));
                        }
                    }
                }
            }
        }
    }
    // The elimination at the middle with its gradient: a frequency's coupling on its field (its every folded drive's, those
    // folded at the atom alone, or as it is) and its response on the path.
    let far = model.far_hamiltonian(&amplitudes, &lights, 1);
    let responses = model.eliminate(cluster, &far, &lights, 1, model.plan(cluster, &lights).mixes());
    let coupling = |frequency: usize, basis: &[Mat], e: usize, b: usize| -> Jet { (0..basis.len()).fold(NONE, |sum, k| plus(sum, scaled(field(frequency, k, false), leaf(basis[k][(e, b)])))) };
    let folded = |frequency: usize, e: usize, b: usize| coupling(frequency, &model.tones[frequency].coupled[0], e, b);
    let response = |frequency: usize, e: usize, y: usize| -> Jet {
        let Some(carrier) = model.carrier_of(frequency) else { return NONE };
        (0..model.carriers[carrier].basis.len()).fold(NONE, |sum, k| {
            let first: Jet = std::array::from_fn(|part| leaf(responses.first[carrier][k][part][(e, y)]));
            let second: Jet = std::array::from_fn(|part| leaf(-responses.second[carrier][k][part][(e, y)]));
            plus(sum, plus(times(field(frequency, k, false), first), times(field(frequency, k, true), second)))
        })
    };
    for (wj, emitted) in model.tones.iter().enumerate() {
        for (wi, absorbed) in model.tones.iter().enumerate() {
            let beat = emitted.beat.plus(-1, &absorbed.beat);
            for x in 0..n {
                for y in 0..n {
                    if !model.term_included(cluster, x, y, &beat) {
                        continue;
                    }
                    let sum = (0..EXCITED).fold(NONE, |sum, e| {
                        let there = times(conjugate(folded(wj, GROUND + e, c.states[x])), response(wi, e, y));
                        let back = times(conjugate(response(wj, e, x)), folded(wi, GROUND + e, c.states[y]));
                        plus(sum, plus(there, back))
                    });
                    add(x, y, scaled(sum, real(-0.5) * phase(&beat, x, y)));
                }
            }
        }
    }
    for cc in &model.carriers {
        let (w, component) = (cc.frequency, Component { frequency: cc.frequency, counter: false });
        let at_the_atom = |e: usize, b: usize| coupling(w, &cc.basis, e, b);
        let _ = &at_the_atom;
        for x in g..n {
            for q in 0..g {
                if model.level(component, c.states[x], c.states[q]) == Level::Explicit {
                    let drive = coupling(w, &model.tones[w].basis[0], c.states[x], c.states[q]);
                    add(x, q, scaled(drive, phase(&cc.beat.negated(), x, q)));
                    add(q, x, scaled(conjugate(drive), phase(&cc.beat, q, x)));
                }
            }
        }
    }
    // The decay: per class and axis its jump on the landing cluster's ground places, its feeds' emission rows on their
    // replicas' 𝒯 at the path's momentum.
    let mut decay = vec![C64::ZERO; n * n];
    let excited = |replica: &crate::model::photon::Replica, e: usize, y: usize| -> C64 {
        let mut z = C64::ZERO;
        for &(channel, place) in replica.channels.iter().filter(|ch| ch.1 == y) {
            let _ = place;
            match channel {
                crate::model::photon::Channel::Near { carrier } => {
                    let f = model.carriers[carrier].frequency;
                    for k in 0..model.carriers[carrier].basis.len() {
                        let (first, second) = (&responses.first[carrier][k][0], &responses.second[carrier][k][0]);
                        z += field(f, k, false)[0] * leaf(first[(e, y)]) + field(f, k, true)[0] * leaf(-second[(e, y)]);
                    }
                }
                crate::model::photon::Channel::Bare { frequency, side } => {
                    for (k, response) in model.tones[frequency].bare_response[side].iter().enumerate() {
                        let c = field(frequency, k, false)[0];
                        z += (if side == 1 { c.conj() } else { c }) * leaf(response[(e, c_states(y))]);
                    }
                }
            }
        }
        if replica.explicit.contains(&y) && GROUND + e == c_states(y) {
            z -= leaf(C64::ONE);
        }
        z
    };
    for class in &model.classes[cluster] {
        let landing = &model.clusters[class.landing];
        for a in 0..3 {
            let mut jump = Mat::zeros(landing.ground, n);
            for (r, bp, phase) in &class.feeds {
                let replica = &model.replicas[cluster][*r];
                let own = leaf(C64::cis(model.omega(phase) * (t + s)));
                let b = landing.states[*bp];
                for y in 0..n {
                    for &e in &replica.rows {
                        jump[(*bp, y)] += leaf(model.emission[a][(b, GROUND + e)]) * excited(replica, e, y) * own;
                    }
                }
            }
            for x in 0..n {
                for y in 0..n {
                    let product: C64 = (0..landing.ground).map(|bp| jump[(bp, x)].conj() * jump[(bp, y)]).sum();
                    decay[x * n + y] += leaf(C64::new(0.0, -0.5)) * product;
                }
            }
        }
    }
    // The states' Doppler frames along the path.
    let frames = model.plan(cluster, &lights).waves.clone();
    let rate = |x: usize| dot3(frames[x], velocity);
    let turn = |x: usize| leaf(C64::cis(-rate(x) * s));
    for x in 0..n {
        for y in 0..n {
            let turned = turn(x) * turn(y).conj();
            hermitian[x * n + y] = scaled(hermitian[x * n + y], turned);
            decay[x * n + y] *= turned;
        }
        hermitian[x * n + x][0] += real(rate(x));
    }
    Written { hermitian, decay, terms: counts.into_iter().max().unwrap_or(0) + 3 * model.classes[cluster].len() }
}

/// The single products along an entry's two chains (`chain`). Of the definitions written out: a field's turn, coefficient and
/// Doppler weight, a coupling's and a response's with their fields, their product and its pair's sum, the half, the tone, the
/// two frames' turns and theirs with the entry, the decay with the Hermitian part, the frame's rate (thirteen). Of the path's:
/// a field's three, a form's product and its half, its slot and the merged value, a block's coefficient, its tone and its
/// value, the two turns and the frame's rate (thirteen).
const SINGLE: usize = 26;

/// The multiply-adds along the longest chain from a factor to an entry, of the definitions written out and of the path's
/// evaluation together: each evaluation errs by at most γ of its own chain times the sum of the entry's terms' moduli (Higham
/// §3.1), so the two differ by at most γ_N of it, N this count. The sums along the chains: a field's over its beams (each
/// evaluation's); a near term's, a form's or a jump's over the excited states, of two products each (each evaluation's); a
/// class's jump over its members; a jump product's over the landing states, and the path's over the axes' too; the entry's
/// over its terms (`Written::terms`) and over its blocks' four flags.
fn chain(model: &Model, plan: &Plan, terms: usize) -> usize {
    let beams = model.light.frequencies.iter().map(|f| f.beams.len()).max().unwrap_or(0);
    let channels = model.replicas.iter().flatten().map(|r| r.channels.len() + r.explicit.len()).max().unwrap_or(0);
    2 * (beams + 2 * EXCITED) + channels * 3 + 4 * GROUND + terms + 4 * plan.blocks.len() + SINGLE
}

/// A path through a point for the tests, in the light's own units -- K its largest wave number: a position within 1/K of
/// the focus on each axis, a time within the recoil's m/(ħK²), a momentum within K, a velocity within ħK/m that is not the
/// momentum's, and the path's times: its middle and two within the recoil's time, over which a beam's plane wave turns by
/// about a radian.
fn drawn(model: &Model, stream: &mut Stream) -> ([f64; 3], f64, [f64; 3], [f64; 3], [f64; 3]) {
    let k = model.light.beams.iter().map(|b| norm3(b.wave_vector())).fold(0.0, f64::max);
    let (speed, time) = (model.hbar_over_mass * k, 1.0 / (model.hbar_over_mass * k * k));
    let mut within = |scale: f64| scale * (2.0 * stream.uniform() - 1.0);
    let r = [within(1.0 / k), within(1.0 / k), within(1.0 / k)];
    let t = within(time);
    let kappa = [within(k), within(k), within(k)];
    let velocity = [within(speed), within(speed), within(speed)];
    (r, t, kappa, velocity, [0.0, within(time), within(time)])
}

/// A Hermitian matrix on n states, row by row: X + X† of uniform entries.
fn hermitian(n: usize, stream: &mut Stream) -> Vec<C64> {
    let x: Vec<C64> = (0..n * n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
    (0..n * n).map(|i| x[i] + x[(i % n) * n + i / n].conj()).collect()
}

/// The form Σ_xy G[x, y] ρ[y, x] of the written-out gradient along the lab axis a with a ρ.
fn form(written: &Written, a: usize, rho: &[C64], n: usize) -> C64 {
    (0..n * n).map(|i| written.hermitian[i][1 + a] * rho[(i % n) * n + i / n]).sum()
}

/// The same with every factor by its modulus.
fn form_size(sizes: &Written, a: usize, rho: &[C64], n: usize) -> f64 {
    (0..n * n).map(|i| sizes.hermitian[i][1 + a].re * rho[(i % n) * n + i / n].abs()).sum()
}

/// A path's generator (step.rs `Middle::generator`) is the model's definitions written out, at the middle and at times
/// along the path, and its force's form with a Hermitian ρ (`Middle::force`) is the written-out Hermitian part's gradient in
/// the middle's position with it, Re Σ ∂_a H[x, y] ρ[y, x] -- entry by entry to the rounding of the entry's sum: γ_N of its
/// terms' moduli (`chain`; the gradient's product rule doubles the chain, and its form with ρ adds its n² terms). The frames'
/// rates are K_s·v. A still cluster's generator is the same at every time along the path, to the same rounding twice. The
/// gradient's matrices (`Middle::gradient`, the optical sector's) are the same entries; and their slope along the path,
/// made in the same pass, is their derivative: against their central difference at ±δ, δ a thousandth over the fastest
/// a term turns at along the path -- every factor along it is a phase e^{iωs}, |ω| at most twice the beams' and the
/// frames' fastest rates and the fastest tone --, which errs by at most (ωδ)²/6 of ω times the entry's moduli, and by
/// the two ends' rounding over 2δ: their sums', and their phases' -- a phase's argument Φ = ω(t + s) is rounded to Φu
/// twice (the sum and the product), which is all of a tone's turn over δ where Φ is a hyperfine beat's at t.
#[test]
fn a_path_s_generator_and_force_are_the_definitions_written_out() {
    let (mut still, mut beating) = (0, 0);
    for (index, (name, model)) in models().into_iter().enumerate() {
        let mut stream = Stream::new(0, Purpose::Initial, index as u64, 0);
        let lights = model.running();
        let (mut generator, mut force, mut most) = (0.0_f64, 0.0_f64, 0);
        for cluster in 0..model.clusters.len() {
            let n = model.clusters[cluster].states.len();
            let plan = model.plan(cluster, &lights);
            let (r, t, kappa, velocity, times) = drawn(&model, &mut stream);
            let rho = hermitian(n, &mut stream);
            let middle = model.assemble(cluster, r, t, &lights);
            let path = middle.path(kappa, velocity);
            let mut rates = vec![0.0; n];
            middle.gauge(&path, &mut rates);
            assert!(rates.iter().zip(&plan.waves).all(|(rate, k)| *rate == dot3(*k, velocity)), "{name}, cluster {cluster}: the frames' rates {rates:?}");
            if middle.still() { still += 1 } else { beating += 1 }
            let mut first: Option<Mat> = None;
            for s in times {
                let mut h = Mat::zeros(n, n);
                middle.generator(&path, s, &mut h);
                let (written, sizes) = (written_out(&model, cluster, r, t, kappa, velocity, s, false), written_out(&model, cluster, r, t, kappa, velocity, s, true));
                let links = chain(&model, &plan, written.terms);
                most = most.max(links);
                let at_rest = first.get_or_insert_with(|| h.clone());
                for i in 0..n * n {
                    let (apart, size) = ((h.data[i] - (written.hermitian[i][0] + written.decay[i])).abs(), sizes.hermitian[i][0].re + sizes.decay[i].re);
                    assert!(apart <= rounding(links) * size, "{name}, cluster {cluster}, s {s}, entry ({}, {}): {apart:e} apart of {size:e} ({links} links)", i / n, i % n);
                    generator = generator.max(if size > 0.0 { apart / size } else { 0.0 });
                    if middle.still() {
                        let moved = (h.data[i] - at_rest.data[i]).abs();
                        assert!(moved <= 2.0 * rounding(links) * size, "{name}, cluster {cluster}, s {s}, entry ({}, {}): a still cluster's generator moved by {moved:e} of {size:e}", i / n, i % n);
                    }
                }
                let pull = middle.force(&path, s, &rho);
                for a in 0..3 {
                    let (apart, size) = ((pull[a] - form(&written, a, &rho, n).re).abs(), form_size(&sizes, a, &rho, n));
                    assert!(apart <= rounding(2 * links + n * n) * size, "{name}, cluster {cluster}, s {s}, axis {a}: the force's form {} is {apart:e} apart of {size:e}", pull[a]);
                    force = force.max(if size > 0.0 { apart / size } else { 0.0 });
                }
                // The gradient's matrices (`Middle::gradient`) are the written-out Hermitian part's gradient, entry by entry.
                let (mut g, mut slope): ([Vec<C64>; 3], [Vec<C64>; 3]) = Default::default();
                middle.gradient(&path, s, &mut g, Some(&mut slope));
                let fastest = |rates: &[f64]| rates.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
                let omega = 2.0 * (fastest(&path.rates.path) + fastest(&path.rates.gauge)) + fastest(&plan.tones);
                let delta = 1e-3 / omega.max(f64::MIN_POSITIVE);
                let phase = (t.abs() + s.abs() + delta) * fastest(&plan.tones) + (s.abs() + delta) * 2.0 * (fastest(&path.rates.path) + fastest(&path.rates.gauge));
                let (mut ahead, mut behind): ([Vec<C64>; 3], [Vec<C64>; 3]) = Default::default();
                middle.gradient(&path, s + delta, &mut ahead, None);
                middle.gradient(&path, s - delta, &mut behind, None);
                for a in 0..3 {
                    for i in 0..n * n {
                        let mirror = (i % n) * n + i / n;
                        let (apart, size) = ((g[a][i] - written.hermitian[i][1 + a]).abs(), sizes.hermitian[i][1 + a].re.max(sizes.hermitian[mirror][1 + a].re));
                        assert!(apart <= rounding(2 * links + 2) * size, "{name}, cluster {cluster}, s {s}, axis {a}, entry ({}, {}): the gradient's matrix {apart:e} apart of {size:e}", i / n, i % n);
                        let central = (ahead[a][i] - behind[a][i]) * (0.5 / delta);
                        let off = (slope[a][i] - central).abs();
                        let bound = omega * size * (1e-6 / 6.0) + 2.0 * (rounding(2 * links + 2) + 2.0 * phase * crate::units::UNIT_ROUNDOFF) * size / (2.0 * delta);
                        assert!(off <= bound, "{name}, cluster {cluster}, s {s}, axis {a}, entry ({}, {}): the gradient's slope {} is {off:e} off its central difference {} (bound {bound:e})", i / n, i % n, slope[a][i].abs(), central.abs());
                    }
                }
            }
        }
        eprintln!("{name}: the generator within {generator:.1e} and the force within {force:.1e} of their terms' moduli; the bounds {:.1e} and {:.1e} at most", rounding(most), rounding(2 * most + STATES * STATES));
    }
    assert!(still > 0 && beating > 0, "still clusters ({still}) and beating ones ({beating})");
}

/// The tones (step.rs `Middle::tones`) are the held force's even part in time. At rest on the middle the force's form is
/// F(s) = Σ_m Re[z_m e^{iΩ_m s}], z_m the gradient's entries of tone m at the middle's time with ρ, and the tones give
/// f_m = Re z_m: ½(F(s) + F(−s)) = Σ_m f_m cos Ω_m s at every s -- at s = 0 their sum is the force's form --, so that
/// ∫_{−τ/2}^{τ/2} F ds = Σ_m f_m τ sinc(Ω_m τ/2), the step's shear. To the rounding of the forms' sums on both sides (the
/// force's bound, with the tones' own sum) and of their phases' arguments: a tone's phase Ω_m(t ± s) is a product of a sum,
/// γ_2 of its radians. The first tone is at rest, and a still cluster has no other.
#[test]
fn the_tones_are_the_held_force_s_even_part() {
    let mut most = 0;
    for (index, (name, model)) in models().into_iter().enumerate() {
        let mut stream = Stream::new(1, Purpose::Initial, index as u64, 0);
        let lights = model.running();
        let mut worst = 0.0_f64;
        for cluster in 0..model.clusters.len() {
            let n = model.clusters[cluster].states.len();
            let (r, t, kappa, _, _) = drawn(&model, &mut stream);
            let rho = hermitian(n, &mut stream);
            let middle = model.assemble(cluster, r, t, &lights);
            let rest = middle.path(kappa, [0.0; 3]);
            let mut tones: Vec<(f64, [f64; 3])> = vec![];
            middle.tones(&rest, &rho, &mut |omega, form| tones.push((omega, form)));
            assert!(tones[0].0 == 0.0 && (!middle.still() || tones.len() == 1), "{name}, cluster {cluster}: the tones {tones:?}");
            most = most.max(tones.len());
            let fastest = tones.iter().map(|tone| tone.0.abs()).fold(0.0, f64::max);
            let sizes = written_out(&model, cluster, r, t, kappa, [0.0; 3], 0.0, true);
            let links = 2 * chain(&model, &model.plan(cluster, &lights), sizes.terms) + n * n + tones.len();
            let times = if fastest > 0.0 { vec![0.0, (1.0 + stream.uniform()) / fastest] } else { vec![0.0] };
            for s in times {
                let (there, back) = (middle.force(&rest, s, &rho), middle.force(&rest, -s, &rho));
                for a in 0..3 {
                    let sum: f64 = tones.iter().map(|(omega, form)| form[a] * (omega * s).cos()).sum();
                    let (apart, size) = ((sum - 0.5 * (there[a] + back[a])).abs(), form_size(&sizes, a, &rho, n));
                    let within = 2.0 * (rounding(links) + rounding(2) * fastest * (t.abs() + s)) * size;
                    assert!(apart <= within, "{name}, cluster {cluster}, s {s}, axis {a}: the tones' {sum} is {apart:e} from the force's even part, of {size:e} (within {within:e})");
                    worst = worst.max(if size > 0.0 { apart / size } else { 0.0 });
                }
            }
        }
        eprintln!("{name}: the tones within {worst:.1e} of the forms' moduli");
    }
    assert!(most > 1, "a cluster that beats at rest");
}

/// A table's middle (tabulate.rs `Model::near_point`: the table's interpolant of a point's values, each straight to its
/// place, and of its responding forms) against the middle of the table's forms at the point (`Model::near_tabulated`,
/// laid by `Site::formed`). Their Hermitian values are the same reals under the same weights, the one summed onto its
/// constant in the interpolation's own fused sum and the other added to it afterwards, and their decays are one product
/// of the same interpolated responses (module documentation, # The excited amplitude): the generators agree entry by
/// entry, and the force's forms, to the rounding of their sums (the bound of the definitions' own test) -- the decay is
/// the photons' rate between the table's nodes as at them, there is no second interpolation of it --; and each counts
/// what it makes: the table's middle its reads of the laid values (`Plan::near_read`, the normalized forms'
/// `Plan::normal_read_work`, its place's coordinates), the forms' middle the forms' reads and their laying
/// (`Plan::formed_work`). On reference scenes' models with drives folded at the atom, their tables made over the reach.
#[test]
fn a_table_s_middle_is_the_middle_of_its_forms() {
    let (mut tables_met, mut axes) = (0, 0);
    for (index, name) in ["R3_DOPPLER", "R4_TW1", "R5_PUMP"].into_iter().enumerate() {
        let (request, model) = (scene(name), model(name, &others, &|_| false));
        let mut stream = Stream::new(2, Purpose::Initial, index as u64, 0);
        let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let p: [f64; GROUND] = std::array::from_fn(|b| if b == state { 1.0 } else { 0.0 });
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        let tables = model.tabulate(&reach, 1 << 12);
        let lights = model.running();
        for cluster in 0..model.clusters.len() {
            let Some(table) = tables.of(cluster, &lights) else { continue };
            tables_met += 1;
            axes += table.axes();
            let (plan, n) = (model.plan(cluster, &lights), model.clusters[cluster].states.len());
            let sections = [(plan.near.hermitian(), 1), (plan.near.responding(), 1)];
            for _ in 0..16 {
                let (r, t, kappa, velocity, times) = drawn(&model, &mut stream);
                let rho = hermitian(n, &mut stream);
                let amplitudes = model.light.amplitudes(r);
                let near = model.near_values(&plan, &amplitudes, &sections, Some(table));
                let (ours, theirs) = (model.middle(&plan, r, t, Some(table)), model.middle_of(&plan, t, &amplitudes, Some(&near)));
                let place = ours.point.borrow().place.as_ref().map_or(0.0, |p| p.1);
                let normal = if plan.fourth.is_some() { plan.normal_read_work(table.axes(), true) } else { 0.0 };
                assert_eq!(ours.work, place + plan.near_read(table.axes()) + normal, "{name}, cluster {cluster}: the table's middle counts its reads");
                assert_eq!(theirs.work, near.work + plan.formed_work(theirs.point.borrow().directions.len()), "{name}, cluster {cluster}: the forms' middle their reads and their laying");
                let (path, other) = (ours.path(kappa, velocity), theirs.path(kappa, velocity));
                let sizes = written_out(&model, cluster, r, t, kappa, velocity, 0.0, true);
                let links = chain(&model, &plan, sizes.terms);
                for s in times {
                    let (mut h, mut g) = (Mat::zeros(n, n), Mat::zeros(n, n));
                    ours.generator(&path, s, &mut h);
                    theirs.generator(&other, s, &mut g);
                    for i in 0..n * n {
                        let (apart, size) = ((h.data[i] - g.data[i]).abs(), sizes.hermitian[i][0].re + sizes.decay[i].re);
                        assert!(apart <= rounding(links) * size, "{name}, cluster {cluster}, s {s}, entry ({}, {}): the generators are {apart:e} apart of {size:e}", i / n, i % n);
                    }
                    let (pull, pulled) = (ours.force(&path, s, &rho), theirs.force(&other, s, &rho));
                    for a in 0..3 {
                        let (apart, size) = ((pull[a] - pulled[a]).abs(), form_size(&sizes, a, &rho, n));
                        assert!(apart <= rounding(2 * links + n * n) * size, "{name}, cluster {cluster}, s {s}, axis {a}: the force's forms are {apart:e} apart of {size:e}");
                    }
                }
            }
        }
    }
    assert!(tables_met > 0 && axes > 0, "tables met ({tables_met}) with axes ({axes})");
}

/// A state's photons made in parts (photon.rs `Gradients`) are the same photons: every class's amplitudes without any
/// gradients (what a jump's draw reads) are the amplitudes of the photons made whole, and one class alone with its
/// gradients (what a jump makes of the class it draws) is that class of them but for the explicit states' part of the
/// second moment's rate, which a class alone does not make -- the same numbers, bit for bit, so a
/// jump's photon (`Dynamics::emit`: the classes' rates to the draw, the drawn class's amplitudes and kick) is the whole
/// photon's, whichever class is drawn. On every model of the tests, with the exact elimination, and on a scene's tables.
#[test]
fn a_state_s_photons_in_parts_are_its_photons() {
    use crate::model::photon::Gradients;
    use crate::model::Run;
    let (mut classes, mut folded) = (0, 0);
    let mut check = |name: &str, model: &Model, run: &Run, tables: Option<&crate::model::tabulate::Tables>, stream: &mut Stream| {
        let lights = model.running();
        for cluster in 0..model.clusters.len() {
            let (plan, n) = (model.plan(cluster, &lights), model.clusters[cluster].states.len());
            let table = tables.and_then(|t| t.of(cluster, &lights));
            let (r, t, kappa, _, _) = drawn(model, stream);
            let at = crate::model::Point { r, kappa, t };
            let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
            // Each count made afresh: a thread keeps its last point's normalized matrices (`Model::quartic`).
            Model::forget_quartic();
            let (whole, work) = model.photons_of(&plan, table, &at, &psi, Gradients::All);
            Model::forget_quartic();
            let counted = model.photons_counted(cluster, &at, &psi, tables);
            assert_eq!((&whole, work), (&counted.0, counted.1), "{name}, cluster {cluster}: the photons by the cluster's own plan and table");
            // A step's scattering reads the same photons where they are made: the same sums in the same order.
            let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
            assert_eq!(model.scattering_of(&plan, table, &at, &psi).0, crate::model::Scattering::of(&whole, norm), "{name}, cluster {cluster}: the scattering");
            let bare = model.photons_of(&plan, table, &at, &psi, Gradients::None).0;
            assert_eq!(bare, run.photon_amplitudes(cluster, &at, &psi));
            // A class of no amplitude is a photon of the whole alone, its gradients' second moment (photon.rs
            // `Model::classes_on`): the classes a draw reads are those of an amplitude.
            let emitting: Vec<&crate::model::Photon> = whole.iter().filter(|p| p.amplitudes.iter().flatten().any(|z| *z != C64::ZERO)).collect();
            assert_eq!(emitting.len(), bare.len(), "{name}, cluster {cluster}: the classes");
            for (drawn, (photon, alone)) in emitting.into_iter().zip(&bare).enumerate() {
                assert!(alone.gradients.iter().flatten().all(|g| g.is_empty()), "{name}, cluster {cluster}, class {}: gradients where none are asked", photon.class);
                assert!((alone.class, &alone.amplitudes) == (photon.class, &photon.amplitudes), "{name}, cluster {cluster}, class {}: the amplitudes alone", photon.class);
                let one = model.photons_of(&plan, table, &at, &psi, Gradients::Of(photon.class)).0;
                assert!(one.len() == 1 && one[0] == *photon, "{name}, cluster {cluster}, class {}: the class alone", photon.class);
                let mut draw = |rates: &[f64]| {
                    assert!(rates.len() == bare.len() && rates.iter().zip(&bare).all(|(rate, photon)| *rate == photon.weight()), "{name}, cluster {cluster}: the classes' rates to a jump's draw");
                    Some(drawn)
                };
                let emitted = crate::step::Dynamics::emit(run, cluster, &at, None, &psi, &mut draw).expect("the class drawn");
                assert!((emitted.landing, emitted.wave_number, emitted.kick) == (photon.landing, photon.wave_number, photon.kick()) && (0..3).all(|a| emitted.amplitudes[a][..] == alone.amplitudes[a][..]), "{name}, cluster {cluster}, class {}: a jump's photon", photon.class);
                classes += 1;
                folded += photon.gradients.iter().flatten().flatten().filter(|z| **z != C64::ZERO).count().min(1);
            }
        }
    };
    for (index, (name, model)) in models().into_iter().enumerate() {
        let mut stream = Stream::new(4, Purpose::Initial, index as u64, 0);
        check(name, &model, &Run::new(&model, [1.0 / GROUND as f64; GROUND]), None, &mut stream);
        // With the normalized generator's every part (the series' first order and the normalization kept): the photons
        // of L = (J(κ) + c J2 E) S⁻¹ with their gradients, in parts and whole.
        let bent = model.rekept(&[model.kept.clone(), vec![Unit::Series, Unit::Normalization { order: 1 }]].concat()).unwrap();
        if (0..bent.clusters.len()).any(|c| bent.plan(c, &bent.running()).fourth.as_ref().is_some_and(|f| f.bends())) {
            check(&format!("{name}, normalized"), &bent, &Run::new(&bent, [1.0 / GROUND as f64; GROUND]), None, &mut stream);
        }
    }
    let (name, request, model) = ("R4_TW1 tabulated", scene("R4_TW1"), model("R4_TW1", &others, &|_| false));
    let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p: [f64; GROUND] = std::array::from_fn(|b| if b == state { 1.0 } else { 0.0 });
    let reach = model.reach(&model.trap(&p, &model.prepared()).unwrap(), &p, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
    let tables = model.tabulate(&reach, 1 << 12);
    check(name, &model, &Run::tabulated(&model, &tables, p), Some(&tables), &mut Stream::new(4, Purpose::Initial, 99, 0));
    assert!(classes > 0 && folded > 0, "classes ({classes}) with folded gradients ({folded})");
}

/// A class's photon written out entry by entry: u_a and ∇u_a (`[i][a]`) on the landing cluster's states, and the sums of
/// their terms' moduli.
struct Entries {
    u: [Vec<C64>; 3],
    du: [[Vec<C64>; 3]; 3],
    sizes: [Vec<f64>; 3],
    slopes: [[Vec<f64>; 3]; 3],
    terms: usize,
}

/// A state's photons by the definition's entries (photon.rs, # Replicas and classes, # The absorbed light's momentum), none
/// of the plan's replicas or classes: per class, feed, axis a and excited state e of the feed's replica, C_a[b', e] times
/// the replica's excited amplitude there -- per channel at a ground place s a carrier's near forms (its response with its
/// gradient, `NearValues::first_order`) times its field component less the Doppler-weighted component times its partner,
/// a bare component's constant response times its component, an explicit state's −1 -- times ψ_s, at the feed's phase;
/// ∇u by the product rule entry by entry. `bound(place)` is the bound of a near form's modulus in the sums of moduli (its
/// own, or what its interpolant's terms can reach).
fn photons_written(model: &Model, plan: &Plan, near: Option<&NearValues>, bound: &dyn Fn(usize) -> f64, amplitudes: &[Amplitude], at: &crate::model::Point, psi: &[C64]) -> Vec<Entries> {
    use crate::model::photon::Channel;
    let fields = model.fields(plan, amplitudes, at.kappa, 1);
    let (start, count) = (plan.near.responding().start, plan.near.responses.len());
    let cluster = &model.clusters[plan.cluster];
    // A near form with its gradient, and their moduli's bounds.
    let form = |place: usize| -> ([C64; 4], [f64; 4]) {
        let near = near.expect("a near form's values");
        let value = near.first_order(place);
        let mut size = [bound(place), 0.0, 0.0, 0.0];
        match &near.directions {
            None => (0..3).for_each(|i| size[1 + i] = value[1 + i].abs()),
            Some(directions) => {
                for (f, d) in directions.iter().enumerate() {
                    (0..3).for_each(|i| size[1 + i] += near.data[(1 + f) * near.len + place].abs() * d[i].abs());
                }
            }
        }
        (value, size)
    };
    let scalar = |frequency: usize, k: usize, conjugate: bool| plan.scalars.iter().position(|s| (s.frequency, s.group, s.component, s.conjugate) == (frequency, None, k, conjugate));
    let one = [C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO];
    // A replica's excited amplitude at e of place y (its value and gradient, and their moduli's bounds), per unit ψ_y.
    let excited = |replica: &crate::model::photon::Replica, e: usize, y: usize| -> ([C64; 4], [f64; 4]) {
        let (mut z, mut size) = ([C64::ZERO; 4], [0.0; 4]);
        let mut add = |c: &[C64], cs: [f64; 4], t: [C64; 4], ts: [f64; 4]| {
            z[0] += c[0] * t[0];
            size[0] += cs[0] * ts[0];
            for i in 0..3 {
                z[1 + i] += c[1 + i] * t[0] + c[0] * t[1 + i];
                size[1 + i] += cs[1 + i] * ts[0] + cs[0] * ts[1 + i];
            }
        };
        let moduli = |c: &[C64]| -> [f64; 4] { std::array::from_fn(|p| c[p].abs()) };
        for &(channel, _) in replica.channels.iter().filter(|ch| ch.1 == y) {
            match channel {
                Channel::Near { carrier } => {
                    let cc = &model.carriers[carrier];
                    if !plan.lights.on(cc.frequency) {
                        continue;
                    }
                    for k in 0..cc.basis.len() {
                        let (Some(s), Some(i)) = (scalar(cc.frequency, k, false), plan.near.responses.iter().position(|r| *r == [carrier, k, e, y])) else { continue };
                        let (c, cw) = (fields.plain(s), fields.doppler(s));
                        let ((t, ts), (t2, ts2)) = (form(start + i), form(start + count + i));
                        add(c, moduli(c), t, ts);
                        let minus: Vec<C64> = cw.iter().map(|x| -*x).collect();
                        add(&minus, moduli(cw), t2, ts2);
                    }
                }
                Channel::Bare { frequency, side } => {
                    if !plan.lights.on(frequency) {
                        continue;
                    }
                    for (k, response) in model.tones[frequency].bare_response[side].iter().enumerate() {
                        let Some(s) = scalar(frequency, k, side == 1) else { continue };
                        let v = response[(e, cluster.states[y])];
                        let c = fields.plain(s);
                        add(c, moduli(c), [v, C64::ZERO, C64::ZERO, C64::ZERO], [v.abs(), 0.0, 0.0, 0.0]);
                    }
                }
            }
        }
        if replica.explicit.contains(&y) && GROUND + e == cluster.states[y] {
            add(&one, [1.0, 0.0, 0.0, 0.0], [-C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO], [1.0, 0.0, 0.0, 0.0]);
        }
        (z, size)
    };
    model.classes[plan.cluster]
        .iter()
        .map(|class| {
            let landing = &model.clusters[class.landing];
            let n = landing.states.len();
            let mut out = Entries {
                u: std::array::from_fn(|_| vec![C64::ZERO; n]),
                du: std::array::from_fn(|_| std::array::from_fn(|_| vec![C64::ZERO; n])),
                sizes: std::array::from_fn(|_| vec![0.0; n]),
                slopes: std::array::from_fn(|_| std::array::from_fn(|_| vec![0.0; n])),
                terms: 0,
            };
            for (r, bp, phase) in &class.feeds {
                let replica = &model.replicas[plan.cluster][*r];
                let phase = C64::cis(model.omega(phase) * at.t);
                let b = landing.states[*bp];
                for (y, &x) in psi.iter().enumerate() {
                    let x = x * phase;
                    for &e in &replica.rows {
                        let (z, size) = excited(replica, e, y);
                        for a in 0..3 {
                            let c = model.emission[a][(b, GROUND + e)];
                            if c == C64::ZERO {
                                continue;
                            }
                            out.u[a][*bp] += c * z[0] * x;
                            out.sizes[a][*bp] += c.abs() * size[0] * x.abs();
                            out.terms += 1;
                            for i in 0..3 {
                                out.du[i][a][*bp] += c * z[1 + i] * x;
                                out.slopes[i][a][*bp] += c.abs() * size[1 + i] * x.abs();
                            }
                        }
                    }
                }
            }
            out
        })
        .collect()
}

/// A state's photons (photon.rs `Model::photons_of`: the replicas' excited amplitudes once, then the classes' emission
/// rows on them) are the definition's entries summed one by one (`photons_written`), amplitudes and gradients, to the
/// rounding of each sum -- γ_N of its terms' moduli, N the entries of the class's row, each a chain of a field's sum over
/// its beams, the response's own sum (a form's over the excited states, or an interpolant's over its stencil), the
/// product rule's four terms and each direction's two, the excited amplitude's sum over the places; and their moments
/// (`Photon::moments`) are those of the written-out vectors. With the exact elimination at the point, on every model of
/// the tests; and from tables made over the reach, whose interpolant of the responses is that of the responses
/// themselves: there a response's terms are bounded by its largest modulus over the table's nodes times the stencil's
/// weights' moduli, at most 3/2 an axis (5/4 inside, table.rs `Axis::weights`), and its slopes are the same numbers in
/// both.
#[test]
fn a_state_s_photons_are_their_entries_summed() {
    use crate::model::photon::Gradients;
    let (mut classes, mut folded, mut near_met, mut axes) = (0, 0, 0, 0);
    // The largest difference met, in unit round-offs of its sum's terms' moduli: with the exact elimination, from tables.
    let mut apart = [0.0_f64; 2];
    let mut check = |name: &str, model: &Model, tables: Option<&crate::model::tabulate::Tables>, stream: &mut Stream| {
        let lights = model.running();
        let beams = model.light.frequencies.iter().map(|f| f.beams.len()).max().unwrap_or(0);
        for cluster in 0..model.clusters.len() {
            let (plan, n) = (model.plan(cluster, &lights), model.clusters[cluster].states.len());
            let table = tables.and_then(|t| t.of(cluster, &lights));
            let len = plan.near.len();
            // Per near form its largest modulus over the table's nodes.
            let largest: Vec<f64> = table.map_or(vec![], |t| {
                let mut most = vec![0.0_f64; len];
                for node in 0..crate::table::Table::nodes(t.table.axes()) {
                    let reals = t.table.node(node).1;
                    most.iter_mut().enumerate().for_each(|(at, m)| *m = m.max(C64::new(reals[2 * at], reals[2 * at + 1]).abs()));
                }
                most
            });
            let d = table.map_or(0, |t| t.axes());
            axes += d;
            for _ in 0..8 {
                let (r, t, kappa, _, _) = drawn(model, stream);
                let at = crate::model::Point { r, kappa, t };
                let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
                let amplitudes = model.light.amplitudes(r);
                // The table's where it holds the point, else the exact elimination's (tabulate.rs, # Axes).
                let table = table.filter(|t| model.near_tabulated(t, &amplitudes, &[(plan.near.responding(), 1)]).is_some());
                let near = match table {
                    _ if len == 0 => None,
                    Some(table) => model.near_tabulated(table, &amplitudes, &[(plan.near.responding(), 1)]),
                    None => Some(model.near_exact(&plan, &amplitudes, 1)),
                };
                near_met += near.is_some() as usize;
                let bound = |place: usize| match (table, &near) {
                    (Some(_), _) => 1.5_f64.powi(d as i32) * largest[place],
                    (None, Some(near)) => near.data[place].abs(),
                    (None, None) => 0.0,
                };
                let written = photons_written(model, &plan, near.as_ref(), &bound, &amplitudes, &at, &psi);
                let photons = model.photons_of(&plan, table, &at, &psi, Gradients::All).0;
                for (place, (class, entries)) in plan.classes.iter().zip(&written).enumerate() {
                    let landing = model.clusters[class.landing].states.len();
                    // The chain to a row's sum: its entries, each a field's sum over its beams, a jump's own sum and
                    // the product rule's terms with a direction's two each.
                    let links = entries.terms.div_ceil(landing.max(1)) + 2 * beams + 2 * EXCITED + 4usize.pow(d as u32) + 4 + 2 * d.max(3) + 8 + 2 * n;
                    let photon = photons.iter().find(|p| p.class == place);
                    let zero = vec![C64::ZERO; landing];
                    for a in 0..3 {
                        for b in 0..landing {
                            let (ours, size) = (photon.map_or(C64::ZERO, |p| p.amplitudes[a][b]), entries.sizes[a][b]);
                            assert!((ours - entries.u[a][b]).abs() <= rounding(links) * size, "{name}, cluster {cluster}, class {place}, u[{a}][{b}]: {ours:?} for {:?} of {size:e}", entries.u[a][b]);
                            let seen = &mut apart[table.is_some() as usize];
                            *seen = seen.max(if size > 0.0 { (ours - entries.u[a][b]).abs() / size / crate::units::UNIT_ROUNDOFF } else { 0.0 });
                            for i in 0..3 {
                                let (ours, size) = (photon.map_or(&zero, |p| &p.gradients[i][a])[b], entries.slopes[i][a][b]);
                                assert!((ours - entries.du[i][a][b]).abs() <= rounding(links) * size, "{name}, cluster {cluster}, class {place}, ∂{i} u[{a}][{b}]: {ours:?} for {:?} of {size:e}", entries.du[i][a][b]);
                                let seen = &mut apart[table.is_some() as usize];
                                *seen = seen.max(if size > 0.0 { (ours - entries.du[i][a][b]).abs() / size / crate::units::UNIT_ROUNDOFF } else { 0.0 });
                            }
                        }
                    }
                    // A class left out is nothing: its entries within their sums' rounding of zero, checked above.
                    let Some(photon) = photon else { continue };
                    assert_eq!((photon.landing, photon.wave_number, photon.absorbed), (class.landing, class.wave_number, class.absorbed));
                    // The moments of the written-out vectors: W, Σ_a Im⟨u_a|∇u_a⟩, Σ_a Re⟨∇u_a|⊗|∇u_a⟩ with the explicit part.
                    let moments = photon.moments();
                    let (u, du) = (&photon.amplitudes, &photon.gradients);
                    let weight: f64 = u.iter().map(|v| crate::linalg::norm_sqr(v)).sum();
                    let scale = |x: &[C64], y: &[C64]| x.iter().zip(y).map(|(p, q)| p.abs() * q.abs()).sum::<f64>();
                    let sum = rounding(2 * 3 * landing + 4);
                    assert!((moments.weight - weight).abs() <= sum * weight, "{name}, cluster {cluster}, class {place}: the weight {} for {weight}", moments.weight);
                    for i in 0..3 {
                        let (mean, size): (f64, f64) = ((0..3).map(|a| crate::linalg::inner(&u[a], &du[i][a]).im).sum(), (0..3).map(|a| scale(&u[a], &du[i][a])).sum());
                        assert!((moments.mean[i] - mean).abs() <= sum * 2.0 * size, "{name}, cluster {cluster}, class {place}: the mean along {i}, {} for {mean}", moments.mean[i]);
                        for j in 0..3 {
                            let (second, size): (f64, f64) = ((0..3).map(|a| crate::linalg::inner(&du[i][a], &du[j][a]).re).sum::<f64>(), (0..3).map(|a| scale(&du[i][a], &du[j][a])).sum::<f64>());
                            assert!((moments.second[i][j] - second).abs() <= sum * 2.0 * size, "{name}, cluster {cluster}, class {place}: the second moment ({i}, {j}), {} for {second}", moments.second[i][j]);
                        }
                    }
                    classes += 1;
                    folded += du.iter().flatten().flatten().filter(|z| **z != C64::ZERO).count().min(1);
                }
            }
        }
    };
    for (index, (name, model)) in models().into_iter().enumerate() {
        check(name, &model, None, &mut Stream::new(5, Purpose::Initial, index as u64, 0));
    }
    for (index, name) in ["R3_DOPPLER", "R4_TW1", "R5_PUMP"].into_iter().enumerate() {
        let (request, model) = (scene(name), model(name, &others, &|_| false));
        let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let p: [f64; GROUND] = std::array::from_fn(|b| if b == state { 1.0 } else { 0.0 });
        let reach = model.reach(&model.trap(&p, &model.prepared()).unwrap(), &p, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        let tables = model.tabulate(&reach, 1 << 12);
        check(&format!("{name} tabulated"), &model, Some(&tables), &mut Stream::new(5, Purpose::Initial, 100 + index as u64, 0));
    }
    assert!(classes > 0 && folded > 0 && near_met > 0 && axes > 0, "classes ({classes}) with folded gradients ({folded}), near forms ({near_met}), tables' axes ({axes})");
    eprintln!("the photons against their entries' sums: at most {:.1} unit round-offs of a sum's terms' moduli apart with the exact elimination, {:.1} from tables ({classes} classes' photons)", apart[0], apart[1]);
}

/// The scattering's table error on a scene's own model (run by name, alone: `SCENE=<a reference scene's name or a native
/// request's path> … the_scattering_s_table_error_on_a_scene -- --ignored --nocapture`): the scene's model as the solve
/// closes it with its near tables (every unit whose fold does not exist over the reach kept, until none fails), and per
/// cluster with a table, at 256 atoms of
/// the scene's temperature in the trap's harmonic approximation, each a random state of the cluster: a step's scattering
/// (photon.rs `Model::scattering_of`) from the table against the exact elimination's at the point -- the force's
/// difference over the classes' summed |Σ_a Im⟨u_a|∇u_a⟩|, the diffusion's largest entry's over the summed second
/// moments' diagonal (`Scattering::size`) --, beside the table's whole read of the generator at the same atoms.
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn the_scattering_s_table_error_on_a_scene() {
    use crate::model::photon::Gradients;
    let name = std::env::var("SCENE").expect("SCENE: a reference scene's name or a request file's path");
    let request = if name.ends_with(".json") { Request::from_native(&serde_json::from_str(&std::fs::read_to_string(&name).unwrap()).unwrap()).unwrap() } else { scene(&name) };
    let light = Light::new(&request);
    let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let (model, _, trap, reach) = Model::closed_at(&request, &light, vec![], state, request.temperature_uk);
    let tables = model.tabulate(&reach, 1 << 20);
    let (lights, kt) = (model.running(), crate::units::KT_PER_UK * request.temperature_uk);
    let mut stream = Stream::new(3, Purpose::Initial, 0, 0);
    for cluster in 0..model.clusters.len() {
        let Some(table) = tables.of(cluster, &lights) else { continue };
        let (plan, n) = (model.plan(cluster, &lights), model.clusters[cluster].states.len());
        // (the mean, the largest) of the force's error, the diffusion's, and the generator's whole read.
        let mut reads = [[0.0_f64; 2]; 3];
        let atoms = 256;
        for _ in 0..atoms {
            let along: [f64; 3] = std::array::from_fn(|i| stream.normal() * (kt * model.hbar_over_mass).sqrt() / trap.frequencies[i]);
            let r: [f64; 3] = std::array::from_fn(|a| trap.centre[a] + (0..3).map(|i| trap.axes[i][a] * along[i]).sum::<f64>());
            let kappa: [f64; 3] = std::array::from_fn(|_| stream.normal() * (kt / model.hbar_over_mass).sqrt());
            let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.normal(), stream.normal())).collect();
            let at = crate::model::Point { r, kappa, t: 0.0 };
            let (ours, exact) = (model.scattering_of(&plan, Some(table), &at, &psi).0, model.scattering_of(&plan, None, &at, &psi).0);
            let norm = crate::linalg::norm_sqr(&psi);
            let means: f64 = model.photons_of(&plan, None, &at, &psi, Gradients::All).0.iter().map(|photon| norm3(photon.moments().mean)).sum::<f64>() / norm;
            let force = norm3(std::array::from_fn(|i| ours.force[i] - exact.force[i])) / means;
            let diffusion = (0..9).map(|i| (ours.diffusion[i / 3][i % 3] - exact.diffusion[i / 3][i % 3]).abs()).fold(0.0, f64::max) / exact.size;
            let sample = crate::model::Sample { cluster, at, psi };
            let amplitudes = model.light.amplitudes(r);
            let action = |middle: &Middle| model.action(&sample, middle);
            let far = action(&model.middle_of(&plan, 0.0, &amplitudes, None));
            let whole = delta(&far, &action(&model.middle_of(&plan, 0.0, &amplitudes, Some(&model.near_exact(&plan, &amplitudes, 1)))), &action(&model.middle(&plan, r, 0.0, Some(table))));
            for (read, x) in reads.iter_mut().zip([force, diffusion, whole]) {
                *read = [read[0] + x / atoms as f64, read[1].max(x)];
            }
        }
        eprintln!(
            "{name} cluster {cluster} ({n} states; {} axes, {} nodes of {} reals, {} of them the responding forms'): the scattering from the table against the exact elimination's -- the force {:.2e} (largest {:.2e}) of the classes' means, the diffusion {:.2e} ({:.2e}) of the second moments | the table's whole read of the generator δ {:.2e} ({:.2e})",
            table.axes(),
            crate::table::Table::nodes(table.table.axes()),
            table.table.width(),
            2 * plan.near.responding().len(),
            reads[0][0],
            reads[0][1],
            reads[1][0],
            reads[1][1],
            reads[2][0],
            reads[2][1]
        );
    }
    let (shown, temperature) = crate::model::tabulate::shares();
    eprintln!("{name}: the tables' shares {shown:.2e} on what is shown, {temperature:.2e} on T");
}

/// The normalized generator's part along a step's path (ELIMINATION v7; module documentation, # The normalized
/// generator): each node's is its own forms' (`Model::node_normal`), so the step's quadrature of it through its three
/// nodes -- Simpson's -- is the path's integral within O(τ⁵), relatively (Kvτ)⁴, where the part held at the middle,
/// τ δ(0), was within O(τ³), (Kvτ)². On PGC3D's F = 2 states at its cooling light's coherent peak near the focus --
/// six beams: an intensity pattern, X turns along a path (the 1D PGC's σ⁺σ⁻ has none) -- with the normalization at order
/// 1 (its excited share 0.01 asks none of the share: taken here to see the part), at a speed of its 30 µK reach
/// (0.05 µm/µs: Kvτ ≈ 0.8 for the pattern's 2k at the run's 0.95 µs) and steps of 0.95 µs and its halves: the decay's
/// normalized part δ(s) = D(s) − Λ(s), D the node's generator's decay i(Ĥ − Ĥ†) and Λ that of the same terms without it
/// (the plan without its normalized generator), against its integral by Simpson's rule on 512 panels.
#[test]
fn the_normalized_part_along_a_path_is_its_nodes_and_its_quadrature_falls_as_the_fourth_power() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("PGC3D_1ms.json");
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let request = Request::from_native(&line).unwrap();
    let light = Light::new(&request);
    let bare = Model::new(&request, &light, &[]).unwrap();
    let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
    let mut model = Model::new(&request, &light, &kept).unwrap();
    let (r, v) = ([0.15, 0.0, 0.15], [0.04, 0.02, 0.025]);
    model.referenced(&[(r, v), ([0.0; 3], [0.0; 3])]);
    model.series.normalization = model.series.normalization.max(1);
    let cluster = model.cluster_of(GROUND - 1);
    let plan = model.plan(cluster, &model.running());
    let fourth = plan.fourth.as_ref().expect("the cluster's fold has its normalized generator");
    assert!(fourth.orders.normalization >= 1, "the normalization takes an order here: {:?}", fourth.orders);
    let without = Arc::new(Plan { fourth: None, ..(*plan).clone() });
    let n = plan.states;
    let kappa = v.map(|x| x / model.hbar_over_mass);
    let (with, bare_middle) = (model.middle(&plan, r, 0.0, None), model.middle(&without, r, 0.0, None));
    let (on, off) = (with.path(kappa, v), bare_middle.path(kappa, v));
    let decay = |m: &Middle, p: &Path, s: f64| -> Vec<C64> {
        let mut h = Mat::zeros(n, n);
        m.generator(p, s, &mut h);
        (0..n * n).map(|i| (h.data[i] - h.data[(i % n) * n + i / n].conj()) * C64::I).collect()
    };
    let part = |s: f64| -> Vec<C64> { decay(&with, &on, s).iter().zip(decay(&bare_middle, &off, s)).map(|(a, b)| *a - b).collect() };
    let norm = |x: &[C64]| x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
    let mut last: Option<(f64, f64)> = None;
    for tau in [0.95, 0.475, 0.2375] {
        let h = 0.5 * tau;
        let panels = 512;
        let mut exact = vec![C64::ZERO; n * n];
        for k in 0..=2 * panels {
            let s = -h + tau * k as f64 / (2 * panels) as f64;
            let w = if k == 0 || k == 2 * panels { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 } * tau / (6 * panels) as f64;
            exact.iter_mut().zip(part(s)).for_each(|(z, x)| *z += x * w);
        }
        let (a, m, b) = (part(-h), part(0.0), part(h));
        let nodes: Vec<C64> = (0..n * n).map(|i| (a[i] + m[i] * 4.0 + b[i]) * (h / 3.0)).collect();
        let held: Vec<C64> = m.iter().map(|z| *z * tau).collect();
        let size = norm(&exact);
        let (node_error, held_error) = (norm(&nodes.iter().zip(&exact).map(|(x, y)| *x - *y).collect::<Vec<_>>()) / size, norm(&held.iter().zip(&exact).map(|(x, y)| *x - *y).collect::<Vec<_>>()) / size);
        println!("tau {tau}: the normalized part's integral {size:.3e}; through the nodes {node_error:.3e} apart, held at the middle {held_error:.3e} apart");
        assert!(node_error < held_error, "the nodes' quadrature is the better");
        if let Some((nodes_before, held_before)) = last {
            println!("    falls {:.2} (nodes), {:.2} (held) a halving", nodes_before / node_error, held_before / held_error);
            assert!(nodes_before / node_error > 10.0, "the nodes' error falls as τ⁴: {}", nodes_before / node_error);
            assert!((3.0..6.0).contains(&(held_before / held_error)), "the held part's error falls as τ²: {}", held_before / held_error);
        }
        last = Some((node_error, held_error));
    }
}

/// The held part's force form (`Middle::held_force`) is the form of its gradient's matrices: at the series' order 1
/// -- on the first cluster whose run holds it (its plan's orders, eliminate.rs `Model::references`): of the 1D
/// PGC at 3 G, refs/requests/PGC1D_TILT_0, its cluster 0 (its F = 2 sublevels, one cluster of Zeeman-split levels,
/// take it no longer), else PGC3D's --
/// the held part −Herm(Y E₀) at a node, its force form on a Hermitian ρ by the forms' terms contracted with E₀ρ and ρY
/// (no gradient matrix made) against Σ_xy ∇held[x, y] ρ[y, x] of the node's whole matrices (`Middle::gradient`, the
/// node's gradients made): at the three nodes of a step along a path, on random states, to the rounding of the sums.
#[test]
fn the_held_part_s_force_form_is_its_gradient_s() {
    let (r, v) = ([0.05, -0.1, 0.0], [0.09, 0.05, 0.07]);
    let points = [(r, v), ([0.0; 3], [0.0; 3])];
    let (model, cluster) = ["PGC1D_TILT_0.json", "PGC3D_1ms.json"]
        .iter()
        .find_map(|file| {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join(file);
            let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let request = Request::from_native(&line).unwrap();
            let light = Light::new(&request);
            let bare = Model::new(&request, &light, &[]).unwrap();
            let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
            // The prior's units at the points, and the series at its first order kept (`Unit::Series`): the held part
            // is anything where a cluster's run takes it.
            let prior = Model::new(&request, &light, &kept).unwrap().prior(&points);
            let mut model = Model::new(&request, &light, &[kept, prior, vec![Unit::Series]].concat()).unwrap();
            model.referenced(&points);
            let running = model.running();
            let cluster = (0..model.clusters().len()).find(|&c| model.plan(c, &running).fourth.as_ref().is_some_and(|f| f.held()));
            println!("{file}: a cluster whose run takes the series' first order: {cluster:?}");
            cluster.map(|c| (model, c))
        })
        .expect("a cluster whose run holds the series' first order");
    let plan = model.plan(cluster, &model.running());
    let fourth = plan.fourth.as_ref().expect("the cluster's fold has its normalized generator");
    assert!(fourth.held(), "the series at order 1: {:?}", fourth.orders);
    let n = plan.states;
    let kappa = v.map(|x| x / model.hbar_over_mass);
    let mut stream = Stream::new(7, Purpose::Initial, 0, 0);
    for s in [-0.165, 0.0, 0.165] {
        for _ in 0..3 {
            let rho = hermitian(n, &mut stream);
            // A fresh middle for each: the force's form first (the node's values alone), then the gradient's matrices.
            let middle = model.middle(&plan, r, 0.0, None);
            let path = middle.path(kappa, v);
            let force = middle.force(&path, s, &rho);
            let mut g: [Vec<C64>; 3] = Default::default();
            middle.gradient(&path, s, &mut g, None);
            let form: [f64; 3] = std::array::from_fn(|a| (0..n * n).map(|i| (g[a][i] * rho[(i % n) * n + i / n]).re).sum());
            let size: f64 = (0..3).map(|a| (0..n * n).map(|i| (g[a][i] * rho[(i % n) * n + i / n]).abs()).sum::<f64>()).fold(0.0, f64::max);
            println!("s {s}: the force's form {force:?}, the gradient's {form:?} (of {size:.3e})");
            for a in 0..3 {
                assert!((force[a] - form[a]).abs() <= 1e-10 * size.max(1e-30), "axis {a} at s {s}: {} against {}", force[a], form[a]);
            }
        }
    }
}

/// The decay is the photons' rate, one definition (module documentation, # The normalized generator): a path's decay at
/// its middle on a state, ψ†i(H − H†)ψ with the normalized generator's part S⁻¹(Λ + Λ_R E + E†Λ_R† + E†Λ_B E)S⁻¹, is
/// Σ_classes ‖L ψ‖² of the photons' operators L = (J(κ) + c J2 E) S⁻¹ (photon.rs) at the same point and momentum -- the
/// survival a step's hazard reads is the jumps' own -- to the rounding of the sums: at the series' order 1 (the 1D PGC at
/// 3 G, refs/requests/PGC1D_TILT_0, q ≈ 0.03) and on PGC3D, the series' first order and the normalization's kept
/// (`Unit::Series`, `Unit::Normalization`), each against the plan without its normalized generator, moving at its reach's speeds.
#[test]
fn the_decay_is_the_photons_rate_with_the_normalized_generator() {
    use crate::model::photon::Gradients;
    for (file, speed) in [("PGC1D_TILT_0.json", 0.09), ("PGC3D_1ms.json", 0.04)] {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join(file);
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let bare = Model::new(&request, &light, &[]).unwrap();
        let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
        let (r, v) = ([0.05, -0.1, 0.03], [speed, 0.5 * speed, 0.7 * speed]);
        let points = [(r, v), ([0.0; 3], [0.0; 3])];
        let prior = Model::new(&request, &light, &kept).unwrap().prior(&points);
        // The series' first order and the normalization's kept: every part of the normalized generator.
        let mut model = Model::new(&request, &light, &[kept, prior, vec![Unit::Series, Unit::Normalization { order: 1 }]].concat()).unwrap();
        model.referenced(&points);
        let cluster = model.cluster_of(GROUND - 1);
        let plan = model.plan(cluster, &model.running());
        let n = plan.states;
        assert!(plan.fourth.as_ref().is_some_and(|f| f.bends()), "{file}: the normalized generator bends the decay: {:?}", plan.fourth.as_ref().map(|f| f.orders));
        let without = Arc::new(Plan { fourth: None, ..(*plan).clone() });
        let kappa = v.map(|x| x / model.hbar_over_mass);
        let mut stream = Stream::new(9, Purpose::Initial, 0, 0);
        for (label, p) in [("with", plan.clone()), ("without", without)] {
            for _ in 0..3 {
                let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
                let middle = model.middle(&p, r, 0.0, None);
                let on = middle.path(kappa, v);
                let mut h = Mat::zeros(n, n);
                middle.generator(&on, 0.0, &mut h);
                let mut rate = 0.0;
                for x in 0..n {
                    for y in 0..n {
                        rate += (psi[x].conj() * (h.data[x * n + y] - h.data[y * n + x].conj()) * C64::I * psi[y]).re;
                    }
                }
                let photons: f64 = model.photons_of(&p, None, &crate::model::Point { r, kappa, t: 0.0 }, &psi, Gradients::None).0.iter().map(|ph| ph.weight()).sum();
                println!("{file} {label}: decay {rate:.6e}, photons {:.3e} apart", (rate - photons) / rate);
                assert!((rate - photons).abs() <= rounding(4 * n * n) * rate, "{file} {label}: the decay {rate} against the photons' {photons}");
            }
        }
    }
}

/// The decay is the jumps' product on random fields and polarizations (the class test; module documentation, # The
/// excited amplitude; photon.rs, # The jumps and the decay): of one 𝒯 the generator's anti-Hermitian part is
/// −(i/2) S⁻¹ Σ 𝒯†D𝒯 S⁻¹ and the photons' operators are C_a 𝒯 S⁻¹ -- the compression's identity i(G − G†) = Σ L†L, which
/// holds for any 𝒯 and any Hermitian S. On every model of the tests -- bare and folded drives, explicit states, kept
/// couplings and coherences (merged classes), clusters of several reference classes --, as they come and with the
/// series' first order and the normalization kept (`Unit::Series`, `Unit::Normalization`), at random points across the
/// light's pattern (the fields' amplitudes and polarizations there), random momenta and random states, from the exact
/// elimination and from a scene's tables between their nodes: ψ†i(H − H†)ψ of the generator at the point (its path's
/// middle) is Σ_classes Σ_a ‖u_a‖² of its photons there, to the rounding of their sums.
#[test]
fn the_decay_is_the_jumps_product_on_random_fields_and_polarizations() {
    use crate::model::photon::Gradients;
    let (mut checked, mut largest) = (0, 0.0_f64);
    let mut check = |name: &str, model: &Model, tables: Option<&crate::model::tabulate::Tables>, stream: &mut Stream| {
        let lights = model.running();
        for cluster in 0..model.clusters.len() {
            let (plan, n) = (model.plan(cluster, &lights), model.clusters[cluster].states.len());
            let table = tables.and_then(|t| t.of(cluster, &lights));
            for _ in 0..6 {
                let (r, t, kappa, velocity, _) = drawn(model, stream);
                let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
                let middle = model.middle(&plan, r, t, table);
                let path = middle.path(kappa, velocity);
                let mut h = Mat::zeros(n, n);
                middle.generator(&path, 0.0, &mut h);
                // The rate's rounding: of the entries' sums, the Hermitian part's whole size among them.
                let (mut rate, mut size) = (0.0, 0.0);
                for x in 0..n {
                    for y in 0..n {
                        let z = psi[x].conj() * (h.data[x * n + y] - h.data[y * n + x].conj()) * C64::I * psi[y];
                        rate += z.re;
                        size += psi[x].abs() * (h.data[x * n + y].abs() + h.data[y * n + x].abs()) * psi[y].abs();
                    }
                }
                let at = crate::model::Point { r, kappa, t };
                let photons_at = |table| -> f64 { model.photons_of(&plan, table, &at, &psi, Gradients::None).0.iter().map(|ph| ph.weight()).sum() };
                let photons = photons_at(table);
                let apart = (rate - photons).abs();
                // From a table the decay is its interpolated pair forms and the photons its interpolated responses' jumps:
                // apart by the table's measured error, each against the exact elimination at the point (tabulate.rs, # The
                // table's error: its whole read).
                let tabulated = match table {
                    None => 0.0,
                    Some(_) => {
                        let exact = model.middle(&plan, r, t, None);
                        let mut g = Mat::zeros(n, n);
                        exact.generator(&exact.path(kappa, velocity), 0.0, &mut g);
                        let rate_exact: f64 = (0..n).flat_map(|x| (0..n).map(move |y| (x, y))).map(|(x, y)| (psi[x].conj() * (g.data[x * n + y] - g.data[y * n + x].conj()) * C64::I * psi[y]).re).sum();
                        (rate - rate_exact).abs() + (photons - photons_at(None)).abs()
                    }
                };
                assert!(apart <= rounding(8 * n * n + 4 * EXCITED * EXCITED) * size.max(photons) + tabulated, "{name}, cluster {cluster} at {r:?}, κ {kappa:?}: the decay {rate:e} against the photons' {photons:e} (the table's error {tabulated:e})");
                largest = largest.max(if photons > 0.0 { apart / photons } else { 0.0 });
                checked += 1;
            }
        }
    };
    for (index, (name, model)) in models().into_iter().enumerate() {
        let mut stream = Stream::new(12, Purpose::Initial, index as u64, 0);
        check(name, &model, None, &mut stream);
        let bent = model.rekept(&[model.kept.clone(), vec![Unit::Series, Unit::Normalization { order: 1 }]].concat()).unwrap();
        check(&format!("{name}, normalized"), &bent, None, &mut stream);
    }
    for (index, name) in ["R3_DOPPLER", "R4_TW1"].into_iter().enumerate() {
        let (request, model) = (scene(name), model(name, &others, &|_| false));
        let state = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let p: [f64; GROUND] = std::array::from_fn(|b| if b == state { 1.0 } else { 0.0 });
        let reach = model.reach(&model.trap(&p, &model.prepared()).unwrap(), &p, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
        let tables = model.tabulate(&reach, 1 << 12);
        check(&format!("{name} tabulated"), &model, Some(&tables), &mut Stream::new(12, Purpose::Initial, 100 + index as u64, 0));
    }
    eprintln!("{checked} states: the decay against the photons' rate within {largest:.2e} of it");
    assert!(checked > 0);
}

/// The beating groups (assemble.rs, # Exact tones) are the generator's along a path: on every test model's clusters whose
/// generator has no normalized part (whose forms' products are in no group), Ĥ(s) less every candidate pair's
/// e^{iνs}G⁺ + e^{−iνs}G⁻ (`Middle::beats`, `Middle::envelopes`) is the same at every time along the path -- what is left
/// is the static pieces and the frames' rates --, and so is the force's form with a Hermitian ρ less every pair's
/// `Middle::tone_force` on ρ at its phase (σ = e^{iνs}ρ), to the rounding of the sums: the entries' own bound (`chain`)
/// on their moduli and the envelopes', and their phases' arguments ν s.
#[test]
fn the_beating_groups_are_the_generator_s_along_a_path() {
    let (mut groups, mut clusters) = (0, 0);
    for (index, (name, model)) in models().into_iter().enumerate() {
        let (g, c) = groups_along_a_path(name, index as u64, &model);
        (groups, clusters) = (groups + g, clusters + c);
    }
    eprintln!("{groups} candidate tone pairs on {clusters} clusters");
    assert!(groups > 0);
}

/// A beating group's pieces are its terms' own entries (assemble.rs `Term::held`): a dense term is a block's entries at
/// one tone that one flag holds, laid on their span, and a block's flags hold different entries -- the decay's flag
/// where a Hermitian one holds none, a near form's flag beside a constant one. A group that took every entry of the
/// block at the tone into each flag's term read past the values of a flag that holds fewer: S2Q's first model and its
/// first attempt's keep (the probe's F′ = 3 block for F = 2 m = −2, −1, 0 and the normalization's first order) panicked
/// in a step's envelopes (the run at index 6 to 94 of a term's 4 values; this test at 15 of 8). On both, every
/// cluster's every candidate pair's envelopes and
/// tone forces are made, and where the generator has no normalized part the groups are the generator's along a path
/// (`groups_along_a_path`).
#[test]
fn a_beating_group_s_pieces_are_its_terms_own_entries() {
    use crate::model::Component;
    use crate::request::GROUND;
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("S2Q.json");
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let req = Request::from_native(&line).unwrap();
    let light = Light::new(&req);
    let chief = (0..GROUND).max_by(|&a, &b| req.initial_density[(a, a)].re.total_cmp(&req.initial_density[(b, b)].re)).unwrap();
    let (first, mut kept, _, _) = Model::closed_at(&req, &light, vec![], chief, req.temperature_uk);
    let state = |label: &str| req.states.iter().position(|s| s == label).expect("a state of the request");
    let block = first.blocks().iter().position(|b| b.contains(&(state("D2-3-0") - GROUND))).expect("the F′ = 3 block");
    let probe = (0..light.frequencies.len()).find(|&f| light.frequencies[f].beams.iter().any(|&l| req.beams[l].id == "laser-probe")).expect("the probe's carrier");
    for m in ["-2", "-1", "0"] {
        kept.push(Unit::Block { block, component: Component { frequency: probe, counter: false }, grounds: crate::model::elim::ground_set(&[state(&format!("ground-2-{m}"))]) });
    }
    kept.push(Unit::Normalization { order: 1 });
    let (keep, _, _, _) = Model::closed_at(&req, &light, kept, chief, req.temperature_uk);
    let (mut groups, mut clusters) = (0, 0);
    for (index, (name, model)) in [("S2Q's first model", first), ("S2Q's first attempt's keep", keep)].into_iter().enumerate() {
        let (g, c) = groups_along_a_path(name, index as u64, &model);
        (groups, clusters) = (groups + g, clusters + c);
    }
    eprintln!("S2Q: {groups} candidate tone pairs on {clusters} clusters");
    assert!(groups > 0);
}

/// `the_beating_groups_are_the_generator_s_along_a_path` on one model, its stream's index `index`: every cluster's
/// candidate pairs' envelopes made, and where its generator has no normalized part the property checked; the pairs and
/// the clusters checked.
fn groups_along_a_path(name: &str, index: u64, model: &Model) -> (usize, usize) {
    let (mut groups, mut clusters) = (0, 0);
    {
        let mut stream = Stream::new(1, Purpose::Initial, index, 0);
        let lights = model.running();
        for cluster in 0..model.clusters.len() {
            let n = model.clusters[cluster].states.len();
            let plan = model.plan(cluster, &lights);
            let (r, t, kappa, velocity, times) = drawn(model, &mut stream);
            let rho = hermitian(n, &mut stream);
            let middle = model.assemble(cluster, r, t, &lights);
            let path = middle.path(kappa, velocity);
            let (mut beats, mut keys) = (vec![], vec![]);
            middle.beats(&path, &mut beats, &mut keys);
            // A label is the beat exactly: two pairs of one label are one beat, and the pairs' labels are distinct.
            assert_eq!(keys.len(), beats.len());
            for (a, ka) in keys.iter().enumerate() {
                assert!(keys[..a].iter().all(|kb| kb != ka), "two pairs of one label");
            }
            let envelopes: Vec<(Mat, Mat)> = (0..beats.len())
                .map(|m| {
                    let (mut plus, mut minus) = (Mat::zeros(n, n), Mat::zeros(n, n));
                    middle.envelopes(&path, m, 0.0, &mut plus, &mut minus);
                    (plus, minus)
                })
                .collect();
            if plan.fourth.as_ref().is_some_and(|fourth| fourth.bends() || fourth.held()) {
                let (re, im): (Vec<C64>, Vec<C64>) = rho.iter().map(|z| (*z, C64::ZERO)).unzip();
                for m in 0..beats.len() {
                    let tone = middle.tone_force(&path, m, &[(0.0, &re, &im)]);
                    assert!(tone.iter().all(|x| x.is_finite()), "{name}, cluster {cluster}: a pair's tone force is a number");
                }
                assert!(envelopes.iter().all(|(p, q)| p.data.iter().chain(&q.data).all(|z| z.re.is_finite() && z.im.is_finite())), "{name}, cluster {cluster}: the envelopes are numbers");
                continue;
            }
            // The Hermitian part: the decay's beats are the static group's, made of the node's 𝒯 (module documentation, #
            // The excited amplitude; spectral.rs, # Which beats are tones), none of a group's.
            let rest = |s: f64| -> Mat {
                let mut h = Mat::zeros(n, n);
                middle.generator(&path, s, &mut h);
                for (beat, (plus, minus)) in beats.iter().zip(&envelopes) {
                    for i in 0..n * n {
                        h.data[i] -= C64::cis(beat * s) * plus.data[i] + C64::cis(-beat * s) * minus.data[i];
                    }
                }
                Mat::from_fn(n, n, |x, y| (h[(x, y)] + h[(y, x)].conj()) * 0.5)
            };
            let form = |s: f64| -> [f64; 3] {
                let mut f = middle.force(&path, s, &rho);
                for (m, beat) in beats.iter().enumerate() {
                    let (sin, cos) = (beat * s).sin_cos();
                    let (re, im): (Vec<C64>, Vec<C64>) = rho.iter().map(|z| (*z * cos, *z * sin)).unzip();
                    let tone = middle.tone_force(&path, m, &[(s, &re, &im)]);
                    (0..3).for_each(|a| f[a] -= tone[a]);
                }
                f
            };
            let links = chain(&model, &plan, written_out(&model, cluster, r, t, kappa, velocity, 0.0, false).terms);
            let (h0, f0) = (rest(0.0), form(0.0));
            for s in times {
                let (h, f, sizes) = (rest(s), form(s), written_out(&model, cluster, r, t, kappa, velocity, s, true));
                let size0 = written_out(&model, cluster, r, t, kappa, velocity, 0.0, true);
                // A phase's argument, a tone's Ω(t + s) and a pair's ν s, rounded with its sum and its product.
                let fastest = plan.tones.iter().fold(0.0_f64, |m, omega| m.max(omega.abs())) * (t.abs() + s.abs()) + beats.iter().fold(0.0_f64, |m, b| m.max((b * s).abs()));
                for i in 0..n * n {
                    let tones: f64 = envelopes.iter().map(|(p, q)| p.data[i].abs() + q.data[i].abs()).sum();
                    let size = sizes.hermitian[i][0].re + sizes.decay[i].re + size0.hermitian[i][0].re + size0.decay[i].re + 2.0 * tones;
                    let (apart, bound) = ((h.data[i] - h0.data[i]).abs(), rounding(links + 4) * size + 4.0 * crate::units::UNIT_ROUNDOFF * fastest * tones);
                    assert!(apart <= bound, "{name}, cluster {cluster}, s {s}, entry ({}, {}): what the groups leave moved by {apart:e} (bound {bound:e})", i / n, i % n);
                }
                for a in 0..3 {
                    let size = form_size(&sizes, a, &rho, n) + form_size(&size0, a, &rho, n);
                    let (apart, bound) = ((f[a] - f0[a]).abs(), 4.0 * (rounding(2 * links + n * n + 8) + crate::units::UNIT_ROUNDOFF * fastest) * size);
                    assert!(apart <= bound, "{name}, cluster {cluster}, s {s}, axis {a}: what the groups leave of the force's form moved by {apart:e} (bound {bound:e})");
                }
            }
            (groups, clusters) = (groups + beats.len(), clusters + 1);
        }
    }
    (groups, clusters)
}

/// The readout inverts the generator's own S (readout.rs `Model::readout_of`; `Model::quartic`): the 1D PGC at 3 G
/// (PGC1D_TILT_0) with every near light at the atom, the prior's units, and the series' first order and the
/// normalization's kept, so that S bends the state. At two points the readout's S⁻¹ -- Σ_{j≤J} b_j X^j of the one
/// metric, the near part of 𝒯's at the responses there and the plan's field components -- is the matrix the photons and
/// the landing take
/// (`Model::quartic`, with its gradients), to rounding; and the ground populations the readout reads of a state ψ are
/// those of x = S⁻¹ψ, in ratio (the readout divides them by the sum with the folded excited populations).
#[test]
fn the_readout_inverts_the_generator_s_own_normalization() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("PGC1D_TILT_0.json");
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let request = Request::from_native(&line).unwrap();
    let light = Light::new(&request);
    let bare = Model::new(&request, &light, &[]).unwrap();
    let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
    let points = [([0.05, -0.1, 0.03], [0.09, 0.045, 0.063]), ([0.0; 3], [0.0; 3])];
    let prior = Model::new(&request, &light, &kept).unwrap().prior(&points);
    let mut model = Model::new(&request, &light, &[kept, prior, vec![Unit::Series, Unit::Normalization { order: 1 }]].concat()).unwrap();
    model.referenced(&points);
    let cluster = model.cluster_of(GROUND - 1);
    let plan = model.plan(cluster, &model.running());
    let fourth = plan.fourth.as_ref().filter(|f| f.bends()).expect("S bends the state");
    let (n, size) = (fourth.n, fourth.n * fourth.n);
    assert!(n > 1, "a cluster of several states");
    let mut stream = Stream::new(11, Purpose::Initial, 0, 0);
    for r in [[0.05, -0.1, 0.03], [-0.21, 0.13, 0.4]] {
        // The generator's at the point, as the photons and the landing take it.
        let mut site = Held::<Site>::take();
        let table = model.pointed(&plan, None, r, &mut site);
        model.quartic(&plan, fourth, table, Some(r), &mut site, false, false);
        let generator: Vec<C64> = Model::quartic_read(|q| q.data[BENT * size..(BENT + 1) * size].to_vec());
        let identity = generator.iter().enumerate().fold(0.0_f64, |m, (i, z)| m.max((*z - if i % (n + 1) == 0 { C64::ONE } else { C64::ZERO }).abs()));
        println!("r {r:?}: S⁻¹ − 1 of size {identity:.2e}");
        assert!(identity > 0.0, "S bends the state");
        // The populations of ψ are x's, x = S⁻¹ψ.
        let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
        let x: Vec<C64> = (0..n).map(|i| (0..n).map(|j| generator[i * n + j] * psi[j]).sum()).collect();
        let read = model.readout(cluster, &crate::model::Point { r, kappa: [0.0; 3], t: 0.0 }, &psi);
        let c = &model.clusters()[cluster];
        let (first, rest) = (c.states[0], &c.states[1..c.ground]);
        for (p, &s) in rest.iter().enumerate() {
            let (shown, own) = (read[s] / read[first], x[p + 1].norm_sqr() / x[0].norm_sqr());
            assert!((shown - own).abs() <= 1e-12 * own.max(1.0), "ground state {s}: read {shown} against x's {own}");
        }
    }
}

/// The plan's counted work by piece (run by name: `SCENE=<name or path> … a_scene_s_plan_work -- --ignored --nocapture`):
/// per cluster of the scene's closed model, the complex multiply-adds a node's and a point's pieces count -- the dense
/// terms' node cost, 𝒯 (`excite`), its first order, the decay, the metric, the normalized part, a scattering's classes.
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn a_scene_s_plan_work() {
    use crate::model::photon::Gradients;
    let name = std::env::var("SCENE").expect("SCENE: a reference scene's name or a request file's path");
    let request = crate::model::testing::scene(&name);
    let light = crate::field::Light::new(&request);
    let state = (0..crate::request::GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let (model, ..) = Model::closed_at(&request, &light, vec![], state, request.temperature_uk);
    let running = model.running();
    for k in 0..model.clusters().len() {
        let plan = model.plan(k, &running);
        let fourth = plan.fourth.as_ref();
        println!(
            "cluster {k}: {} | node {:?} amplitudes {:.0} first {:.0} node normal {:?} quartic {:.0} held force {:.0} | classes (all) {:.0} (none) {:.0} amplitudes {:.0} | costs {:?}",
            plan.described(),
            plan.node,
            plan.amplitudes_work(1, false),
            plan.amplitudes_work(1, true) - plan.amplitudes_work(1, false),
            fourth.map(|f| plan.node_normal_work(f)),
            fourth.map_or(0.0, |f| plan.quartic_work(f, None, false)),
            fourth.map_or(0.0, |f| plan.held_force_work(f, 3)),
            plan.classes_work(Gradients::All),
            plan.classes_work(Gradients::None),
            plan.amplitudes_work(4, false),
            plan.costs(3)
        );
    }
}
