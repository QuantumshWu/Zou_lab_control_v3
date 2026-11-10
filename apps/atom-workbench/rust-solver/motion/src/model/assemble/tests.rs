//! A path's generator, force and tones (assemble.rs `Middle`, `Path`) against the model's definitions written out on dense
//! matrices (`written_out`): none of the plan's blocks, flags, near forms, slots or tones.

use super::*;
use crate::field::Light;
use crate::linalg::norm3;
use crate::random::{Purpose, Stream};
use crate::request::Request;
use crate::step::Middle as _;
use crate::units::rounding;

fn scene(name: &str) -> Request {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    Request::from_native(&line).unwrap()
}

/// A scene's model with every frequency but the trap's a near carrier, and of the units it then folds those `keep` takes kept.
fn near(name: &str, keep: &dyn Fn(&Unit) -> bool) -> Model {
    let request = scene(name);
    let light = Light::new(&request);
    let carriers: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).map(|frequency| Unit::Far { component: Component { frequency, counter: false } }).collect();
    let units = Model::new(&request, &light, &carriers).unwrap().units();
    let kept: Vec<Unit> = carriers.into_iter().chain(units.into_iter().filter(keep)).collect();
    Model::new(&request, &light, &kept).unwrap()
}

/// A scene's model with every light folded far, and of the units it then folds those `keep` takes kept.
fn far(name: &str, keep: &dyn Fn(&Unit) -> bool) -> Model {
    let request = scene(name);
    let light = Light::new(&request);
    let kept: Vec<Unit> = Model::new(&request, &light, &[]).unwrap().units().into_iter().filter(keep).collect();
    Model::new(&request, &light, &kept).unwrap()
}

/// The models the tests run on, every kind of term among them: one far beam (still clusters) and a standing wave of two
/// (beating along a path); crossed far beams with their static couplings kept (clusters of four states in Doppler frames that
/// turn); two far frequencies with their beats kept (tones at rest); near light with its drives of the stretched state kept
/// (explicit states with their shifts, drives and own photons; one carrier, two of two beats, and eight explicit states);
/// near light with its couplings of the ground states kept, of one carrier with the photons' coherences (merged classes) and
/// of two carriers of several beams (the clusters the hyperfine levels, and all eight ground states).
fn models() -> Vec<(&'static str, Model)> {
    let none = |_: &Unit| false;
    let couplings = |u: &Unit| matches!(u, Unit::Term { .. });
    let drives = |u: &Unit| matches!(u, Unit::Drive { ground: 7, .. });
    vec![
        ("R1_TWEEZER far", far("R1_TWEEZER", &none)),
        ("R1_RETRO far", far("R1_RETRO", &none)),
        ("R1_CROSSED far, static couplings kept", far("R1_CROSSED", &couplings)),
        ("R2_RECOIL far, beats and couplings kept", far("R2_RECOIL", &|u| matches!(u, Unit::FarBeat { .. } | Unit::Term { .. }))),
        ("R2_RECOIL near, drives of |2,2> kept", near("R2_RECOIL", &drives)),
        ("R3_DOPPLER near, couplings and coherences kept", near("R3_DOPPLER", &|u| matches!(u, Unit::Term { .. } | Unit::Coherence { .. }))),
        ("R4_TW1 near, couplings kept", near("R4_TW1", &couplings)),
        ("R5_PUMP near, drives of |2,2> kept", near("R5_PUMP", &drives)),
        ("R6_RAMAN_CO near, drives of |2,2> kept", near("R6_RAMAN_CO", &drives)),
        ("R8_R5051 near, couplings kept", near("R8_R5051", &couplings)),
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
/// - the states with every light off, in their frames; each frequency's far form Σ_k'k e_k'* e_k A_k'k; each kept far beat's
///   cross forms a_l₂* a_l₁ C_l₂l₁ (a_l₁* a_l₂ of two counter-rotating components) with their adjoints;
/// - per pair of near carriers −½(V'†R̃ + R̃'†V) on the ground places whose beat the frames or a kept unit hold, V = Σ_k e_k U^f_k
///   and R̃ = Σ_k (e_k T_k − ẽ_k T2_k) of the elimination at the middle; the explicit states' van Vleck shift by the folded
///   drives; the kept drives Σ_k e_k U_k and their adjoints;
/// - the decay −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}, a class's jump the sum of its members' -- a folded far component's
///   e_k J_{a,k} (e_k* counter-rotating), a near carrier's C_a R̃, an explicit state's −C_a --, each at its relative beat;
/// - every entry at its beat's phase at t + s and in its two states' Doppler frames, with their diag(K_s·v).
/// With `moduli` every factor is its modulus: the same sums with nothing cancelling, the scale of their rounding.
fn written_out(model: &Model, cluster: usize, r: [f64; 3], t: f64, kappa: [f64; 3], velocity: [f64; 3], s: f64, moduli: bool) -> Written {
    let c = &model.clusters[cluster];
    let (n, g) = (c.states.len(), c.ground);
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
    // The states with every light off, in their frames; the far forms; the kept far beats.
    for x in 0..n {
        for y in 0..n {
            add(x, y, constant(leaf(model.base[(c.states[x], c.states[y])]) * phase(&zero, x, y)));
        }
        add(x, x, constant(real(-model.omega(&c.frames[x]))));
    }
    for (w, (f, tone)) in model.light.frequencies.iter().zip(&model.tones).enumerate() {
        let rank = f.rank();
        for kl in 0..rank {
            for kr in 0..rank {
                let product = times(conjugate(field(w, kl, false)), field(w, kr, false));
                for x in 0..n {
                    for y in 0..n {
                        add(x, y, scaled(product, leaf(tone.forms[kl * rank + kr][(c.states[x], c.states[y])]) * phase(&zero, x, y)));
                    }
                }
            }
        }
    }
    for unit in &model.kept {
        let Unit::FarBeat { first, second } = unit else { continue };
        let (f1, f2) = (&model.light.frequencies[first.frequency], &model.light.frequencies[second.frequency]);
        let beat = model.axes.component(&model.light, *second).plus(-1, &model.axes.component(&model.light, *first));
        let cross = model.far_cross(*first, *second);
        for (j, &l2) in f2.beams.iter().enumerate() {
            for (i, &l1) in f1.beams.iter().enumerate() {
                let product = if first.counter { times(conjugate(on[l1]), on[l2]) } else { times(conjugate(on[l2]), on[l1]) };
                for x in 0..n {
                    for y in 0..n {
                        let entry = cross[j * f1.beams.len() + i][(c.states[x], c.states[y])];
                        add(x, y, scaled(product, leaf(entry) * phase(&beat, x, y)));
                        add(y, x, scaled(conjugate(product), leaf(entry.conj()) * phase(&beat.negated(), y, x)));
                    }
                }
            }
        }
    }
    // The elimination at the middle with its gradient: a carrier's coupling on its field (folded, or as it is) and its response
    // on the path.
    let far = model.far_hamiltonian(&amplitudes, &lights, 1);
    let responses = model.eliminate(cluster, &far, &amplitudes, &lights, 1, &model.coherent_dressing());
    let coupling = |carrier: usize, folded: bool, e: usize, b: usize| -> Jet {
        let cc = &model.carriers[carrier];
        let basis = if folded { &cc.basis } else { &model.tones[cc.frequency].basis[0] };
        (0..basis.len()).fold(NONE, |sum, k| plus(sum, scaled(field(cc.frequency, k, false), leaf(basis[k][(e, b)]))))
    };
    let response = |carrier: usize, e: usize, y: usize| -> Jet {
        let cc = &model.carriers[carrier];
        (0..cc.basis.len()).fold(NONE, |sum, k| {
            let first: Jet = std::array::from_fn(|part| leaf(responses.first[carrier][k][part][(e, y)]));
            let second: Jet = std::array::from_fn(|part| leaf(-responses.second[carrier][k][part][(e, y)]));
            plus(sum, plus(times(field(cc.frequency, k, false), first), times(field(cc.frequency, k, true), second)))
        })
    };
    for (cj, cp) in model.carriers.iter().enumerate() {
        for (ci, cc) in model.carriers.iter().enumerate() {
            let beat = cp.beat.plus(-1, &cc.beat);
            let (from, to) = (Component { frequency: cc.frequency, counter: false }, Component { frequency: cp.frequency, counter: false });
            for x in 0..g {
                for y in 0..g {
                    if !model.term_included(cluster, x, y, from, to, &beat) {
                        continue;
                    }
                    let sum = (0..EXCITED).fold(NONE, |sum, e| {
                        let there = times(conjugate(coupling(cj, true, GROUND + e, c.states[x])), response(ci, e, y));
                        let back = times(conjugate(response(cj, e, x)), coupling(ci, true, GROUND + e, c.states[y]));
                        plus(sum, plus(there, back))
                    });
                    add(x, y, scaled(sum, real(-0.5) * phase(&beat, x, y)));
                }
            }
        }
    }
    for (ci, cc) in model.carriers.iter().enumerate() {
        for x in g..n {
            for y in g..n {
                let (e, f) = (c.states[x], c.states[y]);
                let sum = (0..GROUND).fold(NONE, |sum, b| {
                    let halves = [e, f].iter().fold(NONE, |halves, &state| {
                        let mut gap: Vec<f64> = far.iter().map(|part| part[(state, state)].re - part[(b, b)].re).collect();
                        gap[0] -= cc.detuning;
                        let reciprocal = jet::reciprocal(&gap);
                        plus(halves, std::array::from_fn(|part| real(0.5 * reciprocal[part])))
                    });
                    plus(sum, times(times(coupling(ci, true, e, b), conjugate(coupling(ci, true, f, b))), halves))
                });
                add(x, y, scaled(sum, phase(&zero, x, y)));
            }
            for q in 0..g {
                if model.drive_kept(ci, c.states[x], c.states[q]) {
                    let drive = coupling(ci, false, c.states[x], c.states[q]);
                    add(x, q, scaled(drive, phase(&cc.beat.negated(), x, q)));
                    add(q, x, scaled(conjugate(drive), phase(&cc.beat, q, x)));
                }
            }
        }
    }
    // The decay: per class and axis its jump on the landing cluster's ground places, the sum of its members'.
    let mut decay = vec![C64::ZERO; n * n];
    for class in &model.classes[cluster] {
        let landing = &model.clusters[class.landing];
        for a in 0..3 {
            let mut jump = Mat::zeros(landing.ground, n);
            for member in &class.members {
                let own = leaf(C64::cis(model.omega(&member.phase) * (t + s)));
                for &(bp, sp) in &member.entries {
                    let (b, state) = (landing.states[bp], c.states[sp]);
                    let amplitude = match model.sources[member.source].origin {
                        Origin::Far { frequency, side, component } => {
                            let e = field(frequency, component, false)[0];
                            (if side == 1 { e.conj() } else { e }) * leaf(model.tones[frequency].jumps[side][component][a][(b, state)])
                        }
                        Origin::Near { carrier, component } => {
                            let f = model.carriers[carrier].frequency;
                            let (e, weighted) = (field(f, component, false)[0], field(f, component, true)[0]);
                            let (first, second) = (&responses.first[carrier][component][0], &responses.second[carrier][component][0]);
                            (0..EXCITED).map(|i| leaf(model.emission[a][(b, GROUND + i)]) * (e * leaf(first[(i, sp)]) + weighted * leaf(-second[(i, sp)]))).sum()
                        }
                        Origin::Explicit { state: e } => leaf(-model.emission[a][(b, e)]),
                    };
                    jump[(bp, sp)] += amplitude * own;
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
    let rate = |x: usize| dot3(c.waves[x], velocity);
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
    2 * (beams + 2 * EXCITED) + model.sources.len() + 4 * GROUND + terms + 4 * plan.blocks.len() + SINGLE
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
/// rates are K_s·v. A still cluster's generator is the same at every time along the path, to the same rounding twice.
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
            assert!(rates.iter().zip(&model.clusters[cluster].waves).all(|(rate, k)| *rate == dot3(*k, velocity)), "{name}, cluster {cluster}: the frames' rates {rates:?}");
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
