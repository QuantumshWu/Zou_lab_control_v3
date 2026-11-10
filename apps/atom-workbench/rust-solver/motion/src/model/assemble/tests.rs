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
    let units = Model::new(&request, &light, &lifted).unwrap().units();
    let kept: Vec<Unit> = lifted.into_iter().chain(units.into_iter().filter(keep)).collect();
    Model::new(&request, &light, &kept).unwrap()
}

/// Every drive of every frequency but the trap's.
fn others(light: &Light, unit: &Unit) -> bool {
    matches!(unit, Unit::Bare { component, .. } if component.frequency != light.trap)
}

/// A scene's model with one frequency's drives at all three levels: of the light that is not the trap's, the drives of the
/// stretched state explicit, and of the others those into every second excited state folded at the atom, the rest bare.
fn three_levels(name: &str) -> Model {
    let request = scene(name);
    let light = Light::new(&request);
    let bare = Model::new(&request, &light, &[]).unwrap();
    let mut kept = vec![];
    for frequency in (0..light.frequencies.len()).filter(|&f| f != light.trap) {
        for unit in bare.bare_units(Component { frequency, counter: false }) {
            let Unit::Bare { component, excited, ground } = unit else { unreachable!() };
            if ground == GROUND - 1 {
                kept.push(Unit::Drive { component, excited, ground });
            } else if excited % 2 == 0 {
                kept.push(unit);
            }
        }
    }
    let model = Model::new(&request, &light, &kept).unwrap();
    for level in [Level::Bare, Level::Atom, Level::Explicit] {
        assert!(model.driven.iter().any(|&(c, e, g)| c.frequency != light.trap && model.level(c, e, g) == level), "{name}: no drive {level:?}");
    }
    assert!(model.clusters.iter().any(|c| c.states.len() - c.ground > 1), "{name}: explicit states that share folded drives");
    model
}

/// The models the tests run on, every kind of term among them. With every drive bare: one beam (still clusters) and a
/// standing wave of two (beating along a path); crossed beams with their static couplings kept (clusters of four states in
/// Doppler frames that turn); two frequencies with every coupling of two states kept, their beats' too (tones at rest). With
/// the drives of the light that is not the trap's folded at the atom: the drives of the stretched state kept (explicit states
/// with their shifts, drives and own photons; one carrier, two of two beats, and eight explicit states); every coupling of
/// two states kept -- of one carrier with the trap's bare light (a bare coupling against a response at the atom), of one
/// with the photons' coherences (merged classes), of two carriers of several beams (all eight ground states). And one
/// frequency's drives at all three levels (`three_levels`).
fn models() -> Vec<(&'static str, Model)> {
    let none = |_: &Unit| false;
    let nothing = |_: &Light, _: &Unit| false;
    let couplings = |u: &Unit| matches!(u, Unit::Term { .. });
    let drives = |u: &Unit| matches!(u, Unit::Drive { ground: 7, .. });
    vec![
        ("R1_TWEEZER bare", model("R1_TWEEZER", &nothing, &none)),
        ("R1_RETRO bare", model("R1_RETRO", &nothing, &none)),
        ("R1_CROSSED bare, couplings kept", model("R1_CROSSED", &nothing, &couplings)),
        ("R2_RECOIL bare, couplings kept", model("R2_RECOIL", &nothing, &couplings)),
        ("R2_RECOIL at the atom, drives of |2,2> kept", model("R2_RECOIL", &others, &drives)),
        ("R2_RECOIL at the atom, couplings kept", model("R2_RECOIL", &others, &couplings)),
        ("R3_DOPPLER at the atom, couplings and coherences kept", model("R3_DOPPLER", &others, &|u| matches!(u, Unit::Term { .. } | Unit::Coherence { .. }))),
        ("R4_TW1 at the atom, couplings kept", model("R4_TW1", &others, &couplings)),
        ("R5_PUMP at the atom, drives of |2,2> kept", model("R5_PUMP", &others, &drives)),
        ("R6_RAMAN_CO at the atom, drives of |2,2> kept", model("R6_RAMAN_CO", &others, &drives)),
        ("R8_R5051 at the atom, couplings kept", model("R8_R5051", &others, &couplings)),
        ("R8_R5051 at three levels", three_levels("R8_R5051")),
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
///   denominator against the other leg's folded coupling; the explicit drives Σ_k e_k U_k and their adjoints;
/// - the decay −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}, a class's jump the sum of its members' -- a component's bare drives'
///   e_k J_{a,k} (e_k* counter-rotating), a carrier's C_a R̃ with its own bare drives' e_k J_{a,k}, an explicit state's
///   −C_a --, each at its relative beat;
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
    let far = model.far_hamiltonian(&amplitudes, &lights, 1, &model.coherent_dressing());
    let responses = model.eliminate(cluster, &far, &amplitudes, &lights, 1, &model.coherent_dressing());
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
            for x in 0..g {
                for y in 0..g {
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
        for x in g..n {
            for y in g..n {
                let (e, f) = (c.states[x], c.states[y]);
                let sum = (0..GROUND).fold(NONE, |sum, b| {
                    let half = |state: usize| -> Jet {
                        let mut gap: Vec<f64> = far.iter().map(|part| part[(state, state)].re - part[(b, b)].re).collect();
                        gap[0] -= cc.detuning;
                        let reciprocal = jet::reciprocal(&gap);
                        std::array::from_fn(|part| real(0.5 * reciprocal[part]))
                    };
                    let right = times(times(at_the_atom(e, b), conjugate(folded(w, f, b))), half(e));
                    let left = times(times(folded(w, e, b), conjugate(at_the_atom(f, b))), half(f));
                    plus(sum, plus(right, left))
                });
                add(x, y, scaled(sum, phase(&zero, x, y)));
            }
            for q in 0..g {
                if model.level(component, c.states[x], c.states[q]) == Level::Explicit {
                    let drive = coupling(w, &model.tones[w].basis[0], c.states[x], c.states[q]);
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
                            let bare = model.tones[f].jumps[0].get(component).map_or(C64::ZERO, |jumps| jumps[a][(b, state)]);
                            let at_the_atom: C64 = if sp < g { (0..EXCITED).map(|i| leaf(model.emission[a][(b, GROUND + i)]) * (e * leaf(first[(i, sp)]) + weighted * leaf(-second[(i, sp)]))).sum() } else { C64::ZERO };
                            at_the_atom + e * leaf(bare)
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
