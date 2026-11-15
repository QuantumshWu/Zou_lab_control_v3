//! The elimination with the levels at the atom on the reference scenes (refs/scenes, built by the app): the generator's
//! two-level rate against the definition's (the self-consistent fold) within the error the readings book, the Λ scheme's Raman coupling against its closed form at either level of its
//! drives, optical pumping's branching against the emission's Clebsch–Gordan weights, and the identities the model holds by
//! construction with carriers and explicit states (the decay is the photons' rate, the readout sums to one, a kept and a
//! folded drive give the same recoil second moment).
use crate::field::Light;
use crate::linalg::{C64, Mat};
use crate::model::{Beat, Component, Lights, Model, Point, Sample, Unit, GROUND, STATES};
use crate::request::Request;
use crate::step::Middle as _;

use super::testing::{agrees, at_the_atom, converged, decay_rate, own, scene, steps};

/// The frequency (an index of the light's) of the beam named `id`.
fn frequency_of(request: &Request, light: &Light, id: &str) -> usize {
    let beam = request.beams.iter().position(|b| b.id == id).unwrap();
    light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap()
}

/// A cluster's generator at a point and momentum, at rest there.
fn total(model: &Model, cluster: usize, at: &Point, lights: &Lights) -> Mat {
    let middle = model.assemble(cluster, at.r, at.t, lights);
    let n = model.clusters()[cluster].states.len();
    let mut h = Mat::zeros(n, n);
    middle.generator(&middle.path(at.kappa, [0.0; 3]), 0.0, &mut h);
    h
}

/// An excited state's decay rate Γ_e = Σ_a ‖C_a[:, e]‖².
fn width(request: &Request, e: usize) -> f64 {
    request.emission.iter().map(|c| (0..GROUND).map(|b| c[(b, e)].norm_sqr()).sum::<f64>()).sum()
}

/// The two-level fold (R2_RECOIL: σ⁺ along B, |2,2⟩ → |3′,3′⟩ the only state it reaches, the probe's drives folded at the
/// atom) as the run evaluates it errs on the definition by no more than the readings book at the state: the generator's
/// rate there -- the probe on less off, −2 Im of the cluster's one entry -- against the definition's, the self-consistent
/// fold's slow eigenvalue (elim.rs `Model::definition_fold`: Löwdin, the exact eigenvalue of the static generator), in
/// relative terms within the error the readings book on every candidate at that state (elim.rs `Model::readings`: the
/// evaluation against the definition, on the series' order, the normalization's and the nearest block) and the probe's far
/// parts (its counter-rotating and outside states' shifts, in the generator and not in the fold: 1e-5); with the series'
/// first order and the normalization's kept, within what the readings book then, and nearer. At κ = −k/2 along the probe
/// its first-order Doppler shift w = (ħ/m)(k·κ + k²/2) vanishes.
#[test]
fn a_two_level_fold_errs_on_the_definition_by_no_more_than_its_readings_book() {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = frequency_of(&request, &light, "probe");
    let b = 7;
    let lifted = at_the_atom(&request, &light, &[probe]);
    let mut errs = vec![];
    for orders in [vec![], vec![Unit::Series, Unit::Normalization { order: 1 }]] {
        let model = Model::new(&request, &light, &[lifted.clone(), orders].concat()).unwrap();
        let cluster = model.cluster_of(b);
        assert_eq!(model.clusters()[cluster].states, vec![b], "the stretched state alone");
        let k = request.beams[light.frequencies[probe].beams[0]].wave_number;
        let mut off = model.running();
        off.0[probe] = crate::model::Lit::Off;
        let units = model.candidates();
        let mut worst: f64 = 0.0;
        for r in [[0.0, 0.0, 0.0], [0.2, -0.1, 0.4], [0.5, 0.3, -0.6]] {
            let at = Point { r, kappa: [0.0, 0.0, -k / 2.0], t: 0.0 };
            let (on, without) = (total(&model, cluster, &at, &model.running()), total(&model, cluster, &at, &off));
            let rate = -2.0 * (on[(0, 0)].im - without[(0, 0)].im);
            let (fold, _) = model.definition_fold(cluster, &light.amplitudes(r)).expect("the fold exists");
            let defined = -2.0 * fold.eigen.values[0].im;
            let error = (rate / defined - 1.0).abs();
            let (read, _) = model.readings(&units, &Sample { cluster, at, psi: vec![C64::ONE] }, None);
            let booked: f64 = read.iter().map(|x| x.population).sum();
            eprintln!("orders {:?}: rate {rate:.9e} against the definition's {defined:.9e}, {error:.3e} apart; booked {booked:.3e}", model.series.orders());
            assert!(error <= booked + 1e-5, "the rate errs by {error:.3e} on the definition, beyond the {booked:.3e} booked");
            worst = worst.max(error);
        }
        errs.push(worst);
    }
    assert!(errs[1] < errs[0], "the series' first order and the normalization's take the rate nearer the definition: {errs:?}");
}

/// A frequency's exact detuning δ_ω as a beat (its index on the request's axes; no frame coordinate).
fn detuning(light: &Light, frequency: usize) -> Beat {
    let mut n: Vec<i64> = light.frequencies[frequency].index.iter().map(|&i| i as i64).collect();
    n.push(0);
    Beat::whole(n)
}

/// The coupling of |1,0⟩ ← |2,0⟩ at the beat of two frequencies, emitting `to`'s and absorbing `from`'s: δ_to − δ_from.
fn raman_unit(light: &Light, to: usize, from: usize) -> Unit {
    let (to, from) = (detuning(light, to), detuning(light, from));
    Unit::Term { a: 1, b: 5, beat: to.plus(-1, &from) }
}

/// The Λ scheme's Raman coupling (R6_RAMAN_CO: D1 σ⁺σ⁺, A from F = 1, B from F = 2, Raman-resonant between |1,0⟩ and |2,0⟩,
/// Δ ≈ −20 GHz): the coupling of the two states at the two frequencies' beat is kept by the prior whatever the level of
/// the light's drives (elim.rs `Model::prior`, its sector at the atom: its distance is the two-photon detuning, and on
/// resonance its fold does not exist -- the link goes up, # Sectors), and kept, the two states form one cluster whose static coupling is the Raman term
/// −½(V_A†R_B + R_A†V_B) = Ω_R/2, to the closed form Σ_e V_A[e,1]* V_B[e,5] · ½(1/(E_5 + δ_B − E_e) + 1/(E_1 + δ_A − E_e))
/// (contract §6 R6, bare energies). With every drive folded with the bare levels -- no light is near here -- what the
/// closed form leaves out is the widths, (Γ/2Δ)² ≈ 2e-8, and the counter-rotating legs, Δ/(2ω) ≈ 3e-5 of each leg and of
/// the pair the fine structure's share of that (a ΔF = 1 coupling is the difference of the two lines): 4e-7 measured,
/// within 1e-5. With the two beams' drives folded at the atom, the dressing and the tweezer's shift of the lines over Δ
/// as well -- 1e-2.
#[test]
fn a_raman_pair_is_kept_on_resonance_and_couples_at_the_closed_form() {
    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let raman = raman_unit(&light, fa, fb);
    let (beam_a, beam_b) = (&request.beams[light.frequencies[fa].beams[0]], &request.beams[light.frequencies[fb].beams[0]]);
    for (level, drives, within) in [("bare", vec![], 1e-5), ("at the atom", at_the_atom(&request, &light, &[fa, fb]), 1e-2)] {
        let folded = Model::new(&request, &light, &drives).unwrap();
        let p = { let mut p = [0.0; GROUND]; p[1] = 1.0; p };
        let trap = folded.trap(&p, &folded.prepared()).unwrap();
        let prior = folded.prior(&[(trap.centre, [0.0; 3])]);
        assert!(prior.contains(&raman), "{level}: the prior keeps the Raman pair: {:?}", prior.iter().filter(|u| matches!(u, Unit::Term { a: 1, b: 5, .. })).collect::<Vec<_>>());
        let model = Model::new(&request, &light, &drives.iter().cloned().chain([raman.clone()]).collect::<Vec<_>>()).unwrap();
        let cluster = model.cluster_of(1);
        assert_eq!(model.cluster_of(5), cluster, "{level}: the pair joins its states");
        let c = &model.clusters()[cluster];
        let (x, y) = (c.states.iter().position(|&s| s == 1).unwrap(), c.states.iter().position(|&s| s == 5).unwrap());
        let middle = model.assemble(cluster, trap.centre, 0.0, &model.running());
        assert!(middle.still(), "{level}: in the pair's frames nothing beats");
        let coupling = total(&model, cluster, &Point { r: trap.centre, kappa: [0.0; 3], t: 0.0 }, &model.running())[(x, y)];
        let (da, db) = (model.omega(&detuning(&light, fa)), model.omega(&detuning(&light, fb)));
        let energy = |s: usize| request.hamiltonian[(s, s)].re;
        let mut closed = C64::ZERO;
        for e in GROUND..32 {
            closed += beam_a.coupling[(e, 1)].conj() * beam_b.coupling[(e, 5)] * (0.5 * (1.0 / (energy(5) + db - energy(e)) + 1.0 / (energy(1) + da - energy(e))));
        }
        assert!(closed.abs() > 0.1, "a Raman coupling of Ω_R/2 ≈ 2π·0.125 MHz: {}", closed.abs());
        eprintln!("{level}: Raman {} against the closed form {} ({:.1e} apart)", coupling.abs(), closed.abs(), (coupling.abs() - closed.abs()).abs() / closed.abs());
        assert!((coupling.abs() - closed.abs()).abs() <= within * closed.abs(), "{level}: Raman {} against the closed form {}", coupling.abs(), closed.abs());
    }
}

/// The frames of a cluster whose kept couplings close a cycle off resonance (cluster.rs, # Frames; ELIMINATION v8): the
/// pair of R6_RAMAN_CO kept at both its beats -- |1,0⟩ and |2,0⟩ coupled at the two lights' Raman beat β and statically
/// -- is a ring of two that misses by β. Its frames hold one of the two couplings static, exactly -- the frames'
/// difference zero or −β on the axes' integers, the stronger coupling's (the strength tree) --, and the other turns at
/// the ring's whole miss β, which the propagation integrates at its combined frequencies (spectral.rs, # Tones). (Karp's
/// frames left each at β/2 before.)
#[test]
fn a_kept_ring_s_frames_hold_one_coupling_static_and_the_other_at_the_whole_miss() {
    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let raman = raman_unit(&light, fa, fb);
    let static_coupling = Unit::Term { a: 1, b: 5, beat: detuning(&light, fa).plus(-1, &detuning(&light, fa)) };
    let model = Model::new(&request, &light, &[raman.clone(), static_coupling]).unwrap();
    let cluster = model.cluster_of(1);
    assert_eq!(model.cluster_of(5), cluster, "the ring joins its states");
    let c = &model.clusters()[cluster];
    let place = |s: usize| c.states.iter().position(|&x| x == s).unwrap();
    let Unit::Term { beat, .. } = &raman else { unreachable!("the Raman unit is a coupling") };
    // The Raman coupling a ← b at β is static where θ_a − θ_b = −β, the static one where it is zero.
    let apart = c.frames[place(1)].plus(-1, &c.frames[place(5)]);
    eprintln!("the frames' difference {:?}: the {} coupling static", apart, if apart.is_zero() { "static" } else { "Raman" });
    assert!(apart.is_zero() || apart == beat.negated(), "the frames hold one coupling static, exactly: {apart:?} against 0 or {:?}", beat.negated());
}

/// Optical pumping's branching (R5_PUMP: D1 σ⁺ F = 2 → F′ = 2 and its repump): from |2,m⟩ the resonant pump reaches |2′,m+1⟩,
/// so its photons land in each ground state b' with the emission's Clebsch–Gordan weight Σ_a |C_a[b', e]|²/Γ_e -- up to the
/// interference of the photons of the other states it reaches off resonance (D1 F′ = 1 at 814 MHz, the D2 states): their
/// amplitudes relative to the resonant one's, ρ = ‖t_others‖/|t_e| with t = X[·,b]/(E − E_b − δ − iΓ/2) (bare energies), move
/// a landing share by at most 2ρ + ρ². |2,2⟩ is dark to the pump (no F′ = 2 state of m′ = 3).
#[test]
fn optical_pumping_lands_by_the_clebsch_gordan_weights() {
    let request = scene("R5_PUMP");
    let light = Light::new(&request);
    let (fp, fr) = (frequency_of(&request, &light, "pump"), frequency_of(&request, &light, "repump"));
    let model = Model::new(&request, &light, &at_the_atom(&request, &light, &[fp, fr])).unwrap();
    // The pump's photons: within the ground states' splitting of its wave number (a Raman photon's 2e-5 of it; the repump's
    // line is 2 % away).
    let pump_k = request.beams[light.frequencies[fp].beams[0]].wave_number;
    let of_pump = |k: f64| (k / pump_k - 1.0).abs() < 1e-4;
    let levels = request.levels().unwrap();
    for m in -2..=1_i8 {
        let b = (0..GROUND).find(|&s| levels[s].f == 2 && levels[s].m == m).unwrap();
        let e = (GROUND..32).find(|&s| levels[s].manifold == crate::request::Manifold::D1 && levels[s].f == 2 && levels[s].m == m + 1).unwrap();
        let cluster = model.cluster_of(b);
        let c = &model.clusters()[cluster];
        let psi: Vec<C64> = c.states.iter().map(|&s| if s == b { C64::ONE } else { C64::ZERO }).collect();
        let photons = model.photons(cluster, &Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, &psi);
        let mut landed = [0.0; GROUND];
        for ph in photons.iter().filter(|ph| of_pump(ph.wave_number)) {
            for (i, &s) in model.clusters()[ph.landing].states.iter().enumerate() {
                if s < GROUND {
                    landed[s] += ph.amplitudes.iter().map(|u| u[i].norm_sqr()).sum::<f64>();
                }
            }
        }
        let total: f64 = landed.iter().sum();
        assert!(total > 0.0, "|2,{m}⟩ is pumped");
        let gamma = width(&request, e);
        let x = &request.beams[light.frequencies[fp].beams[0]].coupling;
        let delta = model.omega(&detuning(&light, fp));
        let t = |s: usize| x[(s, b)].abs() / C64::new(request.hamiltonian[(s, s)].re - request.hamiltonian[(b, b)].re - delta, -width(&request, s) / 2.0).abs();
        let rho = (GROUND..32).filter(|&s| s != e).map(|s| t(s).powi(2)).sum::<f64>().sqrt() / t(e);
        assert!(rho < 0.05, "the resonant state dominates: ρ = {rho}");
        for to in 0..GROUND {
            let weight = request.emission.iter().map(|c| c[(to, e)].norm_sqr()).sum::<f64>() / gamma;
            assert!((landed[to] / total - weight).abs() <= 2.0 * rho + rho * rho, "|2,{m}⟩ → {}: {} against the weight {weight} (ρ {rho})", request.states[to], landed[to] / total);
        }
    }
    let dark = (0..GROUND).find(|&s| levels[s].f == 2 && levels[s].m == 2).unwrap();
    let cluster = model.cluster_of(dark);
    let psi: Vec<C64> = model.clusters()[cluster].states.iter().map(|&s| if s == dark { C64::ONE } else { C64::ZERO }).collect();
    let pumped: f64 = model.photons(cluster, &Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, &psi).iter().filter(|ph| of_pump(ph.wave_number)).map(|ph| ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum();
    // Dark to the pump's D1 line: what is left is its far scattering through D2's |3′,3′⟩, 7 THz away.
    let bright: f64 = {
        let b = (0..GROUND).find(|&s| levels[s].f == 2 && levels[s].m == 0).unwrap();
        let cluster = model.cluster_of(b);
        let psi: Vec<C64> = model.clusters()[cluster].states.iter().map(|&s| if s == b { C64::ONE } else { C64::ZERO }).collect();
        model.photons(cluster, &Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, &psi).iter().filter(|ph| ph.wave_number == pump_k).map(|ph| ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum()
    };
    assert!(pumped <= 1e-9 * bright, "|2,2⟩ is dark to the pump: {pumped} against a bright state's {bright}");
}

/// With carriers, kept drives and explicit states, the model's identities hold: ⟨ψ|i(H − H†)|ψ⟩ = Σ_φ Σ_a ‖J_{a,φ}ψ‖² (to the
/// rounding of the parts' sum), and the readout sums to one -- a transition folded and the same transition kept (an explicit
/// state's absorbed momentum beside its mean is the optical sector's noise, model/optical.rs, and its fold gate is tested there).
#[test]
fn near_and_explicit_states_keep_the_identities() {
    for (name, id) in [("R2_RECOIL", "probe"), ("R3_DOPPLER", "cool-plus")] {
        a_kept_and_a_folded_drive(name, id);
    }
}

fn a_kept_and_a_folded_drive(name: &str, id: &str) {
    let request = scene(name);
    let light = Light::new(&request);
    let probe = frequency_of(&request, &light, id);
    let near = Component { frequency: probe, counter: false };
    let (b, e) = (7, 31);
    let drive = Unit::Drive { component: near, excited: e, ground: b };
    let lifted = at_the_atom(&request, &light, &[probe]);
    let folded = Model::new(&request, &light, &lifted).unwrap();
    let kept = Model::new(&request, &light, &lifted.iter().cloned().chain([drive]).collect::<Vec<_>>()).unwrap();
    let cluster = kept.cluster_of(b);
    assert_eq!(kept.clusters()[cluster].states, vec![b, e], "the drive makes its excited state explicit");
    for r in [[0.1, 0.2, -0.3], [-0.4, 0.05, 0.25]] {
        let at = Point { r, kappa: [0.3, -0.2, 1.1], t: 0.37 };
        for (model, psi) in [(&folded, vec![C64::ONE]), (&kept, vec![C64::new(0.9, 0.1), C64::new(-0.2, 0.35)])] {
            let c = model.cluster_of(b);
            let h = total(model, c, &at, &model.running());
            let rate = decay_rate(&h, &psi);
            let photons: f64 = model.photons(c, &at, &psi).iter().map(|ph| ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum::<f64>()
                / psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
            assert!((rate - photons).abs() <= 64.0 * f64::EPSILON * h.max_abs(), "{name}: hazard {rate} against the photons' {photons}");
            let sum: f64 = model.readout(c, &at, &psi).iter().sum();
            assert!((sum - 1.0).abs() <= 1e-12, "{name}: the populations sum to {sum}");
        }
    }
}

/// The potential's expansion is exact with drives folded at the atom in it: on the reported lights (the trap light coherent, every
/// other light averaged: R3's σ⁺ cooling pair, R5's pump and repump, R4_LIN_1's lin⊥lin cooling and repump on a cluster of
/// three ground states) each part of the gradient is the difference of the value and each entry of the Hessian that of the
/// gradient, each on its own scale (`agrees`: the weak axis's curvature on its own, not the tight axis's), the step chosen
/// by the differences' convergence -- the responses at the atom differentiated twice through their resolvents
/// (elim.rs `Reference::respond`), not differenced. (Differenced at u^{1/4} ℓ, R4_LIN_1's curvature along the tweezer's
/// axis read +0.166 rad/µs/µm² of the near light where it is −3.5·10⁻⁴.)
#[test]
fn the_expansion_s_derivatives_are_exact_with_drives_at_the_atom() {
    for (name, ids) in [("R3_DOPPLER", vec!["cool-plus"]), ("R5_PUMP", vec!["pump", "repump"]), ("R4_LIN_1", vec!["cool-plus", "repump-plus"])] {
        let request = scene(name);
        let light = Light::new(&request);
        let kept = at_the_atom(&request, &light, &ids.iter().map(|id| frequency_of(&request, &light, id)).collect::<Vec<_>>());
        let model = Model::new(&request, &light, &kept).unwrap();
        let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let p = { let mut p = [0.0; GROUND]; p[b] = 1.0; p };
        let lights = model.reported();
        for r in [[0.11, -0.07, 0.23], [-0.3, 0.2, -0.15]] {
            let q = model.expansion(r, &p, &lights);
            let shifted = |a: usize, s: f64| { let mut x = r; x[a] += s; x };
            for a in 0..3 {
                let value = converged(&steps().into_iter().map(|h| (model.potential(shifted(a, h), &p, &lights) - model.potential(shifted(a, -h), &p, &lights)) / (2.0 * h)).collect::<Vec<_>>(), 1);
                assert!(agrees(q.gradient[a], value, q.gradient[a].abs()), "{name} at {r:?}: ∂_{a}V {} against its difference {value:?}", q.gradient[a]);
                for c in 0..3 {
                    let second = converged(&steps().into_iter().map(|h| (model.expansion(shifted(a, h), &p, &lights).gradient[c] - model.expansion(shifted(a, -h), &p, &lights).gradient[c]) / (2.0 * h)).collect::<Vec<_>>(), 1);
                    assert!(agrees(q.hessian[a][c], second, own(&|x, y| q.hessian[x][y], a, c)), "{name} at {r:?}: ∂_{a}∂_{c}V {} against its difference {second:?}", q.hessian[a][c]);
                }
            }
        }
    }
}

/// The near light's derivatives are its values' (the class of the reported trap's defect: the elimination's jets by
/// differences read the forms' rounding times 1/δ², and the potential multiplied it by the field's coefficients): on the
/// scenes whose clusters hold several ground states -- R4_LIN_1's and R4_TW1's lin⊥lin cooling and repump, R6_RAMAN_CO's
/// Raman pair -- each model as the solve closes it with its averaged lights' drives at the atom, at three random
/// points about the trap's centre, with every light coherent and with the reported lights: every near form of every
/// cluster (assemble.rs `Model::near_exact`) and the near potential of each ground state of such a cluster (potential.rs
/// `Model::near_potential`) -- the force, each part of its gradient, against the differences of its values, and each
/// entry of its Hessian against the differences of the gradient (the chain value → gradient → Hessian, each link a first
/// difference: a second difference of the values would read their rounding times 1/h², as the defect did) --, each on its
/// own scale (`agrees`; a form's beside the rounding of the forms of its kind, which no derivative of it resolves), the
/// step chosen by the differences' convergence (`converged`).
#[test]
fn every_near_form_s_and_the_potential_s_jets_are_the_derivatives_of_their_values() {
    use crate::model::{jet, Lit};
    use crate::random::{Purpose, Stream};
    let mut checked = (0, 0);
    for name in ["R4_LIN_1", "R4_TW1", "R6_RAMAN_CO"] {
        let request = scene(name);
        let light = Light::new(&request);
        let chief = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let (closed, kept, trap, _) = Model::closed_at(&request, &light, vec![], chief, request.temperature_uk);
        assert!(closed.clusters().iter().any(|c| c.ground > 1), "{name}: a cluster of several ground states");
        let mut stream = Stream::new(7, Purpose::Initial, 0, 0);
        for lights in [closed.running(), closed.reported()] {
            let averaged = |c: Component| !c.counter && lights.0[c.frequency] == Lit::Averaged;
            let at: Vec<Unit> = closed.driven.iter().filter(|d| averaged(d.0)).map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).collect();
            let model = closed.rekept(&[kept.iter().filter(|u| !matches!(u, Unit::Bare { component, .. } | Unit::Drive { component, .. } if averaged(*component))).cloned().collect::<Vec<_>>(), at].concat()).unwrap();
            let shortest = 1.0 / model.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max);
            // The resolvent's condition on the excited block, κ = ‖K‖‖R₀‖ ≤ max |E_e| over the least half-width: the levels in
            // the frame (the far line's 7 THz beside the near line's) against the line's width.
            let condition = (crate::request::GROUND..STATES).map(|s| model.energies[s].abs()).fold(0.0, f64::max) / (0..crate::model::far::EXCITED).map(|e| 0.5 * model.decay[(e, e)].re).filter(|w| *w > 0.0).fold(f64::INFINITY, f64::min);
            for _ in 0..3 {
                let r: [f64; 3] = std::array::from_fn(|a| trap.centre[a] + [1.0, 0.3, 0.3][a] * (stream.uniform() - 0.5));
                // The displacements the differences read: ±h along each axis, step by step.
                let hs = steps();
                let shifts: Vec<[f64; 3]> = hs.iter().flat_map(|&h| (0..3).flat_map(move |a| [h, -h].map(|s| std::array::from_fn(|b| if a == b { r[b] + s } else { r[b] })))).collect();
                // [derivative: the gradient's three from a jet's values, then the Hessian's six of jet::PAIRS from its
                // gradients along the second axis][step], of the jets `at` gives at the displacements.
                let quotients = |at: &dyn Fn(usize, usize) -> f64| -> Vec<Vec<f64>> {
                    let mut out = vec![vec![0.0; hs.len()]; 9];
                    for (k, &h) in hs.iter().enumerate() {
                        let along = |a: usize, part: usize| (at(6 * k + 2 * a, part) - at(6 * k + 2 * a + 1, part)) / (2.0 * h);
                        for a in 0..3 {
                            out[a][k] = along(a, 0);
                        }
                        for (m, &(a, c)) in jet::PAIRS.iter().enumerate() {
                            out[3 + m][k] = along(c, 1 + a);
                        }
                    }
                    out
                };
                // Every form of every cluster with near forms.
                for cluster in 0..model.clusters().len() {
                    let plan = model.plan(cluster, &lights);
                    if plan.near.len() == 0 {
                        continue;
                    }
                    let jets = model.near_exact(&plan, &model.light.amplitudes(r), 2);
                    let near = &plan.near;
                    let displaced: Vec<crate::model::assemble::NearValues> = shifts.iter().map(|&x| model.near_exact(&plan, &model.light.amplitudes(x), 1)).collect();
                    let jet_at = |i: usize, e: usize| -> [C64; 4] {
                        let mut j = [C64::ZERO; 4];
                        displaced[i].jet(e, &mut j);
                        j
                    };
                    // Each form is exact to the rounding of the matrix it is an entry of (linalg.rs: a modest multiple of u
                    // on its data): the forms of one kind -- S, D and the crossed forms of the pair entries, the reference, the
                    // decay forms, the responses, their partners and the mixed ones -- one matrix, its rounding γ_{E²} (a solve on
                    // Q) times its largest entry, over ℓ^order, ℓ the light's shortest length; and each derivative to that
                    // rounding times the resolvent's condition κ of itself: the floors beside each form's own scale.
                    let (p, m) = (near.pairs.len(), near.responses.len());
                    let mut bounds: Vec<usize> = vec![0];
                    for size in [p, p, near.crossed.len(), near.reference, near.decays.len(), m, m, near.mixed.len()] {
                        bounds.push(bounds.last().unwrap() + size);
                    }
                    let mut value = [C64::ZERO; jet::WHOLE];
                    let values: Vec<C64> = (0..near.len()).map(|e| { jets.jet(e, &mut value); value[0] }).collect();
                    let largest: Vec<f64> = (0..near.len())
                        .map(|e| {
                            let k = bounds.windows(2).position(|w| w[0] <= e && e < w[1]).expect("a kind of form");
                            (bounds[k]..bounds[k + 1]).map(|i| values[i].abs()).fold(0.0, f64::max)
                        })
                        .collect();
                    for e in 0..near.len() {
                        let mut jet = [C64::ZERO; jet::WHOLE];
                        jets.jet(e, &mut jet);
                        for part in [false, true] {
                            let of = |z: C64| if part { z.im } else { z.re };
                            let hessian = |x: usize, y: usize| of(jet[jet::second(x, y)]);
                            for (d, q) in quotients(&|i, part| of(jet_at(i, e)[part])).iter().enumerate() {
                                let order = if d < 3 { 1 } else { 2 };
                                let (estimate, error) = converged(q, 1);
                                let solve = crate::units::rounding(crate::model::far::EXCITED * crate::model::far::EXCITED);
                                let floor = solve * (largest[e] / shortest.powi(order) + condition * of(jet[1 + d]).abs());
                                let scale = if d < 3 { of(jet[1 + d]).abs() } else { own(&hessian, jet::PAIRS[d - 3].0, jet::PAIRS[d - 3].1) };
                                assert!(agrees(of(jet[1 + d]), (estimate, error + floor), scale), "{name} cluster {cluster} form {e} at {r:?}: derivative {d} {} against ({estimate:e}, {error:e}); value {:e}, quotients {q:?}", of(jet[1 + d]), of(values[e]));
                            }
                        }
                        checked.0 += 1;
                    }
                }
                // The near potential of every ground state of a cluster of several: its force and its Hessian.
                for g in (0..GROUND).filter(|&g| model.clusters()[model.cluster_of(g)].ground > 1) {
                    let p: [f64; GROUND] = std::array::from_fn(|b| if b == g { 1.0 } else { 0.0 });
                    let q = model.near_potential(&model.light.amplitudes(r), &p, &lights, 2, None).quadratic;
                    let displaced: Vec<crate::model::Quadratic> = shifts.iter().map(|&x| model.near_potential(&model.light.amplitudes(x), &p, &lights, 1, None).quadratic).collect();
                    let estimates: Vec<(f64, f64)> = quotients(&|i, part| if part == 0 { displaced[i].value } else { displaced[i].gradient[part - 1] }).iter().map(|q| converged(q, 1)).collect();
                    for a in 0..3 {
                        assert!(agrees(q.gradient[a], estimates[a], q.gradient[a].abs()), "{name} ground {g} at {r:?}: the force ∂_{a}V {} against {:?}", q.gradient[a], estimates[a]);
                    }
                    for (m, &(a, c)) in jet::PAIRS.iter().enumerate() {
                        assert!(agrees(q.hessian[a][c], estimates[3 + m], own(&|x, y| q.hessian[x][y], a, c)), "{name} ground {g} at {r:?}: ∂_{a}∂_{c}V {} against {:?}", q.hessian[a][c], estimates[3 + m]);
                    }
                    checked.1 += 1;
                }
            }
        }
    }
    eprintln!("{} forms and {} potentials checked", checked.0, checked.1);
}

/// The light switched on lands the bare state as a jump does (step.rs `Trajectory::new`; model.rs `Model::landed`), and
/// that landing is measured against the definition's and booked (elim.rs `Fold::landing_error`, `Model::readings`): on
/// R2_RECOIL's probe and R4_LIN_1's lin⊥lin cooling and repump folded at the atom (a cluster of one ground state, and of
/// three), with the evaluation's orders as they come (K = 0, J = 0) and with the normalization's and the series' kept, a
/// trajectory started where its threshold lets it land holds the run's landing of its bare state to the bit; that is the
/// evaluation's map the readings measure (assemble.rs `Model::evaluations`, within 10⁻⁹); the trajectory's state against the
/// definition's S(1 + Σ S_f T_f)⁻¹ e_b (`Fold::definitions`, the self-consistent fold's) errs by what the readings
/// measure (2r + r²), the readings book at least that on the candidates, and the normalization's first order lands
/// nearer the definition than the identity of J = 0.
#[test]
fn a_trajectory_s_start_lands_as_the_definition_within_what_the_readings_book() {
    use crate::model::elim::Fold;
    use crate::step::Trajectory;
    for (name, ids) in [("R2_RECOIL", vec!["probe"]), ("R4_LIN_1", vec!["cool-plus", "repump-plus"])] {
        let request = scene(name);
        let light = Light::new(&request);
        let lifted = at_the_atom(&request, &light, &ids.iter().map(|id| frequency_of(&request, &light, id)).collect::<Vec<_>>());
        let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == b { 1.0 } else { 0.0 });
        let mut errors = vec![];
        for orders in [vec![], vec![Unit::Normalization { order: 1 }], vec![Unit::Series, Unit::Normalization { order: 1 }]] {
            let model = Model::new(&request, &light, &[lifted.clone(), orders.clone()].concat()).unwrap();
            let trap = model.trap(&reference, &model.prepared()).unwrap();
            let r = [trap.centre[0] + 0.13, trap.centre[1] - 0.07, trap.centre[2] + 0.05];
            let run = crate::model::Run::new(&model, reference);
            let cluster = model.cluster_of(b);
            let c = &model.clusters()[cluster];
            let place = c.states.iter().position(|&s| s == b).unwrap();
            let bare: Vec<C64> = (0..c.states.len()).map(|i| if i == place { C64::ONE } else { C64::ZERO }).collect();
            let at = Point { r, kappa: [0.0; 3], t: 0.0 };
            let landed: Vec<C64> = run.landed(cluster, &at, &bare).to_vec();
            let index = (0..256).find(|&i| Trajectory::new(&run, b, r, [0.0; 3], 11, i).threshold < crate::linalg::norm_sqr(&landed)).expect("a threshold the landing's norm passes");
            let trajectory = Trajectory::new(&run, b, r, [0.0; 3], 11, index);
            assert_eq!(trajectory.psi.to_vec(), landed, "{name} {orders:?}: the trajectory holds the run's landing");
            // The evaluation's map the readings measure, and the definition's.
            let amplitudes = model.light.amplitudes(r);
            let evaluated = model.evaluations(&[model.plan(cluster, &model.running())], r).swap_remove(0).1.unwrap_or_else(|| Mat::identity(c.states.len()));
            let (definition, levels) = model.definition_fold(cluster, &amplitudes).expect("the definition's fold");
            let defined = definition.definitions(&levels).1.expect("the definition's landing");
            let column = |m: &Mat| -> Vec<C64> { (0..m.rows).map(|i| m[(i, place)]).collect() };
            let norm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let apart = |a: &[C64], d: &[C64]| norm(&a.iter().zip(d).map(|(x, y)| *x - *y).collect::<Vec<_>>());
            assert!(apart(&landed, &column(&evaluated)) <= 1e-9 * norm(&column(&evaluated)), "{name} {orders:?}: the run's landing {landed:?} against the evaluation's map {:?}", column(&evaluated));
            let measured = Fold::landing_error(&evaluated, &defined, c.ground, &bare);
            let rel = apart(&trajectory.psi, &column(&defined)) / norm(&column(&defined));
            assert!((2.0 * rel + rel * rel - measured).abs() <= 1e-6 * measured.max(1e-12), "{name} {orders:?}: the trajectory's error {} against the measured {measured}", 2.0 * rel + rel * rel);
            let (read, _) = model.readings(&model.candidates(), &Sample { cluster, at, psi: bare.clone() }, None);
            let booked: f64 = read.iter().map(|x| x.population).sum();
            eprintln!("{name} {orders:?}: the landing {rel:.3e} off the definition's (a rate's {measured:.3e}); booked {booked:.3e}");
            assert!(booked >= measured * (1.0 - 1e-9), "{name} {orders:?}: booked {booked} below the landing's {measured}");
            errors.push(measured);
        }
        assert!(errors[1] < errors[0], "{name}: the normalization's first order lands nearer the definition: {errors:?}");
    }
}

/// The reported trap is the scene's, not the solve's (SPEC §1.3; potential.rs, # An averaged light): the other lights' smooth
/// shifts are in it as their drives' folds with the levels at the atom wherever those folds exist at the trap's centre,
/// whatever level the model takes the drives at. On the solve's own closure of R3_DOPPLER (a cooling pair in a tweezer),
/// R4_TW1 (cooling and repump light) and R9_PUSH (a push beam 45 linewidths off its line at the centre, in a shallow
/// trap), the closed model against the one with the other lights' drives all folded at the atom and the one with the
/// reference state's drives by them explicit: the reported trap is the same to the bit, softer than the prepared one by the
/// other lights' shift (the bare folds' constants alone would leave the prepared trap's frequencies, and an explicit drive
/// none of its own); and the prepared trap, the dynamics' own, is each model's as it stands.
#[test]
fn the_reported_trap_does_not_show_the_level_a_drive_is_taken_at() {
    for name in ["R3_DOPPLER", "R4_TW1", "R9_PUSH"] {
        let request = scene(name);
        let light = Light::new(&request);
        let others: Vec<usize> = (0..light.frequencies.len()).filter(|&f| f != light.trap).collect();
        let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let p = { let mut p = [0.0; GROUND]; p[b] = 1.0; p };
        // The model closed by its units' errors at the reference state (`closed`).
        let (model, closed, _, _) = Model::closed_at(&request, &light, vec![], b, request.temperature_uk);
        let reported = model.trap(&p, &model.reported()).unwrap();
        let prepared = model.trap(&p, &model.prepared()).unwrap();
        let softer = (0..3).map(|a| 1.0 - reported.lab[a] / prepared.lab[a]).fold(f64::NEG_INFINITY, f64::max);
        assert!(softer > 1e-4, "{name}: the other lights' smooth shift in the reported trap: softer by {softer:.3e}");
        // The other lights' drives all at the atom; and the reference state's drives by them explicit.
        let raised = at_the_atom(&request, &light, &others);
        let drives: Vec<Unit> = raised
            .iter()
            .filter_map(|u| match *u {
                Unit::Bare { component, excited, ground } if ground == b => Some(Unit::Drive { component, excited, ground }),
                _ => None,
            })
            .collect();
        let atom = Model::new(&request, &light, &[closed.clone(), raised.clone()].concat()).unwrap();
        let explicit = Model::new(&request, &light, &[closed.clone(), raised, drives].concat()).unwrap();
        assert!(explicit.clusters()[explicit.cluster_of(b)].states.len() > model.clusters()[model.cluster_of(b)].states.len(), "{name}: the reference state's drives are explicit");
        assert_eq!(reported, atom.trap(&p, &atom.reported()).unwrap(), "{name}: the reported trap with the other lights' drives at the atom");
        assert_eq!(reported, explicit.trap(&p, &explicit.reported()).unwrap(), "{name}: the reported trap with the reference state's drives explicit");
        assert_eq!(prepared, atom.trap(&p, &atom.prepared()).unwrap(), "{name}: the prepared trap");
        eprintln!("{name}: the reported trap {:?} kHz, softer than the prepared one by {softer:.3e}", reported.lab.map(|w| w / std::f64::consts::TAU * 1e3));
    }
}

/// A model built with units kept lists as candidates only those it folds (elim.rs `Model::candidates`): none it keeps, none
/// its clusters hold for free. The solve's closure of R3_DOPPLER from nothing kept (`Model::closed_at`, solve.rs `closed`)
/// lists none of what it kept; in R6_RAMAN_CO with the Raman pair and the drive of its A leg kept, the pair is no candidate
/// any more -- while the two ground states' static coupling, 6.8 GHz off in the pair's frames, is none of the generator's
/// and stays one.
#[test]
fn a_kept_unit_is_no_longer_a_candidate() {
    let request = scene("R3_DOPPLER");
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let (model, kept, _, _) = Model::closed_at(&request, &light, vec![], b, request.temperature_uk);
    let again: Vec<Unit> = model.candidates().into_iter().filter(|u| kept.contains(u)).collect();
    assert!(again.is_empty(), "R3_DOPPLER: kept units listed again: {again:?}");
    eprintln!("R3_DOPPLER: kept {kept:?}");

    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let ca = Component { frequency: fa, counter: false };
    let (beam_a, beam_b) = (&request.beams[light.frequencies[fa].beams[0]], &request.beams[light.frequencies[fb].beams[0]]);
    let e = (GROUND..STATES).max_by(|&x, &y| (beam_a.coupling[(x, 1)] * beam_b.coupling[(x, 5)]).abs().total_cmp(&(beam_a.coupling[(y, 1)] * beam_b.coupling[(y, 5)]).abs())).unwrap();
    let raman = raman_unit(&light, fa, fb);
    let leg_a = Unit::Drive { component: ca, excited: e, ground: 1 };
    let static_coupling = Unit::Term { a: 1, b: 5, beat: detuning(&light, fa).plus(-1, &detuning(&light, fa)) };
    let lifted = at_the_atom(&request, &light, &[fa, fb]);
    let folded = Model::new(&request, &light, &lifted).unwrap().candidates();
    for u in [&raman, &static_coupling] {
        assert!(folded.contains(u), "R6_RAMAN_CO: {u:?} a candidate while folded");
    }
    let kept: Vec<Unit> = lifted.iter().cloned().chain([raman, leg_a]).collect();
    let units = Model::new(&request, &light, &kept).unwrap().candidates();
    for u in &kept {
        assert!(!units.contains(u), "R6_RAMAN_CO: {u:?} listed though kept");
    }
    assert!(units.contains(&static_coupling), "R6_RAMAN_CO: the static coupling, off resonance in the pair's frames, is folded");
}

/// A candidate books no error where the states it couples are empty (elim.rs `errors`: every part weighed by the shares
/// of the eigenstates it joins): R2_RECOIL with its probe's drives at the atom and the atom in |2,2⟩ at the trap's centre
/// -- every coupling of two other states books nothing, on either output.
#[test]
fn a_unit_books_no_error_where_its_states_are_empty() {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = Component { frequency: frequency_of(&request, &light, "probe"), counter: false };
    let model = Model::new(&request, &light, &at_the_atom(&request, &light, &[probe.frequency])).unwrap();
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = { let mut p = [0.0; GROUND]; p[b] = 1.0; p };
    let trap = model.trap(&p, &model.prepared()).unwrap();
    let cluster = model.cluster_of(b);
    let psi = model.clusters()[cluster].states.iter().map(|&s| if s == b { C64::ONE } else { C64::ZERO }).collect();
    let sample = Sample { cluster, at: Point { r: trap.centre, kappa: [0.0; 3], t: 0.0 }, psi };
    let units = model.candidates();
    let (errors, _) = model.readings(&units, &sample, None);
    let mut empty = 0;
    for (u, e) in units.iter().zip(&errors) {
        let Unit::Term { a, b: other, .. } = u else { continue };
        if *a == b || *other == b {
            continue;
        }
        empty += 1;
        assert!(e.population == 0.0 && e.temperature == 0.0, "{u:?} books {e:?} with its states empty");
    }
    assert!(empty > 0, "a candidate coupling of two empty states");
}

/// A step's middle counts the elimination at the atom (assemble.rs # Work; step.rs `Middle::made`, `Middle::scattering`, where
/// a step counts them), and its scattering makes it no second time: R2_RECOIL with its probe's drives folded there counts
/// its middle at the focus with its state's scattering, at a point of its own (the thread's matrices forgotten), at no
/// less than the exact elimination's own count there (`Model::near_exact`) -- the scattering takes the middle's
/// responses (assemble.rs `Site::responses`) and counts less than that elimination.
#[test]
fn the_assembly_s_price_counts_the_elimination_at_the_atom() {
    use crate::step::Middle;
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = Component { frequency: frequency_of(&request, &light, "probe"), counter: false };
    let model = Model::new(&request, &light, &at_the_atom(&request, &light, &[probe.frequency])).unwrap();
    let mut p = [0.0; GROUND];
    p[7] = 1.0;
    let run = crate::model::Run::new(&model, p);
    let amplitudes = model.light.amplitudes([0.0; 3]);
    let mut met = 0;
    for c in 0..model.clusters().len() {
        let plan = model.plan(c, &model.running());
        if plan.near.len() == 0 {
            continue;
        }
        let once = model.near_exact(&plan, &amplitudes, 1).work * crate::decide::COMPLEX;
        Model::forget_quartic();
        let middle = run.middle(c, [0.0; 3], 0.0);
        let first: Vec<C64> = (0..model.clusters()[c].states.len()).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect();
        let scattering = middle.scattering(&middle.path([0.0; 3], [0.0; 3]), &first).1;
        assert!(middle.made() >= once, "cluster {c}: the middle counted {} below its elimination's {once}", middle.made());
        assert!(scattering < once, "cluster {c}: the scattering counted {scattering}, an elimination's {once} or more");
        met += 1;
    }
    assert!(met > 0);
}

/// The normalized generator's decay is its photons' rate with a kept pair of two frequencies (assemble.rs, # The normalized
/// generator: Λ_R = −i(G_R − G_L†), the anti-Hermitian read of the generator's own Doppler terms). R6_RAMAN_CO, |1,0⟩ and
/// |2,0⟩ joined by their kept Raman coupling at the two lights' beat, one light's drives folded at the atom and the other's
/// bare, the series' first order and the normalization's kept (`Unit::Series`, `Unit::Normalization`) on the joined
/// cluster of two references, F = 1 and F = 2 (eliminate.rs `Model::references`: each column's first order of its own
/// place's reference and the other's, ELIMINATION v12): the generator's decay
/// ⟨ψ|i(H − H†)|ψ⟩ on a state of both is the photons' Σ‖u_a‖² at rest and moving, whichever leg is at the atom; and the
/// same with no pair kept, each state's own cluster of one reference, one frequency's terms alone, the series' first
/// order kept too (`Unit::Series`). Without the orders kept the decay is another (the orders change the rate).
#[test]
fn the_normalized_generator_s_decay_is_its_photons_rate_with_a_kept_pair_of_two_frequencies() {
    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let exact = |f: usize| Beat::whole(light.frequencies[f].index.iter().map(|&i| i as i64).chain([0]).collect());
    let raman = Unit::Term { a: 1, b: 5, beat: exact(fa).plus(-1, &exact(fb)) };
    let rates = |model: &Model, g: usize, kappa: [f64; 3]| -> (f64, f64) {
        let c = model.cluster_of(g);
        let psi: Vec<C64> = (0..model.clusters()[c].states.len()).map(|x| C64::cis(0.7 * x as f64) * (0.6 + 0.3 * x as f64)).collect();
        let at = Point { r: [0.05, -0.03, 0.07], kappa, t: 0.0 };
        let photons: f64 = model.photons(c, &at, &psi).iter().map(|ph| ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum();
        (decay_rate(&total(model, c, &at, &model.running()), &psi), photons / psi.iter().map(|z| z.norm_sqr()).sum::<f64>())
    };
    for lifted in [fb, fa] {
        let drives = at_the_atom(&request, &light, &[lifted]);
        let orders = vec![Unit::Series, Unit::Normalization { order: 1 }];
        let second = Model::new(&request, &light, &[drives.clone(), vec![raman.clone()]].concat()).unwrap();
        let normalized = Model::new(&request, &light, &[drives.clone(), vec![raman.clone(), Unit::Series, Unit::Normalization { order: 1 }]].concat()).unwrap();
        assert_eq!(normalized.references(normalized.cluster_of(1), &normalized.running()), vec![0, 1], "the joined cluster's places have two references");
        assert_eq!(normalized.plan(normalized.cluster_of(1), &normalized.running()).fourth.as_ref().map(|f| f.orders.series), Some(1), "the joined cluster's run takes the first order");
        assert_eq!(normalized.clusters()[normalized.cluster_of(1)].states, vec![1, 5], "the kept pair's cluster");
        let alone = Model::new(&request, &light, &[drives, orders].concat()).unwrap();
        for kappa in [[0.0; 3], [3.0, -2.0, 5.0]] {
            let (generator, photons) = rates(&normalized, 1, kappa);
            assert!((generator - rates(&second, 1, kappa).0).abs() > 1e-7 * generator, "the orders change the rate");
            println!("light {lifted} at the atom, κ = {kappa:?}: (generator − photons)/photons {:+.2e}", generator / photons - 1.0);
            assert!((generator - photons).abs() <= 1e-9 * photons, "light {lifted} at the atom, κ = {kappa:?}: the generator's decay {generator:e} against its photons' {photons:e} ({:+.2e})", generator / photons - 1.0);
            for g in [1, 5] {
                let (generator, photons) = rates(&alone, g, kappa);
                assert!((generator - photons).abs() <= 1e-9 * photons, "no pair, state {g}, κ = {kappa:?}: {generator:e} against {photons:e} ({:+.2e})", generator / photons - 1.0);
            }
        }
    }
}

/// A bare fold onto an explicit state is read where the atoms are (elim.rs `Model::bare_mixing`, `Model::readings`,
/// `Model::prior`; model.rs `Model::settled`): R6_RAMAN_CO with its A leg |1,0⟩ → e kept explicit (e the excited state of
/// the pair's largest product of couplings) and its B leg |2,0⟩ → e folded with the bare levels across the pair's
/// two-photon distance d -- |2,0⟩'s level moved off the pair's resonance so that the fold's mixing a/|d| at the trap's
/// centre is a quarter: within 0.15 < a/d < 0.5, so the fold exists (2a/|d| < 1: bare it stays) and its next order
/// 2(a/d)² = 1/8 is beyond the share. There
/// - the levels d is read on (`Model::levels_at`) are the run's own: its generator's static diagonal at rest, with the
///   frames, on both states' clusters;
/// - the readings book on its `Unit::Bare` 2(a/d)² of the populations of its two states, a = |Σ_l a_l(r) X_l[e, w]| at
///   the beams' amplitudes there and d = z_e − E_w − ν on those levels, z_e the mean of e's cluster's ground levels with
///   e's frame (model.rs `Model::read`);
/// - the prior at the centre keeps it, and the model that keeps it folds the drive at the atom.
#[test]
fn a_bare_fold_onto_an_explicit_state_is_read_at_the_state_and_kept_beyond_the_share() {
    use crate::model::Level;
    let mut request = scene("R6_RAMAN_CO");
    let (g, w) = (1, 5);
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let (beam_a, beam_b) = (&request.beams[light.frequencies[fa].beams[0]], &request.beams[light.frequencies[fb].beams[0]]);
    let e = (GROUND..STATES).max_by(|&x, &y| (beam_a.coupling[(x, g)] * beam_b.coupling[(x, w)]).abs().total_cmp(&(beam_a.coupling[(y, g)] * beam_b.coupling[(y, w)]).abs())).unwrap();
    let kept = vec![Unit::Drive { component: Component { frequency: fa, counter: false }, excited: e, ground: g }];
    let link = (Component { frequency: fb, counter: false }, e, w);
    let unit = Unit::Bare { component: link.0, excited: e, ground: w };
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let centre = |light: &Light, model: &Model| {
        let p = { let mut p = [0.0; GROUND]; p[b] = 1.0; p };
        let r = model.trap(&p, &model.prepared()).unwrap().centre;
        let amplitudes = light.amplitudes(r);
        (r, amplitudes.clone(), model.levels_at(&amplitudes))
    };
    // The fold's coupling at the beams' amplitudes there.
    let coupling = |model: &Model, light: &Light, amplitudes: &[crate::field::Amplitude]| {
        let (f, tone) = (&light.frequencies[link.0.frequency], &model.tones[link.0.frequency]);
        f.beams.iter().zip(&tone.couplings[0]).map(|(&l, x)| amplitudes[l].value * x[(e, w)]).sum::<C64>().abs()
    };
    // |2,0⟩ moved so that |d| = 4a at the centre (the pair's own light shifts, a few rad/µs, ride on d as it is read).
    let model = Model::new(&request, &light, &kept).unwrap();
    let (_, amplitudes, levels) = centre(&light, &model);
    let a = coupling(&model, &light, &amplitudes);
    let d = {
        let c = model.clusters().iter().find(|c| c.states.contains(&e)).unwrap();
        let place = c.states.iter().position(|&s| s == e).unwrap();
        model.read(c, place, &levels, true) - levels[w] - model.tones[link.0.frequency].nu[0]
    };
    // E_w less `off` moves d by +off, to 4a on its side.
    let off = 4.0 * a * if d >= 0.0 { 1.0 } else { -1.0 } - d;
    request.hamiltonian[(w, w)] -= C64::real(off);
    let light = Light::new(&request);
    let model = Model::new(&request, &light, &kept).unwrap();
    assert!(model.bare_links().contains(&link), "the B leg's bare fold exists (2a/|d| < 1, every beam at its peak)");
    let (r, amplitudes, levels) = centre(&light, &model);
    let mixing = model.bare_mixing(link, &amplitudes, &levels);
    eprintln!("R6_RAMAN_CO: a {a:.4e} rad/µs, d {d:.4e} moved by {off:.4e}: the mixing {mixing:.4}");
    assert!(mixing > 0.15 && mixing < 0.5, "the mixing {mixing} within 0.15 < a/d < 0.5");
    // The levels against the run's generator: its static diagonal at rest, with each state's frame.
    let at = Point { r, kappa: [0.0; 3], t: 0.0 };
    let level = |s: usize| {
        let cluster = model.clusters().iter().position(|c| c.states.contains(&s)).unwrap();
        let c = &model.clusters()[cluster];
        let place = c.states.iter().position(|&x| x == s).unwrap();
        let h = total(&model, cluster, &at, &model.running());
        h[(place, place)].re + model.omega(&c.frames[place])
    };
    let cluster = model.cluster_of(g);
    let c = &model.clusters()[cluster];
    assert!(c.states.contains(&e) && !c.states.contains(&w), "e explicit with |1,0⟩, |2,0⟩ apart: {:?}", c.states);
    for &s in c.states[..c.ground].iter().chain([&w]) {
        assert!((levels[s] - level(s)).abs() <= 1e-9 * level(s).abs().max(1.0), "state {s}: level {} against the generator's {}", levels[s], level(s));
    }
    // The fold's mixing from its definition on the generator's levels.
    let a = coupling(&model, &light, &amplitudes);
    let place = c.states.iter().position(|&s| s == e).unwrap();
    let z = model.omega(&c.frames[place]) + (0..c.ground).map(|p| level(c.states[p]) - model.omega(&c.frames[p])).sum::<f64>() / c.ground as f64;
    let analytic = a / (z - level(w) - model.tones[link.0.frequency].nu[0]).abs();
    assert!((mixing - analytic).abs() <= 1e-9 * analytic, "the mixing {mixing} against a/|d| {analytic}");
    // The readings at a state of e's cluster: 2(a/d)² of e's population (|2,0⟩ is no state of this cluster).
    let psi: Vec<C64> = c.states.iter().map(|&s| if s == g { C64::real(0.8) } else if s == e { C64::new(0.0, 0.6) } else { C64::ZERO }).collect();
    let pe = 0.36 / psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
    let sample = Sample { cluster, at, psi };
    let units = model.candidates();
    let (errors, _) = model.readings(&units, &sample, None);
    let booked = errors[units.iter().position(|u| *u == unit).expect("the bare fold is a candidate")];
    let expected = 2.0 * analytic * analytic * pe;
    eprintln!("R6_RAMAN_CO: booked {booked:?} against 2(a/d)² p_e = {expected:.6e}");
    assert!((booked.population - expected).abs() <= 1e-9 * expected && booked.temperature == booked.population, "booked {booked:?} against {expected}");
    // Beyond the share at the centre: the prior keeps it, and kept the drive is folded at the atom.
    assert!(2.0 * mixing * mixing > super::elim::share());
    let prior = model.prior(&[(r, [0.0; 3])]);
    assert!(prior.contains(&unit), "the prior keeps the bare fold beyond the share: {prior:?}");
    let lifted = Model::new(&request, &light, &[kept, vec![unit]].concat()).unwrap();
    assert!(lifted.level(link.0, e, w) == Level::Atom, "kept, the drive is folded at the atom");
}
