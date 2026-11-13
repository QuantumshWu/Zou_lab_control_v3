//! The elimination with the levels at the atom on the reference scenes (refs/scenes, built by the app): the Reiter–Sørensen
//! fold against the exact two-level steady state, the Λ scheme's Raman coupling against its closed form at either level of its
//! drives, optical pumping's branching against the emission's Clebsch–Gordan weights, and the identities the model holds by
//! construction with carriers and explicit states (the decay is the photons' rate, the readout sums to one, a kept and a
//! folded drive give the same recoil second moment).
use atom_motion::field::Light;
use atom_motion::linalg::{C64, Mat};
use atom_motion::model::{Beat, Component, Lights, Model, Point, Sample, Unit, GROUND, STATES};
use atom_motion::request::Request;
use atom_motion::step::Middle as _;

mod model_common;
use model_common::{at_the_atom, closed, decay_rate, scene};

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

/// The Reiter–Sørensen fold of a two-level transition (R2_RECOIL: σ⁺ along B, |2,2⟩ → |3′,3′⟩ the only state it reaches) is its
/// linear response: rate Γ|V|²/(Δ² + Γ²/4) and light shift −|V|²Δ/(Δ² + Γ²/4), |V| the probe's coupling (a plane wave: |a| = 1).
/// The exact two-level steady state scatters Γ|V|²/(Δ² + Γ²/4 + 2|V|²): the fold's rate errs by the relative
/// s = 2|V|²/(Δ² + Γ²/4) = 2·rate/Γ exactly -- the drive's measured small parameter S_b (SPEC §1.5), here to rounding. The rate
/// gives Δ² = Γ|V|²/rate − Γ²/4, and the shift is −rate·Δ/Γ with that |Δ| (the probe's far parts -- its counter-rotating
/// Bloch–Siegert shift and its outside states' shift, which the probe on less off also holds -- move it by 1e-6 of itself: within
/// 1e-5). At κ = −k/2 along the probe its first-order Doppler shift w = (ħ/m)(k·κ + k²/2) vanishes.
#[test]
fn a_two_level_fold_is_the_linear_response_and_its_error_the_saturation() {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = frequency_of(&request, &light, "probe");
    let near = Component { frequency: probe, counter: false };
    let model = Model::new(&request, &light, &at_the_atom(&request, &light, &[probe])).unwrap();
    let (b, e) = (7, 31);
    let cluster = model.cluster_of(b);
    assert_eq!(model.clusters()[cluster].states, vec![b], "the stretched state alone");
    let k = request.beams[light.frequencies[probe].beams[0]].wave_number;
    let gamma = width(&request, e);
    let v2 = request.beams[light.frequencies[probe].beams[0]].coupling[(e, b)].norm_sqr();
    let mut off = model.running();
    off.0[probe] = atom_motion::model::Lit::Off;
    for r in [[0.0, 0.0, 0.0], [0.2, -0.1, 0.4], [0.5, 0.3, -0.6]] {
        let at = Point { r, kappa: [0.0, 0.0, -k / 2.0], t: 0.0 };
        let (on, without) = (total(&model, cluster, &at, &model.running()), total(&model, cluster, &at, &off));
        let shift = on[(0, 0)].re - without[(0, 0)].re;
        let rate = -2.0 * (on[(0, 0)].im - without[(0, 0)].im);
        let delta2 = gamma * v2 / rate - gamma * gamma / 4.0;
        assert!(delta2 > 0.0);
        let linear = rate * delta2.sqrt() / gamma;
        assert!((shift.abs() - linear.abs()).abs() <= 1e-5 * linear.abs(), "shift {shift} against the linear response's ±{}", linear.abs());
        let s = 2.0 * rate / gamma;
        let exact = gamma * v2 / (delta2 + gamma * gamma / 4.0 + 2.0 * v2);
        assert!(((rate - exact) / exact - s).abs() <= 1e-12, "the fold's rate error {} against s = {s}", (rate - exact) / exact);
        let sample = Sample { cluster, at, psi: vec![C64::ONE] };
        let error = model.errors(&[Unit::Drive { component: near, excited: e, ground: b }], &[sample])[0][0];
        assert!((error.parameter - s).abs() <= 1e-9 * s, "the measured S {} against the exact relative error s {s}", error.parameter);
        assert!(s > 1e-4 && s < 0.05, "a small saturation: {s}");
    }
}

/// A frequency's exact detuning δ_ω as a beat (its index on the request's axes; no frame coordinate).
fn detuning(light: &Light, frequency: usize) -> Beat {
    let mut n: Vec<i64> = light.frequencies[frequency].index.iter().map(|&i| i as i64).collect();
    n.push(0);
    Beat(n)
}

/// The coupling of |1,0⟩ ← |2,0⟩ at the beat of two frequencies, emitting `to`'s and absorbing `from`'s: δ_to − δ_from.
fn raman_unit(light: &Light, to: usize, from: usize) -> Unit {
    let (to, from) = (detuning(light, to), detuning(light, from));
    Unit::Term { a: 1, b: 5, beat: Beat(to.0.iter().zip(&from.0).map(|(x, y)| x - y).collect()) }
}

/// The Λ scheme's Raman coupling (R6_RAMAN_CO: D1 σ⁺σ⁺, A from F = 1, B from F = 2, Raman-resonant between |1,0⟩ and |2,0⟩,
/// Δ ≈ −20 GHz): the coupling of the two states at the two frequencies' beat is a unit whatever the level of the light's
/// drives, it must be kept (its distance is the two-photon detuning: the gate fails on resonance), and kept, the two states
/// form one cluster whose static coupling is the Raman term −½(V_A†R_B + R_A†V_B) = Ω_R/2, to the closed form
/// Σ_e V_A[e,1]* V_B[e,5] · ½(1/(E_5 + δ_B − E_e) + 1/(E_1 + δ_A − E_e)) (contract §6 R6, bare energies). With every drive
/// folded with the bare levels -- no light is near here -- what the closed form leaves out is the widths, (Γ/2Δ)² ≈ 2e-8, and
/// the counter-rotating legs, Δ/(2ω) ≈ 3e-5 of each leg and of the pair the fine structure's share of that (a ΔF = 1
/// coupling is the difference of the two lines): 4e-7 measured, within 1e-5. With the two beams' drives folded at the atom,
/// the dressing and the tweezer's shift of the lines over Δ as well -- 1e-2.
#[test]
fn a_raman_pair_is_kept_on_resonance_and_couples_at_the_closed_form() {
    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let raman = raman_unit(&light, fa, fb);
    let (beam_a, beam_b) = (&request.beams[light.frequencies[fa].beams[0]], &request.beams[light.frequencies[fb].beams[0]]);
    for (level, drives, within) in [("bare", vec![], 1e-5), ("at the atom", at_the_atom(&request, &light, &[fa, fb]), 1e-2)] {
        let folded = Model::new(&request, &light, &drives).unwrap();
        let units = folded.units();
        assert!(units.contains(&raman), "{level}: the Raman pair is a unit: {:?}", units.iter().filter(|u| matches!(u, Unit::Term { a: 1, b: 5, .. })).collect::<Vec<_>>());
        let p = { let mut p = [0.0; GROUND]; p[1] = 1.0; p };
        let trap = folded.trap(&p, &folded.prepared()).unwrap();
        let cluster = folded.cluster_of(1);
        let psi: Vec<C64> = folded.clusters()[cluster].states.iter().map(|&s| if s == 1 { C64::ONE } else { C64::ZERO }).collect();
        let error = folded.errors(&[raman.clone()], &[Sample { cluster, at: Point { r: trap.centre, kappa: [0.0; 3], t: 0.0 }, psi }])[0][0];
        assert!(!(error.population < 0.5), "{level}: a Raman-resonant pair cannot fold: its error at |1⟩ {error:?}");
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
        let e = (GROUND..32).find(|&s| levels[s].manifold == atom_motion::request::Manifold::D1 && levels[s].f == 2 && levels[s].m == m + 1).unwrap();
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
/// other light averaged: R3's σ⁺ cooling pair, R5's pump and repump) the gradient is the central difference of the value and the
/// Hessian that of the gradient, to the differences' own truncation (h² of the third derivative; h = 1e-4 µm) -- the responses
/// at the atom differentiated twice through their resolvents (jet.rs), not left out.
#[test]
fn the_expansion_s_derivatives_are_exact_with_drives_at_the_atom() {
    for (name, ids) in [("R3_DOPPLER", vec!["cool-plus"]), ("R5_PUMP", vec!["pump", "repump"])] {
        let request = scene(name);
        let light = Light::new(&request);
        let kept = at_the_atom(&request, &light, &ids.iter().map(|id| frequency_of(&request, &light, id)).collect::<Vec<_>>());
        let model = Model::new(&request, &light, &kept).unwrap();
        let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
        let p = { let mut p = [0.0; GROUND]; p[b] = 1.0; p };
        let lights = model.reported();
        let h = 1e-4;
        for r in [[0.11, -0.07, 0.23], [-0.3, 0.2, -0.15]] {
            let q = model.expansion(r, &p, &lights);
            let shifted = |a: usize, s: f64| { let mut x = r; x[a] += s; x };
            for a in 0..3 {
                let value = (model.potential(shifted(a, h), &p, &lights) - model.potential(shifted(a, -h), &p, &lights)) / (2.0 * h);
                assert!((value - q.gradient[a]).abs() <= 1e-6 * q.gradient.iter().map(|x| x.abs()).fold(1e-12, f64::max), "{name}: ∂_{a}V {} against its difference {value}", q.gradient[a]);
                let (up, down) = (model.expansion(shifted(a, h), &p, &lights), model.expansion(shifted(a, -h), &p, &lights));
                let scale = q.hessian.iter().flatten().map(|x| x.abs()).fold(1e-12, f64::max);
                for c in 0..3 {
                    let second = (up.gradient[c] - down.gradient[c]) / (2.0 * h);
                    assert!((second - q.hessian[a][c]).abs() <= 1e-6 * scale, "{name}: ∂_{a}∂_{c}V {} against its difference {second}", q.hessian[a][c]);
                }
            }
        }
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
        let (model, closed, _, _) = closed(&request, &light, vec![], b, request.temperature_uk);
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

/// A model built with units kept lists as candidates only the units it folds (`Model::units`): none it keeps, none its clusters
/// hold for free. The closure by the units' errors from nothing kept (R3_DOPPLER, whose cooling line the tweezer moves by more
/// than its own detuning) adds only units it has not kept each round; in R6_RAMAN_CO with the Raman pair
/// and the drive of its A leg kept, the B leg's drive is in the cluster at a residual beat of exactly zero (θ_e − θ_5 = δ_B)
/// and is no candidate any more -- while the two ground states' static coupling, 6.8 GHz off in the pair's frames, is none of
/// the generator's and stays one.
#[test]
fn a_kept_unit_is_no_longer_a_candidate() {
    let request = scene("R3_DOPPLER");
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut kept: Vec<Unit> = vec![];
    for _ in 0..4 {
        let model = Model::new(&request, &light, &kept).unwrap();
        let units = model.units();
        let again: Vec<&Unit> = units.iter().filter(|u| kept.contains(u)).collect();
        assert!(again.is_empty(), "R3_DOPPLER: kept units listed again: {again:?}");
        let (_, more, _, _) = closed(&request, &light, kept.clone(), b, request.temperature_uk);
        if more.len() == kept.len() {
            break;
        }
        kept = more;
    }
    eprintln!("R3_DOPPLER: kept {kept:?}");

    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let (ca, cb) = (Component { frequency: fa, counter: false }, Component { frequency: fb, counter: false });
    let (beam_a, beam_b) = (&request.beams[light.frequencies[fa].beams[0]], &request.beams[light.frequencies[fb].beams[0]]);
    let e = (GROUND..STATES).max_by(|&x, &y| (beam_a.coupling[(x, 1)] * beam_b.coupling[(x, 5)]).abs().total_cmp(&(beam_a.coupling[(y, 1)] * beam_b.coupling[(y, 5)]).abs())).unwrap();
    let raman = raman_unit(&light, fa, fb);
    let (leg_a, leg_b) = (Unit::Drive { component: ca, excited: e, ground: 1 }, Unit::Drive { component: cb, excited: e, ground: 5 });
    let static_coupling = Unit::Term { a: 1, b: 5, beat: Beat(vec![0; detuning(&light, fa).0.len()]) };
    let lifted = at_the_atom(&request, &light, &[fa, fb]);
    let folded = Model::new(&request, &light, &lifted).unwrap().units();
    for u in [&raman, &leg_a, &leg_b, &static_coupling] {
        assert!(folded.contains(u), "R6_RAMAN_CO: {u:?} a candidate while folded");
    }
    let kept: Vec<Unit> = lifted.iter().cloned().chain([raman, leg_a]).collect();
    let units = Model::new(&request, &light, &kept).unwrap().units();
    for u in kept.iter().chain([&leg_b]) {
        assert!(!units.contains(u), "R6_RAMAN_CO: {u:?} listed though in the generator");
    }
    assert!(units.contains(&static_coupling), "R6_RAMAN_CO: the static coupling, off resonance in the pair's frames, is folded");
}

/// A folded unit books no error where the states it couples are empty (SPEC §1.5: a coupling's mixing m²(p_a + p_b), a drive's
/// s p_g, on the populations and on T alike): R2_RECOIL with its probe's drives at the atom and the atom in |2,2⟩ at the trap's
/// centre -- every coupling of two other states and every drive from another ground state books nothing on either output,
/// however large its measured parameter.
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
    let units = model.units();
    let errors = model.errors(&units, &[sample]);
    let mut measured = 0.0_f64;
    for (u, e) in units.iter().zip(&errors) {
        let states = match u {
            Unit::Term { a, b, .. } => vec![*a, *b],
            Unit::Bare { ground, .. } | Unit::Drive { ground, .. } => vec![*ground],
            _ => continue,
        };
        if states.contains(&b) {
            continue;
        }
        measured = measured.max(e[0].parameter);
        assert!(e[0].population == 0.0 && e[0].temperature == 0.0, "{u:?} books {:?} with its states empty", e[0]);
    }
    eprintln!("the largest parameter measured on empty states: {measured:.3e}");
    assert!(measured > 0.0, "a unit of empty states with a measured parameter");
}

/// The assembly's price counts the elimination at the atom (assemble.rs # Work): R2_RECOIL with its probe's drives folded there
/// prices a generator at no less than the factorizations its elimination makes, one per ground state for the probe's dressing
/// and one for the response of the cluster's ground state -- (n³ − n)/3 complex multiply-adds each on the n excited states.
#[test]
fn the_assembly_s_price_counts_the_elimination_at_the_atom() {
    use atom_motion::step::Dynamics;
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = Component { frequency: frequency_of(&request, &light, "probe"), counter: false };
    let model = Model::new(&request, &light, &at_the_atom(&request, &light, &[probe.frequency])).unwrap();
    let mut p = [0.0; GROUND];
    p[7] = 1.0;
    let run = atom_motion::model::Run::new(&model, p);
    let n = STATES - GROUND;
    let factorizations = (GROUND + 1) as f64 * ((n * n * n - n) / 3) as f64 * atom_motion::decide::COMPLEX;
    assert!(run.assembly_work() >= factorizations, "an assembly priced {} below its factorizations' {factorizations}", run.assembly_work());
}

/// The fourth order's decay is its photons' rate with a kept pair of two frequencies (assemble.rs, # The fourth order:
/// Λ_R = −i(G_R − G_L†), the anti-Hermitian read of the generator's own Doppler terms). R6_RAMAN_CO, |1,0⟩ and |2,0⟩ joined
/// by their kept Raman coupling at the two lights' beat, one light's drives folded at the atom and the other's bare, the
/// cluster's fold at the fourth order: the generator's decay ⟨ψ|i(H − H†)|ψ⟩ on a state of both is the photons'
/// Σ‖u_a‖² at rest and moving, whichever leg is at the atom (with the fold's own Y subtracted in its stead the other
/// frequency's bare leg's Hermitian Doppler partner was read as decay: +6.07e-3 and −6.08e-3 of the rate). The same
/// with no pair kept and each state's fold at the fourth order, one frequency's terms alone, as before.
#[test]
fn the_fourth_order_s_decay_is_its_photons_rate_with_a_kept_pair_of_two_frequencies() {
    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let exact = |f: usize| -> Vec<i64> { light.frequencies[f].index.iter().map(|&i| i as i64).chain([0]).collect() };
    let raman = Unit::Term { a: 1, b: 5, beat: Beat(exact(fa).iter().zip(exact(fb)).map(|(x, y)| x - y).collect()) };
    let rates = |model: &Model, g: usize, kappa: [f64; 3]| -> (f64, f64) {
        let c = model.cluster_of(g);
        let psi: Vec<C64> = (0..model.clusters()[c].states.len()).map(|x| C64::cis(0.7 * x as f64) * (0.6 + 0.3 * x as f64)).collect();
        let at = Point { r: [0.05, -0.03, 0.07], kappa, t: 0.0 };
        let photons: f64 = model.photons(c, &at, &psi).iter().map(|ph| ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum();
        (decay_rate(&total(model, c, &at, &model.running()), &psi), photons / psi.iter().map(|z| z.norm_sqr()).sum::<f64>())
    };
    for lifted in [fb, fa] {
        let drives = at_the_atom(&request, &light, &[lifted]);
        let second = Model::new(&request, &light, &[drives.clone(), vec![raman.clone()]].concat()).unwrap();
        let fourth = Model::new(&request, &light, &[drives.clone(), vec![raman.clone(), Unit::Second { ground: 1 }]].concat()).unwrap();
        assert_eq!(fourth.clusters()[fourth.cluster_of(1)].states, vec![1, 5], "the kept pair's cluster");
        assert!(second.units().contains(&Unit::Second { ground: 1 }) && !fourth.units().contains(&Unit::Second { ground: 1 }), "the cluster's fold has a fourth order, and takes it");
        let alone = Model::new(&request, &light, &[drives, vec![Unit::Second { ground: 1 }, Unit::Second { ground: 5 }]].concat()).unwrap();
        for kappa in [[0.0; 3], [3.0, -2.0, 5.0]] {
            let (generator, photons) = rates(&fourth, 1, kappa);
            assert!((generator - rates(&second, 1, kappa).0).abs() > 1e-7 * generator, "the fourth order changes the rate");
            println!("light {lifted} at the atom, κ = {kappa:?}: (generator − photons)/photons {:+.2e}", generator / photons - 1.0);
            assert!((generator - photons).abs() <= 1e-9 * photons, "light {lifted} at the atom, κ = {kappa:?}: the generator's decay {generator:e} against its photons' {photons:e} ({:+.2e})", generator / photons - 1.0);
            for g in [1, 5] {
                let (generator, photons) = rates(&alone, g, kappa);
                assert!((generator - photons).abs() <= 1e-9 * photons, "no pair, state {g}, κ = {kappa:?}: {generator:e} against {photons:e} ({:+.2e})", generator / photons - 1.0);
            }
        }
    }
}
