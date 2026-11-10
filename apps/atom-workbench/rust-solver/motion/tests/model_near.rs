//! The near light's elimination on the reference scenes (refs/scenes, built by the app): the Reiter–Sørensen fold against the
//! exact two-level steady state, the Λ scheme's Raman coupling against its closed form, optical pumping's branching against the
//! emission's Clebsch–Gordan weights, and the identities the model holds by construction with near carriers and explicit states
//! (the decay is the photons' rate, the readout sums to one, a kept and a folded drive give the same recoil second moment).
use atom_motion::field::Light;
use atom_motion::linalg::{C64, Mat};
use atom_motion::model::{Component, Lights, Model, Point, Sample, Unit, GROUND, STATES};
use atom_motion::request::Request;

mod model_common;
use model_common::{decay_rate, scene};

/// The frequency (an index of the light's) of the beam named `id`.
fn frequency_of(request: &Request, light: &Light, id: &str) -> usize {
    let beam = request.beams.iter().position(|b| b.id == id).unwrap();
    light.frequencies.iter().position(|f| f.beams.contains(&beam)).unwrap()
}

/// The sum of a generator's parts at the point (τ = 0).
fn total(model: &Model, cluster: usize, at: &Point, lights: &Lights) -> Mat {
    let g = model.assemble(cluster, at, lights);
    let n = model.clusters()[cluster].states.len();
    let mut h = Mat::zeros(n, n);
    for p in &g.parts {
        h.add_scaled(C64::ONE, &p.generator);
    }
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
    let model = Model::new(&request, &light, &[Unit::Far { component: near }]).unwrap();
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
        let error = model.errors(&[Unit::Drive { carrier: near, excited: e, ground: b }], &[sample])[0][0];
        assert!((error.parameter - s).abs() <= 1e-9 * s, "the measured S {} against the exact relative error s {s}", error.parameter);
        assert!(s > 1e-4 && s < 0.05, "a small saturation: {s}");
    }
}

/// The Λ scheme's Raman coupling (R6_RAMAN_CO: D1 σ⁺σ⁺, A from F = 1, B from F = 2, Raman-resonant between |1,0⟩ and |2,0⟩,
/// Δ ≈ −20 GHz): its unit must be kept (the gate fails on resonance), and kept, the two states form one cluster whose static
/// coupling is the Raman term −½(V_A†R_B + R_A†V_B) − (i/2)R_A†DR_B = Ω_R/2, to the closed form Σ_e V_A[e,1]* V_B[e,5] ·
/// ½(1/(E_5 + δ_B − E_e) + 1/(E_1 + δ_A − E_e)) (contract §6 R6, bare energies) within its neglected orders: (Γ/2Δ)², the
/// dressing and the tweezer's shift of the lines over Δ -- 1e-2 here.
#[test]
fn a_raman_pair_is_kept_on_resonance_and_couples_at_the_closed_form() {
    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (fa, fb) = (frequency_of(&request, &light, "ramanA"), frequency_of(&request, &light, "ramanB"));
    let (ca, cb) = (Component { frequency: fa, counter: false }, Component { frequency: fb, counter: false });
    let folded = Model::new(&request, &light, &[Unit::Far { component: ca }, Unit::Far { component: cb }]).unwrap();
    let raman = atom_motion::model::Unit::Term { a: 1, b: 5, from: Some(cb), to: Some(ca) };
    let units = folded.units();
    assert!(units.contains(&raman), "the Raman pair is a unit: {:?}", units.iter().filter(|u| matches!(u, Unit::Term { .. })).collect::<Vec<_>>());
    let p = { let mut p = [0.0; GROUND]; p[1] = 1.0; p };
    let trap = folded.trap(&p, &folded.prepared()).unwrap();
    let reach = folded.reach(&trap, &p, request.temperature_uk, 2e-3);
    let gate = folded.gates(&[raman.clone()], &reach)[0];
    assert!(!gate.exists(), "a Raman-resonant pair cannot fold: {gate:?}");
    let model = Model::new(&request, &light, &[Unit::Far { component: ca }, Unit::Far { component: cb }, raman]).unwrap();
    let cluster = model.cluster_of(1);
    assert_eq!(model.cluster_of(5), cluster, "the pair joins its states");
    let c = &model.clusters()[cluster];
    let (x, y) = (c.states.iter().position(|&s| s == 1).unwrap(), c.states.iter().position(|&s| s == 5).unwrap());
    let g = model.assemble(cluster, &Point { r: trap.centre, kappa: [0.0; 3], t: 0.0 }, &model.running());
    let still = g.parts.iter().find(|p| p.omega == 0.0 && p.wave == [0.0; 3]).unwrap();
    let coupling = still.generator[(x, y)];
    let (beam_a, beam_b) = (&request.beams[light.frequencies[fa].beams[0]], &request.beams[light.frequencies[fb].beams[0]]);
    let omega = |f: usize| model.omega(&atom_motion::model::Beat({
        let mut n: Vec<i64> = light.frequencies[f].index.iter().map(|&i| i as i64).collect();
        n.push(0);
        n
    }));
    let (da, db) = (omega(fa), omega(fb));
    let energy = |s: usize| request.hamiltonian[(s, s)].re;
    let mut closed = C64::ZERO;
    for e in GROUND..32 {
        closed += beam_a.coupling[(e, 1)].conj() * beam_b.coupling[(e, 5)] * (0.5 * (1.0 / (energy(5) + db - energy(e)) + 1.0 / (energy(1) + da - energy(e))));
    }
    assert!(closed.abs() > 0.1, "a Raman coupling of Ω_R/2 ≈ 2π·0.125 MHz: {}", closed.abs());
    assert!((coupling.abs() - closed.abs()).abs() <= 1e-2 * closed.abs(), "Raman {} against the closed form {}", coupling.abs(), closed.abs());
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
    let kept = [Unit::Far { component: Component { frequency: fp, counter: false } }, Unit::Far { component: Component { frequency: fr, counter: false } }];
    let model = Model::new(&request, &light, &kept).unwrap();
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
        let delta = model.omega(&atom_motion::model::Beat({
            let mut n: Vec<i64> = light.frequencies[fp].index.iter().map(|&i| i as i64).collect();
            n.push(0);
            n
        }));
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

/// With near carriers, kept drives and explicit states, the model's identities hold: ⟨ψ|i(H − H†)|ψ⟩ = Σ_φ Σ_a ‖J_{a,φ}ψ‖² (to the
/// rounding of the parts' sum), the readout sums to one; and a transition kept and folded gives its photon the same absorption
/// second moment (SPEC §1.6, one rule): the folded photon's Re⟨∇f|∇f⟩/‖f‖² is the explicit one's Re[∇ln σ*⊗∇ln σ],
/// ∇ln σ = ∇ln a − ∇δ/(δ − iΓ/2) either way.
#[test]
fn near_and_explicit_states_keep_the_identities_and_one_recoil_rule() {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = frequency_of(&request, &light, "probe");
    let near = Component { frequency: probe, counter: false };
    let (b, e) = (7, 31);
    let drive = Unit::Drive { carrier: near, excited: e, ground: b };
    let folded = Model::new(&request, &light, &[Unit::Far { component: near }]).unwrap();
    let kept = Model::new(&request, &light, &[Unit::Far { component: near }, drive]).unwrap();
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
            assert!((rate - photons).abs() <= 64.0 * f64::EPSILON * h.max_abs(), "hazard {rate} against the photons' {photons}");
            let sum: f64 = model.readout(c, &at, &psi).iter().sum();
            assert!((sum - 1.0).abs() <= 1e-12, "the populations sum to {sum}");
        }
        // The probe's photon that leaves the atom in |2,2⟩ in either model -- the strongest class by far, the folded drive's
        // in one and the explicit state's in the other --, polarized along x across a direction along y.
        let moment = |model: &Model, psi: &[C64]| {
            let c = model.cluster_of(b);
            let weight = |ph: &atom_motion::model::Photon| ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>();
            let ph = model.photons(c, &at, psi).into_iter().max_by(|x, y| weight(x).total_cmp(&weight(y))).unwrap();
            assert_eq!((ph.landing, ph.explicit.len()), (c, model.clusters()[c].states.len() - 1), "the probe's photon back into |2,2⟩");
            ph.recoil(&[1.0, 0.0, 0.0]).second
        };
        let (m_folded, m_kept) = (moment(&folded, &[C64::ONE]), moment(&kept, &[C64::new(0.6, 0.0), C64::new(0.0, 0.8)]));
        for i in 0..3 {
            for j in 0..3 {
                assert!((m_folded[i][j] - m_kept[i][j]).abs() <= 1e-6 * m_folded[i][i].abs().max(m_folded[j][j].abs()), "M[{i}][{j}]: folded {:?} kept {:?}", m_folded, m_kept);
            }
        }
    }
}

/// The near units' gates over the reach (SPEC §1.5): a weak probe's drives fold (R2_RECOIL's 100Γ detuning, R3_DOPPLER's two
/// beams at s = 0.02 and δ = −Γ/2: S_b of the order of their saturation), their provable bounds above the saturation measured at
/// the trap's centre; the sweep a is bounded too.
#[test]
fn weak_drives_fold_within_their_provable_gates() {
    for (name, ids) in [("R2_RECOIL", vec!["probe"]), ("R3_DOPPLER", vec!["cool-plus"])] {
        let request = scene(name);
        let light = Light::new(&request);
        let kept: Vec<Unit> = ids.iter().map(|id| Unit::Far { component: Component { frequency: frequency_of(&request, &light, id), counter: false } }).collect();
        let model = Model::new(&request, &light, &kept).unwrap();
        let b = 7;
        let p = { let mut p = [0.0; GROUND]; p[b] = 1.0; p };
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, request.temperature_uk, 2e-3);
        let units: Vec<Unit> = model.units().into_iter().filter(|u| matches!(u, Unit::Drive { ground, .. } if *ground == b)).collect();
        assert!(!units.is_empty(), "{name}: drives from |2,2⟩");
        let started = std::time::Instant::now();
        let gates = model.gates(&units, &reach);
        let cluster = model.cluster_of(b);
        let sample = Sample { cluster, at: Point { r: trap.centre, kappa: [0.0; 3], t: 0.0 }, psi: vec![C64::ONE; model.clusters()[cluster].states.len()] };
        let errors = model.errors(&units, &[sample]);
        let measured: f64 = errors.iter().map(|e| e[0].parameter).sum();
        for (u, g) in units.iter().zip(&gates) {
            assert!(g.exists(), "{name}: {u:?} {g:?}");
            assert!(g.saturation >= measured * (1.0 - 1e-9), "{name}: the bound {} below the measured S {measured}", g.saturation);
        }
        eprintln!("{name}: {} drive gates in {:?}, S bound {:.3e} measured {:.3e}, sweep {:.3e}", units.len(), started.elapsed(), gates[0].saturation, measured, gates[0].sweep);
    }
}

/// The potential's expansion is exact with near light in it: on the reported lights (the trap light coherent, every other light
/// averaged: R3's σ⁺ cooling pair, R5's pump and repump) the gradient is the central difference of the value and the Hessian that of
/// the gradient, to the differences' own truncation (h² of the third derivative; h = 1e-4 µm) -- the near light's responses
/// differentiated twice through their resolvents (jet.rs), not left out.
#[test]
fn the_expansion_s_derivatives_are_exact_with_near_light() {
    for (name, ids) in [("R3_DOPPLER", vec!["cool-plus"]), ("R5_PUMP", vec!["pump", "repump"])] {
        let request = scene(name);
        let light = Light::new(&request);
        let kept: Vec<Unit> = ids.iter().map(|id| Unit::Far { component: Component { frequency: frequency_of(&request, &light, id), counter: false } }).collect();
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

/// A model built with units kept lists as candidates only the units it folds (`Model::units`): none it keeps, none its clusters
/// hold for free. The gates' closure from nothing kept (R2_RECOIL, its σ⁺ probe on |2,2⟩ → |3′,3′⟩) adds only units it has not
/// kept each round and ends; in R6_RAMAN_CO with the Raman pair and the drive of its A leg kept, the B leg's drive is in the
/// cluster at a residual beat of exactly zero (θ_e − θ_5 = δ_B) and the two ground states' static coupling is assembled within
/// it: neither is a candidate any more.
#[test]
fn a_kept_unit_is_no_longer_a_candidate() {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let p = { let mut p = [0.0; GROUND]; p[b] = 1.0; p };
    let mut kept: Vec<Unit> = vec![];
    loop {
        let model = Model::new(&request, &light, &kept).unwrap();
        let units = model.units();
        let again: Vec<&Unit> = units.iter().filter(|u| kept.contains(u)).collect();
        assert!(again.is_empty(), "R2_RECOIL: kept units listed again: {again:?}");
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, request.temperature_uk, 2e-3);
        let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            break;
        }
        kept.extend(failing);
    }
    let probe = Component { frequency: frequency_of(&request, &light, "probe"), counter: false };
    eprintln!("R2_RECOIL: kept {kept:?}");
    assert!(kept.contains(&Unit::Far { component: probe }), "R2_RECOIL: the probe's far fold fails: {kept:?}");

    let request = scene("R6_RAMAN_CO");
    let light = Light::new(&request);
    let (ca, cb) = (Component { frequency: frequency_of(&request, &light, "ramanA"), counter: false }, Component { frequency: frequency_of(&request, &light, "ramanB"), counter: false });
    let (beam_a, beam_b) = (&request.beams[light.frequencies[ca.frequency].beams[0]], &request.beams[light.frequencies[cb.frequency].beams[0]]);
    let e = (GROUND..STATES).max_by(|&x, &y| (beam_a.coupling[(x, 1)] * beam_b.coupling[(x, 5)]).abs().total_cmp(&(beam_a.coupling[(y, 1)] * beam_b.coupling[(y, 5)]).abs())).unwrap();
    let raman = Unit::Term { a: 1, b: 5, from: Some(cb), to: Some(ca) };
    let (leg_a, leg_b) = (Unit::Drive { carrier: ca, excited: e, ground: 1 }, Unit::Drive { carrier: cb, excited: e, ground: 5 });
    let static_coupling = Unit::Term { a: 1, b: 5, from: None, to: None };
    let near = [Unit::Far { component: ca }, Unit::Far { component: cb }];
    let folded = Model::new(&request, &light, &near).unwrap().units();
    for u in [&raman, &leg_a, &leg_b, &static_coupling] {
        assert!(folded.contains(u), "R6_RAMAN_CO: {u:?} a candidate while folded");
    }
    let kept: Vec<Unit> = near.iter().cloned().chain([raman, leg_a]).collect();
    let units = Model::new(&request, &light, &kept).unwrap().units();
    for u in kept.iter().chain([&leg_b, &static_coupling]) {
        assert!(!units.contains(u), "R6_RAMAN_CO: {u:?} listed though in the generator");
    }
}

/// A folded unit books no error where the states it couples are empty (SPEC §1.5: a coupling's mixing m²(p_a + p_b), a drive's
/// S_b p_b, on the populations and on T alike): R2_RECOIL with its probe near and the atom in |2,2⟩ at the trap's centre -- every
/// coupling of two other states and every drive from another ground state books nothing on either output, however large its
/// measured parameter.
#[test]
fn a_unit_books_no_error_where_its_states_are_empty() {
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = Component { frequency: frequency_of(&request, &light, "probe"), counter: false };
    let model = Model::new(&request, &light, &[Unit::Far { component: probe }]).unwrap();
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
            Unit::Excited { excited, other } => vec![*excited, *other],
            Unit::Drive { ground, .. } => vec![*ground],
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

/// The assembly's price counts the near light's elimination (assemble.rs # Work): R2_RECOIL with its probe near prices a
/// generator at no less than the factorizations its elimination makes, one per ground state for the probe's dressing and one
/// for the response of the cluster's ground state -- (n³ − n)/3 complex multiply-adds each on the n excited states.
#[test]
fn the_assembly_s_price_counts_the_near_elimination() {
    use atom_motion::step::Dynamics;
    let request = scene("R2_RECOIL");
    let light = Light::new(&request);
    let probe = Component { frequency: frequency_of(&request, &light, "probe"), counter: false };
    let model = Model::new(&request, &light, &[Unit::Far { component: probe }]).unwrap();
    let mut p = [0.0; GROUND];
    p[7] = 1.0;
    let run = atom_motion::model::Run::new(&model, p);
    let n = STATES - GROUND;
    let factorizations = (GROUND + 1) as f64 * ((n * n * n - n) / 3) as f64 * atom_motion::decide::COMPLEX;
    assert!(run.assembly_work() >= factorizations, "an assembly priced {} below its factorizations' {factorizations}", run.assembly_work());
}
