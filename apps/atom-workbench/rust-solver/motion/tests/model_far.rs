//! The model's far light, potential, trap, envelope, reach and assembly on the reference scenes (refs/scenes, built by the
//! app's own model builder): analytic references where the far light is a Gaussian beam's or a standing wave's, and the
//! identities the model holds by construction (the assembly's static diagonal is the potential, its force the potential's
//! gradient, its decay the photons' rate, the readout's populations sum to one).
use atom_motion::field::Light;
use atom_motion::linalg::{dot3, C64, Mat};
use atom_motion::model::{Model, Point, GROUND};
use atom_motion::request::Request;
use atom_motion::step::Middle as _;

mod model_common;
use model_common::{decay_rate, scene};

/// The scene's model with every unit folded, its initial ground state's populations e_b.
fn built(name: &str) -> (Request, Light, Model, [f64; GROUND]) {
    let request = scene(name);
    let light = Light::new(&request);
    let model = Model::new(&request, &light, &[]).expect("the far light folds");
    let b = (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap();
    let mut p = [0.0; GROUND];
    p[b] = 1.0;
    (request, light, model, p)
}

fn close(a: f64, b: f64, relative: f64) -> bool {
    (a - b).abs() <= relative * a.abs().max(b.abs())
}

/// A lone Gaussian tweezer (R1_TWEEZER, π along B): every ground state's far shift is its own multiple of |E|², V = V_min |a(r)|²
/// with |a|² = e^{−2ρ²/(w₀²(1+s))}/(1 + s), s = ζ²/z_R². The descent ends at the focus as it stands (its slope vanishes
/// there exactly), and the analytic Hessian is V_min times |a|²'s, −4/w₀² across and −2/z_R² along: ω_ρ² = 4|V_min|(ħ/m)/w₀²,
/// ω_ζ² = 2|V_min|(ħ/m)/z_R² -- the closed forms of contract §6 R1, exact for the model's paraxial beam.
#[test]
fn a_lone_tweezer_s_trap_frequencies_are_the_gaussian_beam_s() {
    let (request, _, model, p) = built("R1_TWEEZER");
    let trap = model.trap(&p, &model.prepared()).unwrap();
    assert_eq!(trap.centre, [0.0; 3]);
    let beam = &request.beams[request.trap_beam];
    let (w, k) = (beam.waist.unwrap(), beam.wave_number);
    let zr = k * w * w / 2.0;
    let u = -trap.minimum;
    assert!(u > 0.0, "the tweezer holds the state: {}", trap.minimum);
    let hm = model.hbar_over_mass();
    let radial = (4.0 * u * hm).sqrt() / w;
    let axial = (2.0 * u * hm).sqrt() / zr;
    let mut expected = [radial, radial, axial];
    expected.sort_by(|a, b| a.total_cmp(b));
    for i in 0..3 {
        assert!(close(trap.frequencies[i], expected[i], 1e-10), "{:?} against {:?}", trap.frequencies, expected);
    }
    // Along the lab axes: the beam runs along x.
    assert!(close(trap.lab[0], axial, 1e-10) && close(trap.lab[1], radial, 1e-10) && close(trap.lab[2], radial, 1e-10), "{:?}", trap.lab);
}

/// A retro-reflected tweezer (R1_RETRO: R = 1, φ = 0, one polarization) is one coherent field, V = q|a₁ + a₂|², the depth four
/// times one beam's. On the axis |a₁ + a₂|² = 4cos²(kζ − ψ)/(1 + s) with the Gouy phase ψ = atan(ζ/z_R), so its curvature at the
/// focus is −8(k − 1/z_R)² − 8/z_R² (the lattice: k√(8U₀/m) of contract §6 R1 to that Gouy correction), and across it
/// 4e^{−2ρ²/w₀²}, −16/w₀²: ω_lat² = (|V_min|/4)(ħ/m)(8(k − 1/z_R)² + 8/z_R²), ω_ρ² = 4|V_min|(ħ/m)/w₀².
#[test]
fn a_retro_reflected_tweezer_s_lattice_is_the_coherent_standing_wave_s() {
    let (request, _, model, p) = built("R1_RETRO");
    let trap = model.trap(&p, &model.prepared()).unwrap();
    assert!(trap.centre.iter().all(|x| x.abs() < 1e-9), "{:?}", trap.centre);
    let beam = &request.beams[request.trap_beam];
    let (w, k) = (beam.waist.unwrap(), beam.wave_number);
    let zr = k * w * w / 2.0;
    let (u, hm) = (-trap.minimum, model.hbar_over_mass());
    let lattice = (u / 4.0 * hm * (8.0 * (k - 1.0 / zr).powi(2) + 8.0 / (zr * zr))).sqrt();
    let radial = (4.0 * u * hm).sqrt() / w;
    assert!(close(trap.lab[0], lattice, 1e-9), "lattice {} against {}", trap.lab[0], lattice);
    assert!(close(trap.lab[1], radial, 1e-9) && close(trap.lab[2], radial, 1e-9), "{:?} radial {}", trap.lab, radial);
}

/// The proven envelope bounds the prepared potential everywhere (model.rs `Envelope`): at points across the trap and far out,
/// −V_prep(r) ≤ B max_l A_l(r), A_l = 1/(1 + 2ρ²/w₀² + ζ²/z_R²) of the trap light's beams -- the bound the initial ensemble
/// draws under (ensemble.rs) --, a standing wave and a rank-2 crossed field included.
#[test]
fn the_envelope_bounds_the_prepared_potential() {
    for name in ["R1_TWEEZER", "R1_RETRO", "R1_CROSSED", "R1_SIGMA_P"] {
        let (_, light, model, p) = built(name);
        let envelope = model.envelope(&p);
        let lights = model.prepared();
        let mut x = 12345_u64;
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for i in 0..2000 {
            let scale = if i % 2 == 0 { 2.0 } else { 20.0 };
            let r = [next() * scale, next() * scale, next() * scale];
            let v = model.potential(r, &p, &lights);
            let brightest = envelope
                .beams
                .iter()
                .map(|&l| {
                    let beam = &light.beams[l];
                    let (w, z) = (beam.waist.expect("a tweezer"), beam.rayleigh().expect("a tweezer"));
                    let zeta = dot3(beam.direction, r);
                    1.0 / (1.0 + 2.0 * (dot3(r, r) - zeta * zeta) / (w * w) + zeta * zeta / (z * z))
                })
                .fold(0.0, f64::max);
            let bound = envelope.peak() * brightest;
            assert!(-v <= bound * (1.0 + 1e-12) + 1e-12, "{name}: −V = {} above its bound {} at {:?}", -v, bound, r);
        }
    }
}

/// The reach's box holds its sublevel set: on a grid over each face of the box the prepared potential lies above the level.
#[test]
fn the_reach_s_box_holds_its_sublevel_set() {
    for name in ["R1_TWEEZER", "R1_RETRO", "R1_CROSSED"] {
        let (_, _, model, p) = built(name);
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, 100.0, 2e-3);
        let level = reach.minimum + reach.energy;
        assert!(reach.cell.iter().all(|[lo, hi]| lo.is_finite() && hi.is_finite() && lo < hi), "{name}: {:?}", reach.cell);
        let lights = model.prepared();
        for a in 0..3 {
            for side in 0..2 {
                for i in 0..=20 {
                    for j in 0..=20 {
                        let mut r = [0.0; 3];
                        let (b, c) = ((a + 1) % 3, (a + 2) % 3);
                        r[a] = reach.cell[a][side];
                        r[b] = reach.cell[b][0] + (reach.cell[b][1] - reach.cell[b][0]) * i as f64 / 20.0;
                        r[c] = reach.cell[c][0] + (reach.cell[c][1] - reach.cell[c][0]) * j as f64 / 20.0;
                        // Above the level, or on it within its rounding where the box touches the set (a lone beam's axis).
                        assert!(model.potential(r, &p, &lights) > level - 1e-12 * level.abs(), "{name}: a face point {:?} lies in the reach", r);
                    }
                }
            }
        }
    }
}

/// The one assembly holds the potential: its static diagonal on the trap's own state moves from far away to r by V_prep(r) and its
/// force is V_prep's gradient (the far light's terms summed with their phases at the point); its decay is the photons' rate,
/// ⟨ψ|i(H − H†)|ψ⟩ = Σ_φ Σ_a ‖J_{a,φ}ψ‖² (Lindblad by construction); its readout's populations sum to one.
#[test]
fn the_assembly_holds_the_potential_its_force_and_the_photons_rate() {
    for name in ["R1_TWEEZER", "R1_RETRO", "R1_CROSSED", "R1_SIGMA_P"] {
        let (_, _, model, p) = built(name);
        let b = p.iter().position(|&x| x == 1.0).unwrap();
        let cluster = model.cluster_of(b);
        let n = model.clusters()[cluster].states.len();
        let place = model.clusters()[cluster].states.iter().position(|&s| s == b).unwrap();
        let lights = model.running();
        // The generator at a point at rest, and its gradient's entry on the state: the form with ρ = |b⟩⟨b|.
        let total = |r: [f64; 3]| {
            let middle = model.assemble(cluster, r, 0.0, &lights);
            let path = middle.path([0.0; 3], [0.0; 3]);
            let mut h = Mat::zeros(n, n);
            middle.generator(&path, 0.0, &mut h);
            let mut rho = vec![C64::ZERO; n * n];
            rho[place * n + place] = C64::ONE;
            (h, middle.force(&path, 0.0, &rho))
        };
        let far = total([1e4, 1e4, 1e4]).0;
        for r in [[0.13, -0.21, 0.07], [0.4, 0.3, -0.2], [-0.05, 0.6, 0.31]] {
            let (h, f) = total(r);
            let q = model.expansion(r, &p, &model.prepared());
            let moved = h[(place, place)].re - far[(place, place)].re;
            assert!(close(moved, q.value, 1e-9), "{name}: the static diagonal moves {moved} against V {}", q.value);
            for i in 0..3 {
                assert!((f[i] - q.gradient[i]).abs() <= 1e-9 * q.gradient.iter().map(|x| x.abs()).fold(1e-300, f64::max), "{name}: force {:?} against ∇V {:?}", f, q.gradient);
            }
            let psi: Vec<C64> = (0..n).map(|i| C64::new(1.0 + i as f64, 0.5 - i as f64)).collect();
            let at = Point { r, kappa: [0.0; 3], t: 0.0 };
            let rate = decay_rate(&h, &psi);
            let photons: f64 = model.photons(cluster, &at, &psi).iter().map(|ph| ph.amplitudes.iter().map(|u| u.iter().map(|z| z.norm_sqr()).sum::<f64>()).sum::<f64>()).sum::<f64>()
                / psi.iter().map(|z| z.norm_sqr()).sum::<f64>();
            // Equal to the rounding of the parts' sum: the Hermitian parts at ±q are each complex and cancel in the sum's
            // anti-Hermitian part to the rounding of their scale.
            let scale = h.max_abs();
            assert!(rate > 0.0 && (rate - photons).abs() <= 64.0 * f64::EPSILON * scale, "{name}: hazard {rate} against the photons' {photons} (scale {scale})");
            let populations = model.readout(cluster, &at, &psi);
            let sum: f64 = populations.iter().sum();
            assert!((sum - 1.0).abs() <= 1e-13, "{name}: the populations sum to {sum}");
        }
    }
}

/// The far scenes' units: every far component's fold exists over the reach (the 852 nm tweezer's saturation 2m² is 1e-4 or less), the beats
/// of a frequency with itself are no unit, and a static far coupling between ground sublevels has its gate and its measured
/// error at the trap's centre -- the same small parameter, the error m²(p_a + p_b) with m² = S/2 below the gate's bound.
#[test]
fn the_far_scenes_units_fold_within_their_gates() {
    for name in ["R1_TWEEZER", "R1_RETRO", "R1_CROSSED", "R1_SIGMA_P"] {
        let (_, _, model, p) = built(name);
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, 30.0, 2e-3);
        let units = model.units();
        let gates = model.gates(&units, &reach);
        let b = p.iter().position(|&x| x == 1.0).unwrap();
        let cluster = model.cluster_of(b);
        let n = model.clusters()[cluster].states.len();
        let sample = atom_motion::model::Sample { cluster, at: Point { r: trap.centre, kappa: [0.0; 3], t: 0.0 }, psi: vec![C64::ONE; n] };
        let errors = model.errors(&units, &[sample]);
        for ((u, g), e) in units.iter().zip(&gates).zip(&errors) {
            match u {
                atom_motion::model::Unit::Far { .. } => {
                    assert!(g.exists() && g.saturation < 1e-4, "{name}: {u:?} {g:?}");
                    assert!(e[0].parameter <= 0.5 * g.saturation * (1.0 + 1e-9), "{name}: {u:?} measured {} above its bound {}", e[0].parameter, g.saturation);
                }
                atom_motion::model::Unit::Term { .. } => {
                    assert!(g.saturation.is_finite(), "{name}: {u:?} {g:?}");
                    assert!(e[0].parameter <= 0.5 * g.saturation * (1.0 + 1e-9), "{name}: {u:?} measured {} above its bound {}", e[0].parameter, g.saturation);
                }
                _ => {}
            }
        }
    }
}
