//! The selected model's checks (model/selected.rs): its frame keeps the fold's elements slow, its decay is its photons',
//! its force at the frame's time zero is the model's (`Now::force`), a dragged atom's mean force is the static model's
//! (RUSTPLAN §6 C2), and the six scenes' first checkpoint. Each scene's run is the solve's (`solve::run_of`: P by
//! `select::choose`), its P PYSEL's.
use super::*;
use crate::step::{Brownian, Domain, Dynamics, Middle, Midway, Scratch, Trajectory};
use rayon::prelude::*;

/// The six scenes' own requests (V3C_REQ: scratchpad reqprec\req, as model/rs/tests.rs reads them) with PYSEL's P (the
/// levels added to the ground states; pyexp2.status.md A).
fn scenes() -> Vec<(&'static str, Request, Vec<usize>)> {
    let dir = std::path::PathBuf::from(std::env::var("V3C_REQ").expect("V3C_REQ: the directory of the six scenes' requests (scratchpad reqprec\\req)"));
    let chosen: [(&str, &str, &[&str]); 6] = [
        ("tilt0", "tilt0.json", &["D2-3", "D2-2"]),
        ("tilt2", "tilt2.json", &["D2-3", "D2-2"]),
        ("R4", "R4_LIN_4.json", &["D2-2", "D2-3", "D2-1"]),
        ("GMD1C", "GMD1C_1ms.json", &[]),
        ("PGC3D", "PGC3D_1ms.json", &["D2-2", "D2-3"]),
        ("TET", "TET_1ms.json", &["D2-2", "D2-3"]),
    ];
    let only = std::env::var("V3E_SCENE").ok();
    chosen
        .into_iter()
        .filter(|(name, ..)| only.as_deref().is_none_or(|o| o == *name))
        .map(|(name, file, levels)| {
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join(file)).unwrap()).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            // These requests predate the outside forms' transitions, which the scene does not read.
            for form in line["model"]["motion"]["outsideForms"].as_array_mut().unwrap() {
                form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
            }
            let request = Request::from_native(&line).unwrap();
            let keep: Vec<usize> = (0..STATES)
                .filter(|&s| {
                    let label = &request.states[s];
                    label.starts_with("ground") || levels.iter().any(|l| label.starts_with(&format!("{l}-")))
                })
                .collect();
            (name, request, keep)
        })
        .collect()
}

use crate::request::Request;

/// A scene's run at its reference state as the solve makes it (`solve::run_of`, `solve::chief`): P chosen, said against
/// PYSEL's.
fn run_of(request: &Request, keep: &[usize]) -> Selected {
    let chief = crate::solve::chief(request);
    let run = crate::solve::run_of(request, std::array::from_fn(|s| f64::from(u8::from(s == chief)))).unwrap();
    eprintln!("P {} chosen (PYSEL's {}): {}", run.rs.keep.len(), keep.len(), if run.rs.keep == keep { "the same" } else { "apart" });
    run
}

/// A state of P with every amplitude set (a fixed pattern of phases and sizes, normalized).
fn spread(n: usize) -> Vec<C64> {
    let psi: Vec<C64> = (0..n).map(|a| C64::cis(0.7 * a as f64 + 0.3) * (1.0 + 0.37 * ((3 * a) % 5) as f64)).collect();
    let norm = norm_sqr(&psi).sqrt();
    psi.iter().map(|z| *z * (1.0 / norm)).collect()
}

/// The points the checks take: the centre at rest, and a point off it on a moving atom (κ of a few times the thermal
/// speed's of tens of µK, 1/µm).
const POINTS: [(V3, V3); 2] = [([0.0; 3], [0.0; 3]), ([0.31, -0.17, 0.23], [5.0, -8.0, 12.0])];

/// The frame keeps the fold's model slow (model/selected.rs, # One cluster, one frame): every H_eff element of the
/// six scenes' models (PYSEL's P), at the centre at rest and at a moving point off it, has no fast part in the
/// trajectory's frame -- its vector c − θ_a + θ_b's fast index (n_φ, n_tw, n_ν) is zero, as floquet/assemble.rs reads
/// the same element. Said per scene: the elements, the beats, the largest |Ω|.
#[test]
fn the_selected_model_is_slow_in_its_frame() {
    let mut fast = Vec::new();
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        for (r, kappa) in POINTS {
            let model = run.at(r, kappa, Making { model: true, kicks: false, lift: false });
            let (mut elements, mut beyond) = (0, 0);
            for part in &model.effective {
                let table = run.elements(&part.key, Of::Generator);
                for (x, &beat) in table.iter().enumerate() {
                    if part.value.data[x] == C64::ZERO && part.gradient.iter().all(|d| d.data[x] == C64::ZERO) {
                        continue;
                    }
                    elements += 1;
                    let n = run.read(|b| b.exact[beat].0);
                    if n != [0; 3] {
                        beyond += 1;
                        if beyond <= 3 {
                            let (a, b) = (run.rs.keep[x / keep.len()], run.rs.keep[x % keep.len()]);
                            eprintln!("  {name}: key {:?} element {} <- {}: fast {n:?}", part.key, request.states[a], request.states[b]);
                        }
                    }
                }
            }
            let (values, _) = run.components();
            let most = values.iter().map(|v| v.abs()).fold(0.0, f64::max);
            eprintln!("{name} at r {r:?} κ {kappa:?}: P {}, {elements} elements, {beyond} fast; beats {}, largest |Ω| {most:.4e} rad/µs", keep.len(), values.len());
            if beyond > 0 {
                fast.push(format!("{name} {kappa:?}: {beyond} of {elements}"));
            }
        }
    }
    assert!(fast.is_empty(), "elements with a fast part in the frame: {fast:?}");
}

/// The decay is the photons' (model/selected.rs, # The generator by beats): at a moving point off the centre and at the
/// time t = 3.7 µs, i(Ĥ − Ĥ†) of the middle's generator is Σ_class Σ_a L_a(t)†L_a(t) of the same node's photon
/// operators, to the rounding of the products (n u max|Γ|, n the states).
#[test]
fn the_selected_decay_is_its_photons() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let (r, kappa) = POINTS[1];
        let n = keep.len();
        let middle = run.middle(0, r, 3.7);
        let path = middle.path(kappa, [0.0; 3]);
        let mut h = Mat::zeros(n, n);
        middle.generator(&path, 0.0, &mut h);
        let node = middle.node(&path, 0.0, false);
        let (values, _) = run.components();
        let mut gamma = Mat::zeros(n, n);
        for class in &node.classes {
            for axis in &class.axes {
                let mut l = Mat::zeros(n, n);
                for jump in axis {
                    let f = if jump.beat == 0 { C64::ONE } else { C64::cis(-values[jump.beat] * 3.7) };
                    for e in &jump.elements {
                        l.data[e.at] += f * e.value;
                    }
                }
                gamma.add_scaled(C64::ONE, &l.adjoint_product(&l));
            }
        }
        let decay = Mat::from_fn(n, n, |a, b| C64::I * (h[(a, b)] - h[(b, a)].conj()));
        let size = gamma.max_abs();
        let worst = (0..n * n).map(|x| (decay.data[x] - gamma.data[x]).abs()).fold(0.0, f64::max);
        eprintln!("{name}: |i(H − H†) − Σ L†L| {worst:.3e} of {size:.3e}");
        assert!(worst <= n as f64 * crate::units::UNIT_ROUNDOFF * size * 16.0, "{name}: {worst:e} of {size:e}");
    }
}

/// A lone tweezer's model (R1_TWEEZER, R1_SIGMA_P: the frame's reference, no axis of its own; scene.rs `tweezer_axis`,
/// `Rs::new`): its run as the solve makes it, the node at the trap's centre at rest holds the zero beat alone (the
/// light shift is static in the tweezer's frame), its photons in two classes, the tweezer's co-rotating (n_tw = 1)
/// and counter-rotating (n_tw = −1) scattering, every number finite, and its decay i(Ĥ − Ĥ†) the classes' Σ L†L to the
/// rounding of the products and of H's Hermitian part (the request's forms Hermitian to their own rounding: σ⁺'s vector
/// shift holds off-diagonal ones), n u (max|Γ| + max|Ĥ|) times 16 as `the_selected_decay_is_its_photons` bounds it.
#[test]
fn a_lone_tweezer_s_node_is_its_two_photon_classes() {
    for name in ["R1_TWEEZER", "R1_SIGMA_P"] {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let chief = crate::solve::chief(&request);
        let run = crate::solve::run_of(&request, crate::solve::alone(chief)).unwrap();
        let n = run.rs.keep.len();
        let node = run.node([0.0; 3], [0.0; 3], true);
        let finite = |z: &C64| z.re.is_finite() && z.im.is_finite();
        assert!(node.generator.iter().all(|(_, m)| m.data.iter().all(finite)), "{name}: the generator finite");
        let fasts: Vec<[i64; 2]> = node.classes.iter().map(|c| c.fast).collect();
        let mut gamma = Mat::zeros(n, n);
        for class in &node.classes {
            for axis in &class.axes {
                let mut l = Mat::zeros(n, n);
                for jump in axis {
                    for e in &jump.elements {
                        assert!(finite(&e.value) && finite(&e.kick), "{name}: a photon's element finite");
                        l.data[e.at] += e.value;
                    }
                }
                gamma.add_scaled(C64::ONE, &l.adjoint_product(&l));
            }
        }
        let h = &node.generator[0].1;
        let decay = Mat::from_fn(n, n, |a, b| C64::I * (h[(a, b)] - h[(b, a)].conj()));
        let (size, whole) = (gamma.max_abs(), h.max_abs());
        let worst = (0..n * n).map(|x| (decay.data[x] - gamma.data[x]).abs()).fold(0.0, f64::max);
        eprintln!("{name}: P {n}, beats {:?}, classes {fasts:?}; |i(H − H†) − Σ L†L| {worst:.3e} of |Γ| {size:.3e}, |H| {whole:.3e}", node.generator.iter().map(|g| g.0).collect::<Vec<_>>());
        assert_eq!(node.generator.len(), 1, "{name}: the zero beat alone");
        assert!(fasts.contains(&[0, 1]) && fasts.contains(&[0, -1]) && fasts.len() == 2, "{name}: {fasts:?}");
        assert!(size > 0.0 && worst <= n as f64 * crate::units::UNIT_ROUNDOFF * (size + whole) * 16.0, "{name}: {worst:e} of {size:e}, {whole:e}");
    }
}

/// The node is the model's value in its frame (module documentation, # One cluster, one frame): a jump's node (with
/// the photon forms) at the time t, every beat at its phase, against the same fold's components summed at their
/// phases in one group (`Local::contract`, each component's weight e^{−iF_c t}, F_c its key's frequency in the frame)
/// put in the frame, x_a = e^{iθ_a t} ψ_a: H_eff's Hermitian part with θ on its diagonal, H_eff's gradient, each
/// class's axis L(t) and ∇L(t) and its photon form K(t), each channel up to one phase of modulus one (its fast index's,
/// common to its elements). At t = 0 that sum is `Local::now(0)` itself. At rest (κ = 0) F_c is the component's own
/// frequency, its beams' recoil in, and the sum is `now(t)` at every time (the one Doppler shift per beam, model/rs.rs #
/// The frequencies); at a moving point F_c is the frame's, at rest: the node's beats do not hold the Doppler shifts K·κ,
/// its motion is its place along the path. At the centre and off it at rest and at the moving point, t = 0, 0.37 and
/// −1.9 µs. The two are one set of sums taken in two
/// orders (by beats, one group): they differ by rounding alone, bounded as `now_is_the_components_at_their_phases`
/// bounds it -- per component 64u(1 + |F_c t| + 2 max|θ t|) max|X⁽ᶜ⁾| summed over the quantity's components, and κ(V)u
/// relative to its largest entry (10⁻¹⁰).
#[test]
fn the_node_is_the_model_now_in_its_frame() {
    use crate::floquet::frame::FrameKey;
    use crate::model::rs::ModelPart;
    use crate::units::UNIT_ROUNDOFF;
    let mut worst = 0.0_f64;
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let n = keep.len();
        let frequency = |key: &Key| -> f64 {
            let v: FrameKey = key.iter().map(|&k| i64::from(k)).chain(std::iter::once(0)).collect();
            run.value(&run.frame.split(&v))
        };
        for (r, kappa) in [([0.0; 3], [0.0; 3]), (POINTS[1].0, [0.0; 3]), POINTS[1]] {
            let node = run.node(r, kappa, true);
            // The beats as they stand once the node is made (a beat is made where first met).
            let (values, _) = run.components();
            let local = run.rs.local(&run.scene, &run.scene.point_at(r, kappa));
            let components = local.components();
            for t in [0.0, 0.37, -1.9] {
                let phase = |beat: usize| if beat == 0 { C64::ONE } else { C64::cis(-values[beat] * t) };
                let turn = |x: &Mat| Mat::from_fn(n, n, |a, b| C64::cis(run.theta[a] * t) * x[(a, b)] * C64::cis(-run.theta[b] * t));
                let theta = run.theta.iter().map(|x| (x * t).abs()).fold(0.0, f64::max);
                let bound = |parts: &mut dyn Iterator<Item = (&ModelPart, &Mat)>, scale: f64| -> f64 {
                    parts.map(|(p, x)| 64.0 * UNIT_ROUNDOFF * (1.0 + (frequency(&p.key) * t).abs() + 2.0 * theta) * x.max_abs()).sum::<f64>() + 1e-10 * scale
                };
                let rest = kappa == [0.0; 3];
                let sum = local.contract(&|key, f| (Vec::new(), 0.0, C64::cis(-(if rest { f } else { frequency(key) }) * t)), Making { model: true, kicks: true, lift: false });
                if rest {
                    let now = local.now(t);
                    assert!(sum.effective.first().is_none_or(|p| (0..n * n).all(|x| (p.value.data[x] - now.hamiltonian.data[x]).abs() <= 1e-12 * now.hamiltonian.max_abs())), "{name}: at rest the sum is now(t)");
                }
                if t == 0.0 {
                    let now = local.now(0.0);
                    assert!(sum.effective.first().is_none_or(|p| p.value == now.hamiltonian), "{name}: the sum at t = 0 is now(0)");
                }
                let zero = Mat::zeros(n, n);
                let mut ratios = [0.0_f64; 4];
                // H_eff: the generator's Hermitian part with θ on its diagonal.
                let mut g = Mat::zeros(n, n);
                for (beat, m) in &node.generator {
                    g.add_scaled(phase(*beat), m);
                }
                let h = Mat::from_fn(n, n, |a, b| (g[(a, b)] + g[(b, a)].conj()) * 0.5 + if a == b { C64::real(run.theta[a]) } else { C64::ZERO });
                let want = turn(sum.effective.first().map_or(&zero, |p| &p.value));
                let scale = want.max_abs();
                ratios[0] = apart(&h, &want) / bound(&mut components.effective.iter().map(|p| (p, &p.value)), scale).max(f64::MIN_POSITIVE);
                for i in 0..3 {
                    let mut d = Mat::zeros(n, n);
                    for (beat, m) in &node.gradient {
                        d.add_scaled(phase(*beat), &m[i]);
                    }
                    let want = turn(sum.effective.first().map_or(&zero, |p| &p.gradient[i]));
                    let scale = want.max_abs();
                    ratios[1] = ratios[1].max(apart(&d, &want) / bound(&mut components.effective.iter().map(|p| (p, &p.gradient[i])), scale).max(f64::MIN_POSITIVE));
                }
                // The channels: L, ∇L and K of each class's axis against the channel (emission, fast index).
                let mut seen = 0;
                for (class, a) in node.classes.iter().flat_map(|c| (0..3).map(move |a| (c, a))) {
                    let axis = &class.axes[a];
                    let summed = |m: &dyn Fn(&Element) -> C64| -> Mat {
                        let mut out = Mat::zeros(n, n);
                        for jump in axis {
                            for e in &jump.elements {
                                out.data[e.at] += phase(jump.beat) * m(e);
                            }
                        }
                        out
                    };
                    let channel = sum.jumps.iter().find(|c| c.emission == a && c.fast == class.fast);
                    if axis.is_empty() && channel.is_none() {
                        continue;
                    }
                    seen += 1;
                    let channel = channel.expect("the channel in the sum");
                    let part = components.jumps.iter().find(|c| c.emission == a && c.fast == class.fast).expect("the channel's components");
                    let (value, kick) = (&channel.parts[0], channel.kicks.first());
                    let l = summed(&|e| e.value);
                    let want = turn(&value.value);
                    // The channel's phase, read where L is largest.
                    let at = (0..n * n).max_by(|x, y| want.data[*x].abs().total_cmp(&want.data[*y].abs())).expect("an element");
                    let common = l.data[at] * want.data[at].conj() * (1.0 / want.data[at].norm_sqr());
                    let turned = |x: &Mat| {
                        let y = turn(x);
                        Mat::from_fn(n, n, |a, b| common * y[(a, b)])
                    };
                    let scale = want.max_abs();
                    let unit = (common.abs() - 1.0).abs() * scale;
                    ratios[2] = ratios[2].max((apart(&l, &turned(&value.value)) + unit) / bound(&mut part.parts.iter().map(|p| (p, &p.value)), scale).max(f64::MIN_POSITIVE));
                    for i in 0..3 {
                        let d = summed(&|e| e.gradient[i]);
                        let want = turned(&value.gradient[i]);
                        let scale = want.max_abs();
                        ratios[2] = ratios[2].max(apart(&d, &want) / bound(&mut part.parts.iter().map(|p| (p, &p.gradient[i])), scale).max(f64::MIN_POSITIVE));
                    }
                    let k = summed(&|e| e.kick);
                    let want = turned(kick.map_or(&zero, |k| &k.value));
                    let scale = want.max_abs();
                    ratios[3] = ratios[3].max(apart(&k, &want) / bound(&mut part.kicks.iter().map(|p| (p, &p.value)), scale).max(f64::MIN_POSITIVE));
                }
                assert_eq!(seen, sum.jumps.len(), "{name}: the node's channels are the sum's");
                eprintln!("{name} r {r:?} κ {kappa:?} t {t}: of the bound H {:.1e} ∇H {:.1e} L, ∇L {:.1e} K {:.1e} ({seen} channels)", ratios[0], ratios[1], ratios[2], ratios[3]);
                worst = ratios.iter().fold(worst, |w, e| w.max(*e));
            }
        }
    }
    assert!(worst <= 1.0, "the node and the components at their phases apart by {worst:.2} of their rounding's bound");
}

/// The largest entry of |a − b|.
fn apart(a: &Mat, b: &Mat) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, f64::max)
}

/// The force at the frame's time zero is the model's (model/rs.rs `ModelAt::force`, the Lindbladian's force observable):
/// −Re tr(Ĝρ) of the middle (`Middle::force`) with every class's absorbed momentum Σ_a Im⟨u_a|∇u_a⟩ -- what the jumps
/// and the step's scattering carry between them -- at the centre and at a moving point, on a state of P with every
/// amplitude set, to the rounding of the traces.
#[test]
fn the_selected_force_at_time_zero_is_the_model_s() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let n = keep.len();
        let psi = spread(n);
        let mut rho = vec![C64::ZERO; n * n];
        crate::step::density(&psi, &mut rho);
        for (r, kappa) in POINTS {
            let middle = run.middle(0, r, 0.0);
            let path = middle.path(kappa, [0.0; 3]);
            let ehrenfest = middle.force(&path, 0.0, &rho);
            let node = middle.node(&path, 0.0, false);
            let absorbed = run.read(|b| photons(&node, &b.value, 0.0, &psi)).iter().fold([0.0; 3], |s, p| std::array::from_fn(|i| s[i] + p.mean[i]));
            let engine: V3 = std::array::from_fn(|i| -ehrenfest[i] + absorbed[i]);
            let model = run.rs.local(&run.scene, &run.scene.point_at(r, kappa)).now(0.0).force(&rho);
            let size = model.iter().chain(&ehrenfest).map(|x| x.abs()).fold(0.0, f64::max);
            let worst = (0..3).map(|i| (engine[i] - model[i]).abs()).fold(0.0, f64::max);
            eprintln!("{name} κ {kappa:?}: engine {engine:?}, model {model:?}, difference {worst:.2e}");
            assert!(worst <= 1e-12 * size.max(f64::MIN_POSITIVE), "{name}: {worst:e} of {size:e}");
        }
    }
}

struct Anywhere;
impl Domain for Anywhere {
    fn runaway(&self, _: f64) -> bool {
        false
    }
}

/// An f64 from the environment, or its default.
fn env(key: &str, default: f64) -> f64 {
    std::env::var(key).ok().map(|s| s.parse().unwrap()).unwrap_or(default)
}

/// The static model's mean force at a velocity (RUSTPLAN §6 C2, "the static F at the Doppler point"): the checkpoint's
/// solve of the same model (`Rs::at` at the point and momentum, floquet's frame with its Doppler shifts, Liouville
/// assembly, the rounding rule's elements, krylov's solve) -- its force's z component (1/µm/µs), the time-averaged
/// mean force on the atom's path through r at the velocity of κ.
fn static_force(run: &Selected, r: V3, kappa: V3) -> f64 {
    let scene = &run.scene;
    let doppler = scene.doppler(kappa);
    let components = scene.components(r, &doppler, 1.0);
    let frame = crate::floquet::Frame::new(scene, &Hilbert::full(scene, &components), &doppler);
    let liouville = crate::floquet::Liouville::new(&frame, &run.rs.at(scene, &components).hilbert());
    let elements = crate::floquet::select(&liouville);
    let system = crate::floquet::assemble(&liouville, &elements);
    let solved = crate::krylov::solve::solve(&system).expect("a consistent system");
    crate::floquet::readouts(&liouville, &system, &solved.rho).force[2]
}

/// A dragged atom's mean force is the static model's (RUSTPLAN §6 C2: "轨迹平均 = 模型主方程" on the trajectory's level):
/// tilt0 (V3C_REQ's, PYSEL's P20), its tweezer's waist DRAG_WAIST µm (1e5: its light the focus's along the drag), an
/// atom held at the velocity v along z (its momentum set back after every step, its position pinned to z₀ + vt: the
/// master equation's atom, infinitely heavy at exactly v), DRAG_COUNT starts spread over the pattern's period, stepped at
/// τ = DRAG_TAU µs with no Brownian draw (the diffusion's mean is none), its force the steps' momentum's change over
/// DRAG_WINDOW µs after DRAG_BURN µs -- against the static model's at the same starts (`static_force`, their mean), for
/// v of DRAG_VS (m/s), within their interval at the confidence z over the velocities.
#[test]
#[ignore]
fn a_dragged_atom_s_mean_force_is_the_static_model_s() {
    let (_, mut request, keep) = scenes().into_iter().find(|s| s.0 == "tilt0").expect("tilt0");
    let waist = env("DRAG_WAIST", 1e5);
    let t = request.trap_beam;
    request.beams[t].waist = Some(waist);
    let run = run_of(&request, &keep);
    let hm = run.hbar_over_mass();
    let vs: Vec<f64> = std::env::var("DRAG_VS").ok().map_or(vec![-0.1, 0.0, 0.1], |s| s.split(',').map(|x| x.trim().parse().unwrap()).collect());
    let count = env("DRAG_COUNT", 32.0) as u64;
    let (tau, burn, window) = (env("DRAG_TAU", 0.3256), env("DRAG_BURN", 40.0), env("DRAG_WINDOW", 100.0));
    let kz: f64 = request.beams.iter().enumerate().filter(|(l, b)| *l != t && b.direction[2].abs() > 0.5).map(|(_, b)| b.wave_number).sum();
    let period = 2.0 * std::f64::consts::PI / kz;
    let chief = crate::solve::chief(&request);
    eprintln!("drag: P {}, waist {waist}, period {period:.5} µm, τ {tau}, count {count}, burn {burn}, window {window}", keep.len());
    let mut beyond = Vec::new();
    for &v in &vs {
        let clock = std::time::Instant::now();
        let kappa = [0.0, 0.0, v / hm];
        let starts: Vec<f64> = (0..count).map(|i| period * (i as f64 + 0.5) / count as f64).collect();
        let forces: Vec<f64> = starts.par_iter().enumerate().map(|(i, &z0)| {
            let mut t = Trajectory::new(&run, chief, [0.0, 0.0, z0], kappa, 20261119, i as u64);
            let mut scratch = Scratch::default();
            let (mut made, mut from) = (0.0, None);
            while t.time < burn + window - 1e-9 {
                let step = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Anywhere)).expect("a step");
                let m = step.momentum();
                t = step.end;
                if from.is_some() {
                    made += m.whole[2];
                }
                t.kappa = kappa;
                t.r = [0.0, 0.0, z0 + v * t.time];
                if from.is_none() && t.time >= burn - 1e-9 {
                    from = Some(t.time);
                }
            }
            made / (t.time - from.expect("the window"))
        }).collect();
        let n = forces.len() as f64;
        let mean = forces.iter().sum::<f64>() / n;
        let error = (forces.iter().map(|f| (f - mean).powi(2)).sum::<f64>() / (n - 1.0) / n).sqrt();
        let model = starts.par_iter().map(|&z0| static_force(&run, [0.0, 0.0, z0], kappa)).sum::<f64>() / n;
        let within = crate::estimate::Spread { error, groups: n }.interval_over(vs.len() as f64);
        eprintln!("v {v:+.3} m/s: engine F_z {mean:.6e} ± {error:.2e}, static model {model:.6e}, difference {:.2e} (interval {within:.2e}), {:.1} s", mean - model, clock.elapsed().as_secs_f64());
        if (mean - model).abs() > within {
            beyond.push(v);
        }
    }
    assert!(beyond.is_empty(), "velocities beyond the interval: {beyond:?}");
}

/// A solve's first result (`solve::solve_on` on the run), the solve stopped there (its callback's own unwind, caught
/// here), with its wall time: T, the populations and the loss at every output said, and whether all are numbers.
fn first_result(name: &str, request: &Request, run: &Selected) -> bool {
    struct First;
    let clock = std::time::Instant::now();
    let mut first: Option<crate::solve::Solution> = None;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::solve::solve_on(request, run, &mut |s, _| {
            first = Some(s.clone());
            std::panic::panic_any(First);
        })
    }));
    let result = match outcome {
        Ok(Ok(s)) => s,
        Ok(Err(why)) => panic!("{name}: refused: {why}"),
        Err(payload) if payload.is::<First>() => first.expect("the first result"),
        Err(payload) => std::panic::resume_unwind(payload),
    };
    let last = result.temperature_total_uk.len() - 1;
    let finite = result.populations.iter().flatten().chain(&result.loss_probability).chain(&result.temperature_total_uk).all(|x| x.is_finite());
    eprintln!(
        "{name}: first result after {:.1} s: {} trajectories, τ {:.4} µs; T_total {:.3} µK at the last output, loss {:.4}, rows {:?}; all numbers {finite}",
        clock.elapsed().as_secs_f64(),
        result.trajectories,
        result.time_step_us,
        result.temperature_total_uk[last],
        result.loss_probability[last],
        result.populations[last].iter().enumerate().filter(|(_, x)| **x > 1e-3).map(|(s, x)| format!("{}:{x:.3}", request.states[s])).collect::<Vec<_>>()
    );
    finite
}

/// The six scenes' first checkpoint (RUSTPLAN §6 C2): each scene's solve on PYSEL's P (V3E_SCENE one of them) to its first
/// result, which gives T, the populations and the loss at every output, all numbers, no panic.
#[test]
#[ignore]
fn the_six_scenes_first_checkpoint_is_numbers() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        assert!(first_result(name, &request, &run), "{name}: a number that is none");
    }
}

/// The default scene's first solve (RUSTPLAN §6 C2): the app's start-up scene (V3E_DEFAULT: a native request's path, the
/// accept kit's S1_5ms) to its first result, all numbers, its run the solve's (`solve::run_of`).
#[test]
#[ignore]
fn the_default_scene_s_first_solve_is_numbers() {
    let path = std::env::var("V3E_DEFAULT").expect("V3E_DEFAULT: the default scene's native request");
    let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
    let chief = crate::solve::chief(&request);
    let run = crate::solve::run_of(&request, std::array::from_fn(|s| f64::from(u8::from(s == chief)))).unwrap();
    assert!(first_result("default", &request, &run), "a number that is none");
}

/// A step's time on the selected model (a measurement, ignored; V3E_SCENE one of the six): from the reference state at
/// the trap's centre at the thermal speed along x, each on one thread, timed (µs, the least of five): the model at a
/// point -- the fold there (`Rs::local`), its contraction by the lab's frequencies, the whole node (`Selected::node`)
/// and a jump's node (with the photon forms) --, now(t)'s contraction beside them, and a whole step at τ = V3E_TAU µs
/// (0.3256), the three nodes it makes (its middle and its two outer nodes) apart; the step's candidate tones and their
/// largest beat.
#[test]
#[ignore]
fn a_step_s_time_on_the_selected_model() {
    let tau = env("V3E_TAU", 0.3256);
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let chief = crate::solve::chief(&request);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let (r, kappa) = ([0.1, 0.2, 0.3], [speed, 0.0, 0.0]);
        let least = |f: &mut dyn FnMut()| (0..5).map(|_| {
            let clock = std::time::Instant::now();
            f();
            clock.elapsed().as_secs_f64() * 1e6
        }).fold(f64::INFINITY, f64::min);
        let point = run.scene.point_at(r, kappa);
        let local = run.rs.local(&run.scene, &point);
        let fold = least(&mut || {
            std::hint::black_box(run.rs.local(&run.scene, &point));
        });
        let now = least(&mut || {
            std::hint::black_box(local.now(0.37));
        });
        let model = least(&mut || {
            std::hint::black_box(run.at(r, kappa, Making { model: true, kicks: false, lift: false }));
        });
        let node = least(&mut || {
            std::hint::black_box(run.node(r, kappa, false));
        });
        let jump = least(&mut || {
            std::hint::black_box(run.node(r, kappa, true));
        });
        let t = Trajectory::new(&run, chief, r, kappa, 7, 0);
        let mut scratch = Scratch::default();
        let step = least(&mut || {
            let m = Midway::new(&run, &t, tau).unwrap();
            std::hint::black_box(m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Anywhere).unwrap());
        });
        let middle = run.middle(0, r, 0.0);
        let path = middle.path(kappa, kappa.map(|k| run.hbar_over_mass() * k));
        let mut h = Mat::zeros(keep.len(), keep.len());
        for s in [0.0, -0.5 * tau, 0.5 * tau] {
            middle.generator(&path, s, &mut h);
        }
        let (mut beats, mut keys) = (vec![], vec![]);
        middle.beats(&path, &mut beats, &mut keys);
        eprintln!(
            "{name}: P {}; fold {fold:.0}, now's contraction {:.0}, the node's {:.0}, node {node:.0}, a jump's node {jump:.0} µs; step {step:.0} µs, of it three nodes {:.0} and the rest {:.0}; {} candidate tones, the largest {:.4e} rad/µs",
            keep.len(),
            now,
            model - fold.min(model),
            3.0 * node,
            step - 3.0 * node,
            beats.len(),
            beats.iter().copied().fold(0.0, f64::max)
        );
    }
}

#[path = "golden.rs"]
mod golden;

/// The selection's report (`Selected::report`): P state by state, the states left out, and a point not judged said as
/// such -- R10_NODE's edge, where no light reaches the atom (`Selection::notes`) --, never as zero errors.
#[test]
fn the_report_lists_p_the_states_left_out_and_the_notes() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R10_NODE.json");
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let request = Request::from_native(&line).unwrap();
    let run = crate::solve::run_of(&request, crate::solve::alone(crate::solve::chief(&request))).unwrap();
    let report = run.report();
    eprint!("{report}");
    assert!(report.contains("P 的态：") && report.contains("删去 "), "{report}");
    let notes = run.report.notes();
    assert!(!notes.is_empty() && notes.iter().all(|n| report.contains(n.as_str())), "{report}");
    assert!(report.contains("不判"), "{report}");
}
