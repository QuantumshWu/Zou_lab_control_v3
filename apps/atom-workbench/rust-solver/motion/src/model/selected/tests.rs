//! The selected model's checks (model/selected.rs): its frame keeps the fold's elements slow, its decay is its photons',
//! its force at the frame's time zero is the model's (`Now::force`), a dragged atom's mean force is the static model's
//! (RUSTPLAN §6 C2), and the six scenes' first checkpoint. Each scene's run is the solve's (`solve::run_of`: P by
//! `select::choose`), its P PYSEL's.
use super::*;
use crate::step::{Brownian, Domain, Dynamics, Middle, Midway, Scratch, Trajectory};


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
        // The run's own P (`run_of` chooses it; PYSEL's `keep` is said beside it).
        let keep = run.rs.keep.clone();
        for (r, kappa) in POINTS {
            let model = run.at(r, kappa, Making { model: true, effective: true, kicks: false, lift: false, gradients: true });
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
            let most = values.iter().map(|v| v.abs()).fold(0.0, crate::linalg::larger);
            eprintln!("{name} at r {r:?} κ {kappa:?}: P {}, {elements} elements, {beyond} fast; beats {}, largest |Ω| {most:.4e} rad/µs", keep.len(), values.len());
            if beyond > 0 {
                fast.push(format!("{name} {kappa:?}: {beyond} of {elements}"));
            }
        }
    }
    assert!(fast.is_empty(), "elements with a fast part in the frame: {fast:?}");
}

/// The decay is the photons' (model/selected.rs, # The generator by beats): at the centre at rest and at a moving point
/// off it, at the time t = 3.7 µs, i(Ĥ − Ĥ†) of the middle's generator is Σ_class Σ_a L_a(t)†L_a(t) of the same node's
/// photon operators but for the products of two components at a beat the model drops (model/classify.rs, # The drop
/// tier: a beat dropped is dropped from the decay as from H, its bound booked, `Judgment::dropped`), entry by entry
/// within the two computations' own rounding (Higham: a computed sum of K terms errs by γ_K times the sum of their
/// moduli; units.rs `rounding`): each generator entry a ← b and b ← a by its fold's terms' moduli and count at the point
/// (`Selected::structure_at`, the structure's rule: # The structure's rounding) -- their Hermitian parts cancel in
/// i(Ĥ − Ĥ†) but for that rounding --, and the decay's products both ways, Σ_k (Σ_j |ℓ_j,ka|)(Σ_j |ℓ_j,kb|) of the photon
/// components' moduli, K the generator's beats, the components and the n rows of a product. (Until 2026-11-22 the
/// products were all taken and the allowance n u max|Γ| sixteen times over: tilt2, whose judgment drops the beats ±3ν
/// to ±6ν of its Larmor ν = 13.19 rad/µs, read 4.3·10⁻¹³ of Γ's 2.72 -- 30 times its entries' rounding, the dropped
/// beats' products, no rounding.)
#[test]
fn the_selected_decay_is_its_photons() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        // The run's own P (`run_of` chooses it; PYSEL's `keep` is said beside it).
        let n = run.rs.keep.len();
        let (values, _) = run.components();
        let dropped: Vec<f64> = run.read(|b| run.judgment.tiers.iter().filter(|t| t.1 == Tier::Drop).map(|t| b.value[t.0].abs()).collect());
        let kept = |omega: f64| !dropped.iter().any(|&d| (omega.abs() - d).abs() <= 1e-9 * d);
        let t = 3.7;
        for (r, kappa) in POINTS {
            let middle = run.middle(0, r, t);
            let path = middle.path(kappa, [0.0; 3]);
            let mut h = Mat::zeros(n, n);
            middle.generator(&path, 0.0, &mut h);
            let node = middle.node(&path, 0.0);
            // Σ L†L of the components' products at kept beats, and the products' moduli.
            let (mut gamma, mut products, mut terms) = (Mat::zeros(n, n), vec![0.0_f64; n * n], node.generator.len());
            for class in &node.classes {
                for axis in &class.axes {
                    let components: Vec<(f64, Mat, Vec<f64>)> = axis
                        .iter()
                        .map(|jump| {
                            let (omega, f) = if jump.beat == 0 { (0.0, C64::ONE) } else { (values[jump.beat], C64::cis(-values[jump.beat] * t)) };
                            let (mut m, mut size) = (Mat::zeros(n, n), vec![0.0_f64; n * n]);
                            for e in &jump.elements {
                                m.data[e.at] += f * e.value;
                                size[e.at] += e.value.abs();
                            }
                            (omega, m, size)
                        })
                        .collect();
                    terms += axis.len() + n;
                    for (wl, ml, sl) in &components {
                        for (wr, mr, sr) in &components {
                            if !kept(wr - wl) {
                                continue;
                            }
                            gamma.add_scaled(C64::ONE, &ml.adjoint_product(mr));
                            for a in 0..n {
                                for b in 0..n {
                                    products[a * n + b] += (0..n).map(|k| sl[k * n + a] * sr[k * n + b]).sum::<f64>();
                                }
                            }
                        }
                    }
                }
            }
            let decay = Mat::from_fn(n, n, |a, b| C64::I * (h[(a, b)] - h[(b, a)].conj()));
            let (_, scales) = run.structure_at(r, kappa);
            let rounded = |x: usize| scales.generator.iter().map(|(s, k)| crate::units::rounding(k[x] + node.generator.len()) * s[x]).sum::<f64>();
            let gamma_k = crate::units::rounding(terms);
            let (mut worst, mut most) = (0.0_f64, 0.0_f64);
            for a in 0..n {
                for b in 0..n {
                    let off = (decay[(a, b)] - gamma[(a, b)]).abs();
                    let bound = rounded(a * n + b) + rounded(b * n + a) + 2.0 * gamma_k * products[a * n + b];
                    worst = crate::linalg::larger(worst, off);
                    if off > 0. {
                        most = crate::linalg::larger(most, off / bound);
                    }
                    assert!(off <= bound, "{name} at {r:?}, {kappa:?} ({a}, {b}): {off:e} against the rounding's {bound:e}");
                }
            }
            eprintln!("{name} at {r:?}, {kappa:?}: |i(H − H†) − Σ L†L| {worst:.3e} of {:.3e}, at most {most:.3} of its entry's rounding (γ_{terms}); {} beats dropped", gamma.max_abs(), dropped.len());
        }
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
        let worst = (0..n * n).map(|x| (decay.data[x] - gamma.data[x]).abs()).fold(0.0, crate::linalg::larger);
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
/// relative to its largest entry (10⁻¹⁰). The node with its direct entries one by one: what their span leaves out is
/// booked, not rounding, and read by its own test (`the_direct_entries_spanned_are_the_entries_within_what_the_span_books`).
#[test]
fn the_node_is_the_model_now_in_its_frame() {
    use crate::floquet::frame::FrameKey;
    use crate::model::rs::ModelPart;
    use crate::units::UNIT_ROUNDOFF;
    let mut worst = 0.0_f64;
    for (name, request, keep) in scenes() {
        let mut run = run_of(&request, &keep);
        run.lab.spans.clear();
        // The run's own P (`run_of` chooses it; PYSEL's `keep` is said beside it).
        let keep = run.rs.keep.clone();
        let n = keep.len();
        let frequency = |key: &Key| -> f64 {
            let v: FrameKey = key.iter().map(|&k| i64::from(k)).chain(std::iter::once(0)).collect();
            run.value_of(&run.frame.split(&v))
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
                let theta = run.theta.iter().map(|x| (x * t).abs()).fold(0.0, crate::linalg::larger);
                let bound = |parts: &mut dyn Iterator<Item = (&ModelPart, &Mat)>, scale: f64| -> f64 {
                    parts.map(|(p, x)| 64.0 * UNIT_ROUNDOFF * (1.0 + (frequency(&p.key) * t).abs() + 2.0 * theta) * x.max_abs()).sum::<f64>() + 1e-10 * scale
                };
                let rest = kappa == [0.0; 3];
                let sum = local.contract(&|key, f| (Vec::new(), 0.0, C64::cis(-(if rest { f } else { frequency(key) }) * t)), Making { model: true, effective: true, kicks: true, lift: false, gradients: true });
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
                ratios[0] = apart(&h, &want) / crate::linalg::larger(bound(&mut components.effective.iter().map(|p| (p, &p.value)), scale), f64::MIN_POSITIVE);
                for i in 0..3 {
                    let mut d = Mat::zeros(n, n);
                    for (beat, m) in &node.gradient {
                        d.add_scaled(phase(*beat), &m[i]);
                    }
                    let want = turn(sum.effective.first().map_or(&zero, |p| &p.gradient[i]));
                    let scale = want.max_abs();
                    ratios[1] = crate::linalg::larger(ratios[1], apart(&d, &want) / crate::linalg::larger(bound(&mut components.effective.iter().map(|p| (p, &p.gradient[i])), scale), f64::MIN_POSITIVE));
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
                    ratios[2] = crate::linalg::larger(ratios[2], (apart(&l, &turned(&value.value)) + unit) / crate::linalg::larger(bound(&mut part.parts.iter().map(|p| (p, &p.value)), scale), f64::MIN_POSITIVE));
                    for i in 0..3 {
                        let d = summed(&|e| e.gradient[i]);
                        let want = turned(&value.gradient[i]);
                        let scale = want.max_abs();
                        ratios[2] = crate::linalg::larger(ratios[2], apart(&d, &want) / crate::linalg::larger(bound(&mut part.parts.iter().map(|p| (p, &p.gradient[i])), scale), f64::MIN_POSITIVE));
                    }
                    let k = summed(&|e| e.kick);
                    let want = turned(kick.map_or(&zero, |k| &k.value));
                    let scale = want.max_abs();
                    ratios[3] = crate::linalg::larger(ratios[3], apart(&k, &want) / crate::linalg::larger(bound(&mut part.kicks.iter().map(|p| (p, &p.value)), scale), f64::MIN_POSITIVE));
                }
                assert_eq!(seen, sum.jumps.len(), "{name}: the node's channels are the sum's");
                eprintln!("{name} r {r:?} κ {kappa:?} t {t}: of the bound H {:.1e} ∇H {:.1e} L, ∇L {:.1e} K {:.1e} ({seen} channels)", ratios[0], ratios[1], ratios[2], ratios[3]);
                worst = ratios.iter().fold(worst, |w, e| crate::linalg::larger(w, *e));
            }
        }
    }
    assert!(worst <= 1.0, "the node and the components at their phases apart by {worst:.2} of their rounding's bound");
}

/// The largest entry of |a − b|.
fn apart(a: &Mat, b: &Mat) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).abs()).fold(0.0, crate::linalg::larger)
}

/// The force at the frame's time zero is the model's (model/rs.rs `ModelAt::force`, the Lindbladian's force observable):
/// −Re tr(Ĝρ) of the middle (`Middle::force`) with every class's absorbed momentum Σ_a Im⟨u_a|∇u_a⟩ -- what the jumps
/// and the step's scattering carry between them -- at the centre and at a moving point, on a state of P with every
/// amplitude set, to the rounding of the traces: the run's with nothing dropped (model/classify.rs, # The drop tier),
/// and the run's as the solve makes it apart from that by its dropped beats' −Re tr(Ĝ_k ρ) alone (tilt2's ±3ν … ±6ν),
/// its direct entries one by one and K's sideband parts in (what their span and the parts' leaving leave out is booked,
/// not a beat: their own test reads it, `the_direct_entries_spanned_are_the_entries_within_what_the_span_books`).
#[test]
fn the_selected_force_at_time_zero_is_the_model_s() {
    for (name, request, keep) in scenes() {
        let mut dropping = run_of(&request, &keep);
        dropping.whole();
        let run = made_with(&request, &Drops::Only(Vec::new()));
        let dropped: Vec<usize> = dropping.judgment().tiers.iter().filter(|t| t.1 == crate::model::classify::Tier::Drop).map(|t| t.0).collect();
        // The run's own P (`run_of` chooses it; PYSEL's `keep` is said beside it).
        let keep = run.rs.keep.clone();
        let n = keep.len();
        let psi = spread(n);
        let mut rho = vec![C64::ZERO; n * n];
        crate::step::density(&psi, &mut rho);
        for (r, kappa) in POINTS {
            let middle = run.middle(0, r, 0.0);
            let path = middle.path(kappa, [0.0; 3]);
            let ehrenfest = middle.force(&path, 0.0, &rho);
            let node = middle.node(&path, 0.0);
            let absorbed = run.read(|b| photons(&node, &b.value, 0.0, &psi, &mut Vec::new()).iter().map(|p| p.moments).collect::<Vec<_>>()).iter().fold([0.0; 3], |s, p| std::array::from_fn(|i| s[i] + p.mean[i]));
            let engine: V3 = std::array::from_fn(|i| -ehrenfest[i] + absorbed[i]);
            let model = run.rs.local(&run.scene, &run.scene.point_at(r, kappa)).now(0.0).force(&rho);
            let size = model.iter().chain(&ehrenfest).map(|x| x.abs()).fold(0.0, crate::linalg::larger);
            let worst = (0..3).map(|i| (engine[i] - model[i]).abs()).fold(0.0, crate::linalg::larger);
            eprintln!("{name} κ {kappa:?}: engine {engine:?}, model {model:?}, difference {worst:.2e}");
            assert!(worst <= 1e-12 * crate::linalg::larger(size, f64::MIN_POSITIVE), "{name}: {worst:e} of {size:e}");
            // The run as the solve makes it: apart by its dropped beats' Ehrenfest force alone.
            let other = dropping.middle(0, r, 0.0);
            let lost = SelectedMiddle::contract(&node, |beat| dropped.contains(&beat).then_some(C64::ONE), &rho).map(|z| z.re);
            let apart = (0..3).map(|i| (other.force(&other.path(kappa, [0.0; 3]), 0.0, &rho)[i] + lost[i] - ehrenfest[i]).abs()).fold(0.0, crate::linalg::larger);
            eprintln!("{name} κ {kappa:?}: the dropped beats' force {lost:?}, the run's apart from it by {apart:.2e}");
            assert!(apart <= 1e-12 * crate::linalg::larger(size, f64::MIN_POSITIVE), "{name}: the run's force apart by {apart:e} of {size:e}");
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

/// A solve's first result (`solve::solve_on` on the run), the solve stopped there (its callback's own unwind, caught
/// here), with its wall time: T, the populations and the loss at every output said, and whether all are numbers.
fn first_result(name: &str, request: &Request, run: &Selected) -> bool {
    struct First;
    let clock = std::time::Instant::now();
    let mut first: Option<crate::solve::Solution> = None;
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        crate::solve::solve_on(request, run, &mut |m| {
            let crate::solve::Message::Checkpoint(s, _) = m else { return };
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
        // The run's own P (`run_of` chooses it; PYSEL's `keep` is said beside it).
        let keep = run.rs.keep.clone();
        let chief = crate::solve::chief(&request);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let (r, kappa) = ([0.1, 0.2, 0.3], [speed, 0.0, 0.0]);
        let least = |f: &mut dyn FnMut()| (0..5).map(|_| {
            let clock = std::time::Instant::now();
            f();
            clock.elapsed().as_secs_f64() * 1e6
        }).fold(f64::INFINITY, crate::linalg::smaller);
        let point = run.scene.point_at(r, kappa);
        let local = run.rs.local(&run.scene, &point);
        let fold = least(&mut || {
            std::hint::black_box(run.rs.local(&run.scene, &point));
        });
        let now = least(&mut || {
            std::hint::black_box(local.now(0.37));
        });
        let model = least(&mut || {
            std::hint::black_box(run.at(r, kappa, Making { model: true, effective: true, kicks: false, lift: false, gradients: true }));
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
            model - crate::linalg::smaller(fold, model),
            3.0 * node,
            step - 3.0 * node,
            beats.len(),
            beats.iter().copied().fold(0.0, crate::linalg::larger)
        );
    }
}

/// The candidate tones' keys are exact (`step::Middle::beats`; module documentation, # The generator by beats): on the six
/// scenes at the centre at rest and at a moving point, among the static beat, the candidates' members (each key at +ν and
/// its negation at −ν) and every sum of two members -- the propagation's second order --, two keys are equal where and
/// only where their beats are: equal beats to rounding, different ones apart by more than 10⁻⁹ of the largest. Said per
/// scene: the candidates, and the classes of the second order (the distinct keys among them).
#[test]
fn the_candidates_keys_sum_as_their_beats() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        for (r, kappa) in POINTS {
            let middle = run.middle(0, r, 0.0);
            let path = middle.path(kappa, kappa.map(|k| run.hbar_over_mass() * k));
            let mut h = Mat::zeros(run.rs.keep.len(), run.rs.keep.len());
            middle.generator(&path, 0.0, &mut h);
            let (mut beats, mut keys) = (vec![], vec![]);
            middle.beats(&path, &mut beats, &mut keys);
            let width = keys.first().map_or(0, Vec::len);
            assert!(keys.iter().all(|k| k.len() == width), "{name}: one length");
            let mut members: Vec<(Vec<i64>, f64)> = vec![(vec![0; width], 0.0)];
            for (k, &nu) in keys.iter().zip(&beats) {
                members.push((k.clone(), nu));
                members.push((k.iter().map(|x| -x).collect(), -nu));
            }
            let mut all = members.clone();
            for (i, a) in members.iter().enumerate().skip(1) {
                for b in &members[i..] {
                    all.push((a.0.iter().zip(&b.0).map(|(x, y)| x + y).collect(), a.1 + b.1));
                }
            }
            let top = all.iter().map(|x| x.1.abs()).fold(0.0, crate::linalg::larger);
            for (i, a) in all.iter().enumerate() {
                for b in &all[i + 1..] {
                    let apart = (a.1 - b.1).abs();
                    if a.0 == b.0 {
                        assert!(apart <= 64.0 * f64::EPSILON * top, "{name}: one key, beats {} and {} apart", a.1, b.1);
                    } else {
                        assert!(apart > 1e-9 * top, "{name}: keys {:?} and {:?}, one beat {} ({})", a.0, b.0, a.1, b.1);
                    }
                }
            }
            let mut classes: Vec<&Vec<i64>> = all.iter().map(|x| &x.0).collect();
            classes.sort();
            classes.dedup();
            eprintln!("{name} r {r:?}: {} candidates, {} classes of the second order", beats.len(), classes.len());
        }
    }
}

/// The readout is the lift (module documentation, # The rest of the model): `Selected::rows` on the thread's buffers,
/// against the same populations from the model by components (`Selected::at` making the lift, each part's elements by
/// fast index at their slow beats' phases, `rs::lifted`: (M ρ M†)_kk of the density), on the six scenes at the centre
/// at rest and at a moving point, a spread state, three times: the same numbers within the two computations' own
/// rounding. The readout takes |(M ψ)_k|² of the pure state (`Selected::lifted`), the other M ρ M† (Higham: a computed
/// sum of K terms errs by γ_K times the sum of their moduli; units.rs `rounding`): per fast index and row k of Q, both
/// within γ_{2n+6} S_k² of the exact, S_k = Σ_a |M_ka| |ψ_a| / ‖ψ‖ (each element at its phase one product more, the
/// state over its norm one), P's rows within γ_4 |ψ_a|²/‖ψ‖²; over their sum, each row's own and its share of the sum's.
/// (Until 2026-11-23 the readout was the density's to the bit.)
#[test]
fn the_readout_is_the_lift_by_fast_index() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let n = run.rs.keep.len();
        let psi = spread(n);
        for (r, kappa) in POINTS {
            for t in [0.0, 0.37, -1.9] {
                let model = run.at(r, kappa, Making { model: false, effective: false, kicks: false, lift: true, gradients: true });
                let mut rho = vec![C64::ZERO; n * n];
                crate::step::density(&psi, &mut rho);
                let mut expected = [0.0; STATES];
                for (a, &s) in model.states.iter().enumerate() {
                    expected[s] = rho[a * n + a].re;
                }
                let q = model.folded.len();
                let mut by_fast: Vec<([i64; 3], Mat)> = Vec::new();
                for part in &model.lift {
                    let table = run.elements(&part.key, Of::Lift);
                    let beats: Vec<([i64; 3], f64)> = run.read(|b| table.iter().map(|&i| (b.exact[i].0, b.value[i])).collect());
                    for (x, &(fast, value)) in beats.iter().enumerate() {
                        let z = part.value.data[x];
                        if z == C64::ZERO {
                            continue;
                        }
                        let slow = value - run.value_of(&(fast, vec![]));
                        let i = match by_fast.iter().position(|f| f.0 == fast) {
                            Some(i) => i,
                            None => {
                                by_fast.push((fast, Mat::zeros(q, n)));
                                by_fast.len() - 1
                            }
                        };
                        by_fast[i].1.data[x] += z * C64::cis(-slow * t);
                    }
                }
                for (_, m) in &by_fast {
                    let lifted = crate::model::rs::lifted(&model.states, &model.folded, m, &rho);
                    for &s in &model.folded {
                        expected[s] += lifted[s];
                    }
                }
                // Each row's rounding before the sum (the two computations' together), and the sum's.
                let norm = norm_sqr(&psi);
                let mut bound = [0.0; STATES];
                for (a, &s) in model.states.iter().enumerate() {
                    bound[s] = 2.0 * crate::units::rounding(4) * psi[a].norm_sqr() / norm;
                }
                for (_, m) in &by_fast {
                    for (k, &s) in model.folded.iter().enumerate() {
                        let size: f64 = (0..n).map(|a| m.data[k * n + a].abs() * psi[a].abs()).sum::<f64>();
                        bound[s] += 2.0 * crate::units::rounding(2 * n + 6) * size * size / norm;
                    }
                }
                let total: f64 = expected.iter().sum();
                let spread: f64 = bound.iter().sum();
                let rows = run.rows(&crate::model::Point { r, kappa, t }, &psi);
                for s in 0..STATES {
                    let x = expected[s] / total;
                    let allowed = (bound[s] + x * spread) / total + 2.0 * crate::units::UNIT_ROUNDOFF * x;
                    assert!((rows[s] - x).abs() <= allowed, "{name} r {r:?} t {t} row {s}: {} against {x}, apart {:e} beyond {allowed:e}", rows[s], (rows[s] - x).abs());
                }
            }
        }
        eprintln!("{name}: the readout is the lift within its rounding");
    }
}

/// A jump's photon's node (module documentation, # The photons; `SelectedMiddle::photon`) is the whole node's photon
/// classes to the bit -- every element's value, gradient and photon form --, on the six scenes at the centre at rest and
/// at a moving point, for less work.
#[test]
fn a_jump_s_photon_node_is_the_whole_node_s_classes() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        for (r, kappa) in POINTS {
            let (whole, photon) = (run.node(r, kappa, true), run.photon_node(r, kappa));
            assert_eq!(whole.classes.len(), photon.classes.len(), "{name}");
            for (a, b) in whole.classes.iter().zip(&photon.classes) {
                for (x, y) in a.axes.iter().flatten().zip(b.axes.iter().flatten()) {
                    for (e, f) in x.elements.iter().zip(&y.elements) {
                        let bits = |z: C64| (z.re.to_bits(), z.im.to_bits());
                        assert!(bits(e.value) == bits(f.value) && bits(e.kick) == bits(f.kick) && e.gradient.iter().zip(&f.gradient).all(|(g, h)| bits(*g) == bits(*h)), "{name} r {r:?}: an element apart");
                    }
                }
            }
            eprintln!("{name} r {r:?}: the photon's node to the bit, its work {:.3e} against the whole node's {:.3e}", photon.work, whole.work);
            assert!(photon.work < whole.work, "{name}");
        }
    }
}

/// The direct entries spanned and K's sideband parts dropped (module documentation, # The direct entries spanned, # The
/// sideband parts dropped): on the six scenes at the centre at rest and at a moving point, the run's node against the
/// node with the entries one by one and every sideband part in (`Selected::whole`), entry by entry of every slot of the
/// generator and of its gradient: apart by no more than the two's booked Y (`Selected::span_directs`,
/// `Selected::drop_sidebands`) and the two computations' rounding of the entry; said per scene: the spans' outputs,
/// atoms and directions, whether parts were dropped, Y, the largest difference.
#[test]
fn the_direct_entries_spanned_are_the_entries_within_what_the_span_books() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let mut plain = run_of(&request, &keep);
        let spans = (run.lab.spans.len(), run.lab.spans.iter().map(|d| d.atoms.len()).sum::<usize>(), run.lab.spans.iter().map(|d| d.basis.len()).sum::<usize>());
        let parts = plain.whole();
        let booked = run.judgment.dropped - plain_dropped(&run);
        let (node, scales) = run.structure();
        let rounded = |k: usize, x: usize| crate::units::rounding(scales.generator[k].1[x] + 64) * scales.generator[k].0[x];
        let mut worst: f64 = 0.0;
        for (r, kappa) in POINTS {
            let (a, b) = (run.node(r, kappa, false), plain.node(r, kappa, false));
            for (k, ((_, x), (_, y))) in a.generator.iter().zip(&b.generator).enumerate() {
                for (i, (u, v)) in x.data.iter().zip(&y.data).enumerate() {
                    let d = (*u - *v).abs();
                    worst = worst.max(d);
                    assert!(d <= booked + rounded(k.min(scales.generator.len() - 1), i).max(1e-15 * u.abs()), "{name} slot {k} place {i}: {d:e} apart, booked {booked:e}");
                }
            }
        }
        let _ = node;
        eprintln!("{name}: spans (outputs, atoms, directions) {spans:?}, sideband parts dropped {parts}, booked Y {booked:.3e} rad/µs, the largest difference {worst:.3e}");
    }
}

/// How a model's change by its span's booked Y moves a run's trajectories (STEP, the coordinator's of 2026-11-23: a solve
/// spanned took 184 trajectories where the one by one took 240, TET): on PGC3D and TET (V3E_SCENE one of them),
/// V3E_ATOMS atoms (8) of `starts`, each with one noise on the run spanned and on it one by one (`Selected::unspanned`),
/// each on its own model throughout, steps of τ = V3E_TAU µs (0.474) to 1 ms: every 100 µs the largest |Δr| (µm) and
/// |Δκ| (1/µm) over the atoms, the atoms whose jumps differ, and the jumps. A measurement (ignored, by name).
#[test]
#[ignore]
fn the_span_s_trajectories_part() {
    let (atoms, tau) = (env("V3E_ATOMS", 8.0) as u64, env("V3E_TAU", 0.4741346189956611));
    for (name, request, keep) in scenes().into_iter().filter(|(n, ..)| ["PGC3D", "TET"].contains(n)) {
        let run = run_of(&request, &keep);
        let mut plain = run_of(&request, &keep);
        plain.unspanned();
        let from = starts(&run, request.temperature_uk, atoms);
        let mut pairs: Vec<[Trajectory; 2]> = from
            .iter()
            .enumerate()
            .map(|(i, (r, kappa, psi, threshold))| {
                [&run, &plain].map(|m| {
                    let mut t = Trajectory::new(m, m.cluster(0).states[0], *r, *kappa, 97, i as u64);
                    t.psi = psi.iter().copied().collect();
                    t.forms = m.forms(t.cluster, &t.point(), &t.psi);
                    t.threshold = *threshold;
                    t
                })
            })
            .collect();
        let mut noises: Vec<_> = (0..atoms).map(|i| crate::random::Stream::brownian(97, i, 0, 0)).collect();
        let mut scratch: Vec<[Scratch; 2]> = (0..atoms).map(|_| Default::default()).collect();
        let count = (1000.0 / tau).round() as usize;
        let every = (100.0 / tau).round() as usize;
        for k in 1..=count {
            for (i, pair) in pairs.iter_mut().enumerate() {
                let brownian = Brownian::over(tau, &[noises[i].normals(), noises[i].normals()]);
                for (j, m) in [&run, &plain].into_iter().enumerate() {
                    pair[j] = Midway::new(m, &pair[j], tau).and_then(|s| s.attempt(&pair[j], &brownian, &[], &mut scratch[i][j], &Anywhere)).expect("a step").end;
                }
            }
            if k % every == 0 || k == count {
                let dr = pairs.iter().map(|p| (0..3).map(|a| (p[0].r[a] - p[1].r[a]).abs()).fold(0.0, crate::linalg::larger)).fold(0.0, crate::linalg::larger);
                let dk = pairs.iter().map(|p| (0..3).map(|a| (p[0].kappa[a] - p[1].kappa[a]).abs()).fold(0.0, crate::linalg::larger)).fold(0.0, crate::linalg::larger);
                let apart = pairs.iter().filter(|p| p[0].jumps != p[1].jumps).count();
                let jumps: u32 = pairs.iter().map(|p| p[0].jumps).sum();
                eprintln!("SPANPART {name} t {:.1} µs: |Δr| {dr:.3e} µm, |Δκ| {dk:.3e} /µm, atoms whose jumps differ {apart} of {atoms}, jumps {jumps}", pairs[0][0].time);
            }
        }
    }
}

/// The elements between K's fast parts within a block against their gap (STEP item 2 of 2026-11-23, a measurement:
/// ignored, by name): on the six scenes' runs, the blocks cut by K's fast parts (F = 1 and F = 2 apart), at the centre
/// at rest and at a moving point, the node's static slot in its frame -- per pair of parts its elements' largest
/// Hermitian part |H_ab| and decay part |Γ_ab|/2 (the anti-Hermitian part −(i/2)Γ), the least gap |E_a − E_b| of the
/// slot's diagonal between them (the frame's θ apart: the elements' frequency in the frame's diagonal interaction
/// picture), the same with θ back on it, and the largest ratios; the other beats' cross elements' largest against their
/// frequency |ω_k + E_a − E_b|. Said "CROSSGAP".
#[test]
#[ignore]
fn the_cross_parts_against_their_gap() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let n = run.rs.keep.len();
        let parts = run.rs.plan().parts().to_vec();
        let (values, _) = run.components();
        let cut = |a: usize| (run.block_of[a], parts[a]);
        let mut pieces: Vec<(usize, [i64; 2])> = (0..n).map(cut).collect();
        pieces.sort();
        pieces.dedup();
        eprintln!("CROSSGAP {name}: P{n}, blocks {:?}, cut by fast parts {:?}", run.blocks().iter().map(Vec::len).collect::<Vec<_>>(), pieces.iter().map(|p| (0..n).filter(|&a| cut(a) == *p).count()).collect::<Vec<_>>());
        for (r, kappa) in [([0.0; 3], [0.0; 3]), POINTS[1]] {
            let node = run.node(r, kappa, false);
            let Some(stat) = node.generator.iter().find(|g| g.0 == 0).map(|g| &g.1) else { continue };
            let e = |a: usize| stat[(a, a)].re;
            let mut worst: std::collections::BTreeMap<((usize, [i64; 2]), (usize, [i64; 2])), [f64; 6]> = Default::default();
            for a in 0..n {
                for b in 0..n {
                    if run.block_of[a] != run.block_of[b] || parts[a] == parts[b] || cut(a) > cut(b) {
                        continue;
                    }
                    let h = (stat[(a, b)] + stat[(b, a)].conj()) * 0.5;
                    let g = (stat[(a, b)] - stat[(b, a)].conj()) * 0.5;
                    let gap = (e(a) - e(b)).abs();
                    let w = worst.entry((cut(a), cut(b))).or_insert([0.0, 0.0, f64::INFINITY, 0.0, 0.0, f64::INFINITY]);
                    w[5] = w[5].min((e(a) + run.theta[a] - e(b) - run.theta[b]).abs());
                    w[0] = w[0].max(h.abs());
                    w[1] = w[1].max(g.abs());
                    w[2] = w[2].min(gap);
                    w[3] = w[3].max(h.abs() / gap);
                    w[4] = w[4].max(g.abs() / gap);
                }
            }
            // Within each piece: its elements' largest off the diagonal and its diagonal's spread (the remainder's
            // third-order scale, ‖V‖²‖G_int‖/gap²).
            let mut inner = (0.0_f64, 0.0_f64);
            for p in &pieces {
                let members: Vec<usize> = (0..n).filter(|&a| cut(a) == *p).collect();
                for &a in &members {
                    for &b in &members {
                        if a != b {
                            inner.0 = inner.0.max(stat[(a, b)].abs());
                        }
                        inner.1 = inner.1.max((e(a) - e(b)).abs());
                    }
                }
            }
            eprintln!("CROSSGAP {name} r {r:?} κ {kappa:?}: within the pieces, the largest element off the diagonal {:.3e}, the diagonal's spread {:.3e} rad/µs", inner.0, inner.1);
            let mut other: f64 = 0.0;
            for (beat, m) in node.generator.iter().filter(|g| g.0 != 0) {
                for a in 0..n {
                    for b in 0..n {
                        if run.block_of[a] == run.block_of[b] && parts[a] != parts[b] {
                            other = other.max(m[(a, b)].abs() / (values[*beat] + e(a) - e(b)).abs());
                        }
                    }
                }
            }
            for ((p, q), w) in &worst {
                eprintln!("CROSSGAP {name} r {r:?} κ {kappa:?}: parts {p:?}–{q:?}: |H| {:.3e}, |Γ|/2 {:.3e}, gap {:.4e} rad/µs (θ back {:.4e}); |H|/gap {:.3e}, |Γ|/2/gap {:.3e}", w[0], w[1], w[2], w[5], w[3], w[4]);
            }
            eprintln!("CROSSGAP {name} r {r:?} κ {kappa:?}: the other beats' cross elements over their frequency, largest {other:.3e}");
        }
    }
}

/// The blocks cut by K's fast parts, the elements between them left out or turned (STEP item 2 of 2026-11-23, a
/// measurement, the profiling build's: ignored, by name): on PGC3D, TET, tilt0 and tilt2 (V3E_SCENE one of them), the
/// run whole, cut (the elements between the blocks left out) and turned (`Selected::turned`, second-order
/// Schrieffer–Wolff), V3E_ATOMS atoms (4) of `starts` held from any jump, steps of τ = V3E_TAU µs (0.163) with one noise,
/// each on its own model throughout: every 20 µs to V3E_UNTIL µs (100) the largest difference of a state's populations
/// |ψ_a|²/‖ψ‖² from the whole run's and of κ over the largest |κ|. Said "SPLITWALK".
#[cfg(feature = "prof")]
#[test]
#[ignore]
fn the_split_s_walk() {
    let (atoms, tau, until) = (env("V3E_ATOMS", 4.0) as u64, env("V3E_TAU", 0.163), env("V3E_UNTIL", 100.0));
    for (name, request, keep) in scenes().into_iter().filter(|(n, ..)| ["PGC3D", "TET", "tilt0", "tilt2"].contains(n)) {
        let whole = run_of(&request, &keep);
        let mut cut = run_of(&request, &keep);
        let pairs = cut.split_for_test(false);
        let mut turned = run_of(&request, &keep);
        turned.split_for_test(true);
        eprintln!("SPLITWALK {name}: pairs {pairs:?}, blocks whole {:?}, cut {:?}", whole.blocks().iter().map(Vec::len).collect::<Vec<_>>(), cut.blocks().iter().map(Vec::len).collect::<Vec<_>>());
        let from = starts(&whole, request.temperature_uk, atoms);
        let runs = [&whole, &cut, &turned];
        let mut walks: Vec<[Trajectory; 3]> = from
            .iter()
            .enumerate()
            .map(|(i, (r, kappa, psi, _))| {
                runs.map(|m| {
                    let mut t = Trajectory::new(m, m.cluster(0).states[0], *r, *kappa, 97, i as u64);
                    t.psi = psi.iter().copied().collect();
                    t.forms = m.forms(t.cluster, &t.point(), &t.psi);
                    t.threshold = f64::MIN_POSITIVE;
                    t
                })
            })
            .collect();
        let mut noises: Vec<_> = (0..atoms).map(|i| crate::random::Stream::brownian(97, i, 0, 0)).collect();
        let mut scratch: Vec<[Scratch; 3]> = (0..atoms).map(|_| Default::default()).collect();
        let count = (until / tau).round() as usize;
        let every = (20.0 / tau).round().max(1.0) as usize;
        let pops = |t: &Trajectory| -> Vec<f64> {
            let w: f64 = t.psi.iter().map(|z| z.norm_sqr()).sum();
            t.psi.iter().map(|z| z.norm_sqr() / w).collect()
        };
        for k in 1..=count {
            for (i, walk) in walks.iter_mut().enumerate() {
                let brownian = Brownian::over(tau, &[noises[i].normals(), noises[i].normals()]);
                for (j, &m) in runs.iter().enumerate() {
                    walk[j] = Midway::new(m, &walk[j], tau).and_then(|s| s.attempt(&walk[j], &brownian, &[], &mut scratch[i][j], &Anywhere)).expect("a step").end;
                }
            }
            if k % every == 0 || k == count {
                let ks = walks.iter().map(|w| crate::linalg::norm3(w[0].kappa)).fold(0.0, crate::linalg::larger);
                let apart = |j: usize| -> (f64, f64) {
                    let p = walks.iter().map(|w| pops(&w[0]).iter().zip(pops(&w[j])).map(|(a, b)| (a - b).abs()).fold(0.0, crate::linalg::larger)).fold(0.0, crate::linalg::larger);
                    let q = walks.iter().map(|w| (0..3).map(|a| (w[0].kappa[a] - w[j].kappa[a]).abs()).fold(0.0, crate::linalg::larger)).fold(0.0, crate::linalg::larger) / ks;
                    (p, q)
                };
                let ((pc, kc), (pt, kt)) = (apart(1), apart(2));
                eprintln!("SPLITWALK {name} t {:.1} µs: populations apart cut {pc:.3e}, turned {pt:.3e}; κ apart cut {kc:.3e}, turned {kt:.3e}", walks[0][0].time);
            }
        }
    }
}

/// The dropped beats' Y alone (the span's taken out): the pairs the judgment dropped.
fn plain_dropped(run: &Selected) -> f64 {
    run.judgment.pairs.iter().filter(|(slots, _)| slots.iter().all(|&k| run.judgment.tiers[k].1 == crate::model::classify::Tier::Drop)).map(|(_, y)| y).sum()
}

/// The readings on the middle (module documentation, # The readings on the middle): `SelectedMiddle::readout` at the
/// middle is the fold's readout at the point (`Selected::rows`, `READOUT`) to rounding; along the path, on a moving atom
/// at its momentum, it differs from the fold's readout at the path's point by the second order: from s to s/2 the
/// largest row's difference falls by 3–5 (four the second order's), on the six scenes, a spread state, both signs.
#[test]
fn the_readings_on_the_middle_are_the_points_to_the_second_order() {
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let psi = spread(run.rs.keep.len());
        let (r, kappa) = POINTS[1];
        let t = 0.37;
        let velocity = kappa.map(|k| run.hbar_over_mass() * k);
        let middle = run.middle(0, r, t);
        let path = middle.path(kappa, velocity);
        let apart = |s: f64| -> f64 {
            let on = middle.readout(&path, s, &psi).expect("a middle reads its rows");
            let at = run.rows(&crate::model::Point { r: std::array::from_fn(|i| r[i] + velocity[i] * s), kappa, t: t + s }, &psi);
            on.iter().zip(&at).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max)
        };
        let zero = apart(0.0);
        assert!(zero < 1e-14, "{name}: at the middle {zero:e} apart");
        for s in [0.2, -0.2] {
            let (whole, half) = (apart(s), apart(0.5 * s));
            let ratio = whole / half;
            eprintln!("{name}: s {s} apart {whole:.3e}, s/2 {half:.3e}, ratio {ratio:.2}");
            assert!((3.0..=5.0).contains(&ratio) || whole < 1e-13, "{name} s {s}: {whole:e} against {half:e}");
        }
    }
}

/// A reference scene's request (refs/scenes, as step/tests.rs reads them).
fn reference(name: &str) -> Request {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
    Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap()
}

/// The reference scenes the exact blocks are checked on beside the six: lin⊥lin with explicit excited states
/// (R4_LIN_1, R4_LIN_2), optical pumping on D1 with a D2 repump (R5_PUMP) and recoil heating by a red and a blue beam
/// (R9_HEAT) -- the last two the scenes whose levels' frames are settled to their energies (floquet/frame.rs # The
/// frame).
const BLOCKED: [&str; 4] = ["R4_LIN_1", "R4_LIN_2", "R5_PUMP", "R9_HEAT"];

/// The exact blocks' scenes (V3E_SCENE one of them): the six scenes' own requests (`scenes`) and `BLOCKED`.
fn blocked_scenes() -> Vec<(String, Request)> {
    let only = std::env::var("V3E_SCENE").ok();
    let mut cases: Vec<(String, Request)> = if only.as_deref().is_some_and(|o| BLOCKED.contains(&o)) {
        Vec::new()
    } else {
        scenes().into_iter().map(|(name, request, _)| (name.to_string(), request)).collect()
    };
    cases.extend(BLOCKED.into_iter().filter(|n| only.as_deref().is_none_or(|o| o == *n)).map(|n| (n.to_string(), reference(n))));
    cases
}

/// A request's run as the solve makes it (`made`'s), the beats `drops` says dropped.
fn made_with(request: &Request, drops: &Drops) -> Selected {
    let reference = crate::solve::alone(crate::solve::chief(request));
    let scene = crate::scene::Scene::new(request).unwrap();
    let selection = crate::select::choose(&scene).unwrap();
    Selected::made(scene, selection, reference, true, drops).unwrap()
}

/// A request's run on the ground states and the excited states `excited` (labels), the beats `drops` says dropped: a
/// model with explicit excited states, whatever the selection's chain would take (`made_with` is G's, the selection's
/// before its pilot).
fn made_on(request: &Request, excited: &[&str], drops: &Drops) -> Selected {
    let reference = crate::solve::alone(crate::solve::chief(request));
    let scene = crate::scene::Scene::new(request).unwrap();
    let selection = crate::select::choose(&scene).unwrap();
    let mut keep: Vec<usize> = (0..GROUND).collect();
    keep.extend(excited.iter().map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
    keep.sort_unstable();
    Selected::made(scene, crate::select::Selection { keep, ..selection }, reference, true, drops).unwrap()
}

/// tilt0's and tilt2's P19 of 2026-11-21 (the three-point choice's; JUDGE item 2): every D2 F′ = 3 state and D2 F′ = 2
/// but m = −2 -- the model the drop tier's checks were made on.
pub(crate) const TILT_P19: [&str; 11] = ["D2-3--3", "D2-3--2", "D2-3--1", "D2-3-0", "D2-3-1", "D2-3-2", "D2-3-3", "D2-2--1", "D2-2-0", "D2-2-1", "D2-2-2"];

/// A request's run as the solve makes it (`solve::run_of`, its reference state the chief's) in its exact blocks, and
/// (`split` false) the same model with its states one block.
fn made(request: &Request, split: bool) -> Selected {
    let reference = crate::solve::alone(crate::solve::chief(request));
    let scene = crate::scene::Scene::new(request).unwrap();
    let selection = crate::select::choose(&scene).unwrap();
    Selected::made(scene, selection, reference, split, &Drops::Judged).unwrap()
}

/// The explicit model a scene's blocks were accepted on (`the_exact_blocks_hold_at_every_point`), by its states' labels:
/// R4's P20 -- the ground states, D2-1--1, D2-2 but m = 0, D2-3 -- and tilt0's P19 -- the ground states, D2-2 but
/// m = −2, D2-3 -- (the three-point check's, printed in the gate's logs at 24ce4deb); none of another scene.
fn explicit(name: &str, request: &Request) -> Option<Vec<usize>> {
    let excited: &[&str] = match name {
        "R4" => &["D2-1--1", "D2-2--2", "D2-2--1", "D2-2-1", "D2-2-2", "D2-3--3", "D2-3--2", "D2-3--1", "D2-3-0", "D2-3-1", "D2-3-2", "D2-3-3"],
        "tilt0" => &["D2-2--1", "D2-2-0", "D2-2-1", "D2-2-2", "D2-3--3", "D2-3--2", "D2-3--1", "D2-3-0", "D2-3-1", "D2-3-2", "D2-3-3"],
        _ => return None,
    };
    let mut keep: Vec<usize> = (0..GROUND).collect();
    keep.extend(excited.iter().map(|l| request.states.iter().position(|s| s == l).unwrap_or_else(|| panic!("no state {l}"))));
    keep.sort_unstable();
    Some(keep)
}

/// A request's run on the states `keep` (`made`'s, P replaced), its states in their blocks or (`split` false) one.
fn made_split(request: &Request, keep: &[usize], split: bool) -> Selected {
    let reference = crate::solve::alone(crate::solve::chief(request));
    let scene = crate::scene::Scene::new(request).unwrap();
    let selection = crate::select::choose(&scene).unwrap();
    Selected::made(scene, Selection { keep: keep.to_vec(), ..selection }, reference, split, &Drops::Judged).unwrap()
}

/// A run's blocks said: each its ground and other states and its states' labels.
fn said(request: &Request, run: &Selected) -> String {
    run.blocks()
        .iter()
        .map(|b| {
            let ground = b.iter().filter(|&&a| run.rs.keep[a] < GROUND).count();
            format!("{}+{} [{}]", ground, b.len() - ground, b.iter().map(|&a| request.states[run.rs.keep[a]].as_str()).collect::<Vec<_>>().join(" "))
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// The structure's rounding scales are its entries' (model/selected.rs, # The structure's rounding): on the scenes'
/// runs (`blocked_scenes`), every entry of the structure's node -- each beat of the generator off the static beat's
/// diagonal (θ is put back there after), each beat of the gradient -- is within (1 + γ_K) of its scale S, the sum of
/// its terms' moduli, and a nonzero entry has terms; the entries other than zero that are the summation's rounding
/// zeros (|x| ≤ γ_K S, model/classify.rs `zero`) are said per scene with the largest of them over ε times its matrix's
/// largest.
#[test]
fn the_structure_s_scales_bound_its_entries() {
    for (name, request) in blocked_scenes() {
        let run = made(&request, true);
        let n = run.rs.keep.len();
        let (node, scales) = run.structure();
        let (mut zeros, mut largest) = (0usize, 0.0_f64);
        let mut check = |m: &Mat, (scale, terms): &(Vec<f64>, Vec<usize>), diagonal: bool, what: &str| {
            let top = m.max_abs();
            for x in (0..n * n).filter(|&x| !(diagonal && x / n == x % n)) {
                let z = m.data[x].abs();
                let bound = (1.0 + crate::units::rounding(terms[x])) * scale[x];
                assert!(z <= bound, "{name} {what} ({}, {}): |x| {z:e} above its scale {:e} (K {})", x / n, x % n, scale[x], terms[x]);
                assert!(z == 0.0 || terms[x] > 0, "{name} {what} ({}, {}): |x| {z:e} of no terms", x / n, x % n);
                if z != 0.0 && crate::model::classify::zero(m.data[x], scale[x], terms[x]) {
                    zeros += 1;
                    largest = crate::linalg::larger(largest, z / (f64::EPSILON * top));
                }
            }
        };
        for (k, ((beat, m), s)) in node.generator.iter().zip(&scales.generator).enumerate() {
            check(m, s, k == 0, &format!("beat {beat}"));
        }
        for ((beat, g), s) in node.gradient.iter().zip(&scales.gradient) {
            for (a, (m, s)) in g.iter().zip(s).enumerate() {
                check(m, s, false, &format!("gradient beat {beat} axis {a}"));
            }
        }
        eprintln!("{name}: {zeros} entries the summation's rounding alone reads as zeros, the largest {largest:.3e} ε·max");
    }
}

/// The exact blocks hold at every point (model/selected.rs, # Exact blocks): the scenes' runs (`blocked_scenes`), their
/// blocks read at the trap's centre at rest, against the same model with its states one block, at the centre at rest, at
/// a moving point off it (`POINTS`) and at a point farther out at another speed: the model's node with the photon forms
/// there, its every beat's H_eff and gradient, has between two blocks no element beyond the rounding of its matrix (said
/// in units of ε times the matrix's largest entry), and the blocked node is the whole one's with those places zero, bit
/// for bit. The blocks said per scene: R4 (R4_LIN_4) 9 + 11, tilt0 11 + 8 (the decision history's acceptance), each on
/// the explicit model the acceptance was made on, named by its states (`explicit`): the three-point check's P20 of R4
/// and P19 of tilt0, the selection's run models until 429c36cd -- since then the selection gives the solve the full
/// ground model G to run its pilot on (select.rs `choose`, the chain's first), one block or the ground states' own, and
/// a test of its blocks would read nothing of the excited states' (as solve.rs `a_state_s_price_is_its_step_s_count`).
#[test]
fn the_exact_blocks_hold_at_every_point() {
    let points = [POINTS[0], POINTS[1], ([1.3, -0.7, 2.1], [-9.0, 4.0, 6.0])];
    for (name, request) in blocked_scenes() {
        let (run, whole) = match explicit(&name, &request) {
            Some(keep) => (made_split(&request, &keep, true), made_split(&request, &keep, false)),
            None => (made(&request, true), made(&request, false)),
        };
        assert_eq!(run.rs.keep, whole.rs.keep, "{name}: one P");
        let n = run.rs.keep.len();
        eprintln!("{name}: P {n}, {} blocks: {}", run.blocks().len(), said(&request, &run));
        let block_of = &run.block_of;
        let mut worst = 0.0_f64;
        for (r, kappa) in points {
            let (a, b) = (run.node(r, kappa, true), whole.node(r, kappa, true));
            let matrices = |node: &Node| node.generator.iter().map(|(_, m)| m.clone()).chain(node.gradient.iter().flat_map(|(_, g)| g.iter().cloned())).collect::<Vec<Mat>>();
            let (ma, mb) = (matrices(&a), matrices(&b));
            assert_eq!(ma.len(), mb.len(), "{name}: one layout of beats");
            for (x, y) in ma.iter().zip(&mb) {
                let top = y.max_abs();
                for i in 0..n {
                    for j in 0..n {
                        if block_of[i] == block_of[j] {
                            assert!(x[(i, j)] == y[(i, j)], "{name} at {r:?}, {kappa:?}: ({i}, {j}) in a block, {:?} against the whole's {:?}", x[(i, j)], y[(i, j)]);
                        } else {
                            assert!(x[(i, j)] == C64::ZERO, "{name}: ({i}, {j}) between two blocks written");
                            if y[(i, j)] != C64::ZERO {
                                worst = crate::linalg::larger(worst, y[(i, j)].abs() / (f64::EPSILON * top));
                            }
                        }
                    }
                }
            }
        }
        eprintln!("{name}: the whole model's elements between two blocks at most {worst:.3e} ε of their matrix's largest");
        assert!(worst <= 1.0, "{name}: an element between two blocks of {worst} ε of its matrix's largest");
        if name == "R4" {
            assert_eq!(run.blocks().iter().map(Vec::len).collect::<Vec<_>>(), [9, 11], "R4's blocks");
        }
        if name == "tilt0" {
            assert_eq!(run.blocks().iter().map(Vec::len).collect::<Vec<_>>(), [11, 8], "tilt0's blocks");
        }
    }
}

/// Atoms through the same steps on a model (`through`'s): from the trap's centre region in a spread state (`spread`, a
/// phase pattern per atom) at the thermal speed, a random threshold each.
fn starts(run: &Selected, temperature_uk: f64, atoms: u64) -> Vec<(V3, V3, Vec<C64>, f64)> {
    let speed = (crate::units::KT_PER_UK * temperature_uk / run.hbar_over_mass()).sqrt();
    let n = run.cluster(0).states.len();
    (0..atoms)
        .map(|i| {
            let mut stream = crate::random::Stream::new(97, crate::random::Purpose::Initial, i, 0);
            let r: V3 = std::array::from_fn(|_| 0.2 * stream.normal());
            let kappa: V3 = std::array::from_fn(|_| speed * stream.normal());
            let psi: Vec<C64> = spread(n).iter().enumerate().map(|(a, z)| *z * C64::cis(0.3 * (i as f64) * a as f64)).collect();
            (r, kappa, psi, stream.open_uniform())
        })
        .collect()
}

/// The atoms of `starts` on a model, `count` steps of τ each with one noise: each atom's end and the steps' wall time
/// (s, one thread).
fn through(run: &Selected, starts: &[(V3, V3, Vec<C64>, f64)], tau: f64, count: usize) -> (Vec<Trajectory>, f64) {
    let clock = std::time::Instant::now();
    let ends = starts
        .iter()
        .enumerate()
        .map(|(i, (r, kappa, psi, threshold))| {
            let mut t = Trajectory::new(run, run.cluster(0).states[0], *r, *kappa, 97, i as u64);
            t.psi = psi.iter().copied().collect();
            t.forms = run.forms(t.cluster, &t.point(), &t.psi);
            t.threshold = *threshold;
            let (mut scratch, mut noise) = (Scratch::default(), crate::random::Stream::brownian(97, i as u64, 0, 0));
            for _ in 0..count {
                let brownian = Brownian::over(tau, &[noise.normals(), noise.normals()]);
                t = Midway::new(run, &t, tau).and_then(|m| m.attempt(&t, &brownian, &[], &mut scratch, &Anywhere)).expect("a step").end;
            }
            t
        })
        .collect();
    (ends, clock.elapsed().as_secs_f64())
}

/// How far two sets of ends are apart (`through`'s): the largest |Δψ_a| (the states normalized), |Δr| over the largest |r| and |Δκ|
/// over the largest |κ|, each in units of ε; and whether every atom's jumps are the same.
fn ends_apart(a: &[Trajectory], b: &[Trajectory]) -> ([f64; 3], bool) {
    let mut d = [0.0_f64; 3];
    let (rs, ks) = a.iter().fold((0.0_f64, 0.0_f64), |(r, k), t| (crate::linalg::larger(r, crate::linalg::norm3(t.r)), crate::linalg::larger(k, crate::linalg::norm3(t.kappa))));
    for (x, y) in a.iter().zip(b) {
        d[0] = crate::linalg::larger(d[0], x.psi.iter().zip(y.psi.iter()).map(|(p, q)| (*p - *q).abs()).fold(0.0, crate::linalg::larger));
        d[1] = crate::linalg::larger(d[1], (0..3).map(|i| (x.r[i] - y.r[i]).abs()).fold(0.0, crate::linalg::larger) / rs);
        d[2] = crate::linalg::larger(d[2], (0..3).map(|i| (x.kappa[i] - y.kappa[i]).abs()).fold(0.0, crate::linalg::larger) / ks);
    }
    (d.map(|x| x / f64::EPSILON), a.iter().zip(b).all(|(x, y)| x.jumps == y.jumps))
}

/// The atoms of `starts` through `count` steps of τ each with one noise on the run, each step taken from the same state
/// on `whole` (the same model one block) beside it, and the trajectories on `whole` throughout: per kind of step --
/// without a jump on either, with one -- the most two ends of one step are apart (`ends_apart`'s units) and the steps'
/// count, and the most the step on `whole` from the state turned by a phase e^{i}, its end turned back, is apart from
/// the one from the state as it is (the step's own rounding of ψ: the same numbers in another order); the most a step's
/// ψ apart is in units of τ‖Ĥ‖, ‖Ĥ‖ = ‖Ĥ − tr Ĥ/n‖ + |tr Ĥ/n| (Frobenius, the generator at the step's start) the most
/// an eigenvalue λ = λ₀ + tr Ĥ/n can be: its rounding of ε‖Ĥ‖ is a phase of ετ‖Ĥ‖, the floor of any decomposition's
/// propagation (a common diagonal far from zero its own share, a global phase); each model's nodes and wall time over the
/// steps from the same states (s, one thread); and both trajectories' ends.
struct Side {
    step: [([f64; 3], usize); 2],
    phase: [[f64; 3]; 2],
    floor: [f64; 2],
    nodes: [usize; 2],
    time: [f64; 2],
    ends: [Vec<Trajectory>; 2],
}

fn side_by_side(run: &Selected, whole: &Selected, starts: &[(V3, V3, Vec<C64>, f64)], tau: f64, count: usize) -> Side {
    let mut side = Side { step: [([0.0; 3], 0); 2], phase: [[0.0; 3]; 2], floor: [0.0; 2], nodes: [0; 2], time: [0.0; 2], ends: [Vec::new(), Vec::new()] };
    for (i, (r, kappa, psi, threshold)) in starts.iter().enumerate() {
        let start = |model: &Selected| {
            let mut t = Trajectory::new(model, model.cluster(0).states[0], *r, *kappa, 97, i as u64);
            t.psi = psi.iter().copied().collect();
            t.forms = model.forms(t.cluster, &t.point(), &t.psi);
            t.threshold = *threshold;
            t
        };
        let (mut t, mut u) = (start(run), start(whole));
        let mut scratch: [Scratch; 4] = Default::default();
        let mut noise = crate::random::Stream::brownian(97, i as u64, 0, 0);
        for _ in 0..count {
            let brownian = Brownian::over(tau, &[noise.normals(), noise.normals()]);
            let mut taken = Vec::new();
            for (k, model) in [run, whole].into_iter().enumerate() {
                let clock = std::time::Instant::now();
                let step = Midway::new(model, &t, tau).and_then(|m| m.attempt(&t, &brownian, &[], &mut scratch[k], &Anywhere)).expect("a step");
                side.time[k] += clock.elapsed().as_secs_f64();
                side.nodes[k] += step.prices.iter().map(|(_, p)| p.nodes).sum::<usize>();
                taken.push(step.end);
            }
            let (d, _) = ends_apart(&taken[..1], &taken[1..]);
            let kind = usize::from(taken.iter().any(|e| e.jumps != t.jumps));
            side.step[kind].0 = std::array::from_fn(|a| crate::linalg::larger(side.step[kind].0[a], d[a]));
            side.step[kind].1 += 1;
            let n = t.psi.len();
            let middle = run.middle(t.cluster, t.r, t.time);
            let path = middle.path(t.kappa, t.kappa.map(|k| run.hbar_over_mass() * k));
            let mut h = Mat::zeros(n, n);
            middle.generator(&path, 0.0, &mut h);
            let mean = (0..n).map(|x| h[(x, x)]).sum::<C64>() / n as f64;
            let size = (0..n * n).map(|i| (h.data[i] - if i / n == i % n { mean } else { C64::ZERO }).norm_sqr()).sum::<f64>().sqrt() + mean.abs();
            side.floor[kind] = crate::linalg::larger(side.floor[kind], d[0] / (tau * size));
            let mut turned = t.clone();
            turned.psi.iter_mut().for_each(|z| *z = *z * C64::cis(1.0));
            let mut back = Midway::new(whole, &turned, tau).and_then(|m| m.attempt(&turned, &brownian, &[], &mut scratch[3], &Anywhere)).expect("a step").end;
            back.psi.iter_mut().for_each(|z| *z = *z * C64::cis(-1.0));
            let (p, _) = ends_apart(&taken[1..], std::slice::from_ref(&back));
            side.phase[kind] = std::array::from_fn(|a| crate::linalg::larger(side.phase[kind][a], p[a]));
            u = Midway::new(whole, &u, tau).and_then(|m| m.attempt(&u, &brownian, &[], &mut scratch[2], &Anywhere)).expect("a step").end;
            t = taken.swap_remove(0);
        }
        side.ends[0].push(t);
        side.ends[1].push(u);
    }
    side
}

/// The blocks change no step beyond the rounding, and what they save (R1-CLEAN item 3's acceptance; a measurement too):
/// R4, tilt0, R5_PUMP and R9_HEAT (V3E_SCENE one of them), V3E_ATOMS atoms (4) from `starts`, V3E_STEPS steps (12) of
/// τ = V3E_TAU µs (0.05) with one noise, each step from the same state on the run and on the same model one block
/// (`side_by_side`): the two ends of a step without a jump apart by the rounding -- said in units of ε, beside the step's
/// own rounding of ψ (the phase-turned step's), and ψ at most n ετ‖Ĥ‖, the eigenvalues' rounding of an n-term sum each
/// --, those of a step with one said beside them (a jump's time is the root of the survival, which moves it by the
/// rounding over the hazard); the two trajectories' ends and jumps, each on its own model throughout; and a step's nodes
/// and wall time on each (one thread).
#[test]
#[ignore]
fn the_blocks_change_a_step_by_rounding() {
    let (atoms, count, tau) = (env("V3E_ATOMS", 4.0) as u64, env("V3E_STEPS", 12.0) as usize, env("V3E_TAU", 0.05));
    for (name, request) in blocked_scenes().into_iter().filter(|(n, _)| ["R4", "tilt0", "R5_PUMP", "R9_HEAT"].contains(&n.as_str())) {
        let (run, whole) = (made(&request, true), made(&request, false));
        let side = side_by_side(&run, &whole, &starts(&run, request.temperature_uk, atoms), tau, count);
        let (d, jumps) = ends_apart(&side.ends[0], &side.ends[1]);
        let steps = (atoms as usize * count) as f64;
        let [(free, nf), (jumped, nj)] = side.step;
        let [own, own_jumped] = side.phase;
        eprintln!(
            "{name}: blocks {:?}; {atoms} atoms × {count} steps of τ {tau} µs; a step's two ends apart without a jump ({nf}) |Δψ| {:.1} ε = {:.3} ετ‖Ĥ‖, |Δr| {:.1} ε, |Δκ| {:.1} ε (the step's own rounding {:.1}, {:.1}, {:.1} ε), with one ({nj}) {:.1}, {:.1}, {:.1} ε (its own {:.1}, {:.1}, {:.1} ε); the trajectories' ends ({} and {} jumps) {:.1}, {:.1}, {:.1} ε; a step {:.0} nodes and {:.3} ms in blocks, {:.0} nodes and {:.3} ms one block ({:.2}×)",
            run.blocks().iter().map(Vec::len).collect::<Vec<_>>(),
            free[0],
            side.floor[0],
            free[1],
            free[2],
            own[0],
            own[1],
            own[2],
            jumped[0],
            jumped[1],
            jumped[2],
            own_jumped[0],
            own_jumped[1],
            own_jumped[2],
            side.ends[0].iter().map(|t| t.jumps).sum::<u32>(),
            side.ends[1].iter().map(|t| t.jumps).sum::<u32>(),
            d[0],
            d[1],
            d[2],
            side.nodes[0] as f64 / steps,
            1e3 * side.time[0] / steps,
            side.nodes[1] as f64 / steps,
            1e3 * side.time[1] / steps,
            side.time[1] / side.time[0]
        );
        assert!(jumps, "{name}: the trajectories' jumps differ");
        let n = run.rs.keep.len() as f64;
        assert!(side.floor[0] <= n, "{name}: a step's two ends apart by {} ετ‖Ĥ‖ ({free:?} ε)", side.floor[0]);
    }
}

/// An element between two blocks of more than the rounding joins them, and the readings move with it, not at it
/// (R1-CLEAN item 3): R4_LIN_1's request on R4's P20 (`explicit`; the selection's chain takes G for it since P is chosen
/// on the pilot's paths, 429c36cd of 2026-11-22, and G's blocks hold no excited state) with its first beam's coupling
/// between a ground state of F = 2 in one block and
/// an explicit excited state of F′ = 3 in another -- the cooling transition, near resonance -- set to δ (H0 is
/// diagonal in the model's basis: a coupling is a beam's), δ of
/// 10⁻², 10², 10⁴ and 10⁸ times ε times the coupling's largest entry -- the first below the rounding (of the element's
/// beat in the node and of its first-order amplitude at the fold's input alike), the others above it: below the rounding
/// the blocks stand, above it they are one; and `starts`' atoms through V3E_STEPS steps (12), held from any jump (the
/// threshold the least number: a jump's time would move by the rounding over the hazard), apart from δ = 0's by the
/// rounding below it, by an amount that grows with δ above it -- no step at the joining.
#[test]
#[ignore]
fn a_cross_block_element_joins_the_blocks() {
    let (atoms, count, tau) = (env("V3E_ATOMS", 4.0) as u64, env("V3E_STEPS", 12.0) as usize, env("V3E_TAU", 0.05));
    let request = reference("R4_LIN_1");
    let p20 = explicit("R4", &request).expect("R4's P20");
    let run = made_split(&request, &p20, true);
    assert!(run.blocks().len() > 1, "R4_LIN_1 in blocks");
    let keep = &run.rs.keep;
    let blocks = run.blocks();
    let (g, e) = (0..blocks.len())
        .flat_map(|i| (0..blocks.len()).filter(move |&j| j != i).map(move |j| (i, j)))
        .flat_map(|(i, j)| blocks[i].iter().flat_map(move |&g| blocks[j].iter().map(move |&e| (g, e))))
        .find(|&(g, e)| request.states[keep[g]].starts_with("ground-2-") && request.states[keep[e]].starts_with("D2-3-"))
        .unwrap_or_else(|| panic!("a ground state of F = 2 in one block and an excited state of F′ = 3 in the other: blocks {}", said(&request, &run)));
    let top = request.beams[0].coupling.max_abs();
    eprintln!("R4_LIN_1: blocks {}; δ between {} and {} on {}; the coupling's largest {top:.4e}", said(&request, &run), request.states[keep[g]], request.states[keep[e]], request.beams[0].id);
    let from: Vec<_> = starts(&run, request.temperature_uk, atoms).into_iter().map(|(r, kappa, psi, _)| (r, kappa, psi, f64::MIN_POSITIVE)).collect();
    let (base, _) = through(&run, &from, tau, count);
    let mut moved = Vec::new();
    for factor in [1e-2, 1e2, 1e4, 1e8] {
        let delta = factor * f64::EPSILON * top;
        let mut perturbed = request.clone();
        perturbed.beams[0].coupling[(keep[e], keep[g])] += C64::real(delta);
        let other = made_split(&perturbed, &p20, true);
        assert_eq!(&other.rs.keep, keep, "one P");
        let (ends, _) = through(&other, &from, tau, count);
        let (d, jumps) = ends_apart(&ends, &base);
        eprintln!(
            "δ {factor:e} ε·max ({delta:.3e}): blocks {:?}; apart from δ = 0 |Δψ| {:.3e} ε, |Δr| {:.3e} ε, |Δκ| {:.3e} ε, the same jumps {jumps}",
            other.blocks().iter().map(Vec::len).collect::<Vec<_>>(),
            d[0],
            d[1],
            d[2]
        );
        assert_eq!(other.blocks().len() > 1, factor < 1.0, "δ {factor} ε·max: the blocks");
        moved.push(d[0]);
    }
    assert!(moved.windows(2).all(|w| w[1] >= w[0]), "the readings move with δ: {moved:?}");
}

#[path = "golden.rs"]
mod golden;

// STEP item 8: the Doppler series' census (ignored, by name).
#[path = "doppler.rs"]
mod doppler;

#[path = "drops.rs"]
mod drops;

#[path = "steps.rs"]
mod steps;

/// The selection's report (`Selected::report`): P state by state, the states folded, and the selection's notes -- the
/// structural chain, its candidates' sizes from G up (`Selection::notes`) -- on R10_NODE, G's run.
#[test]
fn the_report_lists_p_the_states_left_out_and_the_notes() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R10_NODE.json");
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let request = Request::from_native(&line).unwrap();
    let run = crate::solve::run_of(&request, crate::solve::alone(crate::solve::chief(&request))).unwrap();
    let report = run.report();
    eprint!("{report}");
    assert!(report.contains("P 的态：") && report.contains("折叠 "), "{report}");
    let notes = run.report.notes();
    assert!(!notes.is_empty() && notes.iter().all(|n| report.contains(n.as_str())), "{report}");
    assert!(report.contains("结构链") && run.rs.keep == run.report.chain[0], "{report}");
}


/// The prepared potential's envelope bounds it everywhere (`Potential::peak`, ensemble.rs `Well::envelope`): at points
/// across the trap and far out, −V_prep(r) ≤ B max_l A_l(r), A_l = 1/(1 + 2ρ²/w₀² + ζ²/z_R²) of the trap light's beams
/// (|a_l|² ≤ A_l of a paraxial beam) -- the bound the initial ensemble draws under. R1_TWEEZER and R1_SIGMA_P from their
/// initial state, the run as the solve makes it (`solve::run_of`).
#[test]
fn the_envelope_bounds_the_prepared_potential() {
    use crate::linalg::dot3;
    for name in ["R1_TWEEZER", "R1_SIGMA_P"] {
        let request = crate::model::testing::scene(name);
        let p = crate::solve::alone(crate::solve::chief(&request));
        let run = crate::solve::run_of(&request, p).unwrap();
        let potential = run.prepared(&p);
        let light = run.light();
        let beams: Vec<&crate::field::Geometry> = light.frequencies[light.trap].beams.iter().map(|&l| &light.beams[l]).collect();
        let bound = potential.peak();
        let mut x = 12345_u64;
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for i in 0..2000 {
            let scale = if i % 2 == 0 { 2.0 } else { 20.0 };
            let r = [next() * scale, next() * scale, next() * scale];
            let v = potential.at(r).value;
            let brightest = beams
                .iter()
                .map(|beam| {
                    let (w, z) = (beam.waist.expect("a tweezer"), beam.rayleigh().expect("a tweezer"));
                    let zeta = dot3(beam.direction, r);
                    1.0 / (1.0 + 2.0 * (dot3(r, r) - zeta * zeta) / (w * w) + zeta * zeta / (z * z))
                })
                .fold(0.0, crate::linalg::larger);
            assert!(-v <= bound * brightest * (1.0 + 1e-12) + 1e-12, "{name}: −V = {} above its bound {} at {r:?}", -v, bound * brightest);
        }
    }
}

/// R2-NODE (a measurement, run by name): a walk of R2_STEPS steps (24) at τ = R2_TAU (0.474 µs; halved where a step is
/// refused) from the reference state at a moving point, with the trajectory's own jumps, every step's end said to the
/// bit -- the time, r, κ, ψ, the jumps, the work: two builds' walks compared line by line.
#[test]
#[ignore]
fn r2_walk_dump() {
    let steps = env("R2_STEPS", 24.0) as usize;
    for (name, request, keep) in scenes() {
        let run = run_of(&request, &keep);
        let chief = crate::solve::chief(&request);
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt();
        let mut t = Trajectory::new(&run, chief, [0.1, 0.2, 0.3], [speed, -0.5 * speed, 0.3 * speed], 11, 0);
        let mut scratch = Scratch::default();
        let mut tau = env("R2_TAU", 0.474);
        let mut taken = 0;
        while taken < steps && tau > 1e-3 {
            let attempt = Midway::new(&run, &t, tau).and_then(|m| m.attempt(&t, &Brownian::STILL, &[], &mut scratch, &Anywhere));
            match attempt {
                Ok(step) => {
                    let jumps = step.made.iter().filter(|m| m.kick.is_some()).count();
                    t = step.end;
                    taken += 1;
                    let psi: Vec<String> = t.psi.iter().map(|z| format!("{:?},{:?}", z.re, z.im)).collect();
                    eprintln!("R2WALK {name} {taken} tau {tau:?} time {:?} r {:?} kappa {:?} jumps {jumps} work {:?} psi {}", t.time, t.r, t.kappa, step.work, psi.join(";"));
                }
                Err(e) => {
                    eprintln!("R2WALK {name} refused at tau {tau:?}: {e:?}");
                    tau *= 0.5;
                }
            }
        }
    }
}
