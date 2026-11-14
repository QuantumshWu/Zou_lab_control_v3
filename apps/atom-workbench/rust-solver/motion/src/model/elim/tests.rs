//! The elimination against analytic and exact references (ELIMINATION.md, the brief's analytic tests): each system small
//! enough that its exact static Floquet generator is a few states, diagonalized here.
use super::*;
use crate::linalg::eigen;

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// A level scheme: ground energies, excited energies (in the excited frame), one decay rate Γ per excited state through
/// the emission's first polarization from excited e to ground `lands[e]`, branching √Γ.
fn levels(ground: &[f64], excited: &[f64], gamma: f64, lands: &[usize]) -> Levels {
    let (g, e) = (ground.len(), excited.len());
    let mut emission: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(g, e));
    for (x, &to) in lands.iter().enumerate() {
        emission[0][(to, x)] = C64::real(gamma.sqrt());
    }
    let mut decay = Mat::zeros(e, e);
    for a in &emission {
        decay.add_scaled(C64::ONE, &a.adjoint_product(a));
    }
    Levels { ground: Mat::diagonal(&ground.iter().map(|&x| C64::real(x)).collect::<Vec<_>>()), excited: Mat::diagonal(&excited.iter().map(|&x| C64::real(x)).collect::<Vec<_>>()), decay }
}

/// A class of one beam at rest: coupling entries (excited, ground, value).
fn class(nu: f64, g: usize, e: usize, entries: &[(usize, usize, C64)]) -> Class {
    let mut coupling = Mat::zeros(e, g);
    for &(x, y, v) in entries {
        coupling[(x, y)] = v;
    }
    Class { nu, beams: vec![Beam { coupling, wave: [0.0; 3] }] }
}

fn folded(levels: &Levels, classes: &[Class], slow: &Slow) -> Fold {
    let replicas = Replicas::of(classes, slow, levels.ground.rows);
    fold(levels, classes, slow, &replicas)
}

fn at_rest(n: usize) -> Slow {
    Slow::single(vec![], vec![0.0; n])
}

/// The two-level atom's saturation (v3 3). The slow eigenvalue of the static Floquet generator [[0, g*], [g, −ν − iΓ/2]]
/// is the fold's self-consistent λ to round-off (Löwdin); the slow part of the bare ground state, x = (1 + ST)⁻¹·1, has
/// the exact spectral projection's norm N = |x|²(1 + |T|²). The mean scattering rate is the renewal rate after a jump,
/// 1/E[T] with E[T] = ∫‖e^{−iHt} g‖² dt, equal to the optical Bloch steady state Γ ρ_ee = (Γ/2) s/(1 + s),
/// s = 2|g|²/(ν² + Γ²/4) (checked here: the exact quantities and the steady state agree to round-off).
///
/// What the elimination gives of it: the slow mode alone, launched with the norm N and decaying at γ = −2 Im λ, its
/// waiting-time law P(T > t) = min(1, N e^{−γt}) (step.rs `jump`): E = N/γ (N ≤ 1) or (1 + ln N)/γ (N > 1). It is
/// not exactly the steady state's: what is missing is the homogeneous part of the bare state -- the fast mode, which
/// decays at Γ/2, and its cross term with the slow one: E[T] = N/γ + (fast and cross) exactly (checked), and the
/// clamp where N > 1. Measured below, the rate's relative error is within 9/8 s² (SPEC §1.4's bound of the old fourth
/// order, the same quantity) and is c s² with c → 1 on resonance, −1 far off it: second order in the saturation, the
/// term the error accounts book (ELIMINATION v3 3).
#[test]
fn a_two_level_atom_s_slow_mode_is_exact_and_its_scattering_rate_is_the_steady_state_s_to_second_order() {
    let gamma: f64 = 38.1;
    for &detuning in &[0.0_f64, 1.0, 6.0] {
        for &s in &[0.03_f64, 0.06, 0.12, 0.5, 2.0] {
            let nu = detuning * gamma;
            let g = (s * (nu * nu + 0.25 * gamma * gamma) / 2.0).sqrt();
            let lv = levels(&[0.0], &[0.0], gamma, &[0]);
            let classes = vec![class(nu, 1, 1, &[(0, 0, C64::real(g))])];
            let fold = folded(&lv, &classes, &at_rest(1));
            // The fold exists within Stewart's bound, the slow state's excited share X = |T|² below ¼ (`settle`; its
            // iteration factor is X exactly here): on resonance the slow and the fast mode meet at s = ½ (the
            // exceptional point g = Γ/4), where X = 1.
            let exact_share = {
                let a = (C64::real(g) / (c(nu, 0.5 * gamma) + if fold.exists { fold.generator[(0, 0)] } else { C64::ZERO })).norm_sqr();
                a
            };
            if !fold.exists {
                println!("Δ {detuning} Γ, s {s}: no fold");
                assert!(s >= 0.5, "the fold exists at s = {s}, Δ = {detuning} Γ");
                continue;
            }
            assert!(exact_share <= 0.25 * (1.0 + 1e-9), "a fold beyond Stewart's bound: X = {exact_share}");
            // The exact static Floquet generator and its slow mode.
            let h = Mat::from_fn(2, 2, |i, j| match (i, j) {
                (0, 0) => C64::ZERO,
                (1, 1) => c(-nu, -0.5 * gamma),
                _ => C64::real(g),
            });
            let e = eigen(&h).unwrap();
            let slow = if e.values[0].abs() < e.values[1].abs() { 0 } else { 1 };
            let lambda = fold.generator[(0, 0)];
            assert!((lambda - e.values[slow]).abs() <= 1e-12 * e.values[slow].abs().max(1e-300), "λ {lambda:?} against the slow eigenvalue {:?}", e.values[slow]);
            // The exact projection of the bare ground state on the slow mode and its norm.
            let along = |k: usize| e.inverse[(k, 0)];
            let vector = |k: usize| [e.vectors[(0, k)], e.vectors[(1, k)]];
            let norm = |k: usize, z: C64| (vector(k)[0].norm_sqr() + vector(k)[1].norm_sqr()) * z.norm_sqr();
            let left = fold.left_operators().unwrap();
            let (x, tau) = fold.transients(&left, &[vec![C64::ONE]]).unwrap().remove(0);
            let metric = fold.metric()[(0, 0)].re;
            let model_norm = x[0].norm_sqr() * (1.0 + metric);
            assert!((model_norm - norm(slow, along(slow))).abs() <= 1e-12, "N {model_norm} against {}", norm(slow, along(slow)));
            // The exact mean waiting time from the bare ground state, and the steady state's rate.
            let mut waiting = C64::ZERO;
            for j in 0..2 {
                for k in 0..2 {
                    let (vj, vk) = (vector(j), vector(k));
                    let overlap = vk[0].conj() * vj[0] + vk[1].conj() * vj[1];
                    waiting += along(j) * along(k).conj() * overlap / (C64::new(0.0, 1.0) * (e.values[j] - e.values[k].conj()));
                }
            }
            let steady = 0.5 * gamma * s / (1.0 + s);
            assert!((1.0 / waiting.re - steady).abs() <= 1e-10 * steady, "renewal {} against the steady state {steady}", 1.0 / waiting.re);
            let rate = -2.0 * lambda.im;
            let slow_part = model_norm / rate;
            let transient = waiting.re - slow_part;
            let modelled = if model_norm <= 1.0 { model_norm / rate } else { (1.0 + model_norm.ln()) / rate };
            let error = waiting.re / modelled - 1.0;
            println!("Δ {detuning} Γ, s {s}: λ {:.6e}{:+.6e}i, N {:.6}, rate's relative error {:+.4e} = {:+.3} s², the transient's share of E[T] {:+.3e}", lambda.re, lambda.im, model_norm, error, error / (s * s), transient / waiting.re);
            if s <= 0.12 {
                assert!(error.abs() <= 9.0 / 8.0 * s * s, "the rate's error {error} beyond 9/8 s² at s = {s}, Δ = {detuning} Γ");
            }
            // The transient's closed form (`Fold::transients`) against the exact one, and the error the accounts book
            // (`errors`: at rest the transient's alone, all of it on the one block) against the exact rate's error: to
            // the leading order in the fold, its relative O(X), X = s/2.
            let booked: f64 = errors(&fold, &[vec![0]], &|_| false, &[C64::ONE], [0.0; 3], 0.0, &[]).parts.iter().map(|p| p.1).sum();
            println!("    the transient {transient:+.5e} µs, closed form {tau:+.5e} µs; the rate's error {error:+.4e}, booked {booked:.4e}");
            assert!((tau - transient).abs() <= 2.0 * s * transient.abs(), "the closed form {tau} against {transient}");
            assert!((booked - error.abs()).abs() <= 2.0 * s * error.abs(), "booked {booked} against the rate's error {error}");
        }
    }
}

/// A Λ system (the brief's test 2): g1 (0) and g2 (δ₂) to one excited state e far off resonance, driven by two classes
/// ν1 (from g1) and ν2 (from g2). In the frames θ_g2 = ν1 − ν2 the two drives feed one replica: the static three-level
/// problem, whose two slow eigenvalues the fold reproduces to round-off; at two-photon resonance one of them is the dark
/// state, exactly zero (no light shift, no decay). In the frames at rest the two are two replicas, and their Raman
/// coupling is a P-layer component of size |g1 g2|/|Δ| (to the order (g/Δ)²) at the beat ν1 − ν2.
#[test]
fn a_lambda_system_s_raman_coupling_and_dark_state_are_the_analytic_ones() {
    let (gamma, delta) = (38.1, -2048.0);
    let (g1, g2) = (30.0, 40.0);
    for &two_photon in &[0.0, 0.7] {
        let (nu1, nu2) = (delta, delta - 26.5);
        // g2 at δ₂: two-photon resonance where ν1 − ν2 + 0 = δ₂ + offset.
        let d2 = (nu1 - nu2) + two_photon;
        let lv = levels(&[0.0, d2], &[0.0], gamma, &[0]);
        let classes = vec![class(nu1, 2, 1, &[(0, 0, C64::real(g1))]), class(nu2, 2, 1, &[(0, 1, C64::real(g2))])];
        let slow = Slow::single(vec![], vec![0.0, nu1 - nu2]);
        let fold = folded(&lv, &classes, &slow);
        assert!(fold.exists);
        assert_eq!(fold.replicas.len(), 1, "the two drives feed one replica in these frames");
        // The exact static generator in the frames: g1 at 0, g2 at δ₂ − (ν1 − ν2), e at −ν1 − iΓ/2.
        let h = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (1, 1) => C64::real(d2 - (nu1 - nu2)),
            (2, 2) => c(-nu1, -0.5 * gamma),
            (0, 2) | (2, 0) => C64::real(g1),
            (1, 2) | (2, 1) => C64::real(g2),
            _ => C64::ZERO,
        });
        let exact = eigen(&h).unwrap();
        let mut slow_exact: Vec<C64> = exact.values.iter().copied().filter(|z| z.abs() < 0.5 * delta.abs()).collect();
        let mut mine = fold.eigen.values.clone();
        slow_exact.sort_by(|a, b| a.re.total_cmp(&b.re));
        mine.sort_by(|a, b| a.re.total_cmp(&b.re));
        for (a, b) in mine.iter().zip(&slow_exact) {
            assert!((*a - *b).abs() <= 1e-12 * b.abs().max(1.0), "slow eigenvalue {a:?} against {b:?}");
        }
        if two_photon == 0.0 {
            let dark = mine.iter().fold(f64::INFINITY, |m, z| m.min(z.abs()));
            assert!(dark <= 1e-12 * (g1 * g1 / delta.abs()), "the dark state's eigenvalue {dark}");
        }
        // At rest: two replicas, the Raman component.
        let rest = folded(&lv, &classes, &at_rest(2));
        assert_eq!(rest.replicas.len(), 2);
        let raman = rest.components([0.0; 3]).into_iter().filter(|x| rest.dominant(x.row) != rest.dominant(x.column)).fold(0.0, |m: f64, x| m.max(x.size));
        let analytic = g1 * g2 / delta.abs();
        assert!((raman / analytic - 1.0).abs() <= 4.0 * (g2 / delta).powi(2), "Raman {raman} against g1 g2/|Δ| {analytic}");
    }
}

/// A far-detuned light's shifts (the brief's test 3): three ground states with a small Zeeman spread, five excited
/// states, a coupling with every entry, the light 2·10⁶ rad/µs off: the fold is the second-order perturbation theory's
/// V†(ν − H_E)⁻¹V -- its scalar, vector and tensor parts, the whole matrix -- to the relative order of the levels'
/// spread and shifts over the detuning.
#[test]
fn a_far_detuned_light_s_scalar_and_tensor_shifts_are_second_order_perturbation_theory_s() {
    let (g, e) = (3, 5);
    let lv = levels(&[-1.0, 0.0, 1.0], &[0.0, 3.0, -2.0, 7.0, 1.0], 38.1, &[0, 1, 2, 0, 1]);
    let nu = 2.0e6;
    let mut coupling = Mat::zeros(e, g);
    for x in 0..e {
        for y in 0..g {
            coupling[(x, y)] = c(((x * 7 + y * 3) % 5) as f64 * 40.0 + 30.0, ((x + 2 * y) % 3) as f64 * 25.0 - 20.0);
        }
    }
    let classes = vec![Class { nu, beams: vec![Beam { coupling: coupling.clone(), wave: [0.0; 3] }] }];
    let fold = folded(&lv, &classes, &at_rest(g));
    let mut pt = Mat::zeros(g, g);
    for a in 0..g {
        for b in 0..g {
            pt[(a, b)] = (0..e).map(|x| coupling[(x, a)].conj() * coupling[(x, b)] / c(lv.ground[(b, b)].re + nu - lv.excited[(x, x)].re, 0.5 * lv.decay[(x, x)].re)).sum();
        }
    }
    let mut shift = fold.generator.clone();
    shift.add_scaled(-C64::ONE, &lv.ground);
    let scale = pt.max_abs();
    let order = (shift.max_abs() + 10.0) / nu;
    for a in 0..g {
        for b in 0..g {
            assert!((shift[(a, b)] - pt[(a, b)]).abs() <= 4.0 * order * scale, "({a}, {b}): {:?} against {:?}", shift[(a, b)], pt[(a, b)]);
        }
    }
    assert!(pt[(0, 1)].abs() > 1e-3 * scale, "the tensor part is there to check");
}

/// A component averaged (the brief's test 4): two ground states a (0) and b (δ₀) coupled by a Raman component of two
/// classes at the beat Ω through a far excited state. The exact quasienergies: in the frames θ_b = ν1 − ν2 the problem
/// is static and the fold is exact. The averaged model keeps the frames at rest: its components' reading (a, δ) gives
/// the static shift −|a|²/δ of a (δ = Ω + λ_b − λ_a) to the relative order (a/δ)², and the exact slow state's admixture
/// of b is (a/δ)² to the same order -- the population the averaging leaves out.
#[test]
fn an_averaged_component_s_shift_and_micromotion_are_the_exact_floquet_ones() {
    let (gamma, delta) = (38.1, -3072.0);
    for &(g1, g2, offset) in &[(60.0, 50.0, 8.0), (60.0, 50.0, 3.0), (90.0, 80.0, 6.0)] {
        let (nu1, nu2) = (delta, delta - 26.5);
        let d0 = (nu1 - nu2) + offset;
        let lv = levels(&[0.0, d0], &[0.0], gamma, &[0]);
        let classes = vec![class(nu1, 2, 1, &[(0, 0, C64::real(g1))]), class(nu2, 2, 1, &[(0, 1, C64::real(g2))])];
        let rest = folded(&lv, &classes, &at_rest(2));
        let exact = folded(&lv, &classes, &Slow::single(vec![], vec![0.0, nu1 - nu2]));
        let pa = (0..2).find(|&p| rest.dominant(p) == 0).unwrap();
        let qa = (0..2).find(|&p| exact.dominant(p) == 0).unwrap();
        let component = rest.components([0.0; 3]).into_iter().filter(|x| x.column == (1 - pa) && x.row == pa).fold(None::<Component>, |m, x| if m.map_or(true, |m| x.size > m.size) { Some(x) } else { m }).unwrap();
        let mixing = component.mixing();
        let predicted = -component.size * component.size / component.distance.re;
        let shift = exact.eigen.values[qa].re - rest.eigen.values[pa].re;
        assert!((shift / predicted - 1.0).abs() <= 3.0 * mixing * mixing, "shift {shift} against −|a|²/δ {predicted} (mixing {mixing})");
        let admixture = exact.eigen.vectors[(1, qa)].norm_sqr() / exact.eigen.vectors[(0, qa)].norm_sqr();
        assert!((admixture / (mixing * mixing) - 1.0).abs() <= 3.0 * mixing * mixing, "admixture {admixture} against (a/δ)² {}", mixing * mixing);
    }
}

/// A Raman resonance across a Zeeman splitting (the coordinator's test): two ground states 2ω_L apart, coupled by two
/// classes 2ω_L + δ apart through a far excited state. The lab beat is 26.37 rad/µs; the component's distance on the
/// actual spectrum is δ (and the light shifts' difference); in the frames (cluster.rs `frames`: a chain, every link on
/// the strength tree) it is static, its residual exactly zero -- and the fold in those frames is the exact static
/// problem's (its slow eigenvalues against the exact three-state generator's).
#[test]
fn a_raman_resonance_across_a_zeeman_splitting_asks_the_step_for_its_detuning_not_its_beat() {
    let (gamma, delta, zeeman, detuning) = (38.1, -1536.0, 26.375, 0.375);
    let (nu1, nu2) = (delta, delta - zeeman - detuning);
    let lv = levels(&[0.0, zeeman], &[0.0], gamma, &[0]);
    let classes = vec![class(nu1, 2, 1, &[(0, 0, C64::real(40.0))]), class(nu2, 2, 1, &[(0, 1, C64::real(40.0))])];
    let rest = folded(&lv, &classes, &at_rest(2));
    let share = crate::units::POPULATION_TARGET * (1.0 - crate::units::STEP_SHARE * crate::units::STEP_SHARE).sqrt();
    let explicit: Vec<Component> = rest.components([0.0; 3]).into_iter().filter(|x| x.row != x.column && x.mixing() > share.sqrt()).collect();
    assert!(!explicit.is_empty(), "the resonant Raman component is explicit");
    let beat = |x: &Component| Link { a: x.row, b: x.column, classes: x.classes }.beat(&classes);
    for x in &explicit {
        assert!((beat(x).abs() - (zeeman + detuning)).abs() <= 1e-9, "its lab beat");
        assert!(x.distance.abs() < 0.2 * zeeman, "its distance on the actual spectrum {:?}", x.distance);
    }
    let links: Vec<(usize, usize, f64, f64)> = explicit.iter().map(|x| (rest.dominant(x.row), rest.dominant(x.column), beat(x), x.size)).collect();
    let theta = crate::model::cluster::frames(2, &links, &0.0);
    assert!(links.iter().all(|l| l.2 - (theta[l.0] - theta[l.1]) == 0.0), "a chain: static in its frames");
    let fold = folded(&lv, &classes, &Slow::single(vec![], theta.clone()));
    assert_eq!(fold.replicas.len(), 1);
    let h = Mat::from_fn(3, 3, |i, j| match (i, j) {
        (0, 0) => C64::real(-theta[0]),
        (1, 1) => C64::real(zeeman - theta[1]),
        (2, 2) => c(-nu1 - theta[0], -0.5 * gamma),
        (0, 2) | (2, 0) | (1, 2) | (2, 1) => C64::real(40.0),
        _ => C64::ZERO,
    });
    let exact = eigen(&h).unwrap();
    for z in &fold.eigen.values {
        let nearest = exact.values.iter().fold(f64::INFINITY, |m, w| m.min((*w - *z).abs()));
        assert!(nearest <= 1e-10 * z.abs().max(1.0), "slow eigenvalue {z:?}");
    }
}

fn request(name: &str) -> crate::request::Request {
    let path = format!("{}/refs/requests/{name}.json", env!("CARGO_MANIFEST_DIR"));
    let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).expect("the request file")).expect("JSON");
    crate::request::Request::from_native(&line).expect("a request")
}

/// The prior's 88 atoms of a request (solve.rs `atoms`: the reach's own ensemble at the request's temperature, the
/// pilot's initial atoms), on its model with nothing kept.
fn prior_atoms(req: &crate::request::Request) -> Vec<([f64; 3], [f64; 3])> {
    let light = crate::field::Light::new(req);
    let model = super::super::Model::new(req, &light, &[]).expect("the model");
    crate::solve::atoms(&model, req, &light, &crate::solve::components(req), req.temperature_uk).expect("the prior's atoms")
}

/// A closure's measures at its end (`closure`, its last round): the worst remainder ρ over its fold's ground-like
/// eigenstates at the offsets each spans (`Fold::offsets`: the motion's pattern and the components' combined
/// frequencies at their shares), and the largest mixing of a component criterion two leaves averaged.
fn measures(closed: &Closed, classes: &[Class], velocity: [f64; 3]) -> (f64, f64) {
    let fastest = classes.iter().flat_map(|c| c.beams.iter()).map(|b| crate::linalg::norm3(b.wave)).fold(0.0, f64::max);
    let pattern = 2.0 * fastest * crate::linalg::norm3(velocity);
    let fold = &closed.fold;
    let components = fold.components(velocity);
    let remainder = (0..fold.size())
        .filter(|&p| fold.ground_weight(p) >= 0.5)
        .map(|p| fold.block_parts(&[], p, Some(&fold.offsets(p, pattern, &components, 1.0, &|_| 1.0))).0)
        .fold(0.0, f64::max);
    let mixing = components.iter().map(|x| x.mixing()).filter(|m| 2.0 * m * m <= share()).fold(0.0, f64::max);
    (remainder, mixing)
}

/// The tilt's property (the brief's test 5, on the prior's own atoms): the 1D PGC at 3 G with its π probe tilted by
/// θ = 0, 0.1°, 2° (the user's failure #4 is 2°; refs/requests/PGC1D_TILT_*, made by the app's own request builder),
/// each closure at the 88 atoms the prior reads (`prior_atoms`, the θ = 0 request's). The tilt adds couplings ∝ sin θ:
/// - the partition (criterion one's blocks) is the same at every atom and θ;
/// - at 0.1° the kept links are the untilted ones and ρ moves by O(θ): no coupling counts as "existing" or not;
/// - at 2° the untilted links stay, and what is added are the tilt's components that are near resonance on the actual
///   spectrum at that atom (the Raman-dressed F = 2 levels: |δ| ≈ 0.4–1.2 rad/µs at a ≈ 0.08–0.3, 2m² up to 0.29 --
///   physics the rule keeps, measured 2026-11-14); at the five hand-picked points of before, none.
#[test]
fn a_tilt_leaves_the_partition_and_moves_the_links_only_where_its_couplings_are_resonant() {
    let names = ["PGC1D_TILT_0", "PGC1D_TILT_01", "PGC1D_TILT_2"];
    let requests: Vec<crate::request::Request> = names.iter().map(|n| request(n)).collect();
    let points = prior_atoms(&requests[0]);
    assert_eq!(points.len(), 88);
    let theta = 0.1_f64.to_radians();
    let (mut added, mut most) = (0, 0.0_f64);
    for &(r, v) in &points {
        let (closed, measured): (Vec<Closed>, Vec<(f64, f64)>) = requests
            .iter()
            .map(|req| {
                let light = crate::field::Light::new(req);
                let (lv, cl, _) = system(req, &light, r);
                let blocks = excited_blocks(&req.states);
                let closed = closure(&lv, &cl, &blocks, v, &Kept::default());
                let measured = measures(&closed, &cl, v);
                (closed, measured)
            })
            .unzip();
        for ((name, c), m) in names.iter().zip(&closed).zip(&measured) {
            println!("{name} r {r:?} v {v:?}: blocks {:?}, links {}, rho {:.4e}, averaged mixing {:.4e}", c.kept.blocks, c.kept.links.len(), m.0, m.1);
        }
        let base = &closed[0];
        for c in &closed[1..] {
            assert_eq!(c.kept.blocks, base.kept.blocks, "the partition at {r:?} {v:?}");
            for l in &base.kept.links {
                assert!(c.kept.links.contains(l), "the untilted link {l:?} at {r:?} {v:?}");
            }
        }
        assert_eq!(closed[1].kept.links.len(), base.kept.links.len(), "the links at 0.1° at {r:?} {v:?}");
        let (at0, at01) = (measured[0], measured[1]);
        assert!((at01.0 - at0.0).abs() <= 10.0 * theta * at0.0.max(1e-6), "ρ at 0.1°: {} against {}", at01.0, at0.0);
        added += closed[2].kept.links.len() - base.kept.links.len();
        most = most.max(measured[2].1);
    }
    println!("2°: {added} links added over the 88 atoms; the largest averaged mixing {most:.3e}");
    assert!(2.0 * most * most <= super::share(), "an averaged component beyond the share");
}

/// The frames are a gauge (module documentation, # Groups): at the prior's atoms of the tilted 1D PGC (a few of them),
/// on the closure's kept set, a constant added to the frames of every state of each group -- random, of the size of the
/// Zeeman splittings and beyond -- leaves every decision and every error the same: criterion two's components (labels,
/// classes, mixing), criterion one's remainders, and the error at a state with its parts.
#[test]
fn every_decision_and_error_is_the_same_in_every_gauge() {
    let req = request("PGC1D_TILT_2");
    let light = crate::field::Light::new(&req);
    let blocks = excited_blocks(&req.states);
    let ground = crate::request::GROUND;
    let points = prior_atoms(&req);
    let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for &(r, v) in points.iter().step_by(11) {
        let (lv, cl, _) = system(&req, &light, r);
        let closed = closure(&lv, &cl, &blocks, v, &Kept::default());
        let slow = slow_of(&blocks, &closed.kept, &cl, &lv);
        let fastest = cl.iter().flat_map(|c| c.beams.iter()).map(|b| crate::linalg::norm3(b.wave)).fold(0.0, f64::max);
        let pattern = 2.0 * fastest * crate::linalg::norm3(v);
        let read = |slow: &Slow| {
            let fold = folded(&lv, &cl, slow);
            let label = |p: usize| fold.dominant(p);
            let mut components: Vec<(usize, usize, (Option<usize>, Option<usize>), f64)> = fold.components(v).iter().filter(|x| x.mixing() > 1e-6).map(|x| (label(x.row), label(x.column), x.classes, x.mixing())).collect();
            components.sort_by(|a, b| (a.0, a.1, a.2).cmp(&(b.0, b.1, b.2)).then(a.3.total_cmp(&b.3)));
            let all = fold.components(v);
            let mut rho: Vec<f64> = (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5).map(|p| fold.block_parts(&blocks, p, Some(&fold.offsets(p, pattern, &all, 1.0, &|_| 1.0))).0).collect();
            rho.sort_by(f64::total_cmp);
            let psi: Vec<C64> = (0..fold.size()).map(|s| if s < ground { C64::real(1.0) } else { C64::ZERO }).collect();
            let errors = errors(&fold, &blocks, &|_| false, &psi, v, pattern, &[]);
            let error = errors.parts.iter().map(|p| p.1).sum::<f64>() + errors.outside.iter().sum::<f64>();
            (components, rho, error, errors.parts)
        };
        let base = read(&slow);
        for _ in 0..3 {
            let mut shifted = slow.clone();
            let shifts: Vec<f64> = (0..shifted.frames.len()).map(|_| 200.0 * (uniform() - 0.5)).collect();
            for (f, g) in shifted.frames.iter_mut().zip(&slow.groups) {
                *f += shifts[*g];
            }
            let other = read(&shifted);
            assert_eq!(base.0.len(), other.0.len(), "criterion two's components at {r:?}");
            for (a, b) in base.0.iter().zip(&other.0) {
                assert_eq!((a.0, a.1, a.2), (b.0, b.1, b.2), "a component's labels and classes");
                assert!((a.3 - b.3).abs() <= 1e-8 * a.3.max(1e-3), "a component's mixing {} against {}", a.3, b.3);
            }
            for (a, b) in base.1.iter().zip(&other.1) {
                assert!((a - b).abs() <= 1e-8 * a.max(1e-6), "a remainder {a} against {b}");
            }
            assert!((base.2 - other.2).abs() <= 1e-8 * base.2.max(1e-6), "the error {} against {}", base.2, other.2);
            for (object, x) in &base.3 {
                let y = other.3.iter().find(|o| o.0 == *object).map_or(0.0, |o| o.1);
                assert!((x - y).abs() <= 1e-8 * x.max(1e-6), "the part of {object:?}: {x} against {y}");
            }
        }
    }
}

/// A block of two classes (the coordinator's test): one ground state driven into one excited state by two classes of
/// different frequencies. The block's part of the error at the ground state goes to the classes by their responses'
/// shares (`class_shares`); with the two couplings swapped the larger share swaps with them. Kept for the class of the
/// larger part, the block is explicit at that class's replica: its drive static in the frames (residual zero, A's), the
/// other class's coupling folded at its own replica, where the explicit state is Q.
#[test]
fn a_block_is_kept_for_the_class_of_its_larger_part_and_turns_in_that_class_s_frame() {
    let (gamma, ground) = (38.1, 1);
    let lv = levels(&[0.0], &[0.0], gamma, &[0]);
    let blocks = vec![vec![0]];
    for strong in [0, 1] {
        let nus = [-30.0, -400.0];
        let size = |k: usize| if k == strong { 60.0 } else { 20.0 };
        let classes: Vec<Class> = (0..2).map(|k| class(nus[k], 1, 1, &[(0, 0, C64::real(size(k)))])).collect();
        let fold = folded(&lv, &classes, &at_rest(1));
        let psi = vec![C64::ONE];
        let found = errors(&fold, &blocks, &|_| false, &psi, [0.0; 3], 5.0, &[]);
        let part = |k: usize| found.parts.iter().find(|o| o.0 == Object::Block(0, Some(k))).map_or(0.0, |o| o.1);
        // The responses' shares: |g|²/|ν − iΓ/2|² of each class.
        let response = |k: usize| size(k).powi(2) / (nus[k].powi(2) + 0.25 * gamma * gamma);
        let (larger, smaller) = if response(0) > response(1) { (0, 1) } else { (1, 0) };
        assert!(part(larger) > part(smaller), "the class of the larger response holds the larger part: {} against {}", part(larger), part(smaller));
        // Kept for that class: the drive's link, the frames from it.
        let kept = Kept { blocks: vec![0], links: vec![Link { a: ground, b: 0, classes: (None, Some(larger)) }], ..Kept::default() };
        let slow = slow_of(&blocks, &kept, &classes, &lv);
        assert!(same(classes[larger].nu + slow.frames[0], slow.frames[ground]), "the explicit state turns in its class's frame");
        let fold = folded(&lv, &classes, &slow);
        assert!(fold.direct[(1, 0)].abs() > 0.0, "the kept class's drive is the static generator's");
        let other = fold.replicas.iter().find(|rep| rep.feeds.iter().any(|f| f.class == Some(smaller))).expect("the other class's replica");
        assert!(other.q.contains(&0), "the explicit state is Q at the other class's replica");
    }
}

/// The fold's series about a reference (the evaluation route the engine takes; the self-consistent fold is the definition),
/// on PGC3D's F = 2 states at the point near the focus where its cooling light's coherent field is strongest: the reference
/// z₀ the mean of the states' levels without the near light (H0 and the outside forms there), the ratio q = ‖H − z₀‖/d of the
/// Neumann series. The generator of the series to order K (`Fold::series_generator`) is the fold's within its remainder's
/// bound q^{K+1}/(1 − q) (relatively, against the fold's own part H − A), and the difference falls with K as q's powers.
/// Measured: about the far levels q = 0.548 there and the difference 9.8e-3 at order 0, falling 0.51–0.54 an order;
/// about the levels with the near light's scalar shift q = 0.0054, 7.5e-5 at order 0 and 3.7e-7 at order 1 -- the scalar
/// shift is most of ‖H − z₀‖, and it is the near light's intensity's alone.
#[test]
fn the_fold_s_series_about_a_reference_is_the_fold_within_its_remainder_and_falls_as_its_ratio_s_powers() {
    let req = request("PGC3D_1ms");
    let light = crate::field::Light::new(&req);
    let ground: Vec<usize> = (0..crate::request::GROUND).filter(|&g| crate::request::Level::parse(&req.states[g]).unwrap().f == 2).collect();
    // The cooling light: the co-rotating class of the most beams; its coherent field's peak over a grid near the focus.
    let strength = |r: [f64; 3]| {
        let (_, classes, sources) = system(&req, &light, r);
        let cooling = (0..classes.len()).filter(|&c| !sources[c].1).max_by_key(|&c| classes[c].beams.len()).unwrap();
        let mut sum = Mat::zeros(classes[cooling].beams[0].coupling.rows, classes[cooling].beams[0].coupling.cols);
        for b in &classes[cooling].beams {
            sum.add_scaled(C64::ONE, &b.coupling);
        }
        sum.frobenius()
    };
    let grid: Vec<[f64; 3]> = (0..7_usize.pow(3)).map(|i| [(i % 7) as f64 * 0.05 - 0.15, (i / 7 % 7) as f64 * 0.05 - 0.15, (i / 49) as f64 * 0.05 - 0.15]).collect();
    let peak = grid.iter().copied().max_by(|a, b| strength(*a).total_cmp(&strength(*b))).unwrap();
    let (levels, classes, _) = system(&req, &light, peak);
    let excited: Vec<usize> = (0..levels.excited.rows).collect();
    let lv = Levels { ground: levels.ground.block(&ground, &ground), excited: levels.excited.clone(), decay: levels.decay.clone() };
    let cl: Vec<Class> = classes.iter().map(|c| Class { nu: c.nu, beams: c.beams.iter().map(|b| Beam { coupling: b.coupling.block(&excited, &ground), wave: b.wave }).collect() }).collect();
    let fold = folded(&lv, &cl, &at_rest(ground.len()));
    assert!(fold.exists);
    let n = ground.len();
    let mut own = fold.generator.clone();
    own.add_scaled(-C64::ONE, &fold.direct);
    // Two references: the states' mean level without the near light, and with its scalar shift (the mean of the fold's
    // eigenvalues: what the near light's intensity alone sets, the polarization's vector and tensor parts traceless).
    let far = C64::real((0..n).map(|i| fold.direct[(i, i)].re).sum::<f64>() / n as f64);
    let shifted = C64::real(fold.eigen.values.iter().map(|z| z.re).sum::<f64>() / n as f64);
    // The coordinator's choice: the far levels and one constant c for the whole light, the middle of the near light's
    // scalar shift's range over the grid's points -- no field in it, the tables' dimensions those of the far light.
    let (mut low, mut high) = (f64::INFINITY, f64::NEG_INFINITY);
    for &r in &grid {
        let (levels, classes, _) = system(&req, &light, r);
        let lv = Levels { ground: levels.ground.block(&ground, &ground), excited: levels.excited.clone(), decay: levels.decay.clone() };
        let cl: Vec<Class> = classes.iter().map(|c| Class { nu: c.nu, beams: c.beams.iter().map(|b| Beam { coupling: b.coupling.block(&excited, &ground), wave: b.wave }).collect() }).collect();
        let f = folded(&lv, &cl, &at_rest(ground.len()));
        let shift = (f.eigen.values.iter().map(|z| z.re).sum::<f64>() - (0..ground.len()).map(|i| f.direct[(i, i)].re).sum::<f64>()) / ground.len() as f64;
        low = low.min(shift);
        high = high.max(shift);
    }
    let centred = far + C64::real(0.5 * (low + high));
    println!("the near light's scalar shift over the grid: {low:.3} to {high:.3} rad/us");
    for (name, z0) in [("far levels", far), ("far levels and the range's centre", centred), ("levels with the point's scalar shift", shifted)] {
        let q = fold.series_ratio(z0);
        println!("PGC3D F=2 at {peak:?}, z0 at the {name}: the series' ratio q = {q:.4}");
        assert!(q < 1.0, "the series converges here: q = {q}");
        let mut last = f64::INFINITY;
        for order in 0..=6 {
            let forms = fold.series(z0, order);
            let h = fold.series_generator(&forms, z0).expect("the series' generator settles");
            let mut apart = h.clone();
            apart.add_scaled(-C64::ONE, &fold.generator);
            let relative = apart.frobenius() / own.frobenius();
            // The remainder's bound, and the rounding of the series' sums (n² products an order) beside it.
            let bound = q.powi(order as i32 + 1) / (1.0 - q) + crate::units::rounding(n * n * (order + 2));
            println!("  order {order}: {relative:.3e} of the fold apart, bound {bound:.3e}, against the order before {:.3}", relative / last);
            assert!(relative <= bound, "order {order}: {relative} beyond the bound {bound}");
            if order > 0 && relative > 1e3 * crate::units::UNIT_ROUNDOFF {
                assert!(relative <= 2.0 * q * last, "order {order}: falls by {} where q = {q}", relative / last);
            }
            last = relative;
        }
    }
}

/// The run's route against the definition (ELIMINATION v5, v7; assemble.rs, # The normalized generator): on PGC3D's and the
/// tilted 1D PGC's F = 2 states at three points and at PGC3D's cooling light's coherent peak, the normalized generator
/// the engine makes of the series' forms about z₀ -- N₀ = Σ W′R₀W and N₁ = Σ W′R₀²W, the metric X = Σ (R₀W)†(R₀W), the
/// loss L = Σ (R₀W)†D(R₀W) -- through `assemble::normal`, the Schrieffer–Wolff series, at the least orders K (of the
/// series' ratio q) and J (of the excited share x) whose proven bounds below lie within the share. Against the
/// self-consistent fold's des Cloizeaux generator S H S⁻¹, S = (1 + X)^{1/2} by X's eigenbasis (`Fold::normalization`),
/// and its decay S⁻¹(Σ_f T_f†D T_f)S⁻¹ -- the jumps' product on the slow state, Σ_a ‖C_a T S⁻¹ψ‖²:
/// - the Hermitian part within the series' bound q^{K+1}/(1 − q) (+ x q at K = 1, the fixed point's Y²E) and what the
///   normalization does to it, β = ‖Herm(S H S⁻¹) − Herm H‖ of the fold's own part, and Hermitian by construction (Herm
///   of the route, exactly);
/// - the decay within the series' bound on a rate (2r + r², of T†T) and the normalization's 2ε + ε², ε =
///   |b_{J+1}| x^{J+1}/(1 − x) √(1 + x) (relatively), and Hermitian by construction (S⁻¹ a polynomial of the Hermitian
///   X: S⁻¹ L S⁻¹ Hermitian to the rounding); at each order J from 0 on, within that order's bound -- the difference
///   falls with J as x's powers, down to the series' own;
/// - the landing map S(1 + Y)⁻¹ against the fold's S(1 + Σ S_f T_f)⁻¹;
/// - and the error the run books, the evaluation against the definition measured on states (`Fold::evaluated`,
///   `Fold::defined`, `Fold::evaluation_error`): on random slow states, falling from the series' order 0 to order 1 and
///   with the normalization's order. The bounds are this test's: the run books the measured numbers.
#[test]
fn the_schrieffer_wolff_series_is_the_fold_s_normalized_generator_within_its_remainders_and_hermitian_by_construction() {
  for name in ["PGC3D_1ms", "PGC1D_TILT_2"] {
    let req = request(name);
    let light = crate::field::Light::new(&req);
    let ground: Vec<usize> = (0..crate::request::GROUND).filter(|&g| crate::request::Level::parse(&req.states[g]).unwrap().f == 2).collect();
    let n = ground.len();
    // The cooling light's coherent peak over a grid near the focus: the largest excited share.
    let strength = |r: [f64; 3]| {
        let (_, classes, sources) = system(&req, &light, r);
        let cooling = (0..classes.len()).filter(|&c| !sources[c].1).max_by_key(|&c| classes[c].beams.len()).unwrap();
        let mut sum = Mat::zeros(classes[cooling].beams[0].coupling.rows, classes[cooling].beams[0].coupling.cols);
        for b in &classes[cooling].beams {
            sum.add_scaled(C64::ONE, &b.coupling);
        }
        sum.frobenius()
    };
    let grid: Vec<[f64; 3]> = (0..7_usize.pow(3)).map(|i| [(i % 7) as f64 * 0.05 - 0.15, (i / 7 % 7) as f64 * 0.05 - 0.15, (i / 49) as f64 * 0.05 - 0.15]).collect();
    let peak = grid.iter().copied().max_by(|a, b| strength(*a).total_cmp(&strength(*b))).unwrap();
    for r in [[0.15, 0.0, 0.15], [0.05, -0.1, 0.0], [0.0; 3], peak] {
        let (levels, classes, _) = system(&req, &light, r);
        let excited: Vec<usize> = (0..levels.excited.rows).collect();
        let lv = Levels { ground: levels.ground.block(&ground, &ground), excited: levels.excited.clone(), decay: levels.decay.clone() };
        let cl: Vec<Class> = classes.iter().map(|c| Class { nu: c.nu, beams: c.beams.iter().map(|b| Beam { coupling: b.coupling.block(&excited, &ground), wave: b.wave }).collect() }).collect();
        let fold = folded(&lv, &cl, &at_rest(n));
        assert!(fold.exists);
        let (s, s_inverse) = fold.normalization();
        let definition = s.product(&fold.generator).product(&s_inverse);
        let mut loss = Mat::zeros(n, n);
        for rep in &fold.replicas {
            loss.add_scaled(C64::ONE, &rep.t.adjoint_product(&lv.decay.block(&rep.q, &rep.q).product(&rep.t)));
        }
        let decay = s_inverse.product(&loss).product(&s_inverse);
        // The reference: the point's scalar shift's centre (here the point's own); the orders it asks.
        let far = (0..n).map(|i| fold.direct[(i, i)].re).sum::<f64>() / n as f64;
        let z0 = fold.eigen.values.iter().map(|z| z.re).sum::<f64>() / n as f64;
        let q = fold.series_ratio(C64::real(z0));
        let x_size = fold.evaluated_share(C64::real(z0)).0;
        assert!(q < 1.0 && x_size < 1.0, "{name}: the series converge here");
        // The proven bounds (this test's; the run books what it measures).
        let remainder = |k: usize| {
            let r = q.powi(k as i32 + 1) / (1.0 - q) + if k >= 1 { x_size * q } else { 0.0 };
            (2.0 * r + r * r).min(1.0)
        };
        let rate = |j: usize| {
            let e = binomial(-0.5, j + 1).abs() * x_size.powi(j as i32 + 1) / (1.0 - x_size) * (1.0 + x_size).sqrt();
            (2.0 * e + e * e).max(x_size.powi(j as i32 + 1) / (1.0 - x_size)).min(1.0)
        };
        let order = (0..).find(|&k| remainder(k) <= share()).unwrap();
        assert!(order <= 1, "{name}: the forms hold the order {order}");
        let normalization = (0..).find(|&j| rate(j) <= share()).unwrap();
        let beta = {
            let mut moved = definition.hermitian_part();
            moved.add_scaled(-C64::ONE, &fold.generator.hermitian_part());
            let mut own = fold.generator.hermitian_part();
            own.add_scaled(-C64::ONE, &fold.direct.hermitian_part());
            moved.frobenius() / own.frobenius()
        };
        // The forms (every class, every replica, as the engine's per carrier and basis), from the reference's responses.
        let nus: Vec<f64> = cl.iter().map(|c| c.nu).collect();
        let reference = Reference::new(&lv, &nus, &at_rest(n), z0);
        let (mut n0, mut n1, mut x, mut l) = (Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n));
        let mut responses = vec![];
        for (c, class) in cl.iter().enumerate() {
            let mut w = Mat::zeros(excited.len(), n);
            for b in &class.beams {
                w.add_scaled(C64::ONE, &b.coupling);
            }
            let t: Vec<Mat> = reference.respond(c, &w, n, 1, &[]).into_iter().map(|mut parts| parts.remove(0)).collect();
            n0.add_scaled(C64::ONE, &w.adjoint_product(&t[0]));
            n1.add_scaled(C64::ONE, &w.adjoint_product(&t[1]));
            x.add_scaled(C64::ONE, &t[0].adjoint_product(&t[0]));
            l.add_scaled(C64::ONE, &t[0].adjoint_product(&lv.decay.product(&t[0])));
            responses.push(t);
        }
        let mut whole = fold.direct.clone();
        whole.add_scaled(C64::ONE, &n0);
        let hermitian = whole.hermitian_part();
        let four = |m: &Mat| -> Vec<C64> { m.data.iter().copied().chain(std::iter::repeat_n(C64::ZERO, 3 * n * n)).collect() };
        let made = |orders: Orders| -> Vec<C64> {
            let mut data = vec![C64::ZERO; 14 * n * n];
            crate::model::assemble::normal(&four(&hermitian), &four(&n1), &four(&x), &four(&l), n, [z0, 0.0, 0.0, 0.0], orders, true, true, true, &mut data, &mut vec![]);
            data
        };
        let matrix = |data: &[C64], m: usize| Mat { rows: n, cols: n, data: data[m * n * n..(m + 1) * n * n].to_vec() };
        let orders = Orders { series: order, normalization };
        let data = made(orders);
        let held = matrix(&data, crate::model::assemble::HELD);
        let mut route = hermitian.clone();
        route.add_scaled(C64::ONE, &held);
        let mut apart = route.clone();
        apart.add_scaled(-C64::ONE, &definition.hermitian_part());
        let mut own = definition.hermitian_part();
        own.add_scaled(-C64::ONE, &fold.direct.hermitian_part());
        let relative = apart.frobenius() / own.frobenius();
        let mut skew = route.clone();
        skew.add_scaled(-C64::ONE, &route.adjoint_product(&Mat::identity(n)));
        // The decay at each order J of the normalization: within its bound.
        let generator_bound = q.powi(order as i32 + 1) / (1.0 - q) + if order >= 1 { x_size * q } else { 0.0 } + crate::units::rounding(n * n * (order + 2));
        let series_bound = remainder(order) + crate::units::rounding(n * n * (order + 2));
        // The jumps' product on the slow state, of the wave operator to the series' order: T = R₀W − R₀²W E (E = H − z₀),
        // Σ over the classes of T†D T on S⁻¹.
        let decay_apart = |j: usize| -> (f64, f64) {
            let data = made(Orders { normalization: j, ..orders });
            let (bent, e) = (matrix(&data, crate::model::assemble::BENT), matrix(&data, crate::model::assemble::OFFSET));
            let mut inner = Mat::zeros(n, n);
            for t in &responses {
                let mut wave = t[0].clone();
                if order >= 1 {
                    wave.add_scaled(-C64::ONE, &t[1].product(&e));
                }
                inner.add_scaled(C64::ONE, &wave.adjoint_product(&lv.decay.product(&wave)));
            }
            let route = bent.product(&inner).product(&bent);
            let mut apart = route.clone();
            apart.add_scaled(-C64::ONE, &decay);
            let mut skew = route.clone();
            skew.add_scaled(-C64::ONE, &route.adjoint_product(&Mat::identity(n)));
            (apart.frobenius() / decay.frobenius(), skew.frobenius() / route.frobenius())
        };
        let landing = matrix(&data, crate::model::assemble::LANDING);
        let exact = {
            let left = fold.left_operators().expect("the left operators settle");
            let bares: Vec<Vec<C64>> = (0..n).map(|j| (0..n).map(|i| if i == j { C64::ONE } else { C64::ZERO }).collect()).collect();
            let landed = fold.transients(&left, &bares).expect("the landing's overlap");
            s.product(&Mat::from_fn(n, n, |i, j| landed[j].0[i]))
        };
        let mut landing_apart = landing.clone();
        landing_apart.add_scaled(-C64::ONE, &exact);
        println!(
            "{name} F=2 at {r:?}: z0 - far {:.3} rad/us, q {q:.4}, K {order}; x {x_size:.4}, J {normalization}; beta {beta:.3e}; the Hermitian route {relative:.3e} of the fold's part apart (skew {:.1e}); landing {:.3e} apart",
            z0 - far,
            skew.frobenius(),
            landing_apart.frobenius() / exact.frobenius()
        );
        let hermitian_bound = generator_bound + beta;
        assert!(relative <= hermitian_bound.max(1e-9), "the route's Hermitian generator {relative} apart, beyond {hermitian_bound}");
        assert_eq!(skew.frobenius(), 0.0, "the route's Hermitian part is Hermitian");
        for j in 0..=normalization + 2 {
            let (apart, skew) = decay_apart(j);
            let bound = series_bound + rate(j);
            println!("    J {j}: the decay {apart:.3e} apart, bound {bound:.3e} (skew {skew:.1e})");
            assert!(apart <= bound, "{name} at {r:?}, J {j}: the decay {apart} beyond its bound {bound}");
            assert!(skew <= crate::units::rounding(4 * n * n * (j + 1)), "the decay is Hermitian: {skew}");
        }
        assert!(landing_apart.frobenius() <= (q + x_size * q + rate(normalization)).max(1e-9) * exact.frobenius(), "the landing map {}", landing_apart.frobenius() / exact.frobenius());
        // The errors the run books, measured on random slow states against the definition (`Fold::evaluated` against
        // `Fold::defined`, a point's numbers, not bounds): the evaluation's normalized generator at its orders is this
        // test's route (the same Hermitian part, the same decay), and the error falls from the series' order 0 to 1 and
        // with the normalization's order.
        let defined = fold.defined(&lv);
        let at = |orders: Orders| fold.evaluated(&lv, C64::real(z0), orders);
        let mut apart = at(orders);
        let mut route_whole = route.hermitian_part();
        route_whole.add_scaled(C64::new(0.0, -0.5), &{
            let data = made(orders);
            let (bent, e) = (matrix(&data, crate::model::assemble::BENT), matrix(&data, crate::model::assemble::OFFSET));
            let mut inner = Mat::zeros(n, n);
            for t in &responses {
                let mut wave = t[0].clone();
                if order >= 1 {
                    wave.add_scaled(-C64::ONE, &t[1].product(&e));
                }
                inner.add_scaled(C64::ONE, &wave.adjoint_product(&lv.decay.product(&wave)));
            }
            bent.product(&inner).product(&bent)
        });
        apart.add_scaled(-C64::ONE, &route_whole);
        assert!(apart.frobenius() <= 1e-9 * route_whole.frobenius(), "{name}: the evaluated generator is the route's, {} apart", apart.frobenius() / route_whole.frobenius());
        let mut stream = crate::random::Stream::new(3, crate::random::Purpose::Initial, 0, 0);
        for _ in 0..4 {
            let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
            let error = |series: usize, normalization: usize| Fold::evaluation_error(&at(Orders { series, normalization }), &defined, C64::real(z0), &psi);
            let (first, second) = (error(0, normalization), error(1, normalization));
            let by_j: Vec<f64> = (0..3).map(|j| error(1, j)).collect();
            println!("    measured against the definition: the series' order 0 {first:.3e}, order 1 {second:.3e} (q {q:.1e}); at order 1 the normalization's J 0..2 {by_j:?} (x {x_size:.1e})");
            assert!(first > 0.0 && second < first, "{name}: the series' measured errors {first}, {second}");
            assert!(by_j[1] < by_j[0], "{name}: the normalization's measured errors {by_j:?}");
        }
    }
  }
}

/// A group's replica is solved on its own places (module documentation, # Groups; the coordinator's test): two groups,
/// ground state g0 holding the excited state e explicit (uncoupled to it, so that its eigenvalue is e's own level
/// exactly) and ground state g1 driven into e by a class whose replica sits at e's frame -- where e is Q for g1's group,
/// at the shift that puts g0's group's excited eigenvalue exactly on K's spectrum. The fold is finite, and it is the
/// two groups' folds made apart: their eigenvalues together and g1's response, to round-off.
#[test]
fn a_group_s_replica_is_solved_on_its_own_places_whatever_another_group_holds_on_its_q() {
    let (gamma, nu, g) = (38.1, -120.0, 30.0);
    let lv = levels(&[0.0, 5.0], &[0.0], gamma, &[0]);
    let classes = vec![class(nu, 2, 1, &[(0, 1, C64::real(g))])];
    let slow = Slow { explicit: vec![0], frames: vec![0.0, 0.0, nu], groups: vec![0, 1, 0] };
    let whole = folded(&lv, &classes, &slow);
    assert!(whole.generator.data.iter().all(|z| z.re.is_finite() && z.im.is_finite()), "a finite fold");
    let first = folded(&levels(&[0.0], &[0.0], gamma, &[0]), &[class(nu, 1, 1, &[])], &Slow::single(vec![0], vec![0.0, nu]));
    let second = folded(&levels(&[5.0], &[0.0], gamma, &[0]), &[class(nu, 1, 1, &[(0, 0, C64::real(g))])], &Slow::single(vec![], vec![0.0]));
    let mut apart: Vec<C64> = first.eigen.values.iter().chain(&second.eigen.values).copied().collect();
    let mut together = whole.eigen.values.clone();
    for v in [&mut apart, &mut together] {
        v.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
    }
    for (a, b) in together.iter().zip(&apart) {
        assert!((*a - *b).abs() <= 1e-12 * b.abs().max(1.0), "an eigenvalue {a:?} against {b:?}");
    }
    let response = |fold: &Fold, place: usize| fold.replicas.iter().find(|rep| rep.feeds.iter().any(|f| f.class == Some(0))).map(|rep| rep.t[(0, place)]).expect("the drive's replica");
    let (a, b) = (response(&whole, 1), response(&second, 0));
    assert!((a - b).abs() <= 1e-12 * b.abs(), "g1's response {a:?} against {b:?}");
}

/// The test requests of the levels outside the model (refs/requests/OUTSIDE_776_*, made by the app from R2_RECOIL's scene,
/// the rest worktree's ladder.mjs): R2's σ⁺ probe on |2,2⟩ → |3′,3′⟩ and a 776 nm plane wave of 1 kW/cm² along it, σ⁺,
/// Δ red of |3′,3′⟩ → 5D5/2 (m_J = 5/2, m_I = 3/2). Its coupling there V = ⟨5D5/2‖d‖5P3/2⟩ E₀/(√6 · 2h): 1.76 ea₀ (the
/// app's table), E₀ = √(2I/ε₀c), the stretched pair's 3j symbol 1/√6 -- 399.0 MHz.
const LADDER_V_MHZ: f64 = 399.018;

/// A frequency of a request by one of its beams' ids.
fn frequency_of(request: &crate::request::Request, id: &str) -> usize {
    let light = crate::field::Light::new(request);
    light.frequencies.iter().position(|f| f.beams.iter().any(|&l| request.beams[l].id == id)).unwrap()
}

/// The model of a request with its probe's drives at the atom (`Unit::Bare`) and the units `kept`, and the place among its
/// outside transitions of the ladder beam's 5P3/2→5D5/2 (none without that beam).
fn ladder(request: &crate::request::Request, kept: &[crate::model::Unit]) -> (crate::model::Model, Option<usize>) {
    let light = crate::field::Light::new(request);
    let bare = crate::model::Model::new(request, &light, &[]).unwrap();
    let mut units = bare.bare_units(crate::model::Component { frequency: frequency_of(request, "probe"), counter: false });
    units.extend_from_slice(kept);
    let model = crate::model::Model::new(request, &light, &units).unwrap();
    let beam = request.beams.iter().any(|b| b.id == "ladder").then(|| frequency_of(request, "ladder"));
    let t = beam.and_then(|w| model.outside.iter().position(|(f, t)| *f == w && t.name == "5P3/2→5D5/2"));
    (model, t)
}

/// The transitions' mixing as the engine reads it (request.rs `Transition`, `Model::outside_at`) is the app's (V/Δ)² of the
/// stretched pair, the one sublevel its σ⁺ reaches, and its shift V²/Δ: (399.0/500)² = 0.637 and (399.0/30)² = 176.9 at
/// |3′,3′⟩ -- the field's coordinates, the units and the forms' polarization identity end to end (the counter-rotating
/// term, (V/2ν)² ~ 3e-13, and the forms' rounding apart).
#[test]
fn the_outside_mixing_is_the_stretched_pair_s_coupling_over_its_detuning() {
    for (name, delta) in [("OUTSIDE_776_500", 500.0), ("OUTSIDE_776_30", 30.0)] {
        let request = request(name);
        let (model, t) = ladder(&request, &[]);
        let t = t.expect("the ladder's transition");
        let s = request.states.iter().position(|x| x == "D2-3-3").unwrap();
        let (m, o) = &model.outside_at(&model.light.amplitudes([0.0; 3]))[t];
        let (expected, shift) = ((LADDER_V_MHZ / delta).powi(2), LADDER_V_MHZ * LADDER_V_MHZ / delta);
        let mhz = o[s] / crate::units::RAD_PER_US_PER_MHZ;
        eprintln!("{name}: m {:.6} against (V/Δ)² {expected:.6}; its shift {mhz:.4} MHz against V²/Δ {shift:.4}", m[s]);
        assert!((m[s] / expected - 1.0).abs() <= 1e-5, "{name}: m {} against {expected}", m[s]);
        assert!((mhz.abs() / shift - 1.0).abs() <= 1e-5, "{name}: the shift {mhz} against {shift}");
        assert!((0..crate::request::GROUND).all(|g| m[g] == 0.0), "{name}: no ground state reaches 5D");
    }
}

/// An excited state explicit in the model (the probe's drive |2,2⟩ → |3′,3′⟩ kept) leaks its population into the level:
/// the state √0.7 |2,2⟩ + √0.3 |3′,3′⟩ books at least 0.3 m(3′,3′) for 5P3/2→5D5/2 -- 0.19 at Δ = 500 MHz, beyond the
/// model's share --, and far from the level (Δ = 100 GHz, m = 1.6e-5) every transition a thousandth of the share at most.
#[test]
fn an_explicit_excited_state_leaks_its_population_into_the_outside_level() {
    use crate::model::{Component, Point, Sample, Unit};
    let share = super::share();
    let (b, e) = (7, 31);
    for (name, delta) in [("OUTSIDE_776_FAR", 1e5), ("OUTSIDE_776_500", 500.0)] {
        let request = request(name);
        let drive = Unit::Drive { component: Component { frequency: frequency_of(&request, "probe"), counter: false }, excited: e, ground: b };
        let (model, t) = ladder(&request, &[drive]);
        let t = t.expect("the ladder's transition");
        let cluster = model.cluster_of(b);
        let c = &model.clusters()[cluster];
        assert!(c.states.contains(&e), "{name}: |3′,3′⟩ explicit");
        let psi: Vec<C64> = c.states.iter().map(|&s| C64::real(if s == b { 0.7_f64.sqrt() } else if s == e { 0.3_f64.sqrt() } else { 0.0 })).collect();
        let (_, outside) = model.readings(&[], &Sample { cluster, at: Point { r: [0.0; 3], kappa: [0.0; 3], t: 0.0 }, psi }, None);
        let leaked = 0.3 * (LADDER_V_MHZ / delta).powi(2);
        let largest = outside.iter().map(|x| x.population).fold(0.0, f64::max);
        eprintln!("{name}: booked {:.4e} for {} (0.3 m = {leaked:.4e}; the largest transition {largest:.4e}; the share {share:.4})", outside[t].population, model.outside_label(t));
        assert!(outside[t].population >= leaked * (1.0 - 1e-5), "{name}: {} below the leak {leaked}", outside[t].population);
        if delta < 1e3 {
            assert!(outside[t].population > share);
        } else {
            assert!(largest <= 1e-3 * share, "{name}: far, {largest}");
        }
    }
}

/// The eliminated excited states' level error acts through the fold (`outside_errors`; the coordinator's 2026-11-13
/// example): R2_RECOIL with the 776 nm beam, the atom in |2,2⟩ at the trap's centre and the probe folded at the atom.
/// The beam moves |3′,3′⟩ by V²/Δ (318 MHz at Δ = 500 MHz, 5.3 GHz at 30) with an error m of that, and the probe's
/// response feels it across its distance to the line: booked beyond the share at 500 MHz, more at 30 MHz -- where
/// |3′,3′⟩'s population, the line moved away, is smaller still. Far from the level (100 GHz) and without the beam
/// (R2_RECOIL itself), every transition's error is a thousandth of the share at most.
#[test]
fn a_shifted_excited_level_s_error_is_booked_through_the_fold() {
    use crate::model::{Point, Sample};
    let share = super::share();
    let b = 7;
    let r2 = format!("{}/refs/scenes/R2_RECOIL.json", env!("CARGO_MANIFEST_DIR"));
    let mut booked = vec![];
    for name in [r2.as_str(), "OUTSIDE_776_FAR", "OUTSIDE_776_500", "OUTSIDE_776_30"] {
        let request = if name.ends_with(".json") { crate::request::Request::from_native(&serde_json::from_str(&std::fs::read_to_string(name).unwrap()).unwrap()).unwrap() } else { request(name) };
        let (model, t) = ladder(&request, &[]);
        let p: [f64; crate::request::GROUND] = std::array::from_fn(|s| if s == b { 1.0 } else { 0.0 });
        let centre = model.trap(&p, &model.prepared()).unwrap().centre;
        let cluster = model.cluster_of(b);
        let psi: Vec<C64> = model.clusters()[cluster].states.iter().map(|&s| if s == b { C64::ONE } else { C64::ZERO }).collect();
        let (_, outside) = model.readings(&[], &Sample { cluster, at: Point { r: centre, kappa: [0.0; 3], t: 0.0 }, psi }, None);
        let largest = outside.iter().map(|x| x.population).fold(0.0, f64::max);
        let own = t.map_or(0.0, |t| outside[t].population);
        eprintln!("{}: 5P3/2→5D5/2 booked {own:.4e}, the largest transition {largest:.4e} (share {share:.4})", name.rsplit('/').next().unwrap());
        booked.push((own, largest));
    }
    assert!(booked[0].1 <= 1e-3 * share && booked[1].1 <= 1e-3 * share, "without the beam, or far from the level: {booked:?}");
    assert!(booked[2].0 > share, "Δ = 500 MHz: {:?}", booked[2]);
    assert!(booked[3].0 > booked[2].0, "Δ = 30 MHz books more: {booked:?}");
}

/// The series about a reference evaluated as the definition takes it (module documentation of elim.rs, # The definition
/// and the route it is evaluated on): the tests' reference for the engine's route (`Reference`, `Fold::evaluated`).
impl Fold {
    /// The fold's series about a reference z₀ (the engine's evaluation route; the self-consistent fold is the
    /// definition): the resolvent's Neumann series R(z) = Σ_k (z₀ − z)^k R₀^{k+1}, R₀ = (z₀ + f − K)⁻¹, with z the P
    /// generator H itself (a right factor, as in the Sylvester equation) gives the wave operator
    /// T_f = Σ_k (−1)^k R₀^{k+1} W_f (H − z₀)^k and the generator H = A + Σ_k (−1)^k N_k (H − z₀)^k,
    /// N_k = Σ_f W′_f R₀^{k+1} W_f. The N_k (and the jumps' C R₀^{k+1} W) are bilinear in the fields with coefficients
    /// of the reference levels alone: what a table holds without the near light's coordinates. Returns N_0..N_K.
    fn series(&self, z0: C64, orders: usize) -> Vec<Mat> {
        let n = self.size();
        let mut out = vec![Mat::zeros(n, n); orders + 1];
        for (rep, responses) in self.replicas.iter().zip(self.responses_about(z0, orders)) {
            for (form, x) in out.iter_mut().zip(&responses) {
                form.add_scaled(C64::ONE, &rep.back.product(x));
            }
        }
        out
    }
    /// The generator of the series to order K (`series`): the fixed point of H ← A + Σ_k (−1)^k N_k (H − z₀)^k from A
    /// (`settle`, on n × n matrices alone); None where it does not settle.
    fn series_generator(&self, forms: &[Mat], z0: C64) -> Option<Mat> {
        let n = self.size();
        let (h, exists) = settle(
            &self.direct,
            |h| {
                let mut e = h.clone();
                for i in 0..n {
                    e[(i, i)] -= z0;
                }
                let (mut next, mut power, mut scale) = (self.direct.clone(), Mat::identity(n), self.direct.frobenius());
                for (k, form) in forms.iter().enumerate() {
                    let term = form.product(&power);
                    scale += term.frobenius();
                    next.add_scaled(C64::real(if k % 2 == 0 { 1.0 } else { -1.0 }), &term);
                    power = power.product(&e);
                }
                (next, scale)
            },
            n * n * forms.len(),
        );
        exists.then_some(h)
    }
    /// The series' ratio q = ‖H − z₀‖₂ / d (the spectral norm, the largest singular value; d the least distance of
    /// z₀ + f from the excited block's spectrum over the replicas, 1/‖R₀‖ where the excited block is normal): its
    /// remainder after order K is within q^{K+1}/(1 − q) of the fold, relatively, where q < 1 (the Neumann series'
    /// radius) -- the order an evaluation takes is the least whose remainder is within the share (criterion one's ρ
    /// books it).
    fn series_ratio(&self, z0: C64) -> f64 {
        self.series_norm(z0) / self.series_distance(z0, None)
    }
}

/// A series' radius is read on the blocks its terms hold (`Folded::held_blocks`, `Fold::series_distance`): R₀ = (z₀ + f −
/// K_f)⁻¹ is block-diagonal on K_f's blocks, so a block that neither W nor δK reaches is in no term of the series. S2Q's
/// probe with its F′ = 3 block kept explicit for the three F = 2 states it drives (the first attempt's keep): its σ± light
/// joins |2, m⟩ to |3′, m ± 1⟩ alone, so the kept edges split the block by the parity of m into two clusters, each with
/// the other's half in its Q, nearer z₀ + f than ‖H − z₀‖, in no term of its series. Read on every block, that half's
/// distance gave q ≥ 1 (2.9), and the closure made the hyperfine blocks below explicit one after another -- growing
/// ‖H − z₀‖, never the distance -- until the repump's block move joined a ground driven by both carriers into one block,
/// a 6.57 GHz loop in the frames no step resolves. On the held blocks the series converges and the closure keeps nothing
/// more; and the series' generator to each order K is within q^{K+1}/(1 − q) of the self-consistent fold at q of the held
/// blocks, where an unheld block's distance would bound nothing.
#[test]
fn a_series_radius_is_read_on_the_blocks_its_terms_hold() {
    use super::super::{Component, Model, Unit};
    use crate::request::GROUND;
    let req = request("S2Q");
    let light = crate::field::Light::new(&req);
    let chief = (0..GROUND).max_by(|&a, &b| req.initial_density[(a, a)].re.total_cmp(&req.initial_density[(b, b)].re)).unwrap();
    let (base, mut kept, _, _) = Model::closed_at(&req, &light, vec![], chief, req.temperature_uk);
    let state = |label: &str| req.states.iter().position(|s| s == label).expect("a state of the request");
    let block = base.blocks().iter().position(|b| b.contains(&(state("D2-3-0") - GROUND))).expect("the F′ = 3 block");
    let probe = (0..light.frequencies.len()).find(|&f| light.frequencies[f].beams.iter().any(|&l| req.beams[l].id == "laser-probe")).expect("the probe's carrier");
    let component = Component { frequency: probe, counter: false };
    for m in ["-2", "-1", "0"] {
        kept.push(Unit::Block { block, component, grounds: ground_set(&[state(&format!("ground-2-{m}"))]) });
    }
    kept.push(Unit::Normalization { order: 1 });
    let mut model = Model::new(&req, &light, &kept).expect("the model");
    let points = crate::solve::atoms(&model, &req, &light, &crate::solve::components(&req), req.temperature_uk).expect("the atoms");
    model.referenced(&points);
    // The scene holds the class: a cluster whose Q has a block in no term of its series nearer z₀ + f than ‖H − z₀‖.
    let mut worst: Option<(f64, Fold, C64)> = None;
    let mut unheld = 0;
    for k in 0..model.clusters().len() {
        for &(r, _) in &points {
            let amplitudes = light.amplitudes(r);
            let Some((fold, _, far, _)) = model.cluster_fold(k, &amplitudes, false) else { continue };
            let z0 = C64::real(far + model.series.centre[k]);
            let every = fold.replicas.iter().flat_map(|rep| (0..rep.k.n).map(move |i| (z0 + C64::real(rep.frequency) - rep.k.s[i * rep.k.n + i]).abs())).fold(f64::INFINITY, f64::min);
            let norm = fold.series_norm(z0);
            if every < norm && fold.series_distance(z0, None) > norm {
                unheld += 1;
            }
            let q = fold.series_ratio(z0);
            if worst.as_ref().is_none_or(|w| q > w.0) {
                worst = Some((q, fold, z0));
            }
        }
    }
    println!("S2Q, the probe's F′ = 3 block kept for F = 2 m = −2, −1, 0: {} clusters; {unheld} cluster folds with a block in no term nearer than ‖H − z₀‖; series q = {:.3e}, x = {:.3e}", model.clusters().len(), model.series.ratio, model.series.excited);
    assert!(unheld > 0, "the scene holds a block the series does not hold within ‖H − z₀‖ of z₀ + f");
    assert!(model.series.exists(), "the series converge on the blocks they hold: q = {}, x = {}", model.series.ratio, model.series.excited);
    assert_eq!(model.series_block(), None, "the closure makes no block explicit");
    // The series' orders at its largest ratio converge as that ratio bounds them.
    let (q, fold, z0) = worst.expect("a cluster with a fold");
    let n = fold.size();
    let own = fold.generator.clone();
    for order in 0..=3 {
        let forms = fold.series(z0, order);
        let h = fold.series_generator(&forms, z0).expect("the series' generator settles");
        let mut apart = h.clone();
        apart.add_scaled(-C64::ONE, &fold.generator);
        let relative = apart.frobenius() / own.frobenius();
        let bound = q.powi(order as i32 + 1) / (1.0 - q) + crate::units::rounding(n * n * (order + 2));
        println!("  order {order}: {relative:.3e} of the fold apart, bound {bound:.3e} (q = {q:.3e})");
        assert!(relative <= bound, "order {order}: {relative} beyond the bound {bound}");
    }
}

/// A drive folded with the bare levels onto an explicit state exists where its ratio 2a/|d| is below one (model.rs
/// `Model::settled`, `converges`), and at a two-photon distance d of nothing it is not: R4_LIN_1 at B = 0, its lin⊥lin
/// light driving an excited sublevel from two degenerate ground sublevels -- one drive kept, the other's Raman pair with it
/// is at one energy. The model is made (the bare fold, far.rs `second_order`, is never read at its pole), and the drive
/// goes up one level, to its fold at the atom (`Level::Atom`) -- where the closure reads the pair on the actual spectrum --,
/// for every such pair the light makes.
#[test]
fn a_drive_onto_an_explicit_state_at_no_two_photon_distance_is_folded_at_the_atom() {
    use crate::request::GROUND;
    let path = format!("{}/refs/scenes/R4_LIN_1.json", env!("CARGO_MANIFEST_DIR"));
    let request = crate::request::Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
    let light = crate::field::Light::new(&request);
    let bare = super::super::Model::new(&request, &light, &[]).unwrap();
    let mut pairs = 0;
    for &(c, e, a) in bare.driven.iter().filter(|d| !d.0.counter) {
        for &(_, _, w) in bare.driven.iter().filter(|d| d.0 == c && d.1 == e && d.2 > a && d.2 < GROUND && bare.energies[d.2] == bare.energies[a]) {
            let kept = super::super::Model::new(&request, &light, &[super::super::Unit::Drive { component: c, excited: e, ground: a }]).unwrap();
            assert_eq!(kept.level(c, e, w), super::super::Level::Atom, "{} → {}: the drive from {w} is folded at the atom", a, e);
            pairs += 1;
        }
    }
    assert!(pairs > 0, "R4_LIN_1 drives an excited sublevel from two degenerate ground sublevels");
}

/// The same statement where the distance is small but not nothing (model.rs `Model::settled`): R4_LIN_1's first pair of
/// degenerate ground sublevels a, w driven to one excited sublevel e, w lifted by ε as a small field splits the pair,
/// the drive e ← a kept. The fold of e ← w onto the explicit e is at two-photon distance d = −ε (e is read at a's
/// energy, its frame's beat the photon's): where its ratio 2a/ε is far beyond one the drive goes up to its fold at the
/// atom; where it is well below, the drive stays folded with the bare levels and w keeps its own cluster (how well that
/// fold does is read at the states, `Model::bare_mixing`).
#[test]
fn a_drive_onto_an_explicit_state_at_a_small_two_photon_distance_is_folded_at_the_atom_and_at_a_large_one_bare() {
    use super::super::{Level, Model, Unit};
    use crate::request::GROUND;
    let path = format!("{}/refs/scenes/R4_LIN_1.json", env!("CARGO_MANIFEST_DIR"));
    let request = crate::request::Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
    let light = crate::field::Light::new(&request);
    let bare = Model::new(&request, &light, &[]).unwrap();
    let (c, e, a, w) = bare
        .driven
        .iter()
        .filter(|d| !d.0.counter)
        .find_map(|&(c, e, a)| bare.driven.iter().find(|d| d.0 == c && d.1 == e && d.2 > a && d.2 < GROUND && bare.energies[d.2] == bare.energies[a]).map(|d| (c, e, a, d.2)))
        .expect("R4_LIN_1 drives an excited sublevel from two degenerate ground sublevels");
    let size: f64 = bare.tones[c.frequency].couplings[0].iter().map(|x| x[(e, w)].abs()).sum();
    for (epsilon, exists) in [(1e-3 * size, false), (30.0 * size, true)] {
        let ratio = 2.0 * size / epsilon;
        assert_eq!(super::converges(ratio), exists, "the premise: 2a/ε = {ratio:.3e}");
        let mut split = request.clone();
        split.hamiltonian[(w, w)] += C64::new(epsilon, 0.0);
        let model = Model::new(&split, &light, &[Unit::Drive { component: c, excited: e, ground: a }]).unwrap();
        let level = model.level(c, e, w);
        assert_eq!(level, if exists { Level::Bare } else { Level::Atom }, "{a} → {e} kept, ε = {epsilon:.3e}, 2a/ε = {ratio:.3e}: the drive from {w} at level {level:?}");
        assert_ne!(model.cluster_of(w), model.cluster_of(a), "the drive from {w} joins no cluster");
    }
}

/// A drive goes to the atom by its own part of its class's route (`bare_routes`; the prior, `Model::prior`): the 1D PGC
/// at 3 G (PGC1D_TILT_0) and with its probe tilted 2° (PGC1D_TILT_2). Tilted, every beam's coupling holds parts of
/// ≤ 1.7·10⁻⁸ of its largest at every Δm -- the probe's light shift through the levels outside the model mixing the
/// excited sublevels, physical and of no consequence (its rates' part ~10⁻¹⁶) -- beside the probe's π (1.59 of 83.2) and
/// σ⁻ (2.5·10⁻²). Read by its class's whole route, every drive of a class went to the atom (576 against 106) and the near
/// forms were dense. By each drive's own part, the rest within what no output resolves: untilted 56 of the 106 (the
/// other 50 carry together at most 2.4·10⁻⁴ of their class's response, the largest 1.8·10⁻⁴); tilted, no drive of a
/// part below 10⁻⁶ of its beam's largest coupling at the atom, the probe's π drives among them, and at most the 72
/// physical ones more than untilted.
#[test]
fn a_drive_goes_to_the_atom_by_its_own_part_of_the_route() {
    use crate::request::{Level as Label, GROUND};
    let mut counts = vec![];
    for name in ["PGC1D_TILT_0", "PGC1D_TILT_2"] {
        let req = request(name);
        let light = crate::field::Light::new(&req);
        let chief = (0..GROUND).max_by(|&a, &b| req.initial_density[(a, a)].re.total_cmp(&req.initial_density[(b, b)].re)).unwrap();
        let (_, kept, _, _) = super::super::Model::closed_at(&req, &light, vec![], chief, req.temperature_uk);
        let id = |frequency: usize| light.frequencies[frequency].beams.iter().map(|&l| req.beams[l].id.clone()).collect::<Vec<_>>().join("+");
        let label = |s: usize| Label::parse(&req.states[s]).unwrap();
        // A drive's coupling relative to its beams' largest.
        let relative = |frequency: usize, e: usize, g: usize| {
            light.frequencies[frequency].beams.iter().map(|&l| {
                let c = &req.beams[l].coupling;
                c[(e, g)].abs() / c.max_abs()
            }).fold(0.0, f64::max)
        };
        let bare: Vec<(usize, usize, usize)> = kept.iter().filter_map(|u| if let super::super::Unit::Bare { component, excited, ground } = u { (!component.counter).then_some((component.frequency, *excited, *ground)) } else { None }).collect();
        let smallest = bare.iter().map(|&(f, e, g)| relative(f, e, g)).fold(f64::INFINITY, f64::min);
        let probe = (0..light.frequencies.len()).find(|&f| id(f) == "probe").expect("the probe");
        let pi: Vec<(usize, usize)> = (GROUND..req.states.len()).flat_map(|e| (0..GROUND).map(move |g| (e, g))).filter(|&(e, g)| label(e).m == label(g).m && relative(probe, e, g) > 1e-3).collect();
        let pi_at_atom = pi.iter().filter(|&&(e, g)| bare.contains(&(probe, e, g))).count();
        println!("{name}: {} drives at the atom, the smallest relative coupling among them {smallest:.3e}; the probe's π drives {} of {} at the atom", bare.len(), pi_at_atom, pi.len());
        counts.push((bare.len(), smallest, pi_at_atom, pi.len()));
    }
    let ((untilted, _, _, _), (tilted, smallest, pi_at_atom, pi)) = (counts[0], counts[1]);
    assert!(untilted > 0 && untilted <= 106, "untilted, at most the class routes' 106 drives at the atom: {untilted}");
    assert!(smallest > 1e-6, "tilted, no drive of a part in 10⁸ at the atom: {smallest}");
    assert!(pi > 0 && pi_at_atom > 0, "tilted, the probe's π drives at the atom: {pi_at_atom} of {pi}");
    assert!(tilted <= untilted + 72, "tilted, at most the 72 physical drives more: {tilted}");
}

/// The resolvent's bounds are continuous in K's departure from normality (`SchurBlock::henrici`; module documentation,
/// # First order in the frequency offsets): a ground state driven into two excited states at 50 and 80 rad/µs of decay
/// 38 and 6 rad/µs, joined by a coupling ε -- K = H_E − (i/2)D not normal where ε is not zero, its departure of the
/// order of ε. Criterion one's remainder at the offsets the moving atom spans (`Fold::offsets`, ±10 rad/µs) and the
/// series' ratio q (`Fold::series_distance`) at ε = 10⁻¹², 10⁻⁹, 10⁻⁶ rad/µs are within 10 ε/30 rad/µs (relative)
/// of theirs at ε = 0 -- as the eigenvectors move --, where a judgement of K normal or not collapsed the distance to a
/// rectangle's at a rounding's ε.
#[test]
fn the_resolvent_s_bounds_are_continuous_in_the_folded_block_s_departure_from_normality() {
    let read = |epsilon: f64| -> (f64, f64) {
        let mut excited = Mat::diagonal(&[C64::real(50.0), C64::real(80.0)]);
        excited[(0, 1)] = C64::real(epsilon);
        excited[(1, 0)] = C64::real(epsilon);
        let levels = Levels { ground: Mat::zeros(1, 1), excited, decay: Mat::diagonal(&[C64::real(38.0), C64::real(6.0)]) };
        let classes = vec![class(0.0, 1, 2, &[(0, 0, C64::real(5.0)), (1, 0, C64::real(3.0))])];
        let f = folded(&levels, &classes, &at_rest(1));
        assert!(f.exists);
        let rho: f64 = f.offsets(0, 10.0, &[], 1.0, &|_| 1.0).iter().map(|x| x * x).sum::<f64>().sqrt();
        (rho, f.series_ratio(C64::real(f.eigen.values[0].re)))
    };
    let (rho0, q0) = read(0.0);
    println!("ε 0: ρ {rho0:.6e}, q {q0:.6e}");
    for epsilon in [1e-12, 1e-9, 1e-6] {
        let (rho, q) = read(epsilon);
        let bound = 10.0 * epsilon / 30.0 + 1e-13;
        println!("ε {epsilon:e}: ρ moved {:.3e}, q moved {:.3e} (relative; bound {bound:.3e})", (rho / rho0 - 1.0).abs(), (q / q0 - 1.0).abs());
        assert!((rho / rho0 - 1.0).abs() <= bound, "ε {epsilon}: the remainder moved from {rho0} to {rho}");
        assert!((q / q0 - 1.0).abs() <= bound, "ε {epsilon}: the series' ratio moved from {q0} to {q}");
    }
}

/// Criterion three against the exact master equation (unify part 2): a Λ of |1⟩ at 0 and |2⟩ at 100 rad/µs, an excited
/// state decaying at Γ = 38 rad/µs, three quarters to |1⟩ and a quarter to |2⟩ (on two axes), laser A on |1⟩
/// (3 rad/µs) and B on |2⟩ (1.5 rad/µs), both on resonance -- where the jumps' coherence, not a coherent Raman coupling,
/// makes the dark state (coherent population trapping), and criterion three reads it before criterion two would --,
/// and the fold's frames 0 and 37: the two ground states in one group, the frames following neither the lasers'
/// difference nor the levels. Its photons landing in |1⟩ are two classes, A's from |1⟩ and B's from |2⟩, their
/// frequencies' difference Ω = 63 − δ; their own photons' difference is the Raman detuning δ.
/// - At the Raman resonance, the dark state b|1⟩ − a|2⟩: `start_cross_terms` keeps the pair, where γ/|Ω| -- the
///   readings' old number -- is within the share; and the effective master equation of the two ground states with each
///   class's jumps apart (the class split, its decay too) is beyond the share of the exact three-level one at 30 µs
///   (the dark state's |1⟩ 0.2 against the rate equations' 0.43), as with both classes in one jump it is within the
///   elimination's (a/|Δ + iΓ/2|)².
/// - 20 rad/µs off it: the pair is not kept, and the split is within the share of the exact one.
#[test]
fn a_raman_resonant_photon_coherence_is_kept_and_a_far_one_dropped_as_the_master_equation_says() {
    let (e2, gamma, delta, a, b, beta) = (100.0, 38.0, 0.0, 3.0, 1.5, 37.0);
    let (to1, to2) = (0.75 * gamma, 0.25 * gamma);
    let levels = Levels { ground: Mat::diagonal(&[C64::ZERO, C64::real(e2)]), excited: Mat::zeros(1, 1), decay: Mat::diagonal(&[C64::real(gamma)]) };
    let mut emission: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(3, 3));
    emission[0][(0, 2)] = C64::real(to1.sqrt());
    emission[1][(1, 2)] = C64::real(to2.sqrt());
    let denominator = C64::new(delta, 0.5 * gamma);
    let shift = |x: f64| x * x * delta / denominator.norm_sqr();
    // The Lindblad generator's action on ρ, and ρ at T from |1⟩⟨1| (RK4).
    let lindblad = |h: &Mat, jumps: &[Mat], rho: &Mat| -> Mat {
        let mut out = h.product(rho);
        out.add_scaled(-C64::ONE, &rho.product(h));
        let mut out = out.scaled(C64::new(0.0, -1.0));
        for j in jumps {
            out.add_scaled(C64::ONE, &j.product(rho).product(&j.adjoint()));
            let jj = j.adjoint_product(j);
            out.add_scaled(C64::real(-0.5), &jj.product(rho));
            out.add_scaled(C64::real(-0.5), &rho.product(&jj));
        }
        out
    };
    let evolve = |h: &Mat, jumps: &[Mat], time: f64, dt: f64| -> Mat {
        let n = h.rows;
        let mut rho = Mat::zeros(n, n);
        rho[(0, 0)] = C64::ONE;
        let steps = (time / dt).round() as usize;
        for _ in 0..steps {
            let k1 = lindblad(h, jumps, &rho);
            let mut x = rho.clone();
            x.add_scaled(C64::real(0.5 * dt), &k1);
            let k2 = lindblad(h, jumps, &x);
            let mut x = rho.clone();
            x.add_scaled(C64::real(0.5 * dt), &k2);
            let k3 = lindblad(h, jumps, &x);
            let mut x = rho.clone();
            x.add_scaled(C64::real(dt), &k3);
            let k4 = lindblad(h, jumps, &x);
            for (k, w) in [(&k1, 1.0), (&k2, 2.0), (&k3, 2.0), (&k4, 1.0)] {
                rho.add_scaled(C64::real(dt * w / 6.0), k);
            }
        }
        rho
    };
    let share = share();
    for (name, detuning) in [("resonant", 0.0), ("20 rad/us off", 20.0)] {
        // B's two-photon detuning dx, so that the light-shifted Raman detuning is `detuning` (on resonance no shift).
        let dx = detuning + shift(a) - shift(b);
        // The fold, and its cross terms at the start.
        let classes = vec![class(delta, 2, 1, &[(0, 0, C64::real(a))]), class(delta - e2 + dx, 2, 1, &[(0, 1, C64::real(b))])];
        let slow = Slow { explicit: vec![], frames: vec![0.0, beta], groups: vec![0, 0] };
        let f = fold(&levels, &classes, &slow, &Replicas::of(&classes, &slow, 2));
        assert!(f.exists, "{name}: the fold exists");
        let pairs = start_cross_terms(&f, &emission);
        let raman = |pair: &[(Option<usize>, usize, usize); 2]| pair.contains(&(Some(0), 0, 0)) && pair.contains(&(Some(1), 1, 0));
        let kept = pairs.iter().any(raman);
        // The old number at the equal superposition: the two classes' interference rate over Ω.
        let like: Vec<usize> = (0..f.size()).collect();
        let (images, origins) = f.photons(&emission, &like);
        let find = |o: (PhotonSource, usize)| origins.iter().position(|x| *x == o).expect("the class");
        let (ia, ib) = (find((PhotonSource::Class(0, 0), 0)), find((PhotonSource::Class(1, 1), 0)));
        let psi: Vec<C64> = (0..2).map(|k| f.eigen.vectors[(k, 0)] + f.eigen.vectors[(k, 1)]).collect();
        let norm = crate::linalg::norm_sqr(&psi);
        let sum = |i: usize, x: usize| -> Vec<C64> { (0..2).map(|k| images[i].images[0][x][k] + images[i].images[1][x][k]).collect() };
        let gamma_old = (0..3).map(|x| crate::linalg::inner(&sum(ia, x), &sum(ib, x))).sum::<C64>().abs() / norm;
        let omega = images[ia].frequency - images[ib].frequency;
        let old = gamma_old / omega.abs();
        let landings = vec![Landing::of(&f, &[0, 1])];
        let read = cross_terms(&images, &landings, &f.eigen.values, &[C64::ONE, C64::ONE], norm).into_iter().find(|t| (t.0, t.1) == (ia.min(ib), ia.max(ib))).map_or(0.0, |t| t.2);
        // The exact three-level master equation in the lasers' frame, and the two ground states' with both classes in
        // one jump (the elimination's) and with each class apart (the split's).
        let h3 = Mat::from_fn(3, 3, |i, j| match (i, j) {
            (1, 1) => C64::real(dx),
            (2, 2) => C64::real(-delta),
            (2, 0) | (0, 2) => C64::real(a),
            (2, 1) | (1, 2) => C64::real(b),
            _ => C64::ZERO,
        });
        let c3 = [Mat::from_fn(3, 3, |i, j| if (i, j) == (0, 2) { C64::real(to1.sqrt()) } else { C64::ZERO }), Mat::from_fn(3, 3, |i, j| if (i, j) == (1, 2) { C64::real(to2.sqrt()) } else { C64::ZERO })];
        let exact = evolve(&h3, &c3, 30.0, 2e-4);
        // The excited amplitude each ground state drives at its own frequency, r_j = w_j/(Δ_j + iΓ/2) (Δ_2 = Δ + dx); the
        // second order's Hermitian part (M + M†)/2, M_ij = w_i r_j; the jumps √Γ_k |k⟩⟨r|, split by the ground state
        // each class drives.
        let w = [C64::real(a), C64::real(b)];
        let r = [w[0] / denominator, w[1] / (denominator + C64::real(dx))];
        let mut h2 = Mat::from_fn(2, 2, |i, j| if (i, j) == (1, 1) { C64::real(dx) } else { C64::ZERO });
        h2.add_scaled(C64::ONE, &Mat::from_fn(2, 2, |i, j| (w[i] * r[j] + (w[j] * r[i]).conj()) * C64::real(0.5)));
        let rate = [to1, to2];
        let jump = |to: usize, from: Option<usize>| Mat::from_fn(2, 2, |i, j| if i == to && from.is_none_or(|f| f == j) { C64::real(rate[to].sqrt()) * r[j] } else { C64::ZERO });
        let joined = evolve(&h2, &[jump(0, None), jump(1, None)], 30.0, 1e-3);
        let split = evolve(&h2, &[jump(0, Some(0)), jump(0, Some(1)), jump(1, Some(0)), jump(1, Some(1))], 30.0, 1e-3);
        let (pe, pj, ps) = (exact[(0, 0)].re / (exact[(0, 0)].re + exact[(1, 1)].re), joined[(0, 0)].re, split[(0, 0)].re);
        println!("{name}: |Ω| {:.3}, γ/|Ω| {old:.3e}, the cross term's reading {read:.3e} (kept: {kept}); |1>'s population at 30 us: exact {pe:.4}, one jump {pj:.4}, split {ps:.4}", omega.abs());
        assert!((pj - pe).abs() <= 4.0 * (a / denominator.abs()).powi(2), "{name}: the elimination is the exact one's within (a/|Δ + iΓ/2|)²: {pj} against {pe}");
        if detuning == 0.0 {
            assert!(kept, "{name}: the Raman-resonant cross term is kept");
            assert!(old < share, "{name}: where the old number would drop it ({old})");
            assert!((ps - pe).abs() > share, "{name}: the split is beyond the share of the exact: {ps} against {pe}");
        } else {
            assert!(!kept, "{name}: the far cross term is not kept ({read})");
            assert!((ps - pe).abs() < share, "{name}: the split is within the share of the exact: {ps} against {pe}");
        }
    }
}

/// Criterion three's frequency at the prior's atoms (unify part 2; a measurement): on R6_RAMAN_CO, R6_RAMAN_COUNTER,
/// GMD1C (its request at the path in the environment variable GMD1C), the zero-field probe (S2), PGC1D_TILT_0 and
/// PGC1D_TILT_2, the closed model's coherence units not kept, read at every prior atom on the states of every source
/// cluster's spectrum (`Model::cluster_spectrum`, as the readings read them): each ground-like eigenstate r_p, and each
/// pair's superposition r_p + r_p′. The old number, the two classes' interference rate γ = |Σ_a ⟨u_a|u′_a⟩|/‖ψ‖² over
/// their frequencies' difference Ω, against `cross_terms` -- the dropped term's parts on the landing's and the source's
/// eigenstates, each over its combined frequency ω_c -- and the same parts over Ω (the frequency's change apart from
/// the size's). A unit is kept where its reading exceeds the share; the counts of reads kept by each and where the old
/// and the new differ.
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn the_coherences_old_and_combined_frequencies_at_the_prior_s_atoms() {
    use super::super::{Model, Point, Unit};
    use crate::linalg::norm_sqr;
    use crate::request::GROUND;
    let base = env!("CARGO_MANIFEST_DIR");
    let mut scenes: Vec<(String, String)> = vec![
        ("R6_RAMAN_CO".into(), format!("{base}/refs/scenes/R6_RAMAN_CO.json")),
        ("R6_RAMAN_COUNTER".into(), format!("{base}/refs/scenes/R6_RAMAN_COUNTER.json")),
        ("S2 (zero-field probe)".into(), format!("{base}/refs/requests/S2.json")),
        ("PGC1D_TILT_0".into(), format!("{base}/refs/requests/PGC1D_TILT_0.json")),
        ("PGC1D_TILT_2".into(), format!("{base}/refs/requests/PGC1D_TILT_2.json")),
    ];
    if let Ok(path) = std::env::var("GMD1C") {
        scenes.insert(2, ("GMD1C".into(), path));
    }
    let share = share();
    let dot = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(x, y)| x.conj() * *y).sum::<C64>();
    for (name, path) in scenes {
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).expect("the request file")).expect("JSON");
        let req = crate::request::Request::from_native(&line).expect("a request");
        let light = crate::field::Light::new(&req);
        let chief = (0..GROUND).max_by(|&a, &b| req.initial_density[(a, a)].re.total_cmp(&req.initial_density[(b, b)].re)).unwrap();
        let (model, kept, _, _) = Model::closed_at(&req, &light, vec![], chief, req.temperature_uk);
        let points = crate::solve::atoms(&model, &req, &light, &crate::solve::components(&req), req.temperature_uk).expect("the atoms");
        let hm = crate::units::hbar_over_mass(req.mass_kg);
        let running = model.running();
        let coherence = |classes: &[crate::model::photon::Class], a: usize, b: usize| {
            let (x, y) = (&classes[a].beats[0], &classes[b].beats[0]);
            let (lo, hi) = if x < y { (x.clone(), y.clone()) } else { (y.clone(), x.clone()) };
            Unit::Coherence { landing: model.clusters[classes[a].landing].states[0], first: lo, second: hi }
        };
        // reads; kept by the old, by the parts at Ω, by the new; old ≠ new, of them on a pair's superposition; the
        // largest of each reading; the least |ω_c|/|Ω| is not read here (`cross_terms` sums over it).
        let (mut reads, mut old_kept, mut parts_kept, mut new_kept, mut differ, mut differ_pairs) = (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut most_old, mut most_parts, mut most_new) = (0.0_f64, 0.0_f64, 0.0_f64);
        let mut examples: Vec<String> = vec![];
        let mut units_kept: Vec<Unit> = vec![];
        for &(r, v) in &points {
            let amplitudes = light.amplitudes(r);
            let at = Point { r, kappa: v.map(|x| x / hm), t: 0.0 };
            for s in 0..model.clusters.len() {
                let classes = &model.classes[s];
                if classes.len() < 2 {
                    continue;
                }
                let source = model.cluster_spectrum(s, &amplitudes);
                let plan = model.plan(s, &running);
                let n = source.size();
                let like: Vec<usize> = (0..n).filter(|&p| source.ground_weight(p) >= 0.5).collect();
                let mut landed: Vec<usize> = vec![];
                for class in classes {
                    if !landed.contains(&class.landing) {
                        landed.push(class.landing);
                    }
                }
                let landings: Vec<Landing> = landed
                    .iter()
                    .map(|&l| {
                        let fold = if l == s { source.clone() } else { model.cluster_spectrum(l, &amplitudes) };
                        Landing::of(&fold, &(0..fold.size()).collect::<Vec<_>>())
                    })
                    .collect();
                let mut images: Vec<Images> = classes
                    .iter()
                    .map(|class| {
                        let m = model.clusters[class.landing].states.len();
                        Images { frequency: -model.omega(&class.beats[0]), landing: landed.iter().position(|&l| l == class.landing).unwrap(), images: vec![std::array::from_fn(|_| vec![C64::ZERO; m]); n] }
                    })
                    .collect();
                for &p in &like {
                    let (photons, _) = model.photons_of(&plan, None, &at, &source.right(p), super::super::photon::Gradients::None);
                    for photon in photons {
                        images[photon.class].images[p] = photon.amplitudes;
                    }
                }
                let mut states: Vec<Vec<usize>> = like.iter().map(|&p| vec![p]).collect();
                for (i, &p) in like.iter().enumerate() {
                    for &p2 in &like[i + 1..] {
                        states.push(vec![p, p2]);
                    }
                }
                for state in &states {
                    let mut coefficients = vec![C64::ZERO; n];
                    state.iter().for_each(|&p| coefficients[p] = C64::ONE);
                    let psi: Vec<C64> = (0..n).map(|k| state.iter().map(|&p| source.eigen.vectors[(k, p)]).sum()).collect();
                    let norm = norm_sqr(&psi);
                    // The same parts over Ω: every eigenvalue at zero.
                    let flat: Vec<Landing> = landings.iter().map(|l| Landing { values: vec![C64::ZERO; l.values.len()], left: l.left.clone() }).collect();
                    let at_omega = cross_terms(&images, &flat, &vec![C64::ZERO; n], &coefficients, norm);
                    for (a, b, x) in cross_terms(&images, &landings, &source.eigen.values, &coefficients, norm) {
                        let unit = coherence(classes, a, b);
                        if kept.contains(&unit) {
                            continue;
                        }
                        let summed = |i: usize, axis: usize| -> Vec<C64> { (0..images[i].images[0][axis].len()).map(|k| state.iter().map(|&p| images[i].images[p][axis][k]).sum()).collect() };
                        let omega = images[a].frequency - images[b].frequency;
                        let old = (0..3).map(|axis| dot(&summed(a, axis), &summed(b, axis))).sum::<C64>().abs() / norm / omega.abs();
                        let parts = at_omega.iter().find(|t| (t.0, t.1) == (a, b)).map_or(0.0, |t| t.2);
                        reads += 1;
                        old_kept += usize::from(old > share);
                        parts_kept += usize::from(parts > share);
                        new_kept += usize::from(x > share);
                        (most_old, most_parts, most_new) = (most_old.max(old), most_parts.max(parts), most_new.max(x));
                        if x > share && !units_kept.contains(&unit) {
                            units_kept.push(unit.clone());
                        }
                        if (old > share) != (x > share) {
                            differ += 1;
                            differ_pairs += usize::from(state.len() == 2);
                            if examples.len() < 4 {
                                examples.push(format!("{unit:?} at {r:?}, eigenstates {state:?}: γ/|Ω| {old:.3e}, ΣM/|Ω| {parts:.3e}, ΣM/|ω_c| {x:.3e}, |Ω| {:.4e}", omega.abs()));
                            }
                        }
                    }
                }
            }
        }
        println!("{name}: {} coherence units kept by the prior; {reads} reads at the share {share:.4}: kept old (γ/|Ω|) {old_kept}, parts at Ω {parts_kept}, new (ΣM/|ω_c|) {new_kept}; old ≠ new {differ} ({differ_pairs} on a pair's superposition); {} units the new keeps", kept.iter().filter(|u| matches!(u, Unit::Coherence { .. })).count(), units_kept.len());
        println!("  largest: γ/|Ω| {most_old:.3e}, ΣM/|Ω| {most_parts:.3e}, ΣM/|ω_c| {most_new:.3e}");
        for e in &examples {
            println!("  differs: {e}");
        }
    }
}
