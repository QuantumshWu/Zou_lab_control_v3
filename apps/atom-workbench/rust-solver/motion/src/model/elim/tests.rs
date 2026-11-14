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
    Levels { ground: Mat::diagonal(&ground.iter().map(|&x| C64::real(x)).collect::<Vec<_>>()), excited: Mat::diagonal(&excited.iter().map(|&x| C64::real(x)).collect::<Vec<_>>()), decay, emission }
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
    Slow { explicit: vec![], frames: vec![0.0; n] }
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
                println!("Δ {detuning} Γ, s {s}: no fold (iterations {})", fold.iterations);
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
            let x = fold.landed(&[C64::ONE]).unwrap();
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
            // The transient's closed form (`Fold::transient`) against the exact one, and the error accounts' term
            // (`errors`) against the exact rate's error: to the leading order in the fold, its relative O(X), X = s/2.
            let left = fold.left_operators().unwrap();
            let (_, tau) = fold.transient(&left, &[C64::ONE]).unwrap();
            let booked = errors(&fold, &lv, &[vec![0]], &|_| false, Some(&|_| false), &[C64::ONE], [0.0; 3], 0.0).transient;
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
        let slow = Slow { explicit: vec![], frames: vec![0.0, nu1 - nu2] };
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
        let exact = folded(&lv, &classes, &Slow { explicit: vec![], frames: vec![0.0, nu1 - nu2] });
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

/// The frames (the coordinator's 2026-11-13 rule): a chain of components is static in its optimal frames (residual zero
/// on every link), and a three-state ring is left its cycle's detuning over three on every link -- the linear
/// program's optimum, which no frames beat (each link's residual is the ring's mean).
#[test]
fn the_frames_leave_a_chain_static_and_a_ring_its_mean_cycle_detuning() {
    let chain = [(1, 0, 26.37), (2, 1, 26.41), (3, 2, -3.2)];
    let (theta, residual) = frames(5, &chain);
    assert_eq!(residual, 0.0);
    for &(a, b, omega) in &chain {
        assert!((omega + theta[b] - theta[a]).abs() <= 1e-12, "link {a} ← {b}");
    }
    let ring = [(1, 0, 10.0), (2, 1, 4.0), (0, 2, -11.0)];
    let (theta, residual) = frames(3, &ring);
    let mean: f64 = (10.0 + 4.0 - 11.0) / 3.0;
    assert!((residual - mean.abs()).abs() <= 1e-12, "residual {residual} against {mean}");
    for &(a, b, omega) in &ring {
        assert!(((omega + theta[b] - theta[a]).abs() - mean.abs()).abs() <= 1e-12, "link {a} ← {b}: {}", omega + theta[b] - theta[a]);
    }
    // Two links of one pair at two beats: a two-cycle, half their difference.
    let (_, residual) = frames(2, &[(1, 0, 0.0), (1, 0, 26.37)]);
    assert!((residual - 13.185).abs() <= 1e-12);
}

/// A Raman resonance across a Zeeman splitting (the coordinator's test): two ground states 2ω_L apart, coupled by two
/// classes 2ω_L + δ apart through a far excited state. The lab beat is 26.37 rad/µs; the component's distance on the
/// actual spectrum is δ (and the light shifts' difference); in the optimal frames it is static, the step's
/// requirement its residual, zero -- and the fold in those frames is the exact static problem's (its slow eigenvalues
/// against the exact three-state generator's).
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
    for x in &explicit {
        assert!((x.beat.abs() - (zeeman + detuning)).abs() <= 1e-9, "its lab beat");
        assert!(x.distance.abs() < 0.2 * zeeman, "its distance on the actual spectrum {:?}", x.distance);
    }
    let links: Vec<(usize, usize, f64)> = explicit.iter().map(|x| (rest.dominant(x.row), rest.dominant(x.column), x.beat)).collect();
    let (theta, residual) = frames(2, &links);
    assert_eq!(residual, 0.0, "a chain: static in its frames");
    let fold = folded(&lv, &classes, &Slow { explicit: vec![], frames: theta.clone() });
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

/// The brief's test 5, the tilt's property: the 1D PGC at 3 G with its π probe tilted by θ = 0, 0.1°, 2° (the user's
/// failure #4 is 2°; refs/requests/PGC1D_TILT_*, made by the app's own request builder). The tilt adds couplings
/// ∝ sin θ: the remainder ρ and the components' mixings move by O(θ) at most, continuously, and the partition, the
/// kept links and the frames' residual -- what the step's requirement is made of -- are the same at every θ; the
/// requirement itself moves only by the probe's wave vector's turn, O(θ²) of |K| v. No count of a coupling that
/// "exists" or not: a tilt's small couplings are averaged like any other.
#[test]
fn a_tilt_leaves_the_partition_and_the_step_s_requirement_and_moves_the_numbers_continuously() {
    let names = ["PGC1D_TILT_0", "PGC1D_TILT_01", "PGC1D_TILT_2"];
    let requests: Vec<crate::request::Request> = names.iter().map(|n| request(n)).collect();
    let speed = 0.17;
    let points: Vec<([f64; 3], [f64; 3])> = vec![([0.0; 3], [0.0; 3]), ([0.0; 3], [speed, 0.0, 0.0]), ([0.0; 3], [0.0, speed, 0.0]), ([0.0; 3], [0.0, 0.0, speed]), ([0.05, 0.1, 0.0], [0.0, 0.0, -speed])];
    for (r, v) in points {
        let closed: Vec<Closed> = requests
            .iter()
            .map(|req| {
                let light = crate::field::Light::new(req);
                let (lv, cl, _) = system(req, &light, r);
                closure(&lv, &cl, &blocks(req), v, &Kept::default())
            })
            .collect();
        for (name, c) in names.iter().zip(&closed) {
            println!(
                "{name} r {r:?} v {v:?}: blocks {:?}, links {}, photons {}, residual {:.3e}, requirement {:.4} rad/us, rho {:.4e}, averaged mixing {:.4e}, fold iterations {}",
                c.kept.blocks,
                c.kept.links.len(),
                c.kept.photons.len(),
                c.residual,
                c.requirement,
                c.remainder,
                c.mixing,
                c.fold.iterations
            );
        }
        let base = &closed[0];
        for c in &closed[1..] {
            assert_eq!(c.kept.blocks, base.kept.blocks, "the partition at {r:?} {v:?}");
            assert_eq!(c.kept.links.len(), base.kept.links.len(), "the kept links at {r:?} {v:?}");
            assert!((c.residual - base.residual).abs() <= 1e-9 * (1.0 + base.residual), "the frames' residual");
            assert!((c.requirement - base.requirement).abs() <= 1e-2 * base.requirement.max(1.0), "the step's requirement {} against {}", c.requirement, base.requirement);
        }
        // Continuity: at 0.1° the numbers are the untilted ones to O(θ).
        let theta = 0.1_f64.to_radians();
        assert!((closed[1].remainder - base.remainder).abs() <= 10.0 * theta * base.remainder.max(1e-6), "ρ at 0.1°: {} against {}", closed[1].remainder, base.remainder);
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
    let lv = Levels { ground: levels.ground.block(&ground, &ground), excited: levels.excited.clone(), decay: levels.decay.clone(), emission: std::array::from_fn(|a| levels.emission[a].block(&ground, &excited)) };
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
        let lv = Levels { ground: levels.ground.block(&ground, &ground), excited: levels.excited.clone(), decay: levels.decay.clone(), emission: std::array::from_fn(|a| levels.emission[a].block(&ground, &excited)) };
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

/// The run's route against the definition (ELIMINATION v5; assemble.rs, # The normalized generator): on PGC3D's F = 2
/// states at the cooling light's coherent peak near the focus, the normalized generator the engine makes of the series'
/// forms about z₀ -- N₀ = Σ W′R₀W and N₁ = Σ W′R₀²W, the metric X = Σ (R₀W)†(R₀W), the loss L = Σ (R₀W)†D(R₀W), at the
/// order the ratio asks (`Series::order_of`), through `assemble::normalized` -- against the self-consistent fold's
/// S H S⁻¹, S = (1 + X)^{1/2} (`Fold::normalization`): its Hermitian part within the series' remainder and the
/// normalization's own next order, and the landing map S(1 + Y)⁻¹ against the fold's S(1 + Σ S_f T_f)⁻¹. Measured
/// below; the bound is the remainder q^{K+1}/(1 − q) of the forms' part of the generator, beside the loss form's
/// difference from the anti-Hermitian part of W′R₀W (zero where the excited block is normal with one linewidth).
#[test]
fn the_run_s_normalized_generator_is_the_fold_s_within_the_series_remainder() {
  for name in ["PGC3D_1ms", "PGC1D_TILT_2"] {
    let req = request(name);
    let light = crate::field::Light::new(&req);
    let ground: Vec<usize> = (0..crate::request::GROUND).filter(|&g| crate::request::Level::parse(&req.states[g]).unwrap().f == 2).collect();
    let n = ground.len();
    for r in [[0.15, 0.0, 0.15], [0.05, -0.1, 0.0], [0.0; 3]] {
        let (levels, classes, _) = system(&req, &light, r);
        let excited: Vec<usize> = (0..levels.excited.rows).collect();
        let lv = Levels { ground: levels.ground.block(&ground, &ground), excited: levels.excited.clone(), decay: levels.decay.clone(), emission: std::array::from_fn(|a| levels.emission[a].block(&ground, &excited)) };
        let cl: Vec<Class> = classes.iter().map(|c| Class { nu: c.nu, beams: c.beams.iter().map(|b| Beam { coupling: b.coupling.block(&excited, &ground), wave: b.wave }).collect() }).collect();
        let fold = folded(&lv, &cl, &at_rest(n));
        assert!(fold.exists);
        let (s, s_inverse) = fold.normalization();
        let definition = s.product(&fold.generator).product(&s_inverse);
        // The reference: the far levels' mean and the point's scalar shift's centre (here the point's own).
        let far = (0..n).map(|i| fold.direct[(i, i)].re).sum::<f64>() / n as f64;
        let z0 = fold.eigen.values.iter().map(|z| z.re).sum::<f64>() / n as f64;
        let q = fold.series_ratio(C64::real(z0));
        let order = Series::order_of(q).expect("the series converges here");
        // The forms (every class, every replica, as the engine's per carrier and basis), from the reference's responses.
        let nus: Vec<f64> = cl.iter().map(|c| c.nu).collect();
        let reference = Reference::new(&lv, &nus, &at_rest(n), z0);
        let (mut n0, mut n1, mut x, mut l) = (Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n));
        for (c, class) in cl.iter().enumerate() {
            let mut w = Mat::zeros(excited.len(), n);
            for b in &class.beams {
                w.add_scaled(C64::ONE, &b.coupling);
            }
            let t = reference.respond(c, &w, n, 1);
            n0.add_scaled(C64::ONE, &w.adjoint_product(&t[0]));
            n1.add_scaled(C64::ONE, &w.adjoint_product(&t[1]));
            x.add_scaled(C64::ONE, &t[0].adjoint_product(&t[0]));
            l.add_scaled(C64::ONE, &t[0].adjoint_product(&lv.decay.product(&t[0])));
        }
        let mut whole = fold.direct.clone();
        whole.add_scaled(C64::ONE, &n0);
        let hermitian = whole.hermitian_part();
        let four = |m: Mat| -> [Mat; 4] { [m, Mat::zeros(n, n), Mat::zeros(n, n), Mat::zeros(n, n)] };
        let mut data = vec![C64::ZERO; 14 * n * n];
        crate::model::assemble::normalized(&four(hermitian.clone()), &four(n1.clone()), &four(x.clone()), &four(l.clone()), [z0, 0.0, 0.0, 0.0], order, &mut data);
        let held = Mat { rows: n, cols: n, data: data[crate::model::assemble::HELD * n * n..(crate::model::assemble::HELD + 1) * n * n].to_vec() };
        let mut route = hermitian.clone();
        route.add_scaled(C64::ONE, &held);
        let mut apart = route.clone();
        apart.add_scaled(-C64::ONE, &definition.hermitian_part());
        let mut own = definition.hermitian_part();
        own.add_scaled(-C64::ONE, &fold.direct.hermitian_part());
        let relative = apart.frobenius() / own.frobenius();
        // The normalization's own size against the fold's part, and the landing maps.
        let mut normalization = definition.hermitian_part();
        normalization.add_scaled(-C64::ONE, &fold.generator.hermitian_part());
        let landing = Mat { rows: n, cols: n, data: data[crate::model::assemble::LANDING * n * n..(crate::model::assemble::LANDING + 1) * n * n].to_vec() };
        let exact = fold.landing(&s).expect("the left operators settle");
        let mut landing_apart = landing.clone();
        landing_apart.add_scaled(-C64::ONE, &exact);
        let x_size = crate::linalg::hermitian_eigen(&x).values.iter().fold(0.0_f64, |m, &v| m.max(v));
        println!(
            "{name} F=2 at {r:?}: z0 - far {:.3} rad/us, q {q:.4}, order {order}, largest excited share {x_size:.4}; the normalization moves the fold's generator by {:.3e} of its part; the route's Hermitian generator {relative:.3e} of it apart; landing {:.3e} apart",
            z0 - far,
            normalization.frobenius() / own.frobenius(),
            landing_apart.frobenius() / exact.frobenius()
        );
        let bound = q.powi(order as i32 + 1) / (1.0 - q);
        assert!(relative <= bound.max(1e-9), "the route's generator {relative} apart, beyond the series' remainder {bound}");
        assert!(landing_apart.frobenius() <= (q + x_size * q).max(1e-9) * exact.frobenius(), "the landing map");
    }
  }
}
