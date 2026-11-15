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

/// The static fold: no sector (its turning couplings neither folded nor carried).
fn folded(levels: &Levels, classes: &[Class], slow: &Slow) -> Fold {
    let replicas = Replicas::of(classes, slow, levels.ground.rows);
    fold(levels, classes, slow, &replicas, None)
}

/// The fold with its sectors (# Sectors): every turning coupling folded, or (`carried`) every one carried.
fn sectored(levels: &Levels, classes: &[Class], slow: &Slow, carried: bool) -> Fold {
    let replicas = Replicas::of(classes, slow, levels.ground.rows);
    fold(levels, classes, slow, &replicas, Some(&|_: &Link| carried))
}

/// The P of one cluster of exact frames (each its own size): all its states one group.
fn single(explicit: Vec<usize>, frames: Vec<f64>) -> Slow {
    let (groups, sizes) = (vec![0; frames.len()], frames.iter().map(|f| f.abs()).collect());
    Slow { explicit, frames, sizes, groups }
}

fn at_rest(n: usize) -> Slow {
    single(vec![], vec![0.0; n])
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
            let booked: f64 = errors(&fold, &[vec![0]], &[C64::ONE], 0.0, &[]).parts.iter().map(|p| p.1).sum();
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
/// coupling is a sector's (# Sectors) of size |g1 g2|/|Δ| (to the order (g/Δ)²) at the beat ν1 − ν2.
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
        let slow = single(vec![], vec![0.0, nu1 - nu2]);
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
        let rest = sectored(&lv, &classes, &at_rest(2), true);
        assert_eq!(rest.replicas.len(), 2);
        let raman = rest.carried.iter().flat_map(|s| s.links.iter()).filter(|l| l.1 != l.2).fold(0.0, |m: f64, l| m.max(l.3.abs()));
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

/// A sector's self-consistent fold where its ladder closes is the exact problem (# Sectors; the brief's first analytic
/// test): two ground states of one group in frames θ = (0, 26.5) rad/µs, no light, a lab-static coupling a between them
/// (H_g = [[0, a], [a, d₀]]) that turns at −θ₂ in the frames. g₁(0) couples to g₂ at the sector −θ₂ alone, which couples
/// back to g₁(0) alone: the fold's eigenvalues are the lab's, λ₁ and λ₂ − θ₂, its sector's population on the first the
/// exact eigenvector's |x₂/x₁|², to the rounding (1e-10). The static fold has none of the shift.
#[test]
fn a_sector_s_self_consistent_fold_where_its_ladder_closes_is_the_exact_problem() {
    for &(a, d0) in &[(1.0, 8.0), (1.0, 3.0), (2.0, 5.0)] {
        let theta = 26.5;
        let lv = Levels { ground: Mat::from_fn(2, 2, |i, j| match (i, j) { (0, 1) | (1, 0) => C64::real(a), (1, 1) => C64::real(d0), _ => C64::ZERO }), excited: Mat::zeros(0, 0), decay: Mat::zeros(0, 0) };
        let slow = single(vec![], vec![0.0, theta]);
        let fold = sectored(&lv, &[], &slow, false);
        assert!(fold.exists, "a = {a}, d₀ = {d0}: the fold exists");
        let exact = eigen(&lv.ground).unwrap();
        let (low, high) = if exact.values[0].re < exact.values[1].re { (0, 1) } else { (1, 0) };
        let p = (0..2).find(|&p| fold.dominant(p) == 0).unwrap();
        let q = 1 - p;
        assert!((fold.eigen.values[p] - exact.values[low]).abs() <= 1e-10 * d0, "g₁: {:?} against {:?}", fold.eigen.values[p], exact.values[low]);
        assert!((fold.eigen.values[q] - (exact.values[high] - C64::real(theta))).abs() <= 1e-10 * d0, "g₂: {:?} against {:?}", fold.eigen.values[q], exact.values[high] - C64::real(theta));
        let ratio = (exact.vectors[(1, low)] / exact.vectors[(0, low)]).norm_sqr();
        let held = fold.sectors.iter().map(|s| crate::linalg::norm_sqr(&fold.sector_apply(s, &fold.right(p)))).sum::<f64>() / fold.right(p)[0].norm_sqr();
        assert!((held - ratio).abs() <= 1e-10 * ratio, "the sector's population {held} against |x₂/x₁|² {ratio}");
        let none = folded(&lv, &[], &slow);
        println!("a {a}, d₀ {d0}: λ₁ {:.6e} (exact {:.6e}; the static fold {:.1e}); |x₂/x₁|² {held:.6e}", fold.eigen.values[p].re, exact.values[low].re, none.eigen.values[(0..2).find(|&p| none.dominant(p) == 0).unwrap()].re);
    }
}

/// The Λ scheme of unify 4.7 (the brief's second analytic test): g₁ (0) and g₂ (δ₀) to one excited state e, beam 1 on g₁
/// alone, beam 2 on g₂ alone, their beat Ω = 26.5 rad/µs; in the frames following the beams the problem is static and the
/// fold its exact three-level one. In the frames at rest the Raman coupling a = g₁g₂/Δ turns at Ω, across
/// δ = Ω + λ₁ − λ₂ on the actual spectrum. The table's light shifts and coupling (U₁ = −1.172, a = −0.977 rad/µs at the
/// first point) at Δ = −3072 rad/µs (Γ = 38.1), and with the same U and a at 1000 times the detuning (g ∝ √Δ), where
/// the excited response does not change over δ -- 4.7's numbers, which took U and a as constants:
/// - folded as its sector (# Sectors), g₁'s eigenvalue is the exact one within the sector's own excited response, taken
///   at λ + Ω by the fold's left factor H_s and its V_{−Ω} at the source's frequency: relatively δ/Δ of the shift at
///   Δ = −3072 (criterion one's offset of the sector), far within (a/δ)² of it at 1000 times; its population
///   ‖T′r‖²/(1 + ‖T′r‖²) the exact eigenvector's;
/// - the second order −a²/δ (T′ ≈ R′V) errs on the exact shift by (a/δ)² relatively (1.35 % at 1.36 % at 1000 times);
/// - the fold without it -- what the averaged model was -- leaves out the shift, 9.6 % of g₁'s light shift at
///   a/δ = 0.117, where 2(a/δ)² = 2.7 % was booked; the readings' measure of it (criterion one on the run's fold, which
///   folds no sector, at g₁'s eigenstate: `errors`, `Fold::sector_parts`) is the shift over U₁ within the second
///   order's (a/δ)², beyond 2(a/δ)²;
/// - and the closure keeps the link at all three points (at the first the averaged rule's 2(a/δ)² left it out).
#[test]
fn the_lambda_scheme_s_sector_holds_the_static_shift_the_averaged_model_left_out() {
    let gamma = 38.1;
    for (scale, &(g1, g2, offset, missing)) in [1.0_f64, 1000.0].into_iter().flat_map(|x| [(60.0, 50.0, 8.0, 0.096), (60.0, 50.0, 3.0, 0.22), (90.0, 80.0, 6.0, 0.29)].iter().map(move |p| (x, p))) {
        let (delta, g1, g2) = (-3072.0 * scale, g1 * scale.sqrt(), g2 * scale.sqrt());
        let (nu1, nu2) = (delta, delta - 26.5);
        let d0 = (nu1 - nu2) + offset;
        let lv = levels(&[0.0, d0], &[0.0], gamma, &[0]);
        let classes = vec![class(nu1, 2, 1, &[(0, 0, C64::real(g1))]), class(nu2, 2, 1, &[(0, 1, C64::real(g2))])];
        let (rest, sector) = (folded(&lv, &classes, &at_rest(2)), sectored(&lv, &classes, &at_rest(2), false));
        let exact = folded(&lv, &classes, &single(vec![], vec![0.0, nu1 - nu2]));
        assert!(sector.exists && exact.exists);
        let of = |f: &Fold| (0..2).find(|&p| f.dominant(p) == 0).unwrap();
        let (pr, ps, pe) = (of(&rest), of(&sector), of(&exact));
        let (lr, ls, le) = (rest.eigen.values[pr], sector.eigen.values[ps], exact.eigen.values[pe]);
        let shift = le - lr;
        // The sector's coupling and distance on the actual spectrum.
        let s = sector.sectors.iter().find(|s| s.links.iter().any(|l| (l.1, l.2) == (1, 0))).expect("g₂ ← g₁'s sector");
        let a = s.v[(1, 0)];
        let distance = C64::real(s.frequency) + lr - rest.eigen.values[1 - pr];
        let mixing = a.abs() / distance.abs();
        let second = -(a * s.back[(0, 1)]) / -distance;
        let light = lr - C64::real(lv.ground[(0, 0)].re);
        println!(
            "Δ {delta}, g ({g1:.1}, {g2:.1}), δ₀ − Ω {offset}: a/δ {mixing:.4}, 2(a/δ)² {:.4}; shift {:.5} (of U₁ {:.2} %), the sector {:.3e} apart, the second order {:.3} % apart",
            2.0 * mixing * mixing,
            shift.re,
            100.0 * shift.abs() / light.abs(),
            (ls - le).abs() / shift.abs(),
            100.0 * ((second - shift).abs() / shift.abs())
        );
        assert!((shift.abs() / light.abs() - missing).abs() <= 0.01, "the shift the averaged model leaves out: {} of U₁", shift.abs() / light.abs());
        let offset_part = 2.0 * distance.abs() / delta.abs();
        assert!((ls - le).abs() <= (offset_part + 0.1 * mixing * mixing) * shift.abs(), "the sector's eigenvalue {ls:?} against the exact {le:?}");
        let relative = (second - shift).abs() / shift.abs();
        assert!(relative >= 0.8 * mixing * mixing - offset_part && relative <= 1.1 * mixing * mixing + offset_part, "the second order {relative} against (a/δ)² {}", mixing * mixing);
        // The micromotion's time mean: the sector's population against the exact eigenvector's g₂ part.
        let t = sector.sector_apply(s, &sector.right(ps));
        let held = crate::linalg::norm_sqr(&t) / sector.right(ps)[0].norm_sqr();
        let population = held / (1.0 + held);
        let v = exact.right(pe);
        let true_population = v[1].norm_sqr() / (v[0].norm_sqr() + v[1].norm_sqr());
        println!("    the micromotion's population: the sector {population:.6}, exact {true_population:.6}, (a/δ)² {:.6}", mixing * mixing);
        assert!((population - true_population).abs() <= (0.05 + offset_part) * true_population, "the sector's population {population} against {true_population}");
        // The readings' measure of what the run, which folds no sector, leaves out: criterion one on the run's fold about
        // z₀ at g₁'s bare level, its sector made at it (`errors`, `Fold::sector_parts`), at g₁'s eigenstate -- the
        // light's interference through the sector, a response's relative error (g₂/g₁)(a/δ), the sector's own
        // second-order shift over g₁'s light shift: the shift's relative error to the second order's (a/δ)²; in squares
        // with the population's part, the amplitude's error √(1 + 2‖T′r‖²) − 1 of the populations' 2⟨X′⟩.
        let run = fold_about(&lv, &classes, &at_rest(2), &Replicas::of(&classes, &at_rest(2), 2), Some(&[0.0, 0.0]), Some(&|_: &Link| false));
        let reading: f64 = errors(&run, &[vec![0]], &run.right(of(&run)), 0.0, &[]).links.iter().map(|l| l.1).sum();
        let r = amplitude(reading);
        let expected = (shift.abs() / lr.abs()).hypot(amplitude(2.0 * held));
        println!("    the reading {reading:.4e} (ρ {r:.4e}) against the shift over U₁ with the population's part {expected:.4e}; 2(a/δ)² {:.4e}", 2.0 * mixing * mixing);
        assert!(reading > 2.0 * mixing * mixing, "the reading {reading} within the averaged rule's 2(a/δ)²");
        assert!((r - expected).abs() <= (1.1 * mixing * mixing + offset_part) * expected, "the reading's ρ {r} against the shift's {expected}");
        // The closure keeps the coupling (criterion one on its sector).
        let closed = closure(&lv, &classes, &[vec![0]], [0.0; 3], &Kept::default());
        assert!(closed.kept.links.iter().any(|l| l.a < 2 && l.b < 2 && l.a != l.b), "the closure keeps the Raman coupling: {:?}", closed.kept);
    }
}

/// A sector's fold against the truncated Sambe problem where no frame stills the coupling (the brief's numerical
/// test): both ground states driven by both beams, so that two P states are coupled statically through each beam and at
/// the beat Ω = ν₁ − ν₂ through the two -- the ladder g(0) → g(Ω) → g(2Ω) … does not close. The exact quasienergies: in
/// the frame of e at ν₁, H(t) = H₀ + V₊e^{iΩt} + V₋e^{−iΩt} (V₊ = W₂ on e ← g, V₋ = W₂† on g ← e), its Sambe matrix
/// (H₀ − mΩ on the diagonal, V₊ from sector m + 1 and V₋ from m − 1) truncated at |m| ≤ M until converged. The first
/// harmonic's fold holds each state's quasienergy within (a/δ)² of its sector's shift (the second harmonic and the
/// sector's excited response at λ + Ω); the static fold misses the shift whole.
#[test]
fn a_sector_s_fold_is_the_truncated_sambe_problem_s_within_its_second_harmonic() {
    let (gamma, delta, omega) = (38.1, -3072.0, 26.5);
    let d0 = omega + 8.0;
    let (nu1, nu2) = (delta, delta - omega);
    let lv = levels(&[0.0, d0], &[0.0], gamma, &[0]);
    let (w1, w2) = ([60.0, 30.0], [25.0, 50.0]);
    let classes = vec![class(nu1, 2, 1, &[(0, 0, C64::real(w1[0])), (0, 1, C64::real(w1[1]))]), class(nu2, 2, 1, &[(0, 0, C64::real(w2[0])), (0, 1, C64::real(w2[1]))])];
    let (rest, sector) = (folded(&lv, &classes, &at_rest(2)), sectored(&lv, &classes, &at_rest(2), false));
    assert!(sector.exists);
    let sambe = |m_max: i64| -> Eigen {
        let sectors = (2 * m_max + 1) as usize;
        let n = 3 * sectors;
        let mut q = Mat::zeros(n, n);
        for (k, m) in (-m_max..=m_max).enumerate() {
            let at = 3 * k;
            q[(at + 1, at + 1)] = C64::real(d0);
            q[(at + 2, at + 2)] = C64::new(-nu1, -0.5 * gamma);
            for g in 0..2 {
                q[(at + 2, at + g)] = C64::real(w1[g]);
                q[(at + g, at + 2)] = C64::real(w1[g]);
            }
            for i in 0..3 {
                q[(at + i, at + i)] -= C64::real(m as f64 * omega);
            }
            if k + 1 < sectors {
                for g in 0..2 {
                    q[(at + 2, at + 3 + g)] = C64::real(w2[g]);
                    q[(at + 3 + g, at + 2)] = C64::real(w2[g]);
                }
            }
        }
        eigen(&q).unwrap()
    };
    // The quasienergy whose eigenvector lies most on (g, sector 0).
    let near = |e: &Eigen, m_max: i64, g: usize| -> C64 {
        let at = 3 * m_max as usize + g;
        let k = (0..e.values.len()).max_by(|&x, &y| {
            let weight = |k: usize| e.vectors[(at, k)].norm_sqr() / (0..e.values.len()).map(|i| e.vectors[(i, k)].norm_sqr()).sum::<f64>();
            weight(x).total_cmp(&weight(y))
        }).unwrap();
        e.values[k]
    };
    let (coarse, fine) = (sambe(6), sambe(8));
    for g in 0..2 {
        let (p, r) = ((0..2).find(|&p| sector.dominant(p) == g).unwrap(), (0..2).find(|&p| rest.dominant(p) == g).unwrap());
        let (exact, converged) = (near(&fine, 8, g), near(&coarse, 6, g));
        let (ls, lr) = (sector.eigen.values[p], rest.eigen.values[r]);
        let s = sector.sectors.iter().filter(|s| s.blocks.iter().any(|b| b.1 == sector.groups[p])).map(|s| {
            let t = sector.sector_apply(s, &sector.right(p));
            crate::linalg::norm_sqr(&t)
        }).sum::<f64>();
        let mixing2 = s / sector.right(p)[g].norm_sqr();
        println!("g{}: exact {:.8e} (M 6 against 8: {:.1e}); the sector's fold {:.3e} apart, the static fold {:.3e}; (a/δ)² {mixing2:.3e}", g + 1, exact.re, (exact - converged).abs(), (ls - exact).abs(), (lr - exact).abs());
        assert!((exact - converged).abs() <= 1e-3 * (lr - exact).abs(), "the Sambe truncation converged");
        assert!((ls - exact).abs() <= mixing2 * (lr - exact).abs(), "g{}: the sector's fold {ls:?} against {exact:?}, beyond (a/δ)² of the shift", g + 1);
    }
}

/// One object, one number (the finish brief's item 1). Criterion one's ρ is an amplitude's relative error: of the excited
/// response e = T r_p, wrong by δe with ‖δe‖ = ρ‖e‖, every rate it makes, Γ‖e‖², moves by |2 Re⟨e|δe⟩ + ‖δe‖²| ≤
/// (2ρ + ρ²)Γ‖e‖² -- attained where δe lies along e, as it does on one line --, and a shift ⟨W r_p|e⟩ by at most ρ; the
/// populations are the rates' ratios and T is set by the rates and the shifts' gradients, so the output's error is the
/// rate's 2ρ + ρ² (`rate`). The closure judged ρ against the share and the readings booked 2ρ: an object between the
/// two was let go by the prior and asked for by the readings. Now both take `rate` of one function's ρ
/// (`Fold::criterion_one`):
/// - a two-level atom (Δ = −3Γ, s ≈ 0.06) moving through its light along k: ρ = Δ_v²/(|z ± Δ_v − κ||z − κ|) at the offset
///   Δ_v = 2kv (z = λ + ν, κ = −iΓ/2; # First order in the frequency offsets), the larger sign, analytic; at its
///   eigenstate the readings book 2ρ + ρ² (`errors`, beside the transient; to 1e-12), and the closure makes the block explicit
///   exactly where that is beyond the share -- also where share/2 < ρ < share (√(1 + share) − 1 < ρ), where before it
///   did not;
/// - the Λ scheme of a sector at a far two-photon distance, its link not kept: at the closure's eigenstate the readings'
///   number of the link is the closure's own part of it, to the rounding.
#[test]
fn the_closure_judges_each_object_by_the_number_the_readings_book() {
    let (gamma, k): (f64, f64) = (38.1, 8.0);
    let nu = -3.0 * gamma;
    let g = (0.06 * (nu * nu + 0.25 * gamma * gamma) / 2.0).sqrt();
    let lv = levels(&[0.0], &[0.0], gamma, &[0]);
    let classes = vec![Class { nu, beams: vec![Beam { coupling: Mat::from_fn(1, 1, |_, _| C64::real(g)), wave: [k, 0.0, 0.0] }] }];
    let share = super::share();
    let (mut between, mut beyond, mut within) = (0, 0, 0);
    for step in 1..=40 {
        let v = 0.06 * step as f64;
        let pattern = 2.0 * k * v;
        let fold = folded(&lv, &classes, &at_rest(1));
        let lambda = fold.eigen.values[0];
        let distance = |d: f64| (lambda + C64::new(nu + d, 0.5 * gamma)).abs();
        let rho = [pattern, -pattern].iter().map(|&d| d * d / (distance(d) * distance(0.0))).fold(0.0_f64, f64::max);
        // The readings' block part less the transient's (its part at rest, the offsets' none).
        let on = |pattern: f64| errors(&fold, &[vec![0]], &[C64::ONE], pattern, &[]).parts.iter().map(|x| x.1).sum::<f64>();
        let read = on(pattern) - on(0.0);
        assert!((read - rate(rho)).abs() <= 1e-12 * rate(rho), "v {v}: the readings' {read} against 2ρ + ρ² {}", rate(rho));
        let (total, _) = fold.criterion_one(&[vec![0]], 0, pattern, 1.0, &|_| 1.0);
        assert!((total.sqrt() - rho).abs() <= 1e-12 * rho, "v {v}: the closure's ρ {} against {rho}", total.sqrt());
        let kept = !closure(&lv, &classes, &[vec![0]], [v, 0.0, 0.0], &Kept::default()).kept.blocks.is_empty();
        assert_eq!(kept, rate(rho) > share, "v {v}: ρ {rho}, 2ρ + ρ² {}, the share {share}", rate(rho));
        if amplitude(share) < rho && rho < share {
            between += 1;
        }
        (beyond, within) = (beyond + usize::from(kept), within + usize::from(!kept));
    }
    println!("two levels: {within} velocities within, {beyond} beyond, of them {between} between share/2 and the share");
    assert!(between > 0 && within > 0, "the scan crosses the share, through the window the factor two left");
    // The Λ scheme, its sector 300 rad/µs off the two-photon resonance: the link folded and not kept.
    let (delta, g1, g2) = (-3072.0, 60.0, 50.0);
    let (nu1, nu2) = (delta, delta - 26.5);
    let lv = levels(&[0.0, (nu1 - nu2) + 300.0], &[0.0], gamma, &[0]);
    let classes = vec![class(nu1, 2, 1, &[(0, 0, C64::real(g1))]), class(nu2, 2, 1, &[(0, 1, C64::real(g2))])];
    let closed = closure(&lv, &classes, &[vec![0]], [0.0; 3], &Kept::default());
    assert!(closed.kept.links.is_empty() && !closed.fold.sectors.is_empty(), "the sector folded, its link not kept: {:?}", closed.kept);
    let fold = &closed.fold;
    for p in (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5) {
        let (total, parts) = fold.criterion_one(&[vec![0]], p, 0.0, 1.0, &|_| 1.0);
        let mut prior: Vec<(Link, f64)> = vec![];
        for (l, x) in split(total, parts).filter_map(|(o, x)| if let Object::Link(l) = o { Some((l, x)) } else { None }) {
            match prior.iter_mut().find(|q| q.0 == l || q.0 == l.reversed()) {
                Some(q) => q.1 += x,
                None => prior.push((l, x)),
            }
        }
        let read = errors(fold, &[vec![0]], &fold.right(p), 0.0, &[]);
        for (l, x) in &prior {
            let y = read.links.iter().find(|r| r.0 == *l || r.0 == l.reversed()).map_or(0.0, |r| r.1);
            println!("Λ, eigenstate {p}: the link {l:?}: the closure's {x:.6e}, the readings' {y:.6e}");
            assert!((x - y).abs() <= 1e-12 * x.max(1e-300), "the link {l:?}: the closure's {x} against the readings' {y}");
        }
        assert!(!prior.is_empty(), "the sector's link has its part at eigenstate {p}");
    }
}

/// A Raman resonance across a Zeeman splitting (the coordinator's test): two ground states 2ω_L apart, coupled by two
/// classes 2ω_L + δ apart through a far excited state. The lab beat is 26.37 rad/µs; the coupling's distance on the
/// actual spectrum is δ (and the light shifts' difference): its sector's fold does not exist there, and the closure keeps
/// the link (the existence rule's level up, not a block). In its frames (cluster.rs `frames`: a chain, every link on the
/// strength tree) it is static, its residual exactly zero -- and the fold in those frames is the exact static problem's
/// (its slow eigenvalues against the exact three-state generator's).
#[test]
fn a_raman_resonance_across_a_zeeman_splitting_is_kept_and_static_in_its_frames() {
    let (gamma, delta, zeeman, detuning) = (38.1, -1536.0, 26.375, 0.375);
    let (nu1, nu2) = (delta, delta - zeeman - detuning);
    let lv = levels(&[0.0, zeeman], &[0.0], gamma, &[0]);
    let classes = vec![class(nu1, 2, 1, &[(0, 0, C64::real(40.0))]), class(nu2, 2, 1, &[(0, 1, C64::real(40.0))])];
    assert!(!sectored(&lv, &classes, &at_rest(2), false).exists, "the resonant sector's fold does not exist");
    let closed = closure(&lv, &classes, &[vec![0]], [0.0; 3], &Kept::default());
    assert!(closed.kept.blocks.is_empty(), "no block is made explicit for it");
    let raman: Vec<&Link> = closed.kept.links.iter().filter(|l| l.a < 2 && l.b < 2 && l.a != l.b).collect();
    assert!(!raman.is_empty(), "the resonant Raman coupling is kept");
    let slow = slow_of(&[vec![0]], &closed.kept, &classes, &lv);
    for l in &raman {
        assert!((l.beat(&classes).abs() - (zeeman + detuning)).abs() <= 1e-9, "its lab beat");
        assert!(l.beat(&classes) - (slow.frames[l.a] - slow.frames[l.b]) == 0.0, "a chain: static in its frames");
    }
    let fold = &closed.fold;
    assert_eq!(fold.replicas.len(), 1);
    let theta = &slow.frames;
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
/// eigenstates at the offsets each spans (`Fold::offsets`: the motion's pattern and the carried sectors' sidebands at
/// their shares), and the worst part of the sectors it folds (`Fold::sector_parts`, the square root of an eigenstate's
/// sum).
fn measures(closed: &Closed, classes: &[Class], velocity: [f64; 3]) -> (f64, f64) {
    let fastest = classes.iter().flat_map(|c| c.beams.iter()).map(|b| crate::linalg::norm3(b.wave)).fold(0.0, f64::max);
    let pattern = 2.0 * fastest * crate::linalg::norm3(velocity);
    let fold = &closed.fold;
    let like: Vec<usize> = (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
    let remainder = like.iter().map(|&p| fold.block_parts(&[], p, Some(&fold.offsets(p, pattern, 1.0, &|_| 1.0))).0).fold(0.0, f64::max);
    let sectors = like.iter().map(|&p| fold.sector_parts(p, pattern).iter().map(|x| x.1).sum::<f64>().sqrt()).fold(0.0, f64::max);
    (remainder, sectors)
}

/// The tilt's property (the brief's test 5, on the prior's own atoms): the 1D PGC at 3 G with its π probe tilted by
/// θ = 0, 0.1°, 2° (the user's failure #4 is 2°; refs/requests/PGC1D_TILT_*, made by the app's own request builder),
/// each closure at the 88 atoms the prior reads (`prior_atoms`, the θ = 0 request's). The tilt adds couplings ∝ sin θ:
/// - the partition (criterion one's blocks) is the same at every atom and θ;
/// - at 0.1° the untilted links stay and ρ moves by O(θ): no coupling counts as "existing" or not; what 0.1° adds 2°
///   holds -- keeps, or holds static in its frames -- a tilt coupling's sector near a two-photon resonance, whose
///   first-order part (the light's interference through it, # Sectors) grows as θ;
/// - at 2° the untilted links stay, and what is added are the tilt's couplings that are near resonance on the actual
///   spectrum at that atom (the Raman-dressed F = 2 levels: |δ| ≈ 0.4–1.2 rad/µs at a ≈ 0.08–0.3 -- physics the rule
///   keeps, measured 2026-11-14); what is left folded is within the share (criterion one on the sectors).
#[test]
fn a_tilt_leaves_the_partition_and_moves_the_links_only_where_its_couplings_are_resonant() {
    let names = ["PGC1D_TILT_0", "PGC1D_TILT_01", "PGC1D_TILT_2"];
    let requests: Vec<crate::request::Request> = names.iter().map(|n| request(n)).collect();
    let points = prior_atoms(&requests[0]);
    assert_eq!(points.len(), 88);
    let theta = 0.1_f64.to_radians();
    let (mut added, mut most) = (0, 0.0_f64);
    let blocks = excited_blocks(&requests[0].states);
    let ground = crate::request::GROUND;
    // Whether a closure holds a link: kept, or static in its frames -- its two states one group, its lab beat their
    // frames' difference (`slow_of`).
    let holds = |c: &Closed, l: &Link, cl: &[Class], lv: &Levels| {
        let slow = slow_of(&blocks, &c.kept, cl, lv);
        let place = |s: usize| if s < ground { Some(s) } else { slow.explicit.iter().position(|&e| e == s - ground).map(|i| ground + i) };
        c.kept.links.contains(l) || c.kept.links.contains(&l.reversed()) || matches!((place(l.a), place(l.b)), (Some(a), Some(b)) if slow.groups[a] == slow.groups[b] && equal(l.beat(cl) - slow.frames[a] + slow.frames[b], l.size(cl) + slow.sizes[a] + slow.sizes[b]))
    };
    for &(r, v) in &points {
        let read: Vec<(Closed, (f64, f64), Levels, Vec<Class>)> = requests
            .iter()
            .map(|req| {
                let light = crate::field::Light::new(req);
                let (lv, cl, _) = system(req, &light, r);
                let closed = closure(&lv, &cl, &blocks, v, &Kept::default());
                let measured = measures(&closed, &cl, v);
                (closed, measured, lv, cl)
            })
            .collect();
        let closed: Vec<&Closed> = read.iter().map(|x| &x.0).collect();
        let measured: Vec<(f64, f64)> = read.iter().map(|x| x.1).collect();
        for ((name, c), m) in names.iter().zip(closed.iter()).zip(&measured) {
            println!("{name} r {r:?} v {v:?}: blocks {:?}, links {}, rho {:.4e}, folded sectors {:.4e}", c.kept.blocks, c.kept.links.len(), m.0, m.1);
        }
        let base = closed[0];
        for c in &closed[1..] {
            assert_eq!(c.kept.blocks, base.kept.blocks, "the partition at {r:?} {v:?}");
            for l in &base.kept.links {
                assert!(c.kept.links.contains(l), "the untilted link {l:?} at {r:?} {v:?}");
            }
        }
        for l in closed[1].kept.links.iter().filter(|l| !base.kept.links.contains(l)) {
            assert!(holds(closed[2], l, &read[2].3, &read[2].2), "a link at 0.1° that 2° does not hold, {l:?}, at {r:?} {v:?}");
        }
        let (at0, at01) = (measured[0], measured[1]);
        assert!((at01.0 - at0.0).abs() <= 10.0 * theta * at0.0.max(1e-6), "ρ at 0.1°: {} against {}", at01.0, at0.0);
        added += closed[2].kept.links.len() - base.kept.links.len();
        most = most.max(measured[2].1);
    }
    println!("2°: {added} links added over the 88 atoms; the largest folded sectors' part {most:.3e}");
    assert!(rate(most) <= super::share(), "a folded sector beyond the share");
}

/// Each link's part of criterion one moves continuously with the tilt (the coordinator's 2026-11-14 statement): the tilt's
/// couplings are the probe's, linear in sin θ, so along the family of the 0° and the 2° requests' levels and couplings in
/// the proportions 1 − s and s (θ = 2s° to the order θ²), on one structure -- the 0° closure's kept set, its groups and
/// frames held --, every link's part at an atom is a smooth function of s. Read at s = 0, ¼, ½, ¾, 1 against s + h for
/// h = 10⁻³ and 10⁻⁴: the largest change of any link's part falls with h at least fivefold (a jump would not fall), at the
/// prior's atoms (every eleventh). Where a decision changes with θ it is where a part crosses the share continuously:
/// the 2° closure's links beyond the 0°'s are each kept at an eigenstate whose error, 2ρ + ρ² (`rate`), is beyond the
/// share on the 2° fold with the 0° set kept -- printed, with the count.
#[test]
fn a_tilt_moves_every_link_s_part_continuously() {
    let (zero, two) = (request("PGC1D_TILT_0"), request("PGC1D_TILT_2"));
    let (l0, l2) = (crate::field::Light::new(&zero), crate::field::Light::new(&two));
    let blocks = excited_blocks(&zero.states);
    let points = prior_atoms(&zero);
    let blend = |a: &Mat, b: &Mat, s: f64| {
        let mut m = a.scaled(C64::real(1.0 - s));
        m.add_scaled(C64::real(s), b);
        m
    };
    let (mut beyond, mut added) = (0, 0);
    for &(r, v) in points.iter().step_by(11) {
        let ((lv0, cl0, _), (lv2, cl2, _)) = (system(&zero, &l0, r), system(&two, &l2, r));
        assert_eq!(cl0.len(), cl2.len());
        let at = |s: f64| -> (Levels, Vec<Class>) {
            let lv = Levels { ground: blend(&lv0.ground, &lv2.ground, s), excited: blend(&lv0.excited, &lv2.excited, s), decay: lv0.decay.clone() };
            let cl = cl0.iter().zip(&cl2).map(|(a, b)| Class { nu: a.nu, beams: a.beams.iter().zip(&b.beams).map(|(x, y)| Beam { coupling: blend(&x.coupling, &y.coupling, s), wave: x.wave }).collect() }).collect();
            (lv, cl)
        };
        let fastest = cl0.iter().flat_map(|c| c.beams.iter()).map(|b| crate::linalg::norm3(b.wave)).fold(0.0, f64::max);
        let pattern = 2.0 * fastest * crate::linalg::norm3(v);
        let base = closure(&lv0, &cl0, &blocks, v, &Kept::default()).kept;
        // Every link's part, summed over the ground-like eigenstates, on the 0° structure at s.
        let parts = |s: f64| -> Vec<(Link, f64)> {
            let (lv, cl) = at(s);
            let slow = slow_of(&blocks, &base, &cl, &lv);
            let f = fold(&lv, &cl, &slow, &Replicas::of(&cl, &slow, lv.ground.rows), Some(&|l: &Link| base.links.contains(l)));
            let mut out: Vec<(Link, f64)> = vec![];
            for p in (0..f.size()).filter(|&p| f.ground_weight(p) >= 0.5) {
                for (l, x) in f.sector_parts(p, pattern) {
                    match out.iter_mut().find(|o| o.0 == l || o.0 == l.reversed()) {
                        Some(o) => o.1 += x,
                        None => out.push((l, x)),
                    }
                }
            }
            out
        };
        let moved = |a: &[(Link, f64)], b: &[(Link, f64)]| a.iter().map(|x| (x.1 - b.iter().find(|y| y.0 == x.0 || y.0 == x.0.reversed()).map_or(0.0, |y| y.1)).abs()).fold(0.0_f64, f64::max);
        for s in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let here = parts(s);
            let sign = if s < 1.0 { 1.0 } else { -1.0 };
            let (coarse, fine) = (moved(&here, &parts(s + sign * 1e-3)), moved(&here, &parts(s + sign * 1e-4)));
            let largest = here.iter().map(|x| x.1).fold(0.0_f64, f64::max);
            assert!(fine <= coarse / 5.0 + crate::units::rounding(1024) * largest, "at {r:?}, s {s}: a link's part moved {coarse:.3e} over 10⁻³ and {fine:.3e} over 10⁻⁴");
        }
        // The 2° closure's links beyond the 0°'s: each eigenstate's error at 2° on the 0° structure.
        let kept2 = closure(&lv2, &cl2, &blocks, v, &Kept::default()).kept;
        let more: Vec<&Link> = kept2.links.iter().filter(|l| !base.links.contains(l) && !base.links.contains(&l.reversed())).collect();
        if !more.is_empty() {
            let slow = slow_of(&blocks, &base, &cl2, &lv2);
            let f = fold(&lv2, &cl2, &slow, &Replicas::of(&cl2, &slow, lv2.ground.rows), Some(&|l: &Link| base.links.contains(l)));
            let worst = (0..f.size()).filter(|&p| f.ground_weight(p) >= 0.5).map(|p| rate(f.criterion_one(&blocks, p, pattern, 1.0, &|_| 1.0).0.sqrt())).fold(0.0_f64, f64::max);
            println!("at {r:?}: 2° keeps {} links beyond the 0°'s {}; on the 0° structure at 2° the worst eigenstate's error {worst:.4e} (the share {:.4e}){}", more.len(), base.links.len(), super::share(), if f.exists { "" } else { ", where the fold does not exist" });
            beyond += usize::from(worst > super::share() || !f.exists);
            added += 1;
        }
    }
    println!("{added} atoms where 2° keeps more than 0°; at {beyond} of them the 0° structure is beyond the share at 2°");
    assert_eq!(beyond, added, "2° keeps more only where the 0° structure is beyond the share at 2°");
}

/// The frames are a gauge (module documentation, # Groups): at the prior's atoms of the tilted 1D PGC (a few of them),
/// on two structures -- the closure's first round (no link kept: its groups and frames), where it makes its moves, and
/// its kept set --, a constant added to the frames of every state of each group -- random, of the size of the Zeeman
/// splittings and beyond -- leaves every decision the same, object by object (the moves criterion one makes at
/// each ground-like eigenstate, `closure`'s rule), and every number the same within its rounding: the sectors' parts by
/// link, the blocks' remainders, and the error at a state with its parts.
///
/// The rounding a number is the same within. A gauge is an exact change of variables (H → H − c on a group), but each
/// gauge's fold is its own computation, settled to its floor ε_H = γ_terms s (`settle`: the change of an iteration
/// within the rounding of its operations, terms = (n_P + n_E)², at the scale s of its terms, of the size of ‖H‖_F;
/// PGC1D's first round stops so at its tenth step, contracting by 0.058 a step); an
/// eigenvector r_p of H is then fixed to ε_H/g_p, g_p its eigenvalue's least distance from the others of its group (first
/// order in the perturbation), and a sector's response at p, read off its Sylvester solve, to ε_H/δ_p, δ_p = min |Ω +
/// λ_p' − λ_q| the solve's separation over its blocks' eigenvalue pairs -- beside its own arithmetic's γ_terms κ, κ the
/// eigenvectors' condition: the vector a whose components' squares are p's parts by link (x_l = |a_l|², X_p = ‖a‖² their
/// whole) is fixed to η_p ‖a‖, η_p = γ_terms κ + ε_H/g_p + ε_H/δ_p, the error of a vector, not of each component: a
/// part to 2√(x_l X_p) η_p + X_p η_p² (a small part beside a large one moves with the large one's error; PGC1D's first
/// round, a link of 1.8·10⁻³ beside 0.32 at one eigenstate, ‖H‖_F = 7.5·10⁴ of its hyperfine frames: 1.2·10⁻¹⁰ between
/// gauges against 2x η of 4·10⁻¹²); and a sector's frequency Ω is a sum of frames and classes' frequencies -- of a
/// counter-rotating class the optical frequency's 2.4·10⁹ rad/µs --, known to its terms' rounding δΩ = γ_STATES s_Ω
/// (`equal`'s: what two frequencies are one within), so a part of 1/d², d the least distance of the sector's pole from
/// p's offsets, moves by 2x δΩ/d (PGC1D's first round: Ω = 26.37 of replica frequencies of 2.4·10⁹, d = 0.57, a part
/// of 391 apart by 1.3·10⁻⁶ relatively, between gauges whose frames differ by 10⁻³; by 10⁻⁹, where the replicas'
/// frequencies round alike, 10⁻⁹). Two gauges' parts of a link differ by at most the sum of their floors over the
/// eigenstates; the remainders and the errors, made of the same eigenvectors, relatively by γ_terms κ + 2ε_H/g of the
/// least g among the eigenstates read, the errors besides by their sectors' parts' floors (a rate 2ρ + ρ² of an
/// eigenstate's total t moves by (1 + 1/√t) of t's); and every part by γ_terms of the whole sum's scale, a link of a structural zero
/// being a rounding's. (Read part by part at 10⁻⁸ relatively, the tilt's link of 3.6·10⁻⁹ at an eigenstate near its
/// neighbour moved 8.6·10⁻⁴ between gauges.)
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
    let share = super::share();
    let mut decided = 0;
    for &(r, v) in points.iter().step_by(11) {
        let (lv, cl, _) = system(&req, &light, r);
        let closed = closure(&lv, &cl, &blocks, v, &Kept::default());
        let slow = slow_of(&blocks, &closed.kept, &cl, &lv);
        let none = Kept::default();
        let first = slow_of(&blocks, &none, &cl, &lv);
        let fastest = cl.iter().flat_map(|c| c.beams.iter()).map(|b| crate::linalg::norm3(b.wave)).fold(0.0, f64::max);
        let pattern = 2.0 * fastest * crate::linalg::norm3(v);
        // Per link (either way round) its part summed over the ground-like eigenstates and its floor; the remainders; the
        // relative floor of what is made of the eigenvectors; the error at a state and its parts; the moves, sorted.
        type Read = (Vec<(Link, f64, f64)>, Vec<f64>, f64, (f64, Vec<((usize, Option<usize>), f64)>, f64), Vec<String>);
        let read = |slow: &Slow, kept: &Kept| -> Read {
            let fold = fold(&lv, &cl, slow, &Replicas::of(&cl, slow, lv.ground.rows), Some(&|l: &Link| kept.links.contains(l)));
            let n = fold.size();
            let gamma = crate::units::rounding((n + lv.excited.rows).pow(2));
            let (floor, own) = (gamma * fold.generator.frobenius(), gamma * fold.eigen.condition);
            let gap = |p: usize| (0..n).filter(|&q| q != p && fold.groups[q] == fold.groups[p]).map(|q| (fold.eigen.values[q] - fold.eigen.values[p]).abs()).fold(f64::INFINITY, f64::min);
            let like: Vec<usize> = (0..n).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
            let least = like.iter().map(|&p| gap(p)).fold(f64::INFINITY, f64::min);
            let mut sectors: Vec<(Link, f64, f64)> = vec![];
            let mut moves: Vec<String> = vec![];
            // The error's floor (`errors`: a rate 2ρ + ρ², of the total t at an eigenstate, moves by (1 + 1/√t) of t's).
            let mut error_floor = 0.0;
            // The separation of the Sylvester solves eigenstate p's sector parts are read from: |Ω + λ_p' − λ_q| over
            // every block from p's group, p' of that group and q of the block's target (a solve's error is its
            // separation's, of all its eigenvalues' pairs, not p's alone).
            let (groups, values) = (&fold.groups, &fold.eigen.values);
            let distance = |p: usize| {
                fold.sectors
                    .iter()
                    .flat_map(|s| s.blocks.iter().filter(move |b| b.1 == groups[p]).flat_map(move |b| (0..n).filter(move |&o| groups[o] == b.1).flat_map(move |o| (0..n).filter(move |&q| groups[q] == b.0).map(move |q| (C64::real(s.frequency) + values[o] - values[q]).abs()))))
                    .fold(f64::INFINITY, f64::min)
            };
            for &p in &like {
                let parts = fold.sector_parts(p, pattern);
                let (whole, eta) = (parts.iter().map(|x| x.1).sum::<f64>(), own + floor / gap(p) + floor / distance(p));
                let mut at_p = 0.0;
                for (l, x) in parts {
                    // The sector's frequency to its terms' rounding (`equal`'s), at the least distance of its pole from p's
                    // offsets: the part, of 1/d², moves by 2x δΩ/d.
                    let sector = fold.sectors.iter().find(|s| s.links.iter().any(|m| m.0 == l)).expect("a part's sector");
                    let lambda = fold.eigen.values[p];
                    let mut pole = f64::INFINITY;
                    for (to, _) in fold.targets(sector, p) {
                        let k = &fold.stills[to];
                        for q in 0..k.n {
                            for o in [0.0, pattern, -pattern] {
                                pole = pole.min((lambda + C64::real(sector.frequency + o) - k.s[q * k.n + q]).abs());
                            }
                        }
                    }
                    let beat = 2.0 * x * crate::units::rounding(crate::request::STATES) * sector.scale / pole;
                    let rounding = 2.0 * (x * whole).sqrt() * eta + whole * eta * eta + beat;
                    at_p += rounding;
                    match sectors.iter_mut().find(|s| s.0 == l || s.0 == l.reversed()) {
                        Some(s) => (s.1, s.2) = (s.1 + x, s.2 + rounding),
                        None => sectors.push((l, x, rounding)),
                    }
                }
                // The moves at this eigenstate (`closure`): its parts largest first until what is left is within the share.
                let (total, mut parts) = fold.criterion_one(&blocks, p, pattern, 1.0, &|_| 1.0);
                if rate(total.sqrt()) < 1.0 && total > 0.0 {
                    error_floor += (1.0 + 1.0 / total.sqrt()) * at_p;
                }
                if rate(total.sqrt()) > share {
                    parts.sort_by(|x, y| y.1.total_cmp(&x.1));
                    let mut rest = total;
                    for (o, x) in parts {
                        if !(rate(rest.max(0.0).sqrt()) > share) {
                            break;
                        }
                        rest -= x;
                        moves.push(match o {
                            Object::Block { block, class, .. } => format!("block {block} {class:?}"),
                            Object::Link(l) => format!("{:?}", if (l.a, l.b) <= (l.b, l.a) { l } else { l.reversed() }),
                        });
                    }
                }
            }
            moves.sort();
            moves.dedup();
            // And every part's rounding at the scale of the whole sum: a link of a structural zero, its part a rounding's.
            let whole: f64 = sectors.iter().map(|x| x.1).sum();
            sectors.iter_mut().for_each(|x| x.2 += gamma * whole);
            let mut rho: Vec<f64> = like.iter().map(|&p| fold.block_parts(&blocks, p, Some(&fold.offsets(p, pattern, 1.0, &|_| 1.0))).0).collect();
            rho.sort_by(f64::total_cmp);
            let psi: Vec<C64> = (0..n).map(|s| if s < ground { C64::real(1.0) } else { C64::ZERO }).collect();
            let errors = errors(&fold, &blocks, &psi, pattern, &[]);
            let error = errors.parts.iter().map(|p| p.1).sum::<f64>() + errors.links.iter().map(|p| p.1).sum::<f64>() + errors.outside.iter().sum::<f64>();
            (sectors, rho, own + 2.0 * floor / least, (error, errors.parts, error_floor), moves)
        };
        for (slow, kept) in [(&first, &none), (&slow, &closed.kept)] {
            let base = read(slow, kept);
            for _ in 0..3 {
                let mut shifted = slow.clone();
                let shifts: Vec<f64> = (0..shifted.frames.len()).map(|_| 200.0 * (uniform() - 0.5)).collect();
                // A frame moved by c is a sum of one term more: its terms' size grows by |c| (`Sized`).
                for ((f, size), g) in shifted.frames.iter_mut().zip(shifted.sizes.iter_mut()).zip(&slow.groups) {
                    (*f, *size) = (*f + shifts[*g], *size + shifts[*g].abs());
                }
                let other = read(&shifted, kept);
                assert_eq!(base.4, other.4, "the moves at {r:?}");
                decided += base.4.len();
                // Every link's part the same within the two folds' floors, either way round.
                for (a, b) in [(&base.0, &other.0), (&other.0, &base.0)] {
                    for x in a.iter() {
                        let (y, floor) = b.iter().find(|y| y.0 == x.0 || y.0 == x.0.reversed()).map_or((0.0, 0.0), |y| (y.1, y.2));
                        assert!((x.1 - y).abs() <= x.2 + floor, "the part of the link {:?}: {} against {y} (floors {:.2e}, {floor:.2e})", x.0, x.1, x.2);
                    }
                }
                let relative = base.2 + other.2;
                for (a, b) in base.1.iter().zip(&other.1) {
                    assert!((a - b).abs() <= relative * a.abs(), "a remainder {a} against {b}");
                }
                // The error and its parts: of the eigenvectors, relatively, and of the sectors' parts, their floors.
                let floors = base.3 .2 + other.3 .2;
                assert!((base.3 .0 - other.3 .0).abs() <= relative * base.3 .0 + floors, "the error {} against {} (floors {floors:.2e})", base.3 .0, other.3 .0);
                for (object, x) in &base.3 .1 {
                    let y = other.3 .1.iter().find(|o| o.0 == *object).map_or(0.0, |o| o.1);
                    assert!((x - y).abs() <= relative * x + floors, "the part of {object:?}: {x} against {y}");
                }
            }
        }
    }
    println!("{decided} moves, the same in every gauge");
    assert!(decided > 0, "the first rounds make moves");
}

/// A block of two classes (the coordinator's test): one ground state driven into one excited state by two classes of
/// different frequencies. The block's part of the error at the ground state goes to the classes by their responses'
/// shares (`class_shares`); with the two couplings swapped the larger share swaps with them. Kept for the class of the
/// larger part, the block is explicit at that class's replica: its drive static in the frames (residual zero, A's), the
/// other class's coupling a turning coupling of P at its residual beat -- the explicit state is P at every frequency,
/// its copy at the other class's a Sambe sector (ELIMINATION v12) --: no replica feeds it, no Q holds the explicit state,
/// and the sector of that beat holds the other class's link.
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
        let found = errors(&fold, &blocks, &psi, 5.0, &[]);
        let part = |k: usize| found.parts.iter().find(|o| o.0 == (0, Some(k))).map_or(0.0, |o| o.1);
        // The responses' shares: |g|²/|ν − iΓ/2|² of each class.
        let response = |k: usize| size(k).powi(2) / (nus[k].powi(2) + 0.25 * gamma * gamma);
        let (larger, smaller) = if response(0) > response(1) { (0, 1) } else { (1, 0) };
        assert!(part(larger) > part(smaller), "the class of the larger response holds the larger part: {} against {}", part(larger), part(smaller));
        // Kept for that class: the drive's link, the frames from it.
        let kept = Kept { blocks: vec![0], links: vec![Link { a: ground, b: 0, classes: (None, Some(larger)) }], ..Kept::default() };
        let slow = slow_of(&blocks, &kept, &classes, &lv);
        assert!(equal(classes[larger].nu + slow.frames[0] - slow.frames[ground], classes[larger].nu.abs() + slow.sizes[0] + slow.sizes[ground]), "the explicit state turns in its class's frame");
        let fold = folded(&lv, &classes, &slow);
        assert!(fold.direct[(1, 0)].abs() > 0.0, "the kept class's drive is the static generator's");
        assert!(fold.replicas.iter().all(|rep| rep.feeds.iter().all(|f| f.class != Some(smaller)) && !rep.q.contains(&0)), "no replica folds the explicit state, at any frequency");
        let sectors = sectored(&lv, &classes, &slow, false);
        let beat = classes[smaller].nu + slow.frames[0] - slow.frames[ground];
        let link = Link { a: ground, b: 0, classes: (None, Some(smaller)) };
        assert!(sectors.sectors.iter().any(|s| equal(s.frequency - beat, s.scale + classes[smaller].nu.abs() + slow.sizes[0] + slow.sizes[ground]) && s.links.iter().any(|l| l.0 == link)), "the other class's drive is the sector of its residual beat {beat}");
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
        let q = fold.series_bound(z0);
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
///   `Fold::definitions`, `Fold::evaluation_error`): on random slow states, falling from the series' order 0 to order 1 and
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
        let q = fold.series_bound(C64::real(z0));
        let x_size = fold.evaluated_share(&vec![z0; n]).0;
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
        let reference = Reference::new(&lv, &nus, &at_rest(n), &vec![z0; n]);
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
            let mut data = vec![C64::ZERO; (crate::model::assemble::FIRST + 1) * n * n];
            crate::model::assemble::normal(&four(&hermitian), &four(&n1), &[], 1, &four(&x), &four(&l), n, &vec![[z0, 0.0, 0.0, 0.0]; n], orders, true, true, &mut data, &mut vec![]);
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
        // The errors the run books, measured on random slow states against the definition (the route's whole normalized
        // generator -- the run's `normal` and the decay of 𝒯 = 𝒯⁽⁰⁾ + 𝒯⁽¹⁾ of the same forms, assemble.rs `Middle::generator`
        // -- against `Fold::definitions`, a point's numbers, not bounds): the error falls from the series' order 0 to 1 and
        // with the normalization's order.
        let defined = fold.definitions(&lv).0;
        let at = |orders: Orders| {
            let data = made(orders);
            let (bent, e) = (matrix(&data, crate::model::assemble::BENT), matrix(&data, crate::model::assemble::OFFSET));
            let mut whole = hermitian.clone();
            whole.add_scaled(C64::ONE, &matrix(&data, crate::model::assemble::HELD));
            let mut inner = Mat::zeros(n, n);
            for t in &responses {
                let mut wave = t[0].clone();
                if orders.series >= 1 {
                    wave.add_scaled(-C64::ONE, &t[1].product(&e));
                }
                inner.add_scaled(C64::ONE, &wave.adjoint_product(&lv.decay.product(&wave)));
            }
            let mut out = whole.hermitian_part();
            out.add_scaled(C64::new(0.0, -0.5), &bent.product(&inner).product(&bent));
            out
        };
        let mut stream = crate::random::Stream::new(3, crate::random::Purpose::Initial, 0, 0);
        for _ in 0..4 {
            let psi: Vec<C64> = (0..n).map(|_| C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
            let error = |series: usize, normalization: usize| Fold::evaluation_error(&at(Orders { series, normalization }), &defined, &vec![z0; n], &psi);
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
    let slow = Slow { explicit: vec![0], frames: vec![0.0, 0.0, nu], sizes: vec![0.0, 0.0, nu.abs()], groups: vec![0, 1, 0] };
    let whole = folded(&lv, &classes, &slow);
    assert!(whole.generator.data.iter().all(|z| z.re.is_finite() && z.im.is_finite()), "a finite fold");
    let first = folded(&levels(&[0.0], &[0.0], gamma, &[0]), &[class(nu, 1, 1, &[])], &single(vec![0], vec![0.0, nu]));
    let second = folded(&levels(&[5.0], &[0.0], gamma, &[0]), &[class(nu, 1, 1, &[(0, 0, C64::real(g))])], &single(vec![], vec![0.0]));
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
/// response feels it across its distance to the line: booked beyond the share at 500 MHz, no less at 30 MHz -- where
/// |3′,3′⟩'s population, the line moved away, is smaller still (each at most all of the rate, `rate`). Far from the level (100 GHz) and without the beam
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
    assert!(booked[3].0 >= booked[2].0, "Δ = 30 MHz books no less: {booked:?}");
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
        for (rep, responses) in self.replicas.iter().zip(self.responses_about(&vec![z0.re; n], orders)) {
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
    fn series_bound(&self, z0: C64) -> f64 {
        self.series_bound_about(&vec![z0.re; self.size()])
    }
    /// The series' bound about Z₀ = diag(z_y), q = ‖H − Z₀‖₂/d, d the least distance of z_y + f over the places of
    /// each replica's group (ELIMINATION v11): its remainder after order K within q^{K+1}/(1 − q) where E = H − Z₀ is
    /// normal enough for ‖E‖ to bound its spectrum -- the tests' proven bound; the engine takes the step's own radius
    /// (`Fold::series_ratio`, ELIMINATION v12).
    fn series_bound_about(&self, z0: &[f64]) -> f64 {
        self.series_norm(z0) / self.series_distance(z0, None)
    }
    /// ‖H − Z₀‖₂ (`series_bound_about`).
    fn series_norm(&self, z0: &[f64]) -> f64 {
        let mut e = self.generator.clone();
        for i in 0..self.size() {
            e[(i, i)] -= C64::real(z0[i]);
        }
        crate::linalg::hermitian_eigen(&e.adjoint_product(&e)).values.iter().fold(0.0_f64, |m, &x| m.max(x)).sqrt()
    }
    /// The series' generator about Z₀ = diag(z_y) to order K (ELIMINATION v11): column s of the Sylvester equation,
    /// (z_s + f − K)T[:, s] = W_s − (T E)[:, s], E = H − Z₀, gives T = Σ_k T⁽ᵏ⁾, T⁽⁰⁾[:, s] = R₀(s)W_s,
    /// T⁽ᵏ⁾[:, s] = −R₀(s)(T⁽ᵏ⁻¹⁾E)[:, s] -- with one reference T_f = Σ_k (−1)^k R₀^{k+1} W_f E^k (`series`) --, and
    /// H = A + Σ_f W′_f T_f its fixed point in H (`settle`); None where it does not settle.
    fn series_generator_about(&self, z0: &[f64], orders: usize) -> Option<Mat> {
        let n = self.size();
        let solve = |rep: &Folded, x: &Mat| -> Mat {
            let mut out = Mat::zeros(x.rows, n);
            for &s in &self.members[rep.group] {
                let column: Vec<C64> = (0..x.rows).map(|i| x[(i, s)]).collect();
                let solved = rep.k.resolvent(C64::real(z0[s] + rep.frequency), &column);
                for i in 0..x.rows {
                    out[(i, s)] = solved[i];
                }
            }
            out
        };
        let (h, exists) = settle(
            &self.direct,
            |h| {
                let mut e = h.clone();
                for i in 0..n {
                    e[(i, i)] -= C64::real(z0[i]);
                }
                let (mut next, mut scale) = (self.direct.clone(), self.direct.frobenius());
                for rep in &self.replicas {
                    let mut term = solve(rep, &rep.w);
                    let mut t = term.clone();
                    for _ in 0..orders {
                        term = solve(rep, &term.product(&e)).scaled(-C64::ONE);
                        t.add_scaled(C64::ONE, &term);
                    }
                    let form = rep.back.product(&t);
                    scale += form.frobenius();
                    next.add_scaled(C64::ONE, &form);
                }
                (next, scale)
            },
            n * n * (orders + 1),
        );
        exists.then_some(h)
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
    let mut worst: Option<(f64, Fold, Vec<f64>)> = None;
    let mut unheld = 0;
    for k in 0..model.clusters().len() {
        for &(r, _) in &points {
            let amplitudes = light.amplitudes(r);
            let Some((fold, _, far, _)) = model.cluster_fold(k, &amplitudes, false) else { continue };
            let z0 = model.reference_of(k, &far);
            let mut every = f64::INFINITY;
            for rep in &fold.replicas {
                for &y in &fold.members[rep.group] {
                    for i in 0..rep.k.n {
                        every = every.min((C64::real(z0[y] + rep.frequency) - rep.k.s[i * rep.k.n + i]).abs());
                    }
                }
            }
            let norm = fold.series_norm(&z0);
            if every < norm && fold.series_distance(&z0, None) > norm {
                unheld += 1;
            }
            let q = fold.series_bound_about(&z0);
            if worst.as_ref().is_none_or(|w| q > w.0) {
                worst = Some((q, fold, z0));
            }
        }
    }
    println!("S2Q, the probe's F′ = 3 block kept for F = 2 m = −2, −1, 0: {} clusters; {unheld} cluster folds with a block in no term nearer than ‖H − z₀‖; series q = {:?}, x = {:.3e}", model.clusters().len(), model.series.ratios, model.series.excited);
    assert!(unheld > 0, "the scene holds a block the series does not hold within ‖H − z₀‖ of z₀ + f");
    assert!(model.series.exists() && (0..model.clusters().len()).all(|k| model.series.holds(k)), "the series converge on the blocks they hold: q = {:?}, x = {}", model.series.ratios, model.series.excited);
    assert_eq!(model.series_block(), None, "the closure makes no block explicit");
    // The series' orders at its largest ratio converge as that ratio bounds them.
    let (q, fold, z0) = worst.expect("a cluster with a fold");
    let n = fold.size();
    let own = fold.generator.clone();
    for order in 0..=3 {
        let h = fold.series_generator_about(&z0, order).expect("the series' generator settles");
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
        let rho: f64 = f.offsets(0, 10.0, 1.0, &|_| 1.0).iter().map(|x| x * x).sum::<f64>().sqrt();
        (rho, f.series_bound(C64::real(f.eigen.values[0].re)))
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
        let slow = Slow { explicit: vec![], frames: vec![0.0, beta], sizes: vec![0.0, beta], groups: vec![0, 0] };
        let f = fold(&levels, &classes, &slow, &Replicas::of(&classes, &slow, 2), None);
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
/// cluster's spectrum (its fold about its reference, `Model::clusters_fold`): each ground-like eigenstate r_p, and each
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
                let source = model.clusters_fold(&[s], &amplitudes, true, false, true, false).expect("a cluster's fold").0;
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
                        let fold = if l == s { source.clone() } else { model.clusters_fold(&[l], &amplitudes, true, false, true, false).expect("a cluster's fold").0 };
                        Landing::of(&fold, &(0..fold.size()).collect::<Vec<_>>())
                    })
                    .collect();
                let mut images: Vec<Images> = classes
                    .iter()
                    .map(|class| {
                        let m = model.clusters[class.landing].states.len();
                        Images { frequency: -model.omega(&class.beats[0]), scale: model.axes.size(&class.beats[0]), landing: landed.iter().position(|&l| l == class.landing).unwrap(), images: vec![std::array::from_fn(|_| vec![C64::ZERO; m]); n] }
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

/// The kept units by kind on the scenes the brief runs (a measurement): PGC1D_TILT_0, PGC1D_TILT_2 and PGC3D_1ms, each
/// closed as the solve closes it at its reference state (`Model::closed_at`), with the closure's wall time -- the links
/// kept (`Unit::Term`: static in the clusters' frames, or turning as tones), the drives at the atom (`Unit::Bare`), the
/// explicit drives and blocks, the coherences, the evaluation's orders.
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn the_kept_units_by_kind_on_the_brief_s_scenes() {
    use super::super::Unit;
    for name in ["PGC1D_TILT_0", "PGC1D_TILT_2", "PGC3D_1ms"] {
        let req = request(name);
        let light = crate::field::Light::new(&req);
        let chief = (0..crate::request::GROUND).max_by(|&a, &b| req.initial_density[(a, a)].re.total_cmp(&req.initial_density[(b, b)].re)).unwrap();
        let started = std::time::Instant::now();
        let (model, kept, _, _) = super::super::Model::closed_at(&req, &light, vec![], chief, req.temperature_uk);
        let wall = started.elapsed().as_secs_f64();
        let count = |f: &dyn Fn(&Unit) -> bool| kept.iter().filter(|u| f(u)).count();
        let terms: Vec<&Unit> = kept.iter().filter(|u| matches!(u, Unit::Term { .. })).collect();
        let turning = terms.iter().filter(|u| if let Unit::Term { a, b, beat } = u {
            let k = model.cluster_of(*a);
            let c = &model.clusters()[k];
            let place = |s: usize| c.states.iter().position(|&x| x == s);
            match (place(*a), place(*b)) { (Some(x), Some(y)) => !beat.plus(1, &c.frames[x]).plus(-1, &c.frames[y]).is_zero(), _ => false }
        } else { false }).count();
        println!(
            "{name}: closed in {wall:.2} s; kept {}: Term {} ({turning} turning in the frames), Bare {}, Drive {}, Block {}, Coherence {}, Series {}, Normalization {}; clusters {:?}",
            kept.len(), terms.len(), count(&|u| matches!(u, Unit::Bare { .. })), count(&|u| matches!(u, Unit::Drive { .. })), count(&|u| matches!(u, Unit::Block { .. })), count(&|u| matches!(u, Unit::Coherence { .. })), count(&|u| matches!(u, Unit::Series)), count(&|u| matches!(u, Unit::Normalization { .. })),
            model.clusters().iter().map(|c| c.states.len()).collect::<Vec<_>>()
        );
    }
}

/// Where the model's building and its readings spend their time (a measurement, the finish brief's item 4): on
/// PGC1D_TILT_0, PGC1D_TILT_2 and PGC3D_1ms, at the prior's atoms --
/// - the closure's first round (nothing kept: every ground state its own group): its groups, its sectors and their
///   links, how many of them are at the beat zero of one class (the same class absorbed and emitted, the two states'
///   frames equal), the fold's wall time without sectors and with them, and the whole closure's;
/// - the model closed as the solve closes it, its wall time;
/// - at a state of each cluster at each atom (its ground states in equal parts): a reading's wall time, and its parts --
///   the run's fold of the whole atom (`Model::atom_fold`) and its sectors, `errors` on it, the definition's fold and
///   its `definitions`.
#[test]
#[ignore = "a measurement, not a check: run by name"]
fn the_closure_s_and_the_readings_cost_by_part() {
    use super::super::{Point, Sample};
    use std::time::Instant;
    let seconds = |t: Instant| t.elapsed().as_secs_f64();
    for name in ["PGC1D_TILT_0", "PGC1D_TILT_2", "PGC3D_1ms"] {
        let req = request(name);
        let light = crate::field::Light::new(&req);
        let blocks = excited_blocks(&req.states);
        let points = prior_atoms(&req);
        let (mut groups, mut sectors, mut links, mut zero_same, mut zero_links) = (0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut t_static, mut t_sectors, mut t_closure) = (0.0, 0.0, 0.0);
        for &(r, v) in &points {
            let (lv, cl, _) = system(&req, &light, r);
            let ground = lv.ground.rows;
            let slow = slow_of(&blocks, &Kept::default(), &cl, &lv);
            let replicas = Replicas::of(&cl, &slow, ground);
            let mut distinct = slow.groups.clone();
            distinct.sort();
            distinct.dedup();
            groups += distinct.len();
            let t = Instant::now();
            let _ = fold(&lv, &cl, &slow, &replicas, None);
            t_static += seconds(t);
            let t = Instant::now();
            let f = fold(&lv, &cl, &slow, &replicas, Some(&|_: &Link| false));
            t_sectors += seconds(t);
            sectors += f.sectors.len();
            for s in &f.sectors {
                links += s.links.len();
                if equal(s.frequency, s.scale) {
                    let same_class = s.links.iter().filter(|l| l.0.classes.0 == l.0.classes.1).count();
                    if same_class == s.links.len() {
                        zero_same += 1;
                    }
                    zero_links += same_class;
                }
            }
            let t = Instant::now();
            let _ = closure(&lv, &cl, &blocks, v, &Kept::default());
            t_closure += seconds(t);
        }
        let k = points.len() as f64;
        println!("{name}: {} atoms; the first round a closure: {:.1} groups, {:.1} sectors of {:.1} links, of them at the beat zero of one class {:.1} sectors ({:.1} links); fold {:.2} ms without sectors, {:.2} ms with; the whole closure {:.2} ms", points.len(), groups as f64 / k, sectors as f64 / k, links as f64 / k, zero_same as f64 / k, zero_links as f64 / k, 1e3 * t_static / k, 1e3 * t_sectors / k, 1e3 * t_closure / k);
        let chief = (0..crate::request::GROUND).max_by(|&a, &b| req.initial_density[(a, a)].re.total_cmp(&req.initial_density[(b, b)].re)).unwrap();
        let t = Instant::now();
        let (model, kept, _, _) = super::super::Model::closed_at(&req, &light, vec![], chief, req.temperature_uk);
        println!("{name}: closed in {:.2} s (wall, the pool's), {} units kept; clusters {:?}", seconds(t), kept.len(), model.clusters().iter().map(|c| c.states.len()).collect::<Vec<_>>());
        let units = model.candidates();
        let hm = model.hbar_over_mass();
        let (mut reads, mut whole, mut run, mut errs, mut definition, mut defined, mut folded) = (0usize, 0.0, 0.0, 0.0, 0.0, 0.0, 0usize);
        for &(r, v) in &points {
            let amplitudes = light.amplitudes(r);
            for (cluster, c) in model.clusters().iter().enumerate().filter(|(_, c)| c.ground > 0) {
                let n = c.states.len();
                let psi: Vec<C64> = (0..n).map(|s| if s < c.ground { C64::real((c.ground as f64).sqrt().recip()) } else { C64::ZERO }).collect();
                let sample = Sample { cluster, at: Point { r, kappa: v.map(|x| x / hm), t: 0.0 }, psi: psi.clone() };
                let t = Instant::now();
                let _ = model.readings(&units, &sample, None);
                whole += seconds(t);
                reads += 1;
                let t = Instant::now();
                let (atom, states) = model.atom_fold(&amplitudes);
                run += seconds(t);
                folded += atom.sectors.len();
                let mut on = vec![C64::ZERO; atom.size()];
                for (i, &x) in c.states.iter().enumerate() {
                    on[states.iter().position(|&y| y == x).unwrap()] = psi[i];
                }
                let t = Instant::now();
                let _ = errors(&atom, &blocks, &on, 0.0, &[]);
                errs += seconds(t);
                let t = Instant::now();
                let Some((d, lv)) = model.definition_fold(cluster, &amplitudes) else { continue };
                definition += seconds(t);
                let t = Instant::now();
                let _ = d.definitions(&lv);
                defined += seconds(t);
            }
        }
        let k = reads as f64;
        println!("{name}: {reads} readings, {:.3} ms each: the atom's run fold {:.3} ({:.1} sectors), errors {:.3}, the definition's fold {:.3}, its definitions {:.3}", 1e3 * whole / k, 1e3 * run / k, folded as f64 / k, 1e3 * errs / k, 1e3 * definition / k, 1e3 * defined / k);
    }
}

/// R4_LIN_4 closed at its request's state (testing.rs `Model::closed_at`), its prior's atoms, and its clusters whose
/// places have more than one reference class (eliminate.rs `Model::references`) among those with a normalized generator.
fn r4_lin_4() -> (crate::model::Model, Vec<([f64; 3], [f64; 3])>, Vec<usize>) {
    let req = crate::model::testing::scene("R4_LIN_4");
    let light = crate::field::Light::new(&req);
    let state = (0..crate::request::GROUND).max_by(|&x, &y| req.initial_density[(x, x)].re.total_cmp(&req.initial_density[(y, y)].re)).unwrap();
    let (model, ..) = crate::model::Model::closed_at(&req, &light, vec![], state, req.temperature_uk);
    let points = crate::solve::atoms(&model, &req, &light, &crate::solve::components(&req), req.temperature_uk).expect("the atoms");
    let running = model.running();
    let many: Vec<usize> = (0..model.clusters().len()).filter(|&k| model.plan(k, &running).fourth.is_some() && model.references(k, &running).iter().any(|&a| a > 0)).collect();
    (model, points, many)
}

/// The run's series' first order about each place's own reference (ELIMINATION v12; assemble.rs, # The normalized
/// generator), measured as the readings measure it: on R4_LIN_4's clusters of more than one reference class -- its F = 1
/// and F = 2 ground states with the explicit states, their far levels apart in the frames -- the model with
/// `Unit::Series` kept takes the first order there (its plans' orders), `Unit::Series` is a candidate of the model without
/// it, and at the prior's atoms the run's own evaluation (assemble.rs `Model::evaluation`: the plan's static normalized
/// generator at rest, the one function the readings call) at order 1 is no farther from the definition (`Fold::
/// definitions`, `Fold::evaluation_error`) than at order 0; the run's decay is its photons' rate on the cluster's states.
#[test]
fn a_cluster_of_more_than_one_reference_takes_the_first_order_of_each_place_s_reference() {
    use crate::model::{Point, Unit};
    use crate::step::Middle as _;
    let (model, points, many) = r4_lin_4();
    assert!(!many.is_empty(), "R4_LIN_4 has a cluster of more than one reference");
    assert!(model.kept.contains(&Unit::Series) || model.candidates().contains(&Unit::Series), "Series a unit");
    let kept = model.rekept(&[model.kept.clone(), vec![Unit::Series, Unit::Normalization { order: 1 }]].concat()).expect("the model with the first orders");
    let running = kept.running();
    let mut read = 0;
    for &k in &many {
        let plan = kept.plan(k, &running);
        assert_eq!(plan.fourth.as_ref().map(|f| f.orders.series), Some(1), "cluster {k}'s run takes the first order");
        let c = &kept.clusters()[k];
        let n = c.states.len();
        for &(r, _) in points.iter().step_by(9) {
            let amplitudes = kept.light.amplitudes(r);
            let (Some((_, _, far, _)), Some((definition, whole))) = (kept.cluster_fold(k, &amplitudes, true), kept.definition_fold(k, &amplitudes)) else { continue };
            let z0 = kept.reference_of(k, &far);
            let defined = definition.definitions(&whole).0;
            let psi: Vec<C64> = (0..n).map(|i| C64::cis(0.4 * i as f64) * (0.5 + 0.2 * i as f64)).collect();
            let error = |series: usize| {
                let orders = Orders { series, ..kept.series.orders() };
                Fold::evaluation_error(&kept.evaluation(&kept.plan_at(k, &running, orders), r).0, &defined, &z0, &psi)
            };
            let (zero, first) = (error(0), error(1));
            println!("cluster {k} at {r:?}: the run's evaluation against the definition at order 0 {zero:.3e}, at order 1 {first:.3e}");
            assert!(first <= zero, "cluster {k} at {r:?}: order 1 {first:e} beyond order 0 {zero:e}");
            // The decay is the photons' rate.
            let middle = kept.assemble(k, r, 0.0, &running);
            let path = middle.path([0.0; 3], [0.0; 3]);
            let mut h = Mat::zeros(n, n);
            middle.generator(&path, 0.0, &mut h);
            let rate = crate::linalg::inner(&psi, &Mat::from_fn(n, n, |a, b| C64::I * (h[(a, b)] - h[(b, a)].conj())).apply(&psi)).re;
            let photons: f64 = kept.photons(k, &Point { r, kappa: [0.0; 3], t: 0.0 }, &psi).iter().map(|ph| ph.weight()).sum();
            assert!((rate - photons).abs() <= 1e-9 * photons, "cluster {k} at {r:?}: the decay {rate:e} against the photons' {photons:e}");
            read += 1;
        }
    }
    assert!(read > 0);
}

/// The series about each place's own reference (ELIMINATION v11; module documentation, # The definition and the route
/// it is evaluated on): on R4_LIN_4's cluster of F = 1 and F = 2 ground states, where one reference for the cluster had
/// q ≈ 1.6, at the prior's atoms the ratio q = ‖H − Z₀‖₂/d about Z₀ = diag(z_y) (`Fold::series_norm`,
/// `Fold::series_distance`) is below one, and the wave operator's series about it, T⁽⁰⁾[:, s] = R₀(s)W_s,
/// T⁽ᵏ⁾[:, s] = −R₀(s) Σ_t T⁽ᵏ⁻¹⁾[:, t] E[t, s] (E = H − Z₀ of the fold's own H: (z_s + f − K)T[:, s] = W_s −
/// (T E)[:, s]), is within q^{K+1}/(1 − q) ‖T⁽⁰⁾‖ of the fold's own T after each order K = 0..3 -- its remainders
/// falling as q's powers.
#[test]
fn the_series_about_each_place_s_reference_falls_as_its_ratio_s_powers_on_a_joined_cluster() {
    let (model, points, many) = r4_lin_4();
    let mut read = 0;
    let mut largest = 0.0_f64;
    for &k in &many {
        for &(r, _) in points.iter().step_by(5) {
            let amplitudes = model.light.amplitudes(r);
            let Some((fold, _, far, _)) = model.cluster_fold(k, &amplitudes, false) else { continue };
            assert!(fold.exists, "the cluster's fold exists at {r:?}");
            let z0 = model.reference_of(k, &far);
            let n = fold.size();
            let q = fold.series_norm(&z0) / fold.series_distance(&z0, None);
            largest = largest.max(q);
            assert!(q < 1.0, "q {q} at {r:?}");
            let mut e = fold.generator.clone();
            for i in 0..n {
                e[(i, i)] -= C64::real(z0[i]);
            }
            let (mut first, mut remainders) = (0.0_f64, vec![0.0_f64; 4]);
            for rep in &fold.replicas {
                // R₀(s) on every column of x, column s at its own reference.
                let solve = |x: &Mat| -> Mat {
                    let mut out = Mat::zeros(x.rows, n);
                    for &s in &fold.members[rep.group] {
                        let column: Vec<C64> = (0..x.rows).map(|i| x[(i, s)]).collect();
                        let solved = rep.k.resolvent(C64::real(z0[s] + rep.frequency), &column);
                        for i in 0..x.rows {
                            out[(i, s)] = solved[i];
                        }
                    }
                    out
                };
                let mut term = solve(&rep.w);
                first += term.frobenius().powi(2);
                let mut sum = term.clone();
                for (order, remainder) in remainders.iter_mut().enumerate() {
                    if order > 0 {
                        term = solve(&term.product(&e)).scaled(-C64::ONE);
                        sum.add_scaled(C64::ONE, &term);
                    }
                    let mut apart = rep.t.clone();
                    apart.add_scaled(-C64::ONE, &sum);
                    *remainder += apart.frobenius().powi(2);
                }
            }
            let first = first.sqrt();
            for (order, remainder) in remainders.iter().enumerate() {
                let bound = q.powi(order as i32 + 1) / (1.0 - q) * first + crate::units::rounding(n * n * (order + 2)) * first;
                assert!(remainder.sqrt() <= bound, "at {r:?}, K {order}: the remainder {:.3e} beyond {bound:.3e} (q {q:.3e})", remainder.sqrt());
            }
            if read % 8 == 0 {
                println!("at {r:?}: q {q:.3e}; the remainders after K = 0..3, relative to T⁽⁰⁾: {:?}", remainders.iter().map(|x| format!("{:.2e}", x.sqrt() / first)).collect::<Vec<_>>());
            }
            read += 1;
        }
    }
    println!("{read} atoms read; the largest q {largest:.3e}");
    assert!(read > 0);
}

///   classes, its own photons' and its carried sectors'.
#[test]
fn an_explicit_state_is_p_at_every_frequency_and_its_copies_sectors() {
    let mut stream = crate::random::Stream::new(11, crate::random::Purpose::Initial, 0, 0);
    let mut u = || stream.uniform();
    let (mut explicit_seen, mut turning_seen) = (0, 0);
    for _ in 0..40 {
        let ng = 2 + (3.0 * u()) as usize;
        let (b0, b1) = (2 + (2.0 * u()) as usize, 2 + (2.0 * u()) as usize);
        let ne = b0 + b1;
        let grounds: Vec<f64> = (0..ng).map(|_| 60.0 * (u() - 0.5)).collect();
        let excited: Vec<f64> = (0..ne).map(|e| if e < b0 { 0.0 } else { 300.0 } + 17.6 * e as f64).collect();
        let lands: Vec<usize> = (0..ne).map(|e| e % ng).collect();
        let mut lv = levels(&grounds, &excited, 38.1, &lands);
        for (lo, hi) in [(0, b0), (b0, ne)] {
            for x in lo..hi {
                for y in x + 1..hi {
                    let v = C64::new(3.0 * (u() - 0.5), 3.0 * (u() - 0.5));
                    lv.excited[(x, y)] = v;
                    lv.excited[(y, x)] = v.conj();
                }
            }
        }
        let blocks = vec![(0..b0).collect::<Vec<_>>(), (b0..ne).collect::<Vec<_>>()];
        let count = 2 + (2.0 * u()) as usize;
        let mut classes: Vec<Class> = vec![];
        for _ in 0..count {
            let mut entries: Vec<(usize, usize, C64)> = vec![];
            for e in 0..ne {
                for g in 0..ng {
                    if u() < 0.6 {
                        entries.push((e, g, C64::new(60.0 * u(), 20.0 * (u() - 0.5))));
                    }
                }
            }
            let nu = 80.0 * (u() - 0.5) - 17.6 * (b0 as f64) * u();
            classes.push(class(nu, ng, ne, &entries));
        }
        let velocity = [u() - 0.5, u() - 0.5, u() - 0.5];
        let closed = closure(&lv, &classes, &blocks, velocity, &Kept::default());
        let fold = &closed.fold;
        let place_of = |e: usize| fold.explicit.iter().position(|&x| x == e).map(|i| ng + i);
        explicit_seen += usize::from(!fold.explicit.is_empty());
        for rep in &fold.replicas {
            for &e in &rep.q {
                assert!(place_of(e).is_none_or(|p| fold.members[rep.group].iter().all(|&m| m != p)), "Q at {} holds explicit state {e} of its group", rep.frequency);
            }
        }
        for p in (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5) {
            let group = fold.groups[p];
            for (b, ..) in fold.block_parts(&blocks, p, None).1 {
                assert!(blocks[b].iter().all(|&e| place_of(e).is_none_or(|x| fold.members[group].iter().all(|&m| m != x))), "a part on block {b}, explicit in eigenstate {p}'s group");
            }
        }
        // Every turning entry of an explicit state is one sector's link (folded or carried), none a replica's feed.
        let slow = slow_of(&blocks, &closed.kept, &classes, &lv);
        // Criterion three's classes: the run's alone.
        let emission: [Mat; 3] = std::array::from_fn(|a| Mat::from_fn(ng, ng + ne, |g, x| if a == 0 && x >= ng && lands[x - ng] == g { C64::real(38.1_f64.sqrt()) } else { C64::ZERO }));
        let keys = |f: &Fold| f.photons(&emission, &(0..f.size()).collect::<Vec<_>>()).0.iter().map(|c| (c.landing, c.frequency, c.scale)).collect::<Vec<_>>();
        let (still, left_out) = (keys(&folded(&lv, &classes, &slow)), keys(&sectored(&lv, &classes, &slow, false)));
        assert!(left_out.iter().all(|k| still.iter().any(|x| x.0 == k.0 && equal(x.1 - k.1, x.2 + k.2))), "a left-out sector makes no photon class: {left_out:?} against {still:?}");
        let like: Vec<usize> = (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
        let (images, _) = fold.photons(&emission, &like);
        assert!(start_cross_terms(fold, &emission).len() <= images.len() * images.len(), "the kept pairs are of the run's classes");
        let n = fold.size();
        let label = |s: usize| if s < ng { s } else { ng + fold.explicit[s - ng] };
        for x in ng..n {
            for y in 0..n {
                if slow.groups[x] != slow.groups[y] || x == y {
                    continue;
                }
                let entries: Vec<(f64, f64, (Option<usize>, Option<usize>))> = if y >= ng {
                    if lv.excited[(fold.explicit[x - ng], fold.explicit[y - ng])] != C64::ZERO { vec![(slow.frames[y] - slow.frames[x], slow.sizes[x] + slow.sizes[y], (None, None))] } else { vec![] }
                } else {
                    classes.iter().enumerate().filter(|(_, c)| c.beams[0].coupling[(fold.explicit[x - ng], y)] != C64::ZERO).map(|(k, c)| (c.nu + slow.frames[y] - slow.frames[x], c.nu.abs() + slow.sizes[x] + slow.sizes[y], (None, Some(k)))).collect()
                };
                for (beat, scale, cls) in entries.into_iter().filter(|e| !equal(e.0, e.1)) {
                    let link = Link { a: label(x), b: label(y), classes: cls };
                    let holding = fold.sectors.iter().chain(&fold.carried).filter(|s| equal(s.frequency - beat, s.scale + scale) && s.links.iter().any(|l| l.0 == link)).count();
                    assert_eq!(holding, 1, "the turning entry {link:?} at {beat} is one sector's link");
                    turning_seen += 1;
                }
            }
        }
    }
    println!("{explicit_seen} of 40 closures with explicit states, {turning_seen} turning entries of explicit states, each one sector's link");
    assert!(explicit_seen > 0 && turning_seen > 0, "the schemes reach explicit states and their turning entries");
}

/// A cluster's series order is a level of its own (ELIMINATION v12; `Series::holds`, `Model::first_order_holds`): on
/// PGC1D_TILT_2's model of the finish head's refusal -- its first model and its first attempt's keep (the normalization's
/// first order, the F′ = 3 block for both PGC lights from F = 2: a cluster of F = 2 and seven explicit states, its
/// drives up to 83 rad/µs) --
/// - the model exists, the series' ratio (`Fold::series_ratio`: the step's spectral radius) below one at every prior atom
///   on that cluster (about 0.08, where with the explicit states' copies in Q it was 2.3–2.4 and their bound ‖E‖/d 6.5),
///   and the series' own terms fall within its powers;
/// - per cluster, its first order a level it can take exactly where its own series converges;
/// - the evaluation at order 0 is within the share of the definition on every eigenstate, ground-like and explicit-like,
///   and order 1 nearer still.
#[test]
fn a_cluster_s_series_order_is_its_own_level_and_tilt2_s_explicit_cluster_converges() {
    use super::super::{Component, Model, Unit};
    let req = request("PGC1D_TILT_2");
    let light = crate::field::Light::new(&req);
    let initial = crate::solve::components(&req);
    let mut kept = vec![];
    let _ = crate::solve::closed(&req, &light, &mut kept, &initial, req.temperature_uk).expect("the first model");
    kept.push(Unit::Normalization { order: 1 });
    for f in [0, 1] {
        kept.push(Unit::Block { block: 5, component: Component { frequency: f, counter: false }, grounds: 248 });
    }
    let mut model = Model::new(&req, &light, &kept).expect("the model");
    let points = crate::solve::atoms(&model, &req, &light, &initial, req.temperature_uk).expect("atoms");
    model.referenced(&points);
    assert!(model.series.exists(), "the model exists: {:?}", model.series.stop);
    let running = model.running();
    for c in 0..model.clusters.len() {
        assert_eq!(model.series.holds(c), model.series.ratios[c] < 1.0);
        assert_eq!(model.first_order_holds(c), model.series.holds(c), "cluster {c}");
    }
    let k = (0..model.clusters.len()).max_by_key(|&k| model.clusters[k].states.len()).unwrap();
    assert!(model.clusters[k].states.len() == 12 && model.series.holds(k), "the explicit cluster's series converges: {:?}", model.series.ratios);
    let share = super::share();
    let (mut radius, mut worst) = (0.0_f64, [[0.0_f64; 2]; 2]);
    for &(r, _) in &points {
        let amplitudes = light.amplitudes(r);
        let (fold, _, far, left) = model.cluster_fold(k, &amplitudes, true).unwrap();
        let z0 = model.reference_of(k, &far);
        let rho = fold.series_ratio(&z0, Some(&left));
        radius = radius.max(rho);
        // The series' terms, T⁽ᵏ⁺¹⁾[:, s] = −R₀(s)(T⁽ᵏ⁾E)[:, s], twenty steps on: within (2ρ)²⁰ of the first.
        let mut e = fold.generator.clone();
        for y in 0..fold.size() {
            e[(y, y)] -= C64::real(z0[y]);
        }
        for (rep, r0) in fold.replicas.iter().zip(fold.responses_about(&z0, 0)) {
            let (mut t, first) = (r0[0].clone(), r0[0].frobenius());
            for _ in 0..20 {
                let te = t.product(&e);
                let mut next = Mat::zeros(t.rows, t.cols);
                for &s in &fold.members[rep.group] {
                    let column: Vec<C64> = (0..t.rows).map(|x| -te[(x, s)]).collect();
                    let solved = rep.k.resolvent(C64::real(z0[s] + rep.frequency), &column);
                    for x in 0..t.rows {
                        next[(x, s)] = solved[x];
                    }
                }
                t = next;
            }
            assert!(t.frobenius() <= first * (2.0 * rho).powi(20), "the terms fall as the ratio's powers");
        }
        let (definition, whole) = model.definition_fold(k, &amplitudes).unwrap();
        let defined = definition.definitions(&whole).0;
        let orders = model.orders_of(k);
        for (order, evaluated) in [0, 1].map(|series| (series, model.evaluation(&model.plan_at(k, &running, Orders { series, ..orders }), r).0)) {
            for p in 0..definition.size() {
                let x = Fold::evaluation_error(&evaluated, &defined, &z0, &definition.right(p));
                let kind = usize::from(definition.ground_weight(p) < 0.5);
                worst[order][kind] = worst[order][kind].max(x);
            }
        }
    }
    println!("the explicit cluster: series' ratio {radius:.4e} at most; the evaluation's error at order 0 ground-like {:.3e}, explicit-like {:.3e}; at order 1 {:.3e}, {:.3e} (share {share:.4})", worst[0][0], worst[0][1], worst[1][0], worst[1][1]);
    // What a reading costs there, criterion three on the run's classes alone (the measure the module documentation quotes).
    let units = model.candidates();
    let n = model.clusters[k].states.len();
    let (r, v) = points[0];
    let sample = super::super::Sample { cluster: k, at: super::super::Point { r, kappa: v.map(|x| x / model.hbar_over_mass), t: 0.0 }, psi: (0..n).map(|x| if x == 0 { C64::ONE } else { C64::ZERO }).collect() };
    let started = std::time::Instant::now();
    let _ = model.readings(&units, &sample, None);
    let classes = model.classes[k].len();
    println!("a reading in {:.3} s: {classes} photon classes of the cluster, {} pairs at most", started.elapsed().as_secs_f64(), classes * (classes - 1) / 2);
    assert!(radius < 1.0);
    assert!(worst[0][0] <= share && worst[0][1] <= share, "order 0 within the share: {worst:?}");
    assert!(worst[1][0] <= worst[0][0] && worst[1][1] <= worst[0][1], "order 1 nearer: {worst:?}");
}

