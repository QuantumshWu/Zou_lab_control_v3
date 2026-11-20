//! The krylov module's named tests: the three zero norms of the root fix (`frx.py`), each level of the route forced
//! by the structure that makes the level before it fail, and a solve against the exact solution.

use super::*;
use crate::linalg::C64;

fn csr(n: usize, entries: &[(usize, usize, f64)]) -> Csr {
    let mut start = vec![0usize; n + 1];
    let (mut column, mut value) = (Vec::new(), Vec::new());
    for i in 0..n {
        for &(r, c, v) in entries {
            if r == i {
                column.push(c);
                value.push(C64::real(v));
            }
        }
        start[i + 1] = column.len();
    }
    Csr { start, column, value }
}

fn system(n: usize, entries: &[(usize, usize, f64)], rhs: &[f64], harmonic: &[usize]) -> System {
    System {
        matrix: csr(n, entries),
        rhs: rhs.iter().map(|&v| C64::real(v)).collect(),
        harmonic: harmonic.to_vec(),
        readouts: Readouts {
            force: Row { index: vec![0], value: vec![C64::ONE] },
            rate: Row::default(),
            populations: (0..n).map(|i| (format!("{i}"), Row { index: vec![i], value: vec![C64::ONE] })).collect(),
        },
    }
}

/// A 2 × 2 system with two harmonics, one slow unknown coupled to one fast one, a nonzero rhs.
fn coupled() -> System {
    system(2, &[(0, 0, 0.5), (0, 1, 1.0), (1, 0, 1.0), (1, 1, 20.0)], &[1.0, 0.0], &[0, 1])
}

/// β = ‖W b‖ = 0: x = 0 exactly, certified, no iteration (frx.py fix 1).
#[test]
fn krylov_zero_rhs_returns_zero_certified() {
    let mut s = coupled();
    s.rhs = vec![C64::ZERO; 2];
    let out = solve(&s).unwrap();
    assert!(out.certified);
    assert!(out.x.iter().all(|&v| v == C64::ZERO));
    let run = out.route[0].run.as_ref().unwrap();
    assert_eq!((run.stop, run.iterations, run.zeta), (Stop::ZeroRhs, 0, 0.0));
    assert_eq!(out.route.len(), 1);
    assert!(out.readings.populations.iter().all(|(_, v)| *v == 0.0));
}

/// h_{k+1,k} = 0 (frx.py fix 2): a diagonal A with no disc containing 0 is all fast, P = D_f = A; with powers of two
/// and b along one unknown every operation is exact, the first Arnoldi step closes the Krylov space and the iteration
/// ends there with the exact solution, certified.
#[test]
fn krylov_arnoldi_breakdown_ends_exact() {
    let s = system(2, &[(0, 0, 2.0), (1, 1, 4.0)], &[4.0, 0.0], &[0, 0]);
    let out = solve(&s).unwrap();
    let run = out.route[0].run.as_ref().unwrap();
    assert_eq!((run.stop, run.iterations), (Stop::Breakdown, 1));
    assert!(out.certified);
    assert_eq!(out.x, vec![C64::real(2.0), C64::ZERO]);
}

/// A zero row (frx.py fix 3): b_i = 0 gives weight 1 and a row that counts 0 in the backward error; b_i ≠ 0 is an
/// inconsistent system, refused with the row.
#[test]
fn krylov_zero_row_weight_one_or_inconsistent() {
    let a = csr(2, &[(1, 0, 1.0), (1, 1, 2.0)]);
    let rows = Rows::new(&a);
    let x = [C64::real(1.0), C64::real(1.0)];
    let b = [C64::ZERO, C64::real(3.0)];
    assert!(rows.check(&b).is_ok());
    assert_eq!(rows.backward(&x, &b, &[C64::ZERO, C64::ZERO]), (0.0, 0.0));
    let s = system(2, &[(1, 0, 1.0), (1, 1, 2.0)], &[1.0, 3.0], &[0, 0]);
    assert_eq!(solve(&s).unwrap_err(), Inconsistent { rows: vec![0] });
}

/// Level 2 forced: both unknowns are slow (row 0's disc holds 0, row 1's |z − 1| ≤ 1 meets it), the harmonic
/// block of unknown 0 is structurally empty (A_00 is not stored) while S = A is regular: the harmonic level cannot be
/// factorised, the exact S solves, certified.
#[test]
fn krylov_route_falls_back_to_exact_schur() {
    let s = system(2, &[(0, 1, 1.0), (1, 0, 1.0), (1, 1, 1.0)], &[1.0, 2.0], &[0, 1]);
    let out = solve(&s).unwrap();
    let levels: Vec<Level> = out.route.iter().map(|a| a.level).collect();
    assert_eq!(levels, vec![Level::Harmonic, Level::ExactSchur]);
    assert!(out.route[0].run.is_none() && out.route[0].cross);
    assert!(out.certified);
    // x0 + x1 = 2, x1 = 1
    assert!((out.x[0] - C64::real(1.0)).abs() <= 16.0 * f64::EPSILON);
    assert!((out.x[1] - C64::real(1.0)).abs() <= 16.0 * f64::EPSILON);
}

/// Level 3 forced: slow = {0} (its disc |z − 0.2| ≤ 2 holds 0; the fast discs |z − 10| ≤ 2 are apart), the Schur
/// complement S = 0.2 − 1/10 − 1/10 is exactly 0 (a singular pivot) while A is regular (det 1.8); one harmonic, so
/// the exact S is the same operator and is skipped: the direct LU solves, certified.
#[test]
fn krylov_route_falls_back_to_direct() {
    let e = [(0, 0, 0.2), (0, 1, 1.0), (0, 2, 1.0), (1, 0, 1.0), (1, 1, 10.0), (1, 2, 1.0), (2, 0, 1.0), (2, 1, 1.0), (2, 2, 10.0)];
    let s = system(3, &e, &[1.0, 0.0, 0.0], &[0, 0, 0]);
    let out = solve(&s).unwrap();
    let levels: Vec<Level> = out.route.iter().map(|a| a.level).collect();
    assert_eq!(levels, vec![Level::Harmonic, Level::Direct]);
    assert!(!out.route[0].cross);
    assert!(out.route[0].run.as_ref().map_or(true, |r| !r.certified));
    assert!(out.certified);
    // A⁻¹ e₁ = (99, −9, −9) / 1.8
    for (v, want) in out.x.iter().zip([99.0 / 1.8, -9.0 / 1.8, -9.0 / 1.8]) {
        assert!((*v - C64::real(want)).abs() <= 64.0 * f64::EPSILON * 55.0, "{v:?} vs {want}");
    }
}

/// A slow-fast system across two harmonics, solved at level 1 and certified; its residual is at rounding against
/// the exact solution by Cramer's rule.
#[test]
fn krylov_harmonic_level_certified() {
    let s = coupled();
    let out = solve(&s).unwrap();
    assert_eq!(out.route.len(), 1);
    assert_eq!(out.route[0].level, Level::Harmonic);
    assert!(out.certified);
    // det = 0.5·20 − 1 = 9: x = (20, −1) / 9
    assert!((out.x[0] - C64::real(20.0 / 9.0)).abs() <= 16.0 * f64::EPSILON);
    assert!((out.x[1] - C64::real(-1.0 / 9.0)).abs() <= 16.0 * f64::EPSILON);
}
