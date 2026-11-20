//! The krylov module's named tests: the three zero norms of the root fix (`frx.py`), each route forced by the
//! structure that makes the one before it fail, and a slow-fast system across two harmonics solved on route 1.

use super::*;
use crate::floquet::{Elements, Sparse, System};
use crate::linalg::C64;

fn sparse(n: usize, entries: &[(usize, usize, f64)]) -> Sparse {
    let mut indptr = vec![0usize; n + 1];
    let (mut indices, mut values) = (Vec::new(), Vec::new());
    for i in 0..n {
        for &(r, c, v) in entries {
            if r == i {
                indices.push(c);
                values.push(C64::real(v));
            }
        }
        indptr[i + 1] = indices.len();
    }
    Sparse { rows: n, cols: n, indptr, indices, values }
}

/// A bordered system: `matrix` over the elements and the multiplier (the last unknown), τ (K1, K2 are not read).
fn system(n: usize, entries: &[(usize, usize, f64)], trace: &[f64], harmonic: &[usize]) -> System {
    System {
        elements: Elements::default(),
        matrix: sparse(n, entries),
        k1: sparse(n - 1, &[]),
        k2: sparse(n - 1, &[]),
        trace: trace.iter().map(|&t| C64::real(t)).collect(),
        harmonic: harmonic.to_vec(),
    }
}

fn close(a: &[C64], b: &[f64], tolerance: f64) -> bool {
    a.iter().zip(b).all(|(p, &q)| (*p - C64::real(q)).abs() <= tolerance)
}

/// A slow-fast system across two harmonics: element 0 (the population, slow) coupled to element 1 (fast, its disc
/// |z − 20| ≤ 1 apart from the slow discs |z − 0.5| ≤ 2, |z| ≤ 1), the border.
fn coupled() -> System {
    let e = [(0, 0, 0.5), (0, 1, 1.0), (0, 2, 1.0), (1, 0, 1.0), (1, 1, 20.0), (2, 0, 1.0)];
    system(3, &e, &[1.0, 0.0], &[0, 1, 0])
}

/// Route 1 on a slow-fast system with two harmonics: certified; ρ = (1, −1/20) (row 2: ρ0 = 1; row 1: ρ0 + 20ρ1 = 0).
#[test]
fn krylov_harmonic_route_certified() {
    let out = solve(&coupled()).unwrap();
    assert_eq!(out.route, Route::Krylov);
    assert!(out.certified);
    assert!(close(&out.rho, &[1.0, -0.05], 8.0 * f64::EPSILON));
}

/// β = ‖W b‖ = 0 (frx.py fix 1): x = 0 exactly, certified, no iteration.
#[test]
fn krylov_zero_rhs_returns_zero_certified() {
    let s = coupled();
    let a = &s.matrix;
    let (slow, _) = split(a);
    let p = Schur::new(a, &slow, &s.harmonic).unwrap();
    let run = gmres(a, &[C64::ZERO; 3], &p, &Rows::new(a));
    assert_eq!((run.stop, run.iterations, run.zeta, run.certified), (Stop::ZeroRhs, 0, 0.0, true));
    assert!(run.x.iter().all(|&v| v == C64::ZERO));
}

/// h_{k+1,k} = 0 (frx.py fix 2): a diagonal A with no disc containing 0 is all fast, P = D_f = A; with powers of two
/// and b along one unknown every operation is exact, the first Arnoldi step closes the Krylov space and the iteration
/// ends there with the exact solution, certified.
#[test]
fn krylov_arnoldi_breakdown_ends_exact() {
    let a = sparse(2, &[(0, 0, 2.0), (1, 1, 4.0)]);
    let rows = Rows::new(&a);
    let (slow, _) = split(&a);
    assert_eq!(slow, vec![false, false]);
    let p = Schur::new(&a, &slow, &[0, 0]).unwrap();
    let run = gmres(&a, &[C64::real(4.0), C64::ZERO], &p, &rows);
    assert_eq!((run.stop, run.iterations, run.certified), (Stop::Breakdown, 1, true));
    assert_eq!(run.x, vec![C64::real(2.0), C64::ZERO]);
}

/// A zero row (frx.py fix 3): b_i = 0 gives weight 1 and a row that counts 0 in the backward error; b_i ≠ 0 is an
/// inconsistent system, refused with its row (here ρ's: the border row of a system whose τ is empty reads 0 = 1).
#[test]
fn krylov_zero_row_weight_one_or_inconsistent() {
    let a = sparse(2, &[(1, 0, 1.0), (1, 1, 2.0)]);
    let rows = Rows::new(&a);
    let x = [C64::real(1.0), C64::real(1.0)];
    let b = [C64::ZERO, C64::real(3.0)];
    assert!(rows.inconsistent(&b).is_empty());
    assert_eq!(rows.backward(&x, &b, &[C64::ZERO, C64::ZERO]), (0.0, 0.0));
    let s = system(2, &[(0, 0, 1.0)], &[0.0], &[0, 0]);
    assert_eq!(solve(&s).unwrap_err(), Inconsistent { rows: vec![1] });
}

/// Route 2 forced: elements 0, 1 are both populations, all unknowns slow (every disc holds 0); element 1's harmonic
/// block is structurally empty (A_11 not stored) while S = A is regular (det 1): route 1 cannot be factorised, the
/// exact S solves, certified. A x = (0, 0, 1): ρ = (1, 0), the multiplier −1.
#[test]
fn krylov_route_falls_back_to_exact_schur() {
    let e = [(0, 0, 1.0), (0, 1, 1.0), (0, 2, 1.0), (1, 0, 1.0), (1, 2, 1.0), (2, 0, 1.0), (2, 1, 1.0)];
    let s = system(3, &e, &[1.0, 1.0], &[0, 1, 0]);
    let out = solve(&s).unwrap();
    let routes: Vec<Route> = out.attempts.iter().map(|t| t.route).collect();
    assert_eq!(routes, vec![Route::Krylov, Route::ExactSchur]);
    assert!(out.attempts[0].cross);
    assert!(out.attempts[0].unfactorised.is_some() || !out.attempts[0].run.as_ref().unwrap().certified);
    assert_eq!(out.route, Route::ExactSchur);
    assert!(out.certified);
    assert!(close(&out.rho, &[1.0, 0.0], 16.0 * f64::EPSILON));
}

/// Route 3 forced: slow = {0, 1, border} (their discs hold 0), fast = {2, 3} (discs |z − 10| ≤ 2, apart); the Schur
/// complement's (0, 0) entry 0.2 − 1/10 − 1/10 is exactly 0, so S's rows 0 and 1 are both (0, 0, 1): singular, while
/// A is regular (its minor on {0, 2, 3} has det 1.8). One harmonic, so the exact S is route 1's operator and is
/// skipped: the direct LU solves, certified. A x = e_border: ρ = (0, 1, 0, 0).
#[test]
fn krylov_route_falls_back_to_direct() {
    let e = [
        (0, 0, 0.2), (0, 2, 1.0), (0, 3, 1.0), (0, 4, 1.0),
        (1, 4, 1.0),
        (2, 0, 1.0), (2, 2, 10.0), (2, 3, 1.0),
        (3, 0, 1.0), (3, 2, 1.0), (3, 3, 10.0),
        (4, 0, 1.0), (4, 1, 1.0),
    ];
    let s = system(5, &e, &[1.0, 1.0, 0.0, 0.0], &[0; 5]);
    let out = solve(&s).unwrap();
    let rho = &out.attempts;
    assert_eq!(rho.iter().map(|t| t.route).collect::<Vec<_>>(), vec![Route::Krylov, Route::Direct]);
    assert_eq!(rho[0].slow, 3);
    assert!(!rho[0].cross);
    assert!(rho[0].run.as_ref().map_or(true, |r| !r.certified));
    assert_eq!(out.route, Route::Direct);
    assert!(out.certified);
    assert!(close(&out.rho, &[0.0, 1.0, 0.0, 0.0], 64.0 * f64::EPSILON));
}
