//! The solve and its fallback chain (FASTREF.md §§ 5, 8.2–8.3; `fr.solve` with `frx.py`): the steady state ρ,
//! certified (ζ ≤ 1) or solved again by the next route. Only ρ: the check reads F, R and the populations; the force's
//! response y serves the diffusion alone, which the check does not judge (the user: the diffusion is out of the
//! model's gate), so it is not solved.
//!
//! One GMRES (`gmres`) under three preconditioners, tried in order until one is certified:
//! 1. `Route::Krylov` -- the Gershgorin split, D_f for the fast-fast block, S on its harmonic-diagonal blocks;
//! 2. `Route::ExactSchur` -- the same split, S exact (one block); tried only if S has entries outside the harmonic
//!    blocks (otherwise it is route 1's operator: one harmonic, M = 0);
//! 3. `Route::Direct` -- every unknown slow, one block: P = A, the direct sparse LU (GMRES's first iterate is A⁻¹b; it
//!    then refines to the floor and is certified like the others).
//!
//! All three solve the same system to the same stop and certificate; the route only changes the work. The result is
//! the first certified attempt's iterate; if none is certified, the attempt of least ζ, reported not certified. The
//! record kept 96 of 96 solves certified on route 1 (FASTREF.md §8).

use super::gmres::{gmres, Outcome, Rows};
use super::precond::Schur;
use super::split::split;
use super::Inconsistent;
use crate::floquet::System;
use crate::linalg::C64;
use std::time::Instant;

/// The route a solve took.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Route {
    /// GMRES on the harmonic-block Schur preconditioner, certified.
    Krylov,
    /// The exact Schur complement.
    ExactSchur,
    /// Direct sparse LU of the whole system.
    Direct,
}

/// One attempt: the route, its preconditioner (slow unknowns, blocks, whether S had entries outside the blocks,
/// faer's error when a block could not be factorised), the GMRES run (its `x` moved into `Solved` when accepted), the
/// clock (the preconditioner's construction and factorisation, the iteration).
#[derive(Clone, Debug, PartialEq)]
pub struct Attempt {
    pub route: Route,
    pub slow: usize,
    pub blocks: usize,
    pub cross: bool,
    pub unfactorised: Option<String>,
    pub run: Option<Outcome>,
    pub seconds_setup: f64,
    pub seconds_iterate: f64,
}

/// A solved system: ρ on the system's elements (the multiplier dropped); the route of the accepted solution; the
/// GMRES rounds of every attempt; the accepted solution's certificate ζ and whether it is certified; every attempt;
/// the split (slow unknowns, disc-pair tests, seconds).
#[derive(Clone, Debug, PartialEq)]
pub struct Solved {
    pub rho: Vec<C64>,
    pub route: Route,
    pub rounds: usize,
    /// The certificate ζ of the accepted solution.
    pub certificate: f64,
    pub certified: bool,
    pub attempts: Vec<Attempt>,
    pub slow: usize,
    pub split_tests: usize,
    pub seconds_split: f64,
}

/// fr.solve for ρ: Bd [ρ; μ] = [0; 1] by the route above; an inconsistent system (a zero row with b_i ≠ 0) is refused.
pub fn solve(system: &System) -> Result<Solved, Inconsistent> {
    let a = &system.matrix;
    let n = a.rows;
    let mut b = vec![C64::ZERO; n];
    b[n - 1] = C64::ONE;
    let rows = Rows::new(a);
    let inconsistent = rows.inconsistent(&b);
    if !inconsistent.is_empty() {
        return Err(Inconsistent { rows: inconsistent });
    }
    let clock = Instant::now();
    let (slow, split_tests) = split(a);
    let seconds_split = clock.elapsed().as_secs_f64();
    let one = vec![0usize; n];
    let everything = vec![true; n];
    let mut attempts: Vec<Attempt> = Vec::new();
    let mut accepted: Option<usize> = None;
    let mut cross = true;
    for route in [Route::Krylov, Route::ExactSchur, Route::Direct] {
        // the exact S is route 1's operator when S has no entry outside the harmonic blocks
        if route == Route::ExactSchur && !cross {
            continue;
        }
        let (mask, group) = match route {
            Route::Krylov => (&slow[..], &system.harmonic[..]),
            Route::ExactSchur => (&slow[..], &one[..]),
            Route::Direct => (&everything[..], &one[..]),
        };
        let clock = Instant::now();
        let p = Schur::new(a, mask, group);
        let mut attempt = Attempt {
            route,
            slow: 0,
            blocks: 0,
            cross: false,
            unfactorised: None,
            run: None,
            seconds_setup: clock.elapsed().as_secs_f64(),
            seconds_iterate: 0.0,
        };
        match p {
            Err(u) => {
                attempt.cross = u.cross;
                attempt.unfactorised = Some(u.reason);
            }
            Ok(p) => {
                (attempt.slow, attempt.blocks) = p.shape();
                attempt.cross = p.cross;
                let clock = Instant::now();
                let run = gmres(a, &b, &p, &rows);
                attempt.seconds_iterate = clock.elapsed().as_secs_f64();
                attempt.run = Some(run);
            }
        }
        if route == Route::Krylov {
            cross = attempt.cross;
        }
        let certified = attempt.run.as_ref().is_some_and(|r| r.certified);
        attempts.push(attempt);
        if certified {
            accepted = Some(attempts.len() - 1);
            break;
        }
    }
    let zeta = |t: &Attempt| t.run.as_ref().map_or(f64::INFINITY, |r| r.zeta);
    let pick = accepted.or_else(|| {
        (0..attempts.len()).filter(|&k| attempts[k].run.is_some()).min_by(|&p, &q| zeta(&attempts[p]).total_cmp(&zeta(&attempts[q])))
    });
    let rounds = attempts.iter().filter_map(|t| t.run.as_ref()).map(|r| r.iterations).sum();
    let (rho, route, certificate) = match pick {
        Some(k) => {
            let route = attempts[k].route;
            let run = attempts[k].run.as_mut().unwrap();
            let x = std::mem::take(&mut run.x);
            (x[..n - 1].to_vec(), route, run.zeta)
        }
        None => (vec![C64::ZERO; n - 1], Route::Direct, f64::INFINITY),
    };
    Ok(Solved {
        rho,
        route,
        rounds,
        certificate,
        certified: accepted.is_some(),
        attempts,
        slow: slow.iter().filter(|&&s| s).count(),
        split_tests,
        seconds_split,
    })
}
