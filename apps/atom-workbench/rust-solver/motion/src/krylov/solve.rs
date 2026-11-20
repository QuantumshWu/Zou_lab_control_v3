//! The solve and its route (FASTREF.md §§ 5, 8.2–8.3; `fr.solve` with `frx.py`).
//!
//! One GMRES (`gmres`) under three preconditioners, tried in order until one is certified:
//! 1. `Level::Harmonic` -- the Gershgorin split, D_f for the fast-fast block, S on its harmonic-diagonal blocks;
//! 2. `Level::ExactSchur` -- the same split, S exact (one block); tried only if S has entries outside the harmonic
//!    blocks (otherwise it is level 1's operator: one harmonic, M = 0);
//! 3. `Level::Direct` -- every unknown slow, one block: P = A, the direct sparse LU (GMRES's first iterate is
//!    A⁻¹ b; it then refines to the floor and is certified like the others).
//!
//! All three solve the same system to the same stop and certificate; the route only changes the work. The result is
//! the first certified attempt's iterate; if none is certified, the attempt of least ζ, reported as not certified.
//! The record kept 96 of 96 solves certified at level 1 (FASTREF.md §8).

use super::gmres::{gmres, Outcome, Rows};
use super::precond::Schur;
use super::split::split;
use super::{Inconsistent, Readings, System};
use crate::linalg::C64;
use std::time::Instant;

/// A preconditioner level of the route.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Level {
    Harmonic,
    ExactSchur,
    Direct,
}

/// One attempt of the route: its level, the slow unknowns and blocks of its preconditioner, whether S had entries
/// outside the blocks, the GMRES run (`None` when a block's LU failed; its `x` is moved into `Solved`), and its clock
/// (the preconditioner's construction and factorisation, the iteration).
#[derive(Clone, Debug)]
pub struct Attempt {
    pub level: Level,
    pub slow: usize,
    pub blocks: usize,
    pub cross: bool,
    pub run: Option<Outcome>,
    pub seconds_setup: f64,
    pub seconds_iterate: f64,
}

/// The solve: the solution, its readouts, whether it is certified, the route, the split (slow unknowns, disc-pair
/// tests, seconds).
#[derive(Clone, Debug)]
pub struct Solved {
    pub x: Vec<C64>,
    pub readings: Readings,
    pub certified: bool,
    pub route: Vec<Attempt>,
    pub slow: usize,
    pub split_tests: usize,
    pub seconds_split: f64,
}

/// fr.solve: the system's solution to the rounding floor, certified, by the route above; an inconsistent system
/// (a zero row with b_i ≠ 0) is refused.
pub fn solve(system: &System) -> Result<Solved, Inconsistent> {
    let a = &system.matrix;
    let b = &system.rhs;
    let n = a.dim();
    let rows = Rows::new(a);
    rows.check(b)?;
    let clock = Instant::now();
    let (slow, split_tests) = split(a);
    let seconds_split = clock.elapsed().as_secs_f64();
    let one = vec![0usize; n];
    let everything = vec![true; n];
    let mut route: Vec<Attempt> = Vec::new();
    let mut levels = vec![Level::Harmonic, Level::ExactSchur, Level::Direct];
    let mut chosen: Option<usize> = None;
    while !levels.is_empty() {
        let level = levels.remove(0);
        let (mask, group) = match level {
            Level::Harmonic => (&slow, &system.harmonic),
            Level::ExactSchur => (&slow, &one),
            Level::Direct => (&everything, &one),
        };
        let clock = Instant::now();
        let p = Schur::new(a, mask, group);
        let seconds_setup = clock.elapsed().as_secs_f64();
        let cross = match &p {
            Ok(p) => p.cross,
            Err(u) => u.cross,
        };
        if level == Level::Harmonic && !cross {
            levels.retain(|&l| l != Level::ExactSchur);
        }
        let mut attempt = Attempt {
            level,
            slow: mask.iter().filter(|&&s| s).count(),
            blocks: 0,
            cross,
            run: None,
            seconds_setup,
            seconds_iterate: 0.0,
        };
        if let Ok(p) = p {
            attempt.blocks = p.shape().1;
            let clock = Instant::now();
            let run = gmres(a, b, &p, &rows);
            attempt.seconds_iterate = clock.elapsed().as_secs_f64();
            let certified = run.certified;
            attempt.run = Some(run);
            route.push(attempt);
            if certified {
                chosen = Some(route.len() - 1);
                break;
            }
        } else {
            route.push(attempt);
        }
    }
    let pick = chosen.or_else(|| {
        (0..route.len())
            .filter(|&k| route[k].run.is_some())
            .min_by(|&p, &q| zeta(&route[p]).total_cmp(&zeta(&route[q])))
    });
    let x = match pick {
        Some(k) => std::mem::take(&mut route[k].run.as_mut().unwrap().x),
        None => vec![C64::ZERO; n],
    };
    for attempt in route.iter_mut() {
        if let Some(run) = attempt.run.as_mut() {
            run.x = Vec::new();
        }
    }
    let readings = system.readouts.read(&x);
    Ok(Solved {
        x,
        readings,
        certified: chosen.is_some(),
        route,
        slow: slow.iter().filter(|&&s| s).count(),
        split_tests,
        seconds_split,
    })
}

fn zeta(attempt: &Attempt) -> f64 {
    attempt.run.as_ref().map_or(f64::INFINITY, |r| r.zeta)
}
