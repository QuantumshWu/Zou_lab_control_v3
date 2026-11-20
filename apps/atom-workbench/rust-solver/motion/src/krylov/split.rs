//! The spectral split (FASTREF.md §2.5; `fr.gap_split_sweep`).
//!
//! Every unknown y has the Gershgorin disc D_y = {z : |z − c_y| ≤ R_y}, c_y = A_yy, R_y = Σ_{z≠y} |A_yz|. The union of
//! the discs falls into connected components; by Gershgorin's theorem in its separation form (Varga, *Geršgorin and
//! His Circles*, 2004, Thm 1.6) k discs whose union meets no other disc hold exactly k eigenvalues. Slow = the
//! component that contains 0 (the steady state; the border row's disc is centred at 0): it holds as many eigenvalues
//! as unknowns and is separated from the rest of the spectrum by a gap. Fast = all others.
//!
//! The component is grown breadth first from the discs that contain 0. Candidates are found by a sweep in Im c: two
//! discs meet only if |Im c_y − Im c_z| ≤ R_y + R_z ≤ R_y + max R, so with the discs sorted by Im c a frontier disc
//! is tested only against the window [Im c_y − R_y − max R, Im c_y + R_y + max R] -- the same component as testing all
//! pairs (the prototype asserted equality at every point), with far fewer tests.

use super::row;
use crate::floquet::Sparse;
use crate::linalg::C64;

/// The slow mask (true = slow: in the Gershgorin component of 0) and the number of disc-pair tests made.
pub fn split(a: &Sparse) -> (Vec<bool>, usize) {
    let n = a.rows;
    let mut centre = vec![C64::ZERO; n];
    let mut radius = vec![0.0f64; n];
    for i in 0..n {
        let (c, v) = row(a, i);
        for (&j, &x) in c.iter().zip(v) {
            if j == i {
                centre[i] += x;
            } else {
                radius[i] += x.abs();
            }
        }
    }
    let widest = radius.iter().cloned().fold(0.0, f64::max);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&p, &q| centre[p].im.total_cmp(&centre[q].im));
    let im: Vec<f64> = order.iter().map(|&p| centre[p].im).collect();
    let mut slow: Vec<bool> = (0..n).map(|i| centre[i].abs() <= radius[i]).collect();
    let mut frontier: Vec<usize> = (0..n).filter(|&i| slow[i]).collect();
    let mut tests = 0usize;
    while !frontier.is_empty() {
        let mut next = Vec::new();
        for &i in &frontier {
            let lo = im.partition_point(|&t| t < centre[i].im - radius[i] - widest);
            let hi = im.partition_point(|&t| t <= centre[i].im + radius[i] + widest);
            for &j in &order[lo..hi] {
                if slow[j] {
                    continue;
                }
                tests += 1;
                if (centre[j] - centre[i]).abs() <= radius[j] + radius[i] {
                    slow[j] = true;
                    next.push(j);
                }
            }
        }
        frontier = next;
    }
    (slow, tests)
}
