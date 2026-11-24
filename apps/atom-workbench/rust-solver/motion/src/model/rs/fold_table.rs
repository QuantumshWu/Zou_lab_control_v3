//! The folded blocks tabled in the tweezer's intensity s (FOLDDEN item 2 and its prototype; P4-TABLE): what a node's
//! fold (`Local::evaluate`) makes of a block's generator that depends on s alone -- its spectrum, and the intensity's
//! T̂_s = V⁻¹(∂G₀/∂s)V for the force -- made once for the plan (`Plan::table`, at its first evaluation) and read at
//! every point.
//!
//! # What depends on s alone
//! A folded block's generator is G₀ = F + s S + Σ_c x_c S_c: F its part no point changes (`Block::fixed`), S the
//! entries of the factors that are the tweezer's intensity (`Plan::intensity`: its unit fold's `Power` and its own
//! outside form's |a_tw|², both s), and per coefficient c of the tweezer's Doppler series (the powers (±δ)^j of rs.rs
//! `values_into`, each in front of the intensity) its entries S_c, x_c = s k_c, k_c the point's coefficient. The
//! Doppler parts are small -- |x_c|‖S_c‖ of 10⁻⁶ rad/µs against the clusters' gaps of 10²–10³ (FOLDDEN) -- and are
//! taken to first order (# The small parts). So is another beam's own intensity f_l = |a_l|² where it varies (a
//! Gaussian beam's outside form's light shift, `Plan::own`): one at its focus, it enters as F + S_l (in F) and the small
//! part x_l S_l, x_l = f_l − 1 -- 10⁻⁶ to 10⁻⁵ where an atom is held in a tweezer at a millimetre beam's centre (tilt0's,
//! tilt2's) --, its T̂_l made at the node on the table's eigenbasis (rs/fold.rs). A block whose generator holds any
//! other varying factor (a beam's amplitude, two beams' product) is not tabled; its node decomposes it as before.
//!
//! # Clusters
//! The block's states are cut into clusters (`clusters`): states whose eigenvalues come within τ = ‖S‖/4 of each other
//! somewhere on s ∈ [0, 1] are one -- each eigenpair standing for the state its spectral projector weighs most, each
//! state taken once --, on a grid fine enough that no eigenvalue moves τ/2 between two of its points (doubled from 8
//! intervals until the eigenvalues' largest move between neighbours says so). A cluster c's spectral projector P_c(s) =
//! Σ_{k∈c} v_k w_k is unique and analytic in s wherever the clusters stay apart, however its own states cross; its basis
//! in the gauge of its own states,
//!   W_c = P_c[:, c],  W̃_c = (P_c[c, c])⁻¹ P_c[c, :]  (W̃_c W_c = 1, W_c W̃_c = P_c),
//! and its block B_c = W̃_c G₀ W_c about a fixed centre c_c (the mean of its states' Re G₀ at s = ½). The clusters found
//! are the hyperfine levels (PGC3D's 24 states: D1 F′ = 1, 2 and D2 F′ = 0…3, of 3, 5, 1, 3, 5, 7).
//!
//! # The table
//! Per piece [a, b] of s the Chebyshev coefficients (`chebyshev`, at the Lobatto points) of W, W̃, the clusters' B_c − c_c
//! and Ŝ = W̃ S W -- W and Ŝ by cluster (`blocked`: each cluster's columns an n × m block, as the products with U take
//! them). A piece holds where every quantity's last two coefficients are within its rounding floor -- n u times
//! the moduli of the products that made it at the samples (`Sample::floors`): the node's own rounding --, the degree
//! then cut to the last coefficient above it. The degree is doubled from 8 while the table still pays (# Where it pays);
//! where the last two no longer fall by half at a doubling, or the halves' by half of the piece's, the samples' own
//! rounding is reached (a plateau: GMD1C's Ŝ at 5 floors) and the piece is taken at it, each quantity's floor raised to
//! its plateau's height; else the piece is halved. Measured (P4-TABLE): the 24-state blocks one piece of degree 8
//! (GMD1C's 7), PGC3D P11's 21 and TET P18's 14 the same, R4's 7-state blocks four pieces of degree 6.
//!
//! # The small parts
//! To first order in x_σ (σ a Doppler coefficient, or a beam's own intensity): Ŝ_σ = W̃ S_σ W, whose cluster blocks go
//! into B_c exactly; the clusters' first-order mixing K_σ (B_c′K − KB_c = −Ŝ_σ,c′c, c′ ≠ c, solved in B's eigenbases) as
//! W₁ = W K_σ and W̃₁ = −K_σ W̃; and T̂'s first order C_σ = Ŝ K_σ − K_σ Ŝ, beside k_σ Ŝ_σ, a Doppler part's own share of
//! ∂G₀/∂s -- together k_σ E_σ, E_σ = s C_σ + Ŝ_σ; a beam's has no share of ∂G₀/∂s, its E_σ = C_σ at x_σ. Each is tabled as the main quantities, to its main counterpart's floor over the largest
//! x the guard lets in; the row [W₁ | W̃₁ | Ŝ_σ's cluster blocks | E_σ] lies as the main one, so that a node adds x_σ
//! (k_σ for E_σ) times its sums onto the main sums. The second order moves an eigenvalue by (x_σ A_σ)² g, A_σ = ‖Ŝ_σ‖/g
//! the part off the clusters' blocks over their least gap g at the piece's samples (`Piece::guards`): a node takes the
//! table where (x_σ A_σ)² is within the least of B's floors over g (`BlockTable::relative`; PGC3D P8 2.5·10⁻¹³, its
//! A_σ ≤ 1.3·10⁻⁹: |x| up to some 400 rad/µs, a hundred times a held atom's), else it decomposes the block as before
//! (`fill`'s None).
//!
//! # At a node (`BlockTable::fill`)
//! The sums at s (T_j(t) by the three-term recurrence, each coefficient row once); each active small part's first order
//! onto them; each cluster's block decomposed exactly (spectrum.rs: about its own centre), U_c; then V = W diag(U_c),
//! V⁻¹ = diag(U_c⁻¹) W̃, the eigenvalues c_c plus the cluster's, T̂_s = diag(U_c⁻¹) Ŝ diag(U_c).
//!
//! # Where it pays
//! A block is tabled only where a node's counted work off the table -- the sums, the small parts', the clusters'
//! decompositions, the products with U -- is below the work of its generator, its spectrum and its T̂'s made at the
//! node (each measured at s = ½), the small parts counted that are active (their coefficients not zero) at the point the
//! plan is first evaluated at (the structure's: the centre at rest): nothing is said about a block's size or a scene.

use std::sync::Arc;

use super::super::spectrum::Spectrum;
use super::{Plan, Values};
use crate::linalg::{self, C64, Lu, Mat};
use crate::units::UNIT_ROUNDOFF;

#[cfg(test)]
thread_local! {
    /// A check's switch (tests alone, the thread's): 0 the tables taken, 1 every block decomposed at the node, 2 the
    /// same with every entry of its generator moved by a unit round-off (the rounding's floor of what follows).
    pub(crate) static MODE: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// The plan's tables, per block (None where it is not tabled, and why).
#[derive(Clone, Debug, Default)]
pub(crate) struct Table {
    pub blocks: Vec<Option<Arc<BlockTable>>>,
    pub why: Vec<(usize, String)>,
    /// The making's wall time (s).
    pub seconds: f64,
}

/// A cluster: its states' positions in the block (sorted), its first column in the clusters' order, its centre c_c.
#[derive(Clone, Debug)]
pub(crate) struct Cluster {
    pub states: Vec<usize>,
    pub at: usize,
    pub centre: f64,
}

/// A small part (module documentation, # The small parts): a Doppler coefficient's, or a beam's own intensity's.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum Small {
    /// The coefficient c: x_c = s k_c.
    Doppler(usize),
    /// The factor f: x_f = f − 1.
    Own(usize),
}

impl Small {
    /// Its x at a node of the intensity s, and the factor its E_σ is taken at (`fill`).
    fn at(self, s: f64, values: &Values) -> (f64, f64) {
        match self {
            Small::Doppler(c) => (s * values.coefficients[c], values.coefficients[c]),
            Small::Own(f) => {
                let x = values.factors[f].0.re - 1.0;
                (x, x)
            }
        }
    }
}

/// One piece of s: its ends, its degrees and coefficients (degree + 1 rows): the main row [W | W̃ | B_c − c_c | Ŝ], the
/// small parts' [W₁ | W̃₁ | Ŝ_σ,cc | E_σ] side by side; the largest tail over its floor (above one: taken at the plateau).
#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub a: f64,
    pub b: f64,
    pub degree: usize,
    pub small_degree: usize,
    main: Vec<C64>,
    smalls: Vec<C64>,
    /// The largest tail over its floor, the quantity of the largest main tail (`main_ranges`' order) and whether the
    /// piece is taken at its plateau: the census's (the tests').
    #[cfg_attr(not(test), allow(dead_code))]
    pub tail: f64,
    #[cfg_attr(not(test), allow(dead_code))]
    pub worst: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    pub stalled: bool,
    /// Per small part its guard on the piece, A_σ: the largest ‖Ŝ_σ‖∞ (off the clusters' blocks) over the clusters'
    /// least gap at the piece's samples.
    pub guards: Vec<f64>,
}

/// A block's table (module documentation).
#[derive(Clone, Debug)]
pub(crate) struct BlockTable {
    pub size: usize,
    pub clusters: Vec<Cluster>,
    /// Σ_c m_c², the clusters' blocks' entries.
    squares: usize,
    pub smalls: Vec<Small>,
    pub pieces: Vec<Piece>,
    /// B's relative floor (the guard's bound on the second order).
    pub relative: f64,
    /// A node's counted work off the table (its active small parts', # Where it pays) and made at the node (complex
    /// multiply-adds): the census's (the tests').
    #[cfg_attr(not(test), allow(dead_code))]
    pub work: (f64, f64),
}

/// The block's generator's parts (module documentation, # What depends on s alone): F, S and the small parts' S_c.
struct Parts {
    f: Mat,
    s: Mat,
    smalls: Vec<(Small, Mat)>,
}

impl Parts {
    /// The parts of block b of the plan; Err where an entry is of a factor other than the intensity's and the beams'
    /// own intensities (these of no Doppler coefficient).
    fn of(plan: &Plan, b: usize) -> Result<Parts, String> {
        let block = &plan.blocks[b];
        let n = block.rows.len();
        let mut parts = Parts { f: Mat { rows: n, cols: n, data: block.fixed.clone() }, s: Mat::zeros(n, n), smalls: Vec::new() };
        for e in &block.rest {
            let (f, scale) = plan.generator_terms[e.term as usize];
            let at = e.row as usize * n + e.col as usize;
            let z = scale * e.value;
            let small = if plan.intensity.contains(&f) {
                if e.coefficient == 0 {
                    parts.s.data[at] += z;
                    continue;
                }
                Small::Doppler(e.coefficient as usize)
            } else if plan.own.contains(&f) && e.coefficient == 0 {
                // One at the beam's focus: F + S_f, and x_f S_f.
                parts.f.data[at] += z;
                Small::Own(f)
            } else {
                return Err(format!("随位置变化的因子（{f}）既不是光镊光强，也不是各光束自己的光强"));
            };
            let i = parts.smalls.iter().position(|x| x.0 == small).unwrap_or_else(|| {
                parts.smalls.push((small, Mat::zeros(n, n)));
                parts.smalls.len() - 1
            });
            parts.smalls[i].1.data[at] += z;
        }
        parts.smalls.sort_by_key(|x| x.0);
        Ok(parts)
    }

    /// G₀ = F + s S.
    fn at(&self, s: f64) -> Mat {
        let mut g = self.f.clone();
        g.add_scaled(C64::real(s), &self.s);
        g
    }
}

/// |a| |b| entrywise (moduli), n × n.
fn moduli_product(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for k in 0..n {
            let x = a[i * n + k];
            if x != 0.0 {
                for j in 0..n {
                    out[i * n + j] += x * b[k * n + j];
                }
            }
        }
    }
    out
}

fn moduli(m: &Mat) -> Vec<f64> {
    m.data.iter().map(|z| z.abs()).collect()
}

/// The eigen indices' clusters at a decomposition: each eigenpair k to the cluster whose states its projector's diagonal
/// weighs most (Re Σ_{i∈c} v_ik w_ki); None where a cluster does not get as many as it has states.
fn assigned(sp: &Spectrum, clusters: &[Cluster]) -> Option<Vec<usize>> {
    let n = sp.vectors.rows;
    let of: Vec<usize> = (0..n)
        .map(|k| {
            let weight = |c: usize| clusters[c].states.iter().map(|&i| (sp.vectors[(i, k)] * sp.inverse[(k, i)]).re).sum::<f64>();
            (0..clusters.len()).max_by(|&a, &b| weight(a).total_cmp(&weight(b))).unwrap()
        })
        .collect();
    clusters.iter().enumerate().all(|(c, x)| of.iter().filter(|&&o| o == c).count() == x.states.len()).then_some(of)
}

/// Each eigenpair's state: the matching by the projectors' diagonals, largest first, each state once.
fn states_of(sp: &Spectrum) -> Vec<usize> {
    let n = sp.vectors.rows;
    let mut cand: Vec<(f64, usize, usize)> = (0..n).flat_map(|k| (0..n).map(move |i| (k, i))).map(|(k, i)| ((sp.vectors[(i, k)] * sp.inverse[(k, i)]).abs(), k, i)).collect();
    cand.sort_by(|a, b| b.0.total_cmp(&a.0));
    let (mut state, mut used) = (vec![usize::MAX; n], vec![false; n]);
    for (_, k, i) in cand {
        if state[k] == usize::MAX && !used[i] {
            state[k] = i;
            used[i] = true;
        }
    }
    state
}

/// The clusters (module documentation, # Clusters) of the generator F + sS on [0, 1]: Err where a decomposition fails.
fn clusters(parts: &Parts) -> Result<Vec<Cluster>, String> {
    let n = parts.f.rows;
    let norm = (0..n).map(|i| (0..n).map(|j| parts.s[(i, j)].abs()).sum::<f64>()).fold(0.0, linalg::larger);
    // A block that does not move with s: its states within their rounding (n u of the decomposition's scale) are one.
    let scale = parts.at(0.5).max_abs();
    let tau = linalg::larger(0.25 * norm, n as f64 * UNIT_ROUNDOFF * scale);
    let mut intervals = 8usize;
    loop {
        let mut root: Vec<usize> = (0..n).collect();
        fn find(root: &mut [usize], mut i: usize) -> usize {
            while root[i] != i {
                root[i] = root[root[i]];
                i = root[i];
            }
            i
        }
        let mut sp = Spectrum::default();
        let mut spectrum_before: Option<Vec<C64>> = None;
        let mut fastest = 0.0_f64;
        for k in 0..=intervals {
            let s = k as f64 / intervals as f64;
            if !sp.decompose(&parts.at(s)) {
                return Err(format!("s = {s} 处得不到谱"));
            }
            let state = states_of(&sp);
            // Each state's eigenvalue (its eigenpair's).
            let mut lambda = vec![C64::ZERO; n];
            for kk in 0..n {
                lambda[state[kk]] = C64::real(sp.centre[kk]) + sp.values[kk];
            }
            for i in 0..n {
                for j in i + 1..n {
                    if (lambda[i] - lambda[j]).abs() < tau {
                        let (a, b) = (find(&mut root, i), find(&mut root, j));
                        root[a] = b;
                    }
                }
            }
            // The move: the two spectra's distance (each eigenvalue to the nearest of the other's, both ways), which no
            // matching of the eigenpairs to the states enters.
            let now: Vec<C64> = (0..n).map(|kk| C64::real(sp.centre[kk]) + sp.values[kk]).collect();
            if let Some(b) = &spectrum_before {
                let near = |x: &C64, set: &[C64]| set.iter().map(|y| (*x - *y).abs()).fold(f64::INFINITY, linalg::smaller);
                for x in &now {
                    fastest = linalg::larger(fastest, near(x, b));
                }
                for x in b {
                    fastest = linalg::larger(fastest, near(x, &now));
                }
            }
            spectrum_before = Some(now);
        }
        if intervals >= 1 << 12 {
            return Err("每一种网格的相邻点之间本征值都移动超过 τ/2".into());
        }
        if !(fastest > 0.5 * tau) {
            let mut by: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
            for i in 0..n {
                let r = find(&mut root, i);
                by.entry(r).or_default().push(i);
            }
            let mut list: Vec<Vec<usize>> = by.into_values().collect();
            list.sort();
            let half = parts.at(0.5);
            let mut at = 0;
            return Ok(list
                .into_iter()
                .map(|states| {
                    let centre = states.iter().map(|&i| half[(i, i)].re).sum::<f64>() / states.len() as f64;
                    let c = Cluster { at, centre, states };
                    at += c.states.len();
                    c
                })
                .collect());
        }
        intervals *= 2;
    }
}

/// One sample of a block at s (module documentation): the main row, the small parts' row, each main quantity's rounding
/// floor (absolute, `main_ranges`' order), the clusters' least gap and each small part's ‖Ŝ_σ‖∞ off the clusters' blocks.
struct Sample {
    main: Vec<C64>,
    small: Vec<C64>,
    floors: Vec<f64>,
    gap: f64,
    norms: Vec<f64>,
}

/// The main quantities' ranges in a row, their floors' order: [W, W̃, B_c per cluster, Ŝ] (a small part's row lies the
/// same: [W₁, W̃₁, Ŝ_σ's cluster blocks, E_σ]).
fn main_ranges(n: usize, clusters: &[Cluster]) -> Vec<(usize, usize)> {
    let n2 = n * n;
    let mut out = vec![(0, n2), (n2, 2 * n2)];
    let mut at = 2 * n2;
    for c in clusters {
        let m = c.states.len();
        out.push((at, at + m * m));
        at += m * m;
    }
    out.push((at, at + n2));
    out
}

/// A block's sample at s (module documentation).
fn sample(parts: &Parts, clusters: &[Cluster], s: f64) -> Result<Sample, String> {
    let n = parts.f.rows;
    let n2 = n * n;
    let g = parts.at(s);
    let mut sp = Spectrum::default();
    if !sp.decompose(&g) {
        return Err(format!("s = {s} 处得不到谱"));
    }
    let of = assigned(&sp, clusters).ok_or_else(|| format!("s = {s} 处各簇的本征对与它们的态对不上"))?;
    let mut w = Mat::zeros(n, n);
    let mut wt = Mat::zeros(n, n);
    let (va, via) = (moduli(&sp.vectors), moduli(&sp.inverse));
    let mut p_scale = 0.0_f64;
    let mut p_moduli = vec![0.0_f64; n2];
    // Each eigen index's eigenvalue, and per cluster its eigen indices.
    let lambda: Vec<C64> = (0..n).map(|k| C64::real(sp.centre[k]) + sp.values[k]).collect();
    let members: Vec<Vec<usize>> = (0..clusters.len()).map(|c| (0..n).filter(|&k| of[k] == c).collect()).collect();
    for (c, cl) in clusters.iter().enumerate() {
        let mut p = Mat::zeros(n, n);
        p_moduli.fill(0.0);
        for &k in &members[c] {
            for i in 0..n {
                let v = sp.vectors[(i, k)];
                let va_ik = va[i * n + k];
                for j in 0..n {
                    p[(i, j)] += v * sp.inverse[(k, j)];
                    p_moduli[i * n + j] += va_ik * via[k * n + j];
                }
            }
        }
        p_scale = p_moduli.iter().copied().fold(p_scale, linalg::larger);
        let lu = Lu::new(&p.block(&cl.states, &cl.states)).ok_or_else(|| format!("s = {s} 处 P_c[c, c] 奇异"))?;
        for (j, &st) in cl.states.iter().enumerate() {
            for i in 0..n {
                w[(i, cl.at + j)] = p[(i, st)];
            }
        }
        for i in 0..n {
            let mut x: Vec<C64> = cl.states.iter().map(|&r| p[(r, i)]).collect();
            lu.solve(&mut x);
            for (j, v) in x.into_iter().enumerate() {
                wt[(cl.at + j, i)] = v;
            }
        }
    }
    let (wa, wta) = (moduli(&w), moduli(&wt));
    let gw = g.product(&w);
    let full = wt.product(&gw);
    let hat = wt.product(&parts.s.product(&w));
    // The floors: n ε times the moduli of the products that made each.
    let ga = moduli(&g);
    let full_scale = moduli_product(&wta, &moduli_product(&ga, &wa, n), n);
    let hat_scale = moduli_product(&wta, &moduli_product(&moduli(&parts.s), &wa, n), n);
    let nf = n as f64;
    let floor = |x: f64| nf * UNIT_ROUNDOFF * x;
    let w_floor = floor(p_scale);
    let mut floors = vec![w_floor, w_floor];
    let mut main = Vec::with_capacity(3 * n2 + n2);
    blocked(&w, clusters, &mut main);
    main.extend_from_slice(&wt.data);
    for cl in clusters {
        let m = cl.states.len();
        let mut top = 0.0_f64;
        for i in 0..m {
            for j in 0..m {
                let (r, cc) = (cl.at + i, cl.at + j);
                let mut v = full[(r, cc)];
                if i == j {
                    v -= C64::real(cl.centre);
                }
                main.push(v);
                top = linalg::larger(top, full_scale[r * n + cc]);
            }
        }
        floors.push(floor(top));
    }
    blocked(&hat, clusters, &mut main);
    floors.push(floor(hat_scale.iter().copied().fold(0.0, linalg::larger)));
    // The clusters' least gap.
    let mut gap = f64::INFINITY;
    for k in 0..n {
        for l in 0..n {
            if of[k] != of[l] {
                gap = linalg::smaller(gap, (lambda[k] - lambda[l]).abs());
            }
        }
    }
    // The small parts' first order: per cluster its eigenbasis U_c = W̃_c V_c, U_c⁻¹ = V⁻¹_c W_c.
    let mut small = Vec::new();
    let mut norms = Vec::new();
    if !parts.smalls.is_empty() {
        let mut u: Vec<Mat> = Vec::with_capacity(clusters.len());
        let mut ui: Vec<Mat> = Vec::with_capacity(clusters.len());
        for (c, cl) in clusters.iter().enumerate() {
            let m = cl.states.len();
            u.push(Mat::from_fn(m, m, |i, j| (0..n).map(|q| wt[(cl.at + i, q)] * sp.vectors[(q, members[c][j])]).sum()));
            ui.push(Mat::from_fn(m, m, |i, j| (0..n).map(|q| sp.inverse[(members[c][i], q)] * w[(q, cl.at + j)]).sum()));
        }
        for (kind, sm) in &parts.smalls {
            let sh = wt.product(&sm.product(&w));
            let mut k = Mat::zeros(n, n);
            let mut off = 0.0_f64;
            for (c1, a1) in clusters.iter().enumerate() {
                for (c2, a2) in clusters.iter().enumerate() {
                    if c1 == c2 {
                        continue;
                    }
                    let (m1, m2) = (a1.states.len(), a2.states.len());
                    let blk = Mat::from_fn(m1, m2, |i, j| sh[(a1.at + i, a2.at + j)]);
                    for i in 0..m1 {
                        off = linalg::larger(off, (0..m2).map(|j| blk[(i, j)].abs()).sum::<f64>());
                    }
                    // K̂ = −(U₁⁻¹ Ŝ U₂) ⊘ (λ₁ᵢ − λ₂ⱼ); K = U₁ K̂ U₂⁻¹.
                    let hb = ui[c1].product(&blk.product(&u[c2]));
                    let kh = Mat::from_fn(m1, m2, |i, j| -hb[(i, j)] * (lambda[members[c1][i]] - lambda[members[c2][j]]).recip());
                    let kb = u[c1].product(&kh.product(&ui[c2]));
                    for i in 0..m1 {
                        for j in 0..m2 {
                            k[(a1.at + i, a2.at + j)] = kb[(i, j)];
                        }
                    }
                }
            }
            norms.push(off);
            let w1 = w.product(&k);
            let wt1 = k.product(&wt).scaled(C64::real(-1.0));
            // E_σ = s C_σ + Ŝ_σ, C_σ = Ŝ K − K Ŝ: the node's T̂ takes k_σ E_σ (x_σ C_σ and k_σ Ŝ_σ, x_σ = s k_σ); of a
            // beam's own intensity E_σ = C_σ at x_σ.
            let mut e = hat.product(&k);
            e.add_scaled(C64::real(-1.0), &k.product(&hat));
            if let Small::Doppler(_) = kind {
                e = e.scaled(C64::real(s));
                e.add_scaled(C64::ONE, &sh);
            }
            blocked(&w1, clusters, &mut small);
            small.extend_from_slice(&wt1.data);
            for cl in clusters {
                let m = cl.states.len();
                for i in 0..m {
                    for j in 0..m {
                        small.push(sh[(cl.at + i, cl.at + j)]);
                    }
                }
            }
            blocked(&e, clusters, &mut small);
        }
    }
    Ok(Sample { main, small, floors, gap, norms })
}

/// An n × n matrix's columns by cluster onto `out`: per cluster its n × m block of columns, row by row (the layout the
/// node's products with U take, `BlockTable::fill`).
fn blocked(m: &Mat, clusters: &[Cluster], out: &mut Vec<C64>) {
    let n = m.rows;
    for cl in clusters {
        for i in 0..n {
            out.extend_from_slice(&m.data[i * n + cl.at..i * n + cl.at + cl.states.len()]);
        }
    }
}

/// Chebyshev coefficients of rows sampled at the Lobatto points t_i = cos(πi/N), i = 0…N (N + 1 rows of `width`).
fn chebyshev(f: &[&[C64]], degree: usize) -> Vec<C64> {
    let width = f[0].len();
    let n = degree.max(1);
    let mut c = vec![C64::ZERO; (degree + 1) * width];
    if degree == 0 {
        c.copy_from_slice(f[0]);
        return c;
    }
    for j in 0..=degree {
        let wj = if j == 0 || j == degree { 0.5 } else { 1.0 };
        for (i, fi) in f.iter().enumerate() {
            let wi = if i == 0 || i == degree { 0.5 } else { 1.0 };
            let x = (std::f64::consts::PI * ((i * j) % (2 * n)) as f64 / n as f64).cos() * wi * wj * 2.0 / n as f64;
            for (o, v) in c[j * width..(j + 1) * width].iter_mut().zip(fi.iter()) {
                *o += *v * x;
            }
        }
    }
    c
}

/// Per quantity (its range in the row and its floor) the largest of its last two coefficients over the floor -- the
/// largest of them and its quantity --, and the degree its coefficients need (the last above the floor).
fn tails(c: &[C64], width: usize, degree: usize, ranges: &[(usize, usize)], floors: &[f64]) -> (f64, usize, usize) {
    let (mut worst, mut which, mut need) = (0.0_f64, 0usize, 0usize);
    for (q, (&(lo, hi), &floor)) in ranges.iter().zip(floors).enumerate() {
        if hi == lo {
            continue;
        }
        let top = |j: usize| (lo..hi).map(|e| c[j * width + e].abs()).fold(0.0, linalg::larger);
        let tail = if degree == 0 { 0.0 } else { linalg::larger(top(degree), top(degree - 1)) };
        let ratio = if floor > 0.0 { tail / floor } else if tail > 0.0 { f64::INFINITY } else { 0.0 };
        if ratio > worst {
            (worst, which) = (ratio, q);
        }
        need = need.max((0..=degree).rev().find(|&j| top(j) > floor).unwrap_or(0));
    }
    (worst, which, need)
}

/// The Lobatto points of [a, b] at degree N, from b down.
fn lobatto(a: f64, b: f64, degree: usize) -> Vec<f64> {
    (0..=degree).map(|i| 0.5 * (a + b) + 0.5 * (b - a) * (std::f64::consts::PI * i as f64 / degree.max(1) as f64).cos()).collect()
}

/// A node's counted work off the table at degree d and small degree e (module documentation, # Where it pays): the
/// sums (real × complex, half a complex multiply-add), each small part's, the clusters' decompositions (measured), the
/// products with U (V, V⁻¹ and T̂'s two).
fn tabled_work(n: usize, squares: usize, smalls: usize, d: usize, e: usize, decompositions: f64) -> f64 {
    let n2 = (n * n) as f64;
    0.5 * (d + 1) as f64 * (3.0 * n2 + squares as f64) + smalls as f64 * (0.5 * (e + 1) as f64 * 4.0 * n2 + 4.0 * n2) + decompositions + 4.0 * (n * squares) as f64
}

/// A fit of a piece (`Fitting::fit`): its samples, coefficients, degree, the largest tail over its floor and the
/// quantity it is of, the degree its coefficients need, the floors, and whether its tail stalled (the samples'
/// rounding: a plateau).
struct Fit {
    samples: Vec<Sample>,
    c: Vec<C64>,
    degree: usize,
    tail: f64,
    worst: usize,
    need: usize,
    floors: Vec<f64>,
    stalled: bool,
}

/// What a block's fits share: its parts, clusters, the main row's ranges, and whether a degree pays.
struct Fitting<'a> {
    parts: &'a Parts,
    clusters: &'a [Cluster],
    ranges: Vec<(usize, usize)>,
    pays: &'a dyn Fn(usize) -> bool,
}

impl Fitting<'_> {
    /// The fit of [a, b] (module documentation, # The table): degree 8, doubled while its tail is above its floor and the
    /// doubled degree still pays, or until a doubling no longer halves its tail; each doubling's samples the last's and
    /// the new points between them.
    fn fit(&self, a: f64, b: f64) -> Result<Fit, String> {
        let mut degree = 8usize;
        let mut samples: Vec<Sample> = lobatto(a, b, degree).into_iter().map(|s| sample(self.parts, self.clusters, s)).collect::<Result<_, _>>()?;
        let mut last = f64::INFINITY;
        loop {
            let floors: Vec<f64> = (0..self.ranges.len()).map(|q| samples.iter().map(|x| x.floors[q]).fold(0.0, linalg::larger)).collect();
            let rows: Vec<&[C64]> = samples.iter().map(|x| x.main.as_slice()).collect();
            let c = chebyshev(&rows, degree);
            let (tail, worst, need) = tails(&c, rows[0].len(), degree, &self.ranges, &floors);
            let stalled = tail > 1.0 && !(tail < 0.5 * last);
            if tail <= 1.0 || stalled || !(self.pays)(2 * degree) {
                return Ok(Fit { samples, c, degree, tail, worst, need, floors, stalled });
            }
            last = tail;
            let next = lobatto(a, b, 2 * degree);
            let mut more = Vec::with_capacity(2 * degree + 1);
            let mut old = samples.into_iter();
            for (i, s) in next.into_iter().enumerate() {
                more.push(if i % 2 == 0 { old.next().expect("the last degree's point") } else { sample(self.parts, self.clusters, s)? });
            }
            samples = more;
            degree *= 2;
        }
    }

    /// The pieces of [a, b] from its fit: the fit itself where it holds or has stalled; else its halves, unless their
    /// tails are not below half its own (the samples' rounding reached by halving: the plateau, taken at the fit).
    fn pieces(&self, a: f64, b: f64, fit: Fit, depth: usize, out: &mut Vec<(f64, f64, Fit)>) -> Result<(), String> {
        if fit.tail <= 1.0 || fit.stalled {
            out.push((a, b, fit));
            return Ok(());
        }
        let m = 0.5 * (a + b);
        if depth >= 20 || !(a < m && m < b) {
            return Err(format!("[{a:.6}, {b:.6}]：不收敛（尾部为下限的 {:.1e}，量 {}）", fit.tail, fit.worst));
        }
        let (left, right) = (self.fit(a, m)?, self.fit(m, b)?);
        if !(linalg::larger(left.tail, right.tail) < 0.5 * fit.tail) {
            out.push((a, b, Fit { stalled: true, ..fit }));
            return Ok(());
        }
        self.pieces(a, m, left, depth + 1, out)?;
        self.pieces(m, b, right, depth + 1, out)
    }
}

impl BlockTable {
    /// Block b's table (module documentation), or why not; `values` the point it is first made at, whose active small
    /// parts (their coefficients not zero there) a node's work is counted with.
    fn new(plan: &Plan, b: usize, values: &Values) -> Result<BlockTable, String> {
        let parts = Parts::of(plan, b)?;
        let n = parts.f.rows;
        let n2 = n * n;
        let clusters = clusters(&parts)?;
        let squares: usize = clusters.iter().map(|c| c.states.len().pow(2)).sum();
        // The work made at the node: the generator's entries, its spectrum, the intensity's T̂ (one per varying factor
        // of the block, as `Local::evaluate` makes them), at s = ½.
        let made = {
            let mut sp = Spectrum::default();
            if !sp.decompose(&parts.at(0.5)) {
                return Err("s = ½ 处得不到谱".into());
            }
            let factors = plan.blocks[b].varying.len();
            plan.blocks[b].rest.len() as f64 + sp.work + 2.0 * factors as f64 * (n * n * n) as f64
        };
        // The clusters' decompositions' work, and the guard's floor (module documentation, # The small parts): the
        // second order's shift of an eigenvalue, (x A)² g at the least gap g, within the least of B's floors -- so
        // (x A)² within that floor over g --, at s = ½.
        let half = sample(&parts, &clusters, 0.5)?;
        let decompositions = {
            let mut work = 0.0;
            for (c, r) in clusters.iter().zip(&main_ranges(n, &clusters)[2..]) {
                let m = c.states.len();
                let block = Mat { rows: m, cols: m, data: half.main[r.0..r.1].to_vec() };
                let mut sp = Spectrum::default();
                if !sp.decompose(&block) {
                    return Err("s = ½ 处有一簇的块得不到谱".into());
                }
                work += sp.work;
            }
            work
        };
        let ranges = main_ranges(n, &clusters);
        let small_width = 3 * n2 + squares;
        let small_ranges: Vec<(usize, usize)> = (0..parts.smalls.len()).flat_map(|q| [(0, n2), (n2, 2 * n2), (2 * n2, 2 * n2 + squares), (2 * n2 + squares, small_width)].map(|(a, b)| (q * small_width + a, q * small_width + b))).collect();
        let k = parts.smalls.len();
        // Active: a Doppler part's coefficient not zero at the point, a beam's own intensity everywhere off its focus --
        // whose T̂ the node makes on the table's eigenbasis.
        let active = parts.smalls.iter().filter(|(c, _)| match c { Small::Doppler(c) => values.coefficients.get(*c).is_some_and(|x| *x != 0.0), Small::Own(_) => true }).count();
        let owns = parts.smalls.iter().filter(|(c, _)| matches!(c, Small::Own(_))).count();
        let decompositions = decompositions + 2.0 * owns as f64 * (n * n * n) as f64;
        let pays = |d: usize| tabled_work(n, squares, active, d, d, decompositions) < made;
        if !pays(0) {
            return Err(format!("不划算：0 阶时 {:.0} 对 {made:.0}", tabled_work(n, squares, active, 0, 0, decompositions)));
        }
        let b_floor = half.floors[2..2 + clusters.len()].iter().copied().fold(f64::INFINITY, linalg::smaller);
        let relative = if half.gap.is_finite() { b_floor / half.gap } else { f64::INFINITY };
        let fitting = Fitting { parts: &parts, clusters: &clusters, ranges: main_ranges(n, &clusters), pays: &pays };
        let mut fits = Vec::new();
        let whole = fitting.fit(0.0, 1.0)?;
        fitting.pieces(0.0, 1.0, whole, 0, &mut fits)?;
        let width = 3 * n2 + squares;
        let mut pieces = Vec::with_capacity(fits.len());
        for (a, bb, fit) in fits {
            let Fit { samples, c, degree, tail, worst, mut need, mut floors, stalled } = fit;
            // At a plateau each quantity's floor is its plateau's height -- its largest coefficient past half the
            // degree, the samples' own rounding -- and the degree is cut where its coefficients reach it.
            if stalled {
                for (q, &(lo, hi)) in ranges.iter().enumerate() {
                    let height = (degree / 2..=degree).flat_map(|j| (lo..hi).map(move |e| j * width + e)).map(|i| c[i].abs()).fold(0.0, linalg::larger);
                    // A plateau above √u of its quantity's largest entry is no rounding (a cluster's projector that
                    // jumps): not tabled.
                    let top = samples.iter().flat_map(|x| x.main[lo..hi].iter().map(|z| z.abs())).fold(0.0, linalg::larger);
                    if height > UNIT_ROUNDOFF.sqrt() * top {
                        return Err(format!("[{a:.6}, {bb:.6}]：量 {q} 停在其最大元的 {:.1e}", height / top));
                    }
                    floors[q] = linalg::larger(floors[q], height);
                }
                need = tails(&c, width, degree, &ranges, &floors).2;
            }
            let mut main = c;
            main.truncate((need + 1) * width);
            // The small parts: each its guard on the piece, each quantity to its main counterpart's floor over the
            // largest x the guard lets in.
            let (mut smalls, mut small_degree, mut small_tail, mut guards) = (Vec::new(), 0usize, 0.0_f64, vec![0.0_f64; k]);
            if k > 0 {
                let gap = samples.iter().map(|x| x.gap).fold(f64::INFINITY, linalg::smaller);
                for (q, g) in guards.iter_mut().enumerate() {
                    *g = samples.iter().map(|x| x.norms[q]).fold(0.0, linalg::larger) / gap;
                }
                let mut targets = Vec::with_capacity(4 * k);
                let b_floor = floors[2..2 + clusters.len()].iter().copied().fold(f64::INFINITY, linalg::smaller);
                let s_floor = floors[ranges.len() - 1];
                for g in &guards {
                    let x = if *g > 0.0 { relative.sqrt() / g } else { 1.0 };
                    targets.extend([linalg::smaller(b_floor, s_floor) / x, floors[0] / x, floors[1] / x, s_floor / x]);
                }
                let rows: Vec<&[C64]> = samples.iter().map(|x| x.small.as_slice()).collect();
                let cs = chebyshev(&rows, degree);
                let (t, _, need_s) = tails(&cs, rows[0].len(), degree, &small_ranges, &targets);
                small_tail = t;
                small_degree = need_s;
                smalls = cs;
                smalls.truncate((small_degree + 1) * small_width * k);
            }
            pieces.push(Piece { a, b: bb, degree: need, small_degree, main, smalls, tail: linalg::larger(tail, small_tail), worst, stalled, guards });
        }
        let d = pieces.iter().map(|p| p.degree).max().unwrap_or(0);
        let e = pieces.iter().map(|p| p.small_degree).max().unwrap_or(0);
        let off = tabled_work(n, squares, active, d, e, decompositions);
        if !(off < made) {
            return Err(format!("不划算：{off:.0} 对 {made:.0}（{d} 阶，小量 {e}）"));
        }
        let smalls = parts.smalls.iter().map(|(c, _)| *c).collect();
        Ok(BlockTable { size: n, clusters, squares, smalls, pieces, relative, work: (off, made) })
    }

    /// The block's spectrum and the intensity's T̂_s (where `gradients`) at the node's values (module documentation,
    /// # At a node), into `spectrum` and `hat`; `s` the intensity there. Returns the counted work, or None where a small
    /// part's x is beyond its guard or a cluster's block has no spectrum (the node then decomposes the block itself).
    pub(crate) fn fill(&self, s: f64, values: &Values, gradients: bool, spectrum: &mut Spectrum, hat: &mut Mat, scratch: &mut Scratch) -> Option<f64> {
        let n = self.size;
        let n2 = n * n;
        let squares = self.squares;
        let width = 3 * n2 + squares;
        let at = self.pieces.partition_point(|p| p.b < s).min(self.pieces.len() - 1);
        let p = &self.pieces[at];
        // The guard: every active small part's second order within B's relative floor.
        let k = self.smalls.len();
        scratch.x.clear();
        for (sm, guard) in self.smalls.iter().zip(&p.guards) {
            let (x, _) = sm.at(s, values);
            if x != 0.0 && !((x * guard).powi(2) <= self.relative) {
                return None;
            }
            scratch.x.push(x);
        }
        let t = (2.0 * s - p.a - p.b) / (p.b - p.a);
        // The main sums [W | W̃ | B | Ŝ] (Ŝ where the gradients are made), each active small part's onto them: x_σ times
        // its [W₁ | W̃₁ | Ŝ_σ,cc], k_σ times its E_σ.
        let lean = 2 * n2 + squares;
        let used = if gradients { width } else { lean };
        let Scratch { x: xs, value, t: tj, factors, spectra, block, temp, u, ui, .. } = scratch;
        chebyshev_at(t, p.degree.max(p.small_degree), tj);
        value.clear();
        value.resize(used, C64::ZERO);
        onto(&p.main, width, 0, used, &tj[..=p.degree], value);
        let mut work = 0.5 * ((p.degree + 1) * used) as f64;
        for (q, sm) in self.smalls.iter().enumerate() {
            let x = xs[q];
            if x == 0.0 {
                continue;
            }
            let base = q * width;
            factors.clear();
            factors.extend(tj[..=p.small_degree].iter().map(|t| x * t));
            onto(&p.smalls[base..], k * width, 0, lean, factors, &mut value[..lean]);
            if gradients {
                let (_, d) = sm.at(s, values);
                factors.clear();
                factors.extend(tj[..=p.small_degree].iter().map(|t| d * t));
                onto(&p.smalls[base..], k * width, lean, width - lean, factors, &mut value[lean..]);
            }
            work += 0.5 * ((p.small_degree + 1) * used) as f64;
        }
        // Each cluster's block decomposed: U, U⁻¹ (kept by cluster), its eigenvalues about the cluster's centre.
        if spectra.len() < self.clusters.len() {
            spectra.resize_with(self.clusters.len(), Spectrum::default);
        }
        spectrum.centre.clear();
        spectrum.values.clear();
        spectrum.leaves.clear();
        u.clear();
        ui.clear();
        let mut off = 2 * n2;
        for (c, cl) in self.clusters.iter().enumerate() {
            let m = cl.states.len();
            (block.rows, block.cols) = (m, m);
            block.data.clear();
            block.data.extend_from_slice(&value[off..off + m * m]);
            off += m * m;
            let sp = &mut spectra[c];
            if !sp.decompose(block) {
                return None;
            }
            work += sp.work;
            u.extend_from_slice(&sp.vectors.data);
            ui.extend_from_slice(&sp.inverse.data);
            for i in 0..m {
                spectrum.centre.push(cl.centre + sp.centre[i]);
                spectrum.values.push(sp.values[i]);
            }
            spectrum.leaves.push(m);
        }
        // V = W U (each cluster's columns W_c U_c), V⁻¹ = U⁻¹ W̃ (its rows U_c⁻¹ W̃_c), T̂ = U⁻¹ (Ŝ U): the product kernel's.
        for mat in [&mut spectrum.vectors, &mut spectrum.inverse] {
            (mat.rows, mat.cols) = (n, n);
            mat.data.clear();
            mat.data.resize(n2, C64::ZERO);
        }
        let (wb, rest) = value.split_at(n2);
        let wt = &rest[..n2];
        temp.clear();
        temp.resize(n2, C64::ZERO);
        let (mut uo, mut bo) = (0, 0);
        for cl in &self.clusters {
            let (m, start) = (cl.states.len(), cl.at);
            let (uc, uic) = (&u[uo..uo + m * m], &ui[uo..uo + m * m]);
            linalg::fused_product(&wb[bo..bo + n * m], uc, (n, m, m), &mut temp[..n * m]);
            for i in 0..n {
                spectrum.vectors.data[i * n + start..i * n + start + m].copy_from_slice(&temp[i * m..(i + 1) * m]);
            }
            linalg::fused_product(uic, &wt[start * n..(start + m) * n], (m, m, n), &mut spectrum.inverse.data[start * n..(start + m) * n]);
            uo += m * m;
            bo += n * m;
        }
        work += (2 * n * squares) as f64;
        if gradients {
            let sb = &value[lean..width];
            (hat.rows, hat.cols) = (n, n);
            hat.data.clear();
            hat.data.resize(n2, C64::ZERO);
            block.data.clear();
            block.data.resize(n * self.clusters.iter().map(|c| c.states.len()).max().unwrap_or(0), C64::ZERO);
            let (mut uo, mut bo) = (0, 0);
            for cl in &self.clusters {
                let (m, start) = (cl.states.len(), cl.at);
                linalg::fused_product(&sb[bo..bo + n * m], &u[uo..uo + m * m], (n, m, m), &mut block.data[..n * m]);
                for i in 0..n {
                    temp[i * n + start..i * n + start + m].copy_from_slice(&block.data[i * m..(i + 1) * m]);
                }
                uo += m * m;
                bo += n * m;
            }
            let mut uo = 0;
            for cl in &self.clusters {
                let (m, start) = (cl.states.len(), cl.at);
                linalg::fused_product(&ui[uo..uo + m * m], &temp[start * n..(start + m) * n], (m, m, n), &mut hat.data[start * n..(start + m) * n]);
                uo += m * m;
            }
            work += (2 * n * squares) as f64;
        }
        spectrum.work = work;
        Some(work)
    }
}

/// T_j(t), j = 0…d, by the three-term recurrence.
fn chebyshev_at(t: f64, degree: usize, out: &mut Vec<f64>) {
    out.clear();
    out.push(1.0);
    if degree >= 1 {
        out.push(t);
    }
    for j in 2..=degree {
        let next = 2.0 * t * out[j - 1] - out[j - 2];
        out.push(next);
    }
}

/// out += Σ_j f_j c_j[from..from + count], the coefficients' rows `width` apart (one row per factor f_j): each row a
/// real multiple onto the reals of out.
fn onto(c: &[C64], width: usize, from: usize, count: usize, f: &[f64], out: &mut [C64]) {
    // SAFETY: C64 is two f64 (`repr(C)`): a slice of n of them is 2n reals.
    let o = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut f64, 2 * count) };
    for (j, t) in f.iter().enumerate() {
        let row = &c[j * width + from..j * width + from + count];
        let r = unsafe { std::slice::from_raw_parts(row.as_ptr() as *const f64, 2 * count) };
        for (a, b) in o.iter_mut().zip(r) {
            *a += t * b;
        }
    }
}

/// A node's buffers for the tables (`BlockTable::fill`).
#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    x: Vec<f64>,
    value: Vec<C64>,
    t: Vec<f64>,
    factors: Vec<f64>,
    spectra: Vec<Spectrum>,
    block: Mat,
    temp: Vec<C64>,
    u: Vec<C64>,
    ui: Vec<C64>,
}

impl Table {
    /// Every block's table, or why not (module documentation), first made at `values` (`BlockTable::new`).
    pub(crate) fn new(plan: &Plan, values: &Values) -> Table {
        let clock = std::time::Instant::now();
        let mut out = Table::default();
        for b in 0..plan.blocks.len() {
            match BlockTable::new(plan, b, values) {
                Ok(t) => out.blocks.push(Some(Arc::new(t))),
                Err(why) => {
                    out.blocks.push(None);
                    out.why.push((b, why));
                }
            }
        }
        out.seconds = clock.elapsed().as_secs_f64();
        out
    }

    /// The tables' memory (bytes of coefficients): the census's (the tests').
    #[cfg(test)]
    pub(crate) fn bytes(&self) -> usize {
        self.blocks.iter().flatten().map(|t| t.pieces.iter().map(|p| (p.main.len() + p.smalls.len()) * 16).sum::<usize>()).sum()
    }
}

#[cfg(test)]
impl Table {
    /// A check's timing: per tabled block at s with the coefficients `k`, the least of `repeat` of its fill (`fill`, the
    /// gradients made) and of its generator assembled, decomposed and T̂_s made at the node (µs).
    pub(crate) fn timed(&self, plan: &Plan, s: f64, k: &[f64], repeat: usize) -> Vec<(usize, f64, f64)> {
        let least = |f: &mut dyn FnMut()| {
            f();
            (0..repeat)
                .map(|_| {
                    let clock = std::time::Instant::now();
                    f();
                    clock.elapsed().as_secs_f64() * 1e6
                })
                .fold(f64::INFINITY, linalg::smaller)
        };
        let values = Values { coefficients: k.to_vec(), ..Default::default() };
        let mut out = Vec::new();
        for (b, t) in self.blocks.iter().enumerate() {
            let Some(t) = t else { continue };
            let parts = Parts::of(plan, b).unwrap();
            let (mut spectrum, mut hat, mut scratch) = (Spectrum::default(), Mat::default(), Scratch::default());
            let tabled = least(&mut || {
                std::hint::black_box(t.fill(s, &values, true, &mut spectrum, &mut hat, &mut scratch));
            });
            let (mut direct, mut g, mut x, mut h) = (Spectrum::default(), Mat::default(), Mat::default(), Mat::default());
            let made = least(&mut || {
                g.clone_from(&parts.f);
                g.add_scaled(C64::real(s), &parts.s);
                for (c, m) in &parts.smalls {
                    if let Small::Doppler(c) = c {
                        g.add_scaled(C64::real(s * values.coefficients.get(*c).copied().unwrap_or(0.0)), m);
                    }
                }
                direct.decompose(&g);
                for _ in 0..plan.blocks[b].varying.len() {
                    x = parts.s.product(&direct.vectors);
                    h = direct.inverse.product(&x);
                }
                std::hint::black_box(&h);
            });
            out.push((b, tabled, made));
        }
        out
    }
}
