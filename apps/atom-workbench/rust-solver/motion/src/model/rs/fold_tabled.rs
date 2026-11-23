//! The node's contraction off tables of pair products (BRIEF_STEP3 item 4; OLDNEW's prototype, rs/fold_pairs.rs, made
//! the product's): what `Local::evaluate` and `Out::contract` make of the folded blocks at a point, read off tables in
//! the tweezer's intensity s made once for the run, with the bound of what the reading differs by (`Reading`).
//!
//! # The structure
//! A pair (leg sum w, amplitude sum z) writes W M_z, M_z = Σ_{m∈z} R(E_a + f_m) X_m column by column (rs.rs, # The fold),
//! R(z) = (z − G₀)⁻¹ the folded block's resolvent. Each member's frequency is the sum's at rest f₀ plus its shift, and
//! the column's energy E_a is the table's E_a(s) plus its own shift ΔE_a; δ = f_m − f₀ + ΔE_a, and
//!   R(z₀ + δ) = Σ_J (−δ)^J R(z₀)^{J+1},   z₀ = E_a(s) + f₀,
//! kept to `orders` terms (J < O). The legs and the amplitudes are sums of fixed sparse matrices (an entry list's term
//! and coefficient of one member) times the point's numbers (the factors, the coefficients): each sum's fixed matrices
//! span a basis (Gram–Schmidt, the Frobenius product; a carrier's beams of one frequency span at most the dipole's three
//! components), and a pair is
//!   W M_z = Σ_{k′,k,J} φ_k′ ψ_k^{(J)} P_{k′kJ}(s),   P_{k′kJ} = L_k′ R(z₀)^{J+1} U_k,
//! φ the leg's coordinates, ψ^{(J)} = Σ_m (−δ_m)^J × the member's coordinates, the small matrices P tabled in s
//! (Chebyshev pieces); its gradient Σ (∇φ ψ + φ ∇ψ) P + φ ψ P′ ∇s. The lift M on Q is the same with the leg the
//! identity on the block's rows. The direct entries are the contraction's own (`Out::direct`).
//!
//! # What the tables hold at rest
//! A tabled block's generator at the tables' values is G₀(s) = F + s S: the intensity factors at s, every beam's own
//! intensity at one, the coefficients at rest (the first one, every Doppler power none). A block whose generator holds
//! any other varying factor is not tabled, and a plan with such a block has no tables (`Tables::new`'s None): its nodes
//! are the fold's own. At a point the generator is G₀(s) + ΔG, ΔG the tweezer's Doppler powers' entries and a Gaussian
//! beam's own intensity's less one (`Reading::moved`).
//!
//! # The bounds
//! A node off the tables differs from the fold's own by four parts, each bounded at the node (`Tables::read`):
//! - the Doppler series' tail: per eigen-term ρ_k^O/(1 − ρ_k) of its leading term, ρ_k = |δ|/|z₀ − λ_k|, at most
//!   `Reading::tail` with ρ the largest |δ| over the piece's least gap;
//! - the gap between the samples (Bauer–Fike): on a tabled block G₀(s) − G₀(s_i) = (s − s_i) S, so every eigenvalue of
//!   G₀(s) lies within κ_i |s − s_i| ‖S‖ of one of G₀(s_i)'s, κ_i = ‖V‖‖V⁻¹‖ at the sample, and z₀ moves by |B_a| |s −
//!   s_i| (E_a(s) = A_a + B_a s): the least gap over the piece is at least min_i g_i − max_i (κ_i ‖S‖ + max_a |B_a|) h,
//!   h = ½(b − a) sin(π/2m) the farthest any s of the piece [a, b] lies from its nearest of the m Chebyshev nodes. A piece
//!   whose bound is not positive covers nothing;
//! - ΔG (above): R′ − R = R′ ΔG R, ‖R‖ ≤ κ/g, so relative to R's scale κ/g, ‖R′ − R‖ ≤ ε = κ‖ΔG‖/(g − κ‖ΔG‖) (Frobenius
//!   norms bound the 2-norms); its gradient's part, R ∇ΔG R, κ‖∇ΔG‖/g on the gradients (per µm). No node is read where
//!   κ‖ΔG‖ ≥ g;
//! - the interpolation: strict, on each table entry (# The interpolation's bound), and the spans' dropped share
//!   (`Tables::dropped`).
//! The tail and ΔG's part are relative to each eigen-term's modulus -- an element's own scale, the moduli of the terms
//! it is the sum of (`Out::scales`) --, the spans' to the sums' scale (the output's largest), the interpolation's
//! absolute on each output (`Interpolated`) and over its largest element at the node (`Interpolated::relative`, what the
//! run books). A gradient's tail is the series' terms' own s-derivatives, (J + 1) R^{J+2} on the J-th:
//! Σ_{J≥O} (J + 1) ρ^J = ρ^O (O + 1 − Oρ)/(1 − ρ)², and the terms' gradients through δ's, O ρ^{O−1}/(1 − ρ)² times
//! |∇ΔE|/g (`Reading::shift_gradient`) on the value's scale. Measured (model/selected/tabled.rs, 40 walk points each of
//! PGC3D, TET, tilt0, tilt2 at 1–3 terms): every element of H, the channels and the lift, values and gradients, within
//! these bounds, at most 0.99 of them (tilt0's channels: the bound is reached).
//!
//! # The interpolation's bound
//! An entry of a table, L R(s)^p U with p = J + 1 (L a leg's basis vector, the identity's rows for a lift, U an
//! amplitude's: rows and columns of norm at most one), is analytic in s wherever z₀(s) is no eigenvalue of G₀(s); both
//! are linear in s. At a sample s_i with G₀ = V Λ V⁻¹ and s = s_i + δ, z₀ − G₀ = V(D⁻¹ + δ(B_a − Ŝ))V⁻¹, D = (z₀ − Λ)⁻¹,
//! Ŝ = V⁻¹SV, so the entry is (LV)[(I + δK)⁻¹D]^p(V⁻¹U), K = D(B_a − Ŝ): within c (r/(1 − |δ|k))^p where c bounds
//! ‖(LV)_row‖ ‖(V⁻¹U)_col‖, r = ‖D‖ = 1/g and k ≥ ‖K‖₂ (its diagonal's largest, (B_a − λ′)/(z₀ − λ), plus its
//! off-diagonal part's Frobenius norm) -- κ once, through LV and V⁻¹U, not to the p-th power, and the ellipse set by
//! how fast z₀ − λ closes, not by ‖S‖. Every point of the Bernstein ellipse E_ρ of the piece (foci its ends, ρ the sum
//! of its half axes over the half width h) lies within h((ρ − 1/ρ)/2 + sin(π/2m)) of a sample; there the entry is at most
//! M(ρ), and the interpolant at the m first-kind points misses by at most 4M ρ^{−(m−1)}/(ρ − 1), its slope by
//! 4M/h Σ_{k≥m} k² ρ^{−k} (every Chebyshev coefficient a_k ≤ 2M ρ^{−k} aliases onto one below degree m, |T_k′| ≤ k²),
//! the least over ρ (`bernstein`); the samples' rounding (γ_n of their terms' moduli, n = |B|(O + 1) + m) adds the
//! Lebesgue constant's (2/π) ln m + 1 times it, and (m − 1)² over h on the slope. Per table and term on each piece
//! (`Piece::bounds`); a node sums Σ |φ| |ψ| E over each output's tables -- each |ψ| its largest over the columns -- and
//! Σ (|∇φ| |ψ| + |φ| |∇ψ|) E + |φ| |ψ| E′ |∇s| on its gradients (`Interpolated`). Measured (2026-11-23, the same walks,
//! the pieces of # The pieces): 1e-8 to 1.4e-7 of each output's largest scale on the values and 3e-6 to 3.4e-5 on the
//! gradients; over the output's largest element at the node (what is booked) at most 5e-8, 2.1e-7, 8.8e-8, 1.7e-7 on
//! the values and 1.6e-5, 4.2e-5, 2.0e-5, 4.0e-5 on the gradients (PGC3D, TET, tilt0, tilt2) -- beside the Doppler
//! series' tail at two terms, about 2e-5 and 5e-3. The nearest singularity lies 38 to 72 half widths off. As a rate of
//! the drop tier (`Selected::rate`, each output's Frobenius norm) the same bound would be 1.1e-7, 6.8e-7, 1.3e-6,
//! 5.9e-6 /µs, 20 to 800 times the dropped terms' Y: a smooth relative change of the outputs, booked with the tail as
//! one. Its first form, with ‖R(s_i)‖ ≤ κ/g and z₀ − G₀ moving by |B_a| + ‖S‖ (κ and ‖S‖ Frobenius, κ to the p-th
//! power), left ρ ≈ 1: 2e2 of the entries' scale, no bound.
//!
//! # The pieces
//! A piece [a, b] of s holds where its strict bound (`Piece::worst`: per table and term its value's bound over its
//! entries' own bound c ‖D‖^p, and its slope's over theirs c p ‖D‖^p ‖K‖, the J-th weighed by the reach's ratio
//! ρ_reach^J) is within the target: the run's budget where it is known (model/selected.rs, # The tables' budget: the
//! rates' relative error on the values, the friction's on the slopes -- the interpolation never adds more to a node
//! than the series' tail it is held to), else the Doppler series' tail at the tables' terms at the reach (the largest
//! ratio of the members' shifts there to the sums' gaps at s = 1) and at one term fewer. Else it is halved, but where
//! the halves' bounds no longer fall by half -- the samples' rounding reached -- the piece is taken at its bound; one
//! s's round-off cannot halve is taken at its bound too, and where that is none no node reads it. (Until 2026-11-23 the
//! slopes held to nothing: R4's G, ρ 0.08 thermal, laid to the reach's tail in 16 pieces, booked 14.5 % of T at its
//! nodes' gradients. And before, the last two Chebyshev coefficients against the pair's largest, the series' own
//! estimate and no bound -- on each output's largest scale 5 to 230 times the strict bound on the values, 1/45 to 2/3
//! of it on the gradients.)
use super::*;

/// One fixed matrix of a sum on its layout: its member's piece, its term (into `Plan::terms[piece]`), its coefficient,
/// and its coordinates on the sum's basis.
#[derive(Clone, Debug)]
struct Atom {
    piece: usize,
    term: usize,
    coefficient: usize,
    coords: Vec<C64>,
}

/// A sum's basis: its atoms, the basis's size, and an amplitude sum's frequency at rest f₀ (its first member's piece).
#[derive(Clone, Debug, Default)]
struct Basis {
    atoms: Vec<Atom>,
    rank: usize,
    f0: f64,
}

/// What a table writes: an H pair's outputs (its index in `Grouping::h_pairs`), a jump pair's channel (in `j_pairs`), or
/// a lift group's rows (`Grouping::lift`: its block and group).
#[derive(Clone, Copy, Debug, PartialEq)]
enum Target {
    H(usize),
    Jump(usize),
    Lift(usize),
}

/// A pair's tables: what it writes, its leg sum (none for a lift: the identity on the block's rows), its amplitude sum,
/// its rows and columns, the Chebyshev coefficients [piece][k′][k][J][d][i·cols + j].
#[derive(Clone, Debug)]
struct Table {
    target: Target,
    w: Option<usize>,
    z: usize,
    rows: usize,
    cols: usize,
    coef: Vec<C64>,
}

/// A piece of s: its range, per amplitude sum the least gap's lower bound over it (`# The bounds`), per block κ at its
/// samples' largest, and its tables' tail relative to each pair's largest coefficient (`# The pieces`).
#[derive(Clone, Debug)]
pub(crate) struct Piece {
    pub range: [f64; 2],
    gaps: Vec<f64>,
    kappa: Vec<f64>,
    /// The strict bound's worst over its tables (`Tables::layout`, # The pieces): [the largest value's bound over its
    /// entries' own bound c ‖D‖^p, the largest slope's over theirs c p ‖D‖^p ‖K‖], the J-th term's weighed by the
    /// reach's ratio to the J.
    pub worst: [f64; 2],
    /// Per table and Doppler term J (`[t·orders + J]`) the strict bound of its entries' interpolation (`bernstein`, with
    /// the samples' rounding through the interpolant), [the value's, the slope's in s], absolute.
    pub bounds: Vec<[f64; 2]>,
    /// The least 1/(‖K‖ h) over the sums: how far the nearest singularity of the tables' entries lies, in the piece's
    /// half widths (the ellipses' room; the tests' measurement).
    #[cfg_attr(not(test), allow(dead_code))]
    pub room: f64,
}

/// The tables of a run's fold (module documentation): made once (`Tables::layout`, `Layout::fill`), read by every node of
/// the run.
#[derive(Clone, Debug)]
pub(crate) struct Tables {
    pub degree: usize,
    /// The Doppler series' terms the tables hold (J < `orders`).
    pub orders: usize,
    pub pieces: Vec<Piece>,
    /// The spans' dropped share, the largest over the sums (relative to each kind's largest atom).
    #[cfg_attr(not(test), allow(dead_code))]
    pub dropped: f64,
    zs: Vec<Basis>,
    ws: Vec<Basis>,
    tables: Vec<Table>,
    /// K's energies the tables are made at, E_a(s) = A_a + B_a s.
    e_rest: Vec<f64>,
    e_slope: Vec<f64>,
    pub build_seconds: f64,
    pub bytes: usize,
    /// The counted complex multiply-adds of its making, layout and fill.
    pub work: f64,
}

/// A table build before its coefficients (`Tables::layout`): the sums' bases and their vectors, the tables, the pieces
/// with their bounds, K's energies and the blocks' slopes S; what making it took and what filling it takes (complex
/// multiply-adds, `Layout::fill`), and its wall time.
#[derive(Clone, Debug)]
pub(crate) struct Layout {
    degree: usize,
    orders: usize,
    pub pieces: Vec<Piece>,
    dropped: f64,
    zs: Vec<Basis>,
    ws: Vec<Basis>,
    z_vectors: Vec<Vec<Vec<C64>>>,
    w_vectors: Vec<Vec<Vec<C64>>>,
    tables: Vec<Table>,
    e_rest: Vec<f64>,
    e_slope: Vec<f64>,
    used: Vec<usize>,
    s_mats: Vec<Option<Mat>>,
    pub work: f64,
    pub price: f64,
    seconds: f64,
}

/// A point's reading of the tables (`Tables::read`): its piece, s and ∇s, the largest Doppler ratio ρ, and the
/// relative bound ε of ΔG on R and its gradient's (per µm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Reading {
    pub piece: usize,
    pub s: f64,
    pub ds: [f64; 3],
    pub rho: f64,
    pub moved: f64,
    pub moved_gradient: f64,
    /// The largest |∇ΔE_a| over the least gap (per µm): what the series' terms' gradients take of δ's.
    pub shift_gradient: f64,
    /// Its counted complex multiply-adds.
    pub work: f64,
}

impl Reading {
    /// The Doppler series' tail at `orders` terms relative to each eigen-term's leading one, ρ^O/(1 − ρ).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn tail(&self, orders: usize) -> f64 {
        self.rho.powi(orders as i32) / (1.0 - self.rho)
    }
}

/// A node's buffers (one a thread): φ and its gradient per leg sum; ψ per amplitude sum, order, basis vector and
/// column; ΔE_a and its gradient; the Chebyshev values; a table's value and slope at s; a pair's sums.
#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    /// The last contraction's strict bound of what its interpolation leaves (# The interpolation's bound), absolute, on
    /// every element of an output: per H output and per channel output [the value's, the gradient's (per µm)], and per
    /// lift group (its values' and gradients').
    pub interpolated: Interpolated,
    phi: Vec<Vec<(C64, [C64; 3])>>,
    psi: Vec<Vec<Vec<Vec<(C64, [C64; 3])>>>>,
    /// Per amplitude sum, term and basis vector the largest |ψ| over its columns and of its gradient (`interpolated`).
    psi_most: Vec<Vec<[f64; 2]>>,
    /// Per block ΔG's Frobenius norm and its gradient's (`read`).
    sizes: Vec<Option<(f64, f64)>>,
    de: Vec<(f64, [f64; 3])>,
    t: Vec<f64>,
    dt: Vec<f64>,
    px: Vec<C64>,
    py: Vec<C64>,
    value: Vec<C64>,
    slope: Vec<C64>,
    grad: [Vec<C64>; 3],
}

/// The strict bound of a piece's Chebyshev interpolant (# The interpolation's bound): of an entry of L R(s)^p U on a
/// piece of half width `half` at m first-kind points, where at every sample s_i the entry is (L V)[(I + δK)⁻¹D]^p(V⁻¹U)
/// at s = s_i + δ -- V, D = diag(1/(z₀ − λ)) the sample's, K = D(B − V⁻¹SV) the move of z₀ − G₀ per unit s in its
/// eigenbasis -- with ‖(LV)_row‖ ‖(V⁻¹U)_col‖ ≤ c, ‖D‖ ≤ r and ‖K‖ ≤ k: [the value's, the slope's in s]. Each ellipse E_ρ
/// of the piece (foci its ends, ρ the sum of its half axes over the half width) lies within h D(ρ), D(ρ) = (ρ − 1/ρ)/2 +
/// sin(π/2m), of a sample, where the Neumann series in δK bounds the entry by M(ρ) = c (r/(1 − k h D(ρ)))^p; the
/// interpolant at the m points misses by at most 4M ρ^{−(m−1)}/(ρ − 1) and its slope by 4M/h Σ_{k≥m} k² ρ^{−k} (each
/// Chebyshev coefficient a_k ≤ 2M ρ^{−k} folds onto one of degree below m, T_k′ ≤ k²), the least over ρ up to where the
/// series stops converging. Unbounded where no ellipse is left.
fn bernstein(c: f64, r: f64, k: f64, half: f64, m: usize, power: i32) -> [f64; 2] {
    let spacing = (std::f64::consts::PI / (2.0 * m as f64)).sin();
    // The largest ρ: k h D(ρ) = 1, D's sinh part y = 1/(k h) − sin(π/2m).
    let y = 1.0 / (k * half) - spacing;
    if !(y > 0.0) || !(r > 0.0) || !c.is_finite() {
        return [f64::INFINITY; 2];
    }
    let most = y + (y * y + 1.0).sqrt();
    let mut best = [f64::INFINITY; 2];
    // ρ − 1 on a geometric grid from 10⁻⁸ of the widest to it (the bound has one minimum between ρ → 1, where
    // 1/(ρ − 1) grows, and ρ → its largest, where M does).
    const GRID: usize = 96;
    for g in 1..GRID {
        let rho = 1.0 + (most - 1.0) * 1e-8f64.powf(1.0 - g as f64 / GRID as f64);
        let d = 0.5 * (rho - 1.0 / rho) + spacing;
        let resolvent = r / (1.0 - k * half * d);
        if !(resolvent > 0.0 && resolvent.is_finite()) {
            continue;
        }
        let big = c * resolvent.powi(power);
        let q = 1.0 / rho;
        let n = m as f64;
        let value = 4.0 * big * q.powi(m as i32 - 1) / (rho - 1.0);
        let tail = q.powi(m as i32) * (n * n / (1.0 - q) + 2.0 * n * q / (1.0 - q).powi(2) + q * (1.0 + q) / (1.0 - q).powi(3));
        let slope = 4.0 * big * tail / half;
        best = [best[0].min(value), best[1].min(slope)];
    }
    best
}

/// A node's bound of its interpolation (`Scratch::interpolated`): per H output, per channel output and per lift group
/// [value, gradient] on each of its elements, absolute (rad/µs, the gradient's per µm).
#[derive(Clone, Debug, Default)]
pub(crate) struct Interpolated {
    pub h: Vec<[f64; 2]>,
    pub channels: Vec<[f64; 2]>,
    pub lift: Vec<[f64; 2]>,
    /// The largest over the outputs (H's, the channels', the lift's groups) of each one's bound over its largest
    /// |value| and |gradient| at the node: [values', gradients'] -- relative to the output's largest element, what the
    /// run books of it beside the Doppler series' tail.
    pub relative: [f64; 2],
}

/// T_d(x) and dT_d/dx for d = 0..=D.
fn chebyshev(x: f64, degree: usize, t: &mut Vec<f64>, dt: &mut Vec<f64>) {
    t.clear();
    dt.clear();
    t.push(1.0);
    dt.push(0.0);
    if degree >= 1 {
        t.push(x);
        dt.push(1.0);
    }
    let (mut t0, mut t1) = (1.0, x);
    let (mut u0, mut u1) = (1.0, 2.0 * x);
    for d in 2..=degree {
        let t2 = 2.0 * x * t1 - t0;
        (t0, t1) = (t1, t2);
        t.push(t2);
        // dT_d/dx = d U_{d−1}.
        dt.push(d as f64 * u1);
        let u2 = 2.0 * x * u1 - u0;
        (u0, u1) = (u1, u2);
    }
}

/// Gram–Schmidt of a list of vectors, each with the bound b of its point coefficient over the reach: in the order of
/// ‖v‖b, a vector's residual r kept as a new direction where ‖r‖b exceeds `limit`, else left out -- its share ‖r‖b over
/// `top`, summed, is the span's dropped share (returned). The basis and each vector's coordinates on it.
fn span(vectors: &[Vec<C64>], bounds: &[f64], limit: f64, top: f64) -> (Vec<Vec<C64>>, Vec<Vec<C64>>, f64) {
    let dot = |a: &[C64], b: &[C64]| a.iter().zip(b).fold(C64::ZERO, |s, (x, y)| s + x.conj() * *y);
    let sizes: Vec<f64> = vectors.iter().zip(bounds).map(|(v, b)| dot(v, v).re.sqrt() * b).collect();
    let mut order: Vec<usize> = (0..vectors.len()).collect();
    order.sort_by(|&a, &b| sizes[b].total_cmp(&sizes[a]));
    let mut basis: Vec<Vec<C64>> = Vec::new();
    let mut coords: Vec<Vec<C64>> = vec![Vec::new(); vectors.len()];
    let mut dropped = 0.0;
    for &i in &order {
        let v = &vectors[i];
        // Twice (Gram–Schmidt reorthogonalised): the residual and its coordinates.
        let mut r = v.clone();
        let mut c: Vec<C64> = vec![C64::ZERO; basis.len()];
        for _ in 0..2 {
            for (k, q) in basis.iter().enumerate() {
                let x = dot(q, &r);
                c[k] += x;
                for (a, b) in r.iter_mut().zip(q) {
                    *a -= x * *b;
                }
            }
        }
        let norm = dot(&r, &r).re.sqrt();
        if norm * bounds[i] > limit {
            let q: Vec<C64> = r.iter().map(|x| *x * (1.0 / norm)).collect();
            basis.push(q);
            c.push(C64::real(norm));
        } else if top > 0.0 {
            dropped += norm * bounds[i] / top;
        }
        coords[i] = c;
    }
    let rank = basis.len();
    for c in coords.iter_mut() {
        c.resize(rank, C64::ZERO);
    }
    (basis, coords, dropped)
}

/// A sum's atoms: per atom its key (piece, term, coefficient) and its fixed matrix on the sum's layout.
type Atoms = (Vec<(usize, usize, usize)>, Vec<Vec<C64>>);

fn atom_at(keys: &mut Vec<(usize, usize, usize)>, mats: &mut Vec<Vec<C64>>, key: (usize, usize, usize), size: usize) -> usize {
    keys.iter().position(|k| *k == key).unwrap_or_else(|| {
        keys.push(key);
        mats.push(vec![C64::ZERO; size]);
        keys.len() - 1
    })
}

impl Tables {
    /// G₀ on block b at the values `v` (as `Local::evaluate` assembles it).
    fn generator(plan: &Plan, b: usize, v: &Values) -> Mat {
        let block = &plan.blocks[b];
        let size = block.rows.len();
        let mut g = Mat::zeros(size, size);
        g.data.copy_from_slice(&block.fixed);
        for e in &block.rest {
            let (f, scale) = plan.generator_terms[e.term as usize];
            let c = e.value * v.coefficients[e.coefficient as usize];
            g.data[e.row as usize * size + e.col as usize] += scale * v.factors[f].0 * c;
        }
        g
    }

    /// K's energies at the values `v` (the diagonal's entries, as `Local::evaluate`'s).
    fn energies(plan: &Plan, v: &Values) -> Vec<f64> {
        let mut e = vec![0.0; plan.n];
        for d in &plan.diagonal {
            let (f, scale) = plan.generator_terms[d.term as usize];
            e[d.row as usize] += (scale * v.factors[f].0 * (d.value * v.coefficients[d.coefficient as usize])).re;
        }
        e
    }

    /// The values the tables are made at: `rest`'s (a point at rest), the tweezer's intensity s, every beam's own
    /// intensity one, the coefficients at rest.
    fn at_intensity(plan: &Plan, rest: &Values, s: f64) -> Values {
        let mut v = rest.clone();
        for &f in &plan.intensity {
            v.factors[f] = (C64::real(s), [C64::ZERO; 3]);
        }
        for &f in &plan.own {
            v.factors[f] = (C64::ONE, [C64::ZERO; 3]);
        }
        for (i, c) in v.coefficients.iter_mut().enumerate() {
            *c = if i == 0 { 1.0 } else { 0.0 };
        }
        v
    }

    /// The tweezer's intensity s at the values `v` (a plan with tables has one; the tests').
    #[cfg(test)]
    pub(crate) fn intensity(plan: &Plan, v: &Values) -> f64 {
        v.factors[plan.intensity[0]].0.re
    }

    /// The tweezer's intensity s at the values `v`, none where the plan has none.
    pub(crate) fn intensity_of(plan: &Plan, v: &Values) -> Option<f64> {
        plan.intensity.first().map(|&f| v.factors[f].0.re)
    }

    /// Whether a factor is one the tables hold at s or at one (the intensity, a beam's own intensity).
    fn held(plan: &Plan, f: usize) -> bool {
        plan.intensity.contains(&f) || plan.own.contains(&f)
    }

    /// One sample of the tables at the intensity s (`layout`, `fill`): K's energies there, each used block's spectrum
    /// and, where `turned`, its V⁻¹SV; the counted complex multiply-adds. None where a block does not decompose.
    fn sample(plan: &Plan, rest: &Values, used: &[usize], s_mats: &[Option<Mat>], s: f64, turned: bool) -> Option<(Vec<f64>, Vec<Option<Spectrum>>, Vec<Mat>, f64)> {
        let v = Tables::at_intensity(plan, rest, s);
        let energies = Tables::energies(plan, &v);
        let (mut spectra, mut hats): (Vec<Option<Spectrum>>, Vec<Mat>) = (vec![None; plan.blocks.len()], vec![Mat::default(); plan.blocks.len()]);
        let mut work = (plan.diagonal.len() + plan.n) as f64;
        for &bl in used {
            if spectra[bl].is_some() {
                continue;
            }
            let g = Tables::generator(plan, bl, &v);
            let mut sp = Spectrum::default();
            if !sp.decompose(&g) {
                return None;
            }
            let size = g.rows;
            work += sp.work + (size * size + plan.blocks[bl].rest.len()) as f64;
            if turned {
                if let Some(s) = &s_mats[bl] {
                    hats[bl] = sp.inverse.product(&s.product(&sp.vectors));
                    work += (2 * size * size * size) as f64;
                }
            }
            spectra[bl] = Some(sp);
        }
        Some((energies, spectra, hats, work))
    }

    /// The layout of a plan's grouping's tables (module documentation): their sums' bases, the tables, and the pieces of
    /// s on [`low`, 1] with their strict bounds (# The interpolation's bound, # The pieces), from the blocks'
    /// decompositions alone -- what filling them takes is then known (`Layout::price`) before it is done. `rest` the
    /// values at a point at rest (the members' frequencies f₀), `reach` the run's (`Rs::reach`: the pieces' Doppler
    /// shifts at its momentum, the coefficients' largest), `orders` the Doppler series' terms kept, `degree` the pieces'
    /// Chebyshev degree. None where a block a sum is on is not tabled (its generator varies by another factor than the
    /// intensities), the plan has no intensity, or a block does not decompose.
    pub(crate) fn layout(plan: &Plan, grouping: &Grouping, rest: &Values, reach: &Reach, orders: usize, degree: usize, low: f64, target: Option<[f64; 2]>) -> Option<Layout> {
        let clock = std::time::Instant::now();
        if plan.intensity.is_empty() || !(low < 1.0) {
            return None;
        }
        let used: Vec<usize> = grouping.zs.iter().map(|s| s.block).collect();
        let untabled = used.iter().any(|&b| {
            plan.blocks[b].rest.iter().any(|e| {
                let f = plan.generator_terms[e.term as usize].0;
                plan.varying[f] && !Tables::held(plan, f)
            })
        });
        if untabled {
            return None;
        }
        // An atom's point coefficient's bound over the reach: its scale, its factor at most one (a beam's amplitude, an
        // intensity) or its value at rest, and its coefficient's largest.
        let bound = |piece: usize, term: usize, coefficient: usize| -> f64 {
            let (f, scale) = plan.terms[piece][term];
            scale.abs() * rest.factors[f].0.abs().max(1.0) * reach.coefficients[coefficient].abs()
        };
        // The sums' atoms.
        let mut z_atoms: Vec<Atoms> = Vec::new();
        for s in grouping.zs.iter() {
            let (size, cols) = (plan.blocks[s.block].rows.len(), s.index.len());
            let (mut keys, mut mats) = (Vec::new(), Vec::new());
            for (member, map) in &s.members {
                let r = &plan.amplitudes[*member];
                for e in &r.entries {
                    let Some(&(_, to)) = map.iter().find(|x| x.0 == e.col as usize) else { continue };
                    let at = atom_at(&mut keys, &mut mats, (r.piece, e.term as usize, e.coefficient as usize), size * cols);
                    mats[at][e.row as usize * cols + to] += e.value;
                }
            }
            z_atoms.push((keys, mats));
        }
        let mut w_atoms: Vec<Atoms> = Vec::new();
        for s in grouping.ws.iter() {
            let (size, rows) = (plan.blocks[s.block].rows.len(), s.index.len());
            let (mut keys, mut mats) = (Vec::new(), Vec::new());
            for (member, map) in &grouping.wmaps[s.at] {
                let r = &plan.legs[*member];
                for e in &r.entries {
                    let to = map[e.row as usize];
                    if to == u32::MAX {
                        continue;
                    }
                    let at = atom_at(&mut keys, &mut mats, (r.piece, e.term as usize, e.coefficient as usize), rows * size);
                    mats[at][to as usize * size + e.col as usize] += e.value;
                }
            }
            w_atoms.push((keys, mats));
        }
        // The spans' scale, one for each kind over its sums: an amplitude sum's atoms ‖A‖b over the least |z₀ − λ| of its
        // block at s = 1 (its resolvent's scale), a leg sum's ‖L‖b; what is left out is rounding against the whole
        // model's: below n u of the largest, n the atoms of the sum.
        let at_one = Tables::at_intensity(plan, rest, 1.0);
        let e_one = Tables::energies(plan, &at_one);
        let f0_of = |s: &Sum| s.members.first().map_or(0.0, |(m, _)| rest.frequencies[plan.amplitudes[*m].piece]);
        let mut work = 0.0;
        let mut gap_one: Vec<f64> = Vec::with_capacity(grouping.zs.len());
        for s in &grouping.zs {
            let mut sp = Spectrum::default();
            if !sp.decompose(&Tables::generator(plan, s.block, &at_one)) {
                return None;
            }
            work += sp.work;
            let f0 = f0_of(s);
            gap_one.push(s.index.iter().flat_map(|&col| (0..sp.centre.len()).map(move |k| (col, k))).map(|(col, k)| (C64::real(e_one[col] + f0 - sp.centre[k]) - sp.values[k]).abs()).fold(f64::INFINITY, f64::min));
        }
        let norm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let top_of = |(keys, mats): &Atoms| keys.iter().zip(mats).map(|(&(p, t, c), m)| norm(m) * bound(p, t, c)).fold(0.0, f64::max);
        let z_top = z_atoms.iter().zip(&gap_one).map(|(a, g)| top_of(a) / g).fold(0.0, f64::max);
        let w_top = w_atoms.iter().map(top_of).fold(0.0, f64::max);
        let mut dropped: f64 = 0.0;
        let mut basis_of = |(keys, mats): Atoms, f0: f64, top: f64, vectors: &mut Vec<Vec<Vec<C64>>>| -> Basis {
            let bounds: Vec<f64> = keys.iter().map(|&(p, t, c)| bound(p, t, c)).collect();
            let limit = keys.len() as f64 * UNIT_ROUNDOFF * top;
            let (basis, coords, d) = span(&mats, &bounds, limit, top);
            dropped = dropped.max(d);
            vectors.push(basis);
            Basis { atoms: keys.iter().zip(coords).map(|(&(piece, term, coefficient), coords)| Atom { piece, term, coefficient, coords }).collect(), rank: vectors.last().map_or(0, Vec::len), f0 }
        };
        let (mut zs, mut z_vectors): (Vec<Basis>, Vec<Vec<Vec<C64>>>) = (Vec::new(), Vec::new());
        for ((s, atoms), g) in grouping.zs.iter().zip(z_atoms).zip(&gap_one) {
            zs.push(basis_of(atoms, f0_of(s), z_top * g, &mut z_vectors));
        }
        let (mut ws, mut w_vectors): (Vec<Basis>, Vec<Vec<Vec<C64>>>) = (Vec::new(), Vec::new());
        for atoms in w_atoms {
            ws.push(basis_of(atoms, 0.0, w_top, &mut w_vectors));
        }
        // The tables: H's pairs, the jumps', and the lift's sums (the identity leg), each on a made block.
        let mut tables: Vec<Table> = Vec::new();
        for (i, x) in grouping.h_pairs.iter().enumerate() {
            tables.push(Table { target: Target::H(i), w: Some(x.w), z: x.z, rows: grouping.ws[x.w].index.len(), cols: grouping.zs[x.z].index.len(), coef: Vec::new() });
        }
        for (i, x) in grouping.j_pairs.iter().enumerate() {
            tables.push(Table { target: Target::Jump(i), w: Some(x.w), z: x.z, rows: grouping.ws[x.w].index.len(), cols: grouping.zs[x.z].index.len(), coef: Vec::new() });
        }
        for (l, (b, _, mine)) in grouping.lift.iter().enumerate() {
            for &z in mine {
                tables.push(Table { target: Target::Lift(l), w: None, z, rows: plan.blocks[*b].rows.len(), cols: grouping.zs[z].index.len(), coef: Vec::new() });
            }
        }
        tables.retain(|t| plan.made(grouping.zs[t.z].block, true));
        // Per block S = ∂G₀/∂s (G₀ is linear in s on a tabled block), and its Frobenius norm.
        let at_zero = Tables::at_intensity(plan, rest, 0.0);
        let s_mats: Vec<Option<Mat>> = (0..plan.blocks.len())
            .map(|b| {
                used.contains(&b).then(|| {
                    let (g0, mut g1) = (Tables::generator(plan, b, &at_zero), Tables::generator(plan, b, &at_one));
                    g1.data.iter_mut().zip(&g0.data).for_each(|(x, y)| *x -= *y);
                    g1
                })
            })
            .collect();
        let slope: Vec<f64> = s_mats.iter().map(|s| s.as_ref().map_or(0.0, |s| norm(&s.data))).collect();
        let e0 = Tables::energies(plan, &at_zero);
        let e_slope: Vec<f64> = e0.iter().zip(&e_one).map(|(a, b)| b - a).collect();
        let b_most = e_slope.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let m = degree + 1;
        let largest_block = used.iter().map(|&b| plan.blocks[b].rows.len()).max().unwrap_or(0);
        // The interpolation's target: the Doppler series' tail at the tables' terms at the reach (the largest ratio its
        // members' shifts there take of the sums' gaps at s = 1: each member's frequency at rest off the sum's and its
        // Doppler shift at the reach's momentum), so that a piece never adds more to a node's bound than the series it
        // holds takes at the run's fastest atom.
        let reach_rho = grouping
            .zs
            .iter()
            .zip(&gap_one)
            .flat_map(|(s, g)| {
                s.members.iter().map(move |(m, _)| {
                    let piece = plan.amplitudes[*m].piece;
                    ((rest.frequencies[piece] - f0_of(s)).abs() + reach.doppler[piece] * reach.kappa) / g
                })
            })
            .fold(0.0, f64::max);
        // Where the run's budget is known, it (# The pieces).
        let tail = |o: i32| if reach_rho < 1.0 { reach_rho.powi(o) / (1.0 - reach_rho) } else { 0.0 };
        let target = target.unwrap_or([tail(orders as i32), tail(orders as i32 - 1)]);
        // A sample element's rounding: γ_n of the moduli of the terms it is the sum of, n the operations behind it and
        // its transform.
        let floor = {
            let n = (largest_block * (orders + 1) + m) as f64;
            n * UNIT_ROUNDOFF / (1.0 - n * UNIT_ROUNDOFF)
        };
        let lebesgue = 2.0 / std::f64::consts::PI * (m as f64).ln() + 1.0;
        let xs: Vec<f64> = (0..m).map(|n| (std::f64::consts::PI * (n as f64 + 0.5) / m as f64).cos()).collect();
        // One piece [a, b] from its samples' decompositions (# The interpolation's bound): per sum the largest ‖D‖ and
        // ‖K‖ of its columns, per leg sum (and per lift block, the identity's) the largest ‖(LV)_row‖, per amplitude sum
        // the largest ‖(V⁻¹U)_col‖; per table and term its bound; the gaps' lower bounds per sum and κ per block (# The
        // bounds). And its counted work.
        let bounded = |a: f64, b: f64| -> Option<(Piece, f64)> {
            let mut work = 0.0;
            let (mut gaps, mut kappa) = (vec![f64::INFINITY; grouping.zs.len()], vec![0.0f64; plan.blocks.len()]);
            let (mut resolvent, mut moves) = (vec![0.0f64; grouping.zs.len()], vec![0.0f64; grouping.zs.len()]);
            let (mut legs, mut amplitudes, mut identity) = (vec![0.0f64; grouping.ws.len()], vec![0.0f64; grouping.zs.len()], vec![0.0f64; plan.blocks.len()]);
            for &x in &xs {
                let (energies, spectra, hats, w) = Tables::sample(plan, rest, &used, &s_mats, 0.5 * (a + b) + 0.5 * (b - a) * x, true)?;
                work += w;
                for &bl in &used {
                    let sp = spectra[bl].as_ref().expect("a used block decomposed");
                    kappa[bl] = kappa[bl].max(norm(&sp.vectors.data) * norm(&sp.inverse.data));
                    let size = sp.centre.len();
                    identity[bl] = identity[bl].max(sp.vectors.data.chunks(size).map(norm).fold(0.0, f64::max));
                }
                for (z, sum) in grouping.zs.iter().enumerate() {
                    let sp = spectra[sum.block].as_ref().expect("a sum's block decomposed");
                    let hat = &hats[sum.block];
                    let size = sp.centre.len();
                    let mut least = f64::INFINITY;
                    for &col in &sum.index {
                        let z0 = energies[col] + zs[z].f0;
                        // ‖K‖₂ ≤ its diagonal's largest + its off-diagonal part's Frobenius norm.
                        let (mut diagonal, mut off) = (0.0f64, 0.0f64);
                        for k in 0..size {
                            let w = (C64::real(z0 - sp.centre[k]) - sp.values[k]).abs();
                            least = least.min(w);
                            for l in 0..size {
                                let x = hat.data[k * size + l];
                                if k == l {
                                    diagonal = diagonal.max((C64::real(e_slope[col]) - x).abs() / w);
                                } else {
                                    off += x.norm_sqr() / (w * w);
                                }
                            }
                        }
                        moves[z] = moves[z].max(diagonal + off.sqrt());
                    }
                    gaps[z] = gaps[z].min(least);
                    resolvent[z] = resolvent[z].max(1.0 / least);
                    // V⁻¹U's columns.
                    let cols = sum.index.len();
                    for u in &z_vectors[z] {
                        let vu = linalg_product(&sp.inverse.data, u, size, size, cols);
                        for j in 0..cols {
                            amplitudes[z] = amplitudes[z].max((0..size).map(|k| vu[k * cols + j].norm_sqr()).sum::<f64>().sqrt());
                        }
                    }
                    work += (cols * size * (size + 1) * z_vectors[z].len()) as f64;
                }
                for (w, sum) in grouping.ws.iter().enumerate() {
                    // A leg sum on a block no amplitude sum is on is in no table.
                    let Some(sp) = spectra[sum.block].as_ref() else { continue };
                    let (size, rows) = (sp.centre.len(), sum.index.len());
                    for l in &w_vectors[w] {
                        let lv = linalg_product(l, &sp.vectors.data, rows, size, size);
                        legs[w] = legs[w].max(lv.chunks(size).map(norm).fold(0.0, f64::max));
                    }
                    work += (rows * size * (size + 1) * w_vectors[w].len()) as f64;
                }
            }
            // The gaps' lower bounds over the piece (# The bounds).
            let h = 0.5 * (b - a) * (std::f64::consts::PI / (2.0 * m as f64)).sin();
            let moving = |z: usize| kappa[grouping.zs[z].block] * slope[grouping.zs[z].block] + b_most;
            let gaps: Vec<f64> = gaps.iter().enumerate().map(|(z, g)| g - moving(z) * h).collect();
            // The strict bounds: per table and term c, ‖D‖ and ‖K‖ (`bernstein`), and the samples' rounding carried
            // through the interpolant -- the Lebesgue constant of m first-kind points, at most (2/π) ln m + 1, and the
            // slope's by Markov's (m − 1)² over the half width --, of an entry's own bound c ‖D‖^p. The piece's worst:
            // per table and term its value's bound over that, the J-th weighed by the reach's ratio to the J (what it
            // adds beside the leading term at the run's fastest atom).
            let half = 0.5 * (b - a);
            let (mut bounds, mut worst): (Vec<[f64; 2]>, [f64; 2]) = (Vec::with_capacity(tables.len() * orders), [0.0; 2]);
            for t in &tables {
                let side = t.w.map_or(identity[grouping.zs[t.z].block], |w| legs[w]) * amplitudes[t.z];
                for jj in 0..orders {
                    let p = jj as i32 + 1;
                    let [value, slope_bound] = bernstein(side, resolvent[t.z], moves[t.z], half, m, p);
                    let scale = side * resolvent[t.z].powi(p);
                    let rounding = (lebesgue + 1.0) * floor * scale;
                    bounds.push([value + rounding, slope_bound + ((m - 1) * (m - 1)) as f64 * rounding / half]);
                    if scale > 0.0 {
                        let weight = reach_rho.powi(jj as i32);
                        worst[0] = worst[0].max(weight * (value + rounding) / scale);
                        // An entry's slope in s is within c p ‖D‖^p ‖K‖ (the Neumann series' first order).
                        let slope_scale = scale * p as f64 * moves[t.z];
                        if slope_scale > 0.0 {
                            worst[1] = worst[1].max(weight * (slope_bound + ((m - 1) * (m - 1)) as f64 * rounding / half) / slope_scale);
                        }
                    }
                }
            }
            let room = moves.iter().map(|k| 1.0 / (k * half)).fold(f64::INFINITY, f64::min);
            Some((Piece { range: [a, b], gaps, kappa, worst, bounds, room }, work))
        };
        // The pieces: halved until their strict bound is within the target, or no longer falls by half (the samples'
        // rounding reached: # The pieces); one s's round-off cannot halve is taken at its bound.
        let first = bounded(low, 1.0)?;
        work += first.1;
        let mut todo = vec![first.0];
        let mut pieces: Vec<Piece> = Vec::new();
        while let Some(piece) = todo.pop() {
            let [a, b] = piece.range;
            let c = 0.5 * (a + b);
            if (piece.worst[0] <= target[0] && piece.worst[1] <= target[1]) || !(c > a && c < b) {
                pieces.push(piece);
                continue;
            }
            let (left, right) = (bounded(a, c)?, bounded(c, b)?);
            work += left.1 + right.1;
            // Halving helps where every part beyond its target falls by half.
            let falls = (0..2).all(|k| piece.worst[k] <= target[k] || left.0.worst[k].max(right.0.worst[k]) <= 0.5 * piece.worst[k]);
            if !falls {
                pieces.push(piece);
            } else {
                todo.push(right.0);
                todo.push(left.0);
            }
        }
        pieces.sort_by(|x, y| x.range[0].total_cmp(&y.range[0]));
        // What filling takes (`fill`, complex multiply-adds): per piece and sample the decompositions -- each counted as
        // its first sample's --, LV per leg sum and V⁻¹U per amplitude sum, per table its products (per element and
        // basis pair |B| terms of the series' `orders` powers, the reciprocal, the product); per piece and table its
        // Chebyshev transform.
        let decompositions = {
            let mut once = 0.0;
            let mut seen = vec![false; plan.blocks.len()];
            for &b in &used {
                if !std::mem::replace(&mut seen[b], true) {
                    let mut sp = Spectrum::default();
                    if sp.decompose(&Tables::generator(plan, b, &Tables::at_intensity(plan, rest, low))) {
                        once += sp.work;
                    }
                }
            }
            once
        };
        let per_sample: f64 = decompositions
            + grouping.ws.iter().zip(&w_vectors).filter(|(s, _)| used.contains(&s.block)).map(|(s, v)| (s.index.len() * plan.blocks[s.block].rows.len().pow(2) * v.len()) as f64).sum::<f64>()
            + grouping.zs.iter().zip(&z_vectors).map(|(s, v)| (s.index.len() * plan.blocks[s.block].rows.len().pow(2) * v.len()) as f64).sum::<f64>()
            + tables.iter().map(|t| (t.w.map_or(1, |w| ws[w].rank) * zs[t.z].rank * t.rows * t.cols * plan.blocks[grouping.zs[t.z].block].rows.len() * (orders + 2)) as f64).sum::<f64>();
        let transform: f64 = tables.iter().map(|t| (t.w.map_or(1, |w| ws[w].rank) * zs[t.z].rank * orders * m * m * t.rows * t.cols) as f64).sum();
        let price = pieces.len() as f64 * (m as f64 * per_sample + transform);
        Some(Layout { degree, orders, pieces, dropped, zs, ws, z_vectors, w_vectors, tables, e_rest: e0, e_slope, used, s_mats, work, price, seconds: clock.elapsed().as_secs_f64() })
    }
}

impl Layout {
    /// The tables of a layout (`Tables::layout`): per piece the samples at its m Chebyshev points -- each table's
    /// entries Σ_q (LV)_iq (V⁻¹U)_qj (z₀ − λ_q)^{−(J+1)} for J < `orders` -- and their Chebyshev coefficients. None
    /// where a block does not decompose.
    pub(crate) fn fill(self, plan: &Plan, grouping: &Grouping, rest: &Values) -> Option<Tables> {
        let clock = std::time::Instant::now();
        let Layout { degree, orders, pieces, dropped, zs, ws, z_vectors, w_vectors, mut tables, e_rest, e_slope, used, s_mats, work, seconds, .. } = self;
        let m = degree + 1;
        let xs: Vec<f64> = (0..m).map(|n| (std::f64::consts::PI * (n as f64 + 0.5) / m as f64).cos()).collect();
        let mut filled = 0.0;
        for p in &pieces {
            let [a, b] = p.range;
            let mut samples: Vec<Vec<C64>> = tables.iter().map(|t| vec![C64::ZERO; t.w.map_or(1, |w| ws[w].rank) * zs[t.z].rank * orders * m * t.rows * t.cols]).collect();
            for (n, &x) in xs.iter().enumerate() {
                let (energies, spectra, _, w) = Tables::sample(plan, rest, &used, &s_mats, 0.5 * (a + b) + 0.5 * (b - a) * x, false)?;
                filled += w;
                // LV per leg sum and basis vector (rows × |B|), V⁻¹U per amplitude sum and basis vector (|B| × cols).
                let mut lvs: Vec<Vec<Vec<C64>>> = Vec::with_capacity(grouping.ws.len());
                for (s, vectors) in grouping.ws.iter().zip(&w_vectors) {
                    let Some(sp) = spectra[s.block].as_ref() else {
                        lvs.push(Vec::new());
                        continue;
                    };
                    let (size, rows) = (sp.centre.len(), s.index.len());
                    filled += (vectors.len() * rows * size * size) as f64;
                    lvs.push(vectors.iter().map(|l| linalg_product(l, &sp.vectors.data, rows, size, size)).collect());
                }
                let mut vus: Vec<Vec<Vec<C64>>> = Vec::with_capacity(grouping.zs.len());
                for (s, vectors) in grouping.zs.iter().zip(&z_vectors) {
                    let sp = spectra[s.block].as_ref().expect("a sum's block decomposed");
                    let (size, cols) = (sp.centre.len(), s.index.len());
                    filled += (vectors.len() * cols * size * size) as f64;
                    vus.push(vectors.iter().map(|u| linalg_product(&sp.inverse.data, u, size, size, cols)).collect());
                }
                for (out, t) in samples.iter_mut().zip(&tables) {
                    let block = grouping.zs[t.z].block;
                    let sp = spectra[block].as_ref().expect("a sum's block decomposed");
                    let size = sp.centre.len();
                    let (rows, cols, per) = (t.rows, t.cols, t.rows * t.cols);
                    let rz = zs[t.z].rank;
                    let z0: Vec<f64> = grouping.zs[t.z].index.iter().map(|&col| energies[col] + zs[t.z].f0).collect();
                    let lambda: Vec<C64> = (0..size).map(|k| C64::real(sp.centre[k]) + sp.values[k]).collect();
                    let identity = [sp.vectors.data.clone()];
                    let lv: &[Vec<C64>] = match t.w {
                        Some(w) => &lvs[w],
                        None => &identity,
                    };
                    let vu = &vus[t.z];
                    let rw = lv.len();
                    let mut sums = vec![C64::ZERO; orders];
                    for kp in 0..rw {
                        for k in 0..rz {
                            for i in 0..rows {
                                for j in 0..cols {
                                    sums.iter_mut().for_each(|x| *x = C64::ZERO);
                                    for q in 0..size {
                                        let d = (C64::real(z0[j]) - lambda[q]).recip();
                                        let mut x = lv[kp][i * size + q] * vu[k][q * cols + j];
                                        for t in sums.iter_mut() {
                                            x = x * d;
                                            *t += x;
                                        }
                                    }
                                    for (jj, t) in sums.iter().enumerate() {
                                        out[((((kp * rz + k) * orders + jj) * m) + n) * per + i * cols + j] = *t;
                                    }
                                }
                            }
                        }
                    }
                    filled += (rw * rz * rows * cols * size * (orders + 2)) as f64;
                }
            }
            // Chebyshev coefficients from the samples at the Gauss points.
            for (smp, t) in samples.iter().zip(tables.iter_mut()) {
                let per = t.rows * t.cols;
                let count = smp.len() / (m * per);
                let mut coef = vec![C64::ZERO; smp.len()];
                for c in 0..count {
                    for d in 0..m {
                        let scale = if d == 0 { 1.0 / m as f64 } else { 2.0 / m as f64 };
                        for (n, x) in xs.iter().enumerate() {
                            let w = (d as f64 * x.acos()).cos() * scale;
                            for e in 0..per {
                                coef[(c * m + d) * per + e] += smp[(c * m + n) * per + e] * w;
                            }
                        }
                    }
                }
                filled += (count * m * m * per) as f64;
                t.coef.extend_from_slice(&coef);
            }
        }
        let bytes = tables.iter().map(|t| t.coef.len() * std::mem::size_of::<C64>()).sum();
        Some(Tables { degree, orders, pieces, dropped, zs, ws, tables, e_rest, e_slope, build_seconds: seconds + clock.elapsed().as_secs_f64(), bytes, work: work + filled })
    }
}

/// a (r × k) times b (k × c), row-major.
fn linalg_product(a: &[C64], b: &[C64], r: usize, k: usize, c: usize) -> Vec<C64> {
    let mut o = vec![C64::ZERO; r * c];
    for i in 0..r {
        for q in 0..k {
            let x = a[i * k + q];
            if x != C64::ZERO {
                for j in 0..c {
                    o[i * c + j] += x * b[q * c + j];
                }
            }
        }
    }
    o
}

impl Tables {
    /// The tables' reading at the values `v` (`Reading`; # The bounds): None where s lies outside the pieces, the
    /// piece's gap bound is not positive, or ΔG reaches a gap.
    pub(crate) fn read(&self, plan: &Plan, grouping: &Grouping, v: &Values, scratch: &mut Scratch) -> Option<Reading> {
        let f = plan.intensity[0];
        let (s, ds) = (v.factors[f].0.re, v.factors[f].1.map(|d| d.re));
        let piece = self.pieces.iter().position(|p| s >= p.range[0] && s <= p.range[1])?;
        let p = &self.pieces[piece];
        // A piece whose interpolation no ellipse bounds (`bernstein`): the node is the fold's own.
        if !p.worst.iter().all(|w| w.is_finite()) {
            return None;
        }
        self.shifts(plan, v, s, ds, scratch);
        let de = scratch.de.iter().map(|x| x.0.abs()).fold(0.0, f64::max);
        let de_gradient = scratch.de.iter().map(|x| (x.1[0] * x.1[0] + x.1[1] * x.1[1] + x.1[2] * x.1[2]).sqrt()).fold(0.0, f64::max);
        let least_gap = p.gaps.iter().copied().fold(f64::INFINITY, f64::min);
        let mut rho: f64 = 0.0;
        for (z, (b, sum)) in self.zs.iter().zip(&grouping.zs).enumerate() {
            if !plan.made(sum.block, true) {
                continue;
            }
            let gap = p.gaps[z];
            if gap <= 0.0 {
                return None;
            }
            for (member, _) in &sum.members {
                let delta = v.frequencies[plan.amplitudes[*member].piece] - b.f0;
                rho = rho.max((delta.abs() + de) / gap);
            }
        }
        if rho >= 1.0 {
            return None;
        }
        // ΔG per block (its Frobenius norm and its gradient's): the entries the tables hold at rest, at the point less at
        // the tables' values; per amplitude sum its block's against the sum's gap.
        let (mut moved, mut moved_gradient): (f64, f64) = (0.0, 0.0);
        // The reading's count (complex multiply-adds): K's shifts -- each diagonal entry's value and gradients, each
        // state's table energy --, the ratio over the members, ΔG's entries' value and three gradients.
        let mut work = 4.0 * (plan.diagonal.len() + plan.n) as f64 + REAL * (6 * plan.n + 3 * grouping.zs.iter().map(|z| z.members.len()).sum::<usize>()) as f64;
        let sizes = &mut scratch.sizes;
        sizes.clear();
        sizes.resize(plan.blocks.len(), None);
        for (z, sum) in grouping.zs.iter().enumerate() {
            let b = sum.block;
            if !plan.made(b, true) {
                continue;
            }
            let (value, gradient) = *sizes[b].get_or_insert_with(|| {
                work += 8.0 * plan.blocks[b].rest.len() as f64;
                let (mut value, mut gradient) = (0.0, 0.0);
                for e in &plan.blocks[b].rest {
                    let (fi, scale) = plan.generator_terms[e.term as usize];
                    let (c_point, c_table) = (v.coefficients[e.coefficient as usize], if e.coefficient == 0 { 1.0 } else { 0.0 });
                    let (x, g) = v.factors[fi];
                    let (x_table, g_table) = if plan.own.contains(&fi) { (C64::ONE, [C64::ZERO; 3]) } else { (x, g) };
                    value += (scale * e.value * (x * c_point - x_table * c_table)).norm_sqr();
                    for a in 0..3 {
                        gradient += (scale * e.value * (g[a] * c_point - g_table[a] * c_table)).norm_sqr();
                    }
                }
                (value.sqrt(), gradient.sqrt())
            });
            let (kappa, gap) = (p.kappa[b], p.gaps[z]);
            let size = kappa * value;
            if size >= gap {
                return None;
            }
            moved = moved.max(size / (gap - size));
            moved_gradient = moved_gradient.max(kappa * gradient / gap);
        }
        Some(Reading { piece, s, ds, rho, moved, moved_gradient, shift_gradient: de_gradient / least_gap, work })
    }

    /// ΔE_a and its gradient at the values `v`: K's energies there less the tables' E_a(s).
    fn shifts(&self, plan: &Plan, v: &Values, s: f64, ds: [f64; 3], scratch: &mut Scratch) {
        scratch.de.clear();
        scratch.de.extend((0..plan.n).map(|a| (-(self.e_rest[a] + self.e_slope[a] * s), ds.map(|d| -self.e_slope[a] * d))));
        for d in &plan.diagonal {
            let (f, scale) = plan.generator_terms[d.term as usize];
            let (x, g) = v.factors[f];
            let c = d.value * v.coefficients[d.coefficient as usize];
            let e = &mut scratch.de[d.row as usize];
            e.0 += (scale * x * c).re;
            for a in 0..3 {
                e.1[a] += (scale * g[a] * c).re;
            }
        }
    }

    /// The contraction at the values `v` into `out` (as `Out::contract` lays it, `making` a node's, the weights one),
    /// the tables read at `reading` to `orders` Doppler terms (at most the tables'). Returns the counted complex
    /// multiply-adds.
    pub(crate) fn contract(&self, plan: &Plan, grouping: &Grouping, v: &Values, reading: &Reading, orders: usize, making: Making, out: &mut Out, scratch: &mut Scratch) -> f64 {
        let orders = orders.min(self.orders);
        let one = |_: usize| C64::ONE;
        let mut work = out.direct(plan, grouping, v, &one, making);
        let moved = MOVED;
        // The point's coordinates: per atom its factor × scale × coefficient (and gradient).
        let point = |piece: usize, term: usize, coefficient: usize| -> (C64, [C64; 3]) {
            let (f, scale) = plan.terms[piece][term];
            let (x, g) = v.factors[f];
            let c = v.coefficients[coefficient];
            (scale * x * c, g.map(|d| scale * d * c))
        };
        let Scratch { interpolated, phi, psi, psi_most, de, t, dt, px, py, value, slope, grad, .. } = scratch;
        phi.resize_with(self.ws.len(), Vec::new);
        for (b, phi) in self.ws.iter().zip(phi.iter_mut()) {
            phi.clear();
            phi.resize(b.rank, (C64::ZERO, [C64::ZERO; 3]));
            work += 4.0 * moved * b.rank as f64;
            for a in &b.atoms {
                if v.coefficients[a.coefficient] == 0.0 {
                    continue;
                }
                let (x, g) = point(a.piece, a.term, a.coefficient);
                for (p, c) in phi.iter_mut().zip(&a.coords) {
                    p.0 += *c * x;
                    for d in 0..3 {
                        p.1[d] += *c * g[d];
                    }
                }
                work += 4.0 + 4.0 * a.coords.len() as f64;
            }
        }
        psi.resize_with(self.zs.len(), Vec::new);
        for ((b, psi), sum) in self.zs.iter().zip(psi.iter_mut()).zip(&grouping.zs) {
            psi.resize_with(orders, Vec::new);
            for o in psi.iter_mut() {
                o.resize_with(b.rank, Vec::new);
                for p in o.iter_mut() {
                    p.clear();
                    p.resize(sum.index.len(), (C64::ZERO, [C64::ZERO; 3]));
                }
            }
            work += 4.0 * moved * (orders * b.rank * sum.index.len()) as f64;
            if !plan.made(sum.block, making.model) {
                continue;
            }
            for a in &b.atoms {
                if v.coefficients[a.coefficient] == 0.0 {
                    continue;
                }
                let (x, g) = point(a.piece, a.term, a.coefficient);
                let delta = v.frequencies[a.piece] - b.f0;
                for (j, &col) in sum.index.iter().enumerate() {
                    // (−(δ + ΔE_a))^J and its gradient −J (−(δ + ΔE_a))^{J−1} ∇ΔE_a.
                    let (minus, dminus) = (-(delta + de[col].0), de[col].1.map(|d| -d));
                    let (mut power, mut previous) = (1.0, 0.0);
                    for (jj, o) in psi.iter_mut().enumerate() {
                        let dpower: [f64; 3] = dminus.map(|d| jj as f64 * previous * d);
                        let xp = x * power;
                        let gp: [C64; 3] = std::array::from_fn(|d| g[d] * power + x * dpower[d]);
                        for (p, c) in o.iter_mut().zip(&a.coords) {
                            let q = &mut p[j];
                            q.0 += *c * xp;
                            for d in 0..3 {
                                q.1[d] += *c * gp[d];
                            }
                        }
                        work += 4.0 * a.coords.len() as f64 + 4.0;
                        previous = power;
                        power *= minus;
                    }
                }
            }
        }
        // Per sum, term and basis vector the largest |ψ| and |∇ψ| over its columns (the interpolation's bound): each
        // number's modulus and its gradient's.
        psi_most.resize_with(self.zs.len(), Vec::new);
        for (most, psi) in psi_most.iter_mut().zip(psi.iter()) {
            most.clear();
            most.extend(psi.iter().flat_map(|o| o.iter()).map(|p| p.iter().fold([0.0, 0.0], |m: [f64; 2], q| [m[0].max(q.0.abs()), m[1].max((q.1[0].norm_sqr() + q.1[1].norm_sqr() + q.1[2].norm_sqr()).sqrt())])));
            // Per number its modulus and its gradient's norm: some ten real operations.
            work += 10.0 * REAL * most.len() as f64 * psi.first().and_then(|o| o.first()).map_or(0, Vec::len) as f64;
        }
        interpolated.h.clear();
        interpolated.h.resize(grouping.outputs.len(), [0.0; 2]);
        interpolated.channels.clear();
        interpolated.channels.resize(grouping.channel_outputs.len(), [0.0; 2]);
        interpolated.lift.clear();
        interpolated.lift.resize(grouping.labels.len(), [0.0; 2]);
        let ds_norm = (reading.ds[0] * reading.ds[0] + reading.ds[1] * reading.ds[1] + reading.ds[2] * reading.ds[2]).sqrt();
        // The Chebyshev values at s on the piece, and their slope in s.
        let p = &self.pieces[reading.piece];
        let [pa, pb] = p.range;
        let x = if pb > pa { (2.0 * reading.s - pa - pb) / (pb - pa) } else { 0.0 };
        chebyshev(x, self.degree, t, dt);
        for d in dt.iter_mut() {
            *d *= 2.0 / (pb - pa);
        }
        let m = self.degree + 1;
        let ds = reading.ds;
        if making.lift {
            for (l, g) in out.lift.iter_mut() {
                zero(l, plan.nq, plan.n);
                for d in g.iter_mut() {
                    zero(d, if making.gradients { plan.nq } else { 0 }, plan.n);
                }
            }
            work += moved * ((if making.gradients { 4 } else { 1 }) * plan.nq * plan.n * out.lift.len()) as f64;
        }
        for (ti, tb) in self.tables.iter().enumerate() {
            let jump = tb.w.and_then(|w| grouping.ws[w].jump);
            let wanted = match tb.target {
                Target::H(_) => making.model && making.effective,
                Target::Jump(_) => making.model && !(jump.is_some_and(|(_, which)| which == 1) && !making.kicks),
                Target::Lift(_) => making.lift,
            };
            if !wanted || !plan.made(grouping.zs[tb.z].block, making.model) {
                continue;
            }
            let per = tb.rows * tb.cols;
            let (rw, rz) = (tb.w.map_or(1, |w| self.ws[w].rank), self.zs[tb.z].rank);
            value.clear();
            value.resize(per, C64::ZERO);
            slope.clear();
            slope.resize(per, C64::ZERO);
            for g in grad.iter_mut() {
                g.clear();
                g.resize(per, C64::ZERO);
            }
            let psi = &psi[tb.z];
            let piece_stride = rw * rz * self.orders * m * per;
            for kp in 0..rw {
                let a = tb.w.map_or((C64::ONE, [C64::ZERO; 3]), |w| phi[w][kp]);
                for k in 0..rz {
                    for (jj, o) in psi.iter().enumerate() {
                        let base = reading.piece * piece_stride + (((kp * rz + k) * self.orders + jj) * m) * per;
                        let tab = &tb.coef[base..base + m * per];
                        // The table and its slope at s, element by element along its rows (contiguous).
                        px.clear();
                        px.resize(per, C64::ZERO);
                        py.clear();
                        py.resize(per, C64::ZERO);
                        for d in 0..m {
                            let (td, tdd) = (t[d], dt[d]);
                            for ((x, y), z) in px.iter_mut().zip(py.iter_mut()).zip(&tab[d * per..(d + 1) * per]) {
                                *x += *z * td;
                                *y += *z * tdd;
                            }
                        }
                        for (j, da) in o[k].iter().enumerate() {
                            let c = a.0 * da.0;
                            let dc: [C64; 3] = std::array::from_fn(|x| a.1[x] * da.0 + a.0 * da.1[x]);
                            for i in 0..tb.rows {
                                let e = i * tb.cols + j;
                                let x = px[e];
                                value[e] += c * x;
                                slope[e] += c * py[e];
                                for a in 0..3 {
                                    grad[a][e] += dc[a] * x;
                                }
                            }
                        }
                        // The table at s and its slope (2m a element), the coordinates' product and its gradients
                        // (four), and per element its value, slope and three gradients.
                        work += (2 * m * per) as f64 + 4.0 * tb.cols as f64 + 5.0 * per as f64;
                    }
                }
            }
            for e in 0..per {
                for a in 0..3 {
                    grad[a][e] += slope[e] * ds[a];
                }
            }
            work += 3.0 * per as f64;
            // What its interpolation leaves on each element (# The interpolation's bound): Σ |φ| |ψ| E, its gradient
            // Σ (|∇φ| |ψ| + |φ| |∇ψ|) E + |φ| |ψ| E′ |∇s|, each |ψ| its largest over the columns.
            let bounds = &p.bounds[ti * self.orders..];
            let most = &psi_most[tb.z];
            let (mut sums, mut through, mut slopes) = (0.0, 0.0, 0.0);
            for jj in 0..psi.len() {
                let [e, e_slope] = bounds[jj];
                for k in 0..rz {
                    let [x, dx] = most[jj * rz + k];
                    sums += x * e;
                    through += dx * e;
                    slopes += x * e_slope;
                }
            }
            let (mut bound, mut gradient_bound) = (0.0, 0.0);
            for kp in 0..rw {
                let (x, g) = tb.w.map_or((1.0, 0.0), |w| {
                    let a = phi[w][kp];
                    (a.0.abs(), (a.1[0].norm_sqr() + a.1[1].norm_sqr() + a.1[2].norm_sqr()).sqrt())
                });
                bound += x * sums;
                gradient_bound += g * sums + x * (through + slopes * ds_norm);
            }
            work += REAL * (6 * psi.len() * rz + 10 * rw) as f64;
            match tb.target {
                Target::H(i) => {
                    let pair = &grouping.h_pairs[i];
                    // ½ and its conjugate: each half on one output, both on one where the two are one.
                    let halves = if pair.out == pair.conjugate { [(pair.out, 1.0), (pair.out, 0.0)] } else { [(pair.out, 0.5), (pair.conjugate, 0.5)] };
                    for (o, share) in halves {
                        let x = &mut interpolated.h[o];
                        x[0] += share * bound;
                        x[1] += share * gradient_bound;
                    }
                }
                Target::Jump(i) => {
                    let x = &mut interpolated.channels[grouping.j_pairs[i].out];
                    x[0] += bound;
                    x[1] += gradient_bound;
                }
                Target::Lift(l) => {
                    let x = &mut interpolated.lift[grouping.lift[l].1];
                    x[0] += bound;
                    x[1] += gradient_bound;
                }
            }
            match tb.target {
                Target::H(i) => {
                    let pair = &grouping.h_pairs[i];
                    for e in 0..per {
                        let (x, dx) = (value[e] * 0.5, [grad[0][e] * 0.5, grad[1][e] * 0.5, grad[2][e] * 0.5]);
                        let (k, kc) = (pair.places[e] as usize, pair.conjugate_places[e] as usize);
                        let o = &mut out.h[pair.out];
                        o.value[k] += x;
                        for a in 0..3 {
                            o.gradient[a][k] += dx[a];
                        }
                        let o = &mut out.h[pair.conjugate];
                        o.value[kc] += x.conj();
                        for a in 0..3 {
                            o.gradient[a][kc] += dx[a].conj();
                        }
                    }
                    work += per as f64 * (2.0 + 8.0 * moved);
                }
                Target::Jump(i) => {
                    let pair = &grouping.j_pairs[i];
                    let o = &mut out.channels[pair.out];
                    for e in 0..per {
                        let k = pair.places[e] as usize;
                        o.value[k] += value[e];
                        for a in 0..3 {
                            o.gradient[a][k] += grad[a][e];
                        }
                    }
                    work += 4.0 * moved * per as f64;
                }
                Target::Lift(l) => {
                    let (b, group, _) = &grouping.lift[l];
                    let (lv, lg) = &mut out.lift[*group];
                    let n = plan.n;
                    for (i, &row) in plan.blocks[*b].rows.iter().enumerate() {
                        for (j, &a) in grouping.zs[tb.z].index.iter().enumerate() {
                            let e = i * tb.cols + j;
                            lv.data[row * n + a] += value[e];
                            if making.gradients {
                                for d in 0..3 {
                                    lg[d].data[row * n + a] += grad[d][e];
                                }
                            }
                        }
                    }
                    work += moved * ((if making.gradients { 4 } else { 1 }) * per) as f64;
                }
            }
        }
        // Each output's bound over its largest element at the node (`Interpolated::relative`): one pass over the
        // outputs' values and gradients.
        let mut relative = [0.0f64; 2];
        let mut over = |bound: [f64; 2], values: &mut dyn Iterator<Item = C64>, gradients: &mut dyn Iterator<Item = C64>| {
            let most = values.fold(0.0f64, |m, z| m.max(z.abs()));
            let gmost = gradients.fold(0.0f64, |m, z| m.max(z.abs()));
            if most > 0.0 {
                relative[0] = relative[0].max(bound[0] / most);
            }
            if gmost > 0.0 {
                relative[1] = relative[1].max(bound[1] / gmost);
            }
        };
        let mut elements = 0usize;
        for (o, &b) in out.h.iter().zip(&interpolated.h).chain(out.channels.iter().zip(&interpolated.channels)) {
            if b != [0.0; 2] {
                over(b, &mut o.value.iter().copied(), &mut o.gradient.iter().flat_map(|g| g.iter().copied()));
                elements += o.value.len();
            }
        }
        if making.lift {
            for ((v, g), &b) in out.lift.iter().zip(&interpolated.lift) {
                if b != [0.0; 2] {
                    over(b, &mut v.data.iter().copied(), &mut g.iter().flat_map(|m| m.data.iter().copied()));
                    elements += v.data.len();
                }
            }
        }
        interpolated.relative = relative;
        // Per element its modulus and its three gradients': some twelve real operations.
        work += 12.0 * REAL * elements as f64;
        out.work = work;
        work
    }
}
