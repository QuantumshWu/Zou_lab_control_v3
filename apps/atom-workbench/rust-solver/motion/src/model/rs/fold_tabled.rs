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
//! - the interpolation: each piece's tables' last two Chebyshev coefficients against the pair's largest (`Piece::tail`,
//!   the series' own estimate of what the interpolant leaves out), and the spans' dropped share (`Tables::dropped`).
//! The tail and ΔG's part are relative to each eigen-term's modulus -- an element's own scale, the moduli of the terms
//! it is the sum of (`Out::scales`) --, the rest to the sums' scale (the output's largest). A gradient's tail is the
//! series' terms' own s-derivatives, (J + 1) R^{J+2} on the J-th: Σ_{J≥O} (J + 1) ρ^J = ρ^O (O + 1 − Oρ)/(1 − ρ)², and
//! the terms' gradients through δ's, O ρ^{O−1}/(1 − ρ)² times |∇ΔE|/g (`Reading::shift_gradient`) on the value's scale.
//! Measured (model/selected/tabled.rs, 40 walk points each of PGC3D, TET, tilt0, tilt2 at 1–3 terms): every element of H,
//! the channels and the lift, values and gradients, within these bounds, at most 0.99 of them (tilt0's channels: the
//! bound is reached).
//!
//! # The pieces
//! A piece [a, b] of s holds where its tail is within the Doppler series' tail at the tables' terms at the reach (the
//! largest ratio of the members' shifts there to the sums' gaps at s = 1: a piece never adds more to a node's bound than
//! the series takes at the run's fastest atom), or where every table's last two coefficients are within the rounding of
//! the samples that made them (γ_n of the moduli of their terms, n = |B|(O + 1) + m); else it is halved, but where the
//! halves' tails no longer fall by half the piece is taken at its tail. (The last two coefficients of a degree-4 piece
//! fall as its width cubed: tables to the samples' rounding take PGC3D 1154 pieces and 1.3 GB, to the reach's tail 8 and
//! 9 MB.)
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
    pub tail: f64,
}

/// The tables of a run's fold (module documentation): made once (`Tables::new`), read by every node of the run.
#[derive(Clone, Debug)]
pub(crate) struct Tables {
    pub degree: usize,
    /// The Doppler series' terms the tables hold (J < `orders`).
    pub orders: usize,
    pub pieces: Vec<Piece>,
    /// The spans' dropped share, the largest over the sums (relative to each kind's largest atom).
    pub dropped: f64,
    zs: Vec<Basis>,
    ws: Vec<Basis>,
    tables: Vec<Table>,
    /// K's energies the tables are made at, E_a(s) = A_a + B_a s.
    e_rest: Vec<f64>,
    e_slope: Vec<f64>,
    pub build_seconds: f64,
    pub bytes: usize,
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
}

impl Reading {
    /// The Doppler series' tail at `orders` terms relative to each eigen-term's leading one, ρ^O/(1 − ρ).
    pub(crate) fn tail(&self, orders: usize) -> f64 {
        self.rho.powi(orders as i32) / (1.0 - self.rho)
    }
}

/// A node's buffers (one a thread): φ and its gradient per leg sum; ψ per amplitude sum, order, basis vector and
/// column; ΔE_a and its gradient; the Chebyshev values; a table's value and slope at s; a pair's sums.
#[derive(Clone, Debug, Default)]
pub(crate) struct Scratch {
    phi: Vec<Vec<(C64, [C64; 3])>>,
    psi: Vec<Vec<Vec<Vec<(C64, [C64; 3])>>>>,
    de: Vec<(f64, [f64; 3])>,
    t: Vec<f64>,
    dt: Vec<f64>,
    px: Vec<C64>,
    py: Vec<C64>,
    value: Vec<C64>,
    slope: Vec<C64>,
    grad: [Vec<C64>; 3],
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

    /// The tweezer's intensity s at the values `v` (a plan with tables has one).
    pub(crate) fn intensity(plan: &Plan, v: &Values) -> f64 {
        v.factors[plan.intensity[0]].0.re
    }

    /// Whether a factor is one the tables hold at s or at one (the intensity, a beam's own intensity).
    fn held(plan: &Plan, f: usize) -> bool {
        plan.intensity.contains(&f) || plan.own.contains(&f)
    }

    /// The tables of a plan's grouping (module documentation): `rest` the values at a point at rest (the members'
    /// frequencies f₀), `reach` at a point at the reach's speed (the atoms' point coefficients' bounds), `orders` the
    /// Doppler series' terms kept, `degree` the pieces' Chebyshev degree in s on [`low`, 1]. None where a block a sum is
    /// on is not tabled (its generator varies by another factor than the intensities) or the plan has no intensity.
    pub(crate) fn new(plan: &Plan, grouping: &Grouping, rest: &Values, reach: &Values, orders: usize, degree: usize, low: f64) -> Option<Tables> {
        let clock = std::time::Instant::now();
        if plan.intensity.is_empty() {
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
        // An atom's point coefficient's bound: its factor's and coefficient's moduli, the larger of rest's and reach's.
        let bound = |piece: usize, term: usize, coefficient: usize| -> f64 {
            let (f, scale) = plan.terms[piece][term];
            let x = rest.factors[f].0.abs().max(reach.factors[f].0.abs());
            let c = rest.coefficients[coefficient].abs().max(reach.coefficients[coefficient].abs());
            scale.abs() * x * c
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
        let mut gap_one: Vec<f64> = Vec::with_capacity(grouping.zs.len());
        for s in &grouping.zs {
            let mut sp = Spectrum::default();
            if !sp.decompose(&Tables::generator(plan, s.block, &at_one)) {
                return None;
            }
            let f0 = f0_of(s);
            gap_one.push(s.index.iter().flat_map(|&col| (0..sp.centre.len()).map(move |k| (col, k))).map(|(col, k)| (C64::real(e_one[col] + f0 - sp.centre[k]) - sp.values[k]).abs()).fold(f64::INFINITY, f64::min));
        }
        let norm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let top_of = |(keys, mats): &Atoms| keys.iter().zip(mats).map(|(&(p, t, c), m)| norm(m) * bound(p, t, c)).fold(0.0, f64::max);
        let z_top = z_atoms.iter().zip(&gap_one).map(|(a, g)| top_of(a) / g).fold(0.0, f64::max);
        let w_top = w_atoms.iter().map(top_of).fold(0.0, f64::max);
        let mut dropped: f64 = 0.0;
        let mut z_vectors = Vec::new();
        let mut basis_of = |(keys, mats): Atoms, f0: f64, top: f64, vectors: &mut Vec<Vec<Vec<C64>>>| -> Basis {
            let bounds: Vec<f64> = keys.iter().map(|&(p, t, c)| bound(p, t, c)).collect();
            let limit = keys.len() as f64 * UNIT_ROUNDOFF * top;
            let (basis, coords, d) = span(&mats, &bounds, limit, top);
            dropped = dropped.max(d);
            vectors.push(basis);
            Basis { atoms: keys.iter().zip(coords).map(|(&(piece, term, coefficient), coords)| Atom { piece, term, coefficient, coords }).collect(), rank: vectors.last().map_or(0, Vec::len), f0 }
        };
        let mut zs: Vec<Basis> = Vec::new();
        for ((s, atoms), g) in grouping.zs.iter().zip(z_atoms).zip(&gap_one) {
            zs.push(basis_of(atoms, f0_of(s), z_top * g, &mut z_vectors));
        }
        let mut w_vectors = Vec::new();
        let mut ws: Vec<Basis> = Vec::new();
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
        // Per block S = ∂G₀/∂s (G₀ is linear in s on a tabled block).
        let slope: Vec<f64> = (0..plan.blocks.len())
            .map(|b| {
                if !used.contains(&b) {
                    return 0.0;
                }
                let (g0, g1) = (Tables::generator(plan, b, &Tables::at_intensity(plan, rest, 0.0)), Tables::generator(plan, b, &at_one));
                g0.data.iter().zip(&g1.data).map(|(a, b)| (*b - *a).norm_sqr()).sum::<f64>().sqrt()
            })
            .collect();
        let e0 = Tables::energies(plan, &Tables::at_intensity(plan, rest, 0.0));
        let e_slope: Vec<f64> = e0.iter().zip(&e_one).map(|(a, b)| b - a).collect();
        let b_most = e_slope.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        let m = degree + 1;
        let largest_block = used.iter().map(|&b| plan.blocks[b].rows.len()).max().unwrap_or(0);
        // The interpolation's target: the Doppler series' tail at the tables' terms at the reach (the largest ratio its
        // members' shifts there take of the sums' gaps at s = 1), so that a piece never adds more to a node's bound than
        // the series it holds takes at the run's fastest atom.
        let reach_rho = grouping
            .zs
            .iter()
            .zip(&gap_one)
            .flat_map(|(s, g)| s.members.iter().map(move |(m, _)| (reach.frequencies[plan.amplitudes[*m].piece] - f0_of(s)).abs() / g))
            .fold(0.0, f64::max);
        let target = if reach_rho < 1.0 { reach_rho.powi(orders as i32) / (1.0 - reach_rho) } else { 0.0 };
        // A sample element's rounding: γ_n of the moduli of the terms it is the sum of (`moduli`), n the operations
        // behind it and its transform.
        let floor = {
            let n = (largest_block * (orders + 1) + m) as f64;
            n * UNIT_ROUNDOFF / (1.0 - n * UNIT_ROUNDOFF)
        };
        // One piece [a, b]: its tables' coefficients, its gaps' lower bounds per z sum, κ per block, its tail.
        let make = |a: f64, b: f64| -> Option<(Vec<Vec<C64>>, Piece, bool)> {
            let xs: Vec<f64> = (0..m).map(|n| (std::f64::consts::PI * (n as f64 + 0.5) / m as f64).cos()).collect();
            let mut gaps: Vec<f64> = vec![f64::INFINITY; grouping.zs.len()];
            let mut kappa: Vec<f64> = vec![0.0; plan.blocks.len()];
            let mut samples: Vec<Vec<C64>> = tables.iter().map(|t| vec![C64::ZERO; t.w.map_or(1, |w| ws[w].rank) * zs[t.z].rank * orders * m * t.rows * t.cols]).collect();
            // Per table the largest sum of its sample elements' terms' moduli.
            let mut moduli: Vec<f64> = vec![0.0; tables.len()];
            for (n, &x) in xs.iter().enumerate() {
                let sv = 0.5 * (a + b) + 0.5 * (b - a) * x;
                let v = Tables::at_intensity(plan, rest, sv);
                let energies = Tables::energies(plan, &v);
                let mut spectra: Vec<Option<Spectrum>> = vec![None; plan.blocks.len()];
                for &bl in &used {
                    if spectra[bl].is_some() {
                        continue;
                    }
                    let mut sp = Spectrum::default();
                    if !sp.decompose(&Tables::generator(plan, bl, &v)) {
                        return None;
                    }
                    kappa[bl] = kappa[bl].max(norm(&sp.vectors.data) * norm(&sp.inverse.data));
                    spectra[bl] = Some(sp);
                }
                for (z, sum) in grouping.zs.iter().enumerate() {
                    let sp = spectra[sum.block].as_ref().expect("a sum's block decomposed");
                    for &col in &sum.index {
                        let z0 = energies[col] + zs[z].f0;
                        for k in 0..sp.centre.len() {
                            gaps[z] = gaps[z].min((C64::real(z0 - sp.centre[k]) - sp.values[k]).abs());
                        }
                    }
                }
                for ((out, t), modulus) in samples.iter_mut().zip(&tables).zip(moduli.iter_mut()) {
                    let block = grouping.zs[t.z].block;
                    let sp = spectra[block].as_ref().expect("a sum's block decomposed");
                    let size = plan.blocks[block].rows.len();
                    let (rows, cols, per) = (t.rows, t.cols, t.rows * t.cols);
                    let rz = zs[t.z].rank;
                    let z0: Vec<f64> = grouping.zs[t.z].index.iter().map(|&col| energies[col] + zs[t.z].f0).collect();
                    let lambda: Vec<C64> = (0..size).map(|k| C64::real(sp.centre[k]) + sp.values[k]).collect();
                    // L V per leg basis vector (rows × |B|): the identity's V for a lift.
                    let lv: Vec<Vec<C64>> = match t.w {
                        Some(w) => w_vectors[w]
                            .iter()
                            .map(|l| {
                                let mut o = vec![C64::ZERO; rows * size];
                                for i in 0..rows {
                                    for q in 0..size {
                                        let x = l[i * size + q];
                                        if x != C64::ZERO {
                                            for k in 0..size {
                                                o[i * size + k] += x * sp.vectors.data[q * size + k];
                                            }
                                        }
                                    }
                                }
                                o
                            })
                            .collect(),
                        None => vec![sp.vectors.data.clone()],
                    };
                    let rw = lv.len();
                    // V⁻¹ U per amplitude basis vector (|B| × cols).
                    let vu: Vec<Vec<C64>> = z_vectors[t.z]
                        .iter()
                        .map(|u| {
                            let mut o = vec![C64::ZERO; size * cols];
                            for k in 0..size {
                                for q in 0..size {
                                    let x = sp.inverse.data[k * size + q];
                                    if x != C64::ZERO {
                                        for j in 0..cols {
                                            o[k * cols + j] += x * u[q * cols + j];
                                        }
                                    }
                                }
                            }
                            o
                        })
                        .collect();
                    let (mut sums, mut moduli_sums) = (vec![C64::ZERO; orders], vec![0.0; orders]);
                    for kp in 0..rw {
                        for k in 0..rz {
                            for i in 0..rows {
                                for j in 0..cols {
                                    sums.iter_mut().for_each(|x| *x = C64::ZERO);
                                    moduli_sums.iter_mut().for_each(|x| *x = 0.0);
                                    for q in 0..size {
                                        let d = (C64::real(z0[j]) - lambda[q]).recip();
                                        let mut x = lv[kp][i * size + q] * vu[k][q * cols + j];
                                        for (t, a) in sums.iter_mut().zip(moduli_sums.iter_mut()) {
                                            x = x * d;
                                            *t += x;
                                            *a += x.abs();
                                        }
                                    }
                                    *modulus = moduli_sums.iter().fold(*modulus, |a, &b| a.max(b));
                                    for (jj, t) in sums.iter().enumerate() {
                                        out[((((kp * rz + k) * orders + jj) * m) + n) * per + i * cols + j] = *t;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // The gaps' lower bounds over the piece (# The bounds).
            let h = 0.5 * (b - a) * (std::f64::consts::PI / (2.0 * m as f64)).sin();
            let moving = |z: usize| kappa[grouping.zs[z].block] * slope[grouping.zs[z].block] + b_most;
            let gaps: Vec<f64> = gaps.iter().enumerate().map(|(z, g)| g - moving(z) * h).collect();
            // Chebyshev coefficients from the samples at the Gauss points; each table's tail against its pair's largest,
            // and whether every table's is within its samples' rounding.
            let (mut tail, mut fine): (f64, bool) = (0.0, true);
            let coefs: Vec<Vec<C64>> = samples
                .iter()
                .zip(&tables)
                .zip(&moduli)
                .map(|((smp, t), modulus)| {
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
                    let top = coef.iter().map(|z| z.abs()).fold(0.0, f64::max);
                    if top > 0.0 {
                        for c in 0..count {
                            for e in 0..per {
                                let last = coef[(c * m + m - 1) * per + e].abs() + if m > 1 { coef[(c * m + m - 2) * per + e].abs() } else { 0.0 };
                                tail = tail.max(last / top);
                                fine &= last <= floor * modulus;
                            }
                        }
                    }
                    coef
                })
                .collect();
            Some((coefs, Piece { range: [a, b], gaps, kappa, tail }, fine))
        };
        // The pieces: halved until their tail is within the floor, or no longer falls by half (# The pieces).
        let mut todo = vec![(low, 1.0, make(low, 1.0)?)];
        let mut done: Vec<(Vec<Vec<C64>>, Piece)> = Vec::new();
        while let Some((a, b, (coefs, piece, fine))) = todo.pop() {
            if fine || piece.tail <= target {
                done.push((coefs, piece));
                continue;
            }
            let c = 0.5 * (a + b);
            let (left, right) = (make(a, c)?, make(c, b)?);
            if left.1.tail.max(right.1.tail) > 0.5 * piece.tail {
                done.push((coefs, piece));
            } else {
                todo.push((c, b, right));
                todo.push((a, c, left));
            }
        }
        done.sort_by(|x, y| x.1.range[0].total_cmp(&y.1.range[0]));
        for (coefs, _) in &done {
            for (t, c) in tables.iter_mut().zip(coefs) {
                t.coef.extend_from_slice(c);
            }
        }
        let pieces: Vec<Piece> = done.into_iter().map(|x| x.1).collect();
        let bytes = tables.iter().map(|t| t.coef.len() * std::mem::size_of::<C64>()).sum();
        Some(Tables { degree, orders, pieces, dropped, zs, ws, tables, e_rest: e0, e_slope, build_seconds: clock.elapsed().as_secs_f64(), bytes })
    }

    /// The tables' reading at the values `v` (`Reading`; # The bounds): None where s lies outside the pieces, the
    /// piece's gap bound is not positive, or ΔG reaches a gap.
    pub(crate) fn read(&self, plan: &Plan, grouping: &Grouping, v: &Values, scratch: &mut Scratch) -> Option<Reading> {
        let f = plan.intensity[0];
        let (s, ds) = (v.factors[f].0.re, v.factors[f].1.map(|d| d.re));
        let piece = self.pieces.iter().position(|p| s >= p.range[0] && s <= p.range[1])?;
        let p = &self.pieces[piece];
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
        let mut sizes: Vec<Option<(f64, f64)>> = vec![None; plan.blocks.len()];
        for (z, sum) in grouping.zs.iter().enumerate() {
            let b = sum.block;
            if !plan.made(b, true) {
                continue;
            }
            let (value, gradient) = *sizes[b].get_or_insert_with(|| {
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
        Some(Reading { piece, s, ds, rho, moved, moved_gradient, shift_gradient: de_gradient / least_gap })
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
        let Scratch { phi, psi, de, t, dt, px, py, value, slope, grad } = scratch;
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
        for tb in &self.tables {
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
        out.work = work;
        work
    }
}
