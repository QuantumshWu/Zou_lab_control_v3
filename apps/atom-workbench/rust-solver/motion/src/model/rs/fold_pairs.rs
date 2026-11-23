//! OLDNEW's prototype (BRIEF_OLDNEW item C; tests alone, no product path reaches it): the node's contraction (`Out` of
//! `Local::evaluate` + `Out::contract`) off tables of pair products made once, the old engine's structure under strict
//! bounds.
//!
//! # The structure
//! A pair (leg sum w, amplitude sum z) writes W M_z, M_z = Σ_{m∈z} R(E_a + f_m) X_m column by column (rs.rs, # The fold).
//! Each member's frequency is the sum's at rest f₀ plus its shift δ_m (the Doppler shift and the members' recoil apart),
//! and R(z₀ + δ) = Σ_J (−δ)^J R(z₀)^{J+1} -- the tail after J within ρ^{J+1}/(1 − ρ) of the leading term, ρ = |δ|/|z₀ − λ|
//! (STEP's census). The legs and the amplitudes are sums of fixed sparse matrices (an entry's term and coefficient) times
//! point numbers (the factors, the coefficients, the weights): each sum's fixed matrices span a basis (Gram–Schmidt, the
//! Frobenius product; a carrier's beams of one frequency span at most the dipole's three components, the old engine's
//! field basis), and a pair is
//!   W M_z = Σ_{k′,k,J} φ_k′ ψ_k^{(J)} P_{k′kJ}(s),   P_{k′kJ} = L_k′ R(z₀)^{J+1} U_k   (rows × cols, column a at its z₀),
//! φ the leg's coordinates, ψ^{(J)} = Σ_m (−δ_m)^J × the member's coordinates: per pair (r_w r_z (J+1)) small matrices
//! tabled in the tweezer's intensity s (Chebyshev, degree `degree`, on [0, 1]); the gradient Σ (∇φ ψ + φ ∇ψ) P + φ ψ P′ ∇s.
//! The other varying parts of G₀ -- the tweezer's Doppler powers, a Gaussian beam's own intensity -- are held at rest in
//! the tables (their share is what `Pairs::check` reads against the node). The direct entries are the contraction's own.
use super::*;

/// One fixed sparse matrix of a sum (an entry list's term and coefficient of one member) on the sum's layout: its
/// member's piece, its term (into `Plan::terms[piece]`), its coefficient, and its coordinates on the sum's basis.
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
    /// The basis (dense on the sum's layout: an amplitude sum |B| × cols, a leg sum rows × |B|), kept for the build.
    vectors: Vec<Vec<C64>>,
}

/// A pair's tables: H (true) or a jump's, its index in the grouping's pairs, its sums, sizes, the Chebyshev coefficients
/// [piece][k′][k][J][d][i·cols + j].
#[derive(Clone, Debug)]
struct PairTable {
    h: bool,
    at: usize,
    w: usize,
    z: usize,
    rows: usize,
    cols: usize,
    coef: Vec<C64>,
}

/// One output's direct entries spanned (a check's alternative to the entries one by one): H's output (true) or a
/// channel's, its index, its places, its atoms (piece, term, coefficient) with their coordinates and the basis on its
/// places.
#[derive(Clone, Debug)]
struct Direct {
    h: bool,
    out: usize,
    places: Vec<usize>,
    atoms: Vec<Atom>,
    basis: Vec<Vec<C64>>,
    which: usize,
}

/// The tables (module documentation), their structure and what the build took.
#[derive(Clone, Debug, Default)]
pub(crate) struct Pairs {
    pub degree: usize,
    pub orders: usize,
    /// The pieces of s the tables hold (each [a, b] a Chebyshev series of `degree`), split until every table's last two
    /// coefficients are within `S_TOL` of its largest.
    pub pieces: Vec<[f64; 2]>,
    /// The spans' dropped bound, the largest over the sums (relative to each sum's largest atom).
    pub dropped: f64,
    /// A check's switch: H's pairs across two fast parts of K left out (`breakdown`).
    pub split: bool,
    /// A check's switch: the pairs on K's sideband blocks left out.
    pub noside: bool,
    /// The direct entries spanned per output, and the switch that takes them instead of the entries one by one.
    directs: Vec<Direct>,
    pub spanned: bool,
    /// A check's switch: no channel (their pairs and direct entries) made -- what the step reads of them taken off the
    /// quadratic forms' tables instead (`Quad`).
    pub channels_off: bool,
    zs: Vec<Basis>,
    ws: Vec<Basis>,
    pairs: Vec<PairTable>,
    /// Per piece, z sum and column the least |z₀ − λ| over the piece's samples (the Doppler tail's ρ = |δ| over it).
    gaps: Vec<Vec<Vec<f64>>>,
    /// Blocks whose G₀ holds a varying factor other than the intensity and the beams' own intensities: not tabled.
    pub untabled: Vec<usize>,
    pub build_seconds: f64,
    pub bytes: usize,
    /// Per pair (r_w, r_z, rows × cols): the counts the node's cost goes by.
    pub sizes: Vec<(usize, usize, usize)>,
    /// K's energies the tables are made at, E_a(s) = A_a + B_a s (the diagonal's entries at the intensity s, every
    /// other factor at rest): a point's E_a less it, ΔE_a, goes into the series with the members' shifts.
    e_rest: Vec<f64>,
    e_slope: Vec<f64>,
    /// Per amplitude sum: (atoms, rank of all, rank of the first coefficient's atoms alone).
    pub kinds: Vec<(usize, usize, usize)>,
    /// The per-node scratch: φ and its gradient per leg sum; ψ per amplitude sum, order, basis vector and column.
    phi: Vec<Vec<(C64, [C64; 3])>>,
    psi: Vec<Vec<Vec<Vec<(C64, [C64; 3])>>>>,
    de: Vec<(f64, [f64; 3])>,
    pub work: f64,
    /// The last contractions' time by part (µs, summed until read): the coordinates, the pairs, the direct entries.
    pub parts: [f64; 3],
}

/// T_d(x) and dT_d/ds (x = 2s − 1) for d = 0..=D.
fn chebyshev(s: f64, degree: usize, t: &mut Vec<f64>, dt: &mut Vec<f64>) {
    let x = 2.0 * s - 1.0;
    t.clear();
    dt.clear();
    let (mut t0, mut t1) = (1.0, x);
    let (mut u0, mut u1) = (1.0, 2.0 * x);
    t.push(1.0);
    dt.push(0.0);
    if degree >= 1 {
        t.push(x);
        dt.push(2.0);
    }
    for d in 2..=degree {
        let t2 = 2.0 * x * t1 - t0;
        (t0, t1) = (t1, t2);
        t.push(t2);
        // dT_d/dx = d U_{d−1}.
        dt.push(2.0 * d as f64 * u1);
        let u2 = 2.0 * x * u1 - u0;
        (u0, u1) = (u1, u2);
    }
}

thread_local! {
    /// The accepted vectors' residuals over their norms in the last `span`s (a diagnostic).
    pub(crate) static RESIDUALS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Gram–Schmidt of a list of vectors, each with the bound b of its point coefficient over the reach: in the order of
/// ‖v‖b, a vector's residual r kept as a new direction where ‖r‖b exceeds `tol` times the largest ‖v‖b, else left out --
/// its share ‖r‖b over the largest, summed, is the span's dropped bound (returned). The basis and each vector's
/// coordinates on it.
fn span(vectors: &[Vec<C64>], bounds: &[f64], limit: f64, top: f64) -> (Vec<Vec<C64>>, Vec<Vec<C64>>, f64) {
    let dot = |a: &[C64], b: &[C64]| a.iter().zip(b).fold(C64::ZERO, |s, (x, y)| s + x.conj() * *y);
    let sizes: Vec<f64> = vectors.iter().zip(bounds).map(|(v, b)| dot(v, v).re.sqrt() * b).collect();
    let mut order: Vec<usize> = (0..vectors.len()).collect();
    order.sort_by(|&a, &b| sizes[b].total_cmp(&sizes[a]));
    let mut basis: Vec<Vec<C64>> = Vec::new();
    let mut coords: Vec<Vec<C64>> = vec![Vec::new(); vectors.len()];
    let mut dropped = 0.0;
    for &t in &order {
        let mut r = vectors[t].clone();
        let mut cs = vec![C64::ZERO; basis.len()];
        for _ in 0..2 {
            for (k, u) in basis.iter().enumerate() {
                let p = dot(u, &r);
                cs[k] += p;
                for (x, y) in r.iter_mut().zip(u) {
                    *x -= p * *y;
                }
            }
        }
        let rest = dot(&r, &r).re.sqrt();
        if rest * bounds[t] > limit && top > 0.0 {
            RESIDUALS.with(|x| x.borrow_mut().push(rest * bounds[t] / top));
            let inv = 1.0 / rest;
            basis.push(r.iter().map(|x| *x * inv).collect());
            cs.push(C64::real(rest));
        } else if top > 0.0 {
            dropped += rest * bounds[t] / top;
        }
        coords[t] = cs;
    }
    let rank = basis.len();
    for c in coords.iter_mut() {
        c.resize(rank, C64::ZERO);
    }
    (basis, coords, dropped)
}

impl Pairs {
    /// G₀ on block b and K's energies at the values `v` (as `Local::evaluate` assembles them).
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

    fn energies(plan: &Plan, v: &Values) -> Vec<f64> {
        let mut e = vec![0.0; plan.n];
        for d in &plan.diagonal {
            let (f, scale) = plan.generator_terms[d.term as usize];
            e[d.row as usize] += (scale * v.factors[f].0 * (d.value * v.coefficients[d.coefficient as usize])).re;
        }
        e
    }

    /// The values the tables are made at: the structure point's (`rest`, at rest), the tweezer's intensity s, every beam's
    /// own intensity one, the coefficients at rest.
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

    /// The tables of a plan's grouping (module documentation): `rest` the values at a point at rest (the members'
    /// frequencies f₀), `reach` at a point at the reach's speed (the atoms' point coefficients' bounds), `orders` the
    /// Doppler series' terms kept (J + 1), `degree` the pieces' Chebyshev degree in s on [`low`, 1].
    pub(crate) fn new(plan: &Plan, grouping: &Grouping, rest: &Values, reach: &Values, orders: usize, degree: usize, low: f64) -> Pairs {
        const S_TOL: f64 = 1e-14;
        let clock = std::time::Instant::now();
        // The blocks whose G₀ varies by another factor than the intensities.
        let untabled: Vec<usize> = (0..plan.blocks.len())
            .filter(|&b| {
                plan.blocks[b].rest.iter().any(|e| {
                    let f = plan.generator_terms[e.term as usize].0;
                    plan.varying[f] && !plan.intensity.contains(&f) && !plan.own.contains(&f)
                })
            })
            .collect();
        // An atom's point coefficient's bound: its factor's and coefficient's moduli, the larger of rest's and reach's.
        let bound = |piece: usize, term: usize, coefficient: usize| -> f64 {
            let (f, scale) = plan.terms[piece][term];
            let x = rest.factors[f].0.abs().max(reach.factors[f].0.abs());
            let c = rest.coefficients[coefficient].abs().max(reach.coefficients[coefficient].abs());
            scale.abs() * x * c
        };
        // The sums' atoms: per sum its keys (piece, term, coefficient) and fixed matrices on its layout.
        let mut z_atoms: Vec<(Vec<(usize, usize, usize)>, Vec<Vec<C64>>)> = Vec::new();
        for s in grouping.zs.iter() {
            let size = plan.blocks[s.block].rows.len();
            let cols = s.index.len();
            let mut keys: Vec<(usize, usize, usize)> = Vec::new();
            let mut mats: Vec<Vec<C64>> = Vec::new();
            for (member, map) in &s.members {
                let r = &plan.amplitudes[*member];
                for e in &r.entries {
                    let Some(&(_, to)) = map.iter().find(|x| x.0 == e.col as usize) else { continue };
                    let key = (r.piece, e.term as usize, e.coefficient as usize);
                    let at = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                        keys.push(key);
                        mats.push(vec![C64::ZERO; size * cols]);
                        keys.len() - 1
                    });
                    mats[at][e.row as usize * cols + to] += e.value;
                }
            }
            z_atoms.push((keys, mats));
        }
        let mut w_atoms: Vec<(Vec<(usize, usize, usize)>, Vec<Vec<C64>>)> = Vec::new();
        for s in grouping.ws.iter() {
            let size = plan.blocks[s.block].rows.len();
            let rows = s.index.len();
            let mut keys: Vec<(usize, usize, usize)> = Vec::new();
            let mut mats: Vec<Vec<C64>> = Vec::new();
            for (member, map) in &grouping.wmaps[s.at] {
                let r = &plan.legs[*member];
                for e in &r.entries {
                    let to = map[e.row as usize];
                    if to == u32::MAX {
                        continue;
                    }
                    let key = (r.piece, e.term as usize, e.coefficient as usize);
                    let at = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                        keys.push(key);
                        mats.push(vec![C64::ZERO; rows * size]);
                        keys.len() - 1
                    });
                    mats[at][to as usize * size + e.col as usize] += e.value;
                }
            }
            w_atoms.push((keys, mats));
        }
        // The spans' scale (one for every sum, so that what is left out is small against the whole model, not against
        // its own sum): an amplitude sum's atoms ‖A‖b over the least |z₀ − λ| of its block at s = 1 (its resolvent's
        // scale), a leg sum's ‖L‖b; each kind's largest over the sums.
        let at_one = Pairs::at_intensity(plan, rest, 1.0);
        let e_one = Pairs::energies(plan, &at_one);
        let gap_of: Vec<f64> = grouping
            .zs
            .iter()
            .map(|s| {
                if untabled.contains(&s.block) {
                    return f64::INFINITY;
                }
                let mut sp = super::super::spectrum::Spectrum::default();
                assert!(sp.decompose(&Pairs::generator(plan, s.block, &at_one)));
                let f0 = s.members.first().map_or(0.0, |(m, _)| rest.frequencies[plan.amplitudes[*m].piece]);
                s.index.iter().flat_map(|&col| (0..sp.centre.len()).map(move |k| (col, k))).map(|(col, k)| (C64::real(e_one[col] + f0 - sp.centre[k]) - sp.values[k]).abs()).fold(f64::INFINITY, f64::min)
            })
            .collect();
        let norm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let top_of = |(keys, mats): &(Vec<(usize, usize, usize)>, Vec<Vec<C64>>)| keys.iter().zip(mats).map(|(&(p, t, c), m)| norm(m) * bound(p, t, c)).fold(0.0, f64::max);
        let z_top = z_atoms.iter().zip(&gap_of).map(|(a, g)| top_of(a) / g).fold(0.0, f64::max);
        let w_top = w_atoms.iter().map(top_of).fold(0.0, f64::max);
        let tol: f64 = std::env::var("OLDNEW_SPAN_TOL").ok().map_or(1e-12, |x| x.parse().unwrap());
        let mut dropped: f64 = 0.0;
        let mut basis_of = |(keys, mats): (Vec<(usize, usize, usize)>, Vec<Vec<C64>>), f0: f64, limit: f64, top: f64, label: &str| -> Basis {
            let bounds: Vec<f64> = keys.iter().map(|&(p, t, c)| bound(p, t, c)).collect();
            RESIDUALS.with(|r| r.borrow_mut().clear());
            let (vectors, coords, d) = span(&mats, &bounds, limit, top);
            dropped = dropped.max(d);
            if std::env::var("OLDNEW_ATOMS").is_ok() {
                eprintln!("OLDNEW-SPAN {label}: {} atoms, rank {}, dropped {d:.1e}, kept residuals {:?}", keys.len(), vectors.len(), RESIDUALS.with(|r| r.borrow().iter().map(|x| format!("{x:.1e}")).collect::<Vec<_>>()));
            }
            Basis { atoms: keys.iter().zip(coords).map(|(&(piece, term, coefficient), coords)| Atom { piece, term, coefficient, coords }).collect(), rank: vectors.len(), f0, vectors }
        };
        let mut zs: Vec<Basis> = Vec::new();
        for (zi, (s, atoms)) in grouping.zs.iter().zip(z_atoms).enumerate() {
            let f0 = s.members.first().map_or(0.0, |(m, _)| rest.frequencies[plan.amplitudes[*m].piece]);
            let g = gap_of[zi];
            zs.push(basis_of(atoms, f0, tol * z_top * g, z_top * g, &format!("z {zi} block {} cols {}", s.block, s.index.len())));
        }
        let mut ws: Vec<Basis> = Vec::new();
        for (wi, (s, atoms)) in grouping.ws.iter().zip(w_atoms).enumerate() {
            ws.push(basis_of(atoms, 0.0, tol * w_top, w_top, &format!("w {wi} block {} rows {}", s.block, s.index.len())));
        }
        // The pairs to table.
        let n_h = grouping.h_pairs.len();
        let list: Vec<(bool, usize, usize, usize)> = grouping
            .h_pairs
            .iter()
            .enumerate()
            .map(|(i, x)| (true, i, x.w, x.z))
            .chain(grouping.j_pairs.iter().enumerate().map(|(i, x)| (false, i, x.w, x.z)))
            .filter(|&(_, _, _, z)| !untabled.contains(&grouping.zs[z].block))
            .collect();
        let _ = n_h;
        // One piece [a, b]: per pair its coefficients [k′][k][J][d][e], and per z sum per column its least gap; whether
        // every table's last two coefficients are within S_TOL of its largest.
        let m = degree + 1;
        let piece = |a: f64, b: f64| -> (Vec<Vec<C64>>, Vec<Vec<f64>>, bool) {
            let xs: Vec<f64> = (0..m).map(|n| (std::f64::consts::PI * (n as f64 + 0.5) / m as f64).cos()).collect();
            let mut gaps: Vec<Vec<f64>> = grouping.zs.iter().map(|s| vec![f64::INFINITY; s.index.len()]).collect();
            let mut samples: Vec<Vec<C64>> = list
                .iter()
                .map(|&(_, _, w, z)| vec![C64::ZERO; ws[w].rank * zs[z].rank * orders * m * grouping.ws[w].index.len() * grouping.zs[z].index.len()])
                .collect();
            for (n, &x) in xs.iter().enumerate() {
                let sv = 0.5 * (a + b) + 0.5 * (b - a) * x;
                let v = Pairs::at_intensity(plan, rest, sv);
                let energies = Pairs::energies(plan, &v);
                let spectra: Vec<Option<super::super::spectrum::Spectrum>> = (0..plan.blocks.len())
                    .map(|bl| {
                        if untabled.contains(&bl) {
                            return None;
                        }
                        let mut sp = super::super::spectrum::Spectrum::default();
                        assert!(sp.decompose(&Pairs::generator(plan, bl, &v)));
                        Some(sp)
                    })
                    .collect();
                for (z, sum) in grouping.zs.iter().enumerate() {
                    let Some(sp) = spectra[sum.block].as_ref() else { continue };
                    for (j, &col) in sum.index.iter().enumerate() {
                        let z0 = energies[col] + zs[z].f0;
                        for k in 0..sp.centre.len() {
                            gaps[z][j] = gaps[z][j].min((C64::real(z0 - sp.centre[k]) - sp.values[k]).abs());
                        }
                    }
                }
                for (out, &(_, _, w, z)) in samples.iter_mut().zip(&list) {
                    let block = grouping.zs[z].block;
                    let sp = spectra[block].as_ref().unwrap();
                    let size = plan.blocks[block].rows.len();
                    let (rows, cols) = (grouping.ws[w].index.len(), grouping.zs[z].index.len());
                    let (bw, bz) = (&ws[w], &zs[z]);
                    let (rw, rz, per) = (bw.rank, bz.rank, rows * cols);
                    let z0: Vec<f64> = grouping.zs[z].index.iter().map(|&col| energies[col] + bz.f0).collect();
                    let lambda: Vec<C64> = (0..size).map(|k| C64::real(sp.centre[k]) + sp.values[k]).collect();
                    let lv: Vec<Vec<C64>> = bw
                        .vectors
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
                        .collect();
                    let vu: Vec<Vec<C64>> = bz
                        .vectors
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
                    for kp in 0..rw {
                        for k in 0..rz {
                            for i in 0..rows {
                                for j in 0..cols {
                                    let mut sums = vec![C64::ZERO; orders];
                                    for q in 0..size {
                                        let d = (C64::real(z0[j]) - lambda[q]).recip();
                                        let mut x = lv[kp][i * size + q] * vu[k][q * cols + j];
                                        for t in sums.iter_mut() {
                                            x = x * d;
                                            *t += x;
                                        }
                                    }
                                    for (jj, t) in sums.into_iter().enumerate() {
                                        out[((((kp * rz + k) * orders + jj) * m) + n) * per + i * cols + j] = t;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // Chebyshev coefficients from the samples at the Gauss points; each table's tail against its largest.
            let mut fine = true;
            let s_tol: f64 = std::env::var("OLDNEW_S_TOL").ok().map_or(S_TOL, |x| x.parse().unwrap());
            let coefs: Vec<Vec<C64>> = samples
                .iter()
                .zip(&list)
                .map(|(smp, &(_, _, w, z))| {
                    let per = grouping.ws[w].index.len() * grouping.zs[z].index.len();
                    let tables = ws[w].rank * zs[z].rank * orders;
                    let mut coef = vec![C64::ZERO; smp.len()];
                    let mut tails = Vec::with_capacity(tables);
                    for t in 0..tables {
                        for d in 0..m {
                            let scale = if d == 0 { 1.0 / m as f64 } else { 2.0 / m as f64 };
                            for (n, x) in xs.iter().enumerate() {
                                let c = (d as f64 * x.acos()).cos() * scale;
                                for e in 0..per {
                                    coef[(t * m + d) * per + e] += smp[(t * m + n) * per + e] * c;
                                }
                            }
                        }
                        let tail = (0..per).map(|e| coef[(t * m + m - 1) * per + e].abs() + if m > 1 { coef[(t * m + m - 2) * per + e].abs() } else { 0.0 }).fold(0.0, f64::max);
                        tails.push(tail);
                    }
                    // Each table's tail against the pair's largest coefficient (a direction of a sum whose share is tiny
                    // is held to the pair's scale, not to its own).
                    let top = coef.iter().map(|z| z.abs()).fold(0.0, f64::max);
                    if tails.iter().any(|&x| x > s_tol * top) {
                        fine = false;
                    }
                    coef
                })
                .collect();
            (coefs, gaps, fine)
        };
        // The pieces: halved until fine (or a width of 2⁻¹², taken as it is).
        let mut todo = vec![[low, 1.0]];
        let mut done: Vec<([f64; 2], Vec<Vec<C64>>, Vec<Vec<f64>>)> = Vec::new();
        while let Some([a, b]) = todo.pop() {
            let (coefs, gaps, fine) = piece(a, b);
            if fine || b - a < 1.0 / 64.0 {
                done.push(([a, b], coefs, gaps));
            } else {
                let c = 0.5 * (a + b);
                todo.push([c, b]);
                todo.push([a, c]);
            }
        }
        done.sort_by(|x, y| x.0[0].total_cmp(&y.0[0]));
        let pieces: Vec<[f64; 2]> = done.iter().map(|x| x.0).collect();
        let gaps: Vec<Vec<Vec<f64>>> = done.iter().map(|x| x.2.clone()).collect();
        let mut pairs: Vec<PairTable> = list
            .iter()
            .map(|&(h, at, w, z)| PairTable { h, at, w, z, rows: grouping.ws[w].index.len(), cols: grouping.zs[z].index.len(), coef: Vec::new() })
            .collect();
        for (_, coefs, _) in &done {
            for (p, c) in pairs.iter_mut().zip(coefs) {
                p.coef.extend_from_slice(c);
            }
        }
        let sizes = pairs.iter().map(|p| (ws[p.w].rank, zs[p.z].rank, p.rows * p.cols)).collect();
        let bytes = pairs.iter().map(|p| p.coef.len() * 16).sum();
        let phi = ws.iter().map(|b| vec![(C64::ZERO, [C64::ZERO; 3]); b.rank]).collect();
        let psi = zs.iter().zip(&grouping.zs).map(|(b, s)| vec![vec![vec![(C64::ZERO, [C64::ZERO; 3]); s.index.len()]; b.rank]; orders]).collect();
        let e0 = Pairs::energies(plan, &Pairs::at_intensity(plan, rest, 0.0));
        let e1 = Pairs::energies(plan, &Pairs::at_intensity(plan, rest, 1.0));
        let e_slope: Vec<f64> = e0.iter().zip(&e1).map(|(a, b)| b - a).collect();
        let kinds = zs.iter().map(|b| (b.atoms.len(), b.rank, b.atoms.iter().filter(|a| a.coefficient == 0).count())).collect();
        // The direct entries per output, spanned as the sums are (their scale the largest over every output's atoms).
        let mut groups: Vec<(bool, usize, usize, Vec<usize>, Vec<(usize, usize, usize)>, Vec<Vec<(usize, C64)>>)> = Vec::new();
        for p in 0..plan.keys.len() {
            let which = plan.jump[p].map_or(0, |x| x.1);
            for (i, (e, _)) in plan.direct[p].iter().enumerate() {
                let (h, o, k) = if p < plan.hamiltonian {
                    (true, grouping.h_direct[p], grouping.h_direct_places[p][i] as usize)
                } else {
                    let (o, k) = grouping.j_direct[p - plan.hamiltonian][i];
                    (false, o, k as usize)
                };
                let g = match groups.iter().position(|x| x.0 == h && x.1 == o && x.2 == which) {
                    Some(g) => g,
                    None => {
                        groups.push((h, o, which, Vec::new(), Vec::new(), Vec::new()));
                        groups.len() - 1
                    }
                };
                let gr = &mut groups[g];
                let place = gr.3.iter().position(|&x| x == k).unwrap_or_else(|| {
                    gr.3.push(k);
                    gr.3.len() - 1
                });
                let key = (p, e.term as usize, e.coefficient as usize);
                let a = gr.4.iter().position(|x| *x == key).unwrap_or_else(|| {
                    gr.4.push(key);
                    gr.5.push(Vec::new());
                    gr.4.len() - 1
                });
                gr.5[a].push((place, e.value));
            }
        }
        let dense = |entries: &[(usize, C64)], n: usize| {
            let mut v = vec![C64::ZERO; n];
            for &(i, x) in entries {
                v[i] += x;
            }
            v
        };
        // Each kind's own scale (H's outputs and the channels' are apart by orders of magnitude).
        let d_top_of = |h: bool| {
            groups
                .iter()
                .filter(|g| g.0 == h)
                .flat_map(|g| g.4.iter().zip(&g.5).map(move |(&(p, t, c), e)| norm(&dense(e, g.3.len())) * bound(p, t, c)))
                .fold(0.0, f64::max)
        };
        let d_tops = [d_top_of(false), d_top_of(true)];
        let d_tol: f64 = std::env::var("OLDNEW_DIRECT_TOL").ok().map_or(1e-12, |x| x.parse().unwrap());
        let directs: Vec<Direct> = groups
            .into_iter()
            .map(|(h, out, which, places, keys, entries)| {
                let mats: Vec<Vec<C64>> = entries.iter().map(|e| dense(e, places.len())).collect();
                let bounds: Vec<f64> = keys.iter().map(|&(p, t, c)| bound(p, t, c)).collect();
                let d_top = d_tops[h as usize];
                let (basis, coords, d) = span(&mats, &bounds, d_tol * d_top, d_top);
                dropped = dropped.max(d);
                Direct { h, out, places, atoms: keys.iter().zip(coords).map(|(&(piece, term, coefficient), coords)| Atom { piece, term, coefficient, coords }).collect(), basis, which }
            })
            .collect();
        Pairs { parts: [0.0; 3], degree, orders, pieces, dropped, split: false, noside: false, directs, spanned: false, channels_off: false, zs, ws, pairs, gaps, untabled, build_seconds: clock.elapsed().as_secs_f64(), bytes, sizes, e_rest: e0, e_slope, kinds, phi, psi, de: Vec::new(), work: 0.0 }
    }

    /// Whether a point's intensity lies within the tables (else its node is the fold's own, `Local::evaluate`).
    pub(crate) fn covers(&self, plan: &Plan, v: &Values) -> bool {
        let s = plan.intensity.first().map_or(1.0, |&f| v.factors[f].0.re);
        s >= self.pieces[0][0] && s <= self.pieces[self.pieces.len() - 1][1]
    }

    /// The piece of s a point falls in.
    fn piece_of(&self, s: f64) -> usize {
        self.pieces.iter().position(|p| s <= p[1]).unwrap_or(self.pieces.len() - 1)
    }

    /// The largest Doppler ratio ρ = |δ_m|/|z₀ − λ| over the sums' members and columns at the values `v` (against the
    /// least gap over the build's samples), and the tail's relative bound ρ^{orders}/(1 − ρ).
    pub(crate) fn rho(&mut self, plan: &Plan, grouping: &Grouping, v: &Values) -> (f64, f64) {
        self.shifts(plan, v);
        let de = self.de.iter().map(|x| x.0.abs()).fold(0.0, f64::max);
        let mut rho: f64 = 0.0;
        let sv = plan.intensity.first().map_or(0.0, |&f| v.factors[f].0.re);
        let at = self.piece_of(sv);
        for (z, (b, s)) in self.zs.iter().zip(&grouping.zs).enumerate() {
            let gap = self.gaps[at][z].iter().copied().fold(f64::INFINITY, f64::min);
            if !gap.is_finite() {
                continue;
            }
            for (member, _) in &s.members {
                let delta = v.frequencies[plan.amplitudes[*member].piece] - b.f0;
                rho = rho.max((delta.abs() + de) / gap);
            }
        }
        (rho, rho.powi(self.orders as i32) / (1.0 - rho))
    }

    /// ΔE_a and its gradient at the values `v`: K's energies there (the diagonal's entries, as `Local::evaluate`'s)
    /// less the tables' E_a(s).
    fn shifts(&mut self, plan: &Plan, v: &Values) {
        let s = plan.intensity.first().map_or(0.0, |&f| v.factors[f].0.re);
        let ds: [f64; 3] = plan.intensity.first().map_or([0.0; 3], |&f| v.factors[f].1.map(|d| d.re));
        self.de.clear();
        self.de.extend((0..plan.n).map(|a| (-(self.e_rest[a] + self.e_slope[a] * s), ds.map(|d| -self.e_slope[a] * d))));
        for d in &plan.diagonal {
            let (f, scale) = plan.generator_terms[d.term as usize];
            let (x, g) = v.factors[f];
            let c = d.value * v.coefficients[d.coefficient as usize];
            let e = &mut self.de[d.row as usize];
            e.0 += (scale * x * c).re;
            for a in 0..3 {
                e.1[a] += (scale * g[a] * c).re;
            }
        }
    }

    /// The contraction at the values `v` into `out` (as `Out::contract` lays it, `making` a node's); `orders` the Doppler
    /// terms taken at this point (at most the tables').
    pub(crate) fn contract(&mut self, plan: &Plan, grouping: &Grouping, v: &Values, out: &mut Out, making: Making, orders: usize) {
        let clock = std::time::Instant::now();
        let orders = orders.min(self.orders);
        let mut work = 0.0;
        out.h.resize_with(grouping.outputs.len(), Default::default);
        out.channels.resize_with(grouping.channel_outputs.len(), Default::default);
        for (c, list) in out.h.iter_mut().zip(&grouping.lists) {
            c.zeroed(list.len());
        }
        for (c, list) in out.channels.iter_mut().zip(&grouping.channel_lists) {
            c.zeroed(list.len());
        }
        // The point's coordinates: per atom its factor × scale × coefficient (and gradient), per amplitude sum with the
        // member's (−δ)^J.
        let point = |piece: usize, term: usize, coefficient: usize| -> (C64, [C64; 3]) {
            let (f, scale) = plan.terms[piece][term];
            let (x, g) = v.factors[f];
            let c = v.coefficients[coefficient];
            (scale * x * c, g.map(|d| scale * d * c))
        };
        for (b, phi) in self.ws.iter().zip(self.phi.iter_mut()) {
            for p in phi.iter_mut() {
                *p = (C64::ZERO, [C64::ZERO; 3]);
            }
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
                    work += 4.0;
                }
            }
        }
        self.shifts(plan, v);
        let de = &self.de;
        for ((b, psi), sum) in self.zs.iter().zip(self.psi.iter_mut()).zip(&grouping.zs) {
            for o in psi.iter_mut() {
                for p in o.iter_mut() {
                    for x in p.iter_mut() {
                        *x = (C64::ZERO, [C64::ZERO; 3]);
                    }
                }
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
                    for (jj, o) in psi.iter_mut().enumerate().take(orders) {
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
                        work += 4.0 * a.coords.len() as f64;
                        previous = power;
                        power *= minus;
                    }
                }
            }
        }
        let s = plan.intensity.first().map_or(0.0, |&f| v.factors[f].0.re);
        let ds: [f64; 3] = plan.intensity.first().map_or([0.0; 3], |&f| v.factors[f].1.map(|d| d.re));
        self.parts[0] += clock.elapsed().as_secs_f64() * 1e6;
        let clock = std::time::Instant::now();
        let (mut t, mut dt) = (Vec::new(), Vec::new());
        let at = self.piece_of(s);
        let [pa, pb] = self.pieces[at];
        chebyshev(0.5 * ((2.0 * s - pa - pb) / (pb - pa) + 1.0), self.degree, &mut t, &mut dt);
        for d in dt.iter_mut() {
            *d /= pb - pa;
        }
        let m = self.degree + 1;
        let mut value = Vec::new();
        let mut slope = Vec::new();
        let mut grad: [Vec<C64>; 3] = Default::default();
        let (mut px, mut py): (Vec<C64>, Vec<C64>) = (Vec::new(), Vec::new());
        for pt in &self.pairs {
            let jump = grouping.ws[pt.w].jump;
            if !pt.h && (!making.model || (jump.is_some_and(|(_, which)| which == 1) && !making.kicks)) {
                continue;
            }
            if pt.h && !(making.model && making.effective) {
                continue;
            }
            if !pt.h && self.channels_off {
                continue;
            }
            // (A check's: the pairs on K's sideband blocks left out, their share read against the node.)
            if self.noside && plan.blocks[grouping.zs[pt.z].block].sideband.is_some() {
                continue;
            }
            // (A check's: K split into its fast parts, H's pairs across two of them left out.)
            if pt.h && self.split && grouping.ws[pt.w].part != grouping.zs[pt.z].part {
                continue;
            }
            let (rw, rz, per) = (self.ws[pt.w].rank, self.zs[pt.z].rank, pt.rows * pt.cols);
            value.clear();
            value.resize(per, C64::ZERO);
            slope.clear();
            slope.resize(per, C64::ZERO);
            for g in grad.iter_mut() {
                g.clear();
                g.resize(per, C64::ZERO);
            }
            let phi = &self.phi[pt.w];
            let psi = &self.psi[pt.z];
            for kp in 0..rw {
                for k in 0..rz {
                    for (jj, o) in psi.iter().enumerate().take(orders) {
                        let a = phi[kp];
                        let base = (((at * rw + kp) * rz + k) * self.orders + jj) * m * per;
                        let tab = &pt.coef[base..base + m * per];
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
                            for i in 0..pt.rows {
                                let e = i * pt.cols + j;
                                let x = px[e];
                                value[e] += c * x;
                                slope[e] += c * py[e];
                                for a in 0..3 {
                                    grad[a][e] += dc[a] * x;
                                }
                            }
                        }
                        work += per as f64 * (m as f64 + 5.0);
                    }
                }
            }
            for e in 0..per {
                for a in 0..3 {
                    grad[a][e] += slope[e] * ds[a];
                }
            }
            if pt.h {
                let pair = &grouping.h_pairs[pt.at];
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
            } else {
                let pair = &grouping.j_pairs[pt.at];
                let o = &mut out.channels[pair.out];
                for e in 0..per {
                    let k = pair.places[e] as usize;
                    o.value[k] += value[e];
                    for a in 0..3 {
                        o.gradient[a][k] += grad[a][e];
                    }
                }
            }
        }
        // The direct entries (`Out::contract`'s own).
        self.parts[1] += clock.elapsed().as_secs_f64() * 1e6;
        let clock = std::time::Instant::now();
        let factor = |p: usize, term: usize| -> (C64, [C64; 3], bool) {
            let (f, s) = plan.terms[p][term];
            let (x, g) = v.factors[f];
            (x * s, g.map(|d| d * s), g.iter().any(|d| *d != C64::ZERO))
        };
        if self.spanned {
            for d in &self.directs {
                if (d.h && !(making.model && making.effective)) || (!d.h && (self.channels_off || !making.model || (d.which == 1 && !making.kicks))) {
                    continue;
                }
                let mut c = vec![(C64::ZERO, [C64::ZERO; 3]); d.basis.len()];
                for a in &d.atoms {
                    let k = v.coefficients[a.coefficient];
                    if k == 0.0 {
                        continue;
                    }
                    let (x, dx, _) = factor(a.piece, a.term);
                    for (ck, z) in c.iter_mut().zip(&a.coords) {
                        ck.0 += *z * x * k;
                        for q in 0..3 {
                            ck.1[q] += *z * dx[q] * k;
                        }
                    }
                }
                let o = if d.h { &mut out.h[d.out] } else { &mut out.channels[d.out] };
                for (ck, b) in c.iter().zip(&d.basis) {
                    for (&place, z) in d.places.iter().zip(b) {
                        o.value[place] += ck.0 * *z;
                        for q in 0..3 {
                            o.gradient[q][place] += ck.1[q] * *z;
                        }
                    }
                }
            }
        } else if making.model && making.effective {
            for p in 0..plan.hamiltonian {
                let o = &mut out.h[grouping.h_direct[p]];
                for ((e, _), &k) in plan.direct[p].iter().zip(&grouping.h_direct_places[p]) {
                    let (x, dx, moving) = factor(p, e.term as usize);
                    let c = e.value * v.coefficients[e.coefficient as usize];
                    let k = k as usize;
                    o.value[k] += x * c;
                    if moving {
                        for (g, d) in o.gradient.iter_mut().zip(dx) {
                            g[k] += d * c;
                        }
                    }
                }
            }
        }
        if making.model && !self.spanned {
            for p in plan.hamiltonian..plan.keys.len() {
                let (_, which) = plan.jump[p].unwrap();
                if which == 1 && !making.kicks {
                    continue;
                }
                for ((e, _), &(o, k)) in plan.direct[p].iter().zip(&grouping.j_direct[p - plan.hamiltonian]) {
                    let (x, dx, moving) = factor(p, e.term as usize);
                    let c = e.value * v.coefficients[e.coefficient as usize];
                    let (o, k) = (&mut out.channels[o], k as usize);
                    o.value[k] += x * c;
                    if moving {
                        for (g, d) in o.gradient.iter_mut().zip(dx) {
                            g[k] += d * c;
                        }
                    }
                }
            }
        }
        self.parts[2] += clock.elapsed().as_secs_f64() * 1e6;
        self.work = work;
        out.work = work;
    }

    /// The counts: per pair (r_w, r_z, elements), the tables' (r_w r_z (J + 1)) small matrices, their elements, and
    /// the atoms of every sum.
    pub(crate) fn census(&self) -> (usize, usize, usize, usize) {
        let tables: usize = self.sizes.iter().map(|&(a, b, _)| a * b * self.orders).sum();
        let elements: usize = self.sizes.iter().map(|&(a, b, e)| a * b * self.orders * e).sum();
        let atoms: usize = self.zs.iter().chain(&self.ws).map(|b| b.atoms.len()).sum();
        let ranks: usize = self.zs.iter().chain(&self.ws).map(|b| b.rank).sum();
        let _ = ranks;
        (self.sizes.len(), tables, elements, atoms)
    }

    /// The table elements per order by kind (diagnostic): [H on Q, H on K's sidebands, channels on Q, channels on the
    /// sidebands, photon forms (kicks) on Q and on the sidebands], each (pairs, elements), and H's elements across two
    /// fast parts of K (F = 1 and F = 2 apart: rows of one part, columns of another).
    pub(crate) fn breakdown(&self, plan: &Plan, grouping: &Grouping) -> ([(usize, usize); 6], (usize, usize)) {
        let mut out = [(0usize, 0usize); 6];
        let mut cross = (0, 0);
        for (pt, &(rw, rz, per)) in self.pairs.iter().zip(&self.sizes) {
            let side = plan.blocks[grouping.zs[pt.z].block].sideband.is_some() as usize;
            let kind = if pt.h { 0 } else if grouping.ws[pt.w].jump.is_some_and(|(_, w)| w == 1) { 4 } else { 2 };
            out[kind + side].0 += 1;
            out[kind + side].1 += rw * rz * per;
            if pt.h && grouping.ws[pt.w].part != grouping.zs[pt.z].part {
                cross.0 += 1;
                cross.1 += rw * rz * per;
            }
        }
        (out, cross)
    }

    /// The direct entries: their count one by one, and spanned (atoms, basis vectors × places).
    pub(crate) fn direct_census(&self, plan: &Plan) -> (usize, usize, usize) {
        let entries = plan.direct.iter().map(Vec::len).sum();
        let atoms = self.directs.iter().map(|d| d.atoms.len()).sum();
        let spanned = self.directs.iter().map(|d| d.basis.len() * d.places.len()).sum();
        (entries, atoms, spanned)
    }

    /// An amplitude sum's atoms (diagnostic): (piece, factor, coefficient, |coordinates|), and its block and columns.
    pub(crate) fn atoms_of(&self, plan: &Plan, grouping: &Grouping, z: usize) -> (usize, Vec<usize>, Vec<(usize, usize, usize, f64)>) {
        let s = &grouping.zs[z];
        (s.block, s.index.clone(), self.zs[z].atoms.iter().map(|a| (a.piece, plan.terms[a.piece][a.term].0, a.coefficient, a.coords.iter().map(|c| c.norm_sqr()).sum::<f64>().sqrt())).collect())
    }

    /// The sums' ranks against their atoms (amplitude sums, leg sums).
    pub(crate) fn ranks(&self) -> (Vec<(usize, usize)>, Vec<(usize, usize)>) {
        (self.zs.iter().map(|b| (b.atoms.len(), b.rank)).collect(), self.ws.iter().map(|b| (b.atoms.len(), b.rank)).collect())
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// OLDNEW follow-up (the coordinator's 11-23 question): what a step reads of the channels -- the decay ΣL†L, the hazard
// ⟨ψ|ΣL†L|ψ⟩ and the scattering moments Σ Im⟨u|∇u⟩, Σ Re⟨∇u|∇u⟩ -- are quadratic forms of the channels, so they can be
// tabled as pairs of the channels' own tables, the single channels then made at a jump alone. Per photon class (a
// channel output's fast index, its `which` 0) and emission axis, its contributions: each channel pair's tables (k′, k, J)
// and each spanned direct basis vector; per ordered pair (p, q) of one class and axis with rows in common, four tables
// Q = T_p†T_q, Q^s = T_p†T_q′, Q^t = T_p′†T_q, Q^ss = T_p′†T_q′ over those rows, on cols_p × cols_q, per piece of s at
// twice the linear tables' degree (the products of two of their pieces, exactly). At a point, per class
//   D = Σ c_p* c_q Q,   A_i = Σ c_p* (∂_i c_q Q + c_q ∂_i s Q^s),
//   B_ij = Σ ∂_i c_p* ∂_j c_q Q + ∂_i c_p* c_q ∂_j s Q^s + c_p* ∂_j c_q ∂_i s Q^t + c_p* c_q ∂_i s ∂_j s Q^ss,
// so that the decay is −(i/2) Σ_class D, the class's weight ψ†Dψ, its mean Im ψ†A_iψ, its second moment Re ψ†B_ijψ.

/// A contribution to a class's channel on one axis (above).
#[derive(Clone, Debug)]
enum Source {
    Table { pair: usize, kp: usize, k: usize, jj: usize },
    Direct { d: usize, k: usize, values: Vec<C64> },
}

#[derive(Clone, Debug)]
struct Contribution {
    source: Source,
    class: usize,
    axis: usize,
    rows: Vec<usize>,
    cols: Vec<usize>,
}

/// An ordered pair of contributions (above): the rows in common (their places in each), and the four tables' Chebyshev
/// coefficients [piece][table][d][j1·|cols_q| + j2].
#[derive(Clone, Debug)]
struct QuadPair {
    p: usize,
    q: usize,
    coef: Vec<C64>,
}

/// The quadratic forms' tables (above), and at a point per class D, A_i, B_ij (n × n each).
#[derive(Clone, Debug, Default)]
pub(crate) struct Quad {
    contributions: Vec<Contribution>,
    pub classes: Vec<[i64; 2]>,
    pairs: Vec<QuadPair>,
    degree: usize,
    pub bytes: usize,
    pub elements: usize,
    pub build_seconds: f64,
    pub d: Vec<Vec<C64>>,
    pub a: Vec<[Vec<C64>; 3]>,
    pub b: Vec<[Vec<C64>; 6]>,
    pub work: f64,
}

/// The six (i ≤ j) pairs of axes.
pub(crate) const PAIRS6: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

impl Pairs {
    /// A table contribution's value and slope at x ∈ [−1, 1] of a piece (rows × cols, contiguous).
    fn table_at(&self, pair: usize, kp: usize, k: usize, jj: usize, at: usize, x: f64, out: &mut Vec<C64>, slope: &mut Vec<C64>) {
        let pt = &self.pairs[pair];
        let (rw, rz, per, m) = (self.ws[pt.w].rank, self.zs[pt.z].rank, pt.rows * pt.cols, self.degree + 1);
        let [pa, pb] = self.pieces[at];
        let (mut t, mut dt) = (Vec::new(), Vec::new());
        chebyshev(0.5 * (x + 1.0), self.degree, &mut t, &mut dt);
        let base = (((at * rw + kp) * rz + k) * self.orders + jj) * m * per;
        out.clear();
        out.resize(per, C64::ZERO);
        slope.clear();
        slope.resize(per, C64::ZERO);
        for d in 0..m {
            for e in 0..per {
                let z = pt.coef[base + d * per + e];
                out[e] += z * t[d];
                slope[e] += z * (dt[d] / (pb - pa));
            }
        }
    }

    /// The quadratic forms' tables of these tables (above).
    pub(crate) fn quad(&self, plan: &Plan, grouping: &Grouping) -> Quad {
        let clock = std::time::Instant::now();
        let n = plan.n;
        let mut classes: Vec<[i64; 2]> = Vec::new();
        let mut class_of = |fast: [i64; 2]| -> usize {
            classes.iter().position(|c| *c == fast).unwrap_or_else(|| {
                classes.push(fast);
                classes.len() - 1
            })
        };
        let mut contributions: Vec<Contribution> = Vec::new();
        for (pi, pt) in self.pairs.iter().enumerate() {
            if pt.h {
                continue;
            }
            let pair = &grouping.j_pairs[pt.at];
            let (emission, fast, which, _, _) = &grouping.channel_outputs[pair.out];
            if *which != 0 || (self.noside && plan.blocks[grouping.zs[pt.z].block].sideband.is_some()) {
                continue;
            }
            let class = class_of(*fast);
            let (rw, rz) = (self.ws[pt.w].rank, self.zs[pt.z].rank);
            for kp in 0..rw {
                for k in 0..rz {
                    for jj in 0..self.orders {
                        contributions.push(Contribution { source: Source::Table { pair: pi, kp, k, jj }, class, axis: *emission, rows: grouping.ws[pt.w].index.clone(), cols: grouping.zs[pt.z].index.clone() });
                    }
                }
            }
        }
        for (di, d) in self.directs.iter().enumerate() {
            if d.h || d.which != 0 {
                continue;
            }
            let (emission, fast, _, _, _) = &grouping.channel_outputs[d.out];
            let class = class_of(*fast);
            let list = &grouping.channel_lists[d.out];
            let mut rows: Vec<usize> = d.places.iter().map(|&p| list[p] as usize / n).collect();
            let mut cols: Vec<usize> = d.places.iter().map(|&p| list[p] as usize % n).collect();
            rows.sort_unstable();
            rows.dedup();
            cols.sort_unstable();
            cols.dedup();
            for (k, b) in d.basis.iter().enumerate() {
                let mut values = vec![C64::ZERO; rows.len() * cols.len()];
                for (&p, z) in d.places.iter().zip(b) {
                    let x = list[p] as usize;
                    let (i, j) = (rows.binary_search(&(x / n)).unwrap(), cols.binary_search(&(x % n)).unwrap());
                    values[i * cols.len() + j] += *z;
                }
                contributions.push(Contribution { source: Source::Direct { d: di, k, values }, class, axis: *emission, rows: rows.clone(), cols: cols.clone() });
            }
        }
        // The ordered pairs of one class and axis with rows in common, and their tables piece by piece.
        let degree = 2 * self.degree;
        let m = degree + 1;
        let xs: Vec<f64> = (0..m).map(|i| (std::f64::consts::PI * (i as f64 + 0.5) / m as f64).cos()).collect();
        let mut pairs = Vec::new();
        let (mut tp, mut sp, mut tq, mut sq) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let value = |c: &Contribution, at: usize, x: f64, out: &mut Vec<C64>, slope: &mut Vec<C64>| match &c.source {
            Source::Table { pair, kp, k, jj } => self.table_at(*pair, *kp, *k, *jj, at, x, out, slope),
            Source::Direct { values, .. } => {
                out.clear();
                out.extend_from_slice(values);
                slope.clear();
                slope.resize(values.len(), C64::ZERO);
            }
        };
        for (p, cp) in contributions.iter().enumerate() {
            for (q, cq) in contributions.iter().enumerate() {
                if cp.class != cq.class || cp.axis != cq.axis {
                    continue;
                }
                let common: Vec<(usize, usize)> = cp.rows.iter().enumerate().filter_map(|(i, r)| cq.rows.iter().position(|x| x == r).map(|j| (i, j))).collect();
                if common.is_empty() {
                    continue;
                }
                let (c1, c2) = (cp.cols.len(), cq.cols.len());
                let per = c1 * c2;
                let mut coef = vec![C64::ZERO; self.pieces.len() * 4 * m * per];
                for at in 0..self.pieces.len() {
                    let mut samples = vec![C64::ZERO; 4 * m * per];
                    for (ni, &x) in xs.iter().enumerate() {
                        value(cp, at, x, &mut tp, &mut sp);
                        value(cq, at, x, &mut tq, &mut sq);
                        for j1 in 0..c1 {
                            for j2 in 0..c2 {
                                let mut acc = [C64::ZERO; 4];
                                for &(i1, i2) in &common {
                                    let (a, a1) = (tp[i1 * c1 + j1].conj(), sp[i1 * c1 + j1].conj());
                                    let (b, b1) = (tq[i2 * c2 + j2], sq[i2 * c2 + j2]);
                                    acc[0] += a * b;
                                    acc[1] += a * b1;
                                    acc[2] += a1 * b;
                                    acc[3] += a1 * b1;
                                }
                                for (kind, v) in acc.iter().enumerate() {
                                    samples[(kind * m + ni) * per + j1 * c2 + j2] = *v;
                                }
                            }
                        }
                    }
                    for kind in 0..4 {
                        for d in 0..m {
                            let scale = if d == 0 { 1.0 / m as f64 } else { 2.0 / m as f64 };
                            for (ni, x) in xs.iter().enumerate() {
                                let c = (d as f64 * x.acos()).cos() * scale;
                                for e in 0..per {
                                    coef[((at * 4 + kind) * m + d) * per + e] += samples[(kind * m + ni) * per + e] * c;
                                }
                            }
                        }
                    }
                }
                pairs.push(QuadPair { p, q, coef });
            }
        }
        let elements = pairs.iter().map(|x| 4 * contributions[x.p].cols.len() * contributions[x.q].cols.len()).sum();
        let bytes = pairs.iter().map(|x| x.coef.len() * 16).sum();
        let nc = classes.len();
        Quad {
            contributions,
            classes,
            pairs,
            degree,
            bytes,
            elements,
            build_seconds: clock.elapsed().as_secs_f64(),
            d: vec![vec![C64::ZERO; n * n]; nc],
            a: (0..nc).map(|_| std::array::from_fn(|_| vec![C64::ZERO; n * n])).collect(),
            b: (0..nc).map(|_| std::array::from_fn(|_| vec![C64::ZERO; n * n])).collect(),
            work: 0.0,
        }
    }
}

impl Quad {
    /// Its pairs of contributions.
    pub(crate) fn pairs_len(&self) -> usize {
        self.pairs.len()
    }

    /// Per class D, A_i, B_ij at the point the tables' coordinates were last made at (`Pairs::contract` just before,
    /// with the same values `v`): the contributions' coefficients per column, then every pair's four tables at s.
    pub(crate) fn contract(&mut self, pairs: &Pairs, plan: &Plan, v: &Values, orders: usize) {
        let n = plan.n;
        let mut work = 0.0;
        for x in self.d.iter_mut().chain(self.a.iter_mut().flatten()).chain(self.b.iter_mut().flatten()) {
            x.fill(C64::ZERO);
        }
        let s = plan.intensity.first().map_or(0.0, |&f| v.factors[f].0.re);
        let ds: [f64; 3] = plan.intensity.first().map_or([0.0; 3], |&f| v.factors[f].1.map(|d| d.re));
        let at = pairs.piece_of(s);
        let [pa, pb] = pairs.pieces[at];
        let x = (2.0 * s - pa - pb) / (pb - pa);
        let (mut t, mut dt) = (Vec::new(), Vec::new());
        chebyshev(0.5 * (x + 1.0), self.degree, &mut t, &mut dt);
        let m = self.degree + 1;
        // The coefficients per contribution and column (and their gradients); None where its order is not taken.
        let factor = |p: usize, term: usize| -> (C64, [C64; 3]) {
            let (f, sc) = plan.terms[p][term];
            let (x, g) = v.factors[f];
            (x * sc, g.map(|d| d * sc))
        };
        let coeffs: Vec<Option<Vec<(C64, [C64; 3])>>> = self
            .contributions
            .iter()
            .map(|c| match &c.source {
                Source::Table { pair, kp, k, jj } => {
                    if *jj >= orders {
                        return None;
                    }
                    let pt = &pairs.pairs[*pair];
                    let a = pairs.phi[pt.w][*kp];
                    Some(pairs.psi[pt.z][*jj][*k].iter().map(|b| (a.0 * b.0, std::array::from_fn(|i| a.1[i] * b.0 + a.0 * b.1[i]))).collect())
                }
                Source::Direct { d, k, .. } => {
                    let cols = c.cols.len();
                    let dir = &pairs.directs[*d];
                    let mut acc = (C64::ZERO, [C64::ZERO; 3]);
                    for at in &dir.atoms {
                        let kk = v.coefficients[at.coefficient];
                        if kk == 0.0 {
                            continue;
                        }
                        let (x, dx) = factor(at.piece, at.term);
                        acc.0 += at.coords[*k] * x * kk;
                        for i in 0..3 {
                            acc.1[i] += at.coords[*k] * dx[i] * kk;
                        }
                    }
                    Some(vec![acc; cols])
                }
            })
            .collect();
        let mut q4: [Vec<C64>; 4] = Default::default();
        for pr in &self.pairs {
            let (Some(cp), Some(cq)) = (&coeffs[pr.p], &coeffs[pr.q]) else { continue };
            let (ap, aq) = (&self.contributions[pr.p], &self.contributions[pr.q]);
            let (c1, c2) = (ap.cols.len(), aq.cols.len());
            let per = c1 * c2;
            for (kind, q) in q4.iter_mut().enumerate() {
                q.clear();
                q.resize(per, C64::ZERO);
                let base = (at * 4 + kind) * m * per;
                for d in 0..m {
                    let td = t[d];
                    for (o, z) in q.iter_mut().zip(&pr.coef[base + d * per..base + (d + 1) * per]) {
                        *o += *z * td;
                    }
                }
            }
            work += (4 * m * per) as f64 * 0.5;
            let cls = ap.class;
            for j1 in 0..c1 {
                let (u, du) = (cp[j1].0.conj(), cp[j1].1.map(C64::conj));
                for j2 in 0..c2 {
                    let (w, dw) = cq[j2];
                    let e = j1 * c2 + j2;
                    let (q0, qs, qt, qss) = (q4[0][e], q4[1][e], q4[2][e], q4[3][e]);
                    let place = ap.cols[j1] * n + aq.cols[j2];
                    self.d[cls][place] += u * w * q0;
                    for i in 0..3 {
                        self.a[cls][i][place] += u * (dw[i] * q0 + w * qs * ds[i]);
                    }
                    for (r, &(i, j)) in PAIRS6.iter().enumerate() {
                        self.b[cls][r][place] += du[i] * dw[j] * q0 + du[i] * w * qs * ds[j] + u * dw[j] * qt * ds[i] + u * w * qss * (ds[i] * ds[j]);
                    }
                    work += 20.0;
                }
            }
        }
        self.work = work;
    }
}

/// Per class (the order of `Quad::classes`) D, A_i, B_ij from a contraction's channel outputs (its `which` 0): each
/// class's channels summed over their keys per emission axis, then Σ_axis L†L, L†∂_iL, ∂_iL†∂_jL.
pub(crate) fn quadratic_of(out: &Out, grouping: &Grouping, classes: &[[i64; 2]], n: usize) -> (Vec<Vec<C64>>, Vec<[Vec<C64>; 3]>, Vec<[Vec<C64>; 6]>) {
    let nc = classes.len();
    assert!(grouping.channel_outputs.iter().all(|o| o.0 < 3), "three emission axes");
    let mut l: Vec<[[Vec<C64>; 4]; 3]> = (0..nc).map(|_| std::array::from_fn(|_| std::array::from_fn(|_| vec![C64::ZERO; n * n]))).collect();
    for (o, c) in out.channels.iter().enumerate() {
        let (emission, fast, which, _, _) = &grouping.channel_outputs[o];
        if *which != 0 {
            continue;
        }
        let Some(cls) = classes.iter().position(|x| x == fast) else { continue };
        for (k, &x) in grouping.channel_lists[o].iter().enumerate() {
            let x = x as usize;
            l[cls][*emission][0][x] += c.value[k];
            for i in 0..3 {
                l[cls][*emission][i + 1][x] += c.gradient[i][k];
            }
        }
    }
    let mut d = vec![vec![C64::ZERO; n * n]; nc];
    let mut a: Vec<[Vec<C64>; 3]> = (0..nc).map(|_| std::array::from_fn(|_| vec![C64::ZERO; n * n])).collect();
    let mut b: Vec<[Vec<C64>; 6]> = (0..nc).map(|_| std::array::from_fn(|_| vec![C64::ZERO; n * n])).collect();
    // (X†Y)[j1, j2] = Σ_i X[i, j1]* Y[i, j2].
    let product = |x: &[C64], y: &[C64], out: &mut [C64]| {
        for i in 0..n {
            for j1 in 0..n {
                let u = x[i * n + j1].conj();
                if u == C64::ZERO {
                    continue;
                }
                for j2 in 0..n {
                    out[j1 * n + j2] += u * y[i * n + j2];
                }
            }
        }
    };
    for cls in 0..nc {
        for ax in 0..3 {
            let m = &l[cls][ax];
            product(&m[0], &m[0], &mut d[cls]);
            for i in 0..3 {
                product(&m[0], &m[i + 1], &mut a[cls][i]);
            }
            for (r, &(i, j)) in PAIRS6.iter().enumerate() {
                product(&m[i + 1], &m[j + 1], &mut b[cls][r]);
            }
        }
    }
    (d, a, b)
}
