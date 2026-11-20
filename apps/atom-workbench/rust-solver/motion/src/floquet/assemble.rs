//! Lindblad → sparse Liouville by lattice components, the bordered system on the elements, and the readouts (td_s.Liou,
//! mdlat.PerM / assemble / readouts, fr.assemble / readouts). One function for the reference and the model.
//!
//! In the frame (an operator's entry (r, c) at key v moves to v − θ_r + θ_c, θ the states' frame vectors; the static
//! part's diagonal less θ_a·w), on ρ as a vector (x = a·n + b for ρ_ab):
//! - L = −i(H⊗1 − 1⊗Hᵀ) + Σ_j Σ_fg [C_f ⊗ C_g* − ½(C_g†C_f ⊗ 1 + 1 ⊗ (C_g†C_f)ᵀ)] at f − g, per channel j;
//! - K1_i = −½(∂_iH⊗1 + 1⊗∂_iHᵀ) and the jumps' gradient terms (∂_iC ⊗ C* − C ⊗ ∂_iC*)/2i + ¼i(Z⊗1 − 1⊗Zᵀ),
//!   Z = C†∂_iC, ∂_iC†C, for i = x, y, z (the force vector, RUSTPLAN correction "select by L, the force vector");
//! - readout rows: each state's population, each lift row's (Σ_fg tr(M_g[q]* M_f[q] ρ)), each channel's rate tr(C†C ρ), the
//!   force's components tr(K1_i ρ).
//! Every frame vector is mapped to its lattice key (m, n) (frame.rs `Lattice`) and the components of one key summed.
//! The system on the admitted elements y = (m, n, x): [[L + i f(m, n), τ₀*], [τ₀ᵀ, 0]] (ρ; λ) = (0; 1), τ₀ the k = 0
//! populations; F_i = τ₀·K1_iρ. Only ρ is solved (the diffusion is not judged).

use std::collections::HashMap;

use super::frame::FrameKey;
use super::{Element, Elements, Frame, Hilbert, Lattice, Part, Sparse};
use crate::linalg::{C64, Mat};

/// A lattice key (m, n).
pub type LatticeKey = (Vec<i64>, [i64; 3]);

/// One lattice component of the Liouvillian: L and K1_i (d × d, d = n²), each absent where zero.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Superoperators {
    pub l: Option<Sparse>,
    pub k1: [Option<Sparse>; 3],
}

/// A Hilbert model's Liouvillian on its lattice, and its readout rows.
#[derive(Clone, Debug)]
pub struct Liouville {
    /// n states; d = n².
    pub states: usize,
    pub lattice: Lattice,
    /// The components by lattice key.
    pub components: HashMap<LatticeKey, Superoperators>,
    /// The readout rows by lattice key (rows × d): the populations (n), the lift's rows, the channels' rates, the force's
    /// three components.
    pub rows: HashMap<LatticeKey, Sparse>,
    /// The lift rows' and the channels' counts.
    pub lifted: usize,
    pub channels: usize,
    /// The levels over the 32 states, and each model state's and each lift row's state among the 32.
    pub groups: Vec<Vec<usize>>,
    pub of_state: Vec<usize>,
    pub of_row: Vec<usize>,
    /// L's diagonal at the zero key (λ₀ of the rounding rule).
    pub diagonal: Vec<C64>,
}

/// Triplets (row, column, value).
type Triplets = Vec<(u32, u32, C64)>;

/// Triplets summed into compressed rows, zeros dropped.
fn compress(rows: usize, cols: usize, mut t: Triplets) -> Sparse {
    t.sort_by_key(|&(r, c, _)| (r, c));
    let mut out = Sparse { rows, cols, indptr: vec![0; rows + 1], indices: Vec::new(), values: Vec::new() };
    let mut i = 0;
    while i < t.len() {
        let (r, c) = (t[i].0, t[i].1);
        let mut v = C64::ZERO;
        while i < t.len() && t[i].0 == r && t[i].1 == c {
            v += t[i].2;
            i += 1;
        }
        if v != C64::ZERO {
            out.indices.push(c as usize);
            out.values.push(v);
            out.indptr[r as usize + 1] += 1;
        }
    }
    for r in 0..rows {
        out.indptr[r + 1] += out.indptr[r];
    }
    out
}

/// What the frame vectors carry before they are mapped onto the lattice: per vector, L, K1, K2 and the readout rows.
#[derive(Default)]
struct Terms {
    keys: Vec<FrameKey>,
    of: HashMap<FrameKey, usize>,
    which: Vec<[Triplets; 4]>,
    rows: Vec<Triplets>,
}

impl Terms {
    fn at(&mut self, v: &FrameKey) -> usize {
        if let Some(&i) = self.of.get(v) {
            return i;
        }
        self.keys.push(v.clone());
        self.of.insert(v.clone(), self.keys.len() - 1);
        self.which.push(Default::default());
        self.rows.push(Vec::new());
        self.keys.len() - 1
    }
}

/// An operator list in the frame: each nonzero entry (r, c) of a part at key v moves to v − θ_r + θ_c; the entries are
/// gathered per frame vector, the vectors in sorted order per part (td_s.to_frame).
fn to_frame<'a>(parts: impl IntoIterator<Item = (&'a [i32], &'a Mat)>, theta: &[&FrameKey]) -> Vec<(FrameKey, Mat)> {
    let mut out: Vec<(FrameKey, Mat)> = Vec::new();
    for (key, x) in parts {
        let mut entries: Vec<(FrameKey, usize, usize)> = Vec::new();
        for r in 0..x.rows {
            for c in 0..x.cols {
                if x[(r, c)] != C64::ZERO {
                    let v: FrameKey = (0..theta[r].len()).map(|i| super::frame::element(key.get(i).map_or(0, |&k| k as i64), theta[r][i], theta[c][i])).collect();
                    entries.push((v, r, c));
                }
            }
        }
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (v, r, c) in entries {
            let i = match out.iter().position(|o| o.0 == v) {
                Some(i) => i,
                None => {
                    out.push((v, Mat::zeros(x.rows, x.cols)));
                    out.len() - 1
                }
            };
            out[i].1[(r, c)] += x[(r, c)];
        }
    }
    out
}

fn difference(a: &FrameKey, b: &FrameKey) -> FrameKey {
    a.iter().zip(b).map(|(x, y)| x - y).collect()
}

/// Σ over pairs (f ∈ a, g ∈ b) of s·(X_f ⊗ Y_g*) at f − g, into `terms`' list `which`; dense per vector, then its nonzeros.
fn pairs(terms: &mut Terms, which: usize, a: &[(FrameKey, Mat)], b: &[(FrameKey, Mat)], s: C64, n: usize) {
    let d = n * n;
    let mut groups: Vec<(FrameKey, Vec<(usize, usize)>)> = Vec::new();
    for (i, (va, _)) in a.iter().enumerate() {
        for (j, (vb, _)) in b.iter().enumerate() {
            let v = difference(va, vb);
            match groups.iter_mut().find(|g| g.0 == v) {
                Some(g) => g.1.push((i, j)),
                None => groups.push((v, vec![(i, j)])),
            }
        }
    }
    for (v, list) in groups {
        let mut dense = vec![C64::ZERO; d * d];
        for (i, j) in list {
            let (x, y) = (&a[i].1, &b[j].1);
            let xs: Vec<(usize, usize, C64)> = nonzero(x);
            let ys: Vec<(usize, usize, C64)> = nonzero(y);
            for &(p, q, xv) in &xs {
                for &(r, t, yv) in &ys {
                    dense[(p * n + r) * d + q * n + t] += xv * yv.conj();
                }
            }
        }
        let at = terms.at(&v);
        for (k, z) in dense.into_iter().enumerate() {
            if z != C64::ZERO {
                terms.which[at][which].push(((k / d) as u32, (k % d) as u32, z * s));
            }
        }
    }
}

fn nonzero(x: &Mat) -> Vec<(usize, usize, C64)> {
    (0..x.rows).flat_map(|r| (0..x.cols).map(move |c| (r, c))).filter(|&(r, c)| x[(r, c)] != C64::ZERO).map(|(r, c)| (r, c, x[(r, c)])).collect()
}

/// Σ over pairs (f ∈ a, g ∈ b) of Y_g† X_f at f − g.
fn products(a: &[(FrameKey, Mat)], b: &[(FrameKey, Mat)]) -> Vec<(FrameKey, Mat)> {
    let mut out: Vec<(FrameKey, Mat)> = Vec::new();
    for (va, x) in a {
        for (vb, y) in b {
            let v = difference(va, vb);
            let z = y.adjoint_product(x);
            match out.iter_mut().find(|o| o.0 == v) {
                Some(o) => o.1.add_scaled(C64::ONE, &z),
                None => out.push((v, z)),
            }
        }
    }
    out.retain(|o| o.1.data.iter().any(|&z| z != C64::ZERO));
    out
}

/// s·(X ⊗ 1) + t·(1 ⊗ Xᵀ) into `which` at v.
fn sides(terms: &mut Terms, which: usize, v: &FrameKey, x: &Mat, s: C64, t: C64) {
    let n = x.rows;
    let at = terms.at(v);
    for (a, c, z) in nonzero(x) {
        for b in 0..n {
            // (X ⊗ 1)[(a, b), (c, b)] = X_ac; (1 ⊗ Xᵀ)[(b, a), (b, c)] = X_ca... written per entry X_ac: row b·n + c, col b·n + a.
            terms.which[at][which].push(((a * n + b) as u32, (c * n + b) as u32, z * s));
            terms.which[at][which].push(((b * n + c) as u32, (b * n + a) as u32, z * t));
        }
    }
}

/// A row on ρ that reads tr(O ρ) for each entry of O: position c·n + r holds O_rc.
fn trace_row(terms: &mut Terms, row: usize, v: &FrameKey, o: &Mat) {
    let n = o.rows;
    let at = terms.at(v);
    for (r, c, z) in nonzero(o) {
        terms.rows[at].push((row as u32, (c * n + r) as u32, z));
    }
}

impl Liouville {
    /// The Liouvillian of a Hilbert model in the frame of its point (td_s.Liou + mdlat.PerM): the reference and the model
    /// at one point take the same frame (`Frame::new` on the reference).
    pub fn new(frame: &Frame, hilbert: &Hilbert) -> Liouville {
        let n = hilbert.states.len();
        let d = n * n;
        let theta: Vec<&FrameKey> = hilbert.states.iter().map(|&s| &frame.theta[s]).collect();
        let size = frame.values.len();
        let zero: FrameKey = vec![0; size];
        let l = 0;
        let mut terms = Terms::default();
        let i = C64::I;
        // H, with the frame's diagonal at the zero vector.
        let mut h = to_frame(hilbert.hamiltonian.iter().map(|p| (p.key.as_slice(), &p.value)), &theta);
        if !h.iter().any(|o| o.0 == zero) {
            h.push((zero.clone(), Mat::zeros(n, n)));
        }
        let shift: Vec<f64> = theta.iter().map(|t| t.iter().zip(&frame.values).map(|(&k, w)| k as f64 * w).sum()).collect();
        let at = h.iter().position(|o| o.0 == zero).expect("the zero vector");
        for a in 0..n {
            h[at].1[(a, a)] -= C64::real(shift[a]);
        }
        for (v, x) in &h {
            sides(&mut terms, l, v, x, -i, i);
        }
        for axis in 0..3 {
            for (v, x) in &to_frame(hilbert.hamiltonian.iter().map(|p| (p.key.as_slice(), &p.gradient[axis])), &theta) {
                sides(&mut terms, 1 + axis, v, x, C64::real(-0.5), C64::real(-0.5));
            }
        }
        // Readout rows: the populations at the zero vector.
        let z0 = terms.at(&zero);
        for a in 0..n {
            terms.rows[z0].push((a as u32, (a * n + a) as u32, C64::ONE));
        }
        // The lift's rows.
        let (of_row, lifted) = match &hilbert.lift {
            Some(lift) => {
                let rows = lift.folded.len();
                for q in 0..rows {
                    let mut ops: Vec<(Vec<i32>, Mat)> = Vec::new();
                    for (kf, mf) in &lift.amplitudes {
                        for (kg, mg) in &lift.amplitudes {
                            let key: Vec<i32> = kf.iter().zip(kg).map(|(x, y)| x - y).collect();
                            ops.push((key, Mat::from_fn(n, n, |a, b| mg[(q, a)].conj() * mf[(q, b)])));
                        }
                    }
                    for (v, o) in &to_frame(ops.iter().map(|(k, o)| (k.as_slice(), o)), &theta) {
                        trace_row(&mut terms, n + q, v, o);
                    }
                }
                (lift.folded.clone(), rows)
            }
            None => (Vec::new(), 0),
        };
        // The jumps, per channel.
        let channels = hilbert.jumps.len();
        for (j, jump) in hilbert.jumps.iter().enumerate() {
            let c = to_frame(jump.iter().map(|p: &Part| (p.key.as_slice(), &p.value)), &theta);
            pairs(&mut terms, l, &c, &c, C64::ONE, n);
            for (v, z) in &products(&c, &c) {
                sides(&mut terms, l, v, z, C64::real(-0.5), C64::real(-0.5));
                trace_row(&mut terms, n + lifted + j, v, z);
            }
            for axis in 0..3 {
                let dc = to_frame(jump.iter().map(|p: &Part| (p.key.as_slice(), &p.gradient[axis])), &theta);
                if dc.is_empty() {
                    continue;
                }
                pairs(&mut terms, 1 + axis, &dc, &c, C64::new(0.0, -0.5), n);
                pairs(&mut terms, 1 + axis, &c, &dc, C64::new(0.0, 0.5), n);
                for (v, z) in products(&c, &dc).iter().chain(&products(&dc, &c)) {
                    sides(&mut terms, 1 + axis, v, z, C64::new(0.0, 0.25), C64::new(0.0, -0.25));
                }
            }
        }
        // The force rows: τ·K1_i at each vector of K1_i.
        let force = n + lifted + channels;
        for t in 0..terms.keys.len() {
            for axis in 0..3 {
                let row: Triplets = terms.which[t][1 + axis]
                    .iter()
                    .filter(|&&(r, _, _)| (r as usize) % (n + 1) == 0)
                    .map(|&(_, c, z)| ((force + axis) as u32, c, z))
                    .collect();
                terms.rows[t].extend(row);
            }
        }
        // The lattice of every vector, and the components summed per lattice key.
        let lattice = Lattice::new(frame, &terms.keys);
        let mut components: HashMap<LatticeKey, [Triplets; 4]> = HashMap::new();
        let mut rows: HashMap<LatticeKey, Triplets> = HashMap::new();
        for (t, v) in terms.keys.iter().enumerate() {
            let key = lattice.key(frame, v);
            let entry = components.entry(key.clone()).or_default();
            for w in 0..4 {
                entry[w].extend(std::mem::take(&mut terms.which[t][w]));
            }
            rows.entry(key).or_default().extend(std::mem::take(&mut terms.rows[t]));
        }
        let height = force + 3;
        let components: HashMap<LatticeKey, Superoperators> = components
            .into_iter()
            .map(|(k, [a, x, y, z])| {
                let make = |t: Triplets| Some(compress(d, d, t)).filter(|s| !s.values.is_empty());
                (k, Superoperators { l: make(a), k1: [make(x), make(y), make(z)] })
            })
            .collect();
        let rows = rows.into_iter().map(|(k, t)| (k, compress(height, d, t))).filter(|(_, s)| !s.values.is_empty()).collect();
        let m0 = (vec![0; lattice.slow.len()], [0; 3]);
        let mut diagonal = vec![C64::ZERO; d];
        if let Some(Some(s)) = components.get(&m0).map(|c| c.l.as_ref()) {
            for (x, z) in diagonal.iter_mut().enumerate() {
                for p in s.indptr[x]..s.indptr[x + 1] {
                    if s.indices[p] == x {
                        *z = s.values[p];
                    }
                }
            }
        }
        Liouville {
            states: n,
            lattice,
            components,
            rows,
            lifted,
            channels,
            groups: hilbert.groups.clone(),
            of_state: hilbert.states.clone(),
            of_row,
            diagonal,
        }
    }
}

/// The bordered sparse system on the elements.
#[derive(Clone, Debug, PartialEq)]
pub struct System {
    /// The unknowns' elements (the bordering multiplier is the last unknown, index elements.len()).
    pub elements: Elements,
    /// [[L + i f, τ₀*], [τ₀ᵀ, 0]], (N + 1) × (N + 1), N = elements' count.
    pub matrix: Sparse,
    /// K1_i on the elements (N × N), i = x, y, z.
    pub k1: [Sparse; 3],
    /// τ₀ over the elements: 1 at the k = 0 populations.
    pub trace: Vec<C64>,
    /// Each unknown's harmonic, N + 1 of them (fr.harmonic_ids): the index of its slow lattice point m (not the (m, n)
    /// sector) among the system's sorted distinct m; the multiplier's row takes the populations' m₀.
    pub harmonic: Vec<usize>,
}

/// The elements of each sector (m, n): their unknowns' indices and their Liouville indices x, with x → unknown.
struct Sectors {
    of: HashMap<LatticeKey, (Vec<usize>, Vec<usize>, Vec<i64>)>,
}

impl Sectors {
    fn new(elements: &[Element], d: usize) -> Sectors {
        let mut of: HashMap<LatticeKey, (Vec<usize>, Vec<usize>, Vec<i64>)> = HashMap::new();
        for (i, y) in elements.iter().enumerate() {
            let e = of.entry((y.m.clone(), y.n)).or_insert_with(|| (Vec::new(), Vec::new(), vec![-1; d]));
            e.0.push(i);
            e.1.push(y.x);
            e.2[y.x] = i as i64;
        }
        Sectors { of }
    }
}

/// One superoperator on the elements: for each component (m, n) and each source sector (j, n₀) its columns' entries landing
/// in the sector (j + m, n₀ + n), where admitted.
fn on_elements(parts: &Liouville, sectors: &Sectors, pick: impl Fn(&Superoperators) -> Option<&Sparse>) -> Triplets {
    let mut t = Triplets::new();
    for ((m, n), ops) in &parts.components {
        let Some(x) = pick(ops) else { continue };
        for ((j0, n0), (ii, xs, _)) in &sectors.of {
            let target: LatticeKey = (j0.iter().zip(m).map(|(a, b)| a + b).collect(), std::array::from_fn(|k| n0[k] + n[k]));
            let Some((_, _, map)) = sectors.of.get(&target) else { continue };
            // Columns xs of x: walk x's rows (CSR) and keep the entries in admitted columns.
            let mut column = vec![-1i64; x.cols];
            for (k, &c) in xs.iter().enumerate() {
                column[c] = ii[k] as i64;
            }
            for r in 0..x.rows {
                if map[r] < 0 {
                    continue;
                }
                for p in x.indptr[r]..x.indptr[r + 1] {
                    let c = column[x.indices[p]];
                    if c >= 0 {
                        t.push((map[r] as u32, c as u32, x.values[p]));
                    }
                }
            }
        }
    }
    t
}

/// The system on the admitted elements.
pub fn assemble(parts: &Liouville, elements: &Elements) -> System {
    let n = parts.states;
    let d = n * n;
    let el = &elements.order;
    let count = el.len();
    let sectors = Sectors::new(el, d);
    let mut a = on_elements(parts, &sectors, |s| s.l.as_ref());
    for (i, y) in el.iter().enumerate() {
        a.push((i as u32, i as u32, C64::new(0.0, parts.lattice.frequency(&y.m, &y.n))));
    }
    let m0 = vec![0i64; parts.lattice.slow.len()];
    let population = |s: usize| el.iter().position(|y| y.m == m0 && y.n == [0; 3] && y.x == s * n + s).expect("the populations are admitted");
    let mut trace = vec![C64::ZERO; count];
    for s in 0..n {
        let p = population(s);
        trace[p] = C64::ONE;
        a.push((p as u32, count as u32, C64::ONE));
        a.push((count as u32, p as u32, C64::ONE));
    }
    let mut keys: Vec<&Vec<i64>> = el.iter().map(|y| &y.m).collect();
    keys.sort();
    keys.dedup();
    let index = |m: &Vec<i64>| keys.binary_search(&m).expect("a key");
    let mut harmonic: Vec<usize> = el.iter().map(|y| index(&y.m)).collect();
    harmonic.push(index(&el[population(0)].m));
    System {
        elements: elements.clone(),
        matrix: compress(count + 1, count + 1, a),
        k1: std::array::from_fn(|i| compress(count, count, on_elements(parts, &sectors, |s| s.k1[i].as_ref()))),
        trace,
        harmonic,
    }
}

/// What a solved point reads (td_s.pack, t4_td.readouts): the levels' populations (the kept states' and the lift rows',
/// over their sum), the rates, the force vector.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Readouts {
    /// Per level (`Hilbert::groups`), normalized by the total.
    pub levels: Vec<f64>,
    /// Per kept state and per lift row, over the same total.
    pub populations: Vec<f64>,
    pub folded: Vec<f64>,
    /// The scattering rate per channel (1/µs) and their sum R.
    pub rates: Vec<f64>,
    pub rate: f64,
    /// F_i = τ₀·K1_iρ (rad/µs/µm: ħ = 1).
    pub force: [f64; 3],
}

fn apply(s: &Sparse, v: &[C64]) -> Vec<C64> {
    (0..s.rows).map(|r| (s.indptr[r]..s.indptr[r + 1]).map(|p| s.values[p] * v[s.indices[p]]).sum()).collect()
}

/// The readouts of a solved point: ρ on the system's elements (the multiplier dropped).
pub fn readouts(parts: &Liouville, system: &System, rho: &[C64]) -> Readouts {
    let dot = |a: &[C64], b: &[C64]| a.iter().zip(b).map(|(&x, &z)| x * z).sum::<C64>();
    let force = std::array::from_fn(|i| dot(&system.trace, &apply(&system.k1[i], rho)).re);
    // The time averages: each row component at (m, n) reads the elements of the sector (−m, −n).
    let n = parts.states;
    let height = n + parts.lifted + parts.channels + 3;
    let mut avg = vec![C64::ZERO; height];
    let mut by_sector: HashMap<LatticeKey, Vec<(usize, usize)>> = HashMap::new();
    for (i, e) in system.elements.order.iter().enumerate() {
        by_sector.entry((e.m.iter().map(|x| -x).collect(), e.n.map(|x| -x))).or_default().push((e.x, i));
    }
    for (key, s) in &parts.rows {
        let Some(list) = by_sector.get(key) else { continue };
        let mut at = vec![usize::MAX; s.cols];
        for &(x, i) in list {
            at[x] = i;
        }
        for r in 0..s.rows {
            for p in s.indptr[r]..s.indptr[r + 1] {
                let i = at[s.indices[p]];
                if i != usize::MAX {
                    avg[r] += s.values[p] * rho[i];
                }
            }
        }
    }
    let pops: Vec<f64> = avg[..n].iter().map(|z| z.re).collect();
    let folded: Vec<f64> = avg[n..n + parts.lifted].iter().map(|z| z.re).collect();
    let rates: Vec<f64> = avg[n + parts.lifted..n + parts.lifted + parts.channels].iter().map(|z| z.re).collect();
    let total: f64 = pops.iter().sum::<f64>() + folded.iter().sum::<f64>();
    let levels = parts
        .groups
        .iter()
        .map(|g| {
            let kept: f64 = (0..n).filter(|&a| g.contains(&parts.of_state[a])).map(|a| pops[a]).sum();
            let lifted: f64 = (0..parts.lifted).filter(|&q| g.contains(&parts.of_row[q])).map(|q| folded[q]).sum();
            (kept + lifted) / total
        })
        .collect();
    Readouts {
        levels,
        populations: pops.iter().map(|p| p / total).collect(),
        folded: folded.iter().map(|p| p / total).collect(),
        rate: rates.iter().sum(),
        rates,
        force,
    }
}
