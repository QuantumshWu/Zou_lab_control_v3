//! The fold's step (B2; replan §3, onefold.status.md's plan of 2026-11-16): the same object as `Fold::at` -- the same
//! blocks, rows, splits, passes, readings' rule and compression --, evaluated in a decomposition kept along the path and
//! shared by the fold's atoms, in buffers laid out once per structure (`Workspace`): a step asks the heap for nothing
//! (making or taking up a decomposition is the cache's, not the step's). `Fold::at` stays the definition and the tests'
//! truth; at a fresh cache the two agree to the rounding.
//!
//! # What a step does
//! 1. The light at the point: every beam's amplitude a_l, every frequency's field F, 𝒦 and H_GG (the request's levels with
//!    the outside forms of F there), every component's coefficients on its field's basis e_k = Σ_l a_l γ_lk and every beam's
//!    Doppler shift. Every entry the blocks and P take in x coordinates is a slot (`Slot`): 𝒦's, H_GG's, or a drive
//!    Σ_k e_k U_k[x, t] (its conjugate), less a constant; the slots are laid out once and filled every step.
//! 2. Each block's decomposition (`Cache`, keyed like `Path`: its solved rows, and the block where its copies dress it):
//!    the Schur form V T V† of K = Ĥ_SS + Ĥ_SC R₀ Ĥ_CS + o on its solved rows (the exact copies' couplings and the couplings
//!    between the lines left out), R₀ = (z₀ − Ĥ_CC)⁻¹ at its reference. Its diagonal is moved by the first order of the
//!    change, κ_i = T_ii + Σ_t δc_t (V†M_tV)_ii, term by term (`Term`): the outside forms c = F_p*F_q on the excited rows,
//!    and the centred copies' dressing c = e_k e_k′* with M = U_k R₀ U_k′† -- R₀ the reference's. The exact copies' own
//!    rows move by their change itself, V_D†δK_DD V_D (few rows), their couplings with the excited rows are taken at the
//!    point (Woodbury, fold.rs # The exact copies).
//! 3. The passes as `Fold::at` makes them (two on diagonal generators, a third on the Bloch generator where the next one
//!    reads beyond the share): every beam's source a_l Σ_k γ_lk Y_k (Y_k = V†U_k, kept with the decomposition) and the
//!    static one V†(B_S + Ĥ_SC R₀ B_C) solved in the basis -- on a diagonal generator each column a division by
//!    h_j − w_l − κ_i --, the centred rows T_C = R₀(Ĥ_CS V Z + B_C) = X_C†Z + R₀B_C (X_C = V†Ĥ_SC R₀), and the Bloch generator
//!    Ĥ_PP + W†Z + Ê Z + Ĥ_PC T_C (W = Σ_l of the beams' sources, the beams' emission's adjoint; Ê the static emission's
//!    rows in the basis).
//! 4. The reuse's reading, row by row and line by line (fold.rs # One decomposition along the path), its norms from the
//!    terms: a row's error in a column is at most Σ_L off_iL ‖Z_L∖i‖ / |c − κ_i|, off_iL = ‖(V†(K_ref − K_kept)V)[i, L∖i]‖
//!    + Σ_t |δc_t| ‖M̃_t[i, L∖i]‖ -- what the decomposition left out at its reference (on its own line what the Schur form
//!    dropped, on the other line the couplings between the lines), exactly, and the change since; and R₀'s own change
//!    since the reference, δR₀ ≤ ‖R₀‖²‖δM‖/(1 − ‖R₀‖‖δM‖) with
//!    δM the change of z₀ − Ĥ_CC, through the dressing on the solved rows and on the centred rows themselves (their
//!    population's error). Beyond the share a block takes the nearest kept decomposition, then a new one, as `Fold::at`.
//! 5. The compression, the object's whole (DESIGN §1.3): A = Φ†ĤΦ with Φ = [1; T] and Ĥ at the point, every term of it --
//!    the couplings between the lines the decomposition left out, the decay D exactly, the dressing through the centred
//!    rows --, in the coordinates where Ĥ is sparse: T_S = VZ once, then A = Bloch + (W†Z)† + T†(ĤT + B) and M = 1 + T†T.
//!    Its anti-Hermitian part is −(i/2)Σ_o ε_o†Dε_o for any T, so the decay is the jumps' products by construction.
//!    S = M^{1/2}, G = S⁻¹AS⁻¹.
//!
//! Not here yet (the plan's later steps): the force's adjoint, the three nodes and the photons, the tones.
use super::*;
use crate::field::Amplitude;

/// What an entry at the point is (x coordinates): 𝒦's (two excited states), H_GG's (two ground states), a component's
/// drive Σ_k e_k U_k[x, t] (component, excited state, ground state), or its conjugate.
#[derive(Clone, Copy, Debug)]
enum Entry {
    Excited(usize, usize),
    Ground(usize, usize),
    Drive(usize, usize, usize),
    Back(usize, usize, usize),
}

/// An entry of a sparse matrix at the point: its row and column, what it is, the constant it is less, its value.
#[derive(Clone, Copy, Debug)]
struct Slot {
    row: usize,
    col: usize,
    what: Entry,
    less: f64,
    value: C64,
}

/// A term of K's change on the excited rows (module documentation, 2): its coefficient -- an outside form's F_p*F_q
/// (frequency, p, q), or a copy component's e_k e_k′* (component, k, k′) --, its value at the reference, and of
/// M̃ = V†MV the diagonal and per row and group the norm of the row's rest on the group (its own group less itself).
enum Coefficient {
    Form(usize, usize, usize),
    Dressing(usize, usize, usize),
}

struct Term {
    what: Coefficient,
    at: C64,
    diag: Vec<C64>,
    norms: Vec<f64>,
}

/// A block's sources in a basis: per source component its Y_k = V†U_k on its source places (solved rows × n); per copy
/// component its Ỹ_k = V†U_k on the copies' columns (solved rows × (exact + centred copies)), X_k = Ỹ_k[:, centred] R₀ and
/// per row Ỹ_k's norm on the centred columns.
struct Ranks {
    sources: Vec<(usize, Vec<Mat>)>,
    copies: Vec<(usize, Vec<Mat>, Vec<Mat>, Vec<Vec<f64>>)>,
}

/// A kept decomposition (module documentation, 2): its key; K kept at its reference (x coordinates, the exact copies'
/// couplings out; the nearest one's distance); its Schur form; per basis row its group (its line, or the exact copies')
/// and its Schur part; the terms; per row and group the norm of what the decomposition leaves out on the other lines at
/// its reference (V†(K_ref − K_kept)V's); its reference's R₀, z₀, Ĥ_CC, ‖R₀‖_F, the exact copies' K and their Schur vectors V_D
/// (exact × exact); per block its ranks.
struct Basis {
    rows: Vec<Row>,
    own: Option<usize>,
    k: Mat,
    factored: Factored,
    group: Vec<usize>,
    part: Vec<usize>,
    terms: Vec<Term>,
    cross: Vec<f64>,
    r0: Mat,
    z0: f64,
    cc: Mat,
    r0_norm: f64,
    kdd: Mat,
    vd: Mat,
    ranks: Vec<Option<Ranks>>,
}

/// The decompositions kept, shared by a fold's atoms (fold.rs `Path`, the same rule): `current` per key the basis the
/// atom stands on (an atom's own: clear it for another atom), `made` and `looked` the decompositions made and the kept
/// ones taken up again.
#[derive(Default)]
pub(crate) struct Cache {
    bases: Vec<Basis>,
    pub(crate) current: Vec<((Vec<Row>, Option<usize>), usize)>,
    pub(crate) made: usize,
    pub(crate) looked: usize,
}

/// A step's result (module documentation, 5), P's n × n matrices: A, the metric less one, S⁻¹, G; the passes solved and their ratio (≥ 1: the
/// self-consistency does not exist); per block its reuse reading (rate units); the complex multiply-adds by phase.
pub(crate) struct Step {
    pub(crate) a: Mat,
    pub(crate) metric: Mat,
    pub(crate) normal: Mat,
    pub(crate) generator: Mat,
    pub(crate) passes: usize,
    pub(crate) ratio: f64,
    pub(crate) reuse: Vec<f64>,
    pub(crate) work: f64,
    pub(crate) phases: [(&'static str, f64); 6],
}

/// A block's layout and buffers (module documentation): its sizes (Q rows m, solved s, of them excited ne and exact
/// copies nd, centred nc); its slots -- Ĥ on its Q rows, the static sources (row, P place), the static emission (P place,
/// row), its part of Ĥ_PP; its centre's terms (a ground source: (state, its frame), an explicit state: (state, o)); its
/// splits (None the static one, else (component, beam)) with per source component its places; the P places a beam
/// sources (W's columns), the static sources' places, the P rows the static emission from its solved rows reaches; per
/// solved row its group and the groups' count. Then the step's buffers.
struct Lay {
    m: usize,
    s: usize,
    ne: usize,
    nd: usize,
    nc: usize,
    inner: Vec<Slot>,
    absorb: Vec<Slot>,
    emit: Vec<Slot>,
    centre: Vec<(bool, usize, f64)>,
    splits: Vec<Option<(usize, usize)>>,
    places: Vec<(usize, Vec<usize>)>,
    sourced: Vec<usize>,
    statics: Vec<usize>,
    emitting: Vec<usize>,
    group: Vec<usize>,
    groups: usize,
    // The step's.
    basis: usize,
    kappa: Vec<C64>,
    coefficients: Vec<C64>,
    off: Vec<f64>,
    rho: Vec<f64>,
    dr0: f64,
    ycf: f64,
    b0c: f64,
    shifts: Vec<f64>,
    live: Vec<bool>,
    b: Vec<Mat>,
    z: Vec<Mat>,
    bu: Mat,
    ztot: Mat,
    zx: Mat,
    w: Mat,
    est: Mat,
    xc: Mat,
    yc: Mat,
    ydc: Mat,
    coupling: Mat,
    back: Mat,
    b0: Mat,
    r0b: Mat,
    tc: Mat,
    t: Mat,
    y: Mat,
    dkdd: Mat,
    sizes: Vec<f64>,
    err: Vec<f64>,
    norms: Vec<f64>,
    column: Column,
    reading: f64,
}

/// The buffers of the exact copies' column solve (fold.rs `Eliminated::column`): its right-hand side r, per copy the
/// excited rows' response p_a to its coupling, the Schur complement and its pivots, the excited rows' y, the copies' x,
/// the column.
struct Column {
    r: Vec<C64>,
    p: Mat,
    pv: Vec<C64>,
    schur: Vec<C64>,
    pivots: Vec<usize>,
    y: Vec<C64>,
    x: Vec<C64>,
    out: Vec<C64>,
}

/// The step's buffers, laid out once per fold (`Fold::workspace`): the light at the point, P's slots, every block's
/// layout, the P matrices of the passes and the compression, and the step's result.
pub(crate) struct Workspace {
    amplitudes: Vec<Amplitude>,
    fields: Vec<[C64; 3]>,
    excited: Mat,
    ground: Mat,
    e: Vec<Vec<C64>>,
    f: Vec<Vec<C64>>,
    doppler: Vec<Vec<f64>>,
    pp: Vec<Slot>,
    blocks: Vec<Lay>,
    hpp: Mat,
    h: Mat,
    hs: [Mat; 3],
    bloch: Mat,
    beams: Mat,
    tri: Mat,
    u: Mat,
    q: Vec<C64>,
    v: Vec<C64>,
    sums: Vec<C64>,
    rhs: Vec<C64>,
    eigen: Mat,
    vectors: Mat,
    temp: Mat,
    whole: Vec<f64>,
    renew: Vec<(Vec<Row>, Option<usize>)>,
    step: Step,
}

/// A drive at the point, Σ_k e_k U_k[x, t] of component g.
fn drive(e: &[Vec<C64>], components: &[Component], g: usize, x: usize, t: usize) -> C64 {
    components[g].basis.iter().zip(&e[g]).map(|(u, ek)| *ek * u[(x, t)]).sum()
}

fn fill(slots: &mut [Slot], excited: &Mat, ground: &Mat, e: &[Vec<C64>], components: &[Component], work: &mut f64) {
    for s in slots.iter_mut() {
        let v = match s.what {
            Entry::Excited(x, y) => excited[(x, y)],
            Entry::Ground(a, b) => ground[(a, b)],
            Entry::Drive(g, x, t) => {
                *work += e[g].len() as f64;
                drive(e, components, g, x, t)
            }
            Entry::Back(g, x, t) => {
                *work += e[g].len() as f64;
                drive(e, components, g, x, t).conj()
            }
        };
        s.value = v - C64::real(s.less);
    }
}

/// V†x of the solved rows' columns `cols` (x: solved rows × any), part by part, into `out` (those columns).
fn to_cols(f: &Factored, x: &Mat, cols: &[usize], out: &mut Mat, work: &mut f64) {
    for (idx, at, u) in &f.parts {
        for b in 0..idx.len() {
            for &j in cols {
                let mut acc = C64::ZERO;
                for (a, &i) in idx.iter().enumerate() {
                    acc += u[(a, b)].conj() * x[(i, j)];
                }
                out[(at + b, j)] = acc;
            }
        }
        *work += (idx.len() * idx.len() * cols.len()) as f64;
    }
}

/// V z (z: basis rows × n) into `out`'s first rows (the solved rows'), part by part.
fn from_rows(f: &Factored, z: &Mat, out: &mut Mat, work: &mut f64) {
    let n = z.cols;
    for (idx, at, u) in &f.parts {
        for (a, &i) in idx.iter().enumerate() {
            for j in 0..n {
                let mut acc = C64::ZERO;
                for b in 0..idx.len() {
                    acc += u[(a, b)] * z[(at + b, j)];
                }
                out[(i, j)] = acc;
            }
        }
        *work += (idx.len() * idx.len() * n) as f64;
    }
}

/// out = a b (or out += a b).
fn product(out: &mut Mat, a: &Mat, b: &Mat, add: bool, work: &mut f64) {
    if !add {
        out.data.fill(C64::ZERO);
    }
    for i in 0..a.rows {
        for k in 0..a.cols {
            let x = a[(i, k)];
            if x == C64::ZERO {
                continue;
            }
            for j in 0..b.cols {
                out.data[i * out.cols + j] += x * b.data[k * b.cols + j];
            }
        }
    }
    *work += (a.rows * a.cols * b.cols) as f64;
}

/// out = a b† (a: r × k, b: c × k).
fn product_adj(out: &mut Mat, a: &Mat, b: &Mat, work: &mut f64) {
    for i in 0..a.rows {
        for j in 0..b.rows {
            let mut acc = C64::ZERO;
            for k in 0..a.cols {
                acc += a[(i, k)] * b[(j, k)].conj();
            }
            out[(i, j)] = acc;
        }
    }
    *work += (a.rows * a.cols * b.rows) as f64;
}

/// Z with Z T_H − (T_K + w) Z = B (fold.rs `sylvester`) into z; T_H's entries above the diagonal that are zero cost
/// nothing (a diagonal generator: a division a column).
#[allow(clippy::too_many_arguments)]
fn sylvester_into(tk: &Mat, diag: &[C64], ends: &[usize], th: &Mat, w: f64, b: &Mat, z: &mut Mat, rhs: &mut [C64], work: &mut f64) {
    let (m, n) = (b.rows, b.cols);
    for j in 0..n {
        for k in 0..m {
            rhs[k] = b[(k, j)];
        }
        for i in 0..j {
            let t = th[(i, j)];
            if t != C64::ZERO {
                for k in 0..m {
                    rhs[k] -= z[(k, i)] * t;
                }
                *work += m as f64;
            }
        }
        let c = th[(j, j)] - C64::real(w);
        for k in (0..m).rev() {
            let mut acc = rhs[k];
            for l in k + 1..ends[k] {
                acc += tk[(k, l)] * z[(l, j)];
            }
            z[(k, j)] = if acc == C64::ZERO { C64::ZERO } else { acc / (c - diag[k]) };
            *work += (ends[k] - k) as f64;
        }
    }
}

/// One column of the block's solve with exact copies (fold.rs `Eliminated::column`): (c − K)z = r, the excited rows first
/// in the basis, then the copies' Schur complement, into `col.out`.
fn column(f: &Factored, kappa: &[C64], ne: usize, nd: usize, coupling: &Mat, back: &Mat, c: C64, r: &[C64], col: &mut Column, work: &mut f64) -> bool {
    let (t, ends) = (&f.t, &f.ends);
    let tri = |src: &dyn Fn(usize) -> C64, z: &mut [C64], work: &mut f64| {
        for k in (0..ne).rev() {
            let mut acc = src(k);
            for l in k + 1..ends[k].min(ne) {
                acc += t[(k, l)] * z[l];
            }
            z[k] = if acc == C64::ZERO { C64::ZERO } else { acc / (c - kappa[k]) };
            *work += (ends[k].min(ne) - k) as f64;
        }
    };
    tri(&|k| r[k], &mut col.y, work);
    for a in 0..nd {
        tri(&|k| coupling[(k, a)], &mut col.pv, work);
        for k in 0..ne {
            col.p[(a, k)] = col.pv[k];
        }
    }
    for a in 0..nd {
        for d in 0..nd {
            let (i, j) = (ne + a, ne + d);
            let own = if a == d { c - kappa[i] } else { -t[(i, j)] };
            col.schur[a * nd + d] = own - (0..ne).map(|k| back[(a, k)] * col.p[(d, k)]).sum::<C64>();
        }
        col.x[a] = r[ne + a] + (0..ne).map(|k| back[(a, k)] * col.y[k]).sum::<C64>();
    }
    *work += (nd * nd * ne + nd * ne) as f64;
    if !crate::linalg::factor(&mut col.schur, &mut col.pivots, nd) {
        return false;
    }
    crate::linalg::solve(&col.schur, &col.pivots, nd, &mut col.x, 1);
    *work += (nd * nd * nd) as f64 / 3.0 + (nd * nd) as f64;
    for k in 0..ne {
        col.out[k] = col.y[k] + (0..nd).map(|a| col.p[(a, k)] * col.x[a]).sum::<C64>();
    }
    for a in 0..nd {
        col.out[ne + a] = col.x[a];
    }
    *work += (nd * ne) as f64;
    true
}

impl Fold {
    /// The step's buffers for this fold (module documentation): every block's slots and sizes, laid out once.
    pub(crate) fn workspace(&self) -> Workspace {
        let n = self.size();
        let ng = self.ground.len();
        let mut formed = vec![false; STATES * STATES];
        for list in &self.entries {
            for (_, _, entries) in list {
                for &(i, j, _) in entries {
                    formed[i * STATES + j] = true;
                }
            }
        }
        let excited_nz = |x: usize, y: usize| self.base_excited[(x, y)] != C64::ZERO || formed[(GROUND + x) * STATES + GROUND + y];
        let ground_nz = |a: usize, b: usize| self.base_ground[(a, b)] != C64::ZERO || formed[a * STATES + b];
        let couples = |g: usize, x: usize, t: usize| {
            let c = &self.components[g];
            c.basis.iter().any(|u| u[(x, t)] != C64::ZERO) || c.couplings.iter().any(|u| u[(x, t)] != C64::ZERO)
        };
        let slot = |row: usize, col: usize, what: Entry, less: f64| Slot { row, col, what, less, value: C64::ZERO };
        let lines = self.part_of.iter().copied().max().map_or(0, |x| x + 1);
        let mut pp: Vec<Slot> = vec![];
        for p in 0..ng {
            for q in (0..ng).filter(|&q| self.frames[q] == self.frames[p]) {
                if p == q || ground_nz(self.ground[p], self.ground[q]) {
                    pp.push(slot(p, q, Entry::Ground(self.ground[p], self.ground[q]), if p == q { self.frame_values[p] } else { 0.0 }));
                }
            }
        }
        let mut blocks = vec![];
        for b in &self.blocks {
            let m = b.rows.len();
            let s = b.solved;
            let ne = b.rows[..s].iter().filter(|r| matches!(r, Row::Excited(_))).count();
            let (nd, nc) = (s - ne, m - s);
            let mut inner = vec![];
            for (i, ri) in b.rows.iter().enumerate() {
                for (j, rj) in b.rows.iter().enumerate() {
                    match (*ri, *rj) {
                        (Row::Excited(x), Row::Excited(y)) if i == j || excited_nz(x, y) => inner.push(slot(i, j, Entry::Excited(x, y), if i == j { b.value } else { 0.0 })),
                        (Row::Excited(x), Row::Copy { state, component }) if couples(component, x, state) => inner.push(slot(i, j, Entry::Drive(component, x, state), 0.0)),
                        (Row::Copy { state, component }, Row::Excited(y)) if couples(component, y, state) => inner.push(slot(i, j, Entry::Back(component, y, state), 0.0)),
                        (Row::Copy { state: a, component: g }, Row::Copy { state: c, component: h }) if g == h && (i == j || ground_nz(a, c)) => {
                            inner.push(slot(i, j, Entry::Ground(a, c), if i == j { b.value - self.components[g].nu } else { 0.0 }))
                        }
                        _ => {}
                    }
                }
            }
            let mut absorb = vec![];
            let mut emit = vec![];
            let at_copy = |component: usize, p: usize| self.frames[p] == b.offset.plus(-1, &self.components[component].beat);
            for (i, r) in b.rows.iter().enumerate() {
                for &p in &b.explicit {
                    let x = self.explicit[p - ng];
                    match *r {
                        Row::Excited(y) => {
                            if excited_nz(y, x) {
                                absorb.push(slot(i, p, Entry::Excited(y, x), 0.0));
                            }
                            if excited_nz(x, y) {
                                emit.push(slot(p, i, Entry::Excited(x, y), 0.0));
                            }
                        }
                        Row::Copy { state, component } => {
                            if couples(component, x, state) {
                                absorb.push(slot(i, p, Entry::Back(component, x, state), 0.0));
                                emit.push(slot(p, i, Entry::Drive(component, x, state), 0.0));
                            }
                        }
                    }
                }
                if let Row::Copy { state, component } = *r {
                    for p in (0..ng).filter(|&p| at_copy(component, p)) {
                        if ground_nz(state, self.ground[p]) {
                            absorb.push(slot(i, p, Entry::Ground(state, self.ground[p]), 0.0));
                        }
                        if ground_nz(self.ground[p], state) {
                            emit.push(slot(p, i, Entry::Ground(self.ground[p], state), 0.0));
                        }
                    }
                }
            }
            for &p in &b.explicit {
                let x = self.explicit[p - ng];
                for &q in &b.explicit {
                    let y = self.explicit[q - ng];
                    if p == q || excited_nz(x, y) {
                        pp.push(slot(p, q, Entry::Excited(x, y), if p == q { b.value } else { 0.0 }));
                    }
                }
                for q in 0..ng {
                    for (g, comp) in self.components.iter().enumerate() {
                        if self.frames[q].plus(1, &comp.beat) == b.offset && couples(g, x, self.ground[q]) {
                            pp.push(slot(p, q, Entry::Drive(g, x, self.ground[q]), 0.0));
                            pp.push(slot(q, p, Entry::Back(g, x, self.ground[q]), 0.0));
                        }
                    }
                }
            }
            let mut centre: Vec<(bool, usize, f64)> = b.sources.iter().map(|&(p, _)| (false, self.ground[p], self.frame_values[p])).collect();
            centre.extend(b.explicit.iter().map(|&p| (true, self.explicit[p - ng], b.value)));
            let mut splits = vec![None];
            let mut places = vec![];
            for (c, comp) in self.components.iter().enumerate() {
                let here: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
                if here.is_empty() {
                    continue;
                }
                splits.extend((0..comp.beams.len()).map(|j| Some((c, j))));
                places.push((c, here));
            }
            let mut sourced: Vec<usize> = places.iter().flat_map(|x| x.1.iter().copied()).collect();
            sourced.sort();
            sourced.dedup();
            let mut statics: Vec<usize> = absorb.iter().map(|x: &Slot| x.col).collect();
            statics.sort();
            statics.dedup();
            let mut emitting: Vec<usize> = emit.iter().filter(|x: &&Slot| x.col < s).map(|x| x.row).collect();
            emitting.sort();
            emitting.dedup();
            let group: Vec<usize> = b.rows[..s].iter().map(|r| if let Row::Excited(y) = r { self.part_of[*y] } else { lines }).collect();
            let groups = lines + 1;
            let z = |r: usize, c: usize| Mat::zeros(r, c);
            blocks.push(Lay {
                m,
                s,
                ne,
                nd,
                nc,
                inner,
                absorb,
                emit,
                centre,
                live: vec![false; splits.len()],
                shifts: vec![0.0; splits.len()],
                b: (0..splits.len()).map(|_| z(s, n)).collect(),
                z: (0..splits.len()).map(|_| z(s, n)).collect(),
                splits,
                places,
                sourced,
                statics,
                emitting,
                group,
                groups,
                basis: 0,
                kappa: vec![C64::ZERO; s],
                coefficients: vec![],
                off: vec![0.0; s * groups],
                rho: vec![0.0; s],
                dr0: 0.0,
                ycf: 0.0,
                b0c: 0.0,
                bu: z(s, n),
                ztot: z(s, n),
                zx: z(s, n),
                w: z(s, n),
                est: z(n, s),
                xc: z(s, nc),
                yc: z(s, nd + nc),
                ydc: z(nd, nc),
                coupling: z(ne, nd),
                back: z(nd, ne),
                b0: z(m, n),
                r0b: z(nc, n),
                tc: z(nc, n),
                t: z(m, n),
                y: z(m, n),
                dkdd: z(nd, nd),
                sizes: vec![0.0; n],
                err: vec![0.0; s * n],
                norms: vec![0.0; groups],
                column: Column { r: vec![C64::ZERO; s], p: z(nd, ne), pv: vec![C64::ZERO; ne], schur: vec![C64::ZERO; nd * nd], pivots: vec![0; nd], y: vec![C64::ZERO; ne], x: vec![C64::ZERO; nd], out: vec![C64::ZERO; s] },
                reading: 0.0,
            });
        }
        let z = |r: usize, c: usize| Mat::zeros(r, c);
        Workspace {
            amplitudes: vec![Amplitude { value: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] }; self.light.beams.len()],
            fields: vec![[C64::ZERO; 3]; self.light.frequencies.len()],
            excited: z(EXCITED, EXCITED),
            ground: z(GROUND, GROUND),
            e: self.components.iter().map(|c| vec![C64::ZERO; c.basis.len()]).collect(),
            f: self.components.iter().map(|c| vec![C64::ZERO; c.beams.len()]).collect(),
            doppler: self.components.iter().map(|c| vec![0.0; c.beams.len()]).collect(),
            pp,
            blocks,
            hpp: z(n, n),
            h: z(n, n),
            hs: [z(n, n), z(n, n), z(n, n)],
            bloch: z(n, n),
            beams: z(n, n),
            tri: z(n, n),
            u: z(n, n),
            q: vec![C64::ZERO; n * n],
            v: vec![C64::ZERO; n],
            sums: vec![C64::ZERO; n],
            rhs: vec![C64::ZERO; self.blocks.iter().map(|b| b.solved).max().unwrap_or(0)],
            eigen: z(n, n),
            vectors: z(n, n),
            temp: z(n, n),
            whole: vec![0.0; n],
            renew: Vec::with_capacity(self.blocks.len() * 3),
            step: Step { a: z(n, n), metric: z(n, n), normal: z(n, n), generator: z(n, n), passes: 0, ratio: 0.0, reuse: vec![0.0; self.blocks.len()], work: 0.0, phases: [("place", 0.0), ("select", 0.0), ("prepare", 0.0), ("passes", 0.0), ("reading", 0.0), ("compression", 0.0)] },
        }
    }

    /// The light at the point (module documentation, 1), every slot filled.
    fn place(&self, r: [f64; 3], kappa: [f64; 3], ws: &mut Workspace, work: &mut f64) {
        for (a, b) in ws.amplitudes.iter_mut().zip(&self.light.beams) {
            *a = b.amplitude_to(r, 0);
        }
        for (f, freq) in ws.fields.iter_mut().zip(&self.light.frequencies) {
            *f = freq.field(&ws.amplitudes).value;
        }
        ws.excited.data.copy_from_slice(&self.base_excited.data);
        ws.ground.data.copy_from_slice(&self.base_ground.data);
        self.forms(&self.entries, &ws.amplitudes, &mut ws.ground, &mut ws.excited, work);
        for (c, comp) in self.components.iter().enumerate() {
            for (j, &l) in comp.beams.iter().enumerate() {
                ws.f[c][j] = comp.factor(ws.amplitudes[l].value);
                let q = &comp.kicks[j];
                ws.doppler[c][j] = self.hbar_over_mass * (q.iter().zip(&kappa).map(|(q, k)| q * k).sum::<f64>() + 0.5 * q.iter().map(|q| q * q).sum::<f64>());
            }
            for k in 0..comp.basis.len() {
                ws.e[c][k] = (0..comp.beams.len()).map(|j| ws.f[c][j] * comp.gamma[j][k]).sum();
            }
            *work += (comp.beams.len() * comp.basis.len()) as f64;
        }
        let Workspace { excited, ground, e, pp, blocks, .. } = ws;
        fill(pp, excited, ground, e, &self.components, work);
        for l in blocks.iter_mut() {
            fill(&mut l.inner, excited, ground, e, &self.components, work);
            fill(&mut l.absorb, excited, ground, e, &self.components, work);
            fill(&mut l.emit, excited, ground, e, &self.components, work);
        }
        ws.hpp.data.fill(C64::ZERO);
        for s in &ws.pp {
            ws.hpp[(s.row, s.col)] += s.value;
        }
    }

    /// A block's centre z₀ at the point (fold.rs `lay`).
    fn centre(&self, l: &Lay, ws: &Workspace) -> f64 {
        let sum: f64 = l.centre.iter().map(|&(explicit, x, less)| if explicit { ws.excited[(x, x)].re - less } else { ws.ground[(x, x)].re - less }).sum();
        sum / l.centre.len().max(1) as f64
    }

    /// A block's K at the point (x coordinates, its solved rows, the exact copies' couplings out: the kept one), its R₀,
    /// z₀ and Ĥ_CC -- where a decomposition is made or the nearest kept one looked for. None where R₀ is singular.
    fn effective_at(&self, o: usize, ws: &Workspace, work: &mut f64) -> Option<(Mat, Mat, f64, Mat)> {
        let l = &ws.blocks[o];
        let (m, s) = (l.m, l.s);
        let mut inner = Mat::zeros(m, m);
        for x in &l.inner {
            inner[(x.row, x.col)] = x.value;
        }
        let z0 = self.centre(l, ws);
        let cs: Vec<usize> = (s..m).collect();
        let ss: Vec<usize> = (0..s).collect();
        let cc = inner.block(&cs, &cs);
        let r0 = inverse(&Mat::from_fn(l.nc, l.nc, |i, j| if i == j { C64::real(z0) } else { C64::ZERO } - cc[(i, j)]), work)?;
        let mut k = inner.block(&ss, &ss);
        if l.nc > 0 {
            let rc = mul(&inner.block(&ss, &cs), &r0, work);
            plus(&mut k, &mul(&rc, &inner.block(&cs, &ss), work));
        }
        let value = self.blocks[o].value;
        for i in 0..s {
            k[(i, i)] += C64::real(value);
            for j in 0..s {
                if (i < l.ne) != (j < l.ne) {
                    k[(i, j)] = C64::ZERO;
                }
            }
        }
        Some((k, r0, z0, cc))
    }

    /// The basis a block stands on at the point (fold.rs `Fold::eliminate`, the same rule): the atom's own; beyond the share
    /// (`renew`) the nearest kept one of its key, then a new one.
    fn select(&self, o: usize, cache: &mut Cache, ws: &mut Workspace, work: &mut f64) -> Option<()> {
        let b = &self.blocks[o];
        let rows = &b.rows[..b.solved];
        let dressed = b.rows.len() > b.solved || rows.iter().any(|r| matches!(r, Row::Copy { .. }));
        let own = dressed.then_some(o);
        let same = |x: &(Vec<Row>, Option<usize>)| x.0 == rows && x.1 == own;
        let current = cache.current.iter().position(|x| same(&x.0));
        let tries = ws.renew.iter().filter(|x| same(x)).count();
        let standing = current.map(|c| cache.current[c].1);
        let mut here: Option<(Mat, Mat, f64, Mat)> = None;
        let nearest = |but: Option<usize>, here: &mut Option<(Mat, Mat, f64, Mat)>, work: &mut f64| -> Option<Option<usize>> {
            if here.is_none() {
                *here = Some(self.effective_at(o, ws, work)?);
            }
            let kept = &here.as_ref().unwrap().0;
            let mut best: Option<(usize, f64)> = None;
            for (i, x) in cache.bases.iter().enumerate().filter(|(i, x)| x.rows == rows && x.own == own && Some(*i) != but) {
                let mut d = kept.clone();
                d.add_scaled(-C64::ONE, &x.k);
                let size = d.frobenius();
                *work += (d.rows * d.cols) as f64;
                if best.is_none_or(|b| size < b.1) {
                    best = Some((i, size));
                }
            }
            Some(best.map(|b| b.0))
        };
        let chosen = match (standing, tries) {
            (Some(at), 0) => Some(at),
            (None, 0) => nearest(None, &mut here, work)?,
            (_, 1) => nearest(standing, &mut here, work)?,
            _ => None,
        };
        if chosen.is_some() && chosen != standing {
            cache.looked += 1;
        }
        let at = match chosen {
            Some(at) => at,
            None => {
                let (kept, r0, z0, cc) = match here {
                    Some(x) => x,
                    None => self.effective_at(o, ws, work)?,
                };
                cache.bases.push(self.make(o, ws, kept, r0, z0, cc, work)?);
                cache.made += 1;
                cache.bases.len() - 1
            }
        };
        match current {
            Some(c) => cache.current[c].1 = at,
            None => cache.current.push(((rows.to_vec(), own), at)),
        }
        if cache.bases[at].ranks[o].is_none() {
            let ranks = self.ranks(o, &cache.bases[at].factored, &cache.bases[at].r0, work);
            cache.bases[at].ranks[o] = Some(ranks);
        }
        ws.blocks[o].basis = at;
        Some(())
    }

    /// A block's sources in a basis (`Ranks`).
    fn ranks(&self, o: usize, f: &Factored, r0: &Mat, work: &mut f64) -> Ranks {
        let b = &self.blocks[o];
        let n = self.size();
        let s = b.solved;
        let ne = b.rows[..s].iter().filter(|r| matches!(r, Row::Excited(_))).count();
        let mut sources = vec![];
        for (ci, comp) in self.components.iter().enumerate() {
            let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == ci).map(|x| x.0).collect();
            if places.is_empty() {
                continue;
            }
            let per = comp
                .basis
                .iter()
                .map(|u| {
                    let mut r = Mat::zeros(s, n);
                    for &p in &places {
                        for i in 0..s {
                            if let Row::Excited(y) = b.rows[i] {
                                r[(i, p)] = u[(y, self.ground[p])];
                            }
                        }
                    }
                    f.to(&r, work)
                })
                .collect();
            sources.push((ci, per));
        }
        let mut copies = vec![];
        let mut gs: Vec<usize> = b.rows[ne..].iter().filter_map(|r| if let Row::Copy { component, .. } = r { Some(*component) } else { None }).collect();
        gs.sort();
        gs.dedup();
        let nd = s - ne;
        for g in gs {
            let (mut ys, mut xs, mut rn) = (vec![], vec![], vec![]);
            for u in &self.components[g].basis {
                let mut c = Mat::zeros(s, b.rows.len() - ne);
                for (col, r) in b.rows[ne..].iter().enumerate() {
                    if let Row::Copy { state, component } = *r {
                        if component == g {
                            for i in 0..ne {
                                if let Row::Excited(y) = b.rows[i] {
                                    c[(i, col)] = u[(y, state)];
                                }
                            }
                        }
                    }
                }
                let y = f.to(&c, work);
                let centred = y.block(&(0..s).collect::<Vec<_>>(), &(nd..y.cols).collect::<Vec<_>>());
                xs.push(mul(&centred, r0, work));
                rn.push((0..s).map(|i| (nd..y.cols).map(|j| y[(i, j)].norm_sqr()).sum::<f64>().sqrt()).collect());
                ys.push(y);
            }
            copies.push((g, ys, xs, rn));
        }
        Ranks { sources, copies }
    }

    /// A decomposition made at the point (module documentation, 2): K's Schur form with the lines' couplings out, its
    /// terms, what it leaves out between the lines there, the copies' reference.
    #[allow(clippy::too_many_arguments)]
    fn make(&self, o: usize, ws: &Workspace, kept: Mat, r0: Mat, z0: f64, cc: Mat, work: &mut f64) -> Option<Basis> {
        let b = &self.blocks[o];
        let l = &ws.blocks[o];
        let (s, ne, nd, groups) = (l.s, l.ne, l.nd, l.groups);
        let rows = b.rows[..s].to_vec();
        let mut k = kept.clone();
        for i in 0..s {
            for j in 0..s {
                if let (Row::Excited(x), Row::Excited(y)) = (rows[i], rows[j]) {
                    if self.part_of[x] != self.part_of[y] {
                        k[(i, j)] = C64::ZERO;
                    }
                }
            }
        }
        let factored = Factored::of(&k, work)?;
        let (mut group, mut part) = (vec![0; s], vec![0; s]);
        for (pi, (idx, at, _)) in factored.parts.iter().enumerate() {
            for a in 0..idx.len() {
                group[at + a] = l.group[idx[0]];
                part[at + a] = pi;
            }
        }
        let rotate = |m: &Mat, work: &mut f64| factored.to(&factored.to(m, work).adjoint(), work).adjoint();
        let norms_of = |t: &Mat, cross_only: bool| -> Vec<f64> {
            let mut out = vec![0.0; s * groups];
            for i in 0..s {
                for j in (0..s).filter(|&j| j != i) {
                    if !cross_only || group[j] != group[i] {
                        out[i * groups + group[j]] += t[(i, j)].norm_sqr();
                    }
                }
            }
            out.iter_mut().for_each(|x| *x = x.sqrt());
            out
        };
        let row_of: Vec<Option<usize>> = (0..EXCITED).map(|x| rows[..ne].iter().position(|r| *r == Row::Excited(x))).collect();
        let mut terms = vec![];
        for (w, list) in self.entries.iter().enumerate() {
            for (p, q, entries) in list {
                let mut m = Mat::zeros(s, s);
                let mut any = false;
                for &(i, j, v) in entries {
                    if i >= GROUND && j >= GROUND {
                        if let (Some(a), Some(c)) = (row_of[i - GROUND], row_of[j - GROUND]) {
                            m[(a, c)] = v;
                            any = true;
                        }
                    }
                }
                if !any {
                    continue;
                }
                let t = rotate(&m, work);
                let field = &ws.fields[w];
                terms.push(Term { what: Coefficient::Form(w, *p, *q), at: field[*p].conj() * field[*q], diag: (0..s).map(|i| t[(i, i)]).collect(), norms: norms_of(&t, false) });
            }
        }
        let ranks = self.ranks(o, &factored, &r0, work);
        for (g, ys, xs, _) in &ranks.copies {
            let centred: Vec<Mat> = ys.iter().map(|y| y.block(&(0..s).collect::<Vec<_>>(), &(nd..y.cols).collect::<Vec<_>>())).collect();
            for (a, x) in xs.iter().enumerate() {
                for (c, y) in centred.iter().enumerate() {
                    let mut t = Mat::zeros(s, s);
                    product_adj(&mut t, x, y, work);
                    terms.push(Term { what: Coefficient::Dressing(*g, a, c), at: ws.e[*g][a] * ws.e[*g][c].conj(), diag: (0..s).map(|i| t[(i, i)]).collect(), norms: norms_of(&t, false) });
                }
            }
        }
        let mut left = kept.clone();
        left.add_scaled(-C64::ONE, &k);
        let cross = norms_of(&rotate(&left, work), true);
        let ds: Vec<usize> = (ne..s).collect();
        let kdd = kept.block(&ds, &ds);
        let mut vd = Mat::zeros(nd, nd);
        for (idx, at, u) in factored.parts.iter().filter(|p| p.0[0] >= ne) {
            for (a, &i) in idx.iter().enumerate() {
                for bb in 0..idx.len() {
                    vd[(i - ne, at + bb - ne)] = u[(a, bb)];
                }
            }
        }
        let r0_norm = r0.frobenius();
        let mut ranks_all: Vec<Option<Ranks>> = (0..self.blocks.len()).map(|_| None).collect();
        ranks_all[o] = Some(ranks);
        Some(Basis { rows, own: (b.rows.len() > s || nd > 0).then_some(o), k: kept, factored, group, part, terms, cross, r0, z0, cc, r0_norm, kdd, vd, ranks: ranks_all })
    }

    /// A block's quantities on its basis at the point (module documentation, 2–4): the moved diagonal, the reading's norms,
    /// the copies' couplings, the sources in the basis, the static emission's rows.
    fn prepare(&self, o: usize, cache: &Cache, ws: &mut Workspace, work: &mut f64) {
        let n = self.size();
        let value = self.blocks[o].value;
        let z0 = {
            let l = &ws.blocks[o];
            if l.nc > 0 { self.centre(l, ws) } else { 0.0 }
        };
        let Workspace { blocks, fields, e, f, doppler, .. } = ws;
        let l = &mut blocks[o];
        let basis = &cache.bases[l.basis];
        let fa = &basis.factored;
        let ranks = basis.ranks[o].as_ref().unwrap();
        let (s, ne, nd, nc, groups) = (l.s, l.ne, l.nd, l.nc, l.groups);
        // The terms' coefficients, the diagonal moved to first order, the reading's norms.
        l.coefficients.clear();
        l.coefficients.extend(basis.terms.iter().map(|t| match t.what {
            Coefficient::Form(w, p, q) => fields[w][p].conj() * fields[w][q],
            Coefficient::Dressing(g, k, c) => e[g][k] * e[g][c].conj(),
        }));
        // A term whose coefficient has not changed adds nothing.
        for i in 0..s {
            l.kappa[i] = fa.t[(i, i)] - C64::real(value);
            let own = basis.group[i];
            for g in 0..groups {
                l.off[i * groups + g] = if g == own { fa.dropped[basis.part[i]] } else { basis.cross[i * groups + g] };
            }
        }
        for (t, c) in basis.terms.iter().zip(&l.coefficients) {
            let d = *c - t.at;
            if d == C64::ZERO {
                continue;
            }
            let delta = d.abs();
            for i in 0..s {
                l.kappa[i] += d * t.diag[i];
                for g in 0..groups {
                    l.off[i * groups + g] += t.norms[i * groups + g] * delta;
                }
            }
            *work += (s * (1 + groups)) as f64;
        }
        // The copies at the point: X_C = V†Ĥ_SC R₀ and Ỹ = V†Ĥ_S,copies, R₀'s change since the reference.
        l.xc.data.fill(C64::ZERO);
        l.yc.data.fill(C64::ZERO);
        for (g, ys, xs, _) in &ranks.copies {
            for (k, (y, x)) in ys.iter().zip(xs).enumerate() {
                let ek = e[*g][k];
                for (a, b) in l.xc.data.iter_mut().zip(&x.data) {
                    *a += ek * *b;
                }
                if nd > 0 {
                    for (a, b) in l.yc.data.iter_mut().zip(&y.data) {
                        *a += ek * *b;
                    }
                }
                *work += (x.data.len() + if nd > 0 { y.data.len() } else { 0 }) as f64;
            }
        }
        if nd > 0 {
            // The exact copies' rows to the centred ones (Ĥ_DC, H_GG), V_D†Ĥ_DC; their K's change on their own rows.
            l.ydc.data.fill(C64::ZERO);
            l.dkdd.data.fill(C64::ZERO);
            for x in l.inner.iter().filter(|x| x.row >= ne && x.row < s) {
                if x.col >= s {
                    for a in 0..nd {
                        l.ydc[(a, x.col - s)] += basis.vd[(x.row - ne, a)].conj() * x.value;
                    }
                } else if x.col >= ne {
                    l.dkdd[(x.row - ne, x.col - ne)] += x.value;
                }
            }
            *work += (nd * nd * nc) as f64;
            // X_C on the exact copies' rows: V_D†Ĥ_DC R₀.
            for a in 0..nd {
                for c in 0..nc {
                    l.xc[(ne + a, c)] = (0..nc).map(|k| l.ydc[(a, k)] * basis.r0[(k, c)]).sum();
                }
            }
            *work += (nd * nc * nc) as f64;
            // K_DD at the point (R₀ the reference's) less the kept one, in the basis: its diagonal and rows.
            let mut hdc = Mat::zeros(nd, nc);
            for x in l.inner.iter().filter(|x| x.row >= ne && x.row < s && x.col >= s) {
                hdc[(x.row - ne, x.col - s)] = x.value;
            }
            let mut hr = Mat::zeros(nd, nc);
            product(&mut hr, &hdc, &basis.r0, false, work);
            let mut dk = Mat::zeros(nd, nd);
            product_adj(&mut dk, &hr, &hdc, work);
            for i in 0..nd {
                for j in 0..nd {
                    dk[(i, j)] += l.dkdd[(i, j)] + if i == j { C64::real(value) } else { C64::ZERO } - basis.kdd[(i, j)];
                }
            }
            let mut left = Mat::zeros(nd, nd);
            for i in 0..nd {
                for j in 0..nd {
                    left[(i, j)] = (0..nd).map(|a| basis.vd[(a, i)].conj() * dk[(a, j)]).sum();
                }
            }
            for i in 0..nd {
                for j in 0..nd {
                    l.dkdd[(i, j)] = (0..nd).map(|a| left[(i, a)] * basis.vd[(a, j)]).sum();
                }
            }
            *work += (2 * nd * nd * nd) as f64;
            for i in 0..nd {
                let r = ne + i;
                l.kappa[r] += l.dkdd[(i, i)];
                let rest: f64 = (0..nd).filter(|&j| j != i).map(|j| l.dkdd[(i, j)].norm_sqr()).sum();
                l.off[r * groups + basis.group[r]] = rest.sqrt() + fa.dropped[basis.part[r]];
            }
            // The couplings of the excited rows and the exact copies in the basis (fold.rs # The exact copies):
            // Ỹ = Ỹ_C[E, D] V_D + X_C[E] (V_D†Ĥ_DC)†, Ỹ′ = (Ỹ_C[E, D] V_D)† + X_C[D] Ỹ_C[E, C]†.
            for i in 0..ne {
                for a in 0..nd {
                    let bare: C64 = (0..nd).map(|c| l.yc[(i, c)] * basis.vd[(c, a)]).sum();
                    let dressed: C64 = (0..nc).map(|c| l.xc[(i, c)] * l.ydc[(a, c)].conj()).sum();
                    l.coupling[(i, a)] = bare + dressed;
                    let back_dressed: C64 = (0..nc).map(|c| l.xc[(ne + a, c)] * l.yc[(i, nd + c)].conj()).sum();
                    l.back[(a, i)] = bare.conj() + back_dressed;
                }
            }
            *work += (ne * nd * (nd + 2 * nc)) as f64;
        }
        // R₀'s change since the reference (module documentation, 4).
        l.dr0 = 0.0;
        l.ycf = 0.0;
        l.rho.iter_mut().for_each(|x| *x = 0.0);
        if nc > 0 {
            // δM, the change of z₀ − Ĥ_CC (its entries are the C × C slots'; the diagonal's are among them).
            let mut dm = 0.0;
            for x in l.inner.iter().filter(|x| x.row >= s && x.col >= s) {
                let shift = if x.row == x.col { C64::real(z0 - basis.z0) } else { C64::ZERO };
                dm += (shift - (x.value - basis.cc[(x.row - s, x.col - s)])).norm_sqr();
            }
            let dm = dm.sqrt();
            let q = basis.r0_norm * dm;
            l.dr0 = if q < 1.0 { basis.r0_norm * basis.r0_norm * dm / (1.0 - q) } else { f64::INFINITY };
            for i in 0..s {
                let mut a = 0.0;
                for (g, _, _, rn) in &ranks.copies {
                    for (k, r) in rn.iter().enumerate() {
                        a += e[*g][k].abs() * r[i];
                    }
                }
                if i >= ne {
                    a += (0..nc).map(|c| l.ydc[(i - ne, c)].norm_sqr()).sum::<f64>().sqrt();
                }
                l.rho[i] = a;
                l.ycf += a * a;
            }
            l.ycf = l.ycf.sqrt();
            for i in 0..s {
                l.rho[i] *= if l.dr0 == 0.0 { 0.0 } else { l.dr0 * l.ycf };
            }
        }
        // The static source in x coordinates, its effective one in the basis V†B_S + X_C B_C, R₀B_C.
        l.b0.data.fill(C64::ZERO);
        for x in &l.absorb {
            l.b0[(x.row, x.col)] = x.value;
        }
        l.b0c = (s..l.m).map(|i| (0..n).map(|j| l.b0[(i, j)].norm_sqr()).sum::<f64>()).sum::<f64>().sqrt();
        {
            let b = &mut l.b[0];
            b.data.fill(C64::ZERO);
            if !l.statics.is_empty() && s > 0 {
                to_cols(fa, &l.b0, &l.statics, b, work);
                for &j in &l.statics {
                    for i in 0..s {
                        b[(i, j)] += (0..nc).map(|c| l.xc[(i, c)] * l.b0[(s + c, j)]).sum::<C64>();
                    }
                }
                *work += (s * nc * l.statics.len()) as f64;
            }
            l.live[0] = b.data.iter().any(|z| *z != C64::ZERO);
            l.r0b.data.fill(C64::ZERO);
            for &j in &l.statics {
                for a in 0..nc {
                    l.r0b[(a, j)] = (0..nc).map(|c| basis.r0[(a, c)] * l.b0[(s + c, j)]).sum();
                }
            }
            *work += (nc * nc * l.statics.len()) as f64;
        }
        // The beams' sources a_l Σ_k γ_lk Y_k, W their sum.
        l.w.data.fill(C64::ZERO);
        for sp in 1..l.splits.len() {
            let (c, j) = l.splits[sp].unwrap();
            let comp = &self.components[c];
            l.shifts[sp] = doppler[c][j];
            let per = &ranks.sources.iter().find(|x| x.0 == c).unwrap().1;
            let b = &mut l.b[sp];
            b.data.fill(C64::ZERO);
            let cols = &l.places.iter().find(|x| x.0 == c).unwrap().1;
            for (k, y) in per.iter().enumerate() {
                let x = f[c][j] * comp.gamma[j][k];
                for i in 0..s {
                    for &p in cols {
                        b[(i, p)] += x * y[(i, p)];
                    }
                }
                *work += (s * cols.len()) as f64;
            }
            for i in 0..s {
                for &p in cols {
                    l.w[(i, p)] += b[(i, p)];
                }
            }
            l.live[sp] = true;
        }
        // The static emission's rows from the solved rows in the basis, Ê = E_S V.
        l.est.data.fill(C64::ZERO);
        for x in l.emit.iter().filter(|x| x.col < s) {
            let i = x.col;
            let (pi, a) = fa.parts.iter().enumerate().find_map(|(pi, (idx, _, _))| idx.iter().position(|&r| r == i).map(|a| (pi, a))).unwrap();
            let (idx, at, u) = &fa.parts[pi];
            for bb in 0..idx.len() {
                l.est[(x.row, at + bb)] += x.value * u[(a, bb)];
            }
            *work += idx.len() as f64;
        }
    }

    /// One pass at the P generator `ws.h` (diagonal) or at the Schur form `ws.tri`, `ws.u` (module documentation, 3), and
    /// its Bloch generator into `ws.bloch`, the beams' emission W†Z into `ws.beams`. False where an exact copies' Schur
    /// complement is singular.
    fn pass(&self, cache: &Cache, ws: &mut Workspace, diagonal: bool, work: &mut f64) -> bool {
        let n = self.size();
        let Workspace { blocks, h, tri, u, rhs, bloch, beams, hpp, .. } = ws;
        let th: &Mat = if diagonal { h } else { tri };
        for l in blocks.iter_mut() {
            let basis = &cache.bases[l.basis];
            let fa = &basis.factored;
            let (s, ne, nd) = (l.s, l.ne, l.nd);
            l.ztot.data.fill(C64::ZERO);
            for sp in 0..l.splits.len() {
                if !l.live[sp] || s == 0 {
                    continue;
                }
                let w = l.shifts[sp];
                if !diagonal {
                    product(&mut l.bu, &l.b[sp], u, false, work);
                }
                let src = if diagonal { &l.b[sp] } else { &l.bu };
                let z = &mut l.z[sp];
                if nd == 0 {
                    sylvester_into(&fa.t, &l.kappa, &fa.ends, th, w, src, z, rhs, work);
                } else {
                    for j in 0..n {
                        let r = &mut l.column.r;
                        for k in 0..s {
                            r[k] = src[(k, j)];
                        }
                        for i in 0..j {
                            let t = th[(i, j)];
                            if t != C64::ZERO {
                                for k in 0..s {
                                    r[k] -= z[(k, i)] * t;
                                }
                                *work += s as f64;
                            }
                        }
                        let r = std::mem::take(&mut l.column.r);
                        let fine = column(fa, &l.kappa, ne, nd, &l.coupling, &l.back, th[(j, j)] - C64::real(w), &r, &mut l.column, work);
                        l.column.r = r;
                        if !fine {
                            return false;
                        }
                        for k in 0..s {
                            z[(k, j)] = l.column.out[k];
                        }
                    }
                }
                for (a, b) in l.ztot.data.iter_mut().zip(&z.data) {
                    *a += *b;
                }
            }
            if diagonal {
                l.zx.data.copy_from_slice(&l.ztot.data);
            } else {
                product_adj(&mut l.zx, &l.ztot, u, work);
            }
            // T_C = X_C†Z + R₀B_C.
            for a in 0..l.nc {
                for j in 0..n {
                    l.tc[(a, j)] = l.r0b[(a, j)] + (0..s).map(|i| l.xc[(i, a)].conj() * l.zx[(i, j)]).sum::<C64>();
                }
            }
            *work += (l.nc * s * n) as f64;
            for j in 0..n {
                l.sizes[j] = (0..s).map(|k| l.ztot[(k, j)].norm_sqr()).sum();
            }
        }
        bloch.data.copy_from_slice(&hpp.data);
        beams.data.fill(C64::ZERO);
        for l in blocks.iter() {
            let s = l.s;
            for &p in &l.sourced {
                for j in 0..n {
                    beams[(p, j)] += (0..s).map(|i| l.w[(i, p)].conj() * l.zx[(i, j)]).sum::<C64>();
                }
            }
            for &p in &l.emitting {
                for j in 0..n {
                    bloch[(p, j)] += (0..s).map(|i| l.est[(p, i)] * l.zx[(i, j)]).sum::<C64>();
                }
            }
            for x in l.emit.iter().filter(|x| x.col >= s) {
                for j in 0..n {
                    bloch[(x.row, j)] += x.value * l.tc[(x.col - s, j)];
                }
            }
            *work += ((l.sourced.len() + l.emitting.len()) * s * n + l.emit.len() * n) as f64;
        }
        for (a, b) in bloch.data.iter_mut().zip(&beams.data) {
            *a += *b;
        }
        true
    }

    /// The next pass's reading after a diagonal pass (fold.rs `Stage::next`).
    fn next(&self, ws: &Workspace, work: &mut f64) -> f64 {
        let n = self.size();
        if ws.blocks.iter().any(|l| l.sizes.iter().any(|x| !x.is_finite())) {
            return f64::INFINITY;
        }
        let whole: Vec<f64> = (0..n).map(|j| ws.blocks.iter().map(|l| l.sizes[j]).sum::<f64>()).collect();
        let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
        let mut worst = 0.0_f64;
        for j in 0..n {
            if whole[j] <= floor {
                continue;
            }
            let hj = ws.h[(j, j)];
            let mut err = 0.0;
            for l in &ws.blocks {
                let gap = l.kappa.iter().map(|d| (hj - *d).abs()).fold(f64::INFINITY, f64::min);
                let x = match (0..n).map(|i| (ws.bloch[(i, j)] - if i == j { hj } else { C64::ZERO }).abs() * l.sizes[i].sqrt()).sum::<f64>() {
                    0.0 => 0.0,
                    x => x / gap,
                };
                err += x * x;
                *work += (n + l.s) as f64;
            }
            worst = worst.max(rate((err / whole[j]).sqrt()));
        }
        worst
    }

    /// Per block the reuse's reading on the last pass (module documentation, 4), into `l.reading`.
    fn reuse_reading(&self, ws: &mut Workspace, diagonal: bool, work: &mut f64) {
        let n = self.size();
        let Workspace { blocks, h, tri, whole, .. } = ws;
        let th: &Mat = if diagonal { h } else { tri };
        for j in 0..n {
            whole[j] = blocks.iter().map(|l| l.sizes[j]).sum();
        }
        let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
        for l in blocks.iter_mut() {
            let (s, groups) = (l.s, l.groups);
            l.err.iter_mut().for_each(|x| *x = 0.0);
            for sp in (0..l.splits.len()).filter(|&sp| l.live[sp] && s > 0) {
                let w = l.shifts[sp];
                let z = &l.z[sp];
                for j in 0..n {
                    let c = th[(j, j)] - C64::real(w);
                    l.norms.iter_mut().for_each(|x| *x = 0.0);
                    for k in 0..s {
                        l.norms[l.group[k]] += z[(k, j)].norm_sqr();
                    }
                    let total: f64 = l.norms.iter().sum();
                    for i in 0..s {
                        let own = l.group[i];
                        let mut x = l.rho[i] * total.sqrt();
                        for g in 0..groups {
                            let size = if g == own { (l.norms[g] - z[(i, j)].norm_sqr()).max(0.0) } else { l.norms[g] };
                            x += l.off[i * groups + g] * size.sqrt();
                        }
                        l.err[i * n + j] += if x == 0.0 { 0.0 } else { x / (c - l.kappa[i]).abs() };
                    }
                    *work += (s * (2 + groups)) as f64;
                }
            }
            let mut worst = 0.0_f64;
            for j in (0..n).filter(|&j| whole[j] > floor) {
                let e2: f64 = (0..s).map(|i| l.err[i * n + j].powi(2)).sum();
                let copies = if l.dr0 == 0.0 { 0.0 } else { l.dr0 * (l.ycf * l.sizes[j].sqrt() + l.b0c) };
                worst = worst.max(rate((e2 / whole[j] + copies * copies).sqrt()));
            }
            l.reading = worst;
        }
    }

    /// The step at a point (r, κ) on the fold's shared decompositions (module documentation): A, M, S⁻¹, G in `ws`. None
    /// where a Schur form does not converge, a centring is singular or an exact copies' Schur complement is.
    pub(crate) fn step<'w>(&self, r: [f64; 3], kappa: [f64; 3], cache: &mut Cache, ws: &'w mut Workspace) -> Option<&'w Step> {
        let n = self.size();
        let mut work = 0.0;
        let mut phases = [0.0; 6];
        let mark = |k: usize, work: f64, phases: &mut [f64; 6]| {
            let before: f64 = phases.iter().sum();
            phases[k] += work - before;
        };
        self.place(r, kappa, ws, &mut work);
        mark(0, work, &mut phases);
        let share = super::super::elim::share();
        ws.renew.clear();
        let mut round = 0;
        let passes = loop {
            let made = cache.made;
            for o in 0..self.blocks.len() {
                self.select(o, cache, ws, &mut work)?;
            }
            mark(1, work, &mut phases);
            for o in 0..self.blocks.len() {
                self.prepare(o, cache, ws, &mut work);
            }
            mark(2, work, &mut phases);
            // diag(Ĥ_PP), then the diagonal of its Bloch generator; the Bloch generator itself where the next one reads beyond
            // the share.
            ws.h.data.fill(C64::ZERO);
            (0..n).for_each(|i| ws.h[(i, i)] = ws.hpp[(i, i)]);
            ws.hs[0].data.copy_from_slice(&ws.h.data);
            if !self.pass(cache, ws, true, &mut work) {
                return None;
            }
            ws.h.data.fill(C64::ZERO);
            (0..n).for_each(|i| ws.h[(i, i)] = ws.bloch[(i, i)]);
            ws.hs[1].data.copy_from_slice(&ws.h.data);
            if !self.pass(cache, ws, true, &mut work) {
                return None;
            }
            let next = self.next(ws, &mut work);
            let passes = if next <= share {
                2
            } else {
                ws.hs[2].data.copy_from_slice(&ws.bloch.data);
                ws.tri.data.copy_from_slice(&ws.bloch.data);
                let (w, _) = crate::linalg::schur(&mut ws.tri.data, &mut ws.q, &mut ws.v, &mut ws.sums, n)?;
                work += w as f64;
                for i in 0..n {
                    for j in 0..i {
                        ws.tri[(i, j)] = C64::ZERO;
                    }
                    for k in 0..n {
                        ws.u[(i, k)] = ws.q[k * n + i];
                    }
                }
                if !self.pass(cache, ws, false, &mut work) {
                    return None;
                }
                3
            };
            mark(3, work, &mut phases);
            self.reuse_reading(ws, passes == 2, &mut work);
            mark(4, work, &mut phases);
            let mut beyond = 0;
            for o in 0..self.blocks.len() {
                if ws.blocks[o].reading > share {
                    let x = &cache.bases[ws.blocks[o].basis];
                    let key = (x.rows.clone(), x.own);
                    if !ws.renew[ws.renew.len() - beyond..].contains(&key) {
                        ws.renew.push(key);
                        beyond += 1;
                    }
                }
            }
            if beyond == 0 || round == 2 || cache.made > made {
                break passes;
            }
            round += 1;
        };
        // The passes' ratio (the self-consistency's existence, ≥ 1 none): of the last two changes.
        let difference = |a: &Mat, b: &Mat| a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>().sqrt();
        let ratio = difference(&ws.bloch, &ws.hs[passes - 1]) / difference(&ws.hs[passes - 1], &ws.hs[passes - 2]);
        // The compression: T_S = VZ, A = Bloch + (W†Z)† + T†(ĤT + B), M = T†T.
        let Workspace { blocks, bloch, beams, step, .. } = ws;
        let (a, metric) = (&mut step.a, &mut step.metric);
        for i in 0..n {
            for j in 0..n {
                a[(i, j)] = bloch[(i, j)] + beams[(j, i)].conj();
            }
        }
        metric.data.fill(C64::ZERO);
        for l in blocks.iter_mut() {
            let basis = &cache.bases[l.basis];
            from_rows(&basis.factored, &l.zx, &mut l.t, &mut work);
            for c in 0..l.nc {
                for j in 0..n {
                    l.t[(l.s + c, j)] = l.tc[(c, j)];
                }
            }
            l.y.data.copy_from_slice(&l.b0.data);
            for x in &l.inner {
                for j in 0..n {
                    l.y.data[x.row * n + j] += x.value * l.t.data[x.col * n + j];
                }
            }
            work += (l.inner.len() * n) as f64;
            for k in 0..l.m {
                for i in 0..n {
                    let ti = l.t[(k, i)].conj();
                    if ti == C64::ZERO {
                        continue;
                    }
                    for j in 0..n {
                        a.data[i * n + j] += ti * l.y.data[k * n + j];
                        metric.data[i * n + j] += ti * l.t.data[k * n + j];
                    }
                }
            }
            work += (2 * l.m * n * n) as f64;
        }
        // S⁻¹ = (1 + X)^{−1/2}, G = S⁻¹AS⁻¹.
        let Workspace { eigen, vectors, temp, step, .. } = ws;
        eigen.data.copy_from_slice(&step.metric.data);
        crate::linalg::hermitian_in_place(&mut eigen.data, &mut vectors.data, n);
        work += 10.0 * (n * n * n) as f64;
        for i in 0..n {
            for j in 0..n {
                step.normal[(i, j)] = (0..n).map(|k| vectors[(i, k)] * vectors[(j, k)].conj() / (1.0 + eigen[(k, k)].re).sqrt()).sum();
            }
        }
        work += (n * n * n) as f64;
        product(temp, &step.normal, &step.a, false, &mut work);
        product(&mut step.generator, temp, &step.normal, false, &mut work);
        mark(5, work, &mut phases);
        step.passes = passes;
        step.ratio = ratio;
        for (o, l) in ws.blocks.iter().enumerate() {
            step.reuse[o] = l.reading;
        }
        step.work = work;
        for (p, x) in step.phases.iter_mut().zip(phases) {
            p.1 = x;
        }
        Some(&ws.step)
    }
}

impl Workspace {
    /// The step's response of block o in x coordinates (its Q rows × n): the last step's T.
    pub(crate) fn response(&self, o: usize) -> &Mat {
        &self.blocks[o].t
    }
}

#[cfg(test)]
impl Step {
    /// Its A, metric, S⁻¹, G and passes, owned (the tests hold them past the workspace's next use).
    pub(crate) fn clone_parts(&self) -> (Mat, Mat, Mat, Mat, usize) {
        (self.a.clone(), self.metric.clone(), self.normal.clone(), self.generator.clone(), self.passes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reuse (module documentation, 2 and 4): a decomposition made at one point of R4_LIN_4 serves a point 40 nm off
    /// with its diagonal moved term by term; against the exact first order there, diag(V†(K − K_basis)V) with K at the
    /// point itself (R₀ there too), each row's moved diagonal is the exact one to the rounding but for R₀'s own change, and
    /// each row's rest on each group, ‖(V†(K − K_basis)V)[i, L∖i]‖, is within the reading's norm of it, off_iL + ρ_i. By
    /// the rule (every copy centred) and with every copy exact.
    #[test]
    fn the_reused_diagonal_is_the_change_s_first_order_and_the_reading_bounds_the_rest() {
        let req = crate::model::testing::scene("R4_LIN_4");
        let share = crate::model::elim::share();
        let rule = Choice::of(&req, share).unwrap();
        let light = Light::new(&req);
        let axes = Axes::new(&req).unwrap();
        let parts: Vec<Part> = components(&req, &light, &axes).unwrap().iter().map(|c| c.part).collect();
        let mut exact = vec![];
        for &a in &parts {
            for s in 0..GROUND {
                for &e in &parts {
                    for f in [1, 2] {
                        exact.push(((a, s, e, f), Status::Decomposed));
                    }
                }
            }
        }
        let (mut worst_diag, mut worst_rest, mut checked) = (0.0_f64, 0.0_f64, 0);
        for (name, choice) in [("by the rule", rule.clone()), ("every copy exact", Choice { copies: exact, ..rule.clone() })] {
            for set in Fold::sets(&req, &choice).unwrap() {
                let fold = Fold::new(&req, &choice, set[0]).unwrap();
                let (r0, r1) = ([0.1, -0.2, 0.3], [0.1 + 0.04 / 3f64.sqrt(), -0.2 + 0.04 / 3f64.sqrt(), 0.3 + 0.04 / 3f64.sqrt()]);
                let mut cache = Cache::default();
                let mut ws = fold.workspace();
                let mut work = 0.0;
                fold.place(r0, [0.0; 3], &mut ws, &mut work);
                for o in 0..fold.blocks.len() {
                    fold.select(o, &mut cache, &mut ws, &mut work).unwrap();
                }
                fold.place(r1, [0.0; 3], &mut ws, &mut work);
                for o in 0..fold.blocks.len() {
                    fold.select(o, &mut cache, &mut ws, &mut work).unwrap();
                    fold.prepare(o, &cache, &mut ws, &mut work);
                    let l = &ws.blocks[o];
                    if l.s == 0 {
                        continue;
                    }
                    let basis = &cache.bases[l.basis];
                    let f = &basis.factored;
                    let (kept, ..) = fold.effective_at(o, &ws, &mut work).unwrap();
                    let mut delta = kept.clone();
                    delta.add_scaled(-C64::ONE, &basis.k);
                    let tilde = f.to(&f.to(&delta, &mut work).adjoint(), &mut work).adjoint();
                    // The rounding's scale: K's own (the change is a difference of two K's).
                    let scale = kept.frobenius().max(basis.k.frobenius());
                    let value = fold.blocks[o].value;
                    for i in 0..l.s {
                        let moved = l.kappa[i] - (f.t[(i, i)] - C64::real(value));
                        let gap = (moved - tilde[(i, i)]).abs();
                        let allowed = l.rho[i] + 1e-14 * scale;
                        worst_diag = worst_diag.max(gap / allowed.max(f64::MIN_POSITIVE));
                        assert!(gap <= allowed, "{name}, set {set:?}, block {o}, row {i}: the moved diagonal {gap:.3e} from the first order (allowed {allowed:.3e})");
                        for g in 0..l.groups {
                            let rest: f64 = (0..l.s).filter(|&j| j != i && basis.group[j] == g).map(|j| tilde[(i, j)].norm_sqr()).sum::<f64>().sqrt();
                            let bound = l.off[i * l.groups + g] + l.rho[i] + 1e-14 * scale;
                            worst_rest = worst_rest.max(rest / bound);
                            assert!(rest <= bound, "{name}, set {set:?}, block {o}, row {i}, group {g}: the rest {rest:.3e} beyond the reading's norm {bound:.3e}");
                        }
                        checked += 1;
                    }
                    println!("{name}, set {set:?}, block {o}: {} rows, ‖K‖ {scale:.3e}, ‖δK‖ {:.3e}, R₀'s change bound {:.3e}, terms {}", l.s, delta.frobenius(), l.dr0, basis.terms.len());
                }
            }
        }
        println!("{checked} rows: the moved diagonal against the first order, worst {worst_diag:.2e} of its allowance; the rest against the reading's norm, worst {worst_rest:.2e}");
        assert!(checked > 0);
    }
}
