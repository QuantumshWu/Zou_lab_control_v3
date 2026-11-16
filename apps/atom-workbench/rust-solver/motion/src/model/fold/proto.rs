//! The fold's step by a decomposition kept on the far coordinates, the near light in linear combinations, the energies and
//! the Doppler shifts on the diagonal and the energy's self-consistency by a series (the first road's minimal prototype,
//! BRIEF_PROTO.md; not the run's, the forward alone). The same object as `Fold::at` on the same choice of nodes, with every
//! excited row of a block that solves any row solved (a centred excited row there costs a mode, not a pass); `Fold::at`
//! is the truth of its tests. `Fold::at` and `Fold::step` are not touched.
//!
//! # The far coordinates
//! Every frequency's field at a point on its basis, ε_k = u_k†F (field.rs `Frequency::components`). The outside forms are
//! Σ_kk' ε_k*ε_k' N_kk' (N_kk' = Σ_pq u_kp* u_k'q O_pq), and a centred copy's dressing of a block's excited rows,
//! Ĥ_SC R₀ Ĥ_CS, is bilinear in its emitting component's coefficients, ε (co-rotating) or ε* (counter-rotating): terms of
//! the same pairs (`Pair`), R₀ held at the decomposition's reference. So K on a block's solved rows is, anywhere,
//! K_ref + Σ_t (c_t − c_t,ref) N_t with c_t = ε_k*ε_k' -- exactly, but for R₀'s own change (read).
//!
//! # What is kept (`Cache`, per block, by the far coordinates)
//! A block with solved rows (`Near`): the Schur form V T V† of K_ref + o part by part (unitary, fold.rs `Factored`; the
//! normal parts diagonal); per term the diagonal of V†N_tV and per row and part the norm of the rest; per beam its source in
//! the basis Ỹ_l = V†X_l; per source vector Y_a = V†U_a; per centred copy vector R₀U_a†V (the copies' rows' response);
//! ‖R₀‖. A block whose rows are all centred (`Far`): its response T = R₀B is its field coefficients times kept vectors, so
//! its compression is the pair forms U_a†R₀U_b and U_a†R₀†R₀U_b in e_a*e_b.
//!
//! # A step
//! 1. The light at the point: the beams' amplitudes, the fields' ε, the pairs' coefficients, the components' coefficients
//!    e and beam factors f_l, every beam's Doppler shift w_l, Ĥ_PP.
//! 2. The diagonal moved to first order, κ_i = T_ii − o + Σ_t Re δc_t (V†N_tV)_ii; the rest of the change is read.
//! 3. Ṽ_l = f_l Ỹ_l, beam by beam; Ṽ = Σ_a e_a Y_a.
//! 4. The far blocks' pair forms into H, A and M.
//! 5. The series: Z_l solves Z_l H − (T + w_l − o) Z_l = Ṽ_l with H = Ĥ_PP + Σ Ṽ†Z the Bloch generator. A sweep is
//!    Z_l ← (c − w_l − T)⁻¹(Ṽ_l − Z_l Δ), c the columns' energies on the diagonal (each column at its own, exact; a common centre for all columns, the
//!    brief's z₀, was measured and dropped: on an S2Q set its series took 6–11 sweeps against two, on R4 F=2 its reading
//!    fell short), Δ = H − diag(c) the
//!    series' part, H made again from the sweep's Z: each sweep adds the next order of Δ and of the self-consistency at
//!    once. Its change, column by column over every row of every block (each mode with its own column: no norm of the
//!    whole), reads it: the next change is the last times their ratio, what the rest add the last times r/(1 − r). Two
//!    sweeps at the least (a ratio is read from two changes), then until the rest reads within the share.
//! 6. The readings of what is held: the reuse's (fold.rs # One decomposition along the path, row by row, on the response
//!    the step uses, Z = Σ_l Z_l, at the beams' mean Doppler shift, with the beams' spread as a remainder; R₀'s own change
//!    with it) and the far blocks' (‖R₀‖ times the distance of the energies and the forms from the reference). Beyond the
//!    share the nearest kept decomposition, then a new one (fold.rs `Path`'s rule).
//! 7. The copies' rows T_C = R₀Ĥ_CS VZ: their own part of the compression (z₀T_C†T_C, T_C†T_C; their dressing is the kept
//!    decomposition's) only where it reads beyond the share (‖T_C‖, a copy row's absolute unit). The compression
//!    A = H + Σ(Ṽ†Z)† + Z†(T + δκ − o)Z (+ z₀T_C†T_C) + z₀ M_far, M = Σ Z†Z (+ T_C†T_C) + M_far (less one);
//!    S⁻¹ = (1 + M)^{−1/2} to the rounding (`inverse_root`), G = S⁻¹AS⁻¹. Its anti-Hermitian part is −(i/2)Σε†Dε by
//!    construction: Z†TZ's is Z†V†DVZ (T the Schur form of K_ref, whose anti-Hermitian part is −D/2; V†DV is not stored
//!    apart), the far forms' T†DT (R₀ − R₀† = −iR₀†DR₀).
//!
//! The copies a block solves exactly (the P layer's Woodbury), the explicit states and the static sources are not here
//! (`Fold::prototype_workspace` refuses them): none of the prototype's two scenes has them. Their copies are the tweezer's,
//! whose dressing (the excited states' tweezer shift, ~10Γ) no series about the excited response holds: it is the kept
//! decomposition's.
use super::*;
use crate::field::Amplitude;

/// A pair of one frequency's field coefficients (frequency w, k, k'): its coefficient ε_k*ε_k'.
type Pair = (usize, usize, usize);

/// The step's phases, as BRIEF_PROTO.md splits its time.
pub(crate) const PHASES: [&str; 10] = ["place", "refresh", "Ṽ", "diagonal", "far", "series", "reading", "copies", "compression", "normal"];
const PLACE: usize = 0;
const REFRESH: usize = 1;
const SOURCES: usize = 2;
const DIAGONAL: usize = 3;
const FAR: usize = 4;
const SERIES: usize = 5;
const READING: usize = 6;
const COPIES: usize = 7;
const COMPRESSION: usize = 8;
const NORMAL: usize = 9;

/// The processor's time stamp (its ticks; the measurement converts them by the wall clock).
fn ticks() -> u64 {
    #[cfg(target_arch = "x86_64")]
    unsafe {
        core::arch::x86_64::_rdtsc()
    }
    #[cfg(not(target_arch = "x86_64"))]
    0
}

/// A block with solved rows: its layout (rows s, centred copies nc, its offset; the P places of its sources, as fold.rs
/// `lay` takes its centre; its beam splits (component, beam); its source vectors (component, k); per copy component its
/// centred rows, and its copy vectors (component, k, group); its terms) and the step's buffers.
#[derive(Clone)]
struct Near {
    o: usize,
    s: usize,
    nc: usize,
    value: f64,
    centre: Vec<usize>,
    splits: Vec<(usize, usize)>,
    vectors: Vec<(usize, usize)>,
    groups: Vec<(usize, Vec<usize>)>,
    copies: Vec<(usize, usize, usize)>,
    terms: Vec<usize>,
    // The step's.
    delta: Vec<C64>,
    kappa: Vec<C64>,
    off: Vec<f64>,
    rho: Vec<f64>,
    shifts: Vec<f64>,
    vt: Vec<C64>,
    vtot: Vec<C64>,
    z: Vec<C64>,
    ztot: Vec<C64>,
    zold: Vec<C64>,
    rhs: Vec<C64>,
    inv: Vec<C64>,
    bloch: Vec<C64>,
    tc: Vec<C64>,
    y: Vec<C64>,
    err: Vec<f64>,
    norms: Vec<f64>,
    active: Vec<bool>,
    held: Vec<C64>,
    q: Vec<f64>,
    reading: f64,
}

/// A block whose rows are all centred: its rows m, its centre's P places, its source vectors (component, k), its beam
/// splits (for the Doppler shifts its centring reads), its terms; the step's.
struct Far {
    o: usize,
    m: usize,
    centre: Vec<usize>,
    vectors: Vec<(usize, usize)>,
    splits: Vec<(usize, usize)>,
    terms: Vec<usize>,
    delta: Vec<C64>,
    products: Vec<C64>,
    hf: Vec<C64>,
    mf: Vec<C64>,
    reading: f64,
    centring: f64,
}

/// A near block's decomposition (module documentation, # What is kept).
struct NearBasis {
    at: Vec<C64>,
    weights: Vec<f64>,
    f: Factored,
    part: Vec<usize>,
    diag: Vec<C64>,
    norms: Vec<f64>,
    dressed: Vec<f64>,
    copied: Vec<f64>,
    beams: Vec<Vec<C64>>,
    vectors: Vec<Vec<C64>>,
    copies: Vec<Vec<C64>>,
    copied_rows: Vec<f64>,
    z0: f64,
    r0: f64,
}

/// A far block's pair forms (module documentation, # What is kept): per pair of source vectors (a, b) U_a†R₀U_b and
/// U_a†R₀†R₀U_b (n × n); its R₀ (the tests' response) and ‖R₀‖; the forms' size on its rows per term (R₀'s change).
struct FarBasis {
    at: Vec<C64>,
    weights: Vec<f64>,
    f: Vec<Vec<C64>>,
    g: Vec<Vec<C64>>,
    copied: Vec<f64>,
    resolvent: Mat,
    z0: f64,
    r0: f64,
}

/// An atom's kept decompositions, per block (near and far apart): the kept ones and the one it stands on; the
/// decompositions made and the kept ones taken up again.
#[derive(Default)]
pub(crate) struct Cache {
    near: Vec<Vec<NearBasis>>,
    far: Vec<Vec<FarBasis>>,
    at_near: Vec<Option<usize>>,
    at_far: Vec<Option<usize>>,
    pub(crate) made: usize,
    pub(crate) looked: usize,
}

/// A step's result: A, the metric less one, S⁻¹, G (n × n); the sweeps; the readings (rate units): the series' rest,
/// the largest of the near blocks' reuse, of the far blocks' held forms against `Fold::at` and of their centring against
/// the exact energies; the last changes' largest ratio; the work (complex multiply-adds) and the ticks by phase
/// (`PHASES`), and per sweep.
pub(crate) struct Step {
    pub(crate) a: Mat,
    pub(crate) metric: Mat,
    pub(crate) normal: Mat,
    pub(crate) generator: Mat,
    pub(crate) sweeps: usize,
    pub(crate) series: f64,
    pub(crate) reuse: f64,
    pub(crate) far: f64,
    pub(crate) centring: f64,
    pub(crate) copies: f64,
    pub(crate) ratio: f64,
    pub(crate) made: usize,
    pub(crate) rounds: usize,
    pub(crate) work: [f64; 10],
    pub(crate) ticks: [u64; 10],
    pub(crate) sweep_work: Vec<f64>,
    pub(crate) sweep_ticks: Vec<u64>,
}

/// The prototype's layout and buffers, once per fold (`Fold::prototype_workspace`).
pub(crate) struct Workspace {
    n: usize,
    pairs: Vec<Pair>,
    /// Per pair its entries on Ĥ_PP (flat index, value).
    pp: Vec<Vec<(usize, C64)>>,
    /// Ĥ_PP less the forms (the ground states of one frame, less the frame).
    base: Vec<C64>,
    near: Vec<Near>,
    far: Vec<Far>,
    amplitudes: Vec<Amplitude>,
    eps: Vec<[C64; 3]>,
    coef: Vec<C64>,
    f: Vec<Vec<C64>>,
    w: Vec<Vec<f64>>,
    e: Vec<Vec<C64>>,
    hpp: Vec<C64>,
    hbase: Vec<C64>,
    h: Vec<C64>,
    delta: Vec<C64>,
    c: Vec<C64>,
    change: Vec<f64>,
    previous: Vec<f64>,
    whole: Vec<f64>,
    eigen: Mat,
    vectors: Mat,
    temp: Mat,
    renew: Vec<(bool, usize)>,
    order: Vec<(f64, usize, usize)>,
    step: Step,
}

/// V†MV of a block's solved rows (Schur parts).
fn rotate(f: &Factored, m: &Mat, work: &mut f64) -> Mat {
    f.to(&f.to(m, work).adjoint(), work).adjoint()
}

/// The rate's inverse: the relative amplitude error ρ of a reading r = 2ρ + ρ².
fn unrate(r: f64) -> f64 {
    (1.0 + r).sqrt() - 1.0
}

/// The series' order, mode by mode (module documentation, 5): after the first sweep the next one moves a Schur part P of a
/// block by at most q_P a_j in column j -- the next change is the diagonal times the first sweep's residual against the
/// Bloch generator H¹ it gives, −Z¹(H¹ − c⁰), so row by row against column by column (Cauchy–Schwarz) and beam by beam,
/// q_P = Σ_l (Σ_{i∈P} max_j |c⁰_j − w_l − κ_i|⁻² ‖Z¹_l,i‖²)^½ and a_j = ‖(H¹ − c⁰)_:,j‖. The parts of the smallest next order
/// stay at the first sweep's while what they leave together, (Σ q_P²)^½ max_j a_j/‖Z_j‖, reads within the share (all
/// blocks' parts on one share); the others go on. A part held keeps its rows' part of the Bloch generator (`held`).
/// Returns what the held parts leave (ρ). The excited rows far from the column energies (a centred tier before) are these.
#[allow(clippy::too_many_arguments)]
fn freeze(near: &mut [Near], cache: &Cache, h: &[C64], c: &[C64], whole: &[f64], order: &mut Vec<(f64, usize, usize)>, share: f64, n: usize, work: &mut f64) -> f64 {
    let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
    let mut most = 0.0_f64;
    for j in (0..n).filter(|&j| whole[j] > floor) {
        let a2: f64 = (0..n).map(|i| (h[i * n + j] - if i == j { c[j] } else { C64::ZERO }).norm_sqr()).sum();
        most = most.max((a2 / whole[j]).sqrt());
    }
    order.clear();
    for (x, nb) in near.iter_mut().enumerate() {
        let basis = &cache.near[x][cache.at_near[x].unwrap()];
        let s = nb.s;
        for (pi, (idx, at, _)) in basis.f.parts.iter().enumerate() {
            let mut q = 0.0;
            for (sp, &wl) in nb.shifts.iter().enumerate() {
                let mut sum = 0.0;
                for i in *at..at + idx.len() {
                    let gap = (0..n).map(|j| (c[j] - C64::real(wl) - nb.kappa[i]).norm_sqr()).fold(f64::INFINITY, f64::min);
                    let row: f64 = nb.z[sp * s * n + i * n..sp * s * n + i * n + n].iter().map(|z| z.norm_sqr()).sum();
                    sum += row / gap;
                }
                q += sum.sqrt();
            }
            nb.q[pi] = q;
            order.push((q, x, pi));
        }
        *work += (nb.shifts.len() * s * n) as f64 / 2.0;
    }
    order.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let limit = unrate(share);
    let mut sum = 0.0;
    for &(q, x, pi) in order.iter() {
        if (sum + q * q).sqrt() * most > limit {
            break;
        }
        sum += q * q;
        let nb = &mut near[x];
        let (idx, at, _) = &cache.near[x][cache.at_near[x].unwrap()].f.parts[pi];
        for i in *at..at + idx.len() {
            nb.active[i] = false;
            for p in 0..n {
                let v = nb.vtot[i * n + p].conj();
                if v != C64::ZERO {
                    crate::linalg::mul_add(false, &mut nb.held[p * n..p * n + n], v, &nb.ztot[i * n..i * n + n]);
                }
            }
        }
        *work += (idx.len() * n * n) as f64;
    }
    sum.sqrt() * most
}

/// A near block's copies' rows T_C = Σ_a e_a* (R₀U_a†V) Z into `nb.tc` (module documentation, 7).
fn copy_rows(nb: &mut Near, basis: &NearBasis, e: &[Vec<C64>], n: usize, work: &mut f64) {
    let s = nb.s;
    nb.tc.fill(C64::ZERO);
    for (cv, &(g, k, gi)) in nb.copies.iter().enumerate() {
        let ek = e[g][k].conj();
        let yc = &basis.copies[cv];
        let rows = &nb.groups[gi].1;
        for (a2, &row) in rows.iter().enumerate() {
            let out = &mut nb.tc[row * n..row * n + n];
            for i in 0..s {
                let y = ek * yc[a2 * s + i];
                for (o, z) in out.iter_mut().zip(&nb.ztot[i * n..i * n + n]) {
                    *o += y * *z;
                }
            }
        }
        *work += (rows.len() * s * n) as f64;
    }
}

/// out = a b (n × n, row by row; linalg.rs `fused_product`).
fn multiply(a: &[C64], b: &[C64], out: &mut [C64], n: usize) {
    crate::linalg::fused_product(a, b, (n, n, n), out);
}

/// S⁻¹ = (1 + X)^{−1/2} of the metric less one X (Hermitian, X ≥ 0, n × n) into `y`, by the Newton–Schulz iteration
/// Y ← Y(3 − (1 + X)Y²)/2 from its series' second order Y₀ = 1 − X/2 + 3X²/8, to the rounding -- the one algorithm for it.
/// Every iterate is a polynomial in X, so the iteration is scalar on each eigenvalue x ≥ 0 of X: with u = y√(1 + x) it is
/// u ← u(3 − u²)/2, which goes to 1 from every u₀ in (0, √3) (on (0, 1] it rises to 1; (1, √3) it maps into (0, 1)), and
/// u₀(x) = (1 − x/2 + 3x²/8)√(1 + x) lies in [1, 1.24] for 0 ≤ x ≤ 1 (u₀ reaches √3 only at x ≈ 1.5). The fold exists only
/// where its excitation, X's largest eigenvalue, is below one (DESIGN §3.1), so there it always converges, quadratically:
/// a step's error e goes to −(3/2)e² − e³/2 (Y = M^{−1/2}(1 + e)), and a step that moves Y by m leaves (3/2)m² -- it stops
/// where that is within γ_{4n} of Y's size (the next move's reading). False where a step does not shrink the last one's
/// move: X beyond that range, the fold not existing there (the caller returns none). Products alone (3n³ a step).
fn inverse_root(x: &[C64], n: usize, y: &mut [C64], p: &mut [C64], q: &mut [C64], r: &mut [C64], work: &mut f64) -> bool {
    multiply(x, x, p, n);
    for i in 0..n {
        for j in 0..n {
            y[i * n + j] = if i == j { C64::ONE } else { C64::ZERO } - x[i * n + j] * 0.5 + p[i * n + j] * 0.375;
        }
    }
    *work += (n * n * n) as f64;
    let floor = crate::units::rounding(4 * n);
    let mut last = f64::INFINITY;
    loop {
        multiply(y, y, p, n);
        multiply(x, p, q, n);
        for (a, b) in q.iter_mut().zip(p.iter()) {
            *a += *b;
        }
        multiply(y, q, r, n);
        // The move and Y in the Frobenius norm (at least the 2-norm the error's recursion holds in).
        let (mut moved, mut size) = (0.0_f64, 0.0_f64);
        for (a, b) in y.iter_mut().zip(r.iter()) {
            let next = *a * 1.5 - *b * 0.5;
            moved += (next - *a).norm_sqr();
            size += next.norm_sqr();
            *a = next;
        }
        let (moved, size) = (moved.sqrt(), size.sqrt());
        *work += (3 * n * n * n) as f64;
        if 1.5 * moved * moved <= floor * size {
            return true;
        }
        if moved >= last {
            return false;
        }
        last = moved;
    }
}

impl Fold {
    /// The prototype's layout for this fold (module documentation). Refused where a block holds what the prototype does
    /// not: explicit states, a static source, an exact copy, centred excited rows beside solved ones (the choice's to make
    /// solved), or a copy emitting into P.
    pub(crate) fn prototype_workspace(&self) -> Result<Workspace, String> {
        let n = self.size();
        let ng = self.ground.len();
        if n != ng {
            return Err("原型不含显式态".into());
        }
        let mut pairs: Vec<Pair> = vec![];
        for (w, f) in self.light.frequencies.iter().enumerate() {
            for k in 0..f.rank() {
                for kk in 0..f.rank() {
                    pairs.push((w, k, kk));
                }
            }
        }
        // A pair's contraction of a frequency's forms: Σ_pq u_kp* u_k'q O_pq on the entries `keep` takes, by `place`.
        let contract = |pair: Pair, keep: &dyn Fn(usize, usize) -> Option<(usize, usize)>, rows: usize, cols: usize| -> Mat {
            let (w, k, kk) = pair;
            let u = &self.light.frequencies[w].basis;
            let mut m = Mat::zeros(rows, cols);
            for (p, q, entries) in &self.entries[w] {
                let cu = u[k][*p].conj() * u[kk][*q];
                for &(i, j, v) in entries {
                    if let Some((a, b)) = keep(i, j) {
                        m[(a, b)] += cu * v;
                    }
                }
            }
            m
        };
        let pp: Vec<Vec<(usize, C64)>> = pairs
            .iter()
            .map(|&pair| {
                let m = contract(
                    pair,
                    &|i, j| {
                        let (p, q) = (self.ground.iter().position(|&g| g == i)?, self.ground.iter().position(|&g| g == j)?);
                        (self.frames[p] == self.frames[q]).then_some((p, q))
                    },
                    n,
                    n,
                );
                (0..n * n).filter(|&x| m.data[x] != C64::ZERO).map(|x| (x, m.data[x])).collect()
            })
            .collect();
        let mut base = vec![C64::ZERO; n * n];
        for p in 0..n {
            for q in (0..n).filter(|&q| self.frames[q] == self.frames[p]) {
                base[p * n + q] = self.base_ground[(self.ground[p], self.ground[q])] - if p == q { C64::real(self.frame_values[p]) } else { C64::ZERO };
            }
        }
        let (mut near, mut far) = (vec![], vec![]);
        for (o, b) in self.blocks.iter().enumerate() {
            let m = b.rows.len();
            if m == 0 {
                continue;
            }
            if !b.explicit.is_empty() {
                return Err("原型不含显式态".into());
            }
            let s = b.solved;
            if b.rows[..s].iter().any(|r| matches!(r, Row::Copy { .. })) {
                return Err(format!("块 {o}：原型不含精确拷贝"));
            }
            let centre: Vec<usize> = b.sources.iter().map(|x| x.0).collect();
            let mut splits = vec![];
            let mut vectors = vec![];
            for (c, comp) in self.components.iter().enumerate() {
                if b.sources.iter().any(|x| x.1 == c) {
                    splits.extend((0..comp.beams.len()).map(|j| (c, j)));
                    vectors.extend((0..comp.basis.len()).map(|k| (c, k)));
                }
            }
            // The static split: its sources on the copies (a ground state at a copy's frame), none here.
            for r in &b.rows {
                if let Row::Copy { component, .. } = *r {
                    let omega = b.offset.plus(-1, &self.components[component].beat);
                    if (0..ng).any(|p| self.frames[p] == omega) {
                        return Err(format!("块 {o}：原型不含落在 P 帧上的拷贝"));
                    }
                }
            }
            let terms_of = |keep: &dyn Fn(usize, usize) -> Option<(usize, usize)>, size: usize| -> Vec<usize> {
                (0..pairs.len()).filter(|&t| contract(pairs[t], keep, size, size).frobenius() > 0.0).collect()
            };
            let row_of = |x: usize| b.rows.iter().position(|r| *r == Row::Excited(x));
            if s > 0 {
                if b.rows[s..].iter().any(|r| matches!(r, Row::Excited(_))) {
                    return Err(format!("块 {o}：解的块里有居中的激发行（先把它们改为解）"));
                }
                let nc = m - s;
                let mut groups: Vec<(usize, Vec<usize>)> = vec![];
                for (a, r) in b.rows[s..].iter().enumerate() {
                    if let Row::Copy { component, .. } = *r {
                        match groups.iter_mut().find(|g| g.0 == component) {
                            Some(g) => g.1.push(a),
                            None => groups.push((component, vec![a])),
                        }
                    }
                }
                let copies: Vec<(usize, usize, usize)> = groups.iter().enumerate().flat_map(|(gi, (g, _))| (0..self.components[*g].basis.len()).map(move |k| (*g, k, gi))).collect();
                // A term: a pair whose forms reach the solved rows, or whose frequency a copy component's is (its dressing).
                let mut terms = terms_of(&|i, j| if i >= GROUND && j >= GROUND { Some((row_of(i - GROUND).filter(|&a| a < s)?, row_of(j - GROUND).filter(|&a| a < s)?)) } else { None }, s);
                for (t, &(w, _, _)) in pairs.iter().enumerate() {
                    if groups.iter().any(|(g, _)| self.components[*g].part.frequency == w) && !terms.contains(&t) {
                        terms.push(t);
                    }
                }
                terms.sort();
                let z = |x: usize| vec![C64::ZERO; x];
                near.push(Near {
                    o,
                    s,
                    nc,
                    value: b.value,
                    centre,
                    terms,
                    delta: vec![],
                    kappa: z(s),
                    off: vec![0.0; s * s],
                    rho: vec![0.0; s],
                    shifts: vec![0.0; splits.len()],
                    vt: z(splits.len() * s * n),
                    vtot: z(s * n),
                    z: z(splits.len() * s * n),
                    ztot: z(s * n),
                    zold: z(s * n),
                    rhs: z(s * n),
                    inv: z(s * n),
                    bloch: z(n * n),
                    tc: z(nc * n),
                    y: z(s * n),
                    err: vec![0.0; s * n],
                    norms: vec![0.0; s + 1],
                    active: vec![true; s],
                    held: z(n * n),
                    q: vec![0.0; s + 1],
                    reading: 0.0,
                    splits,
                    vectors,
                    groups,
                    copies,
                });
            } else {
                if b.rows.iter().any(|r| matches!(r, Row::Copy { .. })) {
                    return Err(format!("块 {o}：全居中的块里有拷贝"));
                }
                let terms = terms_of(&|i, j| if i >= GROUND && j >= GROUND { Some((row_of(i - GROUND)?, row_of(j - GROUND)?)) } else { None }, m);
                let v = vectors.len();
                far.push(Far { o, m, centre, splits, terms, delta: vec![], products: vec![C64::ZERO; v * v], hf: vec![C64::ZERO; n * n], mf: vec![C64::ZERO; n * n], reading: 0.0, centring: 0.0, vectors });
            }
        }
        for x in near.iter_mut() {
            x.delta = vec![C64::ZERO; x.terms.len()];
        }
        for x in far.iter_mut() {
            x.delta = vec![C64::ZERO; x.terms.len()];
        }
        let z = |r: usize, c: usize| Mat::zeros(r, c);
        let near_parts: usize = near.iter().map(|x| x.s).sum();
        Ok(Workspace {
            n,
            pp,
            base,
            near,
            far,
            amplitudes: vec![Amplitude { value: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] }; self.light.beams.len()],
            eps: vec![[C64::ZERO; 3]; self.light.frequencies.len()],
            coef: vec![C64::ZERO; pairs.len()],
            pairs,
            f: self.components.iter().map(|c| vec![C64::ZERO; c.beams.len()]).collect(),
            w: self.components.iter().map(|c| vec![0.0; c.beams.len()]).collect(),
            e: self.components.iter().map(|c| vec![C64::ZERO; c.basis.len()]).collect(),
            hpp: vec![C64::ZERO; n * n],
            hbase: vec![C64::ZERO; n * n],
            h: vec![C64::ZERO; n * n],
            delta: vec![C64::ZERO; n * n],
            c: vec![C64::ZERO; n],
            change: vec![0.0; n],
            previous: vec![0.0; n],
            whole: vec![0.0; n],
            eigen: z(n, n),
            vectors: z(n, n),
            temp: z(n, n),
            renew: vec![],
            order: Vec::with_capacity(near_parts),
            step: Step {
                a: z(n, n),
                metric: z(n, n),
                normal: z(n, n),
                generator: z(n, n),
                sweeps: 0,
                series: 0.0,
                reuse: 0.0,
                far: 0.0,
                centring: 0.0,
                copies: 0.0,
                ratio: 0.0,
                made: 0,
                rounds: 0,
                work: [0.0; 10],
                ticks: [0; 10],
                sweep_work: vec![],
                sweep_ticks: vec![],
            },
        })
    }

    /// The light at the point (module documentation, 1).
    fn prototype_place(&self, r: [f64; 3], kappa: [f64; 3], ws: &mut Workspace, work: &mut f64) {
        for (a, b) in ws.amplitudes.iter_mut().zip(&self.light.beams) {
            *a = b.amplitude_to(r, 0);
        }
        for (eps, f) in ws.eps.iter_mut().zip(&self.light.frequencies) {
            let mut field = [C64::ZERO; 3];
            for (&l, e) in f.beams.iter().zip(&f.fields) {
                let a = ws.amplitudes[l].value;
                for q in 0..3 {
                    field[q] += a * e[q];
                }
            }
            for (k, u) in f.basis.iter().enumerate() {
                eps[k] = (0..3).map(|q| u[q].conj() * field[q]).sum();
            }
            *work += (3 * f.beams.len() + 3 * f.rank()) as f64;
        }
        for (x, &(w, k, kk)) in ws.coef.iter_mut().zip(&ws.pairs) {
            *x = ws.eps[w][k].conj() * ws.eps[w][kk];
        }
        *work += ws.pairs.len() as f64;
        for (c, comp) in self.components.iter().enumerate() {
            for (j, &l) in comp.beams.iter().enumerate() {
                ws.f[c][j] = comp.factor(ws.amplitudes[l].value);
                let q = &comp.kicks[j];
                ws.w[c][j] = self.hbar_over_mass * (q.iter().zip(&kappa).map(|(q, k)| q * k).sum::<f64>() + 0.5 * q.iter().map(|q| q * q).sum::<f64>());
            }
            // e_k = ε_k co-rotating, ε_k* counter-rotating (fold.rs `Component::gamma`).
            let eps = &ws.eps[comp.part.frequency];
            for k in 0..comp.basis.len() {
                ws.e[c][k] = if comp.part.counter { eps[k].conj() } else { eps[k] };
            }
        }
        ws.hpp.copy_from_slice(&ws.base);
        for (entries, &x) in ws.pp.iter().zip(&ws.coef) {
            for &(at, v) in entries {
                ws.hpp[at] += x * v;
            }
            *work += entries.len() as f64;
        }
    }

    /// A near block's decomposition at the point (module documentation, # What is kept), from the fold's own matrices
    /// there (`Fold::site`, `lay`, `effective`).
    fn near_basis(&self, nb: &Near, ws: &Workspace, site: &Site, work: &mut f64) -> Option<NearBasis> {
        let (n, s, nc) = (ws.n, nb.s, nb.nc);
        let b = &self.blocks[nb.o];
        let laid = self.lay(nb.o, site, work);
        let e = self.effective(nb.o, &laid, work)?;
        let f = Factored::of(&e.k, work)?;
        let mut part = vec![0; s];
        for (pi, (idx, at, _)) in f.parts.iter().enumerate() {
            for a in 0..idx.len() {
                part[at + a] = pi;
            }
        }
        let parts = f.parts.len().max(1);
        let rows: Vec<usize> = b.rows[..s].iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!() }).collect();
        let copy_state = |a: usize| if let Row::Copy { state, .. } = b.rows[s + a] { state } else { unreachable!() };
        let r0 = &e.resolvent;
        // Per copy component: U_k on (S, C_g) and R₀ on (C_g, C_g).
        let u_sc = |g: usize, k: usize, cg: &[usize]| Mat::from_fn(s, cg.len(), |a, c| self.components[g].basis[k][(rows[a], copy_state(cg[c]))]);
        let mut terms = vec![];
        for &t in &nb.terms {
            let (w, k, kk) = ws.pairs[t];
            let u = &self.light.frequencies[w].basis;
            let mut form = Mat::zeros(s, s);
            let mut cc = Mat::zeros(nc, nc);
            for (p, q, entries) in &self.entries[w] {
                let cu = u[k][*p].conj() * u[kk][*q];
                for &(i, j, v) in entries {
                    if i >= GROUND && j >= GROUND {
                        if let (Some(a), Some(c)) = (rows.iter().position(|&x| x == i - GROUND), rows.iter().position(|&x| x == j - GROUND)) {
                            form[(a, c)] += cu * v;
                        }
                    } else if i < GROUND && j < GROUND {
                        // The copies' own rows (one component's copies among themselves).
                        for (_, cg) in &nb.groups {
                            for &x in cg {
                                for &y in cg {
                                    if copy_state(x) == i && copy_state(y) == j {
                                        cc[(x, y)] += cu * v;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            // The dressing: co-rotating e_k e_kk* = ε_k ε_kk* (the pair (w, kk, k)), counter-rotating ε_k* ε_kk (the pair (w, k, kk)).
            let mut dressing = Mat::zeros(s, s);
            for (g, cg) in &nb.groups {
                let comp = &self.components[*g];
                if comp.part.frequency != w {
                    continue;
                }
                let rg = r0.block(cg, cg);
                let (k1, k2) = if comp.part.counter { (k, kk) } else { (kk, k) };
                let x = u_sc(*g, k1, cg).product(&rg).product(&u_sc(*g, k2, cg).adjoint());
                plus(&mut dressing, &x);
            }
            let mut whole = form.clone();
            plus(&mut whole, &dressing);
            let tw = rotate(&f, &whole, work);
            let td = rotate(&f, &dressing, work);
            terms.push((whole.frobenius(), tw, td, cc.frobenius()));
        }
        let at: Vec<C64> = nb.terms.iter().map(|&t| ws.coef[t]).collect();
        let weights = terms.iter().map(|x| x.0).collect();
        let mut diag = vec![C64::ZERO; terms.len() * s];
        let mut norms = vec![0.0; terms.len() * s * parts];
        let mut dressed = vec![0.0; terms.len() * s];
        for (t, (_, tw, td, _)) in terms.iter().enumerate() {
            for i in 0..s {
                diag[t * s + i] = tw[(i, i)];
                for j in (0..s).filter(|&j| j != i) {
                    norms[(t * s + i) * parts + part[j]] += tw[(i, j)].norm_sqr();
                }
                dressed[t * s + i] = (0..s).map(|j| td[(i, j)].norm_sqr()).sum::<f64>().sqrt();
            }
        }
        norms.iter_mut().for_each(|x| *x = x.sqrt());
        let copied = terms.iter().map(|x| x.3).collect();
        // The sources: per split Ỹ_l = V† Σ_k γ_jk U_k (source columns), per vector Y_a = V†U_a.
        let mut source = |c: usize, mat: &dyn Fn(usize, usize) -> C64| -> Vec<C64> {
            let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
            let m = Mat::from_fn(s, n, |a, p| if places.contains(&p) { mat(rows[a], self.ground[p]) } else { C64::ZERO });
            f.to(&m, work).data
        };
        let beams = nb.splits.iter().map(|&(c, j)| source(c, &|x, t| self.components[c].couplings[j][(x, t)])).collect();
        let vectors = nb.vectors.iter().map(|&(c, k)| source(c, &|x, t| self.components[c].basis[k][(x, t)])).collect();
        // The copies' rows: per copy vector R₀(C_g, C_g) U_k(S, C_g)† V.
        let vfull = f.from(&Mat::identity(s), work);
        let copies: Vec<Vec<C64>> = nb.copies.iter().map(|&(g, k, gi)| r0.block(&nb.groups[gi].1, &nb.groups[gi].1).product(&u_sc(g, k, &nb.groups[gi].1).adjoint()).product(&vfull).data).collect();
        let copied_rows = copies.iter().map(|y: &Vec<C64>| y.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).collect();
        Some(NearBasis { at, weights, f, part, diag, norms, dressed, copied, beams, vectors, copies, copied_rows, z0: laid.z0, r0: r0.frobenius() })
    }

    /// A far block's pair forms at the point (module documentation, # What is kept).
    fn far_basis(&self, fb: &Far, ws: &Workspace, site: &Site, work: &mut f64) -> Option<FarBasis> {
        let n = ws.n;
        let b = &self.blocks[fb.o];
        let laid = self.lay(fb.o, site, work);
        let e = self.effective(fb.o, &laid, work)?;
        let r0 = e.resolvent;
        let rows: Vec<usize> = b.rows.iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!() }).collect();
        let u: Vec<Mat> = fb
            .vectors
            .iter()
            .map(|&(c, k)| {
                let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
                Mat::from_fn(fb.m, n, |a, p| if places.contains(&p) { self.components[c].basis[k][(rows[a], self.ground[p])] } else { C64::ZERO })
            })
            .collect();
        let ru: Vec<Mat> = u.iter().map(|x| r0.product(x)).collect();
        let (mut f, mut g) = (vec![], vec![]);
        for a in 0..u.len() {
            for bb in 0..u.len() {
                f.push(u[a].adjoint_product(&ru[bb]).data);
                g.push(ru[a].adjoint_product(&ru[bb]).data);
            }
        }
        *work += (u.len() * fb.m * fb.m * n + 2 * u.len() * u.len() * fb.m * n * n) as f64;
        let copied = fb
            .terms
            .iter()
            .map(|&t| {
                let (w, k, kk) = ws.pairs[t];
                let uu = &self.light.frequencies[w].basis;
                let mut m = Mat::zeros(fb.m, fb.m);
                for (p, q, entries) in &self.entries[w] {
                    let cu = uu[k][*p].conj() * uu[kk][*q];
                    for &(i, j, v) in entries.iter().filter(|x| x.0 >= GROUND && x.1 >= GROUND) {
                        if let (Some(a), Some(c)) = (rows.iter().position(|&x| x == i - GROUND), rows.iter().position(|&x| x == j - GROUND)) {
                            m[(a, c)] += cu * v;
                        }
                    }
                }
                m.frobenius()
            })
            .collect::<Vec<f64>>();
        Some(FarBasis { at: fb.terms.iter().map(|&t| ws.coef[t]).collect(), weights: copied.clone(), f, g, copied, z0: laid.z0, r0: r0.frobenius(), resolvent: r0 })
    }

    /// The prototype's step at a point (r, κ) on an atom's kept decompositions (module documentation): A, M, S⁻¹, G in the
    /// workspace's `Step`. None where a decomposition or a centring is singular.
    pub(crate) fn prototype<'w>(&self, r: [f64; 3], kappa: [f64; 3], cache: &mut Cache, ws: &'w mut Workspace) -> Option<&'w Step> {
        let n = ws.n;
        let mut work = [0.0_f64; 10];
        let mut tick = [0u64; 10];
        ws.step.sweep_work.clear();
        ws.step.sweep_ticks.clear();
        let mut clock = ticks();
        let lap = |k: usize, clock: &mut u64, tick: &mut [u64; 10]| {
            let now = ticks();
            tick[k] += now - *clock;
            *clock = now;
        };
        self.prototype_place(r, kappa, ws, &mut work[PLACE]);
        lap(PLACE, &mut clock, &mut tick);
        let share = crate::model::elim::share();
        if cache.at_near.len() != ws.near.len() {
            cache.near = (0..ws.near.len()).map(|_| vec![]).collect();
            cache.at_near = vec![None; ws.near.len()];
            cache.far = (0..ws.far.len()).map(|_| vec![]).collect();
            cache.at_far = vec![None; ws.far.len()];
        }
        ws.renew.clear();
        let made_before = cache.made;
        let mut round = 0;
        let mut site: Option<Site> = None;
        let (sweeps, series, ratio, rounds) = loop {
            let made = cache.made;
            // The bases (module documentation, 6): the atom's own; beyond the share the nearest kept, then a new one.
            for kind in [false, true] {
                let count = if kind { ws.far.len() } else { ws.near.len() };
                for x in 0..count {
                    let tries = ws.renew.iter().filter(|y| **y == (kind, x)).count();
                    let standing = if kind { cache.at_far[x] } else { cache.at_near[x] };
                    if let (Some(_), 0) = (standing, tries) {
                        continue;
                    }
                    let (terms, kept): (&[usize], Vec<(&[C64], &[f64])>) = if kind {
                        (&ws.far[x].terms, cache.far[x].iter().map(|b| (&b.at[..], &b.weights[..])).collect())
                    } else {
                        (&ws.near[x].terms, cache.near[x].iter().map(|b| (&b.at[..], &b.weights[..])).collect())
                    };
                    let nearest = |but: Option<usize>, work: &mut f64| -> Option<usize> {
                        let mut best: Option<(usize, f64)> = None;
                        for (i, (at, weights)) in kept.iter().enumerate().filter(|(i, _)| Some(*i) != but) {
                            let d: f64 = terms.iter().enumerate().map(|(t, &p)| (ws.coef[p] - at[t]).abs() * weights[t]).sum();
                            *work += terms.len() as f64;
                            if best.is_none_or(|b| d < b.1) {
                                best = Some((i, d));
                            }
                        }
                        best.map(|b| b.0)
                    };
                    let chosen = match (standing, tries) {
                        (Some(at), 0) => Some(at),
                        (None, 0) => nearest(None, &mut work[REFRESH]),
                        (_, 1) => nearest(standing, &mut work[REFRESH]),
                        _ => None,
                    };
                    if chosen.is_some() && chosen != standing {
                        cache.looked += 1;
                    }
                    let at = match chosen {
                        Some(at) => at,
                        None => {
                            if site.is_none() {
                                site = Some(self.site(r, kappa, &mut work[REFRESH]));
                            }
                            let site = site.as_ref().unwrap();
                            if kind {
                                let basis = self.far_basis(&ws.far[x], ws, site, &mut work[REFRESH])?;
                                cache.far[x].push(basis);
                                cache.far[x].len() - 1
                            } else {
                                let basis = self.near_basis(&ws.near[x], ws, site, &mut work[REFRESH])?;
                                cache.near[x].push(basis);
                                cache.near[x].len() - 1
                            }
                        }
                    };
                    if chosen.is_none() {
                        cache.made += 1;
                    }
                    if kind {
                        cache.at_far[x] = Some(at);
                    } else {
                        cache.at_near[x] = Some(at);
                    }
                }
            }
            lap(REFRESH, &mut clock, &mut tick);
            // The near blocks: the diagonal to first order and what it leaves; their sources (module documentation, 2, 3).
            let Workspace { near, far, coef, f, w, e, hpp, hbase, h, delta, c, change, previous, whole, step, order, .. } = &mut *ws;
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let fa = &basis.f;
                let parts = fa.parts.len().max(1);
                let z0: f64 = nb.centre.iter().map(|&p| hpp[p * n + p].re).sum::<f64>() / nb.centre.len().max(1) as f64;
                for i in 0..s {
                    nb.kappa[i] = fa.t[(i, i)] - C64::real(nb.value);
                    for g in 0..parts {
                        nb.off[i * parts + g] = if g == basis.part[i] { fa.dropped[g] } else { 0.0 };
                    }
                    nb.rho[i] = 0.0;
                }
                let mut dm = (z0 - basis.z0).abs();
                for (t, &p) in nb.terms.iter().enumerate() {
                    let d = coef[p] - basis.at[t];
                    nb.delta[t] = d;
                    if d == C64::ZERO {
                        continue;
                    }
                    let size = d.abs();
                    dm += size * basis.copied[t];
                    let (dg, nm) = (&basis.diag[t * s..(t + 1) * s], &basis.norms[t * s * parts..(t + 1) * s * parts]);
                    for i in 0..s {
                        nb.kappa[i].re += (d * dg[i]).re;
                    }
                    for (o, x) in nb.off[..s * parts].iter_mut().zip(nm) {
                        *o += size * x;
                    }
                    work[DIAGONAL] += (s + s * parts) as f64;
                }
                // R₀'s own change: δR₀ ≤ ‖R₀‖²‖δM‖/(1 − ‖R₀‖‖δM‖) through the dressing, ρ_i = ε_R Σ_t |c_t| ‖(V†N_t,dress V)_i‖.
                if nb.nc > 0 {
                    let x = basis.r0 * dm;
                    let eps_r = if x < 1.0 { x / (1.0 - x) } else { f64::INFINITY };
                    for (t, &p) in nb.terms.iter().enumerate() {
                        let size = coef[p].abs();
                        for i in 0..s {
                            nb.rho[i] += eps_r * size * basis.dressed[t * s + i];
                        }
                    }
                    work[DIAGONAL] += (nb.terms.len() * s) as f64 / 4.0;
                }
            }
            lap(DIAGONAL, &mut clock, &mut tick);
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                for (sp, &(cc, j)) in nb.splits.iter().enumerate() {
                    nb.shifts[sp] = w[cc][j];
                    let fl = f[cc][j];
                    for (v, y) in nb.vt[sp * s * n..(sp + 1) * s * n].iter_mut().zip(&basis.beams[sp]) {
                        *v = fl * *y;
                    }
                }
                nb.vtot.fill(C64::ZERO);
                for (a, &(cc, k)) in nb.vectors.iter().enumerate() {
                    let ek = e[cc][k];
                    for (v, y) in nb.vtot.iter_mut().zip(&basis.vectors[a]) {
                        *v += ek * *y;
                    }
                }
                work[SOURCES] += ((nb.splits.len() + nb.vectors.len()) * s * n) as f64;
            }
            lap(SOURCES, &mut clock, &mut tick);
            // The far blocks' pair forms (module documentation, 4): H_far = Σ e_a*e_b U_a†R₀U_b, M_far likewise.
            hbase.copy_from_slice(hpp);
            for (x, fb) in far.iter_mut().enumerate() {
                let basis = &cache.far[x][cache.at_far[x].unwrap()];
                let v = fb.vectors.len();
                for a in 0..v {
                    for b in 0..v {
                        let (ca, ka) = fb.vectors[a];
                        let (cb, kb) = fb.vectors[b];
                        fb.products[a * v + b] = e[ca][ka].conj() * e[cb][kb];
                    }
                }
                fb.hf.fill(C64::ZERO);
                fb.mf.fill(C64::ZERO);
                for (ab, &x) in fb.products.iter().enumerate() {
                    for ((hf, mf), (ff, gg)) in fb.hf.iter_mut().zip(fb.mf.iter_mut()).zip(basis.f[ab].iter().zip(&basis.g[ab])) {
                        *hf += x * *ff;
                        *mf += x * *gg;
                    }
                }
                for (h, x) in hbase.iter_mut().zip(&fb.hf) {
                    *h += *x;
                }
                for (t, &p) in fb.terms.iter().enumerate() {
                    fb.delta[t] = coef[p] - basis.at[t];
                }
                work[FAR] += (v * v + 2 * v * v * n * n + n * n) as f64;
            }
            lap(FAR, &mut clock, &mut tick);
            // The series (module documentation, 5).
            h.copy_from_slice(hbase);
            for nb in near.iter_mut() {
                nb.ztot.fill(C64::ZERO);
                nb.active.fill(true);
                nb.held.fill(C64::ZERO);
            }
            previous.fill(0.0);
            let mut sweep = 0;
            let mut frozen = 0.0_f64;
            let (series, ratio) = loop {
                let (t0, w0) = (ticks(), work[SERIES]);
                // The columns' energies (H's diagonal, each column at its own) and the series' part of H (off it).
                for j in 0..n {
                    c[j] = h[j * n + j];
                }
                delta.copy_from_slice(h);
                for j in 0..n {
                    delta[j * n + j] = C64::ZERO;
                }
                change.fill(0.0);
                whole.fill(0.0);
                for (x, nb) in near.iter_mut().enumerate() {
                    let basis = &cache.near[x][cache.at_near[x].unwrap()];
                    let s = nb.s;
                    let (t, ends) = (&basis.f.t, &basis.f.ends);
                    // Only the rows still going on move: the held ones keep their response.
                    nb.zold.copy_from_slice(&nb.ztot);
                    for i in (0..s).filter(|&i| nb.active[i]) {
                        nb.ztot[i * n..i * n + n].fill(C64::ZERO);
                    }
                    for sp in 0..nb.splits.len() {
                        let wl = nb.shifts[sp];
                        let zl = &mut nb.z[sp * s * n..(sp + 1) * s * n];
                        let rhs = &mut nb.rhs;
                        rhs.copy_from_slice(&nb.vt[sp * s * n..(sp + 1) * s * n]);
                        if sweep > 0 {
                            for i in (0..s).filter(|&i| nb.active[i]) {
                                for k in 0..n {
                                    let zik = zl[i * n + k];
                                    if zik == C64::ZERO {
                                        continue;
                                    }
                                    crate::linalg::mul_add(true, &mut rhs[i * n..i * n + n], zik, &delta[k * n..k * n + n]);
                                }
                            }
                            work[SERIES] += (nb.active.iter().filter(|&&x| x).count() * n * n) as f64;
                        }
                        // The diagonal: (c_j − w_l − κ_i)⁻¹, each column at its own energy.
                        for i in (0..s).filter(|&i| nb.active[i]) {
                            for j in 0..n {
                                nb.inv[i * n + j] = (c[j] - C64::real(wl) - nb.kappa[i]).recip();
                            }
                        }
                        work[SERIES] += (s * n) as f64;
                        for i in (0..s).rev().filter(|&i| nb.active[i]) {
                            for l in i + 1..ends[i] {
                                crate::linalg::mul_add(false, &mut rhs[i * n..i * n + n], t[(i, l)], &zl[l * n..l * n + n]);
                            }
                            work[SERIES] += ((ends[i] - i - 1) * n + n) as f64;
                            for j in 0..n {
                                zl[i * n + j] = rhs[i * n + j] * nb.inv[i * n + j];
                            }
                        }
                        for i in (0..s).filter(|&i| nb.active[i]) {
                            for (a, b) in nb.ztot[i * n..i * n + n].iter_mut().zip(&zl[i * n..i * n + n]) {
                                *a += *b;
                            }
                        }
                    }
                    // The change and the whole response, column by column; the Bloch generator's part Ṽ†Z.
                    for i in 0..s {
                        for j in 0..n {
                            let zn = nb.ztot[i * n + j];
                            if nb.active[i] {
                                change[j] += (zn - nb.zold[i * n + j]).norm_sqr();
                            }
                            whole[j] += zn.norm_sqr();
                        }
                    }
                    nb.bloch.copy_from_slice(&nb.held);
                    for i in (0..s).filter(|&i| nb.active[i]) {
                        for p in 0..n {
                            let v = nb.vtot[i * n + p].conj();
                            if v == C64::ZERO {
                                continue;
                            }
                            crate::linalg::mul_add(false, &mut nb.bloch[p * n..p * n + n], v, &nb.ztot[i * n..i * n + n]);
                        }
                    }
                    work[SERIES] += (n * s * n + nb.splits.len() * s * n) as f64;
                }
                h.copy_from_slice(hbase);
                for nb in near.iter() {
                    for (a, b) in h.iter_mut().zip(&nb.bloch) {
                        *a += *b;
                    }
                }
                for fb in far.iter() {
                    for j in 0..n {
                        whole[j] += fb.mf[j * n + j].re;
                    }
                }
                sweep += 1;
                if sweep == 1 {
                    frozen = freeze(near, cache, h, c, whole, order, share, n, &mut work[SERIES]);
                }
                // The rest: the last change times r/(1 − r), r the ratio of the last two changes (none after the first).
                let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
                let (mut rest, mut most) = (0.0_f64, 0.0_f64);
                for j in (0..n).filter(|&j| whole[j] > floor) {
                    let ratio = if previous[j] > 0.0 { (change[j] / previous[j]).sqrt() } else { f64::INFINITY };
                    most = most.max(ratio);
                    let x = if ratio < 1.0 { change[j].sqrt() * ratio / (1.0 - ratio) } else { f64::INFINITY };
                    rest = rest.max(rate(x / whole[j].sqrt()));
                }
                previous.copy_from_slice(change);
                step.sweep_ticks.push(ticks() - t0);
                step.sweep_work.push(work[SERIES] - w0);
                // Two sweeps at the least; then within the share, or no longer converging (a ratio at or above one after
                // the second change: the self-consistency or the series does not exist here).
                if sweep >= 2 && (rest <= share || (sweep >= 3 && most >= 1.0)) {
                    break (rate(unrate(rest) + frozen), most);
                }
            };
            lap(SERIES, &mut clock, &mut tick);
            // The readings of what is held (module documentation, 6), at the columns' energies H_jj. The reuse's first-order
            // error of the response the step uses, Z = Σ_l Z_l: Σ_l R_l δK̃ Z_l = R̄ δK̃ Z + Σ_l (R_l − R̄) δK̃ Z_l, R̄ at the
            // beams' mean Doppler shift w̄ -- row by row, row i of δK̃Z at most Σ_P off_iP ‖Z_P∖i‖ (+ ρ_i‖Z‖), the beams'
            // rest at most W|R̄|²/(1 − W|R̄|) (W the largest |w_l − w̄|) times (Σ_P off_iP + ρ_i) Σ_l ‖Z_l‖. Read on the
            // beams one by one against their sum, the reading would grow where the beams' responses cancel (a standing
            // wave's node) though the error cancels with them.
            for (x, nb) in near.iter_mut().enumerate() {
                let basis = &cache.near[x][cache.at_near[x].unwrap()];
                let s = nb.s;
                let parts = basis.f.parts.len().max(1);
                let beams = nb.splits.len();
                let mean = nb.shifts.iter().sum::<f64>() / beams.max(1) as f64;
                let spread = nb.shifts.iter().map(|w| (w - mean).abs()).fold(0.0, f64::max);
                for j in 0..n {
                    nb.norms[..parts].fill(0.0);
                    for i in 0..s {
                        nb.norms[basis.part[i]] += nb.ztot[i * n + j].norm_sqr();
                    }
                    let total: f64 = nb.norms[..parts].iter().sum();
                    let mut each = 0.0;
                    for sp in 0..beams {
                        each += (0..s).map(|i| nb.z[sp * s * n + i * n + j].norm_sqr()).sum::<f64>().sqrt();
                    }
                    for i in 0..s {
                        let own = basis.part[i];
                        let mut e = nb.rho[i] * total.sqrt();
                        let mut all = nb.rho[i];
                        for g in 0..parts {
                            let off = nb.off[i * parts + g];
                            all += off;
                            if off > 0.0 {
                                e += off * if g == own { (nb.norms[g] - nb.ztot[i * n + j].norm_sqr()).max(0.0) } else { nb.norms[g] }.sqrt();
                            }
                        }
                        let gap = (h[j * n + j] - C64::real(mean) - nb.kappa[i]).norm_sqr().sqrt();
                        let rest = if spread < gap { all * each * spread / (gap * (gap - spread)) } else { f64::INFINITY };
                        nb.err[i * n + j] = if e > 0.0 || rest > 0.0 { e / gap + rest } else { 0.0 };
                    }
                }
                work[READING] += (s * n * (beams + parts + 2)) as f64 / 4.0;
                let floor = (crate::units::rounding(n) * whole.iter().sum::<f64>().sqrt()).powi(2);
                nb.reading = (0..n).filter(|&j| whole[j] > floor).map(|j| rate(((0..s).map(|i| nb.err[i * n + j].powi(2)).sum::<f64>() / whole[j]).sqrt())).fold(0.0, f64::max);
            }
            for (x, fb) in far.iter_mut().enumerate() {
                let basis = &cache.far[x][cache.at_far[x].unwrap()];
                let z0: f64 = fb.centre.iter().map(|&p| hpp[p * n + p].re).sum::<f64>() / fb.centre.len().max(1) as f64;
                let held = (z0 - basis.z0).abs() + fb.delta.iter().zip(&basis.copied).map(|(d, c)| d.abs() * c).sum::<f64>();
                let energies = (0..n).map(|j| (h[j * n + j] - C64::real(z0)).abs()).fold(0.0, f64::max) + fb.splits.iter().map(|&(cc, j)| w[cc][j].abs()).fold(0.0, f64::max);
                let read = |d: f64| {
                    let x = basis.r0 * d;
                    if x < 1.0 { rate(x / (1.0 - x)) } else { f64::INFINITY }
                };
                fb.reading = read(held);
                fb.centring = read(energies);
                work[READING] += fb.terms.len() as f64 / 4.0;
            }
            lap(READING, &mut clock, &mut tick);
            let mut renew = vec![];
            for (x, nb) in near.iter().enumerate() {
                if nb.reading > share {
                    renew.push((false, x));
                }
            }
            for (x, fb) in far.iter().enumerate() {
                if fb.reading > share {
                    renew.push((true, x));
                }
            }
            if renew.is_empty() || round == 2 || cache.made > made {
                break (sweep, series, ratio, round + 1);
            }
            ws.renew.extend(renew);
            round += 1;
        };
        // The copies' rows and the compression (module documentation, 7).
        let Workspace { near, far, h, step, eigen, vectors, temp, e, .. } = &mut *ws;
        let (a, metric) = (&mut step.a, &mut step.metric);
        a.data.copy_from_slice(h);
        metric.data.fill(C64::ZERO);
        let mut copies_read = 0.0_f64;
        for (x, nb) in near.iter_mut().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            nb.tc.fill(C64::ZERO);
            if nb.nc == 0 {
                continue;
            }
            // Their rows' own part of the compression (their metric T_C†T_C and z₀T_C†T_C; their dressing is the kept
            // decomposition's) left out where it reads within the share: ‖T_C,j‖ ≤ Σ_a |e_a| ‖R₀U_a†V‖ ‖Z_j‖, a copy row's
            // error absolute (a population's amplitude per unit P amplitude, DESIGN §1.5).
            let size: f64 = nb.copies.iter().zip(&basis.copied_rows).map(|(&(g, k, _), y)| e[g][k].abs() * y).sum();
            let most = (0..n).map(|j| (0..nb.s).map(|i| nb.ztot[i * n + j].norm_sqr()).sum::<f64>()).fold(0.0, f64::max).sqrt();
            let read = rate(size * most);
            copies_read = copies_read.max(read);
            work[COPIES] += (nb.copies.len() + nb.s * n / 4) as f64;
            if read <= share {
                continue;
            }
            copy_rows(nb, basis, e, n, &mut work[COPIES]);
            for k in 0..nb.nc {
                for i in 0..n {
                    let ti = nb.tc[k * n + i].conj();
                    if ti == C64::ZERO {
                        continue;
                    }
                    for j in 0..n {
                        let tkj = nb.tc[k * n + j];
                        metric.data[i * n + j] += ti * tkj;
                        a.data[i * n + j] += ti * tkj * basis.z0;
                    }
                }
            }
            work[COPIES] += (nb.nc * n * n) as f64;
        }
        lap(COPIES, &mut clock, &mut tick);
        for (x, nb) in near.iter_mut().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let s = nb.s;
            let (t, ends) = (&basis.f.t, &basis.f.ends);
            // (Ṽ†Z)†, and Y = (T + δκ − o)Z with κ on the diagonal.
            for p in 0..n {
                for j in 0..n {
                    a.data[p * n + j] += nb.bloch[j * n + p].conj();
                }
            }
            for i in 0..s {
                for j in 0..n {
                    nb.y[i * n + j] = nb.kappa[i] * nb.ztot[i * n + j];
                }
                for l in i + 1..ends[i] {
                    let tij = t[(i, l)];
                    for j in 0..n {
                        nb.y[i * n + j] += tij * nb.ztot[l * n + j];
                    }
                }
                work[COMPRESSION] += ((ends[i] - i) * n) as f64;
            }
            for k in 0..s {
                for i in 0..n {
                    let zi = nb.ztot[k * n + i].conj();
                    if zi == C64::ZERO {
                        continue;
                    }
                    crate::linalg::mul_add(false, &mut a.data[i * n..i * n + n], zi, &nb.y[k * n..k * n + n]);
                    crate::linalg::mul_add(false, &mut metric.data[i * n..i * n + n], zi, &nb.ztot[k * n..k * n + n]);
                }
            }
            work[COMPRESSION] += (2 * s * n * n + n * n) as f64;
        }
        for (x, fb) in far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            for ((am, mm), mf) in a.data.iter_mut().zip(metric.data.iter_mut()).zip(&fb.mf) {
                *am += *mf * basis.z0;
                *mm += *mf;
            }
            work[COMPRESSION] += (n * n) as f64;
        }
        lap(COMPRESSION, &mut clock, &mut tick);
        // S⁻¹ = (1 + X)^{−1/2} (`inverse_root`: it converges wherever the fold exists), G = S⁻¹AS⁻¹.
        if !inverse_root(&metric.data, n, &mut step.normal.data, &mut eigen.data, &mut vectors.data, &mut temp.data, &mut work[NORMAL]) {
            return None;
        }
        multiply(&step.normal.data, &step.a.data, &mut temp.data, n);
        multiply(&temp.data, &step.normal.data, &mut step.generator.data, n);
        work[NORMAL] += (2 * n * n * n) as f64;
        lap(NORMAL, &mut clock, &mut tick);
        step.sweeps = sweeps;
        step.series = series;
        step.ratio = ratio;
        step.reuse = near.iter().map(|x| x.reading).fold(0.0, f64::max);
        step.far = far.iter().map(|x| x.reading).fold(0.0, f64::max);
        step.centring = far.iter().map(|x| x.centring).fold(0.0, f64::max);
        step.copies = copies_read;
        step.made = cache.made - made_before;
        step.rounds = rounds;
        step.work = work;
        step.ticks = tick;
        Some(&ws.step)
    }
}

impl Workspace {
    /// Per block of the fold its response on its Q rows at the last step, x coordinates, in the block's row order (the
    /// tests' comparison with `Fold::at`): a near block's solved rows VZ and its copies' T_C, a far block's R₀B; None for
    /// the blocks the prototype does not hold (no rows).
    pub(crate) fn responses(&self, fold: &Fold, cache: &Cache) -> Vec<Option<Mat>> {
        let n = self.n;
        let mut out: Vec<Option<Mat>> = (0..fold.blocks.len()).map(|_| None).collect();
        let mut w = 0.0;
        for (x, nb) in self.near.iter().enumerate() {
            let basis = &cache.near[x][cache.at_near[x].unwrap()];
            let z = Mat { rows: nb.s, cols: n, data: nb.ztot.clone() };
            let ts = basis.f.from(&z, &mut w);
            let mut rows = Near { tc: vec![C64::ZERO; nb.nc * n], ..nb.clone() };
            copy_rows(&mut rows, basis, &self.e, n, &mut w);
            out[nb.o] = Some(Mat::from_fn(nb.s + nb.nc, n, |i, j| if i < nb.s { ts[(i, j)] } else { rows.tc[(i - nb.s) * n + j] }));
        }
        for (x, fb) in self.far.iter().enumerate() {
            let basis = &cache.far[x][cache.at_far[x].unwrap()];
            let b = &fold.blocks[fb.o];
            let rows: Vec<usize> = b.rows.iter().map(|r| if let Row::Excited(x) = *r { x } else { unreachable!() }).collect();
            let mut src = Mat::zeros(fb.m, n);
            for &(c, k) in &fb.vectors {
                let places: Vec<usize> = b.sources.iter().filter(|x| x.1 == c).map(|x| x.0).collect();
                for a in 0..fb.m {
                    for &p in &places {
                        src[(a, p)] += self.e[c][k] * fold.components[c].basis[k][(rows[a], fold.ground[p])];
                    }
                }
            }
            out[fb.o] = Some(basis.resolvent.product(&src));
        }
        out
    }

    /// The blocks the prototype holds: (near: (block, rows s, copies nc, beam splits, terms), far: (block, rows, terms)).
    #[allow(clippy::type_complexity)]
    pub(crate) fn layout(&self) -> (Vec<(usize, usize, usize, usize, usize)>, Vec<(usize, usize, usize)>) {
        (self.near.iter().map(|x| (x.o, x.s, x.nc, x.splits.len(), x.terms.len())).collect(), self.far.iter().map(|x| (x.o, x.m, x.terms.len())).collect())
    }
}
