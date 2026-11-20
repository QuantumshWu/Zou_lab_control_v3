//! The one fold (RUSTPLAN §2.1–§2.3 and its corrections of 2026-11-19: "模型只定一次", "一个折叠公式，快分量与光镊都由它
//! 折叠"): Reiter–Sørensen second order with harmonics and dissipation (td_s.hilbert_rs, each column at E_a + f), the
//! full reference's and the model's alike -- they differ only in what is folded and at which generator.
//!
//! # The fold
//! A Lindblad model on a set of states by frequency components (`ModelAt`): H's components H⁽ᶜ⁾ e^{−if_c t} taken both
//! ways (each nonzero key with its conjugate at −key), and jump channels L_ch = Σ_c L_ch⁽ᶜ⁾ e^{−if_c t}. A fold keeps the
//! states K and, among their elements, the slow ones: the element b ← a of the component c has the fast index n = Σ_l c_l
//! n_l + n_b − n_a (`Sectors`, the frame's integer fast parts (n_φ, n_tw) of the key coordinates and of the states,
//! floquet/frame.rs), slow where its folding coordinates (`axes`) are zero. Everything else is folded: every component of
//! the other states F (= Q), and the fast elements within K (K's sidebands -- the hyperfine n_φ and the tweezer n_tw).
//! The folded states' unperturbed generator is block diagonal,
//!   G₀ = G_BB − (i/2) Γ_BB,  Γ = Σ_ch Σ_c L_ch⁽ᶜ⁾† L_ch⁽ᶜ⁾  (the static decay),
//! G the static generator the fold is taken at (`generator`), on the blocks B: Q, and K's states by their fast parts (G
//! and Γ do not couple two of them: a static element between them is fast and folded). Column a of K at its energy
//! E_a = Re G_aa and the component's frequency,
//!   M_c[B, a] = (E_a + f_c − G₀_BB)⁻¹ H⁽ᶜ⁾_Ba        (B = Q, or a block of K where c's element into it is fast),
//! and the model on K, slow elements alone,
//!   H_eff = Σ_c [H⁽ᶜ⁾_KK]_slow + ½ Σ_{c′,c} [H⁽ᶜ′⁾_K· M_c + h.c.]_slow,  the pair at key c + c′,
//!   L_(ch,n) = [L_ch⁽ᶜ⁾_KK + Σ_c′ L_ch⁽ᶜ′⁾_K· M_c]_n,  each element in the channel of its fast index n,
//! so that every channel is slow (its elements beat together), and the interference of channels apart by a fast
//! frequency -- and its share of the decay −(i/2)L†L -- is not in the model. A pair whose total is fast is of the fourth
//! order in the couplings and not in the model either. The gradient along r_i is the same formula's derivative,
//!   ∂M_c[B, a] = (E_a + f_c − G₀_BB)⁻¹ (∂H⁽ᶜ⁾_Ba + (∂G₀_BB − ∂E_a) M_c[B, a]).
//!
//! # The two folds
//! - The full reference (`Tweezer`, `reference`): the scene's components and its Cartesian emission on the 32 states, all
//!   kept, the tweezer's sidebands folded (axes n_tw) at the request's H0 (pysel.status.md: the denominators the bare
//!   levels E = diag H0, fixed, not differentiated -- their dependence on the local shifts is of the fourth order, 10⁻⁷
//!   relative, within the one allowed distinction's second order). Its components are a_tw X_tw and a_tw* R_tw and its
//!   denominators fixed, so the fold is linear in them: made once at unit amplitude, its pairs (n_tw = 0) are |a_tw|²
//!   times the unit fold's and its channels (n_tw = ±1) a_tw or a_tw* times them; at a point the amplitude alone is put
//!   in. Six channels: the three emission axes, co- and counter-rotating, apart from the emission's (n_tw = 0).
//! - The model (`Rs::at`): the reference folded again on P at its static part, every fast element folded (axes n_φ,
//!   n_tw): Q at every component, P's hyperfine sidebands. The integrator sees the slow components alone.
//!
//! # At a point: which block how
//! The folded states' generator at a point splits into its connected parts (the pattern of G₀ and ∂G₀), and each part's
//! resolvent is taken one of two ways, both to the rounding of the formula above:
//! - Q's parts: decomposed at the point, G₀ = V diag(μ) V⁻¹ (`linalg::Eigen`), every column's resolvent that one
//!   spectrum. Q's levels lie within the local shifts of resonance (the tweezer's excited shift is of the order of the
//!   detunings), so nothing fixed is near them: the decomposition is the least work at the point.
//! - K's sideband parts: their denominators are a fast frequency -- a hyperfine φ ≈ 4·10⁴ rad/µs, or the tweezer's 10⁸ --
//!   off the local energies, whose couplings within the part (Zeeman, the outside forms' tensor shifts, the tweezer's) are
//!   ≲ 10² rad/µs. Each column's resolvent is the Neumann series about the part's diagonal at the point,
//!   (D_z − N)⁻¹ = Σ_k (D_z⁻¹N)^k D_z⁻¹, D_z = z − diag G₀ (exact at the point), N its off-diagonal: ρ = ‖D_z⁻¹N‖_∞ ≈
//!   2·10⁻³ and less, and the series is summed until its remainder's bound ‖w_k‖ ρ/(1 − ρ) is below the rounding of the
//!   column's sum, u‖M‖_∞ -- the count the ratio's own (a few terms; one where the part is diagonal, as the tweezer's bare
//!   levels are). No decomposition. Where ρ ≥ 1 the series does not converge and the part is decomposed as Q's.
//!
//! Against td_s.hilbert_rs's column-by-column inverse it is the same formula to the rounding of the decomposition, κ(V)u
//! relative (tests, `the_fold_is_hilbert_rs_to_rounding`).

use std::collections::BTreeMap;

use crate::floquet::{Hilbert, Lift, Part, Sectors};
use crate::linalg::{self, C64, Mat, V3};
use crate::request::STATES;
use crate::scene::{Components, Jumps, Key, Scene};
use crate::units::UNIT_ROUNDOFF;

/// One frequency component of a model operator: `value` e^{−i f t}.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelPart {
    pub key: Key,
    /// f (rad/µs), rest frame.
    pub frequency: f64,
    pub value: Mat,
    /// ∂value/∂r_i, i = x, y, z.
    pub gradient: [Mat; 3],
}

/// A jump channel: the Cartesian emission it comes of, the fast index (n_φ, n_tw) its elements share, its components and
/// the same of k∘C (the photon's wave number per element: rad/µm · 1/√µs).
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub emission: usize,
    pub fast: [i64; 2],
    pub parts: Vec<ModelPart>,
    pub kicks: Vec<ModelPart>,
}

/// A Lindblad model at a point by frequency components: what a fold takes and gives.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAt {
    /// Its states (indices into the 32) and the states the fold that made it folded (Q).
    pub states: Vec<usize>,
    pub folded: Vec<usize>,
    /// The levels over the 32 states (indices into the 32): a level's population is its kept states' and its folded
    /// states' lifted ones.
    pub groups: Vec<Vec<usize>>,
    /// H's components, the static one and each other key both ways.
    pub effective: Vec<ModelPart>,
    pub jumps: Vec<Channel>,
    /// M_c on Q (|Q| × n) by components: the folded states' amplitudes.
    pub lift: Vec<ModelPart>,
}

impl Sectors {
    /// The full reference's: the tweezer's coordinate alone (n_tw = 1 on its generator; its counter-rotating key
    /// −e_tw − 2e_ν has n_tw = −1), the states all slow.
    pub fn tweezer(scene: &Scene) -> Sectors {
        let tw = scene.tweezer();
        Sectors { states: vec![[0; 2]; STATES], generators: (0..scene.generators.len()).map(|g| [0, (g == tw) as i64]).collect() }
    }

    /// Σ_l c_l n_l of a key on the folding coordinates.
    fn key(&self, key: &Key, axes: [bool; 2]) -> [i64; 2] {
        std::array::from_fn(|i| if axes[i] { key.iter().zip(&self.generators).map(|(&k, g)| i64::from(k) * g[i]).sum() } else { 0 })
    }

    /// A state's fast part on the folding coordinates.
    fn state(&self, state: usize, axes: [bool; 2]) -> [i64; 2] {
        std::array::from_fn(|i| if axes[i] { self.states[state][i] } else { 0 })
    }
}

/// The fast coordinates the full reference folds: the tweezer's (n_tw).
pub const REFERENCE: [bool; 2] = [false, true];
/// The fast coordinates the model folds: every one.
pub const MODEL: [bool; 2] = [true, true];

/// The levels of the states `of` (indices into the 32): one per ground F and one per excited (line, F′), each its members'
/// places in `of`, in the order of their first member (the request's labels "manifold-F-m").
pub fn levels(labels: &[String], of: &[usize]) -> Vec<Vec<usize>> {
    let mut found: Vec<(&str, Vec<usize>)> = Vec::new();
    for (place, &state) in of.iter().enumerate() {
        let level = labels[state].rsplit_once('-').map_or(labels[state].as_str(), |(level, _)| level);
        let level = if level.ends_with('-') { &level[..level.len() - 1] } else { level };
        match found.iter_mut().find(|(name, _)| *name == level) {
            Some((_, members)) => members.push(place),
            None => found.push((level, vec![place])),
        }
    }
    found.into_iter().map(|(_, members)| members).collect()
}

fn nonzero(m: &Mat) -> bool {
    m.data.iter().any(|z| *z != C64::ZERO)
}

fn sum_keys(a: &Key, b: &Key) -> Key {
    let length = a.len().max(b.len());
    (0..length).map(|i| a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0)).collect()
}

fn negated(key: &Key) -> Key {
    key.iter().map(|k| -k).collect()
}

fn plus(a: [i64; 2], b: [i64; 2]) -> [i64; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

fn zeros(rows: usize, cols: usize) -> [Mat; 3] {
    std::array::from_fn(|_| Mat::zeros(rows, cols))
}

/// a b, the zero entries of a skipped (the scene's operators are block sparse: a coupling ground ← excited, a form
/// within a manifold, an emission ground ← excited).
fn product(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows);
    let mut out = Mat::zeros(a.rows, b.cols);
    if b.cols == 0 {
        return out;
    }
    for (row, x) in out.data.chunks_exact_mut(b.cols).zip(a.data.chunks_exact(a.cols.max(1))) {
        for (x, b) in x.iter().zip(b.data.chunks_exact(b.cols)) {
            if *x != C64::ZERO {
                linalg::mul_add(false, row, *x, b);
            }
        }
    }
    out
}

/// out ← out + a† b, the zero entries of a skipped.
fn adjoint_product_onto(a: &Mat, b: &Mat, scale: C64, out: &mut Mat) {
    assert_eq!((a.rows, a.cols, b.cols), (b.rows, out.rows, out.cols));
    for k in 0..a.rows {
        for i in 0..a.cols {
            let x = a[(k, i)];
            if x != C64::ZERO {
                linalg::mul_add(false, &mut out.data[i * out.cols..(i + 1) * out.cols], scale * x.conj(), &b.data[k * b.cols..(k + 1) * b.cols]);
            }
        }
    }
}

/// Sums of model parts by key, in the keys' order.
#[derive(Default)]
struct Sums(BTreeMap<Key, ModelPart>);

impl Sums {
    /// Adds scale·x into the part of `key` (size × size), x's entry (i, j) -- or, `adjoint`, x's (j, i) conjugated -- at
    /// (rows[i], cols[j]), where `keep(rows[i], cols[j])` holds.
    #[allow(clippy::too_many_arguments)]
    fn add(&mut self, key: &Key, frequency: f64, size: usize, x: (&Mat, [&Mat; 3]), adjoint: bool, rows: &[usize], cols: &[usize], scale: f64, keep: impl Fn(usize, usize) -> bool) {
        let part = self.0.entry(key.clone()).or_insert_with(|| ModelPart { key: key.clone(), frequency, value: Mat::zeros(size, size), gradient: zeros(size, size) });
        let read = |m: &Mat, i: usize, j: usize| if adjoint { m[(j, i)].conj() } else { m[(i, j)] };
        for (i, &b) in rows.iter().enumerate() {
            for (j, &a) in cols.iter().enumerate() {
                if keep(b, a) {
                    part.value[(b, a)] += read(x.0, i, j) * scale;
                    for (g, d) in part.gradient.iter_mut().zip(x.1) {
                        g[(b, a)] += read(d, i, j) * scale;
                    }
                }
            }
        }
    }

    fn parts(self) -> Vec<ModelPart> {
        self.0.into_values().filter(|p| nonzero(&p.value) || p.gradient.iter().any(nonzero)).collect()
    }
}

/// How a connected part of the folded states' generator is resolved at the point (module documentation, # At a point).
enum Resolvent {
    /// Its decomposition G₀ = V diag(μ) V⁻¹.
    Spectrum(linalg::Eigen),
    /// The Neumann series about its diagonal.
    Series,
}

/// A connected part of a block of the folded states at the point: its rows (positions in the source), G₀ and ∂G₀ on it,
/// its resolvent, and the fast part of its states (None: Q, folded at every element).
struct Block {
    rows: Vec<usize>,
    g: Mat,
    changes: [Mat; 3],
    resolvent: Resolvent,
    part: Option<[i64; 2]>,
}

impl Block {
    fn new(rows: Vec<usize>, g: Mat, changes: [Mat; 3], part: Option<[i64; 2]>) -> Block {
        let resolvent = if part.is_some() { Resolvent::Series } else { Resolvent::Spectrum(Block::decompose(&g)) };
        Block { rows, g, changes, resolvent, part }
    }

    fn decompose(g: &Mat) -> linalg::Eigen {
        let mut eigen = linalg::Eigen::default();
        assert!(eigen.decompose(g), "a folded block's G₀ has no eigendecomposition (defective)");
        eigen
    }

    /// (z_j − G₀)⁻¹ y_j on each column j of y.
    fn resolve(&mut self, z: &[f64], y: &Mat) -> Mat {
        if let Resolvent::Series = self.resolvent {
            if let Some(m) = self.series(z, y) {
                return m;
            }
            self.resolvent = Resolvent::Spectrum(Block::decompose(&self.g));
        }
        let Resolvent::Spectrum(eigen) = &self.resolvent else { unreachable!() };
        let mut w = eigen.inverse.product(y);
        for k in 0..w.rows {
            for (j, &zj) in z.iter().enumerate() {
                w[(k, j)] = w[(k, j)] / (C64::real(zj) - eigen.values[k]);
            }
        }
        eigen.vectors.product(&w)
    }

    /// The Neumann series about the diagonal, each column to the rounding of its sum (module documentation); None where
    /// it does not converge (ρ ≥ 1).
    fn series(&self, z: &[f64], y: &Mat) -> Option<Mat> {
        let size = self.rows.len();
        let mut out = Mat::zeros(size, y.cols);
        let (mut w, mut next) = (vec![C64::ZERO; size], vec![C64::ZERO; size]);
        let inf = |v: &[C64]| v.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
        for (j, &zj) in z.iter().enumerate() {
            let d: Vec<C64> = (0..size).map(|i| C64::real(zj) - self.g[(i, i)]).collect();
            let rho = (0..size).map(|i| (0..size).filter(|&k| k != i).map(|k| self.g[(i, k)].abs()).sum::<f64>() / d[i].abs()).fold(0.0, f64::max);
            if rho >= 1.0 {
                return None;
            }
            for i in 0..size {
                w[i] = y[(i, j)] / d[i];
                out[(i, j)] = w[i];
            }
            let tail = rho / (1.0 - rho);
            loop {
                let sum = (0..size).fold(0.0_f64, |m, i| m.max(out[(i, j)].abs()));
                if inf(&w) * tail <= UNIT_ROUNDOFF * sum {
                    break;
                }
                for i in 0..size {
                    next[i] = (0..size).filter(|&k| k != i).map(|k| self.g[(i, k)] * w[k]).sum::<C64>() / d[i];
                }
                std::mem::swap(&mut w, &mut next);
                for i in 0..size {
                    out[(i, j)] += w[i];
                }
            }
        }
        Some(out)
    }
}

/// The connected parts of `rows` under the pattern of the matrices on them (|rows| × |rows|).
fn connected(rows: &[usize], on: &[&Mat]) -> Vec<Vec<usize>> {
    let size = rows.len();
    let mut root: Vec<usize> = (0..size).collect();
    fn find(root: &mut [usize], mut i: usize) -> usize {
        while root[i] != i {
            root[i] = root[root[i]];
            i = root[i];
        }
        i
    }
    for i in 0..size {
        for k in 0..size {
            if i != k && on.iter().any(|m| m[(i, k)] != C64::ZERO) {
                let (a, b) = (find(&mut root, i), find(&mut root, k));
                root[a.max(b)] = a.min(b);
            }
        }
    }
    let mut parts: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for i in 0..size {
        let r = find(&mut root, i);
        parts.entry(r).or_default().push(i);
    }
    parts.into_values().collect()
}

/// One component's amplitude on one block part: M_c[B, cols] and ∂M_c (|B| × |cols|), cols places in K; and the fast
/// parts n_a − n_b of its columns against K's states, the keys an out-leg makes slow elements of with it.
struct Folded {
    component: usize,
    block: usize,
    cols: Vec<usize>,
    m: Mat,
    dm: [Mat; 3],
    slow: Vec<[i64; 2]>,
    /// max|M| and the largest max|∂_i M|.
    size: [f64; 2],
}

/// A leg's block H_K,B (or L_K,B) with its gradients and max|·|, max_i max|∂_i ·|; None where all are zero.
type Leg = Option<(Mat, [Mat; 3], [f64; 2])>;

fn leg(value: &Mat, gradient: &[Mat; 3], rows: &[usize], cols: &[usize]) -> Leg {
    let out = value.block(rows, cols);
    let moved = gradient.each_ref().map(|d| d.block(rows, cols));
    let size = [out.max_abs(), moved.iter().map(Mat::max_abs).fold(0.0, f64::max)];
    (size != [0.0; 2]).then_some((out, moved, size))
}

/// Whether a pair leg·M is below the rounding of the quantity it adds to (the rounding rule, a ≥ ε): each entry of the
/// product within |B| max|leg| max|M|, of its gradient within |B| (max|∂leg| max|M| + max|leg| max|∂M|), against u
/// times the quantity's scale and its gradient's (`scale`, lower bounds: the direct terms' largest entries).
fn negligible(width: usize, leg: [f64; 2], m: [f64; 2], scale: [f64; 2]) -> bool {
    let w = width as f64;
    w * leg[0] * m[0] < UNIT_ROUNDOFF * scale[0] && w * (leg[1] * m[0] + leg[0] * m[1]) < UNIT_ROUNDOFF * scale[1]
}

/// The decay Γ = Σ_ch Σ_c L_ch⁽ᶜ⁾† L_ch⁽ᶜ⁾ of jump channels (size × size) and its gradient.
pub fn decay(jumps: &[Channel], size: usize) -> (Mat, [Mat; 3]) {
    let (mut gamma, mut changes) = (Mat::zeros(size, size), zeros(size, size));
    for p in jumps.iter().flat_map(|c| &c.parts) {
        adjoint_product_onto(&p.value, &p.value, C64::ONE, &mut gamma);
        for (d, dl) in changes.iter_mut().zip(&p.gradient) {
            if nonzero(dl) {
                adjoint_product_onto(dl, &p.value, C64::ONE, d);
                adjoint_product_onto(&p.value, dl, C64::ONE, d);
            }
        }
    }
    (gamma, changes)
}

impl ModelAt {
    /// The generator its folds are taken at: H's static part − (i/2)Γ (`decay`), with its gradient.
    pub fn generator(&self) -> ModelPart {
        let mut g = self.still().clone();
        let (gamma, changes) = decay(&self.jumps, self.states.len());
        g.value.add_scaled(C64::new(0.0, -0.5), &gamma);
        for (d, c) in g.gradient.iter_mut().zip(&changes) {
            d.add_scaled(C64::new(0.0, -0.5), c);
        }
        g
    }
}

/// The one fold (module documentation): of `source`, keep the states `keep` (indices into the 32, among the source's)
/// and their slow elements by `sectors` on the fast coordinates `axes`; fold the rest at the generator `generator` = G −
/// (i/2)Γ (on the source's states, with its gradient: `ModelAt::generator`, or the full reference's made once,
/// `Tweezer::reference`).
pub fn fold(source: &ModelAt, generator: &ModelPart, keep: &[usize], sectors: &Sectors, axes: [bool; 2]) -> ModelAt {
    let place = |state: usize| source.states.iter().position(|&s| s == state).expect("a kept state of the source");
    let mut kept: Vec<usize> = keep.iter().map(|&s| place(s)).collect();
    kept.sort_unstable();
    kept.dedup();
    let size = source.states.len();
    let others: Vec<usize> = (0..size).filter(|p| kept.binary_search(p).is_err()).collect();
    let n = kept.len();
    let part: Vec<[i64; 2]> = source.states.iter().map(|&s| sectors.state(s, axes)).collect();
    // The fast index of an element b ← a (positions in the source) of a key whose own is `keyed`.
    let fast = |keyed: [i64; 2], to: usize, from: usize| -> [i64; 2] { [keyed[0] + part[to][0] - part[from][0], keyed[1] + part[to][1] - part[from][1]] };
    let keyed: Vec<[i64; 2]> = source.effective.iter().map(|c| sectors.key(&c.key, axes)).collect();
    // The blocks of the folded states: Q, and the kept states by their fast parts; each split into its connected parts.
    let mut sets: Vec<(Vec<usize>, Option<[i64; 2]>)> = Vec::new();
    if !others.is_empty() {
        sets.push((others.clone(), None));
    }
    let mut by_part: BTreeMap<[i64; 2], Vec<usize>> = BTreeMap::new();
    for &p in &kept {
        by_part.entry(part[p]).or_default().push(p);
    }
    // A kept block is wanted where some component has a fast element into it.
    for (s, rows) in by_part {
        let wanted = keyed.iter().any(|&k| kept.iter().any(|&a| plus(k, [s[0] - part[a][0], s[1] - part[a][1]]) != [0; 2]));
        if wanted {
            sets.push((rows, Some(s)));
        }
    }
    let mut blocks: Vec<Block> = Vec::new();
    for (rows, s) in sets {
        // G₀ = G − (i/2)Γ on the rows, with its gradient.
        let g = generator.value.block(&rows, &rows);
        let changes = generator.gradient.each_ref().map(|d| d.block(&rows, &rows));
        let on: Vec<&Mat> = std::iter::once(&g).chain(changes.iter()).collect();
        for members in connected(&rows, &on) {
            let sub: Vec<usize> = members.iter().map(|&i| rows[i]).collect();
            blocks.push(Block::new(sub, g.block(&members, &members), changes.each_ref().map(|d| d.block(&members, &members)), s));
        }
    }
    let energies: Vec<f64> = kept.iter().map(|&a| generator.value[(a, a)].re).collect();
    let slopes: [Vec<f64>; 3] = std::array::from_fn(|i| kept.iter().map(|&a| generator.gradient[i][(a, a)].re).collect());
    // M_c and ∂M_c of every component on every block part it folds into.
    let mut kinds: Vec<[i64; 2]> = kept.iter().map(|&p| part[p]).collect();
    kinds.sort_unstable();
    kinds.dedup();
    let mut amplitudes: Vec<Folded> = Vec::new();
    for (index, c) in source.effective.iter().enumerate() {
        for (b, block) in blocks.iter_mut().enumerate() {
            let y = c.value.block(&block.rows, &kept);
            let moved = c.gradient.each_ref().map(|d| d.block(&block.rows, &kept));
            let cols: Vec<usize> = (0..n)
                .filter(|&j| block.part.is_none() || fast(keyed[index], block.rows[0], kept[j]) != [0; 2])
                .filter(|&j| (0..block.rows.len()).any(|i| y[(i, j)] != C64::ZERO || moved.iter().any(|d| d[(i, j)] != C64::ZERO)))
                .collect();
            if cols.is_empty() {
                continue;
            }
            let z: Vec<f64> = cols.iter().map(|&j| energies[j] + c.frequency).collect();
            let all_rows: Vec<usize> = (0..block.rows.len()).collect();
            let m = block.resolve(&z, &y.block(&all_rows, &cols));
            let dm = std::array::from_fn(|i| {
                let mut rhs = moved[i].block(&all_rows, &cols);
                rhs.add_scaled(C64::ONE, &product(&block.changes[i], &m));
                for k in 0..rhs.rows {
                    for (j, &col) in cols.iter().enumerate() {
                        rhs[(k, j)] -= m[(k, j)] * slopes[i][col];
                    }
                }
                block.resolve(&z, &rhs)
            });
            let mut slow: Vec<[i64; 2]> = Vec::new();
            for &j in &cols {
                slow.extend(kinds.iter().map(|k| [part[kept[j]][0] - k[0], part[kept[j]][1] - k[1]]));
            }
            slow.sort_unstable();
            slow.dedup();
            let size = [m.max_abs(), dm.iter().map(Mat::max_abs).fold(0.0, f64::max)];
            amplitudes.push(Folded { component: index, block: b, cols, m, dm, slow, size });
        }
    }
    let all_kept: Vec<usize> = (0..n).collect();
    let slow_pair = |keyed_sum: [i64; 2], b: usize, a: usize| fast(keyed_sum, kept[b], kept[a]) == [0; 2];
    // H_eff: the slow kept elements of each component, and each folded pair's slow half with its conjugate.
    let mut effective = Sums::default();
    let mut scale = [0.0_f64; 2];
    for (index, c) in source.effective.iter().enumerate() {
        let (block, moved) = (c.value.block(&kept, &kept), c.gradient.each_ref().map(|d| d.block(&kept, &kept)));
        scale = [scale[0].max(block.max_abs()), moved.iter().map(Mat::max_abs).fold(scale[1], f64::max)];
        effective.add(&c.key, c.frequency, n, (&block, moved.each_ref()), false, &all_kept, &all_kept, 1.0, |b, a| slow_pair(keyed[index], b, a));
    }
    // The out-legs H⁽ᶜ′⁾_K,B per component and block part, made once.
    let mut legs: BTreeMap<(usize, usize), Leg> = BTreeMap::new();
    for (index2, c2) in source.effective.iter().enumerate() {
        for folded in &amplitudes {
            let c = &source.effective[folded.component];
            let sum = plus(keyed[folded.component], keyed[index2]);
            if folded.slow.binary_search(&sum).is_err() {
                continue;
            }
            let rows = &blocks[folded.block].rows;
            let found = legs.entry((index2, folded.block)).or_insert_with(|| leg(&c2.value, &c2.gradient, &kept, rows));
            let Some((out, moved, size)) = found else { continue };
            if negligible(rows.len(), *size, folded.size, scale) {
                continue;
            }
            let a = product(out, &folded.m);
            let da: [Mat; 3] = std::array::from_fn(|i| {
                let mut d = product(&moved[i], &folded.m);
                d.add_scaled(C64::ONE, &product(out, &folded.dm[i]));
                d
            });
            let (key, frequency) = (sum_keys(&c.key, &c2.key), c.frequency + c2.frequency);
            let cols: Vec<usize> = folded.cols.clone();
            effective.add(&key, frequency, n, (&a, da.each_ref()), false, &all_kept, &cols, 0.5, |b, x| slow_pair(sum, b, x));
            effective.add(&negated(&key), -frequency, n, (&a, da.each_ref()), true, &cols, &all_kept, 0.5, |x, b| slow_pair(sum, b, x));
        }
    }
    // The jumps: each element in the channel of its fast index.
    let mut out: BTreeMap<(usize, [i64; 2]), [Sums; 2]> = BTreeMap::new();
    let mut scales = [[0.0_f64; 2]; 2];
    for channel in &source.jumps {
        for (which, parts) in [&channel.parts, &channel.kicks].into_iter().enumerate() {
            for p in parts {
                let s = &mut scales[which];
                *s = [s[0].max(p.value.block(&kept, &kept).max_abs()), p.gradient.iter().map(|d| d.block(&kept, &kept).max_abs()).fold(s[1], f64::max)];
            }
        }
    }
    for channel in &source.jumps {
        for (which, parts) in [&channel.parts, &channel.kicks].into_iter().enumerate() {
            for p in parts {
                let mut legs: BTreeMap<usize, Leg> = BTreeMap::new();
                let mut add = |key: &Key, frequency: f64, value: &Mat, gradient: [&Mat; 3], cols: &[usize]| {
                    let own = sectors.key(key, axes);
                    let mut indices: Vec<[i64; 2]> = Vec::new();
                    for b in 0..n {
                        for (j, &a) in cols.iter().enumerate() {
                            if value[(b, j)] != C64::ZERO || gradient.iter().any(|d| d[(b, j)] != C64::ZERO) {
                                indices.push(fast(own, kept[b], kept[a]));
                            }
                        }
                    }
                    indices.sort_unstable();
                    indices.dedup();
                    for index in indices {
                        let sums = &mut out.entry((channel.emission, index)).or_default()[which];
                        sums.add(key, frequency, n, (value, gradient), false, &all_kept, cols, 1.0, |b, a| fast(own, kept[b], kept[a]) == index);
                    }
                };
                let (direct, moved) = (p.value.block(&kept, &kept), p.gradient.each_ref().map(|d| d.block(&kept, &kept)));
                add(&p.key, p.frequency, &direct, moved.each_ref(), &all_kept);
                for folded in &amplitudes {
                    let rows = &blocks[folded.block].rows;
                    let found = legs.entry(folded.block).or_insert_with(|| leg(&p.value, &p.gradient, &kept, rows));
                    let Some((from, moved, size)) = found else { continue };
                    if negligible(rows.len(), *size, folded.size, scales[which]) {
                        continue;
                    }
                    let c = &source.effective[folded.component];
                    let value = product(&from, &folded.m);
                    let gradient: [Mat; 3] = std::array::from_fn(|i| {
                        let mut d = product(&moved[i], &folded.m);
                        d.add_scaled(C64::ONE, &product(&from, &folded.dm[i]));
                        d
                    });
                    add(&sum_keys(&p.key, &c.key), p.frequency + c.frequency, &value, gradient.each_ref(), &folded.cols);
                }
            }
        }
    }
    let jumps = out
        .into_iter()
        .map(|((emission, fast), [parts, kicks])| Channel { emission, fast, parts: parts.parts(), kicks: kicks.parts() })
        .filter(|c| !c.parts.is_empty())
        .collect();
    // The lift: M_c on Q's rows, in Q's order.
    let mut lift = Sums::default();
    let q = others.len();
    for folded in amplitudes.iter().filter(|f| blocks[f.block].part.is_none()) {
        let c = &source.effective[folded.component];
        let rows: Vec<usize> = blocks[folded.block].rows.iter().map(|r| others.binary_search(r).expect("a row of Q")).collect();
        let part = lift.0.entry(c.key.clone()).or_insert_with(|| ModelPart { key: c.key.clone(), frequency: c.frequency, value: Mat::zeros(q, n), gradient: zeros(q, n) });
        for (i, &row) in rows.iter().enumerate() {
            for (j, &col) in folded.cols.iter().enumerate() {
                part.value[(row, col)] = folded.m[(i, j)];
                for (g, d) in part.gradient.iter_mut().zip(&folded.dm) {
                    g[(row, col)] = d[(i, j)];
                }
            }
        }
    }
    ModelAt {
        states: kept.iter().map(|&p| source.states[p]).collect(),
        folded: others.iter().map(|&p| source.states[p]).collect(),
        groups: source.groups.clone(),
        effective: effective.parts(),
        jumps,
        lift: lift.parts(),
    }
}

impl ModelAt {
    /// The unfolded model of H's components on the 32 states (each nonzero key both ways), those whose key `fold` folds
    /// left out, and the Cartesian emission's three channels with their photons' forms.
    fn source(components: &Components, jumps: &Jumps, labels: &[String], skip: impl Fn(&Key) -> bool) -> ModelAt {
        let mut effective = Vec::new();
        for c in components.parts.iter().filter(|c| !skip(&c.key)) {
            effective.push(ModelPart { key: c.key.clone(), frequency: c.frequency, value: c.value.clone(), gradient: c.gradient.clone() });
            if c.key.iter().any(|k| *k != 0) {
                effective.push(ModelPart { key: negated(&c.key), frequency: -c.frequency, value: c.value.adjoint(), gradient: c.gradient.each_ref().map(Mat::adjoint) });
            }
        }
        let length = components.parts[0].key.len();
        let states: Vec<usize> = (0..STATES).collect();
        ModelAt { groups: levels(labels, &states), states, folded: Vec::new(), effective, jumps: emission(jumps, length), lift: Vec::new() }
    }

    /// H's static part (the zero key).
    pub fn still(&self) -> &ModelPart {
        self.effective.iter().find(|p| p.key.iter().all(|k| *k == 0)).expect("H's static part")
    }
}

/// The request's Cartesian emission as three channels (n_tw = n_φ = 0), each with its photons' form k∘C.
fn emission(jumps: &Jumps, length: usize) -> Vec<Channel> {
    let none = || zeros(STATES, STATES);
    let part = |value: Mat| ModelPart { key: vec![0; length], frequency: 0.0, value, gradient: none() };
    let photon = |c: &Mat| Mat::from_fn(STATES, STATES, |i, j| c[(i, j)] * jumps.wave_number[(i, j)].re);
    (0..3).map(|a| Channel { emission: a, fast: [0; 2], parts: vec![part(jumps.emission[a].clone())], kicks: vec![part(photon(&jumps.emission[a]))] }).collect()
}

/// The tweezer's sidebands folded once (module documentation, # The two folds): the unit fold's pairs (n_tw = 0, H's
/// part ∝ |a_tw|²) and its channels (n_tw = ±1, ∝ a_tw and a_tw*).
#[derive(Clone, Debug, PartialEq)]
pub struct Tweezer {
    sectors: Sectors,
    shift: Vec<ModelPart>,
    channels: Vec<Channel>,
    /// Σ C_q†C_q of the emission, and the unit fold's channels' Σ Ĵ†Ĵ (∝ |a_tw|²): the reference's decay.
    emission: Mat,
    scattering: Mat,
}

impl Tweezer {
    /// The fold of the tweezer's couplings at unit amplitude and at rest (`Scene::tweezer_couplings`, both ways) with the
    /// emission's channels, kept on the 32 states, at the request's H0.
    pub fn new(scene: &Scene) -> Tweezer {
        let length = scene.generators.len();
        let mut effective = Vec::new();
        for (key, frequency, x) in scene.tweezer_couplings() {
            effective.push(ModelPart { key: key.clone(), frequency, value: x.clone(), gradient: zeros(STATES, STATES) });
            effective.push(ModelPart { key: negated(&key), frequency: -frequency, value: x.adjoint(), gradient: zeros(STATES, STATES) });
        }
        let states: Vec<usize> = (0..STATES).collect();
        let source = ModelAt { groups: levels(&scene.request.states, &states), states: states.clone(), folded: Vec::new(), effective, jumps: emission(scene.jumps(), length), lift: Vec::new() };
        let emission = decay(&source.jumps, STATES).0;
        let mut generator = ModelPart { key: vec![0; length], frequency: 0.0, value: scene.request.hamiltonian.clone(), gradient: zeros(STATES, STATES) };
        generator.value.add_scaled(C64::new(0.0, -0.5), &emission);
        let sectors = Sectors::tweezer(scene);
        let unit = fold(&source, &generator, &states, &sectors, REFERENCE);
        let channels: Vec<Channel> = unit.jumps.into_iter().filter(|c| c.fast != [0; 2]).collect();
        let scattering = decay(&channels, STATES).0;
        Tweezer { shift: unit.effective, channels, sectors, emission, scattering }
    }

    /// The full reference at a point: the scene's components there, the tweezer's left out, their fold put in at its
    /// amplitude a: the pairs times |a|², the channels with n_tw = 1 times a, with −1 times a*, each with its gradient;
    /// and its generator (`ModelAt::generator`) from the decay made once, Σ C†C + |a|² Σ Ĵ†Ĵ.
    pub fn reference(&self, scene: &Scene, components: &Components) -> (ModelAt, ModelPart) {
        let tweezer = |key: &Key| self.sectors.key(key, REFERENCE) != [0; 2];
        let mut source = ModelAt::source(components, scene.jumps(), &scene.request.states, tweezer);
        let (a, da) = components.tweezer;
        let (power, dpower): (f64, [f64; 3]) = (a.norm_sqr(), std::array::from_fn(|i| 2.0 * (a.conj() * da[i]).re));
        // The frequency of a key at the point: the components' own.
        let frequency = |key: &Key, bare: f64| components.parts.iter().find(|c| &c.key == key).map_or(bare, |c| c.frequency);
        for s in &self.shift {
            let at = match source.effective.iter_mut().find(|p| p.key == s.key) {
                Some(p) => p,
                None => {
                    source.effective.push(ModelPart { key: s.key.clone(), frequency: s.frequency, value: Mat::zeros(STATES, STATES), gradient: zeros(STATES, STATES) });
                    source.effective.last_mut().unwrap()
                }
            };
            at.value.add_scaled(C64::real(power), &s.value);
            for (g, d) in at.gradient.iter_mut().zip(dpower) {
                g.add_scaled(C64::real(d), &s.value);
            }
        }
        for channel in &self.channels {
            let (scale, dscale): (C64, [C64; 3]) = if channel.fast[1] > 0 { (a, da) } else { (a.conj(), da.map(C64::conj)) };
            let put = |p: &ModelPart| ModelPart {
                key: p.key.clone(),
                frequency: frequency(&p.key, p.frequency),
                value: p.value.scaled(scale),
                gradient: dscale.map(|d| p.value.scaled(d)),
            };
            let parts: Vec<ModelPart> = channel.parts.iter().map(put).filter(|p| nonzero(&p.value) || p.gradient.iter().any(nonzero)).collect();
            if !parts.is_empty() {
                source.jumps.push(Channel { emission: channel.emission, fast: channel.fast, parts, kicks: channel.kicks.iter().map(put).collect() });
            }
        }
        let mut generator = source.still().clone();
        generator.value.add_scaled(C64::new(0.0, -0.5), &self.emission);
        generator.value.add_scaled(C64::new(0.0, -0.5 * power), &self.scattering);
        for (d, dp) in generator.gradient.iter_mut().zip(dpower) {
            d.add_scaled(C64::new(0.0, -0.5 * dp), &self.scattering);
        }
        (source, generator)
    }
}

/// The full reference for the scene's components at a point: the scene's model on the 32 states with the tweezer's
/// sidebands folded (floquet's `Hilbert::full` is it).
pub fn reference(scene: &Scene, components: &Components) -> ModelAt {
    Tweezer::new(scene).reference(scene, components).0
}

/// The RS model on a kept set P: P, its complement, the frame's sectors and the tweezer's fold, fixed for the run.
#[derive(Clone, Debug, PartialEq)]
pub struct Rs {
    /// P: indices into the 32, sorted (the ground states first).
    pub keep: Vec<usize>,
    /// Q: the folded states, the rest of the 32.
    pub folded: Vec<usize>,
    pub sectors: Sectors,
    tweezer: Tweezer,
}

impl Rs {
    /// The model's fold on P = `keep` with the frame's sectors; the tweezer's fold made here once.
    pub fn new(scene: &Scene, keep: &[usize], sectors: Sectors) -> Rs {
        let mut keep = keep.to_vec();
        keep.sort_unstable();
        keep.dedup();
        let folded = (0..STATES).filter(|i| keep.binary_search(i).is_err()).collect();
        Rs { keep, folded, sectors, tweezer: Tweezer::new(scene) }
    }

    /// The model for the scene's components at a point (`Scene::components_at`, or `Scene::components` for the
    /// checkpoint's): the full reference there, folded on P at its static part.
    pub fn at(&self, scene: &Scene, components: &Components) -> ModelAt {
        let (reference, generator) = self.tweezer.reference(scene, components);
        fold(&reference, &generator, &self.keep, &self.sectors, MODEL)
    }
}

/// The sum of a model operator's components at the frame's time zero, each at its phase there (one), or of their
/// gradients along r_i.
fn summed(parts: &[ModelPart], gradient: Option<usize>, size: usize) -> Mat {
    let mut sum = Mat::zeros(size, size);
    for p in parts {
        sum.add_scaled(C64::ONE, gradient.map_or(&p.value, |i| &p.gradient[i]));
    }
    sum
}

impl ModelAt {
    /// The force on ρ (n × n, row-major) at the frame's time zero, every component at its phase there: the adjoint
    /// contraction −Re tr(∇H_eff ρ) and the folded light's absorbed momentum Σ_ch Im tr(ρ L_ch†∇L_ch) -- the force
    /// observable of the Lindbladian, −∂H + Σ_ch (L_ch†∂L_ch − ∂L_ch†L_ch)/2i, read on ρ.
    pub fn force(&self, rho: &[C64]) -> V3 {
        let n = self.states.len();
        let trace = |x: &Mat| -> C64 { (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).map(|(a, b)| x[(a, b)] * rho[b * n + a]).sum() };
        std::array::from_fn(|i| {
            let mut f = -trace(&summed(&self.effective, Some(i), n)).re;
            for channel in &self.jumps {
                let (l, dl) = (summed(&channel.parts, None, n), summed(&channel.parts, Some(i), n));
                f += trace(&l.adjoint_product(&dl)).im;
            }
            f
        })
    }

    /// The 32 states' populations of ρ (n × n, row-major) lifted through M_c at the frame's time zero: the model's
    /// ρ_aa, Q's (M ρ M†)_kk with M = Σ_c M_c.
    pub fn lift(&self, rho: &[C64]) -> [f64; STATES] {
        let n = self.states.len();
        let mut out = [0.0; STATES];
        for (a, &state) in self.states.iter().enumerate() {
            out[state] = rho[a * n + a].re;
        }
        let mut m = Mat::zeros(self.folded.len(), n);
        for p in &self.lift {
            m.add_scaled(C64::ONE, &p.value);
        }
        let lifted = m.product(&Mat { rows: n, cols: n, data: rho.to_vec() }).product(&m.adjoint());
        for (k, &state) in self.folded.iter().enumerate() {
            out[state] = lifted[(k, k)].re;
        }
        out
    }

    /// The model as a Hilbert model for the checkpoint (floquet): H's components, each channel's components (channels do
    /// not interfere), three-axis gradients, the levels over the 32 and the lift.
    pub fn hilbert(&self) -> Hilbert {
        let part = |p: &ModelPart| Part { key: p.key.clone(), value: p.value.clone(), gradient: p.gradient.clone() };
        Hilbert {
            states: self.states.clone(),
            hamiltonian: self.effective.iter().map(part).collect(),
            jumps: self.jumps.iter().map(|c| c.parts.iter().map(part).collect()).collect(),
            groups: self.groups.clone(),
            lift: (!self.folded.is_empty()).then(|| Lift { folded: self.folded.clone(), amplitudes: self.lift.iter().map(|p| (p.key.clone(), p.value.clone())).collect() }),
        }
    }
}

#[cfg(test)]
mod tests;
