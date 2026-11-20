//! The one fold (RUSTPLAN §2.1–§2.3 and its corrections of 2026-11-19: "模型只定一次", "一个折叠公式，快分量与光镊都由它
//! 折叠"): Reiter–Sørensen second order with harmonics and dissipation (td_s.hilbert_rs, each column at E_a + f), the
//! full reference's and the model's alike -- they differ only in what is folded and at which generator.
//!
//! # The fold
//! A Lindblad model on a set of states by frequency components: H's components H⁽ᶜ⁾ e^{−if_c t} taken both ways (each
//! nonzero key with its conjugate at −key), and jump channels L_ch = Σ_c L_ch⁽ᶜ⁾ e^{−if_c t}. A fold keeps the states K
//! and, among their elements, the slow ones: the element b ← a of the component c has the fast index n = n_c − n_b + n_a
//! (`Sectors::element`, the fast part of its frame vector c − θ_b + θ_a, floquet/frame.rs `element`, the assembly's own;
//! n_c = Σ_l c_l n_l over the key's generators, n_a the state's, the frame's integer fast parts (n_φ, n_tw)), slow where
//! its folding coordinates (`axes`) are zero. Everything else is folded: every component of
//! the other states F (= Q), and the fast elements within K (K's sidebands -- the hyperfine n_φ and the tweezer n_tw).
//! The folded states' unperturbed generator is block diagonal,
//!   G₀ = G_BB − (i/2) Γ_BB,  Γ = Σ_ch Σ_c L_ch⁽ᶜ⁾† L_ch⁽ᶜ⁾  (the static decay),
//! G the static generator the fold is taken at, on the blocks B: Q, and K's states by their fast parts (G and Γ do not
//! couple two of them: a static element between them is fast and folded). Column a of K at its energy E_a = Re G_aa and
//! the component's frequency,
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
//!   in, and every key's frequency is the point's (scene.rs `frequency`: Σ_g key_g f_g at the point's Doppler shifts).
//!   Six channels: the three emission axes, co- and counter-rotating, apart from the emission's (n_tw = 0).
//! - The model (`Rs`): the reference folded again on P at its static part, every fast element folded (axes n_φ, n_tw): Q
//!   at every component, P's hyperfine sidebands.
//!
//! # What is made once and what at a point
//! Every operator of the fold is a sum of terms, a fixed matrix times a factor of the beams' amplitudes at the point
//! (`Scene::terms`, scene.rs `Point`; the tweezer fold's own, times |a_tw|², a_tw, a_tw*). The fixed matrices are made
//! once (`Rs::new`) by where their entries fall -- K × K, Q × K, K × Q, Q × Q (`Split`) --; a point (`Rs::local`) puts in
//! the factors and makes the one thing the formula itself asks there: the folded blocks' resolvents at the local
//! generator (the tweezer shifts Q's levels by the order of the detunings, so nothing fixed is near them).
//!
//! # At a point: which block how
//! The folded states' generator at a point splits into its connected parts (the pattern of G₀ and ∂G₀); each part's
//! resolvent is taken one of two ways, both to the rounding of the formula above:
//! - Q's parts: decomposed at the point, G₀ = V diag(μ) V⁻¹ (`linalg::Eigen`), every column's resolvent that one
//!   spectrum: in its basis M̂_c = V⁻¹M_c = D_c ∘ (V⁻¹H⁽ᶜ⁾_QK), D_c[k, a] = 1/(E_a + f_c − μ_k), and the out-legs
//!   W_c′ = H⁽ᶜ′⁾_KQ V, so that H⁽ᶜ′⁾_KQ M_c = W_c′ M̂_c; the gradient ∂M̂ = V⁻¹∂M_c = D_c ∘ (V⁻¹∂H⁽ᶜ⁾ + Ĝ M̂_c −
//!   M̂_c ∂E), Ĝ = V⁻¹∂G₀V (STEPBENCH's kernel, stepbench.status.md §4: Y_l = V⁻¹X_l, W_l = X_l†V).
//! - K's sideband parts: their denominators are a fast frequency -- a hyperfine φ ≈ 4·10⁴ rad/µs, or the tweezer's 10⁸ --
//!   off the local energies, whose couplings within the part (Zeeman, the outside forms' tensor shifts, the tweezer's) are
//!   ≲ 10² rad/µs. Each column's resolvent is the Neumann series about the part's diagonal at the point,
//!   (D_z − N)⁻¹ = Σ_k (D_z⁻¹N)^k D_z⁻¹, D_z = z − diag G₀ (exact at the point), N its off-diagonal: ρ = ‖D_z⁻¹N‖_∞ ≈
//!   2·10⁻³ and less, and the series is summed until its remainder's bound ‖w_k‖ ρ/(1 − ρ) is below the rounding of the
//!   column's sum, u‖M‖_∞ -- the count the ratio's own (a few terms; one where the part is diagonal, as the tweezer's bare
//!   levels are). No decomposition. Where ρ ≥ 1 the series does not converge and the part is decomposed as Q's (its
//!   results taken back to the states' basis, the part's own).
//!
//! # Two contractions of one fold (`Local`)
//! - By frequency components (`Local::components`, the checkpoint's: the Floquet lattice needs each key apart): every
//!   pair W_c′ M̂_c at its key c + c′, every channel's element at its fast index.
//! - The value at a time t (`Local::now`, the trajectory's): Σ_c (·)⁽ᶜ⁾ e^{−if_c t}, summed before the products --
//!   Z_γ(t) = Σ_{c: n_c = γ} e^{−if_c t} M̂_c and W_γ′(t) = Σ_{c′: n_c′ = γ′} e^{−if_c′ t} W_c′, the products W_γ′ Z_γ by
//!   classes of the fast index, each element kept where it is slow (γ + γ′ − n_b + n_a = 0): the pairs' count is the
//!   classes', not the components', so a moving atom, every beam its own component at its own Doppler shift, costs the
//!   components' sums alone. The jumps the same way, each element into the channel of its fast index π + γ − n_b + n_a.
//!
//! The two are the same sums in two orders; `now` equals Σ components · e^{−ift} to rounding (tests,
//! `now_is_the_components_at_their_phases`). Against td_s.hilbert_rs's column-by-column inverse the fold is the same
//! formula to the rounding of the decomposition, κ(V)u relative (tests, `the_fold_is_hilbert_rs_to_rounding`).

use std::collections::BTreeMap;

use crate::floquet::{Hilbert, Lift, Part, Sectors};
use crate::linalg::{self, C64, Mat, V3};
use crate::request::STATES;
use crate::scene::{Components, Jumps, Key, Point, Scene};
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

/// A Lindblad model at a point by frequency components: what a fold gives by components (`Local::components`).
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

/// out (a.rows × b.cols) = a b (`linalg::fused_product`).
fn product(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows);
    let mut out = Mat::zeros(a.rows, b.cols);
    linalg::fused_product(&a.data, &b.data, (a.rows, a.cols, b.cols), &mut out.data);
    out
}

/// out += a b.
fn product_onto(a: &Mat, b: &Mat, out: &mut Mat) {
    assert_eq!((a.cols, a.rows, b.cols), (b.rows, out.rows, out.cols));
    linalg::fused_product_onto(&a.data, &b.data, (a.rows, a.cols, b.cols), &mut out.data);
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

// ---------------------------------------------------------------------------------------------------------------------
// The fixed matrices and the source at a point.

/// A fixed matrix on the 32 states by its nonzero entries (row, column, value).
pub type Fixed = Vec<(usize, usize, C64)>;

/// A dense matrix's nonzero entries.
pub fn fixed(m: &Mat) -> Fixed {
    (0..m.rows).flat_map(|i| (0..m.cols).map(move |j| (i, j))).filter(|&(i, j)| m[(i, j)] != C64::ZERO).map(|(i, j)| (i, j, m[(i, j)])).collect()
}

/// A fixed matrix's entries by where they fall in a fold of K and Q: K × K (positions in K) by the fast-part difference
/// δ = n_b − n_a of their row b and column a (an element of a component of fast index γ is slow where δ = γ,
/// `Sectors::element`: so a contraction takes whole groups), Q × K (Q's index, K's position), K × Q, Q × Q.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Split {
    kk: Vec<([i64; 2], Fixed)>,
    qk: Fixed,
    kq: Fixed,
    qq: Fixed,
}

impl Split {
    /// `kept[s]`, `others[s]`: state s's position in K or index in Q (usize::MAX where it is not one); `parts` K's fast
    /// parts.
    fn new(entries: &Fixed, kept: &[usize; STATES], others: &[usize; STATES], parts: &[[i64; 2]]) -> Split {
        let mut split = Split::default();
        for &(i, j, v) in entries {
            match (kept[i] != usize::MAX, kept[j] != usize::MAX) {
                (true, true) => {
                    let (b, a) = (kept[i], kept[j]);
                    let delta = [parts[b][0] - parts[a][0], parts[b][1] - parts[a][1]];
                    match split.kk.iter_mut().find(|g| g.0 == delta) {
                        Some(g) => g.1.push((b, a, v)),
                        None => split.kk.push((delta, vec![(b, a, v)])),
                    }
                }
                (false, true) => split.qk.push((others[i], kept[j], v)),
                (true, false) => split.kq.push((kept[i], others[j], v)),
                (false, false) => split.qq.push((others[i], others[j], v)),
            }
        }
        split
    }

    /// Every fixed matrix's split for the kept states `keep` (indices into the 32, sorted), the rest, and the kept
    /// states' fast parts by `sectors` on `axes`.
    pub fn all(library: &[Fixed], keep: &[usize], sectors: &Sectors, axes: [bool; 2]) -> Vec<Split> {
        let (mut kept, mut others) = ([usize::MAX; STATES], [usize::MAX; STATES]);
        let mut q = 0;
        for s in 0..STATES {
            match keep.binary_search(&s) {
                Ok(p) => kept[s] = p,
                Err(_) => {
                    others[s] = q;
                    q += 1;
                }
            }
        }
        let parts: Vec<[i64; 2]> = keep.iter().map(|&s| sectors.state(s, axes)).collect();
        library.iter().map(|m| Split::new(m, &kept, &others, &parts)).collect()
    }
}

/// One term of a source operator at a point: a fixed matrix (an index into the fold's library; its adjoint where
/// `adjoint`) times `scale`, with ∇scale.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    pub matrix: usize,
    pub adjoint: bool,
    pub scale: C64,
    pub gradient: [C64; 3],
}

impl Term {
    pub fn one(matrix: usize, scale: C64) -> Term {
        Term { matrix, adjoint: false, scale, gradient: [C64::ZERO; 3] }
    }

    /// (s F)† = s* F†.
    fn adjoint(self) -> Term {
        Term { matrix: self.matrix, adjoint: !self.adjoint, scale: self.scale.conj(), gradient: self.gradient.map(C64::conj) }
    }

    fn vanishes(&self) -> bool {
        self.scale == C64::ZERO && self.gradient.iter().all(|g| *g == C64::ZERO)
    }
}

/// A frequency component of a source operator: Σ terms at `key`'s frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Piece {
    pub key: Key,
    pub frequency: f64,
    pub terms: Vec<Term>,
}

impl Piece {
    pub fn adjoint(&self) -> Piece {
        Piece { key: negated(&self.key), frequency: -self.frequency, terms: self.terms.iter().map(|t| t.adjoint()).collect() }
    }
}

/// A source jump channel: its emission axis, its fast index, its components and its photons' forms.
#[derive(Clone, Debug, PartialEq)]
pub struct SourceChannel {
    pub emission: usize,
    pub fast: [i64; 2],
    pub parts: Vec<Piece>,
    pub kicks: Vec<Piece>,
}

/// What a fold folds, at a point, on the 32 states: H's components (each nonzero key both ways), the jump channels, and
/// the terms of the generator the fold is taken at, G − (i/2)Γ.
#[derive(Clone, Debug, PartialEq)]
pub struct Source {
    pub hamiltonian: Vec<Piece>,
    pub jumps: Vec<SourceChannel>,
    pub generator: Vec<Term>,
}

/// The dense sum of terms on the 32 states, and of their gradients.
fn dense(library: &[Fixed], terms: &[Term]) -> (Mat, [Mat; 3]) {
    let (mut value, mut gradient) = (Mat::zeros(STATES, STATES), zeros(STATES, STATES));
    for t in terms {
        for &(i, j, v) in &library[t.matrix] {
            let (i, j, v) = if t.adjoint { (j, i, v.conj()) } else { (i, j, v) };
            value[(i, j)] += t.scale * v;
            for (g, d) in gradient.iter_mut().zip(t.gradient) {
                g[(i, j)] += d * v;
            }
        }
    }
    (value, gradient)
}

impl Source {
    /// The source as a dense model on the 32 states (the full reference's `ModelAt`; floquet's `Hilbert::full`) and its
    /// generator.
    pub fn model(&self, library: &[Fixed], groups: &[Vec<usize>]) -> (ModelAt, ModelPart) {
        let part = |p: &Piece| {
            let (value, gradient) = dense(library, &p.terms);
            ModelPart { key: p.key.clone(), frequency: p.frequency, value, gradient }
        };
        let jumps = self.jumps.iter().map(|c| Channel { emission: c.emission, fast: c.fast, parts: c.parts.iter().map(part).collect(), kicks: c.kicks.iter().map(part).collect() }).collect();
        let model = ModelAt { states: (0..STATES).collect(), folded: Vec::new(), groups: groups.to_vec(), effective: self.hamiltonian.iter().map(part).collect(), jumps, lift: Vec::new() };
        let (value, gradient) = dense(library, &self.generator);
        let length = self.hamiltonian.first().map_or(0, |p| p.key.len());
        (model, ModelPart { key: vec![0; length], frequency: 0.0, value, gradient })
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The fold at a point.

/// An entry routed into a fold: row, column (positions in the sets it falls in), value, gradient.
type Entry = (usize, usize, C64, [C64; 3]);

/// A folded block's resolvent (module documentation, # At a point).
#[derive(Clone, Debug)]
enum Basis {
    /// G₀ = V diag(μ) V⁻¹ with V⁻¹'s columns as rows (V⁻¹ᵀ) and Ĝ_iᵀ: the amplitudes and legs in V's basis.
    Spectrum { eigen: linalg::Eigen, inverse: Mat, changes: [Mat; 3] },
    /// The Neumann series about the diagonal, in the states' basis: G₀'s off-diagonal part and its rows' sums of moduli;
    /// the decomposition made where a series does not converge.
    Series { off: Vec<(usize, usize, C64)>, spread: Vec<f64>, eigen: Option<linalg::Eigen> },
}

/// A connected part of the folded states at the point: its rows (Q's indices, or K's positions for a sideband part), its
/// fast part (None: Q, every element folded; Some(s): K's states of fast part s), G₀ and ∂_i G₀ on it (the latter in
/// its basis for a decomposed part, V⁻¹∂G₀V), its resolvent.
#[derive(Clone, Debug)]
struct Block {
    rows: Vec<usize>,
    part: Option<[i64; 2]>,
    g: Mat,
    changes: [Mat; 3],
    basis: Basis,
}

fn transpose(m: &Mat) -> Mat {
    Mat::from_fn(m.cols, m.rows, |i, j| m[(j, i)])
}

impl Block {
    fn new(rows: Vec<usize>, part: Option<[i64; 2]>, g: Mat, dg: [Mat; 3]) -> Block {
        let size = rows.len();
        if part.is_some() {
            let off: Vec<(usize, usize, C64)> = (0..size).flat_map(|i| (0..size).map(move |k| (i, k))).filter(|&(i, k)| i != k && g[(i, k)] != C64::ZERO).map(|(i, k)| (i, k, g[(i, k)])).collect();
            let mut spread = vec![0.0; size];
            for &(i, _, v) in &off {
                spread[i] += v.abs();
            }
            return Block { rows, part, g, changes: dg, basis: Basis::Series { off, spread, eigen: None } };
        }
        let eigen = Block::decompose(&g);
        let changes = dg.each_ref().map(|d| product(&eigen.inverse, &product(d, &eigen.vectors)));
        let basis = Basis::Spectrum { inverse: transpose(&eigen.inverse), changes: changes.each_ref().map(transpose), eigen };
        Block { rows, part, g, changes, basis }
    }

    fn decompose(g: &Mat) -> linalg::Eigen {
        let mut eigen = linalg::Eigen::default();
        assert!(eigen.decompose(g), "a folded block's G₀ has no eigendecomposition (defective)");
        eigen
    }

    /// M̂ and ∂M̂ (|B| × |cols|, the block's basis) of a component at the columns `cols` (positions in K) at the
    /// denominators z_j = E_cols[j] + f: `entries` H⁽ᶜ⁾_B,cols (row in the block, column's index in cols).
    fn amplitude(&mut self, z: &[f64], slopes: [&[f64]; 3], entries: &[Entry]) -> (Mat, [Mat; 3]) {
        let (size, count) = (self.rows.len(), z.len());
        if let Basis::Spectrum { eigen, inverse, changes } = &self.basis {
            // By columns (each a row of the transposes): V⁻¹H and V⁻¹∂H, D∘, then Ĝ M̂ − M̂ ∂E and D∘.
            let (mut y, mut x) = (Mat::zeros(count, size), zeros(count, size));
            for &(r, j, v, dv) in entries {
                let column = &inverse.data[r * size..(r + 1) * size];
                linalg::mul_add(false, &mut y.data[j * size..(j + 1) * size], v, column);
                for (xi, d) in x.iter_mut().zip(dv) {
                    if d != C64::ZERO {
                        linalg::mul_add(false, &mut xi.data[j * size..(j + 1) * size], d, column);
                    }
                }
            }
            // 1/(μ_k − z_j), the denominators' negatives.
            let mut d = Mat::zeros(count, size);
            for j in 0..count {
                linalg::reciprocals(&eigen.values, z[j], C64::ZERO, &mut d.data[j * size..(j + 1) * size]);
            }
            let m = Mat { rows: count, cols: size, data: y.data.iter().zip(&d.data).map(|(a, b)| -(*a * *b)).collect() };
            let dm = std::array::from_fn(|i| {
                let mut t = std::mem::take(&mut x[i]);
                product_onto(&m, &changes[i], &mut t);
                for j in 0..count {
                    let slope = slopes[i][j];
                    for k in 0..size {
                        let at = j * size + k;
                        t.data[at] = -((t.data[at] - m.data[at] * slope) * d.data[at]);
                    }
                }
                transpose(&t)
            });
            return (transpose(&m), dm);
        }
        // The value's right-hand side, then the gradients' three side by side (one series for the three).
        let (mut y, mut x) = (Mat::zeros(size, count), Mat::zeros(size, 3 * count));
        for &(r, j, v, dv) in entries {
            y[(r, j)] += v;
            for (i, d) in dv.into_iter().enumerate() {
                x[(r, i * count + j)] += d;
            }
        }
        let m = self.resolve(z, &y);
        for i in 0..3 {
            let moved = product(&self.changes[i], &m);
            for k in 0..size {
                for j in 0..count {
                    x[(k, i * count + j)] += moved[(k, j)] - m[(k, j)] * slopes[i][j];
                }
            }
        }
        let thrice: Vec<f64> = (0..3).flat_map(|_| z.iter().copied()).collect();
        let all = self.resolve(&thrice, &x);
        let dm = std::array::from_fn(|i| Mat::from_fn(size, count, |k, j| all[(k, i * count + j)]));
        (m, dm)
    }

    /// A series part's (z_j − G₀)⁻¹ y_j on each column j of y, in the states' basis: the Neumann series about the
    /// diagonal (module documentation), every column summed until its remainder's bound is below the rounding of its
    /// sum; where it does not converge (ρ ≥ 1), through the part's decomposition.
    fn resolve(&mut self, z: &[f64], y: &Mat) -> Mat {
        let Basis::Series { off, spread, eigen } = &mut self.basis else { unreachable!("a sideband part's resolvent") };
        let (size, count) = (y.rows, y.cols);
        let inverse = Mat::from_fn(size, count, |i, j| (C64::real(z[j]) - self.g[(i, i)]).recip());
        let tail: Vec<f64> = (0..count)
            .map(|j| {
                let rho = (0..size).map(|i| spread[i] * inverse[(i, j)].abs()).fold(0.0, f64::max);
                if rho < 1.0 { rho / (1.0 - rho) } else { f64::INFINITY }
            })
            .collect();
        if tail.iter().all(|t| t.is_finite()) {
            let mut w = Mat { rows: size, cols: count, data: y.data.iter().zip(&inverse.data).map(|(a, b)| *a * *b).collect() };
            let mut out = w.clone();
            let largest = |m: &Mat, j: usize| (0..size).fold(0.0_f64, |s, i| s.max(m[(i, j)].abs()));
            while (0..count).any(|j| largest(&w, j) * tail[j] > UNIT_ROUNDOFF * largest(&out, j)) {
                let mut next = Mat::zeros(size, count);
                for &(i, k, v) in off.iter() {
                    linalg::mul_add(false, &mut next.data[i * count..(i + 1) * count], v, &w.data[k * count..(k + 1) * count]);
                }
                w = next;
                for (a, b) in w.data.iter_mut().zip(&inverse.data) {
                    *a *= *b;
                }
                out.add_scaled(C64::ONE, &w);
            }
            return out;
        }
        let eigen = eigen.get_or_insert_with(|| Block::decompose(&self.g));
        let mut w = eigen.inverse.product(y);
        for k in 0..size {
            for (j, &zj) in z.iter().enumerate() {
                w[(k, j)] = w[(k, j)] / (C64::real(zj) - eigen.values[k]);
            }
        }
        eigen.vectors.product(&w)
    }

    /// An out-leg into a decomposed part, W = H_rows,B V, and its gradient's ∂H_rows,B V: `entries` H_rows,B (row's index
    /// in rows, row in the block).
    fn leg(&self, rows: usize, entries: &[Entry]) -> (Mat, [Mat; 3]) {
        let Basis::Spectrum { eigen, .. } = &self.basis else { unreachable!("a leg into Q") };
        let size = self.rows.len();
        let (mut w, mut dw) = (Mat::zeros(rows, size), zeros(rows, size));
        for &(i, c, v, dv) in entries {
            let basis = &eigen.vectors.data[c * size..(c + 1) * size];
            linalg::mul_add(false, &mut w.data[i * size..(i + 1) * size], v, basis);
            for (d, x) in dw.iter_mut().zip(dv) {
                if x != C64::ZERO {
                    linalg::mul_add(false, &mut d.data[i * size..(i + 1) * size], x, basis);
                }
            }
        }
        (w, dw)
    }

    /// The states' amplitudes V m of amplitudes m in the block's basis.
    fn back(&self, m: &Mat) -> Mat {
        match &self.basis {
            Basis::Spectrum { eigen, .. } => product(&eigen.vectors, m),
            Basis::Series { .. } => m.clone(),
        }
    }
}

/// The connected parts of `size` rows under the pattern of the matrices on them (size × size).
fn connected(size: usize, on: &[&Mat]) -> Vec<Vec<usize>> {
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

/// A component's amplitude on one block: its piece, M̂ and ∂M̂ (|B| × |cols|, the block's basis), cols positions in K.
#[derive(Clone, Debug)]
struct Amplitude {
    piece: usize,
    block: usize,
    cols: Vec<usize>,
    m: Mat,
    dm: [Mat; 3],
}

/// An out-leg on one block: W = H_rows,B V and ∂H_rows,B V (|rows| × |B|; V the identity on a sideband part), rows
/// positions in K.
#[derive(Clone, Debug)]
struct Leg {
    block: usize,
    rows: Vec<usize>,
    w: Mat,
    dw: [Mat; 3],
}

/// A component at the point: its key, frequency, the fast index of its key, its terms.
#[derive(Clone, Debug)]
struct Held {
    key: Key,
    frequency: f64,
    class: [i64; 2],
    terms: Vec<Term>,
}

/// A jump channel's component at the point (`which`: 0 the channel's, 1 its photons' form), with its out-legs into Q.
#[derive(Clone, Debug)]
struct JumpPart {
    emission: usize,
    which: usize,
    held: Held,
    legs: Vec<Leg>,
}

/// The fold at a point (module documentation): what both contractions read -- the folded blocks with their resolvents,
/// every component's amplitudes and out-legs into Q, and the components' terms, whose K × K entries the contractions read
/// from the fixed matrices' splits (`splits`) as they go.
#[derive(Clone, Debug)]
pub struct Local<'a> {
    splits: &'a [Split],
    /// K and Q (indices into the 32), the levels.
    pub states: Vec<usize>,
    pub folded: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
    /// K's states' fast parts.
    parts: Vec<[i64; 2]>,
    pieces: Vec<Held>,
    blocks: Vec<Block>,
    /// Each K position's sideband block and its row there (usize::MAX where none).
    sideband: Vec<(usize, usize)>,
    amplitudes: Vec<Amplitude>,
    /// H's out-legs into Q's parts with their pieces.
    legs: Vec<(usize, Leg)>,
    jumps: Vec<JumpPart>,
}

/// Routes a component's terms' entries into a fold, by block: the amplitudes' (`amplitude`: row in the block, K's
/// column) -- Q × K, and the K × K entries that are fast (δ ≠ `class`) into the row's sideband part --, and the out-legs'
/// into Q (`leg`: K's row, column in the block). No amplitudes where `class` is None (a jump channel's component).
fn route(splits: &[Split], terms: &[Term], class: Option<[i64; 2]>, q: &[(usize, usize)], sideband: &[(usize, usize)], amplitude: &mut [Vec<Entry>], leg: &mut [Vec<Entry>]) {
    for t in terms {
        let split = &splits[t.matrix];
        let scaled = |v: C64| -> (C64, [C64; 3]) {
            let v = if t.adjoint { v.conj() } else { v };
            (t.scale * v, t.gradient.map(|g| g * v))
        };
        let (into, out) = if t.adjoint { (&split.kq, &split.qk) } else { (&split.qk, &split.kq) };
        if let Some(class) = class {
            for (delta, list) in &split.kk {
                let delta = if t.adjoint { [-delta[0], -delta[1]] } else { *delta };
                if delta == class {
                    continue;
                }
                for &(i, j, v) in list {
                    let (b, a) = if t.adjoint { (j, i) } else { (i, j) };
                    let (block, r) = sideband[b];
                    if block != usize::MAX {
                        let (x, dx) = scaled(v);
                        amplitude[block].push((r, a, x, dx));
                    }
                }
            }
            for &(e0, e1, v) in into {
                // Q × K: (q, a), or, adjoint, the K × Q entry (a, q) transposed.
                let (qi, a) = if t.adjoint { (e1, e0) } else { (e0, e1) };
                let (block, r) = q[qi];
                let (x, dx) = scaled(v);
                amplitude[block].push((r, a, x, dx));
            }
        }
        for &(e0, e1, v) in out {
            let (b, qi) = if t.adjoint { (e1, e0) } else { (e0, e1) };
            let (block, c) = q[qi];
            let (x, dx) = scaled(v);
            leg[block].push((b, c, x, dx));
        }
    }
}

/// The distinct rows (K positions) of out-leg entries, sorted, each entry's row replaced by its index among them.
fn compact(entries: &mut [Entry], place: &mut [usize]) -> Vec<usize> {
    let mut rows: Vec<usize> = entries.iter().map(|e| e.0).collect();
    rows.sort_unstable();
    rows.dedup();
    for (i, &b) in rows.iter().enumerate() {
        place[b] = i;
    }
    for e in entries.iter_mut() {
        e.0 = place[e.0];
    }
    rows
}

/// The one fold (module documentation) of `source` at a point: keep the states `keep` (indices into the 32, sorted) and
/// their slow elements by `sectors` on the fast coordinates `axes`; fold the rest, Q = `folded`, at the source's
/// generator. `splits` the library's fixed matrices split for this K and Q (`Split::all`, on the same sectors and axes).
pub fn fold<'a>(splits: &'a [Split], keep: &[usize], folded: &[usize], source: &Source, sectors: &Sectors, axes: [bool; 2], groups: &[Vec<usize>]) -> Local<'a> {
    let (n, nq) = (keep.len(), folded.len());
    let parts: Vec<[i64; 2]> = keep.iter().map(|&s| sectors.state(s, axes)).collect();
    // G₀ on Q × Q and K × K with its gradient.
    let (mut gq, mut dgq, mut gk, mut dgk) = (Mat::zeros(nq, nq), zeros(nq, nq), Mat::zeros(n, n), zeros(n, n));
    for t in &source.generator {
        let split = &splits[t.matrix];
        let put = |g: &mut Mat, dg: &mut [Mat; 3], (i, j, v): (usize, usize, C64)| {
            let (i, j, v) = if t.adjoint { (j, i, v.conj()) } else { (i, j, v) };
            g[(i, j)] += t.scale * v;
            for (d, x) in dg.iter_mut().zip(t.gradient) {
                d[(i, j)] += x * v;
            }
        };
        for &e in &split.qq {
            put(&mut gq, &mut dgq, e);
        }
        for &e in split.kk.iter().flat_map(|g| &g.1) {
            put(&mut gk, &mut dgk, e);
        }
    }
    let energies: Vec<f64> = (0..n).map(|a| gk[(a, a)].re).collect();
    let slopes: [Vec<f64>; 3] = std::array::from_fn(|i| (0..n).map(|a| dgk[i][(a, a)].re).collect());
    let classes: Vec<[i64; 2]> = source.hamiltonian.iter().map(|p| sectors.key(&p.key, axes)).collect();
    // The blocks: Q's connected parts; K's states by fast part, where some component has a fast element into them, each
    // split into its connected parts.
    let mut blocks: Vec<Block> = Vec::new();
    let (mut q, mut sideband) = (vec![(usize::MAX, usize::MAX); nq], vec![(usize::MAX, usize::MAX); n]);
    for members in connected(nq, &[&gq, &dgq[0], &dgq[1], &dgq[2]]) {
        for (r, &i) in members.iter().enumerate() {
            q[i] = (blocks.len(), r);
        }
        blocks.push(Block::new(members.clone(), None, gq.block(&members, &members), dgq.each_ref().map(|d| d.block(&members, &members))));
    }
    let mut by_part: BTreeMap<[i64; 2], Vec<usize>> = BTreeMap::new();
    for (a, &s) in parts.iter().enumerate() {
        by_part.entry(s).or_default().push(a);
    }
    for (s, rows) in by_part {
        if !classes.iter().any(|&k| parts.iter().any(|&na| Sectors::element(k, s, na) != [0; 2])) {
            continue;
        }
        let g = gk.block(&rows, &rows);
        let dg = dgk.each_ref().map(|d| d.block(&rows, &rows));
        for members in connected(rows.len(), &[&g, &dg[0], &dg[1], &dg[2]]) {
            let sub: Vec<usize> = members.iter().map(|&i| rows[i]).collect();
            for (r, &a) in sub.iter().enumerate() {
                sideband[a] = (blocks.len(), r);
            }
            blocks.push(Block::new(sub.clone(), Some(s), gk.block(&sub, &sub), dgk.each_ref().map(|d| d.block(&sub, &sub))));
        }
    }
    let nb = blocks.len();
    let (mut amp, mut leg): (Vec<Vec<Entry>>, Vec<Vec<Entry>>) = (vec![Vec::new(); nb], vec![Vec::new(); nb]);
    // H's components: their amplitudes and out-legs into Q.
    let mut pieces = Vec::with_capacity(source.hamiltonian.len());
    let (mut amplitudes, mut legs) = (Vec::new(), Vec::new());
    let mut place = vec![usize::MAX; n];
    for (index, piece) in source.hamiltonian.iter().enumerate() {
        let class = classes[index];
        let terms: Vec<Term> = piece.terms.iter().filter(|t| !t.vanishes()).copied().collect();
        amp.iter_mut().chain(leg.iter_mut()).for_each(Vec::clear);
        route(splits, &terms, Some(class), &q, &sideband, &mut amp, &mut leg);
        for (b, entries) in amp.iter_mut().enumerate().filter(|(_, e)| !e.is_empty()) {
            let mut cols: Vec<usize> = entries.iter().map(|e| e.1).collect();
            cols.sort_unstable();
            cols.dedup();
            for (j, &a) in cols.iter().enumerate() {
                place[a] = j;
            }
            for e in entries.iter_mut() {
                e.1 = place[e.1];
            }
            let z: Vec<f64> = cols.iter().map(|&a| energies[a] + piece.frequency).collect();
            let s: [Vec<f64>; 3] = std::array::from_fn(|i| cols.iter().map(|&a| slopes[i][a]).collect());
            let (m, dm) = blocks[b].amplitude(&z, s.each_ref().map(Vec::as_slice), entries);
            amplitudes.push(Amplitude { piece: index, block: b, cols, m, dm });
        }
        for (b, entries) in leg.iter_mut().enumerate().filter(|(_, e)| !e.is_empty()) {
            let rows = compact(entries, &mut place);
            let (w, dw) = blocks[b].leg(rows.len(), entries);
            legs.push((index, Leg { block: b, rows, w, dw }));
        }
        pieces.push(Held { key: piece.key.clone(), frequency: piece.frequency, class, terms });
    }
    // The jumps' components: their out-legs into Q.
    let mut jumps = Vec::new();
    for channel in &source.jumps {
        for (which, list) in [&channel.parts, &channel.kicks].into_iter().enumerate() {
            for piece in list {
                let terms: Vec<Term> = piece.terms.iter().filter(|t| !t.vanishes()).copied().collect();
                leg.iter_mut().for_each(Vec::clear);
                route(splits, &terms, None, &q, &sideband, &mut amp, &mut leg);
                let mut out = Vec::new();
                for (b, entries) in leg.iter_mut().enumerate().filter(|(_, e)| !e.is_empty()) {
                    let rows = compact(entries, &mut place);
                    let (w, dw) = blocks[b].leg(rows.len(), entries);
                    out.push(Leg { block: b, rows, w, dw });
                }
                let held = Held { key: piece.key.clone(), frequency: piece.frequency, class: sectors.key(&piece.key, axes), terms };
                jumps.push(JumpPart { emission: channel.emission, which, held, legs: out });
            }
        }
    }
    Local { splits, states: keep.to_vec(), folded: folded.to_vec(), groups: groups.to_vec(), parts, pieces, blocks, sideband, amplitudes, legs, jumps }
}

/// Model parts by key, each rows × cols, in the keys' order.
struct Sums {
    rows: usize,
    cols: usize,
    parts: BTreeMap<Key, ModelPart>,
}

impl Sums {
    fn new(rows: usize, cols: usize) -> Sums {
        Sums { rows, cols, parts: BTreeMap::new() }
    }

    fn part(&mut self, key: &Key, frequency: f64) -> &mut ModelPart {
        let (rows, cols) = (self.rows, self.cols);
        self.parts.entry(key.clone()).or_insert_with(|| ModelPart { key: key.clone(), frequency, value: Mat::zeros(rows, cols), gradient: zeros(rows, cols) })
    }

    fn done(self) -> Vec<ModelPart> {
        self.parts.into_values().filter(|p| nonzero(&p.value) || p.gradient.iter().any(nonzero)).collect()
    }
}

fn add(value: &mut Mat, gradient: &mut [Mat; 3], (i, j): (usize, usize), x: C64, dx: [C64; 3]) {
    value[(i, j)] += x;
    for (g, d) in gradient.iter_mut().zip(dx) {
        g[(i, j)] += d;
    }
}

/// W M̂ and ∂(W M̂) = ∂W M̂ + W ∂M̂ of an out-leg and an amplitude (or their class sums) on one block.
fn pair(w: &Mat, dw: &[Mat; 3], m: &Mat, dm: &[Mat; 3]) -> (Mat, [Mat; 3]) {
    let p = product(w, m);
    let dp = std::array::from_fn(|i| {
        let mut d = product(&dw[i], m);
        product_onto(w, &dm[i], &mut d);
        d
    });
    (p, dp)
}

/// A class sum on one block: its class, its K indices (columns of an amplitude sum, rows of an out-leg sum), and the sum
/// with its gradient.
struct Summed {
    class: [i64; 2],
    index: Vec<usize>,
    m: Mat,
    dm: [Mat; 3],
}

/// A member of a class sum: (block, class, frequency, K indices, value, gradient).
type Member<'a> = (usize, [i64; 2], f64, &'a [usize], &'a Mat, &'a [Mat; 3]);

/// Sums members by block and class at the time t, each times e^{−if t}, into `out` (one list per block): amplitudes
/// (|B| × |cols|; `rows` false) or out-legs (|rows| × |B|; `rows` true).
fn gather(members: &[Member], sizes: &[usize], n: usize, t: f64, rows: bool, out: &mut [Vec<Summed>]) {
    let mut place = vec![usize::MAX; n];
    for (block, &size) in sizes.iter().enumerate() {
        let mut classes: Vec<[i64; 2]> = members.iter().filter(|m| m.0 == block).map(|m| m.1).collect();
        classes.sort_unstable();
        classes.dedup();
        for class in classes {
            let mine = || members.iter().filter(move |m| m.0 == block && m.1 == class);
            let mut index: Vec<usize> = mine().flat_map(|m| m.3.iter().copied()).collect();
            index.sort_unstable();
            index.dedup();
            for (i, &a) in index.iter().enumerate() {
                place[a] = i;
            }
            let (r, c) = if rows { (index.len(), size) } else { (size, index.len()) };
            let (mut m, mut dm) = (Mat::zeros(r, c), zeros(r, c));
            for &(_, _, frequency, ids, value, gradient) in mine() {
                let phase = C64::cis(-frequency * t);
                if rows {
                    for (i, &b) in ids.iter().enumerate() {
                        let to = place[b] * size;
                        linalg::mul_add(false, &mut m.data[to..to + size], phase, &value.data[i * size..(i + 1) * size]);
                        for (d, g) in dm.iter_mut().zip(gradient) {
                            linalg::mul_add(false, &mut d.data[to..to + size], phase, &g.data[i * size..(i + 1) * size]);
                        }
                    }
                } else {
                    for k in 0..size {
                        for (j, &a) in ids.iter().enumerate() {
                            let to = place[a];
                            m[(k, to)] += phase * value[(k, j)];
                            for (d, g) in dm.iter_mut().zip(gradient) {
                                d[(k, to)] += phase * g[(k, j)];
                            }
                        }
                    }
                }
            }
            out[block].push(Summed { class, index, m, dm });
        }
    }
}

impl Local<'_> {
    /// The K × K entries of a component's terms whose δ = n_b − n_a passes `take` (adjoint terms transposed, their δ
    /// negated): (b, a, value, gradient).
    fn entries(&self, held: &Held, take: impl Fn([i64; 2]) -> bool, mut each: impl FnMut(usize, usize, C64, [C64; 3])) {
        for t in &held.terms {
            for (delta, list) in &self.splits[t.matrix].kk {
                if !take(if t.adjoint { [-delta[0], -delta[1]] } else { *delta }) {
                    continue;
                }
                for &(i, j, v) in list {
                    let (b, a, v) = if t.adjoint { (j, i, v.conj()) } else { (i, j, v) };
                    each(b, a, t.scale * v, t.gradient.map(|g| g * v));
                }
            }
        }
    }

    /// A component's out-legs into the sideband parts (all K's rows, the identity basis), by block: Σ_t phase s_t F_t[K, B]
    /// and its gradient Σ_t phase ∂s_t F_t[K, B], added into `legs`. An H component's (`fast`) from its fast entries
    /// alone: a pair through a sideband part, W_c′[b, c] M_c[c, a] with c of fast part s, is slow (γ + γ′ − n_b + n_a = 0)
    /// only where the leg's own element is fast, its fast index γ′ − n_b + s being minus the amplitude's γ − s + n_a ≠ 0.
    /// A channel's from every entry (each element goes to the channel of its fast index).
    fn sideband_legs(&self, held: &Held, phase: C64, fast: bool, legs: &mut Vec<(usize, Mat, [Mat; 3])>) {
        let n = self.states.len();
        self.entries(held, |delta| !fast || delta != held.class, |b, a, x, dx| {
            let (block, c) = self.sideband[a];
            if block == usize::MAX {
                return;
            }
            let at = match legs.iter().position(|l| l.0 == block) {
                Some(i) => i,
                None => {
                    let size = self.blocks[block].rows.len();
                    legs.push((block, Mat::zeros(n, size), zeros(n, size)));
                    legs.len() - 1
                }
            };
            let (_, w, dw) = &mut legs[at];
            add(w, dw, (b, c), phase * x, dx.map(|d| phase * d));
        });
    }

    /// The model by frequency components (the checkpoint's; module documentation, # Two contractions): H_eff's
    /// components, the channels by (emission, fast index) with their components and photons' forms, the lift by keys.
    pub fn components(&self) -> ModelAt {
        let n = self.states.len();
        let parts = &self.parts;
        let element = |class: [i64; 2], b: usize, a: usize| Sectors::element(class, parts[b], parts[a]);
        let all: Vec<usize> = (0..n).collect();
        // Every out-leg: into Q, made at the fold; into the sideband parts, from the terms.
        let mut scratch = Vec::new();
        let mut legs_of = |held: &Held, fast: bool, into_q: &mut dyn Iterator<Item = &Leg>| -> Vec<Leg> {
            let mut out: Vec<Leg> = into_q.cloned().collect();
            scratch.clear();
            self.sideband_legs(held, C64::ONE, fast, &mut scratch);
            out.extend(scratch.drain(..).map(|(block, w, dw)| Leg { block, rows: all.clone(), w, dw }));
            out
        };
        let mut effective = Sums::new(n, n);
        for p in &self.pieces {
            let part = effective.part(&p.key, p.frequency);
            self.entries(p, |delta| delta == p.class, |b, a, x, dx| add(&mut part.value, &mut part.gradient, (b, a), x, dx));
        }
        for (c2, q) in self.pieces.iter().enumerate() {
            for leg in legs_of(q, true, &mut self.legs.iter().filter(|(c, _)| *c == c2).map(|(_, l)| l)) {
                for amp in self.amplitudes.iter().filter(|m| m.block == leg.block) {
                    let c = &self.pieces[amp.piece];
                    let class = plus(c.class, q.class);
                    if !leg.rows.iter().any(|&b| amp.cols.iter().any(|&a| element(class, b, a) == [0; 2])) {
                        continue;
                    }
                    let (p, dp) = pair(&leg.w, &leg.dw, &amp.m, &amp.dm);
                    let (key, frequency) = (sum_keys(&c.key, &q.key), c.frequency + q.frequency);
                    for half in [false, true] {
                        let part = if half { effective.part(&negated(&key), -frequency) } else { effective.part(&key, frequency) };
                        for (i, &b) in leg.rows.iter().enumerate() {
                            for (j, &a) in amp.cols.iter().enumerate().filter(|&(_, &a)| element(class, b, a) == [0; 2]) {
                                let (x, dx) = (p[(i, j)] * 0.5, dp.each_ref().map(|d| d[(i, j)] * 0.5));
                                if half {
                                    add(&mut part.value, &mut part.gradient, (a, b), x.conj(), dx.map(C64::conj));
                                } else {
                                    add(&mut part.value, &mut part.gradient, (b, a), x, dx);
                                }
                            }
                        }
                    }
                }
            }
        }
        let mut out: BTreeMap<(usize, [i64; 2]), [Sums; 2]> = BTreeMap::new();
        fn sums(out: &mut BTreeMap<(usize, [i64; 2]), [Sums; 2]>, n: usize, emission: usize, fast: [i64; 2]) -> &mut [Sums; 2] {
            out.entry((emission, fast)).or_insert_with(|| [Sums::new(n, n), Sums::new(n, n)])
        }
        for jp in &self.jumps {
            let p = &jp.held;
            self.entries(p, |_| true, |b, a, x, dx| {
                let part = sums(&mut out, n, jp.emission, element(p.class, b, a))[jp.which].part(&p.key, p.frequency);
                add(&mut part.value, &mut part.gradient, (b, a), x, dx);
            });
            for leg in legs_of(p, false, &mut jp.legs.iter()) {
                for amp in self.amplitudes.iter().filter(|m| m.block == leg.block) {
                    let c = &self.pieces[amp.piece];
                    let class = plus(p.class, c.class);
                    let (v, dv) = pair(&leg.w, &leg.dw, &amp.m, &amp.dm);
                    let (key, frequency) = (sum_keys(&p.key, &c.key), p.frequency + c.frequency);
                    for (i, &b) in leg.rows.iter().enumerate() {
                        for (j, &a) in amp.cols.iter().enumerate() {
                            let (x, dx) = (v[(i, j)], dv.each_ref().map(|d| d[(i, j)]));
                            if x != C64::ZERO || dx.iter().any(|d| *d != C64::ZERO) {
                                let part = sums(&mut out, n, jp.emission, element(class, b, a))[jp.which].part(&key, frequency);
                                add(&mut part.value, &mut part.gradient, (b, a), x, dx);
                            }
                        }
                    }
                }
            }
        }
        let jumps = out
            .into_iter()
            .map(|((emission, fast), [parts, kicks])| Channel { emission, fast, parts: parts.done(), kicks: kicks.done() })
            .filter(|c| !c.parts.is_empty())
            .collect();
        let mut lift = Sums::new(self.folded.len(), n);
        for amp in self.amplitudes.iter().filter(|m| self.blocks[m.block].part.is_none()) {
            let (block, c) = (&self.blocks[amp.block], &self.pieces[amp.piece]);
            let (m, dm) = (block.back(&amp.m), amp.dm.each_ref().map(|d| block.back(d)));
            let part = lift.part(&c.key, c.frequency);
            for (i, &row) in block.rows.iter().enumerate() {
                for (j, &col) in amp.cols.iter().enumerate() {
                    add(&mut part.value, &mut part.gradient, (row, col), m[(i, j)], dm.each_ref().map(|d| d[(i, j)]));
                }
            }
        }
        ModelAt { states: self.states.clone(), folded: self.folded.clone(), groups: self.groups.clone(), effective: effective.done(), jumps, lift: lift.done() }
    }

    fn sizes(&self) -> Vec<usize> {
        self.blocks.iter().map(|b| b.rows.len()).collect()
    }

    /// The amplitudes' class sums at the time t on each block, Z_γ = Σ_{n_c = γ} e^{−if_c t} M̂_c.
    fn sums(&self, t: f64) -> Vec<Vec<Summed>> {
        let members: Vec<Member> = self.amplitudes.iter().map(|a| (a.block, self.pieces[a.piece].class, self.pieces[a.piece].frequency, a.cols.as_slice(), &a.m, &a.dm)).collect();
        let mut out: Vec<Vec<Summed>> = (0..self.blocks.len()).map(|_| Vec::new()).collect();
        gather(&members, &self.sizes(), self.states.len(), t, false, &mut out);
        out
    }

    /// The out-legs' class sums at the time t on each block, W_γ′ = Σ_{n_c′ = γ′} e^{−if_c′ t} W_c′, of components
    /// (`held`) with their out-legs into Q: into Q from those, into the sideband parts from the terms (H's from their fast
    /// entries, `fast`; `sideband_legs`).
    fn leg_sums<'b>(&self, t: f64, fast: bool, held: impl Iterator<Item = (&'b Held, &'b [Leg])>) -> Vec<Vec<Summed>> {
        let n = self.states.len();
        let mut out: Vec<Vec<Summed>> = (0..self.blocks.len()).map(|_| Vec::new()).collect();
        let mut members: Vec<Member> = Vec::new();
        // The sideband parts' sums by class, each added to in place.
        let mut by_class: Vec<([i64; 2], Vec<(usize, Mat, [Mat; 3])>)> = Vec::new();
        for (h, into_q) in held {
            members.extend(into_q.iter().map(|l| (l.block, h.class, h.frequency, l.rows.as_slice(), &l.w, &l.dw)));
            let at = match by_class.iter().position(|c| c.0 == h.class) {
                Some(i) => i,
                None => {
                    by_class.push((h.class, Vec::new()));
                    by_class.len() - 1
                }
            };
            self.sideband_legs(h, C64::cis(-h.frequency * t), fast, &mut by_class[at].1);
        }
        let all: Vec<usize> = (0..n).collect();
        for (class, legs) in by_class {
            for (block, m, dm) in legs {
                out[block].push(Summed { class, index: all.clone(), m, dm });
            }
        }
        gather(&members, &self.sizes(), n, t, true, &mut out);
        out
    }

    /// The model's value at the time t (module documentation, # Two contractions): H_eff(t) = Σ_c H⁽ᶜ⁾ e^{−if_c t} with its
    /// gradient, each channel's L(t) with its gradient, and the lift M(t) on Q (|Q| × n), by class sums.
    pub fn now(&self, t: f64) -> Now {
        let n = self.states.len();
        let parts = &self.parts;
        let slow = |class: [i64; 2], b: usize, a: usize| Sectors::element(class, parts[b], parts[a]) == [0; 2];
        let (mut h, mut dh) = (Mat::zeros(n, n), zeros(n, n));
        for p in &self.pieces {
            let phase = C64::cis(-p.frequency * t);
            self.entries(p, |delta| delta == p.class, |b, a, x, dx| add(&mut h, &mut dh, (b, a), phase * x, dx.map(|d| phase * d)));
        }
        let zs = self.sums(t);
        let mut legs: Vec<&[Leg]> = vec![&[]; self.pieces.len()];
        let mut grouped: Vec<Vec<Leg>> = vec![Vec::new(); self.pieces.len()];
        for (c, l) in &self.legs {
            grouped[*c].push(l.clone());
        }
        for (c, g) in grouped.iter().enumerate() {
            legs[c] = g;
        }
        let ws = self.leg_sums(t, true, self.pieces.iter().zip(legs));
        for (zb, wb) in zs.iter().zip(&ws) {
            for w in wb {
                for z in zb {
                    let class = plus(w.class, z.class);
                    let kept: Vec<(usize, usize)> = (0..w.index.len()).flat_map(|i| (0..z.index.len()).map(move |j| (i, j))).filter(|&(i, j)| slow(class, w.index[i], z.index[j])).collect();
                    if kept.is_empty() {
                        continue;
                    }
                    let (p, dp) = pair(&w.m, &w.dm, &z.m, &z.dm);
                    for (i, j) in kept {
                        let (b, a) = (w.index[i], z.index[j]);
                        let x = p[(i, j)] * 0.5;
                        h[(b, a)] += x;
                        h[(a, b)] += x.conj();
                        for (g, d) in dh.iter_mut().zip(&dp) {
                            let y = d[(i, j)] * 0.5;
                            g[(b, a)] += y;
                            g[(a, b)] += y.conj();
                        }
                    }
                }
            }
        }
        let jumps = self.channels(t, 0, &zs);
        let mut lift = Mat::zeros(self.folded.len(), n);
        for (block, zb) in self.blocks.iter().zip(&zs).filter(|(b, _)| b.part.is_none()) {
            let mut z = Mat::zeros(block.rows.len(), n);
            for s in zb {
                for k in 0..block.rows.len() {
                    for (j, &a) in s.index.iter().enumerate() {
                        z[(k, a)] += s.m[(k, j)];
                    }
                }
            }
            let m = block.back(&z);
            for (k, &row) in block.rows.iter().enumerate() {
                lift.data[row * n..(row + 1) * n].copy_from_slice(&m.data[k * n..(k + 1) * n]);
            }
        }
        Now { states: self.states.clone(), folded: self.folded.clone(), hamiltonian: h, gradient: dh, jumps, lift }
    }

    /// The channels' photons' forms at the time t (`now`'s channels' kicks: k∘L per element, folded alike).
    pub fn kicks(&self, t: f64) -> Vec<Moment> {
        self.channels(t, 1, &self.sums(t))
    }

    /// The channels (`which` 0) or their photons' forms (1) at the time t from the amplitudes' class sums: direct, and
    /// each out-leg class sum times each amplitude class sum, every element into the channel of its fast index.
    fn channels(&self, t: f64, which: usize, zs: &[Vec<Summed>]) -> Vec<Moment> {
        let n = self.states.len();
        let parts = &self.parts;
        let index = |class: [i64; 2], b: usize, a: usize| Sectors::element(class, parts[b], parts[a]);
        let mut out: Vec<Moment> = Vec::new();
        // The slot of each (emission, fast index): a table over the fast indices' range, the classes' and parts' sums.
        let reach = self.pieces.iter().map(|p| p.class).chain(self.jumps.iter().map(|j| j.held.class)).chain(parts.iter().copied()).flat_map(|c| [c[0].abs(), c[1].abs()]).max().unwrap_or(0);
        let side = (8 * reach + 1) as usize;
        let mut table = vec![usize::MAX; 3 * side * side];
        let mut at = |out: &mut Vec<Moment>, emission: usize, fast: [i64; 2]| -> usize {
            let cell = emission * side * side + (fast[0] + 4 * reach) as usize * side + (fast[1] + 4 * reach) as usize;
            if table[cell] == usize::MAX {
                out.push(Moment { emission, fast, value: Mat::zeros(n, n), gradient: zeros(n, n) });
                table[cell] = out.len() - 1;
            }
            table[cell]
        };
        for emission in 0..3 {
            let mine: Vec<&JumpPart> = self.jumps.iter().filter(|j| j.emission == emission && j.which == which).collect();
            for jp in &mine {
                let phase = C64::cis(-jp.held.frequency * t);
                self.entries(&jp.held, |_| true, |b, a, x, dx| {
                    let slot = at(&mut out, emission, index(jp.held.class, b, a));
                    let m = &mut out[slot];
                    add(&mut m.value, &mut m.gradient, (b, a), phase * x, dx.map(|d| phase * d));
                });
            }
            let ws = self.leg_sums(t, false, mine.iter().map(|jp| (&jp.held, jp.legs.as_slice())));
            for (zb, wb) in zs.iter().zip(&ws) {
                for w in wb {
                    for z in zb {
                        let class = plus(w.class, z.class);
                        let (p, dp) = pair(&w.m, &w.dm, &z.m, &z.dm);
                        for (i, &b) in w.index.iter().enumerate() {
                            for (j, &a) in z.index.iter().enumerate() {
                                let (x, dx) = (p[(i, j)], dp.each_ref().map(|d| d[(i, j)]));
                                if x != C64::ZERO || dx.iter().any(|d| *d != C64::ZERO) {
                                    let slot = at(&mut out, emission, index(class, b, a));
                                    let m = &mut out[slot];
                                    add(&mut m.value, &mut m.gradient, (b, a), x, dx);
                                }
                            }
                        }
                    }
                }
            }
        }
        out.sort_by(|x, y| (x.emission, x.fast).cmp(&(y.emission, y.fast)));
        out
    }
}

/// A jump channel's value at a time: its emission, its fast index, L(t) and ∂L(t) (n × n).
#[derive(Clone, Debug, PartialEq)]
pub struct Moment {
    pub emission: usize,
    pub fast: [i64; 2],
    pub value: Mat,
    pub gradient: [Mat; 3],
}

/// The model's value at a time t (`Local::now`): H_eff(t) and ∂H_eff(t) (n × n), the channels L(t), the lift M(t) on Q.
#[derive(Clone, Debug, PartialEq)]
pub struct Now {
    pub states: Vec<usize>,
    pub folded: Vec<usize>,
    pub hamiltonian: Mat,
    pub gradient: [Mat; 3],
    pub jumps: Vec<Moment>,
    pub lift: Mat,
}

impl Now {
    /// The force on ρ (n × n, row-major): the adjoint contraction −Re tr(∇H_eff ρ) and the folded light's absorbed
    /// momentum Σ_ch Im tr(ρ L_ch†∇L_ch) -- the force observable of the Lindbladian, −∂H + Σ_ch (L_ch†∂L_ch −
    /// ∂L_ch†L_ch)/2i, read on ρ.
    pub fn force(&self, rho: &[C64]) -> V3 {
        let n = self.states.len();
        let trace = |x: &Mat| -> C64 { (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).map(|(a, b)| x[(a, b)] * rho[b * n + a]).sum() };
        std::array::from_fn(|i| {
            let mut f = -trace(&self.gradient[i]).re;
            for channel in &self.jumps {
                f += trace(&channel.value.adjoint_product(&channel.gradient[i])).im;
            }
            f
        })
    }

    /// The 32 states' populations of ρ (n × n, row-major) lifted through M(t): the model's ρ_aa, Q's (M ρ M†)_kk.
    pub fn lift(&self, rho: &[C64]) -> [f64; STATES] {
        let n = self.states.len();
        let mut out = [0.0; STATES];
        for (a, &state) in self.states.iter().enumerate() {
            out[state] = rho[a * n + a].re;
        }
        let lifted = self.lift.product(&Mat { rows: n, cols: n, data: rho.to_vec() }).product(&self.lift.adjoint());
        for (k, &state) in self.folded.iter().enumerate() {
            out[state] = lifted[(k, k)].re;
        }
        out
    }
}

impl ModelAt {
    /// H's static part (the zero key).
    pub fn still(&self) -> &ModelPart {
        self.effective.iter().find(|p| p.key.iter().all(|k| *k == 0)).expect("H's static part")
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

// ---------------------------------------------------------------------------------------------------------------------
// The emission, the tweezer's fold, the reference and the model.

/// The request's Cartesian emission as fixed matrices: C_a, k∘C_a (the photons' forms), and Σ_a C_a†C_a.
pub fn emission(jumps: &Jumps) -> ([Mat; 3], [Mat; 3], Mat) {
    let photon = |c: &Mat| Mat::from_fn(STATES, STATES, |i, j| c[(i, j)] * jumps.wave_number[(i, j)].re);
    let kicks = jumps.emission.each_ref().map(photon);
    let mut gamma = Mat::zeros(STATES, STATES);
    for c in &jumps.emission {
        adjoint_product_onto(c, c, C64::ONE, &mut gamma);
    }
    (jumps.emission.clone(), kicks, gamma)
}

/// The emission's three channels (n_tw = n_φ = 0) on the library's matrices `parts` and `kicks`.
pub fn emission_channels(length: usize, parts: [usize; 3], kicks: [usize; 3]) -> Vec<SourceChannel> {
    let piece = |m: usize| Piece { key: vec![0; length], frequency: 0.0, terms: vec![Term::one(m, C64::ONE)] };
    (0..3).map(|a| SourceChannel { emission: a, fast: [0; 2], parts: vec![piece(parts[a])], kicks: vec![piece(kicks[a])] }).collect()
}

/// The fixed matrices of the folds and the tweezer's sidebands folded once (module documentation, # The two folds): the
/// library holds the scene's matrices (`Scene::matrix`, at their indices), the emission's C_a, k∘C_a and Σ C_a†C_a, and
/// the unit fold's pairs (n_tw = 0, H's part ∝ |a_tw|²), its channels (n_tw = ±1, ∝ a_tw and a_tw*) and their decay.
#[derive(Clone, Debug, PartialEq)]
pub struct Tweezer {
    library: Vec<Fixed>,
    sectors: Sectors,
    groups: Vec<Vec<usize>>,
    emission: [usize; 3],
    kicks: [usize; 3],
    decay: usize,
    shift: Vec<(Key, usize)>,
    channels: Vec<(usize, [i64; 2], Vec<(Key, usize)>, Vec<(Key, usize)>)>,
    scattering: usize,
}

impl Tweezer {
    /// The fold of the tweezer's couplings at unit amplitude and at rest (`Scene::tweezer_couplings`, both ways) with the
    /// emission's channels, kept on the 32 states, at the request's H0.
    pub fn new(scene: &Scene) -> Tweezer {
        let length = scene.generators.len();
        let mut library: Vec<Fixed> = (0..scene.matrices()).map(|i| fixed(scene.matrix(i))).collect();
        let put = |library: &mut Vec<Fixed>, m: &Mat| {
            library.push(fixed(m));
            library.len() - 1
        };
        let (c, k, gamma) = emission(scene.jumps());
        let emission = c.each_ref().map(|m| put(&mut library, m));
        let kicks = k.each_ref().map(|m| put(&mut library, m));
        let rate = put(&mut library, &gamma);
        let h0 = put(&mut library, &scene.request.hamiltonian);
        let mut hamiltonian = Vec::new();
        for (key, frequency, x) in scene.tweezer_couplings() {
            let piece = Piece { key, frequency, terms: vec![Term::one(put(&mut library, x), C64::ONE)] };
            let adjoint = piece.adjoint();
            hamiltonian.push(piece);
            hamiltonian.push(adjoint);
        }
        let source = Source { hamiltonian, jumps: emission_channels(length, emission, kicks), generator: vec![Term::one(h0, C64::ONE), Term::one(rate, C64::new(0.0, -0.5))] };
        let all: Vec<usize> = (0..STATES).collect();
        let groups = levels(&scene.request.states, &all);
        let sectors = Sectors::tweezer(scene);
        let splits = Split::all(&library, &all, &sectors, REFERENCE);
        let unit = fold(&splits, &all, &[], &source, &sectors, REFERENCE, &groups).components();
        let shift = unit.effective.iter().map(|p| (p.key.clone(), put(&mut library, &p.value))).collect();
        let channels: Vec<Channel> = unit.jumps.into_iter().filter(|c| c.fast != [0; 2]).collect();
        let scattering = put(&mut library, &decay(&channels, STATES).0);
        let mut on = |parts: &[ModelPart]| parts.iter().map(|p| (p.key.clone(), put(&mut library, &p.value))).collect::<Vec<_>>();
        let channels = channels.iter().map(|c| (c.emission, c.fast, on(&c.parts), on(&c.kicks))).collect();
        Tweezer { library, sectors, groups, emission, kicks, decay: rate, shift, channels, scattering }
    }

    /// The full reference's source at a point (scene.rs `Point`): the scene's terms, the tweezer's left out, by key
    /// (each nonzero key both ways); the tweezer's fold put in at its amplitude a: the pairs times |a|², the channels with
    /// n_tw = 1 times a, with −1 times a*; the emission's channels; the generator H's static part − (i/2)(Σ C†C +
    /// |a|² Σ Ĵ†Ĵ).
    pub fn source(&self, scene: &Scene, point: &Point) -> Source {
        let tw = scene.tweezer();
        let a = &point.amplitudes[tw];
        let (power, dpower): (f64, [f64; 3]) = (a.value.norm_sqr(), std::array::from_fn(|i| 2.0 * (a.value.conj() * a.gradient[i]).re));
        let mut pieces: Vec<Piece> = Vec::new();
        for t in point.terms.iter().filter(|t| self.sectors.key(&t.key, REFERENCE) == [0; 2]) {
            let (scale, gradient) = t.factor.at(&point.amplitudes);
            let term = Term { matrix: t.matrix, adjoint: t.adjoint, scale, gradient };
            match pieces.iter_mut().rev().find(|p| p.key == t.key) {
                Some(p) => p.terms.push(term),
                None => pieces.push(Piece { key: t.key.clone(), frequency: t.frequency, terms: vec![term] }),
            }
        }
        let mut hamiltonian = Vec::with_capacity(2 * pieces.len());
        for p in pieces {
            let adjoint = p.key.iter().any(|k| *k != 0).then(|| p.adjoint());
            hamiltonian.push(p);
            hamiltonian.extend(adjoint);
        }
        let shifted = |m: usize| Term { matrix: m, adjoint: false, scale: C64::real(power), gradient: dpower.map(C64::real) };
        for (key, m) in &self.shift {
            match hamiltonian.iter_mut().find(|p| &p.key == key) {
                Some(p) => p.terms.push(shifted(*m)),
                None => hamiltonian.push(Piece { key: key.clone(), frequency: point.frequency(key), terms: vec![shifted(*m)] }),
            }
        }
        let mut jumps = emission_channels(scene.generators.len(), self.emission, self.kicks);
        for (emission, fast, parts, kicks) in &self.channels {
            let (scale, gradient): (C64, [C64; 3]) = if fast[1] > 0 { (a.value, a.gradient) } else { (a.value.conj(), a.gradient.map(C64::conj)) };
            let put = |(key, m): &(Key, usize)| Piece { key: key.clone(), frequency: point.frequency(key), terms: vec![Term { matrix: *m, adjoint: false, scale, gradient }] };
            let parts: Vec<Piece> = parts.iter().map(put).filter(|p| !p.terms[0].vanishes()).collect();
            if !parts.is_empty() {
                jumps.push(SourceChannel { emission: *emission, fast: *fast, parts, kicks: kicks.iter().map(put).collect() });
            }
        }
        let mut generator = hamiltonian.iter().find(|p| p.key.iter().all(|k| *k == 0)).expect("H's static part").terms.clone();
        generator.push(Term::one(self.decay, C64::new(0.0, -0.5)));
        generator.push(Term { matrix: self.scattering, adjoint: false, scale: C64::new(0.0, -0.5 * power), gradient: dpower.map(|d| C64::new(0.0, -0.5 * d)) });
        Source { hamiltonian, jumps, generator }
    }

    /// The unit fold's static part on the 32 states (rad/µs at the tweezer's focus intensity): its ground diagonal the
    /// tweezer's ground shift (`Scene::trap_potential`'s depth), its excited block the tweezer's shifts of the excited
    /// states (`Scene::smooth`).
    pub fn still(&self) -> Mat {
        let zero = self.shift.iter().find(|(key, _)| key.iter().all(|k| *k == 0)).map(|(_, m)| *m);
        zero.map_or_else(|| Mat::zeros(STATES, STATES), |m| dense(&self.library, &[Term::one(m, C64::ONE)]).0)
    }

    /// The full reference at a point as a dense model on the 32 states, with its generator.
    pub fn reference(&self, scene: &Scene, point: &Point) -> (ModelAt, ModelPart) {
        self.source(scene, point).model(&self.library, &self.groups)
    }
}

/// The full reference for the scene's components at a point: the scene's model on the 32 states with the tweezer's
/// sidebands folded (floquet's `Hilbert::full` is it).
pub fn reference(scene: &Scene, components: &Components) -> ModelAt {
    Tweezer::new(scene).reference(scene, &components.point).0
}

/// The RS model on a kept set P: P, its complement, the frame's sectors, the tweezer's fold and the fixed matrices split
/// for P and Q, made once for the run.
#[derive(Clone, Debug, PartialEq)]
pub struct Rs {
    /// P: indices into the 32, sorted (the ground states first).
    pub keep: Vec<usize>,
    /// Q: the folded states, the rest of the 32.
    pub folded: Vec<usize>,
    pub sectors: Sectors,
    tweezer: Tweezer,
    splits: Vec<Split>,
}

impl Rs {
    /// The model's fold on P = `keep` with the frame's sectors; the tweezer's fold and the splits made here once.
    pub fn new(scene: &Scene, keep: &[usize], sectors: Sectors) -> Rs {
        let mut keep = keep.to_vec();
        keep.sort_unstable();
        keep.dedup();
        let folded = (0..STATES).filter(|i| keep.binary_search(i).is_err()).collect();
        let tweezer = Tweezer::new(scene);
        let splits = Split::all(&tweezer.library, &keep, &sectors, MODEL);
        Rs { keep, folded, sectors, tweezer, splits }
    }

    /// The fold at a point (scene.rs `Point`: `Scene::point_at` on a trajectory, `Scene::components(..).point` at the
    /// checkpoint's): the full reference there folded on P at its static part. Its contractions: `Local::now` (the
    /// value at a time) and `Local::components`.
    pub fn local(&self, scene: &Scene, point: &Point) -> Local<'_> {
        let source = self.tweezer.source(scene, point);
        fold(&self.splits, &self.keep, &self.folded, &source, &self.sectors, MODEL, &self.tweezer.groups)
    }

    /// The tweezer's fold made once (`Tweezer::still`).
    pub fn tweezer(&self) -> &Tweezer {
        &self.tweezer
    }

    /// The model by frequency components for the scene's components at a point (the checkpoint's).
    pub fn at(&self, scene: &Scene, components: &Components) -> ModelAt {
        self.local(scene, &components.point).components()
    }
}

#[cfg(test)]
mod tests;
