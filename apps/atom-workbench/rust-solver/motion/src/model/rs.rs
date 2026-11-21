//! The one fold (RUSTPLAN §2.1–§2.3 and its corrections of 2026-11-19: "模型只定一次", "一个折叠公式，快分量与光镊都由它
//! 折叠"): Reiter–Sørensen second order with harmonics and dissipation (td_s.hilbert_rs, each column at E_a + f), the
//! full reference's and the model's alike -- they differ only in what is folded and at which generator. Written linear in
//! the light field (lin.status.md; BRIEF_R1MODEL item 1): its structure made once, a point puts in the field.
//!
//! # The fold
//! A Lindblad model on a set of states by frequency components: H's components H⁽ᶜ⁾ e^{−if_c t} taken both ways (each
//! nonzero key with its conjugate at −key), and jump channels L_ch = Σ_c L_ch⁽ᶜ⁾ e^{−if_c t}. A fold keeps the states K
//! and, among their elements, the slow ones: the element b ← a of the component c has the fast index n = n_c − n_b + n_a
//! (`Sectors::element`, the fast part of its frame vector c − θ_b + θ_a, floquet/frame.rs `element`, the assembly's own;
//! n_c = Σ_l c_l n_l over the key's generators, n_a the state's: n_φ the frame's, n_tw the tweezer's photons, `Rs::new`), slow where
//! its folding coordinates (`axes`) are zero. Everything else is folded: every component of the other states F (= Q),
//! and the fast elements within K (K's sidebands -- the hyperfine n_φ and the tweezer n_tw). The folded states'
//! unperturbed generator is block diagonal,
//!   G₀ = G_BB − (i/2) Γ_BB,  Γ = Σ_ch Σ_c L_ch⁽ᶜ⁾† L_ch⁽ᶜ⁾  (the static decay),
//! G the static generator the fold is taken at -- the zero key's component alone --, on the blocks B: Q, and K's states
//! by their fast parts. Column a of K at its energy E_a = Re G_aa and the component's frequency,
//!   M_c[B, a] = (E_a + f_c − G₀_BB)⁻¹ H⁽ᶜ⁾_Ba        (B = Q, or a block of K where c's element into it is fast),
//! and the model on K, slow elements alone,
//!   H_eff = Σ_c [H⁽ᶜ⁾_KK]_slow + ½ Σ_{c′,c} [H⁽ᶜ′⁾_K· M_c + h.c.]_slow,  the pair at key c + c′,
//!   L_(ch,n) = [L_ch⁽ᶜ⁾_KK + Σ_c′ L_ch⁽ᶜ′⁾_K· M_c]_n,  each element in the channel of its fast index n,
//! so that every channel is slow (its elements beat together), and the interference of channels apart by a fast
//! frequency -- and its share of the decay −(i/2)L†L -- is not in the model. A pair whose total is fast is of the fourth
//! order in the couplings and not in the model either. The gradient along r_i is the same formula's derivative,
//!   ∂M_c[B, a] = (E_a + f_c − G₀_BB)⁻¹ (∂H⁽ᶜ⁾_Ba + (∂G₀_BB − ∂E_a) M_c[B, a]),
//! with the generator's own gradient (the tweezer's intensity, R T R ∇s: 0.02–4 % of the fold's gradient off the focus).
//!
//! # The frequencies: one Doppler shift per beam
//! A key's frequency at the momentum κ is Σ_g c_g f_g, f_g = ω_g − w_g, w_g = ħ(k_g·κ + |k_g|²/2)/m the change of the
//! atom's kinetic energy as it absorbs a photon of beam g (`Scene::doppler`; emission the opposite sign): additive over
//! the key's integer vector, so a pair's frequency is its members' sum and a group of components one sum. The engine is
//! semiclassical: κ is the centre of mass's classical momentum, the internal frequencies defined to first order in ħk/p.
//! For one photon from the ground state the beam's shift is E(p + ħk) − E(p) exactly; for a Raman pair the beams' sum
//! K·κ + (k_m² − k_l²)/2 is the exact energy difference at the two momentum states' mean momentum (equal wavelengths:
//! K·κ̄). Against |K|²/2 at the initial ground state's momentum it differs by the cross term ħK_c·K_c′/m (≤ 2ω_r ≈ 0.05
//! rad/µs), a velocity below one recoil's, below the semiclassical method's own resolution (each jump kicks a recoil's
//! velocity): the choice of the centre of mass's momentum, not an approximation (R1-MODEL, coordinator 2026-11-20).
//!
//! # The two folds
//! - The full reference (`Tweezer`, `reference`): the scene's components and its Cartesian emission on the 32 states, all
//!   kept, the tweezer's sidebands folded (axes n_tw) at the request's H0 (pysel.status.md: the denominators the bare
//!   levels E = diag H0, fixed, not differentiated -- their dependence on the local shifts is of the fourth order, 10⁻⁷
//!   relative, within the one allowed distinction's second order). Its components are a_tw X_tw and a_tw* R_tw, so the
//!   fold is linear in them: made once at unit amplitude, its pairs (n_tw = 0) are |a_tw|² times the unit fold's and its
//!   channels (n_tw = ±1) a_tw or a_tw* times them. The tweezer's own Doppler shift δ = Σ n_b w_b of its key (±w_tw) is in
//!   its denominators: (z − δ)⁻¹ = Σ_j δ^j z^{−(j+1)}, the unit fold made at each power (`ORDERS`) and a point taking the
//!   powers while the next is above the rounding of the first (δ/Δ_tw ≈ 4·10⁻⁹ at the thermal speed: two terms).
//!   Six channels: the three emission axes, co- and counter-rotating, apart from the emission's (n_tw = 0).
//! - The model (`Rs`): the reference folded again on P at its static part, every fast element folded (axes n_φ, n_tw): Q
//!   at every component, P's hyperfine sidebands.
//!
//! # What is made once and what at a point
//! Every operator of the fold is a sum of terms, a fixed matrix (or a sum of them with coefficients, the tweezer's
//! powers of δ) times a factor of the beams' amplitudes at the point (`Scene::terms`; the tweezer fold's own, times
//! |a_tw|², a_tw, a_tw*). The fixed matrices are made once (`Rs::new`) by where their entries fall -- K × K, Q × K,
//! K × Q, Q × Q (`Split`) --, their entries by the rounding rule (`fixed`), and with them the fold's whole structure:
//! the blocks, the routes, the sums, the pairs, the outputs (rs/fold.rs `Plan`, `Grouping`). A point (`Rs::local`) puts in
//! the factors and the frequencies and makes the one thing the formula itself asks there: the folded blocks' resolvents
//! at the local generator, Q's through their spectrum (rs/spectrum.rs: decoupled at the gaps that pay, each leaf about its
//! own centre), K's sideband parts through the Neumann series about their diagonal (their denominators are a hyperfine
//! φ ≈ 4·10⁴ rad/µs or the tweezer's 10⁸ off couplings ≲ 10²; ρ ≈ 2·10⁻³, summed to each column's rounding).
//!
//! # Contractions (`At::contract`)
//! One contraction of the fold at a point, by groups of its components: each component c enters its group g(c) with a
//! weight w_c, a pair's group is the members' summed. The sums are made before the products: per block the amplitudes'
//! Z_{γ,g,s} = Σ_{c: n_c = γ, g(c) = g} w_c M̂_c on the kept states of fast part s and the out-legs' W likewise, each pair
//! of sums one product, kept whole where it is slow (γ + γ′ − s_b + s_a = 0) at the group g + g′; the jumps the same way,
//! each pair whole into the channel of its fast index π + γ − s_b + s_a. The products' count is the groups' and classes',
//! not the components'. Three groupings:
//! - by key, weight one (`At::components`, the checkpoint's: the Floquet lattice needs each key apart);
//! - one group, weight e^{−if_c t} (`At::now`: the value at a time t, Σ_c (·)⁽ᶜ⁾ e^{−if_c t});
//! - by the frequency the integrator times them at (model/selected.rs: a trajectory's node, its components by the lab's
//!   frequency, every beam of one frequency one group whatever its Doppler shift -- the fold's denominators keep it).
//!
//! A contraction makes what its caller reads (`Making`): the model -- H_eff and the channels --, the channels' photon
//! forms, the lift; the rest it leaves empty. The groupings are the same sums in other orders; `now` equals
//! Σ components · e^{−ift} to rounding (tests, `now_is_the_components_at_their_phases`).

use std::collections::BTreeMap;

pub mod fold;
pub mod spectrum;

use crate::floquet::{Hilbert, Lift, Part, Sectors};
use crate::linalg::{C64, Mat, V3};
use crate::request::STATES;
use crate::scene::{self, Components, Jumps, Key, Point, Scene};
use crate::units::UNIT_ROUNDOFF;
pub use fold::{Making, Piece, Source, SourceChannel, Term, Values};

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

/// A Lindblad model at a point by frequency components: what a fold gives by components (`At::components`).
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
    pub(crate) fn key(&self, key: &Key, axes: [bool; 2]) -> [i64; 2] {
        std::array::from_fn(|i| if axes[i] { key.iter().zip(&self.generators).map(|(&k, g)| i64::from(k) * g[i]).sum() } else { 0 })
    }

    /// A state's fast part on the folding coordinates.
    pub(crate) fn state(&self, state: usize, axes: [bool; 2]) -> [i64; 2] {
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

/// The sum of two keys.
fn sum_keys(a: &Key, b: &Key) -> Key {
    let length = a.len().max(b.len());
    (0..length).map(|i| a.get(i).copied().unwrap_or(0) + b.get(i).copied().unwrap_or(0)).collect()
}

fn negated(key: &Key) -> Key {
    key.iter().map(|k| -k).collect()
}

fn zeros(rows: usize, cols: usize) -> [Mat; 3] {
    std::array::from_fn(|_| Mat::zeros(rows, cols))
}

/// out ← out + scale a† b, the zero entries of a skipped.
fn adjoint_product_onto(a: &Mat, b: &Mat, scale: C64, out: &mut Mat) {
    assert_eq!((a.rows, a.cols, b.cols), (b.rows, out.rows, out.cols));
    for k in 0..a.rows {
        for i in 0..a.cols {
            let x = a[(k, i)];
            if x != C64::ZERO {
                crate::linalg::mul_add(false, &mut out.data[i * out.cols..(i + 1) * out.cols], scale * x.conj(), &b.data[k * b.cols..(k + 1) * b.cols]);
            }
        }
    }
}

/// The decay Γ = Σ_ch Σ_c L_ch⁽ᶜ⁾† L_ch⁽ᶜ⁾ of jump channels (size × size) and its gradient: each emission axis's sum made
/// whole before the axes are added (the axes' products cancel where the polarizations' do, exactly so summed: the
/// tilt's Δm = 2 elements, diaga.status.md 4.3).
pub fn decay(jumps: &[Channel], size: usize) -> (Mat, [Mat; 3]) {
    let (mut gamma, mut changes) = (Mat::zeros(size, size), zeros(size, size));
    for axis in 0..3 {
        let (mut g, mut d) = (Mat::zeros(size, size), zeros(size, size));
        for p in jumps.iter().filter(|c| c.emission == axis).flat_map(|c| &c.parts) {
            adjoint_product_onto(&p.value, &p.value, C64::ONE, &mut g);
            for (d, dl) in d.iter_mut().zip(&p.gradient) {
                if nonzero(dl) {
                    adjoint_product_onto(dl, &p.value, C64::ONE, d);
                    adjoint_product_onto(&p.value, dl, C64::ONE, d);
                }
            }
        }
        gamma.add_scaled(C64::ONE, &g);
        for (c, d) in changes.iter_mut().zip(&d) {
            c.add_scaled(C64::ONE, d);
        }
    }
    (gamma, changes)
}

// ---------------------------------------------------------------------------------------------------------------------
// The fixed matrices.

/// A fixed matrix on the 32 states by its entries that count (row, column, value).
pub type Fixed = Vec<(usize, usize, C64)>;

/// A fixed matrix's entries: those that are not zero. The request's matrices are given entry by entry.
pub fn fixed(m: &Mat) -> Fixed {
    (0..m.rows).flat_map(|i| (0..m.cols).map(move |j| (i, j))).filter(|&(i, j)| m[(i, j)] != C64::ZERO).map(|(i, j)| (i, j, m[(i, j)])).collect()
}

/// A matrix the engine makes as a sum of products (the tweezer's unit fold), by the rounding rule: its entries below
/// u·max|x| of the matrix itself are the rounding of a cancellation -- of the polarizations' axes, of the paths through
/// the folded states --, not a coupling, and are zero.
pub fn rounded(m: &Mat) -> Mat {
    let floor = UNIT_ROUNDOFF * m.max_abs();
    Mat { rows: m.rows, cols: m.cols, data: m.data.iter().map(|z| if z.abs() < floor { C64::ZERO } else { *z }).collect() }
}

/// A fixed matrix's entries by where they fall in a fold of K and Q: K × K (positions in K) by the fast-part difference
/// δ = n_b − n_a of their row b and column a (an element of a component of fast index γ is slow where δ = γ,
/// `Sectors::element`), Q × K (Q's index, K's position), K × Q, Q × Q.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Split {
    pub(crate) kk: Vec<([i64; 2], Fixed)>,
    pub(crate) qk: Fixed,
    pub(crate) kq: Fixed,
    pub(crate) qq: Fixed,
}

impl Split {
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

/// The factor a term takes at a point: the scene's (of the beams' amplitudes, scene.rs `Factor`), the tweezer's
/// intensity |a_tw|², or a fixed number with a fixed gradient (a check's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Factor {
    Scene(scene::Factor),
    Power,
    Fixed(C64, [C64; 3]),
}

impl Factor {
    /// The factor of the adjoint term.
    fn conj(self) -> Factor {
        match self {
            Factor::Scene(scene::Factor::Beam(l)) => Factor::Scene(scene::Factor::Conjugate(l)),
            Factor::Scene(scene::Factor::Conjugate(l)) => Factor::Scene(scene::Factor::Beam(l)),
            Factor::Scene(scene::Factor::Product(l, m)) => Factor::Scene(scene::Factor::Product(m, l)),
            Factor::Fixed(v, g) => Factor::Fixed(v.conj(), g.map(C64::conj)),
            other => other,
        }
    }

    /// Whether its gradient can be other than zero (a plane wave's own intensity is one everywhere): the generator's
    /// varying factors make Ĝ (rs/fold.rs).
    fn varying(self, scene: Option<&Scene>) -> bool {
        match self {
            Factor::Scene(scene::Factor::One) => false,
            Factor::Scene(scene::Factor::Product(l, m)) if l == m => scene.is_none_or(|s| s.geometry[l].waist.is_some()),
            Factor::Scene(_) | Factor::Power => true,
            Factor::Fixed(_, g) => g.iter().any(|d| *d != C64::ZERO),
        }
    }

    /// Its value and gradient at the beams' amplitudes (`tweezer` the tweezer's beam).
    fn at(self, scene: Option<&Scene>, a: &[crate::field::Amplitude], tweezer: usize) -> (C64, [C64; 3]) {
        match self {
            Factor::Scene(f) => {
                let (v, g) = f.at(a);
                if self.varying(scene) { (v, g) } else { (v, [C64::ZERO; 3]) }
            }
            Factor::Power => {
                let t = &a[tweezer];
                (C64::real(t.value.norm_sqr()), std::array::from_fn(|i| C64::real(2.0 * (t.value.conj() * t.gradient[i]).re)))
            }
            Factor::Fixed(v, g) => (v, g),
        }
    }
}

/// A list of distinct factors, each term's an index into it.
#[derive(Clone, Debug, Default, PartialEq)]
struct Factors(Vec<Factor>);

impl Factors {
    fn of(&mut self, f: Factor) -> usize {
        match self.0.iter().position(|x| *x == f) {
            Some(i) => i,
            None => {
                self.0.push(f);
                self.0.len() - 1
            }
        }
    }
}

/// A source's dense sum of terms on the 32 states at the values, and of their gradients.
fn dense(library: &[Fixed], values: &Values, terms: &[Term]) -> (Mat, [Mat; 3]) {
    let (mut value, mut gradient) = (Mat::zeros(STATES, STATES), zeros(STATES, STATES));
    for t in terms {
        let (f, df) = values.factors[t.factor];
        for &(m, c) in &t.matrices {
            let c = values.coefficients[c];
            if c == 0.0 {
                continue;
            }
            for &(i, j, v) in &library[m] {
                let (i, j, v) = if t.adjoint { (j, i, v.conj()) } else { (i, j, v) };
                let x = t.scale * v * c;
                value[(i, j)] += f * x;
                for (g, d) in gradient.iter_mut().zip(df) {
                    g[(i, j)] += d * x;
                }
            }
        }
    }
    (value, gradient)
}

impl Source {
    /// The source as a dense model on the 32 states at the values (the full reference's `ModelAt`; floquet's
    /// `Hilbert::full`) and its generator.
    pub fn model(&self, library: &[Fixed], values: &Values, groups: &[Vec<usize>]) -> (ModelAt, ModelPart) {
        let mut frequencies = values.frequencies.iter();
        let mut part = |p: &Piece| {
            let (value, gradient) = dense(library, values, &p.terms);
            ModelPart { key: p.key.clone(), frequency: *frequencies.next().expect("a piece's frequency"), value, gradient }
        };
        let effective: Vec<ModelPart> = self.hamiltonian.iter().map(&mut part).collect();
        let jumps = self.jumps.iter().map(|c| Channel { emission: c.emission, fast: c.fast, parts: c.parts.iter().map(&mut part).collect(), kicks: c.kicks.iter().map(&mut part).collect() }).filter(|c| c.parts.iter().any(|p| nonzero(&p.value) || p.gradient.iter().any(nonzero))).collect();
        let model = ModelAt { states: (0..STATES).collect(), folded: Vec::new(), groups: groups.to_vec(), effective, jumps, lift: Vec::new() };
        let (value, gradient) = dense(library, values, &self.generator);
        let length = self.hamiltonian.first().map_or(0, |p| p.key.len());
        (model, ModelPart { key: vec![0; length], frequency: 0.0, value, gradient })
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The fold at a point: its numbers and its contractions.

/// A fold's structure with its groupings by key and in one group (`Rs`'s, or a check's of a given source).
#[derive(Clone, Debug)]
pub struct Folding {
    pub plan: fold::Plan,
    keys: Vec<Key>,
    by_key: fold::Grouping,
    one: fold::Grouping,
}

impl Folding {
    /// The structure of `source`'s fold keeping `keep` (sorted), the rest Q, by `sectors` on `axes` (rs/fold.rs `Plan`).
    pub fn new(library: &[Fixed], source: &Source, keep: &[usize], sectors: &Sectors, axes: [bool; 2], groups: &[Vec<usize>]) -> Folding {
        let folded: Vec<usize> = (0..STATES).filter(|i| keep.binary_search(i).is_err()).collect();
        let splits = Split::all(library, keep, sectors, axes);
        let plan = fold::Plan::new(&splits, keep, &folded, source, sectors, axes, groups);
        let keys: Vec<Key> = source.pieces().map(|(p, _)| p.key.clone()).collect();
        let by_key = fold::Grouping::new(&plan, &keys);
        let one = fold::Grouping::new(&plan, &vec![Vec::new(); keys.len()]);
        Folding { plan, keys, by_key, one }
    }

    /// The keys of the fold's pieces (`Source::pieces`' order).
    pub fn keys(&self) -> &[Key] {
        &self.keys
    }

    /// The fold at the values (`At`).
    pub fn at(&self, values: &Values) -> At<'_> {
        let mut local = fold::Local::default();
        local.evaluate(&self.plan, values);
        At { rs: self, local }
    }
}

/// The fold at a point (`Rs::local`): its numbers (rs/fold.rs `Local`) on the fold's structure, read by contractions.
#[derive(Clone, Debug)]
pub struct At<'a> {
    rs: &'a Folding,
    pub local: fold::Local,
}

/// A contraction's grouping of the components: of a component's key and frequency its group's key, its frequency and
/// the weight it enters with.
pub type Grouping<'a> = &'a dyn Fn(&Key, f64) -> (Key, f64, C64);

impl At<'_> {
    /// The contraction by a grouping given as a function (a check's: its grouping made here).
    pub fn contract(&self, group: Grouping, making: Making) -> ModelAt {
        let plan = &self.rs.plan;
        let freqs = &self.local.values.frequencies;
        let made: Vec<(Key, f64, C64)> = (0..plan.pieces()).map(|p| group(&self.rs.keys[p], freqs[p])).collect();
        let grouping = fold::Grouping::new(plan, &made.iter().map(|m| m.0.clone()).collect::<Vec<_>>());
        let weights: Vec<C64> = made.iter().map(|m| m.2).collect();
        let mut out = fold::Out::default();
        out.contract(plan, &grouping, &self.local, Some(&weights), making);
        model_of(plan, &grouping, &out, freqs, making)
    }

    /// The model by frequency components (the checkpoint's): every component its own group, its key.
    pub fn components(&self) -> ModelAt {
        let (plan, grouping) = (&self.rs.plan, &self.rs.by_key);
        let mut out = fold::Out::default();
        out.contract(plan, grouping, &self.local, None, Making::ALL);
        model_of(plan, grouping, &out, &self.local.values.frequencies, Making::ALL)
    }

    /// The contraction by one group, each component at its phase e^{−if_c t}.
    fn at_time(&self, t: f64, making: Making) -> ModelAt {
        let (plan, grouping) = (&self.rs.plan, &self.rs.one);
        let weights: Vec<C64> = self.local.values.frequencies.iter().map(|f| C64::cis(-f * t)).collect();
        let mut out = fold::Out::default();
        out.contract(plan, grouping, &self.local, Some(&weights), making);
        model_of(plan, grouping, &out, &self.local.values.frequencies, making)
    }

    /// The model's value at the time t (the trajectory's): one group, each component at its phase e^{−if_c t}: H_eff(t)
    /// with its gradient, each channel's L(t) with its gradient, and the lift M(t) on Q (|Q| × n).
    pub fn now(&self, t: f64) -> Now {
        Now::of(self.at_time(t, Making { model: true, kicks: false, lift: true }))
    }

    /// The channels' photons' forms at the time t (`now`'s channels' kicks: k∘L per element, folded alike).
    pub fn kicks(&self, t: f64) -> Vec<Moment> {
        moments(&self.at_time(t, Making { model: true, kicks: true, lift: false }), true)
    }
}

/// A contraction's outputs as a model by its outputs' keys (each part's frequency its pieces' sum), the parts in key
/// order, nothing that is zero.
pub fn model_of(plan: &fold::Plan, grouping: &fold::Grouping, out: &fold::Out, frequencies: &[f64], making: Making) -> ModelAt {
    let frequency = |pieces: &[usize]| -> f64 { pieces.iter().map(|&p| frequencies[p]).sum() };
    let keep = |p: &ModelPart| nonzero(&p.value) || p.gradient.iter().any(nonzero);
    let mut effective: BTreeMap<Key, ModelPart> = BTreeMap::new();
    if making.model {
        for (((key, pieces), c), list) in grouping.outputs.iter().zip(&out.h).zip(&grouping.lists) {
            let (value, gradient) = c.dense(list, plan.n);
            let f = pieces.iter().map(|&(p, sign)| sign * frequencies[p]).sum();
            effective.insert(key.clone(), ModelPart { key: key.clone(), frequency: f, value, gradient });
        }
    }
    let mut channels: BTreeMap<(usize, [i64; 2]), [BTreeMap<Key, ModelPart>; 2]> = BTreeMap::new();
    if making.model {
        for (((emission, fast, which, key, pieces), c), list) in grouping.channel_outputs.iter().zip(&out.channels).zip(&grouping.channel_lists) {
            if *which == 1 && !making.kicks {
                continue;
            }
            let (value, gradient) = c.dense(list, plan.n);
            let entry = channels.entry((*emission, *fast)).or_default();
            entry[*which].insert(key.clone(), ModelPart { key: key.clone(), frequency: frequency(pieces), value, gradient });
        }
    }
    let jumps = channels
        .into_iter()
        .map(|((emission, fast), [parts, kicks])| Channel { emission, fast, parts: parts.into_values().filter(keep).collect(), kicks: kicks.into_values().filter(keep).collect() })
        .filter(|c| !c.parts.is_empty())
        .collect();
    let mut lift: BTreeMap<Key, ModelPart> = BTreeMap::new();
    if making.lift {
        for ((key, piece), (v, g)) in grouping.labels.iter().zip(&out.lift) {
            if v.rows == 0 {
                continue;
            }
            lift.insert(key.clone(), ModelPart { key: key.clone(), frequency: frequencies[*piece], value: v.clone(), gradient: g.clone() });
        }
    }
    ModelAt {
        states: plan.keep.clone(),
        folded: plan.folded.clone(),
        groups: plan.groups.clone(),
        effective: effective.into_values().filter(keep).collect(),
        jumps,
        lift: lift.into_values().filter(|p| nonzero(&p.value)).collect(),
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

/// The model's value at a time t (`At::now`): H_eff(t) and ∂H_eff(t) (n × n), the channels L(t), the lift M(t) on Q.
#[derive(Clone, Debug, PartialEq)]
pub struct Now {
    pub states: Vec<usize>,
    pub folded: Vec<usize>,
    pub hamiltonian: Mat,
    pub gradient: [Mat; 3],
    pub jumps: Vec<Moment>,
    pub lift: Mat,
}

/// The 32 states' populations of ρ (n × n, row-major, on the model's `states`) with the `folded` states' lifted through
/// M (|Q| × n): the model's ρ_aa, Q's (M ρ M†)_kk. The one readout of a fold's populations.
pub fn lifted(states: &[usize], folded: &[usize], lift: &Mat, rho: &[C64]) -> [f64; STATES] {
    let n = states.len();
    let mut out = [0.0; STATES];
    for (a, &state) in states.iter().enumerate() {
        out[state] = rho[a * n + a].re;
    }
    let m = lift.product(&Mat { rows: n, cols: n, data: rho.to_vec() }).product(&lift.adjoint());
    for (k, &state) in folded.iter().enumerate() {
        out[state] = m[(k, k)].re;
    }
    out
}

/// A one-group contraction's channels (`At::now`) as moments: their parts', or their photons' forms' (`kicks`).
fn moments(model: &ModelAt, kicks: bool) -> Vec<Moment> {
    let n = model.states.len();
    model
        .jumps
        .iter()
        .map(|c| {
            let parts = if kicks { &c.kicks } else { &c.parts };
            let (value, gradient) = parts.first().map_or_else(|| (Mat::zeros(n, n), zeros(n, n)), |p| (p.value.clone(), p.gradient.clone()));
            Moment { emission: c.emission, fast: c.fast, value, gradient }
        })
        .collect()
}

impl Now {
    /// The value of a one-group contraction (`At::now`): its one part of each quantity, zero where it has none.
    fn of(model: ModelAt) -> Now {
        let (n, q) = (model.states.len(), model.folded.len());
        let (hamiltonian, gradient) = model.effective.first().map_or_else(|| (Mat::zeros(n, n), zeros(n, n)), |p| (p.value.clone(), p.gradient.clone()));
        let lift = model.lift.first().map_or_else(|| Mat::zeros(q, n), |p| p.value.clone());
        Now { jumps: moments(&model, false), states: model.states, folded: model.folded, hamiltonian, gradient, lift }
    }

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

    /// The 32 states' populations of ρ (n × n, row-major) lifted through M(t) (`lifted`).
    pub fn lift(&self, rho: &[C64]) -> [f64; STATES] {
        lifted(&self.states, &self.folded, &self.lift, rho)
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

/// The request's Cartesian emission as fixed matrices: C_a, k∘C_a (the photons' forms), and Σ_a C_a†C_a (scene.rs
/// `Jumps::decay`, the one decay).
pub fn emission(jumps: &Jumps) -> ([Mat; 3], [Mat; 3], Mat) {
    let photon = |c: &Mat| Mat::from_fn(STATES, STATES, |i, j| c[(i, j)] * jumps.wave_number[(i, j)].re);
    let kicks = jumps.emission.each_ref().map(photon);
    (jumps.emission.clone(), kicks, jumps.decay())
}

/// The emission's three channels (n_tw = n_φ = 0) on the library's matrices `parts` and `kicks`, factor `one`.
pub fn emission_channels(length: usize, one: usize, parts: [usize; 3], kicks: [usize; 3]) -> Vec<SourceChannel> {
    let piece = |m: usize| Piece { key: vec![0; length], terms: vec![Term::one(one, m)] };
    (0..3).map(|a| SourceChannel { emission: a, fast: [0; 2], parts: vec![piece(parts[a])], kicks: vec![piece(kicks[a])] }).collect()
}

/// The powers of the tweezer's Doppler shift its unit fold is made at, δ⁰ … δ^(ORDERS − 1) (module documentation, # The
/// two folds): a point takes them while the next is above the rounding of the first, and asserts that the last made is
/// not needed (it would be at |δ| ≈ 10³ rad/µs, an atom a thousand times faster than escapes any trap).
pub const ORDERS: usize = 4;

/// The fixed matrices of the folds and the tweezer's sidebands folded once (module documentation, # The two folds): the
/// library holds the scene's matrices (`Scene::matrix`, at their indices), the emission's C_a, k∘C_a and Σ C_a†C_a, and
/// the unit fold's pairs (n_tw = 0, H's part ∝ |a_tw|²), its channels (n_tw = ±1, ∝ a_tw and a_tw*) and their decay, each
/// at the powers of the tweezer's Doppler shift; the source's structure made of them.
#[derive(Clone, Debug, PartialEq)]
pub struct Tweezer {
    library: Vec<Fixed>,
    sectors: Sectors,
    groups: Vec<Vec<usize>>,
    source: Source,
    /// The same source with its channels in the emission's spherical basis (`channel_basis`): the decay's structure.
    spherical: Source,
    /// The spherical basis: per channel q its Cartesian axes' coefficients (ℓ_q = Σ_a U_qa L_a).
    pub basis: [[C64; 3]; 3],
    factors: Factors,
    /// Per sign σ of the tweezer's absorbed photon (+: its keys e_tw, e_tw + 2e_ν; −: their negations) and power j the
    /// largest entry of its matrices (the series' rule), and the static part at δ = 0 (`still`).
    sizes: [[f64; ORDERS]; 2],
    still: Mat,
}

/// The emission's channels in the basis that holds the fewest entries (by the rounding rule, `rounded`) among the
/// Cartesian axes and the spherical bases about each of them, ℓ_± = ∓(L_b ± iL_c)/√2, ℓ_0 = L_a (a, b, c cyclic): the
/// basis of the states' quantization axis, where each channel changes m by one number and the decay Σ_q ℓ_q†ℓ_q =
/// Σ_a L_a†L_a holds no element of the axes' cancellations (Δm = 2 of the Cartesian x and y: diaga.status.md 4.3). Any
/// unitary basis gives the same decay; this one gives it its structure.
pub fn channel_basis(c: &[Mat; 3]) -> [[C64; 3]; 3] {
    let r = std::f64::consts::FRAC_1_SQRT_2;
    let cyclic = |a: usize| -> [[C64; 3]; 3] {
        let (b, cc) = ((a + 1) % 3, (a + 2) % 3);
        let mut u = [[C64::ZERO; 3]; 3];
        u[0][b] = C64::real(-r);
        u[0][cc] = C64::new(0.0, -r);
        u[1][b] = C64::real(r);
        u[1][cc] = C64::new(0.0, -r);
        u[2][a] = C64::ONE;
        u
    };
    let identity = [[C64::ONE, C64::ZERO, C64::ZERO], [C64::ZERO, C64::ONE, C64::ZERO], [C64::ZERO, C64::ZERO, C64::ONE]];
    let count = |u: &[[C64; 3]; 3]| -> usize {
        u.iter()
            .map(|row| {
                let mut m = Mat::zeros(STATES, STATES);
                for (a, &x) in row.iter().enumerate() {
                    if x != C64::ZERO {
                        m.add_scaled(x, &c[a]);
                    }
                }
                rounded(&m).data.iter().filter(|z| **z != C64::ZERO).count()
            })
            .sum()
    };
    let mut best = (count(&identity), identity);
    for a in [2, 0, 1] {
        let u = cyclic(a);
        let n = count(&u);
        if n < best.0 {
            best = (n, u);
        }
    }
    best.1
}

/// Σ_a U_qa M_a of three matrices.
fn combined(row: &[C64; 3], m: [&Mat; 3]) -> Mat {
    let mut out = Mat::zeros(m[0].rows, m[0].cols);
    for (x, mi) in row.iter().zip(m) {
        if *x != C64::ZERO {
            out.add_scaled(*x, mi);
        }
    }
    out
}

/// The coefficient index of the power j ≥ 1 of the sign σ (0: +, 1: −); coefficient 0 is one.
fn power(sign: usize, j: usize) -> usize {
    if j == 0 { 0 } else { 1 + sign * (ORDERS - 1) + (j - 1) }
}

impl Tweezer {
    /// The fold of the tweezer's couplings at unit amplitude (`Scene::tweezer_couplings`, both ways) with the emission's
    /// channels, kept on the 32 states, at the request's H0, at each power of its Doppler shift; and the source's
    /// structure: the scene's terms (the tweezer's left out) by key, each nonzero key both ways, the unit fold's pairs at
    /// |a|², its channels at a and a*, the emission's channels, the generator H's static part − (i/2)(Σ C†C + |a|² Σ Ĵ†Ĵ).
    pub fn new(scene: &Scene) -> Tweezer {
        let length = scene.generators.len();
        let mut library: Vec<Fixed> = (0..scene.matrices()).map(|i| fixed(scene.matrix(i))).collect();
        let put = |library: &mut Vec<Fixed>, m: &Mat| {
            library.push(fixed(m));
            library.len() - 1
        };
        let (c, k, gamma) = emission(scene.jumps());
        let emission_m = c.each_ref().map(|m| put(&mut library, m));
        let kicks_m = k.each_ref().map(|m| put(&mut library, m));
        let rate = put(&mut library, &gamma);
        let unit = unit_fold(scene, &c, &k, &gamma);
        let mut factors = Factors::default();
        let one = factors.of(Factor::Scene(scene::Factor::One));
        let power_f = factors.of(Factor::Power);
        let tw = scene.tweezer();
        let a = factors.of(Factor::Scene(scene::Factor::Beam(tw)));
        let a_conj = factors.of(Factor::Scene(scene::Factor::Conjugate(tw)));
        let sectors = Sectors::tweezer(scene);
        // The scene's terms by key, the tweezer's left out; each nonzero key both ways.
        let zero_doppler = vec![0.0; scene.request.beams.len()];
        let mut pieces: Vec<Piece> = Vec::new();
        for t in scene.terms(&zero_doppler).iter().filter(|t| sectors.key(&t.key, REFERENCE) == [0; 2]) {
            let term = Term { factor: factors.of(Factor::Scene(t.factor)), adjoint: t.adjoint, scale: C64::ONE, matrices: vec![(t.matrix, 0)] };
            match pieces.iter_mut().rev().find(|p| p.key == t.key) {
                Some(p) => p.terms.push(term),
                None => pieces.push(Piece { key: t.key.clone(), terms: vec![term] }),
            }
        }
        let mut hamiltonian = Vec::with_capacity(2 * pieces.len());
        for p in pieces {
            let adjoint = p.key.iter().any(|k| *k != 0).then(|| Piece {
                key: negated(&p.key),
                terms: p.terms.iter().map(|t| Term { factor: factors.of(factors.0[t.factor].conj()), adjoint: !t.adjoint, scale: t.scale.conj(), matrices: t.matrices.clone() }).collect(),
            });
            hamiltonian.push(p);
            hamiltonian.extend(adjoint);
        }
        // The unit fold's pairs by key, each with its powers.
        let mut sizes = [[0.0_f64; ORDERS]; 2];
        for (key, by) in &unit.shift {
            let matrices: Vec<(usize, usize)> = by.iter().map(|(sign, j, m)| {
                sizes[*sign][*j] = crate::linalg::larger(sizes[*sign][*j], m.max_abs());
                (put(&mut library, &rounded(m)), power(*sign, *j))
            }).collect();
            let term = Term { factor: power_f, adjoint: false, scale: C64::ONE, matrices };
            match hamiltonian.iter_mut().find(|p| &p.key == key) {
                Some(p) => p.terms.push(term),
                None => hamiltonian.push(Piece { key: key.clone(), terms: vec![term] }),
            }
        }
        let mut jumps = emission_channels(length, one, emission_m, kicks_m);
        for ((emission, fast), (parts, kicks)) in &unit.channels {
            let factor = if fast[1] > 0 { a } else { a_conj };
            let piece = |library: &mut Vec<Fixed>, sizes: &mut [[f64; ORDERS]; 2], (key, by): (&Key, &Vec<(usize, usize, Mat)>)| Piece {
                key: key.clone(),
                terms: vec![Term {
                    factor,
                    adjoint: false,
                    scale: C64::ONE,
                    matrices: by.iter().map(|(sign, j, m)| {
                        sizes[*sign][*j] = crate::linalg::larger(sizes[*sign][*j], m.max_abs());
                        (put(library, &rounded(m)), power(*sign, *j))
                    }).collect(),
                }],
            };
            let parts = parts.iter().map(|x| piece(&mut library, &mut sizes, x)).collect();
            let kicks = kicks.iter().map(|x| piece(&mut library, &mut sizes, x)).collect();
            jumps.push(SourceChannel { emission: *emission, fast: *fast, parts, kicks });
        }
        // The same channels in the spherical basis: the emission's, and the tweezer's per fast index, key, sign and power.
        let basis = channel_basis(&c);
        let mut spherical_jumps = Vec::new();
        for (q, row) in basis.iter().enumerate() {
            let m = put(&mut library, &rounded(&combined(row, [&c[0], &c[1], &c[2]])));
            spherical_jumps.push(SourceChannel { emission: q, fast: [0; 2], parts: vec![Piece { key: vec![0; length], terms: vec![Term::one(one, m)] }], kicks: Vec::new() });
        }
        let fasts: Vec<[i64; 2]> = {
            let mut f: Vec<[i64; 2]> = unit.channels.keys().map(|(_, fast)| *fast).collect();
            f.sort_unstable();
            f.dedup();
            f
        };
        for fast in fasts {
            let factor = if fast[1] > 0 { a } else { a_conj };
            let axes: Vec<&BTreeMap<Key, Vec<(usize, usize, Mat)>>> = (0..3).map(|axis| &unit.channels[&(axis, fast)].0).collect();
            for (q, row) in basis.iter().enumerate() {
                let mut parts = Vec::new();
                for (key, by) in axes[0] {
                    let mut matrices = Vec::new();
                    for (sign, j, _) in by {
                        let pick = |axis: usize| -> &Mat { &axes[axis][key].iter().find(|x| x.0 == *sign && x.1 == *j).expect("the axes' powers").2 };
                        let m = rounded(&combined(row, [pick(0), pick(1), pick(2)]));
                        matrices.push((put(&mut library, &m), power(*sign, *j)));
                    }
                    parts.push(Piece { key: key.clone(), terms: vec![Term { factor, adjoint: false, scale: C64::ONE, matrices }] });
                }
                spherical_jumps.push(SourceChannel { emission: q, fast, parts, kicks: Vec::new() });
            }
        }
        let mut generator = hamiltonian.iter().find(|p| p.key.iter().all(|k| *k == 0)).expect("H's static part").terms.clone();
        generator.push(Term { factor: one, adjoint: false, scale: C64::new(0.0, -0.5), matrices: vec![(rate, 0)] });
        let scattering: Vec<(usize, usize)> = unit.scattering.iter().map(|(sign, j, m)| (put(&mut library, &rounded(m)), power(*sign, *j))).collect();
        generator.push(Term { factor: power_f, adjoint: false, scale: C64::new(0.0, -0.5), matrices: scattering });
        let varying = factors.0.iter().map(|f| f.varying(Some(scene))).collect();
        let source = Source { hamiltonian, jumps, generator, varying, coefficients: 1 + 2 * (ORDERS - 1) };
        let spherical = Source { jumps: spherical_jumps, ..source.clone() };
        let all: Vec<usize> = (0..STATES).collect();
        let groups = levels(&scene.request.states, &all);
        Tweezer { library, sectors, groups, source, spherical, basis, factors, sizes, still: unit.still }
    }

    /// The source's numbers at a point (scene.rs `Point`): the factors at its amplitudes, the tweezer's Doppler powers
    /// (its key's frequency at rest less at the point: +w_tw for e_tw, −w_tw for −e_tw), each piece's frequency.
    pub fn values(&self, scene: &Scene, point: &Point) -> Values {
        let mut v = Values::default();
        self.values_into(scene, point, &mut v);
        v
    }

    /// `values` into a buffer.
    pub fn values_into(&self, scene: &Scene, point: &Point, v: &mut Values) {
        let tw = scene.tweezer();
        v.factors.clear();
        v.factors.extend(self.factors.0.iter().map(|f| f.at(Some(scene), &point.amplitudes, tw)));
        let delta = scene.generators[tw].frequency - point.rates[tw];
        v.coefficients.clear();
        v.coefficients.resize(self.source.coefficients, 0.0);
        v.coefficients[0] = 1.0;
        for (sign, d) in [delta, -delta].into_iter().enumerate() {
            let sizes = &self.sizes[sign];
            if sizes[0] == 0.0 {
                continue;
            }
            for j in 1..ORDERS {
                let term = d.powi(j as i32);
                if !(term.abs() * sizes[j] < UNIT_ROUNDOFF * sizes[0]) {
                    assert!(j + 1 < ORDERS, "the tweezer's Doppler series needs more than {ORDERS} powers at δ = {d} rad/µs");
                    v.coefficients[power(sign, j)] = term;
                } else {
                    break;
                }
            }
        }
        v.frequencies.clear();
        v.frequencies.extend(self.source.pieces().map(|(p, _)| point.frequency(&p.key)));
    }

    /// The unit fold's static part on the 32 states at rest (rad/µs at the tweezer's focus intensity): its ground
    /// diagonal the tweezer's ground shift (`Scene::trap_potential`'s depth), its excited block the tweezer's shifts of
    /// the excited states (`Scene::smooth`).
    pub fn still(&self) -> Mat {
        self.still.clone()
    }

    /// The full reference at a point as a dense model on the 32 states, with its generator.
    pub fn reference(&self, scene: &Scene, point: &Point) -> (ModelAt, ModelPart) {
        self.source.model(&self.library, &self.values(scene, point), &self.groups)
    }
}

/// The tweezer's unit fold (`Tweezer::new`): its pairs by key with (sign, power, matrix), its channels by (emission, fast
/// index) with their parts and photon forms by key, the decay of its channels by (sign, power), and its static part at
/// rest.
struct Unit {
    shift: BTreeMap<Key, Vec<(usize, usize, Mat)>>,
    channels: BTreeMap<(usize, [i64; 2]), (BTreeMap<Key, Vec<(usize, usize, Mat)>>, BTreeMap<Key, Vec<(usize, usize, Mat)>>)>,
    scattering: Vec<(usize, usize, Mat)>,
    still: Mat,
}

/// The unit fold (module documentation, # The two folds): G₀ = H0 − (i/2)Σ C†C on the 32 states, its connected parts;
/// the tweezer's four components c (co e_tw at ω_tw, its conjugate, the counter-rotating at −e_tw − 2e_ν, its conjugate),
/// each column's resolvent at E_a + f_c by the Neumann series about the diagonal and its powers R^{j+1} H_c; the pairs
/// whose n_tw is zero, ½(H_c′ R^{j+1} H_c + h.c.) at c + c′ and its negation; the channels C_a R^{j+1} H_c and
/// (k∘C_a) R^{j+1} H_c at n_tw of c; the channels' decay Σ_a Σ_key Σ_{j + j′ = J} L_j†L_j′ by power J, each axis whole.
fn unit_fold(scene: &Scene, c: &[Mat; 3], k: &[Mat; 3], gamma: &Mat) -> Unit {
    let h0 = &scene.request.hamiltonian;
    let g = Mat::from_fn(STATES, STATES, |i, j| h0[(i, j)] - gamma[(i, j)] * C64::new(0.0, 0.5));
    let energy: Vec<f64> = (0..STATES).map(|a| g[(a, a)].re).collect();
    let pattern: Vec<(usize, usize)> = (0..STATES).flat_map(|i| (0..STATES).map(move |j| (i, j))).filter(|&(i, j)| i != j && g[(i, j)] != C64::ZERO).collect();
    let blocks = fold::connected_parts(STATES, &pattern);
    let [(co_key, f, x), (cr_key, fr, r)] = scene.tweezer_couplings();
    let comps: [(Key, f64, Mat, i64, usize); 4] = [
        (co_key.clone(), f, x.clone(), 1, 0),
        (negated(&co_key), -f, x.adjoint(), -1, 1),
        (cr_key.clone(), fr, r.clone(), -1, 1),
        (negated(&cr_key), -fr, r.adjoint(), 1, 0),
    ];
    // R^{j+1} H_c for each component and power, M_c[B, a] = (E_a + f_c − G₀_BB)⁻¹ H_Ba on the blocks B and columns a
    // H_c holds (the request's structure: the couplings excited ← ground, `Request::validate`; a power's the same):
    // elsewhere M_c is zero, the resolvent acting on nothing -- and there it may be singular, a ground state's own
    // block at E_a + f_c = E_a where the tweezer is the frame's carrier (f_c = 0: a lone tweezer, scene.rs `tweezer_axis`).
    let resolved: Vec<Vec<Mat>> = comps
        .iter()
        .map(|(_, freq, h, _, _)| {
            let z: Vec<f64> = energy.iter().map(|e| e + freq).collect();
            let held: Vec<(&Vec<usize>, Vec<usize>)> = blocks
                .iter()
                .map(|rows| (rows, (0..STATES).filter(|&a| rows.iter().any(|&r| h[(r, a)] != C64::ZERO)).collect::<Vec<usize>>()))
                .filter(|(_, cols)| !cols.is_empty())
                .collect();
            let mut out = Vec::with_capacity(ORDERS);
            let mut y = h.clone();
            for _ in 0..ORDERS {
                let mut m = Mat::zeros(STATES, STATES);
                for (rows, cols) in &held {
                    let sub = g.block(rows, rows);
                    let yb = y.block(rows, cols);
                    let zb: Vec<f64> = cols.iter().map(|&a| z[a]).collect();
                    let mut mb = Mat::default();
                    fold::neumann_into(&sub, &zb, &yb, &mut mb);
                    for (i, &row) in rows.iter().enumerate() {
                        for (k, &a) in cols.iter().enumerate() {
                            m[(row, a)] = mb[(i, k)];
                        }
                    }
                }
                y = m.clone();
                out.push(m);
            }
            out
        })
        .collect();
    let mut shift: BTreeMap<Key, Vec<(usize, usize, Mat)>> = BTreeMap::new();
    let add = |map: &mut BTreeMap<Key, Vec<(usize, usize, Mat)>>, key: Key, sign: usize, j: usize, m: &Mat| {
        let list = map.entry(key).or_default();
        match list.iter_mut().find(|(s, p, _)| *s == sign && *p == j) {
            Some((_, _, x)) => x.add_scaled(C64::ONE, m),
            None => list.push((sign, j, m.clone())),
        }
    };
    let mut still = Mat::zeros(STATES, STATES);
    for (ci, (key_c, _, _, class_c, sign)) in comps.iter().enumerate() {
        for (key_w, _, h_w, class_w, _) in &comps {
            if class_c + class_w != 0 {
                continue;
            }
            let key = sum_keys(key_c, key_w);
            for (j, m) in resolved[ci].iter().enumerate() {
                let a = h_w.product(m).scaled(C64::real(0.5));
                let adj = a.adjoint();
                if j == 0 && key.iter().all(|x| *x == 0) {
                    still.add_scaled(C64::ONE, &a);
                    still.add_scaled(C64::ONE, &adj);
                }
                add(&mut shift, key.clone(), *sign, j, &a);
                add(&mut shift, negated(&key), *sign, j, &adj);
            }
        }
    }
    type ByKey = BTreeMap<Key, Vec<(usize, usize, Mat)>>;
    let mut channels: BTreeMap<(usize, [i64; 2]), (ByKey, ByKey)> = BTreeMap::new();
    for (ci, (key_c, _, _, class_c, sign)) in comps.iter().enumerate() {
        for axis in 0..3 {
            let entry = channels.entry((axis, [0, *class_c])).or_default();
            for (j, m) in resolved[ci].iter().enumerate() {
                add(&mut entry.0, key_c.clone(), *sign, j, &c[axis].product(m));
                add(&mut entry.1, key_c.clone(), *sign, j, &k[axis].product(m));
            }
        }
    }
    // The decay of the channels by sign and power: each axis whole, then the axes.
    let mut scattering: Vec<(usize, usize, Mat)> = Vec::new();
    for sign in 0..2 {
        for power in 0..ORDERS {
            let mut total = Mat::zeros(STATES, STATES);
            for axis in 0..3 {
                let mut g = Mat::zeros(STATES, STATES);
                for (_, (parts, _)) in channels.iter().filter(|((e, _), _)| *e == axis) {
                    for list in parts.values() {
                        for (s1, j1, l1) in list.iter().filter(|x| x.0 == sign) {
                            for (_, _, l2) in list.iter().filter(|x| x.0 == *s1 && x.1 + j1 == power) {
                                adjoint_product_onto(l1, l2, C64::ONE, &mut g);
                            }
                        }
                    }
                }
                total.add_scaled(C64::ONE, &g);
            }
            if nonzero(&total) {
                scattering.push((sign, power, total));
            }
        }
    }
    Unit { shift, channels, scattering, still }
}

/// The full reference for the scene's components at a point: the scene's model on the 32 states with the tweezer's
/// sidebands folded (floquet's `Hilbert::full` is it).
pub fn reference(scene: &Scene, components: &Components) -> ModelAt {
    Tweezer::new(scene).reference(scene, &components.point).0
}

/// The RS model on a kept set P: P, its complement, the frame's sectors, the tweezer's fold and the fold's structure
/// with its groupings by key and in one group, made once for the run.
#[derive(Clone, Debug)]
pub struct Rs {
    /// P: indices into the 32, sorted (the ground states first).
    pub keep: Vec<usize>,
    /// Q: the folded states, the rest of the 32.
    pub folded: Vec<usize>,
    pub sectors: Sectors,
    tweezer: Tweezer,
    pub folding: Folding,
}

impl Rs {
    /// The model's fold on P = `keep` with the frame's sectors; the tweezer's fold, the splits and the fold's structure
    /// made here once. The tweezer's sidebands are its photons: a generator's n_tw is its count of them (the tweezer's
    /// one, every other generator's none: `Sectors::tweezer`, the reference's), a state's the frame's, zero (θ holds no
    /// tweezer photon: floquet/frame.rs chooses it on the other light). The frame counts them so on the tweezer's own
    /// axis; a tweezer without other light has none (its index zero, its frequency the frame's carrier: scene.rs
    /// `tweezer_axis`), and its co- and counter-rotating channels, 2ν apart, are told apart by the count alone.
    pub fn new(scene: &Scene, keep: &[usize], sectors: Sectors) -> Rs {
        let mut sectors = sectors;
        let tw = scene.tweezer();
        for (g, s) in sectors.generators.iter_mut().enumerate() {
            s[1] = i64::from(g == tw);
        }
        assert!(sectors.states.iter().all(|s| s[1] == 0), "θ holds no tweezer photon");
        let mut keep = keep.to_vec();
        keep.sort_unstable();
        keep.dedup();
        let folded: Vec<usize> = (0..STATES).filter(|i| keep.binary_search(i).is_err()).collect();
        let tweezer = Tweezer::new(scene);
        let folding = Folding::new(&tweezer.library, &tweezer.source, &keep, &sectors, MODEL, &tweezer.groups);
        Rs { keep, folded, sectors, tweezer, folding }
    }

    /// The fold's structure.
    pub fn plan(&self) -> &fold::Plan {
        &self.folding.plan
    }

    /// The fold at a point (scene.rs `Point`: `Scene::point_at` on a trajectory, `Scene::components(..).point` at the
    /// checkpoint's): the full reference there folded on P at its static part. Its contractions: `At::now` (the value at
    /// a time) and `At::components`.
    pub fn local(&self, scene: &Scene, point: &Point) -> At<'_> {
        let mut local = fold::Local::default();
        self.evaluate(scene, point, &mut local);
        At { rs: &self.folding, local }
    }

    /// The fold's numbers at a point into a buffer (`local`'s).
    pub fn evaluate(&self, scene: &Scene, point: &Point, local: &mut fold::Local) {
        let mut values = std::mem::take(&mut local.values);
        self.tweezer.values_into(scene, point, &mut values);
        local.evaluate(&self.folding.plan, &values);
    }

    /// The tweezer's fold made once (`Tweezer::still`).
    pub fn tweezer(&self) -> &Tweezer {
        &self.tweezer
    }

    /// The fold's structure with its channels in the emission's spherical basis (`channel_basis`): the decay's
    /// structure, made where asked for (a trajectory's model, once).
    pub fn spherical(&self) -> fold::Plan {
        Folding::new(&self.tweezer.library, &self.tweezer.spherical, &self.keep, &self.sectors, MODEL, &self.tweezer.groups).plan
    }

    /// The model by frequency components for the scene's components at a point (the checkpoint's).
    pub fn at(&self, scene: &Scene, components: &Components) -> ModelAt {
        self.local(scene, &components.point).components()
    }
}

#[cfg(test)]
mod tests;
