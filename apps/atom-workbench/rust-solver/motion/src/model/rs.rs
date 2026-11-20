//! The one fold (RUSTPLAN §2.1–§2.3 and its corrections of 2026-11-19: "模型只定一次", "一个折叠公式，快分量与光镊都由它
//! 折叠"): Reiter–Sørensen second order with harmonics and dissipation (td_s.hilbert_rs, each column at E_a + f), the
//! full reference's and the model's alike -- they differ only in what is folded.
//!
//! # The fold
//! A Lindblad model on a set of states by frequency components (`ModelAt`): H's static part H⁽⁰⁾ and components H⁽ᶜ⁾
//! e^{−if_c t} taken both ways (each nonzero key with its conjugate at −key), and jump channels L_ch = Σ_c L_ch⁽ᶜ⁾
//! e^{−if_c t}. A fold keeps the states K and, among their elements, the slow ones: the element b ← a of the component c
//! has the fast index n = Σ_l c_l n_l + n_b − n_a (`Sectors`, the frame's integer fast parts of the key coordinates and
//! of the states, floquet/frame.rs), slow where its folding coordinates (`axes`) are zero. Everything else is folded:
//! every component of the other states F (= Q), and the fast elements within K (K's sidebands -- the hyperfine n_φ and
//! the tweezer n_tw). The folded states' unperturbed generator is block diagonal,
//!   G₀ = H⁽⁰⁾_BB − (i/2) Γ_BB,  Γ = Σ_ch Σ_c L_ch⁽ᶜ⁾† L_ch⁽ᶜ⁾  (the static decay),
//! on the blocks B: Q, and K's states by their fast parts (H⁽⁰⁾ and Γ do not couple two of them: a static element
//! between them is fast and folded). Column a of K at its energy E_a = Re H⁽⁰⁾_aa and the component's frequency,
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
//! - The full reference (`reference`): the scene's components and its Cartesian emission on the 32 states, all kept,
//!   the tweezer's sidebands folded (axes n_tw): the tweezer's light shift on every state and its scattering (its jumps
//!   between the ground states, linear in a_tw(r)) -- the one allowed distinction.
//! - The model (`Rs::at`): the reference folded again on P, every fast element folded (axes n_φ, n_tw, n_ν): Q at every
//!   component, P's hyperfine sidebands. The integrator sees the slow components alone.
//!
//! # At a point
//! Each block's G₀ is decomposed once at the point, G₀ = V diag(μ) V⁻¹ (`linalg::Eigen`), and every resolvent of the
//! point is that one spectrum, (z − G₀)⁻¹ = V diag(1/(z − μ)) V⁻¹; the components carry the beams' local amplitudes
//! (H⁽ᶜ⁾ = Σ_l a_l(r) X_l: the contraction). Nothing else is made at a point. Against td_s.hilbert_rs's column-by-column
//! inverse it is the same formula to the rounding of the decomposition, κ(V) u relative (tests,
//! `the_fold_is_hilbert_rs_to_rounding`).

use std::collections::BTreeMap;

use crate::floquet::{Hilbert, Lift, Part};
use crate::linalg::{self, C64, Mat, V3};
use crate::request::STATES;
use crate::scene::{Components, Jumps, Key, Scene};

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

/// A jump channel: the Cartesian emission it comes of, the fast index its elements share, its components and the same
/// of k∘C (the photon's wave number per element: rad/µm · 1/√µs).
#[derive(Clone, Debug, PartialEq)]
pub struct Channel {
    pub emission: usize,
    pub fast: [i64; 3],
    pub parts: Vec<ModelPart>,
    pub kicks: Vec<ModelPart>,
}

/// A Lindblad model at a point by frequency components: what a fold takes and gives.
#[derive(Clone, Debug, PartialEq)]
pub struct ModelAt {
    /// Its states (indices into the 32), the states the fold that made it folded (Q), and its levels (indices into
    /// `states`).
    pub states: Vec<usize>,
    pub folded: Vec<usize>,
    pub groups: Vec<Vec<usize>>,
    /// H's components, the static one and each other key both ways.
    pub effective: Vec<ModelPart>,
    pub jumps: Vec<Channel>,
    /// M_c on Q (|Q| × n) by components: the folded states' amplitudes.
    pub lift: Vec<ModelPart>,
}

/// The frame's integer fast parts (n_φ, n_tw, n_ν) of each of the 32 states' θ_a and of each key coordinate (the
/// scene's generators), floquet/frame.rs: the fast index of an element b ← a of the key c is Σ_l c_l generators_l +
/// states_b − states_a.
#[derive(Clone, Debug, PartialEq)]
pub struct Sectors {
    pub states: Vec<[i64; 3]>,
    pub generators: Vec<[i64; 3]>,
}

impl Sectors {
    fn index(&self, key: &Key, to: usize, from: usize) -> [i64; 3] {
        std::array::from_fn(|i| key.iter().zip(&self.generators).map(|(&k, g)| i64::from(k) * g[i]).sum::<i64>() + self.states[to][i] - self.states[from][i])
    }
}

/// The fast coordinates the full reference folds: the tweezer's (n_tw).
pub const REFERENCE: [bool; 3] = [false, true, false];
/// The fast coordinates the model folds: every one.
pub const MODEL: [bool; 3] = [true, true, true];

/// The RS fold on a kept set P: P, its complement and the frame's sectors, fixed for the run.
#[derive(Clone, Debug, PartialEq)]
pub struct Rs {
    /// P: indices into the 32, sorted (the ground states first).
    pub keep: Vec<usize>,
    /// Q: the folded states, the rest of the 32.
    pub folded: Vec<usize>,
    pub sectors: Sectors,
}

/// The levels of the states `of` (indices into the 32): one per ground F and one per excited (line, F′), each its members'
/// places in `of`, in the order of their first member (the request's labels "manifold-F-m").
pub fn levels(labels: &[String], of: &[usize]) -> Vec<Vec<usize>> {
    let mut found: Vec<(&str, Vec<usize>)> = Vec::new();
    for (place, &state) in of.iter().enumerate() {
        let level = labels[state].rsplit_once('-').map_or(labels[state].as_str(), |(level, _)| level);
        match found.iter_mut().find(|(name, _)| *name == level) {
            Some((_, members)) => members.push(place),
            None => found.push((level, vec![place])),
        }
    }
    found.into_iter().map(|(_, members)| members).collect()
}

/// The jump channels along the analysis axis u (scene_x `build`, recoil 'emission'): the emission projected on e₁, e₂
/// and u, e₁ ⊥ u from x (or y where u is within acos 0.9 of x), e₂ = u × e₁; each with the photon's mean ⟨(n·u)²⟩ of a
/// dipole along it, 2/5, 2/5, 1/5 -- the weight of its momentum along u. The full reference's channels and the model's.
pub fn channels(axis: V3) -> [(V3, f64); 3] {
    let u = axis.map(|x| x / linalg::norm3(axis));
    let e3 = if u[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let along = linalg::dot3(e3, u);
    let e1 = [e3[0] - along * u[0], e3[1] - along * u[1], e3[2] - along * u[2]];
    let e1 = e1.map(|x| x / linalg::norm3(e1));
    [(e1, 0.4), (linalg::cross3(u, e1), 0.4), (u, 0.2)]
}

/// Sums of model parts by key, in the keys' order; an element added where `mask` holds.
#[derive(Default)]
struct Sums(BTreeMap<Key, ModelPart>);

impl Sums {
    fn add(&mut self, key: &Key, frequency: f64, value: &Mat, gradient: [&Mat; 3], scale: f64, mask: impl Fn(usize, usize) -> bool) {
        let (rows, cols) = (value.rows, value.cols);
        let part = self.0.entry(key.clone()).or_insert_with(|| ModelPart {
            key: key.clone(),
            frequency,
            value: Mat::zeros(rows, cols),
            gradient: std::array::from_fn(|_| Mat::zeros(rows, cols)),
        });
        for b in 0..rows {
            for a in (0..cols).filter(|&a| mask(b, a)) {
                part.value[(b, a)] += value[(b, a)] * scale;
                for (g, d) in part.gradient.iter_mut().zip(gradient) {
                    g[(b, a)] += d[(b, a)] * scale;
                }
            }
        }
    }

    fn parts(self) -> Vec<ModelPart> {
        self.0.into_values().filter(|p| nonzero(&p.value) || p.gradient.iter().any(nonzero)).collect()
    }
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

/// The resolvent of one block of the folded states at the point: its rows (positions in the source), its decomposition,
/// and ∂G₀ on it.
struct Block {
    rows: Vec<usize>,
    eigen: linalg::Eigen,
    changes: [Mat; 3],
}

impl Block {
    /// (E_a + f − G₀)⁻¹ y on each column a of y (the columns of `columns`), y on the block's rows: V diag(1/(z_a − μ)) V⁻¹ y.
    fn resolve(&self, energies: &[f64], frequency: f64, y: &Mat, columns: &[bool]) -> Mat {
        let mut w = self.eigen.inverse.product(y);
        for k in 0..w.rows {
            for a in 0..w.cols {
                w[(k, a)] = if columns[a] { w[(k, a)] / (C64::real(energies[a] + frequency) - self.eigen.values[k]) } else { C64::ZERO };
            }
        }
        self.eigen.vectors.product(&w)
    }
}

/// The one fold (module documentation): of `source`, keep the states `keep` (indices into the 32, among the source's)
/// and their slow elements by `sectors` on the fast coordinates `axes`; fold the rest.
pub fn fold(source: &ModelAt, keep: &[usize], sectors: &Sectors, axes: [bool; 3]) -> ModelAt {
    let place = |state: usize| source.states.iter().position(|&s| s == state).expect("a kept state of the source");
    let mut kept: Vec<usize> = keep.iter().map(|&s| place(s)).collect();
    kept.sort_unstable();
    kept.dedup();
    let size = source.states.len();
    let others: Vec<usize> = (0..size).filter(|p| kept.binary_search(p).is_err()).collect();
    let n = kept.len();
    let state = |p: usize| source.states[p];
    // The fast index of an element (positions in the source) on the folding coordinates.
    let fast = |key: &Key, to: usize, from: usize| -> [i64; 3] {
        let index = sectors.index(key, state(to), state(from));
        std::array::from_fn(|i| if axes[i] { index[i] } else { 0 })
    };
    let still = source.effective.iter().find(|p| p.key.iter().all(|k| *k == 0)).expect("H's static part");
    // G₀ = H⁽⁰⁾ − (i/2)Γ and its gradient, on the source's states.
    let mut g0 = still.value.clone();
    let mut dg0 = still.gradient.clone();
    for channel in &source.jumps {
        for p in &channel.parts {
            g0.add_scaled(C64::new(0.0, -0.5), &p.value.adjoint_product(&p.value));
            for (d, dl) in dg0.iter_mut().zip(&p.gradient) {
                d.add_scaled(C64::new(0.0, -0.5), &dl.adjoint_product(&p.value));
                d.add_scaled(C64::new(0.0, -0.5), &p.value.adjoint_product(dl));
            }
        }
    }
    // The blocks of the folded states: Q, and the kept states by their fast parts.
    let mut groups: BTreeMap<[i64; 3], Vec<usize>> = BTreeMap::new();
    for &p in &kept {
        let s = sectors.states[state(p)];
        groups.entry(std::array::from_fn(|i| if axes[i] { s[i] } else { 0 })).or_default().push(p);
    }
    let blocks: Vec<Block> = std::iter::once(others.clone())
        .filter(|rows| !rows.is_empty())
        .chain(groups.into_values())
        .map(|rows| {
            let mut eigen = linalg::Eigen::default();
            assert!(eigen.decompose(&g0.block(&rows, &rows)), "a folded block's G₀ has no eigendecomposition (defective)");
            let changes = dg0.each_ref().map(|d| d.block(&rows, &rows));
            Block { rows, eigen, changes }
        })
        .collect();
    let energies: Vec<f64> = kept.iter().map(|&a| still.value[(a, a)].re).collect();
    let slopes: [Vec<f64>; 3] = std::array::from_fn(|i| kept.iter().map(|&a| still.gradient[i][(a, a)].re).collect());
    // M_c and ∂M_c (rows: every source position, the kept slow ones zero; columns: K) of every component.
    let mut amplitudes: Vec<(usize, Mat, [Mat; 3])> = Vec::new();
    for (index, c) in source.effective.iter().enumerate() {
        let mut m = Mat::zeros(size, n);
        let mut dm: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(size, n));
        let mut any = false;
        for block in &blocks {
            // The columns whose element into the block is folded: every one into Q, the fast ones into a kept block.
            let columns: Vec<bool> = kept.iter().map(|&a| others.binary_search(&block.rows[0]).is_ok() || fast(&c.key, block.rows[0], a) != [0; 3]).collect();
            let y = c.value.block(&block.rows, &kept);
            let moved = c.gradient.each_ref().map(|d| d.block(&block.rows, &kept));
            if !columns.iter().any(|&x| x) || (!nonzero(&y) && !moved.iter().any(nonzero)) {
                continue;
            }
            any = true;
            let amplitude = block.resolve(&energies, c.frequency, &y, &columns);
            for (i, d) in dm.iter_mut().enumerate() {
                let mut z = moved[i].clone();
                z.add_scaled(C64::ONE, &block.changes[i].product(&amplitude));
                for k in 0..z.rows {
                    for a in 0..n {
                        z[(k, a)] -= amplitude[(k, a)] * slopes[i][a];
                    }
                }
                let change = block.resolve(&energies, c.frequency, &z, &columns);
                for (k, &row) in block.rows.iter().enumerate() {
                    for a in 0..n {
                        d[(row, a)] = change[(k, a)];
                    }
                }
            }
            for (k, &row) in block.rows.iter().enumerate() {
                for a in 0..n {
                    m[(row, a)] = amplitude[(k, a)];
                }
            }
        }
        if any {
            amplitudes.push((index, m, dm));
        }
    }
    // H_eff: the slow kept elements of each component, and each folded pair's slow half with its conjugate.
    let mut effective = Sums::default();
    for c in &source.effective {
        let (block, moved) = (c.value.block(&kept, &kept), c.gradient.each_ref().map(|d| d.block(&kept, &kept)));
        effective.add(&c.key, c.frequency, &block, moved.each_ref(), 1.0, |b, a| fast(&c.key, kept[b], kept[a]) == [0; 3]);
    }
    let all: Vec<usize> = (0..size).collect();
    for c2 in &source.effective {
        let (out, moved) = (c2.value.block(&kept, &all), c2.gradient.each_ref().map(|d| d.block(&kept, &all)));
        if !nonzero(&out) && !moved.iter().any(nonzero) {
            continue;
        }
        for (index, m, dm) in &amplitudes {
            let c = &source.effective[*index];
            let (key, frequency) = (sum_keys(&c.key, &c2.key), c.frequency + c2.frequency);
            let a = out.product(m);
            let da: [Mat; 3] = std::array::from_fn(|i| {
                let mut d = moved[i].product(m);
                d.add_scaled(C64::ONE, &out.product(&dm[i]));
                d
            });
            let slow = |b: usize, x: usize| fast(&key, kept[b], kept[x]) == [0; 3];
            effective.add(&key, frequency, &a, da.each_ref(), 0.5, slow);
            let back = negated(&key);
            effective.add(&back, -frequency, &a.adjoint(), da.each_ref().map(Mat::adjoint).each_ref(), 0.5, |b, x| slow(x, b));
        }
    }
    // The jumps: each element in the channel of its fast index.
    let mut out: BTreeMap<(usize, [i64; 3]), [Sums; 2]> = BTreeMap::new();
    for channel in &source.jumps {
        for (which, parts) in [&channel.parts, &channel.kicks].into_iter().enumerate() {
            for p in parts {
                let mut add = |key: &Key, frequency: f64, value: &Mat, gradient: [&Mat; 3]| {
                    let mut indices: Vec<[i64; 3]> = (0..n).flat_map(|b| (0..n).map(move |a| (b, a))).filter(|&(b, a)| value[(b, a)] != C64::ZERO || gradient.iter().any(|d| d[(b, a)] != C64::ZERO)).map(|(b, a)| fast(key, kept[b], kept[a])).collect();
                    indices.sort_unstable();
                    indices.dedup();
                    for index in indices {
                        let sums = &mut out.entry((channel.emission, index)).or_default()[which];
                        sums.add(key, frequency, value, gradient, 1.0, |b, a| fast(key, kept[b], kept[a]) == index);
                    }
                };
                let (direct, moved) = (p.value.block(&kept, &kept), p.gradient.each_ref().map(|d| d.block(&kept, &kept)));
                add(&p.key, p.frequency, &direct, moved.each_ref());
                let (from, moved) = (p.value.block(&kept, &all), p.gradient.each_ref().map(|d| d.block(&kept, &all)));
                for (index, m, dm) in &amplitudes {
                    let c = &source.effective[*index];
                    let value = from.product(m);
                    let gradient: [Mat; 3] = std::array::from_fn(|i| {
                        let mut d = moved[i].product(m);
                        d.add_scaled(C64::ONE, &from.product(&dm[i]));
                        d
                    });
                    add(&sum_keys(&p.key, &c.key), p.frequency + c.frequency, &value, gradient.each_ref());
                }
            }
        }
    }
    let jumps = out
        .into_iter()
        .map(|((emission, fast), [parts, kicks])| Channel { emission, fast, parts: parts.parts(), kicks: kicks.parts() })
        .filter(|c| !c.parts.is_empty())
        .collect();
    let lift = amplitudes
        .iter()
        .map(|(index, m, dm)| {
            let c = &source.effective[*index];
            ModelPart { key: c.key.clone(), frequency: c.frequency, value: m.block(&others, &all[..n]), gradient: dm.each_ref().map(|d| d.block(&others, &all[..n])) }
        })
        .filter(|p| nonzero(&p.value))
        .collect();
    let groups = source
        .groups
        .iter()
        .map(|members| members.iter().filter_map(|p| kept.binary_search(p).ok()).collect::<Vec<usize>>())
        .filter(|members| !members.is_empty())
        .collect();
    ModelAt {
        states: kept.iter().map(|&p| state(p)).collect(),
        folded: others.iter().map(|&p| state(p)).collect(),
        groups,
        effective: effective.parts(),
        jumps,
        lift,
    }
}

impl ModelAt {
    /// The unfolded model of H's components on the 32 states (each nonzero key both ways) and the Cartesian emission's
    /// three channels with their photons' forms: what the full reference folds.
    pub fn source(components: &Components, jumps: &Jumps, labels: &[String]) -> ModelAt {
        let mut effective = Vec::new();
        for c in &components.parts {
            effective.push(ModelPart { key: c.key.clone(), frequency: c.frequency, value: c.value.clone(), gradient: c.gradient.clone() });
            if c.key.iter().any(|k| *k != 0) {
                effective.push(ModelPart { key: negated(&c.key), frequency: -c.frequency, value: c.value.adjoint(), gradient: c.gradient.each_ref().map(Mat::adjoint) });
            }
        }
        let zero: Key = components.parts.iter().find(|c| c.key.iter().all(|k| *k == 0)).expect("H's static part").key.clone();
        let none = || std::array::from_fn(|_| Mat::zeros(STATES, STATES));
        let part = |value: Mat| ModelPart { key: zero.clone(), frequency: 0.0, value, gradient: none() };
        let photon = |c: &Mat| Mat::from_fn(STATES, STATES, |i, j| c[(i, j)] * jumps.wave_number[(i, j)].re);
        let states: Vec<usize> = (0..STATES).collect();
        ModelAt {
            groups: levels(labels, &states),
            states,
            folded: Vec::new(),
            effective,
            jumps: (0..3).map(|a| Channel { emission: a, fast: [0; 3], parts: vec![part(jumps.emission[a].clone())], kicks: vec![part(photon(&jumps.emission[a]))] }).collect(),
            lift: Vec::new(),
        }
    }
}

/// The full reference at r for the momentum κ: the scene's model on the 32 states with the tweezer's sidebands folded
/// (floquet's `Hilbert::full` is it along an axis).
pub fn reference(scene: &Scene, sectors: &Sectors, r: V3, kappa: V3) -> ModelAt {
    let source = ModelAt::source(&scene.components_at(r, kappa), scene.jumps(), &scene.request.states);
    fold(&source, &(0..STATES).collect::<Vec<_>>(), sectors, REFERENCE)
}

impl Rs {
    /// The model's fold on P = `keep` with the frame's sectors.
    pub fn new(keep: &[usize], sectors: Sectors) -> Rs {
        let mut keep = keep.to_vec();
        keep.sort_unstable();
        keep.dedup();
        let folded = (0..STATES).filter(|i| keep.binary_search(i).is_err()).collect();
        Rs { keep, folded, sectors }
    }

    /// The model at r for the momentum κ: the full reference there, folded on P.
    pub fn at(&self, scene: &Scene, r: V3, kappa: V3) -> ModelAt {
        fold(&reference(scene, &self.sectors, r, kappa), &self.keep, &self.sectors, MODEL)
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

    /// The model as a Hilbert model for the checkpoint, along the analysis axis u: gradients along u; each channel's
    /// emissions projected on e₁, e₂, u (`channels`), with their photons' momentum along u, √w (k∘C).
    pub fn hilbert(&self, axis: V3) -> Hilbert {
        let u = axis.map(|x| x / linalg::norm3(axis));
        let along = |p: &ModelPart| {
            let mut g = Mat::zeros(p.value.rows, p.value.cols);
            for (i, d) in p.gradient.iter().enumerate() {
                g.add_scaled(C64::real(u[i]), d);
            }
            g
        };
        let mut fasts: Vec<[i64; 3]> = self.jumps.iter().map(|c| c.fast).collect();
        fasts.sort_unstable();
        fasts.dedup();
        let (mut jumps, mut kicks) = (Vec::new(), Vec::new());
        for fast in fasts {
            for (e, w) in channels(u) {
                let mut sums = [Sums::default(), Sums::default()];
                for channel in self.jumps.iter().filter(|c| c.fast == fast) {
                    for (which, parts) in [&channel.parts, &channel.kicks].into_iter().enumerate() {
                        let scale = e[channel.emission] * if which == 0 { 1.0 } else { w.sqrt() };
                        for p in parts {
                            let g = along(p);
                            sums[which].add(&p.key, p.frequency, &p.value, [&g, &g, &g], scale, |_, _| true);
                        }
                    }
                }
                let [l, k] = sums.map(|s| s.parts().into_iter().map(|p| Part { key: p.key, value: p.value, gradient: p.gradient[0].clone() }).collect::<Vec<_>>());
                jumps.push(l);
                kicks.push(k);
            }
        }
        Hilbert {
            states: self.states.clone(),
            hamiltonian: self.effective.iter().map(|p| Part { key: p.key.clone(), value: p.value.clone(), gradient: along(p) }).collect(),
            jumps,
            kicks,
            groups: self.groups.clone(),
            lift: Some(Lift { folded: self.folded.clone(), amplitudes: self.lift.iter().map(|p| (p.key.clone(), p.value.clone())).collect() }),
        }
    }
}

#[cfg(test)]
mod tests;
