//! The folds with the bare levels (SPEC §1.2): every frequency's coupling bases, the constant form of its field -- the
//! request's outside form and the second-order shift of its drives folded with the bare levels --, linear in T_ω = E_ω E_ω†
//! and exact at the atom, and those drives' Reiter–Sørensen jumps. No light is of this file or of another: each drive of each
//! component is taken at its own level (model.rs `Level`), and what is here of a frequency is its bare drives' part.
//!
//! # The couplings on the field's basis
//! The beams of frequency ω couple linearly in their fields: X_l = Σ_k γ_lk U_k on the basis u_k of the span of their focus
//! fields (field.rs), U_k = Σ_l γ_lk* X_l/σ_k² (`field::Frequency::dual`); the counter-rotating R_l, antilinear in E_l, are
//! R_l = Σ_k γ_lk* Ũ_k with Ũ_k = Σ_l γ_lk R_l/σ_k². At r the field's components are e_k(r) = Σ_l a_l(r) γ_lk, the
//! co-rotating coupling V(r) = Σ_k e_k U_k and the counter-rotating W(r) = Σ_k e_k* Ũ_k. A request whose couplings leave
//! that span beyond their rounding is not the dipole operator contracted with its fields, and is refused (`Bases::new`).
//!
//! # A fold with the bare levels
//! A drive folded here is eliminated as every drive is (eliminate.rs, # The fold's terms), with the levels of H0: the
//! response of a component's bare drives from ground state b is R[:,b] = (K₀ − E_b − ν)⁻¹ V[:,b], K₀ = H0_ee − (i/2) D -- no
//! light in the denominator, so R is linear in the field and its terms are constants. The other folded drives of the
//! component, those with the levels at the atom, have their response there (eliminate.rs); a fold's terms are bilinear in
//! the coupling and the response, so with V^f every folded drive's coupling and R its bare drives' response the constants
//! below are the whole of the bare drives' part, their cross terms with the others' couplings too. A coupling absorbed at ν_B (B e^{−iν_B t}, B excited ← ground) and one
//! emitted at ν_A (A† e^{iν_A t}) make at second order, with the time dependence e^{−i(ν_B − ν_A)t} (`second_order`),
//!
//! ```text
//! ground   (a ← b):  −½ (A†R_B + R_A†B)_ab            the elimination's contraction of the two legs (eliminate.rs `shift`)
//! excited  (f ← e):  ½ Σ_b B_fb A_eb* [1/(z_e − E_b − ν_A) + 1/(z_f − E_b − ν_B)]
//! ```
//!
//! -- the excited block the van Vleck mean through the ground states, which do not decay (real denominators), each
//! excited state at the energy z it is read at (# A fold onto an explicit state). The ground
//! block holds the width: on a line of detuning Δ it is the dispersive −|V|²Δ/(Δ² + Γ²/4), which the real-denominator form
//! −|V|²/Δ misses by (Γ/2Δ)² of itself; the two legs' responses are the jumps' own (below), so a fold's shift and its
//! scattering are one response's. Both blocks are Hermitian forms: C(A, B)† = C(B, A), at the opposite beat. One component,
//! A = B at ν: its static shift C(V, V). Its error is the fold's: O(m²) of the shift (m its size over its distance,
//! units.rs), and what its bare levels leave out.
//!
//! # A fold onto an explicit state
//! Eliminating the ground state b of a folded drive U (e ← b at ν) gives the excited state the self-energy
//! |U|²/(z − E_b − ν) at the energy z of the state that reads it (the exact second order in U is the reading states'
//! resolvent at b's dressed energy). An excited state that is folded itself is read at its own energy, z = E_e: the
//! distance is the drive's one-photon β = E_e − E_b − ν. An explicit one is read by its cluster: its amplitude follows the
//! cluster's ground states a, which a kept drive at ν_V holds ε = E_e − E_a − ν_V from it, and turns with them,
//! z = E_a + ν_V -- the distance is the two-photon one of the Raman pair (a, b) through e,
//!
//! ```text
//! d_R = E_a + ν_V − E_b − ν = β − ε          z_e = θ_e + ⟨E_a − θ_a⟩_a   (model.rs `Model::read`)
//! ```
//!
//! not β. On three levels (g –V→ e ←U– b, Γ = 1, V = 0.5, ε = 1, U = 0.2; exact: the 3 × 3 spectrum) the kept line's
//! rate with |U|²/β is 14 % low at β = 0.5, 38 % at 0.1, 82 % at 0.02, with a pole at β = 0 where nothing is resonant;
//! with the complex one-photon |U|²/(β − iΓ/2) -- the reading of the transient, z = E_e − iΓ/2, which lives 1/Γ -- 16 to
//! 18 % low; with |U|²/d_R within 0.1 to 3.8 %, the pair's own next order. The pole of 1/d_R is the Raman resonance, where
//! the pair is no fold: the drive's gate is the pair's there, of size |U V| over the nearer one-photon distance across
//! |d_R| (units.rs `Model::pair_gate`), and its error the pair's (eliminate.rs `Model::pair_error`). Where |β| ≫ |ε| the
//! two distances are one.
//!
//! # The static form of a frequency, per pair of beams
//! The outside form O(E) = Σ_pq E_p* E_q O_pq (the request's nine matrices as a Hermitian form in the spherical components:
//! O_qq the |E_q|² matrix, O_pq = (X − iY)/2 and O_qp = (X + iY)/2 from the pair's Re and Im matrices) and the folded
//! components' shifts are Hermitian forms in the field, F(e) = Σ_kk' e_k* e_k' A_kk' with
//! A_kk' = Σ_pq (u_k)_p* (u_k')_q O_pq + C(U_k, U_k') (co, folded) + C(Ũ_k', Ũ_k) (counter, folded). With e_k = Σ_l a_l γ_lk,
//! F = Σ_l'l a_l'* a_l P_l'l, P_l'l = Σ_kk' γ_l'k* γ_lk' A_kk': each beam pair's term is the plane-wave phase e^{i(k_l − k_l')·r}
//! times its envelopes, as every term of the generator (model.rs, # A generator along a step's path). H0 holds the outside
//! form at the focus (request.rs), so a frequency's light adds F(r) − O(E(0)) to it, and turned off −O(E(0)): every light off
//! leaves the bare H0.
//!
//! # The jumps of a folded component (Reiter–Sørensen)
//! The excited amplitude a folded component drives from ground state b is −R[:,b] ψ_b, R the bare response above; its photon
//! of axis a is the jump C_a R (eliminate.rs `jump`; an overall sign is no physics; the near light's photons take the same).
//! Linear in the field: per beam J_{a,l} = Σ_k γ_lk C_a R(U_k), of wave vector k_l (the counter-rotating
//! Σ_k γ_lk* C_a R(Ũ_k), of −k_l). Their decay Σ_a J_a†J_a is formed from them at the assembly: the generator is a Lindblad
//! one by construction, and the far light's scattering, about Γm² per unit time, is not dropped (SPEC §1.2).
//!
//! # Two frequencies' constant form
//! Two frequencies ω and ω' make the same form of their two fields at the beat δ_ω − δ_ω' (`constant`): the coefficient of
//! e^ω_k* e^ω'_k' is the mean of their outside forms' cross term (two frequencies whose beat a cluster resolves differ in
//! their forms, made through the far outside states, by that beat over those states' detunings), the co-rotating
//! C(U^ω_k, U^ω'_k') and the counter-rotating C(Ũ^ω'_k', Ũ^ω_k). A frequency's own (ω' = ω, beat zero) is its static form
//! above. Each entry a ← b of it is the generator's where its residual beat in the frames is exactly zero or its unit is
//! kept (`Unit::Term`: the coupling of a and b at that beat, units.rs); folded, it is dropped, its error the unit's.
//! A co-rotating leg against a counter-rotating one beats at twice an optical frequency and is no term of the model, as
//! the request's outside forms hold none: where both folds exist (units.rs, # The model's domain) its mixing is the
//! product of theirs, a fourth order.

use super::eliminate::{jump, shift, Leg};
use super::{Beat, Component, Level};
use crate::field::{Amplitude, Light};
use crate::linalg::{Lu, Mat, C64};
use crate::request::{Request, GROUND, STATES};
use crate::units::{rounding, RAD_PER_US_PER_MHZ};

/// The excited states.
pub(crate) const EXCITED: usize = STATES - GROUND;

/// The exact frequencies of a request: its frequency axes, then ν_frame (MHz) -- the coordinates of every `Beat`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Axes(pub(crate) Vec<f64>);

impl Axes {
    /// The request's axes and its excited states' frame frequency; refused where the excited states do not share one frame
    /// or a ground state's frame is not zero (the frames H0 is written in, request.rs).
    pub(crate) fn new(request: &Request) -> Result<Axes, String> {
        let frames = &request.frame_frequencies_mhz;
        let frame = frames[GROUND];
        if frames[..GROUND].iter().any(|&f| f != 0.0) || frames[GROUND..].iter().any(|&f| f != frame) {
            return Err("请求不符合求解器的接口：基态的旋转系频率（frameFrequenciesMHz）应为 0，激发态应共用一个旋转系频率".into());
        }
        let mut axes = request.frequencies_mhz.clone();
        axes.push(frame);
        Ok(Axes(axes))
    }
    /// The zero beat.
    pub(crate) fn zero(&self) -> Beat {
        Beat(vec![0; self.0.len()])
    }
    /// A beat's value (rad/µs): 2π Σ_k n_k f_k, from its integer coordinates alone (never a difference of two values).
    pub(crate) fn omega(&self, beat: &Beat) -> f64 {
        RAD_PER_US_PER_MHZ * beat.0.iter().zip(&self.0).map(|(&n, &f)| n as f64 * f).sum::<f64>()
    }
    /// A component's frequency ν exactly: the co-rotating δ_ω (the frequency's index), the counter-rotating −(δ_ω + 2ν_frame).
    pub(crate) fn component(&self, light: &Light, c: Component) -> Beat {
        let mut n: Vec<i64> = light.frequencies[c.frequency].index.iter().map(|&x| x as i64).collect();
        n.resize(self.0.len(), 0);
        if c.counter {
            let last = n.len() - 1;
            n[last] = 2;
            n.iter_mut().for_each(|x| *x = -*x);
        }
        Beat(n)
    }
}

impl Beat {
    /// self + s·other.
    pub(crate) fn plus(&self, s: i64, other: &Beat) -> Beat {
        Beat(self.0.iter().zip(&other.0).map(|(&a, &b)| a + s * b).collect())
    }
    pub(crate) fn negated(&self) -> Beat {
        Beat(self.0.iter().map(|&a| -a).collect())
    }
    pub(crate) fn is_zero(&self) -> bool {
        self.0.iter().all(|&a| a == 0)
    }
}

/// One leg of a second-order form (`second_order`): a component's couplings on one basis vector -- those of its folded
/// drives, whatever their level (`coupled`: what a response contracts with), those of its drives folded with the bare levels
/// alone (`bare`) --, and the bare response of the latter (`response`, EXCITED × GROUND).
#[derive(Clone, Copy)]
pub(crate) struct Fold<'a> {
    pub(crate) coupled: &'a Mat,
    pub(crate) bare: &'a Mat,
    pub(crate) response: &'a Mat,
}

/// The second-order form C(A, B) of an emitted leg A at ν_A and an absorbed one B at ν_B (module documentation, # A fold with
/// the bare levels), on all the states (STATES × STATES): the ground block the contraction of the two legs' folded couplings
/// with their bare responses (eliminate.rs `shift` -- the responses at the atom of the drives folded there contract with the
/// same couplings, assemble.rs); the excited block through the ground states with the bare energies `e`, each bare drive's
/// own denominator against the other leg's folded coupling (the drives folded at the atom bring theirs there, assemble.rs),
/// its excited state at the energy it is read at (`read`, per excited state less GROUND; module documentation, # A fold
/// onto an explicit state).
pub(crate) fn second_order(e: &[f64], read: &[f64], a: Fold, nu_a: f64, b: Fold, nu_b: f64) -> Mat {
    let mut out = Mat::zeros(STATES, STATES);
    for x in 0..GROUND {
        for y in 0..GROUND {
            out[(x, y)] = shift(Leg { coupling: a.coupled, state: x, response: a.response, place: x }, Leg { coupling: b.coupled, state: y, response: b.response, place: y });
        }
    }
    for f in GROUND..STATES {
        for g in GROUND..STATES {
            let mut z = C64::ZERO;
            for w in 0..GROUND {
                // Each half holds one drive's own denominator: the emitted leg's g ← w, the absorbed leg's f ← w. An
                // explicit state, read at its cluster's ground energy (not its own), is never read at a Raman pair's
                // two-photon distance of nothing: such a pair is joined (model.rs `Model::settled`, units.rs `Model::joined`).
                let distance = |x: usize, nu: f64| -> f64 {
                    let d = read[x - GROUND] - e[w] - nu;
                    assert!(d != 0.0 || read[x - GROUND] == e[x], "a fold onto an explicit state at a two-photon distance of nothing (state {x}, ground {w}): the pair is joined, never folded");
                    d
                };
                let emitted = b.coupled[(f, w)] * a.bare[(g, w)].conj();
                if emitted != C64::ZERO {
                    z += emitted * (0.5 / distance(g, nu_a));
                }
                let absorbed = b.bare[(f, w)] * a.coupled[(g, w)].conj();
                if absorbed != C64::ZERO {
                    z += absorbed * (0.5 / distance(f, nu_b));
                }
            }
            out[(f, g)] = z;
        }
    }
    out
}

/// A frequency's outside form as a Hermitian form in the spherical components, O_pq (module documentation).
pub(crate) fn outside_form(m: &[Mat; 9]) -> [[Mat; 3]; 3] {
    // The request's order: |E_q|² for q = −1, 0, +1, then per pair (−1, 0), (−1, +1), (0, +1) its Re and Im matrices.
    std::array::from_fn(|p| {
        std::array::from_fn(|q| {
            if p == q {
                return m[q].clone();
            }
            let at = 3 + 2 * (p + q - 1);
            let mut out = m[at].scaled(C64::real(0.5));
            out.add_scaled(C64::new(0.0, if p < q { -0.5 } else { 0.5 }), &m[at + 1]);
            out
        })
    })
}

/// A frequency's couplings on its field's basis: co-rotating U_k and counter-rotating Ũ_k (STATES × STATES, excited ← ground).
#[derive(Clone, Debug)]
pub(crate) struct Bases {
    pub(crate) co: Vec<Mat>,
    pub(crate) counter: Vec<Mat>,
}

impl Bases {
    /// U_k = Σ_l γ_lk* X_l/σ_k², Ũ_k = Σ_l γ_lk R_l/σ_k². Refused where a beam's coupling departs from Σ_k γ_lk U_k (the
    /// counter-rotating from Σ_k γ_lk* Ũ_k) by more than the rounding a consistent request leaves: the request's own contraction
    /// with its field (three terms an entry, γ_3) and this projection (2rn + 3 operations), amplified by at most the condition
    /// σ_max/σ_min of the beams' fields on the basis -- γ_{2rn+3}·cond·(‖X_l‖ + Σ_k |γ_lk|‖U_k‖) in Frobenius norm.
    pub(crate) fn new(request: &Request, light: &Light, frequency: usize) -> Result<Bases, String> {
        let f = &light.frequencies[frequency];
        let (n, r) = (f.beams.len(), f.rank());
        let combine = |counter: bool| -> Vec<Mat> {
            (0..r)
                .map(|k| {
                    let mut m = Mat::zeros(STATES, STATES);
                    for j in 0..n {
                        let beam = &request.beams[f.beams[j]];
                        if counter {
                            m.add_scaled(f.dual(j, k).conj(), &beam.counter_coupling);
                        } else {
                            m.add_scaled(f.dual(j, k), &beam.coupling);
                        }
                    }
                    m
                })
                .collect()
        };
        let (co, counter) = (combine(false), combine(true));
        let (most, least) = f.gram.iter().fold((0.0_f64, f64::INFINITY), |(a, b), &g| (a.max(g), b.min(g)));
        let allowed = rounding(2 * r * n + 3) * (most / least).sqrt();
        for j in 0..n {
            let beam = &request.beams[f.beams[j]];
            for (given, basis, conjugate) in [(&beam.coupling, &co, false), (&beam.counter_coupling, &counter, true)] {
                let mut residual = given.clone();
                let mut scale = given.frobenius();
                for k in 0..r {
                    let g = if conjugate { f.coefficients[j][k].conj() } else { f.coefficients[j][k] };
                    residual.add_scaled(-g, &basis[k]);
                    scale += g.abs() * basis[k].frobenius();
                }
                if residual.frobenius() > allowed * scale {
                    return Err(format!(
                        "{}：光束的耦合与其电场不一致（偏离 {:.1e}）",
                        beam.name,
                        residual.frobenius() / scale.max(f64::MIN_POSITIVE)
                    ));
                }
            }
        }
        Ok(Bases { co, counter })
    }
}

/// The light of a frequency as the folds with the bare levels take it (module documentation): its exact co-rotating
/// frequency δ_ω, its coupling bases, its static form on its field's basis and per beam pair, the outside form at the focus,
/// and its bare drives' responses and jumps on the basis -- each drive at its level (`Tone::fold`).
#[derive(Clone, Debug)]
pub(crate) struct Tone {
    /// δ_ω exactly (its index; no frame coordinate) and its value (rad/µs).
    pub(crate) beat: Beat,
    pub(crate) detuning: f64,
    /// [co-rotating, counter-rotating]: whether any of its drives is folded with the bare levels.
    pub(crate) bare: [bool; 2],
    /// A_k'k (STATES × STATES) at `forms[k' r + k]` on the field's basis: the static form F(e) = Σ_k'k e_k'* e_k A_k'k.
    pub(crate) forms: Vec<Mat>,
    /// P_l'l (STATES × STATES) at `pairs[l' n + l]`, l' and l in the frequency's beam order.
    pub(crate) pairs: Vec<Mat>,
    /// O(E(0)): the outside form at the focus, which H0 holds.
    pub(crate) focus: Mat,
    /// Per component ([co, counter]) and basis vector k its bare drives' jumps (GROUND × GROUND per axis a): the field's jump is
    /// Σ_k e_k J_{a,k} (the counter-rotating Σ_k e_k* J_{a,k}); empty where none is bare.
    pub(crate) jumps: [Vec<[Mat; 3]>; 2],
    /// Per component ([co, counter]) and beam its bare drives' response R_l (EXCITED × GROUND, the readout's).
    pub(crate) responses: [Vec<Mat>; 2],
    /// Per component ([co, counter]) and beam its coupling at the focus, X_l and R_l (STATES × STATES).
    pub(crate) couplings: [Vec<Mat>; 2],
    /// Per component ([co, counter]) its couplings on the field's basis, U_k and Ũ_k (STATES × STATES); those of its folded
    /// drives, whatever their level (the explicit drives' entries zero: what a fold's responses contract with); those of its
    /// bare drives alone; and the bare response of the latter (EXCITED × GROUND).
    pub(crate) basis: [Vec<Mat>; 2],
    pub(crate) coupled: [Vec<Mat>; 2],
    pub(crate) bare_basis: [Vec<Mat>; 2],
    pub(crate) bare_response: [Vec<Mat>; 2],
    /// The outside form O_pq in the spherical components (two frequencies' cross terms, units.rs), and on the field's basis
    /// (the static form without the folds).
    pub(crate) outside: [[Mat; 3]; 3],
    plain: Vec<Mat>,
    /// The beams' coefficients on the basis γ_lk, the field's basis vectors, [δ_ω, −(δ_ω + 2ν_frame)] (rad/µs), K₀ = H0_ee −
    /// (i/2)D, the bare energies and the emission operators: what a fold reads again.
    pub(crate) coefficients: Vec<Vec<C64>>,
    pub(crate) vectors: Vec<[C64; 3]>,
    pub(crate) nu: [f64; 2],
    k0: Mat,
    energies: Vec<f64>,
    emission: [Mat; 3],
    /// Per excited state (less GROUND) the energy its folds' excited block is read at (model.rs `Model::read_made`).
    read: Vec<f64>,
}

/// The bare energies E_s: H0's diagonal (H0 is diagonal in the model basis, request.rs).
pub(crate) fn energies(request: &Request) -> Vec<f64> {
    (0..STATES).map(|s| request.hamiltonian[(s, s)].re).collect()
}

/// D = Σ_a C_a†C_a on the excited states (EXCITED × EXCITED).
pub(crate) fn decay(request: &Request) -> Mat {
    let (ground, excited): (Vec<usize>, Vec<usize>) = ((0..GROUND).collect(), (GROUND..STATES).collect());
    let mut d = Mat::zeros(EXCITED, EXCITED);
    for c in &request.emission {
        let ce = c.block(&ground, &excited);
        d.add_scaled(C64::ONE, &ce.adjoint_product(&ce));
    }
    d
}

impl Tone {
    /// A frequency's light with nothing folded yet: what does not depend on its drives' levels (`Tone::folded` makes the rest).
    pub(crate) fn new(request: &Request, light: &Light, axes: &Axes, frequency: usize) -> Result<Tone, String> {
        let f = &light.frequencies[frequency];
        let bases = Bases::new(request, light, frequency)?;
        let (n, r) = (f.beams.len(), f.rank());
        let beat = axes.component(light, Component { frequency, counter: false });
        let nu = [axes.omega(&beat), axes.omega(&axes.component(light, Component { frequency, counter: true }))];
        let o = outside_form(&request.outside_forms[f.form].matrices);
        // The outside form on the basis (row-major r × r), and its value at the focus.
        let plain: Vec<Mat> = (0..r * r)
            .map(|x| {
                let (k, kk) = (x / r, x % r);
                let mut m = Mat::zeros(STATES, STATES);
                for p in 0..3 {
                    for q in 0..3 {
                        let w = f.basis[k][p].conj() * f.basis[kk][q];
                        if w != C64::ZERO {
                            m.add_scaled(w, &o[p][q]);
                        }
                    }
                }
                m
            })
            .collect();
        let e0: Vec<C64> = (0..r).map(|k| (0..n).map(|j| f.coefficients[j][k]).sum()).collect();
        let mut focus = Mat::zeros(STATES, STATES);
        for k in 0..r {
            for kk in 0..r {
                focus.add_scaled(e0[k].conj() * e0[kk], &plain[k * r + kk]);
            }
        }
        let k0 = {
            let excited: Vec<usize> = (GROUND..STATES).collect();
            let mut k0 = request.hamiltonian.block(&excited, &excited);
            k0.add_scaled(C64::new(0.0, -0.5), &decay(request));
            k0
        };
        let basis = [bases.co, bases.counter];
        let mut tone = Tone {
            beat,
            detuning: nu[0],
            bare: [false; 2],
            forms: vec![],
            pairs: vec![],
            focus,
            jumps: [vec![], vec![]],
            responses: [vec![], vec![]],
            couplings: [vec![], vec![]],
            basis,
            coupled: [vec![], vec![]],
            bare_basis: [vec![], vec![]],
            bare_response: [vec![], vec![]],
            outside: o,
            plain,
            coefficients: f.coefficients.clone(),
            vectors: f.basis.clone(),
            nu,
            k0,
            energies: energies(request),
            emission: request.emission.clone(),
            read: energies(request)[GROUND..].to_vec(),
        };
        tone.couplings = [0, 1].map(|c| tone.per_beam(c, &tone.basis[c]));
        Ok(tone)
    }

    /// The same light with each drive (component [co, counter], excited, ground) at its level (model.rs `Level`), each
    /// excited state read at `read` (model.rs `Model::read_made`).
    pub(crate) fn folded(&self, level: &dyn Fn(usize, usize, usize) -> Level, read: &[f64]) -> Tone {
        let mut tone = self.clone();
        tone.read = read.to_vec();
        tone.fold(level);
        tone
    }

    /// Per beam Σ_k γ_lk M_k of a co-rotating component's matrices on the basis (the counter-rotating Σ_k γ_lk* M_k).
    fn per_beam(&self, side: usize, of: &[Mat]) -> Vec<Mat> {
        self.coefficients
            .iter()
            .map(|gamma| {
                let mut m = Mat::zeros(of[0].rows, of[0].cols);
                for (g, x) in gamma.iter().zip(of) {
                    m.add_scaled(if side == 0 { *g } else { g.conj() }, x);
                }
                m
            })
            .collect()
    }

    /// Whether a component of it ([co, counter]) couples ground state g to excited state e: some beam's coupling is not zero.
    pub(crate) fn drives(&self, counter: bool, e: usize, g: usize) -> bool {
        self.couplings[counter as usize].iter().any(|x| x[(e, g)] != C64::ZERO)
    }

    /// One leg of its second-order forms (`Fold`): a component's basis vector.
    pub(crate) fn leg(&self, side: usize, k: usize) -> Fold<'_> {
        Fold { coupled: &self.coupled[side][k], bare: &self.bare_basis[side][k], response: &self.bare_response[side][k] }
    }

    /// Takes each drive at its level: the bare drives' responses, jumps and excited block, and the static form of every
    /// folded drive's coupling with them (`second_order`).
    fn fold(&mut self, level: &dyn Fn(usize, usize, usize) -> Level) {
        let r = self.basis[0].len();
        let only =|side: usize, m: &Mat, keep: &dyn Fn(Level) -> bool| Mat::from_fn(STATES, STATES, |e, g| if e >= GROUND && g < GROUND && keep(level(side, e, g)) { m[(e, g)] } else { C64::ZERO });
        self.coupled = [0, 1].map(|c| self.basis[c].iter().map(|u| only(c, u, &|l| l != Level::Explicit)).collect());
        self.bare_basis = [0, 1].map(|c| self.basis[c].iter().map(|u| only(c, u, &|l| l == Level::Bare)).collect());
        self.bare = [0, 1].map(|c| self.bare_basis[c].iter().any(|u| u.data.iter().any(|z| *z != C64::ZERO)));
        self.bare_response = [0, 1].map(|c| self.bare_basis[c].iter().map(|u| response(&self.k0, &self.energies, u, self.nu[c])).collect());
        // A_kk': the outside form, the co-rotating C(U_k, U_k') and the counter-rotating C(Ũ_k', Ũ_k).
        let mut a = self.plain.clone();
        for k in 0..r {
            for kk in 0..r {
                a[k * r + kk].add_scaled(C64::ONE, &second_order(&self.energies, &self.read, self.leg(0, k), self.nu[0], self.leg(0, kk), self.nu[0]));
                a[k * r + kk].add_scaled(C64::ONE, &second_order(&self.energies, &self.read, self.leg(1, kk), self.nu[1], self.leg(1, k), self.nu[1]));
            }
        }
        self.pairs = per_pair(self, self, &a);
        self.forms = a;
        self.responses = [0, 1].map(|c| self.per_beam(c, &self.bare_response[c]));
        self.jumps = [0, 1].map(|c| {
            let of = |response: &Mat| -> [Mat; 3] { std::array::from_fn(|a| Mat::from_fn(GROUND, GROUND, |to, b| jump(&self.emission[a], to, response, b))) };
            if self.bare[c] { self.bare_response[c].iter().map(of).collect() } else { vec![] }
        });
    }
}

/// The constant form of two frequencies on their fields' bases (module documentation, # Two frequencies' constant form):
/// C_{k k'} at `[k r' + k']`, the coefficient of e^to_k* e^from_k' at the beat δ_to − δ_from -- the mean of their outside
/// forms' cross term, the co-rotating C(U^to_k, U^from_k') (emitting `to`'s, absorbing `from`'s) and the counter-rotating
/// C(Ũ^from_k', Ũ^to_k) (emitting `from`'s, absorbing `to`'s: the same product of fields at the same beat). A frequency's
/// own is its static form (`Tone::forms`).
pub(crate) fn constant(to: &Tone, from: &Tone) -> Vec<Mat> {
    let (r, rr) = (to.basis[0].len(), from.basis[0].len());
    (0..r * rr)
        .map(|x| {
            let (k, kk) = (x / rr, x % rr);
            let mut m = Mat::zeros(STATES, STATES);
            for p in 0..3 {
                for q in 0..3 {
                    let w = to.vectors[k][p].conj() * from.vectors[kk][q] * 0.5;
                    if w != C64::ZERO {
                        m.add_scaled(w, &to.outside[p][q]);
                        m.add_scaled(w, &from.outside[p][q]);
                    }
                }
            }
            m.add_scaled(C64::ONE, &second_order(&to.energies, &to.read, to.leg(0, k), to.nu[0], from.leg(0, kk), from.nu[0]));
            m.add_scaled(C64::ONE, &second_order(&to.energies, &to.read, from.leg(1, kk), from.nu[1], to.leg(1, k), to.nu[1]));
            m
        })
        .collect()
}

/// A form on two frequencies' bases (`constant`) per pair of their beams: Σ_kk' γ^to_jk* γ^from_ik' C_kk' at `[j n' + i]`, the
/// coefficient of a_j* a_i (j of `to`'s beams, i of `from`'s).
pub(crate) fn per_pair(to: &Tone, from: &Tone, forms: &[Mat]) -> Vec<Mat> {
    let (n, nn, rr) = (to.coefficients.len(), from.coefficients.len(), from.basis[0].len());
    (0..n * nn)
        .map(|x| {
            let (j, i) = (x / nn, x % nn);
            let mut m = Mat::zeros(STATES, STATES);
            for (k, c) in to.coefficients[j].iter().enumerate() {
                for (kk, d) in from.coefficients[i].iter().enumerate() {
                    m.add_scaled(c.conj() * *d, &forms[k * rr + kk]);
                }
            }
            m
        })
        .collect()
}

/// The bare response of a coupling U at ν (module documentation, # A fold with the bare levels): R[:,b] = (K₀ − E_b − ν)⁻¹
/// U[:,b] on the excited states (EXCITED × GROUND).
pub(crate) fn response(k0: &Mat, e: &[f64], u: &Mat, nu: f64) -> Mat {
    let mut response = Mat::zeros(EXCITED, GROUND);
    for b in 0..GROUND {
        let mut column: Vec<C64> = (0..EXCITED).map(|x| u[(GROUND + x, b)]).collect();
        if column.iter().all(|z| *z == C64::ZERO) {
            continue;
        }
        let mut shifted = k0.clone();
        for x in 0..EXCITED {
            shifted[(x, x)] -= C64::real(e[b] + nu);
        }
        // K₀ − z is regular wherever an excited state decays: its anti-Hermitian part −D/2 keeps every eigenvalue off the real axis.
        let lu = Lu::new(&shifted).expect("K₀ − z regular");
        lu.solve(&mut column);
        for x in 0..EXCITED {
            response[(x, b)] = column[x];
        }
    }
    response
}

/// The product a_l'* a_l of two beams' amplitudes (`first` = l', `second` = l) with its gradient and Hessian (jet.rs
/// `multiply`): the coefficient of a beam pair's term, of wave vector k_l − k_l'.
pub(crate) fn pair(first: &Amplitude, second: &Amplitude) -> Amplitude {
    let mut out = [C64::ZERO; super::jet::WHOLE];
    super::jet::multiply(&super::jet::whole(first).map(|z| z.conj()), &super::jet::whole(second), &mut out);
    super::jet::amplitude(&out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::hermitian_eigen;

    /// A deterministic pseudo-random coupling (excited ← ground) of entries of modulus up to `scale`.
    fn coupling(scale: f64, seed: u64) -> Mat {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        Mat::from_fn(STATES, STATES, |e, b| if e >= GROUND && b < GROUND { C64::new(next(), next()) * scale } else { C64::ZERO })
    }

    /// A bare fold's ground block is the second order of the exact Floquet spectrum. One component V e^{−iνt} + h.c. couples the
    /// ground states at E_b to the excited states at E_e − ν (the rotating frame of the absorbed photon): H_F = [[E_g, V†],
    /// [V, E_e − ν]], exactly Hermitian -- levels that do not decay, whose bare response has real denominators. Its ground-like
    /// eigenvalues are those of E_g + C(V, V)_gg up to fourth order in V/Δ: so halving V cuts the difference sixteenfold,
    /// whatever the ground states' degeneracy (the block is a matrix there; its eigenvalues are compared). Bare energies with
    /// the hyperfine and the line spacings of Rb87 at the scale of the frame (rad/µs), ν a far detuning.
    #[test]
    fn a_bare_fold_s_shift_is_the_floquet_spectrum_s_second_order() {
        let e: Vec<f64> = (0..STATES)
            .map(|s| if s < 3 { -42943.0 + 13.0 * (s as f64 - 1.0) } else if s < GROUND { (s as f64 - 5.0) * 13.2 } else if s < 16 { -4.6e6 + 300.0 * s as f64 } else { 1.0e3 * (s as f64 - 24.0) })
            .collect();
        let nu = -2.5e6;
        // Levels that do not decay: K₀ is their energies.
        let k0 = Mat::from_fn(EXCITED, EXCITED, |x, y| if x == y { C64::real(e[GROUND + x]) } else { C64::ZERO });
        let error = |scale: f64| {
            let v = coupling(scale, 7);
            let r = response(&k0, &e, &v, nu);
            let leg = Fold { coupled: &v, bare: &v, response: &r };
            let f = second_order(&e, &e[GROUND..], leg, nu, leg, nu);
            let floquet = Mat::from_fn(STATES, STATES, |x, y| {
                let d = if x == y { C64::real(if x < GROUND { e[x] } else { e[x] - nu }) } else { C64::ZERO };
                d + if x >= GROUND && y < GROUND { v[(x, y)] } else if x < GROUND && y >= GROUND { v[(y, x)].conj() } else { C64::ZERO }
            });
            let exact = hermitian_eigen(&floquet).values;
            let ground: Vec<usize> = (0..GROUND).collect();
            let mut effective = f.block(&ground, &ground);
            for b in 0..GROUND {
                effective[(b, b)] += C64::real(e[b]);
            }
            let second = hermitian_eigen(&effective).values;
            // The ground-like eigenvalues are the eight nearest the ground energies, the excited ones E_e − ν lying ±2e6 away.
            let mut near: Vec<f64> = exact.clone();
            near.sort_by(|a, b| {
                let d = |x: f64| e[..GROUND].iter().map(|g| (x - g).abs()).fold(f64::INFINITY, f64::min);
                d(*a).total_cmp(&d(*b))
            });
            let mut lowest = near[..GROUND].to_vec();
            lowest.sort_by(|a, b| a.total_cmp(b));
            let shift = f.block(&ground, &ground).max_abs();
            (lowest.iter().zip(&second).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max), shift)
        };
        let (big, shift) = error(3000.0);
        let (small, _) = error(1500.0);
        assert!(shift > 1.0, "the coupling shifts the ground states visibly: {shift}");
        assert!(big < 1e-3 * shift, "fourth order: {big} of a shift {shift}");
        let ratio = big / small;
        assert!((ratio - 16.0).abs() < 1.0, "the remainder is fourth order: ratio {ratio} ({big}, {small})");
    }

    /// A bare fold's ground block holds its lines' widths. With excited states that decay each at its own rate, K₀ =
    /// diag(E_e − iΓ_e/2), the bare response is R[e,b] = V[e,b]/(E_e − E_b − ν − iΓ_e/2) and the contraction of two legs is
    /// −½ Σ_e A_ea* B_eb [1/(E_e − E_b − ν_B − iΓ_e/2) + 1/(E_e − E_a − ν_A + iΓ_e/2)], entry by entry to the rounding of its
    /// sum of two products over the excited states (γ of the entry's terms' moduli) -- on the diagonal of one component the
    /// dispersive −Σ_e |V_eb|² Δ_e/(Δ_e² + Γ_e²/4), which the real-denominator −Σ_e |V_eb|²/Δ_e exceeds by (Γ_e/2Δ_e)² of each
    /// term. Two near detunings (lines a few widths off) and two components' beat.
    #[test]
    fn a_bare_fold_s_ground_block_holds_its_lines_widths() {
        let e: Vec<f64> = (0..STATES).map(|s| if s < GROUND { 3.0 * s as f64 } else { 200.0 + 17.0 * s as f64 }).collect();
        let gamma: Vec<f64> = (0..EXCITED).map(|x| 30.0 + x as f64).collect();
        let k0 = Mat::from_fn(EXCITED, EXCITED, |x, y| if x == y { C64::new(e[GROUND + x], -0.5 * gamma[x]) } else { C64::ZERO });
        let (a, b) = (coupling(5.0, 11), coupling(7.0, 12));
        let (nu_a, nu_b) = (310.0, 345.0);
        let (ra, rb) = (response(&k0, &e, &a, nu_a), response(&k0, &e, &b, nu_b));
        let (la, lb) = (Fold { coupled: &a, bare: &a, response: &ra }, Fold { coupled: &b, bare: &b, response: &rb });
        let form = second_order(&e, &e[GROUND..], la, nu_a, lb, nu_b);
        for x in 0..GROUND {
            for y in 0..GROUND {
                let (mut closed, mut size) = (C64::ZERO, 0.0);
                for w in 0..EXCITED {
                    let (absorbed, emitted) = (C64::new(e[GROUND + w] - e[y] - nu_b, -0.5 * gamma[w]), C64::new(e[GROUND + w] - e[x] - nu_a, 0.5 * gamma[w]));
                    let term = a[(GROUND + w, x)].conj() * b[(GROUND + w, y)];
                    closed += term * (absorbed.recip() + emitted.recip()) * -0.5;
                    size += term.abs() * (absorbed.recip().abs() + emitted.recip().abs()) * 0.5;
                }
                assert!((form[(x, y)] - closed).abs() <= rounding(2 * EXCITED + 6) * size, "({x}, {y}): {:?} against {closed:?}", form[(x, y)]);
            }
        }
        let own = second_order(&e, &e[GROUND..], lb, nu_b, lb, nu_b);
        let (mut dispersive, mut real) = (0.0, 0.0);
        for w in 0..EXCITED {
            let (delta, v2) = (e[GROUND + w] - e[0] - nu_b, b[(GROUND + w, 0)].norm_sqr());
            dispersive -= v2 * delta / (delta * delta + 0.25 * gamma[w] * gamma[w]);
            real -= v2 / delta;
        }
        assert!((own[(0, 0)].re - dispersive).abs() <= rounding(2 * EXCITED + 6) * dispersive.abs().max(real.abs()) && own[(0, 0)].im.abs() <= rounding(2 * EXCITED + 6) * real.abs(), "{:?} against {dispersive}", own[(0, 0)]);
        assert!((real - dispersive).abs() > 1e-3 * real.abs(), "lines a few widths off: the width shows ({real} against {dispersive})");
    }

    /// The outside form's Hermitian form in the spherical components reproduces the request's coordinates: for a field E,
    /// Σ_pq E_p* E_q O_pq = Σ_i x_i(E) O_i with x = (|E_q|², then per pair (−1, 0), (−1, +1), (0, +1) Re and Im of E_p* E_q).
    #[test]
    fn the_outside_form_is_its_coordinates_linear_form() {
        let m: [Mat; 9] = std::array::from_fn(|i| {
            let a = coupling(1.0, 100 + i as u64);
            let mut h = a.clone();
            h.add_scaled(C64::ONE, &a.adjoint());
            h
        });
        let field = [C64::new(0.3, -1.2), C64::new(-0.7, 0.4), C64::new(1.1, 0.25)];
        let o = outside_form(&m);
        let mut by_form = Mat::zeros(STATES, STATES);
        for p in 0..3 {
            for q in 0..3 {
                by_form.add_scaled(field[p].conj() * field[q], &o[p][q]);
            }
        }
        let z = |p: usize, q: usize| field[p].conj() * field[q];
        let x = [z(0, 0).re, z(1, 1).re, z(2, 2).re, z(0, 1).re, z(0, 1).im, z(0, 2).re, z(0, 2).im, z(1, 2).re, z(1, 2).im];
        let mut by_coordinates = Mat::zeros(STATES, STATES);
        for i in 0..9 {
            by_coordinates.add_scaled(C64::real(x[i]), &m[i]);
        }
        let mut d = by_form.clone();
        d.add_scaled(-C64::ONE, &by_coordinates);
        assert!(d.max_abs() <= 1e-14 * by_coordinates.max_abs(), "{}", d.max_abs());
    }
}
