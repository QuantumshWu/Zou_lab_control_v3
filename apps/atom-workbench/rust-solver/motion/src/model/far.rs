//! The far light (SPEC §1.2): every frequency's coupling bases, the static shift of its field -- the request's outside form and
//! the van Vleck shift of its folded components --, linear in T_ω = E_ω E_ω† and exact at the atom, and the Reiter–Sørensen
//! jumps of its folded components.
//!
//! # The couplings on the field's basis
//! The beams of frequency ω couple linearly in their fields: X_l = Σ_k γ_lk U_k on the basis u_k of the span of their focus
//! fields (field.rs), U_k = Σ_l γ_lk* X_l/σ_k² (`field::Frequency::dual`); the counter-rotating R_l, antilinear in E_l, are
//! R_l = Σ_k γ_lk* Ũ_k with Ũ_k = Σ_l γ_lk R_l/σ_k². At r the field's components are e_k(r) = Σ_l a_l(r) γ_lk, the
//! co-rotating coupling V(r) = Σ_k e_k U_k and the counter-rotating W(r) = Σ_k e_k* Ũ_k. A request whose couplings leave
//! that span beyond their rounding is not the dipole operator contracted with its fields, and is refused (`Bases::new`).
//!
//! # The van Vleck form (second order, exact real denominators)
//! A coupling absorbed at ν_B (B e^{−iν_B t}, B excited ← ground) and one emitted at ν_A (A† e^{iν_A t}) make at second order,
//! with the time dependence e^{−i(ν_B − ν_A)t},
//!
//! ```text
//! ground   (a ← b):  C(A, B)_ab = ½ Σ_e A_ea* B_eb [1/(E_b + ν_B − E_e) + 1/(E_a + ν_A − E_e)]
//! excited  (f ← e):  C(A, B)_fe = ½ Σ_b B_fb A_eb* [1/(E_e − E_b − ν_A) + 1/(E_f − E_b − ν_B)]
//! ```
//!
//! -- the symmetric (van Vleck) mean of the initial and final denominators, the bare energies E of H0 exact: a far component
//! folds with no gap between what it keeps and what it folds. C(A, B)† = C(B, A), at the opposite beat. One component, A = B
//! at ν: its static shift C(V, V), Hermitian. Its error is the fold's, O(m²) of the shift (m its size over its distance,
//! units.rs); its real denominators hold no scattering, which its jumps carry (below).
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
//! The excited amplitude a folded component drives from ground state b is −R[:,b] ψ_b, R[:,b] = (K₀ − E_b − ν)⁻¹ V[:,b],
//! K₀ = H0_ee − (i/2) D; its photon of axis a is the jump C_a R (an overall sign is no physics; the near light's photons take the
//! same, eliminate.rs). Linear in the field: per beam J_{a,l} = Σ_k γ_lk C_a R(U_k), of wave vector k_l (the counter-rotating
//! Σ_k γ_lk* C_a R(Ũ_k), of −k_l). Their decay Σ_a J_a†J_a is formed from them at the assembly: the generator is a Lindblad
//! one by construction, and the far light's scattering, about Γm² per unit time, is not dropped (SPEC §1.2).
//!
//! # The beat of two far components
//! Two folded components at ν₁ and ν₂ also make the cross terms C(V₂, V₁) at e^{−i(ν₁ − ν₂)t} and C(V₁, V₂) at the opposite
//! beat, with the cross terms of their outside forms. Folded (`Unit::FarBeat`) they are dropped; kept they are a beat of the
//! generator, the cross outside form the mean of the two frequencies' forms: a kept beat is slow, and two frequencies that
//! close differ in their forms, made through the far outside states, by their beat over those states' detunings.

use super::{Beat, Component};
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
            return Err("frameFrequenciesMHz: the ground states' frames must be 0 and the excited states' one frame".into());
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

/// The van Vleck form C(A, B) of an emitted coupling A at ν_A and an absorbed one B at ν_B (module documentation), on all the
/// states (STATES × STATES: its ground and excited blocks), with the bare energies `e`.
pub(crate) fn van_vleck(e: &[f64], a: &Mat, nu_a: f64, b: &Mat, nu_b: f64) -> Mat {
    let mut out = Mat::zeros(STATES, STATES);
    for x in 0..GROUND {
        for y in 0..GROUND {
            let mut z = C64::ZERO;
            for w in GROUND..STATES {
                let (p, q) = (a[(w, x)], b[(w, y)]);
                if p != C64::ZERO && q != C64::ZERO {
                    z += p.conj() * q * (0.5 * (1.0 / (e[y] + nu_b - e[w]) + 1.0 / (e[x] + nu_a - e[w])));
                }
            }
            out[(x, y)] = z;
        }
    }
    for f in GROUND..STATES {
        for g in GROUND..STATES {
            let mut z = C64::ZERO;
            for w in 0..GROUND {
                let (p, q) = (b[(f, w)], a[(g, w)]);
                if p != C64::ZERO && q != C64::ZERO {
                    z += p * q.conj() * (0.5 * (1.0 / (e[g] - e[w] - nu_a) + 1.0 / (e[f] - e[w] - nu_b)));
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

/// The far light of a frequency (module documentation): its exact co-rotating frequency δ_ω, its coupling bases, which of its
/// components fold far, its static form on its field's basis and per beam pair, the outside form at the focus, and its folded
/// components' jumps on the basis.
#[derive(Clone, Debug)]
pub(crate) struct Tone {
    /// δ_ω exactly (its index; no frame coordinate) and its value (rad/µs).
    pub(crate) beat: Beat,
    pub(crate) detuning: f64,
    /// [co-rotating, counter-rotating]: folded far.
    pub(crate) folded: [bool; 2],
    /// A_k'k (STATES × STATES) at `forms[k' r + k]` on the field's basis: the static form F(e) = Σ_k'k e_k'* e_k A_k'k.
    pub(crate) forms: Vec<Mat>,
    /// P_l'l (STATES × STATES) at `pairs[l' n + l]`, l' and l in the frequency's beam order.
    pub(crate) pairs: Vec<Mat>,
    /// O(E(0)): the outside form at the focus, which H0 holds.
    pub(crate) focus: Mat,
    /// Per component ([co, counter]) and basis vector k its folded jumps (GROUND × GROUND per axis a): the field's jump is
    /// Σ_k e_k J_{a,k} (the counter-rotating Σ_k e_k* J_{a,k}); empty where unfolded.
    pub(crate) jumps: [Vec<[Mat; 3]>; 2],
    /// Per component ([co, counter]) and beam its folded responses R_l (EXCITED × GROUND, the readout's); empty where unfolded.
    pub(crate) responses: [Vec<Mat>; 2],
    /// Per component ([co, counter]) and beam its coupling at the focus, X_l and R_l (STATES × STATES).
    pub(crate) couplings: [Vec<Mat>; 2],
    /// Per component ([co, counter]) its couplings on the field's basis, U_k and Ũ_k (STATES × STATES).
    pub(crate) basis: [Vec<Mat>; 2],
    /// The outside form O_pq in the spherical components (the far beats' cross terms, units.rs).
    pub(crate) outside: [[Mat; 3]; 3],
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
    /// A frequency's far light, its components folded as `folded` says ([co, counter]).
    pub(crate) fn new(request: &Request, light: &Light, axes: &Axes, frequency: usize, folded: [bool; 2]) -> Result<Tone, String> {
        let f = &light.frequencies[frequency];
        let bases = Bases::new(request, light, frequency)?;
        let (n, r) = (f.beams.len(), f.rank());
        let e = energies(request);
        let beat = axes.component(light, Component { frequency, counter: false });
        let nu = [axes.omega(&beat), axes.omega(&axes.component(light, Component { frequency, counter: true }))];
        let o = outside_form(&request.outside_forms[f.form].matrices);
        // A_kk' on the basis (row-major r × r): the outside form, the focus value taken before the folds are added.
        let mut a: Vec<Mat> = (0..r * r)
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
                focus.add_scaled(e0[k].conj() * e0[kk], &a[k * r + kk]);
            }
        }
        for k in 0..r {
            for kk in 0..r {
                if folded[0] {
                    a[k * r + kk].add_scaled(C64::ONE, &van_vleck(&e, &bases.co[k], nu[0], &bases.co[kk], nu[0]));
                }
                if folded[1] {
                    a[k * r + kk].add_scaled(C64::ONE, &van_vleck(&e, &bases.counter[kk], nu[1], &bases.counter[k], nu[1]));
                }
            }
        }
        let pairs = (0..n * n)
            .map(|x| {
                let (j, i) = (x / n, x % n);
                let mut m = Mat::zeros(STATES, STATES);
                for k in 0..r {
                    for kk in 0..r {
                        m.add_scaled(f.coefficients[j][k].conj() * f.coefficients[i][kk], &a[k * r + kk]);
                    }
                }
                m
            })
            .collect();
        let k0 = {
            let excited: Vec<usize> = (GROUND..STATES).collect();
            let mut k0 = request.hamiltonian.block(&excited, &excited);
            k0.add_scaled(C64::new(0.0, -0.5), &decay(request));
            k0
        };
        // Per component and beam: its coupling at the focus on the basis, and where folded its response and jumps.
        let weight = |c: usize, j: usize, k: usize| if c == 0 { f.coefficients[j][k] } else { f.coefficients[j][k].conj() };
        let per_beam = |c: usize, of: &dyn Fn(usize) -> Mat, rows: usize, cols: usize| -> Vec<Mat> {
            (0..n)
                .map(|j| {
                    let mut m = Mat::zeros(rows, cols);
                    for k in 0..r {
                        m.add_scaled(weight(c, j, k), &of(k));
                    }
                    m
                })
                .collect()
        };
        let couplings = [0, 1].map(|c| {
            let basis = if c == 0 { &bases.co } else { &bases.counter };
            per_beam(c, &|k| basis[k].clone(), STATES, STATES)
        });
        let folds: [Vec<(Mat, [Mat; 3])>; 2] = [0, 1].map(|c| {
            let basis = if c == 0 { &bases.co } else { &bases.counter };
            if folded[c] { basis.iter().map(|u| far_response(request, &k0, &e, u, nu[c])).collect() } else { vec![] }
        });
        let responses = [0, 1].map(|c| if folded[c] { per_beam(c, &|k| folds[c][k].0.clone(), EXCITED, GROUND) } else { vec![] });
        let jumps = [0, 1].map(|c| folds[c].iter().map(|f| f.1.clone()).collect());
        Ok(Tone { beat, detuning: nu[0], folded, forms: a, pairs, focus, jumps, responses, couplings, basis: [bases.co, bases.counter], outside: o })
    }
}

/// The response R and the jumps C_a R of a coupling U folded at ν: R[:,b] = (K₀ − E_b − ν)⁻¹ U[:,b] on the excited states
/// (EXCITED × GROUND), C_a R (GROUND × GROUND per axis a) (module documentation, # The jumps of a folded component).
fn far_response(request: &Request, k0: &Mat, e: &[f64], u: &Mat, nu: f64) -> (Mat, [Mat; 3]) {
    let mut out: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(GROUND, GROUND));
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
        for (ax, c) in request.emission.iter().enumerate() {
            for to in 0..GROUND {
                out[ax][(to, b)] = (0..EXCITED).map(|x| c[(to, GROUND + x)] * column[x]).sum();
            }
        }
    }
    (response, out)
}

/// The product a_l'* a_l of two beams' amplitudes (`first` = l', `second` = l) with its gradient and Hessian (jet.rs
/// `multiply`): the coefficient of a beam pair's term, of wave vector k_l − k_l'.
pub(crate) fn pair(first: &Amplitude, second: &Amplitude) -> Amplitude {
    let mut out = [C64::ZERO; 10];
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

    /// The van Vleck form is the second order of the exact Floquet spectrum. One component V e^{−iνt} + h.c. couples the ground
    /// states at E_b to the excited states at E_e − ν (the rotating frame of the absorbed photon): H_F = [[E_g, V†], [V, E_e − ν]],
    /// exactly Hermitian. Its ground-like eigenvalues are those of E_g + C(V, V)_gg up to fourth order in V/Δ, and its excited-like
    /// ones those of E_e − ν + C(V, V)_ee: so halving V cuts the difference sixteenfold, whatever the ground states' degeneracy
    /// (the vV block is a matrix there; its eigenvalues are compared). Bare energies with the hyperfine and the line spacings of
    /// Rb87 at the scale of the frame (rad/µs), ν a far detuning.
    #[test]
    fn the_van_vleck_shift_is_the_floquet_spectrum_s_second_order() {
        let e: Vec<f64> = (0..STATES)
            .map(|s| if s < 3 { -42943.0 + 13.0 * (s as f64 - 1.0) } else if s < GROUND { (s as f64 - 5.0) * 13.2 } else if s < 16 { -4.6e6 + 300.0 * s as f64 } else { 1.0e3 * (s as f64 - 24.0) })
            .collect();
        let nu = -2.5e6;
        let error = |scale: f64| {
            let v = coupling(scale, 7);
            let f = van_vleck(&e, &v, nu, &v, nu);
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
