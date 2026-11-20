//! The scene: the one place the request's Hamiltonian is built (RUSTPLAN §1.1, §2.1, §2.3, and its correction "one fold").
//!
//! H(r, t) = H0 − Σ_ω O_ω(E_ω(0)) + Σ_ω O_ω(E_ω(r)) + Σ_l [a_l(r) X_l e^{−i f_l t} + h.c.]
//!          + [a_tw(r)* R_tw e^{−i f̄ t} + h.c.]:
//! - every beam l: its co-rotating coupling X_l, all 32 states (both D lines, both ground F), at its rest-frame frequency
//!   f_l = ω_l − w_l, ω_l = frequency_index·Φ (the request's frames), w_l the beam's Doppler shift;
//! - the tweezer (the one allowed distinction) also its counter-rotating coupling R_tw at f̄ = −(ω_tw + 2ν) + w_tw, ν the
//!   carrier (the last frequency axis); every other beam in the RWA. The tweezer's components are its sidebands: folded
//!   by the one fold (model/rs.rs `fold`), in the reference and the model alike;
//! - the outside forms O_ω (request.rs `OutsideForm`) at each frequency's summed field E_ω(r) = Σ_{l∈ω} a_l(r) E_l, the
//!   focus value O(0) taken out of H0 whole: the request's H0 is the spectrum of the bare atom with its Zeeman shift and
//!   the focus's outside forms, in that sum's eigenbasis (the generator's spectroscopy.ts `staticModelBackground`), so
//!   that with every light off H is the bare atom (pysel.status.md, "H0 约定": to 7.5·10⁻⁸ rad/µs on the six scenes;
//!   taking out the diagonal alone left 0.04–0.09 rad/µs in D2 and lifted the degeneracy of the hyperfine levels at
//!   B = 0); the product conj(a_l) a_l′ B(E_l, E_l′) of two beams of one frequency turns at f_l′ − f_l;
//! - the jumps: the request's Cartesian emission (√Γ included) and each element's photon wave number k_eg = ω_eg/c.
//!
//! Allowed approximations (BRIEF_V3IMPL): the tweezer/non-tweezer split; RWA on non-tweezer light; no non-tweezer coupling
//! outside the 32 states; non-tweezer light on D1/D2 only (`request.rs` `validate`). Everything else to the rounding rule.
//!
//! A frequency is carried exactly as an integer vector (`Key`) over the generators -- every beam, then the carrier ν --:
//! equal frequencies of different beams are different keys, reduced exactly by the floquet frame (floquet/frame.rs). Each
//! beam is its own component, at its own Doppler shift (D2); two beams of one frequency and one Doppler shift beat at zero,
//! so their outside-form product is static.

use crate::linalg::larger;
use crate::field::{Amplitude, Geometry};
use crate::linalg::{dot3, C64, Mat, V3};
use crate::model::Quadratic;
use crate::request::{Request, GROUND, STATES};
use crate::units::{hbar_over_mass, RAD_PER_US_PER_MHZ, SPEED_OF_LIGHT};

/// A frequency as an exact integer vector over the scene's generators (`Scene::generators`): every beam (its rest-frame
/// frequency f_l = ω_l − w_l), then the carrier ν.
pub type Key = Vec<i32>;

/// One frequency generator: a beam, or the carrier.
#[derive(Clone, Debug, PartialEq)]
pub struct Generator {
    /// The beam's index in `Request::beams`; None for the carrier ν.
    pub beam: Option<usize>,
    /// Its frequency on the request's frequency axes, exact: the beam's `frequency_index`, the carrier's unit on the last
    /// axis.
    pub frequency_index: Vec<i32>,
    /// ω = frequency_index · 2π `Request::frequencies_mhz` (rad/µs), in the frames of `Request::frame_frequencies_mhz`.
    pub frequency: f64,
}

/// One frequency component of H at a point: H holds `value` e^{−i f t} (+ h.c. for a nonzero key).
#[derive(Clone, Debug, PartialEq)]
pub struct Component {
    /// Its frequency's key; the zero key is the static part (Hermitian), every other key appears once, its Hermitian
    /// conjugate at −key implied.
    pub key: Key,
    /// f = Σ_g key_g f_g (rad/µs): the frequency in the atom's rest frame at the Doppler shifts asked for.
    pub frequency: f64,
    /// The 32 × 32 matrix at r (rad/µs), the beams' amplitudes a_l(r) (phase e^{ik·r} included) in it.
    pub value: Mat,
    /// ∂value/∂r_i, i = x, y, z (rad/µs/µm).
    pub gradient: [Mat; 3],
}

/// H(r) by its frequency components: the static part first, then each beam's co-rotating coupling (request order), the
/// tweezer's counter-rotating coupling, then the outside forms' products that beat -- the sums by key of the point's
/// terms (`point`), which the fold reads (model/rs.rs; the tweezer's amplitude a_tw(r), its scale applied, is the
/// point's).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Components {
    pub parts: Vec<Component>,
    pub point: Point,
}

/// The factor a term of H takes at a point: of the beams' amplitudes there, a_l(r) (`Point::amplitudes`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Factor {
    One,
    /// a_l.
    Beam(usize),
    /// a_l*.
    Conjugate(usize),
    /// a_l* a_m.
    Product(usize, usize),
}

impl Factor {
    /// Its value and gradient at the beams' amplitudes.
    pub fn at(self, a: &[Amplitude]) -> (C64, [C64; 3]) {
        match self {
            Factor::One => (C64::ONE, [C64::ZERO; 3]),
            Factor::Beam(l) => (a[l].value, a[l].gradient),
            Factor::Conjugate(l) => (a[l].value.conj(), a[l].gradient.map(C64::conj)),
            Factor::Product(l, m) => {
                let (al, am) = (&a[l], &a[m]);
                (al.value.conj() * am.value, std::array::from_fn(|k| al.gradient[k].conj() * am.value + al.value.conj() * am.gradient[k]))
            }
        }
    }
}

/// One term of H: `factor` times one of the scene's fixed matrices (`Scene::matrix`; its adjoint where `adjoint`), in the
/// component of `key` at the rest-frame frequency `frequency`.
#[derive(Clone, Debug, PartialEq)]
pub struct Term {
    pub key: Key,
    pub frequency: f64,
    pub matrix: usize,
    pub adjoint: bool,
    pub factor: Factor,
}

/// H at a point by its terms: the terms at the point's Doppler shifts (`Scene::terms`), the beams' amplitudes there
/// (`Scene::amplitudes`) and the generators' rest-frame frequencies (`Scene::rates`). H's components are their sums by
/// key (`Scene::components`); the fold contracts the factors with its own forms of the fixed matrices (model/rs.rs), so
/// that a point costs the amplitudes and the contraction.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Point {
    pub terms: Vec<Term>,
    pub amplitudes: Vec<Amplitude>,
    pub rates: Vec<f64>,
}

impl Point {
    /// A key's frequency at the point (`frequency`).
    pub fn frequency(&self, key: &Key) -> f64 {
        frequency(&self.rates, key)
    }
}

/// A key's frequency at the generators' rest-frame frequencies f_g: Σ_g key_g f_g (rad/µs). The one rule for every key's
/// frequency -- the terms', and every key the folds make of them (model/rs.rs).
pub fn frequency(rates: &[f64], key: &Key) -> f64 {
    key.iter().zip(rates).map(|(&k, f)| f64::from(k) * f).sum()
}

/// The jumps: the request's emission and the photon wave number of each element.
#[derive(Clone, Debug, PartialEq)]
pub struct Jumps {
    /// C_a, a = x, y, z (1/√µs, 32 × 32, ground ← excited): Σ_a C_a†C_a = Γ on the excited states.
    pub emission: [Mat; 3],
    /// k_ij = ω_ij / c (rad/µm, real; every entry, read where C_a is not zero), ω_ij = E_j − E_i + 2π(ν_j − ν_i) the
    /// frequency of the photon a jump j → i emits, E = diag H0 and ν the states' frames.
    pub wave_number: Mat,
}

/// The scene built from a request.
#[derive(Clone, Debug)]
pub struct Scene {
    pub request: Request,
    /// The keys' coordinates: every beam (request order), then the carrier.
    pub generators: Vec<Generator>,
    pub jumps: Jumps,
    /// Each beam's geometry (request order).
    pub geometry: Vec<Geometry>,
    /// H0 − Σ_ω O_ω(E_ω(0)): the bare atom with its Zeeman shift (every light off).
    base: Mat,
    /// The outside forms' products of their beams (the forms in request order, each its beams', l ≤ m): (l, m,
    /// B(E_l, E_m)).
    products: Vec<(usize, usize, Mat)>,
    /// ħ/m (µm²/µs).
    hbar_over_mass: f64,
}

/// B(E, E′) = Σ_pq E_p* E′_q A_pq of an outside form's nine matrices (A_qq = O_q, A_pq = (X − iY)/2, A_qp = (X + iY)/2
/// for the pair (p, q), p < q): B(E, E) = O(E) (request.rs `OutsideForm`).
pub fn bilinear(m: &[Mat; 9], e: &[C64; 3], f: &[C64; 3]) -> Mat {
    let mut out = Mat::zeros(STATES, STATES);
    for q in 0..3 {
        out.add_scaled(e[q].conj() * f[q], &m[q]);
    }
    for (i, (p, q)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        let (x, y) = (&m[3 + 2 * i], &m[4 + 2 * i]);
        let (pq, qp) = (e[p].conj() * f[q] * 0.5, e[q].conj() * f[p] * 0.5);
        out.add_scaled(pq, x);
        out.add_scaled(pq * C64::new(0.0, -1.0), y);
        out.add_scaled(qp, x);
        out.add_scaled(qp * C64::I, y);
    }
    out
}

/// The tweezer's own frequency axis, the one judgment of it (Scene::new and floquet/frame.rs `Frame::new`): an axis on
/// which the tweezer's index is nonzero and every other generator's zero -- every other beam's, and the carrier's, whose
/// index is the unit on the last axis, so the last axis is never the tweezer's own. The carrier ν is a generator of its
/// own (the excited states' frame; the counter-rotating coupling's 2ν), its count n_ν read off the last axis first
/// (`Frame::split`): an index there is a whole number of carriers, not the tweezer's frequency. The tweezer is the one
/// trap light, and this axis tells its beats with the other light -- its photons' frequency against that light's, the
/// tweezer split's sidebands that the one fold folds (model/rs.rs) -- in the frame's lattice (n_tw). The app's requests
/// hold the carrier last with every beam zero on it (full-master-model.ts: the beats' tree, then the reference's
/// carrier), so that a tweezer beside other light has one where the tree gives it a beat of its own; a lone tweezer is the
/// app's reference laser, its index zero (R1_TWEEZER, R1_SIGMA_P), and has none: with no other light there is no beat
/// to tell, and the fold counts its photons alone (`Rs::new`).
pub fn tweezer_axis(request: &Request) -> Option<usize> {
    let tweezer = request.trap_beam;
    let index = &request.beams[tweezer].frequency_index;
    let carrier = request.frequencies_mhz.len().checked_sub(1)?;
    (0..carrier).find(|&i| index[i] != 0 && request.beams.iter().enumerate().all(|(l, b)| l == tweezer || b.frequency_index[i] == 0))
}

impl Scene {
    /// Builds the scene; Err for a request outside the allowed approximations (`Request::validate`) or one whose tweezer
    /// beside other light has no frequency axis of its own (`tweezer_axis`).
    pub fn new(request: &Request) -> Result<Scene, String> {
        request.validate()?;
        let phi: Vec<f64> = request.frequencies_mhz.iter().map(|f| RAD_PER_US_PER_MHZ * f).collect();
        let axes = phi.len();
        let omega = |index: &[i32]| index.iter().zip(&phi).map(|(&i, p)| i as f64 * p).sum::<f64>();
        let beams = &request.beams;
        if tweezer_axis(request).is_none() && beams.len() > 1 {
            let id = &beams[request.trap_beam].id;
            return Err(format!("The tweezer {id} needs a frequency axis of its own beside the other light (its beats with that light are told by it)"));
        }
        let carrier: Vec<i32> = (0..axes).map(|i| (i + 1 == axes) as i32).collect();
        let generators = beams
            .iter()
            .enumerate()
            .map(|(l, b)| Generator { beam: Some(l), frequency_index: b.frequency_index.clone(), frequency: omega(&b.frequency_index) })
            .chain(std::iter::once(Generator { beam: None, frequency: omega(&carrier), frequency_index: carrier }))
            .collect();
        let geometry = beams.iter().map(|b| Geometry::new(b.direction, b.wave_number, b.waist)).collect();
        let energy: Vec<f64> = (0..STATES).map(|s| request.hamiltonian[(s, s)].re).collect();
        // The outside forms' members and their focus values, taken out of H0 whole.
        let mut base = request.hamiltonian.clone();
        let members: Vec<Vec<usize>> = request
            .outside_forms
            .iter()
            .map(|form| (0..beams.len()).filter(|&l| beams[l].frequency_index == form.frequency_index).collect())
            .collect();
        for (form, ls) in request.outside_forms.iter().zip(&members) {
            let field: [C64; 3] = std::array::from_fn(|q| ls.iter().map(|&l| beams[l].field[q]).sum());
            base.add_scaled(C64::real(-1.0), &bilinear(&form.matrices, &field, &field));
        }
        let frames = &request.frame_frequencies_mhz;
        let wave_number = Mat::from_fn(STATES, STATES, |i, j| {
            C64::real((energy[j] - energy[i] + RAD_PER_US_PER_MHZ * (frames[j] - frames[i])) / SPEED_OF_LIGHT)
        });
        let products = request
            .outside_forms
            .iter()
            .zip(&members)
            .flat_map(|(form, ls)| ls.iter().enumerate().flat_map(move |(i, &l)| ls[i..].iter().map(move |&m| (l, m, bilinear(&form.matrices, &beams[l].field, &beams[m].field)))))
            .collect();
        Ok(Scene {
            request: request.clone(),
            generators,
            jumps: Jumps { emission: request.emission.clone(), wave_number },
            geometry,
            base,
            products,
            hbar_over_mass: hbar_over_mass(request.mass_kg),
        })
    }

    /// Each beam's Doppler shift at the momentum κ (1/µm): w_l = (ħ/m)(k_l·κ + k_l²/2), the change of the atom's kinetic
    /// energy as it absorbs a photon of beam l (D2). The one definition.
    pub fn doppler(&self, kappa: V3) -> Vec<f64> {
        self.geometry
            .iter()
            .map(|g| {
                let k = g.wave_vector();
                self.hbar_over_mass * (dot3(k, kappa) + 0.5 * dot3(k, k))
            })
            .collect()
    }

    /// The tweezer's generator (its beam's): its co-rotating coupling has the key e_tw, its counter-rotating one
    /// −e_tw − 2e_ν, ν the carrier, the last generator.
    pub fn tweezer(&self) -> usize {
        self.request.trap_beam
    }

    /// The tweezer's two couplings at unit amplitude and at rest (no Doppler shift): (key, frequency, matrix) of the
    /// co-rotating X_tw at e_tw, f_tw = ω_tw, and of the counter-rotating R_tw at −e_tw − 2e_ν, −(ω_tw + 2ν). H holds
    /// a_tw X_tw and a_tw* R_tw (`components`); the one fold folds them once (model/rs.rs `Tweezer`).
    pub fn tweezer_couplings(&self) -> [(Key, f64, &Mat); 2] {
        let tw = self.request.trap_beam;
        let carrier = self.generators.len() - 1;
        let beam = &self.request.beams[tw];
        let counter: Key = self.unit(tw).iter().zip(self.unit(carrier)).map(|(x, y)| -x - 2 * y).collect();
        let f = self.generators[tw].frequency;
        [(self.unit(tw), f, &beam.coupling), (counter, -(f + 2.0 * self.generators[carrier].frequency), &beam.counter_coupling)]
    }

    /// H with every light off: the bare atom with its Zeeman shift (H0 − Σ_ω O_ω(E_ω(0))).
    pub fn dark(&self) -> &Mat {
        &self.base
    }

    /// The key of one generator.
    pub fn unit(&self, g: usize) -> Key {
        (0..self.generators.len()).map(|h| (h == g) as i32).collect()
    }

    /// The fixed matrices of H's terms (`Term::matrix`): 0 the base (H0 − Σ_ω O_ω(E_ω(0))), 1 + l beam l's co-rotating
    /// coupling X_l, 1 + L (L beams) the tweezer's counter-rotating R_tw, then the outside forms' products B(E_l, E_m).
    pub fn matrix(&self, index: usize) -> &Mat {
        let beams = &self.request.beams;
        match index {
            0 => &self.base,
            i if i <= beams.len() => &beams[i - 1].coupling,
            i if i == beams.len() + 1 => &beams[self.request.trap_beam].counter_coupling,
            i => &self.products[i - beams.len() - 2].2,
        }
    }

    /// The count of `matrix`'s indices.
    pub fn matrices(&self) -> usize {
        self.request.beams.len() + 2 + self.products.len()
    }

    /// H's terms at the Doppler shifts w_l (rad/µs, one per beam, request order): the static part's -- the base, and
    /// each outside form's own intensity terms |a_l|² B(E_l, E_l) --, each beam's a_l X_l at e_l, f_l = ω_l − w_l, the
    /// tweezer's a_tw* R_tw at −e_tw − 2e_ν, then the products of two beams of one frequency, conj(a_l) a_m B(E_l, E_m) at
    /// e_m − e_l, f_m − f_l (its conjugate the other way). The terms are the same at every Doppler shift (the structure is
    /// made once, model/rs.rs): a product of two beams at one Doppler shift has the frequency zero, it is not the static
    /// part (whose key is zero), and the static generator G₀ of a fold does not hold it.
    pub fn terms(&self, doppler: &[f64]) -> Vec<Term> {
        let beams = self.request.beams.len();
        let tw = self.request.trap_beam;
        let rates = self.rates(doppler);
        let zero: Key = vec![0; self.generators.len()];
        let term = |key: &Key, matrix: usize, adjoint: bool, factor: Factor| Term { key: key.clone(), frequency: frequency(&rates, key), matrix, adjoint, factor };
        let mut terms = vec![term(&zero, 0, false, Factor::One)];
        let mut beating = Vec::new();
        for (i, &(l, m, _)) in self.products.iter().enumerate() {
            let matrix = beams + 2 + i;
            if l == m {
                terms.push(term(&zero, matrix, false, Factor::Product(l, l)));
            } else {
                let key: Key = self.unit(m).iter().zip(self.unit(l)).map(|(x, y)| x - y).collect();
                beating.push(term(&key, matrix, false, Factor::Product(l, m)));
            }
        }
        for l in 0..beams {
            terms.push(term(&self.unit(l), 1 + l, false, Factor::Beam(l)));
        }
        let [_, (key, _, _)] = self.tweezer_couplings();
        terms.push(term(&key, beams + 1, false, Factor::Conjugate(tw)));
        terms.extend(beating);
        terms
    }

    /// The generators' rest-frame frequencies at the Doppler shifts w_l: every beam's f_l = ω_l − w_l, the carrier's ν.
    pub fn rates(&self, doppler: &[f64]) -> Vec<f64> {
        self.generators.iter().map(|g| g.frequency - g.beam.map_or(0.0, |l| doppler[l])).collect()
    }

    /// Each beam's amplitude a_l(r) with its gradient, the tweezer's scaled by `tweezer` (its log-gradient unchanged).
    pub fn amplitudes(&self, r: V3, tweezer: f64) -> Vec<Amplitude> {
        let tw = self.request.trap_beam;
        self.geometry
            .iter()
            .enumerate()
            .map(|(l, g)| {
                let a = g.amplitude_to(r, 1);
                if l == tw {
                    Amplitude { value: a.value * tweezer, gradient: a.gradient.map(|x| x * tweezer), hessian: a.hessian }
                } else {
                    a
                }
            })
            .collect()
    }

    /// H at r by its terms for the momentum κ (1/µm): what the fold reads at a point of a trajectory.
    pub fn point_at(&self, r: V3, kappa: V3) -> Point {
        self.point(r, &self.doppler(kappa), 1.0)
    }

    /// What the fold reads at a point of a trajectory (model/rs.rs `Tweezer::values`) into a buffer: the beams'
    /// amplitudes with their gradients at r and the generators' rest-frame frequencies at the momentum κ; no terms (the
    /// fold's structure is made once).
    pub fn point_into(&self, r: V3, kappa: V3, point: &mut Point) {
        point.terms.clear();
        point.amplitudes.clear();
        point.amplitudes.extend(self.geometry.iter().map(|g| g.amplitude_to(r, 1)));
        point.rates.clear();
        for g in &self.generators {
            let w = g.beam.map_or(0.0, |l| {
                let k = self.geometry[l].wave_vector();
                self.hbar_over_mass * (dot3(k, kappa) + 0.5 * dot3(k, k))
            });
            point.rates.push(g.frequency - w);
        }
    }

    /// H at r by its terms for given Doppler shifts w_l with the tweezer's amplitude scaled by `tweezer` (`components`).
    pub fn point(&self, r: V3, doppler: &[f64], tweezer: f64) -> Point {
        Point { terms: self.terms(doppler), amplitudes: self.amplitudes(r, tweezer), rates: self.rates(doppler) }
    }

    /// H's components at r for the momentum κ (1/µm).
    pub fn components_at(&self, r: V3, kappa: V3) -> Components {
        self.components(r, &self.doppler(kappa), 1.0)
    }

    /// H's components at r for given Doppler shifts w_l (rad/µs, one per beam, request order) with the tweezer's
    /// amplitude scaled by `tweezer` (its log-gradient unchanged): the two-point check's points (select.rs) -- the
    /// centre at 1, the edge at 0, both at w = 0. Each the sum of its terms (`point`), in their order.
    pub fn components(&self, r: V3, doppler: &[f64], tweezer: f64) -> Components {
        let point = self.point(r, doppler, tweezer);
        let mut parts: Vec<Component> = Vec::new();
        for t in &point.terms {
            let at = match parts.iter().rposition(|p| p.key == t.key) {
                Some(i) => i,
                None => {
                    let gradient = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
                    parts.push(Component { key: t.key.clone(), frequency: t.frequency, value: Mat::zeros(STATES, STATES), gradient });
                    parts.len() - 1
                }
            };
            let (c, dc) = t.factor.at(&point.amplitudes);
            let adjoint;
            let m = if t.adjoint {
                adjoint = self.matrix(t.matrix).adjoint();
                &adjoint
            } else {
                self.matrix(t.matrix)
            };
            let part = &mut parts[at];
            part.value.add_scaled(c, m);
            for k in 0..3 {
                part.gradient[k].add_scaled(dc[k], m);
            }
        }
        Components { parts, point }
    }

    /// The jump operators and the photons' wave numbers.
    pub fn jumps(&self) -> &Jumps {
        &self.jumps
    }

    /// The trap potential of each ground state at r (rad/µs): I_tw(r) (depth_g + Re O_tw(E_tw)_gg), `depth` the tweezer
    /// fold's ground shift at unit intensity (model/rs.rs `Tweezer::still`, the one fold), O_tw the tweezer's own outside
    /// shift -- zero where the tweezer is dark --, with its gradient and Hessian from I = |a|², ∇I = 2 Re(a*∇a),
    /// ∇∇I = 2 Re(∇a* ∇aᵀ + a*∇∇a). SPEC §1.3's V = Σ_b p_b W_b, the trap light's: the one source of V_ref, the prepared
    /// wells (only the trap light on) and the loss energy (the tweezer's conservative potential alone); the reported trap
    /// adds the other lights' smooth shifts to it (`running_potential`).
    pub fn trap_potential(&self, r: V3, depth: &[f64; GROUND]) -> [Quadratic; GROUND] {
        let request = &self.request;
        let tw = request.trap_beam;
        let a = self.geometry[tw].amplitude(r);
        let form = request.outside_forms.iter().find(|f| f.frequency_index == request.beams[tw].frequency_index).expect("validated: one form per frequency");
        let field = &request.beams[tw].field;
        let own = bilinear(&form.matrices, field, field);
        let intensity = a.value.norm_sqr();
        let slope: V3 = std::array::from_fn(|i| 2.0 * (a.value.conj() * a.gradient[i]).re);
        let curvature: [[f64; 3]; 3] =
            std::array::from_fn(|i| std::array::from_fn(|j| 2.0 * (a.gradient[i].conj() * a.gradient[j] + a.value.conj() * a.hessian[i][j]).re));
        std::array::from_fn(|g| {
            let d = depth[g] + own[(g, g)].re;
            Quadratic { value: intensity * d, gradient: slope.map(|x| x * d), hessian: curvature.map(|row| row.map(|x| x * d)) }
        })
    }
}

impl Scene {
    /// The other lights' smooth shifts of the ground states, for the reported trap (SPEC §1.3; refs/lib/phys.mjs
    /// `runningTrap`): each beam l of a frequency other than the tweezer's alone -- its own intensity, the phases between
    /// beams averaged --, in second order at the atom's levels in the trap light, with the first order in the trap
    /// light's intensity y (the focus' one): at the beam's focus amplitude,
    ///   U_lg(y) = −Re x†M(y)⁻¹x,  M(y) = K(y) − E_g(y) − f_l,  x = X_l[e, g] on the excited states e,
    ///   K(y) = H0_ee + (y − 1) O_ee + y S_ee − (i/2) D_ee,  E_g(y) = H0_gg + (y − 1) O_gg + y S_gg,
    /// O the tweezer's outside form in its own field, S the tweezer fold's static part at unit intensity (`shift`,
    /// model/rs.rs `Tweezer::still`: its ground diagonal the depth, its excited block the tweezer's shifts of the excited
    /// states through the ground states), D = Σ_a C_a†C_a, f_l the beam's frequency at rest (its recoil in: `doppler` at
    /// κ = 0), and
    ///   U′_lg(1) = Re[(M⁻†x)† (O_ee + S_ee − (O_gg + S_gg)) (M⁻¹x)]   (dM/dy, M at y = 1).
    /// For one excited level alone U is |X|²δ/(δ² + Γ²/4), δ the light's detuning from the line. The trap light moves the
    /// levels and mixes those of one m across the hyperfine levels, so the matrix is kept whole. Returns each such beam's
    /// index and per ground state (U_lg(1), U′_lg(1)) (rad/µs).
    pub fn smooth(&self, shift: &Mat) -> Vec<(usize, [[f64; 2]; GROUND])> {
        let request = &self.request;
        let tw = request.trap_beam;
        let index = &request.beams[tw].frequency_index;
        let form = request.outside_forms.iter().find(|f| &f.frequency_index == index).expect("validated: one form per frequency");
        let field = &request.beams[tw].field;
        let own = bilinear(&form.matrices, field, field);
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let mut decay = Mat::zeros(STATES, STATES);
        for c in &self.jumps.emission {
            decay.add_scaled(C64::ONE, &c.adjoint().product(c));
        }
        let rest = self.rates(&self.doppler([0.0; 3]));
        let ne = excited.len();
        let block = |m: &Mat| m.block(&excited, &excited);
        let (h0, o, s, d) = (block(&request.hamiltonian), block(&own), block(shift), block(&decay));
        request
            .beams
            .iter()
            .enumerate()
            .filter(|(_, b)| &b.frequency_index != index)
            .map(|(l, b)| {
                let shifts = std::array::from_fn(|g| {
                    let x: Vec<C64> = excited.iter().map(|&e| b.coupling[(e, g)]).collect();
                    if x.iter().all(|z| *z == C64::ZERO) {
                        return [0.0; 2];
                    }
                    let level = request.hamiltonian[(g, g)].re + shift[(g, g)].re + rest[l];
                    let m = Mat::from_fn(ne, ne, |i, j| h0[(i, j)] + s[(i, j)] - d[(i, j)] * C64::new(0.0, 0.5) - if i == j { C64::real(level) } else { C64::ZERO });
                    let slope = Mat::from_fn(ne, ne, |i, j| o[(i, j)] + s[(i, j)] - if i == j { C64::real(own[(g, g)].re + shift[(g, g)].re) } else { C64::ZERO });
                    let (right, left) = (crate::linalg::Lu::new(&m).expect("M has an inverse: the excited states decay"), crate::linalg::Lu::new(&m.adjoint()).expect("M† too"));
                    let (mut y, mut w) = (x.clone(), x.clone());
                    right.solve(&mut y);
                    left.solve(&mut w);
                    let value = -crate::linalg::inner(&x, &y).re;
                    let derivative = crate::linalg::inner(&w, &slope.apply(&y)).re;
                    [value, derivative]
                });
                (l, shifts)
            })
            .collect()
    }

    /// The reported trap's potential of each ground state at r (SPEC §1.3, "运行阱"): the trap light's (`trap_potential`)
    /// with the other lights' smooth shifts (`smooth`), each beam's own intensity I_l(r) = |a_l(r)|² times its shift to
    /// first order in the trap light's intensity y(r) = I_tw(r): Σ_l I_l [U_lg + U′_lg (y − 1)], with the gradient and
    /// Hessian of the products. At a centre where the trap light's coefficients are stationary (the focus) the Hessian is
    /// ∂²V_prep (1 + U′/(O_gg + S_gg)) + the beams' own curvature, exactly; elsewhere the trap light's second order on the
    /// lines is left out (SPEC §1.3).
    pub fn running_potential(&self, r: V3, depth: &[f64; GROUND], smooth: &[(usize, [[f64; 2]; GROUND])]) -> [Quadratic; GROUND] {
        let mut wells = self.trap_potential(r, depth);
        let intensity = |a: &Amplitude| -> (f64, V3, [[f64; 3]; 3]) {
            (a.value.norm_sqr(), std::array::from_fn(|i| 2.0 * (a.value.conj() * a.gradient[i]).re), std::array::from_fn(|i| std::array::from_fn(|j| 2.0 * (a.gradient[i].conj() * a.gradient[j] + a.value.conj() * a.hessian[i][j]).re)))
        };
        let (y, dy, ddy) = intensity(&self.geometry[self.request.trap_beam].amplitude(r));
        for &(l, shifts) in smooth {
            let (i, di, ddi) = intensity(&self.geometry[l].amplitude(r));
            for (well, [u, slope]) in wells.iter_mut().zip(shifts) {
                let f = u + slope * (y - 1.0);
                well.value += i * f;
                for a in 0..3 {
                    well.gradient[a] += di[a] * f + i * slope * dy[a];
                    for b in 0..3 {
                        well.hessian[a][b] += ddi[a][b] * f + slope * (di[a] * dy[b] + dy[a] * di[b]) + i * slope * ddy[a][b];
                    }
                }
            }
        }
        wells
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tweezer's own axis (`tweezer_axis`): R1_TWEEZER and R1_SIGMA_P, the app's lone tweezers (the reference laser:
    /// index zero on their one axis, the carrier), have none and are scenes; R4_LIN_4's tweezer has one; on the sample with
    /// a beat axis and the carrier, a tweezer with a beat of its own beside other light, or alone at index zero, is a
    /// scene; one at the reference (index zero), one sharing its beat axis, or one on the carrier's axis alone beside other
    /// light is refused.
    #[test]
    fn a_tweezer_needs_an_axis_of_its_own_beside_other_light() {
        for (name, own) in [("R1_TWEEZER", None), ("R1_SIGMA_P", None), ("R4_LIN_4", Some(1))] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
            let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let request = Request::from_native(&line).unwrap();
            assert_eq!(tweezer_axis(&request), own, "{name}");
            let scene = Scene::new(&request).unwrap_or_else(|e| panic!("{name}: {e}"));
            assert_eq!(scene.request.frequencies_mhz, request.frequencies_mhz);
        }
        let with = |tweezer: [i32; 2], other: Option<[i32; 2]>| {
            let mut r = crate::request::tests::sample();
            r.frequencies_mhz = vec![1.0, 3.8e8];
            r.beams[0].frequency_index = tweezer.to_vec();
            r.outside_forms[0].frequency_index = tweezer.to_vec();
            match other {
                Some(i) => {
                    r.beams[1].frequency_index = i.to_vec();
                    r.outside_forms[1].frequency_index = i.to_vec();
                }
                None => {
                    r.beams.truncate(1);
                    r.outside_forms.truncate(1);
                }
            }
            r
        };
        for (tweezer, other) in [([0, 0], [1, 0]), ([1, 0], [2, 0]), ([0, 1], [1, 0])] {
            let e = Scene::new(&with(tweezer, Some(other))).err().expect("refused");
            assert!(e.contains("axis of its own"), "{e}");
        }
        assert_eq!(tweezer_axis(&with([-4, 0], Some([0, 0]))), Some(0));
        assert!(Scene::new(&with([-4, 0], Some([0, 0]))).is_ok());
        assert!(Scene::new(&with([0, 0], None)).is_ok());
    }

    /// With every light off H is the bare atom (pysel.status.md, "H0 约定"): on R4_LIN_4 (B = 0) the spectrum of
    /// H0 − Σ_ω O_ω(E_ω(0)) is the hyperfine levels, each 2F + 1 fold degenerate, to the rounding of a Hermitian
    /// eigensolver on H, n u ‖H‖_F (n = 32; ‖H‖ ~ 4.5·10⁷ rad/µs of the D lines in one frame): the gaps within a level
    /// below it, the levels' seven gaps far above it. Taking the focus's diagonal alone out (the held convention) left the
    /// D2 levels split by 0.04–0.09 rad/µs.
    #[test]
    fn the_dark_scene_is_the_bare_atom() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R4_LIN_4.json");
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let scene = Scene::new(&Request::from_native(&line).unwrap()).unwrap();
        let dark = scene.dark();
        let values = crate::linalg::hermitian_eigen(dark).values;
        let levels = {
            let mut seen: Vec<(u8, u8)> = scene.request.levels().unwrap().iter().map(|l| (l.manifold as u8, l.f)).collect();
            seen.sort_unstable();
            seen.dedup();
            seen.len()
        };
        let rounding = STATES as f64 * crate::units::UNIT_ROUNDOFF * dark.frobenius();
        let mut gaps: Vec<f64> = values.windows(2).map(|w| w[1] - w[0]).collect();
        gaps.sort_by(|a, b| b.total_cmp(a));
        let (between, within) = gaps.split_at(levels - 1);
        eprintln!("levels {levels}: smallest gap between {:.3e}, largest within {:.3e}, rounding {rounding:.3e} rad/µs", between[levels - 2], within[0]);
        assert!(within[0] <= rounding && between[levels - 2] > rounding);
    }

    /// The one Doppler shift per beam (model/rs.rs, # The frequencies): on R4 (two counter-propagating beams of one
    /// frequency, k_m = −k_l) at a momentum along them, a beam's key holds f_l = ω_l − ħ(k_l·κ + |k_l|²/2)/m and the
    /// product of the two, at e_m − e_l, the beams' sum f_m − f_l = −ħ(k_m − k_l)·κ/m: their recoils cancel (equal
    /// wavelengths), no |K|²/2 = 4 single-photon recoils; at rest it is zero.
    #[test]
    fn a_key_s_frequency_is_its_beams_sum() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R4_LIN_4.json");
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let scene = Scene::new(&Request::from_native(&line).unwrap()).unwrap();
        let hm = scene.hbar_over_mass;
        let (l, m) = (0..scene.geometry.len())
            .flat_map(|l| (0..scene.geometry.len()).map(move |m| (l, m)))
            .find(|&(l, m)| l < m && scene.request.beams[l].frequency_index == scene.request.beams[m].frequency_index && dot3(scene.geometry[l].direction, scene.geometry[m].direction) < -0.5)
            .expect("two counter-propagating beams of one frequency");
        let (kl, km) = (scene.geometry[l].wave_vector(), scene.geometry[m].wave_vector());
        for kappa in [[0.0; 3], kl.map(|x| 3.0 * x), [12.0, -7.0, 40.0]] {
            let point = scene.point_at([0.0; 3], kappa);
            let key: Key = scene.unit(m).iter().zip(scene.unit(l)).map(|(x, y)| x - y).collect();
            let raman = point.frequency(&key);
            let doppler: f64 = (0..3).map(|a| (km[a] - kl[a]) * kappa[a]).sum::<f64>() * hm;
            let single = point.frequency(&scene.unit(l));
            let expected = scene.generators[l].frequency - hm * (dot3(kl, kappa) + 0.5 * dot3(kl, kl));
            println!("κ {kappa:?}: the Raman key {raman:.6e} against −ħK·κ/m {:.6e}; the single photon's {single:.12e} against {expected:.12e}", -doppler);
            assert!((raman + doppler).abs() <= 1e-12 * (1.0 + doppler.abs()));
            assert_eq!(single, expected);
        }
    }

    /// The reported trap (SPEC §1.3) of every reference scene that states trap frequencies (refs/expect, the references
    /// refs/lib/phys.mjs `preparedTrap` / `runningTrap`), judged as refs/check.mjs judges them: |engine − reference| ≤
    /// 2u, u the reference's own uncertainty. The engine's: the running potential (`running_potential`, the tweezer
    /// fold's static part from model/rs.rs `Tweezer::still`) of the reference state -- the initial component of the
    /// largest population --, its descent from the focus (model/potential.rs `descend`, the one descent) on every beam's
    /// shortest length within the trap's region, the frequencies along the lab axes there. A scene the scene refuses is
    /// said with its reason.
    #[test]
    fn the_running_trap_is_the_references() {
        let refs = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs");
        let json = |path: std::path::PathBuf| -> serde_json::Value { serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap() };
        let (mut seen, mut beyond) = (0, vec![]);
        let mut paths: Vec<_> = std::fs::read_dir(refs.join("expect")).unwrap().map(|e| e.unwrap().path()).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
        paths.sort();
        for path in paths {
            let expectation = json(path);
            let Some(series) = expectation["series"].as_array() else { continue };
            let stated: Vec<&serde_json::Value> = series.iter().filter(|s| s["quantity"] == "trapFrequency").collect();
            if stated.is_empty() {
                continue;
            }
            let id = expectation["id"].as_str().unwrap();
            let made = Request::from_native(&json(refs.join(expectation["request"].as_str().unwrap()))).and_then(|request| Scene::new(&request));
            let scene = match made {
                Ok(scene) => scene,
                Err(why) => {
                    println!("{id}: refused: {why}");
                    continue;
                }
            };
            let still = crate::model::rs::Tweezer::new(&scene).still();
            let depth: [f64; GROUND] = std::array::from_fn(|g| still[(g, g)].re);
            let smooth = scene.smooth(&still);
            let request = &scene.request;
            let chief = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
            let k = request.beams.iter().map(|b| b.wave_number).fold(0.0, larger);
            let tw = request.trap_beam;
            let trap_light: Vec<usize> = (0..request.beams.len()).filter(|&l| request.beams[l].frequency_index == request.beams[tw].frequency_index).collect();
            let outside = |r: V3| trap_light.iter().all(|&l| scene.geometry[l].amplitude_to(r, 0).value.norm_sqr() < crate::units::REGION_INTENSITY);
            let expand = |r: V3| {
                let q = scene.running_potential(r, &depth, &smooth)[chief];
                let scale = q.value.abs();
                (q, crate::units::rounding(GROUND + 2 * smooth.len()) * scale)
            };
            let trap = crate::model::potential::descend(1.0 / k, scene.hbar_over_mass, &outside, &expand).unwrap();
            for s in stated {
                let axis = ["x", "y", "z"].iter().position(|a| s["axis"] == *a).unwrap();
                let (value, u) = (s["value"].as_f64().unwrap(), s["u"].as_f64().unwrap());
                let engine = trap.lab[axis] / std::f64::consts::TAU * 1e3;
                println!("{id} {}: engine {engine:.12e} reference {value:.12e} u {u:.3e} (d/u {:.4}; {} other beams)", s["axis"], (engine - value) / u, smooth.len());
                if (engine - value).abs() > 2.0 * u {
                    beyond.push(format!("{id} {}", s["axis"]));
                }
                seen += 1;
            }
        }
        assert!(seen > 0 && beyond.is_empty(), "{} of {seen} axes beyond 2u: {beyond:?}", beyond.len());
    }
}
