//! The scene: the one place the request's Hamiltonian is built (RUSTPLAN §1.1, §2.1, §2.3, and its correction "one fold").
//!
//! H(r, t) = H0 − Σ_ω held(O_ω(E_ω(0))) + Σ_ω O_ω(E_ω(r)) + Σ_l [a_l(r) X_l e^{−i f_l t} + h.c.]
//!          + [a_tw(r)* R_tw e^{−i f̄ t} + h.c.]:
//! - every beam l: its co-rotating coupling X_l, all 32 states (both D lines, both ground F), at its rest-frame frequency
//!   f_l = ω_l − w_l, ω_l = frequency_index·Φ (the request's frames), w_l the beam's Doppler shift;
//! - the tweezer (the one allowed distinction) also its counter-rotating coupling R_tw at f̄ = −(ω_tw + 2ν) + w_tw, ν the
//!   carrier (the last frequency axis); every other beam in the RWA. The tweezer's components are its sidebands: folded
//!   by the one fold (model/rs.rs `fold`), in the reference and the model alike;
//! - the outside forms O_ω (request.rs `OutsideForm`) at each frequency's summed field E_ω(r) = Σ_{l∈ω} a_l(r) E_l, the
//!   focus value's diagonal taken out of H0 (request.rs `held`: H0 holds the diagonal and no more); the product
//!   conj(a_l) a_l′ B(E_l, E_l′) of two beams of one frequency turns at f_l′ − f_l;
//! - the jumps: the request's Cartesian emission (√Γ included) and each element's photon wave number k_eg = ω_eg/c.
//!
//! Allowed approximations (BRIEF_V3IMPL): the tweezer/non-tweezer split; RWA on non-tweezer light; no non-tweezer coupling
//! outside the 32 states; non-tweezer light on D1/D2 only (`request.rs` `validate`). Everything else to the rounding rule.
//!
//! A frequency is carried exactly as an integer vector (`Key`) over the generators -- every beam, then the carrier ν --:
//! equal frequencies of different beams are different keys, reduced exactly by the floquet frame (floquet/frame.rs). Each
//! beam is its own component, at its own Doppler shift (D2); two beams of one frequency and one Doppler shift beat at zero,
//! so their outside-form product is static.

use crate::field::{Amplitude, Geometry};
use crate::linalg::{dot3, C64, Mat, V3};
use crate::model::Quadratic;
use crate::request::{held, Request, GROUND, STATES};
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
/// tweezer's counter-rotating coupling, then the outside forms' products that beat.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Components {
    pub parts: Vec<Component>,
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
    /// H0 − Σ_ω held(O_ω(E_ω(0))).
    base: Mat,
    /// Each outside form's beams (request order), by form.
    members: Vec<Vec<usize>>,
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

impl Scene {
    /// Builds the scene; Err for a request outside the allowed approximations (`Request::validate`) or one whose tweezer
    /// has no frequency axis of its own (its sidebands are told by that axis, floquet/frame.rs).
    pub fn new(request: &Request) -> Result<Scene, String> {
        request.validate()?;
        let tweezer = request.trap_beam;
        let phi: Vec<f64> = request.frequencies_mhz.iter().map(|f| RAD_PER_US_PER_MHZ * f).collect();
        let axes = phi.len();
        let omega = |index: &[i32]| index.iter().zip(&phi).map(|(&i, p)| i as f64 * p).sum::<f64>();
        let beams = &request.beams;
        let own = (0..axes).any(|i| beams[tweezer].frequency_index[i] != 0 && beams.iter().enumerate().all(|(l, b)| l == tweezer || b.frequency_index[i] == 0));
        if !own {
            return Err(format!("The tweezer {} needs a frequency axis of its own (its sidebands are folded by that axis)", beams[tweezer].id));
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
        // The outside forms' members and the focus values' diagonals H0 holds.
        let mut base = request.hamiltonian.clone();
        let members: Vec<Vec<usize>> = request
            .outside_forms
            .iter()
            .map(|form| (0..beams.len()).filter(|&l| beams[l].frequency_index == form.frequency_index).collect())
            .collect();
        for (form, ls) in request.outside_forms.iter().zip(&members) {
            let field: [C64; 3] = std::array::from_fn(|q| ls.iter().map(|&l| beams[l].field[q]).sum());
            base.add_scaled(C64::real(-1.0), &held(&bilinear(&form.matrices, &field, &field)));
        }
        let frames = &request.frame_frequencies_mhz;
        let wave_number = Mat::from_fn(STATES, STATES, |i, j| {
            C64::real((energy[j] - energy[i] + RAD_PER_US_PER_MHZ * (frames[j] - frames[i])) / SPEED_OF_LIGHT)
        });
        Ok(Scene {
            request: request.clone(),
            generators,
            jumps: Jumps { emission: request.emission.clone(), wave_number },
            geometry,
            base,
            members,
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

    /// The key of one generator.
    pub fn unit(&self, g: usize) -> Key {
        (0..self.generators.len()).map(|h| (h == g) as i32).collect()
    }

    /// H's components at r for the momentum κ (1/µm).
    pub fn components_at(&self, r: V3, kappa: V3) -> Components {
        self.components(r, &self.doppler(kappa), 1.0)
    }

    /// H's components at r for given Doppler shifts w_l (rad/µs, one per beam, request order) with the tweezer's
    /// amplitude scaled by `tweezer` (its log-gradient unchanged): the two-point check's points (select.rs) -- the
    /// centre at 1, the edge at 0, both at w = 0.
    pub fn components(&self, r: V3, doppler: &[f64], tweezer: f64) -> Components {
        let request = &self.request;
        let beams = &request.beams;
        let tw = request.trap_beam;
        let amplitude: Vec<Amplitude> = self
            .geometry
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
            .collect();
        let rest = |l: usize| self.generators[l].frequency - doppler[l];
        let carrier = self.generators.len() - 1;
        // The static part: the base and the outside forms' products that do not beat.
        let mut value = self.base.clone();
        let mut gradient: [Mat; 3] = std::array::from_fn(|_| Mat::zeros(STATES, STATES));
        let mut beating = Vec::new();
        for (form, ls) in request.outside_forms.iter().zip(&self.members) {
            for (i, &l) in ls.iter().enumerate() {
                for &m in &ls[i..] {
                    let b = bilinear(&form.matrices, &beams[l].field, &beams[m].field);
                    let (al, am) = (&amplitude[l], &amplitude[m]);
                    let c = al.value.conj() * am.value;
                    let dc: [C64; 3] = std::array::from_fn(|k| al.gradient[k].conj() * am.value + al.value.conj() * am.gradient[k]);
                    if l == m {
                        value.add_scaled(c, &b);
                        for k in 0..3 {
                            gradient[k].add_scaled(dc[k], &b);
                        }
                    } else if doppler[l] == doppler[m] {
                        let adjoint = b.adjoint();
                        value.add_scaled(c, &b);
                        value.add_scaled(c.conj(), &adjoint);
                        for k in 0..3 {
                            gradient[k].add_scaled(dc[k], &b);
                            gradient[k].add_scaled(dc[k].conj(), &adjoint);
                        }
                    } else {
                        let key: Key = self.unit(m).iter().zip(self.unit(l)).map(|(x, y)| x - y).collect();
                        beating.push(Component {
                            key,
                            frequency: rest(m) - rest(l),
                            value: b.scaled(c),
                            gradient: std::array::from_fn(|k| b.scaled(dc[k])),
                        });
                    }
                }
            }
        }
        let mut parts = vec![Component { key: vec![0; self.generators.len()], frequency: 0.0, value, gradient }];
        for (l, b) in beams.iter().enumerate() {
            let a = &amplitude[l];
            parts.push(Component {
                key: self.unit(l),
                frequency: rest(l),
                value: b.coupling.scaled(a.value),
                gradient: std::array::from_fn(|k| b.coupling.scaled(a.gradient[k])),
            });
        }
        let a = &amplitude[tw];
        parts.push(Component {
            key: self.unit(tw).iter().zip(self.unit(carrier)).map(|(x, y)| -x - 2 * y).collect(),
            frequency: -(rest(tw) + 2.0 * self.generators[carrier].frequency),
            value: beams[tw].counter_coupling.scaled(a.value.conj()),
            gradient: std::array::from_fn(|k| beams[tw].counter_coupling.scaled(a.gradient[k].conj())),
        });
        parts.extend(beating);
        Components { parts }
    }

    /// The jump operators and the photons' wave numbers.
    pub fn jumps(&self) -> &Jumps {
        &self.jumps
    }

    /// The trap potential of each ground state at r (rad/µs): I_tw(r) (depth_g + Re O_tw(E_tw)_gg), `depth` the tweezer
    /// fold's ground shift at unit intensity (model/rs.rs `fold`, the one fold), O_tw the tweezer's own outside shift --
    /// zero where the tweezer is dark --, with its gradient and Hessian from I = |a|², ∇I = 2 Re(a*∇a),
    /// ∇∇I = 2 Re(∇a* ∇aᵀ + a*∇∇a). The one source of V_ref, the prepared wells, the loss energy and the reported trap.
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
