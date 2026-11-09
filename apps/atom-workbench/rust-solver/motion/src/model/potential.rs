//! The one potential (SPEC §1.3): V_L(r; p) = Σ_b p_b [h_L(r) − h_{L∖trap}(r)]_bb, h the eliminated generator's static Hermitian
//! part -- the same terms the assembly sums (far.rs, eliminate.rs) --, p the ground populations, L the lights on, the trap light
//! the whole field of the trap beam's frequency. Everything else about the trap reads it: the loss energy, the prepared
//! potential V_prep(r; p) = V_{L_prep}(r; p), the trap's centre, its analytic Hessian and frequencies, the envelope, the reach.
//!
//! # What of h the difference holds
//! Every light's far part is a sum over its beam pairs (far.rs); the difference h_L − h_{L∖trap} holds of it the trap
//! frequency's alone, exactly -- the other frequencies' far terms and every constant (H0, the focus values of the outside
//! forms) cancel. The near light's static shifts on the ground states differ with the trap light on and off (its far shifts
//! move the near lines), so they enter as the difference of the two eliminations (eliminate.rs). The diagonal of the static
//! Hermitian part is Re Σ c(r) M_bb over the static terms: a term and its adjoint pair (c_l'l = c_ll'*, M_l'l = M_ll'†) sum to
//! twice its real part.
//!
//! # The trap (`Model::trap`)
//! The descent from the focus to the local minimum of V_L(r; p) -- the prepared lights' for the initial wells, the running ones'
//! for the reported trap (SPEC §1.3) --: Newton's step along each eigenvector of the analytic Hessian
//! curved upward, downhill by the trust radius along the others (along an eigenvector without slope at a top: as it is), all
//! within the trust radius, from the light's shortest length 1/max k_l up; doubled after a cut step that lowered V, halved
//! where V did not fall. It ends where a step's predicted gain is within the rounding of V's sum; the point is the trap's
//! centre where every curvature there is positive beyond that rounding over the light's shortest length (a bowl), and the
//! request is refused where it is not, or where the descent leaves the trap's region: no minimum holds the atom ("光镊不约束
//! 原子"), and the focus is never taken instead. A focus that is a bowl's bottom is the centre as it stands, bit for bit.

use super::{Envelope, Lights, Model, Quadratic, Trap};
use crate::field::Amplitude;
use crate::linalg::{dot3, symmetric3_eigen};
use crate::request::GROUND;
use crate::units::rounding;

/// A potential's expansion with the scale of its sum (Σ of its terms' moduli): the rounding of its value is
/// `rounding(terms)·scale`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Expansion {
    pub(crate) quadratic: Quadratic,
    pub(crate) scale: f64,
    pub(crate) terms: usize,
}

impl Expansion {
    /// The rounding of its value.
    pub(crate) fn noise(&self) -> f64 {
        rounding(self.terms.max(1)) * self.scale
    }
}

/// Σ_b p_b M_bb.
fn contract(m: &crate::linalg::Mat, p: &[f64; GROUND]) -> crate::linalg::C64 {
    (0..GROUND).filter(|&b| p[b] != 0.0).map(|b| m[(b, b)] * p[b]).sum()
}

impl Model {
    /// The far part of Σ_b p_b [h(r)]_bb of one frequency's light, with its gradient and Hessian: Re Σ_l'l c_l'l(r) [P_l'l]_p
    /// over its beam pairs (far.rs), c_l'l = a_l'* a_l.
    pub(crate) fn far_potential(&self, frequency: usize, amplitudes: &[Amplitude], p: &[f64; GROUND], lights: &Lights) -> Expansion {
        let (f, tone) = (&self.light.frequencies[frequency], &self.tones[frequency]);
        let n = f.beams.len();
        let mut out = Expansion { quadratic: Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] }, scale: 0.0, terms: 0 };
        for j in 0..n {
            for i in 0..n {
                let m = contract(&tone.pairs[j * n + i], p);
                if m == crate::linalg::C64::ZERO || !lights.pair(frequency, j, i) {
                    continue;
                }
                let c = super::far::pair(&amplitudes[f.beams[j]], &amplitudes[f.beams[i]]);
                let q = &mut out.quadratic;
                q.value += (c.value * m).re;
                for x in 0..3 {
                    q.gradient[x] += (c.gradient[x] * m).re;
                    for y in 0..3 {
                        q.hessian[x][y] += (c.hessian[x][y] * m).re;
                    }
                }
                out.scale += (c.value * m).abs();
                out.terms += 1;
            }
        }
        out
    }

    /// V_L(r; p) with its gradient and Hessian, and the rounding of its sum (module documentation).
    pub(crate) fn expand(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights) -> Expansion {
        let trap = self.light.trap;
        if !lights.on(trap) {
            return Expansion { quadratic: Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] }, scale: 0.0, terms: 0 };
        }
        let amplitudes = self.light.amplitudes(r);
        let mut out = self.far_potential(trap, &amplitudes, p, lights);
        let near = self.near_potential(r, &amplitudes, p, lights);
        let q = &mut out.quadratic;
        q.value += near.quadratic.value;
        for x in 0..3 {
            q.gradient[x] += near.quadratic.gradient[x];
            for y in 0..3 {
                q.hessian[x][y] += near.quadratic.hessian[x][y];
            }
        }
        out.scale += near.scale;
        out.terms += near.terms;
        out
    }

    /// V_L(r; p) (rad/µs).
    pub fn potential(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights) -> f64 {
        self.expand(r, p, lights).quadratic.value
    }

    /// V_L(r; p)'s second-order expansion at r, analytic.
    pub fn expansion(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights) -> Quadratic {
        self.expand(r, p, lights).quadratic
    }

    /// The light's shortest length 1/max_l k_l over the lit beams (µm): the scale on which the potential varies (a standing
    /// wave's λ/2π), the descent's first trust radius and the bowl's test length.
    pub(crate) fn shortest_length(&self, lights: &Lights) -> f64 {
        let k = self.light.frequencies.iter().enumerate().filter(|(w, _)| lights.on(*w)).flat_map(|(_, f)| f.beams.iter().map(|&l| self.light.beams[l].wave_number)).fold(0.0, f64::max);
        1.0 / k
    }

    /// The trap of V_L(r; p) for the ground populations p and the lights `lights` (module documentation, # The trap).
    pub fn trap(&self, p: &[f64; GROUND], lights: &Lights) -> Result<Trap, String> {
        let length = self.shortest_length(lights);
        let mut r = [0.0; 3];
        let mut here = self.expand(r, p, lights);
        let mut radius = length;
        loop {
            let (values, vectors) = symmetric3_eigen(&here.quadratic.hessian);
            let along = vectors.map(|e| dot3(e, here.quadratic.gradient));
            let noise = here.noise();
            let bowl = values.iter().all(|&l| l * length * length > noise);
            let steps: [f64; 3] = std::array::from_fn(|i| {
                if values[i] * length * length > noise {
                    -along[i] / values[i]
                } else if along[i] != 0.0 {
                    -along[i].signum() * radius
                } else if values[i] * length * length < -noise {
                    radius
                } else {
                    0.0
                }
            });
            let size = steps.iter().map(|x| x * x).sum::<f64>().sqrt();
            loop {
                let scale = if size > radius { radius / size } else { 1.0 };
                let gain = -(0..3).map(|i| along[i] * steps[i] * scale + 0.5 * values[i] * (steps[i] * scale).powi(2)).sum::<f64>();
                if !(gain > noise) {
                    if !bowl {
                        return Err("光镊不约束原子：准备势从焦点下降没有到达局部极小".into());
                    }
                    let frequencies = values.map(|l| (l * self.hbar_over_mass).sqrt());
                    let lab = std::array::from_fn(|a| (here.quadratic.hessian[a][a].max(0.0) * self.hbar_over_mass).sqrt());
                    return Ok(Trap { centre: r, minimum: here.quadratic.value, axes: vectors, frequencies, lab });
                }
                let next: [f64; 3] = std::array::from_fn(|a| r[a] + (0..3).map(|i| steps[i] * scale * vectors[i][a]).sum::<f64>());
                if self.light.outside(&self.light.amplitudes(next)) {
                    return Err("光镊不约束原子：准备势的下降离开了光镊的区域".into());
                }
                let there = self.expand(next, p, lights);
                if there.quadratic.value < here.quadratic.value {
                    r = next;
                    here = there;
                    if scale < 1.0 {
                        radius *= 2.0;
                    }
                    break;
                }
                radius = radius.min(size) * 0.5;
            }
        }
    }

    /// The prepared potential's proven envelope for ground populations p (model.rs `Envelope`): |Q_l'l| = |[P_l'l]_p| of the
    /// trap frequency's far static form, and, where the trap's co-rotating component is a near carrier, its light shift's
    /// bound per beam pair (eliminate.rs).
    pub fn envelope(&self, p: &[f64; GROUND]) -> Envelope {
        let f = &self.light.frequencies[self.light.trap];
        let tone = &self.tones[self.light.trap];
        let n = f.beams.len();
        let near = self.near_envelope(p);
        let weights = (0..n * n).map(|x| contract(&tone.pairs[x], p).abs() + near.get(x).copied().unwrap_or(0.0)).collect();
        Envelope { beams: f.beams.clone(), weights }
    }
}

impl Envelope {
    /// Σ_l'l |Q_l'l| √(A_l'(r) A_l(r)) at r, A_l = 1/(1 + u_l), u_l = 2ρ²/w₀² + ζ²/z_R² (a plane wave's A = 1): the proven
    /// bound of −V_prep(r; p).
    pub fn bound(&self, light: &crate::field::Light, r: [f64; 3]) -> f64 {
        let a: Vec<f64> = self.beams.iter().map(|&l| peak(&light.beams[l], r).sqrt()).collect();
        let n = self.beams.len();
        (0..n * n).map(|x| self.weights[x] * a[x / n] * a[x % n]).sum()
    }
    /// Σ_l'l |Q_l'l|: the bound's largest value, B in −V_prep ≤ B max_l A_l(r).
    pub fn peak(&self) -> f64 {
        self.weights.iter().sum()
    }
}

/// A_l(r) = 1/(1 + u), u = 2ρ²/w₀² + ζ²/z_R²: the bound of a paraxial beam's |a_l(r)|² (model.rs `Envelope`); 1 for a plane wave.
pub(crate) fn peak(beam: &crate::field::Geometry, r: [f64; 3]) -> f64 {
    match (beam.waist, beam.rayleigh()) {
        (Some(w), Some(z)) => {
            let zeta = dot3(beam.direction, r);
            let rho2 = (dot3(r, r) - zeta * zeta).max(0.0);
            1.0 / (1.0 + 2.0 * rho2 / (w * w) + zeta * zeta / (z * z))
        }
        _ => 1.0,
    }
}
