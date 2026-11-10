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

use super::assemble::{Held, Site, LEFT, RIGHT};
use super::tabulate::Tables;
use super::{jet, Envelope, Lights, Lit, Model, Quadratic, Trap};
use crate::linalg::C64;
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

    /// V_L(r; p) with, to `order`, its gradient and Hessian, and the rounding of its sum (module documentation); the near light from
    /// its tables where they serve (tabulate.rs `Model::near`).
    pub(crate) fn expand(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights, order: usize, tables: Option<&super::tabulate::Tables>) -> Expansion {
        let trap = self.light.trap;
        if !lights.on(trap) {
            return Expansion { quadratic: Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] }, scale: 0.0, terms: 0 };
        }
        // The beams' amplitudes into a thread's spare (assemble.rs `Held`): a step's reference asks the heap for nothing.
        let mut spare = Held::<Site>::take();
        self.light.amplitudes_into(r, &mut spare.beams);
        let amplitudes = &spare.beams;
        let mut out = self.far_potential(trap, amplitudes, p, lights);
        let near = self.near_potential(amplitudes, p, lights, order, tables);
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

    /// The near carriers' part of V_L(r; p) (module documentation, # What of h the difference holds): Σ_b p_b of their static shifts
    /// on the ground states, each in its cluster -- the cluster's Hermitian near forms on its diagonal at a base beat of zero
    /// (assemble.rs `Plan::links`), each times its two field components at zero momentum --, with the trap light on less with it
    /// off, jets to `order`; the near forms from their tables where they serve (to first order).
    pub(crate) fn near_potential(&self, amplitudes: &[Amplitude], p: &[f64; GROUND], lights: &Lights, order: usize, tables: Option<&Tables>) -> Expansion {
        let mut out = Expansion { quadratic: Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] }, scale: 0.0, terms: 0 };
        if self.carriers.iter().all(|x| !lights.on(x.frequency)) {
            return out;
        }
        let mut off = lights.clone();
        off.0[self.light.trap] = Lit::Off;
        let parts = jet::parts(order);
        let unit: Vec<C64> = (0..parts).map(|part| if part == 0 { C64::ONE } else { C64::ZERO }).collect();
        for (lights, sign) in [(lights, 1.0), (&off, -1.0)] {
            for cluster in 0..self.clusters.len() {
                let c = &self.clusters[cluster];
                if c.states[..c.ground].iter().all(|&s| p[s] == 0.0) {
                    continue;
                }
                let plan = self.plan(cluster, lights);
                if plan.links.is_empty() {
                    continue;
                }
                let near = match tables.and_then(|t| t.of(cluster, lights)) {
                    Some(table) if order <= 1 => self.near_tabulated(table, amplitudes, &[(plan.near.hermitian(), order)]),
                    _ => self.near_exact(&plan, amplitudes, order, &self.coherent_dressing()),
                };
                let fields = self.fields(&plan, amplitudes, [0.0; 3], order);
                for link in &plan.links {
                    let (x, y) = link.place;
                    if x != y || x >= c.ground || p[c.states[x]] == 0.0 || !plan.statics[link.block] {
                        continue;
                    }
                    let (left, right) = plan.factors(link.block);
                    let side = |s: Option<usize>, weighted: bool| s.map_or(&unit[..], |s| if weighted { fields.doppler(s) } else { fields.plain(s) });
                    // A whole jet's parts hold any order's.
                    let (mut coefficient, mut form, mut term) = ([C64::ZERO; 10], [C64::ZERO; 10], [C64::ZERO; 10]);
                    jet::multiply(side(left, link.flag == LEFT), side(right, link.flag == RIGHT), &mut coefficient[..parts]);
                    near.jet(link.operand, &mut form[..parts]);
                    if link.conjugate {
                        form.iter_mut().for_each(|z| *z = z.conj());
                    }
                    jet::multiply(&coefficient[..parts], &form[..parts], &mut term[..parts]);
                    let (q, w) = (&mut out.quadratic, sign * p[c.states[x]]);
                    q.value += w * term[0].re;
                    for i in 0..3 {
                        if order >= 1 {
                            q.gradient[i] += w * term[1 + i].re;
                        }
                        for j in 0..3 {
                            if order >= 2 {
                                q.hessian[i][j] += w * term[jet::second(i, j)].re;
                            }
                        }
                    }
                    out.scale += (coefficient[0] * form[0]).abs() * p[c.states[x]];
                    out.terms += 1;
                }
            }
        }
        out
    }

    /// V_L(r; p) (rad/µs).
    pub fn potential(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights) -> f64 {
        self.expand(r, p, lights, 0, None).quadratic.value
    }

    /// V_L(r; p)'s second-order expansion at r, analytic.
    pub fn expansion(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights) -> Quadratic {
        self.expand(r, p, lights, 2, None).quadratic
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
        let mut here = self.expand(r, p, lights, 2, None);
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
                    let lab = std::array::from_fn(|a| (here.quadratic.hessian[a][a] * self.hbar_over_mass).sqrt());
                    return Ok(Trap { centre: r, minimum: here.quadratic.value, axes: vectors, frequencies, lab });
                }
                let next: [f64; 3] = std::array::from_fn(|a| r[a] + (0..3).map(|i| steps[i] * scale * vectors[i][a]).sum::<f64>());
                if self.light.outside(next) {
                    return Err("光镊不约束原子：准备势的下降离开了光镊的区域".into());
                }
                let there = self.expand(next, p, lights, 2, None);
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
        let weights = (0..n * n).map(|x| contract(&tone.pairs[x], p).abs() + near[x]).collect();
        Envelope { beams: f.beams.clone(), weights }
    }
}

impl Envelope {
    /// Σ_l'l |Q_l'l|: B in −V_prep ≤ B max_l A_l(r).
    pub fn peak(&self) -> f64 {
        self.weights.iter().sum()
    }
}
