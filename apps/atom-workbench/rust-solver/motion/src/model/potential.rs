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
//! # An averaged light (`Model::smooth`)
//! The reported trap (SPEC §1.3) is the trap light's potential with the other lights' smooth shifts: each beam's own
//! intensity, their phases averaged (`Lit::Averaged`). Such a light's shift of a ground state changes over the trap only by
//! what the trap light does to its lines -- a plane wave's own intensity is the same everywhere --, which is the movement
//! δΔ of its drives' denominators at the atom. That shift is the drive's fold with the levels at the atom, and it is there
//! wherever that fold exists, whatever the dynamics make of the drive:
//! - folded with the bare levels, the model holds none of it -- a bare fold's shift is a constant (far.rs), the first term
//!   of the series in δΔ/(Δ₀ − iΓ/2) whose whole the fold at the atom is (units.rs, # Levels), and constants cancel in
//!   h_L − h_{L∖trap}; the dynamics take that first term and book its error;
//! - explicit, the model holds it as the cluster's dynamics, no potential of a ground state -- kept because its fold does
//!   not exist, or because the solve bought its accuracy (SPEC §5.2).
//! A reported quantity is the scene's, not the solve's: the level a drive was left at must not show in it. And the reported
//! trap is a local thing, a minimum and its Hessian: what it asks of a fold is that it exists there. So the potential of a
//! light set with an averaged light is read on the same request's model with that light's co-rotating drives all folded at
//! the atom, but those whose fold does not exist at the prepared trap's centre, at rest -- the one gate on that point
//! (units.rs `Model::folds`: a ground state's drives by the tail rule), closed as the solve closes its model: the failing
//! drives explicit, until none fails. A drive on resonance there has no light shift to give (its weak-field formula would
//! be the largest term of the potential, and wrong); it stays dynamics. The couplings of two ground states stay as this
//! model keeps them: they make the clusters the shift is read in, and their folds were judged on this model's own levels
//! over the reach (a model not closed there -- a degenerate partner's coupling still folded -- has no shift to read: every
//! drive's distance holds that partner's channel). Those of two excited states follow the explicit states. The
//! counter-rotating drives are only ever bare (their δΔ against twice the optical frequency). The model is made once per
//! light set and populations.
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
use super::{jet, Component, Envelope, Gate, Level, Lights, Lit, Model, Quadratic, Trap, Unit};
use crate::linalg::C64;
use crate::field::Amplitude;
use crate::linalg::{dot3, symmetric3_eigen};
use crate::request::GROUND;
use crate::units::rounding;
use std::sync::Arc;

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

    /// The model a light set's potential is read on where a light of it is averaged (module documentation, # An averaged
    /// light): the same request's model with the averaged lights' co-rotating drives folded with the levels at the atom, those
    /// explicit whose fold does not exist at the prepared trap's centre at rest; made once per light set and populations.
    /// None where no light is averaged, an averaged one drives nothing, or no prepared trap holds the atom: this model.
    fn smooth(&self, p: &[f64; GROUND], lights: &Lights) -> Option<Arc<Model>> {
        if !lights.0.contains(&Lit::Averaged) {
            return None;
        }
        let averaged = |c: Component| !c.counter && lights.0[c.frequency] == Lit::Averaged;
        let drives: Vec<(Component, usize, usize)> = self.driven.iter().copied().filter(|d| averaged(d.0)).collect();
        if drives.is_empty() {
            return None;
        }
        let mut made = self.smooth.lock().expect("the smooth models' lock");
        if let Some(found) = made.iter().find(|of| of.0 == *lights && of.1 == *p) {
            return Some(found.2.clone());
        }
        let centre = self.trap(p, &self.prepared()).ok()?.centre;
        // What this model keeps beside the averaged lights' drives and the couplings of two excited states, with those
        // drives' bare units: each folded at the atom.
        let among_excited = |u: &Unit| matches!(u, Unit::Term { a, .. } if *a >= GROUND);
        let own: Vec<Unit> = self
            .kept
            .iter()
            .filter(|u| !among_excited(u) && !matches!(u, Unit::Bare { component, .. } | Unit::Drive { component, .. } if averaged(*component)))
            .cloned()
            .chain(drives.iter().map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }))
            .collect();
        // The averaged lights' drives whose fold does not exist at the centre: explicit, with this model's couplings of
        // an excited state they hold to another.
        let mut explicit: Vec<Unit> = vec![];
        let model = loop {
            let held = |s: usize| own.iter().chain(&explicit).any(|u| matches!(u, Unit::Drive { excited, .. } if *excited == s));
            let terms = self.kept.iter().filter(|u| matches!(u, Unit::Term { a, b, .. } if *a >= GROUND && (held(*a) || held(*b))));
            let kept: Vec<Unit> = own.iter().chain(&explicit).chain(terms).cloned().collect();
            let model = self.rekept(&kept).expect("a co-rotating drive may be folded at the atom or explicit");
            let region = model.point_region(centre, 0.0);
            let fails = |&&(c, e, g): &&(Component, usize, usize)| {
                let place = model.ranked[g].iter().find(|d| d.component == c && d.excited == e).expect("a drive is in its ground state's order").group;
                let fold = model.folds(g, place, &region, usize::MAX, 0.0);
                model.level(c, e, g) == Level::Atom && !(Gate { saturation: fold[0], sweep: fold[1] }).exists()
            };
            let failing: Vec<Unit> = drives.iter().filter(fails).map(|&(component, excited, ground)| Unit::Drive { component, excited, ground }).collect();
            if failing.is_empty() {
                break Arc::new(model);
            }
            explicit.extend(failing);
        };
        made.push((lights.clone(), *p, model.clone()));
        Some(model)
    }

    /// V_L(r; p) with, to `order`, its gradient and Hessian (what lies beyond the order is not made, and no number), and the
    /// rounding of its sum (module documentation); the near light from
    /// its tables where they serve (tabulate.rs `Model::near`). With an averaged light it is read on that light set's own model
    /// (`smooth`), exactly.
    pub(crate) fn expand(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights, order: usize, tables: Option<&super::tabulate::Tables>) -> Expansion {
        match self.smooth(p, lights) {
            Some(model) => model.expand_here(r, p, lights, order, None),
            None => self.expand_here(r, p, lights, order, tables),
        }
    }

    /// V_L(r; p) of this model as it stands (`expand`).
    fn expand_here(&self, r: [f64; 3], p: &[f64; GROUND], lights: &Lights, order: usize, tables: Option<&super::tabulate::Tables>) -> Expansion {
        let trap = self.light.trap;
        if !lights.on(trap) {
            return Expansion { quadratic: Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] }, scale: 0.0, terms: 0 };
        }
        // The lit beams' amplitudes to the order asked -- V_L and its derivatives read no other light and no further --, into
        // a thread's spare (assemble.rs `Held`): a step's reference asks the heap for nothing.
        let mut spare = Held::<Site>::take();
        let lit = |beam: usize| self.light.frequencies.iter().enumerate().any(|(w, f)| lights.on(w) && f.beams.contains(&beam));
        self.light.amplitudes_into(r, order, lit, &mut spare.beams);
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
