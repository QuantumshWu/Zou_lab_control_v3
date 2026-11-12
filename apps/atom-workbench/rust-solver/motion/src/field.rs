//! The light (SPEC §1.2, §1.3): each beam's amplitude a_l(r) relative to its focus value with its analytic gradient
//! and Hessian; one coherent field per frequency, E_ω(r) = Σ_{l∈ω} a_l(r) E_l, on the basis of its beams' span; the
//! bounded chart of each frequency's field with its gradient (the coordinates of the near light's tables, model/tabulate.rs);
//! and the trap's region by intensity. The far light's shifts, linear in T_ω = E_ω E_ω†, are the model's (model/far.rs), as
//! are the potential, the trap and the reach.
//!
//! # Beams
//! A plane wave a = e^{ikζ}; a paraxial Gaussian beam focused at the origin a = e^{ikζ} exp(−ρ²/(w₀²q))/q,
//! q = 1 + iζ/z_R, z_R = k w₀²/2, ζ = n·r, ρ² = |r|² − ζ²: the Gouy phase −atan(ζ/z_R) and the wavefront's curvature
//! are in q. Its logarithmic gradient g = ∇ln a = ik n − 2t/(w₀²q) + (i/z_R)(ρ²/(w₀²q²) − 1/q) n (t = r − ζn, the
//! transverse part) stays finite where a underflows; ∇a = a g, ∇∇a = a (g gᵀ + ∇g) with
//! ∇g = −2(1 − nnᵀ)/(w₀²q) + 2i(ntᵀ + tnᵀ)/(z_R w₀² q²) − (1/q² − 2ρ²/(w₀²q³)) nnᵀ/z_R² (symmetric).
//! |a| ≤ 1/|q| ≤ 1 everywhere: a beam is brightest at its focus. Paraxial: no longitudinal field, the polarization the
//! same across the beam (SPEC §8).
//!
//! # One field per frequency
//! The beams of one frequency are one coherent field, their relative phases the request's (SPEC §1.2, §8); the
//! couplings are linear in it: V_ω(r) = Σ_l a_l(r) X_l = Σ_q E_{ω,q}(r) D_q. The span of the beams' focus fields has
//! an orthonormal basis u_k (k < r ≤ 3, the eigenvectors of the Gram matrix G = Σ_l E_l E_l† whose eigenvalues σ_k²
//! are not zero to G's rounding, `span`), E_l = Σ_k γ_lk u_k with γ_lk = u_k†E_l, so E_ω(r) = Σ_k e_k(r) u_k, e_k = Σ_l a_l γ_lk.
//! Two fields are parallel where their span is one direction (`parallel`): the one definition of it, the rank's own.
//! Since Σ_l γ_lk* γ_lk' = σ_k² δ_kk', a quantity Y_l linear in each E_l, Y_l = Σ_k γ_lk Y_k (the couplings X_l), has
//! the basis components Y_k = Σ_l γ_lk* Y_l/σ_k² (`Frequency::dual`), and one antilinear in E_l (R_l = Σ_k γ_lk* Ỹ_k)
//! Ỹ_k = Σ_l γ_lk R_l/σ_k²: the model's check that the couplings agree with the fields is the residual of these.
//!
//! # The chart
//! Each frequency's field in bounded coordinates (SPEC §1.2; field.md §1.1): the intensity I = |E|²/N ∈ [0, 1],
//! N = (Σ_l ‖E_l‖)² (|E| ≤ Σ_l |a_l|‖E_l‖ and |a_l| ≤ 1: a computed I above 1 is its rounding, and is 1); and the
//! polarization w = E/|E| on the basis, its components in
//! the chart's order o (the pivot o₀ first, the basis vector of the largest focus field, then the others by their focus
//! fields) as hyperspherical angles α_i = atan2(√Σ_{j≥i}|e_{o_j}|², |e_{o_{i−1}}|) ∈ [0, π/2] and relative phases
//! φ_i = arg(e_{o_i} e_{o₀}*) ∈ (−π, π], i = 1 … r − 1: 2r − 1 coordinates in the order (I, α₁, φ₁, α₂, φ₂). Every
//! coordinate value is some field's, so a table on the whole ranges meets no atom at an edge; the phases are periodic.
//! The field is a trigonometric polynomial in the coordinates (`Frequency::chart_field`): w_{o₀} = cos α₁,
//! w_{o₁} = sin α₁ cos α₂ e^{iφ₁}, w_{o₂} = sin α₁ sin α₂ e^{iφ₂} (the last without its cosine), e = √(IN) w -- smooth
//! through the chart's poles, where a component vanishes and its phase is any. At a pole the chart's gradient is not
//! defined; its vanishing magnitude's terms are taken as zero there (a tabulated function's derivative along the
//! undefined coordinate vanishes with the magnitude, so the force stays bounded).
//!
//! # The region
//! An atom is outside the trap's region where every beam of the trap light -- the tweezer's frequency, its focused
//! beams: the tweezer, its reflection, a crossed beam of the same laser -- is below `REGION_INTENSITY` of its focus
//! value, |a_l(r)|² < 1/20 (SPEC §1.3, the UI's definition of loss). Only the trap light makes the trap: a probe or a
//! cooling beam given by its waist is focused too, on the millimetre scale, and an atom it pushes out of the tweezer
//! moves along it for metres (√19 z_R) -- counted by its intensity, such an atom would never be lost.

use crate::linalg::{dot3, hermitian_eigen, Mat, C64, V3};
use crate::units::rounding;
use crate::request::Request;

/// A beam's geometry: its unit direction n (lab axes), its wave number k (rad/µm), its waist w₀ (µm; None: a plane
/// wave).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Geometry {
    pub direction: V3,
    pub wave_number: f64,
    pub waist: Option<f64>,
}

/// A beam's amplitude at a point, relative to its focus value, and its analytic gradient (1/µm) and Hessian (1/µm²).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Amplitude {
    pub value: C64,
    pub gradient: [C64; 3],
    pub hessian: [[C64; 3]; 3],
}

impl Geometry {
    /// The request beam's geometry, its direction normalized.
    pub fn new(direction: V3, wave_number: f64, waist: Option<f64>) -> Geometry {
        let l = crate::linalg::norm3(direction);
        Geometry { direction: direction.map(|x| x / l), wave_number, waist }
    }
    /// k n (rad/µm).
    pub fn wave_vector(&self) -> V3 {
        self.direction.map(|x| x * self.wave_number)
    }
    /// The Rayleigh range z_R = k w₀²/2 of a Gaussian beam (µm).
    pub fn rayleigh(&self) -> Option<f64> {
        self.waist.map(|w| self.wave_number * w * w / 2.0)
    }
    /// a(r), ∇a and ∇∇a (module documentation, # Beams): the transverse part t = r − ζn taken directly, so ρ² = |t|²
    /// is never a difference of nearly equal numbers.
    pub fn amplitude(&self, r: V3) -> Amplitude {
        self.amplitude_to(r, 2)
    }
    /// The same to an order, as its caller reads it: the value alone (0), with ∇a (1), with ∇∇a too (2). What lies
    /// beyond the order is zero and is not made; what lies within it is the same numbers at every order.
    pub fn amplitude_to(&self, r: V3, order: usize) -> Amplitude {
        let (k, n) = (self.wave_number, self.direction);
        let zeta = dot3(r, n);
        let wave = C64::cis(k * zeta);
        let ik = C64::new(0.0, k);
        let none = ([C64::ZERO; 3], [[C64::ZERO; 3]; 3]);
        let (value, (g, dg)): (C64, ([C64; 3], [[C64; 3]; 3])) = match self.waist {
            None => (wave, if order >= 1 { (n.map(|x| ik * x), none.1) } else { none }),
            Some(w) => {
                let (w2, rayleigh) = (w * w, k * w * w / 2.0);
                let inverse = C64::new(1.0, zeta / rayleigh).recip();
                let t: V3 = std::array::from_fn(|j| r[j] - zeta * n[j]);
                let rho2 = dot3(t, t);
                let value = wave * (inverse * (-rho2 / w2)).exp() * inverse;
                let across = inverse * (-2.0 / w2);
                // g = (ik + (i/z_R)(ρ²/(w²q²) − 1/q)) n − (2/(w²q)) t.
                let g = || {
                    let along = ik + C64::I * (inverse * inverse * (rho2 / w2) - inverse) / rayleigh;
                    std::array::from_fn(|j| along * n[j] + across * t[j])
                };
                // ∇g = c₁(1 − nnᵀ) + c₂(ntᵀ + tnᵀ) + c₃ nnᵀ.
                let dg = || {
                    let c1 = across;
                    let c2 = C64::I * inverse * inverse * (2.0 / (rayleigh * w2));
                    let c3 = (inverse * inverse * inverse * (2.0 * rho2 / w2) - inverse * inverse) / (rayleigh * rayleigh);
                    std::array::from_fn(|p| {
                        std::array::from_fn(|q| {
                            let unit = if p == q { 1.0 } else { 0.0 };
                            c1 * (unit - n[p] * n[q]) + c2 * (n[p] * t[q] + t[p] * n[q]) + c3 * (n[p] * n[q])
                        })
                    })
                };
                (value, match order { 0 => none, 1 => (g(), none.1), _ => (g(), dg()) })
            }
        };
        Amplitude {
            value,
            gradient: if order >= 1 { g.map(|x| value * x) } else { none.0 },
            hessian: if order >= 2 { std::array::from_fn(|p| std::array::from_fn(|q| value * (g[p] * g[q] + dg[p][q]))) } else { none.1 },
        }
    }
}

/// One frequency's coherent field at a point: E (V/m, spherical components q = −1, 0, +1) and ∇E (`gradient[a][q]` =
/// ∂_a E_q).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldAt {
    pub value: [C64; 3],
    pub gradient: [[C64; 3]; 3],
}

/// A frequency's chart at a point (module documentation, # The chart): the 2r − 1 coordinates (I, α₁, φ₁, α₂, φ₂)
/// and their gradients (`gradient[c][a]` = ∂_a s_c), held in place: those beyond a field's 2r − 1 are zero.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Chart {
    pub value: [f64; 5],
    pub gradient: [[f64; 3]; 5],
}

/// The range of the chart's coordinate c and whether it is periodic: the intensity [0, 1], an angle [0, π/2], a
/// phase [−π, π) (periodic).
pub fn range(c: usize) -> (f64, f64, bool) {
    use std::f64::consts::{FRAC_PI_2, PI};
    if c == 0 {
        (0.0, 1.0, false)
    } else if c % 2 == 1 {
        (0.0, FRAC_PI_2, false)
    } else {
        (-PI, PI, true)
    }
}

/// One frequency's field: its beams, their focus fields, the basis of their span and the chart's data.
#[derive(Clone, Debug, PartialEq)]
pub struct Frequency {
    /// The frequency's coordinates on the request's frequency axes.
    pub index: Vec<i32>,
    /// Its beams (indices into the request's beams).
    pub beams: Vec<usize>,
    /// Its beams' focus fields E_l (V/m), in `beams` order.
    pub fields: Vec<[C64; 3]>,
    /// Its outside form (an index into the request's outside forms).
    pub form: usize,
    /// N = (Σ_l ‖E_l‖)² ((V/m)²): the chart's intensity scale.
    pub norm: f64,
    /// The orthonormal basis u_k of the span of the focus fields (spherical components), k < r ≤ 3, by decreasing σ_k.
    pub basis: Vec<[C64; 3]>,
    /// σ_k², the Gram matrix's eigenvalue of u_k ((V/m)²).
    pub gram: Vec<f64>,
    /// γ_lk = u_k†E_l: per beam (in `beams` order) its components on the basis.
    pub coefficients: Vec<Vec<C64>>,
    /// The chart's component order o: the pivot first (the largest focus component |Σ_l γ_lk|), then by decreasing
    /// focus component.
    pub order: Vec<usize>,
}

/// The span of fields (module documentation, # One field per frequency): the eigenvectors u_k of G = Σ_l E_l E_l† whose
/// eigenvalues σ_k² are not zero to G's rounding, by decreasing σ_k, with their σ_k². The bound of a zero eigenvalue's computed
/// value: the computed decomposition G U = U Λ + R gives every λ_k within ‖R‖_F of an eigenvalue of the computed G (U unitary
/// to its rounding; Weyl), whatever rotations made it, and the computed G lies within γ_n tr G of the exact one (its entries
/// inner products of the n fields, ‖|G|‖_F ≤ tr G), R's own computation within γ₆ tr G (a row of three products, λ_k u_k and
/// their difference, ‖G‖_F ≤ tr G; U's unitarity): λ_k ≤ ‖R‖_F + γ_{n+6} tr G is indistinguishable from zero, and the field it
/// leaves off is at most the square root of that.
fn span(fields: &[[C64; 3]]) -> (Vec<[C64; 3]>, Vec<f64>) {
    let g = Mat::from_fn(3, 3, |p, q| fields.iter().map(|e| e[p] * e[q].conj()).sum());
    let e = hermitian_eigen(&g);
    let mut residual = g.product(&e.vectors);
    for k in 0..3 {
        for q in 0..3 {
            residual[(q, k)] -= e.vectors[(q, k)] * e.values[k];
        }
    }
    let floor = residual.frobenius() + rounding(fields.len() + 6) * g.trace().re;
    let kept: Vec<usize> = (0..3).rev().filter(|&k| e.values[k] > floor).collect();
    (kept.iter().map(|&k| std::array::from_fn(|q| e.vectors[(q, k)])).collect(), kept.iter().map(|&k| e.values[k]).collect())
}

/// Whether two fields are parallel: their span is one direction (`span`).
pub fn parallel(first: &[C64; 3], second: &[C64; 3]) -> bool {
    span(&[*first, *second]).0.len() == 1
}

impl Frequency {
    /// The field of the beams `beams` with focus fields `fields`, on the basis of their span (`span`).
    fn new(index: Vec<i32>, beams: Vec<usize>, fields: Vec<[C64; 3]>, form: usize) -> Frequency {
        let norm = fields.iter().map(|e| e.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()).sum::<f64>().powi(2);
        let (basis, gram) = span(&fields);
        let coefficients: Vec<Vec<C64>> =
            fields.iter().map(|f| basis.iter().map(|u| (0..3).map(|q| u[q].conj() * f[q]).sum()).collect()).collect();
        let focus: Vec<f64> = (0..basis.len()).map(|k| coefficients.iter().map(|c| c[k]).sum::<C64>().abs()).collect();
        let mut order: Vec<usize> = (0..basis.len()).collect();
        order.sort_by(|&a, &b| focus[b].total_cmp(&focus[a]));
        Frequency { index, beams, fields, form, norm, basis, gram, coefficients, order }
    }
    /// r, the dimension of the span of its beams' fields.
    pub fn rank(&self) -> usize {
        self.basis.len()
    }
    /// The chart's dimension 2r − 1.
    pub fn chart_dimension(&self) -> usize {
        2 * self.rank() - 1
    }
    /// The weight of beam j (in `beams` order) in basis component k of a quantity linear in the beams' fields:
    /// γ_jk*/σ_k² (module documentation, # One field per frequency).
    pub fn dual(&self, j: usize, k: usize) -> C64 {
        self.coefficients[j][k].conj() / self.gram[k]
    }
    /// Its field at a point from all beams' amplitudes there (indexed as the request's beams).
    pub fn field(&self, amplitudes: &[Amplitude]) -> FieldAt {
        let mut f = FieldAt { value: [C64::ZERO; 3], gradient: [[C64::ZERO; 3]; 3] };
        for (&l, e) in self.beams.iter().zip(&self.fields) {
            let a = &amplitudes[l];
            for q in 0..3 {
                f.value[q] += a.value * e[q];
                for x in 0..3 {
                    f.gradient[x][q] += a.gradient[x] * e[q];
                }
            }
        }
        f
    }
    /// Its field's components on the basis, e_k = u_k†E, and their gradients (`[k][a]` = ∂_a e_k).
    pub fn components(&self, field: &FieldAt) -> ([C64; 3], [[C64; 3]; 3]) {
        let project = |v: &[C64; 3], u: &[C64; 3]| (0..3).map(|q| u[q].conj() * v[q]).sum::<C64>();
        let of = |k: usize| self.basis.get(k);
        (
            std::array::from_fn(|k| of(k).map_or(C64::ZERO, |u| project(&field.value, u))),
            std::array::from_fn(|k| of(k).map_or([C64::ZERO; 3], |u| std::array::from_fn(|a| project(&field.gradient[a], u)))),
        )
    }
    /// Its chart at a point, from its field there (module documentation, # The chart): I = min(Σ_k |e_k|²/N, 1),
    /// ∇I = Σ_k 2 Re(e_k* ∇e_k)/N;
    /// α = atan2(B, A) with A = |e_{o_{i−1}}|, B = √Σ_{j≥i}|e_{o_j}|², ∇α = (A∇B − B∇A)/(A² + B²), ∇A = Re(e*∇e)/A,
    /// ∇B = Σ_{j≥i} Re(e*∇e)/B; ∇φ_i = Im(e_{o_i}*∇e_{o_i})/|e_{o_i}|² − Im(e_{o₀}*∇e_{o₀})/|e_{o₀}|² -- a vanishing
    /// magnitude's terms zero.
    pub fn chart(&self, field: &FieldAt) -> Chart {
        let (e, de) = self.components(field);
        let (r, o, n) = (self.rank(), &self.order, self.norm);
        let size: [f64; 3] = e.map(|z| z.norm_sqr());
        // Re and Im of e_k* ∂_a e_k.
        let dot: [[f64; 3]; 3] = std::array::from_fn(|k| std::array::from_fn(|a| (e[k].conj() * de[k][a]).re));
        let turn: [[f64; 3]; 3] = std::array::from_fn(|k| std::array::from_fn(|a| (e[k].conj() * de[k][a]).im));
        let over = |x: f64, y: f64| if y > 0.0 { x / y } else { 0.0 };
        let mut chart = Chart { value: [0.0; 5], gradient: [[0.0; 3]; 5] };
        chart.value[0] = (size[..r].iter().sum::<f64>() / n).min(1.0);
        chart.gradient[0] = std::array::from_fn(|a| (0..r).map(|k| 2.0 * dot[k][a]).sum::<f64>() / n);
        for i in 1..r {
            let (first, rest) = (size[o[i - 1]].sqrt(), o[i..].iter().map(|&j| size[j]).sum::<f64>().sqrt());
            chart.value[2 * i - 1] = rest.atan2(first);
            chart.gradient[2 * i - 1] = std::array::from_fn(|a| {
                let d_rest = over(o[i..].iter().map(|&j| dot[j][a]).sum::<f64>(), rest);
                let d_first = over(dot[o[i - 1]][a], first);
                over(first * d_rest - rest * d_first, first * first + rest * rest)
            });
            let z = e[o[i]] * e[o[0]].conj();
            chart.value[2 * i] = if z == C64::ZERO { 0.0 } else { z.arg() };
            chart.gradient[2 * i] = std::array::from_fn(|a| over(turn[o[i]][a], size[o[i]]) - over(turn[o[0]][a], size[o[0]]));
        }
        chart
    }
    /// The field E (V/m, spherical components) at chart coordinates s: e = √(IN) w, w_{o₀} = cos α₁,
    /// w_{o_i} = sin α₁ ⋯ sin α_i cos α_{i+1} e^{iφ_i} (the last without its cosine), E = Σ_k e_k u_k -- a table node's
    /// field.
    pub fn chart_field(&self, s: &[f64]) -> [C64; 3] {
        let (r, o) = (self.rank(), &self.order);
        let size = (s[0] * self.norm).sqrt();
        let mut w = vec![C64::ZERO; r];
        let mut sine = 1.0;
        for i in 0..r {
            let (sin, cos) = if i + 1 < r { s[2 * i + 1].sin_cos() } else { (0.0, 1.0) };
            let phase = if i == 0 { C64::ONE } else { C64::cis(s[2 * i]) };
            w[o[i]] = phase * (size * sine * cos);
            sine *= sin;
        }
        std::array::from_fn(|q| self.basis.iter().zip(&w).map(|(u, x)| *x * u[q]).sum())
    }
}

/// The light of a request: its beams' geometries and one coherent field per frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    /// Per request beam its geometry.
    pub beams: Vec<Geometry>,
    /// Per distinct frequency of the beams its field, in the order of their first beams.
    pub frequencies: Vec<Frequency>,
    /// The trap's frequency (an index into `frequencies`): the one of the tweezer, whose whole field is the trap light
    /// (SPEC §1.3).
    pub trap: usize,
}

impl Light {
    /// The light of a validated request (`Request::validate`): the beams of one `frequency_index` are one field.
    pub fn new(request: &Request) -> Light {
        let beams = request.beams.iter().map(|b| Geometry::new(b.direction, b.wave_number, b.waist)).collect();
        let mut groups: Vec<(Vec<i32>, Vec<usize>)> = vec![];
        for (l, b) in request.beams.iter().enumerate() {
            match groups.iter_mut().find(|(index, _)| *index == b.frequency_index) {
                Some((_, members)) => members.push(l),
                None => groups.push((b.frequency_index.clone(), vec![l])),
            }
        }
        let trap = groups.iter().position(|(_, members)| members.contains(&request.trap_beam)).expect("the trap beam is a beam");
        let frequencies = groups
            .into_iter()
            .map(|(index, members)| {
                let form = request.outside_forms.iter().position(|f| f.frequency_index == index).expect("a validated request has every frequency's form");
                let fields = members.iter().map(|&l| request.beams[l].field).collect();
                Frequency::new(index, members, fields, form)
            })
            .collect();
        Light { beams, frequencies, trap }
    }
    /// Every beam's amplitude at r (µm).
    pub fn amplitudes(&self, r: V3) -> Vec<Amplitude> {
        self.beams.iter().map(|b| b.amplitude(r)).collect()
    }
    /// The beams' amplitudes at r as a caller reads them, into its vector (kept from one point to the next): of the beams
    /// `of` takes to `order` (`Geometry::amplitude_to`), zero of the others.
    pub fn amplitudes_into(&self, r: V3, order: usize, of: impl Fn(usize) -> bool, out: &mut Vec<Amplitude>) {
        let dark = Amplitude { value: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] };
        out.clear();
        out.extend(self.beams.iter().enumerate().map(|(l, b)| if of(l) { b.amplitude_to(r, order) } else { dark }));
    }
    /// Whether the point r is outside the trap's region: every beam of the trap light there below `REGION_INTENSITY` of
    /// its focus value (module documentation, # The region; the trap light holds no plane wave, `Request::validate`).
    pub fn outside(&self, r: V3) -> bool {
        self.frequencies[self.trap].beams.iter().all(|&l| self.beams[l].amplitude_to(r, 0).value.norm_sqr() < crate::units::REGION_INTENSITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::norm3;
    use crate::random::{Purpose, Stream};
    use crate::request::{tests::sample, OutsideForm};
    use crate::units::{REGION_INTENSITY, UNIT_ROUNDOFF};
    use std::f64::consts::PI;

    /// A request's light with these beams (direction, waist, frequency index, focus field), the first the tweezer, each
    /// frequency with its (empty) outside form.
    fn light_of(beams: &[(V3, Option<f64>, i32, [C64; 3])]) -> Light {
        let mut r = sample();
        let template = r.beams[0].clone();
        r.beams = beams
            .iter()
            .map(|&(direction, waist, index, field)| crate::request::Beam { direction, waist, frequency_index: vec![index], field, ..template.clone() })
            .collect();
        let mut indices: Vec<i32> = beams.iter().map(|b| b.2).collect();
        indices.sort();
        indices.dedup();
        r.outside_forms = indices.iter().map(|&i| OutsideForm { frequency_index: vec![i], ..r.outside_forms[0].clone() }).collect();
        r.validate().unwrap();
        Light::new(&r)
    }
    /// Points spread over a few waists and Rayleigh ranges about the focus.
    fn points(count: usize, seed: u64) -> Vec<V3> {
        let mut s = Stream::new(seed, Purpose::Initial, 0, 0);
        (0..count).map(|_| [1.6 * (s.uniform() - 0.5), 1.6 * (s.uniform() - 0.5), 6.0 * (s.uniform() - 0.5)]).collect()
    }
    /// The central difference of f along axis a at r with step h.
    fn central<T: Copy + std::ops::Sub<Output = T> + std::ops::Mul<f64, Output = T>>(f: &dyn Fn(V3) -> T, r: V3, a: usize, h: f64) -> T {
        let (mut plus, mut minus) = (r, r);
        plus[a] += h;
        minus[a] -= h;
        (f(plus) - f(minus)) * (0.5 / h)
    }
    /// The largest logarithmic-derivative scale of a beam near r: |∇ln a| ≤ k + 2ρ/w₀² + (1 + ρ²/w₀²)/z_R, plus the
    /// curvature 2/w₀ -- the K of the difference checks.
    fn scale(beam: &Geometry, r: V3) -> f64 {
        let n = beam.direction;
        let t: V3 = std::array::from_fn(|j| r[j] - dot3(r, n) * n[j]);
        let rho = norm3(t);
        beam.wave_number + beam.waist.map_or(0.0, |w| 2.0 / w + 2.0 * rho / (w * w) + (1.0 + rho * rho / (w * w)) / beam.rayleigh().unwrap())
    }

    /// The relative rounding of the amplitudes near r: their phases k n·r and exponents ρ²/w₀² carry absolute errors of
    /// a few u times their size (as the closed forms' check reads them), γ₄(1 + k|r| + ρ²/w₀²).
    fn own(beams: &[Geometry], r: V3) -> f64 {
        beams
            .iter()
            .map(|b| {
                let t: V3 = std::array::from_fn(|j| r[j] - dot3(r, b.direction) * b.direction[j]);
                rounding(4) * (1.0 + b.wave_number * norm3(r) + b.waist.map_or(0.0, |w| dot3(t, t) / (w * w)))
            })
            .fold(0.0, f64::max)
    }
    /// The central difference's error at step h, relative to the function's size: its truncation (h²/6)K³ for a
    /// function whose third derivative is within K³ of it, and its rounding (own)/h, doubled for the two ends.
    fn difference_error(k: f64, h: f64, own: f64) -> f64 {
        2.0 * (h * h / 6.0 * k.powi(3) + own / h)
    }

    /// The paraxial Gaussian beam's closed forms (module documentation, # Beams): on its axis e^{i(kz − atan(z/z_R))}
    /// /√(1 + z²/z_R²) (the Gouy phase); across its focus the real exp(−ρ²/w₀²); off its axis the modulus
    /// (w₀/w(z)) exp(−ρ²/w(z)²), w(z)² = w₀²(1 + z²/z_R²), and the phase kz − atan(z/z_R) + kρ²/(2R(z)),
    /// R(z) = z(1 + z_R²/z²) (the wavefront's curvature); a plane wave e^{ik n·r}. Each to its phase's rounding,
    /// γ (1 + k|r|).
    #[test]
    fn a_beam_s_amplitude_is_its_closed_form() {
        let (k, w) = (7.37, 0.8);
        let zr = k * w * w / 2.0;
        let beam = Geometry::new([0.0, 0.0, 2.0], k, Some(w));
        let close = |r: V3, expected: C64| {
            let a = beam.amplitude(r).value;
            assert!((a - expected).abs() <= rounding(16) * (1.0 + k * norm3(r)), "{r:?}: {a:?} against {expected:?}");
        };
        for z in [-3.0, -0.4, 0.0, 0.9, 5.0] {
            close([0.0, 0.0, z], C64::cis(k * z - (z / zr).atan()) / (1.0 + (z / zr).powi(2)).sqrt());
        }
        close([0.6, 0.0, 0.0], C64::real((-0.36 / (w * w)).exp()));
        for [x, y, z] in [[0.5, -0.3, 1.7], [0.2, 0.4, -2.5]] {
            let rho2 = x * x + y * y;
            let w2 = w * w * (1.0 + (z / zr).powi(2));
            let curvature = z * (1.0 + (zr / z).powi(2));
            let modulus = (w * w / w2).sqrt() * (-rho2 / w2).exp();
            close([x, y, z], C64::cis(k * z - (z / zr).atan() + k * rho2 / (2.0 * curvature)) * modulus);
        }
        let plane = Geometry::new([0.6, 0.0, 0.8], k, None);
        let r = [0.3, -1.0, 0.83];
        assert!((plane.amplitude(r).value - C64::cis(k * (0.6 * 0.3 + 0.8 * 0.83))).abs() <= rounding(16) * (1.0 + k * norm3(r)));
    }

    /// ∇a and ∇∇a against central differences of a and of the analytic ∇a, a check of the closed forms: at the step
    /// h = u^{1/3}/K (K of `scale`) within `difference_error` of |a| (K|a| for the Hessian).
    #[test]
    fn a_beam_s_derivatives_are_its_amplitude_s() {
        let beams = [Geometry::new([0.0, 0.6, 0.8], 7.37, Some(0.8)), Geometry::new([1.0, 1.0, 0.0], 8.05, Some(1.3)), Geometry::new([0.0, 0.0, -1.0], 7.9, None)];
        let third = UNIT_ROUNDOFF.cbrt();
        for beam in &beams {
            for r in [[0.0, 0.0, 0.0], [0.3, -0.2, 0.7], [-0.9, 0.4, -1.8], [1.6, 0.0, 0.2]] {
                let (a, k) = (beam.amplitude(r), scale(beam, r));
                // To an order (`amplitude_to`): the same numbers within it, zero beyond it.
                let (zero, lower) = (Amplitude { value: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] }, [beam.amplitude_to(r, 0), beam.amplitude_to(r, 1)]);
                assert!(lower[0] == Amplitude { value: a.value, ..zero } && lower[1] == Amplitude { hessian: zero.hessian, ..a } && beam.amplitude_to(r, 2) == a, "{beam:?} {r:?}: the amplitude to an order");
                let h = third / k;
                let within = difference_error(k, h, own(&[*beam], r)) * a.value.abs();
                for x in 0..3 {
                    let d = central(&|p| beam.amplitude(p).value, r, x, h);
                    assert!((d - a.gradient[x]).abs() <= within, "{beam:?} {r:?} ∂{x}: {d:?} {:?} within {within:e}", a.gradient[x]);
                    for y in 0..3 {
                        let d = central(&|p| beam.amplitude(p).gradient[y], r, x, h);
                        assert!((d - a.hessian[x][y]).abs() <= k * within, "{beam:?} {r:?} ∂{x}∂{y}");
                    }
                }
            }
        }
    }

    /// The span's basis: parallel fields have rank 1 (also when their difference is below the Gram matrix's rounding),
    /// two independent ones rank 2, three rank 3; the basis is orthonormal, Σ_k γ_lk u_k gives each field back, and the
    /// dual weights invert the coefficients, Σ_j dual(j, k) γ_jk' = δ_kk'. Two fields are parallel exactly where their
    /// span has rank 1 (`parallel`), whatever their sizes and phases.
    #[test]
    fn the_basis_spans_the_fields() {
        let field = |e: [f64; 3], phase: f64| e.map(|x| C64::cis(phase) * x);
        // Each case: the fields, their rank, and the part of them off the span (the component below the rounding).
        let cases = [
            (vec![field([0.0, 1e5, 0.0], 0.0), field([0.0, 2e5, 0.0], 1.0)], 1, 0.0),
            (vec![field([1e5, 0.0, 0.0], 0.0), field([1e5, 1e-12, 0.0], 0.3)], 1, 1e-12),
            (vec![field([1e3, 0.0, 0.0], 0.0), field([0.0, 1e3, 0.0], 2.0), field([1e3, 1e3, 0.0], -1.0)], 2, 0.0),
            (vec![field([1e3, 0.0, 0.0], 0.0), field([0.0, 1e3, 0.0], 0.5), field([3.0, 0.0, 7e2], 0.1)], 3, 0.0),
        ];
        for (fields, rank, off) in cases {
            let beams: Vec<(V3, Option<f64>, i32, [C64; 3])> = fields.iter().map(|&e| ([0.0, 0.0, 1.0], Some(0.8), -4, e)).collect();
            let light = light_of(&beams);
            let f = &light.frequencies[0];
            assert_eq!(f.rank(), rank, "{fields:?}");
            if fields.len() == 2 {
                assert_eq!(parallel(&fields[0], &fields[1]), rank == 1, "{fields:?}");
            }
            let largest = fields.iter().map(|e| e.iter().map(|z| z.abs()).fold(0.0, f64::max)).fold(0.0, f64::max);
            for (j, e) in fields.iter().enumerate() {
                for q in 0..3 {
                    let back: C64 = (0..rank).map(|k| f.coefficients[j][k] * f.basis[k][q]).sum();
                    assert!((back - e[q]).abs() <= rounding(16) * largest + off, "{fields:?}");
                }
            }
            for k in 0..rank {
                for m in 0..rank {
                    let unit: C64 = (0..3).map(|q| f.basis[k][q].conj() * f.basis[m][q]).sum();
                    let dual: C64 = (0..fields.len()).map(|j| f.dual(j, k) * f.coefficients[j][m]).sum();
                    let delta = if k == m { 1.0 } else { 0.0 };
                    assert!((unit - C64::real(delta)).abs() <= rounding(16), "{fields:?}");
                    assert!((dual - C64::real(delta)).abs() <= rounding(64), "{fields:?}: {dual:?}");
                }
            }
        }
    }

    /// The chart (module documentation, # The chart) of a rank-3 field of three beams: its coordinates lie in their
    /// ranges; the field it maps back to has the same coherence T = E E† (the chart drops only the global phase), to the
    /// rounding of N; its gradient agrees with central differences (the phases' wrapped) where no component is small.
    /// At a pole the field it maps to does not depend on the undefined coordinates: α₁ = 0 leaves T the same for every
    /// φ₁, α₂, φ₂.
    #[test]
    fn the_chart_is_bounded_and_gives_its_field_back() {
        let light = light_of(&[
            ([0.0, 0.0, 1.0], Some(0.8), -4, [C64::new(2e4, 1e4), C64::real(9e4), C64::ZERO]),
            ([0.0, 0.0, -1.0], Some(0.8), -4, [C64::ZERO, C64::new(-3e4, 5e3), C64::new(0.0, 4e4)]),
            ([0.0, 1.0, 0.0], Some(1.5), -4, [C64::real(3e4), C64::ZERO, C64::new(1e4, -2e4)]),
        ]);
        let f = &light.frequencies[0];
        assert_eq!((f.rank(), f.chart_dimension()), (3, 5));
        let coherence = |e: &[C64; 3]| -> Vec<C64> { (0..9).map(|i| e[i / 3] * e[i % 3].conj()).collect() };
        let third = UNIT_ROUNDOFF.cbrt();
        for r in points(40, 3) {
            let field = f.field(&light.amplitudes(r));
            let chart = f.chart(&field);
            for (c, &s) in chart.value.iter().enumerate() {
                let (lo, hi, _) = range(c);
                assert!(lo <= s && s <= hi, "{r:?} s{c} = {s}");
            }
            let (t, back) = (coherence(&field.value), coherence(&f.chart_field(&chart.value)));
            for i in 0..9 {
                assert!((t[i] - back[i]).abs() <= rounding(32) * f.norm, "{r:?} T{i}: {:?} {:?}", t[i], back[i]);
            }
            // The chart's coordinates vary on the length 1/(K/ε), ε = min_k |e_k|/|E| the smallest relative component, and
            // carry the amplitudes' rounding over ε: `difference_error` at K/ε and own/ε.
            let (e, _) = f.components(&field);
            let total = e.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            let smallest = e.iter().map(|z| z.abs() / total).fold(1.0, f64::min);
            let k = light.beams.iter().map(|b| scale(b, r)).fold(0.0, f64::max);
            let h = third / k;
            for a in 0..3 {
                let (mut plus, mut minus) = (r, r);
                plus[a] += h;
                minus[a] -= h;
                let (p, m) = (f.chart(&f.field(&light.amplitudes(plus))).value, f.chart(&f.field(&light.amplitudes(minus))).value);
                for c in 0..5 {
                    let mut d = p[c] - m[c];
                    if range(c).2 {
                        d -= 2.0 * PI * (d / (2.0 * PI)).round();
                    }
                    let d = d / (2.0 * h);
                    let within = difference_error(k / smallest, h, own(&light.beams, r) / smallest);
                    assert!((d - chart.gradient[c][a]).abs() <= within, "{r:?} ∂{a} s{c}: {d:e} {:e} within {within:e}", chart.gradient[c][a]);
                }
            }
        }
        let at_pole = coherence(&f.chart_field(&[0.4, 0.0, 1.0, 0.3, -2.0]));
        for s in [[0.4, 0.0, -2.5, 1.2, 0.7], [0.4, 0.0, 0.0, 0.0, 0.0]] {
            let other = coherence(&f.chart_field(&s));
            assert!((0..9).all(|i| (other[i] - at_pole[i]).abs() <= rounding(32) * f.norm), "{s:?}");
        }
    }

    /// The region (SPEC §1.3): outside where every focused beam's intensity is below 1/20 of its focus value -- on the
    /// tweezer's axis beyond z = z_R√19, across it beyond ρ = w₀√(ln 20/2); a plane wave, bright everywhere, holds no
    /// atom in; a second focused beam does.
    #[test]
    fn the_region_is_where_a_focused_beam_is_bright() {
        let (k, w) = (7.37, 0.8);
        let zr = k * w * w / 2.0;
        let e = [C64::ZERO, C64::real(1e5), C64::ZERO];
        let probe = ([1.0, 0.0, 0.0], None, 0, [C64::real(10.0), C64::ZERO, C64::ZERO]);
        let light = light_of(&[([0.0, 0.0, 1.0], Some(w), -4, e), probe]);
        let outside = |r: V3| light.outside(r);
        let (axial, radial) = (zr * (1.0 / REGION_INTENSITY - 1.0).sqrt(), w * ((1.0 / REGION_INTENSITY).ln() / 2.0).sqrt());
        let margin = 1.0 + 1e-9;
        assert!(!outside([0.0, 0.0, axial / margin]) && outside([0.0, 0.0, axial * margin]));
        assert!(!outside([radial / margin, 0.0, 0.0]) && outside([0.0, radial * margin, 0.0]));
        // A crossed beam of the trap's laser widens the region; a probe focused on the millimetre scale, of another
        // frequency, does not (the cut-over's finding: an atom it pushed out of the tweezer was never lost).
        let crossed = light_of(&[([0.0, 0.0, 1.0], Some(w), -4, e), ([1.0, 0.0, 0.0], Some(w), -4, e), probe]);
        assert!(!crossed.outside([3.0, 0.0, 0.0]) && crossed.outside([0.0, 3.0, 0.0]));
        let wide = ([1.0, 0.0, 0.0], Some(1000.0), 0, [C64::real(10.0), C64::ZERO, C64::ZERO]);
        let probed = light_of(&[([0.0, 0.0, 1.0], Some(w), -4, e), wide]);
        assert!(probed.outside([0.0, 0.0, axial * margin]) && probed.outside([0.0, 0.0, 1e4]));
    }
}
