//! The light (SPEC §1.2, §1.3): each beam's amplitude a_l(r) relative to its focus value with its analytic gradient
//! and Hessian; one coherent field per frequency, E_ω(r) = Σ_{l∈ω} a_l(r) E_l; and the trap's region by intensity. The light's model is the
//! scene's (scene.rs: the couplings and the outside forms at the beams' amplitudes) and its one fold's (model/rs.rs).
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
//! couplings are linear in it: V_ω(r) = Σ_l a_l(r) X_l = Σ_q E_{ω,q}(r) D_q. (Until 2026-11-22 each frequency's field
//! was also given on the orthonormal basis of its beams' span and in a bounded chart -- intensity, hyperspherical angles,
//! relative phases -- for tables over the field's coordinates; nothing read them.)
//!
//! # The region
//! An atom is outside the trap's region where the trap light -- the tweezer, the one beam of its frequency
//! (`Request::validate`: neither its reflection nor another beam of its laser is input) -- is below `REGION_INTENSITY` of
//! its focus value, |a(r)|² < 1/20 (SPEC §1.3): the descent to a trap's centre stays within it (model/potential.rs
//! `descend`). It is no part of the loss (SPEC §1.7: the loss energy above its far value, wherever the atom is). Only
//! the trap light makes the trap: a probe or a cooling beam given by its waist is focused too, on the millimetre scale,
//! and a region by its intensity would reach metres along it (√19 z_R).

use crate::linalg::{dot3, C64, V3};
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

/// One frequency's field: its beams and their focus fields.
#[derive(Clone, Debug, PartialEq)]
pub struct Frequency {
    /// Its beams (indices into the request's beams).
    pub beams: Vec<usize>,
    /// Its beams' focus fields E_l (V/m), in `beams` order.
    pub fields: Vec<[C64; 3]>,
}

/// The light of a request: its beams' geometries and one coherent field per frequency.
#[derive(Clone, Debug, PartialEq)]
pub struct Light {
    /// Per request beam its geometry.
    pub beams: Vec<Geometry>,
    /// Per distinct frequency of the beams its field, in the order of their first beams.
    pub frequencies: Vec<Frequency>,
    /// The trap's frequency (an index into `frequencies`): the one of the tweezer, whose field is the trap light -- the
    /// tweezer alone, the one beam of its frequency (SPEC §1.3, `Request::validate`).
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
            .map(|(_, members)| Frequency { fields: members.iter().map(|&l| request.beams[l].field).collect(), beams: members })
            .collect();
        Light { beams, frequencies, trap }
    }
    /// Whether the point r is outside the trap's region: the trap light there below `REGION_INTENSITY` of its focus value
    /// (module documentation, # The region; the trap light is the tweezer, a Gaussian beam, `Request::validate`).
    pub fn outside(&self, r: V3) -> bool {
        self.frequencies[self.trap].beams.iter().all(|&l| self.beams[l].amplitude_to(r, 0).value.norm_sqr() < crate::units::REGION_INTENSITY)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::larger;
    use crate::linalg::norm3;
    use crate::request::{tests::sample, OutsideForm};
    use crate::units::{rounding, REGION_INTENSITY, UNIT_ROUNDOFF};

    /// A request's light with these beams (direction, waist, frequency index, focus field): the first the tweezer (the
    /// sample's, 852 nm), the others light beside it on D2 (the sample's probe), each frequency with its (empty) outside
    /// form. A valid request: no other beam at the tweezer's frequency (`Request::validate`).
    fn light_of(beams: &[(V3, Option<f64>, i32, [C64; 3])]) -> Light {
        let mut r = sample();
        let (tweezer, other) = (r.beams[0].clone(), r.beams[1].clone());
        r.beams = beams
            .iter()
            .enumerate()
            .map(|(l, &(direction, waist, index, field))| crate::request::Beam { direction, waist, frequency_index: vec![index], field, ..if l == 0 { tweezer.clone() } else { other.clone() } })
            .collect();
        let mut indices: Vec<i32> = beams.iter().map(|b| b.2).collect();
        indices.sort();
        indices.dedup();
        r.outside_forms = indices.iter().map(|&i| OutsideForm { frequency_index: vec![i], ..r.outside_forms[0].clone() }).collect();
        r.validate().unwrap();
        Light::new(&r)
    }
    /// The tweezer of the lights `light_of` makes (852 nm, waist 0.8 µm, at the frequency index −4), and the frequency
    /// index of the light beside it there.
    const TWEEZER: (V3, Option<f64>, i32, [C64; 3]) = ([0.0, 0.0, 1.0], Some(0.8), -4, [C64::ZERO, C64::real(1e5), C64::ZERO]);
    const BESIDE: i32 = 0;
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
            .fold(0.0, larger)
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

    /// The region (SPEC §1.3): outside where the trap light (the tweezer alone) is below 1/20 of its focus value -- on the
    /// tweezer's axis beyond z = z_R√19, across it beyond ρ = w₀√(ln 20/2); a beam of another frequency, focused or a
    /// plane wave, does not widen it.
    #[test]
    fn the_region_is_where_a_focused_beam_is_bright() {
        let (k, w) = (7.37, TWEEZER.1.unwrap());
        let zr = k * w * w / 2.0;
        let probe = ([1.0, 0.0, 0.0], None, BESIDE, [C64::real(10.0), C64::ZERO, C64::ZERO]);
        let light = light_of(&[TWEEZER, probe]);
        let outside = |r: V3| light.outside(r);
        let (axial, radial) = (zr * (1.0 / REGION_INTENSITY - 1.0).sqrt(), w * ((1.0 / REGION_INTENSITY).ln() / 2.0).sqrt());
        let margin = 1.0 + 1e-9;
        assert!(!outside([0.0, 0.0, axial / margin]) && outside([0.0, 0.0, axial * margin]));
        assert!(!outside([radial / margin, 0.0, 0.0]) && outside([0.0, radial * margin, 0.0]));
        // A probe focused on the millimetre scale, of another frequency, does not widen the region (the cut-over's
        // finding: an atom it pushed out of the tweezer was never lost).
        let wide = ([1.0, 0.0, 0.0], Some(1000.0), BESIDE, [C64::real(10.0), C64::ZERO, C64::ZERO]);
        let probed = light_of(&[TWEEZER, wide]);
        assert!(probed.outside([0.0, 0.0, axial * margin]) && probed.outside([0.0, 0.0, 1e4]));
    }
}
