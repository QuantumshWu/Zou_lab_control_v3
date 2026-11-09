//! The near light's elimination (SPEC §1.4): second-order Reiter–Sørensen with Brillouin–Wigner dressing and first-order
//! Doppler, for the co-rotating components the units keep near (the near carriers), evaluated exactly at a point with its
//! first derivatives -- the definition the tables interpolate (SPEC §2.3).
//!
//! # The response
//! A near carrier c (`Carrier`) drives V_c(r) = Σ_l a_l(r) X_l e^{−iδ_c t}; its folded part V^f zeroes the kept drives' entries
//! (kept as units, or for free: a drive inside a cluster whose residual beat is exactly zero). The excited amplitude it drives from
//! a ground state b of the cluster follows adiabatically, −R[:,b] x_b, with (internal.md 1.2)
//!
//! ```text
//! R_c[:,b] = (K + Σ_b(z) − z)⁻¹ V_c^f[:,b]      z = E_b + δ_c + Σ_{d≠c} ΔE_b^d      K = H_ee − (i/2) D
//! ΔE_b^d = −Re[V_d^f[:,b]† (K − E_b − δ_d)⁻¹ V_d^f[:,b]]          Σ_b(z) = Σ_{d, a∉C} V_d^f[:,a] V_d^f[:,a]† / (z − Ẽ_a − δ_d)
//! ```
//!
//! H the far light's Hamiltonian at the point (far.rs: the base and every lit frequency's static form), E_b its ground diagonal,
//! Ẽ_a = E_a + Σ_d ΔE_a^d. The other carriers dress b's energy (Brillouin–Wigner: in the denominator, to all orders); the folded
//! channels to the ground states of other clusters (a ∉ C) are the self-energy of the excited states (a strong beam's Autler–Townes
//! dressing of a probe through a folded Raman pair); the channels to the cluster's own states are its dynamics', none of Σ.
//! Within a carrier the dressing takes T̄_d, the beams' coherence kept within a polarization unit (beams of parallel fields:
//! a standing wave's intensity pattern) and averaged between units (SPEC §1.4): V_d V_d† → Σ_u v_u v_u†, v_u = Σ_{l∈u} a_l X^f_l;
//! the units' interference is the dressing's error, measured at the atom. Per beam the response is T_l, R_c = Σ_l a_l T_l.
//!
//! # First-order Doppler
//! Absorbing beam l raises the excited state's energy by w_l = (ħ/m)(k_l·κ + |k_l|²/2); to first order T_l → T_l − w_l T2_l,
//! T2 = (K + Σ − z)⁻¹ T (internal.md 1.4): in the generator and the photons alike. Its error is the second order,
//! (w_l ‖T2_l‖/‖T_l‖)² measured at the atom (units.rs); no clamp, no switch (SPEC §1.4).
//!
//! # The generator's near terms (assemble.rs)
//! Per pair of carriers (c', c) and of their beams (l', l): a_l'* a_l S_l'l on the cluster's ground states, S_l'l =
//! −½(X^f_l'† T_l + T_l'† X^f_l), at the beat δ_c' − δ_c and wave vector k_l − k_l' -- the light shifts, the Raman and Zeeman
//! couplings of the light, the beats of two carriers; an entry whose residual beat in the frames is not zero only where its unit
//! (`Unit::Term`) is kept. The decay −(i/2)Σ J†J is the photons' (photon.rs): per beam the jumps C_a T_l. The explicit states'
//! energies take the folded drives' van Vleck shift ½ Σ_b V^f[e,b] V^f[f,b]* (1/(E_e − E_b − δ_c) + 1/(E_f − E_b − δ_c)). The kept
//! drives enter as they are: a_l X_l[e,b] e^{−iδ_c t} and its adjoint.
//!
//! # Derivatives
//! Everything above is a resolvent of the far light's H(r) and the fields: ∂_i T = −(K + Σ − z)⁻¹ ∂_i(K + Σ − z) T, with ∂_i K, ∂_i E_b
//! from the far light's exact derivatives (far.rs), ∂_i ΔE from the dressing's own resolvent, ∂_i Σ from the fields' and the
//! energies' -- the force's exact definition at the point.

use super::assemble::Parts;
use super::cluster::Edge;
use super::potential::Expansion;
use super::reach::{distance, Bounds, Span};
use super::{Beat, Component, Gate, Lights, Model, Point, Quadratic, Reach, Sample, Unit, UnitError};
use crate::field::Amplitude;
use crate::linalg::{dot3, Lu, Mat, C64};
use crate::request::{Request, GROUND, STATES};
use super::far::EXCITED;

/// A near carrier: a co-rotating component the units keep near, its beams' couplings and folded couplings, and its polarization
/// units (beams of parallel fields).
#[derive(Clone, Debug)]
pub(crate) struct Carrier {
    pub(crate) frequency: usize,
    /// δ_c exactly and its value (rad/µs).
    pub(crate) beat: Beat,
    pub(crate) detuning: f64,
    /// Its beams (request indices), their couplings X_l and folded couplings X^f_l (STATES × STATES), their units.
    pub(crate) beams: Vec<usize>,
    pub(crate) couplings: Vec<Mat>,
    pub(crate) folded: Vec<Mat>,
    pub(crate) units: Vec<usize>,
}

impl Carrier {
    /// The near carriers of a model: every co-rotating component not folded far, its kept drives' entries zeroed in X^f
    /// (`Model::drive_kept`). Units: beams whose focus fields are parallel to the rounding of their inner product,
    /// 1 − |⟨E, E'⟩|²/(‖E‖²‖E'‖²) ≤ γ₆ (Cauchy–Schwarz's equality).
    pub(crate) fn all(model: &Model, _request: &Request) -> Result<Vec<Carrier>, String> {
        let mut out = vec![];
        for frequency in 0..model.light.frequencies.len() {
            if model.folded(Component { frequency, counter: false }) {
                continue;
            }
            let f = &model.light.frequencies[frequency];
            let tone = &model.tones[frequency];
            let couplings = tone.couplings[0].clone();
            let mut units: Vec<usize> = vec![];
            for (j, e) in f.fields.iter().enumerate() {
                let parallel = |i: usize| {
                    let o = &f.fields[i];
                    let inner: C64 = (0..3).map(|q| o[q].conj() * e[q]).sum();
                    let (n1, n2) = ((0..3).map(|q| o[q].norm_sqr()).sum::<f64>(), (0..3).map(|q| e[q].norm_sqr()).sum::<f64>());
                    1.0 - inner.norm_sqr() / (n1 * n2) <= crate::units::rounding(6)
                };
                let unit = (0..j).find(|&i| parallel(i)).map_or(j, |i| units[i]);
                units.push(unit);
            }
            out.push(Carrier {
                frequency,
                beat: tone.beat.clone(),
                detuning: tone.detuning,
                beams: f.beams.clone(),
                folded: couplings.clone(),
                couplings,
                units,
            });
        }
        Ok(out)
    }
}

/// A matrix with its gradient along the lab axes.
#[derive(Clone, Debug)]
pub(crate) struct Slope {
    pub(crate) value: Mat,
    pub(crate) gradient: [Mat; 3],
}

impl Slope {
    fn zeros(rows: usize, cols: usize) -> Slope {
        Slope { value: Mat::zeros(rows, cols), gradient: std::array::from_fn(|_| Mat::zeros(rows, cols)) }
    }
}

/// The near light at a point on a cluster (module documentation): per carrier (all of them; empty where unlit) and beam the
/// response with its Doppler term on the cluster's ground places, T̃_l = T_l − w_l T2_l (EXCITED × n_g), its gradient, and per
/// carrier and ground place the Doppler term's relative size w‖T2‖/‖T‖ (its error's square root, units.rs); and the far light's
/// H with its gradient (the explicit states' energies).
pub(crate) struct NearAt {
    pub(crate) responses: Vec<Vec<Slope>>,
    pub(crate) doppler: Vec<Vec<f64>>,
    pub(crate) hamiltonian: Slope,
}

impl Model {
    /// The far light's Hamiltonian H(r) on all the states with its gradient: the base and every lit frequency's static form.
    pub(crate) fn far_hamiltonian(&self, amplitudes: &[Amplitude], lights: &Lights) -> Slope {
        let mut h = Slope { value: self.base.clone(), gradient: std::array::from_fn(|_| Mat::zeros(STATES, STATES)) };
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            if !lights.0[w] {
                continue;
            }
            let n = f.beams.len();
            for j in 0..n {
                for i in 0..n {
                    let c = super::far::pair(&amplitudes[f.beams[j]], &amplitudes[f.beams[i]]);
                    h.value.add_scaled(c.value, &tone.pairs[j * n + i]);
                    for x in 0..3 {
                        h.gradient[x].add_scaled(c.gradient[x], &tone.pairs[j * n + i]);
                    }
                }
            }
        }
        h
    }

    /// Whether a carrier's drive e ← b is kept: as a unit, or for free inside a cluster at a residual beat of exactly zero.
    pub(crate) fn drive_kept(&self, carrier: usize, e: usize, b: usize) -> bool {
        let c = &self.carriers[carrier];
        let component = Component { frequency: c.frequency, counter: false };
        if self.kept.contains(&Unit::Drive { carrier: component, excited: e, ground: b }) {
            return true;
        }
        self.clusters.iter().any(|k| {
            let (pe, pb) = (k.states.iter().position(|&s| s == e), k.states.iter().position(|&s| s == b));
            matches!((pe, pb), (Some(pe), Some(pb)) if k.frames[pe].plus(-1, &k.frames[pb]) == c.beat)
        })
    }

    /// The near light at a point on a cluster (`NearAt`), every carrier lit in `lights`.
    pub(crate) fn near_at(&self, cluster: usize, at: &Point, amplitudes: &[Amplitude], lights: &Lights) -> NearAt {
        let c = &self.clusters[cluster];
        let g = c.ground;
        let h = self.far_hamiltonian(amplitudes, lights);
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let mut k = h.value.block(&excited, &excited);
        k.add_scaled(C64::new(0.0, -0.5), &self.decay);
        let dk: [Mat; 3] = std::array::from_fn(|i| h.gradient[i].block(&excited, &excited));
        let energy = |b: usize| h.value[(b, b)].re;
        let slope = |b: usize| -> [f64; 3] { std::array::from_fn(|i| h.gradient[i][(b, b)].re) };
        let lit: Vec<bool> = self.carriers.iter().map(|x| lights.0[x.frequency]).collect();
        let column = |m: &Mat, b: usize| -> Vec<C64> { (0..EXCITED).map(|x| m[(GROUND + x, b)]).collect() };
        // The dressing ΔE_b^d of every ground state by every lit carrier, with its gradient.
        let mut shift = vec![[0.0_f64; GROUND]; self.carriers.len()];
        let mut shift_slope = vec![[[0.0_f64; 3]; GROUND]; self.carriers.len()];
        for (d, carrier) in self.carriers.iter().enumerate() {
            if !lit[d] {
                continue;
            }
            for b in 0..GROUND {
                let mut m = k.clone();
                for x in 0..EXCITED {
                    m[(x, x)] -= C64::real(energy(b) + carrier.detuning);
                }
                let lu = Lu::new(&m).expect("K − z regular: the decay keeps K's spectrum off the real axis");
                let dm: [Mat; 3] = std::array::from_fn(|i| {
                    let mut x = dk[i].clone();
                    for y in 0..EXCITED {
                        x[(y, y)] -= C64::real(slope(b)[i]);
                    }
                    x
                });
                let n = carrier.beams.len();
                let ys: Vec<(Vec<C64>, [Vec<C64>; 3])> = (0..n)
                    .map(|l| {
                        let mut y = column(&carrier.folded[l], b);
                        lu.solve(&mut y);
                        let dy = std::array::from_fn(|i| {
                            let mut z: Vec<C64> = dm[i].apply(&y).into_iter().map(|v| -v).collect();
                            lu.solve(&mut z);
                            z
                        });
                        (y, dy)
                    })
                    .collect();
                for j in 0..n {
                    for i in 0..n {
                        if carrier.units[j] != carrier.units[i] {
                            continue;
                        }
                        let coefficient = super::far::pair(&amplitudes[carrier.beams[j]], &amplitudes[carrier.beams[i]]);
                        let xj = column(&carrier.folded[j], b);
                        let form = crate::linalg::inner(&xj, &ys[i].0);
                        shift[d][b] -= (coefficient.value * form).re;
                        for x in 0..3 {
                            shift_slope[d][b][x] -= (coefficient.gradient[x] * form + coefficient.value * crate::linalg::inner(&xj, &ys[i].1[x])).re;
                        }
                    }
                }
            }
        }
        let dressed = |a: usize| -> f64 { energy(a) + (0..self.carriers.len()).map(|d| shift[d][a]).sum::<f64>() };
        let dressed_slope = |a: usize| -> [f64; 3] { std::array::from_fn(|i| slope(a)[i] + (0..self.carriers.len()).map(|d| shift_slope[d][a][i]).sum::<f64>()) };
        let outside: Vec<usize> = (0..GROUND).filter(|&a| self.membership[a] != cluster).collect();
        let mut responses = vec![];
        let mut doppler = vec![];
        for (ci, carrier) in self.carriers.iter().enumerate() {
            let n = carrier.beams.len();
            if !lit[ci] {
                responses.push(vec![]);
                doppler.push(vec![]);
                continue;
            }
            let mut out: Vec<Slope> = (0..n).map(|_| Slope::zeros(EXCITED, g)).collect();
            let mut sizes = vec![0.0; g];
            let w: Vec<f64> = carrier
                .beams
                .iter()
                .map(|&l| {
                    let kv = self.light.beams[l].wave_vector();
                    self.hbar_over_mass * (dot3(kv, at.kappa) + 0.5 * dot3(kv, kv))
                })
                .collect();
            for (pb, &b) in c.states[..g].iter().enumerate() {
                let z = energy(b) + carrier.detuning + (0..self.carriers.len()).filter(|&d| d != ci).map(|d| shift[d][b]).sum::<f64>();
                let dz: [f64; 3] = std::array::from_fn(|i| slope(b)[i] + (0..self.carriers.len()).filter(|&d| d != ci).map(|d| shift_slope[d][b][i]).sum::<f64>());
                // The self-energy of the folded channels to the other clusters' ground states.
                let mut m = k.clone();
                let mut dm = dk.clone();
                for x in 0..EXCITED {
                    m[(x, x)] -= C64::real(z);
                    for i in 0..3 {
                        dm[i][(x, x)] -= C64::real(dz[i]);
                    }
                }
                for (d, other) in self.carriers.iter().enumerate() {
                    if !lit[d] {
                        continue;
                    }
                    for &a in &outside {
                        let den = z - dressed(a) - other.detuning;
                        let dden: [f64; 3] = std::array::from_fn(|i| dz[i] - dressed_slope(a)[i]);
                        let units: Vec<usize> = {
                            let mut u = other.units.clone();
                            u.sort();
                            u.dedup();
                            u
                        };
                        for unit in units {
                            let members: Vec<usize> = (0..other.beams.len()).filter(|&l| other.units[l] == unit).collect();
                            let mut v = vec![C64::ZERO; EXCITED];
                            let mut dv: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; EXCITED]);
                            for &l in &members {
                                let amp = &amplitudes[other.beams[l]];
                                for x in 0..EXCITED {
                                    let xv = other.folded[l][(GROUND + x, a)];
                                    v[x] += amp.value * xv;
                                    for i in 0..3 {
                                        dv[i][x] += amp.gradient[i] * xv;
                                    }
                                }
                            }
                            for x in 0..EXCITED {
                                for y in 0..EXCITED {
                                    let vv = v[x] * v[y].conj();
                                    m[(x, y)] += vv / den;
                                    for i in 0..3 {
                                        dm[i][(x, y)] += (dv[i][x] * v[y].conj() + v[x] * dv[i][y].conj()) / den - vv * (dden[i] / (den * den));
                                    }
                                }
                            }
                        }
                    }
                }
                let lu = Lu::new(&m).expect("K + Σ − z regular: the decay keeps it off the real axis");
                let (mut t_norm, mut t2_norm) = (0.0_f64, 0.0_f64);
                for l in 0..n {
                    let mut t = column(&carrier.folded[l], b);
                    if t.iter().all(|x| *x == C64::ZERO) {
                        continue;
                    }
                    lu.solve(&mut t);
                    let mut t2 = t.clone();
                    lu.solve(&mut t2);
                    let dt: [Vec<C64>; 3] = std::array::from_fn(|i| {
                        let mut y: Vec<C64> = dm[i].apply(&t).into_iter().map(|v| -v).collect();
                        lu.solve(&mut y);
                        y
                    });
                    let dt2: [Vec<C64>; 3] = std::array::from_fn(|i| {
                        let mt2 = dm[i].apply(&t2);
                        let mut y: Vec<C64> = dt[i].iter().zip(&mt2).map(|(a, b)| *a - *b).collect();
                        lu.solve(&mut y);
                        y
                    });
                    t_norm = t_norm.max(crate::linalg::norm_sqr(&t).sqrt());
                    t2_norm = t2_norm.max(w[l].abs() * crate::linalg::norm_sqr(&t2).sqrt());
                    for x in 0..EXCITED {
                        out[l].value[(x, pb)] = t[x] - t2[x] * w[l];
                        for i in 0..3 {
                            out[l].gradient[i][(x, pb)] = dt[i][x] - dt2[i][x] * w[l];
                        }
                    }
                }
                sizes[pb] = if t_norm > 0.0 { t2_norm / t_norm } else { 0.0 };
            }
            responses.push(out);
            doppler.push(sizes);
        }
        NearAt { responses, doppler, hamiltonian: h }
    }

    /// The near light's terms of a cluster's generator from a point's responses (module documentation, # The generator's near
    /// terms).
    pub(crate) fn add_near_terms(&self, parts: &mut Parts, cluster: usize, near: &NearAt, amplitudes: &[Amplitude], lights: &Lights) {
        let c = &self.clusters[cluster];
        let g = c.ground;
        let zero_wave = self.no_wave();
        // The folded couplings' light shifts and couplings on the ground states.
        for (cj, cp) in self.carriers.iter().enumerate() {
            if !lights.0[cp.frequency] {
                continue;
            }
            for (ci, cc) in self.carriers.iter().enumerate() {
                if !lights.0[cc.frequency] {
                    continue;
                }
                let beat = cp.beat.plus(-1, &cc.beat);
                for (lj, &bj) in cp.beams.iter().enumerate() {
                    for (li, &bi) in cc.beams.iter().enumerate() {
                        let coefficient = super::far::pair(&amplitudes[bj], &amplitudes[bi]);
                        let (xj, xi) = (&cp.folded[lj], &cc.folded[li]);
                        let (tj, ti) = (&near.responses[cj][lj], &near.responses[ci][li]);
                        let entry = |x: usize, y: usize, grad: Option<usize>| -> C64 {
                            if x >= g || y >= g || !self.term_included(cluster, x, y, Component { frequency: cc.frequency, counter: false }, Component { frequency: cp.frequency, counter: false }, &beat) {
                                return C64::ZERO;
                            }
                            let (a, b) = (c.states[x], c.states[y]);
                            let (tiv, tjv) = match grad {
                                None => (&ti.value, &tj.value),
                                Some(i) => (&ti.gradient[i], &tj.gradient[i]),
                            };
                            let mut s = C64::ZERO;
                            for e in 0..EXCITED {
                                s += xj[(GROUND + e, a)].conj() * tiv[(e, y)] + tjv[(e, x)].conj() * xi[(GROUND + e, b)];
                            }
                            s * -0.5
                        };
                        parts.add_varying(&beat, &self.pair_wave(bj, bi), &coefficient, &|x, y| entry(x, y, None), &|i, x, y| entry(x, y, Some(i)));
                    }
                }
            }
        }
        // The explicit states' van Vleck shift by the folded drives, and the kept drives.
        let h = &near.hamiltonian;
        for cc in self.carriers.iter().filter(|x| lights.0[x.frequency]) {
            let zero = self.axes.zero();
            for (lj, &bj) in cc.beams.iter().enumerate() {
                for (li, &bi) in cc.beams.iter().enumerate() {
                    let coefficient = super::far::pair(&amplitudes[bj], &amplitudes[bi]);
                    let (xj, xi) = (&cc.folded[lj], &cc.folded[li]);
                    let shift = |x: usize, y: usize, grad: Option<usize>| -> C64 {
                        if x < g || y < g {
                            return C64::ZERO;
                        }
                        let (e, f) = (c.states[x], c.states[y]);
                        let mut s = C64::ZERO;
                        for b in 0..GROUND {
                            let v = xi[(e, b)] * xj[(f, b)].conj();
                            if v == C64::ZERO {
                                continue;
                            }
                            let de = h.value[(e, e)].re - h.value[(b, b)].re - cc.detuning;
                            let df = h.value[(f, f)].re - h.value[(b, b)].re - cc.detuning;
                            s += v * match grad {
                                None => 0.5 * (1.0 / de + 1.0 / df),
                                Some(i) => {
                                    let (se, sf) = (h.gradient[i][(e, e)].re - h.gradient[i][(b, b)].re, h.gradient[i][(f, f)].re - h.gradient[i][(b, b)].re);
                                    -0.5 * (se / (de * de) + sf / (df * df))
                                }
                            };
                        }
                        s
                    };
                    parts.add_varying(&zero, &self.pair_wave(bj, bi), &coefficient, &|x, y| shift(x, y, None), &|i, x, y| shift(x, y, Some(i)));
                }
            }
            for (li, &bi) in cc.beams.iter().enumerate() {
                let a = &amplitudes[bi];
                let x = &cc.couplings[li];
                let drive = |p: usize, q: usize| -> C64 {
                    let (e, b) = (c.states[p], c.states[q]);
                    if p < g || q >= g || !self.drive_kept(self.carrier_index(cc.frequency), e, b) {
                        return C64::ZERO;
                    }
                    x[(e, b)]
                };
                parts.add(&cc.beat.negated(), &self.beam_wave(bi), a, &drive, true);
                parts.add(&cc.beat, &self.beam_wave(bi).negated(), &super::assemble::conjugate(a), &|p, q| drive(q, p).conj(), true);
            }
        }
        let _ = zero_wave;
    }

    /// A carrier's index from its frequency.
    pub(crate) fn carrier_index(&self, frequency: usize) -> usize {
        self.carriers.iter().position(|c| c.frequency == frequency).expect("a near carrier")
    }

    /// Whether a near coupling of two ground places (x ← y, absorbing `from`, emitting `to`, base beat `beat`) enters the
    /// cluster's generator: at a residual beat of exactly zero in the frames (free), or as a kept unit.
    pub(crate) fn term_included(&self, cluster: usize, x: usize, y: usize, from: Component, to: Component, beat: &Beat) -> bool {
        let c = &self.clusters[cluster];
        if beat.plus(1, &c.frames[x]).plus(-1, &c.frames[y]).is_zero() {
            return true;
        }
        self.kept.contains(&term_unit(c.states[x], c.states[y], Some(from), Some(to)))
    }

    /// The near carriers' part of V_L(r; p) (potential.rs): Σ_b p_b of their static shifts on the ground states (each in its
    /// cluster), with the trap light on less with it off, with its gradient (the Hessian: the coefficients' and the gradients'
    /// products, the responses' second derivatives left out -- module documentation, # Derivatives).
    pub(crate) fn near_potential(&self, r: [f64; 3], amplitudes: &[Amplitude], p: &[f64; GROUND], lights: &Lights) -> Expansion {
        let mut out = Expansion { quadratic: Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] }, scale: 0.0, terms: 0 };
        if self.carriers.iter().all(|x| !lights.0[x.frequency]) {
            return out;
        }
        let mut off = lights.clone();
        off.0[self.light.trap] = false;
        for (lights, sign) in [(lights, 1.0), (&off, -1.0)] {
            for cluster in 0..self.clusters.len() {
                let c = &self.clusters[cluster];
                if c.states[..c.ground].iter().all(|&s| p[s] == 0.0) {
                    continue;
                }
                let at = Point { r, kappa: [0.0; 3], t: 0.0 };
                let near = self.near_at(cluster, &at, amplitudes, lights);
                for (ci, cc) in self.carriers.iter().enumerate() {
                    if !lights.0[cc.frequency] {
                        continue;
                    }
                    for (lj, &bj) in cc.beams.iter().enumerate() {
                        for (li, &bi) in cc.beams.iter().enumerate() {
                            let coefficient = super::far::pair(&amplitudes[bj], &amplitudes[bi]);
                            for (pb, &b) in c.states[..c.ground].iter().enumerate() {
                                if p[b] == 0.0 {
                                    continue;
                                }
                                let s = |m: &Mat, n: &Mat| -> C64 {
                                    let mut z = C64::ZERO;
                                    for e in 0..EXCITED {
                                        z += cc.folded[lj][(GROUND + e, b)].conj() * m[(e, pb)] + n[(e, pb)].conj() * cc.folded[li][(GROUND + e, b)];
                                    }
                                    z * -0.5
                                };
                                let (ti, tj) = (&near.responses[ci][li], &near.responses[ci][lj]);
                                let value = s(&ti.value, &tj.value);
                                let ds: [C64; 3] = std::array::from_fn(|i| s(&ti.gradient[i], &tj.gradient[i]));
                                let q = &mut out.quadratic;
                                let w = sign * p[b];
                                q.value += w * (coefficient.value * value).re;
                                for x in 0..3 {
                                    q.gradient[x] += w * (coefficient.gradient[x] * value + coefficient.value * ds[x]).re;
                                    for y in 0..3 {
                                        q.hessian[x][y] += w * (coefficient.hessian[x][y] * value + coefficient.gradient[x] * ds[y] + coefficient.gradient[y] * ds[x]).re;
                                    }
                                }
                                out.scale += (coefficient.value * value).abs() * p[b];
                                out.terms += 1;
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The near trap light's light-shift bound per beam pair of the trap frequency (potential.rs `Model::envelope`): where the
    /// trap's co-rotating component is near, |[X^f_l'† G X^f_l]_p| with ‖G‖ ≤ 2/Γ_min (a resolvent of an excited block whose
    /// decay is at least Γ_min/2 off the real axis) -- the far fold's bound on the folded near drive.
    pub(crate) fn near_envelope(&self, p: &[f64; GROUND]) -> Vec<f64> {
        let Some(cc) = self.carriers.iter().find(|c| c.frequency == self.light.trap) else { return vec![] };
        let gamma = (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(f64::INFINITY, f64::min);
        let n = cc.beams.len();
        (0..n * n)
            .map(|x| {
                let (j, i) = (x / n, x % n);
                (0..GROUND)
                    .filter(|&b| p[b] != 0.0)
                    .map(|b| {
                        let (a, c) = ((0..EXCITED).map(|e| cc.folded[j][(GROUND + e, b)].norm_sqr()).sum::<f64>().sqrt(), (0..EXCITED).map(|e| cc.folded[i][(GROUND + e, b)].norm_sqr()).sum::<f64>().sqrt());
                        p[b] * a * c * 2.0 / gamma
                    })
                    .sum::<f64>()
            })
            .collect()
    }

    /// The near carriers' folded channels on a cluster at a point (readout.rs): per carrier R = Σ_l a_l T̃_l.
    pub(crate) fn near_channels(&self, cluster: usize, at: &Point, amplitudes: &[Amplitude], lights: &Lights) -> Vec<super::readout::Channels> {
        if self.carriers.iter().all(|x| !lights.0[x.frequency]) {
            return vec![];
        }
        let near = self.near_at(cluster, at, amplitudes, lights);
        let g = self.clusters[cluster].ground;
        self.carriers
            .iter()
            .enumerate()
            .filter(|(_, x)| lights.0[x.frequency])
            .map(|(ci, cc)| {
                let mut response = Mat::zeros(EXCITED, g);
                for (l, &b) in cc.beams.iter().enumerate() {
                    response.add_scaled(amplitudes[b].value, &near.responses[ci][l].value);
                }
                super::readout::Channels { nu: cc.beat.clone(), response }
            })
            .collect()
    }

    /// The near light's part of V_L far from the trap: the near shifts with every focused beam's amplitude zero and the trap
    /// light on, less off -- zero, since far from the trap the trap light is off (its plane waves aside, which the far part takes).
    pub(crate) fn near_asymptote(&self, _p: &[f64; GROUND], _lights: &Lights) -> f64 {
        0.0
    }

    /// The near carriers' photon sources (photon.rs): per carrier and beam, its jumps C_a T̃_l at the point.
    pub(crate) fn near_sources(&self) -> Vec<super::photon::Source> {
        let mut out = vec![];
        for (ci, cc) in self.carriers.iter().enumerate() {
            for (l, &b) in cc.beams.iter().enumerate() {
                out.push(super::photon::Source {
                    frequency: Some(cc.frequency),
                    beam: Some(b),
                    conjugate: false,
                    beat: cc.beat.negated(),
                    wave: self.beam_wave(b),
                    jumps: super::photon::Jumps::Near { carrier: ci, beam: l },
                    folded: true,
                });
            }
        }
        out
    }

    /// The near carriers' units: their drives, and the couplings of two ground states (or one state's beat of two carriers) they
    /// make at second order (`Unit::Term`, canonical: a < b, or a = b with from < to).
    pub(crate) fn near_units(&self) -> Vec<Unit> {
        let mut out = vec![];
        for cc in &self.carriers {
            let component = Component { frequency: cc.frequency, counter: false };
            for e in GROUND..STATES {
                for b in 0..GROUND {
                    if cc.couplings.iter().any(|x| x[(e, b)] != C64::ZERO) {
                        out.push(Unit::Drive { carrier: component, excited: e, ground: b });
                    }
                }
            }
        }
        for cp in &self.carriers {
            for cc in &self.carriers {
                let (from, to) = (Component { frequency: cc.frequency, counter: false }, Component { frequency: cp.frequency, counter: false });
                for a in 0..GROUND {
                    for b in 0..GROUND {
                        if a == b && from == to {
                            continue;
                        }
                        let coupled = (GROUND..STATES).any(|e| cp.couplings.iter().any(|x| x[(e, a)] != C64::ZERO) && cc.couplings.iter().any(|x| x[(e, b)] != C64::ZERO));
                        let unit = term_unit(a, b, Some(from), Some(to));
                        if coupled && !out.contains(&unit) {
                            out.push(unit);
                        }
                    }
                }
            }
        }
        out
    }

    /// A kept near unit's edge: a drive joins its states at θ_e − θ_b = δ_c, its strongest beam's wave vector; a coupling of two
    /// ground states at θ_a − θ_b = δ_from − δ_to, its strongest beam pair's k_l − k_l'.
    pub(crate) fn near_edge(&self, unit: &Unit) -> Option<Edge> {
        match unit {
            Unit::Drive { carrier, excited, ground } => {
                let cc = &self.carriers[self.carrier_index(carrier.frequency)];
                let (best, strength) = cc.beams.iter().zip(&cc.couplings).fold((cc.beams[0], 0.0_f64), |(best, s), (&l, x)| {
                    let v = x[(*excited, *ground)].abs();
                    (if v > s { l } else { best }, s.max(v))
                });
                Some(Edge { a: *excited, b: *ground, beat: cc.beat.clone(), wave: self.beam_wave(best), strength })
            }
            Unit::Term { a, b, from: Some(from), to: Some(to) } => {
                let (cc, cp) = (&self.carriers[self.carrier_index(from.frequency)], &self.carriers[self.carrier_index(to.frequency)]);
                let mut best = (0.0_f64, self.no_wave());
                for (&li, xi) in cc.beams.iter().zip(&cc.couplings) {
                    for (&lj, xj) in cp.beams.iter().zip(&cp.couplings) {
                        let size: f64 = (GROUND..STATES).map(|e| xj[(e, *a)].abs() * xi[(e, *b)].abs()).sum();
                        if size > best.0 {
                            best = (size, self.pair_wave(lj, li));
                        }
                    }
                }
                Some(Edge { a: *a, b: *b, beat: cc.beat.plus(-1, &cp.beat), wave: best.1, strength: best.0 })
            }
            _ => None,
        }
    }

    /// A near unit's gate over the reach (units.rs, photon.rs).
    pub(crate) fn near_gate(&self, unit: &Unit, reach: &Reach) -> Gate {
        match unit {
            Unit::Coherence { .. } => self.coherence_gate(unit, reach),
            Unit::Drive { .. } => self.drive_gate(unit, reach),
            Unit::Term { .. } => self.term_gate(unit, reach),
            _ => unreachable!("the far units' gates are units.rs'"),
        }
    }

    /// A near unit's error at a sample.
    pub(crate) fn near_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        match unit {
            Unit::Coherence { .. } => self.coherence_error(unit, sample),
            Unit::Drive { .. } => self.drive_error(unit, sample),
            Unit::Term { .. } => self.term_error(unit, sample),
            _ => unreachable!("the far units' errors are units.rs'"),
        }
    }

    /// The bound of the near light's differential shift of two states over a cell (the static gates' separations): per lit
    /// carrier Σ_l'l |a_l'||a_l| (‖X^f_l'[:,a]‖‖X^f_l[:,a]‖ + ‖X^f_l'[:,b]‖‖X^f_l[:,b]‖) 2/Γ_min, the resolvent's norm at most 2/Γ_min.
    pub(crate) fn near_spread(&self, a: usize, b: usize, spans: &[Span]) -> f64 {
        let gamma = (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(f64::INFINITY, f64::min);
        let mut out = 0.0;
        for cc in &self.carriers {
            let column = |m: &Mat, s: usize| -> f64 { if s >= GROUND { 0.0 } else { (0..EXCITED).map(|e| m[(GROUND + e, s)].norm_sqr()).sum::<f64>().sqrt() } };
            for (lj, x) in cc.folded.iter().enumerate() {
                for (li, y) in cc.folded.iter().enumerate() {
                    let amp = spans[cc.beams[lj]].most * spans[cc.beams[li]].most;
                    out += amp * (column(x, a) * column(y, a) + column(x, b) * column(y, b)) * 2.0 / gamma;
                }
            }
        }
        out
    }

    /// The frequency whose kept drives feed an explicit state's photons (photon.rs): the first carrier with a kept drive into it.
    pub(crate) fn driving_frequency(&self, source: &super::photon::Source) -> Option<usize> {
        let super::photon::Jumps::Constant(jumps) = &source.jumps else { return None };
        let e = (GROUND..STATES).find(|&e| (0..GROUND).any(|b| (0..3).any(|a| jumps[a][(b, e)] != C64::ZERO)))?;
        self.carriers
            .iter()
            .enumerate()
            .find(|(ci, cc)| (0..GROUND).any(|b| cc.couplings.iter().any(|x| x[(e, b)] != C64::ZERO) && self.drive_kept(*ci, e, b)))
            .map(|(_, cc)| cc.frequency)
    }
}

/// The far light's energies over a cell (the near gates' separations): per state its value at the centre and the bound of how
/// far it moves within the cell, and per excited state the Gershgorin radius of the far couplings among the excited states.
pub(crate) struct CellEnergies {
    pub(crate) centre: Vec<f64>,
    pub(crate) slack: Vec<f64>,
    pub(crate) radius: Vec<f64>,
}

impl Model {
    /// The far light's energies over a cell (`CellEnergies`), every light on: each beam pair's coefficient a_l'* a_l within its
    /// centre value ± (L_l' ρ max|a_l| + |a_l'(c)| L_l ρ) on a bounded cell, within the disk of radius 1 on an unbounded one.
    pub(crate) fn cell_energies(&self, spans: &[Span]) -> CellEnergies {
        let mut centre: Vec<f64> = (0..STATES).map(|s| self.base[(s, s)].re).collect();
        let mut slack = vec![0.0; STATES];
        let mut off = Mat::from_fn(STATES, STATES, |x, y| if x == y { C64::ZERO } else { self.base[(x, y)] });
        let mut off_slack = Mat::zeros(STATES, STATES);
        for (f, tone) in self.light.frequencies.iter().zip(&self.tones) {
            let n = f.beams.len();
            for j in 0..n {
                for i in 0..n {
                    let (sj, si) = (&spans[f.beams[j]], &spans[f.beams[i]]);
                    let (c0, rad) = if sj.spread.is_finite() && si.spread.is_finite() {
                        (sj.centre.conj() * si.centre, sj.spread * si.most + sj.centre.abs() * si.spread)
                    } else {
                        (C64::ZERO, sj.most * si.most)
                    };
                    let p = &tone.pairs[j * n + i];
                    for s in 0..STATES {
                        centre[s] += (c0 * p[(s, s)]).re;
                        slack[s] += rad * p[(s, s)].abs();
                    }
                    for x in GROUND..STATES {
                        for y in GROUND..STATES {
                            if x != y && p[(x, y)] != C64::ZERO {
                                off[(x, y)] += c0 * p[(x, y)];
                                off_slack[(x, y)] += C64::real(rad * p[(x, y)].abs());
                            }
                        }
                    }
                }
            }
        }
        let radius = (0..STATES)
            .map(|x| if x < GROUND { 0.0 } else { (GROUND..STATES).filter(|&y| y != x).map(|y| off[(x, y)].abs() + off_slack[(x, y)].re).sum() })
            .collect();
        CellEnergies { centre, slack, radius }
    }

    /// The least decay rate of the excited states, Γ_min: every folded response's resolvent is at most 2/Γ_min.
    pub(crate) fn least_decay(&self) -> f64 {
        (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(f64::INFINITY, f64::min)
    }

    /// The bound of the dressing ΔE_b^d of ground state b by one carrier over a cell: Σ over its units' beam pairs of
    /// max|a_l'| max|a_l| ‖X^f_l'[:,b]‖ ‖X^f_l[:,b]‖ · 2/Γ_min.
    fn dressing_bound(&self, carrier: &Carrier, b: usize, spans: &[Span]) -> f64 {
        let column = |m: &Mat| (GROUND..STATES).map(|e| m[(e, b)].norm_sqr()).sum::<f64>().sqrt();
        let mut out = 0.0;
        for j in 0..carrier.beams.len() {
            for i in 0..carrier.beams.len() {
                if carrier.units[j] == carrier.units[i] {
                    out += spans[carrier.beams[j]].most * spans[carrier.beams[i]].most * column(&carrier.folded[j]) * column(&carrier.folded[i]);
                }
            }
        }
        out * 2.0 / self.least_decay()
    }

    /// The bound of the near light's force (rad/µs/µm) anywhere: the far light's slope and each carrier's light shift's,
    /// Σ_l'l (L_l' + L_l)‖X_l'‖‖X_l‖ 2/Γ_min, and the scattering's mean recoil k_max Γ_max/2 (the jumps' mean force at most).
    fn force_bound(&self) -> f64 {
        let mut out = self.energy_slope();
        let gamma = self.least_decay();
        let mut k_max = 0.0_f64;
        for carrier in &self.carriers {
            for (j, xj) in carrier.couplings.iter().enumerate() {
                let lj = &self.light.beams[carrier.beams[j]];
                k_max = k_max.max(lj.wave_number);
                for (i, xi) in carrier.couplings.iter().enumerate() {
                    let li = &self.light.beams[carrier.beams[i]];
                    out += (super::reach::lipschitz(lj) + super::reach::lipschitz(li)) * xj.frobenius() * xi.frobenius() * 2.0 / gamma;
                }
            }
        }
        let gamma_max = (0..EXCITED).map(|x| self.decay[(x, x)].re).fold(0.0, f64::max);
        out + k_max * gamma_max / 2.0
    }

    /// A ground state's folded drives' saturation and sweep bounds over a cell (module documentation of units.rs): per lit
    /// carrier and excited state, v_e = Σ_l max|a_l| |X^f_l[e,b]| over |ΔE_e| ≥ √(d² + Γ_e²/4), d the distance of E_e − z_b − k·v
    /// from zero over the cell and the reach's speeds less the excited states' Gershgorin radius and the self-energy's bound;
    /// S = 2 Σ v_e²/|ΔE_e|², a = max (v L + k a_max)/|ΔE_e|².
    fn drive_cell(&self, b: usize, cluster: usize, spans: &[Span], speed: f64) -> (Bounds, Bounds) {
        let energies = self.cell_energies(spans);
        let bounded = spans.iter().all(|s| s.spread.is_finite());
        let outside: Vec<usize> = (0..GROUND).filter(|&a| self.membership[a] != cluster).collect();
        let dress: Vec<f64> = (0..GROUND).map(|a| self.carriers.iter().map(|d| self.dressing_bound(d, a, spans)).sum()).collect();
        let slope = self.energy_slope();
        let acceleration = self.hbar_over_mass * self.force_bound();
        let (mut s, mut a) = (0.0_f64, 0.0_f64);
        for (ci, carrier) in self.carriers.iter().enumerate() {
            let others: f64 = self.carriers.iter().enumerate().filter(|&(d, _)| d != ci).map(|(_, d)| self.dressing_bound(d, b, spans)).sum();
            let (z_lo, z_hi) = (energies.centre[b] - energies.slack[b] + carrier.detuning - others, energies.centre[b] + energies.slack[b] + carrier.detuning + others);
            let k = carrier.beams.iter().map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max);
            // The self-energy's bound: the folded channels to the other clusters.
            let mut sigma = 0.0;
            for d in &self.carriers {
                for &o in &outside {
                    let (lo, hi) = (z_lo - energies.centre[o] - energies.slack[o] - dress[o] - d.detuning, z_hi - energies.centre[o] + energies.slack[o] + dress[o] - d.detuning);
                    let den = distance(lo, hi);
                    let size: f64 = d.beams.iter().zip(&d.folded).map(|(&l, x)| spans[l].most * (GROUND..STATES).map(|e| x[(e, o)].norm_sqr()).sum::<f64>().sqrt()).sum::<f64>().powi(2);
                    sigma += if size == 0.0 { 0.0 } else { size / den };
                }
            }
            for e in GROUND..STATES {
                let v: f64 = carrier.beams.iter().zip(&carrier.folded).map(|(&l, x)| spans[l].most * x[(e, b)].abs()).sum();
                if v == 0.0 {
                    continue;
                }
                let (lo, hi) = (energies.centre[e] - energies.slack[e] - z_hi - k * speed, energies.centre[e] + energies.slack[e] - z_lo + k * speed);
                let d = (distance(lo, hi) - energies.radius[e] - sigma).max(0.0);
                let gamma = self.decay[(e - GROUND, e - GROUND)].re;
                let width = d * d + gamma * gamma / 4.0;
                s += 2.0 * v * v / width;
                a = a.max((speed * 2.0 * slope + k * acceleration) / width);
            }
        }
        let centre = |x: f64| if bounded { x } else { f64::NEG_INFINITY };
        (Bounds { upper: s, centre: centre(s) }, Bounds { upper: a, centre: centre(a) })
    }

    /// A drive's gate over the reach: its ground state's total folded saturation S_b and its drives' sweep (SPEC §1.5).
    pub(crate) fn drive_gate(&self, unit: &Unit, reach: &Reach) -> Gate {
        let Unit::Drive { ground, .. } = unit else { unreachable!() };
        let cluster = self.membership[*ground];
        let saturation = self.bound_over(reach, super::units::EXISTS, &|_c: &super::reach::Cell, spans: &[Span]| self.drive_cell(*ground, cluster, spans, reach.speed).0);
        let sweep = self.bound_over(reach, super::units::EXISTS, &|_c: &super::reach::Cell, spans: &[Span]| self.drive_cell(*ground, cluster, spans, reach.speed).1);
        Gate { saturation, sweep }
    }

    /// The force on a sample's state, −Re⟨ψ|G|ψ⟩/‖ψ‖² of its generator's parts at the point (rad/µs/µm).
    fn sample_force(&self, sample: &Sample) -> [f64; 3] {
        let g = self.assemble(sample.cluster, &sample.at, &self.running());
        let norm: f64 = sample.psi.iter().map(|z| z.norm_sqr()).sum();
        std::array::from_fn(|i| {
            let mut f = C64::ZERO;
            for part in &g.parts {
                f += crate::linalg::inner(&sample.psi, &part.force[i].apply(&sample.psi));
            }
            -f.re / norm
        })
    }

    /// A folded drive's error at a sample (SPEC §1.5): its own saturation s = 2|R_c[e,b]|² at the point (R = Σ_l a_l T̃_l), the
    /// populations' error s p_b, and on T (√2 s + (w‖T2‖/‖T‖)² + a) p_b with its sweep a = |v·∇(E_e − E_b) + k·v̇|/|ΔE|² measured
    /// there, v̇ the state's force over m.
    pub(crate) fn drive_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        let Unit::Drive { carrier, excited, ground } = unit else { unreachable!() };
        let c = &self.clusters[sample.cluster];
        let zero = UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
        let Some(pb) = c.states[..c.ground].iter().position(|&s| s == *ground) else { return zero };
        let p = self.state_populations(sample)[*ground];
        let lights = self.running();
        let amplitudes = self.light.amplitudes(sample.at.r);
        let near = self.near_at(sample.cluster, &sample.at, &amplitudes, &lights);
        let ci = self.carrier_index(carrier.frequency);
        let cc = &self.carriers[ci];
        let r: C64 = cc.beams.iter().enumerate().map(|(l, &bl)| amplitudes[bl].value * near.responses[ci][l].value[(excited - GROUND, pb)]).sum();
        let s = 2.0 * r.norm_sqr();
        let h = &near.hamiltonian;
        let v: [f64; 3] = sample.at.kappa.map(|k| k * self.hbar_over_mass);
        let force = self.sample_force(sample);
        let strongest = cc.beams.iter().zip(&cc.couplings).max_by(|x, y| x.1[(*excited, *ground)].abs().total_cmp(&y.1[(*excited, *ground)].abs())).map(|(&l, _)| l).unwrap();
        let k = self.light.beams[strongest].wave_vector();
        let delta = h.value[(*excited, *excited)].re - h.value[(*ground, *ground)].re - cc.detuning - dot3(k, v);
        let rate: f64 = (0..3).map(|i| v[i] * (h.gradient[i][(*excited, *excited)].re - h.gradient[i][(*ground, *ground)].re)).sum::<f64>() + self.hbar_over_mass * dot3(k, force);
        let gamma = self.decay[(excited - GROUND, excited - GROUND)].re;
        let sweep = rate.abs() / (delta * delta + gamma * gamma / 4.0);
        let doppler = near.doppler[ci][pb].powi(2);
        UnitError { parameter: s, population: s * p, temperature: (std::f64::consts::SQRT_2 * s + doppler + sweep) * p }
    }

    /// A near coupling's (`Unit::Term`) magnitude and beat over a cell: |X_ab| ≤ ½ Σ_e |V_to[e,a]||V_from[e,b]| (1/σ_eb + 1/σ'_ea +
    /// Γ_e/(σ_eb σ'_ea)), σ the drives' denominators' bounds; its beat Ω = E_a − E_b + δ_to − δ_from over the cell, less the
    /// Doppler shift of its strongest beam pair at the reach's speed and the near light's differential shift. S = 2m².
    fn term_cell(&self, a: usize, b: usize, from: Component, to: Component, spans: &[Span], speed: f64) -> Bounds {
        let energies = self.cell_energies(spans);
        let bounded = spans.iter().all(|s| s.spread.is_finite());
        let (cf, ct) = (&self.carriers[self.carrier_index(from.frequency)], &self.carriers[self.carrier_index(to.frequency)]);
        let sigma = |c: &Carrier, g: usize, e: usize| -> f64 {
            let (lo, hi) = (energies.centre[e] - energies.slack[e] - energies.centre[g] - energies.slack[g] - c.detuning, energies.centre[e] + energies.slack[e] - energies.centre[g] + energies.slack[g] - c.detuning);
            let d = (distance(lo, hi) - energies.radius[e]).max(0.0);
            let gamma = self.decay[(e - GROUND, e - GROUND)].re;
            (d * d + gamma * gamma / 4.0).sqrt()
        };
        let mut size = 0.0;
        for e in GROUND..STATES {
            let vf: f64 = cf.beams.iter().zip(&cf.folded).map(|(&l, x)| spans[l].most * x[(e, b)].abs()).sum();
            let vt: f64 = ct.beams.iter().zip(&ct.folded).map(|(&l, x)| spans[l].most * x[(e, a)].abs()).sum();
            if vf == 0.0 || vt == 0.0 {
                continue;
            }
            let (s1, s2) = (sigma(cf, b, e), sigma(ct, a, e));
            let gamma = self.decay[(e - GROUND, e - GROUND)].re;
            size += 0.5 * vt * vf * (1.0 / s1 + 1.0 / s2 + gamma / (s1 * s2));
        }
        let beat = self.omega(&ct.beat.plus(-1, &cf.beat));
        let (lo, hi) = (energies.centre[a] - energies.slack[a] - energies.centre[b] - energies.slack[b] + beat, energies.centre[a] + energies.slack[a] - energies.centre[b] + energies.slack[b] + beat);
        let k = cf.beams.iter().chain(&ct.beams).map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max);
        let sep = distance(lo, hi) - 2.0 * k * speed - self.near_spread(a, b, spans);
        let m = if size == 0.0 { 0.0 } else if sep > 0.0 { size / sep } else { f64::INFINITY };
        let upper = 2.0 * m * m;
        Bounds { upper, centre: if bounded { upper } else { f64::NEG_INFINITY } }
    }

    /// A near coupling's gate over the reach.
    pub(crate) fn term_gate(&self, unit: &Unit, reach: &Reach) -> Gate {
        let Unit::Term { a, b, from: Some(from), to: Some(to) } = unit else { unreachable!("a far static coupling's gate is units.rs'") };
        let saturation = self.bound_over(reach, super::units::EXISTS, &|_c: &super::reach::Cell, spans: &[Span]| self.term_cell(*a, *b, *from, *to, spans, reach.speed));
        Gate { saturation, sweep: 0.0 }
    }

    /// A near coupling's error at a sample: m² (p_a + p_b) with m its magnitude over its beat at the point and the sample's
    /// velocity (the beat moved by the strongest beam pair's q·v), the cell bound's formula on the point.
    pub(crate) fn term_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        let Unit::Term { a, b, from: Some(from), to: Some(to) } = unit else { unreachable!() };
        let cell = super::reach::Cell { centre: sample.at.r, half: [0.0; 3] };
        let speed = self.hbar_over_mass * crate::linalg::norm3(sample.at.kappa);
        let m2 = 0.5 * self.term_cell(*a, *b, *from, *to, &self.spans(&cell), speed).upper;
        let p = self.state_populations(sample);
        UnitError { parameter: m2, population: m2 * (p[*a] + p[*b]), temperature: m2 * (p[*a] + p[*b]) }
    }

    /// The explicit sources' recoil terms of a photon (photon.rs): the explicit state e's photon amplitudes `own`, shared among its
    /// kept drives (carrier c, beam l, ground b of the cluster) by |a_l X_l[e,b] ψ_b|², each with its response's log-gradient
    /// ∇ln σ = ∇ln a_l − ∇δ/(δ − iΓ_e/2), δ = E_e − E_b − δ_c (SPEC §1.6: the explicit state's photon takes the second moment its
    /// drives' responses give it, as the same transition folded would).
    pub(crate) fn explicit_recoil(&self, cluster: usize, at: &Point, source: usize, own: [Vec<C64>; 3], psi: &[C64]) -> Vec<([Vec<C64>; 3], [C64; 3])> {
        let super::photon::Jumps::Constant(jumps) = &self.sources[source].jumps else { return vec![] };
        let Some(e) = (GROUND..STATES).find(|&e| (0..GROUND).any(|b| (0..3).any(|a| jumps[a][(b, e)] != C64::ZERO))) else { return vec![] };
        let c = &self.clusters[cluster];
        let amplitudes = self.light.amplitudes(at.r);
        let h = self.far_hamiltonian(&amplitudes, &self.running());
        let gamma = self.decay[(e - GROUND, e - GROUND)].re;
        let mut drives = vec![];
        for (ci, cc) in self.carriers.iter().enumerate() {
            for (l, &bl) in cc.beams.iter().enumerate() {
                for (pb, &b) in c.states[..c.ground].iter().enumerate() {
                    if cc.couplings[l][(e, b)] == C64::ZERO || !self.drive_kept(ci, e, b) {
                        continue;
                    }
                    let a = &amplitudes[bl];
                    let weight = (a.value * cc.couplings[l][(e, b)] * psi[pb]).norm_sqr();
                    if weight == 0.0 {
                        continue;
                    }
                    let delta = h.value[(e, e)].re - h.value[(b, b)].re - cc.detuning;
                    let g: [C64; 3] = std::array::from_fn(|i| a.gradient[i] / a.value - C64::real(h.gradient[i][(e, e)].re - h.gradient[i][(b, b)].re) / C64::new(delta, -gamma / 2.0));
                    drives.push((weight, g));
                }
            }
        }
        let total: f64 = drives.iter().map(|d| d.0).sum();
        drives
            .into_iter()
            .map(|(w, g)| {
                let share = (w / total).sqrt();
                (own.clone().map(|v| v.into_iter().map(|z| z * share).collect()), g)
            })
            .collect()
    }
}

/// A coupling of two ground states in canonical form (`Unit::Term`): the pair (a ← b absorbing `from`, emitting `to`) and its
/// adjoint (b ← a absorbing `to`, emitting `from`) are one unit, named a < b, or for a = b with from < to.
pub(crate) fn term_unit(a: usize, b: usize, from: Option<Component>, to: Option<Component>) -> Unit {
    if a < b || (a == b && from <= to) {
        Unit::Term { a, b, from, to }
    } else {
        Unit::Term { a: b, b: a, from: to, to: from }
    }
}
