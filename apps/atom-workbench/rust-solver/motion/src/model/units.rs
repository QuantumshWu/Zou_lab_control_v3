//! Units, their existence gates, their measured errors (SPEC §1.5, §5.2).
//!
//! # The gates
//! A fold of a coupling V between two blocks a separation `sep` apart has its invariant subspace where m = ‖V‖/sep < ½
//! (Stewart 1973, internal.md 1B.1), i.e. S = 2m² < ½ (`EXISTS`); a moving atom's fold is still the instantaneous one's to first
//! order where a = |dΔE/dt|/|ΔE|² < ½ -- the gate at the same relative error as the static one (internal.md 1B.1), not an
//! existence theorem. Both are provable bounds over the reach (reach.rs): the coupling's coherent peak, every beam at the
//! largest amplitude its cell allows and all in phase, over the separation's least, the energies' spread and every Doppler
//! shift of the reach's speeds taken against it. Neither condition implies the other; a fold needs both (SPEC §1.5).
//! The separation is the distance of the two blocks' diagonals less the spread of what moves their spectra (Weyl): the far
//! light's static forms and the near carriers' couplings at their peaks (`Model::spread`).
//!
//! # The errors (SPEC §1.5), measured at a trajectory state
//! A unit's small parameter at the state's point and momentum, mapped to the outputs by its leading order: a far component's
//! and a far beat's fold mix m² (populations, and T relative); a coupling of two ground states mixes m²(p_a + p_b); an
//! excited static coupling m²(p_e + p_f); a near drive's fold, its ground state's S_b on the populations and √2 S_b on T
//! (eliminate.rs); a photon coherence m²√(p_i p_j) (photon.rs).
//!
//! # What is no unit
//! Cross terms of a co- and a counter-rotating component, and of a near and a far one, beat at an optical frequency or at
//! the far detuning: their mixing is the product of the two components' (|V||V'|/(Δ Δ') = m m'), below either one's m² that
//! the far units book. They are folded and dropped, booked under the far units.

use super::cluster::Edge;
use super::reach::{distance, doppler, lipschitz, Bounds, Cell, Span};
use crate::linalg::norm3;
use super::{Component, Gate, Model, Reach, Sample, Unit, UnitError, Wave};
use crate::linalg::{Mat, C64};
use crate::request::{GROUND, STATES};

/// Where a fold exists (module documentation): S = 2m² < ½ and a < ½.
pub(crate) const EXISTS: f64 = 0.5;

impl Gate {
    /// Whether the fold exists over the reach: both bounds below ½.
    pub fn exists(&self) -> bool {
        self.saturation < EXISTS && self.sweep < EXISTS
    }
}

/// A coupling's mixing m = size/(distance − spread), infinite where nothing separates.
fn mixing(size: f64, distance: f64, spread: f64) -> f64 {
    let d = distance - spread;
    if size == 0.0 {
        0.0
    } else if d > 0.0 {
        size / d
    } else {
        f64::INFINITY
    }
}

impl Model {
    /// Whether a component is folded far.
    pub(crate) fn folded(&self, c: Component) -> bool {
        self.tones[c.frequency].folded[c.counter as usize]
    }

    /// The spread of what moves the blocks' spectra (module documentation, # The gates): the far static forms' coherent peaks
    /// and the near carriers' couplings at theirs (rad/µs).
    pub(crate) fn spread(&self) -> f64 {
        let far: f64 = self.tones.iter().map(|t| t.pairs.iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        let near: f64 = self.carriers.iter().map(|c| self.tones[c.frequency].couplings[0].iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        far + near
    }

    /// The far light's energies' Lipschitz bound (rad/µs/µm): Σ_ω Σ_l'l ‖P_l'l‖ (L_l' + L_l), |∇(a_l'* a_l)| ≤ L_l' + L_l.
    pub(crate) fn energy_slope(&self) -> f64 {
        let mut out = 0.0;
        for (f, t) in self.light.frequencies.iter().zip(&self.tones) {
            let n = f.beams.len();
            for j in 0..n {
                for i in 0..n {
                    let l = lipschitz(&self.light.beams[f.beams[j]]) + lipschitz(&self.light.beams[f.beams[i]]);
                    out += t.pairs[j * n + i].frobenius() * l;
                }
            }
        }
        out
    }

    /// A far component's static gate, over all space: m = Σ_l ‖X_l‖/(min_{e,b} |E_e − E_b − ν| − spread) over its coupled pairs,
    /// every beam at its peak |a_l| ≤ 1 and in phase; S = 2m². Its sweep needs a reach (`Model::gates`).
    pub(crate) fn far_gate(&self, c: Component) -> Gate {
        let m = self.far_mixing(c, 1.0);
        Gate { saturation: 2.0 * m * m, sweep: 0.0 }
    }

    /// A far component's mixing with its beams' amplitudes bounded by `amplitude` (1: anywhere; the point's: measured).
    fn far_mixing_at(&self, c: Component, amplitudes: &[f64]) -> f64 {
        let (f, tone) = (&self.light.frequencies[c.frequency], &self.tones[c.frequency]);
        let side = c.counter as usize;
        let size: f64 = tone.couplings[side].iter().zip(&f.beams).map(|(m, &l)| m.frobenius() * amplitudes[l]).sum();
        mixing(size, self.far_distance(c), self.spread())
    }
    fn far_mixing(&self, c: Component, amplitude: f64) -> f64 {
        self.far_mixing_at(c, &vec![amplitude; self.light.beams.len()])
    }
    /// min_{e,b} |E_e − E_b − ν| over the pairs a component couples.
    fn far_distance(&self, c: Component) -> f64 {
        let tone = &self.tones[c.frequency];
        let nu = self.omega(&self.axes.component(&self.light, c));
        let couplings = &tone.couplings[c.counter as usize];
        let mut d = f64::INFINITY;
        for e in GROUND..STATES {
            for b in 0..GROUND {
                if couplings.iter().any(|m| m[(e, b)] != C64::ZERO) {
                    d = d.min((self.energies[e] - self.energies[b] - nu).abs());
                }
            }
        }
        d
    }

    /// A far component's sweep bound over the reach: a ≤ (v L_E + |k| (ħ/m) L_E)/sep², the energies' slope L_E along the
    /// reach's speeds and the Doppler shift's rate under the largest force (`energy_slope`).
    fn far_sweep(&self, c: Component, reach: &Reach) -> f64 {
        let k = self.light.frequencies[c.frequency].beams.iter().map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max);
        let slope = self.energy_slope();
        let sep = self.far_distance(c) - self.spread();
        if sep <= 0.0 {
            return f64::INFINITY;
        }
        (reach.speed * slope + k * self.hbar_over_mass * slope) / (sep * sep)
    }

    /// The cross form of two far components of different frequencies (`Unit::FarBeat`), per beam pair (l₂ of `second`,
    /// l₁ of `first`, row-major over `second`'s beams): absorbing `first`, emitting `second`, at e^{−i(ν₁ − ν₂)t} --
    /// C(X_l₂, X_l₁) and the outside forms' mean cross term (far.rs, # The beat of two far components) for two co-rotating
    /// components, C(R_l₂, R_l₁) for two counter-rotating ones (their coefficients a_l₂* a_l₁ and a_l₂ a_l₁*).
    pub(crate) fn far_cross(&self, first: Component, second: Component) -> Vec<Mat> {
        let (f1, f2) = (&self.light.frequencies[first.frequency], &self.light.frequencies[second.frequency]);
        let (t1, t2) = (&self.tones[first.frequency], &self.tones[second.frequency]);
        let side = first.counter as usize;
        let nu1 = self.omega(&self.axes.component(&self.light, first));
        let nu2 = self.omega(&self.axes.component(&self.light, second));
        let mut out = vec![];
        for (j, x2) in t2.couplings[side].iter().enumerate() {
            for (i, x1) in t1.couplings[side].iter().enumerate() {
                let mut m = super::far::van_vleck(&self.energies, x2, nu2, x1, nu1);
                if !first.counter {
                    for p in 0..3 {
                        for q in 0..3 {
                            let w = f2.fields[j][p].conj() * f1.fields[i][q] * 0.5;
                            if w != C64::ZERO {
                                m.add_scaled(w, &t1.outside[p][q]);
                                m.add_scaled(w, &t2.outside[p][q]);
                            }
                        }
                    }
                }
                out.push(m);
            }
        }
        out
    }

    /// The coherent peak of the far light's static coupling between two states, Σ_ω Σ_l'l |[P_l'l]_ab|.
    pub(crate) fn static_size(&self, a: usize, b: usize) -> f64 {
        self.tones.iter().map(|t| t.pairs.iter().map(|m| m[(a, b)].abs()).sum::<f64>()).sum()
    }

    /// Whether a state is an explicit excited state of this model.
    pub(crate) fn explicit(&self, s: usize) -> bool {
        s >= GROUND && self.clusters.iter().any(|c| c.states.contains(&s))
    }

    /// The candidate units of this model (model.rs `Unit`): every component's fold, the beats of two folded far components of one
    /// rotation, the far light's static couplings of two ground states and of an explicit excited state, and the near carriers'
    /// drives, couplings and photon coherences (eliminate.rs, photon.rs).
    pub fn units(&self) -> Vec<Unit> {
        let mut out = vec![];
        let components = self.components();
        for &c in &components {
            out.push(Unit::Far { component: c });
        }
        for (i, &first) in components.iter().enumerate() {
            for &second in &components[i + 1..] {
                if first.counter == second.counter && first.frequency != second.frequency && self.folded(first) && self.folded(second) {
                    out.push(Unit::FarBeat { first, second });
                }
            }
        }
        for a in 0..GROUND {
            for b in a + 1..GROUND {
                if self.static_size(a, b) > 0.0 {
                    out.push(Unit::Term { a, b, from: None, to: None });
                }
            }
        }
        for e in (GROUND..STATES).filter(|&e| self.explicit(e)) {
            for f in (GROUND..STATES).filter(|&f| f != e && !(self.explicit(f) && f < e)) {
                if self.static_size(e, f) > 0.0 {
                    out.push(Unit::Excited { excited: e, other: f });
                }
            }
        }
        out.extend(self.near_units());
        out.extend(self.coherence_units());
        out
    }

    /// A static coupling's (two ground states, or two excited states) mixing bound over a cell: its coherent peak there over the
    /// least separation of the two states' energies there, less the Doppler shift of its parts' largest wave vector at the
    /// reach's speed.
    fn static_cell(&self, a: usize, b: usize, spans: &[Span], speed: f64) -> Bounds {
        let mut size = 0.0;
        let (mut at, mut slack) = (self.energies[a] - self.energies[b], 0.0);
        let mut size_centre = C64::ZERO;
        let mut wave = 0.0_f64;
        for (f, t) in self.light.frequencies.iter().zip(&self.tones) {
            let n = f.beams.len();
            for j in 0..n {
                for i in 0..n {
                    let (lj, li) = (f.beams[j], f.beams[i]);
                    let p = &t.pairs[j * n + i];
                    let pair = super::reach::pair_span(spans, lj, li);
                    let (c, moved) = (pair.centre, pair.radius);
                    size += p[(a, b)].abs() * spans[lj].most * spans[li].most;
                    size_centre += c * p[(a, b)];
                    let d = p[(a, a)] - p[(b, b)];
                    at += (c * d).re;
                    slack += d.abs() * moved;
                    if p[(a, b)] != C64::ZERO {
                        let k = self.wave_value(&self.beam_wave(li).plus(-1, &self.beam_wave(lj)));
                        wave = wave.max(doppler(k, speed));
                    }
                }
            }
        }
        let sep = distance(at - slack, at + slack) - wave - self.near_spread(a, b, spans);
        let upper = mixing(size, sep, 0.0);
        let centre = if spans.iter().all(|s| s.rho.is_finite()) { mixing(size_centre.abs(), at.abs() - wave, 0.0) } else { f64::NEG_INFINITY };
        Bounds { upper: 2.0 * upper * upper, centre: 2.0 * centre * centre }
    }

    /// Each unit's existence gate over the reach (SPEC §1.5), provable bounds at this model's clusters and dressing.
    pub fn gates(&self, units: &[Unit], reach: &Reach) -> Vec<Gate> {
        units
            .iter()
            .map(|u| match u {
                Unit::Far { component } => Gate { saturation: self.far_gate(*component).saturation, sweep: self.far_sweep(*component, reach) },
                Unit::FarBeat { first, second } => {
                    let size: f64 = self.far_cross(*first, *second).iter().map(|m| m.frobenius()).sum();
                    let beat = self.omega(&self.axes.component(&self.light, *first).plus(-1, &self.axes.component(&self.light, *second)));
                    let m = mixing(size, beat.abs(), self.spread());
                    Gate { saturation: 2.0 * m * m, sweep: 0.0 }
                }
                Unit::Term { a, b, from: None, to: None } | Unit::Excited { excited: a, other: b } => {
                    let (a, b) = (*a, *b);
                    let saturation = self.bound_over(reach, EXISTS, &|_cell: &Cell, spans: &[Span]| self.static_cell(a, b, spans, reach.speed));
                    Gate { saturation, sweep: 0.0 }
                }
                _ => self.near_gate(u, reach),
            })
            .collect()
    }

    /// Each folded unit's errors at each sample, [unit][sample]: its small parameter at the sample's point mapped to the outputs
    /// (module documentation, # The errors); zero for a kept unit.
    pub fn errors(&self, units: &[Unit], samples: &[Sample]) -> Vec<Vec<UnitError>> {
        let zero = UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
        units
            .iter()
            .map(|u| {
                samples
                    .iter()
                    .map(|s| {
                        if self.kept.contains(u) {
                            return zero;
                        }
                        match u {
                            Unit::Far { component } => {
                                let amplitudes: Vec<f64> = self.light.amplitudes(s.at.r).iter().map(|a| a.value.abs()).collect();
                                let m = self.far_mixing_at(*component, &amplitudes);
                                UnitError { parameter: m * m, population: m * m, temperature: m * m }
                            }
                            Unit::FarBeat { first, second } => {
                                let amplitudes = self.light.amplitudes(s.at.r);
                                let (f1, f2) = (&self.light.frequencies[first.frequency], &self.light.frequencies[second.frequency]);
                                let cross = self.far_cross(*first, *second);
                                let mut size = 0.0;
                                for (j, &l2) in f2.beams.iter().enumerate() {
                                    for (i, &l1) in f1.beams.iter().enumerate() {
                                        size += cross[j * f1.beams.len() + i].frobenius() * amplitudes[l2].value.abs() * amplitudes[l1].value.abs();
                                    }
                                }
                                let beat = self.omega(&self.axes.component(&self.light, *first).plus(-1, &self.axes.component(&self.light, *second)));
                                let m = mixing(size, beat.abs(), self.spread());
                                UnitError { parameter: m * m, population: m * m, temperature: m * m }
                            }
                            Unit::Term { a, b, from: None, to: None } | Unit::Excited { excited: a, other: b } => {
                                let cell = Cell { centre: s.at.r, half: [0.0; 3] };
                                let speed = self.hbar_over_mass * norm3(s.at.kappa);
                                let m2 = 0.5 * self.static_cell(*a, *b, &self.spans(&cell), speed).centre.max(0.0);
                                let p = self.state_populations(s);
                                UnitError { parameter: m2, population: m2 * (p[*a] + p[*b]), temperature: m2 }
                            }
                            _ => self.near_error(u, s),
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// A sample's populations of the model basis' states on its cluster (|ψ_s|²/‖ψ‖²; zero off its cluster).
    pub(crate) fn state_populations(&self, s: &Sample) -> [f64; STATES] {
        let c = &self.clusters[s.cluster];
        let total: f64 = s.psi.iter().map(|z| z.norm_sqr()).sum();
        let mut p = [0.0; STATES];
        for (i, &x) in c.states.iter().enumerate() {
            p[x] = s.psi[i].norm_sqr() / total;
        }
        p
    }

    /// A kept unit's edge (cluster.rs): the states it joins, their frame difference, the strongest part's wave vector and its
    /// strength.
    pub(crate) fn edge(&self, unit: &Unit) -> Option<Edge> {
        match unit {
            Unit::Term { a, b, from: None, to: None } | Unit::Excited { excited: a, other: b } => {
                let (a, b) = (*a, *b);
                let mut best = (0.0, self.no_wave());
                for (f, t) in self.light.frequencies.iter().zip(&self.tones) {
                    let n = f.beams.len();
                    for j in 0..n {
                        for i in 0..n {
                            let size = t.pairs[j * n + i][(a, b)].abs();
                            if size > best.0 {
                                best = (size, self.beam_wave(f.beams[i]).plus(-1, &self.beam_wave(f.beams[j])));
                            }
                        }
                    }
                }
                Some(Edge { a, b, beat: self.axes.zero(), wave: best.1, strength: self.static_size(a, b) })
            }
            Unit::Far { .. } | Unit::FarBeat { .. } | Unit::Coherence { .. } => None,
            _ => self.near_edge(unit),
        }
    }

    /// The exact wave vector of a beam pair's term, absorbing `absorbed` and emitting `emitted`: k_absorbed − k_emitted.
    pub(crate) fn pair_wave(&self, emitted: usize, absorbed: usize) -> Wave {
        self.beam_wave(absorbed).plus(-1, &self.beam_wave(emitted))
    }

}

