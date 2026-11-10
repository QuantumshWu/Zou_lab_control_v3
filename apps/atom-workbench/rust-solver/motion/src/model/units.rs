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
//! Every fold has both gates: its ΔE is the distance of the two levels it folds across, and on a moving atom dΔE/dt = v·∇ΔE +
//! q·v̇ -- the levels' slopes along the atom's velocity, and the Doppler shift of the fold's wave vector q under the atom's
//! acceleration --, so a ≤ (v L + q a_max)/sep² (`swept`): a drive's per excited state (eliminate.rs `Model::drive_bound`); a
//! coupling's of two states with their own difference's slope and the acceleration in their clusters (`Model::static_bound`,
//! eliminate.rs `Model::term_bound`); a far component's, a far beat's and a photon coherence's, whose static bounds are closed
//! forms over all space, with the slope of any two levels' difference over the reach's cells (`Model::swept_over`), q the wave
//! vectors of the beams they absorb and emit. A carrier's dressing (`Unit::Dressing`) folds nothing -- its T̄ averages the
//! interference of its polarization units in the others' denominators, with no energy denominator of its own --: it has no
//! gate, only its measured error (eliminate.rs `Model::dressing_error`).
//!
//! # The errors (SPEC §1.5), measured at a trajectory state
//! A unit's small parameter at the state's point and momentum, mapped to the outputs by its leading order, the populations and T
//! by the same weight -- the populations of the states it couples, so that a unit whose states are empty books nothing: a far
//! component's and a far beat's fold mix m² (every ground state is the far light's); a coupling of two ground states mixes
//! m²(p_a + p_b); an excited static coupling m²(p_e + p_f); a near drive's fold, its ground state's S_b p_b on the populations and
//! √2 S_b p_b on T (eliminate.rs); a carrier's dressing its relative rate error on the state's rates (eliminate.rs); a photon
//! coherence m²√(p_i p_j) (photon.rs).
//!
//! # What is no unit
//! Cross terms of a co- and a counter-rotating component, and of a near and a far one, beat at an optical frequency or at
//! the far detuning: their mixing is the product of the two components' (|V||V'|/(Δ Δ') = m m'), below either one's m² that
//! the far units book. They are folded and dropped, booked under the far units.

use super::cluster::Edge;
use super::reach::{distance, Region};
use crate::linalg::norm3;
use super::{Cluster, Component, Gate, Model, Reach, Sample, Unit, UnitError, Wave};
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
pub(crate) fn mixing(size: f64, distance: f64, spread: f64) -> f64 {
    let d = distance - spread;
    if size == 0.0 {
        0.0
    } else if d > 0.0 {
        size / d
    } else {
        f64::INFINITY
    }
}

/// A fold's motional parameter a = |dΔE/dt|/|ΔE|² from the bound `rate` of |dΔE/dt| and the least `separation` of |ΔE| (module
/// documentation, # The gates): zero where nothing moves, infinite where nothing separates what does.
pub(crate) fn swept(rate: f64, separation: f64) -> f64 {
    if rate == 0.0 {
        0.0
    } else if separation > 0.0 {
        rate / (separation * separation)
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

    /// A far component's static bound, over all space: m = Σ_l ‖X_l‖/(min_{e,b} |E_e − E_b − ν| − spread) over its coupled pairs,
    /// every beam at its peak |a_l| ≤ 1 and in phase; S = 2m². Its sweep needs a reach (`Model::gates`).
    pub(crate) fn far_saturation(&self, c: Component) -> f64 {
        let m = self.far_mixing(c, 1.0);
        2.0 * m * m
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

    /// The bound over a region of the slope of a state's level under the near light (rad/µs/µm), made once per region and ground
    /// state: a ground state's shifts by the carriers (eliminate.rs `Model::shift_slope`). An explicit excited state's shift by
    /// the folded drives is not in it, as it is not in the static gates' separations (eliminate.rs `Model::near_spread`).
    pub(crate) fn near_slope(&self, s: usize, region: &Region) -> f64 {
        if s < GROUND {
            *region.slopes[s].get_or_init(|| (0..self.carriers.len()).map(|c| self.shift_slope(c, s, region, false)).sum())
        } else {
            0.0
        }
    }

    /// The bound over a region of an atom's acceleration (µm/µs²) in the clusters `within` picks: ħ/m times the force's bound
    /// (eliminate.rs `Model::force_over`).
    pub(crate) fn acceleration(&self, region: &Region, within: &dyn Fn(&Cluster) -> bool) -> f64 {
        self.hbar_over_mass * (0..self.clusters.len()).filter(|&c| within(&self.clusters[c])).map(|c| self.force_over(c, region)).fold(0.0, f64::max)
    }

    /// The sweep bound over a region of a fold between any two states' levels, of wave vector at most `wave` and closed-form
    /// separation `separation` (a far component's, a far beat's, a photon coherence's; module documentation, # The gates):
    /// `swept` of v L + q a_max over the separation less the Doppler shift q v -- L twice the largest slope of a state's level (the
    /// far light's energy, a ground state's near shifts), a_max the largest acceleration in any cluster, both made once per
    /// region.
    fn swept_over(&self, region: &Region, wave: f64, separation: f64) -> f64 {
        let (slope, acceleration) = *region.moving.get_or_init(|| {
            let slope = (0..STATES).map(|s| self.energy_slope_over(s, s, region) + self.near_slope(s, region)).fold(0.0, f64::max);
            (2.0 * slope, self.acceleration(region, &|_| true))
        });
        swept(region.speed * slope + wave * acceleration, separation - wave * region.speed)
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
    /// drives, couplings and photon coherences (eliminate.rs, photon.rs) -- those it folds: a unit kept, or one its clusters hold
    /// for free, is no candidate (`Model::in_generator`).
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
        out.retain(|u| !self.in_generator(u));
        out
    }

    /// Whether a unit's coupling is in this model's generators as it is (SPEC §1.4), nothing of it folded: kept, or held for free
    /// -- a static coupling of two states of one cluster (the assembly takes every far static entry within a cluster,
    /// assemble.rs), a drive or a near coupling within one cluster at a residual beat of exactly zero (eliminate.rs
    /// `Model::drive_kept`, `Model::term_included`).
    fn in_generator(&self, u: &Unit) -> bool {
        let together = |a: usize, b: usize| self.clusters.iter().position(|c| c.states.contains(&a) && c.states.contains(&b));
        self.kept.contains(u)
            || match u {
                Unit::Drive { carrier, excited, ground } => self.drive_kept(self.carrier_index(carrier.frequency), *excited, *ground),
                Unit::Term { a, b, from: None, to: None } | Unit::Excited { excited: a, other: b } => together(*a, *b).is_some(),
                Unit::Term { a, b, from: Some(from), to: Some(to) } => together(*a, *b).is_some_and(|k| {
                    let c = &self.clusters[k];
                    let place = |s: usize| c.states.iter().position(|&x| x == s).unwrap();
                    let beat = self.carriers[self.carrier_index(to.frequency)].beat.plus(-1, &self.carriers[self.carrier_index(from.frequency)].beat);
                    self.term_included(k, place(*a), place(*b), *from, *to, &beat)
                }),
                _ => false,
            }
    }

    /// A static coupling's (two ground states, or two excited states) bounds over a region, its saturation S = 2m² and its sweep
    /// a: its value |Σ_l'l c_l'l [P_l'l]_ab| at the centre plus Σ |[P_l'l]_ab| times each pair's radius (reach.rs `PairSpan`), at
    /// most every beam at its largest amplitude and in phase, over the least separation of the two states' energies there, less
    /// the Doppler shift of its parts' largest wave vector q at the region's speeds and the near light's differential shift; and
    /// `swept` of v |∇(E_a − E_b)| + q a_max over that separation (the two states' far energies and near shifts, the acceleration
    /// in their clusters). On a point, its measured values there. Made once per region (reach.rs `Region::term`).
    fn static_bound(&self, a: usize, b: usize, region: &Region) -> (f64, f64) {
        region.term((a, b, None, None), || self.static_bound_made(a, b, region))
    }
    fn static_bound_made(&self, a: usize, b: usize, region: &Region) -> (f64, f64) {
        let (mut centre, mut moved, mut blind) = (C64::ZERO, 0.0, 0.0);
        let (mut at, mut slack) = (self.base[(a, a)].re - self.base[(b, b)].re, 0.0);
        let mut wave = 0.0_f64;
        for (f, (t, pairs)) in self.light.frequencies.iter().zip(self.tones.iter().zip(&region.pairs)) {
            let n = f.beams.len();
            for (x, (p, pair)) in t.pairs.iter().zip(pairs).enumerate() {
                let (lj, li) = (f.beams[x / n], f.beams[x % n]);
                let v = p[(a, b)];
                centre += pair.centre * v;
                moved += v.abs() * pair.radius;
                blind += v.abs() * region.spans[lj].most * region.spans[li].most;
                let d = p[(a, a)] - p[(b, b)];
                at += (pair.centre * d).re;
                slack += d.abs() * pair.radius;
                if v != C64::ZERO {
                    wave = wave.max(norm3(self.wave_value(&self.pair_wave(lj, li))));
                }
            }
        }
        let size = (centre.abs() + moved).min(blind);
        let separation = distance(at - slack, at + slack) - wave * region.speed - self.near_spread(a, b, region);
        let m = mixing(size, separation, 0.0);
        if size == 0.0 {
            return (0.0, 0.0);
        }
        let slope = self.energy_slope_over(a, b, region) + self.near_slope(a, region) + self.near_slope(b, region);
        let acceleration = self.acceleration(region, &|c| c.states.contains(&a) || c.states.contains(&b));
        (2.0 * m * m, swept(region.speed * slope + wave * acceleration, separation))
    }

    /// Each unit's existence gate over the reach (SPEC §1.5; module documentation, # The gates), provable bounds at this model's
    /// clusters and dressing, proven together over the reach's cells (reach.rs `Model::bounds_over`): the drives' (per ground state
    /// its folded drives' saturation and sweep) and the couplings' both bounds; of a far component, a far beat and a photon
    /// coherence, whose saturation is a closed form over all space, the sweep. A dressing has no gate.
    pub fn gates(&self, units: &[Unit], reach: &Reach) -> Vec<Gate> {
        let mut quantities: Vec<Box<dyn Fn(&Region) -> f64 + '_>> = vec![];
        let mut grounds: Vec<Option<usize>> = vec![None; GROUND];
        // The largest wave number of a frequency's beams; the static bounds' spread.
        let most = |frequency: usize| self.light.frequencies[frequency].beams.iter().map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max);
        let spread = self.spread();
        // Per unit its saturation where that is a closed form, and where its bounds over the reach lie among the quantities: the
        // saturation's (unless closed) and then the sweep's.
        let asked: Vec<(Option<f64>, Option<usize>)> = units
            .iter()
            .map(|u| {
                let first = quantities.len();
                let (closed, wave, separation) = match *u {
                    Unit::Drive { ground, .. } => {
                        let at = *grounds[ground].get_or_insert_with(|| {
                            let cluster = self.membership[ground];
                            quantities.push(Box::new(move |r: &Region| self.drive_bound(ground, cluster, r).0));
                            quantities.push(Box::new(move |r: &Region| self.drive_bound(ground, cluster, r).1));
                            first
                        });
                        return (None, Some(at));
                    }
                    Unit::Term { a, b, from: None, to: None } | Unit::Excited { excited: a, other: b } => {
                        quantities.push(Box::new(move |r: &Region| self.static_bound(a, b, r).0));
                        quantities.push(Box::new(move |r: &Region| self.static_bound(a, b, r).1));
                        return (None, Some(first));
                    }
                    Unit::Term { a, b, from: Some(from), to: Some(to) } => {
                        quantities.push(Box::new(move |r: &Region| self.term_bound(a, b, from, to, r).0));
                        quantities.push(Box::new(move |r: &Region| self.term_bound(a, b, from, to, r).1));
                        return (None, Some(first));
                    }
                    Unit::Term { .. } => unreachable!("a coupling names both of its carriers or neither"),
                    Unit::Dressing { .. } => return (None, None),
                    Unit::Far { component } => (self.far_saturation(component), most(component.frequency), self.far_distance(component) - spread),
                    Unit::FarBeat { first, second } => {
                        let size: f64 = self.far_cross(first, second).iter().map(|m| m.frobenius()).sum();
                        let beat = self.omega(&self.axes.component(&self.light, first).plus(-1, &self.axes.component(&self.light, second))).abs();
                        let m = mixing(size, beat, spread);
                        (2.0 * m * m, most(first.frequency) + most(second.frequency), beat - spread)
                    }
                    Unit::Coherence { ref first, ref second, .. } => {
                        let wave = 2.0 * (0..self.light.frequencies.len()).map(most).fold(0.0, f64::max);
                        (self.coherence_saturation(u), wave, self.omega(&first.plus(-1, second)).abs() - spread)
                    }
                };
                quantities.push(Box::new(move |r: &Region| self.swept_over(r, wave, separation)));
                (Some(closed), Some(first))
            })
            .collect();
        let refs: Vec<&dyn Fn(&Region) -> f64> = quantities.iter().map(|q| q.as_ref() as &dyn Fn(&Region) -> f64).collect();
        let proven = self.bounds_over(reach, EXISTS, &refs);
        asked
            .into_iter()
            .map(|(closed, at)| match (closed, at) {
                (Some(saturation), Some(i)) => Gate { saturation, sweep: proven[i] },
                (None, Some(i)) => Gate { saturation: proven[i], sweep: proven[i + 1] },
                (_, None) => Gate { saturation: 0.0, sweep: 0.0 },
            })
            .collect()
    }

    /// Each candidate unit's errors at each sample (`Model::units`), [unit][sample]: its small parameter at the sample's point
    /// mapped to the outputs (module documentation, # The errors).
    pub fn errors(&self, units: &[Unit], samples: &[Sample]) -> Vec<Vec<UnitError>> {
        units
            .iter()
            .map(|u| {
                samples
                    .iter()
                    .map(|s| match u {
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
                            let m2 = 0.5 * self.static_bound(*a, *b, &self.point_region(s.at.r, self.hbar_over_mass * norm3(s.at.kappa))).0;
                            let p = self.state_populations(s);
                            UnitError { parameter: m2, population: m2 * (p[*a] + p[*b]), temperature: m2 * (p[*a] + p[*b]) }
                        }
                        _ => self.near_error(u, s),
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
            Unit::Far { .. } | Unit::FarBeat { .. } | Unit::Coherence { .. } | Unit::Dressing { .. } => None,
            _ => self.near_edge(unit),
        }
    }

    /// The exact wave vector of a beam pair's term, absorbing `absorbed` and emitting `emitted`: k_absorbed − k_emitted.
    pub(crate) fn pair_wave(&self, emitted: usize, absorbed: usize) -> Wave {
        self.beam_wave(absorbed).plus(-1, &self.beam_wave(emitted))
    }

}
