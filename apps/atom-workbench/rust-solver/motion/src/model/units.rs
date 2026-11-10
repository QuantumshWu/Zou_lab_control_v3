//! Units, their existence gates, their measured errors (SPEC §1.5, §5.2).
//!
//! # One rule
//! No light is far or near. Every drive -- one component's coupling of one ground state to one excited state -- and every
//! second-order coupling of two states at one beat is a unit of its own, judged alone by one statement at its own distance:
//! a fold of a coupling of size `size` across a distance `distance` exists where m = size/distance < ½, S = 2m² < ½
//! (Stewart 1973, internal.md 1B.1; `EXISTS`), and a moving atom's fold is still the instantaneous one's to first order
//! where a = rate/distance² < ½, rate = |dΔE/dt| -- the gate at the same relative error as the static one (internal.md
//! 1B.1), not an existence theorem. `gate` is that statement, the only one; a kind of unit supplies its own size, its least
//! distance over the reach and its rate, all provable bounds over the reach (reach.rs): a coupling's coherent peak, every
//! beam at the largest amplitude its cell allows and all in phase, over the distance's least, the energies' spread and
//! every Doppler shift of the reach's speeds taken against it. Neither condition implies the other; a fold needs both.
//!
//! # Levels
//! A drive is taken at one of three levels (model.rs `Level`), each the more exact one of the one before, and what decides
//! is the accuracy its own denominator needs -- never its frequency, its line or its beam:
//! - folded with the bare levels (`Unit::Bare`): its response is V/(Δ₀ − iΓ/2), Δ₀ = E_e − E_g − ν of H0, a constant
//!   (far.rs). At the atom the denominator is Δ₀ + δΔ(r, v) − iΓ/2 -- δΔ the shift of its two levels by the other light
//!   and the Doppler shift and recoil of its photon --, so the bare response is the first term of a series in
//!   ε = δΔ/(Δ₀ − iΓ/2), and the bare levels are a perturbation theory about H0: what it folds across the bare distance
//!   |Δ₀ − iΓ/2| is everything between the drive's two states that H0 does not hold, the coupling and the movement of the
//!   two levels together. Its own gate is therefore the one gate of the size |V| + |δΔ|, whichever way the levels move,
//!   across the bare distance: below it the fold exists (|V| against what is left of the distance, |Δ₀ − iΓ/2| − |δΔ|) and
//!   the series converges with |ε| < ½; failing, the unit is kept and the drive is folded with the levels at the atom.
//!   Inside the gate its error is the series' first order, measured at the atoms (# The errors).
//! - folded with the levels at the atom (`Unit::Drive`, eliminate.rs): its distance over a region is its actual
//!   denominator's modulus there, at least. Its gate failing, the unit is kept and the drive is explicit.
//! - explicit: in the generator as it is.
//! A drive has exactly one level, the highest its kept units give it, and one more where its two states are one cluster's
//! at a residual beat of exactly zero (`Model::new`); a carrier is a component with a drive not bare, never listed.
//!
//! # The tail rule
//! The folds of one ground state's drives share its population: the fold of a set of them exists where their shares'
//! sum is below ½, each share across its drive's actual distance -- whatever level it is computed at. A drive's gate is
//! therefore the sum of its ground state's folded shares that are not larger than its own -- so that the strongest fails
//! first and is kept first, drives of one share fail together, and a distant line is never kept for a near one's sake --,
//! and a bare drive's takes its own gate's value for its share: that its bare fold exists is its own matter and no other
//! drive's. The order is one fixed order per ground state, by each drive's share with every
//! beam at its peak and the bare levels (`Model::ranked`): a region's shares in a region's order would not be monotone in
//! the region, and the proof over the reach needs that; which of two is kept first is a choice of cost, each gate a
//! statement about a fixed set. Two shares within the rounding of their own arithmetic are one place in the order (drives
//! equal by a symmetry of the request differ in the last bits of their couplings), and its drives of one level take
//! their largest sweep.
//!
//! # A coupling of two states (`Unit::Term`)
//! Everything the folds leave between two states a ← b at one exact beat β (∝ e^{iβt}) is one unit: the constant forms'
//! entry (far.rs `constant`: one frequency's static form at β = 0, with the two states' own coupling when every light is
//! off; two frequencies' cross form at their beat; on the ground states and on the excited ones) and the part of the
//! drives folded at the atom (eliminate.rs). Its size is the
//! bound of that entry, its distance |β + E_a − E_b| over the region less the Doppler shift of its beam pairs' wave
//! vector and the carriers' differential shift of the two states, its rate v|∇(E_a − E_b)| + q a_max. Kept, the entry is
//! the generator's and the two states are one cluster's, their frames the beat apart (`Model::edges`); folded, it is
//! dropped and its error booked. An entry at a residual beat of exactly zero in a cluster's frames is the generator's
//! without a unit.
//!
//! # The model's domain
//! A counter-rotating drive is only ever folded with the bare levels: one kept is refused, and so is a request whose
//! counter-rotating folds do not exist by the one gate, every beam at its peak (`Model::new`). A co-rotating leg against a
//! counter-rotating one is no term of the model (far.rs, # Two frequencies' constant form).
//!
//! # The errors (SPEC §1.5), measured at a trajectory state
//! What the units' errors at one state have in common is made once for all of them (`Measured`): the beams' amplitudes and
//! the point's region, the cluster's forms at the atom there -- one elimination, or its table's interpolant where the run
//! has tables -- with the carriers' responses and the bare drives', the constant forms' energies and their slopes, the
//! force on the state. Each unit's error is then a few numbers' arithmetic, nothing built per unit; only a carrier's
//! dressing takes an elimination of its own (the one with its coherent field). A unit's small parameter at the state's
//! point and momentum is mapped to the outputs by its leading order, the populations and T by the same weight -- the
//! populations of the states it couples, so that a unit whose states are empty books nothing: a drive's fold its own
//! share s p_g on the populations and (√2 s + a) p_g on T; a bare one also 2ε times its share of the state's rates, on
//! both, and ε times its share of the state's light shifts on T (the relative errors it makes of them, as a dressing's;
//! the shares of a state's drives, folded and kept, add up to one: beside an explicit line a distant one's rate is a
//! small part of what moves the populations); a coupling of two states m²(p_a + p_b); a carrier's dressing its relative rate error on the state's rates (eliminate.rs); a photon
//! coherence m²√(p_i p_j) (photon.rs).

use super::assemble::{NearValues, Plan};
use super::cluster::Edge;
use super::eliminate::term_unit;
use super::far::EXCITED;
use super::reach::{distance, pair_span, Cell, Region, TermForm};
use super::readout::Channels;
use super::tabulate::Tables;
use super::{Beat, Cluster, Component, Gate, Level, Model, Reach, Sample, Unit, UnitError, Wave};
use crate::field::Amplitude;
use crate::linalg::norm3;
use crate::linalg::{Mat, C64};
use crate::request::{GROUND, STATES};
use crate::units::rounding;
use std::cell::OnceCell;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Where a fold exists (module documentation): S = 2m² < ½ and a < ½.
pub(crate) const EXISTS: f64 = 0.5;

impl Gate {
    /// Whether the fold exists over the reach: both bounds below ½.
    pub fn exists(&self) -> bool {
        self.saturation < EXISTS && self.sweep < EXISTS
    }
}

/// The one gate (module documentation, # One rule): of a fold of a coupling of size `size` across a distance `distance`
/// that moves at `rate`, S = 2m², m = size/distance, and a = rate/distance² -- zero where there is nothing to fold or
/// nothing moves, infinite where nothing separates what does.
pub(crate) fn gate(size: f64, distance: f64, rate: f64) -> Gate {
    let over = |x: f64, d: f64| if x == 0.0 { 0.0 } else if distance > 0.0 { x / d } else { f64::INFINITY };
    let m = over(size, distance);
    Gate { saturation: 2.0 * m * m, sweep: over(rate, distance * distance) }
}

/// What the units' errors at one trajectory state share (module documentation, # The errors): the state; the beams'
/// amplitudes; the point's region at the state's speed; the cluster's plan under the running lights and its forms at the
/// atom at the point (the Hermitian ones to first order), with each carrier's folded response there; the bare drives'
/// responses there per component (readout.rs `Model::far_channels`); the state's excitation drive by drive -- Σ |R[e, g]|²
/// p_g over its folded drives, bare ones and those at the atom, with its explicit states' populations: its rates' measure;
/// and Σ |V[e, g] R[e, g]| p_g with its kept drives' |V[e, g]| |ψ_e| |ψ_g|: its light shifts' --;
/// the levels' energies at the atom to first order ([part][state], eliminate.rs `Model::far_energies`); the force on the
/// state (rad/µs/µm); its populations; and, made when asked, its cluster's photon classes' jump operators' sizes at the
/// point (photon.rs `Model::class_sizes`) and per carrier the first-order Doppler term's size per ground place.
pub(crate) struct Measured<'a> {
    pub(crate) sample: &'a Sample,
    pub(crate) amplitudes: Vec<Amplitude>,
    pub(crate) region: Region,
    pub(crate) plan: Arc<Plan>,
    pub(crate) near: NearValues,
    pub(crate) responses: Vec<Option<Mat>>,
    pub(crate) bare: Vec<Channels>,
    pub(crate) rates: f64,
    pub(crate) shifts: f64,
    pub(crate) energies: Vec<Vec<f64>>,
    pub(crate) force: [f64; 3],
    pub(crate) populations: [f64; STATES],
    pub(crate) classes: OnceCell<Vec<[f64; 3]>>,
    pub(crate) doppler: Vec<OnceCell<Vec<f64>>>,
}

/// One drive of a ground state in the tail rule's order (module documentation, # The tail rule): its component and excited
/// state, and its place -- drives of one share within rounding share one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Ranked {
    pub(crate) component: Component,
    pub(crate) excited: usize,
    pub(crate) group: usize,
}

/// Two frequencies' constant form (far.rs `constant`): on their fields' bases and per pair of their beams.
pub(crate) struct Cross {
    pub(crate) forms: Vec<Mat>,
    pub(crate) pairs: Vec<Mat>,
}

impl Model {
    /// The spread of what moves the blocks' spectra, a bound over all space (rad/µs): the constant forms' coherent peaks
    /// and the carriers' couplings at theirs; made once (`Model::new`). The photon coherences' distance takes it (photon.rs).
    pub(crate) fn spread(&self) -> f64 {
        self.spread
    }
    pub(crate) fn spread_made(&self) -> f64 {
        let constant: f64 = self.tones.iter().map(|t| t.pairs.iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        let carriers: f64 = self.carriers.iter().map(|c| self.tones[c.frequency].couplings[0].iter().map(|m| m.frobenius()).sum::<f64>()).sum();
        constant + carriers
    }

    /// A drive's denominator with the bare levels, its modulus |E_e − E_g − ν − iΓ_e/2| (rad/µs).
    pub(crate) fn bare_distance(&self, c: Component, excited: usize, ground: usize) -> f64 {
        let detuning = self.energies[excited] - self.energies[ground] - self.tones[c.frequency].nu[c.counter as usize];
        C64::new(detuning, 0.5 * self.decay[(excited - GROUND, excited - GROUND)].re).abs()
    }

    /// A drive's share anywhere: the one gate's S with every beam at its peak and in phase, |a_l| ≤ 1, across its bare
    /// distance.
    pub(crate) fn share_anywhere(&self, c: Component, excited: usize, ground: usize) -> f64 {
        let size: f64 = self.tones[c.frequency].couplings[c.counter as usize].iter().map(|x| x[(excited, ground)].abs()).sum();
        gate(size, self.bare_distance(c, excited, ground), 0.0).saturation
    }

    /// Per ground state its drives in the tail rule's order (module documentation, # The tail rule): by their shares
    /// anywhere, the largest first; two neighbours within the sum of their shares' roundings share a place. A share's
    /// relative rounding: its n beams' moduli and their sum (each entry the request's contraction with its field, three
    /// terms; γ_{n+5}), squared; the detuning's two differences, |E_e| + |E_g| + |ν| rounded once each against the
    /// denominator Δ₀² + Γ²/4 (2|Δ₀| u (|E_e| + |E_g| + |ν|) of it at most); the decay's sum over the ground states and
    /// the quotient.
    pub(crate) fn ranked_made(&self) -> Vec<Vec<Ranked>> {
        (0..GROUND)
            .map(|g| {
                let mut of: Vec<(Component, usize, f64, f64)> = self
                    .driven
                    .iter()
                    .filter(|d| d.2 == g)
                    .map(|&(c, e, _)| {
                        let tone = &self.tones[c.frequency];
                        let (nu, gamma) = (tone.nu[c.counter as usize], self.decay[(e - GROUND, e - GROUND)].re);
                        let detuning = self.energies[e] - self.energies[g] - nu;
                        let moved = 2.0 * rounding(1) * (self.energies[e].abs() + self.energies[g].abs() + nu.abs()) * detuning.abs() / (detuning * detuning + 0.25 * gamma * gamma);
                        let share = self.share_anywhere(c, e, g);
                        (c, e, share, share * (2.0 * rounding(tone.coefficients.len() + 5) + moved + rounding(3 * GROUND + 3)))
                    })
                    .collect();
                of.sort_by(|x, y| y.2.total_cmp(&x.2));
                let mut group = 0;
                (0..of.len())
                    .map(|i| {
                        if i > 0 && of[i - 1].2 - of[i].2 > of[i - 1].3 + of[i].3 {
                            group += 1;
                        }
                        Ranked { component: of[i].0, excited: of[i].1, group }
                    })
                    .collect()
            })
            .collect()
    }

    /// The model's beats: the distinct δ_ω − δ_ω' of its frequencies, each with its pairs (ω, ω') -- the first is zero,
    /// every frequency's with itself.
    pub(crate) fn beats_made(&self) -> Vec<(Beat, Vec<(usize, usize)>)> {
        let mut out: Vec<(Beat, Vec<(usize, usize)>)> = vec![];
        for to in 0..self.tones.len() {
            for from in 0..self.tones.len() {
                let beat = self.tones[to].beat.plus(-1, &self.tones[from].beat);
                match out.iter_mut().find(|b| b.0 == beat) {
                    Some(found) => found.1.push((to, from)),
                    None => out.push((beat, vec![(to, from)])),
                }
            }
        }
        out
    }

    /// Two frequencies' constant form (`Cross`), made once: a frequency's own is its static form.
    pub(crate) fn constants(&self, to: usize, from: usize) -> Arc<Cross> {
        let mut made = self.crosses.lock().expect("the constant forms' lock");
        made.entry((to, from))
            .or_insert_with(|| {
                let (t, f) = (&self.tones[to], &self.tones[from]);
                let forms = if to == from { t.forms.clone() } else { super::far::constant(t, f) };
                Arc::new(Cross { pairs: if to == from { t.pairs.clone() } else { super::far::per_pair(t, f, &forms) }, forms })
            })
            .clone()
    }

    /// The bound over a region of the slope of a state's level under the carriers (rad/µs/µm), made once per region and ground
    /// state: a ground state's shifts by the carriers (eliminate.rs `Model::shift_slope`). An explicit excited state's shift by
    /// the folded drives is not in it, as it is not in the couplings' distances (eliminate.rs `Model::near_spread`).
    pub(crate) fn near_slope(&self, s: usize, region: &Region) -> f64 {
        if s < GROUND {
            *region.slopes[s].get_or_init(|| (0..self.carriers.len()).map(|c| self.shift_slope(c, s, region, false)).sum())
        } else {
            0.0
        }
    }

    /// The bounds over a region of the slopes |∇(E_e − E_b)| of the constant forms' levels (eliminate.rs
    /// `Model::energy_slope_over`) from ground state b to each excited state a folded drive reaches from it (per excited state
    /// less GROUND; the others' are not asked), made once per region and ground state: the carriers' light shifts' slopes and
    /// the drives' sweeps read them.
    pub(crate) fn level_slopes<'r>(&self, b: usize, region: &'r Region) -> &'r [f64; EXCITED] {
        region.levels[b].get_or_init(|| {
            let mut slopes = [0.0; EXCITED];
            for &e in &region.forms.reached(self)[b] {
                slopes[e] = self.energy_slope_over(GROUND + e, b, region);
            }
            slopes
        })
    }

    /// The bound over a region of an atom's acceleration (µm/µs²) in the clusters `within` picks: ħ/m times the force's bound
    /// (eliminate.rs `Model::force_over`).
    pub(crate) fn acceleration(&self, region: &Region, within: &dyn Fn(&Cluster) -> bool) -> f64 {
        self.hbar_over_mass * (0..self.clusters.len()).filter(|&c| within(&self.clusters[c])).map(|c| self.force_over(c, region)).fold(0.0, f64::max)
    }

    /// The gate over a region of a fold between any two states' levels -- a photon coherence's --, of size `size`, wave vector
    /// at most `wave` and a distance `distance` that is a bound over all space: the one gate across the distance less the
    /// Doppler shift q v, at the rate v L + q a_max -- L twice the largest slope of a state's level (the constant forms'
    /// energy, a ground state's shifts by the carriers), a_max the largest acceleration in any cluster, both made once per
    /// region.
    pub(crate) fn gate_over(&self, region: &Region, size: f64, wave: f64, distance: f64) -> Gate {
        let (slope, acceleration) = *region.moving.get_or_init(|| {
            let slope = (0..STATES).map(|s| self.energy_slope_over(s, s, region) + self.near_slope(s, region)).fold(0.0, f64::max);
            (2.0 * slope, self.acceleration(region, &|_| true))
        });
        gate(size, distance - wave * region.speed, region.speed * slope + wave * acceleration)
    }

    /// Whether a state is an explicit excited state of this model.
    pub(crate) fn explicit(&self, s: usize) -> bool {
        s >= GROUND && self.clusters.iter().any(|c| c.states.contains(&s))
    }

    /// The units that fold a component's drives with the bare levels (`Unit::Bare`), whatever their level in this model:
    /// kept, each of them is folded with the levels at the atom.
    pub fn bare_units(&self, component: Component) -> Vec<Unit> {
        self.driven.iter().filter(|d| d.0 == component).map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).collect()
    }

    /// The candidate units of this model (model.rs `Unit`), those it folds: each folded drive at its level; the couplings
    /// of two ground states, and of an explicit excited state with another excited one, at each of the model's beats
    /// (module documentation, # A coupling of two states) that no cluster holds as they are; the carriers' dressings and the
    /// photon coherences (eliminate.rs, photon.rs).
    pub fn units(&self) -> Vec<Unit> {
        let mut out = vec![];
        for &(component, excited, ground) in &self.driven {
            match self.level(component, excited, ground) {
                Level::Bare => out.push(Unit::Bare { component, excited, ground }),
                Level::Atom => out.push(Unit::Drive { component, excited, ground }),
                Level::Explicit => {}
            }
        }
        let mut listed: HashSet<Unit> = HashSet::new();
        for (index, (beat, _)) in self.beats.iter().enumerate() {
            for a in 0..STATES {
                for b in 0..STATES {
                    if !(if a < GROUND { b < GROUND } else { b >= GROUND && (self.explicit(a) || self.explicit(b)) }) {
                        continue;
                    }
                    let unit = term_unit(a, b, beat);
                    if !listed.contains(&unit) && !self.forms.term(self, a, b, index).is_empty() && !self.in_generator(a, b, beat) {
                        listed.insert(unit.clone());
                        out.push(unit);
                    }
                }
            }
        }
        // A carrier's dressing: where its T̄ averages an interference (several polarization units) and an elimination takes
        // it -- some carrier has a drive folded at the atom, whose denominators hold its static form and its light shifts
        // (eliminate.rs, # The response).
        let eliminated = self.carriers.iter().any(|cc| cc.folded.iter().any(|x| x.data.iter().any(|z| *z != C64::ZERO)));
        for cc in self.carriers.iter().filter(|_| eliminated) {
            let mut units = cc.units.clone();
            units.sort();
            units.dedup();
            if units.len() > 1 {
                out.push(Unit::Dressing { carrier: Component { frequency: cc.frequency, counter: false } });
            }
        }
        out.extend(self.coherence_units());
        out.retain(|u| !self.kept.contains(u));
        out
    }

    /// Whether the coupling a ← b at a beat is in this model's generators as it is (SPEC §1.4), nothing of it folded: the
    /// two states in one cluster, at a residual beat of exactly zero or as a kept unit (eliminate.rs `Model::term_included`).
    fn in_generator(&self, a: usize, b: usize, beat: &Beat) -> bool {
        self.clusters.iter().enumerate().any(|(k, c)| {
            let place = |s: usize| c.states.iter().position(|&x| x == s);
            matches!((place(a), place(b)), (Some(x), Some(y)) if self.term_included(k, x, y, beat))
        })
    }

    /// The kept units' edges (cluster.rs): a kept drive at the atom joins its two states at θ_e − θ_g = δ_ω, its strongest
    /// beam's wave vector; a kept coupling a ← b at the beat β joins them at θ_a − θ_b = −β, its strongest beam pair's
    /// k_absorbed − k_emitted, its strength the peak of its entry with every drive bare, Σ_pairs |C_ji[a, b]| (`bare`: the
    /// frequencies with every drive folded with the bare levels -- one measure for every coupling, whatever the drives'
    /// levels turn out; the tree is a choice of cost, cluster.rs). A unit of nothing in this request makes none.
    pub(crate) fn edges(&self, bare: &[super::far::Tone]) -> Vec<Edge> {
        let mut out = vec![];
        let mut crosses: HashMap<(usize, usize), Vec<Mat>> = HashMap::new();
        for unit in &self.kept {
            match unit {
                Unit::Drive { component, excited, ground } if !component.counter && self.driven.contains(&(*component, *excited, *ground)) => {
                    let (f, tone) = (&self.light.frequencies[component.frequency], &self.tones[component.frequency]);
                    let (best, strength) = f.beams.iter().zip(&tone.couplings[0]).fold((f.beams[0], 0.0_f64), |(best, s), (&l, x)| {
                        let v = x[(*excited, *ground)].abs();
                        (if v > s { l } else { best }, s.max(v))
                    });
                    out.push(Edge { a: *excited, b: *ground, beat: tone.beat.clone(), wave: self.beam_wave(best), strength });
                }
                Unit::Term { a, b, beat } => {
                    let (mut best, mut strength) = ((0.0_f64, self.no_wave()), 0.0);
                    for &(to, from) in self.beats.iter().find(|x| x.0 == *beat).map_or(&[][..], |x| &x.1) {
                        let pairs = crosses.entry((to, from)).or_insert_with(|| {
                            if to == from { bare[to].pairs.clone() } else { super::far::per_pair(&bare[to], &bare[from], &super::far::constant(&bare[to], &bare[from])) }
                        });
                        let (bt, bf) = (&self.light.frequencies[to].beams, &self.light.frequencies[from].beams);
                        for (x, m) in pairs.iter().enumerate() {
                            let size = m[(*a, *b)].abs();
                            strength += size;
                            if size > best.0 {
                                best = (size, self.pair_wave(bt[x / bf.len()], bf[x % bf.len()]));
                            }
                        }
                    }
                    if strength > 0.0 {
                        out.push(Edge { a: *a, b: *b, beat: beat.negated(), wave: best.1, strength });
                    }
                }
                _ => {}
            }
        }
        out
    }

    /// A ground state's drives' gates over a region (module documentation, # Levels, # The tail rule), per place of the tail
    /// rule's order: the folded shares from there down and the place's largest sweep, of its drives at the atom; and the
    /// same of its bare drives, each one's own share and sweep across its own distance (the largest of them). A folded
    /// drive's size is v = |Σ_l a_l X_l[e, g]| (reach.rs `Region::coherent`); over the region its detuning lies within
    /// `moved` of its centre value Δ_c = (E_e − E_g)(c) − ν -- the two energies' slack, the other components' carriers'
    /// dressing of g, the Doppler shift k·v of the region's speeds, the excited state's Gershgorin radius and the
    /// self-energy's bound (eliminate.rs) --, so its distance there is at least |max(|Δ_c| − moved, 0) − iΓ_e/2|; a bare
    /// drive's own gate is of v + |Δ_c − Δ₀| + moved across its bare |Δ₀ − iΓ_e/2|; its rate v L + k a_max, L the slope of
    /// E_e − E_g and of the other carriers' dressings of g. Of the order's first `places` places; the weaker ones' shares enter by `rest`, a
    /// bound of their sum.
    fn folds_made(&self, g: usize, region: &Region, places: usize, rest: f64) -> Vec<[f64; 4]> {
        let drives = &region.forms.drives(self)[g];
        let cluster = self.membership[g];
        let (energies, couplings, dressings) = (self.energies(region), self.couplings(region), self.dressings(region));
        let folded = region.forms.folded(self);
        let dress: [f64; GROUND] = std::array::from_fn(|a| dressings.iter().map(|d| d[a]).sum());
        let acceleration = self.hbar_over_mass * self.force_over(cluster, region);
        let slopes = self.level_slopes(g, region);
        // Per component, what its drives of g share: the dressing of g by the other components' carriers and its slope, the
        // largest wave number of its beams, and the self-energy's bound per excited state -- its Gershgorin row,
        // Σ_{d, a∉C} |u_a[e]| ‖u_a‖/|z − Ẽ_a − δ_d| over the carriers' channels folded at the atom to the ground states of the
        // other clusters (only the states a channel reaches move).
        let mut shared: Vec<Option<(f64, f64, f64, [f64; EXCITED])>> = vec![None; 2 * self.tones.len()];
        let mut out = vec![[0.0; 4]; drives.last().map_or(0, |d| d.group + 1).min(places)];
        for d in drives.iter().filter(|d| d.level != Level::Explicit && d.group < places) {
            let (frequency, side) = (d.component / 2, d.component % 2);
            let nu = self.tones[frequency].nu[side];
            let (others, others_slope, k, sigma) = shared[d.component].get_or_insert_with(|| {
                let other = |c: &super::eliminate::Carrier| side == 1 || c.frequency != frequency;
                let others: f64 = self.carriers.iter().zip(dressings).filter(|of| other(of.0)).map(|of| of.1[g]).sum();
                let others_slope: f64 = (0..self.carriers.len()).filter(|&c| other(&self.carriers[c])).map(|c| self.shift_slope(c, g, region, true)).sum();
                let (z_lo, z_hi) = (energies.centre[g] - energies.slack[g] + nu - others, energies.centre[g] + energies.slack[g] + nu + others);
                let mut sigma = [0.0; EXCITED];
                for ((c, cd), of) in self.carriers.iter().zip(couplings).zip(folded) {
                    for o in (0..GROUND).filter(|&o| self.membership[o] != cluster) {
                        let (lo, hi) = (z_lo - energies.centre[o] - energies.slack[o] - dress[o] - c.detuning, z_hi - energies.centre[o] + energies.slack[o] + dress[o] - c.detuning);
                        let (den, size) = (distance(lo, hi), cd.column[o]);
                        if size == 0.0 {
                            continue;
                        }
                        for &x in &of.reached[o] {
                            let reach = cd.entry[o][x];
                            if reach > 0.0 {
                                sigma[x] += reach * size / den;
                            }
                        }
                    }
                }
                let k = self.light.frequencies[frequency].beams.iter().map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max);
                (others, others_slope, k, sigma)
            });
            let (x, e) = (d.excited, GROUND + d.excited);
            let centre = energies.centre[e] - energies.centre[g] - nu;
            let moved = energies.slack[e] + energies.slack[g] + *others + *k * region.speed + energies.radius[e] + sigma[x];
            let (size, rate) = (region.coherent(&d.beams), region.speed * (slopes[x] + *others_slope) + *k * acceleration);
            let actual = gate(size, C64::new((centre.abs() - moved).max(0.0), 0.5 * d.gamma).abs(), rate);
            let place = &mut out[d.group];
            place[0] += actual.saturation;
            if d.level == Level::Bare {
                let own = gate(size + (centre - d.detuning).abs() + moved, d.bare, rate);
                place[2] = place[2].max(own.saturation - actual.saturation);
                place[3] = place[3].max(own.sweep);
            } else {
                place[1] = place[1].max(actual.sweep);
            }
        }
        // The tails: each place's shares with every weaker one's; a bare drive's own share in its own's stead.
        for place in (0..out.len()).rev() {
            let tail = out[place][0] + out.get(place + 1).map_or(rest, |weaker| weaker[0]);
            out[place][0] = tail;
            out[place][2] += tail;
        }
        out
    }

    /// A ground state's drives' gate at a place of the tail rule's order over a region (`folds_made`): [saturation, sweep]
    /// of its drives at the atom, then of its bare ones. The region makes the order's first `places` places once -- those a
    /// proof over many cells still asks (`Model::gates`) --, the weaker ones in them by `rest`, a bound of their shares' sum
    /// over any region; a place beyond them is made with every place.
    pub(crate) fn folds(&self, g: usize, place: usize, region: &Region, places: usize, rest: f64) -> [f64; 4] {
        let kept = region.drives[g].get_or_init(|| self.folds_made(g, region, places, rest));
        kept.get(place).copied().unwrap_or_else(|| self.folds_made(g, region, usize::MAX, 0.0)[place])
    }

    /// The bound over a region of the denominator of a drive e ← s folded at the atom at the detuning δ: its two levels'
    /// distance there, |E_e − E_s − δ| at the centre less their slack and the excited state's Gershgorin radius
    /// (eliminate.rs `CellEnergies`), across Γ_e/2.
    pub(crate) fn denominator_over(&self, region: &Region, e: usize, s: usize, detuning: f64) -> f64 {
        let energies = self.energies(region);
        let apart = (energies.centre[e] - energies.centre[s] - detuning).abs() - energies.slack[e] - energies.slack[s] - energies.radius[e];
        C64::new(apart.max(0.0), 0.5 * self.decay[(e - GROUND, e - GROUND)].re).abs()
    }

    /// A coupling's (`Unit::Term`: a ← b at the model's beat `beat`) saturation and sweep over a region (module documentation,
    /// # A coupling of two states), made once per region. Its size: the constant part's value |H_ab + Σ_ji c_ji C_ji[a, b]| at
    /// the centre (H_ab the two states' own coupling with every light off, a static one's) plus Σ |C_ji[a, b]| times each
    /// pair's radius (reach.rs `PairSpan`), at most every beam at its largest amplitude and in phase; and the part at the atom, per excited state ½ |u'_a[e]| |u_b[e]| (1/σ_eb + 1/σ'_ea), u the two frequencies'
    /// folded couplings' coherent bounds and σ the denominator's bound of each drive folded at the atom (the bare drives'
    /// halves are the constants'). Its distance: |β + E_a − E_b| over the region -- the constant forms' difference, the two
    /// states' own --, less the Doppler shift of its beam pairs' largest wave vector q at the region's speeds and the
    /// carriers' differential shift of the two states; its rate v |∇(E_a − E_b)| + q a_max, the two states' constant energies
    /// and shifts by the carriers (one state's beat moves with neither), the acceleration in their clusters. On a point, its
    /// measured values there.
    pub(crate) fn term_bound(&self, form: &TermForm, a: usize, b: usize, beat: usize, region: &Region) -> (f64, f64) {
        region.term((a, b, beat), || {
            let (mut centre, mut moved, mut blind) = (form.base, 0.0, form.base.abs());
            for &(lj, li, v, size) in &form.pairs {
                let pair = pair_span(&region.spans, lj, li);
                centre += pair.centre * v;
                moved += size * pair.radius;
                blind += size * region.spans[lj].most * region.spans[li].most;
            }
            let mut size = (centre.abs() + moved).min(blind);
            for leg in &form.legs {
                let e = GROUND + leg.excited;
                let halves = leg.from_atom.map_or(0.0, |detuning| 1.0 / self.denominator_over(region, e, b, detuning)) + leg.to_atom.map_or(0.0, |detuning| 1.0 / self.denominator_over(region, e, a, detuning));
                size += 0.5 * region.coherent(&leg.to) * region.coherent(&leg.from) * halves;
            }
            if size == 0.0 {
                return (0.0, 0.0);
            }
            let (mut at, mut slack) = (self.omega(&self.beats[beat].0), 0.0);
            if a != b {
                at += self.base[(a, a)].re - self.base[(b, b)].re;
                for &(frequency, pair, d, modulus) in region.forms.difference(self, a, b) {
                    let pair = &region.pairs[frequency][pair];
                    at += (pair.centre * d).re;
                    slack += modulus * pair.radius;
                }
            }
            let (spread, slope) = if a == b { (0.0, 0.0) } else { (self.near_spread(a, b, region), self.energy_slope_over(a, b, region) + self.near_slope(a, region) + self.near_slope(b, region)) };
            let acceleration = self.acceleration(region, &|c| c.states.contains(&a) || c.states.contains(&b));
            let made = gate(size, distance(at - slack, at + slack) - form.wave * region.speed - spread, region.speed * slope + form.wave * acceleration);
            (made.saturation, made.sweep)
        })
    }

    /// The model's beat a unit's coupling names, its place among `Model::beats`.
    fn beat_index(&self, beat: &Beat) -> Option<usize> {
        self.beats.iter().position(|b| b.0 == *beat)
    }

    /// Each unit's existence gate over the reach (SPEC §1.5; module documentation), provable bounds at this model's clusters
    /// and levels: every kind through the one gate (`gate`) -- a drive's by its ground state's tail (`folds`), a coupling's
    /// (`term_bound`), a photon coherence's (`gate_over`, its size its two classes' jumps' bound, photon.rs). Each is bounded on the
    /// whole space first, every beam at its peak: a bound over any reach, and where both of a unit's are below ½ there its
    /// gate. The others are proven together over the reach's cells, on all the cores (reach.rs
    /// `Model::bounds_over_in_parallel`) -- a ground state's drives' with the shares of those the whole space has settled by
    /// their bound there. A dressing folds nothing and has no gate; a unit of nothing in this model has nothing to fold.
    pub fn gates(&self, units: &[Unit], reach: &Reach) -> Vec<Gate> {
        // What a unit's gate reads: a drive's its ground state, its place in the tail rule's order and whether it is bare; a
        // coupling's its form; a coherence's its unit and its distance over all space.
        enum Read<'u> {
            Fold { ground: usize, place: usize, bare: bool },
            Term { form: Arc<TermForm>, a: usize, b: usize, beat: usize },
            Coherence { unit: &'u Unit, apart: f64 },
            Nothing,
        }
        let reads: Vec<Read> = units
            .iter()
            .map(|u| match *u {
                Unit::Bare { component, excited, ground } | Unit::Drive { component, excited, ground } => {
                    let level = self.ranked[ground].iter().find(|d| d.component == component && d.excited == excited).map(|d| (d.group, self.level(component, excited, ground)));
                    level.filter(|of| of.1 != Level::Explicit).map_or(Read::Nothing, |(place, level)| Read::Fold { ground, place, bare: level == Level::Bare })
                }
                Unit::Term { a, b, ref beat } => self.beat_index(beat).map_or(Read::Nothing, |index| Read::Term { form: self.forms.term(self, a, b, index), a, b, beat: index }),
                Unit::Dressing { .. } => Read::Nothing,
                Unit::Coherence { ref first, ref second, .. } => Read::Coherence { unit: u, apart: self.omega(&first.plus(-1, second)).abs() - self.spread() },
            })
            .collect();
        // The largest wave number of the light's beams: a coherence's two photons'.
        let wave = 2.0 * self.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max);
        let read = |of: &Read, region: &Region, places: usize, rest: f64| -> [f64; 2] {
            match of {
                Read::Fold { ground, place, bare } => {
                    let made = self.folds(*ground, *place, region, places, rest);
                    if *bare { [made[2], made[3]] } else { [made[0], made[1]] }
                }
                Read::Term { form, a, b, beat } => {
                    let made = self.term_bound(form, *a, *b, *beat, region);
                    [made.0, made.1]
                }
                Read::Coherence { unit, apart } => {
                    let made = self.gate_over(region, self.coherence_bound(unit, region), wave, *apart);
                    [made.saturation, made.sweep]
                }
                Read::Nothing => [0.0; 2],
            }
        };
        let whole = self.region(self.spans(&Cell { centre: [0.0; 3], half: [f64::INFINITY; 3] }), reach.speed);
        let everywhere: Vec<[f64; 2]> = reads.iter().map(|of| read(of, &whole, usize::MAX, 0.0)).collect();
        let open: Vec<usize> = (0..units.len()).filter(|&u| !(everywhere[u][0] < EXISTS && everywhere[u][1] < EXISTS)).collect();
        // Per ground state the places its open drives' gates ask of a cell's table, and the weaker places' shares over the
        // whole space.
        let mut places = [0; GROUND];
        for &u in &open {
            if let Read::Fold { ground, place, .. } = reads[u] {
                places[ground] = places[ground].max(place + 1);
            }
        }
        let rest: [f64; GROUND] = std::array::from_fn(|g| if places[g] > 0 && self.ranked[g].last().is_some_and(|d| d.group >= places[g]) { self.folds(g, places[g], &whole, usize::MAX, 0.0)[0] } else { 0.0 });
        let mut quantities: Vec<Box<dyn Fn(&Region) -> f64 + Sync + '_>> = vec![];
        let mut shared: HashMap<(usize, usize, bool), usize> = HashMap::new();
        // Per open unit where its two bounds lie among the quantities: the saturation's, then the sweep's.
        let asked: Vec<usize> = open
            .iter()
            .map(|&u| {
                let of = &reads[u];
                let (key, asks, weaker) = if let Read::Fold { ground, place, bare } = *of { (Some((ground, place, bare)), places[ground], rest[ground]) } else { (None, 0, 0.0) };
                if let Some(found) = key.and_then(|k| shared.get(&k)) {
                    return *found;
                }
                let first = quantities.len();
                quantities.push(Box::new(move |r: &Region| read(of, r, asks, weaker)[0]));
                quantities.push(Box::new(move |r: &Region| read(of, r, asks, weaker)[1]));
                if let Some(k) = key {
                    shared.insert(k, first);
                }
                first
            })
            .collect();
        let refs: Vec<&(dyn Fn(&Region) -> f64 + Sync)> = quantities.iter().map(|q| q.as_ref()).collect();
        let proven = self.bounds_over_in_parallel(reach, EXISTS, &refs);
        let mut out: Vec<Gate> = everywhere.iter().map(|of| Gate { saturation: of[0], sweep: of[1] }).collect();
        for (&u, &at) in open.iter().zip(&asked) {
            out[u] = Gate { saturation: proven[at], sweep: proven[at + 1] };
        }
        out
    }

    /// Each candidate unit's errors at each sample (`Model::units`), [unit][sample]: its small parameter at the sample's point
    /// mapped to the outputs (module documentation, # The errors), the forms at the atom from the exact elimination at the
    /// point.
    pub fn errors(&self, units: &[Unit], samples: &[Sample]) -> Vec<Vec<UnitError>> {
        self.errors_with(units, samples, None)
    }

    /// The units' errors at the samples (`errors`), the forms at the atom from their tables where they serve (tabulate.rs): a
    /// state then costs an interpolation and the units' arithmetic, and one elimination more per dressing unit.
    pub fn errors_with(&self, units: &[Unit], samples: &[Sample], tables: Option<&Tables>) -> Vec<Vec<UnitError>> {
        let mut out: Vec<Vec<UnitError>> = units.iter().map(|_| Vec::with_capacity(samples.len())).collect();
        for sample in samples {
            let at = self.measured(sample, tables);
            for (unit, errors) in units.iter().zip(out.iter_mut()) {
                errors.push(self.unit_error(unit, &at));
            }
        }
        out
    }

    /// What the units' errors at a sample share (`Measured`).
    pub(crate) fn measured<'a>(&self, sample: &'a Sample, tables: Option<&Tables>) -> Measured<'a> {
        let lights = self.running();
        let amplitudes = self.light.amplitudes(sample.at.r);
        let plan = self.plan(sample.cluster, &lights);
        let sections = [(plan.near.hermitian(), 1), (plan.near.jumping(), 0), (plan.near.responding(), 0)];
        let near = self.near_values(&plan, &amplitudes, &sections, tables.and_then(|t| t.of(sample.cluster, &lights)));
        let responses = self.carrier_responses(&plan, &near, &amplitudes, sample.at.kappa);
        // The force on the state at the point: −tr(G ρ) of its density (step.rs `Middle::force`, at rest on the middle).
        let middle = self.middle_of(&plan, sample.at.t, plan.amplitudes(&amplitudes), Some(&near));
        let (n, norm) = (sample.psi.len(), crate::linalg::norm_sqr(&sample.psi));
        let rho: Vec<C64> = (0..n * n).map(|i| sample.psi[i / n] * sample.psi[i % n].conj() / norm).collect();
        let force = crate::step::Middle::force(&middle, &crate::step::Middle::path(&middle, sample.at.kappa, [0.0; 3]), 0.0, &rho).map(|f| -f);
        // The state's folded excitation drive by drive: a bare one's coupling is its response times its bare denominator, one
        // at the atom's is its beams' at the point.
        let bare = self.far_channels(sample.cluster, &amplitudes, &lights);
        let populations = self.state_populations(sample);
        let c = &self.clusters[sample.cluster];
        let (mut rates, mut shifts) = (0.0, 0.0);
        for (y, &g) in c.states[..c.ground].iter().enumerate() {
            for x in 0..EXCITED {
                for channel in &bare {
                    let size = channel.response[(x, y)].norm_sqr();
                    if size > 0.0 {
                        rates += size * populations[g];
                        shifts += size * self.bare_distance(channel.component, GROUND + x, g) * populations[g];
                    }
                }
                for (cc, response) in self.carriers.iter().zip(&responses) {
                    let size = response.as_ref().map_or(0.0, |r| r[(x, y)].abs());
                    if size > 0.0 {
                        let coupling: C64 = cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + x, g)]).sum();
                        rates += size * size * populations[g];
                        shifts += size * coupling.abs() * populations[g];
                    }
                }
            }
        }
        // The kept drives' part of both: an explicit state's population is its excitation as it is, and its coupling's
        // energy |V[e, g]| |ψ_e| |ψ_g| what a fold's |V R| p_g is of ψ_e = −R ψ_g.
        for &e in &c.states[c.ground..] {
            rates += populations[e];
            for &g in &c.states[..c.ground] {
                let coupling: C64 = self
                    .carriers
                    .iter()
                    .filter(|cc| self.level(Component { frequency: cc.frequency, counter: false }, e, g) == Level::Explicit)
                    .flat_map(|cc| cc.beams.iter().zip(&cc.couplings).map(|(&l, m)| amplitudes[l].value * m[(e, g)]))
                    .sum();
                shifts += coupling.abs() * (populations[e] * populations[g]).sqrt();
            }
        }
        Measured {
            sample,
            region: self.point_region(sample.at.r, self.hbar_over_mass * norm3(sample.at.kappa)),
            near,
            responses,
            bare,
            rates,
            shifts,
            energies: self.far_energies(&amplitudes, &lights, 1, &self.coherent_dressing()).0,
            force,
            populations,
            classes: OnceCell::new(),
            doppler: self.carriers.iter().map(|_| OnceCell::new()).collect(),
            amplitudes,
            plan,
        }
    }

    /// A unit's error at a sample from what the sample's units share.
    fn unit_error(&self, unit: &Unit, at: &Measured) -> UnitError {
        match unit {
            Unit::Bare { .. } => self.bare_error(unit, at),
            Unit::Drive { .. } => self.drive_error(unit, at),
            Unit::Term { a, b, beat } => {
                let m2 = self.beat_index(beat).map_or(0.0, |index| 0.5 * self.term_bound(&self.forms.term(self, *a, *b, index), *a, *b, index, &at.region).0);
                let p = &at.populations;
                UnitError { parameter: m2, population: m2 * (p[*a] + p[*b]), temperature: m2 * (p[*a] + p[*b]) }
            }
            Unit::Dressing { carrier } => self.dressing_error(*carrier, at),
            Unit::Coherence { .. } => self.coherence_error(unit, at),
        }
    }

    /// The error at a sample of a drive folded with the bare levels (module documentation, # Levels, # The errors): its own
    /// share s = 2|R⁰[e, g]|² of its ground state, as every fold's -- s p_g on the populations, (√2 s + a) p_g on T, a its
    /// sweep across its denominator at the atom --; and the first order of its denominator, ε = |δΔ|/|Δ₀ − iΓ_e/2|
    /// (eliminate.rs `Model::drive_motion`), the relative error of its response: of the state's rates, quadratic in the
    /// responses, 2ε times its share |R⁰[e, g]|² p_g of them, and of its light shifts, linear, ε times its share
    /// |V R⁰| p_g of those (`Measured::rates`, `Measured::shifts`: shares that add up to one over the state's drives,
    /// folded and kept, so that the units' errors add up to the state's) -- the rates' on the populations, both on T, booked on
    /// the state's ground population as a dressing's relative rate error is. Zero where the sample's cluster does not
    /// hold its ground state.
    fn bare_error(&self, unit: &Unit, at: &Measured) -> UnitError {
        let nothing = UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
        let Unit::Bare { component, excited, ground } = *unit else { return nothing };
        let c = &self.clusters[at.sample.cluster];
        let (Some(y), Some(channel)) = (c.states[..c.ground].iter().position(|&s| s == ground), at.bare.iter().find(|channel| channel.component == component)) else { return nothing };
        let response = channel.response[(excited - GROUND, y)];
        let s = 2.0 * response.norm_sqr();
        let (moved, rate) = self.drive_motion(component, excited, ground, at);
        let own = self.bare_distance(component, excited, ground);
        let gamma = self.decay[(excited - GROUND, excited - GROUND)].re;
        let nu = self.tones[component.frequency].nu[component.counter as usize];
        let sweep = gate(0.0, C64::new(self.energies[excited] - self.energies[ground] - nu + moved, 0.5 * gamma).abs(), rate).sweep;
        let (epsilon, p, held) = (moved.abs() / own, at.populations[ground], at.populations[..GROUND].iter().sum::<f64>());
        let share = |of: f64, all: f64| if all > 0.0 { of * p / all } else { 0.0 };
        let (rates, shifts) = (2.0 * epsilon * share(response.norm_sqr(), at.rates) * held, epsilon * share(response.norm_sqr() * own, at.shifts) * held);
        UnitError { parameter: epsilon, population: s * p + rates, temperature: (std::f64::consts::SQRT_2 * s + sweep) * p + rates + shifts }
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

    /// The exact wave vector of a beam pair's term, absorbing `absorbed` and emitting `emitted`: k_absorbed − k_emitted.
    pub(crate) fn pair_wave(&self, emitted: usize, absorbed: usize) -> Wave {
        self.beam_wave(absorbed).plus(-1, &self.beam_wave(emitted))
    }
}
