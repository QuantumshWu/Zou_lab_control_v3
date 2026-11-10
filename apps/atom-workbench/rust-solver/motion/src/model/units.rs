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
//! their largest sweep. Over a cell, and on a point, every share of the sum is that cell's or that point's own: a
//! point's gate is the value there. (A weaker place's bound from elsewhere in its stead is no value of any point: with
//! the whole space's 0.49 for one of two equal drives, the other's gate read 0.52 "attained" at the trap's centre where
//! the two shares are 0.03 each, and a light 20 linewidths off its line was made explicit.)
//!
//! # What a folded drive does to its level
//! A drive of coupling v folded across the distance d moves its ground level by at most v²/d = m v, and by less than
//! v/2 wherever its fold exists: the shift is the fold's own second order, across the fold's own distance. So what the
//! drives folded at the atom do to a ground level over a region is bounded drive by drive, each across the distance its
//! own gate reads there (`Model::across`, `Model::near_shift`), and the slope of that shift with the response's norm
//! 1/d of the nearest of them (eliminate.rs `Model::shift_slope`). Never across Γ/2, the distance of a resonant drive,
//! whatever the drive: that bound is 2|Δ|/Γ times the shift of a light Δ off its line, grows with the intensity as the
//! shift does, and at 80 mW/cm² of light 20 linewidths off it exceeded the 137 MHz between a cooling and a repump
//! photon -- a coupling's distance read nothing, its gate ∞, and F=1 and F=2 were made one cluster.
//!
//! # The fold's order
//! The second-order fold leaves out one term of the next order, the fourth-order light shift: a state's potentials are
//! too deep by s/2 and the force on it too large by s, s = 2 × its excited share; its photons come too fast by s
//! (exactly, on one line: the force (1 − s + 3/2 s²) and the rate 1/(1 + s) of the second order's). T follows the
//! potentials' modulation, 0.8 to 0.9 s on a Sisyphus pattern -- the diffusion +0.6 s, the friction −0.2 s, both of the
//! one term, not two errors to add --: the force's s bounds it, and that is booked, s p_g on T and on the populations.
//! Where the fold has a fourth order (assemble.rs, # The fourth order; model.rs `Model::fold_frequency`) that part is
//! its own unit, the cluster's (`Unit::Second`): kept, the fold is the fourth order's, the force's error 3/2 s² p_g and
//! the levels' spread over the drive's distance times s (W3's one denominator a column), and the rates' s/2 -- the
//! prompt photon after a jump, not drawn. A drive's own unit then carries what only keeping it explicit removes: that
//! residue, and at either order its motion's part, the sweep a and the Doppler term's next order. Elsewhere a drive's
//! unit carries all of it.
//!
//! # A coupling of two states (`Unit::Term`)
//! Everything the folds leave between two states a ← b at one exact beat β (∝ e^{iβt}) is one unit: the constant forms'
//! entry (far.rs `constant`: one frequency's static form at β = 0, with the two states' own coupling when every light is
//! off; two frequencies' cross form at their beat; on the ground states and on the excited ones) and the part of the
//! drives folded at the atom (eliminate.rs). Its size is the
//! bound of that entry, its distance |β + E_a − E_b| over the region less the Doppler shift of its beam pairs' wave
//! vector and the shifts of the two states by their drives folded at the atom (# What a folded drive does to its
//! level), its rate v|∇(E_a − E_b)| + q a_max. Kept, the entry is
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
//! share s p_g on the populations and (s + a) p_g on T (# The fold's order); a bare one also 2ε times its share of the state's rates, on
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

/// The fourth order's leading errors in a drive's share s (module documentation, # The fold's order; derivation:
/// fourth.status.md): on the force 3/2 s², the next term of the exact (1 − s + 3/2 s²); on the rates s/2, the prompt
/// photon a jump's landing in the bare state leaves, which the step does not draw.
pub(crate) const ORDER_FORCE: f64 = 1.5;
pub(crate) const ORDER_RATES: f64 = 0.5;

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

/// One drive of a ground state over a region (`Model::across`): its coupling's size v, its detuning at the region's
/// centre Δ_c, and how far from it its detuning lies at most over the region.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Across {
    pub(crate) size: f64,
    pub(crate) centre: f64,
    pub(crate) moved: f64,
}

impl Across {
    /// Its distance over the region, at least: |max(|Δ_c| − moved, 0) − iΓ_e/2|.
    pub(crate) fn distance(&self, gamma: f64) -> f64 {
        C64::new((self.centre.abs() - self.moved).max(0.0), 0.5 * gamma).abs()
    }
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
    /// the folded drives is not in it, as it is not in the couplings' distances (`Model::near_shift`).
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

    /// The candidate units of this model (model.rs `Unit`), those it folds: each folded drive at its level; the second
    /// order of each cluster's fold at the atom that has a fourth (model.rs `Model::fold_frequency`); the couplings
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
        // The second order of a cluster's fold at the atom, where the fold has a next order and is not taken to it.
        for (c, cluster) in self.clusters.iter().enumerate() {
            if self.fourth[c].is_none() && self.fold_frequency(c).is_some() {
                out.push(Unit::Second { ground: cluster.states[0] });
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

    /// A ground state's drives over a region as their gates read them (`Across`; module documentation, # Levels), in the
    /// tail rule's order (an explicit one's unread); made once per region and ground state. A folded drive's size is
    /// v = |Σ_l a_l X_l[e, g]| (reach.rs `Region::coherent`); over the region its detuning lies within `moved` of its
    /// centre value Δ_c = (E_e − E_g)(c) − ν -- the two energies' slack and what the eliminations' levels average of them
    /// (eliminate.rs `CellEnergies`), the other components' carriers' dressing of g, the Doppler shift k·v of the region's
    /// speeds, the excited state's Gershgorin radius and the self-energy's bound (eliminate.rs) --, so its distance there
    /// is at least |max(|Δ_c| − moved, 0) − iΓ_e/2|.
    pub(crate) fn across<'r>(&self, g: usize, region: &'r Region) -> &'r [Across] {
        region.across[g].get_or_init(|| {
            let cluster = self.membership[g];
            let (energies, couplings, dressings) = (self.energies(region), self.couplings(region), self.dressings(region));
            let folded = region.forms.folded(self);
            let dress: [f64; GROUND] = std::array::from_fn(|a| dressings.iter().map(|d| d[a]).sum());
            // Per component, what its drives of g share: the dressing of g by the other components' carriers, the largest
            // wave number of its beams, and the self-energy's bound per excited state -- its Gershgorin row,
            // Σ_{d, a∉C} |u_a[e]| ‖u_a‖/|z − Ẽ_a − δ_d| over the carriers' channels folded at the atom to the ground states of
            // the other clusters (only the states a channel reaches move).
            let mut shared: Vec<Option<(f64, f64, [f64; EXCITED])>> = vec![None; 2 * self.tones.len()];
            region.forms.drives(self)[g]
                .iter()
                .map(|d| {
                    if d.level == Level::Explicit {
                        return Across { size: 0.0, centre: 0.0, moved: 0.0 };
                    }
                    let (frequency, side) = (d.component / 2, d.component % 2);
                    let nu = self.tones[frequency].nu[side];
                    let (others, wave, sigma) = shared[d.component].get_or_insert_with(|| {
                        let others: f64 = self.carriers.iter().zip(dressings).filter(|of| self.other(of.0, d.component)).map(|of| of.1[g]).sum();
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
                        (others, self.wave_of(d.component), sigma)
                    });
                    let (x, e) = (d.excited, GROUND + d.excited);
                    let moved = energies.slack[e] + energies.slack[g] + energies.averaged[e] + energies.averaged[g] + *others + *wave * region.speed + energies.radius[e] + sigma[x];
                    Across { size: region.coherent(&d.beams), centre: energies.centre[e] - energies.centre[g] - nu, moved }
                })
                .collect()
        })
    }

    /// Whether a carrier is another component's than `component` (2 frequency + counter): its dressing of a ground state
    /// moves that component's drives (a carrier's own is its fold's).
    fn other(&self, carrier: &super::eliminate::Carrier, component: usize) -> bool {
        component % 2 == 1 || carrier.frequency != component / 2
    }

    /// The largest wave number of a component's beams (1/µm).
    fn wave_of(&self, component: usize) -> f64 {
        self.light.frequencies[component / 2].beams.iter().map(|&l| self.light.beams[l].wave_number).fold(0.0, f64::max)
    }

    /// The bound over a region of the shift of a ground state's level by its drives folded at the atom (rad/µs; module
    /// documentation, # What a folded drive does to its level): Σ v²/distance, each drive's coupling across its own
    /// distance there (`across`); made once per region and state. Nothing for an excited state: an explicit one's shift by
    /// the folded drives is in no distance.
    pub(crate) fn near_shift(&self, s: usize, region: &Region) -> f64 {
        if s >= GROUND {
            return 0.0;
        }
        *region.shifts[s].get_or_init(|| {
            let drives = &region.forms.drives(self)[s];
            drives.iter().zip(self.across(s, region)).filter(|of| of.0.level == Level::Atom).map(|(d, of)| of.size * of.size / of.distance(d.gamma)).sum()
        })
    }

    /// The least distance over a region of a carrier's drives of a ground state folded at the atom (`across`): what the
    /// carrier's response from that state is taken across, at least. None where it has no such drive.
    pub(crate) fn least_across(&self, carrier: usize, b: usize, region: &Region) -> Option<f64> {
        let component = 2 * self.carriers[carrier].frequency;
        let drives = &region.forms.drives(self)[b];
        let folded = drives.iter().zip(self.across(b, region)).filter(|of| of.0.level == Level::Atom && of.0.component == component);
        folded.map(|(d, of)| of.distance(d.gamma)).min_by(f64::total_cmp)
    }

    /// A ground state's drives' gates over a region (module documentation, # Levels, # The tail rule), per place of the tail
    /// rule's order: the folded shares from there down and the place's largest sweep, of its drives at the atom; and the
    /// same of its bare drives, each one's own share and sweep across its own distance (the largest of them). Every share
    /// is the region's own, the weaker places' as the place's: a point's are the values there. A folded drive's gate is
    /// of its size across its distance over the region (`across`); a bare drive's own gate of v + |Δ_c − Δ₀| + moved
    /// across its bare |Δ₀ − iΓ_e/2|; its rate v L + k a_max, L the slope of E_e − E_g and of the other components'
    /// carriers' dressings of g.
    fn folds_made(&self, g: usize, region: &Region) -> Vec<[f64; 4]> {
        let drives = &region.forms.drives(self)[g];
        let acceleration = self.hbar_over_mass * self.force_over(self.membership[g], region);
        let slopes = self.level_slopes(g, region);
        // Per component the slope of the other components' carriers' dressings of g and its beams' largest wave number.
        let mut shared: Vec<Option<(f64, f64)>> = vec![None; 2 * self.tones.len()];
        let mut out = vec![[0.0; 4]; drives.last().map_or(0, |d| d.group + 1)];
        for (d, of) in drives.iter().zip(self.across(g, region)).filter(|of| of.0.level != Level::Explicit) {
            let (others, wave) = *shared[d.component].get_or_insert_with(|| {
                let others: f64 = (0..self.carriers.len()).filter(|&c| self.other(&self.carriers[c], d.component)).map(|c| self.shift_slope(c, g, region, true)).sum();
                (others, self.wave_of(d.component))
            });
            let rate = region.speed * (slopes[d.excited] + others) + wave * acceleration;
            let actual = gate(of.size, of.distance(d.gamma), rate);
            let place = &mut out[d.group];
            place[0] += actual.saturation;
            if d.level == Level::Bare {
                let own = gate(of.size + (of.centre - d.detuning).abs() + of.moved, d.bare, rate);
                place[2] = place[2].max(own.saturation - actual.saturation);
                place[3] = place[3].max(own.sweep);
            } else {
                place[1] = place[1].max(actual.sweep);
            }
        }
        // The tails: each place's shares with every weaker one's; a bare drive's own share in its own's stead.
        for place in (0..out.len()).rev() {
            let tail = out[place][0] + out.get(place + 1).map_or(0.0, |weaker| weaker[0]);
            out[place][0] = tail;
            out[place][2] += tail;
        }
        out
    }

    /// A ground state's drives' gate at a place of the tail rule's order over a region (`folds_made`): [saturation, sweep]
    /// of its drives at the atom, then of its bare ones. The region keeps the order's first `places` places -- those a
    /// proof over many cells still asks (`Model::gates`); every place's value is its own on the region whatever is
    /// kept --, and a place beyond them is made again.
    pub(crate) fn folds(&self, g: usize, place: usize, region: &Region, places: usize) -> [f64; 4] {
        let kept = region.drives[g].get_or_init(|| {
            let mut made = self.folds_made(g, region);
            made.truncate(places);
            made.shrink_to_fit();
            made
        });
        kept.get(place).copied().unwrap_or_else(|| self.folds_made(g, region)[place])
    }

    /// The bound over a region of the denominator of a drive e ← s folded at the atom at the detuning δ: its two levels'
    /// distance there, |E_e − E_s − δ| at the centre less their slack, what the eliminations' levels average of them and
    /// the excited state's Gershgorin radius (eliminate.rs `CellEnergies`), across Γ_e/2.
    pub(crate) fn denominator_over(&self, region: &Region, e: usize, s: usize, detuning: f64) -> f64 {
        let energies = self.energies(region);
        let apart = (energies.centre[e] - energies.centre[s] - detuning).abs() - energies.slack[e] - energies.slack[s] - energies.averaged[e] - energies.averaged[s] - energies.radius[e];
        C64::new(apart.max(0.0), 0.5 * self.decay[(e - GROUND, e - GROUND)].re).abs()
    }

    /// A coupling's (`Unit::Term`: a ← b at the model's beat `beat`) saturation and sweep over a region (module documentation,
    /// # A coupling of two states), made once per region. Its size: the constant part's value |H_ab + Σ_ji c_ji C_ji[a, b]| at
    /// the centre (H_ab the two states' own coupling with every light off, a static one's) plus Σ |C_ji[a, b]| times each
    /// pair's radius (reach.rs `PairSpan`), at most every beam at its largest amplitude and in phase; and the part at the atom, per excited state ½ |u'_a[e]| |u_b[e]| (1/σ_eb + 1/σ'_ea), u the two frequencies'
    /// folded couplings' coherent bounds and σ the denominator's bound of each drive folded at the atom (the bare drives'
    /// halves are the constants'). Its distance: |β + E_a − E_b| over the region -- the constant forms' difference, the two
    /// states' own --, less the Doppler shift of its beam pairs' largest wave vector q at the region's speeds and the
    /// two states' shifts by their drives folded at the atom (`near_shift`); its rate v |∇(E_a − E_b)| + q a_max, the two states' constant energies
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
    /// `Model::bounds_over_in_parallel`) -- a ground state's drive's with every weaker one's share on the same cell: what a
    /// cell's centre attains is a value there, never a bound from elsewhere. A dressing folds nothing and has no gate; a
    /// unit of nothing in this model has nothing to fold.
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
                // A dressing folds nothing; a fold's order is no fold of its own: where the fold exists, so does its next order.
                Unit::Dressing { .. } | Unit::Second { .. } => Read::Nothing,
                Unit::Coherence { ref first, ref second, .. } => Read::Coherence { unit: u, apart: self.omega(&first.plus(-1, second)).abs() - self.spread() },
            })
            .collect();
        // The largest wave number of the light's beams: a coherence's two photons'.
        let wave = 2.0 * self.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max);
        let read = |of: &Read, region: &Region, places: usize| -> [f64; 2] {
            match of {
                Read::Fold { ground, place, bare } => {
                    let made = self.folds(*ground, *place, region, places);
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
        let everywhere: Vec<[f64; 2]> = reads.iter().map(|of| read(of, &whole, usize::MAX)).collect();
        let open: Vec<usize> = (0..units.len()).filter(|&u| !(everywhere[u][0] < EXISTS && everywhere[u][1] < EXISTS)).collect();
        // Per ground state the places its open drives' gates ask of a cell.
        let mut places = [0; GROUND];
        for &u in &open {
            if let Read::Fold { ground, place, .. } = reads[u] {
                places[ground] = places[ground].max(place + 1);
            }
        }
        let mut quantities: Vec<Box<dyn Fn(&Region) -> f64 + Sync + '_>> = vec![];
        let mut shared: HashMap<(usize, usize, bool), usize> = HashMap::new();
        // Per open unit where its two bounds lie among the quantities: the saturation's, then the sweep's.
        let asked: Vec<usize> = open
            .iter()
            .map(|&u| {
                let of = &reads[u];
                let (key, asks) = if let Read::Fold { ground, place, bare } = *of { (Some((ground, place, bare)), places[ground]) } else { (None, 0) };
                if let Some(found) = key.and_then(|k| shared.get(&k)) {
                    return *found;
                }
                let first = quantities.len();
                quantities.push(Box::new(move |r: &Region| read(of, r, asks)[0]));
                quantities.push(Box::new(move |r: &Region| read(of, r, asks)[1]));
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
        let middle = self.middle_of(&plan, sample.at.t, &amplitudes, Some(&near));
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
            Unit::Second { ground } => self.order_error(*ground, at),
            Unit::Term { a, b, beat } => {
                let m2 = self.beat_index(beat).map_or(0.0, |index| 0.5 * self.term_bound(&self.forms.term(self, *a, *b, index), *a, *b, index, &at.region).0);
                let p = &at.populations;
                UnitError { parameter: m2, population: m2 * (p[*a] + p[*b]), temperature: m2 * (p[*a] + p[*b]) }
            }
            Unit::Dressing { carrier } => self.dressing_error(*carrier, at),
            Unit::Coherence { .. } => self.coherence_error(unit, at),
        }
    }

    /// The error at a sample of the second order of a cluster's fold at the atom (`Unit::Second`; module documentation,
    /// # The errors, # The fold's order): Σ s p_g over the drives folded at the atom from the sample's ground states, s =
    /// 2|R[e, g]|² each one's share -- on T, the force's relative error of that order; on the populations, what of the
    /// rates' error s the fourth order takes away, s less its own (`ORDER_RATES` s). Zero off the cluster, and where its
    /// fold is the fourth order's.
    fn order_error(&self, ground: usize, at: &Measured) -> UnitError {
        let cluster = self.membership[ground];
        let Some(frequency) = self.fold_frequency(cluster).filter(|_| at.sample.cluster == cluster && self.fourth[cluster].is_none()) else {
            return UnitError { parameter: 0.0, population: 0.0, temperature: 0.0 };
        };
        let component = Component { frequency, counter: false };
        let c = &self.clusters[cluster];
        let (mut sum, mut held) = (0.0, 0.0);
        if let Some(response) = self.carrier_of(frequency).and_then(|ci| at.responses[ci].as_ref()) {
            for (y, &g) in c.states.iter().enumerate() {
                held += at.populations[g];
                for x in (0..EXCITED).filter(|&x| self.level(component, GROUND + x, g) == Level::Atom) {
                    sum += 2.0 * response[(x, y)].norm_sqr() * at.populations[g];
                }
            }
        }
        UnitError { parameter: if held > 0.0 { sum / held } else { 0.0 }, population: (1.0 - ORDER_RATES) * sum, temperature: sum }
    }

    /// The error at a sample of a drive folded with the bare levels (module documentation, # Levels, # The errors): its own
    /// share s = 2|R⁰[e, g]|² of its ground state, as every fold's -- s p_g on the populations, (s + a) p_g on T, a its
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
        UnitError { parameter: epsilon, population: s * p + rates, temperature: (s + sweep) * p + rates + shifts }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::model::Point;
    use crate::request::Request;

    /// A scene's model (a reference scene by name, or a native request by its path) closed over its request's reach (solve.rs `attempt`: every unit whose gate fails kept).
    fn closed(scene: &str) -> (Model, [f64; 3]) {
        let path = if scene.ends_with(".json") { std::path::PathBuf::from(scene) } else { std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{scene}.json")) };
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let state = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let mut kept: Vec<Unit> = vec![];
        loop {
            let model = Model::new(&request, &light, &kept).unwrap();
            let trap = model.trap(&reference, &model.prepared()).unwrap();
            let reach = model.reach(&trap, &reference, request.temperature_uk, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
            let units = model.units();
            let failing: Vec<Unit> = units.iter().zip(model.gates(&units, &reach)).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
            if failing.is_empty() {
                return (model, trap.centre);
            }
            kept.extend(failing);
        }
    }

    /// What a folded drive does to its ground level is within its own gate's bounds (module documentation, # What a
    /// folded drive does to its level): on the lin-perp-lin scene at 160 mW/cm² a beam, closed, at points across a fringe
    /// of the cooling light and off the trap's axis, the shift of every ground state by its drives folded at the atom --
    /// the elimination's own, −Re Σ_e V[e, g]* R[e, g] over the carriers -- is within `near_shift` on the point, and its
    /// slope along the fringes and across them (a central difference) within `near_slope`. The bounds are the drives'
    /// own: the shift's is below a twentieth of what the coupling would be across Γ/2, a resonant drive's distance.
    #[test]
    fn a_folded_drive_s_shift_and_its_slope_are_within_their_bounds() {
        let (model, centre) = closed("R4_LIN_4");
        let shift = |cluster: usize, place: usize, r: [f64; 3]| -> f64 {
            let c = &model.clusters[cluster];
            let psi: Vec<C64> = (0..c.states.len()).map(|x| if x == place { C64::ONE } else { C64::ZERO }).collect();
            let sample = Sample { cluster, at: Point { r, kappa: [0.0; 3], t: 0.0 }, psi };
            let at = model.measured(&sample, None);
            let g = c.states[place];
            let mut out = 0.0;
            for (cc, response) in model.carriers.iter().zip(&at.responses) {
                let Some(response) = response else { continue };
                for e in 0..EXCITED {
                    let v: C64 = cc.beams.iter().zip(&cc.folded).map(|(&l, m)| at.amplitudes[l].value * m[(GROUND + e, g)]).sum();
                    out -= (v.conj() * response[(e, place)]).re;
                }
            }
            out
        };
        let (mut largest, mut steepest, mut tightest) = (0.0_f64, 0.0_f64, f64::INFINITY);
        for offset in [[0.0, 0.0, 0.0], [0.0, 0.0, 0.05], [0.0, 0.0, 0.1], [0.0, 0.0, 0.17], [0.0, 0.0, 0.25], [0.3, 0.0, 0.07], [0.0, 0.4, 0.31]] {
            let r: [f64; 3] = std::array::from_fn(|i| centre[i] + offset[i]);
            let region = model.point_region(r, 0.0);
            for (cluster, c) in model.clusters.iter().enumerate() {
                for (place, &g) in c.states[..c.ground].iter().enumerate() {
                    let (here, bound) = (shift(cluster, place, r), model.near_shift(g, &region));
                    assert!(here.abs() <= bound * (1.0 + 1e-9), "the shift {here} of ground state {g} at {r:?} beyond its bound {bound}");
                    let coupling: f64 = model.across(g, &region).iter().zip(&region.forms.drives(&model)[g]).filter(|of| of.1.level == Level::Atom).map(|(of, d)| of.size * of.size / (0.5 * d.gamma)).sum();
                    assert!(bound <= coupling / 20.0, "the bound {bound} of ground state {g} against its coupling across Γ/2, {coupling}");
                    largest = largest.max(here.abs());
                    if here.abs() > 1.0 {
                        tightest = tightest.min(bound / here.abs());
                    }
                    let step = 1e-4;
                    for axis in [0, 2] {
                        let moved = |sign: f64| {
                            let mut at = r;
                            at[axis] += sign * step;
                            shift(cluster, place, at)
                        };
                        let slope = (moved(1.0) - moved(-1.0)) / (2.0 * step);
                        let bound = model.near_slope(g, &region);
                        assert!(slope.abs() <= bound * (1.0 + 1e-6) + 1e-6, "the slope {slope} of ground state {g}'s shift along axis {axis} at {r:?} beyond its bound {bound}");
                        steepest = steepest.max(slope.abs());
                    }
                }
            }
        }
        eprintln!("the largest shift {largest:.2} rad/µs, the steepest slope {steepest:.1} rad/µs/µm, the least bound over its shift {tightest:.2}");
        assert!(largest > 20.0 && steepest > 100.0, "the points read a light shift: {largest}, {steepest}");
    }

    /// The slow states of a cluster's generator at a point, the atom at rest: its eigenvalues nearest the real axis, as
    /// many as it has ground states, by their real parts.
    fn slow(model: &Model, cluster: usize, r: [f64; 3]) -> Vec<C64> {
        use crate::step::Middle as _;
        let c = &model.clusters[cluster];
        let middle = model.assemble(cluster, r, 0.0, &model.running());
        let mut h = Mat::zeros(c.states.len(), c.states.len());
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
        let mut values = crate::linalg::eigen(&h).expect("a cluster's generator has its eigenvalues").values;
        values.sort_by(|a, b| b.im.total_cmp(&a.im));
        values.truncate(c.ground);
        values.sort_by(|a, b| a.re.total_cmp(&b.re));
        values
    }

    /// The exact elimination's slow states at a point, the atom at rest: of the no-jump generator of a cluster's ground
    /// states with every excited state -- the cluster's own generator without its drives folded at the atom, those
    /// drives' couplings V, and the excited block K − δ the eliminations take there --, the eigenvalues nearest the real
    /// axis, by their real parts.
    fn eliminated(model: &Model, cluster: usize, carrier: usize, r: [f64; 3]) -> Vec<C64> {
        let n = model.clusters[cluster].ground;
        let mut values = crate::linalg::eigen(&no_jump(model, cluster, carrier, r)).expect("the generator has its eigenvalues").values;
        values.sort_by(|a, b| b.im.total_cmp(&a.im));
        values.truncate(n);
        values.sort_by(|a, b| a.re.total_cmp(&b.re));
        values
    }

    /// The no-jump generator of a cluster's ground states with every excited state at a point, the atom at rest
    /// (`eliminated`): the ground states first.
    fn no_jump(model: &Model, cluster: usize, carrier: usize, r: [f64; 3]) -> Mat {
        use crate::step::Middle as _;
        let (c, cc) = (&model.clusters[cluster], &model.carriers[carrier]);
        let n = c.ground;
        assert!(c.frames.iter().all(|frame| *frame == c.frames[0]), "the cluster's ground states share a frame");
        let amplitudes = model.light.amplitudes(r);
        let middle = model.middle_of(&model.plan(cluster, &model.running()), 0.0, &amplitudes, None);
        let mut ground = Mat::zeros(n, n);
        middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut ground);
        let h = &model.far_hamiltonian(&amplitudes, &model.running(), 0, &model.coherent_dressing())[0];
        let frame = model.omega(&c.frames[0]) + cc.detuning;
        Mat::from_fn(n + EXCITED, n + EXCITED, |x, y| match (x < n, y < n) {
            (true, true) => ground[(x, y)],
            (false, true) => cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + x - n, c.states[y])]).sum(),
            (true, false) => cc.beams.iter().zip(&cc.folded).map(|(&l, m)| amplitudes[l].value * m[(GROUND + y - n, c.states[x])]).sum::<C64>().conj(),
            (false, false) => h[(GROUND + x - n, GROUND + y - n)] - model.decay[(x - n, y - n)] * C64::new(0.0, 0.5) - if x == y { C64::real(frame) } else { C64::ZERO },
        })
    }

    /// Where a jump lands, against the exact elimination (model.rs `Model::landed`; assemble.rs, # The fourth order): the
    /// lin-perp-lin scenes, the cluster F=2 m = −2, 0, 2 with its D2 drives at the atom at the fourth order. A bare
    /// ground state f is the slow subspace's part of it and a fast part that decays at once: exactly, the part is
    /// P (f, 0) with P the spectral projector of the no-jump generator onto its three slow eigenvectors, the prompt
    /// photon's probability 1 − ‖P (f, 0)‖²/‖f‖², and the slow state's coordinates (1 + ½X) times the part's ground
    /// amplitudes. The model's p and Π f are these to the next order, s² -- p itself is about half the landing state's
    /// share --, and the norm is kept on average by construction: ‖Π f‖² + p ‖f‖² = ‖f‖².
    #[test]
    fn a_jump_s_landing_is_the_exact_elimination_s_slow_part() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (closed, centre) = closed(scene);
            let states = closed.clusters[closed.membership[3]].states.clone();
            let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
            let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
            let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }, Unit::Second { ground: 3 }]).collect();
            let fourth = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
            let (cluster, carrier) = (fourth.membership[3], fourth.carrier_of(cooling.frequency).unwrap());
            let plan = fourth.plan(cluster, &fourth.running());
            assert!(plan.fourth.is_some(), "{scene}: the cluster's fold is the fourth order's");
            let (mut s, mut worst, mut turned) = (0.0_f64, 0.0_f64, 0.0_f64);
            for offset in [0.0, 0.05, 0.1, 0.17, 0.26] {
                let r = [centre[0], centre[1], centre[2] + offset];
                let at = Point { r, kappa: [0.0; 3], t: 0.0 };
                let exact = crate::linalg::eigen(&no_jump(&fourth, cluster, carrier, r)).unwrap();
                let mut order: Vec<usize> = (0..exact.values.len()).collect();
                order.sort_by(|&a, &b| exact.values[b].im.total_cmp(&exact.values[a].im));
                for f in [[C64::ONE, C64::ZERO, C64::ZERO], [C64::ZERO, C64::ONE, C64::ZERO], [C64::real(0.6), C64::new(0.0, 0.5), C64::real(-0.4)]] {
                    let size = crate::linalg::norm_sqr(&f);
                    // P (f, 0) = Σ_slow r_i (R⁻¹ (f, 0))_i.
                    let mut part = vec![C64::ZERO; 3 + EXCITED];
                    for &i in &order[..3] {
                        let weight: C64 = (0..3).map(|y| exact.inverse[(i, y)] * f[y]).sum();
                        for (x, z) in part.iter_mut().enumerate() {
                            *z += exact.vectors[(x, i)] * weight;
                        }
                    }
                    let prompt = 1.0 - crate::linalg::norm_sqr(&part) / size;
                    let (slow, p) = fourth.landed(&plan, None, &at, &f);
                    assert!((crate::linalg::norm_sqr(&slow) + p * size - size).abs() <= 1e-12 * size, "{scene}: the norm on average");
                    worst = worst.max((p - prompt).abs());
                    s = s.max(2.0 * prompt);
                    // The slow state's direction: Π f against the exact part's ground amplitudes (to their common first
                    // order they differ by the metric's half, of the order of p itself; the angle between them is s²).
                    let overlap: C64 = (0..3).map(|x| slow[x].conj() * part[x]).sum();
                    let ground: f64 = part[..3].iter().map(|z| z.norm_sqr()).sum();
                    turned = turned.max(1.0 - overlap.norm_sqr() / (crate::linalg::norm_sqr(&slow) * ground));
                }
            }
            eprintln!("{scene}: the largest prompt photon's probability {:.4}; the model's differs by at most {worst:.2e}; the slow state turned by at most {turned:.2e}", 0.5 * s);
            assert!(worst <= 1.5 * s * s, "{scene}: p within the next order: {worst} for s = {s}");
            assert!(turned <= s * s, "{scene}: the slow state within the next order: {turned} for s = {s}");
        }
    }

    /// The fold's orders against the exact elimination, the atom frozen (module documentation, # The fold's order;
    /// assemble.rs, # The fourth order): the lin-perp-lin scenes at 40, 80 and 160 mW/cm² a beam, 20 linewidths red, on
    /// ⁸⁷Rb's real F′ = 1, 2, 3 in the tweezer. The cluster F=2 m = −2, 0, 2 with every cooling drive into F′=3 folded at
    /// the atom, at the second order and at the fourth, against the slow eigenvalues of the same ground states with all
    /// the excited states (`eliminated`; the dressing coherent, so that the levels are the same in both), at points across
    /// a fringe. Of the three slow states, with s the largest share 2 Σ_e |R[e, g]|² met: the potentials err by s/2 of the
    /// light's largest shift there at the second order and by s²/2 at the fourth; the force on them (the potentials'
    /// slope along the fringes, a central difference), relative to the steepest potential met, by s and by 3/2 s²; their
    /// decay rates by 3/2 s and by the next order. And the fourth order's generator is a Lindblad one as the second's: a
    /// state's norm falls at its photons' rate.
    #[test]
    fn the_fourth_order_s_potentials_forces_and_rates_against_the_exact_elimination() {
        for scene in ["R4_LIN_1", "R4_LIN_2", "R4_LIN_4"] {
            let (closed, centre) = closed(scene);
            let states = closed.clusters[closed.membership[3]].states.clone();
            assert_eq!(states, [3, 5, 7], "the cluster of F=2 m = −2, 0, 2");
            // Every cooling drive from the cluster into D2 -- F′ = 1, 2, 3 -- at the atom (the closure leaves the weaker
            // bare), the dressing coherent.
            let cooling = Component { frequency: closed.driven.iter().find(|d| d.2 == 3 && closed.level(d.0, d.1, d.2) == Level::Atom).expect("a drive at the atom").0.frequency, counter: false };
            let line = closed.driven.iter().filter(|&&(c, e, g)| c == cooling && states.contains(&g) && closed.bare_distance(c, e, g) < 5000.0);
            let more: Vec<Unit> = line.map(|&(component, excited, ground)| Unit::Bare { component, excited, ground }).chain([Unit::Dressing { carrier: cooling }]).collect();
            let second = closed.rekept(&[closed.kept.clone(), more].concat()).unwrap();
            let fourth = second.rekept(&[second.kept.clone(), vec![Unit::Second { ground: 3 }]].concat()).unwrap();
            let (cluster, carrier) = (second.membership[3], second.carrier_of(cooling.frequency).unwrap());
            assert_eq!(second.fold_frequency(cluster), Some(cooling.frequency));
            assert_eq!((second.fourth[cluster], fourth.fourth[fourth.membership[3]]), (None, Some(cooling.frequency)));
            assert!(second.units().contains(&Unit::Second { ground: 3 }) && !fourth.units().contains(&Unit::Second { ground: 3 }));
            let (mut s, mut largest, mut steepest) = (0.0_f64, 0.0_f64, 0.0_f64);
            let (mut potential, mut force, mut rate) = ([0.0_f64; 2], [0.0_f64; 2], [0.0_f64; 2]);
            let step = 1e-4;
            for offset in [0.0, 0.03, 0.07, 0.1, 0.15, 0.2, 0.26, 0.33] {
                let r = [centre[0], centre[1], centre[2] + offset];
                let read = |values: &dyn Fn(f64) -> Vec<C64>| -> (Vec<C64>, Vec<f64>) {
                    let (up, down) = (values(step), values(-step));
                    (values(0.0), up.iter().zip(&down).map(|(a, b)| (a.re - b.re) / (2.0 * step)).collect())
                };
                let (exact, pull) = read(&|by| eliminated(&second, cluster, carrier, [r[0], r[1], r[2] + by]));
                let level = {
                    use crate::step::Middle as _;
                    let middle = second.middle_of(&second.plan(cluster, &second.running()), 0.0, &second.light.amplitudes(r), None);
                    let mut h = Mat::zeros(3, 3);
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                    crate::linalg::eigen(&h).unwrap().values
                };
                let mean = level.iter().map(|z| z.re).sum::<f64>() / 3.0;
                let shift = exact.iter().map(|z| (z.re - mean).abs()).fold(0.0, f64::max);
                let decay = exact.iter().map(|z| z.im.abs()).fold(0.0, f64::max);
                largest = largest.max(shift);
                steepest = pull.iter().map(|f| f.abs()).fold(steepest, f64::max);
                for (order, model) in [&second, &fourth].into_iter().enumerate() {
                    let (values, slopes) = read(&|by| slow(model, model.membership[3], [r[0], r[1], r[2] + by]));
                    for i in 0..3 {
                        potential[order] = potential[order].max((values[i].re - exact[i].re).abs() / shift);
                        force[order] = force[order].max((slopes[i] - pull[i]).abs());
                        rate[order] = rate[order].max((values[i].im - exact[i].im).abs() / decay);
                    }
                }
                // The largest share at the point, and the fourth order's hazard identity on a state of the cluster.
                let psi: Vec<C64> = (0..3).map(|x| C64::cis(1.3 * x as f64) * (0.2 + 0.3 * x as f64)).collect();
                let here = Point { r, kappa: [0.0; 3], t: 0.0 };
                let sample = Sample { cluster, at: here, psi: psi.clone() };
                let at = second.measured(&sample, None);
                let response = at.responses[carrier].as_ref().unwrap();
                s = (0..3).map(|y| 2.0 * (0..EXCITED).map(|x| response[(x, y)].norm_sqr()).sum::<f64>()).fold(s, f64::max);
                let plan = fourth.plan(fourth.membership[3], &fourth.running());
                let photons: f64 = fourth.photons_of(&plan, None, &here, &psi, crate::model::photon::Gradients::None).0.iter().map(|p| p.amplitudes.iter().map(|u| crate::linalg::norm_sqr(u)).sum::<f64>()).sum();
                let mut h = Mat::zeros(3, 3);
                {
                    use crate::step::Middle as _;
                    let middle = fourth.assemble(fourth.membership[3], r, 0.0, &fourth.running());
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                }
                let falls: f64 = (0..3).flat_map(|x| (0..3).map(move |y| (x, y))).map(|(x, y)| (psi[x].conj() * C64::I * (h[(x, y)] - h[(y, x)].conj()) * psi[y]).re).sum();
                // To the product of the fourth order and the first-order Doppler term (here the recoil shift, 3e-5 of the
                // detuning, times s), which the photons' product holds and the middle's decay, made at rest, does not.
                assert!((falls - photons).abs() <= 1e-5 * photons, "{scene}: the norm falls at {falls}, the photons' rate is {photons}");
            }
            let force = force.map(|f| f / steepest);
            eprintln!("{scene}: s = {s:.4}, the largest shift {largest:.2} rad/µs: potentials {:.2e} (second), {:.2e} (fourth); forces {:.2e}, {:.2e}; decay rates {:.2e}, {:.2e}", potential[0], potential[1], force[0], force[1], rate[0], rate[1]);
            assert!((0.4 * s..0.6 * s).contains(&potential[0]) && potential[1] <= 0.6 * s * s, "{scene}: the potentials err by s/2 and by s²/2: {potential:?} for s = {s}");
            assert!((0.5 * s..1.0 * s).contains(&force[0]) && force[1] <= 1.5 * s * s, "{scene}: the forces err by s and by 3/2 s² at most: {force:?} for s = {s}");
            assert!((1.2 * s..1.8 * s).contains(&rate[0]) && rate[1] <= 2.5 * s * s, "{scene}: the decay rates err by 3/2 s and by the next order: {rate:?} for s = {s}");
        }
    }

    /// What the model's part of a step costs at the second order and at the fourth (measured; `SCENE`, default R4_LIN_1):
    /// per cluster of the closed model whose fold has a fourth order, from the two models' tables, one middle at a
    /// thermal point, its path, sixteen generators, eight forces, three tones' forms and the state's scattering -- a
    /// product step's calls --, in µs, with the middle's and the scattering's share; and the tables' reals a node.
    #[test]
    #[ignore]
    fn the_fourth_order_s_cost_a_step() {
        use crate::model::Run;
        use crate::step::Middle as _;
        let name = std::env::var("SCENE").unwrap_or("R4_LIN_1".into());
        let (closed, centre) = closed(&name);
        let grounds: Vec<usize> = (0..closed.clusters.len()).filter(|&c| closed.fold_frequency(c).is_some()).map(|c| closed.clusters[c].states[0]).collect();
        let fourth = closed.rekept(&[closed.kept.clone(), grounds.iter().map(|&ground| Unit::Second { ground }).collect()].concat()).unwrap();
        let reference: [f64; GROUND] = std::array::from_fn(|s| if s == grounds[0] { 1.0 } else { 0.0 });
        for (order, model) in [("second", &closed), ("fourth", &fourth)] {
            let trap = model.trap(&reference, &model.prepared()).unwrap();
            let reach = model.reach(&trap, &reference, 30.0, crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64));
            let started = std::time::Instant::now();
            let tables = model.tabulate(&reach, 1 << 20);
            let built = started.elapsed();
            let run = Run::tabulated(model, &tables, reference);
            for &ground in &grounds {
                let cluster = model.membership[ground];
                let n = model.clusters[cluster].states.len();
                let psi: Vec<C64> = (0..n).map(|x| C64::cis(0.7 * x as f64) * (1.0 / n as f64).sqrt()).collect();
                let rho: Vec<C64> = (0..n * n).map(|i| psi[i / n] * psi[i % n].conj()).collect();
                let (rounds, mut h, mut sink) = (20_000, Mat::zeros(n, n), 0.0);
                let mut times = [std::time::Duration::ZERO; 3];
                for round in 0..rounds {
                    let r = [centre[0] + 0.01, centre[1], centre[2] + 0.39 * (round % 97) as f64 / 97.0];
                    let at = Point { r, kappa: [0.3, -0.2, 2.5], t: 0.0 };
                    let from = std::time::Instant::now();
                    let middle = run.middle(cluster, r, 0.0);
                    times[0] += from.elapsed();
                    let path = middle.path(at.kappa, at.kappa.map(|k| k * model.hbar_over_mass));
                    for node in 0..16 {
                        middle.generator(&path, 0.004 * (node as f64 - 7.5), &mut h);
                        sink += h.data[0].re;
                    }
                    for node in 0..8 {
                        sink += middle.force(&path, 0.008 * (node as f64 - 3.5), &rho)[2];
                    }
                    middle.tones(&path, &rho, &mut |_, f| sink += f[2]);
                    times[1] += from.elapsed();
                    let scattering = std::time::Instant::now();
                    sink += run.scattering_at(cluster, &at, &psi, None).force[2];
                    times[2] += scattering.elapsed();
                }
                let each = |t: std::time::Duration| t.as_secs_f64() * 1e6 / rounds as f64;
                let table = tables.of(cluster, &model.running()).map_or(0, |t| t.table.width());
                let counted = run.middle(cluster, [centre[0] + 0.01, centre[1], centre[2] + 0.1], 0.0).work;
                eprintln!("{name} {order}: cluster of ground state {ground}: the middle's counted work {counted:.0} complex multiply-adds, {:.2} ns each", each(times[0]) * 1e3 / counted);
                eprintln!("{name} {order}: cluster of ground state {ground} ({n} states): the model's part of a step {:.2} µs (its middle {:.2}, its scattering {:.2}); table {table} reals a node, {} tables in {built:.2?} [{sink:.3e}]", each(times[1]) + each(times[2]), each(times[0]), each(times[2]), tables.tables.len());
            }
        }
    }
}
