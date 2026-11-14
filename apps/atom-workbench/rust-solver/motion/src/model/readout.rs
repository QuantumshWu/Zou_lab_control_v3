//! The readout (SPEC §1.7): the 32 populations of a cluster's state, those of its slow state x = S⁻¹ψ on its states plus the
//! folded excited populations; and the model in words.
//!
//! # The readout
//! A component's folded drives' excited amplitude from ground state b is −R[:,b] x_b -- its bare drives' and those folded at
//! the atom, each a channel (far.rs, eliminate.rs) --, its channel's excited frequency θ_b + ν (frames). Two channels at
//! one frequency are co-resonant; the others beat and average out (criterion two). Co-resonance is an equivalence of the
//! places (channel p, ground place b) -- one excited frequency θ_b + ν_p a class --: a class's response is
//! C_k = Σ_{(p, b) ∈ k} R_p[:, b] e_bᵀ and its excited amplitude A_k = C_k x.
//!
//! The state the dynamics carries is the normalized ψ = S x (assemble.rs, # The normalized generator; ELIMINATION v7),
//! and the readout inverts the generator's own S: x = S⁻¹ψ, S⁻¹ = Σ_{j≤J} b_j X^j of the slow state's one metric X --
//! the near light's static pair forms at the point, the same matrix a node's is (assemble.rs `Model::bent_at`), whose
//! largest eigenvalue is the series' ratio (elim.rs `Fold::evaluated_share`, `Series`) -- to the model's normalization
//! order J; an explicit state's amplitude its own. (It read before a metric of its own, Y = Σ_k C_k†C_k of every
//! channel with its Doppler partner: a second definition of the metric, which does not invert the generator's S.) The
//! populations are x's |x_b|², the excited states' Σ_k |A_k[i]|² over every channel and the explicit states', over their
//! sum: what is read is the slow state itself in the coordinates the dynamics holds.

use super::assemble::{Held, Plan, Site};
use super::far::EXCITED;
use super::tabulate::NearTable;
use super::{Beat, Cluster, Component, Level, Lights, Model, Point, Sample, Unit, GROUND, STATES};
use crate::field::Amplitude;
use crate::linalg::C64;
use std::collections::HashSet;

impl Model {
    /// The components with bare drives under a light set, in the frequencies' order: (frequency, side), side 0 the
    /// co-rotating and 1 the counter-rotating.
    fn bare_components<'a>(&'a self, lights: &'a Lights) -> impl Iterator<Item = (usize, usize)> + 'a {
        self.tones.iter().enumerate().filter(move |(w, _)| lights.on(*w)).flat_map(|(w, tone)| (0..2).filter(move |&side| tone.bare[side]).map(move |side| (w, side)))
    }

    /// A component's bare drives' response on a cluster at a point (far.rs `Tone::responses`), added onto `response`
    /// (row-major, EXCITED × the cluster's ground places): R = Σ_l a_l R_l (co-rotating) or Σ_l a_l* R_l (counter-rotating).
    fn bare_response(&self, c: &Cluster, amplitudes: &[Amplitude], frequency: usize, side: usize, response: &mut [C64]) {
        let (f, tone) = (&self.light.frequencies[frequency], &self.tones[frequency]);
        for (j, &l) in f.beams.iter().enumerate() {
            let a = if side == 0 { amplitudes[l].value } else { amplitudes[l].value.conj() };
            // Row by row of R_l (EXCITED × GROUND), the cluster's ground places picked from it.
            for (to, from) in response.chunks_exact_mut(c.ground).zip(tone.responses[side][j].data.chunks_exact(GROUND)) {
                for (z, &b) in to.iter_mut().zip(&c.states[..c.ground]) {
                    *z += a * from[b];
                }
            }
        }
    }

    /// The carriers whose drives folded at the atom make a channel on a plan's cluster: the lit ones, where the plan
    /// holds near forms.
    fn folded_carriers<'a>(&'a self, plan: &'a Plan) -> impl Iterator<Item = usize> + 'a {
        (0..self.carriers.len()).filter(move |&d| plan.near.len() > 0 && plan.lights.on(self.carriers[d].frequency))
    }

    /// The readout (module documentation): the 32 populations of a cluster's state ψ at a point, every light on, the
    /// forms at the atom from the exact elimination there.
    pub fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.readout_of(&self.plan(cluster, &self.running()), None, at, psi)
    }

    /// The readout of a state of a plan's cluster under the plan's lights, the running ones, the forms at the atom from
    /// their table where one is given (tabulate.rs `Model::near_responses`). Its channels are the bare drives' per
    /// component (`bare_components`) and then the carriers' (`folded_carriers`); their responses are a thread's spare's
    /// (assemble.rs `Held`) -- the beams, a table's responses, the carriers' field components and the channels, each
    /// written whole here before it is read --, so a readout from a table asks the heap for nothing.
    pub(crate) fn readout_of(&self, plan: &Plan, table: Option<&NearTable>, at: &Point, psi: &[C64]) -> [f64; STATES] {
        let c = &self.clusters[plan.cluster];
        let g = c.ground;
        let mut site = Held::<Site>::take();
        let count = self.channels_at(plan, table, at, &mut site);
        let Site { channels, excited, beams, normal, .. } = &mut *site;
        let size = EXCITED * g;
        // The co-resonant channels and places (p, b1) and (q, b2), θ_b1 + ν_p = θ_b2 + ν_q, and with them a place's class:
        // the first of the places co-resonant with it (itself among them). The cluster's own under the running lights,
        // exact, found once.
        let places = count * g;
        let first = self.resonances[plan.cluster].get_or_init(|| {
            let sums: Vec<Vec<Beat>> = self.channel_beats(plan).iter().map(|nu| c.frames[..g].iter().map(|theta| theta.plus(1, nu)).collect()).collect();
            let mut first = vec![usize::MAX; places];
            for (p, left) in sums.iter().enumerate() {
                for (q, right) in sums.iter().enumerate() {
                    for (b1, one) in left.iter().enumerate() {
                        for (b2, _) in right.iter().enumerate().filter(|(_, two)| one == *two) {
                            let at = &mut first[p * g + b1];
                            *at = (*at).min(q * g + b2);
                        }
                    }
                }
            }
            first
        });
        // Each class's response C_k = Σ_{(p, b) ∈ k} R_p[:, b] e_bᵀ (EXCITED × the ground places, at the class's first
        // place): its excited amplitude A_k = C_k x.
        let n = psi.len();
        excited.clear();
        excited.resize(places * size + 3 * n * n, C64::ZERO);
        let (classes, rest) = excited.split_at_mut(places * size);
        let (bent, scratch) = rest.split_at_mut(n * n);
        for (p, response) in channels.chunks_exact(size).enumerate() {
            for b in 0..g {
                let class = &mut classes[first[p * g + b] * size..][..size];
                for i in 0..EXCITED {
                    class[i * g + b] += response[i * g + b];
                }
            }
        }
        let held: Vec<usize> = (0..places).filter(|&k| first[k] == k).collect();
        // The slow state of the normalized coordinates ψ = S x (assemble.rs, # The normalized generator): x = S⁻¹ψ with
        // the generator's own S⁻¹ = Σ_{j≤J} b_j X^j of the one metric X at the point (`Model::bent_at`; its series'
        // ratio, X's largest eigenvalue, read by elim.rs `Series`), the identity where it bends nothing; every population
        // of it -- the ground places' |x_b|², the excited states' Σ_k |A_k|² over every channel, the explicit states' --
        // over their sum: the slow state's norm to the series' remainder, and the bare drives' excited populations, whose
        // folds are constants the generator's metric does not hold.
        let mut x = [C64::ZERO; STATES];
        x[..n].copy_from_slice(psi);
        if let Some(fourth) = plan.fourth.as_ref().filter(|fourth| fourth.bends()) {
            self.bent_at(plan, fourth, beams, normal, scratch, bent);
            for (b, z) in x.iter_mut().enumerate().take(n) {
                *z = (0..n).map(|y| bent[b * n + y] * psi[y]).sum();
            }
        }
        let mut out = [0.0; STATES];
        for (z, &s) in x.iter().zip(&c.states).take(psi.len()) {
            out[s] = z.norm_sqr();
        }
        for &k in &held {
            let class = &classes[k * size..(k + 1) * size];
            for i in 0..EXCITED {
                out[GROUND + i] += (0..g).map(|b| class[i * g + b] * x[b]).sum::<C64>().norm_sqr();
            }
        }
        let norm: f64 = out.iter().sum();
        out.iter_mut().for_each(|p| *p /= norm);
        out
    }

    /// The channels of a plan's cluster at a point (module documentation, # The readout), into `site.channels` -- each a
    /// response R, row-major, EXCITED × the cluster's ground places, one after another: the bare drives' per component
    /// (`bare_response`), then those folded at the atom per carrier, from their table where one is given, else from the
    /// exact elimination --, the beams' amplitudes at the point into `site.beams`; and the channels' number. Their
    /// excited frequencies' ν, in their order: `channel_beats`.
    fn channels_at(&self, plan: &Plan, table: Option<&NearTable>, at: &Point, site: &mut Site) -> usize {
        let c = &self.clusters[plan.cluster];
        self.light.amplitudes_into(at.r, 0, |_| true, &mut site.beams);
        // The responses of the drives folded at the atom: their table's at the point, or the exact elimination's.
        let folded = self.folded_carriers(plan).count();
        let exact = match table {
            Some(table) if folded > 0 && self.placed(table, site) => {
                self.near_responses(table, site);
                None
            }
            _ => (folded > 0).then(|| self.near_exact(plan, &site.beams, 0)),
        };
        // The slow state's one metric's forms, their values, where the generator's S bends anything: the readout
        // inverts the generator's own S (`Model::bent_at`).
        if plan.fourth.as_ref().is_some_and(|fourth| fourth.bends()) {
            self.normal_read(plan, if exact.is_none() { table } else { None }, exact.as_ref(), site, false);
        }
        let Site { beams: amplitudes, responses, weights, channels, .. } = site;
        let (far, size) = (self.bare_components(&plan.lights).count(), EXCITED * c.ground);
        channels.clear();
        channels.resize((far + folded) * size, C64::ZERO);
        for (p, (frequency, side)) in self.bare_components(&plan.lights).enumerate() {
            self.bare_response(c, amplitudes, frequency, side, &mut channels[p * size..(p + 1) * size]);
        }
        if folded > 0 {
            let start = plan.near.responding().start;
            let near = |i: usize| exact.as_ref().map_or_else(|| C64::new(responses[2 * i], responses[2 * i + 1]), |values| values.data[start + i]);
            let place = |carrier: usize| far + self.folded_carriers(plan).take_while(|&d| d < carrier).count();
            self.responses_each(plan, near, amplitudes, at.kappa, weights, |carrier, e, y, z| channels[place(carrier) * size + e * c.ground + y] += z);
        }
        far + folded
    }

    /// The frequency ν of every channel of a plan's cluster (`channels_at`), in their order: a bare component's own, a
    /// carrier's beat.
    fn channel_beats(&self, plan: &Plan) -> Vec<Beat> {
        let bare = self.bare_components(&plan.lights).map(|(frequency, side)| self.frequency(Component { frequency, counter: side == 1 }));
        bare.chain(self.folded_carriers(plan).map(|d| self.carriers[d].beat.clone())).collect()
    }

    /// The component of every channel of a plan's cluster (`channels_at`), in their order: a bare one's, a carrier's
    /// co-rotating one.
    fn channel_components(&self, plan: &Plan) -> Vec<Component> {
        let bare = self.bare_components(&plan.lights).map(|(frequency, side)| Component { frequency, counter: side == 1 });
        bare.chain(self.folded_carriers(plan).map(|d| Component { frequency: self.carriers[d].frequency, counter: false })).collect()
    }

    /// The state a sample of this model's has in the model `into`, which keeps this one's units and more (solve.rs,
    /// # An attempt's own: a keep's model decides first on the states its attempt left). A unit kept only joins states
    /// and makes excited states explicit (cluster.rs, # Clusters), so the sample's cluster lies in one of `into`'s, and
    /// the state there is the same atom's:
    /// - a state this cluster holds keeps its amplitude, ψ_s = e^{−iθ_s t} x_s in either model's frames (cluster.rs,
    ///   # Frames): x'_s = e^{i(θ'_s − θ_s) t} x_s at the sample's time;
    /// - an excited state explicit there and folded here holds what the fold makes of it through the drives that are
    ///   explicit there (module documentation, # The readout): their channels' −R_p[e, b] x_b, each at its excited
    ///   frequency θ_b + ν_p, so x'_e = −Σ_{p,b} R_p[e, b] e^{i(θ'_e − θ_b − ν_p) t} x_b over the channels p whose drive
    ///   e ← b `into` holds explicit -- the kept drive's channel, static in the frames that keep it, as it is; a drive
    ///   `into` still folds keeps its own excitation folded there, the readout's apart. Its population is then the
    ///   readout's folded one of those channels, and the rates and shifts its units' errors are shares of (elim.rs
    ///   `Model::readings`) are the same in both;
    /// - the other ground states of that cluster hold nothing: between two jumps the atom is in one cluster.
    ///
    /// The sample is read at its point, its own path's middle (Doppler frames at zero).
    pub(crate) fn carried(&self, sample: &Sample, into: &Model) -> Sample {
        let (from, t) = (&self.clusters[sample.cluster], sample.at.t);
        let cluster = into.cluster_of(from.states[0]);
        let to = &into.clusters[cluster];
        let mut psi = vec![C64::ZERO; to.states.len()];
        for (x, (&s, theta)) in sample.psi.iter().zip(from.states.iter().zip(&from.frames)) {
            if let Some(j) = to.states.iter().position(|&r| r == s) {
                psi[j] = *x * C64::cis(self.omega(&to.frames[j].plus(-1, theta)) * t);
            }
        }
        let folded: Vec<usize> = (to.ground..to.states.len()).filter(|&j| !from.states.contains(&to.states[j])).collect();
        if !folded.is_empty() {
            let plan = self.plan(sample.cluster, &self.running());
            let mut site = Held::<Site>::take();
            let count = self.channels_at(&plan, None, &sample.at, &mut site);
            let (beats, components, size) = (self.channel_beats(&plan), self.channel_components(&plan), EXCITED * from.ground);
            for j in folded {
                let e = to.states[j] - GROUND;
                let explicit = |p: usize, b: usize| into.level(components[p], to.states[j], from.states[b]) == Level::Explicit;
                psi[j] = (0..count).flat_map(|p| (0..from.ground).map(move |b| (p, b))).filter(|&(p, b)| explicit(p, b)).fold(C64::ZERO, |z, (p, b)| {
                    let phase = self.omega(&to.frames[j].plus(-1, &from.frames[b]).plus(-1, &beats[p])) * t;
                    z - site.channels[p * size + e * from.ground + b] * sample.psi[b] * C64::cis(phase)
                });
            }
        }
        Sample { cluster, at: sample.at, psi }
    }

    /// V_L(r; p) far from the trap (SPEC §1.7): every focused beam's amplitude zero, the plane waves' terms of one wave vector
    /// (their spatial mean; a term of q ≠ 0 averages out). The part at the atom is zero there: it is those shifts with the
    /// trap light on less with it off (potential.rs), and far from the trap the trap light is off but for its plane waves, whose
    /// part is the constants' above. A request whose trap light holds a plane wave is refused (SPEC §9): for every request
    /// the solve runs this is 0.
    pub fn asymptote(&self, p: &[f64; GROUND], lights: &Lights) -> f64 {
        let trap = self.light.trap;
        if !lights.on(trap) {
            return 0.0;
        }
        let (f, tone) = (&self.light.frequencies[trap], &self.tones[trap]);
        let n = f.beams.len();
        let mut v = 0.0;
        for j in 0..n {
            for i in 0..n {
                let (lj, li) = (f.beams[j], f.beams[i]);
                let plane = self.light.beams[lj].waist.is_none() && self.light.beams[li].waist.is_none();
                if plane && self.beam_wave[lj] == self.beam_wave[li] && lights.pair(trap, j, i) {
                    v += (0..GROUND).map(|b| (tone.pairs[j * n + i][(b, b)] * p[b]).re).sum::<f64>();
                }
            }
        }
        v
    }

    /// A state's label in the model basis (`manifold-F-m`), as the app writes it: "D2 F′=3 m=2", "基态 F=2 m=−1".
    pub(crate) fn state_label(&self, s: usize) -> String {
        let level = crate::request::Level::parse(&self.labels[s]).expect("a validated request's state labels parse (request.rs `Request::levels`)");
        let m = if level.m < 0 { format!("−{}", -level.m) } else { level.m.to_string() };
        match level.manifold {
            crate::request::Manifold::Ground => format!("基态 F={} m={}", level.f, m),
            crate::request::Manifold::D1 => format!("D1 F′={} m={}", level.f, m),
            crate::request::Manifold::D2 => format!("D2 F′={} m={}", level.f, m),
        }
    }

    /// A component's name: its beams' names, and which rotation.
    fn component_label(&self, c: Component) -> String {
        let names: Vec<&str> = self.light.frequencies[c.frequency].beams.iter().map(|&l| self.names[l].as_str()).collect();
        format!("{}{}", names.join("、"), if c.counter { "（反旋转项）" } else { "" })
    }

    /// A beat as the ledger writes it (`frequency_reading`), to the digits that tell the model's beats apart
    /// (`telling_apart`): `Model::beats` holds every beat a unit can carry (elim.rs `Model::candidates`). So no two of a
    /// run's items share a name; and to no fewer than the ledger's readings have (`ledger::READING`), so that a lone
    /// beat still reads as its value (32.4 THz, not 30 THz).
    fn beat_label(&self, beat: &Beat) -> String {
        let mhz = |beat: &Beat| self.omega(beat) / crate::units::RAD_PER_US_PER_MHZ;
        let beats: Vec<f64> = self.beats.iter().map(|(beat, _)| mhz(beat)).collect();
        frequency_reading(mhz(beat), telling_apart(&beats).max(crate::ledger::READING))
    }

    /// A unit's label for the ledger (Chinese).
    pub fn label(&self, unit: &Unit) -> String {
        match unit {
            Unit::Bare { component, excited, ground } => {
                format!("{} → {} 的驱动：按裸能级绝热消去（{}）", self.state_label(*ground), self.state_label(*excited), self.component_label(*component))
            }
            Unit::Drive { component, excited, ground } => {
                format!("{} → {} 的驱动：按原子所在处的能级（含光移）绝热消去（{}）", self.state_label(*ground), self.state_label(*excited), self.component_label(*component))
            }
            Unit::Term { a, b, beat } if beat.is_zero() => format!("{} 与 {} 的静态耦合", self.state_label(*a), self.state_label(*b)),
            Unit::Term { a, b, beat } if a == b => format!("{} 上拍频 {} 的能级拍动", self.state_label(*a), self.beat_label(beat)),
            Unit::Term { a, b, beat } => format!("{} 与 {} 在拍频 {} 上的耦合", self.state_label(*a), self.state_label(*b), self.beat_label(beat)),
            Unit::Coherence { landing, first, second } => {
                format!("落到 {} 所在子空间的两组光子（频率相差 {}）的相干", self.state_label(*landing), self.beat_label(&second.plus(-1, first)).trim_start_matches('−'))
            }
            Unit::Block { block, component, grounds } => format!(
                "激发态 {} 所在的超精细能级块（{}，从 {} 所在的子空间）：绝热消去",
                self.blocks().get(*block).and_then(|b| b.first()).map_or(String::new(), |&e| self.state_label(GROUND + e)),
                self.component_label(*component),
                self.state_label(grounds.trailing_zeros() as usize)
            ),
            Unit::Series => "绝热消去关于参考能量的级数只取零阶（不取一阶 N₁ 项）".to_string(),
            Unit::Normalization { order } => format!("P 坐标的归一化级数只取到 {} 阶（不取 X^{order} 项）", order - 1),
        }
    }

    /// A transition to a level outside the model's label for the ledger (Chinese; `Model::outside`): its light's beams
    /// and its name.
    pub(crate) fn outside_label(&self, t: usize) -> String {
        let (frequency, transition) = &self.outside[t];
        format!("{} 经 32 态以外的跃迁 {} 的二阶静态光移", self.component_label(Component { frequency: *frequency, counter: false }), transition.name)
    }

    /// The model in words (Chinese, the result's method text): its subspaces -- those of one make-up said once, with
    /// their number, in the order they first come --, their explicit excited states, and how many drives are taken at
    /// each level.
    pub fn describe(&self) -> String {
        let mut kinds: Vec<((usize, usize), usize)> = vec![];
        for c in &self.clusters {
            let kind = (c.ground, c.states.len() - c.ground);
            match kinds.iter_mut().find(|(k, _)| *k == kind) {
                Some((_, number)) => *number += 1,
                None => kinds.push((kind, 1)),
            }
        }
        let subspaces: Vec<String> = kinds
            .iter()
            .map(|&((ground, explicit), number)| {
                let holds = if explicit > 0 { format!("{} 个基态和 {} 个显式跟踪的激发态", ground, explicit) } else { format!("{} 个基态", ground) };
                format!("{} 个{}含 {}的子空间", number, if number > 1 { "各" } else { "" }, holds)
            })
            .collect();
        let count = |level: Level| self.driven.iter().filter(|&&(c, e, g)| self.level(c, e, g) == level).count();
        format!(
            "内态分为 {}；每个驱动按它自己的失谐处理：{} 个按裸能级绝热消去（给出光移和散射），{} 个按原子所在处的能级（含光移）绝热消去，{} 个显式跟踪。",
            subspaces.join("、"),
            count(Level::Bare),
            count(Level::Atom),
            count(Level::Explicit)
        )
    }
}

/// A frequency (MHz) as the app reads a frequency difference (src/laser-frequency.ts `naturalDetuningUnit`, src/format.ts
/// `formatNumber`): in the largest of Hz, kHz, MHz, GHz and THz in which it reads at least 1 (zero in MHz), rounded to
/// `digits` significant digits, the trailing zeros dropped, with the true minus sign.
fn frequency_reading(mhz: f64, digits: i32) -> String {
    let magnitude = mhz.abs();
    let (scale, unit) = if magnitude >= 1e6 {
        (1e6, "THz")
    } else if magnitude >= 1e3 {
        (1e3, "GHz")
    } else if magnitude >= 1.0 || magnitude == 0.0 {
        (1.0, "MHz")
    } else if magnitude >= 1e-3 {
        (1e-3, "kHz")
    } else {
        (1e-6, "Hz")
    };
    // The rounded decimal prints as itself, without zeros after it (ledger.rs `significant`).
    format!("{}{} {}", if mhz < 0.0 { "−" } else { "" }, crate::ledger::significant(magnitude / scale, digits), unit)
}

/// The fewest significant digits at which frequencies (MHz) all read differently (`frequency_reading`), two of one
/// double counted once: the digits follow from the frequencies themselves. Two beams 80 MHz apart on a carrier 32 THz
/// from a third take seven (32.35877 THz, 32.35885 THz); a single beat and its opposite take one. No more than the
/// decimal digits a double holds (`f64::DIGITS`): two frequencies alike in all of them read alike.
fn telling_apart(mhz: &[f64]) -> i32 {
    let mut values = mhz.to_vec();
    values.sort_by(f64::total_cmp);
    values.dedup();
    let apart = |digits: &i32| values.iter().map(|&value| frequency_reading(value, *digits)).collect::<HashSet<String>>().len() == values.len();
    (1..f64::DIGITS as i32).find(apart).unwrap_or(f64::DIGITS as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A frequency reads in the unit of its size, rounded in decimal, without trailing zeros and with the true minus
    /// (the cut-over's finding: a 32 THz beat was written "-32358766.391647 MHz").
    #[test]
    fn a_frequency_reads_in_its_own_unit() {
        assert_eq!(frequency_reading(-32358766.391647, 4), "−32.36 THz");
        assert_eq!(frequency_reading(6834.682610904, 6), "6.83468 GHz");
        assert_eq!(frequency_reading(80.0, 7), "80 MHz");
        assert_eq!(frequency_reading(-0.0025, 2), "−2.5 kHz");
        assert_eq!(frequency_reading(3.0e-6, 3), "3 Hz");
        assert_eq!(frequency_reading(0.0, 3), "0 MHz");
        // A digit count below the integer digits rounds them too.
        assert_eq!(frequency_reading(323.6, 1), "300 MHz");
    }

    /// The digits are the fewest that tell a set's frequencies apart: every reading differs at them, and two coincide
    /// at one digit fewer.
    #[test]
    fn the_digits_tell_the_frequencies_apart() {
        let carrier = 32358766.391647;
        for (set, digits) in [(vec![carrier, -carrier, 0.0], 1), (vec![carrier, carrier + 80.0, 80.0, -80.0, 0.0], 7), (vec![6834.68, 6834.68, 6568.0], 2)] {
            assert_eq!(telling_apart(&set), digits, "{set:?}");
            let readings = |d: i32| set.iter().map(|&f| frequency_reading(f, d)).collect::<HashSet<String>>().len();
            let distinct = { let mut v = set.clone(); v.sort_by(f64::total_cmp); v.dedup(); v.len() };
            assert_eq!(readings(digits), distinct, "{set:?}");
            assert!(digits == 1 || readings(digits - 1) < distinct, "{set:?}");
        }
        assert_eq!(frequency_reading(carrier + 80.0, 7), "32.35885 THz");
        assert_eq!(frequency_reading(carrier, 7), "32.35877 THz");
    }
}
