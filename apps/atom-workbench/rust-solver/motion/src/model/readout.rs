//! The readout (SPEC §1.7): the 32 populations of a cluster's state, ρ − ½{Y, ρ} on its states plus the folded excited
//! populations; and the model in words.
//!
//! # The readout
//! A component's folded drives' excited amplitude from ground state b is −R[:,b] x_b -- its bare drives' and those folded at
//! the atom, each a channel (far.rs, eliminate.rs) --, its channel's excited frequency θ_b + ν (frames). Two channels at one frequency are co-resonant; the others beat and average out. With ρ = xx†/‖x‖²,
//! the folded excited populations are Σ (R ρ R†)_ii over co-resonant pairs of channels, and the ground states keep ρ − ½{Y, ρ},
//! Y = Σ R'†R over the same pairs (the metric 1 + Y's first order): the populations sum to one exactly, Re tr(Yρ) =
//! Σ_i (RρR†)_ii. An explicit state's population is its own ρ_ee.
//!
//! Co-resonance is an equivalence of the places (channel p, ground place b) -- one excited frequency θ_b + ν_p a class
//! --, so the pairs' sums are the classes' amplitudes': with a_pb = R_p[:, b] x_b the place's excited amplitude and
//! A_k = Σ_{(p, b) ∈ k} a_pb the class's, Σ over the ordered pairs of a class of a_pb1* a_qb2 is |A_k|², and of the pairs
//! of one place a_pb1* A_k. The excited populations are Σ_k |A_k[i]|²/‖x‖² and a ground place gives up
//! Σ_p Re⟨a_pb, A_k(p, b)⟩/‖x‖²: every place's amplitude is made once and enters two sums, whatever the pairs' number
//! (their number is the classes' sizes squared: 264 pairs of 48 places on GMD1C's 8 + 5, 281 of 40 on PGC3D's F = 2).

use super::assemble::{Held, Plan, Site};
use super::far::EXCITED;
use super::tabulate::NearTable;
use super::{Beat, Cluster, Component, Level, Lights, Model, Point, Unit, GROUND, STATES};
use crate::field::Amplitude;
use crate::linalg::{Mat, C64};
use std::collections::HashSet;

/// A component's bare drives' channels on a cluster at a point: the component and its response R (EXCITED × the cluster's
/// ground places).
pub(crate) struct Channels {
    pub(crate) component: Component,
    pub(crate) response: Mat,
}

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

    /// The bare drives' responses on a cluster at a point, per component that has one (`bare_response`), every light in
    /// `lights`.
    pub(crate) fn far_channels(&self, cluster: usize, amplitudes: &[Amplitude], lights: &Lights) -> Vec<Channels> {
        let c = &self.clusters[cluster];
        self.bare_components(lights)
            .map(|(frequency, side)| {
                let mut response = Mat::zeros(EXCITED, c.ground);
                self.bare_response(c, amplitudes, frequency, side, &mut response.data);
                Channels { component: Component { frequency, counter: side == 1 }, response }
            })
            .collect()
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
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let mut out = [0.0; STATES];
        for (z, &s) in psi.iter().zip(&c.states) {
            out[s] = z.norm_sqr() / norm;
        }
        let mut site = Held::<Site>::take();
        self.light.amplitudes_into(at.r, 0, |_| true, &mut site.beams);
        // The responses of the drives folded at the atom: their table's at the point, or the exact elimination's.
        let folded = self.folded_carriers(plan).count();
        let exact = match table {
            Some(table) if folded > 0 => {
                self.near_responses(table, &mut site);
                None
            }
            _ => (folded > 0).then(|| self.near_exact(plan, &site.beams, 0, &self.coherent_dressing())),
        };
        let Site { beams: amplitudes, responses, weights, channels, resonant: first, excited, .. } = &mut *site;
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
        // The co-resonant channels and places (p, b1, q, b2), θ_b1 + ν_p = θ_b2 + ν_q: the cluster's own under the running
        // lights, exact, found once.
        let resonant = self.resonances[plan.cluster].get_or_init(|| {
            let bare = self.bare_components(&plan.lights).map(|(frequency, side)| self.frequency(Component { frequency, counter: side == 1 }));
            let sums: Vec<Vec<Beat>> = bare.chain(self.folded_carriers(plan).map(|d| self.carriers[d].beat.clone())).map(|nu| c.frames[..c.ground].iter().map(|theta| theta.plus(1, &nu)).collect()).collect();
            let mut pairs = vec![];
            for (p, left) in sums.iter().enumerate() {
                for (q, right) in sums.iter().enumerate() {
                    for (b1, one) in left.iter().enumerate() {
                        pairs.extend(right.iter().enumerate().filter(|(_, two)| one == *two).map(|(b2, _)| [p, b1, q, b2]));
                    }
                }
            }
            pairs
        });
        // A place's class by the first of the places co-resonant with it (itself among them: every place is paired
        // with itself), and the classes' amplitudes A_k = Σ R_p[:, b] x_b, held at the class's first place.
        let places = (far + folded) * c.ground;
        first.clear();
        first.resize(places, usize::MAX);
        for &[p, b1, q, b2] in resonant {
            let at = &mut first[p * c.ground + b1];
            *at = (*at).min(q * c.ground + b2);
        }
        // Place by place the amplitudes a_pb, and after them the classes' sums.
        excited.clear();
        excited.resize(2 * places * EXCITED, C64::ZERO);
        let (own, sums) = excited.split_at_mut(places * EXCITED);
        for (p, response) in channels.chunks_exact(size).enumerate() {
            for (i, row) in response.chunks_exact(c.ground).enumerate() {
                for (b, r) in row.iter().enumerate() {
                    own[(p * c.ground + b) * EXCITED + i] = *r * psi[b];
                }
            }
        }
        for (&class, a) in first.iter().zip(own.chunks_exact(EXCITED)) {
            for (sum, a) in sums[class * EXCITED..(class + 1) * EXCITED].iter_mut().zip(a) {
                *sum += *a;
            }
        }
        // The excited populations |A_k|², a class once (at its first place); and what each ground place gives up,
        // Re⟨a_pb, A_k⟩.
        for ((place, &class), a) in first.iter().enumerate().zip(own.chunks_exact(EXCITED)) {
            let of = &sums[class * EXCITED..(class + 1) * EXCITED];
            if class == place {
                for (i, sum) in of.iter().enumerate() {
                    out[GROUND + i] += sum.norm_sqr() / norm;
                }
            }
            let given: f64 = a.iter().zip(of).map(|(a, sum)| (a.conj() * *sum).re).sum();
            out[c.states[place % c.ground]] -= given / norm;
        }
        out
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
    /// (`telling_apart`): `Model::beats` holds every beat a unit can carry (units.rs `Model::units`). So no two of a
    /// run's items share a name, and no digit is written that tells nothing apart.
    fn beat_label(&self, beat: &Beat) -> String {
        let mhz = |beat: &Beat| self.omega(beat) / crate::units::RAD_PER_US_PER_MHZ;
        let beats: Vec<f64> = self.beats.iter().map(|(beat, _)| mhz(beat)).collect();
        frequency_reading(mhz(beat), telling_apart(&beats))
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
            Unit::Second { ground } => format!("{} 所在子空间按原子所在处能级的绝热消去只取到二阶", self.state_label(*ground)),
            Unit::Term { a, b, beat } if beat.is_zero() => format!("{} 与 {} 的静态耦合", self.state_label(*a), self.state_label(*b)),
            Unit::Term { a, b, beat } if a == b => format!("{} 上拍频 {} 的能级拍动", self.state_label(*a), self.beat_label(beat)),
            Unit::Term { a, b, beat } => format!("{} 与 {} 在拍频 {} 上的耦合", self.state_label(*a), self.state_label(*b), self.beat_label(beat)),
            Unit::Coherence { landing, first, second } => {
                format!("落到 {} 所在子空间的两组光子（频率相差 {}）的相干", self.state_label(*landing), self.beat_label(&second.plus(-1, first)).trim_start_matches('−'))
            }
            Unit::Dressing { carrier } => format!("{} 引起的能级移动（进入绝热消去的失谐）：偏振不同的光束之间的干涉取平均", self.component_label(*carrier)),
        }
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
