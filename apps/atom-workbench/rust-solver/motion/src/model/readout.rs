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
            for x in 0..EXCITED {
                for (y, &b) in c.states[..c.ground].iter().enumerate() {
                    response[x * c.ground + y] += a * tone.responses[side][j][(x, b)];
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
        excited.clear();
        excited.resize(places * EXCITED, C64::ZERO);
        let amplitude = |place: usize, i: usize| channels[(place / c.ground) * size + i * c.ground + place % c.ground] * psi[place % c.ground];
        for (place, &class) in first.iter().enumerate() {
            for (i, sum) in excited[class * EXCITED..(class + 1) * EXCITED].iter_mut().enumerate() {
                *sum += amplitude(place, i);
            }
        }
        // The excited populations |A_k|², a class once (at its first place); and what each ground place gives up,
        // Re⟨a_pb, A_k⟩.
        for (place, &class) in first.iter().enumerate() {
            let of = &excited[class * EXCITED..(class + 1) * EXCITED];
            if class == place {
                for (i, sum) in of.iter().enumerate() {
                    out[GROUND + i] += sum.norm_sqr() / norm;
                }
            }
            let given: f64 = of.iter().enumerate().map(|(i, sum)| (amplitude(place, i).conj() * *sum).re).sum();
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
        format!("{}{}", names.join("、"), if c.counter { "（反旋）" } else { "" })
    }

    /// A beat's value as the ledger writes it (MHz).
    fn beat_label(&self, beat: &Beat) -> String {
        format!("{:.6} MHz", self.omega(beat) / crate::units::RAD_PER_US_PER_MHZ)
    }

    /// A unit's label for the ledger (Chinese).
    pub fn label(&self, unit: &Unit) -> String {
        match unit {
            Unit::Bare { component, excited, ground } => {
                format!("{} → {} 的驱动按裸能级折叠（{}）", self.state_label(*ground), self.state_label(*excited), self.component_label(*component))
            }
            Unit::Drive { component, excited, ground } => {
                format!("{} → {} 的驱动按原子处能级折叠（{}）", self.state_label(*ground), self.state_label(*excited), self.component_label(*component))
            }
            Unit::Second { ground } => format!("{} 所在簇按原子处能级的折叠只取到二阶", self.state_label(*ground)),
            Unit::Term { a, b, beat } if beat.is_zero() => format!("{} 与 {} 的静态耦合", self.state_label(*a), self.state_label(*b)),
            Unit::Term { a, b, beat } if a == b => format!("{} 上拍频 {} 的能级拍动", self.state_label(*a), self.beat_label(beat)),
            Unit::Term { a, b, beat } => format!("{} 与 {} 在拍频 {} 上的耦合", self.state_label(*a), self.state_label(*b), self.beat_label(beat)),
            Unit::Coherence { landing, .. } => format!("落到 {} 所在子空间的两组光子的相干", self.state_label(*landing)),
            Unit::Dressing { carrier } => format!("{} 的缀饰：偏振单元之间的干涉", self.component_label(*carrier)),
        }
    }

    /// The model in words (Chinese, the result's method text): its subspaces, explicit excited states and what is folded.
    pub fn describe(&self) -> String {
        let clusters: Vec<String> = self
            .clusters
            .iter()
            .map(|c| {
                let explicit = c.states.len() - c.ground;
                if explicit > 0 {
                    format!("{} 个基态 + {} 个显式激发态", c.ground, explicit)
                } else {
                    format!("{} 个基态", c.ground)
                }
            })
            .collect();
        let count = |level: Level| self.driven.iter().filter(|&&(c, e, g)| self.level(c, e, g) == level).count();
        format!(
            "内态分为 {} 个子空间（{}）；每个驱动按它自己的失谐处理：{} 个按裸能级的二阶消去（光移和散射）折叠，{} 个按原子处能级的缀饰二阶消去折叠，{} 个显式保留。",
            self.clusters.len(),
            clusters.join("；"),
            count(Level::Bare),
            count(Level::Atom),
            count(Level::Explicit)
        )
    }
}
