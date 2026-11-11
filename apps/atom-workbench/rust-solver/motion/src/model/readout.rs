//! The readout (SPEC §1.7): the 32 populations of a cluster's state, ρ − ½{Y, ρ} on its states plus the folded excited
//! populations; and the model in words.
//!
//! # The readout
//! A component's folded drives' excited amplitude from ground state b is −R[:,b] x_b -- its bare drives' and those folded at
//! the atom, each a channel (far.rs, eliminate.rs) --, its channel's excited frequency θ_b + ν (frames). Two channels at one frequency are co-resonant; the others beat and average out. With ρ = xx†/‖x‖²,
//! the folded excited populations are Σ (R ρ R†)_ii over co-resonant pairs of channels, and the ground states keep ρ − ½{Y, ρ},
//! Y = Σ R'†R over the same pairs (the metric 1 + Y's first order): the populations sum to one exactly, Re tr(Yρ) =
//! Σ_i (RρR†)_ii. An explicit state's population is its own ρ_ee.

use super::{Beat, Component, Level, Lights, Model, Point, Unit, GROUND, STATES};
use crate::linalg::{Mat, C64};

/// A component's folded channels on a cluster at a point: the component, its exact frequency and its response R (EXCITED × the
/// cluster's ground places) -- its bare drives', or those folded at the atom.
pub(crate) struct Channels {
    pub(crate) component: Component,
    pub(crate) nu: Beat,
    pub(crate) response: Mat,
}

impl Model {
    /// The bare drives' responses on a cluster at a point, per component that has one (far.rs `Tone::responses`): R = Σ_l a_l R_l
    /// (co-rotating) or Σ_l a_l* R_l (counter-rotating), every light in `lights`.
    pub(crate) fn far_channels(&self, cluster: usize, amplitudes: &[crate::field::Amplitude], lights: &Lights) -> Vec<Channels> {
        let c = &self.clusters[cluster];
        let mut out = vec![];
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            if !lights.on(w) {
                continue;
            }
            for side in 0..2 {
                if !tone.bare[side] {
                    continue;
                }
                let mut response = Mat::zeros(super::far::EXCITED, c.ground);
                for (j, &l) in f.beams.iter().enumerate() {
                    let a = if side == 0 { amplitudes[l].value } else { amplitudes[l].value.conj() };
                    for x in 0..super::far::EXCITED {
                        for (y, &b) in c.states[..c.ground].iter().enumerate() {
                            response[(x, y)] += a * tone.responses[side][j][(x, b)];
                        }
                    }
                }
                let component = Component { frequency: w, counter: side == 1 };
                out.push(Channels { component, nu: self.frequency(component), response });
            }
        }
        out
    }

    /// The readout (module documentation): the 32 populations of a cluster's state ψ at a point.
    pub fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.readout_with(cluster, at, psi, None)
    }

    /// The readout with the forms at the atom from their tables where they serve (tabulate.rs `Model::near_tabulated`).
    pub(crate) fn readout_with(&self, cluster: usize, at: &Point, psi: &[C64], tables: Option<&super::tabulate::Tables>) -> [f64; STATES] {
        let c = &self.clusters[cluster];
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let mut out = [0.0; STATES];
        for (z, &s) in psi.iter().zip(&c.states) {
            out[s] = z.norm_sqr() / norm;
        }
        let amplitudes = self.light.amplitudes(at.r);
        let lights = self.running();
        let mut channels = self.far_channels(cluster, &amplitudes, &lights);
        channels.extend(self.near_channels(cluster, at, &amplitudes, &lights, tables));
        // The co-resonant channels and places (p, b1, q, b2), θ_b1 + ν_p = θ_b2 + ν_q: the cluster's own under the running
        // lights, exact, found once.
        let resonant = self.resonances[cluster].get_or_init(|| {
            let sums: Vec<Vec<Beat>> = channels.iter().map(|p| c.frames[..c.ground].iter().map(|theta| theta.plus(1, &p.nu)).collect()).collect();
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
        for &[p, b1, q, b2] in resonant {
            let r = psi[b2] * psi[b1].conj() / norm;
            let (left, right) = (&channels[p].response, &channels[q].response);
            let mut y = C64::ZERO;
            for i in 0..super::far::EXCITED {
                let z = left[(i, b1)].conj() * right[(i, b2)];
                y += z;
                out[GROUND + i] += (z * r).re;
            }
            out[c.states[b1]] -= (y * r).re;
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
