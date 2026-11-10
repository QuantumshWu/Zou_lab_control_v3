//! The readout (SPEC §1.7): the 32 populations of a cluster's state, ρ − ½{Y, ρ} on its states plus the folded excited
//! populations; and the model in words.
//!
//! # The readout
//! A folded component's excited amplitude from ground state b is −R[:,b] x_b (far.rs, eliminate.rs), its channel's excited
//! frequency θ_b + ν (frames). Two channels at one frequency are co-resonant; the others beat and average out. With ρ = xx†/‖x‖²,
//! the folded excited populations are Σ (R ρ R†)_ii over co-resonant pairs of channels, and the ground states keep ρ − ½{Y, ρ},
//! Y = Σ R'†R over the same pairs (the metric 1 + Y's first order): the populations sum to one exactly, Re tr(Yρ) =
//! Σ_i (RρR†)_ii. An explicit state's population is its own ρ_ee.

use super::{Beat, Component, Lights, Model, Point, Unit, GROUND, STATES};
use crate::linalg::{Mat, C64};
use crate::request::Level;

/// A folded component's channels on a cluster at a point: its exact frequency and its response R (EXCITED × the cluster's ground
/// places).
pub(crate) struct Channels {
    pub(crate) nu: Beat,
    pub(crate) response: Mat,
}

impl Model {
    /// The far light's folded responses on a cluster at a point (far.rs `Tone::responses`): R = Σ_l a_l R_l (co-rotating) or
    /// Σ_l a_l* R_l (counter-rotating), every light in `lights`.
    pub(crate) fn far_channels(&self, cluster: usize, amplitudes: &[crate::field::Amplitude], lights: &Lights) -> Vec<Channels> {
        let c = &self.clusters[cluster];
        let mut out = vec![];
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            if !lights.on(w) {
                continue;
            }
            for side in 0..2 {
                if !tone.folded[side] {
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
                out.push(Channels { nu: self.axes.component(&self.light, Component { frequency: w, counter: side == 1 }), response });
            }
        }
        out
    }

    /// The readout (module documentation): the 32 populations of a cluster's state ψ at a point.
    pub fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.readout_with(cluster, at, psi, None)
    }

    /// The readout with the near light from its tables where they serve (tabulate.rs `Model::near`).
    pub(crate) fn readout_with(&self, cluster: usize, at: &Point, psi: &[C64], tables: Option<&super::tabulate::Tables>) -> [f64; STATES] {
        let c = &self.clusters[cluster];
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let rho = |x: usize, y: usize| psi[x] * psi[y].conj() / norm;
        let mut out = [0.0; STATES];
        for (x, &s) in c.states.iter().enumerate() {
            out[s] = rho(x, x).re;
        }
        let amplitudes = self.light.amplitudes(at.r);
        let lights = self.running();
        let mut channels = self.far_channels(cluster, &amplitudes, &lights);
        channels.extend(self.near_channels(cluster, at, &amplitudes, &lights, tables));
        let g = c.ground;
        for p in &channels {
            for q in &channels {
                for b1 in 0..g {
                    for b2 in 0..g {
                        // Co-resonant: θ_b1 + ν_p = θ_b2 + ν_q.
                        if c.frames[b1].plus(1, &p.nu) != c.frames[b2].plus(1, &q.nu) {
                            continue;
                        }
                        let r = rho(b2, b1);
                        let mut y = C64::ZERO;
                        for i in 0..super::far::EXCITED {
                            let z = p.response[(i, b1)].conj() * q.response[(i, b2)];
                            y += z;
                            out[GROUND + i] += (z * r).re;
                        }
                        out[c.states[b1]] -= (y * r).re;
                    }
                }
            }
        }
        out
    }

    /// V_L(r; p) far from the trap (SPEC §1.7): every focused beam's amplitude zero, the plane waves' terms of one wave vector
    /// (their spatial mean; a term of q ≠ 0 averages out), and the near light's (eliminate.rs).
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
        v + self.near_asymptote(p, lights)
    }

    /// A state's label in the model basis (`manifold-F-m`), as the app writes it: "D2 F′=3 m=2", "基态 F=2 m=−1".
    pub(crate) fn state_label(&self, s: usize) -> String {
        match Level::parse(&self.labels[s]) {
            Ok(level) => {
                let m = if level.m < 0 { format!("−{}", -level.m) } else { level.m.to_string() };
                match level.manifold {
                    crate::request::Manifold::Ground => format!("基态 F={} m={}", level.f, m),
                    crate::request::Manifold::D1 => format!("D1 F′={} m={}", level.f, m),
                    crate::request::Manifold::D2 => format!("D2 F′={} m={}", level.f, m),
                }
            }
            Err(_) => self.labels[s].clone(),
        }
    }

    /// A component's name: its beams' names, and which rotation.
    fn component_label(&self, c: Component) -> String {
        let names: Vec<&str> = self.light.frequencies[c.frequency].beams.iter().map(|&l| self.names[l].as_str()).collect();
        format!("{}{}", names.join("、"), if c.counter { "（反旋）" } else { "" })
    }

    /// A unit's label for the ledger (Chinese).
    pub fn label(&self, unit: &Unit) -> String {
        match unit {
            Unit::Drive { carrier, excited, ground } => {
                format!("{} → {} 的驱动（{}）", self.state_label(*ground), self.state_label(*excited), self.component_label(*carrier))
            }
            Unit::Term { a, b, from: None, to: None } => format!("{} 与 {} 的远光静态耦合", self.state_label(*a), self.state_label(*b)),
            Unit::Term { a, b, from, to } => {
                let name = |c: &Option<Component>| c.map_or(String::new(), |c| self.component_label(c));
                if a == b {
                    format!("{} 上两组光的拍频（{} 与 {}）", self.state_label(*a), name(from), name(to))
                } else {
                    format!("{} 与 {} 的耦合（吸收 {}，发射 {}）", self.state_label(*a), self.state_label(*b), name(from), name(to))
                }
            }
            Unit::Far { component } => format!("远失谐光的折叠（{}）", self.component_label(*component)),
            Unit::FarBeat { first, second } => format!("两组远失谐光的拍频（{} 与 {}）", self.component_label(*first), self.component_label(*second)),
            Unit::Excited { excited, other } => format!("{} 与 {} 的静态耦合", self.state_label(*excited), self.state_label(*other)),
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
        let near = self.carriers.len();
        let far = self.components().into_iter().filter(|c| self.folded(*c)).count();
        format!(
            "内态分为 {} 个子空间（{}）；{} 个远失谐分量以二阶 van Vleck 移和散射折叠，{} 个近共振载波以缀饰的二阶消去折叠，其余耦合显式保留。",
            self.clusters.len(),
            clusters.join("；"),
            far,
            near
        )
    }
}
