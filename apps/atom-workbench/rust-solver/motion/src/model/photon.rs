//! Photons (SPEC §1.6): the jump operators by class, the photons a state emits, the recoil's two moments.
//!
//! # Sources and classes
//! A source is one part of the jump operators, linear in one beam's field (or constant): a folded component's photons C_a R from
//! one beam's absorption (far.rs, eliminate.rs) or an explicit excited state's C_a. In the base frame its entries b' ← s carry
//! the beat Ω_J -- −ν of the absorbed component, 0 for an explicit state (the excited states' ν_frame, common to every photon,
//! is a global phase) -- and the wave vector q_J (the absorbed beam's k_l; −k_l for a counter-rotating component's a_l*). In the
//! frames (cluster.rs) an entry is at ω̃ = Ω_J + θ_b' − θ_s, its photon's frequency up to the class's common phase; a folded
//! photon and an explicit one of the same light land at the same ω̃ exactly. A class is the entries of one frame frequency
//! landing in one cluster (model.rs `Photon`); two frequencies landing in one cluster are two classes unless their coherence is
//! kept (`Unit::Coherence`), then one, each member at its relative phase e^{i(ω̃ − ω̃₀)t}.
//!
//! # The decay (assemble.rs)
//! −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}: per class and pair of its members (j, i) the term c_j* c_i L_j†L_i, L the members' entries in
//! that class, at the beat Ω_i − Ω_j and the wave vector q_i − q_j, whose residual in the frames is zero within one frequency:
//! the decay is the jumps' own product, the generator Lindblad's by construction, and the hazard ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖² is
//! Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² identically (`hazard`).
//!
//! # The recoil (SPEC §1.6, one rule)
//! The jump into f = Σ_a ε_a u_a has the absorption's conditional mean b = Im⟨f|∇f⟩/‖f‖² (the jump term's first Wigner moment)
//! and its local second moment M = Re⟨∇f|∇f⟩/‖f‖² (the second Kramers–Moyal coefficient of the method's own expansion). A folded
//! source's absorption mean is the jump's to carry: its ∇f is its coefficient's gradient on its entries (the beam's ∇a_l = a_l
//! (ik_l + ∇ln|a_l|)). An explicit state's mean is the Ehrenfest force's (its kept drives' stimulated force), so its part of the
//! kick has mean zero and the second moment of its drives' responses σ_d, Re[∇ln σ_d*⊗∇ln σ_d] by their shares (eliminate.rs).
//! The kick is b_J + ξ, Cov ξ = M − b_J b_Jᵀ, b_J = the folded mean: positive semidefinite by Cauchy–Schwarz.

use super::{Beat, Gate, Lights, Model, Photon, Point, Reach, Recoil, Sample, Unit, UnitError, Wave};
use crate::field::Amplitude;
use crate::linalg::{Mat, C64};
use crate::request::{GROUND, STATES};

/// One source of the jump operators (module documentation): its light (the frequency whose field it needs, None for an explicit
/// state's), its beam (its coefficient a_l, or a_l* where `conjugate`; None: 1), its exact base beat and wave vector, its jumps
/// per axis (GROUND × STATES, landing × source, in the model basis), and whether the jump carries its absorption's mean.
#[derive(Clone, Debug)]
pub(crate) struct Source {
    pub(crate) frequency: Option<usize>,
    pub(crate) beam: Option<usize>,
    pub(crate) conjugate: bool,
    pub(crate) beat: Beat,
    pub(crate) wave: Wave,
    pub(crate) jumps: [Mat; 3],
    pub(crate) folded: bool,
}

/// A member of a class: a source and its entries in the class (landing place in the landing cluster, source place in the
/// source cluster), and its relative beat in a merged class (zero otherwise).
#[derive(Clone, Debug)]
pub(crate) struct Member {
    pub(crate) source: usize,
    pub(crate) entries: Vec<(usize, usize)>,
    pub(crate) phase: Beat,
}

/// A class of a source cluster's photons: its frame frequencies (one, or several kept coherent), its landing cluster, its members.
#[derive(Clone, Debug)]
pub(crate) struct Class {
    pub(crate) beats: Vec<Beat>,
    pub(crate) landing: usize,
    pub(crate) members: Vec<Member>,
}

impl Model {
    /// The far light's sources: per folded component and beam its jumps (far.rs).
    pub(crate) fn far_sources(&self) -> Vec<Source> {
        let mut out = vec![];
        for (frequency, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            for side in 0..2 {
                if !tone.folded[side] {
                    continue;
                }
                let nu = self.axes.component(&self.light, super::Component { frequency, counter: side == 1 });
                for (j, &l) in f.beams.iter().enumerate() {
                    let jumps = std::array::from_fn(|a| {
                        let mut m = Mat::zeros(GROUND, STATES);
                        for x in 0..GROUND {
                            for y in 0..GROUND {
                                m[(x, y)] = tone.jumps[side][j][a][(x, y)];
                            }
                        }
                        m
                    });
                    let wave = if side == 0 { self.beam_wave(l) } else { self.beam_wave(l).negated() };
                    out.push(Source { frequency: Some(frequency), beam: Some(l), conjugate: side == 1, beat: nu.negated(), wave, jumps, folded: true });
                }
            }
        }
        out
    }

    /// The explicit excited states' sources: C_a on each one's column, constant.
    pub(crate) fn explicit_sources(&self) -> Vec<Source> {
        (GROUND..STATES)
            .filter(|&e| self.explicit(e))
            .map(|e| Source {
                frequency: None,
                beam: None,
                conjugate: false,
                beat: self.axes.zero(),
                wave: self.no_wave(),
                jumps: std::array::from_fn(|a| Mat::from_fn(GROUND, STATES, |x, y| if y == e { self.emission[a][(x, e)] } else { C64::ZERO })),
                folded: false,
            })
            .collect()
    }

    /// The classes of a cluster's photons (module documentation, # Sources and classes): its sources' entries from its states,
    /// grouped by frame frequency and landing cluster, the kept coherences merged.
    pub(crate) fn classes(&self, cluster: usize) -> Vec<Class> {
        let c = &self.clusters[cluster];
        let mut out: Vec<Class> = vec![];
        for (index, source) in self.sources.iter().enumerate() {
            for (sp, &s) in c.states.iter().enumerate() {
                for b in 0..GROUND {
                    if (0..3).all(|a| source.jumps[a][(b, s)] == C64::ZERO) {
                        continue;
                    }
                    let landing = self.membership[b];
                    let lc = &self.clusters[landing];
                    let bp = lc.states.iter().position(|&x| x == b).expect("a ground state is in its cluster");
                    let beat = source.beat.plus(1, &lc.frames[bp]).plus(-1, &c.frames[sp]);
                    let class = match out.iter().position(|k| k.landing == landing && k.beats[0] == beat) {
                        Some(at) => at,
                        None => {
                            out.push(Class { beats: vec![beat.clone()], landing, members: vec![] });
                            out.len() - 1
                        }
                    };
                    let members = &mut out[class].members;
                    match members.iter_mut().find(|m| m.source == index) {
                        Some(m) => m.entries.push((bp, sp)),
                        None => members.push(Member { source: index, entries: vec![(bp, sp)], phase: self.axes.zero() }),
                    }
                }
            }
        }
        // The kept coherences: merge the classes of one landing cluster whose frequencies a kept unit joins.
        let mut merged: Vec<Class> = vec![];
        for class in out {
            let key = self.clusters[class.landing].states[0];
            let joined = merged.iter().position(|m| {
                m.landing == class.landing
                    && m.beats.iter().any(|b| {
                        let (lo, hi) = if *b < class.beats[0] { (b.clone(), class.beats[0].clone()) } else { (class.beats[0].clone(), b.clone()) };
                        self.kept.contains(&Unit::Coherence { landing: key, first: lo, second: hi })
                    })
            });
            match joined {
                Some(at) => {
                    let reference = merged[at].beats[0].clone();
                    let phase = class.beats[0].plus(-1, &reference);
                    merged[at].beats.push(class.beats[0].clone());
                    merged[at].members.extend(class.members.into_iter().map(|mut m| {
                        m.phase = phase.clone();
                        m
                    }));
                }
                None => merged.push(class),
            }
        }
        merged
    }

    /// A source's coefficient at a point: a_l(r), a_l(r)* or 1, with its gradient and Hessian.
    pub(crate) fn coefficient(&self, source: &Source, amplitudes: &[Amplitude]) -> Amplitude {
        match source.beam {
            None => Amplitude { value: C64::ONE, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] },
            Some(l) if source.conjugate => {
                let a = &amplitudes[l];
                Amplitude { value: a.value.conj(), gradient: a.gradient.map(|z| z.conj()), hessian: a.hessian.map(|r| r.map(|z| z.conj())) }
            }
            Some(l) => amplitudes[l],
        }
    }

    /// Whether a source's light is on.
    pub(crate) fn lit(&self, source: &Source, lights: &Lights) -> bool {
        source.frequency.map_or(true, |f| lights.0[f])
    }

    /// The photon classes of a cluster's state ψ (its frame amplitudes) at a point, every light on (SPEC §1.6; module
    /// documentation): per class u_a on the landing cluster's states, the folded part's gradients, the explicit sources.
    pub fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        let amplitudes = self.light.amplitudes(at.r);
        let lights = self.running();
        let mut out = vec![];
        for class in &self.classes[cluster] {
            let n = self.clusters[class.landing].states.len();
            let mut u: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n]);
            let mut gradients: [[Vec<C64>; 3]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| vec![C64::ZERO; n]));
            let mut explicit = vec![];
            for m in &class.members {
                let source = &self.sources[m.source];
                if !self.lit(source, &lights) {
                    continue;
                }
                let c = self.coefficient(source, &amplitudes);
                let phase = C64::cis(self.omega(&m.phase) * at.t);
                let mut own: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n]);
                for a in 0..3 {
                    for &(bp, sp) in &m.entries {
                        let s = self.clusters[cluster].states[sp];
                        let b = self.clusters[class.landing].states[bp];
                        let x = source.jumps[a][(b, s)] * psi[sp] * phase;
                        own[a][bp] += x;
                        u[a][bp] += c.value * x;
                        if source.folded {
                            for i in 0..3 {
                                gradients[i][a][bp] += c.gradient[i] * x;
                            }
                        }
                    }
                }
                if !source.folded {
                    explicit.extend(self.explicit_recoil(cluster, at, m.source, own));
                }
            }
            if u.iter().all(|v| v.iter().all(|z| *z == C64::ZERO)) {
                continue;
            }
            out.push(Photon {
                landing: class.landing,
                wave_number: self.class_wave_number(class),
                amplitudes: u,
                gradients,
                explicit,
            });
        }
        out
    }

    /// A class's photon wave number: its light's k = 2π ν/c, the absorbed light's (the photon's frequency differs from it by the
    /// atom's energies, ≤ 1e-7 of it on the D lines) -- the largest beam wave number among its members' lights, the explicit
    /// states' through the frequencies that drive them.
    fn class_wave_number(&self, class: &Class) -> f64 {
        let mut k = 0.0_f64;
        for m in &class.members {
            let source = &self.sources[m.source];
            let frequency = source.frequency.or_else(|| self.driving_frequency(&self.sources[m.source]));
            if let Some(f) = frequency {
                for &l in &self.light.frequencies[f].beams {
                    k = k.max(self.light.beams[l].wave_number);
                }
            }
        }
        k
    }

    /// The candidate photon coherences of this model (`Unit::Coherence`): per source cluster, each pair of its classes landing in
    /// one cluster at two frame frequencies.
    pub(crate) fn coherence_units(&self) -> Vec<Unit> {
        let mut out: Vec<Unit> = vec![];
        for cluster in 0..self.clusters.len() {
            let classes = &self.classes[cluster];
            for (i, a) in classes.iter().enumerate() {
                for b in &classes[i + 1..] {
                    if a.landing != b.landing {
                        continue;
                    }
                    for x in &a.beats {
                        for y in &b.beats {
                            let (lo, hi) = if x < y { (x.clone(), y.clone()) } else { (y.clone(), x.clone()) };
                            let unit = Unit::Coherence { landing: self.clusters[a.landing].states[0], first: lo, second: hi };
                            if !out.contains(&unit) {
                                out.push(unit);
                            }
                        }
                    }
                }
            }
        }
        out
    }

    /// The two classes a coherence unit names, per source cluster: (cluster, class of `first`, class of `second`).
    fn coherence_classes(&self, unit: &Unit) -> Vec<(usize, usize, usize)> {
        let Unit::Coherence { landing, first, second } = unit else { return vec![] };
        let mut out = vec![];
        for cluster in 0..self.clusters.len() {
            let classes = &self.classes[cluster];
            let find = |b: &Beat| classes.iter().position(|k| self.clusters[k.landing].states[0] == *landing && k.beats.contains(b));
            if let (Some(i), Some(j)) = (find(first), find(second)) {
                out.push((cluster, i, j));
            }
        }
        out
    }

    /// A class's jump operators' sizes per axis over a cell's amplitude bounds (or at a point): Σ_members |c| ‖L_a‖_F.
    fn class_size(&self, cluster: usize, class: &Class, most: &dyn Fn(&Source) -> f64) -> [f64; 3] {
        let mut out = [0.0; 3];
        for m in &class.members {
            let source = &self.sources[m.source];
            let c = most(source);
            for a in 0..3 {
                let norm: f64 = m
                    .entries
                    .iter()
                    .map(|&(bp, sp)| source.jumps[a][(self.clusters[class.landing].states[bp], self.clusters[cluster].states[sp])].norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                out[a] += c * norm;
            }
        }
        out
    }

    /// A coherence's mixing with the sources' coefficients bounded by `most`: m = Σ_a |u_a||u'_a|/(|Ω| − spread), Ω the two
    /// frequencies' beat (SPEC §1.5: its error m²√(p_i p_j)), the largest over the source clusters it names.
    fn coherence_mixing(&self, unit: &Unit, most: &dyn Fn(&Source) -> f64) -> f64 {
        let Unit::Coherence { first, second, .. } = unit else { return 0.0 };
        let beat = self.omega(&first.plus(-1, second)).abs();
        let mut m = 0.0_f64;
        for (cluster, i, j) in self.coherence_classes(unit) {
            let classes = &self.classes[cluster];
            let (a, b) = (self.class_size(cluster, &classes[i], most), self.class_size(cluster, &classes[j], most));
            let size: f64 = (0..3).map(|x| a[x] * b[x]).sum();
            let d = beat - self.spread();
            m = m.max(if size == 0.0 { 0.0 } else if d > 0.0 { size / d } else { f64::INFINITY });
        }
        m
    }

    /// A coherence's gate: its mixing with every beam at its peak (|a_l| ≤ 1 everywhere: a bound over any reach).
    pub(crate) fn coherence_gate(&self, unit: &Unit, _reach: &Reach) -> Gate {
        let m = self.coherence_mixing(unit, &|_| 1.0);
        Gate { saturation: 2.0 * m * m, sweep: 0.0 }
    }

    /// A coherence's error at a sample: m² √(p_i p_j), p the populations of the two classes' source states.
    pub(crate) fn coherence_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        let amplitudes = self.light.amplitudes(sample.at.r);
        let m = self.coherence_mixing(unit, &|s| self.coefficient(s, &amplitudes).value.abs());
        let p = self.state_populations(sample);
        let source_weight = |class: &Class| -> f64 {
            class.members.iter().flat_map(|m| m.entries.iter().map(|&(_, sp)| self.clusters[sample.cluster].states[sp])).map(|s| p[s]).fold(0.0, f64::max)
        };
        let mut weight = 0.0_f64;
        for (cluster, i, j) in self.coherence_classes(unit) {
            if cluster == sample.cluster {
                let classes = &self.classes[cluster];
                weight = weight.max((source_weight(&classes[i]) * source_weight(&classes[j])).sqrt());
            }
        }
        UnitError { parameter: m * m, population: m * m * weight, temperature: m * m * weight }
    }
}

impl Photon {
    /// The absorption recoil's moments of the jump into f = Σ_a ε_a u_a (module documentation, # The recoil): the folded part's
    /// mean Im⟨f|∇f_F⟩/‖f‖², and the total second moment Re⟨∇f_F|∇f_F⟩/‖f‖² plus the explicit sources' Re[g*⊗g]‖f_s‖²/‖f‖².
    pub fn recoil(&self, polarization: &[f64; 3]) -> Recoil {
        let n = self.amplitudes[0].len();
        let along = |v: &[Vec<C64>; 3]| -> Vec<C64> { (0..n).map(|x| (0..3).map(|a| v[a][x] * polarization[a]).sum()).collect() };
        let f = along(&self.amplitudes);
        let norm: f64 = f.iter().map(|z| z.norm_sqr()).sum();
        let df: [Vec<C64>; 3] = std::array::from_fn(|i| along(&self.gradients[i]));
        let mean = std::array::from_fn(|i| crate::linalg::inner(&f, &df[i]).im / norm);
        let mut second: [[f64; 3]; 3] = std::array::from_fn(|i| std::array::from_fn(|j| crate::linalg::inner(&df[i], &df[j]).re / norm));
        for (u, g) in &self.explicit {
            let w: f64 = along(u).iter().map(|z| z.norm_sqr()).sum::<f64>() / norm;
            for i in 0..3 {
                for j in 0..3 {
                    second[i][j] += (g[i].conj() * g[j]).re * w;
                }
            }
        }
        Recoil { mean, second }
    }
}

/// The hazard λ = ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖² of a generator H at an instant: Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖², the decay being the jumps'
/// Σ J†J by construction (module documentation, # The decay).
pub fn hazard(h: &Mat, psi: &[C64]) -> f64 {
    let hp = h.apply(psi);
    let hd = h.adjoint().apply(psi);
    let num: C64 = psi.iter().zip(hp.iter().zip(&hd)).map(|(p, (a, b))| p.conj() * (*a - *b)).sum();
    -num.im / psi.iter().map(|z| z.norm_sqr()).sum::<f64>()
}
