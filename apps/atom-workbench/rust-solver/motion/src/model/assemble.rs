//! The one generator assembly (SPEC §1.4): a cluster's generator at a point, by parts of one beat and wave vector, its
//! Hermitian part's gradient beside it. The dynamics, the force, the hazard, the readout and the potential take its terms.
//!
//! # Terms and parts
//! Every term is a coefficient c(r) -- a beam pair's a_l'* a_l, a beam's a_l, or 1 -- times a matrix on the states, of base beat
//! Ω (∝ e^{iΩt}) and wave vector q, both exact (`Beat`, `Wave`): the base (H0 less the outside forms at the focus: every light
//! off), each lit frequency's far static form per beam pair (far.rs), the kept far beats, the photons' decay per class and pair of
//! members (photon.rs), the near light (eliminate.rs). Its entry a ← b goes to the part of residual beat Ω + θ_a − θ_b and
//! residual wave vector q − (K_a − K_b) in the cluster's frames (cluster.rs); the frames' −θ_s join the static part's diagonal.
//! A Hermitian term adds c M to its part and ∇c M to the part's force; a decay term adds −(i/2) c M and no force (the absorption's
//! mean force is the jumps', photon.rs). Each Hermitian term comes with its adjoint (the beam pair (l, l') with (l', l), a beat
//! with its opposite), so the parts pair, (Ω, q) with (−Ω, −q), and G_{−j} = G_j†.
//!
//! # Work
//! The assembly counts its complex multiply-adds as it makes them (`Work::assembly`): four per Hermitian entry (its value and
//! its gradient), one per decay entry, and the decay's products L_j†L_i entry by entry.

use super::photon::Class;
use super::{Beat, Generator, Lights, Model, Part, Point, Sample, Wave, Work};
use crate::field::Amplitude;
use crate::linalg::{Mat, C64};

/// The constant coefficient 1.
pub(crate) fn one() -> Amplitude {
    Amplitude { value: C64::ONE, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] }
}

/// conj(c) of a coefficient with its derivatives.
pub(crate) fn conjugate(c: &Amplitude) -> Amplitude {
    Amplitude { value: c.value.conj(), gradient: c.gradient.map(|z| z.conj()), hessian: c.hessian.map(|r| r.map(|z| z.conj())) }
}

/// The parts of a cluster's generator under assembly (module documentation).
pub(crate) struct Parts<'a> {
    model: &'a Model,
    cluster: usize,
    keys: Vec<(Beat, Wave)>,
    generator: Vec<Mat>,
    force: Vec<[Mat; 3]>,
    pub(crate) work: f64,
}

impl<'a> Parts<'a> {
    pub(crate) fn new(model: &'a Model, cluster: usize) -> Parts<'a> {
        Parts { model, cluster, keys: vec![], generator: vec![], force: vec![], work: 0.0 }
    }
    fn part(&mut self, beat: Beat, wave: Wave) -> usize {
        if let Some(at) = self.keys.iter().position(|(b, w)| *b == beat && *w == wave) {
            return at;
        }
        let n = self.model.clusters[self.cluster].states.len();
        self.keys.push((beat, wave));
        self.generator.push(Mat::zeros(n, n));
        self.force.push(std::array::from_fn(|_| Mat::zeros(n, n)));
        self.keys.len() - 1
    }
    /// Adds a term c M, its entries `entry(x, y)` by the cluster's places: Hermitian (with its force) or a decay.
    pub(crate) fn add(&mut self, beat: &Beat, wave: &Wave, c: &Amplitude, entry: &dyn Fn(usize, usize) -> C64, hermitian: bool) {
        let cluster = &self.model.clusters[self.cluster];
        let n = cluster.states.len();
        for x in 0..n {
            for y in 0..n {
                let m = entry(x, y);
                if m == C64::ZERO {
                    continue;
                }
                let b = beat.plus(1, &cluster.frames[x]).plus(-1, &cluster.frames[y]);
                let w = wave.plus(-1, &cluster.waves_exact[x]).plus(1, &cluster.waves_exact[y]);
                let p = self.part(b, w);
                if hermitian {
                    self.generator[p][(x, y)] += c.value * m;
                    for i in 0..3 {
                        self.force[p][i][(x, y)] += c.gradient[i] * m;
                    }
                    self.work += 4.0;
                } else {
                    self.generator[p][(x, y)] += C64::new(0.0, -0.5) * c.value * m;
                    self.work += 1.0;
                }
            }
        }
    }
    /// Adds a Hermitian term c M(r) whose matrix varies with the point: its entries `value(x, y)` and their gradients
    /// `gradient(i, x, y)`, the force ∇c M + c ∇M.
    pub(crate) fn add_varying(&mut self, beat: &Beat, wave: &Wave, c: &Amplitude, value: &dyn Fn(usize, usize) -> C64, gradient: &dyn Fn(usize, usize, usize) -> C64) {
        let cluster = &self.model.clusters[self.cluster];
        let n = cluster.states.len();
        for x in 0..n {
            for y in 0..n {
                let m = value(x, y);
                if m == C64::ZERO {
                    continue;
                }
                let b = beat.plus(1, &cluster.frames[x]).plus(-1, &cluster.frames[y]);
                let w = wave.plus(-1, &cluster.waves_exact[x]).plus(1, &cluster.waves_exact[y]);
                let p = self.part(b, w);
                self.generator[p][(x, y)] += c.value * m;
                for i in 0..3 {
                    self.force[p][i][(x, y)] += c.gradient[i] * m + c.value * gradient(i, x, y);
                }
                self.work += 7.0;
            }
        }
    }
    /// The parts, the frames' −θ_s on the static part's diagonal.
    pub(crate) fn finish(mut self, at: &Point) -> Generator {
        let model = self.model;
        let cluster = &model.clusters[self.cluster];
        let still = self.part(model.axes.zero(), model.no_wave());
        for (x, theta) in cluster.frames.iter().enumerate() {
            self.generator[still][(x, x)] -= C64::real(model.omega(theta));
        }
        let parts = self
            .keys
            .into_iter()
            .zip(self.generator.into_iter().zip(self.force))
            .map(|((beat, wave), (generator, force))| Part { omega: model.omega(&beat), wave: model.wave_value(&wave), beat, generator, force })
            .collect();
        Generator { cluster: self.cluster, at: *at, parts }
    }
}

impl Model {
    /// The one generator assembly (module documentation).
    pub fn assemble(&self, cluster: usize, at: &Point, lights: &Lights) -> Generator {
        self.assembled(cluster, at, lights).0
    }

    /// The assembly and the multiply-adds it counted.
    pub(crate) fn assembled(&self, cluster: usize, at: &Point, lights: &Lights) -> (Generator, f64) {
        let c = &self.clusters[cluster];
        let amplitudes = self.light.amplitudes(at.r);
        let mut parts = Parts::new(self, cluster);
        let zero = self.axes.zero();
        let still = self.no_wave();
        parts.add(&zero, &still, &one(), &|x, y| self.base[(c.states[x], c.states[y])], true);
        for (w, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            let n = f.beams.len();
            for j in 0..n {
                for i in 0..n {
                    if !lights.pair(w, j, i) {
                        continue;
                    }
                    let p = &tone.pairs[j * n + i];
                    let coefficient = super::far::pair(&amplitudes[f.beams[j]], &amplitudes[f.beams[i]]);
                    parts.add(&zero, &self.pair_wave(f.beams[j], f.beams[i]), &coefficient, &|x, y| p[(c.states[x], c.states[y])], true);
                }
            }
        }
        for unit in &self.kept {
            if let super::Unit::FarBeat { first, second } = unit {
                if !lights.cross(first.frequency, second.frequency) {
                    continue;
                }
                self.add_far_beat(&mut parts, *first, *second, &amplitudes);
            }
        }
        let near = self.carriers.iter().any(|x| lights.on(x.frequency)).then(|| self.near_at(cluster, at, &amplitudes, lights, 1));
        for class in &self.classes[cluster] {
            self.add_decay(&mut parts, cluster, class, &amplitudes, lights, near.as_ref());
        }
        if let Some(near) = &near {
            self.add_near_terms(&mut parts, cluster, near, &amplitudes, lights);
        }
        let work = parts.work;
        (parts.finish(at), work)
    }

    /// A kept far beat's cross terms and their adjoints (units.rs `Model::far_cross`): absorbing `first` and emitting `second` at
    /// the base beat ν₂ − ν₁, coefficient a_l₂* a_l₁ (co-rotating; a_l₂ a_l₁* counter-rotating), wave vector k_l₁ − k_l₂ (its negative).
    fn add_far_beat(&self, parts: &mut Parts, first: super::Component, second: super::Component, amplitudes: &[Amplitude]) {
        let c = &self.clusters[parts.cluster];
        let (f1, f2) = (&self.light.frequencies[first.frequency], &self.light.frequencies[second.frequency]);
        let beat = self.axes.component(&self.light, second).plus(-1, &self.axes.component(&self.light, first));
        let cross = self.far_cross(first, second);
        for (j, &l2) in f2.beams.iter().enumerate() {
            for (i, &l1) in f1.beams.iter().enumerate() {
                let m = &cross[j * f1.beams.len() + i];
                let (coefficient, wave) = if first.counter {
                    (super::far::pair(&amplitudes[l1], &amplitudes[l2]), self.pair_wave(l1, l2))
                } else {
                    (super::far::pair(&amplitudes[l2], &amplitudes[l1]), self.pair_wave(l2, l1))
                };
                parts.add(&beat, &wave, &coefficient, &|x, y| m[(c.states[x], c.states[y])], true);
                parts.add(&beat.negated(), &wave.negated(), &conjugate(&coefficient), &|x, y| m[(c.states[y], c.states[x])].conj(), true);
            }
        }
    }

    /// A class's decay (photon.rs, # The decay): per axis and pair of lit members (j, i), c_j* c_i L_j†L_i on the cluster's places,
    /// at the base beat Ω_i − Ω_j and wave vector q_i − q_j.
    fn add_decay(&self, parts: &mut Parts, cluster: usize, class: &Class, amplitudes: &[Amplitude], lights: &Lights, near: Option<&super::eliminate::NearAt>) {
        let c = &self.clusters[cluster];
        let landing = &self.clusters[class.landing];
        let n = c.states.len();
        for mj in &class.members {
            let sj = &self.sources[mj.source];
            if !self.lit(sj, lights) {
                continue;
            }
            for mi in &class.members {
                let si = &self.sources[mi.source];
                if !self.lit(si, lights) {
                    continue;
                }
                let mut product = Mat::zeros(n, n);
                for a in 0..3 {
                    for &(bj, xj) in &mj.entries {
                        for &(bi, xi) in &mi.entries {
                            if bj == bi {
                                let b = landing.states[bj];
                                let (vj, _) = self.jump(sj, a, b, c.states[xj], xj, near);
                                let (vi, _) = self.jump(si, a, b, c.states[xi], xi, near);
                                product[(xj, xi)] += vj.conj() * vi;
                                parts.work += 1.0;
                            }
                        }
                    }
                }
                let coefficient = super::far::pair(&self.coefficient(sj, amplitudes), &self.coefficient(si, amplitudes));
                let beat = si.beat.plus(-1, &sj.beat);
                let wave = si.wave.plus(-1, &sj.wave);
                parts.add(&beat, &wave, &coefficient, &|x, y| product[(x, y)], false);
            }
        }
    }

    /// The counted work at trajectory states (model.rs `Work`): the assembly's multiply-adds, the clusters' dimension and parts,
    /// means over the samples, every light on.
    pub fn work(&self, samples: &[Sample]) -> Work {
        let lights = self.running();
        let mut out = Work { assembly: 0.0, states: 0.0, parts: 0.0 };
        for s in samples {
            let (g, w) = self.assembled(s.cluster, &s.at, &lights);
            out.assembly += w;
            out.states += self.clusters[s.cluster].states.len() as f64;
            out.parts += g.parts.len() as f64;
        }
        let n = samples.len().max(1) as f64;
        Work { assembly: out.assembly / n, states: out.states / n, parts: out.parts / n }
    }
}
