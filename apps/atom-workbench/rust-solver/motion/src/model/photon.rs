//! Photons (SPEC §1.6): the jump operators by class, the photons a state emits, the recoil's two moments.
//!
//! # Sources and classes
//! A source is one part of the jump operators, linear in one component of a frequency's field (or constant): a folded
//! component's photons C_a R from its field's component e_k on its basis (far.rs, eliminate.rs; the counter-rotating from e_k*),
//! or an explicit excited state's C_a. In the base frame its entries b' ← s carry the beat Ω_J -- −ν of the absorbed component,
//! 0 for an explicit state (the excited states' ν_frame, common to every photon, is a global phase) -- and, beam by beam of its
//! field, the wave vector q_J (the absorbed beam's k_l; −k_l for a counter-rotating component's a_l*). In the frames (cluster.rs)
//! an entry is at ω̃ = Ω_J + θ_b' − θ_s, its photon's frequency up to the class's common phase; a folded photon and an explicit one
//! of the same light land at the same ω̃ exactly. A class is the entries of one frame frequency landing in one cluster (model.rs
//! `Photon`); two frequencies landing in one cluster are two classes unless their coherence is kept (`Unit::Coherence`), then
//! one, each member at its relative phase e^{i(ω̃ − ω̃₀)t}. A near carrier's jump b' ← s exists where its response from s reaches an
//! excited state that decays to b' (assemble.rs `Model::reached`).
//!
//! # The decay (assemble.rs)
//! −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}: per class and pair of its members (j, i) the term e_j* e_i L_j†L_i, L the members' entries in
//! that class, at the beat Ω_i − Ω_j, whose residual in the frames is zero within one frequency: the decay is the jumps' own
//! product, the generator Lindblad's by construction, and the hazard ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖² is Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² identically
//! (`hazard`).
//!
//! # The recoil (SPEC §1.6, one rule)
//! The jump into f = Σ_a ε_a u_a has the absorption's conditional mean b = Im⟨f|∇f⟩/‖f‖² (the jump term's first Wigner moment)
//! and its local second moment M = Re⟨∇f|∇f⟩/‖f‖² (the second Kramers–Moyal coefficient of the method's own expansion). A folded
//! source's absorption mean is the jump's to carry: its ∇f is its field component's gradient on its entries (each beam's ∇a_l = a_l
//! (ik_l + ∇ln|a_l|)) and its jumps' own. An explicit state's mean is the Ehrenfest force's (its kept drives' stimulated force),
//! so its part of the kick has mean zero and the second moment of its drives' responses σ_d, Re[∇ln σ_d*⊗∇ln σ_d] by their shares
//! (eliminate.rs). The kick is b_J + ξ, Cov ξ = M − b_J b_Jᵀ, b_J = the folded mean: positive semidefinite by Cauchy–Schwarz.

use super::far::EXCITED;
use super::{Beat, Gate, Lights, Model, Photon, Point, Recoil, Sample, Unit, UnitError};
use crate::field::Amplitude;
use crate::linalg::{Mat, C64};
use crate::request::{GROUND, STATES};

/// What a source's jumps are (module documentation): a folded far component's on its field's basis vector (`side` 0 the
/// co-rotating, 1 the counter-rotating; constant, far.rs `Tone::jumps`), an explicit excited state's −C_a (the excited amplitude a
/// folded drive makes is −R x_g, so that a class's jump is C_a(R x_g − x_e), the physical photon's sign throughout), or a near
/// carrier's C_a(T_k − w T2_k) at the point (eliminate.rs).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Origin {
    Far { frequency: usize, side: usize, component: usize },
    Explicit { state: usize },
    Near { carrier: usize, component: usize },
}

/// One source of the jump operators (module documentation): what it is and its exact base beat.
#[derive(Clone, Debug)]
pub(crate) struct Source {
    pub(crate) origin: Origin,
    pub(crate) beat: Beat,
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
    /// The sources (module documentation): per folded far component and basis vector of its field, per explicit excited state,
    /// per near carrier and basis vector.
    pub(crate) fn all_sources(&self) -> Vec<Source> {
        let mut out = vec![];
        for (frequency, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            for side in 0..2 {
                if tone.folded[side] {
                    let nu = self.axes.component(&self.light, super::Component { frequency, counter: side == 1 });
                    out.extend((0..f.rank()).map(|component| Source { origin: Origin::Far { frequency, side, component }, beat: nu.negated() }));
                }
            }
        }
        out.extend((GROUND..STATES).filter(|&e| self.explicit(e)).map(|state| Source { origin: Origin::Explicit { state }, beat: self.axes.zero() }));
        for (carrier, cc) in self.carriers.iter().enumerate() {
            out.extend((0..cc.basis.len()).map(|component| Source { origin: Origin::Near { carrier, component }, beat: cc.beat.negated() }));
        }
        out
    }

    /// The frequency whose field a source takes: None for an explicit state's.
    pub(crate) fn source_frequency(&self, source: &Source) -> Option<usize> {
        match source.origin {
            Origin::Far { frequency, .. } => Some(frequency),
            Origin::Near { carrier, .. } => Some(self.carriers[carrier].frequency),
            Origin::Explicit { .. } => None,
        }
    }

    /// Whether a source's light is on.
    pub(crate) fn lit(&self, source: &Source, lights: &Lights) -> bool {
        self.source_frequency(source).map_or(true, |f| lights.on(f))
    }

    /// A far or explicit source's jump b ← s on axis a (constant).
    pub(crate) fn constant_jump(&self, source: &Source, a: usize, b: usize, s: usize) -> C64 {
        match source.origin {
            Origin::Far { frequency, side, component } if s < GROUND => self.tones[frequency].jumps[side][component][a][(b, s)],
            Origin::Explicit { state } if s == state => -self.emission[a][(b, state)],
            Origin::Near { .. } => unreachable!("a near source's jumps are the point's"),
            _ => C64::ZERO,
        }
    }

    /// Whether a source may jump from state s of a cluster to b on some axis: a constant one's nonzero entries; a near one's
    /// wherever its response from s reaches an excited state that decays to b.
    fn may_jump(&self, source: &Source, cluster: usize, b: usize, s: usize) -> bool {
        match source.origin {
            Origin::Near { carrier, component } => {
                s < GROUND && {
                    let reach = self.reached(cluster, carrier, component, s);
                    (0..EXCITED).any(|e| reach[e] && (0..3).any(|a| self.emission[a][(b, GROUND + e)] != C64::ZERO))
                }
            }
            _ => (0..3).any(|a| self.constant_jump(source, a, b, s) != C64::ZERO),
        }
    }

    /// A bound of a source's jump b ← s on axis a anywhere, per unit of its field component: a constant one's entry; a near
    /// one's |C_a[b, ·]| times the norm of its folded coupling's column at s, times 2/Γ_min (its response's resolvent is at most
    /// 2/Γ_min).
    fn jump_bound(&self, source: &Source, a: usize, b: usize, s: usize) -> f64 {
        match source.origin {
            Origin::Near { carrier, component } => {
                let x = &self.carriers[carrier].basis[component];
                let column = (GROUND..STATES).map(|e| x[(e, s)].norm_sqr()).sum::<f64>().sqrt();
                let row = (GROUND..STATES).map(|e| self.emission[a][(b, e)].norm_sqr()).sum::<f64>().sqrt();
                row * column * 2.0 / self.least_decay()
            }
            _ => self.constant_jump(source, a, b, s).abs(),
        }
    }

    /// A source's field component at its largest, every beam at its peak and in phase: Σ_l |γ_lk| (1 for an explicit state's).
    fn source_peak(&self, source: &Source) -> f64 {
        match (self.source_frequency(source), source.origin) {
            (Some(f), Origin::Far { component, .. } | Origin::Near { component, .. }) => self.light.frequencies[f].coefficients.iter().map(|c| c[component].abs()).sum(),
            _ => 1.0,
        }
    }

    /// A source's field component's modulus at a point, |Σ_l a_l γ_lk| (1 for an explicit state's).
    fn source_size(&self, source: &Source, amplitudes: &[Amplitude]) -> f64 {
        match (self.source_frequency(source), source.origin) {
            (Some(f), Origin::Far { component, .. } | Origin::Near { component, .. }) => {
                let f = &self.light.frequencies[f];
                f.beams.iter().zip(&f.coefficients).map(|(&l, c)| amplitudes[l].value * c[component]).sum::<C64>().abs()
            }
            _ => 1.0,
        }
    }

    /// The classes of a cluster's photons (module documentation, # Sources and classes): its sources' entries from its states,
    /// grouped by frame frequency and landing cluster, the kept coherences merged; in the order of their landing clusters and
    /// frame frequencies.
    pub(crate) fn classes(&self, cluster: usize) -> Vec<Class> {
        let c = &self.clusters[cluster];
        let mut out: Vec<Class> = vec![];
        for (index, source) in self.sources.iter().enumerate() {
            for (sp, &s) in c.states.iter().enumerate() {
                for b in 0..GROUND {
                    if !self.may_jump(source, cluster, b, s) {
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
        out.sort_by(|x, y| (x.landing, &x.beats[0]).cmp(&(y.landing, &y.beats[0])));
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

    /// The photon classes of a cluster's state ψ (its frame amplitudes) at a point, every light on (SPEC §1.6; module
    /// documentation): per class u_a on the landing cluster's states, the folded part's gradients, the explicit sources.
    pub fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        self.photons_counted(cluster, at, psi, None).0
    }

    /// The photons (`photons`) and the complex multiply-adds that made them: the near jumps at the point with their gradients
    /// (from their table where one serves, tabulate.rs), the field components, each entry's amplitude and its gradient, the
    /// explicit sources' shares.
    pub(crate) fn photons_counted(&self, cluster: usize, at: &Point, psi: &[C64], tables: Option<&super::tabulate::Tables>) -> (Vec<Photon>, f64) {
        let lights = self.running();
        let plan = self.plan(cluster, &lights);
        let amplitudes = self.light.amplitudes(at.r);
        let len = plan.near.len();
        let near = (len > 0).then(|| self.near_values(&plan, &amplitudes, &[(plan.near.jumping(), 1)], tables.and_then(|t| t.of(cluster, &lights))));
        let fields = self.fields(&plan, &amplitudes, at.kappa, 1);
        // Each field component's value and gradient from its beams, plain and Doppler-weighted.
        let mut work = near.as_ref().map_or(0.0, |n| n.work) + plan.scalars.iter().map(|s| (8 * s.beams.len()) as f64).sum::<f64>();
        // A jump with its gradient: a near form's, or a constant.
        let jump = |place: usize| -> [C64; 4] {
            match &near {
                Some(n) if place < len => n.first_order(place),
                _ => [plan.constants[place - len], C64::ZERO, C64::ZERO, C64::ZERO],
            }
        };
        let one = [C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO];
        let mut energies: Option<Vec<Vec<f64>>> = None;
        let mut out = vec![];
        for class in &plan.classes {
            let n = self.clusters[class.landing].states.len();
            let mut u: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n]);
            let mut gradients: [[Vec<C64>; 3]; 3] = std::array::from_fn(|_| std::array::from_fn(|_| vec![C64::ZERO; n]));
            let mut explicit = vec![];
            for m in &class.members {
                let folded = !matches!(self.sources[m.source].origin, Origin::Explicit { .. });
                let (c, cw): (&[C64], &[C64]) = m.scalar.map_or((&one, &one), |s| (&fields.plain[s], &fields.doppler[s]));
                let phase = C64::cis(m.phase * at.t);
                let mut own: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; if folded { 0 } else { n }]);
                for &(a, bp, sp, first, second) in &m.entries {
                    let x = psi[sp] * phase;
                    let j = jump(first);
                    let mut value = c[0] * j[0];
                    let mut gradient: [C64; 3] = std::array::from_fn(|i| c[1 + i] * j[0] + c[0] * j[1 + i]);
                    if m.doppler {
                        let j2 = jump(second);
                        value -= cw[0] * j2[0];
                        for i in 0..3 {
                            gradient[i] -= cw[1 + i] * j2[0] + cw[0] * j2[1 + i];
                        }
                    }
                    u[a][bp] += value * x;
                    if folded {
                        for i in 0..3 {
                            gradients[i][a][bp] += gradient[i] * x;
                        }
                    } else {
                        own[a][bp] += value * x;
                    }
                    let partner = if m.doppler { 1.0 } else { 0.0 };
                    work += 2.0 + partner + if folded { 3.0 * (3.0 + 2.0 * partner) } else { 0.0 };
                }
                if !folded {
                    let h = energies.get_or_insert_with(|| self.far_energies(&amplitudes, &lights, 1).0);
                    let (shares, w) = self.explicit_recoil(cluster, &amplitudes, h, m.source, own, psi);
                    explicit.extend(shares);
                    work += w;
                }
            }
            if u.iter().all(|v| v.iter().all(|z| *z == C64::ZERO)) {
                continue;
            }
            out.push(Photon { landing: class.landing, wave_number: class.wave_number, amplitudes: u, gradients, explicit });
        }
        (out, work)
    }

    /// A class's photon wave number: its light's k = 2π ν/c, the absorbed light's (the photon's frequency differs from it by the
    /// atom's energies, ≤ 1e-7 of it on the D lines) -- the largest beam wave number among its members' lights, the explicit
    /// states' through the frequencies that drive them.
    pub(crate) fn class_wave_number(&self, class: &Class) -> f64 {
        let mut k = 0.0_f64;
        for m in &class.members {
            let source = &self.sources[m.source];
            if let Some(f) = self.source_frequency(source).or_else(|| self.driving_frequency(source)) {
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

    /// A class's jump operators' sizes per axis with the sources' field components bounded by `most`: Σ_members |e| ‖L_a‖_F.
    fn class_size(&self, cluster: usize, class: &Class, most: &dyn Fn(&Source) -> f64) -> [f64; 3] {
        let mut out = [0.0; 3];
        for m in &class.members {
            let source = &self.sources[m.source];
            let c = most(source);
            for a in 0..3 {
                let norm: f64 = m
                    .entries
                    .iter()
                    .map(|&(bp, sp)| self.jump_bound(source, a, self.clusters[class.landing].states[bp], self.clusters[cluster].states[sp]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                out[a] += c * norm;
            }
        }
        out
    }

    /// A coherence's mixing with the sources' field components bounded by `most`: m = Σ_a |u_a||u'_a|/(|Ω| − spread), Ω the two
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

    /// A coherence's gate: its mixing with every beam at its peak and in phase (|a_l| ≤ 1 everywhere: a bound over any reach).
    pub(crate) fn coherence_gate(&self, unit: &Unit) -> Gate {
        let m = self.coherence_mixing(unit, &|s| self.source_peak(s));
        Gate { saturation: 2.0 * m * m, sweep: 0.0 }
    }

    /// A coherence's error at a sample: m² √(p_i p_j), p the populations of the two classes' source states.
    pub(crate) fn coherence_error(&self, unit: &Unit, sample: &Sample) -> UnitError {
        let amplitudes = self.light.amplitudes(sample.at.r);
        let m = self.coherence_mixing(unit, &|s| self.source_size(s, &amplitudes));
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
/// The anti-Hermitian part B = i(H − H†) is formed entry by entry before it meets ψ, so that the Hermitian part's scale (the
/// states' energies) does not round the rate.
pub fn hazard(h: &Mat, psi: &[C64]) -> f64 {
    let decay = Mat::from_fn(h.rows, h.cols, |x, y| C64::I * (h[(x, y)] - h[(y, x)].conj()));
    crate::linalg::inner(psi, &decay.apply(psi)).re / psi.iter().map(|z| z.norm_sqr()).sum::<f64>()
}
