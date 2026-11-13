//! Photons (SPEC §1.6): the jump operators by class, the photons a state emits, the absorbed light's two moments per unit
//! time.
//!
//! # Sources and classes
//! A source is one part of the jump operators, linear in one component of a frequency's field (or constant): a component's
//! folded drives' photons C_a R from its field's component e_k on its basis (the counter-rotating from e_k*) -- R the whole
//! response of its folded drives, the bare ones' constant and, of a carrier, that of its drives folded at the atom
//! (far.rs, eliminate.rs): one amplitude, one source --, or an explicit excited state's C_a. In the base frame its entries b' ← s carry the beat Ω_J -- −ν of the absorbed component,
//! 0 for an explicit state (the excited states' ν_frame, common to every photon, is a global phase) -- and, beam by beam of its
//! field, the wave vector q_J (the absorbed beam's k_l; −k_l for a counter-rotating component's a_l*). In the frames (cluster.rs)
//! an entry is at ω̃ = Ω_J + θ_b' − θ_s, its photon's frequency up to the class's common phase; a folded photon and an explicit one
//! of the same light land at the same ω̃ exactly. A class is the entries of one frame frequency landing in one cluster (model.rs
//! `Photon`); two frequencies landing in one cluster are two classes unless their coherence is kept (`Unit::Coherence`), then
//! one, each member at its relative phase e^{i(ω̃ − ω̃₀)t}. A carrier's jump b' ← s exists where its response from s reaches an
//! excited state that decays to b' (assemble.rs `Model::reached`).
//!
//! # The decay (assemble.rs)
//! −(i/2) Σ_φ Σ_a J_{a,φ}†J_{a,φ}: per class and pair of its members (j, i) the term e_j* e_i L_j†L_i, L the members' entries in
//! that class, at the beat Ω_i − Ω_j, whose residual in the frames is zero within one frequency: the decay is the jumps' own
//! product, the generator Lindblad's by construction, and the hazard Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² (SPEC §1.4, the photons' own
//! weights) is the propagated norm's rate ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖² identically (step.rs; the identity is the tests').
//! A table's middle takes the decay's sums interpolated, the photons the interpolated jumps: there the identity holds to the
//! table's error, which its reads take (tabulate.rs, # A node's reals, # The table's error).
//!
//! # A state's photons at a point
//! A class's u on the rows a n + b' (the axis a, the landing cluster's n states b') is the sum over its members of the
//! member's field component times its jumps' matrix on the state (assemble.rs, # The jumps' matrices): the matrix on the
//! state once (linalg.rs `combine`: the state's amplitudes on the matrix's columns), then the product with the field's
//! component -- where the member holds a near form with its Doppler partner, the partner's matrix on the same state
//! times the Doppler-weighted component. ∇u takes the same vectors with the components' gradients, and of a member that
//! holds a near form its matrices' slopes on the state, each along its direction (a table's axes' coordinates'
//! gradients, or the lab axes) times the component: the chain rule after the sums, not entry by entry. The matrices at
//! the point are the exact elimination's jumps there with their exact derivatives, or their table's interpolant with its
//! own slopes (tabulate.rs `Model::near_jumps`): one sum either way, the definition's entries (`MemberPlan::entries`)
//! in another order. A class's moments (# The absorbed light's momentum) are ten sums over its rows.
//!
//! # A class's photon
//! Its frequency is the frame's less the class's frame frequency, ω_φ = ω_frame − ω̃ + E_s₀ − E_b₀: the frames are counted from
//! each cluster's first state (cluster.rs), so the bare energies E of the source cluster's first state s₀ and of the landing
//! cluster's b₀ make the frame frequency the photon's own -- the absorbed light's for a state that comes back (Rayleigh), moved by
//! the ground states' splitting for one that does not (Raman), the line's for an explicit state's. The states of a cluster differ
//! from their frames by its residual detunings, a part in 10⁷ of an optical frequency; a class kept coherent over several frame
//! frequencies takes its first, the others a kept beat away. A counter-rotating component's class has ω_φ < 0, its photon the
//! light's own: |k_φ| = |ω_φ|/c (units.rs `wave_number`).
//!
//! # The absorbed light's momentum (SPEC §1.6, one rule)
//! Per unit time the light a class absorbs gives the momentum the local Kramers–Moyal coefficients of the method's own
//! expansion, summed over the emitted photon's polarizations a: the scattering force Σ_a Im⟨u_a|∇u_a⟩ and the second moment's
//! rate Σ_a Re⟨∇u_a|⊗|∇u_a⟩ (`Photon::moments`), neither divided by the class's rate W = Σ_a ‖u_a‖² -- both smooth and
//! finite where the light interferes away (half of the second moment is the evolution without a jump's, and no photon's). A
//! folded source's ∇u is its field component's gradient on its entries (each beam's ∇a_l = a_l (ik_l + ∇ln|a_l|)) and its
//! jumps' own. A jump of the class carries its conditional mean b = Σ_a Im⟨u_a|∇u_a⟩/W as far as one absorbed photon's
//! momentum goes, s b with s = min(1, k_c/|b|) (`Photon::kick`): b is the beams' wave vectors weighted, within their
//! convex hull while the weights are positive, and beyond k_c only where they interfere to both signs -- there it is no
//! photon's momentum. What the jumps do not carry is the step's (`Model::scattering_of`): the force Σ_c (1 − s_c) Σ_a
//! Im⟨u_a|∇u_a⟩ and the diffusion 2D = Σ_c [Σ_a Re⟨∇u_a|⊗|∇u_a⟩ − s_c² W_c b_c b_cᵀ], positive semidefinite by
//! Cauchy–Schwarz. A plane wave's jump carries its k and leaves nothing; a standing wave's carries nothing and leaves
//! k² R₀ sin² kz, the dipole force's fluctuations.
//!
//! An explicit state's jumps are constants and its absorption's mean is the Ehrenfest force's (its kept drives'
//! stimulated force): what its absorbed light gives the momentum beside that mean is the force's own correlation through
//! the optical sector, the trajectory's per-state noise (optical.rs; step.rs, # The optical sector's noise), not a
//! photon's moment here.

use super::assemble::{laid, sized, ClassPlan, Held, MemberPlan, Plan, Site};
use super::far::EXCITED;
use super::own::{norm_of, product, Form, Own};
use super::reach::Region;
use super::tabulate::NearTable;
use super::units::{gate, Measured};
use super::{Beat, Lights, Model, Moments, Photon, Point, Scattering, Unit, UnitError};
use crate::step::Emitted;
use std::sync::Arc;

use crate::linalg::{combine, C64, LANES};
use crate::request::{GROUND, STATES};
use crate::table::MOST_AXES;
use crate::units::{wave_number, RAD_PER_US_PER_MHZ};

/// What a source's jumps are (module documentation): a component's bare drives' on its field's basis vector (`Far`: `side` 0
/// the co-rotating, 1 the counter-rotating; constant, far.rs `Tone::jumps`), an explicit excited state's −C_a (the excited
/// amplitude a folded drive makes is −R x_g, so that a class's jump is C_a(R x_g − x_e), the physical photon's sign
/// throughout), or a carrier's (`Near`): its bare drives' constant and its drives' folded at the atom, C_a(T_k − w T2_k) at
/// the point (eliminate.rs), together.
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

/// Which classes' gradients a state's photons are made with (`Model::photons_of`): every class's (a step's scattering,
/// the one reader of the second moment); one class's, that class the only photon made, for its mean (a jump's: the class
/// drawn and the kick it carries, `Model::emission_of`); or none's, every class's amplitudes alone (what the draw of a
/// class reads: their rates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Gradients {
    All,
    Of(usize),
    None,
}

/// A class of a source cluster's photons: its frame frequencies (one, or several kept coherent), its landing cluster, its members.
#[derive(Clone, Debug)]
pub(crate) struct Class {
    pub(crate) beats: Vec<Beat>,
    pub(crate) landing: usize,
    pub(crate) members: Vec<Member>,
}

impl Model {
    /// The sources (module documentation): per component with a bare drive that is no carrier and basis vector of its field,
    /// per explicit excited state, per carrier and basis vector.
    pub(crate) fn all_sources(&self) -> Vec<Source> {
        let mut out = vec![];
        for (frequency, (f, tone)) in self.light.frequencies.iter().zip(&self.tones).enumerate() {
            for side in 0..2 {
                if tone.bare[side] && !(side == 0 && self.carrier_of(frequency).is_some()) {
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

    /// The constant part of a source's jump b ← s on axis a: a bare or explicit source's jump, and of a carrier's its bare
    /// drives'.
    pub(crate) fn constant_jump(&self, source: &Source, a: usize, b: usize, s: usize) -> C64 {
        let bare = |frequency: usize, side: usize, component: usize| self.tones[frequency].jumps[side].get(component).map_or(C64::ZERO, |jumps| jumps[a][(b, s)]);
        match source.origin {
            Origin::Far { frequency, side, component } if s < GROUND => bare(frequency, side, component),
            Origin::Near { carrier, component } if s < GROUND => bare(self.carriers[carrier].frequency, 0, component),
            Origin::Explicit { state } if s == state => -self.emission[a][(b, state)],
            _ => C64::ZERO,
        }
    }

    /// Whether a source may jump from state s of a cluster to b on some axis: where its constant part is not zero, and a
    /// carrier's wherever its response at the atom from s reaches an excited state that decays to b.
    fn may_jump(&self, source: &Source, cluster: usize, b: usize, s: usize) -> bool {
        let at_the_atom = match source.origin {
            Origin::Near { carrier, component } => {
                s < GROUND && {
                    let reach = self.reached(cluster, carrier, component, s);
                    (0..EXCITED).any(|e| reach[e] && (0..3).any(|a| self.emission[a][(b, GROUND + e)] != C64::ZERO))
                }
            }
            _ => false,
        };
        at_the_atom || (0..3).any(|a| self.constant_jump(source, a, b, s) != C64::ZERO)
    }

    /// A source's jump b ← s on axis a per unit of its field component over a region, complex (own.rs `Form`): its constant
    /// part's entry, and of a carrier's part at the atom Σ_e C_a[b, e] U_k[e, s]/(Δ_es − iΓ_e/2), Δ_es the drive's detuning of
    /// the levels the eliminations take (eliminate.rs `Gap`; own.rs `Own::resolvent`) -- the sum's own form, the excited
    /// states' paths with their phases: far from the lines, where 1/(Δ − iΓ/2) is nearly one number for every excited
    /// state, their sum is the couplings' own sum over the manifold, which may be nothing where each path is not. Beside it,
    /// how far the excited states' far couplings move it (own.rs `Band`): per path |w_e| r_e/(|Δ_es − iΓ_e/2| σ_es), r_e the
    /// excited state's Gershgorin radius and σ_es the denominator's bound (units.rs `Model::denominator_own`).
    fn jump_form(&self, source: &Source, a: usize, b: usize, s: usize, region: &Region) -> (Form<C64>, Own) {
        let mut jump = Form::constant(self.constant_jump(source, a, b, s));
        let mut spread = Own::constant(0.0);
        if let Origin::Near { carrier, component } = source.origin {
            if s < GROUND {
                let cc = &self.carriers[carrier];
                let radii = self.radii(region);
                for e in GROUND..STATES {
                    let w = self.emission[a][(b, e)] * cc.basis[component][(e, s)];
                    if w == C64::ZERO {
                        continue;
                    }
                    let gamma = 0.5 * self.decay[(e - GROUND, e - GROUND)].re;
                    let detuning = self.gap(e, s, region, cc.detuning).taken;
                    jump = jump.add(detuning.resolvent(gamma).turned(w));
                    spread = spread.add(radii[e].div(detuning.hypot(gamma).mul(self.denominator_own(region, e, s, cc.detuning))).scale(w.abs()));
                }
            }
        }
        (jump, spread)
    }

    /// A source's field component over a region, Σ_l a_l γ_lk (reach.rs `Region::field`; 1 for an explicit state's).
    fn source_field(&self, source: &Source, region: &Region) -> Form<C64> {
        match (self.source_frequency(source), source.origin) {
            (Some(f), Origin::Far { component, .. } | Origin::Near { component, .. }) => {
                let f = &self.light.frequencies[f];
                region.field(f.beams.iter().zip(&f.coefficients).map(|(&l, c)| (l, c[component], c[component].abs()))).0
            }
            _ => Form::constant(C64::ONE),
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
    /// documentation): per class u_a on the landing cluster's states and the folded part's gradients.
    pub fn photons(&self, cluster: usize, at: &Point, psi: &[C64]) -> Vec<Photon> {
        self.photons_counted(cluster, at, psi, None).0
    }

    /// The photons (`photons`) with every class's gradients, and the complex multiply-adds that made them; the near jumps
    /// from their table where one serves (tabulate.rs).
    pub(crate) fn photons_counted(&self, cluster: usize, at: &Point, psi: &[C64], tables: Option<&super::tabulate::Tables>) -> (Vec<Photon>, f64) {
        let lights = self.running();
        self.photons_of(&self.plan(cluster, &lights), tables.and_then(|t| t.of(cluster, &lights)), at, psi, Gradients::All)
    }

    /// The photons of a state under a plan's lights (`photons`), made as far as `gradients` says -- every class's amplitudes
    /// with every class's gradients, one class alone with its gradients, or every class's amplitudes without any (a class's
    /// gradients are then empty) --, and the complex multiply-adds that made them: the near jumps at the point (with their
    /// gradients where any class takes them; from their table where one is given), the field components, each entry's
    /// amplitude and its gradient. What a jump shares across its classes is in a thread's spare (assemble.rs `Held`):
    /// beyond the photons' own vectors the heap is asked for nothing.
    pub(crate) fn photons_of(&self, plan: &Arc<Plan>, table: Option<&NearTable>, at: &Point, psi: &[C64], gradients: Gradients) -> (Vec<Photon>, f64) {
        let mut out = Vec::with_capacity(plan.classes.len());
        let work = self.classes_at(plan, table, at, psi, gradients, &mut |place, class, sums, with| {
            let n = class.states;
            let entry = |column: usize, row: usize| C64::new(sums[laid(row, column, 4)], sums[laid(row, column, 4) + LANES]);
            let vector = |column: usize, a: usize| -> Vec<C64> { (0..n).map(|b| entry(column, a * n + b)).collect() };
            let amplitudes = std::array::from_fn(|a| vector(0, a));
            let gradients = std::array::from_fn(|i| std::array::from_fn(|a| if with { vector(1 + i, a) } else { vec![] }));
            out.push(Photon { class: place, landing: class.landing, wave_number: class.wave_number, absorbed: class.absorbed, amplitudes, gradients });
        });
        (out, work)
    }

    /// A jump's photon of a state under a plan's lights at a point (step.rs `Dynamics::emit`, `Emitted`), on the point's
    /// one site, made to first order (`sited`): every class's rate W = Σ_a ‖u_a‖² from its amplitudes alone
    /// (`Gradients::None`: the numbers `photons_of` gives, summed as `Photon::weight` sums them), in the classes' order,
    /// to `draw`; and the class it draws made alone with its gradients (`Gradients::Of`) -- its amplitudes and the mean
    /// a jump of it carries (`Moments::kick`), read from its sums where they are made. None where `draw` picks none.
    /// A jump reads no other class's amplitudes, and none are kept; nothing of the point is made twice; the heap is
    /// asked for nothing (the rates are the site's).
    pub(crate) fn emission_of(&self, plan: &Arc<Plan>, table: Option<&NearTable>, at: &Point, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        let mut site = Held::<Site>::take();
        self.pointed(plan, table, at.r, &mut site);
        let made = self.sited(plan, table, Some(at.r), at.kappa, 1, &mut site, false);
        // u_a on the rows a n + b' (`classes_on`: the sums' first column).
        let amplitude = |sums: &[f64], row: usize| C64::new(sums[laid(row, 0, 4)], sums[laid(row, 0, 4) + LANES]);
        let (mut emitting, mut rates) = (std::mem::take(&mut site.emitting), std::mem::take(&mut site.rates));
        emitting.clear();
        rates.clear();
        self.classes_on(plan, at.t, psi, Gradients::None, &mut site, made, &mut |place, class, sums, _| {
            let n = class.states;
            emitting.push(place);
            rates.push((0..3).map(|a| (0..n).map(|b| amplitude(sums, a * n + b).norm_sqr()).sum::<f64>()).sum());
        });
        let drawn = draw(&rates).map(|pick| emitting[pick]);
        (site.emitting, site.rates) = (emitting, rates);
        let mut emitted = None;
        self.classes_on(plan, at.t, psi, Gradients::Of(drawn?), &mut site, made, &mut |_, class, sums, with| {
            let n = class.states;
            let amplitudes = std::array::from_fn(|a| (0..n).map(|b| amplitude(sums, a * n + b)).collect());
            emitted = Some(Emitted { landing: class.landing, wave_number: class.wave_number, amplitudes, kick: moments(sums, with, class.absorbed).kick });
        });
        Some(emitted.expect("the drawn class at the state it was drawn at"))
    }

    /// The complex multiply-adds of a jump's photon (`emission_of`) of a cluster's state at a point, every light on:
    /// the site once, to first order (`sited`); every class's amplitudes, for the rates; and one class made alone
    /// with its gradients -- the dearest of the cluster's, whichever a jump draws. Counted where `emission_of` makes
    /// them (`sited`, `classes_on`), the site's in the first pass alone.
    pub(crate) fn emission_counted(&self, cluster: usize, at: &Point, psi: &[C64], tables: Option<&super::tabulate::Tables>) -> f64 {
        let lights = self.running();
        let (plan, table) = (self.plan(cluster, &lights), tables.and_then(|t| t.of(cluster, &lights)));
        let mut site = Held::<Site>::take();
        self.pointed(&plan, table, at.r, &mut site);
        let made = self.sited(&plan, table, Some(at.r), at.kappa, 1, &mut site, false);
        let rates = self.classes_on(&plan, at.t, psi, Gradients::None, &mut site, made, &mut |_, _, _, _| {});
        let drawn = (0..plan.classes.len()).map(|place| self.classes_on(&plan, at.t, psi, Gradients::Of(place), &mut site, 0.0, &mut |_, _, _, _| {})).fold(0.0, f64::max);
        rates + drawn
    }

    /// What `photons_of` makes, class by class to `each` as it is made (`classes_on`), on the point's own site
    /// (`Model::pointed`) made to the order the gradients ask (`sited`); and the complex multiply-adds, counted as the
    /// definition's entries (assemble.rs, # Work).
    fn classes_at(&self, plan: &Arc<Plan>, table: Option<&NearTable>, at: &Point, psi: &[C64], gradients: Gradients, each: &mut dyn FnMut(usize, &ClassPlan, &[f64], bool)) -> f64 {
        let mut site = Held::<Site>::take();
        self.pointed(plan, table, at.r, &mut site);
        let made = self.sited(plan, table, Some(at.r), at.kappa, if gradients == Gradients::None { 0 } else { 1 }, &mut site, false);
        self.classes_on(plan, at.t, psi, gradients, &mut site, made, each)
    }

    /// What the photons of a plan's states at a point read of the point's site (assemble.rs `Site`; module
    /// documentation, # A state's photons at a point), whatever the state, made on a site that holds the point
    /// (assemble.rs `Model::pointed`: every beam's amplitude with its gradient, its place on the table) -- a site of
    /// its own, or a middle's, whose forms are `laid` there already: the fourth order's matrices (assemble.rs
    /// `Model::quartic`, the position `r` their key: the step's middle's where the thread still holds them); the
    /// jumps' matrices to `order` -- 0, or 1 with their slopes: a table's, or the exact elimination's --; and the plan's
    /// field components to that order at the momentum κ. What lies within an order is the same numbers at every order.
    /// Returns the site's counted work: the jumps' matrices', the fourth order's, and each field component's value
    /// (and gradient) from its beams, plain and Doppler-weighted.
    #[allow(clippy::too_many_arguments)]
    fn sited(&self, plan: &Arc<Plan>, table: Option<&NearTable>, r: Option<[f64; 3]>, kappa: [f64; 3], order: usize, site: &mut Site, laid: bool) -> f64 {
        let fourth = plan.fourth.as_ref().map_or(0.0, |fourth| self.quartic(plan, fourth, table, r, site, laid));
        let near = match table {
            _ if plan.near.len() == 0 => {
                site.jumped(plan, None);
                0.0
            }
            Some(table) => self.near_jumps(table, order, site),
            None => {
                let near = self.near_exact(plan, &site.beams, order, &self.coherent_dressing());
                site.jumped(plan, Some(&near));
                near.work
            }
        };
        let Site { beams, fields, .. } = site;
        self.fields_into(plan, beams, kappa, order, fields);
        near + fourth + plan.fields_work(order)
    }

    /// The classes of a state's photons at the time t on a point's site (`sited`: its work `made`),
    /// class by class to `each` as each is made -- the class's place and plan, its sums (four columns laid for the lanes on the class's rows a n + b',
    /// assemble.rs `laid`: u, then per lab axis i ∇_i u, which hold nothing where the class takes no gradients), and
    /// whether it took gradients --, a class the state gives nothing left out; and the complex multiply-adds with the
    /// site's, counted as the definition's entries (assemble.rs, # Work). A class made with its gradients asks them of
    /// the site: it is made to first order. Module documentation, # A state's photons at a point: the sums are the site's,
    /// a thread's spare's (`Held`), and a step's scattering (`scattering_of`) and a jump's photon (`emission_of`) read
    /// them where they are made and ask the heap for nothing.
    #[allow(clippy::too_many_arguments)]
    fn classes_on(&self, plan: &Arc<Plan>, t: f64, psi: &[C64], gradients: Gradients, site: &mut Site, made: f64, each: &mut dyn FnMut(usize, &ClassPlan, &[f64], bool)) -> f64 {
        let order = if gradients == Gradients::None { 0 } else { 1 };
        let mut work = made;
        // The fourth order's states (assemble.rs, # The fourth order): (1 − ½X)ψ and Nψ, and with the gradients the
        // three lab axes' −½∂X ψ and ∂N ψ.
        let bent: Option<[[C64; STATES]; 8]> = plan.fourth.as_ref().map(|fourth| {
            let (n, size) = (fourth.n, fourth.n * fourth.n);
            crate::model::Model::quartic_read(|q| {
                let mut out = [[C64::ZERO; STATES]; 8];
                let (near, metric) = (&q.data[..4 * size], &q.data[8 * size..]);
                for part in 0..if order >= 1 { 4 } else { 1 } {
                    for x in 0..n {
                        let (mut bend, mut shift) = (C64::ZERO, C64::ZERO);
                        for (y, z) in psi.iter().enumerate().take(n) {
                            bend += metric[part * size + x * n + y] * *z;
                            shift += near[part * size + x * n + y] * *z;
                        }
                        match part {
                            0 => (out[0][x], out[1][x]) = (psi[x] - bend * 0.5, shift),
                            _ => (out[1 + part][x], out[4 + part][x]) = (bend * -0.5, shift),
                        }
                    }
                }
                work += (2 * size * if order >= 1 { 4 } else { 1 }) as f64;
                out
            })
        });
        // The lab axes along which the fourth order's matrices change at all (a light along one axis: that one).
        let (mut axes, mut count) = ([0; 3], 0);
        for a in 0..3 {
            if bent.as_ref().is_some_and(|b| b[2 + a].iter().chain(&b[5 + a]).any(|z| *z != C64::ZERO)) {
                (axes[count], count) = (a, count + 1);
            }
        }
        let live = &axes[..count];
        let Site { directions, jumps, jump_slopes, fields, member: vectors, class: sums, .. } = site;
        // A member's vectors: its matrix on the state, a point's with its partner's and each direction's two slopes.
        // The most a member makes (below: `base + more` of a near member with the fourth order and the gradients), s
        // the directions' slopes, a table's axes or the three lab axes: 2(1 + s) its jumps' and its partner's with their
        // slopes, 1 + s the fourth order's partner on Nψ with its slopes, and along each of the three lab axes the jumps
        // on −½∂Xψ and the partner on ∂Nψ: 3(1 + s) + 6.
        const MOST: usize = 3 * (1 + if MOST_AXES > 3 { MOST_AXES } else { 3 }) + 6;
        let (width, most) = (jumps.len(), plan.classes.iter().map(|class| class.rows).max().unwrap_or(0));
        sized(vectors, MOST * 2 * most);
        sized(sums, 8 * most);
        let one = [C64::ONE, C64::ZERO, C64::ZERO, C64::ZERO];
        let (mut x, mut weights) = ([C64::ZERO; STATES], [C64::ZERO; 4 * MOST]);
        for (place, class) in plan.classes.iter().enumerate() {
            let with = match gradients {
                Gradients::All => true,
                Gradients::Of(only) if only == place => true,
                Gradients::Of(_) => continue,
                Gradients::None => false,
            };
            let rows = class.rows;
            let sums = &mut sums[..8 * rows];
            // Whether u, and ∇u, hold nothing yet: a class's first member writes them, the others add to them.
            let (mut first, mut flat) = (true, true);
            for m in class.members.iter().filter(|m| !m.columns.is_empty()) {
                let (c, cw): (&[C64], &[C64]) = m.scalar.map_or((&one, &one), |s| (fields.plain(s), fields.doppler(s)));
                // The state on the member's columns, at the member's relative phase in a merged class.
                let (count, matrix) = (m.columns.len(), 2 * rows * m.columns.len());
                if m.phase == 0.0 {
                    x.iter_mut().zip(&m.columns).for_each(|(x, &sp)| *x = psi[sp]);
                } else {
                    let phase = C64::cis(m.phase * t);
                    x.iter_mut().zip(&m.columns).for_each(|(x, &sp)| *x = psi[sp] * phase);
                }
                let x = &x[..count];
                // Its matrices on the state: its jumps', and where they are a point's the partner's and their slopes'.
                let taken = m.folded && with;
                let slopes = if taken && m.near { directions.len() } else { 0 };
                let base = if m.near { 2 * (1 + slopes) } else { 1 };
                // The fourth order's vectors after them: a point's partner on Nψ with its slopes, and with the gradients
                // the jumps on −½∂X ψ and the partner on ∂N ψ along each lab axis; a constant member's jumps on −½∂X ψ.
                let more = match (&bent, m.near) {
                    (None, _) => 0,
                    (Some(_), true) => 1 + slopes + if taken { 2 * live.len() } else { 0 },
                    (Some(_), false) => if taken { live.len() } else { 0 },
                };
                let made = base + more;
                let vectors = &mut vectors[..made * 2 * rows];
                // A state of the cluster on the member's columns, at its phase.
                let gathered = |of: &[C64; STATES]| -> [C64; STATES] {
                    let phase = if m.phase == 0.0 { C64::ONE } else { C64::cis(m.phase * t) };
                    let mut out = [C64::ZERO; STATES];
                    out.iter_mut().zip(&m.columns).for_each(|(z, &sp)| *z = of[sp] * phase);
                    out
                };
                // The jumps act on the state, or on the fourth order's (1 − ½X)ψ.
                let bend = bent.as_ref().map(|b| gathered(&b[0]));
                let on: &[C64] = bend.as_ref().map_or(x, |b| &b[..count]);
                if m.near {
                    combine(false, on, count, &jumps[m.at..m.at + matrix], vectors, made, 0);
                    combine(false, x, count, &jumps[m.at + matrix..m.at + 2 * matrix], vectors, made, 1);
                    for f in 0..slopes {
                        let slope = &jump_slopes[f * width + m.at..][..2 * matrix];
                        combine(false, on, count, &slope[..matrix], vectors, made, 2 + 2 * f);
                        combine(false, x, count, &slope[matrix..], vectors, made, 3 + 2 * f);
                    }
                    if let Some(b) = &bent {
                        let shift = gathered(&b[1]);
                        combine(false, &shift[..count], count, &jumps[m.at + matrix..m.at + 2 * matrix], vectors, made, base);
                        for f in 0..slopes {
                            combine(false, &shift[..count], count, &jump_slopes[f * width + m.at + matrix..][..matrix], vectors, made, base + 1 + f);
                        }
                        for (i, &a) in live.iter().enumerate().filter(|_| taken) {
                            let (bend, shift) = (gathered(&b[2 + a]), gathered(&b[5 + a]));
                            combine(false, &bend[..count], count, &jumps[m.at..m.at + matrix], vectors, made, base + 1 + slopes + 2 * i);
                            combine(false, &shift[..count], count, &jumps[m.at + matrix..m.at + 2 * matrix], vectors, made, base + 2 + slopes + 2 * i);
                        }
                    }
                } else {
                    combine(false, on, count, &plan.emitted[m.at..m.at + matrix], vectors, made, 0);
                    if let (Some(b), true) = (&bent, taken) {
                        for (i, &a) in live.iter().enumerate() {
                            let bend = gathered(&b[2 + a]);
                            combine(false, &bend[..count], count, &plan.emitted[m.at..m.at + matrix], vectors, made, 1 + i);
                        }
                    }
                }
                // u: the field's component on the jumps' vector, less the Doppler-weighted one on the partner's. ∇u of
                // a folded member: the components' gradients on the same vectors, and along each direction the
                // components on the slopes' vectors.
                let weights = &mut weights[..4 * made];
                if more > 0 {
                    weights.iter_mut().for_each(|w| *w = C64::ZERO);
                }
                weights[0] = c[0];
                if m.near {
                    weights[1] = -cw[0];
                    weights[2..made].iter_mut().for_each(|w| *w = C64::ZERO);
                    if bent.is_some() {
                        weights[base] = c[0];
                    }
                }
                if taken {
                    for i in 0..3 {
                        let of = &mut weights[(1 + i) * made..(2 + i) * made];
                        of[0] = c[1 + i];
                        if m.near {
                            of[1] = -cw[1 + i];
                        }
                        for (f, direction) in directions.iter().enumerate().take(slopes) {
                            (of[2 + 2 * f], of[3 + 2 * f]) = (c[0] * direction[i], -cw[0] * direction[i]);
                        }
                        // The fourth order's: the partner on Nψ as the jumps on the state, and this axis' bent states.
                        match (bent.is_some(), m.near) {
                            (true, true) => {
                                of[base] = c[1 + i];
                                for (f, direction) in directions.iter().enumerate().take(slopes) {
                                    of[base + 1 + f] = c[0] * direction[i];
                                }
                                if let Some(at) = live.iter().position(|&a| a == i) {
                                    (of[base + 1 + slopes + 2 * at], of[base + 2 + slopes + 2 * at]) = (c[0], c[0]);
                                }
                            }
                            (true, false) => {
                                if let Some(at) = live.iter().position(|&a| a == i) {
                                    of[1 + at] = c[0];
                                }
                            }
                            _ => {}
                        }
                    }
                }
                match (taken, first == flat) {
                    (false, _) => combine(!first, &weights[..made], made, vectors, sums, 4, 0),
                    (true, true) => combine(!first, weights, made, vectors, sums, 4, 0),
                    (true, false) => {
                        combine(!first, &weights[..made], made, vectors, sums, 4, 0);
                        combine(!flat, &weights[made..], made, vectors, sums, 4, 1);
                    }
                }
                (first, flat) = (false, flat && !taken);
                work += member_work(m, taken, more, rows);
            }
            // A class the state gives nothing: no amplitude, and no gradient either -- at a node of its light a class has
            // no photon and still its gradients' second moment. Gradients no member made are nothing.
            if first {
                continue;
            }
            if with && flat {
                sums.chunks_exact_mut(8 * LANES).for_each(|vector| vector[2 * LANES..].fill(0.0));
            }
            if sums.chunks_exact(8 * LANES).all(|vector| vector[..if with { 8 * LANES } else { 2 * LANES }].iter().all(|x| *x == 0.0)) {
                continue;
            }
            each(place, class, sums, with);
        }
        work
    }

    /// The wave number |k_φ| (1/µm) of the photons of a cluster's class (module documentation, # A class's photon).
    pub(crate) fn class_wave_number(&self, cluster: usize, class: &Class) -> f64 {
        let mut frame = self.axes.zero();
        *frame.0.last_mut().expect("the frame's coordinate") = 1;
        let omega = self.omega(&frame.plus(-1, &class.beats[0])) + self.energies[self.clusters[cluster].states[0]] - self.energies[self.clusters[class.landing].states[0]];
        wave_number((omega / RAD_PER_US_PER_MHZ).abs())
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

    /// A class's jump operators' sizes over a region per axis, ‖u_a‖_F of u_a = Σ_members F L_a on the class's entries: per
    /// member at one relative phase the members' sum, each its field component (`source_field`) times its jump
    /// (`jump_form`), in one own form -- a carrier's components on its field's basis are one field's, whose sum the basis
    /// does not change --, the members at different relative phases (a kept coherence's) apart, their norms added; with
    /// the bands' radii, the field's bound times each jump's.
    fn class_bound(&self, cluster: usize, class: &Class, region: &Region) -> [Own; 3] {
        let (from, landing) = (&self.clusters[cluster], &self.clusters[class.landing]);
        let fields: Vec<Form<C64>> = class.members.iter().map(|m| self.source_field(&self.sources[m.source], region)).collect();
        std::array::from_fn(|a| {
            let mut size = Own::constant(0.0);
            let mut phases: Vec<&Beat> = class.members.iter().map(|m| &m.phase).collect();
            phases.sort();
            phases.dedup();
            for phase in phases {
                // Per entry of the class (landing place, source place) at this phase: the members' sum and its radius.
                let mut entries: Vec<((usize, usize), Form<C64>, Own)> = vec![];
                for (m, field) in class.members.iter().zip(&fields).filter(|of| of.0.phase == *phase) {
                    let source = &self.sources[m.source];
                    for &(bp, sp) in &m.entries {
                        let (jump, spread) = self.jump_form(source, a, landing.states[bp], from.states[sp], region);
                        let (term, spread) = (product(field, &jump, |f: C64, l: C64| f * l), spread.mul(field.norm()));
                        match entries.iter_mut().find(|x| x.0 == (bp, sp)) {
                            Some(x) => (x.1, x.2) = (x.1.add(term), x.2.add(spread)),
                            None => entries.push(((bp, sp), term, spread)),
                        }
                    }
                }
                let forms: Vec<Form<C64>> = entries.iter().map(|x| x.1).collect();
                let spread = entries.iter().fold(Own::constant(0.0), |sum, x| sum.add(x.2.square())).sqrt();
                size = size.add(norm_of(&forms).add(spread));
            }
            size
        })
    }

    /// A coherence's size over a region (the gates', units.rs `Model::gates`): Σ_a ‖L_a‖ ‖L'_a‖ of its two classes' jump
    /// operators, each in its own form there (`class_bound`), the largest over the source clusters it names.
    pub(crate) fn coherence_bound(&self, unit: &Unit, region: &Region) -> Own {
        let mut size: Option<Own> = None;
        for (cluster, i, j) in self.coherence_classes(unit) {
            let classes = &self.classes[cluster];
            let (a, b) = (self.class_bound(cluster, &classes[i], region), self.class_bound(cluster, &classes[j], region));
            let here = (0..3).fold(Own::constant(0.0), |sum, x| sum.add(a[x].mul(b[x])));
            size = Some(size.map_or(here, |size| size.max(here)));
        }
        size.unwrap_or(Own::constant(0.0))
    }

    /// Per class of a sample's cluster and axis, ‖L_a‖_F of its jump operators at the sample's point (units.rs `Measured`):
    /// its lit members' entries summed as the photons sum them (`photons_counted`) -- each its field component there times
    /// its jump, a carrier's with its Doppler partner.
    pub(crate) fn class_sizes(&self, at: &Measured) -> Vec<[f64; 3]> {
        let plan = &at.plan;
        let fields = self.fields(plan, &at.amplitudes, at.sample.at.kappa, 0);
        let len = plan.near.len();
        let jump = |place: usize| if place < len { at.near.first_order(place)[0] + plan.bare(place) } else { plan.constants[place - len] };
        let from = self.clusters[plan.cluster].states.len();
        plan.classes
            .iter()
            .map(|class| {
                let to = self.clusters[class.landing].states.len();
                let mut entries = vec![C64::ZERO; 3 * to * from];
                for m in &class.members {
                    let (c, cw) = m.scalar.map_or((C64::ONE, C64::ONE), |s| (fields.plain(s)[0], fields.doppler(s)[0]));
                    for &(a, bp, sp, first, second) in &m.entries {
                        entries[(a * to + bp) * from + sp] += if m.doppler { c * jump(first) - cw * jump(second) } else { c * jump(first) };
                    }
                }
                std::array::from_fn(|a| entries[a * to * from..(a + 1) * to * from].iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt())
            })
            .collect()
    }

    /// A coherence's error at a sample (units.rs `Measured`): m² √(p_i p_j), m its size at the point -- Σ_a ‖L_a‖ ‖L'_a‖ of
    /// its two classes' jump operators there (`class_sizes`) -- over its distance, the two frequencies' beat less the spread
    /// and the Doppler shift of two photons at the sample's speed (the one gate, units.rs `gate`); p the populations of the
    /// two classes' source states (SPEC §1.5). Zero where the sample's cluster emits no such pair.
    pub(crate) fn coherence_error(&self, unit: &Unit, at: &Measured) -> UnitError {
        let nothing = UnitError::NONE;
        let Unit::Coherence { first, second, .. } = unit else { return nothing };
        let Some(&(cluster, i, j)) = self.coherence_classes(unit).iter().find(|of| of.0 == at.sample.cluster) else { return nothing };
        let sizes = at.classes.get_or_init(|| self.class_sizes(at));
        let size: f64 = (0..3).map(|a| sizes[i][a] * sizes[j][a]).sum();
        let wave = 2.0 * self.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max);
        let apart = self.omega(&first.plus(-1, second)).abs() - self.spread() - wave * at.region.speed;
        let m2 = 0.5 * gate(size, apart, 0.0).saturation;
        let p = &at.populations;
        let classes = &self.classes[cluster];
        let source_weight = |class: &Class| -> f64 { class.members.iter().flat_map(|m| m.entries.iter().map(|&(_, sp)| self.clusters[cluster].states[sp])).map(|s| p[s]).fold(0.0, f64::max) };
        let weight = (source_weight(&classes[i]) * source_weight(&classes[j])).sqrt();
        UnitError { parameter: m2, population: m2 * weight, temperature: m2 * weight, ..UnitError::NONE }
    }
}

impl Model {
    /// The scattering of a state under a plan's lights at a point, per unit time (`Scattering::of` its photons made with
    /// their gradients, `photons_of`), and the complex multiply-adds that made the photons: `scattering_on` the point's
    /// own site.
    pub(crate) fn scattering_of(&self, plan: &Arc<Plan>, table: Option<&NearTable>, at: &Point, psi: &[C64]) -> (Scattering, f64) {
        let mut site = Held::<Site>::take();
        self.pointed(plan, table, at.r, &mut site);
        self.scattering_on(plan, table, Some(at.r), at.kappa, at.t, psi, &mut site, false)
    }

    /// `scattering_of` at the momentum κ and the time t on a site that holds the point (assemble.rs `Model::pointed`):
    /// one of its own, or a step's middle's (assemble.rs `Middle`), whose beams, place on the table and `laid` forms
    /// are the step's already -- what the photons read beside them is made on it (`sited`), and nothing of the point
    /// twice in a step. The photons' moments are read where the classes are made.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scattering_on(&self, plan: &Arc<Plan>, table: Option<&NearTable>, r: Option<[f64; 3]>, kappa: [f64; 3], t: f64, psi: &[C64], site: &mut Site, laid: bool) -> (Scattering, f64) {
        let made = self.sited(plan, table, r, kappa, 1, site, laid);
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let (mut out, mut sizes) = (Scattering::NONE, [0.0; 3]);
        let work = self.classes_on(plan, t, psi, Gradients::All, site, made, &mut |_, class, sums, with| {
            out.add(&mut sizes, &moments(sums, with, class.absorbed), class.states);
        });
        (out.over(sizes, norm), work)
    }

}

/// The counted complex multiply-adds of a member's part of its class's sums (`Model::classes_on`): its entries' amplitudes
/// -- and their Doppler partners' where it takes them, and where the gradients are `taken` theirs, 3(3 + 2 partner) more
/// each --, and the fourth order's `more` vectors, each a matrix on a state and its weights over the class's `rows`.
fn member_work(m: &MemberPlan, taken: bool, more: usize, rows: usize) -> f64 {
    let partner = if m.doppler { 1.0 } else { 0.0 };
    m.entries.len() as f64 * (2.0 + partner + if taken { 3.0 * (3.0 + 2.0 * partner) } else { 0.0 }) + (more * (m.entries.len() + 4 * rows)) as f64
}

impl Plan {
    /// The counted complex multiply-adds of a state's photons' site at a point (`Model::sited`) to `order`, priced without
    /// making it (solve.rs `Pricing`): the near jumps read from a table on `axes` axes whose coordinates there count
    /// `place`; the fourth order's matrices made there where `quartic` -- laid there by the site itself, its forms read
    /// whole (`Plan::quartic_made_work`) -- and none where they are the point's already; the field components.
    pub(crate) fn sited_work(&self, axes: usize, place: f64, order: usize, quartic: bool) -> f64 {
        let near = if self.near.len() == 0 { 0.0 } else { place + super::tabulate::read_work(axes, order, 2 * self.near.jumping().len()) };
        let fourth = self.fourth.as_ref().filter(|_| quartic).map_or(0.0, |fourth| self.quartic_made_work(fourth, axes, place));
        near + fourth + self.fields_work(order)
    }

    /// The counted complex multiply-adds of a state's photons' classes on a site whose jumps have `axes` slopes
    /// (`Model::classes_on`), priced without making them: the fourth order's bent states, and every lit member's part
    /// (`member_work`) of the classes the gradients ask; `live` the lab axes along which the fourth order's matrices
    /// change, which `classes_on` reads off the matrices at the point.
    pub(crate) fn classes_work(&self, axes: usize, gradients: Gradients, live: usize) -> f64 {
        let order = if gradients == Gradients::None { 0 } else { 1 };
        let bent = self.fourth.as_ref().map_or(0.0, |fourth| (2 * fourth.n * fourth.n * if order >= 1 { 4 } else { 1 }) as f64);
        let mut work = bent;
        for (place, class) in self.classes.iter().enumerate() {
            let with = match gradients {
                Gradients::All => true,
                Gradients::Of(only) if only == place => true,
                Gradients::Of(_) => continue,
                Gradients::None => false,
            };
            for m in class.members.iter().filter(|m| !m.columns.is_empty()) {
                let taken = m.folded && with;
                let slopes = if taken && m.near { axes } else { 0 };
                let more = match (&self.fourth, m.near) {
                    (None, _) => 0,
                    (Some(_), true) => 1 + slopes + if taken { 2 * live } else { 0 },
                    (Some(_), false) => if taken { live } else { 0 },
                };
                work += member_work(m, taken, more, class.rows);
            }
        }
        work
    }
}

/// Σ_rows (a b ± c d) of four parts A, B, C, D of a class's four columns laid for the lanes (`Model::classes_at`: each
/// column's real parts, then its imaginary parts, a vector of rows at a time), the sign `sign` (±1): one pass over the
/// rows, a partial sum a lane.
fn lanes<const A: usize, const B: usize, const C: usize, const D: usize>(sums: &[f64], sign: f64) -> f64 {
    let mut lanes = [0.0; LANES];
    for vector in sums.chunks_exact(8 * LANES) {
        for l in 0..LANES {
            lanes[l] = vector[A * LANES + l].mul_add(vector[B * LANES + l], (sign * vector[C * LANES + l]).mul_add(vector[D * LANES + l], lanes[l]));
        }
    }
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

/// The absorbed light's moments of a class per unit time (module documentation, # The absorbed light's momentum) from its
/// sums (`Model::classes_at`: four columns laid for the lanes, u and per lab axis i ∇_i u -- the latter read only `with`
/// gradients) and its absorbed light's wave number k_c: W, the folded part's Σ_a Im⟨u_a|∇u_a⟩, the second moment's rate
/// Σ_a Re⟨∇u_a|⊗|∇u_a⟩, and the mean a jump of the class carries. Each of the ten sums is one pass over the rows, a
/// partial sum a lane.
fn moments(sums: &[f64], with: bool, absorbed: f64) -> Moments {
    debug_assert!(sums.len() % (8 * LANES) == 0);
    // The columns' parts: 0 and 1 u's real and imaginary parts, 2 + 2i and 3 + 2i those of ∇_i u.
    let weight = lanes::<0, 0, 1, 1>(sums, 1.0);
    let (mut mean, mut second) = ([0.0; 3], [[0.0; 3]; 3]);
    if with {
        mean = [lanes::<0, 3, 1, 2>(sums, -1.0), lanes::<0, 5, 1, 4>(sums, -1.0), lanes::<0, 7, 1, 6>(sums, -1.0)];
        let [xx, xy, xz] = [lanes::<2, 2, 3, 3>(sums, 1.0), lanes::<2, 4, 3, 5>(sums, 1.0), lanes::<2, 6, 3, 7>(sums, 1.0)];
        let [yy, yz, zz] = [lanes::<4, 4, 5, 5>(sums, 1.0), lanes::<4, 6, 5, 7>(sums, 1.0), lanes::<6, 6, 7, 7>(sums, 1.0)];
        for (i, row) in [[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]].iter().enumerate() {
            for (j, both) in row.iter().enumerate() {
                second[i][j] += both;
            }
        }
    }
    // |b| against k_c as |mean|² against (k_c W)²: no root is taken within the bound. Beyond it the jump carries k_c
    // along the mean, which is not zero there; within it b itself, mean/W -- of a class without a rate the mean is
    // nothing (its sum's every term holds an amplitude), and so is what its jumps, of which there are none, carry.
    let (square, bound) = (crate::linalg::dot3(mean, mean), absorbed * weight);
    let kick = if square > bound * bound {
        let size = square.sqrt();
        mean.map(|x| absorbed * x / size)
    } else if weight > 0.0 {
        mean.map(|x| x / weight)
    } else {
        [0.0; 3]
    };
    Moments { weight, mean, second, kick }
}

impl Scattering {
    /// The scattering per unit time of a state of the norm ‖ψ‖² = `norm` from its classes' photons made with their
    /// gradients (module documentation, # The absorbed light's momentum): the force and the second moment's rate their
    /// jumps do not carry, over ‖ψ‖². An entry of 2D is summed from each class's 3n products of its gradients (n its
    /// landing cluster's states) and the carried mean's product over the weight's own 3n: `links` counts them, `size`
    /// their moduli on the diagonal.
    pub fn of(photons: &[Photon], norm: f64) -> Scattering {
        let (mut out, mut sizes) = (Scattering::NONE, [0.0; 3]);
        for photon in photons {
            out.add(&mut sizes, &photon.moments(), photon.amplitudes[0].len());
        }
        out.over(sizes, norm)
    }

    /// One class's part (`of`) in the state's norm: its moments on `states` landing states, the diagonal's terms'
    /// moduli to `sizes`.
    fn add(&mut self, sizes: &mut [f64; 3], m: &Moments, states: usize) {
        for i in 0..3 {
            self.force[i] += m.mean[i] - m.weight * m.kick[i];
            sizes[i] += m.second[i][i] + m.weight * m.kick[i] * m.kick[i];
            for j in 0..3 {
                self.diffusion[i][j] += m.second[i][j] - m.weight * m.kick[i] * m.kick[j];
            }
        }
        self.links += 9 * states + 6;
    }

    /// The classes' parts summed (`add`) over the state's norm ‖ψ‖², once: the scattering per unit time.
    fn over(mut self, sizes: [f64; 3], norm: f64) -> Scattering {
        self.force = self.force.map(|x| x / norm);
        self.diffusion = self.diffusion.map(|row| row.map(|x| x / norm));
        self.size = sizes.into_iter().fold(0.0, f64::max) / norm;
        self
    }
}

impl Photon {
    /// The class's rate W = Σ_a ‖u_a‖² in its state's norm.
    pub fn weight(&self) -> f64 {
        self.amplitudes.iter().map(|u| crate::linalg::norm_sqr(u)).sum()
    }

    /// The absorbed light's moments of the class per unit time (module documentation, # The absorbed light's momentum),
    /// of a photon made with its gradients: W, the folded part's Σ_a Im⟨u_a|∇u_a⟩, the second moment's rate Σ_a
    /// Re⟨∇u_a|⊗|∇u_a⟩, and the mean a jump of the class carries.
    pub fn moments(&self) -> Moments {
        // Its vectors as the sums they were made from (`Model::classes_at`): the same rows, the same moments.
        let n = self.amplitudes[0].len();
        let rows = (3 * n).div_ceil(LANES) * LANES;
        let mut sums = [0.0; 8 * 3 * STATES];
        let mut lay = |column: usize, a: usize, vector: &[C64]| {
            for (b, z) in vector.iter().enumerate() {
                let at = laid(a * n + b, column, 4);
                (sums[at], sums[at + LANES]) = (z.re, z.im);
            }
        };
        for a in 0..3 {
            lay(0, a, &self.amplitudes[a]);
            for i in 0..3 {
                lay(1 + i, a, &self.gradients[i][a]);
            }
        }
        let with = self.gradients.iter().flatten().any(|g| !g.is_empty());
        moments(&sums[..8 * rows], with, self.absorbed)
    }

    /// The mean a jump of the class carries (`Moments::kick`): at most one absorbed photon's momentum k_c.
    pub fn kick(&self) -> [f64; 3] {
        self.moments().kick
    }
}
