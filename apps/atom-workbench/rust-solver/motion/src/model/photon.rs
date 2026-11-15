//! Photons (SPEC §1.6): the jump operators by class, the photons a state emits, the absorbed light's two moments per unit
//! time -- of the one excited amplitude the generator's decay is made of (assemble.rs, # The excited amplitude).
//!
//! # Replicas and classes
//! The slow state's excited amplitude is one object, the fold's wave operator 𝒯 (elim.rs, # The fold; ELIMINATION v12):
//! per excited replica -- one excited frequency in the cluster's frames -- the amplitude 𝒯_ρ x its slow state drives there.
//! What makes it (`Replica`): the channels (readout.rs, # The readout) at the ground places they drive into the replica --
//! a component's bare drives (their constant response, far.rs) or a carrier's drives folded at the atom (their response
//! there, eliminate.rs) --, and the explicit excited states of its frame, whose amplitude is their own. A channel ν at the
//! ground place s drives the replica of frame frequency ρ = −ν − θ_s, an explicit state s is at ρ = −θ_s (the excited
//! states' ν_frame, common to every photon, is a global phase). Of the replica's excited states (its rows) the photons land
//! where they decay: on the landing ground state b' of frame θ_b' at the frame frequency ω̃ = ρ + θ_b', its photon's
//! frequency up to the class's common phase. A class is one frame frequency landing in one cluster (model.rs `Photon`): its
//! feeds, each a landing place and the one replica whose photons land there at that frequency. Two frequencies landing in
//! one cluster are two classes unless their coherence is kept (`Unit::Coherence`), then one, each feed at its relative
//! phase e^{i(ω̃ − ω̃₀)t}.
//!
//! # The jumps and the decay
//! A class's jump on the axis a is J_a = Σ_feeds C_a[b', :] 𝒯_ρ S⁻¹ -- the emission's rows on the feed's replica, the
//! normalized coordinates' S⁻¹ (assemble.rs, # The normalized generator) --, and the generator's decay is the same 𝒯's
//! product, −(i/2) S⁻¹ Σ_ρ 𝒯_ρ† D 𝒯_ρ S⁻¹ (D = Σ_a C_a†C_a, a merged class's cross terms at their beats): the
//! compression of the no-jump generator onto the slow subspace (1; 𝒯) has i(G − G†) = Σ L†L for any 𝒯 and any Hermitian
//! S (ELIMINATION v12), so the hazard Σ_φ Σ_a ‖J_{a,φ}ψ‖²/‖ψ‖² is the propagated norm's rate ⟨ψ|i(H − H†)|ψ⟩/‖ψ‖² by
//! construction -- at a node, at a point, from a table or exactly: the decay is made of the same 𝒯 the photons are.
//!
//! # A state's photons at a point
//! The excited amplitude ε_ρ = 𝒯_ρ S⁻¹ψ of every replica once, with its gradient ∇ε_ρ = ∇𝒯_ρ S⁻¹ψ + 𝒯_ρ ∇S⁻¹ψ
//! (assemble.rs `Model::excited_at`); a class's u on the rows a n + b' (the axis a, the landing cluster's n states b') is its
//! feeds' C_a[b', :] ε_ρ, and ∇u the same rows on ∇ε_ρ. A class's moments (# The absorbed light's momentum) are ten sums
//! over its rows.
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
//! folded amplitude's ∇u is its field components' gradients on its responses (each beam's ∇a_l = a_l (ik_l + ∇ln|a_l|))
//! and its responses' own. A jump of the class carries its conditional mean b = Σ_a Im⟨u_a|∇u_a⟩/W as far as one absorbed
//! photon's momentum goes, s b with s = min(1, k_c/|b|) (`Photon::kick`): b is the beams' wave vectors weighted, within their
//! convex hull while the weights are positive, and beyond k_c only where they interfere to both signs -- there it is no
//! photon's momentum. What the jumps do not carry is the step's (`Model::scattering_of`): the force Σ_c (1 − s_c) Σ_a
//! Im⟨u_a|∇u_a⟩ and the diffusion 2D = Σ_c [Σ_a Re⟨∇u_a|⊗|∇u_a⟩ − s_c² W_c b_c b_cᵀ], positive semidefinite by
//! Cauchy–Schwarz. A plane wave's jump carries its k and leaves nothing; a standing wave's carries nothing and leaves
//! k² R₀ sin² kz, the dipole force's fluctuations.
//!
//! An explicit state's amplitude is its own and its absorption's mean is the Ehrenfest force's (its kept drives'
//! stimulated force): what its absorbed light gives the momentum beside that mean is the force's own correlation through
//! the optical sector, the trajectory's per-state noise (optical.rs; step.rs, # The optical sector's noise), not a
//! photon's moment here -- its amplitude has no gradient (S⁻¹ holds the explicit states as they are).

use super::assemble::{laid, sized, ClassPlan, Held, Plan, Site};
use super::far::EXCITED;
use super::tabulate::NearTable;
use super::{Beat, Component, Model, Moments, Photon, Point, Scattering, Unit};
use crate::step::Emitted;
use std::sync::Arc;

use crate::linalg::{C64, LANES};
use crate::request::{GROUND, STATES};
use crate::units::{wave_number, RAD_PER_US_PER_MHZ};

/// A channel of a replica (module documentation, # Replicas and classes; readout.rs, # The readout): a component's bare
/// drives (`side` 0 the co-rotating, 1 the counter-rotating; their constant response, far.rs `Tone::bare_response`), or a
/// carrier's drives folded at the atom (eliminate.rs `Responses`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Channel {
    Bare { frequency: usize, side: usize },
    Near { carrier: usize },
}

/// An excited replica of a cluster (module documentation, # Replicas and classes): its frame frequency ρ, the channels at
/// the ground places that drive it, its explicit places, and the excited states (less GROUND) its amplitude can hold.
#[derive(Clone, Debug)]
pub(crate) struct Replica {
    pub(crate) beat: Beat,
    pub(crate) channels: Vec<(Channel, usize)>,
    pub(crate) explicit: Vec<usize>,
    pub(crate) rows: Vec<usize>,
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

/// A class of a source cluster's photons: its frame frequencies (one, or several kept coherent), its landing cluster, its
/// feeds -- each a replica, the landing place its photons land on at the class's frequency, and its relative beat in a
/// merged class (zero otherwise).
#[derive(Clone, Debug)]
pub(crate) struct Class {
    pub(crate) beats: Vec<Beat>,
    pub(crate) landing: usize,
    pub(crate) feeds: Vec<(usize, usize, Beat)>,
}

impl Model {
    /// A channel's component (module documentation): a bare one's own, a carrier's co-rotating one.
    pub(crate) fn channel_component(&self, channel: Channel) -> Component {
        match channel {
            Channel::Bare { frequency, side } => Component { frequency, counter: side == 1 },
            Channel::Near { carrier } => Component { frequency: self.carriers[carrier].frequency, counter: false },
        }
    }

    /// The excited states (less GROUND) a channel's response from the ground state b can reach on a cluster: a bare
    /// channel's where its constant response is not zero, a carrier's where its response at the atom can be
    /// (assemble.rs `Model::reached`).
    pub(crate) fn channel_rows(&self, cluster: usize, channel: Channel, b: usize) -> [bool; EXCITED] {
        match channel {
            Channel::Bare { frequency, side } => std::array::from_fn(|e| self.tones[frequency].bare_response[side].iter().any(|r| r[(e, b)] != C64::ZERO)),
            Channel::Near { carrier } => (0..self.carriers[carrier].basis.len()).fold([false; EXCITED], |mut out, k| {
                out.iter_mut().zip(self.reached(cluster, carrier, k, b)).for_each(|(o, r)| *o |= r);
                out
            }),
        }
    }

    /// The replicas of a cluster (module documentation, # Replicas and classes), every light on: per component with bare
    /// drives and per carrier, each ground place it reaches an excited state from, at ρ = −ν − θ_s; each explicit state at
    /// ρ = −θ_s; channels and explicit states of one frame frequency one replica. In the order they are first met.
    pub(crate) fn replicas(&self, cluster: usize) -> Vec<Replica> {
        let c = &self.clusters[cluster];
        let mut out: Vec<Replica> = vec![];
        let add = |out: &mut Vec<Replica>, beat: Beat, rows: [bool; EXCITED], what: Result<(Channel, usize), usize>| {
            let at = out.iter().position(|r| r.beat == beat).unwrap_or_else(|| {
                out.push(Replica { beat, channels: vec![], explicit: vec![], rows: vec![] });
                out.len() - 1
            });
            let r = &mut out[at];
            match what {
                Ok(channel) => r.channels.push(channel),
                Err(place) => r.explicit.push(place),
            }
            for (e, _) in rows.iter().enumerate().filter(|x| *x.1) {
                if !r.rows.contains(&e) {
                    r.rows.push(e);
                }
            }
        };
        let mut channels: Vec<Channel> = vec![];
        for (frequency, tone) in self.tones.iter().enumerate() {
            channels.extend((0..2).filter(|&side| tone.bare[side]).map(|side| Channel::Bare { frequency, side }));
        }
        channels.extend((0..self.carriers.len()).map(|carrier| Channel::Near { carrier }));
        for channel in channels {
            let nu = self.frequency(self.channel_component(channel)).negated();
            for (sp, &s) in c.states[..c.ground].iter().enumerate() {
                let rows = self.channel_rows(cluster, channel, s);
                if rows.iter().any(|r| *r) {
                    add(&mut out, nu.plus(-1, &c.frames[sp]), rows, Ok((channel, sp)));
                }
            }
        }
        // An explicit state's rows: its own, and those its static couplings' response reaches (assemble.rs
        // `Model::held_support`, every light on).
        let running = self.running();
        for sp in c.ground..c.states.len() {
            let reach = self.held_support(cluster, sp, &running).1;
            let rows = std::array::from_fn(|e| GROUND + e == c.states[sp] || reach[e]);
            add(&mut out, self.axes.zero().plus(-1, &c.frames[sp]), rows, Err(sp));
        }
        for r in &mut out {
            r.rows.sort();
        }
        out
    }

    /// The classes of a cluster's photons (module documentation, # Replicas and classes): of each replica, each ground state
    /// one of its rows decays to, at ω̃ = ρ + θ_b' landing in b's cluster, grouped by frame frequency and landing cluster,
    /// the kept coherences merged; in the order of their landing clusters and frame frequencies.
    pub(crate) fn classes(&self, cluster: usize) -> Vec<Class> {
        let mut out: Vec<Class> = vec![];
        for (ri, replica) in self.replicas[cluster].iter().enumerate() {
            for b in 0..GROUND {
                if !replica.rows.iter().any(|&e| (0..3).any(|a| self.emission[a][(b, GROUND + e)] != C64::ZERO)) {
                    continue;
                }
                let landing = self.membership[b];
                let lc = &self.clusters[landing];
                let bp = lc.states.iter().position(|&x| x == b).expect("a ground state is in its cluster");
                let beat = replica.beat.plus(1, &lc.frames[bp]);
                let class = match out.iter().position(|k| k.landing == landing && k.beats[0] == beat) {
                    Some(at) => at,
                    None => {
                        out.push(Class { beats: vec![beat.clone()], landing, feeds: vec![] });
                        out.len() - 1
                    }
                };
                out[class].feeds.push((ri, bp, self.axes.zero()));
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
                    merged[at].feeds.extend(class.feeds.into_iter().map(|(r, bp, _)| (r, bp, phase.clone())));
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
    /// gradients are then empty) --, and the complex multiply-adds that made them: the point's excited amplitudes (with
    /// their gradients where any class takes them; the responses from their table where one is given), and each class's
    /// rows on them. What a jump shares across its classes is in a thread's spare (assemble.rs `Held`): beyond the photons'
    /// own vectors the heap is asked for nothing.
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
        let table = self.pointed(plan, table, at.r, &mut site);
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
        let table = self.pointed(&plan, table, at.r, &mut site);
        let made = self.sited(&plan, table, Some(at.r), at.kappa, 1, &mut site, false);
        let rates = self.classes_on(&plan, at.t, psi, Gradients::None, &mut site, made, &mut |_, _, _, _| {});
        let drawn = (0..plan.classes.len()).map(|place| self.classes_on(&plan, at.t, psi, Gradients::Of(place), &mut site, 0.0, &mut |_, _, _, _| {})).fold(0.0, f64::max);
        rates + drawn
    }

    /// What `photons_of` makes, class by class to `each` as it is made (`classes_on`), on the point's own site
    /// (`Model::pointed`) made to the order the gradients ask (`sited`); and the complex multiply-adds, counted as the
    /// operations done (assemble.rs, # Work).
    fn classes_at(&self, plan: &Arc<Plan>, table: Option<&NearTable>, at: &Point, psi: &[C64], gradients: Gradients, each: &mut dyn FnMut(usize, &ClassPlan, &[f64], bool)) -> f64 {
        let mut site = Held::<Site>::take();
        let table = self.pointed(plan, table, at.r, &mut site);
        let made = self.sited(plan, table, Some(at.r), at.kappa, if gradients == Gradients::None { 0 } else { 1 }, &mut site, false);
        self.classes_on(plan, at.t, psi, gradients, &mut site, made, each)
    }

    /// What the photons of a plan's states at a point read of the point's site (assemble.rs `Site`; module
    /// documentation, # A state's photons at a point), whatever the state, made on a site that holds the point
    /// (assemble.rs `Model::pointed`: every beam's amplitude with its gradient, its place on the table) -- a site of
    /// its own, or a middle's, whose forms and responses are `laid` there already: the normalized generator's matrices
    /// (assemble.rs `Model::quartic`, the position `r` their key: the step's middle's where the thread still holds them);
    /// the responses to `order` -- 0, or 1 with their slopes: a table's, or the exact elimination's --; the plan's field
    /// components to that order at the momentum κ; and of them the excited amplitudes' operator 𝒯 of every replica
    /// (assemble.rs `Model::excited_at`). What lies within an order is the same numbers at every order. Returns the
    /// site's counted work.
    #[allow(clippy::too_many_arguments)]
    fn sited(&self, plan: &Arc<Plan>, table: Option<&NearTable>, r: Option<[f64; 3]>, kappa: [f64; 3], order: usize, site: &mut Site, laid: bool) -> f64 {
        // A site of its own is a jump's (or a reading's): its landing follows at the same point (model.rs `Model::landed`),
        // and is made with the photons' matrices; a middle's scattering asks none -- and nothing where S⁻¹ and the
        // landing are the identity (assemble.rs `Fourth::bends`).
        let fourth = plan.fourth.as_ref().filter(|fourth| fourth.bends()).map_or(0.0, |fourth| self.quartic(plan, fourth, table, r, site, laid, !laid));
        // A site of its own counts its place's coordinates here (`Model::placed`); a middle's counted them with its near
        // terms.
        let near = if laid { 0.0 } else { self.responded(plan, table, order, site) + table.and(site.place).map_or(0.0, |(_, work)| work) };
        let Site { beams, fields, .. } = site;
        self.fields_into(plan, beams, kappa, order, fields);
        let excited = self.excited_at(plan, site, order);
        near + fourth + plan.fields_work(order) + excited
    }

    /// The classes of a state's photons at the time t on a point's site (`sited`: its work `made`), class by class to
    /// `each` as each is made -- the class's place and plan, its sums (four columns laid for the lanes on the class's rows
    /// a n + b', assemble.rs `laid`: u, then per lab axis i ∇_i u, which hold nothing where the class takes no
    /// gradients), and whether it took gradients --, a class the state gives nothing left out; and the complex
    /// multiply-adds with the site's (assemble.rs, # Work). A class made with its gradients asks them of the site: it is
    /// made to first order. Module documentation, # A state's photons at a point: the excited amplitudes ε_ρ = 𝒯_ρ S⁻¹ψ
    /// of every replica once with their gradients, then each class's rows C_a[b', :] on its feeds'. The sums are the
    /// site's, a thread's spare's (`Held`), and a step's scattering (`scattering_of`) and a jump's photon (`emission_of`)
    /// read them where they are made and ask the heap for nothing.
    #[allow(clippy::too_many_arguments)]
    fn classes_on(&self, plan: &Arc<Plan>, t: f64, psi: &[C64], gradients: Gradients, site: &mut Site, made: f64, each: &mut dyn FnMut(usize, &ClassPlan, &[f64], bool)) -> f64 {
        let with_any = gradients != Gradients::None;
        let parts = if with_any { 4 } else { 1 };
        let n = psi.len();
        // The normalized state ψ̂ = S⁻¹ψ (assemble.rs, # The normalized generator) and with the gradients its three lab
        // axes' ∂S⁻¹ψ.
        let mut bent = [[C64::ZERO; STATES]; 4];
        bent[0][..n].copy_from_slice(psi);
        let mut work = made;
        if plan.fourth.as_ref().is_some_and(|fourth| fourth.bends()) {
            let size = n * n;
            crate::model::Model::quartic_read(|q| {
                let inverse = &q.data[super::assemble::BENT * size..];
                for (part, out) in bent.iter_mut().enumerate().take(parts) {
                    for (x, z) in out.iter_mut().enumerate().take(n) {
                        *z = (0..n).map(|y| inverse[part * size + x * n + y] * psi[y]).sum();
                    }
                }
            });
            work += (size * parts) as f64;
        }
        let excited = plan.amplitudes(&site.excited, &bent, parts, &mut site.member);
        work += excited;
        let Site { member: amplitudes, class: sums, .. } = site;
        let most = plan.classes.iter().map(|class| class.rows).max().unwrap_or(0);
        sized(sums, 8 * most);
        for (place, class) in plan.classes.iter().enumerate() {
            let with = match gradients {
                Gradients::All => true,
                Gradients::Of(only) if only == place => true,
                Gradients::Of(_) => continue,
                Gradients::None => false,
            };
            let rows = class.rows;
            let sums = &mut sums[..8 * rows];
            sums.fill(0.0);
            let columns = if with { 4 } else { 1 };
            // A merged class's feed at its relative beat: its phase factor once a run of its entries (a feed's are
            // consecutive), one product an entry.
            let mut turned = (0.0, C64::ONE);
            for &(row, at, phase, c) in &class.entries {
                let c = if phase == 0.0 {
                    c
                } else {
                    if turned.0 != phase {
                        turned = (phase, C64::cis(phase * t));
                    }
                    c * turned.1
                };
                for column in 0..columns {
                    let z = c * amplitudes[column * plan.amplitude_len + at];
                    let place = row + column * 2 * LANES;
                    sums[place] += z.re;
                    sums[place + LANES] += z.im;
                }
            }
            work += (class.entries.len() * columns + class.turning) as f64;
            if sums.chunks_exact(8 * LANES).all(|vector| vector[..if with { 8 * LANES } else { 2 * LANES }].iter().all(|x| *x == 0.0)) {
                continue;
            }
            each(place, class, sums, with);
        }
        work
    }

    /// The wave number |k_φ| (1/µm) of the photons of a cluster's class (module documentation, # A class's photon).
    pub(crate) fn class_wave_number(&self, cluster: usize, class: &Class) -> f64 {
        let omega = self.omega(&self.axes.frame().plus(-1, &class.beats[0])) + self.energies[self.clusters[cluster].states[0]] - self.energies[self.clusters[class.landing].states[0]];
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


}

impl Model {
    /// The scattering of a state under a plan's lights at a point, per unit time (`Scattering::of` its photons made with
    /// their gradients, `photons_of`), and the complex multiply-adds that made the photons: `scattering_on` the point's
    /// own site.
    pub(crate) fn scattering_of(&self, plan: &Arc<Plan>, table: Option<&NearTable>, at: &Point, psi: &[C64]) -> (Scattering, f64) {
        let mut site = Held::<Site>::take();
        let table = self.pointed(plan, table, at.r, &mut site);
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
        let mut read = 0.0;
        let work = self.classes_on(plan, t, psi, Gradients::All, site, made, &mut |_, class, sums, with| {
            out.add(&mut sizes, &moments(sums, with, class.absorbed), class.states);
            read += moments_work(class.rows, with);
        });
        (out.over(sizes, norm), work + read)
    }

}

/// The counted complex multiply-adds of a class's moments from its sums over its `rows` (`moments`): the rate, and with
/// the gradients the mean's three and the second moment's six -- each a sum over the rows of two parts' products.
fn moments_work(rows: usize, with: bool) -> f64 {
    ((1 + if with { 9 } else { 0 }) * 2 * rows) as f64 / crate::decide::COMPLEX
}

impl Plan {
    /// The counted complex multiply-adds of a state's photons' site at a point (`Model::sited`) to `order`, priced without
    /// making it (solve.rs `Pricing`): the responses read from a table on `axes` axes whose coordinates there count
    /// `place`; the normalized generator's matrices where they bend anything (assemble.rs `Fourth::bends`) -- on a
    /// middle's site (`laid`: its forms read with the middle) the matrices alone with their gradients, which the step's
    /// middle node makes and its scattering counts (`Plan::quartic_work`); on a site of its own, a jump's, its forms read
    /// and the matrices with the landing that follows there (`Plan::quartic_made_work`) --; the field components; the
    /// excited amplitudes' operator (`Plan::excite_work`).
    pub(crate) fn sited_work(&self, axes: usize, place: f64, order: usize, laid: bool) -> f64 {
        let near = if self.near.len() == 0 || laid { 0.0 } else { place + super::tabulate::read_work(axes, order, 2 * self.near.responding().len()) };
        let fourth = self.fourth.as_ref().filter(|fourth| fourth.bends()).map_or(0.0, |fourth| if laid { self.quartic_work(fourth, Some(axes), false) } else { self.quartic_made_work(fourth, axes, place) });
        let parts = if order >= 1 { 4 } else { 1 };
        let first = if self.fourth.as_ref().is_some_and(|fourth| fourth.held()) { self.excite_first_work(parts, axes) } else { 0.0 };
        near + fourth + self.fields_work(order) + self.excite_work(parts, axes) + first
    }

    /// The counted complex multiply-adds of a state's photons' classes on a site (`Model::classes_on`), priced without
    /// making them: the normalized state and the replicas' amplitudes, and each class the gradients ask its rows -- with
    /// all of them, a scattering's, each class's moments (`Model::scattering_on`).
    pub(crate) fn classes_work(&self, gradients: Gradients) -> f64 {
        let parts = if gradients == Gradients::None { 1 } else { 4 };
        let n = self.states;
        let bent = if self.fourth.as_ref().is_some_and(|fourth| fourth.bends()) { (n * n * parts) as f64 } else { 0.0 };
        let mut work = bent + self.amplitudes_work(parts);
        for (place, class) in self.classes.iter().enumerate() {
            let with = match gradients {
                Gradients::All => true,
                Gradients::Of(only) if only == place => true,
                Gradients::Of(_) => continue,
                Gradients::None => false,
            };
            work += (class.entries.len() * if with { 4 } else { 1 } + class.turning) as f64;
            if gradients == Gradients::All {
                work += moments_work(class.rows, with);
            }
        }
        work
    }
}

// Σ_rows (a b ± c d) of four parts A, B, C, D of a class's four columns laid for the lanes (`Model::classes_at`: each
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
