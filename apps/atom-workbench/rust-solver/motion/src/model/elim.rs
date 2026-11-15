//! The one elimination (ELIMINATION.md §2–§6 with its v2 and v3; SPEC §1.4–§1.5): the excited states the light does not
//! need explicit, folded onto the slow states at a point -- exactly in the static part, to first order in the frequency
//! offsets the motion and the explicit beats bring --, the numbers the two criteria and the model's error read, and the
//! frames.
//!
//! # The system
//! The ground states G (at rest: their frame is the lab's) and the excited states E (in the frame where they turn at
//! ν_frame, request.rs); H0 has no G–E entry and neither has any light: a dipole couples G and E alone. The light is a set
//! of classes (`Class`): one optical frequency, co- or counter-rotating, at ν_c in the excited states' frame; its beams are
//! one coherent field at the point (`Beam`: each beam's coupling E ← G at its amplitude there, with its wave vector).
//! Raman couplings of two ground states, light shifts, the photons' coherences: products of the elimination, none a
//! unit of its own.
//!
//! # P, Q and the frames
//! P is G and the explicit excited states E_x (`Slow`), each state s with its frame θ_s: ψ_s = e^{−iθ_s t} x_s. An entry
//! a ← b that turns as e^{−iΩt} in the lab turns as e^{−i(Ω + θ_b − θ_a)t} in the frames -- its residual beat; it is
//! static where that is zero. A P state's amplitude drives the excited states at a frequency of its own: a ground state s
//! through class c at f = ν_c + θ_s, an explicit state through the excited block's own static couplings at f = θ_s. Each
//! such frequency is a replica (`Replicas`): the excited states' amplitude at it is one Floquet copy, Q_f = E less the
//! explicit states of frame f (those are P at that frequency). A drive of class c into an explicit state of another frame
//! is folded at its replica like any other: the explicit state is a Q state at that frequency.
//!
//! # Groups (gauge invariance)
//! The frames are a gauge: a constant added to the frames of every state of one component of the kept links (a group,
//! `Slow::groups`; one cluster of the run) changes no physics, and no decision may change with it. So the static part is
//! a group's alone: the direct entries of two states of one group and one frame, the explicit drives of one group at
//! residual zero, and the second-order entries through a replica of one group -- a replica is a frequency of one group
//! (`Replicas::of`), its Q the excited states less that group's explicit states of that frame. What couples two groups
//! -- through replicas of the two whatever their frames, or a direct entry -- is a turning coupling of P at its beat, a
//! sector (# Sectors) folded and judged like any other on the actual spectrum (δ = Ω + λ_p − λ_q, the same in every
//! gauge: a group's constant moves its replicas' frequencies and its eigenvalues oppositely). Its eigenstates are each
//! one group's. Before, two groups at equal frames shared replicas, so that which of their couplings were static -- and
//! never judged -- depended on the frames' gauge (the tilted 1D PGC's F = 2 Raman pair of mixing 0.228, kept in one
//! gauge and hidden in another), and an explicit block took drives from both ground hyperfine levels, 6.8 GHz apart
//! (S3: one cluster of 20 states, its static generator spanning 43 411 rad/µs).
//!
//! # The fold (Bloch: exact in the static part)
//! The static generator on P in the frames is H = A + Σ_f W'_f T_f, A the direct static P entries (H0's static entries,
//! the explicit drives at residual zero, −θ on the diagonal, the explicit states' −(i/2)D), W_f (Q_f ← P) the couplings
//! into replica f and W'_f (P ← Q_f) back, and T_f (Q_f ← P) the response: the solution of the Sylvester equation
//!   T_f H − (K_f − f) T_f = W_f,   K_f = (H_E − (i/2)D) on Q_f,
//! that is, in H's eigenbasis, T_f r_p = (λ_p + f − K_f)⁻¹ W_f r_p: each eigenstate of P at its own eigenvalue, and the
//! eigenvalues self-consistent (`fold`). With it (x, T x) is an invariant subspace of the static Floquet generator and H
//! its generator (the Bloch equation; Löwdin: the λ_p are exact eigenvalues of the static generator) -- whatever the
//! saturation, where the fold exists (the iteration contracts). The bare levels' fold and the fold at the atom's levels
//! are this one function taken at a coarser point; there is no other form. The couplings of P that turn in the frames
//! are folded the same way, at their sectors (# Sectors): H = A + Σ_f W'_f T_f + Σ_Ω V_{−Ω} T′_Ω.
//!
//! # Sectors (ELIMINATION v9)
//! What turns in P's frames -- two replicas f ≠ f′ couple P to itself, W′_{f′}T_f at the beat f − f′, and a direct entry
//! of two groups or two frames turns at the frames' difference -- is a coupling V_Ω e^{−iΩt}. In the Floquet (Sambe)
//! space it couples P to a copy of itself shifted by Ω, a sector, exactly as a drive couples P to an excited replica:
//! with ψ(t) = Σ_m e^{−imΩt} φ_m, (ε + Ω)φ_1 = H_s φ_1 + V_Ω φ_0 + …, and folded by the one algebra -- the sector's
//! response T′ solves T′H − (H_s − Ω)T′ = V_Ω (`solve_sylvester`, its left factor the static part H_s, its right factor
//! H), in H's eigenbasis T′r_p = (λ_p + Ω − H_s)⁻¹V_Ω r_p, the distance δ = Ω + λ_p − λ_q the old criterion two's --,
//! self-consistent with the replicas (`fold`: H ← H_s + Σ V_{−Ω}T′_Ω). A sector is the whole of P shifted by Ω, one a
//! beat (`Sector`): H and H_s are each group's alone, so their Schur forms are the groups' and each block of two groups
//! of T′ solves its own equation; and since two groups' blocks are at one beat by the gauge alone, what the sector folds
//! back is each group's own, and every part and share is read block by block. (Laid out by pairs of groups, the tilted
//! 1D PGC's first round -- every ground state its own group -- had 2104 sectors at its prior's atoms; by beats, one a
//! beat.) By construction the fold holds the coupling's static shift V_{−Ω}T′ ≈ |a|²/δ -- first order in a/δ against
//! the light shift U when a is of its size ((a/δ)(a/U)): the Λ scheme at a/δ = 0.117 had 9.6 % of its light shift,
//! 19 % of its force and 19.5 % of its scattering (the interference 2U₂/δ, the shift's anti-Hermitian part) left out
//! where the component was averaged and 2(a/δ)² = 2.7 % booked. Where the ladder closes (the Λ: g₁(0) ↔ g₂(Ω)) the
//! fold is the exact static two-level problem; else what is left is the second harmonic, (a/δ)² of the sector's terms,
//! and the sector's own excited response, at λ + Ω and not λ, criterion one's offset.
//!
//! Folded or kept, by criterion one and the one existence rule, as a block is: where the fold does not exist (δ → 0,
//! `settle`), the object of the largest share goes up -- a link is kept; else the run, which folds no sector (its
//! static evaluation, `fold_about` about a reference), leaves out each sector's whole, its part of criterion one
//! (`Fold::sector_parts`: the light's interference through it, its own scattering, and its population -- the
//! micromotion's time mean, which the display and the next landing leave out), and the link of the largest part goes up
//! as a block does (`closure`). A kept link is the run's: static in its frames where its group's tree holds it, else a
//! tone the propagation follows -- the run's sector fold (spectral.rs), no second one. The sectors of the links the
//! model carries are made too (`Fold::carried`): their sidebands are offsets its amplitudes span (# First order in the
//! frequency offsets). At a run's state the readings take the same function on the run's own fold of the whole atom
//! (`Model::atom_fold`: every cluster a group, about its reference, its sectors made at its generator), so that the
//! couplings of two clusters are read as the prior reads them (`Model::readings`); the definition the evaluation is
//! measured against is the static part's (`Model::definition_fold`). A sector's excited amplitude emits photons at its
//! shifted frequency, a class of their own in criterion three (`Fold::photons`).
//!
//! # The definition and the route it is evaluated on (`Fold::series`)
//! The self-consistent fold above is the definition, and the reference the criteria and the errors are read against. A
//! run evaluates the same function by its series about a reference Z₀ = diag(z_y), a number a place of P: column s of
//! the fold's equation, (z_s + f − K)T[:, s] = W_s − (T E)[:, s] with E = H − Z₀, gives T = Σ_k T⁽ᵏ⁾,
//! T⁽⁰⁾[:, s] = R₀(s)W_s, T⁽ᵏ⁾[:, s] = −R₀(s)(T⁽ᵏ⁻¹⁾E)[:, s], R₀(s) = (z_s + f − K)⁻¹, and H = A + Σ_f W′_f T_f -- one
//! function, not a second form; with one reference for every place T_f = Σ_k (−1)^k R₀^{k+1} W_f E^k and
//! H = A + Σ_k (−1)^k N_k E^k, N_k = Σ_f W′_f R₀^{k+1} W_f, the resolvent's Neumann series. Its radius is the least
//! distance d of z_s + f from the excited block's spectrum over the places; its remainder after order K is within
//! q^{K+1}/(1 − q), q = ‖H − Z₀‖₂/d (`Fold::series_norm`, `Fold::series_distance`), and the order is the least whose
//! remainder is within the share (criterion one's ρ books it). The reference is each place's level in its frame with
//! the far light alone and one constant c of its cluster, the middle of the near light's scalar shift over the prior's
//! atoms: no field in it, the forms are bilinear in the near fields with coefficients of the far light's coordinates
//! alone -- a table's dimensions are the far light's. (PGC3D's F = 2 at its cooling light's coherent peak: about the
//! far levels alone q = 0.548; with the centre c, q = 0.0056, the series' order 0 within 7.9e-5 of the fold, order 1
//! within 4.0e-7. A place's own level, not its cluster's mean: R4_LIN_4's cluster of F = 1 and F = 2 ground states, 855
//! rad/µs apart in its frames, had q ≈ 1.6 about their mean -- ELIMINATION v11.)
//! The first order and the landing's overlap are of two places' resolvents, R₀(s)R₀(t)W_t (`Fold::evaluated`,
//! `Fold::landing_evaluated`): the landing's the run holds as its own forms (assemble.rs `NearLayout::landing`); the
//! generator's first order and its jumps' it holds of one resolvent alone, N₁, exact where a cluster's places have one
//! reference (eliminate.rs `Model::one_reference`) -- elsewhere `Unit::Series` is no candidate and the next order is the
//! nearest block's keep, as every order beyond the first (eliminate.rs `Model::orders_of`). Where no order reaches the share (q
//! near one) the block is made explicit by criterion one.
//!
//! # First order in the frequency offsets
//! A P amplitude whose frequency is λ_p + Δ -- the Doppler shift of a beam, k·v, an explicit beat -- drives
//! (λ_p + Δ + f − K)⁻¹ W r_p = T r_p − Δ T2 r_p + Δ² R² R(λ + Δ) W r_p, T2 the solution of T2 H − (K − f) T2 = T
//! (−∂T/∂f). The first two terms are the model's; the third is the remainder, by the resolvent's identity
//! R(z + Δ) = R − Δ R² + Δ² R² R(z + Δ), z = λ_p + f, exact at any Δ -- no series, no radius. The resolvents of one K
//! commute, so the remainder is Δ² R(z + Δ) v, v = R² W r_p; and K = H_E − (i/2)D on Q is normal where D is each line's
//! excited states' common rate times one (D = Σ C†C: the line's total decay, the same from every excited state of it)
//! and H_E holds no coupling across lines -- every scene's --, so that ‖R(z + Δ) v‖² = Σ_k |⟨e_k|v⟩|²/|z + Δ − κ_k|²
//! exactly, κ_k and e_k K's eigenvalues and eigenvectors, its Schur form's diagonal and columns: one change of v's
//! basis a replica, n_Q numbers an offset; of a block that is not normal (P's static generator, a sector's left factor)
//! the back substitution on its Schur form, exactly the same number where it is (`Schur::resolvent_norm`). Of a Schur
//! block S = Λ + N, N strictly upper, (z − S)⁻¹ = Σ_{k<m} ((z − Λ)⁻¹N)^k (z − Λ)⁻¹ (nilpotent), so ‖(z − S)⁻¹‖ ≤
//! (1 + h)/δ, 1 + h = Σ_{k<m} (‖N‖/δ)^k, δ the least distance of z from Λ (Henrici's departure from normality;
//! `SchurBlock::henrici`): the bound the series' distance takes (`Fold::series_distance`), the normal case's exact
//! number at N = 0, continuous in N -- a block of K is never judged normal or not, its rounding's N a part in 10¹⁶ of
//! 1 + h. (The bound ‖R(z + Δ)‖ ‖v‖ at every offset overstated a sideband
//! near an excited level v does not hold: R3_DOPPLER's cooling pair had eleven blocks made explicit at its prior
//! atoms.) Over the
//! frequencies a P amplitude spans, each component c at its offset δ_c with its share w_c of the amplitude's population,
//! the remainders add as the shares weigh them: the first criterion's relative size is
//!   ρ_p = Σ_c w_c δ_c² ‖R(λ_p + f + δ_c) R² W r_p‖ / ‖R W r_p‖,
//! the replicas added in squares, each with its own sum (`Fold::offsets`, `Fold::block_parts`): one product of the
//! resolvent with the response and, per component, its share and its distance from K's spectrum. At small δ it is the
//! second moment of the amplitude's spectrum over the distance; near a resonance of the folded block it grows without
//! bound, and the block is made explicit by its part as any part makes it; at a share of zero it is nothing -- continuous
//! in every coupling, whatever its offset (a far replica's beat that a tilt's ∝ sin θ coupling brings is no resonance
//! of the block).
//!
//! The offsets a P amplitude spans (`Fold::offsets`; ELIMINATION v8): the pattern a moving atom sees, ±2 k_max v (the
//! beams' Doppler shifts, at most k_max v, and their pattern's turn, |k_l − k_l'| v ≤ 2 k_max v), with the amplitude
//! whole; and every coupling the model carries as it turns, the sectors of its kept links (`Fold::carried`, # Sectors):
//! of a source eigenstate q′ its sector's amplitude on q, c = ⟨l_q|T′r_q′⟩ (a/δ to the first order), at the combined
//! frequency δ = Ω + λ_q′ − λ_q -- the same in every gauge, of the actual spectrum -- with the share min(1, |c|²) of the
//! source's population: that frequency's spectral weight in the state, the part of the response the fold reads at the
//! shifted frequency, and so the part of its remainder there that is the state's (a rate's relative error ρ on a
//! population share w is w ρ); at |δ| ≤ a the mixing is whole and the share one. A sector the model does not carry is
//! no sideband of its amplitude: the run leaves it out, and its whole is its part (# Sectors; read as a sideband,
//! R3_DOPPLER's ground hyperfine beat of 42 943 rad/µs at a mixing of 5.9·10⁻⁴ made blocks explicit, the first-order
//! offset's extrapolation applied where the model applies none). Criterion one reads ρ_p of every ground-like
//! eigenstate (`closure`), the error at a state the same at the state's populations: the main line at its eigenstate's,
//! each sideband at its source's times its share (`errors`). A loop's weak closing edge whose frame beat is B is a
//! sideband of a/B on its eigenstates: its term is (a/B)² B² = a², what the coupling's own first order moves the
//! amplitude by. (The frames' residual -- Karp's least largest, a bound of every residual whatever its size -- was read
//! before, the whole amplitude at it: it depended on the frames, and the frames now hold the strongest couplings static,
//! cluster.rs `frames`, the weakest turning at a loop's whole miss.)
//!
//! # The criteria's numbers (`closure`, `errors`; the photons' on the run's photon classes, `Model::readings`)
//! - The remainder ρ_p of each P eigenstate at the offsets its amplitude spans (# First order in the frequency offsets),
//!   the replicas added in squares (different frequencies), with each excited state's part of it: the excited block that
//!   leaves the most is the one a partition makes explicit; and with them in squares the whole of each sector the run
//!   leaves out, its parts on its links (# Sectors, `Fold::sector_parts`): the link that leaves the most is kept
//!   (criterion one; what was criterion two, the components' 2(a/δ)², is the sectors' population and the shift it left
//!   out). ρ is an amplitude's relative error, and its error on what the outputs are is the rate's, 2ρ + ρ² (`rate`: a
//!   rate Γ‖e‖² of an amplitude e off by ρ moves by at most that, a shift by ρ; the populations are the rates' ratios and
//!   T is set by the rates and the shifts' gradients) -- one number, which the closure judges and the readings book, by
//!   one function (`Fold::criterion_one`): an error that is a rate's or a population's already is in ρ's units the
//!   amplitude's whose rate's error it is (`amplitude`).
//! - The photons' cross terms (criterion three, `cross_terms`): two classes of photons land in one cluster at two
//!   frequencies, and a split into classes drops J_a ρ J_b†. After a jump at t_j its part |r_q⟩⟨r_q′| on the landing's
//!   eigenstates, from the source eigenstates p and p′, turns in t_j at the combined frequency
//!   ω_c = Ω − (λ_q − λ_q′*) + (λ_p − λ_p′*) -- Ω the two classes' frequencies' difference, ω_c the two photons' own,
//!   the same in every gauge --, and of its size M it moves the rates by M/|ω_c| (the secular approximation, first order:
//!   kept above the share). Ω alone is the photons' difference only where the frames join the states: two ground states
//!   the frames do not join, their Raman resonance at ω_c → 0 however far apart Ω puts their classes (PGC1D_TILT_2's
//!   prior atoms: |Ω| 26.4 rad/µs, |ω_c| down to 0.11). A sector's photons are classes of their own at their shifted
//!   frequency (`Fold::photons`): a carried sector's of the run's class of its target, a pair's object its coherence; a
//!   left-out sector's, photons the run has none of, a pair's object the sector's links (`pair_links`). Read at the
//!   start on the closure's fold (`start_cross_terms`, `Model::prior`) and at the run's states on its photon classes and
//!   the atom's fold (`Model::readings`), by one function.
use crate::linalg::{schur, Eigen, Mat, C64};
use crate::units::rounding;

/// The atom's levels at a point (rad/µs): the ground block in the lab frame, the excited block in the excited states' frame,
/// and the decay D = Σ_a C_a†C_a on the excited states (C_a the emission, ground ← excited).
#[derive(Clone, Debug)]
pub(crate) struct Levels {
    pub ground: Mat,
    pub excited: Mat,
    pub decay: Mat,
}

/// One beam of a class at the point: its coupling (excited ← ground, its amplitude there in it) and its wave vector
/// (1/µm; a counter-rotating class's beam takes −k).
#[derive(Clone, Debug)]
pub(crate) struct Beam {
    pub coupling: Mat,
    pub wave: [f64; 3],
}

/// One class of the light: an optical frequency, co- or counter-rotating, ν (rad/µs, in the excited states' frame), its
/// beams one coherent field.
#[derive(Clone, Debug)]
pub(crate) struct Class {
    pub nu: f64,
    pub beams: Vec<Beam>,
}

/// P beside the ground states: the explicit excited states (indices into E), the frame θ_s of every P state (rad/µs;
/// the ground states first, then the explicit ones in their order), and its group: the component of the kept links it
/// lies in (module documentation, # Groups; one cluster's states are one group).
#[derive(Clone, Debug)]
pub(crate) struct Slow {
    pub explicit: Vec<usize>,
    pub frames: Vec<f64>,
    /// Per frame the size of the terms its value is the sum of (`Sized`): what its rounding is of (`equal`).
    pub sizes: Vec<f64>,
    pub groups: Vec<usize>,
}

/// A frequency with the size of the terms it is the sum of (`equal`): the frames of the closure, sums of the kept links'
/// beats along their tree (cluster.rs `frames`), each a difference of two classes' frequencies -- a frame of 26 rad/µs
/// made of the cooling and the repumping lights' 4.1·10⁴ is rounded at the 4.1·10⁴'s, not at its own.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sized {
    pub value: f64,
    pub size: f64,
}

impl super::cluster::Exact for Sized {
    fn plus(&self, s: i64, other: &Sized) -> Sized {
        Sized { value: self.value + s as f64 * other.value, size: self.size + s.unsigned_abs() as f64 * other.size }
    }
}

/// The replicas (module documentation, # P, Q and the frames; # Groups): their frequencies f and groups, the replica
/// each class drives each ground state's amplitude into, and the one each explicit state's static couplings drive. Made
/// of exact frequencies by the caller; `Replicas::of` groups equal values of one group.
#[derive(Clone, Debug)]
pub(crate) struct Replicas {
    pub frequencies: Vec<f64>,
    /// Per replica the size of the terms its frequency is the sum of (a class's and a frame's, `Slow::sizes`; `equal`).
    pub scales: Vec<f64>,
    pub groups: Vec<usize>,
    /// [class][ground state].
    pub driven: Vec<Vec<usize>>,
    /// [explicit state].
    pub held: Vec<usize>,
}

/// Whether a sum of frequencies is nothing: within γ_STATES of the size `scale` of the terms it is the sum of -- a class's
/// frequency and frames, each itself a sum of at most a component per state along the frames' paths (cluster.rs
/// `frames`; `Sized`, `Slow::sizes`, `Replicas::scales`, `Beat::size`). Its own size is no measure: two replicas'
/// frequencies of 26 rad/µs, each the cooling light's and a frame of the repumping light's beat 4.1·10⁴ apart, are
/// rounded at the 4.1·10⁴'s -- read at their own size, PGC1D_TILT_2's held a sector between two replicas that are one,
/// at a beat of 1.8·10⁻¹², and a coupling of a state with itself across it. Two physically different frequencies
/// differ by a beat of the request's frequency axes, many orders more.
pub(crate) fn equal(sum: f64, scale: f64) -> bool {
    sum.abs() <= rounding(crate::request::STATES) * scale
}

impl Replicas {
    /// The replicas of frequencies that are one (`equal`, at the sizes of their terms) within one group.
    pub(crate) fn of(classes: &[Class], slow: &Slow, ground: usize) -> Replicas {
        let (mut frequencies, mut scales, mut groups): (Vec<f64>, Vec<f64>, Vec<usize>) = (vec![], vec![], vec![]);
        let mut place = |f: f64, scale: f64, group: usize| match (0..frequencies.len()).find(|&x| groups[x] == group && equal(frequencies[x] - f, scales[x] + scale)) {
            Some(at) => at,
            None => {
                frequencies.push(f);
                scales.push(scale);
                groups.push(group);
                frequencies.len() - 1
            }
        };
        let driven = classes.iter().map(|c| (0..ground).map(|g| place(c.nu + slow.frames[g], c.nu.abs() + slow.sizes[g], slow.groups[g])).collect()).collect();
        let held = (0..slow.explicit.len()).map(|i| place(slow.frames[ground + i], slow.sizes[ground + i], slow.groups[ground + i])).collect();
        Replicas { frequencies, scales, groups, driven, held }
    }
}

/// What drives one replica: a class's beam from the ground states (`class` Some), or the excited block's static
/// couplings from the explicit states of its frame (`class` None); its coupling W (Q_f ← P), the coupling back W'
/// (P ← Q_f), and its coupling back from every excited state.
#[derive(Clone, Debug)]
pub(crate) struct Feed {
    pub class: Option<usize>,
    pub w: Mat,
    pub back: Mat,
    /// The coupling back from every excited state (P ← E): what the feed's light does to an excited amplitude at another
    /// replica (a P-layer component's emitted leg).
    pub emitted: Mat,
}

/// One replica folded: its frequency, Q_f, the sum of its feeds' W and W', its response T (module documentation).
#[derive(Clone, Debug)]
pub(crate) struct Folded {
    pub frequency: f64,
    /// The size of the terms its frequency is the sum of (`Replicas::scales`).
    pub scale: f64,
    /// Its group (an index into the fold's `members`, # Groups): its feeds' places, where its equation is solved.
    pub group: usize,
    pub q: Vec<usize>,
    pub feeds: Vec<Feed>,
    pub w: Mat,
    pub back: Mat,
    pub t: Mat,
    /// Per class of its feeds (None: the explicit states' own couplings): the class's W, its coupling back from every
    /// excited state (P ← E, its feeds' `emitted`), and its part of T (T = Σ_c T_c) -- what a turning coupling of P
    /// through this replica is made of (`Sectoring`).
    classes: Vec<(Option<usize>, Mat, Mat, Mat)>,
    /// The Schur form of K_f = (H_E − (i/2)D) on Q_f.
    k: Schur,
}

impl Folded {
    /// Which blocks of K_f's Schur form the replica's series holds (`Fold::series_distance`, `Fold::nearest`), in the
    /// form's order: every block of a state its coupling W drives, and every block the forms `joins` (on E, E's indices:
    /// the excited block's part the reference leaves out, δK; none, K_f's blocks alone) join to one held, until none
    /// more. R₀ = (z₀ + f − K_f)⁻¹ is block-diagonal on K_f's blocks, so each term R₀((z − z₀) + δK)R₀ ⋯ R₀W of the
    /// series of (z − K_f − δK)⁻¹W is zero on every other block: its spectrum is in no term, and no part of the
    /// series' radius.
    fn held_blocks(&self, joins: Option<&Mat>) -> Vec<bool> {
        let blocks = &self.k.blocks;
        let mut of = vec![0; self.q.len()];
        for (x, b) in blocks.iter().enumerate() {
            for &s in &b.states {
                of[s] = x;
            }
        }
        let mut held: Vec<bool> = blocks.iter().map(|b| b.states.iter().any(|&s| (0..self.w.cols).any(|c| self.w[(s, c)] != C64::ZERO))).collect();
        if let Some(j) = joins {
            let n = self.q.len();
            let pairs: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).filter(|&(a, b)| of[a] != of[b] && (j[(self.q[a], self.q[b])] != C64::ZERO || j[(self.q[b], self.q[a])] != C64::ZERO)).collect();
            while let Some(&(a, b)) = pairs.iter().find(|&&(a, b)| held[of[a]] != held[of[b]]) {
                held[of[a]] = true;
                held[of[b]] = true;
            }
        }
        held
    }
}

/// A sector of P (module documentation, # Sectors): the whole of P shifted by a beat Ω of its turning couplings, a Sambe
/// copy of it (ELIMINATION v9), folded as a replica is -- V_Ω the couplings into it (∝ e^{−iΩt}), V_{−Ω} those back, and
/// its response T′ of T′H − (H_s − Ω)T′ = V_Ω, H_s the static generator (A and the replicas' fold; `solve`): in H's
/// eigenbasis T′r_p = (λ_p + Ω − H_s)⁻¹V_Ω r_p, its distance δ = Ω + λ_p − λ_q. V_Ω, V_{−Ω} and T′ are on P's places,
/// every group's entries at the beat: H and H_s are each group's alone, so their Schur forms are the groups' (`joined`)
/// and T′'s block of two groups (to, from) solves its own equation, T′_{to,from}H_from − (H_s,to − Ω)T′_{to,from} =
/// V_{to,from} -- one equation a beat, its blocks apart by construction. Two groups' blocks at one beat are one sector by
/// the frames' gauge alone (a group's constant moves its beats), so nothing of the sector joins two of its blocks: what
/// it folds back is each group's own (`within`), and every part, share and existence is read block by block (`blocks`:
/// the pairs of groups its entries lie in, indices into the fold's `members`). V's entries are kept by the link each is
/// (`Link`: its two states and its two classes) with its row and column (P's places) and value.
#[derive(Clone, Debug)]
pub(crate) struct Sector {
    pub frequency: f64,
    /// The sizes of the frequencies its beat is the difference of (`equal`).
    pub scale: f64,
    pub blocks: Vec<(usize, usize)>,
    pub v: Mat,
    pub back: Mat,
    pub links: Vec<(Link, usize, usize, C64)>,
    pub t: Mat,
}

/// m += c b on the rows and columns `rows`, `columns` of m (b the block there).
fn add_block(m: &mut Mat, rows: &[usize], columns: &[usize], c: C64, b: &Mat) {
    for (i, &x) in rows.iter().enumerate() {
        for (j, &y) in columns.iter().enumerate() {
            m[(x, y)] += c * b[(i, j)];
        }
    }
}

/// m += c b on each group's own block (`members`), nothing between two groups (`Sector`: a group's own part of a
/// product through the sector).
fn within(m: &mut Mat, c: C64, b: &Mat, members: &[Vec<usize>]) {
    for places in members {
        add_block(m, places, places, c, &b.block(places, places));
    }
}

/// The Schur form of a matrix that is each group's alone (`members`), from its groups' forms `parts` (`schurs_of`): their
/// blocks one after another on P's places -- block-diagonal, each block upper triangular; no decomposition.
fn joined(parts: &[Schur], members: &[Vec<usize>]) -> Schur {
    let n: usize = members.iter().map(|m| m.len()).sum();
    let (mut blocks, mut dense, mut at) = (vec![], vec![C64::ZERO; n * n], 0);
    for (part, places) in parts.iter().zip(members) {
        for b in &part.blocks {
            blocks.push(SchurBlock { states: b.states.iter().map(|&x| places[x]).collect(), start: at + b.start, s: b.s.clone(), q: b.q.clone(), departure: b.departure });
        }
        for x in 0..part.n {
            for y in 0..part.n {
                dense[(at + x) * n + at + y] = part.s[x * part.n + y];
            }
        }
        at += part.n;
    }
    Schur { n, blocks, s: dense }
}

impl Sector {
    /// T′ at the generator H and the static generator H_s, their Schur forms on P (`joined`): the Sylvester equation of
    /// every block at once (`solve_sylvester`: left H_s, right H, at the shift −Ω).
    fn solve(&mut self, still: &Schur, h: &Schur) {
        self.t = solve_sylvester(still, C64::real(-self.frequency), h, C64::ZERO, &self.v);
    }
    /// The scale of what it folds back, Σ over its blocks of ‖V_{−Ω}‖‖T′‖ there.
    fn scale_of(&self, members: &[Vec<usize>]) -> f64 {
        let norm = |m: &Mat, rows: &[usize], columns: &[usize]| rows.iter().flat_map(|&x| columns.iter().map(move |&y| m[(x, y)].norm_sqr())).sum::<f64>().sqrt();
        self.blocks.iter().map(|&(to, from)| norm(&self.back, &members[from], &members[to]) * norm(&self.t, &members[to], &members[from])).sum()
    }
    /// Its links of the block of the places `to` ← `from`.
    fn links_in<'a>(&'a self, to: &'a [usize], from: &'a [usize]) -> impl Iterator<Item = &'a (Link, usize, usize, C64)> + 'a {
        self.links.iter().filter(move |l| to.contains(&l.1) && from.contains(&l.2))
    }
}

/// A fold's left wave operators (`Fold::left_operators`): per replica S_f (P ← Q_f).
#[derive(Clone, Debug)]
pub(crate) struct Left {
    pub replicas: Vec<Mat>,
}

/// The turning couplings of P made of the replicas' responses (module documentation, # Sectors) and the direct entries
/// that turn (`direct`: row, column, residual beat, value), laid out once a fold as sectors (`Sectoring::make` fills them
/// at each step of its iteration) -- of the links `carried` says are the model's (`which` true) or of the others
/// (false): the entry x ← y of the light of class c′ emitted from replica j's excited amplitude that class c absorbed at
/// replica i, Σ_e C_{c′}[x, e]† T_{i,c}[e, y], at the beat f_i − f_j; each sector the entries of one beat (`equal`), its
/// V_{−Ω} the entries of the opposite beat.
struct Sectoring {
    /// Per sector its beat and the sizes of the frequencies the beat is the difference of, its links (link, row, column:
    /// P's places) and the pairs of groups (to, from) they lie in.
    keys: Vec<(f64, f64)>,
    links: Vec<Vec<(Link, usize, usize)>>,
    blocks: Vec<Vec<(usize, usize)>>,
    /// Per replica i and class of it, per replica j and class of it: the rows x of j's group with their couplings on
    /// Q_i (place on Q_i, C_{c′}[x, e]†), and per row the columns y of i's group with their sector and link (P's places).
    pairs: Vec<(usize, usize, Vec<(usize, Vec<(usize, C64)>, Vec<(usize, usize, usize)>)>)>,
    /// The turning direct entries: (row, column, value, sector, link), P's places.
    direct: Vec<(usize, usize, C64, usize, usize)>,
    /// Per sector the one of the opposite beat, if any; P's size.
    backs: Vec<Option<usize>>,
    n: usize,
}

impl Sectoring {
    #[allow(clippy::too_many_arguments)]
    fn new(folded: &[Folded], direct: &[(usize, usize, f64, C64, f64)], members: &[Vec<usize>], place_group: &[usize], label: &dyn Fn(usize) -> usize, carried: &dyn Fn(&Link) -> bool, which: bool) -> Sectoring {
        let (mut keys, mut links, mut blocks): (Vec<(f64, f64)>, Vec<Vec<(Link, usize, usize)>>, Vec<Vec<(usize, usize)>>) = (vec![], vec![], vec![]);
        let mut place = |x: usize, y: usize, omega: f64, scale: f64, link: Link| -> Option<(usize, usize)> {
            if carried(&link) != which {
                return None;
            }
            let s = match keys.iter().position(|k| equal(k.0 - omega, k.1 + scale)) {
                Some(s) => s,
                None => {
                    keys.push((omega, scale));
                    links.push(vec![]);
                    blocks.push(vec![]);
                    keys.len() - 1
                }
            };
            let block = (place_group[x], place_group[y]);
            if !blocks[s].contains(&block) {
                blocks[s].push(block);
            }
            let l = match links[s].iter().position(|l| l.0 == link && l.1 == x && l.2 == y) {
                Some(l) => l,
                None => {
                    links[s].push((link, x, y));
                    links[s].len() - 1
                }
            };
            Some((s, l))
        };
        let mut pairs = vec![];
        for (i, ri) in folded.iter().enumerate() {
            for (ci, (absorbed, ..)) in ri.classes.iter().enumerate() {
                for rj in folded.iter().enumerate().filter(|&(j, _)| j != i).map(|(_, r)| r) {
                    let (omega, scale) = (ri.frequency - rj.frequency, ri.scale + rj.scale);
                    for (emitted, _, coupling, _) in &rj.classes {
                        let mut rows = vec![];
                        for &x in &members[rj.group] {
                            let row: Vec<(usize, C64)> = ri.q.iter().enumerate().map(|(a, &e)| (a, coupling[(x, e)])).filter(|z| z.1 != C64::ZERO).collect();
                            if row.is_empty() {
                                continue;
                            }
                            let columns: Vec<(usize, usize, usize)> = members[ri.group].iter().filter_map(|&y| place(x, y, omega, scale, Link { a: label(x), b: label(y), classes: (*emitted, *absorbed) }).map(|(s, l)| (y, s, l))).collect();
                            if !columns.is_empty() {
                                rows.push((x, row, columns));
                            }
                        }
                        if !rows.is_empty() {
                            pairs.push((i, ci, rows));
                        }
                    }
                }
            }
        }
        let direct = direct.iter().filter_map(|&(x, y, beat, value, scale)| place(x, y, beat, scale, Link { a: label(x), b: label(y), classes: (None, None) }).map(|(s, l)| (x, y, value, s, l))).collect();
        let backs = keys.iter().map(|k| keys.iter().position(|o| equal(o.0 + k.0, o.1 + k.1))).collect();
        Sectoring { keys, links, blocks, pairs, direct, backs, n: place_group.len() }
    }

    /// The sectors at the replicas' responses as they stand (`Folded::classes`), their responses not yet solved.
    fn make(&self, folded: &[Folded]) -> Vec<Sector> {
        let n = self.n;
        let mut out: Vec<Sector> = self
            .keys
            .iter()
            .zip(&self.links)
            .zip(&self.blocks)
            .map(|((&(frequency, scale), links), blocks)| Sector { frequency, scale, blocks: blocks.clone(), v: Mat::zeros(n, n), back: Mat::zeros(n, n), links: links.iter().map(|&(l, x, y)| (l, x, y, C64::ZERO)).collect(), t: Mat::zeros(n, n) })
            .collect();
        for (i, ci, rows) in &self.pairs {
            let part = &folded[*i].classes[*ci].3;
            for (x, row, columns) in rows {
                for &(y, s, l) in columns {
                    let value = row.iter().map(|&(a, z)| z * part[(a, y)]).sum::<C64>();
                    out[s].v[(*x, y)] += value;
                    out[s].links[l].3 += value;
                }
            }
        }
        for &(x, y, value, s, l) in &self.direct {
            out[s].v[(x, y)] += value;
            out[s].links[l].3 += value;
        }
        let backs: Vec<Mat> = self.backs.iter().map(|b| b.map_or_else(|| Mat::zeros(n, n), |b| out[b].v.clone())).collect();
        for (s, back) in out.iter_mut().zip(backs) {
            s.back = back;
        }
        out
    }
}

/// The fold at a point (module documentation): P's static generator H in the frames, the replicas, and H's
/// eigendecomposition (the eigenvalues λ_p, the right eigenvectors r_p as unit columns, the left ones as the inverse's
/// rows). `exists` is whether the self-consistent iteration contracted to its rounding: where it does not, the fold does
/// not exist at this point (a resonance the partition has to make explicit).
#[derive(Clone, Debug)]
pub(crate) struct Fold {
    pub ground: usize,
    pub explicit: Vec<usize>,
    pub frames: Vec<f64>,
    /// The sizes of the frames' terms (`Slow::sizes`).
    pub sizes: Vec<f64>,
    /// A: the direct static entries.
    pub direct: Mat,
    pub generator: Mat,
    pub replicas: Vec<Folded>,
    /// The sectors of the turning couplings the model does not carry (module documentation, # Sectors): folded into the
    /// generator where the fold is self-consistent; about a reference -- the run's own evaluation, which folds none --
    /// made at its generator, what the run leaves out. And those it carries (its kept links that turn in the frames,
    /// which the run propagates as tones), made alike and folded into nothing -- the offsets they put into the
    /// amplitude's spectrum (`offsets`).
    pub sectors: Vec<Sector>,
    pub carried: Vec<Sector>,
    pub eigen: Eigen,
    /// Whether the fold exists, and whether its static part does (the replicas' alone; `fold`): where only the sectors'
    /// fold does not, the generator is the static part's and the sectors are made at it -- what says which link goes up.
    pub exists: bool,
    pub still_exists: bool,
    /// The reference z₀ of each P place (its group's) where the fold is its series' order 0 about it (`fold`'s
    /// `reference`), none where it is the self-consistent fold.
    pub reference: Option<Vec<f64>>,
    /// Per class and ground state the replica (an index into `replicas`) its drive feeds.
    driven: Vec<Vec<Option<usize>>>,
    /// The groups' places (# Groups), each eigenstate's group, and each place's.
    pub members: Vec<Vec<usize>>,
    pub groups: Vec<usize>,
    places: Vec<usize>,
    /// Where sectors are made, the Schur form of H_s on each group's places (`Sector`; none else).
    stills: Vec<Schur>,
}

/// A Schur form A = Q S Q†, block by block (`Schur::of`): per block of the states A's entries join, its states, its
/// place in the form's order, its own S (upper triangular, row by row) and Q (column by column, on its states), and its
/// departure from normality ‖N‖_F, N its S's strictly upper part (`SchurBlock::henrici`); the whole S row by row in
/// the form's order (block-diagonal).
#[derive(Clone, Debug)]
struct Schur {
    n: usize,
    blocks: Vec<SchurBlock>,
    s: Vec<C64>,
}

#[derive(Clone, Debug)]
struct SchurBlock {
    states: Vec<usize>,
    start: usize,
    s: Vec<C64>,
    q: Vec<C64>,
    departure: f64,
}

impl SchurBlock {
    fn of(states: Vec<usize>, start: usize, s: Vec<C64>, q: Vec<C64>) -> SchurBlock {
        let m = states.len();
        let departure = (0..m).flat_map(|i| (i + 1..m).map(move |j| (i, j))).map(|(i, j)| s[i * m + j].norm_sqr()).sum::<f64>().sqrt();
        SchurBlock { states, start, s, q, departure }
    }
    /// At z, the least distance δ of z from its diagonal Λ and Henrici's factor 1 + h = Σ_{k<m} (ν/δ)^k of its departure
    /// from normality ν (module documentation, # First order in the frequency offsets): ‖(z − S)⁻¹‖ ≤ (1 + h)/δ and
    /// ‖(z − S)⁻¹y‖ ≤ (1 + h)‖(z − Λ)⁻¹y‖, the Neumann series of (z − Λ)⁻¹N, nilpotent -- 1 where N = 0, continuously in N.
    fn henrici(&self, z: C64) -> (f64, f64) {
        let m = self.states.len();
        let delta = (0..m).map(|i| (z - self.s[i * m + i]).abs()).fold(f64::INFINITY, f64::min);
        let x = self.departure / delta;
        let factor = (0..m).fold((0.0, 1.0), |(sum, power), _| (sum + power, power * x)).0;
        (delta, factor)
    }
}

impl Schur {
    /// The Schur form of a square matrix, block by block: the states its entries join (the connected components of its
    /// pattern, a ↔ b where either entry is not zero) are decomposed apart (linalg.rs `schur`) and laid one after another
    /// -- block-diagonal and each block upper triangular is upper triangular, Q block-diagonal on the blocks' states is
    /// unitary --: exactly the whole one's form, its work the blocks' (the excited block's hyperfine and magnetic
    /// sublevels that nothing couples), and so every product with it.
    fn of(a: &Mat) -> Option<Schur> {
        let n = a.rows;
        let mut block: Vec<usize> = (0..n).collect();
        for i in 0..n {
            for j in 0..n {
                if a[(i, j)] != C64::ZERO {
                    let (x, y) = (block[i], block[j]);
                    if x != y {
                        block.iter_mut().filter(|z| **z == y).for_each(|z| *z = x);
                    }
                }
            }
        }
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| (block[i], i));
        let (mut blocks, mut dense) = (vec![], vec![C64::ZERO; n * n]);
        let mut at = 0;
        while at < n {
            let end = (at..n).find(|&k| block[order[k]] != block[order[at]]).unwrap_or(n);
            let states = order[at..end].to_vec();
            let m = states.len();
            let mut h: Vec<C64> = (0..m * m).map(|x| a[(states[x / m], states[x % m])]).collect();
            let (mut q, mut v, mut sums) = (vec![C64::ZERO; m * m], vec![C64::ZERO; m], vec![C64::ZERO; m]);
            schur(&mut h, &mut q, &mut v, &mut sums, m)?;
            for x in 0..m {
                for y in 0..m {
                    dense[(at + x) * n + at + y] = h[x * m + y];
                }
            }
            blocks.push(SchurBlock::of(states, at, h, q));
            at = end;
        }
        Some(Schur { n, blocks, s: dense })
    }
    /// The Schur form of the adjoint from this one: A† = Q S† Q†, S† lower triangular, so with each block's columns of Q
    /// in reversed order it is upper triangular -- no decomposition.
    fn adjoint(&self) -> Schur {
        let n = self.n;
        let mut dense = vec![C64::ZERO; n * n];
        let blocks = self
            .blocks
            .iter()
            .map(|b| {
                let m = b.states.len();
                let s: Vec<C64> = (0..m * m).map(|x| b.s[(m - 1 - x % m) * m + (m - 1 - x / m)].conj()).collect();
                let q: Vec<C64> = (0..m * m).map(|x| b.q[(m - 1 - x / m) * m + x % m]).collect();
                for x in 0..m {
                    for y in 0..m {
                        dense[(b.start + x) * n + b.start + y] = s[x * m + y];
                    }
                }
                SchurBlock::of(b.states.clone(), b.start, s, q)
            })
            .collect();
        Schur { n, blocks, s: dense }
    }
    /// Q† x (x with n rows), into the form's order.
    fn left_in(&self, x: &Mat) -> Mat {
        let mut out = Mat::zeros(self.n, x.cols);
        for b in &self.blocks {
            let m = b.states.len();
            for k in 0..m {
                for (i, &state) in b.states.iter().enumerate() {
                    let c = b.q[k * m + i].conj();
                    if c != C64::ZERO {
                        for j in 0..x.cols {
                            let v = x.data[state * x.cols + j];
                            out.data[(b.start + k) * x.cols + j] += c * v;
                        }
                    }
                }
            }
        }
        out
    }
    /// Q y.
    fn left_out(&self, y: &Mat) -> Mat {
        let mut out = Mat::zeros(self.n, y.cols);
        for b in &self.blocks {
            let m = b.states.len();
            for (i, &state) in b.states.iter().enumerate() {
                for k in 0..m {
                    let c = b.q[k * m + i];
                    if c != C64::ZERO {
                        for j in 0..y.cols {
                            let v = y.data[(b.start + k) * y.cols + j];
                            out.data[state * y.cols + j] += c * v;
                        }
                    }
                }
            }
        }
        out
    }
    /// x Q (x with n columns), into the form's order.
    fn right_in(&self, x: &Mat) -> Mat {
        let mut out = Mat::zeros(x.rows, self.n);
        for b in &self.blocks {
            let m = b.states.len();
            for r in 0..x.rows {
                for k in 0..m {
                    out.data[r * self.n + b.start + k] = b.states.iter().enumerate().map(|(i, &state)| x.data[r * x.cols + state] * b.q[k * m + i]).sum();
                }
            }
        }
        out
    }
    /// y Q†.
    fn right_out(&self, y: &Mat) -> Mat {
        let mut out = Mat::zeros(y.rows, self.n);
        for b in &self.blocks {
            let m = b.states.len();
            for r in 0..y.rows {
                for (i, &state) in b.states.iter().enumerate() {
                    out.data[r * self.n + state] = (0..m).map(|k| y.data[r * y.cols + b.start + k] * b.q[k * m + i].conj()).sum();
                }
            }
        }
        out
    }
    /// (σ − S)⁻¹ y in place (back substitution, block by block).
    /// An entry of nothing on its right is nothing: the block it is in is not driven -- a sector's block of two groups
    /// its couplings do not join is solved with the others' (`Sector::solve`), and at the sector fold's first step H is
    /// H_s, so the empty block of a group with itself at the beat zero would be 0/0.
    fn shifted_solve(&self, sigma: C64, y: &mut [C64]) {
        for b in &self.blocks {
            let m = b.states.len();
            for row in (0..m).rev() {
                let mut acc = y[b.start + row];
                for k in row + 1..m {
                    acc += b.s[row * m + k] * y[b.start + k];
                }
                y[b.start + row] = if acc == C64::ZERO { C64::ZERO } else { acc / (sigma - b.s[row * m + row]) };
            }
        }
    }
    /// ‖(z − A)⁻¹ v‖ exactly, `y` = Q†v in the form's order (`left_in`): the back substitution on each block (Q unitary,
    /// `shifted_solve`) -- where A is normal it is (Σ_k |⟨e_k|v⟩|²/|z − κ_k|²)^{1/2} (module documentation, # First
    /// order in the frequency offsets), continuously in its blocks' departures. (Henrici's bound (1 + h)‖(z − Λ)⁻¹y‖
    /// it was read by grows as (‖N‖/δ)^{m−1}: on the tilted 1D PGC's P of eight ground states, non-normal, a sector's
    /// growth at the moving atom's offsets read 10⁷ where the solve's is 13.)
    fn resolvent_norm(&self, z: C64, y: &[C64]) -> f64 {
        let mut x = y.to_vec();
        self.shifted_solve(z, &mut x);
        crate::linalg::norm_sqr(&x).sqrt()
    }
    /// (σ − A)⁻¹ v.
    fn resolvent(&self, sigma: C64, v: &[C64]) -> Vec<C64> {
        let column = Mat { rows: self.n, cols: 1, data: v.to_vec() };
        let mut y = self.left_in(&column).data;
        self.shifted_solve(sigma, &mut y);
        self.left_out(&Mat { rows: self.n, cols: 1, data: y }).data
    }
}

/// T (H + r) − (L + l) T = W for the Schur forms L = Q_L S_L Q_L† and H = Q_H S_H Q_H† and scalars l, r
/// (Bartels–Stewart): Y = Q_L† T Q_H solves Y (S_H + r) − (S_L + l) Y = Q_L† W Q_H column by column,
/// ((S_H)_jj + r − l − S_L) Y_j = F_j − Σ_{i<j} (S_H)_ij Y_i.
fn solve_sylvester(left: &Schur, l: C64, right: &Schur, r: C64, w: &Mat) -> Mat {
    if left.n == 0 {
        return Mat::zeros(0, w.cols);
    }
    let fq = right.right_in(&left.left_in(w));
    let n = right.n;
    let mut y = Mat::zeros(left.n, n);
    let mut column = vec![C64::ZERO; left.n];
    for j in 0..n {
        for m in 0..left.n {
            let mut acc = fq.data[m * n + j];
            for i in 0..j {
                let c = right.s[i * n + j];
                if c != C64::ZERO {
                    acc -= c * y.data[m * n + i];
                }
            }
            column[m] = acc;
        }
        left.shifted_solve(right.s[j * n + j] + r - l, &mut column);
        for m in 0..left.n {
            y.data[m * n + j] = column[m];
        }
    }
    left.left_out(&right.right_out(&y))
}

/// A replica's Sylvester equation T H − (K − f) T = W (module documentation, # The fold) on its own group's places
/// `places` (# Groups): H's block there
/// (its Schur form `h`), W's columns there; T is nothing on the other groups' places -- no coupling of the replica
/// reaches them (by construction: a replica is a group's), so they are not solved for.
fn sylvester_in(k: &Schur, h: &Schur, f: f64, w: &Mat, places: &[usize]) -> Mat {
    solve_in(k, C64::real(-f), h, C64::ZERO, w, places)
}

/// `solve_sylvester` on a group's places `places` (`sylvester_in`): the right factor's Schur form `right` the group's
/// block, W's columns there, the solution nothing elsewhere.
fn solve_in(left: &Schur, l: C64, right: &Schur, r: C64, w: &Mat, places: &[usize]) -> Mat {
    let rows: Vec<usize> = (0..w.rows).collect();
    let part = solve_sylvester(left, l, right, r, &w.block(&rows, places));
    let mut out = Mat::zeros(w.rows, w.cols);
    for i in 0..w.rows {
        for (j, &x) in places.iter().enumerate() {
            out[(i, x)] = part[(i, j)];
        }
    }
    out
}

/// Per group (`members`) the Schur form of a matrix's block on its places.
fn schurs_of(m: &Mat, members: &[Vec<usize>]) -> Option<Vec<Schur>> {
    members.iter().map(|places| Schur::of(&m.block(places, places))).collect()
}

fn difference(a: &Mat, b: &Mat) -> f64 {
    a.data.iter().zip(&b.data).map(|(x, y)| (*x - *y).norm_sqr()).sum::<f64>().sqrt()
}

/// The one existence statement of every expansion the model takes (ELIMINATION unify, item 1): an expansion in a small
/// quantity ε about a centre -- a resolvent's or a binomial's power series, or the self-consistent fold's fixed point --
/// exists over the range the run needs where its ratio ρ, ‖ε‖ times the growth of its terms (the resolvent's norm, the
/// coefficients' radius), is below one at every point of that range: its remainder after the order K is then within
/// ρ^{K+1}/(1 − ρ) of its scale. The range is where the expansion is evaluated: the reach's atoms and every state read
/// for one taken at the atom, the whole light (every beam at its peak, in phase) for a constant one (the bare levels').
/// Where an expansion does not exist -- its ratio at or beyond one, or no number -- the object it folds goes up one
/// level: a drive from the bare levels to the atom's, a fold to explicit (the closure's move, `closure`, `Series::moved`;
/// at a run's state its error is booked whole on that object, and the keep rule moves it); where there is none above,
/// the request is outside the model's domain (model.rs `Model::settled`).
pub(crate) fn converges(ratio: f64) -> bool {
    ratio < 1.0
}

/// A bare fold's error at its mixing m at a point (`Model::bare_mixing`): where the fold exists there, 2m < 1
/// (`converges`), the population it leaves out -- the fold holds the drive's static shift, |a|²/d, and not its sector's
/// metric (module documentation, # Sectors): the micromotion's time mean m² and what the next landing puts on the other
/// Floquet branch, m², a sector's 2⟨X′⟩ (ELIMINATION v9); else its whole.
pub(crate) fn bare_error(mixing: f64) -> f64 {
    if converges(2.0 * mixing) { 2.0 * mixing * mixing } else { 1.0 }
}

/// The fixed point of x ← step(x) from x₀ (a step gives its value and the scale of its terms), and whether it exists.
/// The iteration's factor at the start is the derivative of the fold in its argument at the unperturbed spectra,
/// W'R²W, the excited share m² of the states it folds (for one line exactly |T|²), read as the iteration's first
/// contraction κ -- the change of the second step over the first's --; Stewart's theorem has the invariant subspace, and
/// the iteration contracting to it, where m = ‖W‖/sep < ½ (ELIMINATION.md §3): the ratio is 2m = 2√κ (`converges`).
/// At one the fold is not this point's -- a slow and a fast mode come together (a two-level atom on resonance at s = ½,
/// its exceptional point, starts at exactly κ = ¼) -- and the partition makes the excited block explicit. Where it
/// exists the iteration goes on to its floor: until a change is within the rounding of a step's operations (`terms`)
/// times the scale, or no longer falls.
fn settle(start: &Mat, mut step: impl FnMut(&Mat) -> (Mat, f64), terms: usize) -> (Mat, bool) {
    let (mut x, mut last, mut iterations, mut first) = (start.clone(), f64::INFINITY, 0, None);
    loop {
        iterations += 1;
        let (next, scale) = step(&x);
        let change = difference(&next, &x);
        if !change.is_finite() {
            // A change that is no number is no fixed point (`converges`); the last iterate that is one is kept.
            return (x, false);
        }
        x = next;
        if change <= rounding(terms) * scale {
            return (x, true);
        }
        match first {
            None => first = Some(change),
            Some(f) if iterations == 2 && !converges(2.0 * (change / f).sqrt()) => return (x, false),
            _ => {}
        }
        if !(change < last) {
            // At its floor.
            return (x, true);
        }
        last = change;
    }
}

/// The fold at a point (module documentation, # The fold): the replicas' couplings, then the self-consistent
/// iteration H ← A + Σ_f W'_f T_f(H) + Σ_Ω V_{−Ω} T′_Ω(H) from H = A (`settle`: `Fold::exists` false where it does not
/// contract within the fold's existence bound) -- or, about a `reference` (each place's z₀, its group's), its series'
/// order 0: one step from H = z₀, T_f = R₀ W_f and H = A + Σ_f W'_f R₀ W_f (module documentation, # The definition and
/// the route it is evaluated on; the run's own evaluation, what its states are read on, which folds no sector).
/// `carried`, where sectors are made (module documentation, # Sectors): which turning links the model carries -- the
/// others are the sectors folded into the self-consistent fold, or, about a reference, made at its generator (what the
/// run leaves out), the carried ones made at its end, their sidebands' (`Fold::offsets`); none: no sector, the static
/// fold alone (what the run's series is taken about, `Model::series_at`).
pub(crate) fn fold(levels: &Levels, classes: &[Class], slow: &Slow, replicas: &Replicas, carried: Option<&dyn Fn(&Link) -> bool>) -> Fold {
    fold_about(levels, classes, slow, replicas, None, carried)
}

/// `fold`, about a reference where one is given.
pub(crate) fn fold_about(levels: &Levels, classes: &[Class], slow: &Slow, replicas: &Replicas, reference: Option<&[f64]>, carried: Option<&dyn Fn(&Link) -> bool>) -> Fold {
    let ground = levels.ground.rows;
    let nx = slow.explicit.len();
    let n = ground + nx;
    let excited = levels.excited.rows;
    let k_full = {
        let mut k = levels.excited.clone();
        k.add_scaled(C64::new(0.0, -0.5), &levels.decay);
        k
    };
    let frame = |s: usize| slow.frames[s];
    let size = |s: usize| slow.sizes[s];
    let together = |x: usize, y: usize| slow.groups[x] == slow.groups[y];
    // The groups' places, in the order they first come.
    let mut members: Vec<Vec<usize>> = vec![];
    for s in 0..n {
        match members.iter_mut().find(|m| slow.groups[m[0]] == slow.groups[s]) {
            Some(m) => m.push(s),
            None => members.push(vec![s]),
        }
    }
    let group_of = |label: usize| members.iter().position(|m| slow.groups[m[0]] == label).expect("a group's places");
    // A: the direct static entries of a group; the turning ones and those of two groups apart (# Groups).
    let mut a = Mat::zeros(n, n);
    let mut turning = vec![];
    let mut put = |a: &mut Mat, row: usize, column: usize, value: C64| {
        if value == C64::ZERO {
            return;
        }
        if together(row, column) && equal(frame(column) - frame(row), size(column) + size(row)) {
            a[(row, column)] += value;
        } else {
            turning.push((row, column, frame(column) - frame(row), value, size(column) + size(row)));
        }
    };
    for r in 0..ground {
        for c in 0..ground {
            put(&mut a, r, c, levels.ground[(r, c)]);
        }
    }
    for (i, &x) in slow.explicit.iter().enumerate() {
        for (j, &y) in slow.explicit.iter().enumerate() {
            put(&mut a, ground + i, ground + j, k_full[(x, y)]);
        }
    }
    for s in 0..n {
        a[(s, s)] -= C64::real(frame(s));
    }
    // The explicit drives of a group at residual zero are A's; the others are folded at their replica (an explicit
    // state of another frame or another group is Q there).
    for class in classes {
        for beam in &class.beams {
            for (i, &x) in slow.explicit.iter().enumerate() {
                for g in 0..ground {
                    let v = beam.coupling[(x, g)];
                    if v != C64::ZERO && together(g, ground + i) && equal(class.nu + frame(g) - frame(ground + i), class.nu.abs() + size(g) + size(ground + i)) {
                        a[(ground + i, g)] += v;
                        a[(g, ground + i)] += v.conj();
                    }
                }
            }
        }
    }
    // The replicas: Q_f and the feeds; one Schur form per distinct Q (K_f is K on Q_f, whatever f).
    let mut forms: Vec<(Vec<usize>, Schur)> = vec![];
    let mut folded: Vec<Folded> = replicas
        .frequencies
        .iter()
        .zip(&replicas.groups)
        .zip(&replicas.scales)
        .map(|((&f, &group), &scale)| {
            let q: Vec<usize> = (0..excited).filter(|&e| !slow.explicit.iter().enumerate().any(|(i, &x)| x == e && slow.groups[ground + i] == group && equal(frame(ground + i) - f, size(ground + i) + scale))).collect();
            if !forms.iter().any(|x| x.0 == q) {
                let k = Schur::of(&k_full.block(&q, &q)).expect("the Schur form of the excited block");
                forms.push((q.clone(), k));
            }
            let (_, k) = forms.iter().find(|x| x.0 == q).expect("its form");
            Folded { frequency: f, scale, group: group_of(group), q, feeds: vec![], w: Mat::zeros(0, 0), back: Mat::zeros(0, 0), t: Mat::zeros(0, 0), classes: vec![], k: k.clone() }
        })
        .collect();
    for (c, class) in classes.iter().enumerate() {
        for beam in &class.beams {
            for (r, rep) in folded.iter_mut().enumerate() {
                let columns: Vec<usize> = (0..ground).filter(|&g| replicas.driven[c][g] == r).collect();
                if columns.is_empty() {
                    continue;
                }
                let w = Mat::from_fn(rep.q.len(), n, |i, s| if columns.contains(&s) { beam.coupling[(rep.q[i], s)] } else { C64::ZERO });
                if w.max_abs() == 0.0 {
                    continue;
                }
                let back = w.adjoint();
                let emitted = Mat::from_fn(n, excited, |s, e| if columns.contains(&s) { beam.coupling[(e, s)].conj() } else { C64::ZERO });
                rep.feeds.push(Feed { class: Some(c), w, back, emitted });
            }
        }
    }
    for (r, rep) in folded.iter_mut().enumerate() {
        let columns: Vec<usize> = (0..nx).filter(|&i| replicas.held[i] == r).collect();
        if columns.is_empty() {
            continue;
        }
        let w = Mat::from_fn(rep.q.len(), n, |i, s| if s >= ground && columns.contains(&(s - ground)) { k_full[(rep.q[i], slow.explicit[s - ground])] } else { C64::ZERO });
        let back = Mat::from_fn(n, rep.q.len(), |s, i| if s >= ground && columns.contains(&(s - ground)) { k_full[(slow.explicit[s - ground], rep.q[i])] } else { C64::ZERO });
        if w.max_abs() == 0.0 && back.max_abs() == 0.0 {
            continue;
        }
        let emitted = Mat::from_fn(n, excited, |s, e| if s >= ground && columns.contains(&(s - ground)) { k_full[(slow.explicit[s - ground], e)] } else { C64::ZERO });
        rep.feeds.push(Feed { class: None, w, back, emitted });
    }
    let kept: Vec<usize> = (0..folded.len()).filter(|&r| !folded[r].feeds.is_empty()).collect();
    folded.retain(|rep| !rep.feeds.is_empty());
    let driven: Vec<Vec<Option<usize>>> = replicas.driven.iter().map(|row| row.iter().map(|&r| kept.iter().position(|&k| k == r)).collect()).collect();
    for rep in folded.iter_mut() {
        let (mut w, mut back) = (Mat::zeros(rep.q.len(), n), Mat::zeros(n, rep.q.len()));
        for feed in &rep.feeds {
            w.add_scaled(C64::ONE, &feed.w);
            back.add_scaled(C64::ONE, &feed.back);
            match rep.classes.iter_mut().find(|x| x.0 == feed.class) {
                Some(x) => {
                    x.1.add_scaled(C64::ONE, &feed.w);
                    x.2.add_scaled(C64::ONE, &feed.emitted);
                }
                None => rep.classes.push((feed.class, feed.w.clone(), feed.emitted.clone(), Mat::zeros(0, 0))),
            }
        }
        rep.w = w;
        rep.back = back;
    }
    // The sectors (module documentation, # Sectors): a place's group, its link label (a ground state's index, an explicit
    // state's ground + its E index), and which turning links the model carries.
    let place_group: Vec<usize> = (0..n).map(|s| group_of(slow.groups[s])).collect();
    let label = |s: usize| if s < ground { s } else { ground + slow.explicit[s - ground] };
    let folds_sectors = carried.is_some() && reference.is_none();
    let carries = |l: &Link| carried.is_some_and(|c| c(l) || c(&l.reversed()));
    let (folding, carrying) = match carried {
        Some(_) => (Some(Sectoring::new(&folded, &turning, &members, &place_group, &label, &carries, false)), Some(Sectoring::new(&folded, &turning, &members, &place_group, &label, &carries, true))),
        None => (None, None),
    };
    // A replica's response; where sectors are made, by class (`Folded::classes`), T = Σ_c T_c.
    let respond = |rep: &mut Folded, sh: &[Schur]| {
        if carried.is_none() {
            rep.t = sylvester_in(&rep.k, &sh[rep.group], rep.frequency, &rep.w, &members[rep.group]);
            return;
        }
        let mut t = Mat::zeros(rep.q.len(), n);
        for part in rep.classes.iter_mut() {
            part.3 = sylvester_in(&rep.k, &sh[rep.group], rep.frequency, &part.1, &members[rep.group]);
            t.add_scaled(C64::ONE, &part.3);
        }
        rep.t = t;
    };
    let mut sectors: Vec<Sector> = vec![];
    // The self-consistent iteration: H ← A + Σ_f W'_f T_f(H) + Σ_Ω V_{−Ω} T′_Ω(H) -- the sectors' couplings made of the
    // replicas' responses at H, their left factor the static part H_s at H --, to the rounding of a Sylvester solution's
    // entries (sums of n_P n_Q products) times the terms' scale; the fold exists where it gets there, contracting.
    let mut step = |h: &Mat, with: bool| {
        let sh = schurs_of(h, &members).expect("the Schur form of P's generator");
        let (mut next, mut scale) = (a.clone(), a.frobenius());
        for rep in folded.iter_mut() {
            respond(rep, &sh);
            next.add_scaled(C64::ONE, &rep.back.product(&rep.t));
            scale += rep.back.frobenius() * rep.t.frobenius();
        }
        if with {
            let still = joined(&schurs_of(&next, &members).expect("the Schur form of P's static generator"), &members);
            let whole = joined(&sh, &members);
            sectors = folding.as_ref().expect("the sectors' layout").make(&folded);
            for s in sectors.iter_mut() {
                s.solve(&still, &whole);
                within(&mut next, C64::ONE, &s.back.product(&s.t), &members);
                scale += s.scale_of(&members);
            }
        }
        (next, scale)
    };
    // The static part's fold first, then the sectors' on it (each its own existence: the object whose fold does not exist
    // goes up, a block where the static part's does not, a link where the sectors' does not -- `closure`); where the
    // sectors' does not, the generator is the static part's, the sectors made at it.
    let terms = (n + excited) * (n + excited);
    let (h, still_exists, exists) = match reference {
        None => {
            let (h, settled) = settle(&a, |h| step(h, false), terms);
            match folds_sectors && settled {
                true => match settle(&h, |h| step(h, true), terms) {
                    (whole, true) => (whole, true, true),
                    _ => (h, true, false),
                },
                false => (h, settled, settled),
            }
        }
        Some(z0) => (step(&Mat::from_fn(n, n, |i, j| if i == j { C64::real(z0[i]) } else { C64::ZERO }), false).0, true, true),
    };
    // The responses at the generator they made it of, at its final form (about a reference: R₀ W, as made), and the static
    // part H_s they make there.
    let sh = schurs_of(&h, &members).expect("the Schur form of P's generator");
    if reference.is_none() {
        for rep in folded.iter_mut() {
            respond(rep, &sh);
        }
    }
    let mut still = a.clone();
    for rep in &folded {
        still.add_scaled(C64::ONE, &rep.back.product(&rep.t));
    }
    // The sectors at the end: folded into the generator where it is self-consistent, about a reference made at it -- what
    // the run, which folds none, leaves out --, and the carried ones, made alike.
    let (mut carried_sectors, mut stills) = (vec![], vec![]);
    if carried.is_some() {
        stills = schurs_of(&still, &members).expect("the Schur form of P's static generator");
        let (still, whole) = (joined(&stills, &members), joined(&sh, &members));
        sectors = folding.as_ref().expect("the sectors' layout").make(&folded);
        sectors.iter_mut().for_each(|s| s.solve(&still, &whole));
        carried_sectors = carrying.as_ref().expect("the sectors' layout").make(&folded);
        carried_sectors.iter_mut().for_each(|s| s.solve(&still, &whole));
    }
    let (eigen, groups) = eigen_of(&h, &members);
    Fold { ground, explicit: slow.explicit.clone(), frames: slow.frames.clone(), sizes: slow.sizes.clone(), direct: a, generator: h, replicas: folded, sectors, carried: carried_sectors, eigen, exists, still_exists, reference: reference.map(|z| z.to_vec()), driven, members, groups, places: place_group, stills }
}

/// H's eigendecomposition and each eigenstate's group: H is a group's alone (# Groups), so its eigenstates are each one
/// group's, decomposed group by group -- two groups' equal eigenvalues mix no eigenvector of one with the other's.
fn eigen_of(h: &Mat, members: &[Vec<usize>]) -> (Eigen, Vec<usize>) {
    let n = h.rows;
    let mut eigen = Eigen::default();
    let mut groups = vec![0; n];
    if members.len() == 1 {
        eigen.decompose(h);
    } else {
        let (mut values, mut vectors, mut inverse) = (vec![C64::ZERO; n], Mat::zeros(n, n), Mat::zeros(n, n));
        let (mut condition, mut work) = (0.0_f64, 0.0);
        let mut next = 0;
        for (g, m) in members.iter().enumerate() {
            let mut part = Eigen::default();
            part.decompose(&h.block(m, m));
            for (k, _) in m.iter().enumerate() {
                values[next + k] = part.values[k];
                groups[next + k] = g;
                for (i, &x) in m.iter().enumerate() {
                    vectors[(x, next + k)] = part.vectors[(i, k)];
                    inverse[(next + k, x)] = part.inverse[(k, i)];
                }
            }
            (condition, work, next) = (condition.max(part.condition), work + part.work, next + m.len());
        }
        (eigen.values, eigen.vectors, eigen.inverse, eigen.condition, eigen.work) = (values, vectors, inverse, condition, work);
    }
    (eigen, groups)
}

impl Fold {
    /// P's size.
    pub(crate) fn size(&self) -> usize {
        self.generator.rows
    }
    /// The right eigenvector r_p (unit) of H.
    pub(crate) fn right(&self, p: usize) -> Vec<C64> {
        (0..self.size()).map(|i| self.eigen.vectors[(i, p)]).collect()
    }
    /// Its ground weight ‖P_G r_p‖².
    pub(crate) fn ground_weight(&self, p: usize) -> f64 {
        (0..self.ground).map(|i| self.eigen.vectors[(i, p)].norm_sqr()).sum()
    }
    /// The P state an eigenstate holds most of.
    pub(crate) fn dominant(&self, p: usize) -> usize {
        (0..self.size()).max_by(|&a, &b| self.eigen.vectors[(a, p)].norm_sqr().total_cmp(&self.eigen.vectors[(b, p)].norm_sqr())).unwrap_or(0)
    }

    /// The offsets eigenstate p's amplitude spans, per replica (module documentation, # First order in the frequency
    /// offsets): Σ_c w_c δ_c² ‖R(λ_p + f + δ_c) v‖/‖v‖, v = R² W r_p, at each replica of p's group
    /// (`Schur::resolvent_norm`; none at the others)
    /// -- `pattern` (2 k_max v) at the share `main`, the larger of its two signs, and the sidebands the model's own turning
    /// couplings put into p (the carried sectors, # Sectors): of each source eigenstate p′ of a carried sector into p's
    /// group, its amplitude on p, c = ⟨l_p|T′r_p′⟩ (a/δ to the first order), at the combined frequency Ω + λ_p′ − λ_p's
    /// real part and the share `source(p′)` min(1, |c|²).
    pub(crate) fn offsets(&self, p: usize, pattern: f64, main: f64, source: &dyn Fn(usize) -> f64) -> Vec<f64> {
        let n = self.size();
        let lambda = self.eigen.values[p];
        let mut into: Vec<(f64, f64)> = vec![];
        for s in &self.carried {
            for &(_, from) in s.blocks.iter().filter(|b| b.0 == self.groups[p]) {
                for q in (0..n).filter(|&q| self.groups[q] == from && source(q) > 0.0) {
                    let t = self.sector_apply(s, &self.right(q));
                    let c: C64 = (0..n).map(|i| self.eigen.inverse[(p, i)] * t[i]).sum();
                    let share = source(q) * c.norm_sqr().min(1.0);
                    if share > 0.0 {
                        into.push((share, (C64::real(s.frequency) + self.eigen.values[q] - lambda).re));
                    }
                }
            }
        }
        if !(main > 0.0) && into.is_empty() {
            return vec![0.0; self.replicas.len()];
        }
        self.replicas
            .iter()
            .map(|rep| {
                if rep.group != self.groups[p] {
                    return 0.0;
                }
                // The remainder's vector v = R(z)² W r_p, in the Schur form's basis (`Schur::resolvent_norm`).
                let z = lambda + C64::real(rep.frequency);
                let v = rep.k.resolvent(z, &rep.k.resolvent(z, &rep.w.apply(&self.right(p))));
                let size = v.iter().map(|x| x.norm_sqr()).sum::<f64>().sqrt();
                if size == 0.0 {
                    return 0.0;
                }
                let y = rep.k.left_in(&Mat { rows: rep.k.n, cols: 1, data: v }).data;
                let term = |delta: f64| delta * delta * rep.k.resolvent_norm(z + C64::real(delta), &y) / size;
                let main = if main > 0.0 { main * term(pattern).max(term(-pattern)) } else { 0.0 };
                main + into.iter().map(|&(share, delta)| share * term(delta)).sum::<f64>()
            })
            .collect()
    }

    /// Replica j's response to a P state of the eigen-coefficients `coefficients` (each eigenstate at its own eigenvalue,
    /// T x = Σ_p c_p R(λ_p + f) W r_p), feed by feed: per feed its class (None: the explicit states' own couplings) and
    /// its part R(λ_p + f) W_feed r_p summed on Q_f.
    pub(crate) fn feed_responses(&self, j: usize, coefficients: &[C64]) -> Vec<(Option<usize>, Vec<C64>)> {
        let rep = &self.replicas[j];
        rep.feeds
            .iter()
            .map(|feed| {
                let mut v = vec![C64::ZERO; rep.q.len()];
                for (p, &c) in coefficients.iter().enumerate().filter(|&(p, _)| self.groups[p] == rep.group) {
                    let x = rep.k.resolvent(self.eigen.values[p] + C64::real(rep.frequency), &feed.w.apply(&self.right(p)));
                    v.iter_mut().zip(&x).for_each(|(a, b)| *a += c * *b);
                }
                (feed.class, v)
            })
            .collect()
    }

    /// T′x of a sector (x and T′x on P's places; module documentation, # Sectors).
    pub(crate) fn sector_apply(&self, s: &Sector, x: &[C64]) -> Vec<C64> {
        s.t.apply(x)
    }

    /// A sector's blocks from eigenstate p's group (`Sector::blocks`): each target group, with its places.
    fn targets<'a>(&'a self, s: &'a Sector, p: usize) -> impl Iterator<Item = (usize, &'a [usize])> + 'a {
        let from = self.groups[p];
        s.blocks.iter().filter(move |b| b.1 == from).map(move |&(to, _)| (to, &self.members[to][..]))
    }

    /// Of a sector's block from eigenstate p's group into the group `to` (its places `places`), each link's weight in
    /// T′r_p, |V[x, y] r_p[y]|² ‖R′(λ_p + Ω) e_x‖² (each entry's own resolvent column, R′ = (z − H_s)⁻¹ on the group).
    fn link_weights(&self, s: &Sector, p: usize, to: usize, places: &[usize]) -> Vec<(Link, f64)> {
        let (r, z) = (self.right(p), self.eigen.values[p] + C64::real(s.frequency));
        let from = &self.members[self.groups[p]];
        s.links_in(places, from).map(|&(link, x, y, value)| (link, (value * r[y]).norm_sqr() * crate::linalg::norm_sqr(&self.stills[to].resolvent(z, &places.iter().map(|&k| if k == x { C64::ONE } else { C64::ZERO }).collect::<Vec<_>>())))).collect()
    }

    /// The folded sectors' part of criterion one at eigenstate p (module documentation, # Sectors), per link, in the
    /// units of ρ_p² (`block_parts`): the run evaluates no sector -- what it carries of them it carries as tones --, so
    /// each one's whole is what it leaves out of p. Block by block of the sector (`Sector`: a target group's), its
    /// excited amplitude e_f′ = T_f′ T′r_p at each replica f′ of the target group (each target eigenstate at its own
    /// frequency: at λ_p + Ω to the first order in the sector's offset, criterion one's), against p's own excited
    /// response ‖R W r_p‖²:
    /// - where λ_p + Ω + f′ is a frequency of p's own replicas (Ω + f′ = f, `same`), e_f′ adds to p's own amplitude
    ///   there -- the light's interference through the sector, its rates' and shifts' first order (the Λ scheme's
    ///   2U₂/δ) --: a response's relative error, ρ² ⊃ ‖e‖²/‖R W r_p‖², as a block's remainder;
    /// - elsewhere it scatters at a frequency of its own: a rate's relative error x = ‖e‖²/‖R W r_p‖² apart, in the
    ///   units of ρ the amplitude's error whose rate's error it is (`amplitude`, `rate`), ρ² ⊃ (√(1 + x) − 1)²;
    /// - and its population ‖T′r_p‖², the time mean of the micromotion, which the display and the next landing each
    ///   leave out: the populations' error 2⟨X′⟩, ρ² ⊃ (√(1 + 2‖T′r_p‖²) − 1)².
    /// Each grows at the offsets the moving atom spans as the sector's response does, g = max(1, ‖R′(z ± pattern)Vr_p‖ /
    /// ‖R′(z)Vr_p‖), by Henrici's bound (`Schur::resolvent_norm`, # First order in the frequency offsets): the
    /// amplitudes by g, the rates and populations by g². A block's part goes to its links by their shares of T′r_p
    /// (`link_weights`).
    pub(crate) fn sector_parts(&self, p: usize, pattern: f64) -> Vec<(Link, f64)> {
        let (r, lambda) = (self.right(p), self.eigen.values[p]);
        let own: f64 = self.replicas.iter().filter(|rep| rep.group == self.groups[p]).map(|rep| crate::linalg::norm_sqr(&rep.k.resolvent(lambda + C64::real(rep.frequency), &rep.w.apply(&r)))).sum();
        let mut out: Vec<(Link, f64)> = vec![];
        for s in &self.sectors {
            let (t, v) = (self.sector_apply(s, &r), s.v.apply(&r));
            let z = lambda + C64::real(s.frequency);
            for (to, places) in self.targets(s, p) {
                let local = |x: &[C64]| places.iter().map(|&i| x[i]).collect::<Vec<C64>>();
                let population = crate::linalg::norm_sqr(&local(&t));
                if !(population > 0.0) {
                    continue;
                }
                let (mut coherent, mut apart) = (0.0, 0.0);
                for rep in self.replicas.iter().filter(|rep| rep.group == to) {
                    let e = crate::linalg::norm_sqr(&rep.t.apply(&t));
                    match self.replicas.iter().any(|own| own.group == self.groups[p] && equal(own.frequency - s.frequency - rep.frequency, own.scale + s.scale + rep.scale)) {
                        true => coherent += e,
                        false => apart += e,
                    }
                }
                let k = &self.stills[to];
                let y = k.left_in(&Mat { rows: places.len(), cols: 1, data: local(&v) }).data;
                let at = k.resolvent_norm(z, &y);
                let g = [pattern, -pattern].iter().map(|&d| k.resolvent_norm(z + C64::real(d), &y) / at).fold(1.0_f64, f64::max);
                let rates = if own > 0.0 { g * g * coherent / own + amplitude(g * g * apart / own).powi(2) } else { 0.0 };
                let rho = rates + amplitude(2.0 * g * g * population).powi(2);
                let weights = self.link_weights(s, p, to, places);
                let total: f64 = weights.iter().map(|w| w.1).sum();
                if total > 0.0 {
                    out.extend(weights.into_iter().map(|(link, w)| (link, rho * w / total)));
                }
            }
        }
        out
    }

    /// Each folded sector's link's first-order mixing at eigenstate p, m = |V[x, y] r_p[y]| ‖R′(λ_p + Ω) e_x‖ (its own
    /// response's norm, a/|δ| of one entry; `link_weights`): where 2m ≥ 1 its own expansion does not exist
    /// (`converges`; `closure`).
    pub(crate) fn sector_mixings(&self, p: usize) -> Vec<(Link, f64)> {
        let mut out = vec![];
        for s in &self.sectors {
            for (to, places) in self.targets(s, p) {
                out.extend(self.link_weights(s, p, to, places).into_iter().map(|(link, w)| (link, w.sqrt())));
            }
        }
        out
    }

    /// A state's shares among the folded sectors' links (`closure`: where only the sectors' fold does not exist, the
    /// link that goes up): over the state's eigenstates by their weights |⟨l_p|ψ⟩|², each block's population ‖T′r_p‖²
    /// there, split among its links as `sector_parts` splits a block's part (`link_weights`).
    pub(crate) fn sector_shares(&self, psi: &[C64]) -> Vec<(Link, f64)> {
        let n = self.size();
        let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| self.eigen.inverse[(p, i)] * psi[i]).sum()).collect();
        let total: f64 = coefficients.iter().map(|z| z.norm_sqr()).sum();
        let mut out: Vec<(Link, f64)> = vec![];
        for s in &self.sectors {
            for p in (0..n).filter(|&p| coefficients[p] != C64::ZERO) {
                let t = self.sector_apply(s, &self.right(p));
                for (to, places) in self.targets(s, p) {
                    let weights = self.link_weights(s, p, to, places);
                    let sum: f64 = weights.iter().map(|w| w.1).sum();
                    if !(sum > 0.0) {
                        continue;
                    }
                    let part = coefficients[p].norm_sqr() / total * places.iter().map(|&i| t[i].norm_sqr()).sum::<f64>();
                    for (link, w) in weights {
                        match out.iter_mut().find(|o| o.0 == link) {
                            Some(o) => o.1 += part * w / sum,
                            None => out.push((link, part * w / sum)),
                        }
                    }
                }
            }
        }
        out
    }

    /// The normalization of P's coordinates (ELIMINATION.md §2: the first order's renormalization, here whole): the
    /// slow state (x, T x) has the norm x†(1 + X)x, so its normalized coordinates are y = S x, S = (1 + X)^{1/2}; returns
    /// S and S⁻¹ (X Hermitian and positive: its eigenbasis).
    pub(crate) fn normalization(&self) -> (Mat, Mat) {
        let x = crate::linalg::hermitian_eigen(&self.metric());
        let n = self.size();
        let make = |f: &dyn Fn(f64) -> f64| Mat::from_fn(n, n, |i, j| (0..n).map(|k| x.vectors[(i, k)] * f(x.values[k].max(0.0)) * x.vectors[(j, k)].conj()).sum());
        (make(&|v| (1.0 + v).sqrt()), make(&|v| 1.0 / (1.0 + v).sqrt()))
    }

    /// The replicas' responses about Z₀ = diag(z_y) to order K (`series`): per replica R₀(y)^{k+1}W on each place y's
    /// column, k ≤ K, R₀(y) = (z_y + f − K)⁻¹ on its group's places.
    pub(crate) fn responses_about(&self, z0: &[f64], orders: usize) -> Vec<Vec<Mat>> {
        self.replicas
            .iter()
            .map(|rep| {
                let mut x = rep.w.clone();
                (0..=orders)
                    .map(|_| {
                        for &j in &self.members[rep.group] {
                            let sigma = C64::real(z0[j] + rep.frequency);
                            let column: Vec<C64> = (0..x.rows).map(|i| x[(i, j)]).collect();
                            let solved = rep.k.resolvent(sigma, &column);
                            for i in 0..x.rows {
                                x[(i, j)] = solved[i];
                            }
                        }
                        x.clone()
                    })
                    .collect()
            })
            .collect()
    }

    /// The products of two places' resolvents on the order-0 responses `responses` (`responses_about`): per replica and
    /// place s of its group R₀(s) R₀(t) W on every column t, R₀(s) = (z_s + f − K)⁻¹ (zero at a place of another group)
    /// -- the series' first order on column s, T¹[:, s] = −Σ_t R₀(s)R₀(t)W_t E[t, s] (the column's own reference: (z_s +
    /// f − K)T[:, s] = W_s − (T E)[:, s]), and of the left wave operator's row s, W′_s R₀(s) (the landing's Y). Where
    /// z_s = z_t, R₀(t)²W_t: the order-1 response, the same numbers (the same solves).
    fn mixed_about(&self, z0: &[f64], responses: &[Vec<Mat>]) -> Vec<Vec<Mat>> {
        let n = self.size();
        self.replicas
            .iter()
            .zip(responses)
            .map(|(rep, r)| {
                (0..n)
                    .map(|s| {
                        let mut x = Mat::zeros(r[0].rows, n);
                        if !self.members[rep.group].contains(&s) {
                            return x;
                        }
                        let sigma = C64::real(z0[s] + rep.frequency);
                        for &t in &self.members[rep.group] {
                            let column: Vec<C64> = (0..x.rows).map(|i| r[0][(i, t)]).collect();
                            let solved = rep.k.resolvent(sigma, &column);
                            for i in 0..x.rows {
                                x[(i, t)] = solved[i];
                            }
                        }
                        x
                    })
                    .collect()
            })
            .collect()
    }

    /// Σ_f W′_f P_f of the products `mixed` (`mixed_about`) at place s: the series' first-order form on column s,
    /// N⁽ˢ⁾[r, t] = Σ_f W′_f R₀(s)R₀(t) W_f -- N₁ where the references are one.
    fn mixed_form(&self, mixed: &[Vec<Mat>], s: usize) -> Mat {
        let n = self.size();
        let mut m = Mat::zeros(n, n);
        for (rep, p) in self.replicas.iter().zip(mixed) {
            m.add_scaled(C64::ONE, &rep.back.product(&p[s]));
        }
        m
    }

    /// ‖H − Z₀‖₂ (`series_ratio`).
    pub(crate) fn series_norm(&self, z0: &[f64]) -> f64 {
        let mut e = self.generator.clone();
        for i in 0..self.size() {
            e[(i, i)] -= C64::real(z0[i]);
        }
        crate::linalg::hermitian_eigen(&e.adjoint_product(&e)).values.iter().fold(0.0_f64, |m, &x| m.max(x)).sqrt()
    }
    /// The series' distance d (`series_ratio`): the least distance of z_y + f from the spectrum of the excited block the
    /// series holds, over the replicas and their groups' places y (`Fold::nearest_held`; `joins` as there).
    pub(crate) fn series_distance(&self, z0: &[f64], joins: Option<&Mat>) -> f64 {
        self.nearest_held(z0, joins).map_or(f64::INFINITY, |x| x.0)
    }
    /// The excited state that sets the series' distance (`series_distance`; E's index): the block criterion one makes
    /// explicit where the series does not converge (`Series::moved`), and on which a run's state books what the
    /// evaluation's orders leave of its error against the definition (`Model::readings`).
    pub(crate) fn nearest(&self, z0: &[f64], joins: Option<&Mat>) -> Option<usize> {
        self.nearest_held(z0, joins).map(|x| x.1)
    }
    /// Of the blocks the series holds (`Folded::held_blocks`) over the replicas, the one nearest z_y + f, y a place of
    /// the replica's group (each column's resolvent is at its own reference), as its resolvent's norm bounds it
    /// (`SchurBlock::henrici`: 1/‖(z_y + f − K_b)⁻¹‖ ≥ δ/(1 + h), the distance of z_y + f from its spectrum where it is
    /// normal): that distance, and of its Schur form's diagonal entry nearest z_y + f the state its Schur vector holds
    /// most of (E's index); none where the series holds no block. A block it does not hold is in no term of the series,
    /// so its distance is no part of the series' radius.
    fn nearest_held(&self, z0: &[f64], joins: Option<&Mat>) -> Option<(f64, usize)> {
        let mut best: Option<(f64, usize)> = None;
        for rep in &self.replicas {
            let held = rep.held_blocks(joins);
            for (b, sigma) in rep.k.blocks.iter().zip(&held).filter(|x| *x.1).map(|x| x.0).flat_map(|b| self.members[rep.group].iter().map(move |&y| (b, C64::real(z0[y] + rep.frequency)))) {
                let (delta, factor) = b.henrici(sigma);
                let distance = delta / factor;
                if best.is_none_or(|x| distance < x.0) {
                    let m = b.states.len();
                    let k = (0..m).min_by(|&x, &y| (sigma - b.s[x * m + x]).abs().total_cmp(&(sigma - b.s[y * m + y]).abs())).expect("a block's state");
                    let most = (0..m).max_by(|&x, &y| b.q[k * m + x].norm_sqr().total_cmp(&b.q[k * m + y].norm_sqr())).expect("a block's state");
                    best = Some((distance, rep.q[b.states[most]]));
                }
            }
        }
        best
    }
    /// The excited share the run's normalization is a series in (`Series`; assemble.rs `normal`, readout.rs): X = Σ_f
    /// (R₀W_f)†(R₀W_f) of the series' order-0 responses about z₀ -- the slow state's one metric, of which the
    /// generator's S = (1 + X)^{1/2} and the readout's S⁻¹ are both made --: its largest eigenvalue x, the ratio of
    /// S^{±1}'s binomial series (`converges`), and the excited state that holds the most of it, of the responses' rows
    /// the largest Σ_s |R₀W[e, s]|² over the replicas (E's index): the block criterion one moves where x is largest and
    /// not below one (`Series::moved`).
    pub(crate) fn evaluated_share(&self, z0: &[f64]) -> (f64, Option<usize>) {
        let n = self.size();
        let responses = self.responses_about(z0, 0);
        let metric = metric_about(&responses, n);
        let mut weight: Vec<(usize, f64)> = vec![];
        for (rep, r) in self.replicas.iter().zip(&responses) {
            for (i, &e) in rep.q.iter().enumerate() {
                let w: f64 = (0..n).map(|s| r[0][(i, s)].norm_sqr()).sum();
                match weight.iter_mut().find(|x| x.0 == e) {
                    Some(x) => x.1 += w,
                    None => weight.push((e, w)),
                }
            }
        }
        let x = crate::linalg::hermitian_eigen(&metric).values.iter().fold(0.0_f64, |m, &v| m.max(v));
        (x, weight.into_iter().filter(|x| x.1 > 0.0).max_by(|a, b| a.1.total_cmp(&b.1)).map(|x| x.0))
    }

    /// The excited share X = Σ_f T_f†T_f, the metric of P's coordinates: the slow state (x, T x) has the norm x†(1 + X)x.
    /// (A sector's population ‖T′x‖² is no part of it: no fold the run's states are normalized or landed on holds a
    /// sector -- the definition is the static part's, the run folds none -- and what it is, the micromotion's time mean
    /// the display and the next landing leave out, is the sector's part of criterion one, `sector_parts`.)
    pub(crate) fn metric(&self) -> Mat {
        let n = self.size();
        let mut x = Mat::zeros(n, n);
        for rep in &self.replicas {
            x.add_scaled(C64::ONE, &rep.t.adjoint_product(&rep.t));
        }
        x
    }

    /// The parts of an eigenstate p's response (or, with the `offsets` it spans per replica, `offsets`, of its remainder
    /// there, ‖R² W r_p‖ by each replica's sum: module documentation, # First order in the frequency offsets) per excited
    /// block and
    /// replica, each split among the replica's classes (`class_shares`): (block, replica, class, part) -- of the replicas
    /// of p's own group; and the response's (or remainder's) size. What the closure moves a block by (`closure`), and
    /// what a state's error on a block is split by (`Fold::state_block_parts`).
    pub(crate) fn block_parts(&self, blocks: &[Vec<usize>], p: usize, offsets: Option<&[f64]>) -> (f64, Vec<(usize, usize, Option<usize>, f64)>) {
        let block_of = |e: usize| blocks.iter().position(|b| b.contains(&e));
            let (n, r) = (self.size(), self.right(p));
            let lambda = self.eigen.values[p];
            let unit: Vec<C64> = (0..n).map(|q| if q == p { C64::ONE } else { C64::ZERO }).collect();
            let mut parts = vec![];
            let (mut num, mut den) = (0.0, 0.0);
            for (j, rep) in self.replicas.iter().enumerate().filter(|(_, rep)| rep.group == self.groups[p]) {
                let wr = rep.w.apply(&r);
                let sigma = lambda + C64::real(rep.frequency);
                let x0 = rep.k.resolvent(sigma, &wr);
                let d: Vec<C64> = match offsets {
                    None => x0.clone(),
                    Some(offsets) => rep.k.resolvent(sigma, &x0).into_iter().map(|z| z * offsets[j]).collect(),
                };
                let mut on = vec![0.0; blocks.len()];
                for (i, &e) in rep.q.iter().enumerate() {
                    if let Some(b) = block_of(e) {
                        on[b] += d[i].norm_sqr();
                    }
                    num += d[i].norm_sqr();
                    den += x0[i].norm_sqr();
                }
                let mut responses = None;
                for (b, &x) in on.iter().enumerate().filter(|(_, x)| **x > 0.0) {
                    let responses = responses.get_or_insert_with(|| self.feed_responses(j, &unit));
                    for (class, share) in class_shares(responses, &rep.q, &blocks[b]) {
                        parts.push((b, j, class, x * share));
                    }
                }
            }
            let scale = if offsets.is_some() { den } else { 1.0 };
            if scale > 0.0 {
                parts.iter_mut().for_each(|x| x.3 /= scale);
            }
            (if offsets.is_some() { if den > 0.0 { (num / den).sqrt() } else { 0.0 } } else { num.sqrt() }, parts)
            }

    /// Criterion one at eigenstate p (module documentation, # The criteria's numbers), the one function the closure judges
    /// and the readings book by (`closure`, `errors`): its ρ_p² -- the remainder at the offsets its amplitude spans
    /// (`offsets`, at the share `main` of its own amplitude and `source` of each source eigenstate's, `block_parts`) and
    /// the sectors the run leaves out (`sector_parts`, of its own amplitude: at the share `main`, its amplitude's error
    /// `main` times its own, in squares) -- and each object's part of it. Its error is `rate` of ρ_p, each object's its
    /// part's share of that (`split`).
    pub(crate) fn criterion_one(&self, blocks: &[Vec<usize>], p: usize, pattern: f64, main: f64, source: &dyn Fn(usize) -> f64) -> (f64, Vec<(Object, f64)>) {
        let offsets = self.offsets(p, pattern, main, source);
        let mut out: Vec<(Object, f64)> = vec![];
        let mut total = 0.0;
        if offsets.iter().any(|x| *x != 0.0) {
            let (rho, parts) = self.block_parts(blocks, p, Some(&offsets));
            total += rho * rho;
            out.extend(parts.into_iter().map(|(block, replica, class, x)| (Object::Block { block, replica, class }, x)));
        }
        if main > 0.0 {
            for (link, x) in self.sector_parts(p, pattern) {
                total += main * main * x;
                out.push((Object::Link(link), main * main * x));
            }
        }
        (total, out)
    }

    /// A state ψ's error `x` on the excited block `on` -- or, None, on every block --, split as an eigenstate's part is
    /// (`block_parts`): over the state's eigenstates by their weights |⟨l_p|ψ⟩|², each by its replicas' and classes'
    /// parts on the block; the (block, class) shares of x (none where nothing of the state reaches the block).
    pub(crate) fn state_block_parts(&self, blocks: &[Vec<usize>], on: Option<usize>, psi: &[C64], x: f64) -> Vec<(usize, Option<usize>, f64)> {
        let n = self.size();
        let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| self.eigen.inverse[(p, i)] * psi[i]).sum()).collect();
        let total: f64 = coefficients.iter().map(|z| z.norm_sqr()).sum();
        let mut shares: Vec<(usize, Option<usize>, f64)> = vec![];
        for p in (0..n).filter(|&p| total > 0.0 && coefficients[p].norm_sqr() > 0.0) {
            let weight = coefficients[p].norm_sqr() / total;
            for (b, _, class, part) in self.block_parts(blocks, p, None).1.into_iter().filter(|part| on.is_none_or(|b| part.0 == b)) {
                match shares.iter_mut().find(|s| s.0 == b && s.1 == class) {
                    Some(s) => s.2 += weight * part,
                    None => shares.push((b, class, weight * part)),
                }
            }
        }
        let sum: f64 = shares.iter().map(|s| s.2).sum();
        if !(sum > 0.0) {
            return vec![];
        }
        shares.into_iter().map(|(b, class, s)| (b, class, x * s / sum)).collect()
    }

    /// The block move where a series does not converge on the excited block b (criterion one's existence, as the closure
    /// moves where a fold does not exist, `closure`): of the ground-like eigenstates' parts on b, the largest of a class
    /// -- its replica's, its eigenstate's group's ground places --; none where no class reaches b.
    pub(crate) fn block_move(&self, blocks: &[Vec<usize>], b: usize) -> Option<(usize, Vec<usize>)> {
        let mut largest: Option<(usize, usize, f64)> = None;
        for p in (0..self.size()).filter(|&p| self.ground_weight(p) >= 0.5) {
            for (_, _, class, x) in self.block_parts(blocks, p, None).1.into_iter().filter(|part| part.0 == b) {
                if let Some(c) = class.filter(|_| x > largest.map_or(0.0, |m| m.2)) {
                    largest = Some((c, p, x));
                }
            }
        }
        let (class, p, _) = largest?;
        let group = self.groups[self.dominant(p)];
        Some((class, self.members[group].iter().copied().filter(|&place| place < self.ground).collect()))
    }

    /// The evaluation's normalized generator at the orders `orders` (assemble.rs `normal`; ELIMINATION v7, v11) of this
    /// fold's series about Z₀ = diag(z_y) -- the reference's forms, what the tables hold --: H₀ = Herm(A + N₀) − (i/2)L,
    /// N₀ = Σ W′R₀(y)W on column y, L = Σ (R₀W)†D(R₀W), at K = 1 H = H₀ − N⁽ˢ⁾(H₀ − Z₀) on each column s (`mixed_form`:
    /// N₁ where the references are one); its Hermitian part Herm H and its decay S_J⁻¹ (Σ T†D T) S_J⁻¹ of the jumps' wave
    /// operator T = R₀W (− Σ_t R₀(s)R₀(t)W_t E[t, s] on column s at K = 1, E = H − Z₀) and S_J⁻¹ = Σ_{j≤J} b_j X^j,
    /// X = Σ (R₀W)†(R₀W): Herm H − (i/2) decay.
    pub(crate) fn evaluated(&self, levels: &Levels, z0: &[f64], orders: Orders) -> Mat {
        let n = self.size();
        let responses = self.responses_about(z0, 0);
        let mixed = if orders.series >= 1 { self.mixed_about(z0, &responses) } else { vec![] };
        let forms = |k: usize| {
            let mut m = Mat::zeros(n, n);
            for (rep, r) in self.replicas.iter().zip(&responses) {
                m.add_scaled(C64::ONE, &rep.back.product(&r[k]));
            }
            m
        };
        let metric = metric_about(&responses, n);
        let mut loss = Mat::zeros(n, n);
        for (rep, r) in self.replicas.iter().zip(&responses) {
            loss.add_scaled(C64::ONE, &r[0].adjoint_product(&levels.decay.block(&rep.q, &rep.q).product(&r[0])));
        }
        let mut whole = self.direct.clone();
        whole.add_scaled(C64::ONE, &forms(0));
        let mut h = whole.hermitian_part();
        h.add_scaled(C64::new(0.0, -0.5), &loss);
        let offset = |h: &Mat| {
            let mut e = h.clone();
            for i in 0..n {
                e[(i, i)] -= C64::real(z0[i]);
            }
            e
        };
        // Column s of a product of the products at s: Σ_t P⁽ˢ⁾[:, t] E[t, s].
        let at_columns = |of: &dyn Fn(usize) -> Mat, e: &Mat, rows: usize| {
            let mut out = Mat::zeros(rows, n);
            for s in 0..n {
                let column = of(s).product(e);
                for r in 0..rows {
                    out[(r, s)] = column[(r, s)];
                }
            }
            out
        };
        if orders.series >= 1 {
            let next = at_columns(&|s| self.mixed_form(&mixed, s), &offset(&h), n);
            h.add_scaled(-C64::ONE, &next);
        }
        let e = offset(&h);
        let mut inner = Mat::zeros(n, n);
        for (x, (rep, r)) in self.replicas.iter().zip(&responses).enumerate() {
            let mut wave = r[0].clone();
            if orders.series >= 1 {
                wave.add_scaled(-C64::ONE, &at_columns(&|s| mixed[x][s].clone(), &e, r[0].rows));
            }
            inner.add_scaled(C64::ONE, &wave.adjoint_product(&levels.decay.block(&rep.q, &rep.q).product(&wave)));
        }
        let mut bent = Mat::identity(n);
        let mut power = Mat::identity(n);
        for j in 1..=orders.normalization {
            power = power.product(&metric);
            bent.add_scaled(C64::real(binomial(-0.5, j)), &power);
        }
        let mut out = h.hermitian_part();
        out.add_scaled(C64::new(0.0, -0.5), &bent.product(&inner).product(&bent));
        out
    }

    /// The definition (ELIMINATION v3 4, v6, v7): of this fold -- the self-consistent one, its excited block whole
    /// (`Model::definition_fold`), the run's static part: what the run's sectors leave out is criterion one's
    /// (`Fold::sector_parts`), read on the run's own fold, not a part of what the evaluation is measured against -- with
    /// one normalization S = (1 + X)^{1/2} of its own metric (`normalization`), the normalized generator, the des
    /// Cloizeaux generator's Hermitian part Herm(S H S⁻¹) and its decay S⁻¹ Λ S⁻¹: Herm(S H S⁻¹) − (i/2) decay; and the
    /// landing map S (1 + Σ_f S_f T_f)⁻¹, S_f the left wave operators (`left_operators`, the landing's overlap of
    /// `transients`): where a bare state lands, in the normalized coordinates -- None where the left operators do not
    /// settle: the landing does not exist there, and the readings book it whole (`Model::readings`, `converges`). Λ is
    /// the norm's loss on the slow invariant subspace, −d/dt x†(1 + X)x = x†Λx: the jumps' product Σ_f T_f†D T_f.
    pub(crate) fn definitions(&self, levels: &Levels) -> (Mat, Option<Mat>) {
        let n = self.size();
        let (s, s_inverse) = self.normalization();
        let mut loss = Mat::zeros(n, n);
        for rep in &self.replicas {
            loss.add_scaled(C64::ONE, &rep.t.adjoint_product(&levels.decay.block(&rep.q, &rep.q).product(&rep.t)));
        }
        let mut generator = s.product(&self.generator).product(&s_inverse).hermitian_part();
        generator.add_scaled(C64::new(0.0, -0.5), &s_inverse.product(&loss).product(&s_inverse));
        let landing = self.left_operators().and_then(|left| {
            let bares: Vec<Vec<C64>> = (0..n).map(|j| (0..n).map(|i| if i == j { C64::ONE } else { C64::ZERO }).collect()).collect();
            let landed = self.landed(&left, &bares)?;
            Some(s.product(&Mat::from_fn(n, n, |i, j| landed[j][i])))
        });
        (generator, landing)
    }

    /// The evaluation's landing map at the orders `orders` (assemble.rs `normal`, model.rs `Model::landed`) of this fold's
    /// series about Z₀ -- the reference's forms, what the tables hold --: S_J Σ_{j≤J} (−Y)^j, Y[s, t] = Σ_f W′_f,s
    /// R₀(s)R₀(t) W_f,t the left and the right wave operators' overlap to the series' leading order (the left one's row s
    /// W′_s R₀(s), the right one's column t R₀(t)W_t: `mixed_about`; N₁ where the references are one), S_J =
    /// Σ_{j≤J} binom(½, j) X^j, X = Σ (R₀W)†(R₀W): the identity at J = 0.
    pub(crate) fn landing_evaluated(&self, z0: &[f64], orders: Orders) -> Mat {
        let n = self.size();
        let responses = self.responses_about(z0, 0);
        let metric = metric_about(&responses, n);
        let mut partner = Mat::zeros(n, n);
        if orders.normalization >= 1 {
            let mixed = self.mixed_about(z0, &responses);
            for s in 0..n {
                let form = self.mixed_form(&mixed, s);
                for t in 0..n {
                    partner[(s, t)] = form[(s, t)];
                }
            }
        }
        let (mut s, mut power, mut sum, mut term) = (Mat::identity(n), Mat::identity(n), Mat::identity(n), Mat::identity(n));
        for j in 1..=orders.normalization {
            power = power.product(&metric);
            s.add_scaled(C64::real(binomial(0.5, j)), &power);
            term = term.product(&partner).scaled(-C64::ONE);
            sum.add_scaled(C64::ONE, &term);
        }
        s.product(&sum)
    }

    /// The evaluation's landing error on a state ψ against the definition (`landing_evaluated`, `definitions`): after
    /// the next jump the atom lands in a bare ground state at the state's own ground populations p_g (as `errors` reads
    /// the transient), and a landing off by r_g = ‖(L_e − L_d) e_g‖/‖L_d e_g‖ is off on the landed state -- its norm, the
    /// next wait's threshold law, and its populations -- by a rate's relative error 2r_g + r_g²: Σ_g p_g (2r_g + r_g²)
    /// over the `ground` places.
    pub(crate) fn landing_error(evaluated: &Mat, defined: &Mat, ground: usize, psi: &[C64]) -> f64 {
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        (0..ground)
            .filter(|&g| psi[g] != C64::ZERO)
            .map(|g| psi[g].norm_sqr() / norm * rate_error(norm_of((0..defined.rows).map(|i| evaluated[(i, g)] - defined[(i, g)])), norm_of((0..defined.rows).map(|i| defined[(i, g)]))))
            .sum()
    }

    /// The evaluation's error on a state ψ against the definition, measured (ELIMINATION v7; a unit's error where the atoms
    /// went): r = ‖(G_e − G_d)ψ‖ / ‖(G_d − z₀)ψ‖ of the two normalized generators (`evaluated`, `defined`), a rate's
    /// relative error 2r + r². Every truncation of the evaluation -- the series' order, the normalization's, the
    /// excited block the reference leaves out (δK), the Hermitian part's normalization -- in the one comparison.
    pub(crate) fn evaluation_error(evaluated: &Mat, defined: &Mat, z0: &[f64], psi: &[C64]) -> f64 {
        let (e, d) = (evaluated.apply(psi), defined.apply(psi));
        rate_error(norm_of(e.iter().zip(&d).map(|(a, b)| *a - *b)), norm_of(d.iter().zip(psi).zip(z0).map(|((d, p), z)| *d - *p * *z)))
    }

    /// The left wave operators S_f (P ← Q_f): the rows (y, y S) span the slow left invariant subspace,
    /// (y, yS)[[A, W'], [W, K − f]] = λ(y, yS) gives M S_f − S_f (K_f − f) = W'_f with M = A + Σ_f S_f W_f -- the right
    /// one's Sylvester for the adjoint (S† M† − (K† − f) S† = W'†), settled the same way (`settle`). Without decay S = T†.
    /// With them the fast invariant subspace is (−S y, y), y on Q. Of the replicas: the folds a state is landed on fold
    /// no sector (`definitions`; the run's). None where the left iteration does not settle.
    pub(crate) fn left_operators(&self) -> Option<Left> {
        let n = self.size();
        let adjoints: Vec<Schur> = self.replicas.iter().map(|rep| rep.k.adjoint()).collect();
        let mut left = Left { replicas: self.replicas.iter().map(|rep| Mat::zeros(n, rep.q.len())).collect() };
        let excited = self.replicas.iter().map(|rep| rep.q.len()).max().unwrap_or(1);
        // About a reference, the series' order 0: S_f = W′_f R₀, one step from M = z₀.
        if let Some(z0) = &self.reference {
            let sm = schurs_of(&Mat::from_fn(n, n, |i, j| if i == j { C64::real(z0[i]) } else { C64::ZERO }), &self.members).expect("a scalar's Schur form");
            for ((rep, k), si) in self.replicas.iter().zip(&adjoints).zip(left.replicas.iter_mut()) {
                *si = sylvester_in(k, &sm[rep.group], rep.frequency, &rep.back.adjoint(), &self.members[rep.group]).adjoint();
            }
            return Some(left);
        }
        // From the right generator, which the left one is similar to, M = (1 + ST) H (1 + ST)⁻¹: within the excited share
        // X of it.
        let (_, settled) = settle(
            &self.generator,
            |m| {
                let sm = schurs_of(&m.adjoint(), &self.members).expect("the Schur form of P's left generator");
                let (mut next, mut scale) = (self.direct.clone(), self.direct.frobenius());
                for ((rep, k), si) in self.replicas.iter().zip(&adjoints).zip(left.replicas.iter_mut()) {
                    *si = sylvester_in(k, &sm[rep.group], rep.frequency, &rep.back.adjoint(), &self.members[rep.group]).adjoint();
                    next.add_scaled(C64::ONE, &si.product(&rep.w));
                    scale += si.frobenius() * rep.w.frobenius();
                }
                (next, scale)
            },
            (n + excited) * (n + excited),
        );
        settled.then_some(left)
    }

    /// Where bare states f land (`transients`): their slow parts' coordinates x₀ = (1 + Σ_f S_f T_f)⁻¹ f; None where the
    /// overlap is singular.
    pub(crate) fn landed(&self, left: &Left, fs: &[Vec<C64>]) -> Option<Vec<Vec<C64>>> {
        let mut overlap = Mat::identity(self.size());
        for (si, rep) in left.replicas.iter().zip(&self.replicas) {
            overlap.add_scaled(C64::ONE, &si.product(&rep.t));
        }
        let lu = crate::linalg::Lu::new(&overlap)?;
        Some(fs.iter().map(|f| {
            let mut x0 = f.clone();
            lu.solve(&mut x0);
            x0
        }).collect())
    }

    /// The transient of bare states f (ELIMINATION v3 3 and 4, one term): what the waiting time for the next photon from
    /// f holds beyond its slow part's, E_exact − ∫‖slow‖² dt. The bare state is its slow part (x₀, T x₀) and its fast
    /// part (−S y₀, y₀), y₀ = −T x₀, x₀ = (1 + Σ_f S_f T_f)⁻¹ f its landing; the fast part is the
    /// homogeneous solution of the excited block, e^{−i(K − f)t} -- it decays at the excited states' Γ/2 within 1/Γ, and
    /// it is the same term whether the bare state is a jump's landing or the light being switched on. Its survival
    /// adds ∫‖fast‖² + 2 Re∫⟨fast|slow⟩; to the leading order in the fold, per replica,
    ///   τ = (T x₀)† 𝓛 (T x₀) − 2 Re (T x₀)† 𝒳 x₀,
    /// 𝓛 = ∫ e^{iK†t} e^{−iKt} dt (𝓛 K − K† 𝓛 = −i; 𝓛 = 1/Γ on a line of one width) and 𝒳 = ∫ e^{i(K−f)†t} (T − S†)
    /// e^{−iHt} dt (𝒳 H − (K† − f) 𝒳 = −i(T − S†)): the norm the fast part holds over its life, and its interference
    /// with the slow part's excited and ground amplitudes. Without decay S† = T and only the first term is left. On one
    /// line it is the whole difference between the mean scattering rate after a jump and the optical Bloch steady state
    /// beside the threshold's clamp (tests). Per f (x₀, τ), the landing's overlap made once, and 𝓛 and 𝒳 of each replica
    /// some landing reaches (T x₀ not nothing; a landing is its group's, `landed`, and reaches none of another group's).
    pub(crate) fn transients(&self, left: &Left, fs: &[Vec<C64>]) -> Option<Vec<(Vec<C64>, f64)>> {
        let s = &left.replicas;
        let landed = self.landed(left, fs)?;
        let h = schurs_of(&self.generator, &self.members)?;
        let solved: Vec<Option<(Mat, Mat)>> = self
            .replicas
            .iter()
            .zip(s)
            .map(|(rep, si)| {
                let nq = rep.q.len();
                if nq == 0 || landed.iter().all(|x0| self.members[rep.group].iter().all(|&i| x0[i] == C64::ZERO)) {
                    return None;
                }
                let k = &rep.k;
                let kh = k.adjoint();
                let identity = Mat::from_fn(nq, nq, |i, j| if i == j { C64::new(0.0, -1.0) } else { C64::ZERO });
                let lyapunov = solve_sylvester(&kh, C64::ZERO, k, C64::ZERO, &identity);
                let mut b = rep.t.clone();
                b.add_scaled(-C64::ONE, &si.adjoint());
                let cross = solve_in(&kh, C64::real(-rep.frequency), &h[rep.group], C64::ZERO, &b.scaled(C64::new(0.0, -1.0)), &self.members[rep.group]);
                Some((lyapunov, cross))
            })
            .collect();
        Some(
            landed
                .into_iter()
                .map(|x0| {
                    let mut tau = 0.0;
                    for (rep, made) in self.replicas.iter().zip(&solved) {
                        let Some((lyapunov, cross)) = made else { continue };
                        let y = rep.t.apply(&x0);
                        tau += crate::linalg::inner(&y, &lyapunov.apply(&y)).re - 2.0 * crate::linalg::inner(&y, &cross.apply(&x0)).re;
                    }
                    (x0, tau)
                })
                .collect(),
        )
    }
}

/// The fold's series about a reference as the run evaluates it (module documentation, # The definition and the route it
/// is evaluated on): per replica its frequency f, Q_f and the Schur form of K_f there -- K the excited block of the
/// levels the reference is taken at (the far light's) --, which replica each class drives each ground state's amplitude
/// into, and the reference Z₀ = diag(z_y), a number a place. The responses R₀(y)^{j+1} W (`respond`) are linear in a
/// class's coupling W with coefficients of these levels alone: what a table holds without the near light's coordinates.
pub(crate) struct Reference {
    replicas: Vec<(f64, Vec<usize>, Schur)>,
    driven: Vec<Vec<usize>>,
    ground: usize,
    excited: usize,
    z0: Vec<f64>,
}

impl Reference {
    /// The reference of the levels `levels` (their excited block and decay; the ground block is the caller's, in Z₀),
    /// the classes' frequencies `nus`, the slow states and their frames, at Z₀ (rad/µs, in the frames: a number a slow
    /// state).
    pub(crate) fn new(levels: &Levels, nus: &[f64], slow: &Slow, z0: &[f64]) -> Reference {
        let ground = levels.ground.rows;
        let excited = levels.excited.rows;
        let classes: Vec<Class> = nus.iter().map(|&nu| Class { nu, beams: vec![] }).collect();
        let replicas = Replicas::of(&classes, slow, ground);
        let mut k = levels.excited.clone();
        k.add_scaled(C64::new(0.0, -0.5), &levels.decay);
        let mut forms: Vec<(Vec<usize>, Schur)> = vec![];
        let out = replicas
            .frequencies
            .iter()
            .zip(&replicas.scales)
            .map(|(&f, &scale)| {
                let q: Vec<usize> = (0..excited).filter(|&e| !slow.explicit.iter().enumerate().any(|(i, &x)| x == e && equal(slow.frames[ground + i] - f, slow.sizes[ground + i] + scale))).collect();
                if !forms.iter().any(|x| x.0 == q) {
                    forms.push((q.clone(), Schur::of(&k.block(&q, &q)).expect("the Schur form of the excited block")));
                }
                let schur = forms.iter().find(|x| x.0 == q).expect("its form").1.clone();
                (f, q, schur)
            })
            .collect();
        Reference { replicas: out, driven: replicas.driven, ground, excited, z0: z0.to_vec() }
    }

    /// The responses of class c's coupling (E × the ground states) to the orders 0..=`orders`, R₀(y)^{j+1} W on every
    /// excited state (zero off each replica's Q) and every slow state y (E × n, n the slow states: the explicit states'
    /// columns are zero, a class couples E ← G alone), R₀(y) = (z_y + f − K_f)⁻¹ of the replica each ground state's
    /// amplitude drives -- order 1 the first order's Doppler partner, −∂T/∂f, column by column --; and their jets along
    /// the position, per order its parts (jet.rs: the value, the gradient's three, the Hessian's six of `jet::PAIRS`),
    /// exact by the resolvent's identity -- not differences (`chain`). Returns [order][part].
    pub(crate) fn respond(&self, class: usize, coupling: &Mat, n: usize, orders: usize, slopes: &[(Vec<f64>, Mat)]) -> Vec<Vec<Mat>> {
        self.chain(class, coupling, n, orders, slopes, &|_, y| y)
    }

    /// The products of two places' resolvents on class c's coupling, R₀(x)R₀(y) W on every ground place y's column, one
    /// a place x of `places` (any of the slow states: a reference class's first, eliminate.rs `Model::references`) -- of
    /// the series' first order at x's reference and of the landing's Y = Σ_f W′_f R₀(x)R₀(y) W_f, the left and the right
    /// wave operators' overlap at their leading order (the left one's row x is W′_x R₀(x), `Fold::landing_evaluated`) --:
    /// [x][part], with their jets as `respond`'s. Where z_x = z_y, the order-1 response R₀(y)² W_y, the same numbers
    /// (the same solves).
    pub(crate) fn mixed(&self, class: usize, coupling: &Mat, n: usize, slopes: &[(Vec<f64>, Mat)], places: &[usize]) -> Vec<Vec<Mat>> {
        places.iter().map(|&x| self.chain(class, coupling, n, 1, slopes, &|order, y| if order == 0 { y } else { x }).pop().expect("order 1")).collect()
    }

    /// The chain x_j = R₀(z_{at(j, y)}) x_{j−1} on ground place y's column, x_{−1} = W (the coupling, the same everywhere:
    /// the near field is the forms' coefficient), to the order `orders`, each order's resolvent at the place `at` names.
    /// With A = z + f − K_f on each replica's Q, A x_j = x_{j−1} differentiated gives
    ///   ∂_i x_j = R₀ (∂_i x_{j−1} − ∂_i A x_j),
    ///   ∂_i∂_k x_j = R₀ (∂_i∂_k x_{j−1} − ∂_i A ∂_k x_j − ∂_k A ∂_i x_j − ∂_i∂_k A x_j),
    /// ∂A = ∂z − ∂K (f is a frame's), `slopes` the derivatives (∂Z₀, ∂K) of the reference (a number a place) and of the
    /// excited block K (its decay constant) from the far light's Hamiltonian, part by part after the value (none: the
    /// values alone). Each part a resolvent per order and driven column, at the rounding of the solves themselves.
    fn chain(&self, class: usize, coupling: &Mat, n: usize, orders: usize, slopes: &[(Vec<f64>, Mat)], at: &dyn Fn(usize, usize) -> usize) -> Vec<Vec<Mat>> {
        let parts = 1 + slopes.len();
        let mut out = vec![vec![Mat::zeros(self.excited, n); parts]; orders + 1];
        for (r, (f, q, schur)) in self.replicas.iter().enumerate() {
            // ∂A on Q at place z's reference applied to a vector: ∂z x − ∂K x, of part p.
            let slope = |p: usize, z: usize, x: &[C64]| -> Vec<C64> {
                let (dz, k) = &slopes[p - 1];
                q.iter().enumerate().map(|(a, &e)| x[a] * dz[z] - q.iter().enumerate().map(|(b, &e2)| k[(e, e2)] * x[b]).sum::<C64>()).collect()
            };
            for g in (0..self.ground).filter(|&g| self.driven[class][g] == r) {
                let column: Vec<C64> = q.iter().map(|&e| coupling[(e, g)]).collect();
                if column.iter().all(|z| *z == C64::ZERO) {
                    continue;
                }
                // The previous order's jet, x_{−1} = W: its value the coupling, its derivatives none.
                let mut before: Vec<Vec<C64>> = (0..parts).map(|p| if p == 0 { column.clone() } else { vec![C64::ZERO; q.len()] }).collect();
                for (j, order) in out.iter_mut().enumerate() {
                    let z = at(j, g);
                    let sigma = C64::real(self.z0[z] + f);
                    let mut x: Vec<Vec<C64>> = vec![schur.resolvent(sigma, &before[0])];
                    for p in 1..parts.min(4) {
                        let a = slope(p, z, &x[0]);
                        x.push(schur.resolvent(sigma, &before[p].iter().zip(&a).map(|(b, a)| *b - *a).collect::<Vec<C64>>()));
                    }
                    for (m, &(i, k)) in super::jet::PAIRS.iter().enumerate().take(parts.saturating_sub(4)) {
                        let p = 4 + m;
                        let (ai, ak, aik) = (slope(1 + i, z, &x[1 + k]), slope(1 + k, z, &x[1 + i]), slope(p, z, &x[0]));
                        let rhs: Vec<C64> = (0..q.len()).map(|a| before[p][a] - ai[a] - ak[a] - aik[a]).collect();
                        x.push(schur.resolvent(sigma, &rhs));
                    }
                    for (p, part) in order.iter_mut().enumerate() {
                        for (a, &e) in q.iter().enumerate() {
                            part[(e, g)] = x[p][a];
                        }
                    }
                    before = x;
                }
            }
        }
        out
    }

    /// The complex multiply-adds of one coupling's responses to `orders` with their jets of `parts` (`respond`): per
    /// driven column, order and part a resolvent's two changes of basis and its back substitution on Q, and per
    /// derivative its products by ∂A -- one a gradient's part, three a Hessian's.
    pub(crate) fn work(&self, orders: usize, parts: usize) -> f64 {
        let q = self.replicas.iter().map(|r| r.1.len()).max().unwrap_or(0) as f64;
        let products = parts.min(4).saturating_sub(1) + 3 * parts.saturating_sub(4);
        (self.ground * (orders + 1)) as f64 * (2.5 * parts as f64 + products as f64) * q * q
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The system at a point, and the point's closure (criterion one on the blocks and the sectors, the frames).

/// The model's share of a target (ELIMINATION v2 4; SPEC §5.2): what the step's share s leaves the model, ε√(1 − s²) --
/// 0.0447 of the populations' 0.05 (absolute), twice that of T's 0.10 (relative).
pub(crate) fn share() -> f64 {
    crate::units::POPULATION_TARGET * (1.0 - crate::units::STEP_SHARE * crate::units::STEP_SHARE).sqrt()
}

/// A request's levels and light at a point r (µm): H0 and every frequency's outside form at the point's field
/// (request.rs `OutsideForm`, its value at the focus taken out of H0), the decay and the emission, and the classes --
/// per frequency of the light its co-rotating part at δ_ω and its counter-rotating part at −(δ_ω + 2ν_frame), each
/// beam's coupling at its amplitude there (the counter-rotating one at its conjugate, its wave vector −k).
pub(crate) fn system(request: &crate::request::Request, light: &crate::field::Light, r: [f64; 3]) -> (Levels, Vec<Class>, Vec<(usize, bool)>) {
    use crate::request::{GROUND, STATES};
    let amplitudes = light.amplitudes(r);
    let focus = light.amplitudes([0.0; 3]);
    let mut h = request.hamiltonian.clone();
    for f in &light.frequencies {
        let (e, e0) = (f.field(&amplitudes).value, f.field(&focus).value);
        let form = super::far::outside_form(&request.outside_forms[f.form].matrices);
        for p in 0..3 {
            for q in 0..3 {
                let weight = e[p].conj() * e[q] - e0[p].conj() * e0[q];
                if weight != C64::ZERO {
                    h.add_scaled(weight, &form[p][q]);
                }
            }
        }
    }
    let (ground, excited): (Vec<usize>, Vec<usize>) = ((0..GROUND).collect(), (GROUND..STATES).collect());
    let emission: [Mat; 3] = std::array::from_fn(|a| request.emission[a].block(&ground, &excited));
    let mut decay = Mat::zeros(excited.len(), excited.len());
    for c in &emission {
        decay.add_scaled(C64::ONE, &c.adjoint_product(c));
    }
    let levels = Levels { ground: h.block(&ground, &ground), excited: h.block(&excited, &excited), decay };
    let omega = |index: &[i32]| std::f64::consts::TAU * index.iter().zip(&request.frequencies_mhz).map(|(&n, f)| n as f64 * f).sum::<f64>();
    let frame = std::f64::consts::TAU * request.frame_frequencies_mhz[GROUND];
    let (mut classes, mut sources) = (vec![], vec![]);
    for (w, f) in light.frequencies.iter().enumerate() {
        let delta = omega(&f.index);
        for counter in [false, true] {
            let beams = f
                .beams
                .iter()
                .map(|&l| {
                    let b = &request.beams[l];
                    let a = amplitudes[l].value;
                    let (coupling, k) = if counter { (b.counter_coupling.block(&excited, &ground).scaled(a.conj()), light.beams[l].wave_vector().map(|x| -x)) } else { (b.coupling.block(&excited, &ground).scaled(a), light.beams[l].wave_vector()) };
                    Beam { coupling, wave: k }
                })
                .filter(|b| b.coupling.max_abs() > 0.0)
                .collect::<Vec<_>>();
            if !beams.is_empty() {
                classes.push(Class { nu: if counter { -(delta + 2.0 * frame) } else { delta }, beams });
                sources.push((w, counter));
            }
        }
    }
    (levels, classes, sources)
}

/// The drives a fold takes at the point rather than at its bare levels (ELIMINATION.md §2: the bare levels' fold is
/// this function at a coarser point), each by its own part of the route's error: of each ground-like eigenstate p, its
/// dominant ground state g at its bare energy E_g (H0's, `bare`, in the frames), and of each class c the relative change
/// of its response ‖[R(E_g + f) − R(λ_p + f)] W_c r_p‖ / ‖R(λ_p + f) W_c r_p‖ over its replicas, split by its drives e ← s:
/// a drive's part |W_c[e, s] r_p[s]| ‖[R(E_g + f) − R(λ_p + f)] e_e‖ over the same norm -- the parts bound the class's
/// whole. The drives moved to the point are the largest parts until what the rest leaves is within `limit` (what no
/// output resolves): a class's drives of a part in 10⁸ of its coupling -- a tilted field's mixing through the levels
/// outside the model (PGC1D_TILT_2) -- stay at the bare levels' constant, at no cost of their own, however much the
/// class's main drives' route asks for the point. Returns (class, excited state's E index, ground place).
pub(crate) fn bare_routes(fold: &Fold, bare: &[f64], limit: f64) -> Vec<(usize, usize, usize)> {
    let mut out: Vec<(usize, usize, usize)> = vec![];
    for p in (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5) {
        let r = fold.right(p);
        let g = fold.dominant(p);
        if g >= fold.ground {
            continue;
        }
        let lambda = fold.eigen.values[p];
        let level = C64::real(bare[g] - fold.frames[g]);
        for (c, row) in fold.driven.iter().enumerate() {
            // (excited, ground, part²) over the class's replicas, and the response's norm².
            let (mut parts, mut den): (Vec<(usize, usize, f64)>, f64) = (vec![], 0.0);
            for (j, rep) in fold.replicas.iter().enumerate().filter(|(_, rep)| rep.group == fold.groups[p]) {
                let columns: Vec<usize> = (0..fold.ground).filter(|&s| row[s] == Some(j)).collect();
                if columns.is_empty() {
                    continue;
                }
                let m = rep.q.len();
                let mut w = Mat::zeros(m, fold.size());
                for f in rep.feeds.iter().filter(|f| f.class == Some(c)) {
                    w.add_scaled(C64::ONE, &f.w);
                }
                let f = C64::real(rep.frequency);
                den += rep.k.resolvent(lambda + f, &w.apply(&r)).iter().map(|a| a.norm_sqr()).sum::<f64>();
                for i in 0..m {
                    let drives: Vec<(usize, C64)> = columns.iter().map(|&s| (s, w[(i, s)] * r[s])).filter(|x| x.1 != C64::ZERO).collect();
                    if drives.is_empty() {
                        continue;
                    }
                    let unit: Vec<C64> = (0..m).map(|k| if k == i { C64::ONE } else { C64::ZERO }).collect();
                    let (at, bared) = (rep.k.resolvent(lambda + f, &unit), rep.k.resolvent(level + f, &unit));
                    let change = at.iter().zip(&bared).map(|(a, b)| (*a - *b).norm_sqr()).sum::<f64>();
                    parts.extend(drives.into_iter().map(|(s, x)| (rep.q[i], s, x.norm_sqr() * change)));
                }
            }
            if den == 0.0 {
                continue;
            }
            let mut sizes: Vec<(usize, usize, f64)> = parts.into_iter().map(|(e, s, x)| (e, s, (x / den).sqrt())).collect();
            sizes.sort_by(|a, b| b.2.total_cmp(&a.2));
            let mut rest: f64 = sizes.iter().map(|x| x.2).sum();
            for (e, s, x) in sizes {
                if rest <= limit {
                    break;
                }
                rest -= x;
                if !out.contains(&(c, e, s)) {
                    out.push((c, e, s));
                }
            }
        }
    }
    out
}

/// A model's series constants (module documentation, # The definition and the route it is evaluated on; ELIMINATION v5,
/// v7): per cluster the centre c of its near light's scalar shift over the atoms read (the reference z₀ is the cluster's
/// far levels' mean and c), the series' largest ratio q over them and the largest excited share x (the evaluation's
/// metric's largest eigenvalue, `Fold::evaluated_share`) -- each series' existence, q < 1 and x < 1 (`converges`: the
/// Neumann series of the resolvent and the binomial one of (1 + X)^{∓1/2}; criterion one's condition, `Series::exists`),
/// nothing booked of them --, and where one does not, the block move on the excited state that stops it -- the one that
/// sets the distance where the ratio is largest (`Fold::nearest`), or that holds the most of the excited share where x is
/// largest (`Fold::evaluated_share`) --; and the
/// orders every evaluation takes, the kept units' (`Series::kept_orders`):
/// their errors are measured where the atoms went (`Model::readings`).
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Series {
    pub centre: Vec<f64>,
    pub ratio: f64,
    pub order: usize,
    pub excited: f64,
    pub normalization: usize,
    /// Where a series does not converge: the block move (`Fold::block_move`) -- the block, the class (a carrier), the
    /// cluster's ground states of the eigenstate's group.
    pub moved: Option<(usize, usize, Vec<usize>)>,
}

impl Series {
    /// The series of no light: every order 0 (a model's own before its atoms are read, `Model::referenced`).
    pub(crate) fn none(clusters: usize) -> Series {
        Series { centre: vec![0.0; clusters], ratio: 0.0, order: 0, excited: 0.0, normalization: 0, moved: None }
    }

    /// Whether both series converge at the atoms read (`converges`): the resolvent's in its argument, q, and the
    /// normalization's binomial one, x (`Series`; where one does not, criterion one makes a block explicit,
    /// `Model::series_block`).
    pub(crate) fn exists(&self) -> bool {
        converges(self.ratio) && converges(self.excited)
    }

    /// The orders the kept units ask (`Unit::Series`, `Unit::Normalization`): the series' first where it is kept, else
    /// order 0; the normalization's highest kept order, else 0.
    pub(crate) fn kept_orders(kept: &[super::Unit]) -> (usize, usize) {
        let normalization = kept.iter().filter_map(|u| if let super::Unit::Normalization { order } = u { Some(*order) } else { None }).max().unwrap_or(0);
        (usize::from(kept.contains(&super::Unit::Series)), normalization)
    }

    /// The orders an evaluation takes.
    pub(crate) fn orders(&self) -> Orders {
        Orders { series: self.order, normalization: self.normalization }
    }
}

/// The orders of the normalized generator's evaluation (ELIMINATION v7; `Series`): the series' K (0, or 1 where
/// `Unit::Series` is kept: the near forms hold N₀ and N₁), the normalization's J. The Hermitian part is the series' own:
/// what the normalization would do to it is in the evaluation's error against the definition (`Fold::evaluation_error`),
/// booked on the block nearest the reference (`Model::readings`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Orders {
    pub series: usize,
    pub normalization: usize,
}

/// A vector's 2-norm.
fn norm_of(v: impl Iterator<Item = C64>) -> f64 {
    v.map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// The one error of an amplitude's relative error r (module documentation, # The criteria's numbers): the rate's,
/// 2r + r², at most all of it (one); a number that is none, all of it.
///
/// An amplitude e -- an excited response T r_p, a generator's action, a landing -- wrong by δe, ‖δe‖ = r‖e‖, moves
/// every rate it makes, Γ‖e‖² (the scattering, the pumping, the hazard), by |2 Re⟨e|δe⟩ + ‖δe‖²| ≤ (2r + r²)‖e‖² --
/// attained where δe lies along e --, and a shift ⟨W r_p|e⟩ by at most r‖W r_p‖‖e‖, r itself where W r_p lies along e
/// (one line). The populations are the rates' ratios and T is set by the rates and the shifts' gradients: a relative
/// rate error x moves the populations by at most x and T by x (SPEC §1.5). So of an amplitude's r the output's error is
/// the rate's 2r + r² -- the shift's r is below it, and no output is the amplitude itself --: what the closure judges
/// and the readings book, one number (`closure`, `errors`).
pub(crate) fn rate(r: f64) -> f64 {
    if r.is_finite() { (2.0 * r + r * r).min(1.0) } else { 1.0 }
}

/// The amplitude's relative error whose rate's error is x (`rate`): √(1 + x) − 1 = x/(√(1 + x) + 1), the second form free
/// of the first's cancellation at small x -- what an error that is a rate's or a population's already (a sector's
/// scattering apart, its population) is in the units of ρ, so that `rate` of it alone is it (`Fold::sector_parts`).
fn amplitude(x: f64) -> f64 {
    x / ((1.0 + x).sqrt() + 1.0)
}

/// A rate's relative error of an amplitude's r = `apart`/`scale` (`rate`): nothing where nothing is apart, all of it
/// where the scale is nothing and something is.
fn rate_error(apart: f64, scale: f64) -> f64 {
    if apart == 0.0 {
        return 0.0;
    }
    rate(apart / scale)
}

/// The slow state's one metric as the run evaluates it (assemble.rs `bent`, `Model::bent_at`): X = Σ_f (R₀W_f)†(R₀W_f)
/// of the series' order-0 responses about z₀ (`Fold::responses_about`) -- the generator's normalization's
/// (`Fold::evaluated`), the landing map's (`Fold::landing_evaluated`) and its series' ratio's (`Fold::evaluated_share`).
fn metric_about(responses: &[Vec<Mat>], n: usize) -> Mat {
    let mut metric = Mat::zeros(n, n);
    for r in responses {
        metric.add_scaled(C64::ONE, &r[0].adjoint_product(&r[0]));
    }
    metric
}

/// The binomial coefficient binom(α, j) = α(α − 1)…(α − j + 1)/j!: the power series' j-th coefficient of (1 + x)^α.
pub(crate) fn binomial(alpha: f64, j: usize) -> f64 {
    (0..j).fold(1.0, |c, i| c * (alpha - i as f64) / (i + 1) as f64)
}

impl super::Model {
    /// The fold of a model's cluster at a point, as the run's series stands for it (`Series`): the levels with the far
    /// light alone (eliminate.rs `Model::far_hamiltonian`), every carrier a class of its coherent field's couplings
    /// folded at the atom and explicit, the cluster's explicit states and frames; with each place's far level in its
    /// frame and what the far levels leave out of the excited block -- the carriers' own static forms there, bilinear
    /// in the near field (eliminate.rs `level_pair`): δK on E.
    pub(crate) fn cluster_fold(&self, cluster: usize, amplitudes: &[crate::field::Amplitude], about: bool) -> Option<(Fold, Levels, Vec<f64>, Mat)> {
        self.clusters_fold(&[cluster], amplitudes, about, false, false, false).map(|(fold, levels, far, left, _)| (fold, levels, far, left))
    }

    /// A cluster's reference at a point, Z₀ = diag(z_y): each place's far level in its frame (`cluster_fold`'s `far`)
    /// and the cluster's centre c (`Series`).
    pub(crate) fn reference_of(&self, cluster: usize, far: &[f64]) -> Vec<f64> {
        far.iter().map(|f| f + self.series.centre[cluster]).collect()
    }

    /// The run's own fold of the whole atom at a point (`Model::readings`): every cluster of ground states (the others' states
    /// are Q to every group, as to every cluster's own fold) one group of P (# Groups), each
    /// about its own reference, as `cluster_fold` -- the run's evaluation, a cluster's eigenstates its run's --, with the
    /// sectors of every turning coupling the model does not carry, of two clusters or off a cluster's generator (# Sectors:
    /// what the run leaves out), and of those it carries; and the model state of each of its places.
    pub(crate) fn atom_fold(&self, amplitudes: &[crate::field::Amplitude]) -> (Fold, Vec<usize>) {
        let all: Vec<usize> = (0..self.clusters.len()).filter(|&k| self.clusters[k].ground > 0).collect();
        let (fold, .., states) = self.clusters_fold(&all, amplitudes, true, false, true, true).expect("the atom's fold");
        (fold, states)
    }

    /// A link of a fold of the model's clusters as the model's unit (`Unit::Term`): its states (labels, # Sectors: a
    /// ground place, or the fold's ground count plus an excited state's E index) as the request's, by the fold's ground
    /// places' states `grounds`; its classes' lab beat (a carrier's, or none's zero).
    pub(crate) fn link_unit(&self, grounds: &[usize], l: &Link) -> super::Unit {
        use crate::request::GROUND;
        let g = grounds.len();
        let state = |label: usize| if label < g { grounds[label] } else { GROUND + label - g };
        let class = |k: Option<usize>| k.map_or(self.axes.zero(), |k| self.carriers[k].beat.clone());
        super::eliminate::term_unit(state(l.a), state(l.b), &class(l.classes.0).plus(-1, &class(l.classes.1)))
    }

    /// The definition at a point (ELIMINATION v6, v7): the cluster's self-consistent fold with its excited block whole --
    /// the far levels and the carriers' own static forms on it (δK, which the series' reference leaves out) -- and its
    /// levels: what every truncation of the evaluation is measured against (`Fold::evaluation_error`). The run's static
    /// part: what the run's sectors leave out is criterion one's, on the run's own fold (`atom_fold`, `errors`).
    pub(crate) fn definition_fold(&self, cluster: usize, amplitudes: &[crate::field::Amplitude]) -> Option<(Fold, Levels)> {
        self.clusters_fold(&[cluster], amplitudes, false, true, false, false).map(|(fold, levels, ..)| (fold, levels))
    }

    /// The fold of the clusters `clusters` at a point, each one group of P (# Groups) in its own frames: P their ground
    /// states in their order, then their explicit states in their order; about each cluster's reference (each place's
    /// far level in its frame and the cluster's centre, `Series`) or self-consistent, its excited block the far levels'
    /// or, `whole`, with the carriers' own static forms on it, with the sectors where `sectors` (the links the model
    /// carries, `link_unit` in its generator: `Model::in_generator`); the fold, its levels, each place's far level in its
    /// frame, those forms, and the model state of each place. None where no carrier couples the clusters at the point,
    /// unless `any`.
    fn clusters_fold(&self, clusters: &[usize], amplitudes: &[crate::field::Amplitude], about: bool, whole: bool, any: bool, sectors: bool) -> Option<(Fold, Levels, Vec<f64>, Mat, Vec<usize>)> {
        use crate::request::{GROUND, STATES};
        let running = self.running();
        let h = &self.far_hamiltonian(amplitudes, &running, 0)[0];
        let grounds: Vec<usize> = clusters.iter().flat_map(|&k| self.clusters[k].states[..self.clusters[k].ground].iter().copied()).collect();
        let explicit: Vec<usize> = clusters.iter().flat_map(|&k| self.clusters[k].states[self.clusters[k].ground..].iter().copied()).collect();
        let g = grounds.len();
        let excited: Vec<usize> = (GROUND..STATES).collect();
        let classes: Vec<Class> = self
            .carriers
            .iter()
            .map(|cc| {
                let mut coupling = Mat::zeros(excited.len(), g);
                for (l, (folded, explicit)) in cc.folded.iter().zip(&cc.explicit).enumerate() {
                    let a = amplitudes[cc.beams[l]].value;
                    coupling.add_scaled(a, &folded.block(&excited, &grounds));
                    coupling.add_scaled(a, &explicit.block(&excited, &grounds));
                }
                Class { nu: cc.detuning, beams: vec![Beam { coupling, wave: [0.0; 3] }] }
            })
            .collect();
        if !any && classes.iter().all(|class| class.beams[0].coupling.max_abs() == 0.0) {
            return None;
        }
        // The carriers' static forms on the excited block.
        let mut left = Mat::zeros(excited.len(), excited.len());
        for cc in &self.carriers {
            let f = &self.light.frequencies[cc.frequency];
            let n = f.beams.len();
            for (x, form) in self.tones[cc.frequency].pairs.iter().enumerate() {
                let z = super::far::pair(&amplitudes[f.beams[x / n]], &amplitudes[f.beams[x % n]]).value;
                if z != C64::ZERO {
                    left.add_scaled(z, &form.block(&excited, &excited));
                }
            }
        }
        let mut block = h.block(&excited, &excited);
        if whole {
            block.add_scaled(C64::ONE, &left);
        }
        let levels = Levels { ground: h.block(&grounds, &grounds), excited: block, decay: self.decay.clone() };
        // Per place its cluster (a position in `clusters`) and frame, ground places first.
        let cluster_of: Vec<usize> = clusters.iter().enumerate().flat_map(|(i, &k)| std::iter::repeat_n(i, self.clusters[k].ground)).chain(clusters.iter().enumerate().flat_map(|(i, &k)| std::iter::repeat_n(i, self.clusters[k].states.len() - self.clusters[k].ground))).collect();
        let states: Vec<usize> = grounds.iter().chain(&explicit).copied().collect();
        let frame = |s: usize| {
            let c = &self.clusters[clusters[cluster_of[s]]];
            &c.frames[c.states.iter().position(|&x| x == states[s]).expect("a cluster's state")]
        };
        let frames: Vec<f64> = (0..states.len()).map(|s| self.omega(frame(s))).collect();
        let sizes: Vec<f64> = (0..states.len()).map(|s| self.axes.size(frame(s))).collect();
        let far: Vec<f64> = (0..states.len()).map(|y| h[(states[y], states[y])].re - frames[y]).collect();
        let slow = Slow { explicit: explicit.iter().map(|&s| s - GROUND).collect(), frames, sizes, groups: cluster_of.clone() };
        let replicas = Replicas::of(&classes, &slow, g);
        let carries = |l: &Link| {
            let super::Unit::Term { a, b, beat } = self.link_unit(&grounds, l) else { unreachable!("a link's unit is a term") };
            self.in_generator(a, b, &beat)
        };
        let reference: Vec<f64> = cluster_of.iter().zip(&far).map(|(&i, f)| f + self.series.centre[clusters[i]]).collect();
        let fold = fold_about(&levels, &classes, &slow, &replicas, about.then_some(&reference[..]), sectors.then_some(&carries as &dyn Fn(&Link) -> bool));
        Some((fold, levels, far, left, states))
    }

    /// The model's series constants at the atoms `points` (`Series`): per cluster the centre of the near light's scalar
    /// shift -- the mean over the fold's ground-like eigenvalues of each one's shift from its places' far levels (by its
    /// weights on them: R4_LIN_4's cluster of F = 1 and F = 2 ground states, 4.3·10⁴ rad/µs apart in one frame, and its
    /// explicit states, F = 1 at one level in its frame with the explicit states it drives, read 3.6·10³ against the two levels'
    /// mean), over the atoms its range's middle --, then at each atom the ratio of the series about Z₀ = diag(z_y), z_y each place's far
    /// level in its frame and c, q = (‖H − Z₀‖₂ + ‖δK‖)/d (the
    /// Neumann series of the resolvent in both its arguments: `Fold::series_norm` and ‖δK‖ of the excited block the
    /// reference leaves out, in the Frobenius norm, a bound of the spectral one, over `Fold::series_distance` on the
    /// blocks the series holds through W and δK), the largest; and the order it asks for. An atom where a fold does not exist (criterion one's: its
    /// closure makes a block explicit) is read as the series' end: q = ∞, as is a ratio that is no number (`converges`).
    /// And at each atom the ratio of the normalization's series, the evaluation's own excited share x about the same z₀
    /// (`Fold::evaluated_share`: the one metric the generator and the readout take), the largest. Where either series
    /// does not converge, the closure's block move on the state that stops it (`Series::moved`).
    pub(crate) fn series_at(&self, points: &[([f64; 3], [f64; 3])]) -> Series {
        use rayon::prelude::*;
        let clusters = self.clusters.len();
        let read: Vec<Vec<Option<(f64, Fold, Vec<f64>, Mat)>>> = points
            .par_iter()
            .map(|&(r, _)| {
                let amplitudes = self.light.amplitudes(r);
                (0..clusters)
                    .map(|cluster| {
                        let (fold, _, far, left) = self.cluster_fold(cluster, &amplitudes, false)?;
                        // Each ground-like eigenvalue's shift from its own places' far levels, by its weights on them.
                        let like: Vec<usize> = (0..fold.size()).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
                        let own = |p: usize| {
                            let r = fold.right(p);
                            r.iter().zip(&far).map(|(z, f)| z.norm_sqr() * f).sum::<f64>() / r.iter().map(|z| z.norm_sqr()).sum::<f64>()
                        };
                        let shift = like.iter().map(|&p| fold.eigen.values[p].re - own(p)).sum::<f64>() / like.len().max(1) as f64;
                        Some((shift, fold, far, left))
                    })
                    .collect()
            })
            .collect();
        let centre: Vec<f64> = (0..clusters)
            .map(|cluster| {
                let (low, high) = read.iter().filter_map(|at| at[cluster].as_ref()).fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), x| (l.min(x.0), h.max(x.0)));
                if low.is_finite() { 0.5 * (low + high) } else { 0.0 }
            })
            .collect();
        // Per atom and cluster the two ratios about z₀, and the excited state that holds the most of the share.
        type Read<'a> = (f64, f64, usize, &'a Fold, Vec<f64>, Option<usize>, &'a Mat);
        let ratios: Vec<Read> = read
            .iter()
            .flat_map(|at| at.iter().enumerate())
            .filter_map(|(cluster, x)| x.as_ref().map(|x| (cluster, x)))
            .map(|(cluster, (_, fold, far, left))| {
                let z0: Vec<f64> = far.iter().map(|f| f + centre[cluster]).collect();
                let q = if fold.exists { (fold.series_norm(&z0) + left.frobenius()) / fold.series_distance(&z0, Some(left)) } else { f64::INFINITY };
                let (x, most) = fold.evaluated_share(&z0);
                let number = |r: f64| if r.is_nan() { f64::INFINITY } else { r };
                (number(q), number(x), cluster, fold, z0, most, left)
            })
            .collect();
        let largest = |pick: fn(&Read) -> f64| ratios.iter().fold((0.0_f64, None), |(m, at), r| if pick(r) > m || at.is_none() { (pick(r).max(m), Some(r)) } else { (m, at) });
        let ((ratio, nearest), (excited, densest)) = (largest(|r| r.0), largest(|r| r.1));
        // Where a series does not converge, the closure's move on the block that stops it (criterion one).
        let blocks = self.blocks();
        let move_on = |cluster: usize, fold: &Fold, e: Option<usize>| -> Option<(usize, usize, Vec<usize>)> {
            let b = e.and_then(|e| blocks.iter().position(|b| b.contains(&e)))?;
            let (class, places) = fold.block_move(&blocks, b)?;
            Some((b, class, places.iter().map(|&place| self.clusters[cluster].states[place]).collect()))
        };
        let moved = if !converges(ratio) {
            nearest.and_then(|(_, _, cluster, fold, z0, _, left)| move_on(*cluster, fold, fold.nearest(z0, Some(left))))
        } else if !converges(excited) {
            densest.and_then(|(_, _, cluster, fold, _, most, _)| move_on(*cluster, fold, *most))
        } else {
            None
        };
        let (order, normalization) = Series::kept_orders(&self.kept);
        Series { centre, ratio, order, excited, normalization, moved }
    }

    /// The excited hyperfine blocks F′ of the model's states (`blocks`), from its labels.
    pub(crate) fn blocks(&self) -> Vec<Vec<usize>> {
        excited_blocks(&self.labels)
    }

    /// Where a series does not converge at the model's atoms (`Series::exists`: criterion one's condition, of the same
    /// kind as a fold that does not exist): the block the closure moves there (`Series::moved`, `Fold::block_move`) --
    /// the block of the state that sets the series' distance where its ratio is largest (q ≥ 1), or of the state that
    /// holds the most of the excited share (x ≥ 1); its class and its eigenstate's group's ground states --; none where
    /// both converge.
    pub(crate) fn series_block(&self) -> Option<super::Unit> {
        let (block, class, grounds) = self.series.moved.as_ref()?;
        Some(super::Unit::Block { block: *block, component: super::Component { frequency: self.carriers[*class].frequency, counter: false }, grounds: ground_set(grounds) })
    }

    /// The model's bare drives onto explicit states (model.rs `Model::settled`; far.rs # A fold onto an explicit state):
    /// the co-rotating drives e ← w folded with the bare levels onto a state e explicit in a cluster -- a constant read
    /// across its two-photon distance, whose error the states read (`bare_mixing`) and the keep rule takes to the atom
    /// (`Unit::Bare`). A counter-rotating one has no level above (the model's domain): no unit.
    pub(crate) fn bare_links(&self) -> Vec<(super::Component, usize, usize)> {
        self.driven.iter().copied().filter(|&(c, e, w)| !c.counter && self.level(c, e, w) == super::Level::Bare && self.explicit(e)).collect()
    }

    /// Every state's level at the beams' amplitudes `amplitudes` (rad/µs, the lab's): the far light's there
    /// (`Model::far_hamiltonian`) with the carriers' own static forms, and on the ground states of a cluster the near
    /// light folds onto at the atom its shift there too (the cluster's fold about its reference, `cluster_fold`: its
    /// static generator less its direct entries). The levels a bare fold onto an explicit state is read at, at a state
    /// (`bare_mixing`).
    pub(crate) fn levels_at(&self, amplitudes: &[crate::field::Amplitude]) -> Vec<f64> {
        use crate::request::STATES;
        let mut h = self.far_hamiltonian(amplitudes, &self.running(), 0).remove(0);
        for cc in &self.carriers {
            let f = &self.light.frequencies[cc.frequency];
            let n = f.beams.len();
            for (x, form) in self.tones[cc.frequency].pairs.iter().enumerate() {
                let z = super::far::pair(&amplitudes[f.beams[x / n]], &amplitudes[f.beams[x % n]]).value;
                if z != C64::ZERO {
                    h.add_scaled(z, form);
                }
            }
        }
        let mut out: Vec<f64> = (0..STATES).map(|s| h[(s, s)].re).collect();
        for (k, c) in self.clusters.iter().enumerate().filter(|(_, c)| c.ground > 0) {
            if let Some((fold, ..)) = self.cluster_fold(k, amplitudes, true) {
                for p in 0..c.ground {
                    out[c.states[p]] += (fold.generator[(p, p)] - fold.direct[(p, p)]).re;
                }
            }
        }
        out
    }

    /// A bare fold onto an explicit state at a point (`bare_links`; far.rs # A fold onto an explicit state): the drive's
    /// coupling there, a = |Σ_l a_l X_l[e, w]| over its frequency's beams at their amplitudes `amplitudes`, across its
    /// two-photon distance there, d = z_e − E_w − ν, z_e the energy e's cluster's ground states read it at (model.rs
    /// `Model::read`) and E_w, both on the levels `levels` there (`levels_at`): its mixing a/|d|. Where the ratio 2a/|d| is
    /// below one (`converges`), the fold's error is its self-energy's next order, 2(a/d)² (`averaged`), a rate's relative
    /// error; where it is not, the fold does not exist there and its error is its whole.
    pub(crate) fn bare_mixing(&self, (c, e, w): (super::Component, usize, usize), amplitudes: &[crate::field::Amplitude], levels: &[f64]) -> f64 {
        let (f, tone) = (&self.light.frequencies[c.frequency], &self.tones[c.frequency]);
        let a = f.beams.iter().zip(&tone.couplings[0]).map(|(&l, x)| amplitudes[l].value * x[(e, w)]).sum::<C64>().abs();
        let cluster = self.clusters.iter().find(|k| k.states.contains(&e)).expect("an explicit state's cluster");
        let place = cluster.states.iter().position(|&s| s == e).expect("its place");
        a / (self.read(cluster, place, levels, true) - levels[w] - tone.nu[0]).abs()
    }

    /// The keep rule's objects of this model (ELIMINATION.md §6: its two kinds, and the photons' cross terms) as the
    /// model's units: every excited block a carrier drives that is not explicit (`Unit::Block`); every P-layer link of
    /// two states of the model's clusters at a beat of two classes (a carrier's co-rotating component or none: lab
    /// beats ν_d − ν_c, ±ν_c and zero) that its generator does not hold as it is (`Unit::Term`); every pair of photon
    /// classes landing in one cluster (`Unit::Coherence`, photon.rs `Model::coherence_units`). The kept ones left out.
    pub(crate) fn candidates(&self) -> Vec<super::Unit> {
        use crate::request::GROUND;
        let mut out: Vec<super::Unit> = vec![];
        for cluster in &self.clusters {
            let grounds = &cluster.states[..cluster.ground];
            for (b, block) in self.blocks().iter().enumerate() {
                if block.iter().any(|&e| cluster.states.contains(&(GROUND + e))) {
                    continue;
                }
                for cc in self.carriers.iter().filter(|cc| cc.folded.iter().any(|m| block.iter().any(|&e| grounds.iter().any(|&g| m[(GROUND + e, g)] != C64::ZERO)))) {
                    out.push(super::Unit::Block { block: b, component: super::Component { frequency: cc.frequency, counter: false }, grounds: ground_set(grounds) });
                }
            }
        }
        let classes: Vec<super::Beat> = std::iter::once(self.axes.zero()).chain(self.carriers.iter().map(|cc| cc.beat.clone())).collect();
        let mut beats: Vec<super::Beat> = vec![];
        for x in &classes {
            for y in &classes {
                let beat = x.plus(-1, y);
                if !beats.contains(&beat) {
                    beats.push(beat);
                }
            }
        }
        let states: Vec<usize> = self.clusters.iter().flat_map(|c| c.states.iter().copied()).collect();
        for beat in &beats {
            for &a in &states {
                for &b in &states {
                    if (a == b && beat.is_zero()) || (a >= GROUND && b >= GROUND) {
                        continue;
                    }
                    let unit = super::eliminate::term_unit(a, b, beat);
                    if !out.contains(&unit) && !self.in_generator(a, b, beat) {
                        out.push(unit);
                    }
                }
            }
        }
        out.extend(self.coherence_units());
        // The bare folds onto explicit states, each taken to the atom by its keep (`bare_links`).
        out.extend(self.bare_links().into_iter().map(|(component, excited, ground)| super::Unit::Bare { component, excited, ground }));
        // The evaluation's next orders where a fold has the normalized generator (`Unit::Series`, `Unit::Normalization`).
        let running = self.running();
        if (0..self.clusters.len()).any(|c| self.plan(c, &running).fourth.is_some()) {
            if self.series.order == 0 {
                out.push(super::Unit::Series);
            }
            out.push(super::Unit::Normalization { order: self.series.normalization + 1 });
        }
        out.retain(|u| !self.kept.contains(u));
        out
    }

    /// The candidates' errors at a trajectory state (`candidates`; ELIMINATION.md §6: the prior's function at a run's
    /// state): criterion one (`errors`, the closure's function: `Fold::criterion_one`) on the run's own fold of the whole
    /// atom at the state's point (`atom_fold`: the sample's cluster's eigenstates its run's), at the state and the offsets
    /// its amplitude spans -- the pattern its moving atom sees, 2 k_max v, and the sidebands of the turning couplings it
    /// carries over their distances from K's spectrum (`Fold::offsets`) --, each excited block's part on its
    /// `Unit::Block`, and the part of every sector the run leaves out -- of two clusters, or off its cluster's generator
    /// -- on its links' `Unit::Term`; and the evaluation against the definition (the cluster's static part,
    /// `Model::definition_fold`). The photons' cross terms on the run's photon classes, which land in every cluster, and
    /// on the left-out sectors' photons (`cross_terms`: of each pair landing in one, the dropped term's parts on the
    /// landing cluster's eigenstates and the state's, each over its combined frequency; v3 1, the secular approximation's
    /// first order), on the pair's `Unit::Coherence` -- or, of a sector's photons, on its links (`pair_links`). A
    /// relative rate error x moves the populations by at most x (absolute) and T by x (relative). Each bare fold onto an
    /// explicit state of the cluster (`bare_links`) at the state's point, its next order there or its whole where it does
    /// not exist there (`bare_error` of `bare_mixing`), on the populations of its two states, on `Unit::Bare`. And beside
    /// the units, per transition to a level outside the model (`Model::outside`), its error at the state (`errors`),
    /// booked so too.
    pub(crate) fn readings(&self, units: &[super::Unit], sample: &super::Sample, tables: Option<&super::tabulate::Tables>) -> (Vec<super::UnitError>, Vec<super::UnitError>) {
        use crate::request::{GROUND, STATES};
        use super::{Unit, UnitError};
        let mut out = vec![UnitError::NONE; units.len()];
        let book = |unit: &Unit, x: f64, out: &mut Vec<UnitError>| {
            if let Some(i) = units.iter().position(|u| u == unit) {
                let e = &mut out[i];
                (e.population, e.temperature) = (e.population + x, e.temperature + x);
            }
        };
        let c = &self.clusters[sample.cluster];
        let g = c.ground;
        let amplitudes = self.light.amplitudes(sample.at.r);
        let speed = crate::linalg::norm3(sample.at.kappa.map(|k| k * self.hbar_over_mass));
        let norm = crate::linalg::norm_sqr(&sample.psi);
        let blocks = self.blocks();
        let block_unit = |b: usize, k: usize| Unit::Block { block: b, component: super::Component { frequency: self.carriers[k].frequency, counter: false }, grounds: ground_set(&c.states[..g]) };
        // The run's fold of the whole atom, the state on its places; criterion one there, and the levels outside the model.
        let (atom, states) = self.atom_fold(&amplitudes);
        let place_of = |s: usize| states.iter().position(|&x| x == s).expect("a state of the atom's fold");
        let grounds = &states[..atom.ground];
        let mut on_atom = vec![C64::ZERO; atom.size()];
        for (i, &s) in c.states.iter().enumerate() {
            on_atom[place_of(s)] = sample.psi[i];
        }
        let outside: Vec<Outside> = self.outside_at(&amplitudes).iter().map(|(m, o)| Outside { slow: states.iter().map(|&s| m[s]).collect(), excited: (GROUND..STATES).map(|s| m[s] * o[s]).collect() }).collect();
        let errors = errors(&atom, &blocks, &on_atom, 2.0 * self.light.beams.iter().map(|b| b.wave_number).fold(0.0, f64::max) * speed, &outside);
        for &((b, class), x) in &errors.parts {
            if let Some(k) = class {
                book(&block_unit(b, k), x, &mut out);
            }
        }
        for (l, x) in &errors.links {
            book(&self.link_unit(grounds, l), *x, &mut out);
        }
        if let Some((fold, levels, far, left)) = self.cluster_fold(sample.cluster, &amplitudes, true) {
            // A part on a block -- or, None, on every block the state reaches --, split among its classes as `errors`
            // splits a block's (`Fold::state_block_parts`).
            let on_blocks = |on: Option<usize>, x: f64, out: &mut Vec<UnitError>| {
                for (b, class, part) in fold.state_block_parts(&blocks, on, &sample.psi, x) {
                    if let Some(k) = class {
                        book(&block_unit(b, k), part, out);
                    }
                }
            };
            // The evaluation against the definition (ELIMINATION v7; `Fold::evaluation_error`, `Fold::landing_error`): one
            // comparison at the run's orders -- the normalized generator, and the landing map a jump and the light's
            // switching on put the atom through (model.rs `Model::landed`) --, and at each next object the part it takes
            // away: the series' first order (`Unit::Series`) where the run takes order 0, the normalization's next
            // (`Unit::Normalization`); the rest (δK, the orders beyond, the Hermitian part's normalization) on the block
            // nearest the reference (`Fold::nearest`), which criterion one makes explicit. Where the definition does not
            // exist at the state (`settle`, `converges`), the evaluation is wrong by all of itself: its whole, a rate's
            // relative error one, on the blocks by the state's excited share, as the closure moves the block of the
            // largest part.
            let z0 = &self.reference_of(sample.cluster, &far)[..];
            let psi = &sample.psi;
            match self.definition_fold(sample.cluster, &amplitudes) {
                Some((definition, _)) if !definition.exists => on_blocks(None, 1.0, &mut out),
                Some((definition, whole)) => {
                    // Where the definition's landing does not exist (its left wave operators do not settle, `converges`),
                    // every landing is wrong by all of itself: the state's ground populations whole, at every order.
                    let unlanded = (0..g).map(|x| psi[x].norm_sqr()).sum::<f64>() / norm;
                    let against = |(defined, landing): &(Mat, Option<Mat>), orders: Orders| {
                        let landed = landing.as_ref().map_or(unlanded, |landing| Fold::landing_error(&fold.landing_evaluated(z0, orders), landing, g, psi));
                        Fold::evaluation_error(&fold.evaluated(&levels, z0, orders), defined, z0, psi) + landed
                    };
                    let defined = definition.definitions(&whole);
                    // The run's orders, every cluster's (assemble.rs, # The normalized generator: the series' first
                    // order of each place's own reference).
                    let own = self.series.orders();
                    let at = against(&defined, own);
                    let series = if own.series == 0 { (at - against(&defined, Orders { series: 1, ..own })).max(0.0) } else { 0.0 };
                    let normalization = (at - against(&defined, Orders { normalization: own.normalization + 1, ..own })).max(0.0);
                    if own.series == 0 {
                        book(&Unit::Series, series, &mut out);
                    }
                    book(&Unit::Normalization { order: own.normalization + 1 }, normalization, &mut out);
                    let nearest = fold.nearest(z0, Some(&left)).and_then(|e| blocks.iter().position(|b| b.contains(&e)));
                    if nearest.is_some() {
                        on_blocks(nearest, (at - series - normalization).max(0.0), &mut out);
                    }
                }
                None => {}
            }
        }
        // The photons' cross terms (criterion three, `cross_terms`) on the atom's fold: the run's photon classes of the
        // cluster, each landing cluster's spectrum its group's, at the state's coefficients on the cluster's eigenstates and
        // each one's photons (linear in the state), on the classes' `Unit::Coherence`; and the photons of the sectors the
        // run leaves out (`Fold::photons`), on their links (`pair_links`).
        if norm > 0.0 && units.iter().any(|u| matches!(u, Unit::Coherence { .. })) {
            let lights = self.running();
            let plan = self.plan(sample.cluster, &lights);
            let table = tables.and_then(|t| t.of(sample.cluster, &lights));
            let n = atom.size();
            let group = atom.places[place_of(c.states[0])];
            let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| atom.eigen.inverse[(p, i)] * on_atom[i]).sum()).collect();
            let source: Vec<usize> = (0..n).filter(|&p| atom.groups[p] == group).collect();
            let landings: Vec<Landing> = (0..atom.members.len()).map(|x| Landing::of(&atom, &(0..n).filter(|&q| atom.groups[q] == x).collect::<Vec<_>>())).collect();
            let classes = &self.classes[sample.cluster];
            let mut images: Vec<Images> = classes.iter().map(|class| Images { frequency: -self.omega(&class.beats[0]), scale: self.axes.size(&class.beats[0]), landing: atom.places[place_of(self.clusters[class.landing].states[0])], images: vec![std::array::from_fn(|_| vec![C64::ZERO; n]); n], blocks: vec![] }).collect();
            for &p in source.iter().filter(|&&p| coefficients[p] != C64::ZERO) {
                let r = atom.right(p);
                let local: Vec<C64> = c.states.iter().map(|&s| r[place_of(s)]).collect();
                let (photons, _) = self.photons_of(&plan, table, &sample.at, &local, super::photon::Gradients::None);
                for photon in photons {
                    let landing = &self.clusters[classes[photon.class].landing];
                    for (axis, u) in photon.amplitudes.iter().enumerate() {
                        for (i, &s) in landing.states.iter().enumerate() {
                            images[photon.class].images[p][axis][place_of(s)] = u[i];
                        }
                    }
                }
            }
            let emission: [Mat; 3] = std::array::from_fn(|a| Mat::from_fn(atom.ground, STATES, |x, y| self.emission[a][(grounds[x], y)]));
            images.extend(atom.photons(&emission, &source).0.into_iter().filter(|class| !class.blocks.is_empty()));
            for (a, b, x) in cross_terms(&images, &landings, &atom.eigen.values, &coefficients, norm, 0.0) {
                let on = pair_links(&atom, &images[a], &images[b]);
                if on.is_empty() {
                    let (first, second) = (&classes[a].beats[0], &classes[b].beats[0]);
                    let (lo, hi) = if first < second { (first.clone(), second.clone()) } else { (second.clone(), first.clone()) };
                    book(&Unit::Coherence { landing: self.clusters[classes[a].landing].states[0], first: lo, second: hi }, x, &mut out);
                }
                for (l, share) in on {
                    book(&self.link_unit(grounds, &l), x * share, &mut out);
                }
            }
        }
        // The bare folds onto this cluster's explicit states (`bare_links`): each at the state's point, its error there
        // (`bare_error` of `bare_mixing`: the coupling at the beams' amplitudes there, the two-photon distance on the
        // levels there) on the populations of its two states, on its `Unit::Bare`.
        let links: Vec<(super::Component, usize, usize)> = self.bare_links().into_iter().filter(|&(component, excited, ground)| c.states.contains(&excited) && units.contains(&Unit::Bare { component, excited, ground })).collect();
        if !links.is_empty() {
            let levels = self.levels_at(&amplitudes);
            let population = |s: usize| c.states.iter().position(|&x| x == s).map_or(0.0, |p| sample.psi[p].norm_sqr() / norm);
            for link in links {
                let (component, excited, ground) = link;
                let x = bare_error(self.bare_mixing(link, &amplitudes, &levels)) * (population(excited) + population(ground));
                book(&Unit::Bare { component, excited, ground }, x, &mut out);
            }
        }
        (out, errors.outside.iter().map(|&x| UnitError { population: x, temperature: x }).collect())
    }

    /// The transitions to the levels outside the model (`Model::outside`).
    pub(crate) fn outside_count(&self) -> usize {
        self.outside.len()
    }

    /// Per transition to a level outside the model (`Model::outside`) its mixing and its shift at a point of the beams'
    /// amplitudes `amplitudes`, per model state: m_s = Σ_i x_i(E) mixing_i[s] and o_s = Σ_i x_i(E) shift_i[s], E its
    /// frequency's coherent field there (request.rs `coordinates`).
    pub(crate) fn outside_at(&self, amplitudes: &[crate::field::Amplitude]) -> Vec<(Vec<f64>, Vec<f64>)> {
        let x: Vec<[f64; 9]> = self.light.frequencies.iter().map(|f| crate::request::coordinates(&f.field(amplitudes).value)).collect();
        let at = |w: usize, form: &[Vec<f64>; 9]| (0..crate::request::STATES).map(|s| (0..9).map(|i| x[w][i] * form[i][s]).sum()).collect::<Vec<f64>>();
        self.outside.iter().map(|(w, t)| (at(*w, &t.mixing), at(*w, &t.shift))).collect()
    }

    /// The model's prior (ELIMINATION.md §3–§4, the brief's "先验"): at the reach's own atoms `points` (position,
    /// velocity) the point's closure (`closure`) from what is kept already, the atoms together until no atom adds
    /// anything; and per drive its fold's route, at the point or at its bare levels (`bare_routes`: its own part, the
    /// rest within the share no output resolves at every atom, `estimate::unseen`). What it keeps, as the model's units:
    /// every drive folded at the point (`Unit::Bare`: its fold taken at the levels there), the explicit drives into the
    /// blocks (`Unit::Drive`), the P-layer links of two ground states (`Unit::Term`, their lab beat exactly), and every bare
    /// fold onto an explicit state whose error at an atom is beyond the share (`Unit::Bare`, `bare_mixing`).
    pub(crate) fn prior(&self, points: &[([f64; 3], [f64; 3])]) -> Vec<super::Unit> {
        use rayon::prelude::*;
        let blocks = self.blocks();
        let bare = &self.energies;
        let unseen = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        let mut kept = Kept::default();
        // The drives moved to the point, by their light's (frequency, counter-rotating) at the atom they were read at.
        let mut routes: Vec<((usize, bool), usize, usize)> = vec![];
        let mut sources: Vec<(usize, bool)> = vec![];
        // The photons' cross terms at the closure's fixed point (`start_cross_terms`): its last round's.
        let mut coherent: Vec<[(Option<usize>, usize, usize); 2]> = vec![];
        loop {
            let read: Vec<(Kept, Vec<((usize, bool), usize, usize)>, Vec<(usize, bool)>, Vec<[(Option<usize>, usize, usize); 2]>)> = points
                .par_iter()
                .map(|&(r, v)| {
                    let (levels, classes, sources) = system(&self.request, &self.light, r);
                    let closed = closure(&levels, &classes, &blocks, v, &kept);
                    let routes: Vec<((usize, bool), usize, usize)> = bare_routes(&closed.fold, bare, unseen).into_iter().map(|(c, e, g)| (sources[c], e, g)).collect();
                    // The photons' cross terms: of the run's classes, their coherences; of a sector the run leaves
                    // out, its links kept, as the closure keeps them (`start_cross_terms`).
                    let (coherent, crossed) = start_cross_terms(&closed.fold, &self.emission);
                    let mut kept = closed.kept;
                    for l in crossed {
                        if !kept.links.contains(&l) && !kept.links.contains(&l.reversed()) {
                            kept.links.push(l);
                        }
                    }
                    (kept, routes, sources, coherent)
                })
                .collect();
            let before = (kept.blocks.len(), kept.links.len());
            coherent.clear();
            for (k, r, s, c) in read {
                for pair in c {
                    if !coherent.contains(&pair) {
                        coherent.push(pair);
                    }
                }
                for b in k.blocks {
                    if !kept.blocks.contains(&b) {
                        kept.blocks.push(b);
                    }
                }
                for l in k.links {
                    if !kept.links.contains(&l) {
                        kept.links.push(l);
                    }
                }
                for d in r {
                    if !routes.contains(&d) {
                        routes.push(d);
                    }
                }
                sources = s;
            }
            if (kept.blocks.len(), kept.links.len()) == before {
                break;
            }
        }
        let mut units = vec![];
        let component = |c: usize| super::Component { frequency: sources[c].0, counter: sources[c].1 };
        for &((frequency, counter), e, g) in &routes {
            let drive = (super::Component { frequency, counter }, crate::request::GROUND + e, g);
            if !counter && self.driven.contains(&drive) {
                units.push(super::Unit::Bare { component: drive.0, excited: drive.1, ground: drive.2 });
            }
        }
        // A cross term's unit (`Unit::Coherence`): its two classes' exact beats in the model's frames -- the light's
        // −ν (none of an explicit state's own decay) and the landing state's frame less the source's, as photon.rs makes a
        // class's --, landing in one cluster.
        let frame = |state: usize| self.clusters.iter().find_map(|c| c.states.iter().position(|&x| x == state).map(|i| c.frames[i].clone()));
        let beat = |(class, from, to): (Option<usize>, usize, usize)| -> Option<super::Beat> {
            let light = class.map_or(self.axes.zero(), |c| self.frequency(component(c)).negated());
            Some(light.plus(1, &frame(to)?).plus(-1, &frame(from)?))
        };
        for [x, y] in &coherent {
            let (Some(a), Some(b)) = (beat(*x), beat(*y)) else { continue };
            if a == b || self.membership[x.2] != self.membership[y.2] {
                continue;
            }
            let (first, second) = if a < b { (a, b) } else { (b, a) };
            units.push(super::Unit::Coherence { landing: self.clusters[self.membership[x.2]].states[0], first, second });
        }
        for l in &kept.links {
            match l.classes {
                (None, Some(c)) if l.a >= crate::request::GROUND => units.push(super::Unit::Drive { component: component(c), excited: l.a, ground: l.b }),
                (emitted, absorbed) if l.a < crate::request::GROUND && l.b < crate::request::GROUND => {
                    let beat = |c: Option<usize>| c.map_or(self.axes.zero(), |c| self.frequency(component(c)));
                    units.push(super::eliminate::term_unit(l.a, l.b, &beat(emitted).plus(-1, &beat(absorbed))));
                }
                _ => {}
            }
        }
        // The bare folds onto explicit states read at the atoms' starts as the readings read them where the atoms go
        // (`bare_links`, `bare_error` of `bare_mixing`), each state full: beyond the share at one of them, to the atom
        // before the first attempt (`Unit::Bare`).
        let links = self.bare_links();
        if !links.is_empty() {
            let worst = points
                .par_iter()
                .map(|&(r, _)| {
                    let amplitudes = self.light.amplitudes(r);
                    let levels = self.levels_at(&amplitudes);
                    links.iter().map(|&l| bare_error(self.bare_mixing(l, &amplitudes, &levels))).collect::<Vec<f64>>()
                })
                .reduce(|| vec![0.0; links.len()], |a, b| a.iter().zip(&b).map(|(x, y)| x.max(*y)).collect());
            for (&(component, excited, ground), &x) in links.iter().zip(&worst) {
                if !(x <= share()) {
                    units.push(super::Unit::Bare { component, excited, ground });
                }
            }
        }
        // Two links can be one unit (a static coupling and one class's own at the lab beat zero).
        let mut distinct: Vec<super::Unit> = vec![];
        for unit in units {
            if !distinct.contains(&unit) {
                distinct.push(unit);
            }
        }
        distinct
    }
}

/// A set of ground states as `Unit::Block` holds it: bit g for ground state g.
pub(crate) fn ground_set(grounds: &[usize]) -> u32 {
    grounds.iter().fold(0, |set, &g| set | 1 << g)
}

/// The excited blocks of the states' labels (`blocks`).
pub(crate) fn excited_blocks(states: &[String]) -> Vec<Vec<usize>> {
    use crate::request::{Level, GROUND};
    let mut out: Vec<(String, Vec<usize>)> = vec![];
    for (e, label) in states[GROUND..].iter().enumerate() {
        let l = Level::parse(label).expect("a request's state label");
        let name = format!("{:?}{}", l.manifold, l.f);
        match out.iter_mut().find(|b| b.0 == name) {
            Some(b) => b.1.push(e),
            None => out.push((name, vec![e])),
        }
    }
    out.into_iter().map(|b| b.1).collect()
}

/// A P-layer link the model keeps (a sector's coupling kept, # Sectors; the explicit drives among them): an entry
/// a ← b of P (a state: a ground state's index, an excited one's GROUND + its index) at its lab beat, made by the
/// emitted and the absorbed class (None: a direct entry, H0's or the excited block's).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Link {
    pub a: usize,
    pub b: usize,
    pub classes: (Option<usize>, Option<usize>),
}

impl Link {
    /// The same coupling's other entry, b ← a, its classes exchanged (at the opposite beat): one link with it.
    pub(crate) fn reversed(&self) -> Link {
        Link { a: self.b, b: self.a, classes: (self.classes.1, self.classes.0) }
    }

    /// Its lab beat Ω (∝ e^{−iΩt}): the absorbed class's frequency less the emitted one's.
    pub(crate) fn beat(&self, classes: &[Class]) -> f64 {
        self.classes.1.map_or(0.0, |c| classes[c].nu) - self.classes.0.map_or(0.0, |c| classes[c].nu)
    }
    /// The size of its beat's terms (`equal`): the two classes' frequencies'.
    pub(crate) fn size(&self, classes: &[Class]) -> f64 {
        self.classes.1.map_or(0.0, |c| classes[c].nu.abs()) + self.classes.0.map_or(0.0, |c| classes[c].nu.abs())
    }

    /// Its strength at the point (rad/µs; the frames' order, cluster.rs `frames`): an explicit drive's coupling, the
    /// class's beams summed, |C_c(e, g)|; a coupling of two P states through two classes its second-order entry through
    /// the excited states the partition folds (all but `explicit`), |Σ_e C₀(e, a)* C₁(e, b) / (ν₁ + E_b − E_e + (i/2)Γ_e)|
    /// on the levels' diagonals -- the entry the fold makes, to its leading order --; none of a P state with itself.
    pub(crate) fn strength(&self, classes: &[Class], levels: &Levels, explicit: &[usize]) -> f64 {
        let ground = levels.ground.rows;
        let coupling = |c: usize, e: usize, g: usize| classes[c].beams.iter().map(|b| b.coupling[(e, g)]).sum::<C64>();
        match self.classes {
            (None, Some(c)) if self.a >= ground && self.b < ground => coupling(c, self.a - ground, self.b).abs(),
            (Some(c), None) if self.b >= ground && self.a < ground => coupling(c, self.b - ground, self.a).abs(),
            (Some(emitted), Some(absorbed)) if self.a < ground && self.b < ground => (0..levels.excited.rows)
                .filter(|e| !explicit.contains(e))
                .map(|e| {
                    let distance = C64::new(classes[absorbed].nu + levels.ground[(self.b, self.b)].re - levels.excited[(e, e)].re, 0.5 * levels.decay[(e, e)].re);
                    coupling(emitted, e, self.a).conj() * coupling(absorbed, e, self.b) / distance
                })
                .sum::<C64>()
                .abs(),
            _ => 0.0,
        }
    }
}

/// What a model keeps explicit (ELIMINATION.md §6: the keep rule's two kinds of objects): excited blocks, and P-layer
/// links -- the explicit components, the explicit drives into the blocks with them.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Kept {
    pub blocks: Vec<usize>,
    pub links: Vec<Link>,
}

/// A point's closure (criterion one and the existence rule at one atom, ELIMINATION.md §3–§4 with v2, v3 and v9): the kept set from
/// `start` on, made larger until the point needs nothing more, with its fold.
#[derive(Clone, Debug)]
pub(crate) struct Closed {
    pub kept: Kept,
    pub fold: Fold,
}

/// The P states (labels as in `Link`) of a kept set, the frames from its links at their strengths (cluster.rs `frames`:
/// the strongest static; `Link::strength`) and its groups (`components` of the links, each explicit block's states one
/// group with them: a block is moved whole, its sublevels coupled by the Zeeman and tensor shifts): the ground states,
/// then the kept blocks' states.
fn slow_of(blocks: &[Vec<usize>], kept: &Kept, classes: &[Class], levels: &Levels) -> Slow {
    let ground = levels.ground.rows;
    let explicit: Vec<usize> = kept.blocks.iter().flat_map(|&b| blocks[b].iter().copied()).collect();
    let place = |s: usize| if s < ground { Some(s) } else { explicit.iter().position(|&e| e == s - ground).map(|i| ground + i) };
    let edges: Vec<(usize, usize, Sized, f64)> = kept.links.iter().filter_map(|l| Some((place(l.a)?, place(l.b)?, Sized { value: l.beat(classes), size: l.size(classes) }, l.strength(classes, levels, &explicit)))).collect();
    let n = ground + explicit.len();
    let sized = super::cluster::frames(n, &edges, &Sized { value: 0.0, size: 0.0 });
    let within = kept.blocks.iter().flat_map(|&b| blocks[b].windows(2).filter_map(|w| Some((place(ground + w[0])?, place(ground + w[1])?))));
    let groups = components(n, edges.iter().map(|e| (e.0, e.1)).chain(within));
    Slow { explicit, frames: sized.iter().map(|f| f.value).collect(), sizes: sized.iter().map(|f| f.size).collect(), groups }
}

/// The closure at one point (`Closed`): from `start`, at the atom's velocity v (µm/µs) --
/// 1. the fold in the frames of the kept links, the turning couplings it does not keep folded as its sectors (# Sectors);
///    where it does not exist (`settle`), the object whose fold does not goes up one level, the one of the largest part
///    of the ground-like eigenstates' share over all of them: where the static part's fold does not exist, an excited
///    block's of the excited share X, made explicit; where only the sectors' does not, a link's of the sectors'
///    populations (`Fold::still_exists`, `Fold::sector_shares`), kept;
/// 2. criterion one: every ground-like eigenstate's error 2ρ_p + ρ_p² within the share (`Fold::criterion_one`, `rate`:
///    the number the readings book there) -- ρ_p its remainder at the offsets its amplitude spans (`Fold::offsets`: the
///    pattern a moving atom sees, 2 k_max v, whole, and the sidebands of the turning couplings the model carries, each
///    over its distance from the folded block's spectrum), and the whole of every sector the run leaves out
///    (`Fold::sector_parts`), in squares --; beyond it the objects with the largest parts go up
///    until the rest is within it: near a resonance of a folded block its part grows without bound, and that block is
///    made explicit; near a two-photon resonance a sector's does (δ → 0), and that link is kept;
/// - a block made explicit for an eigenstate is explicit at the replica that holds its part and in the P component the
///   eigenstate holds (`components`): the drives into it at that replica from that component's ground states are kept
///   (its frame). A drive from another component -- the other ground hyperfine level, 6.8 GHz away -- turns at that
///   component's own frequency, where the block is a Q state folded like any other (ELIMINATION §2–§3, v2 5); a block
///   explicit in one component is joined by another's drives only where that component's own part asks for it;
/// - a link kept is a P-layer coupling of its two states and two classes: the frames take it static where its group's
///   tree holds it (cluster.rs `frames`), else the run carries it as a tone;
/// until nothing more is kept. Each round only adds. (The photons' cross terms keep no block: the prior reads them on the
/// fold this ends with, `start_cross_terms`, and keeps there the links of a left-out sector's photons.)
pub(crate) fn closure(levels: &Levels, classes: &[Class], blocks: &[Vec<usize>], velocity: [f64; 3], start: &Kept) -> Closed {
    let ground = levels.ground.rows;
    let share = share();
    let speed = (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt();
    let fastest = classes.iter().flat_map(|c| c.beams.iter()).map(|b| (b.wave[0] * b.wave[0] + b.wave[1] * b.wave[1] + b.wave[2] * b.wave[2]).sqrt()).fold(0.0, f64::max);
    let pattern = 2.0 * fastest * speed;
    let mut kept = start.clone();
    loop {
        let slow = slow_of(blocks, &kept, classes, levels);
        // An explicit block's drives at its frame from every ground state of its group, as the group grows: they are
        // static in the frames (A's), and the run's cluster holds them as it is.
        let grown: Vec<Link> = slow
            .explicit
            .iter()
            .enumerate()
            .flat_map(|(i, &e)| (0..classes.len()).flat_map(move |c| (0..ground).map(move |g| (i, e, c, g))))
            .filter(|&(i, e, c, g)| slow.groups[g] == slow.groups[ground + i] && equal(classes[c].nu + slow.frames[g] - slow.frames[ground + i], classes[c].nu.abs() + slow.sizes[g] + slow.sizes[ground + i]) && classes[c].beams.iter().any(|beam| beam.coupling[(e, g)] != C64::ZERO))
            .map(|(_, e, c, g)| Link { a: ground + e, b: g, classes: (None, Some(c)) })
            .filter(|l| !kept.links.contains(l))
            .collect();
        if !grown.is_empty() {
            kept.links.extend(grown);
            continue;
        }
        let replicas = Replicas::of(classes, &slow, ground);
        let fold = fold(levels, classes, &slow, &replicas, Some(&|l: &Link| kept.links.contains(l)));
        let n = fold.size();
        let ground_like: Vec<usize> = (0..n).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
        // An object as a move of eigenstate p's (a block of no class -- the explicit states' own couplings -- none).
        let movable = |o: Object, p: usize| match o {
            Object::Block { block, replica, class } => class.map(|c| Move::Block(block, replica, c, p)),
            Object::Link(link) => Some(Move::Link(link)),
        };
        let mut moves: Vec<Move> = vec![];
        if !fold.exists {
            // One move a round: the largest part of any ground-like eigenstate's share -- of an excited block's where the
            // static part's fold does not exist, of a sector's link where only the sectors' does not (`Fold::still_exists`):
            // the blocks' parts of its excited share by replica and class (`Fold::block_parts`), the links' of its
            // sectors' (`Fold::sector_shares`).
            let mut largest: Option<(Move, f64)> = None;
            for &p in &ground_like {
                let blocked = fold.block_parts(blocks, p, None).1.into_iter().filter_map(|(block, replica, class, x)| Some((movable(Object::Block { block, replica, class }, p)?, x)));
                let linked = fold.sector_shares(&fold.right(p)).into_iter().map(|(link, x)| (Move::Link(link), x));
                for (m, x) in blocked.chain(linked) {
                    if matches!(m, Move::Link(_)) == fold.still_exists && x > largest.as_ref().map_or(0.0, |l| l.1) {
                        largest = Some((m, x));
                    }
                }
            }
            // Where only the sectors' fold does not exist, every link whose own expansion does not (`sector_mixings`)
            // goes up at once -- each an object that fails --, else the largest share's.
            let failing: Vec<Link> = match fold.still_exists {
                true => ground_like.iter().flat_map(|&p| fold.sector_mixings(p)).filter(|x| !converges(2.0 * x.1)).map(|x| x.0).collect(),
                false => vec![],
            };
            match failing.is_empty() {
                true => moves.extend(largest.map(|l| l.0)),
                false => failing.into_iter().for_each(|l| moves.push(Move::Link(l))),
            }
        } else {
            // Each eigenstate's error by criterion one (`Fold::criterion_one`, `rate`: the readings' number at it), its
            // objects' parts of ρ² largest first, until what is left is within the share.
            for &p in &ground_like {
                let (total, parts) = fold.criterion_one(blocks, p, pattern, 1.0, &|_| 1.0);
                if !(rate(total.sqrt()) > share) {
                    continue;
                }
                let mut parts: Vec<(Move, f64)> = parts.into_iter().filter_map(|(o, x)| Some((movable(o, p)?, x))).collect();
                parts.sort_by(|x, y| y.1.total_cmp(&x.1));
                let mut rest = total;
                for (m, x) in parts {
                    if !(rate(rest.max(0.0).sqrt()) > share) {
                        break;
                    }
                    rest -= x;
                    if !moves.iter().any(|o| o.same(&m)) {
                        moves.push(m);
                    }
                }
            }
        }
        let mut grew = false;
        for m in moves {
            match m {
                Move::Block(b, j, c, p) => {
                    // The class's drives at the replica that holds the part, from the ground states of the eigenstate's group.
                    let rep = &fold.replicas[j];
                    let held = slow.groups[fold.dominant(p)];
                    let links: Vec<Link> = blocks[b]
                        .iter()
                        .flat_map(|&e| (0..ground).map(move |g| (e, g)))
                        .filter(|&(e, g)| slow.groups[g] == held && replicas.driven[c][g] < replicas.frequencies.len() && equal(replicas.frequencies[replicas.driven[c][g]] - rep.frequency, replicas.scales[replicas.driven[c][g]] + rep.scale) && classes[c].beams.iter().any(|beam| beam.coupling[(e, g)] != C64::ZERO))
                        .map(|(e, g)| Link { a: ground + e, b: g, classes: (None, Some(c)) })
                        .filter(|l| !kept.links.contains(l))
                        .collect();
                    if !links.is_empty() {
                        if !kept.blocks.contains(&b) {
                            kept.blocks.push(b);
                        }
                        kept.links.extend(links);
                        grew = true;
                    }
                }
                Move::Link(link) => {
                    if !kept.links.contains(&link) && !kept.links.contains(&link.reversed()) {
                        kept.links.push(link);
                        grew = true;
                    }
                }
            }
        }
        if grew {
            continue;
        }
        return Closed { kept, fold };
    }
}

/// An object of criterion one (`Fold::criterion_one`): an excited block for one class at a replica (class None: the
/// explicit states' own couplings into it), or a sector's link.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Object {
    Block { block: usize, replica: usize, class: Option<usize> },
    Link(Link),
}

/// An eigenstate's error by criterion one among its objects (`Fold::criterion_one`): of its ρ² `total` and the objects'
/// parts of it, the rate's error of ρ (`rate`), each object's part's share of the parts.
fn split(total: f64, parts: Vec<(Object, f64)>) -> impl Iterator<Item = (Object, f64)> {
    let (error, sum) = (rate(total.sqrt()), parts.iter().map(|x| x.1).sum::<f64>());
    parts.into_iter().filter(move |_| sum > 0.0).map(move |(o, x)| (o, error * x / sum))
}

/// A closure's move (`closure`): an excited block made explicit for a class at a replica (block, replica, class, the
/// eigenstate whose part it is), or a link kept.
#[derive(Clone, Copy, Debug)]
enum Move {
    Block(usize, usize, usize, usize),
    Link(Link),
}

impl Move {
    /// Whether two moves do one thing: a block for one class, or one link either way round.
    fn same(&self, other: &Move) -> bool {
        match (self, other) {
            (Move::Block(b, _, c, _), Move::Block(d, _, e, _)) => b == d && c == e,
            (Move::Link(l), Move::Link(m)) => l == m || *l == m.reversed(),
            _ => false,
        }
    }
}

/// The classes' shares of a replica's part on an excited block (ELIMINATION.md §6: a block is kept for one class of the
/// light, `Unit::Block`): per class its feeds' responses on the block's states `on` (`Fold::feed_responses`, Q's places
/// `q`), ‖R W_feed x|_on‖², over their sum -- their cross terms the classes' interference, no class's alone. None where
/// nothing of the replica reaches the block.
fn class_shares(responses: &[(Option<usize>, Vec<C64>)], q: &[usize], on: &[usize]) -> Vec<(Option<usize>, f64)> {
    let mut out: Vec<(Option<usize>, f64)> = vec![];
    for (class, v) in responses {
        let size: f64 = q.iter().zip(v).filter(|(e, _)| on.contains(e)).map(|(_, z)| z.norm_sqr()).sum();
        match out.iter_mut().find(|o| o.0 == *class) {
            Some(o) => o.1 += size,
            None => out.push((*class, size)),
        }
    }
    let total: f64 = out.iter().map(|o| o.1).sum();
    if !(total > 0.0) {
        return vec![];
    }
    out.iter_mut().for_each(|o| o.1 /= total);
    out
}

/// The components of n places joined by the pairs `joined`: per place the least place of its component.
fn components(n: usize, joined: impl Iterator<Item = (usize, usize)>) -> Vec<usize> {
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(parent: &mut [usize], mut x: usize) -> usize {
        while parent[x] != x {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        x
    }
    for (a, b) in joined {
        let (a, b) = (root(&mut parent, a), root(&mut parent, b));
        parent[a.max(b)] = a.min(b);
    }
    (0..n).map(|x| root(&mut parent, x)).collect()
}

/// A model's error at a state (ELIMINATION.md §6, v2, v3): per excited block, for one class of the light (None: the
/// explicit states' own couplings into it), the part its keep would take away (the fold's terms, each split among the
/// blocks that take it away); per link of a sector the run leaves out, its part of criterion one. (The photons' cross
/// terms are read apart, by `cross_terms`.)
#[derive(Clone, Debug)]
pub(crate) struct Errors {
    pub parts: Vec<((usize, Option<usize>), f64)>,
    /// Per link of a sector the run leaves out, its part of criterion one (`Fold::criterion_one`).
    pub links: Vec<(Link, f64)>,
    /// Per transition to a level outside the model its error (`outside_errors`): no object's part, nothing keeps it.
    pub outside: Vec<f64>,
}

/// One transition to a level outside the model at a state's point (request.rs `Transition`, `Model::outside_at`): its
/// mixing m_s -- the population a state s holds in the level, (coupling/detuning)² summed over its sublevels -- per P
/// state; and per excited state the error of the level that transition's second-order shift o_e puts it at, m_e o_e
/// (rad/µs: the term wrong by its own small parameter, relative).
pub(crate) struct Outside {
    pub slow: Vec<f64>,
    pub excited: Vec<f64>,
}

/// The errors of the transitions to the levels outside the model at a state ψ (P amplitudes; `errors`), per transition,
/// each as criterion one reads what the model holds:
/// - a P state s leaks its population into the level: |ψ_s|²/‖ψ‖² m_s;
/// - the excited states the fold eliminates hold their levels wrong by δK = diag(m_e o_e), which acts on the slow
///   dynamics through the fold's response, as any error of the excited block does: of each eigenstate p of the fold
///   (`fold`, with its share w_p of ψ) ρ_p = ‖R(λ_p) δK T r_p‖/‖T r_p‖ over the replicas in squares -- `Fold::remainder`'s
///   ratio, this error in its stead --, a rate's relative error 2ρ_p + ρ_p² (`rate`), Σ_p w_p (2ρ_p + ρ_p²). Not by the excited states' populations:
///   a transition that moves a level away from the light lowers its population and raises the error it makes.
/// The response T r_p of each eigenstate and replica is made once, then a resolvent per transition that is not zero there.
pub(crate) fn outside_errors(fold: &Fold, weight: &[f64], psi: &[C64], outside: &[Outside]) -> Vec<f64> {
    let norm = crate::linalg::norm_sqr(psi);
    let mut out: Vec<f64> = outside.iter().map(|o| if norm > 0.0 { psi.iter().zip(&o.slow).map(|(z, m)| z.norm_sqr() * m).sum::<f64>() / norm } else { 0.0 }).collect();
    for p in (0..fold.size()).filter(|&p| weight[p] > 0.0) {
        let (r, lambda) = (fold.right(p), fold.eigen.values[p]);
        let responses: Vec<(C64, Vec<C64>)> = fold.replicas.iter().map(|rep| {
            let sigma = lambda + C64::real(rep.frequency);
            (sigma, rep.k.resolvent(sigma, &rep.w.apply(&r)))
        }).collect();
        let held: f64 = responses.iter().map(|(_, x)| crate::linalg::norm_sqr(x)).sum();
        if !(held > 0.0) {
            continue;
        }
        for (o, error) in outside.iter().zip(out.iter_mut()) {
            let mut moved = 0.0;
            for (rep, (sigma, x)) in fold.replicas.iter().zip(&responses) {
                let y: Vec<C64> = rep.q.iter().zip(x).map(|(&e, z)| *z * o.excited[e]).collect();
                if y.iter().any(|z| *z != C64::ZERO) {
                    moved += crate::linalg::norm_sqr(&rep.k.resolvent(*sigma, &y));
                }
            }
            *error += weight[p] * rate((moved / held).sqrt());
        }
    }
    out
}

/// The model's error at a state ψ (P amplitudes in the frames), the frequency range its amplitude spans Δ (rad/µs:
/// the pattern a moving atom sees, 2 k_max v, `closure`), at a run's state (`Model::readings`).
/// - Criterion one (`Fold::criterion_one`, the closure's own function): of each eigenstate the state holds, its ρ_p at
///   the state's populations -- the remainder at the offsets Δ its amplitude spans, weighted by the shares of the state
///   there, and the sectors the run leaves out at its own share --, a rate's relative error 2ρ_p + ρ_p² (`rate`; at
///   ψ = r_p exactly what the closure judges there); each object's part of ρ² is its part: an excited block's, a link's.
/// - The transient (`Fold::transients`): after the next jump the atom lands in a bare ground state -- at the state's
///   own ground populations p_g --, whose waiting time the model takes from its slow part alone with the threshold law
///   P(T > t) = min(1, ‖y(t)‖²) (step.rs `jump`). The exact one is longer by τ and the law's clamp leaves out
///   (N − 1 − ln N)/γ where the slow part's norm N = ‖(x₀, T x₀)‖² is above one (γ its decay there): the rate's relative
///   error is |τ + clamp| over the model's mean wait, N/γ or (1 + ln N)/γ. Its part goes to the blocks in proportion to
///   their share of the landing's excited amplitude T x₀: making a block explicit takes its fast part into the dynamics.
///   Where the landing's slow part does not exist, its whole (`converges`: the one failure).
/// - The levels outside the model (request.rs `Transition`, `outside`): each transition's second-order term errs by its
///   own small parameter m = (coupling/detuning)²: on a P state the population it leaks into the level, on an excited
///   state the fold eliminates the level error m o it makes, through the fold's response as criterion one reads one
///   (`outside_errors`). Irreducible: those levels are not in the model, and no object's part.
/// The photons' cross terms are `cross_terms`' (at the start `start_cross_terms`, at a run's state on its photon
/// classes, `Model::readings`). A relative rate error x moves the populations by at most x (absolute) and T, which the light sets
/// through its rates and shifts, by x relative: the terms add on both.
/// The sectors the run leaves out are criterion one's, on the fold that makes them at the state's point (`Fold::sectors`;
/// # Sectors; at a run's state the run's own fold of the whole atom, `Model::readings`).
pub(crate) fn errors(fold: &Fold, blocks: &[Vec<usize>], psi: &[C64], pattern: f64, outside: &[Outside]) -> Errors {
    let n = fold.size();
    let ground = fold.ground;
    let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| fold.eigen.inverse[(p, i)] * psi[i]).sum()).collect();
    let total: f64 = coefficients.iter().map(|z| z.norm_sqr()).sum();
    let weight: Vec<f64> = coefficients.iter().map(|z| if total > 0.0 { z.norm_sqr() / total } else { 0.0 }).collect();
    let block_of = |e: usize| blocks.iter().position(|b| b.contains(&e));
    let mut parts: Vec<((usize, Option<usize>), f64)> = vec![];
    let add = |object: (usize, Option<usize>), x: f64, parts: &mut Vec<((usize, Option<usize>), f64)>| {
        if x > 0.0 {
            match parts.iter_mut().find(|p| p.0 == object) {
                Some(p) => p.1 += x,
                None => parts.push((object, x)),
            }
        }
    };
    // Criterion one at each eigenstate, at the populations there (`Fold::criterion_one`: the remainder at the pattern's
    // offsets at the eigenstate's own population, each carried sector's sideband at its source's times its share, and
    // the sectors the run leaves out at its own), its error `rate` of its ρ on its objects by their parts (`split`).
    let mut links: Vec<(Link, f64)> = vec![];
    let on_link = |l: Link, x: f64, links: &mut Vec<(Link, f64)>| match links.iter_mut().find(|k| k.0 == l || k.0 == l.reversed()) {
        Some(k) => k.1 += x,
        None => links.push((l, x)),
    };
    for p in 0..n {
        // Where a sector's own expansion does not exist at the eigenstate (`converges`: its first-order mixing 2m ≥ 1, or
        // no number; `Fold::sector_mixings`), the eigenstate's error is all of it, on the links whose expansion does not
        // exist -- the objects the closure keeps there (`closure`).
        if weight[p] > 0.0 {
            let failing: Vec<Link> = fold.sector_mixings(p).into_iter().filter(|x| !converges(2.0 * x.1)).map(|x| x.0).collect();
            if !failing.is_empty() {
                for &l in &failing {
                    on_link(l, weight[p] / failing.len() as f64, &mut links);
                }
                continue;
            }
        }
        let (total, each) = fold.criterion_one(blocks, p, pattern, weight[p], &|q| weight[q]);
        for (o, x) in split(total, each) {
            match o {
                Object::Block { block, class, .. } => add((block, class), x, &mut parts),
                Object::Link(l) => on_link(l, x, &mut links),
            }
        }
    }
    // The transient after the next jump, from the state's own ground populations. Where the landing's slow part does not
    // exist -- the left wave operators' iteration does not contract (`settle`, `converges`), or their overlap with the
    // right ones is singular --, its error is its whole, the landing's rate wrong by all of itself (`converges`: the one
    // failure), on the blocks by the bare landing's excited amplitude.
    let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
    let metric = fold.metric();
    let landings: Vec<usize> = (0..ground).filter(|&g| psi[g].norm_sqr() > 0.0).collect();
    let bares: Vec<Vec<C64>> = landings.iter().map(|&g| (0..n).map(|i| if i == g { C64::ONE } else { C64::ZERO }).collect()).collect();
    let landed = if landings.is_empty() { Some(vec![]) } else { fold.left_operators().and_then(|left| fold.transients(&left, &bares)) };
    for (i, &g) in landings.iter().enumerate() {
        let population = psi[g].norm_sqr() / norm;
        let (x0, x) = match &landed {
            None => (bares[i].clone(), population),
            Some(landed) => {
                let (x0, tau) = &landed[i];
                let held = crate::linalg::inner(x0, &metric.apply(x0)).re;
                let big = crate::linalg::norm_sqr(x0) + held;
                let mx: Vec<C64> = {
                    let hx = fold.generator.apply(x0);
                    let mut v = hx.clone();
                    let m = metric.apply(&hx);
                    v.iter_mut().zip(&m).for_each(|(a, b)| *a += *b);
                    v
                };
                let gamma = -2.0 * crate::linalg::inner(x0, &mx).im / big;
                if !(gamma > 0.0) {
                    continue;
                }
                let (wait, clamp) = if big <= 1.0 { (big / gamma, 0.0) } else { ((1.0 + big.ln()) / gamma, (big - 1.0 - big.ln()) / gamma) };
                (x0.clone(), population * (tau + clamp).abs() / wait)
            }
        };
        // Its parts by the blocks' shares of the landing's excited amplitude, replica by replica, each split among its
        // classes (`class_shares`).
        let mut shares = vec![vec![0.0; blocks.len()]; fold.replicas.len()];
        for (j, rep) in fold.replicas.iter().enumerate() {
            let y = rep.t.apply(&x0);
            for (i, &e) in rep.q.iter().enumerate() {
                if let Some(b) = block_of(e) {
                    shares[j][b] += y[i].norm_sqr();
                }
            }
        }
        let sum: f64 = shares.iter().flatten().sum();
        if sum > 0.0 {
            let coefficients: Vec<C64> = (0..n).map(|p| (0..n).map(|i| fold.eigen.inverse[(p, i)] * x0[i]).sum()).collect();
            for (j, row) in shares.iter().enumerate() {
                let mut responses = None;
                for (b, &share) in row.iter().enumerate().filter(|(_, s)| **s > 0.0) {
                    let responses = responses.get_or_insert_with(|| fold.feed_responses(j, &coefficients));
                    for (class, part) in class_shares(responses, &fold.replicas[j].q, &blocks[b]) {
                        add((b, class), x * share * part / sum, &mut parts);
                    }
                }
            }
        }
    }
    let outside = outside_errors(fold, &weight, psi, outside);
    Errors { parts, links, outside }
}

// ---------------------------------------------------------------------------------------------------------------------
// Criterion three: the photons' cross terms.

/// A landing's spectrum (criterion three, `cross_terms`): the eigenvalues λ_q of the fold its jumps land in, in its
/// frames, and its left eigenvectors l_q (the eigenvectors' inverse's rows) on the places the photons' vectors are on.
#[derive(Clone, Debug)]
pub(crate) struct Landing {
    pub values: Vec<C64>,
    pub left: Mat,
}

impl Landing {
    /// The spectrum of the eigenstates `states` of a fold (all of its own where its jumps land in one cluster, a group's
    /// where the fold is the whole atom's), on all of its places.
    pub(crate) fn of(fold: &Fold, states: &[usize]) -> Landing {
        let n = fold.size();
        Landing { values: states.iter().map(|&q| fold.eigen.values[q]).collect(), left: Mat::from_fn(states.len(), n, |a, x| fold.eigen.inverse[(states[a], x)]) }
    }
}

/// One class of photons (criterion three; photon.rs, # Sources and classes): its frequency κ = −ω̃ -- ω̃ the frame
/// frequency its entries turn at, e^{+iω̃t}, so that the photon's lab frequency is κ less the landing state's frame,
/// up to the class's common phase --, the landing it lands in (an index into the landings), and per source eigenstate
/// p its photon u_a = J_a r_p (a = x, y, z) on the landing's places. Of photons the run leaves out -- those of the
/// sectors it does not carry (`Fold::photons`) --, the sectors' blocks it holds photons of (the sector, among the fold's
/// `sectors`, the source eigenstate and the target group), in the order they were made, whose links' keep takes them
/// into the run (`Fold::class_links`, read only for a pair beyond the share); none of the photons the run has.
#[derive(Clone, Debug)]
pub(crate) struct Images {
    pub frequency: f64,
    /// The size of the terms its frequency is the sum of (`equal`).
    pub scale: f64,
    pub landing: usize,
    pub images: Vec<[Vec<C64>; 3]>,
    pub blocks: Vec<(usize, usize, usize)>,
}

/// The objects a pair of classes' cross term is on where the run leaves one of them out (`Images::links`): the links of
/// each class that has them, half to each where both do, by their shares within the class; none where the run has both
/// (the pair's coherence is then the object, `Unit::Coherence`).
pub(crate) fn pair_links(fold: &Fold, a: &Images, b: &Images) -> Vec<(Link, f64)> {
    let parts: Vec<Vec<(Link, f64)>> = [a, b].into_iter().filter(|c| !c.blocks.is_empty()).map(|c| fold.class_links(c)).collect();
    let mut out: Vec<(Link, f64)> = vec![];
    for links in &parts {
        let total: f64 = links.iter().map(|l| l.1).sum();
        for &(link, w) in links.iter().filter(|_| total > 0.0) {
            let x = w / total / parts.len() as f64;
            match out.iter_mut().find(|o| o.0 == link || o.0 == link.reversed()) {
                Some(o) => o.1 += x,
                None => out.push((link, x)),
            }
        }
    }
    out
}

/// The photons' cross terms a split into classes drops (criterion three; module documentation, # The criteria's numbers),
/// at a state of source eigen-coefficients `coefficients` c_p (of eigenvalues `values` λ_p, the source fold's) and norm
/// ‖ψ‖² `norm`: per pair of classes a < b landing in one landing of frequencies apart (`same`), the relative rate error
/// of leaving out J_a ρ J_b† and its adjoint,
///   Σ_{p,p′,q,q′} |c_p c_p′* Σ_x ⟨l_q|u^a_{x,p}⟩ ⟨l_q′|u^b_{x,p′}⟩*| / ‖ψ‖² / |ω_c|,
///   ω_c = (κ_a − κ_b) − (λ_q − λ_q′*) + (λ_p − λ_p′*):
/// after a jump at t_j the landing density's part |r_q⟩⟨r_q′| of the two classes turns in t_j at ω_c -- the difference
/// of the two photons' frequencies, ν_a + λ_p − λ_q less ν_b + λ_p′ − λ_q′, the same in every gauge --, and a class split
/// averages it over the jumps' times: what it leaves out is its part over |ω_c| (the secular approximation's first
/// order, v3 1). Its size is the dropped operator's in the landing's eigenbasis, of which the rate's interference
/// |Σ_x ⟨u^a_x|u^b_x⟩| is the trace alone; and every pair of source eigenstates, a state's coherence between two of them
/// a cross term at their eigenvalues' difference -- two ground states the frames do not join, their Raman resonance
/// at ω_c → 0 however far apart their classes' frequencies (Ω = κ_a − κ_b). A sector's photons are a class of their
/// own, at their frequency (`Fold::photons`), read by the same ω_c, against the classes the run emits: a pair of two
/// left-out sectors' classes is no split of the run's -- the run has neither photon, and what leaving them out costs
/// is their sectors' rates and populations, criterion one's. Returns (a, b, error) of the pairs read -- of those
/// beyond `floor` where it is a decision's (the prior's, at the share: a pair whose bound is within it is none of its
/// moves): by |Σ_x ⟨l_q|u^a⟩⟨l_q′|u^b⟩*| ≤ Σ_x A^a_x A^b_x, A_x = Σ_q |⟨l_q|u_x⟩|, and |ω_c| ≥ |κ_a − κ_b| − Λ, Λ the
/// largest |λ_q − λ_q′* − (λ_p − λ_p′*)|, a pair's error is within Σ_{p,p′} w Σ_x A^a A^b / (|κ_a − κ_b| − Λ) where
/// |κ_a − κ_b| > Λ (left-out sectors' photon classes, 1.8·10³ an atom of PGC1D_TILT_2, and pairs of them by the
/// million: the prior's cross terms 3.5 s an atom against its closure's 0.1).
pub(crate) fn cross_terms(classes: &[Images], landings: &[Landing], values: &[C64], coefficients: &[C64], norm: f64, floor: f64) -> Vec<(usize, usize, f64)> {
    let held: Vec<usize> = (0..coefficients.len()).filter(|&p| coefficients[p] != C64::ZERO).collect();
    // Per class, per axis, its coefficients on the landing's eigenstates, [p][axis][q] for the held p.
    let on: Vec<Vec<[Vec<C64>; 3]>> = classes
        .iter()
        .map(|class| {
            let landing = &landings[class.landing];
            held.iter().map(|&p| std::array::from_fn(|x| landing.left.apply(&class.images[p][x]))).collect()
        })
        .collect();
    // The classes of any photon from the held eigenstates: a pair with one of none is of no size.
    let active: Vec<usize> = (0..classes.len()).filter(|&a| on[a].iter().any(|axes| axes.iter().any(|v| v.iter().any(|z| *z != C64::ZERO)))).collect();
    // Per class and held eigenstate, per axis, Σ_q |⟨l_q|u_x⟩|; per landing, Λ over the held pairs.
    let sums: Vec<Vec<[f64; 3]>> = on.iter().map(|of| of.iter().map(|axes| std::array::from_fn(|x| axes[x].iter().map(|z| z.abs()).sum())).collect()).collect();
    let spread: Vec<f64> = landings
        .iter()
        .map(|landing| {
            let lambda = &landing.values;
            held.iter().flat_map(|&p| held.iter().map(move |&p2| values[p] - values[p2].conj())).flat_map(|source| lambda.iter().flat_map(move |&l| lambda.iter().map(move |&l2| (l - l2.conj() - source).abs()))).fold(0.0, f64::max)
        })
        .collect();
    let mut out = vec![];
    for (i, &a) in active.iter().enumerate() {
        for &b in &active[i + 1..] {
            let (ca, cb) = (&classes[a], &classes[b]);
            // Two classes the run emits neither of: no split of the run's (the run has neither photon; what leaving
            // them out costs is their sectors' rates and populations, criterion one's, `Fold::sector_parts`).
            if ca.landing != cb.landing || equal(ca.frequency - cb.frequency, ca.scale + cb.scale) || (!ca.blocks.is_empty() && !cb.blocks.is_empty()) {
                continue;
            }
            let lambda = &landings[ca.landing].values;
            let omega = ca.frequency - cb.frequency;
            let apart = omega.abs() - spread[ca.landing];
            if floor > 0.0 && apart > 0.0 {
                let bound: f64 = (0..held.len()).flat_map(|i| (0..held.len()).map(move |k| (i, k))).map(|(i, k)| (coefficients[held[i]] * coefficients[held[k]].conj()).abs() / norm * (0..3).map(|x| sums[a][i][x] * sums[b][k][x]).sum::<f64>()).sum::<f64>() / apart;
                if !(bound > floor) {
                    continue;
                }
            }
            let mut error = 0.0;
            for (i, &p) in held.iter().enumerate() {
                for (k, &p2) in held.iter().enumerate() {
                    let weight = (coefficients[p] * coefficients[p2].conj()).abs() / norm;
                    let source = values[p] - values[p2].conj();
                    for q in 0..lambda.len() {
                        for q2 in 0..lambda.len() {
                            let size = (0..3).map(|x| on[a][i][x][q] * on[b][k][x][q2].conj()).sum::<C64>().abs();
                            if size == 0.0 {
                                continue;
                            }
                            error += weight * size / (C64::real(omega) - (lambda[q] - lambda[q2].conj()) + source).abs();
                        }
                    }
                }
            }
            if error > 0.0 {
                out.push((a, b, error));
            }
        }
    }
    out
}

/// Criterion three at an atom's start (the prior's, `Model::prior`; `cross_terms`): the closure's fold's photons
/// (`Fold::photons`, its sectors' with them) at each ground-like eigenstate full, and at each two of one group in equal
/// parts -- the largest coherence a state holds between them --; every pair of classes beyond the share at one of those
/// states: of two the run has, the pair, each class as (its light's class, none for an explicit state's own decay; the
/// state it comes from; the ground state it lands in) -- states as the fold's: its ground places, and an explicit
/// excited state its E index after them --; of one the run leaves out, its links (`pair_links`), the largest parts
/// first until what is left is within the share, as the closure moves (`closure`).
pub(crate) fn start_cross_terms(fold: &Fold, emission: &[Mat; 3]) -> (Vec<[(Option<usize>, usize, usize); 2]>, Vec<Link>) {
    let (n, ground) = (fold.size(), fold.ground);
    let like: Vec<usize> = (0..n).filter(|&p| fold.ground_weight(p) >= 0.5).collect();
    let (images, origins) = fold.photons(emission, &like);
    let landings: Vec<Landing> = (0..fold.members.len()).map(|g| Landing::of(fold, &(0..n).filter(|&q| fold.groups[q] == g).collect::<Vec<_>>())).collect();
    let class = |(origin, at): (PhotonSource, usize)| match origin {
        PhotonSource::Class(c, g) => (Some(c), g, at),
        PhotonSource::Explicit(place) => (None, ground + fold.explicit[place - ground], at),
    };
    let mut states: Vec<Vec<usize>> = like.iter().map(|&p| vec![p]).collect();
    for (i, &p) in like.iter().enumerate() {
        for &p2 in &like[i + 1..] {
            if fold.groups[p] == fold.groups[p2] {
                states.push(vec![p, p2]);
            }
        }
    }
    let (mut out, mut links): (Vec<[(Option<usize>, usize, usize); 2]>, Vec<Link>) = (vec![], vec![]);
    for state in &states {
        let mut coefficients = vec![C64::ZERO; n];
        state.iter().for_each(|&p| coefficients[p] = C64::ONE);
        let psi: Vec<C64> = (0..n).map(|k| state.iter().map(|&p| fold.eigen.vectors[(k, p)]).sum()).collect();
        for (a, b, x) in cross_terms(&images, &landings, &fold.eigen.values, &coefficients, crate::linalg::norm_sqr(&psi), share()) {
            if !(x > share()) {
                continue;
            }
            let mut on = pair_links(fold, &images[a], &images[b]);
            if on.is_empty() {
                let pair = [class(origins[a]), class(origins[b])];
                if !out.contains(&pair) {
                    out.push(pair);
                }
                continue;
            }
            on.sort_by(|u, v| v.1.total_cmp(&u.1));
            let mut rest = x;
            for (link, part) in on {
                if !(rest > share()) {
                    break;
                }
                rest -= x * part;
                if !links.iter().any(|l| *l == link || *l == link.reversed()) {
                    links.push(link);
                }
            }
        }
    }
    (out, links)
}

/// Where one of a fold's photon classes (`Fold::photons`) comes from, for its unit's exact beat (`Model::prior`): a
/// class's folded jumps from a ground place it drives into the replica, or an explicit place's own decay.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum PhotonSource {
    Class(usize, usize),
    Explicit(usize),
}

impl Fold {
    /// The fold's photons by class (criterion three, `cross_terms`), with the emission `emission` (C_a on the fold's
    /// states, ground ← ground + excited), of the source eigenstates `states` (the others' images empty): each replica's
    /// excited amplitude T_j r_p and each explicit place's own, emitted into every ground place, a class per frequency
    /// κ = f_j − θ_g (an explicit place's own frame for f) and landing -- the group of the ground place, an index into
    /// `Landing`s made by the caller in the order of `self.members` --; each with the place it comes from (`PhotonSource`) and
    /// the ground place it lands in, for its beat. A folded photon and an explicit one at one κ are one class, as the
    /// run's (photon.rs): their beats are one.
    ///
    /// And the sectors' (module documentation, # Sectors): a sector's amplitude T′r_p on a target group, at λ_p + Ω, has
    /// the excited amplitudes T_j T′r_p of that group's replicas and its own on that group's explicit places, and they
    /// emit at the frequency shifted by the beat, κ = f_j + Ω − θ_g -- read by the same ω_c (`cross_terms`). Each is a
    /// class of its own, apart from the fold's own photons: a carried sector's -- the run's tone, whose amplitude is the
    /// run's on the target's places -- of the run's class its origin is, the target's (a pair with its own class one
    /// class, `Model::prior`); a left-out sector's -- photons the run has none of -- with its block's links by their
    /// shares (`Fold::link_weights`; `Images::links`), what keeping takes them into the run.
    pub(crate) fn photons(&self, emission: &[Mat; 3], states: &[usize]) -> (Vec<Images>, Vec<(PhotonSource, usize)>) {
        let (n, ground) = (self.size(), self.ground);
        // A class: its images, origin, landing place, and its kind -- the fold's own photons (0), a carried sector's (1),
        // a left-out one's (2).
        let mut out: Vec<(Images, PhotonSource, usize, u8)> = vec![];
        // The class a frequency of a kind lands in, by its exact bits once found: the first class one with it (`equal`),
        // which no class made later precedes.
        let found: std::cell::RefCell<std::collections::HashMap<(u8, usize, u64), usize>> = Default::default();
        let add = |(frequency, scale): (f64, f64), kind: u8, origin: PhotonSource, at: usize, p: usize, axis: usize, value: C64, out: &mut Vec<(Images, PhotonSource, usize, u8)>| -> Option<usize> {
            if value == C64::ZERO {
                return None;
            }
            let landing = self.places[at];
            let key = (kind, landing, frequency.to_bits());
            let known = found.borrow().get(&key).copied();
            let k = match known.or_else(|| out.iter().position(|c| c.3 == kind && c.0.landing == landing && equal(c.0.frequency - frequency, c.0.scale + scale))) {
                Some(k) => k,
                None => {
                    out.push((Images { frequency, scale, landing, images: vec![std::array::from_fn(|_| vec![C64::ZERO; n]); n], blocks: vec![] }, origin, at, kind));
                    out.len() - 1
                }
            };
            found.borrow_mut().insert(key, k);
            out[k].0.images[p][axis][at] += value;
            Some(k)
        };
        let origin_of = |j: usize, rep: &Folded| {
            let origin = (0..self.driven.len()).find_map(|c| (0..ground).find(|&g| self.driven[c][g] == Some(j)).map(|g| PhotonSource::Class(c, g)));
            origin.or_else(|| (0..self.explicit.len()).find(|&i| equal(self.frames[ground + i] - rep.frequency, self.sizes[ground + i] + rep.scale)).map(|i| PhotonSource::Explicit(ground + i)))
        };
        // The photons of an amplitude x on the places of the group `group` (none: every group) at the beat Ω more (its
        // terms' size `scale`): its replicas' excited amplitudes and its explicit places' own; the classes they make.
        let emit = |p: usize, x: &[C64], group: Option<usize>, (omega, scale): (f64, f64), kind: u8, out: &mut Vec<(Images, PhotonSource, usize, u8)>| -> Vec<usize> {
            let mut made = vec![];
            for (j, rep) in self.replicas.iter().enumerate().filter(|(_, rep)| group.is_none_or(|g| rep.group == g)) {
                let Some(origin) = origin_of(j, rep) else { continue };
                let y = rep.t.apply(x);
                for g in 0..ground {
                    let frequency = (rep.frequency + omega - self.frames[g], rep.scale + scale + self.sizes[g]);
                    for axis in 0..3 {
                        let value: C64 = rep.q.iter().zip(&y).map(|(&e, z)| emission[axis][(g, ground + e)] * *z).sum();
                        made.extend(add(frequency, kind, origin, g, p, axis, value, out));
                    }
                }
            }
            for (i, &e) in self.explicit.iter().enumerate().filter(|(i, _)| group.is_none_or(|g| self.places[ground + i] == g)) {
                for g in 0..ground {
                    let frequency = (self.frames[ground + i] + omega - self.frames[g], self.sizes[ground + i] + scale + self.sizes[g]);
                    for axis in 0..3 {
                        made.extend(add(frequency, kind, PhotonSource::Explicit(ground + i), g, p, axis, emission[axis][(g, ground + e)] * x[ground + i], out));
                    }
                }
            }
            made
        };
        for &p in states {
            let r = self.right(p);
            emit(p, &r, None, (0.0, 0.0), 0, &mut out);
            for (kind, list) in [(1, &self.carried), (2, &self.sectors)] {
                for (index, s) in list.iter().enumerate() {
                    let t = self.sector_apply(s, &r);
                    for (to, places) in self.targets(s, p) {
                        let mut made = emit(p, &t, Some(to), (s.frequency, s.scale), kind, &mut out);
                        // A block's links weigh where its coupling reaches r_p (`link_weights`, made when read).
                        let from = &self.members[self.groups[p]];
                        if kind == 2 && s.links_in(places, from).any(|&(_, _, y, value)| value * r[y] != C64::ZERO) {
                            made.sort();
                            made.dedup();
                            for &k in &made {
                                out[k].0.blocks.push((index, p, to));
                            }
                        }
                    }
                }
            }
        }
        out.into_iter().map(|(images, origin, at, _)| (images, (origin, at))).unzip()
    }

    /// A left-out sector's photon class's links with their shares of the class (`Images::blocks`; `pair_links`): per
    /// block it holds photons of, each link's weight in T′r_p over the block's (`link_weights`), summed link by link.
    pub(crate) fn class_links(&self, class: &Images) -> Vec<(Link, f64)> {
        let mut links: Vec<(Link, f64)> = vec![];
        for &(index, p, to) in &class.blocks {
            let weights = self.link_weights(&self.sectors[index], p, to, &self.members[to]);
            let total: f64 = weights.iter().map(|w| w.1).sum();
            for &(link, w) in weights.iter().filter(|_| total > 0.0) {
                match links.iter_mut().find(|l| l.0 == link) {
                    Some(l) => l.1 += w / total,
                    None => links.push((link, w / total)),
                }
            }
        }
        links
    }
}

#[cfg(test)]
mod tests;
