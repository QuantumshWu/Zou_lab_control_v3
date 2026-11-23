//! The path errors of candidate models (PIPELINE_V3_FINAL §二 A.4, B.7–8; foldstudy ENGFOLD, PATHGAP): along atom
//! paths -- trajectories' positions and momenta in time, the pilot's (solve.rs `walks`) --, each candidate model's
//! internal density matrix and the reference's (the engine's own 32-state model) are advanced on the same path by the
//! same segments, and their readings compared: the displayed rows, the levels' sums, the photon rates by class, the
//! force. The difference on a path is the model's error of what it reads along that path; its window means over the
//! paths, with their spread, are the direct part of the model's error (DESIGN2 A, the path-measured part). Nothing here
//! selects or books: the module and its tests' calls (stage 3 takes its interface).
//!
//! # The sector
//! A model's state is a density matrix on P. P falls into its exact blocks (model/selected.rs, # Exact blocks): every
//! matrix of the generator is block-diagonal, and a jump lands on the ground states. Only the blocks that hold a ground
//! state are ever fed -- the initial state is a ground state's, every jump lands on ground states --, and of their
//! coherences only those a jump makes: the sector is the closure of the pairs (X, X) of the ground-holding blocks under
//! the jumps, a pair (X, Y) wherever a jump takes a pair (A, B) of the sector to ground states of X and Y (on every
//! scene so far every jump maps a block into one block, ENGFOLD: the sector is the diagonal pairs, Σ_X b_X², 576 for
//! PGC3D's 32 states against 1024). Its vector holds each pair's b_X × b_Y matrix row by row, pair after pair.
//!
//! # The generator
//! On the sector dρ/dt = L(t)ρ = −i(H(t)ρ − ρH(t)†) + Σ_{c,a} J_{c,a}(t) ρ J_{c,a}(t)†, H the node's generator (H_eff in
//! the frame with its decay −(i/2)ΣL†L: i(H − H†) = Σ J†J by construction, so the trace is kept) by its exact beats,
//! H(t) = Σ_k G_k e^{−iΩ_k t}, and J_{c,a}(t) a photon class's axis, its components at their beats likewise
//! (model/selected.rs, # The generator by beats, # The photons). The beats are taken exactly at every time; the
//! amplitudes G_k, J are the model's at the path's place and momentum (`Selected::node`). The recycling is one map R(t)
//! from the sector to its ground pairs' entries, made at every time from the jumps' elements.
//!
//! # The nodes
//! The model is made at few nodes and interpolated between them on the blocks (PATHGAP item 3: "segments dense, nodes
//! sparse"): the path's knots are its step ends -- the trajectory's path is C¹ there, its acceleration jumps --, each
//! knot interval cut into the scheme's pieces (`pieces_in`: the least power of two that keeps a piece no longer than the
//! knob's length), the model made at every piece's ends (`Selected::node`), and the
//! amplitudes between them the Lagrange interpolant of degree min(3, pieces) (`lagrange`: linear at one piece an interval)
//! of its own knot interval (never across a knot); the beats' phases exact at the time. H is interpolated by its
//! beats, R by its beats' differences (`Amps`): each such amplitude is smooth along the path, the phases are not.
//! Readouts are made at knots, on the exact node there. The pieces' knob refines the nodes with them -- one Richardson
//! reading for both (select.rs `ladder`).
//!
//! # The readings
//! At a readout, on the node there: the displayed rows -- P's populations with Q's lifted (`Dynamics::readout`, the
//! engine's one readout: it takes a state vector, so ρ is taken through its eigenvectors, the unnormalized rows being
//! linear in ρ), over their sum; the photon rate of each class, Σ_a tr(J_{c,a} ρ J_{c,a}†); the force,
//! −tr(∇H_h ρ) + Σ_{c,a} Im tr(ρ J_{c,a}†∇J_{c,a}) (the Ehrenfest force and the scattered light's absorbed momentum).
//! The channels are the trajectories' own, one set: J_{c,a} and ∇J_{c,a} are the elements of the model's node
//! (`Selected::node`, its classes by Cartesian axis) that a trajectory's photons read (model/selected.rs `photons`,
//! model.rs `Moments::of`: ‖J ψ‖², Im⟨Jψ|∇Jψ⟩, Re⟨∂Jψ|∂Jψ⟩), here in ρ form -- the same sums contracted with ρ, not
//! applied to ψ (through ρ's eigenvectors, each read as a trajectory's state, the readings took 4.7 times as long,
//! PGC3D 2026-11-23) --; −tr(∇H_h ρ) is the model's own contraction (`SelectedMiddle::contract`), and the share of
//! the absorbed mean a jump carries the model's rule (`Moments::new`). The rates and the force are the trajectory's: over tr ρ_P, the state's norm on P, as the integrator takes them; the
//! rows over their sum as shown. (ENGFOLD's prototype took the rates and the force over the rows' sum too; the two
//! differ by the lifted Q population. `Reading::total` holds that sum.) tr ρ_P itself is kept by every model's exact
//! flow -- i(H − H†) = Σ J†J --, so its departure from one along a series is the scheme's (`Series::trace_defect`): a
//! strict lower bound of the scheme's error in the trace norm, read at no cost, and a part of the setting's error
//! (select.rs `settle`; the readings over tr ρ_P and the rows' sum hide it).
//!
//! # The motion's readings
//! What the motion takes of the internal state, beside the rows (select.rs # The motion), each over tr ρ_P:
//! - the heating per lab axis, H_a = (ħ/m)/2 · d⟨κ_a²⟩/dt of the scattered light (`Reading::heating_at`): the emitted
//!   photons' k_c² λ_c ⟨n_a²⟩ -- the dipole pattern tr G − n·Re(G)·n of a class's Cartesian amplitudes (step.rs
//!   `Emitted`), whose ⟨n_a²⟩ λ_c = (2/5) tr G − (1/5) G_aa, G_ab = tr(J_{c,a} ρ J_{c,b}†) --, k_c the class's photon wave
//!   number (its photon forms K = k∘L on a node at the form's point: one class one line); and the absorbed photons'
//!   λ ((k²)_a − k̄_a²) + λ (1 + Q) k̄_a²: their directed part k̄ counted with the photon count's Fano factor 1 + Q, Q
//!   each model's own (# The landing check), the rest as Poisson's. The absorbed photon's k̄ and (k²) are the light's at
//!   the place, not a model's: each beam l but the tweezer's by its share w_l ∝ (‖X_l‖² + ‖R_l‖²)|a_l(r)|², its
//!   coupling's strength at its focus times its profile's intensity there -- the same for every model, whose own rate λ
//!   scales it. (A model's own Σ Re tr(∂J ρ ∂J†) is no such measure: it holds what its folded jumps carry, and an
//!   explicit excited state carries its absorbed momentum in its force, ⟨∇H⟩, nowhere in its jumps -- SAT_03's 32
//!   states read 0.003 of it against the full-ground model's 413 /µm²/µs. From 2026-11-23 to the 23rd's evening the
//!   jumps' own second moment stood here with the force's correlation's diffusion by the quantum regression beside it:
//!   see # The landing check for why that diffusion is no measure of the trajectories' heating.)
//! - the force's power per lab axis F_a v_a, v = (ħ/m)κ the path's -- in a window the force's difference against the
//!   velocity, each relative to the cloud's mean there, what moves T (`compare`). (Its conservative part, a lattice's or a
//!   trap's light force against the motion's oscillation, moves no energy over the motion's periods; on a window
//!   shorter than them it is noise of zero mean, the paths' spread. Taking it out by the models' energies ⟨H⟩ at the
//!   window's ends, −d⟨H⟩/dt along a dressed eigenstate, was tried and is wrong near resonance: ⟨H⟩ holds the excited
//!   dressed states' energies by their populations, and R9_PUSH's lost paths read x = −1.2 of it against the +0.11 of
//!   their rates.)
//!
//! # The landing check
//! A path's reading starts once, from the walk's start; after every jump the atom lands on the ground states and its
//! excitation grows from nothing again -- the photons' statistics, which a folded model has not (its emission is
//! Poisson's) and which a partial level gets wrong between its explicit states and its folded neighbours. So at a
//! window's last reading that leaves it room, time t, the reference's state ρ is mapped to where a jump lands it, ρ_L =
//! R(t)ρ / tr R(t)ρ (`landed`: the recycling, Σ_c J_c ρ J_c† over the jump rate), and each model goes on from ρ_L along
//! the path (`landing`), against its own run there: its excess photons n_ex = ∫(λ(τ|L) − λ̄)dτ (and impulse ∫(F(τ|L) −
//! F̄)dτ, said). By the regression theorem λ(τ|L) = λ̄ g⁽²⁾(τ), so Mandel's Q = 2 n_ex: the count's Fano factor 1 + Q
//! of the directed absorbed part a model's jumps carry. The span is the optical memory 2 ln(2/ε)/Γ, Γ the slowest
//! excited decay (`optical`: the excitation grown to the steady state's to the rounding), the stretch's knots from a
//! quarter of the fastest decay's time growing by 2^{1/4} a knot to the walk's own spacing -- the transient resolved
//! where it is --, one piece a knot; the ground states' slower redistribution after the landing is outside the span.
//!
//! No force correlation's diffusion is read. The quantum regression's D_a = ∫₀^∞ ⟨δF_a(0) δF_a(τ)⟩ dτ, the force as an
//! operator, ½(G_a ρ + ρ G_a) − ⟨G_a⟩ρ its source, holds the operator's own variance ⟨G²⟩ − ⟨G⟩², which a pure state has
//! too; the trajectories (the force ⟨ψ|∇H|ψ⟩ between jumps, select.rs # The motion) have none of it -- their force
//! fluctuates by their jumps alone, its correlation tr(G e^{Lτ} M), M = E[⟨G⟩_ψ |ψ⟩⟨ψ|] over the conditional states,
//! not ½{G, ρ}. The two agree on a resonant two-level cycle (SAT_03's 32 states: 0.49 of Poisson's k²λ/2 on the push
//! axis, where its Q = −0.53 gives 0.47 -- what 1 + Q counts above), not where ground states hold long coherences
//! under different light shifts. Read 2026-11-23 (and removed the same day): at the generator frozen at the time, the
//! atom at rest -- an approximation none of the four allowed --, its correlation time ∫τC/∫C was 30-145 µs on PGC3D's
//! reference against some 6 µs for its atoms to cross λ/2, 2e2-7e4 µs on GMD1C's (its dark states), and D of either
//! sign up to 1.3e3 /µm²/µs: GMD1C's reference heated below nought, its T's relaxation none, G's T error 140 µK at
//! 1000 µs against T 9 µK, the selection past 300 s (the release's 13 s). Run along the walk instead (the generator
//! the atom's own as it moves, D_a = −p ∫(F(τ|ρ + X/p) − F(τ|ρ))dτ), D was of one sign and sane in size but no
//! trajectory's: PGC3D's full-ground model read [229, 52, 69] against its reference's [56, −2, 5] /µm²/µs, a heating
//! several times the reference's, where the two models' T from their own trajectories differ by 0.03 of T's allowance
//! (FOLDTEMP's pair). The jumps' own share of the dipole force's fluctuation, which the trajectories do have, is not
//! read here: select.rs # The motion, what the account leaves out.

//! # The memory
//! The path's length is set by the generator's slowest nonzero decay rate λ (`memory`): the sector Liouvillian's
//! eigenvalues at a point, a rounding zero |μ| ≤ κ_μ γ_d ‖L‖₁ (its first-order perturbation bound, κ_μ its condition)
//! the stationary states'; τ_mem = 1/λ, and a path's length L = min(T_run, 5τ_mem) (`Memory::length`).
//!
//! # The budget
//! Selection needs the model's error A to a small share of the target, not to the rounding: a scheme's discretization
//! error on the fold error only has to lie well within that share, and is booked as a part of A. The exponential
//! scheme takes H's fast turning exactly in its eigenbasis and leaves to the substeps only what the readings change by
//! -- its cost set by the substeps, not by ‖L‖ (# The substeps) --; the selection reads its error against its own
//! refinements (select.rs `ladder`). (The Magnus segments a turn of the fastest coherence each, `Sparse`, the converged
//! scheme they were read against until 2026-11-22, did not end on scenes whose 32 states hold levels far off the
//! light's frame, and the product never took them: removed with ENGFOLD's and the one-exponential-a-step schemes.)
//!
//! # The substeps
//! On a piece of `Scheme::Exponential` the sector's state is held in the eigenbasis of H at the piece's middle t_m, σ =
//! V⁻¹ρV⁻†, where −i(H(t_m)ρ − ρH(t_m)†) is the diagonal Λ, μ_ab = −i(λ_a − conj λ_b); the rest -- H's change from t_m
//! (the path's motion and the beats, of the interpolated amplitudes, # The nodes) and the recycling R(t) -- is the
//! remainder g(t, σ) = N(t)σ, linear in σ:
//! σ' = Λσ + g(t, σ), σ(t + d) = e^{Λd}σ(t) + ∫₀^d e^{Λ(d−s)} g(t + s, σ(t + s)) ds.
//! - Lawson's scheme (the integrating factor, until 2026-11-22) took u = e^{−Λs}σ and RK4 on u' = e^{−Λs}N e^{Λs}u. A
//!   coherence the light drives -- between a dressed ground and a dressed excited state -- is steady where the
//!   readings are (it follows the populations, σ_ab ≈ −(Nσ)_ab/μ_ab), but its u_ab turns at Im μ_ab, the detuning: RK4
//!   had to resolve it, d·|Im μ| ≲ 1, the substeps set by the detuning and not by what the readings change by. The
//!   scheme does not keep the fixed point Λσ* + Nσ* = 0 (Hochbruck and Ostermann 2010, Acta Numerica 19, §2.3). Measured
//!   (SELCOST, the reference against `Sparse` on R4's walk held at rest, past its start): 4.2e-3 at 8.4 substeps a µs,
//!   1.2e-4 at 270; TET's 135 a µs and R4's 270 were the detunings', 250–1000 rad/µs.
//! - The exponential Runge–Kutta scheme of Krogstad (2005; Hochbruck and Ostermann 2010, (2.37)) takes g itself as a
//!   polynomial in s over the substep and integrates e^{Λ(d−s)} exactly by φ_k(Λd) (`phis`, `etd_weights`): constant g
//!   is integrated exactly, so σ* is kept, and its error follows the change of g along the substep -- the populations'
//!   pumping, the motion, the beats, the start's transient. Same work a substep as Lawson's (four remainders),
//!   classical order 4, stiff order 3 (for a linear inhomogeneous problem Σ b_i c_i³/3! ≠ φ₄); ΔH at its stages' times,
//!   R's amplitudes at the substep's middle (its beats' phases at the stages': R moves with the path slowly, and so
//!   errs at order 2 in d, the order the substeps' Richardson reading takes). Measured on the same walk: 1.9e-5 at 8.4
//!   substeps a µs, 2.1e-6 at 16.9.
//! - H's change along the piece is in the remainder, so the generator is continuous in time. Frozen at the piece's
//!   middle instead (until SELCOST), H made a staircase: each step a small start, whose transient the substeps had to
//!   resolve, and whose harmonics drove the coherences where the piece was longer than their period -- R4's reference
//!   stood 0.8–1.1e-3 off the converged scheme at 3 to 16 pieces a step and fell to 1.4e-5 at 20, while 4 against 8
//!   read 5e-5: the pieces' Richardson reading read "converged".
//!
//! # The start
//! A stretch from a start state (a walk's start; the late segment and the scans begin from it too) starts off the slow
//! manifold: its coherences turn at the dressed splittings and decay at the excited decay -- a real transient, which
//! feeds g at those frequencies until it has decayed, and which the substeps must resolve there and nowhere else.
//! Unresolved, the scheme stands on a plateau there, where doubling the substeps changes nothing and the Richardson
//! reading reads "converged": R4 at its start, the reference against the converged scheme 5.2e-3 at 16.9 to 135
//! substeps a µs, 1.0e-3 at 270, 3.8e-4 at 540 and 1080 (the rest the run's).
//! - So over the opening the substeps are counted in turns of the fastest mode the opening resolves, W: `turns`
//!   substeps a turn, one at the least -- d·W ≤ 2π, the stages' half-substep sampling at its Nyquist rate --, off the
//!   plateau, where doubling them reads their error. W is the largest |μ| -- the turning and the decay together, the
//!   rate the lemma of the booking below takes -- at the stretch's start over the modes the
//!   start holds above the rounding and the opening outlasts (`Form::opening`). (Until 2026-11-22 it was |Im μ|, the
//!   turning alone: SAT_03's resonant, overdamped dressed pair has real μ, so W was nought, its first 5 µs one substep
//!   at Γd = 190, and the reference's trace fell to 1.6e-5 there -- the recycling R ρ, which feeds back what the decay
//!   in Λ takes out, integrated by a polynomial over a substep far longer than the decay; doubling the turns changed
//!   nothing. Study A §2.6.) A coherence the start does not hold
//!   needs no turn (S1's 32 states hold levels 4.5e7 rad/µs off the light's frame, which its start holds nothing of: W
//!   140 rad/µs; taken over the whole sector, 0.5 µs of S1 took 148 s), and a mode the opening does not outlast is the
//!   run's (R5_PUMP's ground hyperfine coherences, 4.4e4 rad/µs, held at 2e-5 and decaying at the pumping rate).
//! - The opening lasts while the start's optical coherences have not decayed to the double's unit rounding u: a
//!   coherence between a ground and an excited state decays at half the excited state's rate (for a dressed pair the
//!   two halves' sum, the trace of their decay), so t_open = 2 ln(1/u)/Γ, Γ the slowest decay of P's excited states in
//!   the sector at the start (`Form::opening`); a model whose excited states are all folded holds no such coherence and
//!   opens at once. (The modes between dressed ground states turn slowly or decay slowly and feed the remainder little:
//!   held to the rounding as well, R4's opening grew to 10 µs and TET's to 458 µs, the whole path at the opening's
//!   rate.) Until the opening was so set it was the paths' first window, 0.29 µs on R4, and the transient's rest stood
//!   at 3.8e-4 whatever the run's rate: with 0.5 µs or more, 2.3e-4 at 8.4 a µs -- the pieces' part -- falling with
//!   the pieces.
//! - **Resolve or book** (the model's own judgment, model/classify.rs: zero, drop under a strict bound booked, fold,
//!   keep): a held mode faster than the setting's `cutoff` (|μ_j|) is not resolved -- its part of the start, σ₀ⱼ m_j (m_j = v_a
//!   v_b†, V's columns of the mode's pair of eigenstates), is dropped at the start, and a strict bound of what that does
//!   to a displayed row is booked; the cutoff is the setting's fourth knob, its part that bound, doubled while it is the
//!   largest of the parts (select.rs `settle`): a mode is booked where its bound fits the tolerance, resolved where it
//!   does not, the same for every mode and scene. The bound: the dropped part goes on as δ(t) = Φ(t, 0)σ₀ⱼm_j; with L₀
//!   the piece's frozen part (L₀m_j = μ_j m_j, μ_j = −γ_j + iω_j) and N = L − L₀ (the recycling R and H's change ΔH),
//!   Duhamel gives δ(t)/σ₀ⱼ = e^{μ_j t} m_j + ∫₀ᵗ c(u) e^{μ_j u} du, c(u) = Φ(t, u) N(u) m_j. The first is read from the
//!   first knot on, |·| ≤ ‖m_j‖₁ e^{−γ_j t₁} (‖m_j‖₁ = ‖v_a‖‖v_b‖); the second is the lemma's (model/classify.rs
//!   `by_parts`, with the whole complex rate: the turning and the decay together, |μ_j|), |c| ≤ ‖N m_j‖₁ ≤ ‖R m_j‖₁ +
//!   2‖ΔH‖‖m_j‖₁ (Φ contracts the trace norm), |ċ| ≤ ‖L N m_j‖₁ + ‖Ṅ m_j‖₁, ∫₀ᵗ |ċ| e^{−γ_j u} du ≤ sup|ċ|/γ_j: B_j =
//!   |σ₀ⱼ| (‖m_j‖₁ e^{−γ_j t₁} + (2‖N m_j‖₁ + (‖L R m_j‖₁ + 2‖L‖‖ΔH‖‖m_j‖₁ + 2‖Ḣ‖‖m_j‖₁)/γ_j)/|μ_j|), a displayed row
//!   moving by at most the trace norm (bounded by the entries' moduli). ‖ΔH‖ is H's change over half a piece of the
//!   first knot interval with every beat's amplitude twice, ‖Ḣ‖ the change over that interval and each beat's amplitude
//!   twice times its rate, ‖L‖ the generator's 1-norm bound, L R m_j taken at the start (`Form::opening`).
//!
//! What an exponential piece needs before its substeps reads no state -- the nodes, H at its middle and the blocks'
//! eigendecompositions, the substep's weights (`Frozen`) --, so it is made on the pool, a chunk of knot intervals
//! ahead of the propagation (their nodes first, one job each, then their pieces), which alone runs in order,
//! interpolating R and ΔH from the nodes as it goes (a piece of the opening holds hundreds of substeps); the readings
//! of a chunk are made on the pool after it (`Form::family`, `Form::freeze_family`).
//!
//! A family of runs along one stretch from one state -- a setting and its refinements, each knob doubled alone (select.rs
//! `settle`) -- samples one path: a node is the path's at its time, the same for every scheme that takes it there. So a
//! family makes each node once (`Form::along_family`): on each knot interval its members' pieces are powers of two of one
//! another (`pieces_in`), and every member's nodes are the interval's finest member's at the same times to the bit; a piece is frozen once for the members of
//! the same pieces and substeps there, and a state propagated once for the members whose pieces have all been the same
//! so far -- up to the opening's end the runs that differ only in the rate, whose opening is the same --, parted where
//! their pieces part. Each member's series is its run alone, to the bit (paths/tests.rs `paths_a_family_is_its_members`).
//! (Run alone, a setting's refinements made every node again: the nodes and their amplitudes are nine tenths of a
//! reference run's freezing, TET's 32 states 59 of 65 s over the paths' jobs.)
//!
//! # The errors
//! Per path the candidate's readings less the reference's at the same times, averaged over a time window; over the
//! paths their mean and its standard error (the paths' spread over √P), and per path its worst row (`compare`,
//! `Window`). The windows are octaves of τ_mem from the start's transient to L (`windows`); an output time takes its
//! window's error, past L the last window's (`window_of`): the path has forgotten its start there. The paths are the
//! pilot's trajectories at the run's step (solve.rs `walks`), each from its own initial state.
use crate::linalg::{fused_product, fused_product_onto, hermitian_eigen, larger, mul_add, C64, Eigen, Mat, V3};
use crate::floquet::frame::Ratio;
use crate::model::selected::{Exact, Node, Selected};
use crate::model::Point;
use crate::request::{GROUND, STATES};
use crate::step::Dynamics;
use crate::units::rounding;
use rayon::prelude::*;

/// An atom's path (module documentation): a trajectory's place and momentum at its step ends from its start, and its
/// internal state at its start by the request's states.
#[derive(Clone, Debug, Default)]
pub struct Walk {
    pub index: u64,
    /// The internal state at the first end: (state, amplitude), normalized.
    pub start: Vec<(usize, C64)>,
    /// (t, r, κ) at the step ends (µs, µm, 1/µm), the times increasing.
    pub ends: Vec<[f64; 7]>,
    /// Whether the trajectory lost its atom at its last end (its path ends there; select.rs # The motion).
    pub lost: bool,
    /// Its weight among the paths: its stratum's share of the pilot's trajectories over the stratum's paths (select.rs
    /// # How many paths; one alike where the paths are not stratified).
    pub weight: f64,
}

impl Walk {
    /// The path at the time t: r, and κ = ṙ/(ħ/m), of the cubic Hermite interpolant of r between the step ends with
    /// ṙ = (ħ/m)κ there (ENGFOLD's `path_at`; outside the ends the first or last piece's; a path of one end stands).
    pub fn at(&self, hm: f64, t: f64) -> (V3, V3) {
        let s = &self.ends;
        if s.len() == 1 {
            return ([s[0][1], s[0][2], s[0][3]], [s[0][4], s[0][5], s[0][6]]);
        }
        let i = s.partition_point(|x| x[0] <= t).clamp(1, s.len() - 1) - 1;
        let (a, b) = (&s[i], &s[i + 1]);
        let dt = b[0] - a[0];
        let x = (t - a[0]) / dt;
        let (h00, h10, h01, h11) = (2. * x * x * x - 3. * x * x + 1., x * x * x - 2. * x * x + x, -2. * x * x * x + 3. * x * x, x * x * x - x * x);
        let (d00, d10, d01, d11) = (6. * x * x - 6. * x, 3. * x * x - 4. * x + 1., -6. * x * x + 6. * x, 3. * x * x - 2. * x);
        let r = std::array::from_fn(|k| h00 * a[1 + k] + h10 * dt * hm * a[4 + k] + h01 * b[1 + k] + h11 * dt * hm * b[4 + k]);
        let kappa = std::array::from_fn(|k| (d00 * a[1 + k] + d10 * dt * hm * a[4 + k] + d01 * b[1 + k] + d11 * dt * hm * b[4 + k]) / dt / hm);
        (r, kappa)
    }

    /// The path's last time.
    pub fn end(&self) -> f64 {
        self.ends.last().map_or(0.0, |e| e[0])
    }

    /// The knots on [from, to]: from, the step ends strictly inside, to.
    fn knots(&self, from: f64, to: f64) -> Vec<f64> {
        let mut k = vec![from];
        k.extend(self.ends.iter().map(|e| e[0]).filter(|&t| t > from && t < to));
        k.push(to);
        k
    }
}

/// How a path is cut and where the model is made along it (module documentation, # The nodes, # The substeps).
#[derive(Clone, Copy, Debug)]
pub enum Scheme {
    /// Each knot interval of the path in equal pieces, `pieces` a µs (`pieces_in`: the least power of two that keeps a
    /// piece no longer than 1/`pieces`; one, the interval whole, at none), the model made at the pieces' ends and
    /// interpolated between them (# The nodes): H at a piece's middle integrated exactly in its eigenbasis (block by block), the rest -- H's change
    /// from there, the recycling Σ J ρ J† -- by the exponential Runge–Kutta substeps (# The substeps), over a stretch's
    /// opening `turns` a turn of the fastest mode it resolves -- the start's modes up to `cutoff` (rad/µs), the faster
    /// ones booked by their bound -- and `rate` a µs after it (# The start); a readout at every `read`-th knot and at the
    /// end. The scheme's error is read by its own refinements and booked (# The budget).
    Exponential { turns: f64, rate: f64, pieces: f64, cutoff: f64, read: usize },
}

/// The knot interval [`from`, `to`]'s pieces at `pieces` a µs (`Scheme::Exponential`, # The nodes): the least power of
/// two that keeps a piece no longer than 1/`pieces` -- one at least, the interval whole --; so on one interval two
/// schemes' pieces are powers of two of one another whatever their knobs (a family's nodes shared,
/// `Form::along_family`), and a piece's length is the knob's, not the run's step's. The interval's length is read to
/// its ends' rounding (each end a time rounded once, ε|t|; the product's own, 4ε): a step's interval late in a run is
/// the step to that, not beyond it -- PGC3D's at 1000 µs, 2e-13 of its 0.47 µs over it, took twice the pieces where
/// the knob stood at a power of two a step. (Until 2026-11-23 the knob was the pieces a step: a step twice as long took
/// the ladder one doubling more, over the whole stretch, to the same piece -- R9_PUSH's at 6.06 µs.)
pub fn pieces_in(from: f64, to: f64, pieces: f64) -> usize {
    let x = (to - from) * pieces;
    let rounding = (from.abs() + to.abs()) * f64::EPSILON * pieces + 4.0 * f64::EPSILON * x;
    ((x - rounding).ceil().max(1.0) as usize).next_power_of_two()
}

/// (Tests alone, OLDNEW's selection measurement: the landing check's parts' thread nanoseconds and calls -- the landed
/// state (`landed`), the force correlation's diffusion (`diffusion`) and its LU, the landing's stretch (`landing`) --,
/// each the pair [ns, calls] at 2i, 2i + 1; said by the selection's trace.)
#[cfg(test)]
pub(crate) static CHECKS: [std::sync::atomic::AtomicU64; 8] = [const { std::sync::atomic::AtomicU64::new(0) }; 8];

/// (Tests alone: one part's call timed into `CHECKS` while it lives.)
#[cfg(test)]
struct Checked(usize, std::time::Instant);
#[cfg(test)]
impl Checked {
    fn new(part: usize) -> Checked {
        Checked(part, std::time::Instant::now())
    }
}
#[cfg(test)]
impl Drop for Checked {
    fn drop(&mut self) {
        CHECKS[2 * self.0].fetch_add(self.1.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
        CHECKS[2 * self.0 + 1].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
}

/// The work of advancing models along paths: nodes made, exponentials and their matrix–vector products, segments,
/// readouts (each one's readout calls), and the seconds of each part (one thread's wall time).
#[derive(Clone, Copy, Debug, Default)]
pub struct Counts {
    pub nodes: u64,
    pub exponentials: u64,
    pub products: u64,
    pub segments: u64,
    pub readouts: u64,
    pub readout_calls: u64,
    pub node_seconds: f64,
    pub propagation_seconds: f64,
    pub readout_seconds: f64,
    /// The path's time advanced (µs).
    pub time: f64,
    /// The largest condition of an eigenvector matrix the scheme took (`Scheme::Exponential`).
    pub condition: f64,
    /// The stretch's opening (µs; `Scheme::Exponential`, module documentation # The start).
    pub opening: f64,
    /// The bound of what the start's modes the opening books instead of resolving can do to a displayed row (# The start).
    pub booked: f64,
    /// The counted work (real multiply-adds, `decide::Work`, the trajectories' price's measure -- select.rs # How many
    /// paths): every node's making (`Node::work`, as a step's middle counts it) and its amplitudes on the sector (R's
    /// terms, `Form::amplitudes_work`); each piece's frozen part -- its blocks' eigendecompositions (`Eigen::work`) and
    /// the substeps' weights -- and its two sandwiches; the substeps' products (`Form::product`); the readings
    /// (`Form::read`); and beside a series what the landing check takes (`Form::landing`, `Form::diffusion`).
    pub work: f64,
}

impl Counts {
    pub fn add(&mut self, o: &Counts) {
        self.nodes += o.nodes;
        self.exponentials += o.exponentials;
        self.products += o.products;
        self.segments += o.segments;
        self.readouts += o.readouts;
        self.readout_calls += o.readout_calls;
        self.node_seconds += o.node_seconds;
        self.propagation_seconds += o.propagation_seconds;
        self.readout_seconds += o.readout_seconds;
        self.time += o.time;
        self.condition = larger(self.condition, o.condition);
        self.opening = larger(self.opening, o.opening);
        self.booked = larger(self.booked, o.booked);
        self.work += o.work;
    }
    pub fn seconds(&self) -> f64 {
        self.node_seconds + self.propagation_seconds + self.readout_seconds
    }
}

/// A model's readings at a time of a path (module documentation, # The readings).
#[derive(Clone, Debug)]
pub struct Reading {
    pub t: f64,
    pub r: V3,
    pub kappa: V3,
    /// The rows as shown: over their sum.
    pub rows: [f64; STATES],
    /// Per photon class (`Series::classes`) its rate (1/µs), over tr ρ_P.
    pub rates: Vec<f64>,
    /// The force (rad/µs/µm, dκ/dt), over tr ρ_P.
    pub force: V3,
    /// The rows' sum before their normalization (P's trace with Q's lifted populations), and tr ρ_P.
    pub total: f64,
    pub trace: f64,
    /// Per lab axis the rate of ⟨κ_a²⟩ the emitted photons give, Σ_c k_c² ((2/5) λ_c − (1/5) tr(J_{c,a} ρ J_{c,a}†))
    /// (1/µm²/µs, over tr ρ_P; # The motion's readings).
    pub emitted: V3,
    /// The light's absorbed photon at the place per lab axis, the model's none (# The motion's readings): its mean
    /// Σ_l w_l k_{l,a} (1/µm) and its second moment Σ_l w_l k_{l,a}² (1/µm²), w_l the beams' shares.
    pub kick: V3,
    pub kick2: V3,
}

impl Reading {
    /// The heating per lab axis (rad/µs², d(E/ħ)/dt; # The motion's readings) at the photon count's Mandel Q `q`:
    /// (ħ/m)/2 times the rate of ⟨κ_a²⟩ the scattered light gives -- the emitted photons', and the absorbed ones'
    /// λ ((k²)_a − k̄_a²) + λ (1 + Q) k̄_a².
    pub fn heating_at(&self, hbar_over_mass: f64, q: f64) -> V3 {
        let lambda: f64 = self.rates.iter().sum();
        std::array::from_fn(|a| 0.5 * hbar_over_mass * (self.emitted[a] + lambda * (self.kick2[a] - self.kick[a] * self.kick[a] + (1.0 + q) * self.kick[a] * self.kick[a])))
    }

    /// `heating_at` of Poisson's count (Q = 0).
    pub fn heating(&self, hbar_over_mass: f64) -> V3 {
        self.heating_at(hbar_over_mass, 0.0)
    }

    /// The force's power per lab axis on the path (rad/µs², F_a v_a, v = (ħ/m)κ).
    pub fn work(&self, hbar_over_mass: f64) -> V3 {
        std::array::from_fn(|a| self.force[a] * hbar_over_mass * self.kappa[a])
    }
}

/// A model's readings along one path, its photon classes' fast indices, the work they took, and its state at the end
/// (the sector vector: the path goes on from there, `Form::along_from`).
#[derive(Clone, Debug, Default)]
pub struct Series {
    pub classes: Vec<[i64; 2]>,
    pub readings: Vec<Reading>,
    pub counts: Counts,
    pub state: Vec<C64>,
    /// The sector states at the readings a run was asked to keep (`Form::along_keeping`): the landing check's starts
    /// (# The landing check).
    pub kept: Vec<(f64, Vec<C64>)>,
}

impl Series {
    /// The scheme's trace defect along the series (module documentation, # The readings): the largest |tr ρ_P − 1| over
    /// its readings -- every model's generator keeps the trace (i(H − H†) = Σ J†J), so the defect is the scheme's own,
    /// a strict lower bound of its error in the trace norm, ‖ρ_num − ρ‖₁ ≥ |tr ρ_num − 1|; none of no reading.
    pub fn trace_defect(&self) -> f64 {
        self.readings.iter().map(|r| (r.trace - 1.0).abs()).fold(0.0, larger)
    }
}

/// A model on its sector (module documentation, # The sector, # The generator): the structure every node shares, made
/// once from one node.
pub struct Form<'a> {
    pub model: &'a Selected,
    n: usize,
    /// The ground-holding blocks (places in P), and each place's (block, position in it); usize::MAX off the sector.
    blocks: Vec<Vec<usize>>,
    block_of: Vec<(usize, usize)>,
    /// The sector's block pairs and where each one's matrix starts; the vector's length.
    pairs: Vec<(usize, usize)>,
    offsets: Vec<usize>,
    dim: usize,
    /// H's slots' beats; where each block's entries start within a slot, and a slot's length.
    h_beats: Vec<usize>,
    h_block: Vec<usize>,
    h_slot: usize,
    /// The jumps: their elements' count (in the node's order); the recycling's groups by
    /// the beat Δ = Ω_b − Ω_b′ of two components of one channel (rad/µs; one exact difference one group, its value
    /// correctly rounded -- tilt2's 17 differences met as 51 numbers to the bit, each group R's whole size --: every
    /// (b, b) one, at Δ = 0), at e^{−iΔt}, and its terms -- (group, R's place, two elements): R_group[place] +=
    /// J_e J_e′* --, in the order of their places.
    elements: usize,
    beats_r: Vec<f64>,
    terms: Vec<(u32, u32, u32, u32)>,
    /// R's places some term takes (`terms`; the places of R^T, column-major over rows), group by group, each group's
    /// increasing; the groups' bounds among them (`r_groups`: group g's are r_groups[g]..r_groups[g + 1]); each term's
    /// place among them (`r_slot`). A node's R is held at them alone (`Amps::r`): tilt2's 21 groups hold 64 306 of their
    /// 21 × 36 864 places, TET's one group 32 574 of its 36 864.
    r_places: Vec<u32>,
    r_groups: Vec<usize>,
    r_slot: Vec<u32>,
    /// R (# The generator) as R^T: its columns' and rows' sector places, column-major over rows.
    cols: Vec<usize>,
    rows: Vec<usize>,
    /// The beats' values (rad/µs; `Selected::components`).
    values: Vec<f64>,
    /// The states of P; whether the model folds any state (its rows then lift Q's).
    states: Vec<usize>,
    folds: bool,
    classes: Vec<[i64; 2]>,
    /// Per class its photon's wave number (1/µm; # The motion's readings).
    photon: Vec<f64>,
    /// The beams an atom absorbs from (all but the tweezer's): each its index, its wave vector (1/µm) and its coupling's
    /// strength at its focus, ‖X_l‖²_F + ‖R_l‖²_F (# The motion's readings).
    absorbing: Vec<(usize, V3, f64)>,
    /// A substep's remainder's counted work (real multiply-adds): ΔH's sandwich over the sector's pairs, Σ b_x b_y (b_x +
    /// b_y) complex multiply-adds, and R's places (`Counts::work`); a sandwich's alone (`sandwich`).
    product: f64,
    sandwiched: f64,
    hbar_over_mass: f64,
}

/// A node's amplitudes on the sector (# The nodes): H's slots' blocks and the recycling's R per group of its beats (R^T,
/// at the places its terms take, `Form::r_places`: the others are zero at every node) -- the interpolated quantities: H
/// and R are linear in the Liouvillian, each beat's amplitude a smooth function of the time along the path.
#[derive(Clone, Debug)]
struct Amps {
    h: Vec<C64>,
    r: Vec<C64>,
    /// Per R group and column Σ_rows |R| (the 1-norm's bound of a combination: Σ |w| of these).
    columns: Vec<f64>,
}

impl<'a> Form<'a> {
    /// The model's form (module documentation): its sector from its blocks and its jumps' structure, read on the node at
    /// the point `at` at rest (every node holds the layout's beats and elements, whatever their values there).
    pub fn new(model: &'a Selected, at: V3) -> Form<'a> {
        let node = model.node(at, [0.0; 3], false);
        let states = model.rs.keep.clone();
        let n = states.len();
        let ground: Vec<bool> = states.iter().map(|&s| s < GROUND).collect();
        let blocks: Vec<Vec<usize>> = model.blocks().iter().filter(|b| b.iter().any(|&x| ground[x])).cloned().collect();
        let mut block_of = vec![(usize::MAX, usize::MAX); n];
        for (x, b) in blocks.iter().enumerate() {
            for (i, &p) in b.iter().enumerate() {
                block_of[p] = (x, i);
            }
        }
        // The jumps: per channel (class, axis) its components, each its beat and its elements (flat index, ground row,
        // source column: places in P).
        let mut channels: Vec<Vec<(usize, Vec<(usize, usize, usize)>)>> = Vec::new();
        let mut elements = 0;
        let mut classes = Vec::new();
        for class in &node.classes {
            classes.push(class.fast);
            for axis in &class.axes {
                let mut components = Vec::new();
                for jump in axis {
                    let at: Vec<(usize, usize, usize)> = jump.elements.iter().enumerate().map(|(k, e)| (elements + k, e.at / n, e.at % n)).collect();
                    elements += jump.elements.len();
                    components.push((jump.beat, at));
                }
                channels.push(components);
            }
        }
        // The sector: the closure of the ground-holding blocks' own pairs under the jumps.
        let nb = blocks.len();
        let mut held = vec![false; nb * nb];
        for x in 0..nb {
            held[x * nb + x] = true;
        }
        let of = |p: usize| block_of[p].0;
        loop {
            let mut grown = false;
            for ch in &channels {
                let entries: Vec<(usize, usize)> = ch.iter().flat_map(|c| c.1.iter().map(|e| (e.1, e.2))).collect();
                for &(g, s) in &entries {
                    for &(g2, s2) in &entries {
                        let (bs, bs2, bg, bg2) = (of(s), of(s2), of(g), of(g2));
                        if bs == usize::MAX || bs2 == usize::MAX || bg == usize::MAX || bg2 == usize::MAX {
                            continue;
                        }
                        if held[bs * nb + bs2] && !held[bg * nb + bg2] {
                            held[bg * nb + bg2] = true;
                            grown = true;
                        }
                    }
                }
            }
            if !grown {
                break;
            }
        }
        let mut pairs: Vec<(usize, usize)> = (0..nb).map(|x| (x, x)).collect();
        for x in 0..nb {
            for y in 0..nb {
                if x != y && held[x * nb + y] {
                    pairs.push((x, y));
                }
            }
        }
        let mut offsets = Vec::with_capacity(pairs.len() + 1);
        let mut dim = 0;
        for &(x, y) in &pairs {
            offsets.push(dim);
            dim += blocks[x].len() * blocks[y].len();
        }
        offsets.push(dim);
        let mut pair_of = vec![usize::MAX; nb * nb];
        for (k, &(x, y)) in pairs.iter().enumerate() {
            pair_of[x * nb + y] = k;
        }
        let place = |a: usize, b: usize| -> Option<usize> {
            let ((x, i), (y, j)) = (block_of[a], block_of[b]);
            if x == usize::MAX || y == usize::MAX || pair_of[x * nb + y] == usize::MAX {
                return None;
            }
            Some(offsets[pair_of[x * nb + y]] + i * blocks[y].len() + j)
        };
        // R's rows: the ground pairs of the sector; its columns: the source pairs a channel's two elements take; its
        // terms per pair of components of a channel.
        let mut row_of = vec![u32::MAX; n * n];
        let mut rows = Vec::new();
        let mut col_of = vec![u32::MAX; n * n];
        let mut cols = Vec::new();
        let mut raw: Vec<(usize, usize, usize, usize, usize)> = Vec::new();
        let (values, exact) = model.components();
        let mut beats_r: Vec<f64> = Vec::new();
        let mut keys: Vec<Exact> = Vec::new();
        for ch in &channels {
            for (b1, c1) in ch {
                for (b2, c2) in ch {
                    let key = if b1 == b2 { ([0; 3], Vec::new()) } else { difference(&exact[*b1], &exact[*b2]) };
                    let bp = keys.iter().position(|k| *k == key).unwrap_or_else(|| {
                        beats_r.push(if key.0 == [0; 3] && key.1.is_empty() { 0.0 } else { model.value_of(&key) });
                        keys.push(key);
                        beats_r.len() - 1
                    });
                    for &(e1, g, s) in c1 {
                        for &(e2, g2, s2) in c2 {
                            if let (Some(r), Some(c)) = (place(g, g2), place(s, s2)) {
                                if row_of[g * n + g2] == u32::MAX {
                                    row_of[g * n + g2] = rows.len() as u32;
                                    rows.push(r);
                                }
                                if col_of[s * n + s2] == u32::MAX {
                                    col_of[s * n + s2] = cols.len() as u32;
                                    cols.push(c);
                                }
                                raw.push((bp, g * n + g2, s * n + s2, e1, e2));
                            }
                        }
                    }
                }
            }
        }
        let nr = rows.len();
        let mut terms: Vec<(u32, u32, u32, u32)> = raw.iter().map(|&(bp, gg, ss, e1, e2)| (bp as u32, (col_of[ss] as usize * nr + row_of[gg] as usize) as u32, e1 as u32, e2 as u32)).collect();
        terms.sort_unstable();
        // R's places the terms take, by group (`Form::r_places`).
        let (mut r_places, mut r_groups, mut r_slot) = (Vec::new(), vec![0usize; beats_r.len() + 1], Vec::with_capacity(terms.len()));
        let mut last = None;
        for &(bp, place, ..) in &terms {
            if last != Some((bp, place)) {
                r_places.push(place);
                last = Some((bp, place));
                r_groups[bp as usize + 1] = r_places.len();
            }
            r_slot.push(r_places.len() as u32 - 1);
        }
        for g in 1..r_groups.len() {
            r_groups[g] = r_groups[g].max(r_groups[g - 1]);
        }
        let mut h_block = Vec::with_capacity(nb);
        let mut h_slot = 0;
        for b in &blocks {
            h_block.push(h_slot);
            h_slot += b.len() * b.len();
        }
        // Each class's photon wave number (# The motion's readings): its photon forms K = k∘L on the node at `at` with
        // them, Σ Re(L̄ K)/Σ |L|² over its elements (one class one line: its elements' k differ by the hyperfine
        // splitting over the optical frequency); a class that scatters nothing there, the scene's shortest emitted
        // wavelength's (the largest k).
        let request = &model.scene.request;
        let largest = (0..STATES).flat_map(|i| (0..STATES).map(move |j| (i, j))).filter(|&(i, j)| request.emission.iter().any(|c| c[(i, j)] != C64::ZERO)).map(|(i, j)| model.scene.jumps().wave_number[(i, j)].re.abs()).fold(0.0, larger);
        let photon = model
            .node(at, [0.0; 3], true)
            .classes
            .iter()
            .map(|c| {
                let (mut lk, mut ll) = (0.0, 0.0);
                for e in c.axes.iter().flatten().flat_map(|j| &j.elements) {
                    lk += (e.value.conj() * e.kick).re;
                    ll += e.value.norm_sqr();
                }
                if ll > 0.0 { lk / ll } else { largest }
            })
            .collect();
        let absorbing = request
            .beams
            .iter()
            .enumerate()
            .filter(|&(l, _)| l != request.trap_beam)
            .map(|(l, b)| {
                let size = crate::linalg::norm3(b.direction);
                let strength: f64 = b.coupling.data.iter().chain(&b.counter_coupling.data).map(|z| z.norm_sqr()).sum();
                (l, b.direction.map(|x| b.wave_number * x / size), strength)
            })
            .filter(|x| x.2 > 0.0)
            .collect();
        let sandwiched = crate::decide::COMPLEX * pairs.iter().map(|&(x, y)| (blocks[x].len() * blocks[y].len() * (blocks[x].len() + blocks[y].len())) as f64).sum::<f64>();
        let product = sandwiched + crate::decide::COMPLEX * r_places.len() as f64;
        let form = Form {
            model,
            n,
            blocks,
            block_of,
            pairs,
            offsets,
            dim,
            h_beats: node.generator.iter().map(|g| g.0).collect(),
            h_block,
            h_slot,
            elements,
            beats_r,
            terms,
            r_places,
            r_groups,
            r_slot,
            cols,
            rows,
            values,
            folds: !model.rs.folded.is_empty(),
            states,
            classes,
            photon,
            absorbing,
            product,
            sandwiched,
            hbar_over_mass: model.hbar_over_mass(),
        };
        assert!(form.beats_known(&node), "a node's beat outside the model's values");
        form
    }

    /// A node's amplitudes' counted work (`amps`; real multiply-adds): R's terms, one complex multiply-add each, and H's
    /// slots copied.
    fn amplitudes_work(&self) -> f64 {
        crate::decide::COMPLEX * self.terms.len() as f64 + (self.h_beats.len() * self.h_slot) as f64
    }

    /// The form's sizes (the measurements' line): P's states, the blocks, the sector, H's slots, R and its groups, the
    /// terms, the jumps' elements, the photon classes.
    #[cfg(test)]
    pub(crate) fn shape(&self) -> String {
        format!(
            "n {}, blocks {:?}, pairs {}, dim {}, H beats {} (nonzero {}), H slot {}, R {} x {} by {} beat groups, terms {}, elements {}, classes {}",
            self.n, self.blocks.iter().map(|b| b.len()).collect::<Vec<_>>(), self.pairs.len(), self.dim, self.h_beats.len(), self.h_beats.iter().filter(|&&b| self.values[b] != 0.0).count(), self.h_slot,
            self.cols.len(), self.rows.len(), self.beats_r.len(), self.terms.len(), self.elements, self.classes.len()
        )
    }

    /// The beats' values (rad/µs): H's nonzero ones, and R's groups' (the measurements' line).
    #[cfg(test)]
    pub(crate) fn beat_values(&self) -> (Vec<f64>, Vec<f64>) {
        (self.h_beats.iter().map(|&b| self.values[b]).filter(|&v| v != 0.0).collect(), self.beats_r.clone())
    }

    /// The sector vector's length (module documentation, # The sector).
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The ground-holding blocks' sizes and the sector's pairs.
    pub fn blocks(&self) -> (Vec<usize>, usize) {
        (self.blocks.iter().map(Vec::len).collect(), self.pairs.len())
    }

    /// Whether every beat a node holds has its value.
    fn beats_known(&self, node: &Node) -> bool {
        let v = self.values.len();
        node.generator.iter().all(|g| g.0 < v)
            && node.gradient.iter().all(|g| g.0 < v)
            && node.classes.iter().all(|c| c.axes.iter().all(|a| a.iter().all(|j| j.beat < v)))
    }

    /// A beat's phase e^{−iΩt} at the time t.
    fn phase(&self, beat: usize, t: f64) -> C64 {
        let v = self.values[beat];
        if v == 0.0 {
            C64::ONE
        } else {
            C64::cis(-v * t)
        }
    }

    /// The model's node at the path's place and momentum at the time t.
    fn node_at(&self, walk: &Walk, t: f64) -> Node {
        let (r, kappa) = walk.at(self.hbar_over_mass, t);
        self.model.node(r, kappa, false)
    }

    /// A node's amplitudes on the sector (`Amps`): H's slots' blocks; R per group of beats from the jumps' elements, and
    /// R's columns' bounds (the operator's norm needs them, `Form::op_into`).
    fn amps(&self, node: &Node) -> Amps {
        self.amps_with(node, true)
    }

    /// `amps`, R's columns' bounds made only where `columns` (the exponential scheme's substeps take none: none there).
    /// (Until 2026-11-23 also the blocks' spectral width, which the Magnus segments' rule read: no reader since they went,
    /// 37d0047d, its eigendecompositions made at every dense Liouvillian's node for nothing.)
    fn amps_with(&self, node: &Node, columns: bool) -> Amps {
        let n = self.n;
        let mut h = vec![C64::ZERO; self.h_beats.len() * self.h_slot];
        for (k, (_, m)) in node.generator.iter().enumerate() {
            for (x, b) in self.blocks.iter().enumerate() {
                let at = k * self.h_slot + self.h_block[x];
                let w = b.len();
                for (i, &p) in b.iter().enumerate() {
                    for (j, &q) in b.iter().enumerate() {
                        h[at + i * w + j] = m.data[p * n + q];
                    }
                }
            }
        }
        let mut j = Vec::with_capacity(self.elements);
        for class in &node.classes {
            for axis in &class.axes {
                for jump in axis {
                    j.extend(jump.elements.iter().map(|e| e.value));
                }
            }
        }
        let mut r = vec![C64::ZERO; self.r_places.len()];
        for (&(_, _, e1, e2), &slot) in self.terms.iter().zip(&self.r_slot) {
            r[slot as usize] += j[e1 as usize] * j[e2 as usize].conj();
        }
        let columns = if columns { self.columns_of(&r) } else { Vec::new() };
        Amps { h, r, columns }
    }

    /// R's columns' moduli summed over its rows, per group and column (`Amps::columns`), of R at its places `r`.
    fn columns_of(&self, r: &[C64]) -> Vec<f64> {
        let (nr, nc) = (self.rows.len().max(1), self.cols.len());
        let mut out = vec![0.0; self.beats_r.len() * nc];
        for g in 0..self.beats_r.len() {
            for k in self.r_groups[g]..self.r_groups[g + 1] {
                out[g * nc + self.r_places[k] as usize / nr] += r[k].abs();
            }
        }
        out
    }

    /// z ← z + c R_g of a node's R at its places `r` (`Amps::r`), its group g's, into R^T's places `z`.
    fn r_onto(&self, z: &mut [C64], c: C64, r: &[C64], g: usize) {
        let (from, to) = (self.r_groups[g], self.r_groups[g + 1]);
        for (&place, &x) in self.r_places[from..to].iter().zip(&r[from..to]) {
            z[place as usize] += c * x;
        }
    }

    /// The operator h·L′ of one exponential into `op` (`Op`): its generator Σ (weight × the amplitudes at their phases
    /// at the time) over the list `of` of (node's amplitudes, weight, time) -- a CF4 exponential's c₁, c₂ times each
    /// Gauss point's interpolation weights --, each node's weights merged first, so that its H and R are read once:
    /// H = Σ_k e^{−iΩ_k t} H_k, R = Σ_g e^{−iΔ_g t} R_g. L′ = L − 2 Im(c)·1 for the shift c of H by its mean diagonal
    /// over the sector (−i(cρ − ρc*) = 2 Im(c) ρ, a scalar factor of the exponential); with its 1-norm's bound, column
    /// by column: the off-diagonal moduli of H's two columns, the diagonal's difference, R's column bounded by its
    /// nodes' (the Taylor degree and scaling from a bound: the series is still summed to the rounding).
    fn op_into(&self, of: &[(&Amps, f64, f64)], h: f64, op: &mut Op) {
        op.fit(self);
        let (slots, groups) = (self.h_beats.len(), self.beats_r.len());
        let mut nodes: Vec<&Amps> = Vec::with_capacity(of.len());
        op.wh.clear();
        op.wr.clear();
        for &(a, w, t) in of {
            let i = nodes.iter().position(|x| std::ptr::eq(*x, a)).unwrap_or_else(|| {
                nodes.push(a);
                op.wh.resize(nodes.len() * slots, C64::ZERO);
                op.wr.resize(nodes.len() * groups, C64::ZERO);
                nodes.len() - 1
            });
            for (k, &beat) in self.h_beats.iter().enumerate() {
                op.wh[i * slots + k] += self.phase(beat, t) * w;
            }
            for (g, &delta) in self.beats_r.iter().enumerate() {
                op.wr[i * groups + g] += if delta == 0.0 { C64::real(w) } else { C64::cis(-delta * t) * w };
            }
        }
        op.hc.fill(C64::ZERO);
        for (i, a) in nodes.iter().enumerate() {
            for k in 0..slots {
                mul_add(false, &mut op.hc, op.wh[i * slots + k], &a.h[k * self.h_slot..(k + 1) * self.h_slot]);
            }
        }
        let mut sum = C64::ZERO;
        let mut count = 0;
        for (x, b) in self.blocks.iter().enumerate() {
            let w = b.len();
            for i in 0..w {
                sum += op.hc[self.h_block[x] + i * w + i];
                count += 1;
            }
        }
        let shift = sum * (1.0 / count as f64);
        for (x, b) in self.blocks.iter().enumerate() {
            let w = b.len();
            let at = self.h_block[x];
            let (left, right, diag, off) = (&mut op.a1[x], &mut op.a2[x], &mut op.diag[x], &mut op.off[x]);
            off.fill(0.0);
            for i in 0..w {
                for j in 0..w {
                    let raw = op.hc[at + i * w + j];
                    let z = if i == j { raw - shift } else { raw };
                    left[i * w + j] = C64::new(0.0, -h) * z;
                    // (H − c)† at (j, i), times i h: the right factor of ρ(H − c)†.
                    right[j * w + i] = C64::new(0.0, h) * z.conj();
                    if i == j {
                        diag[i] = z;
                    } else {
                        off[j] += z.abs();
                    }
                }
            }
        }
        // R and its columns' bound: Σ_j Σ_g |h w_jg| Σ_rows |R_jg| ≥ Σ_rows |R's combination|.
        let ncols = self.cols.len();
        op.rt.fill(C64::ZERO);
        op.colsum.fill(0.0);
        for (i, a) in nodes.iter().enumerate() {
            for g in 0..groups {
                let c = op.wr[i * groups + g] * h;
                self.r_onto(&mut op.rt, c, &a.r, g);
                let m = c.abs();
                for (ci, &place) in self.cols.iter().enumerate() {
                    op.colsum[place] += m * a.columns[g * ncols + ci];
                }
            }
        }
        let mut norm = 0.0_f64;
        for (p, &(x, y)) in self.pairs.iter().enumerate() {
            let wy = self.blocks[y].len();
            for k in 0..self.blocks[x].len() {
                for l in 0..wy {
                    let s = h * (op.off[x][k] + op.off[y][l] + (op.diag[x][k] - op.diag[y][l].conj()).abs()) + op.colsum[self.offsets[p] + k * wy + l];
                    norm = larger(norm, s);
                }
            }
        }
        op.eta = (2.0 * shift.im * h).exp();
        op.norm = norm;
    }



    /// The sector vector of a state (normalized), by the request's states. A start is a state of the pilot's model's
    /// cluster -- a ground state with its dressing -- and every model judged holds the pilot's model's states (select.rs
    /// # The candidates), a dressed state in the block of the ground state it dresses: so each start's state is in P and
    /// in a ground-holding block, an invariant asserted (no request reaches it: an initial excited population is refused
    /// with the request, request.rs).
    fn initial(&self, start: &[(usize, C64)]) -> Vec<C64> {
        let mut psi = vec![C64::ZERO; self.n];
        for &(s, z) in start {
            let p = self.states.iter().position(|&x| x == s).unwrap_or_else(|| panic!("a start's state {s} outside P: every model judged holds the pilot's model's states"));
            assert!(self.block_of[p].0 != usize::MAX, "a start's state {s} in no ground-holding block: a dressed state is in its ground state's block");
            psi[p] = z;
        }
        let norm: f64 = psi.iter().map(|z| z.norm_sqr()).sum();
        let mut v = vec![C64::ZERO; self.dim];
        for a in 0..self.n {
            for b in 0..self.n {
                if let Some(p) = self.place(a, b) {
                    v[p] = psi[a] * psi[b].conj() * (1.0 / norm);
                }
            }
        }
        v
    }

    /// The sector place of the element (a, b) of ρ (places in P), None off the sector.
    fn place(&self, a: usize, b: usize) -> Option<usize> {
        let ((x, i), (y, j)) = (self.block_of[a], self.block_of[b]);
        if x == usize::MAX || y == usize::MAX {
            return None;
        }
        let k = self.pairs.iter().position(|&p| p == (x, y))?;
        Some(self.offsets[k] + i * self.blocks[y].len() + j)
    }

    /// ρ on P (n × n, row by row) from the sector vector: zero off the sector.
    fn density(&self, v: &[C64]) -> Vec<C64> {
        let n = self.n;
        let mut rho = vec![C64::ZERO; n * n];
        for (k, &(x, y)) in self.pairs.iter().enumerate() {
            let (bx, by) = (&self.blocks[x], &self.blocks[y]);
            for (i, &a) in bx.iter().enumerate() {
                for (j, &b) in by.iter().enumerate() {
                    rho[a * n + b] = v[self.offsets[k] + i * by.len() + j];
                }
            }
        }
        rho
    }

    /// The readings of the sector state v on the node at the time t (module documentation, # The readings).
    fn read(&self, node: &Node, walk: &Walk, t: f64, v: &[C64], counts: &mut Counts) -> Reading {
        let clock = std::time::Instant::now();
        let n = self.n;
        let rho = self.density(v);
        let trace: f64 = (0..n).map(|a| rho[a * n + a].re).sum();
        let (r, kappa) = walk.at(self.hbar_over_mass, t);
        // The rows: the diagonal where nothing is folded; else through ρ's eigenvectors on the sector's places, each
        // one's rows as shown un-normalized by its own sum (P's part over ‖ψ‖²).
        let mut rows = [0.0; STATES];
        if !self.folds {
            for (a, &s) in self.states.iter().enumerate() {
                rows[s] = rho[a * n + a].re;
            }
        } else {
            let support: Vec<usize> = (0..n).filter(|&a| self.block_of[a].0 != usize::MAX).collect();
            let m = support.len();
            let sub = Mat::from_fn(m, m, |i, j| rho[support[i] * n + support[j]]);
            let eigen = hermitian_eigen(&sub);
            let point = Point { r, kappa, t };
            let mut psi = vec![C64::ZERO; n];
            for k in 0..m {
                let lambda = eigen.values[k];
                psi.fill(C64::ZERO);
                for (i, &a) in support.iter().enumerate() {
                    psi[a] = eigen.vectors[(i, k)];
                }
                let shown = self.model.readout(0, &point, &psi);
                counts.readout_calls += 1;
                let on_p: f64 = self.states.iter().map(|&s| shown[s]).sum();
                for (x, y) in rows.iter_mut().zip(shown) {
                    *x += lambda * y / on_p;
                }
            }
        }
        let total: f64 = rows.iter().sum();
        let rows = rows.map(|x| x / total);
        // The rates and the force on the node at t: J(t), ∇J(t) by class and axis; H_eff's gradient on ρ the model's own
        // contraction (model/selected.rs `SelectedMiddle::contract`, every beat at its phase).
        let mut rates = vec![0.0; self.classes.len()];
        let mut force = crate::model::selected::SelectedMiddle::contract(node, |beat| Some(self.phase(beat, t)), &rho).map(|z| -z.re);
        let mut emitted = [0.0; 3];
        // The entries' pairs read (`Counts::work`: four complex multiply-adds each -- the rate and the absorbed mean on
        // three axes).
        let mut matched = 0usize;
        let mut entries: Vec<(usize, usize, C64, [C64; 3])> = Vec::new();
        for (c, class) in node.classes.iter().enumerate() {
            // Per Cartesian axis of the emission its rate tr(J_a ρ J_a†), for the photons' second moment.
            let mut along = [0.0; 3];
            for (b, axis) in class.axes.iter().enumerate() {
                entries.clear();
                for jump in axis {
                    let f = self.phase(jump.beat, t);
                    for e in &jump.elements {
                        let (g, s) = (e.at / n, e.at % n);
                        let (z, d) = (f * e.value, e.gradient.map(|x| f * x));
                        match entries.iter_mut().find(|x| x.0 == g && x.1 == s) {
                            Some(x) => {
                                x.2 += z;
                                for a in 0..3 {
                                    x.3[a] += d[a];
                                }
                            }
                            None => entries.push((g, s, z, d)),
                        }
                    }
                }
                // tr(J ρ J†) = Σ J[g,u] ρ[u,s] conj(J[g,s]); Im tr(ρ J†∇J) = Im Σ conj(J[g,u]) ∇J[g,s] ρ[s,u] -- the
                // model's photons' rate and absorbed mean (model.rs `Moments::of`) on ρ, read on ρ itself: through ρ's
                // eigenvectors, each a trajectory's state read by the model's own photons, the readings took 4.7 times as
                // long (PGC3D's paths 2026-11-23: 25.6 → 119.5 thread seconds of 282, the numbers the same).
                let mut rate = C64::ZERO;
                let mut absorbed = [C64::ZERO; 3];
                for &(g, u, z, _) in entries.iter() {
                    for &(g2, s, w, d) in entries.iter() {
                        if g2 != g {
                            continue;
                        }
                        matched += 1;
                        rate += z * rho[u * n + s] * w.conj();
                        for a in 0..3 {
                            absorbed[a] += z.conj() * d[a] * rho[s * n + u];
                        }
                    }
                }
                rates[c] += rate.re / trace;
                along[b] = rate.re / trace;
                for a in 0..3 {
                    force[a] += absorbed[a].im;
                }
            }
            // The photons' second moment per lab axis: k_c² λ_c ⟨n_a²⟩, the dipole pattern's ⟨n_a²⟩ λ = (2/5) tr G −
            // (1/5) G_aa (G_ab = tr(J_a ρ J_b†); # The motion's readings).
            let k2 = self.photon[c] * self.photon[c];
            let lambda: f64 = along.iter().sum();
            for a in 0..3 {
                emitted[a] += k2 * (0.4 * lambda - 0.2 * along[a]);
            }
        }
        let force = force.map(|f| f / trace);
        // The reading's counted work: ρ on P, the rows (through ρ's eigenvectors where the model folds: its decomposition
        // and a readout a vector), the gradient's contractions and the entries' pairs.
        let support = if self.folds { (0..n).filter(|&a| self.block_of[a].0 != usize::MAX).count() } else { 0 };
        counts.work += crate::decide::COMPLEX * ((n * n) as f64 + (support * support * support) as f64 + (support * n * n) as f64 + (3 * node.gradient.len() * n * n) as f64 + 4.0 * matched as f64);
        // The light's absorbed photon at the place (# The motion's readings).
        let (mut kick, mut kick2, mut weight) = ([0.0; 3], [0.0; 3], 0.0);
        let light = self.model.light();
        for &(l, k, strength) in &self.absorbing {
            let w = strength * light.beams[l].amplitude_to(r, 0).value.norm_sqr();
            weight += w;
            for a in 0..3 {
                kick[a] += w * k[a];
                kick2[a] += w * k[a] * k[a];
            }
        }
        if weight > 0.0 {
            (kick, kick2) = (kick.map(|x| x / weight), kick2.map(|x| x / weight));
        }
        counts.readouts += 1;
        counts.readout_seconds += clock.elapsed().as_secs_f64();
        Reading { t, r, kappa, rows, rates, force, total, trace, emitted, kick, kick2 }
    }

    /// The model's readings along a path from `from` to `to` from the state `start` (module documentation): its
    /// series, and Err where the state is not the model's.
    /// What the opening of a stretch from the state `start` at `from` books at `pieces` a µs (`pieces_in`) and the cutoff
    /// `cutoff` (# The start), without the stretch: the bound alone.
    pub fn booked_at(&self, walk: &Walk, start: &[(usize, C64)], from: f64, pieces: f64, cutoff: f64) -> Result<f64, String> {
        let v = self.initial(start);
        let knots = walk.knots(from, walk.end().max(from + f64::MIN_POSITIVE));
        let (first, next) = (self.amps_with(&self.node_at(walk, from), false), self.amps_with(&self.node_at(walk, knots[1]), false));
        Ok(self.opening(&first, &next, from, knots[1], pieces_in(from, knots[1], pieces), &v, cutoff)?.2)
    }

    pub fn along(&self, walk: &Walk, start: &[(usize, C64)], from: f64, to: f64, scheme: Scheme) -> Result<Series, String> {
        self.along_from(walk, self.initial(start), from, to, scheme)
    }

    /// `along` at an exponential scheme, its states kept at the readings at the times `keep` (`Series::kept`).
    pub fn along_keeping(&self, walk: &Walk, start: &[(usize, C64)], from: f64, to: f64, scheme: Scheme, keep: &[f64]) -> Result<Series, String> {
        Ok(self.family(walk, self.initial(start), from, to, &[scheme], keep)?.swap_remove(0))
    }

    /// `along_family`, each member's states kept at the readings at the times `keep` (`Series::kept`).
    pub fn along_family_keeping(&self, walk: &Walk, start: &[(usize, C64)], from: f64, to: f64, members: &[Scheme], keep: &[f64]) -> Result<Vec<Series>, String> {
        self.family(walk, self.initial(start), from, to, members, keep)
    }

    /// `along` at several exponential schemes from one start, a family (module documentation, # The nodes): per scheme
    /// its series, each the same as its run alone to the bit, each node made once for them all.
    pub fn along_family(&self, walk: &Walk, start: &[(usize, C64)], from: f64, to: f64, members: &[Scheme]) -> Result<Vec<Series>, String> {
        self.family(walk, self.initial(start), from, to, members, &[])
    }

    /// `along` from a sector state at `from` (a series' end state: the same path gone on, its knots and nodes the
    /// same as in one stretch where `from` is a knot).
    pub fn along_from(&self, walk: &Walk, state: Vec<C64>, from: f64, to: f64, scheme: Scheme) -> Result<Series, String> {
        Ok(self.family(walk, state, from, to, &[scheme], &[])?.swap_remove(0))
    }

    /// The form's excited states' decays at its point (# The landing check): the optical memory 2 ln(2/ε)/Γ_min, Γ_min the
    /// slowest decay −2 Im H_ee of P's excited states on the node there -- the time a ground state's optical coherences
    /// take to grow to the steady state's from nothing, to the rounding --, and the fastest decay Γ_max (1/µs); none of a
    /// model without excited states.
    pub fn optical(&self, at: V3) -> Option<(f64, f64)> {
        let node = self.model.node(at, [0.0; 3], false);
        let n = self.n;
        let decays: Vec<f64> = node.generator.iter().filter(|(beat, _)| self.values[*beat] == 0.0).flat_map(|(_, g)| (0..n).filter(|&a| self.states[a] >= GROUND).map(move |a| -2.0 * g.data[a * n + a].im)).filter(|g| *g > 0.0).collect();
        let (slowest, fastest) = (decays.iter().copied().fold(f64::INFINITY, crate::linalg::smaller), decays.iter().copied().fold(0.0, larger));
        slowest.is_finite().then(|| (2.0 * (2.0 / f64::EPSILON).ln() / slowest, fastest))
    }

    /// The landing state after a jump at the time t from the sector state v (# The landing check): R(t)v, the
    /// recycling's image, as the ground density matrix by the request's ground states over its trace (the jump rate on
    /// v), and that trace; none where no jump is made there.
    pub fn landed(&self, walk: &Walk, t: f64, v: &[C64]) -> Option<([[C64; GROUND]; GROUND], f64)> {
        #[cfg(test)]
        let _timed = Checked::new(0);
        let node = self.node_at(walk, t);
        let amps = self.amps_with(&node, false);
        let mut rt = vec![C64::ZERO; self.cols.len() * self.rows.len()];
        self.r_at(&[(&amps, 1.0)], t, &mut rt);
        let gathered: Vec<C64> = self.cols.iter().map(|&c| v[c]).collect();
        let mut product = vec![C64::ZERO; self.rows.len()];
        fused_product(&gathered, &rt, (1, self.cols.len(), self.rows.len()), &mut product);
        let mut x = vec![C64::ZERO; self.dim];
        for (p, &r) in product.iter().zip(&self.rows) {
            x[r] += *p;
        }
        let rho = self.density(&x);
        let n = self.n;
        let mut out = [[C64::ZERO; GROUND]; GROUND];
        let mut trace = 0.0;
        for a in 0..n {
            for b in 0..n {
                let (sa, sb) = (self.states[a], self.states[b]);
                if sa < GROUND && sb < GROUND {
                    out[sa][sb] = rho[a * n + b];
                    if a == b {
                        trace += rho[a * n + a].re;
                    }
                }
            }
        }
        (trace > 0.0).then(|| (out.map(|row| row.map(|z| z * (1.0 / trace))), trace))
    }

    /// A ground density matrix (by the request's ground states) as a sector state of the form.
    pub fn from_ground(&self, g: &[[C64; GROUND]; GROUND]) -> Vec<C64> {
        let mut v = vec![C64::ZERO; self.dim];
        for a in 0..self.n {
            for b in 0..self.n {
                let (sa, sb) = (self.states[a], self.states[b]);
                if sa < GROUND && sb < GROUND {
                    if let Some(p) = self.place(a, b) {
                        v[p] = g[sa][sb];
                    }
                }
            }
        }
        v
    }

    /// The landing check of one landing (# The landing check): from the landed ground state `landed` at the time t, the
    /// form along the walk for `span` µs at the exponential scheme's knobs `knobs` (one piece a knot), on knots that grow
    /// from a quarter of the fastest excited decay's time 1/Γ (`decay`) by 2^{1/4} a knot -- the transient resolved
    /// where it is, its tail at the walk's own spacing --; against its own run `own` along the walk (its readings
    /// interpolated linearly between the walk's knots): the excess photons n_ex = ∫(λ(τ|L) − λ̄)dτ and the excess impulse
    /// ∫(F(τ|L) − F̄)dτ (1/µm) by the trapezoid, and the stretch's counted work (`Counts::work`). Err where the state is
    /// not the form's.
    #[allow(clippy::too_many_arguments)]
    pub fn landing(&self, walk: &Walk, t: f64, landed: &[[C64; GROUND]; GROUND], span: f64, decay: f64, knobs: (f64, f64, f64), own: &Series, budget: f64) -> Result<(f64, V3, f64, f64), String> {
        #[cfg(test)]
        let _timed = Checked::new(3);
        let hm = self.hbar_over_mass;
        // The Q term's share of the total heating per unit of Q there (`Reading::heating_at`: (ħ/m)/2 λ Q k̄_a² against
        // the whole), summed over the axes as the total T sums them: what a change of Q = 2 n_ex moves in the total T's
        // account, relative.
        let weight = own.readings.iter().min_by(|a, b| (a.t - t).abs().total_cmp(&(b.t - t).abs())).map_or(0.0, |x| {
            let lambda: f64 = x.rates.iter().sum();
            let whole: f64 = x.heating(hm).iter().sum();
            if whole > 0.0 { 0.5 * hm * lambda * x.kick.iter().map(|k| k * k).sum::<f64>() / whole } else { 0.0 }
        });
        // The most a part of the stretch can move n_ex: |λ(τ|L) − λ̄| at most the own run's largest rate there times its length.
        let most = |from: f64, to: f64| own.readings.iter().filter(|x| x.t >= from && x.t <= to).map(|x| x.rates.iter().sum::<f64>()).fold(0.0, larger) * (to - from);
        let end = own.readings.last().map_or(t + span, |x| x.t);
        // A part's change is read as Q's (2 Δn); the budget is the share of the total T's target its account may take.
        let (mut counted, mut work, mut state) = ((0.0, [0.0; 3]), 0.0, self.from_ground(landed));
        let (mut from, mut length, mut change) = (t, span, f64::INFINITY);
        loop {
            let to = (from + length).min(end);
            let (part, impulse, spent, last) = self.landing_part(walk, t, from, to, state, decay, knobs, own)?;
            (state, work) = (last, work + spent);
            let more = part;
            counted = (counted.0 + more, std::array::from_fn(|k| counted.1[k] + impulse[k]));
            // Doubled while the last doubling moved the Q term by more than its budget (`budget`: a share of the total T's
            // relative target) and the own run reads further -- not at all where the whole rest of the run cannot move it so
            // much (`most`, a strict bound, booked so: the reference's, whose jumps carry next to nothing of the absorbed
            // light) --; else the last doubling's change is booked (`Landing::photons_error`).
            let cannot = 2.0 * most(to, end) * weight <= budget;
            let stands = (from > t && 2.0 * more.abs() * weight <= budget) || cannot;
            change = if cannot { most(to, end) } else if from > t { more.abs() } else { change };
            if stands || !(to < end) {
                break;
            }
            (from, length) = (to, to - t);
        }
        Ok((counted.0, counted.1, work, if change.is_finite() { change } else { 0.0 }))
    }

    /// A part [from, to] of a landing's stretch from t (`landing`): from the state `state` there, its excess photons and
    /// impulse over the own run, its counted work and its last state.
    #[allow(clippy::too_many_arguments)]
    fn landing_part(&self, walk: &Walk, t: f64, from: f64, to: f64, state: Vec<C64>, decay: f64, knobs: (f64, f64, f64), own: &Series) -> Result<(f64, V3, f64, Vec<C64>), String> {
        let hm = self.hbar_over_mass;
        let span = to - from;
        let step = walk.ends.windows(2).find(|w| w[1][0] > from).map_or(span, |w| w[1][0] - w[0][0]).min(span);
        let mut times = vec![from];
        // From the landing the transient's knots, a quarter of the fastest decay's time growing by 2^{1/4}; a later part
        // at the walk's own spacing (the transient long past).
        let mut d = if from == t { 0.25 / decay } else { step };
        while *times.last().unwrap() < to {
            let next = (times.last().unwrap() + d.min(step)).min(to);
            times.push(next);
            d *= 2f64.powf(0.25);
        }
        let ends = times.iter().map(|&u| {
            let (r, k) = walk.at(hm, u);
            [u, r[0], r[1], r[2], k[0], k[1], k[2]]
        });
        let dense = Walk { index: walk.index, start: walk.start.clone(), ends: ends.collect(), lost: false, weight: walk.weight };
        let (turns, rate, cutoff) = knobs;
        // One piece a knot interval: the transient's own knots (none a µs, `pieces_in`).
        let series = self.along_from(&dense, state, from, to, Scheme::Exponential { turns, rate, pieces: 0.0, cutoff, read: 1 })?;
        // The own run's readings at a time, linearly between its knots.
        let own_at = |u: f64| -> (f64, V3) {
            let r = &own.readings;
            let i = r.partition_point(|x| x.t <= u).clamp(1, r.len().max(2) - 1);
            let (a, b) = (&r[(i - 1).min(r.len() - 1)], &r[i.min(r.len() - 1)]);
            let s = if b.t > a.t { ((u - a.t) / (b.t - a.t)).clamp(0.0, 1.0) } else { 0.0 };
            let (la, lb) = (a.rates.iter().sum::<f64>(), b.rates.iter().sum::<f64>());
            (la + s * (lb - la), std::array::from_fn(|k| a.force[k] + s * (b.force[k] - a.force[k])))
        };
        let excess = |x: &Reading| -> (f64, V3) {
            let (l, f) = own_at(x.t);
            (x.rates.iter().sum::<f64>() - l, std::array::from_fn(|k| x.force[k] - f[k]))
        };
        let (mut photons, mut impulse) = (0.0, [0.0; 3]);
        for w in series.readings.windows(2) {
            let ((p0, f0), (p1, f1)) = (excess(&w[0]), excess(&w[1]));
            let dt = w[1].t - w[0].t;
            photons += 0.5 * dt * (p0 + p1);
            for k in 0..3 {
                impulse[k] += 0.5 * dt * (f0[k] + f1[k]);
            }
        }
        Ok((photons, impulse, series.counts.work, series.state))
    }

    /// The sector's Liouvillian at the point and time, dense (d × d), column by column through the operator.
    fn liouvillian(&self, r: V3, kappa: V3, t: f64) -> Mat {
        let node = self.model.node(r, kappa, false);
        let amps = self.amps(&node);
        let mut op = Op::default();
        self.op_into(&[(&amps, 1.0, t)], 1.0, &mut op);
        let d = self.dim;
        let mut work = Work::new(self);
        let mut l = Mat::zeros(d, d);
        let mut e = vec![C64::ZERO; d];
        let mut out = vec![C64::ZERO; d];
        for j in 0..d {
            e.fill(C64::ZERO);
            e[j] = C64::ONE;
            op.apply(self, &e, &mut out, &mut work);
            for i in 0..d {
                l[(i, j)] = out[i];
            }
            // The shift's scalar back: L = L′ + 2 Im(c)·1.
            l[(j, j)] += C64::real(op.eta.ln());
        }
        l
    }
}

/// The exact vector of a beat difference Ω_a − Ω_b (model/selected.rs `Exact`): the fast integers' and the slow
/// ratios' differences, the ratios to the longer's length and its trailing zeros cut (one difference, one vector).
fn difference(a: &Exact, b: &Exact) -> Exact {
    let n = std::array::from_fn(|k| a.0[k] - b.0[k]);
    let at = |p: &Vec<Ratio>, k: usize| p.get(k).copied().unwrap_or(Ratio::ZERO);
    let mut p: Vec<Ratio> = (0..a.1.len().max(b.1.len()))
        .map(|k| {
            let (x, y) = (at(&a.1, k), at(&b.1, k));
            Ratio::new(x.num * y.den - y.num * x.den, x.den * y.den)
        })
        .collect();
    while p.last().is_some_and(|r| r.num == 0) {
        p.pop();
    }
    (n, p)
}

/// The scratch of the operator's products: the gathered columns and R's product.
struct Work {
    gathered: Vec<C64>,
    product: Vec<C64>,
}

impl Work {
    fn new(form: &Form) -> Work {
        Work { gathered: vec![C64::ZERO; form.cols.len()], product: vec![C64::ZERO; form.rows.len()] }
    }
}

/// The operator h·L′ on the sector (`Form::op_into`): per block −ih(H − c) and ih(H − c)† (the right factor), h R^T,
/// the shift's factor e^{2 Im(c) h} and the 1-norm's bound; and its making's scratch: the nodes' merged weights per H
/// slot and R group, H's combination, the norm's per-block sums.
#[derive(Default)]
struct Op {
    a1: Vec<Vec<C64>>,
    a2: Vec<Vec<C64>>,
    rt: Vec<C64>,
    eta: f64,
    norm: f64,
    wh: Vec<C64>,
    wr: Vec<C64>,
    hc: Vec<C64>,
    diag: Vec<Vec<C64>>,
    off: Vec<Vec<f64>>,
    colsum: Vec<f64>,
}

impl Op {
    /// Its buffers for the form's blocks, made where they are not yet.
    fn fit(&mut self, form: &Form) {
        if self.a1.len() == form.blocks.len() {
            return;
        }
        self.a1 = form.blocks.iter().map(|b| vec![C64::ZERO; b.len() * b.len()]).collect();
        self.a2 = self.a1.clone();
        self.diag = form.blocks.iter().map(|b| vec![C64::ZERO; b.len()]).collect();
        self.off = form.blocks.iter().map(|b| vec![0.0; b.len()]).collect();
        self.rt = vec![C64::ZERO; form.cols.len() * form.rows.len()];
        self.hc = vec![C64::ZERO; form.h_slot];
        self.colsum = vec![0.0; form.dim];
    }

    /// out = h L′ z: per pair (X, Y) −ih(H_X − c) Z + Z ih(H_Y − c)†, and R's recycling into the ground pairs.
    fn apply(&self, form: &Form, z: &[C64], out: &mut [C64], work: &mut Work) {
        for (k, &(x, y)) in form.pairs.iter().enumerate() {
            let (wx, wy) = (form.blocks[x].len(), form.blocks[y].len());
            let (o, e) = (form.offsets[k], form.offsets[k + 1]);
            fused_product(&self.a1[x], &z[o..e], (wx, wx, wy), &mut out[o..e]);
            fused_product_onto(&z[o..e], &self.a2[y], (wx, wy, wy), &mut out[o..e]);
        }
        let nr = form.rows.len();
        if nr > 0 {
            for (g, &c) in work.gathered.iter_mut().zip(&form.cols) {
                *g = z[c];
            }
            fused_product(&work.gathered, &self.rt, (1, form.cols.len(), nr), &mut work.product);
            for (p, &r) in work.product.iter().zip(&form.rows) {
                out[r] += *p;
            }
        }
    }

}

/// The memory a path's chunk of frozen pieces may hold (bytes; `Form::along_from`): 64 MB, a few chunks' at once on
/// the pool as the paths run side by side.
const FROZEN_BYTES: usize = 64 << 20;

/// The Lagrange interpolant of a knot interval cut in m parts, its nodes at the parts' ends 0 … m (module documentation,
/// # The nodes; `Scheme::Exponential`): at u (in parts, 0 … m) the first of the nearest min(4, m + 1) nodes of the
/// interval, their count and weights -- cubic where the interval holds four nodes or more, of degree m below it (linear
/// at one part).
fn lagrange(m: usize, u: f64) -> (usize, usize, [f64; 4]) {
    let count = (m + 1).min(4);
    let start = if count < 4 { 0 } else { (u.floor().max(0.0) as usize).min(m - 1).saturating_sub(1).min(m - 3) };
    let mut w = [0.0; 4];
    for (k, wk) in w.iter_mut().enumerate().take(count) {
        let xk = (start + k) as f64;
        *wk = (0..count).filter(|&l| l != k).map(|l| (u - (start + l) as f64) / (xk - (start + l) as f64)).product();
    }
    (start, count, w)
}

/// A piece frozen (`Form::freeze`; module documentation, # The substeps): its interval's nodes (`grid`, the interval's
/// start and parts' length -- the amplitudes at any time of it by `lagrange`), its start and substeps (length, count),
/// H(t_m) and per block its eigenvectors V and V⁻¹ with their adjoints, per sector entry the substep's weights
/// (`etd_weights`), the reading's node at its end where it is read there; the largest eigenvector condition, and the
/// seconds it took. It reads no state.
struct Frozen {
    grid: Vec<std::sync::Arc<Amps>>,
    a: f64,
    part: f64,
    p0: f64,
    d: f64,
    n: usize,
    hm: Vec<C64>,
    v: Vec<Vec<C64>>,
    vinv: Vec<Vec<C64>>,
    vadj: Vec<Vec<C64>>,
    vinvadj: Vec<Vec<C64>>,
    weights: Vec<Vec<C64>>,
    condition: f64,
    seconds: f64,
    /// Its counted work (real multiply-adds): the blocks' decompositions and the weights (`Counts::work`).
    work: f64,
}

impl Frozen {
    /// The interval's nodes and their weights at the time t (`lagrange`).
    fn of(&self, t: f64) -> Vec<(&Amps, f64)> {
        let (start, count, w) = lagrange(self.grid.len() - 1, (t - self.a) / self.part);
        (0..count).map(|k| (&*self.grid[start + k], w[k])).collect()
    }
}

/// φ₀ … φ₃ of z (φ₀ = e^z, φ_{k+1}(z) = (φ_k(z) − 1/k!)/z): the Taylor series where |z| < 1 -- 24 terms, the 24th
/// below 1/24! of the first --, the recurrence beyond, its cancellation no worse than a factor 1/|z| ≤ 1 a step.
fn phis(z: C64) -> [C64; 4] {
    if z.abs() < 1.0 {
        let mut out = [C64::ZERO; 4];
        let mut power = C64::ONE;
        let mut factorial = [1.0_f64, 1.0, 2.0, 6.0];
        for j in 0..24 {
            for (o, f) in out.iter_mut().zip(&factorial) {
                *o += power * (1.0 / f);
            }
            power = power * z;
            for (k, f) in factorial.iter_mut().enumerate() {
                *f *= (j + k + 1) as f64;
            }
        }
        out
    } else {
        let e = z.exp();
        let p1 = (e - C64::ONE) / z;
        let p2 = (p1 - C64::ONE) / z;
        let p3 = (p2 - C64::real(0.5)) / z;
        [e, p1, p2, p3]
    }
}

/// The count of `etd_weights`.
const ETD_WEIGHTS: usize = 10;

/// Krogstad's exponential Runge–Kutta weights (module documentation, # The substeps) of an entry of rate μ at the
/// substep d, z = μd: e^z and e^{z/2}; the stages' a₂₁ = ½φ₁(z/2), a₃₁ = ½φ₁(z/2) − φ₂(z/2), a₃₂ = φ₂(z/2), a₄₁ = φ₁(z) −
/// 2φ₂(z), a₄₃ = 2φ₂(z); the step's b₁ = φ₁ − 3φ₂ + 4φ₃, b₂ = b₃ = 2φ₂ − 4φ₃, b₄ = 4φ₃ − φ₂ (at z), the stages at c = 0,
/// ½, ½, 1.
fn etd_weights(mu: C64, d: f64) -> [C64; ETD_WEIGHTS] {
    let [e, p1, p2, p3] = phis(mu * d);
    let [e2, q1, q2, _] = phis(mu * (0.5 * d));
    [e, e2, q1 * 0.5, q1 * 0.5 - q2, q2, p1 - p2 * 2.0, p2 * 2.0, p1 - p2 * 3.0 + p3 * 4.0, p2 * 2.0 - p3 * 4.0, p3 * 4.0 - p2]
}

/// A path's substeps' scratch (`Scheme::Exponential`; `Form::piece_run`): H(t), its change −iΔH(t) and the right factor
/// iΔH(t)† per block, R^T at a substep's stages (one where R holds no beat), the stages' vectors.
struct Stages {
    now: Vec<C64>,
    dh: Vec<Vec<C64>>,
    dha: Vec<Vec<C64>>,
    r: [Vec<C64>; 3],
    rho: Vec<C64>,
    tmp: Vec<C64>,
    bare: Vec<C64>,
    k: [Vec<C64>; 4],
    s: [Vec<C64>; 2],
    gathered: Vec<C64>,
    product: Vec<C64>,
}

impl Stages {
    fn new(form: &Form) -> Stages {
        let d = form.dim;
        let sq = |b: &Vec<usize>| vec![C64::ZERO; b.len() * b.len()];
        Stages {
            now: vec![C64::ZERO; form.h_slot],
            dh: form.blocks.iter().map(sq).collect(),
            dha: form.blocks.iter().map(sq).collect(),
            r: std::array::from_fn(|_| vec![C64::ZERO; form.cols.len() * form.rows.len()]),
            rho: vec![C64::ZERO; d],
            tmp: vec![C64::ZERO; d],
            bare: vec![C64::ZERO; d],
            k: std::array::from_fn(|_| vec![C64::ZERO; d]),
            s: std::array::from_fn(|_| vec![C64::ZERO; d]),
            gathered: vec![C64::ZERO; form.cols.len()],
            product: vec![C64::ZERO; form.rows.len()],
        }
    }
}

impl Form<'_> {
    /// x_XY ← A_X x_XY B_Y for every sector pair (A, B per block, b × b row by row), through `tmp`.
    fn sandwich(&self, a: &[Vec<C64>], b: &[Vec<C64>], x: &[C64], out: &mut [C64], tmp: &mut [C64]) {
        for (k, &(px, py)) in self.pairs.iter().enumerate() {
            let (wx, wy) = (self.blocks[px].len(), self.blocks[py].len());
            let (o, e) = (self.offsets[k], self.offsets[k + 1]);
            fused_product(&a[px], &x[o..e], (wx, wx, wy), &mut tmp[o..e]);
            fused_product(&tmp[o..e], &b[py], (wx, wy, wy), &mut out[o..e]);
        }
    }

    /// H(t) on the blocks of the amplitudes Σ_j w_j a_j at their phases at t, into `out` (Σ b² entries).
    fn h_at(&self, of: &[(&Amps, f64)], t: f64, out: &mut [C64]) {
        out.fill(C64::ZERO);
        for (k, &beat) in self.h_beats.iter().enumerate() {
            let f = self.phase(beat, t);
            for &(a, w) in of {
                mul_add(false, out, f * w, &a.h[k * self.h_slot..(k + 1) * self.h_slot]);
            }
        }
    }

    /// R(t)^T of the amplitudes Σ_j w_j a_j, its groups at their phases at t, into `out`.
    fn r_at(&self, of: &[(&Amps, f64)], t: f64, out: &mut [C64]) {
        out.fill(C64::ZERO);
        for (g, &delta) in self.beats_r.iter().enumerate() {
            let f = if delta == 0.0 { C64::ONE } else { C64::cis(-delta * t) };
            for &(a, w) in of {
                self.r_onto(out, f * w, &a.r, g);
            }
        }
    }

    /// The remainder g = (L(t) − L₀)σ on an eigenbasis vector σ, into `out` (eigenbasis), of the frozen piece `f`: ρ =
    /// VσV†, then −i(ΔHρ − ρΔH†) + Rρ -- ΔH = H(t) − H(t_m) in `lw.dh` (−iΔH) and `lw.dha` (iΔH†, `change_at`), R^T
    /// `rt` --, back by V⁻¹ · V⁻†.
    fn remainder(&self, f: &Frozen, rt: &[C64], lw: &mut Stages, sigma: &[C64], out: &mut [C64]) {
        let (mut rho, mut tmp, mut bare) = (std::mem::take(&mut lw.rho), std::mem::take(&mut lw.tmp), std::mem::take(&mut lw.bare));
        self.sandwich(&f.v, &f.vadj, sigma, &mut rho, &mut tmp);
        for (q, &(px, py)) in self.pairs.iter().enumerate() {
            let (wx, wy) = (self.blocks[px].len(), self.blocks[py].len());
            let (o, e) = (self.offsets[q], self.offsets[q + 1]);
            fused_product(&lw.dh[px], &rho[o..e], (wx, wx, wy), &mut bare[o..e]);
            fused_product_onto(&rho[o..e], &lw.dha[py], (wx, wy, wy), &mut bare[o..e]);
        }
        let nr = self.rows.len();
        if nr > 0 {
            for (g, &c) in lw.gathered.iter_mut().zip(&self.cols) {
                *g = rho[c];
            }
            fused_product(&lw.gathered, rt, (1, self.cols.len(), nr), &mut lw.product);
            for (p, &r) in lw.product.iter().zip(&self.rows) {
                bare[r] += *p;
            }
        }
        self.sandwich(&f.vinv, &f.vinvadj, &bare, out, &mut tmp);
        (lw.rho, lw.tmp, lw.bare) = (rho, tmp, bare);
    }

    /// H's change from the piece's middle at the time t, ΔH = H(t) − H(t_m) of the interpolated amplitudes at their
    /// phases there, per block into `lw.dh` (−iΔH) and `lw.dha` (iΔH†, the right factor).
    fn change_at(&self, f: &Frozen, t: f64, lw: &mut Stages) {
        let mut now = std::mem::take(&mut lw.now);
        self.h_at(&f.of(t), t, &mut now);
        for (x, b) in self.blocks.iter().enumerate() {
            let (w, o) = (b.len(), self.h_block[x]);
            for i in 0..w {
                for j in 0..w {
                    let z = now[o + i * w + j] - f.hm[o + i * w + j];
                    lw.dh[x][i * w + j] = C64::new(0.0, -1.0) * z;
                    lw.dha[x][j * w + i] = C64::new(0.0, 1.0) * z.conj();
                }
            }
        }
        lw.now = now;
    }

    /// Per sector entry its rate μ = −i(λ_a − conj λ_b) of the blocks' eigenvalues `values` (# The substeps).
    fn rates_of(&self, values: &[Vec<C64>]) -> Vec<C64> {
        let mut mus = vec![C64::ZERO; self.dim];
        for (q, &(px, py)) in self.pairs.iter().enumerate() {
            let wy = self.blocks[py].len();
            for (i, la) in values[px].iter().enumerate() {
                for (k, lb) in values[py].iter().enumerate() {
                    mus[self.offsets[q] + i * wy + k] = C64::new(0.0, -1.0) * (*la - lb.conj());
                }
            }
        }
        mus
    }

    /// A stretch's opening (module documentation, # The start) from the state `v` at the time `t`, the amplitudes there
    /// `amps` and at the path's first knot `t1` after it `next`, the first interval's `pieces`, the cutoff `cutoff` (rad/µs): its
    /// length (µs), the time the start's optical coherences take to decay to the double's unit rounding u, 2 ln(1/u)/Γ,
    /// Γ the slowest decay −2 Im H_ee of P's excited states in the sector (none, none: a model whose excited states are
    /// all folded holds no such coherence); of the modes the start holds, |σ₀ⱼ| above u of the largest, that the opening
    /// outlasts, |σ₀ⱼ| e^{Re μ_j t_open} at most that (a mode it does not outlast goes on into the run, which takes it as
    /// it takes every mode then), the fastest |μ_j| -- its turning and its decay together -- up to the cutoff (resolved), the sum of the others' bounds
    /// (`booked`; # The start), and the start with those others' parts dropped. Err where an eigendecomposition fails.
    #[allow(clippy::too_many_arguments)]
    fn opening(&self, amps: &Amps, next: &Amps, t: f64, t1: f64, pieces: usize, v: &[C64], cutoff: f64) -> Result<(f64, f64, f64, Vec<C64>), String> {
        let mut h = vec![C64::ZERO; self.h_slot];
        self.h_at(&[(amps, 1.0)], t, &mut h);
        let mut decay = f64::INFINITY;
        let (mut values, mut vecs, mut vinv, mut vinvadj) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for (x, blk) in self.blocks.iter().enumerate() {
            let (w, o) = (blk.len(), self.h_block[x]);
            for (i, &p) in blk.iter().enumerate() {
                if self.states[p] >= GROUND {
                    decay = decay.min(-2.0 * h[o + i * w + i].im);
                }
            }
            let mut eig = Eigen::default();
            if !eig.decompose(&Mat { rows: w, cols: w, data: h[o..o + w * w].to_vec() }) {
                return Err(format!("{t} µs 处 H 的第 {x} 块做不出本征分解"));
            }
            let (mut a, mut b) = (vec![C64::ZERO; w * w], vec![C64::ZERO; w * w]);
            for i in 0..w {
                for k in 0..w {
                    a[i * w + k] = eig.inverse[(i, k)];
                    b[k * w + i] = eig.inverse[(i, k)].conj();
                }
            }
            values.push(eig.values.clone());
            vecs.push(eig.vectors.clone());
            vinv.push(a);
            vinvadj.push(b);
        }
        if !(decay.is_finite() && decay > 0.0) {
            return Ok((0.0, 0.0, 0.0, v.to_vec()));
        }
        let length = 2.0 * (2.0 / f64::EPSILON).ln() / decay;
        let mus = self.rates_of(&values);
        let (mut sigma, mut tmp) = (vec![C64::ZERO; self.dim], vec![C64::ZERO; self.dim]);
        self.sandwich(&vinv, &vinvadj, v, &mut sigma, &mut tmp);
        let floor = 0.5 * f64::EPSILON * sigma.iter().map(|z| z.abs()).fold(0.0, larger);
        let held: Vec<usize> = (0..self.dim).filter(|&j| sigma[j].abs() > floor && sigma[j].abs() * (mus[j].re * length).exp() <= floor).collect();
        // Each mode by its whole rate |μ_j|, the turning and the decay together (# The start): the remainder follows a
        // decaying mode as fast as a turning one.
        let fastest = held.iter().filter(|&&j| mus[j].abs() <= cutoff).map(|&j| mus[j].abs()).fold(0.0, larger);
        let faster: Vec<usize> = held.iter().copied().filter(|&j| mus[j].abs() > cutoff).collect();
        if faster.is_empty() {
            return Ok((length, fastest, 0.0, v.to_vec()));
        }
        // ‖ΔH‖: H's change over half a piece, and every beat's amplitude twice; ‖Ḣ‖: H's change over the first knot
        // interval, and every beat's amplitude twice times its rate.
        let mut later = vec![C64::ZERO; self.h_slot];
        self.h_at(&[(next, 1.0)], t, &mut later);
        let norm_of = |m: &dyn Fn(usize) -> C64| -> f64 {
            let mut most: f64 = 0.0;
            for (x, blk) in self.blocks.iter().enumerate() {
                let (w, o) = (blk.len(), self.h_block[x]);
                for i in 0..w {
                    let (row, col) = ((0..w).map(|k| m(o + i * w + k).abs()).sum::<f64>(), (0..w).map(|k| m(o + k * w + i).abs()).sum::<f64>());
                    most = larger(most, larger(row, col));
                }
            }
            most
        };
        let change = norm_of(&|k| later[k] - h[k]);
        let beats: Vec<(f64, f64)> = self.h_beats.iter().enumerate().filter(|&(_, &b)| self.values[b] != 0.0).map(|(k, &b)| (2.0 * norm_of(&|e| amps.h[k * self.h_slot + e]), self.values[b].abs())).collect();
        let dh = change / (2.0 * pieces as f64) + beats.iter().map(|b| b.0).sum::<f64>();
        let rate = change / (t1 - t) + beats.iter().map(|b| b.0 * b.1).sum::<f64>();
        // The generator at the start, to take L x of the recycled part (its shift's scalar back: L = L′ + 2 Im(c)).
        let mut op = Op::default();
        let nr = self.rows.len();
        let whole = Amps { columns: self.columns_of(&amps.r), ..amps.clone() };
        self.op_into(&[(&whole, 1.0, t)], 1.0, &mut op);
        let mut work = Work::new(self);
        let (bound_l, shift) = (op.norm, op.eta.ln());
        // R^T at the start, and each faster mode's place: its pair and its two eigenstates.
        let mut rt = vec![C64::ZERO; self.cols.len() * self.rows.len()];
        self.r_at(&[(amps, 1.0)], t, &mut rt);
        let mut gathered = vec![C64::ZERO; self.cols.len()];
        let mut product = vec![C64::ZERO; nr];
        let (mut x, mut lx) = (vec![C64::ZERO; self.dim], vec![C64::ZERO; self.dim]);
        let first_reading = t1 - t;
        let mut booked = 0.0;
        let mut kept = sigma.clone();
        for &j in &faster {
            let q = self.offsets.partition_point(|&o| o <= j) - 1;
            let (px, py) = self.pairs[q];
            let wy = self.blocks[py].len();
            let (a, b) = ((j - self.offsets[q]) / wy, (j - self.offsets[q]) % wy);
            let (va, vb) = (&vecs[px], &vecs[py]);
            let wx = self.blocks[px].len();
            // m_j on the sector: v_a v_b† on the pair (X, Y), nothing elsewhere.
            for (g, &c) in gathered.iter_mut().zip(&self.cols) {
                *g = if c >= self.offsets[q] && c < self.offsets[q + 1] {
                    let (i, k) = ((c - self.offsets[q]) / wy, (c - self.offsets[q]) % wy);
                    va[(i, a)] * vb[(k, b)].conj()
                } else {
                    C64::ZERO
                };
            }
            fused_product(&gathered, &rt, (1, self.cols.len(), nr), &mut product);
            // x = R m_j on the sector, and L x.
            x.fill(C64::ZERO);
            for (p, &r) in product.iter().zip(&self.rows) {
                x[r] += *p;
            }
            op.apply(self, &x, &mut lx, &mut work);
            let recycled: f64 = x.iter().map(|z| z.abs()).sum();
            let moved: f64 = lx.iter().zip(&x).map(|(l, z)| (*l + *z * shift).abs()).sum();
            let size = (0..wx).map(|i| va[(i, a)].norm_sqr()).sum::<f64>().sqrt() * (0..wy).map(|k| vb[(k, b)].norm_sqr()).sum::<f64>().sqrt();
            // c(u) = Φ(t, u) N(u) m_j: |c| ≤ ‖N m_j‖₁, |ċ| ≤ ‖L N m_j‖₁ + ‖Ṅ m_j‖₁ (# The start).
            let ends = 2.0 * (recycled + 2.0 * dh * size);
            let variation = (moved + bound_l * 2.0 * dh * size + 2.0 * rate * size) / -mus[j].re;
            booked += sigma[j].abs() * (size * (mus[j].re * first_reading).exp() + crate::model::classify::by_parts(ends, variation, mus[j]));
            kept[j] = C64::ZERO;
        }
        // The start with the booked modes' parts dropped: V σ V†.
        let (vs, vsadj): (Vec<Vec<C64>>, Vec<Vec<C64>>) = vecs
            .iter()
            .map(|m| {
                let w = m.rows;
                let (mut a, mut b) = (vec![C64::ZERO; w * w], vec![C64::ZERO; w * w]);
                for i in 0..w {
                    for k in 0..w {
                        a[i * w + k] = m[(i, k)];
                        b[k * w + i] = m[(i, k)].conj();
                    }
                }
                (a, b)
            })
            .unzip();
        let mut start = vec![C64::ZERO; self.dim];
        self.sandwich(&vs, &vsadj, &kept, &mut start, &mut tmp);
        Ok((length, fastest, booked, start))
    }

    /// Exponential runs along one stretch from the sector state `state`, a family (module documentation, # The nodes):
    /// the schemes `members`, read alike (on each knot interval their pieces powers of two of one another, `pieces_in`).
    /// Each node made once on the interval's finest member's pieces, each piece frozen once for the members of the same
    /// pieces and substeps there
    /// (`freeze_family`), each state propagated once for a class of members -- at the start those of one opening, split
    /// where their pieces part (`advance`) --; per member its series, the same as its run alone to the bit, the shared
    /// work counted on the first member that takes it (the members' work summed is the family's).
    fn family(&self, walk: &Walk, state: Vec<C64>, from: f64, to: f64, members: &[Scheme], keep: &[f64]) -> Result<Vec<Series>, String> {
        // The family's own terms, every caller's (`along_from`, `along_family_keeping`, the ladder's families): a fault of
        // the program where one fails, asserted -- no refusal a user meets.
        let mut knobs: Vec<(f64, f64, f64, f64)> = Vec::with_capacity(members.len());
        let mut read = 0;
        for s in members {
            let Scheme::Exponential { turns, rate, pieces, cutoff, read: r } = *s;
            assert!(read == 0 || r == read, "a family read alike");
            read = r;
            knobs.push((turns, rate, pieces, cutoff));
        }
        let m = knobs.len();
        assert!(m > 0, "an empty family");
        assert!(knobs.iter().all(|k| k.2 >= 0.0 && k.2.is_finite()), "a family's pieces a µs a number, none below nought: {knobs:?}");
        assert_eq!(state.len(), self.dim, "a state of the form's sector");
        assert!(to > from, "an empty stretch of the path: [{from}, {to}]");
        let mut counts = vec![Counts::default(); m];
        let knots = walk.knots(from, to);
        // The knot intervals, each with whether a reading is made at its end.
        let intervals: Vec<(f64, f64, bool)> = knots.windows(2).enumerate().map(|(i, w)| (w[0], w[1], (i + 1) % read == 0 || i + 2 == knots.len())).collect();
        // The start: its reading (one state for all), its amplitudes, and the first knot's (H's change along a piece).
        let clock = std::time::Instant::now();
        let node = self.node_at(walk, from);
        let at_knot = self.node_at(walk, knots[1]);
        (counts[0].nodes, counts[0].node_seconds, counts[0].work) = (2, clock.elapsed().as_secs_f64(), node.work + at_knot.work + 2.0 * self.amplitudes_work());
        let mut c = Counts::default();
        let reading = self.read(&node, walk, from, &state, &mut c);
        counts[0].add(&c);
        let first = std::sync::Arc::new(self.amps_with(&node, false));
        drop(node);
        let next = self.amps_with(&at_knot, false);
        drop(at_knot);
        // Each member's opening (# The start), one for the members of the same pieces and cutoff: its length, the fastest
        // coherence it resolves, its bound, and the start with the booked modes dropped (the reading at the start is made).
        let mut openings: Vec<((f64, f64), (f64, f64, f64, Vec<C64>))> = Vec::new();
        let mut of: Vec<usize> = Vec::with_capacity(m);
        for (q, k) in knobs.iter().enumerate() {
            let at = match openings.iter().position(|o| o.0 == (k.2, k.3)) {
                Some(at) => at,
                None => {
                    openings.push(((k.2, k.3), self.opening(&first, &next, from, knots[1], pieces_in(from, knots[1], k.2), &state, k.3)?));
                    openings.len() - 1
                }
            };
            (counts[q].opening, counts[q].booked) = (openings[at].1 .0, openings[at].1 .2);
            of.push(at);
        }
        // A member's substeps over a piece, of its start and length (# The start): `turns` a turn of the fastest coherence
        // its opening resolves, in the opening, `rate` a µs after; a product's rounding up never adds one (a rate of whole
        // substeps a piece).
        let substeps = |q: usize, p0: f64, part: f64| -> usize {
            let (turns, rate, ..) = knobs[q];
            let (opening, fastest) = (openings[of[q]].1 .0, openings[of[q]].1 .1);
            let n = if p0 < from + opening { part * fastest * turns / std::f64::consts::TAU } else { part * rate };
            (n * (1.0 - 4.0 * f64::EPSILON)).ceil().max(1.0) as usize
        };
        // The classes: the members of one state, each with it -- at the start those of one opening.
        let mut classes: Vec<(Vec<usize>, Vec<C64>)> = (0..openings.len()).map(|o| ((0..m).filter(|&q| of[q] == o).collect(), openings[o].1 .3.clone())).collect();
        let mut readings: Vec<Vec<Reading>> = vec![vec![reading]; m];
        let mut kept: Vec<Vec<(f64, Vec<C64>)>> = vec![vec![]; m];
        let mut condition = vec![0.0_f64; m];
        // A chunk of intervals frozen on the pool while the one before is propagated; its readings on the pool after it.
        // The chunk: what `FROZEN_BYTES` holds of an interval's nodes and the members' pieces, at most four per thread.
        let most = |pieces: f64| intervals.iter().map(|iv| pieces_in(iv.0, iv.1, pieces)).max().unwrap_or(1);
        let finest = knobs.iter().map(|k| most(k.2)).max().unwrap_or(1);
        let pieces: usize = knobs.iter().map(|k| most(k.2)).sum();
        let bytes = 16 * (finest * (self.h_beats.len() * self.h_slot + self.r_places.len()) + pieces * (5 * self.h_slot + ETD_WEIGHTS * self.dim));
        let chunk = (FROZEN_BYTES / (bytes + 1)).clamp(1, 4 * rayon::current_num_threads());
        let freeze = |range: std::ops::Range<usize>, before: std::sync::Arc<Amps>| self.freeze_family(walk, &intervals[range], before, &knobs, &substeps);
        let freeze = &freeze;
        let mut ahead = Some(freeze(0..chunk.min(intervals.len()), first));
        let mut start = 0;
        while let Some(frozen) = ahead.take() {
            let (taken, reads, last, made, ends) = frozen?;
            counts[0].add(&made);
            for q in 0..m {
                condition[q] = taken[q].iter().map(|f| f.condition).fold(condition[q], larger);
            }
            let end = (start + chunk).min(intervals.len());
            let following = (end < intervals.len()).then(|| end..(end + chunk).min(intervals.len()));
            let read_at: Vec<bool> = reads.iter().map(Option::is_some).collect();
            let now = std::mem::take(&mut classes);
            let (next, moved) = rayon::join(move || following.map(|range| freeze(range, last)), || now.into_par_iter().map(|class| self.advance(class, &taken, &ends, &read_at)).collect::<Vec<_>>());
            // The states read at the intervals' ends, in each member's order of time.
            let mut states = Vec::new();
            for (kept, read, work) in moved {
                classes.extend(kept);
                states.extend(read);
                for (q, c) in work {
                    counts[q].add(&c);
                }
            }
            states.sort_by_key(|x| x.0);
            let read: Vec<(Reading, Counts)> = states
                .par_iter()
                .map(|(i, _, state)| {
                    let (t, node) = reads[*i].as_ref().expect("a reading's node");
                    let mut c = Counts::default();
                    let r = self.read(node, walk, *t, state, &mut c);
                    (r, c)
                })
                .collect();
            for ((r, c), (_, members, state)) in read.into_iter().zip(&states) {
                counts[members[0]].add(&Counts { time: 0.0, ..c });
                let keeping = keep.contains(&r.t);
                for &q in members {
                    if keeping {
                        kept[q].push((r.t, state.clone()));
                    }
                    readings[q].push(r.clone());
                }
            }
            (ahead, start) = (next, end);
        }
        let mut out: Vec<Option<Series>> = (0..m).map(|_| None).collect();
        for (members, state) in classes {
            for &q in &members {
                let mut c = counts[q];
                (c.time, c.condition) = (to - from, condition[q]);
                out[q] = Some(Series { classes: self.classes.clone(), readings: std::mem::take(&mut readings[q]), counts: c, state: state.clone(), kept: std::mem::take(&mut kept[q]) });
            }
        }
        Ok(out.into_iter().map(|s| s.expect("every member in a class")).collect())
    }

    /// A class of a family's members along a chunk (`family`): its state through the members' pieces `pieces` (per
    /// member, in order; `ends` per member where each interval's end among them) while they are the same pieces, split
    /// into classes where they part; the classes at the chunk's
    /// end, the states at its intervals' ends where they are read (`read`: the interval, the class's members, the state),
    /// and the work, on each class's first member.
    #[allow(clippy::type_complexity)]
    fn advance(&self, (members, state): (Vec<usize>, Vec<C64>), pieces: &[Vec<std::sync::Arc<Frozen>>], ends: &[Vec<usize>], read: &[bool]) -> (Vec<(Vec<usize>, Vec<C64>)>, Vec<(usize, Vec<usize>, Vec<C64>)>, Vec<(usize, Counts)>) {
        let mut lw = Stages::new(self);
        let (mut kept, mut states, mut work) = (Vec::new(), Vec::new(), Vec::new());
        let mut todo = vec![(members, state, 0usize)];
        while let Some((members, mut state, mut k)) = todo.pop() {
            let lead = members[0];
            let (bounds, total) = (&ends[lead], pieces[lead].len());
            let mut c = Counts::default();
            let clock = std::time::Instant::now();
            while k < total && members.iter().all(|&q| std::sync::Arc::ptr_eq(&pieces[q][k], &pieces[lead][k])) {
                self.piece_run(&pieces[lead][k], &mut state, &mut lw, &mut c);
                k += 1;
                if let Ok(i) = bounds.binary_search(&k) {
                    if read[i] {
                        states.push((i, members.clone(), state.clone()));
                    }
                }
            }
            c.propagation_seconds += clock.elapsed().as_secs_f64();
            work.push((lead, c));
            if k < total {
                let mut parted: Vec<Vec<usize>> = Vec::new();
                for &q in &members {
                    match parted.iter_mut().find(|g| std::sync::Arc::ptr_eq(&pieces[g[0]][k], &pieces[q][k])) {
                        Some(g) => g.push(q),
                        None => parted.push(vec![q]),
                    }
                }
                for g in parted {
                    todo.push((g, state.clone(), k));
                }
            } else {
                kept.push((members, state));
            }
        }
        (kept, states, work)
    }

    /// A chunk of knot intervals frozen for a family (module documentation, # The nodes; `family`): each interval's
    /// nodes at its finest member's pieces' ends (`pieces_in` of each member's pieces a µs), one job each on the pool --
    /// the node before the chunk's first `before` --; then the pieces (`freeze`), each member's on its own nodes among
    /// them (every (finest / its pieces)-th), one job a piece of the same pieces, place and substeps (`substeps`: of the
    /// member, the piece's start and length), shared by the members that take it. Per member its pieces in order, per
    /// interval its end's node where it is read there, the chunk's last node, the work, and per member where each
    /// interval ends among its pieces.
    #[allow(clippy::type_complexity)]
    fn freeze_family(&self, walk: &Walk, intervals: &[(f64, f64, bool)], before: std::sync::Arc<Amps>, knobs: &[(f64, f64, f64, f64)], substeps: &(dyn Fn(usize, f64, f64) -> usize + Sync)) -> Result<(Vec<Vec<std::sync::Arc<Frozen>>>, Vec<Option<(f64, Node)>>, std::sync::Arc<Amps>, Counts, Vec<Vec<usize>>), String> {
        // Per member and interval its pieces; per interval the finest member's, and where its nodes start among the chunk's.
        let each: Vec<Vec<usize>> = knobs.iter().map(|k| intervals.iter().map(|iv| pieces_in(iv.0, iv.1, k.2)).collect()).collect();
        let finest: Vec<usize> = (0..intervals.len()).map(|i| each.iter().map(|e| e[i]).max().unwrap_or(1)).collect();
        let offset: Vec<usize> = finest
            .iter()
            .scan(0, |sum, &f| {
                *sum += f;
                Some(*sum - f)
            })
            .collect();
        let jobs: Vec<(usize, usize)> = (0..intervals.len()).flat_map(|i| (1..=finest[i]).map(move |j| (i, j))).collect();
        let mut nodes: Vec<(std::sync::Arc<Amps>, Option<Node>, f64, f64)> = jobs
            .par_iter()
            .map(|&(i, j)| {
                let clock = std::time::Instant::now();
                let (a, b, read) = intervals[i];
                let f = finest[i];
                let t = if j == f { b } else { a + j as f64 * (b - a) / f as f64 };
                let node = self.node_at(walk, t);
                #[cfg(test)]
                traced(0, clock);
                #[cfg(test)]
                let part = std::time::Instant::now();
                let amps = self.amps_with(&node, false);
                #[cfg(test)]
                traced(1, part);
                let work = node.work + self.amplitudes_work();
                (std::sync::Arc::new(amps), (j == f && read).then_some(node), clock.elapsed().as_secs_f64(), work)
            })
            .collect();
        let mut counts = Counts { nodes: nodes.len() as u64, node_seconds: nodes.iter().map(|x| x.2).sum(), work: nodes.iter().map(|x| x.3).sum(), ..Counts::default() };
        // The distinct pieces -- (the interval's pieces, interval, place, substeps) -- and each member's among them.
        let mut keys: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut index: std::collections::HashMap<(usize, usize, usize, usize), usize> = std::collections::HashMap::new();
        let mut places: Vec<Vec<usize>> = Vec::with_capacity(knobs.len());
        for q in 0..knobs.len() {
            let mut own = Vec::with_capacity(each[q].iter().sum());
            for (i, &(a, b, _)) in intervals.iter().enumerate() {
                let pieces = each[q][i];
                let part = (b - a) / pieces as f64;
                for j in 0..pieces {
                    let key = (pieces, i, j, substeps(q, a + j as f64 * part, part));
                    own.push(*index.entry(key).or_insert_with(|| {
                        keys.push(key);
                        keys.len() - 1
                    }));
                }
            }
            places.push(own);
        }
        // A member's grid on the interval i: the node before it, then its own nodes among the finest's.
        let frozen: Vec<std::sync::Arc<Frozen>> = {
            let grid = |i: usize, pieces: usize| -> Vec<std::sync::Arc<Amps>> {
                let s = finest[i] / pieces;
                std::iter::once(if i == 0 { before.clone() } else { nodes[offset[i] - 1].0.clone() }).chain((1..=pieces).map(|j| nodes[offset[i] + j * s - 1].0.clone())).collect()
            };
            keys.par_iter().map(|&(pieces, i, j, n)| self.freeze(grid(i, pieces), intervals[i].0, intervals[i].1, j, &|_, _| n).map(std::sync::Arc::new)).collect::<Result<_, _>>()?
        };
        for f in &frozen {
            counts.node_seconds += f.seconds;
            counts.work += f.work;
        }
        let reads = intervals.iter().enumerate().map(|(i, iv)| nodes[offset[i] + finest[i] - 1].1.take().map(|node| (iv.1, node))).collect();
        let last = nodes.last().map_or(before, |x| x.0.clone());
        let taken = places.into_iter().map(|own| own.into_iter().map(|p| frozen[p].clone()).collect()).collect();
        let ends = each
            .iter()
            .map(|e| {
                e.iter()
                    .scan(0, |sum, &n| {
                        *sum += n;
                        Some(*sum)
                    })
                    .collect()
            })
            .collect();
        Ok((taken, reads, last, counts, ends))
    }

    /// A piece frozen (`Frozen`): the `j`-th of the knot interval [a, b] cut in as many pieces as its nodes `grid` at the
    /// pieces' ends (`lagrange`) less one: H(t_m) at its middle and its eigendecomposition per block (σ = V⁻¹ρV⁻†, Λ
    /// diagonal there), its substeps (`substeps`: their count of the piece's start and length) and their weights. Err where an eigendecomposition fails.
    fn freeze(&self, grid: Vec<std::sync::Arc<Amps>>, a: f64, b: f64, j: usize, substeps: &(dyn Fn(f64, f64) -> usize + Sync)) -> Result<Frozen, String> {
        let clock = std::time::Instant::now();
        let pieces = grid.len() - 1;
        let part = (b - a) / pieces as f64;
        let p0 = a + j as f64 * part;
        let tm = p0 + 0.5 * part;
        let mut f = Frozen { grid, a, part, p0, d: 0.0, n: 0, hm: vec![C64::ZERO; self.h_slot], v: vec![], vinv: vec![], vadj: vec![], vinvadj: vec![], weights: vec![], condition: 0.0, seconds: 0.0, work: 0.0 };
        let mut hm = std::mem::take(&mut f.hm);
        self.h_at(&f.of(tm), tm, &mut hm);
        f.hm = hm;
        let mut values: Vec<Vec<C64>> = Vec::with_capacity(self.blocks.len());
        for (x, blk) in self.blocks.iter().enumerate() {
            let w = blk.len();
            let m = Mat { rows: w, cols: w, data: f.hm[self.h_block[x]..self.h_block[x] + w * w].to_vec() };
            let mut eig = Eigen::default();
            if !eig.decompose(&m) {
                return Err(format!("{tm} µs 处 H 的第 {x} 块做不出本征分解"));
            }
            f.condition = larger(f.condition, eig.condition);
            f.work += crate::decide::COMPLEX * eig.work;
            let (mut e, mut ei, mut ea, mut eia) = (vec![C64::ZERO; w * w], vec![C64::ZERO; w * w], vec![C64::ZERO; w * w], vec![C64::ZERO; w * w]);
            for i in 0..w {
                for k in 0..w {
                    let (p, q) = (eig.vectors[(i, k)], eig.inverse[(i, k)]);
                    e[i * w + k] = p;
                    ei[i * w + k] = q;
                    ea[k * w + i] = p.conj();
                    eia[k * w + i] = q.conj();
                }
            }
            f.v.push(e);
            f.vinv.push(ei);
            f.vadj.push(ea);
            f.vinvadj.push(eia);
            values.push(eig.values.clone());
        }
        #[cfg(test)]
        traced(2, clock);
        #[cfg(test)]
        let part_clock = std::time::Instant::now();
        // μ per sector entry, the substeps, and their weights.
        let mus = self.rates_of(&values);
        f.n = substeps(p0, part);
        f.d = part / f.n as f64;
        // The weights: φ₀ … φ₃ of each sector entry's rate (`phis`: at most its series' 24 terms) and the ten weights.
        f.work += crate::decide::COMPLEX * (self.dim * (24 + ETD_WEIGHTS)) as f64;
        f.weights = vec![vec![C64::ZERO; self.dim]; ETD_WEIGHTS];
        for (at, &mu) in mus.iter().enumerate() {
            for (w, x) in f.weights.iter_mut().zip(etd_weights(mu, f.d)) {
                w[at] = x;
            }
        }
        #[cfg(test)]
        traced(3, part_clock);
        f.seconds = clock.elapsed().as_secs_f64();
        Ok(f)
    }

    /// A frozen piece's substeps (module documentation, # The substeps): Krogstad's exponential Runge–Kutta on σ' = Λσ +
    /// g(t, σ) in H(t_m)'s eigenbasis, g the remainder -- H's change from t_m at the stages' times t, t + d/2, t + d,
    /// and the recycling R (its amplitudes at the substep's middle, its beats' phases at the stages') --, in its n
    /// substeps of d, on the state `v`; the amplitudes interpolated from the interval's nodes as the substeps go.
    fn piece_run(&self, f: &Frozen, v: &mut [C64], lw: &mut Stages, counts: &mut Counts) {
        let d = f.d;
        let still = !(self.beats_r.len() > 1 || self.beats_r.first().is_some_and(|x| *x != 0.0));
        // σ = V⁻¹ ρ V⁻†.
        let mut sigma = std::mem::take(&mut lw.s[0]);
        let mut tmp = std::mem::take(&mut lw.tmp);
        self.sandwich(&f.vinv, &f.vinvadj, v, &mut sigma, &mut tmp);
        lw.tmp = tmp;
        let mut k = std::mem::take(&mut lw.k);
        let mut u = std::mem::take(&mut lw.s[1]);
        let mut r = std::mem::take(&mut lw.r);
        let w = &f.weights;
        let (e, e2, a21, a31, a32, a41, a43, b1, b23, b4) = (&w[0], &w[1], &w[2], &w[3], &w[4], &w[5], &w[6], &w[7], &w[8], &w[9]);
        for step in 0..f.n {
            let t = f.p0 + step as f64 * d;
            // R^T at the substep: the amplitudes at its middle, the phases at each stage's time.
            let middle = f.of(t + 0.5 * d);
            for (c, rt) in r.iter_mut().enumerate().take(if still { 1 } else { 3 }) {
                self.r_at(&middle, t + c as f64 * 0.5 * d, rt);
            }
            let rc = |c: usize| if still { 0 } else { c };
            // G1 = g(t, σ); U2 = e^{z/2}σ + d a21 G1; G2 = g(t + d/2, U2).
            self.change_at(f, t, lw);
            self.remainder(f, &r[rc(0)], lw, &sigma, &mut k[0]);
            for i in 0..sigma.len() {
                u[i] = e2[i] * sigma[i] + a21[i] * k[0][i] * d;
            }
            self.change_at(f, t + 0.5 * d, lw);
            self.remainder(f, &r[rc(1)], lw, &u, &mut k[1]);
            // U3 = e^{z/2}σ + d (a31 G1 + a32 G2); G3 = g(t + d/2, U3).
            for i in 0..sigma.len() {
                u[i] = e2[i] * sigma[i] + (a31[i] * k[0][i] + a32[i] * k[1][i]) * d;
            }
            self.remainder(f, &r[rc(1)], lw, &u, &mut k[2]);
            // U4 = e^z σ + d (a41 G1 + a43 G3); G4 = g(t + d, U4).
            for i in 0..sigma.len() {
                u[i] = e[i] * sigma[i] + (a41[i] * k[0][i] + a43[i] * k[2][i]) * d;
            }
            self.change_at(f, t + d, lw);
            self.remainder(f, &r[rc(2)], lw, &u, &mut k[3]);
            // σ ← e^z σ + d (b1 G1 + b23 (G2 + G3) + b4 G4).
            for i in 0..sigma.len() {
                sigma[i] = e[i] * sigma[i] + (b1[i] * k[0][i] + b23[i] * (k[1][i] + k[2][i]) + b4[i] * k[3][i]) * d;
            }
            counts.products += 4;
            counts.work += 4.0 * self.product;
            counts.segments += 1;
        }
        // ρ = V σ V†.
        let mut tmp = std::mem::take(&mut lw.tmp);
        self.sandwich(&f.v, &f.vadj, &sigma, v, &mut tmp);
        lw.tmp = tmp;
        lw.k = k;
        lw.r = r;
        (lw.s[0], lw.s[1]) = (sigma, u);
        counts.exponentials += 1;
        counts.work += 2.0 * self.sandwiched;
    }
}

/// The generator's memory at a point (module documentation, # The memory): its slowest nonzero decay rate (1/µs), the
/// stationary modes' count (the rounding zeros), and the memory time 1/λ.
#[derive(Clone, Copy, Debug)]
pub struct Memory {
    pub rate: f64,
    pub zeros: usize,
    pub time: f64,
    /// The next three nonzero decay rates, increasing (a complex mode and its partner each listed): the fast memory.
    pub rates: [f64; 3],
}

impl Memory {
}

/// The memory of a model at the point r, κ and the time t (`Memory`): None where the sector's Liouvillian does not
/// decompose.
pub fn memory(form: &Form, r: V3, kappa: V3, t: f64) -> Option<Memory> {
    let l = form.liouvillian(r, kappa, t);
    let d = l.rows;
    let mut eigen = Eigen::default();
    if !eigen.decompose(&l) {
        return None;
    }
    let scale = rounding(d) * l.norm1();
    let (mut zeros, mut rate) = (0, f64::INFINITY);
    let mut all: Vec<f64> = Vec::new();
    for (i, mu) in eigen.values.iter().enumerate() {
        // κ_μ = ‖r_i‖₂ ‖l_i‖₂ of its right vector (R's column) and left one (R⁻¹'s row), l_i r_i = 1.
        let right: f64 = (0..d).map(|x| eigen.vectors[(x, i)].norm_sqr()).sum::<f64>().sqrt();
        let left: f64 = (0..d).map(|x| eigen.inverse[(i, x)].norm_sqr()).sum::<f64>().sqrt();
        if mu.abs() <= right * left * scale {
            zeros += 1;
        } else {
            rate = rate.min(-mu.re);
            all.push(-mu.re);
        }
    }
    all.sort_by(f64::total_cmp);
    // The next three nonzero rates (a complex mode's partner holds its rate: L(ρ†) = L(ρ)†).
    let rates = std::array::from_fn(|k| all.get(k + 1).copied().unwrap_or(f64::NAN));
    Some(Memory { rate, zeros, time: 1.0 / rate, rates })
}

/// A path mean and its standard error over the paths (the paths' spread over √P; NaN of one path).
#[derive(Clone, Copy, Debug, Default)]
pub struct Mean {
    pub value: f64,
    pub error: f64,
}

impl Mean {
    /// The mean of the paths' values `x` and its standard error, the paths' spread over √P: none to read on one path
    /// (without bound). Of no path nothing is read: its window is unread (`Window::paths`, `select::direct_at`).
    pub fn of(x: &[f64]) -> Mean {
        Mean::weighted(x, &vec![1.0; x.len()])
    }

    /// The weighted mean of the paths' values `x` by their weights `w` (their strata's, `Walk::weight`) and its standard
    /// error, n/(n − 1) Σ w²(x − m)² / (Σ w)² -- the paths' spread over √P where the weights are alike; none to read on
    /// one path.
    pub fn weighted(x: &[f64], w: &[f64]) -> Mean {
        let p = x.len() as f64;
        let sum: f64 = w.iter().sum();
        let value = x.iter().zip(w).map(|(x, w)| x * w).sum::<f64>() / sum;
        if x.len() < 2 {
            return Mean { value, error: f64::INFINITY };
        }
        let var = x.iter().zip(w).map(|(x, w)| w * w * (x - value) * (x - value)).sum::<f64>() / (sum * sum) * p / (p - 1.0);
        Mean { value, error: var.sqrt() }
    }
}

/// A candidate's direct error over one time window (module documentation): per quantity its window mean's difference
/// to the reference on each path, and those over the paths (`Mean`) -- over the `paths` that read in it, a lost atom's
/// path ending early: a window no path reads (every path's atom lost before it) reads nothing, and its means are no
/// numbers no reader takes.
#[derive(Clone, Debug, Default)]
pub struct Window {
    pub from: f64,
    pub to: f64,
    pub paths: usize,
    /// Every displayed row; each level's sum (its states' labels without the last part); the worst row of the means.
    pub rows: Vec<Mean>,
    /// The window's rows by its parts (`parts`): the window halved while its two halves' rows differ beyond their
    /// statistics -- the direct part reads each part's (select.rs `direct`), so that a change inside a long window is
    /// not averaged away.
    pub parts: Vec<Part>,
    pub levels: Vec<(String, Mean)>,
    pub worst: (usize, Mean),
    /// Per path its worst row's window mean |Δ| (for the distribution and the account).
    pub worst_per_path: Vec<f64>,
    /// Per photon class (the reference's, by fast index) its rate's difference (1/µs), the reference's mean rate, and
    /// the total rate's relative difference.
    pub rates: Vec<([i64; 2], Mean, f64)>,
    pub rate: Mean,
    /// The force's difference per lab axis (rad/µs/µm), and the rms difference over the rms force per path.
    pub force: [Mean; 3],
    pub force_rms: Mean,
    /// The motion's (# The motion's readings; select.rs # The motion): per lab axis the reference's heating H_a and the
    /// candidate's thermal energy rate's difference δP_a = δH_a + (δF_a − δF̄_a)(v_a − v̄_a) (rad/µs²), the bars the
    /// window's means over the paths read in it -- the cloud's own motion, which T is read about (estimate.rs, # The
    /// cloud's own motion): a force along it, or the same force on every atom, moves the cloud, not its T; the force's
    /// covariance with the velocity over the cloud is what heats or cools it.
    pub heating: [Mean; 3],
    pub energy_rate: [Mean; 3],
    /// Per path read in the window (its place among the paths) its δP_a, the reference's H_a and the candidate's own H_a
    /// (rad/µs²; the heating alone, the work apart): the total T's error weighs the axes by each one's response -- its
    /// relaxation the energy balance of the model the pilot runs, select.rs `motion_at` --, and reads its rates over the
    /// paths' whole forgotten stretches (select.rs `Stationary`), its spread over the paths that sum's own.
    pub energy_rates: Vec<(usize, [f64; 3], [f64; 3], [f64; 3])>,
    /// Per path read in the window its mean force difference δF_a and the reference's own mean force F_a there (1/µm/µs):
    /// the push's error is read on their regression (select.rs `motion_at`, # The push's error).
    pub forces: Vec<(f64, [f64; 3], [f64; 3])>,
}

/// A part of a window (`parts`): its span (from, to], the paths read in it and every displayed row's mean difference
/// over them.
#[derive(Clone, Debug, Default)]
pub struct Part {
    pub from: f64,
    pub to: f64,
    pub paths: usize,
    pub rows: Vec<Mean>,
}

/// A window's parts (select.rs, # The direct part): (from, to] halved while its two halves' rows differ -- on the paths
/// that read both, each path's difference of its two halves' means of a row, their mean beyond Student's t at the run's
/// confidence among the rows times its standard error (`Mean::weighted`, paired: what a path's place makes of both
/// halves cancels) --, each half again; a part ends where no two paths read both its halves (a part of one reading a
/// path is not halved). Per part its rows' means over the paths read in it, by the paths' `weights`. The windows grow
/// as the memory's octaves (`windows`): a change inside a long one -- R9_PUSH's loss front at 14-34 µs inside (12, 25]
/// and (25, 50] -- read by its mean alone is averaged away, the direct part taking the window's mean (until 2026-11-23).
fn parts(candidate: &[Series], reference: &[Series], from: f64, to: f64, weights: &[f64]) -> Vec<Part> {
    // Per path read in (a, b], its place and its rows' mean difference there (the two series read at the same times).
    let rows_in = |a: f64, b: f64| -> Vec<(usize, [f64; STATES])> {
        candidate
            .iter()
            .zip(reference)
            .enumerate()
            .filter_map(|(k, (c, r))| {
                let (lo, hi) = (r.readings.partition_point(|y| y.t <= a), r.readings.partition_point(|y| y.t <= b));
                let hi = hi.min(c.readings.len());
                (lo < hi).then(|| {
                    let mut sum = [0.0; STATES];
                    for (x, y) in c.readings[lo..hi].iter().zip(&r.readings[lo..hi]) {
                        for s in 0..STATES {
                            sum[s] += x.rows[s] - y.rows[s];
                        }
                    }
                    (k, sum.map(|v| v / (hi - lo) as f64))
                })
            })
            .collect()
    };
    let weight = |k: usize| weights.get(k).copied().unwrap_or(1.0);
    let mean_of = |x: &[(usize, f64)]| Mean::weighted(&x.iter().map(|p| p.1).collect::<Vec<_>>(), &x.iter().map(|p| weight(p.0)).collect::<Vec<_>>());
    let q_rows = |paths: usize| crate::estimate::quantile(crate::estimate::tail_among(STATES as f64), Some(paths as u64 - 1));
    let mut out = vec![];
    let mut open = vec![(from, to)];
    while let Some((a, b)) = open.pop() {
        let m = 0.5 * (a + b);
        let (first, second) = (rows_in(a, m), rows_in(m, b));
        let both: Vec<(usize, [f64; STATES], [f64; STATES])> = first.iter().filter_map(|(k, x)| second.iter().find(|(j, _)| j == k).map(|(_, y)| (*k, *x, *y))).collect();
        let split = both.len() >= 2 && {
            let q = q_rows(both.len());
            (0..STATES).any(|s| {
                let d = mean_of(&both.iter().map(|(k, x, y)| (*k, x[s] - y[s])).collect::<Vec<_>>());
                d.error.is_finite() && d.value.abs() > q * d.error
            })
        };
        if split {
            // The later half first on the stack: the parts come out in time's order.
            open.push((m, b));
            open.push((a, m));
            continue;
        }
        let read = rows_in(a, b);
        out.push(Part { from: a, to: b, paths: read.len(), rows: (0..STATES).map(|s| mean_of(&read.iter().map(|(k, x)| (*k, x[s])).collect::<Vec<_>>())).collect() });
    }
    out
}

/// A landing check's numbers at a time of a path (# The landing check), the candidate's first and the reference's:
/// each one's Mandel Q of its photons from the reference's landing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Statistics {
    pub t: f64,
    pub q: [f64; 2],
    /// Each model's Q's last doubling's change (`Form::landing`), booked beside its Q (`compare`).
    pub q_error: [f64; 2],
}

impl Statistics {
    /// Nothing read: Poisson's count.
    pub const NONE: Statistics = Statistics { t: 0.0, q: [0.0; 2], q_error: [0.0; 2] };

    /// A path's landing check over the window (from, to]: the mean of its landings there, else its nearest one's; none
    /// read of no landing.
    pub fn within(landings: &[Statistics], from: f64, to: f64) -> Statistics {
        let inside: Vec<&Statistics> = landings.iter().filter(|x| x.t > from && x.t <= to).collect();
        let near = |x: &&Statistics| (x.t - from).abs().min((x.t - to).abs());
        let at: Vec<&Statistics> = if inside.is_empty() { landings.iter().min_by(|a, b| near(a).total_cmp(&near(b))).into_iter().collect() } else { inside };
        if at.is_empty() {
            return Statistics::NONE;
        }
        let k = at.len() as f64;
        Statistics {
            t: at.iter().map(|x| x.t).sum::<f64>() / k,
            q: std::array::from_fn(|m| at.iter().map(|x| x.q[m]).sum::<f64>() / k),
            q_error: std::array::from_fn(|m| at.iter().map(|x| x.q_error[m]).sum::<f64>() / k),
        }
    }
}

/// The windows of a path of the length `length` (µs): (0, e₁], (e₁, 2e₁], … doubling up to the length, e₁ the
/// memory time halved down to the first below four times the path's `step` -- the start's transient resolved where it
/// changes fastest, the error's approach to its stationary value (the memory's scale) in octaves; the last window ends
/// at the length.
pub fn windows(length: f64, memory: f64, step: f64) -> Vec<(f64, f64)> {
    let mut e = memory;
    while e > 4.0 * step {
        e *= 0.5;
    }
    let mut edges = vec![0.0];
    while e < length {
        edges.push(e);
        e *= 2.0;
    }
    edges.push(length);
    edges.windows(2).map(|x| (x[0], x[1])).collect()
}

/// The window an output time takes its error from (PIPELINE_V3_FINAL B.8: "the time window it falls in"): the window
/// (from, to] holding t; past the last window's end the last window -- a path of L = 5τ_mem has forgotten its start
/// there, the error stationary --; None at or before the first window's start (every model starts in the same state).
pub fn window_of(windows: &[(f64, f64)], t: f64) -> Option<usize> {
    if windows.is_empty() || t <= windows[0].0 {
        return None;
    }
    Some(windows.iter().position(|w| t <= w.1).unwrap_or(windows.len() - 1))
}

/// The candidate's direct errors against the reference on the same paths (`Window` per window `(from, to]`):
/// `candidate[p]` and `reference[p]` read along path p at the same times; `labels` the request's states. A window's
/// statistics are over the paths read in it.
pub fn compare(candidate: &[Series], reference: &[Series], windows: &[(f64, f64)], labels: &[String], hbar_over_mass: f64, landings: &[Vec<Statistics>], weights: &[f64]) -> Vec<Window> {
    let mut levels: Vec<(String, Vec<usize>)> = Vec::new();
    for (s, l) in labels.iter().enumerate() {
        let level = l.rsplit_once('-').map_or(l.as_str(), |x| x.0).trim_end_matches('-').to_string();
        match levels.iter_mut().find(|x| x.0 == level) {
            Some(x) => x.1.push(s),
            None => levels.push((level, vec![s])),
        }
    }
    // The photon classes of the first path that reads any (a path that ended before the windows reads none).
    let classes = &reference.iter().find(|s| !s.readings.is_empty()).unwrap_or(&reference[0]).classes;
    windows
        .iter()
        .map(|&(from, to)| {
            let p = candidate.len();
            let mut rows = vec![vec![0.0; p]; STATES];
            let mut lv = vec![vec![0.0; p]; levels.len()];
            let mut rates = vec![vec![0.0; p]; classes.len()];
            let mut rate_ref = vec![0.0; classes.len()];
            let mut rate = vec![0.0; p];
            let mut force = vec![[0.0; 3]; p];
            let mut own_force = vec![[0.0; 3]; p];
            let mut force_rms = vec![0.0; p];
            let (mut heating, mut energy_rate, mut own_heating) = (vec![[0.0; 3]; p], vec![[0.0; 3]; p], vec![[0.0; 3]; p]);
            let mut read = vec![false; p];
            // The cloud's mean velocity and mean force's difference in the window over the paths that read in it (# The
            // motion's readings).
            let (mut sum, mut pushed, mut count) = ([0.0; 3], [0.0; 3], 0.0);
            for (c, r) in candidate.iter().zip(reference) {
                for (x, y) in c.readings.iter().zip(&r.readings).filter(|(_, y)| y.t > from && y.t <= to) {
                    for a in 0..3 {
                        sum[a] += hbar_over_mass * y.kappa[a];
                        pushed[a] += x.force[a] - y.force[a];
                    }
                    count += 1.0;
                }
            }
            let (cloud, push): ([f64; 3], [f64; 3]) = (sum.map(|x| if count > 0.0 { x / count } else { 0.0 }), pushed.map(|x| if count > 0.0 { x / count } else { 0.0 }));
            for (k, (c, r)) in candidate.iter().zip(reference).enumerate() {
                let pairs: Vec<(&Reading, &Reading)> = c.readings.iter().zip(&r.readings).filter(|(x, _)| x.t > from && x.t <= to).collect();
                assert!(pairs.iter().all(|(x, y)| x.t == y.t), "the two series read at the same times");
                let q = pairs.len() as f64;
                read[k] = !pairs.is_empty();
                // The path's landing check in the window (# The landing check): its landings' there, else its nearest one's;
                // none read where it has none.
                let at = landings.get(k).map_or(Statistics::NONE, |l| Statistics::within(l, from, to));
                let (mut df2, mut f2, mut total_m, mut total_r) = (0.0, 0.0, 0.0, 0.0);
                let mut booked = [0.0; 3];
                for (x, y) in &pairs {
                    let (hx, hy) = (x.heating_at(hbar_over_mass, at.q[0]), y.heating_at(hbar_over_mass, at.q[1]));
                    // What Q's booked change moves in each model's heating (`Statistics::q_error`).
                    let (bx, by) = (x.heating_at(hbar_over_mass, at.q_error[0]), y.heating_at(hbar_over_mass, at.q_error[1]));
                    let (px, py) = (x.heating(hbar_over_mass), y.heating(hbar_over_mass));
                    for a in 0..3 {
                        booked[a] += ((bx[a] - px[a]).abs() + (by[a] - py[a]).abs()) / q;
                    }
                    for a in 0..3 {
                        heating[k][a] += hy[a] / q;
                        own_heating[k][a] += hx[a] / q;
                        energy_rate[k][a] += ((hx[a] - hy[a]) + (x.force[a] - y.force[a] - push[a]) * (hbar_over_mass * y.kappa[a] - cloud[a])) / q;
                    }
                    for s in 0..STATES {
                        rows[s][k] += (x.rows[s] - y.rows[s]) / q;
                    }
                    for (l, (_, states)) in levels.iter().enumerate() {
                        lv[l][k] += states.iter().map(|&s| x.rows[s] - y.rows[s]).sum::<f64>() / q;
                    }
                    for (ci, fast) in classes.iter().enumerate() {
                        let mine = c.classes.iter().position(|f| f == fast).map_or(0.0, |i| x.rates[i]);
                        rates[ci][k] += (mine - y.rates[ci]) / q;
                        rate_ref[ci] += y.rates[ci] / q / p as f64;
                    }
                    total_m += x.rates.iter().sum::<f64>() / q;
                    total_r += y.rates.iter().sum::<f64>() / q;
                    for a in 0..3 {
                        let d = x.force[a] - y.force[a];
                        force[k][a] += d / q;
                        own_force[k][a] += y.force[a] / q;
                        df2 += d * d / q;
                        f2 += y.force[a] * y.force[a] / q;
                    }
                }
                // Q's booked change, away from nought (the bound the landing's stretch leaves).
                for a in 0..3 {
                    energy_rate[k][a] += booked[a].copysign(energy_rate[k][a]);
                }
                rate[k] = total_m / total_r - 1.0;
                force_rms[k] = (df2 / f2).sqrt();
            }
            // The paths read in the window (a lost atom's path ends early).
            let held: Vec<usize> = (0..p).filter(|&k| read[k]).collect();
            let weight = |k: usize| weights.get(k).copied().unwrap_or(1.0);
            let of = |x: &[f64]| Mean::weighted(&held.iter().map(|&k| x[k]).collect::<Vec<_>>(), &held.iter().map(|&k| weight(k)).collect::<Vec<_>>());
            let row_means: Vec<Mean> = rows.iter().map(|x| of(x)).collect();
            let worst_row = (0..STATES).max_by(|&a, &b| row_means[a].value.abs().total_cmp(&row_means[b].value.abs())).unwrap();
            let worst_per_path = held.iter().map(|&k| rows.iter().map(|x| x[k].abs()).fold(0.0, larger)).collect();
            Window {
                from,
                to,
                paths: held.len(),
                worst: (worst_row, row_means[worst_row]),
                rows: row_means,
                parts: parts(candidate, reference, from, to, weights),
                levels: levels.iter().zip(&lv).map(|((name, _), x)| (name.clone(), of(x))).collect(),
                worst_per_path,
                rates: classes.iter().zip(&rates).zip(&rate_ref).map(|((f, x), r)| (*f, of(x), *r * p as f64 / held.len() as f64)).collect(),
                rate: of(&rate),
                force: std::array::from_fn(|a| of(&force.iter().map(|f| f[a]).collect::<Vec<_>>())),
                force_rms: of(&force_rms),
                heating: std::array::from_fn(|a| of(&heating.iter().map(|h| h[a]).collect::<Vec<_>>())),
                energy_rate: std::array::from_fn(|a| of(&energy_rate.iter().map(|h| h[a]).collect::<Vec<_>>())),
                energy_rates: held.iter().map(|&k| (k, energy_rate[k], heating[k], own_heating[k])).collect(),
                forces: held.iter().map(|&k| (weight(k), force[k], own_force[k])).collect(),
            }
        })
        .collect()
}




/// The freezes' parts' nanoseconds over every thread (the measurements' trace, select.rs `traced`): the nodes, their
/// amplitudes, the eigendecompositions, the substeps' weights, R and ΔH at the stages.
#[cfg(test)]
pub(crate) static FREEZE: [std::sync::atomic::AtomicU64; 5] = [const { std::sync::atomic::AtomicU64::new(0) }; 5];

/// A freeze's part `k` (`FREEZE`) since `from`.
#[cfg(test)]
fn traced(k: usize, from: std::time::Instant) {
    FREEZE[k].fetch_add(from.elapsed().as_nanos() as u64, std::sync::atomic::Ordering::Relaxed);
}

/// The freezes' parts (`FREEZE`) in seconds, and zeroed.
#[cfg(test)]
pub(crate) fn freeze_parts() -> [f64; 5] {
    std::array::from_fn(|k| FREEZE[k].swap(0, std::sync::atomic::Ordering::Relaxed) as f64 * 1e-9)
}

#[cfg(test)]
thread_local! {
    /// What a test asks of the pilot's paths (solve.rs `walks`: how many, to when) and what it was handed.
    pub(crate) static WALKS: std::cell::RefCell<Option<(usize, f64, Vec<Walk>)>> = const { std::cell::RefCell::new(None) };
}

/// The pilot's paths to a test that asks for them (`WALKS`; solve.rs `attempt`).
#[cfg(test)]
pub(crate) fn hand_out(walks: &dyn Fn(usize, f64) -> Vec<Walk>) {
    WALKS.with(|w| {
        if let Some((count, until, out)) = w.borrow_mut().as_mut() {
            *out = walks(*count, *until);
        }
    });
}

#[cfg(test)]
mod tests;
