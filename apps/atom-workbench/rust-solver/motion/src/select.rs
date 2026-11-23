//! The selection of P (PIPELINE_V3_FINAL §二 A.3, A.5–A.6, B.7–B.11, C.13–C.15): the candidates by structure, each
//! judged by what it does to the run's readings along the atoms' own paths against the 32-state model (paths.rs), and
//! the candidate of least total work chosen among those whose model error A leaves the targets room.
//! The trajectories' parts (the pilot the paths come from, T's memory read on the run's own trajectories, the work)
//! are the solve's (solve.rs, # The selection); what is here is the model's side and the accounts.
//!
//! # The candidates
//! Every excited state e has an importance: its first-order excitation Σ_g p_g Σ_ω |V_eg|²/|E_g + f − E_e + iΓ_e/2|²
//! by the light other than the tweezer's (`excitation`: the population it would hold to the first order, the fold's
//! leading term weighed by the ground states its atoms come from), the largest over the places the pilot's atoms were
//! seen at with their ground populations there (`importance`; before the pilot, `amplitudes`: over the three structural
//! points -- the tweezer's focus with its light full (s = 1), the focus with it off (s = 0), and the place of half its
//! intensity along the direction the other light changes fastest (`middle`) -- from every ground state alike). No
//! equation is solved for it: it is the coupling over the detuning, where the atoms are. (By the amplitude alone,
//! |V|/|Δ + iΓ/2|, PGC3D's repumper resonant on F = 1 → F′ = 2 put F′ = 2 first, 0.27 against 0.06–0.11 for F′ = 3,
//! while 90 % of the atoms are in F = 2: the three-point choice of 2026-11-21 kept F′ = 3 m = 0, ±1 and folded every
//! F′ = 2 state.) The structural chain (`chain`) folds the excited states one at a time in increasing importance, from
//! the 32 states down to the full-ground model G: the run starts on G, and where G's fold runs away at every pilot's start
//! the attempt climbs it to the first model that does not (solve.rs `piloted`). It chooses nothing.
//!
//! The candidates the choice weighs (solve.rs `choose_on_the_pilot`, the user's of 2026-11-22, PLAN_UNIFIED §三 2) are
//! any subset of the excited states the pilot's model folds, by what each does measured, and the verification decides:
//! - each addable state -- one the light excites where the pilot's atoms were, its importance above nought -- alone added
//!   to the pilot's model, judged on the first `NOMINATION_WALKS` paths (`judge_on`) at its point estimates (`Reach::Point`):
//!   its contribution the misfit it takes away -- the model error over its target, the largest over the times and the
//!   readings -- per work it adds a µs, and whether it meets every look there;
//! - each single that does is judged alone on every path and weighed, a candidate: where a single state is enough, the
//!   singles' needs at their point estimates are the run's own statistics' (R4 2026-11-23: 771 to 794 trajectories each,
//!   the pilot's own 780), and what tells their W apart is their direct part's spread over the paths, which two paths do
//!   not see -- on every path 833 to 3133 trajectories; the one first by its misfit on two paths was 13th of 24 by its W,
//!   and the cheapest took none away there (its contribution is where other atoms went: the lattice's polarization);
//! - the states in the order of those singles' W (the least first), then the others' contribution (those taking none
//!   away after them, by their importance), the nominees the order's prefixes, the 32 states last: a group is a prefix,
//!   whatever its states (no chain by structure, no bisection, no rule of mirror pairs or of whole levels);
//! - each prefix from the first one predicted to fit -- the pilot's misfit less what its states took away alone, below
//!   one (the rates and the forces add over single states to 0.1–1 percentage points, the rows not: Study A §2.2) -- is
//!   judged on every path, the next where it is not complete; above a complete one while a nominee's least W stands
//!   below the least found.
//! Where the pilot's model is complete and no model with more states could work less (its work a µs at the ladder's first
//! rung no less than the pilot's model's), nothing is nominated. What stays is the runaway's check of the steps (step.rs,
//! # The domain): a fold that diverges where an atom goes scatters beyond every state of the atom there and is booked
//! whole.
//!
//! # The paths
//! The candidates are judged along `WALKS` of the pilot's trajectories (solve.rs `walks`; P1-PATHS): each is advanced
//! by the candidate and by the 32-state model in the ground-holding exact blocks' density matrices along the same path
//! (paths.rs `Form::along`), by the exponential scheme in H's eigenbasis (paths.rs, # The substeps), and their readings compared
//! in windows of the memory time (paths.rs `windows`, `compare`). A path is read at its start for L = min(T, 5 τ_mem),
//! τ_mem the slowest decay of the full-ground model's generator, the longest over the paths' starts (`memory`; P1 §5: it
//! differs with the place by up to 2.4 times) -- the start's transient and the internal state's forgetting of it --, and,
//! where the run is longer than 2L, again over its last L from the walk's own start state (`Paths::late_from`): its last
//! window reads the model where the atoms are late in the run, which the start's windows cannot (the motion's own
//! memory is the longer: R4's atoms heat through the run).
//!
//! The scheme's setting (`Setting`) -- the substeps a turn of the fastest coherence over a stretch's opening (the
//! start's transient, paths.rs # The start), the substeps a µs after it, and the pieces a step -- is settled by its own
//! error, not fixed
//! (`settle`): along a stretch the reference is advanced at the setting and at each knob
//! doubled, each knob's part of the setting's error the largest change of a displayed row from doubling it alone
//! (Richardson's 2^p/(2^p − 1), p = 2: the amplitudes' linear interpolation at one piece a step, R's at the substep's
//! middle -- paths.rs # The nodes, # The substeps; a rate no lower than one substep a piece, so
//! that where twice the pieces raise a rate to that floor the pieces' part is read at the raised rates,
//! `refinements`), and beside them the trace's part, the setting's own run's trace defect (paths.rs
//! `Series::trace_defect`: every model's flow keeps tr ρ_P, so its departure is a strict lower bound of the scheme's
//! error -- one the knobs' parts cannot see, the rows read over their sum); from
//! the coarsest (`COARSEST`), the knob of the largest part is doubled -- where the trace's is the largest, the knob whose
//! refinement lowers it most -- while the parts' sum is beyond `TOLERANCE` -- s⁴
//! of the population target, the share the numerical tolerances inside a step take of theirs (units.rs `QUADRATURE`)
//! --, to `FINEST` at the most (a bound on the work, not on the accuracy: stopped there short of the tolerance, the
//! setting is said not converged and the reference's parts are booked whole in every candidate's A,
//! `Settled::unmet`). The stretch is the one the setting is taken for: the longest path over the whole of its L, where a
//! knob's error has accumulated over every piece the paths take (the propagator contracts the trace norm, so each
//! piece's own error adds and none grows: over a run's L at a slow memory -- GMD1C's 34 ms, S1's 0.75 s -- the sum of
//! hundreds of µs of them, which no short piece holds). The ladder climbs first on its first `CALIBRATION` µs, where a
//! run costs a fraction of the whole length's, and goes on from there over the whole stretch, where it stops: the
//! first piece decides nothing, it only starts the second stage near its end (a setting that meets the tolerance there
//! is checked there in one round of the refinements, the setting's own run being the path's). The trace is read on
//! every path, not on the calibration's alone: where another stretch's defect is beyond the tolerance at the setting,
//! the ladder goes on along that stretch from the setting, and the paths are run again at what it ends at (`Paths::new`);
//! what defect is left at the ladder's top is booked in every candidate's A, the reference's largest over the paths
//! beside the candidate's own (`Paths::trace`, `judge`). (Until 2026-11-22 the trace was read nowhere: the reference's
//! fell to 1.6e-5 on SAT_03's first 5 µs and drifted to 0.55 along S1's paths, 1.04 along tilt0's -- Study A §2.6.)
//! Every path and every candidate take that setting. The same refinements of each candidate along the same stretch give its
//! calibration (`Calibration`): the change of its rows' difference to the reference, read alike, booked beside the
//! direct part. (Until 2026-11-22 the setting was fixed at
//! 128 substeps a µs and two pieces, P1-PATHS's measure on R4, and the calibration read against the converged
//! `Scheme::sparse`, whose segments follow the fastest coherence: on S1 -- its 32 states hold levels far off the light's
//! frame -- 0.5 µs of it took more than 190 s, where the whole setting ladder takes a second. The refinements
//! converge to the same limit: H exact in each piece's eigenbasis as the pieces shorten, the exponential Runge–Kutta
//! as the substeps do. Until SELCOST the substeps were Lawson's, whose rate had to resolve the detunings, paths.rs #
//! The substeps.)
//!
//! The reference is the 32-state model itself (B.8), not a whole-level base: under P1's scheme its path cost 10–20 % more
//! than the base's (P1 R2.3), and with it nothing remains of a base's own error to check request by request.
//!
//! # The direct part
//! Per output time t, its window's (`paths::window_of`) error of each displayed row, |mean| + q·se over the paths
//! (`paths::Mean`), q Student's t at the run's confidence (the normal's two-sided tail at `CONFIDENCE`) with the paths
//! read in the window less one degrees of freedom -- the paths' own statistics, not the normal's z (at four paths z = 2
//! covers 86 %, the t quantile 95.4 %) --, the largest over the rows, and the scheme's calibration beside it; past L the
//! larger of the start's last window's and the late segment's last window's, and past the late segment's start the
//! latter's. Of one path the spread is unknown: its upper end is without bound.
//!
//! # How many paths
//! The paths are the pilot's trajectories in two strata by the time each held its atom (solve.rs `walks`): those held to
//! the run's end -- whose paths read the late windows -- and those lost before it -- whose paths went where atoms are
//! lost, the trap's rim --, half the paths from each (all from one where the other has none), each path weighed by its
//! stratum's share of the pilot's trajectories over the stratum's paths (`paths::Walk::weight`): a window's mean is
//! their weighted mean over the paths read in it, its standard error n/(n − 1) Σ w²(x − m)²/(Σ w)² (`paths::Mean::
//! weighted`). (Until 2026-11-22 the pilot's first trajectories: S1's four were all lost by 1.5 ms, its last windows
//! unread and booked whole -- 950 times the work of its full-ground model; and scans of the trap's axes marked a region
//! the paths did not reach -- 10–80 % of the selection, a straight line's error, not a path's: Study A §2.3.)
//!
//! From `WALKS`, twice as many where it pays: a candidate complete at its account's lower end, |mean| − q·se (none below
//! zero), and not at its upper -- its fate not decided at the paths' statistics -- would save, decided complete, the
//! least W found -- or the 32 states', complete by construction, where that is less -- less its own W at its lower end; twice the paths cost the reference's and every weighed candidate's
//! counted work on them again (`paths::Counts::work`, the trajectories' price's measure). Doubled where the saving is
//! more than that, up to all of the pilot's trajectories, the candidates judged again on them -- and before anything is
//! built on an open fate: the pilot's model's or a nominee's; else its upper end stands, said. (Until 2026-11-22 doubled
//! whatever it cost: S2's selection, its full-ground model open on the loss, doubled to 15 paths and ran past 300 s.)
//!
//! A state of the pilot's from which a candidate takes no step -- its fold runs away there (step.rs # The domain) -- is
//! an unresolved atom's: its trajectory stops there and works nothing more, and its share of the pilot's states,
//! weighed as the price is, is booked whole in the candidate's account (solve.rs `priced_with`, # The unresolved atoms).
//! (Until 2026-11-22 one such state made the model's price without bound: p5d5's full-ground model, one of its 254
//! states at the trap's rim, Study A §2.4.)
//!
//! # The motion
//! The model's internal error moves the atoms, and where they go moves the readings the motion makes: the loss, T per
//! lab axis, the push and the mean velocity. Along the paths the candidate and the reference read beside the rows what
//! the motion takes of the internal state (paths.rs # The motion's readings): per lab axis the heating H_a (the scattered
//! light's momentum diffusion) and the force's power F_a v_a, so that a window holds per axis the reference's H_a and the
//! candidate's energy rate's difference δP_a = δH_a + (δF_a − δF̄_a)(v_a − v̄_a), the bars the cloud's means there
//! (paths.rs `Window`: the force's covariance with the velocity over the cloud). The readings' errors follow to the first order by the
//! time-scaling identity (the user's of 2026-11-22, PLAN_UNIFIED §三 4; Study A §2.1.4): the motion and the internal
//! state's generator 𝓛, a reading O(t) = ⟨μ₀, e^{t𝓛} f⟩; a candidate's 𝓛′ = 𝓛 + x 𝓛_m, the mechanism m scaled by 1 + x,
//! moves it by δO(t) = x ∫₀ᵗ ⟨μ₀ e^{s𝓛} 𝓛_m e^{(t−s)𝓛} f⟩ ds = x ∂O/∂ln s_m (Duhamel); a reading the mechanism alone
//! drives moves by x t Ȯ(t), and a steady T = D/α by δT/T = x_D − x_α (`motion_at`):
//! - **T**, the total's (the axes' mean, the one T's target is held to: the user's of 2026-11-22), of each lab axis's
//!   response, the axes weighed per path (the sum's spread its own): the energy of the axis E_a = k_B T_a (the trap's
//!   equipartition) gains H_a + W_a; the candidate's
//!   by δP_a more. It relaxes at α_a = (H_a − Ė_a)/E_a (the reference's heating less the pilot's own Ė_a, none below
//!   nought), so δE_a(t) = δP_a (1 − e^{−α_a t})/α_a: δT_a = δP_a t/k_B on an axis that does not cool (α = 0: x t Ṫ of
//!   the heating's x), δT_a/T_a = δP_a/H_a at the steady state (x_D − x_α: the heating's relative difference less the
//!   friction's, the work's difference over the friction's power, which is H_a there).
//! - **The photons' statistics and the force's fluctuations** (paths.rs # The landing check): at each window's last
//!   reading that leaves it room, the reference's state mapped to its landing after a jump, each model on from there --
//!   its Mandel Q, of the directed absorbed part its jumps carry --, and each model's force correlation's diffusion at
//!   its own state there, in its heating (`Paths::landings`, `judge`: SAT_03's resonant two-level cycle, the reference's
//!   Q = −0.53, the full-ground model's 0; the latter heats the push axis 2.1 times as much, Study A's prediction 2.10
//!   of the measured 1.80 ± 0.5).
//! - T's rates are each path's over its whole stretches past the memory time (`Stationary`): a window shorter than the
//!   motion's periods holds the conservative exchange, bounded over any time, which a short window's rate times an
//!   axis's whole response would take for heating.
//! - **The loss**: an atom is lost where its energy reaches the escape value; the energy rate's relative difference x
//!   moves its time by x of it, δL(t) = x t L̇(t). x is the lost atoms' own: per lost path its mean energy rate's
//!   difference Σ_a (δH_a + δF_a v_a) over the energy it gained to its loss per µs, (E(loss) −
//!   E(0))/t_loss on the reference's prepared potential -- the paths' time mean over every atom thins out the trap's rim,
//!   where an atom about to be lost is (Study A §2.1.4: S2's loss 55–75 % predicted so) --; over every path's heating
//!   where fewer than two paths' atoms are lost (no spread to read of fewer) (`Judged::escape`).
//! - **The push and the mean velocity** per lab axis: the force's window mean difference δF_a, the push's error
//!   (ħ/m)|δF_a| (µm/µs²), and the mean velocity's (ħ/m)|δF_a| min(t, 1/ω_a) (m/s) -- a constant force's response of a
//!   mean velocity in a harmonic well, (δF/mω) sin ωt, at most that.
//! Each a window's |mean| + q·se over the paths (Student's t, # The direct part), the window the direct part takes at the
//! time. Point estimates of the first order, not bounds: the second order is x² (2 % at x = 15 %), and a mechanism mixed
//! of several is read as their whole energy rate. (Until 2026-11-22 the motion's feedback was set to nought, the user's
//! of that day while no bound could be had: R9_PUSH's loss 0.18 above the reference, SAT_03's push ×1.30, S2's loss
//! 0.55 against 0.42 booked nothing.)
//!
//! ## T's memory, read on the run's own trajectories
//! One flag is left of the momentum-scaling response: T's memory not shorter than the run (FOLDTEMP §R2.3, "缩放响应在
//! 运行末仍可读出"): whether the run's end still remembers its initial state. It is read on the product's own
//! trajectories, no trajectory more (solve.rs `memory_at_the_end`; until the coordinator's item 7 of 2026-11-22 two more
//! runs of the pilot's atoms at 1 ± η did it, 4–52 % of the chosen W):
//! - Any perturbation of the initial state not orthogonal to the slow modes asks it. A perturbation p_ε of the draws'
//!   density moves a reading by R_O(t) = d/dε ∫ O(t; x₀) p_ε(x₀) dx₀ = E[O(t) s(x₀)] (the likelihood-ratio identity), s =
//!   ∂_ε log p_ε at ε = 0 the draw's score, the dynamics after t = 0 unchanged; E[s] = 0, so R_O = Cov(O(t), s), and
//!   for an estimate that is a smooth function of the run's sums -- T, a U-statistic of the held atoms' κ and κ² over
//!   their number -- R is its derivative with every trajectory reweighed by 1 + εs.
//! - The perturbation is the initial kinetic energy's exponential tilt, p_ε ∝ p e^{−εK} normalized, K the classical
//!   draw's kinetic energy in k_BT (ensemble.rs, # Its draw; `Ensemble::draw_kinetic`): s = ⟨K⟩ − K. Like the momenta's
//!   scaling it changes the initial state's temperature, and overlaps the slow modes as much; unlike it, its score is
//!   bounded -- the draw's density (e^{y−K} − 1)₊ holds 0 ≤ K ≤ y, y the depth of the place, at most the well's -- and its
//!   variance finite. (The momenta's scaling, by the same identity, had s = 2K e^{y−K}/(e^{y−K} − 1) − 3: it grows as
//!   2y/(y − K) at the escape energy, where the density vanishes linearly, and its second moment diverges as the
//!   logarithm; the groups' jackknife of such a heavy tail reads too small an error. The coordinator's item 8.) The
//!   Wigner map (# The Wigner factor) is a fixed map of the classical draw, so the mapped draws are tilted by the same
//!   factor, and a mixture's component is the trajectory's internal state: the tilt is of each component's own density.
//! - ⟨K⟩ is the run's own mean K̄ over its trajectories (failed or not, at t = 0), so Σ s = 0 over the run: its sums'
//!   counts are not reweighed to the first order. A ratio of reweighed sums is unmoved by a constant added to s, so K̄'s
//!   own sampling error enters only through the U-statistic's (n − 1) -- a relative 1/n of the response --, and the
//!   sample covariance's bias is the same 1/n against the jackknife's 1/√n: both negligible.
//! - The question's scale (`decide`): the momenta's scaling by a unit moved T(0) by 2T(0), and the flag asked whether
//!   that much at t = 0 still moves the end's T by more than its allowance. The tilt's response is put on that scale by
//!   its own start: the end's memory m = 2T(0) R_T(end)/R_T(0), T(0) the run's, its error the groups' jackknife of the
//!   ratio itself. Read there (|m| > z·se), flagged; within the allowance at its upper end, or not told, no flag.
//!
use crate::field::Light;
use crate::linalg::{larger, symmetric3_eigen, C64, Mat, V3};
use crate::model::rs::{self, decay, ModelAt};
use crate::model::selected::Selected;
use crate::paths::{self, Form, Scheme, Series, Walk, Window};
use crate::request::{GROUND, STATES};
use crate::scene::{Components, Scene};
use crate::step::Dynamics;
use crate::units::{CONFIDENCE, POPULATION_TARGET, STEP_SHARE};
use rayon::prelude::*;
use std::time::Instant;

/// The points the structural amplitudes are read at: the centre (0), the edge (1), the middle (2) (module
/// documentation, # The candidates).
pub const POINTS: usize = 3;

/// The paths a single state's addition is judged on for its nomination (module documentation, # The candidates): two,
/// the fewest whose mean is one over paths -- the nomination reads point estimates (the rates and the forces add over
/// single states to 0.1–1 percentage points on two paths, Study A §2.2), and every nominee is judged again on every path,
/// every single that meets every look on these alone.
pub const NOMINATION_WALKS: usize = 2;

/// The pilot's paths a candidate is first judged on (BRIEF_P3_SELECT item 2: P1-PATHS's path-to-path spread is at most
/// 2e-3, one or two paths reach a quarter of the target, four cover the start's draws), doubled where the judgment asks
/// (module documentation, # How many paths).
pub const WALKS: usize = 4;

/// The paths' scheme's tolerance (module documentation, # The paths): the largest error of a displayed row the
/// reference may take from the scheme's setting on the calibration's piece -- s⁴ of the population target, as the
/// numerical tolerances inside a step take s⁴ of theirs (units.rs `QUADRATURE`): 2.0e-3 at p = 2.
pub const TOLERANCE: f64 = STEP_SHARE * STEP_SHARE * STEP_SHARE * STEP_SHARE * POPULATION_TARGET;

/// A setting of the paths' scheme (module documentation, # The paths): the substeps a turn of the fastest coherence over
/// a stretch's opening, the substeps a µs after it (paths.rs, # The start), and the pieces a step.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Setting {
    pub turns: f64,
    pub rate: f64,
    pub pieces: usize,
    /// The opening's cutoff (rad/µs): the start's modes up to it resolved, the faster ones booked by their bound
    /// (paths.rs, # The start).
    pub cutoff: f64,
}

/// The knobs of a setting (`Setting`): the opening's substeps, the run's substeps, the pieces, the opening's cutoff --
/// the last one's part not read from a run but the bound of what the opening books (`Settled`).
pub const KNOBS: usize = 4;

/// The knobs a setting's refinements run (`refinements`): every knob but the cutoff.
const RUN_KNOBS: usize = 3;

impl Setting {
    /// The setting with its knob `k` (`KNOBS`) doubled.
    fn doubled(self, k: usize) -> Setting {
        let mut s = self;
        match k {
            0 => s.turns *= 2.0,
            1 => s.rate *= 2.0,
            2 => s.pieces *= 2,
            _ => s.cutoff *= 2.0,
        }
        s
    }

    /// Whether its knob `k` is below `FINEST`'s.
    fn open(&self, k: usize) -> bool {
        match k {
            0 => self.turns < FINEST.turns,
            1 => self.rate < FINEST.rate,
            2 => self.pieces < FINEST.pieces,
            _ => self.cutoff < FINEST.cutoff,
        }
    }
}

/// The setting the paths' scheme starts from (module documentation, # The paths): a substep a turn of the fastest
/// coherence over the opening (its Nyquist rate, paths.rs # The start), 8 a µs after it, one piece a step, the opening
/// resolving the start's modes up to what those 8 a µs resolve (2π × 8 rad/µs) -- each knob doubled from there as its
/// error asks.
pub const COARSEST: Setting = Setting { turns: 1.0, rate: 8.0, pieces: 1, cutoff: std::f64::consts::TAU * 8.0 };

/// The finest setting (module documentation, # The paths): 64 substeps a turn, 4096 a µs (32 times P1-PATHS's 128), 64
/// pieces a step -- a bound on the work, never a measure of accuracy: a setting settled there short of `TOLERANCE` is
/// said ("到顶未收敛") and its reference's error parts booked whole in every candidate's A (`Settled::unmet`).
pub const FINEST: Setting = Setting { turns: 64.0, rate: 4096.0, pieces: 64, cutoff: f64::INFINITY };

/// A refinement's change read as the coarser setting's error (module documentation, # The paths): Richardson's
/// 2^p/(2^p − 1) at the scheme's lowest order p = 2 -- the amplitudes' linear interpolation at one piece a step, the
/// recycling's amplitudes at the substep's middle (the cubic interpolation and the exponential Runge–Kutta err at order
/// 3 or more).
const RICHARDSON: f64 = 4.0 / 3.0;

/// The paths' scheme at a setting, read at every `read`-th knot.
fn scheme(setting: Setting, read: usize) -> Scheme {
    Scheme::Exponential { turns: setting.turns, rate: setting.rate, pieces: setting.pieces, cutoff: setting.cutoff, read }
}

/// The setting ladder's first stage (µs; module documentation, # The paths): a stretch's first piece, where the
/// readings change fastest (P1-PATHS: one path, 10 µs), the ladder climbed there before it goes on over the whole
/// stretch, which decides -- its length moves the ladder's cost alone, not the setting it ends at.
pub const CALIBRATION: f64 = 10.0;

/// The measurements' trace (SELCOST): `SELCOST_TRACE` set, each job's work said.
#[cfg(test)]
pub(crate) fn traced() -> bool {
    std::env::var_os("SELCOST_TRACE").is_some()
}

/// One job's work in the trace (`traced`): its wall seconds and its series' counts.
#[cfg(test)]
pub(crate) fn trace_job(what: &str, s: &Series, wall: f64) {
    if traced() {
        let c = &s.counts;
        eprintln!(
            "    [trace] {what}: {:.1} µs, wall {wall:.2} s; thread s {:.2} (freeze {:.2} + propagation {:.2} + readouts {:.2}); {} nodes, {} substeps, {} products, {} readouts",
            c.time, c.seconds(), c.node_seconds, c.propagation_seconds, c.readout_seconds, c.nodes, c.segments, c.products, c.readouts
        );
    }
}

/// A run's states: P, the structural chain it is one of (module documentation, # The candidates), the states'
/// structural amplitudes, the chain's time, and what the choice made of the chain where it was made (`Choice`: after the
/// pilot, solve.rs # The selection).
#[derive(Clone, Debug)]
pub struct Selection {
    /// P (indices into the 32).
    pub keep: Vec<usize>,
    /// The chain from the full-ground model G (first) up to the 32 states (last), each sorted.
    pub chain: Vec<Vec<usize>>,
    /// Per state its structural amplitude (zero on the ground states).
    pub amplitudes: [f64; STATES],
    pub seconds: f64,
    pub choice: Option<std::sync::Arc<Choice>>,
}

impl Selection {
    /// The full-ground model's states: the chain's first, the model the scout and the pilot run.
    pub fn ground(&self) -> &[usize] {
        &self.chain[0]
    }

    /// What the report says beside the numbers (Chinese): the chain, and the choice where it was made.
    pub fn notes(&self) -> Vec<String> {
        let mut notes = vec![format!("结构链 {} 档：{}", self.chain.len(), self.chain.iter().map(|c| c.len().to_string()).collect::<Vec<_>>().join(" → "))];
        if let Some(choice) = &self.choice {
            notes.extend(choice.report.iter().cloned());
        }
        notes
    }
}

/// The selection's chain for a scene (module documentation, # The candidates): its states' structural amplitudes and the
/// chain; P the full-ground model, until the choice (solve.rs, # The selection) takes another.
pub fn choose(scene: &Scene) -> Result<Selection, String> {
    let clock = Instant::now();
    let amplitudes = amplitudes(scene);
    let chain = chain(&amplitudes);
    Ok(Selection { keep: chain[0].clone(), chain, amplitudes, seconds: clock.elapsed().as_secs_f64(), choice: None })
}

/// The chain (module documentation, # The candidates) by the states' structural amplitudes: from the ground states up,
/// the excited states added in decreasing amplitude, those equal to the rounding at once, to all 32.
pub fn chain(amplitudes: &[f64; STATES]) -> Vec<Vec<usize>> {
    let mut excited: Vec<usize> = (GROUND..STATES).collect();
    excited.sort_by(|&x, &y| amplitudes[y].total_cmp(&amplitudes[x]).then(x.cmp(&y)));
    let mut keep: Vec<usize> = (0..GROUND).collect();
    let mut chain = vec![keep.clone()];
    let tie = crate::units::rounding(STATES);
    let mut i = 0;
    while i < excited.len() {
        let top = amplitudes[excited[i]];
        let mut j = i + 1;
        while j < excited.len() && (top - amplitudes[excited[j]]).abs() <= tie * larger(top.abs(), amplitudes[excited[j]].abs()) {
            j += 1;
        }
        keep.extend(&excited[i..j]);
        keep.sort_unstable();
        chain.push(keep.clone());
        i = j;
    }
    chain
}

/// The structural points' components (module documentation, # The candidates), the atom at rest (no Doppler shift): the
/// centre, the tweezer full, and the edge, dark, both at the focus' place; the middle at `middle`, every light there as
/// it is (the tweezer at its half intensity).
pub fn points(scene: &Scene) -> [Components; POINTS] {
    let zero = vec![0.0; scene.request.beams.len()];
    [scene.components([0.0; 3], &zero, 1.0), scene.components([0.0; 3], &zero, 0.0), scene.components(middle(scene), &zero, 1.0)]
}

/// The middle point's place (µm from the focus): n̂ the top eigenvector of M = Σ_w Re(∇E_w ∇E_w†) at the focus over the
/// light other than the tweezer's -- the direction its pattern changes fastest (Rayleigh–Ritz; Horn & Johnson, Thm
/// 4.2.2), the tweezer's axis where M = 0 --, its sign the one that makes its largest component positive; along it the
/// distance where the tweezer's paraxial Gaussian intensity is half its peak, its one root found by bisection to the
/// rounding (the intensity falls strictly along every ray from the focus).
pub fn middle(scene: &Scene) -> V3 {
    let light = Light::new(&scene.request);
    let mut m = [[0.0_f64; 3]; 3];
    for frequency in light.frequencies.iter().enumerate().filter(|&(w, _)| w != light.trap).map(|(_, f)| f) {
        let mut g = [[C64::ZERO; 3]; 3];
        for (&l, field) in frequency.beams.iter().zip(&frequency.fields) {
            let a = light.beams[l].amplitude_to([0.0; 3], 1).gradient;
            for (gi, ai) in g.iter_mut().zip(a) {
                for (x, e) in gi.iter_mut().zip(field) {
                    *x += ai * *e;
                }
            }
        }
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += (0..3).map(|q| (g[i][q] * g[j][q].conj()).re).sum::<f64>();
            }
        }
    }
    let tweezer = &light.beams[light.frequencies[light.trap].beams[0]];
    let (values, vectors) = symmetric3_eigen(&m);
    let n = if values[2] > 0.0 {
        let v = vectors[2];
        let big = (0..3).fold(0, |b, i| if v[i].abs() > v[b].abs() { i } else { b });
        if v[big] < 0.0 { v.map(|x| -x) } else { v }
    } else {
        tweezer.direction
    };
    let below = |t: f64| tweezer.amplitude_to(n.map(|x| t * x), 0).value.norm_sqr() - 0.5;
    let (mut lo, mut hi) = (0.0, tweezer.rayleigh().expect("the tweezer is a Gaussian beam (Request::validate)"));
    while below(hi) >= 0.0 {
        (lo, hi) = (hi, 2.0 * hi);
    }
    loop {
        let mid = 0.5 * (lo + hi);
        if mid <= lo || mid >= hi {
            break;
        }
        if below(mid) >= 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let t = if below(lo).abs() <= below(hi).abs() { lo } else { hi };
    n.map(|x| t * x)
}

/// Per state its importance at the structural points (module documentation, # The candidates), before any atom is
/// seen: the largest over the three of its first-order excitation by the light other than the tweezer's from every
/// ground state alike (`excitation`, p_g = 1/8); zero on the ground states. The chain the pilot's choice is made on takes
/// the pilot's states instead (`importance`).
pub fn amplitudes(scene: &Scene) -> [f64; STATES] {
    let tw = scene.request.trap_beam;
    let even = [1.0 / GROUND as f64; GROUND];
    let mut score = [0.0_f64; STATES];
    for point in points(scene) {
        let reference = rs::reference(scene, &point);
        for (s, a) in score.iter_mut().zip(excitation(scene, &reference, &point, &|l| l != tw, &even)) {
            *s = larger(*s, a);
        }
    }
    score
}

/// The first-order excitation of each excited state at a point by the light `beams` takes, from the ground populations
/// `p`: Σ_g p_g Σ_± |V^±_eg|²/|E_g ± f − E_e + iΓ_e/2|² -- V the coupling of a frequency's beams together (its field:
/// the beams of one frequency index summed, so that beams cancelling at the point cancel), at f and its conjugate at −f;
/// E the reference's static energies at the point (the tweezer's shift in), Γ its decay; a beam with no component at the
/// point couples nothing there --: the population the state would hold to the first order, the fold's leading term
/// weighed by the ground states it takes its atoms from.
pub fn excitation(scene: &Scene, reference: &ModelAt, point: &Components, beams: &dyn Fn(usize) -> bool, p: &[f64; GROUND]) -> [f64; STATES] {
    let request = &scene.request;
    let still = &reference.still().value;
    let gamma = decay(&reference.jumps, STATES).0;
    let mut out = [0.0_f64; STATES];
    let mut seen: Vec<&Vec<i32>> = Vec::new();
    for (l, beam) in request.beams.iter().enumerate().filter(|&(l, _)| beams(l)) {
        if seen.contains(&&beam.frequency_index) {
            continue;
        }
        seen.push(&beam.frequency_index);
        let mut v = Mat::zeros(STATES, STATES);
        for m in (0..request.beams.len()).filter(|&m| beams(m) && request.beams[m].frequency_index == beam.frequency_index) {
            let key = scene.unit(m);
            if let Some(c) = point.parts.iter().find(|c| c.key == key) {
                v.add_scaled(C64::ONE, &c.value);
            }
        }
        let f = scene.generators[l].frequency;
        for (x, f) in [(v.clone(), f), (v.adjoint(), -f)] {
            for e in GROUND..STATES {
                for g in 0..GROUND {
                    if x[(e, g)] != C64::ZERO {
                        let away = C64::new(still[(g, g)].re + f - still[(e, e)].re, 0.5 * gamma[(e, e)].re);
                        out[e] += p[g] * x[(e, g)].norm_sqr() / away.norm_sqr();
                    }
                }
            }
        }
    }
    out
}

/// Each excited state's importance on the chain (module documentation, # The candidates): the largest, over the places
/// `places` an atom was seen at with its ground populations there (the pilot's states: an atom at rest there, the
/// tweezer's light as it is), of its first-order excitation by the light other than the tweezer's (`excitation`). Where
/// the atoms go and from the states they are in: a repumper resonant on F = 1 → F′ = 2 gives F′ = 2 the larger
/// amplitude (PGC3D: 0.27 against 0.06–0.11 for F′ = 3), but the atoms are in F = 2 (90 %), whose F′ = 3 holds the
/// excitation; a lattice's polarization reaches states between the structural points (R4's F′ = 3 stretched states).
pub fn importance(scene: &Scene, places: &[(V3, [f64; GROUND])]) -> [f64; STATES] {
    let tw = scene.request.trap_beam;
    let zero = vec![0.0; scene.request.beams.len()];
    places
        .par_iter()
        .map(|(r, p)| {
            let point = scene.components(*r, &zero, 1.0);
            excitation(scene, &rs::reference(scene, &point), &point, &|l| l != tw, p)
        })
        .reduce(|| [0.0; STATES], |a, b| std::array::from_fn(|s| larger(a[s], b[s])))
}

// ---------------------------------------------------------------------------------------------------------------------
// The paths and the candidates' judgment.

/// The tweezer's intensity at a place, relative to its peak (s = |a_tw(r)|²).
pub fn intensity(light: &Light, r: V3) -> f64 {
    light.beams[light.frequencies[light.trap].beams[0]].amplitude_to(r, 0).value.norm_sqr()
}

//// What every candidate is judged against (module documentation, # The paths): the walks cut at L, their windows, the
/// reference's form, the scheme's setting (`settle`), the reference's series along the walks at it, the calibration --
/// the walk it was read along (the longest), its length and the reference's series there at the setting's runs --, the
/// least tweezer intensity the walks reached, the memory time, and the work it took.
pub struct Paths<'a> {
    pub walks: Vec<Walk>,
    pub length: f64,
    pub memory: f64,
    pub windows: Vec<(f64, f64)>,
    pub reference: Form<'a>,
    pub settled: Settled,
    series: Vec<Series>,
    calibration: (usize, f64, Vec<Series>),
    /// The late segment (module documentation, # The paths): its start, its windows and the reference's series there;
    /// none where the run is no longer than twice L.
    late: Option<(f64, Vec<(f64, f64)>, Vec<Series>)>,
    pub least: f64,
    /// The largest bound the reference's openings book over its stretches (paths.rs, # The start).
    pub booked: f64,
    /// The reference's largest trace defect over its stretches at the setting (paths.rs `Series::trace_defect`).
    pub trace: f64,
    /// The landing check (module documentation, # The motion): per stretch -- the walks' starts, then their late
    /// segments -- its landings with the reference's excess photons and impulse; and the check's span and the fastest
    /// excited decay (paths.rs `Form::optical`; none of a reference without excited states).
    pub landings: Vec<Vec<Landing>>,
    pub optical: Option<(f64, f64)>,
    /// The reference's counted work along its paths (paths.rs `Counts::work`; # How many paths).
    pub work: f64,
    pub seconds: f64,
    pub products: u64,
}

/// A landing of the landing check (module documentation, # The motion): its time, the reference's landed ground state
/// (by the request's ground states, its trace one), the reference's excess photons n_ex = ∫(λ(τ|L) − λ̄)dτ and impulse
/// ∫(F(τ|L) − F̄)dτ (1/µm) from it (paths.rs `Form::landing`), and the reference's force correlation's diffusion at
/// its own state there (paths.rs `Form::diffusion`; none where its frozen generator has no inverse there).
#[derive(Clone, Debug)]
pub struct Landing {
    pub t: f64,
    pub landed: [[C64; GROUND]; GROUND],
    pub photons: f64,
    pub impulse: V3,
    pub diffusion: Option<V3>,
    /// The reference's counted work on it (`paths::Counts::work`): the landing's stretch and the diffusion's solve.
    pub work: f64,
}

/// The paths' scheme as settled (`settle`; module documentation, # The paths): its setting, the calibration's opening (µs), the
/// reference's error parts along the calibration's stretch (per knob, `KNOBS`), the calibration's runs -- the setting
/// first, its refinements after -- and per knob the two of them its part is read between (`refinements`), the settings
/// run, and their work and time.
#[derive(Clone, Debug)]
pub struct Settled {
    pub setting: Setting,
    pub opening: f64,
    pub errors: [f64; KNOBS],
    /// The setting's trace defect along the stretch (paths.rs `Series::trace_defect`, the models' largest): a part of
    /// its error beside the knobs' (module documentation, # The paths).
    pub trace: f64,
    pub runs: Vec<Setting>,
    pub pairs: [(usize, usize); KNOBS],
    pub tried: usize,
    pub products: u64,
    /// The ladder's runs' counted work (`paths::Counts::work`).
    pub work: f64,
    pub seconds: f64,
}

impl Settled {
    /// Where the setting stopped at `FINEST` with its parts still beyond `TOLERANCE` (the ladder's top, not
    /// converged): their sum, booked in every candidate's A beside its own calibration (`judge`, `Paths::itself`); none
    /// where the parts met the tolerance.
    pub fn unmet(&self) -> Option<f64> {
        let sum: f64 = self.errors.iter().sum::<f64>();
        // The trace's part decides with the knobs' and is booked on its own, over every path (`Paths::trace`).
        (sum + self.trace > TOLERANCE).then_some(sum)
    }

    /// The scheme at the setting, read at every `read`-th knot.
    pub fn scheme(&self, read: usize) -> Scheme {
        scheme(self.setting, read)
    }

    /// The ladder's second stage after its first (`settle`): the second's setting and parts, the two's work together.
    fn after(self, first: &Settled) -> Settled {
        Settled { tried: self.tried + first.tried, products: self.products + first.products, work: self.work + first.work, seconds: self.seconds + first.seconds, ..self }
    }
}

/// A setting's refinements (module documentation, # The paths): its runs -- the setting, and each knob doubled, raised
/// to one substep a piece (`own`) --, and per knob the two runs its part is read between: the substeps' doubled at the
/// setting's pieces; the pieces' doubled at one rate, the doubled pieces' own -- where twice the pieces raise the rate
/// to that floor, against the setting's pieces at the raised rate, one run more where that is none of the others. (Read against the setting itself, the pieces' part held the substeps' wherever the floor doubled them too:
/// TET's ladder doubled the pieces to 16 while its substeps erred, the pieces' own part at 4 a step 1.24e-3, SELCOST.)
fn refinements(setting: Setting, own: &dyn Fn(Setting) -> Setting) -> (Vec<Setting>, [(usize, usize); KNOBS]) {
    let mut runs: Vec<Setting> = std::iter::once(setting).chain((0..RUN_KNOBS).map(|k| own(setting.doubled(k)))).collect();
    let pieces = RUN_KNOBS;
    let partner = Setting { rate: runs[pieces].rate, ..setting };
    let at = runs.iter().position(|&s| s == partner).unwrap_or_else(|| {
        runs.push(partner);
        runs.len() - 1
    });
    // The cutoff's part is the setting's own bound (no pair: `settle`).
    (runs, std::array::from_fn(|k| if k + 1 == pieces { (at, pieces) } else if k < RUN_KNOBS { (0, k + 1) } else { (0, 0) }))
}


/// `settle`'s ladder along the walk's stretch of `piece` µs from `begin` (from the walk's own start state there) from the
/// setting `from` (raised to its own rate floor).
#[allow(clippy::too_many_arguments)]
fn ladder(forms: &[&Form], walk: &Walk, begin: f64, piece: f64, step: f64, starts: &[(&Walk, f64)], from: Setting) -> Result<(Settled, Vec<Vec<Series>>), String> {
    let clock = Instant::now();
    // A family of settings' runs, each model's (paths.rs `Form::along_family`: each node made once for them all): per
    // setting the models' series.
    let family = |settings: &[Setting]| -> Result<Vec<Vec<Series>>, String> {
        let schemes: Vec<Scheme> = settings.iter().map(|&s| scheme(s, 1)).collect();
        let made: Vec<Vec<Series>> = forms.par_iter().map(|f| f.along_family(walk, &walk.start, begin, begin + piece, &schemes)).collect::<Result<_, _>>()?;
        let mut made: Vec<std::vec::IntoIter<Series>> = made.into_iter().map(Vec::into_iter).collect();
        Ok(settings.iter().map(|_| made.iter_mut().map(|x| x.next().expect("a member's series")).collect()).collect())
    };
    let own = |s: Setting| Setting { rate: larger(s.rate, s.pieces as f64 / step), ..s };
    // The cutoff's part: the largest over the stretches the setting takes (`starts`: each from its walk's own start) of
    // what their openings book, the models' together.
    let booked = |s: Setting| -> Result<f64, String> {
        let each: Vec<f64> = starts.par_iter().map(|&(w, from)| forms.iter().map(|f| f.booked_at(w, &w.start, from, s.pieces, s.cutoff)).sum::<Result<f64, String>>()).collect::<Result<_, _>>()?;
        Ok(each.into_iter().fold(0.0, larger))
    };
    let mut setting = own(from);
    // The setting's own runs where they are in hand (the refinement it was), else made in its refinements' family.
    let mut at: Option<Vec<Series>> = None;
    let work = |x: &[Series]| x.iter().map(|s| s.counts.products).sum::<u64>();
    let (mut tried, mut products, mut counted) = (0, 0, 0.0);
    loop {
        let (runs, pairs) = refinements(setting, &own);
        #[cfg(test)]
        let rung = Instant::now();
        let made = family(if at.is_some() { &runs[1..] } else { &runs[..] })?;
        #[cfg(test)]
        if traced() {
            let sum = |f: &dyn Fn(&crate::paths::Counts) -> f64| made.iter().flatten().map(|s| f(&s.counts)).sum::<f64>();
            eprintln!("    [trace] family of {}: wall {:.2} s, thread s {:.2} (freeze {:.2} + propagation {:.2} + readouts {:.2})", made.len(), rung.elapsed().as_secs_f64(), sum(&|c| c.seconds()), sum(&|c| c.node_seconds), sum(&|c| c.propagation_seconds), sum(&|c| c.readout_seconds));
        }
        tried += made.len();
        products += made.iter().map(|s| work(s)).sum::<u64>();
        counted += made.iter().flatten().map(|s| s.counts.work).sum::<f64>();
        let series: Vec<Vec<Series>> = at.take().into_iter().chain(made).collect();
        let mut errors = pairs.map(|(i, j)| RICHARDSON * change(&series[i], &series[j]));
        // The cutoff's part: what the setting's openings book.
        errors[RUN_KNOBS] = booked(setting)?;
        // The trace's part: the setting's own run's defect, the models' largest (paths.rs `Series::trace_defect`: a strict
        // lower bound of its error, which the rows' normalization hides from the knobs' parts).
        let defect = |i: usize| series[i].iter().map(Series::trace_defect).fold(0.0, larger);
        let trace = defect(0);
        #[cfg(test)]
        if traced() {
            eprintln!("    [trace] settle {setting:?}: parts {:?}, trace {trace:.2e}", errors.map(|e| format!("{e:.2e}")));
        }
        let open: [bool; KNOBS] = std::array::from_fn(|k| setting.open(k));
        if errors.iter().sum::<f64>() + trace <= TOLERANCE || !open.iter().any(|&o| o) {
            let opening = series[0][0].counts.opening;
            return Ok((Settled { setting, opening, errors, trace, runs, pairs, tried, products, work: counted, seconds: clock.elapsed().as_secs_f64() }, series));
        }
        // The knob of the largest part; where the trace's is the largest, the run's knob whose doubling lowers it most.
        let largest = (0..KNOBS).filter(|&k| open[k]).max_by(|&a, &b| errors[a].total_cmp(&errors[b])).expect("an open knob");
        let k = if trace > errors[largest] { (0..RUN_KNOBS).filter(|&k| open[k]).min_by(|&a, &b| defect(1 + a).total_cmp(&defect(1 + b))).unwrap_or(largest) } else { largest };
        if k < RUN_KNOBS {
            setting = runs[1 + k];
            at = series.into_iter().nth(1 + k);
        } else {
            // The cutoff's part is the setting's own bound: raised while it stays the largest, and the setting run once.
            let others = errors[..RUN_KNOBS].iter().copied().fold(0.0, larger);
            loop {
                setting = setting.doubled(k);
                if !(setting.open(k) && booked(setting)? > others) {
                    break;
                }
            }
            // Its own runs made with its refinements, the next family.
        }
    }
}

/// The largest change between two runs, along one path at the times both read: of a displayed row of one model (one
/// series each), or of a row's difference of the second model to the first (two).
fn change(a: &[Series], b: &[Series]) -> f64 {
    let at = |s: &Series, t: f64| s.readings.iter().find(|y| y.t == t).map(|y| y.rows);
    let mut e = 0.0;
    for x in &a[0].readings {
        let Some(y) = at(&b[0], x.t) else { continue };
        let (u, v) = match (a.get(1), b.get(1)) {
            (Some(p), Some(q)) => match (at(p, x.t), at(q, x.t)) {
                (Some(u), Some(v)) => (u, v),
                _ => continue,
            },
            _ => ([0.0; STATES], [0.0; STATES]),
        };
        for k in 0..STATES {
            e = larger(e, ((u[k] - x.rows[k]) - (v[k] - y[k])).abs());
        }
    }
    e
}


impl<'a> Paths<'a> {
    /// The reference's part (`Paths`): the walks `walks` (the pilot's, from their own starts, through the run) at their
    /// start, [0, L], L = min(`run`, 5 τ_mem) -- `memory` the memory time (`paths::memory`, the longest over the walks'
    /// starts) --, and at the run's end, [run − L, run], each from the walk's own start state (the late segment: its
    /// last window has forgotten that start), where the run is longer than twice L; the windows of their step `step`;
    /// the scheme's setting along the longest walk's [0, L] (`settle`; the first such where several are), and the
    /// reference along the walks at it -- the other walks' on the pool beside the setting's second stage, at the first
    /// stage's setting, and again where the second stage moved it.
    pub fn new(reference: &'a Selected, at: V3, walks: Vec<Walk>, memory: f64, run: f64, step: f64, light: &Light) -> Result<Paths<'a>, String> {
        let clock = Instant::now();
        let form = Form::new(reference, at);
        let length = run.min(5.0 * memory);
        let windows = paths::windows(length, memory, step);
        let late_from = (run > 2.0 * length).then_some(run - length);
        // The stretches the setting takes: every walk that took a step from its start, and from the late segment's where
        // there is one.
        let starts: Vec<(&Walk, f64)> = walks.iter().filter(|w| w.end() > 0.0).map(|w| (w, 0.0)).chain(late_from.iter().flat_map(|&f| walks.iter().filter(move |w| f < w.end()).map(move |w| (w, f)))).collect();
        // The calibration's walk: the longest (the first of them), along its whole [0, L].
        let reach = |w: &Walk| length.min(w.end());
        let longest = (0..walks.len()).fold(0, |best, k| if reach(&walks[k]) > reach(&walks[best]) { k } else { best });
        let stretch = reach(&walks[longest]);
        if !(stretch > 0.0) {
            return Err("no path of the pilot's goes past its start (every one's atom lost or its trajectory ended at its first step)".into());
        }
        #[cfg(test)]
        if traced() {
            eprintln!("    [trace] reference form: {}; beats H {:?}, R {:?}", form.shape(), form.beat_values().0, form.beat_values().1);
            eprintln!("    [trace] walks: {:?} ends, step {step}, ends at {:?}", walks.iter().map(|w| w.ends.len()).collect::<Vec<_>>(), walks.iter().map(|w| w.end()).collect::<Vec<_>>());
        }
        #[cfg(test)]
        let pool = Instant::now();
        #[cfg(test)]
        let _ = crate::paths::freeze_parts();
        // One job a walk's start and its end, the calibration's walk's too (the same run as the setting's own there, its
        // landing states kept); on the pool.
        let n = walks.len();
        let jobs: Vec<usize> = (0..n * (1 + usize::from(late_from.is_some()))).collect();
        let stretch_of = |j: usize| -> (usize, f64, f64) { if j < n { (j, 0.0, reach(&walks[j])) } else { (j - n, late_from.expect("a late segment"), run.min(walks[j - n].end())) } };
        // The landing check's times (module documentation, # The motion): per stretch one landing a window, at its last
        // reading that leaves the check's span inside the stretch.
        let optical = form.optical(at);
        let keeps: Vec<Vec<f64>> = jobs
            .iter()
            .map(|&j| {
                let (w, from, to) = stretch_of(j);
                let Some((span, _)) = optical.filter(|_| to > from) else { return vec![] };
                let reads: Vec<f64> = walks[w].ends.iter().map(|e| e[0]).filter(|&t| t > from && t < to).chain([to]).collect();
                let mut at: Vec<f64> = windows.iter().filter_map(|&(a, b)| reads.iter().copied().filter(|&t| t > from + a && t <= from + b && t + span <= to).last()).collect();
                at.dedup();
                at
            })
            .collect();
        let others = |setting: Setting| -> Vec<Result<Series, String>> {
            let scheme = scheme(setting, 1);
            jobs.par_iter()
                .map(|&j| {
                    #[cfg(test)]
                    let clock = Instant::now();
                    let out = if j < n {
                        let w = &walks[j];
                        // A walk that took no step (its trajectory ended at its first) reads nothing.
                        if reach(w) > 0.0 { form.along_keeping(w, &w.start, 0.0, reach(w), scheme, &keeps[j]) } else { Ok(Series::default()) }
                    } else {
                        // A walk that ended before the late segment (its atom lost) reads nothing there.
                        let (w, from) = (&walks[j - n], late_from.expect("a late segment"));
                        if from < w.end() { form.along_keeping(w, &w.start, from, run.min(w.end()), scheme, &keeps[j]) } else { Ok(Series::default()) }
                    };
                    #[cfg(test)]
                    if let Ok(s) = &out {
                        trace_job(&format!("reference walk {} {}", j % n, if j < n { "start" } else { "late" }), s, clock.elapsed().as_secs_f64());
                    }
                    out
                })
                .collect()
        };
        // The setting (`settle`): its first stage, then its second beside the other walks at the first stage's setting.
        let piece = CALIBRATION.min(stretch);
        let (first, at) = ladder(&[&form], &walks[longest], 0.0, piece, step, &starts, COARSEST)?;
        let (mut settled, calibrated, done) = if stretch > piece {
            let (whole, done) = rayon::join(|| ladder(&[&form], &walks[longest], 0.0, stretch, step, &starts, first.setting), || others(first.setting));
            let (whole, at) = whole?;
            let done = if whole.setting == first.setting { done } else { others(whole.setting) };
            (whole.after(&first), at, done)
        } else {
            let done = others(first.setting);
            (first, at, done)
        };
        let mut calibrated: Vec<Series> = calibrated.into_iter().map(|mut x| x.swap_remove(0)).collect();
        #[cfg(test)]
        if traced() {
            eprintln!("    [trace] reference jobs: {:.2} s wall on {} threads; freezes' parts (nodes, amplitudes, eigen, weights, R and ΔH) {:.2?} s", pool.elapsed().as_secs_f64(), rayon::current_num_threads(), crate::paths::freeze_parts());
        }
        let mut done: Vec<Series> = done.into_iter().collect::<Result<_, _>>()?;
        // The trace on every path, not the calibration's alone (module documentation, # The paths): where some other
        // stretch's defect is beyond the tolerance at the setting, the setting is not converged there -- the ladder goes
        // on along that stretch from the setting, the calibration's walk is run again at the new setting's refinements and
        // every other stretch at it, until every stretch's defect is within the tolerance or the ladder stands.
        loop {
            let worst = jobs.iter().zip(&done).map(|(&j, s)| (j, s.trace_defect())).fold((longest, 0.0), |a, b| if b.1 > a.1 { b } else { a });
            if !(worst.1 > TOLERANCE) || worst.0 == longest {
                break;
            }
            let (w, from, to) = stretch_of(worst.0);
            let (again, _) = ladder(&[&form], &walks[w], from, to - from, step, &starts, settled.setting)?;
            if again.setting == settled.setting {
                break;
            }
            settled = again.after(&settled);
            let schemes: Vec<Scheme> = settled.runs.iter().map(|&s| scheme(s, 1)).collect();
            let (made, more) = rayon::join(|| form.along_family(&walks[longest], &walks[longest].start, 0.0, stretch, &schemes), || others(settled.setting));
            (calibrated, done) = (made?, more.into_iter().collect::<Result<_, _>>()?);
        }
        // The landing check of the reference (module documentation, # The motion): at each kept state the landing after a
        // jump, and the reference's excess photons and impulse from it.
        let knobs = (settled.setting.turns, settled.setting.rate, settled.setting.cutoff);
        let landings: Vec<Vec<Landing>> = match optical {
            None => vec![vec![]; jobs.len()],
            Some((span, decay)) => jobs
                .par_iter()
                .map(|&j| -> Result<Vec<Landing>, String> {
                    let (w, ..) = stretch_of(j);
                    done[j]
                        .kept
                        .iter()
                        .filter_map(|(t, v)| form.landed(&walks[w], *t, v).map(|(landed, _)| (*t, v, landed)))
                        .map(|(t, v, landed)| {
                            form.landing(&walks[w], t, &landed, span, decay, knobs, &done[j]).map(|(photons, impulse, spent)| {
                                let (diffusion, solved) = form.diffusion(&walks[w], t, v);
                                Landing { t, landed, photons, impulse, diffusion, work: spent + solved }
                            })
                        })
                        .collect()
                })
                .collect::<Result<_, _>>()?,
        };
        let mut done = done.into_iter();
        let series: Vec<Series> = (0..n).map(|_| done.next().expect("a walk")).collect();
        let late_series: Vec<Series> = done.collect();
        let late = late_from.map(|from| (from, windows.iter().map(|&(a, b)| (from + a, from + b)).collect(), late_series));
        let least = walks.iter().flat_map(|w| w.ends.iter().map(|e| intensity(light, [e[1], e[2], e[3]]))).fold(f64::INFINITY, crate::linalg::smaller);
        let products = series.iter().chain(late.iter().flat_map(|l| l.2.iter())).map(|s| s.counts.products).sum::<u64>() + settled.products;
        let booked = series.iter().chain(late.iter().flat_map(|l| l.2.iter())).map(|s| s.counts.booked).fold(0.0, larger);
        let trace = series.iter().chain(late.iter().flat_map(|l| l.2.iter())).map(Series::trace_defect).fold(0.0, larger);
        let work = series.iter().chain(late.iter().flat_map(|l| l.2.iter())).map(|s| s.counts.work).sum::<f64>() + landings.iter().flatten().map(|l| l.work).sum::<f64>() + settled.work;
        Ok(Paths { walks, length, memory, windows, reference: form, settled, series, calibration: (longest, stretch, calibrated), late, least, booked, trace, landings, optical, work, seconds: clock.elapsed().as_secs_f64(), products })
    }

    /// The calibration's stretch (µs): the longest walk's [0, L], where the setting was read.
    pub fn calibrated_over(&self) -> f64 {
        self.calibration.1
    }

    /// Where the late segment starts (µs), where there is one.
    pub fn late_from(&self) -> Option<f64> {
        self.late.as_ref().map(|l| l.0)
    }

    /// The reference judged against itself (`Judged`): no error at all -- the 32 states are the reference, and relative to
    /// itself a model errs by nothing. No windows (its direct part zero at every time, `direct_at`), no calibration: the
    /// paths' scheme errs where a candidate is compared with the reference (`Settled::unmet`, `Paths::booked`, booked in
    /// every other candidate's calibration, `judge`), not in the reference's own model. (Until 2026-11-22 it booked the
    /// scheme's unmet sum and the openings' bound as its own, and read its windows against themselves: no number where no
    /// path read a window -- R9_PUSH's, every atom pushed out by 45 µs.)
    pub fn itself(&self, keep: Vec<usize>) -> Judged {
        Judged { keep, windows: vec![], late: vec![], calibration: Calibration::default(), escape: None, statistics: None, stationary: None, seconds: 0.0, products: 0, work: 0.0 }
    }
}

/// The scheme's error on the fold's error (module documentation, # The paths): along the calibration's stretch the largest
/// change of a row's difference to the reference under each refinement of the setting, read as its error part
/// (`RICHARDSON`), the two parts added -- booked beside the direct part --, and of the total rate's relative difference
/// alike (said).
#[derive(Clone, Copy, Debug, Default)]
pub struct Calibration {
    pub rows: f64,
    pub rate: f64,
}



/// One candidate judged along the paths (`judge`): its states, its windows' errors, its calibration, its motion's
/// numbers, and the work and time its judgment took.
#[derive(Clone, Debug)]
pub struct Judged {
    pub keep: Vec<usize>,
    pub windows: Vec<Window>,
    /// The late segment's windows (their times the run's), none where it has none.
    pub late: Vec<Window>,
    pub calibration: Calibration,
    /// The loss's energy rate's relative difference x over the paths (module documentation, # The motion): per path its
    /// mean energy rate's difference over the energy its atom gained to its loss per µs (the lost paths', where at least
    /// two are; else every path's over its heating), with the paths' count; none where no path read.
    pub escape: Option<(paths::Mean, usize)>,
    /// The landing check's Mandel Q, its mean over the landings, the candidate's and the reference's (module
    /// documentation, # The motion); none where no landing was read.
    pub statistics: Option<[f64; 2]>,
    /// The energy rates T's error reads (`Stationary`); none where no path read past its memory time.
    pub stationary: Option<Stationary>,
    pub seconds: f64,
    pub products: u64,
    /// Its paths' counted work (paths.rs `Counts::work`): what more paths cost (# How many paths).
    pub work: f64,
}

/// A candidate model judged against the reference along the paths (`Judged`): its series along the walks at the paths'
/// setting and on the calibration's piece at the setting's two refinements, every job on the pool.
pub fn judge(paths: &Paths, model: &Selected, at: V3) -> Result<Judged, String> {
    judge_on(paths, model, at, paths.walks.len())
}

/// `judge` on the paths' first `walks` walks alone (the nomination's, # The candidates): the others' series none, read in
/// no window; the calibration where its walk is among them.
pub fn judge_on(paths: &Paths, model: &Selected, at: V3, walks: usize) -> Result<Judged, String> {
    let clock = Instant::now();
    let form = Form::new(model, at);
    let (longest, stretch, ref calibrated) = paths.calibration;
    let scheme = paths.settled.scheme(1);
    // The calibration's walk along its stretch: the setting's own run and its refinements, one family (the setting's own
    // the walk's series).
    let family: Vec<Scheme> = paths.settled.runs.iter().map(|&s| self::scheme(s, 1)).collect();
    let n = paths.walks.len();
    let run = paths.late.as_ref().map_or(0.0, |(from, ..)| from + paths.length);
    let jobs: Vec<usize> = (0..n * (1 + usize::from(paths.late.is_some()))).collect();
    // Each stretch's states kept at the reference's landings (the candidate's own force correlation there).
    let keep = |j: usize| -> Vec<f64> { paths.landings.get(j).map_or(vec![], |l| l.iter().map(|x| x.t).collect()) };
    let done: Vec<Result<Vec<Series>, String>> = jobs
        .par_iter()
        .map(|&j| {
            #[cfg(test)]
            let clock = Instant::now();
            let out = match j {
                j if j % n >= walks => Ok(vec![Series::default()]),
                j if j == longest => {
                    let w = &paths.walks[j];
                    form.along_family_keeping(w, &w.start, 0.0, stretch, &family, &keep(j))
                }
                j if j < n => {
                    let w = &paths.walks[j];
                    let reach = paths.length.min(w.end());
                    (if reach > 0.0 { form.along_keeping(w, &w.start, 0.0, reach, scheme, &keep(j)) } else { Ok(Series::default()) }).map(|s| vec![s])
                }
                j => {
                    let (w, from) = (&paths.walks[j - n], paths.late.as_ref().expect("a late segment").0);
                    (if from < w.end() { form.along_keeping(w, &w.start, from, run.min(w.end()), scheme, &keep(j)) } else { Ok(Series::default()) }).map(|s| vec![s])
                }
            };
            #[cfg(test)]
            if let Ok(made) = &out {
                for s in made {
                    trace_job(&format!("P{} job {j}", model.rs.keep.len()), s, clock.elapsed().as_secs_f64());
                }
            }
            out
        })
        .collect();
    let mut done = done.into_iter();
    let (mut finer, mut series) = (Vec::new(), Vec::with_capacity(n));
    for j in 0..n {
        let mut made = done.next().expect("a walk")?;
        if j == longest {
            finer = made.split_off(1);
        }
        series.push(made.swap_remove(0));
    }
    let late_series: Vec<Series> = done.map(|x| x.map(|mut x| x.swap_remove(0))).collect::<Result<_, _>>()?;
    let labels = &model.scene.request.states;
    let hm = model.hbar_over_mass();
    // The landing check (module documentation, # The motion): the candidate from each of the reference's landings -- its
    // photons' Mandel Q = 2 n_ex --, and its force correlation's diffusion at its own state there; the reference's beside
    // (paths.rs `Statistics`). Where either model's frozen generator has no inverse, neither's diffusion is read there.
    let knobs = (paths.settled.setting.turns, paths.settled.setting.rate, paths.settled.setting.cutoff);
    let own_kept = |j: usize| -> &Series { if j < n { &series[j] } else { &late_series[j - n] } };
    // The landing checks' counted work, summed on the pool (whole multiply-adds).
    let landing_work = std::sync::atomic::AtomicU64::new(0);
    let landed: Vec<Vec<paths::Statistics>> = match paths.optical {
        None => vec![vec![]; paths.landings.len()],
        Some((span, decay)) => paths
            .landings
            .par_iter()
            .enumerate()
            .map(|(j, list)| -> Result<Vec<paths::Statistics>, String> {
                let (walk, own) = (&paths.walks[if j < n { j } else { j - n }], own_kept(j));
                // A stretch the candidate did not read (`judge_on`'s others): no landing check.
                if own.readings.is_empty() {
                    return Ok(vec![]);
                }
                list.iter()
                    .map(|l| {
                        let (photons, _, spent) = form.landing(walk, l.t, &l.landed, span, decay, knobs, own)?;
                        let (mine, solved) = own.kept.iter().find(|x| x.0 == l.t).map_or((None, 0.0), |(t, v)| form.diffusion(walk, *t, v));
                        let diffusion = match (mine, l.diffusion) {
                            (Some(c), Some(r)) => [c, r],
                            _ => [[0.0; 3]; 2],
                        };
                        Ok((paths::Statistics { t: l.t, q: [2.0 * photons, 2.0 * l.photons], diffusion }, spent + solved))
                    })
                    .collect::<Result<Vec<_>, String>>()
                    .map(|x| x.into_iter().unzip::<_, _, Vec<_>, Vec<f64>>())
                    .map(|(stats, works)| { landing_work.fetch_add((works.iter().sum::<f64>()) as u64, std::sync::atomic::Ordering::Relaxed); stats })
            })
            .collect::<Result<_, _>>()?,
    };
    let (landed_start, landed_late) = landed.split_at(n.min(landed.len()));
    let escape = escape_of(paths, &series, &late_series, landed_start, landed_late, hm);
    let weights: Vec<f64> = paths.walks.iter().map(|w| w.weight).collect();
    let windows = paths::compare(&series, &paths.series, &paths.windows, labels, hm, landed_start, &weights);
    let late = paths.late.as_ref().map_or(vec![], |(_, w, s)| paths::compare(&late_series, s, w, labels, hm, landed_late, &weights));
    let statistics = {
        let all: Vec<&paths::Statistics> = landed.iter().flatten().collect();
        (!all.is_empty()).then(|| [0, 1].map(|m| all.iter().map(|x| x.q[m]).sum::<f64>() / all.len() as f64))
    };
    // The motion's windows, said where a test asks (`SEL_MOTION`).
    #[cfg(test)]
    if std::env::var_os("SEL_MOTION").is_some() {
        let m = |x: &paths::Mean| format!("{:+.3e}±{:.1e}", x.value, x.error);
        eprintln!("    [motion] P{} landing Q (candidate, reference) mean {:?}; per stretch (t, Q, D_c, D_r) {:?}", model.rs.keep.len(), statistics, landed.iter().map(|l| l.iter().map(|x| (x.t, x.q, x.diffusion)).collect::<Vec<_>>()).collect::<Vec<_>>());
        for w in windows.iter().chain(&late) {
            eprintln!(
                "    [motion] P{} ({:.2}, {:.2}] {} paths: rate {}; dP_a [{}, {}, {}]; H_a [{:.3e}, {:.3e}, {:.3e}]; dF_a [{}, {}, {}]",
                model.rs.keep.len(), w.from, w.to, w.paths, m(&w.rate), m(&w.energy_rate[0]), m(&w.energy_rate[1]), m(&w.energy_rate[2]), w.heating[0].value, w.heating[1].value, w.heating[2].value, m(&w.force[0]), m(&w.force[1]), m(&w.force[2])
            );
        }
        eprintln!("    [motion] P{} escape {:?}", model.rs.keep.len(), escape);
        if let (Some(x), Some(y)) = (series.first().and_then(|s| s.readings.get(3)), paths.series.first().and_then(|s| s.readings.get(3))) {
            eprintln!("    [motion] P{} reading t {:.3}: rates {:.4e} / {:.4e}, force {:?} / {:?}, absorbed {:?} / {:?}, carried {:?} / {:?}, emitted {:?} / {:?}, kappa {:?}, hm {hm:e}", model.rs.keep.len(), x.t, x.rates.iter().sum::<f64>(), y.rates.iter().sum::<f64>(), x.force, y.force, x.absorbed, y.absorbed, x.carried, y.carried, x.emitted, y.emitted, x.kappa);
        }
    }
    // The calibration: the calibration's walk's own series at the setting along its stretch (the same scheme from the
    // same start) and its refinements, against the reference's.
    let candidate: Vec<&Series> = std::iter::once(&series[longest]).chain(&finer).collect();
    let mut calibration = if finer.is_empty() { Calibration::default() } else { calibrate(&candidate, calibrated, &paths.settled.pairs) };
    // The ladder's top unmet, and what the openings book, the reference's and the candidate's (paths.rs, # The start);
    // the trace defects on every path, the reference's and the candidate's (paths.rs `Series::trace_defect`: the scheme's
    // error read where no refinement ran).
    let own = |f: fn(&Series) -> f64| series.iter().chain(&late_series).map(f).fold(0.0, larger);
    calibration.rows += paths.settled.unmet().unwrap_or(0.0) + paths.booked + own(|s| s.counts.booked) + paths.trace + own(Series::trace_defect);
    let products = series.iter().chain(&late_series).chain(&finer).map(|s| s.counts.products).sum();
    let stationary = Stationary::of(&windows, &late, paths.memory, paths.late_from(), &weights);
    let work = series.iter().chain(&late_series).chain(&finer).map(|s| s.counts.work).sum::<f64>() + landing_work.load(std::sync::atomic::Ordering::Relaxed) as f64;
    Ok(Judged { keep: model.rs.keep.clone(), windows, late, calibration, escape, statistics, stationary, seconds: clock.elapsed().as_secs_f64(), products, work })
}

/// The loss's x of a candidate (module documentation, # The motion; `Judged::escape`) of its series along the walks'
/// starts `candidate` and late segments `late` against the reference's: per path the mean over its read stretches of
/// the energy rate's difference -- per reading Σ_a (δH_a + δF_a v_a) by the trapezoid --, over the energy its atom gained to its loss per µs
/// on the reference's prepared potential where it was lost, (E(loss) − E(0))/t_loss, else over the reference's heating's
/// mean there; the lost paths' where at least two are, else every path's.
fn escape_of(paths: &Paths, candidate: &[Series], late: &[Series], landed: &[Vec<paths::Statistics>], landed_late: &[Vec<paths::Statistics>], hm: f64) -> Option<(paths::Mean, usize)> {
    let reference = paths.reference.model;
    let well = reference.prepared(&crate::solve::alone(crate::solve::chief(&reference.scene.request)));
    let energy = |e: &[f64; 7]| 0.5 * hm * (e[4] * e[4] + e[5] * e[5] + e[6] * e[6]) + well.at([e[1], e[2], e[3]]).value;
    let late_reference = paths.late.as_ref().map(|l| &l.2);
    let per_path = |k: usize| -> Option<(bool, f64)> {
        let (mut integral, mut time, mut heat) = (0.0, 0.0, 0.0);
        let stretches = std::iter::once((&candidate[k], &paths.series[k], landed.get(k))).chain(late.get(k).zip(late_reference.and_then(|l| l.get(k))).map(|(c, r)| (c, r, landed_late.get(k))));
        for (c, r, f) in stretches {
            // The stretch's landing check, its mean over its landings (none read where none).
            let f = f.map_or(paths::Statistics::NONE, |f| paths::Statistics::within(f, f64::NEG_INFINITY, f64::INFINITY));
            let pairs: Vec<(&paths::Reading, &paths::Reading)> = c.readings.iter().zip(&r.readings).collect();
            let (Some(first), Some(last)) = (pairs.first(), pairs.last()) else { continue };
            if !(last.0.t > first.0.t) {
                continue;
            }
            let rate = |(x, y): &(&paths::Reading, &paths::Reading)| -> (f64, f64) {
                let (hx, hy, wx, wy) = (x.heating_at(hm, f.q[0], f.diffusion[0]), y.heating_at(hm, f.q[1], f.diffusion[1]), x.work(hm), y.work(hm));
                ((0..3).map(|a| hx[a] - hy[a] + wx[a] - wy[a]).sum(), hy.iter().sum())
            };
            for w in pairs.windows(2) {
                let (d0, h0) = rate(&w[0]);
                let (d1, h1) = rate(&w[1]);
                let dt = w[1].0.t - w[0].0.t;
                integral += 0.5 * dt * (d0 + d1);
                heat += 0.5 * dt * (h0 + h1);
            }
            time += last.0.t - first.0.t;
        }
        if !(time > 0.0) {
            return None;
        }
        let walk = &paths.walks[k];
        let gained = match (walk.lost, walk.ends.first(), walk.ends.last()) {
            (true, Some(a), Some(b)) if b[0] > a[0] => (energy(b) - energy(a)) / (b[0] - a[0]),
            _ => 0.0,
        };
        if walk.lost && gained > 0.0 {
            Some((true, integral / time / gained))
        } else {
            (heat > 0.0).then(|| (false, integral / heat))
        }
    };
    let all: Vec<(bool, f64)> = (0..candidate.len()).filter_map(per_path).collect();
    let lost: Vec<f64> = all.iter().filter(|x| x.0).map(|x| x.1).collect();
    let x: Vec<f64> = if lost.len() >= 2 { lost } else { all.iter().map(|x| x.1).collect() };
    (!x.is_empty()).then(|| (paths::Mean::of(&x), x.len()))
}


/// The calibration (`Calibration`) of a candidate's series along the calibration's stretch at the paths' runs
/// (`candidate`: the setting first, its refinements after), against the reference's alike (`reference`), read at the
/// times all read (the knots): per knob the largest change of a row's difference to the reference between its two runs
/// (`pairs`, `refinements`), and of the total rate's relative difference, read as its error part (`RICHARDSON`), the
/// parts added.
fn calibrate(candidate: &[&Series], reference: &[Series], pairs: &[(usize, usize); KNOBS]) -> Calibration {
    fn at(s: &Series, t: f64) -> Option<&paths::Reading> {
        s.readings.iter().find(|r| r.t == t)
    }
    let total = |r: &paths::Reading| r.rates.iter().sum::<f64>();
    let mut parts = [Calibration::default(); KNOBS];
    for x in &candidate[0].readings {
        for (part, &(i, j)) in parts.iter_mut().zip(pairs) {
            let (Some(a), Some(b), Some(u), Some(v)) = (at(candidate[i], x.t), at(&reference[i], x.t), at(candidate[j], x.t), at(&reference[j], x.t)) else { continue };
            for q in 0..STATES {
                part.rows = larger(part.rows, ((a.rows[q] - b.rows[q]) - (u.rows[q] - v.rows[q])).abs());
            }
            part.rate = larger(part.rate, ((total(a) / total(b) - 1.0) - (total(u) / total(v) - 1.0)).abs());
        }
    }
    Calibration { rows: RICHARDSON * parts.iter().map(|p| p.rows).sum::<f64>(), rate: RICHARDSON * parts.iter().map(|p| p.rate).sum::<f64>() }
}

// ---------------------------------------------------------------------------------------------------------------------
// The accounts.

/// Whether T's memory is not shorter than the run, of its memory at the run's end `r` (µK, on the momenta's-scaling
/// scale: T(0) moved by 2T(0); module documentation, # T's memory, read on the run's own trajectories) and its standard
/// error `se`, T's allowance there `allowance` (µK): yes where it is read there, |r| > z·se (FOLDTEMP §R2.3: "缩放响应
/// 在运行末仍可读出"); no where even its upper end |r| + z·se is within the allowance -- a unit kick of the initial
/// state's temperature moves the end's T by less than its target; none otherwise (no flag: the run's groups do not tell).
pub fn decide(r: f64, se: f64, allowance: f64) -> Option<bool> {
    if !(r.is_finite() && se.is_finite()) {
        None
    } else if r.abs() > CONFIDENCE * se {
        Some(true)
    } else if r.abs() + CONFIDENCE * se < allowance {
        Some(false)
    } else {
        None
    }
}

/// Which end of a path mean an account reads (module documentation, # The direct part, # How many paths): its upper
/// end (the one booked), its lower (a fate decided at both is the paths' statistics'), or the point estimate itself
/// (the nomination's, # The candidates).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Reach {
    Upper,
    Lower,
    Point,
}

/// A path mean's end (module documentation, # The direct part): |mean| + q·se, |mean| − q·se (none below zero), or
/// |mean| (`reach`), q Student's t at the run's confidence with the `paths` read less one degrees of freedom; of fewer
/// than two paths q is without bound (the spread unknown).
fn end(m: &paths::Mean, paths: usize, reach: Reach) -> f64 {
    let q = if paths >= 2 { crate::estimate::quantile(crate::estimate::tail_among(1.0), Some(paths as u64 - 1)) } else { f64::INFINITY };
    let spread = if m.error == 0.0 { 0.0 } else { q * m.error.abs() };
    match reach {
        Reach::Upper => m.value.abs() + spread,
        Reach::Lower => larger(m.value.abs() - spread, 0.0),
        Reach::Point => m.value.abs(),
    }
}

/// The direct part of a window (module documentation, # The direct part): per row its end (`end`).
fn direct(w: &Window, reach: Reach) -> Vec<f64> {
    w.rows.iter().map(|m| end(m, w.paths, reach)).collect()
}

/// The windows a candidate's error at the time t > 0 is read in (module documentation, # The direct part): the time's
/// window (`paths::window_of` on the start's windows `windows`); past L the start's last and the late segment's last
/// window, past the late segment's start the latter alone; each one that some path reads. None of the reference itself
/// (no windows).
fn windows_at<'a>(judged: &'a Judged, windows: &[(f64, f64)], t: f64) -> Vec<&'a Window> {
    if judged.windows.is_empty() {
        return vec![];
    }
    let length = windows.last().map_or(0.0, |w| w.1);
    let at: Vec<&Window> = match (judged.late.first().map(|w| w.from), judged.windows.last(), judged.late.last()) {
        (Some(from), Some(last), Some(late_last)) if t > length => {
            if t > from { vec![late_last] } else { vec![last, late_last] }
        }
        _ => vec![&judged.windows[paths::window_of(windows, t).unwrap_or(0)]],
    };
    at.into_iter().filter(|w| w.paths > 0).collect()
}

/// A candidate's direct error at the time t (module documentation, # The direct part): the largest row's |mean| + z·se
/// of the time's window (`paths::window_of` on the start's windows `windows`), past L the larger of the start's last
/// window's and the late segment's last window's, past the late segment's start the latter's; the scheme's calibration
/// beside it. Zero at t = 0, where every model starts in the same state; none where the window it takes no path reads
/// (`paths::Window::paths`: every path's atom lost before it) -- nothing held there on the paths to err on, no number.
pub fn direct_at(judged: &Judged, windows: &[(f64, f64)], t: f64) -> Option<f64> {
    direct_bound_at(judged, windows, t, Reach::Upper)
}

/// `direct_at` at an end (`Reach`) of each row, the calibration beside it (# How many paths: a fate decided at both ends
/// is the paths' statistics').
pub fn direct_bound_at(judged: &Judged, windows: &[(f64, f64)], t: f64, reach: Reach) -> Option<f64> {
    if t <= 0.0 || judged.windows.is_empty() {
        return Some(0.0);
    }
    let read = windows_at(judged, windows, t);
    (!read.is_empty()).then(|| read.iter().flat_map(|w| direct(w, reach)).fold(0.0, larger) + judged.calibration.rows)
}

/// The pilot's course the motion's errors are read on (module documentation, # The motion; `motion_at`): its times --
/// the request's outputs, where the choice weighs its candidates and the product books their error, or the analysis
/// times --, its loss and its T per lab axis there (none where it holds no atom), the reported trap's frequencies per
/// lab axis (rad/µs), and per time and lab axis T's window's half width where the times are the outputs (T read there
/// is its window's mean: its slope is read across the window, `temperature_slope`).
#[derive(Clone, Debug, Default)]
pub struct Course {
    pub times: Vec<f64>,
    pub loss: Vec<f64>,
    pub temperature: Vec<Option<[f64; 3]>>,
    pub frequencies: [f64; 3],
    pub halves: Option<Vec<[f64; 3]>>,
}

impl Course {
    /// The loss at the time t on the line through the times read, past the last on its last interval's line (the
    /// first order where the course reads nothing), within [0, 1].
    fn loss_at(&self, t: f64) -> f64 {
        let n = self.times.len();
        if n < 2 {
            return self.loss.first().copied().unwrap_or(0.0);
        }
        let b = self.times.iter().position(|&x| x >= t).unwrap_or(n - 1).max(1);
        let a = b - 1;
        let s = (t - self.times[a]) / (self.times[b] - self.times[a]);
        (self.loss[a] + s * (self.loss[b] - self.loss[a])).clamp(0.0, 1.0)
    }

    /// The loss's change at the k-th time where the run's clock runs 1 + x times as fast (module documentation, # The
    /// motion: the time-scaling identity itself, L(t(1 + x)) − L(t), not its first order x t L̇ -- the loss's front read
    /// at the resolution its course is read at, no slope taken of it).
    fn loss_shifted(&self, k: usize, x: f64) -> f64 {
        self.loss_at(self.times[k] * (1.0 + x)) - self.loss[k]
    }

    /// T's slope along the lab axis a at the k-th time (µK/µs): across T's window there where the course has its half
    /// widths (`halves`: T at an output is its window's mean, the slope of neighbours closer than that the noise's), the
    /// line through T read at t − h and t + h, each on the line through the times read where it is read; else the
    /// neighbours' (`slope`).
    fn temperature_slope(&self, k: usize, a: usize) -> Option<f64> {
        let value = |i: usize| self.temperature[i].map(|x| x[a]);
        let Some(h) = self.halves.as_ref().map(|h| h[k][a]).filter(|h| *h > 0.0) else { return self.slope(k, &value) };
        let at = |t: f64| -> Option<f64> {
            let b = self.times.iter().position(|&x| x >= t)?;
            if b == 0 {
                return value(0);
            }
            let (va, vb) = (value(b - 1)?, value(b)?);
            Some(va + (t - self.times[b - 1]) / (self.times[b] - self.times[b - 1]) * (vb - va))
        };
        let t = self.times[k];
        let (lo, hi) = ((t - h).max(0.0), (t + h).min(*self.times.last()?));
        match (at(lo), at(hi)) {
            (Some(x), Some(y)) if hi > lo => Some((y - x) / (hi - lo)),
            _ => self.slope(k, &value),
        }
    }

    /// A reading's slope at the k-th time, of its values at the times (`value`, none where not read): the central
    /// difference of the neighbours read, one-sided at an end; none of fewer than two.
    fn slope(&self, k: usize, value: &dyn Fn(usize) -> Option<f64>) -> Option<f64> {
        let n = self.times.len();
        let before = (0..k).rev().find(|&i| value(i).is_some());
        let after = (k + 1..n).find(|&i| value(i).is_some());
        let (a, b) = match (before, after) {
            (Some(a), Some(b)) => (a, b),
            (Some(a), None) => (a, k),
            (None, Some(b)) => (k, b),
            (None, None) => return None,
        };
        let (va, vb) = (value(a)?, value(b)?);
        (self.times[b] > self.times[a]).then(|| (vb - va) / (self.times[b] - self.times[a]))
    }
}

/// The motion's errors at a time (module documentation, # The motion): on the loss (absolute), on the total T (µK: the
/// one T's target is held to, the user's of 2026-11-22), on the push per lab axis (µm/µs²) and on the mean velocity per
/// lab axis (m/s); and T's per lab axis (µK, each axis's own window end: said, not booked).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Motion {
    pub loss: f64,
    pub temperature_total: f64,
    pub push: [f64; 3],
    pub velocity: [f64; 3],
    pub temperature: [f64; 3],
}

/// The energy rates the total T's error reads (module documentation, # The motion): per path the candidate's δP_a and
/// the reference's H_a over the whole of its stretches past the memory time -- the start's from τ_mem on, the late
/// segment's from its own τ_mem on --, each window's per-path means weighed by the window's length. A window shorter
/// than the motion's periods holds the conservative exchange's oscillation (a trap's or a lattice's force against the
/// atom's swing), which moves no energy over a period and is bounded over any time: read as a rate on one short window
/// and multiplied by an axis's whole response time, it is noise taken for heating (tilt0's tweezer axis, 15.6 kHz: its
/// 22 µs windows read δP_x ± 4e-3 /µs² against the 2e-3 of its rates, 108 µK over 1 ms at the window's upper end).
#[derive(Clone, Debug, PartialEq)]
pub struct Stationary {
    pub rates: Vec<[f64; 3]>,
    pub heating: Vec<[f64; 3]>,
    /// Each path's weight (`paths::Walk::weight`).
    pub weights: Vec<f64>,
}

impl Stationary {
    /// The start's windows `windows` and the late segment's `late` (from `late_from`), past the memory time `memory`.
    pub fn of(windows: &[Window], late: &[Window], memory: f64, late_from: Option<f64>, weights: &[f64]) -> Option<Stationary> {
        let mut sums: std::collections::BTreeMap<usize, ([f64; 3], [f64; 3], f64)> = std::collections::BTreeMap::new();
        let forgotten = windows.iter().filter(|w| w.from >= memory).chain(late.iter().filter(|w| late_from.is_some_and(|f| w.from >= f + memory)));
        for w in forgotten {
            let span = w.to - w.from;
            for &(k, rate, heating) in &w.energy_rates {
                let entry = sums.entry(k).or_insert(([0.0; 3], [0.0; 3], 0.0));
                for a in 0..3 {
                    entry.0[a] += span * rate[a];
                    entry.1[a] += span * heating[a];
                }
                entry.2 += span;
            }
        }
        let read: Vec<(&usize, &([f64; 3], [f64; 3], f64))> = sums.iter().filter(|x| x.1 .2 > 0.0).collect();
        (!read.is_empty()).then(|| Stationary {
            rates: read.iter().map(|x| x.1 .0.map(|v| v / x.1 .2)).collect(),
            heating: read.iter().map(|x| x.1 .1.map(|v| v / x.1 .2)).collect(),
            weights: read.iter().map(|x| weights.get(*x.0).copied().unwrap_or(1.0)).collect(),
        })
    }
}

/// A candidate's motion's errors at the course's k-th time (module documentation, # The motion), each at its window's
/// end (`reach`): none at t = 0 or of the reference itself (no windows), or where no path reads the
/// time's windows.
pub fn motion_at(judged: &Judged, windows: &[(f64, f64)], course: &Course, k: usize, hbar_over_mass: f64, reach: Reach) -> Option<Motion> {
    let t = course.times[k];
    let read = windows_at(judged, windows, t);
    if t <= 0.0 || read.is_empty() {
        return None;
    }
    let most = |of: &dyn Fn(&Window) -> f64| read.iter().map(|w| of(w)).fold(0.0, larger);
    // The loss: L(t(1 + x)) − L(t), x at its end, of its own sign.
    let loss = judged.escape.as_ref().map_or(0.0, |(x, n)| course.loss_shifted(k, end(x, *n, reach).copysign(x.value)).abs());
    // T per lab axis: δE_a = δP_a (1 − e^{−α t})/α, α = (H_a − Ė_a)/E_a, the rates over the paths' forgotten stretches
    // (`Stationary`, the time's own windows where none is); the total T's, the axes' mean, per path (its spread the sum's
    // own).
    let (mut temperature, mut temperature_total) = ([0.0; 3], 0.0);
    if let Some(at) = course.temperature[k] {
        let kt = crate::units::KT_PER_UK;
        let own: Option<Stationary> = judged.stationary.clone().or_else(|| {
            let w = read.last()?;
            Some(Stationary { rates: w.energy_rates.iter().map(|x| x.1).collect(), heating: w.energy_rates.iter().map(|x| x.2).collect(), weights: vec![1.0; w.energy_rates.len()] })
        });
        if let Some(own) = own {
            let paths = own.rates.len();
            let weight: f64 = own.weights.iter().sum();
            let heating: [f64; 3] = std::array::from_fn(|a| own.heating.iter().zip(&own.weights).map(|(h, w)| w * h[a]).sum::<f64>() / weight);
            let held: [f64; 3] = std::array::from_fn(|a| {
                let warming = kt * course.temperature_slope(k, a).unwrap_or(0.0);
                let energy = kt * at[a];
                let alpha = if energy > 0.0 { larger((heating[a] - warming) / energy, 0.0) } else { 0.0 };
                if alpha * t > 0.0 { -(-alpha * t).exp_m1() / alpha } else { t }
            });
            for a in 0..3 {
                temperature[a] = end(&paths::Mean::weighted(&own.rates.iter().map(|r| r[a]).collect::<Vec<_>>(), &own.weights), paths, reach) * held[a] / kt;
            }
            let each: Vec<f64> = own.rates.iter().map(|p| (0..3).map(|a| p[a] * held[a]).sum::<f64>() / (3.0 * kt)).collect();
            temperature_total = end(&paths::Mean::weighted(&each, &own.weights), paths, reach);
        }
    }
    let push: [f64; 3] = std::array::from_fn(|a| hbar_over_mass * most(&|w| end(&w.force[a], w.paths, reach)));
    let velocity = std::array::from_fn(|a| push[a] * if course.frequencies[a] > 0.0 { t.min(1.0 / course.frequencies[a]) } else { t });
    Some(Motion { loss, temperature_total, push, velocity, temperature })
}

/// What the choice made of the chain (solve.rs, # The selection): P, its chain, its judgment, the paths' windows, the
/// memory time and L, and the report's lines -- every candidate judged with its direct part, its cost and why it was or
/// was not chosen, the stages' times.
#[derive(Clone, Debug)]
pub struct Choice {
    pub keep: Vec<usize>,
    /// The chain the choice was made on: by the importance at the pilot's states (`importance`).
    pub chain: Vec<Vec<usize>>,
    pub judged: Judged,
    pub windows: Vec<(f64, f64)>,
    pub memory: f64,
    pub length: f64,
    pub report: Vec<String>,
}

impl Choice {
    /// P's direct error at the time t (`direct_at`): on the rows shown (the motion's readings' are `motion`); none where
    /// its window reads no path.
    pub fn at(&self, t: f64) -> Option<f64> {
        direct_at(&self.judged, &self.windows, t)
    }

    /// P's motion's errors at the course's k-th time (`motion_at`, at the upper end): on the run's own pilot's course.
    pub fn motion(&self, course: &Course, k: usize, hbar_over_mass: f64) -> Option<Motion> {
        motion_at(&self.judged, &self.windows, course, k, hbar_over_mass, Reach::Upper)
    }
}

/// The key the choice is kept under (BRIEF_P3_SELECT item 6: the scene's physics): the request without its times --
/// its length, its outputs and the samples they make. The worker is one process for every request (main.rs), so a scene
/// solved again with other times takes its choice from here (`CHOICES`).
pub fn key(request: &crate::request::Request) -> String {
    let mut r = request.clone();
    r.times_us = Vec::new();
    format!("{r:?}")
}

#[cfg(test)]
thread_local! {
    /// Candidates beside the chain a test asks the choice to judge and weigh for comparison, never to choose (solve.rs
    /// `choose_on_the_pilot`): each its states.
    pub(crate) static ALSO: std::cell::RefCell<Vec<Vec<usize>>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The choices made (`key`), in the order they were made.
static CHOICES: std::sync::Mutex<Vec<(String, std::sync::Arc<Choice>)>> = std::sync::Mutex::new(Vec::new());

/// The choice kept for a request (`key`), where one was made.
pub fn kept(request: &crate::request::Request) -> Option<std::sync::Arc<Choice>> {
    let key = key(request);
    CHOICES.lock().unwrap_or_else(|e| e.into_inner()).iter().find(|(k, _)| *k == key).map(|(_, c)| c.clone())
}

/// A choice kept for its request (`key`).
pub fn keep(request: &crate::request::Request, choice: std::sync::Arc<Choice>) {
    let key = key(request);
    let mut list = CHOICES.lock().unwrap_or_else(|e| e.into_inner());
    list.retain(|(k, _)| *k != key);
    list.push((key, choice));
}

#[cfg(test)]
pub(crate) mod tests;
