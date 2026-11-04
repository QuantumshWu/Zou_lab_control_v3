//! A 3D motion solve: the model, its approximations' ledger, the time step from a step-doubling
//! check, then trajectories in sequential batches until the standard errors meet the targets at
//! every output time (as QuTiP's mcsolve does with its target tolerance), each batch's result handed
//! out as a checkpoint with the wall time the remaining trajectories will take.
//!
//! # Targets
//! The semiclassical method is accurate to about `TEMPERATURE_TARGET` in T and
//! `POPULATION_TARGET` in the populations in its regime; the trajectories are run until each
//! output's total estimated error (below) lies within those targets: every population's and the
//! loss's at every output time, and every temperature's (each axis, relative) at the output times
//! in the regime -- outside it the method's own error is larger and the temperature is a reference
//! only.
//!
//! # Ledger and the total error
//! Every a-priori approximation's estimate enters the solve's ledger (ledger.rs), reported against
//! its budget, the a-priori share of the population target. The decisions take that share: the
//! couplings' and the far components' folds (ground.rs), a Gaussian beam's intensity axis (model.rs)
//! and, per step, the first order in the Doppler shift -- what the budget leaves after the model's
//! estimates and the pilot's elimination, or the budget on its own where the elimination alone exceeds it (the exact
//! response beyond, engine.rs). The elimination's η² -- each output's mean
//! excited population, the local expansion parameter ⟨Y⟩ of each atom's state -- is measured along
//! the trajectories and is no decision yet: it enters the total (validated on resonant probes against
//! the population core: optical pumping's population error 0.4–1.1 η², a cycling transition's 2η⁴).
//!
//! The fold's charge at an output is its error over the populations' memory before it. A step's defect
//! reaches a later time through the problem's own propagator (the error transport of an ODE's global
//! error; a dissipative problem's propagator decays with its relaxation, and the a-posteriori estimates
//! weigh each defect by that decay): an error the fold makes while the atoms are bright stays in their
//! populations until they relax, and an atom's populations relax through the photons it scatters. An atom
//! pumped into a dark state keeps its transient's error (an optical pumping's, the validation's M1: 0.028
//! at 5 µs), one that scatters forgets it within a few photons. So the charge at output i is the largest
//! over j ≤ i of output j's 2η² -- its error over the interval after j, where the larger of the interval's
//! two ends bounds it -- times the mean over the atoms held at i of e^{−n}, n an atom's jumps since the
//! end of that interval (`Memory`, the pilot's atoms): at least its own 2η², and at most the largest since
//! t = 0. (QuTiP's solvers control a step's local error and leave the global error, as an elimination's
//! validity, to the user's convergence checks.)
//!
//! An output's total estimated error is the ledger's total there (its η², its mean (kv/σ)² and the
//! profiles at the atoms with the model's estimates), the time step's bias (its step-doubling share)
//! and CONFIDENCE standard errors: the populations' against the population target (absolute), T's
//! against the temperature target (relative, where the method holds) -- the fold's charge in either
//! its own (`fold_charges`: at fourth order the populations' S², T's still S, the force folding at
//! second order). The trajectories are run until
//! the totals meet the targets -- the statistics within what the ledger and the step leave (# Shares) --
//! and an output whose total exceeds a target all the same is flagged (`Ledger::flagged`; a
//! checkpoint's statistics still run, so only its a-priori excess is).
//!
//! # The fold's choice
//! The rule's candidates (ground.rs `Ground::allowances`) from all folded down, and on the CPU the
//! first's fourth order, are costed on the pilot's states carried into their clusters (`carried`): each
//! with its fold's charge as measured (`measured_charge`: per output the charge the run's ledger books, with
//! the memory). A candidate's cost is its step check's (below) on those states -- the step it would take --:
//! its counted work per interval there times the trajectories its rooms at that step's bias need and the weight
//! of the totals it leaves (`predicted`: per output, the pilot's statistics against what the batches take, the
//! statistic they bind on -- a mean charge over the outputs under-predicted the D1 gray molasses' need twice over,
//! its early transient), over the run: continuous in its estimates (# Shares). The least is taken,
//! one keeping more of the model (its fourth order, a rung) only for a saving beyond the need's own relative
//! error (`need_error`, as a finer step, `kept`); the ladder's rungs are costed only where the folded models'
//! cost would fall by more than that error without the estimates a rung can change (their totals at the pilot's
//! step with only `Candidate::fixed`): below, no rung -- more of the model explicit, more work per step -- can
//! win. The run goes on from those states (no pilot runs again); no timing enters, so a request gives the same
//! choice on any machine.
//!
//! # Time step
//! The trap's harmonic part is moved exactly (engine.rs); what the step must resolve is the light's
//! pattern crossed by the moving atom and the internal state's response to it, which carries the
//! light's friction. The step is chosen by step doubling on the deterministic flow (as an adaptive
//! ODE solver's local error, QuTiP's way): states are sampled along a pilot of a few trajectories
//! at the step the bound rule's search (below) finds on the pilot's own initial states, from the
//! phase k_max v_T of the light's pattern at the initial temperature advancing sqrt(s ε_T) per step --
//! a pilot coarser than its own criteria heats its states, and the run's check on them goes finer
//! (the 3D PGC at the phase's 0.32 us: the orbits' T error 4-5 times the run's at equal steps) --, and
//! from each one step of dt is compared with two of dt/2, without jumps. A step meets the targets
//! when (4/3 of) the differences meet their tolerances, the step's share s of each target's room
//! (# Shares):
//! - the internal state's populations after one waiting time between jumps (the pilot's mean),
//!   within its tolerance. Bounded by one step's difference times the steps of
//!   that time; where the motion's criteria below are met, measured as well: from each of a
//!   subsample of the states (one per pilot trajectory) the jump-free flow over that time (at most
//!   the run) in steps of dt against steps of dt/2, the largest difference of the cluster's
//!   populations, their mean over the subsample plus CONFIDENCE standard errors -- in a gray
//!   molasses the atoms wait 50-140 us in dark states, and the bound over-counts the step's error
//!   there several times;
//! - the step's light impulse, relative, within T's tolerance;
//! - T's own step error, the orbits' (`orbits`): from one state per pilot trajectory bound to the trap
//!   (the atoms whose T the step's error stays on: an unbound one leaves within a transit of the trap's
//!   region) the jump-free flow over one period of the trap's slowest axis, the span of the solve's T
//!   windows, in steps of dt against steps of dt/2, ψ kept normalized; per axis the mean κ² over it
//!   compared, the probes' mean relative difference plus CONFIDENCE standard errors, within T's
//!   tolerance. The integrator keeps the conservative motion's energy error bounded rather
//!   than accumulating, so that a step's energy error summed over the run (the former friction
//!   criterion) over-counts T's error many times -- the tweezer alone at 0.45 us 0.99 of its share
//!   against the orbits' 0.13, and the retro-reflected tweezer refined to 0.0076 us where T's measured
//!   error asks 0.03–0.05 us --;
//!   with the orbits, the impulse and the populations' criterion, the 3D PGC and the tetrahedral MOT
//!   hold T within 0.4 % and 1.0 % of a quarter of their steps at the outputs where it binds (4096
//!   trajectories on shared streams);
//! - the heating the jumps carry, their counterpart of the orbits (`jumps`), within T's tolerance: the step holds a (sub-)step's jump hazard R at its middle, so the jumps fall evenly over a
//!   step instead of by the hazard along it, and a jump's expected deposit -- its cross term
//!   c = (ħ²/m) κ·E[Δκ], E[Δκ] the mean kick of its photons -- takes a first-order profile over its place
//!   in the step, of slope S = −E[c ΔR]/E[R] per step (ΔR the hazard's change over the jump-free step at
//!   the state's ψ: the generator's, which the step holds; E over the sampled states). Over a step the
//!   profile cancels, but a jump that follows another within its step (by the chance R(1 − u) dt after
//!   one at u) keeps the first one's place in it: per jump a net −(R_j dt/12) S, R_j = E[R²]/E[R] the
//!   hazard at the jumps -- relative to the jumps' recoil per jump, which bounds T's relative error
//!   whether the motion accumulates their heating or its cooling balances it. On the retro-reflected
//!   tweezer with a resonant probe (the 951 kHz lattice axis turns the velocity by 0.3 rad per step):
//!   the profile's slope 5.85 µK per 0.0526 us step predicted against 5.80 measured on the jumps, and
//!   the net with the probe at half, the same and twice its intensity −0.020, −0.071 and −0.187 µK per
//!   jump against −0.019, −0.064 and −0.185 measured (the step against itself with the hazard taken
//!   along the step, 4096 trajectories on shared streams; a step's first jumps agree, the net is their
//!   followers'); the step the line took there, 0.054 us, left T −3.1 ± 1.9 % at its binding outputs
//!   against 0.0132 us, beyond its 1 % share (−1.3 ± 1.5 % with the near light's dressing, against 0.0083
//!   us, 8192 trajectories); the jumps' criterion takes it to 0.033 us, T −0.1 ± 1.5 % there. A doubling
//!   of the deposit itself cancels by the motion's time reversal (the cross term is odd in v) and misses it;
//! - the loss, within the populations' tolerance: an escape over the trap's top moves with the motion's energy -- the pilot's
//!   escaped share times D/k_BT per relative error of it (`escape`) --, by the larger of the orbits'
//!   and the jumps' errors.
//! T's criteria take the step's share of T's room where T binds (`Binding`) and the target itself where
//! it is a reference only (`tolerances`): its share would refine for a value no target takes, while a
//! step that leaves the motion unresolved still moves the loss (the retro-reflected tweezer over 10 us
//! at the stiff axis' period, 1 us: 0.9 % of its atoms lost, T 160 % high, against 0.016 us).
//! All scale as dt², so a step that misses is shrunk by the square root of its excess and checked
//! again, and one that meets them with room is grown the same way, by whatever factor, until a step
//! comes up twice: first by the population criterion's bound, then from the largest step that met
//! the targets by it with the population error measured -- by the excess of the criteria evaluated
//! (`next_step`): where the motion or the loss missed, the population was not measured, and its bound,
//! which over-counts several times where the measured search runs, is left out (a step that missed by
//! 1 % on the motion went back by the bound's ninefold excess to a third of its size). Every checked
//! step that met the targets is a candidate, and the largest is kept (`kept`) -- the field's rule, the
//! largest step that meets the tolerances --, a finer one only where it is predicted to cost less by
//! more than the prediction's own uncertainty: the relative standard error of the trajectories the
//! pilot predicts (`need_error`, the jackknife of its statistics over its groups). A met step's bias is at
//! most its share of the run's room (# Shares): it leaves the statistics their balance by construction. The
//! same rule is taken once more over every step from
//! the largest to the finest candidate, their ratios the largest's as dt², and its pick checked too: the
//! choice moves with its estimates, not with the steps the search happened to check. The prediction
//! (`predicted`, the fold's choice's weighed cost): its work per interval
//! -- counted on the sampled states (`interval_work`): its steps' advances, each the generator's and
//! the force's multiply-adds and table pairs of the state's cluster and two half steps' series, or in a
//! cluster that takes the exact step the segment it builds and its grid's nodes (`exact_step`), and
//! per jump its photons and the two advances it redoes, each unit at a cost measured once (no timing
//! enters a solve: a request gives the same step and the same result on any machine and load) --,
//! times the trajectories the statistics need -- the pilot's statistics per output against the rooms the step's
//! bias leaves them -- and the weight of the totals beyond the targets. Kept by the
//! least predicted cost alone, the choice flipped where two candidates' costs crossed on a flat
//! landscape -- the D1 gray molasses forced all-folded between 0.25 and 0.083 us as its cooling
//! beams' couplings moved by 1e-3 (at 0.091 us it had run 2392 trajectories in 280 CPU-s against
//! 0.25 us's 1416 in 75): the largest met step moves with the criteria, and on a flat landscape the
//! coarse step's certain work is kept -- with the couplings at 1 ± 1e-3 and 1 ± 1e-2 the molasses keeps
//! 0.25 us at 1 us outputs and 0.222–0.25 us at 10 us outputs; forced all-folded at 1 us outputs, where
//! the a-priori T estimate leaves the coarse step's bias little room, 0.25, 0.143, 0.2, 0.25 and
//! 0.25 us. The step's exponentials are summed to the step's share of its own
//! population tolerance, s² ε dt/waiting, instead of to double precision. A trajectory takes at most
//! `MOST_STEPS` steps over the run: neither the first step nor the search goes finer, and a step that
//! bound sets is kept even where it misses its shares, its measured bias booked as any step's (the
//! flag, where a total exceeds its target, says why).
//!
//! # Shares
//! The targets are the totals: the a-priori estimates (the ledger's, each booked as measured), the step's bias and
//! CONFIDENCE standard errors together within each target, ε. The trajectories go as the inverse square of the room R
//! the estimates leave the bias and the statistics, the work per trajectory as 1/dt, and the bias as c dt^p (p =
//! `ORDER`, the integrator's): the cost W N ∝ 1/(dt (R − b)²) is least at b = R/(1 + 2p), the step's share s =
//! `STEP_SHARE` of the room -- 1/5 at second order; where fixed costs per unit time (the jumps, a multirate grid's
//! nodes) make the work go as dt^−w, w < 1, at R w/(w + 2p), which the step's predicted costs weigh among the steps
//! checked (`kept`). (MLMC splits a mean-square target into the bias' and the variance's halves; a total linear in
//! the bias and the standard error splits as here.) The numerical tolerances within a step -- its exponentials'
//! truncation, the beats its exact step leaves out, the impulse's quadrature (engine.rs `QUADRATURE`) -- take the
//! step's share of the step's own tolerance, s² ε: their work grows with the logarithm of their tolerance, so that
//! their optimum lies at a few per cent of it, its terms' share of the work. The model's approximations take no
//! fixed share: each is booked as measured against the target, and chosen by its cost (the fold's candidates, the
//! tables' refinement, model.rs `tabulate`); a model whose estimate the whole target cannot hold is refused (the
//! GPU's undressed response, gpu/program.rs). The gain's guard (model.rs `Model::lindblad`) is no such estimate: the
//! generator is of Lindblad form by construction, its round-off the limit of what it may read.
//!
//! As the a-priori estimates A near ε that work diverges, as R^−(2+1/p), while past ε no work meets the target: the rule
//! that held the statistics to the sliver below the target and gave them the whole target above it was a cliff (the
//! shallow probe's T, 8.99 % a priori and 0.88 % of bias: 0.13 % left, 1.75M trajectories, where a little more a priori
//! would have taken a few hundred). One weighed cost orders the decisions instead -- the batches', the step's and the
//! fold's: the work times (E/ε)^k beyond the target, E the predicted total (`beyond`). Its least over the statistics
//! alone, work ∝ S^−2, gives them what the floor F = A + b leaves of the target where that is at least 2F/(k − 2), and
//! that balance beyond (`room`): the total exceeds its target where the floor takes more than 1 − 2/k of it, and is
//! flagged as before. Over the run's errors together, work ∝ R^−(2+1/p), the least is R = A/(2p) beyond (1 − s)ε
//! (`run_room`), the step's share s of it its tolerance (`tolerances`), and at that bias the statistics' own balance is
//! exactly the rest: one optimum. The exponent k = (2 + 1/p)/s puts that turn where the run's errors -- what its work
//! moves -- would be left less than s of the total, the least share the optimum gives one of them (the step's bias beside
//! the statistics); beyond the target the run keeps that share, at most the target itself: the statistics never take
//! more than the whole target, since the a-priori estimates bound their errors and can lie far above them (F9's
//! interference at 20, PGCP_10's tables at 1.25), while what the statistics carry the result shows. The first order in
//! the Doppler shift takes the step's share of the run's room beside the other estimates (`doppler_allowance`), a
//! truncation per step as the step's own error. The candidates' weighed costs (# The fold's choice, # Time step) are
//! the same work times weight: a model whose total misses its target is taken only for a saving its weight outweighs --
//! 2 % beyond for 28 %, a quarter beyond for 16 times --, and every decision moves continuously with its estimates.
//!
//! # Batches
//! Every batch is cut into groups of `TASK` trajectories (lib.rs `Stratum`), a task on the CPU and a
//! block on the GPU: each group's initial states are a Latin hypercube, so its mean of the initial
//! energies, and of what they leave at early times, varies far less than independent draws'; the
//! groups are independent, so the jackknife over them measures the error. One size for every group
//! gives every trajectory the same share of the variance, so the need that a batch's errors predict
//! holds for the trajectories still to come; groups of 8 keep most of the hypercube's gain (the
//! trajectories T needs at early times fall by 10-40% from groups of 1 to groups of 8, little more
//! beyond) while the jackknife's groups grow with the trajectories. T, ⟨v⟩, n̄ and ⟨r²⟩ are each
//! atom's means over one trap period of the axis around each output (lib.rs `simulate`): its kinetic
//! energy sloshes into potential energy and back twice a period, so over a period the spread of T
//! between atoms is their energies' alone.
//!
//! The first standard errors stand on `FIRST_GROUPS` groups on any machine: the pilot's trajectories,
//! each a group of one (with the harmonic control, at the pilot's step), predict the need, and the
//! first batch is that many groups of `TASK`, at most the need the pilot predicts, at least the
//! device's width -- a machine of more threads runs it in fewer rounds, not with more trajectories.
//! Each later batch is the predicted remaining need -- the trajectories the errors so far say the
//! targets need, less those run --, at least the device's width and at most as many as have run: the
//! need is the largest of many outputs' estimated errors, biased up while few groups stand behind it,
//! so it is followed at most to twice the trajectories it rests on. No margin is added above the
//! prediction: a batch more costs the device's tail and the statistics, a few trajectories' time,
//! where a margin of the estimate's own noise would cost tens of percent of the trajectories.
//!
//! # Control variates
//! Each output's κ² and populations are estimated with control variates of exactly known means
//! (control.rs, stats.rs `estimate`): the harmonic twin's for T, always (it costs a few products per
//! output), and the held twin's for the populations. β is the regression on the control over every
//! trajectory so far, held when the jackknife leaves a group out: its errors -- what the stopping rule
//! takes -- are the corrected values' spread over the groups; where a control follows nothing, β ≈ 0.
//! The held
//! twin costs a trajectory of its own: it runs in the first batch, and is kept when the trajectories
//! that batch says the targets need with it -- the populations' or T's, whichever needs more --,
//! times its share of the work (counted as the step's, `held_work`), are fewer than without: where T
//! decides the trajectories its control of the populations saves none. The first batch's groups are
//! Latin hypercubes, so its T need is the run's, not inflated by independent draws' spread at early
//! times.
//!
//! # Regime
//! A point particle has no zero-point motion: below a few vibrational quanta per axis the method
//! cools too far. Against the 1D GMW references of the tweezer PGC (the atom on the line through
//! the focus; measured with v2026.11.2.6's engine, whose 3D PGC example this one reproduces within
//! its standard errors): heating (TW1, k_B T from 6 to 16 ħω over 10 ms) T within 6 % (mean +2 %) and the
//! populations within 0.03; cooling (TW2) T within 4.5 % down to k_B T ≈ 5 ħω, 6 % low at 4.6 ħω,
//! 7-10 % low at 4 ħω and 10-55 % low below 3.5 ħω, the populations within 0.045 but for the first
//! 50 µs of optical pumping (up to 0.066). (With x alone free, the reference's own motion: 3.4–7.7 % low at 5.6–4 ħω, the
//! engine's absorbed recoils (engine.rs # Absorption) at their incoherent second moment there; the rest is what the kicks leave out.) Against the 2D quantum trajectories of the same scene
//! (x and y free) within 8 % at 4.6 ħω and 15 % low at 3.1 ħω. Each output time says whether it
//! is in the regime k_B T ≥ `CLASSICAL` ħω on every axis (with T the engine's own, which lies below
//! the quantum T where the method cools too far, so the mark comes early rather than late).
//!
//! # Device
//! The batches run on the device asked for (`Device`): the CPU's threads, or the GPU (gpu/mod.rs),
//! which runs a batch's trajectories as the CPU does -- the same groups, initial states, random
//! streams and sums. The two devices make a solve's choices alike: the model, the ledger, the pilot
//! and the time step on the CPU in double precision either way, the statistics and their stopping
//! rule. A batch is whole groups: on the CPU at least one (its threads take the trajectories one at
//! a time, so a batch costs its trajectories and one tail whatever its size), and on the GPU whole
//! waves -- a group in each of its resident blocks --, a wave costing about what one trajectory does,
//! so that there the first batch is usually the last. A GPU that cannot run (none, an older driver, a
//! model beyond its kernel) leaves the solve on the CPU with the reason
//! (`Solution::device_note`); one that fails during the batches has the solve redone on the CPU.
use crate::engine::{column_norm,node_phase,series,Engine,Settings,Trajectory};
use crate::gpu;
use crate::ground::{ExcitedKind,Ground,Kind};
use crate::ledger::Ledger;
use crate::model::{Exact,Model,Trap,MOST_MIXED,MOST_STATES,NOT_LINDBLAD};
use crate::control::{Harmonic,Twin};
use crate::stats::{summarize,Estimate,Group,Means,Scales,KT_PER_UK};
use crate::{pilot_start,sample_states,simulate,Initial,Outcome,Variates,TASK};
use crate::{Complex,Request,STATES};
use rayon::prelude::*;
use std::time::Instant;

/// Where a solve's trajectories run (the request's `options.device`).
#[derive(Clone,Copy,Debug,PartialEq,Eq)]
pub enum Device{Cpu,Gpu}

/// The method's accuracy targets: T relative, populations and loss absolute.
pub const TEMPERATURE_TARGET:f64=0.10;
pub const POPULATION_TARGET:f64=0.05;
/// Statistics are run until this many standard errors lie within the targets.
pub const CONFIDENCE:f64=2.;
/// The integrator's order p (engine.rs: the flow, the kick and the internal state's half steps in Strang's
/// splitting): a step's bias over the run and its doubling's difference go as dt^p, Richardson's 2^p/(2^p − 1)
/// the difference's share of the bias.
pub const ORDER:f64=2.;
/// The step's share s of the run's room beside the a-priori estimates (module documentation, # Shares): 1/(1 + 2p), its
/// bias' cost-optimal share beside the statistics.
pub const STEP_SHARE:f64=1./(1.+2.*ORDER);
/// The power k of a total's weight beyond its target, (E/ε)^k (module documentation, # Shares): the run's work exponent
/// 2 + 1/p over the step's share, (1 + 2p)²/p.
pub const BEYOND:f64=(2.+1./ORDER)/STEP_SHARE;
/// The validated regime: k_B T ≥ CLASSICAL ħω along every axis (T the engine's own), where the 1D
/// and 2D quantum references put T within about 10 % (validation, see the module documentation).
pub const CLASSICAL:f64=4.;
/// The most trajectories a solve runs; a solve stopped there returns its result with the reason.
pub const MOST_TRAJECTORIES:u64=1<<14;
/// The most steps a trajectory takes over the run, a bound on the work as `MOST_TRAJECTORIES` is (a
/// solve takes at most their product): neither the first step nor the step search goes finer, and a
/// step it bounds that misses its shares is kept with its measured bias, the reason in the flag where
/// a total exceeds its target (`ledger`).
pub const MOST_STEPS:usize=1<<24;
/// The jackknife's groups behind a solve's first standard errors, on any machine (# Batches): the
/// pilot's trajectories, each a group of one, and at most the first batch's groups of `TASK` -- the
/// twin's decision and the first prediction stand on them (formerly two per thread of 16 threads).
pub const FIRST_GROUPS:u64=32;

pub const NOT_CONFINED:&str="光镊不约束原子：3D 运动计算光镊中的单个原子，需要在焦点处形成势阱的光镊";

/// Distributions along one axis on one grid, at some of the times.
#[derive(Clone,Debug,PartialEq)]
pub struct Distributions{pub grid:Vec<f64>,pub samples:Vec<usize>,pub probabilities:Vec<Vec<f64>>}

/// A 3D motion solve's result, at every time of the request (per lab axis).
/// Everything but `loss_probability` describes the atom still held.
#[derive(Clone,Debug)]
pub struct Solution{
    pub populations:Vec<Vec<f64>>,
    /// Per time, the largest standard error of the populations and the loss.
    pub sample_errors:Vec<f64>,
    pub temperature_uk:Vec<[f64;3]>,
    pub temperature_errors_uk:Vec<[f64;3]>,
    /// T_total = (T_x + T_y + T_z)/3 and its standard error (stats.rs `Estimate`).
    pub temperature_total_uk:Vec<f64>,
    pub temperature_total_errors_uk:Vec<f64>,
    pub mean_velocity_mps:Vec<[f64;3]>,
    /// ω_a/2π (kHz) at the focus of the reference state's trap per lab axis, and n̄ per axis relative to it.
    pub trap_frequency_khz:[f64;3],
    pub mean_occupation:Vec<[f64;3]>,
    pub loss_probability:Vec<f64>,
    /// Per axis: velocities (m/s) and positions (um from the focus).
    pub velocity:[Distributions;3],
    pub position:[Distributions;3],
    /// Per time: in the validated regime (k_B T ≥ CLASSICAL ħω on every axis).
    pub classical:Vec<bool>,
    pub trajectories:u64,
    /// The trajectories the statistics need by the latest batch's estimate: `trajectories` is the
    /// first whole batch at or beyond it (a batch at most triples the trajectories run).
    pub needed:u64,
    pub time_step_us:f64,
    pub method:String,
    pub reason:Option<String>,
    /// The a-priori approximations and their estimated errors (ledger.rs).
    pub approximations:Ledger,
    /// The device the trajectories ran on, and why it is not the one asked for (Chinese).
    pub device:Device,
    pub device_note:Option<String>,
}

/// An estimate in the validated regime: k_B T ≥ CLASSICAL ħω along every axis (ω per axis).
fn classical(e:&Estimate,frequencies:[f64;3])->bool{(0..3).all(|a|KT_PER_UK*e.temperature[a]>=CLASSICAL*frequencies[a])}

/// An output's statistical errors at CONFIDENCE: the populations' and the loss's largest (absolute)
/// and, where the method holds, T's largest relative one over the axes (None outside: a reference).
fn statistics(e:&Estimate,frequencies:[f64;3])->(f64,Option<f64>){
    let p=CONFIDENCE*e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x));
    let t=classical(e,frequencies).then(||CONFIDENCE*(0..3).map(|a|if e.temperature[a]>0.{e.temperature_error[a]/e.temperature[a]}else{0.}).fold(0.,f64::max));
    (p,t)
}
/// The room an error whose work goes as its −α-th power takes beside a floor its work does not move (module documentation,
/// # Shares): what the floor leaves of the target, or where that is less, the weighed cost's least αF/(k − α);
/// at most the target itself.
fn room_of(target:f64,floor:f64,exponent:f64)->f64{(target-floor).max(exponent*floor/(BEYOND-exponent)).min(target)}
/// The statistics' room (their trajectories go as S^−2) beside an output's a-priori estimates (`Accounts::apriori`) and the
/// step's bias: what they leave of the target, or their balance beyond 1 − 2/k of it (the output is then flagged).
fn room(target:f64,floor:f64)->f64{room_of(target,floor,2.)}
/// The weight of a total at its target (module documentation, # Shares): 1 within it, (E/ε)^k beyond.
fn beyond(target:f64,total:f64)->f64{(total/target).max(1.).powf(BEYOND)}
/// What a run predicts at a candidate step (module documentation, # Shares): the trajectories its rooms need --
/// per output after t = 0 and binding target (T where the pilot's estimate is in the regime) the pilot's statistic over its
/// room beside the a-priori total there and the step's bias, the largest squared, times the pilot's trajectories, at least a
/// group (`TASK`, a run's least) and at most `MOST_TRAJECTORIES` -- and the weight of the largest total at that need, its
/// statistics the pilot's scaled to it, and of the model's resonance (`resonance`, the elimination's a-priori bound over
/// the reach, which the ledger flags beyond the population target: the measured charge can miss where the light nears
/// resonance -- F5's retro-reflected tweezer folded whole at a 9 % bound, its charge 3 %, against its explicit rung).
fn predicted(statistics:&[(f64,Option<f64>)],apriori:&[[f64;2]],bias:[f64;2],binds:[bool;2],pilot:f64,resonance:f64)->(f64,f64){
    let terms:Vec<(f64,f64,f64)>=(1..statistics.len()).flat_map(|i|{let(p,t)=statistics[i];let a=apriori[i];
        [Some((POPULATION_TARGET,p,a[0]+bias[0])),t.filter(|_|binds[1]).map(|t|(TEMPERATURE_TARGET,t,a[1]+bias[1]))]}).flatten().collect();
    let need=(pilot*terms.iter().map(|&(target,s,floor)|(s/room(target,floor)).powi(2)).fold(0.,f64::max)).clamp(TASK as f64,MOST_TRAJECTORIES as f64);
    let weight=terms.iter().map(|&(target,s,floor)|beyond(target,floor+if s>0.{s*(pilot/need).sqrt()}else{0.})).fold(beyond(POPULATION_TARGET,resonance),f64::max);
    (need,weight)
}
/// The run's room beside the a-priori estimates -- the step's bias and the statistics together, their work going as
/// R^−(2+1/p) at their cost-optimal split (module documentation, # Shares): what the estimates leave of the target, or
/// beyond (1 − s) of it their balance A/(2p); at most the target. The step takes its share s (`tolerances`).
fn run_room(target:f64,apriori:f64)->f64{room_of(target,apriori,2.+1./ORDER)}
/// The largest ratio of the statistical errors to their room over every output after t = 0 (the
/// request's initial state, `start`): 1 meets the targets.
fn worst_ratio(estimates:&[Estimate],accounts:&Accounts,frequencies:[f64;3])->f64{let[p,t]=worst_ratios(estimates,accounts,frequencies);p.max(t)}
/// The same per kind: the populations' and the loss's, and T's.
fn worst_ratios(estimates:&[Estimate],accounts:&Accounts,frequencies:[f64;3])->[f64;2]{
    estimates.iter().enumerate().skip(1).map(|(i,e)|{
        let(p,t)=statistics(e,frequencies);let a=accounts.apriori[i];
        [p/room(POPULATION_TARGET,a[0]+accounts.bias[0]),t.map_or(0.,|t|t/room(TEMPERATURE_TARGET,a[1]+accounts.bias[1]))]
    }).fold([0.;2],|w,r|[w[0].max(r[0]),w[1].max(r[1])])
}

/// The distribution sample indices: evenly spread over the outputs, the first and last among them.
fn listed(outputs:usize)->Vec<usize>{
    let count=outputs.min(21);let mut v:Vec<usize>=(0..count).map(|i|((i*(outputs-1)) as f64/(count-1).max(1) as f64).round() as usize).collect();v.dedup();v
}

/// Histograms of a value of the samples (a coordinate along one axis) at the listed times on one
/// grid symmetric about 0 that holds all of them (the bins' centres; an odd count, 0 at the centre).
/// A listed time where no atom is held carries the one before (`assemble`); the first is t = 0,
/// where every atom is.
fn histograms(samples:&[Vec<([f64;3],[f64;3])>],listed:&[usize],value:&dyn Fn(&([f64;3],[f64;3]))->f64)->Distributions{
    let reach=samples.iter().flatten().map(|s|value(s).abs()).fold(0.,f64::max).max(f64::MIN_POSITIVE);
    let count=samples.iter().map(|s|s.len()).max().unwrap_or(0);
    // Bins: about the square root of the samples, odd, at least 21.
    let bins=(((count as f64).sqrt() as usize)|1).max(21);let width=2.*reach/(bins-1) as f64;
    let grid:Vec<f64>=(0..bins).map(|i|(i as f64-(bins/2) as f64)*width).collect();
    let mut last:Option<Vec<f64>>=None;
    let probabilities=samples.iter().map(|s|{
        if s.is_empty(){return last.clone().unwrap_or_else(||{let mut p=vec![0.;bins];p[bins/2]=1.;p});}
        let mut p=vec![0.;bins];
        for x in s{let i=((value(x)/width).round() as isize+(bins/2) as isize).clamp(0,bins as isize-1) as usize;p[i]+=1./s.len() as f64;}
        last=Some(p.clone());p
    }).collect();
    Distributions{grid,samples:listed.to_vec(),probabilities}
}

/// A time in the words of the app (μs below a millisecond, ms above).
fn when(us:f64)->String{if us<1e3{format!("{us:.0} μs")}else{format!("{:.2} ms",us/1e3)}}

/// The model's approximations (fixed before the run): the far components' fold, the couplings'
/// fold, the near light's dressing left out, the Gaussian beams' profiles and the tables.
fn model_ledger(model:&Model,request:&Request)->Ledger{
    let g=&model.ground;let mut ledger=Ledger::new();let(far,near,statics)=g.fold_estimates();
    let folded=g.components.iter().filter(|c|c.folded&&c.size>0.).count();
    ledger.add("远失谐分量（光镊、反旋项）",far,format!("二阶折叠（van Vleck 能移），{folded} 个分量"));
    // The drives' fold, its charge; H's couplings between excited states.
    let explicit:usize=g.clusters.iter().map(|c|c.excited.len()).sum();
    let count=|drive:bool|g.excited_items.iter().filter(|x|matches!(x.kind,ExcitedKind::Drive(_))==drive).fold((0,0),|(k,f),x|if x.folded{(k,f+1)}else{(k+1,f)});
    let(kept,fold)=count(true);
    let order=if model.fourth{"四阶折叠（误差界为饱和参数之和的平方）"}else{"二阶折叠（误差界为饱和参数之和）"};
    ledger.add(FOLD,fold_charge(model,g.charge),if explicit==0{format!("{fold} 个驱动全部{order}，饱和参数按光场图样平均")}
        else{format!("{fold} 个驱动{order}，{kept} 个显式保留、{explicit} 个激发态精确传播：按预计总耗时选取")});
    let(kept,fold)=count(false);
    if kept+fold>0{ledger.add("激发态之间的耦合（光镊混合超精细能级）",statics,format!("{fold} 个二阶折叠，{kept} 个显式保留"));}
    let(kept,fold)=g.items.iter().filter(|i|matches!(i.kind,Kind::Static{..}|Kind::Raman{..})).fold((0,0),|(k,f),i|if i.folded{(k,f+1)}else{(k+1,f)});
    let sizes:Vec<String>=g.clusters.iter().map(|c|c.dimension().to_string()).collect();
    ledger.add("基态之间的耦合（塞曼子能级、拉曼项、两组光的拍频）",near,format!("{fold} 对二阶折叠，{kept} 对显式保留；子空间 {}",sizes.join("+")));
    if model.dressing>0.{ledger.add("近共振光的相互缀饰（彼此的光移）",model.dressing,"各近共振光的光移全阶计入彼此的分母；同频多束光的干涉与缀饰的下一阶略去".into());}
    if g.unmixed_error>0.{ledger.add("近共振光的静态基态块（块内混合）",g.unmixed_error,"混合可略的块按各态自身能量（每态一个分解），块内非对角部分对各通道响应的改变按其界计入".into());}
    if model.fourth_dropped>0.{ledger.add("四阶折叠的快拍频项",model.fourth_dropped,"略去（与二阶相同的截断频率）".into());}
    let names=|beams:&[usize]|beams.iter().map(|&l|request.beams[l].name.as_str()).collect::<Vec<_>>().join(" + ");
    for &(u,error,_)in &model.profiles{ledger.add(&format!("「{}」的光强分布",names(&g.units[u].beams)),error,"取焦点处的光强".into());}
    for &u in &model.focused[1..]{ledger.add(&format!("「{}」的光强分布",names(&g.units[u].beams)),0.,"按位置插值（强度维度）".into());}
    // The interference the units of beams of one frequency leave out (ground.rs `coherence`).
    for x in &g.interference{
        let action=match(x.near,x.coherent){
            (true,true)=>"近共振光场按相干叠加精确计入；远失谐部分（反旋项、32 态以外能级的光移）按主偏振合成，偏离主偏振的剩余分量略去",
            (true,false)=>"近共振光场按相干叠加精确计入；远失谐部分（反旋项、32 态以外能级的光移）按各光束光强相加，其偏振干涉略去",
            (false,true)=>"同频光束按主偏振合成一个光场（驻波精确）；偏离主偏振的剩余分量略去",
            (false,false)=>"同频而偏振不同：远失谐光移按各光束光强相加，干涉项（偏振光晶格）略去"};
        ledger.add(&format!("「{}」的干涉",names(&x.beams)),x.estimate(g),action.into());
    }
    ledger.add("聚焦光的光强插值表",model.table_error,match(model.table_limit,model.table_stalled){
        (Some(n),true)=>format!("三次插值；网格加密不再减小误差，止于每维 {n} 个节点"),(Some(n),false)=>format!("三次插值；网格细化到工作量上限（每维 {n} 个节点）为止"),
        (None,_)=>format!("三次插值（每维 {} 个节点：再加密的代价超过其误差所省的轨迹）",model.tables.nodes)});
    ledger
}

/// A solve's error accounts so far: per output and target (the populations, T) the a-priori estimates'
/// total (the model's, and there the fold's charge as measured, the mean (kv/σ)² and the profiles at
/// the atoms) and the fold's charge in it (`fold_charges`), and the time step's bias on the populations
/// and on T.
struct Accounts{apriori:Vec<[f64;2]>,excited:Vec<[f64;2]>,bias:[f64;2]}

/// The ledger of a solve's estimates so far, and its accounts: the model's items, the elimination's
/// largest η², the largest mean (kv/σ)² and the profiles at the atoms. With `last`, the outputs whose
/// total estimated error -- the a-priori total, the step's bias and the statistics -- exceeds a
/// target are flagged; before it, those whose a-priori total and bias alone do.
fn ledger(model:&Model,base:&Ledger,estimates:&[Estimate],times:&[f64],allowance:f64,bias:[f64;2],frequencies:[f64;3],last:bool,memory:&Memory)->(Ledger,Accounts){
    let mut ledger=base.clone();let charge=fold_charge(model,model.ground.charge);
    // The fold's error, charged as measured along the trajectories: the folded excited population's 2η² at
    // each output with the populations' memory of the earlier ones (`Memory::charges`) -- what the fold's
    // bound takes on the paths they go (an atom that gathers where the light is bright, as red-detuned
    // molasses holds it at its lattice's bright sites, meets more than the pattern's mean). The a-priori
    // estimate over the reach decided the fold (`Candidate`).
    let measured=memory.charges(&estimates.iter().map(|e|fold_charges(model.fourth,2.*e.folded)).collect::<Vec<_>>());
    let profiles:Vec<f64>=estimates.iter().map(|e|e.profile).collect();let dopplers:Vec<f64>=estimates.iter().map(|e|e.doppler).collect();
    let largest=|v:&[f64]|v.iter().skip(1).copied().fold(0.,f64::max);
    ledger.add("多普勒与反冲频移",largest(&dopplers),format!("一阶（kv/σ）²；单步超过 {:.1e} 时用精确响应",allowance));
    if !model.profiles.is_empty(){ledger.add("聚焦光在原子处的强度（沿轨迹）",largest(&profiles),"按峰值强度的平面波".into());}
    let met=largest(&measured.iter().map(|m|m[0]).collect::<Vec<f64>>());
    if let Some(item)=ledger.items.iter_mut().find(|i|i.label==FOLD){
        item.action=format!("{}；计入沿轨迹实测的饱和参数 2η²{}（先验估计 {:.2}%{}）{}",item.action,if model.fourth{" 的平方"}else{""},100.*charge,if met>charge{"，实测高于先验估计"}else{""},
            if model.fourth{"；温度仍计 2η²（力按二阶折叠）"}else{""});item.estimate=met;
    }
    let peak:f64=model.profiles.iter().map(|p|p.1).sum();let fixed=base.used()-peak-charge-cut(base);
    let apriori:Vec<[f64;2]>=(0..estimates.len()).map(|i|if i==0{[0.;2]}else{measured[i].map(|m|fixed+peak.max(profiles[i])+dopplers[i]+m)}).collect();
    let accounts=Accounts{apriori,excited:measured,bias};
    // The outputs beyond a target: (output, the quantity, its total, the a-priori share, the statistics).
    let over:Vec<(usize,&str,f64,f64,f64)>=(1..estimates.len()).filter_map(|i|{
        let(p,t)=statistics(&estimates[i],frequencies);let a=accounts.apriori[i];let(p,t)=if last{(p,t)}else{(0.,t.map(|_|0.))};
        let population=(a[0]+bias[0]+p>POPULATION_TARGET).then_some((i,"布居",a[0]+bias[0]+p,a[0]+bias[0],p));
        let temperature=t.and_then(|t|(a[1]+bias[1]+t>TEMPERATURE_TARGET).then_some((i,"温度",a[1]+bias[1]+t,a[1]+bias[1],t)));
        population.or(temperature)}).collect();
    // Light near resonance where the atoms go (`resonance`).
    let resonance=resonance(model);
    // The initial ensemble's flag, on its own (`initial_flag`, the base ledger's): the loss and the populations
    // depend on its cut, T on its kinetic temperature.
    let initial=base.flagged.clone();
    ledger.flagged=match(resonance,initial){(Some(r),Some(i))=>Some(format!("{r}；{i}")),(r,i)=>r.or(i)};
    if !over.is_empty(){
        let(first,last_over)=(over[0].0,over[over.len()-1].0);
        let span=if over.len()==last_over-first+1&&last_over==estimates.len()-1{format!("{} 起",when(times[first]))}
            else{format!("{} 至 {} 之间 {} 个时刻",when(times[first]),when(times[last_over]),over.len())};
        let worst=over.iter().max_by(|a,b|(a.2/if a.1=="布居"{POPULATION_TARGET}else{TEMPERATURE_TARGET}).total_cmp(&(b.2/if b.1=="布居"{POPULATION_TARGET}else{TEMPERATURE_TARGET}))).unwrap();
        let target=if worst.1=="布居"{POPULATION_TARGET}else{TEMPERATURE_TARGET};
        let over=format!("{span}估计误差超出目标：{}误差最大 {:.1}%（目标 {:.0}%；近似与步长 {:.1}%，其中消去激发态 {:.1}%；统计 {:.1}%）",
            worst.1,100.*worst.2,100.*target,100.*worst.3,100.*accounts.excited[worst.0][if worst.1=="布居"{0}else{1}],100.*worst.4);
        ledger.flagged=Some(match ledger.flagged.take(){Some(r)=>format!("{r}；{over}"),None=>over});
    }
    (ledger,accounts)
}

/// Solves a request on the device asked for (module documentation, # Device). `checkpoint` receives
/// every batch's complete result and the estimated wall time (ms) the rest will take.
pub fn solve(request:&Request,device:Device,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,String>{
    // The GPU starts in the background while the model is made.
    if device==Device::Gpu{gpu::prepare();}
    match run(request,device,checkpoint){
        Err(Stop::Gpu(error))=>{
            let mut solution=run(request,Device::Cpu,checkpoint).map_err(Stop::message)?;
            solution.device_note=Some(format!("GPU 计算出错（{error}）：本次在 CPU 上计算"));Ok(solution)
        }
        other=>other.map_err(Stop::message),
    }
}

/// Why a solve stopped: its own error, or its GPU's (the solve is then redone on the CPU).
enum Stop{Error(String),Gpu(String)}
impl From<String> for Stop{fn from(error:String)->Stop{Stop::Error(error)}}
impl Stop{fn message(self)->String{match self{Stop::Error(e)|Stop::Gpu(e)=>e}}}

fn run(request:&Request,device:Device,checkpoint:&mut dyn FnMut(&Solution,Option<f64>))->Result<Solution,Stop>{
    let times=&request.times_us;let outputs=times.len();
    if outputs<2||times[0]!=0.{return Err(Stop::Error("A 3D motion solve needs at least two times, the first at 0".into()));}
    let interval=times[1]-times[0];
    if times.windows(2).any(|w|((w[1]-w[0])-interval).abs()>1e-9*interval){return Err(Stop::Error("A 3D motion solve needs evenly spaced times".into()));}
    let listed=listed(outputs);let seed=0x3D_5EED;
    let pilot=FIRST_GROUPS;
    let duration=times[outputs-1];
    // The fold of the excited states (module documentation, # The fold's choice): the rule's candidates from
    // all folded down, their order the a-priori estimates'; the pilot runs the first, each candidate's charge
    // is what its folded drives meet at the pilot's states (`measured_charge`), and of those whose charge
    // leaves the statistics room the one of least predicted cost is taken.
    let mut ladder=Ground::allowances(request)?.into_iter();let mut failed=None;
    // The fourth order is a candidate on the CPU (the GPU's kernel does not read its tables' slabs yet,
    // gpu/program.rs).
    let fourth=device==Device::Cpu;
    let Some(mut pilot_candidate)=next_candidate(request,&mut ladder,false,&mut failed)else{return Err(Stop::Error(failed.unwrap_or_else(||"No model of the request".into())))};
    pilot_candidate.settle();
    // A generator that gains on the ladder's first rung, every drive folded, is the fold's construction failing
    // (model.rs `Model::lindblad`), not a regime to escalate into explicit states: the solve stops with why.
    if let Some(e)=failed.as_ref().filter(|e|e.starts_with(NOT_LINDBLAD)){return Err(Stop::Error(e.clone()));}
    // The reference state: the largest initial population; its trap sets ω per axis.
    let(trap,depth)=pilot_candidate.reference();
    // A trap the model gives only where the model holds: a strong near beam whose light shift anti-traps the
    // reference state is the elimination's resonance first (the validation's F8).
    if trap.frequencies.iter().any(|&w|!(w>0.)){return Err(Stop::Error(match resonance(&pilot_candidate.model){Some(r)=>format!("{r}：光镊的势阱无从判断"),None=>NOT_CONFINED.into()}));}
    // The search's first step: the light pattern's phase k_max v_T per step at sqrt(s × the temperature
    // target), or without light the trap's own ω.
    let thermal=(KT_PER_UK*request.temperature_uk*pilot_candidate.model.hbar_over_mass).sqrt();
    let pattern=pilot_candidate.model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
    let rate=if pattern>0.{pattern}else{trap.frequencies.iter().copied().fold(0.,f64::max)};
    // The finest step the work bound allows (`MOST_STEPS`), in steps per interval.
    let most=(MOST_STEPS/(outputs-1)).max(1);
    let first=((interval*rate/first_phase()).ceil() as usize).clamp(1,most);
    // The loss's region is the reference trap's (engine.rs `Settings::region`).
    let plain=|steps:usize,truncation:f64|{let mut s=Settings::new(interval/steps as f64,&trap,truncation,true);s.region=depth;s};
    // The pilot's exponentials to double precision; the beats its exact steps leave out within the run's share,
    // the step's population budget over a waiting time between jumps (engine.rs `Segment::resolve`, `running`):
    // before it measures one, the one its atoms start with -- the inverse of its initial states' mean jump rate,
    // the no-jump decay ⟨ψ|i(H − H†)|ψ⟩/⟨ψ|ψ⟩ at their points, at most the run. Where the atoms then go dark the
    // waiting grows and the pilot's share is looser by as much: a beat it leaves out errs by at most that growth
    // times the run's share in an atom's populations, which the pilot's uses take far coarser -- its states
    // sample the step's check (at the run's settings), its fold's charge is the percent of its 2η², its jumps and
    // statistics their counts' --, and the run takes the waiting time the pilot measures.
    let piloting=|steps:usize,waiting:f64|{let mut s=plain(steps,f64::EPSILON);s.beats=STEP_SHARE*STEP_SHARE*POPULATION_TARGET*s.dt/waiting;s.horizon=waiting;s};
    let starts:Vec<Trajectory>=(0..pilot).map(|index|pilot_start(&pilot_candidate.model,&plain(first,f64::EPSILON),&pilot_candidate.initial,seed,index)).collect();
    let start_rate=starts.iter().map(|t|{let m=&pilot_candidate.model;let g=m.generator(t.cluster,&m.point(t.r,t.kappa),0.);let n=g.n;
        let(mut form,mut norm)=(Complex::ZERO,0.);for a in 0..n{norm+=t.psi[a].norm_sqr();for b in 0..n{form+=t.psi[a].conj()*g.hamiltonian[a*n+b]*t.psi[b];}}-2.*form.im/norm}).sum::<f64>()/pilot as f64;
    let start_waiting=if start_rate>0.{(1./start_rate).min(duration)}else{duration};
    // The trap's slowest axis's period: the span of T's step error (`orbits`).
    let period=std::f64::consts::TAU/trap.frequencies.iter().copied().fold(f64::INFINITY,f64::min);
    // A run's settings at `steps` per interval: its exponentials summed to the step's share of its own population
    // tolerance, s² ε dt/waiting, and the beats left out within it over the waiting time; the first order in the
    // Doppler shift within `allowance` (`doppler_allowance`).
    let running=|steps:usize,waiting:f64,allowance:f64|{let dt=interval/steps as f64;let mut s=plain(steps,STEP_SHARE*STEP_SHARE*POPULATION_TARGET*dt/waiting);s.doppler=allowance;s.horizon=waiting;s};
    // The pilot's step (module documentation, # Time step): the bound rule's search on its own initial states from the
    // first step, at the a-priori estimates of its model -- the largest step met there, else the search's last.
    let steps={let c=&pilot_candidate;let allowance=doppler_allowance(&c.base,&c.model);
        let binds=[true,Binding::of(&starts,c.model.hbar_over_mass,trap.lab).temperature];let tolerances=tolerances(binds[1],&c.charges.iter().map(|x|x.map(|x|c.others()+x)).collect::<Vec<_>>());
        let probes=bound_probes(&c.model,&starts);
        let doubled=|k:usize|{let r=step_excess(&Engine::new(&c.model,running(k,start_waiting,allowance)),&starts,start_waiting,&probes,period,0.,tolerances);Check{steps:k,population:r[0],loss:r[1],motion:r[2]}};
        let mut bound=vec![];search(first,most,&[],&doubled,None,&mut bound);
        bound.iter().filter(|c|c.met()).map(|c|c.steps).min().unwrap_or(bound[bound.len()-1].steps)};
    let pilot_harmonic=Harmonic::new(&pilot_candidate.model,&piloting(steps,start_waiting),&pilot_candidate.initial,steps,outputs);
    let(pilot_states,pilot_jumps,pilot_groups)=sample_states(&pilot_candidate.model,&piloting(steps,start_waiting),&pilot_candidate.initial,seed,pilot,steps,outputs,&pilot_harmonic);
    // The fold's memory (module documentation, # Ledger): per pilot atom its jumps at the outputs; the pilot's
    // initial states, where the charge starts.
    let memory=Memory::of(&pilot_states,interval,outputs);
    let charge_of=|from:&Model,m:&Model|measured_charge(from,m,&pilot_states,&starts,&memory,interval,outputs);
    let charge=charge_of(&pilot_candidate.model,&pilot_candidate.model);pilot_candidate.charged(charge);
    // The targets the fold's choice weighs (`Binding`): the populations' always, T's where the pilot's atoms
    // reach the method's regime -- elsewhere T is a reference only, and its charge must not choose.
    let binds=[true,Binding::of(&pilot_states,pilot_candidate.model.hbar_over_mass,trap.lab).temperature];
    // The pilot's statistics per output (its trajectories each a group of one, with the harmonic control): CONFIDENCE
    // standard errors, the populations' and the loss's largest and T's where it is in the regime (`statistics`). The
    // trajectories a model needs are (statistic/room)² times the pilot's at the output and target that bind (`predicted`).
    let pilot_statistics:Vec<(f64,Option<f64>)>={
        let scales=Scales{hbar_over_mass:pilot_candidate.model.hbar_over_mass,frequencies:trap.lab};
        let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        let mut e=summarize(&pilot_groups,&scales,&means);start(&mut e[0],&pilot_candidate.model,&pilot_candidate.initial);
        e.iter().map(|x|statistics(x,trap.lab)).collect()
    };
    // The relative standard error of the need the pilot predicts (`need_error`): the margin a finer step must save
    // by (`kept`).
    let need={let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        need_error(&pilot_groups,&Scales{hbar_over_mass:pilot_candidate.model.hbar_over_mass,frequencies:trap.lab},&means,&pilot_candidate.model,&pilot_candidate.initial,trap.lab,binds)};
    let pilot_waiting=pilot as f64*duration/(pilot_jumps.max(1) as f64);let pilot_horizon=pilot_waiting.min(duration);
    // The beats the pilot's exact steps left out (engine.rs `Segment::resolve`): their bound per unit time, at
    // most the run's share over its waiting time (`booked_beats`).
    let pilot_beats=booked_beats(&pilot_states,pilot_waiting);
    // A candidate's weighed cost on the pilot's states carried into its clusters (module documentation, # The fold's
    // choice): its step check there (`step_check`), every step weighed by its counted work per interval times the
    // trajectories its rooms at its bias need and the weight of the totals they leave (`predicted`) over the run; at the
    // step it keeps, that cost, the check's own work, its work per interval, its bias and its a-priori totals per output.
    let evaluate=|c:&Candidate,states:&[Trajectory],begun:&[Trajectory]|->Costed{
        let base=gained(&c.base,&c.model,states);let(others,allowance)=(unfolded(&base,&c.model),doppler_allowance(&base,&c.model));
        let settings=|s:usize|running(s,pilot_waiting,allowance);let parts=state_work(&c.model,states,begun);
        let counted:std::cell::RefCell<Vec<(usize,f64)>>=Default::default();
        let work=|s:usize|->f64{if let Some(&(_,w))=counted.borrow().iter().find(|x|x.0==s){return w;}
            let w=interval_work(&c.model,&parts,&settings(s),interval,pilot_waiting);counted.borrow_mut().push((s,w));w};
        let apriori:Vec<[f64;2]>=c.charges.iter().map(|x|x.map(|x|others+x+pilot_beats*pilot_horizon)).collect();
        let tolerances=tolerances(binds[1],&apriori);let bias=|k:&Check|{let r=k.ratios();[r[0]*tolerances[0],r[1]*tolerances[1]]};
        let cost=|k:&Check|{let(n,weight)=predicted(&pilot_statistics,&apriori,bias(k),binds,pilot as f64,c.model.resonance);work(k.steps)*n*weight*(outputs-1) as f64};
        let checked=step_check(&c.model,states,&settings,steps,most,pilot_waiting,pilot_horizon,pilot as usize,&pilot_groups,depth,period,tolerances,&cost,need);
        let chosen=checked.chosen;
        Costed{cost:cost(&chosen),check:checked_work(&checked,states.len(),pilot as usize,pilot_horizon,interval,period,&work),work:work(chosen.steps),bias:bias(&chosen),apriori,checked}
    };
    // The candidates (module documentation, # The fold's choice), each costed where the pilot kept states
    // (atoms all lost before the first output measure none): the pilot's model and its fourth order on the
    // pilot's own states, then the ladder's on them carried into its clusters (`relaxed`). A rung is costed while
    // its step check's least work and the least work any step leaves it -- the multirate grid's nodes and the
    // jumps (`least_work`), or a costed earlier rung's work at its step where larger: a later rung keeps more
    // states explicit, and costs more per step and per node -- times the fewest trajectories it can need and their
    // weight (its totals with no step bias) could still cost less: its check is spent whether it is taken or not.
    let measured=!pilot_states.is_empty();
    let pilot_cost=measured.then(||evaluate(&pilot_candidate,&pilot_states,&starts));
    // The fourth order is built where it can pay its check and run: its work per interval at least the pilot's
    // model's at its step (its generator adds the fourth order's), its check's at least that model's, its totals
    // at least those of its least charges -- per output the pilot's T charge squared for the populations
    // (`fold_charges`; a mean of squares is at least the square of the mean) with no step bias, and T's the second
    // order's with the pilot's model's T bias: the fourth order folds the force at the second, so that T's charge
    // and the motion's step error are the second order's.
    let promise=pilot_cost.as_ref().map_or(true,|p|{
        let apriori:Vec<[f64;2]>=p.apriori.iter().zip(&pilot_candidate.charges).map(|(a,c)|[a[0]-c[0]+fold_charges(true,c[1])[0],a[1]]).collect();
        let(n,weight)=predicted(&pilot_statistics,&apriori,[0.,p.bias[1]],binds,pilot as f64,pilot_candidate.model.resonance);p.work*n*weight*((outputs-1) as f64)+p.check<p.cost});
    let fourth_cost=if fourth&&measured&&promise{Candidate::new(request,pilot_candidate.allowance,true).ok().map(|mut c|{c.charged(charge_of(&pilot_candidate.model,&c.model));
        let into=|v:&[Trajectory]|->Vec<Trajectory>{v.iter().map(|t|carried(&pilot_candidate.model,&c.model,t)).collect()};
        let(states,begun)=(into(&pilot_states),into(&starts));let costed=evaluate(&c,&states,&begun);(costed,c,states,begun)})}else{None};
    let mut least=pilot_cost.as_ref().map_or(f64::INFINITY,|p|p.cost).min(fourth_cost.as_ref().map_or(f64::INFINITY,|f|f.0.cost));
    // The ladder's rungs where the folded models' cost could fall by more than the prediction's uncertainty without the
    // estimates a rung can change: at the pilot's model's step, its totals with those no candidate changes alone
    // (`Candidate::fixed`) and the beats. Below that no rung -- more of the model explicit, more work per step -- can win
    // (DEFAULT's, THREE's: a near tie of explicit and folded at 1.3 s of the startup's checks).
    let open=pilot_cost.as_ref().is_some_and(|p|{let fixed=vec![[pilot_candidate.fixed()+pilot_beats*pilot_horizon;2];outputs];
        let(n,weight)=predicted(&pilot_statistics,&fixed,p.bias,binds,pilot as f64,0.);least>p.work*n*weight*(outputs-1) as f64*(1.+need)});
    let(mut rung,mut seen,mut shape)=(None,0_f64,vec![]);
    if open{
        while let Some(mut c)=next_candidate(request,&mut ladder,false,&mut failed){
            // A rung whose closure folds the same drives as the last one's is the same model.
            let folds:Vec<bool>=c.model.ground.excited_items.iter().map(|x|x.folded).collect();if folds==shape{continue;}shape=folds;
            c.charged(charge_of(&pilot_candidate.model,&c.model));
            let at=running(steps,pilot_waiting,doppler_allowance(&c.base,&c.model));
            let into=|v:&[Trajectory]|relaxed(&c.model,&at,v.iter().map(|t|carried(&pilot_candidate.model,&c.model,t)).collect());
            let(states,begun)=(into(&pilot_states),into(&starts));let parts=state_work(&c.model,&states,&begun);
            let apriori:Vec<[f64;2]>=c.charges.iter().map(|x|x.map(|x|c.others()+x+pilot_beats*pilot_horizon)).collect();
            let(fewest,weight)=predicted(&pilot_statistics,&apriori,[0.;2],binds,pilot as f64,c.model.resonance);
            let floor=|work:f64|work.max(seen)*fewest*weight*(outputs-1) as f64;
            // Its check's least: one doubling at the first step (`checked_step`). A rung is taken only for a saving beyond
            // the need's error: one whose floor cannot make it is passed over with its check unspent.
            let probes=bound_probes(&c.model,&states);
            let check=checked_step(states.len(),probes.len(),steps,interval,period)*interval_work(&c.model,&parts,&at,interval,pilot_waiting)/steps as f64;
            if floor(least_work(&c.model,&parts,&at,interval,pilot_waiting))*(1.+need)+check>=least{break;}
            // That doubling, made: the motion's and the loss's criteria, as dt², bound the steps the check can keep from below
            // (the population's bound may yet be measured smaller), and the work there any step's it keeps -- the explicit
            // states' exact steps resolve what the folded model's step left to the fold (CONTROL_CYCLE's first rung: 118 steps
            // per interval against the folded 34, its evaluation ten CPU-s for a cost twice the least's).
            let tolerances=tolerances(binds[1],&apriori);
            let r=step_excess(&Engine::new(&c.model,at.clone()),&states,pilot_waiting,&probes,period,escape(&c.model,&states,&pilot_groups,depth),tolerances);
            let fewest_steps=((steps as f64*r[1].max(r[2]).sqrt()).ceil() as usize).clamp(1,most);
            // There the motion's and the loss's biases, as dt², and the trajectories they leave the statistics: the rung's
            // cost at the largest step its check can keep -- a near tie (RETRO's ladder, ten rungs within its error of the
            // first) is passed over unchecked; one that saves beyond the error is costed.
            let x=(steps as f64/fewest_steps as f64).powi(2);
            let(n,weight)=predicted(&pilot_statistics,&apriori,[r[1]*x*tolerances[0],r[2]*x*tolerances[1]],binds,pilot as f64,c.model.resonance);
            if interval_work(&c.model,&parts,&running(fewest_steps,pilot_waiting,at.doppler),interval,pilot_waiting).max(seen)*n*weight*(outputs-1) as f64*(1.+need)+check>=least{break;}
            let costed=evaluate(&c,&states,&begun);seen=seen.max(costed.work);
            // A rung keeps more of the model explicit: taken only for a saving beyond the prediction's own uncertainty.
            if costed.cost*(1.+need)<least{least=costed.cost;rung=Some((c,states,begun,costed.checked));}
        }
    }
    // The choice: the least weighed cost, one keeping more of the model only for a saving beyond the need's error; with
    // none costed, the pilot's model. Its run takes the pilot's states carried into its clusters, the pilot's statistics
    // and jumps the prediction's (the models' rates differ by the fold's S, the physical atoms' jumps alike), and the step
    // its check took on them: no pilot runs again.
    let fourth_wins=fourth_cost.as_ref().is_some_and(|f|pilot_cost.as_ref().map_or(true,|p|f.0.cost*(1.+need)<p.cost));
    let(chosen,states,begun,known)=if let Some((c,states,begun,checked))=rung{(c,states,begun,Some(checked))}
        else if fourth_wins{let(f,c,states,begun)=fourth_cost.unwrap();(c,states,begun,Some(f.checked))}
        else{(pilot_candidate,pilot_states,starts,pilot_cost.map(|p|p.checked))};
    let(jumps,pilot_groups)=(pilot_jumps,pilot_groups);
    let mut chosen=chosen;chosen.settle();
    let Candidate{model,initial,base,charges,..}=chosen;
    // The generator against its jumps at the pilot's states (`gain`): booked where its decay falls below
    // the jumps' rate.
    let mut base=gained(&base,&model,&states);
    // The beats the exact steps left out (engine.rs `Segment::resolve`), booked as their bound per unit time
    // over the horizon: within the exponentials' share by construction (`booked_beats`).
    let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    let waiting=pilot as f64*duration/(jumps.max(1) as f64);let horizon=waiting.min(duration);
    let pilot_beats=booked_beats(&states,waiting);
    let allowance=doppler_allowance(&base,&model);
    let settings=|steps:usize|running(steps,waiting,allowance);
    // The step (`step_check`; on the pilot's states as the choice checked it): the a-priori total the steps'
    // costs take -- the ledger's estimates with the fold's charge as the pilot measured it. The motion's criteria
    // hold T's step error within T's share where T binds (`Binding`, at the chosen model's pilot), and within the
    // T target itself where T is a reference only: its share would refine for a value the targets do not take,
    // while a step that leaves the motion unresolved still moves the loss (`tolerances`).
    let binds=[true,Binding::of(&states,model.hbar_over_mass,trap.lab).temperature];
    let apriori:Vec<[f64;2]>=charges.iter().map(|x|x.map(|c|unfolded(&base,&model)+c+pilot_beats*horizon)).collect();let parts=state_work(&model,&states,&begun);
    let checked=match known{Some(checked)=>checked,None=>{let tolerances=tolerances(binds[1],&apriori);
        let cost=|k:&Check|{let r=k.ratios();let(n,weight)=predicted(&pilot_statistics,&apriori,[r[0]*tolerances[0],r[1]*tolerances[1]],binds,pilot as f64,model.resonance);
            interval_work(&model,&parts,&settings(k.steps),interval,waiting)*n*weight};
        step_check(&model,&states,&settings,steps,most,waiting,horizon,pilot as usize,&pilot_groups,depth,period,tolerances,&cost,need)}};
    let tolerances=checked.tolerances;
    let chosen=checked.chosen;
    let(steps,ratios)=(chosen.steps,chosen.ratios());
    let dt=interval/steps as f64;let settings=settings(steps);
    // A step the work bound set that misses its shares: why, for the flag.
    let bounded=(steps==most&&!chosen.met()).then(||format!("步数已达上限（每条轨迹 {MOST_STEPS} 步）：步长 {dt:.3e} μs 的误差为其份额的 {:.1} 倍",chosen.excess()));
    // The step's bias: per target its tolerance times its measured excess there.
    let bias=[ratios[0]*tolerances[0],ratios[1]*tolerances[1]];
    // The device: the GPU when asked for and able, the model on it; else the CPU, with why.
    let(gpu,note)=match device{
        Device::Cpu=>(None,None),
        Device::Gpu=>match gpu::gpu().and_then(|g|{let program=g.lock().unwrap_or_else(|e|e.into_inner()).program(&model,&settings)?;Ok((g,program))}){
            Ok(on)=>(Some(on),None),
            Err(reason)=>(None,Some(format!("{reason}：本次在 CPU 上计算"))),
        },
    };
    let used=if gpu.is_some(){Device::Gpu}else{Device::Cpu};
    // On the GPU the exact response leaves the near light's dressing out (gpu/program.rs): its bound.
    if gpu.is_some()&&model.undressed>0.{base.add("GPU 精确响应中的基态静态耦合",model.undressed,"多普勒一阶之外的精确响应按各基态自身能量计算，未计入基态静态块的本征基（GPU）".into());}
    // The control variates (control.rs): the harmonic twin's, and the held twin where the trap beam's
    // intensity is the pilot's mean, when the generator is static.
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    let intensity=states.iter().map(|t|model.geometry[model.trap_beam].intensity(t.r)).sum::<f64>()/states.len().max(1) as f64;
    let mut twin=Twin::new(&model,intensity);
    let twin_means=twin.as_ref().map(|t|t.means(&model,&request.initial_density,interval,outputs));
    let means=|held:bool|->Vec<Means>{(0..outputs).map(|i|Means{harmonic:harmonic.mean[i],twin:twin_means.as_ref().filter(|_|held).map(|m|m[i])}).collect()};
    let batch=|first:u64,count:u64,twin:Option<&Twin>|->Result<Outcome,Stop>{
        let variates=Variates{harmonic:&harmonic,twin};
        match &gpu{
            None=>Ok(simulate(&model,&settings,&initial,seed,first,count,TASK,steps,outputs,&listed,Some(&variates))),
            Some((g,program))=>g.lock().unwrap_or_else(|e|e.into_inner()).simulate(program,&model,&settings,&initial,seed,first,count,TASK,steps,outputs,&listed,Some(&variates))
                .map_err(Stop::Gpu),
        }
    };
    // A batch is whole groups: on the CPU at least one, on the GPU a group in each of its resident
    // blocks -- such a batch costs about what one trajectory does --, a whole number of its widths.
    let width=match &gpu{Some((g,_))=>g.lock().unwrap_or_else(|e|e.into_inner()).width(1),None=>TASK};
    let unit=if gpu.is_some(){width}else{TASK};
    // The first batch: FIRST_GROUPS groups, at most the pilot's predicted need (its trajectories each
    // a group, with the harmonic control), at least the device's width.
    let pilot_need={
        let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        let mut e=summarize(&pilot_groups,&scales,&means);start(&mut e[0],&model,&initial);
        let(_,accounts)=ledger(&model,&booked(&base,pilot_beats*horizon),&e,times,allowance,bias,trap.lab,false,&memory);
        (pilot as f64*worst_ratio(&e,&accounts,trap.lab).powi(2)).min(MOST_TRAJECTORIES as f64).ceil() as u64
    };
    let clock=Instant::now();
    let count=(FIRST_GROUPS*TASK).min(pilot_need).max(width).div_ceil(unit)*unit;
    let first=batch(0,count,twin.as_ref())?;
    let wall=clock.elapsed().as_secs_f64();
    // The held twin is kept where its control pays for itself: the trajectories the first batch says
    // the targets need with it, times its share of the work (counted as the step's, `held_work`),
    // against those they need without.
    if twin.is_some(){
        let needed=|means:&[Means]|{let mut e=summarize(&first.groups,&scales,means);start(&mut e[0],&model,&initial);
            let(_,accounts)=ledger(&model,&booked(&base,pilot_beats.max(first.beats)*horizon),&e,times,allowance,bias,trap.lab,false,&memory);worst_ratio(&e,&accounts,trap.lab).powi(2)};
        let share=held_work(&parts,&settings,interval,waiting)/interval_work(&model,&parts,&settings,interval,waiting);
        if needed(&means(true))*(1.+share)>=needed(&means(false)){twin=None;}
    }
    let means=means(twin.is_some());
    // Batches until the targets are met.
    let mut outcome=first;let mut done=count;let mut per_trajectory=wall/count as f64;
    loop{
        let mut estimates=summarize(&outcome.groups,&scales,&means);start(&mut estimates[0],&model,&initial);
        let(_,accounts)=ledger(&model,&booked(&base,pilot_beats.max(outcome.beats)*horizon),&estimates,times,allowance,bias,trap.lab,false,&memory);
        let ratio=worst_ratio(&estimates,&accounts,trap.lab);
        let wanted=(done as f64*ratio*ratio).ceil() as u64;let needed=wanted.min(MOST_TRAJECTORIES);
        let(mut ledger,_)=ledger(&model,&booked(&base,pilot_beats.max(outcome.beats)*horizon),&estimates,times,allowance,bias,trap.lab,needed<=done,&memory);
        if let(Some(note),Some(flag))=(&bounded,&ledger.flagged){ledger.flagged=Some(format!("{note}；{flag}"));}
        let mut solution=assemble(&model,&estimates,&outcome,&listed,times,trap.lab,done,dt,&request.initial_density,ledger,twin.is_some());
        solution.device=used;solution.device_note=note.clone();solution.needed=wanted;
        if needed<=done{
            if ratio>1.{solution.reason=Some(format!("统计误差为目标的 {ratio:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}"));}
            return Ok(solution);
        }
        checkpoint(&solution,Some(per_trajectory*(needed-done) as f64*1e3));
        let next=next_batch(needed,done,width,unit);
        let clock=Instant::now();
        let more=batch(done,next,twin.as_ref())?;
        per_trajectory=clock.elapsed().as_secs_f64()/next as f64;
        outcome.groups.extend(more.groups);for(a,b)in outcome.samples.iter_mut().zip(more.samples){a.extend(b);}outcome.jumps+=more.jumps;outcome.doppler=outcome.doppler.max(more.doppler);outcome.beats=outcome.beats.max(more.beats);
        done+=next;
    }
}

/// The bound per unit time of the beats a pilot's exact steps left out (engine.rs `Segment::resolve`), at most the
/// run's share over its waiting time: the run leaves out beats within that share by construction, and the pilot's
/// share is looser where its horizon was shorter than the waiting time it measured (`piloting`).
fn booked_beats(states:&[Trajectory],waiting:f64)->f64{states.iter().map(|t|t.beats).fold(0.,f64::max).min(STEP_SHARE*STEP_SHARE*POPULATION_TARGET/waiting)}

/// The model's ledger with the beats the exact steps left out, `bound` their largest over the horizon.
fn booked(base:&Ledger,bound:f64)->Ledger{
    let mut ledger=base.clone();
    if bound>0.{ledger.add("多速率步略去的残余拍频",bound,"整步影响之界（Duhamel 积分的分部积分）在指数截断份额内的拍频对略去，其余按 CF4 精确传播".into());}
    ledger
}

/// The elimination's a-priori bound over the atoms' reach (model.rs `Model::resonance`) beyond the population
/// target: light near resonance somewhere the atoms go -- the measured η² can miss it, and the model's
/// trap there is no trap's.
fn resonance(model:&Model)->Option<String>{
    (model.resonance>POPULATION_TARGET).then(||format!("光在原子所到之处接近共振：消去激发态的先验界（基态的激发态权重）达 {:.0}%（目标 {:.0}%），近共振处消去激发态不成立",
        100.*model.resonance,100.*POPULATION_TARGET))
}

/// The ledger's item of a generator whose decay falls below its jumps' rate (`gain`).
const GAIN:&str="生成元的增益";
/// A model's ledger with its gain at sampled states booked where it has one (`gain`).
fn gained(base:&Ledger,model:&Model,states:&[Trajectory])->Ledger{
    let mut ledger=base.clone();let gain=gain(model,states);
    if gain>0.{ledger.add(GAIN,gain,"沿试探轨迹测得：无跳跃的衰减低于跳跃率（i(H − H†) − ΣL†L 的负本征值对最大跳跃率之比）".into());}
    ledger
}
/// A model's ledger's estimates but the fold's charge and the initial ensemble's cut: what the fold's charge
/// and the step's bias add to (`Accounts::apriori`).
fn unfolded(base:&Ledger,model:&Model)->f64{base.used()-cut(base)-fold_charge(model,model.ground.charge)}
/// The first order in the Doppler shift's allowance: the step's share of the run's room beside the model's other
/// estimates on the population target (`run_room`; the fold's charge takes its own) -- a truncation per step, as the
/// step's own error; its largest met along the trajectories is booked, and one hot atom's must not take the run's room.
fn doppler_allowance(base:&Ledger,model:&Model)->f64{STEP_SHARE*run_room(POPULATION_TARGET,unfolded(base,model))}
/// A model's gain at sampled states (model.rs `Model::decay_balance`, at most FIRST_GROUPS × 8 of them): the
/// largest −λ of D = i(H − H†) − Σ L†L beyond its eigenvalues' round-off, relative to the largest jump rate
/// met -- the relative error of the jump rate where the no-jump decay misses it (a decay no jump carries,
/// the validation's F10 before the near light's dressing); 0 for a Lindblad generator, unbounded for a
/// gain without jumps.
fn gain(model:&Model,states:&[Trajectory])->f64{
    let every=(states.len()/(FIRST_GROUPS as usize*8)).max(1);
    let(gain,rate)=states.par_iter().step_by(every).map(|t|{let(low,rate,scale)=model.decay_balance(t.cluster,&model.point(t.r,t.kappa),t.time);
        let n=model.dimension(t.cluster) as f64;((-low-64.*f64::EPSILON*n*scale).max(0.),rate)}).reduce(||(0.,0.),|a,b|(a.0.max(b.0),a.1.max(b.1)));
    if gain==0.{0.}else if rate>0.{gain/rate}else{f64::INFINITY}
}

/// The ledger's item of the drives' fold (its charge a priori; the solve's ledger charges it as measured).
const FOLD:&str="消去激发态";
/// The ledger's item of the initial ensemble's cut (ensemble.rs): the share of the traps' thermal
/// distributions over their regions (U_h ≤ D, Maxwell's momenta) that the bound ensembles leave out --
/// the unbound atoms and the lowering near the trap's top. The temperature asked for against the trap
/// sets it; no choice of the solve changes it, so it stays out of the totals the solve's choices take
/// (the fold's room, the Doppler allowance, the statistics' room) and out of the population headline of
/// the flag: beyond the budget it is flagged on its own (`initial_flag`).
const CUT:&str="初始热分布（束缚态之外）";
/// The initial ensemble's flag, every output's (carried by the base ledger, `Candidate::new`): its cut
/// beyond the budget (`CUT`), and its kinetic temperature where it departs from T by more than T's share
/// -- a trap a few k_BT deep holds no bound ensemble as hot as T (none of a Gaussian beam's is hotter
/// than 0.166 D/k_B, ensemble.rs), the reference state's depth in k_BT.
fn initial_flag(initial:&Initial)->Option<String>{
    let cut=(initial.cut>STEP_SHARE*POPULATION_TARGET).then(||format!("初始热分布中能量高于阱顶（未束缚）及阱顶附近的 {:.0}% 原子不计入（取束缚原子的降低等温分布）",100.*initial.cut));
    let ratio=initial.kinetic();let set=initial.temperature_uk;
    let shallow=((ratio-1.).abs()>STEP_SHARE*TEMPERATURE_TARGET).then(||{
        let k=(0..initial.states.len()).max_by(|&a,&b|initial.states[a].2.total_cmp(&initial.states[b].2)).unwrap();
        let depth=initial.ensembles[k].as_ref().map_or(initial.depths[k],|e|e.depth)/(KT_PER_UK*set);let t=initial.temperature();
        // One decimal from one up (none for the setting's whole numbers), two significant digits below.
        let digits=|x:f64|if (x-x.round()).abs()<=1e-9*x.abs(){format!("{x:.0}")}else if x>=0.95{format!("{x:.1}")}else{let d=(-x.log10()).ceil().max(0.) as usize+1;format!("{x:.d$}")};
        format!("势阱较浅（D ≈ {} k_BT）：束缚的热分布动能温度为 {} μK，{}设定的 {} μK",digits(depth),digits(t.iter().sum::<f64>()/3.),if ratio<1.{"低于"}else{"高于"},digits(set))});
    match(cut,shallow){(Some(c),Some(s))=>Some(format!("{c}；{s}")),(c,s)=>c.or(s)}
}
/// The cut's share in a ledger (`CUT`), zero without one.
fn cut(ledger:&Ledger)->f64{ledger.items.iter().find(|i|i.label==CUT).map_or(0.,|i|i.estimate)}
/// The fold's charge for the saturation S its folded drives meet: its populations' error bound, S at
/// second order and S² at fourth (ground.rs; against the population core 0.2-1.0 S² where the series
/// converges, S ≲ 0.5 -- beyond, the charge exceeds the target and the rule keeps the drives).
fn fold_charge(model:&Model,saturation:f64)->f64{fold_charges(model.fourth,saturation)[0]}
/// The fold's charges per target, the populations' (`fold_charge`) and T's, for the saturation S its folded drives meet at
/// second or fourth order: T's is S at either -- the force folds at second order. T is the ratio of the momentum diffusion and the friction, each of the fold's error S, so the moving PGC's z below exceeds it; its charge comes with the rooms' combination of errors (engine.rs # Absorption, both models on one kick:
/// the explicit states' T against the all-folded T, second-half means over 1 ms on the axes above 4ħω: a tetrahedral MOT's z
/// +0.03 ± 0.7 % and x +0.6 ± 1.7 % at S = 2.1 %; the moving 3D PGC's x −2.2 ± 0.8 % and z +5.2 ± 1.0 % at S = 2.05 %; a gray
/// molasses's z +1.4 ± 4.9 % and x −1.6 ± 2.4 % at S = 3.4 %).
fn fold_charges(fourth:bool,saturation:f64)->[f64;2]{[if fourth{saturation*saturation}else{saturation},saturation]}
/// The targets in the order of the rooms: the populations' (absolute) and T's (relative).
const TARGETS:[f64;2]=[POPULATION_TARGET,TEMPERATURE_TARGET];

/// A candidate of the fold (module documentation, # The fold's choice): the model at one of the rule's
/// allowances, its initial state, its ledger, and the fold's charge per output and target (`fold_charges`) -- a
/// priori at every output, or as measured (`Candidate::charged`).
struct Candidate{model:Model,initial:Initial,base:Ledger,allowance:f64,charges:Vec<[f64;2]>,settled:bool}
/// A candidate's weighed cost (`evaluate` in `run`): at the step its check keeps, its work over the run times the weight of
/// its totals (`predicted`), its check's own work, its work per interval, the step's bias and its a-priori totals per output.
struct Costed{cost:f64,check:f64,work:f64,bias:[f64;2],apriori:Vec<[f64;2]>,checked:Steps}
impl Candidate{
    fn new(request:&Request,allowance:f64,fourth:bool)->Result<Candidate,String>{
        let model=Model::with_order(request,allowance,fourth)?;let initial=Initial::deferred(&model,request)?;let base=model_ledger(&model,request);
        let charges=vec![fold_charges(model.fourth,model.ground.charge);request.times_us.len()];
        Ok(Candidate{model,initial,base,allowance,charges,settled:false})
    }
    /// The initial ensemble's quadrature (ensemble.rs), which none of the fold's choices reads, for the pilot's and
    /// the chosen model alone: the share of the traps' thermal distributions it leaves out, and its flag
    /// (`initial_flag`) on the base ledger, every output's.
    fn settle(&mut self){
        if self.settled{return;}self.settled=true;self.initial.integrate(&self.model);let(initial,base)=(&self.initial,&mut self.base);
        if initial.cut>0.{base.add(CUT,initial.cut,"取势阱中束缚原子的降低等温分布 f ∝ e^(−E/k_BT) − 1（E 为相对阱外的机械能，E < 0）：相对势阱区（谐振近似低于阱深处）内的热分布，未束缚与阱顶附近的原子不计入；丢失与布居依赖这一截断".into());}
        base.flagged=initial_flag(&initial);
    }
    /// Its fold's charges as measured (`measured_charge`) in place of the a-priori estimate.
    fn charged(&mut self,charges:Vec<[f64;2]>){self.charges=charges;}
    /// Its ledger's estimates but the fold's.
    fn others(&self)->f64{unfolded(&self.base,&self.model)}
    /// The estimates no candidate of the fold changes: the far components' fold and the beams' interference, the same in
    /// every model of the request. (The others move with the fold: a folded drive's pole leaves the tables no convergence
    /// -- RETRO's every drive folded, its table stalled at 33 nodes with 1.27 booked --, and a rung keeps it.)
    fn fixed(&self)->f64{let g=&self.model.ground;g.fold_estimates().0+g.interference.iter().map(|x|x.estimate(g)).sum::<f64>()}
    /// Its reference state, the largest initial population: its trap (ω per axis) and depth.
    fn reference(&self)->(Trap,f64){let i=&self.initial;let k=(0..i.states.len()).max_by(|&a,&b|i.states[a].2.total_cmp(&i.states[b].2)).unwrap();(i.traps[k],i.depths[k])}
}
/// The next candidate along the rule's allowances that builds (a model that fails, its tables not
/// converging, is passed over with its error kept).
fn next_candidate(request:&Request,ladder:&mut dyn Iterator<Item=f64>,fourth:bool,failed:&mut Option<String>)->Option<Candidate>{
    for a in ladder{match Candidate::new(request,a,fourth){Ok(c)=>return Some(c),Err(e)=>{failed.get_or_insert(e);}}}
    None
}
/// The fold's charges a candidate's folded drives would take per output and target (`fold_charges`), at the pilot's
/// states carried into its clusters (`carried`) and at its initial ones: per output their mean 2η² (twice the folded
/// excited population; each output holds few of the pilot's states, and their largest is their extreme, not the
/// run's), an output the pilot kept no states at holding the last one's, with the memory (`Memory::charges`) -- what
/// the run's ledger books there. What the atoms meet rather than the light's pattern extrapolated (an atom held in the
/// dark states of a gray molasses meets little of its bright state's saturation once there, all of it before; one
/// gathered at a red-detuned lattice's bright sites meets more than its mean). The a-priori charge at every output
/// where the pilot kept no state after t = 0.
fn measured_charge(from:&Model,to:&Model,states:&[Trajectory],starts:&[Trajectory],memory:&Memory,interval:f64,outputs:usize)->Vec<[f64;2]>{
    let charge=|t:&Trajectory|{let x=carried(from,to,t);let p=to.point(x.r,x.kappa);fold_charges(to.fourth,2.*to.readout(x.cluster,&p,&x.psi,x.time).1)};
    // Per output the states' mean charges.
    let(mut profile,mut held)=(vec![[0.;2];outputs],vec![false;outputs]);
    for group in by_output(starts).into_iter().chain(by_output(states)){
        let i=((group[0].time/interval).round() as usize).min(outputs-1);let n=group.len() as f64;held[i]=true;
        let charges:Vec<[f64;2]>=group.par_iter().map(|&t|charge(t)).collect();
        for c in &charges{for k in 0..2{profile[i][k]+=c[k]/n;}}
    }
    if !held.iter().skip(1).any(|&h|h){return vec![fold_charges(to.fourth,to.ground.charge);outputs];}
    for i in 1..outputs{if !held[i]{profile[i]=profile[i-1];}}
    memory.charges(&profile)
}

/// The fold's memory (module documentation, # Ledger), from a pilot's atoms: per atom its jumps up to each
/// output (at its last state kept there or before) and from each output on (at its first state kept there or
/// after: where the pilot keeps every few outputs' states, the jumps in between are at most their difference),
/// and the last output it was kept at -- held there.
struct Memory{upto:Vec<Vec<u64>>,from:Vec<Vec<u64>>,last:Vec<usize>}
impl Memory{
    fn of(states:&[Trajectory],interval:f64,outputs:usize)->Memory{
        let mut atoms:Vec<(u64,Vec<(usize,u64)>)>=vec![];
        for t in states{let at=(((t.time/interval).round() as usize).min(outputs-1),t.jumps);
            match atoms.iter_mut().find(|a|a.0==t.index){Some(a)=>a.1.push(at),None=>atoms.push((t.index,vec![(0,0),at]))}}
        let(mut upto,mut from,mut last)=(vec![],vec![],vec![]);
        for(_,mut kept)in atoms{
            kept.sort();let u:Vec<u64>=(0..outputs).map(|i|kept.iter().rev().find(|x|x.0<=i).map_or(0,|x|x.1)).collect();
            from.push((0..outputs).map(|j|kept.iter().find(|x|x.0>=j).map_or(u[j],|x|x.1)).collect());upto.push(u);last.push(kept[kept.len()-1].0);
        }
        Memory{upto,from,last}
    }
    /// The share of an error made by output j that the atoms held at output i ≥ j still carry: per atom e^{−n}, n
    /// its jumps in between, their mean; all of it where none is held at i.
    fn kept(&self,i:usize,j:usize)->f64{
        let(mut sum,mut count)=(0.,0.);
        for a in (0..self.last.len()).filter(|&a|self.last[a]>=i){sum+=(-(self.upto[a][i].saturating_sub(self.from[a][j]) as f64)).exp();count+=1.;}
        if count>0.{sum/count}else{1.}
    }
    /// The fold's charges per output and target with the memory: at each output the largest of its own and each
    /// earlier output's times the share kept since the end of the interval after it, over which that one's
    /// error is made.
    fn charges(&self,values:&[[f64;2]])->Vec<[f64;2]>{
        let top:[f64;2]=std::array::from_fn(|k|values.iter().map(|v|v[k]).fold(0.,f64::max));
        (0..values.len()).map(|i|{let mut most=values[i];
            for j in (0..i).rev(){let kept=self.kept(i,j+1);if (0..2).all(|k|top[k]*kept<=most[k]){break;}for k in 0..2{most[k]=most[k].max(values[j][k]*kept);}}
            most}).collect()
    }
}

/// A pilot's states grouped by the output they were kept at, in time order: a trajectory's time is the sum of its
/// steps, so one output's times differ between trajectories in their last bits -- within the rounding of
/// `MOST_STEPS` additions, far below an interval (`measured_charge`, `Binding::of`).
pub fn by_output(states:&[Trajectory])->Vec<Vec<&Trajectory>>{
    let mut sorted:Vec<&Trajectory>=states.iter().collect();sorted.sort_by(|a,b|a.time.total_cmp(&b.time));
    let mut groups:Vec<Vec<&Trajectory>>=vec![];
    for t in sorted{match groups.last_mut(){Some(g)if t.time-g[0].time<=MOST_STEPS as f64*f64::EPSILON*t.time.abs()=>g.push(t),_=>groups.push(vec![t])}}
    groups
}

/// A trajectory's state carried from one model into another's clusters: into the cluster of its
/// most populated ground state, its amplitudes on that cluster's ground states and on the explicit
/// excited states the two clusters share.
fn carried(from:&Model,to:&Model,t:&Trajectory)->Trajectory{
    let(f,g)=(&from.ground.clusters[t.cluster],&to.ground);let ng=f.states.len();
    let most=(0..ng).max_by(|&a,&b|t.psi[a].norm_sqr().total_cmp(&t.psi[b].norm_sqr())).unwrap();
    let k=g.cluster[f.states[most]];let c=&g.clusters[k];
    let mut out=t.clone();out.cluster=k;out.psi=[Complex::ZERO;crate::model::MOST_MIXED];
    for(x,&b)in f.states.iter().enumerate(){if let Some(y)=c.position(b){out.psi[y]=t.psi[x];}}
    for(x,&e)in f.excited.iter().enumerate(){if let Some(y)=c.excited_position(e){out.psi[y]=t.psi[ng+x];}}
    out
}

/// A rung's pilot states carried into its clusters (`carried`), let settle where they hold explicit excited
/// states: those start bare, and the exact step's dressing transient -- the excited amplitudes' build-up,
/// ringing at the detuning and damped at half the excited states' decay -- is no step's error, while the run's
/// states between their jumps are dressed. Each evolves without jumps (`Engine::coast` at `settings`) until the
/// transient's amplitude is within the step's share s of itself, e^{−Γτ/2} ≤ s at the slowest decay Γ of its cluster's
/// explicit states (the generator's −2 Im of their diagonal), its norm restored.
fn relaxed(model:&Model,settings:&Settings,states:Vec<Trajectory>)->Vec<Trajectory>{
    let slowest:Vec<f64>=(0..model.clusters()).map(|k|{if !model.mixed(k){return 0.;}let(n,n0)=(model.dimension(k),model.ground_dimension(k));
        let g=model.generator(k,&model.point([0.;3],[0.;3]),0.);(n0..n).map(|e|-2.*g.hamiltonian[e*n+e].im).fold(f64::INFINITY,f64::min)}).collect();
    let engine=Engine::new(model,settings.clone());
    states.into_par_iter().map(|t|{let decay=slowest[t.cluster];if !(decay>0.&&decay.is_finite()){return t;}
        let mut x=engine.coast(&t,2.*(1./STEP_SHARE).ln()/decay,1);let n=model.dimension(x.cluster);
        let norm=x.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();for z in x.psi[..n].iter_mut(){*z=*z*(1./norm);}x}).collect()
}

/// The batch after `done` trajectories (module documentation, # Batches): the predicted remaining
/// need, at least the device's `width` and at most `done` -- the need followed at most to twice the
/// trajectories it rests on --, in whole `unit`s (groups on the CPU, waves on the GPU).
fn next_batch(needed:u64,done:u64,width:u64,unit:u64)->u64{needed.saturating_sub(done).clamp(width,done.max(width)).div_ceil(unit)*unit}

/// The targets that bind a solve: the populations and the loss at every output; T only at the outputs
/// in the method's regime (k_B T ≥ CLASSICAL ħω on every axis, as `classical`), outside which T is a
/// reference only (仅供参考). Here from a pilot's states at the outputs: T binds where some output's
/// states, their T per axis m⟨δv²⟩/k_B = ħ²⟨δκ²⟩/m, lie in the regime.
#[derive(Clone,Copy,Debug)]
pub struct Binding{pub temperature:bool}
impl Binding{
    pub fn of(states:&[Trajectory],hbar_over_mass:f64,frequencies:[f64;3])->Binding{
        let temperature=by_output(states).iter().any(|at|{let n=at.len() as f64;
            n>1.&&(0..3).all(|a|{let mean=at.iter().map(|t|t.kappa[a]).sum::<f64>()/n;
                let variance=at.iter().map(|t|(t.kappa[a]-mean).powi(2)).sum::<f64>()/(n-1.);hbar_over_mass*variance>=CLASSICAL*frequencies[a]})
        });
        Binding{temperature}
    }
}

/// The step search (module documentation): from `first` each step's check (`check`, or the `known` one where
/// it was checked) and the next step from it (`next_step`), until a step comes up twice, at most 8 checks;
/// with `measure` (the measured search) the population error measured where the population decides a step.
fn search(first:usize,most:usize,known:&[Check],check:&dyn Fn(usize)->Check,measure:Option<&dyn Fn(usize)->f64>,checked:&mut Vec<Check>){
    let mut steps=first;
    for _ in 0..8{
        if checked.iter().any(|c|c.steps==steps){break;}
        let mut c=known.iter().find(|c|c.steps==steps).copied().unwrap_or_else(||check(steps));
        if let Some(m)=measure.filter(|_|c.decided_by_population()){c.population=m(steps);}
        checked.push(c);steps=next_step(&c,measure.is_some(),most);
    }
}

/// The search's next step after a check (module documentation): the square root of its excess away (all scale
/// as dt²) -- in the measured search the excess of the criteria evaluated: where the motion or the loss missed,
/// the population was not measured, and its bound, which over-counts several times where that search runs, is
/// left out.
fn next_step(c:&Check,measured:bool,most:usize)->usize{
    let missed=c.motion.max(c.loss);let excess=if measured&&missed>1.{missed}else{c.excess()};
    ((c.steps as f64*excess.sqrt()).ceil() as usize).clamp(1,most)
}

/// The step kept among the checked ones that met the targets (module documentation): the largest (where checked twice,
/// its later check, the population measured), or the largest of those whose weighed cost (`cost`) lies within the
/// prediction's own uncertainty (`error`, relative) of the least: a finer step only for a saving beyond it. None without
/// one.
fn kept(candidates:&[Check],cost:&dyn Fn(&Check)->f64,error:f64)->Option<Check>{
    let largest=*candidates.iter().rev().min_by_key(|c|c.steps)?;
    let costs:Vec<(Check,f64)>=candidates.iter().filter(|c|c.steps>=largest.steps).map(|c|(*c,cost(c))).collect();
    let least=costs.iter().map(|x|x.1).fold(f64::INFINITY,f64::min);
    costs.iter().rev().filter(|x|x.1<=least*(1.+error)).min_by_key(|x|x.0.steps).map(|x|x.0).or(Some(largest))
}

/// The relative standard error of the trajectories the pilot predicts (module documentation): each target's need goes
/// as its statistic squared (`statistics`: the largest over the outputs of CONFIDENCE standard errors, from the pilot's
/// groups), so twice the statistic's relative error, its jackknife over the groups; the largest over the targets that
/// bind. Zero where fewer than three groups or no statistic stand.
fn need_error(groups:&[Group],scales:&Scales,means:&[Means],model:&Model,initial:&Initial,frequencies:[f64;3],binds:[bool;2])->f64{
    let statistic=|groups:&[Group]|->[f64;2]{let mut e=summarize(groups,scales,means);start(&mut e[0],model,initial);
        e.iter().skip(1).map(|x|statistics(x,frequencies)).fold([0.;2],|m,(p,t)|[m[0].max(p),m[1].max(t.unwrap_or(0.))])};
    let n=groups.len();if n<3{return 0.;}
    let all=statistic(groups);
    let left:Vec<[f64;2]>=(0..n).into_par_iter().map(|g|statistic(&groups.iter().enumerate().filter(|&(i,_)|i!=g).map(|(_,x)|x.clone()).collect::<Vec<Group>>())).collect();
    (0..2).filter(|&k|binds[k]&&all[k]>0.).map(|k|{let mean=left.iter().map(|x|x[k]).sum::<f64>()/n as f64;
        2.*((n-1) as f64/n as f64*left.iter().map(|x|(x[k]-mean).powi(2)).sum::<f64>()).sqrt()/all[k]}).fold(0.,f64::max)
}

/// A checked step (module documentation): its steps per interval and its ratios to the tolerances
/// (`tolerances`) -- the population criterion's (its bound's, or where measured the upper confidence bound
/// of its measured error), the loss's (absolute as the populations') and the motion's (the largest of the
/// impulse's, the orbits' and the jumps'); 1 meets each.
#[derive(Clone,Copy,Debug)]
struct Check{steps:usize,population:f64,loss:f64,motion:f64}
impl Check{
    /// Per target, the populations' (with the loss's) and T's.
    fn ratios(&self)->[f64;2]{[self.population.max(self.loss),self.motion]}
    fn excess(&self)->f64{self.population.max(self.loss).max(self.motion)}
    fn met(&self)->bool{self.excess()<=1.}
    /// The population criterion decides the step: its ratio the largest, the others met.
    fn decided_by_population(&self)->bool{self.motion.max(self.loss)<=1.&&self.population>self.motion.max(self.loss)}
}

/// A model's step check on sampled states (module documentation, # Time step): the bound rule's search from
/// `first` (`search`, the next step from the criteria evaluated, `next_step`), then, where the population criterion
/// decides that rule's step, the search from it with the population error measured where it decides. Kept: the
/// largest met step, a finer one only for a predicted saving beyond the need's relative error `need` (`kept`; the
/// weighed cost `cost`), and the rule taken once more over every step from the largest to the finest candidate; or the
/// last checked when none met. A met step's bias is at most the step's share of the run's room (`tolerances`): it leaves
/// the statistics their balance by construction. Every check after the searches is the measured list's (for
/// `checked_work`); the tolerances the checks took, for the step's bias.
struct Steps{bound:Vec<Check>,measured:Vec<Check>,chosen:Check,probes:usize,tolerances:[f64;2]}
fn step_check(model:&Model,states:&[Trajectory],settings:&dyn Fn(usize)->Settings,first:usize,most:usize,waiting:f64,horizon:f64,subsample:usize,groups:&[Group],depth:f64,period:f64,
    tolerances:[f64;2],cost:&dyn Fn(&Check)->f64,need:f64)->Steps{
    // The orbits' probes and the loss's sensitivity (`bound_probes`, `escape`), at the pilot's groups.
    let(probes,escape)=(bound_probes(model,states),escape(model,states,groups,depth));
    let doubled=|steps:usize|{let r=step_excess(&Engine::new(model,settings(steps)),states,waiting,&probes,period,escape,tolerances);Check{steps,population:r[0],loss:r[1],motion:r[2]}};
    let measure=|steps:usize|accumulated_excess(&Engine::new(model,settings(steps)),states,horizon,subsample,tolerances[0]);
    let checked=|steps:usize|{let mut c=doubled(steps);if c.decided_by_population(){c.population=measure(steps);}c};
    let mut bound=vec![];search(first,most,&[],&doubled,None,&mut bound);
    let grown=*bound.iter().filter(|c|c.met()).min_by_key(|c|c.steps).unwrap_or(&bound[bound.len()-1]);
    let mut measured=vec![];if grown.decided_by_population(){search(grown.steps,most,&bound,&doubled,Some(&measure as &dyn Fn(usize)->f64),&mut measured);}
    let mut candidates:Vec<Check>=bound.iter().chain(&measured).filter(|c|c.met()).copied().collect();
    let chosen=if candidates.len()>1{
        let mut choice=kept(&candidates,cost,need).unwrap_or(grown);
        // The same rule over every step from the largest to the finest candidate, their ratios the largest's as dt²: its
        // pick is checked too -- the choice moves with its estimates instead of keeping whichever steps the search checked.
        if let Some(largest)=candidates.iter().rev().min_by_key(|c|c.steps).copied(){
            let carried=|s:usize|{let x=(largest.steps as f64/s as f64).powi(2);Check{steps:s,population:largest.population*x,loss:largest.loss*x,motion:largest.motion*x}};
            let range:Vec<Check>=(largest.steps..=candidates.iter().map(|c|c.steps).max().unwrap()).map(carried).collect();
            if let Some(s)=kept(&range,cost,need).map(|c|c.steps).filter(|&s|!candidates.iter().any(|c|c.steps==s)){
                let c=checked(s);measured.push(c);if c.met(){candidates.push(c);choice=kept(&candidates,cost,need).unwrap_or(grown);}
            }
        }
        choice
    }else{candidates.first().copied().unwrap_or(grown)};
    Steps{bound,measured,chosen,probes:probes.len(),tolerances}
}
/// A step check's own work (ns, `step_check`) at `work`'s work per step (per interval over the steps): per
/// checked step a step and two halves on every state and the orbits' jump-free flow over the trap's `period` in
/// steps of dt and of dt/2 from every probe (`checked_step`), and where the population error was measured the
/// jump-free flow over the horizon in steps of dt and of dt/2 from the subsample.
fn checked_work(checked:&Steps,states:usize,subsample:usize,horizon:f64,interval:f64,period:f64,work:&dyn Fn(usize)->f64)->f64{
    let doubled=|c:&Check|checked_step(states,checked.probes,c.steps,interval,period)*work(c.steps)/c.steps as f64;
    checked.bound.iter().map(doubled).sum::<f64>()+checked.measured.iter().map(|c|if checked.bound.iter().any(|b|b.steps==c.steps){0.}else{doubled(c)}
        +3.*subsample.min(states) as f64*(horizon/interval).max(1./c.steps as f64)*work(c.steps)).sum::<f64>()
}
/// The steps a step check's doubling takes at `steps` per interval (`step_excess`): a step and two halves on every
/// state, and from every probe the orbits' over the trap's period in steps of dt and of dt/2.
fn checked_step(states:usize,probes:usize,steps:usize,interval:f64,period:f64)->f64{3.*states as f64+3.*probes as f64*(period*steps as f64/interval).ceil()}

/// The tolerances of a step's criteria (module documentation, # Shares): per target the step's share s of the run's room
/// beside the a-priori totals `apriori` per output (`run_room`), its least over the outputs after t = 0 -- T's where T
/// binds, the T target itself where T is a reference only.
fn tolerances(temperature:bool,apriori:&[[f64;2]])->[f64;2]{
    let share=|k:usize|STEP_SHARE*apriori.iter().skip(1).map(|a|run_room(TARGETS[k],a[k])).fold(TARGETS[k],f64::min);
    [share(0),if temperature{share(1)}else{TEMPERATURE_TARGET}]}

/// The step-doubling excess of a step (module documentation), each criterion over its tolerance
/// (`tolerances`; 1 meets it): the population criterion's bound -- from all the sampled states one step
/// against two halves, the populations' difference times the steps of a waiting time --; the loss's, the
/// escape's sensitivity times the motion's energy error, the orbits' or the jumps' (`escape`); and the
/// motion's, the largest of the light impulse's relative error over one step (the same doublings), the
/// orbits' error over a trap period (`orbits`) and the jumps' (`jumps`).
fn step_excess(engine:&Engine,states:&[Trajectory],waiting:f64,probes:&[Trajectory],period:f64,escape:f64,tolerances:[f64;2])->[f64;3]{
    let dt=engine.settings.dt;let richardson=4./3.;let count=states.len().max(1) as f64;
    // The states' doublings on the threads, summed in their order (the sums do not depend on the threads).
    let doubled:Vec<([f64;2],[f64;3])>=states.par_iter().map(|t|engine.doubling(t,dt)).collect();
    let(mut population,mut difference,mut impulse)=(0.,0.,0.);
    for(motion,internal)in &doubled{difference+=motion[0];impulse+=motion[1];population+=internal[0];}
    let bound=richardson*population/count*(waiting/dt).max(1.);
    let relative=if impulse>0.{richardson*difference/impulse}else{0.};
    let energy=orbits(engine,probes,period).max(jumps(engine,states));
    [bound/tolerances[0],escape*energy/tolerances[0],relative.max(energy)/tolerances[1]]
}

/// The heating the jumps carry, its step error relative to their recoil (module documentation, # Time step):
/// the step holds each (sub-)step's jump hazard R at its middle, so a jump's place u in its step weighs the
/// states along it evenly, not by their hazard -- a first-order profile of its expected deposit over u, of slope
/// S = −E[c ΔR]/E[R] per step (`deposit`'s cross term c, ΔR the hazard's change over the jump-free step at the
/// state's ψ -- the generator's, which the step holds --, E over the sampled states). A jump that follows another
/// within its step, by the chance R(1 − u) dt after one at u, keeps the first one's place: per jump the net
/// −(R_j dt/12) S, R_j = E[R²]/E[R] the hazard at the jumps -- over the recoil per jump E[R r]/E[R], with
/// E[c ΔR] its magnitude plus CONFIDENCE standard errors. Zero without jumps.
pub fn jumps(engine:&Engine,states:&[Trajectory])->f64{
    let(model,dt)=(engine.model,engine.settings.dt);
    let parts:Vec<[f64;4]>=states.par_iter().map(|t|{
        let n=model.dimension(t.cluster);let size=t.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();
        let mut psi=[Complex::ZERO;MOST_MIXED];for i in 0..n{psi[i]=t.psi[i]*(1./size);}
        // The hazard ⟨ψ|i(H − H†)|ψ⟩ of the generator at a point, at the state's time and ψ.
        let hazard=|r:[f64;3],kappa:[f64;3]|{let h=model.generator(t.cluster,&model.point(r,kappa),t.time).hamiltonian;
            (0..n).map(|a|(0..n).map(|b|(psi[a].conj()*(h[a*n+b]-h[b*n+a].conj())*psi[b]*Complex::I).re).sum::<f64>()).sum::<f64>()};
        let rate=hazard(t.r,t.kappa);let end=engine.coast(t,dt,1);let(cross,recoil)=deposit(model,t,&psi,engine.settings.doppler);
        [rate,rate*rate,cross*(hazard(end.r,end.kappa)-rate),rate*recoil]}).collect();
    let n=parts.len() as f64;let sum=|k:usize|parts.iter().map(|x|x[k]).sum::<f64>();
    let(rate,square,heating)=(sum(0),sum(1),sum(3));if !(n>1.&&rate>0.&&heating>0.){return 0.;}
    let mean=sum(2)/n;let error=(parts.iter().map(|x|(x[2]-mean).powi(2)).sum::<f64>()/(n*(n-1.))).sqrt();
    dt/12.*square/rate*(mean.abs()+CONFIDENCE*error)/(heating/n)
}

/// A jump's expected deposit at a state of normalized ψ (rad/us): its cross term (ħ²/m) κ·E[Δκ] and its recoil
/// (ħ²/2m) Σ_φ (w_φ/W)(k_φ² + |E[Δκ|φ]|²), at most (ħ²/2m) E|Δκ|²: per class φ of photons of weight w_φ (W their
/// sum) the photon's own recoil and its absorbed light's mean, E[Δκ|φ] = Im Σ_a ⟨u_a|∇u_a⟩/w_φ (engine.rs
/// `Engine::kick`: over the directions and polarizations the drawn beams' and the emissions' kicks average out).
/// Zero without photons.
fn deposit(model:&Model,t:&Trajectory,psi:&[Complex],doppler:f64)->(f64,f64){
    let point=model.point(t.r,t.kappa);let exact=if model.doppler_ratio(t.kappa)>doppler{Exact::new(model,t.r,t.kappa).ok()}else{None};
    let photons=model.photons(t.cluster,&point,psi,t.time,exact);
    let(mut weight,mut mean,mut recoil)=(0.,[0.;3],0.);
    for(c,(class,u))in model.ground.clusters[t.cluster].classes.iter().zip(&photons.amplitude).enumerate(){
        let nt=model.ground_dimension(class.target);let w:f64=u.iter().map(|v|v[..nt].iter().map(|z|z.norm_sqr()).sum::<f64>()).sum();if !(w>0.){continue;}
        let mut m=[0.;3];
        for a in 0..3{let mut e=[0.;3];e[a]=1.;let(gradient,_)=photons.absorption(c,e,&[Complex::ZERO;MOST_STATES]);for x in 0..3{for i in 0..nt{m[x]+=(u[a][i].conj()*gradient[i][x]).im;}}}
        weight+=w;for x in 0..3{mean[x]+=m[x];}recoil+=w*class.wave_number*class.wave_number+m.iter().map(|v|v*v).sum::<f64>()/w;
    }
    if !(weight>0.){return(0.,0.);}
    let hm=model.hbar_over_mass;(hm*(0..3).map(|a|t.kappa[a]*mean[a]).sum::<f64>()/weight,0.5*hm*recoil/weight)
}

/// The loss's sensitivity to the motion's energy scale: the pilot's escape over the trap's top -- the
/// share of its trajectories lost after a state of theirs was bound (atoms the light heated over the
/// top; an unbound one leaving is no escape) -- times the trap's depth over the bound states' mean
/// kinetic energy per axis, D/k_BT. An escape over a barrier from a thermal tail goes as e^{−D/k_BT}:
/// a relative error δ of the motion's energy (its orbits', `orbits`) moves the trap's top by δD for the
/// orbits that reach it, and the escaped share by D/k_BT δ of itself. Zero where none escaped.
fn escape(model:&Model,states:&[Trajectory],groups:&[Group],depth:f64)->f64{
    let(hm,outputs)=(model.hbar_over_mass,groups.first().map_or(0,|g|g.sums.len()));if outputs==0{return 0.;}
    let bound:Vec<(u64,f64)>=states.par_iter().filter_map(|t|{let p=model.point(t.r,t.kappa);let k:f64=t.kappa.iter().map(|k|k*k).sum();
        (0.5*hm*k+model.trap_potential(t.cluster,&p,&t.psi,t.time)<0.).then_some((t.index,hm*k/3.))}).collect();
    if bound.is_empty(){return 0.;}
    let kt=bound.iter().map(|b|b.1).sum::<f64>()/bound.len() as f64;
    let escaped=groups.iter().enumerate().filter(|(k,g)|g.sums[outputs-1].held==0.&&bound.iter().any(|b|b.0==*k as u64)).count() as f64;
    escaped/groups.len() as f64*depth/kt
}

/// The orbits' probes: per pilot trajectory its middle state among those bound to the trap -- their
/// mechanical energy below the trap's far level (model.rs `Model::trap_potential`) --, the atoms whose T
/// the step's error stays on (an unbound one leaves within a transit of the trap's region).
fn bound_probes(model:&Model,states:&[Trajectory])->Vec<Trajectory>{
    let hm=model.hbar_over_mass;
    let bound:Vec<bool>=states.par_iter().map(|t|{let p=model.point(t.r,t.kappa);0.5*hm*t.kappa.iter().map(|k|k*k).sum::<f64>()+model.trap_potential(t.cluster,&p,&t.psi,t.time)<0.}).collect();
    let mut indices:Vec<u64>=states.iter().map(|t|t.index).collect();indices.sort();indices.dedup();
    indices.iter().filter_map(|&i|{let own:Vec<&Trajectory>=states.iter().zip(&bound).filter(|(t,&b)|t.index==i&&b).map(|(t,_)|t).collect();
        (!own.is_empty()).then(||own[own.len()/2].clone())}).collect()
}

/// The conservative motion's step error on T (module documentation): from each probe the jump-free flow
/// over one period of the trap's slowest axis (the solve's T windows' span, past the run where it is
/// shorter: a window short of a period weighs the orbit's phase, not the step) in steps of dt and of dt/2, ψ kept
/// normalized, the mean κ² per axis over it compared -- 4/3 their difference, the probe's orbit's own
/// error, which the window means of T keep. Per axis the probes' mean difference relative to their mean
/// κ², its magnitude plus CONFIDENCE standard errors; the largest axis. Zero without probes.
fn orbits(engine:&Engine,probes:&[Trajectory],period:f64)->f64{
    if probes.is_empty(){return 0.;}
    // The flow takes no jump over the period: its exponentials and beats are held within the step's share over the
    // period as a run's are over a waiting time (`running`) -- summed past the waiting they were set for, their
    // truncation would be read as the step's T error (the 3D PGC's 2-3 times at the step's share s²).
    let mut held=engine.settings.clone();held.truncation=held.truncation.min(STEP_SHARE*STEP_SHARE*POPULATION_TARGET*held.dt/period);held.beats=held.beats.min(held.truncation);
    let engine=&Engine::new(engine.model,held);
    let(dt,model)=(engine.settings.dt,engine.model);let steps=((period/dt).ceil() as usize).max(1);
    let windows:Vec<[[f64;3];2]>=probes.par_iter().map(|t|{
        let mean=|h:f64,n:usize|->[f64;3]{let mut x=t.clone();let mut sum=[0.;3];let size=model.dimension(x.cluster);
            for _ in 0..n{x=engine.coast(&x,h,1);let norm=x.psi[..size].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();for z in x.psi[..size].iter_mut(){*z=*z*(1./norm);}
                for a in 0..3{sum[a]+=x.kappa[a]*x.kappa[a];}}
            sum.map(|v|v/n as f64)};
        [mean(dt,steps),mean(0.5*dt,2*steps)]}).collect();
    let n=windows.len() as f64;
    (0..3).map(|a|{
        let reference=windows.iter().map(|w|w[1][a]).sum::<f64>()/n;if !(reference>0.){return 0.;}
        let d:Vec<f64>=windows.iter().map(|w|4./3.*(w[0][a]-w[1][a])/reference).collect();let m=d.iter().sum::<f64>()/n;
        m.abs()+CONFIDENCE*(d.iter().map(|x|(x-m).powi(2)).sum::<f64>()/(n*(n-1.).max(1.))).sqrt()
    }).fold(0.,f64::max)
}

/// The population error a step accumulates over `horizon` (module documentation), over its tolerance:
/// from each of `subsample` states spread over the sampled ones the jump-free flow in steps of dt
/// against steps of dt/2, the largest difference of the cluster's populations; their mean plus
/// CONFIDENCE standard errors.
fn accumulated_excess(engine:&Engine,states:&[Trajectory],horizon:f64,subsample:usize,tolerance:f64)->f64{
    let dt=engine.settings.dt;let richardson=4./3.;let count=((horizon/dt).ceil() as usize).max(1);
    let every=(states.len()/subsample.max(1)).max(1);let picked:Vec<&Trajectory>=states.iter().step_by(every).collect();
    let errors:Vec<f64>=picked.par_iter().map(|t|{
        let n=engine.model.dimension(t.cluster);let(one,two)=(engine.coast(t,dt,count),engine.coast(t,0.5*dt,2*count));
        let(a,b)=(one.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),two.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
        richardson*(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max)
    }).collect();
    let m=errors.len().max(1) as f64;let mean=errors.iter().sum::<f64>()/m;
    let spread=(errors.iter().map(|e|(e-mean).powi(2)).sum::<f64>()/(m*(m-1.).max(1.))).sqrt();
    (mean+CONFIDENCE*spread)/tolerance
}

/// What a unit of each of the step's parts costs (ns), for the work model (`interval_work`): measured
/// once on one performance core of the Core Ultra 7 265H (examples/bench_cost.rs) and fitted over
/// DEFAULT, THREE, PGC3D, TET, GMD1C and PGCP_10, each part within 10–20 % of its measured time; only
/// their ratios enter the choice. An advance without its propagation: per call (the point's fields,
/// the flow, the kick), per multiply-add of the generator and the force (`Model::work`) and per table
/// pair.
const ADVANCE_NS:f64=374.;const MULTIPLY_ADD_NS:f64=0.247;const PAIR_NS:f64=29.9;
/// A half step's propagation: per term of its series and per multiply-add (n² a term).
const TERM_NS:f64=4.4;const TERM_MULTIPLY_ADD_NS:f64=0.469;
/// A whole jump (`Engine::jump`): per call (its point, the class's and the polarization's draws, the
/// dipole pattern, the kicks), per multiply-add of every class's photons and per multiply-add of the
/// picked class's absorbed light (`Model::work`). Timed against the advance in the same interleaved
/// rounds on the loaded machine (each the least of 30 rounds × 3 passes), taken at the advance's cost
/// above: within 10 % but DEFAULT's (+21 %); a jump costs 1.5–2.1 times its photons alone.
const JUMP_NS:f64=970.;const PHOTON_MULTIPLY_ADD_NS:f64=0.594;const ABSORPTION_MULTIPLY_ADD_NS:f64=1.48;
/// The held twin's step besides its two products (control.rs): 30–190 ns over DEFAULT, THREE, PGC3D
/// and GMD1C on the same core.
const HELD_NS:f64=100.;

/// An advance's counted work in a cluster (ns, `ADVANCE_NS` and the rest; model.rs `Model::work`), its propagation aside.
pub(crate) fn advance_work(w:&crate::model::Work)->f64{ADVANCE_NS+MULTIPLY_ADD_NS*(w.generator+w.force)+PAIR_NS*w.pairs as f64}
/// The phase the light's pattern advances per first step (module documentation, # Time step): sqrt(s × the temperature
/// target), its T error about the phase squared.
pub(crate) fn first_phase()->f64{(STEP_SHARE*TEMPERATURE_TARGET).sqrt()}
/// A sampled state's parts of the work model (`interval_work`): its cluster's advance without the
/// propagation and its jump (ns), its generator's norm and its dimension; in a cluster that takes the exact
/// step (explicit excited states, beats), the state itself -- its step's segment is probed at each step
/// (`exact_step`); and its weight in the run's time.
struct StateWork{advance:f64,jump:f64,norm:f64,n:usize,exact:Option<Trajectory>,weight:f64}
/// The work model's states (`StateWork`) over the run's time, each in the model's own clusters: the pilot's states at the
/// outputs, and its initial ones (`begun`) at half their weight -- the trapezoid rule over the outputs, whose first
/// interval the atoms start in their initial clusters (the shallow probe's every atom dark at the outputs, in its
/// rungs' one-state clusters, after a start in the explicit ones: its rungs' work counted as the folded model's, their
/// runs 3.4 times it).
fn state_work(model:&Model,states:&[Trajectory],begun:&[Trajectory])->Vec<StateWork>{
    states.par_iter().map(|t|(t,1.)).chain(begun.par_iter().map(|t|(t,0.5))).map(|(t,weight)|{
        let w=model.work(t.cluster);let n=model.dimension(t.cluster);let exact=model.mixed(t.cluster)||model.beating(t.cluster);
        let norm=if exact{0.}else{column_norm(&model.local(t.cluster,&model.point(t.r,t.kappa),t.time).hamiltonian,n)};
        StateWork{advance:advance_work(&w),
            jump:JUMP_NS+PHOTON_MULTIPLY_ADD_NS*w.photons+ABSORPTION_MULTIPLY_ADD_NS*w.absorption,norm,n,exact:exact.then(||t.clone()),weight}
    }).collect()
}
/// The exact step's work (`exact_step`, ns): per complex multiply-add of n × n matrices, per n³ of the spectral
/// path's eigendecomposition with its forms, and per complex exponential of a beat's phase (at the large
/// arguments a run's times give). Against the step's CPU time on one core of the Core Ultra 7 265H (the jump-free
/// step at each of a candidate's states along its trajectories, as examples/multirate.rs `profile` times it):
/// PGC3D's 10-state cluster (spectral at the run's beats share, cached at the old pilot's), PGCP_10's 12-state
/// and 5-state beating clusters and PGC3D's folded one take 1.8-3.3 times their counts on the loaded machine --
/// the counts' ratios, which alone choose, within about ±30 %.
pub(crate) const MATRIX_NS:f64=0.469;const SPECTRAL_N3:f64=60.;const CIS_NS:f64=25.;
/// The exact step's work at a state (ns, engine.rs `Engine::exact`), from the segment the step builds there
/// (`Engine::probe`): per step, every step's -- its advance (the components' gather, the flows and the kick),
/// the beats' bounds (`Segment::resolve`: per pair two commutators, n³ each, of either beat) and the segment's
/// build (the spectral path's eigendecomposition and forms, the Taylor path's propagator, n³ per term of its
/// series to double precision, or a period's cache, per spacing two exponentials and their product) --, and
/// the grid's: per node (amplitude-weighted on the spectral path) its propagation (a phase per state, a
/// propagator's product, or CF4's two combinations of the beats and their series) and the force's and the
/// norm's forms, n² each.
fn exact_step(model:&Model,s:&Settings,t:&Trajectory,x:&StateWork)->[f64;2]{
    let p=Engine::new(model,s.clone()).probe(t.cluster,&model.point(t.r,t.kappa),t.time,s.dt,&t.psi);
    let n=x.n as f64;let(n2,n3)=(n*n,n*n*n);let pairs=p.beats.len() as f64;let kept=2.*p.beats.iter().filter(|b|!b.2).count() as f64;
    let terms=series(1.,node_phase(),f64::EPSILON).1 as f64;let nodes=p.weighted.unwrap_or(p.nodes) as f64;
    // Per path: the build's multiply-adds and phases, a node's multiply-adds and phases (CF4's two combinations
    // take two phases a beat each, the force one a pair).
    let(build,phases,node,turns)=match p.path{
        "spectral"=>(SPECTRAL_N3*n3,0.,4.*n2+n,0.),
        "taylor"=>(terms*n3,0.,5.*n2,0.),
        "cached"=>{let k=model.beats(t.cluster).1.map_or(1.,|period|(period/p.spacing).round().max(1.));(k*((2.*terms+1.)*n3+18.*(1.+kept)*n2),5.*k*kept,5.*n2,0.)}
        _=>(0.,0.,(2.*(1.+kept+terms)+4.+1.5*kept)*n2,4.5*kept),
    };
    [x.advance+MATRIX_NS*(8.*pairs*n3+build)+CIS_NS*phases,nodes*(TERM_MULTIPLY_ADD_NS*node+CIS_NS*turns)]
}
/// The exact states a work count probes at most (`steps_parts`): each probe builds its segment.
const PROBED:usize=256;
/// A step's parts at every state (ns): what every step pays -- the advance with, in a cluster of ground states,
/// its two half steps' series, or the exact step's per step (`exact_step`) -- and what the multirate grid takes
/// over the step, whose nodes per unit time no step changes. The exact states' from at most `PROBED` of them,
/// evenly spread, their mean for the rest.
fn steps_parts(model:&Model,states:&[StateWork],s:&Settings)->Vec<[f64;2]>{
    let exact:Vec<usize>=(0..states.len()).filter(|&i|states[i].exact.is_some()).collect();
    let probed:Vec<[f64;2]>=exact.par_iter().step_by((exact.len()/PROBED).max(1)).map(|&i|exact_step(model,s,states[i].exact.as_ref().unwrap(),&states[i])).collect();
    let mean:[f64;2]=std::array::from_fn(|k|probed.iter().map(|p|p[k]).sum::<f64>()/probed.len().max(1) as f64);
    states.iter().map(|x|if x.exact.is_some(){mean}else{[x.advance+2.*half_step(x,s),0.]}).collect()
}
/// A half step's propagation at a state (ns): its series' sub-steps × terms at its generator's norm,
/// the step's dt and truncation.
fn half_step(x:&StateWork,s:&Settings)->f64{let(sub,terms)=series(x.norm,0.5*s.dt,s.truncation);(sub*terms) as f64*(TERM_NS+TERM_MULTIPLY_ADD_NS*(x.n*x.n) as f64)}
/// A step's work per interval (ns, `ADVANCE_NS` and the rest), counted on the sampled states -- the
/// same whatever the machine and its load: its steps (`steps_parts`), each the generator's and the force's
/// multiply-adds and pairs of the state's cluster and two half steps' series, or the exact step's; and per
/// jump, one every waiting time, the jump itself -- every class's photons, the picked class's absorbed light
/// and the rest -- and two more advances, the step redone to the jump and its rest (an exact step goes on
/// from the jump: its segment in the cluster it lands in). A readout per output is the same at every step and
/// left out.
fn interval_work(model:&Model,states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    let parts=steps_parts(model,states,s);let(mut advance,mut jump)=(0.,0.);
    for(x,p)in states.iter().zip(&parts){let a=p[0]+p[1];advance+=x.weight*a;jump+=x.weight*(x.jump+if x.exact.is_some(){p[0]}else{2.*a});}
    let count=weights(states);
    (interval/s.dt*advance+interval/waiting*jump)/count
}
/// The states' total weight in the run's time (`state_work`), at least one's.
fn weights(states:&[StateWork])->f64{states.iter().map(|x|x.weight).sum::<f64>().max(1.)}
/// The least work per interval any step leaves (ns, `interval_work`): the multirate grid's nodes, whose number
/// per unit time is the generator's and not the step's, and the jumps.
fn least_work(model:&Model,states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    let parts=steps_parts(model,states,s);let(mut grid,mut jump)=(0.,0.);
    for(x,p)in states.iter().zip(&parts){grid+=x.weight*p[1];jump+=x.weight*x.jump;}
    let count=weights(states);
    (interval/s.dt*grid+interval/waiting*jump)/count
}
/// The held twin's work per interval (control.rs, ns): per step its two products by the half step's
/// propagator (n² multiply-adds each) and its step's own (`HELD_NS`); per jump the step redone to it
/// and its rest by the series.
fn held_work(states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    let(mut step,mut jump)=(0.,0.);
    for x in states{step+=x.weight*(HELD_NS+2.*TERM_MULTIPLY_ADD_NS*(x.n*x.n) as f64);jump+=x.weight*4.*half_step(x,s);}
    let count=weights(states);
    (interval/s.dt*step+interval/waiting*jump)/count
}

/// t = 0 is the request's initial state, exactly: per lab axis T is its bound ensembles' kinetic
/// temperature with each trap's thermal Wigner factor (lib.rs `Initial::temperature`), the mean
/// velocity zero -- the trajectories' estimates there carry only the sampling's error.
fn start(e:&mut Estimate,model:&Model,initial:&Initial){
    let t=initial.temperature();let _=model;
    e.temperature=t;e.temperature_error=[0.;3];e.mean_velocity=[0.;3];
    e.temperature_total=t.iter().sum::<f64>()/3.;e.temperature_total_error=0.;
}

/// The solution of the estimates so far.
/// At t = 0 the populations are the initial state's, as the request gives them (the light acts
/// after), and so are T and the mean velocity (`start`). An output where no atom is held (the loss
/// one) has no held atom to describe: its populations, T, ⟨v⟩, n̄ and distributions are the last
/// output's that had one -- each row still a probability vector, the app's p (1 − L) zero there --,
/// outside the validated regime, and the flag says from when.
fn assemble(model:&Model,estimates:&[Estimate],outcome:&Outcome,listed:&[usize],times:&[f64],frequencies:[f64;3],trajectories:u64,dt:f64,initial:&[Complex],mut ledger:Ledger,twin:bool)->Solution{
    let hm=model.hbar_over_mass;
    let velocity=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.1[a]*hm));
    let position=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.0[a]));
    let first:Vec<f64>=(0..STATES).map(|i|initial[i*STATES+i].re).collect();
    let sizes:Vec<String>=model.ground.clusters.iter().map(|c|c.dimension().to_string()).collect();
    let explicit:usize=model.ground.clusters.iter().map(|c|c.excited.len()).sum();
    let internal=if explicit>0{format!("内态（子空间 {}，其中 {explicit} 个激发态显式、精确传播，其余激发态二阶消去）",sizes.join("+"))}else{format!("消去激发态后的基态内态（子空间 {}）",sizes.join("+"))};
    // Each output's source: itself where an atom is held (t = 0 always), else the last that held one.
    let held=|i:usize|i==0||estimates[i].loss<1.;
    let source:Vec<usize>=(0..estimates.len()).scan(0,|last,i|{if held(i){*last=i;}Some(*last)}).collect();
    if let Some(gone)=(0..estimates.len()).find(|&i|!held(i)){
        let note=format!("{} 起已无原子被俘获（丢失概率 1）：此后的布居、温度与分布沿用 {} 的值，仅供参考",when(times[gone]),when(times[source[gone]]));
        ledger.flagged=Some(match ledger.flagged.take(){Some(r)=>format!("{note}；{r}"),None=>note});
    }
    let carried=|f:&dyn Fn(&Estimate)->[f64;3]|->Vec<[f64;3]>{source.iter().map(|&j|f(&estimates[j])).collect()};
    Solution{
        populations:source.iter().map(|&j|if j==0{first.clone()}else{estimates[j].populations.clone()}).collect(),
        sample_errors:estimates.iter().enumerate().map(|(i,e)|if i==0{0.}else{e.population_errors.iter().fold(e.loss_error,|m,&x|m.max(x))}).collect(),
        temperature_uk:carried(&|e|e.temperature),
        temperature_errors_uk:carried(&|e|e.temperature_error),
        temperature_total_uk:source.iter().map(|&j|estimates[j].temperature_total).collect(),
        temperature_total_errors_uk:source.iter().map(|&j|estimates[j].temperature_total_error).collect(),
        mean_velocity_mps:carried(&|e|e.mean_velocity),
        trap_frequency_khz:frequencies.map(|w|w/std::f64::consts::TAU*1e3),
        mean_occupation:carried(&|e|e.occupation),
        loss_probability:{let mut worst=0.;estimates.iter().map(|e|{worst=e.loss.max(worst);worst}).collect()},
        velocity,position,
        classical:estimates.iter().enumerate().map(|(i,e)|held(i)&&classical(e,frequencies)).collect(),
        trajectories,needed:trajectories,time_step_us:dt,
        method:format!("半经典 3D：经典质点（位置、动量）携带{internal}，量子跳跃（自发辐射按偶极辐射方向图反冲，吸收按光束反冲并保持精确的平均力），机械能高于势阱深度且已离开势阱区即算损失，{trajectories} 条轨迹，步长 {dt:.3} μs；控制变量：谐振孪生（T）{}",if twin{"与静止孪生（布居）"}else{""}),
        reason:None,
        approximations:ledger,
        device:Device::Cpu,device_note:None,
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    /// The fold's charge with the memory: a bright transient's charge stays with atoms that go dark (b1/128's
    /// optical pumping, the validation's M1), and atoms that scatter forget it within their next photons.
    #[test]
    fn the_fold_s_charge_stays_with_dark_atoms_and_relaxes_with_the_photons(){
        let atom=|index:u64,jumps:&[u64]|->Vec<Trajectory>{jumps.iter().enumerate().map(|(i,&n)|Trajectory{cluster:0,psi:[Complex::ZERO;crate::model::MOST_MIXED],r:[0.;3],kappa:[0.;3],
            time:(i+1) as f64*5.,threshold:1.,alive:true,jumps:n,doppler:0.,beats:0.,seed:0,index}).collect()};
        let charges=|s:&[f64]|s.iter().map(|&x|[x,x]).collect::<Vec<[f64;2]>>();let first=|c:Vec<[f64;2]>|c.iter().map(|x|x[0]).collect::<Vec<f64>>();
        // Pumped dark within the first interval: two photons, then none.
        let dark=Memory::of(&[atom(0,&[2,2,2,2]),atom(1,&[3,3,3,3])].concat(),5.,5);
        assert_eq!(first(dark.charges(&charges(&[0.076,0.006,0.003,0.001,0.]))),vec![0.076;5]);
        // Scattering on: ten photons an interval, the transient forgotten after the next.
        let bright=Memory::of(&[atom(0,&[10,20,30,40]),atom(1,&[10,20,30,40])].concat(),5.,5);
        let c=first(bright.charges(&charges(&[0.076,0.02,0.02,0.02,0.02])));
        assert_eq!(c[1],0.076);assert!((c[2]-0.02).abs()<=0.076*(-10_f64).exp());assert_eq!(c[4],0.02);
        // Half the atoms dark: half the transient's charge stays.
        let mixed=Memory::of(&[atom(0,&[1,1,1,1]),atom(1,&[10,20,30,40])].concat(),5.,5);
        assert!((first(mixed.charges(&charges(&[0.076,0.,0.,0.,0.])))[4]-0.038).abs()<=0.076*(-10_f64).exp());
        // An output's states differ in their times' last bits: one group each.
        let mut a=atom(0,&[0]);a[0].time=5.000000000000002;let states=[atom(1,&[0]),a,atom(2,&[0,0])].concat();let groups=by_output(&states);
        assert_eq!(groups.iter().map(|g|g.len()).collect::<Vec<_>>(),vec![3,1]);
    }
    /// The statistics' room is continuous through the target (module documentation, # Shares): what the floor
    /// leaves of it, then from 1 − 2/k of it the weighed cost's balance 2F/(k − 2), at most the target -- the shallow probe's
    /// T at 8.99 % a priori and 0.884 % of bias gets 1.88 % instead of the 0.13 % sliver; the run's room is the same rule at
    /// its work exponent, and at the step's share of it the statistics' room is exactly the rest, the one optimum; the
    /// predicted need and weight move continuously as a floor crosses the turn and the target.
    #[test]
    fn the_room_is_continuous_through_the_target(){
        for target in TARGETS{
            let mut last=room(target,0.);assert_eq!(last,target);
            for i in 1..=80_000{let floor=i as f64*1e-4*target;let r=room(target,floor);
                assert!(r>0.&&r<=target&&(r-last).abs()<=1.0001e-4*target,"{floor}: {r} after {last}");last=r;
                if floor<=(1.-2./BEYOND)*target{assert!((r-(target-floor)).abs()<=1e-15);}}
            assert_eq!(room(target,8.*target),target);
            for i in 0..=500{let apriori=i as f64*0.01*target;let run=room_of(target,apriori,2.+1./ORDER);let bias=STEP_SHARE*run;
                if apriori<=2.*ORDER*target{assert!((room(target,apriori+bias)-(run-bias)).abs()<=1e-12*target,"{apriori}: {} against {}",room(target,apriori+bias),run-bias);}
                if apriori<=(1.-STEP_SHARE)*target{assert!((run-(target-apriori)).abs()<=1e-15);}
                else if apriori<=2.*ORDER*target{assert!((run-STEP_SHARE*(apriori+run)).abs()<=1e-12*target);}}
        }
        assert!((room(0.1,0.0899+0.00884)-2.*0.09874/(BEYOND-2.)).abs()<1e-15&&room(0.1,0.0899+0.00884)>0.0187);
        // One output after t = 0, T a reference: the need and the weight across the turn and the target.
        let need=|floor:f64|predicted(&[(0.,None),(0.01,None)],&[[0.;2],[floor,0.]],[0.;2],[true,false],32.,0.);
        let(n,w)=need(0.01);assert!(n==TASK as f64&&w==1.,"{n} {w}");let(n,w)=need(0.04);assert!((n-32.).abs()<1e-9&&(w-1.).abs()<1e-12,"{n} {w}");
        let(mut last,mut weight)=need(0.);
        for i in 1..=2000{let(n,w)=need(i as f64*5e-5);assert!((n-last).abs()<=0.03*last.max(1.)&&(w-weight).abs()<=0.02*weight,"{i}: {n} {w} after {last} {weight}");(last,weight)=(n,w);}
        assert!(weight>1.&&need(0.2).1>need(0.1).1);
    }
    /// After a miss the search moves by the criteria it evaluated (`next_step`): the D1 gray molasses at 10 us
    /// outputs, with the near light's dressing, missed at 33 steps by 1 % on the motion, its population not
    /// measured, and the bound's 8.86 sent it back to 99 steps (0.101 us) -- where without the dressing, a model
    /// 1e-3 apart, 38 met. On a model of its criteria (the bound population 8.86 at 33 steps as dt², the measured
    /// 0.016 at 99 as dt², the motion 0.1075 at 99 and 1.0104 at 33, times 1 + δ) the search keeps 33 or 34
    /// steps whatever δ within ±2 %, never the bound's 99.
    #[test]
    fn a_missed_step_moves_by_the_criteria_it_evaluated(){
        let most=1<<20;
        assert_eq!(next_step(&Check{steps:33,population:8.86,loss:0.,motion:1.0104},true,most),34);
        assert_eq!(next_step(&Check{steps:33,population:8.86,loss:0.,motion:1.0104},false,most),99);
        assert_eq!(next_step(&Check{steps:99,population:0.016,loss:0.,motion:0.1075},true,most),33);
        let power=(1.0104f64/0.1075).ln()/3f64.ln();
        for delta in [-0.02,-1e-3,0.,1e-3,0.02]{
            let check=|s:usize|{let x=s as f64;Check{steps:s,population:8.86*(33./x).powi(2),loss:0.,motion:0.1075*(99./x).powf(power)*(1.+delta)}};
            let measure=|s:usize|0.016*(99./s as f64).powi(2);
            let mut bound=vec![];search(43,most,&[],&check,None,&mut bound);
            let grown=*bound.iter().filter(|c|c.met()).min_by_key(|c|c.steps).unwrap();
            let mut measured=vec![];search(grown.steps,most,&bound,&check,Some(&measure as &dyn Fn(usize)->f64),&mut measured);
            let kept=measured.iter().filter(|c|c.met()).map(|c|c.steps).min().unwrap();
            assert!(grown.steps==99&&(kept==33||kept==34),"δ {delta}: grown {}, kept {kept} steps ({measured:?})",grown.steps);
        }
    }
    /// The step kept (`kept`) is the largest that met the targets, a finer one only for a predicted saving beyond the
    /// prediction's uncertainty: as the finer one's cost crosses the largest's on a flat landscape (the D1 gray
    /// molasses forced all-folded: 4 and 12 steps per 1 us, their costs within a few per cent) the largest stays; a
    /// saving beyond the uncertainty takes the finer one, whichever check of the largest step comes first.
    #[test]
    fn the_largest_met_step_is_kept_but_for_a_clear_saving(){
        let(coarse,measured,fine)=(Check{steps:4,population:6.9,loss:0.,motion:0.90},Check{steps:4,population:0.28,loss:0.,motion:0.90},Check{steps:12,population:0.03,loss:0.,motion:0.11});
        for(x,error,steps)in [(0.97,0.25,4),(1.03,0.25,4),(0.81,0.25,4),(0.79,0.25,12),(0.5,0.25,12),(0.97,0.,12),(1.03,0.,4)]{
            let cost=|c:&Check|if c.steps==4{1.}else{x};
            let k=kept(&[fine,coarse,measured],&cost,error).unwrap();
            assert_eq!(k.steps,steps,"cost {x}, error {error}");if steps==4{assert_eq!(k.population,0.28);}
        }
        assert!(kept(&[],&|_|1.,0.25).is_none());
        // A largest step of no finite cost is not kept.
        let none=|c:&Check|if c.steps==4{f64::INFINITY}else{1.};
        assert_eq!(kept(&[fine,measured],&none,0.25).unwrap().steps,12);
        // A met step leaves the statistics their balance by construction: the all-folded model's a-priori T estimate at 0.0905 of
        // 0.1, the run's room its balance 0.0905/(2p), T's tolerance the step's share s of it -- at that bias the statistics'
        // room is exactly the rest; T a reference, its tolerance the target itself.
        let(t,run)=(tolerances(true,&[[0.;2],[0.,0.0905]]),room_of(TEMPERATURE_TARGET,0.0905,2.+1./ORDER));
        assert!((t[1]-STEP_SHARE*run).abs()<1e-15&&(room(TEMPERATURE_TARGET,0.0905+t[1])-(run-t[1])).abs()<1e-12,"{t:?} {run}");
        assert_eq!(tolerances(false,&[[0.;2],[0.,0.0905]])[1],TEMPERATURE_TARGET);
    }
    /// The batches: the remaining need in whole groups, at least the width, at most doubling; on the
    /// GPU whole waves.
    #[test]
    fn a_batch_is_the_remaining_need_within_its_bounds(){
        assert_eq!(next_batch(300,256,32,TASK),48);
        assert_eq!(next_batch(260,256,32,TASK),32);
        assert_eq!(next_batch(845,256,32,TASK),256);
        assert_eq!(next_batch(700,640,640,640),640);
        assert_eq!(next_batch(4000,1280,640,640),1280);
        assert_eq!(next_batch(1700,1280,640,640),640);
    }
}
