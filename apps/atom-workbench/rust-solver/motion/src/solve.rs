//! The pipeline: request, reach and model, pilot, decisions, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the light; the initial components; the reach and the model with every unit whose existence
//!    gate fails kept (model.rs, SPEC §1.5); the trap the result reports, the outputs' plan, the initial ensemble and
//!    what the reach leaves out of it; T(0) by quadrature; the near light's tables over the reach (SPEC §2.3).
//! 2. The scout: the first member of every pilot group, at the attempt's first step and at twice it (# The step),
//!    screened. Where its atoms outgrow the reach, or a folded unit is to be kept at its states (`keep`), the attempt
//!    ends there and the next begins with that model: a model is settled for the price of a few trajectories, not of a
//!    pilot.
//!    The pilot: the scout's groups whole, `pilot_size` trajectories at the run's two levels (the scout's are its
//!    groups' first members and are not run again). Its groups have numbers of their own, from trajectory
//!    `MOST_TRAJECTORIES` on: beyond every number a group of the product takes, which is its place in the run's list
//!    (`product`). It checks the step -- on further pairs where they are the least work (# The step) --, reads the
//!    reach's test again, the model's units' errors at the states of the first held member of every group, and the
//!    tables' errors at one state of every trajectory (SPEC §5.4; `one_each`). Its trajectories at the run's step stay
//!    in the result: they are trajectories of the run's own model and step.
//! 3. The model's decisions (SPEC §5.2; # The model), the step's error taken at its share: at the scout's states; at
//!    the pilot's on every rung its check does not leave at once, before the check stands there or runs a pair
//!    (`decide`: the check's works are beside the model's errors read then, and an attempt these decisions end has
//!    paid for no pair); and at the pilot's once more when the step has stood, beside the tables' errors. A unit to
//!    keep ends the attempt, and the next begins with its model at the step this one stood at; a keep to undo ends it
//!    too (# The model).
//! 4. The product: batches of whole groups at the settled step, one level, every trajectory run to the last output,
//!    until every held quantity's total lies within its target (# Stopping) or the trajectories reach
//!    `MOST_TRAJECTORIES`.
//! 5. The result (contract §2.2, `assemble`).
//!
//! # The step (SPEC §2.2)
//! One step τ for the run (adapt.rs), which follows from what the scheme must resolve and from nothing of the outputs:
//! an output is read where it falls, inside a step (adapt.rs, # The reads), so outputs a tenth of a step apart take no
//! shorter a step than outputs a hundred steps apart. The ladder's rungs are the run's length (`Plan::end`) over the
//! even numbers, and below a rung its halves: the companion's step is twice the run's, and the run is a whole number of
//! the companion's steps, so no level's last step is a cut one. A step is
//! admissible where its nodes resolve what the scheme samples along an atom's path (`admissible`): the reference flow,
//! ω τ < π at the prepared trap's largest frequency; and the generator, whose change along the path the propagation
//! reads at nodes half a step apart, (|Ω| + |K| v) τ/2 < π for every residual beat Ω and wave vector K of the model's
//! entries at an atom's speed v. The ladder begins at the largest rung that is admissible for the run and whose double
//! is for the companion (`first_rung`): the run's step at the largest speed of the reach the model is judged over; the
//! companion's at the reach's rms speed. An attempt after the first begins at the step the one before stood at, where
//! that is the shorter. The first rung is a rung of the ladder, whose bottom is the step of `MOST_STEPS` steps a
//! trajectory: a request for which no rung down to it is admissible is refused with a message.
//!
//! The step's convergence is checked on the pilot, as a refinement is checked by hand: the pilot's trajectories run at
//! τ and, the same trajectory numbers -- the same initial atoms, the same jumps' draws by their number, the same
//! Brownian path (adapt.rs) --, at 2τ (`Levels`), both read at the same outputs at the same instants. A quantity's
//! step error at an output is e(τ) = [Q(2τ) − Q(τ)]/(2^p − 1); the check reads its size over the run, the mean square
//! M of e over the outputs, estimated without the noise's floor, with the delete-one-group jackknife of that estimate
//! (estimate.rs `between`): one number for the loss, for each row as shown and for the total T, each carried to the
//! outputs by the dynamics itself. An error that changes its sign in time counts by its size.
//!
//! - A comparison of two levels says something only where the coarser one still resolves what the scheme samples: the
//!   first rung's double is admissible for the typical atom, and stable for every atom.
//! - What the step's error takes of a quantity's budget is what the check does not exclude. A quantity's total is
//!   E² = A² + b² + S² ≤ ε² (# Stopping), every part at what its source leaves possible at the confidence z: the
//!   statistics at their interval, and the step at the far end of the read its rung is held by, b² = max(M̂, 0) + t·SE
//!   at one look's interval (`unexcluded`; on what is shown the largest of the rows' and the loss's). The clear part
//!   √(M̂ − t·SE) is what the check has seen: a total that held only that would state less than the check leaves
//!   possible in nearly every run. No share of a target is set aside for the step: the statistics have
//!   ε² − A² − b², and a step that the check reads poorly costs trajectories.
//! - So a rung is not passed or failed against a threshold (but for the limit of a rung whose order is untested,
//!   below); the check chooses by the predicted work that is left (`choose`), read again at every rung and after
//!   every batch of pairs. In trajectories at τ, with N(b) the
//!   trajectories the run's looks need beside the model's error and a step's part b (`needed`: the prediction the
//!   product stops by; no number where b leaves a look no room; the model's error as it is read at the pilot's
//!   states on the rung in hand, `check`):
//!   - standing: N(b) less the trajectories run, b what the pairs read do not exclude;
//!   - the rung j below: 2^j N(b/2^{pj}) -- the run over again at τ/2^j, beside what the order leaves of the step's
//!     part there (what that rung will inherit) -- and the check's trajectories at every rung between;
//!   - more pairs, P in all: ½ a trajectory for each further pair's companion, and the trajectories still to run at
//!     τ, N(b_P) or the pairs', whichever are more. The further pairs are the product's next groups: their finer
//!     level is the product's own. b_P is what P pairs could leave at the best (`narrowed`): were the step's error
//!     the least the read allows -- its near end L at one look, from zero: each quantity's own interval, as at its
//!     far end --, what the read holds above L, value and interval, is the pairs' noise, and falls as 1/P: the far
//!     end L + (b² − L)·P₀/P.
//!   Standing and the rung below are final -- no rung is climbed again, a part once booked is the run's -- and are
//!   priced at what is certain, the far end. Pairs can be taken back, and are run where at the least error the read
//!   allows they would leave less work than the lesser of the two certain choices; else that lesser one. So a step is
//!   left for the rung below, at twice the work, only on what its reads leave no doubt of; a read that the noise has
//!   put high costs the pairs that bring it down, not a rung; and every work is continuous in the reads, the work
//!   chosen is the work paid -- the part is booked and the statistics make up for it --, so where standing and the
//!   rung below turn they cost the same: a read at that turn, whichever side the draw puts it, does not decide what
//!   a run costs. (For one look and a run far above its pilot the turn is at 1/(1 − β) = 2/(1 − β/2^{2p}),
//!   β = b²/(ε² − A²) = 16/31 at p = 2: a consequence, no constant of the solve.)
//!   What the pairs could do is a hope, and it is bounded: no more of them are asked on a rung than cost what the
//!   rung below does -- they are lost with the rung, a trajectory and a half each --, so whatever they then show the
//!   check has spent at most twice the work of the rung taken at once (`choose`).
//!   Where nothing has a finite work -- no rung below at the steps' cap, no number of pairs within a run's
//!   trajectories, a read that is no number -- the work's bounds are used up (SPEC §5.1): the step stands, flagged,
//!   and the accounts hold what the check does not exclude all the same; the ledger flags the outputs that puts
//!   beyond their targets.
//! - The scout (eleven single trajectories) books nothing, and the pilot is to come on the rung it leaves standing.
//!   Its one choice is to leave the rung before a pilot is spent on it: it is taken only where the read shows the
//!   rung below to be the lesser work -- `choose` beside the least step error the read leaves possible, its clear
//!   part among the family's looks (`clear`), and no pairs to ask. The works' difference grows with the error: less
//!   work below at the near end is less for every error the read allows.
//! - A rung reached by a halving inherits the read of the rung above, lowered by the order (`lowered`: a mean square
//!   goes as τ^{2p}), and is held by whichever of the two reads has the lesser far end (`held`); the scout's
//!   inheritance serves the pilot on the rung the scout left standing. Its own read tests what both e and the
//!   inheritance rest on, e ∝ τ^p between the levels: where it lies clearly above the inherited one (`refuted`) the
//!   order does not hold at these steps -- neither read is of the step's error, nothing can be booked or predicted
//!   at this rung -- and the step is halved. An error that falls faster than the order (a coarser level outside its
//!   asymptotic regime) leaves the inherited read too high, and the rung's own holds it.
//! - A rung that inherits no read -- an attempt's first: two levels -- has its order untested: there
//!   e = [Q(2τ) − Q(τ)]/(2^p − 1) is an assumption, and two levels that are both too coarse can agree. What such a
//!   rung books is no bound of its error (measured, SPEC §2.2: the total T of a first rung read below 1 % and 2 %
//!   where runs a rung below show 2 to 13 %; a Raman transfer's rows read at 2.9 % where the analytic result lies
//!   10 % off). So the first rung's rule is kept as it was: it stands, or takes pairs, only where the part it books
//!   is within the step's share (`Bound::share`; units.rs `STEP_SHARE`), which leaves the target a factor 1 + 2p of
//!   the mean square above an estimate that the order has not been tested for. A convention, not a bound: the test
//!   of the order is the three levels of the rung below.
//! - The step's part is fixed after the pilot, the same at every output; the product runs one level and reads no
//!   step.
//! - The check's resolution -- √(t·SE), the rms error the pairs cannot tell from none -- is said in the result's
//!   method line beside the two steps compared and the number of pairs (`Checked`).
//! - Before the check is read the step's part is taken at the share of a step chosen continuously (units.rs
//!   `STEP_SHARE`: the least-work split of a budget between a step and its statistics): in the pilot's size, in the
//!   model's decisions (# The model) and in the results sent before the pilot's read. Of the steps it decides only
//!   what a rung of untested order may book (above).
//!
//! It is a check of the error's size over the run, not a bound at an output: an error confined to a short stretch of
//! the run is diluted by the run's length in the mean square; e rests on the scheme's order, which only three levels
//! test -- a first rung that stands is taken on two, within the step's share.
//! A bound per output was built and measured (the product at
//! both levels, a60231dd): the pairs part within 100 µs in six beams' cooling whatever the noise they share, so the
//! difference's error at the late outputs is two independent ensembles', and the measurement cost 30 to 240 % of a run
//! on top of it; pooling it over the outputs books less than the truth where the error grows in time (SPEC §2.2).
//!
//! # Outputs and windows
//! The outputs are the request's times, from 0 and evenly spaced. At an output a trajectory is read inside the step
//! the output falls in (`run`; step.rs `Step::at`): its rows as shown, its (r, κ) for the listed distributions, the
//! pilot's kept state, and its loss test there. T's window about an output (estimate.rs `half_window`) takes a
//! trajectory's κ and κ² over the cuts of its steps' paths (step.rs `Segment::moments`: the path with the kick spread
//! over the step, of the scheme's order at any cut); two outputs' windows never overlap (each at most half an interval
//! on either side), the first is the instant t = 0, and the run goes on through the last output's window (`Plan`).
//!
//! # The model (SPEC §5.2)
//! The units are the model's (model.rs), listed again in every model a keep closes. A unit whose fold does not exist over
//! the reach is kept, and so is one whose measured error leaves an output no room for the statistics. Beyond these the
//! model keeps a unit where keeping it pays (decide.rs `pays`): the run's predicted total work W = w·N, N the trajectories
//! its looks need beside its errors and the step's share (`Look::need` on the pilot's estimates) and w the
//! counted work a trajectory spends per µs (`rate`, the step's count through the model's generator at the pilot's states),
//! falls by a share of itself beyond that share's spread, the jackknife over the pilot's groups (`Pricing`) -- the kept
//! model's W counting the pilot its attempt runs again: every keep is a new attempt, a new model and its own pilot. A
//! kept model is built and rated only for a unit whose keep would pay were it to cost no more per µs than the fold
//! (`Pricing::saving_at_all`: the trajectories its error's removal saves, less the new pilot): a fold is made to cost less than
//! what it folds, so a keep that pays by its rate alone is not looked for -- where one exists the fold is at fault, not
//! the choice.
//!
//! The price of a keep knows the kept model's work a step, not the step it needs: the pricing reads no step. So a keep
//! made for its predicted saving is tried, and undone where its model's check is about to take a step at which it
//! costs more than the model before it did, (1 − share)·τ_before/τ > 1 (`Attempt::Undo`; `attempt`): the fold's error
//! stands again, measured and booked. A unit once kept for its saving is not priced again in the solve, undone or not
//! (`barred`). A model's rate is its mean over the run's held states, read at one state of every pilot trajectory,
//! the states spread evenly over the run (`one_each`), the same for every model.
//!
//! A unit is read at the states only where its error can matter. A unit whose gate bounds its error over the reach
//! (units.rs `Gate::error_bound`) is booked at that bound at every output, unread, as long as such bounds, the smallest
//! first, add up to no more than the share the most trajectories a solve runs leave unseen (`booked`; estimate.rs
//! `unseen`): what is shown is never booked statistics below that share (`Estimate::shown_statistics`), so what the
//! unread units' bounds overstate together lies below what any run resolves. The others are read, once per state: a
//! state's errors are kept as read and their means per output made once for every decision that asks (`Measured`).
//!
//! # Stopping (SPEC §3)
//! Every quantity held to a target -- at every output after t = 0 what is shown, and the total T -- has the total
//! E² = A² + b² + S² (ledger.rs; `Look`): A the model's error, from the units measured at the pilot's states and the
//! tables' errors there, and b the step's part, what the pilot's check does not exclude of its rms over the run
//! (# The step), both fixed by the pilot; S the statistics' interval. The run stops where every one of them lies
//! within its target: the test the ledger flags an output by (`Look::met`). The stop reads the run's standard errors
//! and nothing else of its estimates. A look that no number of trajectories brings within its target (A² + b² ≥ ε²:
//! the model's decisions have kept what they could, and the check had no step or pairs left to choose) is not chased:
//! the ledger flags it.
//!
//! The trajectories a look needs (`Look::need`): the statistics may take the room R² = ε² − A² − b², and fall as
//! N^{−1/2}; on what is shown they are no less than the unseen share at N (estimate.rs `unseen`, `draws_for`). A batch
//! brings the trajectories to the number the looks predict, at most to twice the number the prediction rests on
//! (`batch`). The same prediction, beside each step's part the check could book, is what the check chooses the step
//! by (# The step, `needed`).
//!
//! # T's target is on the total temperature
//! The user's target on T, 10 % relative at the confidence z, is on the total temperature: the mean of the three lab
//! axes' T (`Motion::temperature_total`), its statistics the jackknife of that mean. It is held at every output after
//! t = 0 where it is read -- its step's size read by the check, what the check does not exclude of it booked, its
//! look met or chased (`looks`), the total flagged where it exceeds the target (ledger.rs). Each axis's T is shown with its
//! own statistics and held to nothing: an axis that carries little of the energy does not set the run's length.
//! The validated regime k_BT_a ≥ `CLASSICAL` ħω_a is reported per output and axis (`Solution::classical`) and decides
//! nothing in the solve.
//!
//! The axis's frequency is the reported trap's along the lab axis, ω_a² = (ħ/m) ∂²V/∂x_a² at its bottom (model.rs
//! `Trap::lab`): T_a is the variance of κ along the lab axis over that frequency's period (# Outputs and windows), n̄_a
//! and the result's `trapFrequencyKHz` are read against it, and so is the regime -- one frequency for everything said of
//! an axis; the principal axes' own would test temperatures that are not estimated. In a trap whose principal axes e_i
//! are tilted against the lab axes it is the root mean square of the principal frequencies the axis is made of,
//! ω_a² = Σ_i (e_i·x_a)² ω_i², no less than their mean of the same weights (Jensen): with every mode at its own boundary,
//! k_BT_i = C ħω_i, the axis's T_a = Σ_i (e_i·x_a)² T_i = C ħ Σ_i (e_i·x_a)² ω_i/k_B ≤ C ħω_a/k_B, so a tilted axis is
//! reported inside the regime only where the modes it is made of are inside it in that weighted mean, never further
//! out.
//!
//! # Checkpoints
//! The decisions take counts, never the wall clock, so a request gives the same result on any machine. The wall clock only
//! times the messages (`Messages`): one clock for the whole solve, set at every message; whenever `CHECKPOINT` has passed
//! since the last one, the trajectories that are running (`run_groups`: the scout's, the pilot's, a batch's) send a
//! complete result of what the attempt holds at its step and of the groups finished so far in their order (a prefix:
//! groups chosen by their number alone). A result is passed on where it rests on two groups at least (the jackknife's
//! least for a spread: groups, of however many members each -- the scout's have one) and on no fewer trajectories than
//! the attempt's last one -- a pilot's first groups at a finer step are no better than the result before them --; else
//! the last result goes out again, and nothing where the solve has sent none yet: the message then stays due (the clock
//! is set by a message, not by a look) and goes out with the first group that makes a result pass. Between the
//! product's batches each total is a message: in a solve shorter than `CHECKPOINT` the first message is the product's
//! first total, after the pilot (SPEC §9). A message is made by loops that run on the trajectories' own pool, between
//! two trajectories of a thread (`run_groups`). What runs no trajectories sends nothing and looks at no clock: the
//! model's construction and its gates, the tables and their refinement, the units' reads at the states, the pricing
//! of keeps, T(0)'s quadrature.
use crate::adapt;
use crate::decide::{assured, pays, Read};
use crate::ensemble::{Ensemble, Quadrature, Start, Well, GROUP};
use crate::estimate::{between, draws_for, half_window, occupation, spread, summarize, unseen, Between, Estimate, Fate, Group, Held, Scales, Spread};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, span, when, Account, Ledger};
use crate::linalg::V3;
use crate::model::{Envelope, Lights, Model, Run, Sample, Trap, Unit, GROUND};
use crate::request::STATES;
use crate::step::{Dynamics, Scratch, Segment, Trajectory};
use crate::units::{hbar_over_mass, CLASSICAL, CONFIDENCE, KT_PER_UK, MOST_STEPS, MOST_TRAJECTORIES, ORDER, STEP_SHARE, TEMPERATURE_TARGET};
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::f64::consts::{PI, TAU};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The population target the app checks a result against by equality (contract §2.2 `requestedTolerance`).
pub use crate::units::POPULATION_TARGET;
use crate::request::Request;

/// The runs' seed, the key of every trajectory's streams (random.rs): any fixed value, fixed so that a request gives the
/// same result every time.
const SEED: u64 = 0;
/// The output times the distributions are given at (contract §3.4: at most 21, t = 0 and the last among them): the
/// contract's number, for the display alone -- no estimate reads a distribution.
const DISTRIBUTION_TIMES: usize = 21;
/// Half the time the app waits for a message before it kills the worker (contract §2.5: 120 s): the checkpoints'
/// interval. The other half is left to what runs between two messages and sends none (module documentation,
/// # Checkpoints): a margin, not a bound derived from those stages' work. It times messages and decides no result.
const CHECKPOINT: Duration = Duration::from_secs(60);
/// What the engine did with a folded unit (the ledger's `action`, shown by the app): its booked error is read at the
/// pilot's states (SPEC §5.2).
const FOLD_MEASURED: &str = "二阶消去（折叠），沿试探轨迹实测";
/// What the engine did with a folded unit it did not read (`booked`): its gate's bound over the reach is its error.
const FOLD_BOUNDED: &str = "二阶消去（折叠），按可达范围上的界记账";
/// What the engine did with a near table's error (SPEC §5.4): refined over the reach when it was built, and read
/// against the elimination itself at the pilot's states.
const TABLE_MEASURED: &str = "按可达范围加密的表，在试探状态上对照消去本身实测";

/// A distribution along one lab axis on one grid (strictly increasing, at least two points) at some of the output
/// times (`samples`: increasing indices, the last output among them), each the probability at every grid point
/// (contract §2.2 `momentum`, `position`).
#[derive(Clone, Debug, PartialEq)]
pub struct Distributions {
    pub grid: Vec<f64>,
    pub samples: Vec<usize>,
    pub probabilities: Vec<Vec<f64>>,
}

/// A solve's result at every output time of the request, per lab axis x, y, z (contract §2.2). Everything but the
/// loss describes the atom still held.
#[derive(Clone, Debug)]
pub struct Solution {
    /// The held atom's populations, p = P/(1 − L) per output (row 0 the request's initial diagonal), before the
    /// adapter's projection onto the simplex.
    pub populations: Vec<[f64; STATES]>,
    /// Per output the total estimated error of what is shown, the rows as shown and the loss (ledger.rs `Account`).
    pub estimated_errors: Vec<f64>,
    /// T per lab axis (µK) and its standard error, none where fewer than two atoms or groups read it; T_total =
    /// (T_x + T_y + T_z)/3 and its own standard error (estimate.rs `Motion`).
    pub temperature_uk: Vec<[f64; 3]>,
    pub temperature_errors_uk: Vec<[Option<f64>; 3]>,
    pub temperature_total_uk: Vec<f64>,
    pub temperature_total_errors_uk: Vec<Option<f64>>,
    /// The mean velocity (m/s) and n̄ per lab axis, n̄ relative to `trap_frequency_khz`.
    pub mean_velocity_mps: Vec<[f64; 3]>,
    pub mean_occupation: Vec<[f64; 3]>,
    /// ω/2π (kHz) along each lab axis at the bottom of the reference state's trap.
    pub trap_frequency_khz: [f64; 3],
    /// The probability that the atom is lost by each output, never decreasing.
    pub loss_probability: Vec<f64>,
    /// Velocities (m/s) and positions (µm from the focus) per lab axis.
    pub velocity: [Distributions; 3],
    pub position: [Distributions; 3],
    /// Per output and lab axis: in the validated regime, k_BT_a ≥ CLASSICAL ħω_a at the run's own T_a (reporting only; the
    /// contract's `classical` is every axis's, contract §2.2).
    pub classical: Vec<[bool; 3]>,
    /// Per output: the total T's estimated error (relative, ledger.rs `Account`: model, step and statistics, weighed by
    /// the held share), none where one of the three is not read. T's target is on the total T, the mean over the three
    /// lab axes; each axis's T is shown with its statistics.
    pub temperature_estimated_errors: Vec<Option<f64>>,
    pub trajectories: u64,
    /// The run's step τ (µs, adapt.rs).
    pub time_step_us: f64,
    pub method: String,
    /// What keeps the worst output beyond the population target (Chinese): present where its total exceeds it.
    pub reason: Option<String>,
    pub approximations: Ledger,
}

/// The solve of a request (module documentation): every checkpoint handed to `checkpoint` with the estimated wall time
/// left (ms) where the run's pace gives one, the final solution returned; partial results are never withheld.
pub fn solve(request: &Request, checkpoint: &mut dyn FnMut(&Solution, Option<f64>)) -> Result<Solution, String> {
    request.validate()?;
    let light = Light::new(request);
    let (mut kept, mut reach, mut step) = (vec![], request.temperature_uk, None);
    // The last keep made for its predicted saving (`Attempt::Keep`): the units kept before it, the step then and the
    // saving's share; and the units such a keep has been undone for (`Attempt::Undo`), not priced again.
    let (mut priced, mut barred): (Option<(Vec<Unit>, f64, f64)>, Vec<Unit>) = (None, vec![]);
    let mut messages = Messages { send: checkpoint, clock: Instant::now(), last: None, least: 0 };
    loop {
        // An attempt's results improve on its own last one (module documentation, # Checkpoints).
        messages.least = 0;
        match attempt(request, &light, &mut kept, reach, step, priced.as_ref().map(|p| (p.1, p.2)), &barred, &mut messages)? {
            Attempt::Grow(t, tau) => (reach, step) = (t, Some(tau)),
            Attempt::Keep(u, tau, share) => {
                priced = share.map(|share| (kept.clone(), tau, share));
                barred.extend(share.map(|_| u.clone()));
                kept.push(u);
                step = Some(tau);
            }
            Attempt::Undo => {
                let (before, tau, _) = priced.take().expect("only an attempt told of a priced keep undoes one");
                (kept, step) = (before, Some(tau));
            }
            Attempt::Done(s) => return Ok(s),
        }
    }
}

/// The solve's messages to the app (module documentation, # Checkpoints).
struct Messages<'a> {
    send: &'a mut dyn FnMut(&Solution, Option<f64>),
    /// The last message's time.
    clock: Instant,
    /// The last result sent.
    last: Option<Solution>,
    /// The trajectories a result of this attempt must rest on to be sent.
    least: u64,
}
impl Messages<'_> {
    /// A result of the solve so far, of `groups` groups and `trajectories` trajectories, with the wall time left (ms)
    /// where the run's pace gives one: made (`made`) and sent where it rests on two groups at least -- the jackknife's
    /// least for a spread, whatever the members a group has so far (the scout's groups have one) -- and improves on the
    /// attempt's last one, else the last result again. A result that is not sent is not made: with no group in there is
    /// none to make.
    fn result(&mut self, groups: usize, trajectories: u64, remaining: Option<f64>, made: impl FnOnce() -> Solution) {
        if groups >= 2 && trajectories >= self.least {
            let s = made();
            self.least = trajectories;
            (self.send)(&s, remaining);
            self.last = Some(s);
            self.clock = Instant::now();
        } else {
            self.again();
        }
    }
    /// Nothing new: the last result again. With none yet there is nothing to send and the clock stands: the message
    /// stays due, and goes out with the first result that passes (`run_groups`).
    fn again(&mut self) {
        if let Some(s) = &self.last {
            (self.send)(s, None);
            self.clock = Instant::now();
        }
    }
    /// The time until the next message is due.
    fn due(&self) -> Duration {
        CHECKPOINT.saturating_sub(self.clock.elapsed())
    }
}

/// How one attempt at a model ends (`attempt`): the pilot's atoms outgrew the reach the model was judged over (its new
/// temperature, µK), a unit is to be kept -- each with the step the attempt stood at, the next attempt's first rung; a
/// keep with the share of the work it was predicted to save where that is why it is kept (`keep`) --, the last such
/// keep is undone (its model's step, once checked, makes it dearer than the model before it), or the solution.
enum Attempt {
    Grow(f64, f64),
    Keep(Unit, f64, Option<f64>),
    Undo,
    Done(Solution),
}

/// The initial components (SPEC §2.5): the ground states the request's initial density holds and their populations
/// normalized over them -- one at least: a validated request's density is diagonal and on the ground states
/// (request.rs `Request::validate`).
fn components(request: &Request) -> Vec<(usize, f64)> {
    let diagonal: Vec<f64> = (0..GROUND).map(|s| request.initial_density[(s, s)].re).collect();
    let total: f64 = diagonal.iter().sum();
    (0..GROUND).filter(|&s| diagonal[s] > 0.).map(|s| (s, diagonal[s] / total)).collect()
}

/// The ground populations of a basis state alone.
fn alone(state: usize) -> [f64; GROUND] {
    std::array::from_fn(|s| if s == state { 1. } else { 0. })
}

/// One initial component's preparation well (ensemble.rs `Well`) from the model: its prepared potential -- zero at the far
/// level where the trap light is Gaussian beams, the ensemble refusing any other --, the trap, the envelope's beams and peak.
struct Prepared<'a> {
    model: &'a Model,
    light: &'a Light,
    populations: [f64; GROUND],
    lights: Lights,
    trap: Trap,
    envelope: Envelope,
}
impl<'a> Prepared<'a> {
    fn new(model: &'a Model, light: &'a Light, populations: [f64; GROUND]) -> Result<Prepared<'a>, String> {
        Ok(Prepared { trap: model.trap(&populations, &model.prepared())?, envelope: model.envelope(&populations), lights: model.prepared(), model, light, populations })
    }
}
impl Well for Prepared<'_> {
    fn potential(&self, r: V3) -> (f64, V3) {
        let q = self.model.expansion(r, &self.populations, &self.lights);
        (q.value, q.gradient)
    }
    fn trap(&self) -> (V3, [V3; 3], V3) {
        (self.trap.centre, self.trap.axes, self.trap.frequencies)
    }
    fn envelope(&self) -> (Vec<Geometry>, f64) {
        (self.envelope.beams.iter().map(|&l| self.light.beams[l]).collect(), self.envelope.peak())
    }
}

/// Whether a step is admissible for an atom of a speed (module documentation, # The step): its nodes resolve what the
/// scheme samples along the atom's path.
/// - The reference flow, at the prepared trap's largest frequency ω (rad/µs): ω τ < π -- there the two half flows of a
///   step turn the motion by π and any kick between them grows. The same for every atom.
/// - The generator: the propagation reads its change along the path at three nodes, the step's middle and its two
///   ends (step.rs, # The kick), τ/2 apart, and takes it for the parabola through them. An entry that turns at the rate
///   ρ along the path -- its residual beat Ω in the cluster's frames and its residual wave vector K at the atom's
///   velocity, ρ = |Ω + K·v| ≤ |Ω| + |K| v (`sampled`: the model's own entries, per beat the largest |K|) -- advances by
///   ρ τ/2 from node to node; at π the nodes read it as its alias. So (|Ω| + |K| v) τ/2 < π for every entry.
/// Both are edges -- where the scheme stops meaning anything --, not accuracies within a share: how well a step below
/// them does is the check's to say (the pilot's two levels). The frequency is the prepared trap's alone: the curvature
/// of the near light's own wells is no condition here.
fn admissible(tau: f64, frequency: f64, sampled: &[[f64; 2]], speed: f64) -> bool {
    frequency * tau < PI && sampled.iter().all(|[beat, wave]| 0.5 * (beat + wave * speed) * tau < PI)
}

/// The ladder's first rung (module documentation, # The step): the largest rung -- the run's length `run` (µs) over an
/// even number, its double a whole part of the run too -- that is admissible for the run and whose double is admissible
/// for the companion (`admissible`), at the speeds `fastest` and `typical` (µm/µs) of the reach the model is judged over.
/// (Admissibility is a bound on the step from above: the least even number is found by counting up.)
/// - The run's step is the result's: it is admissible for every atom the model is judged for, at the reach's largest
///   speed (model.rs `Reach::speed`) -- one domain for the model's folds and for the step.
/// - The companion at twice the step is a probe of the step, not a result: it must be stable, which the trap's
///   resonance decides for every atom alike, and resolve the light for the typical atom, at the reach's rms speed
///   (`typical_speed`) -- two steps one of which reads the light's alias at most atoms agree or differ for no reason of
///   the step, while the few fastest atoms beyond its limit weigh in the difference by their share.
/// Where the trap's resonance binds, the first rung is the rung below the largest stable one. Half the run at the
/// most: the companion's step is a rung itself, the run its one step then. Nothing of the outputs enters: they are read
/// inside the steps (adapt.rs, # The reads). The ladder has one bottom, the step at which a trajectory takes `MOST_STEPS`
/// steps, and the first rung is a rung of it: none where no rung down to that one is admissible -- the light or the
/// trap asks for a shorter step than the cap allows, or a frequency or a speed is no number (nothing is admissible
/// then). The solve refuses such a request (`attempt`): a step that is not admissible computes nothing.
fn first_rung(run: f64, frequency: f64, sampled: &[[f64; 2]], fastest: f64, typical: f64) -> Option<f64> {
    (1..=MOST_STEPS / 2).map(|pairs| run / (2 * pairs) as f64).find(|&step| admissible(step, frequency, sampled, fastest) && admissible(2. * step, frequency, sampled, typical))
}

/// The rms speed (µm/µs) of atoms at the temperature T (µK): √(3 (ħ/m) k_BT/ħ), the reach's typical atom at the
/// mean kinetic temperature of the ensemble the reach is made for (model.rs `Reach::kinetic_uk`).
fn typical_speed(hbar_over_mass: f64, temperature_uk: f64) -> f64 {
    (3. * hbar_over_mass * KT_PER_UK * temperature_uk).sqrt()
}

/// The reads a step is judged by (module documentation, # The step): the mean squares over the run of the step's
/// error on the loss, on every row as shown and on the total T where the levels read it. One family: what is clear of
/// one of them is clear among them all.
fn judged(read: &Between) -> Vec<Read> {
    std::iter::once(read.loss).chain(read.shown).chain(read.temperature).collect()
}

/// Every read of a check, each made by `f`.
fn each(read: &Between, f: impl Fn(&Read) -> Read) -> Between {
    Between { loss: f(&read.loss), shown: read.shown.each_ref().map(&f), temperature: read.temperature.as_ref().map(&f) }
}

/// A read as the scheme's order gives it at half the step (module documentation, # The step): a mean square goes as
/// the step to twice the order, so each read's value and spread over 2^{2p} -- what the rung below inherits.
fn lowered(read: &Between) -> Between {
    let part = 0.5f64.powi(2 * ORDER);
    each(read, |r| Read { value: part * r.value, spread: Spread { error: part * r.spread.error, ..r.spread } })
}

/// A rung's own read as more pairs could give it at the best (module documentation, # The step), on N pairs where it
/// has `part`·N: the step's error being the least the read allows -- its mean square L the read's near end, from
/// zero, at one look's interval: each quantity's own, as its far end is (`unexcluded`) --, what the read holds above
/// L is the pairs' noise. The further pairs then read L, and the whole read L + (max(M̂, 0) − L)·part; its spread
/// is the noise floor's, which falls as the floor does, as 1/N: the spread at `part` of itself. So the far end goes
/// from what it is, at part = 1, towards L as 1/N. (Where an error is resolved its spread falls as N^{−1/2} only:
/// more pairs are then asked than pay, a batch at a time, and each batch is read before the next.) A read that is
/// no number stays none.
fn narrowed(read: &Between, part: f64) -> Between {
    let value = |r: &Read| {
        let least = r.least(1.).max(0.);
        least + (r.value.max(0.) - least) * part
    };
    each(read, |r| Read { value: if r.value.is_nan() { r.value } else { value(r) }, spread: Spread { error: part * r.spread.error, ..r.spread } })
}

/// Whether a rung's own read refutes the read it inherits (module documentation, # The step): some look's mean square
/// lies clearly above what the order allows it -- its near end above the inherited read's far end, both among the
/// family's looks. The step's error does not fall at the scheme's order between these levels. The order is tested
/// look by look, on the looks both rungs read: the loss and every row always; the total T where both read it -- a T
/// read at one rung alone has nothing to be compared with, is no look of this family, and keeps no other look from
/// being tested.
fn refuted(own: &Between, inherited: &Between) -> bool {
    let both: Vec<(Read, Read)> =
        std::iter::once((own.loss, inherited.loss)).chain(own.shown.iter().copied().zip(inherited.shown)).chain(own.temperature.zip(inherited.temperature)).collect();
    let looks = both.len() as f64;
    both.iter().any(|(own, inherited)| own.least(looks) > inherited.most(looks))
}

/// What a rung is held by (module documentation, # The step): per look, of its own read and the one it inherits from
/// the rung above (`lowered`), the one whose far end is the lesser -- each bounds the rung's mean square.
fn held(own: &Between, inherited: Option<&Between>) -> Between {
    let Some(inherited) = inherited else { return own.clone() };
    let lesser = |a: &Read, b: &Read| if b.most(1.) < a.most(1.) { *b } else { *a };
    Between {
        loss: lesser(&own.loss, &inherited.loss),
        shown: std::array::from_fn(|s| lesser(&own.shown[s], &inherited.shown[s])),
        temperature: own.temperature.map(|a| inherited.temperature.map_or(a, |b| lesser(&a, &b))),
    }
}

/// The step's part of an output's accounts (module documentation, # The step): on what is shown -- the largest of the
/// rows' and the loss's, a probability's error at most 1 -- and on the total T (relative). The step's share until the
/// pilot's two levels are read; then what their check does not exclude (`unexcluded`), the same at every output,
/// unbounded on T where no output reads it at both levels.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Bound {
    shown: f64,
    temperature: f64,
}
impl Bound {
    /// The step's share of each target (`step_bound`).
    fn share() -> Bound {
        let [shown, temperature] = step_bound();
        Bound { shown, temperature }
    }
    /// The part the scheme's order leaves of it `rungs` halvings below: an error goes as the step to the order.
    fn lowered(&self, rungs: u32) -> Bound {
        let part = 0.5f64.powi(ORDER * rungs as i32);
        Bound { shown: part * self.shown, temperature: part * self.temperature }
    }
}

/// What a check does not exclude (module documentation, # The step; SPEC §5.1), from the reads a rung is held by
/// (`held`): each quantity's far end, the rms the read leaves possible at the confidence z, √(max(M̂, 0) + t·SE) at
/// one look's interval -- every quantity's statement is its own (decide.rs, # Many reads) --, the largest on what is
/// shown, and the total T's; unbounded where a read excludes nothing. This is the step's part of the accounts.
fn unexcluded(held: &Between) -> Bound {
    let far = |read: &Read| read.most(1.).sqrt();
    Bound { shown: held.shown.iter().map(far).fold(far(&held.loss), f64::max).min(1.), temperature: held.temperature.map_or(f64::INFINITY, |t| far(&t)) }
}

/// What a check has seen (module documentation, # The step): the clear part of each rms among the family's looks --
/// √(M̂ − t·SE) at the family's interval, where positive: clear is one statement, the largest of many noisy reads
/// (decide.rs, # Many reads) --, the largest on what is shown, and the total T's (none where it is not read). The
/// least step error the reads leave possible.
fn clear(read: &Between) -> Bound {
    let looks = judged(read).len() as f64;
    let near = |read: &Read| read.least(looks).max(0.).sqrt();
    Bound { shown: read.shown.iter().map(near).fold(near(&read.loss), f64::max).min(1.), temperature: read.temperature.map_or(0., |t| near(&t)) }
}

/// A check's resolution (module documentation, # The step): the rms error over the run that its pairs cannot tell from
/// none, √(t·SE) of the mean square at one look, the largest on what is shown (absolute) and the total T's (relative;
/// unbounded where it is not read).
fn resolution(read: &Between) -> [f64; 2] {
    let of = |read: &Read| read.spread.interval().sqrt();
    [read.shown.iter().map(of).fold(of(&read.loss), f64::max), read.temperature.map_or(f64::INFINITY, |t| of(&t))]
}

/// The step's check as the result tells it (module documentation, # The step): the companion's step (µs), the pairs
/// compared, and the check's resolution (`resolution`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Checked {
    companion: f64,
    pairs: u64,
    resolution: [f64; 2],
}

/// The trajectories a run is predicted to take beside the step's part `step` (module documentation, # The step): the
/// largest need of `looks` (`Look::need`: the prediction the product stops by), no fewer than the pilot; no number
/// where the step's part leaves a look no room. `looks`: those some number of trajectories brings within their
/// targets beside the model's error alone -- the others no step meets, and the ledger flags them. The total T's looks
/// take part where the step's part on T is a number: a T the levels do not read has no account (ledger.rs) and is
/// held to nothing.
fn needed(looks: &[Look], step: &Bound) -> f64 {
    let need = |l: Look| if l.fits() { l.need() } else { f64::INFINITY };
    looks.iter().filter(|l| l.shown || step.temperature.is_finite()).map(|l| need(l.beside(step))).fold(pilot_size() as f64, f64::max)
}

/// What a check does with its rung (module documentation, # The step).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    /// The step stands on the pairs read.
    Stand,
    /// More pairs at the step and its double: the pairs to have.
    Pairs(u64),
    /// The rung below.
    Rung,
    /// Nothing has a finite predicted work: no number of pairs within a run's trajectories could leave the looks
    /// room, and no rung of the ladder below does.
    Neither,
}

/// The check's choice by the predicted work that is left (module documentation, # The step), in trajectories at the
/// step τ -- a trajectory's steps go as 1/τ. `now`: the step's part of the accounts as the pairs read give it;
/// `could`: that part as a number of pairs could give it at the best (`narrowed`); `limit`: the most this rung may
/// book, where it has one (`check`: a rung whose order is untested); `have`: the pairs read; `most`: the most the
/// check may come to (`have`: it asks none); `run`: the trajectories its rung's check began with; `rungs`: the rungs
/// the ladder has below.
/// - Standing: the trajectories still to run at τ beside `now`, need − have (`needed`); not to be had where `now` is
///   beyond the rung's limit, on what is shown or on a total T that is read.
/// - The rung j below: the run over again at τ/2^j beside the part the order leaves of `now` there
///   (`Bound::lowered`: what that rung inherits; it is held to no more), 2^j·need, and the check's trajectories at
///   every rung between, (2^j − 2)·run (the rung above the last is its companion). What has been run at τ is spent in
///   either case: here it counts for the run, there it is the companion.
/// - P > `have` pairs, whole groups: their companions at 2τ, half a trajectory each, and the trajectories still to
///   run at τ, the run's beside `could` at P or the pairs', whichever are more: ½(P − have) + max(need, P) − have;
///   not to be had where `could` at P is beyond the rung's limit.
/// Standing and the rung below are final -- no rung is climbed again, and a part once booked is the run's --, and
/// both are priced at what is certain, `now`. Pairs are the one choice that can be taken back: they are run where, at
/// the least error the reads allow, they would leave less work than the lesser of the two certain ones -- a batch of
/// them, towards their best number and at most to twice the pairs the prediction rests on (`batch`), and the check
/// is read again. Else the lesser of the certain ones, the rung in hand where they are equal. So a step is left for
/// the rung below, at twice the work, only on what the reads leave no doubt of; a read that the noise has put high
/// costs the pairs that bring it down, not a rung; and where standing and the rung below turn they cost the same
/// (the part is booked, and the run's statistics make up for it): a read at that turn does not decide a run's cost.
///
/// What the pairs could do is a hope, and it is bounded: the rung's further pairs are lost with it -- a trajectory
/// at τ and half a one at 2τ each, 3/2 (P − run) --, and no more of them are asked than cost what the rung below
/// does, P ≤ run + below/(3/2). Whatever the pairs then show, the check has spent at most twice the work of the rung
/// taken at once -- the least any rule can promise that does not know what the pairs will show (the rule of the
/// rented skis: rent until the rent paid is the price).
fn choose(looks: &[Look], now: &Bound, could: &dyn Fn(u64) -> Bound, limit: Option<&Bound>, have: u64, most: u64, run: u64, rungs: u32) -> Choice {
    let within = |part: &Bound| limit.map_or(true, |limit| part.shown <= limit.shown && !(part.temperature.is_finite() && part.temperature > limit.temperature));
    let here = |part: &Bound| if within(part) { needed(looks, part) } else { f64::INFINITY };
    let standing = here(now).max(have as f64) - have as f64;
    let below = (1..=rungs)
        .map(|j| {
            let parts = 2f64.powi(j as i32);
            parts * needed(looks, &now.lowered(j)) + (parts - 2.) * run as f64
        })
        .fold(f64::INFINITY, f64::min);
    let on = |pairs: u64| 0.5 * (pairs - have) as f64 + here(&could(pairs)).max(pairs as f64) - have as f64;
    let most = if below < f64::INFINITY { most.min(run + (below / 1.5) as u64) } else { most };
    let (more, pairs) = (have / GROUP + 1..=most / GROUP).map(|g| (on(g * GROUP), g * GROUP)).min_by(|a, b| a.0.total_cmp(&b.0)).unwrap_or((f64::INFINITY, have));
    if more < standing.min(below) {
        Choice::Pairs(batch(pairs as f64, have))
    } else if below < standing {
        Choice::Rung
    } else if standing < f64::INFINITY {
        Choice::Stand
    } else {
        Choice::Neither
    }
}

/// The estimates, the listed outputs' atoms and the counts of some groups.
fn pooled(parts: &[&Part], scales: &Scales) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts) {
    let estimates = summarize(parts, scales);
    let mut listed: Vec<Vec<(V3, V3)>> = vec![vec![]; parts.first().map_or(0, |p| p.listed.len())];
    let mut counts = Counts::default();
    for p in parts {
        for (l, x) in listed.iter_mut().zip(&p.listed) {
            l.extend_from_slice(x);
        }
        counts.add(&p.counts);
    }
    (estimates, listed, counts)
}

/// A message while the pilot's members come in: the result of what is in so far -- the parts in hand and those of the
/// members now running, each as a group of its own (the last result again where they are fewer than two groups,
/// `Messages::result`).
fn so_far(context: &Context, have: &[Part], more: &[&Part], messages: &mut Messages) {
    let all: Vec<&Part> = have.iter().chain(more.iter().copied()).collect();
    let (e, l, c) = pooled(&all, &context.scales);
    let n = e.first().map_or(0., |x| x.trajectories + x.failed) as u64;
    messages.result(all.len(), n, None, || assemble(context, &e, &l, &c, n));
}

/// The reach's falsifier (SPEC §1.8; decide.rs: a booked bound that a read can only refute): the pilot's T where it has
/// outgrown the reach's temperature `reach` (µK: the largest kinetic temperature along an axis of the ensemble the
/// reach was made for, model.rs `Reach::kinetic_uk`) -- some output's T above it beyond its interval among all the T read
/// (Šidák over the outputs and axes) --; then the largest T read, the reach's new temperature. A T that reads above the
/// reach within its interval leaves the reach standing, as every unreadable falsifier's read does.
fn outgrown(estimates: &[Estimate], reach: f64) -> Option<f64> {
    let read: Vec<&crate::estimate::Motion> = estimates.iter().skip(1).filter_map(|e| e.motion.as_ref()).collect();
    let looks = 3. * read.len() as f64;
    let beyond = read.iter().any(|m| (0..3).any(|a| m.temperature[a] - m.temperature_errors[a].interval_over(looks) > reach));
    beyond.then(|| read.iter().flat_map(|m| m.temperature).fold(reach, f64::max))
}

/// Whether a lab axis's temperature (µK) lies in the validated regime of that axis's frequency (rad/µs), k_BT ≥
/// `CLASSICAL` ħω (SPEC §0, §3): what the result reports as classical, per axis; reporting only.
fn classical(temperature: f64, frequency: f64) -> bool {
    KT_PER_UK * temperature >= CLASSICAL * frequency
}

/// The states the model is read at and the errors there of the units read, `units` (SPEC §5.2): at every output the
/// first member of every group of `parts` that still holds its atom -- one a group keeps every group in a mean's spread,
/// and a unit's error is a function of the state; the first held one, so that an output some atom reaches is read
/// wherever a group has one there --, with each state's output and group, and per state the units' errors in their
/// order. A state `known` already holds -- the same trajectory at the same output and step, the scout's inside the pilot
/// -- keeps the errors read there: a unit's error is a function of the state.
///
/// The errors are one table (`Measured::errors`): the states' rows one after another, a row the units' errors on the
/// populations and on T, each row written on the pool where it is read. One block of memory, not one a state: at a
/// thousand outputs the states are ten thousand, and that many blocks took the calling thread 0.2 to 0.6 s to free.
struct Sampled {
    /// Each state's trajectory and step.
    of: Vec<(u64, f64)>,
    at: Vec<usize>,
    from: Vec<usize>,
    errors: Vec<[f64; 2]>,
}
fn sampled(model: &Model, units: &[Unit], tables: &crate::model::tabulate::Tables, parts: &[&Part], known: Option<Sampled>) -> Sampled {
    let mut states: Vec<&State> = vec![];
    let (mut at, mut from) = (vec![], vec![]);
    for (g, p) in parts.iter().enumerate() {
        for (k, s) in p.samples.iter().enumerate() {
            if let Some(state) = s.iter().min_by_key(|x| x.index) {
                states.push(state);
                at.push(k);
                from.push(g);
            }
        }
    }
    let key = |index: u64, at: usize, tau: f64| (index, at, tau.to_bits());
    let read: std::collections::HashMap<(u64, usize, u64), usize> =
        known.iter().flat_map(|k| k.of.iter().zip(&k.at).enumerate().map(|(j, (&(index, tau), &at))| (key(index, at, tau), j))).collect();
    let width = units.len();
    let mut errors = vec![[0.; 2]; states.len() * width];
    // A state's row (the table is empty where no unit is read).
    errors.par_chunks_mut(width.max(1)).zip(&states).zip(&at).for_each(|((row, state), &k)| match (&known, read.get(&key(state.index, k, state.tau))) {
        (Some(known), Some(&j)) => row.copy_from_slice(&known.errors[j * width..][..width]),
        _ => row.iter_mut().zip(model.errors_at(units, &state.sample, Some(tables))).for_each(|(row, e)| *row = [e.population, e.temperature]),
    });
    Sampled { of: states.iter().map(|s| (s.index, s.tau)).collect(), at, from, errors }
}

/// How a folded unit's error is booked (module documentation, # The model): read at the pilot's states -- its place
/// among the units read --, or its gate's bound over the reach.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Booked {
    Read(usize),
    Bound(f64),
}

/// How each folded unit is booked, from the units' bounds over the reach, none where a unit's gate bounds no error of
/// its (units.rs `Gate::error_bound`): the bounded units taken by their bounds, the smallest first (equal ones in the
/// units' order), are booked at them while the bounds so booked add up to no more than the unseen share of
/// `MOST_TRAJECTORIES` draws, z²/(N + z²) -- the least statistics a solve ever books on what is shown; every other
/// unit is read. So the model's error at an output is at most that share above the one with every unit read, on the
/// populations (absolute) and on T (relative) alike.
fn booked(bounds: &[Option<f64>]) -> Vec<Booked> {
    let mut bounded: Vec<(f64, usize)> = bounds.iter().enumerate().filter_map(|(u, b)| b.map(|b| (b, u))).collect();
    bounded.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut unread: Vec<Option<f64>> = vec![None; bounds.len()];
    let mut sum = 0.;
    for (bound, u) in bounded {
        sum += bound;
        if sum > unseen(MOST_TRAJECTORIES as f64) {
            break;
        }
        unread[u] = Some(bound);
    }
    let mut read = 0;
    unread
        .iter()
        .map(|bound| match *bound {
            Some(bound) => Booked::Bound(bound),
            None => {
                read += 1;
                Booked::Read(read - 1)
            }
        })
        .collect()
}

/// The folded unit to keep, if any (SPEC §5.2), from the units' errors `measured` at the states of the trajectories
/// `parts` (their `estimates`; the units' mean errors over them `means`, per output and unit) and the errors `fixed`
/// beside them, the step's taken at its share (a pilot's measured
/// bound is its own noise): a unit whose error leaves an output no room for the statistics -- of the
/// outputs left none, the unit of the largest weight there (its errors on what is shown and on T where they were read,
/// weighed by the held share as the totals are, each over its target, added up) --; else, of the units whose error's removal alone would pay
/// for the `again` trajectories its model runs before it stands where this one does, each rated with its kept model,
/// the one whose keeping saves the most predicted work beyond its spread -- with that saving's share, and never a unit
/// of `barred` (a keep of this kind once made is not priced again: undone or not, its price has been read).
#[allow(clippy::too_many_arguments)]
fn keep(
    request: &Request,
    light: &Light,
    kept: &[Unit],
    model: &Model,
    reference: [f64; GROUND],
    centre: V3,
    units: &[Unit],
    measured: &Measured,
    means: &[Vec<[f64; 2]>],
    parts: &[&Part],
    estimates: &[Estimate],
    fixed: &Fixed,
    scales: &Scales,
    unsampled: [f64; 2],
    again: f64,
    barred: &[Unit],
) -> Option<(usize, Option<f64>)> {
    let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
    // A unit weighs where it was read: an output with no state read carries every unit's bound alike, and tells none
    // of them apart.
    let mut read = vec![false; fixed.model.len()];
    measured.output.iter().for_each(|&k| read[k] = true);
    let crowded: Vec<usize> = crowded(&looks(estimates, fixed, &vec![Bound::share(); fixed.model.len()])).into_iter().filter(|&k| held[k] > 0. && read[k]).collect();
    let weights: Vec<f64> = (0..units.len()).map(|u| crowded.iter().map(|&k| held[k] * (means[k][u][0] / POPULATION_TARGET + means[k][u][1] / TEMPERATURE_TARGET)).sum()).collect();
    if let Some(u) = (0..units.len()).max_by(|&a, &b| weights[a].total_cmp(&weights[b])).filter(|&u| weights[u] > 0.) {
        return Some((u, None));
    }
    let mut pricing = Pricing::new(request, light, kept, model, reference, centre, units, measured, means, parts, estimates, fixed, scales, unsampled, again);
    let saving = pricing.saving_at_all();
    let mut ranked: Vec<Candidate> = if saving.is_empty() {
        vec![]
    } else {
        pricing.own = pricing.rates(model);
        let pricing = &pricing;
        let others = pricing.leave_one_out();
        saving.into_par_iter().filter(|&u| !barred.contains(&units[u]) && pays(0., pricing.saving(&others, u, &pricing.own))).filter_map(|u| pricing.priced(&others, u)).collect()
    };
    ranked.sort_by(|a, b| assured(b.saving).total_cmp(&assured(a.saving)));
    ranked.first().filter(|c| pays(0., c.saving)).map(|c| (c.unit, Some(c.saving.value)))
}

/// One attempt at the solve with the units `kept` and the reach at `reach` µK (module documentation, # The pipeline). The
/// units its model's closure keeps join `kept`, the solve's one set: a unit an attempt keeps after its pilot is a unit of
/// the closed model (a drive of a carrier the closure made near), and the next attempt's model is built from both.
/// `step`: the step the attempt before stood at, this one's first rung. `priced`: the model's last unit was kept for a
/// predicted saving -- the step of the model before it and the saving's share of the work. The price knew the kept
/// model's work a step, not the step it needs (pricing reads no step); where the check then takes a step at which the
/// kept model costs more than the one before it, (1 − share)·τ_before/τ > 1, the keep is undone (`Attempt::Undo`): the
/// fold's error stands, measured and booked as before.
#[allow(clippy::too_many_arguments)]
fn attempt(
    request: &Request,
    light: &Light,
    kept: &mut Vec<Unit>,
    reach: f64,
    step: Option<f64>,
    priced: Option<(f64, f64)>,
    barred: &[Unit],
    messages: &mut Messages,
) -> Result<Attempt, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request);
    // The reference state (SPEC §2.1): the initial component of the largest population, of equal ones the last.
    let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).expect("a validated request's initial state holds a ground state");
    let reference = alone(initial[chief].0);
    // The model: every unit whose fold does not exist over the reach kept (SPEC §1.5), closed again until none fails (a
    // model lists only what it folds, model.rs `units`). A model's units ask nothing of its trap and reach: the two are
    // made side by side on the pool.
    let (model, trap, judged, units, gates) = loop {
        let model = Model::new(request, light, kept)?;
        let reached = || model.trap(&reference, &model.prepared()).map(|trap| (model.reach(&trap, &reference, reach, unseen(MOST_TRAJECTORIES as f64)), trap));
        let (reached, units) = rayon::join(reached, || model.units());
        let (judged, trap) = reached?;
        let gates = model.gates(&units, &judged);
        let failing: Vec<Unit> = units.iter().zip(&gates).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
        if failing.is_empty() {
            break (model, trap, judged, units, gates);
        }
        kept.extend(failing);
    };
    // The folded units: which are read at the states, which booked at their gates' bounds (`booked`).
    let booking = booked(&units.iter().zip(&gates).map(|(u, g)| g.error_bound(u)).collect::<Vec<_>>());
    let reading: Vec<Unit> = units.iter().zip(&booking).filter(|(_, b)| matches!(b, Booked::Read(_))).map(|(u, _)| u.clone()).collect();
    // The trap every reported quantity describes (SPEC §1.3: trapFrequencyKHz, n̄, the validated regime and T's
    // windows): the trap light's potential with the other lights' phase-averaged shifts at the reference state
    // (`Model::reported`). Where it has no minimum the descent reaches -- the atoms are not held by the light they run in: a
    // mathematical object that does not exist, not a kind of scene -- the prepared trap's is reported, and flagged.
    //
    // What the reach the model was judged over leaves out of the initial ensemble (model/reach.rs `Model::left_out`): of
    // the atoms and of their kinetic energy, by the ensemble's quadrature of its own density -- no atom is drawn --, a
    // level of it with no more evaluations of the potential than the most trajectories a solve runs have initial atoms.
    //
    // The reported trap, what the reach leaves out and the entries the first rung reads (`admissible`) ask nothing of
    // one another: the three are made side by side on the pool.
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, light, alone(s))).collect::<Result<_, _>>()?;
    let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm)?;
    let (reported, (unsampled, entries)) =
        rayon::join(|| model.trap(&reference, &model.reported()), || rayon::join(|| model.left_out(&judged, &ensemble, MOST_TRAJECTORIES), || model.sampled(&model.running())));
    let (report, unheld) = match reported {
        Ok(t) => (t, false),
        Err(_) => (trap.clone(), true),
    };
    let scales = Scales { hbar_over_mass: hm, frequencies: report.lab };
    let plan = Plan::new(&request.times_us, report.lab);
    let outputs = plan.times.len();
    let row: [f64; STATES] = std::array::from_fn(|s| request.initial_density[(s, s)].re);
    let depth = ensemble.depths()[chief];
    let mut ledger = Ledger::default();
    ledger.note = Some(format!("初始系综：各初态在自己的准备阱中的束缚原子（降低的等温分布乘每轴谐振子的 Wigner 因子，T = {} μK），参考态阱深 {:.1} k_BT", request.temperature_uk, depth));
    if unheld {
        ledger.flag("运行中的势没有极小：阱频与 n̄ 按准备阱报告".into());
    }
    let exact = |kinetic: &[Quadrature]| -> (V3, f64) {
        (ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>()), kinetic.iter().map(|k| k.difference / k.eta).fold(0., f64::max))
    };
    // T(0) and n̄(0) by quadrature (ensemble.rs `Ensemble::kinetic`), twice. Here with no budget: the two levels it takes
    // at least -- an error needs two to differ --, for the results the pilot's stages send. After the pilot with as many
    // evaluations of the potential as the pilot took steps: a step evaluates the potential's expansion at least once (its
    // reference flow's, SPEC §2.1), so the quadrature costs no more than the pilot the solve has just run, in any trap;
    // where that budget ends it before its levels agree, their difference is T(0)'s booked error (`Fixed::start`).
    let (momentum, start) = exact(&ensemble.kinetic(0));
    // The ladder's first rung over the reach the model is judged over: an attempt that begins at the step the one
    // before stood at begins no higher than its own first rung (a reach that has grown lowers it).
    let first_rung = first_rung(plan.end, trap.frequencies.iter().copied().fold(0., f64::max), &entries, judged.speed, typical_speed(hm, judged.kinetic_uk.iter().sum::<f64>() / 3.))
        .ok_or_else(|| format!("时间步长：每条轨迹至多 {MOST_STEPS} 步之内没有可用的步长（阱的共振或光场沿原子路径的变化要求更小的步长；或者阱频、原子速率不是有限数）"))?;
    // What the reach's falsifier compares the T read with: the largest kinetic temperature along an axis of the
    // ensemble the reach was made for (model.rs `Reach::kinetic_uk`: T_r where the trap is classical, the zero point's
    // where it is cold) -- no lab axis's T is above it while the atoms are that ensemble.
    let held = judged.kinetic_uk.iter().copied().fold(reach, f64::max);
    let mut context = Context {
        momentum,
        fixed: Fixed { model: vec![[0.; 2]; outputs], start },
        step: vec![Bound::share(); outputs],
        checked: None,
        tau: step.map_or(first_rung, |step| step.min(first_rung)),
        plan: Plan { cell: 2. * step.map_or(first_rung, |step| step.min(first_rung)), ..plan },
        scales,
        initial: row,
        ledger,
        method: format!("半经典 3D：{}", model.describe().trim_end_matches('。')),
    };

    // The near light's tables (SPEC §2.3, §5.4; model/tabulate.rs): refined over the reach to the largest ensemble's
    // resolution, no grid of more nodes than the middles the pilot takes at the attempt's first step and at its double,
    // pilot_size·(end/τ + end/(2τ)) -- a node counted as a middle without a table. The bound is on each grid by
    // itself, and against the pilot's middles, not the run's: it keeps a grid's nodes below the pilot's middles
    // before a trajectory has run; it is not the work the table saves. What a grid held back misses is read at the
    // atoms below.
    let most = (pilot_size() as f64 * (context.plan.end / context.tau + context.plan.end / (2. * context.tau))).ceil() as usize;
    let mut tables = model.tabulate(&judged, most);
    let run = Run::tabulated(&model, &tables, reference);

    // The pilot's groups (module documentation, # The pipeline): the run's first trajectories, at two levels.
    let first = MOST_TRAJECTORIES / GROUP;
    let numbers: Vec<u64> = (first..first + pilot_size() / GROUP).collect();
    // A priced keep's model at the step `step` against the model before it at its own (see `priced`).
    let dearer = |step: f64| priced.is_some_and(|(before, share)| (1. - share) * before / step > 1.);
    let mut levels = Levels::default();

    // The scout (module documentation, # The pipeline): the first member of every pilot group at both levels, screened;
    // where its atoms outgrow the reach or a unit is to be kept at its states, the attempt ends here -- for the price of
    // the scout, not of a pilot. The pilot's read of the same things below stands for the result.
    if matches!(settle(&run, &ensemble, &mut context, &mut levels, &numbers, 0..1, &dearer, messages), Settled::Dearer) {
        return Ok(Attempt::Undo);
    }
    // What ends the attempt at some groups' states (`parts`; a model's pilot is `again` trajectories): its atoms
    // outgrown the reach, or a unit to keep (`keep`) at the model's errors there -- which are the errors the results
    // book from then on, and the pilot's those its check weighs the step beside (`check`). Else the states' read, for
    // the next one to take over (`sampled`).
    let decide = |context: &mut Context, parts: &[&Part], known: Option<Sampled>, again: f64| -> Result<Sampled, Attempt> {
        let estimates = pooled(parts, &context.scales).0;
        if let Some(t) = outgrown(&estimates, held) {
            return Err(Attempt::Grow(t, context.tau));
        }
        let read = sampled(&model, &reading, &tables, parts, known);
        let measured = Measured { errors: &read.errors, output: &read.at, group: &read.from, booked: &booking };
        let means = measured.means(outputs, None);
        context.fixed.model = model_error(&sums(&means, unsampled), unsampled);
        match keep(request, light, kept, &model, reference, trap.centre, &units, &measured, &means, parts, &estimates, &context.fixed, &context.scales, unsampled, again, barred) {
            Some((u, share)) => Err(Attempt::Keep(units[u].clone(), context.tau, share)),
            None => Ok(read),
        }
    };
    let scouted = match decide(&mut context, &levels.fine.iter().collect::<Vec<_>>(), None, numbers.len() as f64) {
        Ok(read) => read,
        Err(end) => return Ok(end),
    };
    // The pilot: its groups' other members at both levels, and the step's check on them. On every rung the check
    // does not leave at once it asks for the model's errors at the pilot's states there, which its works are
    // beside; the model's decisions at those states are made with them, before the check stands or runs a pair: an
    // attempt they end has paid for neither pairs nor product. The reads the check's last rung is held by, flagged
    // where it ended with nothing left to choose.
    let (mut settled, mut scouted) = (settle(&run, &ensemble, &mut context, &mut levels, &numbers, 1..GROUP, &dearer, messages), scouted);
    while matches!(settled, Settled::Asks) {
        scouted = match decide(&mut context, &levels.fine.iter().collect::<Vec<_>>(), Some(scouted), pilot_size() as f64) {
            Ok(read) => read,
            Err(end) => return Ok(end),
        };
        settled = check(&run, &ensemble, &mut context, &mut levels, &numbers, GROUP, true, &dearer, messages);
    }
    let held_by = match settled {
        Settled::Dearer => return Ok(Attempt::Undo),
        Settled::Unmet(held) => {
            context.ledger.flag(unmet());
            held
        }
        Settled::Standing(held) => held,
        Settled::Asks => unreachable!("the check has been given what it asked for"),
    };
    let tau = context.tau;
    let run = {
        let pilot: Vec<&Part> = levels.fine.iter().collect();
        let (estimates, _, counts) = pooled(&pilot, &context.scales);
        if let Some(t) = outgrown(&estimates, held) {
            return Ok(Attempt::Grow(t, tau));
        }
        let steps = counts.steps + levels.coarse.iter().map(|p| p.counts.steps).sum::<u64>();
        (context.momentum, context.fixed.start) = exact(&ensemble.kinetic(steps));

        // The model's errors at the pilot's states (SPEC §5.2): each folded unit's per output, the held atoms' mean over
        // the first member of every group -- one a group keeps every group in the mean's spread, and a unit's error is a
        // function of the state, read as well on eleven states an output as on eighty-eight.
        let Sampled { at, from, errors, .. } = sampled(&model, &reading, &tables, &pilot, Some(scouted));
        let measured = Measured { errors: &errors, output: &at, group: &from, booked: &booking };
        let means = measured.means(outputs, None);
        context.fixed.model = model_error(&sums(&means, unsampled), unsampled);
        // The tables' errors (SPEC §5.4): each table's whole error against the elimination itself, its mean over the
        // run's held states read at one state of every pilot trajectory (`one_each`, `over_the_run`) -- a state on another
        // cluster reads none of it. A table
        // beyond the tables' shares has the axis of its largest read halved and is read again, until it is within them or
        // its next grid would exceed the work's bound; what it reads then is booked on every output, and flagged where it
        // still lies beyond.
        let read_at = one_each(&pilot);
        let (shown_share, temperature_share) = crate::model::tabulate::shares();
        drop(run);
        let mut stuck = vec![false; tables.tables.len()];
        let booked = loop {
            let reads: Vec<Vec<crate::model::tabulate::TableRead>> = read_at.par_iter().map(|pick| model.table_reads(&tables, std::slice::from_ref(&pick.state.sample))).collect();
            let mean = |f: &dyn Fn(&crate::model::tabulate::TableRead) -> f64, t: usize| over_the_run(&read_at, reads.iter().map(|r| f(&r[t])), None);
            let whole: Vec<[f64; 2]> = (0..tables.tables.len()).map(|t| [mean(&|r| r.whole[0].population, t), mean(&|r| r.whole[0].temperature, t)]).collect();
            let beyond = (0..whole.len()).find(|&t| !stuck[t] && (whole[t][0] > shown_share || whole[t][1] > temperature_share));
            let Some(t) = beyond else { break whole };
            let axes = tables.tables[t].axes();
            let weight = |f: usize| mean(&|r| r.axes[f][0].population / shown_share + r.axes[f][0].temperature / temperature_share, t);
            let worst = (0..axes).max_by(|&a, &b| weight(a).total_cmp(&weight(b)));
            match worst.and_then(|f| model.halved(&tables.tables[t], f, most)) {
                Some(finer) => tables.tables[t].table = finer,
                None => stuck[t] = true,
            }
        };
        for (t, table) in tables.tables.iter().enumerate() {
            let label = format!(
                "近共振光的表（{} 所在子空间{}）",
                model.state_label(model.clusters()[table.layout.cluster].states[0]),
                if table.layout.lights == model.running() { "" } else { "，光镊关闭时" }
            );
            for k in 1..outputs {
                context.fixed.model[k] = [(context.fixed.model[k][0] + booked[t][0]).min(1.), context.fixed.model[k][1] + booked[t][1]];
            }
            context.ledger.item(&label, booked[t][0].max(booked[t][1]), TABLE_MEASURED);
            if stuck[t] {
                context.ledger.flag(format!("{label}的误差超出表的份额，而加密已到工作量的界：按实测记账"));
            }
        }
        // The step's check (module documentation, # The step): what it does not exclude into the accounts, its
        // resolution into the result's words.
        let (fine, coarse) = levels.paired();
        let resolution = resolution(&between(&fine, &coarse, &context.scales));
        (context.step, context.checked) = (vec![unexcluded(&held_by); outputs], Some(Checked { companion: 2. * tau, pairs: fine.len() as u64 * GROUP, resolution }));
        // A unit to keep at the pilot's states (`keep`): its model then runs a pilot again.
        if let Some((u, share)) = keep(request, light, kept, &model, reference, trap.centre, &units, &measured, &means, &pilot, &estimates, &context.fixed, &context.scales, unsampled, pilot_size() as f64, barred) {
            return Ok(Attempt::Keep(units[u].clone(), tau, share));
        }
        // The ledger's items: each folded unit's error on what is shown and on T (its largest over the outputs, weighed by
        // the pilot's held share as the totals are; an output where the pilot holds no atom weighs nothing: its units'
        // errors are their bounds, `Measured::means`) with the item's place for its unit and how it was booked, and the
        // unsampled share.
        let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
        let mut largest = vec![0.0_f64; units.len()];
        for k in (1..outputs).filter(|&k| held[k] > 0.) {
            for (most, m) in largest.iter_mut().zip(&means[k]) {
                *most = most.max(m[0].max(m[1]) * held[k]);
            }
        }
        for ((unit, most), how) in units.iter().zip(largest).zip(&booking) {
            context.ledger.item(&model.label(unit), most, if matches!(how, Booked::Read(_)) { FOLD_MEASURED } else { FOLD_BOUNDED });
        }
        context.ledger.item("可达范围之外的初始原子份额", unsampled[0], "按初始系综的密度求积（含求积两级之差），计入布居的模型误差");
        context.ledger.item("可达范围之外的初始原子的动能份额", unsampled[1], "按初始系综的密度求积（含求积两级之差），计入 T 的模型误差");
        Run::tabulated(&model, &tables, reference)
    };

    // The product (module documentation, # Stopping): on from the pilot's trajectories at the run's step, the run's
    // first; the pilot's companions have done their part, and its states have been read.
    let mut parts = levels.fine;
    forget(&mut parts);
    parts.append(&mut levels.further[0]);
    let (e, l, c) = product(&run, &ensemble, &context, &mut parts, &numbers, messages);
    // A point of the prepared potential below the bottom the ensemble was drawn about (ensemble.rs, # Its draw).
    if ensemble.deeper() > 0 {
        context.ledger.flag("准备势在所取阱底之外还有更深的地方：初始系综按该阱底的深度封顶".into());
    }
    Ok(Attempt::Done(assemble(&context, &e, &l, &c, parts.len() as u64 * GROUP)))
}

/// The flag of a step whose check ended with nothing left to choose (module documentation, # The step; SPEC §5.1):
/// what the check does not exclude leaves some quantity's statistics no room, and no number of pairs within a run's
/// trajectories and no rung of the ladder is predicted to leave it one -- or the error does not fall at the scheme's
/// order on the ladder's last rung. What the check does not exclude is booked.
fn unmet() -> String {
    format!("时间步长的收敛检查没有排除的误差占满了目标（或误差没有按格式的阶下降），而加轨迹对（上限 {MOST_TRAJECTORIES} 条）与减半步长（每条轨迹上限 {MOST_STEPS} 步）都已无余地：按检查没有排除的上限记账")
}

/// The pilot's two levels (module documentation, # The step): its groups at the step τ and the same groups, in the same
/// order, at 2τ; the further pairs a check took (`settle`), at the two levels: the groups the product runs next, by
/// their place in its list; and the read the rung in hand inherits from the rung above (`lowered`), where a check of
/// this attempt came down from one -- the scout's serves the pilot on the rung the scout left standing: the same
/// model at the same steps.
#[derive(Default)]
struct Levels {
    fine: Vec<Part>,
    coarse: Vec<Part>,
    further: [Vec<Part>; 2],
    above: Option<Between>,
}
impl Levels {
    /// The finer level's groups and their companions: the pilot's, then the further pairs'.
    fn paired(&self) -> (Vec<&Part>, Vec<&Part>) {
        (self.fine.iter().chain(&self.further[0]).collect(), self.coarse.iter().chain(&self.further[1]).collect())
    }
}

/// The members `members` of the groups `numbers` at the step `tau`, each trajectory through the plan: a hand-out of
/// its own (`run_groups`).
#[allow(clippy::too_many_arguments)]
fn level(run: &Run, ensemble: &Ensemble, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Part> {
    let hand = Hand { groups: numbers, members, shape: Shape::of(plan, keep), record: &|i| trajectory(run, ensemble, tau, plan, keep, i) };
    run_groups(&[hand], messages, progress).pop().expect("the hand-out's one level")
}

/// The members `members` of the groups `numbers` at the step `tau` (their states kept where `keep`) and at twice it,
/// the two levels of a check (module documentation, # The step): one hand-out (`run_groups`) -- a check runs both
/// whatever either shows --, the finer level first, whose trajectories take twice the steps. `progress` takes the finer
/// level's groups.
#[allow(clippy::too_many_arguments)]
fn pair(run: &Run, ensemble: &Ensemble, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> [Vec<Part>; 2] {
    let fine = Hand { groups: numbers, members: members.clone(), shape: Shape::of(plan, keep), record: &|i| trajectory(run, ensemble, tau, plan, keep, i) };
    let coarse = Hand { groups: numbers, members, shape: Shape::of(plan, false), record: &|i| trajectory(run, ensemble, 2. * tau, plan, false, i) };
    let mut levels = run_groups(&[fine, coarse], messages, progress).into_iter();
    std::array::from_fn(|_| levels.next().expect("the hand-out's two levels"))
}

/// The states kept in some groups' parts freed, on the pool, a group's by itself: a state is a block of memory of its
/// own, and a pilot keeps one for every trajectory at every output.
fn forget(parts: &mut [Part]) {
    parts.par_iter_mut().for_each(|part| part.samples = vec![]);
}

/// Some groups' further members joined to those a level holds of them (all of them where it holds none).
fn absorb(level: &mut Vec<Part>, more: Vec<Part>) {
    if level.is_empty() {
        *level = more;
    } else {
        level.iter_mut().zip(more).for_each(|(p, m)| p.absorb(m));
    }
}

/// How a check leaves the step (`check`): standing, or with nothing left to choose -- each with the reads the rung is
/// held by (`held`) --; asking for the model's errors at the pilot's states on the rung in hand; or at a step that
/// makes a priced keep dearer than the model before it (`attempt`).
#[derive(Clone, Debug)]
enum Settled {
    Standing(Between),
    Unmet(Between),
    Asks,
    Dearer,
}

/// The members `members` of the pilot's groups `numbers` at both levels (`pair`), joined to those the levels hold, and
/// the step's check on them (`check`), the model's errors at their states not read yet.
#[allow(clippy::too_many_arguments)]
fn settle(run: &Run, ensemble: &Ensemble, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: std::ops::Range<u64>, dearer: &dyn Fn(f64) -> bool, messages: &mut Messages) -> Settled {
    let [fine, coarse] = pair(run, ensemble, &context.plan, context.tau, numbers, members.clone(), true, messages, &mut |prefix, messages| so_far(context, &levels.fine, prefix, messages));
    absorb(&mut levels.fine, fine);
    absorb(&mut levels.coarse, coarse);
    check(run, ensemble, context, levels, numbers, members.end, false, dearer, messages)
}

/// The step's check (module documentation, # The step) on the pilot's groups `numbers`, the first `members` of each
/// in at both levels, read again at every rung and after every batch of pairs. The looks are the run's as the finer
/// level predicts them (`needed`); the choice is the one of least predicted work (`choose`):
/// - on the pilot (all the members of a group in: the last word on the step), among this rung on the pairs read, on
///   more pairs -- the product's next groups at both levels --, and the rungs below: standing and the rungs beside
///   what the reads the rung is held by (`held`) do not exclude (`unexcluded`), more pairs beside what they could
///   leave of it (`narrowed`). The works are beside the model's error at the pilot's states on the rung in hand
///   (`current`: `Context::fixed` holds it). Where it is not read there, the looks are taken beside no model error,
///   and only the rung below is chosen on them -- a model's error takes of every look's room, and the larger step's
///   part feels it the more: a rung left beside none is left beside any --; any other choice ends the check asking
///   for the model's errors (`Settled::Asks`), with which the model's own decisions at those states come first
///   (`attempt`);
/// - on the scout, which books nothing and has the pilot to come, between this rung and the rungs below alone, beside
///   what its read has seen (`clear`) and no model error (its states are read after it): the rung is left before its
///   pilot only where the read shows the rung below to be the lesser work, with the least step error it leaves
///   possible;
/// - a rung whose own read refutes the order it inherits by (`refuted`) is left: neither read is of its error;
/// - a rung that inherits no read -- an attempt's first: two levels, the order untested -- stands or takes pairs
///   only within the step's share (`Bound::share`; module documentation, # The step): the first rung's rule.
/// The rung below: the level at τ becomes the companion, and the members so far run anew at τ/2 alone; the finer level
/// keeps its states (the model is read at them). A rung at which a trajectory would take more than `MOST_STEPS` steps
/// does not exist. Where nothing is left to choose -- no finite work, or the order refuted on the ladder's last rung
/// -- the check ends there (`Settled::Unmet`): the step stands, flagged. Nor is a rung taken at which a priced keep's
/// model costs more than the model before it (`dearer`): the attempt ends, and the keep is undone.
#[allow(clippy::too_many_arguments)]
fn check(run: &Run, ensemble: &Ensemble, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: u64, mut current: bool, dearer: &dyn Fn(f64) -> bool, messages: &mut Messages) -> Settled {
    // The trajectories a rung's check begins with.
    let begins = numbers.len() as u64 * members;
    loop {
        let (fine, coarse) = levels.paired();
        let read = between(&fine, &coarse, &context.scales);
        let failed = levels.above.as_ref().is_some_and(|inherited| refuted(&read, inherited));
        let inherited = levels.above.as_ref().filter(|_| !failed);
        let held = held(&read, inherited);
        // The rung below, and the rungs the ladder has below this one.
        let below = 0.5 * context.tau;
        let rungs = (1..).take_while(|&j| context.plan.end / (context.tau * 0.5f64.powi(j)) <= MOST_STEPS as f64).count() as u32;
        if !failed {
            // The run's looks as the pairs' finer level predicts them, beside the model's error alone -- beside none
            // where it is not read at this rung's states.
            let (have, estimates) = (fine.len() as u64 * members, pooled(&fine, &context.scales).0);
            let none = Bound { shown: 0., temperature: 0. };
            let unread = Fixed { model: vec![[0.; 2]; estimates.len()], start: context.fixed.start };
            let looks: Vec<Look> = looks(&estimates, if current { &context.fixed } else { &unread }, &vec![none; estimates.len()]).into_iter().filter(Look::fits).collect();
            // A rung that inherits no read has its order untested: it books no more than the step's share.
            let share = Bound::share();
            let limit = inherited.is_none().then_some(&share);
            let choice = if members == GROUP {
                choose(&looks, &unexcluded(&held), &|pairs| unexcluded(&self::held(&narrowed(&read, have as f64 / pairs as f64), inherited)), limit, have, MOST_TRAJECTORIES, begins, rungs)
            } else {
                choose(&looks, &clear(&read), &|_| clear(&read), limit, have, have, begins, rungs)
            };
            match choice {
                Choice::Rung => {}
                _ if members == GROUP && !current => return Settled::Asks,
                Choice::Stand => return Settled::Standing(held),
                Choice::Neither => return Settled::Unmet(held),
                Choice::Pairs(pairs) => {
                    let next: Vec<u64> = (have / GROUP..pairs / GROUP).collect();
                    let more = pair(run, ensemble, &context.plan, context.tau, &next, 0..GROUP, false, messages, &mut |prefix, messages| {
                        so_far(context, &levels.fine, &levels.further[0].iter().chain(prefix.iter().copied()).collect::<Vec<_>>(), messages)
                    });
                    levels.further.iter_mut().zip(more).for_each(|(further, more)| further.extend(more));
                    continue;
                }
            }
        }
        if rungs == 0 {
            return Settled::Unmet(held);
        } else if dearer(below) {
            return Settled::Dearer;
        }
        levels.above = Some(lowered(&held));
        (context.tau, current) = (below, false);
        // The level at the step is the companion from here on: its states are read no more.
        forget(&mut levels.fine);
        let finer = level(run, ensemble, &context.plan, below, numbers, 0..members, true, messages, &mut |prefix, messages| so_far(context, &[], prefix, messages));
        levels.coarse = std::mem::replace(&mut levels.fine, finer);
        levels.further = Default::default();
    }
}

/// The trajectories a batch brings the product to, for the `need` that the estimates of `have` trajectories predict:
/// whole groups, more than `have`, at most twice as many, and `MOST_TRAJECTORIES` at most.
///
/// At most twice: the need is predicted from variances read on those trajectories' groups (the pilot's eleven at first,
/// whose largest read over the outputs runs high), and a batch takes the product no further than to twice what its
/// prediction rests on. What that bounds: the batches accumulate -- every trajectory of the product is in the result --,
/// so the product ends on the first total that meets the targets. Each total is at most g times one that did not meet
/// them, so the last total is at most g times the least that meets them, whatever was predicted: at g = 2 no more than
/// half of the work is spent past the need, and a need N times the pilot's size is reached in log₂ N batches. The
/// bound holds at any g > 1 and does not single out two -- a smaller factor spends less past the need and reads more
/// often --: the factor is chosen, not derived.
fn batch(need: f64, have: u64) -> u64 {
    let whole = ((need / GROUP as f64).ceil() as u64).saturating_mul(GROUP);
    whole.clamp(have + GROUP, 2 * have).min(MOST_TRAJECTORIES)
}

/// The product (module documentation, # Stopping): batches of whole groups at the run's step, on from the pilot's
/// groups (their numbers `pilot`; the product's own follow in the list's order), to the number the looks predict
/// (`batch`), until every look that some number of trajectories can meet is met, or the trajectories reach
/// `MOST_TRAJECTORIES`. Each batch's result is a checkpoint; returns the estimates, the listed atoms and the counts at
/// its end.
fn product(run: &Run, ensemble: &Ensemble, context: &Context, parts: &mut Vec<Part>, pilot: &[u64], messages: &mut Messages) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts) {
    let (started, from) = (Instant::now(), parts.len() as u64 * GROUP);
    // The group at the list's place g.
    let number = |g: u64| pilot.get(g as usize).copied().unwrap_or(g);
    // The wall time (ms) the trajectories from `done` to `want` take at the product's pace so far; none before its own
    // first trajectories are in.
    let left = |done: u64, want: u64| (done > from).then(|| started.elapsed().as_secs_f64() * 1e3 / (done - from) as f64 * (want - done) as f64);
    loop {
        let all: Vec<&Part> = parts.iter().collect();
        let n = all.len() as u64 * GROUP;
        let (e, l, c) = pooled(&all, &context.scales);
        let chased: Vec<Look> = looks(&e, &context.fixed, &context.step).into_iter().filter(Look::fits).collect();
        if chased.iter().all(Look::met) || n >= MOST_TRAJECTORIES {
            return (e, l, c);
        }
        let want = batch(chased.iter().map(Look::need).fold(0., f64::max), n);
        messages.result(all.len(), n, left(n, want), || assemble(context, &e, &l, &c, n));
        let numbers: Vec<u64> = (n / GROUP..want / GROUP).map(number).collect();
        let new = level(run, ensemble, &context.plan, context.tau, &numbers, 0..GROUP, false, messages, &mut |prefix, messages| {
            let all: Vec<&Part> = parts.iter().chain(prefix.iter().copied()).collect();
            let done = all.len() as u64 * GROUP;
            let (e, l, c) = pooled(&all, &context.scales);
            messages.result(all.len(), done, left(done, want), || assemble(context, &e, &l, &c, done));
        });
        parts.extend(new);
    }
}

/// The pilot's size: the fewest whole groups whose unseen share fits the statistics' room in the population target beside
/// the step's share alone, z²/(N + z²) ≤ √(1 − s²) ε (`room`; estimate.rs `unseen` and its inverse `draws_for`, units.rs
/// `STEP_SHARE`). The pilot is the run's first trajectories, and no run ends on fewer: what is shown has statistics no
/// less than the unseen share (estimate.rs `Estimate::shown_statistics`), so below this number its look is not met
/// beside the step's share whatever the trajectories show and however exact the model. N ≥ z²(1/(√(1 − s²) ε) − 1) =
/// 85.4 at z = 2, s² = 1/(1 + 2p) = 1/5, ε = 0.05: eleven groups of `GROUP` = 8, 88 trajectories.
fn pilot_size() -> u64 {
    (draws_for(room(POPULATION_TARGET, 0., step_bound()[0])) / GROUP as f64).ceil() as u64 * GROUP
}

/// The outputs and T's windows (module documentation, # Outputs and windows): per output and lab axis the window's half
/// width, the outputs the distributions are taken at, and the run's end, the last window's.
struct Plan {
    times: Vec<f64>,
    halves: Vec<V3>,
    listed: Vec<usize>,
    end: f64,
    /// The cells of a trajectory's Brownian path (adapt.rs `Path`), µs: the companion's step at the attempt's first
    /// step (its first rung, or the shorter step the attempt before stood at), a whole part of the run. Every step of
    /// the attempt -- a rung's, its companion's -- is a cell over a power of two, so every level reads the same path.
    /// The run until the first rung is known.
    cell: f64,
}
impl Plan {
    /// The plan of the request's times with the trap's frequency per lab axis (rad/µs).
    fn new(times: &[f64], frequencies: V3) -> Plan {
        let (interval, last) = (times[1] - times[0], times.len() - 1);
        let halves: Vec<V3> = times.iter().map(|&t| frequencies.map(|w| half_window(w, interval, t))).collect();
        let count = times.len().min(DISTRIBUTION_TIMES);
        let mut listed: Vec<usize> = (0..count).map(|i| ((i * last) as f64 / (count - 1) as f64).round() as usize).collect();
        listed.dedup();
        let end = times[last] + halves[last].iter().fold(0., |a: f64, &b| a.max(b));
        Plan { end, times: times.to_vec(), halves, listed, cell: end }
    }
}

/// Counts over trajectories: the steps taken, and the trajectories the scheme could not advance (adapt.rs `advance`'s
/// refusal) with the earliest time one stopped at.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    steps: u64,
    failed: u64,
    since: Option<f64>,
}
impl Counts {
    fn add(&mut self, o: &Counts) {
        self.steps += o.steps;
        self.failed += o.failed;
        self.since = match (self.since, o.since) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
}

/// A pilot trajectory's state at an output (model.rs `Sample`) with the run's step and the trajectory's number: where
/// the units' errors are measured and a model is priced (`rate`).
#[derive(Clone)]
struct State {
    sample: Sample,
    tau: f64,
    index: u64,
}

/// What one trajectory leaves: what became of it by every output (estimate.rs `Fate`), its (r, κ) at the listed outputs
/// where held, its states at the outputs where held (`keep`: the pilot's, for the model's errors and prices), its counts.
struct Record {
    outputs: Vec<Fate>,
    listed: Vec<Option<(V3, V3)>>,
    samples: Vec<Option<State>>,
    counts: Counts,
}

/// One trajectory from its initial state through the plan at the step τ (module documentation, # Outputs and windows):
/// advanced to the run's end by steps that know no output, and read at every output where it falls (adapt.rs, # The
/// reads). A trajectory the scheme cannot advance (adapt.rs) fails from the first output it does not reach.
fn run<D: Dynamics>(model: &D, start: Start, index: u64, tau: f64, plan: &Plan, keep: bool) -> Record {
    let outputs = plan.times.len();
    let interval = plan.times[1] - plan.times[0];
    let widest = plan.halves.iter().flatten().fold(0., |a: f64, &b| a.max(b));
    let mut t = Trajectory::new(model, start.state, start.r, start.kappa, SEED, index);
    let mut scratch = Scratch::default();
    // The trajectory's Brownian path in the plan's cells, its grid the multiples of τ (a cell over a power of two):
    // the same path at every rung and at its companion.
    let mut path = adapt::Path::new(SEED, index, plan.cell, (plan.cell / tau).log2().round() as u32);
    // Per output and lab axis: the window's time held, ∫κ dt and ∫κ² dt.
    let mut moments = vec![[[0.; 3]; 3]; outputs];
    let mut out = vec![Fate::Lost; outputs];
    let mut listed = vec![None; plan.listed.len()];
    let mut samples: Vec<Option<State>> = vec![None; if keep { outputs } else { 0 }];
    let mut steps = 0;
    let mut record = |k: usize, t: &Trajectory, out: &mut Vec<Fate>| {
        // The atom at the output's instant; from the first output on its κ and κ² are its window's means (below).
        out[k] = Fate::Held(Held { rows: model.readout(t.cluster, &t.point(), &t.psi), kappa: t.kappa, kappa2: t.kappa.map(|x| x * x) });
        if let Ok(j) = plan.listed.binary_search(&k) {
            listed[j] = Some((t.r, t.kappa));
        }
        if keep {
            let sample = Sample { cluster: t.cluster, at: t.point(), psi: t.psi.to_vec() };
            samples[k] = Some(State { sample, tau, index });
        }
    };
    record(0, &t, &mut out);
    // The first output the trajectory does not reach held, and whether the scheme could not advance it.
    let mut reached = 1;
    let failed = {
        let mut read = |k: usize, t: &Trajectory| {
            record(k + 1, t, &mut out);
            reached = k + 2;
        };
        // A step's path as far as the trajectory is on it, into the windows it cuts.
        let mut each = |s: &Segment, until: f64| {
            steps += 1;
            let (s0, s1) = (s.start, until);
            let first = (((s0 - widest) / interval).floor().max(1.)) as usize;
            let last = (((s1 + widest) / interval).ceil() as usize).min(outputs - 1);
            for k in first..=last {
                // The step's moments over an axis's cut of it (`Segment::moments` gives every axis's): made once for
                // the axes whose windows cut the step alike.
                let mut made: [Option<((f64, f64), (V3, V3))>; 3] = [None; 3];
                for a in 0..3 {
                    let h = plan.halves[k][a];
                    let (lo, hi) = (s0.max(plan.times[k] - h), s1.min(plan.times[k] + h));
                    if hi > lo {
                        let (m1, m2) = match made[..a].iter().flatten().find(|(cut, _)| *cut == (lo, hi)) {
                            Some((_, same)) => *same,
                            None => s.moments(lo, hi),
                        };
                        made[a] = Some(((lo, hi), (m1, m2)));
                        let w = &mut moments[k][a];
                        w[0] += hi - lo;
                        w[1] += m1[a];
                        w[2] += m2[a];
                    }
                }
            }
        };
        // Through the last output's window; a failure there leaves every output read.
        adapt::advance(model, tau, &mut t, &mut scratch, &mut path, plan.end, &plan.times[1..], &mut read, &mut each).is_err()
    };
    // From the output it does not reach: failed, or lost.
    if reached < outputs {
        out[reached..].fill(if failed { Fate::Failed } else { Fate::Lost });
    }
    for (k, o) in out.iter_mut().enumerate().skip(1) {
        if let Fate::Held(h) = o {
            let w = &moments[k];
            h.kappa = std::array::from_fn(|a| w[a][1] / w[a][0]);
            h.kappa2 = std::array::from_fn(|a| w[a][2] / w[a][0]);
        }
    }
    let counts = Counts { steps, failed: u64::from(failed), since: failed.then_some(t.time) };
    Record { outputs: out, listed, samples, counts }
}

/// What a group of trajectories leaves: its sums (estimate.rs `Group`), its held atoms' (r, κ) at the listed outputs, its
/// states at the outputs (the pilot's), its counts.
struct Part {
    sums: Group,
    listed: Vec<Vec<(V3, V3)>>,
    samples: Vec<Vec<State>>,
    counts: Counts,
}
impl std::borrow::Borrow<Group> for &Part {
    fn borrow(&self) -> &Group {
        &self.sums
    }
}
impl Part {
    /// The same groups' further members, after this part's.
    fn absorb(&mut self, other: Part) {
        self.sums.absorb(other.sums);
        for (l, x) in self.listed.iter_mut().zip(other.listed) {
            l.extend(x);
        }
        for (s, x) in self.samples.iter_mut().zip(other.samples) {
            s.extend(x);
        }
        self.counts.add(&other.counts);
    }
}

/// What a group's records fold into (`Part`): the outputs, the listed outputs, and whether the states are kept.
#[derive(Clone, Copy)]
struct Shape {
    outputs: usize,
    listed: usize,
    keep: bool,
}
impl Shape {
    fn of(plan: &Plan, keep: bool) -> Shape {
        Shape { outputs: plan.times.len(), listed: plan.listed.len(), keep }
    }
    /// A group's records added in their order.
    fn fold(&self, records: impl Iterator<Item = Record>) -> Part {
        let mut part = Part {
            sums: Group::new(self.outputs),
            listed: vec![vec![]; self.listed],
            samples: vec![vec![]; if self.keep { self.outputs } else { 0 }],
            counts: Counts::default(),
        };
        for r in records {
            part.sums.add(&r.outputs);
            for (l, x) in part.listed.iter_mut().zip(r.listed) {
                l.extend(x);
            }
            for (s, x) in part.samples.iter_mut().zip(r.samples) {
                s.extend(x);
            }
            part.counts.add(&r.counts);
        }
        part
    }
}

/// Trajectory `index` of a run at the step τ: its initial state from the ensemble (ensemble.rs `Ensemble::draw`), run
/// through the plan (`run`). A function of the index and the step: the same index at another step is the same atom on
/// the same streams (module documentation, # The step).
fn trajectory(model: &Run, ensemble: &Ensemble, tau: f64, plan: &Plan, keep: bool, index: u64) -> Record {
    run(model, ensemble.draw(SEED, index), index, tau, plan, keep)
}

/// A level of a hand-out (`run_groups`): the members `members` of the groups of the numbers `groups` (all of them:
/// 0..GROUP), each trajectory made by `record` from its number, a group's records folded as `shape` says.
struct Hand<'a> {
    groups: &'a [u64],
    members: std::ops::Range<u64>,
    shape: Shape,
    record: &'a (dyn Fn(u64) -> Record + Sync),
}
impl Hand<'_> {
    /// The members run of a group.
    fn size(&self) -> u64 {
        self.members.end - self.members.start
    }
    /// The level's trajectories.
    fn count(&self) -> u64 {
        self.groups.len() as u64 * self.size()
    }
    /// The number of the level's k-th trajectory: member k mod size of the k / size-th group.
    fn number(&self, k: u64) -> u64 {
        self.groups[(k / self.size()) as usize] * GROUP + self.members.start + k % self.size()
    }
}

/// One hand-out of trajectories: the levels `levels`, their trajectories in one list -- level after level, each level's
/// in its numbers' order -- run on the solve's one pool of threads (rayon's: the pool its decisions' loops run on between
/// the trajectories, every core unless the process is told otherwise; what a thread keeps for the model,
/// model/assemble.rs `Held`, it keeps from one call to the next). Each thread takes the list's next trajectory as it ends
/// one, so a faster core takes more of them, the groups finish nearly in order and the first after about one
/// trajectory's time. The threads have no trajectory left only at the list's end: what is known to be run together is
/// handed out together (`pair`), the level of the longer trajectories first, so that the threads that wait for the
/// list's last trajectories wait for its shortest. Each group's records are added in its members' order on this thread
/// (`Shape::fold`, nothing depends on the threads), each level's groups returned in their order; whenever the solve's
/// next message is due (`Messages::due`), `progress` takes the first level's groups finished so far in order and sends
/// the result of what is in with them (module documentation, # Checkpoints); where it has nothing to send the message
/// stays due, and is asked for again when the next group is in. A message is made on this thread by loops that run on
/// the pool (`pooled`, `accounts`), whose threads are all in their trajectories: between two trajectories a thread
/// takes what waits for the pool, so a message waits for the next trajectory to end on any thread and not for the
/// list's end.
fn run_groups(levels: &[Hand], messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Vec<Part>> {
    // Where each level's trajectories begin in the list, and the list's length.
    let starts: Vec<u64> = levels.iter().scan(0, |next, level| Some(std::mem::replace(next, *next + level.count()))).collect();
    let stop: u64 = levels.iter().map(Hand::count).sum();
    // The list's k-th trajectory: its level and its place there.
    let place = |k: u64| {
        let level = starts.partition_point(|&start| start <= k) - 1;
        (level, k - starts[level])
    };
    let next = AtomicU64::new(0);
    let threads = rayon::current_num_threads().min(stop as usize);
    let mut waiting: Vec<Vec<Vec<Option<Record>>>> = levels.iter().map(|level| (0..level.groups.len()).map(|_| (0..level.size()).map(|_| None).collect()).collect()).collect();
    let mut parts: Vec<Vec<Option<Part>>> = levels.iter().map(|level| (0..level.groups.len()).map(|_| None).collect()).collect();
    // This thread stays here for the records and the messages; the pool's threads run the trajectories.
    rayon::in_place_scope(|scope| {
        let (sender, receiver) = mpsc::channel::<(usize, u64, Record)>();
        for _ in 0..threads {
            let (sender, next, place) = (sender.clone(), &next, &place);
            scope.spawn(move |_| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= stop {
                    break;
                }
                let (level, k) = place(k);
                if sender.send((level, k, (levels[level].record)(levels[level].number(k)))).is_err() {
                    break;
                }
                // What else waits for the pool: a message's loops (below).
                rayon::yield_now();
            });
        }
        drop(sender);
        let mut left: usize = levels.iter().map(|level| level.groups.len()).sum();
        // Whether the message that is due was asked for and had nothing to send (`Messages::again`), with no group in
        // since: it stays due, and is asked for again with the next group.
        let mut asked = false;
        while left > 0 {
            let due = messages.due();
            if due.is_zero() && !asked {
                let prefix: Vec<&Part> = parts.first().map_or(vec![], |first| first.iter().map_while(Option::as_ref).collect());
                progress(&prefix, messages);
                asked = messages.due().is_zero();
                continue;
            }
            // The next record: until the message is due, or -- due, with nothing to send -- whenever it comes.
            let received = if due.is_zero() { receiver.recv().map_err(|_| mpsc::RecvTimeoutError::Disconnected) } else { receiver.recv_timeout(due) };
            match received {
                Ok((level, k, r)) => {
                    let size = levels[level].size();
                    let (g, m) = ((k / size) as usize, (k % size) as usize);
                    let members = &mut waiting[level][g];
                    members[m] = Some(r);
                    if members.iter().all(Option::is_some) {
                        parts[level][g] = Some(levels[level].shape.fold(members.iter_mut().map(|r| r.take().expect("a member's record"))));
                        left -= 1;
                        asked = false;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });
    parts.into_iter().map(|level| level.into_iter().map(|p| p.expect("a group's trajectories ended without their part")).collect()).collect()
}

/// What the pilot fixes for the run (module documentation, # Stopping), per output: the model's error on the held atoms
/// (populations absolute, T relative; T's unbounded where no pilot atom reads it); and T(0)'s relative error, its
/// quadrature's last two levels' difference (ensemble.rs `Ensemble::kinetic`).
#[derive(Clone)]
struct Fixed {
    model: Vec<[f64; 2]>,
    start: f64,
}

/// The step's share of the targets on what is shown and on T (relative), s·ε (SPEC §2.2, §4): what its error may take
/// before its check is read (module documentation, # The step): the pilot's size rests on it, the model's decisions
/// take it for the step's error, and a result is booked at it until the pilot's two levels are read. And the most a
/// rung of untested order may book (`check`).
fn step_bound() -> [f64; 2] {
    [STEP_SHARE * POPULATION_TARGET, STEP_SHARE * TEMPERATURE_TARGET]
}

/// The room the statistics may take of a target ε beside a model error A and a step error b, √(ε² − A² − b²) (module
/// documentation, # Stopping); none where the two fill the target.
fn room(target: f64, model: f64, step: f64) -> f64 {
    (target * target - model * model - step * step).max(0.).sqrt()
}

/// One quantity held to its target at one output (module documentation, # Stopping): what is shown, or the total T.
/// Its account (ledger.rs: the model's error A, the step's part b, the statistics S, as the ledger books them)
/// against its target ε; and what its need takes beside them: the statistics' jackknife interval and the
/// trajectories it was read on, and on what is shown the unseen share, which the statistics are no less than. `held`:
/// the share of the step's part (`Bound`) that its account takes -- all of it on what is shown, the held share on T.
#[derive(Clone, Copy, Debug)]
struct Look {
    output: usize,
    target: f64,
    account: Account,
    interval: f64,
    trajectories: f64,
    shown: bool,
    held: f64,
}
impl Look {
    /// The look beside another step's part: its account with that part in the step's place.
    fn beside(&self, step: &Bound) -> Look {
        Look { account: Account { step: self.held * if self.shown { step.shown } else { step.temperature }, ..self.account }, ..*self }
    }
    /// Whether its total lies within its target: the ledger's own test of an output (ledger.rs `Ledger::close`).
    fn met(&self) -> bool {
        self.account.total() <= self.target
    }
    /// The room its statistics may take (`room`).
    fn room(&self) -> f64 {
        room(self.target, self.account.model, self.account.step)
    }
    /// Whether some number of trajectories brings it within its target: the model's error and the step's leave a room.
    fn fits(&self) -> bool {
        self.room() > 0.
    }
    /// The trajectories it needs: its interval, falling as N^{−1/2}, within its room; on what is shown no fewer than the
    /// draws that leave the room unseen (estimate.rs `draws_for`). None where no number fits.
    fn need(&self) -> f64 {
        let room = self.room();
        let draws = self.trajectories * (self.interval / room).powi(2);
        if self.shown { draws.max(draws_for(room)) } else { draws }
    }
}

/// The looks of a run (module documentation, # Stopping) by its estimates, the pilot's fixed errors and the step's
/// part: at every output after t = 0 what is shown, and the total T (the mean over the three lab axes: what T's target
/// is on). A T without an account -- one of its parts is unread -- is not met; where it is its statistics that are
/// unread, its need is predicted from the normal theory of one axis's variance over the N h atoms held, the relative
/// interval z √(2/(N h)) -- the total's is at most that (three axes' mean; equal where one axis holds all the energy).
/// The outputs' looks are made on the pool, each output's by itself, and come in the outputs' order.
fn looks(estimates: &[Estimate], fixed: &Fixed, step: &[Bound]) -> Vec<Look> {
    let (population, temperature) = accounts(estimates, fixed, step);
    estimates
        .par_iter()
        .enumerate()
        .skip(1)
        .flat_map_iter(|(k, e)| {
            let (h, n) = (1. - e.loss, e.trajectories);
            let shown = Look { output: k, target: POPULATION_TARGET, account: population[k], interval: e.shown_interval(), trajectories: n, shown: true, held: 1. };
            let account = temperature[k].unwrap_or(Account { model: h * fixed.model[k][1], step: h * step[k].temperature, statistics: f64::INFINITY });
            let interval = if account.statistics.is_finite() { account.statistics } else { h * CONFIDENCE * (2. / (n * h)).sqrt() };
            [shown, Look { output: k, target: TEMPERATURE_TARGET, account, interval, trajectories: n, shown: false, held: h }]
        })
        .collect()
}

/// The outputs where some look leaves the statistics no room (SPEC §5.2): no number of trajectories fits.
fn crowded(looks: &[Look]) -> Vec<usize> {
    let mut outputs: Vec<usize> = looks.iter().filter(|l| !l.fits()).map(|l| l.output).collect();
    outputs.dedup();
    outputs
}

/// Each output's accounts of what is shown and of the total T (ledger.rs `Account`) by the run's estimates, the model's
/// errors on the held atoms per output (`Fixed`) and the step's part, T's where read: the held share 1 − L weighs the
/// model's error on the rows as shown, and T's whole accounts (module documentation, # Stopping); t = 0 is the
/// request's state, without error, and the ensemble's moments, T's by quadrature: its accounts are the quadrature's
/// error alone (`Fixed::start`), in the discretization's part (SPEC §4). The model's error on what is shown is at most
/// 1, a probability's. T has an account where its statistics, the model's error and its step error are all read. The
/// outputs' accounts are made on the pool, each output's by itself.
fn accounts(estimates: &[Estimate], fixed: &Fixed, step: &[Bound]) -> (Vec<Account>, Vec<Option<Account>>) {
    estimates
        .par_iter()
        .zip(&fixed.model)
        .zip(step)
        .enumerate()
        .map(|(k, ((e, a), b))| {
            if k == 0 {
                return (Account::default(), Some(Account { step: fixed.start, ..Account::default() }));
            }
            let h = 1. - e.loss;
            let population = Account { model: (a[0] * h).min(1.), step: b.shown, statistics: e.shown_statistics() };
            let (relative, bound) = (e.temperature_interval(), b.temperature);
            let temperature = (relative.is_finite() && a[1].is_finite() && bound.is_finite()).then_some(Account { model: h * a[1], step: h * bound, statistics: h * relative });
            (population, temperature)
        })
        .unzip()
}

/// What the pilot measured of the model's units (SPEC §5.2): the errors of the units read at the states (`Sampled`: the
/// states' rows one after another, a row the read units' errors on the populations and on T in their order), each state's
/// output and group, and how each folded unit is booked (`booked`).
struct Measured<'a> {
    errors: &'a [[f64; 2]],
    output: &'a [usize],
    group: &'a [usize],
    booked: &'a [Booked],
}
impl Measured<'_> {
    /// Per output and unit the unit's error on the held atoms (on their populations, absolute; on T, relative). A unit
    /// read: the mean over the states of the pilot's groups but `left`; where they hold no atom nothing is measured, and
    /// the error is its bound there, 1 on a population and none on T (the run's atoms at such an output are at most the
    /// share the pilot left unseen, and T has no account without a pilot's T: its look fits no number of trajectories
    /// and is not chased). A unit booked at its gate's bound: that bound, at every output. The outputs are made on the
    /// pool, each by itself: an output's sums take its states in the states' order.
    fn means(&self, outputs: usize, left: Option<usize>) -> Vec<Vec<[f64; 2]>> {
        let read = self.booked.iter().filter(|b| matches!(b, Booked::Read(_))).count();
        // Each output's states, in their order.
        let mut states: Vec<Vec<usize>> = vec![vec![]; outputs];
        for (state, &k) in self.output.iter().enumerate() {
            states[k].push(state);
        }
        states
            .par_iter()
            .map(|states| {
                let (mut sum, mut count) = (vec![[0.; 2]; read], 0.);
                for &state in states.iter().filter(|&&state| Some(self.group[state]) != left) {
                    for (s, e) in sum.iter_mut().zip(&self.errors[state * read..][..read]) {
                        s[0] += e[0];
                        s[1] += e[1];
                    }
                    count += 1.;
                }
                self.booked
                    .iter()
                    .map(|b| match *b {
                        Booked::Bound(bound) => [bound; 2],
                        Booked::Read(place) if count > 0. => [sum[place][0] / count, sum[place][1] / count],
                        Booked::Read(_) => [1., f64::INFINITY],
                    })
                    .collect()
            })
            .collect()
    }
}

/// The units' errors (`Measured::means`) added up per output (populations, T; SPEC §4), every folded unit's and what the
/// reach leaves out of the initial ensemble (of the atoms, of their kinetic energy; model/reach.rs `Model::left_out`).
fn sums(means: &[Vec<[f64; 2]>], unsampled: [f64; 2]) -> Vec<[f64; 2]> {
    means.iter().map(|units| units.iter().fold(unsampled, |a, m| [a[0] + m[0], a[1] + m[1]])).collect()
}

/// The model's error at an output after t = 0 from the sum `whole` of its units' there (`sums`), less one unit's where
/// `without` gives its error there and the units' number: a population's at most 1. A read error leaves the sum by
/// subtraction -- one output's work, not every unit's again; at an output no state was read at, every read unit's
/// error is its bound alike (`Measured::means`), and the others' bounds stand.
fn model_error_at(whole: [f64; 2], without: Option<([f64; 2], usize)>, unsampled: [f64; 2]) -> [f64; 2] {
    let less = |i: usize| match without {
        None => whole[i],
        Some((unit, _)) if unit[i].is_finite() => whole[i] - unit[i],
        Some((_, units)) => if units > 1 { f64::INFINITY } else { unsampled[i] },
    };
    [less(0).min(1.), less(1)]
}

/// The model's error per output from the sums `whole` of its units' (`model_error_at`): none at t = 0, the request's
/// state.
fn model_error(whole: &[[f64; 2]], unsampled: [f64; 2]) -> Vec<[f64; 2]> {
    (0..whole.len()).map(|k| if k == 0 { [0.; 2] } else { model_error_at(whole[k], None, unsampled) }).collect()
}

/// The counted work (decide.rs `Work`) a trajectory spends per µs at a pilot state under a model: one step's -- the
/// integrator's (adapt.rs `advance_work`) on the path of the state's own momentum, velocity and τ through `middle`, its
/// cluster's middle at the trap's centre (a price needs the generator's sizes, not the state's own point: one middle a
/// cluster serves every state of a model), and its middles' (`assembly`, the model's count of one) -- over its τ.
fn rate(run: &Run, assembly: f64, middle: &crate::model::Middle, state: &State) -> f64 {
    let at = state.sample.at;
    let velocity = at.kappa.map(|k| run.hbar_over_mass() * k);
    let (work, assemblies) = adapt::advance_work(middle, at.kappa, velocity, state.tau);
    (work + assemblies as f64 * assembly) / state.tau
}

/// A state a mean over the run is read at (`one_each`): a pilot trajectory's at one output, with its group's place
/// among the pilot's and its weight in the mean, the outputs after t = 0 its trajectory holds its atom at.
struct Pick<'a> {
    state: &'a State,
    group: usize,
    weight: f64,
}

/// The states a mean over the run is read at -- the rates a model is priced by (`Pricing`), a table's whole error
/// (`attempt`) --: one state of every trajectory of `parts`, the pilot's groups with their states kept.
///
/// What is read is the mean of a function f of the state over the run's held states, Σ_i Σ_{k ∈ H_i} f_ik / Σ_i |H_i|,
/// H_i the outputs after t = 0 that trajectory i holds its atom at: a trajectory's work is its rate over the time it
/// is held, and an error is booked on the atoms that are held. A trajectory is the ensemble's independent draw and its
/// states along the run are not, so each trajectory gives one state, f at one output of H_i for the whole of its sum:
/// f_i·|H_i|, the one-point rule, and the mean is Σ_i |H_i| f_i / Σ_i |H_i| (`over_the_run`). For the rule to read the
/// whole run the points are spread evenly over it: trajectory i takes the output of H_i that lies the fraction
/// u_i = (ρ_i + ½)/n of the way through it, ρ_i its rank among the n trajectories -- the midpoints of n equal parts of
/// the run. The rank of the member m of the group at the place g of G is ρ = m·G + g: a group's own members lie a
/// whole 1/(members) of the run apart, and the groups are shifted against one another by 1/n. So every group reads
/// the whole run by itself -- leaving one out, as the pricing's jackknife does, leaves no stretch of the run unread
/// and reads the run's own change in time as no spread --, and all of them together read it at n evenly spaced
/// points. A function of the trajectory's number alone: its group's place and its own place in the group. A
/// trajectory that holds its atom at no output after t = 0 has no state to give, and weighs nothing.
fn one_each<'a>(parts: &[&'a Part]) -> Vec<Pick<'a>> {
    let groups = parts.len();
    let mut picks = vec![];
    for (group, part) in parts.iter().enumerate() {
        // The group's trajectories in its members' order: every one is held at t = 0.
        let members = part.samples.first().map_or(&[][..], Vec::as_slice);
        for (member, first) in members.iter().enumerate() {
            let held: Vec<&State> = part.samples.iter().skip(1).filter_map(|at| at.iter().find(|state| state.index == first.index)).collect();
            let through = ((member * groups + group) as f64 + 0.5) / (members.len() * groups) as f64;
            if let Some(&state) = held.get((through * held.len() as f64) as usize) {
                picks.push(Pick { state, group, weight: held.len() as f64 });
            }
        }
    }
    picks
}

/// The mean over the run's held states of the values read at `picks` (`one_each`: Σ_i |H_i| f_i / Σ_i |H_i|), of the
/// groups but `left`; zero where no state is read -- nothing is held to err or to work on.
fn over_the_run(picks: &[Pick], values: impl Iterator<Item = f64>, left: Option<usize>) -> f64 {
    let (sum, weight) = picks.iter().zip(values).filter(|(pick, _)| Some(pick.group) != left).fold((0., 0.), |a, (pick, value)| (a.0 + pick.weight * value, a.1 + pick.weight));
    if weight > 0. { sum / weight } else { 0. }
}

/// A unit to keep (`Pricing::priced`) and its predicted saving.
struct Candidate {
    unit: usize,
    saving: Read,
}

/// The pilot's groups but one (or all of them) as the pricing reads them (`Pricing`): the estimates and the units' mean
/// errors over them (per output and unit), and the current model's W there.
struct Level<'a> {
    left: Option<usize>,
    estimates: std::borrow::Cow<'a, [Estimate]>,
    means: std::borrow::Cow<'a, [Vec<[f64; 2]>]>,
    /// The units' mean errors added up (`sums`).
    whole: Vec<[f64; 2]>,
    /// Per output the trajectories its looks need at the current model's error, and the outputs after t = 0 by that
    /// need, the largest first.
    needs: Vec<f64>,
    order: Vec<usize>,
    /// The trajectories the current model needs there: the largest of `needs`.
    need: f64,
    /// What the reach leaves out (`sums`) and T(0)'s error (`Fixed::start`), as the looks take them.
    unsampled: [f64; 2],
    start: f64,
}
impl<'a> Level<'a> {
    /// The level of the groups but `left` from their estimates and the units' mean errors over them, beside the errors
    /// `fixed` and the step's share.
    fn new(left: Option<usize>, estimates: std::borrow::Cow<'a, [Estimate]>, means: std::borrow::Cow<'a, [Vec<[f64; 2]>]>, fixed: &Fixed, unsampled: [f64; 2]) -> Level<'a> {
        let whole = sums(&means, unsampled);
        let fixed = Fixed { model: model_error(&whole, unsampled), ..fixed.clone() };
        let mut needs = vec![0.0_f64; whole.len()];
        for look in looks(&estimates, &fixed, &vec![Bound::share(); whole.len()]) {
            needs[look.output] = needs[look.output].max(look.need());
        }
        let mut order: Vec<usize> = (1..whole.len()).collect();
        order.sort_by(|&a, &b| needs[b].total_cmp(&needs[a]));
        let need = needs.iter().copied().fold(0., f64::max);
        Level { left, estimates, means, whole, needs, order, need, unsampled, start: fixed.start }
    }

    /// The trajectories the looks need beside the step's share and a model's error `model` (per output) that is nowhere
    /// more than the level's own -- a unit's removed, every unit's. A look's need falls with the model's error, so an
    /// output needs no more than it does at the level's own (`needs`): the outputs are asked by that need, the largest
    /// first, and no further once the next one's is no more than what those before it need now. The largest over every
    /// output, for one or two outputs' looks.
    fn need_beside(&self, model: &dyn Fn(usize) -> [f64; 2]) -> f64 {
        let mut most = 0.;
        for &k in &self.order {
            if self.needs[k] <= most {
                break;
            }
            let fixed = Fixed { model: vec![[0.; 2], model(k)], start: self.start };
            most = looks(&[self.estimates[0].clone(), self.estimates[k].clone()], &fixed, &[Bound::share(); 2]).iter().map(Look::need).fold(most, f64::max);
        }
        most
    }

    /// The trajectories the looks need beside the units' errors but unit `u`'s and the step's share.
    fn need_without(&self, u: usize) -> f64 {
        self.need_beside(&|k| model_error_at(self.whole[k], Some((self.means[k][u], self.means[k].len())), self.unsampled))
    }

    /// The trajectories the looks need beside the step's share alone: every unit kept, what the reach leaves out left.
    fn need_alone(&self) -> f64 {
        self.need_beside(&|_| model_error_at(self.unsampled, None, self.unsampled))
    }
}

/// The pricing of keeps (SPEC §5.2; module documentation, # The model): the predicted saving of total work W = w·N by
/// keeping a folded unit, as a share of the current model's -- N the trajectories the looks need at the model's
/// errors and the step's share (`Look::need` on the pilot's estimates), and for the kept model the pilot its new attempt runs as well (the
/// current one's is spent); w the mean of `rate` --, spread by the jackknife over the pilot's groups (each recomputed
/// without one: its estimates, the units' errors, the rates). A share, not the work saved: both models' N are the one
/// variance the pilot estimates over their rooms (N ∝ σ̂²), and σ̂²'s spread, common to both, scales the work saved
/// without ever changing its sign; in the share it cancels, and what spreads is what the keep changes -- the rooms and
/// the rates. A model's w is the mean of its rates over the run's held states, read at one state of every pilot
/// trajectory (`one_each`, `over_the_run`), the same states for every model.
struct Pricing<'a> {
    request: &'a Request,
    light: &'a Light,
    kept: &'a [Unit],
    model: &'a Model,
    reference: [f64; GROUND],
    centre: V3,
    units: &'a [Unit],
    measured: &'a Measured<'a>,
    pilot: &'a [&'a Part],
    fixed: &'a Fixed,
    scales: &'a Scales,
    unsampled: [f64; 2],
    /// The states the rates are read at and the current model's rates there.
    rated: Vec<Pick<'a>>,
    own: Vec<f64>,
    /// The trajectories a kept model's attempt runs again before it stands where this one does.
    again: f64,
    /// Every group together.
    all: Level<'a>,
}

impl<'a> Pricing<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        request: &'a Request,
        light: &'a Light,
        kept: &'a [Unit],
        model: &'a Model,
        reference: [f64; GROUND],
        centre: V3,
        units: &'a [Unit],
        measured: &'a Measured<'a>,
        means: &'a [Vec<[f64; 2]>],
        pilot: &'a [&'a Part],
        estimates: &'a [Estimate],
        fixed: &'a Fixed,
        scales: &'a Scales,
        unsampled: [f64; 2],
        again: f64,
    ) -> Pricing<'a> {
        let rated = one_each(pilot);
        let all = Level::new(None, estimates.into(), means.into(), fixed, unsampled);
        Pricing { request, light, kept, model, reference, centre, units, measured, pilot, fixed, scales, unsampled, rated, own: vec![], again, all }
    }

    /// A model's rates at the rated states (`rate`): the current model's (`own`: read once a unit's keep saves at
    /// all -- `saving_at_all` asks no rate --, before any saving is priced), or a kept one's. Each state on the middle
    /// of the cluster that holds its own cluster's first state in that model, made where a rated state asks for it.
    fn rates(&self, model: &Model) -> Vec<f64> {
        let run = Run::new(model, self.reference);
        let assembly = run.assembly_work();
        let mut middles: Vec<Option<crate::model::Middle>> = (0..model.clusters().len()).map(|_| None).collect();
        self.rated
            .iter()
            .map(|pick| {
                let cluster = run.cluster_of(self.model.clusters()[pick.state.sample.cluster].states[0]);
                rate(&run, assembly, middles[cluster].get_or_insert_with(|| run.middle(cluster, self.centre, 0.)), pick.state)
            })
            .collect()
    }

    /// The jackknife's levels: the pilot's groups but each one in turn, the levels made on the pool, each by itself.
    fn leave_one_out(&self) -> Vec<Level<'a>> {
        (0..self.pilot.len())
            .into_par_iter()
            .map(|left| {
                let parts: Vec<&Part> = self.pilot.iter().enumerate().filter(|(g, _)| *g != left).map(|(_, p)| *p).collect();
                Level::new(Some(left), pooled(&parts, self.scales).0.into(), self.measured.means(self.fixed.model.len(), Some(left)).into(), self.fixed, self.unsampled)
            })
            .collect()
    }

    /// The mean over the run of the rates at the rated states of the groups but `left` (`over_the_run`).
    fn mean(&self, rates: &[f64], left: Option<usize>) -> f64 {
        over_the_run(&self.rated, rates.iter().copied(), left)
    }

    /// The share of the current model's W at a level that keeping unit `u` saves at the rates `rates` of its kept model:
    /// a model's W (per µs of run) is its mean rate times its product's trajectories (`need`), the kept model's with the
    /// pilot its new attempt runs.
    fn share(&self, level: &Level, u: usize, rates: &[f64]) -> f64 {
        1. - self.mean(rates, level.left) * (level.need_without(u) + self.again) / (self.mean(&self.own, level.left) * level.need)
    }

    /// The saving of keeping unit `u` at the rates `rates` of its kept model, spread over the jackknife's levels `others`.
    fn saving(&self, others: &[Level], u: usize, rates: &[f64]) -> Read {
        Read { value: self.share(&self.all, u, rates), spread: spread(others.iter().map(|l| self.share(l, u, rates)), others.len() as f64) }
    }

    /// The units whose keep would save at all were its model to cost no more per µs than the current one -- the
    /// trajectories its error's removal saves, less the new pilot (module documentation, # The model). A saving that is
    /// none before its spread pays at no confidence: only these are read on the jackknife's levels and given a kept
    /// model.
    fn saving_at_all(&self) -> Vec<usize> {
        // No unit's removal saves more than the removal of them all (a look's need falls with the model's error):
        // where even that does not pay for the new pilot, none is tried.
        if self.all.need_alone() + self.again >= self.all.need {
            return vec![];
        }
        (0..self.units.len()).into_par_iter().filter(|&u| self.all.need_without(u) + self.again < self.all.need).collect()
    }

    /// The rates of the model that keeps unit `u`, built for it, at the rated states: a keep costs less per µs than the
    /// fold it replaces or far more (a drive kept across a fine-structure splitting, its generator's norm ~10⁷ rad/µs).
    /// None for a unit the model refuses to keep (its fold must exist, `Model::new`): its fold's error stands.
    fn kept_rates(&self, u: usize) -> Option<Vec<f64>> {
        let with = Model::new(self.request, self.light, &[self.kept, std::slice::from_ref(&self.units[u])].concat()).ok()?;
        Some(self.rates(&with))
    }

    /// Unit `u` with its kept model built and rated, its saving spread over the jackknife's levels `others`.
    fn priced(&self, others: &[Level], u: usize) -> Option<Candidate> {
        let rates = self.kept_rates(u)?;
        Some(Candidate { unit: u, saving: self.saving(others, u, &rates) })
    }
}

/// Histograms of one coordinate of the held atoms at the listed outputs on one grid symmetric about 0 that holds them all
/// (the bins' centres): a listed output with no atom carries the one before it. The number of bins is for the display
/// alone -- no estimate reads a histogram --: the square-root choice, ⌊√n⌋ for the n atoms of the fullest listed output
/// (bins of about √n atoms each), made odd so that 0 is a bin's centre, and three at least -- the contract's grid has two
/// points or more (contract §2.2), and the least odd count that has is three.
fn histograms(listed: &[Vec<(V3, V3)>], outputs: &[usize], value: &dyn Fn(&(V3, V3)) -> f64) -> Distributions {
    let reach = listed.iter().flatten().map(|s| value(s).abs()).fold(f64::MIN_POSITIVE, f64::max);
    let most = listed.iter().map(Vec::len).fold(0, usize::max);
    let bins = ((most as f64).sqrt() as usize | 1).max(3);
    let width = 2. * reach / (bins - 1) as f64;
    let grid: Vec<f64> = (0..bins).map(|i| (i as f64 - (bins / 2) as f64) * width).collect();
    let mut last: Vec<f64> = (0..bins).map(|i| if i == bins / 2 { 1. } else { 0. }).collect();
    let probabilities = listed
        .iter()
        .map(|s| {
            if !s.is_empty() {
                last = vec![0.; bins];
                for x in s {
                    let i = ((value(x) / width).round() as isize + (bins / 2) as isize).clamp(0, bins as isize - 1) as usize;
                    last[i] += 1. / s.len() as f64;
                }
            }
            last.clone()
        })
        .collect();
    Distributions { grid, samples: outputs.to_vec(), probabilities }
}

/// What the result's assembly takes beside the estimates: the plan, the run's step (µs), the scales, the request's initial
/// diagonal, the exact moments at t = 0 ((ħ/m)⟨κ_a²⟩ per lab axis, rad/µs), the pilot's fixed errors, the step's part of
/// every output's accounts -- the step's share until the pilot's check is read (`Checked`) --, the ledger's items and
/// note, and the method's description.
struct Context {
    plan: Plan,
    tau: f64,
    scales: Scales,
    initial: [f64; STATES],
    momentum: V3,
    fixed: Fixed,
    step: Vec<Bound>,
    checked: Option<Checked>,
    ledger: Ledger,
    method: String,
}

/// The result of the estimates of `trajectories` trajectories (contract §2.2): at t = 0 the request's initial state and the
/// ensemble's exact moments (T, and n̄ as of one atom with that mean, estimate.rs `occupation`); an output with no atom
/// held carries the last one's populations, one with fewer than two atoms the last T read (its error unknown); each
/// flagged.
fn assemble(c: &Context, estimates: &[Estimate], listed: &[Vec<(V3, V3)>], counts: &Counts, trajectories: u64) -> Solution {
    let (times, outputs) = (&c.plan.times, c.plan.times.len());
    let hm = c.scales.hbar_over_mass;
    let w = c.scales.frequencies;
    let classical = |t: &V3| -> [bool; 3] { std::array::from_fn(|a| classical(t[a], w[a])) };
    let mut ledger = c.ledger.clone();
    let (population, temperature) = accounts(estimates, &c.fixed, &c.step);
    // The step's item: the largest it is booked at over the accounts (weighed by the held share as the totals are). And
    // the check in the method's words: the two steps compared, the pairs, the resolution.
    let mut step = String::new();
    if let Some(check) = &c.checked {
        let largest = (1..outputs).flat_map(|k| temperature[k].iter().map(|a| a.step).chain([population[k].step])).fold(0., f64::max);
        let share = (1 << ORDER) - 1;
        ledger.item(&format!("时间步长 τ = {} μs（全程不变）", c.tau), largest, &format!("轨迹在 τ 与 2τ 下逐条配对比较：两者之差的 1/{share} 在全程各输出时刻上的方均根（已扣除统计噪声的本底），取检查没有排除的上限"));
        let relative = if check.resolution[1].is_finite() { format!("、在温度上低于 {} 时分辨不出", percent(check.resolution[1])) } else { " 时分辨不出，温度上读不出".into() };
        step = format!("（收敛检查：与 {} μs 在 {} 对同号轨迹上比较，看两者之差在全程各输出时刻上的方均根（已扣除统计噪声的本底）；步长误差的方均根在布居上低于 {}{relative}；计入总误差的是检查没有排除的上限：它说的是全程的方均根，不是逐时刻的误差界）", check.companion, check.pairs, percent(check.resolution[0]));
    }
    ledger.close(times, &population, &temperature);
    // T and the motion at t = 0: exact.
    let start_t = c.momentum.map(|m| m / KT_PER_UK);
    let mut s = Solution {
        populations: Vec::with_capacity(outputs),
        estimated_errors: population.iter().map(Account::total).collect(),
        temperature_uk: Vec::with_capacity(outputs),
        temperature_errors_uk: Vec::with_capacity(outputs),
        temperature_total_uk: Vec::with_capacity(outputs),
        temperature_total_errors_uk: Vec::with_capacity(outputs),
        mean_velocity_mps: Vec::with_capacity(outputs),
        mean_occupation: Vec::with_capacity(outputs),
        trap_frequency_khz: w.map(|x| x / TAU * 1e3),
        loss_probability: estimates.iter().enumerate().map(|(k, e)| if k == 0 { 0. } else { e.loss }).collect(),
        velocity: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.1[a] * hm)),
        position: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.0[a])),
        classical: Vec::with_capacity(outputs),
        temperature_estimated_errors: temperature.iter().map(|a| a.map(|a| a.total())).collect(),
        trajectories,
        time_step_us: c.tau,
        method: format!("{}；{trajectories} 条轨迹（每组 {GROUP} 条拉丁超立方），步长 {} μs{step}。", c.method, c.tau),
        reason: None,
        approximations: ledger,
    };
    let (mut empty, mut unread) = (None, vec![]);
    for (k, e) in estimates.iter().enumerate() {
        if k == 0 {
            s.populations.push(c.initial);
            s.temperature_uk.push(start_t);
            s.temperature_errors_uk.push([Some(0.); 3]);
            s.temperature_total_uk.push(start_t.iter().sum::<f64>() / 3.);
            s.temperature_total_errors_uk.push(Some(0.));
            s.mean_velocity_mps.push([0.; 3]);
            s.mean_occupation.push(std::array::from_fn(|a| occupation(c.momentum[a], 1., w[a])));
            s.classical.push(classical(&start_t));
            continue;
        }
        let last = k - 1;
        s.populations.push(if e.held > 0. { e.populations } else { s.populations[last] });
        if e.held == 0. && empty.is_none() {
            empty = Some(k);
        }
        match &e.motion {
            Some(m) => {
                let read = |x: &crate::estimate::Spread| x.read().then_some(x.error);
                s.temperature_uk.push(m.temperature);
                s.temperature_errors_uk.push(m.temperature_errors.each_ref().map(read));
                s.temperature_total_uk.push(m.temperature_total);
                s.temperature_total_errors_uk.push(read(&m.temperature_total_error));
                s.mean_velocity_mps.push(m.mean_velocity);
                s.mean_occupation.push(m.occupation);
                s.classical.push(classical(&m.temperature));
                if m.temperature_errors.iter().any(|x| !x.read()) && e.held > 0. {
                    unread.push(k);
                }
            }
            None => {
                let (t, total, v, n) = (s.temperature_uk[last], s.temperature_total_uk[last], s.mean_velocity_mps[last], s.mean_occupation[last]);
                s.temperature_uk.push(t);
                s.temperature_errors_uk.push([None; 3]);
                s.temperature_total_uk.push(total);
                s.temperature_total_errors_uk.push(None);
                s.mean_velocity_mps.push(v);
                s.mean_occupation.push(n);
                s.classical.push([false; 3]);
                if e.held > 0. {
                    unread.push(k);
                }
            }
        }
    }
    if let Some(k) = empty {
        s.approximations.flag(format!("{} 起已无原子被俘获（丢失概率 1）：此后的布居、温度与分布沿用此前的值，仅供参考", when(times[k])));
    }
    if !unread.is_empty() {
        s.approximations.flag(format!("{}：被俘获的原子不足两个组，温度沿用此前读到的值，误差未知", span(&unread, times)));
    }
    // T shown with its statistics where its model error or its step error is unread.
    let unbooked: Vec<usize> = (1..outputs)
        .filter(|&k| estimates[k].motion.as_ref().is_some_and(|m| m.temperature_errors.iter().all(Spread::read)) && !(c.fixed.model[k][1].is_finite() && c.step[k].temperature.is_finite()))
        .collect();
    if !unbooked.is_empty() {
        s.approximations.flag(format!("{}：这些时刻被俘获的原子不足，模型或步长对温度的误差读不出", span(&unbooked, times)));
    }
    if let Some(t) = counts.since {
        s.approximations.flag(format!("{} 条轨迹的状态在数值上无法再推进（最早在 {}）：此后不计入结果", counts.failed, when(t)));
    }
    // The worst output beyond the population target: what keeps it there.
    if let Some((k, a)) = population.iter().enumerate().max_by(|x, y| x.1.total().total_cmp(&y.1.total())) {
        if a.total() > POPULATION_TARGET {
            let cause = if a.statistics >= a.model.max(a.step) {
                if trajectories >= MOST_TRAJECTORIES {
                    format!("统计误差为目标的 {:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}", a.statistics / POPULATION_TARGET)
                } else {
                    format!("统计误差 {}（{trajectories} 条轨迹，计算未完成）", percent(a.statistics))
                }
            } else if a.model >= a.step {
                format!("模型近似误差 {}", percent(a.model))
            } else {
                format!("时间步长误差 {}", percent(a.step))
            };
            s.reason = Some(format!("{}估计误差 {}（目标 {}）：{cause}", when(times[k]), percent(a.total()), percent(POPULATION_TARGET)));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pilot is the smallest ensemble of whole groups that sees a share as large as the statistics' share of the
    /// population target, z²(1/(√(1 − s²) ε) − 1) = 85.4 trajectories at z = 2, s² = 1/5, ε = 0.05: whole groups whose
    /// unseen share fits that room, where a group fewer's does not.
    #[test]
    fn the_pilot_sees_the_target() {
        let (n, room) = (pilot_size(), (1. - STEP_SHARE * STEP_SHARE).sqrt() * POPULATION_TARGET);
        assert!(n % GROUP == 0 && unseen(n as f64) <= room && unseen((n - GROUP) as f64) > room, "{n}");
    }

    /// The first rung is the largest even part of the run's length that is admissible for the run and whose double is
    /// for the companion (`first_rung`). Rb-87 at 30 µK: the reach's largest speed 0.2185 µm/µs, its rms speed 0.0928.
    /// - No light, a 74.65 kHz trap, a run of 10 µs: π/ω = 6.70 µs bounds the companion, 3.35 µs the run's step: the
    ///   run over 4, 2.5 µs. A 1 kHz trap bounds nothing: half the run.
    /// - A pattern of 2k = 16.1 µm⁻¹: the run below 2π/(K v_max) = 1.79 µs, the companion below 4.20 µs, so the run
    ///   below 1.79 µs: the run's length over 6, 1.667 µs. At √10 k the run below 1.13 µs: its length over 10, 1 µs.
    /// - In every case the rung is admissible with its double, and the next larger even part is not.
    /// - The first rung is a rung of the ladder down to `MOST_STEPS` steps: there is none for a run of 10⁹ µs in the
    ///   74.65 kHz trap (3.35 µs at the most: 3·10⁸ steps), nor at a frequency or a speed that is no number or
    ///   unbounded -- and the count ends.
    #[test]
    fn the_first_rung_is_admissible_and_its_double_is_for_the_typical_atom() {
        let khz = |f: f64| std::f64::consts::TAU * f * 1e-3;
        let hm = hbar_over_mass(1.4432e-25);
        let fastest = |t: f64| (2. * (MOST_TRAJECTORIES as f64 / (CONFIDENCE * CONFIDENCE) + 1.).ln() * KT_PER_UK * t * hm).sqrt();
        let (fast, typical) = (fastest(30.), typical_speed(hm, 30.));
        assert!((fast - 0.2185).abs() < 5e-4 && (typical - 0.0928).abs() < 5e-4, "{fast} {typical}");
        assert_eq!(first_rung(10., khz(74.65), &[], fast, typical), Some(2.5));
        assert_eq!(first_rung(10., khz(1.), &[[0., 0.]], fast, typical), Some(5.));
        assert!((TAU / (16.1 * fast) - 1.79).abs() < 5e-3 && (TAU / (16.1 * typical) - 4.20).abs() < 1e-2);
        assert_eq!(first_rung(10., khz(74.65), &[[0., 16.1]], fast, typical), Some(10. / 6.));
        let k = 16.106 / 2. * 10f64.sqrt();
        assert!((TAU / (k * fast) - 1.13).abs() < 5e-3 && (TAU / (k * typical) - 2.66).abs() < 1e-2);
        assert_eq!(first_rung(10., khz(100.), &[[0., k]], fast, typical), Some(1.));
        let both = |tau: f64, w: f64, sampled: &[[f64; 2]], fast: f64, typical: f64| admissible(tau, w, sampled, fast) && admissible(2. * tau, w, sampled, typical);
        for (run, w, sampled, fast, typical) in [
            (10., khz(910.7), vec![[0., 16.1]], fast, typical),
            (20., khz(79.), vec![[0., 16.1]], fastest(141.), typical_speed(hm, 141.)),
            (16., khz(20.3), vec![[0., 32.2]], fast, typical),
            (10., khz(74.65), vec![[0., 16.1]], 0.0089, 0.0038),
            (10., khz(74.65), vec![[0., 0.], [2., 16.1], [0.5, 8.]], fast, typical),
            (100., khz(100.), vec![[0., k]], fast, typical),
        ] {
            let tau = first_rung(run, w, &sampled, fast, typical).unwrap();
            let pairs = (run / (2. * tau)).round();
            assert!((run / (2. * pairs) - tau).abs() <= 1e-12 * tau && both(tau, w, &sampled, fast, typical), "{run} {tau}");
            assert!(pairs == 1. || !both(run / (2. * (pairs - 1.)), w, &sampled, fast, typical), "{run} {tau}: a larger rung is admissible");
        }
        assert!(admissible(1.25, khz(74.65), &[[0., 16.1]], fast) && !admissible(2.5, khz(74.65), &[[0., 16.1]], fast) && admissible(2.5, khz(74.65), &[[0., 16.1]], typical));
        assert_eq!(first_rung(1e9, khz(74.65), &[], fast, typical), None);
        assert_eq!(first_rung(10., f64::NAN, &[], fast, typical), None);
        assert_eq!(first_rung(10., khz(74.65), &[[0., 16.1]], f64::INFINITY, typical), None);
        assert_eq!(first_rung(10., khz(74.65), &[[0., 16.1]], fast, f64::NAN), None);
    }

    /// A read on unlimited groups (one look's interval is z of the error), and one on the pilot's eleven.
    fn read(value: f64, error: f64) -> Read {
        Read { value, spread: Spread { error, groups: f64::INFINITY } }
    }
    fn pilot(value: f64, error: f64) -> Read {
        Read { value, spread: Spread { error, groups: 11. } }
    }
    /// The reads between two levels with nothing on the loss and the rows, and `t` on the total T.
    fn at(t: Option<Read>) -> Between {
        Between { loss: read(0., 0.), shown: [read(0., 0.); STATES], temperature: t }
    }

    /// One look on the total T that needs `need` trajectories beside no step error: read on 800, its interval
    /// ε √(need/800), every atom held.
    fn needing(need: f64) -> Vec<Look> {
        vec![look(0., 0., TEMPERATURE_TARGET * (need / 800.).sqrt())]
    }
    /// The step's part on T whose square is the share `beta` of the target's.
    fn part(beta: f64) -> Bound {
        Bound { shown: 0., temperature: beta.sqrt() * TEMPERATURE_TARGET }
    }

    /// The trajectories a run is predicted to take beside a step's part (`needed`): a look that needs 400 beside none
    /// needs 400/(1 − β) beside the share β of its target's square, and no number from β = 1 on; no run is predicted
    /// below the pilot; a total T the levels do not read is no look of the check, while a row whose read excludes
    /// nothing (a bound of 1) leaves what is shown no room.
    #[test]
    fn a_run_s_need_beside_a_step_s_part() {
        let close = |a: f64, b: f64| (a / b - 1.).abs() < 1e-12;
        assert!(close(needed(&needing(400.), &part(0.)), 400.) && close(needed(&needing(400.), &part(0.5)), 800.) && close(needed(&needing(400.), &part(0.75)), 1600.));
        assert_eq!((needed(&needing(400.), &part(1.)), needed(&needing(400.), &part(1.2))), (f64::INFINITY, f64::INFINITY));
        assert_eq!(needed(&needing(20.), &part(0.5)), pilot_size() as f64);
        let unread = Bound { shown: 0., temperature: f64::INFINITY };
        assert_eq!(needed(&needing(400.), &unread), pilot_size() as f64);
        let shown = vec![Look { target: POPULATION_TARGET, shown: true, ..look(0., 0., 0.01) }];
        assert_eq!(needed(&shown, &Bound { shown: 1., temperature: 0. }), f64::INFINITY);
    }

    /// The check's choice between a rung and the rungs below (`choose`), no pairs to ask and nothing run yet, one look
    /// that needs 4000 trajectories beside no step error.
    /// - One rung below: standing costs 1/(1 − β) of that need and the rung 2/(1 − β/16), equal at β = 16/31: the step
    ///   stands a hundredth below it and is halved a hundredth above -- at the turn the two cost the same.
    /// - β = 4 leaves no room here; a rung below it leaves 1 − 1/4: the rung. With no rung below: nothing to choose.
    /// - β = 20 leaves none one rung below either (20/16), and 1 − 20/256 two below: the rung where the ladder has
    ///   two, nothing to choose where it has one.
    #[test]
    fn a_step_stands_where_standing_is_the_lesser_work() {
        let turn: f64 = 16. / 31.;
        let at = |beta: f64, rungs: u32| choose(&needing(4000.), &part(beta), &|_| part(beta), None, 0, 0, 0, rungs);
        assert!((1. / (1. - turn) - 2. / (1. - turn / 16.)).abs() < 1e-12);
        assert_eq!((at(turn - 0.01, 1), at(turn + 0.01, 1), at(turn + 0.01, 0)), (Choice::Stand, Choice::Rung, Choice::Stand));
        assert_eq!((at(4., 1), at(4., 0)), (Choice::Rung, Choice::Neither));
        assert_eq!((at(20., 2), at(20., 1)), (Choice::Rung, Choice::Neither));
    }

    /// The check's choice on a pilot of 88 pairs (`choose`), a run that needs 350 trajectories beside no step error,
    /// one rung below. A read whose far end is the share s of the target's square, nothing of it clear, books s now,
    /// and could book s·88/P on P pairs (what it holds falling as 1/P). In trajectories at τ, P pairs leave
    /// ½(P − 88) + 350/(1 − s·88/P) − 88:
    /// - s = 0.05: standing leaves 280.4, eight pairs more 282.8: the step stands;
    /// - s = 0.15: standing leaves 323.8, 112 pairs 320.8 (104: 320.9; 120: 321.3): pairs, to 112;
    /// - s = 0.6: standing leaves 787, the rung 2·350/(1 − 0.6/16) = 727, 248 pairs 436.7: pairs, a batch of them, to
    ///   twice the pairs read; with no pairs to ask, the rung; with neither, the step stands at its 787.
    /// A read the noise may have put high -- a far end of 0.9, nothing of it clear: it could book 0.9·88/P -- leaves
    /// 3412 standing and 742 a rung below, and 493.1 on 312 pairs: pairs, not the rung. The same far end with 0.6 of
    /// it clear could book no less than 0.6, 875 trajectories on any number of pairs: the rung.
    /// A run that its pilot bounds (it needs 20): at β = 0.9 it needs 200, 112 more, against the pilot twice over at
    /// the rung below, 176: it stands; at β = 0.95 it needs 400, 312 more: the rung.
    #[test]
    fn a_check_asks_for_pairs_where_they_could_be_the_least_work() {
        let pilot = |now: f64, least: f64, most: u64, rungs: u32| choose(&needing(350.), &part(now), &|pairs| part(least + (now - least) * 88. / pairs as f64), None, 88, most, 88, rungs);
        let none = |s: f64, most: u64, rungs: u32| pilot(s, 0., most, rungs);
        let left = |least: f64, s: f64, pairs: f64| 0.5 * (pairs - 88.) + 350. / (1. - least - s * 88. / pairs) - 88.;
        assert!((left(0., 0.05, 88.) - 280.4).abs() < 0.05 && (left(0., 0.05, 96.) - 282.8).abs() < 0.05);
        assert_eq!(none(0.05, MOST_TRAJECTORIES, 1), Choice::Stand);
        assert!((left(0., 0.15, 88.) - 323.8).abs() < 0.05 && (left(0., 0.15, 112.) - 320.8).abs() < 0.05 && left(0., 0.15, 104.) > left(0., 0.15, 112.) && left(0., 0.15, 120.) > left(0., 0.15, 112.));
        assert_eq!(none(0.15, MOST_TRAJECTORIES, 1), Choice::Pairs(112));
        assert!((left(0., 0.6, 88.) - 787.).abs() < 0.5 && (left(0., 0.6, 248.) - 436.7).abs() < 0.05 && left(0., 0.6, 240.) > left(0., 0.6, 248.) && left(0., 0.6, 256.) > left(0., 0.6, 248.));
        assert_eq!((none(0.6, MOST_TRAJECTORIES, 1), none(0.6, 88, 1), none(0.6, 88, 0)), (Choice::Pairs(176), Choice::Rung, Choice::Stand));
        assert!((350. / 0.1 - 88. - 3412f64).abs() < 0.5 && (2. * 350. / (1. - 0.9 / 16.) - 742f64).abs() < 0.5);
        assert!((left(0., 0.9, 312.) - 493.1).abs() < 0.05 && left(0., 0.9, 304.) > left(0., 0.9, 312.) && left(0., 0.9, 320.) > left(0., 0.9, 312.));
        assert_eq!((pilot(0.9, 0., MOST_TRAJECTORIES, 1), pilot(0.9, 0.6, MOST_TRAJECTORIES, 1)), (Choice::Pairs(176), Choice::Rung));
        let small = |beta: f64| choose(&needing(20.), &part(beta), &|_| part(beta), None, 88, 88, 88, 1);
        assert_eq!((small(0.9), small(0.95)), (Choice::Stand, Choice::Rung));
    }

    /// A rung whose order is untested books no more than its limit (`choose`), here 0.2 of the target's square on the
    /// total T; a run that needs 4000 trajectories, 88 pairs read. A part of 0.3 is below the turn of the works
    /// (16/31) and would stand; over the limit it does not: with no pairs to ask, the rung below; with neither,
    /// nothing is left. Pairs that could bring it within -- 0.3·88/P ≤ 0.2 from 136 pairs on -- are asked, towards
    /// the number that leaves the least work among those (488: ½ a trajectory a pair against 4000/(1 − 26.4/P)),
    /// a batch of them. At the limit itself the step stands; a total T that is not read is not held to it.
    #[test]
    fn an_untested_rung_books_no_more_than_its_limit() {
        let limit = part(0.2);
        let on = |now: f64, limit: Option<&Bound>, most: u64, rungs: u32| choose(&needing(4000.), &part(now), &|pairs| part(now * 88. / pairs as f64), limit, 88, most, 88, rungs);
        assert_eq!((on(0.3, None, 88, 1), on(0.3, Some(&limit), 88, 1), on(0.3, Some(&limit), 88, 0)), (Choice::Stand, Choice::Rung, Choice::Neither));
        let left = |pairs: f64| 0.5 * (pairs - 88.) + 4000. / (1. - 26.4 / pairs) - 88.;
        assert!(0.3 * 88. / 128. > 0.2 && 0.3 * 88. / 136. < 0.2 && left(480.) > left(488.) && left(496.) > left(488.));
        assert_eq!(on(0.3, Some(&limit), MOST_TRAJECTORIES, 1), Choice::Pairs(176));
        assert_eq!(on(0.2, Some(&limit), 88, 1), Choice::Stand);
        let unread = Bound { shown: 0., temperature: f64::INFINITY };
        assert_eq!(choose(&needing(4000.), &unread, &|_| unread, Some(&limit), 88, 88, 88, 1), Choice::Stand);
    }

    /// The pairs a check asks for are bounded by the rung below (`choose`): a run that needs 4000 trajectories, a read
    /// whose far end is 0.9 of the target's square, nothing of it clear; the rung below costs 2·4000/(1 − 0.9/16) =
    /// 8477, so no more pairs are asked than 88 + 8477/1.5 = 5739. On 88 pairs read, more are asked (their best
    /// number, some 875, is within the bound); on 5744, none -- the rung, however much the pairs could still promise; and
    /// with no rung below there is no bound: pairs still, to the number at which the pairs are the run itself (9168:
    /// 4000/(1 − 0.9·5744/P) = P there).
    #[test]
    fn the_pairs_asked_cost_no_more_than_the_rung_below() {
        let on = |have: u64, rungs: u32| choose(&needing(4000.), &part(0.9), &|pairs| part(0.9 * have as f64 / pairs as f64), None, have, MOST_TRAJECTORIES, 88, rungs);
        assert!((2. * 4000. / (1. - 0.9 / 16.) - 8477f64).abs() < 0.5);
        assert_eq!((on(88, 1), on(5744, 1), on(5744, 0)), (Choice::Pairs(176), Choice::Rung, Choice::Pairs(9168)));
    }

    /// The scout leaves its rung only on what its read shows (`settle`: `choose` beside `clear`, no pairs to ask):
    /// eleven trajectories, a run that needs 4000 on T and on what is shown. A mean square of 0.5 ± 0.04 on the
    /// total T is clear among the family's looks, and what is clear of it leaves the 10 % target no room at this
    /// rung nor at the one below (over 16), and room two below (over 256): the rung is left. The same mean square
    /// under a standard error of 0.2 is clear of nothing: the rung stands for the pilot, far beyond the target though
    /// its far end lies; so does a read below zero, one that is no number, and one whose spread is unread. A row clear
    /// of the populations' target is shown as T is.
    #[test]
    fn the_scout_leaves_a_rung_on_what_its_read_shows() {
        let mut looks = needing(4000.);
        looks.push(Look { target: POPULATION_TARGET, shown: true, ..look(0., 0., POPULATION_TARGET * 5f64.sqrt()) });
        let scout = |read: &Between, rungs: u32| choose(&looks, &clear(read), &|_| clear(read), None, 11, 11, 11, rungs);
        let shown = at(Some(pilot(0.5, 0.04)));
        let seen = clear(&shown).temperature.powi(2);
        assert!(seen / 16. > TEMPERATURE_TARGET.powi(2) && seen / 256. < TEMPERATURE_TARGET.powi(2), "{seen}");
        assert_eq!((scout(&shown, 2), scout(&shown, 1)), (Choice::Rung, Choice::Neither));
        for unclear in [pilot(0.5, 0.2), pilot(-0.5, 0.01), read(f64::NAN, 0.), Read { value: 1., spread: Spread::UNREAD }] {
            assert_eq!(scout(&at(Some(unclear)), 2), Choice::Stand, "{unclear:?}");
        }
        assert_eq!(scout(&at(None), 2), Choice::Stand);
        let mut row = at(None);
        row.shown[3] = pilot(0.01, 0.0005);
        assert_eq!(scout(&row, 2), Choice::Rung);
        assert_eq!(judged(&shown).len(), 1 + STATES + 1);
    }

    /// A rung's own read against the one it inherits (`held`, `refuted`). The rung is held, look by look, by the read
    /// of the lesser far end; a T that its own levels do not read is not held by an inherited one. The own read
    /// refutes the order where it lies clearly above the inherited one among the family's looks: a row's mean square
    /// of 2.7e-4 ± 0.3e-4 where the rung above's 4.8e-4 ± 1.7e-4 allows 3.0e-5 ± 1.1e-5 (a Raman transfer at 2.5 µs
    /// against 5 µs: the step does not resolve the Doppler phase, and the error does not fall); not where its interval
    /// reaches the inherited read's, nor where it lies below it. A total T that one of the two rungs does not read is
    /// not tested, and the row is tested all the same.
    #[test]
    fn a_rung_s_own_read_tests_the_order_it_inherits_by() {
        let mut above = at(Some(read(0.016, 0.0032)));
        above.shown[1] = pilot(4.8e-4, 1.7e-4);
        let inherited = lowered(&above);
        let part = 1. / 16.;
        assert!((inherited.shown[1].value - 4.8e-4 * part).abs() < 1e-18 && (inherited.shown[1].spread.error - 1.7e-4 * part).abs() < 1e-18 && inherited.shown[1].spread.groups == 11.);
        let mut own = at(Some(read(0., 0.01)));
        own.shown[1] = pilot(2.7e-4, 0.3e-4);
        let by = held(&own, Some(&inherited));
        assert_eq!((by.temperature, by.shown[1], by.loss), (inherited.temperature, inherited.shown[1], own.loss));
        assert_eq!(held(&own, None).shown[1], own.shown[1]);
        assert_eq!(held(&at(None), Some(&inherited)).temperature, None);
        assert!(refuted(&own, &inherited));
        let (without, mut above_without) = (Between { temperature: None, ..own.clone() }, inherited.clone());
        assert!(refuted(&without, &inherited));
        above_without.temperature = None;
        assert!(refuted(&own, &above_without));
        own.shown[1] = pilot(2.7e-4, 1.0e-4);
        assert!(!refuted(&own, &inherited));
        own.shown[1] = pilot(1e-5, 1e-6);
        assert!(!refuted(&own, &inherited) && !refuted(&at(None), &inherited));
    }

    /// What a check books, what it has seen, and what it resolves (`unexcluded`, `clear`, `resolution`).
    /// - Booked is what the reads do not exclude: the root of each far end at one look, a read below zero's from zero
    ///   -- T's 0.0016 ± 0.0002 on unlimited groups: √0.002; a row's −1e-4 ± 0.5e-4: √1e-4 --; unbounded on a T that
    ///   is not read, or whose spread is not.
    /// - Seen is the clear part among the family's looks (Student's t at ten degrees of freedom and Šidák's tail over
    ///   the family): a mean square of 0.0144 on the total T (an rms of 12 %) with a standard error of 0.004 on eleven
    ///   groups leaves nothing clear; of 0.36 with 0.04, the root of 0.36 less the family's interval. A loss clear at
    ///   0.04 is seen on what is shown; a T that is not read shows nothing.
    /// - The resolution is the root of one look's t·SE: T's, and what is shown from the rows and the loss.
    /// - More pairs could narrow a read's spread and bring its value towards the least it allows -- its near end at
    ///   one look, nothing of a read within its own interval of zero --, both as 1/N (`narrowed`): on four times the
    ///   pairs a quarter of the spread and of what the value holds above the least; on the pairs read, the read.
    ///   The order leaves a quarter of a step's part a rung below (`Bound::lowered`, p = 2).
    #[test]
    fn the_step_s_check_books_what_it_does_not_exclude() {
        let mut dim = at(Some(read(0.0016, 0.0002)));
        dim.shown[4] = read(-1e-4, 0.5e-4);
        let far = unexcluded(&dim);
        assert!((far.temperature - 0.002f64.sqrt()).abs() < 1e-15 && (far.shown - 1e-4f64.sqrt()).abs() < 1e-15, "{far:?}");
        assert_eq!((unexcluded(&at(None)).temperature, unexcluded(&at(Some(Read { value: 0.1, spread: Spread::UNREAD }))).temperature), (f64::INFINITY, f64::INFINITY));

        let unclear = at(Some(pilot(0.0144, 0.004)));
        assert_eq!(clear(&unclear), Bound { shown: 0., temperature: 0. });
        let one = Spread { error: 0.004, groups: 11. }.interval();
        assert!(resolution(&unclear) == [0., one.sqrt()] && one > CONFIDENCE * 0.004);
        let seen = at(Some(pilot(0.36, 0.04)));
        let family = Spread { error: 0.04, groups: 11. }.interval_over(judged(&seen).len() as f64);
        assert!((clear(&seen).temperature - (0.36 - family).sqrt()).abs() < 1e-15 && family > Spread { error: 0.04, groups: 11. }.interval());
        let mut lossy = at(None);
        (lossy.loss, lossy.shown[2]) = (read(0.04, 0.0001), pilot(1e-4, 4e-4));
        let step = clear(&lossy);
        assert!(step.shown > 0.15 && step.shown < 0.2 && step.temperature == 0., "{step:?}");
        assert!((resolution(&lossy)[0] - Spread { error: 4e-4, groups: 11. }.interval().sqrt()).abs() < 1e-15 && resolution(&lossy)[1] == f64::INFINITY);

        let more = narrowed(&seen, 0.25).temperature.unwrap();
        let own = Spread { error: 0.04, groups: 11. }.interval();
        assert!((more.value - (0.36 - 0.75 * own)).abs() < 1e-15 && more.spread == Spread { error: 0.01, groups: 11. }, "{more:?}");
        assert!((narrowed(&seen, 1.).temperature.unwrap().value - 0.36).abs() < 1e-15 && narrowed(&at(Some(pilot(0.0044, 0.004))), 0.5).temperature == Some(pilot(0.0022, 0.002)));
        assert_eq!(narrowed(&at(Some(pilot(-0.3, 0.004))), 0.5).temperature, Some(pilot(0., 0.002)));
        assert!(narrowed(&at(Some(read(f64::NAN, 0.))), 0.5).temperature.unwrap().value.is_nan());
        assert_eq!(Bound { shown: 0.16, temperature: 0.32 }.lowered(1), Bound { shown: 0.04, temperature: 0.08 });
        assert_eq!(part(0.64).lowered(2).temperature, part(0.64).temperature / 16.);
    }

    /// A look on T at 800 trajectories: the model's error, the step's part, the statistics' interval (each
    /// relative).
    fn look(model: f64, step: f64, interval: f64) -> Look {
        Look { output: 1, target: TEMPERATURE_TARGET, account: Account { model, step, statistics: interval }, interval, trajectories: 800., shown: false, held: 1. }
    }

    /// The trajectories a look needs (`Look::need`): its interval within the room the model's error and the step's
    /// part leave, N₀ (S/R)²; it is met exactly where it needs no more than it has; no number fits where the two
    /// fill the target; on what is shown the need is no less than the draws that leave the room unseen. And a batch
    /// (`batch`): the need in whole groups, more than the product has, at most twice it, at most the cap.
    #[test]
    fn a_look_needs_the_trajectories_that_fit_its_room() {
        for l in [look(0., 0., 0.12), look(0.03, 0.04, 0.08), look(0.03, 0.04, 0.09), look(0.02, 0., 0.3)] {
            let need = 800. * l.interval.powi(2) / (l.target.powi(2) - l.account.model.powi(2) - l.account.step.powi(2));
            assert!(l.fits() && (l.need() / need - 1.).abs() < 1e-12, "{l:?}: {} against {need}", l.need());
            assert_eq!(l.met(), l.need() <= l.trajectories, "{l:?}");
        }
        assert!(!look(0.06, 0.09, 0.01).fits() && !look(0.11, 0., 0.01).fits() && look(0.06, 0.07, 0.01).fits());
        let shown = Look { target: POPULATION_TARGET, shown: true, ..look(0.03, 0.02, 0.) };
        assert_eq!(shown.need(), draws_for(room(POPULATION_TARGET, 0.03, 0.02)));
        assert_eq!((batch(1147.3, 800), batch(300., 800), batch(5000., 800), batch(f64::INFINITY, 800), batch(1e9, MOST_TRAJECTORIES - 8)), (1152, 808, 1600, 1600, MOST_TRAJECTORIES));
    }

    /// An output's estimate with T = `temperature` (µK) per lab axis and the total T's standard error `error` (µK), read
    /// on unlimited groups (the interval is z of the error), every atom held, nothing else to chase.
    fn estimate(temperature: V3, error: f64, trajectories: f64) -> Estimate {
        let exact = Spread { error: 0., groups: f64::INFINITY };
        Estimate {
            trajectories,
            held: trajectories,
            loss: 0.,
            loss_error: exact,
            shown_errors: [exact; STATES],
            populations: [0.; STATES],
            motion: Some(crate::estimate::Motion {
                temperature,
                temperature_errors: [exact; 3],
                temperature_total: temperature.iter().sum::<f64>() / 3.,
                temperature_total_error: Spread { error, groups: f64::INFINITY },
                occupation: [0.; 3],
                mean_velocity: [0.; 3],
            }),
            failed: 0.,
        }
    }

    /// The looks and the accounts are of what is shown and of the total T. A pilot that reads 25, 6 and 10 µK on the
    /// axes has the total 13.67 µK; with a model error of 3 % and a step's part of 2 % there is one look on what
    /// is shown and one on T at every output after t = 0. An interval of 1 % on the total neither keeps the run from
    /// stopping nor asks for a trajectory; one of 15 % does both, by (interval/room)², whatever the axes' own errors
    /// are. A model error or a step part that fills the target crowds its output. T's account holds the three parts,
    /// and the result's total is it.
    #[test]
    fn the_looks_and_the_accounts_are_of_the_total_temperature() {
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.03]], start: 0. };
        let none = Bound { shown: 0., temperature: 0. };
        let step = [none, Bound { shown: 0., temperature: 0.02 }];
        let (n, total) = (800., (25. + 6. + 10.) / 3.);
        let quiet = estimate([25., 6., 10.], 0.005 * total, n);
        let run = [quiet.clone(), quiet];
        let all = looks(&run, &fixed, &step);
        assert_eq!(all.iter().map(|l| (l.output, l.shown, l.account.step)).collect::<Vec<_>>(), [(1, true, 0.), (1, false, 0.02)]);
        assert!(all.iter().all(|l| l.met() && l.need() < n) && crowded(&all).is_empty());
        let noisy = estimate([25., 6., 10.], 0.075 * total, n);
        let run = [noisy.clone(), noisy];
        let t = looks(&run, &fixed, &step)[1];
        let need = n * (CONFIDENCE * 0.075 / room(TEMPERATURE_TARGET, 0.03, 0.02)).powi(2);
        assert!(!t.met() && (t.need() - need).abs() < 1e-9 * need, "{} against {need}", t.need());
        let wide = [none, Bound { shown: 0., temperature: 0.4 }];
        assert!(!looks(&run, &fixed, &wide)[1].fits() && crowded(&looks(&run, &fixed, &wide)) == [1]);
        let mut full = Fixed { model: fixed.model.clone(), start: 0. };
        full.model[1][1] = 0.11;
        assert!(crowded(&looks(&run, &full, &step)) == [1]);
        let account = accounts(&run, &fixed, &step).1[1].unwrap();
        assert!(account.model == 0.03 && account.step == 0.02 && (account.statistics - CONFIDENCE * 0.075).abs() < 1e-14, "{account:?}");
    }

    /// How the folded units are booked (`booked`): of the units with a bound, those of the smallest bounds that add up
    /// within the unseen share of the most trajectories, 4/16388 = 2.44e-4, are booked at them -- 1e-9, 1e-5 and 1e-4
    /// here, 1.1e-4 together; the next, 2e-4, would bring them beyond it, and it is read with everything after it and
    /// with the units no gate bounds, each at its place among those read, in the units' order.
    #[test]
    fn the_units_whose_bounds_fit_the_unseen_share_are_not_read() {
        let made = booked(&[None, Some(1e-5), Some(1e-4), Some(2e-4), None, Some(1e-9), Some(1.)]);
        assert_eq!(made, [Booked::Read(0), Booked::Bound(1e-5), Booked::Bound(1e-4), Booked::Read(1), Booked::Read(2), Booked::Bound(1e-9), Booked::Read(3)]);
        assert!(1e-9 + 1e-5 + 1e-4 <= unseen(MOST_TRAJECTORIES as f64) && 1e-9 + 1e-5 + 1e-4 + 2e-4 > unseen(MOST_TRAJECTORIES as f64));
        assert_eq!(booked(&[Some(0.), Some(0.)]), [Booked::Bound(0.), Booked::Bound(0.)]);
    }

    /// The units' errors per output (`Measured::means`), their sums and a unit's removal (`sums`, `model_error_at`).
    /// Three units, the second booked at its bound 1e-5 (a state's row: the two read units' errors); two groups, output 1
    /// read in both, output 2 in the first alone, output 3 in neither. A read unit's error is the mean over the states of
    /// the output, of the first group's
    /// with the second left out, and its bound (1, none) where no state is read; the bounded unit's is its bound
    /// everywhere. The sums add what the reach leaves out; a read unit leaves a sum by subtraction; where it was not
    /// read, its population's bound leaves as a read error does, and on T the others' bounds stand -- alone, what the
    /// reach leaves out.
    #[test]
    fn the_states_errors_are_the_outputs_means_and_sums() {
        let errors = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08], [0.02, 0.04], [0.06, 0.08]];
        let booking = [Booked::Read(0), Booked::Bound(1e-5), Booked::Read(1)];
        let measured = Measured { errors: &errors, output: &[1, 2, 1], group: &[0, 0, 1], booked: &booking };
        let means = measured.means(4, None);
        assert_eq!(means[1], [[(0.01 + 0.02) / 2., (0.02 + 0.04) / 2.], [1e-5; 2], [(0.03 + 0.06) / 2., (0.04 + 0.08) / 2.]]);
        assert_eq!(means[2], [[0.05, 0.06], [1e-5; 2], [0.07, 0.08]]);
        assert_eq!(means[3], [[1., f64::INFINITY], [1e-5; 2], [1., f64::INFINITY]]);
        assert_eq!(measured.means(4, Some(1))[1], [[0.01, 0.02], [1e-5; 2], [0.03, 0.04]]);
        assert_eq!(measured.means(4, Some(0))[2], [[1., f64::INFINITY], [1e-5; 2], [1., f64::INFINITY]]);
        let unsampled = [1e-3, 2e-3];
        let whole = sums(&means, unsampled);
        assert_eq!(whole[1], [1e-3 + means[1][0][0] + 1e-5 + means[1][2][0], 2e-3 + means[1][0][1] + 1e-5 + means[1][2][1]]);
        let all = model_error(&whole, unsampled);
        assert_eq!((all[0], all[1], all[3]), ([0.; 2], whole[1], [1., f64::INFINITY]));
        assert_eq!(model_error_at(whole[1], Some((means[1][2], 3)), unsampled), [whole[1][0] - means[1][2][0], whole[1][1] - means[1][2][1]]);
        assert_eq!(model_error_at(whole[3], Some((means[3][0], 3)), unsampled), [1., f64::INFINITY]);
        assert_eq!(model_error_at(whole[3], Some((means[3][0], 1)), unsampled), [(whole[3][0] - 1.).min(1.), unsampled[1]]);
    }

    /// The trajectories a level's looks need without a unit (`Level::need_without`) and with every unit kept
    /// (`Level::need_alone`) are the largest need of all the looks at that model's error, to the bit: the outputs not
    /// asked could not have raised it. Seven outputs whose total T is read to 1.5 to 3.6 %, three units whose errors
    /// on T differ from output to output, so that the neediest output changes with the unit removed.
    #[test]
    fn a_level_s_need_without_a_unit_is_the_largest_of_all_its_looks() {
        let outputs = 8;
        let estimates: Vec<Estimate> = (0..outputs).map(|k| estimate([20., 10., 15.], 15. * (0.015 + 0.0035 * ((k * 5) % 7) as f64), 88.)).collect();
        let means: Vec<Vec<[f64; 2]>> = (0..outputs).map(|k| (0..3).map(|u| [0.001 * u as f64, 0.004 * (((k + 2 * u) * 3) % 5) as f64 + 0.03 * ((k + u) % 3 / 2) as f64]).collect()).collect();
        let (fixed, unsampled) = (Fixed { model: vec![[0.; 2]; outputs], start: 0. }, [2.5e-4, 1e-3]);
        let level = Level::new(None, estimates.as_slice().into(), means.as_slice().into(), &fixed, unsampled);
        let direct = |model: &dyn Fn(usize) -> [f64; 2]| -> f64 {
            let fixed = Fixed { model: (0..outputs).map(|k| if k == 0 { [0.; 2] } else { model(k) }).collect(), start: 0. };
            looks(&estimates, &fixed, &vec![Bound::share(); outputs]).iter().map(Look::need).fold(0., f64::max)
        };
        assert_eq!(level.need, direct(&|k| model_error_at(level.whole[k], None, unsampled)));
        let needs: Vec<f64> = (0..3).map(|u| level.need_without(u)).collect();
        for u in 0..3 {
            assert_eq!(needs[u], direct(&|k| model_error_at(level.whole[k], Some((means[k][u], 3)), unsampled)), "unit {u}");
            assert!(needs[u] <= level.need, "unit {u}: {} of {}", needs[u], level.need);
        }
        assert!(needs.iter().any(|&n| n < level.need) && (needs[0] != needs[1] || needs[1] != needs[2]), "{needs:?} of {}", level.need);
        assert_eq!(level.need_alone(), direct(&|_| model_error_at(unsampled, None, unsampled)));
        assert!(level.need_alone() < needs.iter().copied().fold(f64::INFINITY, f64::min));
    }

    /// A pilot's groups with their states kept: `groups` groups of `members` trajectories, numbered as the pilot's are,
    /// each held at the outputs up to `held(group, member)` of `outputs` after t = 0. A state's time is its output's
    /// place, its step its group's.
    fn kept(groups: usize, members: usize, outputs: usize, held: &dyn Fn(usize, usize) -> usize) -> Vec<Part> {
        (0..groups)
            .map(|g| {
                let state = |m: usize, k: usize| State {
                    sample: Sample { cluster: 0, at: crate::model::Point { r: [0.; 3], kappa: [0.; 3], t: k as f64 }, psi: vec![] },
                    tau: g as f64,
                    index: MOST_TRAJECTORIES + (g as u64) * GROUP + m as u64,
                };
                let samples = (0..=outputs).map(|k| (0..members).filter(|&m| k <= held(g, m)).map(|m| state(m, k)).collect()).collect();
                Part { sums: Group::new(outputs + 1), listed: vec![], samples, counts: Counts::default() }
            })
            .collect()
    }

    /// The states a mean over the run is read at (`one_each`) are one of every trajectory, spread evenly over the
    /// run whatever the number of outputs -- 1000, 400, 100 (the counts whose picks lay 8 apart in one stretch of
    /// the run), 88, 8 and 3 after t = 0 --, for the pilot's eleven groups of eight and for the scout's eleven of one:
    /// - with every atom held throughout, each trajectory gives one state; cut the run into as many equal parts as
    ///   there are trajectories and every part's outputs hold one pick (the n-th part's the pick of the n-th rank),
    ///   so the picks' mean place is the run's middle to within half a part, and no output is picked twice while
    ///   the outputs are no fewer than the trajectories;
    /// - each group by itself reads the whole run: its members' picks lie one a member's share of the run;
    /// - a trajectory lost on the way gives a state where it is held, at the same fraction of what it holds, and
    ///   weighs by the outputs it holds; one lost before the first output gives none;
    /// - the mean over the run (`over_the_run`) is the mean over the held states: of a value that is the output's
    ///   place, with every trajectory of a group held as far as its group's place among the groups allows, the picks'
    ///   weighted mean meets the held states' own mean to within half a part of the longest-held trajectory, and
    ///   without a group it is the mean of the others.
    #[test]
    fn the_run_is_read_at_one_state_of_every_trajectory_spread_over_it() {
        for (groups, members) in [(11, GROUP as usize), (11, 1)] {
            let n = groups * members;
            for outputs in [1000usize, 400, 100, 88, 8, 3] {
                let parts = kept(groups, members, outputs, &|_, _| outputs);
                let all: Vec<&Part> = parts.iter().collect();
                let picks = one_each(&all);
                assert_eq!(picks.len(), n);
                assert!(picks.iter().all(|p| p.weight == outputs as f64 && p.state.tau == p.group as f64));
                let mut places: Vec<usize> = picks.iter().map(|p| p.state.sample.at.t as usize).collect();
                // Each group's members, in their order, a member's share of the run apart.
                for g in 0..groups {
                    let own: Vec<usize> = picks.iter().filter(|p| p.group == g).map(|p| p.state.sample.at.t as usize).collect();
                    assert!(own.iter().enumerate().all(|(m, &k)| k >= 1 + m * outputs / members && k <= 1 + (m + 1) * outputs / members), "{outputs} outputs, group {g}: {own:?}");
                }
                places.sort();
                assert!(places.iter().enumerate().all(|(rank, &k)| k == 1 + ((rank as f64 + 0.5) / n as f64 * outputs as f64) as usize), "{outputs} outputs: {places:?}");
                if outputs >= n {
                    assert!(places.windows(2).all(|w| w[0] < w[1]), "{outputs} outputs: a pick twice");
                }
                let mean = over_the_run(&picks, picks.iter().map(|p| p.state.sample.at.t), None);
                assert!((mean - 0.5 * (outputs + 1) as f64).abs() <= 0.5 * (outputs as f64 / n as f64).max(1.), "{outputs} outputs: {mean}");
            }
        }
        // Losses: group g's trajectories are held to the output 100 (g + 1)/11, its last member to none.
        let (groups, members, outputs) = (11usize, GROUP as usize, 100usize);
        let last = |g: usize, m: usize| if m + 1 == members { 0 } else { outputs * (g + 1) / groups };
        let parts = kept(groups, members, outputs, &last);
        let all: Vec<&Part> = parts.iter().collect();
        let picks = one_each(&all);
        assert_eq!(picks.len(), groups * (members - 1));
        for p in &picks {
            let held = outputs * (p.group + 1) / groups;
            let member = (p.state.index % GROUP) as usize;
            let through = ((member * groups + p.group) as f64 + 0.5) / (groups * members) as f64;
            assert_eq!((p.weight, p.state.sample.at.t as usize), (held as f64, 1 + (through * held as f64) as usize));
        }
        let place = |picks: &[Pick], left: Option<usize>| over_the_run(picks, picks.iter().map(|p| p.state.sample.at.t), left);
        let exact = |left: Option<usize>| {
            let (sum, count) = (0..groups).filter(|&g| Some(g) != left).fold((0., 0.), |a, g| {
                let held = (outputs * (g + 1) / groups) as f64;
                (a.0 + (members - 1) as f64 * held * (held + 1.) / 2., a.1 + (members - 1) as f64 * held)
            });
            sum / count
        };
        for left in [None, Some(0), Some(10)] {
            assert!((place(&picks, left) - exact(left)).abs() <= 0.5 * outputs as f64 / (members - 1) as f64, "{left:?}: {} against {}", place(&picks, left), exact(left));
        }
        assert_eq!(over_the_run(&[], std::iter::empty(), None), 0.);
    }

    /// The plan: the windows half a period (at most half an interval) cut at t = 0, the listed outputs evenly spread with
    /// the first and the last, the run's end the last window's.
    #[test]
    fn the_plan_holds_the_windows_and_the_listed_outputs() {
        let times: Vec<f64> = (0..101).map(|k| k as f64).collect();
        let w = std::f64::consts::TAU / 0.5;
        let p = Plan::new(&times, [w, 0.1, w]);
        assert_eq!(p.halves[0], [0.; 3]);
        assert_eq!(p.halves[7], [0.25, 0.5, 0.25]);
        assert_eq!((p.listed.len(), p.listed[0], p.listed[20], p.listed[1]), (21, 0, 100, 5));
        assert_eq!(p.end, 100.5);
        let p = Plan::new(&[0., 2.], [w; 3]);
        assert_eq!((p.listed.clone(), p.end), (vec![0, 1], 2.25));
    }

    /// The histograms hold every atom, sum to one, put 0 at a bin's centre, and carry a listed output with no atom.
    #[test]
    fn the_histograms_hold_the_atoms() {
        let listed = vec![vec![([0.; 3], [1., 0., 0.]), ([0.; 3], [-3., 0., 0.]), ([0.; 3], [0.2, 0., 0.])], vec![]];
        let d = histograms(&listed, &[0, 4], &|x| x.1[0]);
        assert_eq!(d.grid.len(), 3);
        assert_eq!(d.grid[1], 0.);
        assert_eq!(d.probabilities[1], d.probabilities[0]);
        assert!((d.probabilities[0].iter().sum::<f64>() - 1.).abs() < 1e-15);
    }

    /// What the messages pass on (`Messages::result`): a result of two groups at least -- of one member each as well as
    /// of eight -- and of no fewer trajectories than the attempt's last; any other is not made, and the last one goes out
    /// again (nothing where there is none). With no group in there are no estimates to make one from (`so_far`).
    #[test]
    fn a_result_goes_out_from_two_groups_on() {
        let context = Context {
            plan: Plan::new(&[0., 1.], [1.; 3]),
            tau: 0.5,
            scales: Scales { hbar_over_mass: 1., frequencies: [1.; 3] },
            initial: [0.; STATES],
            momentum: [1.; 3],
            fixed: Fixed { model: vec![[0.; 2]; 2], start: 0. },
            step: vec![Bound::share(); 2],
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let shape = Shape::of(&context.plan, false);
        let record = |i: u64| {
            let held = Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3] };
            Record { outputs: vec![Fate::Held(held.clone()), Fate::Held(held)], listed: vec![Some(([0.; 3], [i as f64; 3])); shape.listed], samples: vec![], counts: Counts::default() }
        };
        let whole = [shape.fold((0..GROUP).map(record)), shape.fold((GROUP..2 * GROUP).map(record))];
        let scout: Vec<Part> = (0..3).map(|i| shape.fold(std::iter::once(record(i)))).collect();
        let mut sent = vec![];
        let mut send = |s: &Solution, _: Option<f64>| sent.push(s.trajectories);
        let mut messages = Messages { send: &mut send, clock: Instant::now() - CHECKPOINT, last: None, least: 0 };
        // No group, one group: nothing, and the message stays due. Three groups of one member: a result, and the clock is
        // set. Two of them: fewer trajectories, the last again. Two whole groups, in hand or coming in: results.
        so_far(&context, &[], &[], &mut messages);
        so_far(&context, &whole[..1], &[], &mut messages);
        assert!(messages.due().is_zero());
        so_far(&context, &scout, &[], &mut messages);
        assert!(!messages.due().is_zero());
        so_far(&context, &scout[..2], &[], &mut messages);
        so_far(&context, &whole, &[], &mut messages);
        so_far(&context, &whole[..1], &[&whole[1]], &mut messages);
        messages.result(1, 4 * GROUP, None, || unreachable!("no result is made of one group"));
        assert_eq!(sent, [3, 3, 2 * GROUP, 2 * GROUP, 2 * GROUP]);
    }

    /// A message that falls due inside a hand-out is made while the trajectories run: its loops run on the pool
    /// (`pooled`), whose threads take them between two trajectories (`run_groups`) -- not at the list's end, when all
    /// but the last few trajectories are over. Eight trajectories a thread, each a few milliseconds of waiting; a
    /// message due from the start, made by a loop on the pool (of two items: one alone runs where it is called) that
    /// counts the trajectories over by then: not half of them (all but a thread's worth where the threads keep to
    /// their trajectories: 122 of 128 on 16 threads).
    #[test]
    fn a_message_due_is_made_while_the_trajectories_run() {
        let shape = Shape { outputs: 1, listed: 0, keep: false };
        let over = AtomicU64::new(0);
        let record = |_: u64| {
            std::thread::sleep(Duration::from_millis(3));
            over.fetch_add(1, Ordering::Relaxed);
            Record { outputs: vec![Fate::Lost], listed: vec![], samples: vec![], counts: Counts::default() }
        };
        let mut silent = |_: &Solution, _: Option<f64>| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now() - CHECKPOINT, last: None, least: 0 };
        let numbers: Vec<u64> = (0..rayon::current_num_threads() as u64).collect();
        let mut seen = vec![];
        let levels = run_groups(&[Hand { groups: &numbers, members: 0..GROUP, shape, record: &record }], &mut messages, &mut |_, messages| {
            seen.push((0..2).into_par_iter().map(|_| over.load(Ordering::Relaxed)).max().expect("the loop's two items"));
            messages.again();
        });
        assert_eq!(levels[0].len(), numbers.len());
        assert!(2 * seen[0] < numbers.len() as u64 * GROUP, "{seen:?} of {}", numbers.len() as u64 * GROUP);
        // Nothing was sent (there is no last result): the message stayed due, and was asked for again as the groups came
        // in, once a group at the most.
        assert!(messages.due().is_zero() && seen.len() > 1 && seen.len() <= numbers.len(), "{seen:?}");
    }

    /// The groups come back in their order whatever the threads do: 37 groups' records (five of a pilot's numbers, then a
    /// product's) -- each trajectory's own, its rows marking its number -- folded by `run_groups` equal the same records
    /// folded one after another; so do the levels of one hand-out, each with its own members of the groups.
    #[test]
    fn the_groups_come_back_in_order() {
        let shape = Shape { outputs: 2, listed: 1, keep: false };
        let record = |i: u64| {
            let mut rows = [0.; STATES];
            rows[(i % STATES as u64) as usize] = 1.;
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3] };
            let outputs = vec![Fate::Held(held.clone()), if i % 3 > 0 { Fate::Held(held) } else { Fate::Lost }];
            Record { outputs, listed: vec![Some(([i as f64; 3], [0.; 3]))], samples: vec![], counts: Counts { steps: i, ..Counts::default() } }
        };
        let mut silent = |_: &Solution, _: Option<f64>| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0 };
        let numbers: Vec<u64> = (2048..2053).chain(5..37).collect();
        let members = [0..GROUP, 1..3];
        let hands = members.clone().map(|members| Hand { groups: &numbers, members, shape, record: &record });
        let levels = run_groups(&hands, &mut messages, &mut |_, _| {});
        assert_eq!(levels.len(), 2);
        for (parts, members) in levels.iter().zip(members) {
            assert_eq!(parts.len(), 37);
            for (part, number) in parts.iter().zip(&numbers) {
                let one = shape.fold(members.clone().map(|m| record(number * GROUP + m)));
                assert_eq!(part.sums, one.sums);
                assert_eq!(part.listed, one.listed);
                assert_eq!(part.counts.steps, one.counts.steps);
            }
        }
    }

    /// What the reach leaves out of a tweezer's ensemble by quadrature (model/reach.rs `Model::left_out`, ensemble.rs
    /// `Ensemble::beyond`) is what a count over its drawn atoms finds -- drawn here alone, for the test. R4_TW2 (one
    /// component, 30 µK, its tweezer 35 k_BT deep), the reach at the request's temperature: of the initial atoms of the
    /// most trajectories a solve runs, the share outside the reach (`Model::holds`) and their share of Σκ² are the
    /// quadrature's within z of the count's own deviations (the atoms' binomial; the energy's ratio linearized) and the
    /// quadrature's difference between its last two levels.
    #[test]
    fn the_reach_s_share_by_quadrature_is_the_count_of_its_drawn_atoms() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R4_TW2.json");
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let model = Model::new(&request, &light, &[]).unwrap();
        let initial = components(&request);
        assert_eq!(initial.len(), 1, "one initial component");
        let reference = alone(initial[0].0);
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, &light, alone(s)).unwrap()).collect();
        let hm = hbar_over_mass(request.mass_kg);
        let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
        let prepared = model.prepared();
        let [atoms, energy] = ensemble.beyond(&|r| model.room(&reach, r, &prepared), MOST_TRAJECTORIES);
        let booked = model.left_out(&reach, &ensemble, MOST_TRAJECTORIES);
        assert_eq!(booked, [atoms.value + atoms.difference, energy.value + energy.difference], "the booked shares are the quadrature's with its differences");
        let n = MOST_TRAJECTORIES;
        let read: Vec<(f64, bool)> = (0..n).into_par_iter().map(|index| { let start = ensemble.draw(SEED, index); (start.kappa.iter().map(|k| k * k).sum(), !model.holds(&reach, start.r, start.kappa)) }).collect();
        let count = read.iter().filter(|atom| atom.1).count() as f64;
        let (all, outside): (f64, f64) = read.iter().fold((0., 0.), |(all, outside), &(e, out)| (all + e, if out { outside + e } else { outside }));
        let ratio = outside / all;
        let spread: f64 = read.iter().map(|&(e, out)| (if out { e } else { 0. } - ratio * e).powi(2)).sum::<f64>().sqrt() / all;
        eprintln!("outside the reach: {count} of {n} atoms drawn ({:.3e}) against the quadrature's {:.3e} ± {:.1e}; their kinetic energy's share {ratio:.3e} ± {spread:.1e} against {:.3e} ± {:.1e}", count / n as f64, atoms.value, atoms.difference, energy.value, energy.difference);
        assert!(count > 0., "some atoms are drawn outside the reach");
        assert!((count / n as f64 - atoms.value).abs() <= CONFIDENCE * count.sqrt() / n as f64 + atoms.difference, "the atoms' share: {} drawn of {n} against {:e} ± {:e}", count, atoms.value, atoms.difference);
        assert!((ratio - energy.value).abs() <= CONFIDENCE * spread + energy.difference, "the kinetic energy's share: {ratio:e} ± {spread:e} drawn against {:e} ± {:e}", energy.value, energy.difference);
    }

    /// The initial ensembles of the conservative reference traps (refs/scenes R1, the app's own requests: a lone tweezer, a
    /// standing wave, crossed beams of one frequency, a σ⁺ tweezer) in the model's prepared wells under the model's proven
    /// envelopes: 2¹² draws each read the exact moments (ħ/m)⟨κ_a²⟩ = Σ_b p_b η_b k_BT (S_b²)_aa per lab axis within their
    /// intervals over the groups at the confidence z -- twelve looks, three axes of four traps -- and the quadrature's own
    /// error as the solve books it for T(0), its last two levels' relative difference (`Fixed::start`), here on 2¹⁶
    /// evaluations, fewer than a pilot's steps in these traps (88 trajectories of 800 steps or more). η lies in (0, 1]:
    /// the lowered isothermal's kinetic temperature, below the Boltzmann one.
    #[test]
    fn the_reference_traps_draw_their_exact_moments() {
        let names = ["R1_TWEEZER", "R1_RETRO", "R1_CROSSED", "R1_SIGMA_P"];
        for name in names {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
            let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let request = Request::from_native(&line).unwrap();
            let light = Light::new(&request);
            let model = Model::new(&request, &light, &[]).unwrap();
            let initial = components(&request);
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, &light, alone(s)).unwrap()).collect();
            let hm = hbar_over_mass(request.mass_kg);
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
            let kinetic = ensemble.kinetic(1 << 16);
            assert!(kinetic.iter().all(|k| k.eta > 0. && k.eta <= 1.), "{name}: η {kinetic:?}");
            let exact = ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>());
            let quadrature = kinetic.iter().map(|k| k.difference / k.eta).fold(0., f64::max);
            let draws: Vec<V3> = (0..1 << 12).map(|i| ensemble.draw(SEED, i).kappa.map(|k| hm * k * k)).collect();
            for a in 0..3 {
                let groups: Vec<f64> = draws.chunks(GROUP as usize).map(|members| members.iter().map(|d| d[a]).sum::<f64>() / GROUP as f64).collect();
                let g = groups.len() as f64;
                let mean = groups.iter().sum::<f64>() / g;
                let error = (groups.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (g - 1.) / g).sqrt();
                let within = Spread { error, groups: g }.interval_over((3 * names.len()) as f64) + quadrature * exact[a];
                assert!((mean - exact[a]).abs() <= within, "{name} axis {a}: {mean} against {} ± {within} (quadrature {quadrature:e})", exact[a]);
            }
        }
    }

}
