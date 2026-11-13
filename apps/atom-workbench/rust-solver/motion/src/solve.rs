//! The pipeline: request, reach and model, pilot, decisions, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the light; the initial components; the model with the units kept so far and those whose
//!    folds do not exist where the atoms start (`closed`; # The model's prior) -- which folds hold beyond that is
//!    measured at the pilot's states (# The model) --; the reach (model/reach.rs); the trap the
//!    result reports, the outputs' plan, the initial ensemble and
//!    what the reach leaves out of it; the near light's tables over the reach (SPEC §2.3). What of
//!    these rests on the model alone is made once for a model and serves every attempt at it; what rests on the
//!    reach or on the step is an attempt's own (`solve`, `Setting`).
//! 2. The scout: the first member of every pilot group, at the attempt's first step and at twice it (# The step),
//!    screened. Where a folded unit is to be kept at its states (`keep`), the attempt ends there and the next begins
//!    with that model: a model is settled for the price of a few trajectories, not of a pilot.
//!    The pilot: the scout's groups whole, `pilot_size` trajectories at the run's two levels (the scout's are its
//!    groups' first members and are not run again). Its groups have numbers of their own, from trajectory
//!    `MOST_TRAJECTORIES` on: beyond every number a group of the product takes, which is its place in the run's list
//!    (`product`). It checks the step -- on two levels and a third above where it is cheap, and on further pairs where
//!    they pay (# The step) --, reads the reach's falsifier (`outgrown`: where its atoms have outgrown
//!    the reach the attempt ends, and the next begins with the reach of the temperature read; on the pilot's atoms --
//!    a reach is grown to a temperature, which the scout's eleven atoms tell to √(2/11), and the work goes as its
//!    root), the model's units' errors at every state of its run level at the analysis times, and the tables' errors at
//!    one state of every trajectory there (SPEC §5.4; `one_each`) -- every read and decision at the analysis times, not
//!    at the outputs (# The analysis times). Its trajectories at the run's step stay in the result: they are
//!    trajectories of the run's own model and step.
//!    The falsifier reads the run's own level alone, so the pilot's own level is run before its companion and read as
//!    it comes in (`own_level`): where its atoms refute the reach the attempt ends there, and the rest of the pilot --
//!    its companion, the step's check at a model the next attempt judges anew over the grown reach -- is never run
//!    (`attempt`); so is a rung below's own level read, where the check goes down (`check`). The own level is run after
//!    the scout's keep: a model whose units' errors leave a look no room is no model of the atoms, and their T is that
//!    model's. Where the scout's atoms already refute the reach, its members double a hand-out (`batch`), and the reach
//!    is read after each. The scout's refutation is the read that selected the attempt for ending; its eleven atoms,
//!    one a group, are confirmed by the next eleven, read with them, before the reach grows to the temperature they read
//!    (a reach grown on the scout's own read, R3_DOPPLER, over-read its atoms' T by up to 2.4 times and cost 16 % more
//!    work). What the falsifier needs is significance, which the doubled atoms give where it stands; what more atoms
//!    would buy is the grown temperature read closer, and a reach read low is refuted again by the next pilot's atoms,
//!    read high costs its own pilot -- both the next attempt's to tell. (Measured, S3 at 30 µK: on 22 atoms the reach
//!    grows to 99.7 µK, on 44 to 95.7, on 88 to 113.0, none refuted by the next pilot's 88; 957, 949 and 1128 CPU-s in
//!    all, the same result.) What an attempt so ended leaves the next is the reach's temperature and the states of the
//!    run's level (# An attempt's own).
//! 3. The model's decisions (SPEC §5.2; # The model), at the scout's states and again at the pilot's -- at its first rung
//!    whose own read is in and not clearly beyond, before its check goes on (# Each decision as its evidence is in) --,
//!    the step's error as far as it is read (# The model: the step's part): units to keep end the attempt, and the next
//!    begins with their model, deciding first on the states this one's run level left (# An attempt's own).
//! 4. The product: batches of whole groups at the settled step, one level, every trajectory run to the run's end and
//!    read at the pilot's analysis times beside the outputs, until every held quantity's total lies within its target
//!    there (# Stopping) or the trajectories reach
//!    `MOST_TRAJECTORIES`. Its atoms that leave the reach are read where they are, and what they carry is booked
//!    (# The reach's atoms); where that leaves a look no room, the reach grows and the next attempt begins.
//! 5. The result (contract §2.2, `assemble`).
//!
//! # An attempt's own
//! What an attempt's end leaves valid of it, by one rule for every end: a reach outgrown (`Attempt::Grow`) or units
//! kept (`Attempt::Keep`), the attempt hands the next the states of its run level (`Carried`: the scout's, the pilot's
//! own level as far as it ran, or the pilot's), and the next attempt decides on them first, before it runs a
//! trajectory -- as the old engine carried its pilot's states into the grown reach's choice. The model's own set-up
//! (`Setting`) serves every attempt at the model while no unit is kept. A grown reach changes, for the same model: what
//! the reach leaves out of the ensemble, the kick's curvature and the first rung (the reach's largest speed, and its
//! temperature the prior's), the tables (refined over the reach's ranges: other nodes, another interpolant -- part of
//! the generator a trajectory is integrated with) and the Brownian path's cell (the first rung's companion). A kept
//! unit changes the model at every state -- a kept unit is explicit everywhere --, and every read of the pilot is the
//! old model's: its trajectories, the step's reads (the old generator's error), the units' reads, the tables (other
//! clusters), its statistics, and the ensemble's draw (the prepared potential holds the folded drives' shifts alone).
//! What stays is where the atoms went: their states, carried into the next model where it keeps more (`Carried::into`;
//! model/readout.rs `Model::carried`: what this model held as it was, in the next one's frames, and an excited state
//! made explicit at what the fold made of it). They are read for a decision and booked nowhere: the result's errors are
//! read on its own pilot's states.
//!
//! So a keep is a fixed point on the states it was read on: the units the reads ask are kept, the kept model reads the
//! same atoms again -- the units it still folds, whose errors a keep changes beside them (a kept line read through a
//! folded drive's self-energy, model/eliminate.rs `Model::pair_error`; a state's rates shared with an explicit line) --,
//! and keeps again where they still leave a look no room, before its own atoms run. (Kept at once on the assumption
//! that the other units' errors stand, the 1D PGC example's next pilot read them 6 to 30 times larger -- [18] 0.0006 →
//! 0.0070 --, and it kept again, a pilot each.) Where the carried states ask nothing they stand for the scout's keep,
//! and the attempt runs its pilot. Nothing of the step stays: every attempt's ladder begins at its own first rung and
//! goes down on its own reads alone (# The step) -- a descent read on another model or another reach is that model's or
//! reach's. (S3c over 1 ms: its 30 µK reach's companion at 1.43 µs left the domain and the check went down to 0.358 µs,
//! which the next attempt, at 65 µK and a first rung of 0.971 µs, began at, twice the product's steps.) No pilot's
//! trajectories carry over, so none is made to: a pilot on a refuted reach is not spent instead (# The pipeline: its
//! own level, read before its companion).
//!
//! # Each decision as its evidence is in
//! A decision that ends an attempt is taken where what it reads is in, not after the work it would make useless. The
//! pilot's keep reads the units' errors and the tables' at the states of the pilot's run level, the optical sector's
//! booking, the pilot's statistics, and the step's error as the rung's own reads hold it (# The model: the step's
//! part) -- not the rest of the step's check: what the check does past the rung's own two levels, where it stands,
//! leaves the step within its share (the reads are the ones made: they are kept, and the keep is not asked again);
//! where the check goes down a rung instead, its states are the same trajectories at half the step, read again, and
//! the keep asked again on them. So the pilot runs as far as the first rung whose own read is in and not clearly beyond
//! (`Settled::Pending`), the keep is read there -- the reach's falsifier before it, on the pilot's own level as it came
//! in (`own_level`) --, and only an attempt they leave standing pays for the rest of its check -- the third level at
//! 4τ, pairs, a rung below. (Read
//! after the whole check, S3's probe kept sixteen units after its check had gone down to 0.187 µs -- the third level,
//! the rung below, 127 CPU-s of a model that was then replaced --, and its next attempt began at 0.187 µs.) The check's
//! levels stay on the tables they were run on; a table its reads refine is the product's (`PilotRead::refined`).
//! What a keep or a growth leaves valid of the attempt is said in # An attempt's own.
//!
//! # The step (SPEC §2.2)
//! One step τ for the run (adapt.rs), which follows from what the scheme must resolve and from nothing of the outputs:
//! an output is read where it falls, inside a step (adapt.rs, # The reads), so outputs a tenth of a step apart take no
//! shorter a step than outputs a hundred steps apart. The ladder's rungs are the run's length (`Plan::end`) over the
//! even numbers, and below a rung its halves: the companion's step is twice the run's, and the run is a whole number of
//! the companion's steps, so no level's last step is a cut one. A step is
//! admissible where its nodes resolve what the scheme samples along an atom's path (`admissible`): the reference flow
//! with the kick between its halves, whose linear map about a minimum is bounded below its first edge, cos ωτ −
//! (ω_k²τ²/2) sin(ωτ)/(ωτ) > −1 at the prepared trap's largest frequency ω and the kick's largest curvature ω_k² as
//! far as the model bounds it (model/reach.rs `Model::kick_curvature`: the states' own traps and the bare folds'
//! fringes; the wells of the light folded at the atom and of explicit states are the check's to find) -- ω τ < π for
//! a kick without curvature, ω_k τ < 2 on a flat reference --; and the generator, whose change along the path the propagation
//! reads at nodes half a step apart, (|Ω| + |K| v) τ/2 < π for every residual beat Ω and wave vector K of the model's
//! entries at an atom's speed v. The ladder begins at the largest rung within the physics prior's step that is
//! admissible for the run and whose double is for the companion (`first_rung`): the run's step at the largest speed of
//! the reach the model is judged over; the companion's at the reach's rms speed, below the same edge; and the prior,
//! the old engine's first step (`prior`): T's step error goes as the square of the phase k_max v_T τ the light's
//! pattern advances in a step at the reach's temperature -- the request's until the atoms grow it, one temperature
//! for every speed the first rung is taken at --, and the step may take its share of T's allowance -- so the
//! ladder does not begin at an edge's rung the check then leaves (PGC3D: 1.425 µs, left on the pilot for 0.7125 µs,
//! 11.2 CPU-s). Every attempt begins at its own first rung, and goes down only on its own τ/2τ reads (# An attempt's
//! own). The first rung is a rung of the ladder, whose bottom is the step of `MOST_STEPS` steps a trajectory: a
//! request for which no rung down to it is admissible is refused with a message.
//!
//! The step's convergence is checked on the pilot, as a refinement is checked by hand: the pilot's trajectories run at
//! τ and, the same trajectory numbers -- the same initial atoms, the same jumps' draws by their number, the same
//! Brownian path (adapt.rs) --, at 2τ (`Levels`), both read at the same instants: the analysis times' candidates, the
//! companion's step ends (# The analysis times), where both levels' states are their steps' own ends. Of two levels at
//! h and 2h a quantity's difference at a time is d(h) = Q(2h) − Q(h); a read (estimate.rs `between`) takes
//! d/(2^p − 1) over the run -- its mean square M over the run, by the trapezoid over the two levels' own analysis times
//! (`read_step`: each time weighed by half its neighbours' interval), estimated without the noise's floor, with the
//! delete-one-group jackknife of that estimate --: one read for the loss, for each row as shown and for the total T,
//! each carried by the dynamics itself. An error that changes its sign in time counts by its size. (Over the outputs,
//! as it was, the read changed with their number: a mean over 100 or over 1000 of them.)
//!
//! ## Two levels, three levels
//! A difference is an error only by what is known of the coarser level. With e(h) = Q(h) − Q(0), d(h) = e(2h) − e(h),
//! and d(h)/(2^p − 1) is e(h) where the order holds between the two levels, e(2h) = 2^p e(h). Nothing in d tests that:
//! - two levels that are both too coarse agree. Measured: e038 (S3-like, 0.38 mW, 3 G) at 0.589 µs read within the
//!   share against 1.178 µs -- the companion under-heats with it --, and its T(1 ms) is 79.1 ± 2.8 µK against the
//!   converged 89.2 ± 2.6 (τ 0.0895 and 0.179 agree): −11.3 %, outside the target. A two-level atom in a resonant
//!   standing wave at Ω₀ = 10Γ (mass ×100, held at an antinode; the reach grown to 309 µK) was taken at 1.30 µs, a
//!   seventeenth of the beat period π/(kv) at the reach's rms speed, and its T(200 µs) is 29.8 against the 23.2 µK of
//!   0.162 µs (+28 %);
//! - a read of other levels is no verdict for these. Measured: m20 (S3 at −20 MHz) at 0.392 µs read its loss against
//!   0.784 µs at an estimate of 1.95 times the share's square (far end 7.7), stood on the rung above's read lowered by
//!   the order, booked its clear part, none, and its loss at 600 µs is 0.587 against the converged 0.724 (−5.2σ). Its
//!   own read alone does not catch it: on 352 pairs it reads 0.69 b, within -- the two levels agree well enough --,
//!   where the rung above's read, lowered to it, is 2.0 b: the order is not shown, and the check goes down (to
//!   0.098 µs, the loss at 600 µs 0.717).
//! The order is a property of three levels: e ∝ h^p across them makes the successive differences shrink by 2^p at
//! every output, d(2h) = 2^p d(h). It is the observed-order test that grid-convergence studies hold Richardson's
//! extrapolation to (Roache's grid convergence index takes a level's error from two grids only with a safety factor
//! of three, from three whose observed order is the scheme's with 1.25), and the guard of multilevel Monte Carlo
//! (Giles), which takes the larger of the finest difference and the next one lowered by the order, so that a
//! difference small by chance or by two coarse levels' agreement is not taken for convergence.
//!
//! A rung τ is read by its own pair (τ, 2τ) and, where it is cheap, by a third level above, (2τ, 4τ), its read carried
//! to τ by the order (`carried`) -- lowered 2^{2p} times in the mean square, noise and all. The triple is (4τ, 2τ, τ), τ
//! its finest step. The third level is the rung left above where the check came down from it (a level the ladder ran),
//! else the pilot's trajectories at 4τ, a quarter of a trajectory each (`above`: the ladder has the rung above). Where
//! the order holds across the triple both reads are of one mean square, e(τ)'s. A rung stands (`verdict`) where both
//! estimates are within the share, M̂ ≤ b on every look (`within`) -- the larger of the two, as multilevel Monte Carlo
//! takes the larger of its finest difference and the next one lowered by the order, so that two levels that agree by
//! being both too coarse are not taken for converged where the third shows the error --, and it is left where the
//! third's estimate is beyond: the order is not shown above it. Where no cheap third level is to be had -- a first rung
//! at the ladder's top, a rung above whose paths left the model's domain or whose read is no number --, the rung stands
//! on its own two levels: the change between levels within the tolerance on the estimates, as an adaptive integrator's
//! local control and a refinement checked by hand take it (the user's approximate convergence, QuTiP-like: delivered
//! fast, the risk of two coarse levels that agree accepted there). No level below a rung is run to read it: τ/2 is a
//! level of the run only where the ladder takes it. (Read as the middle of a triple below, (2τ, τ, τ/2), the third
//! level took the pilot again at twice its steps -- S3 113 to 133 CPU-s an attempt, PGC3D 12 of its 55 --, and on 88
//! pairs its read, raised 16 times, was a lottery: a clean rung refused about two times in five.) The rung books what it
//! stands by: per look the larger estimate, √max(M̂, 0) (`checked`, `worse`) -- at most the share, and the number the
//! criterion held. What three levels cannot tell: an error the same at all three -- the scheme saturated across a factor
//! four of the step. Where every difference is below the noise no number of levels tells it from convergence.
//!
//! ## The ladder
//! - Clearly beyond: a rung whose own read lies clearly beyond the square b of the step's share among the family's few
//!   looks (`screen`) is left: the step is halved, the level at τ becomes the companion and only τ/2 is run anew; the
//!   rung below's third level above is the rung left's own read, lowered. Down to the step at which a trajectory would
//!   take `MOST_STEPS` steps, where the step stands as it is, flagged, and books what its reads do not exclude.
//! - The scout's eleven pairs leave a rung so, and certify nothing: a rung they leave standing is the pilot's to read.
//! - On the pilot (all the members of a group in: the last word on the step), a rung not clearly beyond (`verdict`):
//!   - its own estimate within: where it has no third level on the pilot's pairs and the ladder has the rung above, the
//!     third level above; it stands where that one's estimate is within, and is left where it is beyond; with no third
//!     level to be had, it stands on its own two;
//!   - its own estimate beyond the share and not clearly -- the noise hides the difference --: more pairs, where they
//!     could bring it within at the best (`remedy`), the product's next groups at both levels, read again after every
//!     batch. A rung below is no remedy for the noise: it reads its own pair's, and where the pairs part early that is
//!     the two ensembles' own, the same at every step (estimate.rs, # The step's error; measured, R4_LIN_2: rows at far
//!     ends of 1.5 to 2.5 b at 0.43, 0.22, 0.11 and 0.054 µs alike). Where pairs cannot see it either, its estimate as
//!     it reads is beyond the share, and the rung is left, as an adaptive integrator leaves a step whose error estimate
//!     is beyond its tolerance. (Stood unable to see at TET's ladder top, 1.40 µs, a row's far end booked 3.8 % of the
//!     5 % and left the statistics a room of 0.9 %: 4272 trajectories, 211 CPU-s against 39.) At the ladder's bottom
//!     the check ends unable to see (SPEC §5.1): the step stands, flagged, and books what its reads do not exclude.
//! - A read is never carried from one rung to another as a verdict: a rung's third level is a read of levels it holds
//!   itself, of the same model at the same steps -- the scout's, where the pilot has none of its own and the ladder
//!   no rung above --, and it certifies nothing alone. Nor from one attempt to the next: every attempt's ladder is its
//!   own, from its own first rung (# An attempt's own).
//! - A comparison of two levels says something only where the coarser one still resolves what the scheme samples: the
//!   first rung's double is admissible for the typical atom, and stable for every atom. The companion's atoms beyond
//!   its light's limit weigh in the difference by their share (`Domain::companion`): no level leaves the comparison.
//!
//! What enters a quantity's account, at every time alike, is what the reads the rung stands by say (`checked`); it
//! is fixed after the pilot, and the product runs one level and reads no step. The check's resolution -- √(t·SE) of that
//! read, the rms error the pairs cannot tell from none -- is said in the result's method line beside the steps compared
//! and the number of pairs (`Checked`).
//!
//! It is a check of the order, not a bound: an error confined to a short stretch of the run is diluted by the run's
//! length in the mean square. A bound per output was built and measured (the product at
//! both levels, a60231dd): the pairs part within 100 µs in six beams' cooling whatever the noise they share, so the
//! difference's error at the late outputs is two independent ensembles', and the measurement cost 30 to 240 % of a run
//! on top of it; pooling it over the outputs books less than the truth where the error grows in time (SPEC §2.2).
//!
//! # Outputs and windows
//! The outputs are the request's times, from 0 and evenly spaced: what is shown, and nothing the solve decides on
//! (# The analysis times). At an output a trajectory is read inside the step the output falls in (`run`; step.rs
//! `Step::at`): its rows as shown, its (r, κ) for the listed distributions, and its window's T sums -- cheap sums, no
//! state kept and nothing read of the model. T's window about an output (estimate.rs `half_window`: one period of the
//! axis, cut at t = 0) takes a trajectory's κ and κ² over the cuts of its steps' paths (step.rs `Segment::moments`: the
//! path with the kick spread over the step, of the scheme's order at any cut); the first is the instant t = 0, and the
//! run goes on through the last output's window (`Plan`). Outputs closer than a period have overlapping windows, so a
//! window is not summed by itself: per lab axis the trajectory's running integrals of 1, κ and κ² from t = 0 are kept
//! at every window's two ends, and a window is their difference (`run`) -- each step's path integrated once, and once
//! more up to each end that falls inside it, whatever the windows' overlap.
//!
//! The times the decisions read (# The analysis times) are read the same way, their windows cut at the run's end as
//! well, so that they stay centred (the trajectories run to the run's end and no further); a time that is an output
//! too is read once.
//!
//! The output at t = 0 is read as every other one is: T, n̄ and the mean velocity there are the estimates of the
//! trajectories' initial atoms (their instant κ: the window at t = 0 has no length) -- the same estimator on the same
//! atoms as at the outputs after it. An output's T is a function of its held atoms' κ over its window; as t → 0 the
//! window shrinks to the instant and each κ(t) = κ(0) + O(t) along the flow, so T(t) tends to T(0) on the same atoms:
//! T is continuous at t = 0, and what it does by the first output is the dynamics' alone. Its error there is its
//! statistics' (no step taken, no model acted: `accounts`). What is shown at t = 0 is the request's state, exact:
//! every trajectory starts in its component's state, and the readout's dressing by the lights -- the eliminated
//! excited amplitudes following the ground state (model/readout.rs) -- has not happened at the instant they are
//! switched on. (T(0) was the ensemble's moment by quadrature (ensemble.rs, now the tests' reference for the draws),
//! with the pilot's steps for its budget: another estimator, and a short run's budget leaves it unconverged -- the
//! user's 200 µK tweezer of 7.5 k_BT, 0.1 µs: two levels of 2 and 4 nodes a coordinate 22.5 % apart, 179.5 µK against
//! the converged 171.08 µK (2¹⁵ draws: 170.8 ± 0.5), and the run's 144 atoms read 164.5 µK 10⁻⁴ µs later.)
//!
//! # The analysis times
//! Every read and decision of the solve stands on the analysis times T_a (analysis.rs; SPEC §3.1), whatever the
//! outputs: the reads and their families are of a number that the readings' own change in time asks, not of the output
//! grid (as it was: the reads at every output cost in proportion to their number -- the pgc3d example 5 s at 100
//! outputs, 25 s at 1000; a run of 0.1 µs at 1000 outputs, one step, 69.9 s --, and a keep was the largest of eleven
//! states' point estimates over as many outputs as there were, decided anew by their number: 0, 11 and 7 units kept at
//! 100, 199 and 1000; r_samples.status.md).
//! - The candidates are the companion's step ends at the attempt's first rung, m·2τ₀ to the run's end (`grid`): a
//!   step end of every level the attempt runs -- its rungs and their companions; the third level at 4τ reads half of
//!   them inside its steps (adapt.rs, # The reads). Every level of the pilot is read at every candidate (`Plan::reads`;
//!   the run level keeps its states there), the product at the analysis times alone.
//! - The analysis times of some levels are the halving's of their readings (`analysed`; analysis.rs, # The rule): from
//!   the run's two ends, an interval is halved while a reading at its middle, its quarters -- and, where it is long, the
//!   nested middles below them (next) -- lies off the line through its ends beyond its own jackknife interval among the
//!   interval's looks and the line's rounding. The
//!   readings are what the decisions read of the levels (estimate.rs `readings`): the run level's loss, rows as shown
//!   and T per lab axis, and, where a read compares two levels, the step's error of each of them. The step's check
//!   stands on its two levels' times (`read_step`); the reach's falsifier, the model's decisions and the product on the
//!   run level's.
//! - The physical scale bounds the reads' spacing, not the intervals: the ends are the run's two ends alone -- the
//!   lights are switched on at t = 0 and stay so: the request has no other switching time --, and an interval is read
//!   at points no farther apart than a quarter of T's shortest window, one period of the fastest lab axis
//!   (`Plan::window`), its nested middles (analysis.rs: a half's reads its parent's, a feature found stays found). T at
//!   a time is its windows' mean (estimate.rs `half_window`), so no change of it is narrower than the shortest window,
//!   and reads a quarter of it apart meet every such change wherever it lies -- none is invisible to them. A feature
//!   narrower than that, away from the lights' switching, is not promised (a hypothesis: between the switches the
//!   readings follow the atoms' motion, the trap's periods its scale and the fastest axis's the shortest; the internal
//!   state's own transients begin at a switch, an end). No number of the times is chosen: the window is T's own, and a
//!   read is no end -- an interval whose reads lie on its line stands whole and adds no time. (First built with the
//!   intervals bounded instead, by the slowest axis's window, every end a time: PGC3D 1 ms read at 26 analysis times,
//!   2200 rows of the units, and took 1.23 times 717cb386's CPU over three alternating pairs.)
//! - The units' errors are no reading: a state's read costs about a millisecond (PGC3D reads 1185 units), and the
//!   halving reads three times as many times as it keeps. A unit's error at a state is a function of its place, speed
//!   and internal state, whose ensemble the rows and T read (a hypothesis, not tested by itself: an error of the
//!   model's narrower in time than those is stepped over).
//! - What stands on them: the step's mean square (estimate.rs `between`, each time by its share of the run); every
//!   state of the pilot's run level there, each group's every member held and every atom lost by then (`sampled`), the
//!   model's error and its families -- the looks', `crowded`, `tight`, `needed`, `Doubt`'s (two looks a time), the
//!   needs `Level` and `Pricing` read --; the reach's falsifier (`outgrown`: a T per lab axis and a count outside at
//!   every analysis time) and the temperature it grows to; the product's looks, batches, keeps and atoms apart; the
//!   tables' reads and the prices at one state of every trajectory among them (`one_each`). States carried to the next
//!   attempt are at their own analysis times (`Carried`), and the next attempt decides there.
//! - What the result shows at an output: its own estimates and their standard errors; its accounts -- the model's
//!   error, the step's part, the statistics and T's allowance -- on the line through the accounts at the analysis
//!   times (`assemble`, `accounts_along`), so that the ledger flags where the stop reads, whatever the outputs. (Judged
//!   on its own statistics, a noisy estimate of the same spread, the pgc3d example stopped at the analysis times was
//!   flagged at 11 of 1000 outputs, at most 5.4 % with 5.1 % of statistics, and 2 of 100, its reason "not finished".)
//!
//! # The model (SPEC §5.2)
//! The units are the model's (model.rs), listed again in every model a keep closes. Every model the solve runs is
//! closed first by its prior (# The model's prior): the folds that do not exist where the atoms start are kept before a
//! trajectory runs. Which folds hold beyond that is measured, as an adaptive integrator's tolerance is (QuTiP's way): a
//! fold that does not hold where the atoms went reads its error there -- its own small parameter at the state. A keep
//! is one comparison of the units' measured errors against what keeping them costs, over the whole set the reads give
//! (`keep`), made once and never undone: the kept set only grows, and every keep is a new attempt, a new model and its
//! own pilot.
//!
//! ## The model's prior (SPEC §1.5)
//! The old engine's first rung: every fold that exists is folded (Stewart, S = 2m² < ½, and the sweep a < ½), every
//! other unit kept -- a drive taken up its levels (bare, at the atom, explicit) as far as the first whose fold exists
//! (`closed`, `failing`, model/units.rs `Model::failing`). Where a fold exists is read where the atoms are, at points,
//! not proven over the reach: at the reach's own atoms, the draws of its thermal ensemble at its temperature T_r with
//! the run's seed, as many as the pilot runs (`atoms`) -- at the request's temperature the pilot's own initial atoms --,
//! each at its place and its speed, as a state's errors are read (model/units.rs `Measured`). A gate bounded below ½
//! over the whole space at the atoms' largest speed, every beam at its peak, holds at every one of them and is not read
//! again; the others are read at each atom, a value there: the fold does not exist where it reaches ½ at one of them.
//! The units that fail are kept, and the model they make read again, until none fails. Every model is closed so before
//! its first attempt, and again where its reach grows (`Attempt::Grow`): a grown reach is another ensemble's atoms, and
//! what fails among them is kept as an attempt's keep is, with the states the attempt left (# An attempt's own).
//! - The points are no number of the prior's own: the thermal ensemble's positions and speeds in a conservative trap are
//!   where and how the atoms are, and what only a rare fast atom meets is the measured errors' to answer -- the reads
//!   confirm the prior, or keep beside it (# The model). Their count is the pilot's; the prior costs a gate's
//!   evaluation an atom and open unit (PGC3D, R4_LIN_2: 88 atoms, 0.1 s a closure; 63 and 29 units kept), and nothing is
//!   subdivided. (Read at the reach's largest speed at each place in its worst direction, a probe tilted 2° off the 1D
//!   PGC kept the Δm = 1 Raman couplings its tilt makes -- S 0.007 at rest, 3.9 at 0.48 µm/µs, where 2kv closes their
//!   two-photon distance -- and a ground sublevels' coupling at beat zero by its sweep at that speed: the model's first
//!   rung fell from 0.108 to 0.038 µs and the solve took twice the untilted one's time. Read at 30 µK's atoms after its
//!   reach grew to 191 µK, S2Q's resonant probe, folded at the atom to the fourth order, was read nowhere on the trap's
//!   flanks where the tweezer brings it through resonance, and its table was refined there to 9 GB.)
//! - A gate at a point is its value there, of the levels there: a distance reads nothing only where two levels cross
//!   there, never because a neighbourhood's steep gradient puts nothing in a cell's range. (Proven over cells, a
//!   probe tilted 2° off a 1D PGC in a tweezer read infinite gates for its ground sublevels' couplings at beat zero --
//!   the tweezer's 236 rad/µs a micrometre over cells that had to shrink to 0.01–0.05 µm, 340 000 of them, 208 s a
//!   model --, where at the points S ≤ 2·10⁻⁴.) Nothing rests on an exact zero: a coupling a broken symmetry makes is
//!   judged by its size at the points, and folds where it is small.
//! - A photon coherence's gate is read at an atom's place alone (photon.rs `Model::coherence_gate`: its classes' jump
//!   operators there, the exact elimination's at rest); over a region no bound of them is made.
//! (The first model keeping nothing, as it was, ran the scout and the pilot on a wrong model and kept by their reads,
//! an attempt each: PGC3D_1ms 3 attempts and 52.5 CPU-s where the prior's line, 717cb386, took one and 12.9; TET_1ms
//! 4 and 120.6 against 16.6.)
//!
//! ## One rule for a keep: the run's total work
//! A keep is chosen by the run's expected total work, wherever the looks are (`needed`): a set's kept model's work a µs
//! times the trajectories it runs -- its pilot again and its product beside the error it leaves --, against keeping
//! nothing, the current model's work a µs times its product's trajectories, where every look has room for the
//! statistics; where some look has none, no number of trajectories fits beside the errors, and keeping nothing is no
//! option. The trajectories follow from the room: a keep that removes error shortens the run. A set kept where nothing
//! is an option saves beyond that saving's spread over the pilot's groups (decide.rs `pays`; `Pricing::share_of`).
//! - The order: at the far end of the error the units taken so far leave, the analysis times where the need binds -- those
//!   without room, else those of the largest need (`binding`) --, and there the units by their weight (their errors on
//!   what is shown and on T, weighed by the held share as the totals are, each over its target) for the work a µs of
//!   their kept model, the most first; and of the sets so led through, the one of the least expected work, none of it
//!   to spare (`least_expected`): its model's work a µs over its pilot and its product, and the chance that the next
//!   pilot still reads a look without room -- the doubt the reads leave on the error, `Doubt::chance` -- times the work
//!   of keeping again. The order ends where no unit weighs where the need binds, or where no larger set can be
//!   expected to cost less than the least found. (Taken one at a time, a keep an attempt, S3's probe kept fourteen bare
//!   drives over fourteen attempts, each a model and a scout; taken by weight alone, it kept its carrier's dressing with
//!   them, three to four times dearer a step; taken until the error read had room, e038 kept 6, 2, 1 and 1 units over
//!   four pilots; taken to the far end of the error, S3c's pilot kept ten units where one gave room.) The product asks
//!   the same where a look the pilot left room has none at the product's statistics (`product`): it ends with the
//!   set. A look the model leaves room whose need a run's `MOST_TRAJECTORIES` do not meet is not chased (`Look::fits`)
//!   -- read again at every batch, as its need is. (Asked where what the atoms outside the reach carry left a look no
//!   room, R3_DOPPLER's product kept two units for T's looks over its last ten outputs, the error of its atoms heated
//!   past the reach -- the kept model's check took half the step, 52 → 91 CPU-s --: that is the reach's to answer, at
//!   the product's end.)
//! - Asked only where some look had no room, as it was, GMD1C's pilot read a model error of 0.047 against the room's
//!   0.0498, and its units, priced one at a time (a candidate where its keep alone paid for the pilot again), kept
//!   none: its 800 trajectories took 549 CPU-s, where its forty drives kept explicit together -- none of them pays
//!   alone, the first one's model dearer a µs by its cluster's growth -- leave 0.0107 and 176 trajectories, 100 CPU-s.
//! - The work: w the counted work a trajectory spends per µs, at the pilot's states (`Pricing::rates`): its steps'
//!   count through the model's generator (step.rs `Price`), its middles' assembly and scattering, and its jumps'
//!   photons at the trajectory's own rate of jumps, all as the run counts them with near tables -- the attempt's for
//!   the current model, the axes its tables would take for a kept one (model.rs `Model::priced`, tabulate.rs
//!   `Model::priced_axes`). A kept model is priced by its work formula, of its structure alone: its clusters, its
//!   drives' levels and its plans, the counts the run's are -- no table is made and nothing eliminated. (Rated with the
//!   tables its attempt would make, S3's pilot made 38 kept models' tables and 2259 floors, a model and its
//!   elimination each, 263 CPU-s a pricing. Priced without tables, by the exact elimination, R4_LIN_2's current model
//!   read 138 times its counted work, and a keep's coordinates -- its tables' axes, an interpolation's work going as
//!   four to their number -- were unseen: S3's carrier's dressing priced as the bare drives, where with its tables it
//!   is twice them, and its run's steps 2.4 times dearer.)
//! - Pricing costs work of its own -- it builds and rates kept models --, so it is spent only where the most any keep
//!   could save pays beyond its spread over the pilot's groups (`Pricing::could_save`, decide.rs `pays`: the old
//!   engine's gate on every spend): every unit's error removed, its pilot run again, at the current model's work a µs,
//!   which a kept model's is not below where a fold costs less than what it folds. (Gated on the least work a µs a kept
//!   model can have instead, the step alone, a twentieth of the current model's, PGC3D's scout and pilot priced keeps
//!   for 0.7 and 1.6 s of a 5.6 s run, where the most any keep could save is 16 % ± 36 %.) The scout and the pilot
//!   price (`Reads::priced`); the product and carried states ask the room alone -- a
//!   carried state's attempt runs its own pilot whatever it keeps, so any saving would pay there (every unit priced
//!   at every attempt's start: R2_RECOIL 98 s, a kept model of 5.7 GB). A keep that pays by its rate alone is not
//!   looked for: a fold is made to cost less than what it folds, and where one does not the fold is at fault, not the
//!   choice -- the order takes the units that weigh.
//! - On the scout, whose reads are eleven states a time, both are judged at the model's error they leave no doubt
//!   of: every model's error less the jackknife interval of the current one's over the scout's groups, at the family
//!   of the analysis times' reads (`doubt`, `clear`); the pilot reads the same units on all its states. A unit's error is
//!   heavy-tailed over the states, and the reads that put the room's error high put the work it asks high with them:
//!   a need is the statistics' interval over the room, and where a point estimate all but fills the target the need
//!   runs to any number, every one of the jackknife's levels with it -- the saving reads near one with no spread.
//!   (Measured, R4_LIN_2 at 76.5 µK, the room judged at the near end and the pricing at the point estimate: the scout
//!   priced one side of a mirror pair of bare drives and then its carrier's dressing at a need of 168 421, its near
//!   end leaving the room whole, and the product ran 1352 trajectories where the model kept on the pilot's reads ran
//!   552.)
//!
//! A model's work a µs counts its prepared potential beside its step: a step's reference evaluates it once at the least,
//! and a trajectory's initial draw as many times as the current ensemble's draws took proposals, each evaluation at its
//! counted work (potential.rs `Expansion::work`: a near light's elimination where the trap light's drives are folded at
//! the atom, 3.2·10⁻⁴ s against the far form's 2.6·10⁻⁷ s on R2_RECOIL); and a keep's expected total holds its model's
//! setup, the prior's atoms and the reach's left-out quadrature at the current attempt's counts (`Setup`,
//! `Formula::setup`). (Priced by the step alone, R2_RECOIL's scout kept the tweezer's drive at the atom at 3619 a µs, the
//! cheapest, and the next model's setup ran past 280 s.)
//!
//! The price of a keep knows the kept model's work a step at the step the model stands at, or at the kept model's own
//! first rung where that is shorter: the beats a keep brings into the generator ask it before any check reads it, and a
//! keep that no rung within `MOST_STEPS` runs is refused, as a fold that does not exist is (`Formula::kept`; a
//! coherence of photons an optical frequency apart, 4.8·10⁹ rad/µs, asks τ < 1.3·10⁻⁹ µs: 5341246b kept such ones for
//! the user's 5 mW probe, and its next model had no rung at any run from 0.1 to 1000 µs). A kept model's check may
//! take a shorter step still. That is the accuracy the kept model asks, read as every model's is, and it is not traded back:
//! a keep undone where its check took a shorter step (as it was) is a decision taken again on the same reads, and it
//! went back and forth -- on PGC3D, with a batch of bare drives booked too high, some twenty keeps and undos, a model,
//! its gates and a pilot each (bounds3.status.md). A model's rate is its mean over the run's held states, read at one
//! state of every pilot trajectory, the states spread evenly over the run (`one_each`), the same for every model.
//!
//! ## The units booked at their bounds
//! A unit is read at the states only where its error can matter. Its error at a state is its small parameter there
//! mapped to what is shown by the leading order (model/units.rs, # The errors); where that parameter has a bound over
//! the whole space -- the prior's screen (# The model's prior): every beam at its peak and all in phase, at the largest
//! speed of the states read, so that the bound holds at each of them, an atom heated past the reach's speeds too --, the
//! same map at that bound and at the populations' largest weight bounds the error at every one of those states, on what
//! is shown and on T alike (model/units.rs `Model::bounds`: a coupling of two states, m²(p_a + p_b) ≤ S/2 for two states
//! and S for a level's own beat). A unit so bounded is booked at that bound at every analysis time read, unread, as long as
//! such bounds, the smallest first, add up to no more than the share the most trajectories a solve runs leave unseen
//! (`booked`; estimate.rs `unseen`, z²/(N + z²) = 2.44·10⁻⁴ at N = `MOST_TRAJECTORIES`): what is shown is never booked
//! statistics below that share (`Estimate::shown_statistics`), so what the unread units' bounds overstate together lies
//! below what any run resolves; on T, relative, it is a four-hundredth of the 10 % target. The others are read, once
//! per state: a state's errors are kept as read and their means per analysis time made once for every decision that asks
//! (`Measured`). The ledger names each unit's booking (`FOLD_MEASURED`, `FOLD_BOUNDED`). (Every unit read, as it was
//! after the gates over the reach went, GMD1C's scout read its 2130 units at 11 011 states, 14.5 of its 39 s, and the
//! run took 1.6 times the line's CPU: its 927 couplings of two states, each bounded at 2·10⁻¹¹ or less over the whole
//! space, 3·10⁻¹⁰ together, were 77 % of a state's reads. The line, 717cb386, booked them by their bounds over the
//! reach. Booked at their bounds over the whole space, the scout reads 1203 units there in 0.8–1.1 s, and the run takes
//! 0.75 times the line's CPU: its trajectories and its step the line's, its T within their errors.)
//!
//! ## Where the atoms went
//! The model is judged where the atoms went, not only where the held ones are at the analysis times (`sampled`). What is
//! shown at a time is every atom's fate there -- a held atom's rows, a lost one's loss --, so the model errs on it by the
//! mean over the atoms of the error of each one's fate (`Measured::means`): the held atoms' at their states there, by
//! the held share 1 − L, and the lost atoms' along their paths, by L. A lost atom's fate was decided along its path
//! from t = 0 to its loss: a rate's relative error x along it moves its loss's time by x of it to the first order, and
//! its fate's error is the path's mean of x, read at the two states the run holds of it -- at t = 0 and where it was
//! found lost -- by the trapezoid (`Booking::path`). T is the held atoms' alone. A run's price is read where its atoms
//! were too (`one_each`): a trajectory weighs by the time it holds its atom, and one lost before the first time read by
//! its state at t = 0. (Read on the held atoms alone, S2Q's first model -- nothing kept, its resonant probe's drives
//! folded by the bare levels -- lost every atom within 10 µs, read no state at any output and booked its error as
//! none, and the solve stood on every atom lost by 50 µs, where half of them are held to 0.53 ms; priced at the held
//! states alone, every keep of it cost nothing, and it kept a coherence of photons 736 THz apart, which no step
//! resolves.)
//!
//! ## The step's part
//! The room a keep reads is what the step leaves the model: E² = A² + c² + S² ≤ ε² leaves the statistics room while
//! A < √(ε² − c²), c the step's error. Before the step's reads are in -- the scout, the carried states -- c is its
//! share s·ε: the check leaves the step it stands at within its share (`verdict`). Once the pilot's rung has its own
//! read, c is that read's estimate, at most the share (`standing`): the step's error the run is to book, which the
//! check does not raise past the share. (717cb386, PGC3D at 1000 outputs: the model's threshold on what is shown is 0.0447 at the
//! share, 0.0499 at its read step error, 0.0022; the pilot's keep stood on 39 outputs at 0.0450–0.0497, within the
//! second -- the share's, not the run's: r_samples.)
//!
//! # The reach's atoms (SPEC §1.8)
//! The reach -- the thermal ensemble at its temperature -- is what the run is made for: the first rung is admissible at
//! its largest speed, the tables are refined over its ranges, and the pilot's atoms are its draws. Whether the model's
//! folds hold there is read at the atoms' starts (# The model's prior) and measured where they went, not proven
//! (# The model). Where the atoms go is not proven before they go there either:
//! a step asks its domain two cheap things (`Domain`; step.rs, # The domain).
//! - Whether a state scatters beyond every state of the atom: a fold far outside its validity, whose rate grows without
//!   bound (the released runaway: a drive folded at the atom past its Doppler series' radius, 10⁵ jumps/µs against
//!   Γ/2). The trajectory stops there (`run`): unresolved from then on, its state as it stood. Its states are the
//!   pilot's to read like any other, and the units whose errors leave a look no room there are kept by the same rule
//!   (# The model): the guard keeps nothing itself.
//! - Whether the step resolves the light along its path at the atom's speed (`resolves`): a step that does not is
//!   taken, and its trajectory is unresolved from it on (step.rs `Trajectory::unresolved`) -- every later state of it
//!   is a wrong step's descendant.
//! What an unresolved atom shows is booked whole: its share of the held atoms on what is shown -- an atom's row is a
//! probability, so its error is at most one, whatever its state, and the mean over the held atoms errs by at most the
//! share --, its share of the held atoms' kinetic energy on T -- the part of T it makes, which bounds its error while
//! its true kinetic energy is within twice the one computed (a hypothesis on T, not a bound). The loss of an atom lost
//! after such a step is no held atom's: as every model error's, it is in what the held atoms' errors are weighed by
//! (the held share, `accounts`), and not booked apart.
//!
//! The atoms the run takes past the reach are booked as the old engine booked them (`beyond_reach`): where a time's
//! T along a lab axis is above the reach's T_r, the share of a thermal ensemble at that T past the reach's level beyond
//! the share s the reach leaves out at its own temperature, s^(T_r/T) − s, on what is shown and on T alike -- their
//! folds unread. It is in the looks: the run stops on totals that hold it. Where the product ends with a look that it
//! leaves no room for the statistics and that has room without it, the reach grows to the bound atoms' temperature
//! read where that is above it (`attempt`), and the next attempt begins; else the item stands booked and the ledger
//! flags the output. The reach's falsifier reads the pilot's bound atoms (`outgrown`, estimate.rs `bound_motion`): a
//! held atom whose prepared energy is at the far value or above at a time is leaving, no draw of the distribution
//! the reach is made for.
//!
//! # Stopping (SPEC §3)
//! Every quantity held to a target -- at every analysis time after t = 0 what is shown, and the total T -- has the total
//! E² = A² + c² + S² (ledger.rs; `Look`): A the model's error, from the units measured at the pilot's states -- its
//! held atoms' and its lost atoms' paths (# The model: where the atoms went) -- and the tables' errors there, and c
//! what the step's check leaves open (its rms over the run), both fixed by the pilot -- A with what
//! the run's atoms outside the reach carry, read before each batch's looks (# The reach's atoms) --; S the
//! statistics' interval. The run stops where every one of them lies
//! within its target: the test the ledger flags an output by, at the outputs (`Look::met`). The stop reads the run's standard errors,
//! its atoms outside the reach, and nothing else of its estimates. A look that no run's trajectories bring within its
//! target -- its need beyond `MOST_TRAJECTORIES`, or A² + c² ≥ ε² (`Look::fits`): the model's decisions have kept
//! what they could -- is not chased: the ledger flags it.
//!
//! The trajectories a look needs (`Look::need`): the statistics may take the room R² = ε² − A² − c², and fall as
//! N^{−1/2}; on what is shown they are no less than the unseen share at N (estimate.rs `unseen`, `draws_for`). A batch
//! brings the trajectories to the number the looks predict, at most to twice the number the prediction rests on
//! (`batch`).
//!
//! # T's target is on the total temperature
//! The user's target on T, 10 % relative at the confidence z, is on the total temperature: the mean of the three lab
//! axes' T (`Motion::temperature_total`), its statistics the jackknife of that mean. It is held at every analysis time after
//! t = 0 where it is read, as its allowance there (# T's allowance) -- its step's size checked against the step's share,
//! what the check leaves open booked, its look met or chased (`looks`), the total flagged where it exceeds the allowance
//! (ledger.rs). Each axis's T is shown with its own statistics and held to nothing: an axis that carries little of the
//! energy does not set the run's length.
//!
//! # T's allowance
//! The point particle has an error of its own on T that no step and no trajectory takes away. In a trap it has no zero
//! point: it holds T down to nothing where a quantum oscillator's kinetic temperature stays at ħω/2k_B per axis (the
//! old engine's references read T 10 % low at k_BT = 4ħω and 46 % at 1.2ħω); in sub-Doppler cooling it has no lower
//! bound, where the full quantum treatment bottoms out at about 5–6 ħk rms per axis, 30 k_BT_rec
//! (`SUB_DOPPLER_RECOILS`) -- there the semiclassical T is off by O(1). The scene's scale of that error is T_sc =
//! max(30 T_rec, ħω̄/2k_B) (estimate.rs `semiclassical`): k_BT_rec = (ħk)²/m at the mean k² of the photons the
//! trajectories' jumps emit -- the recoil the jumps give, the transition that scatters most's where one does (D1 and
//! D2 differ by 4 % in T_rec) --, and ħω̄/2k_B the mean over the lab axes of the reported trap's zero point (the
//! frequencies T's windows and n̄ are read against, the old engine's ω), the error of the total T, the axes' mean. The
//! numerical error and the model's own are independent, so an output's total T is held to the allowance δT =
//! √((ε_T T)² + T_sc²), relative √(ε_T² + (T_sc/T)²) (`allowance`): the user's 10 % where T ≫ T_sc, the model's own
//! error scale where T is near or below it -- nothing is spent where the model cannot be right, and nothing switches:
//! the allowance is smooth in T, and so are the step's reads, the trajectories a look needs and the units' weights it
//! enters. It is T's target wherever T's target enters: the looks and the ledger (`looks`, `accounts`, ledger.rs), the
//! step's reads (estimate.rs `between`: each output's error over its own allowance, the step's share s of it,
//! `step_bound`), the units' weights for a keep (`weights`). The populations and the loss keep their absolute target.
//! (PGC3D's tweezer, 100 and 19 kHz: ħω̄/2k_B = 1.8 µK, 30 T_rec = 10.9 µK for Rb87 on D2, an allowance of 81 % at its
//! 13.5 µK; 19 % at S3's 66 µK, 16 % at 89 µK.)
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
//! model's construction, the tables and their refinement, the units' reads at the states, the pricing
//! of keeps.
use crate::adapt;
use crate::analysis::analysis_times;
use crate::decide::{pays, Outcome, Read};
use crate::ensemble::{Ensemble, Evaluations, Start, Well, GROUP};
use crate::estimate::{allowance, between, bound_motion, draws_for, half_window, readings, semiclassical, spread, summarize, unseen, Between, Estimate, Fate, Group, Held, Scales, Spread};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, significant, span, when, Account, Ledger};
use crate::linalg::{norm3, V3};
use crate::model::optical::Unmade;
use crate::model::units::destined;
use crate::model::tabulate::{Key, Tables};
use crate::model::{Envelope, Lights, Model, Priced, Run, Sample, Trap, Unit, UnitError, GROUND};
use crate::request::STATES;
use crate::step::{stumpff, Dynamics, Price, Refused, Scratch, Segment, Trajectory};
use crate::units::{hbar_over_mass, CLASSICAL, CONFIDENCE, KT_PER_UK, MOST_STEPS, MOST_TRAJECTORIES, ORDER, STEP_SHARE};
use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
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
const FOLD_MEASURED: &str = "绝热消去（二阶），误差沿先导轨迹实测";
/// What the engine did with a folded unit it did not read (`booked`): its gate's bound over the whole space at the
/// speeds of the states read is its error.
const FOLD_BOUNDED: &str = "绝热消去（二阶），误差按全空间上的上界（每束光取峰值且同相，速度取所读状态的最大值）计入误差预算";
/// What the engine did with a near table's error (SPEC §5.4): refined over the reach when it was built, and read
/// against the elimination itself at the pilot's states.
const TABLE_MEASURED: &str = "在原子的活动范围上加密的插值表，误差在先导轨迹的状态上与直接计算对照实测";

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
///
/// A model is made of the units kept, and attempted over a reach (`attempt`) until an attempt changes the units or
/// gives the solution. What an attempt takes splits by what it rests on:
/// - on the model alone -- the model with what it holds once made (its plans, its folds), its units, the reference
///   state's prepared trap, every component's well and the initial ensemble drawn in them, the reported trap, the
///   entries the first rung reads (`Setting`): made once for a model, and the same at every attempt at it;
/// - on the reach as well -- what the reach leaves out of the ensemble, the kick's curvature, the first rung -- and on
///   the step the attempt begins at too -- the Brownian path's cell, the tables' bound and so the tables, every
///   trajectory and all that is read on them: an attempt's own.
/// So a reach that has grown (`Attempt::Grow`) is attempted with the model as it stands, and units kept
/// (`Attempt::Keep`) make the next model. Every attempt's end hands the next the states its run level left (`Carried`),
/// carried into the next model where the units changed (`Carried::into`): the next attempt decides on them first
/// (module documentation, # An attempt's own).
pub fn solve(request: &Request, checkpoint: &mut dyn FnMut(&Solution, Option<f64>)) -> Result<Solution, String> {
    request.validate()?;
    let light = Light::new(request);
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request);
    // The reference state (SPEC §2.1): the initial component of the largest population, of equal ones the last.
    let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).expect("a validated request's initial state holds a ground state");
    let reference = alone(initial[chief].0);
    // The units kept and the reach's temperature -- both only grow --, the model they make, closed by its prior (`closed`),
    // and the states the last attempt's run level left (`Carried`).
    let (mut kept, mut reach) = (vec![], request.temperature_uk);
    let mut model = closed(request, &light, &mut kept, &initial, reach)?;
    let mut carried: Option<Carried> = None;
    let mut messages = Messages { send: checkpoint, clock: Instant::now(), last: None, least: 0 };
    loop {
        // The units the next model keeps, and the states this model's last attempt left (an attempt's keep).
        let (more, left) = {
            // A model's units ask nothing of its trap: the two are made side by side on the pool.
            let (trap, units) = rayon::join(|| model.trap(&reference, &model.prepared()), || model.units());
            let trap = trap?;
            // The reported trap and the entries the first rung reads ask nothing of one another: side by side on the pool.
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, &light, alone(s))).collect::<Result<_, _>>()?;
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm)?;
            let (reported, entries) = rayon::join(|| model.trap(&reference, &model.reported()), || model.sampled(&model.running()));
            let setting = Setting { model: &model, reference, units: &units, trap: &trap, ensemble: &ensemble, depth: ensemble.depths()[chief], reported: reported.ok(), entries };
            loop {
                // An attempt's results improve on its own last one (module documentation, # Checkpoints).
                messages.least = 0;
                match attempt(request, &setting, &kept, reach, carried.take(), &mut messages)? {
                    // A grown reach is the model's prior's again: its atoms are another ensemble's, and what fails
                    // among them is kept as an attempt's keep is (# The model's prior).
                    Attempt::Grow(t, left) => {
                        reach = t;
                        let more = failing(&model, request, &light, &initial, reach)?;
                        if !more.is_empty() {
                            break (more, left);
                        }
                        carried = Some(left);
                    }
                    Attempt::Keep(units, left) => break (units, left),
                    Attempt::Done(s) => return Ok(s),
                }
            }
        };
        kept.extend(more);
        let next = closed(request, &light, &mut kept, &initial, reach)?;
        carried = Some(left.into(&model, &next));
        model = next;
    }
}

/// The model of the units `kept` and of every unit whose fold does not exist where the reach's atoms are, added to
/// `kept` (module documentation, # The model's prior; `failing`): the units that fail kept, and the model they make
/// read again, until none fails -- a drive up its levels as far as the first whose fold exists (model.rs `Level`).
fn closed(request: &Request, light: &Light, kept: &mut Vec<Unit>, initial: &[(usize, f64)], temperature: f64) -> Result<Model, String> {
    loop {
        let model = Model::new(request, light, kept)?;
        let more = failing(&model, request, light, initial, temperature)?;
        if more.is_empty() {
            return Ok(model);
        }
        kept.extend(more);
    }
}

/// The atoms a model's prior reads its gates at (module documentation, # The model's prior): the draws of the reach's
/// own ensemble -- the initial components in their wells, the lowered isothermal at the reach's temperature
/// `temperature` (µK; ensemble.rs) -- of the pilot's trajectories, `MOST_TRAJECTORIES` on, the run's seed: at the
/// request's temperature the pilot's own initial atoms. Each its place and its speed |v| = (ħ/m)|κ|.
fn atoms(model: &Model, request: &Request, light: &Light, initial: &[(usize, f64)], temperature: f64) -> Result<Vec<(V3, f64)>, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(model, light, alone(s))).collect::<Result<_, _>>()?;
    let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), temperature, hm)?;
    Ok((MOST_TRAJECTORIES..MOST_TRAJECTORIES + pilot_size())
        .into_par_iter()
        .map(|index| {
            let start = ensemble.draw(SEED, index);
            (start.r, hm * start.kappa.iter().map(|k| k * k).sum::<f64>().sqrt())
        })
        .collect())
}

/// The model's units whose folds do not exist where the reach of the temperature `temperature` (µK) has its atoms
/// (module documentation, # The model's prior): the gates (model/units.rs `Model::failing`) at its atoms (`atoms`),
/// each at its place and its speed -- a state's errors are read so (model/units.rs `Measured`) --, bounded first over
/// the whole space at the fastest of them.
fn failing(model: &Model, request: &Request, light: &Light, initial: &[(usize, f64)], temperature: f64) -> Result<Vec<Unit>, String> {
    let points = atoms(model, request, light, initial, temperature)?;
    let fastest = points.iter().fold(0., |most: f64, p| most.max(p.1));
    Ok(model.failing(&model.units(), &points, fastest))
}

/// What an attempt takes of its model alone (`solve`): nothing of it rests on the reach the model is judged over or
/// on the step, so it is made once for a model and is the same at every attempt at it.
struct Setting<'a> {
    model: &'a Model,
    /// The reference state's ground populations (SPEC §2.1).
    reference: [f64; GROUND],
    /// The model's folded units (model.rs `units`).
    units: &'a [Unit],
    /// The reference state's prepared trap.
    trap: &'a Trap,
    /// The initial ensemble, every component in its own well, and the depth at the reference state's bottom (k_BT).
    ensemble: &'a Ensemble<'a>,
    depth: f64,
    /// The trap every reported quantity describes (SPEC §1.3: trapFrequencyKHz, n̄, the validated regime and T's
    /// windows): the trap light's potential with the other lights' phase-averaged shifts at the reference state
    /// (`Model::reported`). None where it has no minimum the descent reaches -- the atoms are not held by the light
    /// they run in: a mathematical object that does not exist, not a kind of scene --: the prepared trap's is
    /// reported then, and flagged.
    reported: Option<Trap>,
    /// The entries the first rung reads (`admissible`; model/assemble.rs `Model::sampled`).
    entries: Vec<[f64; 2]>,
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

/// How one attempt at a model ends (`attempt`): the reach is outgrown by the run's atoms (the reach's new temperature,
/// µK: the model is attempted again as it stands, `solve`), units are to be kept (`keep`) -- each with the states the
/// attempt's run level left, the next attempt's first evidence (module documentation, # An attempt's own) --, or the
/// solution.
enum Attempt {
    Grow(f64, Carried),
    Keep(Vec<Unit>, Carried),
    Done(Solution),
}

/// What an attempt's end hands the next (module documentation, # An attempt's own): the groups of its run level with
/// their states -- the scout's, a pilot's own level as far as it ran, or the pilot's --, and whether they are few, the
/// scout's (`Reads::Carried`: judged at what they leave no doubt of). Nothing of the step: the next attempt's ladder
/// begins at its own first rung, and goes down only on its own reads (# The step).
struct Carried {
    parts: Vec<Part>,
    few: bool,
}
impl Carried {
    /// The states as the model `to` holds them, where they are the model `from`'s (`Model::carried`: the next model
    /// keeps this one's units and more), every state of every group -- each a function of its own state alone, on the
    /// pool.
    fn into(mut self, from: &Model, to: &Model) -> Carried {
        let carry = |state: &mut State| state.sample = from.carried(&state.sample, to);
        self.parts.par_iter_mut().for_each(|part| {
            part.samples.iter_mut().flatten().for_each(carry);
            part.lost.iter_mut().for_each(|(_, state)| carry(state));
        });
        self
    }
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
/// - The reference flow and the kick between its halves, at the prepared trap's largest frequency ω (rad/µs) and the
///   kick's largest curvature `kick` = ω_k² = (ħ/m) λ_max ∂∂(Ĥ_h − V_ref) (rad²/µs²; model/reach.rs
///   `Model::kick_curvature`, with the derivation, what of it the model bounds and what it leaves to the check).
///   About a minimum a step is the linear map R(ωτ/2) K R(ωτ/2), its
///   trace 2 cos ωτ − ω_k²τ² sin(ωτ)/(ωτ) (measured on the crate's step: step/tests.rs), and the motion is bounded
///   while the trace is above −2: on the first stretch, ωτ < π, the trace over two is above −1. It falls with ω and
///   with ω_k², so the largest of each stands for every axis and state. A kick without curvature leaves ω τ < π --
///   there the two half flows turn the motion by π and any kick between them grows --, a softer well than the
///   reference's (ω_k² < 0) the same; a stiffer one ends the stretch below π, (ωτ/2) tan(ωτ/2) = ω²/ω_k², and on a
///   flat reference at the kick's own ω_k τ = 2. The same for every atom.
/// - The generator: the propagation reads its change along the path at three nodes, the step's middle and its two
///   ends (step.rs, # The kick), τ/2 apart, and takes it for the parabola through them. An entry that turns at the rate
///   ρ along the path -- its residual beat Ω in the cluster's frames and its residual wave vector K at the atom's
///   velocity, ρ = |Ω + K·v| ≤ |Ω| + |K| v (`sampled`: the model's own entries, per beat the largest |K|) -- advances by
///   ρ τ/2 from node to node; at π the nodes read it as its alias. So (|Ω| + |K| v) τ/2 < π for every entry.
/// Both are edges -- where the scheme stops meaning anything --, not accuracies within a share: how well a step below
/// them does is the check's to say (the pilot's two levels). They are not edges of one kind (step.rs, # The domain):
/// beyond the first (`stable`) the step multiplies every error by e^μ, cosh μ = |trace|/2, a step after a step -- no
/// atom's step means anything there --; beyond the second (`resolves`) the nodes read an alias, a parabola through
/// three samples of the entry, at most 5/4 of it anywhere and Hermitian where it is: the atom goes on under a wrong
/// generator as bounded as the true one, its step inaccurate, not unstable. So a step of an atom past the second is
/// taken, and what the atom shows is booked whole from it on (# The reach's atoms); past the first it is not.
fn admissible(tau: f64, frequency: f64, kick: f64, sampled: &[[f64; 2]], speed: f64) -> bool {
    stable(tau, frequency, kick) && resolves(tau, sampled, speed)
}

/// Whether a step of τ is below the edge of its reference's turn and its kick's curvature (`admissible`): the same for
/// every atom of the trap's largest frequency and the kick's largest curvature, whatever its speed.
fn stable(tau: f64, frequency: f64, kick: f64) -> bool {
    let turn = frequency * tau;
    // cos ωτ and sin(ωτ)/(ωτ): the Stumpff functions the reference's flow is made of (step.rs).
    let (cos, sinc, _) = stumpff(turn * turn);
    turn < PI && cos - 0.5 * kick * tau * tau * sinc > -1.
}

/// Whether a step of τ resolves the generator's turn along the path of an atom of a speed (`admissible`): every entry's
/// (|Ω| + |K| v) τ/2 below π.
fn resolves(tau: f64, sampled: &[[f64; 2]], speed: f64) -> bool {
    sampled.iter().all(|[beat, wave]| 0.5 * (beat + wave * speed) * tau < PI)
}

/// The step the physics prior allows (module documentation, # The step; the old engine's first step), µs: T's step
/// error, relative, goes as the square of the phase the light's pattern advances along an atom's path in a step,
/// (k v τ)², and the step may take its share of T's allowance (`step_bound`, # T's allowance), k_max v_T τ = √(s δT/T)
/// -- at the carriers' largest wave number `wave` (1/µm; model/eliminate.rs `Model::carrier_wave_number`), the rms
/// speed along an axis at the temperature `temperature` (µK), v_T = √(k_BT/m) -- the reach's (`attempt`: its
/// ensemble's mean kinetic temperature, the typical atom's, `typical_speed`; the request's until the atoms grow it, as
/// the old engine's at the initial temperature) --, and T's allowance there at the
/// scale T_sc of that light's recoil (estimate.rs `semiclassical`: one photon of k_max). No bound where the light has no
/// pattern or the atoms no speed. (PGC3D, Rb87 at 30 µK on D2: k_max v_T = 0.43 rad/µs, T_sc = 10.9 µK, δT/T = 0.375:
/// τ ≤ 0.95 µs, where the ladder's top by admissibility alone, 1.425 µs, was left on the pilot and 0.7125 µs stood.)
fn prior(wave: f64, hbar_over_mass: f64, temperature: f64, scales: &Scales) -> f64 {
    let speed = (hbar_over_mass * KT_PER_UK * temperature).sqrt();
    (step_bound()[1] * allowance(temperature, semiclassical([1., wave * wave], scales))).sqrt() / (wave * speed)
}

/// The ladder's first rung (module documentation, # The step): the largest rung -- the run's length `run` (µs) over an
/// even number, its double a whole part of the run too -- that is no longer than the physics prior's step `prior`
/// (`prior`), is admissible for the run and whose double is admissible for the companion (`admissible`), at the speeds
/// `fastest` and `typical` (µm/µs) of the reach. (Both are bounds on the step from above: the least even number is
/// found by counting up, from the prior's.)
/// - The run's step is the result's: it is admissible for every atom the model is judged for, at the reach's largest
///   speed (model.rs `Reach::speed`) -- one domain for the model's folds and for the step.
/// - The companion at twice the step is a probe of the step, not a result: it must be stable, which the step's edge
///   decides for every atom alike -- the same edge as the run's, at the same frequency and curvature --, and resolve
///   the light for the typical atom, at the reach's rms speed
///   (`typical_speed`) -- two steps one of which reads the light's alias at most atoms agree or differ for no reason of
///   the step, while the few fastest atoms beyond its limit weigh in the difference by their share.
/// Where the step's edge binds, the first rung is the rung below the largest stable one. Half the run at the
/// most: the companion's step is a rung itself, the run its one step then. Nothing of the outputs enters: they are read
/// inside the steps (adapt.rs, # The reads). The ladder has one bottom, the step at which a trajectory takes `MOST_STEPS`
/// steps, and the first rung is a rung of it: none where no rung down to that one is admissible -- the light or the
/// trap asks for a shorter step than the cap allows, or a frequency, a curvature or a speed is no number (nothing is
/// admissible then). The solve refuses such a request (`attempt`), and a keep whose model has none is refused
/// (`Formula::kept`): a step that is not admissible computes nothing.
///
/// The count of pairs n (the rung run/(2n)) starts where a rung can first be admissible, each bound read off its
/// condition: within the prior, run/(2n) ≤ prior; the reference's turn, ωτ < π for the rung and for its double,
/// n > run ω/π; the nodes, ½ ρ τ < π at the largest rate ρ = |Ω| + |K| v of the entries (`resolves`), n > run ρ/(4π) at
/// the fastest speed and n > run ρ/(2π) for the double at the typical one. Below that n no rung is admissible, so the
/// rung found is the scan's from n = 1; and a light that asks a beat no rung within the cap resolves is told at once,
/// not by counting to the cap (a keep of photons 736 THz apart asks 10⁻⁹ µs: measured, 5341246b's second model of the
/// user's 5 mW probe, 14.8 s to say it).
fn first_rung(run: f64, prior: f64, frequency: f64, kick: f64, sampled: &[[f64; 2]], fastest: f64, typical: f64) -> Option<f64> {
    if !sampled.iter().flatten().chain([&frequency, &kick, &fastest, &typical]).all(|x| x.is_finite()) {
        return None;
    }
    let rate = |speed: f64| sampled.iter().map(|[beat, wave]| beat + wave * speed).fold(0., f64::max);
    // The prior's bound is the rule itself (nothing else checks it): its ceiling. The others are admissibility's, which
    // decides at each n: their floors start the count at or below the first n each allows, the rounding of the products
    // taken one before it at most.
    let least = [(run / (2. * prior)).ceil(), (run * frequency / PI).floor(), (run * rate(fastest) / (4. * PI)).floor(), (run * rate(typical) / (2. * PI)).floor()].into_iter().fold(1., f64::max);
    (least <= (MOST_STEPS / 2) as f64)
        .then(|| (least as u64..=MOST_STEPS / 2).map(|pairs| run / (2 * pairs) as f64).find(|&step| admissible(step, frequency, kick, sampled, fastest) && admissible(2. * step, frequency, kick, sampled, typical)))
        .flatten()
}

/// What a model's first rung reads beside the entries its steps sample (`first_rung`): the run's length, the physics
/// prior's step, the trap's largest frequency, the kick's curvature over the reach and the reach's speeds -- an
/// attempt's, the same for every model priced in it (`Formula::kept`: a kept model's kick, a new curvature of the
/// light's potential, is its own attempt's to read; its entries -- the beats a keep brings into the generator -- are
/// read here).
#[derive(Clone, Copy, Debug)]
struct Ladder {
    run: f64,
    prior: f64,
    frequency: f64,
    kick: f64,
    fastest: f64,
    typical: f64,
}
/// What an attempt's model costs in evaluations of its prepared potential beside its steps' own work (`Formula::rate`,
/// `Formula::kept`), counted on the current attempt: per trajectory's initial draw the proposals it reads
/// (`Ensemble::evaluations`), and per attempt its setup's -- the prior's atoms, the pilot's number of draws
/// (`atoms`), and the reach's left-out quadrature (`Model::left_out`), as many as the current attempt's took. A kept
/// model's attempt makes the same evaluations, each at its own work. (Each step reads the potential at its reference
/// once at the least, step.rs `middle`: that one is the rate's.)
#[derive(Clone, Copy, Debug)]
struct Setup {
    per_draw: f64,
    evaluations: f64,
}
impl Setup {
    fn of(drawn: Evaluations, left_out: f64) -> Setup {
        Setup { per_draw: drawn.per_draw, evaluations: pilot_size() as f64 * drawn.per_draw + left_out }
    }
}

impl Ladder {
    /// The first rung of a model whose steps sample `sampled` (`first_rung`).
    fn first(&self, sampled: &[[f64; 2]]) -> Option<f64> {
        first_rung(self.run, self.prior, self.frequency, self.kick, sampled, self.fastest, self.typical)
    }
}

/// The rms speed (µm/µs) of atoms at the temperature T (µK): √(3 (ħ/m) k_BT/ħ), the reach's typical atom at the
/// mean kinetic temperature of the ensemble the reach is made for (model.rs `Reach::kinetic_uk`) -- and no faster than
/// the reach's fastest atom, `fastest` (model.rs `Reach::speed`): a reach holds the trap's depth at most, while its
/// temperature's s²T grows without bound (a reach grown to 11 498 µK read a typical speed of 1.82 µm/µs against its
/// largest 0.397).
fn typical_speed(hbar_over_mass: f64, temperature_uk: f64, fastest: f64) -> f64 {
    (3. * hbar_over_mass * KT_PER_UK * temperature_uk).sqrt().min(fastest)
}

/// What a level's steps ask of the model's domain (SPEC §1.8; module documentation, # The reach's atoms; step.rs, # The
/// domain): the model and the reach -- where an atom lies against it (`placed`) --, ħ/m, the entries `resolves` reads,
/// whether the level's step is the run's (`result`: its rungs and those below; a companion's atoms beyond its light's
/// limit weigh in its difference by their share), and the fastest decay of the model's excited states, `widest`
/// (1/µs): no state of the atom scatters faster than that, wholly excited (`runaway`).
#[derive(Clone, Copy)]
struct Domain<'a> {
    model: &'a Model,
    reach: &'a crate::model::Reach,
    hbar_over_mass: f64,
    sampled: &'a [[f64; 2]],
    widest: f64,
    result: bool,
}
impl Domain<'_> {
    /// Whether an atom at r with the momentum κ lies outside the reach, and whether it is leaving (`Model::situated`).
    fn placed(&self, r: V3, kappa: V3) -> [bool; 2] {
        let [held, leaving] = self.model.situated(self.reach, r, kappa);
        [!held, leaving]
    }
    /// The same domain for a companion's level (`result` off).
    fn companion(&self) -> Self {
        Domain { result: false, ..*self }
    }
}
impl crate::step::Domain for Domain<'_> {
    /// On a level whose step is the run's, whether the step resolves the light along its path at its speed
    /// (`resolves`): a step below the first rung's edge at the reach's largest speed does at any speed up to it.
    fn resolves(&self, _: &crate::step::Flow, (_, kappa): (V3, V3), half: f64) -> bool {
        let speed = self.hbar_over_mass * kappa.iter().map(|k| k * k).sum::<f64>().sqrt();
        !self.result || resolves(2. * half.abs(), self.sampled, speed)
    }
    /// A hazard above the fastest excited state's decay, wholly excited: a fold far outside its validity (step.rs, # The
    /// domain).
    fn runaway(&self, hazard: f64) -> bool {
        hazard > self.widest
    }
}

/// The reads a step is judged by (module documentation, # The step): the mean squares over the run of the step's
/// error on the loss, on every row as shown and on the total T, each with the square of the step's share of its
/// target. One family: what is clear of one of them is clear among them all.
fn judged(read: &Between) -> Vec<(Read, f64)> {
    let [sp, st] = step_bound();
    let mut looks = vec![(read.loss, sp * sp)];
    looks.extend(read.shown.iter().map(|x| (*x, sp * sp)));
    looks.extend(read.temperature.map(|t| (t, st * st)));
    looks
}

/// The screen of a step on the reads between its two levels (module documentation, # The step; decide.rs, # Many
/// reads): whether the family of its reads is beyond -- some read beyond the square of the step's share among all the
/// family's looks. The step is halved where it is.
fn screen(read: &Between) -> bool {
    let reads = judged(read);
    reads.iter().any(|(read, bound)| read.outcome(*bound, reads.len() as f64) == Outcome::Beyond)
}

/// A read as the scheme's order carries it to the step `ratio` times its own (module documentation, # The step): a
/// mean square goes as the step to twice the order, so each read's value and spread times ratio^{2p}. To half the
/// step, over 2^{2p}: a rung's read as the rung below's third level above. To twice the step, 2^{2p} times: a read is
/// of its finer level's error, and this is the same difference read for its coarser level -- the read below as its
/// rung's third level (`Levels::told`).
fn carried(read: &Between, ratio: f64) -> Between {
    let part = ratio.powi(2 * ORDER);
    let carry = |r: &Read| Read { value: part * r.value, spread: Spread { error: part * r.spread.error, ..r.spread } };
    Between { loss: carry(&read.loss), shown: read.shown.each_ref().map(carry), temperature: read.temperature.as_ref().map(carry) }
}

/// Whether a read's every look is within the step's share as it reads (module documentation, # The step): its
/// estimate M̂ -- unbiased, the noise's floor taken out -- at most the square b of the share; a look that is no number
/// is not. The convergence test as a refinement is checked by hand and as an adaptive integrator's local control takes
/// it: the change between two levels within the tolerance, on the estimate.
fn within(read: &Between) -> bool {
    judged(read).iter().all(|(read, bound)| read.value <= *bound)
}

/// What the check makes of a rung on its reads (module documentation, # The step; `verdict`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Within on its own read and on its third level's -- or, with no cheap third level, on its own two levels, as an
    /// adaptive integrator's local control takes them; on the scout, not clearly beyond: the rung stands.
    Stands,
    /// Its own read clearly beyond the step's share, or its third level's estimate beyond it (the order is not shown
    /// above it): the rung is left.
    Left,
    /// On the pilot, its own read within and no third level on the pilot's pairs, where the ladder has the rung above
    /// and that rung's paths stay in the model's domain: the pilot's trajectories at 4τ, a quarter of a trajectory each.
    Above,
    /// Its own read's estimate beyond the share, but not clearly: the noise hides the difference -- the pairs that
    /// could see through it (`remedy`), else the rung is left.
    Unseeing,
}

/// What the check makes of a rung (module documentation, # The step; `settle`) on its own `read` and its third level
/// above carried to its step, `above` (none where it has none); `pilot`: whether the read is the pilot's, the last word
/// on the step (the scout's leaves only on what is clearly beyond); `upward`: whether a third level above on the pilot's
/// own pairs is to be had (the ladder has the rung above, the rung has none yet, and the rung above's read was not
/// unread). Every
/// estimate is taken as it reads (`within`). The third level is had where it is cheap -- the rung the ladder left
/// above, or the pilot at 4τ --; where none is, the rung stands on its own two levels.
fn verdict(read: &Between, above: Option<&Between>, pilot: bool, upward: bool) -> Verdict {
    if screen(read) {
        Verdict::Left
    } else if !pilot {
        Verdict::Stands
    } else if !within(read) {
        Verdict::Unseeing
    } else if let Some(above) = above {
        if within(above) { Verdict::Stands } else { Verdict::Left }
    } else if upward {
        Verdict::Above
    } else {
        Verdict::Stands
    }
}

/// What a rung's reads say together (module documentation, # The step): per look, of its own read and its third
/// level's carried to its step, the read of the larger estimate -- each is the rung's mean square where the order
/// holds, and the larger is the one the rung stands by --, or of the larger far end where the check could not see
/// (`far`). A T the rung's own levels do not read stays unread.
fn worse(own: &Between, third: Option<&Between>, far: bool) -> Between {
    let Some(third) = third else { return own.clone() };
    let size = |r: &Read| if far { r.most(1.) } else if r.value.is_nan() { f64::INFINITY } else { r.value };
    let larger = |a: &Read, b: &Read| if size(b) > size(a) { *b } else { *a };
    Between {
        loss: larger(&own.loss, &third.loss),
        shown: std::array::from_fn(|s| larger(&own.shown[s], &third.shown[s])),
        temperature: own.temperature.map(|a| third.temperature.map_or(a, |b| larger(&a, &b))),
    }
}

/// Whether a read excludes something on every look (module documentation, # The step): each far end a number. A look
/// that is no number, or whose spread is unread, excludes nothing; no pairs and no rung make it say more.
fn readable(read: &Between) -> bool {
    judged(read).iter().all(|(read, _)| read.most(1.) < f64::INFINITY)
}

/// What a read asks for where it does not read within the step's share (module documentation, # The step): the pairs
/// on which it could, as a factor on the pairs it was read on; none where it reads within. A look
/// reads within where its far end F = max(M̂, 0) + t·SE at one look is at most the square b of the step's share
/// (decide.rs `Read::outcome`). Where it does not, the pairs are asked for what they could do at the best: were the
/// look's error the least its read allows -- its near end L at one look, from zero: its own interval, as at its far
/// end --, what the read holds above L, value and interval, is the pairs' noise, and falls as the floor does, as 1/N.
/// The far end on N pairs is then L + (F − L)·N₀/N: within b from N = N₀ (F − L)/(b − L) on, on no number where L
/// itself is at b or above. The largest factor over the looks. (A value taken as it reads -- the rule before -- makes
/// a read that the noise has put above b ask for no pairs, and the rung below was taken on it: measured, SPEC §2.2.)
fn blind(own: &Between) -> Option<f64> {
    judged(own)
        .into_iter()
        .filter(|(read, bound)| read.outcome(*bound, 1.) != Outcome::Within)
        .map(|(read, bound)| {
            // A read that is no number, or whose spread is unread, excludes nothing: no pairs bring it within.
            if !read.most(1.).is_finite() {
                return f64::INFINITY;
            }
            let least = read.least(1.).max(0.);
            if least < bound { (read.most(1.) - least) / (bound - least) } else { f64::INFINITY }
        })
        .reduce(f64::max)
}

/// The pairs a rung's own read asks for where its estimate lies beyond the step's share but not clearly -- the noise
/// hides the difference (module documentation, # The step; `Verdict::Unseeing`): `factor`, the pairs it could come
/// within on at the best over the `have` it was read on (`blind`), wherever a run's trajectories can take them, a batch
/// at a time (`batch`: at most twice the pairs the read rests on) and read again; none where they cannot -- the rung
/// below is then read instead, or the check ends unable to see (SPEC §5.1). A rung below is no remedy for the noise:
/// it reads its own pair's, and where the pairs part early that is the two ensembles' own, the same at every step
/// (measured, R4_LIN_2: rows at far ends of 1.5 to 2.5 b at 0.43, 0.22, 0.11 and 0.054 µs alike).
fn remedy(factor: f64, have: u64) -> Option<u64> {
    let pairs = factor * have as f64;
    (pairs <= MOST_TRAJECTORIES as f64 && have < MOST_TRAJECTORIES).then(|| batch(pairs, have))
}

/// The step's part of an output's accounts (module documentation, # The step): on what is shown -- the largest of the
/// rows' and the loss's, a probability's error at most 1 -- and on the total T (relative, weighed by the held share, over
/// T's allowance at the output: the check reads T's error so, estimate.rs `between`, and the account takes it times the
/// output's allowance, `accounts`). The step's share until the
/// pilot's levels are read; then what the reads the rung stands by do not exclude, the rms over the run (`checked`,
/// `Levels::told`), the same at every output, unbounded on T where no output reads it at both levels.
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
}

/// The step's check as the result tells it (module documentation, # The step), of the reads the rung stands by
/// (`Levels::told`): the third level's step (µs) -- four times the run's; none where the rung stood on its own two
/// levels --, the pairs of the rung's own two levels, and the check's resolution -- the rms error over the run
/// that the read booked cannot tell from none, √(t·SE) of its mean square, the largest on what is shown (absolute) and
/// on T (a share of its allowance; unbounded where it is not read); and whether the check saw (the rung stands on its
/// estimates) or ended unable to see, capped or not.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Checked {
    third: Option<f64>,
    pairs: u64,
    resolution: [f64; 2],
    saw: bool,
}

/// The step's account as the pilot leaves it (module documentation, # The step), from the reads the rung stands by --
/// its own and its third level's, carried to its step (`Levels::told`) --, and whether the check `saw` (the rung
/// stands on its estimates): where it did, per look the larger estimate, √max(M̂, 0) (`worse`, `estimated`) -- the number the criterion
/// holds against the share's square, so what is booked and what stood the rung are one --; where it did not (at the
/// steps' cap, or unable to see), what the reads do not exclude, the far end (`unexcluded`). And the check's resolution,
/// one look's √(t·SE) of the read booked. A read that is no number -- no output counts a trajectory at both levels
/// (estimate.rs `between`) -- is unbounded, as T's is where no output reads it.
fn checked(own: &Between, third: Option<&Between>, saw: bool) -> (Bound, [f64; 2]) {
    let read = worse(own, third, !saw);
    let resolution = |read: &Read| read.spread.interval().sqrt();
    let shown = read.shown.iter().map(resolution).fold(resolution(&read.loss), f64::max);
    (if saw { estimated(&read) } else { unexcluded(&read) }, [shown, read.temperature.map_or(f64::INFINITY, |t| resolution(&t))])
}

/// What a read estimates: each quantity's √max(M̂, 0), the rms of its step error as read, the largest on what is shown,
/// and the total T's; unbounded where a read is no number.
fn estimated(read: &Between) -> Bound {
    let size = |read: &Read| if read.value.is_nan() { f64::INFINITY } else { read.value.max(0.).sqrt() };
    Bound { shown: probability_error(read.shown.iter().map(size).fold(size(&read.loss), f64::max)), temperature: read.temperature.map_or(f64::INFINITY, |t| size(&t)) }
}

/// The step's part of the accounts as far as a rung's reads hold it (module documentation, # The model; `keep`): per
/// look its estimate √max(M̂, 0) (`estimated`), at most the step's share. The check leaves the step it stands at within
/// its share on its estimates (`verdict`) -- a rung beyond is left, and the run stands at a rung below --, so the share
/// bounds the step's part the run will book, and a read below it is that part as read.
fn standing(read: &Between) -> Bound {
    let (read, share) = (estimated(read), Bound::share());
    Bound { shown: read.shown.min(share.shown), temperature: read.temperature.min(share.temperature) }
}

/// What a read does not exclude (decide.rs `Read::most`): each quantity's far end, the rms it leaves open,
/// √(max(M̂, 0) + t·SE) at one look's interval -- the number `blind` holds against the step's share --, the largest on
/// what is shown, and the total T's; unbounded where a read excludes nothing.
fn unexcluded(read: &Between) -> Bound {
    let far = |read: &Read| read.most(1.).sqrt();
    Bound { shown: probability_error(read.shown.iter().map(far).fold(far(&read.loss), f64::max)), temperature: read.temperature.map_or(f64::INFINITY, |t| far(&t)) }
}

/// What is shown of some groups: the estimates at the outputs, the listed outputs' atoms and the counts.
fn pooled(parts: &[&Part], scales: &Scales) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts) {
    let estimates = summarize(&outputs_of(parts), scales);
    let mut listed: Vec<Vec<(V3, V3)>> = vec![vec![]; parts.first().map_or(0, |p| p.listed.len())];
    for p in parts {
        for (l, x) in listed.iter_mut().zip(&p.listed) {
            l.extend_from_slice(x);
        }
    }
    (estimates, listed, counts_of(parts))
}

/// The counts of some groups.
fn counts_of(parts: &[&Part]) -> Counts {
    parts.iter().fold(Counts::default(), |mut counts, p| {
        counts.add(&p.counts);
        counts
    })
}

/// A message while the pilot's members come in: the result of what is in so far -- the parts in hand and those of the
/// members now running, each as a group of its own (the last result again where they are fewer than two groups,
/// `Messages::result`).
fn so_far(context: &Context, have: &[Part], more: &[&Part], messages: &mut Messages) {
    let all: Vec<&Part> = have.iter().chain(more.iter().copied()).collect();
    // Every trajectory is held at t = 0.
    let n = all.iter().map(|p| p.sums.sums[0].trajectories + p.sums.sums[0].failed).sum::<f64>() as u64;
    messages.result(all.len(), n, None, || {
        let (e, l, c) = pooled(&all, &context.scales);
        // The estimates at the analysis times the context holds, among the times its parts are read at.
        let at: Vec<usize> = context.at.iter().map(|t| context.plan.reads.iter().position(|r| r == t).expect("an analysis time is a time read")).collect();
        let reads = summarize(&all.iter().map(|p| p.reads.at(&at)).collect::<Vec<Group>>(), &context.scales);
        assemble(context, &e, &reads, &l, &c, n)
    });
}

/// The analysis times' candidates (module documentation, # The analysis times): the companion's step ends at the
/// attempt's first rung `rung`, m·2τ for m = 0 … M -- a step end of every level the attempt runs, its rungs' and their
/// companions' --, the last the run's end `end` (`first_rung`: the run is M of them).
fn grid(end: f64, rung: f64) -> Vec<f64> {
    let last = (end / (2. * rung)).round() as usize;
    (0..=last).map(|m| if m == last { end } else { (2 * m) as f64 * rung }).collect()
}

/// The analysis times (module documentation, # The analysis times; analysis.rs `analysis_times`) of some groups'
/// sums `fine` at the candidates `times`, with their companion's `coarse` where the read compares two levels (else
/// empty): the places of the times that stand, from the run's two ends -- the lights are switched on at t = 0, and the
/// request switches them nowhere else --, every interval read no farther apart than a quarter of T's shortest window
/// `window` (`Plan::window`).
fn analysed(times: &[f64], window: f64, fine: &[&Group], coarse: &[&Group], scales: &Scales) -> Vec<usize> {
    analysis_times(times, &[], window, &mut |k| readings(fine, coarse, k, scales))
}

/// The step's read between two levels of the same groups (`between`) at their analysis times (`analysed`, the two
/// levels' readings): the groups' sums there, each time weighed by its share of the run.
fn read_step(times: &[f64], window: f64, fine: &[&Part], coarse: &[&Part], scales: &Scales) -> Between {
    let (fine, coarse) = (reads_of(fine), reads_of(coarse));
    let at = analysed(times, window, &fine, &coarse, scales);
    let there = |groups: &[&Group]| -> Vec<Group> { groups.iter().map(|g| g.at(&at)).collect() };
    between(&there(&fine), &there(&coarse), scales, &at.iter().map(|&k| times[k]).collect::<Vec<_>>())
}

/// Of the bound atoms some groups hold at each of their times read (estimate.rs `Sums::bound_outside`: the leaving ones
/// are no draw of the distribution a reach is made for, module documentation, # The reach's atoms): how many lie outside
/// the reach, and how many there are.
fn outside(groups: &[Group]) -> Vec<(f64, f64)> {
    (0..groups.first().map_or(0, |g| g.sums.len()))
        .map(|k| groups.iter().map(|g| g.sums[k].bound_outside()).fold((0., 0.), |a, b| (a.0 + b.0, a.1 + b.1)))
        .collect()
}

/// The reach's falsifier (SPEC §1.8; decide.rs: a booked bound that a read can only refute). The reach is made for an
/// ensemble -- the bound atoms' at its temperature, of which it leaves out the share `left` -- and the model is judged
/// over it as holding them. The pilot reads the two things this says of its bound atoms (`motion`, estimate.rs
/// `bound_motion`; module documentation, # The reach's atoms: a leaving atom is no draw of the distribution a reach is
/// made for, and no reading of a temperature), at the run level's analysis times (module documentation, # The analysis
/// times), as one family of looks (Šidák over all of them, each one-sided at the confidence z):
/// - no lab axis's T at a time after t = 0 is above the largest kinetic temperature along an axis of that ensemble,
///   `reach` (µK; model.rs `Reach::kinetic_uk`): refuted by a T read above it beyond its interval;
/// - of the bound atoms held at a time, those outside the reach (`outside`: per time the count and the atoms) are a
///   binomial's of the share `left` at the most: refuted by a count whose tail (estimate.rs `at_least`) is below the
///   look's -- among the 101 looks of 26 times, two of the pilot's 88. This read estimates no spread. (Read at every
///   output for T and at the listed ones for the count, as it was, the family was the outputs' -- six hundred looks of
///   two hundred --, and the counts were of other times at another number of outputs.) T's interval over six hundred
///   looks was 6.4 standard errors, each estimated on eleven groups: a T twice the reach's passed it wherever that
///   estimate fell high, with a tenth of the atoms outside (measured, SPEC §1.8).
///
/// Either refutes; the reach's new temperature is then the largest T read. Where no T read is above `reach` the
/// atoms outside are no hotter ensemble's (a cloud driven out carries its energy in its mean motion): the falsifier
/// has no reach to offer them, and the reach stands.
fn outgrown(outside: &[(f64, f64)], left: f64, motion: &[Option<crate::estimate::Motion>], reach: f64) -> Option<f64> {
    let read: Vec<&crate::estimate::Motion> = motion.iter().skip(1).flatten().collect();
    let looks = (3 * read.len() + outside.len()) as f64;
    let hot = read.iter().any(|m| (0..3).any(|a| m.temperature[a] - m.temperature_errors[a].interval_over(looks) > reach));
    let tail = 0.5 * crate::estimate::tail_among(looks);
    let crowded = outside.iter().any(|&(count, atoms)| crate::estimate::at_least(count, atoms, left) < tail);
    (hot || crowded).then(|| hottest(motion)).filter(|t| *t > reach)
}

/// The temperature a reach grows to (SPEC §1.8): the largest T of the bound atoms (`bound_motion`) read along a lab
/// axis at a time; no number where none is read.
fn hottest(motion: &[Option<crate::estimate::Motion>]) -> f64 {
    motion.iter().skip(1).flatten().flat_map(|m| m.temperature).fold(f64::NEG_INFINITY, f64::max)
}

/// What the run's unresolved atoms and its atoms past the reach add to the model's error at every time (SPEC §1.8,
/// §4; module documentation, # The reach's atoms), on what is shown (absolute: held atoms, by the held share 1 − L, as
/// `Measured::means` books them) and on T (relative):
/// - the unresolved atoms, whole: their share of the held atoms on what is shown, of the kinetic energy on T
///   (estimate.rs `Estimate::unresolved`);
/// - the atoms past the reach, as the old engine booked them (`beyond_reach`), on both.
fn apart(estimates: &[Estimate], reach: f64, share: f64, cap: f64) -> Vec<[f64; 2]> {
    estimates
        .iter()
        .map(|e| {
            let beyond = beyond_reach(e, reach, share, cap);
            [(1. - e.loss) * (e.unresolved[0] + beyond), e.unresolved[1] + beyond]
        })
        .collect()
}

/// The share of a time's atoms past the reach (module documentation, # The reach's atoms): of a thermal ensemble at
/// the time's largest T along a lab axis -- at most `cap` (µK), the trap's depth over ln(1/s), past which an atom is
/// the loss's --, what lies past the reach's level beyond the share s (`share`) the reach leaves out at its own T_r
/// (`reach`, µK: the largest kinetic temperature along an axis of the ensemble it is made for), s^(T_r/T) − s. Zero
/// within the reach, or with no T read.
fn beyond_reach(e: &Estimate, reach: f64, share: f64, cap: f64) -> f64 {
    let t = e.motion.map_or(0., |m| m.temperature.iter().copied().fold(0., f64::max)).min(cap);
    if t > reach { share.powf(reach / t) - share } else { 0. }
}

/// Whether a lab axis's temperature (µK) lies in the validated regime of that axis's frequency (rad/µs), k_BT ≥
/// `CLASSICAL` ħω (SPEC §0, §3): what the result reports as classical, per axis; reporting only.
fn classical(temperature: f64, frequency: f64) -> bool {
    KT_PER_UK * temperature >= CLASSICAL * frequency
}

/// A state's model error as the run books it (`Measured::means`, `apart`): on the populations (absolute) and on T
/// (relative) what its units book there, of which the part a flow's rate error moves to its far side
/// (`UnitError::moved`, `into`; model/units.rs, # A rate's error on the populations) is the source side's as the
/// state sees it, and is taken against the time's populations (model/units.rs `destined`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Booking {
    error: [f64; 2],
    moved: f64,
    into: u32,
}
impl Booking {
    const NONE: Booking = Booking { error: [0.; 2], moved: 0., into: 0 };

    /// A unit's error at a state. One without bound or that is no number -- a fold read where it does not exist: a
    /// coupling across a distance of nothing, a drive at a state far outside its validity, the runaway's (# The reach's
    /// atoms) -- is the state's whole, as an unresolved atom is booked: every row of the atom (a probability's range) and
    /// its whole part of T, none of it a flow's.
    fn of(e: &UnitError) -> Booking {
        if [e.population, e.temperature, e.moved].iter().any(|x| !x.is_finite()) {
            return Booking { error: [1.; 2], moved: 0., into: 0 };
        }
        Booking { error: [e.population, e.temperature], moved: e.moved, into: e.into }
    }

    /// The error of a lost atom's fate (module documentation, # The model): a unit's error along its path, of which the
    /// run holds two states -- at t = 0 (`start`) and at its loss (`end`) --, their mean: the path's mean by the
    /// trapezoid on those two. Whole, none of it a flow's: what a flow moves between ground states is the held atoms'
    /// (`at`), and an atom lost errs by its fate.
    fn path(start: &Booking, end: &Booking) -> Booking {
        Booking { error: std::array::from_fn(|i| 0.5 * (start.error[i] + end.error[i])), moved: 0., into: 0 }
    }

    /// What is booked at a time of the held populations `p` (estimate.rs `Estimate::populations`).
    fn at(&self, p: &[f64; STATES]) -> [f64; 2] {
        let less = self.moved * (1. - destined(self.into, p));
        [self.error[0] - less, self.error[1] - less]
    }
}

/// How a folded unit's error is booked (module documentation, # The model: the units booked at their bounds): read at
/// the states -- its place among the units read --, or at its bound.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Booked {
    Read(usize),
    Bound(f64),
}

/// How each folded unit is booked, from the units' bounds at the states read, none where a unit's gate bounds no error
/// of its (model/units.rs `Model::bounds`): the bounded units taken by their bounds, the smallest first (equal ones in
/// the units' order), are booked at them while the bounds so booked add up to no more than the unseen share of
/// `MOST_TRAJECTORIES` draws, z²/(N + z²) -- the least statistics a solve ever books on what is shown; every other unit
/// is read. So the model's error at a time is at most that share above the one with every unit read, on what is
/// shown (absolute) and on T (relative) alike. A bound that is no number is no bound: it and every larger one are read.
fn booked(bounds: &[Option<f64>]) -> Vec<Booked> {
    let mut bounded: Vec<(f64, usize)> = bounds.iter().enumerate().filter_map(|(u, b)| b.map(|b| (b, u))).collect();
    bounded.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut unread: Vec<Option<f64>> = vec![None; bounds.len()];
    let mut sum = 0.;
    for (bound, u) in bounded {
        sum += bound;
        if !(sum <= unseen(MOST_TRAJECTORIES as f64)) {
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

/// The states the model is read at and the errors there of the units read of `units` (SPEC §5.2; module documentation,
/// # The model: where the atoms went), at the analysis times the groups `parts` are read at (`Part::at`). At every
/// analysis time after t = 0, of every group:
/// - every member that holds its atom there: the error of its rows, read at its state there;
/// - and every member lost by then: the error of its fate, read along its path at the two states the run holds of
///   it, at t = 0 and where it was found lost (adapt.rs, # The reads: its last state) -- their mean (`Booking::path`).
/// At t = 0 no model has acted (`model_error`): its states are read as a lost atom's path's start alone. A held atom's
/// state is read for its time alone, a lost atom's path once for every time after its loss: the reads are the table's
/// rows by number (`Sampled::row`). A row `known` already holds -- the same trajectory at the same time, or its path, at
/// the same step: the scout's inside the pilot -- keeps the errors read there, where it read every unit read here: a
/// unit's error is a function of the state.
///
/// Which units are read (`booked`) is decided on the states' own bounds: each unit's (model/units.rs `Model::bounds`)
/// at the largest speed of the states read -- every state's speed is within it, so a bound holds at each, as an atom's
/// that heated past the reach's speeds too.
///
/// The errors are one table (`Measured::errors`): the rows one after another, a row the units' read errors on the
/// populations and on T as each books them there (`Booking`), each row written on the pool where it is read. One block
/// of memory, not one a state: the states are thousands, and as many blocks took the calling thread 0.2 to 0.6 s to
/// free.
struct Sampled {
    /// Each row's trajectory, its time's bits (`PATH` for a lost atom's path) and its step.
    of: Vec<(u64, u64, f64)>,
    /// How each unit is booked (`booked`), in the units' order.
    booked: Vec<Booked>,
    errors: Vec<Booking>,
    /// Each read's analysis time (its place among them), its group, its row, and whether it reads a lost atom's fate.
    at: Vec<usize>,
    from: Vec<usize>,
    row: Vec<usize>,
    lost: Vec<bool>,
}
impl Sampled {
    /// The errors of the model's folded units as the decisions read them (`Measured`).
    fn measured(&self) -> Measured<'_> {
        Measured { errors: &self.errors, output: &self.at, group: &self.from, row: &self.row, lost: &self.lost, booked: &self.booked }
    }

    /// A row's length: the units read.
    fn width(&self) -> usize {
        read_units(&self.booked)
    }
}

/// The units read among those booked so (`booked`).
fn read_units(booked: &[Booked]) -> usize {
    booked.iter().filter(|b| matches!(b, Booked::Read(_))).count()
}
/// The time a lost atom's path's row stands at (`Sampled::of`): none of the run's.
const PATH: u64 = u64::MAX;
fn sampled(model: &Model, units: &[Unit], tables: &crate::model::tabulate::Tables, parts: &[&Part], known: Option<Sampled>) -> Sampled {
    // A row's states: a held atom's at a time, or a lost atom's at t = 0 and at its loss.
    let mut rows: Vec<(u64, u64, f64, &State, Option<&State>)> = vec![];
    let (mut at, mut from, mut row, mut lost) = (vec![], vec![], vec![], vec![]);
    for (g, p) in parts.iter().enumerate() {
        // The group's lost atoms, each a row: the first time it is lost at, its row.
        let mut paths: Vec<(usize, usize)> = vec![];
        for (first, last) in &p.lost {
            if let Some(start) = p.samples.first().and_then(|s| s.iter().find(|x| x.index == last.index)) {
                paths.push((*first, rows.len()));
                rows.push((last.index, PATH, last.tau, start, Some(last)));
            }
        }
        for (k, s) in p.samples.iter().enumerate().skip(1) {
            for state in s {
                (at.push(k), from.push(g), row.push(rows.len()), lost.push(false));
                rows.push((state.index, state.sample.at.t.to_bits(), state.tau, state, None));
            }
            for &(_, r) in paths.iter().filter(|(first, _)| *first <= k) {
                (at.push(k), from.push(g), row.push(r), lost.push(true));
            }
        }
    }
    // The units read and those booked at their bounds, at the largest speed of the states read.
    let speed = rows.iter().flat_map(|&(_, _, _, state, last)| std::iter::once(state).chain(last)).map(|s| norm3(s.sample.at.kappa)).fold(0., f64::max) * model.hbar_over_mass();
    let booked = booked(&model.bounds(units, speed));
    let reading: Vec<Unit> = units.iter().zip(&booked).filter(|of| matches!(of.1, Booked::Read(_))).map(|of| of.0.clone()).collect();
    // The place in a row of `known` of each unit read here, where it read them all.
    let places: Option<Vec<usize>> = known.as_ref().and_then(|k| booked.iter().zip(&k.booked).filter(|of| matches!(of.0, Booked::Read(_))).map(|of| if let Booked::Read(q) = *of.1 { Some(q) } else { None }).collect());
    let key = |index: u64, at: u64, tau: f64| (index, at, tau.to_bits());
    let read: HashMap<(u64, u64, u64), usize> = known.iter().filter(|_| places.is_some()).flat_map(|k| k.of.iter().enumerate().map(|(j, &(index, at, tau))| (key(index, at, tau), j))).collect();
    let width = reading.len();
    let mut errors = vec![Booking::NONE; rows.len() * width];
    let errors_at = |state: &State| model.errors_at(&reading, &state.sample, Some(tables)).iter().map(Booking::of).collect::<Vec<_>>();
    // A row (the table is empty where no unit is read).
    errors.par_chunks_mut(width.max(1)).zip(&rows).for_each(|(out, &(index, time, tau, state, last))| match (&known, &places, read.get(&key(index, time, tau))) {
        (Some(known), Some(places), Some(&j)) => {
            let row = &known.errors[j * known.width()..][..known.width()];
            out.iter_mut().zip(places).for_each(|(out, &q)| *out = row[q]);
        }
        _ => match last {
            None => out.copy_from_slice(&errors_at(state)),
            Some(last) => out.iter_mut().zip(errors_at(state)).zip(errors_at(last)).for_each(|((out, a), b)| *out = Booking::path(&a, &b)),
        },
    });
    Sampled { of: rows.iter().map(|&(index, time, tau, _, _)| (index, time, tau)).collect(), booked, errors, at, from, row, lost }
}

/// Whose reads a keep is judged on (`keep`): the scout's few, the pilot's, the product's statistics beside the pilot's
/// reads of the units, or the states an attempt before left (# An attempt's own), few where they are a scout's.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reads {
    Scout,
    Pilot,
    Product,
    Carried { few: bool },
}
impl Reads {
    /// Whether the reads are few, one state a group and time, judged at what they leave no doubt of (`Doubt`).
    fn few(self) -> bool {
        matches!(self, Reads::Scout | Reads::Carried { few: true })
    }
    /// Whether the work a keep saves is asked of them, beside the room (`keep`): on the scout and the pilot, whose keep
    /// runs again what its attempt ran. Not in the product -- the pilot priced the rest --, nor on carried states: their
    /// attempt runs its own pilot whatever it keeps, so every keep's re-run is nothing (`again` = 0) and any unit's
    /// smallest saving pays, every unit priced at every attempt's start (R2_RECOIL: 98 s of pricing, and one unit's
    /// kept model asked 5.7 GB); the work is the pilot's to price on the run's own reads.
    fn priced(self) -> bool {
        matches!(self, Reads::Scout | Reads::Pilot)
    }
}

/// The folded units to keep, if any (SPEC §5.2; module documentation, # The model), from the units' errors `measured`
/// at the states of the trajectories `parts` (their `estimates`; the units' mean errors over them `means`, per time
/// and unit) and the errors `fixed` beside them, the step's part `step` as far as it is read (`standing`; its share
/// before its reads are in), by one rule: of the sets the units' order leads through (`needed`), the one of the least
/// expected total work -- a set's kept model's work a µs times the trajectories it runs, its pilot again (`again`) and
/// its product beside the error it leaves --, against keeping nothing: the current model's work a µs times its
/// product's trajectories, where its looks have room for the statistics; none where some look has none (no number of
/// trajectories fits beside its errors). A set kept where nothing is an option saves beyond its saving's spread over
/// the pilot's groups (decide.rs `pays`; `Pricing::share_of`). Keeps are priced (`Reads::priced`) on the scout and
/// the pilot, and only where the most any keep could save pays beyond its spread (`Pricing::could_save`: every unit's
/// error removed, its pilot run again, at the current model's work a µs). On the scout, whose
/// reads are few, all of it is judged at the model's error they leave no doubt of (`Doubt`, `clear`): the room, and
/// the work the current model's error asks; the units taken to the far end of the doubt (`raised`). In the product
/// (`Reads::Product`) and on carried states the room alone is asked: the pilot has priced the rest, and the carried
/// states' attempt runs its own pilot whatever it keeps (every keep's re-run nothing, any saving would pay).
#[allow(clippy::too_many_arguments)]
fn keep(
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
    step: Bound,
    scales: &Scales,
    unsampled: [f64; 2],
    tables: &Tables,
    ladder: &Ladder,
    setup: Setup,
    again: f64,
    reads: Reads,
) -> Option<Vec<usize>> {
    let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
    // A unit weighs where it was read: a time with no state read carries every unit's bound alike, and tells none
    // of them apart. And where the units' errors crowd it: the share of trajectories not counted (estimate.rs, # What is
    // counted) is no unit's, and no keep takes it away.
    let mut read = vec![false; fixed.model.len()];
    measured.output.iter().for_each(|&k| read[k] = true);
    let counted: Vec<Estimate> = estimates.iter().map(|e| Estimate { uncounted: 0., ..e.clone() }).collect();
    // On the scout the model's error is judged at what its few reads leave no doubt of (`Doubt`): the room and the
    // pricing alike -- a keep is never undone, and the reads that put the room's error high put the work the current
    // model asks high with them; the pilot reads the same units on all its states.
    let doubt = || Doubt::new(measured, fixed.model.len(), parts, scales, unsampled);
    let scouted = reads.few().then(doubt);
    let near = scouted.as_ref().map_or_else(|| vec![[0.; 2]; fixed.model.len()], |d| d.interval(&[]));
    let mut pricing = Pricing::new(Formula::new(model, kept, units, reference, centre, tables, *ladder, setup), measured, means, parts, estimates, fixed, step, scales, unsampled, again, &near);
    // Keeping nothing: no option where some look has no room for the statistics; else the current model's work a µs
    // times its product's trajectories, where some keep could save at all.
    let crowded = !weighing(&counted, &clear(fixed, &near), step, means, &held, &read).is_empty();
    // Keeping nothing: no option where some look has no room for the statistics; else the current model's work a µs
    // times its product's trajectories -- where the most any keep could save pays (`could_save`): the jackknife's
    // levels are read for that, and for the set's saving after.
    let mut others = vec![];
    let none = if crowded {
        None
    } else {
        if !reads.priced() {
            return None;
        }
        pricing.own = pricing.own();
        others = pricing.leave_one_out();
        if !pays(0., pricing.could_save(&others)) {
            return None;
        }
        Some(pricing.mean(&pricing.own, None) * pricing.all.need)
    };
    let doubts = scouted.unwrap_or_else(doubt);
    let far = |set: &[usize]| raised(&without(fixed, means, &read, set), &doubts.interval(set));
    // A kept set's work a µs (its mean rate, made once a set) and the bound below every set's (`Pricing::least`); the
    // product's trajectories beside the error it leaves (`Level::need_without_all`), and the chance that the next
    // pilot reads a look without room (`Doubt::chance`).
    let rates: std::cell::RefCell<HashMap<Vec<usize>, Option<[f64; 2]>>> = std::cell::RefCell::new(HashMap::new());
    let rate = |set: &[usize]| -> Option<[f64; 2]> {
        let mut key = set.to_vec();
        key.sort_unstable();
        if let Some(&made) = rates.borrow().get(&key) {
            return made;
        }
        let made = pricing.rates(&key).map(|(r, setup)| [pricing.mean(&r, None), setup]);
        rates.borrow_mut().insert(key, made);
        made
    };
    let price = |set: &[usize], beyond: &[usize]| -> [f64; 2] {
        let point = without(fixed, means, &read, set);
        [pricing.all.need_without_all(set), doubts.chance(set, &point, &counted, step, &read, beyond)]
    };
    let costs = Costs { least: pricing.least(), floor: pricing.all.need_alone(), again: pricing.again, none };
    let set = needed(&counted, step, means, &held, &read, &costs, &rate, &|set| without(fixed, means, &read, set), &far, &price);
    if set.is_empty() {
        return None;
    }
    if none.is_some() {
        let (with, _) = pricing.rates(&set)?;
        let saving = Read { value: pricing.share_of(&pricing.all, pricing.all.need_without_all(&set), &with), spread: spread(others.iter().map(|l| pricing.share_of(l, l.need_without_all(&set), &with)), others.len() as f64) };
        if !pays(0., saving) {
            return None;
        }
    }
    Some(set)
}

/// One attempt at the solve (module documentation, # The pipeline) with the model of the units `kept` as `setting` holds
/// it, over the reach of the temperature `reach` µK. `carried`: the states the attempt before left, in this model
/// (# An attempt's own) -- decided on first.
fn attempt(request: &Request, setting: &Setting, kept: &[Unit], reach: f64, carried: Option<Carried>, messages: &mut Messages) -> Result<Attempt, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let Setting { model, reference, units, trap, ensemble, depth, .. } = *setting;
    // The reach (model/reach.rs): the ensemble at its temperature, of which it leaves out the share the most trajectories
    // of a solve leave unseen.
    let share = unseen(MOST_TRAJECTORIES as f64);
    let judged = &model.reach(trap, &reference, reach, share);
    let fastest = judged.speed;
    let entries = &setting.entries;
    // The ensemble's proposals so far that read deeper than their well's bottom: the attempts' before this one.
    let deeper = ensemble.deeper();
    // What the reach leaves out of the initial ensemble (model/reach.rs `Model::left_out`): of the atoms and of their
    // kinetic energy, by the ensemble's quadrature of its own density -- no atom is drawn --, a level of it with no more
    // evaluations of the potential than the most trajectories a solve runs have initial atoms. It and the kick's
    // curvature over the reach (`admissible`), of the beams' largest moduli over its box, ask nothing of one another:
    // side by side on the pool.
    let integrated = ensemble.evaluations().integrated;
    let (unsampled, kick) = rayon::join(|| model.left_out(judged, ensemble, MOST_TRAJECTORIES), || model.kick_curvature(judged, &model.moduli(judged), trap));
    // A keep's model's setup as this attempt's counts it (`Setup`), its draws' proposals read when the keep is asked.
    let setup = {
        let left_out = ensemble.evaluations().integrated - integrated;
        move || Setup::of(ensemble.evaluations(), left_out)
    };
    let (report, unheld) = (setting.reported.as_ref().unwrap_or(trap), setting.reported.is_none());
    let scales = Scales { hbar_over_mass: hm, frequencies: report.lab };
    let plan = Plan::new(&request.times_us, report.lab);
    let row: [f64; STATES] = std::array::from_fn(|s| request.initial_density[(s, s)].re);
    let mut ledger = Ledger::default();
    ledger.note = Some(format!("初始系综：每个初态分量的原子束缚在它自己的势阱里（降低的等温分布 f ∝ e^(−E/k_BT) − 1，只含束缚原子，阱深比 k_BT 大得多时即热分布；每个主轴再乘谐振子的 Wigner 因子，T = {} μK）；布居最大的初态的阱深为 {:.1} k_BT", request.temperature_uk, depth));
    if unheld {
        ledger.flag("计算所用光场中的势没有极小：阱频、n̄、温度的窗口（一个阱周期）与验证范围都按初始势阱给出".into());
    }
    // The ladder's first rung, the attempt's first step (# An attempt's own): within the physics prior's step, and
    // admissible over the reach at its largest speed -- every speed of it the reach's, at the mean kinetic temperature
    // of the ensemble the reach is made for (model.rs `Reach::kinetic_uk`).
    let kinetic = judged.kinetic_uk.iter().sum::<f64>() / 3.;
    let (frequency, typical, end) = (trap.frequencies.iter().copied().fold(0., f64::max), typical_speed(hm, kinetic, fastest), plan.end);
    let ladder = Ladder { run: end, prior: prior(model.carrier_wave_number(), hm, kinetic, &scales), frequency, kick, fastest, typical };
    let first_rung = ladder
        .first(entries)
        .ok_or_else(|| format!("找不到可用的时间步长：在每条轨迹的步数上限（{MOST_STEPS} 步）之内，没有步长能分辨阱中的振荡、光场中势阱的曲率和光场沿原子路径的变化（或者阱频、曲率、原子速率不是有限数）"))?;
    // What the reach's falsifier compares the T read with: the largest kinetic temperature along an axis of the
    // ensemble the reach was made for (model.rs `Reach::kinetic_uk`: T_r where the trap is classical, the zero point's
    // where it is cold) -- no lab axis's T is above it while the atoms are that ensemble.
    let held = judged.kinetic_uk.iter().copied().fold(reach, f64::max);
    // And the share of its atoms it takes to lie outside the reach: the share the reach is made to leave out of its own
    // ensemble, and no less than what it is known to leave out of the initial one.
    let left = unsampled[0].max(share);
    // What the atoms past the reach add to the model's error (`apart`): the trap's depth (µK) over ln(1/s) at most.
    let cap = depth * request.temperature_uk / share.recip().ln();
    let beyond = |e: &[Estimate]| apart(e, held, share, cap);
    // Whether the rung above a step is a rung of the ladder (`first_rung`'s test at twice the step): the check then
    // reads that rung on a third level, at 4τ (`settle`): a cell of the Brownian path.
    let above = |tau: f64| 4. * tau <= end && admissible(2. * tau, frequency, kick, entries, fastest) && admissible(4. * tau, frequency, kick, entries, typical);
    // What the attempt's steps ask of the model's domain (`Domain`): the fastest decay of the model's excited states, no
    // state of the atom's faster.
    let widest = (0..model.decay.rows).map(|x| model.decay[(x, x)].re).fold(0., f64::max);
    let domain = Domain { model, reach: judged, hbar_over_mass: hm, sampled: entries, widest, result: true };
    // The analysis times' candidates (# The analysis times): every level of the attempt is read there while the pilot
    // runs.
    let (candidates, window) = (grid(end, first_rung), plan.window);
    let mut context = Context {
        at: vec![0., end],
        fixed: Fixed { model: vec![[0.; 2]; 2] },
        outside: vec![[0.; 2]; 2],
        step: Bound::share(),
        checked: None,
        tau: first_rung,
        plan: Plan { cell: if above(first_rung) { 4. } else { 2. } * first_rung, ..plan },
        scales,
        initial: row,
        ledger,
        method: format!("三维半经典运动：{}", model.describe().trim_end_matches('。')),
    };
    context.plan.read_at(candidates.clone());

    // The near light's tables (SPEC §2.3, §5.4; model/tabulate.rs): refined over the reach to the largest ensemble's
    // resolution, as far as their functions' structure asks (model/tabulate.rs, # Refinement). The work's bound decides
    // which coordinates are axes -- no first grid of more nodes than the middles the pilot takes at the attempt's first
    // step and at its double, pilot_size·(end/τ + end/(2τ)), a node counted as a middle without a table --, and is the
    // ceiling no grid passes; how far an axis is refined below it is the table's (its reads), not the run's. What a
    // coordinate held back misses is read at the atoms below.
    let most = (pilot_size() as f64 * (context.plan.end / context.tau + context.plan.end / (2. * context.tau))).ceil() as usize;
    let mut tables = model.tabulate(judged, most);
    let run = Run::tabulated(model, &tables, reference);

    // The pilot's groups (module documentation, # The pipeline): the run's first trajectories, at two levels.
    let first = MOST_TRAJECTORIES / GROUP;
    let numbers: Vec<u64> = (first..first + pilot_size() / GROUP).collect();
    let scales = context.scales;
    // A run level's analysis times (# The analysis times: its own readings), and whether its atoms have outgrown the
    // reach there (`outgrown`) -- the temperature it then grows to, with those times: the attempt ends there, and
    // nothing further is spent on a step of that reach (`own_level`, `check`).
    let analysed_alone = |level: &[&Part]| analysed(&candidates, window, &reads_of(level), &[], &scales);
    let outgrown_at = |level: &[&Part], at: &[usize]| {
        let here: Vec<Group> = level.iter().map(|p| p.reads.at(at)).collect();
        outgrown(&outside(&here), left, &bound_motion(&here, &scales), held)
    };
    let grown = |level: &[&Part]| -> Option<(f64, Vec<usize>)> {
        let at = analysed_alone(level);
        outgrown_at(level, &at).map(|t| (t, at))
    };
    let mut levels = Levels::default();

    // The model's decision at some groups' states before the pilot's own (`keep`; module documentation, # The model),
    // at their analysis times (`Part::at`), the step's error at its share -- its own reads are not in --: the units to
    // keep, and the states' read (`Sampled`). `again`: the trajectories a keep runs again of what this attempt has run.
    let decided = |parts: &[&Part], again: f64, reads: Reads| -> (Option<Vec<Unit>>, Sampled) {
        let estimates = summarize(&reads_of(parts), &scales);
        let times = estimates.len();
        let read = sampled(model, units, &tables, parts, None);
        let measured = read.measured();
        let means = measured.means(times, None, &estimates);
        let fixed = Fixed { model: model_error(&measured.combined(times, None, &estimates), unsampled) };
        let keeps = keep(kept, model, reference, trap.centre, units, &measured, &means, parts, &estimates, &fixed, Bound::share(), &scales, unsampled, &tables, &ladder, setup(), again, reads);
        (keeps.map(|u| u.iter().map(|&u| units[u].clone()).collect()), read)
    };
    // The states the attempt before left (# An attempt's own), at their own analysis times: decided on before a
    // trajectory runs, for the room (the work is this attempt's pilot's to price, `Reads::priced`). A keep they ask ends
    // the attempt there, handing the same states on, so that a model is settled on them, keep after keep, before its
    // own atoms run; none asked stands for the scout's keep.
    let decided_before = match carried {
        Some(carried) => {
            let parts: Vec<&Part> = carried.parts.iter().collect();
            let (asked, _) = decided(&parts, 0., Reads::Carried { few: carried.few });
            drop(parts);
            if let Some(units) = asked {
                return Ok(Attempt::Keep(units, carried));
            }
            true
        }
        None => false,
    };

    // The scout (module documentation, # The pipeline): the first member of every pilot group at both levels, screened;
    // where a unit is to be kept at its states, the attempt ends here -- for the price of the scout, not of a pilot.
    // The pilot's read of the same below stands for the result.
    settle(&run, ensemble, &domain, &mut context, &mut levels, &numbers, 0..1, &above, &|pilot| grown(pilot).is_some(), None, false, messages);
    // The scout's keep first, at the scout's analysis times: a model whose units' errors measured at the scout's states
    // leave a look no room is no model of those atoms, and what they read of the reach is that model's (with nothing
    // kept, PGC3D's and TET's scouts heated their atoms to 153 and 198 µK, and the reach grown for them stayed: 1.7 and
    // 4.2 times the released line's CPU).
    let scouted = if decided_before {
        None
    } else {
        let scout: Vec<&Part> = levels.fine.iter().collect();
        let here = parts_at(&scout, &analysed_alone(&scout));
        let (asked, read) = decided(&here.iter().collect::<Vec<_>>(), scout.len() as f64, Reads::Scout);
        drop(scout);
        if let Some(units) = asked {
            return Ok(Attempt::Keep(units, Carried { parts: here, few: true }));
        }
        Some(read)
    };
    // The pilot's own level is run before its companion and read by the reach's falsifier as it comes in (`own_level`):
    // where its atoms refute the reach, the attempt ends with the reach grown to the temperature they read, and the
    // rest of the pilot -- its companion, the step's check at a model judged over a reach the run does not keep -- is
    // never run. Where the scout's atoms already refute it, its members double a hand-out; else they are one hand-out.
    // The pilot's check goes on from that level, its companion run then (`settle`, `ran`).
    let doubling = grown(&levels.fine.iter().collect::<Vec<_>>()).is_some();
    let ran = match own_level(&run, ensemble, &domain, &context, &levels.fine, &numbers, doubling, &grown, messages) {
        Ok(own) => own,
        Err((t, states)) => return Ok(Attempt::Grow(t, Carried { parts: states, few: false })),
    };
    // The pilot: its groups' other members at both levels, screened, and the step's check (module documentation,
    // # The step) -- first as far as its own two levels: at the first rung whose own read is in and not clearly beyond,
    // the decisions the pilot's reads hold are taken (# Each decision as its evidence is in), and the check goes on
    // past them only where they leave the attempt standing. Whether it saw: a step that stands uncertified -- at the
    // steps' cap, or unable to see -- is flagged.
    let mut settled = settle(&run, ensemble, &domain, &mut context, &mut levels, &numbers, 1..GROUP, &above, &|pilot| grown(pilot).is_some(), Some(ran), true, messages);
    drop(run);
    let mut scouted = scouted;
    // The pilot's reads at its run level's analysis times, made at the rung they were made at; made again only where
    // the check went on to another rung.
    let mut made: Option<(f64, Vec<usize>, Vec<Part>, PilotRead)> = None;
    let (at, pilot, read) = loop {
        let tau = context.tau;
        let (at, pilot, read) = match made.take() {
            Some((made_at, at, pilot, read)) if made_at == tau => (at, pilot, read),
            _ => {
                let level: Vec<&Part> = levels.fine.iter().collect();
                let at = analysed_alone(&level);
                if let Some(t) = outgrown_at(&level, &at) {
                    return Ok(Attempt::Grow(t, Carried { parts: parts_at(&level, &at), few: false }));
                }
                let pilot = parts_at(&level, &at);
                let read = pilot_reads(model, units, &tables, &pilot.iter().collect::<Vec<_>>(), scouted.take(), unsampled, most, &scales);
                (at, pilot, read)
            }
        };
        let fixed = Fixed { model: read.model.clone() };
        let measured = read.measured();
        let means = measured.means(at.len(), None, &read.estimates);
        // The step's part as far as the rung's own reads hold it (`standing`; # The model).
        let step = {
            let (own, third, _, _) = levels.told(&candidates, window, tau, &scales);
            standing(&worse(&own, third.as_ref(), false))
        };
        // Units to keep at the pilot's states (`keep`): their model then decides on these states first.
        let refs: Vec<&Part> = pilot.iter().collect();
        let asked = keep(kept, model, reference, trap.centre, units, &measured, &means, &refs, &read.estimates, &fixed, step, &scales, unsampled, read.refined.as_ref().unwrap_or(&tables), &ladder, setup(), pilot_size() as f64, Reads::Pilot);
        drop(refs);
        if let Some(u) = asked {
            return Ok(Attempt::Keep(u.iter().map(|&u| units[u].clone()).collect(), Carried { parts: pilot, few: false }));
        }
        if !matches!(settled, Settled::Pending) {
            break (at, pilot, read);
        }
        made = Some((tau, at, pilot, read));
        let run = Run::tabulated(model, &tables, reference);
        settled = check(&run, ensemble, &domain, &mut context, &mut levels, &numbers, GROUP, &above, &|pilot| grown(pilot).is_some(), false, messages);
    };
    let saw = match settled {
        Settled::Capped => {
            context.ledger.flag(capped());
            false
        }
        Settled::Unseen => {
            context.ledger.flag(unseen_at_the_end());
            false
        }
        _ => true,
    };
    let mut read = read;
    if let Some(refined) = read.refined.take() {
        tables = refined;
    }
    let tau = context.tau;
    // The step's check (module documentation, # The step): what the reads the rung stands by do not exclude into the
    // accounts, their resolution into the result's words (`Levels::told`, `checked`).
    let (own, third, step_of_third, pairs) = levels.told(&candidates, window, tau, &scales);
    let (step, resolution) = checked(&own, third.as_ref(), saw);
    (context.step, context.checked) = (step, Some(Checked { third: step_of_third, pairs, resolution, saw }));
    // The decisions stand on the pilot's analysis times from here on: the product reads its trajectories there, and
    // the run's model error is the pilot's there (# The analysis times). The levels' states at the candidates are
    // read no more; the further pairs' groups are the product's, at the same times.
    let times: Vec<f64> = at.iter().map(|&k| candidates[k]).collect();
    let further: Vec<Part> = levels.further[0].iter().map(|p| p.at(&at)).collect();
    drop(levels);
    context.plan.read_at(times.clone());
    let n = times.len();
    context.at = times;
    context.outside = vec![[0.; 2]; n];
    // The pilot's groups with their states: the units were read there, and a keep's price in the product (`keep`).
    let pilot_refs: Vec<&Part> = pilot.iter().collect();
    let (estimates, counts) = (&read.estimates, &read.counts);
    context.fixed.model = read.model.clone();
    let measured = read.measured();
    let means = measured.means(n, None, estimates);
    for (t, table) in tables.tables.iter().enumerate() {
        let label = format!(
            "近共振光的插值表（{} 所在的子空间{}）",
            model.state_label(model.clusters()[table.layout.cluster].states[0]),
            if table.layout.lights == model.running() { "" } else { "，光镊关闭时" }
        );
        context.ledger.item(&label, read.tables[t][0].max(read.tables[t][1]), TABLE_MEASURED);
        match read.stuck[t] {
            Some(Stuck::Order) => context.ledger.flag(format!("{label}的误差超出误差预算中分给插值表的部分，而它在已分辨的网格上加密后误差下降不超过未分辨结构的两倍（不收敛），或再加密一次就超过节点数上限（{most} 个）：按实测的误差计入误差预算")),
            Some(Stuck::Held) => context.ledger.flag(format!("{label}的误差超出误差预算中分给插值表的部分，而超出的部分来自取焦点处的值的坐标，不是插值的误差（加密网格不能降低它）：按实测的误差计入误差预算")),
            Some(Stuck::Axes) => context.ledger.flag(format!("{label}的误差超出误差预算中分给插值表的部分，而它没有可以加密的坐标（它依赖的量都取焦点处的值）：按实测的误差计入误差预算")),
            None => {}
        }
    }
    // The optical sector's noise (ledger.rs `optical`), its items; its booking on T is in the model's error the reads
    // made (`pilot_reads`).
    crate::ledger::optical(&mut context.ledger, counts.optical);
    // Units to keep in the product (`keep`), read with the product's estimates and the step's part as the pilot booked it.
    let keeping = |estimates: &[Estimate], fixed: &Fixed, step: Bound| keep(kept, model, reference, trap.centre, units, &measured, &means, &pilot_refs, estimates, fixed, step, &scales, unsampled, &tables, &ladder, setup(), pilot_size() as f64, Reads::Product);
    {
        // The ledger's items: each folded unit's error on what is shown and on T (its largest over the analysis times,
        // weighed by the pilot's held share as the totals are; a time where the pilot holds no atom weighs nothing: its
        // units' errors are their bounds, `Measured::means`) with how it was booked (`booked`), and the unsampled share.
        let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
        let mut largest = vec![0.0_f64; units.len()];
        for k in (1..n).filter(|&k| held[k] > 0.) {
            for (most, m) in largest.iter_mut().zip(&means[k]) {
                *most = most.max(m[0].max(held[k] * m[1]));
            }
        }
        for ((unit, most), how) in units.iter().zip(largest).zip(measured.booked) {
            context.ledger.item(&model.label(unit), most, if matches!(how, Booked::Read(_)) { FOLD_MEASURED } else { FOLD_BOUNDED });
        }
        context.ledger.item("初始时就在估计的活动范围之外的原子所占的比例", unsampled[0], "对初始分布数值积分得到（含积分最后两级之差），计入布居的模型误差");
        context.ledger.item("初始时就在估计的活动范围之外的原子占初始动能的比例", unsampled[1], "对初始分布数值积分得到（含积分最后两级之差），计入温度的模型误差");
    }
    let run = Run::tabulated(model, &tables, reference);

    // The product (module documentation, # Stopping): on from the pilot's trajectories at the run's step, the run's
    // first; the pilot's companions have done their part, and its states have been read.
    let mut parts = pilot.clone();
    forget(&mut parts);
    parts.extend(further);
    let booked = context.fixed.clone();
    let (e, l, c, reads) = match product(&run, ensemble, &domain, &mut context, &mut parts, &numbers, &beyond, &keeping, messages) {
        Ok(product) => product,
        Err(u) => return Ok(Attempt::Keep(u.iter().map(|&u| units[u].clone()).collect(), Carried { parts: pilot, few: false })),
    };
    // The reach outgrown as far as the product reads (# The reach's atoms): some look has no room for the statistics
    // with what the atoms past the reach add, and has room without it. The reach grows to the temperature read where
    // that is above it, and the next attempt begins; else the account stands as booked, flagged.
    if looks(&reads, &context.fixed, context.step).iter().zip(&looks(&reads, &booked, context.step)).any(|(with, without)| !with.fits() && without.fits()) {
        if let Some(t) = Some(hottest(&bound_motion(&reads_of(&parts.iter().collect::<Vec<&Part>>()), &context.scales))).filter(|t| *t > held) {
            return Ok(Attempt::Grow(t, Carried { parts: pilot, few: false }));
        }
    }
    // A point of the prepared potential below the bottom the ensemble was drawn about (ensemble.rs, # Its draw).
    if ensemble.deeper() > deeper {
        context.ledger.flag("初态的势在所取的阱底之外还有更深的地方：初始系综只取到该阱底的阱深为止".into());
    }
    Ok(Attempt::Done(assemble(&context, &e, &reads, &l, &c, parts.len() as u64 * GROUP)))
}

/// The pilot's reads at the rung it stands at (`attempt`, `pilot_reads`): its estimates and counts, the units' errors
/// at its states (`Sampled`), the model's error per time -- the units',
/// the tables' and the optical sector's booking on T --, each table's booked error and why its refinement ended where it
/// is beyond its shares; and the tables refined, where a read refined one (the pilot's own stay as its levels ran on
/// them: a check that goes on reads its further levels on the tables of the levels it compares them with).
struct PilotRead {
    estimates: Vec<Estimate>,
    counts: Counts,
    sampled: Sampled,
    model: Vec<[f64; 2]>,
    tables: Vec<[f64; 2]>,
    stuck: Vec<Option<Stuck>>,
    refined: Option<Tables>,
}
impl PilotRead {
    /// The errors of the model's folded units as the decisions read them (`Measured`).
    fn measured(&self) -> Measured<'_> {
        self.sampled.measured()
    }
}

/// The pilot's reads (`PilotRead`) of its groups `pilot`, the folded units `units` read at its states (the scout's
/// `known` kept), on the pilot's tables `tables`, refined where their reads ask -- a copy, made at the first refinement
/// (`PilotRead::refined`).
/// - The model's errors at the pilot's states (SPEC §5.2): each folded unit's per time, the held atoms' mean over the
///   first member of every group -- one a group keeps every group in the mean's spread, and a unit's error is a function
///   of the state, read as well on eleven states a time as on eighty-eight.
/// - The tables' errors (SPEC §5.4): each table's whole error against the elimination itself, its mean over the run's
///   held states read at one state of every pilot trajectory (`one_each`, `over_the_run`) -- a state on another cluster
///   reads none of it. A table beyond the tables' shares has the axis of its largest read halved and is read again,
///   until it is within them, or it has no axis to halve (its coordinates all held at the focus: model/tabulate.rs,
///   # Axes), or what is beyond is not its axes' interpolation -- the whole less every axis's own read still beyond the
///   shares: a coordinate held at the focus, which no halving lowers --, or its worst axis's read on a grid it resolves
///   fell by no more than an unresolved peak's (model/tabulate.rs, # Refinement: it is not converging); what it reads
///   then is booked on every time, and flagged where it still lies beyond -- with which of these ended its refinement
///   (`Stuck`).
/// - The optical sector's noise (ledger.rs `optical`): its booking on T at every time, a part of the model's error
///   that no unit kept takes away.
#[allow(clippy::too_many_arguments)]
fn pilot_reads(model: &Model, units: &[Unit], tables: &Tables, pilot: &[&Part], known: Option<Sampled>, unsampled: [f64; 2], most: usize, scales: &Scales) -> PilotRead {
    let (estimates, counts) = (summarize(&reads_of(pilot), scales), counts_of(pilot));
    let sampled = sampled(model, units, tables, pilot, known);
    let measured = sampled.measured();
    let mut error = model_error(&measured.combined(estimates.len(), None, &estimates), unsampled);
    let read_at = one_each(pilot);
    let (shown_share, temperature_share) = crate::model::tabulate::shares();
    let mut stuck: Vec<Option<Stuck>> = vec![None; tables.tables.len()];
    let mut before: Vec<Vec<Option<f64>>> = tables.tables.iter().map(|table| vec![None; table.axes()]).collect();
    let mut refined: Option<Tables> = None;
    let booked = loop {
        let current = refined.as_ref().unwrap_or(tables);
        let reads: Vec<Vec<crate::model::tabulate::TableRead>> = read_at.par_iter().map(|pick| model.table_reads(current, std::slice::from_ref(&pick.state.sample))).collect();
        let mean = |f: &dyn Fn(&crate::model::tabulate::TableRead) -> f64, t: usize| over_the_run(&read_at, reads.iter().map(|r| f(&r[t])), None);
        let whole: Vec<[f64; 2]> = (0..current.tables.len()).map(|t| [mean(&|r| r.whole[0].population, t), mean(&|r| r.whole[0].temperature, t)]).collect();
        let beyond = (0..whole.len()).find(|&t| stuck[t].is_none() && (whole[t][0] > shown_share || whole[t][1] > temperature_share));
        let Some(t) = beyond else { break whole };
        // Each axis's read over the shares, on what is shown and on T.
        let axes: Vec<[f64; 2]> = (0..current.tables[t].axes()).map(|f| [mean(&|r| r.axes[f][0].population, t) / shown_share, mean(&|r| r.axes[f][0].temperature, t) / temperature_share]).collect();
        let held = (0..2).any(|k| whole[t][k] / [shown_share, temperature_share][k] - axes.iter().map(|a| a[k]).sum::<f64>() > 1.);
        let worst = (0..axes.len()).max_by(|&a, &b| (axes[a][0] + axes[a][1]).total_cmp(&(axes[b][0] + axes[b][1])));
        let finer = match worst {
            None => Err(Stuck::Axes),
            Some(_) if held => Err(Stuck::Held),
            Some(f) => model.halved_by(&current.tables[t], f, axes[f][0].max(axes[f][1]), &mut before[t][f], most).ok_or(Stuck::Order),
        };
        match finer {
            Ok(finer) => refined.get_or_insert_with(|| tables.clone()).tables[t].table = finer,
            Err(why) => stuck[t] = Some(why),
        }
    };
    // A table's error is read at the held atoms' states: on what is shown, by their share (`Measured::means`).
    for table in &booked {
        for (e, x) in error.iter_mut().zip(&estimates).skip(1) {
            *e = [probability_error(e[0] + (1. - x.loss) * table[0]), e[1] + table[1]];
        }
    }
    let optical = crate::ledger::optical(&mut Ledger::default(), counts.optical);
    for e in error.iter_mut().skip(1) {
        e[1] += optical;
    }
    PilotRead { estimates, counts, sampled, model: error, tables: booked, stuck, refined }
}

/// Why a table beyond the tables' shares is refined no further (`attempt`): its worst axis's read on a grid it resolves
/// fell by no more than an unresolved peak's (it is not converging: model/tabulate.rs, # Refinement), or its halved grid
/// would pass the work bound; what is
/// beyond is the coordinates it holds at the focus, not its axes' interpolation; or the table has no axis -- nothing to
/// halve.
#[derive(Clone, Copy)]
enum Stuck {
    Order,
    Held,
    Axes,
}

/// The flag of a step whose error lies clearly beyond what it may take at the steps' cap (module documentation, # The
/// step): what its reads do not exclude is booked.
fn capped() -> String {
    format!("时间步长的误差明显超出误差预算中分给它的部分，而每条轨迹的步数已达上限 {MOST_STEPS}，步长无法再减半：按检查不能排除的最大误差计入误差预算")
}

/// The flag of a step whose check ended unable to see (module documentation, # The step; SPEC §5.1): no number of
/// pairs within a run's trajectories and no rung of the ladder is predicted to certify it; what its reads do not
/// exclude is booked.
fn unseen_at_the_end() -> String {
    format!("步长收敛检查没有结论（三个步长的比较既没有明显超出误差预算中分给步长的部分，也不能确定在其内且误差随步长按二次方变化），而增加成对的先导轨迹（上限 {MOST_TRAJECTORIES} 条）和把步长再减半（每条轨迹上限 {MOST_STEPS} 步）都已不可能或不再有用：按检查不能排除的最大误差计入误差预算")
}

/// The pilot's two levels (module documentation, # The step): its groups at the step τ and the same groups, in the same
/// order, at 2τ; the further pairs a check took (`settle`), at the two levels: the groups the product runs next, by
/// their place in its list. The rung's third level above, `above`: the read between 2τ and 4τ carried to τ -- the
/// rung left above it, read on the pilot's pairs where the check came down from it there (`above_pilot`), on the
/// scout's where the scout did, or the pilot's trajectories at 4τ (`settle`); and whether the rung above the rung in
/// hand was read and is unread (`above_unread`: its read excludes nothing on some look).
#[derive(Default)]
struct Levels {
    fine: Vec<Part>,
    coarse: Vec<Part>,
    further: [Vec<Part>; 2],
    above: Option<Between>,
    above_pilot: bool,
    above_unread: bool,
}
impl Levels {
    /// The finer level's groups and their companions: the pilot's, then the further pairs'.
    fn paired(&self) -> (Vec<&Part>, Vec<&Part>) {
        (self.fine.iter().chain(&self.further[0]).collect(), self.coarse.iter().chain(&self.further[1]).collect())
    }

    /// The reads the check of the rung in hand, at the step `tau`, is told and booked by (`checked`, `Checked`): its
    /// own, between τ and 2τ on all its pairs; its third level's carried to τ (`carried`), the level above's -- none
    /// where it has none --; the third level's step, and the own read's pairs.
    ///
    /// Two levels give a difference, and an error only by what is known of the other level. With e(τ) = Q(τ) − Q(0)
    /// a quantity's error at an output:
    /// - against 2τ, d₂ = Q(2τ) − Q(τ) = e(2τ) − e(τ). The order between the two levels, e(2τ) = 2^p e(τ), makes it
    ///   e(τ) = d₂/(2^p − 1), which is what `between` reads. Of d₂ the part 2^p/(2^p − 1) is the companion's own error:
    ///   two levels that are both too coarse agree (d₂ small, e(τ) not), and a companion beyond the steps at which the
    ///   order holds reads e(τ) high by any factor;
    /// - between 2τ and 4τ, d₄ = e(4τ) − e(2τ) = (2^p − 1) e(2τ) under the order, the coarser level's error, whose
    ///   mean square is the rung's 2^{2p} times: the read carried to half its step.
    /// Where the order holds across the three levels the two reads are one number. Neither is taken alone, nor the
    /// lesser of them: they are of one mean square only where the order holds, which they test against each other
    /// (`verdict`), and the lesser of two noisy reads is biased low. Both are booked as one: the larger (`worse`).
    /// The own read stands on its two levels' analysis times among the candidates `times` (`read_step`).
    fn told(&self, times: &[f64], window: f64, tau: f64, scales: &Scales) -> (Between, Option<Between>, Option<f64>, u64) {
        let (fine, coarse) = self.paired();
        let (own, pairs) = (read_step(times, window, &fine, &coarse, scales), fine.len() as u64 * GROUP);
        (own, self.above.clone(), self.above.as_ref().map(|_| 4. * tau), pairs)
    }
}

/// The members `members` of the groups `numbers` at the step `tau`, each trajectory through the plan in the `domain`
/// (a companion's level takes `Domain::companion`): a hand-out of its own (`run_groups`).
#[allow(clippy::too_many_arguments)]
fn level(run: &Run, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Part> {
    let hand = Hand { groups: numbers, members, shape: Shape::of(plan, keep), record: &|i| trajectory(run, ensemble, domain, tau, plan, keep, i) };
    run_groups(&[hand], messages, progress).pop().expect("the hand-out's one level")
}

/// The members `members` of the groups `numbers` at the step `tau` (their states kept where `keep`) and at twice it,
/// the two levels of a check (module documentation, # The step): one hand-out (`run_groups`) -- a check runs both
/// whatever either shows --, the finer level first, whose trajectories take twice the steps; the coarser one is the
/// companion's (`Domain::companion`). `progress` takes the finer level's groups.
#[allow(clippy::too_many_arguments)]
fn pair(run: &Run, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> [Vec<Part>; 2] {
    let companion = domain.companion();
    let fine = Hand { groups: numbers, members: members.clone(), shape: Shape::of(plan, keep), record: &|i| trajectory(run, ensemble, domain, tau, plan, keep, i) };
    let coarse = Hand { groups: numbers, members, shape: Shape::of(plan, false), record: &|i| trajectory(run, ensemble, &companion, 2. * tau, plan, false, i) };
    let mut levels = run_groups(&[fine, coarse], messages, progress).into_iter();
    std::array::from_fn(|_| levels.next().expect("the hand-out's two levels"))
}

/// The pilot's own level (module documentation, # The pipeline): the other members of the pilot's groups `numbers` at
/// the attempt's step, joined to the scout's `scout`, run before their companion, the reach's falsifier reading them
/// as they come in (`grown`, `outgrown`): it reads the run's own level alone -- the bound atoms' T and their count
/// outside the reach, on whole groups --, never a companion nor the step's check, so its read here is the one the whole
/// pilot makes. Where the scout's atoms already refuted the reach (`doubling`) the members double a hand-out (`batch`)
/// and the reach is read after each -- the scout's refutation is the read that selected the attempt for ending, and
/// its eleven atoms, one a group, are confirmed by the next eleven read with them --; else they are one hand-out, read
/// once. (Read on the groups as they finish: S3c's reads at 30 µK, 62–68 µK at 3.7–4.2 µK on eleven groups, would
/// refute it at the ninth at the earliest, when the threads have begun every trajectory of the level -- nothing saved,
/// and a look more at each prefix.) The level; or, where the atoms in hand refute the reach, the temperature it grows
/// to and those atoms' groups, their states the next attempt's to decide on (`Attempt::Grow`).
#[allow(clippy::too_many_arguments)]
fn own_level(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &Context, scout: &[Part], numbers: &[u64], doubling: bool, grown: &dyn Fn(&[&Part]) -> Option<(f64, Vec<usize>)>, messages: &mut Messages) -> Result<Vec<Part>, (f64, Vec<Part>)> {
    let (tau, plan) = (context.tau, &context.plan);
    let mut joined = scout.to_vec();
    let mut members = 1;
    loop {
        let more = if doubling { (2 * members).min(GROUP) } else { GROUP };
        let hand = Hand { groups: numbers, members: members..more, shape: Shape::of(plan, true), record: &|i| trajectory(run, ensemble, domain, tau, plan, true, i) };
        let mut rest = run_groups(&[hand], messages, &mut |prefix, messages| so_far(context, &joined, prefix, messages));
        absorb(&mut joined, rest.pop().expect("the hand-out's one level"));
        members = more;
        let level: Vec<&Part> = joined.iter().collect();
        if let Some((t, at)) = grown(&level) {
            return Err((t, parts_at(&level, &at)));
        }
        if members == GROUP {
            return Ok(joined);
        }
    }
}

/// The states kept in some groups' parts freed, on the pool, a group's by itself: a state is a block of memory of its
/// own, and a pilot keeps one for every trajectory at every output, and one more for every atom it lost.
fn forget(parts: &mut [Part]) {
    parts.par_iter_mut().for_each(|part| (part.samples, part.lost) = (vec![], vec![]));
}

/// Some groups' further members joined to those a level holds of them (all of them where it holds none).
fn absorb(level: &mut Vec<Part>, more: Vec<Part>) {
    if level.is_empty() {
        *level = more;
    } else {
        level.iter_mut().zip(more).for_each(|(p, m)| p.absorb(m));
    }
}

/// How the pilot's check leaves the step (`settle`): standing -- within on its reads, or on its own two levels where no
/// cheap third level is to be had; or ended where the pilot's atoms refute the reach (`check`) --; clearly beyond its share at the steps' cap; unable to see with no remedy left --
/// each of these two standing, what its reads do not exclude booked (`Levels::told`, `checked`), and flagged.
#[derive(Clone, Debug)]
enum Settled {
    Standing,
    Capped,
    Unseen,
    /// On the pilot, asked to stop there (`until`): the rung's own read is in and not clearly beyond, and the check
    /// would go on past its own two levels -- the decisions the pilot's reads hold come first (`attempt`).
    Pending,
}

/// The members `members` of the pilot's groups `numbers` at both levels (`pair`), joined to those the levels hold, and the
/// step's check (module documentation, # The step), read again at every rung and after every batch of pairs:
/// - a rung whose own read lies clearly beyond the square of the step's share among all the looks (`screen`) is left:
///   the step is halved -- the level at τ becomes the companion, and the members so far run anew at τ/2 alone, its
///   third level above the rung left's own read (`carried`). The finer level keeps its states (the model is read at
///   them);
/// - else, on the scout (which has the pilot to come), the rung stands. On the pilot (all the members of a group in:
///   the last word on the step), every estimate as it reads (`verdict`):
///   - its own estimate within and its third level's too: it stands; its third level's beyond (the order is not shown
///     above it): it is left;
///   - its own estimate within and no third level, where `above` says the rung above is a rung of the ladder and it was
///     not read unread: the pilot's trajectories at 4τ, a quarter of a trajectory each, are its third level; where they
///     read no number (`readable`), the rung above is unread;
///   - its own estimate within and no third level to be had: it stands on its own two levels;
///   - its own estimate beyond the share and not clearly (`Verdict::Unseeing`): the pairs of `remedy` -- the product's
///     next groups at both levels, read again after every batch --; where they cannot see it, it is left: its estimate
///     as it reads is beyond the share.
/// A rung below at which a trajectory would take more than `MOST_STEPS` steps does not exist: a check that is clearly
/// beyond there leaves the step capped, one whose noise no pairs see through ends unable to see (`Settled::Unseen`).
/// On the pilot a rung below's own level, once in, is read by the reach's falsifier first (`outgrown`, `own_level`):
/// where its atoms refute the reach, the check ends there (`Settled::Standing`) and the attempt with it (`attempt`) --
/// no level and no pair is run for a step of a reach the run does not keep.
///
/// `ran`: the finer level of the members already run, joined to the groups the levels hold (`attempt`: the pilot's own
/// level, run first and read by the reach's falsifier before its companion, `own_level`) -- its companion alone is
/// run then, and the check goes on as from a pair: a trajectory
/// is a function of its number and step (`trajectory`), not of the hand-out it was run in. `until`: on the pilot, the
/// check stops where the rung's own read is in, not clearly beyond, and it would go on past its two levels
/// (`Settled::Pending`; `check` takes it up again).
#[allow(clippy::too_many_arguments)]
fn settle(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: std::ops::Range<u64>, above: &dyn Fn(f64) -> bool, outgrown: &dyn Fn(&[&Part]) -> bool, ran: Option<Vec<Part>>, until: bool, messages: &mut Messages) -> Settled {
    // The finer level joined to the groups the levels hold, and its companion.
    let (joined, coarse) = match ran {
        Some(joined) => {
            let companion = domain.companion();
            let (tau, plan) = (context.tau, &context.plan);
            let hand = Hand { groups: numbers, members: members.clone(), shape: Shape::of(plan, false), record: &|i| trajectory(run, ensemble, &companion, 2. * tau, plan, false, i) };
            let mut coarse = run_groups(&[hand], messages, &mut |_, messages| so_far(context, &joined, &[], messages));
            (joined, coarse.pop().expect("the hand-out's one level"))
        }
        None => {
            let [fine, coarse] = pair(run, ensemble, domain, &context.plan, context.tau, numbers, members.clone(), true, messages, &mut |prefix, messages| so_far(context, &levels.fine, prefix, messages));
            let mut joined = std::mem::take(&mut levels.fine);
            absorb(&mut joined, fine);
            (joined, coarse)
        }
    };
    levels.fine = joined;
    absorb(&mut levels.coarse, coarse);
    check(run, ensemble, domain, context, levels, numbers, members.end, above, outgrown, until, messages)
}

/// The step's check of the levels in hand (`settle`) on the members `..members` of the pilot's groups `numbers` (the
/// pilot's where they are all its members); `until` as `settle` takes it.
#[allow(clippy::too_many_arguments)]
fn check(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: u64, above: &dyn Fn(f64) -> bool, outgrown: &dyn Fn(&[&Part]) -> bool, until: bool, messages: &mut Messages) -> Settled {
    // Whether the check is the pilot's: all the members of a group in.
    let pilot = members == GROUP;
    loop {
        // The rung's own read, at its two levels' analysis times.
        let (times, window) = (&context.plan.reads, context.plan.window);
        let read = {
            let (fine, coarse) = levels.paired();
            read_step(times, window, &fine, &coarse, &context.scales)
        };
        // The rung below, and whether the ladder has it.
        let below = 0.5 * context.tau;
        let lowest = context.plan.end / below > MOST_STEPS as f64;
        let upward = !levels.above_pilot && above(context.tau) && !levels.above_unread;
        let verdict = verdict(&read, levels.above.as_ref(), pilot, upward);
        if verdict == Verdict::Stands {
            return Settled::Standing;
        }
        if until && pilot && !screen(&read) {
            return Settled::Pending;
        }
        if verdict == Verdict::Above {
            // The third level above on the pilot's trajectories: the rung above's coarser level, at 4τ, against its
            // finer one, this rung's companion. Where its read excludes nothing on some look (`readable`: no output
            // counts a trajectory at both levels), the rung above is unread.
            let coarser = level(run, ensemble, &domain.companion(), &context.plan, 4. * context.tau, numbers, 0..GROUP, false, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages));
            let read = read_step(&context.plan.reads, context.plan.window, &levels.coarse.iter().collect::<Vec<_>>(), &coarser.iter().collect::<Vec<_>>(), &context.scales);
            if readable(&read) {
                (levels.above, levels.above_pilot) = (Some(carried(&read, 0.5)), true);
            } else {
                levels.above_unread = true;
            }
            continue;
        }
        let have = levels.paired().0.len() as u64 * GROUP;
        let pairs = if verdict == Verdict::Unseeing { blind(&read).and_then(|factor| remedy(factor, have)) } else { None };
        if let Some(pairs) = pairs {
            // The own read's estimate beyond the share, and not clearly: the noise hides the difference, and the pairs
            // could bring it within at the best (`blind`, `remedy`) -- a rung below reads its own pair's noise, and where
            // the pairs part early that is the two ensembles' own. Where pairs cannot, the estimate as it reads is beyond
            // the share: the rung is left, as an adaptive integrator leaves a step whose error estimate is beyond its
            // tolerance.
            let next: Vec<u64> = (have / GROUP..pairs / GROUP).collect();
            let more = pair(run, ensemble, domain, &context.plan, context.tau, &next, 0..GROUP, false, messages, &mut |prefix, messages| {
                so_far(context, &levels.fine, &levels.further[0].iter().chain(prefix.iter().copied()).collect::<Vec<_>>(), messages)
            });
            levels.further.iter_mut().zip(more).for_each(|(further, more)| further.extend(more));
            continue;
        }
        // The rung in hand is left. At the ladder's bottom a read clearly beyond leaves the step capped, one that only
        // could not be seen through, unable to see.
        if lowest {
            return if verdict == Verdict::Unseeing { Settled::Unseen } else { Settled::Capped };
        }
        // The rung below's third level above: this rung's own read, lowered.
        let inherits = Some(carried(&read, 0.5));
        context.tau = below;
        let finer = level(run, ensemble, domain, &context.plan, below, numbers, 0..members, true, messages, &mut |prefix, messages| so_far(context, &[], prefix, messages));
        // The level at the step is the companion from here on: its states are read no more.
        forget(&mut levels.fine);
        levels.coarse = std::mem::replace(&mut levels.fine, finer);
        levels.further = Default::default();
        (levels.above, levels.above_pilot, levels.above_unread) = (inherits, pilot, false);
        // The pilot's own level at the rung taken is in: the reach's falsifier reads it before anything more is spent
        // on the step (`own_level`). Where its atoms refute the reach the check ends there, and the attempt with it.
        if pilot && outgrown(&levels.fine.iter().collect::<Vec<_>>()) {
            return Settled::Standing;
        }
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
    // A need that is no number asks as much as a need without bound (`Look::need` makes none).
    let need = if need.is_nan() { f64::INFINITY } else { need };
    let whole = ((need / GROUP as f64).ceil() as u64).saturating_mul(GROUP);
    whole.clamp(have + GROUP, 2 * have).min(MOST_TRAJECTORIES)
}

/// The product (module documentation, # Stopping): batches of whole groups at the run's step, on from the pilot's
/// groups (their numbers `pilot`; the product's own follow in the list's order), to the number the looks predict
/// (`batch`), until every look a run's trajectories can meet (`Look::fits`) is met, or the trajectories reach
/// `MOST_TRAJECTORIES`. Before every batch's looks the unresolved atoms and those past the reach are booked (`apart`,
/// given the estimates): what they add joins the model's error the pilot fixed, in the looks and in the results. Where the units'
/// error the pilot booked leaves a look no room (`crowded`) at the product's statistics, at an output its first batch's
/// did not, the units the room needs there are asked for (`keeping`: the pilot's reads of the units, the product's
/// statistics), and where there are some the product ends with them -- what the atoms past the reach add is the reach's,
/// read at the product's end (`attempt`; module documentation, # The reach's atoms); the trajectories are not doubled
/// after a look the run's `MOST_TRAJECTORIES` cannot meet. Each batch's
/// result is a checkpoint; returns the estimates, the listed atoms and the counts at its end -- or the units to keep.
#[allow(clippy::too_many_arguments)]
fn product(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &mut Context, parts: &mut Vec<Part>, pilot: &[u64], apart: &dyn Fn(&[Estimate]) -> Vec<[f64; 2]>, keeping: &dyn Fn(&[Estimate], &Fixed, Bound) -> Option<Vec<usize>>, messages: &mut Messages) -> Result<(Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts, Vec<Estimate>), Vec<usize>> {
    let (started, from) = (Instant::now(), parts.len() as u64 * GROUP);
    // The model's error as the pilot fixed it.
    let booked = context.fixed.model.clone();
    // The group at the list's place g.
    let number = |g: u64| pilot.get(g as usize).copied().unwrap_or(g);
    // The wall time (ms) the trajectories from `done` to `want` take at the product's pace so far; none before its own
    // first trajectories are in.
    let left = |done: u64, want: u64| (done > from).then(|| started.elapsed().as_secs_f64() * 1e3 / (done - from) as f64 * (want - done) as f64);
    // The analysis times whose looks no run's trajectories fit and that a keep was asked for: from the product's first
    // batch on, which the pilot's keep has read.
    let mut asked: Option<Vec<usize>> = None;
    loop {
        let all: Vec<&Part> = parts.iter().collect();
        let n = all.len() as u64 * GROUP;
        // The estimates at the analysis times: what the looks and the decisions read (# The analysis times).
        let e = summarize(&reads_of(&all), &context.scales);
        context.outside = apart(&e);
        context.fixed.model = booked.iter().zip(&context.outside).map(|(model, more)| [probability_error(model[0] + more[0]), model[1] + more[1]]).collect();
        let looks = looks(&e, &context.fixed, context.step);
        // The units' error, as the pilot booked it, leaving a look no room at the product's statistics: the units' to
        // answer. What the atoms past the reach add is the reach's (# The reach's atoms: read at the product's end).
        let units = Fixed { model: booked.clone(), ..context.fixed.clone() };
        let lost = crowded(&self::looks(&e, &units, context.step));
        match &mut asked {
            None => asked = Some(lost),
            Some(asked) => {
                let new: Vec<usize> = lost.into_iter().filter(|k| !asked.contains(k)).collect();
                if !new.is_empty() {
                    asked.extend(new);
                    if let Some(units) = keeping(&e, &units, context.step) {
                        return Err(units);
                    }
                }
            }
        }
        let chased: Vec<Look> = looks.into_iter().filter(Look::fits).collect();
        if chased.iter().all(Look::met) || n >= MOST_TRAJECTORIES {
            let (shown, l, c) = pooled(&all, &context.scales);
            return Ok((shown, l, c, e));
        }
        let want = batch(chased.iter().map(Look::need).fold(0., f64::max), n);
        messages.result(all.len(), n, left(n, want), || {
            let (shown, l, c) = pooled(&all, &context.scales);
            assemble(context, &shown, &e, &l, &c, n)
        });
        let numbers: Vec<u64> = (n / GROUP..want / GROUP).map(number).collect();
        let new = level(run, ensemble, domain, &context.plan, context.tau, &numbers, 0..GROUP, false, messages, &mut |prefix, messages| {
            let all: Vec<&Part> = parts.iter().chain(prefix.iter().copied()).collect();
            let done = all.len() as u64 * GROUP;
            messages.result(all.len(), done, left(done, want), || {
                let (shown, l, c) = pooled(&all, &context.scales);
                assemble(context, &shown, &summarize(&reads_of(&all), &context.scales), &l, &c, done)
            });
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
/// width, the outputs the distributions are taken at, and the run's end, the last window's; and the times the
/// decisions read the trajectories at (# The analysis times) with T's windows about them.
struct Plan {
    times: Vec<f64>,
    halves: Vec<V3>,
    listed: Vec<usize>,
    end: f64,
    /// The cells of a trajectory's Brownian path (adapt.rs `Path`), µs: the companion's step at the attempt's first
    /// step (its first rung), a whole part of the run. Every step of
    /// the attempt -- a rung's, its companion's -- is a cell over a power of two, so every level reads the same path.
    /// The run until the first rung is known.
    cell: f64,
    /// The times the decisions read (µs, from t = 0 to the run's end; module documentation, # The analysis times): the
    /// analysis times' candidates while the pilot runs (`grid`), the analysis times in the product; and per time and lab
    /// axis T's window's half width there, cut at both ends of the run (`Plan::read_at`).
    reads: Vec<f64>,
    read_halves: Vec<V3>,
    /// T's shortest window (µs): one period of the fastest lab axis, four times the farthest the analysis times' reads
    /// lie apart (`analysed`).
    window: f64,
    frequencies: V3,
}
impl Plan {
    /// The plan of the request's times with the trap's frequency per lab axis (rad/µs), read at the run's two ends until
    /// the attempt's grid is known (`read_at`).
    fn new(times: &[f64], frequencies: V3) -> Plan {
        let last = times.len() - 1;
        let halves: Vec<V3> = times.iter().map(|&t| frequencies.map(|w| half_window(w, t))).collect();
        let count = times.len().min(DISTRIBUTION_TIMES);
        let mut listed: Vec<usize> = (0..count).map(|i| ((i * last) as f64 / (count - 1) as f64).round() as usize).collect();
        listed.dedup();
        let end = times[last] + halves[last].iter().fold(0., |a: f64, &b| a.max(b));
        let window = TAU / frequencies.iter().copied().fold(0., f64::max);
        let mut plan = Plan { end, times: times.to_vec(), halves, listed, cell: end, reads: vec![], read_halves: vec![], window, frequencies };
        plan.read_at(vec![0., end]);
        plan
    }

    /// The plan read at the times `reads` (increasing, from t = 0 to the run's end): T's window about each, half a
    /// period of the axis on either side and cut at both ends of the run (estimate.rs `half_window`), so that it is
    /// centred on its time -- the trajectories run to the run's end and no further.
    fn read_at(&mut self, reads: Vec<f64>) {
        self.read_halves = reads.iter().map(|&t| self.frequencies.map(|w| half_window(w, t).min(self.end - t))).collect();
        self.reads = reads;
    }
}

/// Counts over trajectories: the steps taken -- a step with jumps is one step, whatever the pieces its path is handed
/// on by (adapt.rs, # The jumps) --, and the trajectories that ended with no outcome (estimate.rs, # What is counted),
/// by why: the scheme could not advance them (adapt.rs `advance`'s refusal) or a step's optical sector made no K (step.rs
/// `Refused::Optical`, by its `Unmade`); and the trajectories stopped where a state of theirs scattered beyond every
/// state of the atom (`Domain::runaway`), unresolved from there on (`run`).
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    steps: u64,
    failed: Stopped,
    unmade: [Stopped; 3],
    runaway: Stopped,
    /// The optical sector's sums over the trajectories (ledger.rs `optical`; step.rs `Trajectory::booked`, `pending`).
    optical: [f64; 4],
}
impl Counts {
    fn add(&mut self, o: &Counts) {
        self.steps += o.steps;
        self.failed.add(&o.failed);
        for (stopped, other) in self.unmade.iter_mut().zip(&o.unmade) {
            stopped.add(other);
        }
        self.runaway.add(&o.runaway);
        for (sum, x) in self.optical.iter_mut().zip(o.optical) {
            *sum += x;
        }
    }
}

/// Trajectories that stopped for one reason (`Counts`): how many, and the earliest time one stopped at.
#[derive(Clone, Copy, Debug, Default)]
struct Stopped {
    count: u64,
    since: Option<f64>,
}
impl Stopped {
    /// One trajectory stopped at the time t.
    fn at(t: f64) -> Stopped {
        Stopped { count: 1, since: Some(t) }
    }
    fn add(&mut self, o: &Stopped) {
        self.count += o.count;
        self.since = match (self.since, o.since) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
    }
}

/// A pilot trajectory's state at a time read (model.rs `Sample`) with the run's step and the trajectory's number: where
/// the units' errors are measured and a model is priced (`Pricing::rates`).
#[derive(Clone)]
struct State {
    sample: Sample,
    tau: f64,
    index: u64,
    /// The trajectory's jumps up to the state.
    jumps: u32,
}

/// What one trajectory leaves: what became of it by every output (estimate.rs `Fate`), its (r, κ) at the listed outputs
/// where held; what became of it by every time the decisions read (`Plan::reads`), its states there where held and,
/// where it was lost, the first of those times it is lost at and its last state, where it was found lost (`keep`: the
/// pilot's, for the model's errors and prices; module documentation, # The model: where the atoms went); its counts.
struct Record {
    outputs: Vec<Fate>,
    listed: Vec<Option<(V3, V3)>>,
    reads: Vec<Fate>,
    samples: Vec<Option<State>>,
    lost: Option<(usize, State)>,
    counts: Counts,
    /// Its photons and their Σ k_φ² (step.rs `Trajectory::scattered`).
    scattered: [f64; 2],
}

/// One trajectory's T windows (module documentation, # Outputs and windows) about some times: per lab axis the running
/// integrals from t = 0 of the time held, κ and κ² (µs, 1/µm·µs, 1/µm²·µs), and per time and axis their values at its
/// window's two ends -- the window's time held, ∫κ dt and ∫κ² dt are the differences. An axis's windows' lower ends come
/// in the times' order, and so do their upper ends (a window is cut at t = 0, and at the run's end alike): per axis and
/// side, the next time whose end is to come. Time 0 is the instant t = 0 and has none.
struct Windows<'a> {
    times: &'a [f64],
    halves: &'a [V3],
    running: [[f64; 3]; 3],
    ends: Vec<[[[f64; 3]; 2]; 3]>,
    next: [[usize; 2]; 3],
}
impl<'a> Windows<'a> {
    fn new(times: &'a [f64], halves: &'a [V3]) -> Windows<'a> {
        Windows { times, halves, running: [[0.; 3]; 3], ends: vec![[[[0.; 3]; 2]; 3]; times.len()], next: [[1; 2]; 3] }
    }
    /// The lower (side 0) or upper (1) end of time k's window on axis a.
    fn end(&self, k: usize, a: usize, side: usize) -> f64 {
        let (t, h) = (self.times[k], self.halves[k][a]);
        if side == 0 { t - h } else { t + h }
    }
    /// The path's piece from `from` to `to` (the next piece starts where this one ends), with its integrals of κ and κ²
    /// per lab axis between two of its times (step.rs `Segment::moments`): each end inside it reads the integrals up to
    /// it -- once for the ends that coincide, the axes of one frequency and the windows cut at t = 0 --, and the running
    /// integrals take the whole piece.
    fn add(&mut self, from: f64, to: f64, moments: &dyn Fn(f64, f64) -> (V3, V3)) {
        let outputs = self.ends.len();
        let mut made: Vec<(f64, (V3, V3))> = vec![];
        for a in 0..3 {
            for side in 0..2 {
                while self.next[a][side] < outputs && self.end(self.next[a][side], a, side) <= to {
                    let k = self.next[a][side];
                    let at = self.end(k, a, side).max(from);
                    let (m1, m2) = match made.iter().find(|(x, _)| *x == at) {
                        Some((_, same)) => *same,
                        None => {
                            let m = moments(from, at);
                            made.push((at, m));
                            m
                        }
                    };
                    let r = self.running[a];
                    self.ends[k][a][side] = [r[0] + (at - from), r[1] + m1[a], r[2] + m2[a]];
                    self.next[a][side] += 1;
                }
            }
        }
        let (m1, m2) = moments(from, to);
        for a in 0..3 {
            let r = self.running[a];
            self.running[a] = [r[0] + (to - from), r[1] + m1[a], r[2] + m2[a]];
        }
    }
    /// The trajectory has ended (run through, lost, or stopped): the ends it did not reach -- within a window, or past
    /// its last piece by the rounding -- take the running integrals where it ended, the window over its held part.
    fn close(&mut self) {
        for a in 0..3 {
            for side in 0..2 {
                for e in &mut self.ends[self.next[a][side]..] {
                    e[a][side] = self.running[a];
                }
                self.next[a][side] = self.ends.len();
            }
        }
    }
    /// Time k's window (k ≥ 1) per lab axis: its time held, ∫κ dt and ∫κ² dt.
    fn window(&self, k: usize) -> [[f64; 3]; 3] {
        let e = &self.ends[k];
        std::array::from_fn(|a| std::array::from_fn(|i| e[a][1][i] - e[a][0][i]))
    }
    /// The atoms held at the times `fates` (time 0 the instant t = 0) as the windows read them: from the first time on
    /// an atom's κ and κ² are its window's means -- over its held part, which is no empty stretch after t = 0
    /// (estimate.rs `half_window`); a window of no length is the instant's own κ and κ².
    fn mean(&self, fates: &mut [Fate]) {
        for (k, o) in fates.iter_mut().enumerate().skip(1) {
            if let Fate::Held(h) = o {
                let w = self.window(k);
                for a in (0..3).filter(|&a| w[a][0] > 0.) {
                    (h.kappa[a], h.kappa2[a]) = (w[a][1] / w[a][0], w[a][2] / w[a][0]);
                }
            }
        }
    }
}

/// The times after t = 0 of two increasing lists from t = 0, merged: each with its place in either list, a time that is
/// in both once.
fn merged(first: &[f64], second: &[f64]) -> Vec<(f64, [Option<usize>; 2])> {
    let (mut i, mut j, mut out) = (1, 1, vec![]);
    while i < first.len() || j < second.len() {
        let (a, b) = (first.get(i).copied().unwrap_or(f64::INFINITY), second.get(j).copied().unwrap_or(f64::INFINITY));
        if a == b {
            out.push((a, [Some(i), Some(j)]));
            (i, j) = (i + 1, j + 1);
        } else if a < b {
            out.push((a, [Some(i), None]));
            i += 1;
        } else {
            out.push((b, [None, Some(j)]));
            j += 1;
        }
    }
    out
}

/// One trajectory from its initial state through the plan at the step τ (module documentation, # Outputs and windows):
/// advanced to the run's end by steps that know no output, and read at every output and at every time the decisions
/// read (`Plan::reads`) where it falls (adapt.rs, # The reads) -- a time of both read once. A trajectory the scheme
/// cannot advance (adapt.rs) fails from the first time it does not reach. One whose state scatters beyond every state
/// of the atom (`Domain::runaway`: a fold far outside its validity) stops there: it is unresolved from then on, held at
/// every later time as it stood (module documentation, # The reach's atoms).
fn run<D: Dynamics>(model: &D, domain: &Domain, start: Start, index: u64, tau: f64, plan: &Plan, keep: bool) -> Record {
    let sizes = [plan.times.len(), plan.reads.len()];
    let mut t = Trajectory::new(model, start.state, start.r, start.kappa, SEED, index);
    let mut scratch = Scratch::default();
    // The trajectory's Brownian path in the plan's cells, its grid the multiples of τ (a cell over a power of two):
    // the same path at every rung and at its companion.
    let mut path = adapt::Path::new(SEED, index, plan.cell, (plan.cell / tau).log2().round() as u32);
    let mut windows = [Windows::new(&plan.times, &plan.halves), Windows::new(&plan.reads, &plan.read_halves)];
    let mut fates = sizes.map(|n| vec![Fate::Lost; n]);
    let mut listed = vec![None; plan.listed.len()];
    let mut samples: Vec<Option<State>> = vec![None; sizes[1]];
    // The times read after t = 0, the outputs' and the decisions', with their places among each.
    let at = merged(&plan.times, &plan.reads);
    let times: Vec<f64> = at.iter().map(|a| a.0).collect();
    // The steps taken, and the start of the last one counted.
    let (mut steps, mut counted) = (0, None);
    let mut record = |places: [Option<usize>; 2], t: &Trajectory, fates: &mut [Vec<Fate>; 2]| {
        // The atom at the time's instant; from the first time on its κ and κ² are its window's means (`Windows::mean`).
        // An unresolved one is booked whole (`apart`). Leaving or bound, at the instant.
        let [beyond, leaving] = domain.placed(t.r, t.kappa);
        let outside = !t.unresolved && beyond;
        let held = Fate::Held(Held { rows: model.readout(t.cluster, &t.point(), &t.psi), kappa: t.kappa, kappa2: t.kappa.map(|x| x * x), outside, unresolved: t.unresolved, leaving });
        if let Some(k) = places[0] {
            fates[0][k] = held.clone();
            if let Ok(j) = plan.listed.binary_search(&k) {
                listed[j] = Some((t.r, t.kappa));
            }
        }
        if let Some(k) = places[1] {
            fates[1][k] = held;
            if keep {
                samples[k] = Some(State { sample: Sample { cluster: t.cluster, at: t.point(), psi: t.psi.to_vec() }, tau, index, jumps: t.jumps });
            }
        }
    };
    record([Some(0), Some(0)], &t, &mut fates);
    // The first output and the first time read the trajectory does not reach held; whether the scheme could not
    // advance it, and whether it stopped where a state scattered beyond every state of the atom.
    let mut reached = [1, 1];
    let mut unmade = None;
    let (failed, runaway) = {
        let mut read = |i: usize, t: &Trajectory| {
            record(at[i].1, t, &mut fates);
            for (reached, place) in reached.iter_mut().zip(at[i].1) {
                if let Some(k) = place {
                    *reached = k + 1;
                }
            }
        };
        // A step's path as far as the trajectory is on it, into T's windows: piece by piece, from the step's start and
        // from each of its jumps -- the step is counted at the first of them.
        let mut each = |s: &Segment, until: f64| {
            if counted != Some(s.step.0) {
                (steps, counted) = (steps + 1, Some(s.step.0));
            }
            for w in &mut windows {
                w.add(s.start, until, &|from, to| s.moments(from, to));
            }
        };
        // Through the last output's window; a failure there leaves every output read.
        match adapt::advance(model, domain, tau, &mut t, &mut scratch, &mut path, plan.end, &times, &mut read, &mut each) {
            Ok(()) => (false, false),
            Err(Refused::Runaway) => (false, true),
            Err(Refused::Optical(why)) => {
                unmade = Some(why);
                (true, false)
            }
            Err(_) => (true, false),
        }
    };
    // A runaway stops where it stood: unresolved, held so at every time it did not reach.
    if runaway {
        t.unresolved = true;
        for k in reached[0]..sizes[0] {
            record([Some(k), None], &t, &mut fates);
        }
        for k in reached[1]..sizes[1] {
            record([None, Some(k)], &t, &mut fates);
        }
        reached = sizes;
    }
    // From the time it does not reach: failed, or lost -- a lost atom's last state kept where states are, its fate's
    // to read (`sampled`).
    let mut lost = None;
    for ((fates, &reached), &size) in fates.iter_mut().zip(&reached).zip(&sizes) {
        fates[reached.min(size)..].fill(if failed { Fate::Failed } else { Fate::Lost });
    }
    if reached[1] < sizes[1] && keep && !failed {
        lost = Some((reached[1], State { sample: Sample { cluster: t.cluster, at: t.point(), psi: t.psi.to_vec() }, tau, index, jumps: t.jumps }));
    }
    for (w, fates) in windows.iter_mut().zip(&mut fates) {
        w.close();
        w.mean(fates);
    }
    let stopped = |why: bool| if why { Stopped::at(t.time) } else { Stopped::default() };
    let optical = [t.booked[0], t.booked[1], t.booked[2], -(t.pending[0][0] + t.pending[1][1] + t.pending[2][2])];
    let counts = Counts {
        steps,
        failed: stopped(failed && unmade.is_none()),
        unmade: [Unmade::Schur, Unmade::Singular, Unmade::Unfinite].map(|why| stopped(unmade == Some(why))),
        runaway: stopped(runaway),
        optical,
    };
    let [outputs, reads] = fates;
    Record { outputs, listed, reads, samples, lost, counts, scattered: t.scattered }
}

/// What a group of trajectories leaves: its sums at the outputs and at the times the decisions read (estimate.rs
/// `Group`), its held atoms' (r, κ) at the listed outputs, its states at the times read and its lost atoms' last states
/// with the first of those times each is lost at (the pilot's: every member's), its counts.
#[derive(Clone)]
struct Part {
    sums: Group,
    reads: Group,
    listed: Vec<Vec<(V3, V3)>>,
    samples: Vec<Vec<State>>,
    lost: Vec<(usize, State)>,
    counts: Counts,
}
impl Part {
    /// The same groups' further members, after this part's.
    fn absorb(&mut self, other: Part) {
        self.sums.absorb(other.sums);
        self.reads.absorb(other.reads);
        for (l, x) in self.listed.iter_mut().zip(other.listed) {
            l.extend(x);
        }
        for (s, x) in self.samples.iter_mut().zip(other.samples) {
            s.extend(x);
        }
        self.lost.extend(other.lost);
        self.counts.add(&other.counts);
    }

    /// The part at the places `at` of its times read alone (increasing, the first and the last among them: analysis
    /// times, # The analysis times): its sums and states there, a lost atom's first place the first of them at or
    /// after its own; its outputs as they are.
    fn at(&self, at: &[usize]) -> Part {
        Part {
            sums: self.sums.clone(),
            reads: self.reads.at(at),
            listed: self.listed.clone(),
            samples: at.iter().map(|&k| self.samples[k].clone()).collect(),
            lost: self.lost.iter().map(|(first, state)| (at.partition_point(|&k| k < *first), state.clone())).collect(),
            counts: self.counts,
        }
    }
}

/// Some parts at the places `at` of their times read (`Part::at`).
fn parts_at(parts: &[&Part], at: &[usize]) -> Vec<Part> {
    parts.par_iter().map(|p| p.at(at)).collect()
}

/// The groups' sums at the outputs: what is shown.
fn outputs_of<'a>(parts: &[&'a Part]) -> Vec<&'a Group> {
    parts.iter().map(|p| &p.sums).collect()
}

/// The groups' sums at the times the decisions read.
fn reads_of<'a>(parts: &[&'a Part]) -> Vec<&'a Group> {
    parts.iter().map(|p| &p.reads).collect()
}

/// What a group's records fold into (`Part`): the outputs, the times read, the listed outputs, and whether the states
/// are kept.
#[derive(Clone, Copy)]
struct Shape {
    outputs: usize,
    reads: usize,
    listed: usize,
    keep: bool,
}
impl Shape {
    fn of(plan: &Plan, keep: bool) -> Shape {
        Shape { outputs: plan.times.len(), reads: plan.reads.len(), listed: plan.listed.len(), keep }
    }
    /// A group's records added in their order.
    fn fold(&self, records: impl Iterator<Item = Record>) -> Part {
        let mut part = Part {
            sums: Group::new(self.outputs),
            reads: Group::new(self.reads),
            listed: vec![vec![]; self.listed],
            samples: vec![vec![]; self.reads],
            lost: vec![],
            counts: Counts::default(),
        };
        for r in records {
            part.sums.add(&r.outputs);
            part.sums.scatter(r.scattered);
            part.reads.add(&r.reads);
            part.reads.scatter(r.scattered);
            for (l, x) in part.listed.iter_mut().zip(r.listed) {
                l.extend(x);
            }
            if self.keep {
                for (s, x) in part.samples.iter_mut().zip(r.samples) {
                    s.extend(x);
                }
                part.lost.extend(r.lost);
            }
            part.counts.add(&r.counts);
        }
        part
    }
}

/// Trajectory `index` of a run at the step τ: its initial state from the ensemble (ensemble.rs `Ensemble::draw`), run
/// through the plan (`run`). A function of the index and the step: the same index at another step is the same atom on
/// the same streams (module documentation, # The step).
fn trajectory(model: &Run, ensemble: &Ensemble, domain: &Domain, tau: f64, plan: &Plan, keep: bool, index: u64) -> Record {
    run(model, domain, ensemble.draw(SEED, index), index, tau, plan, keep)
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
    // Per level and group the members' records as they come in, and how many are in (a member not run counts in).
    let mut waiting: Vec<Vec<(Vec<Option<Record>>, u64)>> = levels.iter().map(|level| (0..level.groups.len()).map(|_| ((0..level.size()).map(|_| None).collect(), 0)).collect()).collect();
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
                let record = (levels[level].record)(levels[level].number(k));
                if sender.send((level, k, record)).is_err() {
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
                    let (members, count) = &mut waiting[level][g];
                    members[m] = Some(r);
                    *count += 1;
                    if *count == size {
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

/// What the pilot fixes for the run (module documentation, # Stopping), per time: the model's error on the held atoms
/// (populations absolute, T relative; T's unbounded where no pilot atom reads it) -- to which the product adds what
/// its atoms outside the reach carry (`Context::outside`) --; none at t = 0, where no model has acted (`model_error`).
#[derive(Clone)]
struct Fixed {
    model: Vec<[f64; 2]>,
}

/// The step's share of the targets (SPEC §2.2, §4): on what is shown s·ε, and on T s itself -- of T's allowance at each
/// time, which T's reads and bounds are taken over (# T's allowance; estimate.rs `between`): what its error may take
/// -- the pilot is screened against it, the model's decisions take it for the step's error, and a result is booked at
/// it until its two levels are read.
fn step_bound() -> [f64; 2] {
    [STEP_SHARE * POPULATION_TARGET, STEP_SHARE]
}

/// The room the statistics may take of a target ε beside a model error A and a step error b, √(ε² − A² − b²) (module
/// documentation, # Stopping); none where the two fill the target, or where either is no number -- it bounds nothing.
fn room(target: f64, model: f64, step: f64) -> f64 {
    let rest = target * target - model * model - step * step;
    if rest > 0. { rest.sqrt() } else { 0. }
}

/// An error on a probability as it is booked (SPEC §4): at most one, the probability's whole range; one that is no
/// number is that whole -- nothing bounds it less (estimate.rs, # What is counted). Every cap of an error on what is
/// shown is this one, so that no cap turns a number that is none into a bound.
fn probability_error(error: f64) -> f64 {
    if error.is_nan() { 1. } else { error.min(1.) }
}

/// One quantity held to its target at one time (module documentation, # Stopping): what is shown, or the total T.
/// Its account (ledger.rs: the model's error A, what the step's check leaves open c, the statistics S, as the ledger
/// books them) against its target ε; and what its need takes beside them: the statistics' jackknife interval and the
/// trajectories it was read on, and on what is shown the unseen share, which the statistics are no less than.
#[derive(Clone, Copy, Debug)]
struct Look {
    output: usize,
    target: f64,
    account: Account,
    interval: f64,
    trajectories: f64,
    shown: bool,
}
impl Look {
    /// Whether its total lies within its target: the ledger's own test of a time (ledger.rs `Ledger::close`).
    fn met(&self) -> bool {
        self.account.total() <= self.target
    }
    /// The room its statistics may take (`room`).
    fn room(&self) -> f64 {
        room(self.target, self.account.model, self.account.step)
    }
    /// Whether the trajectories a run may take bring it within its target (module documentation, # Stopping): its need
    /// within `MOST_TRAJECTORIES` -- a look whose model's error and step's leave its statistics a room no run fills is
    /// not met by running, whatever room there is, and is not chased (`product`).
    fn fits(&self) -> bool {
        self.need() <= MOST_TRAJECTORIES as f64
    }
    /// The trajectories it needs: its interval, falling as N^{−1/2}, within its room; on what is shown no fewer than the
    /// draws that leave the room unseen (estimate.rs `draws_for`). Unbounded where the model's error and the step's leave
    /// no room, or where its interval is unread -- a number, always.
    fn need(&self) -> f64 {
        let room = self.room();
        if !(room > 0.) || !self.interval.is_finite() {
            return f64::INFINITY;
        }
        let draws = self.trajectories * (self.interval / room).powi(2);
        if self.shown { draws.max(draws_for(room)) } else { draws }
    }
}

/// The looks of a run (module documentation, # Stopping) by its estimates, the pilot's fixed errors and the step's
/// part: at every time after t = 0 what is shown, and the total T (the mean over the three lab axes: what T's target
/// is on). A T whose model's or step's part is unread is not met, and no number of trajectories meets it. Where it is
/// its statistics that are unread (too few groups hold an atom), they are predicted from the normal theory of one
/// axis's variance over the N h atoms held, the relative interval z √(2/(N h)) -- the total's is at most that (three
/// axes' mean; equal where one axis holds all the energy) --, weighed by the held share as a read interval is: one
/// number for whether the look is met and for the trajectories it needs (`Look::met`, `Look::need`). (Unread, it was
/// booked unbounded for the one and predicted for the other: S2Q's times at 2.4--2.6 ms, about one atom held of
/// three thousand, were never met and asked for one to three trajectories, and its product grew by a group a batch,
/// 293 batches, from 568 trajectories to 2912.) Where no atom is held there is no T, and no look of it: N h = 0 would
/// ask that interval of nothing. The times' looks are made on the pool, each time's by itself, and come in the
/// times' order.
fn looks(estimates: &[Estimate], fixed: &Fixed, step: Bound) -> Vec<Look> {
    let (population, temperature) = accounts(estimates, fixed, step);
    estimates
        .par_iter()
        .enumerate()
        .skip(1)
        .flat_map_iter(|(k, e)| {
            let (h, n) = (1. - e.loss, e.trajectories);
            let shown = Look { output: k, target: POPULATION_TARGET, account: population[k], interval: e.shown_interval(), trajectories: n, shown: true };
            let held = (e.held > 0.).then(|| {
                let read = Some(e.temperature_interval()).filter(|i| i.is_finite());
                let account = temperature[k].unwrap_or_else(|| temperature_account(e, fixed.model[k][1], &step, read.unwrap_or(CONFIDENCE * (2. / (n * h)).sqrt())));
                Look { output: k, target: e.temperature_target, account, interval: account.statistics, trajectories: n, shown: false }
            });
            std::iter::once(shown).chain(held)
        })
        .collect()
}

/// The times where some look's model error and step's share leave the statistics no room (SPEC §5.2): no number
/// of trajectories fits beside them -- what the model's decisions answer (`keep`), on the errors booked, not on a
/// statistics' read: a keep is never undone, and a need read off a few groups' interval is no certain thing (a look
/// that a run cannot meet within `MOST_TRAJECTORIES` is not chased, `Look::fits`, and is read again at every batch).
fn crowded(looks: &[Look]) -> Vec<usize> {
    let mut outputs: Vec<usize> = looks.iter().filter(|l| !(l.room() > 0.)).map(|l| l.output).collect();
    outputs.dedup();
    outputs
}

/// The times whose looks leave the statistics no room (`crowded`) at the model's errors `fixed` beside the step's
/// part `step`, at the `estimates` (every trajectory counted), among those `read`: where a unit can be told to weigh
/// (`weights`) -- a time with no state read carries every unit's bound alike, and tells none of them apart.
fn tight(estimates: &[Estimate], fixed: &Fixed, step: Bound, read: &[bool]) -> Vec<usize> {
    crowded(&looks(estimates, fixed, step)).into_iter().filter(|&k| read[k]).collect()
}

/// Of the units' errors `means` (per time and unit), per unit its weight at the `times` (`tight`): its errors on
/// what is shown (the held and the lost atoms', `Measured::means`) and on T, weighed by the held share `held` as T's
/// total is, each over its target there -- T's its allowance at the estimates' T (# T's allowance) --, added up. No T
/// weighs where no atom is held.
fn weights(means: &[Vec<[f64; 2]>], estimates: &[Estimate], held: &[f64], outputs: &[usize]) -> Vec<f64> {
    let temperature = |k: usize, u: usize| if held[k] > 0. { held[k] * means[k][u][1] / estimates[k].temperature_target } else { 0. };
    (0..means.first().map_or(0, Vec::len)).map(|u| outputs.iter().map(|&k| means[k][u][0] / POPULATION_TARGET + temperature(k, u)).sum()).collect()
}

/// The units that weigh where the looks have no room (`weights`), the candidates for the set they need (`needed`).
fn weighing(estimates: &[Estimate], fixed: &Fixed, step: Bound, means: &[Vec<[f64; 2]>], held: &[f64], read: &[bool]) -> Vec<usize> {
    weights(means, estimates, held, &tight(estimates, fixed, step, read)).iter().enumerate().filter(|(_, w)| **w > 0.).map(|(u, _)| u).collect()
}

/// The doubt the reads of a few groups leave on the model's error per time (`keep`), on what is shown and on T: the
/// model's error as every group but one gives it -- the units' as `measured` at their states and what the reach leaves
/// out, `unsampled` (`model_error`), with the estimates of the others (their populations a state's booking is taken
/// at, `Booking::at`) -- whose jackknife over the groups is the interval the reads leave on the error, at the family of
/// the times' two reads (`interval`). Its near end is the error less it (`clear`): what the reads leave no doubt of,
/// as a rung is left only on that (# The step). Its far end is the error and it (`raised`): what the reads leave no
/// doubt is not exceeded -- of the error a set of units leaves, the set's errors taken out of every group's
/// (`interval` of the set). The scout reads eleven states a time, one a group, and a unit's error is heavy-tailed
/// over the states (it is largest at the atoms that are hottest): S3's scout read its carrier's dressing as filling the
/// room on eleven states, where its pilot's eighty-eight booked it at 2.7 % beside 3.7 % in all.
struct Doubt<'a> {
    measured: &'a Measured<'a>,
    /// Per group left out, the others' estimates and their model's error per time.
    left: Vec<(Vec<Estimate>, Vec<[f64; 2]>)>,
    outputs: usize,
}
impl<'a> Doubt<'a> {
    fn new(measured: &'a Measured<'a>, outputs: usize, parts: &[&Part], scales: &Scales, unsampled: [f64; 2]) -> Doubt<'a> {
        // Group by group: a group's units' means are every time's every unit's, and they are summed at once.
        let left = (0..parts.len())
            .map(|g| {
                let rest: Vec<&Part> = parts.iter().enumerate().filter(|(h, _)| *h != g).map(|(_, p)| *p).collect();
                let estimates = summarize(&reads_of(&rest), scales);
                let error = model_error(&measured.combined(outputs, Some(g), &estimates), unsampled);
                (estimates, error)
            })
            .collect();
        Doubt { measured, left, outputs }
    }

    /// Per time the jackknife interval over the groups, at the family of the times' two reads, of the model's error
    /// less the units `set`'s (none: of the whole error); none where no group reads it (a spread that is no number).
    fn interval(&self, set: &[usize]) -> Vec<[f64; 2]> {
        let rest = self.replicates(set);
        let looks = 2. * self.outputs.saturating_sub(1) as f64;
        (0..self.outputs).map(|k| std::array::from_fn(|i| spread(rest.iter().map(|r| r[k][i]), self.left.len() as f64).interval_over(looks))).collect()
    }

    /// Per group left out, per time the model's error less the units `set`'s (`remainder`): the jackknife's
    /// replicates.
    fn replicates(&self, set: &[usize]) -> Vec<Vec<[f64; 2]>> {
        self.left
            .par_iter()
            .enumerate()
            .map(|(g, (estimates, error))| {
                let less = self.measured.means_of(self.outputs, Some(g), estimates, set);
                error.iter().zip(&less).map(|(e, l)| remainder(*e, l.iter())).collect()
            })
            .collect()
    }

    /// The chance that the next pilot reads the model's error the units `set` leave, `point` (`without`), with some
    /// look among the `estimates` (`tight`: where an atom is held and a state read) without room, but at the times
    /// `beyond` (`needed`). The estimate's error and the next pilot's own are independent draws of the reads' spread,
    /// whose jackknife gives its draws over all the times together: the deviations δ_g = √(G − 1)(r_g − r̄) of the
    /// replicates, of mean square the jackknife's variance; so the next read is the point less one draw and plus
    /// another, Â − δ_g + δ_h, over the G² pairs, and the chance is the share of them that leave a look no room. An
    /// time whose replicates are no number (no group's states read it) has no draw.
    fn chance(&self, set: &[usize], point: &Fixed, estimates: &[Estimate], step: Bound, read: &[bool], beyond: &[usize]) -> f64 {
        let rest = self.replicates(set);
        let g = rest.len();
        let mean: Vec<[f64; 2]> = (0..self.outputs).map(|k| std::array::from_fn(|i| rest.iter().map(|r| r[k][i]).sum::<f64>() / g as f64)).collect();
        let deviation = |a: usize, k: usize, i: usize| -> f64 {
            let d = ((g - 1) as f64).sqrt() * (rest[a][k][i] - mean[k][i]);
            if d.is_finite() { d } else { 0. }
        };
        let pairs: Vec<(usize, usize)> = (0..g).flat_map(|a| (0..g).map(move |b| (a, b))).collect();
        let short = pairs
            .par_iter()
            .filter(|&&(a, b)| {
                let model = point.model.iter().enumerate().map(|(k, m)| if k == 0 { *m } else { std::array::from_fn(|i| (m[i] - deviation(a, k, i) + deviation(b, k, i)).max(0.)) }).map(|m: [f64; 2]| [probability_error(m[0]), m[1]]).collect();
                tight(estimates, &Fixed { model, ..point.clone() }, step, read).iter().any(|k| !beyond.contains(k))
            })
            .count();
        short as f64 / pairs.len() as f64
    }
}

/// An error less the `doubt` its reads leave on it, no less than nothing; an error that is no number stays as it is
/// (`Doubt`).
fn lowered(error: f64, doubt: f64) -> f64 {
    let near = error - doubt;
    if near.is_nan() { error } else { near.max(0.) }
}

/// The model's errors `fixed` per time at their near ends (`Doubt`, `lowered`): what the reads leave no doubt of.
fn clear(fixed: &Fixed, doubt: &[[f64; 2]]) -> Fixed {
    Fixed { model: fixed.model.iter().zip(doubt).map(|(m, d)| [lowered(m[0], d[0]), lowered(m[1], d[1])]).collect() }
}

/// The model's errors `fixed` per time at their far ends (`Doubt`): each with the doubt its reads leave on it, a
/// population's at most one; where the doubt is no number (no group reads it), as it is.
fn raised(fixed: &Fixed, doubt: &[[f64; 2]]) -> Fixed {
    let far = |error: f64, doubt: f64| if doubt.is_nan() { error } else { error + doubt };
    Fixed { model: fixed.model.iter().zip(doubt).map(|(m, d)| [probability_error(far(m[0], d[0])), far(m[1], d[1])]).collect() }
}

/// A time's model's error `error` less the errors of some units there, `less`, no less than nothing, a population's
/// at most one (`without`, `Doubt::interval`); no number where a unit's error there is unbounded -- the states do not
/// read the time (`Measured::means`), and its error is its bound, which no unit's removal lowers.
fn remainder<'a>(error: [f64; 2], less: impl Iterator<Item = &'a [f64; 2]>) -> [f64; 2] {
    let less = less.fold([0.; 2], |a, m| [a[0] + m[0], a[1] + m[1]]);
    if less[1].is_infinite() {
        return [f64::NAN; 2];
    }
    [probability_error((error[0] - less[0]).max(0.)), (error[1] - less[1]).max(0.)]
}

/// What a set's expected work is weighed with (`needed`): the least work a µs a kept model can have (`Pricing::least`),
/// the trajectories the looks need with every unit's error removed (`Level::need_alone`), the trajectories a kept
/// model runs again before it stands where this one does, and the expected work of keeping nothing (none where some
/// look has no room for the statistics).
struct Costs {
    least: f64,
    floor: f64,
    again: f64,
    none: Option<f64>,
}

/// The times whose looks set the run's need at the model's errors `fixed` (`needed`): of the looks at times read
/// that the run's trajectories bring within their targets (`Look::fits`), those of the largest need.
fn binding(estimates: &[Estimate], fixed: &Fixed, step: Bound, read: &[bool]) -> Vec<usize> {
    let fitting: Vec<Look> = looks(estimates, fixed, step).into_iter().filter(|l| read[l.output] && l.fits()).collect();
    let most = fitting.iter().map(Look::need).fold(0., f64::max);
    let mut outputs: Vec<usize> = fitting.iter().filter(|l| l.need() >= most).map(|l| l.output).collect();
    outputs.dedup();
    outputs
}

/// The units to keep (module documentation, # The model; `keep`), by the run's expected total work. The order: at the
/// far end of the model's error the units taken so far leave (`far`: the error and the doubt its reads leave on it,
/// `Doubt`), the times where the need binds -- those whose looks leave the statistics no room, and where every look
/// has room those of the largest need (`binding`) --, and there the unit whose weight is the most for the work a µs of
/// its kept model (`work`: its mean rate; none where its keep is refused, never taken) is taken next: the error that
/// sets the run's length is removed where the run it leaves costs the least. The order ends where no unit weighs where
/// the need binds, or where no larger set can be expected to cost less than the least found: a set with room costs
/// at least its own work a µs -- no larger set's is less, its clusters are no smaller (`Price::grown`) -- times its
/// pilot again and the trajectories the looks need with every error removed (`Costs::floor`). (By weight alone, S3's
/// probe took its carrier's dressing with thirteen bare drives on its scout -- the generator's assembly, the jumps and
/// the photons three to four times dearer a step --, where the bare drives alone leave room. By the work a keep adds,
/// the difference of two rates, PGC3D took 129 units whose keeps cost nothing at the step it stood at, their weights a
/// trifle, before any that the room needed, and its model had no step left: a keep's work is the kept model's, at any
/// step. Asked only where some look had no room, GMD1C's pilot read a model error of 0.047 against the room's 0.0498
/// and kept none: its 800 trajectories took 549 CPU-s, where its forty drives kept explicit, their errors removed
/// together -- none of them pays alone --, leave 0.0107 and 176 trajectories, 100 CPU-s.)
///
/// The set: of the order's leading sets, the one of the least expected work (`price`: per set the work a µs of its
/// kept model, the product's trajectories beside the error it leaves, and the chance that the next pilot reads a look
/// without room, `Doubt::chance`), and of it none to spare (`least_expected`); none where keeping nothing is expected
/// to cost no more. Kept, a set's model runs a pilot again (`again`) and its product; where the next pilot reads a
/// look without room it keeps again, of the larger sets the one of the least expected work -- the largest, another
/// pilot and its product at its own work. A cost against a cost: the error read and its far end are the two ends of
/// it. (Taken until the error read had room, e038's pilots kept 6, 2, 1 and 1 units over four attempts, a pilot and its
/// reads each, 440 CPU-s, the next pilot reading the times left with no room as often as not; taken to the far end of
/// the error, R4_LIN_2's pilot kept six units where one did, and the next pilot three more.) Times that no set has
/// room at -- every unit that weighs there kept -- are no set's to cover (the ledger flags them). A set that leaves a
/// look no room at the error as read (`point`) is no candidate where a larger one leaves room there: its model decides
/// first on these same states (module documentation, # An attempt's own), and their reads ask the rest of the room at
/// once -- a round of a model, its tables and its reads for a set the reads already refuse. (Priced as the next
/// pilot's chance alone, a keep on states that run nothing again, `again` = 0, costs a set short of room as much as
/// the larger set it leads to, and S2Q kept its first 37 units in 37 rounds.)
///
/// A kept set's rate (`work`: its work formula, `Pricing::rates`) is a kept model's structure and its plans, made where a
/// set asks for it, and once. `least` bounds every set's rate from below (`Pricing::least`), so a unit's weight over it
/// bounds its value from above: the units are asked in the order of their weights, and no further once the best value
/// made is no less than the next one's bound -- the unit taken is the one of the most value, as were every rate made.
/// A set's expected work is its rate times what it runs, and a set that the bound leaves above the least found -- of
/// the larger sets, or the set kept -- is never the least: its rate is not made.
#[allow(clippy::too_many_arguments)]
fn needed(estimates: &[Estimate], step: Bound, means: &[Vec<[f64; 2]>], held: &[f64], read: &[bool], costs: &Costs, work: &dyn Fn(&[usize]) -> Option<[f64; 2]>, point: &dyn Fn(&[usize]) -> Fixed, far: &dyn Fn(&[usize]) -> Fixed, price: &dyn Fn(&[usize], &[usize]) -> [f64; 2]) -> Vec<usize> {
    let tight_at = |set: &[usize]| tight(estimates, &far(set), step, read);
    // The times no set has room at: those tight with every unit kept that weighs where the far end is tight.
    let every: Vec<usize> = weights(means, estimates, held, &tight_at(&[])).iter().enumerate().filter(|&(_, w)| *w > 0.).map(|(u, _)| u).collect();
    let beyond = tight_at(&every);
    let open = |set: &[usize]| -> Vec<usize> {
        let fixed = far(set);
        let short: Vec<usize> = tight(estimates, &fixed, step, read).into_iter().filter(|k| !beyond.contains(k)).collect();
        if short.is_empty() { binding(estimates, &fixed, step, read) } else { short }
    };
    let short = |set: &[usize]| tight(estimates, &point(set), step, read).iter().any(|k| !beyond.contains(k));
    let (mut taken, mut best): (Vec<usize>, f64) = (vec![], costs.none.unwrap_or(f64::INFINITY));
    let mut made: HashMap<usize, Option<f64>> = HashMap::new();
    loop {
        // A set with room: what it is expected to cost as the last one kept, and whether a larger one could cost less.
        if !taken.is_empty() && !short(&taken) {
            let Some([w, setup]) = work(&taken) else { break };
            let [n, p] = price(&taken, &beyond);
            best = best.min(w * (costs.again * (1. + p) + n) + setup);
            if w * (costs.again + costs.floor) + setup >= best {
                break;
            }
        }
        let outputs = open(&taken);
        if outputs.is_empty() {
            break;
        }
        let weights = weights(means, estimates, held, &outputs);
        let weighing: Vec<usize> = (0..weights.len()).filter(|&u| weights[u] > 0. && !taken.contains(&u)).collect();
        // Error per work a µs (a model's rate is some work: every step costs some), bounded by the bound's.
        let mut asked: Vec<(usize, f64)> = weighing.iter().map(|&u| (u, weights[u] / costs.least)).collect();
        asked.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
        let mut chosen: Option<(usize, f64)> = None;
        for (u, most) in asked {
            if chosen.is_some_and(|(_, value)| value > most) {
                break;
            }
            if let Some(c) = *made.entry(u).or_insert_with(|| work(&[u]).map(|w| w[0])) {
                let value = weights[u] / c;
                // Of equal values the last unit, as a scan of them all in their order takes it.
                if chosen.is_none_or(|(b, v)| value > v || (value == v && u > b)) {
                    chosen = Some((u, value));
                }
            }
        }
        let Some((u, _)) = chosen else { break };
        taken.push(u);
    }
    least_expected(&taken, &|u| made[&u].unwrap_or(0.), costs, work, &|set| price(set, &beyond), &short)
}

/// Of the sets the order `taken` leads through, the one of the least expected work, and of it none to spare (`needed`):
/// a set kept, its model runs a pilot again and then its product, of the work a µs `w` of its kept model (`work`, none
/// where its keep is refused; `Costs::least` below every set's), the trajectories `again` and `n` the looks need beside
/// the error it leaves; and where the next pilot still reads a look without room (its chance `p`; `price` gives [n, p]),
/// the next keep is chosen again among the larger sets, the least of theirs -- for the largest, another pilot and its
/// product at its own work. E_j = w_j (again + (1 − p_j) n_j) + p_j min_{i > j} E_i. None where keeping nothing is
/// expected to cost no more (`Costs::none`). Then each unit is left out, the dearest first (`single`: its own kept
/// model's work a µs), where the set without it is expected to cost less. Of the sets, those `short` of room at the
/// error as read are none where another is not, nor is a set left so (`needed`).
fn least_expected(taken: &[usize], single: &dyn Fn(usize) -> f64, costs: &Costs, work: &dyn Fn(&[usize]) -> Option<[f64; 2]>, price: &dyn Fn(&[usize]) -> [f64; 2], short: &dyn Fn(&[usize]) -> bool) -> Vec<usize> {
    let again = costs.again;
    // The first set with room at the error as read: the sets before it are no candidates.
    let from = (0..taken.len()).find(|&j| !short(&taken[..=j]));
    // E is w a + b: where the bound below w leaves it above `against`, the least of the sets it is weighed against, the
    // set is not the least, and its rate is not made (none).
    let expect = |set: &[usize], next: Option<f64>, against: f64| -> Option<f64> {
        let [n, p] = price(set);
        let (a, b) = match next {
            Some(next) => (again + (1. - p) * n, p * next),
            None => (again * (1. + p) + n, 0.),
        };
        if costs.least * a + b > against {
            return None;
        }
        let [w, setup] = work(set)?;
        Some(w * a + setup + b)
    };
    let mut expected = vec![f64::INFINITY; taken.len()];
    let mut after = f64::INFINITY;
    for j in (0..taken.len()).rev() {
        if let Some(e) = expect(&taken[..=j], after.is_finite().then_some(after), after.min(costs.none.unwrap_or(f64::INFINITY))) {
            expected[j] = e;
        }
        after = after.min(expected[j]);
    }
    let Some(best) = (from.unwrap_or(0)..taken.len()).filter(|&j| expected[j].is_finite()).min_by(|&a, &b| expected[a].total_cmp(&expected[b])) else { return vec![] };
    if costs.none.is_some_and(|none| none <= expected[best]) {
        return vec![];
    }
    let (mut chosen, mut least) = (taken[..=best].to_vec(), expected[best]);
    // The set is the room's together: a unit taken early for its value may be needed no more once those after it are
    // kept. Each unit is left out, the dearest first, where the set without it is expected to cost less -- its next
    // pilot, where it reads a look without room, keeping again what the set holds.
    let mut dearest = chosen.clone();
    dearest.sort_by(|&a, &b| single(b).total_cmp(&single(a)).then(b.cmp(&a)));
    for u in dearest {
        let rest: Vec<usize> = chosen.iter().copied().filter(|&v| v != u).collect();
        if rest.is_empty() || (from.is_some() && short(&rest)) {
            continue;
        }
        if let Some(value) = expect(&rest, Some(least), least).filter(|&value| value < least) {
            (chosen, least) = (rest, value);
        }
    }
    chosen
}

/// The model's errors `fixed` less those of the units `set` (`means`) at the times read -- a read time's errors are
/// numbers (`Measured::means`) --, no less than nothing (`needed`).
fn without(fixed: &Fixed, means: &[Vec<[f64; 2]>], read: &[bool], set: &[usize]) -> Fixed {
    let model = fixed
        .model
        .iter()
        .enumerate()
        .map(|(k, m)| {
            if k == 0 || !read[k] {
                return *m;
            }
            remainder(*m, set.iter().map(|&u| &means[k][u]))
        })
        .collect();
    Fixed { model }
}

/// Each time's accounts of what is shown and of the total T (ledger.rs `Account`) by the run's estimates, the model's
/// errors per time (`Fixed`: on what is shown every atom's, `Measured::means`; on T the held atoms') and the step's
/// part, T's where read: the held share 1 − L weighs T's whole accounts (module documentation, # Stopping); at t = 0
/// what is shown is the request's state, without error, and T the initial atoms' estimate (module documentation,
/// # Times and windows): no step is taken and no model has acted there, so T's account is its statistics alone.
/// The share of the time's trajectories not counted is booked whole on what is shown, in the model's part
/// (estimate.rs, # What is counted: what no number of trajectories takes away), and leaves T no account
/// (`temperature_model`). The model's error on what is shown is at most 1, a probability's. T has an account where its
/// statistics, the model's error and its step error are all read. The times' accounts are made on the pool, each
/// time's by itself.
fn accounts(estimates: &[Estimate], fixed: &Fixed, step: Bound) -> (Vec<Account>, Vec<Option<Account>>) {
    estimates
        .par_iter()
        .zip(&fixed.model)
        .enumerate()
        .map(|(k, (e, a))| {
            if k == 0 {
                let temperature = temperature_account(e, 0., &Bound { shown: 0., temperature: 0. }, e.temperature_interval());
                return (Account::default(), temperature.statistics.is_finite().then_some(temperature));
            }
            let population = Account { model: probability_error(a[0] + e.uncounted), step: step.shown, statistics: e.shown_statistics() };
            let temperature = temperature_account(e, a[1], &step, e.temperature_interval());
            (population, (temperature.model.is_finite() && temperature.step.is_finite() && temperature.statistics.is_finite()).then_some(temperature))
        })
        .unzip()
}

/// T's account at a time (`accounts`) from its estimate, the units' error on T there `units`, the step's part `bound`
/// (a share of T's allowance) and the statistics' relative interval `interval`: the model's part (`temperature_model`)
/// and the statistics' weighed by the held share, the step's read so already (estimate.rs `between`), times the
/// allowance.
fn temperature_account(e: &Estimate, units: f64, bound: &Bound, interval: f64) -> Account {
    let h = 1. - e.loss;
    Account { model: weighed(h, temperature_model(e, units)), step: bound.temperature * e.temperature_target, statistics: weighed(h, interval) }
}

/// The model's part of T's account at a time whose units' error on T is `units` (relative): theirs where every
/// trajectory is counted; unbounded where some is not -- the part of T the held atoms of unknown fate make is bounded by
/// nothing the run read (estimate.rs, # What is counted).
fn temperature_model(e: &Estimate, units: f64) -> f64 {
    if e.uncounted > 0. { f64::INFINITY } else { units }
}

/// An error of the held atoms weighed by their share h, as an account weighs it: unbounded stays so at any share -- h
/// times it would be no number at h = 0.
fn weighed(h: f64, error: f64) -> f64 {
    if error.is_finite() { h * error } else { f64::INFINITY }
}

/// What the pilot measured of the model's units (SPEC §5.2): the errors of the folded units read at the states
/// (`Sampled`: the rows one after another, a row the read units' errors on the populations and on T in their order),
/// each read's time, group and row, whether it reads a lost atom's fate, and how each folded unit is booked
/// (`booked`).
struct Measured<'a> {
    errors: &'a [Booking],
    output: &'a [usize],
    group: &'a [usize],
    row: &'a [usize],
    lost: &'a [bool],
    booked: &'a [Booked],
}
impl Measured<'_> {
    /// Per time and unit the unit's error on what is shown (absolute) and on the held atoms' T (relative), of the
    /// reads of the pilot's groups but `left` (module documentation, # The model: where the atoms went). What is shown
    /// at a time is every atom's fate there -- a held atom's rows, P_s = E[A ρ_s], and a lost one's loss, L = E[1 −
    /// A] (estimate.rs, # The outputs are expectations) --, so the model errs on it by the mean over the atoms of the
    /// error of each one's fate: (1 − L) times the held atoms' mean, each at its state there, and L times the lost
    /// atoms', each along its path (`Booking::path`). T is the held atoms' alone. Where no read stands for a time
    /// the error is its bound there, 1 on what is shown and none on T (the run's atoms there are at most the share the
    /// pilot left unseen, and T has no account without a pilot's T: its look fits no number of trajectories and is not
    /// chased).
    /// A held state's booking is taken at the populations `estimates` give there (`Booking::at`: those of the groups the
    /// mean is over), and so is the share L. A unit booked at its bound (`booked`) errs by it at every read, held or
    /// lost, on what is shown and on T: it is booked as a read one whose every state reads its bound. The times are
    /// made on the pool, each by itself: a time's sums take its reads in their order.
    fn means(&self, outputs: usize, left: Option<usize>, estimates: &[Estimate]) -> Vec<Vec<[f64; 2]>> {
        self.means_of(outputs, left, estimates, &(0..self.booked.len()).collect::<Vec<_>>())
    }

    /// Per time the folded units' combined error (`means`): at each read, the state's own error -- the model with every
    /// unit folded against the model with all of them kept, there: every folded unit's error at the state together, each
    /// booked as `means` books it --, and their means. The units' errors at a state are its fold's parts, which add up to
    /// the state's (model/units.rs, # The errors: their shares of the state's rates and shifts add up to one), not signed
    /// differences; so the combined error is the units' means added, read in one pass -- the bounded units' bounds
    /// (`booked`) at every read beside the read ones'.
    fn combined(&self, outputs: usize, left: Option<usize>, estimates: &[Estimate]) -> Vec<[f64; 2]> {
        let bounded: f64 = self.booked.iter().map(|b| if let Booked::Bound(bound) = *b { bound } else { 0. }).sum();
        self.reduced(outputs, left, estimates, 1, &|row, p, lost, out| {
            out[0] = row.iter().fold([bounded; 2], |a, e| {
                let e = if lost { e.error } else { e.at(p) };
                [a[0] + e[0], a[1] + e[1]]
            });
        })
        .into_iter()
        .map(|o| o[0])
        .collect()
    }

    /// `means` of the units `units` alone, in their order.
    fn means_of(&self, outputs: usize, left: Option<usize>, estimates: &[Estimate], units: &[usize]) -> Vec<Vec<[f64; 2]>> {
        self.reduced(outputs, left, estimates, units.len(), &|row, p, lost, out| {
            for (out, &u) in out.iter_mut().zip(units) {
                *out = match self.booked[u] {
                    Booked::Read(place) if lost => row[place].error,
                    Booked::Read(place) => row[place].at(p),
                    Booked::Bound(bound) => [bound; 2],
                };
            }
        })
    }

    /// Per time `n` errors (`means`): each read gives them (`each`: its row of the read units, the time's held
    /// populations, whether it reads a lost atom's fate, into its `n` pairs), and the held reads' means and the lost
    /// reads' make them, by the held share there.
    fn reduced(&self, outputs: usize, left: Option<usize>, estimates: &[Estimate], n: usize, each: &(dyn Fn(&[Booking], &[f64; STATES], bool, &mut [[f64; 2]]) + Sync)) -> Vec<Vec<[f64; 2]>> {
        let width = read_units(self.booked);
        // Each time's reads, in their order.
        let mut reads: Vec<Vec<usize>> = vec![vec![]; outputs];
        for (read, &k) in self.output.iter().enumerate() {
            reads[k].push(read);
        }
        reads
            .par_iter()
            .enumerate()
            .map(|(k, reads)| {
                let p = &estimates[k].populations;
                let (mut held, mut lost, mut one) = (vec![[0.; 2]; n], vec![0.; n], vec![[0.; 2]; n]);
                let (mut holding, mut losing) = (0., 0.);
                for &read in reads.iter().filter(|&&read| Some(self.group[read]) != left) {
                    each(&self.errors[self.row[read] * width..][..width], p, self.lost[read], &mut one);
                    if self.lost[read] {
                        lost.iter_mut().zip(&one).for_each(|(sum, e)| *sum += e[0]);
                        losing += 1.;
                    } else {
                        held.iter_mut().zip(&one).for_each(|(sum, e)| *sum = [sum[0] + e[0], sum[1] + e[1]]);
                        holding += 1.;
                    }
                }
                // Every held atom's group reads its first held one, every lost atom's its first lost one: a kind none
                // reads is none of these groups' atoms, and weighs nothing -- no T where no atom is held. Read by
                // neither, the time is unread.
                if holding == 0. && losing == 0. {
                    return vec![[1., f64::INFINITY]; n];
                }
                let h = 1. - estimates[k].loss;
                let mean = |sum: f64, count: f64| if count > 0. { sum / count } else { 0. };
                held.iter().zip(&lost).map(|(held, lost)| [h * mean(held[0], holding) + (1. - h) * mean(*lost, losing), mean(held[1], holding)]).collect()
            })
            .collect()
    }
}

/// The units' errors (`Measured::means`) added up per time (populations, T; SPEC §4), every folded unit's and what the
/// reach leaves out of the initial ensemble (of the atoms, of their kinetic energy; model/reach.rs `Model::left_out`): the
/// whole a keep's pricing takes a unit's error out of (`Level`). The error booked is the combined one, measured
/// (`model_error`); the two are one number where the states read every time alike.
fn sums(means: &[Vec<[f64; 2]>], unsampled: [f64; 2]) -> Vec<[f64; 2]> {
    means.iter().map(|units| units.iter().fold(unsampled, |a, m| [a[0] + m[0], a[1] + m[1]])).collect()
}

/// The model's error at a time after t = 0 from the sum `whole` of its units' there and what the reach leaves out
/// (`sums`): a population's at most 1.
fn model_error_at(whole: [f64; 2]) -> [f64; 2] {
    [probability_error(whole[0]), whole[1]]
}

/// The model's error booked per time (SPEC §4; module documentation, # The model): the folded units' combined error
/// measured at the states (`Measured::combined`) and what the reach leaves out of the initial ensemble (of the atoms, of
/// their kinetic energy; model/reach.rs `Model::left_out`), a population's at most 1; none at t = 0, the request's state.
fn model_error(combined: &[[f64; 2]], unsampled: [f64; 2]) -> Vec<[f64; 2]> {
    combined.iter().enumerate().map(|(k, c)| if k == 0 { [0.; 2] } else { model_error_at([c[0] + unsampled[0], c[1] + unsampled[1]]) }).collect()
}

/// The bandwidth of a model's cluster's generator by its levels (`Pricing::rated_by`, step.rs `Price::grown`): the largest
/// |λ_a* − λ_b| over its states' levels in its frames, E_s − θ_s − iΓ_s/2 (Γ an explicit excited state's decay) -- what
/// its couplings and the light's shifts move them by left out: a state's own bandwidth, its cluster's before, is the
/// bound below.
fn bandwidth(model: &Model, cluster: usize) -> f64 {
    let c = &model.clusters()[cluster];
    let levels: Vec<crate::linalg::C64> = c
        .states
        .iter()
        .zip(&c.frames)
        .map(|(&s, frame)| crate::linalg::C64::new(model.energies[s] - model.omega(frame), if s < GROUND { 0. } else { -0.5 * model.decay[(s - GROUND, s - GROUND)].re }))
        .collect();
    levels.iter().flat_map(|a| levels.iter().map(move |b| (a.conj() - *b).abs())).fold(0., f64::max)
}

/// A state a mean over the run is read at (`one_each`): a pilot trajectory's at one time, with its group's place
/// among the pilot's and its weight in the mean, the time its trajectory holds its atom (µs).
struct Pick<'a> {
    state: &'a State,
    group: usize,
    weight: f64,
    /// The trajectory's jumps per µs over the time it holds its atom: its last state's jumps over that state's time.
    jumps: f64,
}

/// The states a mean over the run is read at -- the rates a model is priced by (`Pricing`), a table's whole error
/// (`attempt`) --: one state of every trajectory of `parts`, the pilot's groups with their states kept.
///
/// What is read is the mean of a function f of the state over the time the run's atoms are held, Σ_i ∫ f_i dt /
/// Σ_i T_i, T_i the time trajectory i holds its atom -- to its loss (its last state, `Part::lost`), or through the run:
/// a trajectory's work is its rate over the time it is held, and an error is booked on the atoms that are held. A
/// trajectory is the ensemble's independent draw and its states along the run are not, so each trajectory gives one
/// state, f at one time of H_i, the times after t = 0 it holds its atom at, for the whole of its time: f_i·T_i, the
/// one-point rule, and the mean is Σ_i T_i f_i / Σ_i T_i (`over_the_run`). One that is lost before the first time
/// gives its state at t = 0, the start of the path its loss ends (module documentation, # The model: where the atoms
/// went), for its own time: a run whose atoms are all lost so is priced at the states they were at, not at nothing.
/// For the rule to read the whole run the points are spread evenly over it: trajectory i takes the time of H_i that
/// lies the fraction
/// u_i = (ρ_i + ½)/n of the way through it, ρ_i its rank among the n trajectories -- the midpoints of n equal parts of
/// the run. The rank of the member m of the group at the place g of G is ρ = m·G + g: a group's own members lie a
/// whole 1/(members) of the run apart, and the groups are shifted against one another by 1/n. So every group reads
/// the whole run by itself -- leaving one out, as the pricing's jackknife does, leaves no stretch of the run unread
/// and reads the run's own change in time as no spread --, and all of them together read it at n evenly spaced
/// points. A function of the trajectory's number alone: its group's place and its own place in the group. A
/// trajectory that ends with no outcome before the first time (estimate.rs, # What is counted) has no state to give,
/// and weighs nothing.
fn one_each<'a>(parts: &[&'a Part]) -> Vec<Pick<'a>> {
    let groups = parts.len();
    let mut picks = vec![];
    for (group, part) in parts.iter().enumerate() {
        // The group's trajectories in its members' order: every one is held at t = 0.
        let members = part.samples.first().map_or(&[][..], Vec::as_slice);
        for (member, first) in members.iter().enumerate() {
            let held: Vec<&State> = part.samples.iter().skip(1).filter_map(|at| at.iter().find(|state| state.index == first.index)).collect();
            let lost = part.lost.iter().find(|(_, state)| state.index == first.index).map(|(_, state)| state);
            // Its states to pick from, and its last: where it was lost, its last state is where it was found lost.
            let states = if held.is_empty() { lost.map(|_| vec![first]).unwrap_or_default() } else { held };
            let Some(last) = lost.or(states.last().copied()) else { continue };
            let through = ((member * groups + group) as f64 + 0.5) / (members.len() * groups) as f64;
            let state = states[((through * states.len() as f64) as usize).min(states.len() - 1)];
            picks.push(Pick { state, group, weight: last.sample.at.t, jumps: f64::from(last.jumps) / last.sample.at.t });
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

/// The pilot's groups but one (or all of them) as the pricing reads them (`Pricing`): the estimates and the units' mean
/// errors over them (per time and unit), and the current model's W there; every model's error taken less the doubt
/// the reads leave on it (`doubt`: the scout's; none on the pilot).
struct Level<'a> {
    left: Option<usize>,
    estimates: std::borrow::Cow<'a, [Estimate]>,
    means: std::borrow::Cow<'a, [Vec<[f64; 2]>]>,
    /// The units' mean errors added up (`sums`).
    whole: Vec<[f64; 2]>,
    /// Per time the trajectories its looks need at the current model's error, and the times after t = 0 by that
    /// need, the largest first.
    needs: Vec<f64>,
    order: Vec<usize>,
    /// The trajectories the current model needs there: the largest of `needs`.
    need: f64,
    /// What the reach leaves out (`sums`), as the looks take it, and the step's part (`keep`).
    unsampled: [f64; 2],
    step: Bound,
    /// Per time the doubt on the model's error (`lowered`).
    doubt: Vec<[f64; 2]>,
}
impl<'a> Level<'a> {
    /// The level of the groups but `left` from their estimates and the units' mean errors over them, beside the step's
    /// part `step`, the model's errors (the units' with what the reach leaves out) less the `doubt` per time.
    fn new(left: Option<usize>, estimates: std::borrow::Cow<'a, [Estimate]>, means: std::borrow::Cow<'a, [Vec<[f64; 2]>]>, step: Bound, unsampled: [f64; 2], doubt: &[[f64; 2]]) -> Level<'a> {
        let whole = sums(&means, unsampled);
        let fixed = clear(&Fixed { model: model_error(&sums(&means, [0.; 2]), unsampled) }, doubt);
        let mut needs = vec![0.0_f64; whole.len()];
        for look in looks(&estimates, &fixed, step) {
            if look.fits() {
                needs[look.output] = needs[look.output].max(look.need());
            }
        }
        let mut order: Vec<usize> = (1..whole.len()).collect();
        order.sort_by(|&a, &b| needs[b].total_cmp(&needs[a]));
        let need = needs.iter().copied().fold(0., f64::max);
        Level { left, estimates, means, whole, needs, order, need, unsampled, step, doubt: doubt.to_vec() }
    }

    /// The trajectories the looks need beside the step's part and a model's error `model` (per time, less the level's
    /// doubt) that is nowhere more than the level's own -- a unit's removed, every unit's. A look's need falls with the
    /// model's error, so a time needs no more than it does at the level's own (`needs`): the times are asked by that
    /// need, the largest first, and no further once the next one's is no more than what those before it need now. The
    /// largest over every time, for one or two times' looks.
    fn need_beside(&self, model: &dyn Fn(usize) -> [f64; 2]) -> f64 {
        let mut most = 0.;
        for &k in &self.order {
            if self.needs[k] <= most {
                break;
            }
            let (m, d) = (model(k), self.doubt[k]);
            let fixed = Fixed { model: vec![[0.; 2], [lowered(m[0], d[0]), lowered(m[1], d[1])]] };
            most = looks(&[self.estimates[0].clone(), self.estimates[k].clone()], &fixed, self.step).iter().filter(|l| l.fits()).map(Look::need).fold(most, f64::max);
        }
        most
    }

    /// The trajectories the looks need beside the units' errors but those of the units `set` and the step's share: a read
    /// error leaves the sum by subtraction (`sums`); at a time no state was read at every unit's error is its bound
    /// alike (`Measured::means`), its population's leaves as a read one does, and on T the other units' bounds stand --
    /// with every unit removed, what the reach leaves out.
    fn need_without_all(&self, set: &[usize]) -> f64 {
        self.need_beside(&|k| {
            let less = set.iter().fold([0.; 2], |a, &u| [a[0] + self.means[k][u][0], a[1] + self.means[k][u][1]]);
            let rest = if set.len() < self.means[k].len() { f64::INFINITY } else { self.unsampled[1] };
            [probability_error(self.whole[k][0] - less[0]), if less[1].is_finite() { self.whole[k][1] - less[1] } else { rest }]
        })
    }

    /// The trajectories the looks need beside the step's share alone: every unit kept, what the reach leaves out left.
    fn need_alone(&self) -> f64 {
        self.need_beside(&|_| model_error_at(self.unsampled))
    }
}

/// The work formula (module documentation, # The model): the counted work a trajectory spends per µs at a state under a
/// model, and under the models that keep more units, as the run counts it -- the step's own count through its cluster's
/// generator (step.rs `Price`) with the model's counts at a node of the cluster, a middle with a step's scattering over
/// τ, and the jumps' photons at the trajectory's own rate of jumps (model.rs `Model::priced`) --, with near tables: the
/// current model's own (the attempt's), and for a kept model the axes its tables would take (tabulate.rs
/// `Model::priced_axes`). A kept model is priced of its structure alone -- its clusters, its drives' levels and its
/// plans (`Model::rekept`): no table is made and nothing eliminated.
struct Formula<'a> {
    model: &'a Model,
    kept: &'a [Unit],
    units: &'a [Unit],
    reference: [f64; GROUND],
    centre: V3,
    /// The current model's near tables; per cluster its table's axes and every coordinate it was made over, the axes'
    /// number and their coordinates' count at a point (`Model::table_axes`; none where it has no table).
    tables: &'a Tables,
    axes: Vec<Option<(Vec<Key>, Vec<Key>, (usize, f64))>>,
    /// Where the atoms were (the rated states' places, set by `Pricing::new`), and the range each coordinate moves across
    /// over them, as a kept model's axes ask it (`Model::priced_axes`).
    points: Vec<V3>,
    moving: Mutex<HashMap<Key, f64>>,
    /// The attempt's ladder: a model's first rung (`rung`); and its setup's counts (`potential`).
    ladder: Ladder,
    setup: Setup,
}

impl<'a> Formula<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(model: &'a Model, kept: &'a [Unit], units: &'a [Unit], reference: [f64; GROUND], centre: V3, tables: &'a Tables, ladder: Ladder, setup: Setup) -> Formula<'a> {
        let running = model.running();
        let axes = (0..model.clusters().len()).map(|c| tables.of(c, &running).map(|t| model.table_axes(t))).collect();
        Formula { model, kept, units, reference, centre, tables, axes, points: vec![], moving: Mutex::new(HashMap::new()), ladder, setup }
    }

    /// A model's first rung on the attempt's ladder (`Ladder::first`), of the entries its steps sample under the running
    /// lights (model/assemble.rs `Model::sampled`); none where no rung within `MOST_STEPS` is admissible.
    fn rung(&self, model: &Model) -> Option<f64> {
        self.ladder.first(&model.sampled(&model.running()))
    }

    /// A model's prepared potential, one evaluation's counted work (potential.rs `Expansion::work`) as a step's
    /// reference reads it: at the trap's centre, of the reference's populations, to the second order -- a near light's
    /// elimination there where the trap light's drives are folded at the atom (milliseconds), a far form's few sums where
    /// they are not (R2_RECOIL: 3.2·10⁻⁴ s against 2.6·10⁻⁷ s, measured).
    fn potential(&self, model: &Model) -> f64 {
        model.expand(self.centre, &self.reference, &model.prepared(), 2, None).work
    }

    /// The step's price (step.rs `Price`) at states of the current model, each its cluster, its point and its τ: on the
    /// middle of its cluster at the trap's centre with the tables -- a price needs the generator's sizes and its
    /// spectrum, not the state's own point: one middle a cluster serves every state --, on the path of the state's own
    /// momentum and velocity.
    fn prices(&self, states: &[(usize, crate::model::Point, f64)]) -> Vec<Price> {
        let run = Run::tabulated(self.model, self.tables, self.reference);
        let mut middles: Vec<Option<crate::model::Middle>> = (0..self.model.clusters().len()).map(|_| None).collect();
        states
            .iter()
            .map(|&(cluster, at, tau)| {
                let middle = middles[cluster].get_or_insert_with(|| run.middle(cluster, self.centre, 0.));
                Price::of(middle, at.kappa, at.kappa.map(|k| run.hbar_over_mass() * k), tau)
            })
            .collect()
    }

    /// The current model priced on its tables' axes (`Model::priced`).
    fn own(&self) -> Priced {
        self.model.priced(&self.axes.iter().map(|a| a.as_ref().map_or((0, 0.), |a| a.2)).collect::<Vec<_>>())
    }

    /// The model that keeps the units `set` beside the kept ones, of its structure alone (`Model::rekept`), and its price
    /// (`Model::priced`), each cluster's near forms on the axes its table would take (`Model::priced_axes`: those of the
    /// current tables of the clusters it holds -- of every current table where none of them has one --, and the
    /// coordinates new to them that weigh there beyond the tables' shares), and its first rung (`rung`). None where the
    /// model refuses the keep (`Model::new`) or no rung is admissible for it: a keep whose beats no step within
    /// `MOST_STEPS` resolves -- a coherence of photons an optical frequency apart, a beat of 4.8·10⁹ rad/µs asking
    /// τ < 1.3·10⁻⁹ µs -- makes a model no run can step, its work unbounded (`Pricing::rated_by`). And its potential's
    /// work an evaluation (`potential`).
    fn kept(&self, set: &[usize]) -> Option<(Model, Priced, f64, f64)> {
        let with = self.model.rekept(&[self.kept, &set.iter().map(|&u| self.units[u].clone()).collect::<Vec<_>>()].concat()).ok()?;
        let rung = self.rung(&with)?;
        let moves = |key: Key, moved: &dyn Fn() -> f64| -> f64 {
            if let Some(&known) = self.moving.lock().expect("the coordinates' moves").get(&key) {
                return known;
            }
            let made = moved();
            self.moving.lock().expect("the coordinates' moves").insert(key, made);
            made
        };
        let axes: Vec<(usize, f64)> = (0..with.clusters().len())
            .map(|c| {
                let held = |o: usize| with.cluster_of(self.model.clusters()[o].states[0]) == c;
                let tabled = self.axes.iter().enumerate().any(|(o, table)| table.is_some() && held(o));
                let (mut axes, mut known) = (vec![], vec![]);
                for (o, table) in self.axes.iter().enumerate() {
                    if let Some((a, k, _)) = table.as_ref().filter(|_| held(o) || !tabled) {
                        axes.extend_from_slice(a);
                        known.extend_from_slice(k);
                    }
                }
                with.priced_axes(c, &self.points, &axes, &known, &moves)
            })
            .collect();
        let priced = with.priced(&axes);
        let potential = self.potential(&with);
        Some((with, priced, rung, potential))
    }

    /// The rate at a state of the current model's cluster `cluster` -- the step's price there `price` (`prices`), its τ
    /// and its trajectory's jumps a µs -- under a model priced `priced`: the price grown where the model's cluster that
    /// holds the state's is larger (`Price::grown`, its bandwidth at least its levels' own, `bandwidth`), with that
    /// cluster's counts at a node, the middle with a step's scattering over τ, and the photons.
    ///
    /// Beside the step, the model's prepared potential (`potential`: its work an evaluation): a step's reference, and the
    /// trajectory's initial draw over the run (`Setup::per_draw` evaluations).
    #[allow(clippy::too_many_arguments)]
    fn rate(&self, model: &Model, priced: &Priced, cluster: usize, price: &Price, tau: f64, jumps: f64, potential: f64) -> f64 {
        let states = &self.model.clusters()[cluster].states;
        let c = model.cluster_of(states[0]);
        let held = &model.clusters()[c];
        let price = if held.states == *states { *price } else { price.grown(held.states.len(), held.ground, price.bandwidth.max(bandwidth(model, c))) };
        (price.work(priced.costs[c]) + priced.assembly) / tau + jumps * priced.photons + potential * (1. / tau + self.setup.per_draw / self.ladder.run)
    }

    /// A model's setup a µs of the run (the rate's unit: the expected total work's, `least_expected`): its potential's
    /// evaluations at the attempt's setup (`Setup::evaluations`), each at its work, over the run's length.
    fn setup(&self, potential: f64) -> f64 {
        potential * self.setup.evaluations / self.ladder.run
    }
}

/// The pricing of keeps (SPEC §5.2; module documentation, # The model): the predicted saving of total work W = w·N by
/// keeping a folded unit, as a share of the current model's -- N the trajectories the looks need at the model's
/// errors and the step's share (`Look::need` on the pilot's estimates), and for the kept model the pilot its new attempt runs as well (the
/// current one's is spent); w the mean of its rates (`rates`) --, spread by the jackknife over the pilot's groups (each recomputed
/// without one: its estimates, the units' errors, the rates). A share, not the work saved: both models' N are the one
/// variance the pilot estimates over their rooms (N ∝ σ̂²), and σ̂²'s spread, common to both, scales the work saved
/// without ever changing its sign; in the share it cancels, and what spreads is what the keep changes -- the rooms and
/// the rates. A model's w is the mean of its rates over the run's held states, read at one state of every pilot
/// trajectory (`one_each`, `over_the_run`), the same states for every model.
struct Pricing<'a> {
    formula: Formula<'a>,
    measured: &'a Measured<'a>,
    pilot: &'a [&'a Part],
    fixed: &'a Fixed,
    scales: &'a Scales,
    unsampled: [f64; 2],
    /// The states the rates are read at, the step's price at each by the current model (`Formula::prices`, made once
    /// asked), and the current model's rates there.
    rated: Vec<Pick<'a>>,
    prices: OnceLock<Vec<Price>>,
    own: Vec<f64>,
    /// The trajectories a kept model's attempt runs again before it stands where this one does.
    again: f64,
    /// Every group together.
    all: Level<'a>,
}

impl<'a> Pricing<'a> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        formula: Formula<'a>,
        measured: &'a Measured<'a>,
        means: &'a [Vec<[f64; 2]>],
        pilot: &'a [&'a Part],
        estimates: &'a [Estimate],
        fixed: &'a Fixed,
        step: Bound,
        scales: &'a Scales,
        unsampled: [f64; 2],
        again: f64,
        doubt: &[[f64; 2]],
    ) -> Pricing<'a> {
        let rated = one_each(pilot);
        let mut formula = formula;
        formula.points = rated.iter().map(|pick| pick.state.sample.at.r).collect();
        let all = Level::new(None, estimates.into(), means.into(), step, unsampled, doubt);
        Pricing { formula, measured, pilot, fixed, scales, unsampled, rated, prices: OnceLock::new(), own: vec![], again, all }
    }

    /// The step's price at each rated state by the current model (`Formula::prices`).
    fn prices(&self) -> &[Price] {
        self.prices.get_or_init(|| self.formula.prices(&self.rated.iter().map(|pick| (pick.state.sample.cluster, pick.state.sample.at, pick.state.tau)).collect::<Vec<_>>()))
    }

    /// A model's rates at the rated states (`Formula::rate`), each at the step the model would run at there: the state's
    /// τ, or the model's first rung `rung` where that is shorter -- the step the beats a keep brings in ask before any
    /// check reads it (`Formula::kept`). The current model's rung is at least the τ its states ran at.
    fn rated_by(&self, model: &Model, priced: &Priced, rung: f64, potential: f64) -> Vec<f64> {
        self.rated.iter().zip(self.prices()).map(|(pick, price)| self.formula.rate(model, priced, pick.state.sample.cluster, price, pick.state.tau.min(rung), pick.jumps, potential)).collect()
    }

    /// The current model's rates (`Formula::own`).
    fn own(&self) -> Vec<f64> {
        let rung = self.formula.rung(self.formula.model).expect("the current model's attempt stands on an admissible rung");
        self.rated_by(self.formula.model, &self.formula.own(), rung, self.formula.potential(self.formula.model))
    }

    /// The rates of the model that keeps the units `set` beside the kept ones (`Formula::kept`), and its setup a µs of the
    /// run (`Formula::setup`: the current model's is spent); none for a set the model refuses to keep (its folds must
    /// exist, `Model::new`) or no step within the cap runs (its beats): their folds' errors stand.
    fn rates(&self, set: &[usize]) -> Option<(Vec<f64>, f64)> {
        self.formula.kept(set).map(|(with, priced, rung, potential)| (self.rated_by(&with, &priced, rung, potential), self.formula.setup(potential)))
    }

    /// A bound below every kept set's mean rate (`needed`): the step's price at each rated state with no count at a
    /// node, no middle and no photon -- what no keep takes away: a keep's clusters are at least as large (`Price::grown`).
    fn least(&self) -> f64 {
        let rates: Vec<f64> = self.rated.iter().zip(self.prices()).map(|(pick, price)| price.work([0.; 3]) / pick.state.tau).collect();
        self.mean(&rates, None)
    }

    /// The jackknife's levels: the pilot's groups but each one in turn, the levels made on the pool, each by itself.
    fn leave_one_out(&self) -> Vec<Level<'a>> {
        (0..self.pilot.len())
            .into_par_iter()
            .map(|left| {
                let parts: Vec<&Part> = self.pilot.iter().enumerate().filter(|(g, _)| *g != left).map(|(_, p)| *p).collect();
                let estimates = summarize(&reads_of(&parts), self.scales);
                let means = self.measured.means(self.fixed.model.len(), Some(left), &estimates);
                Level::new(Some(left), estimates.into(), means.into(), self.all.step, self.unsampled, &self.all.doubt)
            })
            .collect()
    }

    /// The mean over the run of the rates at the rated states of the groups but `left` (`over_the_run`).
    fn mean(&self, rates: &[f64], left: Option<usize>) -> f64 {
        over_the_run(&self.rated, rates.iter().copied(), left)
    }

    /// The share of the current model's W at a level that a kept model saves, which needs `need` trajectories at the
    /// rates `rates`: a model's W (per µs of run) is its mean rate times its product's trajectories (`need`), the kept
    /// model's with the pilot its new attempt runs.
    fn share_of(&self, level: &Level, need: f64, rates: &[f64]) -> f64 {
        let spent = self.mean(&self.own, level.left) * level.need;
        if spent > 0. { 1. - self.mean(rates, level.left) * (need + self.again) / spent } else { 0. }
    }

    /// The most any keep could save, spread over the jackknife's levels `others` (`keep`): every unit's error removed
    /// (`Level::need_alone`), its pilot run again, at the current model's work a µs -- a kept model's is no less: a fold
    /// is made to cost less than what it folds, and a keep that pays by its rate alone is not looked for (module
    /// documentation, # One rule for a keep). The old engine's gate on every spend (cost.rs `pays`: what it could save
    /// beyond the need's own error), read before a kept model is built: pricing builds and rates them.
    fn could_save(&self, others: &[Level]) -> Read {
        Read { value: self.share_of(&self.all, self.all.need_alone(), &self.own), spread: spread(others.iter().map(|l| self.share_of(l, l.need_alone(), &self.own)), others.len() as f64) }
    }

}

/// Histograms of one coordinate of the held atoms at the listed outputs on one grid symmetric about 0 that holds them all
/// (the bins' centres), every bin as wide as the grid's spacing: a listed output with no atom carries the one before it.
/// The number of bins is for the display alone -- no estimate reads a histogram --: the square-root choice, ⌊√n⌋ for the
/// n atoms of the fullest listed output (bins of about √n atoms each), made odd so that 0 is a bin's centre, and three at
/// least -- the contract's grid has two points or more (contract §2.2), and the least odd count that has is three.
fn histograms(listed: &[Vec<(V3, V3)>], outputs: &[usize], value: &dyn Fn(&(V3, V3)) -> f64) -> Distributions {
    let reach = listed.iter().flatten().map(|s| value(s).abs()).fold(f64::MIN_POSITIVE, f64::max);
    let most = listed.iter().map(Vec::len).fold(0, usize::max);
    let bins = ((most as f64).sqrt() as usize | 1).max(3);
    // Every bin as wide as the grid's spacing w = 2·reach/bins, the bins tiling [−reach, reach] exactly (the outermost
    // centred w/2 inside the farthest atom): a share over its bin's width is the density, as the app reads it -- every
    // cell, the end ones too, one spacing wide (src/motion-chart.ts `density`). Bins centred on ±reach would hold atoms on
    // their inner halves alone, and read half the density there.
    let width = 2. * reach / bins as f64;
    let grid: Vec<f64> = (0..bins).map(|i| (i as f64 - (bins / 2) as f64) * width).collect();
    let mut last: Vec<f64> = (0..bins).map(|i| if i == bins / 2 { 1. } else { 0. }).collect();
    let probabilities = listed
        .iter()
        .map(|s| {
            // A bin's share is its count over the atoms', one division: n added shares of 1/n can round past one, and a
            // probability above one is no probability (the app's reader refuses it).
            if !s.is_empty() {
                let mut counts = vec![0_usize; bins];
                for x in s {
                    counts[((value(x) / width).round() as isize + (bins / 2) as isize).clamp(0, bins as isize - 1) as usize] += 1;
                }
                last = counts.iter().map(|&c| c as f64 / s.len() as f64).collect();
            }
            last.clone()
        })
        .collect();
    Distributions { grid, samples: outputs.to_vec(), probabilities }
}

/// What the result's assembly takes beside the estimates: the plan, the run's step (µs), the scales, the request's initial
/// diagonal, the analysis times (µs; the run's two ends until the pilot's are read) and the pilot's fixed errors at
/// them, the step's part of every account -- the step's share until the pilot's two levels are `checked` --, the
/// ledger's items and note, and the method's description.
struct Context {
    plan: Plan,
    tau: f64,
    scales: Scales,
    initial: [f64; STATES],
    at: Vec<f64>,
    fixed: Fixed,
    /// What the atoms outside the reach add to the model's error at each analysis time (`apart`), in `fixed` with the
    /// pilot's: nothing until the product reads them.
    outside: Vec<[f64; 2]>,
    step: Bound,
    checked: Option<Checked>,
    ledger: Ledger,
    method: String,
}

/// Where a time t lies on the line through the analysis times `at` (µs, from t = 0; module documentation, # The analysis
/// times): the two analysis times about it and its share of the way from the first to the second -- held at the first
/// analysis time after t = 0 before it (what is booked at t = 0 is no model's and no step's, `accounts`, and the line
/// from it is neither's) and at the last after it; t = 0 at its own.
fn bracket(at: &[f64], t: f64) -> (usize, usize, f64) {
    if t <= 0. {
        return (0, 0, 0.);
    }
    let j = at.partition_point(|&a| a < t).max(2).min(at.len() - 1);
    if j < 2 {
        return (1, 1, 0.);
    }
    (j - 1, j, ((t - at[j - 1]) / (at[j] - at[j - 1])).clamp(0., 1.))
}

/// The point at the share s of the way from a to b; a value that is no bound on one side is none on the line between
/// (unbounded stays so), and the ends are the values themselves.
fn lerp(a: f64, b: f64, s: f64) -> f64 {
    if s == 0. { a } else if s == 1. { b } else { (1. - s) * a + s * b }
}

/// Values at the analysis times `at` at the times `times` (`bracket`): the line through them.
fn interpolated(at: &[f64], values: &[[f64; 2]], times: &[f64]) -> Vec<[f64; 2]> {
    times
        .iter()
        .map(|&t| {
            let (a, b, s) = bracket(at, t);
            std::array::from_fn(|i| lerp(values[a][i], values[b][i], s))
        })
        .collect()
}

/// Accounts at the analysis times `at` at the times `times` (`bracket`): each part on the line through them; a T with no
/// account on a side has none between.
fn accounts_along(at: &[f64], population: &[Account], temperature: &[Option<Account>], times: &[f64]) -> (Vec<Account>, Vec<Option<Account>>) {
    let mix = |x: &Account, y: &Account, s: f64| Account { model: lerp(x.model, y.model, s), step: lerp(x.step, y.step, s), statistics: lerp(x.statistics, y.statistics, s) };
    times
        .iter()
        .map(|&t| {
            let (a, b, s) = bracket(at, t);
            let temperature = match (temperature[a], temperature[b]) {
                (Some(x), Some(y)) => Some(mix(&x, &y, s)),
                (x, _) if s == 0. => x,
                (_, y) if s == 1. => y,
                _ => None,
            };
            (mix(&population[a], &population[b], s), temperature)
        })
        .unzip()
}

/// A step (µs) of a run of the length `run` as the result writes it. A step is the run over a whole number N of steps
/// (`first_rung`, and halvings of it), so its digits say nothing beyond N; it is written to one significant digit more
/// than N has decimal digits, and N is the run over the step as written, rounded. Rounded to d significant digits the
/// step reads τ(1 + ε) with |ε| ≤ ½·10^(1−d); the run over that is N/(1 + ε), within N|ε|/(1 − |ε|) of N, which is less
/// than ½ where 10^(d−1) ≥ N + 1 -- d − 1 the digits of N. (30 steps of a 100.05 µs run: 3.34 µs; 1548 of 1005 µs:
/// 0.64922 µs. The run ends with the last output's window, `Plan::end`.)
fn step_reading(run: f64, step: f64) -> String {
    let count = (run / step).round() as u64;
    significant(step, count.to_string().len() as i32 + 1).to_string()
}

/// The result of the estimates of `trajectories` trajectories (contract §2.2): what is shown at t = 0 the request's
/// initial state, the motion there the initial atoms' as at every output (module documentation, # Outputs and windows);
/// an output with no atom held carries the last one's populations, one with fewer than two atoms the last T read with
/// its judgement of the validated regime (`classical`: a statement about that T) and n̄, their error unknown; one whose T is its own but whose
/// jackknife reads no error (some group's removal leaves fewer than two atoms) says so; each flagged, apart.
fn assemble(c: &Context, estimates: &[Estimate], reads: &[Estimate], listed: &[Vec<(V3, V3)>], counts: &Counts, trajectories: u64) -> Solution {
    let (times, outputs) = (&c.plan.times, c.plan.times.len());
    let hm = c.scales.hbar_over_mass;
    let w = c.scales.frequencies;
    let classical = |t: &V3| -> [bool; 3] { std::array::from_fn(|a| classical(t[a], w[a])) };
    let mut ledger = c.ledger.clone();
    // The accounts the run's decisions and its stop stand on, at the analysis times of the estimates `reads`, and at the
    // outputs the line through them (# The analysis times): the ledger flags where the stop reads -- not an output's own
    // statistics, a noisy estimate of the same spread, the largest of a thousand of them above it --, and so do T's
    // allowances. The model's error at the outputs likewise, for what is read of it there.
    let (at_reads, at_temperature) = accounts(reads, &c.fixed, c.step);
    let (population, temperature) = accounts_along(&c.at, &at_reads, &at_temperature, times);
    let allowed: Vec<f64> = times.iter().map(|&t| bracket(&c.at, t)).map(|(a, b, s)| lerp(reads[a].temperature_target, reads[b].temperature_target, s)).collect();
    let fixed = Fixed { model: interpolated(&c.at, &c.fixed.model, times) };
    // The step's item: the largest it is booked at over the accounts (weighed by the held share as the totals are). And
    // the check in the method's words: the steps compared, the pairs, the resolution, and what is booked -- what the
    // reads the rung stands by do not exclude (`Levels::told`, `checked`).
    let mut step = String::new();
    if let Some(check) = &c.checked {
        let largest = (1..outputs).flat_map(|k| temperature[k].iter().map(|a| a.step).chain([population[k].step])).fold(0., f64::max);
        let (doubled, third) = (step_reading(c.plan.end, 2. * c.tau), check.third.map(|s| ("4τ", step_reading(c.plan.end, s))));
        let levels = third.as_ref().map_or(format!("τ 与 2τ（{doubled} μs）"), |(name, at)| format!("τ、2τ（{doubled} μs）与 {name}（{at} μs）"));
        let how = if check.saw {
            "相邻两级之差按步长误差与步长的二次方成正比各自换算成 τ 的误差，两个估计都在误差预算分给步长的部分之内才用这个步长；计入的是两者中较大的估计"
        } else {
            "检查没有结论（三个步长的比较既没有确定在误差预算分给步长的部分之内，也没有明显超出），而增加成对轨迹和把步长再减半都已不可能或不再有用；计入的是检查不能排除的最大值（方均根加上它的置信区间）"
        };
        ledger.item(&format!("时间步长 τ = {} μs（全程不变）", step_reading(c.plan.end, c.tau)), largest, &format!("步长收敛检查：先导轨迹用 {levels} 各算一遍、逐条比较，取差在各输出时刻上的方均根（已扣除统计噪声）；{how}"));
        let relative = if check.resolution[1].is_finite() { format!("、在温度上低于其允许误差的 {} 时检查分辨不出", percent(check.resolution[1])) } else { " 时检查分辨不出，在温度上无法估计".into() };
        step = format!("（步长收敛检查：{} 对先导轨迹用 {levels} 各算一遍（同一初始原子、同一组随机数），相邻两级之差在各输出时刻上的方均根（已扣除统计噪声）按二阶换算成本步长的误差；{how}；步长误差的方均根在布居上低于 {}{relative}）", check.pairs, percent(check.resolution[0]));
    }
    // What the unresolved atoms and those past the reach add (`apart`): its largest over the analysis times, weighed by
    // the held share.
    for (i, what) in ["布居", "温度"].into_iter().enumerate() {
        let largest = (1..reads.len()).map(|k| c.outside[k][i] * (1. - reads[k].loss)).fold(0., f64::max);
        ledger.item(&format!("计算中跑出估计的活动范围的原子（{what}）"), largest, "在每个输出时刻统计：曾有一步因速率太大而时间步长对它不适用、或因散射率超过原子任何状态的上限而停下的原子（从那一步起），按它们所占的比例全部计入（布居上是原子数的比例，温度上是动能的比例）；温度高于活动范围的温度 T_r 时，再按热分布在活动范围之外的比例 s^(T_r/T) − s 计入（s 是活动范围本身留在外面的比例），这些原子上的折叠不另行判定");
    }
    ledger.close(times, &population, &temperature, &allowed);
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
        loss_probability: estimates.iter().map(|e| e.loss).collect(),
        velocity: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.1[a] * hm)),
        position: std::array::from_fn(|a| histograms(listed, &c.plan.listed, &|x| x.0[a])),
        classical: Vec::with_capacity(outputs),
        trajectories,
        time_step_us: c.tau,
        method: format!("{}；{trajectories} 条轨迹（每 {GROUP} 条一组，组内按拉丁超立方抽取初始原子），时间步长 {} μs{step}。", c.method, step_reading(c.plan.end, c.tau)),
        reason: None,
        approximations: ledger,
    };
    // The first output where every trajectory counted is lost; the outputs no trajectory is counted at (estimate.rs,
    // # What is counted), and those where T is read and some trajectory is not counted.
    let (mut empty, mut carried, mut unread) = (None, vec![], vec![]);
    let none: Vec<usize> = (1..outputs).filter(|&k| estimates[k].trajectories == 0.).collect();
    let partly: Vec<usize> = (1..outputs).filter(|&k| estimates[k].uncounted > 0. && estimates[k].motion.is_some()).collect();
    for (k, e) in estimates.iter().enumerate() {
        // At t = 0 every trajectory holds its atom (`run`), and a result rests on two at least: nothing is carried there.
        let last = || k.checked_sub(1).expect("t = 0 holds every trajectory's atom, two at least in a result");
        s.populations.push(if k == 0 { c.initial } else if e.held > 0. { e.populations } else { s.populations[last()] });
        if e.held == 0. && e.trajectories > 0. && empty.is_none() {
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
                let last = last();
                let (t, total, v, n) = (s.temperature_uk[last], s.temperature_total_uk[last], s.mean_velocity_mps[last], s.mean_occupation[last]);
                s.temperature_uk.push(t);
                s.temperature_errors_uk.push([None; 3]);
                s.temperature_total_uk.push(total);
                s.temperature_total_errors_uk.push(None);
                s.mean_velocity_mps.push(v);
                s.mean_occupation.push(n);
                s.classical.push(s.classical[last]);
                if e.held > 0. {
                    carried.push(k);
                }
            }
        }
    }
    if let Some(k) = empty {
        s.approximations.flag(format!("{} 起已无原子留在光阱中（丢失概率 1）：此后的布居、温度与分布沿用此前的值，仅供参考", when(times[k])));
    }
    if !carried.is_empty() {
        s.approximations.flag(format!("{}：仍在光阱中的原子不足两个，温度与 n̄ 沿用此前的值，误差未知", span(&carried, times)));
    }
    if !unread.is_empty() {
        s.approximations.flag(format!("{}：仍在光阱中的原子太少（去掉任一组后不足两个），温度的统计误差未知", span(&unread, times)));
    }
    // T shown with its statistics where its model error or its step error is unread.
    let unbooked: Vec<usize> = (1..outputs)
        .filter(|&k| estimates[k].motion.as_ref().is_some_and(|m| m.temperature_errors.iter().all(Spread::read)) && !(fixed.model[k][1].is_finite() && c.step.temperature.is_finite()))
        .collect();
    if !unbooked.is_empty() {
        s.approximations.flag(format!("{}：这些时刻仍在光阱中的原子太少，无法估计模型近似或时间步长给温度带来的误差", span(&unbooked, times)));
    }
    // The trajectories that ended with no outcome, by why (`Counts`), and what that leaves (estimate.rs, # What is
    // counted).
    let unmade = "在某一步上，光学扇区的噪声算子 K 做不出";
    for (stopped, why) in [
        (&counts.failed, "的状态在数值上无法再推进".to_string()),
        (&counts.unmade[0], format!("{unmade}（激发态块或基态块的 Schur 形式不收敛；这不应发生，是引擎的缺陷）")),
        (&counts.unmade[1], format!("{unmade}（ℳ 奇异：按构造 Q 扇区以约 Γ/2 或更快弛豫，ℳ 应当可逆；这不应发生，是引擎的缺陷）")),
        (&counts.unmade[2], format!("{unmade}（算出的数不是有限数；这不应发生，是引擎的缺陷）")),
    ] {
        if let Some(t) = stopped.since {
            s.approximations.flag(format!("{} 条轨迹{why}（最早在 {}）：此后它们不计入估计，所占的比例全部计入布居与丢失的误差", stopped.count, when(t)));
        }
    }
    if let Some(t) = counts.runaway.since {
        s.approximations.flag(format!("{} 条轨迹的散射率超过了原子任何状态可能的上限（某个绝热消去远离它的适用范围，最早在 {}）：它们在那里停下，此后按停下时的状态计入，所占的比例全部计入误差", counts.runaway.count, when(t)));
    }
    if !partly.is_empty() {
        s.approximations.flag(format!("{}：有轨迹不计入估计，温度的误差无法估计", span(&partly, times)));
    }
    if !none.is_empty() {
        s.approximations.flag(format!("{}：没有一条轨迹计入估计，丢失概率、布居与温度沿用此前的值，布居与丢失的误差至少 100%", span(&none, times)));
    }
    // The worst output beyond the population target: what keeps it there -- where no trajectory is counted, that; else
    // the largest part, the model's with the share not counted that it holds.
    if let Some((k, a)) = population.iter().enumerate().max_by(|x, y| x.1.total().total_cmp(&y.1.total())) {
        if a.total() > POPULATION_TARGET {
            let uncounted = estimates[k].uncounted;
            let cause = if estimates[k].trajectories == 0. {
                "没有一条轨迹计入估计".to_string()
            } else if a.statistics >= a.model.max(a.step) {
                if trajectories >= MOST_TRAJECTORIES {
                    format!("统计误差为目标的 {:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}", a.statistics / POPULATION_TARGET)
                } else {
                    format!("统计误差 {}（{trajectories} 条轨迹，计算未完成）", percent(a.statistics))
                }
            } else if a.model >= a.step && uncounted > 0. {
                format!("模型近似误差 {}（其中不计入估计的轨迹占 {}）", percent(a.model), percent(uncounted))
            } else if a.model >= a.step {
                format!("模型近似误差 {}", percent(a.model))
            } else {
                format!("时间步长误差 {}", percent(a.step))
            };
            s.reason = Some(format!("t = {} 处布居的估计误差 {}（目标 {}）：{cause}", when(times[k]), percent(a.total()), percent(POPULATION_TARGET)));
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::TEMPERATURE_TARGET;

    /// A step as written gives its count back (`step_reading`): the run over the reading, rounded, is the number of
    /// steps, for every rung -- the run over an even number, halved down to the steps' cap -- of runs short and long;
    /// and the digits are no more than that takes: with one fewer some rungs' counts are lost.
    #[test]
    fn a_step_as_written_gives_its_count_back() {
        let default = 100. + 100. / 1998.;
        assert_eq!((step_reading(default, default / 30.), step_reading(default, default / 15.)), ("3.34".to_string(), "6.67".to_string()));
        assert_eq!(step_reading(1005., 1005. / 1548.), "0.64922");
        let mut fewer = 0;
        for run in [default, 1005., 20.025, 1e7] {
            for pairs in 1..=64_u64 {
                for count in (0..).map(|halvings| 2 * pairs << halvings).take_while(|&count| count <= MOST_STEPS) {
                    let step = run / count as f64;
                    let read: f64 = step_reading(run, step).parse().unwrap();
                    assert_eq!((run / read).round() as u64, count, "{run} µs over {count}: {read}");
                    fewer += ((run / significant(step, count.to_string().len() as i32)).round() as u64 != count) as u32;
                }
            }
        }
        assert!(fewer > 0);
    }

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
    /// - The step's edge with a kick's curvature ω_k² (`admissible`): at ωτ = π without one and for a softer well, below
    ///   π for a stiffer one, at ω_kτ = 2 on a flat reference.
    /// - The first rung is a rung of the ladder down to `MOST_STEPS` steps: there is none for a run of 10⁹ µs in the
    ///   74.65 kHz trap (3.35 µs at the most: 3·10⁸ steps), nor at a frequency or a speed that is no number or
    ///   unbounded -- and the count ends.
    #[test]
    fn the_first_rung_is_admissible_and_its_double_is_for_the_typical_atom() {
        let khz = |f: f64| std::f64::consts::TAU * f * 1e-3;
        let hm = hbar_over_mass(1.4432e-25);
        let fastest = |t: f64| (2. * (MOST_TRAJECTORIES as f64 / (CONFIDENCE * CONFIDENCE) + 1.).ln() * KT_PER_UK * t * hm).sqrt();
        let (fast, typical) = (fastest(30.), typical_speed(hm, 30., fastest(30.)));
        assert!((fast - 0.2185).abs() < 5e-4 && (typical - 0.0928).abs() < 5e-4, "{fast} {typical}");
        assert_eq!(first_rung(10., f64::INFINITY, khz(74.65), 0., &[], fast, typical), Some(2.5));
        assert_eq!(first_rung(10., f64::INFINITY, khz(1.), 0., &[[0., 0.]], fast, typical), Some(5.));
        assert!((TAU / (16.1 * fast) - 1.79).abs() < 5e-3 && (TAU / (16.1 * typical) - 4.20).abs() < 1e-2);
        assert_eq!(first_rung(10., f64::INFINITY, khz(74.65), 0., &[[0., 16.1]], fast, typical), Some(10. / 6.));
        let k = 16.106 / 2. * 10f64.sqrt();
        assert!((TAU / (k * fast) - 1.13).abs() < 5e-3 && (TAU / (k * typical) - 2.66).abs() < 1e-2);
        assert_eq!(first_rung(10., f64::INFINITY, khz(100.), 0., &[[0., k]], fast, typical), Some(1.));
        let both = |tau: f64, w: f64, kick: f64, sampled: &[[f64; 2]], fast: f64, typical: f64| admissible(tau, w, kick, sampled, fast) && admissible(2. * tau, w, kick, sampled, typical);
        for (run, w, kick, sampled, fast, typical) in [
            (10., khz(910.7), 0., vec![[0., 16.1]], fast, typical),
            (20., khz(79.), 0., vec![[0., 16.1]], fastest(141.), typical_speed(hm, 141., fastest(141.))),
            (16., khz(20.3), 0., vec![[0., 32.2]], fast, typical),
            (10., khz(74.65), 0., vec![[0., 16.1]], 0.0089, 0.0038),
            (10., khz(74.65), 0., vec![[0., 0.], [2., 16.1], [0.5, 8.]], fast, typical),
            (100., khz(100.), 0., vec![[0., k]], fast, typical),
            (100., khz(20.), 0.35 * 0.35, vec![[0., 16.1]], 0.05, 0.017),
            (100., khz(74.65), -0.5 * khz(74.65).powi(2), vec![], fast, typical),
        ] {
            let tau = first_rung(run, f64::INFINITY, w, kick, &sampled, fast, typical).unwrap();
            let pairs = (run / (2. * tau)).round();
            assert!((run / (2. * pairs) - tau).abs() <= 1e-12 * tau && both(tau, w, kick, &sampled, fast, typical), "{run} {tau}");
            assert!(pairs == 1. || !both(run / (2. * (pairs - 1.)), w, kick, &sampled, fast, typical), "{run} {tau}: a larger rung is admissible");
        }
        assert!(admissible(1.25, khz(74.65), 0., &[[0., 16.1]], fast) && !admissible(2.5, khz(74.65), 0., &[[0., 16.1]], fast) && admissible(2.5, khz(74.65), 0., &[[0., 16.1]], typical));
        // The step's edge: at π for a kick without curvature and for a softer well; below π for a stiffer one, where
        // (θ/2) tan(θ/2) = ω²/ω_k² (0.5477 π at ω_k = ω, 0.3056 π at ω_k² = 4 ω²); the kick's own ω_k τ = 2 on a flat
        // reference.
        let w = khz(100.);
        let edge = |kick: f64, lo: f64, hi: f64| admissible(lo * PI / w, w, kick, &[], 0.) && !admissible(hi * PI / w, w, kick, &[], 0.);
        assert!(edge(0., 0.9999, 1.0001) && edge(-0.5 * w * w, 0.9999, 1.0001) && edge(-9. * w * w, 0.9999, 1.0001));
        assert!(edge(w * w, 0.5477, 0.5478) && edge(4. * w * w, 0.3056, 0.3057) && edge(0.1 * w * w, 0.9096, 0.9097));
        assert!(admissible(1.9999 / w, 0., w * w, &[], 0.) && !admissible(2.0001 / w, 0., w * w, &[], 0.));
        // The constructed example (numbers.status.md §1.1): 1 µK in a tweezer of 20 kHz and a light well of
        // ω_k = 0.35 rad/µs, K = 16.1 /µm, v_max = 0.050 µm/µs. The trap alone allows 25 µs, the light 7.8 µs; the
        // step's edge is at 5.5 µs.
        assert!(admissible(7.7, khz(20.), 0., &[[0., 16.1]], 0.05) && admissible(5.4, khz(20.), 0.35 * 0.35, &[[0., 16.1]], 0.05) && !admissible(5.6, khz(20.), 0.35 * 0.35, &[[0., 16.1]], 0.05));
        assert_eq!(first_rung(1e9, f64::INFINITY, khz(74.65), 0., &[], fast, typical), None);
        assert_eq!(first_rung(10., f64::INFINITY, f64::NAN, 0., &[], fast, typical), None);
        assert_eq!(first_rung(10., f64::INFINITY, khz(74.65), f64::NAN, &[], fast, typical), None);
        assert_eq!(first_rung(10., f64::INFINITY, khz(74.65), 0., &[[0., 16.1]], f64::INFINITY, typical), None);
        assert_eq!(first_rung(10., f64::INFINITY, khz(74.65), 0., &[[0., 16.1]], fast, f64::NAN), None);
    }

    /// The typical atom is no faster than the reach's fastest (`typical_speed`), so the first rung of a reach grown
    /// towards its top -- its temperature without bound, its set and its fastest speed bounded -- goes to the one at
    /// the set's own speed and no lower (m20: a reach of 11 498 µK, typical 1.82 µm/µs against its largest 0.397): over
    /// 30 µK to 10⁸ µK at the fastest speed of a reach that saturates at 0.397 µm/µs, the first rung falls with the
    /// temperature and never below the one whose companion is taken at 0.397 -- where the set is at its top, that one.
    #[test]
    fn the_typical_atom_is_no_faster_than_the_set_s_fastest() {
        let (hm, khz) = (hbar_over_mass(1.4432e-25), |f: f64| std::f64::consts::TAU * f * 1e-3);
        let (top, sampled) = (0.397, [[0., 16.1]]);
        let at_top = first_rung(1000., f64::INFINITY, khz(100.), 0., &sampled, top, top).unwrap();
        let mut last = f64::INFINITY;
        for t in (0..=70).map(|i| 30. * 10f64.powf(i as f64 / 10.)) {
            let fastest = top * (1. - (-t / 300.).exp());
            let tau = first_rung(1000., f64::INFINITY, khz(100.), 0., &sampled, fastest, typical_speed(hm, t, fastest)).unwrap();
            assert!(tau >= at_top && tau <= last * (1. + 1e-12), "{t} µK: {tau} µs after {last}, at the set's speed {at_top}");
            if t >= 1e5 {
                assert_eq!(tau, at_top, "{t} µK");
            }
            last = tau;
        }
    }

    /// The ladder begins within the physics prior (`prior`, `first_rung`): T's step error (k_max v_T τ)² at the step's
    /// share of T's allowance at the reach's temperature, the request's to begin with. PGC3D's Rb87 at 30 µK on D2 (k = 8.0528 /µm; its tweezer's
    /// 100, 100 and 19 kHz, a zero point of 1.8 µK below 30 T_rec = 10.9 µK): δT/T = 0.375 and τ ≤ 0.950 µs, where its
    /// ladder's top by the edges alone, 1.425 µs, was left. Over temperatures, wave numbers and traps the phase a step
    /// of the prior advances, squared, is that share to the rounding, and no light's pattern bounds nothing. Over runs,
    /// priors and edges the first rung is the largest even part of the run within the prior that the edges admit with
    /// its double -- the next larger one is beyond the prior or not admissible --, it is the edges' own rung wherever
    /// the prior is above that, and it never exceeds the prior.
    #[test]
    fn the_first_rung_is_the_prior_s_where_the_edges_allow_it() {
        let khz = |f: f64| std::f64::consts::TAU * f * 1e-3;
        let hm = hbar_over_mass(1.4432e-25);
        let pgc = Scales { hbar_over_mass: hm, frequencies: [khz(100.), khz(100.), khz(19.)] };
        let (k, t) = (8.0528, 30.);
        assert!((semiclassical([0.; 2], &pgc) - 1.75).abs() < 0.01 && (semiclassical([1., k * k], &pgc) - 10.86).abs() < 0.01);
        let tau = prior(k, hm, t, &pgc);
        assert!((allowance(t, semiclassical([1., k * k], &pgc)) - 0.3755).abs() < 1e-4 && (tau - 0.950).abs() < 1e-3, "{tau}");
        for (k, t, frequencies) in [(8.0528, 30., pgc.frequencies), (7.3699, 1.5, [khz(150.); 3]), (8.0528, 300., [khz(20.), khz(80.), khz(5.)]), (16.1, 0.2, [khz(400.), khz(400.), khz(90.)])] {
            let scales = Scales { hbar_over_mass: hm, frequencies };
            let phase = k * (hm * KT_PER_UK * t).sqrt() * prior(k, hm, t, &scales);
            let share = step_bound()[1] * allowance(t, semiclassical([1., k * k], &scales));
            assert!((phase * phase / share - 1.).abs() < 1e-12, "{k} {t}: {phase} {share}");
        }
        assert_eq!(prior(0., hm, t, &pgc), f64::INFINITY);
        let fastest = |t: f64| (2. * (MOST_TRAJECTORIES as f64 / (CONFIDENCE * CONFIDENCE) + 1.).ln() * KT_PER_UK * t * hm).sqrt();
        let (fast, typical) = (fastest(30.), typical_speed(hm, 30., fastest(30.)));
        let both = |tau: f64, w: f64, kick: f64, sampled: &[[f64; 2]], fast: f64, typical: f64| admissible(tau, w, kick, sampled, fast) && admissible(2. * tau, w, kick, sampled, typical);
        for (run, w, kick, sampled) in [
            (1026.3, khz(100.), 0., vec![[0., 16.1]]),
            (10., khz(74.65), 0., vec![[0., 16.1]]),
            (100., khz(20.), 0.35 * 0.35, vec![[0., 16.1]]),
            (100., khz(910.7), 0., vec![[0., 0.], [2., 16.1], [0.5, 8.]]),
            (5000., khz(1.), 0., vec![]),
        ] {
            let edges = first_rung(run, f64::INFINITY, w, kick, &sampled, fast, typical).unwrap();
            for prior in [1e-3, 0.0954, 0.3581, 0.7125, 0.950, 1.425, 3.7, 1e3, f64::INFINITY] {
                let tau = first_rung(run, prior, w, kick, &sampled, fast, typical).unwrap();
                let pairs = (run / (2. * tau)).round();
                assert!((run / (2. * pairs) - tau).abs() <= 1e-12 * tau && tau <= prior * (1. + 1e-12) && tau <= edges && both(tau, w, kick, &sampled, fast, typical), "{run} {prior}: {tau}");
                assert!(pairs == 1. || run / (2. * (pairs - 1.)) > prior || !both(run / (2. * (pairs - 1.)), w, kick, &sampled, fast, typical), "{run} {prior}: {tau}, a larger rung is within both");
                assert!(prior < edges || tau == edges, "{run} {prior}: {tau} below the edges' {edges}");
            }
        }
    }

    /// Reads of held atoms' states, a row of the table each, in their order (`Measured`).
    fn rows(n: usize) -> (Vec<usize>, Vec<bool>) {
        ((0..n).collect(), vec![false; n])
    }

    /// `n` units, every one read, each at its place in a row (`Measured`).
    fn all_read(n: usize) -> Vec<Booked> {
        (0..n).map(Booked::Read).collect()
    }

    /// The units booked at their bounds (`booked`): the smallest bounds first, equal ones in the units' order, while they
    /// add up to no more than the unseen share of the most trajectories; the rest read, numbered in the units' order --
    /// a unit without a bound, one past the share, an infinite bound and one that is no number. And what is so booked
    /// (`Measured`): a bounded unit is booked as a read one whose every state reads its bound -- at every read output,
    /// held or lost, on what is shown and on T, in the units' means and in the combined error alike --, and an output
    /// none of the groups reads is unread for it too.
    #[test]
    fn units_bounded_within_the_unseen_share_are_booked_at_their_bounds() {
        let share = unseen(MOST_TRAJECTORIES as f64);
        let bounds = [Some(0.4 * share), None, Some(0.3 * share), Some(f64::NAN), Some(0.3 * share), Some(0.1 * share), Some(f64::INFINITY)];
        let (third, tenth) = (Booked::Bound(0.3 * share), Booked::Bound(0.1 * share));
        assert_eq!(booked(&bounds), vec![Booked::Read(0), Booked::Read(1), third, Booked::Read(2), third, tenth, Booked::Read(3)]);
        assert_eq!(booked(&[Some(0.6 * share), Some(0.6 * share)]), vec![Booked::Bound(0.6 * share), Booked::Read(0)]);
        // Two units: the first read, the second at the bound b -- against both read, the second's every state at b.
        let b = 0.7 * share;
        let read = |error: [f64; 2], moved: f64| Booking { error, moved, into: 0b11 };
        let first = [read([0.01, 0.02], 0.004), read([0.03, 0.01], 0.), read([0.02, 0.05], 0.01)];
        let both: Vec<Booking> = first.iter().flat_map(|&a| [a, read([b, b], 0.)]).collect();
        let (output, group, row, lost) = ([1, 1, 2], [0, 1, 0], [0, 1, 2], [false, true, false]);
        let read_both = Measured { errors: &both, output: &output, group: &group, row: &row, lost: &lost, booked: &all_read(2) };
        let bounded = Measured { errors: &first, output: &output, group: &group, row: &row, lost: &lost, booked: &[Booked::Read(0), Booked::Bound(b)] };
        let estimates: Vec<Estimate> = (0..4).map(|_| Estimate { loss: 0.25, populations: std::array::from_fn(|s| if s < 2 { 0.1 } else { 0. }), ..estimate([10.; 3], 0., 8.) }).collect();
        for left in [None, Some(0), Some(1)] {
            assert_eq!(bounded.means(4, left, &estimates), read_both.means(4, left, &estimates), "left {left:?}");
            for (k, (x, y)) in bounded.combined(4, left, &estimates).iter().zip(read_both.combined(4, left, &estimates)).enumerate() {
                for i in 0..2 {
                    assert!(x[i] == y[i] || (x[i] - y[i]).abs() <= 1e-15 * y[i].abs(), "left {left:?}, output {k}: {x:?} against {y:?}");
                }
            }
        }
        assert_eq!(bounded.means(4, None, &estimates)[3], vec![[1., f64::INFINITY]; 2]);
        assert_eq!(bounded.means(4, None, &estimates)[2][1], [0.75 * b, b]);
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

    /// The screen's outcomes on the mean squares between two levels, each against the square b of the step's share.
    /// The step is not halved where every mean square fits b, where a small one lies under a wide interval, where
    /// one lies above b but not clearly so among the looks, nor on a read below zero, however far; it is halved where
    /// a mean square is clear of b: the total T's, the loss's and a row's against the populations' share's square. The
    /// family is the run's few sizes (the loss, the rows, T): on the pilot's eleven groups a mean square of 50 b with a
    /// standard error of 5 b is clear among them, and one of 15 b -- clear at one look -- is not. A read that is no
    /// number is not beyond.
    #[test]
    fn the_screen_halves_the_step_on_a_clear_mean_square_alone() {
        let halved = |read: &Between| screen(read);
        let [sp, st] = step_bound();
        let (bp, bt) = (sp * sp, st * st);
        assert!(!halved(&at(Some(read(0.5 * bt, 0.1 * bt)))));
        assert!(!halved(&at(Some(read(0.5 * bt, bt)))));
        assert!(!halved(&at(Some(read(2. * bt, bt)))));
        assert!(halved(&at(Some(read(4. * bt, 0.01 * bt)))));
        assert!(!halved(&at(Some(read(-4. * bt, 0.01 * bt)))));
        assert!(!halved(&at(None)));
        let lost = Between { loss: read(2. * bp, 0.01 * bp), ..at(None) };
        let mut row = at(None);
        row.shown[3] = read(2. * bp, 0.01 * bp);
        assert!(halved(&lost) && halved(&row));
        assert_eq!(judged(&at(Some(pilot(50. * bt, 5. * bt)))).len(), 1 + STATES + 1);
        assert!(halved(&at(Some(pilot(50. * bt, 5. * bt)))) && !halved(&at(Some(pilot(50. * bt, 15. * bt)))) && pilot(50. * bt, 15. * bt).least(1.) > bt);
        assert!(!halved(&at(Some(read(f64::NAN, 0.)))) && !halved(&at(Some(Read { value: 1., spread: Spread::UNREAD }))));
    }

    /// What a rung of untested order asks for (`blind`), on the total T against b = s² = 0.2, the step's share of T's
    /// allowance squared (unlimited groups, t = 2). A far end within b asks nothing. A read of none with an interval of 2b could be within on twice
    /// its pairs -- what it holds falls as 1/N --; a read below zero the same: its far end is taken from zero. A mean
    /// square of b/2 under an interval of 2b has nothing clear of it: the far end 2.5 b could fall to b on 2.5 times
    /// the pairs. One of 1.5 b under an interval of b -- above b, and not clearly: the noise may have put it there --
    /// has b/2 clear: (2.5 − 0.5)/(1 − 0.5) = 4 times the pairs, where the value taken as it reads asked for none. One
    /// whose near end is at b or above could be within on no number of pairs. A row asks against the populations'
    /// share; the largest asking of the looks is taken. A look that is unread or no number excludes nothing
    /// (`readable`): no number of pairs.
    #[test]
    fn a_rung_of_untested_order_asks_for_the_pairs_that_could_bring_it_within() {
        let b = step_bound()[1].powi(2);
        let asks = |t: Read| blind(&at(Some(t)));
        assert_eq!(asks(read(0., 0.25 * b)), None);
        let close = |asked: Option<f64>, factor: f64| asked.is_some_and(|f| (f - factor).abs() < 1e-12);
        assert!(close(asks(read(0., b)), 2.) && close(asks(read(-1.5 * b, b)), 2.) && close(asks(read(0.5 * b, b)), 2.5) && close(asks(read(1.5 * b, 0.5 * b)), 4.));
        assert_eq!((asks(read(3. * b, 0.5 * b)), asks(read(1.5 * b, 0.2 * b))), (Some(f64::INFINITY), Some(f64::INFINITY)));
        let bp = step_bound()[0].powi(2);
        let mut row = at(Some(read(0., b)));
        row.shown[1] = read(0.5 * bp, bp);
        assert!(close(blind(&row), 2.5) && readable(&row));
        row.shown[1] = Read { value: 0.3, spread: Spread::UNREAD };
        assert!(blind(&row) == Some(f64::INFINITY) && !readable(&row));
        row.shown[1] = read(f64::NAN, 0.);
        assert!(blind(&row) == Some(f64::INFINITY) && !readable(&row) && readable(&at(None)));
    }

    /// The pairs a rung's own read asks for where the noise hides its difference (`remedy`): the factor `blind` gives,
    /// in whole groups, at most twice the pairs it has a batch -- 1.05 times 88 is 96, 7.1 times 88 a first batch of 176
    /// --, wherever a run can take them; none past the most trajectories a solve runs, nor for a read that excludes
    /// nothing.
    #[test]
    fn the_pairs_a_rung_asks_for_are_those_it_could_see_on_at_the_best() {
        assert_eq!((remedy(1.05, 88), remedy(7.1, 88), remedy(200., 88), remedy(f64::INFINITY, 88)), (Some(96), Some(176), None, None));
        assert_eq!(remedy(1.01, MOST_TRAJECTORIES), None);
    }

    /// A read as a run logged it: its value and its far end at one look, each over the bound b of its look, on `groups`
    /// groups (the far end's interval Student's t at one look: the standard error back from it).
    fn logged(value: f64, most: f64, groups: f64, b: f64) -> Read {
        let t = Spread { error: 1., groups }.interval();
        Read { value: value * b, spread: Spread { error: (most - value.max(0.)) * b / t, groups } }
    }

    /// A rung within on two levels reads the cheap third level where there is one, and else stands on its two (`verdict`;
    /// module documentation, # The step). e038's base at 0.589 µs read its total T against 1.178 µs at v/b −0.055 with a
    /// far end of 0.83 b on the pilot's eleven groups -- within --, and its T(1 ms) came out 79.1 against the converged
    /// 89.2 µK (−11.3 %) where two levels were all it was read on. The scout stands on it (the pilot reads it); the
    /// pilot reads its third level above where the ladder has the rung above and that rung stays in the model's domain,
    /// and else stands on its two levels, as an adaptive integrator does. A coarser pair that shows the error the two
    /// levels share -- a plateau of the error over τ and 2τ, read at 2τ against 4τ as 32 ± 1 b, lowered to 2 b, beyond
    /// the share -- does not show the order: the rung is left (e038's base path is caught by its 4τ). A third level
    /// within: the rung stands. Clearly beyond on its own read: left, on the scout as on the pilot.
    #[test]
    fn a_rung_reads_its_cheap_third_level_and_else_stands_on_its_two() {
        let b = step_bound()[1].powi(2);
        let own = at(Some(logged(-0.055, 0.83, 11., b)));
        assert!(blind(&own).is_none() && !screen(&own) && within(&own));
        assert_eq!((verdict(&own, None, false, false), verdict(&own, None, true, true), verdict(&own, None, true, false)), (Verdict::Stands, Verdict::Above, Verdict::Stands));
        let coarser = carried(&at(Some(pilot(32. * b, b))), 0.5);
        assert!(screen(&coarser) && verdict(&own, Some(&coarser), true, false) == Verdict::Left);
        let quiet = carried(&at(Some(pilot(4. * b, b))), 0.5);
        assert_eq!(verdict(&own, Some(&quiet), true, false), Verdict::Stands);
        let beyond = at(Some(pilot(4. * b, 0.1 * b)));
        assert_eq!((verdict(&beyond, None, false, false), verdict(&beyond, Some(&quiet), true, false)), (Verdict::Left, Verdict::Left));
    }

    /// A read that cannot see buys its pairs, and an upper level that shows the error leaves the rung (`verdict`,
    /// `blind`, `checked`). m20 at 0.392 µs read its loss against 0.784 µs at v/b 1.95 with a far end of 7.70 b on the
    /// pilot's eleven groups; the rung above's read, 0.784 against 1.568 µs, was 32.0 b with a far end of 49.1 b, lowered
    /// to the rung 2.0 and 3.1 b. Standing on the lesser read, a rule before booked none of it and the loss at 600 µs
    /// came out 0.587 against the converged 0.724. Its own estimate is beyond the share and not clearly: the noise hides
    /// it, and it asks for the pairs that could bring it within at the best (`blind`: 7.7 times); on 352 pairs it read
    /// 0.69 b, within, where the rung above's 2.0 b is not: the rung is left (m20 is caught by its upper level). Were it
    /// to stand unable to see, it would book what its reads leave open, √7.70 of the share on what is shown, and where it
    /// stood the larger estimate.
    #[test]
    fn a_read_that_cannot_see_buys_pairs_and_an_upper_level_beyond_leaves_the_rung() {
        let [sp, _] = step_bound();
        let bp = sp * sp;
        let (mut own, mut above) = (at(None), at(None));
        (own.loss, above.loss) = (logged(1.952, 7.702, 11., bp), logged(32.045, 49.088, 11., bp));
        let above = carried(&above, 0.5);
        assert!(!screen(&above) && verdict(&own, Some(&above), true, false) == Verdict::Unseeing);
        assert!(blind(&own).is_some_and(|factor| (factor - 7.702).abs() < 1e-9), "{:?}", blind(&own));
        let ((open, _), (estimate, _)) = (checked(&own, Some(&above), false), checked(&own, Some(&above), true));
        assert!((open.shown - (7.702 * bp).sqrt()).abs() < 1e-12 && open.shown > sp, "{open:?}");
        assert!((estimate.shown - (32.045 / 16. * bp).sqrt()).abs() < 1e-12, "{estimate:?}");
        let mut more = at(None);
        more.loss = logged(0.69, 1.3, 44., bp);
        assert_eq!(verdict(&more, Some(&above), true, false), Verdict::Left);
    }

    /// A rung that stands, and what it books (`verdict`, `worse`, `checked`). The total T read 0.2 ± 0.1 b at the rung
    /// and 0.15 ± 0.02 b from the coarser pair, lowered: both estimates within -- the rung stands, and books per look the
    /// larger estimate, here the own read's √0.2 b, its resolution √(2·0.1) b; a check that ended unable to see books
    /// the far end, √(0.2 + 2·0.1) b. A third level that reads no T holds nothing against the rung's T. A row the third
    /// reads larger is booked by the third's read. A read that is no number is unbounded: the whole on what is shown.
    #[test]
    fn a_rung_that_stands_books_the_larger_estimate() {
        let b = step_bound()[1].powi(2);
        let (own, third) = (at(Some(read(0.2 * b, 0.1 * b))), at(Some(read(0.15 * b, 0.02 * b))));
        assert_eq!(verdict(&own, Some(&third), true, true), Verdict::Stands);
        let (step, resolution) = checked(&own, Some(&third), true);
        assert!((step.temperature - (0.2 * b).sqrt()).abs() < 1e-15 && step.temperature < step_bound()[1] && step.shown == 0., "{step:?}");
        assert!((checked(&own, Some(&third), false).0.temperature - (0.4 * b).sqrt()).abs() < 1e-15);
        assert!((resolution[1] - (0.2 * b).sqrt()).abs() < 1e-15 && resolution[0] == 0., "{resolution:?}");
        assert_eq!((verdict(&own, Some(&at(None)), true, false), verdict(&at(None), Some(&third), true, false)), (Verdict::Stands, Verdict::Stands));
        let mut wide = third.clone();
        wide.shown[4] = read(1e-4, 1e-4);
        let ((step, _), (open, _)) = (checked(&own, Some(&wide), true), checked(&own, Some(&wide), false));
        assert!((step.shown - 1e-4f64.sqrt()).abs() < 1e-15 && (open.shown - 3e-4f64.sqrt()).abs() < 1e-15, "{step:?} {open:?}");
        let none = Between { loss: Read { value: f64::NAN, spread: Spread::UNREAD }, ..own.clone() };
        assert_eq!((checked(&none, Some(&third), true).0.shown, unexcluded(&at(None)).temperature, estimated(&at(None)).temperature), (1., f64::INFINITY, f64::INFINITY));
    }

    /// The reads a rung is told and booked by (`Levels::told`): its own on all its pairs -- the pilot's and the further
    /// pairs', 360 --, and its third level's carried to its step: the level above's as it is held (already lowered),
    /// the third level's step four times the rung's. A rung without has no third level to tell.
    #[test]
    fn a_rung_is_told_by_its_own_read_and_its_third_level_s() {
        let (b, tau, scales) = (step_bound()[1].powi(2), 1.4585, Scales { hbar_over_mass: 1., frequencies: [1.; 3] });
        let pilot_groups = || kept(11, GROUP as usize, 4, &|_, _| 4);
        let mut levels = Levels { fine: pilot_groups(), coarse: pilot_groups(), further: [kept(34, GROUP as usize, 4, &|_, _| 4), kept(34, GROUP as usize, 4, &|_, _| 4)], ..Levels::default() };
        let times = [0., 1., 2., 3., 4.];
        let (_, third, step, pairs) = levels.told(&times, f64::INFINITY, tau, &scales);
        assert_eq!((third.is_none(), step, pairs), (true, None, 360));
        let lowered = carried(&at(Some(pilot(0.8 * b, 0.1 * b))), 0.5);
        levels.above = Some(lowered.clone());
        let (_, third, step, _) = levels.told(&times, f64::INFINITY, tau, &scales);
        assert_eq!((third.unwrap().temperature, step), (lowered.temperature, Some(4. * tau)));
    }

    /// A look on T at 800 trajectories: the model's error, what the step's check leaves open, the statistics' interval (each
    /// relative).
    fn look(model: f64, step: f64, interval: f64) -> Look {
        Look { output: 1, target: TEMPERATURE_TARGET, account: Account { model, step, statistics: interval }, interval, trajectories: 800., shown: false }
    }

    /// The trajectories a look needs (`Look::need`): its interval within the room the model's error and the step's
    /// clear part leave, N₀ (S/R)²; it is met exactly where it needs no more than it has; no number fits where the two
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

    /// The units the looks need kept (`needed`), in one pass, none to spare. One output after t = 0, its T read exactly, the step at its
    /// share: T's look has room while the model's error on T is below √(1 − 1/5) of the target, 0.0894. The error read
    /// without a doubt on it: the far end is the error, and the next pilot reads it again -- its chance of a look
    /// without room one where the set leaves none, else none --; a set's work a µs the sum of its units', its product
    /// nothing beside its pilot (the reads exact), and keeping nothing no option where the look has no room, else
    /// nothing to spend. Three units of errors 0.06, 0.05 and 0.04 on T beside 0.05 of the rest (0.2), each adding the
    /// same work: the heaviest first, 0.14 and 0.09 still leave no room, 0.05 does -- all three, together: a set that
    /// leaves no room costs its pilot and then a larger set's. With 0.02, 0.08 and 0.03 beside 0.02 (0.15): the second
    /// alone, 0.07. With room from the start, none; at an output not read, or where no atom is held, none.
    #[test]
    fn the_units_the_looks_need_kept_are_taken_by_error_per_work_until_they_have_room() {
        let estimates = vec![estimate([10.; 3], 0., 88.), estimate([10.; 3], 0., 88.)];
        // A set's rate the sum of its units', and a bound below every set's.
        let needed = |estimates: &[Estimate], fixed: &Fixed, means: &[Vec<[f64; 2]>], held: &[f64], read: &[bool], least: f64, work: &dyn Fn(&[usize]) -> Option<f64>| {
            let price = |set: &[usize], beyond: &[usize]| -> [f64; 2] {
                let short = tight(estimates, &without(fixed, means, read, set), Bound::share(), read).iter().any(|k| !beyond.contains(k));
                [0., f64::from(u8::from(short))]
            };
            let step = Bound::share();
            let none = tight(estimates, fixed, step, read).is_empty().then_some(0.);
            needed(estimates, step, means, held, read, &Costs { least, floor: 0., again: 1., none }, &|set| work(set).map(|w| [w, 0.]), &|set| without(fixed, means, read, set), &|set| without(fixed, means, read, set), &price)
        };
        let summed = |rate: &dyn Fn(usize) -> Option<f64>, set: &[usize]| set.iter().map(|&u| rate(u)).sum::<Option<f64>>();
        let case = |errors: [f64; 3], rest: f64, held: f64, read: bool| {
            let means = vec![vec![[0.; 2]; 3], errors.iter().map(|&e| [0., e]).collect::<Vec<_>>()];
            let fixed = Fixed { model: vec![[0.; 2], [0., errors.iter().sum::<f64>() + rest]] };
            needed(&estimates, &fixed, &means, &[1., held], &[false, read], 1., &|set| summed(&|_| Some(1.), set))
        };
        assert_eq!(case([0.06, 0.05, 0.04], 0.05, 1., true), [0, 1, 2]);
        assert_eq!(case([0.02, 0.08, 0.03], 0.02, 1., true), [1]);
        assert_eq!(case([0.02, 0.03, 0.01], 0.02, 1., true), Vec::<usize>::new());
        assert_eq!(case([0.06, 0.05, 0.04], 0.05, 1., false), Vec::<usize>::new());
        assert_eq!(case([0.06, 0.05, 0.04], 0.05, 0., true), Vec::<usize>::new());
        // By error per work: the second unit's kept model three times dearer than the others', it is taken last -- the
        // first and the third leave 0.10, no room yet --; a unit whose keep is refused, never; one whose model is the
        // cheaper first.
        let means = vec![vec![[0.; 2]; 3], vec![[0., 0.06], [0., 0.05], [0., 0.04]]];
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.2]] };
        let with = |cost: [Option<f64>; 3]| needed(&estimates, &fixed, &means, &[1., 1.], &[false, true], cost.iter().flatten().copied().fold(f64::INFINITY, f64::min), &|set| summed(&|u| cost[u], set));
        assert_eq!(with([Some(1.), Some(3.), Some(1.)]), [0, 2, 1]);
        assert_eq!(with([Some(1.), None, Some(1.)]), [0, 2]);
        assert_eq!(with([Some(1.), Some(0.5), Some(1.)]), [1, 0, 2]);
        // A rate is made where a set asks for it, once, against a bound below every rate: the same set as with every rate
        // made. Twelve units of errors 0.004 to 0.048 on T (0.312 in all), their kept models' rates 1 to 7 by their place,
        // and the bound a half to the whole of the least: the units kept are the same, and at the least itself the
        // units' rates made fewer than the units that weigh.
        let means = vec![vec![[0.; 2]; 12], (0..12).map(|u| [0., 0.004 * (1 + (u * 5) % 12) as f64]).collect::<Vec<_>>()];
        let fixed = Fixed { model: vec![[0.; 2], [0., means[1].iter().map(|m| m[1]).sum::<f64>()]] };
        let rate = |u: usize| Some(1. + (u * 7 % 12) as f64 / 2.);
        let every: Vec<Option<f64>> = (0..12).map(rate).collect();
        let all = needed(&estimates, &fixed, &means, &[1., 1.], &[false, true], 0., &|set| summed(&rate, set));
        let lowest = every.iter().flatten().copied().fold(f64::INFINITY, f64::min);
        for share in [0.5, 0.8, 1.] {
            let made = std::cell::RefCell::new(std::collections::HashSet::<Vec<usize>>::new());
            let lazy = needed(&estimates, &fixed, &means, &[1., 1.], &[false, true], share * lowest, &|set| {
                let mut key = set.to_vec();
                key.sort_unstable();
                made.borrow_mut().insert(key);
                summed(&rate, set)
            });
            assert_eq!(lazy, all, "the bound at {share} of the least rate");
            let units = made.borrow().iter().filter(|set| set.len() == 1).count();
            assert!(units <= 12 && (share < 1. || units < 12), "the bound at {share} of the least rate: {units} units' rates made");
        }
        assert!(all.len() > 1 && all.len() < 12, "{all:?}");
        // None of the set to spare: of errors 0.03 and 0.30 on T beside 0.05, the first's kept model twenty times
        // cheaper, both are taken, the first for its value; the second alone leaves the room, and the first is spared.
        let means = vec![vec![[0.; 2]; 3], vec![[0., 0.03], [0., 0.30], [0., 0.]]];
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.38]] };
        let rate = [Some(0.1), Some(2.), Some(1.)];
        assert_eq!(needed(&estimates, &fixed, &means, &[1., 1.], &[false, true], 0.1, &|set| summed(&|u| rate[u], set)), [1]);
    }

    /// The set kept is the one of the least expected work (`needed`): the error read and its far end are its two ends.
    /// One output, T's look with room below 0.0894; three units of 0.04 each beside 0.05 (0.17), a pilot of 88 and a
    /// product of 500 trajectories whatever the set. The next pilot's chance of a look without room: one with one unit
    /// kept (0.13 left), 0.05 with two (0.09, at the edge), none with three. Each unit adding 1 % of the work a µs, two are
    /// kept -- a pilot more at 5 % costs less than the third unit's work over the product and the pilot; where the
    /// third unit adds nothing, all three; where each adds 30 %, still two: a set short of room costs a pilot and a larger
    /// set's work. The least of the three, against the sum written out.
    #[test]
    fn the_set_kept_is_the_one_of_the_least_expected_work() {
        let estimates = vec![estimate([10.; 3], 0., 88.), estimate([10.; 3], 0., 88.)];
        let means = vec![vec![[0.; 2]; 3], vec![[0., 0.04]; 3]];
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.17]] };
        let (held, read) = ([1., 1.], [false, true]);
        // The work a µs of a set of n units: one and the first n of `more`.
        let pick = |more: [f64; 3]| {
            let price = |set: &[usize], _: &[usize]| -> [f64; 2] { [500., [1., 1., 0.05, 0.][set.len()]] };
            let work = |set: &[usize]| Some([1. + more[..set.len()].iter().sum::<f64>(), 0.]);
            let costs = Costs { least: 1., floor: 500., again: 88., none: None };
            needed(&estimates, Bound::share(), &means, &held, &read, &costs, &work, &|_| Fixed { model: vec![[0.; 2]; 2] }, &|set| without(&fixed, &means, &read, set), &price).len()
        };
        // E3 = w3 (88 + 500), E2 = w2 (88 + 0.95 · 500) + 0.05 E3, E1 = w1 · 88 + min(E2, E3).
        let least = |more: [f64; 3]| {
            let w = [1. + more[0], 1. + more[0] + more[1], 1. + more.iter().sum::<f64>()];
            let e3 = w[2] * 588.;
            let e2 = w[1] * (88. + 475.) + 0.05 * e3;
            let e = [w[0] * 88. + e2.min(e3), e2, e3];
            (0..3).min_by(|&a, &b| e[a].total_cmp(&e[b])).unwrap() + 1
        };
        for (more, kept) in [([0.01; 3], 2), ([0.01, 0.01, 0.], 3), ([0.3; 3], 2)] {
            assert_eq!((least(more), pick(more)), (kept, kept), "{more:?}");
        }
    }

    /// Keeps are judged by the run's total work wherever the looks have room (`needed`, `Costs::none`), not only where
    /// they have none: GMD1C's class -- a model's error spread over many units, none of whose keeps pays alone, the
    /// first one's dearer a µs by its cluster's growth and every further one's by little (drives into one excited state,
    /// explicit once). One output, T read at 88 trajectories to a relative interval of 0.0937, the step at its share; the
    /// model's error on T 0.085 -- forty units of 0.0015 and 0.025 of the rest --, room for the statistics of 0.0278: the
    /// run needs about 1000 trajectories at the current work a µs, one. A set of k units: work a µs 1 + 0.5 + 0.001 k,
    /// its pilot (88) again and its product beside the error it leaves. Every unit's keep alone costs more than keeping
    /// nothing; the set kept costs less, and it is a large one. Where a keep's first unit multiplies the work a µs by
    /// thirty, nothing is kept: no set is expected to cost less.
    #[test]
    fn units_that_pay_only_together_are_kept_by_the_run_s_total_work() {
        let units = 40;
        let estimates = vec![estimate([10.; 3], 0., 88.), estimate([10.; 3], 0.4685, 88.)];
        let means = vec![vec![[0.; 2]; units], vec![[0., 0.0015]; units]];
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.06 + 0.025]] };
        let (held, read, step) = ([1., 1.], [false, true], Bound::share());
        let need = |set: &[usize]| looks(&estimates, &without(&fixed, &means, &read, set), step).iter().filter(|l| l.fits()).map(Look::need).fold(0., f64::max);
        let price = |set: &[usize], _: &[usize]| -> [f64; 2] { [need(set), 0.] };
        let (again, none) = (88., need(&[]));
        assert!((none / 1000. - 1.).abs() < 0.05, "the current model's need {none}");
        for growth in [0.5, 30.] {
            let work = |set: &[usize]| Some([1. + if set.is_empty() { 0. } else { growth + 0.001 * set.len() as f64 }, 0.]);
            let expected = |set: &[usize]| work(set).unwrap()[0] * (again + need(set));
            assert!((0..units).all(|u| expected(&[u]) > none), "growth {growth}: a unit's keep alone pays");
            let costs = Costs { least: 1., floor: need(&(0..units).collect::<Vec<_>>()), again, none: Some(none) };
            let kept = needed(&estimates, step, &means, &held, &read, &costs, &work, &|set| without(&fixed, &means, &read, set), &|set| without(&fixed, &means, &read, set), &price);
            if growth < 1. {
                assert!(kept.len() > units / 2 && expected(&kept) < none, "growth {growth}: kept {} units, expected {} against {none}", kept.len(), expected(&kept));
            } else {
                assert!(kept.is_empty(), "growth {growth}: kept {kept:?}");
            }
        }
    }

    /// The set kept on recorded prices (`least_expected`) is the one the kept models' runs choose: S3's pilot at 139 µK
    /// (2026-11-13), the order its keeps were taken in -- fifteen bare drives and, third, the cooling pair's dressing --,
    /// each set's product and the next pilot's chance of a look without room, and each set's work a µs by the formula and
    /// by its kept model's run with its own tables made. The formula prices every set with the dressing at 2.40 M, as the
    /// runs whose tables took its relative phase as an axis; the tables took it on 6 of 21 such sets (2.07 M on the
    /// others). Either way all sixteen are kept, the next pilot sure of room.
    #[test]
    fn the_set_kept_on_s3_s_recorded_prices_is_the_tabled_models_one() {
        // (the set by its places in the order, n, p, w by the formula, w by the run).
        let rows: [(u16, f64, f64, f64, f64); 31] = [
            (0xffff, 180.8, 0.0000, 2395525.1, 2065165.2),
            (0x7fff, 182.1, 0.0000, 2395525.1, 2065006.2),
            (0x3fff, 183.6, 0.0083, 2395525.1, 2065135.9),
            (0x1fff, 185.7, 0.0165, 2395525.1, 2065106.5),
            (0x0fff, 188.2, 0.0331, 2395525.1, 2396523.9),
            (0x07ff, 190.9, 0.0992, 2395525.1, 2393038.7),
            (0x03ff, 194.0, 0.1901, 2395525.1, 2391410.1),
            (0x01ff, 197.5, 0.2893, 2395525.1, 2394924.2),
            (0x00ff, 202.1, 0.5041, 2395525.1, 2386979.1),
            (0x007f, 208.5, 0.7851, f64::NAN, f64::NAN),
            (0x003f, 216.6, 0.9504, f64::NAN, f64::NAN),
            (0x001f, 230.7, 1.0000, f64::NAN, f64::NAN),
            (0x000f, 246.7, 1.0000, f64::NAN, f64::NAN),
            (0x0007, 268.1, 1.0000, f64::NAN, f64::NAN),
            (0x0003, 454.1, 1.0000, f64::NAN, f64::NAN),
            (0x0001, 699.0, 1.0000, f64::NAN, 2067345.8),
            (0xfffb, 184.0, 0.5289, 2067287.0, 2064093.0),
            (0xbfff, 182.1, 0.0000, 2395525.1, 2065253.4),
            (0xfff7, 184.7, 0.0083, 2395525.1, 2392360.4),
            (0xffdf, 185.0, 0.0000, 2395525.1, 2392536.6),
            (0xffef, 184.5, 0.0000, 2395525.1, 2070196.7),
            (0xfbff, 182.5, 0.0000, 2395525.1, 2064842.1),
            (0xffbf, 183.6, 0.0000, 2395525.1, 2065135.9),
            (0xf7ff, 182.5, 0.0000, 2395525.1, 2068200.3),
            (0xefff, 182.5, 0.0000, 2395525.1, 2395735.8),
            (0xfffd, 188.0, 0.0496, 2395525.1, 2065400.2),
            (0xff7f, 183.4, 0.0000, 2395525.1, 2065224.0),
            (0xfdff, 182.6, 0.0000, 2395525.1, 2065282.7),
            (0xfeff, 182.9, 0.0000, 2395525.1, 2068642.4),
            (0xfffe, 190.4, 0.1818, 2395525.1, 2064959.6),
            (0xdfff, 182.4, 0.0000, 2395525.1, 2065165.2),
        ];
        // Each unit's own kept model's work a µs, by the formula and by its run.
        let singles: [[f64; 2]; 16] = std::array::from_fn(|u| match u {
            2 => [2394452.8, 2395396.9],
            _ => [2067287.0, [2067345.8, 2060234.9, 0., 2067374.4, 2063182.3, 2054222.4, 2067357.0, 2067217.1, 2063251.7, 2064400.3, 2067287.0, 2063868.7, 2063182.3, 2067287.0, 2064517.6, 2067885.3][u]],
        });
        let row = |set: &[usize]| {
            let mask = set.iter().fold(0_u16, |m, &u| m | 1 << u);
            rows.iter().find(|r| r.0 == mask).unwrap_or_else(|| panic!("a set not recorded: {mask:#06x}"))
        };
        let kept = |by: usize| {
            let work = |set: &[usize]| Some(if by == 0 { row(set).3 } else { row(set).4 }).filter(|w| w.is_finite()).map(|w| [w, 0.]);
            least_expected(&(0..16).collect::<Vec<_>>(), &|u| singles[u][by], &Costs { least: 1509565.7, floor: 0., again: 88., none: None }, &work, &|set| [row(set).1, row(set).2], &|_| false)
        };
        assert_eq!((kept(0), kept(1)), ((0..16).collect(), (0..16).collect()));
    }

    /// The chance the keep is priced by is the next pilot's own (`Doubt::chance`, `needed`): a pilot of eleven groups of
    /// eight states, two outputs after t = 0, twelve units whose errors on T are heavy-tailed over the states -- a unit's
    /// error a_u X, X = (ln U)² of a uniform U, of mean 2 and variance 20 --, 0.17 in all where T's look has room below
    /// 0.0894. Over 40 pilots, each with the next pilot drawn on states of its own, and for each set of the heaviest
    /// units: the chance the pilot's jackknife gives that the next pilot reads the set's remaining error without room,
    /// against how often it does -- within the binomial's own spread of 40 draws, at every set (four units: 6 short, 11.1
    /// predicted; three: 38, 36.2). The error read alone (the set that leaves it room, no doubt on it) is short in a
    /// tenth of the pilots or more, as its chance says (6 of 40, 8.3): a keep taken to the error read asks a keep again
    /// that often, and the price of it is the next pilot's.
    #[test]
    fn the_next_pilot_s_chance_of_a_look_without_room_is_its_reads_own() {
        let (groups, members, units, outputs, pilots) = (11_usize, 8_usize, 12_usize, 3_usize, 40);
        let scale: Vec<f64> = (0..units).map(|u| 0.17 / 2. * (units - u) as f64 / (units * (units + 1) / 2) as f64).collect();
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let estimates = vec![estimate([10.; 3], 0., 88.); outputs];
        let (step, read) = (Bound::share(), vec![false, true, true]);
        // A pilot's reads: per state (output after t = 0, group, member) every unit's error on T.
        let draw = |seed: u64| -> (Vec<Booking>, Vec<usize>, Vec<usize>) {
            let mut stream = crate::random::Stream::jump(seed, 7, 0);
            let (mut errors, mut at, mut from) = (vec![], vec![], vec![]);
            for k in 1..outputs {
                for g in 0..groups {
                    for _ in 0..members {
                        errors.extend(scale.iter().map(|a| Booking { error: [0., a * stream.open_uniform().ln().powi(2)], ..Booking::NONE }));
                        at.push(k);
                        from.push(g);
                    }
                }
            }
            (errors, at, from)
        };
        let part = || {
            let mut sums = Group::new(outputs);
            let held = Fate::Held(Held { rows: [0.; STATES], kappa: [0.; 3], kappa2: [1.; 3], outside: false, unresolved: false, leaving: false });
            (0..members).for_each(|_| sums.add(&vec![held.clone(); outputs]));
            Part { reads: sums.clone(), sums, listed: vec![], samples: vec![vec![]; outputs], lost: vec![], counts: Counts::default() }
        };
        let parts: Vec<Part> = (0..groups).map(|_| part()).collect();
        let parts: Vec<&Part> = parts.iter().collect();
        let short = |set: &[usize], next: &[Vec<[f64; 2]>]| -> bool {
            let model = (0..outputs).map(|k| if k == 0 { [0.; 2] } else { [0., (0..units).filter(|u| !set.contains(u)).map(|u| next[k][u][1]).sum()] }).collect();
            !tight(&estimates, &Fixed { model }, step, &read).is_empty()
        };
        let order: Vec<usize> = (0..units).collect();
        let (mut predicted, mut seen) = (vec![0.; units + 1], vec![0.; units + 1]);
        let (mut read_short, mut read_predicted) = (0., 0.);
        for trial in 0..pilots as u64 {
            let (errors, at, from) = draw(2 * trial);
            let (row, lost) = rows(at.len());
            let measured = Measured { errors: &errors, output: &at, group: &from, row: &row, lost: &lost, booked: &all_read(units) };
            let means = measured.means(outputs, None, &estimates);
            let fixed = Fixed { model: model_error(&measured.combined(outputs, None, &estimates), [0.; 2]) };
            let doubt = Doubt::new(&measured, outputs, &parts, &scales, [0.; 2]);
            let (errors, at, from) = draw(2 * trial + 1);
            let (row, lost) = rows(at.len());
            let next = Measured { errors: &errors, output: &at, group: &from, row: &row, lost: &lost, booked: &all_read(units) }.means(outputs, None, &estimates);
            for j in 0..=units {
                let set = &order[..j];
                predicted[j] += doubt.chance(set, &without(&fixed, &means, &read, set), &estimates, step, &read, &[]);
                seen[j] += f64::from(u8::from(short(set, &next)));
            }
            // The set the error read asks: the heaviest units until it has room.
            let at_read = (0..=units).find(|&j| tight(&estimates, &without(&fixed, &means, &read, &order[..j]), step, &read).is_empty()).unwrap_or(units);
            read_short += f64::from(u8::from(short(&order[..at_read], &next)));
            read_predicted += doubt.chance(&order[..at_read], &without(&fixed, &means, &read, &order[..at_read]), &estimates, step, &read, &[]);
        }
        for j in 0..=units {
            let p = predicted[j] / pilots as f64;
            let spread = (pilots as f64 * p * (1. - p)).sqrt();
            assert!((seen[j] - predicted[j]).abs() <= 2. * spread + 1., "{j} units: {} of {pilots} short, {:.1} predicted", seen[j], predicted[j]);
        }
        let p = read_predicted / pilots as f64;
        let spread = (pilots as f64 * p * (1. - p)).sqrt();
        assert!(read_short >= pilots as f64 / 10. && (read_short - read_predicted).abs() <= 2. * spread + 1., "the error read's set short in {read_short} of {pilots}, {read_predicted:.1} predicted");
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
            uncounted: 0.,
            outside: [0.; 2],
            unresolved: [0.; 2],
            temperature_target: TEMPERATURE_TARGET,
        }
    }

    /// The share not counted (estimate.rs, # What is counted) in the accounts and in the result. Four groups of two
    /// trajectories at t = 0 and three outputs, every atom held in state 0 until it ends with no outcome: the first of
    /// each group after the first output, the second after the second. The units' error booked at 1 % on the held
    /// atoms' populations and 2 % on T, the step at its share. At the first output the accounts are the units' and T
    /// has one; at the second what is shown takes the uncounted half whole, 1 % + 50 %, and T has none; at the third,
    /// none counted, what is shown is booked whole, its model's part and its statistics' at one. Those two outputs are
    /// crowded, by the share: the looks a unit's keep reads (`keep`, the estimates without it) crowd none. The result:
    /// every number finite (the app refuses one that is not), the loss and the populations at the third output the
    /// second's, its error at least the whole, and the flags say why -- the trajectories that ended by their reasons
    /// (a step's optical sector that made no K by its own), counts and first times, the output where T has no account,
    /// the outputs none is counted at -- and nothing of an atom lost; the reason a result is partial is the output none
    /// is counted at.
    #[test]
    fn the_share_not_counted_is_booked_whole_and_an_output_none_counts_at_says_so() {
        let groups: Vec<Group> = (0..4)
            .map(|g| {
                let mut group = Group::new(4);
                let held = Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [1. + g as f64; 3], outside: false, unresolved: false, leaving: false });
                group.add(&[held.clone(), held.clone(), Fate::Failed, Fate::Failed]);
                group.add(&[held.clone(), held.clone(), held, Fate::Failed]);
                group
            })
            .collect();
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let estimates = summarize(&groups, &scales);
        let (fixed, step) = (Fixed { model: vec![[0.01, 0.02]; 4] }, Bound::share());
        let (population, temperature) = accounts(&estimates, &fixed, step);
        assert!((population[1].model - 0.01).abs() < 1e-15 && temperature[1].is_some_and(|t| (t.model - 0.02).abs() < 1e-15), "{population:?} {temperature:?}");
        assert!((population[2].model - 0.51).abs() < 1e-15 && temperature[2].is_none());
        assert!((population[3].model, population[3].statistics) == (1., 1.) && temperature[3].is_none());
        assert_eq!(crowded(&looks(&estimates, &fixed, step)), [2, 3]);
        let counted: Vec<Estimate> = estimates.iter().map(|e| Estimate { uncounted: 0., ..e.clone() }).collect();
        assert!(crowded(&looks(&counted, &fixed, step)).is_empty());
        let plan = Plan::new(&[0., 10., 20., 30.], [1.; 3]);
        let listed = vec![vec![]; plan.listed.len()];
        let context = Context {
            at: plan.times.clone(),
            plan,
            tau: 1.,
            scales,
            initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))),
            fixed,
            outside: vec![[0.; 2]; 4],
            step,
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let unmade = [Stopped::default(), Stopped { count: 2, since: Some(14.) }, Stopped::default()];
        let counts = Counts { steps: 0, failed: Stopped { count: 3, since: Some(12.5) }, unmade, runaway: Stopped { count: 5, since: Some(15.) }, optical: [0.0; 4] };
        let s = assemble(&context, &estimates, &estimates, &listed, &counts, 8);
        assert!(s.estimated_errors.iter().chain(&s.loss_probability).chain(s.populations.iter().flatten()).all(|x| x.is_finite()), "{s:?}");
        assert!(s.loss_probability[3] == s.loss_probability[2] && s.populations[3] == s.populations[2] && s.estimated_errors[3] >= 1., "{s:?}");
        let flagged = s.approximations.flagged.expect("flagged");
        for said in ["3 条轨迹的状态在数值上无法再推进（最早在 12.5 μs）：此后它们不计入估计", "2 条轨迹在某一步上，光学扇区的噪声算子 K 做不出（ℳ 奇异", "5 条轨迹的散射率超过了原子任何状态可能的上限（某个绝热消去远离它的适用范围，最早在 15 μs）：它们在那里停下", "20 μs：有轨迹不计入估计，温度的误差无法估计", "30 μs 起：没有一条轨迹计入估计"] {
            assert!(flagged.contains(said), "{said} in {flagged}");
        }
        assert!(!flagged.contains("已无原子留在光阱中"), "{flagged}");
        assert!(s.reason.as_ref().is_some_and(|r| r.starts_with("t = 30 μs") && r.ends_with("：没有一条轨迹计入估计")), "{:?}", s.reason);
    }

    /// The two ways T goes unread, said apart (`assemble`): an output whose atoms sit in one group (two of them) has its
    /// own T with no error -- the jackknife, every group's removal leaving fewer than two atoms, reads none --; an output
    /// of one atom carries the last T, n̄, mean velocity and the judgement of the validated regime that goes with that T
    /// (`classical`), not a fresh "below 4ħω". Five groups of four: every atom held at 10 μs, one group's two at 20 μs,
    /// one atom at 30 μs, none at 40 μs.
    #[test]
    fn a_t_carried_keeps_its_judgement_and_a_t_unread_is_its_own() {
        let held = |g: usize, m: usize| Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [100. + (4 * g + m) as f64; 3], outside: false, unresolved: false, leaving: false });
        let groups: Vec<Group> = (0..5)
            .map(|g| {
                let mut group = Group::new(5);
                for m in 0..4 {
                    group.add(&[held(g, m), held(g, m), if g == 0 && m < 2 { held(g, m) } else { Fate::Lost }, if g == 0 && m == 0 { held(g, m) } else { Fate::Lost }, Fate::Lost]);
                }
                group
            })
            .collect();
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let estimates = summarize(&groups, &scales);
        let times = [0., 10., 20., 30., 40.];
        let plan = Plan::new(&times, [1.; 3]);
        let listed = vec![vec![]; plan.listed.len()];
        let context = Context {
            at: plan.times.clone(),
            plan,
            tau: 1.,
            scales,
            initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))),
            fixed: Fixed { model: vec![[0.01, 0.02]; 5] },
            outside: vec![[0.; 2]; 5],
            step: Bound::share(),
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let s = assemble(&context, &estimates, &estimates, &listed, &Counts::default(), 20);
        let own = estimates[2].motion.as_ref().expect("two atoms: a T").temperature;
        assert!(own != estimates[1].motion.as_ref().unwrap().temperature && s.temperature_uk[2] == own && s.temperature_errors_uk[2] == [None; 3], "{s:?}");
        assert!(estimates[3].motion.is_none() && s.classical[2] == [true; 3], "{:?}", s.classical);
        assert!(s.temperature_uk[3] == s.temperature_uk[2] && s.mean_occupation[3] == s.mean_occupation[2] && s.mean_velocity_mps[3] == s.mean_velocity_mps[2] && s.classical[3] == s.classical[2], "{s:?}");
        let flagged = s.approximations.flagged.expect("flagged");
        for said in ["20 μs：仍在光阱中的原子太少（去掉任一组后不足两个），温度的统计误差未知", "30 μs：仍在光阱中的原子不足两个，温度与 n̄ 沿用此前的值，误差未知"] {
            assert!(flagged.contains(said), "{said} in {flagged}");
        }
        assert!(!flagged.contains("20 μs：仍在光阱中的原子不足两个") && !flagged.contains("30 μs：仍在光阱中的原子太少"), "{flagged}");
    }

    /// Every estimator and every account on every kind of output a run can hold (estimate.rs, # What is counted): a
    /// ratio over a count, a share or a weight that can be none gives a number or an explicit "unread", never one that
    /// is no number, and no cap or extremum turns one that is none into a bound. Two levels of five groups of four
    /// trajectories, each output one kind: every atom held; half lost; all lost; all held and unresolved; all held
    /// outside the reach; half not counted; none counted; one group holding two atoms; two groups holding (T's read on
    /// two groups); atoms that do not move (T zero); one atom held in all. At both levels, with the units' errors
    /// booked at a number, at none (an output no pilot state was read at: [1, ∞]) and at no number:
    /// - each estimate's loss, populations and shares are numbers, the loss and the share in [0, 1]; T's interval a
    ///   number or unbounded; what is shown errs by its statistics at most one;
    /// - each account of what is shown is a number at most √3 (three parts of a probability's at most one), T's a
    ///   number or none; each look's need a number or unbounded, and a batch of it the product's next count;
    /// - the step's reads are numbers or "no number" with an unread spread; what the check books of them and what it
    ///   does not exclude are numbers or unbounded on T, numbers at most one on what is shown;
    /// - the result's errors, losses and populations are numbers, and the outputs of no atom, of none counted and of T
    ///   without an account are said.
    #[test]
    fn every_estimator_and_account_is_a_number_or_says_why_on_every_kind_of_output() {
        let kinds = 11;
        let held = |g: usize, m: usize, scale: f64, still: bool, outside: bool, unresolved: bool| {
            let y = if still { 0. } else { scale * (1. + (g * 4 + m) as f64) };
            Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == (g + m) % 8))), kappa: [if still { 0. } else { 0.1 * m as f64 }; 3], kappa2: [y; 3], outside, unresolved, leaving: false })
        };
        let level = |scale: f64| -> Vec<Group> {
            (0..5)
                .map(|g| {
                    let mut group = Group::new(kinds + 1);
                    for m in 0..4 {
                        let fates: Vec<Fate> = (0..=kinds)
                            .map(|k| match k {
                                0 | 1 => held(g, m, scale, false, false, false),
                                2 => if m % 2 == 0 { Fate::Lost } else { held(g, m, scale, false, false, false) },
                                3 => Fate::Lost,
                                4 => held(g, m, scale, false, false, true),
                                5 => held(g, m, scale, false, true, false),
                                6 => if m % 2 == 0 { Fate::Failed } else { held(g, m, scale, false, false, false) },
                                7 => Fate::Failed,
                                8 => if g == 0 && m < 2 { held(g, m, scale, false, false, false) } else { Fate::Lost },
                                9 => if g < 2 && m < 2 { held(g, m, scale, false, false, false) } else { Fate::Lost },
                                10 => held(g, m, scale, true, false, false),
                                _ => if g == 0 && m == 0 { held(g, m, scale, false, false, false) } else { Fate::Lost },
                            })
                            .collect();
                        group.add(&fates);
                    }
                    group
                })
                .collect()
        };
        let (fine, coarse) = (level(1.), level(1.3));
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let number = |x: f64| x.is_finite();
        let number_or_unbounded = |x: f64| x.is_finite() || x == f64::INFINITY;
        for groups in [&fine, &coarse] {
            let estimates = summarize(groups, &scales);
            for (k, e) in estimates.iter().enumerate() {
                assert!(number(e.loss) && (0. ..=1.).contains(&e.loss) && (0. ..=1.).contains(&e.uncounted) && e.populations.iter().all(|p| number(*p)), "output {k}: {e:?}");
                assert!(e.outside.iter().chain(&e.unresolved).all(|x| number(*x)) && number_or_unbounded(e.temperature_interval()), "output {k}: {e:?}");
                assert!(number(e.shown_statistics()) && e.shown_statistics() <= 1., "output {k}: {e:?}");
            }
            for booked in [[0.01, 0.02], [1., f64::INFINITY], [f64::NAN; 2]] {
                let fixed = Fixed { model: vec![booked; kinds + 1] };
                let step = Bound::share();
                let (population, temperature) = accounts(&estimates, &fixed, step);
                assert!(population.iter().all(|a| number(a.total()) && a.total() <= 3f64.sqrt()), "{booked:?}: {population:?}");
                assert!(temperature.iter().flatten().all(|a| number(a.total())), "{booked:?}: {temperature:?}");
                for l in looks(&estimates, &fixed, step) {
                    assert!(number_or_unbounded(l.need()) && batch(l.need(), 88) >= 96 && batch(l.need(), 88) <= 176, "{booked:?}: {l:?} needs {}", l.need());
                }
            }
            let s = assemble(&Context {
                at: (0..=kinds).map(|k| 10. * k as f64).collect(),
                plan: Plan::new(&(0..=kinds).map(|k| 10. * k as f64).collect::<Vec<_>>(), [1.; 3]),
                tau: 1.,
                scales,
                initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))),
                fixed: Fixed { model: vec![[0.01, 0.02]; kinds + 1] },
                outside: vec![[0.; 2]; kinds + 1],
                step: Bound::share(),
                checked: None,
                ledger: Ledger::default(),
                method: String::new(),
            }, &estimates, &estimates, &vec![vec![]; Plan::new(&(0..=kinds).map(|k| 10. * k as f64).collect::<Vec<_>>(), [1.; 3]).listed.len()], &Counts::default(), 20);
            assert!(s.estimated_errors.iter().chain(&s.loss_probability).chain(s.populations.iter().flatten()).all(|x| number(*x)), "{s:?}");
            let flagged = s.approximations.flagged.expect("flagged");
            assert!(flagged.contains("30 μs") && flagged.contains("70 μs：没有一条轨迹计入估计") && flagged.contains("有轨迹不计入估计，温度的误差无法估计"), "{flagged}");
        }
        let read = between(&fine, &coarse, &scales, &(0..=kinds).map(|k| 10. * k as f64).collect::<Vec<_>>());
        let reads: Vec<&Read> = std::iter::once(&read.loss).chain(&read.shown).chain(read.temperature.iter()).collect();
        assert!(reads.iter().all(|r| (number(r.value) && (number(r.spread.error) || !r.spread.read())) || (r.value.is_nan() && !r.spread.read())), "{read:?}");
        let none = Between { loss: Read { value: f64::NAN, spread: Spread::UNREAD }, ..read.clone() };
        for b in [checked(&read, None, true).0, checked(&none, Some(&read), false).0, unexcluded(&read), unexcluded(&none), estimated(&read), estimated(&none)] {
            assert!(number(b.shown) && b.shown <= 1. && number_or_unbounded(b.temperature), "{b:?}");
        }
        assert_eq!(checked(&none, None, true).0.shown, 1., "a read that is no number excludes nothing: the whole");
        assert!(blind(&none).is_some_and(|f| f == f64::INFINITY));
        assert_eq!(batch(f64::NAN, 88), 176);
        assert_eq!((probability_error(f64::NAN), probability_error(2.), probability_error(0.3)), (1., 1., 0.3));
    }

    /// The reach's falsifier (`outgrown`) on a pilot of 88 atoms, 200 outputs after t = 0 and 21 listed ones (621
    /// looks), a reach made for 30.04 µK that leaves out s = 2.44e-4.
    /// - Every T at the reach's and no atom outside: the reach stands. A T of 75 µK read exactly: refuted, to 75.
    /// - The T's own read fails where its error is estimated high: 75 ± 12 µK on eleven groups is within 6.4 errors of
    ///   the reach. The count then: two of 88 outside happen by chance once in 4400 (the tail (88 s)²/2), not rare
    ///   enough among 621 looks (one-sided 3.7e-5 each); three do not: refuted, to the largest T read.
    /// - The measured case (R3_DOPPLER, seed 3): the hottest look 38.78 ± 1.79 µK, 6.4 errors reach to 27.3; six of
    ///   88 outside: refuted.
    /// - A family of eleven looks (three outputs, two listed; a T of 75 ± 20 µK) is refuted by two of 88.
    /// - Atoms outside with no T above the reach's -- a cloud driven out -- are offered no reach: it stands.
    /// - Six of 88 outside a reach that leaves out a twentieth refute nothing.
    #[test]
    fn a_reach_is_refuted_by_a_t_above_it_or_by_its_atoms_count_outside() {
        let s = unseen(MOST_TRAJECTORIES as f64);
        let run = |outputs: usize, hottest: f64, error: f64| -> Vec<Option<crate::estimate::Motion>> {
            (0..=outputs).map(|k| {
                let mut e = estimate([20., 20., if k == outputs { hottest } else { 30. }], 0., 88.);
                e.motion.as_mut().unwrap().temperature_errors = [Spread { error, groups: if error > 0. { 11. } else { f64::INFINITY } }; 3];
                e.motion
            }).collect()
        };
        let counts = |listed: usize, n: f64| -> Vec<(f64, f64)> { (0..listed).map(|j| (if j + 1 == listed { n } else { 0. }, 88.)).collect() };
        assert_eq!((outgrown(&counts(21, 0.), s, &run(200, 30., 0.), 30.04), outgrown(&counts(21, 0.), s, &run(200, 75., 0.), 30.04)), (None, Some(75.)));
        let look = 0.5 * crate::estimate::tail_among(621.);
        assert!(75. - Spread { error: 12., groups: 11. }.interval_over(621.) < 30.04 && (look / 3.7e-5 - 1.).abs() < 0.02);
        assert!(crate::estimate::at_least(2., 88., s) > look && crate::estimate::at_least(3., 88., s) < look);
        assert_eq!((outgrown(&counts(21, 2.), s, &run(200, 75., 12.), 30.04), outgrown(&counts(21, 3.), s, &run(200, 75., 12.), 30.04)), (None, Some(75.)));
        assert!((38.78 - Spread { error: 1.79, groups: 11. }.interval_over(621.) - 27.3).abs() < 0.1);
        assert_eq!((outgrown(&counts(21, 0.), s, &run(200, 38.78, 1.79), 30.04), outgrown(&counts(21, 6.), s, &run(200, 38.78, 1.79), 30.04)), (None, Some(38.78)));
        assert_eq!((outgrown(&counts(2, 1.), s, &run(3, 75., 20.), 30.04), outgrown(&counts(2, 2.), s, &run(3, 75., 20.), 30.04)), (None, Some(75.)));
        assert_eq!(outgrown(&counts(21, 88.), s, &run(200, 30., 0.), 30.04), None);
        assert_eq!(outgrown(&counts(21, 6.), 0.05, &run(200, 75., 12.), 30.04), None);
    }

    /// The looks and the accounts are of what is shown and of the total T. A pilot that reads 25, 6 and 10 µK on the
    /// axes has the total 13.67 µK; with a model error of 3 % and a step's clear part of 2 % there is one look on what
    /// is shown and one on T at every output after t = 0. An interval of 1 % on the total neither keeps the run from
    /// stopping nor asks for a trajectory; one of 15 % does both, by (interval/room)², whatever the axes' own errors
    /// are. A model error or a step part that fills the target crowds its output. T's account holds the three parts,
    /// and the result's total is it. (The estimates' allowance is the plain 10 %; the step's bound on T is a share of
    /// it, 0.2 of the allowance for 2 %.)
    #[test]
    fn the_looks_and_the_accounts_are_of_the_total_temperature() {
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.03]] };
        let step = Bound { shown: 0., temperature: 0.02 / TEMPERATURE_TARGET };
        let (n, total) = (800., (25. + 6. + 10.) / 3.);
        let quiet = estimate([25., 6., 10.], 0.005 * total, n);
        let run = [quiet.clone(), quiet];
        let all = looks(&run, &fixed, step);
        assert_eq!(all.iter().map(|l| (l.output, l.shown)).collect::<Vec<_>>(), [(1, true), (1, false)]);
        assert!(all[0].account.step == 0. && (all[1].account.step - 0.02).abs() < 1e-17);
        assert!(all.iter().all(|l| l.met() && l.need() < n) && crowded(&all).is_empty());
        let noisy = estimate([25., 6., 10.], 0.075 * total, n);
        let run = [noisy.clone(), noisy];
        let t = looks(&run, &fixed, step)[1];
        let need = n * (CONFIDENCE * 0.075 / room(TEMPERATURE_TARGET, 0.03, 0.02)).powi(2);
        assert!(!t.met() && (t.need() - need).abs() < 1e-9 * need, "{} against {need}", t.need());
        let wide = Bound { shown: 0., temperature: 4. };
        assert!(!looks(&run, &fixed, wide)[1].fits() && crowded(&looks(&run, &fixed, wide)) == [1]);
        let mut full = Fixed { model: fixed.model.clone() };
        full.model[1][1] = 0.11;
        assert!(crowded(&looks(&run, &full, step)) == [1]);
        let account = accounts(&run, &fixed, step).1[1].unwrap();
        assert!(account.model == 0.03 && (account.step - 0.02).abs() < 1e-17 && (account.statistics - CONFIDENCE * 0.075).abs() < 1e-14, "{account:?}");
    }

    /// A T whose statistics no group reads is held as one whose statistics are the normal theory's prediction, in whether
    /// its look is met and in the trajectories it needs alike (`looks`): S2Q at 2.5 ms, 568 trajectories and a held
    /// share of 3·10⁻⁴, met -- it neither keeps the product running nor asks for a trajectory --; at a held share of
    /// one on 100 trajectories, neither met nor within its need. Met exactly where its need is within the trajectories
    /// it has, on either side.
    #[test]
    fn an_unread_t_of_no_weight_neither_blocks_nor_asks() {
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.01]] };
        let unread = |held: f64, n: f64| {
            let mut e = estimate([20.; 3], 1., n);
            e.loss = 1. - held;
            e.motion.as_mut().unwrap().temperature_total_error = Spread::UNREAD;
            looks(&[e.clone(), e], &fixed, Bound::share())[1]
        };
        let (faint, full) = (unread(3e-4, 568.), unread(1., 100.));
        assert!(!faint.shown && faint.met() && faint.need() <= 568., "{faint:?}");
        assert!(!full.met() && full.need() > 100. && full.fits(), "{full:?}");
        for look in [faint, full] {
            assert_eq!(look.met(), look.need() <= look.trajectories, "{look:?}");
        }
    }

    /// T's allowance (module documentation, # T's allowance) and what it drives are smooth in T. With the scale T_sc of a
    /// trap's zero point alone (no photon: ω = 2k_BT_sc/ħ on every axis, T_sc = 10 µK): the absolute allowance
    /// √((0.1 T)² + T_sc²) moves by no more than a tenth of T's own move over 0.05 to 20 T_sc, it is 10 % of T far above
    /// T_sc and T_sc far below; the step's read of a coarser level 5 % hotter at every atom, the trajectories T's look
    /// needs at an interval of 7.5 % beside a model error of 3 % and the step's share, and a unit's weight there all move
    /// by a factor within (1 ± 1 %) between T and T(1 + 10⁻³) -- nothing switches at T_sc. And the set kept changes where
    /// a look's room closes and nowhere else: with the model's error on T at 30 %, T's look is crowded (a keep is asked
    /// for) above the T at which √(1 − s²) times the allowance is 30 %, not at T_sc.
    #[test]
    fn t_s_allowance_and_all_it_drives_are_smooth_in_t() {
        let scale = 10.;
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [2. * KT_PER_UK * scale; 3] };
        assert!((crate::estimate::semiclassical([0.; 2], &scales) - scale).abs() < 1e-12);
        let ts: Vec<f64> = (0..=3000).map(|i| 0.05 * scale * 400f64.powf(i as f64 / 3000.)).collect();
        let absolute = |t: f64| t * allowance(t, scale);
        for w in ts.windows(2) {
            assert!((absolute(w[1]) - absolute(w[0])).abs() <= 0.1 * (w[1] - w[0]) + 1e-12, "{w:?}");
        }
        assert!((allowance(1000. * scale, scale) / TEMPERATURE_TARGET - 1.).abs() < 1e-4 && (absolute(0.01 * scale) / scale - 1.).abs() < 1e-5);
        // A level of four groups of four atoms at T (κ = 0, ⟨κ²⟩ = T: ħ/m is k_B/ħ) and its companion 5 % hotter.
        let level = |t: f64, hotter: f64| -> Vec<Group> {
            (0..4)
                .map(|g| {
                    let mut group = Group::new(2);
                    for m in 0..4 {
                        let kappa2 = t * hotter * (1. + 0.01 * (4 * g + m) as f64);
                        let held = Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [kappa2; 3], outside: false, unresolved: false, leaving: false });
                        group.add(&[held.clone(), held]);
                    }
                    group
                })
                .collect()
        };
        let at = |t: f64| {
            let fine = level(t, 1.);
            let read = between(&fine, &level(t, 1.05), &scales, &[0., 1.]).temperature.expect("T read at both levels").value;
            let mut e = estimate([t; 3], 0.075 * t / CONFIDENCE, 800.);
            e.temperature_target = allowance(t, scale);
            let fixed = Fixed { model: vec![[0.; 2], [0., 0.03]] };
            let need = looks(&[e.clone(), e.clone()], &fixed, Bound::share())[1].need();
            let weight = weights(&[vec![[0.; 2]], vec![[0.01, 0.03]]], &[e.clone(), e.clone()], &[1., 1.], &[1])[0];
            let crowded = !crowded(&looks(&[e.clone(), e], &Fixed { model: vec![[0.; 2], [0., 0.3]] }, Bound::share())).is_empty();
            (read, need, weight, crowded)
        };
        let mut last = at(ts[0]);
        let mut turned = vec![];
        for (i, &t) in ts.iter().enumerate().skip(1) {
            let now = at(t);
            for (name, a, b) in [("the step's read", last.0, now.0), ("the need", last.1, now.1), ("the weight", last.2, now.2)] {
                assert!((b / a - 1.).abs() < 0.01, "{name} at {t} µK: {a} then {b}");
            }
            if now.3 != last.3 {
                turned.push(ts[i - 1]);
            }
            last = now;
        }
        // Where √(1 − s²) δT/T = 0.3: (0.3² / (1 − s²) − 0.1²) T² = T_sc².
        let closes = scale / (0.09 / (1. - STEP_SHARE * STEP_SHARE) - 0.01).sqrt();
        assert!(turned.len() == 1 && (turned[0] / closes - 1.).abs() < 3e-3, "the set kept turns at {turned:?}, the room closes at {closes} µK");
    }

    /// The units' errors per output (`Measured::means`) and their sums (`sums`, `model_error`). Two units (a state's
    /// row: their errors); two groups, output 1 read in both, output 2 in the first alone, output 3 in neither. A unit's
    /// error is the mean over the states of the output, of the first group's with the second left out, and its bound
    /// (1, none) where no state is read. The sums add what the reach leaves out, and are the combined error booked.
    #[test]
    fn the_states_errors_are_the_outputs_means_and_sums() {
        let errors = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08], [0.02, 0.04], [0.06, 0.08]].map(|error| Booking { error, ..Booking::NONE });
        let (row, lost) = rows(3);
        let measured = Measured { errors: &errors, output: &[1, 2, 1], group: &[0, 0, 1], row: &row, lost: &lost, booked: &all_read(2) };
        let estimates: Vec<Estimate> = (0..4).map(|_| estimate([10.; 3], 0., 8.)).collect();
        let means = measured.means(4, None, &estimates);
        assert_eq!(means[1], [[(0.01 + 0.02) / 2., (0.02 + 0.04) / 2.], [(0.03 + 0.06) / 2., (0.04 + 0.08) / 2.]]);
        assert_eq!(means[2], [[0.05, 0.06], [0.07, 0.08]]);
        assert_eq!(means[3], [[1., f64::INFINITY], [1., f64::INFINITY]]);
        assert_eq!(measured.means(4, Some(1), &estimates)[1], [[0.01, 0.02], [0.03, 0.04]]);
        assert_eq!(measured.means(4, Some(0), &estimates)[2], [[1., f64::INFINITY], [1., f64::INFINITY]]);
        let unsampled = [1e-3, 2e-3];
        let whole = sums(&means, unsampled);
        assert_eq!(whole[1], [1e-3 + means[1][0][0] + means[1][1][0], 2e-3 + means[1][0][1] + means[1][1][1]]);
        let all = model_error(&measured.combined(4, None, &estimates), unsampled);
        assert!((0..2).all(|i| (all[1][i] - whole[1][i]).abs() <= 1e-15), "{all:?} against {whole:?}");
        assert_eq!((all[0], all[3]), ([0.; 2], [1., f64::INFINITY]));
    }

    /// A flow's moved population booked against the output's populations (`destined`, `Booking::at`; model/units.rs,
    /// # A rate's error on the populations). A source side holding 0.9 of the atom in one ground state, the flow's far
    /// side two ground states (a hyperfine level's) holding 0.04 and 0.06; a flow of rate error x = 0.2 read at states of
    /// the source side, each booking x times its cluster's ground population 1: over the ensemble x(1 − s) = 0.18 for the
    /// far side's share s = 0.1, where the first order is x s(1 − s) = 0.018. The run books x min(s, 1 − s) = 0.02, s/(1 − s)
    /// of the states' mean -- within 1/(1 − s) of the first order --; on a far side holding half or more, all of it; the
    /// rest of a unit's error as read; a state that names no side as read. A unit's error without bound or that is no
    /// number is the state's whole.
    #[test]
    fn a_flow_is_booked_by_the_lesser_of_its_two_sides() {
        let mut p = [0.; STATES];
        (p[0], p[1], p[2]) = (0.9, 0.04, 0.06);
        let (x, s, far) = (0.2, 0.1, 1 << 1 | 1 << 2);
        let state = Booking { error: [0.03 + x, 0.05 + x], moved: x, into: far };
        let booked = (1. - s) * x * destined(far, &p);
        assert!((booked - x * s).abs() < 1e-15 && booked >= x * s * (1. - s) && booked <= x * s + 1e-15, "{booked}");
        let at = state.at(&p);
        assert!((at[0] - (0.03 + x * s / (1. - s))).abs() < 1e-15 && (at[1] - (0.05 + x * s / (1. - s))).abs() < 1e-15, "{at:?}");
        (p[0], p[1], p[2]) = (0.4, 0.3, 0.3);
        assert_eq!(state.at(&p), state.error);
        assert_eq!(Booking { into: 0, ..state }.at(&p), state.error);
        // A unit's error without bound or that is no number at a state is the state's whole, none of it a flow's.
        for bad in [f64::INFINITY, f64::NAN] {
            for e in [UnitError { population: bad, ..UnitError::NONE }, UnitError { temperature: bad, ..UnitError::NONE }, UnitError { moved: bad, into: far, ..UnitError::NONE }] {
                assert_eq!(Booking::of(&e), Booking { error: [1.; 2], moved: 0., into: 0 }, "{e:?}");
            }
        }
    }

    /// The trajectories a level's looks need without some units (`Level::need_without_all`) and with every unit kept
    /// (`Level::need_alone`) are the largest need of all the looks at that model's error, to the bit: the outputs not
    /// asked could not have raised it. Seven outputs whose total T is read to 1.5 to 3.6 %, three units whose errors
    /// on T differ from output to output, so that the neediest output changes with the unit removed.
    #[test]
    fn a_level_s_need_without_units_is_the_largest_of_all_its_looks() {
        let outputs = 8;
        let estimates: Vec<Estimate> = (0..outputs).map(|k| estimate([20., 10., 15.], 15. * (0.015 + 0.0035 * ((k * 5) % 7) as f64), 88.)).collect();
        let means: Vec<Vec<[f64; 2]>> = (0..outputs).map(|k| (0..3).map(|u| [0.001 * u as f64, 0.004 * (((k + 2 * u) * 3) % 5) as f64 + 0.03 * ((k + u) % 3 / 2) as f64]).collect()).collect();
        let unsampled = [2.5e-4, 1e-3];
        let level = Level::new(None, estimates.as_slice().into(), means.as_slice().into(), Bound::share(), unsampled, &vec![[0.; 2]; outputs]);
        let direct = |model: &dyn Fn(usize) -> [f64; 2]| -> f64 {
            let fixed = Fixed { model: (0..outputs).map(|k| if k == 0 { [0.; 2] } else { model(k) }).collect() };
            looks(&estimates, &fixed, Bound::share()).iter().map(Look::need).fold(0., f64::max)
        };
        // An output's error less unit u's (every error here is read: a sum leaves it by subtraction).
        let less = |k: usize, u: usize| -> [f64; 2] { model_error_at([level.whole[k][0] - means[k][u][0], level.whole[k][1] - means[k][u][1]]) };
        assert_eq!(level.need, direct(&|k| model_error_at(level.whole[k])));
        let needs: Vec<f64> = (0..3).map(|u| level.need_without_all(&[u])).collect();
        for u in 0..3 {
            assert_eq!(needs[u], direct(&|k| less(k, u)), "unit {u}");
            assert!(needs[u] <= level.need, "unit {u}: {} of {}", needs[u], level.need);
        }
        assert!(needs.iter().any(|&n| n < level.need) && (needs[0] != needs[1] || needs[1] != needs[2]), "{needs:?} of {}", level.need);
        assert_eq!(level.need_alone(), direct(&|_| model_error_at(unsampled)));
        assert!(level.need_alone() < needs.iter().copied().fold(f64::INFINITY, f64::min));
        // A level read less a doubt (the scout's, `doubt`) needs what its looks need at every model's error so lowered,
        // to the bit: half of the error on T left where the doubt is half of it; and where the doubt is the whole error,
        // nothing -- no unit's removal then saves a trajectory.
        let half: Vec<[f64; 2]> = level.whole.iter().map(|w| [0., 0.5 * w[1]]).collect();
        let doubted = Level::new(None, estimates.as_slice().into(), means.as_slice().into(), Bound::share(), unsampled, &half);
        assert_eq!(doubted.need, direct(&|k| [model_error_at(level.whole[k])[0], 0.5 * level.whole[k][1]]));
        for u in 0..3 {
            assert_eq!(doubted.need_without_all(&[u]), direct(&|k| { let m = less(k, u); [m[0], (m[1] - 0.5 * level.whole[k][1]).max(0.)] }), "unit {u} at half the error's doubt");
        }
        let whole: Vec<[f64; 2]> = level.whole.clone();
        let doubted = Level::new(None, estimates.as_slice().into(), means.as_slice().into(), Bound::share(), unsampled, &whole);
        assert!((0..3).all(|u| doubted.need_without_all(&[u]) == doubted.need) && doubted.need_alone() == doubted.need, "{}", doubted.need);
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
                    // Three jumps a unit of time.
                    jumps: 3 * k as u32,
                };
                let samples = (0..=outputs).map(|k| (0..members).filter(|&m| k <= held(g, m)).map(|m| state(m, k)).collect()).collect();
                Part { sums: Group::new(outputs + 1), reads: Group::new(outputs + 1), listed: vec![], samples, lost: vec![], counts: Counts::default() }
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
    ///   weighs by the time it holds; one that ends before the first output with no outcome gives none, one lost there
    ///   gives its state at t = 0 and weighs by the time to its loss;
    /// - the mean over the run (`over_the_run`) is the mean over the held states: of a value that is the output's
    ///   place, with every trajectory of a group held as far as its group's place among the groups allows, the picks'
    ///   weighted mean meets the held states' own mean to within half a part of the longest-held trajectory, and
    ///   without a group it is the mean of the others;
    /// - a pick carries its trajectory's jumps per unit time over what it holds: three, of trajectories that jump
    ///   three times a unit of time, wherever they are lost.
    #[test]
    fn the_run_is_read_at_one_state_of_every_trajectory_spread_over_it() {
        for (groups, members) in [(11, GROUP as usize), (11, 1)] {
            let n = groups * members;
            for outputs in [1000usize, 400, 100, 88, 8, 3] {
                let parts = kept(groups, members, outputs, &|_, _| outputs);
                let all: Vec<&Part> = parts.iter().collect();
                let picks = one_each(&all);
                assert_eq!(picks.len(), n);
                assert!(picks.iter().all(|p| p.weight == outputs as f64 && p.state.tau == p.group as f64 && p.jumps == 3.));
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
            assert_eq!((p.weight, p.state.sample.at.t as usize, p.jumps), (held as f64, 1 + (through * held as f64) as usize, 3.));
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

    /// The plan: the windows half a period -- whatever the outputs' interval, longer than it on a slow axis -- cut at
    /// t = 0, the listed outputs evenly spread with the first and the last, the run's end the last window's.
    #[test]
    fn the_plan_holds_the_windows_and_the_listed_outputs() {
        let times: Vec<f64> = (0..101).map(|k| k as f64).collect();
        let w = std::f64::consts::TAU / 0.5;
        let p = Plan::new(&times, [w, 0.1, w]);
        assert_eq!(p.halves[0], [0.; 3]);
        assert_eq!(p.halves[7], [0.25, 7., 0.25]);
        assert_eq!(p.halves[40], [0.25, 0.5 * std::f64::consts::TAU / 0.1, 0.25]);
        assert_eq!((p.listed.len(), p.listed[0], p.listed[20], p.listed[1]), (21, 0, 100, 5));
        assert_eq!(p.end, 100. + 0.5 * std::f64::consts::TAU / 0.1);
        let p = Plan::new(&[0., 2.], [w; 3]);
        assert_eq!((p.listed.clone(), p.end), (vec![0, 1], 2.25));
    }

    /// T's windows by the running integrals at their ends (`Windows`) against the closed forms of a path
    /// κ_a = A_a cos(ω_a t + φ_a): outputs 1 µs apart, periods of 0.7, 2.3 and 23 µs on the three axes (windows shorter
    /// and longer than the interval, overlapping, cut at t = 0), the path in pieces of a step of 0.37 µs and a jump inside
    /// every third step. Run through to the plan's end, every window's time, ∫κ and ∫κ² within 1e-12 of its own scale;
    /// an atom lost at 17.6 µs, the windows that reach past it over their held part.
    #[test]
    fn the_windows_are_their_ends_differences() {
        let w = [TAU / 0.7, TAU / 2.3, TAU / 23.];
        let (amp, phase) = ([1.3, 0.4, 2.1], [0.3, -1.1, 2.0]);
        let times: Vec<f64> = (0..31).map(|k| k as f64).collect();
        let plan = Plan::new(&times, w);
        let antiderivative = |a: usize, t: f64| -> [f64; 3] {
            let x = w[a] * t + phase[a];
            [t, amp[a] * x.sin() / w[a], amp[a] * amp[a] * (0.5 * t + (2. * x).sin() / (4. * w[a]))]
        };
        let moments = |from: f64, to: f64| -> (V3, V3) {
            let d = |a: usize| { let (p, q) = (antiderivative(a, from), antiderivative(a, to)); [q[1] - p[1], q[2] - p[2]] };
            (std::array::from_fn(|a| d(a)[0]), std::array::from_fn(|a| d(a)[1]))
        };
        // The pieces: steps of τ, the third of every three cut by a jump at 0.43 τ.
        let tau = 0.37;
        let mut cuts = vec![0.];
        for i in 0.. {
            let (start, step_end) = (i as f64 * tau, ((i + 1) as f64 * tau).min(plan.end));
            if i % 3 == 2 && start + 0.43 * tau < step_end {
                cuts.push(start + 0.43 * tau);
            }
            cuts.push(step_end);
            if step_end >= plan.end {
                break;
            }
        }
        for stop in [plan.end, 17.6] {
            let mut windows = Windows::new(&plan.times, &plan.halves);
            for piece in cuts.windows(2).filter(|p| p[0] < stop) {
                windows.add(piece[0], piece[1].min(stop), &moments);
            }
            windows.close();
            for k in 1..times.len() {
                for a in 0..3 {
                    let (lo, hi) = (times[k] - plan.halves[k][a], (times[k] + plan.halves[k][a]).min(stop));
                    let exact: [f64; 3] = if hi > lo { let (p, q) = (antiderivative(a, lo), antiderivative(a, hi)); std::array::from_fn(|i| q[i] - p[i]) } else { [0.; 3] };
                    let got = windows.window(k)[a];
                    for i in 0..3 {
                        assert!((got[i] - exact[i]).abs() <= 1e-12 * stop * amp[a].max(1.).powi(2), "stop {stop}, output {k}, axis {a}, moment {i}: {} against {}", got[i], exact[i]);
                    }
                }
            }
        }
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
        // Every atom in one bin is a share of exactly one, however many atoms (S2Q's checkpoint: 1.0000000000000002 from
        // n added shares of 1/n).
        for n in 1..2000 {
            let one = histograms(&[vec![([0.; 3], [0.; 3]); n]], &[0], &|x| x.1[0]);
            assert!(one.probabilities[0].iter().all(|&p| p <= 1.) && one.probabilities[0].contains(&1.), "{n}");
        }
    }

    /// A uniform density reads uniform in every bin, the end ones too: N atoms one to each of N equal cells of [−1, 1]
    /// (at the cells' centres, so that none sits on a bin's edge) have the density N/2, and every bin -- the end ones too
    /// -- holds its width's worth, N·w/2, to within one atom, its share over the grid's spacing w (the density as the app
    /// reads it, every cell one spacing wide: src/motion-chart.ts `density`) is ½ to within 1/(N w). The bins tile the
    /// atoms' reach. (Bins centred on ±reach held about half the end bin's worth.)
    #[test]
    fn a_uniform_density_reads_uniform_to_the_ends() {
        for n in [9_usize, 100, 1001, 4096] {
            let atoms: Vec<(V3, V3)> = (0..n).map(|j| ([0.; 3], [-1. + (2 * j + 1) as f64 / n as f64, 0., 0.])).collect();
            let reach = 1. - 1. / n as f64;
            let d = histograms(&[atoms], &[0], &|x| x.1[0]);
            let (bins, spacing) = (d.grid.len(), d.grid[1] - d.grid[0]);
            assert!(d.grid.windows(2).all(|g| (g[1] - g[0] - spacing).abs() < 1e-12) && d.grid[bins / 2] == 0., "{n}: an even grid about 0");
            assert!((spacing * bins as f64 - 2. * reach).abs() < 1e-12, "{n}: the bins tile [−reach, reach]");
            let worth = n as f64 * spacing / 2.;
            for (i, p) in d.probabilities[0].iter().enumerate() {
                assert!((p * n as f64 - worth).abs() <= 1. + 1e-9, "{n} atoms, bin {i} of {bins}: {} against {worth}", p * n as f64);
                assert!((p / spacing - 0.5).abs() <= 1. / (n as f64 * spacing) + 1e-12, "{n} atoms, bin {i}: density {}", p / spacing);
            }
        }
    }

    /// What the messages pass on (`Messages::result`): a result of two groups at least -- of one member each as well as
    /// of eight -- and of no fewer trajectories than the attempt's last; any other is not made, and the last one goes out
    /// again (nothing where there is none). With no group in there are no estimates to make one from (`so_far`).
    #[test]
    fn a_result_goes_out_from_two_groups_on() {
        let context = Context {
            at: vec![0., Plan::new(&[0., 1.], [1.; 3]).end],
            plan: Plan::new(&[0., 1.], [1.; 3]),
            tau: 0.5,
            scales: Scales { hbar_over_mass: 1., frequencies: [1.; 3] },
            initial: [0.; STATES],
            fixed: Fixed { model: vec![[0.; 2]; 2] },
            outside: vec![[0.; 2]; 2],
            step: Bound::share(),
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let shape = Shape::of(&context.plan, false);
        let record = |i: u64| {
            let held = Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], outside: false, unresolved: false, leaving: false };
            Record { outputs: vec![Fate::Held(held.clone()), Fate::Held(held.clone())], reads: vec![Fate::Held(held.clone()), Fate::Held(held)], listed: vec![Some(([0.; 3], [i as f64; 3])); shape.listed], samples: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2] }
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
        let shape = Shape { outputs: 1, reads: 1, listed: 0, keep: false };
        let over = AtomicU64::new(0);
        let record = |_: u64| {
            std::thread::sleep(Duration::from_millis(3));
            over.fetch_add(1, Ordering::Relaxed);
            Record { outputs: vec![Fate::Lost], reads: vec![Fate::Lost], listed: vec![], samples: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2] }
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

    /// The model's prior (`closed`; module documentation, # The model's prior) is a fixed point of the gates where the
    /// atoms are: on R4_LIN_2 (lin⊥lin cooling, its folds failing at the atoms' speeds) and PGC3D (six beams' cooling
    /// and a repump in a tweezer), closed from nothing kept, every unit the model folds has its gate below ½ at each of
    /// the reach's atoms (`atoms`: at the request's temperature the pilot's initial ones), each at its place and speed
    /// (`Model::failing` finds none) -- each gate a value there, a number, whatever its neighbourhood does --, and the
    /// units kept are among those that failed at one of them as their models were read. Printed: the units kept and the
    /// time the prior took.
    #[test]
    fn the_prior_s_model_folds_only_what_exists_at_every_atom_s_start() {
        let scenes = [("scenes", "R4_LIN_2"), ("requests", "PGC3D_1ms")];
        for (dir, name) in scenes {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join(dir).join(format!("{name}.json"));
            let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
            let light = Light::new(&request);
            let initial = components(&request);
            let started = Instant::now();
            let mut kept = vec![];
            let model = closed(&request, &light, &mut kept, &initial, request.temperature_uk).unwrap();
            let took = started.elapsed().as_secs_f64();
            // The pilot's initial atoms again, as `closed` reads them.
            let points = atoms(&model, &request, &light, &initial, request.temperature_uk).unwrap();
            let fastest = points.iter().fold(0., |most: f64, p| most.max(p.1));
            let units = model.units();
            assert!(model.failing(&units, &points, fastest).is_empty(), "{name}: the closed model folds a unit that fails at an atom's start");
            // Every unit the model folds is read at every atom's start as a number below ½ (the whole space's bound
            // stands for the point where it is below ½ already).
            let reads = model.judged(&units);
            for &(r, v) in &points {
                let region = model.point_region(r, v);
                for (u, read) in reads.iter().enumerate() {
                    let gate = model.gate_of(read, &region);
                    assert!(gate.saturation.is_finite() && gate.sweep.is_finite() && gate.exists(), "{name}: {:?} at {r:?}, speeds to {v}: {gate:?}", units[u]);
                }
            }
            eprintln!("{name}: {} units kept by the prior, {} folded, {} atoms' starts, {took:.2} s", kept.len(), units.len(), points.len());
        }
    }

    /// The work formula prices a model as its run counts it (`Formula`). R4_LIN_2's model with its cooling pair's drives into
    /// F′=3 at the atom, over its 30 µK reach,
    /// with its tables: the current model priced from its plans on its tables' axes (`Model::priced`) is its run's own
    /// count, cluster by cluster at a node, and the middle with its scattering and a jump's photons at the focus, to the
    /// bit. A kept model -- a bare drive taken to the atom, a cluster's fold to its fourth order, a carrier's dressing
    /// made coherent -- priced of its structure alone is its run's with its own tables made, at states of every cluster
    /// at rest and at the reach's speed along each lab axis, the coordinates new to a kept model read over the trap's
    /// centre and halfway to the reach's box along each axis: the bare drive's and the dressing's within 1 % (measured:
    /// to the bit, and 0.3 % below), the fourth order's above its run's and within 15 % (11.5 %: its matrices counted
    /// once at the step's middle, which the run's middle at the focus finds made by its scattering, and their changes
    /// along every lab axis the beams change along); and the three in their runs' order.
    #[test]
    fn the_work_formula_prices_a_kept_model_as_its_run_counts_it() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R4_LIN_2.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let initial = components(&request);
        let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).unwrap();
        let reference = alone(initial[chief].0);
        // The cooling pair's drives into the D2 F′=3 states at the atom (a cluster's fold at the atom, its fourth order and
        // its carrier's dressing are units; the others stay bare).
        let bare = Model::new(&request, &light, &[]).unwrap();
        let cooling = (0..light.frequencies.len()).filter(|&f| f != light.trap).max_by_key(|&f| light.frequencies[f].beams.len()).unwrap();
        let kept: Vec<Unit> = bare.bare_units(crate::model::Component { frequency: cooling, counter: false }).into_iter().filter(|u| matches!(u, Unit::Bare { excited, .. } if bare.labels[*excited].starts_with("D2-3"))).collect();
        let model = Model::new(&request, &light, &kept).unwrap();
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, 30., unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let (tau, end) = (0.4335, 800.);
        let most = (pilot_size() as f64 * (end / tau + end / (2. * tau))).ceil() as usize;
        let reach = &reach;
        let tables = model.tabulate(reach, most);
        let ladder = Ladder { run: end, prior: f64::INFINITY, frequency: trap.frequencies.iter().copied().fold(0., f64::max), kick: 0., fastest: reach.speed, typical: reach.speed };
        let mut formula = Formula::new(&model, &kept, &units, reference, trap.centre, &tables, ladder, Setup { per_draw: 1., evaluations: 0. });
        // Where the atoms were: the trap's centre and halfway to the reach's box's faces along each lab axis.
        formula.points = std::iter::once(trap.centre)
            .chain((0..3).flat_map(|a| reach.cell[a].map(|edge| std::array::from_fn(|b| if b == a { 0.5 * (trap.centre[b] + edge) } else { trap.centre[b] }))))
            .collect();
        // The current model: its counts priced are its run's.
        let run = Run::tabulated(&model, &tables, reference);
        let own = formula.own();
        for c in 0..model.clusters().len() {
            assert_eq!(own.costs[c], crate::step::Middle::work(&run.middle(c, trap.centre, 0.)), "cluster {c}'s counts at a node");
        }
        assert_eq!((own.assembly, own.photons), (run.assembly_work(), run.photon_work()), "a middle with its scattering, and a jump's photons");
        // States of every cluster, at rest and at the reach's speed along each lab axis.
        let speed = reach.speed / model.hbar_over_mass();
        let states: Vec<(usize, crate::model::Point, f64)> = (0..model.clusters().len())
            .filter(|&c| model.clusters()[c].ground > 0)
            .flat_map(|c| (0..4).map(move |a| (c, crate::model::Point { r: trap.centre, kappa: std::array::from_fn(|b| if a == b + 1 { speed } else { 0. }), t: 0. }, tau)))
            .collect();
        let prices = formula.prices(&states);
        let mean = |rates: &[f64]| rates.iter().sum::<f64>() / rates.len() as f64;
        // A kept set's mean rate over the states by the formula, and by its run with its own tables; a jump a µs.
        let rates = |set: &[usize]| -> [f64; 2] {
            let (with, priced, _, potential) = formula.kept(set).unwrap();
            let priced: Vec<f64> = states.iter().zip(&prices).map(|(&(c, _, tau), price)| formula.rate(&with, &priced, c, price, tau, 1., potential)).collect();
            let with = Model::new(&request, &light, &[&kept[..], &set.iter().map(|&u| units[u].clone()).collect::<Vec<_>>()].concat()).unwrap();
            let tables = with.tabulate(reach, most);
            let run = Run::tabulated(&with, &tables, reference);
            let made: Vec<f64> = states
                .iter()
                .map(|&(c, at, tau)| {
                    let middle = run.middle(run.cluster_of(model.clusters()[c].states[0]), trap.centre, 0.);
                    let price = Price::of(&middle, at.kappa, at.kappa.map(|k| run.hbar_over_mass() * k), tau);
                    (price.work(crate::step::Middle::work(&middle)) + run.assembly_work()) / tau + run.photon_work()
                })
                .collect();
            [mean(&priced), mean(&made)]
        };
        let first = |kind: &dyn Fn(&Unit) -> bool| units.iter().position(kind).expect("a unit of the kind");
        let bare = rates(&[first(&|u| matches!(u, Unit::Bare { component, .. } if !component.counter))]);
        let fourth = rates(&[first(&|u| matches!(u, Unit::Second { .. }))]);
        let dressing = rates(&[first(&|u| matches!(u, Unit::Dressing { .. }))]);
        eprintln!("formula against run: bare {bare:?}, fourth order {fourth:?}, dressing {dressing:?}");
        for (name, [priced, made]) in [("bare drive", bare), ("dressing", dressing)] {
            assert!((priced / made - 1.).abs() < 0.01, "{name}: {priced} against its run's {made}");
        }
        assert!(fourth[0] >= fourth[1] && fourth[0] < 1.15 * fourth[1], "fourth order: {} against its run's {}", fourth[0], fourth[1]);
        let order = |i: usize| { let mut o = [0, 1, 2]; o.sort_by(|&a, &b| [bare, fourth, dressing][a][i].total_cmp(&[bare, fourth, dressing][b][i])); o };
        assert_eq!(order(0), order(1), "the keeps in their runs' order");
    }

    /// The groups come back in their order whatever the threads do: 37 groups' records (five of a pilot's numbers, then a
    /// product's) -- each trajectory's own, its rows marking its number -- folded by `run_groups` equal the same records
    /// folded one after another; so do the levels of one hand-out, each with its own members of the groups.
    #[test]
    fn the_groups_come_back_in_order() {
        let shape = Shape { outputs: 2, reads: 2, listed: 1, keep: false };
        let record = |i: u64| {
            let mut rows = [0.; STATES];
            rows[(i % STATES as u64) as usize] = 1.;
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], outside: false, unresolved: false, leaving: false };
            let outputs = vec![Fate::Held(held.clone()), if i % 3 > 0 { Fate::Held(held) } else { Fate::Lost }];
            Record { reads: outputs.clone(), outputs, listed: vec![Some(([i as f64; 3], [0.; 3]))], samples: vec![], lost: None, counts: Counts { steps: i, ..Counts::default() }, scattered: [0.; 2] }
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
                assert_eq!((&part.sums, &part.reads), (&one.sums, &one.reads));
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

    /// R2_RECOIL's tweezer and σ⁺ probe, the probe's |2,2⟩ → |3′,3′⟩ moved onto resonance at the trap's centre and its
    /// couplings scaled to |V| = Γ/10 there -- its fold at rest exists --; with the probe's drives folded at the atom (the
    /// units kept), the ground state the probe cycles.
    fn resonant_probe() -> (Request, Light, Vec<Unit>, usize) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R2_RECOIL.json");
        let mut request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let (b, e) = (7, 31);
        let probe = request.beams.iter().position(|beam| beam.id == "probe").unwrap();
        let frequency = light.frequencies.iter().position(|f| f.beams.contains(&probe)).unwrap();
        let bare = Model::new(&request, &light, &[]).unwrap();
        let centre = bare.trap(&alone(b), &bare.prepared()).unwrap().centre;
        let gamma = bare.decay[(e - GROUND, e - GROUND)].re;
        let size = request.beams[probe].coupling[(e, b)].abs() * light.amplitudes(centre)[probe].value.abs();
        let beam = &mut request.beams[probe];
        for m in [&mut beam.coupling, &mut beam.counter_coupling] {
            let scaled = std::mem::replace(m, crate::linalg::Mat::zeros(STATES, STATES));
            m.add_scaled(crate::linalg::C64::real(0.1 * gamma / size), &scaled);
        }
        for _ in 0..3 {
            let light = Light::new(&request);
            let bare = Model::new(&request, &light, &[]).unwrap();
            let at = Model::new(&request, &light, &bare.bare_units(crate::model::Component { frequency, counter: false })).unwrap();
            let h = at.far_hamiltonian(&at.light.amplitudes(centre), &at.running(), 1, &at.coherent_dressing());
            let delta = h[0][(e, e)].re - h[0][(b, b)].re - at.carriers[0].detuning;
            request.hamiltonian[(e, e)] -= crate::linalg::C64::real(delta);
        }
        let light = Light::new(&request);
        let kept = Model::new(&request, &light, &[]).unwrap().bare_units(crate::model::Component { frequency, counter: false });
        (request, light, kept, b)
    }

    /// The runaway's guard, on its class (module documentation, # The reach's atoms; step.rs, # The domain): a fold far
    /// outside its validity scatters without bound -- the released engine's resonant probe folded at the atom, its first-order
    /// Doppler series diverging at atoms the probe pushed out, 10⁵ jumps/µs. R2_RECOIL's resonant weak probe
    /// (`resonant_probe`), its drives folded at the atom; atoms of |2,2⟩ through the trap's centre at 1 to 1000 times the
    /// reach's largest speed along each lab axis, run for 0.8 µs at 0.005 µs with the attempt's domain. The state's folded
    /// excitation at its first step's middle (model/units.rs `Measured::rates`: Σ_e |R[e, g]|² at its momentum, the
    /// tweezer's shift moving the probe's line by linewidths within a micrometre along the beam) tells the class: an atom of excitation
    /// above 2 -- scattering at twice the fastest rate of any state of the atom -- stops at its first step, is counted, and
    /// is held from then on as it stood, unresolved (booked whole), its rows numbers; one below ½ runs to the end
    /// untouched. Both occur. The guard reads nothing but the hazard the step has: no unit is named or kept by it.
    #[test]
    fn a_fold_scattering_beyond_any_state_stops_its_atom_held_unresolved() {
        let (request, light, kept, b) = resonant_probe();
        let model = Model::new(&request, &light, &kept).unwrap();
        let reference = alone(b);
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let entries = model.sampled(&model.running());
        let widest = (0..model.decay.rows).map(|x| model.decay[(x, x)].re).fold(0., f64::max);
        let domain = Domain { model: &model, reach: &reach, hbar_over_mass: model.hbar_over_mass(), sampled: &entries, widest, result: true };
        let plan = Plan::new(&[0., 0.2, 0.4, 0.6, 0.8], trap.lab);
        let run = Run::new(&model, reference);
        let cluster = model.cluster_of(b);
        let psi: Vec<crate::linalg::C64> = model.clusters()[cluster].states.iter().map(|&s| if s == b { crate::linalg::C64::ONE } else { crate::linalg::C64::ZERO }).collect();
        let (mut stopped, mut ran) = (0, 0);
        for (i, factor) in [1., 10., 30., 100., 300., 1000.].into_iter().enumerate() {
            for a in 0..3 {
                let kappa: V3 = std::array::from_fn(|x| if x == a { factor * reach.speed / model.hbar_over_mass() } else { 0. });
                let middle: V3 = std::array::from_fn(|x| trap.centre[x] + 0.5 * TAU_GUARD * model.hbar_over_mass() * kappa[x]);
                let excitation = model.measured(&Sample { cluster, at: crate::model::Point { r: middle, kappa, t: 0. }, psi: psi.clone() }, None).rates;
                let record = run_trajectory(&run, &domain, Start { state: b, r: trap.centre, kappa }, (3 * i + a) as u64, TAU_GUARD, &plan);
                if excitation > 2. {
                    stopped += 1;
                    assert_eq!(record.counts.runaway.count, 1, "an atom of excitation {excitation} at {factor} × the reach's speed runs on");
                    for (k, fate) in record.outputs.iter().enumerate().skip(1) {
                        match fate {
                            Fate::Held(h) => assert!(h.unresolved && h.rows.iter().all(|x| x.is_finite()), "output {k}: {h:?}"),
                            other => panic!("output {k} of a stopped atom not held: {other:?}"),
                        }
                    }
                }
                if excitation < 0.5 {
                    ran += 1;
                    assert_eq!(record.counts.runaway.count, 0, "an atom of excitation {excitation} at {factor} × the reach's speed stopped");
                }
            }
        }
        eprintln!("R2_RECOIL's resonant probe at the atom: {stopped} atoms beyond any state stopped, {ran} within ran");
        assert!(stopped > 0 && ran > 0, "{stopped} stopped, {ran} ran");
    }

    /// The decisions' reads are the outputs' onlookers (module documentation, # The analysis times; adapt.rs, # The
    /// reads): R2_RECOIL's atoms in |2,2⟩ at the trap's centre, at a tenth to the whole of the reach's largest speed
    /// along each lab axis, run through 2 µs read at 3, 41 or 401 outputs, at 800 steps to the run's end (the last
    /// output's window), and at the candidates m·2τ: their fates and states there are the same to the bit whatever the
    /// outputs, and so are their counts; an output at a candidate's time reads the candidate's fate (their windows are
    /// one where the output's is not cut by the run's end).
    #[test]
    fn the_decisions_reads_are_the_same_whatever_the_outputs() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R2_RECOIL.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let model = Model::new(&request, &light, &[]).unwrap();
        let b = 7;
        let reference = alone(b);
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let entries = model.sampled(&model.running());
        let widest = (0..model.decay.rows).map(|x| model.decay[(x, x)].re).fold(0., f64::max);
        let domain = Domain { model: &model, reach: &reach, hbar_over_mass: model.hbar_over_mass(), sampled: &entries, widest, result: true };
        let run = Run::new(&model, reference);
        let end = Plan::new(&[0., 2.], trap.lab).end;
        let tau = end / 800.;
        let plan = |outputs: usize| -> Plan {
            let mut plan = Plan::new(&(0..=outputs).map(|k| 2. * k as f64 / outputs as f64).collect::<Vec<_>>(), trap.lab);
            plan.cell = 2. * tau;
            plan.read_at(grid(plan.end, tau));
            plan
        };
        let plans = [plan(2), plan(40), plan(400)];
        let (mut shared, mut atoms) = (0, 0);
        for factor in [0.1, 0.4, 1.] {
            for a in 0..3 {
                let kappa: V3 = std::array::from_fn(|x| if x == a { factor * reach.speed / model.hbar_over_mass() } else { 0. });
                let records: Vec<Record> = plans.iter().map(|plan| run_trajectory(&run, &domain, Start { state: b, r: trap.centre, kappa }, (a + 3) as u64, tau, plan)).collect();
                for (r, plan) in records.iter().zip(&plans) {
                    assert_eq!((&r.reads, r.counts.steps), (&records[0].reads, records[0].counts.steps), "{} outputs against 2", plan.times.len() - 1);
                    for (k, t) in plan.times.iter().enumerate() {
                        if let Some(j) = plan.reads.iter().position(|x| x == t) {
                            assert_eq!(r.outputs[k], r.reads[j], "output {k} at the candidate {j}");
                            shared += 1;
                        }
                    }
                }
                let kept: Vec<Record> = plans.iter().map(|plan| super::run(&run, &domain, Start { state: b, r: trap.centre, kappa }, (a + 3) as u64, tau, plan, true)).collect();
                let states = |r: &Record| r.samples.iter().map(|s| s.as_ref().map(|s| (s.sample.clone(), s.tau.to_bits(), s.index, s.jumps))).collect::<Vec<_>>();
                assert!(kept.iter().all(|r| states(r) == states(&kept[0]) && r.lost.as_ref().map(|l| l.0) == kept[0].lost.as_ref().map(|l| l.0)));
                atoms += 1;
            }
        }
        eprintln!("{atoms} atoms through three output grids; {shared} outputs at a candidate's time");
        assert!(shared > 0);
    }

    /// The analysis times' candidates and their reads (module documentation, # The analysis times; `grid`, `analysed`):
    /// the companion's step ends m·2τ to the run's end. Readings the same in every group (no noise) that do not change
    /// split nothing: the run's two ends stand, whatever the window. Where the run's rows change for 5 µs at 313 µs, the
    /// reads a quarter of T's shortest window apart at most find it -- a 40 µs window, reads 7.8 µs apart: times about
    /// it --, and with a window longer than the run, the middle and the quarters alone, it is unseen.
    #[test]
    fn the_reads_are_no_farther_apart_than_a_quarter_of_t_s_shortest_window() {
        let candidates = grid(1000., 0.5);
        assert!(candidates.len() == 1001 && candidates[1000] == 1000. && candidates.windows(2).all(|w| (w[1] - w[0] - 1.).abs() < 1e-12));
        let held = Fate::Held(Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [0.; 3], kappa2: [1.; 3], outside: false, unresolved: false, leaving: false });
        let groups: Vec<Group> = (0..11)
            .map(|_| {
                let mut g = Group::new(candidates.len());
                (0..GROUP).for_each(|_| g.add(&vec![held.clone(); candidates.len()]));
                g
            })
            .collect();
        let refs: Vec<&Group> = groups.iter().collect();
        let scales = Scales { hbar_over_mass: 1., frequencies: [1.; 3] };
        assert_eq!((analysed(&candidates, 40., &refs, &refs, &scales), analysed(&candidates, 2000., &refs, &[], &scales)), (vec![0, 1000], vec![0, 1000]));
        let other = Fate::Held(Held { rows: std::array::from_fn(|s| if s == 1 { 1. } else { 0. }), ..match &held { Fate::Held(h) => h.clone(), _ => unreachable!() } });
        let changed: Vec<Group> = (0..11)
            .map(|_| {
                let mut g = Group::new(candidates.len());
                (0..GROUP).for_each(|_| g.add(&(0..candidates.len()).map(|k| if (311..=315).contains(&k) { other.clone() } else { held.clone() }).collect::<Vec<_>>()));
                g
            })
            .collect();
        let refs: Vec<&Group> = changed.iter().collect();
        let seen = analysed(&candidates, 40., &refs, &[], &scales);
        assert!(seen.iter().any(|&k| (305..=321).contains(&k)) && seen.len() > 2, "{seen:?}");
        assert_eq!(analysed(&candidates, 2000., &refs, &[], &scales), [0, 1000]);
    }

    /// The model's error at the outputs is the line through its values at the analysis times (`interpolated`): held at
    /// the first one after t = 0 before it and at the last after it, t = 0's own at t = 0, unbounded on a side
    /// unbounded; with the run's two ends alone, the end's value at every output after t = 0.
    #[test]
    fn the_outputs_read_the_line_through_the_analysis_times() {
        let (at, values) = ([0., 1., 3.], [[0., 0.], [0.1, 1.], [0.3, f64::INFINITY]]);
        let got = interpolated(&at, &values, &[0., 0.5, 1., 2., 3., 4.]);
        assert_eq!(got[..3], [[0., 0.], [0.1, 1.], [0.1, 1.]]);
        assert!((got[3][0] - 0.2).abs() < 1e-15 && got[3][1] == f64::INFINITY && got[4] == [0.3, f64::INFINITY] && got[5] == [0.3, f64::INFINITY], "{got:?}");
        assert_eq!(interpolated(&[0., 2.], &[[0.; 2], [0.05, 0.1]], &[0., 0.7, 2.]), [[0.; 2], [0.05, 0.1], [0.05, 0.1]]);
        // The accounts likewise, part by part: the ledger's at an output are the stop's at the analysis times about it;
        // a T with no account on a side has none between, and its own at an analysis time.
        let account = |x: f64| Account { model: x, step: 0.02, statistics: 2. * x };
        let (population, temperature) = accounts_along(&at, &[Account::default(), account(0.01), account(0.03)], &[Some(account(0.)), Some(account(0.01)), None], &[0., 0.5, 2., 3.]);
        assert_eq!((population[1], population[3]), (account(0.01), account(0.03)));
        assert!((population[2].model - 0.02).abs() < 1e-15 && (population[2].statistics - 0.04).abs() < 1e-15 && population[2].step == 0.02, "{population:?}");
        assert_eq!(temperature, [Some(account(0.)), Some(account(0.01)), None, None]);
    }

    /// The guard's class test's step (µs).
    const TAU_GUARD: f64 = 0.005;

    /// One trajectory through the plan (`run`) with its states not kept.
    fn run_trajectory(run: &Run, domain: &Domain, start: Start, index: u64, tau: f64, plan: &Plan) -> Record {
        super::run(run, domain, start, index, tau, plan, false)
    }

    /// The model's error booked is the folded units' combined error measured at the states (`Measured::combined`,
    /// `model_error`), on its class: random pilots' reads -- 11 groups, a state an output and group, 40 units of random
    /// errors on the populations and on T, a random part of them a flow's moved population with a random far side, some
    /// outputs read by no group, each group left out in turn -- book at every output the mean over the states of each
    /// state's own error, every unit's there booked at the output's populations together; it is the units' means added
    /// (the whole a keep's pricing takes them out of, `sums`) to the rounding, and where no state reads an output, its
    /// bound (1, none).
    #[test]
    fn the_model_s_error_booked_is_the_units_combined_error_at_the_states() {
        let mut stream = crate::random::Stream::jump(11, 3, 0);
        let (groups, units, outputs) = (11_usize, 40_usize, 6_usize);
        for trial in 0..20 {
            let mut errors = vec![];
            let (mut output, mut group) = (vec![], vec![]);
            for g in 0..groups {
                for k in 1..outputs {
                    if (k + trial) % 5 == 0 || stream.uniform() < 0.1 {
                        continue;
                    }
                    for _ in 0..units {
                        let error = [0.01 * stream.uniform(), 0.02 * stream.uniform()];
                        let moved = if stream.uniform() < 0.5 { error[0].min(error[1]) * stream.uniform() } else { 0. };
                        errors.push(Booking { error, moved, into: if moved > 0. { 1 << (stream.uniform() * 8.) as u32 } else { 0 } });
                    }
                    output.push(k);
                    group.push(g);
                }
            }
            let (row, lost) = rows(output.len());
            let measured = Measured { errors: &errors, output: &output, group: &group, row: &row, lost: &lost, booked: &all_read(units) };
            let estimates: Vec<Estimate> = (0..outputs)
                .map(|_| {
                    let mut e = estimate([10.; 3], 0., 88.);
                    let mut p = [0.; STATES];
                    p[..8].iter_mut().for_each(|x| *x = stream.uniform());
                    let total: f64 = p.iter().sum();
                    e.populations = p.map(|x| x / total);
                    e
                })
                .collect();
            let unsampled = [1e-4 * stream.uniform(), 1e-4 * stream.uniform()];
            for left in std::iter::once(None).chain((0..groups).map(Some)) {
                let booked = model_error(&measured.combined(outputs, left, &estimates), unsampled);
                let whole = sums(&measured.means(outputs, left, &estimates), unsampled);
                let read = |k: usize| output.iter().zip(&group).any(|(&o, &g)| o == k && Some(g) != left);
                assert_eq!(booked[0], [0.; 2], "t = 0 books nothing");
                for k in 1..outputs {
                    if read(k) {
                        assert!((0..2).all(|i| (booked[k][i] - whole[k][i]).abs() <= 1e-14 * whole[k][i]), "trial {trial} output {k}: {:?} against the means' sum {:?}", booked[k], whole[k]);
                    } else {
                        assert_eq!(booked[k], [1., f64::INFINITY], "trial {trial} output {k} read by no state");
                    }
                }
            }
        }
    }

    /// The model is judged where the atoms went (module documentation, # The model; `Measured::means`), on its class:
    /// random pilots' reads -- 11 groups, five outputs after t = 0 of every kind of loss (none, some, every atom lost),
    /// per group a held atom's state where the group holds one and a lost atom's path where it lost one, three units of
    /// random errors, each group left out in turn -- book on what is shown (1 − L) times the held reads' mean and L times
    /// the lost reads', on T the held reads' alone (none where no atom is held), and the combined error is the units'
    /// means added. Every atom lost (S2Q's first model, every atom out within 10 µs): what is shown books the lost
    /// atoms' paths, the output is read, and a unit whose fate error leaves it no room weighs there -- where the held
    /// reads alone booked nothing at all.
    #[test]
    fn every_atom_s_fate_is_read_held_or_lost() {
        let mut stream = crate::random::Stream::jump(13, 5, 0);
        let (groups, units, outputs) = (11_usize, 3_usize, 6_usize);
        let lost_share = [0., 0., 0.3, 0.7, 1., 1.];
        let (mut errors, mut at, mut from, mut row, mut lost) = (vec![], vec![], vec![], vec![], vec![]);
        for k in 1..outputs {
            for g in 0..groups {
                for (kind, present) in [(false, lost_share[k] < 1.), (true, lost_share[k] > 0.)] {
                    if !present {
                        continue;
                    }
                    row.push(errors.len() / units);
                    errors.extend((0..units).map(|_| Booking { error: [0.2 * stream.uniform(), 0.1 * stream.uniform()], ..Booking::NONE }));
                    (at.push(k), from.push(g), lost.push(kind));
                }
            }
        }
        let measured = Measured { errors: &errors, output: &at, group: &from, row: &row, lost: &lost, booked: &all_read(units) };
        let estimates: Vec<Estimate> = (0..outputs).map(|k| Estimate { loss: lost_share[k], ..estimate([10.; 3], 0., 88.) }).collect();
        for left in std::iter::once(None).chain((0..groups).map(Some)) {
            let means = measured.means(outputs, left, &estimates);
            let combined = measured.combined(outputs, left, &estimates);
            for k in 1..outputs {
                let reads = |kind: bool| (0..at.len()).filter(|&i| at[i] == k && lost[i] == kind && Some(from[i]) != left).collect::<Vec<_>>();
                let (held, gone) = (reads(false), reads(true));
                let mean = |of: &[usize], u: usize, i: usize| if of.is_empty() { 0. } else { of.iter().map(|&r| errors[row[r] * units + u].error[i]).sum::<f64>() / of.len() as f64 };
                let l = lost_share[k];
                for u in 0..units {
                    let shown = (1. - l) * mean(&held, u, 0) + l * mean(&gone, u, 0);
                    assert!((means[k][u][0] - shown).abs() <= 1e-15 && (means[k][u][1] - mean(&held, u, 1)).abs() <= 1e-15, "output {k} unit {u} left {left:?}: {:?}", means[k][u]);
                }
                let summed: [f64; 2] = std::array::from_fn(|i| means[k].iter().map(|m| m[i]).sum());
                assert!((combined[k][0] - summed[0]).abs() <= 1e-14 && (combined[k][1] - summed[1]).abs() <= 1e-14, "output {k}: {:?} against {summed:?}", combined[k]);
            }
        }
        // Every atom lost by output 4: what is shown is the lost paths' mean, read; a unit of fate error 0.4 there
        // leaves the look no room, and it alone weighs.
        let (errors, at, from, row, lost) = (vec![Booking { error: [0.4, 0.], ..Booking::NONE }, Booking::NONE], vec![4; groups], (0..groups).collect::<Vec<_>>(), vec![0; groups], vec![true; groups]);
        let measured = Measured { errors: &errors, output: &at, group: &from, row: &row, lost: &lost, booked: &all_read(2) };
        let means = measured.means(outputs, None, &estimates);
        assert!((means[4][0][0] - 0.4).abs() < 1e-15 && means[4][0][1] == 0. && means[4][1] == [0., 0.], "{:?}", means[4]);
        let fixed = Fixed { model: model_error(&measured.combined(outputs, None, &estimates), [0.; 2]) };
        let read: Vec<bool> = (0..outputs).map(|k| k == 4).collect();
        let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
        let step = Bound::share();
        assert_eq!(tight(&estimates, &fixed, step, &read), [4]);
        assert_eq!(weighing(&estimates, &fixed, step, &means, &held, &read), [0]);
    }

    /// A run is priced where its atoms were (`one_each`): a trajectory weighs by the time it holds its atom, and one lost
    /// before the first output gives its state at t = 0 for its own time. Eleven groups of eight, every atom lost at
    /// 0.1 to 0.9 of the first output's time (S2Q's first model): every trajectory gives its state at t = 0, weighs by
    /// its time to the loss, and its jumps a µs are its last state's; a mean over the run of a value reads that value,
    /// where the held states alone read nothing (every keep priced at nothing: S2Q's first keep was a coherence of photons
    /// 736 THz apart, which no step resolves).
    #[test]
    fn a_run_whose_atoms_are_all_lost_is_priced_where_they_were() {
        let (groups, members, outputs) = (11_usize, GROUP as usize, 4_usize);
        let parts: Vec<Part> = (0..groups)
            .map(|g| {
                let state = |m: usize, t: f64, jumps: u32| State { sample: Sample { cluster: 0, at: crate::model::Point { r: [0.; 3], kappa: [0.; 3], t }, psi: vec![] }, tau: 0.5, index: MOST_TRAJECTORIES + (g * members + m) as u64, jumps };
                let loss = |m: usize| 0.1 + 0.8 * ((g * members + m) % 9) as f64 / 8.;
                let mut samples = vec![vec![]; outputs + 1];
                samples[0] = (0..members).map(|m| state(m, 0., 0)).collect();
                let lost = (0..members).map(|m| (1, state(m, loss(m), 6))).collect();
                Part { sums: Group::new(outputs + 1), reads: Group::new(outputs + 1), listed: vec![], samples, lost, counts: Counts::default() }
            })
            .collect();
        let all: Vec<&Part> = parts.iter().collect();
        let picks = one_each(&all);
        assert_eq!(picks.len(), groups * members);
        for p in &picks {
            let lost = &parts[p.group].lost.iter().find(|(_, s)| s.index == p.state.index).unwrap().1;
            assert_eq!((p.state.sample.at.t, p.weight, p.jumps), (0., lost.sample.at.t, 6. / lost.sample.at.t));
        }
        assert!((over_the_run(&picks, picks.iter().map(|_| 2.5), None) - 2.5).abs() < 1e-15);
        // An atom that ends with no outcome before the first output (no state, no loss) gives none.
        let mut silent = parts[0].clone();
        silent.lost.clear();
        assert!(one_each(&[&silent]).is_empty());
    }

    /// A state carried into a model that keeps more (`Model::carried`, module documentation, # An attempt's own) is the
    /// same atom there, on its class: R2_RECOIL's resonant weak probe, its drives folded at the atom, and the model that
    /// keeps the probe's drive |2,2⟩ → |3′,3⟩ explicit. States of |2,2⟩'s cluster -- random amplitudes on its ground
    /// states, at points about the trap's centre, at times 0 to 40 µs (the frames' phases) -- keep their ground
    /// populations and coherences in the other's frames (each ground state's population to the rounding, the phases'
    /// differences as the beats say), and the excited state made explicit holds the population the probe's fold gave
    /// it: the readout's folded population there, but for the part of the channels the keep leaves folded (the trap
    /// light's, its own class in the readout: ~10⁻⁴ of it).
    #[test]
    fn a_state_carried_into_a_model_that_keeps_more_is_the_same_atom() {
        let (request, light, kept, b) = resonant_probe();
        let e = 31;
        let folded = Model::new(&request, &light, &kept).unwrap();
        let probe = request.beams.iter().position(|beam| beam.id == "probe").unwrap();
        let frequency = light.frequencies.iter().position(|f| f.beams.contains(&probe)).unwrap();
        let explicit = Model::new(&request, &light, &[kept.clone(), vec![Unit::Drive { component: crate::model::Component { frequency, counter: false }, excited: e, ground: b }]].concat()).unwrap();
        let centre = folded.trap(&alone(b), &folded.prepared()).unwrap().centre;
        let (from, to) = (folded.cluster_of(b), explicit.cluster_of(b));
        assert!(explicit.clusters()[to].states.contains(&e) && !folded.clusters()[from].states.contains(&e), "the keep makes the excited state explicit");
        let mut stream = crate::random::Stream::jump(17, 2, 0);
        for trial in 0..12 {
            let n = folded.clusters()[from].states.len();
            let psi: Vec<crate::linalg::C64> = (0..n).map(|_| crate::linalg::C64::new(stream.uniform() - 0.5, stream.uniform() - 0.5)).collect();
            let at = crate::model::Point { r: std::array::from_fn(|a| centre[a] + 0.3 * (stream.uniform() - 0.5)), kappa: [0.; 3], t: 40. * trial as f64 / 11. };
            let sample = Sample { cluster: from, at, psi };
            let carried = folded.carried(&sample, &explicit);
            assert_eq!(carried.cluster, to);
            let norm: f64 = sample.psi.iter().map(|z| z.norm_sqr()).sum();
            let place = |c: &crate::model::Cluster, s: usize| c.states.iter().position(|&x| x == s).unwrap();
            for (i, &s) in folded.clusters()[from].states.iter().enumerate() {
                let j = place(&explicit.clusters()[to], s);
                assert!((carried.psi[j].norm_sqr() - sample.psi[i].norm_sqr()).abs() <= 1e-15 * norm, "trial {trial}: state {s}");
            }
            let readout = folded.readout(from, &at, &sample.psi);
            let excited = carried.psi[place(&explicit.clusters()[to], e)].norm_sqr() / norm;
            assert!(excited <= readout[e] * (1. + 1e-12) && excited >= readout[e] * (1. - 1e-3), "trial {trial}: {excited} against the readout's {}", readout[e]);
        }
    }

    /// A set short of room at the error as read is never kept where a larger one has room (`least_expected`, `needed`):
    /// its model decides first on the same states, and their reads ask the rest at once. On keeps that run nothing again
    /// (`again` = 0, the carried states), every set's next keep certain where it is short and the work a µs rising with
    /// the set, 200 random orders: the set kept is the shortest that has room, or one larger, and never one short of it;
    /// where no set has room, the least expected one as before.
    #[test]
    fn a_set_short_of_room_at_the_error_read_is_never_kept_where_a_larger_has_room() {
        let mut stream = crate::random::Stream::jump(19, 4, 0);
        for trial in 0..200 {
            let units = 2 + (stream.uniform() * 10.) as usize;
            let taken: Vec<usize> = (0..units).collect();
            let room = (stream.uniform() * (units + 1) as f64) as usize;
            let short = |set: &[usize]| set.len() < room;
            let rates: Vec<f64> = (0..units).map(|_| 0.01 + stream.uniform()).collect();
            let work = |set: &[usize]| Some([1. + set.iter().map(|&u| rates[u]).sum::<f64>(), 0.]);
            let need = 100. + 400. * stream.uniform();
            let price = |set: &[usize]| [need, if short(set) { 1. } else { 0.1 * stream_free(trial, set.len()) }];
            let chosen = least_expected(&taken, &|u| rates[u], &Costs { least: 1., floor: 0., again: 0., none: None }, &work, &price, &short);
            if room <= units {
                assert!(!chosen.is_empty() && !short(&chosen), "trial {trial}: {chosen:?} short of room at {room}");
            }
        }
    }

    /// A chance of a short read that depends on the trial and the set alone (`a_set_short_of_room_...`).
    fn stream_free(trial: usize, len: usize) -> f64 {
        crate::random::Stream::jump(23, trial as u64, len as u32).uniform()
    }

    /// The room a keep reads is what the step leaves the model as far as its reads hold it (`standing`, module
    /// documentation, # The model: the step's part): at most the step's share on every look, the read's estimate where
    /// it is below. PGC3D at 1000 outputs: a model error of 0.0480 on what is shown has no room beside the step's share
    /// (its threshold 0.0447) and has room beside its read step error of 0.0022 (threshold 0.0499); a read beyond the
    /// share, not clearly (the rung to be left), books the share: the rung the check stands at is within it.
    #[test]
    fn the_room_a_keep_reads_is_what_the_step_as_read_leaves() {
        let b = step_bound()[0].powi(2);
        let below = Between { loss: read(0.0022f64.powi(2), 1e-7), shown: std::array::from_fn(|_| read(0.0022f64.powi(2), 1e-7)), temperature: Some(read(0.01, 1e-4)) };
        let above = Between { loss: read(1.5 * b, 0.5 * b), ..below.clone() };
        let (low, high, share) = (standing(&below), standing(&above), Bound::share());
        assert!((low.shown - 0.0022).abs() < 1e-12 && (low.temperature - 0.1).abs() < 1e-12, "{low:?}");
        assert_eq!(high, Bound { shown: share.shown, temperature: 0.1 });
        assert!((room(POPULATION_TARGET, 0., share.shown) - 0.04472).abs() < 1e-5 && (room(POPULATION_TARGET, 0., low.shown) - 0.04995).abs() < 1e-5);
        let estimates = vec![estimate([10.; 3], 0., 88.), estimate([10.; 3], 0., 88.)];
        let fixed = Fixed { model: vec![[0.; 2], [0.0480, 0.]] };
        assert_eq!((tight(&estimates, &fixed, share, &[false, true]), tight(&estimates, &fixed, low, &[false, true])), (vec![1], vec![]));
    }

    /// t = 0 is read as every output is (module documentation, # Outputs and windows): the result's T per axis, the
    /// total, their errors, n̄, the mean velocity and the validated regime at an output are the same function of that
    /// output's sums at t = 0 as after it -- no other estimator there. Random runs (3 to 12 groups of 1 to 8 atoms,
    /// every atom held in the request's state, random κ and window moments ⟨κ²⟩ ≥ ⟨κ⟩²) whose last output's fates are
    /// t = 0's: the two outputs show the same numbers, bit for bit, and what is shown at t = 0 is the request's state.
    /// T's account at t = 0 is its statistics alone (`accounts`: no step taken, no model acted), weighed and read as
    /// at every output.
    #[test]
    fn t_0_is_read_as_every_output_is() {
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut uniform = move || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let state = |s: usize| -> [f64; STATES] { std::array::from_fn(|r| f64::from(u8::from(r == s))) };
        for _ in 0..200 {
            let (groups, members) = (3 + (uniform() * 10.) as usize, 1 + (uniform() * 8.) as usize);
            let mut fate = || {
                let kappa: V3 = std::array::from_fn(|_| 3. * (uniform() - 0.5));
                let kappa2: V3 = std::array::from_fn(|a| kappa[a] * kappa[a] * (1. + uniform()));
                Fate::Held(Held { rows: state(0), kappa, kappa2, outside: false, unresolved: false, leaving: false })
            };
            let parts: Vec<Group> = (0..groups)
                .map(|_| {
                    let mut group = Group::new(3);
                    for _ in 0..members {
                        let (first, middle) = (fate(), fate());
                        group.add(&[first.clone(), middle, first]);
                    }
                    group
                })
                .collect();
            let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: std::array::from_fn(|_| 0.5 + uniform()) };
            let estimates = summarize(&parts, &scales);
            let plan = Plan::new(&[0., 10., 20.], scales.frequencies);
            let listed = vec![vec![]; plan.listed.len()];
            let context = Context {
                at: plan.times.clone(),
                plan,
                tau: 1.,
                scales,
                initial: state(0),
                fixed: Fixed { model: vec![[0.; 2], [0.01, 0.02], [0.; 2]] },
                outside: vec![[0.; 2]; 3],
                step: Bound::share(),
                checked: None,
                ledger: Ledger::default(),
                method: String::new(),
            };
            let s = assemble(&context, &estimates, &estimates, &listed, &Counts::default(), (groups * members) as u64);
            assert!(s.temperature_uk[0] == s.temperature_uk[2] && s.temperature_errors_uk[0] == s.temperature_errors_uk[2], "{s:?}");
            assert!(s.temperature_total_uk[0] == s.temperature_total_uk[2] && s.temperature_total_errors_uk[0] == s.temperature_total_errors_uk[2], "{s:?}");
            assert!(s.mean_velocity_mps[0] == s.mean_velocity_mps[2] && s.mean_occupation[0] == s.mean_occupation[2] && s.classical[0] == s.classical[2], "{s:?}");
            assert!(s.populations[0] == state(0) && s.populations[2] == state(0) && s.loss_probability[0] == 0., "{s:?}");
            let e = &estimates[0];
            let interval = e.temperature_interval();
            let account = accounts(&estimates, &context.fixed, context.step).1[0];
            if interval.is_finite() {
                let a = account.expect("T read at t = 0 has an account");
                assert!(a.model == 0. && a.step == 0. && a.statistics == interval, "{a:?} against {interval}");
            } else {
                assert!(account.is_none(), "{account:?}");
            }
        }
    }

    /// A keep is priced at the step its model would run at, and refused where no rung within `MOST_STEPS` runs it
    /// (`Formula::kept`, `Pricing::rated_by`), one rule for every unit: here the photon coherences of R9_HEAT's model
    /// (its probe's, its repump's and the tweezer's photons, co- and counter-rotating: beats |Δω| from 7.6·10³ to
    /// 4.8·10⁹ rad/µs), one unit of each beat, on the ladders of runs of 0.1 to 1000 µs. A kept coherence brings its beat
    /// into the generator: its model has a rung exactly where the cap allows the companion's resolution of it, run·|Δω|/(2π)
    /// ≤ MOST_STEPS/2 pairs, and the rung then lies below π/|Δω| -- the tweezer's 2.0·10⁸ rad/µs at 0.1 µs, 3·10⁶ steps; an
    /// optical beat of 4.4·10⁹ at no run. On the formula's 10 µs run every coherence |Δω| ≥ 10⁸ apart is refused (never
    /// kept), and its read (the one gate at the trap's centre at the reach's speed, photon.rs `Model::coherence_gate`:
    /// (size/(|Δω| − spread − Doppler))², the fold's own error) is below 1/(|Δω| τ), the share of its interference a step
    /// of the current model's first rung τ would keep; one at a beat the cap resolves is priced at its model's own first
    /// rung, below the current one and below π/|Δω|, where its rate is the higher.
    #[test]
    fn a_keep_whose_beats_no_rung_resolves_is_refused_and_one_that_a_rung_resolves_is_priced_at_it() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R9_HEAT.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let initial = components(&request);
        let reference = alone(initial[0].0);
        let model = Model::new(&request, &light, &[]).unwrap();
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let tables = model.tabulate(&reach, 10_000);
        let hm = model.hbar_over_mass();
        let kick = model.kick_curvature(&reach, &model.moduli(&reach), &trap);
        let typical = typical_speed(hm, reach.kinetic_uk.iter().sum::<f64>() / 3., reach.speed);
        let ladder = |run: f64| Ladder { run, prior: f64::INFINITY, frequency: trap.frequencies.iter().copied().fold(0., f64::max), kick, fastest: reach.speed, typical };
        let own = ladder(10.).first(&model.sampled(&model.running())).expect("the bare model runs");
        let mut formula = Formula::new(&model, &[], &units, reference, trap.centre, &tables, ladder(10.), Setup { per_draw: 1., evaluations: 0. });
        formula.points = vec![trap.centre];
        let region = model.point_region(trap.centre, reach.speed);
        let mut seen: Vec<f64> = vec![];
        let (mut optical, mut resolvable) = (0, 0);
        for (i, unit) in units.iter().enumerate() {
            let Unit::Coherence { first, second, .. } = unit else { continue };
            let beat = (model.omega(first) - model.omega(second)).abs();
            if seen.iter().any(|b| (b - beat).abs() <= 1e-9 * beat) {
                continue;
            }
            seen.push(beat);
            let with = model.rekept(std::slice::from_ref(unit)).unwrap();
            let entries = with.sampled(&with.running());
            for run in [0.1, 1., 10., 100., 1000.] {
                match ladder(run).first(&entries) {
                    None => assert!(run * beat / TAU > (MOST_STEPS / 2) as f64, "{unit:?} (|Δω| {beat:e}) has no rung at {run} µs"),
                    Some(rung) => assert!(run * beat / TAU <= (MOST_STEPS / 2) as f64 && rung < PI / beat, "{unit:?} (|Δω| {beat:e}): rung {rung} at {run} µs"),
                }
            }
            if beat >= 1e8 {
                optical += 1;
                assert!(formula.kept(&[i]).is_none(), "{unit:?} is priced");
                let read = model.gate_of(&crate::model::units::Read::Coherence(unit.clone()), &region).saturation;
                assert!(read <= 1. / (beat * own), "{unit:?}: read {read:e} against 1/(|Δω| τ) {:e}", 1. / (beat * own));
            } else {
                resolvable += 1;
                let (_, priced, rung, potential) = formula.kept(&[i]).expect("a coherence a rung resolves is priced");
                assert!(rung < own && rung < PI / beat, "{unit:?}: its rung {rung} against the current {own} and π/|Δω| {}", PI / beat);
                let run = Run::tabulated(&model, &tables, reference);
                let middle = run.middle(0, trap.centre, 0.);
                let price = Price::of(&middle, [0.; 3], [0.; 3], own);
                assert!(formula.rate(&with, &priced, 0, &price, rung, 0., potential) > formula.rate(&with, &priced, 0, &price, own, 0., potential), "{unit:?}");
            }
        }
        assert!(optical >= 6 && resolvable >= 1, "{optical} optical and {resolvable} resolvable beats: {seen:?}");
    }

    /// R2_RECOIL's model as its prior closes it, and its trap light's drive F=2 m=2 → F′=3 m=3 (a unit: kept, it is folded
    /// with the levels at the atom, model.rs `Unit::Bare`).
    fn trap_drive_at_the_atom() -> (Request, Light, Vec<Unit>, Unit) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R2_RECOIL.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let initial = components(&request);
        let mut kept = vec![];
        closed(&request, &light, &mut kept, &initial, request.temperature_uk).unwrap();
        let (b, e) = (request.states.iter().position(|s| s == "ground-2-2").unwrap(), request.states.iter().position(|s| s == "D2-3-3").unwrap());
        let drive = Unit::Bare { component: crate::model::Component { frequency: light.trap, counter: false }, excited: e, ground: b };
        (request, light, kept, drive)
    }

    /// The trap light's drive folded at the atom moves the well by its levels' shifts over its detuning, and its envelope
    /// follows (eliminate.rs `Model::near_envelope`: line by line, each its own distance less the spread): B above the
    /// bare fold's by less than k_BT -- what the rejection sampling's acceptance pays for, as a power of the excess in
    /// k_BT -- and still above −V at the well's points along the lab axes; the sampling then reads proposals a draw of the
    /// bare model's order (within a factor 3; measured 4.9 against 2.6), and its 2⁹ draws read the exact
    /// moments of the model's own lowered isothermal -- (ħ/m)⟨κ_a²⟩ by its quadrature (`Ensemble::kinetic`) -- per lab
    /// axis within their interval over the groups at the confidence z (three looks) and the quadrature's own difference.
    /// (With the dissipation's 2/Γ_min the envelope was 4.2·10⁹ rad/µs against 392.6, and no atom was drawn in 140 s.)
    #[test]
    fn a_trap_drive_folded_at_the_atom_bounds_its_well_as_tightly_and_draws_its_own_ensemble() {
        let (request, light, kept, drive) = trap_drive_at_the_atom();
        let initial = components(&request);
        let reference = alone(initial[0].0);
        let hm = hbar_over_mass(request.mass_kg);
        let models = [Model::new(&request, &light, &kept).unwrap(), Model::new(&request, &light, &[&kept[..], std::slice::from_ref(&drive)].concat()).unwrap()];
        let mut read: Vec<(f64, f64)> = vec![];
        for (m, model) in models.iter().enumerate() {
            let well = Prepared::new(model, &light, reference).unwrap();
            let bound = well.envelope().1;
            let centre = well.trap().0;
            for d in [0., 0.3, 0.7, 1.5, 3.] {
                for a in 0..3 {
                    let mut r = centre;
                    r[a] += d;
                    assert!(-well.potential(r).0 <= bound, "model {m}: −V {} above B {bound} at {r:?}", -well.potential(r).0);
                }
            }
            let ensemble = Ensemble::new(&[(initial[0].0, 1., &well as &dyn Well)], request.temperature_uk, hm).unwrap();
            let draws: Vec<V3> = (0..1 << 9).map(|i| ensemble.draw(SEED, i).kappa.map(|k| hm * k * k)).collect();
            read.push((bound, ensemble.evaluations().per_draw));
            if m == 1 {
                let kinetic = ensemble.kinetic(1 << 12);
                let exact = ensemble.momentum(&[kinetic[0].eta]);
                let quadrature = kinetic[0].difference / kinetic[0].eta;
                for a in 0..3 {
                    let groups: Vec<f64> = draws.chunks(GROUP as usize).map(|g| g.iter().map(|d| d[a]).sum::<f64>() / GROUP as f64).collect();
                    let g = groups.len() as f64;
                    let mean = groups.iter().sum::<f64>() / g;
                    let error = (groups.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (g - 1.) / g).sqrt();
                    let within = Spread { error, groups: g }.interval_over(3.) + quadrature * exact[a];
                    assert!((mean - exact[a]).abs() <= within, "axis {a}: {mean} against {} ± {within}", exact[a]);
                }
            }
        }
        let [(bare, bare_draw), (atom, atom_draw)] = [read[0], read[1]];
        eprintln!("envelope {bare} -> {atom}; proposals a draw {bare_draw} -> {atom_draw}");
        assert!(atom >= bare && atom - bare <= KT_PER_UK * request.temperature_uk, "B {atom} against the bare fold's {bare}");
        assert!(atom_draw <= 3. * bare_draw, "{atom_draw} proposals a draw against {bare_draw}");
    }

    /// A keep's price holds its model's prepared potential by count (`Formula::potential`, `Formula::rate`,
    /// `Formula::setup`): the trap drive folded at the atom makes each evaluation an elimination, its count many times the
    /// bare model's (3.2·10⁻⁴ s against 2.6·10⁻⁷ s measured), and a step's reference alone then outweighs the step's own
    /// work -- its rate at the step is more than twice what it is without the potential --, and its setup (the prior's
    /// atoms and the reach's quadrature, at their counts) is in its price.
    #[test]
    fn a_keep_that_makes_the_potential_an_elimination_is_priced_by_it() {
        let (request, light, kept, drive) = trap_drive_at_the_atom();
        let initial = components(&request);
        let reference = alone(initial[0].0);
        let model = Model::new(&request, &light, &kept).unwrap();
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, unseen(MOST_TRAJECTORIES as f64));
        let units = model.units();
        let tables = model.tabulate(&reach, 10_000);
        let ladder = Ladder { run: 1000., prior: f64::INFINITY, frequency: trap.frequencies.iter().copied().fold(0., f64::max), kick: 0., fastest: reach.speed, typical: reach.speed };
        let setup = Setup { per_draw: 2., evaluations: pilot_size() as f64 * 2. + 4096. };
        let mut formula = Formula::new(&model, &kept, &units, reference, trap.centre, &tables, ladder, setup);
        formula.points = vec![trap.centre];
        let u = units.iter().position(|x| *x == drive).expect("the trap drive is a unit");
        let (with, priced, rung, potential) = formula.kept(&[u]).expect("the trap drive at the atom is priced");
        let own = formula.potential(&model);
        eprintln!("potential's work an evaluation {own} -> {potential}");
        assert!(potential > 100. * own, "{potential} against the bare fold's {own}");
        let tau = rung.min(0.25);
        let state = reference.iter().position(|&p| p == 1.).unwrap();
        let c = model.cluster_of(state);
        let run = Run::tabulated(&model, &tables, reference);
        let price = Price::of(&run.middle(c, trap.centre, 0.), [0.; 3], [0.; 3], tau);
        let (with_potential, without) = (formula.rate(&with, &priced, c, &price, tau, 0., potential), formula.rate(&with, &priced, c, &price, tau, 0., 0.));
        eprintln!("rate {without} -> {with_potential}");
        assert!(with_potential > 2. * without, "{with_potential} against {without} without the potential");
        assert!((formula.setup(potential) - potential * setup.evaluations / ladder.run).abs() <= 1e-12 * formula.setup(potential));
    }
}
