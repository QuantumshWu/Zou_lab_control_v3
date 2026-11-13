//! The pipeline: request, reach and model, pilot, decisions, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the light; the initial components; the reach and the model with every unit whose existence
//!    gate fails kept (model.rs, SPEC §1.5); the trap the result reports, the outputs' plan, the initial ensemble and
//!    what the reach leaves out of it; T(0) by quadrature; the near light's tables over the reach (SPEC §2.3). What of
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
//!    root), the model's units' errors at the states of the first held member of every group, and the tables' errors
//!    at one state of every trajectory (SPEC §5.4; `one_each`). Its trajectories at the run's step stay in the
//!    result: they are trajectories of the run's own model and step.
//!    The falsifier reads the run's own level alone, so the pilot's own level is run before its companion and read as
//!    it comes in (`own_level`): where its atoms refute the reach the attempt ends there, and the rest of the pilot --
//!    its companion, the step's check at a model the next attempt judges anew over the grown reach -- is never run
//!    (`attempt`); so is a rung below's own level read, where the check goes down (`check`). Where the scout's atoms
//!    already refute the reach, the own level is run before the scout's keep (the units' errors are read at the states
//!    of a reach that stands), its members doubling a hand-out (`batch`), and the reach read after each. The scout's
//!    refutation is the read that selected the attempt for ending; its eleven atoms, one a group, are confirmed by the
//!    next eleven, read with them, before the reach grows to the temperature they read (a reach grown on the scout's own
//!    read, R3_DOPPLER, over-read its atoms' T by up to 2.4 times and cost 16 % more work). What the falsifier needs is
//!    significance, which the doubled atoms give where it stands; what more atoms would buy is the grown temperature
//!    read closer, and a reach read low is refuted again by the next pilot's atoms, read high costs its own proofs --
//!    both the next attempt's to tell. (Measured, S3 at 30 µK: on 22 atoms the reach grows to 99.7 µK, on 44 to 95.7, on
//!    88 to 113.0, none refuted by the next pilot's 88; 957, 949 and 1128 CPU-s in all, the same result.) What the work
//!    of an attempt so ended leaves the next is the reach's temperature and the states of the atoms that refuted it,
//!    for the next attempt's decisions (# An attempt's own).
//! 3. The model's decisions (SPEC §5.2; # The model), at the scout's states and again at the pilot's -- at its first rung
//!    whose own read is in and not clearly beyond, before its check goes on (# Each decision as its evidence is in) --,
//!    the step's error taken at its share: units to keep end the attempt, and the next begins with their model at the
//!    step this one stood at (# The model). An attempt after a growth decides first on the states the refuting atoms
//!    left, before its scout (# An attempt's own).
//! 4. The product: batches of whole groups at the settled step, one level, every trajectory run to the last output,
//!    until every held quantity's total lies within its target (# Stopping) or the trajectories reach
//!    `MOST_TRAJECTORIES`. Its atoms that leave the reach are read where they are, and what they carry is booked
//!    (# The reach's atoms); where that leaves a look no room, the reach grows and the next attempt begins.
//! 5. The result (contract §2.2, `assemble`).
//!
//! # An attempt's own
//! What a reach that grows leaves valid of the attempt before it. The model's own set-up (`Setting`) serves every
//! attempt at the model while no unit is kept. A grown reach changes, for the same model: the gates (proofs over more
//! cells) and with them which units are read and which booked at their bounds, what the reach leaves out of the
//! ensemble, the kick's curvature and the first rung (the reach's largest speed), the tables (refined over the
//! reach's ranges: other nodes, another interpolant -- part of the generator a trajectory is integrated with) and the
//! Brownian path's cell (the first rung's companion). An earlier pilot's trajectories are then of another generator,
//! on another tree of the path, and at a step the new attempt need not stand at (R3_DOPPLER: the first rung over
//! 90 µK is 0.83 µs, below the pilot's 1.43 µs). Where the grown reach keeps units (their folds fail over it), the
//! model changes at every state -- a kept unit is explicit everywhere --, and every read of the pilot is the old
//! model's: its trajectories, the step's reads (the old generator's error), the units' reads (of folds the new model
//! may not hold), the tables (other clusters), its statistics, and the ensemble's draw (the prepared potential holds
//! the folded drives' shifts alone). What stays is what the reads say of the atoms: the reach's verdict and the
//! temperature it grows to, and where the atoms went -- the states of the run's level whose atoms refuted the reach
//! (`Attempt::Grow`), a pilot's of the model as it stands. Where no unit is kept over the grown set, the next attempt
//! reads the model's decision on them first (`keep`, at the units' errors there with its own tables and booking), as
//! the old engine carried its pilot's states into the grown reach's choice: a keep they ask ends the attempt before it
//! runs a trajectory, and none asked stands for the scout's keep on its eleven states. They are read for a decision
//! and booked nowhere: the result's errors are read on its own pilot's states. The step does not stay: a descent the
//! check took on a reach the atoms refuted, or whose domain its paths left, is that reach's -- its domain, its tables,
//! its ladder --, and the next attempt begins at its own first rung (`Attempt::stood`; S3c over 1 ms: its 30 µK
//! reach's companion at 1.43 µs left the domain and the check went down to 0.358 µs, which the next attempt, at 65 µK
//! and a first rung of 0.971 µs, began at, twice the product's steps). Measured (2026-11-12, S1, S2, S2Q, S3, S3c over
//! 1 and 10 ms, e038, e077, PGC3D_1ms, R3_DOPPLER): every growth after a pilot kept units (S3 82 → 84, S3c over 10 ms
//! 69 → 75) or lowered the first rung below the pilot's step (R3_DOPPLER); the others grew on the scout's paths not
//! proven. No pilot's trajectories carry over, so none is made to: a pilot on a refuted reach is not spent instead
//! (# The pipeline: its own level, read before its companion).
//!
//! # Each decision as its evidence is in
//! A decision that ends an attempt is taken where what it reads is in, not after the work it would make useless. The
//! pilot's keep reads the units' errors and the tables' at the states of the pilot's run level, the optical sector's
//! booking, the pilot's statistics, and the step at its share -- not the step's check: nothing of it changes with what
//! the check does past the rung's own two levels, where it stands (the reads are the ones made: they are kept, and the
//! keep is not asked again); where the check goes down a rung instead, its states are the same trajectories at half
//! the step, read again, and the keep asked again on them. So the pilot runs as far as the first rung whose own read is
//! in and not clearly beyond (`Settled::Pending`), the reach's falsifier and the keep are read there, and only an
//! attempt they leave standing pays for the rest of its check -- the third level at 4τ, pairs, a rung below. (Read
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
//! pattern advances in a step at the request's temperature, and the step may take its share of T's allowance -- so the
//! ladder does not begin at an edge's rung the check then leaves (PGC3D: 1.425 µs, left on the pilot for 0.7125 µs,
//! 11.2 CPU-s). An attempt after a keep begins at the step the one before stood at, where that is the shorter; after
//! a growth, at its own first rung (# An attempt's own). The first rung is a rung of the ladder, whose bottom is the
//! step of `MOST_STEPS` steps a trajectory: a request for which no rung down to it is admissible is refused with a
//! message.
//!
//! The step's convergence is checked on the pilot, as a refinement is checked by hand: the pilot's trajectories run at
//! τ and, the same trajectory numbers -- the same initial atoms, the same jumps' draws by their number, the same
//! Brownian path (adapt.rs) --, at 2τ (`Levels`), both read at the same outputs at the same instants. Of two levels at
//! h and 2h a quantity's difference at an output is d(h) = Q(2h) − Q(h); a read (estimate.rs `between`) takes
//! d/(2^p − 1) over the run -- its mean square M over the outputs, estimated without the noise's floor, with the
//! delete-one-group jackknife of that estimate --: one read for the loss, for each row as shown and for the total T,
//! each carried to the outputs by the dynamics itself. An error that changes its sign in time counts by its size.
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
//!   no rung above --, and it certifies nothing alone. An attempt that begins at the step the attempt before stood at
//!   (after a keep) reads that rung anew: the reads before were of another model (# An attempt's own).
//! - A comparison of two levels says something only where the coarser one still resolves what the scheme samples: the
//!   first rung's double is admissible for the typical atom, and stable for every atom. Where a path of the coarser
//!   level is not proven, the coarser one has left the model's domain for that atom: the two levels are apart by more
//!   than any share, and the rung is left as one beyond, unread -- the rung below has no third level above, and no
//!   third level reads the step that left (# The reach's atoms: where the finer level's path is not proven either, the
//!   rung below's decides whether the reach grows).
//!
//! What enters a quantity's account, at every output alike, is what the reads the rung stands by say (`checked`); it
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
//! The outputs are the request's times, from 0 and evenly spaced. At an output a trajectory is read inside the step
//! the output falls in (`run`; step.rs `Step::at`): its rows as shown, its (r, κ) for the listed distributions, the
//! pilot's kept state, and its loss test there. T's window about an output (estimate.rs `half_window`: one period of the
//! axis, cut at t = 0) takes a trajectory's κ and κ² over the cuts of its steps' paths (step.rs `Segment::moments`: the
//! path with the kick spread over the step, of the scheme's order at any cut); the first is the instant t = 0, and the
//! run goes on through the last output's window (`Plan`). Outputs closer than a period have overlapping windows, so a
//! window is not summed by itself: per lab axis the trajectory's running integrals of 1, κ and κ² from t = 0 are kept
//! at every window's two ends, and a window is their difference (`run`) -- each step's path integrated once, and once
//! more up to each end that falls inside it, whatever the windows' overlap.
//!
//! # The model (SPEC §5.2)
//! The units are the model's (model.rs), listed again in every model a keep closes. A unit whose fold does not exist over
//! the reach is kept. Beyond that a keep is one comparison of the units' booked errors against what keeping them costs,
//! over the whole set the reads give (`keep`), made once and never undone: the kept set only grows, and every keep is
//! a new attempt, a new model and its own pilot.
//! - The errors no number of trajectories fits beside: where the model's error and the step's share leave a look no
//!   room for the statistics (`crowded`), the units are taken by their weight there (their errors on what is shown and
//!   on T, weighed by the held share as the totals are, each over its target) for the work a µs of their kept model,
//!   the most first, as far as every look has room at the far end of the error they leave; and of the sets so led
//!   through, the one of the least expected work is kept, none of it to spare (`needed`): its model's work a µs over
//!   its pilot and its product, and the chance that the next pilot still reads a look without room -- the doubt the
//!   reads leave on the error, `Doubt::chance` -- times the work of keeping again. (Taken one at a time, a keep an
//!   attempt, S3's probe kept fourteen bare drives over fourteen attempts, each a model, its gates and a scout; taken
//!   by weight alone, it kept its carrier's dressing with them, three to four times dearer a step; taken until the
//!   error read had room, e038 kept 6, 2, 1 and 1 units over four pilots; taken to the far end of the error, S3c's
//!   pilot kept ten units where one gave room.) The product asks the same where a look the pilot left room has none at the product's
//!   statistics (`product`): it ends with the set. A look the model leaves room whose need a run's
//!   `MOST_TRAJECTORIES` do not meet is not chased (`Look::fits`) -- read again at every batch, as its need is. (Asked
//!   where what the atoms outside the reach carry left a look no room, R3_DOPPLER's product kept two units for T's
//!   looks over its last ten outputs, the error of its atoms heated past the reach -- the kept model's check took half
//!   the step, 52 → 91 CPU-s --: that is the reach's to answer, at the product's end.)
//! - Else the work: the run's predicted total work W = w·N, N the trajectories its looks need beside its errors and the
//!   step's share (`Look::need` on the pilot's estimates) and w the counted work a trajectory spends per µs, at the
//!   pilot's states (`Pricing::rates`): its steps' count through the model's generator (step.rs `Price`), its
//!   middles' assembly and scattering, and its jumps' photons at the trajectory's own rate of jumps, all as the run
//!   counts them with near tables -- the attempt's for the current model, the axes its tables would take for a kept one
//!   (model.rs `Model::priced`, tabulate.rs `Model::priced_axes`) --, and the kept model's W counting the pilot its
//!   attempt runs again. A kept model is priced by its work formula, of its structure alone: its clusters, its drives'
//!   levels and its plans, the counts the run's are -- no table is made and nothing eliminated. (Rated with the tables
//!   its attempt would make, S3's pilot made 38 kept models' tables and 2259 floors, a model and its elimination each,
//!   263 CPU-s a pricing. Priced without tables, by the exact elimination, R4_LIN_2's current model read 138 times its
//!   counted work, and a keep's coordinates -- its tables' axes, an interpolation's work going as four to their number
//!   -- were unseen: S3's carrier's dressing priced as the bare drives, where with its tables it is twice them, and its
//!   run's steps 2.4 times dearer.) A unit is a candidate where its keep alone saves beyond that saving's spread (decide.rs
//!   `pays`; the jackknife over the pilot's groups, `Pricing`), and the set kept is, of the candidates taken by their
//!   saving, the largest first, the one whose keep together saves the most beyond its spread: their errors leave the
//!   looks together, and their kept model is rated once (`Pricing::joint`). A kept model is built and rated only for a
//!   unit whose keep would pay were it to cost no more per µs than the fold (`Pricing::saving_at_all`: the trajectories
//!   its error's removal saves, less the new pilot): a fold is made to cost less than what it folds, so a keep that
//!   pays by its rate alone is not looked for -- where one exists the fold is at fault, not the choice.
//! - On the scout, whose reads are eleven states an output, both are judged at the model's error they leave no doubt
//!   of: every model's error less the jackknife interval of the current one's over the scout's groups, at the family
//!   of the outputs' reads (`doubt`, `clear`); the pilot reads the same units on all its states. A unit's error is
//!   heavy-tailed over the states, and the reads that put the room's error high put the work it asks high with them:
//!   a need is the statistics' interval over the room, and where a point estimate all but fills the target the need
//!   runs to any number, every one of the jackknife's levels with it -- the saving reads near one with no spread.
//!   (Measured, R4_LIN_2 at 76.5 µK, the room judged at the near end and the pricing at the point estimate: the scout
//!   priced one side of a mirror pair of bare drives and then its carrier's dressing at a need of 168 421, its near
//!   end leaving the room whole, and the product ran 1352 trajectories where the model kept on the pilot's reads ran
//!   552.)
//!
//! The price of a keep knows the kept model's work a step at the step the model stands at; a kept model's check may
//! take a shorter one. That is the accuracy the kept model asks, read as every model's is, and it is not traded back:
//! a keep undone where its check took a shorter step (as it was) is a decision taken again on the same reads, and it
//! went back and forth -- on PGC3D, with a batch of bare drives booked too high, some twenty keeps and undos, a model,
//! its gates and a pilot each (bounds3.status.md). A model's rate is its mean over the run's held states, read at one
//! state of every pilot trajectory, the states spread evenly over the run (`one_each`), the same for every model.
//!
//! A unit is read at the states only where its error can matter. A unit whose gate bounds its error over the reach
//! (units.rs `Gate::error_bound`) is booked at that bound at every output, unread, as long as such bounds, the smallest
//! first, add up to no more than the share the most trajectories a solve runs leave unseen (`booked`; estimate.rs
//! `unseen`): what is shown is never booked statistics below that share (`Estimate::shown_statistics`), so what the
//! unread units' bounds overstate together lies below what any run resolves. The others are read, once per state: a
//! state's errors are kept as read and their means per output made once for every decision that asks (`Measured`).
//!
//! # The reach's atoms (SPEC §1.8)
//! The proof set -- the reach and its tubes (`Proven`; model/reach.rs, # Tubes) -- is where the model's statements are
//! made before a trajectory runs: the folds exist (the gates are proofs over each of its parts), the first rung is
//! admissible at its largest speed; the tables are refined over the reach's ranges. Of these, what is wrong without
//! bound where it fails is proven for every step of a held trajectory (step.rs, # The domain; `Domain`): a fold that
//! does not exist, and a step past the edge of its reference's turn and its kick's curvature. A step asks the model
//! everything at points of its straight path and at one momentum, so before its propagation the path is placed --
//! within the reach, where its largest prepared energy is within the reach's level (with its box within the trap's
//! region where the reach holds only a lower level outside it, model/reach.rs, # The trap's region), or within a tube,
//! and nothing more is read; else at its own state, by the statements the set's proofs make over a cell, made along
//! the path at its speed (model/reach.rs `Model::holds_along`: every unit the whole space at that speed leaves open has
//! its fold), and the step stable there. A step that is not proven is not taken: the hand-out ends (`run_groups`), and
//! so does the attempt, with the set grown to hold the paths not proven (`Attempt::Grow`): a path some temperature's
//! reach holds grows the reach to it (model/reach.rs `holding`, `Model::outside_to`); a path none holds -- at the
//! reach's top or above it: leaving the trap, or bound beyond every temperature's quantile -- is held by its own tube,
//! the path's box at its energy, and the reach stands at its temperature. The model is judged again over the set -- a
//! fold that does not exist there is kept, at the level the level rule gives (SPEC §1.5) -- and the next attempt
//! begins. A state that fails is held by the set that ends its attempt, and the set and the units kept only grow. (Held
//! by the reach's level raised within the trap's region to such a path's energy, the reach proved the folds at every
//! place of the region at every speed up to the path's, which no atom of the run is at, and its temperature no longer
//! read its level: S3's probe at −20 MHz proved them 319 CPU-s and kept five units, the first rung fell from 1.43 to
//! 0.78 µs; and the falsifier, reading the leaving atoms' T, grew S2's reach to 331 µK and halved its step.) The
//! reach's temperature is its bound atoms': the falsifier reads the bound atoms alone (`outgrown`, estimate.rs
//! `bound_motion`: a held atom whose prepared energy is at the far value or above at an output is leaving, held by
//! its path's proofs, no draw of the distribution the reach is made for), and so does the product's end
//! (`attempt`); a leaving atom is computed as any other, and what it carries is read where it is (`apart`). The paths a reach holds are the run's own: a level at the run's
//! step or below it. A companion's level (twice the step, or four times it as a third level) is the step's probe, not
//! the atoms': where its path is not proven and the rung's own is, the companion's step has left the model's domain
//! where the rung's did not -- its error, the levels apart by more than any share --, and the rung is left as one
//! beyond (`settle`, # The step); the reach is not grown for it. Where the rung's own paths left the domain too, the
//! two levels say nothing of each other, and the finer step decides: the rung is left, and the reach grows for the
//! rung below's own paths where they leave the domain as well (the next attempt beginning at its own first rung, # An
//! attempt's own), else the rung below is left in turn, its companion
//! the level that departed (`own_departure`). A growth so costs one level of the rung below at the most, and a step's
//! departure grows nothing. (Grown for its paths, the lab's probe scenes S3 grew
//! the reach from 30 µK to 145--666 µK on the scout's companions at 2.5--2.9 µs, which heated their atoms past the
//! trap's depth within 0.03--5 ms where the run's own level held them: the reach's largest speed then set a step 2.3--4
//! times shorter, its proofs took up to three quarters of the solve, and its edge, past the depth, kept the probe's
//! excited states explicit.)
//!
//! What is accuracy is booked where the atom is. The tables' error at a state is read there, as every unit's is (an
//! interpolant of the elimination's own nodes is bounded by them; beyond the ranges the tables were verified over its
//! error is unverified, not unbounded). A step of the run that does not resolve the light along its path at its speed
//! has no read at a state: such a step is taken, and its trajectory is unresolved from it on (step.rs
//! `Trajectory::unresolved`) -- every later state of it is a wrong step's descendant. What it shows is booked whole:
//! its share of the held atoms on what is shown -- an atom's row is a probability, so its error is at most one,
//! whatever its state, and the mean over the held atoms errs by at most the share --, its share of the held atoms'
//! kinetic energy on T -- the part of T it makes, which bounds its error while its true kinetic energy is within twice
//! the one computed (its step's force within 5/4 of the true one's bound, step.rs; a hypothesis on T, not a bound). The
//! loss of an atom lost after such a step is no held atom's: as every model error's, it is in what the held atoms'
//! errors are weighed by (the held share, `accounts`), and not booked apart.
//!
//! Atoms outside the reach whose steps are proven at their own states are computed as any other, and what they carry is
//! read where they are. Every trajectory places its atom at every output (`Domain::placed`), and an output's estimate
//! holds the share of the held atoms unresolved and of those outside the reach, resolved, each with its share of the
//! kinetic energy (estimate.rs `Estimate::unresolved`, `Estimate::outside`); a group keeps the state of its first atom
//! outside (`Shape::fold`). Before each batch's looks the product reads them (`apart`), and the model's error at an
//! output is the pilot's with the unresolved atoms whole and the atoms outside, by their shares, times what the model's
//! error read at their states -- every folded unit's and the tables', at most the whole of what an atom carries --
//! exceeds the error booked. The booked error is already a mean over the held atoms wherever they are (the pilot's
//! states are not chosen by the reach), so only the excess is theirs: a cloud driven out whole (every atom outside) adds
//! nothing twice. The read is the product's own, at as many states an output as the pilot reads the model at. It is in
//! the looks: the run stops on totals that hold it. Where the product ends with a look that it leaves no room for the
//! statistics and that has room without it, the reach is outgrown as far as the product reads: it grows to the bound
//! atoms' temperature read where that is above it (`attempt`), else the item stands booked and the ledger flags the
//! output.
//!
//! Measured (2026-11-11, seeds' runs of the reference scenes, before every step was placed). Six beams' Doppler heating
//! in a tweezer, the reach grown to 84--139 µK, 448--704 atoms, 200 outputs: at most 0.2--1.1 % of the held atoms
//! outside at an output, with 1.9--5.6 % of the kinetic energy (the heated tail is heavier than thermal), the farthest
//! at 1.44 of the reach's level and 1.04 of its largest speed; the model's error at 113 outside states 0.77 % on what is
//! shown and 2.2 % on T against 0.65 % and 0.88 % at the states inside, the tables' 1e-5 at both; over a reach raised to
//! hold the farthest atom none of the 200 gates fails. An atom held at a lattice's antinode: one or two of 176 outside,
//! 1e-5 added. A cloud pushed out of its trap: every atom outside from 40 µs on, 6e-5 added. Six beams' sub-Doppler
//! cooling, gray molasses, a Raman transfer: none outside.
//!
//! # Stopping (SPEC §3)
//! Every quantity held to a target -- at every output after t = 0 what is shown, and the total T -- has the total
//! E² = A² + c² + S² (ledger.rs; `Look`): A the model's error, from the units measured at the pilot's states and the
//! tables' errors there, and c what the step's check leaves open (its rms over the run), both fixed by the pilot -- A with what
//! the run's atoms outside the reach carry, read before each batch's looks (# The reach's atoms) --; S the
//! statistics' interval. The run stops where every one of them lies
//! within its target: the test the ledger flags an output by (`Look::met`). The stop reads the run's standard errors,
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
//! axes' T (`Motion::temperature_total`), its statistics the jackknife of that mean. It is held at every output after
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
//! model's construction and its gates, the tables and their refinement, the units' reads at the states, the pricing
//! of keeps, T(0)'s quadrature.
use crate::adapt;
use crate::decide::{assured, pays, Outcome, Read};
use crate::ensemble::{Ensemble, Quadrature, Start, Well, GROUP};
use crate::estimate::{allowance, between, bound_motion, draws_for, half_window, occupation, semiclassical, spread, summarize, unseen, Between, Estimate, Fate, Group, Held, Scales, Spread};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, significant, span, when, Account, Ledger};
use crate::linalg::V3;
use crate::model::optical::Unmade;
use crate::model::reach::{holding, top, Cell};
use crate::model::units::{destined, Coverings, Judged};
use crate::model::tabulate::{Key, Tables};
use crate::model::{Envelope, Gate, Lights, Model, Priced, Reach, Run, Sample, Trap, Unit, UnitError, GROUND};
use crate::request::STATES;
use crate::step::{stumpff, Dynamics, Price, Refused, Scratch, Segment, Trajectory, Tube};
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
/// What the engine did with a folded unit it did not read (`booked`): its gate's bound over the reach is its error.
const FOLD_BOUNDED: &str = "绝热消去（二阶），误差按原子活动范围上的上界计入误差预算";
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
///   state's prepared trap, every component's well and the initial ensemble drawn in them, T(0)'s quadrature at its
///   least, the reported trap, the entries the first rung reads (`Setting`): made once for a model, and the same at
///   every attempt at it;
/// - on the proof set as well -- the reach and its tubes, the gates over them (`Proven`), how the units are booked,
///   what the reach leaves out of the ensemble, the kick's curvature, the first rung -- and on the step the attempt
///   begins at too -- the Brownian path's cell, the tables' bound and so the tables, every trajectory and all that is
///   read on them: an attempt's own.
/// So a proof set that has grown (`Attempt::Grow`) is attempted with the model as it stands, unless a fold then fails
/// over it: such units are kept (SPEC §1.5) and the model is closed again, as it is before its first attempt, until
/// none fails (a model lists only what it folds, model.rs `units`).
pub fn solve(request: &Request, checkpoint: &mut dyn FnMut(&Solution, Option<f64>)) -> Result<Solution, String> {
    request.validate()?;
    let light = Light::new(request);
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request);
    // The reference state (SPEC §2.1): the initial component of the largest population, of equal ones the last.
    let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).expect("a validated request's initial state holds a ground state");
    let reference = alone(initial[chief].0);
    // The units kept, the reach's temperature and the prepared energy it holds outside the trap's region (model/reach.rs,
    // # The trap's region), the tubes (# Tubes) -- all only grow --, the step the last attempt stood at where it kept
    // units (`Attempt::stood`).
    let (mut kept, mut reach, mut beyond, mut tubes, mut step) = (vec![], request.temperature_uk, f64::NEG_INFINITY, Vec::<Tube>::new(), None);
    // The last model's proof set: where the next model's proofs begin (`Proven`).
    let mut last: Option<Proven> = None;
    let mut messages = Messages { send: checkpoint, clock: Instant::now(), last: None, least: 0 };
    loop {
        let model = Model::new(request, &light, &kept)?;
        // A model's units ask nothing of its trap: the two are made side by side on the pool.
        let (trap, units) = rayon::join(|| model.trap(&reference, &model.prepared()), || model.units());
        let trap = trap?;
        // The reach of a temperature that holds a prepared energy outside the trap's region, and the proof set of it and
        // the tubes with the model's gates over it (`Proven`), and the units whose folds do not exist there.
        let region = |reach: f64, beyond: f64| {
            let made = model.reach(&trap, &reference, reach, unseen(MOST_TRAJECTORIES as f64));
            if beyond > made.minimum + made.outside { model.outside_to(&made, beyond) } else { made }
        };
        let failing = |proven: &Proven| -> Vec<Unit> { units.iter().zip(&proven.gates).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect() };
        let mut proven = Proven::new(&model, &units, region(reach, beyond), &tubes, last.as_ref());
        // The units this model is to be closed again with: those that fail over the proof set, or the one an attempt keeps.
        let mut more = failing(&proven);
        if more.is_empty() {
            // The reported trap and the entries the first rung reads ask nothing of one another: side by side on the pool.
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&model, &light, alone(s))).collect::<Result<_, _>>()?;
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm)?;
            let (reported, entries) = rayon::join(|| model.trap(&reference, &model.reported()), || model.sampled(&model.running()));
            let judged_units = model.judged(&units);
            let setting = Setting { model: &model, reference, units: &units, gates: &judged_units, trap: &trap, ensemble: &ensemble, depth: ensemble.depths()[chief], reported: reported.ok(), entries, start: exact(&ensemble, &ensemble.kinetic(0)) };
            // The states of the run's level whose atoms refuted the last attempt's reach: the next attempt at this model
            // decides on them first (# An attempt's own).
            let mut carried: Option<Vec<Part>> = None;
            while more.is_empty() {
                // An attempt's results improve on its own last one (module documentation, # Checkpoints).
                messages.least = 0;
                let ended = attempt(request, &setting, &proven, &kept, reach, step, carried.take(), &mut messages)?;
                step = ended.stood();
                match ended {
                    Attempt::Grow(t, outside, new, states) => {
                        // The set grown holds the one before it: the reach's level outside the region kept, the tubes.
                        beyond = beyond.max(proven.region.minimum + proven.region.outside).max(outside);
                        tubes.extend(&new);
                        (reach, carried) = (t, states);
                        proven.grow(&model, &units, region(reach, beyond), &new);
                        more = failing(&proven);
                    }
                    Attempt::Keep(units, _) => more = units,
                    Attempt::Done(s) => return Ok(s),
                }
            }
        }
        kept.extend(more);
        last = Some(proven);
    }
}

/// The proof set (module documentation, # The reach's atoms; model/reach.rs, # Tubes) with the model's gates over it:
/// the temperature's reach and the tubes, each a part proven from its own box (model/units.rs `Model::gates`), a unit's
/// gate the larger of its parts' bounds, each statement's. A part is proven once for a model: a set grown proves what it
/// adds alone -- the reach where its temperature or its level outside the region grew, and the new tubes --, and every
/// part it held keeps its proof, which the same model gives again over the same part to the bit. A part proven again --
/// the reach grown, or every part after a keep -- begins every unit's statement on the cells its last proof of the
/// part ended on (model/reach.rs, # A proof begun where one ended): what is new is proven, what is not is read again.
#[derive(Clone)]
struct Proven {
    region: Reach,
    tubes: Vec<Reach>,
    /// Per part its units' gates and the cells their proofs ended on: the reach's, then each tube's in their order.
    parts: Vec<(Vec<Gate>, Coverings)>,
    /// Per unit its gate over the whole set.
    gates: Vec<Gate>,
}
impl Proven {
    /// The set of the reach `region` and the tubes `tubes`, its gates over the model's `units`, each part's proofs
    /// begun where the set `from`'s of the same part ended -- the model's before a keep (`solve`).
    fn new(model: &Model, units: &[Unit], region: Reach, tubes: &[Tube], from: Option<&Proven>) -> Proven {
        let mut proven = Proven { parts: vec![model.gates_from(units, &region, from.map(|f| &f.parts[0].1))], region, tubes: vec![], gates: vec![] };
        proven.add(model, units, tubes, from);
        proven
    }
    /// The set grown to the reach `region` and the further tubes `tubes`: the reach proven again where it is another,
    /// from its own proofs' cells.
    fn grow(&mut self, model: &Model, units: &[Unit], region: Reach, tubes: &[Tube]) {
        if region != self.region {
            self.parts[0] = model.gates_from(units, &region, Some(&self.parts[0].1));
            self.region = region;
        }
        self.add(model, units, tubes, None);
    }
    /// The tubes `tubes` added, each proven -- from where the set `from`'s proofs of it ended, where it has the tube --;
    /// the units' gates over the set.
    fn add(&mut self, model: &Model, units: &[Unit], tubes: &[Tube], from: Option<&Proven>) {
        for tube in tubes {
            let made = model.tube(&self.region, tube);
            let before = from.and_then(|f| f.parts.get(self.parts.len())).map(|part| &part.1);
            self.parts.push(model.gates_from(units, &made, before));
            self.tubes.push(made);
        }
        self.gates = (0..units.len()).map(|u| self.parts.iter().fold(Gate { saturation: 0., sweep: 0. }, |most, (part, _)| Gate { saturation: most.saturation.max(part[u].saturation), sweep: most.sweep.max(part[u].sweep) })).collect();
    }
    /// The set's parts: the reach, then the tubes.
    fn parts(&self) -> impl Iterator<Item = &Reach> {
        std::iter::once(&self.region).chain(&self.tubes)
    }
    /// The largest speed in the set (model.rs `Reach::speed`): of the reach and of every tube.
    fn fastest(&self) -> f64 {
        self.tubes.iter().map(|t| t.speed).fold(self.region.speed, f64::max)
    }
}

/// What the paths not proven of a level ask of the proof set (step.rs `Unproven`; module documentation, # The reach's
/// atoms): the largest prepared energies a temperature's reach is to hold within the trap's region and outside it (−∞:
/// nothing), and the tubes of the paths no temperature's reach holds.
#[derive(Clone, Debug, PartialEq)]
struct Asks {
    region: f64,
    outside: f64,
    tubes: Vec<Tube>,
}
impl Asks {
    /// What one step not proven asks.
    fn of(step: &crate::step::Unproven) -> Asks {
        Asks { region: step.region, outside: step.outside, tubes: step.tube.into_iter().collect() }
    }
    /// What two sets of steps not proven ask together: the larger energies, and both's tubes.
    fn join(a: Option<Asks>, b: Option<Asks>) -> Option<Asks> {
        match (a, b) {
            (Some(x), Some(y)) => Some(Asks { region: x.region.max(y.region), outside: x.outside.max(y.outside), tubes: [x.tubes, y.tubes].concat() }),
            (x, y) => x.or(y),
        }
    }
}

/// What an attempt takes of its model alone (`solve`): nothing of it rests on the reach the model is judged over or
/// on the step, so it is made once for a model and is the same at every attempt at it.
struct Setting<'a> {
    model: &'a Model,
    /// The reference state's ground populations (SPEC §2.1).
    reference: [f64; GROUND],
    /// The model's folded units (model.rs `units`), and their gates as functions of a region (model/units.rs `Judged`).
    units: &'a [Unit],
    gates: &'a Judged<'a>,
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
    /// T(0) and n̄(0) by quadrature with no budget (`exact`): the two levels it takes at least -- an error needs two
    /// to differ --, for the results an attempt's pilot sends before its own (`attempt`).
    start: (V3, f64),
}

/// T(0)'s quadrature as a result takes it (ensemble.rs `Ensemble::kinetic`): (ħ/m)⟨κ_a²⟩ per lab axis, and the largest
/// of the components' differences between their last two levels over their values, T(0)'s booked error
/// (`Fixed::start`).
fn exact(ensemble: &Ensemble, kinetic: &[Quadrature]) -> (V3, f64) {
    (ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>()), kinetic.iter().map(|k| k.difference / k.eta).fold(0., f64::max))
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

/// How one attempt at a model ends (`attempt`): the proof set the model was judged over is outgrown -- by the pilot's
/// atoms, or by a step not proven (`Domain`) -- (the reach's new temperature, µK, the prepared energy it is to hold
/// outside the trap's region, rad/µs, −∞ where nothing asks, the tubes to add, and the states of the run's level whose
/// atoms refuted the reach, where they did: the model is attempted again as it stands unless a fold fails over the
/// grown set, `solve`, and decides on those states first, # An attempt's own); units are to be kept (`keep`), with the
/// step the attempt stood at; or the solution.
enum Attempt {
    Grow(f64, f64, Vec<Tube>, Option<Vec<Part>>),
    Keep(Vec<Unit>, f64),
    Done(Solution),
}
impl Attempt {
    /// The step the next attempt may begin at (`begins`): after a keep, the one this attempt stood at -- over the same
    /// reach --; after a growth none. A step the check took on a reach the atoms refuted or whose domain its paths left
    /// is that reach's -- its domain, its tables, its ladder --, and the next attempt begins at its own first rung.
    fn stood(&self) -> Option<f64> {
        match self {
            Attempt::Keep(_, tau) => Some(*tau),
            Attempt::Grow(..) | Attempt::Done(_) => None,
        }
    }
}

/// An attempt's first step (module documentation, # The step): its first rung, or the step the attempt before stood
/// at where it has one and that is the shorter (`Attempt::stood`).
fn begins(stood: Option<f64>, first_rung: f64) -> f64 {
    stood.map_or(first_rung, |step| step.min(first_rung))
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
/// speed along an axis at the request's temperature `temperature` (µK), v_T = √(k_BT/m), and T's allowance there at the
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
/// `fastest` and `typical` (µm/µs) of the reach the model is judged over. (Both are bounds on the step from above: the
/// least even number is found by counting up, from the prior's.)
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
/// admissible then). The solve refuses such a request (`attempt`): a step that is not admissible computes nothing.
fn first_rung(run: f64, prior: f64, frequency: f64, kick: f64, sampled: &[[f64; 2]], fastest: f64, typical: f64) -> Option<f64> {
    // The fewest pairs whose rung is within the prior: run/(2n) ≤ prior from n = run/(2 prior) on (none past the cap).
    let least = (run / (2. * prior)).ceil().max(1.).min((MOST_STEPS / 2 + 1) as f64) as u64;
    (least..=MOST_STEPS / 2).map(|pairs| run / (2 * pairs) as f64).find(|&step| admissible(step, frequency, kick, sampled, fastest) && admissible(2. * step, frequency, kick, sampled, typical))
}

/// The rms speed (µm/µs) of atoms at the temperature T (µK): √(3 (ħ/m) k_BT/ħ), the reach's typical atom at the
/// mean kinetic temperature of the ensemble the reach is made for (model.rs `Reach::kinetic_uk`) -- and no faster than
/// the proof set's fastest atom, `fastest` (`Proven::fastest`): no speed a step is taken for is one the set does not
/// hold. (A reach grown towards its top holds the trap's depth at most, while its temperature's s²T grows without bound:
/// m20's scout asked for a path 0.6 % below the top, the reach grew to 11 498 µK, its typical speed 1.82 µm/µs against
/// its largest 0.397, and the companion's limit set a first rung of 0.107 µs, τ 0.0537 and a product of 1093 CPU-s.)
fn typical_speed(hbar_over_mass: f64, temperature_uk: f64, fastest: f64) -> f64 {
    (3. * hbar_over_mass * KT_PER_UK * temperature_uk).sqrt().min(fastest)
}

/// Where a level's steps are proven (SPEC §1.8; module documentation, # The reach's atoms; step.rs, # The domain): the
/// model with its units' gates (`judged`; those the whole space at the proof set's largest speed `fastest` leaves
/// `open`, and all of them, `every`), the proof set it was judged over -- the reach and its tubes (`Proven`) --, the
/// prepared energy above the trap's bottom at and above which no temperature's reach holds a path (`top`, as `holding`
/// reads it: model/reach.rs, # Tubes), the trap, and the edge of the step -- the trap's largest frequency, the kick's curvature over the set with the beams' largest
/// moduli it was bounded by (`moduli`), and the entries `resolves` reads.
/// Proven is what is wrong without bound where it fails (step.rs, # The domain): the folds exist, at every level -- one
/// that does not is no model, and its series diverges (a fold's rate went as the Doppler shift squared, SPEC §1.8) --,
/// and the step is stable, the edge of every level (`stable`, `first_rung`). What is accuracy is booked, not proven: the
/// tables' error, an interpolant of the elimination's own nodes, is read at the atoms outside the reach as every unit's
/// is (`apart`); a step that does not resolve the light at the atom's speed is taken, and on a level whose step is the
/// run's (`result`: its rungs and those below) the atom is unresolved from it on and booked whole (`apart`). A
/// companion's atoms beyond its light's limit weigh in its difference by their share. (Held to the tables' verified
/// ranges, PGC3D's pilot's companion at 2.6 µs, which heats two atoms to 1.8 and 2.0 times the reach's level, grew the
/// reach from 30 to 54 µK and the run's step fell from 0.649 to 0.532 µs, 64 % more work; held to the light's resolution
/// at their speeds, R3_DOPPLER's run took 6 attempts and 2.7 times the steps, the reach growing by atoms 1--6 % faster than
/// its largest speed, each grown reach's product finding a faster one.)
#[derive(Clone, Copy)]
struct Domain<'a> {
    model: &'a Model,
    reach: &'a crate::model::Reach,
    tubes: &'a [crate::model::Reach],
    top: f64,
    fastest: f64,
    trap: &'a Trap,
    judged: &'a Judged<'a>,
    open: &'a [usize],
    every: &'a [usize],
    hbar_over_mass: f64,
    frequency: f64,
    kick: f64,
    moduli: &'a [f64],
    sampled: &'a [[f64; 2]],
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
    /// A step's path (step.rs, # The domain) is proven where it lies within the proof set: within the reach -- the
    /// largest prepared energy on it, the reference's potential at its highest along the path (`Flow::highest`, the
    /// reach's potential: the reference state's prepared one) and (ħ/m)κ_m²/2, within the reach's level outside the
    /// trap's region, or within its level and the path's box within the region (model/reach.rs, # The trap's region) --,
    /// or within a tube -- its box within the tube's, its energy within the tube's level (model/reach.rs, # Tubes): there
    /// the set's proofs hold and the first rung is admissible, and nothing more is read. Else at its own state, by the
    /// same statements the set's proofs make over a cell, made along the path at the path's speed
    /// (`Model::holds_along`): every unit whose gate the whole space at that speed does not settle has its fold there
    /// (of a speed within the set's, the units the whole space at the set's largest speed leaves open, else every one);
    /// and the step is stable there -- the kick's curvature the set's, unless a beam is stronger over the path's box than
    /// over the set's boxes, and then the curvature with the stronger moduli (model/reach.rs `Model::kick_curvature`). A
    /// step so proven is taken; whether it resolves the light along its path at its speed is told where the level's step
    /// is the run's (`resolves`).
    ///
    /// Else the set is to hold the path (`Unproven`, `attempt`). Where some temperature's reach holds its energy (below
    /// `top`), the reach: within the region its energy; outside it the same, where the path lies outside the region (its
    /// box does not meet it) or where the reach already holds its energy within it -- what fails then is of the path's
    /// part outside the region. A path that crosses the region's edge above the reach's level asks for the region first:
    /// an atom outside the region above the far value is lost (SPEC §1.7), and an unbound atom crossing the edge on its
    /// way out, held outside the region, would make the reach the whole space. Where none does, its own tube: the path's
    /// box at its energy -- the reach's level is no atom's of that energy.
    fn proven(&self, flow: &crate::step::Flow, (r, kappa): (V3, V3), half: f64) -> Result<bool, crate::step::Unproven> {
        let (model, reach) = (self.model, self.reach);
        let v = kappa.map(|k| self.hbar_over_mass * k);
        let energy = flow.highest(r, v, half) + 0.5 * self.hbar_over_mass * kappa.iter().map(|k| k * k).sum::<f64>();
        if energy <= reach.minimum + reach.outside {
            return Ok(true);
        }
        let path = Cell { centre: r, half: v.map(|x| (x * half).abs()) };
        let spans = model.spans(&path);
        let [meets, within] = model.region_of(&spans);
        let held = energy <= reach.minimum + reach.energy;
        if held && within || self.tubes.iter().any(|tube| Model::tube_holds(tube, &path, energy)) {
            return Ok(true);
        }
        let speed = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        let units = if speed <= self.fastest { self.open } else { self.every };
        let kick = if spans.iter().zip(self.moduli).all(|(span, most)| span.most <= *most) {
            self.kick
        } else {
            model.kick_curvature(reach, &spans.iter().zip(self.moduli).map(|(span, most)| span.most.max(*most)).collect::<Vec<f64>>(), self.trap)
        };
        if stable(2. * half.abs(), self.frequency, kick) && model.holds_along(self.judged, units, r, v, half, speed) {
            return Ok(!self.result || resolves(2. * half.abs(), self.sampled, speed));
        }
        Err(if energy - reach.minimum >= self.top {
            crate::step::Unproven { region: f64::NEG_INFINITY, outside: f64::NEG_INFINITY, tube: Some(Tube { centre: r, half: path.half, level: energy }) }
        } else {
            crate::step::Unproven { region: energy, outside: if !within && (!meets || held) { energy } else { f64::NEG_INFINITY }, tube: None }
        })
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
    /// above it): the rung is left -- as one whose companion's paths were not proven is (`settle`).
    Left,
    /// On the pilot, its own read within and no third level on the pilot's pairs, where the ladder has the rung above
    /// and that rung's paths stay in the model's domain: the pilot's trajectories at 4τ, a quarter of a trajectory each.
    Above,
    /// Its own read's estimate beyond the share, but not clearly: the noise hides the difference -- the pairs that
    /// could see through it (`remedy`), else the rung is left.
    Unseeing,
}

/// What the check makes of a rung (module documentation, # The step; `settle`) on its own `read` and its third level
/// above carried to its step, `above` (none where it has none) -- a rung whose companion's paths left the model's domain
/// has no read and is left (`settle`) --; `pilot`: whether the read is the pilot's, the last word on the step (the scout's
/// leaves only on what is clearly beyond); `upward`: whether a third level above on the pilot's own pairs is to be had
/// (the ladder has the rung above, the rung has none yet, and the rung above did not leave the model's domain). Every
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

/// What a read does not exclude (decide.rs `Read::most`): each quantity's far end, the rms it leaves open,
/// √(max(M̂, 0) + t·SE) at one look's interval -- the number `blind` holds against the step's share --, the largest on
/// what is shown, and the total T's; unbounded where a read excludes nothing.
fn unexcluded(read: &Between) -> Bound {
    let far = |read: &Read| read.most(1.).sqrt();
    Bound { shown: probability_error(read.shown.iter().map(far).fold(far(&read.loss), f64::max)), temperature: read.temperature.map_or(f64::INFINITY, |t| far(&t)) }
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

/// Of the bound atoms the groups `parts` hold at every listed output (`Model::situated`: the leaving ones are no draw of
/// the distribution a reach is made for, module documentation, # The reach's atoms): how many lie outside the reach,
/// and how many there are.
fn outside(model: &Model, reach: &crate::model::Reach, parts: &[&Part]) -> Vec<(f64, f64)> {
    let listed = parts.first().map_or(0, |p| p.listed.len());
    (0..listed)
        .map(|j| {
            parts.iter().flat_map(|p| &p.listed[j]).map(|(r, kappa)| model.situated(reach, *r, *kappa)).filter(|[_, leaving]| !leaving).fold((0., 0.), |(outside, all), [held, _]| (outside + f64::from(u8::from(!held)), all + 1.))
        })
        .collect()
}

/// The reach's falsifier (SPEC §1.8; decide.rs: a booked bound that a read can only refute). The reach is made for an
/// ensemble -- the bound atoms' at its temperature, of which it leaves out the share `left` -- and the model is judged
/// over it as holding them. The pilot reads the two things this says of its bound atoms (`motion`, estimate.rs
/// `bound_motion`; module documentation, # The reach's atoms: a leaving atom is held by its tube, and is no reading of a
/// temperature), as one family of looks (Šidák over all of them, each one-sided at the confidence z):
/// - no lab axis's T at an output is above the largest kinetic temperature along an axis of that ensemble, `reach`
///   (µK; model.rs `Reach::kinetic_uk`): refuted by a T read above it beyond its interval;
/// - of the bound atoms held at a listed output, those outside the reach (`outside`: per listed output the count and
///   the atoms) are a binomial's of the share `left` at the most: refuted by a count whose tail (estimate.rs
///   `at_least`) is below the look's -- among the six hundred looks of two hundred outputs, three of the pilot's 88.
///   This read estimates no spread. The T's interval is 6.4 standard errors there, each estimated on eleven groups: a
///   T twice the reach's passes it wherever that estimate falls high, with a tenth of the atoms outside (measured,
///   SPEC §1.8).
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
/// axis at an output; no number where none is read.
fn hottest(motion: &[Option<crate::estimate::Motion>]) -> f64 {
    motion.iter().skip(1).flatten().flat_map(|m| m.temperature).fold(f64::NEG_INFINITY, f64::max)
}

/// What the atoms outside the reach add to the model's error at every output (SPEC §1.8, §4; module documentation,
/// # The reach's atoms), on the held atoms' populations (absolute) and on T (relative), from the run's `estimates`
/// (their shares apart, estimate.rs `Estimate::outside`, `Estimate::unresolved`), the outside states its groups `parts`
/// keep and the model's error `booked` there (`Fixed::model`, the pilot's):
/// - the unresolved atoms, whole: their share of the atoms on what is shown, of the kinetic energy on T;
/// - the atoms outside, resolved, by their share of the atoms and of the kinetic energy, times what the model's error
///   at their states -- `read` at a state: every folded unit's and the tables', booked at the output's populations
///   (`Booking::at`), at most the whole of what the atom carries -- exceeds the error booked. The mean over the first outside state of each of the first groups that hold
///   one at the output, as many groups as the pilot has: as many states an output as the model's booked error was read
///   at. A state's read is kept (`known`, by trajectory and output): it is read once in a run.
fn apart(estimates: &[Estimate], parts: &[&Part], booked: &[[f64; 2]], read: &(dyn Fn(&State) -> Booking + Sync), known: &mut HashMap<(u64, usize), Booking>) -> Vec<[f64; 2]> {
    let groups = (pilot_size() / GROUP) as usize;
    let states: Vec<Vec<&State>> = (0..estimates.len()).map(|k| parts.iter().filter(|_| k > 0).filter_map(|p| p.samples.get(k).and_then(|s| s.first())).take(groups).collect()).collect();
    let new: Vec<(u64, usize, &State)> = states.iter().enumerate().flat_map(|(k, at)| at.iter().map(move |state| (state.index, k, *state))).filter(|(index, k, _)| !known.contains_key(&(*index, *k))).collect();
    let errors: Vec<Booking> = new.par_iter().map(|(_, _, state)| read(state)).collect();
    known.extend(new.iter().zip(errors).map(|((index, k, _), e)| ((*index, *k), e)));
    estimates
        .iter()
        .zip(&states)
        .zip(booked)
        .enumerate()
        .map(|(k, ((e, states), booked))| {
            std::array::from_fn(|i| {
                // No outside state read at the output: no atom outside there (the first of every group is kept).
                let excess = if states.is_empty() {
                    0.
                } else {
                    let mean = states.iter().map(|state| known[&(state.index, k)].at(&e.populations).map(probability_error)[i]).sum::<f64>() / states.len() as f64;
                    (mean - booked[i]).max(0.)
                };
                e.unresolved[i] + e.outside[i] * excess
            })
        })
        .collect()
}

/// Whether a lab axis's temperature (µK) lies in the validated regime of that axis's frequency (rad/µs), k_BT ≥
/// `CLASSICAL` ħω (SPEC §0, §3): what the result reports as classical, per axis; reporting only.
fn classical(temperature: f64, frequency: f64) -> bool {
    KT_PER_UK * temperature >= CLASSICAL * frequency
}

/// A state's model error as the run books it (`Measured::means`, `apart`): on the populations (absolute) and on T
/// (relative) what its units book there, of which the part a flow's rate error moves to its far side
/// (`UnitError::moved`, `into`; model/units.rs, # A rate's error on the populations) is the source side's as the
/// state sees it, and is taken against the output's populations (model/units.rs `destined`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Booking {
    error: [f64; 2],
    moved: f64,
    into: u32,
}
impl Booking {
    const NONE: Booking = Booking { error: [0.; 2], moved: 0., into: 0 };

    /// A unit's error at a state.
    fn of(e: &UnitError) -> Booking {
        Booking { error: [e.population, e.temperature], moved: e.moved, into: e.into }
    }

    /// Several units' at one state: their flows have one far side, the state's (model/units.rs `Model::moved_into`).
    fn add(self, o: &Booking) -> Booking {
        Booking { error: [self.error[0] + o.error[0], self.error[1] + o.error[1]], moved: self.moved + o.moved, into: self.into | o.into }
    }

    /// What is booked at an output of the held populations `p` (estimate.rs `Estimate::populations`).
    fn at(&self, p: &[f64; STATES]) -> [f64; 2] {
        let less = self.moved * (1. - destined(self.into, p));
        [self.error[0] - less, self.error[1] - less]
    }
}

/// The states the model is read at and the errors there of the units read, `units` (SPEC §5.2): at every output the
/// first member of every group of `parts` that still holds its atom -- one a group keeps every group in a mean's spread,
/// and a unit's error is a function of the state; the first held one, so that an output some atom reaches is read
/// wherever a group has one there --, with each state's output and group, and per state the units' errors in their
/// order. A state `known` already holds -- the same trajectory at the same output and step, the scout's inside the pilot
/// -- keeps the errors read there: a unit's error is a function of the state.
///
/// The errors are one table (`Measured::errors`): the states' rows one after another, a row the units' errors on the
/// populations and on T as each books them there (`Booking`), each row written on the pool where it is read. One block of memory, not one a state: at a
/// thousand outputs the states are ten thousand, and that many blocks took the calling thread 0.2 to 0.6 s to free.
struct Sampled {
    /// Each state's trajectory and step.
    of: Vec<(u64, f64)>,
    at: Vec<usize>,
    from: Vec<usize>,
    errors: Vec<Booking>,
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
    let mut errors = vec![Booking::NONE; states.len() * width];
    // A state's row (the table is empty where no unit is read).
    errors.par_chunks_mut(width.max(1)).zip(&states).zip(&at).for_each(|((row, state), &k)| match (&known, read.get(&key(state.index, k, state.tau))) {
        (Some(known), Some(&j)) => row.copy_from_slice(&known.errors[j * width..][..width]),
        _ => row.iter_mut().zip(model.errors_at(units, &state.sample, Some(tables))).for_each(|(row, e)| *row = Booking::of(&e)),
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

/// Whose reads a keep is judged on (`keep`): the scout's few, the pilot's, or the product's statistics beside the
/// pilot's reads of the units.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Reads {
    Scout,
    Pilot,
    Product,
}

/// The folded units to keep, if any (SPEC §5.2; module documentation, # The model), from the units' errors `measured`
/// at the states of the trajectories `parts` (their `estimates`; the units' mean errors over them `means`, per output
/// and unit) and the errors `fixed` beside them, the step's taken at its share (a pilot's measured bound is its own
/// noise): the units the looks need kept to have room for the statistics (`needed`); else, of the units whose error's
/// removal alone would pay for the `again` trajectories its model runs before it stands where this one does, each
/// rated with its kept model and paying beyond its spread, the leading ones by their saving whose keep together saves
/// the most beyond its spread (`Pricing::joint`). On the scout, whose reads are few, both are judged at the model's
/// error they leave no doubt of (`Doubt`, `clear`): the room, and the work the current model's error asks. The units
/// the room needs are taken to the far end of the doubt (`needed`, `raised`): the next pilot's read of the error they
/// leave has room. In the product (`Reads::Product`) the room alone is asked: a look the pilot left room that the
/// units' error the pilot booked leaves none at the product's statistics is the room's, which a keep gives where one
/// does, and the attempt ends with it (`product`) -- the pilot has priced the rest.
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
    scales: &Scales,
    unsampled: [f64; 2],
    tabled: (&Tables, &Reach),
    again: f64,
    reads: Reads,
) -> Option<Vec<usize>> {
    let held: Vec<f64> = estimates.iter().map(|e| 1. - e.loss).collect();
    // A unit weighs where it was read: an output with no state read carries every unit's bound alike, and tells none
    // of them apart. And where the units' errors crowd it: the share of trajectories not counted (estimate.rs, # What is
    // counted) is no unit's, and no keep takes it away.
    let mut read = vec![false; fixed.model.len()];
    measured.output.iter().for_each(|&k| read[k] = true);
    let counted: Vec<Estimate> = estimates.iter().map(|e| Estimate { uncounted: 0., ..e.clone() }).collect();
    // On the scout the model's error is judged at what its few reads leave no doubt of (`Doubt`): the room and the
    // pricing alike -- a keep is never undone, and the reads that put the room's error high put the work the current
    // model asks high with them; the pilot reads the same units on all its states.
    let doubt = || Doubt::new(measured, fixed.model.len(), parts, scales, unsampled);
    let scouted = (reads == Reads::Scout).then(doubt);
    let near = scouted.as_ref().map_or_else(|| vec![[0.; 2]; fixed.model.len()], |d| d.interval(&[]));
    let mut pricing = Pricing::new(Formula::new(model, kept, units, reference, centre, tabled), measured, means, parts, estimates, fixed, scales, unsampled, again, &near);
    // The units that weigh where the looks have no room, each with the work a µs of its kept model (`needed`; none where
    // its model is refused, `Model::new`), the set they need taken to the far end of the error it leaves.
    let room = clear(fixed, &near);
    if !weighing(&counted, &room, means, &held, &read).is_empty() {
        let spread = scouted.unwrap_or_else(doubt);
        let far = |set: &[usize]| raised(&without(fixed, means, &read, set), &spread.interval(set));
        // A kept set's work a µs (its mean rate, made once a set) and the bound below every set's (`Pricing::least`); the
        // product's trajectories beside the error it leaves (`Level::need_without_all`), and the chance that the next
        // pilot reads a look without room (`Doubt::chance`).
        let rates: std::cell::RefCell<HashMap<Vec<usize>, Option<f64>>> = std::cell::RefCell::new(HashMap::new());
        let rate = |set: &[usize]| -> Option<f64> {
            let mut key = set.to_vec();
            key.sort_unstable();
            if let Some(&made) = rates.borrow().get(&key) {
                return made;
            }
            let made = pricing.rates(&key).map(|r| pricing.mean(&r, None));
            rates.borrow_mut().insert(key, made);
            made
        };
        let price = |set: &[usize], beyond: &[usize]| -> [f64; 2] {
            let point = without(fixed, means, &read, set);
            [pricing.all.need_without_all(set), spread.chance(set, &point, &counted, &held, &read, beyond)]
        };
        let needed = needed(&counted, means, &held, &read, pricing.least(), &rate, &far, &price, pricing.again);
        return (!needed.is_empty()).then_some(needed);
    }
    if reads == Reads::Product {
        return None;
    }
    let saving = pricing.saving_at_all();
    if saving.is_empty() {
        return None;
    }
    pricing.own = pricing.own();
    let pricing = &pricing;
    let others = pricing.leave_one_out();
    let screened: Vec<usize> = saving.into_par_iter().filter(|&u| pays(0., pricing.saving(&others, u, &pricing.own))).collect();
    let mut ranked: Vec<Candidate> = screened.into_iter().filter_map(|u| pricing.priced(&others, u)).filter(|c| pays(0., c.saving)).collect();
    ranked.sort_by(|a, b| assured(b.saving).total_cmp(&assured(a.saving)));
    let ranked: Vec<usize> = ranked.into_iter().map(|c| c.unit).collect();
    (!ranked.is_empty()).then(|| pricing.joint(&others, &ranked))
}

/// One attempt at the solve (module documentation, # The pipeline) with the model of the units `kept` as `setting`
/// holds it, judged over the proof set `proven` -- the reach of the temperature `reach` µK and the tubes --, where none
/// of its gates fails. `step`: the step the attempt before stood at where it may begin there (`Attempt::stood`);
/// `carried`: the states of the run's level whose atoms refuted the reach before this one, where they did.
#[allow(clippy::too_many_arguments)]
fn attempt(
    request: &Request,
    setting: &Setting,
    proven: &Proven,
    kept: &[Unit],
    reach: f64,
    step: Option<f64>,
    carried: Option<Vec<Part>>,
    messages: &mut Messages,
) -> Result<Attempt, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let Setting { model, reference, units, trap, ensemble, depth, .. } = *setting;
    let (judged, gates, fastest) = (&proven.region, &proven.gates, proven.fastest());
    let entries = &setting.entries;
    // The ensemble's proposals so far that read deeper than their well's bottom: the attempts' before this one.
    let deeper = ensemble.deeper();
    // The folded units: which are read at the states, which booked at their gates' bounds (`booked`).
    let booking = booked(&units.iter().zip(gates).map(|(u, g)| g.error_bound(u)).collect::<Vec<_>>());
    let reading: Vec<Unit> = units.iter().zip(&booking).filter(|(_, b)| matches!(b, Booked::Read(_))).map(|(u, _)| u.clone()).collect();
    // What the reach the model was judged over leaves out of the initial ensemble (model/reach.rs `Model::left_out`): of
    // the atoms and of their kinetic energy, by the ensemble's quadrature of its own density -- no atom is drawn --, a
    // level of it with no more evaluations of the potential than the most trajectories a solve runs have initial atoms.
    // It and the kick's curvature over the proof set (`admissible`), of the beams' largest moduli over its boxes, ask
    // nothing of one another: side by side on the pool.
    let moduli = model.moduli(proven.parts());
    let (unsampled, kick) = rayon::join(|| model.left_out(judged, ensemble, MOST_TRAJECTORIES), || model.kick_curvature(judged, &moduli, trap));
    let (report, unheld) = (setting.reported.as_ref().unwrap_or(trap), setting.reported.is_none());
    let scales = Scales { hbar_over_mass: hm, frequencies: report.lab };
    let plan = Plan::new(&request.times_us, report.lab);
    let outputs = plan.times.len();
    let row: [f64; STATES] = std::array::from_fn(|s| request.initial_density[(s, s)].re);
    let mut ledger = Ledger::default();
    ledger.note = Some(format!("初始系综：每个初态分量的原子束缚在它自己的势阱里（降低的等温分布 f ∝ e^(−E/k_BT) − 1，只含束缚原子，阱深比 k_BT 大得多时即热分布；每个主轴再乘谐振子的 Wigner 因子，T = {} μK）；布居最大的初态的阱深为 {:.1} k_BT", request.temperature_uk, depth));
    if unheld {
        ledger.flag("计算所用光场中的势没有极小：阱频、n̄、温度的窗口（一个阱周期）与验证范围都按初始势阱给出".into());
    }
    let (momentum, start) = setting.start;
    // The ladder's first rung: within the physics prior's step at the request's temperature, and admissible over the
    // proof set the model is judged over, at its largest speed. An attempt that begins at the step the one before stood
    // at -- after a keep, over the same reach -- begins no higher than its own first rung.
    let prior = prior(model.carrier_wave_number(), hm, request.temperature_uk, &scales);
    let first_rung = first_rung(plan.end, prior, trap.frequencies.iter().copied().fold(0., f64::max), kick, entries, fastest, typical_speed(hm, judged.kinetic_uk.iter().sum::<f64>() / 3., fastest))
        .ok_or_else(|| format!("找不到可用的时间步长：在每条轨迹的步数上限（{MOST_STEPS} 步）之内，没有步长能分辨阱中的振荡、光场中势阱的曲率和光场沿原子路径的变化（或者阱频、曲率、原子速率不是有限数）"))?;
    // What the reach's falsifier compares the T read with: the largest kinetic temperature along an axis of the
    // ensemble the reach was made for (model.rs `Reach::kinetic_uk`: T_r where the trap is classical, the zero point's
    // where it is cold) -- no lab axis's T is above it while the atoms are that ensemble.
    let held = judged.kinetic_uk.iter().copied().fold(reach, f64::max);
    // And the share of its atoms it takes to lie outside the reach: the share the reach is made to leave out of its own
    // ensemble, and no less than what it is known to leave out of the initial one.
    let left = unsampled[0].max(unseen(MOST_TRAJECTORIES as f64));
    // The attempt's first step (`begins`), and whether the rung above a step is a rung of the ladder (`first_rung`'s
    // test at twice the step): the attempt's first step has it where the step the attempt before stood at, or the
    // prior, sets it below the edges. The check then reads that rung on a third level, at 4τ (`settle`): a cell of the
    // Brownian path.
    let begins = begins(step, first_rung);
    let (frequency, typical, end) = (trap.frequencies.iter().copied().fold(0., f64::max), typical_speed(hm, judged.kinetic_uk.iter().sum::<f64>() / 3., fastest), plan.end);
    let above = |tau: f64| 4. * tau <= end && admissible(2. * tau, frequency, kick, entries, fastest) && admissible(4. * tau, frequency, kick, entries, typical);
    // Where the attempt's steps are proven (`Domain`): of the units' gates, those the whole space at the proof set's
    // largest speed leaves open, and all of them; at and above the reach's top, no temperature's reach holds a path.
    let share = unseen(MOST_TRAJECTORIES as f64);
    let (open, every): (Vec<usize>, Vec<usize>) = (model.everywhere(setting.gates, fastest).1, (0..units.len()).collect());
    let domain = Domain { model, reach: judged, tubes: &proven.tubes, top: top(trap, share), fastest, trap, judged: setting.gates, open: &open, every: &every, hbar_over_mass: hm, frequency, kick, moduli: &moduli, sampled: entries, result: true };
    // An attempt that ends on a step not proven: the proof set that holds its paths (model/reach.rs, # The trap's
    // region, # Tubes) -- the temperature whose level holds the paths' energy within the region (below the top, where
    // one does), the level outside it, and the tubes of the paths no temperature's reach holds.
    let hold = |asked: Asks| Attempt::Grow(holding(trap, asked.region, share, reach), asked.outside, asked.tubes, None);
    let mut context = Context {
        momentum,
        fixed: Fixed { model: vec![[0.; 2]; outputs], start },
        outside: vec![[0.; 2]; outputs],
        step: vec![Bound::share(); outputs],
        checked: None,
        tau: begins,
        plan: Plan { cell: if above(begins) { 4. } else { 2. } * begins, ..plan },
        scales,
        initial: row,
        ledger,
        method: format!("三维半经典运动：{}", model.describe().trim_end_matches('。')),
    };

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
    // Whether the pilot's atoms have outgrown the reach (`outgrown`), and the temperature it then grows to: the
    // attempt ends there, and nothing further is spent on a step of that reach (`own_level`, `check`).
    let scales = context.scales;
    let grown = |pilot: &[&Part]| outgrown(&outside(model, judged, pilot), left, &bound_motion(pilot, &scales), held);
    let refuted = |(t, states): (f64, Vec<Part>)| Attempt::Grow(t, f64::NEG_INFINITY, vec![], Some(states));
    let mut levels = Levels::default();

    // The model's decision at some groups' states before the pilot's own (`keep`; module documentation, # The model):
    // the units' errors read there and the model's error they make, the trajectories `again` that a keep runs again of
    // what this attempt has run, and whose reads they are; the units to keep, and the states' read (`Sampled`).
    let start = context.fixed.start;
    let decided = |parts: &[&Part], again: f64, reads: Reads| -> (Option<Vec<Unit>>, Sampled) {
        let estimates = pooled(parts, &scales).0;
        let read = sampled(model, &reading, &tables, parts, None);
        let measured = Measured { errors: &read.errors, output: &read.at, group: &read.from, booked: &booking };
        let means = measured.means(outputs, None, &estimates);
        let fixed = Fixed { model: model_error(&sums(&means, unsampled), unsampled), start };
        let keeps = keep(kept, model, reference, trap.centre, units, &measured, &means, parts, &estimates, &fixed, &scales, unsampled, (&tables, judged), again, reads);
        (keeps.map(|u| u.iter().map(|&u| units[u].clone()).collect()), read)
    };
    // The states the run's level left where its atoms refuted the reach before this one (# An attempt's own): this
    // model's atoms where they went, a pilot's of them, read for its decision before the attempt runs a trajectory --
    // a keep they ask ends it at nothing run again --; they are the decision the scout's eleven states would make here.
    let decided_before = match &carried {
        Some(states) => match decided(&states.iter().collect::<Vec<_>>(), 0., Reads::Pilot).0 {
            Some(units) => return Ok(Attempt::Keep(units, context.tau)),
            None => true,
        },
        None => false,
    };
    drop(carried);

    // The scout (module documentation, # The pipeline): the first member of every pilot group at both levels, screened;
    // where a unit is to be kept at its states, the attempt ends here -- for the price of the scout, not of a pilot.
    // The pilot's read of the same below stands for the result.
    if let Settled::Unproven(asked) = settle(&run, ensemble, &domain, &mut context, &mut levels, &numbers, 0..1, &above, &|pilot| grown(pilot).is_some(), None, false, messages) {
        return Ok(hold(asked));
    }
    // The pilot's own level is run before its companion and read by the reach's falsifier as it comes in (`own_level`):
    // where its atoms refute the reach, the attempt ends with the reach grown to the temperature they read, and the
    // rest of the pilot -- its companion, the step's check at a model judged over a reach the run does not keep -- is
    // never run. Where the scout's atoms already refute it, before the scout's keep, its members doubling a hand-out:
    // the units' errors are read at the states of a reach that stands -- at a refuted one, of atoms past it (S3's
    // scout at 30 µK kept its carrier's dressing on atoms heated past the reach); else after it, one hand-out. Where
    // they do not refute it, or a path of theirs was not proven, which their companion's says what of (`settle`,
    // `own_departure`), the pilot's check goes on from that level, its companion run then (`settle`, `ran`).
    let doubling = grown(&levels.fine.iter().collect::<Vec<_>>()).is_some();
    let mut ran = None;
    if doubling {
        match own_level(&run, ensemble, &domain, &context, &levels.fine, &numbers, true, &grown, messages) {
            Ok(own) => ran = Some(own),
            Err(at) => return Ok(refuted(at)),
        }
    }
    let scouted = if decided_before {
        None
    } else {
        match decided(&levels.fine.iter().collect::<Vec<_>>(), levels.fine.len() as f64, Reads::Scout) {
            (Some(units), _) => return Ok(Attempt::Keep(units, context.tau)),
            (None, read) => Some(read),
        }
    };
    let ran = match ran {
        Some(own) => own,
        None => match own_level(&run, ensemble, &domain, &context, &levels.fine, &numbers, false, &grown, messages) {
            Ok(own) => own,
            Err(at) => return Ok(refuted(at)),
        },
    };
    // The pilot: its groups' other members at both levels, screened, and the step's check (module documentation,
    // # The step) -- first as far as its own two levels: at the first rung whose own read is in and not clearly beyond,
    // the decisions the pilot's reads hold are taken (# Each decision as its evidence is in), and the check goes on
    // past them only where they leave the attempt standing. Whether it saw: a step that stands uncertified -- at the
    // steps' cap, or unable to see -- is flagged.
    let mut settled = settle(&run, ensemble, &domain, &mut context, &mut levels, &numbers, 1..GROUP, &above, &|pilot| grown(pilot).is_some(), Some(ran), true, messages);
    drop(run);
    let mut scouted = scouted;
    // The pilot's reads, made at the rung they were made at; made again only where the check went on to another rung.
    let mut made: Option<(f64, PilotRead)> = None;
    let read = loop {
        if let Settled::Unproven(asked) = settled {
            return Ok(hold(asked));
        }
        let tau = context.tau;
        if let Some(t) = grown(&levels.fine.iter().collect::<Vec<_>>()) {
            return Ok(refuted((t, std::mem::take(&mut levels.fine))));
        }
        let pilot: Vec<&Part> = levels.fine.iter().collect();
        let read = match made.take() {
            Some((at, read)) if at == tau => read,
            _ => pilot_reads(model, &reading, &booking, &tables, &pilot, &levels.coarse, scouted.take(), ensemble, unsampled, outputs, most, &context.scales),
        };
        let fixed = Fixed { model: read.model.clone(), start: read.start.1 };
        let measured = read.measured(&booking);
        let means = measured.means(outputs, None, &read.estimates);
        // Units to keep at the pilot's states (`keep`): their model then runs a pilot again.
        if let Some(u) = keep(kept, model, reference, trap.centre, units, &measured, &means, &pilot, &read.estimates, &fixed, &scales, unsampled, (read.refined.as_ref().unwrap_or(&tables), judged), pilot_size() as f64, Reads::Pilot) {
            return Ok(Attempt::Keep(u.iter().map(|&u| units[u].clone()).collect(), tau));
        }
        if !matches!(settled, Settled::Pending) {
            break read;
        }
        made = Some((tau, read));
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
    // The pilot's groups with their states: the units were read there, and a keep's price in the product (`keep`).
    let pilot: Vec<&Part> = levels.fine.iter().collect();
    let (estimates, counts) = (&read.estimates, &read.counts);
    (context.momentum, context.fixed.start) = read.start;
    context.fixed.model = read.model.clone();
    let measured = read.measured(&booking);
    let means = measured.means(outputs, None, estimates);
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
    // The step's check (module documentation, # The step): what the reads the rung stands by do not exclude into the
    // accounts, their resolution into the result's words (`Levels::told`, `checked`).
    let (own, third, step_of_third, pairs) = levels.told(tau, &context.scales);
    let (step, resolution) = checked(&own, third.as_ref(), saw);
    (context.step, context.checked) = (vec![step; outputs], Some(Checked { third: step_of_third, pairs, resolution, saw }));
    // The optical sector's noise (ledger.rs `optical`), its items; its booking on T is in the model's error the reads
    // made (`pilot_reads`).
    crate::ledger::optical(&mut context.ledger, counts.optical);
    // Units to keep in the product (`keep`), read with the product's estimates.
    let keeping = |estimates: &[Estimate], fixed: &Fixed, reads: Reads| keep(kept, model, reference, trap.centre, units, &measured, &means, &pilot, estimates, fixed, &scales, unsampled, (&tables, judged), pilot_size() as f64, reads);
    {
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
        context.ledger.item("初始时就在估计的活动范围之外的原子所占的比例", unsampled[0], "对初始分布数值积分得到（含积分最后两级之差），计入布居的模型误差");
        context.ledger.item("初始时就在估计的活动范围之外的原子占初始动能的比例", unsampled[1], "对初始分布数值积分得到（含积分最后两级之差），计入温度的模型误差");
    }
    let run = Run::tabulated(model, &tables, reference);

    // The product (module documentation, # Stopping): on from the pilot's trajectories at the run's step, the run's
    // first; the pilot's companions have done their part, and its states have been read -- but those outside the
    // reach, which the product reads with its own (# The reach's atoms): at a state every folded unit's error and the
    // tables', at most the whole of what the atom carries.
    let mut parts = levels.fine.clone();
    outside_alone(&mut parts);
    parts.append(&mut levels.further[0]);
    let booked = context.fixed.clone();
    let read = |state: &State| -> Booking {
        let errors = model.errors_at(units, &state.sample, Some(&tables));
        let reads = model.table_reads(&tables, std::slice::from_ref(&state.sample));
        errors.iter().chain(reads.iter().map(|table| &table.whole[0])).fold(Booking::NONE, |sum, e| sum.add(&Booking::of(e)))
    };
    let (e, l, c) = match product(&run, ensemble, &domain, &mut context, &mut parts, &numbers, &read, &|e, fixed| keeping(e, fixed, Reads::Product), messages) {
        Ok(product) => product,
        Err(Ended::Unproven(asked)) => return Ok(hold(asked)),
        Err(Ended::Keep(u)) => return Ok(Attempt::Keep(u.iter().map(|&u| units[u].clone()).collect(), context.tau)),
    };
    // The reach outgrown as far as the product reads (# The reach's atoms): some look has no room for the statistics
    // with what the atoms outside the reach carry, and has room without it. The reach grows to the temperature read
    // where that is above it, and the next attempt begins, deciding on the pilot's states first; else the account
    // stands as read, flagged.
    if looks(&e, &context.fixed, &context.step).iter().zip(&looks(&e, &booked, &context.step)).any(|(with, without)| !with.fits() && without.fits()) {
        if let Some(t) = Some(hottest(&bound_motion(&parts.iter().collect::<Vec<&Part>>(), &context.scales))).filter(|t| *t > held) {
            return Ok(refuted((t, std::mem::take(&mut levels.fine))));
        }
    }
    // A point of the prepared potential below the bottom the ensemble was drawn about (ensemble.rs, # Its draw).
    if ensemble.deeper() > deeper {
        context.ledger.flag("初态的势在所取的阱底之外还有更深的地方：初始系综只取到该阱底的阱深为止".into());
    }
    Ok(Attempt::Done(assemble(&context, &e, &l, &c, parts.len() as u64 * GROUP)))
}

/// The pilot's reads at the rung it stands at (`attempt`, `pilot_reads`): its estimates and counts, T(0) and its
/// quadrature's error (`exact`), the units' errors at its states (`Sampled`), the model's error per output -- the units',
/// the tables' and the optical sector's booking on T --, each table's booked error and why its refinement ended where it
/// is beyond its shares; and the tables refined, where a read refined one (the pilot's own stay as its levels ran on
/// them: a check that goes on reads its further levels on the tables of the levels it compares them with).
struct PilotRead {
    estimates: Vec<Estimate>,
    counts: Counts,
    start: (V3, f64),
    sampled: Sampled,
    model: Vec<[f64; 2]>,
    tables: Vec<[f64; 2]>,
    stuck: Vec<Option<Stuck>>,
    refined: Option<Tables>,
}
impl PilotRead {
    /// The units' errors as the decisions read them (`Measured`), of the units booked so (`booking`).
    fn measured<'a>(&'a self, booking: &'a [Booked]) -> Measured<'a> {
        Measured { errors: &self.sampled.errors, output: &self.sampled.at, group: &self.sampled.from, booked: booking }
    }
}

/// The pilot's reads (`PilotRead`) of its groups `pilot` and their companions `coarse`, the units `reading` read at its
/// states (the scout's `known` kept), on the pilot's tables `tables`, refined where their reads ask -- a copy, made at
/// the first refinement (`PilotRead::refined`).
/// - T(0) and n̄(0) by quadrature again (`Setting::start`), with as many evaluations of the potential as the pilot took
///   steps: a step evaluates the potential's expansion at least once (its reference flow's, SPEC §2.1), so the
///   quadrature costs no more than the pilot the solve has just run, in any trap; where that budget ends it before its
///   levels agree, their difference is T(0)'s booked error (`Fixed::start`).
/// - The model's errors at the pilot's states (SPEC §5.2): each folded unit's per output, the held atoms' mean over the
///   first member of every group -- one a group keeps every group in the mean's spread, and a unit's error is a function
///   of the state, read as well on eleven states an output as on eighty-eight.
/// - The tables' errors (SPEC §5.4): each table's whole error against the elimination itself, its mean over the run's
///   held states read at one state of every pilot trajectory (`one_each`, `over_the_run`) -- a state on another cluster
///   reads none of it. A table beyond the tables' shares has the axis of its largest read halved and is read again,
///   until it is within them, or it has no axis to halve (its coordinates all held at the focus: model/tabulate.rs,
///   # Axes), or what is beyond is not its axes' interpolation -- the whole less every axis's own read still beyond the
///   shares: a coordinate held at the focus, which no halving lowers --, or its worst axis's read on a grid it resolves
///   fell by no more than an unresolved peak's (model/tabulate.rs, # Refinement: it is not converging); what it reads
///   then is booked on every output, and flagged where it still lies beyond -- with which of these ended its refinement
///   (`Stuck`).
/// - The optical sector's noise (ledger.rs `optical`): its booking on T at every output, a part of the model's error
///   that no unit kept takes away.
#[allow(clippy::too_many_arguments)]
fn pilot_reads(model: &Model, reading: &[Unit], booking: &[Booked], tables: &Tables, pilot: &[&Part], coarse: &[Part], known: Option<Sampled>, ensemble: &Ensemble, unsampled: [f64; 2], outputs: usize, most: usize, scales: &Scales) -> PilotRead {
    let (estimates, _, counts) = pooled(pilot, scales);
    let steps = counts.steps + coarse.iter().map(|p| p.counts.steps).sum::<u64>();
    let start = exact(ensemble, &ensemble.kinetic(steps));
    let sampled = sampled(model, reading, tables, pilot, known);
    let measured = Measured { errors: &sampled.errors, output: &sampled.at, group: &sampled.from, booked: booking };
    let mut error = model_error(&sums(&measured.means(outputs, None, &estimates), unsampled), unsampled);
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
    for table in &booked {
        for e in error.iter_mut().skip(1) {
            *e = [probability_error(e[0] + table[0]), e[1] + table[1]];
        }
    }
    let optical = crate::ledger::optical(&mut Ledger::default(), counts.optical);
    for e in error.iter_mut().skip(1) {
        e[1] += optical;
    }
    PilotRead { estimates, counts, start, sampled, model: error, tables: booked, stuck, refined }
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
/// scout's where the scout did, or the pilot's trajectories at 4τ (`settle`). And what the paths of each of the rung's
/// two levels that left the model's domain ask of a reach (`departed`, `pair`; module documentation, # The reach's
/// atoms): where the companion's did, the rung is left, its read unread; and whether the rung above the rung in hand is
/// a step whose paths did (`above_departs`): no third level reads it.
#[derive(Default)]
struct Levels {
    fine: Vec<Part>,
    coarse: Vec<Part>,
    further: [Vec<Part>; 2],
    above: Option<Between>,
    above_pilot: bool,
    departed: [Option<Asks>; 2],
    above_departs: bool,
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
    fn told(&self, tau: f64, scales: &Scales) -> (Between, Option<Between>, Option<f64>, u64) {
        let (fine, coarse) = self.paired();
        let (own, pairs) = (between(&fine, &coarse, scales), fine.len() as u64 * GROUP);
        (own, self.above.clone(), self.above.as_ref().map(|_| 4. * tau), pairs)
    }
}

impl Levels {
    /// The paths of a pair's two levels that left the model's domain, joined to what the rung's levels held (`pair`).
    fn depart(&mut self, [fine, coarse]: [Option<Asks>; 2]) {
        self.departed = [Asks::join(self.departed[0].take(), fine), Asks::join(self.departed[1].take(), coarse)];
    }
}

/// What a pair's paths not proven ask of the reach (module documentation, # The reach's atoms): where the finer level's
/// left the model's domain and its companion's did not, the finer level's -- the run's own, which the coarser step did
/// not leave the domain at; else nothing yet. Where both left it, the rung is left and the finer level becomes the
/// companion of the rung below (`settle`): the departure is the reach's where the finer step confirms it, a step's error
/// where it does not.
fn own_departure(departed: &[Option<Asks>; 2]) -> Option<Asks> {
    departed[0].clone().filter(|_| departed[1].is_none())
}

/// The members `members` of the groups `numbers` at the step `tau`, each trajectory through the plan in the `domain`
/// (a companion's level takes `Domain::companion`): a hand-out of its own (`run_groups`); or, where a step of it was not
/// proven, what a reach is to hold for its paths.
#[allow(clippy::too_many_arguments)]
fn level(run: &Run, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Result<Vec<Part>, Asks> {
    let hand = Hand { groups: numbers, members, shape: Shape::of(plan, keep), whole: domain.result, record: &|i| trajectory(run, ensemble, domain, tau, plan, keep, i) };
    match run_groups(&[hand], messages, progress) {
        (mut parts, mut unproven) => match unproven.pop().flatten() {
            Some(asked) => Err(asked),
            None => Ok(parts.pop().expect("the hand-out's one level")),
        },
    }
}

/// The members `members` of the groups `numbers` at the step `tau` (their states kept where `keep`) and at twice it,
/// the two levels of a check (module documentation, # The step): one hand-out (`run_groups`) -- a check runs both
/// whatever either shows --, the finer level first, whose trajectories take twice the steps; the coarser one is the
/// companion's (`Domain::companion`). `progress` takes the finer level's groups. With the two levels, what a reach is to
/// hold for each level's paths not proven (`run_groups`): which of them asks it of the reach is `settle`'s (module
/// documentation, # The reach's atoms; `own_departure`).
#[allow(clippy::too_many_arguments)]
fn pair(run: &Run, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> ([Vec<Part>; 2], [Option<Asks>; 2]) {
    let companion = domain.companion();
    let fine = Hand { groups: numbers, members: members.clone(), shape: Shape::of(plan, keep), whole: true, record: &|i| trajectory(run, ensemble, domain, tau, plan, keep, i) };
    let coarse = Hand { groups: numbers, members, shape: Shape::of(plan, false), whole: false, record: &|i| trajectory(run, ensemble, &companion, 2. * tau, plan, false, i) };
    let (levels, unproven) = run_groups(&[fine, coarse], messages, progress);
    let (mut levels, mut unproven) = (levels.into_iter(), unproven.into_iter());
    (std::array::from_fn(|_| levels.next().expect("the hand-out's two levels")), std::array::from_fn(|_| unproven.next().expect("the hand-out's two levels")))
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
/// and a look more at each prefix.) The level, with what its paths
/// not proven ask of a reach; or, where the atoms in hand refute the reach, the temperature it grows to and those
/// atoms' groups, their states the next attempt's to decide on (`Attempt::Grow`).
#[allow(clippy::too_many_arguments)]
fn own_level(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &Context, scout: &[Part], numbers: &[u64], doubling: bool, grown: &dyn Fn(&[&Part]) -> Option<f64>, messages: &mut Messages) -> Result<(Vec<Part>, Option<Asks>), (f64, Vec<Part>)> {
    let (tau, plan) = (context.tau, &context.plan);
    let mut joined = scout.to_vec();
    let mut members = 1;
    loop {
        let more = if doubling { (2 * members).min(GROUP) } else { GROUP };
        let hand = Hand { groups: numbers, members: members..more, shape: Shape::of(plan, true), whole: true, record: &|i| trajectory(run, ensemble, domain, tau, plan, true, i) };
        let (mut rest, mut unproven) = run_groups(&[hand], messages, &mut |prefix, messages| so_far(context, &joined, prefix, messages));
        absorb(&mut joined, rest.pop().expect("the hand-out's one level"));
        members = more;
        if let Some(asked) = unproven.pop().flatten() {
            return Ok((joined, Some(asked)));
        }
        if let Some(t) = grown(&joined.iter().collect::<Vec<_>>()) {
            return Err((t, joined));
        }
        if members == GROUP {
            return Ok((joined, None));
        }
    }
}

/// The states kept in some groups' parts freed, on the pool, a group's by itself: a state is a block of memory of its
/// own, and a pilot keeps one for every trajectory at every output.
fn forget(parts: &mut [Part]) {
    parts.par_iter_mut().for_each(|part| part.samples = vec![]);
}

/// The states kept in some groups' parts freed (`forget`) but, at every output, the first that lies outside the reach:
/// what a group of the product keeps (`Shape::fold`).
fn outside_alone(parts: &mut [Part]) {
    parts.par_iter_mut().for_each(|part| part.samples.iter_mut().for_each(|states| *states = states.iter().filter(|state| state.outside).min_by_key(|state| state.index).cloned().into_iter().collect()));
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
/// cheap third level is to be had; or ended where the pilot's atoms refute the reach (`check`) --; clearly beyond its
/// share at the steps' cap; unable to see with no remedy left --
/// each of these two standing, what its reads do not exclude booked (`Levels::told`, `checked`), and flagged --; or not
/// at all: a step of its trajectories was not proven, and the attempt ends with what the reach is to hold for it
/// (`run_groups`).
#[derive(Clone, Debug)]
enum Settled {
    Standing,
    Capped,
    Unseen,
    Unproven(Asks),
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
/// - a path of its companion's level left the model's domain (not proven, `pair`; `Levels::departed`): the rung is
///   left as one beyond, its read unread -- the rung below has no third level above, and no third level reads the step
///   that departed (`Levels::above_departs`), nor one whose own path is not proven. Where the rung's own level's paths
///   left it too, that level is the rung below's companion: the reach grows for the rung below's own paths where they
///   leave it as well (`level`), else the rung below is left in turn (`own_departure`; module documentation, # The
///   reach's atoms); at the ladder's bottom for the rung's own paths, the finest there are;
/// - else, on the scout (which has the pilot to come), the rung stands. On the pilot (all the members of a group in:
///   the last word on the step), every estimate as it reads (`verdict`):
///   - its own estimate within and its third level's too: it stands; its third level's beyond (the order is not shown
///     above it): it is left;
///   - its own estimate within and no third level, where `above` says the rung above is a rung of the ladder and that
///     rung's paths did not leave the model's domain: the pilot's trajectories at 4τ, a quarter of a trajectory each,
///     are its third level; where they leave the domain or read no number (`readable`), the rung above is unread;
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
/// `ran`: the finer level of the members already run, joined to the groups the levels hold, with what its paths not
/// proven ask of a reach (`attempt`: the pilot's own level, run first and read by the reach's falsifier before its
/// companion, `own_level`) -- its companion alone is run then, and the check goes on as from a pair: a trajectory
/// is a function of its number and step (`trajectory`), not of the hand-out it was run in. `until`: on the pilot, the
/// check stops where the rung's own read is in, not clearly beyond, and it would go on past its two levels
/// (`Settled::Pending`; `check` takes it up again).
#[allow(clippy::too_many_arguments)]
fn settle(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: std::ops::Range<u64>, above: &dyn Fn(f64) -> bool, outgrown: &dyn Fn(&[&Part]) -> bool, ran: Option<(Vec<Part>, Option<Asks>)>, until: bool, messages: &mut Messages) -> Settled {
    // The finer level joined to the groups the levels hold, its companion, and what each level's paths not proven ask.
    let (joined, coarse, departed) = match ran {
        Some((joined, asked)) => {
            let companion = domain.companion();
            let (tau, plan) = (context.tau, &context.plan);
            let hand = Hand { groups: numbers, members: members.clone(), shape: Shape::of(plan, false), whole: false, record: &|i| trajectory(run, ensemble, &companion, 2. * tau, plan, false, i) };
            let (mut coarse, mut unproven) = run_groups(&[hand], messages, &mut |_, messages| so_far(context, &joined, &[], messages));
            (joined, coarse.pop().expect("the hand-out's one level"), [asked, unproven.pop().flatten()])
        }
        None => {
            let ([fine, coarse], departed) = pair(run, ensemble, domain, &context.plan, context.tau, numbers, members.clone(), true, messages, &mut |prefix, messages| so_far(context, &levels.fine, prefix, messages));
            let mut joined = std::mem::take(&mut levels.fine);
            absorb(&mut joined, fine);
            (joined, coarse, departed)
        }
    };
    if let Some(asked) = own_departure(&departed) {
        return Settled::Unproven(asked);
    }
    levels.fine = joined;
    absorb(&mut levels.coarse, coarse);
    levels.depart(departed);
    check(run, ensemble, domain, context, levels, numbers, members.end, above, outgrown, until, messages)
}

/// The step's check of the levels in hand (`settle`) on the members `..members` of the pilot's groups `numbers` (the
/// pilot's where they are all its members); `until` as `settle` takes it.
#[allow(clippy::too_many_arguments)]
fn check(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: u64, above: &dyn Fn(f64) -> bool, outgrown: &dyn Fn(&[&Part]) -> bool, until: bool, messages: &mut Messages) -> Settled {
    // Whether the check is the pilot's: all the members of a group in.
    let pilot = members == GROUP;
    loop {
        // The rung's own read; none where its companion departed (a departed level is not run to its end).
        let departs = levels.departed[1].is_some();
        let read = (!departs).then(|| {
            let (fine, coarse) = levels.paired();
            between(&fine, &coarse, &context.scales)
        });
        // The rung below, and whether the ladder has it.
        let below = 0.5 * context.tau;
        let lowest = context.plan.end / below > MOST_STEPS as f64;
        let upward = !levels.above_pilot && above(context.tau) && !levels.above_departs;
        let verdict = read.as_ref().map_or(Verdict::Left, |read| verdict(read, levels.above.as_ref(), pilot, upward));
        if verdict == Verdict::Stands {
            return Settled::Standing;
        }
        if until && pilot && read.as_ref().is_some_and(|read| !screen(read)) {
            return Settled::Pending;
        }
        if verdict == Verdict::Above {
            // The third level above on the pilot's trajectories: the rung above's coarser level, at 4τ, against its
            // finer one, this rung's companion. Where its paths leave the model's domain (the level ends at the first
            // that does), or its read excludes nothing on some look (`readable`: no output counts a trajectory at both
            // levels), the rung above is unread.
            match level(run, ensemble, &domain.companion(), &context.plan, 4. * context.tau, numbers, 0..GROUP, false, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages)) {
                Ok(coarser) => {
                    let read = between(&levels.coarse.iter().collect::<Vec<_>>(), &coarser.iter().collect::<Vec<_>>(), &context.scales);
                    if readable(&read) {
                        (levels.above, levels.above_pilot) = (Some(carried(&read, 0.5)), true);
                    } else {
                        levels.above_departs = true;
                    }
                }
                Err(_) => levels.above_departs = true,
            }
            continue;
        }
        let have = levels.paired().0.len() as u64 * GROUP;
        let pairs = if verdict == Verdict::Unseeing { read.as_ref().and_then(blind).and_then(|factor| remedy(factor, have)) } else { None };
        if let Some(pairs) = pairs {
            // The own read's estimate beyond the share, and not clearly: the noise hides the difference, and the pairs
            // could bring it within at the best (`blind`, `remedy`) -- a rung below reads its own pair's noise, and where
            // the pairs part early that is the two ensembles' own. Where pairs cannot, the estimate as it reads is beyond
            // the share: the rung is left, as an adaptive integrator leaves a step whose error estimate is beyond its
            // tolerance.
            let next: Vec<u64> = (have / GROUP..pairs / GROUP).collect();
            let (more, departed) = pair(run, ensemble, domain, &context.plan, context.tau, &next, 0..GROUP, false, messages, &mut |prefix, messages| {
                so_far(context, &levels.fine, &levels.further[0].iter().chain(prefix.iter().copied()).collect::<Vec<_>>(), messages)
            });
            if let Some(asked) = own_departure(&departed) {
                return Settled::Unproven(asked);
            }
            levels.depart(departed);
            levels.further.iter_mut().zip(more).for_each(|(further, more)| further.extend(more));
            continue;
        }
        // The rung in hand is left. At the ladder's bottom the paths of the rung's own level that left the model's
        // domain are the finest there are: the reach is to hold them; a read clearly beyond leaves the step capped, one
        // that only could not be seen through, unable to see.
        // Above the bottom the rung's own named units go down with it (`levels.departed` below): they are taken here
        // only at the bottom.
        if lowest {
            if let Some(asked) = levels.departed[0].take() {
                return Settled::Unproven(asked);
            }
            return if verdict == Verdict::Unseeing { Settled::Unseen } else { Settled::Capped };
        }
        // The rung below's third level above: this rung's own read, lowered -- none where its companion departed.
        let inherits = read.as_ref().map(|read| carried(read, 0.5));
        context.tau = below;
        let finer = match level(run, ensemble, domain, &context.plan, below, numbers, 0..members, true, messages, &mut |prefix, messages| so_far(context, &[], prefix, messages)) {
            Ok(finer) => finer,
            Err(asked) => return Settled::Unproven(asked),
        };
        // The level at the step is the companion from here on: its states are read no more.
        forget(&mut levels.fine);
        levels.coarse = std::mem::replace(&mut levels.fine, finer);
        levels.further = Default::default();
        (levels.above, levels.above_pilot) = (inherits, pilot);
        // The rung above the one taken is the step whose companion departed, where it did: no third level reads it.
        // The level at the step is the companion now, with what its own paths asked.
        levels.above_departs = departs;
        levels.departed = [None, levels.departed[0].take()];
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

/// How the product ends before its estimates stand (`product`): a step of a batch was not proven, with what the proof
/// set is to hold for it (`run_groups`); or a look its model's error leaves no room has room where units are kept,
/// those units (`keep`).
enum Ended {
    Unproven(Asks),
    Keep(Vec<usize>),
}

/// The product (module documentation, # Stopping): batches of whole groups at the run's step, on from the pilot's
/// groups (their numbers `pilot`; the product's own follow in the list's order), to the number the looks predict
/// (`batch`), until every look a run's trajectories can meet (`Look::fits`) is met, or the trajectories reach
/// `MOST_TRAJECTORIES`. Before every batch's looks the atoms outside the reach are read (`apart`, with `read` at a
/// state): what they carry joins the model's error the pilot fixed, in the looks and in the results. Where the units'
/// error the pilot booked leaves a look no room (`crowded`) at the product's statistics, at an output its first batch's
/// did not, the units the room needs there are asked for (`keeping`: the pilot's reads of the units, the product's
/// statistics), and where there are some the product ends with them -- what the atoms outside carry is the reach's,
/// read at the product's end (`attempt`; module documentation, # The reach's atoms); the trajectories are not doubled
/// after a look the run's `MOST_TRAJECTORIES` cannot meet. Each batch's
/// result is a checkpoint; returns the estimates, the listed atoms and the counts at its end -- or how it ended before
/// (`Ended`).
#[allow(clippy::too_many_arguments)]
fn product(run: &Run, ensemble: &Ensemble, domain: &Domain, context: &mut Context, parts: &mut Vec<Part>, pilot: &[u64], read: &(dyn Fn(&State) -> Booking + Sync), keeping: &dyn Fn(&[Estimate], &Fixed) -> Option<Vec<usize>>, messages: &mut Messages) -> Result<(Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts), Ended> {
    let (started, from) = (Instant::now(), parts.len() as u64 * GROUP);
    // The model's error as the pilot fixed it, and the outside states read so far.
    let (booked, mut known) = (context.fixed.model.clone(), HashMap::new());
    // The group at the list's place g.
    let number = |g: u64| pilot.get(g as usize).copied().unwrap_or(g);
    // The wall time (ms) the trajectories from `done` to `want` take at the product's pace so far; none before its own
    // first trajectories are in.
    let left = |done: u64, want: u64| (done > from).then(|| started.elapsed().as_secs_f64() * 1e3 / (done - from) as f64 * (want - done) as f64);
    // The outputs whose looks no run's trajectories fit and that a keep was asked for: from the product's first batch
    // on, which the pilot's keep has read.
    let mut asked: Option<Vec<usize>> = None;
    loop {
        let all: Vec<&Part> = parts.iter().collect();
        let n = all.len() as u64 * GROUP;
        let (e, l, c) = pooled(&all, &context.scales);
        context.outside = apart(&e, &all, &booked, read, &mut known);
        context.fixed.model = booked.iter().zip(&context.outside).map(|(model, more)| [probability_error(model[0] + more[0]), model[1] + more[1]]).collect();
        let looks = looks(&e, &context.fixed, &context.step);
        // The units' error, as the pilot booked it, leaving a look no room at the product's statistics: the units' to
        // answer. What the atoms outside the reach carry is the reach's (# The reach's atoms: read at the product's end).
        let units = Fixed { model: booked.clone(), ..context.fixed.clone() };
        let lost = crowded(&self::looks(&e, &units, &context.step));
        match &mut asked {
            None => asked = Some(lost),
            Some(asked) => {
                let new: Vec<usize> = lost.into_iter().filter(|k| !asked.contains(k)).collect();
                if !new.is_empty() {
                    asked.extend(new);
                    if let Some(units) = keeping(&e, &units) {
                        return Err(Ended::Keep(units));
                    }
                }
            }
        }
        let chased: Vec<Look> = looks.into_iter().filter(Look::fits).collect();
        if chased.iter().all(Look::met) || n >= MOST_TRAJECTORIES {
            return Ok((e, l, c));
        }
        let want = batch(chased.iter().map(Look::need).fold(0., f64::max), n);
        messages.result(all.len(), n, left(n, want), || assemble(context, &e, &l, &c, n));
        let numbers: Vec<u64> = (n / GROUP..want / GROUP).map(number).collect();
        let new = level(run, ensemble, domain, &context.plan, context.tau, &numbers, 0..GROUP, false, messages, &mut |prefix, messages| {
            let all: Vec<&Part> = parts.iter().chain(prefix.iter().copied()).collect();
            let done = all.len() as u64 * GROUP;
            let (e, l, c) = pooled(&all, &context.scales);
            messages.result(all.len(), done, left(done, want), || assemble(context, &e, &l, &c, done));
        })
        .map_err(Ended::Unproven)?;
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
    /// step (`begins`), a whole part of the run. Every step of
    /// the attempt -- a rung's, its companion's -- is a cell over a power of two, so every level reads the same path.
    /// The run until the first rung is known.
    cell: f64,
}
impl Plan {
    /// The plan of the request's times with the trap's frequency per lab axis (rad/µs).
    fn new(times: &[f64], frequencies: V3) -> Plan {
        let last = times.len() - 1;
        let halves: Vec<V3> = times.iter().map(|&t| frequencies.map(|w| half_window(w, t))).collect();
        let count = times.len().min(DISTRIBUTION_TIMES);
        let mut listed: Vec<usize> = (0..count).map(|i| ((i * last) as f64 / (count - 1) as f64).round() as usize).collect();
        listed.dedup();
        let end = times[last] + halves[last].iter().fold(0., |a: f64, &b| a.max(b));
        Plan { end, times: times.to_vec(), halves, listed, cell: end }
    }
}

/// Counts over trajectories: the steps taken -- a step with jumps is one step, whatever the pieces its path is handed
/// on by (adapt.rs, # The jumps) --, and the trajectories that ended with no outcome (estimate.rs, # What is counted),
/// by why: the scheme could not advance them (adapt.rs `advance`'s refusal), a step's optical sector made no K (step.rs
/// `Refused::Optical`, by its `Unmade`), or a step of theirs was not proven (`Domain`) -- in a result sent while their
/// attempt runs.
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    steps: u64,
    failed: Stopped,
    unmade: [Stopped; 3],
    unproven: Stopped,
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
        self.unproven.add(&o.unproven);
        for (sum, x) in self.optical.iter_mut().zip(o.optical) {
            *sum += x;
        }
    }
}

/// Trajectories that ended with no outcome for one reason (`Counts`): how many, and the earliest time one stopped at.
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

/// A trajectory's state at an output (model.rs `Sample`) with the run's step, the trajectory's number and whether the
/// atom lies outside the reach there, resolved (`Domain::outside`, estimate.rs `Held`): where the units' errors are
/// measured and a model is priced (`Pricing::rates`) -- a pilot trajectory's at every output; any trajectory's where its atom is
/// outside the reach (`apart`).
#[derive(Clone)]
struct State {
    sample: Sample,
    tau: f64,
    index: u64,
    outside: bool,
    /// The trajectory's jumps up to the state.
    jumps: u32,
}

/// What one trajectory leaves: what became of it by every output (estimate.rs `Fate`), its (r, κ) at the listed outputs
/// where held, its states at the outputs where held (`keep`: the pilot's, for the model's errors and prices; without
/// it those where its atom lies outside the reach), its counts; and where a step of it was not proven, what a reach is
/// to hold for that step (step.rs `Unproven`: within the trap's region and outside it).
struct Record {
    outputs: Vec<Fate>,
    listed: Vec<Option<(V3, V3)>>,
    samples: Vec<Option<State>>,
    counts: Counts,
    unproven: Option<crate::step::Unproven>,
    /// Its photons and their Σ k_φ² (step.rs `Trajectory::scattered`).
    scattered: [f64; 2],
}

/// One trajectory's T windows (module documentation, # Outputs and windows): per lab axis the running integrals from
/// t = 0 of the time held, κ and κ² (µs, 1/µm·µs, 1/µm²·µs), and per output and axis their values at its window's two
/// ends -- the window's time held, ∫κ dt and ∫κ² dt are the differences. An axis's windows' lower ends come in the
/// outputs' order, and so do their upper ends (a window is cut at t = 0 alone): per axis and side, the next output whose
/// end is to come. Output 0 is the instant t = 0 and has none.
struct Windows<'a> {
    plan: &'a Plan,
    running: [[f64; 3]; 3],
    ends: Vec<[[[f64; 3]; 2]; 3]>,
    next: [[usize; 2]; 3],
}
impl<'a> Windows<'a> {
    fn new(plan: &'a Plan) -> Windows<'a> {
        Windows { plan, running: [[0.; 3]; 3], ends: vec![[[[0.; 3]; 2]; 3]; plan.times.len()], next: [[1; 2]; 3] }
    }
    /// The lower (side 0) or upper (1) end of output k's window on axis a.
    fn end(&self, k: usize, a: usize, side: usize) -> f64 {
        let (t, h) = (self.plan.times[k], self.plan.halves[k][a]);
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
    /// Output k's window (k ≥ 1) per lab axis: its time held, ∫κ dt and ∫κ² dt.
    fn window(&self, k: usize) -> [[f64; 3]; 3] {
        let e = &self.ends[k];
        std::array::from_fn(|a| std::array::from_fn(|i| e[a][1][i] - e[a][0][i]))
    }
}

/// One trajectory from its initial state through the plan at the step τ (module documentation, # Outputs and windows):
/// advanced to the run's end by steps that know no output, and read at every output where it falls (adapt.rs, # The
/// reads). A trajectory the scheme cannot advance (adapt.rs) fails from the first output it does not reach.
fn run<D: Dynamics>(model: &D, domain: &Domain, start: Start, index: u64, tau: f64, plan: &Plan, keep: bool) -> Record {
    let outputs = plan.times.len();
    let mut t = Trajectory::new(model, start.state, start.r, start.kappa, SEED, index);
    let mut scratch = Scratch::default();
    // The trajectory's Brownian path in the plan's cells, its grid the multiples of τ (a cell over a power of two):
    // the same path at every rung and at its companion.
    let mut path = adapt::Path::new(SEED, index, plan.cell, (plan.cell / tau).log2().round() as u32);
    let mut windows = Windows::new(plan);
    let mut out = vec![Fate::Lost; outputs];
    let mut listed = vec![None; plan.listed.len()];
    let mut samples: Vec<Option<State>> = vec![None; outputs];
    // The steps taken, and the start of the last one counted.
    let (mut steps, mut counted) = (0, None);
    let mut record = |k: usize, t: &Trajectory, out: &mut Vec<Fate>| {
        // The atom at the output's instant; from the first output on its κ and κ² are its window's means (below). An
        // unresolved one is booked whole, its state not read (`apart`). Leaving or bound, at the instant.
        let [beyond, leaving] = domain.placed(t.r, t.kappa);
        let outside = !t.unresolved && beyond;
        out[k] = Fate::Held(Held { rows: model.readout(t.cluster, &t.point(), &t.psi), kappa: t.kappa, kappa2: t.kappa.map(|x| x * x), outside, unresolved: t.unresolved, leaving });
        if let Ok(j) = plan.listed.binary_search(&k) {
            listed[j] = Some((t.r, t.kappa));
        }
        if keep || outside {
            let sample = Sample { cluster: t.cluster, at: t.point(), psi: t.psi.to_vec() };
            samples[k] = Some(State { sample, tau, index, outside, jumps: t.jumps });
        }
    };
    record(0, &t, &mut out);
    // The first output the trajectory does not reach held; whether the scheme could not advance it, and what a reach is
    // to hold for a step its domain did not prove (step.rs `Unproven`): the trajectory ends there unfinished, and so
    // does the attempt (`run_groups`).
    let mut reached = 1;
    let mut unmade = None;
    let (failed, unproven) = {
        let mut read = |k: usize, t: &Trajectory| {
            record(k + 1, t, &mut out);
            reached = k + 2;
        };
        // A step's path as far as the trajectory is on it, into T's windows: piece by piece, from the step's start and
        // from each of its jumps -- the step is counted at the first of them.
        let mut each = |s: &Segment, until: f64| {
            if counted != Some(s.step.0) {
                (steps, counted) = (steps + 1, Some(s.step.0));
            }
            windows.add(s.start, until, &|from, to| s.moments(from, to));
        };
        // Through the last output's window; a failure there leaves every output read.
        match adapt::advance(model, domain, tau, &mut t, &mut scratch, &mut path, plan.end, &plan.times[1..], &mut read, &mut each) {
            Ok(()) => (false, None),
            Err(Refused::Unproven(step)) => (true, Some(step)),
            Err(Refused::Optical(why)) => {
                unmade = Some(why);
                (true, None)
            }
            Err(_) => (true, None),
        }
    };
    // From the output it does not reach: failed, or lost.
    if reached < outputs {
        out[reached..].fill(if failed { Fate::Failed } else { Fate::Lost });
    }
    windows.close();
    for (k, o) in out.iter_mut().enumerate().skip(1) {
        if let Fate::Held(h) = o {
            let w = windows.window(k);
            // A held atom was held over its window's part before the output, which is no empty stretch after t = 0
            // (estimate.rs `half_window`); a window of no length is the instant's own κ and κ².
            for a in (0..3).filter(|&a| w[a][0] > 0.) {
                (h.kappa[a], h.kappa2[a]) = (w[a][1] / w[a][0], w[a][2] / w[a][0]);
            }
        }
    }
    let stopped = |why: bool| if why { Stopped::at(t.time) } else { Stopped::default() };
    let optical = [t.booked[0], t.booked[1], t.booked[2], -(t.pending[0][0] + t.pending[1][1] + t.pending[2][2])];
    let counts = Counts {
        steps,
        failed: stopped(failed && unproven.is_none() && unmade.is_none()),
        unmade: [Unmade::Schur, Unmade::Singular, Unmade::Unfinite].map(|why| stopped(unmade == Some(why))),
        unproven: stopped(unproven.is_some()),
        optical,
    };
    Record { outputs: out, listed, samples, counts, unproven, scattered: t.scattered }
}



/// What a group of trajectories leaves: its sums (estimate.rs `Group`), its held atoms' (r, κ) at the listed outputs, its
/// states at the outputs (the pilot's: every member's; else the first that lies outside the reach), its counts.
#[derive(Clone)]
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
            samples: vec![vec![]; self.outputs],
            counts: Counts::default(),
        };
        for r in records {
            part.sums.add(&r.outputs);
            part.sums.scatter(r.scattered);
            for (l, x) in part.listed.iter_mut().zip(r.listed) {
                l.extend(x);
            }
            for (s, x) in part.samples.iter_mut().zip(r.samples) {
                if self.keep || s.is_empty() {
                    s.extend(x);
                }
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
/// 0..GROUP), each trajectory made by `record` from its number, a group's records folded as `shape` says; `whole`:
/// whether every path of it not proven is asked for -- the run's own level's, which a reach is to hold --, else the
/// level ends at its first (a companion's, the step's probe: that it left the model's domain is all it says).
struct Hand<'a> {
    groups: &'a [u64],
    members: std::ops::Range<u64>,
    shape: Shape,
    whole: bool,
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
///
/// A trajectory a step of which its domain did not prove ends there (`Domain`; module documentation, # The reach's
/// atoms): the attempt's model is no model of that step. A level `whole` goes on to its end, every trajectory to its own
/// end or its first step not proven, and tells what the proof set is to hold for all of that level's paths not proven
/// (`Asks::join`, in the list's order): a function of the list, not of the threads, and one set that holds every such
/// path of the level. (Held to the first path not proven, the reach of the default tweezer with a resonant probe grew
/// five times over 5 ms, each attempt proving the region anew; to all of them, three.) A companion's level (not
/// `whole`) says only that it left the domain -- its step's error, the rung left unread (`settle`) --: it ends at its
/// first path not proven, its trajectories not yet begun are not run, and it returns no groups. Which level's paths a
/// reach is to hold is the caller's (`level`, `pair`): a companion's are its step's, not the atoms'. (S3c's third level
/// at 4τ = 2.865 µs ran all its 88 trajectories to their first step not proven, 43 of them, at 4.3 ms of proofs a
/// step: 68.7 CPU-s for a level that was then unread.)
fn run_groups(levels: &[Hand], messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> (Vec<Vec<Part>>, Vec<Option<Asks>>) {
    // Where each level's trajectories begin in the list, and the list's length.
    let starts: Vec<u64> = levels.iter().scan(0, |next, level| Some(std::mem::replace(next, *next + level.count()))).collect();
    let stop: u64 = levels.iter().map(Hand::count).sum();
    // The list's k-th trajectory: its level and its place there.
    let place = |k: u64| {
        let level = starts.partition_point(|&start| start <= k) - 1;
        (level, k - starts[level])
    };
    let next = AtomicU64::new(0);
    // The levels that ended at a path not proven (not `whole`).
    let ended: Vec<std::sync::atomic::AtomicBool> = levels.iter().map(|_| std::sync::atomic::AtomicBool::new(false)).collect();
    // What each level's trajectories' steps not proven ask of the proof set, by their places in the level.
    let mut unproven: Vec<Vec<Option<crate::step::Unproven>>> = levels.iter().map(|level| vec![None; level.count() as usize]).collect();
    let threads = rayon::current_num_threads().min(stop as usize);
    // Per level and group the members' records as they come in, and how many are in (a member not run counts in).
    let mut waiting: Vec<Vec<(Vec<Option<Record>>, u64)>> = levels.iter().map(|level| (0..level.groups.len()).map(|_| ((0..level.size()).map(|_| None).collect(), 0)).collect()).collect();
    let mut parts: Vec<Vec<Option<Part>>> = levels.iter().map(|level| (0..level.groups.len()).map(|_| None).collect()).collect();
    // This thread stays here for the records and the messages; the pool's threads run the trajectories.
    rayon::in_place_scope(|scope| {
        let (sender, receiver) = mpsc::channel::<(usize, u64, Option<Record>)>();
        for _ in 0..threads {
            let (sender, next, place, ended) = (sender.clone(), &next, &place, &ended);
            scope.spawn(move |_| loop {
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= stop {
                    break;
                }
                let (level, k) = place(k);
                let record = (!ended[level].load(Ordering::Relaxed)).then(|| (levels[level].record)(levels[level].number(k)));
                if !levels[level].whole && record.as_ref().is_some_and(|r| r.unproven.is_some()) {
                    ended[level].store(true, Ordering::Relaxed);
                }
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
                    if let Some(mut r) = r {
                        unproven[level][k as usize] = r.unproven.take();
                        members[m] = Some(r);
                    }
                    *count += 1;
                    if *count == size {
                        // A group with a member not run is of a level that ended: none of that level's is returned.
                        if members.iter().all(Option::is_some) {
                            parts[level][g] = Some(levels[level].shape.fold(members.iter_mut().map(|r| r.take().expect("a member's record"))));
                        }
                        left -= 1;
                        asked = false;
                    }
                }
                Err(mpsc::RecvTimeoutError::Timeout) => {}
                Err(mpsc::RecvTimeoutError::Disconnected) => break,
            }
        }
    });
    let asks = unproven.iter().map(|level| level.iter().flatten().fold(None, |asked, step| Asks::join(asked, Some(Asks::of(step))))).collect();
    let parts = parts.into_iter().zip(&ended).map(|(level, ended)| if ended.load(Ordering::Relaxed) { vec![] } else { level.into_iter().map(|p| p.expect("a group's trajectories ended without their part")).collect() }).collect();
    (parts, asks)
}

/// What the pilot fixes for the run (module documentation, # Stopping), per output: the model's error on the held atoms
/// (populations absolute, T relative; T's unbounded where no pilot atom reads it) -- to which the product adds what
/// its atoms outside the reach carry (`Context::outside`) --; and T(0)'s relative error, its quadrature's last two
/// levels' difference (ensemble.rs `Ensemble::kinetic`).
#[derive(Clone)]
struct Fixed {
    model: Vec<[f64; 2]>,
    start: f64,
}

/// The step's share of the targets (SPEC §2.2, §4): on what is shown s·ε, and on T s itself -- of T's allowance at each
/// output, which T's reads and bounds are taken over (# T's allowance; estimate.rs `between`): what its error may take
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

/// One quantity held to its target at one output (module documentation, # Stopping): what is shown, or the total T.
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
    /// Whether its total lies within its target: the ledger's own test of an output (ledger.rs `Ledger::close`).
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
/// part: at every output after t = 0 what is shown, and the total T (the mean over the three lab axes: what T's target
/// is on). A T whose model's or step's part is unread is not met, and no number of trajectories meets it. Where it is
/// its statistics that are unread (too few groups hold an atom), they are predicted from the normal theory of one
/// axis's variance over the N h atoms held, the relative interval z √(2/(N h)) -- the total's is at most that (three
/// axes' mean; equal where one axis holds all the energy) --, weighed by the held share as a read interval is: one
/// number for whether the look is met and for the trajectories it needs (`Look::met`, `Look::need`). (Unread, it was
/// booked unbounded for the one and predicted for the other: S2Q's outputs at 2.4--2.6 ms, about one atom held of
/// three thousand, were never met and asked for one to three trajectories, and its product grew by a group a batch,
/// 293 batches, from 568 trajectories to 2912.) Where no atom is held there is no T, and no look of it: N h = 0 would
/// ask that interval of nothing. The outputs' looks are made on the pool, each output's by itself, and come in the
/// outputs' order.
fn looks(estimates: &[Estimate], fixed: &Fixed, step: &[Bound]) -> Vec<Look> {
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
                let account = temperature[k].unwrap_or_else(|| temperature_account(e, fixed.model[k][1], &step[k], read.unwrap_or(CONFIDENCE * (2. / (n * h)).sqrt())));
                Look { output: k, target: e.temperature_target, account, interval: account.statistics, trajectories: n, shown: false }
            });
            std::iter::once(shown).chain(held)
        })
        .collect()
}

/// The outputs where some look's model error and step's share leave the statistics no room (SPEC §5.2): no number
/// of trajectories fits beside them -- what the model's decisions answer (`keep`), on the errors booked, not on a
/// statistics' read: a keep is never undone, and a need read off a few groups' interval is no certain thing (a look
/// that a run cannot meet within `MOST_TRAJECTORIES` is not chased, `Look::fits`, and is read again at every batch).
fn crowded(looks: &[Look]) -> Vec<usize> {
    let mut outputs: Vec<usize> = looks.iter().filter(|l| !(l.room() > 0.)).map(|l| l.output).collect();
    outputs.dedup();
    outputs
}

/// The outputs whose looks leave the statistics no room (`crowded`) at the model's errors `fixed` beside the step's
/// share, at the `estimates` (every trajectory counted), among those `read` where the held share `held` is some: where
/// a unit can be told to weigh (`weights`) -- an output with no state read carries every unit's bound alike, and tells
/// none of them apart.
fn tight(estimates: &[Estimate], fixed: &Fixed, held: &[f64], read: &[bool]) -> Vec<usize> {
    crowded(&looks(estimates, fixed, &vec![Bound::share(); fixed.model.len()])).into_iter().filter(|&k| held[k] > 0. && read[k]).collect()
}

/// Of the units' errors `means` (per output and unit), per unit its weight at the `outputs` (`tight`): its errors on
/// what is shown and on T, weighed by the held share `held` as the totals are, each over its target there -- T's its
/// allowance at the estimates' T (# T's allowance) --, added up.
fn weights(means: &[Vec<[f64; 2]>], estimates: &[Estimate], held: &[f64], outputs: &[usize]) -> Vec<f64> {
    (0..means.first().map_or(0, Vec::len)).map(|u| outputs.iter().map(|&k| held[k] * (means[k][u][0] / POPULATION_TARGET + means[k][u][1] / estimates[k].temperature_target)).sum()).collect()
}

/// The units that weigh where the looks have no room (`weights`), the candidates for the set they need (`needed`).
fn weighing(estimates: &[Estimate], fixed: &Fixed, means: &[Vec<[f64; 2]>], held: &[f64], read: &[bool]) -> Vec<usize> {
    weights(means, estimates, held, &tight(estimates, fixed, held, read)).iter().enumerate().filter(|(_, w)| **w > 0.).map(|(u, _)| u).collect()
}

/// The doubt the reads of a few groups leave on the model's error per output (`keep`), on what is shown and on T: the
/// model's error as every group but one gives it -- the units' as `measured` at their states and what the reach leaves
/// out, `unsampled` (`model_error`), with the estimates of the others (their populations a state's booking is taken
/// at, `Booking::at`) -- whose jackknife over the groups is the interval the reads leave on the error, at the family of
/// the outputs' two reads (`interval`). Its near end is the error less it (`clear`): what the reads leave no doubt of,
/// as a rung is left only on that (# The step). Its far end is the error and it (`raised`): what the reads leave no
/// doubt is not exceeded -- of the error a set of units leaves, the set's errors taken out of every group's
/// (`interval` of the set). The scout reads eleven states an output, one a group, and a unit's error is heavy-tailed
/// over the states (it is largest at the atoms that are hottest): S3's scout read its carrier's dressing as filling the
/// room on eleven states, where its pilot's eighty-eight booked it at 2.7 % beside 3.7 % in all.
struct Doubt<'a> {
    measured: &'a Measured<'a>,
    /// Per group left out, the others' estimates and their model's error per output.
    left: Vec<(Vec<Estimate>, Vec<[f64; 2]>)>,
    outputs: usize,
}
impl<'a> Doubt<'a> {
    fn new(measured: &'a Measured<'a>, outputs: usize, parts: &[&Part], scales: &Scales, unsampled: [f64; 2]) -> Doubt<'a> {
        // Group by group: a group's units' means are every output's every unit's, and they are summed at once.
        let left = (0..parts.len())
            .map(|g| {
                let rest: Vec<&Part> = parts.iter().enumerate().filter(|(h, _)| *h != g).map(|(_, p)| *p).collect();
                let estimates = pooled(&rest, scales).0;
                let error = model_error(&sums(&measured.means(outputs, Some(g), &estimates), unsampled), unsampled);
                (estimates, error)
            })
            .collect();
        Doubt { measured, left, outputs }
    }

    /// Per output the jackknife interval over the groups, at the family of the outputs' two reads, of the model's error
    /// less the units `set`'s (none: of the whole error); none where no group reads it (a spread that is no number).
    fn interval(&self, set: &[usize]) -> Vec<[f64; 2]> {
        let rest = self.replicates(set);
        let looks = 2. * self.outputs.saturating_sub(1) as f64;
        (0..self.outputs).map(|k| std::array::from_fn(|i| spread(rest.iter().map(|r| r[k][i]), self.left.len() as f64).interval_over(looks))).collect()
    }

    /// Per group left out, per output the model's error less the units `set`'s (`remainder`): the jackknife's
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
    /// look among the `estimates` (`tight`: where an atom is held and a state read) without room, but at the outputs
    /// `beyond` (`needed`). The estimate's error and the next pilot's own are independent draws of the reads' spread,
    /// whose jackknife gives its draws over all the outputs together: the deviations δ_g = √(G − 1)(r_g − r̄) of the
    /// replicates, of mean square the jackknife's variance; so the next read is the point less one draw and plus
    /// another, Â − δ_g + δ_h, over the G² pairs, and the chance is the share of them that leave a look no room. An
    /// output whose replicates are no number (no group's states read it) has no draw.
    fn chance(&self, set: &[usize], point: &Fixed, estimates: &[Estimate], held: &[f64], read: &[bool], beyond: &[usize]) -> f64 {
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
                tight(estimates, &Fixed { model, ..point.clone() }, held, read).iter().any(|k| !beyond.contains(k))
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

/// The model's errors `fixed` per output at their near ends (`Doubt`, `lowered`): what the reads leave no doubt of.
fn clear(fixed: &Fixed, doubt: &[[f64; 2]]) -> Fixed {
    Fixed { model: fixed.model.iter().zip(doubt).map(|(m, d)| [lowered(m[0], d[0]), lowered(m[1], d[1])]).collect(), ..fixed.clone() }
}

/// The model's errors `fixed` per output at their far ends (`Doubt`): each with the doubt its reads leave on it, a
/// population's at most one; where the doubt is no number (no group reads it), as it is.
fn raised(fixed: &Fixed, doubt: &[[f64; 2]]) -> Fixed {
    let far = |error: f64, doubt: f64| if doubt.is_nan() { error } else { error + doubt };
    Fixed { model: fixed.model.iter().zip(doubt).map(|(m, d)| [probability_error(far(m[0], d[0])), far(m[1], d[1])]).collect(), ..fixed.clone() }
}

/// An output's model's error `error` less the errors of some units there, `less`, no less than nothing, a population's
/// at most one (`without`, `Doubt::interval`); no number where a unit's error there is unbounded -- the states do not
/// read the output (`Measured::means`), and its error is its bound, which no unit's removal lowers.
fn remainder<'a>(error: [f64; 2], less: impl Iterator<Item = &'a [f64; 2]>) -> [f64; 2] {
    let less = less.fold([0.; 2], |a, m| [a[0] + m[0], a[1] + m[1]]);
    if less[1].is_infinite() {
        return [f64::NAN; 2];
    }
    [probability_error((error[0] - less[0]).max(0.)), (error[1] - less[1]).max(0.)]
}

/// The units the looks need kept (module documentation, # The model; `keep`). The order: while a look leaves the
/// statistics no room at the far end of the model's error the units taken so far leave (`far`: the error and the doubt
/// its reads leave on it, `Doubt`), the unit whose weight there is the most for the work a µs of its kept model
/// (`cost`: its mean rate; none where its keep is refused, never taken) is taken next, until every such look has room
/// or no unit weighs there: the error that leaves no room is removed where the run it leaves costs the least. (By
/// weight alone, S3's probe took its carrier's dressing with thirteen bare drives on its scout -- the generator's
/// assembly, the jumps and the photons three to four times dearer a step --, where the bare drives alone leave room.
/// By the work a keep adds, the difference of two rates, PGC3D took 129 units whose keeps cost nothing at the step it
/// stood at, their weights a trifle, before any that the room needed, and its model had no step left: a keep's work is
/// the kept model's, at any step.)
///
/// The set: of the order's leading sets, the one of the least expected work (`price`: per set the work a µs of its
/// kept model, the product's trajectories beside the error it leaves, and the chance that the next pilot reads a look
/// without room, `Doubt::chance`), and of it none to spare. Kept, a set's model runs a pilot again (`again`) and its
/// product; where the next pilot reads a look without room it keeps again, of the larger sets the one of the least
/// expected work -- the largest, another pilot and its product at its own work. A cost against a cost: the error read
/// and its far end are the two ends of it. (Taken until the error read had room, e038's pilots kept 6, 2, 1 and 1 units
/// over four attempts, a pilot and its reads each, 440 CPU-s, the next pilot reading the outputs left with no room as
/// often as not; taken to the far end of the error, R4_LIN_2's pilot kept six units where one did, and the next pilot
/// three more.) Outputs that no set has room at -- every unit that weighs there kept -- are no set's to cover (the
/// ledger flags them).
///
/// A kept set's rate (`work`: its work formula, `Pricing::rates`) is a kept model's structure and its plans, made where a
/// set asks for it, and once. `least` bounds every set's rate from below (`Pricing::least`), so a unit's weight over it
/// bounds its value from above: the units are asked in the order of their weights, and no further once the best value
/// made is no less than the next one's bound -- the unit taken is the one of the most value, as were every rate made.
/// A set's expected work is its rate times what it runs, and a set that the bound leaves above the least found -- of
/// the larger sets, or the set kept -- is never the least: its rate is not made.
#[allow(clippy::too_many_arguments)]
fn needed(estimates: &[Estimate], means: &[Vec<[f64; 2]>], held: &[f64], read: &[bool], least: f64, work: &dyn Fn(&[usize]) -> Option<f64>, far: &dyn Fn(&[usize]) -> Fixed, price: &dyn Fn(&[usize], &[usize]) -> [f64; 2], again: f64) -> Vec<usize> {
    let tight_at = |set: &[usize]| tight(estimates, &far(set), held, read);
    // The outputs no set has room at: those tight with every unit kept that weighs where the far end is tight.
    let every: Vec<usize> = weights(means, estimates, held, &tight_at(&[])).iter().enumerate().filter(|&(_, w)| *w > 0.).map(|(u, _)| u).collect();
    let beyond = tight_at(&every);
    let open = |set: &[usize]| -> Vec<usize> { tight_at(set).into_iter().filter(|k| !beyond.contains(k)).collect() };
    let mut taken: Vec<usize> = vec![];
    let mut made: HashMap<usize, Option<f64>> = HashMap::new();
    loop {
        let outputs = open(&taken);
        if outputs.is_empty() {
            break;
        }
        let weights = weights(means, estimates, held, &outputs);
        let weighing: Vec<usize> = (0..weights.len()).filter(|&u| weights[u] > 0. && !taken.contains(&u)).collect();
        // Error per work a µs (a model's rate is some work: every step costs some), bounded by the bound's.
        let mut asked: Vec<(usize, f64)> = weighing.iter().map(|&u| (u, weights[u] / least)).collect();
        asked.sort_by(|a, b| b.1.total_cmp(&a.1).then(b.0.cmp(&a.0)));
        let mut best: Option<(usize, f64)> = None;
        for (u, most) in asked {
            if best.is_some_and(|(_, value)| value > most) {
                break;
            }
            if let Some(c) = *made.entry(u).or_insert_with(|| work(&[u])) {
                let value = weights[u] / c;
                // Of equal values the last unit, as a scan of them all in their order takes it.
                if best.is_none_or(|(b, v)| value > v || (value == v && u > b)) {
                    best = Some((u, value));
                }
            }
        }
        let Some((u, _)) = best else { break };
        taken.push(u);
    }
    least_expected(&taken, &|u| made[&u].unwrap_or(0.), least, work, &|set| price(set, &beyond), again)
}

/// Of the sets the order `taken` leads through, the one of the least expected work, and of it none to spare (`needed`):
/// a set kept, its model runs a pilot again and then its product, of the work a µs `w` of its kept model (`work`, none
/// where its keep is refused; `least` below every set's), the trajectories `again` and `n` the looks need beside the
/// error it leaves; and where the next pilot still reads a look without room (its chance `p`; `price` gives [n, p]),
/// the next keep is chosen again among the larger sets, the least of theirs -- for the largest, another pilot and its
/// product at its own work. E_j = w_j (again + (1 − p_j) n_j) + p_j min_{i > j} E_i. Then each unit is left out, the
/// dearest first (`single`: its own kept model's work a µs), where the set without it is expected to cost less.
fn least_expected(taken: &[usize], single: &dyn Fn(usize) -> f64, least: f64, work: &dyn Fn(&[usize]) -> Option<f64>, price: &dyn Fn(&[usize]) -> [f64; 2], again: f64) -> Vec<usize> {
    // E is w a + b: where the bound below w leaves it above `against`, the least of the sets it is weighed against, the
    // set is not the least, and its rate is not made (none).
    let expect = |set: &[usize], next: Option<f64>, against: f64| -> Option<f64> {
        let [n, p] = price(set);
        let (a, b) = match next {
            Some(next) => (again + (1. - p) * n, p * next),
            None => (again * (1. + p) + n, 0.),
        };
        if least * a + b > against {
            return None;
        }
        Some(work(set)? * a + b)
    };
    let mut expected = vec![f64::INFINITY; taken.len()];
    let mut after = f64::INFINITY;
    for j in (0..taken.len()).rev() {
        if let Some(e) = expect(&taken[..=j], after.is_finite().then_some(after), after) {
            expected[j] = e;
        }
        after = after.min(expected[j]);
    }
    let Some(best) = (0..taken.len()).filter(|&j| expected[j].is_finite()).min_by(|&a, &b| expected[a].total_cmp(&expected[b])) else { return vec![] };
    let (mut chosen, mut least) = (taken[..=best].to_vec(), expected[best]);
    // The set is the room's together: a unit taken early for its value may be needed no more once those after it are
    // kept. Each unit is left out, the dearest first, where the set without it is expected to cost less -- its next
    // pilot, where it reads a look without room, keeping again what the set holds.
    let mut dearest = chosen.clone();
    dearest.sort_by(|&a, &b| single(b).total_cmp(&single(a)).then(b.cmp(&a)));
    for u in dearest {
        let rest: Vec<usize> = chosen.iter().copied().filter(|&v| v != u).collect();
        if rest.is_empty() {
            continue;
        }
        if let Some(value) = expect(&rest, Some(least), least).filter(|&value| value < least) {
            (chosen, least) = (rest, value);
        }
    }
    chosen
}

/// The model's errors `fixed` less those of the units `set` (`means`) at the outputs read -- a read output's errors are
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
    Fixed { model, ..fixed.clone() }
}

/// Each output's accounts of what is shown and of the total T (ledger.rs `Account`) by the run's estimates, the model's
/// errors on the held atoms per output (`Fixed`) and the step's part, T's where read: the held share 1 − L weighs the
/// model's error on the rows as shown, and T's whole accounts (module documentation, # Stopping); t = 0 is the
/// request's state, without error, and the ensemble's moments, T's by quadrature: its accounts are the quadrature's
/// error alone (`Fixed::start`), in the discretization's part (SPEC §4). The share of the output's trajectories not
/// counted is booked whole on what is shown, in the model's part (estimate.rs, # What is counted: what no number of
/// trajectories takes away), and leaves T no account (`temperature_model`). The model's error on what is shown is at
/// most 1, a probability's. T has an account where its statistics, the model's error and its step error are all
/// read. The outputs' accounts are made on the pool, each output's by itself.
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
            let population = Account { model: probability_error(a[0] * h + e.uncounted), step: b.shown, statistics: e.shown_statistics() };
            let temperature = temperature_account(e, a[1], b, e.temperature_interval());
            (population, (temperature.model.is_finite() && temperature.step.is_finite() && temperature.statistics.is_finite()).then_some(temperature))
        })
        .unzip()
}

/// T's account at an output (`accounts`) from its estimate, the units' error on T there `units`, the step's part `bound`
/// (a share of T's allowance) and the statistics' relative interval `interval`: the model's part (`temperature_model`)
/// and the statistics' weighed by the held share, the step's read so already (estimate.rs `between`), times the
/// allowance.
fn temperature_account(e: &Estimate, units: f64, bound: &Bound, interval: f64) -> Account {
    let h = 1. - e.loss;
    Account { model: weighed(h, temperature_model(e, units)), step: bound.temperature * e.temperature_target, statistics: weighed(h, interval) }
}

/// The model's part of T's account at an output whose units' error on T is `units` (relative): theirs where every
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

/// What the pilot measured of the model's units (SPEC §5.2): the errors of the units read at the states (`Sampled`: the
/// states' rows one after another, a row the read units' errors on the populations and on T in their order), each state's
/// output and group, and how each folded unit is booked (`booked`).
struct Measured<'a> {
    errors: &'a [Booking],
    output: &'a [usize],
    group: &'a [usize],
    booked: &'a [Booked],
}
impl Measured<'_> {
    /// Per output and unit the unit's error on the held atoms (on their populations, absolute; on T, relative). A unit
    /// read: the mean over the states of the pilot's groups but `left`; where they hold no atom nothing is measured, and
    /// the error is its bound there, 1 on a population and none on T (the run's atoms at such an output are at most the
    /// share the pilot left unseen, and T has no account without a pilot's T: its look fits no number of trajectories
    /// and is not chased). A unit booked at its gate's bound: that bound, at every output. A state's booking is taken at
    /// the populations `estimates` give there (`Booking::at`: those of the groups the mean is over). The outputs are
    /// made on the pool, each by itself: an output's sums take its states in the states' order.
    fn means(&self, outputs: usize, left: Option<usize>, estimates: &[Estimate]) -> Vec<Vec<[f64; 2]>> {
        self.means_of(outputs, left, estimates, &(0..self.booked.len()).collect::<Vec<_>>())
    }

    /// `means` of the units `units` alone, in their order.
    fn means_of(&self, outputs: usize, left: Option<usize>, estimates: &[Estimate], units: &[usize]) -> Vec<Vec<[f64; 2]>> {
        let read = self.booked.iter().filter(|b| matches!(b, Booked::Read(_))).count();
        // Each output's states, in their order.
        let mut states: Vec<Vec<usize>> = vec![vec![]; outputs];
        for (state, &k) in self.output.iter().enumerate() {
            states[k].push(state);
        }
        // Each unit's place among the units read, where it is read.
        let places: Vec<Option<usize>> = units.iter().map(|&u| if let Booked::Read(place) = self.booked[u] { Some(place) } else { None }).collect();
        states
            .par_iter()
            .enumerate()
            .map(|(k, states)| {
                let (mut sum, mut count) = (vec![[0.; 2]; units.len()], 0.);
                for &state in states.iter().filter(|&&state| Some(self.group[state]) != left) {
                    let row = &self.errors[state * read..][..read];
                    for (s, place) in sum.iter_mut().zip(&places) {
                        if let Some(place) = *place {
                            let e = row[place].at(&estimates[k].populations);
                            s[0] += e[0];
                            s[1] += e[1];
                        }
                    }
                    count += 1.;
                }
                units
                    .iter()
                    .zip(&sum)
                    .map(|(&u, s)| match self.booked[u] {
                        Booked::Bound(bound) => [bound; 2],
                        Booked::Read(_) if count > 0. => [s[0] / count, s[1] / count],
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
    [probability_error(less(0)), less(1)]
}

/// The model's error per output from the sums `whole` of its units' (`model_error_at`): none at t = 0, the request's
/// state.
fn model_error(whole: &[[f64; 2]], unsampled: [f64; 2]) -> Vec<[f64; 2]> {
    (0..whole.len()).map(|k| if k == 0 { [0.; 2] } else { model_error_at(whole[k], None, unsampled) }).collect()
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

/// A state a mean over the run is read at (`one_each`): a pilot trajectory's at one output, with its group's place
/// among the pilot's and its weight in the mean, the outputs after t = 0 its trajectory holds its atom at.
struct Pick<'a> {
    state: &'a State,
    group: usize,
    weight: f64,
    /// The trajectory's jumps per µs over the outputs it holds its atom at: its last held state's jumps over that
    /// state's time.
    jumps: f64,
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
            if let (Some(&state), Some(last)) = (held.get((through * held.len() as f64) as usize), held.last()) {
                picks.push(Pick { state, group, weight: held.len() as f64, jumps: f64::from(last.jumps) / last.sample.at.t });
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
/// errors over them (per output and unit), and the current model's W there; every model's error taken less the doubt
/// the reads leave on it (`doubt`: the scout's; none on the pilot).
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
    /// Per output the doubt on the model's error (`lowered`).
    doubt: Vec<[f64; 2]>,
}
impl<'a> Level<'a> {
    /// The level of the groups but `left` from their estimates and the units' mean errors over them, beside the errors
    /// `fixed` and the step's share, the model's errors less the `doubt` per output.
    fn new(left: Option<usize>, estimates: std::borrow::Cow<'a, [Estimate]>, means: std::borrow::Cow<'a, [Vec<[f64; 2]>]>, fixed: &Fixed, unsampled: [f64; 2], doubt: &[[f64; 2]]) -> Level<'a> {
        let whole = sums(&means, unsampled);
        let fixed = clear(&Fixed { model: model_error(&whole, unsampled), ..fixed.clone() }, doubt);
        let mut needs = vec![0.0_f64; whole.len()];
        for look in looks(&estimates, &fixed, &vec![Bound::share(); whole.len()]) {
            if look.fits() {
                needs[look.output] = needs[look.output].max(look.need());
            }
        }
        let mut order: Vec<usize> = (1..whole.len()).collect();
        order.sort_by(|&a, &b| needs[b].total_cmp(&needs[a]));
        let need = needs.iter().copied().fold(0., f64::max);
        Level { left, estimates, means, whole, needs, order, need, unsampled, start: fixed.start, doubt: doubt.to_vec() }
    }

    /// The trajectories the looks need beside the step's share and a model's error `model` (per output, less the level's
    /// doubt) that is nowhere more than the level's own -- a unit's removed, every unit's. A look's need falls with the
    /// model's error, so an output needs no more than it does at the level's own (`needs`): the outputs are asked by that
    /// need, the largest first, and no further once the next one's is no more than what those before it need now. The
    /// largest over every output, for one or two outputs' looks.
    fn need_beside(&self, model: &dyn Fn(usize) -> [f64; 2]) -> f64 {
        let mut most = 0.;
        for &k in &self.order {
            if self.needs[k] <= most {
                break;
            }
            let (m, d) = (model(k), self.doubt[k]);
            let fixed = Fixed { model: vec![[0.; 2], [lowered(m[0], d[0]), lowered(m[1], d[1])]], start: self.start };
            most = looks(&[self.estimates[0].clone(), self.estimates[k].clone()], &fixed, &[Bound::share(); 2]).iter().filter(|l| l.fits()).map(Look::need).fold(most, f64::max);
        }
        most
    }

    /// The trajectories the looks need beside the units' errors but unit `u`'s and the step's share.
    fn need_without(&self, u: usize) -> f64 {
        self.need_beside(&|k| model_error_at(self.whole[k], Some((self.means[k][u], self.means[k].len())), self.unsampled))
    }

    /// The trajectories the looks need beside the units' errors but those of the units `set` and the step's share: at an
    /// output where some of them is unread, the bounds of the units read left there (`model_error_at`).
    fn need_without_all(&self, set: &[usize]) -> f64 {
        self.need_beside(&|k| {
            let less = set.iter().fold([0.; 2], |a, &u| [a[0] + self.means[k][u][0], a[1] + self.means[k][u][1]]);
            let rest = if set.len() < self.means[k].len() { f64::INFINITY } else { self.unsampled[1] };
            [probability_error(self.whole[k][0] - less[0]), if less[1].is_finite() { self.whole[k][1] - less[1] } else { rest }]
        })
    }

    /// The trajectories the looks need beside the step's share alone: every unit kept, what the reach leaves out left.
    fn need_alone(&self) -> f64 {
        self.need_beside(&|_| model_error_at(self.unsampled, None, self.unsampled))
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
    /// The current model's near tables and the reach they are made over; per cluster its table's axes and every
    /// coordinate it was made over, the axes' number and their coordinates' count at a point (`Model::table_axes`; none
    /// where it has no table).
    tables: &'a Tables,
    reach: &'a Reach,
    axes: Vec<Option<(Vec<Key>, Vec<Key>, (usize, f64))>>,
    /// The range each coordinate moves across over the reach, as a kept model's axes ask it (`Model::priced_axes`).
    moving: Mutex<HashMap<Key, f64>>,
}

impl<'a> Formula<'a> {
    fn new(model: &'a Model, kept: &'a [Unit], units: &'a [Unit], reference: [f64; GROUND], centre: V3, (tables, reach): (&'a Tables, &'a Reach)) -> Formula<'a> {
        let running = model.running();
        let axes = (0..model.clusters().len()).map(|c| tables.of(c, &running).map(|t| model.table_axes(t))).collect();
        Formula { model, kept, units, reference, centre, tables, reach, axes, moving: Mutex::new(HashMap::new()) }
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
    /// coordinates new to them that weigh there beyond the tables' shares). None where the model refuses the keep
    /// (`Model::new`).
    fn kept(&self, set: &[usize]) -> Option<(Model, Priced)> {
        let with = self.model.rekept(&[self.kept, &set.iter().map(|&u| self.units[u].clone()).collect::<Vec<_>>()].concat()).ok()?;
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
                with.priced_axes(c, self.reach, &axes, &known, &moves)
            })
            .collect();
        let priced = with.priced(&axes);
        Some((with, priced))
    }

    /// The rate at a state of the current model's cluster `cluster` -- the step's price there `price` (`prices`), its τ
    /// and its trajectory's jumps a µs -- under a model priced `priced`: the price grown where the model's cluster that
    /// holds the state's is larger (`Price::grown`, its bandwidth at least its levels' own, `bandwidth`), with that
    /// cluster's counts at a node, the middle with a step's scattering over τ, and the photons.
    fn rate(&self, model: &Model, priced: &Priced, cluster: usize, price: &Price, tau: f64, jumps: f64) -> f64 {
        let states = &self.model.clusters()[cluster].states;
        let c = model.cluster_of(states[0]);
        let held = &model.clusters()[c];
        let price = if held.states == *states { *price } else { price.grown(held.states.len(), held.ground, price.bandwidth.max(bandwidth(model, c))) };
        (price.work(priced.costs[c]) + priced.assembly) / tau + jumps * priced.photons
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
        scales: &'a Scales,
        unsampled: [f64; 2],
        again: f64,
        doubt: &[[f64; 2]],
    ) -> Pricing<'a> {
        let rated = one_each(pilot);
        let all = Level::new(None, estimates.into(), means.into(), fixed, unsampled, doubt);
        Pricing { formula, measured, pilot, fixed, scales, unsampled, rated, prices: OnceLock::new(), own: vec![], again, all }
    }

    /// The step's price at each rated state by the current model (`Formula::prices`).
    fn prices(&self) -> &[Price] {
        self.prices.get_or_init(|| self.formula.prices(&self.rated.iter().map(|pick| (pick.state.sample.cluster, pick.state.sample.at, pick.state.tau)).collect::<Vec<_>>()))
    }

    /// A model's rates at the rated states (`Formula::rate`).
    fn rated_by(&self, model: &Model, priced: &Priced) -> Vec<f64> {
        self.rated.iter().zip(self.prices()).map(|(pick, price)| self.formula.rate(model, priced, pick.state.sample.cluster, price, pick.state.tau, pick.jumps)).collect()
    }

    /// The current model's rates (`Formula::own`).
    fn own(&self) -> Vec<f64> {
        self.rated_by(self.formula.model, &self.formula.own())
    }

    /// The rates of the model that keeps the units `set` beside the kept ones (`Formula::kept`); none for a set the
    /// model refuses to keep (its folds must exist, `Model::new`): their folds' errors stand.
    fn rates(&self, set: &[usize]) -> Option<Vec<f64>> {
        self.formula.kept(set).map(|(with, priced)| self.rated_by(&with, &priced))
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
                let estimates = pooled(&parts, self.scales).0;
                let means = self.measured.means(self.fixed.model.len(), Some(left), &estimates);
                Level::new(Some(left), estimates.into(), means.into(), self.fixed, self.unsampled, &self.all.doubt)
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
        self.share_of(level, level.need_without(u), rates)
    }

    /// The share of the current model's W at a level that a kept model saves, which needs `need` trajectories at the
    /// rates `rates` (`share`).
    fn share_of(&self, level: &Level, need: f64, rates: &[f64]) -> f64 {
        let spent = self.mean(&self.own, level.left) * level.need;
        if spent > 0. { 1. - self.mean(rates, level.left) * (need + self.again) / spent } else { 0. }
    }

    /// The units to keep of the candidates `ranked` (each paying by itself, by their saving, the largest first): of the
    /// sets of their leading ones, the one whose keep together saves the most beyond its spread over the jackknife's
    /// levels `others` -- its errors leave the looks together, and its kept model is rated once --; the leading one
    /// where no larger set does better, or where a larger set's model is refused (`Model::new`): no candidate after it
    /// is taken then.
    fn joint(&self, others: &[Level], ranked: &[usize]) -> Vec<usize> {
        let mut best: (usize, f64) = (1, f64::NEG_INFINITY);
        for j in 1..=ranked.len() {
            let set = &ranked[..j];
            let Some(rates) = self.rates(set) else { break };
            let saving = Read { value: self.share_of(&self.all, self.all.need_without_all(set), &rates), spread: spread(others.iter().map(|l| self.share_of(l, l.need_without_all(set), &rates)), others.len() as f64) };
            if pays(0., saving) && assured(saving) > best.1 {
                best = (j, assured(saving));
            }
        }
        ranked[..best.0].to_vec()
    }

    /// The saving of keeping unit `u` at the rates `rates` of its kept model, spread over the jackknife's levels `others`.
    fn saving(&self, others: &[Level], u: usize, rates: &[f64]) -> Read {
        Read { value: self.share(&self.all, u, rates), spread: spread(others.iter().map(|l| self.share(l, u, rates)), others.len() as f64) }
    }

    /// The units whose keep would save at all were its model to cost no more per µs than the current one -- the
    /// trajectories its error's removal saves, less the new pilot (module documentation, # The model). A saving that is
    /// none before its spread pays at no confidence: only these are read on the jackknife's levels and priced.
    fn saving_at_all(&self) -> Vec<usize> {
        // No unit's removal saves more than the removal of them all (a look's need falls with the model's error):
        // where even that does not pay for the new pilot, none is tried.
        if self.all.need_alone() + self.again >= self.all.need {
            return vec![];
        }
        (0..self.formula.units.len()).into_par_iter().filter(|&u| self.all.need_without(u) + self.again < self.all.need).collect()
    }

    /// Unit `u` with its kept model rated (`rates`; a keep costs less per µs than the fold it replaces or far more -- a
    /// drive kept across a fine-structure splitting, its generator's norm ~10⁷ rad/µs), its saving spread over the
    /// jackknife's levels `others`.
    fn priced(&self, others: &[Level], u: usize) -> Option<Candidate> {
        let rates = self.rates(&[u])?;
        Some(Candidate { unit: u, saving: self.saving(others, u, &rates) })
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
/// diagonal, the exact moments at t = 0 ((ħ/m)⟨κ_a²⟩ per lab axis, rad/µs), the pilot's fixed errors, the step's part of
/// every output's accounts -- the step's share until the pilot's two levels are `checked` --, the ledger's items and
/// note, and the method's description.
struct Context {
    plan: Plan,
    tau: f64,
    scales: Scales,
    initial: [f64; STATES],
    momentum: V3,
    fixed: Fixed,
    /// What the atoms outside the reach add to the model's error per output (`apart`), in `fixed` with the pilot's:
    /// nothing until the product reads them.
    outside: Vec<[f64; 2]>,
    step: Vec<Bound>,
    checked: Option<Checked>,
    ledger: Ledger,
    method: String,
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

/// The result of the estimates of `trajectories` trajectories (contract §2.2): at t = 0 the request's initial state and the
/// ensemble's exact moments (T, and n̄ as of one atom with that mean, estimate.rs `occupation`); an output with no atom
/// held carries the last one's populations, one with fewer than two atoms the last T read with its judgement of the
/// validated regime (`classical`: a statement about that T) and n̄, their error unknown; one whose T is its own but whose
/// jackknife reads no error (some group's removal leaves fewer than two atoms) says so; each flagged, apart.
fn assemble(c: &Context, estimates: &[Estimate], listed: &[Vec<(V3, V3)>], counts: &Counts, trajectories: u64) -> Solution {
    let (times, outputs) = (&c.plan.times, c.plan.times.len());
    let hm = c.scales.hbar_over_mass;
    let w = c.scales.frequencies;
    let classical = |t: &V3| -> [bool; 3] { std::array::from_fn(|a| classical(t[a], w[a])) };
    let mut ledger = c.ledger.clone();
    let (population, temperature) = accounts(estimates, &c.fixed, &c.step);
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
    // What the atoms outside the reach carry (`apart`): its largest over the outputs, weighed by the held share.
    for (i, what) in ["布居", "温度"].into_iter().enumerate() {
        let largest = (1..outputs).map(|k| c.outside[k][i] * (1. - estimates[k].loss)).fold(0., f64::max);
        ledger.item(&format!("计算中跑出估计的活动范围的原子（{what}）"), largest, "在每个输出时刻统计：曾有一步因速率太大而时间步长对它不适用的原子（从那一步起），按它们所占的比例全部计入（布居上是原子数的比例，温度上是动能的比例）；其余的按所占的比例乘以在它们的状态上实测的模型误差（只计超出已计入的部分）");
    }
    ledger.close(times, &population, &temperature, &estimates.iter().map(|e| e.temperature_target).collect::<Vec<f64>>());
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
        .filter(|&k| estimates[k].motion.as_ref().is_some_and(|m| m.temperature_errors.iter().all(Spread::read)) && !(c.fixed.model[k][1].is_finite() && c.step[k].temperature.is_finite()))
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
        (&counts.unproven, "走出了本轮计算的活动范围，模型在它们的路径上未经证明（本轮随后扩大活动范围重新计算）".to_string()),
    ] {
        if let Some(t) = stopped.since {
            s.approximations.flag(format!("{} 条轨迹{why}（最早在 {}）：此后它们不计入估计，所占的比例全部计入布居与丢失的误差", stopped.count, when(t)));
        }
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

    /// The typical atom is no faster than the proof set's fastest (`typical_speed`), so the first rung of a reach grown
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
    /// share of T's allowance at the request's temperature. PGC3D's Rb87 at 30 µK on D2 (k = 8.0528 /µm; its tweezer's
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

    /// A descent decided on a refuted reach does not carry into the next attempt (`Attempt::stood`, `begins`): after any
    /// growth -- the falsifier's, with the refuting atoms' states or without, at the product's end, or for a path not
    /// proven, with a tube -- the next attempt begins at its own first rung; after a keep, over the same reach, at the
    /// step it stood at where that is the shorter. (S3c over 1 ms, seed 2: its 30 µK reach's check went down to 0.3581
    /// µs; at 65 µK the first rung was 0.9712 µs, and the next attempt began at 0.3581, twice the product's steps.)
    #[test]
    fn the_next_attempt_begins_at_its_own_first_rung_after_any_growth() {
        let first_rung = 0.9712;
        let tube = Tube { centre: [0.; 3], half: [1.; 3], level: 3. };
        for grown in [Attempt::Grow(65.179, f64::NEG_INFINITY, vec![], Some(vec![])), Attempt::Grow(65.179, f64::NEG_INFINITY, vec![], None), Attempt::Grow(30., 12., vec![tube], None)] {
            assert_eq!(grown.stood(), None);
            assert_eq!(begins(grown.stood(), first_rung), first_rung);
            assert!((0..8).map(|k| begins(grown.stood(), first_rung) / f64::from(1 << k)).all(|rung| (rung - 0.3581).abs() > 1e-3), "the refuted reach's 0.3581 µs is no rung of the next ladder");
        }
        assert_eq!(begins(Attempt::Keep(vec![], 0.3581).stood(), first_rung), 0.3581);
        assert_eq!(begins(Attempt::Keep(vec![], 1.4325).stood(), first_rung), first_rung);
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
        let (_, third, step, pairs) = levels.told(tau, &scales);
        assert_eq!((third.is_none(), step, pairs), (true, None, 360));
        let lowered = carried(&at(Some(pilot(0.8 * b, 0.1 * b))), 0.5);
        levels.above = Some(lowered.clone());
        let (_, third, step, _) = levels.told(tau, &scales);
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
    /// nothing beside its pilot (the reads exact). Three units of errors 0.06, 0.05 and 0.04 on T beside 0.05 of the
    /// rest (0.2), each adding the same work: the heaviest first, 0.14 and 0.09 still leave no room, 0.05 does -- all
    /// three, together: a set that leaves no room costs its pilot and then a larger set's. With 0.02, 0.08 and 0.03
    /// beside 0.02 (0.15): the second alone, 0.07. With room from the start, none; at an output not read, or where no
    /// atom is held, none.
    #[test]
    fn the_units_the_looks_need_kept_are_taken_by_error_per_work_until_they_have_room() {
        let estimates = vec![estimate([10.; 3], 0., 88.), estimate([10.; 3], 0., 88.)];
        // A set's rate the sum of its units', and a bound below every set's.
        let needed = |estimates: &[Estimate], fixed: &Fixed, means: &[Vec<[f64; 2]>], held: &[f64], read: &[bool], least: f64, work: &dyn Fn(&[usize]) -> Option<f64>| {
            let price = |set: &[usize], beyond: &[usize]| -> [f64; 2] {
                let short = tight(estimates, &without(fixed, means, read, set), held, read).iter().any(|k| !beyond.contains(k));
                [0., f64::from(u8::from(short))]
            };
            needed(estimates, means, held, read, least, work, &|set| without(fixed, means, read, set), &price, 1.)
        };
        let summed = |rate: &dyn Fn(usize) -> Option<f64>, set: &[usize]| set.iter().map(|&u| rate(u)).sum::<Option<f64>>();
        let case = |errors: [f64; 3], rest: f64, held: f64, read: bool| {
            let means = vec![vec![[0.; 2]; 3], errors.iter().map(|&e| [0., e]).collect::<Vec<_>>()];
            let fixed = Fixed { model: vec![[0.; 2], [0., errors.iter().sum::<f64>() + rest]], start: 0. };
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
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.2]], start: 0. };
        let with = |cost: [Option<f64>; 3]| needed(&estimates, &fixed, &means, &[1., 1.], &[false, true], cost.iter().flatten().copied().fold(f64::INFINITY, f64::min), &|set| summed(&|u| cost[u], set));
        assert_eq!(with([Some(1.), Some(3.), Some(1.)]), [0, 2, 1]);
        assert_eq!(with([Some(1.), None, Some(1.)]), [0, 2]);
        assert_eq!(with([Some(1.), Some(0.5), Some(1.)]), [1, 0, 2]);
        // A rate is made where a set asks for it, once, against a bound below every rate: the same set as with every rate
        // made. Twelve units of errors 0.004 to 0.048 on T (0.312 in all), their kept models' rates 1 to 7 by their place,
        // and the bound a half to the whole of the least: the units kept are the same, and at the least itself the
        // units' rates made fewer than the units that weigh.
        let means = vec![vec![[0.; 2]; 12], (0..12).map(|u| [0., 0.004 * (1 + (u * 5) % 12) as f64]).collect::<Vec<_>>()];
        let fixed = Fixed { model: vec![[0.; 2], [0., means[1].iter().map(|m| m[1]).sum::<f64>()]], start: 0. };
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
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.38]], start: 0. };
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
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.17]], start: 0. };
        let (held, read) = ([1., 1.], [false, true]);
        // The work a µs of a set of n units: one and the first n of `more`.
        let pick = |more: [f64; 3]| {
            let price = |set: &[usize], _: &[usize]| -> [f64; 2] { [500., [1., 1., 0.05, 0.][set.len()]] };
            let work = |set: &[usize]| Some(1. + more[..set.len()].iter().sum::<f64>());
            needed(&estimates, &means, &held, &read, 1., &work, &|set| without(&fixed, &means, &read, set), &price, 88.).len()
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
            let work = |set: &[usize]| Some(if by == 0 { row(set).3 } else { row(set).4 }).filter(|w| w.is_finite());
            least_expected(&(0..16).collect::<Vec<_>>(), &|u| singles[u][by], 1509565.7, &work, &|set| [row(set).1, row(set).2], 88.)
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
        let (held, read) = (vec![1.; outputs], vec![false, true, true]);
        let booked: Vec<Booked> = (0..units).map(Booked::Read).collect();
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
            Part { sums, listed: vec![], samples: vec![vec![]; outputs], counts: Counts::default() }
        };
        let parts: Vec<Part> = (0..groups).map(|_| part()).collect();
        let parts: Vec<&Part> = parts.iter().collect();
        let short = |set: &[usize], next: &[Vec<[f64; 2]>]| -> bool {
            let model = (0..outputs).map(|k| if k == 0 { [0.; 2] } else { [0., (0..units).filter(|u| !set.contains(u)).map(|u| next[k][u][1]).sum()] }).collect();
            !tight(&estimates, &Fixed { model, start: 0. }, &held, &read).is_empty()
        };
        let order: Vec<usize> = (0..units).collect();
        let (mut predicted, mut seen) = (vec![0.; units + 1], vec![0.; units + 1]);
        let (mut read_short, mut read_predicted) = (0., 0.);
        for trial in 0..pilots as u64 {
            let (errors, at, from) = draw(2 * trial);
            let measured = Measured { errors: &errors, output: &at, group: &from, booked: &booked };
            let means = measured.means(outputs, None, &estimates);
            let fixed = Fixed { model: model_error(&sums(&means, [0.; 2]), [0.; 2]), start: 0. };
            let doubt = Doubt::new(&measured, outputs, &parts, &scales, [0.; 2]);
            let (errors, at, from) = draw(2 * trial + 1);
            let next = Measured { errors: &errors, output: &at, group: &from, booked: &booked }.means(outputs, None, &estimates);
            for j in 0..=units {
                let set = &order[..j];
                predicted[j] += doubt.chance(set, &without(&fixed, &means, &read, set), &estimates, &held, &read, &[]);
                seen[j] += f64::from(u8::from(short(set, &next)));
            }
            // The set the error read asks: the heaviest units until it has room.
            let at_read = (0..=units).find(|&j| tight(&estimates, &without(&fixed, &means, &read, &order[..j]), &held, &read).is_empty()).unwrap_or(units);
            read_short += f64::from(u8::from(short(&order[..at_read], &next)));
            read_predicted += doubt.chance(&order[..at_read], &without(&fixed, &means, &read, &order[..at_read]), &estimates, &held, &read, &[]);
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
        let (fixed, step) = (Fixed { model: vec![[0.01, 0.02]; 4], start: 0. }, vec![Bound::share(); 4]);
        let (population, temperature) = accounts(&estimates, &fixed, &step);
        assert!((population[1].model - 0.01).abs() < 1e-15 && temperature[1].is_some_and(|t| (t.model - 0.02).abs() < 1e-15), "{population:?} {temperature:?}");
        assert!((population[2].model - 0.51).abs() < 1e-15 && temperature[2].is_none());
        assert!((population[3].model, population[3].statistics) == (1., 1.) && temperature[3].is_none());
        assert_eq!(crowded(&looks(&estimates, &fixed, &step)), [2, 3]);
        let counted: Vec<Estimate> = estimates.iter().map(|e| Estimate { uncounted: 0., ..e.clone() }).collect();
        assert!(crowded(&looks(&counted, &fixed, &step)).is_empty());
        let plan = Plan::new(&[0., 10., 20., 30.], [1.; 3]);
        let listed = vec![vec![]; plan.listed.len()];
        let context = Context {
            plan,
            tau: 1.,
            scales,
            initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))),
            momentum: [1.; 3],
            fixed,
            outside: vec![[0.; 2]; 4],
            step,
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let unmade = [Stopped::default(), Stopped { count: 2, since: Some(14.) }, Stopped::default()];
        let counts = Counts { steps: 0, failed: Stopped { count: 3, since: Some(12.5) }, unmade, unproven: Stopped { count: 5, since: Some(15.) }, optical: [0.0; 4] };
        let s = assemble(&context, &estimates, &listed, &counts, 8);
        assert!(s.estimated_errors.iter().chain(&s.loss_probability).chain(s.populations.iter().flatten()).all(|x| x.is_finite()), "{s:?}");
        assert!(s.loss_probability[3] == s.loss_probability[2] && s.populations[3] == s.populations[2] && s.estimated_errors[3] >= 1., "{s:?}");
        let flagged = s.approximations.flagged.expect("flagged");
        for said in ["3 条轨迹的状态在数值上无法再推进（最早在 12.5 μs）：此后它们不计入估计", "2 条轨迹在某一步上，光学扇区的噪声算子 K 做不出（ℳ 奇异", "5 条轨迹走出了本轮计算的活动范围", "20 μs：有轨迹不计入估计，温度的误差无法估计", "30 μs 起：没有一条轨迹计入估计"] {
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
            plan,
            tau: 1.,
            scales,
            initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))),
            momentum: [1.; 3],
            fixed: Fixed { model: vec![[0.01, 0.02]; 5], start: 0. },
            outside: vec![[0.; 2]; 5],
            step: vec![Bound::share(); 5],
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let s = assemble(&context, &estimates, &listed, &Counts::default(), 20);
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
                let fixed = Fixed { model: vec![booked; kinds + 1], start: 0. };
                let step = vec![Bound::share(); kinds + 1];
                let (population, temperature) = accounts(&estimates, &fixed, &step);
                assert!(population.iter().all(|a| number(a.total()) && a.total() <= 3f64.sqrt()), "{booked:?}: {population:?}");
                assert!(temperature.iter().flatten().all(|a| number(a.total())), "{booked:?}: {temperature:?}");
                for l in looks(&estimates, &fixed, &step) {
                    assert!(number_or_unbounded(l.need()) && batch(l.need(), 88) >= 96 && batch(l.need(), 88) <= 176, "{booked:?}: {l:?} needs {}", l.need());
                }
            }
            let s = assemble(&Context {
                plan: Plan::new(&(0..=kinds).map(|k| 10. * k as f64).collect::<Vec<_>>(), [1.; 3]),
                tau: 1.,
                scales,
                initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))),
                momentum: [1.; 3],
                fixed: Fixed { model: vec![[0.01, 0.02]; kinds + 1], start: 0. },
                outside: vec![[0.; 2]; kinds + 1],
                step: vec![Bound::share(); kinds + 1],
                checked: None,
                ledger: Ledger::default(),
                method: String::new(),
            }, &estimates, &vec![vec![]; Plan::new(&(0..=kinds).map(|k| 10. * k as f64).collect::<Vec<_>>(), [1.; 3]).listed.len()], &Counts::default(), 20);
            assert!(s.estimated_errors.iter().chain(&s.loss_probability).chain(s.populations.iter().flatten()).all(|x| number(*x)), "{s:?}");
            let flagged = s.approximations.flagged.expect("flagged");
            assert!(flagged.contains("30 μs") && flagged.contains("70 μs：没有一条轨迹计入估计") && flagged.contains("有轨迹不计入估计，温度的误差无法估计"), "{flagged}");
        }
        let read = between(&fine, &coarse, &scales);
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

    /// What the atoms outside the reach add to the model's error (`apart`), in a run whose model's error is booked
    /// at 2 % on what is shown and 3 % on T. At the first output after t = 0 a tenth of the atoms are outside with a
    /// fifth of the kinetic energy, none unresolved; three groups hold an outside state there, read at 1 %, 5 % and 9 %
    /// on what is shown and at 3 %, 13 % and 23 % on T: the means, 5 % and 13 %, exceed the booked error by 3 % and
    /// 10 %, and the atoms add 0.1·0.03 and 0.2·0.10. At the second, 1 % of the atoms with 4 % of the energy are outside
    /// and as many unresolved; the outside state reads below the booked error: the unresolved add their whole shares. At
    /// the third no atom is outside: nothing; nor at t = 0, where half of them are, as many unresolved, and no state is
    /// read: the unresolved add their whole shares. A state is read once in a run, and of the groups that hold an outside
    /// state at an output only as many as the pilot has are read: a twelfth group's state, read at the whole, changes
    /// nothing.
    #[test]
    fn the_atoms_outside_the_reach_add_what_their_states_read_beyond_the_booked_error() {
        let with = |outside: [f64; 2], unresolved: [f64; 2]| Estimate { outside, unresolved, ..estimate([10.; 3], 0., 800.) };
        let estimates = [with([0.5; 2], [0.5; 2]), with([0.1, 0.2], [0.; 2]), with([0.01, 0.04], [0.01, 0.04]), with([0.; 2], [0.; 2])];
        let state = |index: u64, k: usize| State { sample: Sample { cluster: 0, at: crate::model::Point { r: [0.; 3], kappa: [0.; 3], t: k as f64 }, psi: vec![] }, tau: 1., index, outside: true, jumps: 0 };
        let part = |states: [Option<u64>; 4]| Part { sums: Group::new(4), listed: vec![], samples: states.iter().enumerate().map(|(k, index)| index.map(|index| state(index, k)).into_iter().collect()).collect(), counts: Counts::default() };
        let parts = [part([Some(0), Some(0), None, None]), part([Some(8), Some(8), Some(9), None]), part([None, None, None, None]), part([None, Some(24), None, None])];
        let all: Vec<&Part> = parts.iter().collect();
        let booked = [[0.02, 0.03]; 4];
        let reads = AtomicU64::new(0);
        let read = |state: &State| -> Booking {
            reads.fetch_add(1, Ordering::Relaxed);
            let error = match state.index {
                0 => [0.01, 0.03],
                8 => [0.05, 0.13],
                9 => [0.001, 0.002],
                24 => [0.09, 0.23],
                _ => [1.; 2],
            };
            Booking { error, ..Booking::NONE }
        };
        let mut known = HashMap::new();
        let adds = apart(&estimates, &all, &booked, &read, &mut known);
        let near = |a: [f64; 2], b: [f64; 2]| (a[0] - b[0]).abs() < 1e-15 && (a[1] - b[1]).abs() < 1e-15;
        assert!(near(adds[0], [0.5; 2]) && near(adds[1], [0.1 * 0.03, 0.2 * 0.10]) && near(adds[2], [0.01, 0.04]) && near(adds[3], [0.; 2]), "{adds:?}");
        assert_eq!((reads.load(Ordering::Relaxed), known.len()), (4, 4));
        assert_eq!(apart(&estimates, &all, &booked, &read, &mut known), adds);
        assert_eq!(reads.load(Ordering::Relaxed), 4);
        let many: Vec<Part> = (0..12).map(|g| part([None, Some(if g < 11 { 8 } else { 100 }), None, None])).collect();
        let adds = apart(&estimates, &many.iter().collect::<Vec<_>>(), &booked, &read, &mut HashMap::new());
        assert!(near(adds[1], [0.1 * 0.03, 0.2 * 0.10]), "{adds:?}");
    }

    /// What a group keeps of its trajectories' states (`Shape::fold`, `outside_alone`). Four trajectories at two
    /// outputs; the second and the fourth are outside the reach at the last output and leave their states there,
    /// the pilot's leave every state. A product's group keeps, per output, the first state it is handed: the second
    /// trajectory's; a pilot's keeps them all, and of those `outside_alone` leaves the first that is outside -- the
    /// same state.
    #[test]
    fn a_group_keeps_its_first_state_outside_the_reach() {
        let plan = Plan::new(&[0., 1.], [1.; 3]);
        let record = |index: u64, keep: bool| {
            let outside = index % 2 == 1;
            let held = |outside: bool| Fate::Held(Held { rows: [0.; STATES], kappa: [0.; 3], kappa2: [0.; 3], outside, unresolved: false, leaving: false });
            let state = |k: usize, outside: bool| State { sample: Sample { cluster: 0, at: crate::model::Point { r: [0.; 3], kappa: [0.; 3], t: k as f64 }, psi: vec![] }, tau: 1., index, outside, jumps: 0 };
            let samples = vec![keep.then(|| state(0, false)), (keep || outside).then(|| state(1, outside))];
            Record { outputs: vec![held(false), held(outside)], listed: vec![None; plan.listed.len()], samples, counts: Counts::default(), unproven: None, scattered: [0.; 2] }
        };
        let kept = |part: &Part| -> Vec<Vec<u64>> { part.samples.iter().map(|states| states.iter().map(|state| state.index).collect()).collect() };
        let product = Shape::of(&plan, false).fold((0..4).map(|i| record(i, false)));
        assert_eq!(kept(&product), [vec![], vec![1]]);
        assert_eq!(product.sums.sums[1].outside[0], 2.);
        let mut pilot = [Shape::of(&plan, true).fold((0..4).map(|i| record(i, true)))];
        assert_eq!(kept(&pilot[0]), [vec![0, 1, 2, 3], vec![0, 1, 2, 3]]);
        outside_alone(&mut pilot);
        assert_eq!(kept(&pilot[0]), kept(&product));
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
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.03]], start: 0. };
        let none = Bound { shown: 0., temperature: 0. };
        let step = [none, Bound { shown: 0., temperature: 0.02 / TEMPERATURE_TARGET }];
        let (n, total) = (800., (25. + 6. + 10.) / 3.);
        let quiet = estimate([25., 6., 10.], 0.005 * total, n);
        let run = [quiet.clone(), quiet];
        let all = looks(&run, &fixed, &step);
        assert_eq!(all.iter().map(|l| (l.output, l.shown)).collect::<Vec<_>>(), [(1, true), (1, false)]);
        assert!(all[0].account.step == 0. && (all[1].account.step - 0.02).abs() < 1e-17);
        assert!(all.iter().all(|l| l.met() && l.need() < n) && crowded(&all).is_empty());
        let noisy = estimate([25., 6., 10.], 0.075 * total, n);
        let run = [noisy.clone(), noisy];
        let t = looks(&run, &fixed, &step)[1];
        let need = n * (CONFIDENCE * 0.075 / room(TEMPERATURE_TARGET, 0.03, 0.02)).powi(2);
        assert!(!t.met() && (t.need() - need).abs() < 1e-9 * need, "{} against {need}", t.need());
        let wide = [none, Bound { shown: 0., temperature: 4. }];
        assert!(!looks(&run, &fixed, &wide)[1].fits() && crowded(&looks(&run, &fixed, &wide)) == [1]);
        let mut full = Fixed { model: fixed.model.clone(), start: 0. };
        full.model[1][1] = 0.11;
        assert!(crowded(&looks(&run, &full, &step)) == [1]);
        let account = accounts(&run, &fixed, &step).1[1].unwrap();
        assert!(account.model == 0.03 && (account.step - 0.02).abs() < 1e-17 && (account.statistics - CONFIDENCE * 0.075).abs() < 1e-14, "{account:?}");
    }

    /// A T whose statistics no group reads is held as one whose statistics are the normal theory's prediction, in whether
    /// its look is met and in the trajectories it needs alike (`looks`): S2Q at 2.5 ms, 568 trajectories and a held
    /// share of 3·10⁻⁴, met -- it neither keeps the product running nor asks for a trajectory --; at a held share of
    /// one on 100 trajectories, neither met nor within its need. Met exactly where its need is within the trajectories
    /// it has, on either side.
    #[test]
    fn an_unread_t_of_no_weight_neither_blocks_nor_asks() {
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.01]], start: 0. };
        let unread = |held: f64, n: f64| {
            let mut e = estimate([20.; 3], 1., n);
            e.loss = 1. - held;
            e.motion.as_mut().unwrap().temperature_total_error = Spread::UNREAD;
            looks(&[e.clone(), e], &fixed, &[Bound::share(); 2])[1]
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
            let read = between(&fine, &level(t, 1.05), &scales).temperature.expect("T read at both levels").value;
            let mut e = estimate([t; 3], 0.075 * t / CONFIDENCE, 800.);
            e.temperature_target = allowance(t, scale);
            let fixed = Fixed { model: vec![[0.; 2], [0., 0.03]], start: 0. };
            let need = looks(&[e.clone(), e.clone()], &fixed, &[Bound::share(); 2])[1].need();
            let weight = weights(&[vec![[0.; 2]], vec![[0.01, 0.03]]], &[e.clone(), e.clone()], &[1., 1.], &[1])[0];
            let crowded = !crowded(&looks(&[e.clone(), e], &Fixed { model: vec![[0.; 2], [0., 0.3]], start: 0. }, &[Bound::share(); 2])).is_empty();
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
        let errors = [[0.01, 0.02], [0.03, 0.04], [0.05, 0.06], [0.07, 0.08], [0.02, 0.04], [0.06, 0.08]].map(|error| Booking { error, ..Booking::NONE });
        let booking = [Booked::Read(0), Booked::Bound(1e-5), Booked::Read(1)];
        let measured = Measured { errors: &errors, output: &[1, 2, 1], group: &[0, 0, 1], booked: &booking };
        let estimates: Vec<Estimate> = (0..4).map(|_| estimate([10.; 3], 0., 8.)).collect();
        let means = measured.means(4, None, &estimates);
        assert_eq!(means[1], [[(0.01 + 0.02) / 2., (0.02 + 0.04) / 2.], [1e-5; 2], [(0.03 + 0.06) / 2., (0.04 + 0.08) / 2.]]);
        assert_eq!(means[2], [[0.05, 0.06], [1e-5; 2], [0.07, 0.08]]);
        assert_eq!(means[3], [[1., f64::INFINITY], [1e-5; 2], [1., f64::INFINITY]]);
        assert_eq!(measured.means(4, Some(1), &estimates)[1], [[0.01, 0.02], [1e-5; 2], [0.03, 0.04]]);
        assert_eq!(measured.means(4, Some(0), &estimates)[2], [[1., f64::INFINITY], [1e-5; 2], [1., f64::INFINITY]]);
        let unsampled = [1e-3, 2e-3];
        let whole = sums(&means, unsampled);
        assert_eq!(whole[1], [1e-3 + means[1][0][0] + 1e-5 + means[1][2][0], 2e-3 + means[1][0][1] + 1e-5 + means[1][2][1]]);
        let all = model_error(&whole, unsampled);
        assert_eq!((all[0], all[1], all[3]), ([0.; 2], whole[1], [1., f64::INFINITY]));
        assert_eq!(model_error_at(whole[1], Some((means[1][2], 3)), unsampled), [whole[1][0] - means[1][2][0], whole[1][1] - means[1][2][1]]);
        assert_eq!(model_error_at(whole[3], Some((means[3][0], 3)), unsampled), [1., f64::INFINITY]);
        assert_eq!(model_error_at(whole[3], Some((means[3][0], 1)), unsampled), [(whole[3][0] - 1.).min(1.), unsampled[1]]);
    }

    /// A flow's moved population booked against the output's populations (`destined`, `Booking::at`; model/units.rs,
    /// # A rate's error on the populations). A source side holding 0.9 of the atom in one ground state, the flow's far
    /// side two ground states (a hyperfine level's) holding 0.04 and 0.06; a flow of rate error x = 0.2 read at states of
    /// the source side, each booking x times its cluster's ground population 1: over the ensemble x(1 − s) = 0.18 for the
    /// far side's share s = 0.1, where the first order is x s(1 − s) = 0.018. The run books x min(s, 1 − s) = 0.02, s/(1 − s)
    /// of the states' mean -- within 1/(1 − s) of the first order --; on a far side holding half or more, all of it; the
    /// rest of a unit's error as read; a state that names no side as read.
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
        let unit = UnitError { parameter: 0.1, population: 0.25, temperature: 0.3, moved: 0.2, into: far };
        assert_eq!(Booking::of(&unit).add(&Booking { error: [0.01, 0.02], ..Booking::NONE }), Booking { error: [0.26, 0.32], moved: 0.2, into: far });
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
        let level = Level::new(None, estimates.as_slice().into(), means.as_slice().into(), &fixed, unsampled, &vec![[0.; 2]; outputs]);
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
        // A level read less a doubt (the scout's, `doubt`) needs what its looks need at every model's error so lowered,
        // to the bit: half of the error on T left where the doubt is half of it; and where the doubt is the whole error,
        // nothing -- no unit's removal then saves a trajectory.
        let half: Vec<[f64; 2]> = level.whole.iter().map(|w| [0., 0.5 * w[1]]).collect();
        let doubted = Level::new(None, estimates.as_slice().into(), means.as_slice().into(), &fixed, unsampled, &half);
        assert_eq!(doubted.need, direct(&|k| [model_error_at(level.whole[k], None, unsampled)[0], 0.5 * level.whole[k][1]]));
        for u in 0..3 {
            assert_eq!(doubted.need_without(u), direct(&|k| { let m = model_error_at(level.whole[k], Some((means[k][u], 3)), unsampled); [m[0], (m[1] - 0.5 * level.whole[k][1]).max(0.)] }), "unit {u} at half the error's doubt");
        }
        let whole: Vec<[f64; 2]> = level.whole.clone();
        let doubted = Level::new(None, estimates.as_slice().into(), means.as_slice().into(), &fixed, unsampled, &whole);
        assert!((0..3).all(|u| doubted.need_without(u) == doubted.need) && doubted.need_alone() == doubted.need, "{}", doubted.need);
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
                    outside: false,
                    // Three jumps a unit of time.
                    jumps: 3 * k as u32,
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
            let mut windows = Windows::new(&plan);
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
            plan: Plan::new(&[0., 1.], [1.; 3]),
            tau: 0.5,
            scales: Scales { hbar_over_mass: 1., frequencies: [1.; 3] },
            initial: [0.; STATES],
            momentum: [1.; 3],
            fixed: Fixed { model: vec![[0.; 2]; 2], start: 0. },
            outside: vec![[0.; 2]; 2],
            step: vec![Bound::share(); 2],
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let shape = Shape::of(&context.plan, false);
        let record = |i: u64| {
            let held = Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], outside: false, unresolved: false, leaving: false };
            Record { outputs: vec![Fate::Held(held.clone()), Fate::Held(held)], listed: vec![Some(([0.; 3], [i as f64; 3])); shape.listed], samples: vec![], counts: Counts::default(), unproven: None, scattered: [0.; 2] }
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
            Record { outputs: vec![Fate::Lost], listed: vec![], samples: vec![], counts: Counts::default(), unproven: None, scattered: [0.; 2] }
        };
        let mut silent = |_: &Solution, _: Option<f64>| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now() - CHECKPOINT, last: None, least: 0 };
        let numbers: Vec<u64> = (0..rayon::current_num_threads() as u64).collect();
        let mut seen = vec![];
        let (levels, unproven) = run_groups(&[Hand { groups: &numbers, members: 0..GROUP, shape, whole: true, record: &record }], &mut messages, &mut |_, messages| {
            seen.push((0..2).into_par_iter().map(|_| over.load(Ordering::Relaxed)).max().expect("the loop's two items"));
            messages.again();
        });
        assert_eq!(unproven, [None], "every step proven");
        assert_eq!(levels[0].len(), numbers.len());
        assert!(2 * seen[0] < numbers.len() as u64 * GROUP, "{seen:?} of {}", numbers.len() as u64 * GROUP);
        // Nothing was sent (there is no last result): the message stayed due, and was asked for again as the groups came
        // in, once a group at the most.
        assert!(messages.due().is_zero() && seen.len() > 1 && seen.len() <= numbers.len(), "{seen:?}");
    }

    /// Steps not proven end a hand-out with what all of a level's ask of the proof set together (`run_groups`,
    /// `Asks::join`), each level's apart: of 40 groups of eight on two levels (members 0..8 and 2..5), the trajectories of
    /// the numbers 157, 230, 231 and 233 have one on the first level (each asking its own number within the region, the
    /// 230th also its number less a hundred outside it; the 233rd and the 157th asking their own tubes, at their numbers),
    /// and of the numbers 155, 156 and 159 on the second -- each a coarser level's own: its paths ask nothing of the
    /// first's. The hand-out runs every trajectory and tells 231 and 130 with the tubes of 157 and 233, in the list's
    /// order (the 157th's group, 19, comes after the 233rd's, 29), for the first level, and 156 and −∞ for the second
    /// (the number 159 is not its member) -- the same at every run, whatever the threads did.
    #[test]
    fn steps_not_proven_end_the_hand_out_with_what_each_level_asks() {
        let shape = Shape { outputs: 1, listed: 0, keep: false };
        let numbers: Vec<u64> = (20..40).chain(0..20).collect();
        let tube = |i: u64| Tube { centre: [i as f64; 3], half: [1.; 3], level: i as f64 };
        for _ in 0..3 {
            let ran = AtomicU64::new(0);
            let record = |level: usize| {
                let ran = &ran;
                move |i: u64| {
                    ran.fetch_add(1, Ordering::Relaxed);
                    let ends: &[u64] = if level == 0 { &[157, 230, 231, 233] } else { &[155, 156, 159] };
                    let unproven = ends.contains(&i).then(|| match i {
                        157 | 233 => crate::step::Unproven { region: f64::NEG_INFINITY, outside: f64::NEG_INFINITY, tube: Some(tube(i)) },
                        _ => crate::step::Unproven { region: i as f64, outside: if i == 230 { i as f64 - 100. } else { f64::NEG_INFINITY }, tube: None },
                    });
                    Record { outputs: vec![Fate::Lost], listed: vec![], samples: vec![], counts: Counts::default(), unproven, scattered: [0.; 2] }
                }
            };
            let (first, second) = (record(0), record(1));
            let mut silent = |_: &Solution, _: Option<f64>| {};
            let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0 };
            let hands = [Hand { groups: &numbers, members: 0..GROUP, shape, whole: true, record: &first }, Hand { groups: &numbers, members: 2..5, shape, whole: true, record: &second }];
            let (levels, told) = run_groups(&hands, &mut messages, &mut |_, _| {});
            assert_eq!(told, [Some(Asks { region: 231., outside: 130., tubes: vec![tube(233), tube(157)] }), Some(Asks { region: 156., outside: f64::NEG_INFINITY, tubes: vec![] })]);
            assert_eq!(levels.iter().map(Vec::len).collect::<Vec<_>>(), [numbers.len(), numbers.len()]);
            assert_eq!(ran.load(Ordering::Relaxed), numbers.len() as u64 * (GROUP + 3));
        }
    }

    /// The work formula prices a model as its run counts it (`Formula`). R4_LIN_2's model closed over its 30 µK reach,
    /// with its tables: the current model priced from its plans on its tables' axes (`Model::priced`) is its run's own
    /// count, cluster by cluster at a node, and the middle with its scattering and a jump's photons at the focus, to the
    /// bit. A kept model -- a bare drive taken to the atom, a cluster's fold to its fourth order, a carrier's dressing
    /// made coherent -- priced of its structure alone is its run's with its own tables made, at states of every cluster
    /// at rest and at the reach's speed along each lab axis: the bare drive's and the dressing's within 1 % (measured:
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
        let mut kept = vec![];
        let (model, units, trap, proven) = closed(&request, &light, &mut kept, reference, 30., &[]);
        let (tau, end) = (0.4335, 800.);
        let most = (pilot_size() as f64 * (end / tau + end / (2. * tau))).ceil() as usize;
        let reach = &proven.region;
        let tables = model.tabulate(reach, most);
        let formula = Formula::new(&model, &kept, &units, reference, trap.centre, (&tables, reach));
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
            let (with, priced) = formula.kept(set).unwrap();
            let priced: Vec<f64> = states.iter().zip(&prices).map(|(&(c, _, tau), price)| formula.rate(&with, &priced, c, price, tau, 1.)).collect();
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

    /// A model closed over the proof set of a temperature's reach and the tubes `tubes` (`solve`): the units failing
    /// over it kept, until none fails.
    fn closed(request: &Request, light: &Light, kept: &mut Vec<Unit>, reference: [f64; GROUND], temperature: f64, tubes: &[Tube]) -> (Model, Vec<Unit>, Trap, Proven) {
        loop {
            let model = Model::new(request, light, kept).unwrap();
            let (trap, units) = (model.trap(&reference, &model.prepared()).unwrap(), model.units());
            let proven = Proven::new(&model, &units, model.reach(&trap, &reference, temperature, unseen(MOST_TRAJECTORIES as f64)), tubes, None);
            let failing: Vec<Unit> = units.iter().zip(&proven.gates).filter(|(_, g)| !g.exists()).map(|(u, _)| u.clone()).collect();
            if failing.is_empty() {
                return (model, units, trap, proven);
            }
            kept.extend(failing);
        }
    }

    /// The domain of a proof set as an attempt makes it (`attempt`), handed to `f`.
    fn with_domain<R>(model: &Model, units: &[Unit], proven: &Proven, trap: &Trap, f: impl FnOnce(&Domain) -> R) -> R {
        let judged = model.judged(units);
        let fastest = proven.fastest();
        let (open, every) = (model.everywhere(&judged, fastest).1, (0..units.len()).collect::<Vec<_>>());
        let moduli = model.moduli(proven.parts());
        let kick = model.kick_curvature(&proven.region, &moduli, trap);
        let entries = model.sampled(&model.running());
        let frequency = trap.frequencies.iter().copied().fold(0., f64::max);
        let top = top(trap, unseen(MOST_TRAJECTORIES as f64));
        let domain = Domain { model, reach: &proven.region, tubes: &proven.tubes, top, fastest, trap, judged: &judged, open: &open, every: &every, hbar_over_mass: model.hbar_over_mass(), frequency, kick, moduli: &moduli, sampled: &entries, result: true };
        f(&domain)
    }

    /// A path no temperature's reach holds is held by its tube, and moves nothing else (module documentation, # The
    /// reach's atoms; model/reach.rs, # Tubes) -- the class, on S2Q's tweezer and probe (its sanity scene), its model
    /// closed over its 30 µK reach. Steps of 1 µs through 400 random points of the reach's box, in random directions, at
    /// prepared energies from 0.4 to 1.15 of the trap's depth above its bottom (past the depth, the atom leaving):
    /// - every step the domain takes lies in the proof set: within the reach, or held along its own path (`holds_along`,
    ///   stable at the set's kick);
    /// - every step it does not take asks the reach to hold its energy where some temperature's reach does (below the
    ///   top), and else its own tube -- its box at its energy -- and nothing of the reach;
    /// - the set grown by such tubes keeps the reach -- its level its temperature's quantile -- and the reach's gates to
    ///   the bit, and each tube's gates are the tube's own, proven from scratch;
    /// - the model closed again over the grown set takes every one of those steps.
    #[test]
    fn a_path_no_reach_holds_is_held_by_its_tube_and_moves_nothing_else() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("S2Q.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let initial = components(&request);
        let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).unwrap();
        let reference = alone(initial[chief].0);
        let mut kept = vec![];
        let (model, units, trap, proven) = closed(&request, &light, &mut kept, reference, 30., &[]);
        let (share, hm, prepared) = (unseen(MOST_TRAJECTORIES as f64), model.hbar_over_mass(), model.prepared());
        let (top, depth, half) = (top(&trap, share), -trap.minimum, 0.5);
        let mut stream = crate::random::Stream::jump(5, 1, 0);
        let (mut asked, mut accepted, mut warmer) = (vec![], 0, 0);
        with_domain(&model, &units, &proven, &trap, |domain| {
            for _ in 0..400 {
                let r: V3 = std::array::from_fn(|a| proven.region.cell[a][0] + (proven.region.cell[a][1] - proven.region.cell[a][0]) * stream.uniform());
                let kinetic = trap.minimum + (0.4 + 0.75 * stream.uniform()) * depth - model.potential(r, &reference, &prepared);
                let n = stream.normals();
                let norm = n.iter().map(|x| x * x).sum::<f64>().sqrt();
                if kinetic <= 0. || norm == 0. {
                    continue;
                }
                let kappa: V3 = n.map(|x| x / norm * (2. * kinetic / hm).sqrt());
                let v = kappa.map(|k| hm * k);
                let flow = crate::step::Flow::new(&model.expansion(r, &reference, &prepared), r, hm);
                let energy = flow.highest(r, v, half) + 0.5 * hm * kappa.iter().map(|k| k * k).sum::<f64>();
                let cell = Cell { centre: r, half: v.map(|x| (x * half).abs()) };
                match crate::step::Domain::proven(domain, &flow, (r, kappa), half) {
                    Ok(_) => {
                        accepted += 1;
                        let reach = &proven.region;
                        let within = energy <= reach.minimum + reach.outside || (energy <= reach.minimum + reach.energy && model.region_of(&model.spans(&cell))[1]);
                        let speed = v.iter().map(|x| x * x).sum::<f64>().sqrt();
                        let along = model.holds_along(domain.judged, if speed <= domain.fastest { domain.open } else { domain.every }, r, v, half, speed);
                        assert!(within || along, "a step taken outside the proof set: r {r:?} v {v:?} energy {energy}");
                    }
                    Err(step) if energy - trap.minimum >= top => {
                        let tube = Tube { centre: r, half: cell.half, level: energy };
                        assert!(step.tube == Some(tube) && step.region == f64::NEG_INFINITY && step.outside == f64::NEG_INFINITY, "{step:?} at {energy} above the top {top}");
                        asked.push((tube, r, kappa));
                    }
                    Err(step) => {
                        assert!(step.tube.is_none() && step.region == energy, "{step:?} at {energy} below the top {top}");
                        warmer += 1;
                    }
                }
            }
        });
        asked.truncate(4);
        assert!(!asked.is_empty() && accepted > 0 && warmer > 0, "{} tubes, {accepted} taken, {warmer} asking a warmer reach", asked.len());
        let tubes: Vec<Tube> = asked.iter().map(|a| a.0).collect();
        let mut grown = proven.clone();
        grown.grow(&model, &units, proven.region.clone(), &tubes);
        assert!(grown.region == proven.region && grown.region.energy == model.reach(&trap, &reference, 30., share).energy && grown.parts[0].0 == proven.parts[0].0, "the reach and its gates as they were");
        for (tube, (gates, _)) in grown.tubes.iter().zip(&grown.parts[1..]) {
            assert_eq!(gates, &model.gates(&units, tube), "a tube's gates are its own");
        }
        let (model, units, trap, proven) = closed(&request, &light, &mut kept, reference, 30., &tubes);
        with_domain(&model, &units, &proven, &trap, |domain| {
            for (tube, r, kappa) in &asked {
                let flow = crate::step::Flow::new(&model.expansion(*r, &reference, &model.prepared()), *r, hm);
                assert!(crate::step::Domain::proven(domain, &flow, (*r, *kappa), half).is_ok(), "the step of the tube {tube:?} taken");
            }
        });
    }

    /// A proof begun where one ended is the proof from the box (model/reach.rs, # A proof begun where one ended; model
    /// /units.rs `Model::gates_from`) -- the class, on S2Q's tweezer and probe, its model closed over its 30 µK reach:
    /// - over the same reach, every unit's gate begun on the cells its proof ended on is the gate proven from the box, to
    ///   the bit, and so is every one over the reach grown to 60 µK (another box: proven from it);
    /// - over the 30 µK reach after a keep (the unit of the largest gate kept, the model closed again), every gate
    ///   begun where its statement reads over the box what it read before decides as the gate from the box does -- the
    ///   fold exists by the one where it does by the other --, and both bound the statement's values at the centres of
    ///   the cells either proof ended on that lie in the reach; some statements read the same, some another.
    #[test]
    fn a_proof_begun_where_one_ended_is_the_proof_from_the_box() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("S2Q.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let initial = components(&request);
        let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).unwrap();
        let reference = alone(initial[chief].0);
        let share = unseen(MOST_TRAJECTORIES as f64);
        let mut kept = vec![];
        let (model, units, trap, _) = closed(&request, &light, &mut kept, reference, 30., &[]);
        let cold = model.reach(&trap, &reference, 30., share);
        let (gates, ended) = model.gates_from(&units, &cold, None);
        let (again, _) = model.gates_from(&units, &cold, Some(&ended));
        assert_eq!(again, gates, "the same statements over the same reach");
        // Decided alike, and bounds of the values at the cells' centres in the reach.
        let alike = |model: &Model, units: &[Unit], reach: &crate::model::Reach, from: &Coverings, what: &str| {
            let (begun, begun_ended) = model.gates_from(units, reach, Some(from));
            let (boxed, boxed_ended) = model.gates_from(units, reach, None);
            let judged = model.judged(units);
            let prepared = model.prepared();
            for (u, (b, x)) in begun.iter().zip(&boxed).enumerate() {
                assert_eq!(b.exists(), x.exists(), "{what}: {:?} begun {b:?} from the box {x:?}", units[u]);
            }
            for (name, ended) in [("begun", &begun_ended), ("from the box", &boxed_ended)] {
                for (unit, cells) in &ended.of {
                    let u = units.iter().position(|v| v == unit).unwrap();
                    for cell in cells[0].cells.iter().chain(cells[1].cells.iter()).step_by(7) {
                        let room = model.room(reach, cell.centre, &prepared);
                        if room >= 0. {
                            let [saturation, sweep] = model.gate_of(&judged, u, &model.point_region(cell.centre, (2. * model.hbar_over_mass() * room).sqrt().min(reach.speed)));
                            assert!(saturation <= begun[u].saturation.max(boxed[u].saturation) && sweep <= begun[u].sweep.max(boxed[u].sweep), "{what}, {name}: {unit:?} at {:?}: {saturation} {sweep} above {:?} {:?}", cell.centre, begun[u], boxed[u]);
                        }
                    }
                }
            }
            // The statements that read the same over the box, and those that read another.
            let same = begun_ended.of.iter().filter(|(unit, now)| from.of.get(*unit).is_some_and(|before| (0..2).all(|w| before[w].reads(now[w].whole)))).count();
            (begun, boxed, same, begun_ended.of.len())
        };
        let warm = model.reach(&trap, &reference, 60., share);
        let (begun, boxed, _, _) = alike(&model, &units, &warm, &ended, "grown to 60 µK");
        assert_eq!(begun, boxed, "another box: from the box");
        let largest = (0..units.len()).filter(|&u| gates[u].exists()).max_by(|&a, &b| gates[a].saturation.max(gates[a].sweep).total_cmp(&gates[b].saturation.max(gates[b].sweep))).unwrap();
        kept.push(units[largest].clone());
        let (after, units_after, trap_after, _) = closed(&request, &light, &mut kept, reference, 30., &[]);
        let reach_after = after.reach(&trap_after, &reference, 30., share);
        assert_eq!(reach_after.cell, cold.cell, "a keep leaves the reach's box");
        let (_, _, same, of) = alike(&after, &units_after, &reach_after, &ended, "after a keep");
        assert!(same > 0 && same < of, "{same} of {of} statements read the same over the box after a keep");
    }

    /// What a pair's paths not proven ask of the reach (`own_departure`; module documentation, # The reach's atoms): the
    /// finer level's where its companion's are all proven -- the run's own level left the domain where the coarser step
    /// did not --; nothing where the companion's left it too (the rung below decides) or alone (a step's departure), or
    /// where nothing left it.
    #[test]
    fn a_reach_holds_the_paths_of_the_run_s_own_level_alone() {
        let asks = |region: f64, outside: f64| Some(Asks { region, outside, tubes: vec![] });
        let (fine, coarse) = (asks(12., f64::NEG_INFINITY), asks(900., 3.));
        assert_eq!(own_departure(&[fine.clone(), None]), fine);
        assert_eq!(own_departure(&[fine.clone(), coarse.clone()]), None);
        assert_eq!(own_departure(&[None, coarse.clone()]), None);
        assert_eq!(own_departure(&[None, None]), None);
        let mut levels = Levels::default();
        levels.depart([None, coarse]);
        levels.depart([fine.clone(), asks(4., 7.)]);
        assert_eq!(levels.departed, [fine, asks(900., 7.)], "a rung's levels hold the most every pair of them asked");
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
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], outside: false, unresolved: false, leaving: false };
            let outputs = vec![Fate::Held(held.clone()), if i % 3 > 0 { Fate::Held(held) } else { Fate::Lost }];
            Record { outputs, listed: vec![Some(([i as f64; 3], [0.; 3]))], samples: vec![], counts: Counts { steps: i, ..Counts::default() }, unproven: None, scattered: [0.; 2] }
        };
        let mut silent = |_: &Solution, _: Option<f64>| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0 };
        let numbers: Vec<u64> = (2048..2053).chain(5..37).collect();
        let members = [0..GROUP, 1..3];
        let hands = members.clone().map(|members| Hand { groups: &numbers, members, shape, whole: true, record: &record });
        let (levels, unproven) = run_groups(&hands, &mut messages, &mut |_, _| {});
        assert_eq!(unproven, [None, None], "every step proven");
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
