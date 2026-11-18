//! The pipeline: request, run, pilot, step's check, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the initial components; the run (model/fold/run.rs `Onefold`: the fold at the atom, one
//!    object for every step of the solve, its structure decided at the points the atoms are at -- which excited nodes
//!    are explicit is the run's own decision at its decision times, from its own readings, DESIGN §11 --); every initial
//!    component's prepared well and the initial ensemble drawn in them, the reported trap, the kick's curvature
//!    (`solve`, `Setting`); the outputs' plan, the first rung (# The step). Nothing of the model is decided before a
//!    trajectory runs: no prior, no table, no unit kept.
//! 2. The scout: the first member of every pilot group, at the first rung and at twice it (# The step), screened; where
//!    its rung stands on its reads, its states price the rungs (## The ladder) and the pilot runs at the one the run is
//!    expected to spend least on. The pilot: the scout's groups whole, `pilot_size` trajectories at the run's two levels
//!    (the scout's are its groups' first members and are not run again). Its groups have numbers of their own, from
//!    trajectory `MOST_TRAJECTORIES` on: beyond every number a group of the product takes, which is its place in the
//!    run's list (`product`). It checks the step -- on two levels, a third above and a rung below where they pay, and on
//!    further pairs where they pay (# The step) --, and reads the fold's readings at every state of its run level at the
//!    analysis times (# The model) -- every read at the analysis times, not at the outputs (# The analysis times). Its
//!    trajectories at the run's step stay in the result: they are trajectories of the run and its step.
//! 3. The product: batches of whole groups at the settled step, one level, every trajectory run to the run's end and
//!    read at the pilot's analysis times beside the outputs, until every held quantity's total lies within its target
//!    there (# Stopping) or the trajectories reach `MOST_TRAJECTORIES`.
//! 4. The result (contract §2.2, `assemble`).
//!
//! # Each read as its evidence is in
//! The step's check stops where the rung's own read is in and not clearly beyond (`Settled::Pending`): its course --
//! stand, the third level at 4τ, pairs, a rung below -- is priced at the states the run holds by then (the scout's, then
//! the pilot's run level's at its analysis times: `Prices`), and only then goes on. The check's levels are read on the
//! run they were run on; nothing of them is carried to another rung but its read, as the ladder says.
//!
//! # The step (SPEC §2.2)
//! One step τ for the run (adapt.rs), which follows from what the scheme must resolve and from nothing of the outputs:
//! an output is read where it falls, inside a step (adapt.rs, # The reads), so outputs a tenth of a step apart take no
//! shorter a step than outputs a hundred steps apart. The ladder's rungs are the run's length (`Plan::end`) over the
//! even numbers, and below a rung its halves: the companion's step is twice the run's, and the run is a whole number of
//! the companion's steps, so no level's last step is a cut one. A step is
//! below its edge (`stable`): the reference flow with the kick between its halves, whose linear map about a minimum is bounded below its first edge, cos ωτ −
//! (ω_k²τ²/2) sin(ωτ)/(ωτ) > −1 at the prepared trap's largest frequency ω and the kick's largest curvature ω_k² as
//! far as the run bounds it (`kick_curvature`: the ground states' own prepared wells against the reference's; the light
//! folded at the atom and the explicit states' wells are the check's to find) -- ω τ < π for
//! a kick without curvature, ω_k τ < 2 on a flat reference. The generator asks no edge of its own: the propagation
//! integrates every beat of its entries exactly at the combined frequencies of the spectrum (spectral.rs, # Tones;
//! ELIMINATION v8), so a step resolves no beat and aliases none, and what it leaves of the light -- the envelopes'
//! cubic, the second Magnus term weighted by the tones' amplitudes, the kick's quadrature at its bandwidth -- is the
//! step's error, which the check reads. The ladder begins at the largest rung within the physics prior's step that is
//! below the edge with its double (`first_rung`) -- the largest the run may take, not the one it takes: the scout's
//! states price the rungs from there down, and the pilot runs at the one of the least expected work (## The ladder) --;
//! and the prior,
//! the old engine's first step (`prior`): T's step error goes as the square of the phase k_max v_T τ the light's
//! pattern advances in a step at the initial ensemble's kinetic temperature -- one temperature for every speed the
//! first rung is taken at --, and the step may take its share of T's allowance -- so the
//! ladder does not begin at an edge's rung the check then leaves (PGC3D: 1.425 µs, left on the pilot for 0.7125 µs,
//! 11.2 CPU-s). The ladder begins at its first rung, and goes down only on its own τ/2τ reads and its states' prices.
//! The first rung is a rung of the ladder, whose bottom is the step of
//! `MOST_STEPS` steps a trajectory: a request for which no rung down to it is below the edge is refused with a message.
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
//! A rung τ is read by its own pair (τ, 2τ) and, where it pays, by a third level above, (2τ, 4τ), its read carried
//! to τ by the order (`carried`) -- lowered 2^{2p} times in the mean square, noise and all. The triple is (4τ, 2τ, τ), τ
//! its finest step. The third level is the rung left above where the check came down from it (a level the ladder ran,
//! left on its reads or for its price), else the pilot's trajectories at 4τ where the rung buys them (`above`: the
//! ladder has the rung above; `buys`: their price no more than what a rung below would add, ## The ladder). Where
//! the order holds across the triple both reads are of one mean square, e(τ)'s. A rung stands (`verdict`) where both
//! estimates are within the share, M̂ ≤ b on every look (`within`) -- the larger of the two, as multilevel Monte Carlo
//! takes the larger of its finest difference and the next one lowered by the order, so that two levels that agree by
//! being both too coarse are not taken for converged where the third shows the error --, and it is left where the
//! third's estimate is beyond: the order is not shown above it. Where no third level is to be had or bought -- a first
//! rung at the ladder's top, a rung above whose paths left the model's domain or whose read is no number, a third level
//! dearer than a rung below would make the run --, the rung stands
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
//! - Clearly beyond: a rung whose own read -- or its third level above, the rung left's read lowered -- lies clearly
//!   beyond the square b of the step's share among the family's few looks (`screen`) is left: the step is halved, the level at τ becomes the companion and only τ/2 is run anew; the
//!   rung below's third level above is the rung left's own read, lowered. Down to the step at which a trajectory would
//!   take `MOST_STEPS` steps, where the step stands as it is, flagged, and books what its reads do not exclude.
//! - The run's work. Of the rungs a check's reads let stand, the run takes the one it is expected to spend least on
//!   from here -- the product and the check together --, not the largest. A trajectory's work a µs w(h) with every
//!   step at h, each step priced at its own step (`work_at`: the step's counted work as a step taken from each state
//!   counts it, step.rs `priced`, `Price`, and its photons), at the states in hand (`one_each`: the scout's, then the
//!   pilot's); and the product's trajectories N its looks ask (`most_needed`, # Stopping: the pilot's statistics, before
//!   them the scout's) -- one N for every rung, the need beside the
//!   step's part as read: a rung below's smaller step error, which leaves the statistics more room, is not counted, and
//!   the choice errs toward the larger step; so does it where a long step's caches make it dearer than its count (a
//!   count is the operations done, no cache's effect: step.rs `Price`). With P the pilot's trajectories, P_m those of them run at both of the
//!   rung's levels (the scout's one a group, or all), n the pairs at the rung and r the rung below's level run (the
//!   scout's pairs at τ/2, `foresee`):
//!   - staying: the pilot's companions still to run and the run level to the most of N, P and n, (P − P_m) w(2τ) +
//!     (max(N, P, n) − n) w(τ) (`staying`), and the third level P w(4τ) where the rung buys it;
//!   - the k-th rung below, τ_k = τ/2^k, reached a rung at a time (each rung's own level its next one's companion):
//!     the members in hand at every rung on the way, the pilot's companions at τ_{k−1} and its run level at τ_k,
//!     P_m Σ_{j≤k} w(τ_j) + (P − P_m) w(τ_{k−1}) + (max(N, P) − P_m) w(τ_k) − r w(τ_1); the least of them
//!     (`descending`), priced down to where the levels on the way alone cost more than the least found (every other
//!     term is no less than zero) or to the ladder's bottom.
//!   The rung below is taken where some rung below is expected to cost less than staying (`course`): a smaller step is
//!   the more accurate one, and the rung below stands only on its own reads, as every rung does, so no error enters
//!   the choice -- it is the work's alone. Its third level above is this rung's read, lowered (`carried`).
//! - The third level above shows the order above a rung by its own pair (2τ, 4τ); a rung below shows it by this
//!   rung's read, inherited. So the third level is worth no more than what the least rung below adds to staying, and
//!   the rung buys it where its price P w(4τ) is no more than that, on the pilot's position (`buys`; on the scout the
//!   pilot's purchase to come is priced into staying). Where it costs more, the rung stands on its own two levels: the
//!   QuTiP-like path, as where a ladder's top has no third level. At the ladder's bottom, with no rung below, it buys
//!   it wherever there is one.
//! - Why priced. The ladder took every step at one price, w(h) ∝ 1/h: then the largest rung that stands is the
//!   cheapest -- a rung below adds at least N w(τ) to staying, the steps doubled -- and the third level, a quarter of
//!   the pilot's steps, P w(τ)/4, always less than that: the ladder as it was, to the bit, wherever a step's price is
//!   the same at every step (`a_flat_price_leaves_the_ladder_as_it_was`). Since the propagation integrates the tones
//!   exactly (spectral.rs, # Which beats are tones), a step's price grows with the step: a beat is a tone where its
//!   change through the nodes would leave more than its treatments and wherever |ν|h ≥ π, more of them the longer the
//!   step, and their products' classes, 1 + 2T + 2P, with them. GMD1C's pictures
//!   cost 1.6·10⁴ at τ = 0.95 µs, 3.7·10⁵ at 2τ and 2.2·10⁶ at 4τ, 140 times τ's: its third level, a quarter of the
//!   pilot's steps, took 23 s of a 29.6 s run (9.2 s before the tones) -- the steps' count is no measure of the work.
//! - The scout's eleven pairs leave a rung clearly beyond so, and certify nothing: a rung they leave standing is the
//!   pilot's to read. Its course is the scout's to price: its states price the rungs (# Each read as its evidence is
//!   in), and where a rung below is expected to cost the run less the scout goes down --
//!   its level at τ the rung below's companion, its read the rung below's third level, lowered --, so that the pilot's
//!   levels run at the rung it is to stand at, not at one it is to leave for its price.
//! - On the pilot (all the members of a group in: the last word on the step), a rung not clearly beyond (`verdict`):
//!   - its own estimate within: its course (`course`) -- the rung below where it costs less; else the third level
//!     above where the rung has none and buys it -- it stands where that one's estimate is within, and is left where
//!     it is beyond --; else it stands on its own two levels, or on its third where it has one;
//!   - its own estimate beyond the share and not clearly: the noise hides the difference, or the error itself lies
//!     near the share -- two causes, two answers, each priced from what is read (`remedy`). With F and L a look's far
//!     and near ends at one look (L from zero), b the share's square and N₀ the pairs read: at the best the error's
//!     mean square is L, and what the read holds above it is the pairs' noise, falling as 1/N (`blind`) --
//!     - more pairs at the rung, the product's next groups at both levels, read again after every batch: the far end
//!       within b on N_τ = N₀ (F − L)/(b − L) pairs, on no number where L is at b;
//!     - the rung below: there the error's mean square is L/2^{2p} where the order holds, and the noise is what the
//!       pairs' differences make of it at that step. Pairs that stay together differ by the step's own error along each
//!       atom's path, of the order τ^p, and their noise falls with the error, 2^{2p} times in the mean square; pairs
//!       that have parted carry the two ensembles' own, the same at every step (estimate.rs, # The step's error;
//!       measured, R4_LIN_2: rows at far ends of 1.5 to 2.5 b at 0.43, 0.22, 0.11 and 0.054 µs alike) -- a rung below is
//!       no remedy for that noise. Which of the two a run's pairs are is not assumed but measured, on the scout's pairs
//!       -- the first member of every pilot group -- run at τ/2 and again at τ and 2τ: their reads (τ/2, τ) and (τ, 2τ)
//!       on the same pairs give the noise's ratio between the two rungs, by the jackknife over the same groups
//!       (`foresee`), and the rung's own read, its error lowered by the order and its noise by that ratio, is the rung
//!       below's foreseen read, within on N_{τ/2} of its own pairs (`foreseen`). The ratio's expectation lies within
//!       [2^{−2p}, 1]: halving the step, a pair that stays together lowers its noise by the order's factor, one that has
//!       parted keeps it, and no pair's noise grows -- the noise of a mean of squared differences goes as the squares,
//!       τ^{2p} for the first kind and the ensembles' own for the second, and a run's pairs are a mixture of the two. So
//!       the ratio read is taken within that range: the scout's eleven pairs read it with their own noise, beyond one
//!       where the noise is mostly two ensembles' (measured: PGC3D 16.6, the 1D PGC tilted 2° 3.2), and a ratio taken
//!       above one foresees more pairs at the rung below than its pairs can need (the tilt: about 21 s of further pairs,
//!       measured by the check); below 2^{−2p}, it foresees fewer than they can do with. (Assumed the same at every step, p10d5's
//!       0.25 µs, on its pilot's 88 pairs at F ≈ 8 b and L = 0, asked 700 pairs at either rung and took the pairs:
//!       704 of them, 4.9 s of an 11.9 s run, before L showed and the rung below stood on the pilot's own 88 pairs --
//!       its pairs, 5 µs long, never part. Measured so, the scout's pairs read its T's noise at τ/2 0.24 times τ's:
//!       N_{τ/2} = 170 against N_τ = 700, the rung below at once, and the run took 6.0 s.)
//!     The rung takes the one the run is expected to spend less on from here, at a trajectory's work a µs at each step
//!     (`Prices`: at the pilot's states, every step at its own τ) and the product's trajectories N_p the pilot's looks ask
//!     (# Stopping): the pairs' companions at 2τ and
//!     the run level at τ to the more of N_τ and N_p -- the further pairs' run level is the product's -- (and the
//!     pilot at 4τ where the rung is still to read its third level and buys it), against the run level at τ/2 to the
//!     most of N_p, N_{τ/2} and the pilot's, beyond the scout's pairs already run there, and the further pairs'
//!     companions at τ beyond the pilot's (its level at τ is the rung below's companion). Where the pairs have parted, L ≪ b asks the
//!     same pairs at either rung, and the rung below costs its run level at τ/2 beside them: the pairs; as L nears b,
//!     N_τ grows without bound and N_{τ/2} does not: the rung below; where the pairs stay together, the rung below's
//!     noise is the order's less, and it is had on few pairs. The foreseen read decides the work alone, never the
//!     step: the rung below stands only on its own read, as every rung does, so a ratio read wrong costs time, never
//!     an error. The scout's pairs at τ/2 are the rung below's first members where the check goes down: none is run
//!     twice. Where pairs cannot see it -- beyond a run's trajectories, or on no number --, its estimate as it reads
//!     is beyond the share, and the rung is left, as an adaptive integrator leaves a step whose error estimate is
//!     beyond its tolerance. (Stood unable to see at TET's ladder top, 1.40 µs, a row's far end booked 3.8 % of the 5 %
//!     and left the statistics a room of 0.9 %: 4272 trajectories, 211 CPU-s against 39.) At the ladder's bottom, with
//!     no rung below, the pairs are taken where they can be, and else the check ends unable to see (SPEC §5.1): the
//!     step stands, flagged, and books what its reads do not exclude.
//! - A read is never carried from one rung to another as a verdict: a rung's third level is a read of levels it holds
//!   itself, of the same run at the same steps -- the scout's, where the pilot has none of its own and the ladder no
//!   rung above --, and it certifies nothing alone.
//! - A comparison of two levels says something only where the coarser one is a step of the scheme at all: the first
//!   rung's double is below the step's edge for every atom (`stable`); what it leaves of the light is the
//!   comparison's to read.
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
//!   the run level keeps its trajectories' checkpoints, next), the product at the analysis times alone.
//! - The run level's states are not kept at the candidates -- its trajectories times the candidates, in proportion to
//!   the run's length: PGC3D's 10 ms, 88 × 15 400 states, 345 MB --, but its trajectories' checkpoints (`Checkpoint`:
//!   the trajectory, its Brownian path and what its scratch carries, step.rs `Carry`, at a step's end: adapt.rs
//!   `advance`'s stops), at t = 0 and at the first step's end at or past every multiple of T's shortest window
//!   (`Plan::window`); a state at an analysis time is made again from
//!   the last checkpoint before it once the times are known (`replay`, `parts_at`), the trajectory's own to the bit
//!   (adapt.rs, # The reads: a read is an onlooker, a stop no step's end). The window is the scale the reads are spaced
//!   by already (a quarter of it, next): the checkpoints are a quarter of the first round's reads in number, about 1 kB
//!   each against a state's 0.26 kB at each of the window/(2τ₀) candidates between two of them (15 for PGC3D); and a
//!   state costs a window's steps at the most again -- |T_a| windows a trajectory against the run's end/window (PGC3D
//!   1 ms: 13 of 100).
//! - The analysis times of some levels are the halving's of their readings (`analysed`; analysis.rs, # The rule): from
//!   the run's two ends, an interval is halved while a reading at its middle, its quarters -- and, where it is long, the
//!   nested middles below them (next) -- lies off the line through its ends beyond its own jackknife interval among the
//!   interval's looks and the line's rounding. The
//!   readings are what the decisions read of the levels (estimate.rs `readings`): the run level's loss, rows as shown
//!   and T per lab axis, and, where a read compares two levels, the step's error of each of them. The step's check
//!   stands on its two levels' times (`read_step`); the model's error and the product on the run level's.
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
//! - The fold's readings are no reading of the times: a state's is the books' profile its trajectory holds there
//!   (`booked_at`), and the halving reads three times as many times as it keeps. A reading at a state is a function of its place, speed, internal
//!   state and representation, whose ensemble the rows and T read (a hypothesis, not tested by itself: an error of the
//!   fold narrower in time than those is stepped over).
//! - What stands on them: the step's mean square (estimate.rs `between`, each time by its share of the run); every
//!   state of the pilot's run level there, each group's every member held and every atom lost by then (`sampled`), the
//!   model's error and the looks'; the product's looks, batches and unresolved atoms; the prices at one state of every
//!   trajectory among them (`one_each`).
//! - What the result shows at an output: its own estimates and their standard errors; its accounts -- the model's
//!   error, the step's part, the statistics and T's allowance -- on the line through the accounts at the analysis
//!   times (`assemble`, `accounts_along`), so that the ledger flags where the stop reads, whatever the outputs. (Judged
//!   on its own statistics, a noisy estimate of the same spread, the pgc3d example stopped at the analysis times was
//!   flagged at 11 of 1000 outputs, at most 5.4 % with 5.1 % of statistics, and 2 of 100, its reason "not finished".)
//!
//! # The model (SPEC §5.2)
//! The model is the run's fold at the atom (model/fold/run.rs `Onefold`), one object: which excited nodes are explicit,
//! how far each series sweeps, which decompositions are kept, are the run's decisions at its decision times, each the
//! cheapest representation within the share by the fold's own readings there (DESIGN §11; run.rs # The nodes risen and
//! fallen). The solve decides nothing of it: no prior, no table, no unit kept, no reach -- what the old engine decided
//! before and between its attempts (a model closed at its atoms' starts, tables refined over a reach, units kept by the
//! run's total work, a reach grown where its atoms outgrew it) is the run's at every point. What the solve does is book
//! what the fold says of itself where the atoms went, as an adaptive integrator's tolerance is measured (QuTiP's way):
//! the fold's readings at the pilot's states (`sampled`, `booked_at`): the books' profile each state's trajectory holds
//! there (unify2 §2.7: per account the reading its cluster's last decision left, what every step books over its cell,
//! step.rs `Trajectory::model`; no forward made) -- the series' rest, the reuse of a decomposition along the path, the
//! centred rows, what the nodes left out leave (model/fold/proto.rs `KINDS`; `READINGS`; a landing's transient is booked
//! exactly on its modes, read by none: model/fold/proto/transient.rs), each a
//! rate's relative error at the state --, combined as one component's are, ρ² = Σ_k ρ_k² (the one combination of the
//! decision's rule and of the books, DESIGN §1.5). A reading x books x on the populations (absolute) and x on T
//! (relative), as a rate wrong by x moves them (`Booking::of`); one that is no number, or a state whose representation
//! holds no decision there, the state's whole. The held forwards that read their series' ratio at one between two
//! decisions (unresolved, unify2 §2.7) are counted and said (`Counts::unresolved`).
//!
//! ## Where the atoms went
//! The model is judged where the atoms went, not only where the held ones are at the analysis times (`sampled`). What is
//! shown at a time is every atom's fate there -- a held atom's rows, a lost one's loss --, so the model errs on it by the
//! mean over the atoms of the error of each one's fate (`Measured::means`): the held atoms' at their states there, by
//! the held share 1 − L, and the lost atoms' along their paths, by L. A lost atom's fate was decided along its path
//! from t = 0 to where it passes the potential's far value -- past it the atom runs out of the trap's region whatever
//! the rates there, and is found lost where it leaves it (adapt.rs, # The reads) --: a rate's relative error x along
//! that path moves its loss's time by x of it to the first order, and its fate's error is the path's mean of x, read
//! where the run reads every atom and at the path's end: at t = 0, at the analysis times it was held at, and at its
//! last state the run keeps below the far value (`approached`: its last checkpoint not leaving, within one window of
//! the passage; `leaving`: the reference state's prepared energy there) -- by the trapezoid on them (`Booking::path`).
//! The state it was found lost in is read nowhere: its fate is sealed there. (Read there, as half of the path, the 1D
//! PGC tilted 3° over 10 ms -- three of four atoms lost by 4 ms, each found at the trap's rim, 2 µm off its axis, where
//! its strong near light (3 to 5 Γ, 38 I_sat a beam) left the old model's fold wrong by 0.32–0.48 against 0.004–0.032
//! at every held state -- booked half its error there.) An atom held near the rim is read there where an analysis time
//! finds it, as every held atom. T is the held atoms' alone. A run's price is read where its atoms were too
//! (`one_each`): a trajectory weighs by the time it holds its atom -- a lost one's to its path's end --, and one lost
//! before the first time read by its state at t = 0.
//!
//! # The unresolved atoms
//! A step asks its domain one cheap thing (`Domain`; step.rs, # The domain): whether a state scatters beyond every state
//! of the atom, its hazard above the fastest excited state's decay wholly excited (`Widest`). The trajectory stops there
//! (`run`): unresolved from then on, its state as it stood. What an unresolved atom shows is booked whole (`apart`): its
//! share of the held atoms on what is shown -- an atom's row is a probability, so its error is at most one, whatever its
//! state, and the mean over the held atoms errs by at most the share --, its share of the held atoms' kinetic energy on
//! T -- the part of T it makes, which bounds its error while its true kinetic energy is within twice the one computed (a
//! hypothesis on T, not a bound). The loss of an atom lost after such a step is no held atom's: as every model error's,
//! it is in what the held atoms' errors are weighed by (the held share, `accounts`), and not booked apart.
//!
//! # Exits
//! Two exits, each said where it is met, none carried on as a number (unify2 §7):
//! - the request's (refused before a trajectory runs, with its reason): the run's construction (model/fold/run.rs
//!   `Onefold::new`: a set the prototype does not hold, a far block that does not exist at the focus), a prepared trap
//!   that does not hold the atom (`trap_of`), no step below the edge (`first_rung`);
//! - a trajectory's (`run`), where the run's fold does not exist for it: its start lands nowhere (no node left to rise:
//!   its held forms no number, model/fold/run.rs `Dynamics::started`; or its fast part's photon none, step.rs
//!   `Trajectory::new`), its readout at a time read is no number, or outside the trap's region its loss energy is none
//!   -- the loss test could not tell (`Dynamics::lost`). It ends there with no outcome (`Counts::nowhere`, `Fate::Failed`
//!   from then on), counted, its share booked whole as every trajectory not counted is (estimate.rs, # What is counted),
//!   and the ledger says how many and from when.
//! Where every trajectory ends so, nothing is integrated, and the trajectory's exit is the request's: the pilot's
//! trajectories, all of them ended with no outcome before the first output after t = 0, refuse the request with the
//! reasons they ended by (`nothing_integrated`) -- a result would be t = 0 carried to every output (`assemble`), and no
//! result of that is passed on as a checkpoint either (`Messages::result`). Nor does the step's check decide or price
//! anything on a read none of whose looks reads (`Verdict::Unread`, `read_looks`): no rung below and no pairs make a
//! trajectory the model ended count, so the ladder is not walked down on it (tilt2 at 24d346ce: its every start
//! landed nowhere, the pilot walked the ladder to its bottom, 7.96·10⁻⁵ µs, in zero steps, and the result carried
//! t = 0's T to every output with no account -- T's total read 0, "finished").
//!
//! # Stopping (SPEC §3)
//! Every quantity held to a target -- at every analysis time after t = 0 what is shown, and the total T -- has the total
//! E² = A² + c² + S² (ledger.rs; `Look`): A the model's error, from the fold's readings at the pilot's states -- its
//! held atoms' and its lost atoms' paths (# The model: where the atoms went) -- and the optical sector's booking, and c
//! what the step's check leaves open (its rms over the run), both fixed by the pilot -- A with what the run's
//! unresolved atoms carry, read before each batch's looks (# The unresolved atoms) --; S the statistics' interval. The
//! run stops where every one of them lies within its target: the test the ledger flags an output by, at the outputs
//! (`Look::met`). The stop reads the run's standard errors, its unresolved atoms, and nothing else of its estimates. A
//! look that no run's trajectories bring within its target -- its need beyond `MOST_TRAJECTORIES`, or A² + c² ≥ ε²
//! (`Look::fits`): the run's own decisions hold the fold within its share at every point, and what they leave is the
//! model's error booked -- is not chased: the ledger flags it.
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
//! the allowance is smooth in T, and so are the step's reads and the trajectories a look needs. It is T's target wherever T's target enters: the looks and the ledger (`looks`, `accounts`, ledger.rs), the
//! step's reads (estimate.rs `between`: each output's error over its own allowance, the step's share s of it,
//! `step_bound`). The populations and the loss keep their absolute target.
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
//! run's construction, the fold's readings at the states, the prices.
use crate::adapt;
use crate::analysis::analysis_times;
use crate::decide::{Outcome, Read};
use crate::ensemble::{Ensemble, Start, Well, GROUP};
use crate::estimate::{allowance, between, draws_for, half_window, readings, semiclassical, summarize, Between, Estimate, Fate, Group, Held, Scales, Spread};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, significant, span, when, Account, Ledger};
use crate::linalg::{symmetric3_eigen, V3};
use crate::model::fold::run::{Onefold, Reference};
use crate::model::optical::Unmade;
use crate::model::potential::descend;
use crate::model::{Sample, Trap, GROUND};
use crate::request::STATES;
use crate::step::{stumpff, Carry, Dynamics, Refused, Scratch, Segment, Trajectory};
use crate::units::{hbar_over_mass, CLASSICAL, CONFIDENCE, KT_PER_UK, MOST_STEPS, MOST_TRAJECTORIES, ORDER, STEP_SHARE};
use rayon::prelude::*;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::f64::consts::{PI, TAU};
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// The population target the app checks a result against by equality (contract §2.2 `requestedTolerance`).
pub use crate::units::POPULATION_TARGET;
use crate::request::Request;

/// The runs' seed, the key of every trajectory's streams (random.rs): any fixed value, fixed so that a request gives the
/// same result every time.
pub(crate) const SEED: u64 = 0;
/// The output times the distributions are given at (contract §3.4: at most 21, t = 0 and the last among them): the
/// contract's number, for the display alone -- no estimate reads a distribution.
const DISTRIBUTION_TIMES: usize = 21;
/// Half the time the app waits for a message before it kills the worker (contract §2.5: 120 s): the checkpoints'
/// interval. The other half is left to what runs between two messages and sends none (module documentation,
/// # Checkpoints): a margin, not a bound derived from those stages' work. It times messages and decides no result.
const CHECKPOINT: Duration = Duration::from_secs(60);
/// What the engine did with the fold's error (the ledger's `action`, shown by the app): each of its readings read at the
/// pilot's states where the atoms went (module documentation, # The model).
const FOLD_MEASURED: &str = "按原子所在处折叠（一个对象），这一项误差由折叠在先导轨迹的状态上自己读出（原子到过的地方）";

/// The fold's readings a state's error is booked by (model/fold/proto.rs `KINDS`, the accounts' kinds, rate units; module
/// documentation, # The model), each with the ledger's name of it; the state's error is their combination (√Σρ_k²).
const READINGS: [&str; crate::step::READS] = [
    "折叠的级数（H 自洽的下一遍）",
    "分解沿路径的复用",
    "居中（近块的居中拷贝、远块的能级）",
    "略去的结点（激发能级、远块）",
];

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
/// The run is one object, made once for the request and the same at every step of the solve (model/fold/run.rs
/// `Onefold`: the fold at the atom, its structure the points' own): what the solve takes of it beside the steps -- the
/// prepared potentials and their traps, the initial ensemble drawn in them, the reported trap, the kick's curvature
/// (`Setting`) -- is made from it here, and the pipeline runs once (`attempt`). A request it refuses (the run's
/// construction, a trap that does not hold the atom, no step below the edge) is refused with its reason, before a
/// trajectory runs.
pub fn solve(request: &Request, checkpoint: &mut dyn FnMut(&Solution, Option<f64>)) -> Result<Solution, String> {
    request.validate()?;
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request);
    // The reference state (SPEC §2.1): the initial component of the largest population, of equal ones the last.
    let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).expect("a validated request's initial state holds a ground state");
    let reference = alone(initial[chief].0);
    let run = Onefold::new(request, reference)?;
    // The start's decision's report (model/fold/decide.rs), to stderr.
    eprint!("{}", run.report());
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&run, alone(s))).collect::<Result<_, _>>()?;
    let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm)?;
    let trap = &wells[chief].trap;
    let reported = reported(&run, &reference).ok();
    let setting = Setting { run: &run, trap, ensemble: &ensemble, depth: ensemble.depths()[chief], reported, kick: kick_curvature(&run, trap) };
    let mut messages = Messages { send: checkpoint, clock: Instant::now(), last: None, least: 0 };
    let solved = attempt(request, &setting, &mut messages);
    // The run's counts (adapt.rs `tally`): the safety net's readings beyond the share and absent folds, to stderr.
    eprintln!("{}", crate::adapt::counted(true));
    solved
}

/// What the solve takes of its run beside the steps (`solve`), made once.
struct Setting<'a> {
    run: &'a Onefold,
    /// The reference state's prepared trap.
    trap: &'a Trap,
    /// The initial ensemble, every component in its own well, and the depth at the reference state's bottom (k_BT).
    ensemble: &'a Ensemble<'a>,
    depth: f64,
    /// The trap every reported quantity describes (SPEC §1.3: trapFrequencyKHz, n̄, the validated regime and T's
    /// windows): the trap light's potential with the other lights' phase-averaged shifts at the reference state
    /// (model/fold/run.rs `Onefold::reported`). None where it has no minimum the descent reaches -- the atoms are not
    /// held by the light they run in: a mathematical object that does not exist, not a kind of scene --: the prepared
    /// trap's is reported then, and flagged.
    reported: Option<Trap>,
    /// The kick's largest curvature at the trap (`kick_curvature`).
    kick: f64,
}

/// The solve's messages to the app (module documentation, # Checkpoints).
struct Messages<'a> {
    send: &'a mut dyn FnMut(&Solution, Option<f64>),
    /// The last message's time.
    clock: Instant,
    /// The last result sent.
    last: Option<Solution>,
    /// The trajectories a result must rest on to be sent.
    least: u64,
}
impl Messages<'_> {
    /// A result of the solve so far, of `groups` groups and `trajectories` trajectories, with the wall time left (ms)
    /// where the run's pace gives one: made (`made`) and sent where it rests on two groups at least -- the jackknife's
    /// least for a spread, whatever the members a group has so far (the scout's groups have one) --, shows something
    /// beyond t = 0 (`integrated`: some output after it counts a trajectory; else every output after it would be
    /// t = 0's carried, # Exits) and improves on the last one, else the last result again. A result that is not sent is
    /// not made: with no group in there is none to make.
    fn result(&mut self, groups: usize, trajectories: u64, integrated: bool, remaining: Option<f64>, made: impl FnOnce() -> Solution) {
        if groups >= 2 && integrated && trajectories >= self.least {
            let s = made();
            self.least = trajectories;
            // The run's counts so far with each result (adapt.rs `counted`), to stderr; their tally at the solve's end.
            eprintln!("{}", crate::adapt::counted(false));
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

/// The initial components (SPEC §2.5): the ground states the request's initial density holds and their populations
/// normalized over them -- one at least: a validated request's density is diagonal and on the ground states
/// (request.rs `Request::validate`).
pub(crate) fn components(request: &Request) -> Vec<(usize, f64)> {
    let diagonal: Vec<f64> = (0..GROUND).map(|s| request.initial_density[(s, s)].re).collect();
    let total: f64 = diagonal.iter().sum();
    (0..GROUND).filter(|&s| diagonal[s] > 0.).map(|s| (s, diagonal[s] / total)).collect()
}

/// The ground populations of a basis state alone.
pub(crate) fn alone(state: usize) -> [f64; GROUND] {
    std::array::from_fn(|s| if s == state { 1. } else { 0. })
}

/// One initial component's preparation well (ensemble.rs `Well`): its prepared potential -- the trap light's held forms
/// on the run's fold at its ground populations (model/fold/run.rs `Onefold::prepared`), zero at the far level where the
/// trap light is Gaussian beams, the ensemble refusing any other --, its trap (`trap_of`), the trap light's beams and
/// the envelope's peak (`Reference::peak`).
struct Prepared<'a> {
    light: &'a Light,
    potential: Reference,
    trap: Trap,
}
impl<'a> Prepared<'a> {
    fn new(run: &'a Onefold, populations: [f64; GROUND]) -> Result<Prepared<'a>, String> {
        let (light, potential) = (run.light(), run.prepared(&populations));
        let trap = trap_of(run, &potential, &|w| w == light.trap)?;
        Ok(Prepared { light, potential, trap })
    }
}
impl Well for Prepared<'_> {
    fn potential(&self, r: V3) -> (f64, V3) {
        let q = self.potential.at(self.light, r);
        (q.value, q.gradient)
    }
    fn trap(&self) -> (V3, [V3; 3], V3) {
        (self.trap.centre, self.trap.axes, self.trap.frequencies)
    }
    fn envelope(&self) -> (Vec<Geometry>, f64) {
        (self.light.frequencies[self.light.trap].beams.iter().map(|&l| self.light.beams[l]).collect(), self.potential.peak())
    }
}

/// The trap of a potential of the run's light (model/potential.rs `descend`, the one descent): from the shortest length
/// of the beams of the frequencies `lit` (their pattern's scale), within the trap light's region.
fn trap_of(run: &Onefold, potential: &Reference, lit: &dyn Fn(usize) -> bool) -> Result<Trap, String> {
    let light = run.light();
    let k = light.frequencies.iter().enumerate().filter(|(w, _)| lit(*w)).flat_map(|(_, f)| f.beams.iter().map(|&l| light.beams[l].wave_number)).fold(0., f64::max);
    descend(1. / k, run.hbar_over_mass(), &|r| light.outside(r), &|r| potential.rounded(light, r))
}

/// The reported trap of the ground populations p (SPEC §1.3; `Setting::reported`): the trap of the run's reported
/// potential (model/fold/run.rs `Onefold::reported`: the trap light's held forms with the other lights' phase-averaged
/// ones), every light's beams' pattern its scale (`trap_of`).
pub(crate) fn reported(run: &Onefold, p: &[f64; GROUND]) -> Result<Trap, String> {
    trap_of(run, &run.reported(p)?, &|_| true)
}

/// The kick's largest curvature (`stable`), ω_k² = (ħ/m) λ_max ∂∂(V_s − V_ref) at the trap's centre over the ground
/// states s (rad/µs²): how much stiffer than the reference's a state's own prepared well is there (`Onefold::prepared`),
/// exactly. What else the kick's potential curves by -- the folded light's shifts along the pattern, an explicit
/// excited state's well -- is the step's check's to find, as every part of the force is (step.rs, # The kick).
/// Negative where every state's well is the softer one.
fn kick_curvature(run: &Onefold, trap: &Trap) -> f64 {
    let reference = run.reference(trap.centre).hessian;
    (0..GROUND)
        .map(|s| {
            let own = run.prepared(&alone(s)).at(run.light(), trap.centre).hessian;
            let more: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| own[a][b] - reference[a][b]));
            symmetric3_eigen(&more).0.into_iter().fold(f64::NEG_INFINITY, f64::max)
        })
        .fold(f64::NEG_INFINITY, f64::max)
        * run.hbar_over_mass()
}

/// The kinetic temperature along each of a trap's principal axes of a thermal ensemble at `temperature` (µK): T where
/// the trap is classical, the zero point's ħω/2k_B where it is cold -- s_i² T, s_i² = x coth x, x = ħω_i/2k_BT, a
/// harmonic oscillator's mean kinetic energy at that temperature.
fn kinetic_uk(trap: &Trap, temperature: f64) -> V3 {
    let kt = KT_PER_UK * temperature;
    trap.frequencies.map(|omega| {
        let x = 0.5 * omega / kt;
        temperature * x / x.tanh()
    })
}

/// The largest wave number of the beams of every frequency but the trap's (1/µm): the light's pattern an atom crosses
/// at its speed -- the trap's own frequency holds the atom, its pattern the trap's -- (`prior`). None, zero.
fn pattern_wave_number(light: &Light) -> f64 {
    light.frequencies.iter().enumerate().filter(|(w, _)| *w != light.trap).flat_map(|(_, f)| f.beams.iter().map(|&l| light.beams[l].wave_number)).fold(0., f64::max)
}

/// Whether a step of τ is below the edge of its reference's turn and its kick's curvature (module documentation, # The
/// step): the reference flow and the kick between its halves, at the prepared trap's largest frequency ω (rad/µs) and
/// the kick's largest curvature `kick` = ω_k² = (ħ/m) λ_max ∂∂(Ĥ_h − V_ref) (rad²/µs²; `kick_curvature`: what of it
/// the run bounds, the rest the check's to find). About a
/// minimum a step is the linear map R(ωτ/2) K R(ωτ/2), its trace 2 cos ωτ − ω_k²τ² sin(ωτ)/(ωτ) (measured on the
/// crate's step: step/tests.rs), and the motion is bounded while the trace is above −2: on the first stretch, ωτ < π,
/// the trace over two is above −1. It falls with ω and with ω_k², so the largest of each stands for every axis and
/// state. A kick without curvature leaves ω τ < π -- there the two half flows turn the motion by π and any kick between
/// them grows --, a softer well than the reference's (ω_k² < 0) the same; a stiffer one ends the stretch below π,
/// (ωτ/2) tan(ωτ/2) = ω²/ω_k², and on a flat reference at the kick's own ω_k τ = 2. The same for every atom. An edge --
/// where the scheme stops meaning anything --, not an accuracy within a share: beyond it the step multiplies every
/// error by e^μ, cosh μ = |trace|/2, a step after a step (step.rs, # The domain), and how well a step below it does is
/// the check's to say (the pilot's two levels). The light's beats ask no edge: the propagation integrates them exactly
/// (module documentation, # The step).
fn stable(tau: f64, frequency: f64, kick: f64) -> bool {
    let turn = frequency * tau;
    // cos ωτ and sin(ωτ)/(ωτ): the Stumpff functions the reference's flow is made of (step.rs).
    let (cos, sinc, _) = stumpff(turn * turn);
    turn < PI && cos - 0.5 * kick * tau * tau * sinc > -1.
}

/// The step the physics prior allows (module documentation, # The step; the old engine's first step), µs: T's step
/// error, relative, goes as the square of the phase the light's pattern advances along an atom's path in a step,
/// (k v τ)², and the step may take its share of T's allowance (`step_bound`, # T's allowance), k_max v_T τ = √(s δT/T)
/// -- at the pattern's largest wave number `wave` (1/µm; `pattern_wave_number`), the rms speed along an axis at the
/// temperature `temperature` (µK), v_T = √(k_BT/m) -- the initial ensemble's mean kinetic temperature in its trap
/// (`kinetic_uk`), as the old engine's at the initial temperature: a run whose atoms heat past it is the check's to
/// take down the ladder --, and T's allowance there at the
/// scale T_sc of that light's recoil (estimate.rs `semiclassical`: one photon of k_max). No bound where the light has no
/// pattern or the atoms no speed. (PGC3D, Rb87 at 30 µK on D2: k_max v_T = 0.43 rad/µs, T_sc = 10.9 µK, δT/T = 0.375:
/// τ ≤ 0.95 µs, where the ladder's top by the edges alone, 1.425 µs, was left on the pilot and 0.7125 µs stood.)
fn prior(wave: f64, hbar_over_mass: f64, temperature: f64, scales: &Scales) -> f64 {
    let speed = (hbar_over_mass * KT_PER_UK * temperature).sqrt();
    (step_bound()[1] * allowance(temperature, semiclassical([1., wave * wave], scales))).sqrt() / (wave * speed)
}

/// The ladder's first rung (module documentation, # The step): the largest rung -- the run's length `run` (µs) over an
/// even number, its double a whole part of the run too -- that is no longer than the physics prior's step `prior`
/// (`prior`) and is below the step's edge with its double (`stable`): the run's step and its companion's, a probe of the
/// step, not a result, at twice it -- the same edge for every atom, at the trap's largest frequency `frequency` and the
/// kick's largest curvature `kick`. (Both are bounds on the step from above: the least even number is found by
/// counting up, from the prior's.) Where the step's edge binds, the first rung is the rung below the largest stable one.
/// Half the run at the most: the companion's step is a rung itself, the run its one step then. Nothing of the outputs
/// enters: they are read inside the steps (adapt.rs, # The reads). The ladder has one bottom, the step at which a
/// trajectory takes `MOST_STEPS` steps, and the first rung is a rung of it: none where no rung down to that one is
/// stable -- the trap asks for a shorter step than the cap allows, or a frequency or a curvature is no number (nothing
/// is stable then). The solve refuses such a request (`attempt`): a step beyond the edge computes nothing.
///
/// The count of pairs n (the rung run/(2n)) starts where a rung can first be one, each bound read off its condition:
/// within the prior, run/(2n) ≤ prior; the reference's turn, ωτ < π for the rung and for its double, n > run ω/π.
/// Below that n no rung is, so the rung found is the scan's from n = 1.
fn first_rung(run: f64, prior: f64, frequency: f64, kick: f64) -> Option<f64> {
    if !frequency.is_finite() || !kick.is_finite() {
        return None;
    }
    // The prior's bound is the rule itself (nothing else checks it): its ceiling. The edge's decides at each n: its
    // floor starts the count at or below the first n it allows, the rounding of the product taken one before it at most.
    let least = [(run / (2. * prior)).ceil(), (run * frequency / PI).floor()].into_iter().fold(1., f64::max);
    (least <= (MOST_STEPS / 2) as f64)
        .then(|| (least as u64..=MOST_STEPS / 2).map(|pairs| run / (2 * pairs) as f64).find(|&step| stable(step, frequency, kick) && stable(2. * step, frequency, kick)))
        .flatten()
}

/// What the first rung reads (`first_rung`): the run's length, the physics prior's step, the trap's largest frequency
/// and the kick's curvature.
#[derive(Clone, Copy, Debug)]
struct Ladder {
    run: f64,
    prior: f64,
    frequency: f64,
    kick: f64,
}
impl Ladder {
    /// The first rung (`first_rung`).
    fn first(&self) -> Option<f64> {
        first_rung(self.run, self.prior, self.frequency, self.kick)
    }
}

/// What a level's steps ask of the run's domain (step.rs, # The domain): the runaway's rule (`Widest`), and the trap
/// light's region, outside of which a trajectory's loss is tested (`run`: its loss energy read there).
#[derive(Clone, Copy)]
struct Domain<'a> {
    light: &'a Light,
    widest: Widest,
}

/// The fastest decay of the request's excited states (1/µs): no state of the atom scatters faster than that, wholly
/// excited -- a state's hazard above it is a fold far outside its validity (step.rs, # The domain): the runaway's rule,
/// of the run's steps (`Domain`) and of those a price takes (`work_at`).
#[derive(Clone, Copy)]
struct Widest(f64);
impl Widest {
    fn of(request: &Request) -> Widest {
        let decay = crate::model::far::decay(request);
        Widest((0..decay.rows).map(|x| decay[(x, x)].re).fold(0., f64::max))
    }
}
impl crate::step::Domain for Widest {
    fn runaway(&self, hazard: f64) -> bool {
        hazard > self.0
    }
}
impl crate::step::Domain for Domain<'_> {
    /// A hazard above the fastest excited state's decay, wholly excited (`Widest`).
    fn runaway(&self, hazard: f64) -> bool {
        self.widest.runaway(hazard)
    }
}

/// Whether an atom at r with the momentum κ is leaving (module documentation, # Where the atoms went): its prepared
/// energy, (ħ/m)κ²/2 + V_ref(r) of the reference state (the splitting's reference, `Dynamics::reference`), at the
/// potential's far value, nothing, or above -- past it the atom runs out of the trap's region whatever the rates there.
fn leaving<D: Dynamics>(model: &D, r: V3, kappa: V3) -> bool {
    0.5 * model.hbar_over_mass() * kappa.iter().map(|k| k * k).sum::<f64>() + model.reference(r).value >= 0.
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

/// The looks of the family (`judged`) that read: a look no time reads at both levels -- none counts a trajectory
/// there, or of T none holds two atoms (estimate.rs `between`) -- is no look of it. It excludes nothing, and no step
/// makes it read: a trajectory the model ended no shorter step cures (`Fate::Failed`), and one its step ended reads
/// unbounded instead (`Fate::Refused`); nor do pairs (`asks`). So it neither holds a rung nor leaves one.
fn read_looks(read: &Between) -> Vec<(Read, f64)> {
    judged(read).into_iter().filter(|(read, _)| !read.value.is_nan()).collect()
}

/// The screen of a step on the reads between its two levels (module documentation, # The step; decide.rs, # Many
/// reads): whether the family of its reads is beyond -- some read beyond the square of the step's share among all the
/// family's looks that read (`read_looks`). The step is halved where it is.
fn screen(read: &Between) -> bool {
    let reads = read_looks(read);
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

/// Whether a read's every look that reads (`read_looks`) is within the step's share as it reads (module documentation,
/// # The step): its estimate M̂ -- unbiased, the noise's floor taken out -- at most the square b of the share. The
/// convergence test as a refinement is checked by hand and as an adaptive integrator's local control takes it: the
/// change between two levels within the tolerance, on the estimate.
fn within(read: &Between) -> bool {
    read_looks(read).iter().all(|(read, bound)| read.value <= *bound)
}

/// What the check makes of a rung on its reads (module documentation, # The step; `verdict`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Verdict {
    /// Within on its own read and on its third level's -- or, with no third level to be had, on its own two levels, as
    /// an adaptive integrator's local control takes them; on the scout, not clearly beyond: the rung stands on its
    /// reads, and takes its course (`course`).
    Stands,
    /// Its own read clearly beyond the step's share, or its third level's estimate beyond it (the order is not shown
    /// above it): the rung is left.
    Left,
    /// On the pilot, its own read within and no third level on the pilot's pairs, where the ladder has the rung above
    /// and that rung's paths stay in the model's domain: the pilot's trajectories at 4τ, where the rung buys them
    /// (`course`, `buys`).
    Above,
    /// Its own read's estimate beyond the share, but not clearly: the noise hides the difference -- the pairs that
    /// could see through it where they are expected to cost the run less than the rung below (`remedy`), else the
    /// rung is left.
    Unseeing,
    /// No look of its own read reads (`read_looks`): its levels count no trajectory at both at any time read. No pairs
    /// and no rung below make it read, so nothing is decided or priced on it: the check ends unable to see where it is
    /// (`Settled::Unseen`).
    Unread,
}

/// What the check makes of a rung (module documentation, # The step; `settle`) on its own `read` and its third level
/// above carried to its step, `above` (none where it has none); `pilot`: whether the read is the pilot's, the last word
/// on the step (the scout's leaves only on what is clearly beyond -- its own read, or its third level's: a rung whose
/// third level is clearly beyond is left on the scout as on the pilot, and no pairs at the rung can bring that read
/// within); `upward`: whether a third level above on the pilot's own pairs is to be had (the ladder has the rung above,
/// the rung has none yet, and the rung above's read was not unread). Every
/// estimate is taken as it reads (`within`). The third level is the rung the ladder left above, or the pilot at 4τ
/// where the rung buys it (`course`); where none is, the rung stands on its own two levels. What a rung that stands
/// does next is its course's (`course`).
fn verdict(read: &Between, above: Option<&Between>, pilot: bool, upward: bool) -> Verdict {
    if read_looks(read).is_empty() {
        Verdict::Unread
    } else if screen(read) || above.is_some_and(screen) {
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
/// (decide.rs `Read::outcome`). Where it does not, the pairs are asked for what they could do at the best (`asks`):
/// were the look's error the least its read allows -- its near end L at one look, from zero: its own interval, as at
/// its far end (`ends`) --, what the read holds above L, value and interval, is the pairs' noise, and falls as the
/// floor does, as 1/N. The largest factor over the looks that read (`read_looks`: one no time reads asks for nothing
/// pairs could give). (A value taken as it reads -- the rule before -- makes a read that the noise has put above b ask
/// for no pairs, and the rung below was taken on it: measured, SPEC §2.2.)
fn blind(own: &Between) -> Option<f64> {
    read_looks(own).into_iter().filter(|(read, bound)| read.outcome(*bound, 1.) != Outcome::Within).map(|(read, bound)| asks(ends(&read), bound)).reduce(f64::max)
}

/// A look's far and near ends at one look, [F, L] (decide.rs `Read::most`, `Read::least`), the near end from zero:
/// what the read does not exclude, and the least error it allows.
fn ends(read: &Read) -> [f64; 2] {
    [read.most(1.), read.least(1.).max(0.)]
}

/// The pairs a look of the ends [F, L] could read within the bound b on at the best (`blind`), as a factor on those it
/// was read on: the far end on N pairs is L + (F − L)·N₀/N, within b from N = N₀ (F − L)/(b − L) on; on no number where
/// L itself is at b or above, nor where F is no number -- a look that is no number, or whose spread is unread,
/// excludes nothing.
fn asks([far, near]: [f64; 2], bound: f64) -> f64 {
    if far.is_finite() && near < bound { (far - near) / (bound - near) } else { f64::INFINITY }
}

/// The pairs a rung's own read could see on at the best (`blind`), where a run's trajectories can take them: none past
/// `MOST_TRAJECTORIES`, nor on no number (a near end at the share, a look unread).
fn seeing(own: &Between, have: u64) -> Option<f64> {
    let at = blind(own)? * have as f64;
    (at <= MOST_TRAJECTORIES as f64 && have < MOST_TRAJECTORIES).then_some(at)
}

/// What the rung below is foreseen to ask, from the rung's own read `own` on the pairs it rests on (module
/// documentation, ## The ladder; `remedy`): the factor on those pairs that would bring it within at the best (`asks`),
/// none where it is foreseen within on them. Per look, the rung's ends [F, L] at the rung below: the error at the best
/// L lowered by the order, L/2^{2p} (`carried`), and the noise above it, F − L, times the ratio of the noise below to
/// the noise at the rung, measured on the same pairs -- the scout's, at τ/2 against τ (`half`) and at τ against 2τ
/// (`rung`), each standard error the jackknife over the same groups (`foresee`): [L/2^{2p} + r (F − L), L/2^{2p}].
/// Pairs that stay together differ by the step's own error along each atom's path, of the order τ^p, and their noise
/// falls with the error (r = 2^{−2p}); pairs that have parted carry the two ensembles' own, the same at every step
/// (r = 1). The ratio is read, not assumed, and taken within [2^{−2p}, 1], where its expectation lies (module
/// documentation, ## The ladder): it prices the work alone, never the step. A look with no noise at the rung has none
/// below; one whose ratio the scout's pairs do not read (no T where the rung reads one) asks for no number of pairs.
fn foreseen(own: &Between, half: &Between, rung: &Between) -> Option<f64> {
    let lowered = 0.5_f64.powi(2 * ORDER);
    let look = |read: &Read, ratio: f64, bound: f64| -> Option<f64> {
        let [far, near] = ends(read);
        let below = [lowered * near + if far > near { ratio * (far - near) } else { 0. }, lowered * near];
        (!(below[0] <= bound)).then(|| asks(below, bound))
    };
    let ratio = |h: &Read, at: &Read| (h.spread.error / at.spread.error).clamp(lowered, 1.);
    let [sp, st] = step_bound();
    let mut looks = vec![look(&own.loss, ratio(&half.loss, &rung.loss), sp * sp)];
    looks.extend((0..STATES).map(|s| look(&own.shown[s], ratio(&half.shown[s], &rung.shown[s]), sp * sp)));
    looks.extend(own.temperature.map(|t| look(&t, half.temperature.zip(rung.temperature).map_or(f64::NAN, |(h, at)| ratio(&h, &at)), st * st)));
    looks.into_iter().flatten().reduce(f64::max)
}

/// What the step's check weighs its courses by (`course`, `remedy`; module documentation, ## The ladder): `work`, a
/// trajectory's work a µs on the run with every step taken at the step given, each priced at its own step (`work_at`:
/// at the states in hand, the scout's or the pilot's, `one_each`); and `product`, the trajectories the product is to
/// end on as the statistics in hand predict them (`most_needed`: the looks' need, # Stopping) -- the pilot's, or before
/// them the scout's.
struct Prices<'a> {
    work: &'a dyn Fn(f64) -> f64,
    product: f64,
}

/// A trajectory's work a µs at every step at the states `rated` (`one_each`), the mean over the run's held states
/// (`over_the_run`): at each, the step of τ taken from it on the run, as its count is made (step.rs `priced`: its
/// stretches' counts, each with its cluster's counts at a node, `Onefold::costs`) over τ, and its jumps' counted work a
/// µs over its run (`Pick::jumping`: their photons and landings as they were made, step.rs `Trajectory::jumped`) -- a
/// jump's work is the jump's whatever the step, so the step's count holds none of it: a landing's clock counted in the one
/// step a state is priced at made the price a draw of whether that step held a jump (tilt0's scout: w(τ/2) below w(τ) by
/// 5 %, and the ladder went down a rung for it); past any work where no step of τ can be taken from it (refused as the
/// run's would be, the runaway's rule the run's, `Widest`). The splitting's reference is a sum over the trap light's beam pairs, no part of the price. The step's
/// check's `Prices::work`, each step's made once -- a check prices the same steps again as it goes on --, the states'
/// steps on the pool, each by itself.
fn work_at<'a>(run: &'a Onefold, widest: Widest, rated: &'a [Pick<'a>]) -> impl Fn(f64) -> f64 + 'a {
    let made = std::cell::RefCell::new(Vec::<(f64, f64)>::new());
    move |tau: f64| {
        if let Some(&(_, work)) = made.borrow().iter().find(|(at, _)| *at == tau) {
            return work;
        }
        let rates: Vec<f64> = rated
            .par_iter()
            .map(|pick| match crate::step::priced(run, &pick.state.trajectory(), tau, &widest) {
                Ok(stretches) => stretches.iter().map(|(_, price)| price.work(price.costs)).sum::<f64>() / tau + pick.jumping,
                Err(_) => f64::INFINITY,
            })
            .collect();
        let work = over_the_run(rated, rates.into_iter(), None);
        made.borrow_mut().push((tau, work));
        work
    }
}

/// What the run holds of a rung's levels where its check takes its course (`course`, `remedy`; module documentation,
/// ## The ladder), in trajectories: the pilot's at each of the rung's two levels once its groups are whole, `pilot`;
/// those of them run at both, `members` (the scout's, one a group, or all the pilot's); the pairs at the rung, `pairs`
/// (those and the further pairs); and the rung below's level already run, `below` (the scout's pairs at τ/2,
/// `foresee`).
#[derive(Clone, Copy, Debug)]
struct Holding {
    pilot: f64,
    members: f64,
    pairs: f64,
    below: f64,
}

/// The run's expected work from here where it stays at the rung τ in hand (module documentation, ## The ladder): the
/// pilot's companions still to run at 2τ, and its run level at τ to the most of the product's trajectories, the
/// pilot's and the pairs in hand (the further pairs' run level is the product's), (P − P_m) w(2τ) + (max(N, P, n) − n)
/// w(τ). The third level above is the course's to add where the rung buys it (`course`).
fn staying(prices: &Prices, tau: f64, holding: Holding) -> f64 {
    let Holding { pilot, members, pairs, .. } = holding;
    (pilot - members) * (prices.work)(2. * tau) + (prices.product.max(pilot).max(pairs) - pairs) * (prices.work)(tau)
}

/// The run's expected work from here at the least of the rungs below the rung τ in hand (module documentation,
/// ## The ladder); without bound where there is none -- the ladder's bottom is the step at which a trajectory of the
/// run's length `end` takes `MOST_STEPS` steps (`check`). The ladder goes down a rung at a time, each rung's own level
/// its next one's companion, so the k-th rung below, τ_k = τ/2^k, costs the members in hand at every rung on the way,
/// the pilot's companions at τ_{k−1}, and its run level at τ_k to the more of the product's trajectories and the
/// pilot's, less the rung below's level already run: P_m Σ_{j≤k} w(τ_j) + (P − P_m) w(τ_{k−1}) + (max(N, P) − P_m)
/// w(τ_k) − r w(τ_1) -- the pairs at the rung are no level of a rung below. Every term past the first k is no less than
/// zero, so no rung at or below τ_k costs less than P_m Σ_{j≤k} w(τ_j) − r w(τ_1): the rungs are priced down to
/// where that is no less than the least found, or to the bottom.
fn descending(prices: &Prices, tau: f64, end: f64, holding: Holding) -> f64 {
    let Holding { pilot, members, below, .. } = holding;
    let run = prices.product.max(pilot);
    let (mut least, mut on_the_way, mut nearest) = (f64::INFINITY, 0., None);
    let (mut step, mut above) = (tau, (prices.work)(tau));
    while end / (0.5 * step) <= MOST_STEPS as f64 {
        step *= 0.5;
        let here = (prices.work)(step);
        on_the_way += members * here;
        let floor = on_the_way - below * *nearest.get_or_insert(here);
        if !(floor < least) {
            break;
        }
        least = least.min(floor + (pilot - members) * above + (run - members) * here);
        above = here;
    }
    least
}

/// Whether a rung buys its third level above, the pilot's trajectories at 4τ (module documentation, ## The ladder): at
/// its price P w(4τ) no more than what the least rung below adds to staying, at the pilot's whole `holding` -- the
/// third level shows the order above the rung by its own pair (2τ, 4τ), a rung below shows it by this rung's read
/// (`carried`), and the third is worth no more than that. At the ladder's bottom a rung below adds without bound.
fn buys(prices: &Prices, tau: f64, end: f64, holding: Holding) -> bool {
    holding.pilot * (prices.work)(4. * tau) <= descending(prices, tau, end, holding) - staying(prices, tau, holding)
}

/// A rung's course where its reads stand it (module documentation, ## The ladder; `course`, `check`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Course {
    /// It stands on its levels: its own two, or those and its third level above.
    Stand,
    /// Its third level above is read first, the pilot's trajectories at 4τ: bought (`buys`).
    Third,
    /// The rung below: some rung below is expected to cost the run less from here (`descending`).
    Below,
}

/// A rung's course where its reads stand it -- not clearly beyond on the scout, within on the pilot -- by the run's
/// expected work from here (module documentation, ## The ladder), at the run's length `end` and what the run holds of
/// the rung's levels (`holding`); `third`: whether the rung has a third level above to read (the ladder has the rung
/// above, the rung none yet, and the rung above's paths stay in the model's domain). The rung below where some rung
/// below costs less than staying (`descending`, `staying`); else its third level where the rung buys it (`buys`, at
/// the pilot's position: on the scout the pilot's purchase to come, priced into staying -- the course there is the
/// pilot's to take); else it stands on its own two levels.
fn course(prices: &Prices, tau: f64, end: f64, holding: Holding, third: bool) -> Course {
    let whole = Holding { members: holding.pilot, pairs: holding.pairs.max(holding.pilot), ..holding };
    let bought = third && buys(prices, tau, end, whole);
    let stay = staying(prices, tau, holding) + if bought { holding.pilot * (prices.work)(4. * tau) } else { 0. };
    if descending(prices, tau, end, holding) < stay {
        Course::Below
    } else if bought {
        Course::Third
    } else {
        Course::Stand
    }
}

/// The rung below as a rung that cannot see foresees it (`remedy`): the factor on the pairs in hand that would bring
/// it within at the best (`foreseen`; none where it is foreseen within on them), and the trajectories of its level
/// already run -- the scout's pairs at τ/2 (`foresee`).
#[derive(Clone, Copy)]
struct Below {
    asks: Option<f64>,
    ran: u64,
}

/// The pairs a rung's own read asks for where its estimate lies beyond the step's share but not clearly -- the noise
/// hides the difference, or the error lies near the share (module documentation, ## The ladder; `Verdict::Unseeing`)
/// --, a batch of them at a time (`batch`: at most twice the `have` pairs the read rests on) and read again; or none:
/// the rung at τ is left for the rung `below` (none at the ladder's bottom, where the check then ends unable to see,
/// SPEC §5.1).
///
/// The pairs at the rung, N_τ = N₀ (F − L)/(b − L) at the best (`seeing`), wherever a run's trajectories can take them;
/// the rung below's, N_{τ/2}, those its foreseen ends ask (`Below::asks`: the error lowered by the order, the noise as
/// the scout's pairs read it there, `foreseen`). Each is priced at the run's expected work from here (`prices`: a
/// trajectory's work a µs at each step, the product's trajectories N_p), the pilot's `pilot` pairs the ones a rung
/// below keeps -- its level at τ becomes the companion of its own at τ/2, and the further pairs are freed:
/// - at the rung: the further pairs' companions at 2τ, its run level at τ beyond the `have` in hand -- the pairs' or
///   the product's, whichever are more: the further pairs' run level is the product's --, and where it has a third
///   level above to read on the pilot (`third`: it has none, and the ladder has the rung above) and buys it on the
///   pairs it then holds (`buys`, at the run's length `end`), the pilot at 4τ;
/// - below: the run level at τ/2 -- the pilot's, the pairs' or the product's, whichever are more -- beyond the
///   trajectories it has run (`Below::ran`), and the further pairs' companions at τ beyond the pilot's.
/// The rung below is taken where it is expected to cost less; else the pairs, as where no rung below is. Where no
/// pairs can be had, none: the rung below. What the rung below is foreseen to read decides the work alone, never the
/// step: the rung below stands only on its own read (`check`).
#[allow(clippy::too_many_arguments)]
fn remedy(own: &Between, have: u64, pilot: u64, below: Option<Below>, prices: &Prices, tau: f64, end: f64, third: bool) -> Option<u64> {
    let (n, pilot) = (have as f64, pilot as f64);
    let at = seeing(own, have)?;
    let pairs = batch(at, have);
    let Some(below) = below else { return Some(pairs) };
    let (work, product) = (prices.work, prices.product);
    let half = below.asks.map_or(0., |factor| factor * n);
    let third = third && buys(prices, tau, end, Holding { pilot, members: pilot, pairs: at, below: below.ran as f64 });
    let stay = (at - n) * work(2. * tau) + (at.max(product) - n) * work(tau) + if third { pilot * work(4. * tau) } else { 0. };
    let down = (product.max(half).max(pilot) - below.ran as f64) * work(0.5 * tau) + (half - pilot).max(0.) * work(tau);
    (stay <= down).then_some(pairs)
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
/// levels -- and the pairs it was read on (the scout's eleven where the scout went down a rung, the pilot's at 4τ), the
/// pairs of the rung's own two levels, and the check's resolution -- the rms error over the run
/// that the read booked cannot tell from none, √(t·SE) of its mean square, the largest on what is shown (absolute) and
/// on T (a share of its allowance; unbounded where it is not read); and whether the check saw (the rung stands on its
/// estimates) or ended unable to see, capped or not.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Checked {
    third: Option<(f64, u64)>,
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

/// The step's part of the accounts as far as a rung's reads hold it (`attempt`: the product's need the check is priced at): per
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

/// Whether some groups show anything beyond t = 0 (module documentation, # Exits): some output after it counts a
/// trajectory -- its atom held or lost there under the run's model. Where none does, every trajectory ended with no
/// outcome before the first output after t = 0: nothing was integrated, and every output after it would be t = 0's
/// carried (`assemble`).
fn integrated(parts: &[&Part]) -> bool {
    parts.iter().any(|p| p.sums.sums.iter().skip(1).any(|s| s.trajectories > 0.))
}

/// The request's exit where nothing is integrated (module documentation, # Exits): the pilot's `trajectories`, every
/// one ended with no outcome before the first output after t = 0, `first` (µs), and why (`ended`).
fn nothing_integrated(counts: &Counts, trajectories: u64, first: f64) -> String {
    format!("没有一条轨迹被积分：先导的 {trajectories} 条轨迹都在 t = 0 之后的第一个输出时刻（{}）之前没有结果地结束，t = 0 之后没有可给的结果（{}）", when(first), ended(counts, false).join("；"))
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
    messages.result(all.len(), n, integrated(&all), None, || {
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

/// What the run's unresolved atoms add to the model's error at every time (SPEC §4; module documentation, # The
/// unresolved atoms), whole: their share of the held atoms on what is shown (absolute: by the held share 1 − L, as
/// `Measured::means` books the held atoms), their share of the kinetic energy on T (relative; estimate.rs
/// `Estimate::unresolved`).
fn apart(estimates: &[Estimate]) -> Vec<[f64; 2]> {
    estimates.iter().map(|e| [(1. - e.loss) * e.unresolved[0], e.unresolved[1]]).collect()
}

/// Whether a lab axis's temperature (µK) lies in the validated regime of that axis's frequency (rad/µs), k_BT ≥
/// `CLASSICAL` ħω (SPEC §0, §3): what the result reports as classical, per axis; reporting only.
fn classical(temperature: f64, frequency: f64) -> bool {
    KT_PER_UK * temperature >= CLASSICAL * frequency
}

/// A state's model error as the run books it (`Measured::means`): on the populations (absolute) and on T (relative)
/// what one of its fold's readings books there.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Booking {
    error: [f64; 2],
}
impl Booking {
    const NONE: Booking = Booking { error: [0.; 2] };
    const WHOLE: Booking = Booking { error: [1.; 2] };

    /// A reading x at a state (rate units: a rate's relative error, model/fold/proto.rs `Cache::profile`): a rate wrong by x
    /// moves a population by x at most, absolute, and T by x, relative. One that is no number or without bound -- a
    /// fold read where it does not exist -- is the state's whole, as an unresolved atom is booked: every row of the atom
    /// (a probability's range) and its whole part of T.
    fn of(x: f64) -> Booking {
        if x.is_finite() { Booking { error: [x; 2] } } else { Booking::WHOLE }
    }

    /// The error of a lost atom's fate (module documentation, # Where the atoms went): a reading's mean along the path
    /// its fate was decided along, by the trapezoid on the states the run holds of it there (`nodes`: their times,
    /// increasing from t = 0, and their errors) -- one state, its own: an atom lost errs by its fate.
    fn path(nodes: &[(f64, Booking)]) -> Booking {
        let span = nodes.last().map_or(0., |n| n.0) - nodes.first().map_or(0., |n| n.0);
        if !(span > 0.) {
            return nodes.first().map_or(Booking::NONE, |n| n.1);
        }
        let sum = nodes.windows(2).fold([0.; 2], |s, w| std::array::from_fn(|i| s[i] + 0.5 * (w[1].0 - w[0].0) * (w[0].1.error[i] + w[1].1.error[i])));
        Booking { error: sum.map(|x| x / span) }
    }
}

/// The end of the path a lost atom's fate was decided along (module documentation, # Where the atoms went): of its
/// checkpoints `trail` (at t = 0 and every window on, `run`) the last whose state is not leaving -- its prepared energy
/// below the potential's far value (`leaving`) --, the first where every one is. Its last state that the dynamics still
/// decide the fate of: past the far value the atom runs out of the trap's region whatever the rates there, and is found
/// lost where it leaves it.
fn approached(trail: &[Checkpoint], leaving: &dyn Fn(&Trajectory) -> bool) -> usize {
    trail.iter().rposition(|c| !leaving(&c.trajectory)).unwrap_or(0)
}

/// The states a lost atom's path is read at (`sampled`, `Booking::path`), in their times' order, each with its place
/// among the held states `samples` (per analysis time, the first t = 0's) where it is one of them: its state at t = 0,
/// its held states at the analysis times before the time `first` it is first lost at and before its path's end, and
/// the path's end `end` (`approached`) where it is later than all of them. None where the run holds no state of it at
/// t = 0.
fn held_path<'a>(samples: &'a [Vec<State>], first: usize, end: &'a State) -> Option<Vec<(&'a State, Option<(usize, usize)>)>> {
    let start = samples.first()?.iter().find(|x| x.index == end.index)?;
    let mut along = vec![(start, None)];
    for (k, at) in samples.iter().enumerate().take(first).skip(1) {
        if let Some(i) = at.iter().position(|x| x.index == end.index && x.sample.at.t < end.sample.at.t) {
            along.push((&at[i], Some((k, i))));
        }
    }
    if end.sample.at.t > along[along.len() - 1].0.sample.at.t {
        along.push((end, None));
    }
    Some(along)
}

/// The fold's readings at a state as the books take them (module documentation, # The model): the books' profile at its
/// time -- per account (`READINGS`) the reading its cluster's last decision left on the trajectory's representation,
/// what each of its steps books over the cell (unify2 §2.7; model/fold/run.rs `Onefold::profile`; step.rs
/// `Trajectory::model`), no forward made -- and their combination (√Σρ_k², the books' whole), each booked as a state's
/// error is (`Booking::of`); the state's whole where its representation holds no decision on its cluster.
fn booked_at(run: &Onefold, state: &State) -> Vec<Booking> {
    match run.profile(state.sample.cluster, &state.held) {
        Some(profile) => {
            let mut out: Vec<Booking> = profile.into_iter().map(Booking::of).collect();
            // A cell with an unresolved step (unify2 §2.7): that account reads ρ = 1 at its states, said as it is.
            if state.unresolved {
                *out.last_mut().expect("the combination") = Booking::WHOLE;
            }
            out
        }
        None => vec![Booking::WHOLE; READINGS.len() + 1],
    }
}

/// The states the model is read at and the fold's readings there (SPEC §5.2; module documentation, # Where the atoms
/// went), at the analysis times the groups `parts` are read at (`Part::at`). At every analysis time after t = 0, of
/// every group:
/// - every member that holds its atom there: the error of its rows, read at its state there;
/// - and every member lost by then: the error of its fate, its readings along the path its fate was decided along, read
///   at the states the run holds of that path -- at t = 0, at the analysis times it was held at before the path's end,
///   and at the path's end, its last state below the potential's far value (`approached`) -- by the trapezoid
///   (`Booking::path`); never where it was found lost, beyond the trap's region past the far value.
/// At t = 0 no model has acted (`model_error`): its states are read as a lost atom's path's start alone. A state is read
/// once -- a held atom's for its time, and as a node of its path where it is lost later --, and a lost atom's path is
/// made of its states' rows, no read of its own, for every time after its loss: the reads are the table's rows by
/// number (`Sampled::row`).
///
/// The errors are one table (`Measured::errors`): the rows one after another, a row the readings' errors and their
/// combination at a state (`booked_at`), each state's row written on the pool where it is read, the paths' rows after
/// them. One block of memory, not one a state: the states are thousands, and as many blocks took the calling thread 0.2
/// to 0.6 s to free.
struct Sampled {
    errors: Vec<Booking>,
    /// Each read's analysis time (its place among them), its row, and whether it reads a lost atom's fate.
    at: Vec<usize>,
    row: Vec<usize>,
    lost: Vec<bool>,
}
impl Sampled {
    /// The fold's errors as the decisions read them (`Measured`).
    fn measured(&self) -> Measured<'_> {
        Measured { errors: &self.errors, output: &self.at, row: &self.row, lost: &self.lost }
    }
}

fn sampled(run: &Onefold, parts: &[&Part]) -> Sampled {
    // The states read -- a held atom's at a time, a lost atom's at t = 0 and at its path's end --, and per lost atom its
    // path's nodes (their times and rows); per group its held states' rows by time and its lost atoms' first times lost
    // and paths.
    let mut rows: Vec<&State> = vec![];
    let mut paths: Vec<Vec<(f64, usize)>> = vec![];
    let mut layout: Vec<(Vec<Vec<usize>>, Vec<(usize, usize)>)> = vec![];
    for p in parts {
        let mut held: Vec<Vec<usize>> = vec![vec![]; p.samples.len()];
        for (k, s) in p.samples.iter().enumerate().skip(1) {
            for state in s {
                held[k].push(rows.len());
                rows.push(state);
            }
        }
        let mut lost_here = vec![];
        for (first, end) in &p.lost {
            let Some(along) = held_path(&p.samples, *first, end) else { continue };
            let mut nodes = vec![];
            for (state, place) in along {
                let r = match place {
                    Some((k, i)) => held[k][i],
                    None => {
                        rows.push(state);
                        rows.len() - 1
                    }
                };
                nodes.push((state.sample.at.t, r));
            }
            lost_here.push((*first, paths.len()));
            paths.push(nodes);
        }
        layout.push((held, lost_here));
    }
    let states = rows.len();
    let (mut at, mut row, mut lost) = (vec![], vec![], vec![]);
    for (held, lost_here) in &layout {
        for (k, h) in held.iter().enumerate().skip(1) {
            for &r in h {
                (at.push(k), row.push(r), lost.push(false));
            }
            for &(_, j) in lost_here.iter().filter(|(first, _)| *first <= k) {
                (at.push(k), row.push(states + j), lost.push(true));
            }
        }
    }
    let width = READINGS.len() + 1;
    let mut errors = vec![Booking::NONE; (states + paths.len()) * width];
    // A state's row, then each path's of its nodes' rows.
    let (read_rows, path_rows) = errors.split_at_mut(states * width);
    read_rows.par_chunks_mut(width).zip(&rows).for_each(|(out, state)| out.copy_from_slice(&booked_at(run, state)));
    for (nodes, out) in paths.iter().zip(path_rows.chunks_mut(width)) {
        for (u, out) in out.iter_mut().enumerate() {
            *out = Booking::path(&nodes.iter().map(|&(t, r)| (t, read_rows[r * width + u])).collect::<Vec<_>>());
        }
    }
    Sampled { errors, at, row, lost }
}

/// The solve's pipeline on its run (module documentation, # The pipeline): the scout, the pilot and the step's check,
/// the product, the result.
fn attempt(request: &Request, setting: &Setting, messages: &mut Messages) -> Result<Solution, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let Setting { run, trap, ensemble, depth, kick, .. } = *setting;
    let (report, unheld) = (setting.reported.as_ref().unwrap_or(trap), setting.reported.is_none());
    let scales = Scales { hbar_over_mass: hm, frequencies: report.lab };
    let plan = Plan::new(&request.times_us, report.lab);
    let row: [f64; STATES] = std::array::from_fn(|s| request.initial_density[(s, s)].re);
    // The ensemble's proposals so far that read deeper than their well's bottom.
    let deeper = ensemble.deeper();
    let mut ledger = Ledger::default();
    ledger.note = Some(format!("初始系综：每个初态分量的原子束缚在它自己的势阱里（降低的等温分布 f ∝ e^(−E/k_BT) − 1，只含束缚原子，阱深比 k_BT 大得多时即热分布；每个主轴再乘谐振子的 Wigner 因子，T = {} μK）；布居最大的初态的阱深为 {:.1} k_BT", request.temperature_uk, depth));
    if unheld {
        ledger.flag("计算所用光场中的势没有极小：阱频、n̄、温度的窗口（一个阱周期）与验证范围都按初始势阱给出".into());
    }
    // The ladder's first rung (# The step): within the physics prior's step at the initial ensemble's mean kinetic
    // temperature in the reference state's trap (`kinetic_uk`), and below the step's edge.
    let kinetic = kinetic_uk(trap, request.temperature_uk).iter().sum::<f64>() / 3.;
    let (frequency, end) = (trap.frequencies.iter().copied().fold(0., f64::max), plan.end);
    let ladder = Ladder { run: end, prior: prior(pattern_wave_number(run.light()), hm, kinetic, &scales), frequency, kick };
    let first_rung = ladder
        .first()
        .ok_or_else(|| format!("找不到可用的时间步长：在每条轨迹的步数上限（{MOST_STEPS} 步）之内，没有步长能分辨阱中的振荡和光场中势阱的曲率（或者阱频、曲率不是有限数）"))?;
    // Whether the rung above a step is a rung of the ladder (`first_rung`'s test at twice the step): the check then
    // reads that rung on a third level, at 4τ (`settle`): a cell of the Brownian path.
    let above = |tau: f64| 4. * tau <= end && stable(2. * tau, frequency, kick) && stable(4. * tau, frequency, kick);
    // What the steps ask of the run's domain (`Domain`).
    let domain = Domain { light: run.light(), widest: Widest::of(request) };
    // The analysis times' candidates (# The analysis times): every level is read there while the pilot runs.
    let (candidates, window) = (grid(end, first_rung), plan.window);
    let mut context = Context {
        at: vec![0., end],
        fixed: Fixed { model: vec![[0.; 2]; 2] },
        unresolved: vec![[0.; 2]; 2],
        step: Bound::share(),
        checked: None,
        tau: first_rung,
        plan: Plan { cell: if above(first_rung) { 4. } else { 2. } * first_rung, ..plan },
        scales,
        initial: row,
        ledger,
        method: "三维半经典运动：内态按原子所在处的光折叠（基态全部保留，激发态在每个判断时刻按折叠自己的读数升为显式或降回折叠）".into(),
    };
    context.plan.read_at(candidates.clone());

    // The pilot's groups (module documentation, # The pipeline): the run's first trajectories, at two levels.
    let first = MOST_TRAJECTORIES / GROUP;
    let numbers: Vec<u64> = (first..first + pilot_size() / GROUP).collect();
    let scales = context.scales;
    // A run level's analysis times (# The analysis times: its own readings).
    let analysed_alone = |level: &[&Part]| analysed(&candidates, window, &reads_of(level), &[], &scales);
    let mut levels = Levels::default();

    // The scout (module documentation, # The pipeline): the first member of every pilot group at both levels, screened.
    // Where its rung stands on its reads, its course by the run's work from here, priced at its states at its analysis
    // times -- the rung, or one below where that is expected to cost the run less: the product's trajectories its looks
    // ask beside the fold's readings at those states (`Prices`).
    let scouting = settle(run, ensemble, &domain, &mut context, &mut levels, &numbers, 0..1, &above, messages);
    if matches!(scouting, Settled::Pending) {
        let here = {
            let scout: Vec<&Part> = levels.fine.iter().collect();
            parts_at(run, &domain, context.tau, &context.plan, &scout, &analysed_alone(&scout))
        };
        let refs: Vec<&Part> = here.iter().collect();
        let estimates = summarize(&reads_of(&refs), &scales);
        let read = sampled(run, &refs);
        let need = most_needed(&looks(&estimates, &Fixed { model: model_error(&read.measured().combined(estimates.len(), &estimates)) }, Bound::share()));
        let rated = one_each(&refs);
        let work = work_at(run, domain.widest, &rated);
        check(run, ensemble, &domain, &mut context, &mut levels, &numbers, 1, &above, Some(&Prices { work: &work, product: need }), messages);
    }
    // The pilot: its groups' other members at both levels, screened, and the step's check (module documentation,
    // # The step) -- first as far as its own two levels: at the first rung whose own read is in and not clearly beyond,
    // its states are read, and the check goes on from there priced at them. Whether it saw: a step that stands
    // uncertified -- at the steps' cap, or unable to see -- is flagged.
    let mut settled = settle(run, ensemble, &domain, &mut context, &mut levels, &numbers, 1..GROUP, &above, messages);
    // Nothing integrated (# Exits): the pilot's every trajectory ended with no outcome before the first output after
    // t = 0 -- no step, no rung and no further trajectory is checked or priced on what shows nothing --, the request's
    // exit with its reason.
    {
        let pilot: Vec<&Part> = levels.fine.iter().collect();
        if !integrated(&pilot) {
            return Err(nothing_integrated(&counts_of(&pilot), pilot.len() as u64 * GROUP, context.plan.times[1]));
        }
    }
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
                let pilot = parts_at(run, &domain, tau, &context.plan, &level, &at);
                let read = pilot_reads(run, &pilot.iter().collect::<Vec<_>>(), &scales);
                (at, pilot, read)
            }
        };
        if !matches!(settled, Settled::Pending) {
            break (at, pilot, read);
        }
        // The check goes on priced (`Prices`): its courses -- stand, the third level above, a rung below, more pairs
        // where the noise hides its rung's difference -- by the run's expected work, a trajectory's work a µs at each
        // step at the pilot's states (`work_at`), and the product's trajectories its looks ask beside the step's part
        // as far as the rung's own reads hold it (`standing`).
        let step = {
            let (own, third, _, _) = levels.told(&candidates, window, tau, &scales);
            standing(&worse(&own, third.as_ref(), false))
        };
        let product = most_needed(&looks(&read.estimates, &Fixed { model: read.model.clone() }, step));
        let (_, _, pilot, _) = &*made.insert((tau, at, pilot, read));
        let rated = one_each(&pilot.iter().collect::<Vec<_>>());
        let work = work_at(run, domain.widest, &rated);
        settled = check(run, ensemble, &domain, &mut context, &mut levels, &numbers, GROUP, &above, Some(&Prices { work: &work, product }), messages);
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
    let tau = context.tau;
    // The step's check (module documentation, # The step): what the reads the rung stands by do not exclude into the
    // accounts, their resolution into the result's words (`Levels::told`, `checked`).
    let (own, third, step_of_third, pairs) = levels.told(&candidates, window, tau, &scales);
    let (step, resolution) = checked(&own, third.as_ref(), saw);
    (context.step, context.checked) = (step, Some(Checked { third: step_of_third, pairs, resolution, saw }));
    // The decisions stand on the pilot's analysis times from here on: the product reads its trajectories there, and
    // the run's model error is the pilot's there (# The analysis times). The levels' checkpoints are read no more;
    // the further pairs' groups are the product's, at the same times.
    let times: Vec<f64> = at.iter().map(|&k| candidates[k]).collect();
    let further: Vec<Part> = levels.further[0].iter().map(|p| p.at(&at)).collect();
    drop(levels);
    context.plan.read_at(times.clone());
    let n = times.len();
    context.at = times;
    context.unresolved = vec![[0.; 2]; n];
    context.fixed.model = read.model.clone();
    // The optical sector's noise (ledger.rs `optical`), its items; its booking on T is in the model's error the reads
    // made (`pilot_reads`).
    crate::ledger::optical(&mut context.ledger, read.counts.optical);
    {
        // The ledger's items: each of the fold's readings' error on what is shown and on T (its largest over the
        // analysis times, weighed by the pilot's held share as the totals are; a time where the pilot holds no atom
        // weighs nothing: its errors are their bounds, `Measured::means`).
        let held: Vec<f64> = read.estimates.iter().map(|e| 1. - e.loss).collect();
        let means = read.sampled.measured().means(n, &read.estimates);
        for (i, label) in READINGS.iter().enumerate() {
            let most = (1..n).filter(|&k| held[k] > 0.).map(|k| means[k][i][0].max(held[k] * means[k][i][1])).fold(0., f64::max);
            context.ledger.item(label, most, FOLD_MEASURED);
        }
    }

    // The product (module documentation, # Stopping): on from the pilot's trajectories at the run's step, the run's
    // first; the pilot's companions have done their part, and its states have been read.
    let mut parts = pilot;
    forget(&mut parts);
    parts.extend(further);
    let (e, l, c, reads) = product(run, ensemble, &domain, &mut context, &mut parts, &numbers, messages);
    // A point of the prepared potential below the bottom the ensemble was drawn about (ensemble.rs, # Its draw).
    if ensemble.deeper() > deeper {
        context.ledger.flag("初态的势在所取的阱底之外还有更深的地方：初始系综只取到该阱底的阱深为止".into());
    }
    Ok(assemble(&context, &e, &reads, &l, &c, parts.len() as u64 * GROUP))
}

/// The pilot's reads at the rung it stands at (`attempt`, `pilot_reads`): its estimates and counts, the fold's readings
/// at its states (`Sampled`), and the model's error per time -- the readings' combination and the optical sector's
/// booking on T.
struct PilotRead {
    estimates: Vec<Estimate>,
    counts: Counts,
    sampled: Sampled,
    model: Vec<[f64; 2]>,
}

/// The pilot's reads (`PilotRead`) of its groups `pilot`:
/// - the model's error at the pilot's states (SPEC §5.2; # The model): the fold's readings' combination per time, the
///   held atoms' mean and the lost atoms' along their paths (`Measured::combined`);
/// - the optical sector's noise (ledger.rs `optical`): its booking on T at every time, a part of the model's error.
fn pilot_reads(run: &Onefold, pilot: &[&Part], scales: &Scales) -> PilotRead {
    let (estimates, counts) = (summarize(&reads_of(pilot), scales), counts_of(pilot));
    let sampled = sampled(run, pilot);
    let mut model = model_error(&sampled.measured().combined(estimates.len(), &estimates));
    let optical = crate::ledger::optical(&mut Ledger::default(), counts.optical);
    for e in model.iter_mut().skip(1) {
        e[1] += optical;
    }
    PilotRead { estimates, counts, sampled, model }
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
/// scout's where the scout did, or the pilot's trajectories at 4τ (`settle`) --, and the pairs it was read on
/// (`above_pairs`: the scout's, the pilot's, or with the further pairs); and whether the rung above the rung in
/// hand was read and is unread (`above_unread`: its read excludes nothing on some look).
///
/// And where the rung in hand could not see and the rung below was weighed against more pairs (`foresee`), the scout's
/// pairs at τ/2 -- the rung below's level of the first member of every pilot group, its states kept -- and the two
/// reads on those pairs, (τ/2, τ) and (τ, 2τ) (`half`).
#[derive(Default)]
struct Levels {
    fine: Vec<Part>,
    coarse: Vec<Part>,
    further: [Vec<Part>; 2],
    above: Option<Between>,
    above_pilot: bool,
    above_pairs: u64,
    above_unread: bool,
    half: Option<(Vec<Part>, Between, Between)>,
}
impl Levels {
    /// The finer level's groups and their companions: the pilot's, then the further pairs'.
    fn paired(&self) -> (Vec<&Part>, Vec<&Part>) {
        (self.fine.iter().chain(&self.further[0]).collect(), self.coarse.iter().chain(&self.further[1]).collect())
    }

    /// The reads the check of the rung in hand, at the step `tau`, is told and booked by (`checked`, `Checked`): its
    /// own, between τ and 2τ on all its pairs; its third level's carried to τ (`carried`), the level above's -- none
    /// where it has none --; the third level's step and the pairs it was read on, and the own read's pairs.
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
    fn told(&self, times: &[f64], window: f64, tau: f64, scales: &Scales) -> (Between, Option<Between>, Option<(f64, u64)>, u64) {
        let (fine, coarse) = self.paired();
        let (own, pairs) = (read_step(times, window, &fine, &coarse, scales), fine.len() as u64 * GROUP);
        (own, self.above.clone(), self.above.as_ref().map(|_| (4. * tau, self.above_pairs)), pairs)
    }
}

/// The members `members` of the groups `numbers` at the step `tau`, each trajectory through the plan in the `domain`
/// (a companion's level takes `Domain::companion`): a hand-out of its own (`run_groups`).
#[allow(clippy::too_many_arguments)]
fn level(run: &Onefold, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Part> {
    let hand = Hand { groups: numbers, members, shape: Shape::of(plan, keep), record: &|i| trajectory(run, ensemble, domain, tau, plan, keep, i) };
    run_groups(&[hand], messages, progress).pop().expect("the hand-out's one level")
}

/// The members `members` of the groups `numbers` at the step `tau` (their states kept where `keep`) and at twice it,
/// the two levels of a check (module documentation, # The step): one hand-out (`run_groups`) -- a check runs both
/// whatever either shows --, the finer level first, whose trajectories take twice the steps; the coarser one is the
/// companion's (`Domain::companion`). `progress` takes the finer level's groups.
#[allow(clippy::too_many_arguments)]
fn pair(run: &Onefold, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> [Vec<Part>; 2] {
    let fine = Hand { groups: numbers, members: members.clone(), shape: Shape::of(plan, keep), record: &|i| trajectory(run, ensemble, domain, tau, plan, keep, i) };
    let coarse = Hand { groups: numbers, members, shape: Shape::of(plan, false), record: &|i| trajectory(run, ensemble, domain, 2. * tau, plan, false, i) };
    let mut levels = run_groups(&[fine, coarse], messages, progress).into_iter();
    std::array::from_fn(|_| levels.next().expect("the hand-out's two levels"))
}

/// The states and checkpoints kept in some groups' parts freed, on the pool, a group's by itself: each is a block of
/// memory of its own -- a pilot's states one for every trajectory at every analysis time, and one more for every atom it
/// lost; its checkpoints one for every trajectory and window of the run.
fn forget(parts: &mut [Part]) {
    parts.par_iter_mut().for_each(|part| (part.samples, part.trails, part.lost) = (vec![], vec![], vec![]));
}

/// Some groups' further members joined to those a level holds of them (all of them where it holds none).
fn absorb(level: &mut Vec<Part>, more: Vec<Part>) {
    if level.is_empty() {
        *level = more;
    } else {
        level.iter_mut().zip(more).for_each(|(p, m)| p.absorb(m));
    }
}

/// How the check leaves the step (`settle`): standing -- within on its reads, or on its own two levels where no third
/// level above is to be had or bought (`course`) --; clearly beyond its share at the steps' cap; unable to see with no
/// remedy left -- each of these two standing, what its reads do not exclude booked (`Levels::told`, `checked`), and
/// flagged.
#[derive(Clone, Debug)]
enum Settled {
    Standing,
    Capped,
    Unseen,
    /// Asked to stop there (`check` with no `Prices`): the rung's own read is in and not clearly beyond, and its course
    /// is priced -- stand, its third level, a rung below, pairs -- once the states it is priced at are read (`attempt`:
    /// the scout's, the pilot's).
    Pending,
}

/// The members `members` of the pilot's groups `numbers` at both levels (`pair`), joined to those the levels hold, and the
/// step's check (module documentation, # The step), read again at every rung and after every batch of pairs:
/// - a rung whose own read lies clearly beyond the square of the step's share among all the looks (`screen`) is left:
///   the step is halved -- the level at τ becomes the companion, and the members so far run anew at τ/2 alone, its
///   third level above the rung left's own read (`carried`). The finer level keeps its checkpoints (the model is read at
///   its states, `parts_at`);
/// - else, on the scout (which has the pilot to come), the rung stands on its reads, and takes its course by the run's
///   work from here (`course`): it stands, or goes down where a rung below is expected to cost the run less. On the
///   pilot (all the members of a group in: the last word on the step), every estimate as it reads (`verdict`):
///   - its own estimate within and its third level's too: its course (`course`): it stands, or goes down where a rung
///     below costs less; its third level's beyond (the order is not shown above it): it is left;
///   - its own estimate within and no third level, where `above` says the rung above is a rung of the ladder and it was
///     not read unread: its course -- the rung below where it costs less; else the pilot's trajectories at 4τ are its
///     third level where the rung buys them (`buys`: their price no more than what the least rung below adds), and
///     where they read no number (`readable`) the rung above is unread; else it stands on its own two levels;
///   - its own estimate within and no third level to be had: its course, standing on its own two levels or going down;
///   - its own estimate beyond the share and not clearly (`Verdict::Unseeing`): the pairs of `remedy` -- the product's
///     next groups at both levels, read again after every batch -- where they are expected to cost the run less than
///     the rung below; else, or where they cannot see it, it is left: its estimate as it reads is beyond the share.
/// A rung below at which a trajectory would take more than `MOST_STEPS` steps does not exist: a check that is clearly
/// beyond there leaves the step capped, one whose noise no pairs see through ends unable to see (`Settled::Unseen`).
///
/// On the scout and on the pilot alike the check stops where the rung's own read is in and not clearly beyond
/// (`Settled::Pending`: `check` takes it up again, priced, once the states it is priced at are read).
#[allow(clippy::too_many_arguments)]
fn settle(run: &Onefold, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: std::ops::Range<u64>, above: &dyn Fn(f64) -> bool, messages: &mut Messages) -> Settled {
    let [fine, coarse] = pair(run, ensemble, domain, &context.plan, context.tau, numbers, members.clone(), true, messages, &mut |prefix, messages| so_far(context, &levels.fine, prefix, messages));
    absorb(&mut levels.fine, fine);
    absorb(&mut levels.coarse, coarse);
    check(run, ensemble, domain, context, levels, numbers, members.end, above, None, messages)
}

/// The step's check of the levels in hand (`settle`) on the members `..members` of the pilot's groups `numbers` (the
/// pilot's where they are all its members). `prices`: what the check's courses are weighed by -- stand, the third level
/// above, a rung below (`course`), and where the noise hides a rung's difference more pairs (`remedy`) -- at the states
/// in hand (`attempt`: the scout's, then the pilot's), made once the check has its own rung's read; none before them,
/// where the check stops at that read (`Settled::Pending`, as `settle` takes it). The scout's check reads no rung it
/// cannot see: where it is not clearly beyond it stands, or goes down where that costs the run less.
#[allow(clippy::too_many_arguments)]
fn check(run: &Onefold, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: u64, above: &dyn Fn(f64) -> bool, prices: Option<&Prices>, messages: &mut Messages) -> Settled {
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
        crate::prof::note(|| {
            let looks: Vec<String> = judged(&read).iter().map(|(r, b)| format!("{:.3e}±{:.2e}/{:.3e}", r.value, r.spread.error, b)).collect();
            let above: Vec<String> = levels.above.as_ref().map_or(vec![], |a| judged(a).iter().map(|(r, b)| format!("{:.3e}±{:.2e}/{:.3e}", r.value, r.spread.error, b)).collect());
            format!("ladder tau {} pilot {pilot} members {members} groups {} verdict {verdict:?} screen {} looks [{}] above [{}]", context.tau, numbers.len(), screen(&read), looks.join(" "), above.join(" "))
        });
        // A read of which no look reads decides nothing and prices nothing (`Verdict::Unread`): the check ends unable to
        // see where it is.
        if verdict == Verdict::Unread {
            return Settled::Unseen;
        }
        // A rung its reads do not leave takes its course by the run's work from here: without its prices the check
        // stops there (`Settled::Pending`).
        if prices.is_none() && !screen(&read) {
            return Settled::Pending;
        }
        // What the run holds of the rung's levels (`Holding`).
        let groups = numbers.len() as u64;
        let holding = Holding {
            pilot: (groups * GROUP) as f64,
            members: (groups * members) as f64,
            pairs: (groups * members + levels.further[0].len() as u64 * GROUP) as f64,
            below: levels.half.as_ref().map_or(0., |_| groups as f64),
        };
        if let (Verdict::Stands | Verdict::Above, Some(prices)) = (verdict, prices) {
            crate::prof::note(|| {
                let w: Vec<String> = [0.25, 0.5, 1., 2., 4.].iter().map(|f| format!("w({:.4})={:.4e}", f * context.tau, (prices.work)(f * context.tau))).collect();
                format!("ladder tau {} prices product {:.1} {:?} staying {:.4e} descending {:.4e} course {:?} [{}]", context.tau, prices.product, holding, staying(prices, context.tau, holding), descending(prices, context.tau, context.plan.end, holding), course(prices, context.tau, context.plan.end, holding, upward && levels.above.is_none()), w.join(" "))
            });
            match course(prices, context.tau, context.plan.end, holding, upward && levels.above.is_none()) {
                Course::Third if pilot => {
                    // The third level above on the pilot's trajectories: the rung above's coarser level, at 4τ, against
                    // its finer one, this rung's companion. Where its read excludes nothing on some look (`readable`: no
                    // output counts a trajectory at both levels), the rung above is unread.
                    let coarser = level(run, ensemble, domain, &context.plan, 4. * context.tau, numbers, 0..GROUP, false, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages));
                    let read = read_step(&context.plan.reads, context.plan.window, &levels.coarse.iter().collect::<Vec<_>>(), &coarser.iter().collect::<Vec<_>>(), &context.scales);
                    if readable(&read) {
                        (levels.above, levels.above_pilot, levels.above_pairs) = (Some(carried(&read, 0.5)), true, groups * GROUP);
                    } else {
                        levels.above_unread = true;
                    }
                    continue;
                }
                // On the scout the third level is the pilot's to read.
                Course::Stand | Course::Third => return Settled::Standing,
                Course::Below => {}
            }
        }
        // The own read's estimate beyond the share, and not clearly (`Verdict::Unseeing`, the pilot's alone: its check
        // has its prices here): the noise hides the difference, or the error lies near the share. The pairs that could
        // bring it within at the best, where they are expected to cost the run less than the rung below -- its error
        // the order's less, its noise as the scout's pairs read it there (`foresee`, `foreseen`, `remedy`); the rung
        // below is weighed where both are to be had. Where the rung below is cheaper, or pairs cannot see, the rung is
        // left, as an adaptive integrator leaves a step whose error estimate is beyond its tolerance.
        let have = holding.pairs as u64;
        let pairs = match (verdict, prices) {
            (Verdict::Unseeing, Some(prices)) => {
                if !lowest && seeing(&read, have).is_some() && levels.half.is_none() {
                    levels.half = Some(foresee(run, ensemble, domain, context, levels, numbers, messages));
                }
                let below = levels.half.as_ref().map(|(_, half, rung)| Below { asks: foreseen(&read, half, rung), ran: numbers.len() as u64 });
                remedy(&read, have, numbers.len() as u64 * members, below, prices, context.tau, context.plan.end, upward && levels.above.is_none())
            }
            _ => None,
        };
        if let Some(pairs) = pairs {
            let next: Vec<u64> = (have / GROUP..pairs / GROUP).collect();
            let more = pair(run, ensemble, domain, &context.plan, context.tau, &next, 0..GROUP, false, messages, &mut |prefix, messages| {
                so_far(context, &levels.fine, &levels.further[0].iter().chain(prefix.iter().copied()).collect::<Vec<_>>(), messages)
            });
            levels.further.iter_mut().zip(more).for_each(|(further, more)| further.extend(more));
            continue;
        }
        // The rung in hand is left: its reads leave it, or a rung below is expected to cost the run less (`course`). At
        // the ladder's bottom -- where no rung below is cheaper, there being none -- a read clearly beyond leaves the step
        // capped, one that only could not be seen through, unable to see.
        if lowest {
            return if verdict == Verdict::Unseeing { Settled::Unseen } else { Settled::Capped };
        }
        // The rung below's third level above: this rung's own read, lowered.
        let inherits = Some(carried(&read, 0.5));
        context.tau = below;
        // The level at the step is the companion from here on: its states are read no more; its own companion and the
        // further pairs are no level of the rung below's -- all freed before its level runs, which holds as much again.
        forget(&mut levels.fine);
        (levels.coarse, levels.further) = Default::default();
        // Its own level: where the rung below was foreseen on the scout's pairs, those are its first members, run.
        let finer = match levels.half.take() {
            Some((mut first, _, _)) => {
                let rest = level(run, ensemble, domain, &context.plan, below, numbers, 1..members, true, messages, &mut |prefix, messages| so_far(context, &first, prefix, messages));
                absorb(&mut first, rest);
                first
            }
            None => level(run, ensemble, domain, &context.plan, below, numbers, 0..members, true, messages, &mut |prefix, messages| so_far(context, &[], prefix, messages)),
        };
        levels.coarse = std::mem::replace(&mut levels.fine, finer);
        (levels.above, levels.above_pilot, levels.above_pairs, levels.above_unread) = (inherits, pilot, have, false);
    }
}

/// The scout's pairs -- the first member of every pilot group `numbers` -- at the rung below, τ/2, their states kept
/// (they are its level's first members where the check goes down, `check`), and at the rung's own two levels again,
/// τ and 2τ: one hand-out (`run_groups`), the longer trajectories first. A group's sums hold its members together
/// (`Part::absorb`), so the scout's pairs at τ and 2τ are run anew -- the same trajectories to the bit, a trajectory is
/// a function of its number and step (`trajectory`) -- for the two reads on the same pairs, (τ/2, τ) and (τ, 2τ), whose
/// noise's ratio foresees the rung below (`foreseen`). Their cost is the scout's pairs at three steps, a few dozen of
/// the rung's trajectories against the hundreds of pairs it weighs.
fn foresee(run: &Onefold, ensemble: &Ensemble, domain: &Domain, context: &Context, levels: &Levels, numbers: &[u64], messages: &mut Messages) -> (Vec<Part>, Between, Between) {
    let (tau, plan) = (context.tau, &context.plan);
    let half = |i| trajectory(run, ensemble, domain, 0.5 * tau, plan, true, i);
    let own = |i| trajectory(run, ensemble, domain, tau, plan, false, i);
    let double = |i| trajectory(run, ensemble, domain, 2. * tau, plan, false, i);
    let hands = [
        Hand { groups: numbers, members: 0..1, shape: Shape::of(plan, true), record: &half },
        Hand { groups: numbers, members: 0..1, shape: Shape::of(plan, false), record: &own },
        Hand { groups: numbers, members: 0..1, shape: Shape::of(plan, false), record: &double },
    ];
    let further: Vec<&Part> = levels.further[0].iter().collect();
    let mut made = run_groups(&hands, messages, &mut |_, messages| so_far(context, &levels.fine, &further, messages)).into_iter();
    let (half, own, double) = (made.next().expect("the scout's pairs at τ/2"), made.next().expect("at τ"), made.next().expect("at 2τ"));
    let (times, window, scales) = (&plan.reads, plan.window, &context.scales);
    let reads = {
        let [h, o, d] = [&half, &own, &double].map(|level| level.iter().collect::<Vec<&Part>>());
        (read_step(times, window, &h, &o, scales), read_step(times, window, &o, &d, scales))
    };
    (half, reads.0, reads.1)
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
/// `MOST_TRAJECTORIES`. Before every batch's looks the unresolved atoms are booked (`apart`): what they add joins the
/// model's error the pilot fixed, in the looks and in the results. Each batch's result is a checkpoint; returns the
/// estimates, the listed atoms and the counts at its end, and the estimates at the analysis times.
#[allow(clippy::too_many_arguments)]
fn product(run: &Onefold, ensemble: &Ensemble, domain: &Domain, context: &mut Context, parts: &mut Vec<Part>, pilot: &[u64], messages: &mut Messages) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts, Vec<Estimate>) {
    let (started, from) = (Instant::now(), parts.len() as u64 * GROUP);
    // The model's error as the pilot fixed it.
    let booked = context.fixed.model.clone();
    // The group at the list's place g.
    let number = |g: u64| pilot.get(g as usize).copied().unwrap_or(g);
    // The wall time (ms) the trajectories from `done` to `want` take at the product's pace so far; none before its own
    // first trajectories are in.
    let left = |done: u64, want: u64| (done > from).then(|| started.elapsed().as_secs_f64() * 1e3 / (done - from) as f64 * (want - done) as f64);
    loop {
        let all: Vec<&Part> = parts.iter().collect();
        let n = all.len() as u64 * GROUP;
        // The estimates at the analysis times: what the looks and the decisions read (# The analysis times).
        let e = summarize(&reads_of(&all), &context.scales);
        context.unresolved = apart(&e);
        context.fixed.model = booked.iter().zip(&context.unresolved).map(|(model, more)| [probability_error(model[0] + more[0]), model[1] + more[1]]).collect();
        let chased: Vec<Look> = looks(&e, &context.fixed, context.step).into_iter().filter(Look::fits).collect();
        if chased.iter().all(Look::met) || n >= MOST_TRAJECTORIES {
            let (shown, l, c) = pooled(&all, &context.scales);
            return (shown, l, c, e);
        }
        let want = batch(most_needed(&chased), n);
        messages.result(all.len(), n, integrated(&all), left(n, want), || {
            let (shown, l, c) = pooled(&all, &context.scales);
            assemble(context, &shown, &e, &l, &c, n)
        });
        let numbers: Vec<u64> = (n / GROUP..want / GROUP).map(number).collect();
        let new = level(run, ensemble, domain, &context.plan, context.tau, &numbers, 0..GROUP, false, messages, &mut |prefix, messages| {
            let all: Vec<&Part> = parts.iter().chain(prefix.iter().copied()).collect();
            let done = all.len() as u64 * GROUP;
            messages.result(all.len(), done, integrated(&all), left(done, want), || {
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
pub(crate) fn pilot_size() -> u64 {
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
/// by why: the scheme could not advance them (adapt.rs `advance`'s refusal), a step's optical sector made no K (step.rs
/// `Refused::Optical`, by its `Unmade`), or the run's fold does not exist where they were (`run`: a start that lands
/// nowhere, a loss energy that is no number -- the trajectory's one exit, said, never carried on as a number); and the
/// trajectories stopped where a state of theirs scattered beyond every state of the atom (`Domain::runaway`),
/// unresolved from there on (`run`); and the steps whose held forward read its series' ratio at one between two
/// decisions (step.rs `Trajectory::unresolved_steps`; unify2 §2.7: booked at ρ = 1 in the trajectory's books, counted).
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    steps: u64,
    unresolved: u64,
    /// The trajectories ended by their steps (`Fate::Refused`), by a jump that gives no photon, by the optical sector.
    refused: Stopped,
    failed: Stopped,
    unmade: [Stopped; 3],
    nowhere: Stopped,
    runaway: Stopped,
    /// The optical sector's sums over the trajectories (ledger.rs `optical`; step.rs `Trajectory::booked`, `pending`).
    optical: [f64; 4],
}
impl Counts {
    fn add(&mut self, o: &Counts) {
        self.steps += o.steps;
        self.unresolved += o.unresolved;
        self.refused.add(&o.refused);
        self.failed.add(&o.failed);
        for (stopped, other) in self.unmade.iter_mut().zip(&o.unmade) {
            stopped.add(other);
        }
        self.nowhere.add(&o.nowhere);
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

/// A pilot trajectory's state at a time read (model.rs `Sample`) with the trajectory's number: where
/// the fold's readings are read and a step is priced (`booked_at`, `work_at`) -- with what of the trajectory a step from
/// it reads: its next jump, its held forms, its representation of the fold (step.rs `Held`: shared, a handle) and its
/// switches' balance.
#[derive(Clone)]
struct State {
    sample: Sample,
    index: u64,
    /// The trajectory's jumps up to the state, their counted work (step.rs `Trajectory::jumped`), the threshold of its next
    /// one and what its last landing owes it.
    jumps: u32,
    jumped: f64,
    threshold: f64,
    owed: Option<crate::step::Owed>,
    /// Its held forms (step.rs `Forms`: the next step's path momentum).
    forms: crate::step::Forms,
    /// Its representation (step.rs `Trajectory::held`).
    held: crate::step::Held,
    /// Its next decision time, and whether a step of its cell was unresolved (step.rs `Trajectory::next`,
    /// `Trajectory::unresolved_cell`: its account reads ρ = 1 there, `booked_at`).
    next: f64,
    unresolved: bool,
}
impl State {
    /// A trajectory's state as it stands.
    fn of(t: &Trajectory) -> State {
        State {
            sample: Sample { cluster: t.cluster, at: t.point(), psi: t.psi.to_vec() },
            index: t.index,
            jumps: t.jumps,
            jumped: t.jumped,
            threshold: t.threshold,
            owed: t.owed.clone(),
            forms: t.forms.clone(),
            held: t.held.clone(),
            next: t.next,
            unresolved: t.unresolved_cell,
        }
    }

    /// The trajectory at the state, as far as a step from it reads it (`step::priced`): its place, momentum, internal
    /// state and time, its next jump's threshold, its held forms, its representation and its balance, and its streams.
    fn trajectory(&self) -> Trajectory {
        let Sample { cluster, at, psi } = &self.sample;
        Trajectory {
            r: at.r,
            kappa: at.kappa,
            psi: psi.iter().copied().collect(),
            cluster: *cluster,
            time: at.t,
            threshold: self.threshold,
            owed: self.owed.clone(),
            jumps: self.jumps,
            scattered: [0.; 2],
            lost: None,
            unresolved: false,
            seed: SEED,
            index: self.index,
            work: 0.,
            jumped: 0.,
            pending: [[0.; 3]; 3],
            booked: [0.; 3],
            model: [0.; crate::step::MODEL],
            unresolved_steps: 0,
            next: self.next,
            unresolved_cell: self.unresolved,
            held: self.held.clone(),
            forms: self.forms.clone(),
        }
    }
}

/// A kept trajectory's whole state at a step's end (module documentation, # The analysis times; adapt.rs `advance`'s
/// stops): the trajectory, its Brownian path and what its scratch carries to the next step (step.rs `Carry`: the next
/// step's route is weighed against it), from which it is advanced again to the bit (`replay`).
#[derive(Clone)]
struct Checkpoint {
    trajectory: Trajectory,
    path: adapt::Path,
    carry: Carry,
}

/// What one trajectory leaves: what became of it by every output (estimate.rs `Fate`), its (r, κ) at the listed outputs
/// where held; what became of it by every time the decisions read (`Plan::reads`); where its states are kept (`keep`:
/// the pilot's run level, for the model's errors and prices; module documentation, # The model: where the atoms went),
/// its checkpoints (`trail`, module documentation, # The analysis times) and, where it was lost, the first of those
/// times it is lost at and the end of the path its fate was decided along (`approached`); its counts.
struct Record {
    outputs: Vec<Fate>,
    listed: Vec<Option<(V3, V3)>>,
    reads: Vec<Fate>,
    trail: Vec<Checkpoint>,
    lost: Option<(usize, State)>,
    counts: Counts,
    /// Its photons and their Σ k_φ² (step.rs `Trajectory::scattered`).
    scattered: [f64; 2],
}

/// One trajectory's T windows (module documentation, # Outputs and windows) about some times, and what became of it by
/// each time: per lab axis the running integrals from t = 0 of the time held, κ and κ² (µs, 1/µm·µs, 1/µm²·µs), and
/// per time and axis their values at its window's two ends -- the window's time held, ∫κ dt and ∫κ² dt are the
/// differences. An axis's windows' lower ends come in the times' order, and so do their upper ends (a window is cut at
/// t = 0, and at the run's end alike): per axis and side, the next time whose end is to come. Time 0 is the instant
/// t = 0 and has none. A time's ends are kept only while its window or its atom is not in (`open`): its atom takes its
/// window's mean as soon as both are (`settle`) -- a trajectory keeps the windows that are open, a period of the slowest
/// axis's times, not one a time read (a pilot's 15 400 for PGC3D's 10 ms).
struct Windows<'a> {
    times: &'a [f64],
    halves: &'a [V3],
    running: [[f64; 3]; 3],
    fates: Vec<Fate>,
    /// From the time `first` on: per time its ends' values as far as they are read, and whether its atom is in.
    open: VecDeque<([[[f64; 3]; 2]; 3], bool)>,
    first: usize,
    next: [[usize; 2]; 3],
}
impl<'a> Windows<'a> {
    fn new(times: &'a [f64], halves: &'a [V3]) -> Windows<'a> {
        Windows { times, halves, running: [[0.; 3]; 3], fates: vec![Fate::Lost; times.len()], open: VecDeque::new(), first: 1, next: [[1; 2]; 3] }
    }
    /// The lower (side 0) or upper (1) end of time k's window on axis a.
    fn end(&self, k: usize, a: usize, side: usize) -> f64 {
        let (t, h) = (self.times[k], self.halves[k][a]);
        if side == 0 { t - h } else { t + h }
    }
    /// Time k's place among the open ones (k ≥ `first`), opened as far as it.
    fn slot(&mut self, k: usize) -> &mut ([[[f64; 3]; 2]; 3], bool) {
        while self.open.len() <= k - self.first {
            self.open.push_back(([[[0.; 3]; 2]; 3], false));
        }
        &mut self.open[k - self.first]
    }
    /// The atom at time k as it is read there (time 0's the instant's).
    fn record(&mut self, k: usize, fate: Fate) {
        self.fates[k] = fate;
        if k > 0 {
            self.slot(k).1 = true;
            self.settle();
        }
    }
    /// The path's piece from `from` to `to` (the next piece starts where this one ends), with its integrals of κ and κ²
    /// per lab axis between two of its times (step.rs `Segment::moments`): each end inside it reads the integrals up to
    /// it -- once for the ends that coincide, the axes of one frequency and the windows cut at t = 0 --, and the running
    /// integrals take the whole piece.
    fn add(&mut self, from: f64, to: f64, moments: &dyn Fn(f64, f64) -> (V3, V3)) {
        let outputs = self.times.len();
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
                    self.slot(k).0[a][side] = [r[0] + (at - from), r[1] + m1[a], r[2] + m2[a]];
                    self.next[a][side] += 1;
                }
            }
        }
        let (m1, m2) = moments(from, to);
        for a in 0..3 {
            let r = self.running[a];
            self.running[a] = [r[0] + (to - from), r[1] + m1[a], r[2] + m2[a]];
        }
        self.settle();
    }
    /// The open times whose windows and atoms are in, from the first: each atom takes its window's mean (`mean`), and
    /// the time is open no more. A window is in where its upper ends are on every axis (a lower end comes before its
    /// upper one).
    fn settle(&mut self) {
        while let Some(&(ends, true)) = self.open.front() {
            if self.next.iter().any(|n| n[1] <= self.first) {
                break;
            }
            mean(&mut self.fates[self.first], &ends);
            self.open.pop_front();
            self.first += 1;
        }
    }
    /// The trajectory has ended (run through, lost, or stopped), what became of it by every time in: the ends it did
    /// not reach -- within a window, or past its last piece by the rounding -- take the running integrals where it
    /// ended, the window over its held part, and every open time's atom takes its window's mean. A time none of whose
    /// ends was read has a window of no length (and no atom read: `Fate::Lost` or `Fate::Failed`).
    fn close(mut self) -> Vec<Fate> {
        for a in 0..3 {
            for side in 0..2 {
                for e in self.open.iter_mut().skip(self.next[a][side] - self.first) {
                    e.0[a][side] = self.running[a];
                }
            }
        }
        for (k, (ends, _)) in (self.first..).zip(&self.open) {
            mean(&mut self.fates[k], ends);
        }
        self.fates
    }
}

/// An atom held at a time after t = 0 as its window reads it, its window's two ends per lab axis `ends`: its κ and κ²
/// are the window's means -- over its held part, which is no empty stretch after t = 0 (estimate.rs `half_window`); a
/// window of no length is the instant's own κ and κ².
fn mean(fate: &mut Fate, ends: &[[[f64; 3]; 2]; 3]) {
    if let Fate::Held(h) = fate {
        let w: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|i| ends[a][1][i] - ends[a][0][i]));
        for a in (0..3).filter(|&a| w[a][0] > 0.) {
            (h.kappa[a], h.kappa2[a]) = (w[a][1] / w[a][0], w[a][2] / w[a][0]);
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
/// every later time as it stood (module documentation, # The unresolved atoms). One the run's fold does not exist for
/// -- its start lands nowhere (its held forms no number: model/fold/run.rs `Dynamics::started`, step.rs
/// `Trajectory::new`), or at a time read its readout is no number, or outside the trap's region its loss energy is none
/// (`Dynamics::lost` could not tell) -- ends there with no outcome, said (`Counts::nowhere`; # Exits): at t = 0 it is
/// the atom in its initial state, its own populations. Where its states are kept (`keep`), it is
/// advanced from checkpoint to checkpoint (`Checkpoint`: at t = 0 and at the first step's end at or past every multiple
/// of T's shortest window, module documentation, # The analysis times) -- the same trajectory to the bit (adapt.rs
/// `advance`: a stop is no step's end) --, and its states are made again from them (`replay`); else it is advanced
/// through at once.
fn run<D: Dynamics>(model: &D, domain: &Domain, start: Start, index: u64, tau: f64, plan: &Plan, keep: bool) -> Record {
    let _p = crate::prof::scope(crate::prof::SOLVE_TRAJECTORY);
    let sizes = [plan.times.len(), plan.reads.len()];
    let mut t = Trajectory::begun(model, start.state, start.r, start.kappa, SEED, index, plan.cell);
    // The time the run's fold is found not to exist for it (# Exits): at its start where its start lands nowhere.
    let nowhere = std::cell::Cell::new((!t.forms.held.iter().all(|x| x.is_finite()) || !t.psi.iter().all(|z| z.re.is_finite() && z.im.is_finite())).then_some(0.));
    let mut scratch = Scratch::default();
    // The trajectory's Brownian path in the plan's cells, its grid the multiples of τ (a cell over a power of two):
    // the same path at every rung and at its companion.
    let mut path = adapt::Path::new(SEED, index, plan.cell, (plan.cell / tau).log2().round() as u32);
    // The outputs' windows and the decisions' (the trajectory's fates with them), taken up by its reads and its path's
    // pieces alike.
    let windows = std::cell::RefCell::new([Windows::new(&plan.times, &plan.halves), Windows::new(&plan.reads, &plan.read_halves)]);
    let mut listed = vec![None; plan.listed.len()];
    let mut trail: Vec<Checkpoint> = vec![];
    // The times read after t = 0, the outputs' and the decisions', with their places among each.
    let at = merged(&plan.times, &plan.reads);
    let times: Vec<f64> = at.iter().map(|a| a.0).collect();
    // The steps taken, and the start of the last one counted.
    let (mut steps, mut counted) = (0, None);
    let mut record = |places: [Option<usize>; 2], t: &Trajectory, shown: Option<&[f64; STATES]>| {
        // The atom at the time's instant; from the first time on its κ and κ² are its window's means (`mean`).
        // An unresolved one is booked whole (`apart`). Where the run's fold does not exist for it: no outcome from then
        // on, and at t = 0 its own state's populations.
        let _p = crate::prof::scope(crate::prof::SOLVE_RECORD);
        let mut rows = [0.; STATES];
        if nowhere.get().is_none() {
            rows = shown.copied().unwrap_or_else(|| model.readout(t.cluster, &t.point(), &t.psi, &t.held));
            if !rows.iter().all(|x| x.is_finite()) || (domain.light.outside(t.r) && !model.energy(t.cluster, &t.point(), &t.psi).is_finite()) {
                nowhere.set(Some(t.time));
            }
        }
        if nowhere.get().is_some() {
            let total = crate::linalg::norm_sqr(&t.psi);
            rows = [0.; STATES];
            for (z, &s) in t.psi.iter().zip(&model.cluster(t.cluster).states) {
                rows[s] = z.norm_sqr() / total;
            }
        }
        let held = match nowhere.get() {
            Some(_) if t.time > 0. => Fate::Failed,
            _ => Fate::Held(Held { rows, kappa: t.kappa, kappa2: t.kappa.map(|x| x * x), unresolved: t.unresolved }),
        };
        let mut windows = windows.borrow_mut();
        if let Some(k) = places[0] {
            windows[0].record(k, held.clone());
            if let Ok(j) = plan.listed.binary_search(&k) {
                listed[j] = Some((t.r, t.kappa));
            }
        }
        if let Some(k) = places[1] {
            windows[1].record(k, held);
        }
    };
    record([Some(0), Some(0)], &t, None);
    // The first output and the first time read the trajectory does not reach held; whether the scheme could not
    // advance it, and whether it stopped where a state scattered beyond every state of the atom.
    let mut reached = [1, 1];
    let mut unmade = None;
    let (failed, runaway, refused) = {
        let mut read = |i: usize, t: &Trajectory, shown: Option<&[f64; STATES]>| {
            record(at[i].1, t, shown);
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
            for w in windows.borrow_mut().iter_mut() {
                w.add(s.start, until, &|from, to| s.moments(from, to));
            }
        };
        // Through the last output's window -- a failure there leaves every output read --, from checkpoint to checkpoint
        // where the states are kept: stopped at the first step's end at or past each multiple of the window.
        let every = if keep { plan.window } else { plan.end };
        let mut advanced = Ok(());
        while advanced.is_ok() && t.lost.is_none() && t.time < plan.end && nowhere.get().is_none() {
            if keep {
                trail.push(Checkpoint { trajectory: t.clone(), path: path.clone(), carry: scratch.carry() });
            }
            let stop = ((t.time / every).floor() + 1.) * every;
            advanced = adapt::advance(model, domain, tau, &mut t, &mut scratch, &mut path, plan.end, stop, &times, &mut read, &mut each);
        }
        // Of what ends a trajectory with no outcome, its step's own failures (`Fate::Refused`) apart from the model's.
        match advanced {
            Ok(()) => (false, false, false),
            Err(Refused::Runaway) => (false, true, false),
            Err(Refused::Optical(why)) => {
                unmade = Some(why);
                (true, false, false)
            }
            Err(Refused::Overflow | Refused::Reference) => (true, false, true),
            Err(Refused::Emission) => (true, false, false),
        }
    };
    // A runaway stops where it stood: unresolved, held so at every time it did not reach.
    if runaway {
        t.unresolved = true;
        for k in reached[0]..sizes[0] {
            record([Some(k), None], &t, None);
        }
        for k in reached[1]..sizes[1] {
            record([None, Some(k)], &t, None);
        }
        reached = sizes;
    }
    // From the time it does not reach: failed, or lost -- a lost atom's last state held below the potential's far
    // value kept where states are, the end of the path its fate was decided along (`approached`), its fate's to read
    // (`sampled`).
    let (failed, nowhere) = (failed || nowhere.get().is_some(), nowhere.get());
    let mut lost = None;
    let mut windows = windows.into_inner();
    for ((w, &reached), &size) in windows.iter_mut().zip(&reached).zip(&sizes) {
        w.fates[reached.min(size)..].fill(if refused { Fate::Refused } else if failed { Fate::Failed } else { Fate::Lost });
    }
    if reached[1] < sizes[1] && keep && !failed {
        lost = Some((reached[1], State::of(&trail[approached(&trail, &|c| leaving(model, c.r, c.kappa))].trajectory)));
    }
    let [outputs, reads] = windows.map(Windows::close);
    let stopped = |why: bool| if why { Stopped::at(t.time) } else { Stopped::default() };
    let optical = [t.booked[0], t.booked[1], t.booked[2], -(t.pending[0][0] + t.pending[1][1] + t.pending[2][2])];
    let counts = Counts {
        steps,
        unresolved: t.unresolved_steps as u64,
        refused: stopped(refused),
        failed: stopped(failed && !refused && unmade.is_none() && nowhere.is_none()),
        unmade: [Unmade::Schur, Unmade::Singular, Unmade::Unfinite].map(|why| stopped(unmade == Some(why))),
        nowhere: nowhere.map_or_else(Stopped::default, Stopped::at),
        runaway: stopped(runaway),
        optical,
    };
    Record { outputs, listed, reads, trail, lost, counts, scattered: t.scattered }
}

/// A kept trajectory's states at the times `times` (increasing, from t = 0; µs) made again from its checkpoints `trail`
/// (`run`, module documentation, # The analysis times) at the step τ, toward the run's end `end`, in the `domain` it ran
/// in: each time from the last checkpoint before it -- or from where the trajectory stands, advanced again so far,
/// where that is later --, stopped at the first step's end at or past it (adapt.rs `advance`). The states `run` reads
/// there to the bit (adapt.rs, # The reads: a read is an onlooker, a stop no step's end): none after the trajectory's
/// loss or failure, and after a runaway the state it stopped in at every time it did not reach (`run`); none at all of a
/// trajectory whose start lands nowhere (no checkpoint).
fn replay<D: Dynamics>(model: &D, domain: &Domain, tau: f64, end: f64, trail: &[Checkpoint], times: &[f64]) -> Vec<Option<State>> {
    let mut states: Vec<Option<State>> = vec![None; times.len()];
    // A trajectory whose start lands nowhere has no checkpoint (`run`: its fate sealed at t = 0, counted in
    // `Counts::nowhere` and said): nothing is made again of it.
    let Some(first) = trail.first() else { return states };
    states[0] = Some(State::of(&first.trajectory));
    // The first time not read, and the trajectory as far as it is advanced again, with its path and scratch.
    let (mut reached, mut going): (usize, Option<(Trajectory, adapt::Path, Scratch)>) = (1, None);
    while reached < times.len() {
        let next = times[reached];
        let from = &trail[trail.partition_point(|c| c.trajectory.time < next) - 1];
        let (mut trajectory, mut path, mut scratch) = match going.take() {
            Some(going) if going.0.time >= from.trajectory.time => going,
            _ => (from.trajectory.clone(), from.path.clone(), Scratch::resumed(&from.carry)),
        };
        let advanced = adapt::advance(model, domain, tau, &mut trajectory, &mut scratch, &mut path, end, next, times, &mut |k, s, _| (states[k], reached) = (Some(State::of(s)), k + 1), &mut |_, _| {});
        match advanced {
            Ok(()) if trajectory.lost.is_none() => going = Some((trajectory, path, scratch)),
            Err(Refused::Runaway) => {
                states[reached..].fill(Some(State::of(&trajectory)));
                break;
            }
            _ => break,
        }
    }
    states
}

/// What a group of trajectories leaves: its sums at the outputs and at the times the decisions read (estimate.rs
/// `Group`), its held atoms' (r, κ) at the listed outputs; where its states are kept (the pilot's run level: every
/// member's), its trajectories' checkpoints in its members' order (`Record::trail`) and, at the analysis times
/// (`parts_at`), its states there; its lost atoms' paths' ends (`approached`) with the first of the times read each is
/// lost at; its counts.
#[derive(Clone)]
struct Part {
    sums: Group,
    reads: Group,
    listed: Vec<Vec<(V3, V3)>>,
    samples: Vec<Vec<State>>,
    trails: Vec<Vec<Checkpoint>>,
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
        self.trails.extend(other.trails);
        self.lost.extend(other.lost);
        self.counts.add(&other.counts);
    }

    /// The part at the places `at` of its times read alone (increasing, the first and the last among them: analysis
    /// times, # The analysis times): its sums there, no state yet (`parts_at` makes them), a lost atom's first place
    /// the first of them at or after its own; its outputs as they are.
    fn at(&self, at: &[usize]) -> Part {
        Part {
            sums: self.sums.clone(),
            reads: self.reads.at(at),
            listed: self.listed.clone(),
            samples: vec![vec![]; at.len()],
            trails: vec![],
            lost: self.lost.iter().map(|(first, state)| (at.partition_point(|&k| k < *first), state.clone())).collect(),
            counts: self.counts,
        }
    }
}

/// Some groups of a level whose states are kept at the places `at` of their times read (`Part::at`), each with its
/// trajectories' states there made again from their checkpoints (`replay`), in its members' order -- on the pool,
/// trajectory by trajectory --: the level's model `run`, its `domain` and step `tau`, and the plan it ran through.
fn parts_at(run: &Onefold, domain: &Domain, tau: f64, plan: &Plan, parts: &[&Part], at: &[usize]) -> Vec<Part> {
    let times: Vec<f64> = at.iter().map(|&k| plan.reads[k]).collect();
    parts
        .par_iter()
        .map(|p| {
            let mut part = p.at(at);
            let mut made: Vec<Vec<Option<State>>> = p.trails.par_iter().map(|trail| replay(run, domain, tau, plan.end, trail, &times)).collect();
            for (j, states) in part.samples.iter_mut().enumerate() {
                states.extend(made.iter_mut().filter_map(|m| m[j].take()));
            }
            part
        })
        .collect()
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
    /// A group's part before its first record.
    fn empty(&self) -> Part {
        Part {
            sums: Group::new(self.outputs),
            reads: Group::new(self.reads),
            listed: vec![vec![]; self.listed],
            samples: vec![],
            trails: vec![],
            lost: vec![],
            counts: Counts::default(),
        }
    }
    /// A group's next record added to its part (the records in their members' order).
    fn add(&self, part: &mut Part, r: Record) {
        part.sums.add(&r.outputs);
        part.sums.scatter(r.scattered);
        part.reads.add(&r.reads);
        part.reads.scatter(r.scattered);
        for (l, x) in part.listed.iter_mut().zip(r.listed) {
            l.extend(x);
        }
        if self.keep {
            part.trails.push(r.trail);
            part.lost.extend(r.lost);
        }
        part.counts.add(&r.counts);
    }
}

/// Trajectory `index` of a run at the step τ: its initial state from the ensemble (ensemble.rs `Ensemble::draw`), run
/// through the plan (`run`). A function of the index and the step: the same index at another step is the same atom on
/// the same streams (module documentation, # The step).
fn trajectory(model: &Onefold, ensemble: &Ensemble, domain: &Domain, tau: f64, plan: &Plan, keep: bool, index: u64) -> Record {
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
/// the trajectories, every core unless the process is told otherwise). Each thread takes the list's next trajectory as it ends
/// one, so a faster core takes more of them, the groups finish nearly in order and the first after about one
/// trajectory's time. The threads have no trajectory left only at the list's end: what is known to be run together is
/// handed out together (`pair`), the level of the longer trajectories first, so that the threads that wait for the
/// list's last trajectories wait for its shortest. Each group's records are added in its members' order on this thread
/// as soon as those before it are in -- a record waits only for its group's members before it, never for the whole
/// group (a record holds its trajectory's fates at every time read) -- (`Shape::add`, nothing depends on the threads),
/// each level's groups returned in their order; whenever the solve's
/// next message is due (`Messages::due`), `progress` takes the first level's groups finished so far in order and sends
/// the result of what is in with them (module documentation, # Checkpoints); where it has nothing to send the message
/// stays due, and is asked for again when the next group is in. A message is made on this thread by loops that run on
/// the pool (`pooled`, `accounts`), whose threads are all in their trajectories: between two trajectories a thread
/// takes what waits for the pool, so a message waits for the next trajectory to end on any thread and not for the
/// list's end. A trajectory's panic -- the engine's defect -- ends the hand-out: no trajectory more is handed out, and
/// once the threads' trajectories in hand end it unwinds to the caller (the worker answers the request with it, main.rs
/// # Defects).
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
    // Per level and group its part as far as its records are added, the members' records that came in before their
    // turn, and the next member to add.
    let mut waiting: Vec<Vec<(Option<Part>, Vec<Option<Record>>, u64)>> = levels.iter().map(|level| (0..level.groups.len()).map(|_| (None, (0..level.size()).map(|_| None).collect(), 0)).collect()).collect();
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
                // A trajectory's panic is the engine's defect: no trajectory more is handed out, and the panic unwinds
                // to the hand-out's caller once the threads' trajectories in hand end (rayon's scope hands it on).
                let record = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (levels[level].record)(levels[level].number(k)))) {
                    Ok(record) => record,
                    Err(payload) => {
                        next.store(stop, Ordering::Relaxed);
                        std::panic::resume_unwind(payload)
                    }
                };
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
                    let (size, shape) = (levels[level].size(), &levels[level].shape);
                    let (g, m) = ((k / size) as usize, (k % size) as usize);
                    let (part, early, next) = &mut waiting[level][g];
                    early[m] = Some(r);
                    while let Some(r) = early.get_mut(*next as usize).and_then(Option::take) {
                        shape.add(part.get_or_insert_with(|| shape.empty()), r);
                        *next += 1;
                    }
                    if *next == size {
                        parts[level][g] = part.take();
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
/// its unresolved atoms carry (`Context::unresolved`) --; none at t = 0, where no model has acted (`model_error`).
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
            let shown = Look { target: POPULATION_TARGET, account: population[k], interval: e.shown_interval(), trajectories: n, shown: true };
            let held = (e.held > 0.).then(|| {
                let read = Some(e.temperature_interval()).filter(|i| i.is_finite());
                let account = temperature[k].unwrap_or_else(|| temperature_account(e, fixed.model[k][1], &step, read.unwrap_or(CONFIDENCE * (2. / (n * h)).sqrt())));
                Look { target: e.temperature_target, account, interval: account.statistics, trajectories: n, shown: false }
            });
            std::iter::once(shown).chain(held)
        })
        .collect()
}

/// The trajectories the looks a run's trajectories can meet need (`Look::fits`, `Look::need`), the most of them: what
/// the product's next batch is made for (`batch`), and the product the step's check weighs its choice by (`Prices`).
fn most_needed(looks: &[Look]) -> f64 {
    looks.iter().filter(|l| l.fits()).map(Look::need).fold(0., f64::max)
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

/// T's account at a time (`accounts`) from its estimate, the model's error on T there `model`, the step's part `bound`
/// (a share of T's allowance) and the statistics' relative interval `interval`: the model's part (`temperature_model`)
/// and the statistics' weighed by the held share, the step's read so already (estimate.rs `between`), times the
/// allowance.
fn temperature_account(e: &Estimate, model: f64, bound: &Bound, interval: f64) -> Account {
    let h = 1. - e.loss;
    Account { model: weighed(h, temperature_model(e, model)), step: bound.temperature * e.temperature_target, statistics: weighed(h, interval) }
}

/// The model's part of T's account at a time whose model's error on T is `model` (relative): it where every
/// trajectory is counted; unbounded where some is not -- the part of T the held atoms of unknown fate make is bounded by
/// nothing the run read (estimate.rs, # What is counted).
fn temperature_model(e: &Estimate, model: f64) -> f64 {
    if e.uncounted > 0. { f64::INFINITY } else { model }
}

/// An error of the held atoms weighed by their share h, as an account weighs it: unbounded stays so at any share -- h
/// times it would be no number at h = 0.
fn weighed(h: f64, error: f64) -> f64 {
    if error.is_finite() { h * error } else { f64::INFINITY }
}

/// What the pilot measured of the fold (SPEC §5.2; module documentation, # The model): the fold's readings and their
/// combination read at the states (`Sampled`: the rows one after another, a row the readings' errors on the populations
/// and on T in `READINGS`' order, their combination last), each read's time, whether it reads a lost atom's fate.
struct Measured<'a> {
    errors: &'a [Booking],
    output: &'a [usize],
    row: &'a [usize],
    lost: &'a [bool],
}
impl Measured<'_> {
    /// Per time and reading (`READINGS`) its error on what is shown (absolute) and on the held atoms' T (relative)
    /// (module documentation, # The model: where the atoms went). What is shown at a time is every atom's fate there --
    /// a held atom's rows, P_s = E[A ρ_s], and a lost one's loss, L = E[1 − A] (estimate.rs, # The outputs are
    /// expectations) --, so the model errs on it by the mean over the atoms of the error of each one's fate: (1 − L)
    /// times the held atoms' mean, each at its state there, and L times the lost atoms', each along its path
    /// (`Booking::path`). T is the held atoms' alone. Where no read stands for a time the error is its bound there, 1 on
    /// what is shown and none on T (the run's atoms there are at most the share the pilot left unseen, and T has no
    /// account without a pilot's T: its look fits no number of trajectories and is not chased). The share L is the one
    /// `estimates` give there. The times are made on the pool, each by itself: a time's sums take its reads in their
    /// order.
    fn means(&self, outputs: usize, estimates: &[Estimate]) -> Vec<Vec<[f64; 2]>> {
        self.reduced(outputs, estimates, READINGS.len(), &|row, out| {
            for (out, e) in out.iter_mut().zip(row) {
                *out = e.error;
            }
        })
    }

    /// Per time the fold's error (`means`): at each read the state's readings combined (`booked_at`: its row's last, the
    /// books' own combination), and their means.
    fn combined(&self, outputs: usize, estimates: &[Estimate]) -> Vec<[f64; 2]> {
        self.reduced(outputs, estimates, 1, &|row, out| out[0] = row[READINGS.len()].error).into_iter().map(|o| o[0]).collect()
    }

    /// Per time `n` errors (`means`): each read gives them (`each`: its row into its `n` pairs), and the held reads'
    /// means and the lost reads' make them, by the held share there.
    fn reduced(&self, outputs: usize, estimates: &[Estimate], n: usize, each: &(dyn Fn(&[Booking], &mut [[f64; 2]]) + Sync)) -> Vec<Vec<[f64; 2]>> {
        let width = READINGS.len() + 1;
        // Each time's reads, in their order.
        let mut reads: Vec<Vec<usize>> = vec![vec![]; outputs];
        for (read, &k) in self.output.iter().enumerate() {
            reads[k].push(read);
        }
        reads
            .par_iter()
            .enumerate()
            .map(|(k, reads)| {
                let (mut held, mut lost, mut one) = (vec![[0.; 2]; n], vec![0.; n], vec![[0.; 2]; n]);
                let (mut holding, mut losing) = (0., 0.);
                for &read in reads {
                    each(&self.errors[self.row[read] * width..][..width], &mut one);
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

/// The model's error booked per time (SPEC §4; module documentation, # The model): the fold's combined readings
/// measured at the states (`Measured::combined`), a population's at most 1; none at t = 0, the request's state.
fn model_error(combined: &[[f64; 2]]) -> Vec<[f64; 2]> {
    combined.iter().enumerate().map(|(k, c)| if k == 0 { [0.; 2] } else { [probability_error(c[0]), c[1]] }).collect()
}

/// A state a mean over the run is read at (`one_each`): a pilot trajectory's at one time, with its group's place
/// among the pilot's and its weight in the mean, the time its trajectory holds its atom (µs).
struct Pick<'a> {
    state: &'a State,
    group: usize,
    weight: f64,
    /// The trajectory's jumps' counted work a µs over the time it holds its atom: its last state's (`State::jumped`) over
    /// that state's time.
    jumping: f64,
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
            // Its states to pick from, and its last: where it was lost, its path's end (`approached`).
            let states = if held.is_empty() { lost.map(|_| vec![first]).unwrap_or_default() } else { held };
            let Some(last) = lost.or(states.last().copied()) else { continue };
            let through = ((member * groups + group) as f64 + 0.5) / (members.len() * groups) as f64;
            let state = states[((through * states.len() as f64) as usize).min(states.len() - 1)];
            picks.push(Pick { state, group, weight: last.sample.at.t, jumping: last.jumped / last.sample.at.t });
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
    /// What the unresolved atoms add to the model's error at each analysis time (`apart`), in `fixed` with the pilot's:
    /// nothing until the product reads them.
    unresolved: Vec<[f64; 2]>,
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

/// The trajectories that ended with no outcome (`Counts`; module documentation, # Exits), by why, in the result's words:
/// each cause met, how many and from when; `booked`, with what that leaves in a result (their share booked whole on
/// what is shown) -- else the cause alone, as the request's exit says it (`nothing_integrated`).
fn ended(counts: &Counts, booked: bool) -> Vec<String> {
    let unmade = "在某一步上，光学扇区的噪声算子 K 做不出";
    let [after, there] = if booked { ["：此后它们不计入估计，所占的比例全部计入布居与丢失的误差", "：它们在那里结束，此后不计入估计，所占的比例全部计入布居与丢失的误差"] } else { [""; 2] };
    let mut out = vec![];
    for (stopped, why) in [
        (&counts.refused, "的一步在所取步长下算出的数不是有限数（状态或参考流），在时间的舍入以内减半步长也推进不了".to_string()),
        (&counts.failed, "的跳跃给不出光子（模型在该状态上失败，缩短步长无济于事）".to_string()),
        (&counts.unmade[0], format!("{unmade}（激发态块或基态块的 Schur 形式不收敛；这不应发生，是引擎的缺陷）")),
        (&counts.unmade[1], format!("{unmade}（ℳ 奇异：按构造 Q 扇区以约 Γ/2 或更快弛豫，ℳ 应当可逆；这不应发生，是引擎的缺陷）")),
        (&counts.unmade[2], format!("{unmade}（算出的数不是有限数；这不应发生，是引擎的缺陷）")),
    ] {
        if let Some(t) = stopped.since {
            out.push(format!("{} 条轨迹{why}（最早在 {}）{after}", stopped.count, when(t)));
        }
    }
    if let Some(t) = counts.nowhere.since {
        out.push(format!("{} 条轨迹所在处的折叠不存在（起点落不到任何簇上、读出不是有限数、或在光阱区域外丢失能量不是有限数，最早在 {}）{there}", counts.nowhere.count, when(t)));
    }
    out
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
        let (doubled, third) = (step_reading(c.plan.end, 2. * c.tau), check.third.map(|(s, pairs)| (step_reading(c.plan.end, s), pairs)));
        let levels = third.as_ref().map_or(format!("τ 与 2τ（{doubled} μs）"), |(at, _)| format!("τ、2τ（{doubled} μs）与 4τ（{at} μs）"));
        // The pairs at each level: the third level's may be fewer than the rung's own (the scout's, or the pilot's
        // beside the further pairs).
        let paired = match &third {
            Some((at, pairs)) if *pairs < check.pairs => format!("{} 对先导轨迹用 τ 与 2τ（{doubled} μs）各算一遍、其中 {pairs} 对又用 4τ（{at} μs）算一遍", check.pairs),
            _ => format!("{} 对先导轨迹用 {levels} 各算一遍", check.pairs),
        };
        let how = if check.saw {
            "相邻两级之差按步长误差与步长的二次方成正比各自换算成 τ 的误差，两个估计都在误差预算分给步长的部分之内才用这个步长；计入的是两者中较大的估计"
        } else {
            "检查没有结论（三个步长的比较既没有确定在误差预算分给步长的部分之内，也没有明显超出），而增加成对轨迹和把步长再减半都已不可能或不再有用；计入的是检查不能排除的最大值（方均根加上它的置信区间）"
        };
        ledger.item(&format!("时间步长 τ = {} μs（全程不变）", step_reading(c.plan.end, c.tau)), largest, &format!("步长收敛检查：先导轨迹用 {levels} 各算一遍、逐条比较，取差在各输出时刻上的方均根（已扣除统计噪声）；{how}"));
        let relative = if check.resolution[1].is_finite() { format!("、在温度上低于其允许误差的 {} 时检查分辨不出", percent(check.resolution[1])) } else { " 时检查分辨不出，在温度上无法估计".into() };
        step = format!("（步长收敛检查：{paired}（同一初始原子、同一组随机数），相邻两级之差在各输出时刻上的方均根（已扣除统计噪声）按二阶换算成本步长的误差；{how}；步长误差的方均根在布居上低于 {}{relative}）", percent(check.resolution[0]));
    }
    // What the unresolved atoms add (`apart`): its largest over the analysis times, weighed by the held share.
    for (i, what) in ["布居", "温度"].into_iter().enumerate() {
        let largest = (1..reads.len()).map(|k| c.unresolved[k][i] * (1. - reads[k].loss)).fold(0., f64::max);
        ledger.item(&format!("计算中停下的原子（{what}）"), largest, "在每个输出时刻统计：因散射率超过原子任何状态的上限而停下的原子（从那一步起），按它们所占的比例全部计入（布居上是原子数的比例，温度上是动能的比例）");
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
    for said in ended(counts, true) {
        s.approximations.flag(said);
    }
    if counts.unresolved > 0 {
        s.approximations.flag(format!("保持的前向有 {} 步（共 {} 步）未解析：两次决定之间级数之比到 1，折叠在格内可能失去存在；这些步的误差不在折叠的读数内", counts.unresolved, counts.steps));
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
    use crate::estimate::unseen;
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

    /// The first rung and its double are below the step's edge (`first_rung`, `stable`), the first rung the largest
    /// such: over runs, traps and kicks the rung is an even part of the run, it and its double stable, and the next
    /// larger one not. The edge: at π for a kick without curvature and for a softer well; below π for a stiffer one,
    /// where (θ/2) tan(θ/2) = ω²/ω_k² (0.5477 π at ω_k = ω, 0.3056 π at ω_k² = 4 ω²); the kick's own ω_k τ = 2 on a
    /// flat reference. The constructed example (numbers.status.md §1.1): 1 µK in a tweezer of 20 kHz and a light well
    /// of ω_k = 0.35 rad/µs; the trap alone allows 25 µs, the step's edge is at 5.5 µs. No rung within the steps' cap,
    /// or a frequency or a curvature that is no number: none.
    #[test]
    fn the_first_rung_and_its_double_are_below_the_step_s_edge() {
        let khz = |f: f64| std::f64::consts::TAU * f * 1e-3;
        assert_eq!(first_rung(10., f64::INFINITY, khz(74.65), 0.), Some(2.5));
        assert_eq!(first_rung(10., f64::INFINITY, khz(1.), 0.), Some(5.));
        let both = |tau: f64, w: f64, kick: f64| stable(tau, w, kick) && stable(2. * tau, w, kick);
        for (run, w, kick) in [(10., khz(910.7), 0.), (20., khz(79.), 0.), (16., khz(20.3), 0.), (100., khz(20.), 0.35 * 0.35), (100., khz(74.65), -0.5 * khz(74.65).powi(2))] {
            let tau = first_rung(run, f64::INFINITY, w, kick).unwrap();
            let pairs = (run / (2. * tau)).round();
            assert!((run / (2. * pairs) - tau).abs() <= 1e-12 * tau && both(tau, w, kick), "{run} {tau}");
            assert!(pairs == 1. || !both(run / (2. * (pairs - 1.)), w, kick), "{run} {tau}: a larger rung is below the edge");
        }
        let w = khz(100.);
        let edge = |kick: f64, lo: f64, hi: f64| stable(lo * PI / w, w, kick) && !stable(hi * PI / w, w, kick);
        assert!(edge(0., 0.9999, 1.0001) && edge(-0.5 * w * w, 0.9999, 1.0001) && edge(-9. * w * w, 0.9999, 1.0001));
        assert!(edge(w * w, 0.5477, 0.5478) && edge(4. * w * w, 0.3056, 0.3057) && edge(0.1 * w * w, 0.9096, 0.9097));
        assert!(stable(1.9999 / w, 0., w * w) && !stable(2.0001 / w, 0., w * w));
        assert!(stable(24.9, khz(20.), 0.) && stable(5.4, khz(20.), 0.35 * 0.35) && !stable(5.6, khz(20.), 0.35 * 0.35));
        assert_eq!(first_rung(1e9, f64::INFINITY, khz(74.65), 0.), None);
        assert_eq!(first_rung(10., f64::INFINITY, f64::NAN, 0.), None);
        assert_eq!(first_rung(10., f64::INFINITY, khz(74.65), f64::NAN), None);
    }

    /// The ladder begins within the physics prior (`prior`, `first_rung`): T's step error (k_max v_T τ)² at the step's
    /// share of T's allowance at the reach's temperature, the request's to begin with. PGC3D's Rb87 at 30 µK on D2 (k = 8.0528 /µm; its tweezer's
    /// 100, 100 and 19 kHz, a zero point of 1.8 µK below 30 T_rec = 10.9 µK): δT/T = 0.375 and τ ≤ 0.950 µs, where its
    /// ladder's top by the edges alone, 1.425 µs, was left. Over temperatures, wave numbers and traps the phase a step
    /// of the prior advances, squared, is that share to the rounding, and no light's pattern bounds nothing. Over runs,
    /// priors and edges the first rung is the largest even part of the run within the prior that the edge admits with
    /// its double -- the next larger one is beyond the prior or the edge --, it is the edge's own rung wherever the
    /// prior is above that, and it never exceeds the prior.
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
        let both = |tau: f64, w: f64, kick: f64| stable(tau, w, kick) && stable(2. * tau, w, kick);
        for (run, w, kick) in [(1026.3, khz(100.), 0.), (10., khz(74.65), 0.), (100., khz(20.), 0.35 * 0.35), (100., khz(910.7), 0.), (5000., khz(1.), 0.)] {
            let edges = first_rung(run, f64::INFINITY, w, kick).unwrap();
            for prior in [1e-3, 0.0954, 0.3581, 0.7125, 0.950, 1.425, 3.7, 1e3, f64::INFINITY] {
                let tau = first_rung(run, prior, w, kick).unwrap();
                let pairs = (run / (2. * tau)).round();
                assert!((run / (2. * pairs) - tau).abs() <= 1e-12 * tau && tau <= prior * (1. + 1e-12) && tau <= edges && both(tau, w, kick), "{run} {prior}: {tau}");
                assert!(pairs == 1. || run / (2. * (pairs - 1.)) > prior || !both(run / (2. * (pairs - 1.)), w, kick), "{run} {prior}: {tau}, a larger rung is within both");
                assert!(prior < edges || tau == edges, "{run} {prior}: {tau} below the edges' {edges}");
            }
        }
    }

    /// Reads of held atoms' states, a row of the table each, in their order (`Measured`).
    fn rows(n: usize) -> (Vec<usize>, Vec<bool>) {
        ((0..n).collect(), vec![false; n])
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
    /// share; the largest asking of the looks is taken. A look whose spread is unread excludes nothing (`readable`): no
    /// number of pairs. A look that is no number -- no time reads it -- is no look (`read_looks`): it asks for nothing,
    /// and the others ask as they do; it still excludes nothing.
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
        assert!(close(blind(&row), 2.) && !readable(&row) && readable(&at(None)));
    }

    /// The run's work a step's check weighs its choice by (`Prices`) in the tests: a trajectory's work a µs as its steps'
    /// count, 1/h, and the product's trajectories.
    fn steps(product: f64) -> (Box<dyn Fn(f64) -> f64>, f64) {
        (Box::new(|h: f64| 1. / h), product)
    }

    /// Where no rung below is (the ladder's bottom), or no pairs see (`remedy`), the rung asks as it did before the
    /// rung below was priced: the factor `blind` gives, in whole groups, at most twice the pairs it has a batch, wherever
    /// a run can take them -- 1.05 times 88 is 96, 7.1 times 88 a first batch of 176, 4 times 352 is 704 (a read of
    /// 1.5 b ± 0.5 b on the total T against b = s², unlimited groups, t = 2: (2.5 − 0.5)/(1 − 0.5)) --; none past the
    /// most trajectories a solve runs, nor for a read whose near end is at the share or that excludes nothing -- the
    /// rung is then left, or at the bottom the check ends unable to see -- whatever the rung below is foreseen to ask.
    #[test]
    fn where_no_rung_below_is_or_no_pairs_see_the_rung_asks_as_before() {
        let b = step_bound()[1].powi(2);
        let (work, product) = steps(300.);
        let prices = Prices { work: &*work, product };
        let close = |asked: Option<f64>, factor: f64| asked.is_some_and(|f| (f - factor).abs() < 1e-12);
        let wide = at(Some(read(1.5 * b, 0.5 * b)));
        // A read of none whose far end is f b asks f times its pairs.
        let factor = |f: f64| at(Some(read(0., 0.5 * f * b)));
        assert!(close(blind(&wide), 4.) && close(blind(&factor(1.05)), 1.05) && close(blind(&factor(7.1)), 7.1));
        let bottom = |r: &Between, have: u64| remedy(r, have, 88, None, &prices, 0.25, 1000., true);
        assert_eq!((bottom(&factor(1.05), 88), bottom(&factor(7.1), 88), bottom(&wide, 352)), (Some(96), Some(176), Some(704)));
        let near = at(Some(read(3. * b, 0.5 * b)));
        let none = at(Some(read(f64::NAN, 0.)));
        for below in [None, Some(Below { asks: None, ran: 11 }), Some(Below { asks: Some(f64::INFINITY), ran: 11 })] {
            let asks = |r: &Between, have: u64| remedy(r, have, 88, below, &prices, 0.25, 1000., true);
            assert_eq!((asks(&factor(200.), 88), asks(&near, 88), asks(&none, 88), asks(&factor(1.01), MOST_TRAJECTORIES)), (None, None, None, None));
        }
    }

    /// The rung below as the scout's pairs foresee it (`foreseen`), for a read on the total T of 3.975 b ± 1.9875 b --
    /// its near end at none, its far end 7.95 b, p10d5's at 0.25 µs on its pilot's 88 pairs --: where the scout's pairs
    /// stay together, their noise at τ/2 against τ is the order's less, 2^{−2p} (their read the rung's own, lowered,
    /// `carried`), and the rung below's far end 7.95/16 b is within on the pairs in hand (asks nothing); where they have
    /// parted, the noise is the same at both (a read of an error lowered by the order beside the same standard error),
    /// and the rung below asks the same 7.95 times the pairs as the rung. A rung whose error lies near its share,
    /// 1.04 b ± 0.025 b (near end 0.99 b, far end 1.09 b: ten times its pairs at the rung), foresees the rung below within
    /// either way: its error the order's less. A T the scout's pairs do not read leaves the rung below unread; the
    /// looks with no noise (the loss and the rows here) ask nothing below.
    #[test]
    fn the_rung_below_is_foreseen_with_the_noise_the_scout_s_pairs_read_there() {
        let b = step_bound()[1].powi(2);
        let own = at(Some(read(3.975 * b, 1.9875 * b)));
        let near = at(Some(read(1.04 * b, 0.025 * b)));
        assert!((blind(&own).unwrap() - 7.95).abs() < 1e-12 && (blind(&near).unwrap() - 10.).abs() < 1e-9);
        let rung = at(Some(pilot(2. * b, 3. * b)));
        let together = carried(&rung, 0.5);
        let parted = at(Some(Read { value: 2. * b / 16., ..pilot(0., 3. * b) }));
        assert_eq!(foreseen(&own, &together, &rung), None);
        assert!(foreseen(&own, &parted, &rung).is_some_and(|f| (f - 7.95).abs() < 1e-12));
        assert_eq!((foreseen(&near, &together, &rung), foreseen(&near, &parted, &rung)), (None, None));
        assert_eq!(foreseen(&own, &at(None), &rung), Some(f64::INFINITY));
    }

    /// The noise ratio the scout's pairs read is taken within [2^{−2p}, 1], where its expectation lies (`foreseen`; module
    /// documentation, ## The ladder): halving the step, pairs that stay together lower their noise by the order's factor,
    /// pairs that have parted keep it, and no pair's noise grows. Read at 16.6 (PGC3D's scout) the rung below is foreseen
    /// as on pairs that have parted, r = 1: the same 7.95 times the pairs as the rung for the read of 3.975 b ± 1.9875 b;
    /// read at a quarter of 2^{−2p}, as on pairs that stay together, r = 2^{−2p}: for a read whose far end is 32 b, twice
    /// the pairs in hand, where the ratio as read would have foreseen it within on them.
    #[test]
    fn the_noise_ratio_the_scout_s_pairs_read_is_taken_within_its_range() {
        let b = step_bound()[1].powi(2);
        let rung = at(Some(pilot(2. * b, 3. * b)));
        let noise = |ratio: f64| at(Some(Read { value: 2. * b / 16., ..pilot(0., ratio * 3. * b) }));
        let own = at(Some(read(3.975 * b, 1.9875 * b)));
        assert_eq!(foreseen(&own, &noise(16.6), &rung), foreseen(&own, &noise(1.), &rung));
        assert!(foreseen(&own, &noise(1.), &rung).is_some_and(|f| (f - 7.95).abs() < 1e-12));
        let wide = at(Some(read(16. * b, 8. * b)));
        let lowered = 0.5_f64.powi(2 * ORDER);
        assert_eq!(foreseen(&wide, &noise(0.25 * lowered), &rung), foreseen(&wide, &noise(lowered), &rung));
        assert!(foreseen(&wide, &noise(lowered), &rung).is_some_and(|f| (f - 2.).abs() < 1e-9));
    }

    /// Two causes of a read that cannot see, two answers (`remedy`; module documentation, ## The ladder), priced at the
    /// steps' count (1/h a µs), the pilot's 88 pairs in hand at τ = 0.25 µs (the third level at 4τ to run where it
    /// stands), the scout's 11 pairs run at τ/2:
    /// - p10d5's read (`the_rung_below_is_foreseen_with_the_noise_the_scout_s_pairs_read_there`), its product 143
    ///   trajectories: 700 pairs at the rung, 612 more at τ and 2τ and the pilot at 4τ, 3758 µs⁻¹ of work; where the
    ///   pairs stay together the rung below's pilot sees it, and its product at τ/2, 132 trajectories beyond the scout's,
    ///   is 1056: the rung below; where they have parted, the rung below asks the same 700 pairs, at τ/2 and τ, 7955: the
    ///   pairs, 176 a first batch;
    /// - an error near the share (1.04 b ± 0.025 b) with parted pairs: 880 pairs at the rung, 4840, against the rung below
    ///   seen on the pilot: the rung below.
    /// And the choice is no step in the read: an estimate of 1.05 b on parted pairs, a product of 300, its interval from
    /// 2.1 b down to 0.12 b -- the near end from none up to 0.99 b --, the choice moves once, from the pairs to the rung
    /// below, and never back.
    #[test]
    fn a_rung_that_cannot_see_takes_the_rung_below_where_its_pairs_stay_together_or_its_error_is_near_its_share() {
        let b = step_bound()[1].powi(2);
        let (work, _) = steps(0.);
        let rung = at(Some(pilot(2. * b, 3. * b)));
        let (together, parted) = (carried(&rung, 0.5), at(Some(Read { value: 2. * b / 16., ..pilot(0., 3. * b) })));
        let asks = |r: &Between, half: &Between, product: f64| {
            let prices = Prices { work: &*work, product };
            remedy(r, 88, 88, Some(Below { asks: foreseen(r, half, &rung), ran: 11 }), &prices, 0.25, 1000., true)
        };
        let own = at(Some(read(3.975 * b, 1.9875 * b)));
        let near = at(Some(read(1.04 * b, 0.025 * b)));
        for r in [&own, &near] {
            assert!(verdict(r, None, true, true) == Verdict::Unseeing && !screen(r));
        }
        assert_eq!((asks(&own, &together, 143.), asks(&own, &parted, 143.), asks(&near, &parted, 143.)), (None, Some(176), None));
        let choices: Vec<bool> = (0..=99).map(|k| asks(&at(Some(read(1.05 * b, (0.525 - 0.005 * k as f64) * b))), &parted, 300.).is_some()).collect();
        let flips = choices.windows(2).filter(|w| w[0] != w[1]).count();
        assert!(choices[0] && !choices[99] && flips == 1, "{choices:?}");
    }

    /// Where a step's price is the same at every step, w(h) = c/h, the ladder is the one it was (`course`, `buys`; module
    /// documentation, ## The ladder): a rung its reads stand stays -- on the scout's position and on the pilot's, the
    /// pilot reading its third level above wherever it has one to read --, whatever the price c, the step, the product,
    /// the pairs in hand and the rung below's level run: a rung below adds at least N c/τ to staying and the third
    /// level costs P c/(4τ), less than that. `remedy` prices the third level where it is bought (`buys`): wherever there
    /// is one, as it did. At the ladder's bottom (6.1·10⁻⁵ µs of a 1 ms run: no rung below within `MOST_STEPS`) the same.
    #[test]
    fn a_flat_price_leaves_the_ladder_as_it_was() {
        let end = 1000.;
        for c in [1e-3, 1., 2.9e5] {
            let work = move |h: f64| c / h;
            for product in [0., 50., 88., 287., 5000., MOST_TRAJECTORIES as f64] {
                let prices = Prices { work: &work, product };
                for tau in [6.1e-5, 0.0954, 0.95, 3.7, 125.] {
                    for (members, pairs, below) in [(11., 11., 0.), (88., 88., 0.), (88., 88., 11.), (88., 352., 11.), (88., 1408., 0.)] {
                        let holding = Holding { pilot: 88., members, pairs, below };
                        for third in [false, true] {
                            let expected = if third { Course::Third } else { Course::Stand };
                            assert_eq!(course(&prices, tau, end, holding, third), expected, "{c} {product} {tau} {holding:?} {third}");
                        }
                        assert!(members < 88. || buys(&prices, tau, end, holding), "{c} {product} {tau} {holding:?}");
                    }
                }
            }
        }
    }

    /// A step whose price grows with the step takes the rung of the least work (`course`, `descending`; module
    /// documentation, ## The ladder): the ladder taken a rung at a time -- from the scout's position (its eleven pairs
    /// run at the first rung, 1 µs, and its double), then from the pilot's -- ends at the rung of the least work by the
    /// definition, the levels the run has still to run added up: the scout's pairs at every rung on the way, the pilot's
    /// other members at the rung and at its companion, the product beyond the pilot; over products from the pilot's 88
    /// to a run's most. A price p(h) = 1 + (h/h₀)⁴ a step (h₀ = 0.25 µs: 257 at the first rung, 2 at h₀) takes two or
    /// three rungs below the first; the same price with the rung at 0.5 µs a hundred times dearer goes past that rung
    /// where the rungs below it pay for it, though that rung alone costs more than staying; a flat price stays at the
    /// first rung. The first rung here has no third level to read (the ladder's top).
    #[test]
    fn a_step_whose_price_grows_with_the_step_takes_the_rung_of_the_least_work() {
        let (end, first, pilot, scout) = (1000., 1., 88., 11.);
        let steep = |h: f64| (1. + (h / 0.25).powi(4)) / h;
        let dear = move |h: f64| if h == 0.5 { 100. * steep(h) } else { steep(h) };
        let flat = |h: f64| 1. / h;
        let rung = |k: i32| first * 0.5_f64.powi(k);
        let at = |members: f64| Holding { pilot, members, pairs: members, below: 0. };
        let ladder = |prices: &Prices| -> f64 {
            let mut tau = first;
            while course(prices, tau, end, at(scout), false) == Course::Below {
                tau *= 0.5;
            }
            while course(prices, tau, end, at(pilot), false) == Course::Below {
                tau *= 0.5;
            }
            tau
        };
        let prices: [&dyn Fn(f64) -> f64; 3] = [&steep, &dear, &flat];
        for (p, work) in prices.into_iter().enumerate() {
            for product in [88., 200., 1000., MOST_TRAJECTORIES as f64] {
                let total = |k: i32| (1..=k).map(|j| scout * work(rung(j))).sum::<f64>() + (pilot - scout) * (work(rung(k)) + work(rung(k - 1))) + (product.max(pilot) - pilot) * work(rung(k));
                let rungs = (0..).take_while(|&k| end / rung(k) <= MOST_STEPS as f64);
                let best = rungs.min_by(|&a, &b| total(a).total_cmp(&total(b))).unwrap();
                let taken = ladder(&Prices { work, product });
                assert_eq!(taken, rung(best), "price {p}, product {product}: {:?}", (0..6).map(total).collect::<Vec<_>>());
                match p {
                    0 => assert!(taken == 0.25 || taken == 0.125, "{taken}"),
                    1 => assert!(taken < 0.5 && total(1) > total(0), "{taken}"),
                    _ => assert_eq!(taken, first),
                }
            }
        }
    }

    /// A third level whose price is more than what a rung below would add is not read: the rung stands on its own two
    /// levels (`course`, `buys`; module documentation, ## The ladder). The pilot's 88 pairs at 0.95 µs of a 1 ms run, a
    /// product of 287 trajectories, a flat price but the third level's steps dearer by a factor f: the rung below adds
    /// (N + P)/τ to staying's (N − P)/τ, so the third level, 88 f/(4τ), is read up to f = 4(N + P)/P = 17.05 and not beyond --
    /// one flip, and never the rung below, which is no cheaper. At the ladder's bottom (no rung below) it is read
    /// whatever its price. On the scout's position the pilot's purchase to come is priced into staying: the rung below
    /// adds 336.5/τ to the scout's staying (its companions at 2τ not yet run), so where the pilot would buy a third level
    /// dearer than that -- from f = 336.5/22 = 15.3 to the pilot's 17.05 -- the scout goes down instead, and beyond it,
    /// where the pilot would not buy it, the scout stands.
    #[test]
    fn a_third_level_dearer_than_a_rung_below_is_not_read() {
        let (end, tau, product) = (1000., 0.95, 287.);
        let (pilot, scout) = (Holding { pilot: 88., members: 88., pairs: 88., below: 0. }, Holding { pilot: 88., members: 11., pairs: 11., below: 0. });
        let priced = |f: f64, tau: f64, holding: Holding| {
            let work = move |h: f64| if h == 4. * tau { f / h } else { 1. / h };
            course(&Prices { work: &work, product }, tau, end, holding, true)
        };
        let factors: Vec<f64> = (0..=400).map(|k| 1. + 0.1 * k as f64).collect();
        let courses: Vec<Course> = factors.iter().map(|&f| priced(f, tau, pilot)).collect();
        let flips = courses.windows(2).filter(|w| w[0] != w[1]).count();
        let flip = 4. * (product + 88.) / 88.;
        assert!(courses[0] == Course::Third && courses[400] == Course::Stand && flips == 1, "{courses:?}");
        assert!(factors.iter().zip(&courses).all(|(&f, &c)| c == if f <= flip { Course::Third } else { Course::Stand }), "{flip}");
        let bottom = end / MOST_STEPS as f64 * 1.5;
        assert!(end / (0.5 * bottom) > MOST_STEPS as f64 && priced(1e6, bottom, pilot) == Course::Third);
        assert_eq!([1., 15.2, 15.4, 17., 17.1, 1e6].map(|f| priced(f, tau, scout)), [Course::Third, Course::Third, Course::Below, Course::Below, Course::Stand, Course::Stand]);
    }

    /// A read as a run logged it: its value and its far end at one look, each over the bound b of its look, on `groups`
    /// groups (the far end's interval Student's t at one look: the standard error back from it).
    fn logged(value: f64, most: f64, groups: f64, b: f64) -> Read {
        let t = Spread { error: 1., groups }.interval();
        Read { value: value * b, spread: Spread { error: (most - value.max(0.)) * b / t, groups } }
    }

    /// A rung within on two levels reads its third level where there is one, and else stands on its two (`verdict`;
    /// module documentation, # The step). e038's base at 0.589 µs read its total T against 1.178 µs at v/b −0.055 with a
    /// far end of 0.83 b on the pilot's eleven groups -- within --, and its T(1 ms) came out 79.1 against the converged
    /// 89.2 µK (−11.3 %) where two levels were all it was read on. The scout stands on it (the pilot reads it); the
    /// pilot reads its third level above where the ladder has the rung above and that rung stays in the model's domain,
    /// and else stands on its two levels, as an adaptive integrator does. A coarser pair that shows the error the two
    /// levels share -- a plateau of the error over τ and 2τ, read at 2τ against 4τ as 32 ± 1 b, lowered to 2 b, beyond
    /// the share -- does not show the order: the rung is left (e038's base path is caught by its 4τ). A third level
    /// within: the rung stands. Clearly beyond on its own read, or on its third level's: left, on the scout as on the pilot
    /// (R4_LIN_4's scout at 0.474 µs carried the left rung's loss lowered to 9.9 ± 1.5 b; it stood on its own read, and
    /// its pilot ran there before the rung below).
    #[test]
    fn a_rung_reads_its_third_level_and_else_stands_on_its_two() {
        let b = step_bound()[1].powi(2);
        let own = at(Some(logged(-0.055, 0.83, 11., b)));
        assert!(blind(&own).is_none() && !screen(&own) && within(&own));
        assert_eq!((verdict(&own, None, false, false), verdict(&own, None, true, true), verdict(&own, None, true, false)), (Verdict::Stands, Verdict::Above, Verdict::Stands));
        let coarser = carried(&at(Some(pilot(32. * b, b))), 0.5);
        assert!(screen(&coarser) && verdict(&own, Some(&coarser), true, false) == Verdict::Left);
        assert_eq!(verdict(&own, Some(&coarser), false, false), Verdict::Left);
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
    /// the third level's step four times the rung's and the pairs it was read on (the scout's eleven, where the scout
    /// went down a rung). A rung without has no third level to tell.
    #[test]
    fn a_rung_is_told_by_its_own_read_and_its_third_level_s() {
        let (b, tau, scales) = (step_bound()[1].powi(2), 1.4585, Scales { hbar_over_mass: 1., frequencies: [1.; 3] });
        let pilot_groups = || kept(11, GROUP as usize, 4, &|_, _| 4);
        let mut levels = Levels { fine: pilot_groups(), coarse: pilot_groups(), further: [kept(34, GROUP as usize, 4, &|_, _| 4), kept(34, GROUP as usize, 4, &|_, _| 4)], ..Levels::default() };
        let times = [0., 1., 2., 3., 4.];
        let (_, third, step, pairs) = levels.told(&times, f64::INFINITY, tau, &scales);
        assert_eq!((third.is_none(), step, pairs), (true, None, 360));
        let lowered = carried(&at(Some(pilot(0.8 * b, 0.1 * b))), 0.5);
        (levels.above, levels.above_pairs) = (Some(lowered.clone()), 11);
        let (_, third, step, _) = levels.told(&times, f64::INFINITY, tau, &scales);
        assert_eq!((third.unwrap().temperature, step), (lowered.temperature, Some((4. * tau, 11))));
    }

    /// A look on T at 800 trajectories: the model's error, what the step's check leaves open, the statistics' interval (each
    /// relative).
    fn look(model: f64, step: f64, interval: f64) -> Look {
        Look { target: TEMPERATURE_TARGET, account: Account { model, step, statistics: interval }, interval, trajectories: 800., shown: false }
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
            unresolved: [0.; 2],
            temperature_target: TEMPERATURE_TARGET,
        }
    }

    /// The share not counted (estimate.rs, # What is counted) in the accounts and in the result. Four groups of two
    /// trajectories at t = 0 and three outputs, every atom held in state 0 until it ends with no outcome: the first of
    /// each group after the first output, the second after the second. The model's error booked at 1 % on the held
    /// atoms' populations and 2 % on T, the step at its share. At the first output the accounts are the model's and T
    /// has one; at the second what is shown takes the uncounted half whole, 1 % + 50 %, and T has none; at the third,
    /// none counted, what is shown is booked whole, its model's part and its statistics' at one: those two outputs' looks
    /// on what is shown have no room. The result:
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
                let held = Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [1. + g as f64; 3], unresolved: false });
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
        assert_eq!(looks(&estimates, &fixed, step).iter().map(|l| l.room() > 0.).collect::<Vec<_>>(), [true, true, false, false, false]);
        let plan = Plan::new(&[0., 10., 20., 30.], [1.; 3]);
        let listed = vec![vec![]; plan.listed.len()];
        let context = Context {
            at: plan.times.clone(),
            plan,
            tau: 1.,
            scales,
            initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))),
            fixed,
            unresolved: vec![[0.; 2]; 4],
            step,
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let unmade = [Stopped::default(), Stopped { count: 2, since: Some(14.) }, Stopped::default()];
        let counts = Counts { steps: 0, unresolved: 0, refused: Stopped { count: 4, since: Some(11.) }, failed: Stopped { count: 3, since: Some(12.5) }, unmade, nowhere: Stopped { count: 6, since: Some(16.) }, runaway: Stopped { count: 5, since: Some(15.) }, optical: [0.0; 4] };
        let s = assemble(&context, &estimates, &estimates, &listed, &counts, 8);
        assert!(s.estimated_errors.iter().chain(&s.loss_probability).chain(s.populations.iter().flatten()).all(|x| x.is_finite()), "{s:?}");
        assert!(s.loss_probability[3] == s.loss_probability[2] && s.populations[3] == s.populations[2] && s.estimated_errors[3] >= 1., "{s:?}");
        let flagged = s.approximations.flagged.expect("flagged");
        for said in ["4 条轨迹的一步在所取步长下算出的数不是有限数（状态或参考流），在时间的舍入以内减半步长也推进不了（最早在 11 μs）：此后它们不计入估计", "3 条轨迹的跳跃给不出光子（模型在该状态上失败，缩短步长无济于事）（最早在 12.5 μs）：此后它们不计入估计", "2 条轨迹在某一步上，光学扇区的噪声算子 K 做不出（ℳ 奇异", "5 条轨迹的散射率超过了原子任何状态可能的上限（某个绝热消去远离它的适用范围，最早在 15 μs）：它们在那里停下", "6 条轨迹所在处的折叠不存在（起点落不到任何簇上、读出不是有限数、或在光阱区域外丢失能量不是有限数，最早在 16 μs）：它们在那里结束", "20 μs：有轨迹不计入估计，温度的误差无法估计", "30 μs 起：没有一条轨迹计入估计"] {
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
        let held = |g: usize, m: usize| Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [100. + (4 * g + m) as f64; 3], unresolved: false });
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
            unresolved: vec![[0.; 2]; 5],
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
    /// trajectories, each output one kind: every atom held; half lost; all lost; all held and unresolved; every atom held
    /// again; half not counted; none counted; one group holding two atoms; two groups holding (T's read on two groups);
    /// atoms that do not move (T zero); one atom held in all. At both levels, with the model's errors
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
        let held = |g: usize, m: usize, scale: f64, still: bool, unresolved: bool| {
            let y = if still { 0. } else { scale * (1. + (g * 4 + m) as f64) };
            Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == (g + m) % 8))), kappa: [if still { 0. } else { 0.1 * m as f64 }; 3], kappa2: [y; 3], unresolved })
        };
        let level = |scale: f64| -> Vec<Group> {
            (0..5)
                .map(|g| {
                    let mut group = Group::new(kinds + 1);
                    for m in 0..4 {
                        let fates: Vec<Fate> = (0..=kinds)
                            .map(|k| match k {
                                0 | 1 | 5 => held(g, m, scale, false, false),
                                2 => if m % 2 == 0 { Fate::Lost } else { held(g, m, scale, false, false) },
                                3 => Fate::Lost,
                                4 => held(g, m, scale, false, true),
                                6 => if m % 2 == 0 { Fate::Failed } else { held(g, m, scale, false, false) },
                                7 => Fate::Failed,
                                8 => if g == 0 && m < 2 { held(g, m, scale, false, false) } else { Fate::Lost },
                                9 => if g < 2 && m < 2 { held(g, m, scale, false, false) } else { Fate::Lost },
                                10 => held(g, m, scale, true, false),
                                _ => if g == 0 && m == 0 { held(g, m, scale, false, false) } else { Fate::Lost },
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
                assert!(e.unresolved.iter().all(|x| number(*x)) && number_or_unbounded(e.temperature_interval()), "output {k}: {e:?}");
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
                unresolved: vec![[0.; 2]; kinds + 1],
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
        assert_eq!(blind(&none), blind(&Between { loss: Read { value: 0., spread: Spread { error: 0., groups: f64::INFINITY } }, ..read.clone() }), "a look no time reads asks for nothing");
        assert_eq!(batch(f64::NAN, 88), 176);
        assert_eq!((probability_error(f64::NAN), probability_error(2.), probability_error(0.3)), (1., 1., 0.3));
    }

    /// The looks and the accounts are of what is shown and of the total T. A pilot that reads 25, 6 and 10 µK on the
    /// axes has the total 13.67 µK; with a model error of 3 % and a step's clear part of 2 % there is one look on what
    /// is shown and one on T at every output after t = 0. An interval of 1 % on the total neither keeps the run from
    /// stopping nor asks for a trajectory; one of 15 % does both, by (interval/room)², whatever the axes' own errors
    /// are. A model error or a step part that fills the target leaves its look no room. T's account holds the three parts,
    /// and the result's total is it. (The estimates' allowance is the plain 10 %; the step's bound on T is a share of
    /// it, 0.2 of the allowance for 2 %.)
    #[test]
    fn the_looks_and_the_accounts_are_of_the_total_temperature() {
        let fixed = Fixed { model: vec![[0.; 2], [0., 0.03]] };
        let step = Bound { shown: 0., temperature: 0.02 / TEMPERATURE_TARGET };
        let (n, total) = (800., (25. + 6. + 10.) / 3.);
        let quiet = estimate([25., 6., 10.], 0.005 * total, n);
        let run = [quiet.clone(), quiet];
        let crowded = |all: &[Look]| all.iter().filter(|l| !(l.room() > 0.)).count();
        let all = looks(&run, &fixed, step);
        assert_eq!(all.iter().map(|l| l.shown).collect::<Vec<_>>(), [true, false]);
        assert!(all[0].account.step == 0. && (all[1].account.step - 0.02).abs() < 1e-17);
        assert!(all.iter().all(|l| l.met() && l.need() < n) && crowded(&all) == 0);
        let noisy = estimate([25., 6., 10.], 0.075 * total, n);
        let run = [noisy.clone(), noisy];
        let t = looks(&run, &fixed, step)[1];
        let need = n * (CONFIDENCE * 0.075 / room(TEMPERATURE_TARGET, 0.03, 0.02)).powi(2);
        assert!(!t.met() && (t.need() - need).abs() < 1e-9 * need, "{} against {need}", t.need());
        let wide = Bound { shown: 0., temperature: 4. };
        assert!(!looks(&run, &fixed, wide)[1].fits() && crowded(&looks(&run, &fixed, wide)) == 1);
        let mut full = Fixed { model: fixed.model.clone() };
        full.model[1][1] = 0.11;
        assert!(crowded(&looks(&run, &full, step)) == 1);
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
    /// needs at an interval of 7.5 % beside a model error of 3 % and the step's share both move by a factor within
    /// (1 ± 1 %) between T and T(1 + 10⁻³) -- nothing switches at T_sc. And a look's room closes at one T and nowhere
    /// else: with the model's error on T at 30 %, T's look has no room above the T at which √(1 − s²) times the allowance
    /// is 30 %, not at T_sc.
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
                        let held = Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [kappa2; 3], unresolved: false });
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
            let crowded = !(looks(&[e.clone(), e], &Fixed { model: vec![[0.; 2], [0., 0.3]] }, Bound::share())[1].room() > 0.);
            (read, need, crowded)
        };
        let mut last = at(ts[0]);
        let mut turned = vec![];
        for (i, &t) in ts.iter().enumerate().skip(1) {
            let now = at(t);
            for (name, a, b) in [("the step's read", last.0, now.0), ("the need", last.1, now.1)] {
                assert!((b / a - 1.).abs() < 0.01, "{name} at {t} µK: {a} then {b}");
            }
            if now.2 != last.2 {
                turned.push(ts[i - 1]);
            }
            last = now;
        }
        // Where √(1 − s²) δT/T = 0.3: (0.3² / (1 − s²) − 0.1²) T² = T_sc².
        let closes = scale / (0.09 / (1. - STEP_SHARE * STEP_SHARE) - 0.01).sqrt();
        assert!(turned.len() == 1 && (turned[0] / closes - 1.).abs() < 3e-3, "the look turns at {turned:?}, the room closes at {closes} µK");
    }

    /// A state at the time t of the trajectory `index` with `jumps` jumps, each of a unit of counted work: no step reads it
    /// (nothing of the run's).
    fn bare_state(index: u64, t: f64, jumps: u32) -> State {
        State { sample: Sample { cluster: 0, at: crate::model::Point { r: [0.; 3], kappa: [0.; 3], t }, psi: vec![] }, index, jumps, jumped: f64::from(jumps), threshold: 0.5, owed: None, forms: crate::step::Forms::default(), held: crate::step::Held::default(), next: 0., unresolved: false }
    }

    /// The fold's readings per output (`Measured::means`) and their combination (`Measured::combined`, `model_error`):
    /// two states' rows -- each reading's error and their combination last --, output 1 read at both, output 2 at the
    /// first alone, output 3 at neither. A reading's error is the mean over the output's states, and its bound (1, none)
    /// where no state is read; the error booked is the mean of the states' combinations, none at t = 0.
    #[test]
    fn the_states_errors_are_the_outputs_means() {
        let width = READINGS.len() + 1;
        let row = |x: f64| -> Vec<Booking> { (0..width).map(|i| Booking::of(x * (1 + i) as f64)).collect() };
        let errors: Vec<Booking> = [row(0.01), row(0.02)].concat();
        let measured = Measured { errors: &errors, output: &[1, 2, 1], row: &[0, 0, 1], lost: &[false; 3] };
        let estimates: Vec<Estimate> = (0..4).map(|_| estimate([10.; 3], 0., 8.)).collect();
        let means = measured.means(4, &estimates);
        assert!(means[1].len() == READINGS.len() && (0..READINGS.len()).all(|i| (means[1][i][0] - 0.015 * (1 + i) as f64).abs() < 1e-15 && (means[2][i][1] - 0.01 * (1 + i) as f64).abs() < 1e-15), "{means:?}");
        assert_eq!(means[3], vec![[1., f64::INFINITY]; READINGS.len()]);
        let all = model_error(&measured.combined(4, &estimates));
        assert!((all[1][0] - 0.015 * width as f64).abs() < 1e-15 && (all[2][1] - 0.01 * width as f64).abs() < 1e-15, "{all:?}");
        assert_eq!((all[0], all[3]), ([0.; 2], [1., f64::INFINITY]));
    }

    /// A reading without bound or that is no number at a state is the state's whole (`Booking::of`): every row of the
    /// atom and its whole part of T; a reading x books x on both.
    #[test]
    fn a_reading_without_bound_is_the_state_s_whole() {
        for bad in [f64::INFINITY, f64::NAN] {
            assert_eq!(Booking::of(bad), Booking { error: [1.; 2] });
        }
        assert_eq!(Booking::of(0.01), Booking { error: [0.01; 2] });
    }

    /// A lost atom's fate is read along the path it was decided along (module documentation, # Where the atoms went;
    /// `approached`, `held_path`, `Booking::path`): analysis times 0, 10, 20, 30 and 40 µs, checkpoints every 6 µs.
    /// - An atom that escapes: held at 0 and 10 µs, past the far value from 24 µs on (`leaving`), found lost at 27 µs
    ///   (first lost at 30): its path ends at its last checkpoint below the far value, 18 µs, and is read at 0, 10 and
    ///   18 µs -- not at 20 µs, a state the run holds of it after its path's end, nor where it was found lost. Its
    ///   error is the trapezoid's mean on those three, whatever the errors past the far value (one there, a fold's
    ///   whole, as at the trap's rim of the 1D PGC tilted 3°).
    /// - An atom that goes near the rim and back, held there at 10 µs with a large error, and is lost later (first at
    ///   40, past the far value from 36 µs): its rim is a node of its path, weighed as the trapezoid weighs it.
    /// - An atom held near the rim at 20 µs and never lost: that state is a held row like every other, its error in the
    ///   held atoms' mean there (`Measured::means`).
    /// - An atom past the far value at every checkpoint (none below it) is read at t = 0 alone.
    #[test]
    fn a_lost_atom_s_fate_is_read_on_its_path_below_the_far_value_and_the_rim_where_held() {
        let state = |index: u64, t: f64| bare_state(index, t, 0);
        let trail = |index: u64, until: f64| -> Vec<Checkpoint> {
            (0..).map(|j| 6. * j as f64).take_while(|&t| t < until).map(|t| Checkpoint { trajectory: state(index, t).trajectory(), path: adapt::Path::new(SEED, index, 2., 2), carry: Scratch::default().carry() }).collect()
        };
        // The escaping atom (0), the one that goes to the rim and back (1), the one held at the rim (2).
        let leaving = |from: f64| move |t: &Trajectory| t.time >= from;
        let (first, second) = (trail(0, 27.), trail(1, 38.));
        let ends = [&first[approached(&first, &leaving(24.))], &second[approached(&second, &leaving(36.))]];
        assert_eq!((ends[0].trajectory.time, ends[1].trajectory.time), (18., 30.), "the paths' ends: the last checkpoints below the far value");
        assert_eq!(approached(&trail(3, 27.), &leaving(0.)), 0, "past the far value throughout: its path is its start");
        let ends = [State::of(&ends[0].trajectory), State::of(&ends[1].trajectory)];
        let times = [0., 10., 20., 30., 40.];
        let held = |index: u64, last: usize| (0..=last).map(move |k| (k, state(index, times[k])));
        let mut samples: Vec<Vec<State>> = vec![vec![]; times.len()];
        for (k, s) in held(0, 2).chain(held(1, 3)).chain(held(2, 4)) {
            samples[k].push(s);
        }
        let path0 = held_path(&samples, 3, &ends[0]).unwrap();
        assert_eq!(path0.iter().map(|(s, place)| (s.sample.at.t, *place)).collect::<Vec<_>>(), vec![(0., None), (10., Some((1, 0))), (18., None)]);
        let path1 = held_path(&samples, 4, &ends[1]).unwrap();
        assert_eq!(path1.iter().map(|(s, _)| s.sample.at.t).collect::<Vec<_>>(), vec![0., 10., 20., 30.], "the rim at 10 µs a node; its end at its last checkpoint below the far value, 30 µs");
        // The errors: a few parts in a hundred where the fold holds, 0.4 at the rim, one past the far value.
        let error = |index: u64, t: f64| -> Booking {
            let x = match (index, t) {
                (1, 10.) | (2, 20.) => 0.4,
                (0, t) if t >= 24. => 1.,
                (_, t) => 0.01 + 0.001 * t,
            };
            Booking { error: [x, x] }
        };
        let mean = |path: &[(&State, Option<(usize, usize)>)]| Booking::path(&path.iter().map(|(s, _)| (s.sample.at.t, error(s.index, s.sample.at.t))).collect::<Vec<_>>());
        let trapezoid = |nodes: &[(f64, f64)]| nodes.windows(2).map(|w| 0.5 * (w[1].0 - w[0].0) * (w[0].1 + w[1].1)).sum::<f64>() / (nodes[nodes.len() - 1].0 - nodes[0].0);
        let expected = [trapezoid(&[(0., 0.01), (10., 0.02), (18., 0.028)]), trapezoid(&[(0., 0.01), (10., 0.4), (20., 0.03), (30., 0.04)])];
        for (path, expected) in [mean(&path0), mean(&path1)].into_iter().zip(expected) {
            assert!(path.error.iter().all(|x| (x - expected).abs() <= 1e-15), "{path:?} against {expected}");
        }
        assert!(mean(&path0).error[0] < 0.03, "nothing of the error past the far value");
        assert_eq!(Booking::path(&[(0., Booking { error: [0.2, 0.3] })]), Booking { error: [0.2, 0.3] }, "a path of its start alone");
        // The held atom at the rim: a held row at 20 µs, beside the second atom's there (the first is lost by then, its
        // path its row).
        let errors: Vec<Booking> = [error(1, 20.), error(2, 20.), mean(&path0)].iter().flat_map(|&b| vec![b; READINGS.len() + 1]).collect();
        let measured = Measured { errors: &errors, output: &[2, 2, 2], row: &[0, 1, 2], lost: &[false, false, true] };
        let estimates: Vec<Estimate> = (0..3).map(|k| Estimate { loss: if k == 2 { 1. / 3. } else { 0. }, ..estimate([10.; 3], 0., 3.) }).collect();
        let at = measured.means(3, &estimates)[2][0][0];
        let expected = (2. / 3.) * 0.5 * (error(1, 20.).error[0] + 0.4) + (1. / 3.) * mean(&path0).error[0];
        assert!((at - expected).abs() <= 1e-15, "the held rim's error in the held atoms' mean: {at} against {expected}");
    }

    /// A pilot's groups with their states kept: `groups` groups of `members` trajectories, numbered as the pilot's are,
    /// each held at the outputs up to `held(group, member)` of `outputs` after t = 0. A state's time is its output's
    /// place.
    fn kept(groups: usize, members: usize, outputs: usize, held: &dyn Fn(usize, usize) -> usize) -> Vec<Part> {
        (0..groups)
            .map(|g| {
                // Three jumps a unit of time.
                let state = |m: usize, k: usize| bare_state(MOST_TRAJECTORIES + (g as u64) * GROUP + m as u64, k as f64, 3 * k as u32);
                let samples = (0..=outputs).map(|k| (0..members).filter(|&m| k <= held(g, m)).map(|m| state(m, k)).collect()).collect();
                Part { sums: Group::new(outputs + 1), reads: Group::new(outputs + 1), listed: vec![], samples, trails: vec![], lost: vec![], counts: Counts::default() }
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
                assert!(picks.iter().all(|p| p.weight == outputs as f64 && (p.state.index - MOST_TRAJECTORIES) / GROUP == p.group as u64 && p.jumping == 3.));
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
            assert_eq!((p.weight, p.state.sample.at.t as usize, p.jumping), (held as f64, 1 + (through * held as f64) as usize, 3.));
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
    /// every third step. An atom read at every time, its κ and κ² zero at the instant, takes its window's means whether it
    /// is read before the piece its time lies in is added or after it. Run through to the plan's end, every window's
    /// mean κ and κ² within 1e-12 of its own scale; an atom lost at 17.6 µs, the windows that reach past it over their
    /// held part, no atom after it. The times kept open are never more than a window's span of them and two.
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
        let atom = Fate::Held(Held { rows: [0.; STATES], kappa: [0.; 3], kappa2: [0.; 3], unresolved: false });
        let span = 2. * plan.halves.iter().flatten().copied().fold(0., f64::max);
        for stop in [plan.end, 17.6] {
            let mut windows = Windows::new(&plan.times, &plan.halves);
            windows.record(0, atom.clone());
            let (mut next, mut most) = (1, 0);
            for piece in cuts.windows(2).filter(|p| p[0] < stop) {
                let to = piece[1].min(stop);
                // The times in the piece read before it is added (the odd ones) and after it (the even ones).
                let inside: Vec<usize> = (next..times.len()).take_while(|&k| times[k] <= to).collect();
                inside.iter().filter(|&&k| k % 2 == 1).for_each(|&k| windows.record(k, atom.clone()));
                windows.add(piece[0], to, &moments);
                inside.iter().filter(|&&k| k % 2 == 0).for_each(|&k| windows.record(k, atom.clone()));
                next += inside.len();
                most = most.max(windows.open.len());
            }
            let fates = windows.close();
            for (k, fate) in fates.iter().enumerate().skip(1) {
                let Fate::Held(h) = fate else {
                    assert!(times[k] > stop && *fate == Fate::Lost, "stop {stop}, output {k}: {fate:?}");
                    continue;
                };
                assert!(times[k] <= stop, "stop {stop}, output {k} held after the loss");
                for a in 0..3 {
                    let (lo, hi) = (times[k] - plan.halves[k][a], (times[k] + plan.halves[k][a]).min(stop));
                    let exact: [f64; 3] = { let (p, q) = (antiderivative(a, lo), antiderivative(a, hi)); std::array::from_fn(|i| q[i] - p[i]) };
                    let scale = 1e-12 * stop * amp[a].max(1.).powi(2) / exact[0];
                    assert!((h.kappa[a] - exact[1] / exact[0]).abs() <= scale && (h.kappa2[a] - exact[2] / exact[0]).abs() <= scale, "stop {stop}, output {k}, axis {a}: {} and {} against {} and {}", h.kappa[a], h.kappa2[a], exact[1] / exact[0], exact[2] / exact[0]);
                }
            }
            assert!(most as f64 <= span + 2., "{most} times open against a span of {span} µs");
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
            unresolved: vec![[0.; 2]; 2],
            step: Bound::share(),
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let shape = Shape::of(&context.plan, false);
        let record = |i: u64| {
            let held = Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], unresolved: false };
            Record { outputs: vec![Fate::Held(held.clone()), Fate::Held(held.clone())], reads: vec![Fate::Held(held.clone()), Fate::Held(held)], listed: vec![Some(([0.; 3], [i as f64; 3])); shape.listed], trail: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2] }
        };
        let whole = [folded(&shape, (0..GROUP).map(record)), folded(&shape, (GROUP..2 * GROUP).map(record))];
        let scout: Vec<Part> = (0..3).map(|i| folded(&shape, std::iter::once(record(i)))).collect();
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
        messages.result(1, 4 * GROUP, true, None, || unreachable!("no result is made of one group"));
        assert_eq!(sent, [3, 3, 2 * GROUP, 2 * GROUP, 2 * GROUP]);
    }

    /// A run that integrates nothing hands out nothing (module documentation, # Exits; tilt2 at 24d346ce: every start
    /// landed nowhere, zero steps, the ladder walked to its bottom, t = 0's T carried to every output and "finished").
    /// Groups whose every trajectory ended with no outcome before the first output after t = 0 -- held there, failed
    /// after: a start that lands nowhere -- show nothing beyond t = 0 (`integrated`), make no checkpoint (`so_far`: the
    /// message stays due), and the request's exit says how many ended, why and from when (`nothing_integrated`). One
    /// trajectory counted after t = 0 among them is a result. And the step's check decides nothing on a read none of
    /// whose looks reads (`Verdict::Unread`, on the scout and on the pilot: the check ends unable to see where it is,
    /// no rung below), while a look no time reads among looks that do neither holds the rung nor leaves it: the pilot's
    /// rung whose loss and rows read within stands with its T unread (it went down the ladder before: its far end no
    /// number asked for no number of pairs).
    #[test]
    fn a_run_that_integrates_nothing_hands_out_nothing() {
        let context = Context {
            at: vec![0., Plan::new(&[0., 1.], [1.; 3]).end],
            plan: Plan::new(&[0., 1.], [1.; 3]),
            tau: 0.5,
            scales: Scales { hbar_over_mass: 1., frequencies: [1.; 3] },
            initial: [0.; STATES],
            fixed: Fixed { model: vec![[0.; 2]; 2] },
            unresolved: vec![[0.; 2]; 2],
            step: Bound::share(),
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let shape = Shape::of(&context.plan, false);
        let held = |i: u64| Fate::Held(Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], unresolved: false });
        let nowhere = |i: u64| Record {
            outputs: vec![held(i), Fate::Failed],
            reads: vec![held(i), Fate::Failed],
            listed: vec![None; shape.listed],
            trail: vec![],
            lost: None,
            counts: Counts { nowhere: Stopped::at(0.), ..Counts::default() },
            scattered: [0.; 2],
        };
        let none = [folded(&shape, (0..GROUP).map(nowhere)), folded(&shape, (GROUP..2 * GROUP).map(nowhere))];
        let all: Vec<&Part> = none.iter().collect();
        assert!(!integrated(&all));
        let mut sent = vec![];
        let mut send = |s: &Solution, _: Option<f64>| sent.push(s.trajectories);
        let mut messages = Messages { send: &mut send, clock: Instant::now() - CHECKPOINT, last: None, least: 0 };
        so_far(&context, &none, &[], &mut messages);
        assert!(messages.due().is_zero(), "no result of what shows nothing beyond t = 0");
        let reason = nothing_integrated(&counts_of(&all), 2 * GROUP, context.plan.times[1]);
        println!("{reason}");
        assert!(reason.starts_with("没有一条轨迹被积分：先导的 16 条轨迹都在 t = 0 之后的第一个输出时刻（1 μs）之前") && reason.contains("16 条轨迹所在处的折叠不存在") && reason.contains("最早在 0 μs"), "{reason}");
        let one = folded(&shape, std::iter::once(Record { outputs: vec![held(0), held(0)], reads: vec![held(0), held(0)], ..nowhere(0) }));
        let some: Vec<&Part> = none.iter().chain(std::iter::once(&one)).collect();
        assert!(integrated(&some));
        so_far(&context, &none, &[&one], &mut messages);
        assert!(!messages.due().is_zero());
        assert_eq!(sent, [2 * GROUP + 1]);
        // The step's reads.
        let unread = Read { value: f64::NAN, spread: Spread::UNREAD };
        let nothing = Between { loss: unread, shown: [unread; STATES], temperature: Some(unread) };
        assert!(read_looks(&nothing).is_empty() && blind(&nothing).is_none() && !screen(&nothing));
        for (pilot, upward) in [(false, false), (true, false), (true, true)] {
            assert_eq!(verdict(&nothing, None, pilot, upward), Verdict::Unread);
        }
        let b = step_bound()[1].powi(2);
        assert_eq!(verdict(&at(Some(unread)), Some(&at(Some(read(0.1 * b, 0.01 * b)))), true, false), Verdict::Stands);
        assert_eq!(verdict(&at(Some(unread)), None, true, true), Verdict::Above);
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
            Record { outputs: vec![Fate::Lost], reads: vec![Fate::Lost], listed: vec![], trail: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2] }
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

    /// A scene's request (refs/scenes, or refs/requests where `requests`), its run at the ground state `state` (the
    /// reference state) and that state's prepared trap, as the solve makes them (`solve`, `Prepared`).
    fn scene_run(name: &str, requests: bool, state: usize) -> (Request, Onefold, Trap) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join(if requests { "requests" } else { "scenes" }).join(format!("{name}.json"));
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let run = Onefold::new(&request, alone(state)).unwrap();
        let trap = Prepared::new(&run, alone(state)).unwrap().trap;
        (request, run, trap)
    }

    /// The initial ground state of a request of the largest population (the solve's reference state).
    fn chief(request: &Request) -> usize {
        (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap()
    }

    /// The rms speed along an axis of a request's atoms at its temperature, as a momentum over ħ (1/µm).
    fn thermal(request: &Request, run: &Onefold) -> f64 {
        (KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt()
    }

    /// The PGC3D_1ms trajectory that hung on 2cd4e77d (e2e.status R1: its decisions rose by the state's population read off
    /// a forward that did not exist, up to all six nodes -- the tweezer's D1 F′ = 1, 2 among them, 25 THz off --, and a
    /// stretch planned 1.56·10⁷ Taylor steps): the reference state at the trap's centre, three thermal speeds along z,
    /// trajectory 9, τ = end/800, the cells 2τ. It runs to the plan's end (or its loss), each cell's decisions taking the
    /// remedy their attribution names (unify2 §3.4; a fold that does not exist rises the node of the most mixing of the
    /// part that diverges): the explicit states of every cluster it holds said, and none of the tweezer's line.
    #[test]
    fn the_pgc3d_trajectory_that_hung_runs_through_on_its_own_light_s_nodes() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("PGC3D_1ms.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let b = chief(&request);
        let (request, run, trap) = scene_run("PGC3D_1ms", true, b);
        let domain = Domain { light: run.light(), widest: Widest::of(&request) };
        let speed = thermal(&request, &run);
        let mut plan = Plan::new(&[0., 100., 200.], trap.lab);
        let tau = plan.end / 800.;
        plan.cell = 2. * tau;
        let kappa: V3 = [0., 0., 3. * speed];
        let index = 9;
        let mut t = Trajectory::begun(&run, b, trap.centre, kappa, SEED, index, plan.cell);
        let mut path = adapt::Path::new(SEED, index, plan.cell, (plan.cell / tau).log2().round() as u32);
        let mut scratch = Scratch::default();
        let started = std::time::Instant::now();
        // The tweezer's line: the excited states its frequency drives from the ground (the request's couplings).
        let trap_light = run.light().trap;
        let tweezer: Vec<usize> = (0..crate::request::STATES - GROUND).filter(|&e| run.light().frequencies[trap_light].beams.iter().any(|&l| (0..GROUND).any(|g| request.beams[l].coupling[(GROUND + e, g)].abs() > 0.0))).collect();
        let mut held: Vec<Vec<usize>> = vec![];
        let mut ended = Ok(());
        while t.time < plan.end && t.lost.is_none() {
            let stop = t.time + 10.;
            ended = adapt::advance(&run, &domain, tau, &mut t, &mut scratch, &mut path, plan.end, stop, &[], &mut |_, _, _| {}, &mut |_, _| {});
            let explicit: Vec<usize> = run.cluster(t.cluster).states.iter().filter(|&&s| s >= GROUND).map(|&s| s - GROUND).collect();
            if !held.contains(&explicit) {
                held.push(explicit);
            }
            if ended.is_err() {
                break;
            }
        }
        eprintln!("the trajectory to {:.1} µs of {:.1} ({:?}; lost {:?}), {} jumps, {} unresolved steps, in {:.1} s; the explicit excited states of the clusters it held {held:?}; the tweezer's line {tweezer:?}", t.time, plan.end, ended, t.lost, t.jumps, t.unresolved_steps, started.elapsed().as_secs_f64());
        assert!(ended.is_ok(), "it runs through");
        assert!(held.iter().flatten().all(|e| !tweezer.contains(e)), "no node of the tweezer's line rises");
    }

    /// The price a step's check weighs its courses by is the run's count (`work_at`): R4_LIN_2's run, at states of its
    /// atoms at the trap's centre at rest and at one to three times the thermal speed along each lab axis, held (no jump,
    /// each its own representation: its start's), at two steps -- a trajectory's work a µs priced at every state is that
    /// state's step's own count (`Step::work`, its scratch primed as a run's) over τ and its jumps' counted work a µs
    /// (`Pick::jumping`), the mean over the states by their weights, to the rounding.
    #[test]
    fn a_state_s_price_is_its_step_s_count() {
        let (request, run, trap) = scene_run("R4_LIN_2", false, 0);
        let b = chief(&request);
        let speed = thermal(&request, &run);
        let widest = Widest::of(&request);
        let states: Vec<State> = (0..7)
            .map(|i| {
                let kappa: V3 = std::array::from_fn(|a| if i > 0 && a == (i - 1) % 3 { (1 + (i - 1) / 3) as f64 * speed } else { 0. });
                let t = Trajectory { threshold: f64::MIN_POSITIVE, ..Trajectory::new(&run, b, trap.centre, kappa, SEED, i as u64) };
                State::of(&t)
            })
            .collect();
        let picks: Vec<Pick> = states.iter().enumerate().map(|(i, state)| Pick { state, group: i, weight: 1. + i as f64, jumping: 1e5 * i as f64 }).collect();
        for tau in [0.2, 0.4335] {
            let counted = |from: &Trajectory| {
                let (through, mut scratch) = (crate::step::Midway::new(&run, from, tau, None).unwrap(), Scratch::default());
                through.attempt(from, &crate::step::Brownian::STILL, &[], &mut scratch, &widest).unwrap();
                through.attempt(from, &crate::step::Brownian::STILL, &[], &mut scratch, &widest).unwrap().work
            };
            let made = over_the_run(&picks, picks.iter().map(|p| counted(&p.state.trajectory()) / tau + p.jumping), None);
            let priced = work_at(&run, widest, &picks)(tau);
            eprintln!("R4_LIN_2 at {tau} µs: priced {priced:.6e}, counted {made:.6e} a µs");
            assert!((priced - made).abs() <= 1e-12 * made, "{tau}: {priced} against {made}");
        }
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
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], unresolved: false };
            let outputs = vec![Fate::Held(held.clone()), if i % 3 > 0 { Fate::Held(held) } else { Fate::Lost }];
            Record { reads: outputs.clone(), outputs, listed: vec![Some(([i as f64; 3], [0.; 3]))], trail: vec![], lost: None, counts: Counts { steps: i, ..Counts::default() }, scattered: [0.; 2] }
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
                let one = folded(&shape, members.clone().map(|m| record(number * GROUP + m)));
                assert_eq!((&part.sums, &part.reads), (&one.sums, &one.reads));
                assert_eq!(part.listed, one.listed);
                assert_eq!(part.counts.steps, one.counts.steps);
            }
        }
    }

    /// A trajectory that panics ends its hand-out with its own panic, at once (`run_groups`): 37 groups of eight
    /// trajectories of 40 ms each, the fourth of the first group panicking with its message -- the hand-out's caller
    /// unwinds with that panic before half the time the other trajectories would take on the pool (no trajectory more
    /// is handed out: before, all of them ran first, a product's batch minutes, and the app waited on a worker that
    /// said nothing), and the pool serves the next hand-out.
    #[test]
    fn a_trajectory_s_panic_ends_its_hand_out_with_it() {
        let shape = Shape { outputs: 2, reads: 2, listed: 1, keep: false };
        let numbers: Vec<u64> = (2048..2053).chain(5..37).collect();
        let each = Duration::from_millis(40);
        let record = |i: u64| {
            assert!(i != 2048 * GROUP + 3, "a trajectory's defect at {i}");
            std::thread::sleep(each);
            Record { reads: vec![Fate::Lost; 2], outputs: vec![Fate::Lost; 2], listed: vec![None], trail: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2] }
        };
        let mut silent = |_: &Solution, _: Option<f64>| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0 };
        let started = Instant::now();
        let hands = [Hand { groups: &numbers, members: 0..GROUP, shape, record: &record }];
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_groups(&hands, &mut messages, &mut |_, _| {})));
        let took = started.elapsed();
        let message = outcome.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
        eprintln!("the hand-out ended after {took:?}: {message}");
        let whole = each * (numbers.len() as u32 * GROUP as u32).div_ceil(rayon::current_num_threads() as u32);
        assert!(message.contains("a trajectory's defect at 16387") && took < whole / 2, "{took:?} against {whole:?}: {message}");
        let again = run_groups(&[Hand { groups: &numbers[1..2], members: 0..GROUP, shape, record: &record }], &mut messages, &mut |_, _| {});
        assert_eq!(again[0].len(), 1);
    }

    /// The initial ensembles of the conservative reference traps (refs/scenes R1, the app's own requests: a lone tweezer, a
    /// standing wave, crossed beams of one frequency, a σ⁺ tweezer) in the run's prepared wells (`Prepared`: the trap
    /// light's held forms on the fold) under their envelopes (`Reference::peak`): 2¹² draws each read the exact moments (ħ/m)⟨κ_a²⟩ = Σ_b p_b η_b k_BT (S_b²)_aa per lab axis within their
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
            let run = Onefold::new(&request, alone(chief(&request))).unwrap();
            let initial = components(&request);
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&run, alone(s)).unwrap()).collect();
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

    /// The runaway's guard on the run (step.rs, # The domain): R2_RECOIL's atoms of |2,2⟩ through the trap's centre at 1
    /// to 1000 times the thermal speed along each lab axis, run for 0.8 µs at 0.005 µs in the solve's domain. An atom whose
    /// state's hazard at its first step's middle is above the fastest excited state's decay, wholly excited, stops there,
    /// is counted and held from then on as it stood, unresolved (booked whole), its rows numbers; one whose hazard is
    /// below half of it runs to the end untouched. The fold at the atom rises a node where its readings leave the share
    /// (model/fold/run.rs # The nodes risen and fallen), so its rate stays a state's: how many stop is said, not asked.
    #[test]
    fn a_fold_scattering_beyond_any_state_stops_its_atom_held_unresolved() {
        let (request, run, trap) = scene_run("R2_RECOIL", false, 7);
        let b = 7;
        let widest = Widest::of(&request);
        let domain = Domain { light: run.light(), widest };
        let plan = Plan::new(&[0., 0.2, 0.4, 0.6, 0.8], trap.lab);
        let speed = thermal(&request, &run);
        let (mut stopped, mut ran) = (0, 0);
        for (i, factor) in [1., 10., 30., 100., 300., 1000.].into_iter().enumerate() {
            for a in 0..3 {
                let kappa: V3 = std::array::from_fn(|x| if x == a { factor * speed } else { 0. });
                let start = Trajectory::new(&run, b, trap.centre, kappa, SEED, (3 * i + a) as u64);
                let middle: V3 = std::array::from_fn(|x| trap.centre[x] + 0.5 * TAU_GUARD * run.hbar_over_mass() * kappa[x]);
                let excitation = {
                    let m = run.middle(start.cluster, middle, 0., &start.held, false);
                    let mut h = crate::linalg::Mat::zeros(start.psi.len(), start.psi.len());
                    crate::step::Middle::generator(&m, &crate::step::Middle::path(&m, kappa, [0.; 3]), 0., &mut h);
                    crate::model::testing::decay_rate(&h, &start.psi) / widest.0
                };
                let record = run_trajectory(&run, &domain, Start { state: b, r: trap.centre, kappa }, (3 * i + a) as u64, TAU_GUARD, &plan);
                if record.counts.runaway.count > 0 {
                    stopped += 1;
                    for (k, fate) in record.outputs.iter().enumerate().skip(1) {
                        match fate {
                            Fate::Held(h) => assert!(h.unresolved && h.rows.iter().all(|x| x.is_finite()), "output {k}: {h:?}"),
                            other => panic!("output {k} of a stopped atom not held: {other:?}"),
                        }
                    }
                }
                if excitation > 2. {
                    assert_eq!(record.counts.runaway.count, 1, "an atom of excitation {excitation} at {factor} × the thermal speed runs on");
                }
                if excitation < 0.5 {
                    ran += 1;
                    assert_eq!(record.counts.runaway.count, 0, "an atom of excitation {excitation} at {factor} × the thermal speed stopped");
                }
            }
        }
        eprintln!("R2_RECOIL on the run: {stopped} atoms beyond any state stopped, {ran} within ran");
        assert!(ran > 0, "{stopped} stopped, {ran} ran");
    }

    /// The decisions' reads are the outputs' onlookers (module documentation, # The analysis times; adapt.rs, # The
    /// reads): R2_RECOIL's atoms in |2,2⟩ at the trap's centre, at a third to three times the thermal speed along each lab
    /// axis, run through 2 µs read at 3, 41 or 401 outputs, at 800 steps to the run's end (the last output's window), and
    /// at the candidates m·2τ: their fates and states there are the same to the bit whatever the outputs, and so are their
    /// counts; an output at a candidate's time reads the candidate's fate (their windows are one where the output's is not
    /// cut by the run's end).
    #[test]
    fn the_decisions_reads_are_the_same_whatever_the_outputs() {
        let (request, run, trap) = scene_run("R2_RECOIL", false, 7);
        let b = 7;
        let domain = Domain { light: run.light(), widest: Widest::of(&request) };
        let speed = thermal(&request, &run);
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
        for factor in [0.3, 1.2, 3.] {
            for a in 0..3 {
                let kappa: V3 = std::array::from_fn(|x| if x == a { factor * speed } else { 0. });
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
                let states = |r: &Record, plan: &Plan| replay(&run, &domain, tau, plan.end, &r.trail, &plan.reads).iter().map(|s| s.as_ref().map(|s| (s.sample.clone(), s.index, s.jumps))).collect::<Vec<_>>();
                assert!(kept.iter().zip(&plans).all(|(r, plan)| states(r, plan) == states(&kept[0], &plans[0]) && r.lost.as_ref().map(|l| l.0) == kept[0].lost.as_ref().map(|l| l.0)));
                atoms += 1;
            }
        }
        eprintln!("{atoms} atoms through three output grids; {shared} outputs at a candidate's time");
        assert!(shared > 0);
    }

    /// A kept trajectory's states are its own, made again from its checkpoints (module documentation, # The analysis
    /// times; `run`, `replay`; adapt.rs `advance`: a stop is no step's end) -- its representation with them (step.rs
    /// `Held`, carried by the checkpoint's trajectory: the decompositions a decision made and its switches' balance) --,
    /// on its class: R2_RECOIL's atoms in |2,2⟩ at the trap's centre at 1 to 100 times the thermal speed along each lab
    /// axis through 2 µs at 800 steps, and PGC3D_1ms's at a half to twice it through 200 µs (tens of jumps an atom, the
    /// representation's decisions at every cell's end, the steps' routes chosen by what the scratch carries, step.rs
    /// `Carry`) -- atoms held to the end and lost --, read at the candidates m·2τ, the checkpoints a fifth of the run
    /// less a little apart (off the step's grid). Kept, a trajectory's fates, listed atoms and counts are those of the
    /// same trajectory advanced through at once; and its states replayed at every candidate, and at a few of them, are to
    /// the bit those that one advance through the run reads there -- none after its loss.
    #[test]
    fn a_kept_trajectory_s_states_are_made_again_from_its_checkpoints_to_the_bit() {
        let key = |s: &Option<State>| s.as_ref().map(|s| format!("{:?} {} {} {:?}", s.sample, s.index, s.jumps, s.forms));
        let (mut ends, mut lost, mut most) = (0, 0, 0);
        let started = Instant::now();
        for (name, requests, last, factors) in [("R2_RECOIL", false, 2., [1., 10., 100.]), ("PGC3D_1ms", true, 200., [0.5, 1., 2.])] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join(if requests { "requests" } else { "scenes" }).join(format!("{name}.json"));
            let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
            let b = chief(&request);
            let (request, run, trap) = scene_run(name, requests, b);
            let domain = Domain { light: run.light(), widest: Widest::of(&request) };
            let speed = thermal(&request, &run);
            let mut plan = Plan::new(&[0., 0.5 * last, last], trap.lab);
            let tau = plan.end / 800.;
            plan.cell = 2. * tau;
            plan.read_at(grid(plan.end, tau));
            plan.window = plan.end / 5.3;
            for factor in factors {
                for a in 0..3 {
                    let kappa: V3 = std::array::from_fn(|x| if x == a { factor * speed } else { 0. });
                    let (start, index) = (Start { state: b, r: trap.centre, kappa }, (a + 7) as u64);
                    let [through, record] = [false, true].map(|keep| super::run(&run, &domain, start, index, tau, &plan, keep));
                    assert_eq!((&record.outputs, &record.reads, &record.listed, format!("{:?}", record.counts)), (&through.outputs, &through.reads, &through.listed, format!("{:?}", through.counts)), "{name}: kept against through at {factor} × along {a}");
                    assert!(record.trail[0].trajectory.time == 0. && record.trail.windows(2).all(|w| w[0].trajectory.time < w[1].trajectory.time));
                    most = most.max(record.trail.len());
                    // The states one advance through the run reads at the candidates.
                    let mut t = Trajectory::new(&run, b, start.r, start.kappa, SEED, index);
                    let mut path = adapt::Path::new(SEED, index, plan.cell, (plan.cell / tau).log2().round() as u32);
                    let mut direct: Vec<Option<State>> = vec![None; plan.reads.len()];
                    direct[0] = Some(State::of(&t));
                    let advanced = adapt::advance(&run, &domain, tau, &mut t, &mut Scratch::default(), &mut path, plan.end, plan.end, &plan.reads, &mut |k, s, _| direct[k] = Some(State::of(s)), &mut |_, _| {});
                    assert!(advanced.is_ok(), "{name} at {factor} × along {a}: {advanced:?}");
                    if t.lost.is_some() {
                        lost += 1;
                    } else {
                        ends += 1;
                    }
                    let few: Vec<usize> = (0..plan.reads.len()).filter(|k| k % 97 == 0 || k + 1 == plan.reads.len()).collect();
                    let times: Vec<f64> = few.iter().map(|&k| plan.reads[k]).collect();
                    let again = replay(&run, &domain, tau, plan.end, &record.trail, &plan.reads);
                    let sparse = replay(&run, &domain, tau, plan.end, &record.trail, &times);
                    assert!(again.iter().zip(&direct).all(|(x, y)| key(x) == key(y)), "{name}: every candidate at {factor} × along {a}");
                    assert!(sparse.iter().zip(&few).all(|(x, &k)| key(x) == key(&direct[k])), "{name}: a few candidates at {factor} × along {a}");
                    eprintln!("{name} at {factor} × along {a}: {} jumps, {}, {:.1} s", t.jumps, if t.lost.is_some() { "lost" } else { "held" }, started.elapsed().as_secs_f64());
                }
            }
        }
        eprintln!("{ends} atoms held to the end, {lost} lost; {most} checkpoints at the most");
        assert!(ends > 0 && lost > 0 && most >= 5, "{ends} held, {lost} lost; {most} checkpoints");
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
        let held = Fate::Held(Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [0.; 3], kappa2: [1.; 3], unresolved: false });
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

    /// A group's records added in their order (`Shape::add`).
    fn folded(shape: &Shape, records: impl Iterator<Item = Record>) -> Part {
        records.fold(shape.empty(), |mut part, r| {
            shape.add(&mut part, r);
            part
        })
    }

    /// One trajectory through the plan (`run`) with its states not kept.
    fn run_trajectory(run: &Onefold, domain: &Domain, start: Start, index: u64, tau: f64, plan: &Plan) -> Record {
        super::run(run, domain, start, index, tau, plan, false)
    }

    /// The model's error booked is the fold's readings combined at each state, measured at the states
    /// (`Measured::combined`, `model_error`), on its class: random pilots' reads -- 11 groups, a state an output and group,
    /// random readings and their combinations, some outputs read by no group -- book at every output the mean over the
    /// states of each state's own combination (its row's last), and where no state reads an output, its bound (1, none);
    /// t = 0 books nothing.
    #[test]
    fn the_model_s_error_booked_is_the_states_combined_readings() {
        let mut stream = crate::random::Stream::jump(11, 3, 0);
        let (groups, outputs, width) = (11_usize, 6_usize, READINGS.len() + 1);
        for trial in 0..20 {
            let (mut errors, mut output) = (vec![], vec![]);
            for _ in 0..groups {
                for k in 1..outputs {
                    if (k + trial) % 5 == 0 || stream.uniform() < 0.1 {
                        continue;
                    }
                    errors.extend((0..width).map(|_| Booking { error: [0.01 * stream.uniform(), 0.02 * stream.uniform()] }));
                    output.push(k);
                }
            }
            let (row, lost) = rows(output.len());
            let measured = Measured { errors: &errors, output: &output, row: &row, lost: &lost };
            let estimates: Vec<Estimate> = (0..outputs).map(|_| estimate([10.; 3], 0., 88.)).collect();
            let booked = model_error(&measured.combined(outputs, &estimates));
            assert_eq!(booked[0], [0.; 2], "t = 0 books nothing");
            for k in 1..outputs {
                let reads: Vec<usize> = (0..output.len()).filter(|&i| output[i] == k).collect();
                if reads.is_empty() {
                    assert_eq!(booked[k], [1., f64::INFINITY], "trial {trial} output {k} read by no state");
                    continue;
                }
                let mean = |i: usize| reads.iter().map(|&r| errors[r * width + width - 1].error[i]).sum::<f64>() / reads.len() as f64;
                assert!((0..2).all(|i| (booked[k][i] - mean(i)).abs() <= 1e-15), "trial {trial} output {k}: {:?}", booked[k]);
            }
        }
    }

    /// The model is judged where the atoms went (module documentation, # The model; `Measured::means`), on its class:
    /// random pilots' reads -- 11 groups, five outputs after t = 0 of every kind of loss (none, some, every atom lost),
    /// per group a held atom's state where the group holds one and a lost atom's path where it lost one, random readings
    /// -- book on what is shown (1 − L) times the held reads' mean and L times the lost reads', on T the held reads' alone
    /// (none where no atom is held), each reading and their combination alike. Every atom lost (S2Q's first model, every
    /// atom out within 10 µs): what is shown books the lost atoms' paths, the output is read, and a fate's error of 0.4
    /// leaves its look no room -- where the held reads alone booked nothing at all.
    #[test]
    fn every_atom_s_fate_is_read_held_or_lost() {
        let mut stream = crate::random::Stream::jump(13, 5, 0);
        let (groups, outputs, width) = (11_usize, 6_usize, READINGS.len() + 1);
        let lost_share = [0., 0., 0.3, 0.7, 1., 1.];
        let (mut errors, mut at, mut row, mut lost) = (vec![], vec![], vec![], vec![]);
        for k in 1..outputs {
            for _ in 0..groups {
                for (kind, present) in [(false, lost_share[k] < 1.), (true, lost_share[k] > 0.)] {
                    if !present {
                        continue;
                    }
                    row.push(errors.len() / width);
                    errors.extend((0..width).map(|_| Booking { error: [0.2 * stream.uniform(), 0.1 * stream.uniform()] }));
                    (at.push(k), lost.push(kind));
                }
            }
        }
        let measured = Measured { errors: &errors, output: &at, row: &row, lost: &lost };
        let estimates: Vec<Estimate> = (0..outputs).map(|k| Estimate { loss: lost_share[k], ..estimate([10.; 3], 0., 88.) }).collect();
        let (means, combined) = (measured.means(outputs, &estimates), measured.combined(outputs, &estimates));
        for k in 1..outputs {
            let reads = |kind: bool| (0..at.len()).filter(|&i| at[i] == k && lost[i] == kind).collect::<Vec<_>>();
            let (held, gone) = (reads(false), reads(true));
            let mean = |of: &[usize], u: usize, i: usize| if of.is_empty() { 0. } else { of.iter().map(|&r| errors[row[r] * width + u].error[i]).sum::<f64>() / of.len() as f64 };
            let l = lost_share[k];
            for u in 0..width {
                let shown = (1. - l) * mean(&held, u, 0) + l * mean(&gone, u, 0);
                let booked = if u < READINGS.len() { means[k][u] } else { combined[k] };
                assert!((booked[0] - shown).abs() <= 1e-15 && (booked[1] - mean(&held, u, 1)).abs() <= 1e-15, "output {k} column {u}: {booked:?}");
            }
        }
        // Every atom lost by output 4: what is shown is the lost paths' mean, read; a fate's error of 0.4 there leaves the
        // look no room.
        let errors = vec![Booking { error: [0.4, 0.] }; width];
        let (at, row, lost) = (vec![4; groups], vec![0; groups], vec![true; groups]);
        let measured = Measured { errors: &errors, output: &at, row: &row, lost: &lost };
        let means = measured.means(outputs, &estimates);
        assert!((means[4][0][0] - 0.4).abs() < 1e-15 && means[4][0][1] == 0., "{:?}", means[4]);
        let fixed = Fixed { model: model_error(&measured.combined(outputs, &estimates)) };
        let looks = looks(&estimates, &fixed, Bound::share());
        assert!(!(looks.iter().filter(|l| l.shown).nth(3).unwrap().room() > 0.), "output 4's look on what is shown");
    }

    /// A run is priced where its atoms were (`one_each`): a trajectory weighs by the time it holds its atom, and one lost
    /// before the first output gives its state at t = 0 for its own time. Eleven groups of eight, every atom lost at
    /// 0.1 to 0.9 of the first output's time (S2Q's first model): every trajectory gives its state at t = 0, weighs by
    /// its time to the loss, and its jumps a µs are its last state's; a mean over the run of a value reads that value,
    /// where the held states alone read nothing.
    #[test]
    fn a_run_whose_atoms_are_all_lost_is_priced_where_they_were() {
        let (groups, members, outputs) = (11_usize, GROUP as usize, 4_usize);
        let parts: Vec<Part> = (0..groups)
            .map(|g| {
                let state = |m: usize, t: f64, jumps: u32| bare_state(MOST_TRAJECTORIES + (g * members + m) as u64, t, jumps);
                let loss = |m: usize| 0.1 + 0.8 * ((g * members + m) % 9) as f64 / 8.;
                let mut samples = vec![vec![]; outputs + 1];
                samples[0] = (0..members).map(|m| state(m, 0., 0)).collect();
                let lost = (0..members).map(|m| (1, state(m, loss(m), 6))).collect();
                Part { sums: Group::new(outputs + 1), reads: Group::new(outputs + 1), listed: vec![], samples, trails: vec![], lost, counts: Counts::default() }
            })
            .collect();
        let all: Vec<&Part> = parts.iter().collect();
        let picks = one_each(&all);
        assert_eq!(picks.len(), groups * members);
        for p in &picks {
            let lost = &parts[p.group].lost.iter().find(|(_, s)| s.index == p.state.index).unwrap().1;
            assert_eq!((p.state.sample.at.t, p.weight, p.jumping), (0., lost.sample.at.t, 6. / lost.sample.at.t));
        }
        assert!((over_the_run(&picks, picks.iter().map(|_| 2.5), None) - 2.5).abs() < 1e-15);
        // An atom that ends with no outcome before the first output (no state, no loss) gives none.
        let mut silent = parts[0].clone();
        silent.lost.clear();
        assert!(one_each(&[&silent]).is_empty());
    }

    /// The room the check's product is priced at is what the step leaves the model as far as its reads hold it
    /// (`standing`; `attempt`): at most the step's share on every look, the read's estimate where it is below. PGC3D at
    /// 1000 outputs: a model error of 0.0480 on what is shown has no room beside the step's share (its threshold 0.0447)
    /// and has room beside its read step error of 0.0022 (threshold 0.0499); a read beyond the share, not clearly (the
    /// rung to be left), books the share: the rung the check stands at is within it.
    #[test]
    fn the_room_the_check_prices_at_is_what_the_step_as_read_leaves() {
        let b = step_bound()[0].powi(2);
        let below = Between { loss: read(0.0022f64.powi(2), 1e-7), shown: std::array::from_fn(|_| read(0.0022f64.powi(2), 1e-7)), temperature: Some(read(0.01, 1e-4)) };
        let above = Between { loss: read(1.5 * b, 0.5 * b), ..below.clone() };
        let (low, high, share) = (standing(&below), standing(&above), Bound::share());
        assert!((low.shown - 0.0022).abs() < 1e-12 && (low.temperature - 0.1).abs() < 1e-12, "{low:?}");
        assert_eq!(high, Bound { shown: share.shown, temperature: 0.1 });
        assert!((room(POPULATION_TARGET, 0., share.shown) - 0.04472).abs() < 1e-5 && (room(POPULATION_TARGET, 0., low.shown) - 0.04995).abs() < 1e-5);
        let estimates = vec![estimate([10.; 3], 0., 88.), estimate([10.; 3], 0., 88.)];
        let fixed = Fixed { model: vec![[0.; 2], [0.0480, 0.]] };
        assert!(!(looks(&estimates, &fixed, share)[0].room() > 0.) && looks(&estimates, &fixed, low)[0].room() > 0.);
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
                Fate::Held(Held { rows: state(0), kappa, kappa2, unresolved: false })
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
                unresolved: vec![[0.; 2]; 3],
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

}
