//! The pipeline: request, run, pilot, step's check, batches, checkpoints, result (SPEC §3, §5, §9).
//!
//! # The pipeline
//! 1. The request's checks; the initial components; the selection's structural chain (select.rs `choose`) and the run
//!    on its cheapest candidate, the full-ground model G (model/selected.rs `Selected`: the scene, P and its RS fold,
//!    one object for every step of an attempt) -- or on the choice kept for the scene (# The selection) --; every
//!    initial component's prepared well and the initial ensemble drawn in them, the reported trap, the kick's curvature
//!    (`solve`, `Setting`); the outputs' plan, the first rung (# The step).
//! 2. The scout: the first member of every pilot group, at the first rung and at twice it -- or at half it, ## A companion
//!    at τ/2 -- (# The step), screened; where
//!    its rung stands on its reads, its states price the rungs (## The ladder) and the pilot runs at the one the run is
//!    expected to spend least on. The pilot: the scout's groups whole, `pilot_size` trajectories at the run's two levels
//!    (the scout's are its groups' first members and are not run again). Its groups have numbers of their own, from
//!    trajectory `MOST_TRAJECTORIES` on: beyond every number a group of the product takes, which is its place in the
//!    run's list (`product`). It checks the step -- on two levels, a third above and a rung below where they pay, and on
//!    further pairs where they pay (# The step) --, every read at the analysis times, not at the outputs (# The analysis
//!    times). Its trajectories at the run's step stay in the result: they are trajectories of the run and its step.
//!    On G, where no choice is kept, the pilot's trajectories then choose P (# The selection): G chosen, the product goes
//!    on from them; another, the attempt is made again on it, its scout and pilot its own.
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
//! Brownian path (adapt.rs) --, at 2τ, or at τ/2 where that reads the rung at less work for its precision (## A companion at
//! τ/2) (`Levels`), both read at the same instants: the analysis times' candidates, the
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
//! ladder has the rung above; `buys`: their price no more than what a rung below would add, ## The ladder) -- as
//! many of the pilot's groups as its read asks at the best, three at the least (`Third::pairs`; on more of them where
//! it cannot see, as further pairs are taken for an own read): its pairs are its read's precision's, not the pilot's
//! number (tilt2's third level on all 88 of its trajectories was a third of its pilot's work, each trajectory's jumps
//! as dear at 4τ as at τ). Where
//! the order holds across the triple both reads are of one mean square, e(τ)'s. A rung stands (`verdict`) where both
//! estimates are within the share, M̂ ≤ b on every look (`within`) -- the larger of the two, as multilevel Monte Carlo
//! takes the larger of its finest difference and the next one lowered by the order, so that two levels that agree by
//! being both too coarse are not taken for converged where the third shows the error --, and it is left where the
//! third's read is clearly beyond: the order is not shown above it. A third level's estimate beyond the share and not
//! clearly is what an own read's is, the noise hiding it: its pairs, or the rung below, by their price (`remedy`) --
//! one rule for both reads. Where no third level is to be had or bought -- a first
//! rung at the ladder's top, a rung above whose paths left the model's domain or whose read is no number, a third level
//! dearer than a rung below would make the run --, the rung stands
//! on its own two levels: the change between levels within the tolerance on the estimates, as an adaptive integrator's
//! local control and a refinement checked by hand take it (the user's approximate convergence, QuTiP-like: delivered
//! fast, the risk of two coarse levels that agree accepted there). The third level bought on the pilot lies above the
//! rung's pair or below it -- (2τ, 4τ) above, (τ/2, τ) below; against τ/2, (τ, 2τ) and (τ/4, τ/2) --, three adjacent
//! levels either way, at the side whose read costs the less for its precision (`third_of`, `read_cost`): the pairs its
//! read needs, its lift above or below and its factor on the rung's own noise, at its levels' work (## A companion at
//! τ/2; the factor foretold by the states in hand where that level is not run, measured where it is) -- the one rule the
//! companion is chosen by. Above,
//! the triple holds the rung's own pair at its finest; a coarser level that took most of its steps at their halves is out
//! of the order's range there, and its read can only not see (PGC3D, TET at 0.948 µs: 3.79 µs read 6 to 8 times the
//! share's square, the rung's own read within). Below, its read is raised 2^{2p} times and is the noisier where the pairs
//! have parted (read so on 88 pairs a clean rung was refused about two times in five, S3 113 to 133 CPU-s an attempt):
//! its noise is in its read's spread, which the verdict reads as every read's. A level held below the rung is the rung
//! below's own level, or its companion: its states are kept, and the ladder going down runs it no more. The rung books what it
//! stands by: per look the larger estimate, √max(M̂, 0) (`checked`, `worse`) -- at most the share, and the number the
//! criterion held. What three levels cannot tell: an error the same at all three -- the scheme saturated across a factor
//! four of the step. Where every difference is below the noise no number of levels tells it from convergence.
//!
//! ## A companion at τ/2
//! A difference d(h) = Q(2h) − Q(h) is (2^p − 1) e(h) only where both levels took their steps at their own lengths. The
//! advance takes a refused step at its half (adapt.rs `halving`), so a level some of whose steps were halved errs as
//! its steps do: to the leading order the steps' local errors add up along the run, and a level of the step s errs by
//! the time mean of C ℓ^p over the lengths ℓ it took, C s^p r with r = 1 − Σ ℓ (1 − (ℓ/s)^p)/T (adapt.rs `Halved`: the
//! halved steps' shortfall in the order over the time run) -- r = 1 − f (1 − 2^{−p}) where its halved steps took one half
//! each over a share f of its time. Where the coarser of two levels is so made, d = (2^p r − 1) e(h) and the plain read
//! d/(2^p − 1) is low by (2^p r − 1)/(2^p − 1), 1 − f for steps halved once; the read takes it back (`read_step`,
//! `amplified`): its value and its jackknife spread, both of mean squares, times ((2^p − 1)/(2^p r − 1))² -- e(h) =
//! d/((1 − f)(2^p − 1)) for steps halved once. r is measured on the trajectories the read is made on, not chosen, and is
//! one where no step was halved. The finer level's halved steps make e(h) smaller and d larger: the read errs high, and
//! is left so. (Taken as it is, the coarser level halved, a read is low about 1 − f times: on the scouts' steps GMD1C,
//! PGC3D and TET at 1.897 µs refused 5.6, 7.9 and 3.3 % of their attempts; motion3d_status/rewrite/prof1.status.md.)
//! The read so taken back is the noisier, its noise raised with its value, so where the level at 2τ took many steps at
//! their halves the rung is read against τ/2 instead, on the same trajectories -- the same initial atoms, jumps' draws and
//! Brownian path, as the companion at 2τ is (`Levels::companion`, `halve_companion`). Between τ/2 and τ, d' = Q(τ) −
//! Q(τ/2) = e(τ) − e(τ/2) = (1 − 2^{−p}) e(τ) where the order holds, so
//!   e(τ) = d'·2^p/(2^p − 1) = 2^p·[d'/(2^p − 1)]:
//! the read of the pair (τ/2, τ), d'/(2^p − 1) = e(τ/2) (estimate.rs `between`, its finer level τ/2's error; the level at
//! τ its coarser, its own halved steps taken into it as above), carried to τ by the order -- 2^{2p} times in the mean
//! square, value and spread (`carried` at the ratio two), as a third level's read is carried down by the same factor --:
//! one estimator and its jackknife over the same groups, taken one rung lower and raised to the rung. The relation of the
//! two: against 2τ, e(τ) = d/(2^p − 1) with d = (2^p − 1) e(τ); against τ/2, e(τ) = d'·2^p/(2^p − 1) with d' = (2^p − 1)
//! e(τ)/2^p -- one e(τ) where the order holds over (τ/2, τ, 2τ). Against τ/2 the read is of a level at τ whose steps all
//! took their length: where the run's took some at their halves, its error is the less, and the read errs high. Where
//! the pairs have parted, their difference carries the two ensembles' own noise, the same at either rung, and the read
//! raised 2^{2p} times is so much the noisier than the one against 2τ; where they stay together the noise falls with the
//! error and the two are alike (## The ladder: the noise's ratio). Its estimate decides as every read's does (`verdict`),
//! its noise priced as every read's is (`remedy`).
//! Which of the two: the one whose read costs the less work for its precision (`against_half`, `read_cost`): the pairs
//! it needs to read within the share at the best, at its levels' work a µs as priced, its refused attempts in it
//! (`Prices::work`). A read's noise is raised with its value -- by its factor a for its coarser level's halved steps, and
//! by its lift (τ/h)^{2p} from its pair's finer step h to the rung's --, so where its pairs have parted, their difference
//! the two ensembles' own noise at either step, it needs λ a times the pairs the rung's own read's noise asks with its
//! lift and factor undone (`raw_pairs`): against 2τ λ = 1 and a = a(2τ), against τ/2 λ = 2^{2p} and a = a(τ) -- the
//! τ/2 read costs 16 times the pairs before its factor where the read against 2τ costs a(2τ) times, on the pilot's pairs
//! at least. Where the pairs stay together the noise falls with the error and the lift is no cost: the rule errs toward
//! 2τ there. (Priced at w·a alone, the lift left out, it took the side below for GMD1C's and PGC3D's third levels, whose
//! reads on parted pairs came out ±3.5 b against the ±0.03 b above.) It is decided on a priced pass of
//! the check (`check`), before the read decides anything: where 2τ took any step at its half the check stops for its
//! prices (`Settled::Pending`) whatever the read against 2τ says -- it may be of no comparison of the steps at all (R4
//! at 0.474 µs, 0.948 µs's time 44 % on halved steps: a row at 7.1 ± 1.8 b against 2τ taken back, 2.2 ± 0.6 b as it is,
//! 0.06 ± 0.16 b against τ/2 on the same 44 pairs; the coarser level beyond the order's range where it was halved) --,
//! and it is read against τ/2 at once
//! where that read is without bound (2^p r ≤ 1: the level at 2τ no coarser than τ). At the ladder's bottom, where τ/2 is
//! beyond the steps' cap, the read against 2τ is the one there is. The companion at τ/2 costs twice the rung's steps
//! where the one at 2τ cost half, and the ladder prices every rung at the companion it would be read against (`staying`,
//! `descending`). A rung read against τ/2 has no third level above: the level above its companion at 2τ took many
//! steps at their halves, and twice its step takes more; it stands on its two levels, or on the third it inherited
//! from the rung left above (that rung's read, 2τ against 4τ, 4τ's halved steps taken back). Its rung below is its
//! companion's own level, read against τ -- the rung's own read, uncarried: the rung below's read is had with no
//! trajectory more -- or against τ/4 where that costs the less. What the reads against 2τ say, as they are and taken
//! back, is noted beside the read against τ/2 on the same pairs wherever the companion turns (`halve_companion`, the
//! feature `prof`), for the data to judge by; it decides nothing. (Decided first so that any halved step of 2τ turned
//! the companion: the scenes whose companions halve a step in 10⁴ -- tilt0's 0.652 µs, R4's 0.474 µs, S1's 0.948 µs --
//! would have run their pilots at twice the work for reads low by less than 10⁻³.)
//!
//! ## The ladder
//! - A refused step: a step the scheme refuses at a rung's length -- the decomposition's for an entry of no route
//!   (spectral.rs, # The split: neither the following's series nor the hops' converges on it, a length beyond both
//!   radii), a propagation whose norm grows, a state or a reference flow that is no number -- is taken at its half, and
//!   so on (adapt.rs `halving`): an exact adaptive subdivision, the trajectory's error at those steps the shorter
//!   steps' own, so no rung is left for it. What it costs is the rung's price: a state's step is priced as the run takes
//!   it, at the length it is taken at (`work_at`), so a rung whose steps are refused costs what their halves do, and it
//!   is left only where its reads leave it or a rung below is expected to cost the run less (`course`). A companion
//!   whose steps were halved so errs as its steps do: its read takes them back, or the rung is read against τ/2 (## A companion at
//!   τ/2) where that costs the less. (Left on its
//!   first refused step, as the ladder was before -- the scout's own level holding a step of no route, and a companion's
//!   refused step priced without bound --, GMD1C, PGC3D and TET stood on their reads at 0.948 µs and went down to
//!   0.474 µs for their companions' refused steps at 1.897 µs (5.6, 7.9 and 3.3 % of their steps), and R4 left 0.474 µs
//!   for 6 of its 18 561 steps: 35 to 51 % of their runs' wall time; motion3d_status/rewrite/prof1.status.md.) What the
//!   rung leaves of the readings is its read's to tell: the decomposition's own remainders on a step (spectral.rs
//!   `Picture::remainder`) bound the internal state's error over the step alone, and the readings' differences at the
//!   rungs a run stands at are the splitting's second order, which they do not see (R4: 1e-6 to 1e-3 of the state a
//!   step at 0.474 µs, against T_z 7.7 ± 2.3 µK apart from 0.237 µs), so no rung is left by them.
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
//!   rung's levels (the scout's one a group, or all), n the pairs at the rung, c the rung's companion's step (2τ, or
//!   τ/2: ## A companion at τ/2) and r_j the trajectories run at τ/2^j beside the members at τ (the scout's pairs at
//!   τ/2 or τ/4, `foresee`; the members at τ/2 where that is the companion):
//!   - staying: the pilot's companions still to run and the run level to the most of N, P and n, (P − P_m) w(c) +
//!     (max(N, P, n) − n) w(τ) (`staying`), and the third level N₃ w(4τ) where the rung buys it, N₃ the pairs its read
//!     asks in whole groups (`Third::pairs`);
//!   - the k-th rung below, τ_k = τ/2^k, reached a rung at a time (each rung's own level its next one's companion, or
//!     the next one's own level its companion): the members in hand at every rung on the way, the pilot's run level at
//!     τ_k and its companions at τ_{k−1} -- or at τ_{k+1}, where that reads it at less work for its precision: τ's halved steps as its members
//!     took them, τ/2's where it is run, a level not yet run as one whose steps are whole (a halving is known once it has run, and a
//!     finer step is refused less) --, every level beyond what it holds already,
//!     Σ_{j<k} (P_m − r_j)⁺ w(τ_j) + (max(N, P) − r_k)⁺ w(τ_k) + the companions' (P − what they hold)⁺ w(c_k); the
//!     least of them (`descending`), priced down to where the levels on the way alone cost more than the least found
//!     (every other term is no less than zero) or to the ladder's bottom. (Each rung's companion the rung above, as the
//!     ladder was: P_m Σ_{j≤k} w(τ_j) + (P − P_m) w(τ_{k−1}) + (max(N, P) − P_m) w(τ_k) − r_1 w(τ_1), the same.)
//!   The rung below is taken where some rung below is expected to cost less than staying (`course`): a smaller step is
//!   the more accurate one, and the rung below stands only on its own reads, as every rung does, so no error enters
//!   the choice -- it is the work's alone. Its third level above is this rung's read, lowered (`carried`).
//! - Two paths from a rung its reads stand (`course`): verified there -- its companions and, where it has one to read,
//!   its third level, each at its work for its read's precision (`read_cost`: the pairs its noise needs, lifted and
//!   taken back as its pair is, at its levels' work with the refused attempts in it), and its run level to the product
//!   --, or verified at the least rung below, whose third level is this rung's read, inherited and lowered, for
//!   nothing. The rung takes the cheaper, on the scout before the pilot is bought and again after every stretch the check
//!   runs, at what is held then (what is spent is no part of either path). A rung with no third level to be had (a
//!   ladder's top with no level below to read) stands on its own two levels: the QuTiP-like path. At the ladder's bottom,
//!   with no rung below, it reads the third wherever there is one. (Before, a third dearer than the rung below added
//!   left the rung on its own two levels, and a third's price was its plain work: PGC3D's scout stayed at 0.948 µs for a
//!   third at 3.79 µs priced 1.2e9 -- its read's precision there 3.7e9, out of the order's range, unable to see -- and went
//!   down only after 190 s.)
//! - The third level above as read, not foretold (`Scouted`): where the path at a rung is the cheaper on the scout with
//!   its third level above as foretold -- its pairs the rung's own noise lifted and taken back by the factor the price
//!   states foretell --, the scout reads that level first on its own pairs, the eleven trajectories at 4τ, 11 w(4τ)
//!   (held where the pilot then reads it there), and the two paths are weighed again on that read, by the same rule:
//!   its noise and its factor as measured -- the pairs it asks at the best (`blind`; none where it reads within) -- and
//!   whether it stands in the order's range of the rung's own read at all (`in_range`). Under the order the two reads
//!   are of one mean square: the third's, carried to τ, is the own one's times (d₂/d₁)²/2^{2p}, d₁ = Q(2τ) − Q(τ) and
//!   d₂ = Q(4τ) − Q(2τ). Where on some look the near end of one lies above the far end of the other -- each read with
//!   its own statistical error, no threshold of its own --, the ratio of the differences is not 2^p: the steps above
//!   the rung are out of the order's range, no pairs make that read the rung's, and the path at the rung takes the
//!   side below as foretold, or is without bound. (Foretold, TET's third at 3.79 µs was cheap, its factor foretold
//!   too small; the pilot ran there 200 s to find it out of the range, one row 6.0 ± 2.1 of the share's square against
//!   the own read's 0.10 ± 0.09 on 88 pairs; motion3d_status/rewrite/r2ladder.status.md.)
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
//!   - its own estimate within, and its third level's where it has one: its course (`course`) -- the rung below where
//!     it costs less; else the third level above where the rung has none and buys it --; else it stands on its own two
//!     levels, or on its third where it has one;
//!   - its own estimate, or its third level's, beyond the share and not clearly: one rule for both reads, each the
//!     rung's mean square where the order holds. The third level's pairs are at its own levels, 2τ and 4τ: where it
//!     is not read on the pilot's trajectories at 4τ (the scout's eleven pairs lowered, or the rung left above's), it
//!     is read there first, P w(4τ); beyond the pilot's, further pairs at the rung's three levels (`remedy`). With
//!     the own read within, the rung below is foreseen within on the pilot's pairs (its error the order's less, its
//!     noise no more), and costs its run level. For the own read:
//!     the noise hides the difference, or the error itself lies
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
//! axis, cut at t = 0) takes a trajectory's κ and κ² over the cuts of its steps' paths (step.rs `Segment::stretch`: the
//! path with the kick spread over the step, of the scheme's order at any cut), and its shape -- its κ on the window's
//! Legendre polynomials, which T takes the cloud's own motion out by (estimate.rs, # The cloud's own motion); the first
//! is the instant t = 0, and the run goes on through the last output's window (`Plan`). Outputs closer than a period
//! have overlapping windows, so a window is not summed by itself: per lab axis the trajectory's running integrals of 1,
//! κ and κ² from t = 0, and its running moments of κ against the powers of the time about a grid of origins
//! (estimate.rs `Shapes`), are kept at every window's two ends, and a window is their difference (`run`) -- each piece
//! of a step's path evaluated once (step.rs `Stretch`: its panels' sums, and up to each end inside it the panel's
//! polynomial), whatever the windows' overlap.
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
//! excited amplitudes following the ground state (the lift, model/rs.rs `ModelAt::lift`) -- has not happened at the instant they are
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
//!   the trajectory and its Brownian path at a step's end: adapt.rs `advance`'s stops), at t = 0 and at the first step's end at or past every multiple of T's shortest window
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
//! - The model's error is no reading of the times: it is the choice's account at each of them (# The selection).
//! - What stands on them: the step's mean square (estimate.rs `between`, each time by its share of the run); the
//!   pilot's states there, the model's error and the looks'; the product's looks, batches and unresolved atoms; the prices at one state of every
//!   trajectory among them (`one_each`).
//! - What the result shows at an output: its own estimates and their standard errors; its accounts -- the model's
//!   error, the step's part, the statistics and T's allowance -- on the line through the accounts at the analysis
//!   times (`assemble`, `accounts_along`), so that the ledger flags where the stop reads, whatever the outputs. (Judged
//!   on its own statistics, a noisy estimate of the same spread, the pgc3d example stopped at the analysis times was
//!   flagged at 11 of 1000 outputs, at most 5.4 % with 5.1 % of statistics, and 2 of 100, its reason "not finished".)
//!
//! # The model (SPEC §5.2)
//! An attempt's model is one object, made before its trajectories run (RUSTPLAN §3.2; model/selected.rs `Selected`),
//! and decides nothing during it. What the solve books of the model is its choice's account (# The selection; select.rs
//! `Choice::at`, `Choice::motion`), at every analysis time: on what is shown the path-measured direct part, and the
//! motion's -- the loss's, T's per lab axis, by the time-scaling identity on the run's own pilot's course (select.rs,
//! # The motion; `course_of`) --; beside them
//! and the bounds of `bounded`. An
//! attempt with no choice -- G's before it chooses, a test's model of its own -- books none and says so (the ledger's
//! flag): its error is not measured.
//!
//! # The selection (PIPELINE_V3_FINAL §二 B.7–B.11; select.rs)
//! Where no choice is kept for the scene (select.rs `kept`: the request without its times), the attempt on G chooses P
//! once its pilot stands (`choose_on_the_pilot`). Where G's fold scatters beyond every state of the atom at every one of
//! its pilot's paths' starts (step.rs, # The domain: the runaway, which no step's length helps; SAT_3's resonant
//! D2-3-3, its first-order excitation 1.5), no path goes past its start and nothing can be chosen on them: the attempt
//! is made on the structural chain's next model, and so on up (`piloted`; the top folds nothing), and the choice is made
//! on that pilot among the chain's models above it (G and the others below it ran away where the atoms start).
//! 1. The paths: `select::WALKS` of the pilot's trajectories at the run's step (`walks`) -- twice as many, and so on up
//!    to all of them, while a candidate weighed is complete at its direct part's lower end and not at its upper (its
//!    fate open at the paths' statistics, Student's t at the paths less one; select.rs, # How many paths), every
//!    candidate judged again on them before any is scouted --, cut at L = min(T, 5 τ_mem),
//!    τ_mem the full-ground model's memory time, the longest over their starts; the 32-state reference along them, at
//!    the setting read along the longest of them (select.rs `Paths`); the paths stratified by the time their atoms
//!    were held (`walks`: those held to the end and those lost, half each, each weighed by its stratum's share).
//! 2. T's memory is not read here: the flag of its being no shorter than the run is the product's, on the run's own
//!    trajectories at its end (`memory_at_the_end`; select.rs, ## T's memory, read on the run's own trajectories).
//! 3. The candidates (select.rs # The candidates): the pilot's model judged, its parts at the analysis times and the
//!    looks they leave (`looks`); a candidate whose looks that the exact model's would meet are met by its account is
//!    complete, its work W the product's trajectories its looks need times its work over the run, and -- not the
//!    pilot's -- the scout's and the pilot's again on it. Where a model with more states could work less, each addable
//!    state's addition alone is judged on two paths; those that meet every look there are judged alone on every path and
//!    weighed (what tells their W apart is the paths' spread, which two paths do not see); the nominees are the prefixes
//!    of the states' order -- those singles by their W, then the others by the misfit each takes away per work it adds
//!    --, and from the first predicted to fit each is judged on every path until one is complete, above it while its
//!    least W -- at no model error, at the ladder's first rung's step, its pilot again -- stands below the least found.
//! 4. The complete ones but the pilot's are scouted at their own step in the order of their least W while it stands
//!    below the least found; P is the complete candidate of least W. The choice is kept for the scene, and said (stderr
//!    and the ledger's items): every candidate's parts, work and fate, the stages' times.
//! The pilot's step check is priced at no model error while the choice is not made: the error is measured on the
//! paths of the step it settles.
//!
//! ## Where the atoms went
//! The model is judged where the atoms went: the paths are the pilot's trajectories in two strata by the time each held
//! its atom (`walks`; select.rs # How many paths) -- those held to the run's end, whose paths read the late windows, and
//! those lost before it, whose paths went to the trap's rim --, each weighed by its stratum's share. A lost atom's fate
//! is its loss, and what the model's error does to its time is the motion's, booked by the time-scaling identity with
//! the lost paths' own x (select.rs, # The motion): R9_PUSH's full-ground model scatters 11 to 12 % above the 32 states
//! on its paths and pushes its atoms out that much early, its loss 0.18 above the reference at 26 µs. (Until 2026-11-22
//! the paths were the pilot's first trajectories, scans of the trap's axes marked a region the paths did not reach, and
//! the atoms' share in it booked its error: the scans took 10–80 % of the selection and measured a straight line's
//! error, not a path's -- 10 to 100 times the paths' window means --, and S1's four first atoms were all lost by
//! 1.5 ms, its last windows unread and booked whole -- Study A §2.3.) A run's price is read where its atoms were
//! (`one_each`): a trajectory weighs by the time it holds its atom -- a lost one's to its path's end --, and one lost
//! before the first time read by its state at t = 0.
//!
//! # The unresolved atoms
//! A step asks its domain one cheap thing (`Domain`; step.rs, # The domain): whether a state scatters beyond every state
//! of the atom, its hazard above the fastest excited state's decay wholly excited (`Widest`). The trajectory stops there
//! (`run`): unresolved from then on, its state as it stood. What an unresolved atom shows is booked whole (`bounded`):
//! its share of the held atoms on what is shown -- an atom's row is a probability, so its error is at most one, whatever
//! its state, and the mean over the held atoms errs by at most the share --; on T the strict bound of that share of the
//! held atoms moved anywhere in the bound set (estimate.rs `Estimate::moved`: a held atom's kinetic energy is at most
//! K_max, SPEC §1.7), with the trajectories not counted, each as one held atom of unknown state (estimate.rs, # What is
//! counted) -- one bound for every held atom whose state the run does not know. (Until 2026-11-21 its share of the held
//! atoms' kinetic energy: a bound only while its true kinetic energy was within twice the computed one.) The loss of an
//! atom lost after such a step is no held atom's: as every model error's, it is in what the held atoms' errors are
//! weighed by (the held share, `accounts`), and not booked apart.
//!
//! # Exits
//! Two exits, each said where it is met, none carried on as a number (unify2 §7):
//! - the request's (refused before a trajectory runs, with its reason): the run's construction (scene.rs `Scene::new`,
//!   select.rs `choose`, model/selected.rs `Selected::new`), a prepared trap
//!   that does not hold the atom (`trap_of`), no step below the edge (`first_rung`);
//! - a trajectory's (`run`), where the model gives it no number: its start's held forms none, or its fast part's photon
//!   none (step.rs `Trajectory::new`), or its readout at a time read is no number. It ends there with no outcome
//!   (`Counts::nowhere`, `Fate::Failed`
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
//! E² = A² + c² + S² (ledger.rs; `Look`): A the model's error, the choice's account at the time (# The model, # The
//! selection) and the optical sector's booking, and c
//! what the step's check leaves open (its rms over the run), both fixed by the pilot -- A with what the atoms whose
//! state the run does not know carry, read with every look (# The unresolved atoms; `bounded`) --; S the statistics'
//! interval. The
//! run stops where every one of them lies within its target: the test the ledger flags an output by, at the outputs
//! (`Look::met`). The stop reads the run's standard errors, its unresolved atoms, and nothing else of its estimates. A
//! look that no run's trajectories bring within its target -- its need beyond `MOST_TRAJECTORIES`, or A² + c² ≥ ε²
//! (`Look::fits`): the model's error booked is the selected P's account, which no number of trajectories lowers -- is
//! not chased: the ledger flags it.
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
//! two trajectories of a thread (`run_groups`). What runs no trajectories sends no result and looks at no clock: the
//! run's construction, the model's error at the states, the prices.
//!
//! # Steps
//! Beside the results, the solve says where its pipeline stands (`Stage`, `Messages::stage`) -- what no result can say
//! where no trajectory runs: the selection, the model, and which of the scout, the pilot and the product's batches the
//! trajectories are of. A stage is sent as a step begins (`solve`: the model, and the selection where it is the kept
//! one; `attempt`: the scout and the pilot; `choose_on_the_pilot`: the selection; `product`: every batch, with its
//! trajectories, the number it brings the run to and how far its looks stand from their targets) and as the step's
//! check moves to another rung (`check`) -- when it changes, never otherwise. The order (the app's, src/solve-progress.ts):
//! - a new selection: Model (the structural chain and G's model are the model's making) → Scout (G's) → Pilot →
//!   Select (along the pilot's paths; a candidate's short scout is part of it and says nothing of its own) → where the
//!   choice is not G, Model → Scout → Pilot → Product; where it is G, Product;
//! - the choice kept for the scene: Select (at once) → Model → Scout → Pilot → Product.
//! Select and Model have no rung (`Stage::tau` none: the app takes a step there for no progress); the scout, the pilot
//! and the product are at theirs.
//! It decides nothing, is no part of the result and is kept nowhere: its one reader is the app's course of a running
//! solve (src/solve-progress.ts).
use crate::adapt;
use crate::analysis::analysis_times;
use crate::decide::{Outcome, Read};
use crate::ensemble::{Ensemble, Start, Well, GROUP};
use crate::estimate::{self, allowance, between, draws_for, half_window, readings, semiclassical, summarize, Between, Escape, Estimate, Fate, Group, Held, Local, Scales, Shapes, Spread, NO_SHAPE, SHAPE};
use crate::field::{Geometry, Light};
use crate::ledger::{percent, significant, span, when, Account, Ledger};
use crate::linalg::{symmetric3_eigen, V3};
use crate::model::selected::{Potential, Selected};
use crate::select::{Choice, Selection};
use crate::model::optical::Unmade;
use crate::model::potential::descend;
use crate::model::{Sample, Trap, GROUND};
use crate::request::STATES;
use crate::step::{stumpff, Dynamics, Integrals, Refused, Scratch, Segment, Trajectory};
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
/// What the engine did with each part of the model's error (the ledger's items' `action`, shown by the app; module
/// documentation, # The model, # The selection): the paths' direct part.
const MODEL_DIRECT: &str = "沿先导的原子路径，候选模型与 32 态模型推进内态密度矩阵、比较显示的各行（按记忆时间分窗的路径平均 + 2 倍路径间统计误差），另加路径格式对其自身细化（子步、分段各加倍）的校准";
/// The motion's part of the model's error (module documentation, # The model; select.rs, # The motion): the ledger's
/// item, read by the app's error budget list.
const MODEL_MOTION: [&str; 2] = ["模型误差：内态误差经运动对读数的影响", "沿路径实测散射率、力与能量变化率，按时间缩放恒等式一阶换算成丢失与每个方向温度的误差（路径平均 + t 分位数 × 路径间统计误差；丢失按丢失原子自己的能量变化率）"];


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

/// The solve of a request (module documentation): every message handed to `send` -- each checkpoint with the estimated
/// wall time left (ms) where the run's pace gives one, and each step the pipeline enters (# Steps) --, the final
/// solution returned; partial results are never withheld.
///
/// An attempt's run is one object, the same at every step of the attempt (model/selected.rs `Selected`: the scene, P
/// and its fold): what the solve takes of it beside the steps -- the prepared potentials and their traps, the initial
/// ensemble drawn in them, the reported trap, the kick's curvature (`Setting`) -- is made from it (`solve_with`), and
/// the pipeline runs on it (`attempt`). The first is the choice kept for the scene, or else G (where G's fold runs away
/// at every start of its pilot, the chain's first model that does not, `piloted`), whose pilot makes the choice (# The
/// selection): it chosen, its attempt is the solve; another, the attempt is made again on it. A request
/// it refuses (the scene, a trap that does not hold the atom, no step below the edge) is refused with its reason,
/// before a trajectory runs.
pub fn solve(request: &Request, send: &mut dyn FnMut(Message)) -> Result<Solution, String> {
    request.validate()?;
    let mut messages = Messages { send, clock: Instant::now(), last: None, least: 0, stage: None };
    let reference = alone(chief(request));
    // The choice kept for the scene (# Steps: Select at once, then the model).
    if let Some(choice) = crate::select::kept(request) {
        messages.stage(Stage { step: Step::Select, tau: None, batch: None });
        messages.stage(Stage { step: Step::Model, tau: None, batch: None });
        let scene = crate::scene::Scene::new(request)?;
        let selection = crate::select::choose(&scene)?;
        let run = Selected::new(scene, Selection { keep: choice.keep.clone(), choice: Some(choice.clone()), ..selection }, reference)?;
        return solved(solve_with(request, &run, Choosing::Made(Some(choice)), &mut messages));
    }
    // A new selection (# Steps: the structural chain and G's model are the model's making).
    messages.stage(Stage { step: Step::Model, tau: None, batch: None });
    let (attempted, selection) = piloted(request, reference, Choosing::Make, &mut messages)?;
    match attempted {
        Attempted::Solved(s) => Ok(s),
        Attempted::Scouted(_) => unreachable!("an attempt that makes the choice is no scout"),
        Attempted::RanAway => unreachable!("`piloted` goes up the chain past a model that ran away"),
        Attempted::Chose(choice) => {
            messages.stage(Stage { step: Step::Model, tau: None, batch: None });
            let run = Selected::new(crate::scene::Scene::new(request)?, Selection { keep: choice.keep.clone(), choice: Some(choice.clone()), ..selection }, reference)?;
            solved(solve_with(request, &run, Choosing::Made(Some(choice)), &mut messages))
        }
    }
}

/// An attempt that was to solve (`Choosing::Made`): its solution.
fn solved(attempted: Result<Attempted, String>) -> Result<Solution, String> {
    match attempted? {
        Attempted::Solved(s) => Ok(s),
        Attempted::Chose(_) | Attempted::Scouted(_) | Attempted::RanAway => unreachable!("an attempt on a choice made solves"),
    }
}

/// The attempt that makes the choice (`Choosing::Make`; the tests' `Choose`) on the cheapest model of the structural
/// chain whose pilot goes past its start (# The selection; PIPELINE_V3_FINAL §二 B.7: the scout and the pilot on the
/// cheapest candidate): G, and where every one of G's paths stopped at its start, its fold scattering beyond every state
/// of the atom there (`Attempted::RanAway`; step.rs, # The domain: no length of a step makes the fold hold), the chain's
/// next model, and so on up -- the chain's top folds nothing, and no state of it scatters beyond the atom's. Each model
/// made again is the pipeline's Model step again. With it the selection it was taken from (the structural chain).
fn piloted(request: &Request, reference: [f64; GROUND], choosing: Choosing, messages: &mut Messages) -> Result<(Attempted, Selection), String> {
    let selection = crate::select::choose(&crate::scene::Scene::new(request)?)?;
    let top = selection.chain.len() - 1;
    for place in 0..=top {
        if place > 0 {
            messages.stage(Stage { step: Step::Model, tau: None, batch: None });
        }
        let run = Selected::new(crate::scene::Scene::new(request)?, Selection { keep: selection.chain[place].clone(), ..selection.clone() }, reference)?;
        match solve_with(request, &run, choosing.clone(), messages)? {
            Attempted::RanAway if place < top => eprintln!(
                "  注：P{} 的折叠在先导每条路径的起点就超过原子任何状态的散射率（失控，任何步长都一样），没有一条路径走过起点：先导改在链上的下一档 P{} 上重做",
                selection.chain[place].len(),
                selection.chain[place + 1].len()
            ),
            Attempted::RanAway => return Err("the chain's top ran away at its pilot's every start (it folds nothing: no state of it can)".into()),
            attempted => return Ok((attempted, selection)),
        }
    }
    unreachable!("the chain's top ends the climb")
}

/// What an attempt does with the choice of P (module documentation, # The selection): takes the one made, or none
/// (`Made`: its model's error booked from it, or not measured); makes it on its pilot and goes on to the product where
/// it chose the attempt's own model (`Make`), or stops there whatever it chose (`Choose`: the tests' `chosen`); or a
/// candidate's own scout alone, its rung read and nothing chosen (`Scout`: the choice's pricing of a candidate at its
/// own step, `choose_on_the_pilot`).
#[derive(Clone)]
enum Choosing {
    Made(Option<std::sync::Arc<Choice>>),
    Make,
    Scout,
    #[cfg(test)]
    Choose,
}

/// How an attempt ended: its solution, the choice of another model (or, `Choosing::Choose`, of any), (`Scout`) the
/// rung its scout stands at (µs), or -- where it was to make the choice -- its model's fold beyond every state of the
/// atom at every one of its pilot's paths' starts, so that no path went past its start and nothing is chosen on them
/// (`choose_on_the_pilot`; `piloted` takes the chain's next model).
enum Attempted {
    Solved(Solution),
    Chose(std::sync::Arc<Choice>),
    Scouted(f64),
    RanAway,
}

/// The run of a request at the reference populations `reference`: its scene and the selection's cheapest candidate,
/// the full-ground model G, with its fold -- the run the scout and the pilot are made on where no choice is kept (# The
/// selection) --: the tests' run of a model, whose error is not measured.
#[cfg(test)]
pub(crate) fn run_of(request: &Request, reference: [f64; GROUND]) -> Result<Selected, String> {
    let scene = crate::scene::Scene::new(request)?;
    let selection = crate::select::choose(&scene)?;
    Selected::new(scene, selection, reference)
}

/// The run of a request on the P its selection chooses (# The selection), at the reference populations `reference`:
/// the choice kept for the scene, or made on the pilot (`piloted`: G's, or the chain's first model that does not run
/// away; its attempt stopped there) and kept. The tests' chosen run.
#[cfg(test)]
pub(crate) fn chosen(request: &Request, reference: [f64; GROUND]) -> Result<Selected, String> {
    let scene = crate::scene::Scene::new(request)?;
    let selection = crate::select::choose(&scene)?;
    let choice = match crate::select::kept(request) {
        Some(choice) => choice,
        None => {
            let mut send = |_: Message| {};
            let mut messages = Messages { send: &mut send, clock: Instant::now(), last: None, least: 0, stage: None };
            match piloted(request, alone(chief(request)), Choosing::Choose, &mut messages)?.0 {
                Attempted::Chose(choice) => choice,
                Attempted::Solved(_) | Attempted::Scouted(_) | Attempted::RanAway => unreachable!("an attempt that chooses stops at its choice"),
            }
        }
    };
    Selected::new(scene, Selection { keep: choice.keep.clone(), choice: Some(choice), ..selection }, reference)
}

/// The reference state (SPEC §2.1): the initial component of the largest population, of equal ones the last.
pub(crate) fn chief(request: &Request) -> usize {
    let initial = components(request);
    let chief = (0..initial.len()).max_by(|&a, &b| initial[a].1.total_cmp(&initial[b].1)).expect("a validated request's initial state holds a ground state");
    initial[chief].0
}

/// The solve of a request on a run of its own (the tests'): its error booked from the choice kept for the scene where
/// that choice is this run's P, and else not measured (# The model).
#[cfg(test)]
pub(crate) fn solve_on(request: &Request, run: &Selected, send: &mut dyn FnMut(Message)) -> Result<Solution, String> {
    let choice = crate::select::kept(request).filter(|c| c.keep == run.rs.keep);
    let mut messages = Messages { send, clock: Instant::now(), last: None, least: 0, stage: None };
    solved(solve_with(request, run, Choosing::Made(choice), &mut messages))
}

/// An attempt of a request on a run (`solve`): the run's report to stderr, its setting, the pipeline (`attempt`).
fn solve_with(request: &Request, run: &Selected, choosing: Choosing, messages: &mut Messages) -> Result<Attempted, String> {
    let hm = hbar_over_mass(request.mass_kg);
    let initial = components(request);
    let chief = initial.iter().position(|&(s, _)| s == chief(request)).expect("the reference state is a component");
    let reference = alone(initial[chief].0);
    eprint!("{}", run.report());
    let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(run, alone(s))).collect::<Result<_, _>>()?;
    let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm)?;
    let trap = &wells[chief].trap;
    let reported = reported(run, &reference).ok();
    let setting = Setting { run, trap, ensemble: &ensemble, depth: ensemble.depths()[chief], reported, kick: kick_curvature(run, trap) };
    let attempted = attempt(request, &setting, messages, choosing);
    // The run's counts (adapt.rs `tally`), to stderr.
    eprintln!("{}", crate::adapt::counted(true));
    attempted
}

/// What the solve takes of its run beside the steps (`solve`), made once.
struct Setting<'a> {
    run: &'a Selected,
    /// The reference state's prepared trap.
    trap: &'a Trap,
    /// The initial ensemble, every component in its own well, and the depth at the reference state's bottom (k_BT).
    ensemble: &'a Ensemble<'a>,
    depth: f64,
    /// The trap every reported quantity describes (SPEC §1.3: trapFrequencyKHz, n̄, the validated regime and T's
    /// windows): the reference state's potential (model/selected.rs `Selected::prepared`, `Scene::trap_potential`: one
    /// potential, RUSTPLAN §2.3), its trap at every light's pattern's scale (`reported`). None where it has no minimum
    /// the descent reaches -- the atoms are not
    /// held by the light they run in: a mathematical object that does not exist, not a kind of scene --: the prepared
    /// trap's is reported then, and flagged.
    reported: Option<Trap>,
    /// The kick's largest curvature at the trap (`kick_curvature`).
    kick: f64,
}

/// The solve's messages to the app (module documentation, # Checkpoints).
struct Messages<'a> {
    send: &'a mut dyn FnMut(Message),
    /// The last result's time: the clock times results alone, never a stage.
    clock: Instant,
    /// The last result sent.
    last: Option<Solution>,
    /// The trajectories a result must rest on to be sent.
    least: u64,
    /// The last stage sent (# Steps).
    stage: Option<Stage>,
}

/// A message of the solve to the app: a checkpoint -- a complete result so far, with the wall time left (ms) where the
/// run's pace gives one (# Checkpoints) -- or the step the pipeline stands at (# Steps).
pub enum Message<'a> {
    Checkpoint(&'a Solution, Option<f64>),
    Stage(&'a Stage),
}

/// The steps of the pipeline a stage names (module documentation, # The pipeline): the selection of P -- its structural
/// chain before the scout, the choice on the pilot's paths after it (# The selection) --, the model (its fold, the
/// prepared wells and the initial ensemble), the scout, the pilot and the product.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Select,
    Model,
    Scout,
    Pilot,
    Product,
}

/// Where the pipeline stands (# Steps): its step, the rung τ (µs) the scout, the pilot or the product runs at, and the
/// product's batch.
#[derive(Clone, Debug, PartialEq)]
pub struct Stage {
    pub step: Step,
    pub tau: Option<f64>,
    pub batch: Option<Batch>,
}

/// A batch of the product (`product`): its number from 1, the trajectories in hand and the number it brings the run to
/// (`batch`), and its looks' largest total over its target -- every look within its target at 1 or below, which stops
/// the run (`Look::met`) --, none where that is no number.
#[derive(Clone, Debug, PartialEq)]
pub struct Batch {
    pub number: u64,
    pub trajectories: u64,
    pub want: u64,
    pub worst: Option<f64>,
}

impl Messages<'_> {
    /// The pipeline at `stage` (# Steps): sent where it is not the last stage sent -- a step begun, or its numbers
    /// changed --, never again otherwise.
    fn stage(&mut self, stage: Stage) {
        if self.stage.as_ref() != Some(&stage) {
            (self.send)(Message::Stage(&stage));
            self.stage = Some(stage);
        }
    }
    /// The step's check moves to the rung `tau` (`check`): the step in hand, at that rung -- where it is one that has a
    /// rung, the scout, the pilot or the product (# Steps): a candidate's short scout runs under Select, which has none.
    fn rung(&mut self, tau: f64) {
        if let Some(stage) = self.stage.as_ref().filter(|s| matches!(s.step, Step::Scout | Step::Pilot | Step::Product)) {
            let moved = Stage { tau: Some(tau), ..stage.clone() };
            self.stage(moved);
        }
    }
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
            (self.send)(Message::Checkpoint(&s, remaining));
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
            (self.send)(Message::Checkpoint(s, None));
            self.clock = Instant::now();
        }
    }
    /// The time until the next message is due.
    fn due(&self) -> Duration {
        CHECKPOINT.saturating_sub(self.clock.elapsed())
    }
    /// Between the selection's stages that run no trajectory (# The selection): where a message is due, the last
    /// result again, or with none yet the stage the pipeline stands at again -- the app hears from the worker within its
    /// wait (`CHECKPOINT`).
    fn alive(&mut self) {
        if !self.due().is_zero() {
            return;
        }
        if self.last.is_some() {
            self.again();
        } else if let Some(stage) = &self.stage {
            (self.send)(Message::Stage(stage));
            self.clock = Instant::now();
        }
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

/// One initial component's preparation well (ensemble.rs `Well`): its prepared potential -- the tweezer's on the run's
/// ground populations (model/selected.rs `Selected::prepared`), zero far from the trap --, its trap (`trap_of`), the
/// trap light's beams and the envelope's peak (`Potential::peak`).
struct Prepared<'a> {
    light: &'a Light,
    potential: Potential<'a>,
    trap: Trap,
}
impl<'a> Prepared<'a> {
    fn new(run: &'a Selected, populations: [f64; GROUND]) -> Result<Prepared<'a>, String> {
        let (light, potential) = (run.light(), run.prepared(&populations));
        let trap = trap_of(run, &potential, &|w| w == light.trap).map_err(|why| format!("准备阱不存在：{why}（准备阱是光镊在这些基态布居上的势：它的 RS 折叠光移与外层形式）"))?;
        Ok(Prepared { light, potential, trap })
    }
}
impl Well for Prepared<'_> {
    fn potential(&self, r: V3) -> (f64, V3) {
        let q = self.potential.at(r);
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
fn trap_of(run: &Selected, potential: &Potential, lit: &dyn Fn(usize) -> bool) -> Result<Trap, String> {
    let light = run.light();
    let k = light.frequencies.iter().enumerate().filter(|(w, _)| lit(*w)).flat_map(|(_, f)| f.beams.iter().map(|&l| light.beams[l].wave_number)).fold(0., crate::linalg::larger);
    descend(1. / k, run.hbar_over_mass(), &|r| light.outside(r), &|r| potential.rounded(r))
}

/// The reported trap of the ground populations p (SPEC §1.3; `Setting::reported`): the trap of their potential
/// (`Selected::prepared`), every light's beams' pattern its scale (`trap_of`).
pub(crate) fn reported(run: &Selected, p: &[f64; GROUND]) -> Result<Trap, String> {
    trap_of(run, &run.running(p), &|_| true)
}

/// The kick's largest curvature (`stable`), ω_k² = (ħ/m) λ_max ∂∂(V_s − V_ref) at the trap's centre over the ground
/// states s (rad/µs²): how much stiffer than the reference's a state's own prepared well is there (`Selected::prepared`),
/// exactly. What else the kick's potential curves by -- the folded light's shifts along the pattern, an explicit
/// excited state's well -- is the step's check's to find, as every part of the force is (step.rs, # The kick).
/// Negative where every state's well is the softer one.
fn kick_curvature(run: &Selected, trap: &Trap) -> f64 {
    let reference = run.reference(trap.centre).hessian;
    (0..GROUND)
        .map(|s| {
            let own = run.prepared(&alone(s)).at(trap.centre).hessian;
            let more: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| own[a][b] - reference[a][b]));
            symmetric3_eigen(&more).0.into_iter().fold(f64::NEG_INFINITY, crate::linalg::larger)
        })
        .fold(f64::NEG_INFINITY, crate::linalg::larger)
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
    light.frequencies.iter().enumerate().filter(|(w, _)| *w != light.trap).flat_map(|(_, f)| f.beams.iter().map(|&l| light.beams[l].wave_number)).fold(0., crate::linalg::larger)
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
    let least = [(run / (2. * prior)).ceil(), (run * frequency / PI).floor()].into_iter().fold(1., crate::linalg::larger);
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

/// What a level's steps ask of the run's domain (step.rs, # The domain): the runaway's rule (`Widest`).
#[derive(Clone, Copy)]
struct Domain {
    widest: Widest,
}

/// The fastest decay of the request's excited states (1/µs): no state of the atom scatters faster than that, wholly
/// excited -- a state's hazard above it is a fold far outside its validity (step.rs, # The domain): the runaway's rule,
/// of the run's steps (`Domain`) and of those a price takes (`work_at`).
#[derive(Clone, Copy)]
struct Widest(f64);
impl Widest {
    fn of(request: &Request) -> Widest {
        let decay = crate::scene::decay(&request.emission);
        Widest((0..decay.rows).map(|x| decay[(x, x)].re).fold(0., crate::linalg::larger))
    }
}
impl crate::step::Domain for Widest {
    fn runaway(&self, hazard: f64) -> bool {
        hazard > self.0
    }
}
impl crate::step::Domain for Domain {
    /// A hazard above the fastest excited state's decay, wholly excited (`Widest`).
    fn runaway(&self, hazard: f64) -> bool {
        self.widest.runaway(hazard)
    }
}

/// The held atoms' kinetic energy's bound K_max (rad/µs; SPEC §1.7; DESIGN2 §2.3), made once with the model: a held
/// atom's loss energy is at most the escape value, zero (`Dynamics::lost`), so its kinetic energy is at most
/// −V_prep(r; p), at most the prepared potential's deepest over every place and every ground population p -- V is the
/// ground states' wells' convex sum, each the tweezer's intensity, largest at its focus, times its coefficient
/// (`Scene::trap_potential`) --: the deepest ground state's well at the focus, the trap's depth U₀ (1043 µK on GMD1C,
/// PGC3D and TET, 1073 on R4, 1500 on tilt). None where no ground state is held there.
fn kinetic_bound(run: &Selected) -> f64 {
    (0..GROUND).map(|g| -run.prepared(&alone(g)).at([0.; 3]).value).fold(0., crate::linalg::larger)
}

/// The growth rate Y (1/µs) of the trace distance by which the model's dropped terms may move the state, D(t) ≤
/// ½(e^{2Yt} − 1) (DESIGN2 §2.2; DROPRES §2; model/classify.rs `distance`): the model's own bound
/// (`Selected::dropped`), zero where it drops nothing.
fn dropped_rate(run: &Selected) -> f64 {
    run.dropped()
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
    scaled(read, ratio.powi(2 * ORDER))
}

/// A read of mean squares times a factor, each look's value and spread (`carried`, `read_step`).
fn scaled(read: &Between, factor: f64) -> Between {
    let carry = |r: &Read| Read { value: factor * r.value, spread: Spread { error: factor * r.spread.error, ..r.spread } };
    Between { loss: carry(&read.loss), shown: read.shown.each_ref().map(carry), temperature: read.temperature.as_ref().map(carry) }
}

/// The factor a read of two levels takes for its coarser level's halved steps, of that level's groups `coarser` (module
/// documentation, ## A companion at τ/2): with r = 1 − shortfall/time run (adapt.rs `Halved`, `Counts::time`) the
/// coarser level errs by 2^p r times the finer one's where the finer took its steps whole, so their difference is
/// d = (2^p r − 1) e and the read's d/(2^p − 1) is low by (2^p r − 1)/(2^p − 1) -- by 1 − f where its steps were halved
/// once over a share f of its time. The factor is its inverse squared, the read's being of mean squares:
/// ((2^p − 1)/(2^p r − 1))², one where no step was halved, without bound where 2^p r ≤ 1 (the coarser level no coarser
/// than the finer: the difference tells nothing).
fn amplified<'a>(coarser: impl IntoIterator<Item = &'a Part>) -> f64 {
    let (shortfall, time) = coarser.into_iter().fold((0., 0.), |(s, t), p| (s + p.counts.halved.shortfall, t + p.counts.time));
    factor_of(shortfall, time)
}

/// `amplified`'s factor of a level whose halved steps' shortfall in the order is `shortfall` over the time `time` (µs):
/// ((2^p − 1)/(2^p r − 1))², r = 1 − shortfall/time; one over no time.
fn factor_of(shortfall: f64, time: f64) -> f64 {
    let (order, r) = (2_f64.powi(ORDER), if time > 0. { 1. - shortfall / time } else { 1. });
    let short = order * r - 1.;
    if short > 0. { ((order - 1.) / short).powi(2) } else { f64::INFINITY }
}

/// The pairs a read of no lift and no halved steps needs to read within the step's share at the best -- its noise
/// alone: its far end t·SE at one look within the share's square b, the noise falling as 1/N (`asks`) --, as the rung's
/// own read `read` on its `pairs` foretells it, its own lift and factor `carried` undone (module documentation, ## A
/// companion at τ/2): the largest over the looks that read. A pair's read whose noise is the same pairs' -- the pairs
/// parted, the difference of two ensembles' estimates at either step -- carried by the lift λ = (τ/h)^{2p} from its finer
/// step h to the rung's and taken back by its factor a needs λ a times as many (`read_cost`).
fn raw_pairs(read: &Between, pairs: f64, carried: f64) -> f64 {
    judged(read).into_iter().map(|(read, bound)| read.spread.interval() / bound).filter(|x| x.is_finite()).fold(0., crate::linalg::larger) * pairs / carried
}

/// The work a read of a pair costs for its precision (module documentation, ## A companion at τ/2; `against_half`,
/// `third_of`): the pairs it needs within the share at the best (`raw_pairs` × its lift × its factor), or the pilot's
/// `pilot` where those need fewer, at its level of the step `step` beyond the `held` there and at its other level
/// `other` beyond the `beside` there (the pilot's, or the run level's trajectories where that is the other level: what
/// the run holds there anyway) -- each at its work a µs as priced, its refused attempts in it (`Prices::work`); without
/// bound where its factor is.
#[allow(clippy::too_many_arguments)]
fn read_cost(prices: &Prices, step: f64, held: f64, other: f64, beside: f64, pilot: f64, raw: f64, lift: f64, factor: f64) -> f64 {
    if !(factor < f64::INFINITY) {
        return f64::INFINITY;
    }
    let pairs = crate::linalg::larger(pilot, raw * lift * factor);
    crate::linalg::larger(pairs - held, 0.) * (prices.work)(step) + crate::linalg::larger(pairs - beside, 0.) * (prices.work)(other)
}

/// A rung's companion's work for its read's precision (`read_cost`): against 2τ of no lift, against τ/2 lifted 2^{2p};
/// its factor `factor`, its level held by `held`, the run level beside it held to `run`.
#[allow(clippy::too_many_arguments)]
fn companion_cost(prices: &Prices, tau: f64, companion: Companion, held: f64, run: f64, pilot: f64, raw: f64, factor: f64) -> f64 {
    let lift = if companion == Companion::Half { 2_f64.powi(2 * ORDER) } else { 1. };
    read_cost(prices, companion.step(tau), held, tau, run, pilot, raw, lift, factor)
}

/// Whether a rung at the step `step` is read against half its step rather than twice it (module documentation, ## A
/// companion at τ/2): where the read against τ/2 costs the less for its precision (`read_cost`) -- the read against 2h
/// of no lift and the factor `double` of 2h's halved steps, its level held by `held[0]`; the read against h/2 lifted
/// 2^{2p} and of the factor `half` of h's, its level held by `held[1]`; each its pairs for the share at the best, the
/// rung's own noise `raw` foretelling theirs (`raw_pairs`), on the `pilot` pairs at least --, and wherever the factor
/// against 2h is without bound. The one rule the companion and the third level are chosen by (`third_of`).
#[allow(clippy::too_many_arguments)]
fn against_half(prices: &Prices, step: f64, double: f64, half: f64, raw: f64, pilot: f64, held: [f64; 2]) -> bool {
    !(double < f64::INFINITY) || companion_cost(prices, step, Companion::Half, held[1], pilot, pilot, raw, half) < companion_cost(prices, step, Companion::Double, held[0], pilot, pilot, raw, double)
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
    /// Its own read's estimate, or its third level's, beyond the share, but not clearly: the noise hides the difference
    /// -- the pairs that could see through it where they are expected to cost the run less than the rung below
    /// (`remedy`), else the rung is left. One rule for both reads: each is the rung's mean square where the order holds.
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
///
/// The two reads are judged by one rule: each is the rung's mean square where the order holds, so what is clear of
/// one is clear of the other. Clearly beyond, either: left. Its estimate beyond and not clearly, either: the noise
/// hides it (`Verdict::Unseeing`), and the pairs that could see through it are weighed against the rung below
/// (`remedy`) -- a third level's estimate beyond the share is no more a verdict than the rung's own is. (Left on its
/// estimate alone, the third level decided on the noise: R4_LIN_4's third level at 0.237 µs, the scout's eleven
/// pairs at 0.474 and 0.948 µs lowered, read 3.36 ± 3.59 and 8.78 ± 5.44 (10⁻⁴, the share's square 5·10⁻⁴) on a row
/// in two runs that differ in nothing the step resolves -- 0.8σ apart, one rung apart: 0.237 against 0.118 µs, 90
/// against 196 CPU-s.)
fn verdict(read: &Between, above: Option<&Between>, pilot: bool, upward: bool) -> Verdict {
    if read_looks(read).is_empty() {
        Verdict::Unread
    } else if screen(read) || above.is_some_and(screen) {
        Verdict::Left
    } else if !pilot {
        Verdict::Stands
    } else if !within(read) || above.is_some_and(|above| !within(above)) {
        Verdict::Unseeing
    } else if above.is_some() {
        Verdict::Stands
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
    read_looks(own).into_iter().filter(|(read, bound)| read.outcome(*bound, 1.) != Outcome::Within).map(|(read, bound)| asks(ends(&read), bound)).reduce(crate::linalg::larger)
}

/// A look's far and near ends at one look, [F, L] (decide.rs `Read::most`, `Read::least`), the near end from zero:
/// what the read does not exclude, and the least error it allows.
fn ends(read: &Read) -> [f64; 2] {
    [read.most(1.), crate::linalg::larger(read.least(1.), 0.)]
}

/// The pairs a look of the ends [F, L] could read within the bound b on at the best (`blind`), as a factor on those it
/// was read on: the far end on N pairs is L + (F − L)·N₀/N, within b from N = N₀ (F − L)/(b − L) on; on no number where
/// L itself is at b or above, nor where F is no number -- a look that is no number, or whose spread is unread,
/// excludes nothing.
fn asks([far, near]: [f64; 2], bound: f64) -> f64 {
    if far.is_finite() && near < bound { (far - near) / (bound - near) } else { f64::INFINITY }
}

/// Whether a rung's third level, carried to its step, stands in the order's range of its own read (module
/// documentation, ## The ladder; `Scouted`): under the order the two are of one mean square, the third's the own one's
/// times (d₂/d₁)²/2^{2p}; on no look the near end of one above the far end of the other (`ends`, each read with its own
/// statistical error, at one look). A look either does not read excludes nothing.
fn in_range(own: &Between, third: &Between) -> bool {
    let apart = |a: &Read, b: &Read| {
        let ([far_a, near_a], [far_b, near_b]) = (ends(a), ends(b));
        near_b > far_a || near_a > far_b
    };
    !(apart(&own.loss, &third.loss) || own.shown.iter().zip(&third.shown).any(|(a, b)| apart(a, b)) || own.temperature.zip(third.temperature).is_some_and(|(a, b)| apart(&a, &b)))
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
    looks.into_iter().flatten().reduce(crate::linalg::larger)
}

/// What the step's check weighs its courses by (`course`, `remedy`; module documentation, ## The ladder): `work`, a
/// trajectory's work a µs on the run with every step taken at the step given, each priced at its own step (`work_at`:
/// at the states in hand, the scout's or the pilot's, `one_each`); and `product`, the trajectories the product is to
/// end on as the statistics in hand predict them (`most_needed`: the looks' need, # Stopping) -- the pilot's, or before
/// them the scout's; and `halving`, the factor a read takes for a level at the step given as its pair's coarser level
/// (`amplified`), as the same states foretell it before the level runs (`work_at`: the lengths their steps were taken
/// at) -- what a level not yet run is weighed by where a read's precision is priced (`against_half`, `Third`).
struct Prices<'a> {
    work: &'a dyn Fn(f64) -> f64,
    halving: &'a dyn Fn(f64) -> f64,
    product: f64,
}

/// A trajectory's work a µs at every step at the states `rated` (`one_each`), the mean over the run's held states
/// (`over_the_run`): at each, the step of τ taken from it as the run takes it, as its count is made (step.rs `priced`:
/// its stretches' counts, each with its middle's counts at a node, `step::Middle::work`) over its length, and its jumps'
/// counted work a µs over its run (`Pick::jumping`: their photons and landings as they were made, step.rs
/// `Trajectory::jumped`) -- a jump's work is the jump's whatever the step, so the step's count holds none of it: a
/// landing's clock counted in the one step a state is priced at made the price a draw of whether that step held a jump
/// (tilt0's scout: w(τ/2) below w(τ) by 5 %, and the ladder went down a rung for it). A step the scheme refuses at τ is
/// taken as the advance takes it, at its half and so on (adapt.rs `halving`), and priced at the length it is taken at:
/// its count over that length. That is what a refused step costs the run: refused at t, the run takes τ/2 there and goes
/// on at τ from t + τ/2 -- one step more over the run --, and a state whose step is taken at τ/2 prices the run's work a
/// µs there at twice the steps. And each attempt refused on the way is the work it did before its refusal (`refused_at`,
/// step.rs `Scratch::refused`): a refusal for an entry of no route comes at the decomposition (step.rs `pictured`),
/// after the step's middle and its generator at the three nodes are made -- most of a step where the model at a point
/// is (36 to 79 % of a step: motion3d_status/rewrite/prof1.status.md) --, one for a growing norm after the propagation's
/// stops. (Priced without it, GMD1C's pilot at 3.79 µs, most of its steps refused, took 48 s against the 23 s its price
/// foretold, and R4's 0.948 µs, 44 % of its time on halved steps, priced 10 % below 0.474 µs: motion3d_status/rewrite/
/// r2ladder.status.md.) Past any work
/// where no length of a step can be taken from the state or where its step ends the trajectory (a runaway, a jump with
/// no photon, an optical sector with no K: refused as the run's would be, the runaway's rule the run's, `Widest`). The
/// splitting's reference is a sum over the trap light's beam pairs, no part of the price. The step's
/// check's `Prices::work`, each step's made once -- a check prices the same steps again as it goes on --, the states'
/// steps on the pool, each by itself. Beside it the factor a read takes for the level at τ as its coarser level, as the
/// states foretell its halved steps (`Prices::halving`, `factor_of`): each state's step one of the level's, of the
/// length it was taken at, weighed by the time its trajectory holds its atom -- a state whose step is halved stands for
/// the stretch the run takes at the half.
fn work_at<'a>(run: &'a Selected, widest: Widest, rated: &'a [Pick<'a>]) -> impl Fn(f64) -> [f64; 2] + 'a {
    let made = std::cell::RefCell::new(Vec::<(f64, [f64; 2])>::new());
    move |tau: f64| {
        if let Some(&(_, priced)) = made.borrow().iter().find(|(at, _)| *at == tau) {
            return priced;
        }
        // Per state its work a µs and the length its step was taken at (none where none was).
        let taken: Vec<(f64, Option<f64>)> = rated
            .par_iter()
            .map(|pick| {
                let from = pick.state.trajectory();
                // The work of the attempts refused on the way, each by what it did (step.rs `Scratch::refused`).
                let mut refused = 0.;
                match adapt::halving(from.time, tau, |length| crate::step::priced(run, &from, length, &widest).inspect_err(|_| refused += refused_at(run, &from, length, &widest))) {
                    Ok((stretches, length)) => ((stretches.iter().map(|(_, price)| price.work(price.costs)).sum::<f64>() + refused) / length + pick.jumping, Some(length)),
                    Err(_) => (f64::INFINITY, None),
                }
            })
            .collect();
        let work = over_the_run(rated, taken.iter().map(|t| t.0), None);
        // The level's halved steps as its states foretell them: each state one of its steps, weighed by its time held.
        let (shortfall, time) = rated.iter().zip(&taken).filter_map(|(pick, t)| t.1.map(|l| (pick.weight, l))).fold((0., 0.), |(s, t), (w, l)| (s + w * l * (1. - (l / tau).powi(ORDER)), t + w * l));
        let priced = [work, factor_of(shortfall, time)];
        made.borrow_mut().push((tau, priced));
        priced
    }
}

/// The work a refused attempt of the step of `length` from the trajectory `from` did before its refusal (step.rs
/// `Scratch::refused`: read about the attempt, on a scratch of its own as `step::priced` takes it), for `work_at`;
/// none where the step's middle is refused before any attempt (`Midway::new`).
fn refused_at(run: &Selected, from: &Trajectory, length: f64, widest: &Widest) -> f64 {
    crate::step::Midway::new(run, from, length).map_or(0., |through| {
        let mut scratch = Scratch::default();
        let before = scratch.refused();
        let _ = through.attempt(from, &crate::step::Brownian::STILL, &[], &mut scratch, widest);
        scratch.refused() - before
    })
}

/// What the run holds of a rung's levels where its check takes its course (`course`, `remedy`; module documentation,
/// ## The ladder), in trajectories: the pilot's at each of the rung's two levels once its groups are whole, `pilot`;
/// those of them run at both, `members` (the scout's, one a group, or all the pilot's); the pairs at the rung, `pairs`
/// (those and the further pairs); the rung's companion (`Companion`: at 2τ, or at τ/2); the trajectories run at τ/2
/// and at τ/4 beside the members at τ, `below` (the scout's pairs at τ/2 or τ/4, `foresee`; the members at τ/2 where
/// that is the companion); and the factor a read takes for the levels at τ and at τ/2 as its coarser level, of their
/// halved steps on what they hold (`amplified`; one at a level not run) -- what the rungs below are read against
/// (`against_half`; ## A companion at τ/2); and the pairs a read of no lift and no halved steps needs within the share,
/// as the rung's own read's noise foretells it (`raw_pairs`): what every read's precision is priced by.
#[derive(Clone, Copy, Debug)]
struct Holding {
    pilot: f64,
    members: f64,
    pairs: f64,
    companion: Companion,
    below: [f64; 2],
    amplified: [f64; 2],
    raw: f64,
    factor: f64,
}

/// The run's expected work from here where it stays at the rung τ in hand (module documentation, ## The ladder): the
/// pilot's companions still to run at the companion's step c (2τ, or τ/2), and its run level at τ to the most of the
/// product's trajectories, the pilot's and the pairs in hand (the further pairs' run level is the product's), (P − P_m)
/// w(c) + (max(N, P, n) − n) w(τ). The third level above is the course's to add where the rung buys it (`course`).
fn staying(prices: &Prices, tau: f64, holding: Holding) -> f64 {
    let Holding { pilot, members, pairs, companion, raw, factor, .. } = holding;
    let run = crate::linalg::larger(crate::linalg::larger(prices.product, pilot), pairs);
    companion_cost(prices, tau, companion, members, run, pilot, raw, factor) + (run - pairs) * (prices.work)(tau)
}

/// The run's expected work from here at the least of the rungs below the rung τ in hand (module documentation,
/// ## The ladder); without bound where there is none -- the ladder's bottom is the step at which a trajectory of the
/// run's length `end` takes `MOST_STEPS` steps (`check`). The ladder goes down a rung at a time, each rung's own level
/// its next one's companion -- or the next one's own level its companion, where that reads it at less work for its
/// precision (`against_half`; ## A companion at τ/2): τ's halved steps as its members took them, τ/2's where it is
/// run, a level not run as one whose steps are whole (a halving is known once a level has run) --, so the
/// k-th rung below, τ_k = τ/2^k, costs the members in hand at every rung on the way, its run level at τ_k to the more
/// of the product's trajectories and the pilot's, and the pilot's companions at τ_{k−1}, or at τ_{k+1}: each level
/// beyond the trajectories it holds already (the members at τ, `Holding::below` at τ/2 and τ/4), Σ_{j<k} (P_m −
/// r_j)⁺ w(τ_j) + (max(N, P) − r_k)⁺ w(τ_k) + (P − held)⁺ w(c_k) -- the pairs at the rung are no level of a rung
/// below. Every term is no less than zero, and every rung at or below τ_k holds the members at τ_1 … τ_k, so none costs
/// less than Σ_{j≤k} (P_m − r_j)⁺ w(τ_j): the rungs are priced down to where that is no less than the least found, or
/// to the bottom. Each rung's companion the rung above, it is the ladder as it was, P_m Σ_{j≤k} w(τ_j) + (P − P_m)
/// w(τ_{k−1}) + (max(N, P) − P_m) w(τ_k) − r_1 w(τ_1).
fn descending(prices: &Prices, tau: f64, end: f64, holding: Holding) -> f64 {
    let Holding { pilot, members, below, amplified, raw, .. } = holding;
    let run = crate::linalg::larger(prices.product, pilot);
    let at = |k: i32| tau * 0.5_f64.powi(k);
    // The trajectories held at τ_k beside the members at τ (k ≥ 1).
    let held = |k: i32| below.get(k as usize - 1).copied().unwrap_or(0.);
    let more = |need: f64, have: f64| crate::linalg::larger(need - have, 0.);
    let (mut least, mut floor, mut k) = (f64::INFINITY, 0., 0);
    while end / at(k + 1) <= MOST_STEPS as f64 {
        k += 1;
        // The members at τ_k: on the way to every rung at or below it.
        let way = more(members, held(k)) * (prices.work)(at(k));
        floor += way;
        if !(floor < least) {
            break;
        }
        // Rung k's companion: τ_{k−1} -- its members held there, on the way or at τ --, or τ_{k+1} where that reads it
        // at less work for its precision, the factors of τ's and τ/2's halved steps as held (`against_half`).
        let factor = |j: i32| if j < 2 { amplified[j as usize] } else { (prices.halving)(at(j)) };
        let above = if k == 1 { members } else { crate::linalg::larger(members, held(k - 1)) };
        let companion = if against_half(prices, at(k), factor(k - 1), factor(k), raw, pilot, [above, held(k + 1)]) {
            companion_cost(prices, at(k), Companion::Half, held(k + 1), run, pilot, raw, factor(k))
        } else {
            companion_cost(prices, at(k), Companion::Double, above, run, pilot, raw, factor(k - 1))
        };
        least = crate::linalg::smaller(least, floor - way + more(run, held(k)) * (prices.work)(at(k)) + companion);
    }
    least
}

/// A third level a rung can read on the pilot's trajectories (module documentation, ## Two levels, three levels): the
/// step of its level beside the rung's own pair -- above it, twice the pair's coarser step (4τ, or 2τ for a rung read
/// against τ/2), or below it, half the pair's finer step (τ/2, or τ/4): three adjacent levels either way --, the pilot's
/// trajectories held there, and the factor its pair's read takes for that pair's coarser level's halved steps
/// (`amplified`: measured where that level is held, foretold by the states in hand where it is not, `Prices::halving`).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Third {
    step: f64,
    held: f64,
    factor: f64,
    beside: f64,
    lift: f64,
    /// The pairs its read needs at the best as read on the scout's own pairs (`blind`, `Scouted`; without bound out of
    /// the order's range, `in_range`), where it was: in place of the noise, the lift and the factor foretold.
    measured: Option<f64>,
}
impl Third {
    /// The pairs its read is run on (module documentation, ## Two levels, three levels; `third_on_the_pilot`): those it
    /// asks at the best -- measured on the scout's pairs (`measured`), or the rung's own noise `raw` lifted and taken back
    /// by its factor --, above the rung in whole groups of the pilot's `pilot`, `THIRD_GROUPS` at the least, and beyond
    /// the pilot's, further pairs; below it the pilot's at the least, its level the rung below's own or its companion's,
    /// every member of it.
    fn pairs(&self, pilot: f64, raw: f64) -> f64 {
        let asks = self.measured.unwrap_or(raw * self.lift * self.factor);
        let groups = crate::linalg::larger((asks / GROUP as f64).ceil(), THIRD_GROUPS as f64) * GROUP as f64;
        if self.step > self.beside && groups <= pilot { groups } else { crate::linalg::larger(asks, pilot) }
    }

    /// Its work on the pilot's `pilot` trajectories: those of its pairs still to run at its step (`pairs`).
    fn work(&self, prices: &Prices, pilot: f64, raw: f64) -> f64 {
        crate::linalg::larger(self.pairs(pilot, raw).min(pilot) - self.held, 0.) * (prices.work)(self.step)
    }

    /// Its work for its read's precision (`read_cost`): its pairs (`pairs`) at its level, and those beyond the pilot's
    /// `pilot` at the rung's level beside it; without bound where its factor is.
    fn cost(&self, prices: &Prices, pilot: f64, raw: f64) -> f64 {
        if self.measured.is_none() && !(self.factor < f64::INFINITY) {
            return f64::INFINITY;
        }
        let pairs = self.pairs(pilot, raw);
        crate::linalg::larger(pairs - self.held, 0.) * (prices.work)(self.step) + crate::linalg::larger(pairs - pilot, 0.) * (prices.work)(self.beside)
    }
}

/// The least of the pilot's groups a third level above the rung is read on (`Third::pairs`): a read's spread is over
/// its groups, and there is none on fewer than three (estimate.rs `between`).
const THIRD_GROUPS: u64 = 3;

/// The pilot's groups the pairs `pairs` fill, of the `groups` it has: the whole groups, all of them at the most.
fn groups_for(pairs: f64, groups: usize) -> usize {
    if pairs.is_finite() { ((pairs / GROUP as f64).ceil() as usize).clamp(1, groups) } else { groups }
}

/// The third level a rung reads where it reads one (module documentation, ## Two levels, three levels): of the sides
/// it can have, `sides` (above, below), the one whose read costs the less for its precision (`Third::cost`) -- three
/// adjacent levels show the order either way, and a side whose coarser level took most of its steps at their halves,
/// out of the order's range, is so much the dearer (PGC3D and TET at 0.948 µs: their third level at 3.79 µs read 6 to
/// 8 times the share's square, unable to see, where the rung's own read was within; motion3d_status/rewrite/
/// r2ladder.status.md) --; the one above where the two cost alike, as the ladder was; none where neither is to be had
/// or both are without bound. `scouted`: the side above as the scout read it on its own pairs (`Scouted`: its pairs
/// measured, `Third::measured`), in place of its foretelling where it is at that side's step.
fn third_of(prices: &Prices, tau: f64, holding: Holding, sides: [bool; 2], scouted: Option<Third>) -> Option<Third> {
    let Holding { pilot, companion, below, amplified, raw, .. } = holding;
    // The lift of a pair whose finer step is h, to τ: (τ/h)^{2p}.
    let lift = |h: f64| (tau / h).powi(2 * ORDER);
    let (up, down) = match companion {
        Companion::Double => (Third { step: 4. * tau, held: 0., factor: (prices.halving)(4. * tau), beside: 2. * tau, lift: lift(2. * tau), measured: None }, Third { step: 0.5 * tau, held: below[0], factor: amplified[0], beside: tau, lift: lift(0.5 * tau), measured: None }),
        Companion::Half => (Third { step: 2. * tau, held: 0., factor: (prices.halving)(2. * tau), beside: tau, lift: 1., measured: None }, Third { step: 0.25 * tau, held: below[1], factor: amplified[1], beside: 0.5 * tau, lift: lift(0.25 * tau), measured: None }),
    };
    let up = scouted.filter(|scouted| scouted.step == up.step).unwrap_or(up);
    [(sides[0], up), (sides[1], down)]
        .into_iter()
        .filter(|(side, third)| *side && third.cost(prices, pilot, raw) < f64::INFINITY)
        .map(|(_, third)| third)
        .reduce(|a, b| if b.cost(prices, pilot, raw) < a.cost(prices, pilot, raw) { b } else { a })
}

/// Whether a rung buys its third level, the pilot's trajectories at the third's step (module documentation, ## The
/// ladder; `third_of`): at its cost for its read's precision (`Third::cost`) no more than what the least rung below
/// adds to staying, at the pilot's whole `holding` -- the third level shows the order about the rung by its own pair,
/// a rung below shows it by this rung's read (`carried`), and the third is worth no more than that. At the ladder's
/// bottom a rung below adds without bound.
fn buys(prices: &Prices, tau: f64, end: f64, holding: Holding, third: Third) -> bool {
    third.cost(prices, holding.pilot, holding.raw) <= descending(prices, tau, end, holding) - staying(prices, tau, holding)
}

/// A rung's course where its reads stand it (module documentation, ## The ladder; `course`, `check`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Course {
    /// It stands on its levels: its own two, or those and its third level above.
    Stand,
    /// Its third level is read first, the pilot's trajectories at its step: bought (`buys`, `third_of`).
    Third,
    /// The rung below: some rung below is expected to cost the run less from here (`descending`).
    Below,
}

/// A rung's course where its reads stand it -- not clearly beyond on the scout, within on the pilot -- by the run's
/// expected work from here (module documentation, ## The ladder), at the run's length `end` and what the run holds of
/// the rung's levels (`holding`); `third`: the third level the rung would read, where it has none yet (`third_of`). The
/// rung below where some rung below costs less than staying (`descending`, `staying`); else its third level where the
/// rung buys it (`buys`, at the pilot's position: on the scout the pilot's purchase to come, its work priced into
/// staying -- the course there is the pilot's to take); else it stands on its own two levels.
fn course(prices: &Prices, tau: f64, end: f64, holding: Holding, third: Option<Third>) -> Course {
    // The two paths from here: verified at τ -- its companions, its third level where it has one to read, each for its
    // read's precision, and its run level to the product --, or at the least rung below (`descending`: its third level
    // this rung's read, lowered).
    let stay = staying(prices, tau, holding) + third.map_or(0., |third| third.cost(prices, holding.pilot, holding.raw));
    if descending(prices, tau, end, holding) < stay {
        Course::Below
    } else if third.is_some() {
        Course::Third
    } else {
        Course::Stand
    }
}

/// The rung below as a rung that cannot see foresees it (`remedy`): the factor on the pairs in hand that would bring
/// it within at the best (`foreseen`; none where it is foreseen within on them), the trajectories of its level
/// already run -- the scout's pairs at τ/2 (`foresee`), or the pilot's where τ/2 is the rung's companion --, and its
/// companion's step with the trajectories run there: τ, the pilot's level, or τ/4 where that costs the less (the
/// scout's pairs there where `foresee` ran them; ## A companion at τ/2).
#[derive(Clone, Copy)]
struct Below {
    asks: Option<f64>,
    ran: u64,
    companion: (f64, u64),
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
/// - at the rung: the further pairs' companions at the companion's step (2τ, or τ/2), its run level at τ beyond the
///   `have` in hand -- the pairs' or the product's, whichever are more: the further pairs' run level is the product's
///   --, and where it has a third level above to read on the pilot (`third`: it has none, and the ladder has the rung
///   above) and buys it on the pairs it then holds (`buys`, at the run's length `end`), the pilot at 4τ;
/// - below: the run level at τ/2 -- the pilot's, the pairs' or the product's, whichever are more -- beyond the
///   trajectories it has run (`Below::ran`), and its companions -- the pilot's and the further pairs' -- beyond those
///   it holds (`Below::companion`: at τ, the pilot's; at τ/4 where that costs the less, `against_half`).
/// The rung below is taken where it is expected to cost less; else the pairs, as where no rung below is. Where no
/// pairs can be had, none: the rung below. What the rung below is foreseen to read decides the work alone, never the
/// step: the rung below stands only on its own read (`check`).
///
/// The rung's third level above, where its estimate is beyond the share and not clearly (`upper`), by the same rule:
/// its pairs N₃ at the best (`seeing`, on the pairs it was read on), at its own levels, 2τ and 4τ. A third level the
/// rung holds at 4τ (`Upper::held`: the pilot's trajectories there, `Levels::coarser`) takes them as further pairs
/// at the rung's every level -- one batch, the own read's and its: the same pairs at τ, 2τ and 4τ --, a further
/// pair the more w(4τ); one it does not hold (the scout's eleven pairs lowered, or the rung left above's) is read
/// on the pilot's trajectories at 4τ first (`Remedy::Upper`, P w(4τ), where the rung above is a rung and its read
/// was not unread: `Upper::buys`), and takes further pairs beyond the pilot's: N₃ − P. The rung's own read
/// within, the rung below is foreseen within on the pilot's pairs whatever the noise's ratio there (its error the
/// order's less, its noise no more than the rung's: `foreseen` asks nothing), and its third level is this rung's
/// own read, lowered: it costs its run level at τ/2 (`check`: `Below { asks: None, ran: 0, .. }`).
///
/// `holding`: what the run holds of the rung's levels (`Holding`): the pilot's `pilot` pairs, the `pairs` in hand, the
/// rung's companion -- the further pairs' companions are at its step c, 2τ or τ/2 (## A companion at τ/2) --, and
/// what a third level bought is priced on (`buys`). With the companion at 2τ and the rung below's at τ, the ladder's
/// rule as it was.
#[allow(clippy::too_many_arguments)]
fn remedy(own: &Between, upper: Option<Upper>, holding: Holding, below: Option<Below>, prices: &Prices, tau: f64, end: f64, third: Option<Third>) -> Option<Remedy> {
    let have = holding.pairs as u64;
    let (n, pilot) = (holding.pairs, holding.pilot);
    // The own read's pairs where its estimate is beyond (`seeing`: none it can see on, the rung below); none asked
    // where it is within.
    let own_at = if within(own) { None } else { Some(seeing(own, have)?) };
    // The third level's pairs: those its read asks, at its levels -- the pilot's at 4τ first where it holds none, as
    // many of its groups as the read asks (`THIRD_GROUPS` at the least).
    let (upper_at, buy, coarser, bought) = match upper {
        Some(upper) if !within(upper.read) => {
            let need = seeing(upper.read, upper.pairs)?;
            if !upper.held && !upper.buys {
                return None;
            }
            let bought = (groups_for(need, (pilot / GROUP as f64) as usize).max(THIRD_GROUPS as usize) as f64 * GROUP as f64).min(pilot);
            let base = if upper.held { upper.pairs as f64 } else { bought };
            (Some(n + crate::linalg::larger(need - base, 0.)), !upper.held, true, bought)
        }
        upper => (None, false, upper.is_some_and(|upper| upper.held), 0.),
    };
    // Nothing that cannot see asks nothing; pairs beyond a run's trajectories are none to be had.
    let at = match (own_at, upper_at) {
        (None, None) => return None,
        (own, upper) => crate::linalg::larger(own.unwrap_or(n), upper.unwrap_or(n)),
    };
    if !buy && !(at <= MOST_TRAJECTORIES as f64 && have < MOST_TRAJECTORIES) {
        return None;
    }
    let next = if buy { Remedy::Upper } else { Remedy::Pairs(batch(at, have)) };
    let Some(below) = below else { return Some(next) };
    let (work, product) = (prices.work, prices.product);
    let half = below.asks.map_or(0., |factor| factor * n);
    let third = third.filter(|third| upper.is_none() && buys(prices, tau, end, Holding { members: pilot, pairs: at, ..holding }, *third));
    let extra = upper.map_or(0., |upper| upper.step);
    let stay = (at - n) * work(holding.companion.step(tau))
        + (crate::linalg::larger(at, product) - n) * work(tau)
        + third.map_or(0., |third| third.work(prices, pilot, holding.raw))
        + if buy { crate::linalg::larger(bought - upper.map_or(0., |upper| upper.ran as f64), 0.) * work(extra) } else { 0. }
        + if coarser { (at - n) * work(extra) } else { 0. };
    // The rung below's run level at τ/2 and its companion's, each beyond what it holds.
    let (step, held) = below.companion;
    let down = (crate::linalg::larger(crate::linalg::larger(product, half), pilot) - below.ran as f64) * work(0.5 * tau) + crate::linalg::larger(crate::linalg::larger(half, pilot) - held as f64, 0.) * work(step);
    (stay <= down).then_some(next)
}

/// A rung's third level as `remedy` weighs it, where it has one: its read carried to the rung, the pairs it was read
/// on, whether the rung holds its extra level -- the pilot's trajectories at its step, `Levels::extra`: further pairs
/// extend it --, and where it does not, whether the pilot's trajectories can read it at the side `third_of` takes (a
/// side to be had, its read on them not unread), with the trajectories already run there (`ran`: the scout's,
/// `Scouted`); and that level's step, held or to be run.
#[derive(Clone, Copy)]
struct Upper<'a> {
    read: &'a Between,
    pairs: u64,
    held: bool,
    buys: bool,
    ran: u64,
    step: f64,
}

/// What `remedy` takes where a rung cannot see: the third level on the pilot's trajectories at its step, or further
/// pairs at the rung's levels, to so many trajectories (`batch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Remedy {
    Upper,
    Pairs(u64),
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
/// on T (a share of its allowance; unbounded where it is not read); the rung's companion (2τ, or τ/2: module
/// documentation, ## A companion at τ/2); and whether the check saw (the rung stands on its
/// estimates) or ended unable to see, capped or not.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Checked {
    third: Option<([f64; 2], u64)>,
    pairs: u64,
    resolution: [f64; 2],
    saw: bool,
    companion: Companion,
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
    let shown = read.shown.iter().map(resolution).fold(resolution(&read.loss), crate::linalg::larger);
    (if saw { estimated(&read) } else { unexcluded(&read) }, [shown, read.temperature.map_or(f64::INFINITY, |t| resolution(&t))])
}

/// What a read estimates: each quantity's √max(M̂, 0), the rms of its step error as read, the largest on what is shown,
/// and the total T's; unbounded where a read is no number.
fn estimated(read: &Between) -> Bound {
    let size = |read: &Read| if read.value.is_nan() { f64::INFINITY } else { crate::linalg::larger(read.value, 0.).sqrt() };
    Bound { shown: probability_error(read.shown.iter().map(size).fold(size(&read.loss), crate::linalg::larger)), temperature: read.temperature.map_or(f64::INFINITY, |t| size(&t)) }
}

/// The step's part of the accounts as far as a rung's reads hold it (`attempt`: the product's need the check is priced at): per
/// look its estimate √max(M̂, 0) (`estimated`), at most the step's share. The check leaves the step it stands at within
/// its share on its estimates (`verdict`) -- a rung beyond is left, and the run stands at a rung below --, so the share
/// bounds the step's part the run will book, and a read below it is that part as read.
fn standing(read: &Between) -> Bound {
    let (read, share) = (estimated(read), Bound::share());
    Bound { shown: crate::linalg::smaller(read.shown, share.shown), temperature: crate::linalg::smaller(read.temperature, share.temperature) }
}

/// What a read does not exclude (decide.rs `Read::most`): each quantity's far end, the rms it leaves open,
/// √(max(M̂, 0) + t·SE) at one look's interval -- the number `blind` holds against the step's share --, the largest on
/// what is shown, and the total T's; unbounded where a read excludes nothing.
fn unexcluded(read: &Between) -> Bound {
    let far = |read: &Read| read.most(1.).sqrt();
    Bound { shown: probability_error(read.shown.iter().map(far).fold(far(&read.loss), crate::linalg::larger)), temperature: read.temperature.map_or(f64::INFINITY, |t| far(&t)) }
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
/// levels' readings): the groups' sums there, each time weighed by its share of the run; of the finer level's error,
/// the coarser one's halved steps taken into it (`amplified`; module documentation, ## A companion at τ/2) -- the
/// finer one's err on the side of more and are not.
fn read_step(times: &[f64], window: f64, fine: &[&Part], coarse: &[&Part], scales: &Scales) -> Between {
    scaled(&read_as_it_is(times, window, fine, coarse, scales), amplified(coarse.iter().copied()))
}

/// `read_step`'s read before its coarser level's halved steps are taken into it: `between` of the two levels as they
/// are.
fn read_as_it_is(times: &[f64], window: f64, fine: &[&Part], coarse: &[&Part], scales: &Scales) -> Between {
    let (fine, coarse) = (reads_of(fine), reads_of(coarse));
    let at = analysed(times, window, &fine, &coarse, scales);
    let there = |groups: &[&Group]| -> Vec<Group> { groups.iter().map(|g| g.at(&at)).collect() };
    between(&there(&fine), &there(&coarse), scales, &at.iter().map(|&k| times[k]).collect::<Vec<_>>())
}

/// Whether a lab axis's temperature (µK) lies in the validated regime of that axis's frequency (rad/µs), k_BT ≥
/// `CLASSICAL` ħω (SPEC §0, §3): what the result reports as classical, per axis; reporting only.
fn classical(temperature: f64, frequency: f64) -> bool {
    KT_PER_UK * temperature >= CLASSICAL * frequency
}






/// The solve's pipeline on its run (module documentation, # The pipeline): the scout, the pilot and the step's check,
/// the choice where it is made on this pilot (# The selection), the product, the result.
fn attempt(request: &Request, setting: &Setting, messages: &mut Messages, choosing: Choosing) -> Result<Attempted, String> {
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
    // The choice the model's error is booked from (module documentation, # The model): the one made, or none -- not
    // measured, said in the ledger (the ledger without it kept for when this pilot makes the choice).
    let mut choice: Option<std::sync::Arc<Choice>> = match &choosing {
        Choosing::Made(c) => c.clone(),
        _ => None,
    };
    let clean = ledger.clone();
    if choice.is_none() {
        ledger.flag(unmeasured());
    }
    // The ladder's first rung (# The step): within the physics prior's step at the initial ensemble's mean kinetic
    // temperature in the reference state's trap (`kinetic_uk`), and below the step's edge.
    let kinetic = kinetic_uk(trap, request.temperature_uk).iter().sum::<f64>() / 3.;
    let (frequency, end) = (trap.frequencies.iter().copied().fold(0., crate::linalg::larger), plan.end);
    let ladder = Ladder { run: end, prior: prior(pattern_wave_number(run.light()), hm, kinetic, &scales), frequency, kick };
    let first_rung = ladder
        .first()
        .ok_or_else(|| format!("找不到可用的时间步长：在每条轨迹的步数上限（{MOST_STEPS} 步）之内，没有步长能分辨阱中的振荡和光场中势阱的曲率（或者阱频、曲率不是有限数）"))?;
    // Whether the rung above a step is a rung of the ladder (`first_rung`'s test at twice the step): the check then
    // reads that rung on a third level, at 4τ (`settle`): a cell of the Brownian path.
    let above = |tau: f64| 4. * tau <= end && stable(2. * tau, frequency, kick) && stable(4. * tau, frequency, kick);
    // What the steps ask of the run's domain (`Domain`).
    let domain = Domain { widest: Widest::of(request) };
    // The analysis times' candidates (# The analysis times): every level is read there while the pilot runs.
    let (candidates, window) = (grid(end, first_rung), plan.window);
    let mut context = Context {
        at: vec![0., end],
        fixed: Fixed::new(vec![[0.; 2]; 2], &[0., end], dropped_rate(run), Escape::new(kinetic_bound(run), hm)),
        step: Bound::share(),
        checked: None,
        tau: first_rung,
        plan: Plan { cell: if above(first_rung) { 4. } else { 2. } * first_rung, ..plan },
        scales,
        initial: row,
        ledger,
        method: "三维半经典运动：内态在选出的态集 P 上演化（P 里的态显式保留，P 以外的能级按 RS 折叠），P 由先导的原子路径上对 32 态模型的实测误差与运动反馈上界选出，模型在每次尝试开头只构造一次".into(),
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
    // ask beside the model's error at those states (`Prices`).
    // A candidate's short scout is the selection's, and says no Scout of its own (# Steps).
    if !matches!(choosing, Choosing::Scout) {
        messages.stage(Stage { step: Step::Scout, tau: Some(context.tau), batch: None });
    }
    let scouting = settle(run, ensemble, &domain, &mut context, &mut levels, &numbers, 0..1, &above, None, messages);
    // A candidate's short scout (`Choosing::Scout`): the rung its own two levels stand on (`settle`: down the ladder
    // to the first rung whose own read is in and not clearly beyond), the check's priced course not taken -- a scout
    // with it took PGC3D's P20 34-68 s, R4's P26 87 s.
    if matches!(choosing, Choosing::Scout) {
        return Ok(Attempted::Scouted(context.tau));
    }
    // The scout's states where its check is priced (`Settled::Pending`): its prices, and the pilot's levels' hand-out's
    // order after them (`foretold`).
    let here = matches!(scouting, Settled::Pending).then(|| {
        let scout: Vec<&Part> = levels.fine.iter().collect();
        let at = analysed_alone(&scout);
        (parts_at(run, &domain, context.tau, &context.plan, &scout, &at), at)
    });
    let refs: Vec<&Part> = here.iter().flat_map(|(here, _)| here.iter()).collect();
    let rated = one_each(&refs);
    let priced = work_at(run, domain.widest, &rated);
    let work = |t: f64| priced(t)[0];
    if let Some((_, at)) = &here {
        let times: Vec<f64> = at.iter().map(|&k| candidates[k]).collect();
        let read = pilot_reads(&refs, &scales, choice.as_deref(), &times, &context.fixed.escape);
        let fixed = context.fixed.with(read.model, &times);
        let need = most_needed(&looks(&read.estimates, &fixed, Bound::share()));
        let halving = |t: f64| priced(t)[1];
        check(run, ensemble, &domain, &mut context, &mut levels, &numbers, 1, &above, Some(&Prices { work: &work, halving: &halving, product: need }), messages);
    }
    // The pilot: its groups' other members at both levels, screened, and the step's check (module documentation,
    // # The step) -- first as far as its own two levels: at the first rung whose own read is in and not clearly beyond,
    // its states are read, and the check goes on from there priced at them. Whether it saw: a step that stands
    // uncertified -- at the steps' cap, or unable to see -- is flagged.
    messages.stage(Stage { step: Step::Pilot, tau: Some(context.tau), batch: None });
    let mut settled = settle(run, ensemble, &domain, &mut context, &mut levels, &numbers, 1..GROUP, &above, here.as_ref().map(|_| &work as &dyn Fn(f64) -> f64), messages);
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
                let times: Vec<f64> = at.iter().map(|&k| candidates[k]).collect();
                let read = pilot_reads(&pilot.iter().collect::<Vec<_>>(), &scales, choice.as_deref(), &times, &context.fixed.escape);
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
            let (own, third, _, _) = levels.told(&candidates, window, &scales);
            standing(&worse(&own, third.as_ref(), false))
        };
        let product = most_needed(&looks(&read.estimates, &context.fixed.with(read.model.clone(), &at.iter().map(|&k| candidates[k]).collect::<Vec<_>>()), step));
        let (_, _, pilot, _) = &*made.insert((tau, at, pilot, read));
        let rated = one_each(&pilot.iter().collect::<Vec<_>>());
        let priced = work_at(run, domain.widest, &rated);
        let (work, halving) = (|t: f64| priced(t)[0], |t: f64| priced(t)[1]);
        settled = check(run, ensemble, &domain, &mut context, &mut levels, &numbers, GROUP, &above, Some(&Prices { work: &work, halving: &halving, product }), messages);
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
    // The pilot's paths at the run's step, handed out where a test asks for them (`walks`; PIPELINE_V3_FINAL §二 B.7).
    #[cfg(test)]
    crate::paths::hand_out(&|count, until| walks(run, &domain, tau, context.plan.end, &levels.fine.iter().collect::<Vec<_>>(), count, until));
    // The step's check (module documentation, # The step): what the reads the rung stands by do not exclude into the
    // accounts, their resolution into the result's words (`Levels::told`, `checked`).
    let (own, third, step_of_third, pairs) = levels.told(&candidates, window, &scales);
    let (step, resolution) = checked(&own, third.as_ref(), saw);
    (context.step, context.checked) = (step, Some(Checked { third: step_of_third, pairs, resolution, saw, companion: levels.companion }));
    crate::prof::note(|| format!("ladder settled tau {tau} companion {:?} saw {saw} third {step_of_third:?} pairs {pairs}: the step's part booked {:.3e} on what is shown, {:.3e} of T's allowance; resolution {resolution:?}", levels.companion, step.shown, step.temperature));
    // The decisions stand on the pilot's analysis times from here on: the product reads its trajectories there, and
    // the run's model error is the pilot's there (# The analysis times). The levels' checkpoints are read no more;
    // the further pairs' groups are the product's, at the same times.
    let times: Vec<f64> = at.iter().map(|&k| candidates[k]).collect();
    let further: Vec<Part> = levels.further[0].iter().map(|p| p.at(&at)).collect();
    context.plan.read_at(times.clone());
    // The choice (module documentation, # The selection), where this pilot makes it: G chosen, the product goes on
    // from here with the model's error booked from it; another, the attempt ends with it.
    let mut read = read;
    if choice.is_none() && !matches!(choosing, Choosing::Made(_)) {
        let pilot_refs: Vec<&Part> = pilot.iter().collect();
        let Some(made) = choose_on_the_pilot(request, setting, &context, first_rung, &levels.fine, &pilot_refs, &read, &numbers, &times, messages)? else {
            return Ok(Attempted::RanAway);
        };
        let made = std::sync::Arc::new(made);
        crate::select::keep(request, made.clone());
        let stop = {
            #[cfg(test)]
            {
                matches!(choosing, Choosing::Choose)
            }
            #[cfg(not(test))]
            {
                false
            }
        };
        if stop || made.keep != run.rs.keep {
            return Ok(Attempted::Chose(made));
        }
        eprint!("{}", made.report.iter().map(|l| format!("  注：{l}\n")).collect::<String>());
        // The ledger as it was but the flag of an error not measured, with what the pilot's check flagged since.
        let flagged = std::mem::replace(&mut context.ledger, clean).flagged;
        if let Some(f) = flagged {
            for part in f.split('；').filter(|p| *p != unmeasured()) {
                context.ledger.flag(part.to_string());
            }
        }
        read = pilot_reads(&pilot_refs, &scales, Some(&made), &times, &context.fixed.escape);
        choice = Some(made);
    }
    drop(levels);
    let n = times.len();
    context.at = times;
    context.fixed = context.fixed.with(read.model.clone(), &context.at);
    // The optical sector's noise (ledger.rs `optical`), its items; its booking on T is in the model's error the reads
    // made (`pilot_reads`).
    crate::ledger::optical(&mut context.ledger, read.counts.optical);
    {
        // The ledger's items: each part of the model's error on what is shown and on T (its largest over the analysis
        // times, weighed by the pilot's held share as the totals are; a time where the pilot holds no atom weighs nothing).
        let held: Vec<f64> = read.estimates.iter().map(|e| 1. - e.loss).collect();
        if let Some(c) = &choice {
            // Where the paths read no direct part, the held atoms' bound (`model_of`).
            let direct = (1..n).filter(|&k| held[k] != 0.).map(|k| c.at(context.at[k]).unwrap_or(held[k])).fold(0., crate::linalg::larger);
            context.ledger.item("模型误差：路径实测的直接部分", direct, MODEL_DIRECT);
            for flag in model_flags(c, &read, request) {
                context.ledger.flag(flag);
            }
        }
        // The motion's part (select.rs, # The motion): its largest over the times, on the loss and on T (by the held share).
        let course = course_of(&read.estimates, &context.at, &context.scales);
        let motion = choice.as_ref().map_or(0., |c| {
            (1..n)
                .filter(|&k| held[k] != 0.)
                .filter_map(|k| {
                    let m = c.motion(&course, k, context.scales.hbar_over_mass)?;
                    let t = read.estimates[k].motion.filter(|x| x.temperature_total > 0.).map_or(0., |x| m.temperature_total / x.temperature_total);
                    Some(crate::linalg::larger(m.loss, held[k] * t))
                })
                .fold(0., crate::linalg::larger)
        });
        context.ledger.item(MODEL_MOTION[0], motion, MODEL_MOTION[1]);
    }

    // The product (module documentation, # Stopping): on from the pilot's trajectories at the run's step, the run's
    // first; the pilot's companions have done their part, and its states have been read.
    let mut parts = pilot;
    forget(&mut parts);
    parts.extend(further);
    let (e, l, c, reads) = product(run, ensemble, &domain, &mut context, &mut parts, &numbers, messages);
    // T's memory not shorter than the run, read on the run's own trajectories (`memory_at_the_end`): flagged where it is.
    if choice.is_some() {
        let allowance = reads.last().and_then(|e| e.motion.map(|m| m.temperature_total * e.temperature_target)).unwrap_or(f64::NAN);
        let (r, se, decided) = memory_at_the_end(&parts.iter().collect::<Vec<_>>(), &context.scales, allowance);
        // At the run's end no atom held: no T there, its memory unread (no flag).
        let said = if allowance.is_finite() && r.is_finite() && se.is_finite() {
            format!("m = {r:.2} ± {se:.2} µK，允许量 {allowance:.2} µK：{}", match decided { Some(true) => "塌掉（运行末仍可读出）", Some(false) => "没塌掉（运行末的上端在允许量以内）", None => "定不下来，不出旗标" })
        } else {
            "运行末没有被抱住的原子，T 无读数，不出旗标".to_string()
        };
        eprintln!("  注：运行末 T 的记忆（运行自己的 {} 条轨迹，初态动能倾斜的似然比）：{said}", parts.len() as u64 * GROUP);
        if decided == Some(true) {
            context.ledger.flag("温度的记忆时间不短于运行（运行自己轨迹上的动量缩放响应在运行末仍可读出：Λ 塌掉）".to_string());
        }
    }
    // A point of the prepared potential below the bottom the ensemble was drawn about (ensemble.rs, # Its draw).
    if ensemble.deeper() > deeper {
        context.ledger.flag("初态的势在所取的阱底之外还有更深的地方：初始系综只取到该阱底的阱深为止".into());
    }
    Ok(Attempted::Solved(assemble(&context, &e, &reads, &l, &c, parts.len() as u64 * GROUP)))
}

/// The flag of a model whose error is not measured (module documentation, # The model): G's before its pilot chooses,
/// a test's model of its own.
fn unmeasured() -> String {
    "模型误差尚未测得（选态在先导之后做；这之前的结果不含模型误差）".to_string()
}

/// What the choice says of itself that a result is less for (module documentation, # The selection; PIPELINE_V3_FINAL
/// C.15): T still drifting at the run's end, the paths run through the whole request (the memory of the dark states longer than a fifth of it), and
/// nothing more.
fn model_flags(choice: &Choice, read: &PilotRead, request: &Request) -> Vec<String> {
    let mut flags = vec![];
    // T still drifting at the end: its last two analysis times apart by more than their intervals together.
    let held: Vec<(f64, f64)> = read.estimates.iter().filter_map(|e| e.motion.map(|m| (m.temperature_total, m.temperature_total_error.interval()))).collect();
    if let [.., (a, ea), (b, eb)] = held[..] {
        if (b - a).abs() > ea.hypot(eb) {
            flags.push(format!("温度在运行末段仍在漂移：最后两个分析时刻差 {:.2} µK，超过它们的误差棒 {:.2} µK", b - a, ea.hypot(eb)));
        }
    }
    let length = request.times_us.last().copied().unwrap_or(0.);
    if 5. * choice.memory >= length {
        flags.push(format!("暗态的长记忆：内态记忆时间 {:.3} µs，路径跑满整个请求（{length} µs）", choice.memory));
    }
    flags
}

/// The pilot's reads at the rung it stands at (`attempt`, `pilot_reads`): its estimates and counts, and the model's
/// error per time -- the choice's parts and the optical sector's booking on T.
struct PilotRead {
    estimates: Vec<Estimate>,
    counts: Counts,
    model: Vec<[f64; 2]>,
}

/// The pilot's reads (`PilotRead`) of its groups `pilot` at the times `times`: the model's error (SPEC §5.2; # The
/// model), the choice's parts at each time (`Choice::at`, `Choice::motion` on the pilot's own course; none where there is
/// no choice), and the optical sector's noise (ledger.rs `optical`), its booking on T at every time, a part of it.
fn pilot_reads(pilot: &[&Part], scales: &Scales, choice: Option<&Choice>, times: &[f64], escape: &Escape) -> PilotRead {
    let (estimates, counts) = (summarize(&reads_of(pilot), scales), counts_of(pilot));
    let optical = crate::ledger::optical(&mut Ledger::default(), counts.optical);
    // On the reference itself (P all 32 states) no model error at any time, as the choice weighed it (`choose_on_the_pilot`).
    let course = course_of(&estimates, times, scales);
    let model = if choice.is_some_and(|c| c.keep.len() == STATES) { vec![[0.; 2]; times.len()] } else { model_of(&|t| choice.map_or(Some(0.), |c| c.at(t)), &|k| choice.and_then(|c| c.motion(&course, k, scales.hbar_over_mass)), &estimates, times, optical, escape) };
    PilotRead { estimates, counts, model }
}

/// The model's error per time (`pilot_reads`) of its direct part `direct(t)`, its motion's errors at the k-th time
/// `motion(k)` (select.rs `Motion`) and the optical sector's booking `optical` on T: on what is shown the larger of the
/// direct part (the rows) and the motion's on the loss, at most one; on T the optical booking and the motion's on the
/// total T, relative to it; none at t = 0, the request's state. Where the paths read no direct part (no path's atom held
/// there, `select::direct_at`) the held atoms' rows are unread: their bound, the share of the atoms held there (1 − L:
/// none where every atom is lost, nothing held to err on).
fn model_of(direct: &dyn Fn(f64) -> Option<f64>, motion: &dyn Fn(usize) -> Option<crate::select::Motion>, estimates: &[Estimate], times: &[f64], optical: f64, escape: &Escape) -> Vec<[f64; 2]> {
    estimates
        .iter()
        .enumerate()
        .map(|(k, e)| {
            if k == 0 {
                return [0.; 2];
            }
            let m = motion(k).unwrap_or_default();
            let temperature = e.motion.filter(|x| x.temperature_total > 0.).map_or(0., |x| m.temperature_total / x.temperature_total);
            let _ = escape;
            [probability_error(crate::linalg::larger(direct(times[k]).unwrap_or(1. - e.loss), m.loss)), optical + temperature]
        })
        .collect()
}

/// The pilot's course (select.rs `Course`) of its estimates at the times: its loss, its T per lab axis where it holds
/// atoms, and the reported trap's frequencies.
fn course_of(estimates: &[Estimate], times: &[f64], scales: &Scales) -> crate::select::Course {
    crate::select::Course { times: times.to_vec(), loss: estimates.iter().map(|e| e.loss).collect(), temperature: estimates.iter().map(|e| e.motion.map(|m| m.temperature)).collect(), frequencies: scales.frequencies }
}

// ---------------------------------------------------------------------------------------------------------------------
// The selection on the pilot (module documentation, # The selection).

/// A trajectory's work a µs at the step τ at the states `rated` on a model `model` of the chain (`work_at`'s count at
/// one step): each state's place, momentum and time, its internal state the ground state of its largest population
/// (the state's own where the model is the one it was run on), the step taken from it as the run takes it, and its
/// jumps' work as on the pilot's model (a jump's work is its photons' and landings'); and the share of the states,
/// weighed as the price is (`over_the_run`), from which no step is taken -- a fold that runs away there (step.rs # The domain): such a state's trajectory stops there, unresolved, and works nothing on;
/// its share is booked whole in the candidate's account (`choose_on_the_pilot`, # The unresolved atoms). (Until 2026-11-22
/// one such state made the model's price without bound: p5d5's G, one of its 254 pilot's states at the trap's rim, s =
/// 0.37; Study A §2.4.)
fn priced_with(model: &Selected, widest: Widest, rated: &[Pick], tau: f64, own: bool) -> (f64, f64) {
    let taken: Vec<f64> = rated
        .par_iter()
        .map(|pick| {
            let mine = pick.state.trajectory();
            let from = if own {
                mine
            } else {
                let states = &pick.state.sample.psi;
                let ground = (0..GROUND).max_by(|&a, &b| states.get(a).map_or(0., |z| z.norm_sqr()).total_cmp(&states.get(b).map_or(0., |z| z.norm_sqr()))).unwrap_or(0);
                let mut t = Trajectory::new(model, ground, mine.r, mine.kappa, SEED, mine.index);
                (t.time, t.threshold, t.jumps) = (mine.time, mine.threshold, mine.jumps);
                t
            };
            let mut refused = 0.;
            match adapt::halving(from.time, tau, |length| crate::step::priced(model, &from, length, &widest).inspect_err(|_| refused += refused_at(model, &from, length, &widest))) {
                Ok((stretches, length)) => (stretches.iter().map(|(_, price)| price.work(price.costs)).sum::<f64>() + refused) / length + pick.jumping,
                Err(_) => f64::INFINITY,
            }
        })
        .collect();
    (over_the_run(rated, taken.iter().map(|&x| if x.is_finite() { x } else { 0. }), None), over_the_run(rated, taken.iter().map(|&x| if x.is_finite() { 0. } else { 1. }), None))
}

/// T's memory at the run's end, read on the run's own trajectories at no trajectory more (select.rs, # T's memory, read
/// on the run's own trajectories): R_T(t) = dT(t)/dε of the initial tilt e^{−εK}, by the likelihood ratio the run's T
/// estimate with its reads reweighed by 1 + ε(K̄ − K) (`Group::tilted`; K each trajectory's classical kinetic energy,
/// ensemble.rs `Ensemble::draw_kinetic`, K̄ the run's own mean) -- the difference at ±h over 2h --; T's memory at the
/// end the response there over its start's, R_T(end)/R_T(0), and on the momenta's-scaling scale the question asks it on
/// (`select::decide`: a unit scaling moves T(0) by 2T(0)) m = 2T(0)·R_T(end)/R_T(0) (µK), its standard error the groups'
/// jackknife of that ratio; the flag's question against T's allowance at the end `allowance` (µK). The memory m and its
/// error, and the answer.
fn memory_at_the_end(parts: &[&Part], scales: &Scales, allowance: f64) -> (f64, f64, Option<bool>) {
    let h = MEMORY_TILT;
    // K̄: every trajectory's K over their number, at t = 0 (every trajectory counted there, failed or not).
    let total = |g: &Group| g.sums.first().map_or(0., |s| s.trajectories + s.failed);
    let mean = parts.iter().map(|p| total(&p.kinetic)).sum::<f64>() / parts.iter().map(|p| total(&p.reads)).sum::<f64>();
    let tilted = |sign: f64| -> Vec<Group> { parts.iter().map(|p| p.reads.tilted(&p.reads, sign * h * mean).tilted(&p.kinetic, -sign * h)).collect() };
    let (plus, minus) = (tilted(1.), tilted(-1.));
    let (a, b): (Vec<&Group>, Vec<&Group>) = (plus.iter().collect(), minus.iter().collect());
    let plain: Vec<&Group> = parts.iter().map(|p| &p.reads).collect();
    let none: [&Group; 0] = [];
    let last = parts.first().map_or(0, |p| p.reads.sums.len()).saturating_sub(1);
    let t = |r: &[crate::analysis::Reading], g: Option<usize>| (0..3).map(|q| g.map_or(r[1 + STATES + q].whole, |g| r[1 + STATES + q].leave[g])).sum::<f64>() / 3.;
    let at = |k: usize| (readings(&a, &none, k, scales), readings(&b, &none, k, scales));
    let ((x0, y0), (x1, y1)) = (at(0), at(last));
    let start = readings(&plain, &none, 0, scales);
    let m = |g: Option<usize>| 2. * t(&start, g) * (t(&x1, g) - t(&y1, g)) / (t(&x0, g) - t(&y0, g));
    let groups = parts.len();
    let (r, se) = (m(None), crate::estimate::spread((0..groups).map(|g| m(Some(g))).collect::<Vec<_>>().into_iter(), groups as f64).error);
    (r, se, crate::select::decide(r, se, allowance))
}

/// The reweighing's step h of `memory_at_the_end`: the run's estimate is smooth in it, and its difference over ±h is
/// the derivative to O(h²) with a rounding of ε/h -- 2⁻²⁰ leaves both near 10⁻¹².
const MEMORY_TILT: f64 = 1. / (1u64 << 20) as f64;

/// A line of the choice's report (`Choice::report`), said as it is made (stderr: the solve's own log).
fn noted(report: &mut Vec<String>, line: String) {
    eprintln!("  选态：{line}");
    report.push(line);
}

/// One candidate as the choice weighs it (`choose_on_the_pilot`): its place on the chain, its judgment, its work a µs,
/// the model's error per time it books, the product's trajectories its looks need, whether its looks meet what the
/// exact model's would (on what is shown, on T), its total work W where it is complete, and its fate in words.
struct Weighed {
    place: usize,
    judged: crate::select::Judged,
    price: f64,
    need: f64,
    work: Option<f64>,
    fate: String,
    /// Whether its fate is open at the paths' statistics: complete at its direct part's lower end and not at its upper
    /// (select.rs, # How many paths).
    open: bool,
    /// Its W were its fate decided complete: at its account's lower end (# How many paths).
    settled_low: f64,
}

/// The choice of P on the pilot of the attempt's model -- G, or the chain's first whose pilot went past its start
/// (`piloted`) -- (module documentation, # The selection): the paths; the pilot's model judged first, then -- where it
/// is not complete -- the least complete candidate of the chain above it found by bisection (the chain nested, each
/// candidate's error no less than the next's: a candidate complete, every one above it taken complete), then the chain
/// above that while a candidate's least W stands below the least found, every candidate judged on the way weighed; the
/// complete ones scouted at their own step while their least W stands below the least found; P the complete candidate
/// of least W. None where no path went past its start because the model's fold scattered beyond
/// every state of the atom there (a pilot's trajectory ran away at t = 0, step.rs # The domain): nothing to choose on.
#[allow(clippy::too_many_arguments)]
fn choose_on_the_pilot(request: &Request, setting: &Setting, context: &Context, first: f64, fine: &[Part], pilot: &[&Part], read: &PilotRead, numbers: &[u64], times: &[f64], messages: &mut Messages) -> Result<Option<Choice>, String> {
    use crate::select::{direct_at, direct_bound_at, judge, Paths, Reach, WALKS};
    let clock = Instant::now();
    let run = setting.run;
    let trap = setting.trap;
    let hm = run.hbar_over_mass();
    let tau = context.tau;
    let domain = Domain { widest: Widest::of(request) };
    let light = run.light();
    let centre = trap.centre;
    messages.stage(Stage { step: Step::Select, tau: None, batch: None });
    let reference = alone(chief(request));
    let length = request.times_us.last().copied().unwrap_or(0.);
    let mut report: Vec<String> = vec![];
    // The chain on the first-order excitation where the pilot's atoms went, from the ground states they were in
    // (select.rs `importance`; a model's cluster holds its ground states first, model.rs `Cluster`), from the pilot's own
    // model up: every model of it holds the pilot's states, the pilot's first (place 0, the attempt's run).
    let started = Instant::now();
    let places: Vec<(V3, [f64; GROUND])> = pilot
        .iter()
        .flat_map(|p| p.samples.iter().flatten())
        .map(|s| {
            let n: f64 = s.sample.psi.iter().map(|z| z.norm_sqr()).sum();
            (s.sample.at.r, std::array::from_fn(|g| s.sample.psi.get(g).map_or(0., |z| z.norm_sqr() / n)))
        })
        .collect();
    let amplitudes = crate::select::importance(&run.scene, &places);
    let own = &run.rs.keep;
    let labels = &request.states;
    // The excited states a candidate may add (select.rs # The candidates): those the pilot's model folds that the light
    // excites where its atoms were (a first-order excitation of nought couples nothing there).
    let addable: Vec<usize> = (GROUND..STATES).filter(|s| !own.contains(s) && amplitudes[*s] > 0.).collect();
    noted(
        &mut report,
        format!(
            "先导立定于 τ = {tau:.4} µs，{} 个分析时刻；先导的模型 P{}{}；可加回的激发态 {} 个（先导的 {} 个状态上一阶激发不为零），{:.1} s",
            times.len(), run.rs.keep.len(), if run.rs.keep.len() == GROUND { "（G）" } else { "（G 的折叠在先导起点失控，结构链上第一个走得动的）" }, addable.len(), places.len(), started.elapsed().as_secs_f64()
        ),
    );
    // 1. The paths: the pilot's first trajectories' starts, the memory time there (the longest), the walks to L -- as
    // many as the judgment's statistics ask (select.rs, # How many paths): from `WALKS`, doubled while a candidate's fate
    // is open, up to the pilot's trajectories, every candidate judged again on them.
    let fine_refs: Vec<&Part> = fine.iter().collect();
    let available = fine.iter().flat_map(|p| p.trails.iter()).filter(|t| !t.is_empty()).count();
    // The candidates' models and their prices a µs, each made once for the choice: neither depends on the paths, which
    // are doubled and every candidate judged again on them, not made or priced again. A model by its states; a price by
    // the model's states, the step and whether the pilot's states are its own (`priced_with`: the price and the share
    // no step leaves).
    let rated = one_each(pilot);
    let built: std::sync::Mutex<std::collections::HashMap<Vec<usize>, std::sync::Arc<Selected>>> = Default::default();
    let costed: std::sync::Mutex<std::collections::HashMap<(Vec<usize>, u64, bool), (f64, f64)>> = Default::default();
    let make = |keep: &[usize]| -> Result<std::sync::Arc<Selected>, String> {
        if let Some(m) = built.lock().expect("the models' cache").get(keep) {
            return Ok(m.clone());
        }
        let model = std::sync::Arc::new(Selected::new(crate::scene::Scene::new(request)?, Selection { keep: keep.to_vec(), ..run.report.clone() }, reference)?);
        built.lock().expect("the models' cache").insert(keep.to_vec(), model.clone());
        Ok(model)
    };
    let price = |model: &Selected, step: f64, own: bool| -> (f64, f64) {
        let key = (model.rs.keep.clone(), step.to_bits(), own);
        if let Some(&p) = costed.lock().expect("the prices' cache").get(&key) {
            return p;
        }
        let p = priced_with(model, domain.widest, &rated, step, own);
        costed.lock().expect("the prices' cache").insert(key, p);
        p
    };
    let full = make(&(0..STATES).collect::<Vec<_>>())?;
    let (head, mut doubled, mut count) = (report, Vec::<String>::new(), WALKS.min(available));
    // A test's fixed count of paths (`SEL_WALKS`): that many, never doubled (the measurements' probe).
    #[cfg(test)]
    let pinned = std::env::var("SEL_WALKS").ok().map(|w| w.parse::<usize>().expect("SEL_WALKS: a count").min(available));
    #[cfg(not(test))]
    let pinned: Option<usize> = None;
    if let Some(w) = pinned {
        count = w;
    }
    loop {
        let mut report = head.clone();
        report.extend(doubled.iter().cloned());
        let firsts: Vec<&Checkpoint> = fine.iter().flat_map(|p| p.trails.iter()).filter_map(|t| t.first()).take(count).collect();
        let form = crate::paths::Form::new(run, centre);
        let memory = firsts.iter().filter_map(|c| crate::paths::memory(&form, c.trajectory.r, c.trajectory.kappa, c.trajectory.time)).map(|m| m.time).fold(0., crate::linalg::larger);
        let memory = if memory > 0. { memory } else { length };
        drop(form);
        let walked = walks(run, &domain, tau, context.plan.end, &fine_refs, count, length);
        // No path past its start, the model's fold beyond every state of the atom where a pilot's atom started: the chain's
        // next model is the pilot's (`piloted`). (No path past its start otherwise -- every atom lost at its first step --
        // is said by `Paths::new`.)
        if walked.iter().all(|w| !(w.end() > 0.)) && read.counts.runaway.since == Some(0.) {
            return Ok(None);
        }
        let paths = Paths::new(&full, centre, walked, memory, length, tau, light)?;
        let settled = &paths.settled;
        noted(
            &mut report,
            format!(
                "路径：{} 条先导路径，L = {:.1} µs（τ_mem {:.3} µs，各路径起点上最长），{} 个窗{}；格式 指数 Runge–Kutta：开头（{:.2} µs，起点的光学相干衰减到舍入）最快的模（转动加衰减）每转 {} 个子步（只分辨到 {:.0} rad/µs，更快的按严格界记账）、其后每 µs {:.1} 个，每步 {} 段（按误差定：参照的误差 开头子步 {:.1e} + 其后子步 {:.1e} + 分段 {:.1e} + 开头记账 {:.1e} + 迹偏差 {:.1e}（最长一条路径全长 {:.1} µs 上，先在前 {} µs 上爬梯；迹偏差每条路径都查），容差 {:.1e}{}；试了 {} 种设置，{:.1} s）；参照沿全部路径的最大迹偏差 {:.1e}（记进每个候选的 A）；参照 32 态 {:.1} s",
                paths.walks.len(), paths.length, memory, paths.windows.len(), paths.late_from().map_or(String::new(), |f| format!("，另有运行末的一段 [{f:.0}, {length}] µs")),
                settled.opening, settled.setting.turns, settled.setting.cutoff, settled.setting.rate, settled.setting.pieces, settled.errors[0], settled.errors[1], settled.errors[2], settled.errors[3], settled.trace, paths.calibrated_over(), crate::select::CALIBRATION, crate::select::TOLERANCE,
                settled.unmet().map_or(String::new(), |e| format!("；到顶未收敛：各旋钮误差之和 {e:.1e} 照实记进每个候选的 A")), settled.tried, settled.seconds, paths.trace, paths.seconds
            ),
        );
        messages.alive();
        // 2. T's memory on the pilot's own trajectories, said beside the choice (the run's flag is the product's).
        {
            let end = read.estimates.last().and_then(|e| e.motion.map(|m| m.temperature_total * e.temperature_target)).unwrap_or(f64::NAN);
            let (r, se, decided) = memory_at_the_end(pilot, &context.scales, end);
            let held: Vec<(f64, f64)> = read.estimates.iter().filter_map(|e| e.motion.map(|m| (m.temperature_total, m.temperature_total_error.interval()))).collect();
            let drift = match held[..] {
                [.., (a, ea), (b, eb)] if ea.hypot(eb).is_finite() => format!("末两个分析时刻 T 差 {:.2} µK、误差棒 {:.2} µK：{}", b - a, ea.hypot(eb), if (b - a).abs() > ea.hypot(eb) { "漂移（出旗标）" } else { "不漂移" }),
                [.., (a, _), (b, _)] => format!("末两个分析时刻 T 差 {:.2} µK、误差棒无读数：定不下来", b - a),
                _ => "末段无 T".to_string(),
            };
            // At the run's end no atom held: no T there, its memory unread.
            let memory = if end.is_finite() && r.is_finite() && se.is_finite() {
                format!("运行末 m = {r:.2} ± {se:.2} µK，允许量 {end:.2} µK：{}", match decided { Some(true) => "塌掉", Some(false) => "没塌掉", None => "定不下来" })
            } else {
                "运行末没有被抱住的原子，T 无读数".to_string()
            };
            noted(&mut report, format!("先导自己 {} 条轨迹上的 T 记忆（初态动能倾斜的似然比，只作对照；旗标在正式运行末读）：{memory}；{drift}", pilot.len() as u64 * GROUP));
        }
        // 3. The candidates.
        let fixed = |model: Vec<[f64; 2]>| context.fixed.with(model, times);
        let base = looks(&read.estimates, &fixed(vec![[0.; 2]; times.len()]), context.step);
        let need0 = most_needed(&base);
        let ran = (numbers.len() as u64 * GROUP) as f64;
        // The run's work over a trajectory's time (µs: the plan's end, every trajectory runs to it); the scout's and the
        // pilot's again on a model not G (two levels, the companion's at most half the run level's work).
        let span = context.plan.end;
        let again = |place: usize, price: f64| if place == 0 { 0. } else { 1.5 * ran * price * span };
        let met = |looks: &[Look]| -> [bool; 2] {
            let mut met = [true; 2];
            for (b, l) in base.iter().zip(looks) {
                if b.fits() && !l.fits() {
                    met[usize::from(!l.shown)] = false;
                }
            }
            met
        };
        let optical = crate::ledger::optical(&mut Ledger::default(), read.counts.optical);
        // The pilot's course the motion's errors are read on (select.rs, # The motion).
        let course = course_of(&read.estimates, times, &context.scales);
        // A candidate's model error per time at its direct part's upper end (the one booked) and at its lower.
        let accounts = |j: &crate::select::Judged| -> [Vec<[f64; 2]>; 2] {
            [Reach::Upper, Reach::Lower].map(|reach| model_of(&|t| direct_bound_at(j, &paths.windows, t, reach), &|k| crate::select::motion_at(j, &paths.windows, &course, k, hm, reach), &read.estimates, times, optical, &context.fixed.escape))
        };
        // A candidate's account at its point estimates (the nomination's, select.rs # The candidates), and its misfit there:
        // its model error over its target, the largest over the times and the two kinds -- what is shown over the
        // population target, T over its allowance there.
        let point = |j: &crate::select::Judged| -> Vec<[f64; 2]> { model_of(&|t| direct_bound_at(j, &paths.windows, t, Reach::Point), &|k| crate::select::motion_at(j, &paths.windows, &course, k, hm, Reach::Point), &read.estimates, times, optical, &context.fixed.escape) };
        let misfit = |point: &[[f64; 2]]| -> f64 { (1..times.len()).map(|k| crate::linalg::larger(point[k][0] / POPULATION_TARGET, point[k][1] / read.estimates[k].temperature_target)).fold(0., crate::linalg::larger) };
        // An account with the states its model cannot step from booked whole (`priced_with`): on what is shown their share,
        // on T the bound set's move of it (# The unresolved atoms).
        let unresolved = |mut m: Vec<[f64; 2]>, runaway: f64| -> Vec<[f64; 2]> {
            if runaway > 0. {
                for (k, e) in m.iter_mut().enumerate().skip(1) {
                    *e = [probability_error(e[0] + runaway), e[1] + read.estimates[k].moved(runaway, &context.fixed.escape)];
                }
            }
            m
        };
        // What an account asks of the run: the kinds of look it meets of those the run's own meets (`met`), the trajectories
        // its looks need, and its W where it meets them all -- that need at its price a µs over the run, the pilot's again
        // beside where it is not the pilot's model.
        let asks = |m: Vec<[f64; 2]>, price: f64, place: usize| -> ([bool; 2], f64, Option<f64>) {
            let own = looks(&read.estimates, &fixed(m), context.step);
            let (met, need) = (met(&own), most_needed(&own));
            (met, need, (met == [true; 2]).then(|| need * price * span + again(place, price)))
        };
        // A candidate's name: its states' count, and the excited state it adds where it adds one to the pilot's model.
        let named = |keep: &[usize]| -> String {
            match keep.iter().find(|s| !own.contains(s)) {
                Some(&e) if keep.len() == own.len() + 1 => format!("P{}（+{}）", keep.len(), labels[e]),
                _ => format!("P{}", keep.len()),
            }
        };
        let account = |j: &crate::select::Judged| -> Vec<[f64; 2]> { accounts(j).into_iter().next().expect("the upper end") };
        // The candidates (select.rs # The candidates): the pilot's model first (place 0), the nominees above it.
        let mut chain: Vec<Vec<usize>> = vec![own.clone()];
        let mut models: Vec<Option<std::sync::Arc<Selected>>> = vec![None];
        let mut weighed: Vec<Weighed> = vec![];
        // A candidate judged on every path, its price a µs and the share of the pilot's states it cannot step from
        // (`priced_with`), and the seconds it took. The 32 states are the reference itself: no model error at any time
        // (`Paths::itself`), its looks the run's own (`base`) -- complete by construction, whatever its atoms do (the
        // invariant the choice stands on).
        let judgment = |place: usize, keep: &[usize], model: &Selected| -> Result<(crate::select::Judged, f64, f64, f64), String> {
            let started = Instant::now();
            let (price, runaway) = price(model, tau, place == 0);
            let judged = if keep.len() == STATES { paths.itself(keep.to_vec()) } else { judge(&paths, model, centre)? };
            Ok((judged, price, runaway, started.elapsed().as_secs_f64()))
        };
        // A candidate weighed on its judgment.
        let booked = |place: usize, (judged, price, runaway, seconds): (crate::select::Judged, f64, f64, f64), report: &mut Vec<String>| -> Weighed {
            let exact = judged.keep.len() == STATES;
            let [model_error, lower] = if exact { [vec![[0.; 2]; times.len()], vec![[0.; 2]; times.len()]] } else { accounts(&judged) };
            // The states it cannot step from: unresolved, booked whole.
            let (model_error, lower) = (unresolved(model_error, runaway), unresolved(lower, runaway));
            #[cfg(test)]
            if std::env::var_os("SEL_MOTION").is_some() {
                for (k, e) in read.estimates.iter().enumerate() {
                    eprintln!("    [account] P{} t {:.2}: upper {:.3e} / {:.3e}, lower {:.3e} / {:.3e}; T {:?}, L {:.4}, T's target {:.3}", judged.keep.len(), times[k], model_error[k][0], model_error[k][1], lower[k][0], lower[k][1], e.motion.map(|m| m.temperature_total), e.loss, e.temperature_target);
                }
            }
            let (met, need, work) = asks(model_error, price, place);
            // Complete at the direct part's lower end as well where it is not at its upper: its fate open at the paths'
            // statistics (select.rs, # How many paths); what it would cost were its fate decided complete, its need there.
            let (at_least, need_low, _) = asks(lower, price, place);
            let open = met != [true; 2] && at_least == [true; 2];
            let settled_low = need_low * price * span + again(place, price);
            debug_assert!(!exact || met == [true; 2], "the reference itself meets every look the run's own does");
            // Its fate in words: complete where every look the run's own meets (no model error, `base`) is still met at its
            // model error -- the room that error leaves the statistics filled within the trajectories' cap (`Look::fits`) --,
            // else the first kind of look it leaves unmet, what is shown (the populations and the loss) before T.
            let fate = match met {
                [true, true] => "模型误差留给统计的余量够用",
                [true, false] => "温度：模型误差留给统计的余量在轨迹上限内不够，不选",
                _ => "显示量（布居、丢失）：模型误差留给统计的余量在轨迹上限内不够，不选",
            }
            .to_string();
            let runaway_said = if runaway > 0. { format!("（先导的状态里有 {} 走不了一步，按未解决整份记账）", percent(runaway)) } else { String::new() };
            noted(report, format!("候选 {}：每 µs 代价 {price:.3e}{runaway_said}，{fate}，{seconds:.1} s", named(&judged.keep)));
            Weighed { place, judged, price, need, work, fate, open, settled_low }
        };
        // A candidate judged and weighed (its model made where it is not the pilot's).
        let weigh = |place: usize, keep: &[usize], models: &mut Vec<Option<std::sync::Arc<Selected>>>, report: &mut Vec<String>| -> Result<Weighed, String> {
            if place > 0 && models[place].is_none() {
                models[place] = Some(make(keep)?);
            }
            Ok(booked(place, judgment(place, keep, models[place].as_deref().unwrap_or(run))?, report))
        };
        weighed.push(weigh(0, own, &mut models, &mut report)?);
        messages.alive();
        let mut found = weighed[0].work.unwrap_or(f64::INFINITY);
        // A fate open at the paths' statistics is settled on more paths before anything is built on it (# How many paths):
        // where more paths can be had, the nomination and the search wait for them.
        let more_paths = count < available && pinned.is_none();
        let settle_first = weighed[0].open && more_paths;
        // A model with more states works no less a µs at the same step than the pilot's (its states more): at the ladder's
        // first rung -- the longest its own scout may stand at --, at no model error, its pilot again, its W is no less than
        // that bound. Where the pilot's model is complete below it, no nominee can cost less, and none is made.
        let price_own = price(run, first, true).0;
        let floor = need0 * price_own * span + again(1, price_own);
        let mut predicted: Vec<f64> = vec![f64::NAN];
        // The chain's first prefix of the nomination (after the singles weighed alone), and whether a fate open at the paths'
        // statistics waits for more paths before any nominee is judged.
        let mut prefixes = 1;
        let mut settle = settle_first;
        if floor < found && !addable.is_empty() && !settle_first {
            // The nomination (select.rs # The candidates): each addable state alone added to the pilot's model, judged on the
            // first `NOMINATION_WALKS` paths at its point estimates -- the misfit it takes away per work it adds a µs, and
            // whether it meets every look there --; those that do judged alone on every path and weighed; the states in the
            // order of those singles' W, the least first, then the others' misfit taken away per work, those that take none
            // away after them by their excitation; the nominees the order's prefixes, the 32 states last.
            let clock = Instant::now();
            let few = crate::select::NOMINATION_WALKS.min(paths.walks.len());
            let price_tau = price(run, tau, true).0;
            // A single whose least W -- at no model error, at the ladder's first rung, its pilot again -- is no lower than
            // the least found, and every nominee holding it, can cost no less: it is not judged, and nominated last.
            let made: Vec<(usize, std::sync::Arc<Selected>, bool)> = addable
                .par_iter()
                .map(|&e| -> Result<(usize, std::sync::Arc<Selected>, bool), String> {
                    let mut keep = own.clone();
                    keep.push(e);
                    keep.sort_unstable();
                    let model = make(&keep)?;
                    let least = price(&model, first, false).0;
                    Ok((e, model, need0 * least * span + again(1, least) < found))
                })
                .collect::<Result<_, _>>()?;
            let mine = if made.iter().any(|x| x.2) { misfit(&point(&crate::select::judge_on(&paths, run, centre, few)?)) } else { f64::NAN };
            let with = |e: usize| -> Vec<usize> {
                let mut keep = own.clone();
                keep.push(e);
                keep.sort_unstable();
                keep
            };
            // Per single: its state, the misfit it takes away, the work it adds a µs, whether it meets every look at its point
            // estimates on those paths (its unresolved states booked whole, `asks`), and its model.
            let singles: Vec<(usize, f64, f64, bool, std::sync::Arc<Selected>)> = made
                .into_par_iter()
                .map(|(e, model, open)| -> Result<(usize, f64, f64, bool, std::sync::Arc<Selected>), String> {
                    if !open {
                        return Ok((e, f64::NAN, f64::INFINITY, false, model));
                    }
                    let (price, runaway) = price(&model, tau, false);
                    let at = point(&crate::select::judge_on(&paths, &model, centre, few)?);
                    let taken = mine - misfit(&at);
                    let (met, _, _) = asks(unresolved(at, runaway), price, 1);
                    Ok((e, taken, price - price_tau, met == [true; 2], model))
                })
                .collect::<Result<_, _>>()?;
            let mut order: Vec<usize> = (0..singles.len()).collect();
            let score = |i: usize| if singles[i].1 > 0. { singles[i].1 / singles[i].2.max(f64::MIN_POSITIVE) } else { f64::NEG_INFINITY };
            order.sort_by(|&a, &b| score(b).total_cmp(&score(a)).then(amplitudes[singles[b].0].total_cmp(&amplitudes[singles[a].0])).then(singles[a].0.cmp(&singles[b].0)));
            // The singles that meet every look at their point estimates there: their need there is the run's own statistics'
            // (R4 2026-11-23: 771 to 794 trajectories each, the pilot's 780), and what tells their W apart is the paths'
            // spread, which two paths do not see (on every path 833 to 3133; the one first by misfit on two paths 7th of ten by
            // its W). Each is judged on every path, together, and weighed alone.
            let alone: Vec<usize> = order.iter().copied().filter(|&i| singles[i].3).collect();
            let clock_alone = Instant::now();
            let judged_alone: Vec<_> = alone.par_iter().map(|&i| judgment(1, &with(singles[i].0), &singles[i].4)).collect::<Result<_, _>>()?;
            let info: Vec<(usize, f64)> = singles.iter().map(|s| (s.0, s.1)).collect();
            let mut singles: Vec<Option<std::sync::Arc<Selected>>> = singles.into_iter().map(|s| Some(s.4)).collect();
            let mut alone_work: Vec<(usize, f64)> = vec![];
            for (&i, j) in alone.iter().zip(judged_alone) {
                let place = chain.len();
                chain.push(with(info[i].0));
                models.push(singles[i].take());
                predicted.push(mine - info[i].1);
                let w = booked(place, j, &mut report);
                found = found.min(w.work.unwrap_or(f64::INFINITY));
                settle |= w.open && more_paths;
                alone_work.push((i, w.work.unwrap_or(f64::INFINITY)));
                weighed.push(w);
            }
            // The order the prefixes are made in: the singles weighed alone by their W (the least first; those not complete
            // after them as they were), then the others as they were.
            alone_work.sort_by(|a, b| a.1.total_cmp(&b.1));
            let ordered: Vec<usize> = alone_work.iter().map(|a| a.0).chain(order.iter().copied().filter(|i| !alone.contains(i))).collect();
            prefixes = chain.len();
            let mut keep = own.clone();
            let mut left = mine;
            for (k, &i) in ordered.iter().enumerate() {
                let (e, taken) = info[i];
                keep.push(e);
                keep.sort_unstable();
                if taken > 0. {
                    left -= taken;
                }
                // The first prefix is a single: weighed alone already, or the nominee with the single's own model.
                if k == 0 && alone.contains(&i) {
                    continue;
                }
                chain.push(keep.clone());
                models.push(if k == 0 { singles[i].take() } else { None });
                predicted.push(left);
            }
            if chain.last().map(|c| c.len()) != Some(STATES) {
                chain.push((0..STATES).collect());
                models.push(None);
                predicted.push(0.);
            }
            let said = if mine.is_nan() {
                format!("提名：每个可加回的激发态单独加进先导的模型，最小 W 都不低于已找到的 {found:.3e}，都不判；{:.1} s", clock.elapsed().as_secs_f64())
            } else {
                format!(
                    "提名（每个可加回的激发态单独加进先导的模型，在 {few} 条路径上按点估计判：去掉的失配 / 每 µs 多出的代价，最小 W 不低于已找到的不判；点估计下显示量与温度都够的 {} 个单独在全部路径上判、按 W 排在前面，{:.1} s）：先导模型的失配 {mine:.3}；{}；{:.1} s",
                    alone.len(),
                    clock_alone.elapsed().as_secs_f64(),
                    ordered.iter().map(|&i| alone_work.iter().find(|a| a.0 == i).map_or(format!("{}", labels[info[i].0]), |a| format!("{}（W {}）", labels[info[i].0], if a.1.is_finite() { format!("{:.2e}", a.1) } else { "不够".to_string() }))).collect::<Vec<_>>().join(" → "),
                    clock.elapsed().as_secs_f64()
                )
            };
            noted(&mut report, said);
            messages.alive();
        }
        // Where nothing is complete yet, the 32 states are a candidate (complete by construction).
        if !found.is_finite() && !settle_first && chain.last().map(|c| c.len()) != Some(STATES) {
            chain.push((0..STATES).collect());
            models.push(None);
            predicted.push(0.);
        }
        // The prefixes from the first one predicted to fit (its misfit, the pilot's less what its states took away alone,
        // below one), each judged on every path, until one is complete; above it, while a nominee's least W stands below the
        // least found (its states more, its least W no less).
        let start = (prefixes..chain.len()).find(|&p| predicted[p] < 1.).unwrap_or(chain.len().saturating_sub(1)).max(prefixes);
        for place in if settle { chain.len()..chain.len() } else { start..chain.len() } {
            if models[place].is_none() {
                models[place] = Some(make(&chain[place])?);
            }
            let least = price(models[place].as_deref().expect("its model"), first, false).0;
            let lower = need0 * least * span + again(place, least);
            if !(lower < found) {
                noted(&mut report, format!("提名的 {} 的最小 W（A = 0、第一级步长 {first:.4} µs）{lower:.3e} 已不低于已找到的最小 W {found:.3e}：再往上不判", named(&chain[place])));
                break;
            }
            let keep = chain[place].clone();
            let w = weigh(place, &keep, &mut models, &mut report)?;
            messages.alive();
            found = found.min(w.work.unwrap_or(f64::INFINITY));
            let open = w.open;
            weighed.push(w);
            // An open fate settled on more paths first (# How many paths).
            if open && more_paths {
                break;
            }
        }
        // A fate open at the paths' statistics (select.rs, # How many paths): the paths doubled, up to the pilot's
        // trajectories, and every candidate judged again on them -- before any is scouted. At all of them, its upper end
        // stands, said.
        // The paths doubled only where it pays (select.rs # How many paths): an open candidate decided complete would save
        // the least W found less its own at its account's lower end; twice the paths cost the reference's and every
        // weighed candidate's work on them again, as much more as the paths are (`paths::Counts::work`). Doubled where some
        // open candidate's saving is more than that; else its upper end stands, said.
        let open: Vec<&Weighed> = weighed.iter().filter(|w| w.open).collect();
        if !open.is_empty() {
            let names = open.iter().map(|w| named(&w.judged.keep)).collect::<Vec<_>>().join("、");
            let next = (2 * count).min(available);
            let spent = paths.work + weighed.iter().map(|w| w.judged.work).sum::<f64>();
            let more = spent * (next - count) as f64 / count as f64;
            // Against the least W found, or the 32 states' -- complete by construction -- where that is less.
            let top = {
                let price = price(&full, tau, false).0;
                need0 * price * span + again(1, price)
            };
            let saving = open.iter().map(|w| found.min(top) - w.settled_low).fold(f64::NEG_INFINITY, crate::linalg::larger);
            if count < available && pinned.is_none() && saving > more {
                let line = format!("{count} 条路径上 {names} 够用与否在路径的统计（t 分布，自由度为窗内路径数减一）下定不下来，判清可省 W {saving:.3e}，多测 {next} 条路径的功 {more:.3e}：路径加倍，候选重判");
                eprintln!("  选态：{line}");
                doubled.push(line);
                count = next;
                continue;
            }
            noted(&mut report, format!("{count} 条路径上 {names} 够用与否定不下来；判清可省 W {saving:.3e}，加倍到 {next} 条的功 {more:.3e}{}：按直接部分的上端记", if count < available { "，不划算" } else { "（已用先导的全部路径）" }));
        }
        // The complete candidates but G at their own step: G's is G's own, and a model with explicit excited states may
        // stand at a longer one (R4: G stands at 0.1185 µs, the 2026-11-21 P20 at 0.474). Each one's scout alone on its own
        // model (`Choosing::Scout`: the first member of every pilot group at its ladder's first rung and the companion, down
        // the ladder to the first rung its own reads stand on, as the attempt's scout, the priced course not taken), its work a µs at the rung the scout stands at, and its W again
        // there with the scout's work beside -- one member of each pilot group at two levels, the companion's at most half.
        // A scout is dear (R4's P26: 87 s), so they are taken in the order of each one's least W -- its work a µs at the
        // ladder's first rung `first`, the longest step its scout may stand at -- while that stands below the least W found
        // (G's at its own step where G is complete, the scouted ones' at theirs); the rest keep their W at G's step, which is
        // no less than their least.
        let scout_of = |price: f64| 1.5 * numbers.len() as f64 * price * span;
        let mut order: Vec<(usize, f64)> = (0..weighed.len())
            .filter(|&i| weighed[i].work.is_some() && weighed[i].place != 0 && weighed[i].place != usize::MAX)
            .map(|i| {
                let w = &weighed[i];
                let least = price(models[w.place].as_deref().expect("a candidate's model"), first, false).0;
                (i, w.need * least * span + again(w.place, least) + scout_of(least))
            })
            .collect();
        order.sort_by(|a, b| a.1.total_cmp(&b.1));
        let mut best = weighed.iter().find(|w| w.place == 0).and_then(|w| w.work).unwrap_or(f64::INFINITY);
        for (i, least) in order {
            if !(least < best) {
                weighed[i].fate += &format!("；第一级步长 {first:.4} µs 上的最小 W {least:.3e} 已不低于最小的 W {best:.3e}，不侦察");
                continue;
            }
            let clock = Instant::now();
            let model = models[weighed[i].place].as_deref().expect("a candidate's model");
            let rung = match solve_with(request, model, Choosing::Scout, messages)? {
                Attempted::Scouted(t) => t,
                _ => unreachable!("a scout's attempt stops at its rung"),
            };
            messages.stage(Stage { step: Step::Select, tau: None, batch: None });
            let price = price(model, rung, false).0;
            let scout = scout_of(price);
            let entry = &mut weighed[i];
            let w = entry.need * price * span + again(entry.place, price) + scout;
            best = crate::linalg::smaller(best, w);
            entry.fate += &format!("；自己的侦察立定于 τ = {rung:.4} µs（先导 {tau:.4}）：每 µs 代价 {:.3e} → {price:.3e}，W {:.3e} → {w:.3e}（含侦察 {scout:.3e}），{:.1} s", entry.price, entry.work.unwrap_or(f64::NAN), clock.elapsed().as_secs_f64());
            (entry.price, entry.work) = (price, Some(w));
            noted(&mut report, format!("{}：{}", named(&entry.judged.keep), entry.fate));
            messages.alive();
        }
        // P: the complete candidate of least W (the first of them where several are) -- one is complete at least, the 32 states by
        // construction; no candidate's W is ever taken back (a scout's replaces it).
        let w = |i: usize| weighed[i].work.unwrap_or(f64::INFINITY);
        let complete = (0..weighed.len()).find(|&i| weighed[i].work.is_some()).expect("a complete candidate: the 32 states are the reference itself");
        let chosen = (0..weighed.len()).filter(|&i| weighed[i].work.is_some()).fold(complete, |best, i| if w(i).total_cmp(&w(best)).then(i.cmp(&best)).is_lt() { i } else { best });
        // The tests' candidates beside the chain (select.rs `ALSO`): judged and weighed for comparison, never chosen.
        #[cfg(test)]
        for keep in crate::select::ALSO.with(|a| a.borrow().clone()) {
            let model = Selected::new(crate::scene::Scene::new(request)?, Selection { keep, ..run.report.clone() }, reference)?;
            let judged = judge(&paths, &model, centre)?;
            let price = price(&model, tau, false).0;
            let model_error = account(&judged);
            let own = looks(&read.estimates, &fixed(model_error.clone()), context.step);
            let (met, need) = (met(&own), most_needed(&own));
            weighed.push(Weighed { place: usize::MAX, judged, price, need, work: None, fate: format!("对照，不参与选（显示量{}、温度{}）", if met[0] { "够" } else { "不够" }, if met[1] { "够" } else { "不够" }), open: false, settled_low: f64::INFINITY });
        }
        // The report: every candidate weighed with its parts at their largest over the times.
        for w in &weighed {
            // The direct part where the paths read it (where they read none, the held atoms' bound there: `model_of`).
            let direct = times.iter().filter_map(|&t| direct_at(&w.judged, &paths.windows, t)).fold(0., crate::linalg::larger);
            // The motion's parts, each its largest over the times (select.rs, # The motion).
            let motion_of = |k: usize| crate::select::motion_at(&w.judged, &paths.windows, &course, k, hm, Reach::Upper);
            let motions: Vec<crate::select::Motion> = (1..times.len()).filter_map(motion_of).collect();
            let most = |of: &dyn Fn(&crate::select::Motion) -> f64| motions.iter().map(of).fold(0., crate::linalg::larger);
            let axes = |of: &dyn Fn(&crate::select::Motion, usize) -> f64| format!("[{:.2e}, {:.2e}, {:.2e}]", most(&|m| of(m, 0)), most(&|m| of(m, 1)), most(&|m| of(m, 2)));
            let motion = if motions.is_empty() {
                String::new()
            } else {
                format!(
                    "；运动：丢失 {:.2e}（x {}）、总 T {:.2} µK（各方向 {} µK，不记账；落点检查的 Q 候选 / 参照 {}）、推力 {} µm/µs²、平均速度 {} m/s",
                    most(&|m| m.loss),
                    w.judged.escape.as_ref().map_or("—".to_string(), |(x, n)| format!("{:+.3} ± {:.3}，{n} 条", x.value, x.error)),
                    most(&|m| m.temperature_total),
                    axes(&|m, a| m.temperature[a]),
                    w.judged.statistics.map_or("—".to_string(), |[c, r]| format!("{c:+.3} / {r:+.3}")),
                    axes(&|m, a| m.push[a]),
                    axes(&|m, a| m.velocity[a])
                )
            };
            noted(
                &mut report,
                format!(
                    "{}{}：直接 {:.2e}（校准 {:.1e}，率 {:.1e}）{motion}；need {:.0}、W {}；{}",
                    named(&w.judged.keep),
                    if w.place == weighed[chosen].place { "（选中）" } else { "" },
                    direct,
                    w.judged.calibration.rows,
                    w.judged.calibration.rate,
                    w.need,
                    w.work.map_or("—".to_string(), |x| format!("{x:.3e}")),
                    w.fate
                ),
            );
        }
        noted(&mut report, format!("选态用时 {:.1} s", clock.elapsed().as_secs_f64()));
        let w = weighed.swap_remove(chosen);
        return Ok(Some(Choice { keep: w.judged.keep.clone(), chain: chain.clone(), judged: w.judged, windows: paths.windows.clone(), memory, length: paths.length, report }));
    }
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

/// The level a rung is read against (module documentation, ## A companion at τ/2): at twice its step, or at half of it
/// where that reads the rung at less work for its precision (`against_half`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum Companion {
    /// 2τ: e(τ) = [Q(2τ) − Q(τ)]/(2^p − 1) (estimate.rs `between`), its halved steps taken back (`read_step`).
    #[default]
    Double,
    /// τ/2: e(τ) = [Q(τ) − Q(τ/2)]·2^p/(2^p − 1), the read of (τ/2, τ) carried to τ (`carried`).
    Half,
}
impl Companion {
    /// The companion's step at the rung τ.
    fn step(self, tau: f64) -> f64 {
        match self {
            Companion::Double => 2. * tau,
            Companion::Half => 0.5 * tau,
        }
    }
}

/// The scout's pairs a rung foresaw its rung below on (`foresee`): their level at the step `step`, its states kept --
/// the rung below's own level's first members, or where the rung below is read against τ/4 its companion's (## A
/// companion at τ/2) --, and the two reads on the same pairs, the finer one's (`half`) and the coarser one's (`rung`).
struct Foreseen {
    step: f64,
    first: Vec<Part>,
    half: Between,
    rung: Between,
}

/// The pilot's two levels (module documentation, # The step): its groups at the step τ and the same groups, in the same
/// order, at the companion's step (`companion`: 2τ, or τ/2 where that reads the rung at less work for its precision, ## A
/// companion at τ/2), `coarse`; the further pairs a check took (`settle`), at the two levels: the groups the product
/// runs next, by their place in its list. The rung's third level, `above`: a read of the pair beside the rung's own
/// carried to τ -- the rung left above it (2τ against 4τ), read on the pilot's pairs where the check came down from it
/// there (`above_pilot`), on the scout's where the scout did, or the pilot's trajectories at the third's step
/// (`third_on_the_pilot`: above the pair or below it, `third_of`) --, its pair's two steps (`above_steps`) and the pairs
/// it was read on (`above_pairs`: the scout's, the pilot's, or with the further pairs); and per side, above and below,
/// whether no third level is to be had there on the pilot's own pairs (`third_none`): its read on them was unread (it
/// excludes nothing on some look), or, above, the rung above stood against τ/2, its own companion at 2τ -- this rung's
/// 4τ -- having taken many steps at their halves.
///
/// And where the rung in hand could not see and the rung below was weighed against more pairs (`foresee`), the scout's
/// pairs there and the two reads on them (`half`).
///
/// Where the third level is read on the pilot's trajectories (`check`: bought, or where it could not see), they are
/// held, `extra` at `extra_step`, and every further pair taken from then on is run there too, `further[2]` -- the last
/// of the further pairs at the rung's levels (those taken before have none) --: the third level is then read again on
/// all of them (`Levels::third`). An extra level below the rung is a rung below's own level, or its companion: its
/// states are kept, and it is not run again where the ladder goes down.
///
/// Where the scout read the rung's third level above on its own pairs (`scouted`, `Scouted`), its trajectories
/// there: the first members of the pilot's at that step, where the pilot reads it there.
#[derive(Default)]
struct Levels {
    fine: Vec<Part>,
    coarse: Vec<Part>,
    companion: Companion,
    extra: Vec<Part>,
    extra_step: f64,
    further: [Vec<Part>; 3],
    above: Option<Between>,
    above_pilot: bool,
    above_steps: [f64; 2],
    above_pairs: u64,
    third_none: [bool; 2],
    half: Option<Foreseen>,
    scouted: Option<Scouted>,
    /// The scout's trajectories at the third level's step of the pilot's groups the extra level does not hold yet
    /// (`third_on_the_pilot`: their first members, `extra_members` of each), kept for the groups it goes on to.
    extra_first: Vec<Part>,
    extra_members: u64,
}

/// The rung's third level above as the scout read it on its own pairs (module documentation, ## The ladder; `check`,
/// `third_on_the_scout`): the rung `tau` it was read for, the level's step, the members of the pilot's groups run there
/// (`members`, the scout's) and those trajectories -- none from where the pilot holds them (`third_on_the_pilot`).
struct Scouted {
    tau: f64,
    step: f64,
    members: u64,
    level: Vec<Part>,
}
impl Scouted {
    /// The trajectories run at its step that the pilot's level there has yet to hold.
    fn ran(&self) -> f64 {
        (self.level.len() as u64 * self.members) as f64
    }
}

impl Levels {
    /// The run level's groups and their companions: the pilot's, then the further pairs'.
    fn paired(&self) -> (Vec<&Part>, Vec<&Part>) {
        (self.fine.iter().chain(&self.further[0]).collect(), self.coarse.iter().chain(&self.further[1]).collect())
    }

    /// The rung's own read on all its pairs at their analysis times among the candidates `times` (`read_step`): against
    /// 2τ, the read between the two levels; against τ/2, the read between τ/2 and τ carried to τ by the order (module
    /// documentation, ## A companion at τ/2).
    fn own(&self, times: &[f64], window: f64, scales: &Scales) -> Between {
        let (fine, coarse) = self.paired();
        match self.companion {
            Companion::Double => read_step(times, window, &fine, &coarse, scales),
            Companion::Half => carried(&read_step(times, window, &coarse, &fine, scales), 2.),
        }
    }

    /// The third level on the trajectories the rung holds at its extra level (`extra`, `further[2]`) against their
    /// neighbours at the rung's own level next to it -- the coarser of the two where the extra level lies above, the finer
    /// where it lies below --, the pair's read carried to the rung's step τ (`carried`: from the pair's finer step), at
    /// the analysis times' candidates `times`; the pair's two steps and the pairs it is read on.
    fn third(&self, tau: f64, times: &[f64], window: f64, scales: &Scales) -> (Between, [f64; 2], u64) {
        let extra: Vec<&Part> = self.extra.iter().chain(&self.further[2]).collect();
        self.third_at(tau, self.extra_step, &extra, self.further[2].len(), GROUP, times, window, scales)
    }

    /// The third level on the trajectories `extra` at the step `step` -- the last `further` of them the further pairs',
    /// their groups of `members` members, as the rung's level beside it holds them -- against the rung's level next to
    /// it (`third`).
    #[allow(clippy::too_many_arguments)]
    fn third_at(&self, tau: f64, step: f64, extra: &[&Part], further: usize, members: u64, times: &[f64], window: f64, scales: &Scales) -> (Between, [f64; 2], u64) {
        let other = self.companion.step(tau);
        let (lower, upper) = if other > tau { ((tau, 0), (other, 1)) } else { ((other, 1), (tau, 0)) };
        // The rung's level beside the extra one: the pilot's groups the extra level holds (`third_on_the_pilot`: all of
        // them, or as many as its read asked) and its further pairs that hold the extra level too.
        let (next, k) = if step > upper.0 { upper } else { lower };
        let start = self.further[k].len() - further;
        let beside: Vec<&Part> = [&self.fine, &self.coarse][k].iter().take(extra.len() - further).chain(&self.further[k][start..]).collect();
        let (finer, coarser, steps) = if step > next { (beside, extra.to_vec(), [next, step]) } else { (extra.to_vec(), beside, [step, next]) };
        let pairs = finer.len() as u64 * members;
        (carried(&read_step(times, window, &finer, &coarser, scales), tau / steps[0]), steps, pairs)
    }

    /// The reads the check of the rung in hand, at the step `tau`, is told and booked by (`checked`, `Checked`): its
    /// own on all its pairs (`own`: against 2τ, or against τ/2 carried to τ); its third level's carried to τ (`carried`)
    /// -- none where it has none --; the third level's pair's steps and the pairs it was read on, and the own read's
    /// pairs.
    ///
    /// Two levels give a difference, and an error only by what is known of the other level. With e(τ) = Q(τ) − Q(0)
    /// a quantity's error at an output:
    /// - against 2τ, d₂ = Q(2τ) − Q(τ) = e(2τ) − e(τ). The order between the two levels, e(2τ) = 2^p e(τ), makes it
    ///   e(τ) = d₂/(2^p − 1), which is what `between` reads. Of d₂ the part 2^p/(2^p − 1) is the companion's own error:
    ///   two levels that are both too coarse agree (d₂ small, e(τ) not), and a companion beyond the steps at which the
    ///   order holds reads e(τ) high by any factor;
    /// - against τ/2, d' = Q(τ) − Q(τ/2) = (1 − 2^{−p}) e(τ) under the order: e(τ) = 2^p d'/(2^p − 1), the read of the
    ///   pair (τ/2, τ) carried to its coarser level (module documentation, ## A companion at τ/2);
    /// - a third pair beside the own one, (2τ, 4τ) above or (τ/2, τ) below (or, against τ/2, (τ, 2τ) and (τ/4, τ/2)):
    ///   under the order its read is of its finer level's error, the rung's carried by the ratio of the steps,
    ///   2^{2p} times in the mean square a halving.
    /// Where the order holds across the levels the reads are one number. Neither is taken alone, nor the lesser of
    /// them: they are of one mean square only where the order holds, which they test against each other (`verdict`),
    /// and the lesser of two noisy reads is biased low. Both are booked as one: the larger (`worse`). The own read
    /// stands on its two levels' analysis times among the candidates `times` (`read_step`).
    fn told(&self, times: &[f64], window: f64, scales: &Scales) -> (Between, Option<Between>, Option<([f64; 2], u64)>, u64) {
        (self.own(times, window, scales), self.above.clone(), self.above.as_ref().map(|_| (self.above_steps, self.above_pairs)), self.paired().0.len() as u64 * GROUP)
    }
}

/// The members `members` of the groups `numbers` at the step `tau`, each trajectory through the plan in the `domain`
/// (a companion's level takes `Domain::companion`): a hand-out of its own (`run_groups`).
#[allow(clippy::too_many_arguments)]
fn level(run: &Selected, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, numbers: &[u64], members: std::ops::Range<u64>, keep: bool, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Part> {
    let hand = Hand { groups: numbers, members, shape: Shape::of(plan, keep), record: &|i| trajectory(run, ensemble, domain, tau, plan, keep, i), work: 1. };
    run_groups(&[hand], 0, messages, progress).pop().expect("the hand-out's one level")
}

/// Some levels of the groups `numbers`, the members `members`, one hand-out (`run_groups`): the steps `steps`, each with
/// whether its states are kept, the dearest trajectories first as foretold at `prices` (`foretold`; with none, the
/// finest level, whose trajectories take the most steps). `progress` takes the first level's groups of `steps` (the run
/// level's); returned in the order of `steps`.
#[allow(clippy::too_many_arguments)]
fn levels_of(run: &Selected, ensemble: &Ensemble, domain: &Domain, plan: &Plan, steps: &[(f64, bool)], numbers: &[u64], members: std::ops::Range<u64>, prices: Option<&dyn Fn(f64) -> f64>, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Vec<Part>> {
    let records: Vec<_> = steps.iter().map(|&(step, keep)| move |i| trajectory(run, ensemble, domain, step, plan, keep, i)).collect();
    let hands: Vec<Hand> = steps.iter().zip(&records).map(|(&(step, keep), record)| Hand { groups: numbers, members: members.clone(), shape: Shape::of(plan, keep), record, work: foretold(prices, step) }).collect();
    run_groups(&hands, 0, messages, progress)
}

/// The members `members` of the groups `numbers` at the step `tau` (their states kept) and at its companion's step
/// (2τ, or τ/2: its states kept too, the rung below's own level), the two levels of a check (module documentation,
/// # The step): one hand-out (`levels_of`) -- a check runs both whatever either shows --, the dearer level first as
/// foretold at `prices` (`foretold`). `progress` takes the level at τ's groups.
#[allow(clippy::too_many_arguments)]
fn pair(run: &Selected, ensemble: &Ensemble, domain: &Domain, plan: &Plan, tau: f64, companion: Companion, numbers: &[u64], members: std::ops::Range<u64>, prices: Option<&dyn Fn(f64) -> f64>, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> [Vec<Part>; 2] {
    let mut levels = levels_of(run, ensemble, domain, plan, &[(tau, true), (companion.step(tau), companion == Companion::Half)], numbers, members, prices, messages, progress).into_iter();
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
/// - the rung is read against 2τ, its read taking 2τ's halved steps into it (`read_step`), or, on a priced pass, against
///   τ/2 where that costs the less for the read's precision (`against_half`, `halve_companion`; module documentation,
///   ## A companion at τ/2) -- none beyond the steps' cap;
/// - a rung whose own read lies clearly beyond the square of the step's share among all the looks (`screen`) is left:
///   the step is halved -- the level at τ becomes the companion, and the members so far run anew at τ/2 alone (or, the
///   rung read against τ/2, that level is the rung below's own), its third level above the rung left's own read
///   (`carried`; none where it was read against τ/2: that read is the rung below's own). The finer level keeps its
///   checkpoints (the model is read at its states, `parts_at`);
/// - else, on the scout (which has the pilot to come), the rung stands on its reads, and takes its course by the run's
///   work from here (`course`): it stands, or goes down where a rung below is expected to cost the run less -- where it
///   would stand with its third level above as foretold, that level read on the scout's own pairs first and the paths
///   weighed again on its read (`third_on_the_scout`, `Scouted`; module documentation, ## The ladder). On the
///   pilot (all the members of a group in: the last word on the step), every estimate as it reads (`verdict`):
///   - its own estimate within and its third level's too: its course (`course`): it stands, or goes down where a rung
///     below costs less; its third level's clearly beyond (the order is not shown above it): it is left; its third
///     level's beyond and not clearly: as its own read's (`Verdict::Unseeing`, below);
///   - its own estimate within and no third level, where `above` says the rung above is a rung of the ladder, the rung
///     is read against 2τ and the rung above was not read unread: its course -- the rung below where it costs less;
///     else the pilot's trajectories at 4τ are its third level where the rung buys them (`buys`: their price no more
///     than what the least rung below adds), and where they read no number (`readable`) the rung above is unread; else
///     it stands on its own two levels;
///   - its own estimate within and no third level to be had: its course, standing on its own two levels or going down;
///   - its own estimate, or its third level's, beyond the share and not clearly (`Verdict::Unseeing`): the pairs of
///     `remedy` -- the third level on the pilot's trajectories at 4τ where it is not read there (`upper_on_the_pilot`),
///     the product's next groups at the rung's levels (τ and its companion's, and 4τ where it holds that level), read
///     again after every batch -- where they are expected to cost the run less than the rung below; else, or where they
///     cannot see it, it is left: its estimate as it reads is beyond the share.
/// A rung below at which a trajectory would take more than `MOST_STEPS` steps does not exist: a check that is clearly
/// beyond there leaves the step capped, one whose noise no pairs see through ends unable to see (`Settled::Unseen`).
///
/// On the scout and on the pilot alike the check stops where the rung's own read is in and not clearly beyond
/// (`Settled::Pending`: `check` takes it up again, priced, once the states it is priced at are read).
#[allow(clippy::too_many_arguments)]
fn settle(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: std::ops::Range<u64>, above: &dyn Fn(f64) -> bool, prices: Option<&dyn Fn(f64) -> f64>, messages: &mut Messages) -> Settled {
    let [fine, coarse] = pair(run, ensemble, domain, &context.plan, context.tau, levels.companion, numbers, members.clone(), prices, messages, &mut |prefix, messages| so_far(context, &levels.fine, prefix, messages));
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
fn check(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &mut Context, levels: &mut Levels, numbers: &[u64], members: u64, above: &dyn Fn(f64) -> bool, prices: Option<&Prices>, messages: &mut Messages) -> Settled {
    // Whether the check is the pilot's: all the members of a group in.
    let pilot = members == GROUP;
    loop {
        // The rung below, and whether the ladder has it.
        let below = 0.5 * context.tau;
        let lowest = context.plan.end / below > MOST_STEPS as f64;
        // The rung's companion (## A companion at τ/2): 2τ, its read taking its halved steps into it (`read_step`), or
        // τ/2 where that reads the rung at less work for its precision (`against_half`) -- at the work as priced, so on
        // a priced pass, before the read decides anything (`undecided`) --, and wherever the read against 2τ is without
        // bound, whatever the price. None beyond the steps' cap, where the read against 2τ is the one there is.
        let mut undecided = false;
        if levels.companion == Companion::Double && !lowest {
            // The two reads' noise foretold by the read against 2τ's own, its factor undone (`raw_pairs`).
            let (double, half, raw) = {
                let (own, companion) = levels.paired();
                let double = amplified(companion.iter().copied());
                (double, amplified(own.iter().copied()), raw_pairs(&levels.own(&context.plan.reads, context.plan.window, &context.scales), own.len() as f64 * GROUP as f64, double))
            };
            let (pilot_pairs, held) = ((numbers.len() as u64 * GROUP) as f64, [(numbers.len() as u64 * members) as f64, levels.half.as_ref().filter(|f| f.step == below).map_or(0., |_| numbers.len() as f64)]);
            if prices.map_or(!(double < f64::INFINITY), |prices| against_half(prices, context.tau, double, half, raw, pilot_pairs, held)) {
                halve_companion(run, ensemble, domain, context, levels, numbers, members, messages);
            } else {
                undecided = prices.is_none() && double > 1.;
            }
        }
        // The rung's own read, at its two levels' analysis times.
        let (times, window) = (&context.plan.reads, context.plan.window);
        let read = levels.own(times, window, &context.scales);
        // The sides a third level on the pilot's own pairs can be had at (`third_of`): above -- 4τ, or 2τ where the
        // rung is read against τ/2 -- where the ladder has that rung; below -- τ/2, or τ/4 -- within the steps' cap.
        let tau = context.tau;
        let (up, down) = match levels.companion {
            Companion::Double => (above(tau), !lowest),
            Companion::Half => (above(0.5 * tau), context.plan.end / (0.25 * tau) <= MOST_STEPS as f64),
        };
        let sides = [up && !levels.third_none[0], down && !levels.third_none[1]];
        let upward = !levels.above_pilot && sides.iter().any(|&side| side);
        let verdict = verdict(&read, levels.above.as_ref(), pilot, upward);
        crate::prof::note(|| {
            let looks: Vec<String> = judged(&read).iter().map(|(r, b)| format!("{:.3e}±{:.2e}/{:.3e}", r.value, r.spread.error, b)).collect();
            let above: Vec<String> = levels.above.as_ref().map_or(vec![], |a| judged(a).iter().map(|(r, b)| format!("{:.3e}±{:.2e}/{:.3e}", r.value, r.spread.error, b)).collect());
            let halved = |level: &[Part]| level.iter().fold((0, 0.), |(n, f), p| (n + p.counts.halved.steps, f + p.counts.halved.time / level.iter().map(|q| q.counts.time).sum::<f64>()));
            format!("ladder tau {} pilot {pilot} members {members} groups {} companion {:?} verdict {verdict:?} screen {} halved (steps, share of time) {:?} / {:?} factor {:.5} looks [{}] above [{}]", context.tau, numbers.len(), levels.companion, screen(&read), halved(&levels.fine), halved(&levels.coarse), { let (fine, coarse) = levels.paired(); amplified(if levels.companion == Companion::Double { coarse } else { fine }) }, looks.join(" "), above.join(" "))
        });
        // A read of which no look reads decides nothing and prices nothing (`Verdict::Unread`): the check ends unable to
        // see where it is.
        if verdict == Verdict::Unread {
            return Settled::Unseen;
        }
        // A rung its reads do not leave takes its course by the run's work from here: without its prices the check
        // stops there (`Settled::Pending`) -- and so it does, whatever its read, where its companion is still to be
        // chosen on the prices: a companion at 2τ that took steps at their halves may be no comparison of the steps
        // at all (R4 at 0.474 µs, 0.948 µs's time 44 % on halved steps: one row read 7.1 ± 1.8 of the share's square
        // against 2τ taken back, 0.06 ± 0.16 against τ/2 on the same 44 pairs; motion3d_status/rewrite/
        // r2ladder.status.md).
        if prices.is_none() && (undecided || !screen(&read)) {
            return Settled::Pending;
        }
        // What the run holds of the rung's levels (`Holding`): beside the members at τ and at the companion's step,
        // the scout's pairs `foresee` ran at τ/2 or τ/4, the companion's members where it is τ/2; and the factors a read
        // takes for the levels at τ and τ/2 as its coarser level, of their halved steps on the pilot's groups -- what the
        // rungs below are read against (`against_half`).
        let groups = numbers.len() as u64;
        let ran_at = |step: f64| levels.half.as_ref().filter(|f| f.step == step);
        let halved_companion = levels.companion == Companion::Half;
        let holding = Holding {
            pilot: (groups * GROUP) as f64,
            members: (groups * members) as f64,
            pairs: (groups * members + levels.further[0].len() as u64 * GROUP) as f64,
            companion: levels.companion,
            below: [if halved_companion { (groups * members) as f64 } else { ran_at(below).map_or(0., |_| groups as f64) }, ran_at(0.5 * below).map_or(0., |_| groups as f64)],
            amplified: [amplified(&levels.fine), if halved_companion { amplified(&levels.coarse) } else { ran_at(below).map_or_else(|| prices.map_or(1., |prices| (prices.halving)(below)), |f| amplified(&f.first)) }],
            // The rung's own read's noise, its lift and factor undone: what every read's precision is priced by.
            raw: raw_pairs(&read, (groups * members + levels.further[0].len() as u64 * GROUP) as f64, if halved_companion { 2_f64.powi(2 * ORDER) * amplified(levels.paired().0.iter().copied()) } else { amplified(levels.paired().1.iter().copied()) }),
            factor: amplified(if halved_companion { levels.paired().0 } else { levels.paired().1 }),
        };
        // The side above as the scout read it on its own pairs (`Scouted`), where that read is the rung's third level
        // still -- not yet read on the pilot's trajectories --: its pairs measured (`Third::measured`) -- those its read
        // asks at the best (`blind`), and without bound where it stands out of the order's range of the rung's own read
        // (`in_range`) --, the trajectories run there held; none to read where it reads within.
        let scouted = levels.scouted.as_ref().filter(|scouted| scouted.tau == tau && !levels.above_pilot).zip(levels.above.as_ref());
        let scouted_up = scouted.and_then(|(scouted, above)| {
            let asks = if in_range(&read, above) { blind(above).map(|factor| factor * levels.above_pairs as f64) } else { Some(f64::INFINITY) };
            asks.map(|pairs| Third { step: scouted.step, held: scouted.ran(), factor: 1., beside: levels.above_steps[0], lift: 1., measured: Some(pairs) })
        });
        // The third level the rung would read on the pilot's pairs, where it has none yet: the side of the less cost for
        // its read's precision (`third_of`). Where the scout read it above, the side of the less cost by that read --
        // the side below as foretold where it is cheaper --, and the side above as read where neither is to be had: the
        // path at the rung costs what that read asks, without bound where no pairs make it the rung's.
        let third = match (prices, scouted) {
            (Some(prices), Some(_)) => scouted_up.map(|up| third_of(prices, tau, holding, sides, Some(up)).unwrap_or(up)),
            (prices, _) => prices.and_then(|prices| third_of(prices, tau, holding, sides, None)).filter(|_| upward && levels.above.is_none()),
        };
        // On the scout, where its rung stands and the path at it is the cheaper with its third level above as foretold
        // (module documentation, ## The ladder): that level on the scout's own pairs first (`third_on_the_scout`), and
        // the paths weighed again on its read.
        if let (false, Verdict::Stands, Some(prices), Some(up)) = (pilot, verdict, prices, third) {
            if up.step > tau && up.measured.is_none() && levels.scouted.as_ref().is_none_or(|scouted| scouted.tau != tau) && course(prices, tau, context.plan.end, holding, third) != Course::Below {
                third_on_the_scout(run, ensemble, domain, context, levels, numbers, up.step, members, messages);
                continue;
            }
        }
        if let (Verdict::Stands | Verdict::Above, Some(prices)) = (verdict, prices) {
            crate::prof::note(|| {
                let w: Vec<String> = [0.25, 0.5, 1., 2., 4.].iter().map(|f| format!("w({:.4})={:.4e}", f * context.tau, (prices.work)(f * context.tau))).collect();
                let cost = third.map(|third| (third.step, third.measured, third.cost(prices, holding.pilot, holding.raw)));
                format!("ladder tau {} prices product {:.1} {:?} staying {:.4e} descending {:.4e} third (step, measured pairs, cost) {cost:?} course {:?} [{}]", context.tau, prices.product, holding, staying(prices, context.tau, holding), descending(prices, context.tau, context.plan.end, holding), course(prices, context.tau, context.plan.end, holding, third), w.join(" "))
            });
            match course(prices, context.tau, context.plan.end, holding, third) {
                Course::Third if pilot => {
                    let third = third.expect("a third bought");
                    third_on_the_pilot(run, ensemble, domain, context, levels, numbers, third.step, groups_for(third.pairs(holding.pilot, holding.raw), numbers.len()), messages);
                    continue;
                }
                // On the scout the third level is the pilot's to read.
                Course::Stand | Course::Third => return Settled::Standing,
                Course::Below => {}
            }
        }
        // The own read's estimate, or its third level's, beyond the share and not clearly (`Verdict::Unseeing`, the
        // pilot's alone: its check has its prices here): the noise hides the difference, or the error lies near the
        // share. The pairs that could bring it within at the best, where they are expected to cost the run less than the
        // rung below -- its error the order's less, its noise as the scout's pairs read it there (`foresee`, `foreseen`,
        // `remedy`; the own read within, the rung below is foreseen within on the pilot's pairs, none of its level run);
        // the rung below is weighed where both are to be had. Where the rung below is cheaper, or pairs cannot see, the
        // rung is left, as an adaptive integrator leaves a step whose error estimate is beyond its tolerance.
        let have = holding.pairs as u64;
        // A third level above the rung held on some of the pilot's groups whose read cannot see: on more of them first,
        // as many as it asks and twice as many at the least (every group at the most), as further pairs are taken for an
        // own read -- the extra level alone runs.
        if let (Verdict::Unseeing, Some(above)) = (verdict, levels.above.as_ref().filter(|above| !within(above) && levels.above_pilot)) {
            if levels.extra_step > tau && !levels.extra.is_empty() && levels.extra.len() < numbers.len() {
                let asks = seeing(above, levels.above_pairs).unwrap_or(f64::INFINITY);
                let groups = groups_for(asks, numbers.len()).max(2 * levels.extra.len()).min(numbers.len());
                third_on_the_pilot(run, ensemble, domain, context, levels, numbers, levels.extra_step, groups, messages);
                continue;
            }
        }
        let remedied = match (verdict, prices) {
            (Verdict::Unseeing, Some(prices)) => {
                let own_within = within(&read);
                // The rung below's companion: the level at τ, the pilot's; or τ/4 where that reads it at less work for its
                // precision (`against_half`), the scout's pairs there where `foresee` ran them.
                let quarter = against_half(prices, below, holding.amplified[0], holding.amplified[1], holding.raw, holding.pilot, [holding.pilot, holding.below[1]]);
                let companion_below = |held: u64| if quarter { (0.5 * below, held) } else { (context.tau, groups * GROUP) };
                let below = match levels.companion {
                    _ if lowest => None,
                    // Read against τ/2, its rung below against τ: the rung below's read is this one uncarried, on the same
                    // pairs -- its error and its noise the order's less, exactly (`carried`).
                    Companion::Half if !quarter => Some(Below { asks: foreseen_below(&read), ran: groups * members, companion: companion_below(0) }),
                    companion => {
                        // The scout's pairs at the pair's finer level's half, and at its two levels again: τ/2, τ and 2τ
                        // against 2τ; τ/4, τ/2 and τ against τ/2, where the rung below is read against τ/4.
                        let lower = if companion == Companion::Half { below } else { context.tau };
                        if !own_within && seeing(&read, have).is_some() && levels.half.is_none() {
                            levels.half = Some(foresee(run, ensemble, domain, context, levels, numbers, lower, prices.work, messages));
                        }
                        let ran = if companion == Companion::Half { groups * members } else { 0 };
                        match &levels.half {
                            Some(f) => Some(Below { asks: foreseen(&read, &f.half, &f.rung), ran: if companion == Companion::Half { ran } else { groups }, companion: companion_below(if f.step == 0.5 * below { groups } else { 0 }) }),
                            None => own_within.then_some(Below { asks: None, ran, companion: companion_below(0) }),
                        }
                    }
                };
                // The third level held on the pilot's trajectories, or the side it would be read at there.
                let held = !levels.extra.is_empty();
                let reading = third_of(prices, tau, holding, if levels.above_pilot { [false; 2] } else { sides }, scouted_up);
                let upper = levels.above.as_ref().map(|read| Upper { read, pairs: levels.above_pairs, held, buys: !held && reading.is_some(), ran: if held { 0 } else { levels.scouted.as_ref().filter(|scouted| scouted.tau == tau && reading.is_some_and(|third| third.step == scouted.step)).map_or(0, |scouted| scouted.ran() as u64) }, step: if held { levels.extra_step } else { reading.map_or(4. * tau, |third| third.step) } });
                let remedied = remedy(&read, upper, holding, below, prices, context.tau, context.plan.end, third);
                crate::prof::note(|| format!("ladder tau {} remedy {remedied:?} have {have} own within {own_within} upper {:?}", context.tau, upper.map(|u| (u.pairs, u.held, u.buys, within(u.read), seeing(u.read, u.pairs)))));
                remedied
            }
            _ => None,
        };
        match remedied {
            Some(Remedy::Upper) => {
                let third = third_of(prices.expect("a remedy is priced"), tau, holding, sides, scouted_up).expect("a third to read");
                let asks = levels.above.as_ref().and_then(|above| seeing(above, levels.above_pairs)).unwrap_or(f64::INFINITY);
                let groups = groups_for(crate::linalg::larger(asks, (THIRD_GROUPS * GROUP) as f64), numbers.len());
                third_on_the_pilot(run, ensemble, domain, context, levels, numbers, third.step, if third.step > tau { groups } else { numbers.len() }, messages);
                continue;
            }
            Some(Remedy::Pairs(pairs)) => {
                // Further pairs at the rung's every level: τ and its companion's step, and its third level's where it
                // holds that level.
                let next: Vec<u64> = (have / GROUP..pairs / GROUP).collect();
                let upper = !levels.extra.is_empty();
                // The third level held on some of the pilot's groups: on all of them first (further pairs follow the
                // pilot's at every level).
                if upper && levels.extra.len() < numbers.len() {
                    third_on_the_pilot(run, ensemble, domain, context, levels, numbers, levels.extra_step, numbers.len(), messages);
                }
                let mut steps = vec![(tau, false), (levels.companion.step(tau), false)];
                if upper {
                    steps.push((levels.extra_step, false));
                }
                let more = levels_of(run, ensemble, domain, &context.plan, &steps, &next, 0..GROUP, prices.map(|p| p.work), messages, &mut |prefix, messages| {
                    so_far(context, &levels.fine, &levels.further[0].iter().chain(prefix.iter().copied()).collect::<Vec<_>>(), messages)
                });
                levels.further.iter_mut().zip(more).for_each(|(further, more)| further.extend(more));
                if upper {
                    let (read, _, pairs) = levels.third(tau, &context.plan.reads, context.plan.window, &context.scales);
                    (levels.above, levels.above_pairs) = (Some(read), pairs);
                }
                continue;
            }
            None => {}
        }
        // The rung in hand is left: its reads leave it, or a rung below is expected to cost the run less (`course`). At
        // the ladder's bottom -- where no rung below is cheaper, there being none -- a read clearly beyond leaves the step
        // capped, one that only could not be seen through, unable to see.
        if lowest {
            return if verdict == Verdict::Unseeing { Settled::Unseen } else { Settled::Capped };
        }
        // The rung below's third level above: this rung's own read against 2τ, lowered. Read against τ/2, it is the
        // rung below's own read, no third level of it -- and the rung below has none above to buy either, the level at
        // 2τ having taken steps at their halves.
        let halved_companion = levels.companion == Companion::Half;
        let inherits = (!halved_companion).then(|| carried(&read, 0.5));
        context.tau = below;
        messages.rung(below);
        // The level at the step is the companion from here on: its states are read no more; its own companion (where
        // it is 2τ) and the further pairs are no level of the rung below's -- all freed before its level runs, which
        // holds as much again.
        forget(&mut levels.fine);
        // A third level held below the rung -- the pilot's trajectories at τ/2, or at τ/4 against τ/2 -- is the rung
        // below's own level or its companion's, its states kept: kept for it, run no more.
        let extra = (levels.extra_step < tau).then(|| std::mem::take(&mut levels.extra)).filter(|extra| !extra.is_empty());
        (levels.extra, levels.further, levels.scouted, levels.extra_first) = Default::default();
        if halved_companion {
            // The companion at τ/2 is the rung below's own level, its states kept. A foresight at τ/4 is its companion's
            // first members, where it is read against τ/4 (`halve_companion`), and a third held at τ/4 its whole.
            levels.half = levels.half.take().filter(|f| f.step == 0.5 * below);
            std::mem::swap(&mut levels.fine, &mut levels.coarse);
            if let Some(extra) = extra {
                (levels.extra, levels.extra_step) = (extra, 0.5 * below);
            }
        } else if let Some(extra) = extra {
            // The third held at τ/2 is the rung below's own level, every member of it.
            levels.half = None;
            levels.coarse = std::mem::replace(&mut levels.fine, extra);
        } else {
            // Its own level: where the rung below was foreseen on the scout's pairs, those are its first members, run.
            let finer = match levels.half.take().filter(|f| f.step == below) {
                Some(Foreseen { mut first, .. }) => {
                    let rest = level(run, ensemble, domain, &context.plan, below, numbers, 1..members, true, messages, &mut |prefix, messages| so_far(context, &first, prefix, messages));
                    absorb(&mut first, rest);
                    first
                }
                None => level(run, ensemble, domain, &context.plan, below, numbers, 0..members, true, messages, &mut |prefix, messages| so_far(context, &[], prefix, messages)),
            };
            levels.coarse = std::mem::replace(&mut levels.fine, finer);
        }
        levels.companion = Companion::Double;
        (levels.above, levels.above_pilot, levels.above_steps, levels.above_pairs, levels.third_none) = (inherits, pilot, [tau, 2. * tau], have, [halved_companion, false]);
    }
}

/// The rung below of a rung read against τ/2 whose rung below is read against τ, as `remedy` weighs it (`Below::asks`):
/// its read is the rung's own uncarried, on the same pairs -- its error and its noise lowered by the order exactly --,
/// so its foreseen ends are its own (`foreseen` at the noise's ratio 2^{−2p}, the read lowered against itself).
fn foreseen_below(read: &Between) -> Option<f64> {
    foreseen(read, &carried(read, 0.5), read)
}

/// The rung's companion at τ/2 in place of 2τ, where that reads the rung at less work for its precision (`check`,
/// `against_half`; module documentation, ## A companion at τ/2): the pilot's groups `numbers`, their members
/// `..members`, at τ/2 -- the first of them those `foresee` ran there, where it did -- with their states kept (the rung
/// below's own level, where the ladder goes down), and the further pairs' groups (the product's next ones, by their
/// place in its list) at τ/2. The level at 2τ is freed, and no third level above is read on from here (`coarser`): the
/// level at 4τ took more steps at their halves. What the reads against 2τ say beside the read against τ/2 on the same
/// pairs -- as it is and with its halved steps taken into it (`read_as_it_is`, `read_step`), per look its value and
/// standard error, with the share of the companion's time on halved steps and the read's factor -- is noted
/// (`prof::note`): a record for the data to judge by, nothing decided on it.
#[allow(clippy::too_many_arguments)]
fn halve_companion(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &Context, levels: &mut Levels, numbers: &[u64], members: u64, messages: &mut Messages) {
    let (tau, plan, scales) = (context.tau, &context.plan, &context.scales);
    let (hard, corrected, factor, share) = {
        let (fine, coarse) = levels.paired();
        let (halved, time) = coarse.iter().fold((0., 0.), |(h, t), p| (h + p.counts.halved.time, t + p.counts.time));
        (read_as_it_is(&plan.reads, plan.window, &fine, &coarse, scales), levels.own(&plan.reads, plan.window, scales), amplified(coarse.iter().copied()), halved / time)
    };
    let first = levels.half.take().filter(|f| f.step == 0.5 * tau);
    // A third level held at τ/2 on the pilot's trajectories is the companion's whole, its read the rung's own from here.
    let held = (levels.extra_step == 0.5 * tau && !levels.extra.is_empty()).then(|| std::mem::take(&mut levels.extra));
    if held.is_some() && levels.above_steps == [0.5 * tau, tau] {
        (levels.above, levels.above_pilot) = (None, false);
    }
    let halves = match (held, first) {
        (Some(held), _) => held,
        (None, Some(Foreseen { mut first, .. })) => {
            if members > 1 {
                let rest = level(run, ensemble, domain, plan, 0.5 * tau, numbers, 1..members, true, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages));
                absorb(&mut first, rest);
            }
            first
        }
        (None, None) => level(run, ensemble, domain, plan, 0.5 * tau, numbers, 0..members, true, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages)),
    };
    let further: Vec<u64> = (numbers.len() as u64..(numbers.len() + levels.further[0].len()) as u64).collect();
    let further_halves = if further.is_empty() { vec![] } else { level(run, ensemble, domain, plan, 0.5 * tau, &further, 0..GROUP, false, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages)) };
    (levels.coarse, levels.further[1], levels.companion) = (halves, further_halves, Companion::Half);
    (levels.extra, levels.further[2]) = Default::default();
    crate::prof::note(|| {
        let half = levels.own(&plan.reads, plan.window, scales);
        let looks: Vec<String> = judged(&hard)
            .iter()
            .zip(judged(&corrected))
            .zip(judged(&half))
            .map(|(((h, b), (c, _)), (s, _))| format!("{:.3e}±{:.2e} {:.3e}±{:.2e} {:.3e}±{:.2e} /{:.3e}", h.value, h.spread.error, c.value, c.spread.error, s.value, s.spread.error, b))
            .collect();
        format!("ladder tau {tau} companion halved: 2τ's time on halved steps {share:.4e}, its read's factor {factor:.5}; against 2τ as it is, corrected, against τ/2 [{}]", looks.join(" | "))
    });
}

/// The third level on the pilot's trajectories (`check`: bought, `Course::Third`, or where the third level the rung has
/// could not see, `Remedy::Upper`): the first `groups` of the pilot's groups `numbers` at the third's step `step`
/// (`third_of`: above the rung's pair or below it; below it every group, its level a rung below's own or its
/// companion's), held (`Levels::extra`; their states kept below the rung), read against the rung's level beside it on
/// the same groups (`Levels::third`) -- the first members those the scout ran there, where it read the level on its
/// own pairs (`Scouted`); a level held on fewer groups goes on to the more. Where its read excludes nothing on some
/// look (`readable`: no output counts a trajectory at both levels), that side is unread.
#[allow(clippy::too_many_arguments)]
fn third_on_the_pilot(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &Context, levels: &mut Levels, numbers: &[u64], step: f64, groups: usize, messages: &mut Messages) {
    let tau = context.tau;
    let groups = if step < tau { numbers.len() } else { groups.clamp(1, numbers.len()) };
    if !(levels.extra_step == step && !levels.extra.is_empty()) {
        let first = levels.scouted.as_mut().filter(|scouted| scouted.tau == tau && scouted.step == step && !scouted.level.is_empty()).map(|scouted| (std::mem::take(&mut scouted.level), scouted.members));
        levels.extra = match first {
            Some((mut first, members)) => {
                // The scout's members of the groups beyond kept for them (`Levels::extra_first`).
                (levels.extra_first, levels.extra_members) = (first.split_off(groups.min(first.len())), members);
                let rest = level(run, ensemble, domain, &context.plan, step, &numbers[..groups], members..GROUP, step < tau, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages));
                absorb(&mut first, rest);
                first
            }
            None => {
                levels.extra_first.clear();
                level(run, ensemble, domain, &context.plan, step, &numbers[..groups], 0..GROUP, step < tau, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages))
            }
        };
        (levels.extra_step, levels.further[2]) = (step, vec![]);
    } else {
        further_groups(run, ensemble, domain, context, levels, numbers, step, groups, messages);
    }
    let (mut read, mut steps, mut pairs) = levels.third(tau, &context.plan.reads, context.plan.window, &context.scales);
    // A read on some of the pilot's groups that excludes nothing on some look: on all of them before the side is unread.
    if !readable(&read) && levels.extra.len() < numbers.len() {
        further_groups(run, ensemble, domain, context, levels, numbers, step, numbers.len(), messages);
        (read, steps, pairs) = levels.third(tau, &context.plan.reads, context.plan.window, &context.scales);
    }
    crate::prof::note(|| {
        let (halved, time) = levels.extra.iter().fold((0., 0.), |(h, t), p| (h + p.counts.halved.time, t + p.counts.time));
        let looks: Vec<String> = judged(&read).iter().map(|(r, b)| format!("{:.3e}±{:.2e}/{:.3e}", r.value, r.spread.error, b)).collect();
        format!("ladder tau {tau} third level at {step}: its time on halved steps {:.4e}, the pair's coarser factor {:.5}; read [{}]", halved / time, amplified(if step > tau { &levels.extra } else if levels.companion == Companion::Double { &levels.fine } else { &levels.coarse }), looks.join(" "))
    });
    if readable(&read) {
        (levels.above, levels.above_pilot, levels.above_steps, levels.above_pairs) = (Some(read), true, steps, pairs);
    } else {
        levels.third_none[usize::from(step < tau)] = true;
    }
}

/// The extra level held on the pilot's first groups (`third_on_the_pilot`) on to its first `groups`: the scout's
/// trajectories there (`Levels::extra_first`) with their groups' other members, then whole groups, one hand-out.
#[allow(clippy::too_many_arguments)]
fn further_groups(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &Context, levels: &mut Levels, numbers: &[u64], step: f64, groups: usize, messages: &mut Messages) {
    let held = levels.extra.len();
    if !(held < groups) {
        return;
    }
    let tau = context.tau;
    let took = (groups - held).min(levels.extra_first.len());
    let mut first: Vec<Part> = levels.extra_first.drain(..took).collect();
    let records = [|i| trajectory(run, ensemble, domain, step, &context.plan, step < tau, i)];
    let hands = [
        Hand { groups: &numbers[held..held + took], members: levels.extra_members..GROUP, shape: Shape::of(&context.plan, step < tau), record: &records[0], work: 1. },
        Hand { groups: &numbers[held + took..groups], members: 0..GROUP, shape: Shape::of(&context.plan, step < tau), record: &records[0], work: 1. },
    ];
    let mut made = run_groups(&hands, 0, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages)).into_iter();
    let (rest, whole) = (made.next().expect("the scout's groups' other members"), made.next().expect("the whole groups"));
    absorb(&mut first, rest);
    levels.extra.extend(first);
    levels.extra.extend(whole);
}

/// The rung's third level above on the scout's own pairs (`check`; module documentation, ## The ladder): the members
/// `..members` of the pilot's groups `numbers` -- the scout's -- at the third's step `step`, read against the rung's
/// level beside it on the same pairs (`Levels::third_at`), kept (`Scouted`: the pilot's first members there, where it
/// reads the level). Its read is the rung's third level (`above`, on the scout's pairs) where it reads (`readable`);
/// where it does not, eleven pairs show nothing of it, and the level is foretold as before. What it reads beside the
/// rung's own read -- per look its value and standard error, with the level's time on halved steps and its factor --
/// is noted (`prof::note`).
#[allow(clippy::too_many_arguments)]
fn third_on_the_scout(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &Context, levels: &mut Levels, numbers: &[u64], step: f64, members: u64, messages: &mut Messages) {
    let (tau, plan, scales) = (context.tau, &context.plan, &context.scales);
    let level = level(run, ensemble, domain, plan, step, numbers, 0..members, step < tau, messages, &mut |_, messages| so_far(context, &levels.fine, &[], messages));
    let (read, steps, pairs) = levels.third_at(tau, step, &level.iter().collect::<Vec<_>>(), 0, members, &plan.reads, plan.window, scales);
    crate::prof::note(|| {
        let (halved, time) = level.iter().fold((0., 0.), |(h, t), p| (h + p.counts.halved.time, t + p.counts.time));
        let own = levels.own(&plan.reads, plan.window, scales);
        let looks: Vec<String> = judged(&read).iter().zip(judged(&own)).map(|((r, b), (o, _))| format!("{:.3e}±{:.2e} own {:.3e}±{:.2e} /{:.3e}", r.value, r.spread.error, o.value, o.spread.error, b)).collect();
        format!("ladder tau {tau} scout's third level at {step} on {pairs} pairs: its time on halved steps {:.4e}, its factor {:.5}; in the order's range {}, asks {:?}; read [{}]", halved / time, amplified(&level), in_range(&own, &read), blind(&read), looks.join(" | "))
    });
    if readable(&read) {
        (levels.above, levels.above_pilot, levels.above_steps, levels.above_pairs) = (Some(read), false, steps, pairs);
    }
    levels.scouted = Some(Scouted { tau, step, members, level });
}

/// The scout's pairs -- the first member of every pilot group `numbers` -- at half the pair's finer step `lower` (the rung
/// τ against 2τ; τ/2 for a rung against τ/2 whose rung below is read against
/// τ/4: module documentation, ## A companion at τ/2), their states kept (they are the rung below's own level's first
/// members, or its companion's, where the check goes down, `check`), and at the pair's two levels again, `lower` and
/// twice it: one hand-out (`run_groups`), the longer trajectories first. A group's sums hold its members together
/// (`Part::absorb`), so the scout's pairs at the pair's levels are run anew -- the same trajectories to the bit, a
/// trajectory is a function of its number and step (`trajectory`) -- for the two reads on the same pairs, the finer
/// pair's and the pair's own, whose noise's ratio foresees the rung below (`foreseen`). Their cost is the scout's pairs
/// at three steps, a few dozen of the rung's trajectories against the hundreds of pairs it weighs.
#[allow(clippy::too_many_arguments)]
fn foresee(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &Context, levels: &Levels, numbers: &[u64], lower: f64, prices: &dyn Fn(f64) -> f64, messages: &mut Messages) -> Foreseen {
    let plan = &context.plan;
    let half = |i| trajectory(run, ensemble, domain, 0.5 * lower, plan, true, i);
    let own = |i| trajectory(run, ensemble, domain, lower, plan, false, i);
    let double = |i| trajectory(run, ensemble, domain, 2. * lower, plan, false, i);
    let work = |step: f64| foretold(Some(prices), step);
    let hands = [
        Hand { groups: numbers, members: 0..1, shape: Shape::of(plan, true), record: &half, work: work(0.5 * lower) },
        Hand { groups: numbers, members: 0..1, shape: Shape::of(plan, false), record: &own, work: work(lower) },
        Hand { groups: numbers, members: 0..1, shape: Shape::of(plan, false), record: &double, work: work(2. * lower) },
    ];
    let further: Vec<&Part> = levels.further[0].iter().collect();
    let mut made = run_groups(&hands, 0, messages, &mut |_, messages| so_far(context, &levels.fine, &further, messages)).into_iter();
    let (first, own, double) = (made.next().expect("the scout's pairs at the finer pair's half"), made.next().expect("at the finer"), made.next().expect("at the coarser"));
    let (times, window, scales) = (&plan.reads, plan.window, &context.scales);
    let [h, o, d] = [&first, &own, &double].map(|level| level.iter().collect::<Vec<&Part>>());
    let (half, rung) = (read_step(times, window, &h, &o, scales), read_step(times, window, &o, &d, scales));
    Foreseen { step: 0.5 * lower, first, half, rung }
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
/// `MOST_TRAJECTORIES`; every look with what the atoms of unknown state carry (`bounded`). Each batch's result is a
/// checkpoint; returns the
/// estimates, the listed atoms and the counts at its end, and the estimates at the analysis times.
#[allow(clippy::too_many_arguments)]
fn product(run: &Selected, ensemble: &Ensemble, domain: &Domain, context: &mut Context, parts: &mut Vec<Part>, pilot: &[u64], messages: &mut Messages) -> (Vec<Estimate>, Vec<Vec<(V3, V3)>>, Counts, Vec<Estimate>) {
    let (started, from) = (Instant::now(), parts.len() as u64 * GROUP);
    // The group at the list's place g.
    let number = |g: u64| pilot.get(g as usize).copied().unwrap_or(g);
    // The wall time (ms) the trajectories from `done` to `want` take at the product's pace so far; none before its own
    // first trajectories are in.
    let left = |done: u64, want: u64| (done > from).then(|| started.elapsed().as_secs_f64() * 1e3 / (done - from) as f64 * (want - done) as f64);
    let mut batches = 0;
    loop {
        let all: Vec<&Part> = parts.iter().collect();
        let n = all.len() as u64 * GROUP;
        // The estimates at the analysis times: what the looks and the decisions read (# The analysis times).
        let e = summarize(&reads_of(&all), &context.scales);
        let chased: Vec<Look> = looks(&e, &context.fixed, context.step).into_iter().filter(Look::fits).collect();
        if chased.iter().all(Look::met) || n >= MOST_TRAJECTORIES {
            let (shown, l, c) = pooled(&all, &context.scales);
            return (shown, l, c, e);
        }
        let want = batch(most_needed(&chased), n);
        // The batch begun (# Steps): its trajectories, and how far its looks stand from their targets.
        batches += 1;
        let worst = chased.iter().map(Look::over).fold(0., crate::linalg::larger);
        messages.stage(Stage { step: Step::Product, tau: Some(context.tau), batch: Some(Batch { number: batches, trajectories: n, want, worst: worst.is_finite().then_some(worst) }) });
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
        let end = times[last] + halves[last].iter().fold(0., |a: f64, &b| crate::linalg::larger(a, b));
        let window = TAU / frequencies.iter().copied().fold(0., crate::linalg::larger);
        let mut plan = Plan { end, times: times.to_vec(), halves, listed, cell: end, reads: vec![], read_halves: vec![], window, frequencies };
        plan.read_at(vec![0., end]);
        plan
    }

    /// The plan read at the times `reads` (increasing, from t = 0 to the run's end): T's window about each, half a
    /// period of the axis on either side and cut at both ends of the run (estimate.rs `half_window`), so that it is
    /// centred on its time -- the trajectories run to the run's end and no further.
    fn read_at(&mut self, reads: Vec<f64>) {
        self.read_halves = reads.iter().map(|&t| self.frequencies.map(|w| crate::linalg::smaller(half_window(w, t), self.end - t))).collect();
        self.reads = reads;
    }
}

/// Counts over trajectories: the steps taken -- a step with jumps is one step, whatever the pieces its path is handed
/// on by (adapt.rs, # The jumps) --, and the trajectories that ended with no outcome (estimate.rs, # What is counted),
/// by why: the scheme could not advance them (adapt.rs `advance`'s refusal), a step's optical sector made no K (step.rs
/// `Refused::Optical`, by its `Unmade`), or the model gives no number where they were (`run`: a start that lands
/// nowhere, a readout that is no number -- the trajectory's one exit, said, never carried on as a number); and the
/// trajectories stopped where a state of theirs scattered beyond every state of the atom (`Domain::runaway`),
/// unresolved from there on (`run`).
#[derive(Clone, Copy, Debug, Default)]
struct Counts {
    steps: u64,
    /// The trajectories ended by their steps (`Fate::Refused`), by a jump that gives no photon, by the optical sector.
    refused: Stopped,
    failed: Stopped,
    unmade: [Stopped; 3],
    nowhere: Stopped,
    runaway: Stopped,
    /// The optical sector's sums over the trajectories (ledger.rs `optical`; step.rs `Trajectory::booked`, `pending`).
    optical: [f64; 4],
    /// The steps the scheme refused at the level's length, each taken at a half of it or less, with the time they span
    /// and its shortfall in the order (adapt.rs `Halved`), and the time the trajectories ran (µs, each to where it ended):
    /// what a comparison takes a level's error for (module documentation, ## A companion at τ/2; `amplified`).
    halved: adapt::Halved,
    time: f64,
}
impl Counts {
    fn add(&mut self, o: &Counts) {
        self.steps += o.steps;
        self.halved.add(&o.halved);
        self.time += o.time;
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
            (Some(a), Some(b)) => Some(crate::linalg::smaller(a, b)),
            (a, b) => a.or(b),
        };
    }
}

/// A pilot trajectory's state at a time read (model.rs `Sample`) with the trajectory's number: where the model's error
/// is read and a step is priced (`booked_at`, `work_at`) -- with what of the trajectory a step from it reads: its next
/// jump and its held forms.
#[derive(Clone)]
struct State {
    sample: Sample,
    index: u64,
    /// The trajectory's jumps up to the state and the threshold of its next one (its jumps' counted work is its end's,
    /// `Record::end`).
    jumps: u32,
    threshold: f64,
    /// Its held forms (step.rs `Forms`: the next step's path momentum).
    forms: crate::step::Forms,
}
impl State {
    /// A trajectory's state as it stands.
    fn of(t: &Trajectory) -> State {
        State {
            sample: Sample { cluster: t.cluster, at: t.point(), psi: t.psi.to_vec() },
            index: t.index,
            jumps: t.jumps,
            threshold: t.threshold,
            forms: t.forms.clone(),
        }
    }

    /// The trajectory at the state, as far as a step from it reads it (`step::priced`): its place, momentum, internal
    /// state and time, its next jump's threshold, its held forms and its streams.
    fn trajectory(&self) -> Trajectory {
        let Sample { cluster, at, psi } = &self.sample;
        Trajectory {
            r: at.r,
            kappa: at.kappa,
            psi: psi.iter().copied().collect(),
            cluster: *cluster,
            time: at.t,
            threshold: self.threshold,
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
            forms: self.forms.clone(),
        }
    }
}

/// A kept trajectory's whole state at a step's end (module documentation, # The analysis times; adapt.rs `advance`'s
/// stops): the trajectory and its Brownian path, from which it is advanced again to the bit with a fresh scratch
/// (`replay`; step.rs `Scratch`: nothing of a step is carried to the next).
#[derive(Clone)]
struct Checkpoint {
    trajectory: Trajectory,
    path: adapt::Path,
}

/// What one trajectory leaves: what became of it by every output (estimate.rs `Fate`), its (r, κ) at the listed outputs
/// where held; what became of it by every time the decisions read (`Plan::reads`); where its states are kept (`keep`:
/// the pilot's run level, for the model's errors and prices; module documentation, # The model: where the atoms went),
/// its checkpoints (`trail`, module documentation, # The analysis times) and, where it was lost, the first of those
/// times it is lost at and the end of the path its fate was decided along (its last checkpoint, `run`); its counts.
struct Record {
    outputs: Vec<Fate>,
    listed: Vec<Option<(V3, V3)>>,
    reads: Vec<Fate>,
    trail: Vec<Checkpoint>,
    lost: Option<(usize, State)>,
    counts: Counts,
    /// Its photons and their Σ k_φ² (step.rs `Trajectory::scattered`).
    scattered: [f64; 2],
    /// Its classical draw's kinetic energy K in k_BT (ensemble.rs `Ensemble::draw_kinetic`; none where the run does not
    /// draw it): the statistic of the initial tilt T's memory is read by (`memory_at_the_end`).
    kinetic: f64,
    /// Its number, the time the run holds its atom, and its jumps' counted work by then (step.rs `Trajectory::jumped`):
    /// what a mean over the run weighs it by and its jumps' rate a µs there (`one_each`). The time held: to its loss --
    /// the moment its energy crossed zero, step.rs `Trajectory::lost` --, to where it ended with no outcome, else through
    /// the run's end (a runaway is held, unresolved, at every time it did not reach).
    end: (u64, f64, f64),
}

/// One trajectory's T windows (module documentation, # Outputs and windows) about some times, and what became of it by
/// each time: per lab axis the running integrals from t = 0 of the time held, κ and κ² (µs, 1/µm·µs, 1/µm²·µs), and
/// per time and axis their values at its window's two ends -- the window's time held, ∫κ dt and ∫κ² dt are the
/// differences --; and per lab axis the running moments for the windows' shapes (estimate.rs `Shapes`), the window's
/// origin's at its two ends. An axis's windows' lower ends come in the times' order, and so do their upper ends (a
/// window is cut at t = 0, and at the run's end alike): per axis and side, the next time whose end is to come. Time 0
/// is the instant t = 0 and has none. A time's ends are kept only while its window or its atom is not in (`open`): its
/// atom takes its window's mean as soon as both are (`settle`) -- a trajectory keeps the windows that are open, a period
/// of the slowest axis's times, not one a time read (a pilot's 15 400 for PGC3D's 10 ms).
struct Windows<'a> {
    times: &'a [f64],
    halves: &'a [V3],
    running: [[f64; 3]; 3],
    shapes: [Shapes; 3],
    fates: Vec<Fate>,
    /// From the time `first` on: per time its ends' values as far as they are read, whether its atom is in, and its
    /// ends' shape moments (none where its origin was never met).
    open: VecDeque<([[[f64; 3]; 2]; 3], bool, ShapeEnds)>,
    first: usize,
    next: [[usize; 2]; 3],
}
/// A window's shape moments at its two ends per lab axis: its origin and the origin's moments there (`Shapes::at`).
type ShapeEnds = [[Option<(i64, [f64; SHAPE + 1])>; 2]; 3];
impl<'a> Windows<'a> {
    /// The windows of half widths `halves` about `times`: per lab axis the shapes' grid is the axis's longest half window.
    fn new(times: &'a [f64], halves: &'a [V3]) -> Windows<'a> {
        let shapes = std::array::from_fn(|a| {
            let longest = halves.iter().map(|h| h[a]).fold(0., crate::linalg::larger);
            Shapes::new(a, if longest > 0. { longest } else { f64::INFINITY })
        });
        Windows { times, halves, running: [[0.; 3]; 3], shapes, fates: vec![Fate::Lost; times.len()], open: VecDeque::new(), first: 1, next: [[1; 2]; 3] }
    }
    /// The lower (side 0) or upper (1) end of time k's window on axis a.
    fn end(&self, k: usize, a: usize, side: usize) -> f64 {
        let (t, h) = (self.times[k], self.halves[k][a]);
        if side == 0 { t - h } else { t + h }
    }
    /// Time k's place among the open ones (k ≥ `first`), opened as far as it.
    fn slot(&mut self, k: usize) -> &mut ([[[f64; 3]; 2]; 3], bool, ShapeEnds) {
        while self.open.len() <= k - self.first {
            self.open.push_back(([[[0.; 3]; 2]; 3], false, [[None; 2]; 3]));
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
    /// The path's piece from `from` to `to` (the next piece starts where this one ends), with its integrals from `from`
    /// to any time of it (step.rs `Stretch::to`: of κ and κ² per lab axis, and the shapes' moments): each end inside it
    /// reads the integrals and its origin's moments up to it -- once for the ends that coincide, the axes of one
    /// frequency and the windows cut at t = 0 --, and the running integrals and every origin's moments take the whole
    /// piece.
    fn add(&mut self, from: f64, to: f64, along: &dyn Fn(f64) -> Integrals) {
        let outputs = self.times.len();
        for shapes in self.shapes.iter_mut() {
            shapes.reach(from, to);
        }
        let mut made: Vec<(f64, Integrals)> = vec![];
        for a in 0..3 {
            for side in 0..2 {
                while self.next[a][side] < outputs && self.end(self.next[a][side], a, side) <= to {
                    let k = self.next[a][side];
                    let at = crate::linalg::larger(self.end(k, a, side), from);
                    let i = match made.iter().position(|(x, _)| *x == at) {
                        Some(i) => i,
                        None => {
                            made.push((at, along(at)));
                            made.len() - 1
                        }
                    };
                    let (m, r) = (made[i].1, self.running[a]);
                    let q = self.shapes[a].origin(self.end(k, a, 0));
                    let shape = self.shapes[a].at(q, &m.local).map(|x| (q, x));
                    let slot = self.slot(k);
                    (slot.0[a][side], slot.2[a][side]) = ([r[0] + (at - from), r[1] + m.one[a], r[2] + m.two[a]], shape);
                    self.next[a][side] += 1;
                }
            }
        }
        let whole = along(to);
        for a in 0..3 {
            let r = self.running[a];
            self.running[a] = [r[0] + (to - from), r[1] + whole.one[a], r[2] + whole.two[a]];
            self.shapes[a].add(&whole.local);
        }
        self.settle();
    }
    /// The open times whose windows and atoms are in, from the first: each atom takes its window's mean (`mean`), and
    /// the time is open no more. A window is in where its upper ends are on every axis (a lower end comes before its
    /// upper one).
    fn settle(&mut self) {
        while let Some(&(ends, true, shape)) = self.open.front() {
            if self.next.iter().any(|n| n[1] <= self.first) {
                break;
            }
            mean(&mut self.fates[self.first], &ends, &shape, &self.shapes);
            self.open.pop_front();
            self.first += 1;
        }
    }
    /// The trajectory has ended (run through, lost, or stopped), what became of it by every time in: the ends it did
    /// not reach -- within a window, or past its last piece by the rounding -- take the running integrals and moments
    /// where it ended, the window over its held part, and every open time's atom takes its window's mean. A time none of
    /// whose ends was read has a window of no length (and no atom read: `Fate::Lost` or `Fate::Failed`).
    fn close(mut self) -> Vec<Fate> {
        for a in 0..3 {
            for side in 0..2 {
                for k in self.next[a][side].max(self.first)..self.first + self.open.len() {
                    let q = self.shapes[a].origin(self.end(k, a, 0));
                    let shape = self.shapes[a].at(q, &Local::NONE).map(|m| (q, m));
                    let e = &mut self.open[k - self.first];
                    (e.0[a][side], e.2[a][side]) = (self.running[a], shape);
                }
            }
        }
        for (k, (ends, _, shape)) in (self.first..).zip(&self.open) {
            mean(&mut self.fates[k], ends, shape, &self.shapes);
        }
        self.fates
    }
}

/// An atom held at a time after t = 0 as its window reads it, its window's two ends per lab axis `ends` and its shape
/// moments there: its κ and κ² are the window's means and its shape the window's (estimate.rs `shape`) -- over its held
/// part, which is no empty stretch after t = 0 (estimate.rs `half_window`); a window of no length is the instant's own
/// κ and κ², and no shape.
fn mean(fate: &mut Fate, ends: &[[[f64; 3]; 2]; 3], shape: &ShapeEnds, shapes: &[Shapes; 3]) {
    if let Fate::Held(h) = fate {
        let w: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|i| ends[a][1][i] - ends[a][0][i]));
        for a in (0..3).filter(|&a| w[a][0] != 0.) {
            (h.kappa[a], h.kappa2[a]) = (w[a][1] / w[a][0], w[a][2] / w[a][0]);
            if let [Some((q, low)), Some((_, high))] = &shape[a] {
                h.shape[a] = estimate::shape(shapes[a].spacing(), *q, ends[a][0][0], ends[a][1][0], low, high);
            }
        }
    }
}

// The times after t = 0 of two increasing lists from t = 0, merged: each with its place in either list, a time that is
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
/// -- its start lands nowhere (its forms at the start no number: step.rs `Trajectory::new`), or at a time read its readout is no number -- ends there with no outcome, said (`Counts::nowhere`; # Exits): at t = 0 it is
/// the atom in its initial state, its own populations. Where its states are kept (`keep`), it is
/// advanced from checkpoint to checkpoint (`Checkpoint`: at t = 0 and at the first step's end at or past every multiple
/// of T's shortest window, module documentation, # The analysis times) -- the same trajectory to the bit (adapt.rs
/// `advance`: a stop is no step's end) --, and its states are made again from them (`replay`); else it is advanced
/// through at once.
fn run<D: Dynamics>(model: &D, domain: &Domain, start: Start, index: u64, tau: f64, plan: &Plan, keep: bool) -> Record {
    let _p = crate::prof::scope(crate::prof::SOLVE_TRAJECTORY);
    let sizes = [plan.times.len(), plan.reads.len()];
    let mut t = Trajectory::new(model, start.state, start.r, start.kappa, SEED, index);
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
        // An unresolved one is booked whole (`bounded`). Where the run's fold does not exist for it: no outcome from then
        // on, and at t = 0 its own state's populations.
        let _p = crate::prof::scope(crate::prof::SOLVE_RECORD);
        let mut rows = [0.; STATES];
        if nowhere.get().is_none() {
            rows = shown.copied().unwrap_or_else(|| model.readout(t.cluster, &t.point(), &t.psi));
            if !rows.iter().all(|x| x.is_finite()) {
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
            _ => Fate::Held(Held { rows, kappa: t.kappa, kappa2: t.kappa.map(|x| x * x), shape: NO_SHAPE, unresolved: t.unresolved }),
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
            let stretch = s.stretch(s.start, until);
            for w in windows.borrow_mut().iter_mut() {
                w.add(s.start, until, &|at| stretch.to(at));
            }
        };
        // Through the last output's window -- a failure there leaves every output read --, from checkpoint to checkpoint
        // where the states are kept: stopped at the first step's end at or past each multiple of the window.
        let every = if keep { plan.window } else { plan.end };
        let mut advanced = Ok(());
        while advanced.is_ok() && t.lost.is_none() && t.time < plan.end && nowhere.get().is_none() {
            if keep {
                trail.push(Checkpoint { trajectory: t.clone(), path: path.clone() });
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
            Err(Refused::Overflow | Refused::Reference | Refused::Propagation(_)) => (true, false, true),
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
    // From the time it does not reach: failed, or lost -- a lost atom's last checkpoint where states are kept, a state
    // the loss test found held (every checkpoint after t = 0 is one: the loop stops at a checkpoint only while the atom
    // is held), the end of the path its fate was decided along (module documentation, # Where the atoms went), its
    // fate's to read (`sampled`).
    let (failed, nowhere) = (failed || nowhere.get().is_some(), nowhere.get());
    let mut lost = None;
    let mut windows = windows.into_inner();
    for ((w, &reached), &size) in windows.iter_mut().zip(&reached).zip(&sizes) {
        w.fates[reached.min(size)..].fill(if refused { Fate::Refused } else if failed { Fate::Failed } else { Fate::Lost });
    }
    if reached[1] < sizes[1] && keep && !failed {
        lost = trail.last().map(|c| (reached[1], State::of(&c.trajectory)));
    }
    let [outputs, reads] = windows.map(Windows::close);
    let stopped = |why: bool| if why { Stopped::at(t.time) } else { Stopped::default() };
    let optical = [t.booked[0], t.booked[1], t.booked[2], -(t.pending[0][0] + t.pending[1][1] + t.pending[2][2])];
    let counts = Counts {
        steps,
        refused: stopped(refused),
        failed: stopped(failed && !refused && unmade.is_none() && nowhere.is_none()),
        unmade: [Unmade::Schur, Unmade::Singular, Unmade::Unfinite].map(|why| stopped(unmade == Some(why))),
        nowhere: nowhere.map_or_else(Stopped::default, Stopped::at),
        runaway: stopped(runaway),
        optical,
        halved: path.halved(),
        time: t.time,
    };
    let held = match t.lost {
        Some(at) => at,
        None if failed || refused => t.time,
        None => plan.end,
    };
    Record { outputs, listed, reads, trail, lost, counts, scattered: t.scattered, kinetic: 0., end: (index, held, t.jumped) }
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
            _ => (from.trajectory.clone(), from.path.clone(), Scratch::default()),
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

/// The atom paths of a level whose states are kept (PIPELINE_V3_FINAL §二 B.7; paths.rs `Walk`): `count` of its
/// trajectories in two strata by the time each held its atom (`Part::ends`; select.rs # How many paths) -- those held
/// to the run's end and those lost before it --, half from each (all from one where the other has none, the other's
/// rest where one has too few), each stratum's in its members' order; each path weighs its stratum's share of the
/// level's trajectories over the stratum's paths. Each made again from its first checkpoint at the step τ toward the
/// run's end `end` (`replay`'s way: the same trajectory to the bit), one step at a time -- advanced to the first step's
/// end past its time (adapt.rs `advance`: a stop is no step's end) --, its (t, r, κ) at every step's end as far as the
/// first at or past `until`, its loss or its failure; and its state at t = 0 by the request's states. Reads and changes
/// nothing of the run.
fn walks(run: &Selected, domain: &Domain, tau: f64, end: f64, parts: &[&Part], count: usize, until: f64) -> Vec<crate::paths::Walk> {
    // The level's trajectories with their times held, in their members' order, and their strata.
    let all: Vec<(&Checkpoint, f64)> = parts.iter().flat_map(|p| p.trails.iter().zip(&p.ends).filter_map(|(t, e)| t.first().map(|c| (c, e.1)))).collect();
    let (held, gone): (Vec<&(&Checkpoint, f64)>, Vec<&(&Checkpoint, f64)>) = all.iter().partition(|x| !(x.1 < end));
    let from_gone = gone.len().min(count.saturating_sub(held.len().min(count / 2)).max(count - held.len().min(count)));
    let from_held = held.len().min(count - from_gone);
    let total = all.len() as f64;
    let chosen: Vec<(&Checkpoint, f64)> = held
        .iter()
        .take(from_held)
        .map(|x| (x.0, held.len() as f64 / total / from_held as f64))
        .chain(gone.iter().take(from_gone).map(|x| (x.0, gone.len() as f64 / total / from_gone as f64)))
        .collect();
    chosen
        .par_iter()
        .map(|&(first, weight)| {
            let (mut t, mut path, mut scratch) = (first.trajectory.clone(), first.path.clone(), Scratch::default());
            let states = &run.cluster(t.cluster).states;
            let norm = crate::linalg::norm_sqr(&t.psi).sqrt();
            let start = t.psi.iter().zip(states).filter(|(z, _)| z.norm_sqr() > 0.).map(|(z, &s)| (s, *z * (1. / norm))).collect();
            let mut ends = vec![[t.time, t.r[0], t.r[1], t.r[2], t.kappa[0], t.kappa[1], t.kappa[2]]];
            while t.time < until && t.time < end && t.lost.is_none() {
                let stop = t.time.next_up();
                if adapt::advance(run, domain, tau, &mut t, &mut scratch, &mut path, end, stop, &[], &mut |_, _, _| {}, &mut |_, _| {}).is_err() {
                    break;
                }
                ends.push([t.time, t.r[0], t.r[1], t.r[2], t.kappa[0], t.kappa[1], t.kappa[2]]);
            }
            crate::paths::Walk { index: t.index, start, ends, lost: t.lost.is_some(), weight }
        })
        .collect()
}

/// What a group of trajectories leaves: its sums at the outputs and at the times the decisions read (estimate.rs
/// `Group`), its held atoms' (r, κ) at the listed outputs; where its states are kept (the pilot's run level: every
/// member's), its trajectories' checkpoints in its members' order (`Record::trail`) and, at the analysis times
/// (`parts_at`), its states there; its lost atoms' paths' ends (`Record::lost`) with the first of the times read each
/// is lost at; its trajectories' times held and jumps' work by then (`Record::end`); its counts.
#[derive(Clone)]
struct Part {
    sums: Group,
    reads: Group,
    kinetic: Group,
    listed: Vec<Vec<(V3, V3)>>,
    samples: Vec<Vec<State>>,
    trails: Vec<Vec<Checkpoint>>,
    lost: Vec<(usize, State)>,
    ends: Vec<(u64, f64, f64)>,
    counts: Counts,
}
impl Part {
    /// The same groups' further members, after this part's.
    fn absorb(&mut self, other: Part) {
        self.sums.absorb(other.sums);
        self.reads.absorb(other.reads);
        self.kinetic.absorb(other.kinetic);
        for (l, x) in self.listed.iter_mut().zip(other.listed) {
            l.extend(x);
        }
        for (s, x) in self.samples.iter_mut().zip(other.samples) {
            s.extend(x);
        }
        self.trails.extend(other.trails);
        self.lost.extend(other.lost);
        self.ends.extend(other.ends);
        self.counts.add(&other.counts);
    }

    /// The part at the places `at` of its times read alone (increasing, the first and the last among them: analysis
    /// times, # The analysis times): its sums there, no state yet (`parts_at` makes them), a lost atom's first place
    /// the first of them at or after its own; its outputs as they are.
    fn at(&self, at: &[usize]) -> Part {
        Part {
            sums: self.sums.clone(),
            reads: self.reads.at(at),
            kinetic: self.kinetic.at(at),
            listed: self.listed.clone(),
            samples: vec![vec![]; at.len()],
            trails: vec![],
            lost: self.lost.iter().map(|(first, state)| (at.partition_point(|&k| k < *first), state.clone())).collect(),
            ends: self.ends.clone(),
            counts: self.counts,
        }
    }
}

/// Some groups of a level whose states are kept at the places `at` of their times read (`Part::at`), each with its
/// trajectories' states there made again from their checkpoints (`replay`), in its members' order -- on the pool,
/// trajectory by trajectory --: the level's model `run`, its `domain` and step `tau`, and the plan it ran through.
fn parts_at(run: &Selected, domain: &Domain, tau: f64, plan: &Plan, parts: &[&Part], at: &[usize]) -> Vec<Part> {
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
            kinetic: Group::new(self.reads),
            listed: vec![vec![]; self.listed],
            samples: vec![],
            trails: vec![],
            lost: vec![],
            ends: vec![],
            counts: Counts::default(),
        }
    }
    /// A group's next record added to its part (the records in their members' order).
    fn add(&self, part: &mut Part, r: Record) {
        part.sums.add(&r.outputs);
        part.sums.scatter(r.scattered);
        part.reads.add(&r.reads);
        part.reads.scatter(r.scattered);
        part.kinetic.add_weighted(&r.reads, r.kinetic);
        part.kinetic.scatter(r.scattered.map(|x| x * r.kinetic));
        for (l, x) in part.listed.iter_mut().zip(r.listed) {
            l.extend(x);
        }
        if self.keep {
            part.trails.push(r.trail);
            part.lost.extend(r.lost);
            part.ends.push(r.end);
        }
        part.counts.add(&r.counts);
    }
}

/// Trajectory `index` of a run at the step τ: its initial state from the ensemble (ensemble.rs `Ensemble::draw`), run
/// through the plan (`run`). A function of the index and the step: the same index at another step is the same atom on
/// the same streams (module documentation, # The step).
fn trajectory(model: &Selected, ensemble: &Ensemble, domain: &Domain, tau: f64, plan: &Plan, keep: bool, index: u64) -> Record {
    let (start, kinetic) = ensemble.draw_kinetic(SEED, index);
    Record { kinetic, ..run(model, domain, start, index, tau, plan, keep) }
}

/// A level of a hand-out (`run_groups`): the members `members` of the groups of the numbers `groups` (all of them:
/// 0..GROUP), each trajectory made by `record` from its number, a group's records folded as `shape` says, and a
/// trajectory's work as foretold (`work`: what the hand-out orders its levels by; `foretold`).
struct Hand<'a> {
    groups: &'a [u64],
    members: std::ops::Range<u64>,
    shape: Shape,
    record: &'a (dyn Fn(u64) -> Record + Sync),
    work: f64,
}

/// A trajectory's work at the step `step` as foretold where a hand-out orders its levels (`run_groups`): its work a µs as
/// priced at the states in hand (`Prices::work`, the scout's or the pilot's), where there are prices; before them its
/// steps, 1/h a µs.
fn foretold(prices: Option<&dyn Fn(f64) -> f64>, step: f64) -> f64 {
    prices.map_or(1. / step, |work| work(step))
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
/// handed out together (`pair`, `levels_of`, `foresee`), the level of the dearer trajectories first as foretold
/// (`Hand::work`, `foretold`: largest first, as a list scheduler takes it), so that the threads that wait for the list's
/// last trajectories wait for its cheapest. (Ordered by their steps alone -- the finer first --, a level whose steps
/// cost the more for its tones came last: GMD1C's at 2τ, its pictures 23 times τ's a step.) Each group's records are added in its members' order on this thread
/// as soon as those before it are in -- a record waits only for its group's members before it, never for the whole
/// group (a record holds its trajectory's fates at every time read) -- (`Shape::add`, nothing depends on the threads),
/// each level's groups returned in their order; whenever the solve's
/// next message is due (`Messages::due`), `progress` takes the level `shown`'s groups finished so far in order and sends
/// the result of what is in with them (module documentation, # Checkpoints); where it has nothing to send the message
/// stays due, and is asked for again when the next group is in. A message is made on this thread by loops that run on
/// the pool (`pooled`, `accounts`), whose threads are all in their trajectories: between two trajectories a thread
/// takes what waits for the pool, so a message waits for the next trajectory to end on any thread and not for the
/// list's end. A trajectory's panic -- the engine's defect -- ends the hand-out: no trajectory more is handed out, and
/// once the threads' trajectories in hand end it unwinds to the caller (the worker answers the request with it, main.rs
/// # Defects).
fn run_groups(levels: &[Hand], shown: usize, messages: &mut Messages, progress: &mut dyn FnMut(&[&Part], &mut Messages)) -> Vec<Vec<Part>> {
    // The levels in the list's order, the dearest trajectories first (as given where they are foretold alike), where
    // each level's trajectories begin in the list, and the list's length.
    let mut order: Vec<usize> = (0..levels.len()).collect();
    order.sort_by(|&a, &b| levels[b].work.total_cmp(&levels[a].work));
    let starts: Vec<u64> = order.iter().scan(0, |next, &level| Some(std::mem::replace(next, *next + levels[level].count()))).collect();
    let stop: u64 = levels.iter().map(Hand::count).sum();
    // The list's k-th trajectory: its level and its place there.
    let place = |k: u64| {
        let at = starts.partition_point(|&start| start <= k) - 1;
        (order[at], k - starts[at])
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
                let prefix: Vec<&Part> = parts.get(shown).map_or(vec![], |level| level.iter().map_while(Option::as_ref).collect());
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
/// (populations absolute, T relative; T's unbounded where no pilot atom reads it), none at t = 0, where no model has
/// acted (`model_error`). And what the model's error holds beside it, by bounds (`bounded`): per time the trace distance
/// D(t) ≤ ½(e^{2Yt} − 1) its dropped terms may move the state by (`dropped_rate`; model/classify.rs `distance`), and the
/// bound set's hold on a held atom (`Escape`,
/// `kinetic_bound`), which bounds T where an atom's state is unknown.
#[derive(Clone)]
struct Fixed {
    model: Vec<[f64; 2]>,
    dropped: Vec<f64>,
    rate: f64,
    escape: Escape,
}
impl Fixed {
    /// The model's errors `model` at the times `times`, with the bounds of the dropped terms' rate Y and of `escape`.
    fn new(model: Vec<[f64; 2]>, times: &[f64], rate: f64, escape: Escape) -> Fixed {
        Fixed { model, dropped: times.iter().map(|&t| crate::model::classify::distance(rate, t)).collect(), rate, escape }
    }
    /// The model's errors `model` at the times `times`, with this run's bounds.
    fn with(&self, model: Vec<[f64; 2]>, times: &[f64]) -> Fixed {
        Fixed::new(model, times, self.rate, self.escape)
    }
    /// The model's error at its k-th time with the estimate e there: its measurement and its bounds (`bounded`).
    fn at(&self, k: usize, e: &Estimate) -> [f64; 2] {
        let more = bounded(e, self.dropped[k], &self.escape);
        [self.model[k][0] + more[0], self.model[k][1] + more[1]]
    }
}

/// What the model's error holds beside its measurement at a time (SPEC §4), each a strict bound, on what is shown
/// (absolute) and on T (relative):
/// - the atoms whose state the run does not know: the trajectories not counted (`Estimate::uncounted`, estimate.rs
///   # What is counted) and the unresolved held atoms (`Estimate::unresolved`, module documentation, # The unresolved
///   atoms) -- on what is shown their share whole, the unresolved ones' by the held share n = 1 − L; on T their share of
///   the held atoms (U + F)/(H + F), H held, U of them unresolved, F not counted (each as one held atom);
/// - the model's dropped terms, which may move the state -- the held atoms with the lost ones, on which the propagator
///   is a contraction -- by D in trace distance (`Fixed::dropped`; DROPRES §2): on what is shown D itself, a row as
///   shown and the loss being observables between 0 and 1 (the held atoms' own populations p = P/n move by D/(n − D),
///   DESIGN2 §2.2); on T a share D/(n − D) of the held atoms.
/// On T one bound at the shares' sum (the triangle inequality of the total variation): the held atoms' distribution
/// moved by it within the bound set (estimate.rs `Estimate::moved`). Nothing where every state is known and nothing
/// dropped, to the bit.
fn bounded(e: &Estimate, dropped: f64, escape: &Escape) -> [f64; 2] {
    let held = 1. - e.loss;
    let unknown = if e.held + e.failed > 0. { (e.unresolved * e.held + e.failed) / (e.held + e.failed) } else { 0. };
    let moved = if dropped == 0. { 0. } else if held > dropped { dropped / (held - dropped) } else { 1. };
    [e.uncounted + held * e.unresolved + dropped, e.moved(unknown + moved, escape)]
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
    if !(rest <= 0.) { rest.sqrt() } else { 0. }
}

/// An error on a probability as it is booked (SPEC §4): at most one, the probability's whole range; one that is no
/// number is that whole -- nothing bounds it less (estimate.rs, # What is counted). Every cap of an error on what is
/// shown is this one, so that no cap turns a number that is none into a bound.
fn probability_error(error: f64) -> f64 {
    if error.is_nan() { 1. } else { crate::linalg::smaller(error, 1.) }
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
    /// Its total over its target, 1 or below where it is met: how far it stands from its target, for the product's
    /// stage alone (# Steps; `met` decides).
    fn over(&self) -> f64 {
        self.account.total() / self.target
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
        if self.shown { crate::linalg::larger(draws, draws_for(room)) } else { draws }
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
            let held = (e.held != 0.).then(|| {
                let read = Some(e.temperature_interval()).filter(|i| i.is_finite());
                let account = temperature[k].unwrap_or_else(|| temperature_account(e, fixed.at(k, e)[1], &step, read.unwrap_or(CONFIDENCE * (2. / (n * h)).sqrt())));
                Look { target: e.temperature_target, account, interval: account.statistics, trajectories: n, shown: false }
            });
            std::iter::once(shown).chain(held)
        })
        .collect()
}

/// The trajectories the looks a run's trajectories can meet need (`Look::fits`, `Look::need`), the most of them: what
/// the product's next batch is made for (`batch`), and the product the step's check weighs its choice by (`Prices`).
fn most_needed(looks: &[Look]) -> f64 {
    looks.iter().filter(|l| l.fits()).map(Look::need).fold(0., crate::linalg::larger)
}

/// Each time's accounts of what is shown and of the total T (ledger.rs `Account`) by the run's estimates, the model's
/// errors per time (`Fixed`: on what is shown every atom's, `Measured::means`; on T the held atoms'; with its bounds,
/// `Fixed::at`) and the step's part, T's where read: the held share 1 − L weighs T's whole accounts (module documentation, # Stopping); at t = 0
/// what is shown is the request's state, without error, and T the initial atoms' estimate (module documentation,
/// # Times and windows): no step is taken and no model has acted there, so T's account is its statistics alone.
/// The share of the time's trajectories not counted is booked whole on what is shown, in the model's part
/// (estimate.rs, # What is counted: what no number of trajectories takes away), and on T by the bound (`bounded`). The
/// model's error on what is shown is at most 1, a probability's. T has an account where its
/// statistics, the model's error and its step error are all read. The times' accounts are made on the pool, each
/// time's by itself.
fn accounts(estimates: &[Estimate], fixed: &Fixed, step: Bound) -> (Vec<Account>, Vec<Option<Account>>) {
    estimates
        .par_iter()
        .zip(&fixed.model)
        .enumerate()
        .map(|(k, (e, _))| {
            if k == 0 {
                let temperature = temperature_account(e, 0., &Bound { shown: 0., temperature: 0. }, e.temperature_interval());
                return (Account::default(), temperature.statistics.is_finite().then_some(temperature));
            }
            let a = fixed.at(k, e);
            let population = Account { model: probability_error(a[0]), step: step.shown, statistics: e.shown_statistics() };
            let temperature = temperature_account(e, a[1], &step, e.temperature_interval());
            (population, (temperature.model.is_finite() && temperature.step.is_finite() && temperature.statistics.is_finite()).then_some(temperature))
        })
        .unzip()
}

/// T's account at a time (`accounts`) from its estimate, the model's error on T there `model` (with its bounds,
/// `Fixed::at`), the step's part `bound` (a share of T's allowance) and the statistics' relative interval `interval`:
/// the model's part and the statistics' weighed by the held share, the step's read so already (estimate.rs
/// `between`), times the allowance.
fn temperature_account(e: &Estimate, model: f64, bound: &Bound, interval: f64) -> Account {
    let h = 1. - e.loss;
    Account { model: weighed(h, model), step: bound.temperature * e.temperature_target, statistics: weighed(h, interval) }
}

/// An error of the held atoms weighed by their share h, as an account weighs it: unbounded stays so at any share -- h
/// times it would be no number at h = 0.
fn weighed(h: f64, error: f64) -> f64 {
    if error.is_finite() { h * error } else { f64::INFINITY }
}



/// A state a mean over the run is read at (`one_each`): a pilot trajectory's at one time, with its group's place
/// among the pilot's and its weight in the mean, the time its trajectory holds its atom (µs, `Record::end`).
struct Pick<'a> {
    state: &'a State,
    group: usize,
    weight: f64,
    /// The trajectory's jumps' counted work a µs over the time it holds its atom: its work by then over that time
    /// (`Record::end`).
    jumping: f64,
}

/// The states a mean over the run is read at -- the rates a model is priced by (`Pricing`), a table's whole error
/// (`attempt`) --: one state of every trajectory of `parts`, the pilot's groups with their states kept.
///
/// What is read is the mean of a function f of the state over the time the run's atoms are held, Σ_i ∫ f_i dt /
/// Σ_i T_i, T_i the time trajectory i holds its atom -- to its loss, the moment the engine found its energy past zero,
/// or to where it ended with no outcome, or through the run's end (`Record::end`): one time for every reader of it --:
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
/// and weighs nothing, as does one that holds its atom for no time (no rate a µs to give). (Until 2026-11-22 a
/// trajectory weighed by its last state's time, the last checkpoint where it was lost: R9_PUSH's pushed atoms, 85 of 88
/// held at 11 µs and half of them at 22 µs, weighed nothing, their checkpoint the start, and their jumps a µs read 0/0.)
fn one_each<'a>(parts: &[&'a Part]) -> Vec<Pick<'a>> {
    let groups = parts.len();
    let mut picks = vec![];
    for (group, part) in parts.iter().enumerate() {
        // The group's trajectories in its members' order: every one is held at t = 0.
        let members = part.samples.first().map_or(&[][..], Vec::as_slice);
        for (member, first) in members.iter().enumerate() {
            let held: Vec<&State> = part.samples.iter().skip(1).filter_map(|at| at.iter().find(|state| state.index == first.index)).collect();
            let lost = part.lost.iter().find(|(_, state)| state.index == first.index).map(|(_, state)| state);
            // Its states to pick from, and its last: where it was lost, its path's end (`Record::lost`).
            let states = if held.is_empty() { lost.map(|_| vec![first]).unwrap_or_default() } else { held };
            // The time it holds its atom and its jumps' work by then (`Record::end`).
            let Some(&(_, time, jumped)) = part.ends.iter().find(|end| end.0 == first.index).filter(|end| end.1 > 0.) else { continue };
            if states.is_empty() {
                continue;
            }
            let through = ((member * groups + group) as f64 + 0.5) / (members.len() * groups) as f64;
            let state = states[((through * states.len() as f64) as usize).min(states.len() - 1)];
            picks.push(Pick { state, group, weight: time, jumping: jumped / time });
        }
    }
    picks
}

/// The mean over the run's held states of the values read at `picks` (`one_each`: Σ_i |H_i| f_i / Σ_i |H_i|), of the
/// groups but `left`; zero where no state is read -- nothing is held to err or to work on.
fn over_the_run(picks: &[Pick], values: impl Iterator<Item = f64>, left: Option<usize>) -> f64 {
    let (sum, weight) = picks.iter().zip(values).filter(|(pick, _)| Some(pick.group) != left).fold((0., 0.), |a, (pick, value)| (a.0 + pick.weight * value, a.1 + pick.weight));
    if weight != 0. { sum / weight } else { 0. }
}

/// Histograms of one coordinate of the held atoms at the listed outputs on one grid symmetric about 0 that holds them all
/// (the bins' centres), every bin as wide as the grid's spacing: a listed output with no atom carries the one before it.
/// The number of bins is for the display alone -- no estimate reads a histogram --: the square-root choice, ⌊√n⌋ for the
/// n atoms of the fullest listed output (bins of about √n atoms each), made odd so that 0 is a bin's centre, and three at
/// least -- the contract's grid has two points or more (contract §2.2), and the least odd count that has is three.
fn histograms(listed: &[Vec<(V3, V3)>], outputs: &[usize], value: &dyn Fn(&(V3, V3)) -> f64) -> Distributions {
    let reach = listed.iter().flatten().map(|s| value(s).abs()).fold(f64::MIN_POSITIVE, crate::linalg::larger);
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
    let [after, there] = if booked { ["：此后它们不计入估计，所占的比例全部计入布居与丢失的误差，温度上按束缚原子动能的上限计入", "：它们在那里结束，此后不计入估计，所占的比例全部计入布居与丢失的误差，温度上按束缚原子动能的上限计入"] } else { [""; 2] };
    let mut out = vec![];
    for (stopped, why) in [
        (&counts.refused, "的一步在所取步长下推进不了（算出的数不是有限数，或内态的本征分解不可用），在时间的舍入以内减半步长也推进不了".to_string()),
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
        out.push(format!("{} 条轨迹所在处的折叠不存在（起点处算出的数不是有限数，或某个读出时刻的读数不是有限数，最早在 {}）{there}", counts.nowhere.count, when(t)));
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
    let model = interpolated(&c.at, &c.fixed.model, times);
    // The step's item: the largest it is booked at over the accounts (weighed by the held share as the totals are). And
    // the check in the method's words: the steps compared, the pairs, the resolution, and what is booked -- what the
    // reads the rung stands by do not exclude (`Levels::told`, `checked`).
    let mut step = String::new();
    if let Some(check) = &c.checked {
        let largest = (1..outputs).flat_map(|k| temperature[k].iter().map(|a| a.step).chain([population[k].step])).fold(0., crate::linalg::larger);
        // A level named by its step: τ, or its multiple of τ with its µs.
        let named = |step: f64| {
            let ratio = step / c.tau;
            let name = [(4., "4τ"), (2., "2τ"), (1., "τ"), (0.5, "τ/2"), (0.25, "τ/4")].into_iter().find(|(r, _)| (ratio / r - 1.).abs() < 1e-9).map_or_else(|| format!("{ratio}τ"), |(_, name)| name.to_string());
            if name == "τ" { name } else { format!("{name}（{} μs）", step_reading(c.plan.end, step)) }
        };
        // The rung's companion: 2τ, or τ/2 where that reads it at less work for its precision (module documentation,
        // ## A companion at τ/2); its third level, where it has one, the pair beside its own -- three adjacent levels,
        // or the rung left above's, 2τ against 4τ, beside a rung read against τ/2.
        let own = [c.tau, check.companion.step(c.tau)];
        let why = if check.companion == Companion::Half { "（2τ 的轨迹里有很多步被拒、按一半步长走，按估计精度的代价改和 τ/2 比较）" } else { "" };
        let mut steps: Vec<f64> = own.into_iter().chain(check.third.iter().flat_map(|(s, _)| *s)).collect();
        steps.sort_by(f64::total_cmp);
        steps.dedup();
        let pair = |s: [f64; 2]| format!("{} 与 {}", named(s[0].min(s[1])), named(s[0].max(s[1])));
        let levels = match check.third {
            Some(_) if steps.len() == 3 => format!("{}、{} 与 {}{why}", named(steps[0]), named(steps[1]), named(steps[2])),
            Some((s, _)) => format!("{}{why}，以及 {}", pair(own), pair(s)),
            None => format!("{}{why}", pair(own)),
        };
        // The pairs at each level: the third level's may be fewer than the rung's own (the scout's, or the pilot's
        // beside the further pairs).
        let paired = match check.third {
            Some((s, pairs)) if pairs < check.pairs => format!("{} 对先导轨迹用 {}{why} 各算一遍、其中 {pairs} 对又用 {} 算一遍", check.pairs, pair(own), pair(s)),
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
    // What the unresolved atoms add (`bounded`, their part alone): its largest over the analysis times, weighed by the
    // held share as the accounts weigh it -- on what is shown their share whole, on T the bound at their share of the
    // held atoms, where T is read.
    let kinetic = c.fixed.escape.kinetic_uk();
    for (i, what) in ["布居", "温度"].into_iter().enumerate() {
        let part = |e: &Estimate| if i == 0 { (1. - e.loss) * e.unresolved } else if e.motion.is_some() { weighed(1. - e.loss, e.moved(e.unresolved, &c.fixed.escape)) } else { 0. };
        let largest = (1..reads.len()).map(|k| part(&reads[k])).fold(0., crate::linalg::larger);
        ledger.item(&format!("计算中停下的原子（{what}）"), largest, &format!("在每个输出时刻统计：因散射率超过原子任何状态的上限而停下的原子（从那一步起），布居上按它们所占的比例全部计入；温度上按束缚原子动能的上限（阱深 U₀ = {kinetic:.0} μK）严格计入：份额 w 的原子在束缚集里任意移动时总温度至多变化 w·2U₀/k_B（另加平均速度与 w 的二阶项）"));
    }
    // What the model's dropped terms add (`bounded`): its largest over the analysis times, weighed by the held share as
    // the accounts weigh T -- an item only where it is more than the rounding (`Ledger::item`), so none where nothing is
    // dropped.
    let dropped: Vec<[f64; 2]> = reads.iter().zip(&c.fixed.dropped).map(|(e, &d)| bounded(e, d, &c.fixed.escape)).collect();
    for (i, what) in ["布居", "温度"].into_iter().enumerate() {
        let largest = (1..reads.len()).map(|k| if i == 0 { dropped[k][0] } else if reads[k].motion.is_some() { weighed(1. - reads[k].loss, dropped[k][1]) } else { 0. }).fold(0., crate::linalg::larger);
        ledger.item(&format!("模型删去的项（{what}）"), largest, &format!("建模型时按严格界删去的项：它们让状态在迹距离上至多偏离 D(t) = ½(e^{{2Yt}} − 1) ≈ t·Y（Y 是模型给出的增长率上界）；布居上计 D，温度上按束缚原子动能的上限（阱深 U₀ = {:.0} μK）严格计入", c.fixed.escape.kinetic_uk()));
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
    // # What is counted).
    let (mut empty, mut carried, mut unread) = (None, vec![], vec![]);
    let none: Vec<usize> = (1..outputs).filter(|&k| estimates[k].trajectories == 0.).collect();
    for (k, e) in estimates.iter().enumerate() {
        // At t = 0 every trajectory holds its atom (`run`), and a result rests on two at least: nothing is carried there.
        let last = || k.checked_sub(1).expect("t = 0 holds every trajectory's atom, two at least in a result");
        s.populations.push(if k == 0 { c.initial } else if e.held != 0. { e.populations } else { s.populations[last()] });
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
                if m.temperature_errors.iter().any(|x| !x.read()) && e.held != 0. {
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
                if e.held != 0. {
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
        .filter(|&k| estimates[k].motion.as_ref().is_some_and(|m| m.temperature_errors.iter().all(Spread::read)) && !(model[k][1].is_finite() && c.step.temperature.is_finite()))
        .collect();
    if !unbooked.is_empty() {
        s.approximations.flag(format!("{}：这些时刻仍在光阱中的原子太少，无法估计模型近似或时间步长给温度带来的误差", span(&unbooked, times)));
    }
    // The trajectories that ended with no outcome, by why (`Counts`), and what that leaves (estimate.rs, # What is
    // counted).
    for said in ended(counts, true) {
        s.approximations.flag(said);
    }
    if let Some(t) = counts.runaway.since {
        s.approximations.flag(format!("{} 条轨迹的散射率超过了原子任何状态可能的上限（某个绝热消去远离它的适用范围，最早在 {}）：它们在那里停下，此后按停下时的状态计入，所占的比例全部计入误差（温度上按束缚原子动能的上限）", counts.runaway.count, when(t)));
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
            } else if a.statistics >= crate::linalg::larger(a.model, a.step) {
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

    mod escape;

    /// The model's errors `model` booked at as many times (`Fixed`), with nothing dropped and a held atom's kinetic
    /// energy bounded at a millikelvin (ħ/m = k_B/ħ: a velocity's square is its temperature).
    fn fixed_of(model: Vec<[f64; 2]>) -> Fixed {
        let times = vec![0.; model.len()];
        Fixed::new(model, &times, 0., Escape::new(KT_PER_UK * 1000., KT_PER_UK))
    }

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

    /// The further pairs a remedy takes (`Remedy::Pairs`); none where it reads the third level on the pilot first.
    fn pairs_of(remedy: Remedy) -> u64 {
        match remedy {
            Remedy::Pairs(pairs) => pairs,
            Remedy::Upper => 0,
        }
    }

    /// The run's work a step's check weighs its choice by (`Prices`) in the tests: a trajectory's work a µs as its steps'
    /// count, 1/h, and the product's trajectories.
    fn steps(product: f64) -> (Box<dyn Fn(f64) -> f64>, f64) {
        (Box::new(|h: f64| 1. / h), product)
    }

    /// What the run holds of a rung read against 2τ whose levels took every step at its length (`Holding`): the pilot's
    /// trajectories, the members run at both levels, the pairs in hand, the scout's pairs at τ/2.
    fn holding(pilot: f64, members: f64, pairs: f64, below: f64) -> Holding {
        Holding { pilot, members, pairs, companion: Companion::Double, below: [below, 0.], amplified: [1.; 2], raw: 0., factor: 1. }
    }

    /// The rung below of a rung at τ read against 2τ whose rung below is read against τ, as `check` makes it
    /// (`Below`): its companion the pilot's 88 trajectories at τ.
    fn below_of(asks: Option<f64>, ran: u64, tau: f64) -> Below {
        Below { asks, ran, companion: (tau, 88) }
    }

    /// The third level above a rung at τ read against 2τ, its steps whole (`Third`): the pilot's trajectories at 4τ.
    fn up(tau: f64) -> Option<Third> {
        Some(Third { step: 4. * tau, held: 0., factor: 1., beside: 2. * tau, lift: 0.5_f64.powi(2 * ORDER), measured: None })
    }

    /// The scout's third level above prices the path at its rung by what it reads (module documentation, ## The ladder;
    /// `Levels::third_at`, `in_range`, `third_of`, `Third::measured`): eleven groups whose row as shown is q + C h² at the
    /// step h -- q and C the trajectory's own --, the scout's one member a group and the pilot's eight, at τ, 2τ and 4τ.
    /// In the order's range, d₂/d₁ = 2^p, the third level's read carried to τ is the own one's, to the rounding; where 4τ
    /// errs three times C (4τ)², d₂/d₁ = 44/3 and its read is (11/3)² = 13.4 times the own one's: apart beyond either's
    /// error at one look, on the scout's eleven pairs as on the pilot's 88 -- out of the order's range. Priced (a flat
    /// price, the pilot's 88 pairs), the side above as read costs the pairs it asks beyond the eleven held at 4τ and the
    /// pilot's at 2τ, in whole groups of the pilot's, three at the least (`Third::pairs`); out of the range no pairs make
    /// it the rung's: the side below, as foretold, where it is to be had, and else the path at the rung is without bound
    /// -- the rung below.
    #[test]
    fn the_scout_s_third_level_prices_the_path_at_its_rung_by_what_it_reads() {
        let (tau, times, scales) = (0.95, [0., 1., 2., 3., 4.], Scales { hbar_over_mass: 1., frequencies: [1.; 3] });
        let shape = Shape { outputs: 2, reads: times.len(), listed: 0, keep: false };
        // A trajectory at the step h, its error at 4τ `out` times the order's.
        let record = |h: f64, out: f64, i: u64| {
            let mut rows = [0.; STATES];
            rows[0] = 0.4 + 0.01 * (i % 7) as f64 + (0.002 + 0.0007 * (i % 5) as f64) * h * h * if h > 3. * tau { out } else { 1. };
            let held = Fate::Held(Held { rows, kappa: [0.; 3], kappa2: [0.; 3], shape: NO_SHAPE, unresolved: false });
            Record { outputs: vec![held.clone(); 2], listed: vec![], reads: vec![held; times.len()], trail: vec![], lost: None, counts: Counts { time: 4., ..Counts::default() }, scattered: [0.; 2], kinetic: 0., end: (0, 0., 0.) }
        };
        for members in [1, GROUP] {
            let level = |h: f64, out: f64| -> Vec<Part> { (0..11).map(|g| folded(&shape, (0..members).map(|m| record(h, out, g * GROUP + m)))).collect() };
            let levels = Levels { fine: level(tau, 1.), coarse: level(2. * tau, 1.), ..Levels::default() };
            let own = levels.own(&times, f64::INFINITY, &scales);
            for (out, range) in [(1., true), (3., false)] {
                let extra = level(4. * tau, out);
                let (third, steps, pairs) = levels.third_at(tau, 4. * tau, &extra.iter().collect::<Vec<_>>(), 0, members, &times, f64::INFINITY, &scales);
                eprintln!("{members} a group, 4τ's error {out} times the order's: third {:.4e} ± {:.2e}, own {:.4e} ± {:.2e}", third.shown[0].value, third.shown[0].spread.error, own.shown[0].value, own.shown[0].spread.error);
                assert_eq!((steps, pairs), ([2. * tau, 4. * tau], 11 * members));
                assert_eq!(in_range(&own, &third), range);
                if range {
                    assert!((third.shown[0].value - own.shown[0].value).abs() <= 1e-9 * own.shown[0].value);
                }
            }
        }
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * a.abs().max(b.abs());
        let (pilot, flat) = (holding(88., 88., 88., 0.), |h: f64| 1. / h);
        let prices = Prices { halving: &|_: f64| 1., work: &flat, product: 287. };
        let scouted = |pairs: f64| Third { step: 4. * tau, held: 11., factor: 1., beside: 2. * tau, lift: 1., measured: Some(pairs) };
        let read = scouted(200.);
        assert!(close(read.cost(&prices, 88., 1.), 189. / (4. * tau) + 112. / (2. * tau)));
        assert!(close(scouted(40.).cost(&prices, 88., 1.), 29. / (4. * tau)));
        assert!(close(scouted(10.).cost(&prices, 88., 1.), 13. / (4. * tau)));
        assert!(close(scouted(60.).work(&prices, 88., 1.), 53. / (4. * tau)));
        assert_eq!(third_of(&prices, tau, pilot, [true, false], Some(read)), Some(read));
        let out = scouted(f64::INFINITY);
        assert_eq!(third_of(&prices, tau, pilot, [true, false], Some(out)), None);
        assert_eq!(third_of(&prices, tau, pilot, [true, true], Some(out)).map(|third| third.step), Some(0.5 * tau));
        assert_eq!(course(&prices, tau, 1000., pilot, Some(out)), Course::Below);
    }

    /// The side a rung reads its third level at (`third_of`; module documentation, ## Two levels, three levels): the one
    /// whose read costs the less for its precision (`read_cost`): the pairs it needs within the share at the best -- the
    /// rung's own noise times its lift, 1/16 above and 16 below, and its factor --, on the pilot's at least, at its level
    /// and beside. At a flat price 1/h, the pilot's 88 pairs, its own noise none: above (4τ: 88/4 against 88·2), as the
    /// ladder was, whatever 4τ's factor short of none; its own read needing just the pilot's pairs (raw 88): above while
    /// 4τ's factor leaves it the less (256: 1408 pairs, 1408/4 + 1320/2 = 1012 against 1408·2 + 1320 = 4136 below), below
    /// where it does not (2000) or is without bound; a side not to be had, or whose factor is without bound, is not
    /// taken; read against τ/2, above is 2τ (lift 1) and below τ/4 (lift 256); none where neither side is.
    #[test]
    fn a_rung_reads_its_third_level_at_the_side_of_the_less_cost_for_its_precision() {
        let (tau, flat) = (1., |h: f64| 1. / h);
        let at = |halving: &dyn Fn(f64) -> f64, holding: Holding, sides: [bool; 2]| third_of(&Prices { work: &flat, halving, product: 300. }, tau, holding, sides, None).map(|third| third.step);
        let whole = |_: f64| 1.;
        let quiet = holding(88., 88., 88., 0.);
        assert_eq!((at(&whole, quiet, [true, true]), at(&|h| if h > 3. { 1e9 } else { 1. }, quiet, [true, true])), (Some(4.), Some(4.)));
        let double = Holding { raw: 88., ..quiet };
        assert_eq!(at(&whole, double, [true, true]), Some(4.));
        assert_eq!(at(&|h| if h > 3. { 256. } else { 1. }, double, [true, true]), Some(4.));
        assert_eq!(at(&|h| if h > 3. { 2000. } else { 1. }, double, [true, true]), Some(0.5));
        assert_eq!((at(&|h| if h > 3. { f64::INFINITY } else { 1. }, double, [true, true]), at(&whole, double, [false, true]), at(&whole, double, [false, false])), (Some(0.5), Some(0.5), None));
        let below = Holding { amplified: [f64::INFINITY, 1.], ..double };
        assert_eq!(at(&|h| if h > 3. { f64::INFINITY } else { 1. }, below, [true, true]), None);
        let half = Holding { companion: Companion::Half, below: [88., 0.], amplified: [3., 1.], ..double };
        assert_eq!((at(&whole, half, [true, true]), at(&|h| if h > 1.5 { 1e5 } else { 1. }, half, [true, true])), (Some(2.), Some(0.25)));
    }

    /// Where no rung below is (the ladder's bottom), or no pairs see (`remedy`), the rung asks as it did before the
    /// rung below was priced: the factor `blind` gives, in whole groups, at most twice the pairs it has a batch, wherever
    /// a run can take them -- 7.1 times 88 a first batch of 176, 4 times 352 is 704 (a read of
    /// 1.5 b ± 0.5 b on the total T against b = s², unlimited groups, t = 2: (2.5 − 0.5)/(1 − 0.5)) --; none past the
    /// most trajectories a solve runs, nor for a read whose near end is at the share or that excludes nothing -- the
    /// rung is then left, or at the bottom the check ends unable to see -- whatever the rung below is foreseen to ask.
    /// A read whose estimate is within (a far end of 1.05 b on a read of 0.525 b) asks nothing of its own: the rung
    /// stands on it (`verdict`), and a remedy is for what cannot see.
    #[test]
    fn where_no_rung_below_is_or_no_pairs_see_the_rung_asks_as_before() {
        let b = step_bound()[1].powi(2);
        let (work, product) = steps(300.);
        let prices = Prices { halving: &|_: f64| 1., work: &*work, product };
        let close = |asked: Option<f64>, factor: f64| asked.is_some_and(|f| (f - factor).abs() < 1e-12);
        let wide = at(Some(read(1.5 * b, 0.5 * b)));
        // A read whose near end is at none and far end at f b asks f times its pairs.
        let factor = |f: f64| at(Some(read(0.5 * f * b, 0.25 * f * b)));
        assert!(close(blind(&wide), 4.) && close(blind(&factor(1.05)), 1.05) && close(blind(&factor(7.1)), 7.1));
        let bottom = |r: &Between, have: u64| remedy(r, None, holding(88., 88., have as f64, 0.), None, &prices, 0.25, 1000., up(0.25)).map(pairs_of);
        assert_eq!((bottom(&factor(1.05), 88), bottom(&factor(7.1), 88), bottom(&wide, 352)), (None, Some(176), Some(704)));
        let near = at(Some(read(3. * b, 0.5 * b)));
        let none = at(Some(read(f64::NAN, 0.)));
        for below in [None, Some(below_of(None, 11, 0.25)), Some(below_of(Some(f64::INFINITY), 11, 0.25))] {
            let asks = |r: &Between, have: u64| remedy(r, None, holding(88., 88., have as f64, 11.), below, &prices, 0.25, 1000., up(0.25)).map(pairs_of);
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
            let prices = Prices { halving: &|_: f64| 1., work: &*work, product };
            remedy(r, None, holding(88., 88., 88., 11.), Some(below_of(foreseen(r, half, &rung), 11, 0.25)), &prices, 0.25, 1000., up(0.25)).map(pairs_of)
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
                let prices = Prices { halving: &|_: f64| 1., work: &work, product };
                for tau in [6.1e-5, 0.0954, 0.95, 3.7, 125.] {
                    for (members, pairs, below) in [(11., 11., 0.), (88., 88., 0.), (88., 88., 11.), (88., 352., 11.), (88., 1408., 0.)] {
                        let holding = holding(88., members, pairs, below);
                        for third in [false, true] {
                            let expected = if third { Course::Third } else { Course::Stand };
                            assert_eq!(course(&prices, tau, end, holding, if third { up(tau) } else { None }), expected, "{c} {product} {tau} {holding:?} {third}");
                        }
                        assert!(members < 88. || buys(&prices, tau, end, holding, up(tau).unwrap()), "{c} {product} {tau} {holding:?}");
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
        let at = |members: f64| holding(pilot, members, members, 0.);
        let ladder = |prices: &Prices| -> f64 {
            let mut tau = first;
            while course(prices, tau, end, at(scout), None) == Course::Below {
                tau *= 0.5;
            }
            while course(prices, tau, end, at(pilot), None) == Course::Below {
                tau *= 0.5;
            }
            tau
        };
        let prices: [&dyn Fn(f64) -> f64; 3] = [&steep, &dear, &flat];
        for (p, work) in prices.into_iter().enumerate() {
            for product in [88., 200., 1000., MOST_TRAJECTORIES as f64] {
                let total = |k: i32| (1..=k).map(|j| scout * work(rung(j))).sum::<f64>() + (pilot - scout) * (work(rung(k)) + work(rung(k - 1))) + (crate::linalg::larger(product, pilot) - pilot) * work(rung(k));
                let rungs = (0..).take_while(|&k| end / rung(k) <= MOST_STEPS as f64);
                let best = rungs.min_by(|&a, &b| total(a).total_cmp(&total(b))).unwrap();
                let taken = ladder(&Prices { work, halving: &|_: f64| 1., product });
                assert_eq!(taken, rung(best), "price {p}, product {product}: {:?}", (0..6).map(total).collect::<Vec<_>>());
                match p {
                    0 => assert!(taken == 0.25 || taken == 0.125, "{taken}"),
                    1 => assert!(taken < 0.5 && total(1) > total(0), "{taken}"),
                    _ => assert_eq!(taken, first),
                }
            }
        }
    }

    /// A rung verified where it stands -- its third level read with its two -- or at the rung below, whichever path costs
    /// the run the less (`course`; module documentation, ## The ladder): the pilot's 88 pairs at 0.95 µs of a 1 ms run, a
    /// product of 287 trajectories, a flat price but the third level's steps dearer by a factor f: the rung below adds
    /// (N + P)/τ to staying's (N − P)/τ, so the third level -- the pairs its read asks, here none beyond its least three
    /// groups, 24 f/(4τ) (`Third::pairs`) -- is read up to f = 4(N + P)/24 = 62.5, and beyond it the rung below is the
    /// cheaper path -- one flip. At the ladder's bottom (no rung below) it is read whatever its price. On the scout's
    /// position the pilot's purchase to come is priced into staying: the rung below adds 336.5/τ to the scout's staying
    /// (its companions at 2τ not yet run), so from f = 336.5/6 = 56.1 on the scout goes down. (Until R2-LADDER a third
    /// dearer than that left the rung to stand on its own two levels; the path at the rung is now verified with its third
    /// wherever one is to be had. Until SELCOST the third was the pilot's 88 trajectories whatever its read asked: the
    /// flips at 17.05 and 15.3.)
    #[test]
    fn a_third_level_dearer_than_a_rung_below_is_not_read() {
        let (end, tau, product) = (1000., 0.95, 287.);
        let (pilot, scout) = (holding(88., 88., 88., 0.), holding(88., 11., 11., 0.));
        let priced = |f: f64, tau: f64, holding: Holding| {
            let work = move |h: f64| if h == 4. * tau { f / h } else { 1. / h };
            course(&Prices { halving: &|_: f64| 1., work: &work, product }, tau, end, holding, up(tau))
        };
        let factors: Vec<f64> = (0..=1000).map(|k| 1. + 0.1 * k as f64).collect();
        let courses: Vec<Course> = factors.iter().map(|&f| priced(f, tau, pilot)).collect();
        let flips = courses.windows(2).filter(|w| w[0] != w[1]).count();
        let flip = 4. * (product + 88.) / (THIRD_GROUPS * GROUP) as f64;
        assert!(courses[0] == Course::Third && courses[1000] == Course::Below && flips == 1, "{courses:?}");
        // The flip on the grid itself (62.5) goes either way to the rounding.
        assert!(factors.iter().zip(&courses).all(|(&f, &c)| (f - flip).abs() < 1e-9 || c == if f <= flip { Course::Third } else { Course::Below }), "{flip}");
        let bottom = end / MOST_STEPS as f64 * 1.5;
        assert!(end / (0.5 * bottom) > MOST_STEPS as f64 && priced(1e6, bottom, pilot) == Course::Third);
        assert_eq!([1., 56., 56.2, 62.4, 62.6, 1e6].map(|f| priced(f, tau, scout)), [Course::Third, Course::Third, Course::Below, Course::Below, Course::Below, Course::Below]);
    }

    /// A rung read against τ/2 reads e(τ) as one read against 2τ does (module documentation, ## A companion at τ/2;
    /// `Levels::own`): eleven groups of eight trajectories whose row as shown is q + C h² at the step h -- q and C of the
    /// trajectory's own, the same atom at every level --, at τ/2, τ and 2τ. Against 2τ, d/(2^p − 1) = C τ²; against τ/2,
    /// (τ/2, τ)'s read C τ²/4 raised by the order, 2^p: both reads' mean squares are (C τ²)² over the groups, to the
    /// rounding, and so are their spreads. The rung's told read is its companion's, with its pairs. A companion at 2τ
    /// that took a share f = 0.3 of its time on steps halved once errs by C (2τ)² (1 − f (1 − 2^{−p})): its read as it is
    /// is (1 − f)² of the mean square, and taken back (`read_step`, `amplified`) the whole of it.
    #[test]
    fn a_rung_read_against_its_half_reads_its_error_as_against_its_double() {
        let (tau, times, scales) = (0.95, [0., 1., 2., 3., 4.], Scales { hbar_over_mass: 1., frequencies: [1.; 3] });
        let shape = Shape { outputs: 2, reads: times.len(), listed: 0, keep: false };
        // A trajectory at the step h a share f of whose 4 µs were on steps halved once.
        let record = |h: f64, f: f64, i: u64| {
            let mut rows = [0.; STATES];
            rows[0] = 0.4 + 0.01 * (i % 7) as f64 + (0.002 + 0.0007 * (i % 5) as f64) * h * h * (1. - f * (1. - 0.5_f64.powi(ORDER)));
            let held = Fate::Held(Held { rows, kappa: [0.; 3], kappa2: [0.; 3], shape: NO_SHAPE, unresolved: false });
            let halved = adapt::Halved { steps: 1, time: 4. * f, shortfall: 4. * f * (1. - 0.5_f64.powi(ORDER)) };
            Record { outputs: vec![held.clone(); 2], listed: vec![], reads: vec![held; times.len()], trail: vec![], lost: None, counts: Counts { time: 4., halved, ..Counts::default() }, scattered: [0.; 2], kinetic: 0., end: (0, 0., 0.) }
        };
        let halved = |h: f64, f: f64| -> Vec<Part> { (0..11).map(|g| folded(&shape, (0..GROUP).map(|m| record(h, f, g * GROUP + m)))).collect() };
        let level = |h: f64| halved(h, 0.);
        let double = Levels { fine: level(tau), coarse: level(2. * tau), ..Levels::default() };
        let half = Levels { fine: level(tau), coarse: level(0.5 * tau), companion: Companion::Half, ..Levels::default() };
        let (against_double, against_half) = (double.own(&times, f64::INFINITY, &scales), half.own(&times, f64::INFINITY, &scales));
        // To the rounding: a level's difference rounds by u q/(C τ²) ~ 10⁻¹⁴ of itself, and the spread is a jackknife of group
        // values a hundredth apart, so much the less exact.
        let close = |a: f64, b: f64, by: f64| (a - b).abs() <= by * a.abs().max(b.abs());
        let (d, h) = (against_double.shown[0], against_half.shown[0]);
        eprintln!("against 2τ {:.6e} ± {:.3e}, against τ/2 {:.6e} ± {:.3e}", d.value, d.spread.error, h.value, h.spread.error);
        assert!(d.value > 0. && close(d.value, h.value, 1e-12) && close(d.spread.error, h.spread.error, 1e-10), "{d:?} {h:?}");
        let finer = read_step(&times, f64::INFINITY, &half.coarse.iter().collect::<Vec<_>>(), &half.fine.iter().collect::<Vec<_>>(), &scales);
        assert_eq!(against_half.shown[0].value, carried(&finer, 2.).shown[0].value);
        let (told, third, _, pairs) = half.told(&times, f64::INFINITY, &scales);
        assert_eq!((told.shown[0].value, third.is_none(), pairs), (h.value, true, 88));
        let taken = Levels { fine: level(tau), coarse: halved(2. * tau, 0.3), ..Levels::default() };
        let (fine, coarse) = taken.paired();
        let (back, raw) = (taken.own(&times, f64::INFINITY, &scales).shown[0], read_as_it_is(&times, f64::INFINITY, &fine, &coarse, &scales).shown[0]);
        eprintln!("against 2τ halved: as it is {:.6e}, taken back {:.6e}", raw.value, back.value);
        assert!(close(raw.value, 0.49 * d.value, 1e-12) && close(back.value, d.value, 1e-12) && close(back.spread.error, d.spread.error, 1e-10), "{raw:?} {back:?}");
    }

    /// The rungs below as the ladder prices them where companions may be at τ/2 (`staying`, `descending`,
    /// `against_half`; module documentation, ## The ladder, ## A companion at τ/2): against a count of every level the
    /// rungs below need beyond what they hold -- the members at every rung on the way, the run level to the most of the
    /// product and the pilot, the pilot's companions at τ_{k−1}, or at τ_{k+1} where that costs the less for the read's
    /// precision (the factors of τ's and τ/2's halved steps as held, one below them) --, the least over the rungs, for
    /// rungs read against 2τ and against τ/2, their levels' factors one, a little above one, without bound, and the
    /// scout's pairs run below, at a price that grows with the step and at a flat one. Staying prices the pilot's
    /// companions at the companion's step. With the flat price c/h a rung read against τ/2 whose rungs below read against
    /// their doubles stands: its rung below costs (max(N, P) − P) c/τ more, nothing where the product is the pilot's.
    #[test]
    fn a_rung_s_companion_at_its_half_is_priced_as_its_levels_are_run() {
        let (end, tau, pilot, scout) = (1000., 1., 88., 11.);
        let steep = |h: f64| (1. + (h / 0.25).powi(4)) / h;
        let flat = |h: f64| 3. / h;
        // The rungs below by their levels' counts.
        let counted = |work: &dyn Fn(f64) -> f64, product: f64, holding: Holding| -> f64 {
            let run = product.max(pilot);
            let factor = |j: usize| holding.amplified.get(j).copied().unwrap_or(1.);
            let w = |j: usize| work(tau * 0.5_f64.powi(j as i32));
            (1..)
                .take_while(|&k| end / (tau * 0.5_f64.powi(k)) <= MOST_STEPS as f64)
                .map(|k| {
                    let k = k as usize;
                    let mut need = vec![0.; k + 2];
                    for slot in need.iter_mut().take(k).skip(1) {
                        *slot = holding.members;
                    }
                    need[k] = run;
                    let have = |j: usize| if j == 0 { holding.members } else { holding.below.get(j - 1).copied().unwrap_or(0.) };
                    // A read's work for its precision: the pairs its noise needs (lift times factor on the rung's own),
                    // the pilot's at least, at its level beyond those held and at the rung's beyond the pilot's.
                    let precision = |j: usize, held: f64, lift: f64, a: f64| {
                        let pairs = pilot.max(holding.raw * lift * a);
                        if a < f64::INFINITY { (pairs - held).max(0.) * w(j) + (pairs - pilot).max(0.) * w(k) } else { f64::INFINITY }
                    };
                    let held = if k == 1 { holding.members } else { holding.members.max(have(k - 1)) };
                    let half = !(factor(k - 1) < f64::INFINITY) || precision(k + 1, have(k + 1), 16., factor(k)) < precision(k - 1, held, 1., factor(k - 1));
                    // The companion's work for its read's precision, its run level beside it held to the run's.
                    let (c, held, lift, a) = if half { (k + 1, have(k + 1), 16., factor(k)) } else { (k - 1, held, 1., factor(k - 1)) };
                    let pairs = pilot.max(holding.raw * lift * a);
                    let companion = (pairs - held).max(0.) * w(c) + (pairs - run).max(0.) * w(k);
                    need.iter().enumerate().map(|(j, &n)| (n - have(j)).max(0.) * w(j)).sum::<f64>() + companion
                })
                .fold(f64::INFINITY, f64::min)
        };
        let half = |members: f64, below: [f64; 2], amplified: [f64; 2]| Holding { pilot, members, pairs: members, companion: Companion::Half, below, amplified, raw: 0., factor: 1. };
        let double = |members: f64, below: [f64; 2], amplified: [f64; 2]| Holding { pilot, members, pairs: members, companion: Companion::Double, below, amplified, raw: 0., factor: 1. };
        let holdings = [
            double(scout, [0., 0.], [1., 1.]),
            double(scout, [scout, 0.], [1.06, 1.]),
            double(pilot, [0., 0.], [f64::INFINITY, 1.]),
            double(pilot, [0., 0.], [40., 1.]),
            half(scout, [scout, 0.], [1., 1.]),
            half(scout, [scout, 0.], [f64::INFINITY, 1.]),
            half(pilot, [pilot, scout], [30., 1.2]),
            half(pilot, [pilot, 0.], [1., f64::INFINITY]),
            Holding { raw: 88., ..double(scout, [scout, 0.], [3.2, 1.]) },
            Holding { raw: 300., ..half(pilot, [pilot, scout], [1.2, 1.]) },
        ];
        let works: [&dyn Fn(f64) -> f64; 2] = [&steep, &flat];
        for work in works {
            for product in [0., 88., 300., 5000.] {
                let prices = Prices { work, halving: &|_: f64| 1., product };
                for holding in holdings {
                    let (priced, count) = (descending(&prices, tau, end, holding), counted(work, product, holding));
                    assert!((priced - count).abs() <= 1e-12 * count, "{product} {holding:?}: {priced} against {count}");
                    let stay = staying(&prices, tau, holding);
                    let lift = if holding.companion == Companion::Half { 16. } else { 1. };
                    let pairs = pilot.max(holding.raw * lift * holding.factor);
                    let expected = (pairs - holding.members) * work(holding.companion.step(tau)) + (pairs - product.max(pilot)).max(0.) * work(tau) + (product.max(pilot) - holding.members) * work(tau);
                    assert!((stay - expected).abs() <= 1e-12 * expected, "{holding:?}: {stay} against {expected}");
                }
            }
        }
        for product in [0., 50., 88., 287., 5000.] {
            let prices = Prices { halving: &|_: f64| 1., work: &flat, product };
            for holding in [half(scout, [scout, 0.], [1., 1.]), half(pilot, [pilot, 0.], [1., 1.])] {
                assert_eq!(course(&prices, tau, end, holding, None), Course::Stand, "{product} {holding:?}");
                let more = descending(&prices, tau, end, holding) - staying(&prices, tau, holding);
                assert!((more - (product.max(pilot) - pilot) * flat(tau)).abs() <= 1e-9 * flat(tau) * pilot, "{product} {holding:?}: {more}");
            }
        }
    }

    /// Which companion a rung is read against (`against_half`, `amplified`, `read_cost`; module documentation, ## A
    /// companion at τ/2): a coarser level whose steps were halved once over a share f of its time raises its read by
    /// 1/(1 − f)²; each read costs the pairs it needs within the share at the best -- the rung's own noise times its lift
    /// and its factor, on the pilot's at least -- at its levels. The pilot's 88 pairs, whose read against 2τ needs just
    /// them (raw 88), at w(h) = 0.948/h: against τ/2 its read is lifted 16 times, 1408 pairs, 1408 w(τ/2) + 1320 w(τ);
    /// against 2τ its raw/(1 − f)² pairs, beyond the 88 held, at w(2τ) + w(τ) -- τ/2 from (1 − f)² = raw/(88 + 4136/1.5)
    /// on, f = 0.824; noise that needs ten times fewer pairs (raw 8.8: 141 pairs against τ/2, 334) from f = 0.832 on: never
    /// while the read against 2τ needs no more than the pilot's. Wherever the read against 2τ is without bound, the coarser
    /// level no coarser (2^p r ≤ 1), whatever the price. A level with no step halved takes no factor, to the bit; one
    /// whose steps were halved twice over (to a quarter) takes the larger factor of its smaller error, r = 1 − f (1 −
    /// 4^{−p}).
    #[test]
    fn a_rung_is_read_against_its_half_where_that_costs_the_less_for_its_precision() {
        let part = |halved: f64, quarter: f64| {
            let tau: f64 = 2.;
            let mut part = Shape { outputs: 1, reads: 1, listed: 0, keep: false }.empty();
            part.counts.time = 1000.;
            for (share, length) in [(halved, 0.5 * tau), (quarter, 0.25 * tau)] {
                part.counts.halved.time += share * 1000.;
                part.counts.halved.shortfall += share * 1000. * (1. - (length / tau).powi(ORDER));
            }
            part
        };
        assert_eq!(amplified(&[part(0., 0.)]), 1.);
        for f in [0.01, 0.3, 0.6] {
            let a = amplified(&[part(f, 0.)]);
            assert!((a * (1. - f).powi(2) - 1.).abs() < 1e-12, "{f}: {a}");
            let deeper = amplified(&[part(0., f)]);
            let r = 1. - f * (1. - 0.25_f64.powi(ORDER));
            assert!(deeper > a && (deeper - (3. / (4. * r - 1.)).powi(2)).abs() < 1e-12 * deeper, "{f}: {deeper}");
        }
        assert_eq!(amplified(&[part(0., 1.)]), f64::INFINITY);
        let rough = |h: f64| 0.948 / h;
        let prices = Prices { work: &rough, halving: &|_: f64| 1., product: 0. };
        let switch = |raw: f64, f: f64| against_half(&prices, 0.948, amplified(&[part(f, 0.)]), 1., raw, 88., [88., 0.]);
        // Against 2τ: (88/(1 − f)² − 88)(0.5 + 1); against τ/2: 1408 · 2 + 1320 · 1 = 4136 -- the raw noise times 16.
        let edge = |raw: f64| {
            let half = 16. * raw;
            let half = crate::linalg::larger(half, 88.);
            let cost = half * 2. + crate::linalg::larger(half - 88., 0.);
            1. - (raw / (88. + cost / 1.5)).sqrt()
        };
        for raw in [88., 8.8] {
            let f = edge(raw);
            assert!(!switch(raw, f - 1e-6) && switch(raw, f + 1e-6), "{raw}: {f}");
        }
        assert!(edge(8.8) > edge(88.) && !switch(0., 0.99) && switch(0., 1.) && !switch(88., 0.));
    }

    /// A read as a run logged it: its value and its far end at one look, each over the bound b of its look, on `groups`
    /// groups (the far end's interval Student's t at one look: the standard error back from it).
    fn logged(value: f64, most: f64, groups: f64, b: f64) -> Read {
        let t = Spread { error: 1., groups }.interval();
        Read { value: value * b, spread: Spread { error: (most - crate::linalg::larger(value, 0.)) * b / t, groups } }
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
        assert_eq!(verdict(&more, Some(&above), true, false), Verdict::Unseeing);
        // Its third level, 2.0 b and not clearly (near end 0.94 b, far end 3.07 b), asks (3.07 − 0.94)/(1 − 0.94) = 35
        // times its pairs at 2τ and 4τ beside those at τ; the rung below, foreseen within on the pilot's pairs, costs
        // its run level: the rung is left (m20 is caught by its upper level, priced).
        let (work, _) = steps(0.);
        let prices = Prices { halving: &|_: f64| 1., work: &*work, product: 352. };
        let upper = Upper { read: &above, pairs: 88, held: true, buys: false, ran: 0, step: 4. * 0.392 };
        assert!(seeing(&above, 88).is_some_and(|n| (n / 88. - 35.).abs() < 2.), "{:?}", seeing(&above, 88));
        assert_eq!(remedy(&more, Some(upper), holding(88., 88., 352., 0.), Some(below_of(None, 0, 0.392)), &prices, 0.392, 600., None), None);
    }

    /// A third level is judged as the rung's own read is (`verdict`, `remedy`; module documentation, ## The ladder):
    /// left only where it is clearly beyond, and where its estimate is beyond and not clearly, the pairs that could see
    /// through it are weighed against the rung below. R4_LIN_4 at 0.237 µs, priced as its run priced it (a µs of a
    /// trajectory 9.61·10⁵ at τ, 6.84·10⁵ at 2τ, 6.92·10⁵ at 4τ, 1.54·10⁶ at τ/2; a product of 785.5): its pilot's own
    /// read within, its third level the scout's eleven pairs at 0.474 and 0.948 µs lowered, a row at 3.36 ± 3.59 in one
    /// run and 8.78 ± 5.44 in another (10⁻⁴; the share's square 5·10⁻⁴) -- 0.8σ apart. The rule before stood the first
    /// and left the second, a rung apart on the noise; now the first stands and the second is read on the pilot's own
    /// trajectories at 4τ (88 pairs, 6.1·10⁷ against the rung below's 5.4·10⁸ more), and so is every estimate from half
    /// the share to one and a half at that noise: none is left on it. On the pilot's pairs a read still beyond and not
    /// clearly takes further pairs at τ, 2τ and 4τ while they cost less than the rung below; one whose near end lies
    /// close to the share (1.04 ± 0.025 of it: ten times the pairs) takes the rung below; one clearly beyond leaves.
    #[test]
    fn a_third_level_that_cannot_see_is_priced_as_the_rung_s_own_read_is() {
        let [sp, _] = step_bound();
        let b = sp * sp;
        let tau = 0.236_966_824_644_549_77;
        let work = move |h: f64| {
            let at = |f: f64| (h - f * tau).abs() < 1e-9 * tau;
            if at(0.5) { 1.5423e6 } else if at(1.) { 9.6112e5 } else if at(2.) { 6.8360e5 } else if at(4.) { 6.9207e5 } else { 1. / h * 2.3e5 }
        };
        let prices = Prices { halving: &|_: f64| 1., work: &work, product: 785.5 };
        let row = |r: Read| {
            let mut between = at(None);
            between.shown[7] = r;
            between
        };
        let own = row(Read { value: -1.495e-4, spread: Spread { error: 2.95e-4, groups: 88. } });
        assert!(within(&own) && !screen(&own));
        let (before, after) = (row(pilot(3.357e-4, 3.59e-4)), row(pilot(8.775e-4, 5.44e-4)));
        let below = Some(below_of(None, 0, tau));
        let scout = |third: &Between| remedy(&own, Some(Upper { read: third, pairs: 11, held: false, buys: true, ran: 0, step: 4. * tau }), holding(88., 88., 88., 0.), below, &prices, tau, 1000., None);
        assert_eq!((verdict(&own, Some(&before), true, true), verdict(&own, Some(&after), true, true)), (Verdict::Stands, Verdict::Unseeing));
        assert_eq!(scout(&after), Some(Remedy::Upper));
        for k in 0..=100 {
            let third = row(pilot((0.5 + 0.01 * k as f64) * b, 5.44e-4));
            let verdict = verdict(&own, Some(&third), true, true);
            assert!(verdict == Verdict::Stands || (verdict == Verdict::Unseeing && scout(&third) == Some(Remedy::Upper)), "{k}: {verdict:?}");
        }
        // Where the rung above is no rung, or its read on the pilot's trajectories was unread, no pairs see it: the
        // rung below.
        assert_eq!(remedy(&own, Some(Upper { read: &after, pairs: 11, held: false, buys: false, ran: 0, step: 4. * tau }), holding(88., 88., 88., 0.), below, &prices, tau, 1000., None), None);
        // On the pilot's own 88 pairs at 4τ, held.
        let held = |third: &Between| remedy(&own, Some(Upper { read: third, pairs: 88, held: true, buys: false, ran: 0, step: 4. * tau }), holding(88., 88., 88., 0.), below, &prices, tau, 1000., None);
        let still = row(Read { value: 5.5e-4, spread: Spread { error: 1.9e-4, groups: 88. } });
        assert_eq!((verdict(&own, Some(&still), true, false), held(&still)), (Verdict::Unseeing, Some(Remedy::Pairs(176))));
        let near = row(read(1.04 * b, 0.025 * b));
        assert_eq!((verdict(&own, Some(&near), true, false), held(&near)), (Verdict::Unseeing, None));
        assert_eq!(verdict(&own, Some(&row(read(4. * b, 0.1 * b))), true, false), Verdict::Left);
        // The rung's own read beyond and not clearly, its third level within: as before, its pairs or the rung below.
        let blind_own = row(read(1.5 * b, 0.5 * b));
        let quiet = row(pilot(0.2 * b, 0.1 * b));
        assert_eq!(verdict(&blind_own, Some(&quiet), true, false), Verdict::Unseeing);
        assert_eq!(remedy(&blind_own, Some(Upper { read: &quiet, pairs: 11, held: false, buys: true, ran: 0, step: 4. * tau }), holding(88., 88., 88., 0.), below, &prices, tau, 1000., None), remedy(&blind_own, None, holding(88., 88., 88., 0.), below, &prices, tau, 1000., None));
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
        let mut levels = Levels { fine: pilot_groups(), coarse: pilot_groups(), further: [kept(34, GROUP as usize, 4, &|_, _| 4), kept(34, GROUP as usize, 4, &|_, _| 4), vec![]], ..Levels::default() };
        let times = [0., 1., 2., 3., 4.];
        let (_, third, step, pairs) = levels.told(&times, f64::INFINITY, &scales);
        assert_eq!((third.is_none(), step, pairs), (true, None, 360));
        let lowered = carried(&at(Some(pilot(0.8 * b, 0.1 * b))), 0.5);
        (levels.above, levels.above_steps, levels.above_pairs) = (Some(lowered.clone()), [2. * tau, 4. * tau], 11);
        let (_, third, step, _) = levels.told(&times, f64::INFINITY, &scales);
        assert_eq!((third.unwrap().temperature, step), (lowered.temperature, Some(([2. * tau, 4. * tau], 11))));
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
                mean_speed: [0.; 3],
            }),
            failed: 0.,
            uncounted: 0.,
            unresolved: 0.,
            temperature_target: TEMPERATURE_TARGET,
        }
    }

    /// The share not counted (estimate.rs, # What is counted) in the accounts and in the result. Four groups of two
    /// trajectories at t = 0 and three outputs, every atom held in state 0 until it ends with no outcome: the first of
    /// each group after the first output, the second after the second. The model's error booked at 1 % on the held
    /// atoms' populations and 2 % on T, the step at its share. At the first output the accounts are the model's and T
    /// has one; at the second what is shown takes the uncounted half whole, 1 % + 50 %, and T the bound of half its held
    /// atoms moved anywhere in the bound set (1 mK, `bounded`), the atoms at rest (`Estimate::moved`); at the third,
    /// none counted, what is shown is booked whole, its model's part and its statistics' at one: those two outputs' looks
    /// on what is shown have no room. The result:
    /// every number finite (the app refuses one that is not), the loss and the populations at the third output the
    /// second's, its error at least the whole, and the flags say why -- the trajectories that ended by their reasons
    /// (a step's optical sector that made no K by its own), counts and first times, the outputs none is counted at --
    /// and nothing of an atom lost, nor of T unbooked; the reason a result is partial is the output none is counted at.
    #[test]
    fn the_share_not_counted_is_booked_whole_and_an_output_none_counts_at_says_so() {
        let groups: Vec<Group> = (0..4)
            .map(|g| {
                let mut group = Group::new(4);
                let held = Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [1. + g as f64; 3], shape: NO_SHAPE, unresolved: false });
                group.add(&[held.clone(), held.clone(), Fate::Failed, Fate::Failed]);
                group.add(&[held.clone(), held.clone(), held, Fate::Failed]);
                group
            })
            .collect();
        let scales = Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] };
        let estimates = summarize(&groups, &scales);
        let (fixed, step) = (fixed_of(vec![[0.01, 0.02]; 4]), Bound::share());
        let (population, temperature) = accounts(&estimates, &fixed, step);
        assert!((population[1].model - 0.01).abs() < 1e-15 && temperature[1].is_some_and(|t| (t.model - 0.02).abs() < 1e-15), "{population:?} {temperature:?}");
        let bound = 0.02 + estimates[2].moved(0.5, &fixed.escape);
        assert!((population[2].model - 0.51).abs() < 1e-15 && temperature[2].is_some_and(|t| (t.model - bound).abs() < 1e-14 * bound), "{temperature:?} against {bound}");
        assert!((estimates[2].moved(0.5, &fixed.escape) - 0.5 * 2000. * 4. / estimates[2].motion.unwrap().temperature_total).abs() < 1e-12 * bound, "at rest, w = 1/2: w (2a)² m/k_B over T");
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
            step,
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let unmade = [Stopped::default(), Stopped { count: 2, since: Some(14.) }, Stopped::default()];
        let counts = Counts { steps: 0, refused: Stopped { count: 4, since: Some(11.) }, failed: Stopped { count: 3, since: Some(12.5) }, unmade, nowhere: Stopped { count: 6, since: Some(16.) }, runaway: Stopped { count: 5, since: Some(15.) }, optical: [0.0; 4], halved: adapt::Halved::default(), time: 0. };
        let s = assemble(&context, &estimates, &estimates, &listed, &counts, 8);
        assert!(s.estimated_errors.iter().chain(&s.loss_probability).chain(s.populations.iter().flatten()).all(|x| x.is_finite()), "{s:?}");
        assert!(s.loss_probability[3] == s.loss_probability[2] && s.populations[3] == s.populations[2] && s.estimated_errors[3] >= 1., "{s:?}");
        let flagged = s.approximations.flagged.expect("flagged");
        for said in ["4 条轨迹的一步在所取步长下推进不了（算出的数不是有限数，或内态的本征分解不可用），在时间的舍入以内减半步长也推进不了（最早在 11 μs）：此后它们不计入估计", "3 条轨迹的跳跃给不出光子（模型在该状态上失败，缩短步长无济于事）（最早在 12.5 μs）：此后它们不计入估计", "2 条轨迹在某一步上，光学扇区的噪声算子 K 做不出（ℳ 奇异", "5 条轨迹的散射率超过了原子任何状态可能的上限（某个绝热消去远离它的适用范围，最早在 15 μs）：它们在那里停下", "6 条轨迹所在处的折叠不存在（起点处算出的数不是有限数，或某个读出时刻的读数不是有限数，最早在 16 μs）：它们在那里结束", "30 μs 起：没有一条轨迹计入估计"] {
            assert!(flagged.contains(said), "{said} in {flagged}");
        }
        assert!(!flagged.contains("已无原子留在光阱中") && !flagged.contains("温度的误差无法估计"), "{flagged}");
        assert!(s.reason.as_ref().is_some_and(|r| r.starts_with("t = 30 μs") && r.ends_with("：没有一条轨迹计入估计")), "{:?}", s.reason);
    }

    /// The two ways T goes unread, said apart (`assemble`): an output whose atoms sit in one group (two of them) has its
    /// own T with no error -- the jackknife, every group's removal leaving fewer than two atoms, reads none --; an output
    /// of one atom carries the last T, n̄, mean velocity and the judgement of the validated regime that goes with that T
    /// (`classical`), not a fresh "below 4ħω". Five groups of four: every atom held at 10 μs, one group's two at 20 μs,
    /// one atom at 30 μs, none at 40 μs.
    #[test]
    fn a_t_carried_keeps_its_judgement_and_a_t_unread_is_its_own() {
        let held = |g: usize, m: usize| Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [100. + (4 * g + m) as f64; 3], shape: NO_SHAPE, unresolved: false });
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
            fixed: fixed_of(vec![[0.01, 0.02]; 5]),
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
            Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == (g + m) % 8))), kappa: [if still { 0. } else { 0.1 * m as f64 }; 3], kappa2: [y; 3], shape: NO_SHAPE, unresolved })
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
                assert!(number(e.unresolved) && number_or_unbounded(e.temperature_interval()), "output {k}: {e:?}");
                assert!(number(e.shown_statistics()) && e.shown_statistics() <= 1., "output {k}: {e:?}");
            }
            for booked in [[0.01, 0.02], [1., f64::INFINITY], [f64::NAN; 2]] {
                let fixed = fixed_of(vec![booked; kinds + 1]);
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
                fixed: fixed_of(vec![[0.01, 0.02]; kinds + 1]),
                step: Bound::share(),
                checked: None,
                ledger: Ledger::default(),
                method: String::new(),
            }, &estimates, &estimates, &vec![vec![]; Plan::new(&(0..=kinds).map(|k| 10. * k as f64).collect::<Vec<_>>(), [1.; 3]).listed.len()], &Counts::default(), 20);
            assert!(s.estimated_errors.iter().chain(&s.loss_probability).chain(s.populations.iter().flatten()).all(|x| number(*x)), "{s:?}");
            let flagged = s.approximations.flagged.expect("flagged");
            assert!(flagged.contains("30 μs") && flagged.contains("70 μs：没有一条轨迹计入估计") && !flagged.contains("温度的误差无法估计"), "{flagged}");
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
        let fixed = fixed_of(vec![[0.; 2], [0., 0.03]]);
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
        let mut full = fixed_of(fixed.model.clone());
        full.model[1][1] = 0.11;
        assert!(crowded(&looks(&run, &full, step)) == 1);
        let account = accounts(&run, &fixed, step).1[1].unwrap();
        assert!(account.model == 0.03 && (account.step - 0.02).abs() < 1e-17 && (account.statistics - CONFIDENCE * 0.075).abs() < 1e-14, "{account:?}");
    }

    /// The model's dropped terms are booked by the trace distance they may move the state by, D = ½(e^{2Yt} − 1)
    /// (`bounded`, `Fixed::at`, model/classify.rs `distance`): at Y = 10⁻⁴/µs, D = 0.010067 at 100 µs -- on what is shown
    /// the measured 1 % plus D; on T the measured 3 % plus the bound at the share D/(1 − D) of the held atoms moved
    /// within the bound set (estimate.rs
    /// `Estimate::moved`, a 1 mK bound), weighed by the held share; in the result an item each, their largest. At Y = 0
    /// every account is the measurement's to the bit, and no item is made.
    #[test]
    fn the_dropped_terms_are_booked_by_their_trace_distance() {
        let escape = Escape::new(1000. * KT_PER_UK, KT_PER_UK);
        let e = estimate([25., 6., 10.], 0.01, 800.);
        let run = [e.clone(), e.clone()];
        let model = vec![[0.; 2], [0.01, 0.03]];
        let step = Bound::share();
        let dropped = Fixed::new(model.clone(), &[0., 100.], 1e-4, escape);
        let (population, temperature) = accounts(&run, &dropped, step);
        let d = crate::model::classify::distance(1e-4, 100.);
        assert!((d - 0.5 * 0.02_f64.exp_m1()).abs() < 1e-17 && d > 0.01, "{d}");
        let w = d / (1. - d);
        let expected = 0.03 + e.moved(w, &escape);
        assert!((population[1].model - (0.01 + d)).abs() < 1e-15 && temperature[1].is_some_and(|t| (t.model - expected).abs() < 1e-14 * expected), "{population:?} {temperature:?} against {expected}");
        assert!(e.moved(w, &escape) > 1., "a millikelvin's bound on 13.7 µK: {}", e.moved(w, &escape));
        let none = Fixed::new(model.clone(), &[0., 100.], 0., escape);
        assert_eq!(format!("{:?}", accounts(&run, &none, step)), format!("{:?}", accounts(&run, &fixed_of(model), step)));
        let items = |fixed: Fixed| {
            let plan = Plan::new(&[0., 100.], [1.; 3]);
            let listed = vec![vec![]; plan.listed.len()];
            let context = Context { at: vec![0., 100.], plan, tau: 1., scales: Scales { hbar_over_mass: KT_PER_UK, frequencies: [1.; 3] }, initial: std::array::from_fn(|s| f64::from(u8::from(s == 0))), fixed, step, checked: None, ledger: Ledger::default(), method: String::new() };
            assemble(&context, &run, &run, &listed, &Counts::default(), 800).approximations.items
        };
        let made = items(dropped);
        let item = |label: &str| made.iter().find(|i| i.label == label).map(|i| i.estimate);
        assert!(item("模型删去的项（布居）") == Some(d) && item("模型删去的项（温度）").is_some_and(|x| (x - e.moved(w, &escape)).abs() < 1e-14 * x), "{made:?}");
        assert!(!items(none).iter().any(|i| i.label.starts_with("模型删去的项")));
    }

    /// The atoms whose state the run does not know are booked with the dropped terms by one bound (`bounded`): 800
    /// atoms held at 13.7 µK, 1 % of them unresolved, 8 trajectories not counted, the model's dropped terms at Y = 10⁻⁴/µs
    /// over 100 µs, D = ½(e^{0.02} − 1) (model/classify.rs `distance`; a 1 mK bound). On what is shown the measured 1 % plus the share not counted, the unresolved share and D; on T the
    /// measured 3 % plus the bound at (8 + 8)/808 + D/(1 − D), more than the two shares' bounds taken apart and added
    /// (the bound is convex in the share). Without the unknown atoms and the dropped terms, the measurement alone.
    #[test]
    fn the_atoms_of_unknown_state_and_the_dropped_terms_share_one_bound() {
        let escape = Escape::new(1000. * KT_PER_UK, KT_PER_UK);
        let mut e = estimate([25., 6., 10.], 0.01, 800.);
        (e.unresolved, e.failed, e.uncounted) = (0.01, 8., 8. / 808.);
        let fixed = Fixed::new(vec![[0.; 2], [0.01, 0.03]], &[0., 100.], 1e-4, escape);
        let run = [e.clone(), e.clone()];
        let (population, temperature) = accounts(&run, &fixed, Bound::share());
        let d = crate::model::classify::distance(1e-4, 100.);
        let (unknown, dropped) = (16. / 808., d / (1. - d));
        let expected = [0.01 + (8. / 808. + 0.01 + d), 0.03 + e.moved(unknown + dropped, &escape)];
        assert!((population[1].model - expected[0]).abs() < 1e-15 && temperature[1].is_some_and(|t| (t.model - expected[1]).abs() < 1e-14 * expected[1]), "{population:?} {temperature:?} against {expected:?}");
        assert!(e.moved(unknown + dropped, &escape) > e.moved(unknown, &escape) + e.moved(dropped, &escape));
        let known = estimate([25., 6., 10.], 0.01, 800.);
        assert_eq!(bounded(&known, 0., &escape), [0.; 2]);
    }

    /// A T whose statistics no group reads is held as one whose statistics are the normal theory's prediction, in whether
    /// its look is met and in the trajectories it needs alike (`looks`): S2Q at 2.5 ms, 568 trajectories and a held
    /// share of 3·10⁻⁴, met -- it neither keeps the product running nor asks for a trajectory --; at a held share of
    /// one on 100 trajectories, neither met nor within its need. Met exactly where its need is within the trajectories
    /// it has, on either side.
    #[test]
    fn an_unread_t_of_no_weight_neither_blocks_nor_asks() {
        let fixed = fixed_of(vec![[0.; 2], [0., 0.01]]);
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
                        let held = Fate::Held(Held { rows: std::array::from_fn(|s| f64::from(u8::from(s == 0))), kappa: [0.; 3], kappa2: [kappa2; 3], shape: NO_SHAPE, unresolved: false });
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
            let fixed = fixed_of(vec![[0.; 2], [0., 0.03]]);
            let need = looks(&[e.clone(), e.clone()], &fixed, Bound::share())[1].need();
            let crowded = !(looks(&[e.clone(), e], &fixed_of(vec![[0.; 2], [0., 0.3]]), Bound::share())[1].room() > 0.);
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
        State { sample: Sample { cluster: 0, at: crate::model::Point { r: [0.; 3], kappa: [0.; 3], t }, psi: vec![] }, index, jumps, threshold: 0.5, forms: crate::step::Forms::default() }
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
                // Each held to its last held output's time, three jumps a unit of time by then.
                let ends = (0..members).map(|m| (MOST_TRAJECTORIES + (g as u64) * GROUP + m as u64, held(g, m) as f64, 3. * held(g, m) as f64)).collect();
                Part { sums: Group::new(outputs + 1), reads: Group::new(outputs + 1), kinetic: Group::new(outputs + 1), listed: vec![], samples, trails: vec![], lost: vec![], ends, counts: Counts::default() }
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
                assert!((mean - 0.5 * (outputs + 1) as f64).abs() <= 0.5 * crate::linalg::larger(outputs as f64 / n as f64, 1.), "{outputs} outputs: {mean}");
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
    /// mean κ and κ² within 1e-12 of its own scale, and its shape (estimate.rs `shape`) its means of κ P_ℓ by a dense
    /// rule within the moments' rounding through the conversion (`amplification`: every summed term's rounding, times
    /// Σ_j |T_ℓj|), zero past its degree; an atom lost at 17.6 µs, the windows that reach past it over their held part,
    /// no atom after it. The times kept open are never more than a window's span of them and two.
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
        let kappa = |t: f64| -> V3 { std::array::from_fn(|a| amp[a] * (w[a] * t + phase[a]).cos()) };
        // Dense Gauss-Legendre over a stretch: its nodes, weights and κ; a piece's integrals from its start (as step.rs
        // `Stretch::to` gives them), κ and κ² exact, the shapes' moments by the dense rule.
        let rule = crate::special::gauss_legendre(60);
        let nodes = |from: f64, to: f64| -> Vec<(f64, f64, V3)> {
            rule.iter().map(|&(x, weight)| { let t = 0.5 * (from + to) + 0.5 * (to - from) * x; (t, 0.5 * (to - from) * weight, kappa(t)) }).collect()
        };
        let along = |from: f64, at: f64| -> Integrals {
            let (one, two) = moments(from, at);
            Integrals { one, two, local: Integrals::of(from, nodes(from, at)).local }
        };
        let longest: V3 = std::array::from_fn(|a| plan.halves.iter().map(|h| h[a]).fold(0., crate::linalg::larger));
        let mut worst = 0_f64;
        // The pieces: steps of τ, the third of every three cut by a jump at 0.43 τ.
        let tau = 0.37;
        let mut cuts = vec![0.];
        for i in 0.. {
            let (start, step_end) = (i as f64 * tau, crate::linalg::smaller((i + 1) as f64 * tau, plan.end));
            if i % 3 == 2 && start + 0.43 * tau < step_end {
                cuts.push(start + 0.43 * tau);
            }
            cuts.push(step_end);
            if step_end >= plan.end {
                break;
            }
        }
        let atom = Fate::Held(Held { rows: [0.; STATES], kappa: [0.; 3], kappa2: [0.; 3], shape: NO_SHAPE, unresolved: false });
        let span = 2. * plan.halves.iter().flatten().copied().fold(0., crate::linalg::larger);
        for stop in [plan.end, 17.6] {
            let mut windows = Windows::new(&plan.times, &plan.halves);
            windows.record(0, atom.clone());
            let (mut next, mut most) = (1, 0);
            for piece in cuts.windows(2).filter(|p| p[0] < stop) {
                let to = crate::linalg::smaller(piece[1], stop);
                // The times in the piece read before it is added (the odd ones) and after it (the even ones).
                let inside: Vec<usize> = (next..times.len()).take_while(|&k| times[k] <= to).collect();
                inside.iter().filter(|&&k| k % 2 == 1).for_each(|&k| windows.record(k, atom.clone()));
                windows.add(piece[0], to, &|at| along(piece[0], at));
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
                    let (lo, hi) = (times[k] - plan.halves[k][a], crate::linalg::smaller(times[k] + plan.halves[k][a], stop));
                    let exact: [f64; 3] = { let (p, q) = (antiderivative(a, lo), antiderivative(a, hi)); std::array::from_fn(|i| q[i] - p[i]) };
                    let scale = 1e-12 * stop * crate::linalg::larger(amp[a], 1.).powi(2) / exact[0];
                    assert!((h.kappa[a] - exact[1] / exact[0]).abs() <= scale && (h.kappa2[a] - exact[2] / exact[0]).abs() <= scale, "stop {stop}, output {k}, axis {a}: {} and {} against {} and {}", h.kappa[a], h.kappa2[a], exact[1] / exact[0], exact[2] / exact[0]);
                    let (centre, half) = (0.5 * (lo + hi), 0.5 * (hi - lo));
                    let degree = estimate::shape_degree(PI * half / longest[a]);
                    // The origin's moments summed every node of every piece since the origin was met (a piece before its
                    // grid point at the most), each term rounded at its power and in the sum.
                    let origin = (lo / longest[a]).floor() * longest[a];
                    let terms = (60 * (cuts.iter().filter(|&&c| c > origin - tau && c < hi).count() + 1) * (SHAPE + 2)) as f64;
                    let gain = amplification(longest[a], lo, hi);
                    for l in 1..=SHAPE {
                        let dense: f64 = if l > degree { 0. } else { nodes(lo, hi).iter().map(|&(t, weight, k)| weight * k[a] * legendre(l, (t - centre) / half)).sum::<f64>() / (hi - lo) };
                        let error = (h.shape[a][l - 1] - dense).abs() / amp[a];
                        let bound = (gain[l] * terms * (hi - origin) / (hi - lo) + 64.) * crate::units::UNIT_ROUNDOFF;
                        worst = worst.max(error / bound);
                        assert!(error <= bound, "stop {stop}, output {k}, axis {a}, degree {l} of {degree}: {} against {dense}, {error:.2e} of the amplitude beyond {bound:.2e}", h.shape[a][l - 1]);
                    }
                }
            }
            assert!(most as f64 <= span + 2., "{most} times open against a span of {span} µs");
        }
        println!("the shapes' largest error over its bound: {worst:.2e}");
    }

    /// The conversion's amplification of a window [lower, upper] on a grid of spacing G (estimate.rs `shape`): per
    /// degree ℓ, Σ_j |T_ℓj| of P_ℓ(αy + β) in the powers of y -- what the rounding of the origin's moments is multiplied
    /// by -- over the window's half width relative to the spacing's.
    fn amplification(spacing: f64, lower: f64, upper: f64) -> [f64; SHAPE + 1] {
        let (h, s) = (0.5 * (upper - lower), 1.5 * spacing);
        let q = (lower / spacing).floor();
        let (alpha, beta) = (s / h, (q * spacing + s - (lower + h)) / h);
        let (mut before, mut now) = ([0.; SHAPE + 2], [0.; SHAPE + 2]);
        (before[0], now[0], now[1]) = (1., beta, alpha);
        let mut out = [1.; SHAPE + 1];
        for l in 1..=SHAPE {
            out[l] = now.iter().map(|c| c.abs()).sum();
            let mut next = [0.; SHAPE + 2];
            for j in 0..=(l + 1).min(SHAPE + 1) {
                let times = beta * now[j] + if j > 0 { alpha * now[j - 1] } else { 0. };
                next[j] = ((2 * l + 1) as f64 * times - l as f64 * before[j]) / (l + 1) as f64;
            }
            (before, now) = (now, next);
        }
        out
    }

    /// P_ℓ(x) by Bonnet's recurrence.
    fn legendre(l: usize, x: f64) -> f64 {
        let (mut before, mut now) = (1., x);
        if l == 0 {
            return 1.;
        }
        for n in 1..l {
            (before, now) = (now, ((2 * n + 1) as f64 * x * now - n as f64 * before) / (n + 1) as f64);
        }
        now
    }

    /// The cloud's own motion is no temperature (estimate.rs, # The cloud's own motion), whatever it is: 64 atoms in eight
    /// groups on the windows of the test above (periods 0.7, 2.3 and 23 µs, outputs 1 µs apart, windows cut at t = 0
    /// and whole), each its own oscillation κ_a = A cos(ω_a t + φ) with a kick of its own at every jump, and on top of
    /// them a path every atom shares -- none, a push (a ramp of 0.6/µm per µs, a fall alike) and a sloshing at 0.86 ω_a
    /// of amplitude 5/µm, the cloud's motion's window variance up to 70 times the atoms' spread. T and its jackknife
    /// per axis and output read with the shared path are those without it within 1e-9 of the paths' scale (the atoms'
    /// spread and the shared path's mean square over the window): the shapes take the path out but for the rounding of
    /// their conversion (bounded in the test above). So is what the atoms whose state the run does not know carry of T:
    /// every seventh atom unresolved, their share of the held atoms (estimate.rs `Estimate::unresolved`, booked on T as
    /// the bound of that share moved: `Estimate::moved`) is the same with the shared path and without it. Read with no
    /// shapes (the estimate of before) T moves with the path by its window variance and twice its window covariance with
    /// the atoms' own mean path, Var_W(g) + 2Cov_W(g, ū) (the U-statistic's algebra), by a dense rule on the pieces within
    /// 1e-9 of the scale -- what the shapes take out.
    #[test]
    fn the_cloud_s_own_motion_is_no_temperature() {
        let w = [TAU / 0.7, TAU / 2.3, TAU / 23.];
        let times: Vec<f64> = (0..31).map(|k| k as f64).collect();
        let plan = Plan::new(&times, w);
        let tau = 0.37;
        let mut cuts = vec![0.];
        for i in 0.. {
            let (start, step_end) = (i as f64 * tau, crate::linalg::smaller((i + 1) as f64 * tau, plan.end));
            if i % 3 == 2 && start + 0.43 * tau < step_end {
                cuts.push(start + 0.43 * tau);
            }
            cuts.push(step_end);
            if step_end >= plan.end {
                break;
            }
        }
        // Each atom's own numbers, a fixed sequence: amplitudes, phases and kick sizes.
        let draw = |i: usize, j: usize| -> f64 { ((i * 7919 + j * 104_729 + 13) as f64 * 0.618_033_988_749_895).fract() };
        let atoms = 64;
        let own = |i: usize, a: usize, t: f64| -> f64 {
            let kicks = (cuts.partition_point(|&c| c <= t) - 1) as f64;
            (0.5 + draw(i, a)) * (w[a] * t + TAU * draw(i, a + 3)).cos() + 0.05 * (draw(i, a + 6) - 0.5) * kicks
        };
        let shared: [&dyn Fn(usize, f64) -> f64; 3] = [&|_, _| 0., &|_, t| 0.6 * t, &|a, t| 5. * (0.86 * w[a] * t + 0.4).sin()];
        let rule = crate::special::gauss_legendre(60);
        let read = |path: &dyn Fn(usize, f64) -> f64, shapes: bool| -> Vec<Estimate> {
            let mut groups: Vec<Group> = (0..8).map(|_| Group::new(times.len())).collect();
            for i in 0..atoms {
                let kappa = |t: f64| -> V3 { std::array::from_fn(|a| own(i, a, t) + path(a, t)) };
                // The path is continuous within a piece (a kick lands at a piece's start): a piece's rule from just
                // inside it.
                let nodes = |from: f64, to: f64| -> Vec<(f64, f64, V3)> {
                    rule.iter().map(|&(x, weight)| { let t = 0.5 * (from + to) + 0.5 * (to - from) * x; (t, 0.5 * (to - from) * weight, kappa(t)) }).collect()
                };
                let along = |from: f64, at: f64| Integrals::of(from, nodes(from, at));
                let mut windows = Windows::new(&plan.times, &plan.halves);
                let at = |t: f64| Fate::Held(Held { rows: [0.; STATES], kappa: kappa(t), kappa2: kappa(t).map(|k| k * k), shape: NO_SHAPE, unresolved: i % 7 == 3 });
                windows.record(0, at(0.));
                let mut next = 1;
                for piece in cuts.windows(2) {
                    windows.add(piece[0], piece[1], &|at| along(piece[0], at));
                    while next < times.len() && times[next] <= piece[1] {
                        windows.record(next, at(times[next]));
                        next += 1;
                    }
                }
                let mut fates = windows.close();
                if !shapes {
                    for f in fates.iter_mut() {
                        if let Fate::Held(h) = f {
                            h.shape = NO_SHAPE;
                        }
                    }
                }
                groups[i % 8].add(&fates);
            }
            summarize(&groups, &Scales { hbar_over_mass: KT_PER_UK, frequencies: w })
        };
        let (alone, alone_before) = (read(shared[0], true), read(shared[0], false));
        let mut worst = 0_f64;
        for (case, path) in shared.iter().enumerate().skip(1) {
            let (with, before) = (read(*path, true), read(*path, false));
            for k in 1..times.len() {
                let (m, a0, b, b0) = (with[k].motion.unwrap(), alone[k].motion.unwrap(), before[k].motion.unwrap(), alone_before[k].motion.unwrap());
                for a in 0..3 {
                    let h = plan.halves[k][a];
                    let (lo, hi) = (times[k] - h, times[k] + h);
                    // The shared path g and the atoms' own mean ū over the window by a dense rule on its pieces.
                    let mut ends = vec![lo];
                    ends.extend(cuts.iter().copied().filter(|&c| c > lo && c < hi));
                    ends.push(hi);
                    let (mut g1, mut g2, mut u1, mut gu) = (0., 0., 0., 0.);
                    for piece in ends.windows(2) {
                        for &(x, wt) in rule.iter() {
                            let (t, wt) = (0.5 * (piece[0] + piece[1]) + 0.5 * (piece[1] - piece[0]) * x, 0.5 * (piece[1] - piece[0]) * wt / (hi - lo));
                            let (g, u) = (path(a, t), (0..atoms).map(|i| own(i, a, t)).sum::<f64>() / atoms as f64);
                            (g1, g2, u1, gu) = (g1 + wt * g, g2 + wt * g * g, u1 + wt * u, gu + wt * g * u);
                        }
                    }
                    let (variance, covariance) = (g2 - g1 * g1, gu - g1 * u1);
                    // The paths' scale: the atoms' spread and the shared path's mean square over the window.
                    let scale = a0.temperature[a] + g2;
                    worst = worst.max((m.temperature[a] - a0.temperature[a]).abs() / scale);
                    assert!((m.temperature[a] - a0.temperature[a]).abs() <= 1e-9 * scale, "case {case}, output {k}, axis {a}: T {} with the shared path against {} without (its variance {variance})", m.temperature[a], a0.temperature[a]);
                    assert!((m.temperature_errors[a].error - a0.temperature_errors[a].error).abs() <= 1e-9 * scale, "case {case}, output {k}, axis {a}: its error {} against {}", m.temperature_errors[a].error, a0.temperature_errors[a].error);
                    let moved = b.temperature[a] - b0.temperature[a];
                    assert!((moved - variance - 2. * covariance).abs() <= 1e-9 * scale, "case {case}, output {k}, axis {a}: with no shapes T moved by {moved} against {variance} + 2 × {covariance}");
                }
                // The unresolved atoms: the same share of the held atoms, what T's bound is booked at.
                assert!(with[k].unresolved == alone[k].unresolved && alone[k].unresolved > 0., "case {case}, output {k}: the unresolved share {} with the shared path against {} without", with[k].unresolved, alone[k].unresolved);
            }
        }
        println!("T with a shared path against T without it: at most {worst:.2e} of the paths' scale");
        // The shared paths are large against the spread: the largest window variance over the atoms' T.
        let sloshing = read(shared[2], false);
        let most = (1..times.len()).map(|k| sloshing[k].motion.unwrap().temperature[2] / alone[k].motion.unwrap().temperature[2]).fold(0., f64::max);
        assert!(most > 10., "the sloshing's reading with no shapes is {most} times the spread at the most");
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
            fixed: fixed_of(vec![[0.; 2]; 2]),
            step: Bound::share(),
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let shape = Shape::of(&context.plan, false);
        let record = |i: u64| {
            let held = Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], shape: NO_SHAPE, unresolved: false };
            Record { outputs: vec![Fate::Held(held.clone()), Fate::Held(held.clone())], reads: vec![Fate::Held(held.clone()), Fate::Held(held)], listed: vec![Some(([0.; 3], [i as f64; 3])); shape.listed], trail: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2], kinetic: 0., end: (0, 0., 0.) }
        };
        let whole = [folded(&shape, (0..GROUP).map(record)), folded(&shape, (GROUP..2 * GROUP).map(record))];
        let scout: Vec<Part> = (0..3).map(|i| folded(&shape, std::iter::once(record(i)))).collect();
        let mut sent = vec![];
        let mut send = |m: Message| if let Message::Checkpoint(s, _) = m { sent.push(s.trajectories) };
        let mut messages = Messages { send: &mut send, clock: Instant::now() - CHECKPOINT, last: None, least: 0, stage: None };
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

    /// The stages (module documentation, # Steps): one as a step begins and as the check moves to another rung, none
    /// where nothing changed -- the same stage again, a rung the step stands at, a rung before any step --; a stage
    /// sets no result's clock.
    #[test]
    fn a_stage_goes_out_when_it_changes_and_never_again() {
        let mut sent = vec![];
        let mut send = |m: Message| if let Message::Stage(s) = m { sent.push(s.clone()) };
        let mut messages = Messages { send: &mut send, clock: Instant::now() - CHECKPOINT, last: None, least: 0, stage: None };
        let at = |step, tau| Stage { step, tau: Some(tau), batch: None };
        let batch = |number, trajectories, want| Stage { step: Step::Product, tau: Some(0.5), batch: Some(Batch { number, trajectories, want, worst: Some(1.5) }) };
        messages.rung(1.);
        messages.stage(at(Step::Scout, 1.));
        messages.stage(at(Step::Scout, 1.));
        messages.rung(0.5);
        messages.rung(0.5);
        messages.stage(at(Step::Pilot, 0.5));
        messages.stage(batch(1, 88, 176));
        messages.stage(batch(2, 176, 352));
        assert!(messages.due().is_zero(), "a stage sets no clock: the result stays due");
        assert_eq!(sent, [at(Step::Scout, 1.), at(Step::Scout, 0.5), at(Step::Pilot, 0.5), batch(1, 88, 176), batch(2, 176, 352)]);
    }

    /// A solve's stages in order (module documentation, # Steps), on R1_TWEEZER's own request over 20 µs, solved twice:
    /// the first a new selection -- Model, then G's Scout and Pilot, Select, and either Product (G chosen) or Model, Scout,
    /// Pilot and Product again --, the second on the choice kept -- Select, Model, Scout, Pilot, Product --; each stage
    /// another than the one before, the steps (consecutive stages of one step taken once) in that order, Select and Model
    /// at no rung and every other stage at one; no checkpoint before the first
    /// Scout. A batch brings the run beyond the trajectories in hand, its numbers count from 1, and its looks stand beyond
    /// their targets (else no batch).
    #[test]
    fn a_solve_says_its_steps_in_order() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R1_TWEEZER.json");
        let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        (line["options"]["durationUs"], line["options"]["samples"]) = (serde_json::json!(20.), serde_json::json!(21));
        let request = Request::from_native(&line).unwrap();
        let fresh = crate::select::kept(&request).is_none();
        for new in [fresh, false] {
            // Each message as it came: a stage, or a checkpoint (None).
            let mut came: Vec<Option<Stage>> = vec![];
            let solved = solve(&request, &mut |m| came.push(match m {
                Message::Stage(s) => Some(s.clone()),
                Message::Checkpoint(..) => None,
            }));
            solved.unwrap();
            let stages: Vec<&Stage> = came.iter().flatten().collect();
            eprintln!("{} messages; stages {stages:?}", came.len());
            assert!(stages.windows(2).all(|w| w[0] != w[1]), "a stage again");
            let mut steps: Vec<Step> = stages.iter().map(|s| s.step).collect();
            steps.dedup();
            use Step::*;
            if new {
                assert!(steps == [Model, Scout, Pilot, Select, Product] || steps == [Model, Scout, Pilot, Select, Model, Scout, Pilot, Product], "{steps:?}");
            } else {
                assert_eq!(steps, [Select, Model, Scout, Pilot, Product]);
            }
            assert!(stages.iter().all(|s| matches!(s.step, Select | Model) == s.tau.is_none()), "a rung where none is, or none where one is");
            let scout = came.iter().position(|m| m.as_ref().is_some_and(|s| s.step == Scout)).unwrap();
            assert!(came[..scout].iter().all(Option::is_some), "no checkpoint before the scout");
            let batches: Vec<&Batch> = stages.iter().filter_map(|s| s.batch.as_ref()).collect();
            assert!(batches.iter().enumerate().all(|(i, b)| b.number == i as u64 + 1 && b.want > b.trajectories && b.worst.is_none_or(|w| w > 1.)), "{batches:?}");
        }
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
            fixed: fixed_of(vec![[0.; 2]; 2]),
            step: Bound::share(),
            checked: None,
            ledger: Ledger::default(),
            method: String::new(),
        };
        let shape = Shape::of(&context.plan, false);
        let held = |i: u64| Fate::Held(Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], shape: NO_SHAPE, unresolved: false });
        let nowhere = |i: u64| Record {
            outputs: vec![held(i), Fate::Failed],
            reads: vec![held(i), Fate::Failed],
            listed: vec![None; shape.listed],
            trail: vec![],
            lost: None,
            counts: Counts { nowhere: Stopped::at(0.), ..Counts::default() },
            scattered: [0.; 2],
            kinetic: 0.,
            end: (i, 0., 0.),
        };
        let none = [folded(&shape, (0..GROUP).map(nowhere)), folded(&shape, (GROUP..2 * GROUP).map(nowhere))];
        let all: Vec<&Part> = none.iter().collect();
        assert!(!integrated(&all));
        let mut sent = vec![];
        let mut send = |m: Message| if let Message::Checkpoint(s, _) = m { sent.push(s.trajectories) };
        let mut messages = Messages { send: &mut send, clock: Instant::now() - CHECKPOINT, last: None, least: 0, stage: None };
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
            Record { outputs: vec![Fate::Lost], reads: vec![Fate::Lost], listed: vec![], trail: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2], kinetic: 0., end: (0, 0., 0.) }
        };
        let mut silent = |_: Message| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now() - CHECKPOINT, last: None, least: 0, stage: None };
        let numbers: Vec<u64> = (0..rayon::current_num_threads() as u64).collect();
        let mut seen = vec![];
        let levels = run_groups(&[Hand { groups: &numbers, members: 0..GROUP, shape, record: &record, work: 1. }], 0, &mut messages, &mut |_, messages| {
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
    /// reference state) and that state's prepared trap, as the solve makes them (`solve`, `run_of`, `Prepared`).
    fn scene_run(name: &str, requests: bool, state: usize) -> (Request, Selected, Trap) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join(if requests { "requests" } else { "scenes" }).join(format!("{name}.json"));
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let run = run_of(&request, alone(state)).unwrap();
        let trap = Prepared::new(&run, alone(state)).unwrap().trap;
        (request, run, trap)
    }

    /// The initial ground state of a request of the largest population (the solve's reference state).
    fn chief(request: &Request) -> usize {
        (0..GROUND).max_by(|&x, &y| request.initial_density[(x, x)].re.total_cmp(&request.initial_density[(y, y)].re)).unwrap()
    }

    /// The rms speed along an axis of a request's atoms at its temperature, as a momentum over ħ (1/µm).
    fn thermal(request: &Request, run: &Selected) -> f64 {
        (KT_PER_UK * request.temperature_uk / run.hbar_over_mass()).sqrt()
    }

    /// A step's count against its time (a measurement, ignored; tilt0's request where the scratchpad holds it): states of
    /// three of tilt0's atoms -- the trap's centre at the thermal speed along x, y and z, run at its step τ -- every two
    /// microseconds over their first 60 µs; from each, 4τ of the run as the run takes it at τ (four steps) and at 4τ
    /// (one), each step made and taken on one thread (`Midway::new` and `attempt`, a scratch carried along each level, no
    /// decision), timed, and counted as the ladder prices it (`Price::parts`, the jumps' `Trajectory::jumped`). Per
    /// level its wall time and its count by part; the prof regions after each (cumulative; nothing without the feature).
    #[test]
    #[ignore]
    fn a_step_s_count_against_its_time() {
        let file = r"C:\Users\eadri\AppData\Local\Temp\claude\c0815e347c54866c\8308d66c-699b-4483-ab9b-7305d93a9f7e\scratchpad\final\req\tilt0.json";
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(file).unwrap()).unwrap()).unwrap();
        let b = chief(&request);
        let run = run_of(&request, alone(b)).unwrap();
        let trap = Prepared::new(&run, alone(b)).unwrap().trap;
        let domain = Domain { widest: Widest::of(&request) };
        let (speed, tau) = (thermal(&request, &run), 0.32610161451737907);
        let mut states = vec![];
        for (index, axis) in [(3u64, 0usize), (5, 1), (7, 2)] {
            let mut kappa = [0.; 3];
            kappa[axis] = speed;
            let mut t = Trajectory::new(&run, b, trap.centre, kappa, SEED, index);
            let mut path = adapt::Path::new(SEED, index, 4. * tau, 2);
            let mut scratch = Scratch::default();
            while t.time < 60. && t.lost.is_none() {
                let stop = t.time + 2.;
                if adapt::advance(&run, &domain, tau, &mut t, &mut scratch, &mut path, 1000., stop, &[], &mut |_, _, _| {}, &mut |_, _| {}).is_err() {
                    break;
                }
                states.push(t.clone());
            }
        }
        eprintln!("{} states{}", states.len(), crate::prof::report());
        for (h, count) in [(tau, 4), (4. * tau, 1)] {
            let mut scratch = Scratch::default();
            let (mut wall, mut parts, mut jumps, mut steps) = (0.0, [0.0; 6], 0.0, 0);
            for s in &states {
                let mut t = s.clone();
                for _ in 0..count {
                    let started = Instant::now();
                    let Ok(m) = crate::step::Midway::new(&run, &t, h) else { break };
                    let Ok(step) = m.attempt(&t, &crate::step::Brownian::STILL, &[], &mut scratch, &domain) else { break };
                    wall += started.elapsed().as_secs_f64();
                    for (_, price) in &step.prices {
                        for (p, x) in parts.iter_mut().zip(price.parts(price.costs)) {
                            *p += x;
                        }
                    }
                    jumps += step.end.jumped - t.jumped;
                    t = step.end;
                    steps += 1;
                }
            }
            let total = parts.iter().sum::<f64>() + jumps;
            eprintln!(
                "COUNT h {h:.4} steps {steps}: wall {:.4} s, {:.2} µs a step; count {total:.4e} ({:.3} ns a count); parts [propagation, impulse, hazard, optical, end, model] {:?}, jumps {jumps:.4e}{}",
                wall,
                1e6 * wall / steps as f64,
                1e9 * wall / total,
                parts.map(|x| format!("{x:.3e}")),
                crate::prof::report()
            );
        }
    }

    /// Whole solves (a measurement, ignored; R2-LADDER item 3): the scenes' requests -- R2LADDER_REQ a folder of
    /// GMD1C_1ms, PGC3D_1ms, TET_1ms, R4_LIN_4, tilt0 and tilt2 .json, R2LADDER_S1 the app's default request (S1_5ms.json)
    /// --, R2LADDER_SCENES some of them (comma separated): each solved through, one at a time (`solve`), every message
    /// with its time, trajectories, step and the time it says is left, and the result's step, trajectories and wall
    /// time. With the feature `prof` the ladder's checks are noted as they are made (`PROFNOTE ladder ...`).
    #[test]
    #[ignore]
    fn r2ladder_solve() {
        let dir = std::path::PathBuf::from(std::env::var("R2LADDER_REQ").expect("R2LADDER_REQ: the requests' folder"));
        let s1 = std::env::var("R2LADDER_S1").ok();
        let files = [("GMD1C", "GMD1C_1ms.json"), ("PGC3D", "PGC3D_1ms.json"), ("TET", "TET_1ms.json"), ("R4", "R4_LIN_4.json"), ("tilt0", "tilt0.json"), ("tilt2", "tilt2.json"), ("S1", "S1")];
        let only = std::env::var("R2LADDER_SCENES").ok();
        for (name, file) in files.into_iter().filter(|(name, _)| only.as_deref().is_none_or(|o| o.split(',').any(|x| x == *name))) {
            let path = if file == "S1" { std::path::PathBuf::from(s1.clone().expect("R2LADDER_S1: the app's default request")) } else { dir.join(file) };
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            if let Some(forms) = line["model"]["motion"]["outsideForms"].as_array_mut() {
                for form in forms {
                    form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
                }
            }
            let request = Request::from_native(&line).unwrap();
            let clock = Instant::now();
            println!("R2-SOLVE {name} begins; threads {}", rayon::current_num_threads());
            let solved = solve(&request, &mut |m| {
                if let Message::Checkpoint(s, remaining) = m {
                    println!("R2-SOLVE {name} {:.1} s: {} trajectories, tau {} us, remaining {remaining:?}", clock.elapsed().as_secs_f64(), s.trajectories, s.time_step_us);
                }
            });
            match solved {
                Ok(s) => println!("R2-SOLVE {name} solved at {:.1} s: tau {} us, {} trajectories; {}", clock.elapsed().as_secs_f64(), s.time_step_us, s.trajectories, s.method),
                Err(e) => println!("R2-SOLVE {name} refused at {:.1} s: {e}", clock.elapsed().as_secs_f64()),
            }
        }
    }

    /// A rung's three reads on the same pairs (a measurement, ignored; R2-LADDER item 2's record): R2LADDER_SCENES' requests
    /// (as `r2ladder_solve`), the pilot's groups -- their first R2LADDER_MEMBERS members (all eight) -- run at τ/2, τ and
    /// 2τ, τ the first rung times R2LADDER_RUNG (a power of one half; 1), set up as the solve sets its attempt up (the
    /// plan, its cells and analysis candidates, the ensemble, the domain): the read against 2τ as it is, the same read
    /// with 2τ's halved steps taken back (`read_step`), and the read against τ/2 (module documentation, ## A companion at
    /// τ/2), each look's value and standard error over its bound, with the share of 2τ's and of τ's time on halved steps
    /// and the factors.
    #[test]
    #[ignore]
    fn r2ladder_companions() {
        let dir = std::path::PathBuf::from(std::env::var("R2LADDER_REQ").expect("R2LADDER_REQ: the requests' folder"));
        let s1 = std::env::var("R2LADDER_S1").ok();
        let members: u64 = std::env::var("R2LADDER_MEMBERS").map_or(GROUP, |m| m.parse().unwrap());
        let rung: f64 = std::env::var("R2LADDER_RUNG").map_or(1., |r| r.parse().unwrap());
        let files = [("GMD1C", "GMD1C_1ms.json"), ("PGC3D", "PGC3D_1ms.json"), ("TET", "TET_1ms.json"), ("R4", "R4_LIN_4.json"), ("tilt0", "tilt0.json"), ("tilt2", "tilt2.json"), ("S1", "S1")];
        let only = std::env::var("R2LADDER_SCENES").expect("R2LADDER_SCENES");
        for (name, file) in files.into_iter().filter(|(name, _)| only.split(',').any(|x| x == *name)) {
            let path = if file == "S1" { std::path::PathBuf::from(s1.clone().expect("R2LADDER_S1")) } else { dir.join(file) };
            let mut line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            line["options"]["device"] = serde_json::json!("cpu");
            if let Some(forms) = line["model"]["motion"]["outsideForms"].as_array_mut() {
                for form in forms {
                    form.as_object_mut().unwrap().entry("transitions").or_insert(serde_json::json!([]));
                }
            }
            let request = Request::from_native(&line).unwrap();
            let clock = Instant::now();
            // As `solve_on` and `attempt` set the attempt up.
            let run = run_of(&request, alone(chief(&request))).unwrap();
            let hm = hbar_over_mass(request.mass_kg);
            let initial = components(&request);
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&run, alone(s)).unwrap()).collect();
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
            let lead = initial.iter().position(|&(s, _)| s == chief(&request)).unwrap();
            let trap = &wells[lead].trap;
            let report = reported(&run, &alone(initial[lead].0)).unwrap_or_else(|_| trap.clone());
            let scales = Scales { hbar_over_mass: hm, frequencies: report.lab };
            let plan = Plan::new(&request.times_us, report.lab);
            let kinetic = kinetic_uk(trap, request.temperature_uk).iter().sum::<f64>() / 3.;
            let (frequency, end) = (trap.frequencies.iter().copied().fold(0., crate::linalg::larger), plan.end);
            let kick = kick_curvature(&run, trap);
            let first_rung = Ladder { run: end, prior: prior(pattern_wave_number(run.light()), hm, kinetic, &scales), frequency, kick }.first().unwrap();
            let above = 4. * first_rung <= end && stable(2. * first_rung, frequency, kick) && stable(4. * first_rung, frequency, kick);
            let domain = Domain { widest: Widest::of(&request) };
            let candidates = grid(end, first_rung);
            let mut plan = Plan { cell: if above { 4. } else { 2. } * first_rung, ..plan };
            plan.read_at(candidates.clone());
            let first = MOST_TRAJECTORIES / GROUP;
            let numbers: Vec<u64> = (first..first + pilot_size() / GROUP).collect();
            let mut silent = |_: Message| {};
            let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0, stage: None };
            let tau = first_rung * rung;
            let [half, own, double] = [0.5 * tau, tau, 2. * tau].map(|step| level(&run, &ensemble, &domain, &plan, step, &numbers, 0..members, false, &mut messages, &mut |_, _| {}));
            let [half, own, double] = [&half, &own, &double].map(|level| level.iter().collect::<Vec<&Part>>());
            let (times, window) = (&plan.reads, plan.window);
            let reads = [read_as_it_is(times, window, &own, &double, &scales), read_step(times, window, &own, &double, &scales), carried(&read_step(times, window, &half, &own, &scales), 2.)];
            let share = |level: &[&Part]| level.iter().map(|p| p.counts.halved.time).sum::<f64>() / level.iter().map(|p| p.counts.time).sum::<f64>();
            println!("R2-COMPANIONS {name} tau {tau} us, {} trajectories a level, {:.1} s: 2τ's time on halved steps {:.4e} (factor {:.5}), τ's {:.4e} (factor {:.5}), τ/2's {:.4e}", own.len() as u64 * members, clock.elapsed().as_secs_f64(), share(&double), amplified(double.iter().copied()), share(&own), amplified(own.iter().copied()), share(&half));
            let looks: Vec<Vec<(Read, f64)>> = reads.iter().map(judged).collect();
            for k in 0..looks[0].len() {
                let b = looks[0][k].1;
                let [a, c, h] = [0, 1, 2].map(|i| looks[i][k].0);
                println!("R2-COMPANIONS {name} look {k}: as it is {:+.4} ± {:.4} b, taken back {:+.4} ± {:.4} b, against τ/2 {:+.4} ± {:.4} b", a.value / b, a.spread.error / b, c.value / b, c.spread.error / b, h.value / b, h.spread.error / b);
            }
        }
    }

    /// The price a step's check weighs its courses by is the run's count (`work_at`): R4_LIN_2's explicit model below,
    /// at states of its atoms at the trap's centre at rest and at one to three times the thermal speed along each lab
    /// axis, held (no jump, each its own representation: its start's), at two steps -- a trajectory's work a µs priced at
    /// every state is that state's step's own count (`Step::work`, its scratch primed as a run's) over its length and its
    /// jumps' counted work a µs (`Pick::jumping`), the mean over the states by their weights, to the rounding. The step is
    /// the run's from the state: where the scheme refuses it, at its half, and so on (adapt.rs `halving`) -- at 0.4335 µs
    /// the decomposition refuses one state's step, and the price is that of its half (before, it was the price without
    /// bound, and a rung with such a step was left whatever its read).
    ///
    /// The model is the one the test was written on, named by its states: the ground states and the excited states 20,
    /// 21 and 23 to 31, the selection's run model of R4_LIN_2 until 429c36cd. Since then the selection gives the solve
    /// the full-ground model G to run its pilot on (select.rs `choose`, the chain's first; `run_of`), P chosen later on
    /// the pilot's paths -- and G, every excited state folded, has no step here that the scheme refuses: on it the
    /// halving and the refused attempts' work this test reads would go unread (2.558·10⁶ a µs at 0.2 µs, every step
    /// whole at 0.4335 µs).
    #[test]
    fn a_state_s_price_is_its_step_s_count() {
        let (request, _, _) = scene_run("R4_LIN_2", false, 0);
        let scene = crate::scene::Scene::new(&request).unwrap();
        let selection = crate::select::choose(&scene).unwrap();
        let keep: Vec<usize> = (0..GROUND).chain([20, 21]).chain(23..STATES).collect();
        let run = Selected::new(scene, Selection { keep, ..selection }, alone(0)).unwrap();
        let trap = Prepared::new(&run, alone(0)).unwrap().trap;
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
        for (tau, halved) in [(0.2, 0), (0.4335, 1)] {
            // The step from a state as the advance takes it: refused, at its half, and so on; its count and the refused
            // attempts' work on the way (step.rs `Scratch::refused`) over its length.
            let counted = |from: &Trajectory| {
                let (mut length, mut refused) = (tau, 0.);
                loop {
                    let (through, mut scratch) = (crate::step::Midway::new(&run, from, length).unwrap(), Scratch::default());
                    match through.attempt(from, &crate::step::Brownian::STILL, &[], &mut scratch, &widest).and_then(|_| through.attempt(from, &crate::step::Brownian::STILL, &[], &mut scratch, &widest)) {
                        Ok(step) => return ((step.work + refused) / length, length, refused),
                        Err(refusal) => {
                            assert!(adapt::halves(&refusal), "{tau}: {refusal:?}");
                            refused += scratch.refused();
                            length *= 0.5;
                        }
                    }
                }
            };
            let steps: Vec<(f64, f64, f64)> = picks.iter().map(|p| counted(&p.state.trajectory())).collect();
            let made = over_the_run(&picks, picks.iter().zip(&steps).map(|(p, (rate, _, _))| rate + p.jumping), None);
            let [priced, factor] = work_at(&run, widest, &picks)(tau);
            let lengths: Vec<f64> = steps.iter().map(|s| s.1).collect();
            // The level's halved steps as the states foretell them: each state's step weighed by its time held.
            let (shortfall, time) = picks.iter().zip(&lengths).fold((0., 0.), |(s, t), (p, &l)| (s + p.weight * l * (1. - (l / tau).powi(ORDER)), t + p.weight * l));
            eprintln!("R4_LIN_2 at {tau} µs: priced {priced:.6e}, counted {made:.6e} a µs; the steps taken at {lengths:?}, refused work {:?}; factor {factor}", steps.iter().map(|s| s.2).collect::<Vec<_>>());
            assert!((priced - made).abs() <= 1e-12 * made, "{tau}: {priced} against {made}");
            assert_eq!(lengths.iter().filter(|&&l| l < tau).count(), halved, "{tau}: {lengths:?}");
            assert!(steps.iter().all(|s| (s.1 < tau) == (s.2 > 0.)), "{tau}: a refused attempt did work, a step taken whole none");
            assert!((factor - factor_of(shortfall, time)).abs() <= 1e-12 * factor && (halved == 0) == (factor == 1.), "{tau}: {factor}");
        }
    }

    /// The groups come back in their order whatever the threads do: 37 groups' records (five of a pilot's numbers, then a
    /// product's) -- each trajectory's own, its rows marking its number -- folded by `run_groups` equal the same records
    /// folded one after another; so do the levels of one hand-out, each with its own members of the groups, the one
    /// foretold the dearer handed out first (`Hand::work`) and returned in its place.
    #[test]
    fn the_groups_come_back_in_order() {
        let shape = Shape { outputs: 2, reads: 2, listed: 1, keep: false };
        let record = |i: u64| {
            let mut rows = [0.; STATES];
            rows[(i % STATES as u64) as usize] = 1.;
            let held = Held { rows, kappa: [i as f64; 3], kappa2: [(i * i) as f64; 3], shape: NO_SHAPE, unresolved: false };
            let outputs = vec![Fate::Held(held.clone()), if i % 3 > 0 { Fate::Held(held) } else { Fate::Lost }];
            Record { reads: outputs.clone(), outputs, listed: vec![Some(([i as f64; 3], [0.; 3]))], trail: vec![], lost: None, counts: Counts { steps: i, ..Counts::default() }, scattered: [0.; 2], kinetic: 0., end: (0, 0., 0.) }
        };
        let mut silent = |_: Message| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0, stage: None };
        let numbers: Vec<u64> = (2048..2053).chain(5..37).collect();
        let members = [0..GROUP, 1..3];
        // The second level foretold the dearer: handed out first, returned in its place.
        let hands = members.clone().map(|members| Hand { groups: &numbers, work: 1. + members.start as f64, members, shape, record: &record });
        let levels = run_groups(&hands, 0, &mut messages, &mut |_, _| {});
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
            Record { reads: vec![Fate::Lost; 2], outputs: vec![Fate::Lost; 2], listed: vec![None], trail: vec![], lost: None, counts: Counts::default(), scattered: [0.; 2], kinetic: 0., end: (0, 0., 0.) }
        };
        let mut silent = |_: Message| {};
        let mut messages = Messages { send: &mut silent, clock: Instant::now(), last: None, least: 0, stage: None };
        let started = Instant::now();
        let hands = [Hand { groups: &numbers, members: 0..GROUP, shape, record: &record, work: 1. }];
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run_groups(&hands, 0, &mut messages, &mut |_, _| {})));
        let took = started.elapsed();
        let message = outcome.err().and_then(|payload| payload.downcast_ref::<String>().cloned()).unwrap_or_default();
        eprintln!("the hand-out ended after {took:?}: {message}");
        let whole = each * (numbers.len() as u32 * GROUP as u32).div_ceil(rayon::current_num_threads() as u32);
        assert!(message.contains("a trajectory's defect at 16387") && took < whole / 2, "{took:?} against {whole:?}: {message}");
        let again = run_groups(&[Hand { groups: &numbers[1..2], members: 0..GROUP, shape, record: &record, work: 1. }], 0, &mut messages, &mut |_, _| {});
        assert_eq!(again[0].len(), 1);
    }

    /// The initial ensembles of the conservative reference traps (refs/scenes R1, the app's own requests: a lone tweezer and a
    /// σ⁺ tweezer) in the run's prepared wells (`Prepared`: the trap light's potential) under their envelopes
    /// (`Potential::peak`): 2¹² draws each read the exact moments (ħ/m)⟨κ_a²⟩ = Σ_b p_b η_b k_BT (S_b²)_aa per lab axis within their
    /// intervals over the groups at the confidence z -- six looks, three axes of two traps -- and the quadrature's own
    /// error as the solve books it for T(0), its last two levels' relative difference (`Fixed::start`), here on 2¹⁶
    /// evaluations, fewer than a pilot's steps in these traps (88 trajectories of 800 steps or more). η lies in (0, 1]:
    /// the lowered isothermal's kinetic temperature, below the Boltzmann one.
    #[test]
    fn the_reference_traps_draw_their_exact_moments() {
        let names = ["R1_TWEEZER", "R1_SIGMA_P"];
        for name in names {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
            let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let request = Request::from_native(&line).unwrap();
            let run = run_of(&request, alone(chief(&request))).unwrap();
            let initial = components(&request);
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&run, alone(s)).unwrap()).collect();
            let hm = hbar_over_mass(request.mass_kg);
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
            let kinetic = ensemble.kinetic(1 << 16);
            assert!(kinetic.iter().all(|k| k.eta > 0. && k.eta <= 1.), "{name}: η {kinetic:?}");
            let exact = ensemble.momentum(&kinetic.iter().map(|k| k.eta).collect::<Vec<_>>());
            let quadrature = kinetic.iter().map(|k| k.difference / k.eta).fold(0., crate::linalg::larger);
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

    /// The draws' tilt statistic (ensemble.rs `Ensemble::draw_kinetic`; select.rs, # T's memory, read on the run's own
    /// trajectories) on the conservative reference traps (as `the_reference_traps_draw_their_exact_moments`), 2¹² draws
    /// each: K within the draw's depth (0 ≤ K ≤ y, so the score ⟨K⟩ − K bounded); with ⟨K⟩ = 3η/2 the classical
    /// ensemble's own (`Ensemble::kinetic`), the score s = ⟨K⟩ − K has none for its mean and E[K s] = −Var K, each within
    /// its interval over the groups at the confidence z (and the quadrature's error), four looks.
    #[test]
    fn the_draws_tilt_scores_their_kinetic_energy() {
        let names = ["R1_TWEEZER", "R1_SIGMA_P"];
        for name in names {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{name}.json"));
            let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
            let request = Request::from_native(&line).unwrap();
            let run = run_of(&request, alone(chief(&request))).unwrap();
            let initial = components(&request);
            let wells: Vec<Prepared> = initial.iter().map(|&(s, _)| Prepared::new(&run, alone(s)).unwrap()).collect();
            let hm = hbar_over_mass(request.mass_kg);
            let ensemble = Ensemble::new(&initial.iter().zip(&wells).map(|(&(s, p), w)| (s, p, w as &dyn Well)).collect::<Vec<_>>(), request.temperature_uk, hm).unwrap();
            let kinetic = ensemble.kinetic(1 << 16);
            // ⟨K⟩ over the components by their weights (K's mean is 3η/2 in each), and the quadrature's error alike.
            let weight: f64 = initial.iter().map(|c| c.1).sum();
            let expected = initial.iter().zip(&kinetic).map(|(c, k)| c.1 * 1.5 * k.eta).sum::<f64>() / weight;
            let quadrature = initial.iter().zip(&kinetic).map(|(c, k)| c.1 * 1.5 * k.difference).sum::<f64>() / weight;
            let k: Vec<f64> = (0..1 << 12).map(|i| ensemble.draw_kinetic(SEED, i).1).collect();
            assert!(k.iter().all(|&k| k >= 0. && k.is_finite()), "{name}: a kinetic energy out of its range");
            let mean = k.iter().sum::<f64>() / k.len() as f64;
            let variance = k.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (k.len() - 1) as f64;
            let looks: [(&str, f64, Box<dyn Fn(f64) -> f64>); 2] = [("s", 0., Box::new(|k| expected - k)), ("K s + Var K", quadrature * mean, Box::new(|k| k * (expected - k) + variance))];
            for (what, slack, x) in &looks {
                let groups: Vec<f64> = k.chunks(GROUP as usize).map(|members| members.iter().map(|&k| x(k)).sum::<f64>() / GROUP as f64).collect();
                let g = groups.len() as f64;
                let m = groups.iter().sum::<f64>() / g;
                let error = (groups.iter().map(|y| (y - m).powi(2)).sum::<f64>() / (g - 1.) / g).sqrt();
                let within = Spread { error, groups: g }.interval_over((looks.len() * names.len()) as f64) + quadrature + slack;
                eprintln!("{name} {what}: {m:.4e} ± {within:.2e} (⟨K⟩ {expected:.4}, mean {mean:.4}, Var {variance:.4})");
                assert!(m.abs() <= within, "{name} {what}: {m} beyond ± {within}");
            }
        }
    }

    /// The runaway's guard on the run (step.rs, # The domain): R2_RECOIL's atoms of |2,2⟩ through the trap's centre at 1
    /// to 1000 times the thermal speed along each lab axis, run for 0.8 µs at 0.005 µs in the solve's domain. An atom whose
    /// state's hazard at its first step's middle is above the fastest excited state's decay, wholly excited, stops there,
    /// is counted and held from then on as it stood, unresolved (booked whole), its rows numbers; one whose hazard is
    /// below half of it runs to the end untouched. The model is made once (P chosen at the checkpoint, model/selected.rs),
    /// so its rate is a state's: how many stop is said, not asked.
    #[test]
    fn a_fold_scattering_beyond_any_state_stops_its_atom_held_unresolved() {
        let (request, run, trap) = scene_run("R2_RECOIL", false, 7);
        let b = 7;
        let widest = Widest::of(&request);
        let domain = Domain { widest };
        let plan = Plan::new(&[0., 0.2, 0.4, 0.6, 0.8], trap.lab);
        let speed = thermal(&request, &run);
        let (mut stopped, mut ran) = (0, 0);
        for (i, factor) in [1., 10., 30., 100., 300., 1000.].into_iter().enumerate() {
            for a in 0..3 {
                let kappa: V3 = std::array::from_fn(|x| if x == a { factor * speed } else { 0. });
                let start = Trajectory::new(&run, b, trap.centre, kappa, SEED, (3 * i + a) as u64);
                let middle: V3 = std::array::from_fn(|x| trap.centre[x] + 0.5 * TAU_GUARD * run.hbar_over_mass() * kappa[x]);
                let excitation = {
                    let m = run.middle(start.cluster, middle, 0.);
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

    /// The loss is the first passage out of the bound set H = {E ≤ 0}, wherever the atom is (SPEC §1.7;
    /// model/selected.rs `lost`): R1_TWEEZER's atoms of its initial state at the trap's centre, along each lab axis at
    /// the speed whose kinetic energy is 1.01 and 0.9 times the depth there, run at 0.01 µs and read at 0, τ/2, τ and
    /// 2τ. The one above is held at t = 0 -- as every atom is, one drawn above the far value too (SPEC §1.7) --, held
    /// at τ/2 inside its first step, and lost at that step's end, its first test: a few nanometres from the focus, deep
    /// inside the trap's region (the old test held it until it left the region, 1.2 waists out). The one below is held
    /// through the run, to the last output's window.
    #[test]
    fn an_atom_above_the_escape_value_is_lost_at_its_first_test_wherever_it_is() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join("R1_TWEEZER.json");
        let b = chief(&Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap());
        let (request, run, trap) = scene_run("R1_TWEEZER", false, b);
        let domain = Domain { widest: Widest::of(&request) };
        let tau = 0.01;
        let mut plan = Plan::new(&[0., 0.5 * tau, tau, 2. * tau], trap.lab);
        plan.cell = 2. * tau;
        let depth = -run.prepared(&alone(b)).at(trap.centre).value;
        assert!(depth > 0., "the tweezer holds the state: {depth}");
        for (share, above) in [(1.01, true), (0.9, false)] {
            for a in 0..3 {
                let kappa: V3 = std::array::from_fn(|x| if x == a { (2. * share * depth / run.hbar_over_mass()).sqrt() } else { 0. });
                let record = run_trajectory(&run, &domain, Start { state: b, r: trap.centre, kappa }, a as u64, tau, &plan);
                let held: Vec<bool> = record.outputs.iter().map(|f| matches!(f, Fate::Held(_))).collect();
                let mut t = Trajectory::new(&run, b, trap.centre, kappa, SEED, a as u64);
                let mut path = adapt::Path::new(SEED, a as u64, plan.cell, (plan.cell / tau).log2().round() as u32);
                let advanced = adapt::advance(&run, &domain, tau, &mut t, &mut Scratch::default(), &mut path, plan.end, plan.end, &[], &mut |_, _, _| {}, &mut |_, _| {});
                assert!(advanced.is_ok(), "{share} of the depth along {a}: {advanced:?}");
                if above {
                    assert_eq!(held, [true, true, false, false], "{share} of the depth along {a}");
                    assert_eq!((t.lost, record.counts.time), (Some(tau), tau), "lost at its first step's end along {a}");
                    let off = t.r.iter().zip(&trap.centre).map(|(x, c)| (x - c) * (x - c)).sum::<f64>().sqrt();
                    assert!(!run.light().outside(t.r) && off < 0.01, "{share} of the depth along {a}: lost {off} µm from the centre");
                } else {
                    assert!(held.iter().all(|&h| h) && t.lost.is_none() && record.counts.time == plan.end, "{share} of the depth along {a}: {held:?}, lost at {:?}", t.lost);
                }
            }
        }
    }

    /// The held atoms' kinetic energy's bound (`kinetic_bound`) is the deepest ground state's well at the focus, and no
    /// ground state's well is deeper anywhere: R1_TWEEZER and PGC3D_1ms, every ground state's prepared potential at
    /// 2000 random points in cubes of two waists and two Rayleigh lengths about the focus no lower than −K_max. PGC3D's
    /// is 1043 µK (DESIGN2 §2.3).
    #[test]
    fn a_held_atom_s_kinetic_energy_is_bounded_by_the_deepest_well() {
        for (name, requests) in [("R1_TWEEZER", false), ("PGC3D_1ms", true)] {
            let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join(if requests { "requests" } else { "scenes" }).join(format!("{name}.json"));
            let b = chief(&Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap());
            let (request, run, _) = scene_run(name, requests, b);
            let cap = kinetic_bound(&run);
            let deepest = (0..GROUND).map(|g| -run.prepared(&alone(g)).at([0.; 3]).value).fold(f64::NEG_INFINITY, f64::max);
            assert!(cap > 0. && cap == deepest, "{name}: {cap} against {deepest}");
            let tw = &request.beams[request.trap_beam];
            let (w, k) = (tw.waist.unwrap(), tw.wave_number);
            let zr = k * w * w / 2.;
            let mut s = crate::random::Stream::new(3, crate::random::Purpose::Initial, 0, 0);
            for i in 0..2000 {
                let reach = if i % 2 == 0 { 2. * w } else { 2. * zr };
                let r: V3 = std::array::from_fn(|_| reach * (2. * s.uniform() - 1.));
                for g in 0..GROUND {
                    let v = run.prepared(&alone(g)).at(r).value;
                    assert!(v >= -cap * (1. + 1e-12), "{name}: ground state {g} at {r:?}: {v} below −{cap}");
                }
            }
            eprintln!("{name}: K_max = {cap} rad/µs = {} µK", cap / KT_PER_UK);
            if requests {
                assert!((cap / KT_PER_UK - 1043.).abs() < 1., "{name}: {} µK", cap / KT_PER_UK);
            }
        }
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
        let domain = Domain { widest: Widest::of(&request) };
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
    /// representation's decisions at every cell's end) -- atoms held to the end and lost --, read at the candidates m·2τ, the checkpoints a fifth of the run
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
            let domain = Domain { widest: Widest::of(&request) };
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
        let held = Fate::Held(Held { rows: std::array::from_fn(|s| if s == 0 { 1. } else { 0. }), kappa: [0.; 3], kappa2: [1.; 3], shape: NO_SHAPE, unresolved: false });
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
    fn run_trajectory(run: &Selected, domain: &Domain, start: Start, index: u64, tau: f64, plan: &Plan) -> Record {
        super::run(run, domain, start, index, tau, plan, false)
    }

    /// A run is priced where its atoms were (`one_each`): a trajectory weighs by the time it holds its atom, and one lost
    /// before the first output gives its state at t = 0 for its own time. Eleven groups of eight, every atom lost at
    /// 0.1 to 0.9 of the first output's time (S2Q's first model), each lost before its first checkpoint after the start:
    /// every trajectory gives its state at t = 0, weighs by its time to the loss (`Record::end`, not its path's end, the
    /// start), and its jumps a µs are its work by then over that time; a mean over the run of a value reads that value,
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
                // Each lost before the first output: its path's end its last checkpoint, the start; its time held to
                // the loss, six jumps' work by then.
                let lost = (0..members).map(|m| (1, state(m, 0., 0))).collect();
                let ends = (0..members).map(|m| (MOST_TRAJECTORIES + (g * members + m) as u64, loss(m), 6.)).collect();
                Part { sums: Group::new(outputs + 1), reads: Group::new(outputs + 1), kinetic: Group::new(outputs + 1), listed: vec![], samples, trails: vec![], lost, ends, counts: Counts::default() }
            })
            .collect();
        let all: Vec<&Part> = parts.iter().collect();
        let picks = one_each(&all);
        assert_eq!(picks.len(), groups * members);
        for p in &picks {
            let end = parts[p.group].ends.iter().find(|e| e.0 == p.state.index).unwrap();
            assert_eq!((p.state.sample.at.t, p.weight, p.jumping), (0., end.1, 6. / end.1));
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
        let fixed = fixed_of(vec![[0.; 2], [0.0480, 0.]]);
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
                Fate::Held(Held { rows: state(0), kappa, kappa2, shape: NO_SHAPE, unresolved: false })
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
                fixed: fixed_of(vec![[0.; 2], [0.01, 0.02], [0.; 2]]),
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
