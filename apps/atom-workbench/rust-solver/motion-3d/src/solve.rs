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
//! the population target. The fold rule's per-item threshold is the a-priori share of it (`BUDGET`, ledger.rs): the
//! couplings' and the far components' folds (ground.rs), a Gaussian beam's intensity axis (model.rs).
//! The first order in the Doppler shift takes, per step, the step's share of the run's room beside the other estimates (`doppler_allowance`; the exact
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
//! over j ≤ i of output j's 2η² (the mean of its states over its neighbours as far as they agree, `steady_means`)
//! -- its error over the interval after j, where the larger of the interval's
//! two ends bounds it -- times the mean over the atoms held at i of e^{−n}, n an atom's jumps since the
//! end of that interval (`Memory`, the pilot's atoms): at least its own 2η², and at most the largest since
//! t = 0. To each output's 2η² it adds what the atoms read leave open of the light's resonance (`with_open`): the elimination's a-priori bound over their reach times the share
//! of atoms that may have met it unseen. (QuTiP's solvers control a step's local error and leave the global error, as an elimination's
//! validity, to the user's convergence checks.)
//!
//! An output's total estimated error is the ledger's total there (its η², its mean (kv/σ)² and the
//! profiles at the atoms with the model's estimates), the time step's bias (its step-doubling share)
//! and CONFIDENCE standard errors, in quadrature (# Shares): the populations' against the population target (absolute),
//! T's against the temperature target (relative, the regime's below it, # Regime) -- the fold's charge in either its own
//! (`fold_charges`: at fourth order the populations' S² and what the transient's delay leaves, T's still √2 S, the force folding at second order). The
//! trajectories are run until the totals meet the targets -- the statistics within what the ledger and the step leave
//! (# Shares) -- and an output whose total exceeds a target all the same is flagged (`Ledger::flagged`; a checkpoint's
//! statistics still run, so only its a-priori excess is).
//!
//! # The fold's choice
//! The rule's candidates (ground.rs `Ground::ladder`) from all folded down, and the
//! first's fourth order (on the GPU past its gate alone: # Device), are costed on the pilot's states carried into their clusters (`carried`): each
//! with its fold's charge as measured (`measured_charge`: per output the charge the run's ledger books, with
//! the memory). A candidate's cost is its step check's (below) on those states -- the step it would take --:
//! its counted work per interval there times the trajectories its rooms at that step's bias need and the weight
//! of the totals it leaves (`predicted`: per output, the pilot's statistics against what the batches take, the
//! statistic they bind on -- a mean charge over the outputs under-predicted the D1 gray molasses' need twice over,
//! its early transient), over the run: continuous in its estimates (# Shares). The least is taken,
//! one keeping more of the model (its fourth order, a rung) only for a saving beyond the need's own relative
//! error (`need_error`, as a finer step, `kept`); the ladder's rungs are costed only where the folded models'
//! cost would fall, without the estimates a rung can change (their totals at the pilot's step with only
//! `Candidate::fixed`), by more than that error and a rung's least check: below, no rung -- more of the model explicit,
//! more work per step, at least one doubling -- can win or pay for its check. A rung's own check, as the pilot's model's at
//! the rung's work, is spent only where what the rung's floor leaves of the least cost exceeds it: on a short run
//! the check, a trap period of the probes' orbits, is the whole run's work at equal steps. A folded model beyond its targets at any
//! step (a pole its tables do not resolve, a charge past the rooms) is costed at the pilot's step with no bias, its check put off
//! until it is the choice: what the step moves of that cost is little beside the weight. The run goes on from those states; no timing enters, so a request gives the same choice on any machine, but for the GPU's fourth order (# Device).
//! A pilot runs again only where it keeps no atom to the first output, its model's own estimates past the targets: no state of it measures a charge or a step, and an atom blown out at once by a folded drive's
//! unsaturated rate says the fold does not hold (a resonant probe of 100 mW/cm² with cooling of 30, every drive folded: all atoms lost within 1 us, its 3+12 rung none by 2 us and 63 % by 100 us). The next rung
//! takes the pilot's place, and the choice goes on from it; a model within its targets has lost its atoms by its physics (`Candidate::past_their_targets`).
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
//!   subsample of the states (one per pilot trajectory, more until the verdict stands, `accumulated_excess`) the jump-free flow over that time (at most
//!   the run) in steps of dt against steps of dt/2, the largest difference of the cluster's
//!   populations, their mean over them plus CONFIDENCE standard errors -- in a gray
//!   molasses the atoms wait 50-140 us in dark states, and the bound over-counts the step's error
//!   there several times;
//! - the step's light impulse, relative, within T's tolerance where T binds: over the states as T's target weighs them (`regime`'s w, `binding_at`), and over all of them
//!   within the T target itself (a step that leaves the motion unresolved still moves the loss). Pooled, it reads where the states are, not where T binds: a D1 gray molasses over 10 ms holds 70 % of its pilot's
//!   states cold, past 3 ms, and T binds within its first ms. At 0.588 us, over 16 pilots, the impulse reads 0.65-1.0 of T's tolerance pooled, 0.92-2.17 over 100-400 us and 0.94-1.78 weighted, where T is
//!   +5.6 ± 1.3 % against a tolerance of 3.9 %. The pilots reading lowest kept 0.71-0.77 us, T +13-20 % at 200-390 us where its target is 10-13 % (against 0.167 us, 8192 trajectories on shared streams);
//!   weighted, 64 pilots keep 0.42-0.63 us, T +6.1 ± 1.3 % at the largest and T's tolerance about at their median, 0.5 us;
//! - T's own step error, the orbits' (`orbits`): from states of the pilot's trajectories bound to the trap
//!   (the atoms whose T the step's error stays on: an unbound one leaves within a transit of the trap's
//!   region; taken as T's target weighs them, first one per trajectory, more until the verdict stands, `orbits`) the jump-free flow over one period of the trap's slowest axis, the span of the solve's T
//!   windows, in steps of dt against steps of dt/2, ψ kept normalized; per axis the mean κ² over it
//!   compared, the probes' mean relative difference plus CONFIDENCE standard errors, within T's
//!   tolerance. The integrator keeps the conservative motion's energy error bounded rather
//!   than accumulating, so that a step's energy error summed over the run (the former friction
//!   criterion) over-counts T's error many times -- the tweezer alone at 0.45 us 0.99 of its share
//!   against the orbits' 0.13, and the retro-reflected tweezer refined to 0.0076 us where T's measured
//!   error asks 0.03–0.05 us --;
//!   with the orbits, the impulse and the populations' criterion, the 3D PGC and the tetrahedral MOT
//!   hold T within 0.4 % and 1.0 % of a quarter of their steps at the outputs where it binds (4096
//!   trajectories on shared streams). A probe's difference counts at the share of atoms that keep that orbit over the period, the flow's no-jump survival ‖ψ‖² at its end, as a statistic stands at the share of atoms it reads (# Batches): a flow no atom follows is no atom's error, and a smaller step cannot pay for it. A flow ends where that share lies within the numerical share of the tolerance.
//!   (A probe at 1 W/cm²: its atoms wait 1.3 us for a jump and the period is 52 us. The flow ran on past them, its survival at the end at most 1e-127, 72 % of its probes past their loss at a median 3.8 us, κ² up a hundred times, and its pairs parted until the step resolved the light's pattern for atoms that fast, k v dt about 0.3:
//!   1.3, 3.0, 0.68, 0.92 and 1.04 tolerances from 0.25 to 0.022 us, 72 of the run's 88 s. Read at the share, under 0.01 of the tolerance at every step, and the step is the impulse's, 0.037 us. Where the atoms keep the orbit -- a trap alone, atoms dark through a period -- the share is 1 and the read the step's own: the shallow probe's T +3.9 % on its worst axis at 2.86 us, and +8.0 % at the 4 us the impulse would take, against 1 us, 4096 trajectories on shared streams);
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
//!   escaped share L by L ln(1/L) per relative error of it (`escape`) --, by the larger of the orbits'
//!   and the jumps' errors.
//! T's criteria take the step's share of T's room at T's target there (`ceiling`, # Regime), at most the T target itself
//! (`tolerances`): the share of a target the method's own error sets would refine for a value no one can read, while a
//! step that leaves the motion unresolved still moves the loss (the retro-reflected tweezer over 10 us
//! at the stiff axis' period, 1 us: 0.9 % of its atoms lost, T 160 % high, against 0.016 us).
//! All go about as dt², so a step that misses is shrunk by the square root of its excess and checked
//! again, and one that meets them with room is grown the same way, by whatever factor -- from the second
//! check on by the power the last two measured (`power`) --, until a step
//! comes up twice: first by the population criterion's bound, as far as the first step it alone misses, then from the largest step that met
//! the targets by it, or that one, with the population error measured -- by the excess of the criteria evaluated
//! (`next_step`): where the motion or the loss missed, the population was not measured, and its bound,
//! which over-counts several times where the measured search runs, is left out (a step that missed by
//! 1 % on the motion went back by the bound's ninefold excess to a third of its size). A bound that a measured error then takes back paid every finer check for nothing: the orbits go as the step's
//! count, and on a PGC with its resonant probe at 100 mW/cm² they took 99.6 % of its checks, two thirds of that at 0.005 us, where the bound met, for a step of 0.05 us (its bound 7 there, its measured error 0.4). Every checked
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
//! 0.25 us. The step's exponentials are summed to the numerical share of the
//! population target over a waiting time, `NUMERICAL_SHARE` ε dt/waiting, instead of to double precision. A trajectory takes at most
//! `MOST_STEPS` steps over the run: neither the first step nor the search goes finer, and a step that
//! bound sets is kept even where it misses its shares, its measured bias booked as any step's (the
//! flag, where a total exceeds its target, says why).
//!
//! The kept step is read once more against T's error over the run itself (`drift`): the orbits' criterion holds one period, and a
//! rate's error adds up over as long as the motion keeps it, which no span set in advance knows. The retro-reflected tweezer's
//! lattice axis heats by an e-fold per 30 us all through its 100 us; at 0.098 us the rate is 10 % low where the orbits' period reads T's
//! tolerance, 4.5 %, and T is −23 ± 2 % where it binds, against 0.0245 us (8192 trajectories on shared streams). So from the orbits' probes
//! a run's own flow, its jumps in on the same streams, in steps of dt and of dt/2, over 1, 2, 4... periods while an error stands beyond
//! its errors (`standing`), up to the run: there
//! about 25 % with errors of 10 % on 32 probes at 0.104 us, 24.6 % with errors of 4.3 % on 256: a steady rate's error stands as far above its errors at every span, each span's cost going with its
//! length, while on a chaotic orbit a pair's errors outgrow every other error within two periods (the 3D PGC's and the D1 molasses's ±3 % on 256 probes over one period, ±10 % over two, ±20-30 % over
//! eight: no count of probes reads T's tolerance there, and proving that nothing stands took most of the read). So the read is the probes' first layer, the others only to confirm what stands beyond
//! its limit, T's tolerance at the probes' times by then (`tolerances`) or the loss's: a stand there moves the whole run's step, and only while the next layer takes fewer steps than what its verdict could move of the run (the tweezer's
//! read at its moved step: 200 thousand against 50 thousand). A size moves the step only where it lies beyond the limit by its errors (`standing`): on a chaotic span they far exceed the limit, and at two standard errors alone a size is no error of the
//! step's -- the gray molasses's 0.204 ± 0.150 at 420 us on all 256 probes, against a limit of 0.083, at 0.556 us, is 0.201 ± 0.159 at 0.417 us; its squares' difference with the errors, taken alone, took one of 64 pilots there finer, and one of 16 of the 3D PGC's from 0.63 to 0.43 us.
//! And the read takes the largest of its sizes over the axes and the spans, so its errors stand at the confidence of those looks (`looks_confidence`, as `steady`'s): at two standard errors, of its 27 looks over the 3D PGC's 10 ms, x's 0.526 ± 0.337 at 6.7 ms on 128 probes, against a limit of 0.1,
//! took one of 48 pilots from 0.625 to 0.329 us, and sizes just past their errors at 4 of its 9 spans held the read through the whole run, then again at the finer step: 73 CPU-s against the others' 0.0-2.5, a wait three times theirs. T's step error there is about that limit all along (x and y from 1.5 ms on: +14.2 ± 1.6 % at
//! 0.70 us, +7.9 ± 1.6 % at 0.50 us, against 0.07 us, 8192 trajectories on shared streams; T's target there 0.77): no count of probes reads it, and a stand of five times it was the noise. At 2.57 standard errors none of the 48 moves or reads for more than 0.9 CPU-s, and the retro-reflected tweezer's error still moves every one of its 32 pilots at their first read (0.044-0.066 us against 0.043-0.066 us).
//! The read is needed only where the error could add up: with the kept step's
//! motion's error per period, summed over the periods up to each output, within that output's limit, no read can change the step; and only where its first layer over its first span pays for what that sum beyond the limit could move of the run, the test the next layers take (`read_pays`).
//! (The resonant probe's PGC, two periods, its jumps parting the pairs within them: ±0.32 on 64 probes against a limit of 0.045, 5 of 12 pilots read at 1.05-1.25 limits, 54 thousand exact steps on one of them for at most 7 thousand of the run's; its T at 0.5 us is 0.4 ± 2 % of 0.125 us's, and the one that stood, 6 limits at 0.5 us, took the step to 0.167 us for nothing.
//! The retro-reflected tweezer's error sums to 12-15 limits: its first read of 6 thousand steps against a million.) More than the limit
//! standing, the step where it would come to it, as dt², its bias as much, is checked again, at least as far as that span: 0.046-0.053 us over the pilot's seeds, T −3.8 ± 1.9 % at
//! 0.045 us and −6.0 ± 1.9 % at 0.052 us where it binds, against 0.013 us (8192 trajectories on shared streams); at 0.104 us, T −23 %, it stands 2-6 limits on 28 pilots of 28. On the other scenes the noise stops the spans within their first few and no step moves.
//!
//! # Shares
//! The targets are the totals, each output's estimated error E² = A² + b² + S²: A the a-priori estimates (the ledger's, each
//! booked as measured), b the step's bias and S the statistics' CONFIDENCE standard errors, within each target ε. The three
//! are independent errors -- the model's, the integrator's, the sample's -- and the field's rule for independent uncertainties
//! (the GUM's) is their root sum of squares, a bound of width B entering as B: at z = 2 that keeps the chance of an error
//! beyond ε at 5.2–6.2 % for any A with the bounds' errors uniform within them (their rectangular distribution, the GUM's
//! type B; 4.6 % for the statistics alone), and A + b + S = 1.65 ε at most at A = 0.45 ε with both at their bounds of one
//! sign and two standard errors on top (a chance of 35 % beyond ε in that corner). The linear sum held every output to that
//! corner: its chance of an error beyond ε is at most 1.6 %, 0.1 % at A = 0.45 ε, for 6.3 times the work there (18 times at
//! A = 0.72 ε). The targets are the method's references, approached as QuTiP's users approach theirs, not guarantees; A's own
//! items stay summed, the bounds of one systematic account: the largest holds at least 90 % of it where it matters, and a root
//! sum of squares would save at most 10 % of the trajectories (GMD1C's; PGC3D's 4.4 %).
//!
//! The trajectories go as S^−2, the work per trajectory as 1/dt, and the bias as c dt^p (p = `ORDER`, the integrator's): with
//! R² = ε² − A² the run's room, the cost W N ∝ 1/(dt S²), S² = R² − b², is least at b² = R²/(1 + 2p), the step's share s =
//! `BIAS_SHARE` of the room -- 1/√5 = 0.447 at second order, the statistics keeping √(1 − s²) = 0.894 of it; where fixed costs
//! per unit time (the jumps, a multirate grid's nodes) make the work go as dt^−w, w < 1, at R²/(1 + 2p/w), which the step's
//! predicted costs weigh among the steps checked (`kept`). (MLMC splits a mean-square target into the bias' and the variance's
//! halves, its cost going as here with its own exponents.) The numerical tolerances within a step -- its exponentials'
//! truncation, the beats its exact step leaves out, the impulse's quadrature (engine.rs `QUADRATURE`), a beating grid's own errors (`grid_phase`) -- take
//! `NUMERICAL_SHARE` of their target, ε/(1 + 2p)²: their work grows with the logarithm of their tolerance, or as its fourth root on a grid of fourth order, so that their
//! optimum lies at a few per cent of it, its terms' share of the work. The model's approximations take no fixed share: each is
//! booked as measured against the target, and chosen by its cost (the fold's candidates, the tables' refinement, model.rs
//! `tabulate`).
//! The gain's guard (model.rs `Model::lindblad`) is no such estimate: the generator is of Lindblad form by construction, its
//! round-off the limit of what it may read, and of what it refuses the norm the gain can show over the run.
//!
//! As the a-priori estimates A near ε that work diverges, as R^−(2+1/p), while past ε no work meets the target: the rule
//! that held the statistics to the sliver below the target and gave them the whole target above it was a cliff (the
//! shallow probe's T, 8.99 % a priori and 0.88 % of bias under the linear sum: 0.13 % left, 1.75M trajectories, where a little
//! more a priori would have taken a few hundred). One weighed cost orders the decisions instead -- the batches', the step's and
//! the fold's: the work times (E/ε)^k beyond the target, E the predicted total (`beyond`). Its least over the statistics alone,
//! work ∝ S^−2, gives them what the floor F² = A² + b² leaves of the target where that is at least 2F²/(k − 2), and that balance
//! beyond (`room`): the total exceeds its target where the floor takes more than 1 − 2/k of ε². Over the run's errors together,
//! work ∝ R^−(2+1/p), the least is R² = αA²/(k − α), α = 2 + 1/p, beyond (1 − α/k) of ε² (`run_room`), the step's share s of it
//! its tolerance (`tolerances`), and at that bias the statistics' own balance is exactly the rest, (1 − s²) R²: one optimum.
//! The exponent k = (2 + 1/p)/s² puts that turn where the run's errors -- what its work moves -- would be left less than s of
//! the total, the least share the optimum gives one of them (the step's bias beside the statistics); beyond the target the
//! run keeps that share, at most the target itself: the statistics never take more than the whole target, since the a-priori
//! estimates bound their errors and can lie far above them (F9's interference at 20, PGCP_10's tables at 1.25), while what the
//! statistics carry the result shows. The first order in the Doppler shift takes the step's share of the run's room beside the
//! other estimates (`doppler_allowance`), a truncation per step as the step's own error. The candidates' weighed costs (# The
//! fold's choice, # Time step) are the same work times weight: a model whose total misses its target is taken only for a
//! saving its weight outweighs -- 2 % beyond for 28 %, a quarter beyond for 16 times --, the weight moving continuously
//! with its estimates. The picks among discrete models and a few gates are not continuous: `past_targets`, `past_their_targets`, the fold rule's `BUDGET` (ledger.rs).
//!
//! # Batches
//! Every batch is cut into groups of `TASK` trajectories (lib.rs `Stratum`), a task on the CPU and a
//! block on the GPU: each group's initial states are a Latin hypercube, so its mean of the initial
//! energies, and of what they leave at early times, varies far less than independent draws'; the
//! groups are independent, so the jackknife over them measures the error. One size for every group
//! gives every trajectory the same share of the variance, so the need that a batch's errors predict
//! holds for the trajectories still to come; groups of 8 keep most of the hypercube's gain (the
//! trajectories T needs at early times fall by 10-40% from groups of 1 to groups of 8, little more
//! beyond) while the jackknife's groups grow with the trajectories. T, ⟨v⟩ and n̄ are each
//! atom's means over one trap period of the axis around each output (lib.rs `simulate`): its kinetic
//! energy sloshes into potential energy and back twice a period, so over a period the spread of T
//! between atoms is their energies' alone.
//!
//! The first standard errors stand on `FIRST_GROUPS` groups on any machine: the pilot's trajectories,
//! each a group of one (with the harmonic control, at the pilot's step), predict the need, and the
//! first batch on the CPU is that many groups of `TASK`, at most the need the pilot predicts -- a machine of more threads runs it in fewer rounds, not with more trajectories --, each as far as the reach it predicts wants it, half of them through the run (`WHOLE_GROUPS`): a whole first batch held the 3D PGC's and the D1 gray molasses's late outputs at 256 trajectories where they need 170 and 50-70.
//! An output the pilot under-predicts gets its trajectories in the next batch, which pass every earlier output too.
//! On the GPU a wave's time grows with its trajectories (a laptop's RTX 5070 at the 3D PGC's step: 2.1 s for 8, 2.7 for 256, 3.96 for 576 -- its clock falls with the load, its cycles per step about stay), and a wave costs at least a trajectory's time as far as it goes, so that its first
//! wave runs what the pilot predicts is needed and the need's own error (`first_ends`), each group as far as the reach wants it: a group that stops early leaves the others its clock. (Over 12 seeds: the need alone ran a second wave on 4 of DEFAULT's and 7 of THREE's, against 1 and 2 with its error; twice its error took the kernel's mean up 4-17 % on GMD1C, TET and PGC3D.)
//! T rests on the atoms still held, and its statistic stands at the share of atoms it reads, its relative error times the held share, as a population's below: read among the held atoms alone a share f would ask 1/f the trajectories, for the T of atoms the light has mostly blown out (a probe at 100 mW/cm², 97 % lost by 25 us: 16384 trajectories, the cap, against 7300-8100). Its need goes as that share, 8 f/R² with x = f N atoms' relative error √(2/x) and R its room, from nothing with no atom held: at the root of the share every count from two asked 8/R² and none or one nothing, so that on a probe at 1 W/cm², 96-98 % lost at its first output of 50 us, a first batch's 2 or 3 held atoms went on to 800-2000 trajectories for T's of a few atoms and one holding fewer stopped at 80; at the share all stop at 80. The reported errors stay T's own: where few atoms remain the output says so.
//! The populations' target is the app's, whose rows are the probabilities p (1 − L) that the atom is in a state and still held, beside the loss L, adding up to one: a population's statistic, and its reported error, are its error among the held atoms times the held share (`shown_error`), the loss's as it is. Read among the held atoms alone they asked those 7300-8100 of that probe, about 1000 now, and of a probe at 10 W/cm², 99.5 % lost by 600 us, 130 000-420 000 for numbers shown at 0.5 % of their size.
//! Trajectories alike read no error: N atoms all lost, or none of them in a state, stand at a jackknife of zero, which says nothing of what none of them met; Wilson's interval at `CONFIDENCE` leaves it up to z²/(N + z²) away (`unseen`). So an output's populations and loss take at least the trajectories that bring that within their room, z²(1/R − 1), 76 at the target (`population_ratio`), and the reported error is at least that far end. (A probe at 1 W/cm², 2.5 % held at its first output of 10 us: its pilot of 32 kept none, its run of 16 read all lost.) It is what a count of none says, and below what any output with counts to read asks but for a run of very few trajectories, whose N it sets.
//! An output's need is its trajectories so far times its worst ratio squared (the statistics over their rooms); a trajectory
//! that reaches an output passes every earlier one, so that the trajectories an output must be passed by -- its reach -- are the
//! largest need from it on, and a batch's trajectories run only as far as the outputs that still want them, half a period past
//! the last (lib.rs `simulate_to`): the outputs where T binds or the fold's memory leaves the populations little room are often
//! the first few (the D1 gray molasses' first 0.4 ms of 10 ms; the 3D PGC's T at its first output alone), and those alone
//! paid for the whole run. Each group of a batch runs as far as its place in the reach (at most doubling each
//! output's trajectories: the need is the largest of many outputs' estimated errors, biased up while few groups stand behind it). The estimates at an output
//! rest on the groups that reach it (stats.rs `summarize`), a standard error's rise at the point where they thin out what the
//! targets allow. The errors the need takes are each output's over its neighbours as far as they agree (`steady`): the largest of many outputs'
//! noisy estimates stood 20-35 % above their common value in the standard error, 1.3-1.8 times in the need. An output's reported error, which the app's marks read against the target, is the same number (`reported_error`): the stop and the marks read one. (The default scene's probe over 100 us at 448 trajectories: 2 SE at single outputs read up to 5.3 % over 14-19 us, a mark on 6 of 32 seeds, where the rows' 2 SD over the seeds is 3.8-4.4 % and the most trajectories in [0, 1] can spread, 2 √(p (1 − p)/N), 4.6 %; over its neighbours it reads 4.7-4.9 %, no mark on any seed.) No margin is added above the prediction: a batch more costs the device's tail and the statistics, a few
//! trajectories' time, where a margin of the estimate's own noise would cost tens of percent of the trajectories.
//!
//! # Control variates
//! Each output's κ² and populations are estimated with control variates of exactly known means
//! (control.rs, stats.rs `estimate`): the harmonic twin's for T, always (it costs a few products per
//! output), and the held twin's for the populations. β is the regression on the control over every
//! trajectory so far, held when the jackknife leaves a group out: its errors -- what the stopping rule
//! takes -- are the corrected values' spread over the groups; where a control follows nothing, β ≈ 0.
//! The held
//! twin costs a trajectory of its own, at its share of the work (counted as the step's, `held_work`). It runs in the first batch where the pilot's need says it can pay for itself: the
//! populations must need more trajectories than T does and than that batch holds, by more than that share (`twin_can_pay`) -- where T
//! decides the trajectories, or the first batch is all a run takes, its control of the populations saves none. It is kept when the trajectories
//! that batch says the targets need with it -- the populations' or T's, whichever needs more --,
//! times its share, are fewer than without. The first batch's groups are
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
//! T's target follows the method's own error continuously (`regime`): a semiclassical expansion's leading term goes in ħ², and
//! the references read C (ħω/k_B T)² per axis --
//! | reference | k_B T/ħω: T's error |
//! |---|---|
//! | TW2 (1D GMW, cooling) | 5: ≤ 4.5 %; 4.6: 6 %; 4: 7-10 %; about 3: −6 %; 1.2: −46 % (−23 % at 2-3 ms between) |
//! | x alone free | 5.6-4: 3.4-7.7 % |
//! | 2D quantum trajectories | 4.6: 8 %; 3.1: 15 % |
//! -- C about 1.1-1.7 from 5 to 3 ħω (0.7 at 1.2). C = 1.6 puts that error at the T target exactly at 4 ħω: `CLASSICAL` is that
//! crossing, not a number of its own. So at an output T's target is ε_T/w, w = min(1, m²), m = k_B T/(CLASSICAL ħω) on the lowest
//! axis: ε_T in the regime and the method's own error below it -- in the need, the totals' weight and the flag alike, and for
//! the step's tolerance through the run's room there (`tolerances`). A yes or no on a noisy T at one output decided the whole
//! run's step and need (the 3D PGC's dt from 0.294 to 0.164 us as its pilot's T_y at the first output crossed 4 ħω, RETRO's need
//! from 9 to over 700 as its T_y did at the last).
//!
//! The step's tolerance reads that weight at T's upper confidence bound, T + CONFIDENCE standard errors per axis (`ceiling`), as every criterion of the step reads its own errors: it is a limit, and no later batch corrects a step. On T itself a pilot's few atoms under-read it where it lies near
//! CLASSICAL: RETRO's T_y of 23.0 ± 4.8 µK at 85 µs where it is 30.5, w 0.59 for 1, T's tolerance 7.6 % for 4.5 %, its step 0.066 us and T_x −11 % on one seed of 32, where the bound keeps 0.043-0.058 us on 96 seeds, T_x at most −9 %; the 3D PGC's T_y at its first output over 10 ms, T's tolerance up to 7.3 % on 17 seeds of 48, its step up to 0.76 us,
//! T_x and T_y +16 %, where the bound keeps 0.46-0.63 us on all 48, +12 % at most. A T that lies below the limit by its errors keeps its target: GMD1C, TET, DEFAULT, THREE and the 1 ms runs keep their steps on 24 seeds each. T's weights over the states (`binding_at`) and the need (`predicted`, the batches' `output_ratios`) keep T itself: a weighing and a prediction want their estimate's
//! middle, and the batches' T, within a few per cent, sets the run's own target.
//!
//! # Device
//! The batches run on the device asked for (`Device`): the CPU's threads, or the GPU (gpu/mod.rs),
//! which runs a batch's trajectories as the CPU does -- the same groups, initial states, random
//! streams and sums. The two devices make a solve's choices alike -- the model, the ledger, the pilot
//! and the time step on the CPU in double precision either way, the statistics and their stopping
//! rule -- but for the fourth order: on the GPU it is built and taken by the kernel's cycles in whole waves of the device's width (`kernel_interval`, `Gpu::width`), so that a GPU's model can differ from the CPU's, and from another GPU's.
//! A batch is whole groups: on the CPU at least one (its threads take the trajectories one at
//! a time, so a batch costs its trajectories and one tail whatever its size), and on the GPU a wave of them,
//! which costs at least a trajectory's time as far as it goes. A GPU that cannot run (none, an older driver, a
//! model beyond its kernel) leaves the solve on the CPU with the reason
//! (`Solution::device_note`); one that fails during the batches has the solve redone on the CPU.
use crate::engine::{column_norm,node_phase,series,Engine,Landings,Settings,Trajectory,QUADRATURE};
use crate::gpu;
use crate::ground::{ExcitedKind,Ground,Kind};
use crate::ledger::Ledger;
use crate::model::{Exact,Model,Trap,MOST_MIXED,MOST_STATES,NOT_LINDBLAD};
use crate::control::{Harmonic,Twin};
use crate::stats::{summarize,Estimate,Group,Means,Scales,Sums,KT_PER_UK};
use crate::{pilot_start,sample_states,simulate_to,Initial,Outcome,Variates,TASK};
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
/// The share s² of the run's room's square the step's bias takes beside the a-priori estimates (module documentation, # Shares):
/// 1/(1 + 2p), its cost-optimal share beside the statistics.
pub const BIAS_VARIANCE:f64=1./(1.+2.*ORDER);
/// The step's share s of the run's room: √`BIAS_VARIANCE`.
pub const BIAS_SHARE:f64=0.447213595499958;
/// The numerical tolerances' share of their target (module documentation, # Shares): 1/(1 + 2p)².
pub const NUMERICAL_SHARE:f64=BIAS_VARIANCE*BIAS_VARIANCE;
/// The power k of a total's weight beyond its target, (E/ε)^k (module documentation, # Shares): the run's work exponent
/// 2 + 1/p over the step's share of the square, (1 + 2p)²/p.
pub const BEYOND:f64=(2.+1./ORDER)/BIAS_VARIANCE;
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
/// The groups of a CPU's first batch that run through the whole run whatever the pilot's need says (`first_ends`), half `FIRST_GROUPS`: a jackknife's variance over g groups errs by √(2/(g − 1)), 36 % here against 25 % on those, and
/// its coverage at CONFIDENCE standard errors is that of t with g − 1 degrees of freedom, 93.6 % here against 94.6 %.
const WHOLE_GROUPS:u64=FIRST_GROUPS/2;

pub const NOT_CONFINED:&str="光镊不约束原子：3D 运动计算光镊中的单个原子，需要在焦点附近形成势阱的光镊";

/// Distributions along one axis on one grid, at some of the times.
#[derive(Clone,Debug,PartialEq)]
pub struct Distributions{pub grid:Vec<f64>,pub samples:Vec<usize>,pub probabilities:Vec<Vec<f64>>}

/// A 3D motion solve's result, at every time of the request (per lab axis).
/// Everything but `loss_probability` describes the atom still held.
#[derive(Clone,Debug)]
pub struct Solution{
    pub populations:Vec<Vec<f64>>,
    /// Per time, the largest standard error of the populations as shown, p (1 − L), and the loss, as the stopping rule reads it, at least what its trajectories leave open (`reported_error`).
    pub sample_errors:Vec<f64>,
    pub temperature_uk:Vec<[f64;3]>,
    pub temperature_errors_uk:Vec<[f64;3]>,
    /// T_total = (T_x + T_y + T_z)/3 and its standard error (stats.rs `Estimate`).
    pub temperature_total_uk:Vec<f64>,
    pub temperature_total_errors_uk:Vec<f64>,
    pub mean_velocity_mps:Vec<[f64;3]>,
    /// ω_a/2π (kHz) at the bottom of the reference state's trap per lab axis, and n̄ per axis relative to it (stats.rs `Estimate::occupation`).
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
    /// first whole batch at or beyond it (a batch at most doubles the trajectories run).
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

/// The weight w of T's target at an output (module documentation, # Regime): its target is ε_T/w, w = min(1, m²), m the lowest
/// axis's margin k_B T/(CLASSICAL ħω).
fn regime(e:&Estimate,frequencies:[f64;3])->f64{weight_at(e.temperature,frequencies)}
/// T's weight at its upper confidence bound, CONFIDENCE standard errors per axis above T, for the step's tolerance (module documentation, # Regime).
fn ceiling(e:&Estimate,frequencies:[f64;3])->f64{weight_at(std::array::from_fn(|a|e.temperature[a]+CONFIDENCE*e.temperature_error[a]),frequencies)}
fn weight_at(temperature:[f64;3],frequencies:[f64;3])->f64{
    (0..3).filter(|&a|frequencies[a]>0.).map(|a|(KT_PER_UK*temperature[a]/(CLASSICAL*frequencies[a])).max(0.)).fold(1_f64,f64::min).powi(2)
}
/// An output's largest standard error of what is shown of the atom (# Batches): a population as the probability p (1 − L) that it is in that state and held, its error among the held atoms times the held share, and the loss's.
fn shown_error(e:&Estimate)->f64{let held=(1.-e.loss).max(0.);e.population_errors.iter().map(|&x|held*x).fold(e.loss_error,f64::max)}
/// An output's reported standard error (# Batches): the statistic the stopping rule and the flags read (`statistics_of`), at least what its trajectories leave open (`unseen`: a reading of all lost says no more than that), in standard errors. The app's marks read it against the target: one number for the stop and the marks.
fn reported_error(statistic:f64,trajectories:f64)->f64{statistic.max(unseen(trajectories))/CONFIDENCE}
/// An output's statistical errors at CONFIDENCE: the populations' as shown and the loss's largest (`shown_error`, absolute) and T's largest
/// relative one over the axes, at the share of atoms it reads (# Batches), with its target's weight there (`regime`).
#[derive(Clone,Copy,Debug)]
struct Statistic{population:f64,temperature:f64,weight:f64}
fn statistics(e:&Estimate,frequencies:[f64;3])->Statistic{
    Statistic{population:CONFIDENCE*shown_error(e),
        temperature:CONFIDENCE*(1.-e.loss).max(0.)*(0..3).map(|a|if e.temperature[a]>0.{e.temperature_error[a]/e.temperature[a]}else{0.}).fold(0.,f64::max),weight:regime(e,frequencies)}
}
/// Whether an output holds the two atoms a temperature takes (`assemble`).
fn reads_temperature(e:&Estimate)->bool{(e.trajectories*(1.-e.loss)).round()>=2.}
/// Per output the last at or before it that `holds` (the first must).
fn last_holding(outputs:usize,holds:&dyn Fn(usize)->bool)->Vec<usize>{(0..outputs).scan(0,|last,i|{if holds(i){*last=i;}Some(*last)}).collect()}
/// An output's statistics as the stopping rule and the flags take them (module documentation, # Batches): `statistics`, each
/// error over its neighbours as far as they agree (`steady`).
fn statistics_of(estimates:&[Estimate],frequencies:[f64;3])->Vec<Statistic>{
    let mut all:Vec<Statistic>=estimates.iter().map(|e|statistics(e,frequencies)).collect();
    let(counts,groups):(Vec<f64>,Vec<f64>)=estimates.iter().map(|e|(e.trajectories,e.groups)).unzip();
    let population=steady(&all.iter().map(|s|s.population).collect::<Vec<f64>>(),&counts,&groups);
    let temperature=steady(&all.iter().map(|s|s.temperature).collect::<Vec<f64>>(),&counts,&groups);
    for(i,s)in all.iter_mut().enumerate(){s.population=population[i];s.temperature=temperature[i];}
    all
}
/// Standard errors per output (`values`, at the trajectories and the groups behind each, the first output's left alone) with their
/// own noise smoothed: a jackknife's variance over g groups has the relative error √(2/(g − 1)), and the need takes their
/// largest, 20-35 % above their common value over hundreds of outputs (with T's windows overlapping, a little less). Each
/// output's variance per trajectory takes the mean over its neighbours within the widest window, doubling from one, that agrees
/// with every narrower one within z √(2/(g − 1)) of the narrower's error, z = √(2 ln n), the largest of n such errors
/// (Lepski's rule): a plateau is smoothed whole, a transient's real peak beyond that noise stays.
fn steady(values:&[f64],counts:&[f64],groups:&[f64])->Vec<f64>{
    let n=values.len();let mut out=values.to_vec();if n<3||values[1..].iter().any(|x|!x.is_finite()){return out;}
    let v:Vec<f64>=(0..n).map(|i|values[i]*values[i]*counts[i]).collect();
    let mut sum=vec![0.;n+1];for i in 1..n{sum[i+1]=sum[i]+v[i];}
    let z=(2.*((n-1) as f64).ln()).sqrt();
    for i in 1..n{
        let noise=(2./(groups[i]-1.).max(1.)).sqrt();let mut narrower=vec![(v[i],1.)];let mut half=1;
        while half<n{
            let(a,b)=(i.saturating_sub(half).max(1),(i+half).min(n-1));let m=(b+1-a) as f64;let x=(sum[b+1]-sum[a])/m;
            if narrower.iter().any(|&(y,k):&(f64,f64)|(x-y).abs()>z*noise*y.min(x)/k.sqrt()){break;}
            narrower.push((x,m));half*=2;
        }
        out[i]=(narrower[narrower.len()-1].0/counts[i]).sqrt();
    }
    out
}
/// The room an error whose work goes as its −α-th power takes beside a floor its work does not move, in quadrature (module
/// documentation, # Shares): what the floor leaves of the target's square, or where that is less, the weighed cost's least
/// αF²/(k − α); at most the target itself.
fn room_of(target:f64,floor:f64,exponent:f64)->f64{let(t,f)=(target*target,floor*floor);(t-f).max(exponent*f/(BEYOND-exponent)).min(t).sqrt()}
/// The statistics' room (their trajectories go as S^−2) beside an output's floor (`floor`): what it leaves of the target, or
/// its balance beyond 1 − 2/k of the target's square (the output is then flagged).
fn room(target:f64,floor:f64)->f64{room_of(target,floor,2.)}
/// An output's floor beside the statistics: its a-priori estimates (`Accounts::apriori`) and the step's bias, in quadrature.
fn floor(apriori:f64,bias:f64)->f64{apriori.hypot(bias)}
/// What N trajectories alike leave open of what none of them met (module documentation, # Batches): the far end of Wilson's interval at `CONFIDENCE` at a count of none or all, z²/(N + z²), where a jackknife reads zero.
fn unseen(trajectories:f64)->f64{let z=CONFIDENCE*CONFIDENCE;z/(trajectories+z)}
/// The trajectories that bring it within `room` (`unseen`): z²(1/room − 1).
fn unseen_need(room:f64)->f64{CONFIDENCE*CONFIDENCE*(1./room-1.).max(0.)}
/// The populations' and the loss's ratio to their room at N trajectories, its square times N the need: their statistic's, at least what brings `unseen` within the room.
fn population_ratio(statistic:f64,room:f64,trajectories:f64)->f64{(statistic/room).max((unseen_need(room)/trajectories).sqrt())}
/// The weight of a total at its target (module documentation, # Shares): 1 within it, (E/ε)^k beyond.
fn beyond(target:f64,total:f64)->f64{(total/target).max(1.).powf(BEYOND)}
/// What a run predicts at a candidate step (module documentation, # Shares, # Batches): the trajectories its rooms need --
/// per output after t = 0 and target (T's its regime's, `regime`) the pilot's statistic over its room beside the a-priori
/// total there and the step's bias, squared, times the pilot's trajectories; per output the largest of its own and every later
/// output's, its reach (`reaches`), at least a first batch (`FIRST_GROUPS` groups, at most the run's need) and at most
/// `MOST_TRAJECTORIES` -- their mean over the outputs, and the weight of the largest total at those trajectories, its
/// statistics the pilot's scaled to them.
fn predicted(statistics:&[Statistic],apriori:&[[f64;2]],bias:[f64;2],pilot:f64)->(f64,f64){
    let terms=|i:usize|{let s=statistics[i];let a=apriori[i];[(POPULATION_TARGET,s.population,floor(a[0],bias[0])),(TEMPERATURE_TARGET/s.weight,s.temperature,floor(a[1],bias[1]))]};
    let needs:Vec<f64>=(0..statistics.len()).map(|i|if i==0{0.}else{let[p,t]=terms(i);pilot*population_ratio(p.1,room(p.0,p.2),pilot).max(t.1/room(t.0,t.2)).powi(2)}).collect();
    let reach=reaches(&needs);let largest=reach.get(1).copied().unwrap_or(0.).clamp(TASK as f64,MOST_TRAJECTORIES as f64);let first=((FIRST_GROUPS*TASK) as f64).min(largest);
    let run=|i:usize|reach[i].clamp(first,MOST_TRAJECTORIES as f64);let n=(statistics.len()-1).max(1) as f64;
    let weight=(1..statistics.len()).flat_map(|i|terms(i).map(|(target,s,floor)|beyond(target,floor.hypot(s*(pilot/run(i)).sqrt())))).fold(1.,f64::max);
    ((1..statistics.len()).map(run).sum::<f64>()/n,weight)
}
/// Whether the a-priori totals alone lie past a target at some output after t = 0: no room for the
/// statistics or a step. The weight (`predicted`) cannot tell: the trajectories a run needs bring its binding total to the target itself, its
/// weight 1 within round-off.
fn past_targets(statistics:&[Statistic],apriori:&[[f64;2]])->bool{
    (1..statistics.len()).any(|i|apriori[i][0]>POPULATION_TARGET||apriori[i][1]>TEMPERATURE_TARGET/statistics[i].weight)
}
/// The trajectories each output must be passed by (module documentation, # Batches): the largest of its need and every later
/// output's.
fn reaches(needs:&[f64])->Vec<f64>{let mut most=0_f64;let mut out:Vec<f64>=needs.iter().rev().map(|&x|{most=most.max(x);most}).collect();out.reverse();out}
/// The run's room beside the a-priori estimates -- the step's bias and the statistics together, their work going as
/// R^−(2+1/p) at their cost-optimal split, in quadrature (module documentation, # Shares): what the estimates leave of the
/// target's square, or beyond (1 − s²) of it their balance αA²/(k − α), α = 2 + 1/p; at most the target. The step takes its share s (`tolerances`).
fn run_room(target:f64,apriori:f64)->f64{room_of(target,apriori,2.+1./ORDER)}
/// The largest ratio of the statistical errors to their room over every output after t = 0 (the
/// request's initial state, `start`): 1 meets the targets.
fn worst_ratio(estimates:&[Estimate],accounts:&Accounts,frequencies:[f64;3])->f64{output_ratios(estimates,accounts,frequencies).into_iter().fold(0.,|m,r|m.max(r[0]).max(r[1]))}
/// The same per output and kind: the populations' and the loss's (`population_ratio`), and T's (zero at t = 0).
fn output_ratios(estimates:&[Estimate],accounts:&Accounts,frequencies:[f64;3])->Vec<[f64;2]>{
    let all=statistics_of(estimates,frequencies);
    (0..estimates.len()).map(|i|{if i==0{return [0.;2];}
        let s=all[i];let a=accounts.apriori[i];
        [population_ratio(s.population,room(POPULATION_TARGET,floor(a[0],accounts.bias[0])),estimates[i].trajectories),s.temperature/room(TEMPERATURE_TARGET/s.weight,floor(a[1],accounts.bias[1]))]
    }).collect()
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
    let order=if model.fourth{"四阶折叠（误差界为饱和参数之和的平方，加跃迁后瞬态光子未尽的一阶余项）"}else{"二阶折叠（误差界为饱和参数之和）"};
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
    if model.effective_dropped>0.{ledger.add("激发态缀饰驱动的快拍频项",model.effective_dropped,"略去（判据与其他耦合的二阶折叠相同）".into());}
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
    let measured=memory.charges(&estimates.iter().map(|e|with_open(fold_charges(model,2.*e.folded),model,e.trajectories)).collect::<Vec<_>>());
    let profiles:Vec<f64>=estimates.iter().map(|e|e.profile).collect();let dopplers:Vec<f64>=estimates.iter().map(|e|e.doppler).collect();
    let largest=|v:&[f64]|v.iter().skip(1).copied().fold(0.,f64::max);
    ledger.add("多普勒与反冲频移",largest(&dopplers),format!("一阶（kv/σ）²；单步超过 {:.1e} 时用精确响应",allowance));
    // What the first order's transfer leaves out (model.rs `Model::first_error`), on T alone: its friction against the recoil heating at the T the atoms have.
    let transfer:Vec<f64>=estimates.iter().map(|e|model.first_error*e.temperature_total).collect();
    ledger.add("一阶多普勒项的转移（温度）",largest(&transfer),"一阶项按各组衰减表的比例给出，超出份额的表项单独保留；其余按其摩擦对反冲加热的比值计入温度".into());
    if !model.profiles.is_empty(){ledger.add("聚焦光在原子处的强度（沿轨迹）",largest(&profiles),"按峰值强度的平面波".into());}
    let met=largest(&measured.iter().map(|m|m[0]).collect::<Vec<f64>>());
    if let Some(item)=ledger.items.iter_mut().find(|i|i.label==FOLD){
        item.action=format!("{}；计入沿轨迹实测的饱和参数 2η²{}（先验估计 {:.2}%{}）与轨迹未及之处的近共振上界{}",item.action,if model.fourth{" 的平方"}else{""},100.*charge,if met>charge{"，实测高于先验估计"}else{""},
            if model.fourth{"；温度仍计 2η²（力按二阶折叠）"}else{""});item.estimate=met;
    }
    let peak:f64=model.profiles.iter().map(|p|p.1).sum();let fixed=base.used()-peak-charge;
    let apriori:Vec<[f64;2]>=(0..estimates.len()).map(|i|if i==0{[0.;2]}else{let[p,t]=measured[i].map(|m|fixed+peak.max(profiles[i])+dopplers[i]+m);[p,t+transfer[i]]}).collect();
    let accounts=Accounts{apriori,excited:measured,bias};
    // The outputs beyond a target: (output, the quantity, its total, the a-priori estimates' and the step's, the statistics, the target).
    let all=statistics_of(estimates,frequencies);
    let over:Vec<(usize,&str,f64,f64,f64,f64)>=(1..estimates.len()).filter_map(|i|{
        let s=all[i];let a=accounts.apriori[i];let(p,t)=if last{(s.population,s.temperature)}else{(0.,0.)};let(fp,ft)=(floor(a[0],bias[0]),floor(a[1],bias[1]));
        let population=(fp.hypot(p)>POPULATION_TARGET).then_some((i,"布居",fp.hypot(p),fp,p,POPULATION_TARGET));
        let temperature=(ft.hypot(t)>TEMPERATURE_TARGET/s.weight).then_some((i,"温度",ft.hypot(t),ft,t,TEMPERATURE_TARGET/s.weight));
        population.or(temperature)}).collect();
    if !over.is_empty(){
        let(first,last_over)=(over[0].0,over[over.len()-1].0);
        let span=if over.len()==last_over-first+1&&last_over==estimates.len()-1{format!("{} 起",when(times[first]))}
            else{format!("{} 至 {} 之间 {} 个时刻",when(times[first]),when(times[last_over]),over.len())};
        let worst=over.iter().max_by(|a,b|(a.2/a.5).total_cmp(&(b.2/b.5))).unwrap();let target=worst.5;
        let over=format!("{span}估计误差超出目标：{}误差最大 {:.1}%（目标 {:.0}%；三项平方和开方：近似与步长 {:.1}%，其中消去激发态 {:.1}%；统计 {:.1}%）",
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
    // The generator before the rule, every candidate's start (`next_candidate`); the rule's allowances from it, past the first -- every drive folded -- only where a rung is wanted.
    let bare=Ground::bare(request)?;let mut rungs=None;
    let mut ladder=std::iter::once(f64::INFINITY).chain(std::iter::from_fn(||rungs.get_or_insert_with(||bare.ladder().into_iter().skip(1)).next()));let mut failed=None;
    let Some(mut pilot_candidate)=next_candidate(request,&bare,&mut ladder,&mut failed,&mut vec![])else{return Err(Stop::Error(failed.unwrap_or_else(||"No model of the request".into())))};
    pilot_candidate.settle();
    // A generator that gains on the ladder's first rung, every drive folded, is the fold's construction failing
    // (model.rs `Model::lindblad`), not a regime to escalate into explicit states: the solve stops with why.
    if let Some(e)=failed.as_ref().filter(|e|e.starts_with(NOT_LINDBLAD)){return Err(Stop::Error(e.clone()));}
    // The reference state: the largest initial population; its trap sets ω per axis, in each model its own (`Candidate::reference`).
    // A trap the model gives only where the model holds: a strong near beam whose fold anti-traps the reference state is the elimination's resonance first (the validation's F8). A model whose own
    // estimates lie past the targets hands over to the next rung, which keeps that drive (a resonant probe of 300 mW/cm², cooling of 30: 146 % of resonance with every drive folded); the last refuses.
    let mut shapes=vec![pilot_candidate.folds()];
    while pilot_candidate.reference().0.frequencies.iter().any(|&w|!(w>0.)){
        let next=if pilot_candidate.past_their_targets(){next_candidate(request,&bare,&mut ladder,&mut failed,&mut shapes)}else{None};
        let Some(next)=next else{return Err(Stop::Error(match resonance(&pilot_candidate.model){Some(r)=>format!("{r}：光镊的势阱无从判断"),None=>NOT_CONFINED.into()}))};
        pilot_candidate=next;pilot_candidate.settle();
    }
    let trap=pilot_candidate.reference().0;
    // The search's first step: the light pattern's phase k_max v_T per step at sqrt(s × the temperature
    // target), or without light the trap's own ω.
    let thermal=(KT_PER_UK*request.temperature_uk*pilot_candidate.model.hbar_over_mass).sqrt();
    let pattern=pilot_candidate.model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max)*thermal;
    let rate=if pattern>0.{pattern}else{trap.frequencies.iter().copied().fold(0.,f64::max)};
    // The finest step the work bound allows (`MOST_STEPS`), in steps per interval.
    let most=(MOST_STEPS/(outputs-1)).max(1);
    let first=((interval*rate/first_phase()).ceil() as usize).clamp(1,most);
    // The loss's region is the reference trap's (engine.rs `Settings::region`).
    let plain=|(trap,depth):(Trap,f64),steps:usize,truncation:f64|{let mut s=Settings::new(interval/steps as f64,&trap,truncation,true);s.region=depth;s};
    // The pilot's exponentials and the beats its exact steps leave out within the run's share,
    // the numerical share of the population target over a waiting time between jumps (engine.rs `Segment::resolve`, `running`):
    // before it measures one, the one its atoms start with -- the inverse of its initial states' mean jump rate,
    // the no-jump decay ⟨ψ|i(H − H†)|ψ⟩/⟨ψ|ψ⟩ at their points, at most the run. Where the atoms then go dark the
    // waiting grows and the pilot's share is looser by as much: a beat it leaves out errs by at most that growth
    // times the run's share in an atom's populations, which the pilot's uses take far coarser -- its states
    // sample the step's check (at the run's settings), its fold's charge is the percent of its 2η², its jumps and
    // statistics their counts' --, and the run takes the waiting time the pilot measures.
    let piloting=|at:(Trap,f64),steps:usize,waiting:f64,phase:f64|{let mut s=plain(at,steps,NUMERICAL_SHARE*POPULATION_TARGET*(interval/steps as f64)/waiting);s.beats=s.truncation;s.horizon=waiting;s.node_phase=phase;s};
    let period=slowest_period(&trap);
    // A run's settings at `steps` per interval: its exponentials summed to the numerical share of the population
    // target over a waiting time, at most the run -- what a step leaves out adds up until a jump or the run's end --, `NUMERICAL_SHARE` ε dt/waiting, and the beats left out within it; the first order in the
    // Doppler shift within `allowance` (`doppler_allowance`); a beating grid at `phase` (`grid_phase`).
    let running=|at:(Trap,f64),steps:usize,waiting:f64,allowance:f64,phase:f64|{let dt=interval/steps as f64;let horizon=waiting.min(duration);let mut s=plain(at,steps,NUMERICAL_SHARE*POPULATION_TARGET*dt/horizon);s.doppler=allowance;s.horizon=horizon;s.node_phase=phase;s};
    // A pilot's model's initial states, the waiting time they start with and its beating grid's phase on them.
    let begin=|c:&mut Candidate|->(Vec<Trajectory>,f64){c.settle();
        let starts:Vec<Trajectory>=(0..pilot).map(|index|pilot_start(&c.model,&plain(c.reference(),first,f64::EPSILON),&c.initial,seed,index)).collect();
        let start_rate=starts.iter().map(|t|{let m=&c.model;let g=m.generator(t.cluster,&m.point(t.r,t.kappa),0.);let n=g.n;
            let(mut form,mut norm)=(Complex::ZERO,0.);for a in 0..n{norm+=t.psi[a].norm_sqr();for b in 0..n{form+=t.psi[a].conj()*g.hamiltonian[a*n+b]*t.psi[b];}}-2.*form.im/norm}).sum::<f64>()/pilot as f64;
        let waiting=if start_rate>0.{(1./start_rate).min(duration)}else{duration};
        c.phase=grid_phase(&c.model,&starts,&running(c.reference(),first,waiting,doppler_allowance(&c.base,&c.model),c.phase),None);(starts,waiting)};
    let(mut starts,mut start_waiting)=begin(&mut pilot_candidate);
    // The pilot's step (module documentation, # Time step): the bound rule's search on its own initial states from the
    // first step, at the a-priori estimates of its model -- the largest step met there, else the search's last. With explicit states their exact steps take the populations, whose bound over-counts many
    // times there, and bare states ring (`relaxed`; a resonant probe of 100 mW/cm²: 502 steps per us for the choice's 5): that first step, the light's pattern's.
    let steps=if pilot_candidate.model.ground.clusters.iter().any(|x|!x.excited.is_empty()){first}else{let c=&pilot_candidate;let allowance=doppler_allowance(&c.base,&c.model);
        let weights={let mut e=Estimate::default();start(&mut e,&c.initial);vec![regime(&e,trap.lab);outputs]};let tolerances=tolerances(&weights,&c.charges.iter().map(|x|x.map(|x|c.others()+x)).collect::<Vec<_>>());
        let binding=vec![1.;starts.len()];let probes=bound_probes(&c.model,&starts,&binding);
        let deposits=Deposits::default();
        let doubled=|k:usize|{let r=step_excess(&Engine::new(&c.model,running(c.reference(),k,start_waiting,allowance,c.phase)),&starts,&binding,start_waiting,&probes,period,0.,tolerances,&deposits,(FIRST_GROUPS*TASK) as f64*duration);Check{steps:k,population:r[0],loss:r[1],motion:r[2]}};
        let mut bound=vec![];search(first,most,&[],&doubled,None,false,&mut bound);
        bound.iter().filter(|c|c.met()).map(|c|c.steps).min().unwrap_or(bound[bound.len()-1].steps)};
    // The pilot's atoms. A pilot that keeps none to the first output measures nothing the choice can read: a model whose own estimates lie past the targets (`Candidate::past_their_targets`)
    // hands it to the next rung, at the same step, what the choice's check starts from; a model within them has lost its atoms by its physics.
    let(pilot_harmonic,pilot_states,pilot_jumps,pilot_groups)=loop{
        let harmonic=Harmonic::new(&pilot_candidate.model,&piloting(pilot_candidate.reference(),steps,start_waiting,pilot_candidate.phase),&pilot_candidate.initial,steps,outputs);
        let(states,jumps,groups)=sample_states(&pilot_candidate.model,&piloting(pilot_candidate.reference(),steps,start_waiting,pilot_candidate.phase),&pilot_candidate.initial,seed,pilot,steps,outputs,&harmonic);
        if states.is_empty()&&pilot_candidate.past_their_targets(){if let Some(next)=next_candidate(request,&bare,&mut ladder,&mut failed,&mut shapes){pilot_candidate=next;(starts,start_waiting)=begin(&mut pilot_candidate);continue;}}
        break(harmonic,states,jumps,groups);
    };
    let(trap,period)={let t=pilot_candidate.reference().0;(t,slowest_period(&t))};
    // The fold's memory (module documentation, # Ledger): per pilot atom its jumps at the outputs; the pilot's
    // initial states, where the charge starts.
    let memory=Memory::of(&pilot_states,interval,outputs);
    let charge_of=|from:&Model,m:&Model|measured_charge(from,m,&pilot_states,&starts,&memory,interval,outputs);
    let charge=charge_of(&pilot_candidate.model,&pilot_candidate.model);pilot_candidate.charged(charge);
        // The pilot's statistics per output (its trajectories each a group of one, with the harmonic control): CONFIDENCE
    // standard errors, the populations' and the loss's largest and T's with its regime's weight (`statistics`). The
    // trajectories a model needs are (statistic/room)² times the pilot's at the output and target that bind (`predicted`).
    let(pilot_statistics,ceilings):(Vec<Statistic>,Vec<f64>)={
        let scales=Scales{hbar_over_mass:pilot_candidate.model.hbar_over_mass,frequencies:trap.lab};
        let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        let mut e=summarize(&pilot_groups,&scales,&means);start(&mut e[0],&pilot_candidate.initial);
        (statistics_of(&e,trap.lab),e.iter().map(|x|ceiling(x,trap.lab)).collect())
    };
    // T's target's weight per output where the pilot's atoms are: over the states, where T's criteria read (`regime`, `binding_at`), and at its upper bound against the step's tolerance (`ceiling`, `tolerances`).
    let weights:Vec<f64>=pilot_statistics.iter().map(|s|s.weight).collect();
    // The relative standard error of the need the pilot predicts (`need_error`): the margin a finer step must save
    // by (`kept`).
    let need={let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        need_error(&pilot_groups,&Scales{hbar_over_mass:pilot_candidate.model.hbar_over_mass,frequencies:trap.lab},&means,&pilot_candidate.initial,trap.lab)};
    let pilot_waiting=pilot as f64*duration/(pilot_jumps.max(1) as f64);let pilot_horizon=pilot_waiting.min(duration);
    // The beats the pilot's exact steps left out (engine.rs `Segment::resolve`): their bound per unit time, at
    // most the run's share over its waiting time (`booked_beats`).
    let pilot_beats=booked_beats(&pilot_states,pilot_horizon);
    // Its grid on its states, as a rung's on its own (`grid_phase`).
    let groups=pilot_candidate.model.ground.cluster.clone();
    pilot_candidate.phase=grid_phase(&pilot_candidate.model,&pilot_states,&running(pilot_candidate.reference(),steps,pilot_waiting,doppler_allowance(&pilot_candidate.base,&pilot_candidate.model),pilot_candidate.phase),Some((&groups,duration)));
    // A candidate's weighed cost on the pilot's states carried into its clusters (module documentation, # The fold's
    // choice): its step check there (`step_check`), every step weighed by its counted work per interval times the
    // trajectories its rooms at its bias need and the weight of the totals they leave (`predicted`) over the run; at the
    // step it keeps, that cost, the check's own work, its work per interval, its bias and its a-priori totals per output.
    let evaluate=|c:&Candidate,states:&[Trajectory],begun:&[Trajectory],first:Option<(Check,[f64;2])>,deposits:&Deposits|->Costed{
        let base=gained(&c.base,&c.model,states);let(others,allowance)=(unfolded(&base,&c.model),doppler_allowance(&base,&c.model));
        let period=slowest_period(&c.reference().0);let settings=|s:usize|running(c.reference(),s,pilot_waiting,allowance,c.phase);let parts=state_work(&c.model,states,begun);let binding=binding_at(states,&weights,interval);
        let counted:std::cell::RefCell<Vec<(usize,f64)>>=Default::default();
        let work=|s:usize|->f64{if let Some(&(_,w))=counted.borrow().iter().find(|x|x.0==s){return w;}
            let w=interval_work(&c.model,&parts,&settings(s),interval,pilot_waiting);counted.borrow_mut().push((s,w));w};
        let apriori:Vec<[f64;2]>=c.charges.iter().map(|x|x.map(|x|others+x+pilot_beats*pilot_horizon)).collect();
        let tolerances=tolerances(&ceilings,&apriori);let bias=|k:&Check|{let r=k.ratios();[r[0]*tolerances[0],r[1]*tolerances[1]]};
        let cost=|k:&Check|{let(n,weight)=predicted(&pilot_statistics,&apriori,bias(k),pilot as f64);work(k.steps)*n*weight*(outputs-1) as f64};
        // Beyond the targets with no step bias at all (a pole its tables do not resolve, a charge past the rooms): what the step can move of its
        // cost is little beside that weight, and its check is put off until it is the choice -- its cost at the first step with no bias.
        if past_targets(&pilot_statistics,&apriori)&&first.is_none(){let(n,weight)=predicted(&pilot_statistics,&apriori,[0.;2],pilot as f64);let at=work(steps);return Costed{cost:at*n*weight*(outputs-1) as f64,check:checked_step(states.len(),bound_probes(&c.model,states,&binding).first,steps,interval,period)*at/steps as f64,work:at,steps,bias:[0.;2],apriori,checked:None};}
        let known:Vec<Check>=first.iter().map(|&(k,t)|Check{steps:k.steps,population:k.population*t[0]/tolerances[0],loss:k.loss*t[0]/tolerances[0],motion:k.motion*t[1]/tolerances[1]}).collect();
        let reach=predicted(&pilot_statistics,&apriori,[0.;2],pilot as f64).0*duration;
        let checked=step_check(&c.model,states,&binding,&settings,first.map_or(steps,|x|x.0.steps),most,pilot_waiting,pilot_horizon,pilot as usize,&pilot_groups,period,tolerances,&cost,need,&known,deposits,reach);
        let chosen=checked.chosen;
        Costed{cost:cost(&chosen),check:checked_work(&checked,states.len(),pilot as usize,pilot_horizon,interval,period,&work),work:work(chosen.steps),steps:chosen.steps,bias:bias(&chosen),apriori,checked:Some(checked)}
    };
    // The candidates (module documentation, # The fold's choice), each costed where the pilot kept states
    // (atoms all lost before the first output measure none): the pilot's model and its fourth order on the
    // pilot's own states, then the ladder's on them carried into its clusters (`relaxed`). A rung is costed while
    // its step check's least work and the least work any step leaves it -- the multirate grid's nodes and the
    // jumps (`least_work`), or a costed earlier rung's work at its step where larger: a later rung keeps more
    // states explicit, and costs more per step and per node -- times the fewest trajectories it can need and their
    // weight (its totals with no step bias) could still cost less: its check is spent whether it is taken or not.
    let measured=!pilot_states.is_empty();
    let pilot_cost=measured.then(||evaluate(&pilot_candidate,&pilot_states,&starts,None,&Deposits::default()));
    // The fourth order is built where it can pay its check and run: its work per interval at least the pilot's
    // model's at its step with its own layout's generators and jumps (`fourth_work`; GMD1C's 2.7 times, its tables eight times the
    // model's to build), its check's at least that model's at that work, its totals
    // at least those of its least charges -- per output what the atoms leave open (`with_open`, the same at either order) and the pilot's T charge squared for the populations
    // (`fold_charges`; a mean of squares is at least the square of the mean) with no step bias, and T's the second
    // order's with the pilot's model's T bias: the fourth order folds the force at the second, so that T's charge
    // and the motion's step error are the second order's. Beside a pilot's model whose check was put off (beyond the targets at
    // any step), only where those totals come within the targets: what lies beyond them is no fold's.
    let promise=pilot_cost.as_ref().map_or(true,|p|{
        let open=open_saturation(pilot_candidate.model.resonance,memory.atoms as f64);
        let apriori:Vec<[f64;2]>=p.apriori.iter().zip(&pilot_candidate.charges).map(|(a,c)|[a[0]-c[0]+open+c[1]*c[1],a[1]]).collect();
        let(n,weight)=predicted(&pilot_statistics,&apriori,[0.,p.bias[1]],pilot as f64);
        if p.checked.is_none()&&past_targets(&pilot_statistics,&apriori){return false;}
        let work=fourth_work(request,&pilot_candidate,&pilot_states,&starts,&running(pilot_candidate.reference(),p.steps,pilot_waiting,doppler_allowance(&pilot_candidate.base,&pilot_candidate.model),pilot_candidate.phase),interval,pilot_waiting);
        work*n*weight*((outputs-1) as f64)+p.check*work/p.work<p.cost});
    // On the GPU a wave takes its width at no further cost and the fourth order pays by the kernel's work, about twice the second's: it is
    // built only where the pilot's model takes more than a wave or lies beyond the targets at any step, and taken only for fewer
    // of the kernel's cycles in whole waves (`kernel_interval`).
    let waves=|x:&Costed,width:f64|(x.cost/(x.work*(outputs-1) as f64)/width).ceil().max(1.);
    let width=||gpu::gpu().map_or(TASK,|g|g.lock().unwrap_or_else(|e|e.into_inner()).width(1)) as f64;
    let promise=promise&&pilot_candidate.allowance==f64::INFINITY&&(device==Device::Cpu||pilot_cost.as_ref().map_or(true,|p|p.checked.is_none()||waves(p,width())>1.));
    let fourth_cost=if measured&&promise{Candidate::new(request,pilot_candidate.model.ground.clone(),pilot_candidate.allowance,true).ok().map(|mut c|{c.charged(charge_of(&pilot_candidate.model,&c.model));
        let into=|v:&[Trajectory]|->Vec<Trajectory>{v.iter().map(|t|carried(&pilot_candidate.model,&c.model,t)).collect()};
        let(states,begun)=(into(&pilot_states),into(&starts));c.phase=grid_phase(&c.model,&states,&running(c.reference(),steps,pilot_waiting,doppler_allowance(&c.base,&c.model),c.phase),Some((&groups,duration)));
        let costed=evaluate(&c,&states,&begun,None,&Deposits::default());(costed,c,states,begun)})}else{None};
    let kernel=|c:&Candidate,states:&[Trajectory],begun:&[Trajectory],x:&Costed|{
        let allowance=doppler_allowance(&gained(&c.base,&c.model,states),&c.model);
        kernel_interval(&c.model,&state_work(&c.model,states,begun),&running(c.reference(),x.steps,pilot_waiting,allowance,c.phase),interval,pilot_waiting)*waves(x,width())};
    let fourth_cost=fourth_cost.filter(|f|device==Device::Cpu||pilot_cost.as_ref().map_or(true,|p|kernel(&f.1,&f.2,&f.3,&f.0)*(1.+need)<kernel(&pilot_candidate,&pilot_states,&starts,p)));
    let mut least=pilot_cost.as_ref().map_or(f64::INFINITY,|p|p.cost).min(fourth_cost.as_ref().map_or(f64::INFINITY,|f|f.0.cost));
    // The ladder's rungs where the folded models' cost could fall, without the estimates a rung can change (at the pilot's
    // model's step, its totals with those no candidate changes alone, `Candidate::fixed`, and the beats), by more than the prediction's
    // uncertainty and a rung's least check: more of the model explicit, more work per step, one doubling at least. Below
    // that no rung can win or pay for its check (DEFAULT's, THREE's: a near tie of explicit and folded at 1.3 s of the
    // startup's checks; F10's a run of 1 CPU-s, a rung's check ten, its opening with the pilot's draws).
    let open=pilot_cost.as_ref().is_some_and(|p|{let fixed=vec![[pilot_candidate.fixed()+pilot_beats*pilot_horizon;2];outputs];
        let(n,weight)=predicted(&pilot_statistics,&fixed,p.bias,pilot as f64);
        least>p.work*n*weight*(outputs-1) as f64*(1.+need)+checked_step(pilot_states.len(),bound_probes(&pilot_candidate.model,&pilot_states,&binding_at(&pilot_states,&weights,interval)).first,p.steps,interval,period)*p.work/p.steps as f64});
    // A check's work per unit of the work per interval: the same steps' doublings at another model's work (the pilot's model's).
    let per_work=pilot_cost.as_ref().map_or(0.,|p|p.check/p.work);
    // Each rung's first doubling by its clusters (`Candidate::clusters`): the same states' dynamics whatever the drives of round-off's
    // size between them, its biases the same at the rung's tolerances.
    let(mut rung,mut doubled)=(None,Vec::<(Vec<(Vec<usize>,Vec<usize>)>,usize,[f64;3],[f64;2])>::new());
    if open{
        while let Some(mut c)=next_candidate(request,&bare,&mut ladder,&mut failed,&mut shapes){
            c.charged(charge_of(&pilot_candidate.model,&c.model));
            // Its grid on the pilot's states carried into it, bare as a jump leaves them (`grid_phase`).
            let period=slowest_period(&c.reference().0);let mut at=running(c.reference(),steps,pilot_waiting,doppler_allowance(&c.base,&c.model),c.phase);
            let into=|v:&[Trajectory]|->Vec<Trajectory>{v.iter().map(|t|carried(&pilot_candidate.model,&c.model,t)).collect()};
            let(bare,begun)=(into(&pilot_states),into(&starts));
            c.phase=grid_phase(&c.model,&bare,&at,Some((&groups,duration)));at.node_phase=c.phase;
            let parts=state_work(&c.model,&bare,&begun);
            let apriori:Vec<[f64;2]>=c.charges.iter().map(|x|x.map(|x|c.others()+x+pilot_beats*pilot_horizon)).collect();
            let(fewest,weight)=predicted(&pilot_statistics,&apriori,[0.;2],pilot as f64);
            // Its check as the pilot's model's at its own work: spent whether it is taken or not. A rung is taken only for a saving
            // beyond the need's error: one whose floor cannot make it, its check paid, is passed over with its check unspent.
            let(at_work,at_least)=(interval_work(&c.model,&parts,&at,interval,pilot_waiting),least_work(&c.model,&parts,&at,interval,pilot_waiting));
            let check=per_work*at_work;
            if at_least*fewest*weight*(outputs-1) as f64*(1.+need)+check>=least{break;}
            let states=relaxed(&c.model,&at,bare);let binding=binding_at(&states,&weights,interval);let probes=bound_probes(&c.model,&states,&binding);
            // That doubling, made, from the coarse side as the grid's (`grid_phase`): at the fewest steps whose work per interval still is at least the work no step changes (W(s) = a s + least_work), at most the pilot's -- a read of the
            // orbits costs as its steps until there, and the pilot's step is no measure of a rung's: on a Λ scheme's rung of 5 explicit states 0.04 us against the 0.62 us it keeps, each check at 0.04-0.05 us 125-155 CPU-s of a search of 830 for a run of 11.
            // The motion's and the loss's criteria, as dt², bound the steps the check can keep from below
            // (the population's bound may yet be measured smaller), and the work there any step's it keeps -- the explicit
            // states' exact steps resolve what the folded model's step left to the fold (CONTROL_CYCLE's first rung: 118 steps
            // per interval against the folded 34, its evaluation ten CPU-s for a cost twice the least's).
            let tolerances=tolerances(&ceilings,&apriori);let clusters=c.clusters();let deposits=Deposits::default();
            let coarse=((at_least*steps as f64/(at_work-at_least)).ceil() as usize).clamp(1,steps);
            let(coarse,r)=match doubled.iter().find(|x|x.0==clusters){
                Some(x)=>(x.1,[x.2[0]*x.3[0]/tolerances[0],x.2[1]*x.3[0]/tolerances[0],x.2[2]*x.3[1]/tolerances[1]]),
                None=>{let read=running(c.reference(),coarse,pilot_waiting,at.doppler,at.node_phase);let r=step_excess(&Engine::new(&c.model,read),&states,&binding,pilot_waiting,&probes,period,escape(&c.model,&states,&pilot_groups),tolerances,&deposits,fewest*duration);doubled.push((clusters,coarse,r,tolerances));(coarse,r)}
            };
            let fewest_steps=((coarse as f64*r[1].max(r[2]).sqrt()).ceil() as usize).clamp(1,most);
            // There the motion's and the loss's biases, as dt², and the trajectories they leave the statistics: the rung's
            // cost at the largest step its check can keep -- a near tie (RETRO's ladder, ten rungs within its error of the
            // first) is passed over unchecked; one that saves beyond the error is costed.
            let x=(coarse as f64/fewest_steps as f64).powi(2);
            let(n,weight)=predicted(&pilot_statistics,&apriori,[r[1]*x*tolerances[0],r[2]*x*tolerances[1]],pilot as f64);
            if interval_work(&c.model,&parts,&running(c.reference(),fewest_steps,pilot_waiting,at.doppler,at.node_phase),interval,pilot_waiting)*n*weight*(outputs-1) as f64*(1.+need)+check>=least{break;}
            let costed=evaluate(&c,&states,&begun,Some((Check{steps:coarse,population:r[0],loss:r[1],motion:r[2]},tolerances)),&deposits);
            // A rung keeps more of the model explicit: taken only for a saving beyond the prediction's own uncertainty.
            if costed.cost*(1.+need)<least{least=costed.cost;rung=Some((c,states,begun,costed.checked));}
        }
    }
    // The choice: the least weighed cost, one keeping more of the model only for a saving beyond the need's error; with
    // none costed, the pilot's model. Its run takes the pilot's states carried into its clusters, the pilot's statistics
    // and jumps the prediction's (the models' rates differ by the fold's S, the physical atoms' jumps alike), and the step
    // its check took on them: no pilot runs again.
    let fourth_wins=fourth_cost.as_ref().is_some_and(|f|pilot_cost.as_ref().map_or(true,|p|f.0.cost*(1.+need)<p.cost));
    let(chosen,states,begun,known)=if let Some((c,states,begun,checked))=rung{(c,states,begun,checked)}
        else if fourth_wins{let(f,c,states,begun)=fourth_cost.unwrap();(c,states,begun,f.checked)}
        else{(pilot_candidate,pilot_states,starts,pilot_cost.and_then(|p|p.checked))};
    let(jumps,pilot_groups)=(pilot_jumps,pilot_groups);
    let mut chosen=chosen;chosen.settle();
    // The run's trap and region are its model's: an all-folded pilot's hold the unsaturated light shift of a near drive the rung keeps (the resonant probe of 100 mW/cm², cooling of 3: 72 kHz and a region
    // of 97 rad/us against the rung's 100 kHz and 137), and T's potential part and the loss read them.
    let reference=chosen.reference();let(trap,period)=(reference.0,slowest_period(&reference.0));
    let Candidate{model,initial,base,charges,phase,..}=chosen;
    // The generator against its jumps at the pilot's states (`gain`): booked where its decay falls below
    // the jumps' rate.
    let base=gained(&base,&model,&states);
    // The beats the exact steps left out (engine.rs `Segment::resolve`), booked as their bound per unit time
    // over the horizon: within the exponentials' share by construction (`booked_beats`).
    let scales=Scales{hbar_over_mass:model.hbar_over_mass,frequencies:trap.lab};
    let waiting=pilot as f64*duration/(jumps.max(1) as f64);let horizon=waiting.min(duration);
    let pilot_beats=booked_beats(&states,horizon);
    let allowance=doppler_allowance(&base,&model);
    let settings=|steps:usize|running(reference,steps,waiting,allowance,phase);
    // The step (`step_check`; on the pilot's states as the choice checked it): the a-priori total the steps'
    // costs take -- the ledger's estimates with the fold's charge as the pilot measured it. The motion's criteria
    // hold T's step error within the step's share of T's room at its target there, at most the T target itself
    // (`ceilings`, `tolerances`): a step that leaves the motion unresolved still moves the loss.
    let apriori:Vec<[f64;2]>=charges.iter().map(|x|x.map(|c|unfolded(&base,&model)+c+pilot_beats*horizon)).collect();let parts=state_work(&model,&states,&begun);
    let checked=match known{Some(checked)=>checked,None=>{let tolerances=tolerances(&ceilings,&apriori);
        let cost=|k:&Check|{let r=k.ratios();let(n,weight)=predicted(&pilot_statistics,&apriori,[r[0]*tolerances[0],r[1]*tolerances[1]],pilot as f64);
            interval_work(&model,&parts,&settings(k.steps),interval,waiting)*n*weight};
        let reach=predicted(&pilot_statistics,&apriori,[0.;2],pilot as f64).0*duration;
        step_check(&model,&states,&binding_at(&states,&weights,interval),&settings,steps,most,waiting,horizon,pilot as usize,&pilot_groups,period,tolerances,&cost,need,&[],&Deposits::default(),reach)}};
    let tolerances=checked.tolerances;
    // The kept step against T's error over the run (`drift`), each output against T's tolerance there and the loss's: read where its error per period could add up past that;
    // where more stands than that, as dt² the step where it would come to it and its bias, checked again at least as far as that span.
    let chosen={let(probes,sensitivity)=(bound_probes(&model,&states,&binding_at(&states,&weights,interval)),escape(&model,&states,&pilot_groups));
        let limits:Vec<f64>=apriori.iter().zip(&ceilings).map(|(a,&w)|temperature_tolerance(w,a[1]).min(if sensitivity>0.{tolerances[0]/sensitivity}else{f64::INFINITY})).collect();
        let limit=|t:f64|limits[((t/interval).round() as usize).clamp(1,outputs-1)];
        let run_steps=|k:&Check|{let r=k.ratios();predicted(&pilot_statistics,&apriori,[r[0]*tolerances[0],r[1]*tolerances[1]],pilot as f64).0*(k.steps*(outputs-1)) as f64};
        let mut through=0.;let mut kept=checked.chosen;
        let could=(1..outputs).map(|i|kept.motion*tolerances[1]*(times[i]/period).max(1.)/limits[i]).fold(0.,f64::max);
        if !read_pays((probes.first*drift_window(period,duration,interval/kept.steps as f64)) as f64,run_steps(&kept),could,1.){kept}else{loop{
            let d=drift(&model,&settings,kept.steps,&probes,period,duration,&limit,through,run_steps(&kept));
            let read=Check{motion:kept.motion.max(d.excess),loss:kept.loss.max(sensitivity*d.stands/tolerances[0]),..kept};
            if d.excess<=1.||kept.steps==most{break read;}
            let finer=((kept.steps as f64*d.excess.powf(1./ORDER)).ceil() as usize).clamp(kept.steps+1,most);let x=(kept.steps as f64/finer as f64).powf(ORDER);
            kept=Check{steps:finer,population:read.population*x,loss:read.loss*x,motion:read.motion*x};through=d.span;
        }}};
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
    // The control variates (control.rs): the harmonic twin's, and the held twin where the trap beam's
    // intensity is the pilot's mean, when the generator is static.
    let harmonic=Harmonic::new(&model,&settings,&initial,steps,outputs);
    // The need the pilot predicts per output and target, the populations' and T's (its trajectories each a group, with the harmonic control).
    let pilot_needs:Vec<[f64;2]>={
        let means:Vec<Means>=(0..outputs).map(|i|Means{harmonic:pilot_harmonic.mean[i],twin:None}).collect();
        let mut e=summarize(&pilot_groups,&scales,&means);start(&mut e[0],&initial);
        let(_,accounts)=ledger(&model,&booked(&base,pilot_beats*horizon),&e,times,allowance,bias,trap.lab,false,&memory);
        output_ratios(&e,&accounts,trap.lab).iter().map(|r|r.map(|x|(pilot as f64*x*x).min(MOST_TRAJECTORIES as f64))).collect()
    };
    let largest_need:[f64;2]=std::array::from_fn(|k|pilot_needs.iter().fold(0_f64,|m,n|m.max(n[k])));
    // The first batch, each group as far as the reach wants it (module documentation, # Batches): on the CPU FIRST_GROUPS groups, at most the pilot's need -- another batch costs a tail --;
    // on the GPU the pilot's need and its error -- another wave costs a whole trajectory's time, as far as it goes --, at least FIRST_GROUPS groups through the run, at most what its resident blocks hold.
    let pilot_reach=reaches(&pilot_needs.iter().map(|n|n[0].max(n[1])).collect::<Vec<f64>>());
    let ends:Vec<usize>=match &gpu{
        Some((g,_))=>{let width=g.lock().unwrap_or_else(|e|e.into_inner()).width(1).max(FIRST_GROUPS*TASK);let margin=1.+need;
            let wanted:Vec<f64>=pilot_reach.iter().map(|x|x*margin).collect();
            first_ends(&wanted,(wanted[1].ceil() as u64).clamp(FIRST_GROUPS*TASK,width).div_ceil(TASK),FIRST_GROUPS)}
        None=>first_ends(&pilot_reach,(FIRST_GROUPS*TASK).min(pilot_reach[1].ceil() as u64).div_ceil(TASK).max(1),WHOLE_GROUPS),
    };
    let count=ends.len() as u64*TASK;
    // The held twin where it can pay for itself by the pilot's need (`twin_can_pay`): spent in the first batch either way, it held 4-10 % of the CPU's solves and 5-11 % of the kernel's on scenes that never kept it.
    let twin_share=||held_work(&parts,&settings,interval,waiting)/interval_work(&model,&parts,&settings,interval,waiting);
    let intensity=states.iter().map(|t|model.geometry[model.trap_beam].intensity(t.r)).sum::<f64>()/states.len().max(1) as f64;
    let mut twin=if twin_can_pay(count as f64,largest_need,0.){Twin::new(&model,intensity).filter(|_|twin_can_pay(count as f64,largest_need,twin_share()))}else{None};
    let twin_means=twin.as_ref().map(|t|t.means(&model,&request.initial_density,interval,outputs));
    let means=|held:bool|->Vec<Means>{(0..outputs).map(|i|Means{harmonic:harmonic.mean[i],twin:twin_means.as_ref().filter(|_|held).map(|m|m[i])}).collect()};
    // A batch's trajectories, each group through its output `ends`.
    let batch=|first:u64,count:u64,ends:&[usize],twin:Option<&Twin>|->Result<Outcome,Stop>{
        let variates=Variates{harmonic:&harmonic,twin};
        match &gpu{
            None=>Ok(simulate_to(&model,&settings,&initial,seed,first,count,TASK,steps,outputs,&listed,Some(&variates),ends)),
            Some((g,program))=>{let last=ends.iter().copied().max().unwrap_or(outputs-1);let short:Vec<usize>=listed.iter().copied().filter(|&l|l<=last).collect();
                let mut out=g.lock().unwrap_or_else(|e|e.into_inner()).simulate(program,&model,&settings,&initial,seed,first,count,TASK,steps,last+1,&short,Some(&variates),Some(ends)).map_err(Stop::Gpu)?;
                for group in &mut out.groups{group.sums.resize(outputs,Sums::zero());}out.samples.resize(listed.len(),vec![]);Ok(out)}
        }
    };
    let clock=Instant::now();
    let first=batch(0,count,&ends,twin.as_ref())?;
    let wall=clock.elapsed().as_secs_f64();
    // The held twin is kept where its control pays for itself: the trajectories the first batch says
    // the targets need with it, times its share of the work (counted as the step's, `held_work`),
    // against those they need without.
    if twin.is_some(){
        let needed=|means:&[Means]|{let mut e=summarize(&first.groups,&scales,means);start(&mut e[0],&initial);
            let(_,accounts)=ledger(&model,&booked(&base,pilot_beats.max(first.beats)*horizon),&e,times,allowance,bias,trap.lab,false,&memory);worst_ratio(&e,&accounts,trap.lab).powi(2)};
        if needed(&means(true))*(1.+twin_share())>=needed(&means(false)){twin=None;}
    }
    let means=means(twin.is_some());
    // Batches until the targets are met, each output's own (module documentation, # Batches). The pace: a batch's seconds per
    // trajectory and output interval it ran.
    let mut outcome=first;let mut done=count;let mut pace=wall/ends.iter().map(|&e|TASK*e as u64).sum::<u64>().max(1) as f64;
    loop{
        let mut estimates=summarize(&outcome.groups,&scales,&means);start(&mut estimates[0],&initial);
        let(_,accounts)=ledger(&model,&booked(&base,pilot_beats.max(outcome.beats)*horizon),&estimates,times,allowance,bias,trap.lab,false,&memory);
        let counts:Vec<u64>=estimates.iter().map(|e|e.trajectories as u64).collect();
        let worst=output_ratios(&estimates,&accounts,trap.lab);let ratio=worst.iter().fold(0_f64,|m,r|m.max(r[0]).max(r[1]));
        let reach=reaches(&worst.iter().zip(&counts).map(|(r,&n)|n as f64*r[0].max(r[1]).powi(2)).collect::<Vec<f64>>());
        let wanted=reach[1].ceil() as u64;let reach:Vec<u64>=reach.iter().map(|x|(x.ceil() as u64).min(MOST_TRAJECTORIES)).collect();
        let last=(1..outputs).rev().find(|&i|reach[i]>counts[i]);
        let(mut ledger,_)=ledger(&model,&booked(&base,pilot_beats.max(outcome.beats)*horizon),&estimates,times,allowance,bias,trap.lab,last.is_none(),&memory);
        if let(Some(note),Some(flag))=(&bounded,&ledger.flagged){ledger.flagged=Some(format!("{note}；{flag}"));}
        let mut solution=assemble(&model,&estimates,&outcome,&listed,times,trap.lab,done,counts[outputs-1],dt,&request.initial_density,ledger,twin.is_some());
        solution.device=used;solution.device_note=note.clone();solution.needed=wanted;
        if last.is_none(){
            if ratio>1.{solution.reason=Some(format!("统计误差为目标的 {ratio:.1} 倍：轨迹数已达上限 {MOST_TRAJECTORIES}"));}
            return Ok(solution);
        }
        let remaining:u64=(1..outputs).map(|i|reach[i].saturating_sub(counts[i])).sum();
        checkpoint(&solution,Some(pace*remaining as f64*1e3));
        let ends=next_groups(&reach,&counts);
        let next=ends.len() as u64*TASK;
        let clock=Instant::now();
        let more=batch(done,next,&ends,twin.as_ref())?;
        pace=clock.elapsed().as_secs_f64()/ends.iter().map(|&e|TASK*e as u64).sum::<u64>().max(1) as f64;
        outcome.groups.extend(more.groups);for(a,b)in outcome.samples.iter_mut().zip(more.samples){a.extend(b);}outcome.jumps+=more.jumps;outcome.beats=outcome.beats.max(more.beats);
        done+=next;
    }
}

/// Whether the held twin can pay for itself (module documentation, # Control variates), from the trajectories the pilot predicts the populations and T need: its control lowers the populations'
/// errors alone, at `share` of the work on every trajectory it runs beside, so that the run comes at best to the larger of the first batch and T's need, against the larger of them and the populations' without it.
fn twin_can_pay(first:f64,[population,temperature]:[f64;2],share:f64)->bool{(1.+share)*first.max(temperature)<first.max(population).max(temperature)}

/// The bound per unit time of the beats a pilot's exact steps left out (engine.rs `Segment::resolve`), at most the
/// run's share over its horizon (`running`): the run leaves out beats within that share by construction, and the pilot's
/// share is looser where its horizon was shorter than the waiting time it measured (`piloting`).
fn booked_beats(states:&[Trajectory],horizon:f64)->f64{states.iter().map(|t|t.beats).fold(0.,f64::max).min(NUMERICAL_SHARE*POPULATION_TARGET/horizon)}

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
/// A model's ledger's estimates but the fold's charge: what the fold's charge and the step's bias add to (`Accounts::apriori`).
fn unfolded(base:&Ledger,model:&Model)->f64{base.used()-fold_charge(model,model.ground.charge)}
/// The first order in the Doppler shift's allowance: the step's share of the run's room beside the model's other
/// estimates on the population target (`run_room`; the fold's charge takes its own) -- a truncation per step, as the
/// step's own error; its largest met along the trajectories is booked, and one hot atom's must not take the run's room.
fn doppler_allowance(base:&Ledger,model:&Model)->f64{BIAS_SHARE*run_room(POPULATION_TARGET,unfolded(base,model))}
/// A model's gain at sampled states (model.rs `Model::decay_balance`, at most FIRST_GROUPS × 8 of them): the
/// largest −λ of D = i(H − H†) − Σ L†L beyond its eigenvalues' round-off, relative to the largest jump rate
/// met -- the relative error of the jump rate where the no-jump decay misses it (a decay no jump carries,
/// the validation's F10 before the near light's dressing); 0 for a Lindblad generator, unbounded for a
/// gain without jumps.
fn gain(model:&Model,states:&[Trajectory])->f64{
    let every=(states.len()/(FIRST_GROUPS as usize*8)).max(1);
    let(gain,rate)=states.par_iter().step_by(every).map(|t|{let(low,rate,scale)=model.decay_balance(t.cluster,&model.point(t.r,t.kappa),t.time);
        let n=model.dimension(t.cluster) as f64;((-low-crate::linear::round_off(n*scale)).max(0.),rate)}).reduce(||(0.,0.),|a,b|(a.0.max(b.0),a.1.max(b.1)));
    if gain==0.{0.}else if rate>0.{gain/rate}else{f64::INFINITY}
}

/// The ledger's item of the drives' fold (its charge a priori; the solve's ledger charges it as measured).
const FOLD:&str="消去激发态";
/// What the request's setup leaves of its thermal distributions, on every output (the base ledger's `note`, `Candidate::settle`): no choice of the solve changes it and
/// it is no error of the solve's, so that it stays out of the ledger's items, the totals and the flag. The share of the traps' thermal distributions over their regions (U_h ≤ D, Maxwell's
/// momenta) that the bound ensembles leave out (ensemble.rs) -- the unbound atoms and the lowering near the trap's top --, beyond `BIAS_VARIANCE` of the population target, and the ensemble's kinetic temperature where it
/// departs from T by more than T's share: a trap a few k_BT deep holds no bound ensemble as hot as T (none of a Gaussian beam's is hotter than 0.166 D/k_B, ensemble.rs), the reference state's depth in k_BT.
fn initial_note(initial:&Initial)->Option<String>{
    let cut=(initial.cut>BIAS_VARIANCE*POPULATION_TARGET).then(||format!("初始热分布中能量高于阱顶（未束缚）及阱顶附近的 {:.0}% 原子不计入（取束缚原子的降低等温分布 f ∝ e^(−E/k_BT) − 1，E 为相对阱外的机械能）；丢失与布居依赖这一截断",100.*initial.cut));
    let ratio=initial.kinetic();let set=initial.temperature_uk;
    let shallow=((ratio-1.).abs()>BIAS_VARIANCE*TEMPERATURE_TARGET).then(||{
        let k=initial.reference();
        let depth=initial.ensembles[k].as_ref().map_or(initial.depths[k],|e|e.depth)/(KT_PER_UK*set);let t=initial.temperature();
        // One decimal from one up (none for the setting's whole numbers), two significant digits below.
        let digits=|x:f64|if (x-x.round()).abs()<=1e-9*x.abs(){format!("{x:.0}")}else if x>=0.95{format!("{x:.1}")}else{let d=(-x.log10()).ceil().max(0.) as usize+1;format!("{x:.d$}")};
        format!("势阱较浅（D ≈ {} k_BT）：束缚的热分布动能温度为 {} μK，{}设定的 {} μK",digits(depth),digits(t.iter().sum::<f64>()/3.),if ratio<1.{"低于"}else{"高于"},digits(set))});
    match(cut,shallow){(Some(c),Some(s))=>Some(format!("{c}；{s}")),(c,s)=>c.or(s)}
}
/// The fold's charge for the saturation S its folded drives meet: its populations' error bound, S at
/// second order and S² + ς S at fourth (ground.rs; against the population core 0.2-1.5 S² where the series
/// converges, S ≲ 0.5 -- beyond, the charge exceeds the target and the rule keeps the drives).
fn fold_charge(model:&Model,saturation:f64)->f64{fold_charges(model,saturation)[0]}
/// A measured fold's charges (`fold_charges`) with what the atoms it reads leave open of the light's resonance: the elimination's a-priori bound over their reach (model.rs `Model::resonance`), the largest excited weight a ground state can take there, times the share of
/// atoms that may have met it unseen (`unseen`), at second order whatever the fold's: a line the light brings near holds no series. The bound itself says where the light could go, and the charge what the atoms did: the 10 G tweezer's bound of 7 %, its charge 0.47 %, took the rung of two explicit states at twelve times the work
/// by its own weight, T within 1.4 standard errors. Where the pilot keeps few atoms, or the bound passes one, this is the charge: a saturating drive's fold can no longer pass unread.
fn with_open(charges:[f64;2],model:&Model,atoms:f64)->[f64;2]{let open=open_saturation(model.resonance,atoms);[charges[0]+open,charges[1]+std::f64::consts::SQRT_2*open]}
/// The saturation 2η² a bound on the excited weight leaves open over this many atoms (`with_open`).
fn open_saturation(resonance:f64,atoms:f64)->f64{2.*resonance*unseen(atoms)}
/// The fold's charges per target, the populations' (`fold_charge`: at fourth order S² and the first order the transient's delay leaves, ς S, model.rs `Model::ringing_spread`) and T's, for the saturation S its folded drives meet at second or
/// fourth order: T's is √2 S at either -- T is the ratio of the momentum diffusion and the friction, each a rate or a force's
/// gradient of the fold's error S, independent errors in quadrature as the totals' (module documentation, # Shares), and the force
/// folds at second order. Measured gaps, the explicit states' T against the all-folded T on one kick, second-half means over 1 ms
/// on the axes above 4ħω, per S: a tetrahedral MOT's z 0.01 ± 0.33 and x 0.29 ± 0.81 (S = 2.1 %), the moving 3D PGC's x −1.07 ± 0.39
/// and z 2.54 ± 0.49 (S = 2.05 %), a gray molasses's z 0.41 ± 1.44 and x −0.47 ± 0.71 (S = 3.4 %): their root mean square, its noise
/// taken out, 0.8 S -- what a bound of √2 S with the error uniform within it gives (the 0.82 S of the totals' reading of a bound),
/// the PGC's z alone 2.3 standard errors beyond it; the populations within 0.5 S.
fn fold_charges(model:&Model,saturation:f64)->[f64;2]{[if model.fourth{saturation*saturation+model.ringing_spread*saturation}else{saturation},std::f64::consts::SQRT_2*saturation]}

/// A candidate of the fold (module documentation, # The fold's choice): the model at one of the rule's
/// allowances, its initial state, its ledger, and the fold's charge per output and target (`fold_charges`) -- a
/// priori at every output, or as measured (`Candidate::charged`); its beating grid's phase (`grid_phase`).
struct Candidate{model:Model,initial:Initial,base:Ledger,allowance:f64,charges:Vec<[f64;2]>,settled:bool,phase:f64}
/// A candidate's weighed cost (`evaluate` in `run`): at the step its check keeps, its work over the run times the weight of
/// its totals (`predicted`), its check's own work, its work per interval there, that step, its bias and the a-priori totals per output.
/// The check itself (`Steps`); none yet where it was put off, beyond the targets whatever its step (`evaluate`).
struct Costed{cost:f64,check:f64,work:f64,steps:usize,bias:[f64;2],apriori:Vec<[f64;2]>,checked:Option<Steps>}
impl Candidate{
    fn new(request:&Request,ground:Ground,allowance:f64,fourth:bool)->Result<Candidate,String>{
        let model=Model::of_ground(request,ground,fourth)?;let initial=Initial::deferred(&model,request)?;let base=model_ledger(&model,request);
        let charges=vec![fold_charges(&model,model.ground.charge);request.times_us.len()];
        Ok(Candidate{model,initial,base,allowance,charges,settled:false,phase:node_phase()})
    }
    /// The initial ensemble's quadrature (ensemble.rs), which none of the fold's choices reads, for the pilot's and
    /// the chosen model alone, and its note on the base ledger (`initial_note`).
    fn settle(&mut self){
        if self.settled{return;}self.settled=true;self.initial.integrate(&self.model);self.base.note=initial_note(&self.initial);
    }
    /// Its fold's charges as measured (`measured_charge`) in place of the a-priori estimate.
    fn charged(&mut self,charges:Vec<[f64;2]>){self.charges=charges;}
    /// Its ledger's estimates but the fold's.
    fn others(&self)->f64{unfolded(&self.base,&self.model)}
    /// The estimates no candidate of the fold changes: the far components' fold and the beams' interference, the same in
    /// every model of the request. (The others move with the fold: a folded drive's line, narrow across the intensities, takes the tables more nodes
    /// -- RETRO's every drive folded holds 0.55 at 33 nodes and 1.4e-3 at 513, the rung that keeps its drive 1.4e-3 at 65.)
    fn fixed(&self)->f64{let g=&self.model.ground;g.fold_estimates().0+g.interference.iter().map(|x|x.estimate(g)).sum::<f64>()}
    /// Its clusters: each one's ground states and explicit excited states.
    fn clusters(&self)->Vec<(Vec<usize>,Vec<usize>)>{self.model.ground.clusters.iter().map(|c|(c.states.clone(),c.excited.clone())).collect()}
    /// Which of its drives and couplings of the excited states fold.
    fn folds(&self)->Vec<bool>{self.model.ground.excited_items.iter().map(|x|x.folded).collect()}
    /// Whether its own estimates, a priori, lie past the population target at some output, or the resonance does: all that a pilot which keeps no atom leaves to read (T's target
    /// has no regime without atoms; its charges are a priori until `charged`).
    fn past_their_targets(&self)->bool{self.model.resonance>POPULATION_TARGET||self.charges.iter().any(|x|self.others()+x[0]>POPULATION_TARGET)}
    /// Its reference state, the largest initial population: its trap (ω per axis) and depth.
    fn reference(&self)->(Trap,f64){let i=&self.initial;let k=i.reference();(i.traps[k],i.depths[k])}
}
/// The next candidate along the rule's allowances that builds (a model that fails, its tables not
/// converging, is passed over with its error kept), past the rungs whose closure folds the same drives as one in `shapes`, the same model, before their tables.
fn next_candidate(request:&Request,bare:&Ground,ladder:&mut dyn Iterator<Item=f64>,failed:&mut Option<String>,shapes:&mut Vec<Vec<bool>>)->Option<Candidate>{
    for a in ladder{
        let ground=match bare.clustered(a){Ok(ground)=>ground,Err(e)=>{failed.get_or_insert(e);continue;}};
        let folds:Vec<bool>=ground.excited_items.iter().map(|x|x.folded).collect();if shapes.contains(&folds){continue;}
        match Candidate::new(request,ground,a,false){Ok(c)=>{shapes.push(folds);return Some(c);},Err(e)=>{failed.get_or_insert(e);}}
    }
    None
}
/// A trap's slowest axis's period: the span of T's step error (`orbits`).
fn slowest_period(trap:&Trap)->f64{std::f64::consts::TAU/trap.frequencies.iter().copied().fold(f64::INFINITY,f64::min)}
/// The fold's charges a candidate's folded drives would take per output and target (`fold_charges`), at the pilot's
/// states carried into its clusters (`carried`) and at its initial ones: per output their mean 2η² (twice the folded
/// excited population; each output holds few of the pilot's states, and their largest is their extreme, not the
/// run's) over its neighbours as far as they agree (`steady_means`: an atom's bright moment among dark ones held GMD1C's
/// charge at 6.1 % where its states' mean is 0.08 %), with the memory (`Memory::charges`) -- what
/// the run's ledger books there. What the atoms meet rather than the light's pattern extrapolated (an atom held in the
/// dark states of a gray molasses meets little of its bright state's saturation once there, all of it before; one
/// gathered at a red-detuned lattice's bright sites meets more than its mean). The a-priori charge at every output
/// where the pilot kept no state after t = 0.
fn measured_charge(from:&Model,to:&Model,states:&[Trajectory],starts:&[Trajectory],memory:&Memory,interval:f64,outputs:usize)->Vec<[f64;2]>{
    let charge=|t:&Trajectory|{let x=carried(from,to,t);let p=to.point(x.r,x.kappa);fold_charges(to,2.*to.readout(x.cluster,&p,&x.psi,x.time).1)};
    // Per output the states' charges: their sums, their counts, and over the outputs after t = 0 their first two moments and largest.
    let(mut sums,mut counts,mut moments,mut largest)=(vec![[0.;2];outputs],vec![0.;outputs],[[0.;2];2],[0_f64;2]);
    for group in by_output(starts).into_iter().chain(by_output(states)){
        let i=((group[0].time/interval).round() as usize).min(outputs-1);counts[i]+=group.len() as f64;
        let charges:Vec<[f64;2]>=group.par_iter().map(|&t|charge(t)).collect();
        for c in &charges{for k in 0..2{sums[i][k]+=c[k];if i>0{moments[0][k]+=c[k];moments[1][k]+=c[k]*c[k];largest[k]=largest[k].max(c[k]);}}}
    }
    let held:f64=counts[1..].iter().sum();if held==0.{return vec![fold_charges(to,to.ground.charge);outputs];}
    let means:[Vec<f64>;2]=std::array::from_fn(|k|{let sum:Vec<f64>=sums.iter().map(|s|s[k]).collect();let mean=moments[0][k]/held;
        steady_means(&sum,&counts,(moments[1][k]/held-mean*mean).max(0.).sqrt(),largest[k])});
    memory.charges(&(0..outputs).map(|i|with_open([means[0][i],means[1][i]],to,memory.atoms as f64)).collect::<Vec<[f64;2]>>())
}
/// Means per output (`sums` over `counts` states, the first output's left alone) over their neighbours as far as they agree: the mean
/// within the widest window, doubling from the output's own, that agrees with every narrower one within Bernstein's bound on that one's
/// noise, z σ/√m + z² b/(6 m) -- σ the states' standard deviation, b their largest, m the narrower's states, z = √(2 ln n), over n such
/// means (Lepski's rule, as `steady`'s for the errors): the charge of atoms mostly dark has a tail far beyond a normal law's, and one bright
/// state alone at an output is no peak. Noise over a plateau is averaged whole, a transient's real peak beyond it stays. An output
/// without states takes its nearest's.
fn steady_means(sums:&[f64],counts:&[f64],deviation:f64,largest:f64)->Vec<f64>{
    let n=sums.len();let mut out=vec![if counts[0]>0.{sums[0]/counts[0]}else{0.};n];
    let(mut total,mut held)=(vec![0.;n+1],vec![0.;n+1]);for i in 1..n{total[i+1]=total[i]+sums[i];held[i+1]=held[i]+counts[i];}
    let z=(2.*((n-1).max(1) as f64).ln()).sqrt();
    for i in 1..n{
        let(mut narrower,mut half)=(Vec::<(f64,f64)>::new(),0);
        loop{
            let(a,b)=(i.saturating_sub(half).max(1),(i+half).min(n-1));let m=held[b+1]-held[a];
            if m>0.{let x=(total[b+1]-total[a])/m;if narrower.iter().any(|&(y,k)|(x-y).abs()>z*deviation/k.sqrt()+z*z*largest/(6.*k)){break;}narrower.push((x,m));}
            if a==1&&b==n-1{break;}
            half=(2*half).max(1);
        }
        out[i]=narrower.last().map_or(0.,|x|x.0);
    }
    out
}

/// The fold's memory (module documentation, # Ledger), from a pilot's atoms: per atom its jumps up to each
/// output (at its last state kept there or before) and from each output on (at its first state kept there or
/// after: where the pilot keeps every few outputs' states, the jumps in between are at most their difference),
/// and the last output it was kept at -- held there. Per output its atoms' in one row; e^{−n} per whole n, down to where it vanishes.
struct Memory{atoms:usize,upto:Vec<u64>,from:Vec<u64>,last:Vec<usize>,decay:Vec<f64>}
impl Memory{
    fn of(states:&[Trajectory],interval:f64,outputs:usize)->Memory{
        let mut atoms:Vec<(u64,Vec<(usize,u64)>)>=vec![];
        for t in states{let at=(((t.time/interval).round() as usize).min(outputs-1),t.jumps);
            match atoms.iter_mut().find(|a|a.0==t.index){Some(a)=>a.1.push(at),None=>atoms.push((t.index,vec![(0,0),at]))}}
        let count=atoms.len();let(mut upto,mut from,mut last)=(vec![0;outputs*count],vec![0;outputs*count],vec![0;count]);
        for(a,(_,mut kept))in atoms.into_iter().enumerate(){
            kept.sort();last[a]=kept[kept.len()-1].0;
            // The last state kept at or before each output, and the first at or after it.
            let(mut before,mut after)=(0,0);
            for i in 0..outputs{
                while before+1<kept.len()&&kept[before+1].0<=i{before+=1;}
                while after<kept.len()&&kept[after].0<i{after+=1;}
                upto[i*count+a]=kept[before].1;from[i*count+a]=if after<kept.len(){kept[after].1}else{kept[before].1};
            }
        }
        Memory{atoms:count,upto,from,last,decay:(0..).map(|n:u32|(-f64::from(n)).exp()).take_while(|&x|x>0.).collect()}
    }
    /// The fold's charges per output and target with the memory: at each output the largest of its own and each
    /// earlier output's times the share kept since the end of the interval after it, over which that one's
    /// error is made -- of an error made by output j, per atom held at output i ≥ j e^{−n}, n its jumps in between, their mean; all of it where none is held at i.
    fn charges(&self,values:&[[f64;2]])->Vec<[f64;2]>{
        let top:[f64;2]=std::array::from_fn(|k|values.iter().map(|v|v[k]).fold(0.,f64::max));
        (0..values.len()).into_par_iter().map(|i|{let mut most=values[i];
            let held:Vec<(usize,u64)>=(0..self.atoms).filter(|&a|self.last[a]>=i).map(|a|(a,self.upto[i*self.atoms+a])).collect();
            for j in (0..i).rev(){
                let from=&self.from[(j+1)*self.atoms..(j+2)*self.atoms];
                let kept=if held.is_empty(){1.}else{held.iter().map(|&(a,upto)|self.decay.get(upto.saturating_sub(from[a]) as usize).copied().unwrap_or(0.)).sum::<f64>()/held.len() as f64};
                if (0..2).all(|k|top[k]*kept<=most[k]){break;}for k in 0..2{most[k]=most[k].max(values[j][k]*kept);}
            }
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
/// transient's amplitude is within 1/(1 + 2p) of itself, e^{−Γτ/2} = 1/(1 + 2p) at the slowest decay Γ of its cluster's
/// explicit states (the generator's −2 Im of their diagonal), its norm restored.
fn relaxed(model:&Model,settings:&Settings,states:Vec<Trajectory>)->Vec<Trajectory>{
    let slowest:Vec<f64>=(0..model.clusters()).map(|k|{if !model.mixed(k){return 0.;}let(n,n0)=(model.dimension(k),model.ground_dimension(k));
        let g=model.generator(k,&model.point(model.centre(),[0.;3]),0.);(n0..n).map(|e|-2.*g.hamiltonian[e*n+e].im).fold(f64::INFINITY,f64::min)}).collect();
    let engine=Engine::new(model,settings.clone());
    states.into_par_iter().map(|t|{let decay=slowest[t.cluster];if !(decay>0.&&decay.is_finite()){return t;}
        let mut x=engine.coast(&t,2.*(1./BIAS_VARIANCE).ln()/decay,1);let n=model.dimension(x.cluster);
        let norm=x.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();for z in x.psi[..n].iter_mut(){*z=*z*(1./norm);}x}).collect()
}

/// The first batch's groups (module documentation, # Batches), each its last output: of `count` groups the g-th, after `TASK` g trajectories, as far as an output wants it (`reach`, the trajectories each output wants passed by); the first `whole` through the run, on which the
/// first standard errors at its late outputs stand.
fn first_ends(reach:&[f64],count:u64,whole:u64)->Vec<usize>{
    let last=reach.len()-1;
    (0..count).map(|g|if g<whole{last}else{(1..=last).rev().find(|&i|((g*TASK) as f64)<reach[i]).unwrap_or(1)}).collect()
}
/// A batch's groups (module documentation, # Batches), each its last output: the g-th, after its `TASK` g trajectories
/// before it, as far as an output still wants it -- its reach beyond its trajectories so far, at most doubled.
fn next_groups(reach:&[u64],counts:&[u64])->Vec<usize>{
    let mut ends=vec![];
    loop{let before=ends.len() as u64*TASK;
        match (1..reach.len()).rev().find(|&i|counts[i]+before<reach[i].min(2*counts[i])){Some(i)=>ends.push(i),None=>return ends}}
}

/// The step search (module documentation): from `first` each step's check (`check`, or the `known` one where
/// it was checked) and the next step from it (`next_step`) -- from the second check on by the excess's own power over the last two,
/// the criteria going as dt^q with q from 2 to 3 where the square root alone crept up on a step in ten checks (`power`) --,
/// until a step comes up twice, at most 8 checks;
/// with `measure` (the measured search) the population error measured where the population decides a step; with `hand_over`, a bound's search that a measured one takes up, until the first step its bound alone misses.
fn search(first:usize,most:usize,known:&[Check],check:&dyn Fn(usize)->Check,measure:Option<&dyn Fn(usize)->f64>,hand_over:bool,checked:&mut Vec<Check>){
    let mut steps=first;
    for _ in 0..8{
        if checked.iter().any(|c|c.steps==steps){break;}
        let mut c=known.iter().find(|c|c.steps==steps).copied().unwrap_or_else(||check(steps));
        if let Some(m)=measure.filter(|_|c.decided_by_population()){c.population=m(steps);}
        let q=checked.last().map_or(2.,|last:&Check|power(last,&c));
        checked.push(c);if hand_over&&c.decided_by_population()&&c.population>1.{break;}
        steps=next_step(&c,measure.is_some(),most,q);
    }
}

/// The search's next step after a check (module documentation): the square root of its excess away (all scale
/// as dt²) -- in the measured search the excess of the criteria evaluated: where the motion or the loss missed,
/// the population was not measured, and its bound, which over-counts several times where that search runs, is
/// left out. The excess goes as dt^q (`power`).
fn next_step(c:&Check,measured:bool,most:usize,q:f64)->usize{
    let missed=c.motion.max(c.loss);let excess=if measured&&missed>1.{missed}else{c.excess()};
    ((c.steps as f64*excess.powf(1./q)).ceil() as usize).clamp(1,most)
}
/// The power q of dt the excess goes as between two checks at other steps, between the integrator's order and twice it:
/// another criterion's turn at the largest, or the states' noise, must not throw the next step far.
fn power(a:&Check,b:&Check)->f64{
    let(ex,st)=((a.excess()/b.excess()).ln(),(b.steps as f64/a.steps as f64).ln());
    if st==0.||!ex.is_finite(){return ORDER;}(ex/st).clamp(ORDER,2.*ORDER)
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

/// The relative standard error of the trajectories the pilot predicts (module documentation): the need goes as the largest
/// statistic over its target squared (`statistics`, CONFIDENCE standard errors from the pilot's groups, T's target its
/// regime's), so twice that statistic's relative error, its jackknife over the groups. Zero where fewer than three groups or no
/// statistic stand.
fn need_error(groups:&[Group],scales:&Scales,means:&[Means],initial:&Initial,frequencies:[f64;3])->f64{
    let statistic=|groups:&[&Group]|->f64{let mut e=summarize(groups,scales,means);start(&mut e[0],initial);
        statistics_of(&e,frequencies).iter().skip(1).map(|s|(s.population/POPULATION_TARGET).max(s.weight*s.temperature/TEMPERATURE_TARGET)).fold(0.,f64::max)};
    let n=groups.len();if n<3{return 0.;}
    let all=statistic(&groups.iter().collect::<Vec<&Group>>());if !(all>0.){return 0.;}
    let left:Vec<f64>=(0..n).into_par_iter().map(|g|statistic(&groups.iter().enumerate().filter(|&(i,_)|i!=g).map(|(_,x)|x).collect::<Vec<&Group>>())).collect();
    let mean=left.iter().sum::<f64>()/n as f64;
    2.*((n-1) as f64/n as f64*left.iter().map(|x|(x-mean).powi(2)).sum::<f64>()).sqrt()/all
}

/// A checked step (module documentation): its steps per interval and its ratios to the tolerances
/// (`tolerances`) -- the population criterion's (its bound's, or where measured the upper confidence bound
/// of its measured error), the loss's (absolute as the populations') and the motion's (the largest of the
/// impulse's, the orbits' and the jumps', and on the kept step `drift`'s); 1 meets each.
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
/// `first` (`search`, the next step from the criteria evaluated, `next_step`) up to the first step its bound alone misses, then, where the population criterion
/// decides that rule's step, the search from it with the population error measured where it decides. Kept: the
/// largest met step, a finer one only for a predicted saving beyond the need's relative error `need` (`kept`; the
/// weighed cost `cost`), and the rule taken once more over every step from the largest to the finest candidate; or the
/// last checked when none met. A met step's bias is at most the step's share of the run's room (`tolerances`): it leaves
/// the statistics their balance by construction. Every check after the searches is the measured list's (for
/// `checked_work`); the tolerances the checks took, for the step's bias. `known`: checks already made, at those tolerances. `binding`: T's weight at each state (`binding_at`). `reach`: the time (us) the trajectories the run predicts take together (`orbits`).
struct Steps{bound:Vec<Check>,measured:Vec<Check>,chosen:Check,probes:usize,tolerances:[f64;2]}
fn step_check(model:&Model,states:&[Trajectory],binding:&[f64],settings:&dyn Fn(usize)->Settings,first:usize,most:usize,waiting:f64,horizon:f64,subsample:usize,groups:&[Group],period:f64,
    tolerances:[f64;2],cost:&dyn Fn(&Check)->f64,need:f64,known:&[Check],deposits:&Deposits,reach:f64)->Steps{
    // The orbits' probes and the loss's sensitivity (`bound_probes`, `escape`), at the pilot's groups.
    let(probes,escape)=(bound_probes(model,states,binding),escape(model,states,groups));
    let doubled=|steps:usize|{let r=step_excess(&Engine::new(model,settings(steps)),states,binding,waiting,&probes,period,escape,tolerances,deposits,reach);Check{steps,population:r[0],loss:r[1],motion:r[2]}};
    let measure=|steps:usize|accumulated_excess(&Engine::new(model,settings(steps)),states,horizon,subsample,tolerances[0]);
    let checked=|steps:usize|{let mut c=doubled(steps);if c.decided_by_population(){c.population=measure(steps);}c};
    let mut bound=vec![];search(first,most,known,&doubled,None,true,&mut bound);
    let grown=*bound.iter().filter(|c|c.met()).min_by_key(|c|c.steps).unwrap_or(&bound[bound.len()-1]);
    let mut measured=vec![];if grown.decided_by_population(){search(grown.steps,most,&bound,&doubled,Some(&measure as &dyn Fn(usize)->f64),false,&mut measured);}
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
    Steps{bound,measured,chosen,probes:probes.first,tolerances}
}
/// A step check's own work (ns, `step_check`) at `work`'s work per step (per interval over the steps): per
/// checked step a step and two halves on every state and the orbits' jump-free flow over the trap's `period` in
/// steps of dt and of dt/2 from every probe (`checked_step`), and where the population error was measured the
/// jump-free flow over the horizon in steps of dt and of dt/2 from the subsample (the first layer of `accumulated_excess`).
fn checked_work(checked:&Steps,states:usize,subsample:usize,horizon:f64,interval:f64,period:f64,work:&dyn Fn(usize)->f64)->f64{
    let doubled=|c:&Check|checked_step(states,checked.probes,c.steps,interval,period)*work(c.steps)/c.steps as f64;
    checked.bound.iter().map(doubled).sum::<f64>()+checked.measured.iter().map(|c|if checked.bound.iter().any(|b|b.steps==c.steps){0.}else{doubled(c)}
        +3.*subsample.min(states) as f64*(horizon/interval).max(1./c.steps as f64)*work(c.steps)).sum::<f64>()
}
/// The steps a step check's doubling takes at `steps` per interval (`step_excess`): a step and two halves on every
/// state, and from every probe the orbits' over the trap's period in steps of dt and of dt/2.
fn checked_step(states:usize,probes:usize,steps:usize,interval:f64,period:f64)->f64{3.*states as f64+3.*probes as f64*(period*steps as f64/interval).ceil()}

/// The tolerances of a step's criteria (module documentation, # Shares): per target the step's share s of the run's room
/// beside the a-priori totals `apriori` per output (`run_room`), its least over the outputs after t = 0 -- T's at its target ε_T/w
/// there (`weights`, `ceiling`), at most ε_T: a step that leaves the motion unresolved still moves the loss.
fn tolerances(weights:&[f64],apriori:&[[f64;2]])->[f64;2]{
    let population=BIAS_SHARE*apriori.iter().skip(1).map(|a|run_room(POPULATION_TARGET,a[0])).fold(POPULATION_TARGET,f64::min);
    let temperature=apriori.iter().zip(weights).skip(1).map(|(a,&w)|temperature_tolerance(w,a[1])).fold(TEMPERATURE_TARGET,f64::min);
    [population,temperature]}
/// T's tolerance at an output (`tolerances`): the step's share of the run's room at its target there, ε_T/w (`regime`), beside its a-priori total, at most ε_T.
fn temperature_tolerance(weight:f64,apriori:f64)->f64{(BIAS_SHARE*run_room(TEMPERATURE_TARGET/weight,apriori)).min(TEMPERATURE_TARGET)}

/// The step-doubling excess of a step (module documentation), each criterion over its tolerance
/// (`tolerances`; 1 meets it): the population criterion's bound -- from all the sampled states one step
/// against two halves, the populations' difference times the steps of a waiting time --; the loss's, the
/// escape's sensitivity times the motion's energy error, the orbits' or the jumps' (`escape`); and the
/// motion's, the largest of the light impulse's relative error over one step (the same doublings, over the states as T's target weighs them, `binding`), the
/// orbits' error over a trap period (`orbits`, `reach` the time the run's trajectories take together) and the jumps' (`jumps`).
fn step_excess(engine:&Engine,states:&[Trajectory],binding:&[f64],waiting:f64,probes:&Probes,period:f64,escape:f64,tolerances:[f64;2],deposits:&Deposits,reach:f64)->[f64;3]{
    let dt=engine.settings.dt;let richardson=4./3.;let count=states.len().max(1) as f64;
    // The states' doublings on the threads, summed in their order (the sums do not depend on the threads).
    let doubled:Vec<([f64;2],[f64;3],([f64;3],[f64;3]))>=states.par_iter().map(|t|engine.doubling(t,dt)).collect();
    // The impulse's relative error: over the states as T's target weighs them against its tolerance where it binds, and over all against the T target itself.
    let(mut population,mut difference,mut impulse,mut whole)=(0.,0.,0.,[0.;2]);
    for((motion,internal,_),&w)in doubled.iter().zip(binding){difference+=w*motion[0];impulse+=w*motion[1];whole[0]+=motion[0];whole[1]+=motion[1];population+=internal[0];}
    let bound=richardson*population/count*(waiting/dt).max(1.);
    let relative=if impulse>0.{richardson*difference/impulse}else{0.};let anywhere=if whole[1]>0.{richardson*whole[0]/whole[1]}else{0.};
    // The motion's error is the largest of its criteria: below the others' -- the loss's reads the energy's alone -- the orbits' moves nothing.
    let heat=heating(engine,states,&deposits.of(engine.model,states,engine.settings.doppler),&doubled.iter().map(|d|d.2).collect::<Vec<_>>());
    let floor=if escape>0.{heat}else{heat.max(relative).max(anywhere*tolerances[1]/TEMPERATURE_TARGET)};
    let energy=orbits(engine,probes,period,tolerances[1].min(if escape>0.{tolerances[0]/escape}else{f64::INFINITY}),floor,reach/dt).max(heat);
    [bound/tolerances[0],escape*energy/tolerances[0],(relative.max(energy)/tolerances[1]).max(anywhere/TEMPERATURE_TARGET)]
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
    let ends:Vec<([f64;3],[f64;3])>=states.par_iter().map(|t|{let end=engine.coast(t,engine.settings.dt,1);(end.r,end.kappa)}).collect();
    heating(engine,states,&deposits(engine.model,states,engine.settings.doppler),&ends)
}
/// A sampled state's ψ normalized, and its hazard ⟨ψ|i(H − H†)|ψ⟩ with the generator at a point, at the state's time.
fn normalized(model:&Model,t:&Trajectory)->[Complex;MOST_MIXED]{
    let n=model.dimension(t.cluster);let size=t.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>().sqrt();
    let mut psi=[Complex::ZERO;MOST_MIXED];for i in 0..n{psi[i]=t.psi[i]*(1./size);}psi
}
fn hazard(model:&Model,t:&Trajectory,psi:&[Complex],r:[f64;3],kappa:[f64;3])->f64{
    let n=model.dimension(t.cluster);let h=model.generator(t.cluster,&model.point(r,kappa),t.time).hamiltonian;
    (0..n).map(|a|(0..n).map(|b|(psi[a].conj()*(h[a*n+b]-h[b*n+a].conj())*psi[b]*Complex::I).re).sum::<f64>()).sum::<f64>()
}
/// Per sampled state what of `jumps` no step changes, at a Doppler allowance: its hazard and its expected deposit's cross term and recoil (`deposit`).
fn deposits(model:&Model,states:&[Trajectory],doppler:f64)->Vec<[f64;3]>{
    states.par_iter().map(|t|{let psi=normalized(model,t);let(cross,recoil)=deposit(model,t,&psi,doppler);[hazard(model,t,&psi,t.r,t.kappa),cross,recoil]}).collect()
}
/// `deposits` kept over a candidate's checks on its sampled states, at the Doppler allowance last asked for: every check's but for the step.
#[derive(Default)]
struct Deposits(std::cell::RefCell<Option<(f64,Vec<[f64;3]>)>>);
impl Deposits{
    fn of(&self,model:&Model,states:&[Trajectory],doppler:f64)->std::cell::Ref<'_,Vec<[f64;3]>>{
        if !self.0.borrow().as_ref().is_some_and(|x|x.0==doppler){*self.0.borrow_mut()=Some((doppler,deposits(model,states,doppler)));}
        std::cell::Ref::map(self.0.borrow(),|x|&x.as_ref().unwrap().1)
    }
}
/// `jumps` from its `deposits` and each state's (r, κ) at the end of its jump-free step: per state the hazard's change over that step.
fn heating(engine:&Engine,states:&[Trajectory],deposits:&[[f64;3]],ends:&[([f64;3],[f64;3])])->f64{
    let(model,dt)=(engine.model,engine.settings.dt);
    let parts:Vec<[f64;4]>=states.par_iter().zip(deposits).zip(ends).map(|((t,&[rate,cross,recoil]),&(r,kappa))|{
        let psi=normalized(model,t);
        [rate,rate*rate,cross*(hazard(model,t,&psi,r,kappa)-rate),rate*recoil]}).collect();
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

/// The loss's sensitivity to the motion's energy scale: of the pilot's trajectories the share L lost after a state of theirs was
/// bound (atoms the light heated over the top; an unbound one leaving is no escape), L ln(1/L). An escape over a barrier from a
/// thermal tail goes as L = e^{−D/k_BT}: a relative error δ of the motion's energy moves L by (D/k_BT) δ of itself, and the D/k_BT
/// of the tail the atoms were lost from is the one L itself measures, ln(1/L) -- the bound atoms' mean energy need not be that
/// tail's (a gray molasses at 2 µK with one atom of 32 lost: D/k_BT of over a hundred on that mean, its step 0.385 us against
/// 0.625 us with none lost, the loss at either within 0.01 % at 8192 trajectories). Zero where none escaped.
fn escape(model:&Model,states:&[Trajectory],groups:&[Group])->f64{
    let(hm,outputs)=(model.hbar_over_mass,groups.first().map_or(0,|g|g.sums.len()));if outputs==0{return 0.;}
    let bound:Vec<u64>=states.par_iter().filter_map(|t|{let p=model.point(t.r,t.kappa);let k:f64=t.kappa.iter().map(|k|k*k).sum();
        (0.5*hm*k+model.trap_potential(t.cluster,&p,&t.psi,t.time)<0.).then_some(t.index)}).collect();
    let share=groups.iter().enumerate().filter(|(k,g)|g.sums[outputs-1].held==0.&&bound.contains(&(*k as u64))).count() as f64/groups.len() as f64;
    if share>0.{-share*share.ln()}else{0.}
}

/// T's weight w at each sampled state (`regime`): its output's, where the pilot's atoms are.
fn binding_at(states:&[Trajectory],weights:&[f64],interval:f64)->Vec<f64>{states.iter().map(|t|weights[((t.time/interval).round() as usize).min(weights.len()-1)]).collect()}
/// The orbits' probes: per pilot trajectory its states bound to the trap -- their mechanical energy below the trap's far
/// level (model.rs `Model::trap_potential`) --, the atoms whose T the step's error stays on (an unbound one leaves within a
/// transit of the trap's region): in layers (`orbits`), as T's target weighs them (`binding`, `layer_place`) -- T's step error is read where T binds --: first the middle of their weights, then evenly between those taken, at most `TASK`
/// states per trajectory.
pub struct Probes{states:Vec<Trajectory>,first:usize}
pub fn bound_probes(model:&Model,states:&[Trajectory],binding:&[f64])->Probes{
    let hm=model.hbar_over_mass;
    let bound:Vec<bool>=states.par_iter().map(|t|{let p=model.point(t.r,t.kappa);0.5*hm*t.kappa.iter().map(|k|k*k).sum::<f64>()+model.trap_potential(t.cluster,&p,&t.psi,t.time)<0.}).collect();
    let mut indices:Vec<u64>=states.iter().map(|t|t.index).collect();indices.sort();indices.dedup();
    let own:Vec<(Vec<&Trajectory>,Vec<f64>)>=indices.iter().map(|&i|states.iter().zip(&bound).zip(binding).filter(|((t,&b),_)|t.index==i&&b).map(|((t,_),&w)|(t,w)).unzip()).filter(|v:&(Vec<&Trajectory>,Vec<f64>)|!v.0.is_empty()).collect();
    let mut probes=vec![];let mut taken=vec![vec![];own.len()];
    for j in 1..=TASK as u32{for((v,w),seen)in own.iter().zip(&mut taken){let k=layer_place(j,w);if !seen.contains(&k){seen.push(k);probes.push(v[k].clone());}}}
    Probes{first:own.len(),states:probes}
}
/// The j-th layer's place among a trajectory's states of these weights: j's radical inverse in base 2 (1/2, 1/4, 3/4, 1/8, ...) of their sum, so that the states are taken as they weigh; where they have none, of their number.
fn layer_place(j:u32,weights:&[f64])->usize{
    let mut f=0.;let(mut x,mut w)=(j,0.5);while x>0{if x&1==1{f+=w;}x>>=1;w*=0.5;}
    let total:f64=weights.iter().sum();if !(total>0.){return ((f*weights.len() as f64) as usize).min(weights.len()-1);}
    let mut sum=0.;for(k,w)in weights.iter().enumerate(){sum+=w;if sum>f*total{return k;}}weights.len()-1
}

/// The conservative motion's step error on T (module documentation): from each probe the jump-free flow
/// over one period of the trap's slowest axis (the solve's T windows' span, past the run where it is
/// shorter: a window short of a period weighs the orbit's phase, not the step) in steps of dt and of dt/2, ψ kept
/// normalized, the mean κ² per axis over it compared -- 4/3 their difference, the probe's orbit's own
/// error, which the window means of T keep. Per axis the probes' mean difference relative to their mean
/// κ² and its CONFIDENCE standard errors, each probe's difference at the share of atoms that keep its orbit (the flow's no-jump survival at its end, module documentation): their sum, the largest axis. Zero without probes. A few probes' orbits are
/// chaotic -- dt and dt/2 part ways, their κ² windows 20-40 % apart at the step the mean differs by 1 % --, so that with one
/// probe per pilot trajectory the standard error rested on those few and the 3D PGC's step moved by 1.8 times between its
/// pilot's draws: the probes' layers (`bound_probes`) are added, doubling their number, until the errors are within the
/// step's share of `tolerance` or the magnitude less the errors lies beyond it, or none are left (the same steps within 6 %). Where its magnitude and errors lie within `floor`, what the step's other
/// criteria hold the motion's error to, it reads no further, from a group's trajectories' probes up: the flows are the dearest of a check on an exact step (of PGCP_10's search all of its 36 000 exact steps), and most stand below the others.
/// And a layer is read only while its flows take fewer steps than what the read's errors could still move of the `run` (its steps at this step, `read_pays`, as `drift`'s layers): between the sizes the errors leave, as far as they lie above `floor`. Where a period is several runs
/// a layer of 8 probes is a tenth of the run, and a read of every layer ten of them at every check (a Λ scheme's rung over 100 us at a slowest period of 440 us: 90 CPU-s a check at its step, against a run of 11).
fn orbits(engine:&Engine,probes:&Probes,period:f64,tolerance:f64,floor:f64,run:f64)->f64{
    if probes.states.is_empty(){return 0.;}
    // The flow takes no jump over the period: its exponentials are held within the step's share over the period as a
    // run's are over a waiting time (`running`) -- summed past the waiting they were set for, their truncation would be read
    // as the step's T error (the 3D PGC's 2-3 times at the step's share s²). The beats stay the run's: the flows at dt and
    // at dt/2 leave out the same ones, their bounds going as the share with the step, so what they leave out reads no step error, and held tighter
    // the read was of another integrator than the run's, at many times its work (a saturated probe's rung, 2 ms a step on a grid of 3000 nodes where the run takes its plain path).
    let mut held=engine.settings.clone();held.truncation=held.truncation.min(NUMERICAL_SHARE*POPULATION_TARGET*held.dt/period);
    let engine=&Engine::new(engine.model,held);
    let(dt,model)=(engine.settings.dt,engine.model);let steps=((period/dt).ceil() as usize).max(1);
    // A flow's mean κ² per axis, its no-jump survival and its steps: at most `n`, ending where the survival lies within `least`.
    let flow=|t:&Trajectory,h:f64,n:usize,least:f64|->([f64;3],f64,usize){
        let mut x=t.clone();let size=model.dimension(x.cluster);let square=|x:&Trajectory|x.psi[..size].iter().map(|z|z.norm_sqr()).sum::<f64>();
        let(mut sum,mut kept,mut last)=([0.;3],1_f64,square(&x));let mut taken=0;
        while taken<n&&kept>least{
            x=engine.coast(&x,h,1);let now=square(&x);kept*=now/last;let root=now.sqrt();for z in x.psi[..size].iter_mut(){*z=*z*(1./root);}last=1.;
            for a in 0..3{sum[a]+=x.kappa[a]*x.kappa[a];}taken+=1;}
        (sum.map(|v|v/taken as f64),kept,taken)};
    let pair=|t:&Trajectory|{let(a,kept,taken)=flow(t,dt,steps,NUMERICAL_SHARE*tolerance);let(b,also,_)=flow(t,0.5*dt,2*taken,0.);at_share(a,b,kept.min(also))};
    let first=probes.first.min(probes.states.len());let mut windows:Vec<[[f64;3];2]>=vec![];let mut count=first.min(TASK as usize);
    loop{
        windows.extend(probes.states[windows.len()..count].par_iter().map(|t|pair(t)).collect::<Vec<_>>());
        let axes=paired_error(&windows);
        let bound=axes.iter().map(|x|x[0]+x[1]).fold(0.,f64::max);
        if bound<=floor{return bound;}
        if count>=first&&(axes.iter().map(|x|x[1]).fold(0.,f64::max)<=BIAS_SHARE*tolerance||axes.iter().any(|x|x[0]-x[1]>tolerance)||count==probes.states.len()){return bound;}
        let next=if count<first{(2*count).min(first)}else{(2*count).min(probes.states.len())};
        let(high,low)=(axes.iter().map(|x|x[0]+x[1]).fold(0.,f64::max)/tolerance,axes.iter().map(|x|x[0]-x[1]).fold(floor,f64::max)/tolerance);
        if !read_pays(((next-count)*steps) as f64,run,high,low){return bound;}
        count=next;
    }
}

/// A pair's windows, at dt and at dt/2, with their difference at a share of the atoms (`orbits`): what `paired_error` reads of it.
fn at_share(coarse:[f64;3],fine:[f64;3],share:f64)->[[f64;3];2]{[std::array::from_fn(|a|fine[a]+share*(coarse[a]-fine[a])),fine]}

/// Per axis the magnitude of the paired windows' error, 4/3 the difference of the mean κ² at dt (a window's first) and at dt/2
/// relative to the latter's mean, and its CONFIDENCE standard errors. Zeros on an axis without κ².
fn paired_error(windows:&[[[f64;3];2]])->[[f64;2];3]{
    let n=windows.len() as f64;
    std::array::from_fn(|a|{
        let reference=windows.iter().map(|w|w[1][a]).sum::<f64>()/n;if !(reference>0.){return [0.;2];}
        let d:Vec<f64>=windows.iter().map(|w|4./3.*(w[0][a]-w[1][a])/reference).collect();let m=d.iter().sum::<f64>()/n;
        [m.abs(),CONFIDENCE*(d.iter().map(|x|(x-m).powi(2)).sum::<f64>()/(n*(n-1.).max(1.))).sqrt()]
    })
}

/// What of an error stands against a limit: its magnitude less its errors up to the limit, and from there on the limit and what lies beyond it, the squares' difference with the errors, as the budget's quadrature: zero at the noise, at the limit where the magnitude lies beyond it by its errors, the magnitude far above them.
fn standing(error:[f64;2],limit:f64)->f64{let beyond=error[0]-limit;if beyond>error[1]{limit+(beyond*beyond-error[1]*error[1]).sqrt()}else{(error[0]-error[1]).max(0.)}}

/// The spans in succession on which nothing stands that stop T's step error over the run (`drift`). A real error x, its errors at a span e (z standard errors, z the read's `looks_confidence`), reads quiet with the
/// chance Φ(z (1 − x/e)); a span's errors are two or three times its last's, so that the spans are nearly independent and n of them leave that chance to the n-th power. It comes to
/// Φ(−`CONFIDENCE`), the 2.3 % every other statistic here leaves on one side, at x/e = 1.78, 1.40, 1.22, 1.11 for n = 1, 2, 3, 4 at z = 2.57. Each span doubles the read, and doubling the probes instead would lower that x by √2 = 1.41:
/// a second span lowers it by 1.27, a third by 1.15, so two is the last that pays about as much. (The retro-reflected tweezer's error on the first layer of probes stands 1.2-2.6 times its errors at
/// 0.05-0.10 us; with the step forced to 0.104, 0.078, 0.069 and 0.062 us the read moves it on 11, 12, 7 and 4 of 12 seeds at z = 2.33, on the same seeds 11, 12, 9 and 4 at 2.)
const QUIET_SPANS:usize=2;
/// The errors of a read that takes the largest of `looks` noisy sizes, in standard errors: z = √(2 ln n), the largest of n such errors (as `steady`), at least `CONFIDENCE`.
fn looks_confidence(looks:f64)->f64{(2.*looks.ln()).sqrt().max(CONFIDENCE)}
/// The steps of `drift`'s spans' unit: the trap's slowest period, at most the run.
fn drift_window(period:f64,duration:f64,dt:f64)->usize{((period.min(duration)/dt).ceil() as usize).max(1)}
/// Whether `drift`'s further reads of so many steps pay (each in steps of dt and of dt/2, three times that): a stand ρ times the limit takes the run ρ^(1/p) − 1 of its `run` steps more, what could stand against what already does.
fn read_pays(steps:f64,run:f64,could:f64,top:f64)->bool{3.*steps<=run*(could.powf(1./ORDER)-top.powf(1./ORDER))}
/// What of T's step error over the run stands (`drift`): at the span where it comes closest to or furthest past its limit there, its share of that limit, its size and the span.
#[derive(Clone,Copy,Debug,Default)]
pub struct Drift{pub excess:f64,pub stands:f64,pub span:f64}
/// T's step error over the run (module documentation): from each probe a run's own flow, its jumps in on the same streams, in
/// steps of dt and of dt/2, over 1, 2, 4... periods of the trap's slowest axis up to the run, each pair carried on from where it
/// stopped, the mean κ² per axis over its last period compared (`paired_error`, its errors at the confidence of the read's looks, the axes times the spans, `looks_confidence`), against `limit` at the probes' times by then (their
/// tolerances' harmonic mean: the pooled error is the mean of theirs). A span reads the probes' first layer, more only for what could raise the largest stand beyond the limit, which sets the new step --
/// a stand beyond it, or a size that the errors of all the layers would leave beyond it --: until, on more probes than the first layer's, the largest stand's errors lie within the step's share of
/// its size and it lies beyond the limit by them (the pairs' differences are heavy-tailed: 39 ± 26 % on 64 probes of the retro-reflected tweezer's quiet axis, ±7 % on all), or nothing is left that could,
/// or no layers are left, or the next layer takes more steps than what it could settle moves of the `run` (its steps at this step): a stand ρ times the limit takes the run ρ^(1/p) − 1 of them more.
/// The spans go on until nothing stands (`standing`) on `QUIET_SPANS` in succession, and at least `through`: a step checked again as far as the coarser step's stood.
pub fn drift(model:&Model,settings:&dyn Fn(usize)->Settings,steps:usize,probes:&Probes,period:f64,duration:f64,limit:&dyn Fn(f64)->f64,through:f64,run:f64)->Drift{
    if probes.states.is_empty(){return Drift::default();}
    let(coarse,fine)=(Engine::new(model,settings(steps)),Engine::new(model,settings(2*steps)));let dt=coarse.settings.dt;
    let window=drift_window(period,duration,dt);let end=((duration/dt).ceil() as usize).max(window);
    let wider=looks_confidence(3.*(1.+(end as f64/window as f64).log2().ceil()))/CONFIDENCE;
    struct Pair{coarse:Trajectory,fine:Trajectory,done:usize,window:Option<[[f64;3];2]>}
    // A flow's next `count` steps, the κ² of the last `tail`; none once its atom is lost.
    let flow=|e:&Engine,x:&mut Trajectory,count:usize,tail:usize|->Option<[f64;3]>{let mut sum=[0.;3];
        for i in 0..count{if !x.alive{return None;}e.step(x);if i+tail>=count{for a in 0..3{sum[a]+=x.kappa[a]*x.kappa[a];}}}
        x.alive.then(||sum.map(|v|v/tail as f64))};
    let(total,first)=(probes.states.len(),probes.first.min(probes.states.len()));
    let mut pairs:Vec<Pair>=vec![];let(mut reach,mut most,mut quiet)=(window,Drift::default(),0);
    loop{
        let mut count=first;
        let(largest,at)=loop{
            if pairs.len()<count{pairs.extend(probes.states[pairs.len()..count].iter().map(|t|Pair{coarse:t.clone(),fine:t.clone(),done:0,window:None}));}
            pairs[..count].par_iter_mut().for_each(|p|{let left=reach-p.done;let tail=window.min(left);
                if left>0{let(a,b)=(flow(&coarse,&mut p.coarse,left,tail),flow(&fine,&mut p.fine,2*left,2*tail));p.done=reach;p.window=a.zip(b).map(|(a,b)|[a,b]);}});
            let windows:Vec<[[f64;3];2]>=pairs[..count].iter().filter_map(|p|p.window).collect();let axes=if windows.len()<2{[[0.;2];3]}else{paired_error(&windows).map(|[size,errors]|[size,wider*errors])};
            let at=count as f64/probes.states[..count].iter().map(|t|1./limit(t.time+reach as f64*dt)).sum::<f64>();
            let largest=axes.into_iter().max_by(|a,b|standing(*a,at).total_cmp(&standing(*b,at))).unwrap();
            let(layers,top)=((count as f64/total as f64).sqrt(),most.excess.max(1.));let could=axes.iter().map(|x|x[0]-x[1]*layers).fold(0.,f64::max)/at;
            let pays=read_pays((((2*count).min(total)-count)*reach) as f64,run,could,top);
            if could<=top||!pays||count==total||(largest[0]-largest[1]>at&&largest[1]<=BIAS_SHARE*largest[0]&&count>first){break(largest,at);}
            count=(2*count).min(total);
        };
        let stands=standing(largest,at);
        if stands/at>most.excess{most=Drift{excess:stands/at,stands,span:reach as f64*dt};}
        quiet=if stands>0.{0}else{quiet+1};
        if reach>=end||(quiet>=QUIET_SPANS&&reach as f64*dt>=through){return most;}
        reach=(2*reach).min(end).max(reach+window);
    }
}

/// The population error a step accumulates over `horizon` (module documentation), over its tolerance:
/// from states spread over the sampled ones the jump-free flow in steps of dt
/// against steps of dt/2, the largest difference of the cluster's populations; their mean plus
/// CONFIDENCE standard errors. Heavy-tailed in a gray molasses, its atoms dark but for a few bright ones, so in layers as the orbits' (`orbits`): `subsample` states, then
/// doubling, evenly between those taken, at most `TASK` times as many, until the errors are within the step's share of the tolerance or the magnitude less the errors lies beyond it. (The D1 molasses's
/// step over 76 seeds at 10 us outputs: 0.24-0.77 us on 32 states alone, one seed's few bright states holding it at 0.24 us where the populations' true error stays within half the tolerance to 0.71 us
/// against 0.167 us; 0.56-0.77 us so, 90 % within 0.56-0.63 us either way.)
fn accumulated_excess(engine:&Engine,states:&[Trajectory],horizon:f64,subsample:usize,tolerance:f64)->f64{
    let dt=engine.settings.dt;let richardson=4./3.;let count=((horizon/dt).ceil() as usize).max(1);
    let flow=|t:&&Trajectory|->f64{
        let n=engine.model.dimension(t.cluster);let(one,two)=(engine.coast(t,dt,count),engine.coast(t,0.5*dt,2*count));
        let(a,b)=(one.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>(),two.psi[..n].iter().map(|z|z.norm_sqr()).sum::<f64>());
        richardson*(0..n).map(|i|(one.psi[i].norm_sqr()/a-two.psi[i].norm_sqr()/b).abs()).fold(0.,f64::max)
    };
    let mut stride=(states.len()/subsample.max(1)).max(1);let most=TASK as usize*(states.len()/stride).max(1);
    let mut errors:Vec<f64>=states.iter().step_by(stride).collect::<Vec<&Trajectory>>().par_iter().map(flow).collect();
    loop{
        let m=errors.len().max(1) as f64;let mean=errors.iter().sum::<f64>()/m;
        let spread=CONFIDENCE*(errors.iter().map(|e|(e-mean).powi(2)).sum::<f64>()/(m*(m-1.).max(1.))).sqrt();
        if spread<=BIAS_SHARE*tolerance||mean-spread>tolerance||stride==1||errors.len()>=most{return (mean+spread)/tolerance;}
        errors.extend(states.iter().skip(stride/2).step_by(stride).collect::<Vec<&Trajectory>>().par_iter().map(flow).collect::<Vec<f64>>());stride/=2;
    }
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
pub(crate) fn first_phase()->f64{(BIAS_VARIANCE*TEMPERATURE_TARGET).sqrt()}
/// A sampled state's parts of the work model (`interval_work`): its cluster, its advance without the
/// propagation and its jump (ns), its generator's norm and its dimension; in a cluster with explicit excited states or beats,
/// the state itself -- the exact step's segment is probed at each step (`exact_step`), where the settings have its cluster take it (`Engine::takes_exact`: a cluster's beats all left out at the model's reach leave it the plain step) --;
/// and its weight in the run's time.
struct StateWork{cluster:usize,advance:f64,jump:f64,norm:f64,n:usize,exact:Option<Trajectory>,weight:f64,kernel:[f64;2]}
/// A cluster's jump in the work model (`StateWork`).
fn jump_work(w:&crate::model::Work)->f64{JUMP_NS+PHOTON_MULTIPLY_ADD_NS*w.photons+ABSORPTION_MULTIPLY_ADD_NS*w.absorption}
/// The work per interval (`interval_work`) of a candidate's fourth order, its tables unbuilt: its states' advances and jumps from
/// the fourth order's layout (`Model::layout_work`), the rest the candidate's own. The candidate's own work where the layout fails.
fn fourth_work(request:&Request,c:&Candidate,states:&[Trajectory],begun:&[Trajectory],s:&Settings,interval:f64,waiting:f64)->f64{
    let mut parts=state_work(&c.model,states,begun);let engine=Engine::new(&c.model,s.clone());
    if let Ok(works)=Model::layout_work(request,c.model.ground.clone(),true){for x in parts.iter_mut().filter(|x|!engine.takes_exact(x.cluster)){let w=&works[x.cluster];x.advance=advance_work(w);x.jump=jump_work(w);}}
    interval_work(&c.model,&parts,s,interval,waiting)
}
/// The work model's states (`StateWork`) over the run's time, each in the model's own clusters: the pilot's states at the
/// outputs, and its initial ones (`begun`) at half their weight -- the trapezoid rule over the outputs, whose first
/// interval the atoms start in their initial clusters (the shallow probe's every atom dark at the outputs, in its
/// rungs' one-state clusters, after a start in the explicit ones: its rungs' work counted as the folded model's, their
/// runs 3.4 times it).
fn state_work(model:&Model,states:&[Trajectory],begun:&[Trajectory])->Vec<StateWork>{
    states.par_iter().map(|t|(t,1.)).chain(begun.par_iter().map(|t|(t,0.5))).map(|(t,weight)|{
        let w=model.work(t.cluster);let n=model.dimension(t.cluster);
        let norm=if model.mixed(t.cluster){0.}else{column_norm(&model.local(t.cluster,&model.point(t.r,t.kappa),t.time).hamiltonian,n)};
        StateWork{cluster:t.cluster,advance:advance_work(&w),jump:jump_work(&w),norm,n,exact:(!model.is_static(t.cluster)).then(||t.clone()),weight,kernel:model.kernel_work(t.cluster)}
    }).collect()
}
/// The exact step's work (`exact_step`, ns): per complex multiply-add of n × n matrices, per n³ of the spectral
/// path's eigendecomposition with its forms, and per complex exponential of a beat's phase (at the large
/// arguments a run's times give). Against the step's CPU time on one core of the Core Ultra 7 265H (the jump-free
/// step at each of a candidate's states along its trajectories, as examples/multirate.rs `profile` times it):
/// PGC3D's 10-state cluster (spectral at the run's beats share), PGCP_10's 12-state
/// and 5-state beating clusters and PGC3D's folded one take 1.2-3.3 times their counts on the loaded machine --
/// the counts' ratios, which alone choose, within about ±30 %.
pub(crate) const MATRIX_NS:f64=0.469;const SPECTRAL_N3:f64=60.;const CIS_NS:f64=25.;
/// The exact step's work at a state (ns, engine.rs `Engine::exact`), from the segment the step builds there
/// (`Engine::probe`): per step, every step's -- its advance (the components' gather, the flows and the kick),
/// the beats' bounds (`Segment::resolve`: per pair two commutators, n³ each, of either beat, where its boundary term
/// does not settle it; else its share of the components' and the entries' n² per part, about as many) and the segment's
/// build (the spectral path's eigendecomposition and forms, the Taylor path's propagator, n³ per term of its
/// series to double precision) --, and
/// the grid's: per node (amplitude-weighted on the spectral path) its propagation (a phase per state, a
/// propagator's product, or CF4's two combinations of the beats and their series, `Entries::node_work`) and the force's and the
/// norm's forms, n² each.
fn exact_step(engine:&Engine,t:&Trajectory,x:&StateWork)->[f64;2]{
    let(model,s)=(engine.model,&engine.settings);let p=engine.probe(t.cluster,&model.point(t.r,t.kappa),t.time,s.dt,&t.psi);
    let n=x.n as f64;let(n2,n3)=(n*n,n*n*n);let pairs=p.beats.len() as f64;
    let terms=series(1.,1.,f64::EPSILON).1 as f64;let nodes=p.weighted.unwrap_or(p.nodes) as f64;
    // Per path: the build's multiply-adds, a node's multiply-adds and phases.
    let(build,node,turns)=match p.path{
        "spectral"=>(SPECTRAL_N3*n3,4.*n2+n,0.),
        "taylor"=>(terms*n3,5.*n2,0.),
        _=>(0.,p.node[0],p.node[1]),
    };
    [x.advance+MATRIX_NS*(8.*pairs*n3+build),nodes*(TERM_MULTIPLY_ADD_NS*node+CIS_NS*turns)]
}
/// The exact states a work count probes at most (`steps_parts`): each probe builds its segment.
const PROBED:usize=256;
/// The phase of a candidate's beating grid (engine.rs `Settings::node_phase`): the largest whose errors meet their shares on its
/// states -- at most `PROBED` of them, evenly spread, each one jump-free step against a grid of half the spacing, 16/15
/// their difference, CF4's and Simpson's order. The populations' largest difference, its mean plus CONFIDENCE standard
/// errors, within the numerical share of the population target (`Settings::beats`: over the waiting time, `NUMERICAL_SHARE` ε), and the impulse's within `QUADRATURE`
/// of its size. And the same step from the states their jumps leave (`Engine::landed`): a bare state rings at the drives' detunings where the probes' own states have relaxed, so that the grid's force on it errs by far more, a kick of its own per jump, of
/// random sign: a heat of (ΔI/k)² of the photon's recoil, within `QUADRATURE`. (Off these alone a strongly driven rung of 11 explicit states read 0.08 of its share at phase 30.8, and its T_x stood 25 % high at any step: 16.7 of it, ΔI = 0.65 k, on the states a jump leaves; T within its error from phase 10.) From the coarsest grid down (`largest_met`): a read costs as its nodes, so that a search pays about what its last read costs. From the quadrature's bound upward its first read, far finer than the phase it came to, was
/// nearly all of its work (of a 16-state rung that lost by 147 times, 583 of its 606 CPU-s, against the run's 40): beyond the fourth order's range the grid aliases the force's beats.
/// And, on what it finds, what the grid lands over the run between the groups of ground states `landing` names with the run's span (`landed_excess`; a pilot's first read of its starts takes none), the same
/// search below that phase where it misses. The bound's own where no state of `at` takes a beating grid.
pub fn grid_phase(model:&Model,states:&[Trajectory],at:&Settings,landing:Option<(&[usize],f64)>)->f64{
    if !(0..model.clusters()).any(|k|model.beating(k)){return node_phase();}
    let engine=Engine::new(model,at.clone());let picked:Vec<&Trajectory>=states.iter().step_by((states.len()/PROBED).max(1)).collect();
    let coarsest=picked.par_iter().filter_map(|t|engine.coarsest_phase(t)).reduce(||0.,f64::max);if !(coarsest>0.){return node_phase();}
    let landed:Vec<Trajectory>=picked.par_iter().map(|t|engine.landed(t)).collect();
    let recoil=model.carriers.iter().map(|c|c.wave_number).fold(0.,f64::max).powi(2);
    let excess=|phase:f64|->f64{
        let errors:Vec<[f64;3]>=picked.par_iter().filter_map(|t|engine.grid_error(t,phase)).collect();
        let m=errors.len() as f64;let sum=|k:usize|errors.iter().map(|e|e[k]).sum::<f64>();let mean=sum(0)/m;
        let spread=(errors.iter().map(|e|(e[0]-mean).powi(2)).sum::<f64>()/(m*(m-1.).max(1.))).sqrt();
        let kicks:Vec<f64>=landed.par_iter().filter_map(|t|engine.grid_error(t,phase)).map(|e|(16./15.*e[1]).powi(2)).collect();
        let heat=kicks.iter().sum::<f64>()/(kicks.len().max(1) as f64*recoil.max(f64::MIN_POSITIVE));
        (16./15.*((mean+CONFIDENCE*spread)/at.beats).max(sum(1)/sum(2).max(f64::MIN_POSITIVE)/QUADRATURE)).max(heat/QUADRATURE)
    };
    let phase=largest_met(coarsest,excess(coarsest),&excess);
    // What it lands between the groups over the run.
    let Some((groups,span))=landing else{return phase};let count=groups.iter().max().map_or(0,|g|g+1);if count<2{return phase;}
    let moved=|phase:f64|landed_excess(&engine,&picked,phase,groups,count,span);
    let r=moved(phase);if r<=1.{return phase;}
    largest_met(phase,r,&moved)
}
/// The largest phase a read's excess meets, from one read at the coarsest phase in question: a miss goes down by the fourth root of its excess, at least to half its spacing -- its errors saturate on a grid that aliases --, and then between the
/// largest phase met and the smallest missed on the logarithms' line, until one's excess lies within a factor two of its share (`grid_phase`). Past the quadrature's bound, where its share may be out of the grid's reach, or past the first phase met, at most 8 reads.
fn largest_met(mut phase:f64,mut r:f64,read:&dyn Fn(f64)->f64)->f64{
    // The largest phase met and the smallest missed, each with its excess.
    let(mut met,mut missed)=((0_f64,0_f64),(f64::INFINITY,f64::INFINITY));let mut reads=0;
    loop{
        if r<=1.{met=(phase,r);if r>=0.5||!missed.0.is_finite(){break;}}else{missed=(phase,r);}
        if met.0>0.||phase<=2.*node_phase(){reads+=1;if reads>8{break;}}
        phase=if met.0>0.{met.0*(missed.0/met.0).powf(((0.75/met.1).ln()/(missed.1/met.1).ln()).clamp(0.1,0.9))}else{phase*r.recip().powf(0.25).min(0.5)};
        r=read(phase);
    }
    if met.0>0.{met.0}else{phase}
}
/// The reads of a step's landings per probe (`Engine::grid_landings`): their errors stop falling at about this many, where the probes' own spread is what is left (F = 1 over the PGC with a resonant probe's
/// run, on its rung with F' = 2 and F' = 1 explicit at the phase its shares take: 0.0072, 0.0055, 0.0049, 0.0046, 0.0046 at 2, 4, 8, 16, 32 reads).
const LANDING_READS:usize=8;
/// What a beating grid at `phase` moves between the groups of ground states over `span`, over the populations' numerical share. A jump-free step's populations cannot read it: within a cluster the state's
/// fall gives back in its jumps what it moves in the populations (the same PGC's chosen rung, 1.7e-4 of a population per step of a third of a microsecond at its phase, 5 % if it added up, its run's populations
/// within 0.005), and the states a far-detuned explicit state pumps into lie in another cluster. So the one-step master equation on each probe, on this grid against one of half its spacing:
/// per group the jumps' landings, the no-jump norm's fall −ln‖ψ‖² shared out as its reads' sources (`Landings`) per cluster -- a jump's own share is a ratio of the photons' weights at its time, and the mean of that
/// ratio over a state's beats stands above the ratio of their means --, per unit time, the group it starts in taking the rest, through the groups' response ∫ e^{Rs} ds (`response`, R the rates between them on the finer grid: it adds up until
/// they relax, at most over the run). Per group the mean over the probes -- their beats average out, as a run's do -- plus CONFIDENCE standard errors, 16/15 their difference. (Against pairs of runs on one seed, 100 us:
/// the rung with F' = 2 and F' = 1 explicit at phases 12.2, 9.9, 6.1: F = 1 read 0.048, 0.041, 0.001, run 0.027 ± 0.005, 0.023 ± 0.008, within 0.012; its rung with three states of F' = 2 explicit at 9.7, 6.3: 0.008, 0.0004, run
/// 0.005 ± 0.006, 0.001 ± 0.003; the chosen rung at its 2.9: 0.0002, run -0.002 ± 0.006.) Within a group the populations relax in about ten such steps (p(2,2) on a rung beside the chosen one, -0.04 at -4e-3 per step), and there the
/// share over the waiting time (`grid_phase`) holds them to a few of the numerical share.
fn landed_excess(engine:&Engine,picked:&[&Trajectory],phase:f64,groups:&[usize],count:usize,span:f64)->f64{
    let dt=engine.settings.dt;let clusters=engine.model.clusters();
    let reads:Vec<(Landings,Option<Landings>)>=picked.par_iter().enumerate().map(|(j,t)|{
        let draws:[f64;LANDING_READS]=std::array::from_fn(|k|((j*LANDING_READS+k+1) as f64*0.618033988749895).fract());
        engine.grid_landings(t,phase,groups,count,&draws)}).collect();
    // Per cluster and grid the norm's fall per unit of its photons, the finer grid's first.
    let(mut fall,mut photons)=([vec![0.;clusters],vec![0.;clusters]],[vec![0.;clusters],vec![0.;clusters]]);
    for(t,(x,y))in picked.iter().zip(&reads){for(k,z)in [x,y.as_ref().unwrap_or(x)].into_iter().enumerate(){fall[k][t.cluster]-=z.left.max(f64::MIN_POSITIVE).ln();photons[k][t.cluster]+=z.sources.iter().sum::<f64>();}}
    let scale=|k:usize,c:usize|if photons[k][c]>0.{fall[k][c]/photons[k][c]}else{0.};
    // The rates between the groups, on the finer grid: of the probes in each.
    let(mut rates,mut probes)=(vec![0.;count*count],vec![0_f64;count]);
    for(t,r)in picked.iter().zip(&reads){let x=&r.0;probes[x.from]+=1.;
        for h in (0..count).filter(|&h|h!=x.from){let w=scale(0,t.cluster)*x.sources[h]/dt;rates[h*count+x.from]+=w;rates[x.from*count+x.from]-=w;}}
    for g in 0..count{if probes[g]>0.{for h in 0..count{rates[h*count+g]/=probes[g];}}}
    let response=response(&rates,count,span);
    // Per probe the groups' change's error per unit time, through the response.
    let moved:Vec<Vec<f64>>=picked.iter().zip(&reads).map(|(t,(x,y))|{let mut d=vec![0.;count];
        if let Some(y)=y{let own=x.from;for h in (0..count).filter(|&h|h!=own){d[h]=(scale(1,t.cluster)*y.sources[h]-scale(0,t.cluster)*x.sources[h]+y.stay[h]-x.stay[h])/dt;d[own]-=d[h];}}
        (0..count).map(|h|(0..count).map(|g|response[h*count+g]*d[g]).sum()).collect()}).collect();
    let m=picked.len() as f64;
    16./15.*(0..count).map(|h|{let mean=moved.iter().map(|d|d[h]).sum::<f64>()/m;
        mean.abs()+CONFIDENCE*(moved.iter().map(|d|(d[h]-mean).powi(2)).sum::<f64>()/(m*(m-1.).max(1.))).sqrt()}).fold(0.,f64::max)/(NUMERICAL_SHARE*POPULATION_TARGET)
}
/// ∫₀^span e^{Rs} ds of rates R (m × m, row-major): the series over a span whose ‖R s‖ lies within one, then doubling it, S(2s) = S(s) + e^{Rs} S(s), e^{2Rs} = (e^{Rs})², e^{Rs} = 1 + R S(s).
fn response(rates:&[f64],m:usize,span:f64)->Vec<f64>{
    let product=|a:&[f64],b:&[f64]|->Vec<f64>{let mut out=vec![0.;m*m];for i in 0..m{for k in 0..m{for j in 0..m{out[i*m+j]+=a[i*m+k]*b[k*m+j];}}}out};
    let size=(0..m).map(|g|(0..m).map(|h|rates[h*m+g].abs()).sum::<f64>()).fold(0.,f64::max)*span;
    let doublings=if size>1.{size.log2().ceil() as i32}else{0};let h=span/2_f64.powi(doublings);
    let unit:Vec<f64>=(0..m*m).map(|k|if k/m==k%m{1.}else{0.}).collect();
    let mut term:Vec<f64>=unit.iter().map(|x|x*h).collect();let mut s=term.clone();
    for k in 2..24{term=product(rates,&term).iter().map(|x|x*h/k as f64).collect();for(a,b)in s.iter_mut().zip(&term){*a+=*b;}}
    let mut e:Vec<f64>=product(rates,&s).iter().zip(&unit).map(|(a,b)|a+b).collect();
    for _ in 0..doublings{let next=product(&e,&s);for(a,b)in s.iter_mut().zip(&next){*a+=*b;}e=product(&e,&e);}
    s
}
/// A step's parts at every state (ns): what every step pays -- the advance with, taking the plain step,
/// its two half steps' series, or the exact step's per step (`exact_step`) -- and what the multirate grid takes
/// over the step, whose nodes per unit time no step changes; and whether it takes the exact step. The exact states' from at most `PROBED` of them,
/// evenly spread, their mean for the rest.
fn steps_parts(model:&Model,states:&[StateWork],s:&Settings)->Vec<([f64;2],bool)>{
    let engine=Engine::new(model,s.clone());
    let exact:Vec<usize>=(0..states.len()).filter(|&i|engine.takes_exact(states[i].cluster)).collect();
    let probed:Vec<[f64;2]>=exact.par_iter().step_by((exact.len()/PROBED).max(1)).map(|&i|exact_step(&engine,states[i].exact.as_ref().unwrap(),&states[i])).collect();
    let mean:[f64;2]=std::array::from_fn(|k|probed.iter().map(|p|p[k]).sum::<f64>()/probed.len().max(1) as f64);
    states.iter().map(|x|if engine.takes_exact(x.cluster){(mean,true)}else{([x.advance+2.*half_step(x,s),0.],false)}).collect()
}
/// What a jump adds to an exact step (`exact_step`'s parts, ns): the step redone to its crossing and its rest build their segments, and the grid up to the crossing goes over again, half a step's on average.
fn exact_redo(p:&[f64;2])->f64{2.*p[0]+0.5*p[1]}
/// A half step's propagation at a state (ns): its series' sub-steps × terms at its generator's norm,
/// the step's dt and truncation.
fn half_step(x:&StateWork,s:&Settings)->f64{let(sub,terms)=series(x.norm,0.5*s.dt,s.truncation);(sub*terms) as f64*(TERM_NS+TERM_MULTIPLY_ADD_NS*(x.n*x.n) as f64)}
/// A step's work per interval (ns, `ADVANCE_NS` and the rest), counted on the sampled states -- the
/// same whatever the machine and its load: its steps (`steps_parts`), each the generator's and the force's
/// multiply-adds and pairs of the state's cluster and two half steps' series, or the exact step's; and per
/// jump, one every waiting time, the jump itself -- every class's photons, the picked class's absorbed light
/// and the rest -- and two more advances, the step redone to the jump and its rest (of an exact step, `exact_redo`). A readout per output is the same at every step and
/// left out.
fn interval_work(model:&Model,states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    let parts=steps_parts(model,states,s);let(mut advance,mut jump)=(0.,0.);
    for(x,(p,exact))in states.iter().zip(&parts){let a=p[0]+p[1];advance+=x.weight*a;jump+=x.weight*(x.jump+if *exact{exact_redo(p)}else{2.*a});}
    let count=weights(states);
    (interval/s.dt*advance+interval/waiting*jump)/count
}
/// A step's work per interval in the GPU's kernel (a warp's cycles, model.rs `Model::kernel_work`), as `interval_work`: the exact step's
/// at `EXACT_CYCLES` per ns of its count, PGCP_10's 1.6M cycles a step in the 12-state cluster against its 130 us.
fn kernel_interval(model:&Model,states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    const EXACT_CYCLES:f64=12.;
    let parts=steps_parts(model,states,s);let(mut advance,mut jump)=(0.,0.);
    for(x,(p,exact))in states.iter().zip(&parts){
        let a=if *exact{EXACT_CYCLES*(p[0]+p[1])}else{x.kernel[0]};
        advance+=x.weight*a;jump+=x.weight*(x.kernel[1]+if *exact{EXACT_CYCLES*exact_redo(p)}else{2.*a});
    }
    (interval/s.dt*advance+interval/waiting*jump)/weights(states)
}
/// The states' total weight in the run's time (`state_work`), at least one's.
fn weights(states:&[StateWork])->f64{states.iter().map(|x|x.weight).sum::<f64>().max(1.)}
/// The least work per interval any step leaves (ns, `interval_work`): the multirate grid's nodes, whose number
/// per unit time is the generator's and not the step's, and the jumps.
fn least_work(model:&Model,states:&[StateWork],s:&Settings,interval:f64,waiting:f64)->f64{
    let parts=steps_parts(model,states,s);let(mut grid,mut jump)=(0.,0.);
    for(x,(p,_))in states.iter().zip(&parts){grid+=x.weight*p[1];jump+=x.weight*x.jump;}
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
fn start(e:&mut Estimate,initial:&Initial){
    let t=initial.temperature();
    e.temperature=t;e.temperature_error=[0.;3];e.mean_velocity=[0.;3];
    e.temperature_total=t.iter().sum::<f64>()/3.;e.temperature_total_error=0.;
}

/// The solution of the estimates so far.
/// At t = 0 the populations are the initial state's, as the request gives them (the light acts
/// after), and so are T and the mean velocity (`start`). An output where no atom is held (the loss
/// one) has no held atom to describe: its populations, T, ⟨v⟩, n̄ and distributions are the last
/// output's that had one -- each row still a probability vector, the app's p (1 − L) zero there --,
/// outside the validated regime, and the flag says from when. T takes two atoms, a variance: one atom reads a variance of zero and an error of zero, which is no T. An output holding fewer has none of its
/// own: its T, ⟨v⟩ and n̄ are the last output's that read one, its T's error none (infinite here, as a single group's; null in the contract, and the app says unknown), outside the validated regime, and the flag says from when.
fn assemble(model:&Model,estimates:&[Estimate],outcome:&Outcome,listed:&[usize],times:&[f64],frequencies:[f64;3],trajectories:u64,through:u64,dt:f64,initial:&[Complex],mut ledger:Ledger,twin:bool)->Solution{
    let hm=model.hbar_over_mass;
    let velocity=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.1[a]*hm));
    let position=std::array::from_fn(|a|histograms(&outcome.samples,listed,&|s|s.0[a]));
    let first:Vec<f64>=(0..STATES).map(|i|initial[i*STATES+i].re).collect();
    let sizes:Vec<String>=model.ground.clusters.iter().map(|c|c.dimension().to_string()).collect();
    let explicit:usize=model.ground.clusters.iter().map(|c|c.excited.len()).sum();
    let internal=if explicit>0{format!("内态（子空间 {}，其中 {explicit} 个激发态显式、精确传播，其余激发态二阶消去）",sizes.join("+"))}else{format!("消去激发态后的基态内态（子空间 {}）",sizes.join("+"))};
    // Each output's source: itself where an atom is held (t = 0 always), else the last that held one.
    let held=|i:usize|i==0||estimates[i].loss<1.;let read=|i:usize|i==0||reads_temperature(&estimates[i]);
    let(source,measured)=(last_holding(estimates.len(),&held),last_holding(estimates.len(),&read));
    let statistic=statistics_of(estimates,frequencies);
    let mut flag=|note:String|ledger.flagged=Some(match ledger.flagged.take(){Some(r)=>format!("{note}；{r}"),None=>note});
    if let Some(gone)=(0..estimates.len()).find(|&i|!held(i)){flag(format!("{} 起已无原子被俘获（丢失概率 1）：此后的布居、温度与分布沿用 {} 的值，仅供参考",when(times[gone]),when(times[source[gone]])));}
    if let Some(few)=(0..estimates.len()).find(|&i|held(i)&&!read(i)){flag(format!("{} 起有时刻被俘获的原子不足两个：这些时刻的温度沿用此前的值，误差未知",when(times[few])));}
    let carried=|f:&dyn Fn(&Estimate)->[f64;3]|->Vec<[f64;3]>{measured.iter().map(|&j|f(&estimates[j])).collect()};
    Solution{
        populations:source.iter().map(|&j|if j==0{first.clone()}else{estimates[j].populations.clone()}).collect(),
        sample_errors:estimates.iter().zip(&statistic).enumerate().map(|(i,(e,s))|if i==0{0.}else{reported_error(s.population,e.trajectories)}).collect(),
        temperature_uk:carried(&|e|e.temperature),
        temperature_errors_uk:estimates.iter().enumerate().map(|(i,e)|if read(i){e.temperature_error}else{[f64::INFINITY;3]}).collect(),
        temperature_total_uk:measured.iter().map(|&j|estimates[j].temperature_total).collect(),
        temperature_total_errors_uk:estimates.iter().enumerate().map(|(i,e)|if read(i){e.temperature_total_error}else{f64::INFINITY}).collect(),
        mean_velocity_mps:carried(&|e|e.mean_velocity),
        trap_frequency_khz:frequencies.map(|w|w/std::f64::consts::TAU*1e3),
        mean_occupation:carried(&|e|e.occupation),
        loss_probability:{let mut worst=0.;estimates.iter().map(|e|{worst=e.loss.max(worst);worst}).collect()},
        velocity,position,
        classical:estimates.iter().enumerate().map(|(i,e)|read(i)&&classical(e,frequencies)).collect(),
        trajectories,needed:trajectories,time_step_us:dt,
        method:format!("半经典 3D：经典质点（位置、动量）携带{internal}，量子跳跃（自发辐射按偶极辐射方向图反冲，吸收按光束反冲并保持精确的平均力），机械能高于势阱深度且已离开势阱区即算损失，{trajectories} 条轨迹{}，步长 {dt:.3} μs；控制变量：谐振孪生（T）{}",if through<trajectories{format!("（走完全程的 {through} 条）")}else{String::new()},if twin{"与静止孪生（布居）"}else{""}),
        reason:None,
        approximations:ledger,
        device:Device::Cpu,device_note:None,
    }
}

#[cfg(test)]
mod tests{
    use super::*;
    const TARGETS:[f64;2]=[POPULATION_TARGET,TEMPERATURE_TARGET];
    /// The fold's charge with the memory: a bright transient's charge stays with atoms that go dark (b1/128's
    /// optical pumping, the validation's M1), and atoms that scatter forget it within their next photons.
    #[test]
    fn past_the_targets_reads_the_estimates_not_the_weight_s_round_off(){
        // A run's need brings its binding total to the target itself: its weight is 1 within round-off, on either side.
        let statistics:Vec<Statistic>=(0..5).map(|i|Statistic{population:0.2+0.01*i as f64,temperature:0.05,weight:1.}).collect();
        let within=vec![[0.02,0.03];5];
        let(_,weight)=predicted(&statistics,&within,[0.;2],32.);
        assert!((weight-1.).abs()<1e-12&&!past_targets(&statistics,&within),"{}",weight-1.);
        let mut past=within.clone();past[3][0]=1.2*POPULATION_TARGET;assert!(past_targets(&statistics,&past));
        let mut past=within.clone();past[2][1]=1.2*TEMPERATURE_TARGET;assert!(past_targets(&statistics,&past));
        // T's target is its regime's, and t = 0's estimates are the request's own.
        let mut wide=statistics.clone();wide[2].weight=0.5;let mut past=within.clone();past[2][1]=1.2*TEMPERATURE_TARGET;assert!(!past_targets(&wide,&past));
        let mut past=within.clone();past[0]=[1.;2];assert!(!past_targets(&statistics,&past));
    }
    /// A resonance's bound counts at the share of atoms that may have met it unseen (`with_open`): over a pilot's 32 atoms the 10 G tweezer's 7 % leaves the populations' target its room, a bound of 23 % takes all of it and a saturating
    /// drive's passes it far; continuous in the bound and in the atoms, and over a run's atoms hardly anything.
    #[test]
    fn a_resonance_s_bound_counts_at_the_share_of_atoms_that_may_have_met_it_unseen(){
        assert!((open_saturation(0.0721,32.)-0.016).abs()<1e-3&&open_saturation(0.0721,32.)<0.5*POPULATION_TARGET);
        assert!((open_saturation(0.225,32.)-POPULATION_TARGET).abs()<1e-12);
        assert!(open_saturation(1.73,8.)>10.*POPULATION_TARGET&&open_saturation(51.,32.)>100.*POPULATION_TARGET);
        assert!(open_saturation(0.0721,1000.)<1e-3&&open_saturation(0.,32.)==0.);
        let mut last=open_saturation(0.1,1.);for n in 2..=2000{let x=open_saturation(0.1,n as f64);assert!(x<last&&last-x<0.03);last=x;}
    }
    /// What stands against a limit (`standing`): nothing within its errors, short of the limit until it lies beyond it by them -- a chaotic span's size two standard errors above nothing, far beyond
    /// the limit, moves no step --, continuous through it and the magnitude far above.
    #[test]
    fn a_stand_passes_its_limit_only_beyond_it_by_its_errors(){
        let limit=0.1;
        assert_eq!(standing([0.1,0.2],limit),0.);assert!((standing([0.2958,0.2032],limit)-0.0926).abs()<1e-12);
        let far=standing([0.25,0.04],limit);assert!(far>0.24&&far<0.25,"{far}");
        let mut last=0.;for i in 0..4000{let m=(i as f64+0.5)*1e-4;let s=standing([m,0.1],limit);assert!(s>=last&&s-last<0.006&&(s<=limit)==(m<=0.2),"{m}: {s} after {last}");last=s;}
    }
    /// A read's errors at the confidence of its looks (`looks_confidence`): the single statistics' at few looks, continuous, and on a chaotic span -- the limit 0.6 of the standard error, as the 3D PGC's over 10 ms at 128 probes, 27 looks -- a size past the
    /// limit by them on about one run in 25, where at two standard errors it is about one in 5.
    #[test]
    fn a_read_s_errors_stand_at_the_confidence_of_its_looks(){
        assert_eq!(looks_confidence(3.),CONFIDENCE);assert_eq!(looks_confidence(6.),CONFIDENCE);assert!((looks_confidence(15.)-2.327).abs()<1e-3&&(looks_confidence(27.)-2.567).abs()<1e-3);
        let mut last=looks_confidence(1.);for i in 1..=4000{let x=looks_confidence(1.+0.01*i as f64);assert!(x>=last&&x-last<0.01);last=x;}
        let mut rng=crate::rng::Rng::new(9,crate::rng::Purpose::Initial,0,0);let(runs,looks,limit)=(4000,27,0.6);
        let stands=|z:f64,rng:&mut crate::rng::Rng|(0..runs).filter(|_|(0..looks).map(|_|crate::rng::normal_quantile(rng.interior()).abs()).fold(0.,f64::max)>limit+z).count() as f64/runs as f64;
        let(wide,narrow)=(stands(looks_confidence(27.),&mut rng),stands(CONFIDENCE,&mut rng));assert!(wide<0.06&&narrow>0.15&&narrow<0.3,"{wide} {narrow}");
    }
    /// An orbit's read counts at the share of atoms that keep it (`at_share`): all of it at one, in proportion below, and nothing of a flow no atom follows, however its pair parted.
    #[test]
    fn an_orbit_s_read_counts_at_the_share_of_atoms_that_keep_it(){
        let pairs=|share:f64|{let w:Vec<[[f64;3];2]>=(0..8).map(|i|at_share([2.;3],[1.+0.1*i as f64;3],share)).collect();paired_error(&w)[0]};
        let whole=pairs(1.);let half=pairs(0.5);assert!((half[0]-0.5*whole[0]).abs()<1e-12&&(half[1]-0.5*whole[1]).abs()<1e-12);
        assert_eq!(pairs(0.),[0.;2]);assert!(whole[0]>0.5,"{whole:?}");
        assert_eq!(at_share([3.,4.,5.],[1.,2.,3.],1.)[0],[3.,4.,5.]);assert_eq!(at_share([3.,4.,5.],[1.,2.,3.],0.)[0],[1.,2.,3.]);
    }
    /// T's statistic stands at the share of atoms it reads: its relative error at no loss, a quarter of it with a quarter of the atoms held, nothing where none is; and a few atoms held, their relative error √(2/x), ask no more trajectories than the batch that holds them, as none or one do: no cliff.
    #[test]
    fn t_s_statistic_stands_at_the_share_of_atoms_it_reads(){
        let mut e=Estimate::default();e.temperature=[10.;3];e.temperature_error=[1.;3];let at=|e:&Estimate|statistics(e,[1.;3]).temperature;
        let whole=at(&e);assert!((whole-CONFIDENCE*0.1).abs()<1e-15,"{whole}");
        e.loss=0.75;assert!((at(&e)-0.25*whole).abs()<1e-15);e.loss=1.;assert_eq!(at(&e),0.);
        let room=room(TEMPERATURE_TARGET,BIAS_SHARE*TEMPERATURE_TARGET);
        for held in 2..=6{let x=f64::from(held);let mut few=Estimate::default();few.trajectories=80.;few.loss=1.-x/80.;few.temperature=[10.;3];few.temperature_error=[10.*(2./x).sqrt();3];assert!(at(&few)<room,"{held}: {}",at(&few));}
    }
    /// A population's statistic is its error as the app shows it: among the held atoms at no loss, a quarter of that with a quarter of the atoms held, the loss's own where that is the larger; a single group's infinite errors stay so, lost or not.
    #[test]
    fn a_population_s_statistic_is_its_error_as_shown(){
        let mut e=Estimate::default();e.population_errors=vec![0.2,0.1];e.loss_error=0.01;let at=|e:&Estimate|statistics(e,[1.;3]).population;
        assert!((at(&e)-CONFIDENCE*0.2).abs()<1e-15);
        e.loss=0.75;assert!((at(&e)-CONFIDENCE*0.05).abs()<1e-15);
        e.loss=0.99;assert!((at(&e)-CONFIDENCE*0.01).abs()<1e-15);
        e.loss=1.;assert_eq!(at(&e),CONFIDENCE*0.01);
        e.population_errors=vec![f64::INFINITY;2];e.loss_error=f64::INFINITY;assert_eq!(at(&e),f64::INFINITY);e.loss=0.5;assert_eq!(at(&e),f64::INFINITY);
    }
    /// What trajectories alike leave open (`unseen`): 16 atoms all lost leave 20 % possible, a jackknife's zero says nothing of it. An output's populations and loss take at least the trajectories that bring it within their room, 76 at the target, in the pilot's prediction and in a run's
    /// ratio alike; beyond them it asks nothing, and a statistic above it stands. The reported error is at least its far end.
    #[test]
    fn trajectories_alike_leave_open_what_none_met(){
        assert!((unseen(16.)-0.2).abs()<1e-15);
        let n=unseen_need(POPULATION_TARGET);assert!((n-76.).abs()<1e-9&&(unseen(n)-POPULATION_TARGET).abs()<1e-15);
        assert!((population_ratio(0.,POPULATION_TARGET,n)-1.).abs()<1e-15&&population_ratio(0.,POPULATION_TARGET,2.*n)<1.);
        assert_eq!(population_ratio(0.1,POPULATION_TARGET,100.),2.);
        let none=[Statistic{population:0.,temperature:0.,weight:1.};3];
        let(need,weight)=predicted(&none,&[[0.;2];3],[0.;2],32.);assert!((need-n).abs()<1e-9&&weight==1.,"{need} {weight}");
        let read=[Statistic{population:0.,temperature:0.,weight:1.},Statistic{population:0.1,temperature:0.,weight:1.},Statistic{population:0.,temperature:0.,weight:1.}];
        let(reach,_)=predicted(&read,&[[0.;2];3],[0.;2],32.);assert!((reach-128.).abs()<1e-9,"{reach}");
        assert!((CONFIDENCE*reported_error(0.,76.)-POPULATION_TARGET).abs()<1e-15);
        assert_eq!(reported_error(CONFIDENCE*0.1,76.),0.1);
    }
    /// A read pays for what it could move of the run (`read_pays`): the resonant probe's PGC at 1.09 limits does not, the retro-reflected tweezer at 14.7 does, nothing within the limit does, and a stand already beyond asks for more.
    #[test]
    fn a_read_pays_for_what_it_could_move(){
        assert!(!read_pays(9984.,97663.,1.093,1.));assert!(read_pays(5856.,433350.,14.67,1.));
        assert!(!read_pays(1.,1e9,1.,1.)&&!read_pays(1.,1e9,0.5,1.));
        assert!(read_pays(9984.,97663.,1.8,1.)&&!read_pays(9984.,97663.,1.8,1.7));
    }
    /// T takes two atoms (`assemble`): one reads a variance of zero with an error of zero, a count of none or one reads no T at any N; an output takes the last that reads one, t = 0 always.
    #[test]
    fn a_temperature_takes_two_atoms(){
        let at=|n:f64,held:f64|{let mut e=Estimate::default();e.trajectories=n;e.loss=1.-held/n;reads_temperature(&e)};
        for n in [3.,80.,1456.,16384.]{assert!(!at(n,0.)&&!at(n,1.)&&at(n,2.),"{n}");}
        assert_eq!(last_holding(5,&|i|i==0||i==2),vec![0,0,2,2,2]);
    }
    /// T's probes fall as its weights do (`bound_probes`): evenly over a trajectory's states where they weigh alike, each state in time, and over their number where none weighs.
    #[test]
    fn t_s_probes_fall_where_its_weights_are(){
        let(flat,none)=(vec![1.;8],vec![0.;8]);
        let places=|w:&[f64],n:u32|(1..=n).map(|j|layer_place(j,w)).collect::<Vec<usize>>();
        assert_eq!(places(&flat,7),vec![4,2,6,1,5,3,7]);assert_eq!(places(&none,7),places(&flat,7));
        // T binding over the first two states of eight, hardly after: the first layers stay there.
        let early=[1.,1.,1e-3,1e-3,1e-3,1e-3,1e-3,1e-3];assert_eq!(places(&early,3),vec![1,0,1]);
        // Each state at its weight's share, up to its last.
        let mut hits=[0;3];for j in 1..=256{hits[layer_place(j,&[1.,2.,1.])]+=1;}assert_eq!(hits,[64,128,64]);
    }
    #[test]
    fn a_charge_s_steady_mean_averages_its_tail_and_keeps_a_real_step(){
        // A plateau of one state per output, dark but for one bright state: 20 standard errors of a normal law, no peak.
        let n=1001;let(mut sums,counts)=(vec![0.001;n],vec![1.;n]);sums[485]=0.06;
        let mean=sums[1..].iter().sum::<f64>()/1000.;let deviation=(sums[1..].iter().map(|x|x*x).sum::<f64>()/1000.-mean*mean).sqrt();
        let out=steady_means(&sums,&counts,deviation,0.06);
        assert!(out[1..].iter().all(|&x|x<0.01),"{} at the bright state, {:?}",out[485],&out[1..4]);
        // Ten times the charge over the first quarter stands on its states.
        let n=257;let mut sums=vec![1.;n];for s in &mut sums[..=64]{*s=10.;}let counts=vec![1.;n];
        let m=(64.*10.+192.)/256.;let deviation=((64.*100.+192.)/256.-m*m as f64).sqrt();
        let out=steady_means(&sums,&counts,deviation,10.);
        assert!((out[8]-10.).abs()<1.5&&(out[200]-1.).abs()<0.1,"{} {}",out[8],out[200]);
        // An output without states takes its nearest's.
        let mut counts=vec![1.;9];let mut sums=vec![1.;9];counts[4]=0.;sums[4]=0.;
        assert!((steady_means(&sums,&counts,0.,1.)[4]-1.).abs()<1e-12);
    }
    #[test]
    fn the_fold_s_charge_stays_with_dark_atoms_and_relaxes_with_the_photons(){
        let atom=|index:u64,jumps:&[u64]|->Vec<Trajectory>{jumps.iter().enumerate().map(|(i,&n)|Trajectory{cluster:0,psi:[Complex::ZERO;crate::model::MOST_MIXED],r:[0.;3],kappa:[0.;3],
            time:(i+1) as f64*5.,threshold:1.,alive:true,jumps:n,beats:0.,seed:0,index}).collect()};
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
    /// leaves of the target's square, then from 1 − 2/k of it the weighed cost's balance 2F²/(k − 2), at most the target; the
    /// run's room is the same rule at its work exponent, and at the step's share of it the statistics' room is exactly the rest,
    /// the one optimum; the predicted need and weight move continuously as a floor crosses the turn and the target.
    #[test]
    fn the_room_is_continuous_through_the_target(){
        assert!((BIAS_SHARE*BIAS_SHARE-BIAS_VARIANCE).abs()<1e-15);
        for target in TARGETS{
            let mut last=room(target,0.);assert_eq!(last,target);
            for i in 1..=80_000{let f=i as f64*1e-4*target;let r=room(target,f);
                assert!(r>0.&&r<=target&&(r-last).abs()<=2e-2*target,"{f}: {r} after {last}");last=r;
                if f*f<=(1.-2./BEYOND)*target*target{assert!((r*r-(target*target-f*f)).abs()<=1e-15*target*target);}}
            assert_eq!(room(target,8.*target),target);
            for i in 0..=500{let apriori=i as f64*0.01*target;let run=run_room(target,apriori);let bias=BIAS_SHARE*run;
                if run<target{assert!((room(target,floor(apriori,bias))-(1.-BIAS_VARIANCE).sqrt()*run).abs()<=1e-12*target,"{apriori}: {} against {}",room(target,floor(apriori,bias)),(1.-BIAS_VARIANCE).sqrt()*run);}
                if apriori*apriori<=(1.-BIAS_VARIANCE)*target*target{assert!((run*run-(target*target-apriori*apriori)).abs()<=1e-15*target*target);}}
        }
        // The shallow probe's T, 8.99 % a priori, just beyond the turn: the run's room 4.50 %, the step's 2.01 % of it, the statistics' 4.02 %.
        let(run,t)=(run_room(0.1,0.0899),BIAS_SHARE*run_room(0.1,0.0899));assert!((run-0.04495).abs()<1e-12&&(room(0.1,floor(0.0899,t))-0.0402).abs()<1e-4,"{run} {t}");
        // One output after t = 0: the need and the weight across the turn and the target.
        let need=|f:f64|predicted(&[Statistic{population:0.,temperature:0.,weight:0.},Statistic{population:0.01,temperature:0.,weight:0.}],&[[0.;2],[f,0.]],[0.;2],32.);
        let(n,w)=need(0.01);assert!((n-unseen_need(room(POPULATION_TARGET,0.01))).abs()<1e-9&&w==1.,"{n} {w}");
        let(mut last,mut weight)=need(0.);
        for i in 1..=2000{let(n,w)=need(i as f64*5e-5);assert!((n-last).abs()<=0.05*last.max(TASK as f64)&&(w-weight).abs()<=0.03*weight,"{i}: {n} {w} after {last} {weight}");(last,weight)=(n,w);}
        assert!(weight>1.&&need(0.2).1>need(0.1).1);
    }
    /// The standard errors' noise over the outputs (`steady`): a plateau of noisy errors comes to its common value, an output's real
    /// peak beyond that noise stays, more trajectories at the first outputs keep their smaller errors, and a step in the error is kept.
    #[test]
    fn the_errors_noise_is_smoothed_over_the_outputs_that_agree(){
        let mut rng=crate::rng::Rng::new(5,crate::rng::Purpose::Initial,0,0);let groups=vec![72.;201];let counts=vec![576.;201];
        let noisy:Vec<f64>=(0..201).map(|_|0.01*(1.+crate::rng::normal_quantile(rng.interior())*(0.5/71_f64).sqrt())).collect();
        let largest=|v:&[f64]|v[1..].iter().copied().fold(0.,f64::max);
        let s=steady(&noisy,&counts,&groups);assert!(largest(&noisy)>1.15*0.01&&largest(&s)<1.05*0.01,"{} {}",largest(&noisy),largest(&s));
        let mut peak=noisy.clone();peak[100]=0.02;let s=steady(&peak,&counts,&groups);assert!(s[100]>0.017,"{}",s[100]);
        let mut fewer=vec![0.01;201];let mut trajectories=vec![576.;201];for i in 21..201{fewer[i]=0.02;trajectories[i]=144.;}
        let s=steady(&fewer,&trajectories,&groups);assert!((s[5]-0.01).abs()<1e-3&&(s[100]-0.02).abs()<2e-3,"{} {}",s[5],s[100]);
        let mut step=vec![0.01;201];for x in &mut step[100..]{*x=0.02;}
        let s=steady(&step,&counts,&groups);assert!((s[50]-0.01).abs()<1e-3&&(s[150]-0.02).abs()<1e-3,"{} {}",s[50],s[150]);
    }
    /// T's target follows the method's own error (module documentation, # Regime): w = min(1, m²) on the lowest axis, continuous;
    /// T's room and tolerance at ε_T/w, the tolerance at most ε_T.
    #[test]
    fn t_s_target_follows_the_method_s_own_error(){
        let w=2.;let limit=CLASSICAL*w/KT_PER_UK;let at=|m:f64|{let mut e=Estimate::default();e.temperature=[1e9,m*limit,3.*limit];regime(&e,[w;3])};
        assert_eq!(at(1.),1.);assert_eq!(at(2.),1.);assert!((at(0.75)-0.5625).abs()<1e-12);assert!((at(0.5)-0.25).abs()<1e-12);assert_eq!(at(0.),0.);
        let mut last=at(0.);for i in 1..=2000{let x=at(i as f64*1e-3);assert!((x-last).abs()<=0.0021&&x>=last,"{i}: {x} after {last}");last=x;}
        let a=[[0.;2],[0.02,0.02]];let t=|w:f64|tolerances(&[w,w],&a)[1];
        assert!((t(1.)-BIAS_SHARE*run_room(TEMPERATURE_TARGET,0.02)).abs()<1e-15&&t(0.)==TEMPERATURE_TARGET&&t(0.25)==TEMPERATURE_TARGET);
        assert!(t(0.6)>t(1.)&&t(0.6)<TEMPERATURE_TARGET);assert!((tolerances(&[0.,1.],&a)[1]-t(1.)).abs()<1e-15||tolerances(&[0.,1.],&a)[1]==t(1.));
        // Its upper bound, for the step's tolerance: T within the limit by its errors reads the limit; no errors read T itself; a group alone, its errors unknown, reads the limit.
        let mut e=Estimate::default();e.temperature=[1e9,0.77*limit,3.*limit];assert!((regime(&e,[w;3])-0.5929).abs()<1e-12);assert!((ceiling(&e,[w;3])-0.5929).abs()<1e-12);
        e.temperature_error=[0.,0.1*limit,0.];assert!((ceiling(&e,[w;3])-0.9409).abs()<1e-12);assert!((regime(&e,[w;3])-0.5929).abs()<1e-12);
        e.temperature_error=[0.,0.2*limit,0.];assert_eq!(ceiling(&e,[w;3]),1.);
        e.temperature_error=[f64::INFINITY;3];assert_eq!(ceiling(&e,[w;3]),1.);
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
        assert_eq!(next_step(&Check{steps:33,population:8.86,loss:0.,motion:1.0104},true,most,2.),34);
        assert_eq!(next_step(&Check{steps:33,population:8.86,loss:0.,motion:1.0104},false,most,2.),99);
        assert_eq!(next_step(&Check{steps:99,population:0.016,loss:0.,motion:0.1075},true,most,2.),33);
        // The excess as dt³: the power over two checks lands on the step at once, where the square root creeps.
        let check=|s:usize|{let x=(100./s as f64).powi(3);Check{steps:s,population:x,loss:0.,motion:0.}};
        let(a,b)=(check(70),check(140));assert!((power(&a,&b)-3.).abs()<1e-12);assert_eq!(next_step(&b,false,most,power(&a,&b)),100);
        let power=(1.0104f64/0.1075).ln()/3f64.ln();
        for delta in [-0.02,-1e-3,0.,1e-3,0.02]{
            let check=|s:usize|{let x=s as f64;Check{steps:s,population:8.86*(33./x).powi(2),loss:0.,motion:0.1075*(99./x).powf(power)*(1.+delta)}};
            let measure=|s:usize|0.016*(99./s as f64).powi(2);
            let mut bound=vec![];search(43,most,&[],&check,None,false,&mut bound);
            let grown=*bound.iter().filter(|c|c.met()).min_by_key(|c|c.steps).unwrap();
            let mut measured=vec![];search(grown.steps,most,&bound,&check,Some(&measure as &dyn Fn(usize)->f64),false,&mut measured);
            // Handed over at the first step the bound alone misses, the same step, for one finer check less.
            let mut handed=vec![];search(43,most,&[],&check,None,true,&mut handed);assert!(handed.last().unwrap().decided_by_population()&&handed.last().unwrap().population>1.&&handed.len()<bound.len(),"{handed:?}");
            let mut over=vec![];search(handed.last().unwrap().steps,most,&handed,&check,Some(&measure as &dyn Fn(usize)->f64),false,&mut over);
            let taken=over.iter().filter(|c|c.met()).map(|c|c.steps).min().unwrap();assert!(taken==33||taken==34,"{over:?}");
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
        // 0.1, T's tolerance the step's share s of the run's room -- at that bias the statistics' room is exactly the rest;
        // T a reference, its tolerance the target itself.
        let(t,run)=(tolerances(&[1.;2],&[[0.;2],[0.,0.0905]]),run_room(TEMPERATURE_TARGET,0.0905));
        assert!((t[1]-BIAS_SHARE*run).abs()<1e-15&&(room(TEMPERATURE_TARGET,floor(0.0905,t[1]))-(1.-BIAS_VARIANCE).sqrt()*run).abs()<1e-12,"{t:?} {run}");
        assert_eq!(tolerances(&[0.;2],&[[0.;2],[0.,0.0905]])[1],TEMPERATURE_TARGET);
    }
    /// The first batch: its first groups through the run, each one beyond them as far as the reach wants it; the batches after it, each output's own need.
    #[test]
    fn a_batch_is_the_need_within_its_bounds(){
        let wave=first_ends(&[0.,800.,800.,100.,100.],72,32);assert_eq!(wave.len(),72);assert!(wave[..32].iter().all(|&e|e==4)&&wave[32..].iter().all(|&e|e==2),"{wave:?}");
        let batch=first_ends(&[0.,500.,200.,100.,100.],32,8);assert!(batch[..13].iter().all(|&e|e==4)&&batch[13..].iter().all(|&e|e<=2),"{batch:?}");assert_eq!((batch[20],batch[31]),(2,1));
        // Each output's own need: the reach is the largest need from an output on; the groups go as far as an output wants them.
        assert_eq!(reaches(&[0.,500.,100.,300.,50.]),vec![500.,500.,300.,300.,50.]);
        assert_eq!(next_groups(&[0,276,276,264,256],&[0,256,256,256,256]),vec![3,2,2]);
        assert_eq!(next_groups(&[0,1000,300,256],&[0,256,256,256]),(0..32).map(|g|if g<6{2}else{1}).collect::<Vec<usize>>());
        assert!(next_groups(&[0,256,100],&[0,256,256]).is_empty());
    }
}
