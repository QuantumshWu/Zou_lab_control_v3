//! The atoms' reach (SPEC §1.8) and provable extrema over it.
//!
//! # The reach
//! The sublevel set {(r, v) : V_prep(r; p) + v²/(2 ħ/m) ≤ V_min + E_r} of the prepared energy, E_r = L k_B T_r the level that
//! holds all but the share s of the trap's lowered isothermal at T_r (# The reach's level): at r the
//! speeds |v| ≤ √(2 (ħ/m)(V_min + E_r − V_prep(r))), at most √(2 (ħ/m) E_r) (`Reach::speed`), the positions where that is real. Its level
//! V_min + E_r is below the far value 0 where the trap holds the reach; there −V_prep ≥ −(V_min + E_r) > 0 on it, and the proven
//! envelope −V_prep ≤ B max_l A_l(r) (model.rs `Envelope`) puts it inside the union of the ellipsoids u_l ≤ B/(−V_min − E_r) − 1
//! of the trap light's focused beams, u_l = xᵀ M_l x, M_l = (2/w₀²)(1 − nnᵀ) + nnᵀ/z_R²: their lab-axis extents √(U (M_l⁻¹)_aa),
//! M_l⁻¹ = (w₀²/2)(1 − nnᵀ) + z_R² nnᵀ, bound `Reach::cell`.
//!
//! # The trap's region
//! An atom is held while it is in the trap's region or bound (SPEC §1.7: lost is outside the region and above the
//! potential's far value; field.rs, # The region). So a level at or above the far value -- an atom that has gained the
//! trap's depth and is still in its light -- is held within the region alone: there the reach is the sublevel set within
//! the region, and outside it the sublevel set of a level below the far value (`Reach::outside`; none at first, a bound
//! atom's where one is to be held, `Model::outside_to`). Two levels, not one: the sublevel sets of the levels just below
//! the far value reach out without bound (an atom at rest far from the focus is bound), and a family clipped to the
//! region above the far value alone would drop, as it grows past it, the bound atoms outside the region it held below
//! it. Both levels only grow, so a reach grown holds the one before it. The region lies within the ellipsoids u_l ≤
//! 1/`REGION_INTENSITY` − 1 of the trap light's beams (|a_l|² ≤ 1/(1 + u_l), model.rs `Envelope`): their extents bound
//! it as the level's do the sublevel set. An outside level at or above the far value -- a held atom that V_run binds and
//! V_prep does not -- leaves the reach unbounded: the whole space, its bounds the global ones.
//!
//! A temperature's level is below the depth (# The reach's level): no temperature's reach holds an atom that has gained
//! the trap's depth, or its top, and is still in the region. Such an atom is held there all the same, and a path of it
//! that the reach's proofs are to hold (solve.rs `Domain`) is held by the region's level raised to its energy, the
//! temperature kept (`Model::inside_to`): within the region the reach then holds every prepared energy up to it -- the
//! path's points among them, its largest energy no more --, and outside it what it held. The family stays nested and
//! only grows: the inside level is the larger of the two, the outside one and the temperature are kept, so the reach so
//! raised holds the one before it, and a later temperature's reach is raised again to the levels asked before (solve.rs
//! keeps them as it keeps the outside ones). The temperature is the atoms' own, its ensemble's statements (the kinetic
//! temperatures, the falsifier's scale) its own; only the set and its speeds, √(2 (ħ/m) E), grow with the path.
//!
//! # The reach's level
//! The reach is the set that holds all but the share s of the initial ensemble -- its bound atoms alone (ensemble.rs,
//! # The lowered isothermal: f ∝ e^{−E/k_BT} − 1 below the far level, nothing above it) --: its level is the (1 − s)
//! quantile of an atom's energy above the trap's bottom in that ensemble as it is drawn (each axis's harmonic-oscillator
//! Wigner factor on the classical draw), in the trap's harmonic form. Along a principal axis i of the prepared trap that
//! holds the atom (frequency ω_i > 0, model.rs `Trap`) the oscillator's two quadratic degrees of freedom hold k_B T E_i
//! classically, E_i a unit exponential, and the Wigner factor scales both by s_i² = x_i coth x_i, x_i = ħω_i/2k_BT (the
//! oscillator's own mean energy over k_B T: one where ħω_i ≪ k_BT, x_i -- the zero point's ½ħω_i -- where it is cold);
//! along an axis the trap does not hold there is the momentum's degree alone, ½ k_BT x² with x a standard normal, of
//! Γ(½). So of the thermal draw the energy over k_B T and the classical one are
//!
//! ```text
//! X = Y + Σ_held s_i² E_i,   C = Y + Σ_held E_i,   Y of Γ(f/2), f the axes not held, C of Γ(k), k = f/2 + the axes held.
//! ```
//!
//! The lowering weighs the draw by (e^{−C} − e^{−d})₊, d = D/k_BT, D = −V_min the trap's depth (the far value is nothing).
//! With one more unit exponential S, (e^{−C} − e^{−d})₊ = ∫₀^∞ e^{−C−S} [C + S < d] dS: the lowered ensemble is the thermal
//! one on the condition C + S < d, C + S of Γ(k + 1). Given C + S = τ the parts over τ are Dirichlet's, so X = τ W, W =
//! Σ_j a_j z_j (a_j the scales, the slack's none) independent of τ; below the least scale a₀ (one where an axis is free)
//! Σ a_j z_j ≤ w forces Σ z_j ≤ 1, so P(W ≤ w) = A w^k there, A = Π s_i^{−2}, the cone's measure. A level L ≤ a₀ d reads
//! W only there at every τ ≥ d: P(X > L, C + S ≥ d) = ∫_d^∞ γ_{k+1}(τ)(1 − A L^k τ^{−k}) dτ = Q(k + 1, d) − A L^k e^{−d}/Γ(k + 1),
//! and
//!
//! ```text
//! P(X > L | C + S < d) = 1 − [P(X ≤ L) − A L^k e^{−d}/Γ(k + 1)] / P(k + 1, d),   L ≤ a₀ d          (`Degrees::above`)
//! ```
//!
//! -- the thermal draw's distribution (`below`) and the incomplete gamma function's (P(k + 1, d) the lowered isothermal's
//! own number, ensemble.rs `kept`). Classical (every s_i = 1) it is ∫_L^d x^{k−1}(e^{−x} − e^{−d}) dx over its integral
//! from nothing, and nothing at the depth; at d → ∞ the thermal draw's tail, P(X > L) = Q(n/2, L) of the n = 6 − f
//! quadratic degrees, for a trap that holds along its three axes Q(3, L) = e^{−L}(1 + L + L²/2): L = 12.89 at s =
//! 2.44e-4. (ln(1/s) = 8.32, the level where the Boltzmann factor is s, is one oscillator's quantile; six degrees leave
//! 1.07 % of the atoms above it. And a level of 12.89 k_B T holds none of an ensemble whose zero point is above it: at ħω
//! = 14, 14, 61 k_BT, s² = 7.2, 7.2, 30.6, the quantile is 271 k_BT.) The thermal draw's distribution is the convolution
//! of its terms, one exponential after another from the smallest scale, P(Y + aE ≤ L) = (1/a)∫₀^L e^{−(L−u)/a} P(Y ≤ u) du,
//! each integral by Gauss–Legendre with its nodes doubled until two levels' difference stops falling, below the mean and
//! as its complement's above it (`below`) -- no difference of exponentials, so scales that are equal or nearly so (a round
//! trap's) are no case of their own, and every part positive, so a hot trap's few atoms below a level are read to their
//! own precision --, and the level by bisection
//! to the format's resolution between nothing and a₀ d (the tail falls monotonically there). Above a₀ d the Wigner widths
//! alone carry atoms, the zero point lifting them above the depth -- outside the validated regime (k_BT ≥ 4ħω_i, where
//! a_max/a₀ ≤ 1.005 and they are fewer than 1.6·10⁻⁴ ≤ s at any depth) --; where more than s are, the level is a_max d,
//! which every atom of the harmonic form is below (X ≤ a_max C < a_max d).
//!
//! The level is below the depth: a reach does not reach the trap's edge, where its light no longer shifts the others. (Taken
//! as the thermal draw's quantile, 12.89 k_BT_r crossed the user's 1.5 mK tweezer's depth at 116 µK; the reach then held
//! the whole region, to its edge, where the probe is 3.3 Γ from its line and its fold does not exist -- every excited
//! state it reaches was kept, a step 50 → 160 µs.) At 1.5 mK and 30 µK the level is 12.91 k_BT (387 µK), at 300 µK 4.93
//! k_BT (1480 µK). It rises with T_r to its top (`top`): at T_r → ∞ every s_i → 1 and the energy over D has the density
//! ∝ u^{k−1}(1 − u) on [0, 1], Beta(k, 2), whose share above q is 1 − (k + 1)q^k + k q^{k+1} = s at q = 0.9936 for three
//! axes held. No temperature's reach holds a level at its top or above: an atom there is at the trap's top or above its
//! depth, leaving or lost (`holding`). The ensemble's kinetic temperature along axis i is s_i² T (`Reach::kinetic_uk`):
//! its speeds' and the reach's falsifier's scale.
//!
//! # What the reach leaves out
//! The level's quantile is the harmonic trap's. What the real trap adds -- the tweezer's anharmonic flanks, the Wigner
//! factors' widths over them -- is read off the initial ensemble's own density, not off a formula for one trap's shape, and
//! by no draw of an atom (`Model::left_out`; ensemble.rs, # What a level leaves out): the share of the atoms beyond the
//! set, the momenta above the kinetic energy A(r) = V_min + E_r − V_prep(r) the set holds at each place integrated over the
//! places, booked on what is shown; and the share of the kinetic energy they hold, on T -- each with its quadrature's own
//! difference between its last two levels. A reach grown with the atoms' temperature leaves out less of the same ensemble.
//!
//! # Provable extrema (`Model::bounds_over`)
//! A quantity's bound over the reach is the largest of its upper bounds over cells covering the reach's box, a cell taken where
//! it may meet the reach (a lower bound of V_prep over it at most the level). A quantity is one function of a region (`Region`):
//! a cell's beams -- each one's value and gradient at the centre and the bounds of how far it moves over the cell (`Span`),
//! |a_l| ≤ 1 -- or its centre point's, its radius zero. Monotone in the beams' bounds, on the cell it bounds the quantity over
//! the cell and on the point it is the quantity's bound there (a drive's coherent peak: every beam at its amplitude there, in
//! phase); the first tends to the second as the cell shrinks. The cell of a quantity's largest bound is halved (# A cell's
//! sides) until
//! the bound answers the question asked -- a threshold decided: below it everywhere (the largest remaining bound is the answer), or
//! a value at or above the question's lower edge reached at a centre in the reach (attained); or an extent known: within a
//! resolution of a value attained at a centre in the reach -- or until it is within the rounding of its centre's value. No
//! point is sampled as a bound. (A threshold near the quantity's true largest value is the expensive question: every cell
//! along the place of that value has to shrink to their difference, their number growing as its inverse cube. A question
//! whose answer may be either side of a band -- a gate, whose fold must be proven below its threshold and whose unit may
//! always be kept, units.rs `KEEP` -- asks the band, and its cells shrink to the band's width at most. A range is asked as
//! an extent, never by thresholds closing in on it.)
//!
//! All the quantities of a model share one tree of cells, and each quantity halves only the cells it still needs -- one
//! decided on the coarse cells costs nothing further. A cell has one pair of halves a side (# A cell's sides), each made when
//! a quantity first asks for it. The tree holds of a cell what the reach says of it (`Node`), a few numbers: a cell's regions
//! -- its beams, beam pairs and the parts the units' bounds have in common (the excited states' Gershgorin radii, the
//! carriers' couplings, resolvents and dressings, the clusters' force bounds, each ground state's drives' gates, the
//! levels' slope and the atoms' acceleration) -- are made where a round reads them, once for all the quantities that ask
//! there, and let go with the round's cell (# Rounds): functions of the cell alone, they are the same whenever made. (Kept in
//! the tree they were 28 kB a cell: the lin-perp-lin scene's 464 215 cells over a reach of 120 µK held 13 GB, and letting
//! them go took 39 s of the proof's 55.) What the bounds read of the model itself -- its forms' entries that are not zero,
//! with their moduli (`Forms`) -- is made once for the model.
//!
//! # Own forms
//! What a proof needs of a bound is that it tends to the quantity's largest value over the cell as fast as the quantity
//! itself varies there: its excess at most a few times the quantity's own variation, Σ_a h_a |∂_a f| and its curvature's
//! term, at every scale (units.rs `a_gate_s_bound_over_a_cell_is_within_its_own_variation_of_its_largest_value`). A bound
//! assembled from the separate ranges of parts that share a term is not that: it is as wide as the parts move, and where
//! the shared term cancels in the quantity it is wide where the quantity is flat, and the proof halves cells far below the
//! light's length before it answers. (The user's experiment, S3: the two-photon distance between F=1 m=0 and m=−1, their
//! 13.2 rad/µs Zeeman splitting, was read as each level's own slack -- the tweezer's scalar light shift, 236 rad/µs a
//! micrometre and the same on both, so nothing of the difference -- and the dressings' moduli: 0.38 rad/µs from nothing at
//! the point, nothing on every cell wider than 0.001 µm; the self-energy read infinite there, a repump drive's share 0.64
//! where it is 0.07, and the gates' proofs over the reach ran past 4000 halvings a quantity and out of memory.) So every
//! bound the gates read is the quantity's own form (own.rs `Own`): its value and gradient at the centre and a remainder of
//! the second order, beside its range -- each part made of its own sum (a level difference of its weights P_xx − P_yy,
//! eliminate.rs `Gap`; a sum of beam pairs of its own gradient and Hessian, `Combination`, `steepness`; a norm of its
//! vector's, `Frame`), and every composite of them in own-form arithmetic, so that two parts that move together
//! cancel at the first order where the quantity does. The speeds' bound is one too (`Region::pace`): a level's slope is
//! largest on the trap's flank and the speed at its bottom, and the product of the two maxima was neither's.
//!
//! # A cell's sides
//! The light does not change alike along a cell's three sides. A beam's amplitude is a = e^{ik·r} b: its plane wave turns by
//! |k·d| over a displacement d, at most Σ_a |k_a| h_a over a cell of half-widths h -- nothing along a side across the beam --,
//! and its envelope b changes over the waist across the beam and over the Rayleigh range along it (`Bends`). A fringe, the
//! coefficient a_j* a_i of two beams, turns only along k_i − k_j. So the bounds over a cell are made side by side, not from
//! the cell's radius times one Lipschitz constant: a pair's coefficient (`PairSpan`) and a sum of beams (`Frame`)
//! each from its value and gradient at the centre to second order, Σ_a h_a |∂_a ·| and half the bound of its second
//! derivative along the cell's displacements, at most the first order's, at most the phase-blind one -- at a crest, where the
//! largest values are, a first-order bound has no slope to read and gives the cell's size where the truth is its square.
//! A beam's modulus has no phase to follow: |b| over a cell is read from the beam's own shape (`Span::moduli`), and so is
//! the pair of a beam with itself, its intensity |b|² -- the far light's shifts by one beam, a trap's potential. (Bounded as
//! a pair of two beams, through the envelope's derivatives -- bounds over all space, the focus's, and of a complex
//! amplitude whose wavefront still curves where its modulus is flat --, a tweezer's intensity 0.16 of its peak one and a
//! half Rayleigh ranges from the focus, within 0.04 of that over a cell of 42 µm by 3 µm, was taken to move by 1.16: no
//! level's distance was known on such a cell, and over a reach grown along the beam the gates' proofs cut 214 000 cells.)
//! And a proof halves its cell along the side that holds its own quantity's bound up the most (`side`): the one along which
//! alone -- the cell's two other half-widths taken to nothing -- the bound is the highest; of sides that hold it up alike (a
//! bound still phase-blind along each, a quantity that does not change over the cell), the longest. Nothing is assumed of
//! how a quantity is made: it is asked. (Asked the other way, which side's removal lowers the bound the most, a bound held
//! at its phase-blind value by two sides at once answers neither and sends every halving to the third.) The cells so fit
//! what each proof resolves: across the fringes of a standing wave they are thin slabs, as wide along the fringes as the
//! envelopes let them be. (Cubes of the fringes' scale would fill a trap's whole volume to settle a bound that changes along
//! one line, and a bound near its threshold needs them small: their number grows as the inverse cube of the distance to it.)
//!
//! # Rounds (`Model::proven_over`)
//! A quantity's proof is a sequence -- its next cell is the one of its largest bound among those its earlier halvings left --,
//! but what a cell holds is a function of the cell alone, and so is a quantity's value on it. The proofs therefore go on in
//! rounds. In a round every proof still open goes on, in its own order, as far as the halves already made for it reach, and
//! then names the cell its order has come to. Where the quantities may be asked from any thread (`AtOnce`:
//! `Model::bounds_over_in_parallel`) it names, ahead of its order, every half the round before made for it whose bound is not
//! settled: whatever the order, each of these has to be halved before the proof can end with its bound settled everywhere, so
//! the cells are halved level by level and the rounds are as many as the cells are deep. The cells named are halved once each,
//! for all the quantities that named them, one after another or on all the cores at once, as the proofs' own steps are. Each
//! proof takes its cells in the order it would take them alone, through the same queue, so its bound is the same number
//! whatever was halved ahead of it and on however many cores. Only a proof that ends otherwise has had cells halved in vain.
//! One whose bound is attained at a centre: the cells not settled above the level at which a centre attains it, for the halves
//! made hold that centre's value from then on (`Proof::sighted`), every cell is settled against it, and the proof asks ahead no
//! more. One that ends at a cell whose bound is within the rounding of its centre's value: the cells it had asked for by then.
//! A unit's statements (its saturation and its sweep) are proven side by side, and a unit one of whose statements is
//! proven at the threshold or above is kept whatever the others read: their proofs end with it, each at the largest bound
//! of its cells left (`Model::statements_over`); which units are kept is the same as theirs proven to the end. (Proven to the
//! end alone, the sweep of a coupling of two Zeeman sublevels of a heavy atom at rest -- B 0, the sublevels crossing where
//! the tweezer's tensor shift does, its saturation infinite there -- halved the cells along the crossing towards a speed's
//! square root that no centre met, and the proof over S3's reach at 1 µK did not end in five minutes.) Every other unit's
//! proofs go to the end though a unit fails: the units failing together are the next model's, the closure's path. (Ended
//! with the round in which a first unit failed, the units failing later were left folded on the next model, and their
//! shares failed others there: the tweezer ×4 variant of S3 kept 114 units where 115 fail together, then 26 more where 6
//! do, 142 at 30 µK where the proofs to the end keep 121 there -- its 1 ms ran past five minutes, against 93 s.)
//!
//! # The kick's curvature (`Model::kick_curvature`)
//! A step is the reference's flow, a kick, the flow again (step.rs); the kick is −τ∇(⟨x|Ĥ_h|x⟩ − V_ref) at the step's
//! middle point r_h, of the state x held in its cluster's Doppler frames there (step.rs, # The kick: the shear). About a
//! minimum the step is the linear map R(ωτ/2) K R(ωτ/2) -- along an axis of the reference's rate ω the kick takes
//! τ w y from κ, w = ∂∂(⟨x|Ĥ_h|x⟩ − V_ref) --, whose trace is 2 cos ωτ − (ħ/m) w τ² sin(ωτ)/(ωτ); the motion is bounded
//! where the trace lies within ±2 (solve.rs `admissible`; measured on the crate's step, step/tests.rs). The map has the
//! eigenvalue −1 where det(D − τW) = 0, D the reference's 2ω_i cot(ω_iτ/2) on its axes and W the kick's Hessian, of any
//! axes: none while (ħ/m) λ_max(W) τ² < 4 (ω_maxτ/2) cot(ω_maxτ/2), which is that trace above −2 at the reference's
//! largest rate and the kick's largest curvature. (The eigenvalue +1 asks W ≤ −D′, D′ = 2ω_i tan(ω_iτ/2) ≥ ω_i²τ: a
//! potential of the atom's own that does not hold it -- the motion's instability, not the step's.) So what the step
//! asks of the model is one number, an upper bound of (ħ/m) λ_max ∂∂(Ĥ_h − V_ref) over the directions and the states
//! (rad²/µs²). What the model bounds of it:
//! - the trap light's own form on a state, less the reference's: the two wells' curvatures where the trap is, at its
//!   centre, as the trap's rates themselves are taken -- exact, with its sign (potential.rs `Model::stiffer`: a softer
//!   well asks nothing);
//! - the fringes of the lights folded with the bare levels, and of every other constant term between two ground
//!   states or two excited states: a matrix times plane waves, each entry's modulus times |q|² of each of its plane
//!   waves' residual wave vectors in the frames, by Gershgorin's rows (assemble.rs `Plan::fringes`), the beams' moduli
//!   at most their largest over the reach's box (`Span::most`). Exact for one pair of beams on one state (a
//!   lattice's wells); blind to the phases between several pairs, which it adds.
//! What it does not bound, and why (measured on the first version, which held both: edge.status.md):
//! - the shifts by the drives folded at the atom. Their bound from what the regions hold -- per pair of beams
//!   |q|² ‖X^f_j[:, a]‖ ‖X^f_i[:, b]‖ over the drives' least distance -- takes the response by its modulus, where the
//!   shift is its Hermitian part (none on resonance), and adds 36 pairs of six beams on five states blind to their
//!   phases: 9 rad²/µs² at the trap's centre of six beams' cooling (135 over the reach's box, whose corners the trap's
//!   light shift carries through the repumper's line) -- a first rung of 0.09 µs where the check lets 0.7 µs stand;
//! - the couplings of the states a cluster holds explicitly. A state held as a superposition of both is kicked by
//!   |q|² |c|; the states an atom is in are not that: its excited amplitude is the drive's answer, and the kick is
//!   the curvature of the level it follows (measured, step/tests.rs), ∂∂E = ⟨∂∂Ĥ_h⟩ + 2 Σ |⟨n|∂Ĥ_h|E⟩|²/(E − E_n) -- of
//!   the fold's size, |q|² |c|²/Δ, for a far drive: the held state's bound is Δ/|c| times that (6.5 rad²/µs² on the
//!   D1 gray molasses' forty explicit drives: 0.39 µs where the check lets 0.73 µs stand);
//! - the envelopes of the lights other than the trap's, the response's own change across the trap, the first-order
//!   Doppler partner: small against |q|², and of either sign.
//! A step whose edge these set is the check's to find (solve.rs, # The step), as every error of a step below its
//! edge is: where a companion lies in such a band its trajectories differ from the run's without bound, and the check
//! halves.

use super::eliminate::{Couplings, Diagonal, Gap, Shift};
use super::own::{Form, Own};
use super::far::EXCITED;
use super::{Component, Envelope, Level, Lights, Model, Reach, Trap};
use crate::field::{Amplitude, Geometry};
use crate::linalg::{dot3, norm3, Mat, C64};
use crate::request::{GROUND, STATES};
use crate::units::{rounding, KT_PER_UK, REGION_INTENSITY};
use rayon::prelude::*;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::{Arc, Mutex, OnceLock};

/// The bounds over all space of a beam's envelope's derivatives, by direction: across the beam and along it (a = e^{ik·r} b the
/// amplitude with its plane-wave phase taken out; all zero for a plane wave). A focused beam's envelope changes over its waist
/// across and over its Rayleigh range along: one number for both would hold a cell to the waist's scale along the beam too.
///
/// For a paraxial Gaussian beam ∇b = b g_b, g_b = −2t/(w₀²q) + (i/z_R)(ρ²/(w₀²q²) − 1/q) n (field.rs, less ik n): its first term
/// across (t ⊥ n), its second along. With |b| = e^{−x}/√(1 + s), x = ρ²/(w₀²(1 + s)), s = ζ²/z_R²:
/// - `slope`, sup |∇b| across and along: |b| 2ρ/(w₀²|q|) = 2√x e^{−x}/(w₀ √(1 + s)) ≤ √(2/e)/w₀ (√x e^{−x} ≤ 1/√(2e)), and
///   |b| |ρ²/(w₀²q²) − 1/q|/z_R ≤ (x e^{−x} + e^{−x})/z_R ≤ (1 + 1/e)/z_R.
/// - `curvature`, sup of the blocks of ∇∇b = b(g_b g_bᵀ + ∇g), ∇g = −2(1 − nnᵀ)/(w₀²q) + 2i(ntᵀ + tnᵀ)/(z_R w₀² q²)
///   − (1/q² − 2ρ²/(w₀²q³)) nnᵀ/z_R²: with √|b| |g_b| ≤ 2e^{−1/2}/w₀ across and 2e^{−1/2}/z_R along (√x e^{−x/2} ≤ e^{−1/2},
///   (x + 1)e^{−x/2} ≤ 2e^{−1/2}), across-across (2e^{−1/2}/w₀)² + 2/w₀²; across-along -- the vector h of n hᵀ + h nᵀ --
///   (2e^{−1/2}/w₀)(2e^{−1/2}/z_R) + 2/(√(2e) z_R w₀) (|b| 2ρ/(z_R w₀²|q|²) ≤ 2√x e^{−x}/(z_R w₀)); along-along
///   (2e^{−1/2}/z_R)² + 2e^{−1/2}/z_R² ((1 + 2x)e^{−x} ≤ 2e^{−1/2}).
/// - `third`, sup of the blocks of ∇∇∇b, ∂_i∂_j∂_k b = b(g_i g_j g_k + g_i ∂_j g_k + g_j ∂_i g_k + g_k ∂_i g_j + ∂_i∂_j g_k),
///   with p = |q| ≥ 1, |b| ≤ e^{−x}, |g_t| = 2√x/w₀, |g_n| ≤ (x + 1)/z_R, |∂_t g_t| ≤ 2/w₀², |∂_n g_t|, |∂_t g_n| ≤ 2√x/(w₀ z_R),
///   |∂_n g_n| ≤ (2x + 1)/z_R², ∂_t∂_t g_t = 0, |∂_t∂_n g_t|, |∂_t∂_t g_n| ≤ 2/(w₀² z_R), |∂_n∂_n g_t|, |∂_t∂_n g_n| ≤
///   4√x/(w₀ z_R²), |∂_n∂_n g_n| ≤ (6x + 2)/z_R³ (each a derivative of g along t or ζ, q' = i/z_R): across³
///   e^{−x}(8x^{3/2} + 12x^{1/2})/w₀³, across² along e^{−x}(4x² + 14x + 4)/(w₀² z_R), across along² e^{−x}x^{1/2}(2x² + 12x +
///   12)/(w₀ z_R²), along³ e^{−x}(x³ + 9x² + 18x + 6)/z_R³, each term at its own largest, e^{−x}x^a ≤ (a/e)^a: 8.43/w₀³,
///   11.32/(w₀² z_R), 11.69/(w₀ z_R²), 18.84/z_R³ -- what a gradient's Taylor remainder takes (`rest`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bends {
    /// sup |∇b| across the beam and along it (1/µm).
    pub(crate) slope: [f64; 2],
    /// sup of ∇∇b's blocks: across-across, across-along, along-along (1/µm²).
    pub(crate) curvature: [f64; 3],
    /// sup of ∇∇∇b's blocks: across³, across² along, across along², along³ (1/µm³).
    pub(crate) third: [f64; 4],
}

impl Bends {
    pub(crate) fn of(beam: &Geometry) -> Bends {
        match (beam.waist, beam.rayleigh()) {
            (Some(w), Some(z)) => {
                let (h, e) = ((-0.5_f64).exp(), std::f64::consts::E);
                // e^{−x} x^a at its largest, (a/e)^a.
                let top = |a: f64| if a == 0.0 { 1.0 } else { (a / e).powf(a) };
                Bends {
                    slope: [(2.0 / e).sqrt() / w, (1.0 + 1.0 / e) / z],
                    curvature: [(2.0 * h / w).powi(2) + 2.0 / (w * w), (2.0 * h / w) * (2.0 * h / z) + 2.0 / ((2.0 * e).sqrt() * z * w), (2.0 * h / z).powi(2) + 2.0 * h / (z * z)],
                    third: [
                        (8.0 * top(1.5) + 12.0 * top(0.5)) / w.powi(3),
                        (4.0 * top(2.0) + 14.0 * top(1.0) + 4.0) / (w * w * z),
                        (2.0 * top(2.5) + 12.0 * top(1.5) + 12.0 * top(0.5)) / (w * z * z),
                        (top(3.0) + 9.0 * top(2.0) + 18.0 * top(1.0) + 6.0) / z.powi(3),
                    ],
                }
            }
            _ => Bends { slope: [0.0; 2], curvature: [0.0; 3], third: [0.0; 4] },
        }
    }
    /// L_b = sup |∇b|, whatever the direction.
    pub(crate) fn lipschitz(&self) -> f64 {
        self.slope[0] + self.slope[1]
    }
}

/// A box of positions: its centre and half-widths (µm); an infinite half-width is the whole axis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Cell {
    pub(crate) centre: [f64; 3],
    pub(crate) half: [f64; 3],
}

impl Reach {
    /// The reach's box as a cell (`Reach::cell`): where the proofs' subdivisions start, and the one cell of a bound taken
    /// over the whole reach at once.
    pub(crate) fn whole(&self) -> Cell {
        Cell { centre: self.cell.map(|[lo, hi]| if (hi - lo).is_finite() { 0.5 * (lo + hi) } else { 0.0 }), half: self.cell.map(|[lo, hi]| 0.5 * (hi - lo)) }
    }
}

impl Cell {
    /// Its half-diagonal ρ: every point of it within ρ of its centre (infinite for an unbounded cell).
    pub(crate) fn radius(&self) -> f64 {
        self.half.iter().map(|h| h * h).sum::<f64>().sqrt()
    }
    pub(crate) fn bounded(&self) -> bool {
        self.radius().is_finite()
    }
    /// Its halves along the side `a`.
    fn halves(&self, a: usize) -> [Cell; 2] {
        let mut half = self.half;
        half[a] *= 0.5;
        [-1.0, 1.0].map(|s| {
            let mut centre = self.centre;
            centre[a] += s * half[a];
            Cell { centre, half }
        })
    }
}

/// A beam over a cell, its amplitude a = e^{ik·r} b written by its envelope b (module documentation of units.rs): a and ∇a at the
/// centre, its wave vector, the cell's half-widths h and its radius ρ (infinite: the whole space, its global bounds), and the
/// bounds over the cell of what its envelope does, d and d′ any two displacements from the centre within the cell (module
/// documentation, # A cell's sides):
/// - `most` and `least`, of |a| = |b|: |b(c)| ± `swing`, within the modulus's own range over the cell (`Span::moduli`);
/// - `slope`, of |∇b|: |∇b(c)| + `twist`, at most L_b;
/// - `swing`, of |∇b·d|, so of |b − b(c)| too: Σ_a h_a |∂_a b(c)| + `bend`, at most the global slopes across the beam and
///   along it times the cell's extents across and along (`Bends`);
/// - `bend`, of |dᵀ ∇∇b d′|, and `twist`, of |∇∇b d|: the curvature's blocks times the cell's extents across and along.
/// A cell's extent across a beam of direction n is Σ_a h_a √(1 − n_a²) and along it Σ_a h_a |n_a|, neither more than ρ.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Span {
    pub(crate) centre: C64,
    pub(crate) gradient: [C64; 3],
    pub(crate) hessian: [[C64; 3]; 3],
    pub(crate) wave: [f64; 3],
    pub(crate) half: [f64; 3],
    pub(crate) rho: f64,
    pub(crate) most: f64,
    pub(crate) least: f64,
    pub(crate) slope: f64,
    pub(crate) swing: f64,
    pub(crate) bend: f64,
    pub(crate) twist: f64,
    pub(crate) third: f64,
}

impl Span {
    /// The least and the largest of a beam's modulus |b| over the points within `along` of the centre `c` along the beam and
    /// within `across` of it across. A plane wave's is 1. A Gaussian beam's is e^{−ρ²/(w₀²u)}/√u, u = 1 + ζ²/z_R² (field.rs):
    /// it falls with the distance ρ from the axis, and along the beam it rises until u = 2ρ²/w₀² and falls from there
    /// (∂_u ln|b| = ρ²/(w₀²u²) − 1/(2u)) -- so the largest is at the least ρ and the u of the range nearest to 2ρ²/w₀², the
    /// least at the largest ρ and an end of u's range.
    fn moduli(beam: &Geometry, c: [f64; 3], along: f64, across: f64) -> [f64; 2] {
        let (Some(w), Some(z)) = (beam.waist, beam.rayleigh()) else { return [1.0; 2] };
        let zeta = dot3(c, beam.direction);
        let axis = norm3(std::array::from_fn(|i| c[i] - zeta * beam.direction[i]));
        let modulus = |rho: f64, u: f64| (-(rho / w).powi(2) / u).exp() / u.sqrt();
        let u = |zeta: f64| 1.0 + (zeta / z).powi(2);
        let (near, far) = (u((zeta.abs() - along).max(0.0)), u(zeta.abs() + along));
        let (inner, outer) = ((axis - across).max(0.0), axis + across);
        [modulus(outer, near).min(modulus(outer, far)), modulus(inner, (2.0 * (inner / w).powi(2)).clamp(near, far))]
    }

    /// A beam of amplitude `value` and gradient `gradient` at a cell's centre over the cell; over the whole space (an
    /// infinite half-width) its global bounds, nothing at a centre.
    fn over(beam: &Geometry, at: &Amplitude, cell: &Cell) -> Span {
        let (k, n, bends, half) = (beam.wave_vector(), beam.direction, Bends::of(beam), cell.half);
        let (value, gradient, hessian) = (at.value, at.gradient, at.hessian);
        let rho = cell.radius();
        if !rho.is_finite() {
            return Span { centre: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3], wave: k, half, rho, most: 1.0, least: 0.0, slope: bends.lipschitz(), swing: f64::INFINITY, bend: f64::INFINITY, twist: f64::INFINITY, third: f64::INFINITY };
        }
        // The envelope's gradient at the centre (its phase e^{ik·c} left on: only moduli are read), and the cell's extents
        // across the beam and along it.
        let envelope: [C64; 3] = std::array::from_fn(|i| gradient[i] - C64::new(0.0, k[i]) * value);
        let across = (0..3).map(|i| half[i] * (1.0 - n[i] * n[i]).max(0.0).sqrt()).sum::<f64>().min(rho);
        let along = (0..3).map(|i| half[i] * n[i].abs()).sum::<f64>().min(rho);
        let [tt, tn, nn] = bends.curvature;
        let bend = tt * across * across + 2.0 * tn * across * along + nn * along * along;
        let twist = tt * across + tn * (across + along) + nn * along;
        // |∇∇∇b[·, d, d]|: its component across and its component along, each over the blocks of d's two extents.
        let [ttt, ttn, tnn, nnn] = bends.third;
        let third = (ttt + ttn) * across * across + 2.0 * (ttn + tnn) * across * along + (tnn + nnn) * along * along;
        let swing = ((0..3).map(|i| half[i] * envelope[i].abs()).sum::<f64>() + bend).min(bends.slope[0] * across + bends.slope[1] * along);
        // The centre's modulus is one of the cell's: the range holds it whatever its own rounding.
        let [least, most] = Span::moduli(beam, cell.centre, along, across);
        Span {
            centre: value,
            gradient,
            hessian,
            wave: k,
            half,
            rho,
            most: (value.abs() + swing).min(most.max(value.abs())),
            least: (value.abs() - swing).max(least.min(value.abs())),
            slope: (envelope.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() + twist).min(bends.lipschitz()),
            swing,
            bend,
            twist,
            third,
        }
    }
    /// The same beam on the cell's centre point alone: its envelope's slope there.
    pub(crate) fn point(&self) -> Span {
        let k = self.wave;
        let slope = (0..3).map(|i| (self.gradient[i] - C64::new(0.0, k[i]) * self.centre).norm_sqr()).sum::<f64>().sqrt();
        Span { most: self.centre.abs(), least: self.centre.abs(), slope, half: [0.0; 3], rho: 0.0, swing: 0.0, bend: 0.0, twist: 0.0, third: 0.0, ..*self }
    }
    /// The bound over a bounded cell of |dᵀ∇∇a′ d| of its amplitude in the frame of a wave vector k₀ (`Frame`), a′ =
    /// e^{−ik₀·d} a(c + d) = e^{ik·c} e^{iq·d} b(c + d), q = k − k₀: |−(q·d)² b + 2i (q·d)(∇b·d) + dᵀ∇∇b d| ≤ τ² max|b| +
    /// 2τ swing + bend, τ its plane wave's turn at q.
    pub(crate) fn bowed(&self, frame: &[f64; 3]) -> f64 {
        let turn = self.turn(&std::array::from_fn(|a| self.wave[a] - frame[a]));
        turn * turn * self.most + 2.0 * turn * self.swing + self.bend
    }
    /// The bound over a bounded cell of |q·d|, the turn of a plane wave of wave vector q from the centre: Σ_a |q_a| h_a.
    pub(crate) fn turn(&self, q: &[f64; 3]) -> f64 {
        (0..3).map(|a| q[a].abs() * self.half[a]).sum()
    }
    /// The bound over a bounded cell of |a′ − a(c)| in the frame of k₀ (`bowed`): |(e^{iq·d} − 1) b + (b − b(c))|, its
    /// plane wave's turn at q = k − k₀ times max|b|, and its envelope's swing.
    pub(crate) fn moved(&self, frame: &[f64; 3]) -> f64 {
        self.turn(&std::array::from_fn(|a| self.wave[a] - frame[a])) * self.most + self.swing
    }
    /// The bound over a bounded cell of |∇a − ∇a(c)| = |∇∇a d| = |e^{ik·r}(−k (k·d) b + i k (∇b·d) + i ∇b (k·d) + ∇∇b d)|:
    /// |k| τ max|b| + |k| swing + τ slope + twist, τ its plane wave's turn.
    pub(crate) fn veer(&self) -> f64 {
        let (k, turn) = (norm3(self.wave), self.turn(&self.wave));
        k * turn * self.most + k * self.swing + turn * self.slope + self.twist
    }
    /// The bound over the cell of |∇a| = |e^{ik·r}(ik b + ∇b)|, every phase blind: |k| max|b| + max|∇b|.
    pub(crate) fn steepest(&self) -> f64 {
        norm3(self.wave) * self.most + self.slope
    }
    /// The bound over a bounded cell of the remainder |∇a(c + d) − ∇a(c) − ∇∇a(c) d| of its gradient's first-order Taylor
    /// form (`rest`, of a = e^{ik·r} b: its envelope's bounds and its turn).
    pub(crate) fn rest(&self) -> f64 {
        let (k, turn) = (norm3(self.wave), self.turn(&self.wave));
        rest(k, turn, self.most, self.swing, self.bend, self.twist, self.slope, self.third)
    }
}

/// A beam pair's coefficient c = a_j'* a_i = e^{iq·r} B, B = b_j* b_i (q = k_i − k_j), over a cell: its value and gradient at the
/// centre, and three bounds over the cell (module documentation, # A cell's sides). With d a displacement from the centre
/// within the cell, τ = Σ_a |q_a| h_a the pair's turn, B̄ = max|b_j| max|b_i|, and of B what its two envelopes' bounds give
/// (`Span`) -- its slope G = max|∇B|, its swing D = max|∇B·d|, its bend Q = max|dᵀ∇∇B d| and its twist T = max|∇∇B d| --:
/// - `veer`, of |∇c − ∇c(c)|: ∇∇c d = e^{iq·r}(−q (q·d) B + i q (∇B·d) + i ∇B (q·d) + ∇∇B d), at most |q| τ B̄ + |q| D + τ G + T.
/// - `slope`, of |∇c|: |∇c(c)| + `veer`, at most |q| B̄ + G.
/// - `curve`, of the remainder |c − c(c) − ∇c(c)·d| of its first-order Taylor form: ½ max|dᵀ∇∇c d| = ½(τ² B̄ + 2τ D + Q)
///   (dᵀ∇∇c d = e^{iq·r}(−(q·d)² B + 2i (q·d)(∇B·d) + dᵀ∇∇B d)) -- what a sum of pairs adds to its own first order
///   (`Region::combined`).
/// - `radius`, of |c − c(c)|: to second order Σ_a h_a |∂_a c(c)| + `curve`, at most the first order's τ B̄ + D, at most
///   |c(c)| + B̄; and of a beam with itself, c = |b|² between its modulus's least and largest squared (`Span::moduli`), at
///   most the farther of the two from c(c).
/// - `most`, of |c|: B̄; `hessian`, ∇∇c at the centre; and `rest`, of the remainder |∇c − ∇c(c) − ∇∇c(c) d| of its
///   gradient's first-order Taylor form (`rest`).
/// - `wave`, q; `swing` and `bend`, D and Q: what its modulus reads in another frame (`Region::modulus`).
/// On a point: its value, its gradient's norm, the rest zero. On the whole space: nothing at the centre, B̄ and |q| B̄ + G.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct PairSpan {
    pub(crate) centre: C64,
    pub(crate) wave: [f64; 3],
    pub(crate) swing: f64,
    pub(crate) bend: f64,
    pub(crate) gradient: [C64; 3],
    pub(crate) hessian: [[C64; 3]; 3],
    pub(crate) radius: f64,
    pub(crate) curve: f64,
    pub(crate) most: f64,
    pub(crate) slope: f64,
    pub(crate) veer: f64,
    pub(crate) rest: f64,
}

/// The bound over a cell of the remainder |∇c(c + d) − ∇c(c) − ∇∇c(c) d| of the gradient's first-order Taylor form of a
/// plane wave times an envelope, c = e^{iq·r} B, with τ = Σ_a |q_a| h_a its turn and of B its bounds B̄ = max|B|,
/// D = max|∇B·d|, Q = max|dᵀ∇∇B d|, T = max|∇∇B d|, G = max|∇B| and E = max|∇∇∇B[·, d, d]|. With ∇c = e^{iq·r}(iqB + ∇B),
/// the remainder is e^{iq·c}(iq X + Y), X = B(c + d) e^{iq·d} − B − i(q·d)B − ∇B·d = B R_e + (∇B·d)(i q·d + R_e) + R_B
/// e^{iq·d} and Y = ∇B(c + d) e^{iq·d} − ∇B − i(q·d)∇B − ∇∇B d = ∇B R_e + ∇∇B d (e^{iq·d} − 1) + R'_B e^{iq·d}, with
/// |R_e| = |e^{iq·d} − 1 − iq·d| ≤ τ²/2, |R_B| ≤ Q/2 and |R'_B| = |∇B(c + d) − ∇B − ∇∇B d| ≤ E/2: at most
/// |q|(B̄τ²/2 + Dτ(1 + τ/2) + Q/2) + Gτ²/2 + Tτ + E/2 -- of the cell's second order throughout.
#[allow(clippy::too_many_arguments)]
fn rest(q: f64, turn: f64, most: f64, swing: f64, bend: f64, twist: f64, steep: f64, third: f64) -> f64 {
    q * (0.5 * most * turn * turn + swing * turn * (1.0 + 0.5 * turn) + 0.5 * bend) + 0.5 * steep * turn * turn + twist * turn + 0.5 * third
}

/// The pair (j, i) of beams over a cell (`PairSpan`).
pub(crate) fn pair_span(spans: &[Span], j: usize, i: usize) -> PairSpan {
    let (a, b) = (&spans[j], &spans[i]);
    let wave: [f64; 3] = std::array::from_fn(|x| b.wave[x] - a.wave[x]);
    let (q, most) = (norm3(wave), a.most * b.most);
    let steep = a.slope * b.most + a.most * b.slope;
    let blind = q * most + steep;
    if !a.rho.is_finite() {
        return PairSpan { centre: C64::ZERO, wave, swing: f64::INFINITY, bend: f64::INFINITY, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3], radius: most, curve: f64::INFINITY, most, slope: blind, veer: f64::INFINITY, rest: f64::INFINITY };
    }
    let turn = a.turn(&wave);
    let swing = a.swing * b.most + a.most * b.swing;
    let bend = a.bend * b.most + 2.0 * a.swing * b.swing + a.most * b.bend;
    let twist = a.twist * b.most + a.slope * b.swing + a.swing * b.slope + a.most * b.twist;
    // |∇∇∇B[·, d, d]| of B = b_j* b_i: the product rule's terms, ∇∇∇b_j b_i + 2 (∇∇b_j d)(∇b_i·d) + (dᵀ∇∇b_j d)∇b_i and the same
    // with j and i exchanged, and b_j ∇∇∇b_i.
    let third = a.third * b.most + 2.0 * a.twist * b.swing + a.bend * b.slope + 2.0 * a.swing * b.twist + a.slope * b.bend + a.most * b.third;
    let centre = a.centre.conj() * b.centre;
    let gradient: [C64; 3] = std::array::from_fn(|x| a.gradient[x].conj() * b.centre + a.centre.conj() * b.gradient[x]);
    let hessian: [[C64; 3]; 3] = std::array::from_fn(|x| {
        std::array::from_fn(|y| a.hessian[x][y].conj() * b.centre + a.gradient[x].conj() * b.gradient[y] + a.gradient[y].conj() * b.gradient[x] + a.centre.conj() * b.hessian[x][y])
    });
    let lean: f64 = (0..3).map(|x| gradient[x].abs() * a.half[x]).sum();
    let veer = q * turn * most + q * swing + turn * steep + twist;
    let curve = 0.5 * (turn * turn * most + 2.0 * turn * swing + bend);
    let radius = (lean + curve).min(turn * most + swing).min(centre.abs() + most);
    let radius = if j == i { radius.min((most - centre.re).max(centre.re - a.least * a.least)) } else { radius };
    let rest = rest(q, turn, most, swing, bend, twist, steep, third);
    PairSpan { centre, wave, swing, bend, gradient, hessian, radius, curve, most, slope: (gradient.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() + veer).min(blind), veer, rest }
}

/// One weighting w of the far light's static forms P_l'l (an entry's, a diagonal difference's), as a sum over the beam pairs
/// reads it: per pair of non-zero weight, in the pairs' order, its frequency and its place among the frequency's pairs, w(P_l'l)
/// and |w(P_l'l)|.
pub(crate) type Weights = Vec<(usize, usize, C64, f64)>;

/// A carrier's folded couplings X^f_l as the regions' bounds read them (eliminate.rs `Model::couplings`, `Model::dressing_shifts`).
pub(crate) struct Folded {
    /// Its polarization units, distinct and ascending.
    pub(crate) units: Vec<usize>,
    /// [ground state][beam]: ‖X^f_l[:, s]‖.
    pub(crate) columns: Vec<Vec<f64>>,
    /// [ground state][unit]: the unit's beams of a non-zero column, in their order -- the beam's place in the carrier's, and its
    /// column's entries that are not zero: the excited state (less GROUND), X^f_l[e, s] and its modulus.
    pub(crate) beams: Vec<Vec<Vec<(usize, Vec<(usize, C64, f64)>)>>>,
    /// [ground state]: the excited states (less GROUND) its beams' couplings reach from it, ascending.
    pub(crate) reached: Vec<Vec<usize>>,
}

/// One drive of a ground state as its gate reads it (units.rs `Model::folds`), in the tail rule's order.
pub(crate) struct DriveForm {
    /// Its component (2 frequency + counter), its excited state (less GROUND), its level and its place in the tail rule's
    /// order (units.rs `Ranked::group`).
    pub(crate) component: usize,
    pub(crate) excited: usize,
    pub(crate) level: Level,
    pub(crate) group: usize,
    /// Its beams' entries that are not zero: the request's beam, X_l[e, g] (of a counter-rotating one its conjugate, as its
    /// field enters) and its modulus.
    pub(crate) beams: Vec<(usize, C64, f64)>,
    /// With the bare levels: its detuning E_e − E_g − ν and its denominator's modulus |E_e − E_g − ν − iΓ_e/2|; Γ_e.
    pub(crate) detuning: f64,
    pub(crate) bare: f64,
    pub(crate) gamma: f64,
}

/// One excited state through which two frequencies' drives couple two ground states at the atom (`TermForm::legs`): the
/// emitted frequency's drive e ← a and the absorbed one's e ← b, each by its beams' entries (the request's beam, X_l and its
/// modulus) and, where it is folded with the levels at the atom, its frequency's detuning δ.
pub(crate) struct TermLeg {
    pub(crate) excited: usize,
    pub(crate) to: Vec<(usize, C64, f64)>,
    pub(crate) from: Vec<(usize, C64, f64)>,
    pub(crate) to_atom: Option<f64>,
    pub(crate) from_atom: Option<f64>,
}

/// A coupling of two states a ← b at one of the model's beats (`Unit::Term`) as its bound reads it (units.rs
/// `Model::term_bound`).
pub(crate) struct TermForm {
    /// Its constant part: at the zero beat the two states' own coupling with every light off (`Model::base`); and per pair
    /// of beams whose entry of the frequencies' constant form (far.rs `constant`) is not zero, the emitted beam and the
    /// absorbed one (the request's), C_ji[a, b] and its modulus.
    pub(crate) base: C64,
    pub(crate) pairs: Vec<(usize, usize, C64, f64)>,
    /// Its part at the atom, of two ground states: per excited state a drive folded there reaches from one of them and a
    /// folded drive of the other frequency from the other.
    pub(crate) legs: Vec<TermLeg>,
    /// The largest |k_i − k_j| of its beam pairs (1/µm).
    pub(crate) wave: f64,
}

impl TermForm {
    /// Whether nothing couples the two states at the beat.
    pub(crate) fn is_empty(&self) -> bool {
        self.base == C64::ZERO && self.pairs.is_empty() && self.legs.is_empty()
    }

    /// An entry of two frequencies' constant form within the rounding of the form's own sums is that rounding, not a
    /// coupling (units.rs `rounding`: a quantity within γ_N of its operands' scale) -- a field's component that is zero comes
    /// as the last bits of its vector's, and so does every product with it: per pair of basis vectors two coefficients times
    /// an entry, itself the outside forms' nine terms and two folds' sums over the excited states, of the forms' largest
    /// entry. The same for every entry of a form: made once with it (units.rs `Model::constants`).
    pub(crate) fn rounding_of(model: &Model, to: usize, from: usize, pairs: &[Mat]) -> f64 {
        let terms = model.tones[to].basis[0].len() * model.tones[from].basis[0].len() * (2 * EXCITED + 11);
        rounding(terms) * pairs.iter().map(|m| m.max_abs()).fold(0.0, f64::max)
    }

    fn new(model: &Model, a: usize, b: usize, beat: usize) -> TermForm {
        let mut out = TermForm { base: C64::ZERO, pairs: vec![], legs: vec![], wave: 0.0 };
        let between = |lj: usize, li: usize| norm3(model.wave_value(&model.pair_wave(lj, li)));
        // The states' own coupling with every light off is H0, diagonal, less the outside forms at the focus: theirs.
        let own = model.base[(a, b)];
        if model.beats[beat].0.is_zero() && own.abs() > (0..model.tones.len()).map(|f| model.constants(f, f).rounding).fold(0.0, f64::max) {
            out.base = own;
        }
        for &(to, from) in &model.beats[beat].1 {
            let (bt, bf) = (&model.light.frequencies[to].beams, &model.light.frequencies[from].beams);
            let constants = model.constants(to, from);
            let least = constants.rounding;
            for (x, m) in constants.pairs.iter().enumerate() {
                let v = m[(a, b)];
                if v.abs() > least {
                    let (lj, li) = (bt[x / bf.len()], bf[x % bf.len()]);
                    out.pairs.push((lj, li, v, v.abs()));
                    out.wave = out.wave.max(between(lj, li));
                }
            }
            if a >= GROUND || b >= GROUND || (a == b && to == from) {
                continue;
            }
            let (ct, cf) = (Component { frequency: to, counter: false }, Component { frequency: from, counter: false });
            let entries = |c: Component, e: usize, s: usize| -> Vec<(usize, C64, f64)> {
                let beams = &model.light.frequencies[c.frequency].beams;
                beams.iter().zip(&model.tones[c.frequency].couplings[0]).map(|(&l, x)| (l, x[(e, s)], x[(e, s)].abs())).filter(|entry| entry.2 != 0.0).collect()
            };
            for e in GROUND..STATES {
                let (lt, lf) = (model.level(ct, e, a), model.level(cf, e, b));
                if lt == Level::Explicit || lf == Level::Explicit || (lt == Level::Bare && lf == Level::Bare) {
                    continue;
                }
                if !model.tones[to].drives(false, e, a) || !model.tones[from].drives(false, e, b) {
                    continue;
                }
                let (to_beams, from_beams) = (entries(ct, e, a), entries(cf, e, b));
                for &(lj, ..) in &to_beams {
                    for &(li, ..) in &from_beams {
                        out.wave = out.wave.max(between(lj, li));
                    }
                }
                out.legs.push(TermLeg {
                    excited: e - GROUND,
                    to: to_beams,
                    from: from_beams,
                    to_atom: (lt == Level::Atom).then(|| model.tones[to].detuning),
                    from_atom: (lf == Level::Atom).then(|| model.tones[from].detuning),
                });
            }
        }
        out
    }
}

/// What the regions' bounds read of the model's forms -- the constant forms P_l'l of each frequency (far.rs), the base, the
/// carriers' couplings folded at the atom (eliminate.rs), the ground states' drives and the couplings of two states (units.rs)
/// --, each as its bound sums it: the entries that are not zero, with their moduli. The same for every region of a model, it
/// is the model's (`Model::forms`), each part made when first asked.
pub(crate) struct Forms {
    /// Per excited state x (less GROUND) the excited states y ≠ x it is coupled to somewhere -- by the base or by a form --,
    /// ascending, each with the base's entry.
    mixed: OnceLock<Vec<Vec<(usize, C64)>>>,
    /// Per pair of states (x, y), row-major: the weighting of the forms' entry [P_l'l]_xy.
    entries: Vec<OnceLock<Weights>>,
    /// Per frequency and beam pair whose form has a diagonal entry that is not zero: the frequency, the pair, and per state
    /// its diagonal entry [P_l'l]_xx (eliminate.rs `Model::gap`).
    diagonals: OnceLock<Vec<(usize, usize, Vec<C64>)>>,
    /// Per carrier.
    folded: OnceLock<Vec<Folded>>,
    /// Per ground state its drives in the tail rule's order (units.rs `Model::ranked`).
    drives: OnceLock<Vec<Vec<DriveForm>>>,
    /// Per coupling asked.
    terms: Mutex<HashMap<Coupling, Arc<TermForm>>>,
    /// Γ_min (eliminate.rs `Model::least_decay`).
    decay: OnceLock<f64>,
    /// [frequency][pair]: whether the levels the eliminations take hold the pair (eliminate.rs `Model::level_pairs`, every
    /// light on).
    taken: OnceLock<Vec<Vec<bool>>>,
}

impl Forms {
    pub(crate) fn new() -> Forms {
        let cells = || (0..STATES * STATES).map(|_| OnceLock::new()).collect();
        Forms { mixed: OnceLock::new(), entries: cells(), diagonals: OnceLock::new(), folded: OnceLock::new(), drives: OnceLock::new(), terms: Mutex::new(HashMap::new()), decay: OnceLock::new(), taken: OnceLock::new() }
    }

    /// A weighting of the model's far static forms (`Weights`).
    fn weights(model: &Model, w: &dyn Fn(&Mat) -> C64) -> Weights {
        let mut out = vec![];
        for (frequency, tone) in model.tones.iter().enumerate() {
            for (pair, p) in tone.pairs.iter().enumerate() {
                let x = w(p);
                if x != C64::ZERO {
                    out.push((frequency, pair, x, x.abs()));
                }
            }
        }
        out
    }

    /// Per excited state (less GROUND) the excited states it is coupled to somewhere, with the base's entries (`Forms::mixed`).
    pub(crate) fn mixed(&self, model: &Model) -> &[Vec<(usize, C64)>] {
        self.mixed.get_or_init(|| {
            (GROUND..STATES)
                .map(|x| (GROUND..STATES).filter(|&y| y != x).map(|y| (y, model.base[(x, y)])).filter(|&(y, base)| base != C64::ZERO || !self.entry(model, x, y).is_empty()).collect())
                .collect()
        })
    }

    /// The weighting of the forms' entry [P_l'l]_xy.
    pub(crate) fn entry(&self, model: &Model, x: usize, y: usize) -> &Weights {
        self.entries[x * STATES + y].get_or_init(|| Forms::weights(model, &|p| p[(x, y)]))
    }

    /// The forms' diagonals per beam pair (`Forms::diagonals`).
    pub(crate) fn diagonals(&self, model: &Model) -> &[(usize, usize, Vec<C64>)] {
        self.diagonals.get_or_init(|| {
            let mut out = vec![];
            for (frequency, tone) in model.tones.iter().enumerate() {
                for (pair, p) in tone.pairs.iter().enumerate() {
                    let w: Vec<C64> = (0..STATES).map(|s| p[(s, s)]).collect();
                    if w.iter().any(|x| *x != C64::ZERO) {
                        out.push((frequency, pair, w));
                    }
                }
            }
            out
        })
    }

    /// Per carrier its couplings folded at the atom (`Folded`).
    pub(crate) fn folded(&self, model: &Model) -> &[Folded] {
        self.folded.get_or_init(|| {
            model
                .carriers
                .iter()
                .map(|cc| {
                    let mut units = cc.units.clone();
                    units.sort();
                    units.dedup();
                    let entries = |l: usize, s: usize| -> Vec<(usize, C64, f64)> {
                        (0..EXCITED).map(|e| (e, cc.folded[l][(GROUND + e, s)])).filter(|&(_, v)| v != C64::ZERO).map(|(e, v)| (e, v, v.abs())).collect()
                    };
                    let columns: Vec<Vec<f64>> =
                        (0..GROUND).map(|s| cc.folded.iter().map(|x| (0..EXCITED).map(|e| x[(GROUND + e, s)].norm_sqr()).sum::<f64>().sqrt()).collect()).collect();
                    let beams = (0..GROUND)
                        .map(|s| units.iter().map(|&unit| (0..cc.folded.len()).filter(|&l| cc.units[l] == unit && columns[s][l] != 0.0).map(|l| (l, entries(l, s))).collect()).collect())
                        .collect();
                    let reached = (0..GROUND).map(|s| (0..EXCITED).filter(|&e| cc.folded.iter().any(|m| m[(GROUND + e, s)] != C64::ZERO)).collect()).collect();
                    Folded { units, columns, beams, reached }
                })
                .collect()
        })
    }

    /// Per ground state its drives in the tail rule's order (`DriveForm`).
    pub(crate) fn drives(&self, model: &Model) -> &[Vec<DriveForm>] {
        self.drives.get_or_init(|| {
            model
                .ranked
                .iter()
                .enumerate()
                .map(|(g, ranked)| {
                    ranked
                        .iter()
                        .map(|d| {
                            let side = d.component.counter as usize;
                            let (f, tone) = (&model.light.frequencies[d.component.frequency], &model.tones[d.component.frequency]);
                            let beams = f.beams.iter().zip(&tone.couplings[side]).map(|(&l, x)| (l, x[(d.excited, g)])).filter(|entry| entry.1 != C64::ZERO);
                            DriveForm {
                                component: 2 * d.component.frequency + side,
                                excited: d.excited - GROUND,
                                level: model.level(d.component, d.excited, g),
                                group: d.group,
                                beams: beams.map(|(l, x)| (l, if side == 1 { x.conj() } else { x }, x.abs())).collect(),
                                detuning: model.energies[d.excited] - model.energies[g] - tone.nu[side],
                                bare: model.bare_distance(d.component, d.excited, g),
                                gamma: model.decay[(d.excited - GROUND, d.excited - GROUND)].re,
                            }
                        })
                        .collect()
                })
                .collect()
        })
    }

    /// What the bound of the coupling a ← b at the model's beat `beat` reads (`TermForm`), made once.
    pub(crate) fn term(&self, model: &Model, a: usize, b: usize, beat: usize) -> Arc<TermForm> {
        if let Some(made) = self.terms.lock().expect("the couplings' forms' lock").get(&(a, b, beat)) {
            return made.clone();
        }
        let made = Arc::new(TermForm::new(model, a, b, beat));
        self.terms.lock().expect("the couplings' forms' lock").entry((a, b, beat)).or_insert(made).clone()
    }

    /// The least decay rate of the excited states, Γ_min.
    pub(crate) fn least_decay(&self, model: &Model) -> f64 {
        *self.decay.get_or_init(|| model.least_decay())
    }

    /// Per frequency and beam pair whether the levels the eliminations take hold it, every light on.
    pub(crate) fn taken(&self, model: &Model) -> &[Vec<bool>] {
        self.taken.get_or_init(|| model.level_pairs(&model.running(), &model.coherent_dressing()))
    }

}

/// A cell's beams, or its centre point's (its radius zero), with what the units' bounds share over it, each part made once for
/// every quantity that asks (module documentation, # Provable extrema): the reach's speeds over it, the beam pairs per frequency,
/// the excited states' Gershgorin radii, the carriers' coherent couplings, resolvents and dressings of the ground states, the
/// clusters' force bounds (eliminate.rs) and the ground states' drives' gates (units.rs).
pub(crate) struct Region {
    pub(crate) spans: Vec<Span>,
    /// The cell's half-widths and its radius (0: a point).
    pub(crate) half: [f64; 3],
    pub(crate) rho: f64,
    /// The bound of the atoms' speed over it (µm/µs): the reach's at its least V_prep (module documentation, # The reach); a
    /// sample's own on its point. And the speed's own form (own.rs): (2 (ħ/m)(level − V_prep))^{1/2} of V_prep's own form,
    /// within the reach's -- a constant on a point, on a path's box and on the whole space.
    pub(crate) speed: f64,
    pub(crate) pace: Own,
    /// Per frequency, its beam pairs (j, i) row-major (`PairSpan`).
    pub(crate) pairs: Vec<Vec<PairSpan>>,
    /// What its bounds read of the model's forms.
    pub(crate) forms: Arc<Forms>,
    /// Per state (eliminate.rs `Model::radii`).
    pub(crate) radii: OnceLock<[Own; STATES]>,
    /// Per carrier.
    pub(crate) couplings: OnceLock<Vec<Couplings>>,
    /// [carrier][ground state]: its dressing's resolvent (eliminate.rs `Model::resolvent_own`).
    pub(crate) resolvents: OnceLock<Vec<[Own; GROUND]>>,
    /// [carrier][ground state].
    pub(crate) dressings: OnceLock<Vec<[Own; GROUND]>>,
    /// [carrier][ground state]: the dressing's shift with its sign (eliminate.rs `Model::dressing_shifts`).
    pub(crate) shifted: OnceLock<Vec<[Shift; GROUND]>>,
    /// Per cluster.
    pub(crate) forces: Vec<OnceLock<Own>>,
    /// Per ground state its drives' sizes and distances (units.rs `Model::across`), the tails of their shares and the gates
    /// of the places asked in the tail rule's order (units.rs `Model::tails`, `Model::folds`).
    pub(crate) across: [OnceLock<Vec<super::units::Across>>; GROUND],
    pub(crate) tails: [OnceLock<Vec<Own>>; GROUND],
    pub(crate) folds: [Mutex<Vec<(usize, [f64; 5])>>; GROUND],
    /// Per ground state its shift by its drives folded at the atom, signed (units.rs `Model::near_level`).
    pub(crate) shifts: [OnceLock<Shift>; GROUND],
    /// The constant forms' diagonal per state, and the levels' differences and their gradients asked, each made once
    /// (eliminate.rs `Model::gap`, `Model::gap_slope`).
    pub(crate) diagonal: OnceLock<Diagonal>,
    pub(crate) gaps: Mutex<HashMap<(usize, usize), Gap>>,
    pub(crate) slopes: Mutex<HashMap<(usize, usize), [Form<[f64; 3]>; 2]>>,
    /// Per coupling of two states asked its saturation and sweep (units.rs `Model::term_bound`).
    pub(crate) terms: Mutex<HashMap<Coupling, (f64, f64)>>,
    /// The model's kept drives (units.rs `Model::kept_over`).
    pub(crate) kept: OnceLock<Vec<super::units::Kept>>,
    /// The slope of any two levels' difference and an atom's acceleration (units.rs `Model::gate_over`).
    pub(crate) moving: OnceLock<(Own, Own)>,
}

/// A coupling of two states a ← b at a beat, its place among the model's beats (`Unit::Term`'s identity in a model).
pub(crate) type Coupling = (usize, usize, usize);

impl Region {
    /// The bound over the region of how far a sum f = Σ_p w_p c_p of beam pairs' coefficients (`PairSpan`, each with its
    /// weight and the weight's modulus) moves from its value at the centre -- of its real part where `real` --, from its own
    /// form: its first order Σ_a h_a |∂_a f(c)| from its gradient at the centre, Σ_p w_p ∇c_p(c), and its Taylor remainder
    /// Σ_p |w_p| `curve`_p; at most Σ_p |w_p| `radius`_p, each pair's own range, which a coarse cell's turn of the phases
    /// leaves the tighter. Never only the latter: two pairs' movements that cancel in the sum -- a level difference's shift
    /// the two levels share, a standing wave's two counter-rotating coefficients at a crest, Re(c₁₂ + c₂₁) = 2cos 2kx flat
    /// where each |∇c| = 2k -- read added there, and the bound reaches its value only on cells far below the light's
    /// length (the gates' proofs over the lin⊥lin scene's reach never ended). Nothing on a point; each pair's range on the
    /// whole space.
    pub(crate) fn combined(&self, terms: impl Iterator<Item = (PairSpan, C64, f64)>) -> Combination {
        Combination::of(&self.half, terms)
    }

    /// ∇f over the region of a sum f = Σ_p w_p c_p of beam pairs' coefficients, in its own form (own.rs `Form`): of its
    /// gradient and Hessian at the centre, Σ_p w_p ∇c_p(c) and Σ_p w_p ∇∇c_p(c), and the gradient's Taylor remainder
    /// Σ_p |w_p| `rest`_p; it moves by Σ_p |w_p| `veer`_p, each pair's gradient's own range, and is at most Σ_p |w_p|
    /// `slope`_p. (With each pair's range alone, a standing wave's term 2|w| cos 2kx read its gradient's change at its node,
    /// where it is flat, as 2k² h |w| -- added over its two counter-rotating pairs, a first order the sum does not have.) On
    /// the whole space, at most the pairs' slopes.
    pub(crate) fn gradient(&self, terms: impl Iterator<Item = (PairSpan, C64, f64)>) -> Form<[C64; 3]> {
        let (mut value, mut hessian) = ([C64::ZERO; 3], [[C64::ZERO; 3]; 3]);
        let (mut veer, mut rest, mut steep) = (0.0, 0.0, 0.0);
        for (pair, w, size) in terms {
            for a in 0..3 {
                value[a] += pair.gradient[a] * w;
                for b in 0..3 {
                    hessian[a][b] += pair.hessian[a][b] * w;
                }
            }
            veer += size * pair.veer;
            rest += size * pair.rest;
            steep += size * pair.slope;
        }
        if !self.rho.is_finite() {
            return Form::whole(steep);
        }
        Form { value, first: std::array::from_fn(|a| hessian[a].map(|z| z * self.half[a])), rest, moved: veer, most: steep }
    }

    /// A sum u = Σ_l a_l x_l of beams' amplitudes times numbers (the request's beam, x_l, |x_l|) and its gradient over the
    /// region, each in its own form (own.rs `Form`): u of its value and slopes at the centre, its Taylor remainder ½ Σ_l |x_l|
    /// max|dᵀ∇∇a_l d| (`Span::bowed`), moving by Σ_l |x_l| max|a_l − a_l(c)| (`Span::moved`) and at most Σ_l |x_l| max|a_l|;
    /// ∇u of its value and second derivatives at the centre, its Taylor remainder Σ_l |x_l| `Span::rest`, moving by Σ_l |x_l|
    /// `Span::veer` and at most Σ_l |x_l| `Span::steepest`. On the whole space, each at most its bound.
    pub(crate) fn field(&self, entries: impl Iterator<Item = (usize, C64, f64)>) -> (Form<C64>, Form<[C64; 3]>) {
        let (mut u, mut slopes, mut second) = (C64::ZERO, [C64::ZERO; 3], [[C64::ZERO; 3]; 3]);
        let mut sums = [0.0; 6];
        let bounded = self.rho.is_finite();
        for (l, x, size) in entries {
            let span = &self.spans[l];
            u += span.centre * x;
            for a in 0..3 {
                slopes[a] += span.gradient[a] * x;
                for b in 0..3 {
                    second[a][b] += span.hessian[a][b] * x;
                }
            }
            let over = if bounded { [span.moved(&[0.0; 3]), span.bowed(&[0.0; 3]), span.most, span.veer(), span.rest(), span.steepest()] } else { [0.0, 0.0, span.most, 0.0, 0.0, span.steepest()] };
            for (sum, x) in sums.iter_mut().zip(over) {
                *sum += size * x;
            }
        }
        if !bounded {
            return (Form::whole(sums[2]), Form::whole(sums[5]));
        }
        let half = self.half;
        (
            Form { value: u, first: std::array::from_fn(|a| slopes[a] * half[a]), rest: 0.5 * sums[1], moved: sums[0], most: sums[2] },
            Form { value: slopes, first: std::array::from_fn(|a| second[a].map(|z| z * half[a])), rest: sums[4], moved: sums[3], most: sums[5] },
        )
    }

    /// A coupling's saturation and sweep over the region, made once for both of its gates' bounds.
    pub(crate) fn term(&self, coupling: Coupling, make: impl FnOnce() -> (f64, f64)) -> (f64, f64) {
        let terms = || self.terms.lock().expect("the couplings' lock");
        if let Some(made) = terms().get(&coupling) {
            return *made;
        }
        let made = make();
        terms().insert(coupling, made);
        made
    }

    /// |u|² = |Σ_l a_l x_l|² over the region of a frequency's beams (the request's beam, x_l, |x_l|), and its gradient, each in
    /// its own form: of its own beam pairs, Σ_{j,i} x_j* x_i c_ji (`PairSpan`; `combined`, `gradient`) -- a beam's own pair its
    /// intensity, read from its modulus with no phase to turn (`Span::moduli`); two beams' pair turning at their wave vectors'
    /// difference. (As the square of |u|'s own form, one beam's plane wave turned u's phase across the cell and put the turn's
    /// square into the intensity's remainder, k²h²|u|², where the intensity does not move.)
    pub(crate) fn intensity(&self, model: &Model, frequency: usize, entries: &[(usize, C64, f64)]) -> (Own, Form<[f64; 3]>) {
        let beams = &model.light.frequencies[frequency].beams;
        let n = beams.len();
        let place = |l: usize| beams.iter().position(|&b| b == l).expect("an entry's beam is its frequency's");
        let pairs = || entries.iter().flat_map(move |&(lj, xj, sj)| entries.iter().map(move |&(li, xi, si)| (self.pairs[frequency][place(lj) * n + place(li)], xj.conj() * xi, sj * si)));
        (self.combined(pairs()).real().within(0.0, f64::INFINITY), self.gradient(pairs()).re())
    }

    /// |Σ_l a_l x_l| over the region in its own form (`Frame`): of the beams `entries` (the request's beam, x_l, |x_l|).
    pub(crate) fn coherent_own(&self, entries: &[(usize, C64, f64)]) -> Own {
        let mut sum = Frame::<1>::of(self, entries.iter().map(|&(l, _, size)| (l, size)));
        for &(l, x, size) in entries {
            sum.add(self, l, size, std::iter::once((0, x)));
        }
        sum.norm(self)
    }

    /// |f + base| of a sum f = Σ_p w_p c_p of beam pairs' coefficients (`PairSpan`, each with its weight and the weight's
    /// modulus) and a constant `base` over the region, in its own form (own.rs `Own::norm`) in the frame of a wave vector q₀
    /// as a sum of beams' is (`Frame`): |f + base| = |e^{−iq₀·(r − c)}(f + base)|, and in that frame a pair turns by τ_p =
    /// Σ_a |q_p,a − q₀,a| h_a and the base by τ₀ = Σ_a |q₀,a| h_a, not each by its own q (a pair's modulus |b_j||b_i| has no
    /// phase to follow, two pairs of one q turn together). Its value and slopes ∂_a f − iq₀,a (f + base) at the centre, its
    /// remainder Σ_p |w_p| ½(τ_p² B̄_p + 2τ_p D_p + Q_p) + |base| ½τ₀²; it moves by Σ_p |w_p| min(τ_p B̄_p + D_p, |c_p(c)| +
    /// B̄_p) + |base| τ₀ at most and is at most |base| + Σ_p |w_p| B̄_p. q₀ the pairs' wave vectors' mean weighted by |w_p| B̄_p,
    /// the base's (q = 0) by |base|. On the whole space, [0, that bound].
    pub(crate) fn modulus(&self, terms: impl Iterator<Item = (PairSpan, C64, f64)> + Clone, base: C64) -> Own {
        let (mut mean, mut weight) = ([0.0; 3], base.abs());
        for (pair, _, size) in terms.clone() {
            for a in 0..3 {
                mean[a] += size * pair.most * pair.wave[a];
            }
            weight += size * pair.most;
        }
        let frame = if weight > 0.0 { mean.map(|x| x / weight) } else { [0.0; 3] };
        let turn = |q: [f64; 3]| -> f64 { (0..3).map(|a| (q[a] - frame[a]).abs() * self.half[a]).sum() };
        let base_turn = turn([0.0; 3]);
        let (mut value, mut slopes) = (base, [C64::ZERO; 3]);
        let (mut rest, mut moved, mut blind) = (0.5 * base_turn * base_turn * base.abs(), base_turn * base.abs(), base.abs());
        for (pair, w, size) in terms {
            value += pair.centre * w;
            for a in 0..3 {
                slopes[a] += pair.gradient[a] * w;
            }
            let tau = turn(pair.wave);
            rest += size * 0.5 * (tau * tau * pair.most + 2.0 * tau * pair.swing + pair.bend);
            moved += size * (tau * pair.most + pair.swing).min(pair.centre.abs() + pair.most);
            blind += size * pair.most;
        }
        let slopes: [C64; 3] = std::array::from_fn(|a| slopes[a] - C64::new(0.0, frame[a]) * value);
        Own::norm(&[value], [&slopes[0..1], &slopes[1..2], &slopes[2..3]], rest, moved, blind, self.half)
    }
}

/// A sum u = Σ_l a_l x_l of beams' amplitudes times vectors x_l (over the excited states) gathered for its norm over a region
/// (`Region::coherent_own`, eliminate.rs `Model::couplings`), in the frame of a wave vector k₀. ‖u‖ = ‖e^{−ik₀·(r − c)} u‖ for
/// any constant k₀, and in that frame a beam's amplitude is e^{ik·c} e^{i(k − k₀)·d} b(c + d): it turns by τ = Σ_a |k_a −
/// k₀,a| h_a, not by its own wave vector's turn (`Span::moved`, `Span::bowed`). So the norm's own form (own.rs `Own::norm`)
/// is of u's value and slopes ∂_a u − ik₀,a u at the centre -- its first order Re⟨u, ∂_a u⟩/‖u‖ the same in every frame --,
/// its Taylor remainder ½ Σ_l ‖x_l‖ max|dᵀ∇∇a′_l d|, within ‖u(c)‖ ± Σ_l ‖x_l‖ max|a′_l − a_l(c)| and at most Σ_l ‖x_l‖
/// max|a_l|. k₀ is the beams' wave vectors' mean weighted by ‖x_l‖ max|a_l|, the minimizer of Σ_l w_l |k_l − k₀|²: one
/// beam does not turn at all (its modulus is its envelope's), a beam with a weak partner turns by the partner's share, a
/// balanced counter-propagating pair by ±k about k₀ = 0. (In the lab's frame, each beam turning by its own k·d, the
/// norm of the user's probe's drive -- its reflection on that line 4·10⁻⁸ of it -- had a remainder ½ τ² ‖u‖ of the cell's
/// square where the modulus moves by the partner's 10⁻⁸: its bound was the beams' phase-blind sum, 8.5·10⁻⁸ of it above
/// the largest value on every cell down to ℓ/4000.)
pub(crate) struct Frame<const N: usize> {
    wave: [f64; 3],
    u: [C64; N],
    slopes: [[C64; N]; 3],
    moved: f64,
    bowed: f64,
    blind: f64,
}

impl<const N: usize> Frame<N> {
    /// An empty sum of N places in the frame of the beams `beams` (the request's beam, ‖x_l‖).
    pub(crate) fn of(region: &Region, beams: impl Iterator<Item = (usize, f64)>) -> Frame<N> {
        let (mut sum, mut weight) = ([0.0; 3], 0.0);
        for (l, size) in beams {
            let span = &region.spans[l];
            let w = size * span.most;
            for a in 0..3 {
                sum[a] += w * span.wave[a];
            }
            weight += w;
        }
        let wave = if weight > 0.0 { sum.map(|x| x / weight) } else { [0.0; 3] };
        Frame { wave, u: [C64::ZERO; N], slopes: [[C64::ZERO; N]; 3], moved: 0.0, bowed: 0.0, blind: 0.0 }
    }

    /// The beam l's amplitude times x_l, of norm `size` and entries (place, value) that are not zero.
    pub(crate) fn add(&mut self, region: &Region, l: usize, size: f64, entries: impl Iterator<Item = (usize, C64)>) {
        let span = &region.spans[l];
        let slopes: [C64; 3] = std::array::from_fn(|a| span.gradient[a] - C64::new(0.0, self.wave[a]) * span.centre);
        for (e, x) in entries {
            self.u[e] += span.centre * x;
            for a in 0..3 {
                self.slopes[a][e] += slopes[a] * x;
            }
        }
        if region.rho.is_finite() {
            self.moved += size * span.moved(&self.wave);
            self.bowed += size * span.bowed(&self.wave);
        }
        self.blind += size * span.most;
    }

    /// ‖u‖ over the region in its own form (own.rs `Own::norm`); on the whole space [0, Σ_l ‖x_l‖ max|a_l|].
    pub(crate) fn norm(&self, region: &Region) -> Own {
        Own::norm(&self.u, [&self.slopes[0], &self.slopes[1], &self.slopes[2]], 0.5 * self.bowed, self.moved, self.blind, region.half)
    }
}

/// A cell of the proofs' tree: its box, the bounds of the reach's speeds over it and at its centre (none on an unbounded
/// cell), whether its centre is in the reach, and per side its halves along it once made (none where a half cannot meet the
/// reach).
struct Node {
    cell: Cell,
    over: f64,
    at: Option<f64>,
    pace: Own,
    inside: bool,
    halves: [Option<[Option<usize>; 2]>; 3],
}

/// A cell's regions (module documentation, # Provable extrema): the cell's and its centre point's (none on an unbounded cell).
/// Each on the heap, not in the frame of the task that holds it: a round's cells are halved by the pool's tasks
/// (`AtOnce::cells`), a task that waits for a part of the round taken by another thread goes on with other parts on its
/// own stack, and a halving holds up to fifteen regions -- by value they were 51 kB a level of that nesting, which
/// grows with the round's cells, and a reach of many cells came to the thread's whole stack (R8_R5051 over a reach of
/// 271 k_B T: 654 kB at the first closure, the stack overflown at the third).
struct Regions {
    over: Box<Region>,
    at: Option<Box<Region>>,
}

/// A cell in a quantity's queue, ordered by its upper bound: the bound, the cell's node, the quantity's value at its centre, and
/// whether the centre is in the reach.
#[derive(Clone, Copy)]
struct Queued {
    upper: f64,
    node: usize,
    centre: f64,
    inside: bool,
}
impl PartialEq for Queued {
    fn eq(&self, o: &Self) -> bool {
        self.upper == o.upper
    }
}
impl Eq for Queued {}
impl PartialOrd for Queued {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Queued {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.upper.total_cmp(&o.upper)
    }
}

/// What the cells of one proving are made from: the model and the reach, and the prepared potential's lights and envelope.
struct Cells<'a> {
    model: &'a Model,
    reach: &'a Reach,
    prepared: Lights,
    envelope: Envelope,
}

impl Cells<'_> {
    /// A cell's node with its regions, none where the cell cannot meet the reach: a lower bound of V_prep over it above the
    /// reach's level -- above its level outside the trap's region where the cell cannot meet the region (module
    /// documentation, # The trap's region).
    fn make(&self, cell: Cell) -> Option<(Node, Regions)> {
        let (model, reach) = (self.model, self.reach);
        // The reach's speeds at a level where V_prep is at least v.
        let speed = |level: f64, v: f64| reach.speed.min((2.0 * model.hbar_over_mass * (level - v).max(0.0)).sqrt());
        let spans = model.spans(&cell);
        let node = if cell.bounded() {
            let centre = model.potential(cell.centre, &reach.populations, &self.prepared);
            let prepared = prepared_own(&self.envelope, &spans, centre);
            let below = prepared.lower();
            // The cell's largest level and its centre's.
            let level = reach.minimum + reach.energy;
            let (most, at) = if reach.outside < reach.energy {
                let beyond = reach.minimum + reach.outside;
                (if model.region_of(&spans)[0] { level } else { beyond }, if model.light.outside(cell.centre) { beyond } else { level })
            } else {
                (level, level)
            };
            if below > most {
                return None;
            }
            // The speeds' own form: (2 (ħ/m)(level − V_prep))^{1/2}, within the reach's.
            let pace = prepared.scale(-2.0 * model.hbar_over_mass).plus(2.0 * model.hbar_over_mass * most).max(Own::constant(0.0)).sqrt().within(0.0, reach.speed);
            Node { cell, over: speed(most, below), at: Some(speed(at, centre)), pace, inside: centre <= at, halves: [None; 3] }
        } else {
            Node { cell, over: reach.speed, at: None, pace: Own::constant(reach.speed), inside: false, halves: [None; 3] }
        };
        let regions = self.over(&node, spans);
        Some((node, regions))
    }

    /// A node's regions from its cell's beams.
    fn over(&self, node: &Node, spans: Vec<Span>) -> Regions {
        let at = node.at.map(|speed| Box::new(self.model.region(spans.iter().map(Span::point).collect(), speed)));
        Regions { over: Box::new(self.model.region_paced(spans, node.over, node.pace)), at }
    }

    /// A node's regions, made again.
    fn regions(&self, node: &Node) -> Regions {
        self.over(node, self.model.spans(&node.cell))
    }
}

/// A sum f = Σ_p w_p c_p of beam pairs' coefficients over a cell of half-widths `half` (`Region::combined`): its value and
/// gradient at the centre, its Taylor remainder's bound Σ_p |w_p| `curve`_p and its pairs' own ranges Σ_p |w_p| `radius`_p.
pub(crate) struct Combination {
    half: [f64; 3],
    value: C64,
    gradient: [C64; 3],
    curve: f64,
    each: f64,
}

impl Combination {
    pub(crate) fn of(half: &[f64; 3], terms: impl Iterator<Item = (PairSpan, C64, f64)>) -> Combination {
        let mut out = Combination { half: *half, value: C64::ZERO, gradient: [C64::ZERO; 3], curve: 0.0, each: 0.0 };
        for (pair, w, size) in terms {
            out.value += pair.centre * w;
            for (g, x) in out.gradient.iter_mut().zip(pair.gradient) {
                *g += x * w;
            }
            out.curve += size * pair.curve;
            out.each += size * pair.radius;
        }
        out
    }

    /// Re f in its own form (own.rs): its value and gradient at the centre and the Taylor remainder, within the pairs' own
    /// ranges -- a coarse cell's turn of the phases leaves those the tighter.
    pub(crate) fn real(&self) -> Own {
        let at = self.value.re;
        Own::new(at, self.gradient.map(|g| g.re), self.curve, self.half).within(at - self.each, at + self.each)
    }
}

/// V_prep(r; p) over a bounded cell in its own form (own.rs), its value at the centre `centre`: its far part Re Σ_l'l Q_l'l
/// c_l'l in its own form and range (`Region::combined`), and the near trap light's shift within each beam pair's radius
/// (`pair_span`) times its bound per pair (potential.rs `Model::envelope`).
fn prepared_own(envelope: &Envelope, spans: &[Span], centre: f64) -> Own {
    let n = envelope.beams.len();
    let pairs = || (0..n * n).map(|x| (x, pair_span(spans, envelope.beams[x / n], envelope.beams[x % n])));
    let far = Combination::of(&spans[envelope.beams[0]].half, pairs().filter(|&(x, _)| envelope.form[x] != C64::ZERO).map(|(x, pair)| (pair, envelope.form[x], envelope.form[x].abs()))).real();
    let near: f64 = pairs().filter(|&(x, _)| envelope.near[x] != 0.0).map(|(x, pair)| envelope.near[x] * pair.radius).sum();
    far.plus(centre - far.value).add(Own::range(0.0, -near, near))
}

/// A quantity on a cell: its value at the centre (−∞ on an unbounded cell, which has none) and its upper bound over the cell.
fn valued<Q: Fn(&Region) -> f64 + ?Sized>(q: &Q, cell: &Regions) -> (f64, f64) {
    (cell.at.as_ref().map_or(f64::NEG_INFINITY, |r| q(r)), q(&cell.over))
}

/// What ends a proof (`Model::proven_over`): `settled(the largest upper bound of the cells left, the largest value attained at a
/// centre in the reach)`, and the question's own size.
#[derive(Clone, Copy)]
struct Rule<'a> {
    settled: &'a (dyn Fn(f64, f64) -> bool + Sync),
    scale: f64,
    /// The statements of one unit, consecutive quantities `together` at a time, and the threshold one bounded at or above
    /// keeps the unit (none: each alone).
    together: Option<(usize, f64)>,
}

/// One quantity's proof (module documentation, # Provable extrema, # Rounds): the cells left, by their upper bounds, and the
/// largest value at a centre in the reach among them and their parents; the halves made for it, by cell, each as it enters the
/// queue, until its order comes to their cell; the largest value at a centre in the reach among all the halves made for it,
/// those waiting too; the cells it has asked to be halved; and its bound once it is decided.
struct Proof {
    queue: BinaryHeap<Queued>,
    attained: f64,
    halved: HashMap<usize, [Option<Queued>; 2]>,
    sighted: f64,
    asked: HashSet<usize>,
    bound: Option<f64>,
}

impl Proof {
    /// A quantity's proof at the proofs' first cell (none: no cell meets the reach, the bound is 0); on an unbounded cell, which
    /// is not halved, the bound is the quantity's over it.
    fn new<Q: Fn(&Region) -> f64 + ?Sized>(q: &Q, first: Option<&(Node, Regions)>) -> Proof {
        let mut proof = Proof { queue: BinaryHeap::new(), attained: f64::NEG_INFINITY, halved: HashMap::new(), sighted: f64::NEG_INFINITY, asked: HashSet::new(), bound: None };
        match first {
            None => proof.bound = Some(0.0),
            Some((node, regions)) => {
                let (centre, upper) = valued(q, regions);
                if node.cell.bounded() {
                    proof.push(Queued { upper, node: 0, centre, inside: node.inside });
                } else {
                    proof.bound = Some(upper);
                }
            }
        }
        proof
    }

    /// A cell into the queue, its centre's value attained where the centre is in the reach.
    fn push(&mut self, cell: Queued) {
        if cell.inside {
            self.attained = self.attained.max(cell.centre);
        }
        self.queue.push(cell);
    }

    /// The proof as far as the halves made for it reach -- with `made`, those the round before made --, and the cells it asks
    /// to be halved next, none once it is decided (module documentation, # Rounds): the cell its order has come to and, with
    /// `ahead`, each of the halves just made that is not settled against the largest value the halves made hold.
    fn advance(&mut self, made: Vec<(usize, [Option<Queued>; 2])>, rule: Rule, ahead: bool) -> Vec<usize> {
        let new: Vec<Queued> = made.iter().flat_map(|(_, halves)| halves.iter().flatten().copied()).collect();
        self.sighted = new.iter().filter(|half| half.inside).fold(self.sighted, |sighted, half| sighted.max(half.centre));
        self.halved.extend(made);
        while self.bound.is_none() {
            let Some(&top) = self.queue.peek() else {
                self.bound = Some(0.0);
                break;
            };
            if (rule.settled)(top.upper, self.attained) || top.upper - top.centre <= rounding(1) * rule.scale.max(top.centre.abs()) {
                self.bound = Some(top.upper);
                break;
            }
            let Some(halves) = self.halved.remove(&top.node) else {
                let unsettled = new.iter().filter(|half| ahead && !(rule.settled)(half.upper, self.sighted)).map(|half| half.node);
                return [top.node].into_iter().chain(unsettled).filter(|&cell| self.asked.insert(cell)).collect();
            };
            self.queue.pop();
            halves.into_iter().flatten().for_each(|half| self.push(half));
        }
        vec![]
    }
}

/// A cell that proofs ask to be halved in a round: its node, and the quantities asking.
struct Asked {
    node: usize,
    quantities: Vec<usize>,
}

/// A cell halved in a round: per side its halves where this round made them (none where a half cannot meet the reach), and
/// per quantity asking the side it halves the cell along (`side`), whether each half's centre is in the reach, and its values
/// on each half (`valued`).
struct Halved {
    made: [Option<[Option<Node>; 2]>; 3],
    of: Vec<(usize, [bool; 2], [Option<(f64, f64)>; 2])>,
}

/// The side a quantity's proof halves a cell along (module documentation, # A cell's sides), from the cell's regions along
/// each of its sides alone: the side along which the quantity's bound is the highest, of sides alike the longest.
fn side<Q: Fn(&Region) -> f64 + ?Sized>(q: &Q, cell: &Cell, along: &[Box<Region>; 3]) -> usize {
    let bound = along.each_ref().map(|region| q(region));
    (0..3).max_by(|&i, &j| bound[i].total_cmp(&bound[j]).then(cell.half[i].total_cmp(&cell.half[j]))).expect("a cell has three sides")
}

/// A round of the proofs (module documentation, # Rounds): what its cells are made from, the cells made before it, the quantities.
struct Round<'a, Q: ?Sized> {
    cells: &'a Cells<'a>,
    nodes: &'a [Node],
    quantities: &'a [&'a Q],
}

impl<Q: Fn(&Region) -> f64 + ?Sized> Round<'_, Q> {
    /// A cell halved for the quantities asking, each along its own side: the halves made where they are not yet, and those
    /// quantities' values on theirs -- each half's regions made once for them all, those of an earlier round's halves again.
    fn halve(&self, asked: &Asked) -> Halved {
        let (cells, node) = (self.cells, &self.nodes[asked.node]);
        let amplitudes = cells.model.light.amplitudes(node.cell.centre);
        let along: [Box<Region>; 3] = std::array::from_fn(|a| Box::new(cells.model.along(&amplitudes, node, a)));
        let sides: Vec<usize> = asked.quantities.iter().map(|&q| side(self.quantities[q], &node.cell, &along)).collect();
        let mut made: [Option<[Option<Node>; 2]>; 3] = [None, None, None];
        // Per side asked, each half that meets the reach: whether its centre is in the reach, and its regions.
        let halves: [Option<[Option<(bool, Regions)>; 2]>; 3] = std::array::from_fn(|a| {
            sides.contains(&a).then(|| match node.halves[a] {
                Some(halves) => halves.map(|half| half.map(|half| (self.nodes[half].inside, cells.regions(&self.nodes[half])))),
                None => {
                    let [(lower, first), (upper, second)] = node.cell.halves(a).map(|half| cells.make(half).map(|(half, regions)| ((half.inside, regions), half)).unzip());
                    made[a] = Some([first, second]);
                    [lower, upper]
                }
            })
        });
        let of = asked
            .quantities
            .iter()
            .zip(&sides)
            .map(|(&q, &a)| {
                let halves = halves[a].as_ref().expect("the halves of a side asked").each_ref().map(Option::as_ref);
                (a, halves.map(|half| half.is_some_and(|half| half.0)), halves.map(|half| half.map(|half| valued(self.quantities[q], &half.1))))
            })
            .collect();
        Halved { made, of }
    }
}

/// The pace of a proving (module documentation, # Rounds): whether its proofs ask for cells ahead of their order, and how a
/// round's independent pieces are gone through -- the proofs' own steps (each with what the round before made for it), the
/// halving of the cells asked for.
trait Pace<Q: ?Sized> {
    const AHEAD: bool;
    fn proofs<T: Send, R: Send>(proofs: &mut [Proof], made: Vec<T>, step: impl Fn(&mut Proof, T) -> R + Sync) -> Vec<R>;
    fn cells(round: &Round<Q>, asked: &[Asked]) -> Vec<Halved>;
}

/// One after another, no cell ahead of a proof's order: the pace of quantities that only their own thread may ask.
struct InTurn;

impl<Q: Fn(&Region) -> f64 + ?Sized> Pace<Q> for InTurn {
    const AHEAD: bool = false;
    fn proofs<T: Send, R: Send>(proofs: &mut [Proof], made: Vec<T>, step: impl Fn(&mut Proof, T) -> R + Sync) -> Vec<R> {
        proofs.iter_mut().zip(made).map(|(proof, made)| step(proof, made)).collect()
    }
    fn cells(round: &Round<Q>, asked: &[Asked]) -> Vec<Halved> {
        asked.iter().map(|cell| round.halve(cell)).collect()
    }
}

/// On all the cores at once, each proof asking ahead of its order: the pace of quantities that any thread may ask.
struct AtOnce;

impl<Q: Fn(&Region) -> f64 + Sync + ?Sized> Pace<Q> for AtOnce {
    const AHEAD: bool = true;
    fn proofs<T: Send, R: Send>(proofs: &mut [Proof], made: Vec<T>, step: impl Fn(&mut Proof, T) -> R + Sync) -> Vec<R> {
        proofs.par_iter_mut().zip(made).map(|(proof, made)| step(proof, made)).collect()
    }
    fn cells(round: &Round<Q>, asked: &[Asked]) -> Vec<Halved> {
        asked.par_iter().map(|cell| round.halve(cell)).collect()
    }
}

/// Whether a quantity's proof against a threshold is decided (`Model::bounds_over`), by the largest upper bound of the cells left
/// and the largest value attained at a centre in the reach: below the threshold everywhere, or attained at or above `attains`
/// (the threshold itself where the question has no band).
fn decided(threshold: f64, attains: f64) -> impl Fn(f64, f64) -> bool + Sync {
    move |upper, attained| upper < threshold || attained >= attains
}

/// P(Y + Σ_k a_k E_k ≤ level) at a level above zero: Y of Γ(`shape`) (nothing at shape zero), E_k unit exponentials of
/// the scales `scales` (module documentation, # The reach's level), by the convolution's recursion over the scales from
/// the last one in -- (1/a)∫₀^L e^{−(L−u)/a} P(rest ≤ u) du on the Gauss–Legendre nodes `nodes` of [−1, 1] in v, u = L v²
/// (a Γ of half-integer shape has its distribution entire in √u, not in u); with nothing left, Y's P(shape, u) (`lower`)
/// or one, and one exponential alone 1 − e^{−L/a}. Every part is positive: the distribution is read to its own relative
/// precision however small it is, as a hot trap's few atoms below a level ask (`Degrees::above`).
fn head_at(shape: f64, scales: &[f64], level: f64, nodes: &[(f64, f64)]) -> f64 {
    let Some((&a, rest)) = scales.split_last() else {
        return if shape > 0.0 { lower(shape, level) } else { 1.0 };
    };
    if shape == 0.0 && rest.is_empty() {
        return -(-level / a).exp_m1();
    }
    nodes.iter().map(|&(x, w)| { let v = 0.5 * (x + 1.0); let u = level * v * v; w * level * v / a * (-(level - u) / a).exp() * head_at(shape, rest, u, nodes) }).sum()
}

/// P(Y + Σ_k a_k E_k > level) at a level above zero (`head_at`'s complement), by the same recursion -- e^{−L/a} +
/// (1/a)∫₀^L e^{−(L−u)/a} P(rest > u) du in v, u = L v² --; with nothing left Y's Q(shape, u) or nothing.
fn tail_at(shape: f64, scales: &[f64], level: f64, nodes: &[(f64, f64)]) -> f64 {
    let Some((&a, rest)) = scales.split_last() else {
        return if shape > 0.0 { crate::special::upper(shape, level) } else { 0.0 };
    };
    let within: f64 = if shape > 0.0 || !rest.is_empty() {
        nodes.iter().map(|&(x, w)| { let v = 0.5 * (x + 1.0); let u = level * v * v; w * level * v / a * (-(level - u) / a).exp() * tail_at(shape, rest, u, nodes) }).sum()
    } else {
        0.0
    };
    (-level / a).exp() + within
}

/// P(s, x), the regularized lower incomplete gamma function at a whole or half-integer s: the lowered isothermal's momenta
/// below its escape at the depth x (ensemble.rs `kept`, whose series takes the leading power exactly) and the power it
/// subtracts.
fn lower(s: f64, x: f64) -> f64 {
    crate::ensemble::kept(s, x, x, x) + (s * x.ln() - x).exp() / crate::special::gamma(s + 1.0)
}

/// The Gauss–Legendre rules of 2, 4, 8, … nodes (`below`), each made where it is first asked: a quantile's bisection
/// asks the same few rules at every level.
#[derive(Default)]
struct Rules(Vec<Vec<(f64, f64)>>);

impl Rules {
    /// The rule of two nodes doubled `times` times.
    fn doubled(&mut self, times: usize) -> &[(f64, f64)] {
        while self.0.len() <= times {
            self.0.push(crate::special::gauss_legendre(2 << self.0.len()));
        }
        &self.0[times]
    }

    /// A recursion's value (`head_at`, `tail_at`) on rules of nodes doubled from two until two levels' difference stops
    /// falling: the integrands are entire, the rule converges geometrically, and a difference no smaller than the one
    /// before is the round-off's floor (or nothing).
    fn converged(&mut self, at: impl Fn(&[(f64, f64)]) -> f64) -> f64 {
        let mut last = at(self.doubled(0));
        let mut before = f64::INFINITY;
        for times in 1.. {
            let next = at(self.doubled(times));
            let difference = (next - last).abs();
            if !(difference < before) {
                break;
            }
            (last, before) = (next, difference);
        }
        last
    }
}

/// P(Y + Σ_k a_k E_k ≤ level), as the incomplete gamma function's two expansions are taken (special.rs), each where it
/// converges: up to the mean f/2 + Σ a_k its own recursion (`head_at`), whose exponential spans its interval and whose
/// small values keep their own precision; above it one less the tail's (`tail_at`), whose values are small there -- the
/// distribution's recursion at a level of many scales has its weight in the last 1/L of its interval, which the first
/// rules do not see.
fn below(shape: f64, scales: &[f64], level: f64, rules: &mut Rules) -> f64 {
    if level <= shape + scales.iter().sum::<f64>() {
        rules.converged(|nodes| head_at(shape, scales, level, nodes))
    } else {
        1.0 - rules.converged(|nodes| tail_at(shape, scales, level, nodes))
    }
}

/// The drawn energy's degrees in units of k_BT (module documentation, # The reach's level): Y's shape f/2, the held axes'
/// scales s_i² ascending, and the depth d = D/k_BT.
struct Degrees {
    shape: f64,
    scales: Vec<f64>,
    depth: f64,
}

impl Degrees {
    /// k: half the classical draw's quadratic degrees, f/2 and one a held axis.
    fn half(&self) -> f64 {
        self.shape + self.scales.len() as f64
    }

    /// The least scale a₀ -- one where an axis is free.
    fn least(&self) -> f64 {
        if self.shape > 0.0 { 1.0 } else { self.scales[0] }
    }

    /// The largest scale.
    fn most(&self) -> f64 {
        self.scales.last().copied().unwrap_or(1.0).max(1.0)
    }

    /// P(X > L) of the lowered isothermal at a level L ≤ a₀ d (module documentation, # The reach's level): 1 − [P(X ≤ L) −
    /// A L^k e^{−d}/Γ(k + 1)]/P(k + 1, d), A = Π_i s_i^{−2}, of the thermal draw's distribution (`below`).
    fn above(&self, level: f64, rules: &mut Rules) -> f64 {
        let k = self.half();
        let cone: f64 = self.scales.iter().map(|a| a.ln()).sum();
        let cut = (k * level.ln() - self.depth - cone).exp() / crate::special::gamma(k + 1.0);
        1.0 - (below(self.shape, &self.scales, level, rules) - cut) / lower(k + 1.0, self.depth)
    }

    /// The level (in units of k_BT) that the share `share` of the lowered isothermal lies above, 0 < share < 1 (module
    /// documentation, # The reach's level): its tail falls monotonically from one at nothing to the least scale's depth
    /// a₀ d, and the bracket between them is halved until its two ends are neighbours in the format; where the zero point
    /// lifts more than the share above a₀ d, the largest scale's depth, which every atom of the harmonic form is below.
    fn level(&self, share: f64) -> f64 {
        let mut rules = Rules::default();
        let mut above = |level: f64| self.above(level, &mut rules) > share;
        let (mut lo, mut hi) = (0.0, self.least() * self.depth);
        if above(hi) {
            return self.most() * self.depth;
        }
        loop {
            let middle = 0.5 * (lo + hi);
            if !(lo < middle && middle < hi) {
                return hi;
            }
            if above(middle) { lo = middle } else { hi = middle }
        }
    }
}

/// The drawn energy's degrees of a trap at temperature T_r (µK) (`Degrees`), and the squares s_i² per axis: per principal
/// axis along which the trap holds the atom an exponential scaled by the Wigner factor's s_i² = x_i coth x_i, x_i =
/// ħω_i/2k_BT_r, per axis it does not hold the momentum's degree alone; its depth −V_min over k_BT_r.
fn degrees(trap: &Trap, temperature_uk: f64) -> (Degrees, [f64; 3]) {
    let kt = KT_PER_UK * temperature_uk;
    let squares = trap.frequencies.map(|omega| { let x = 0.5 * omega / kt; if x > 0.0 { x / x.tanh() } else { 1.0 } });
    let mut scales: Vec<f64> = (0..3).filter(|&i| trap.frequencies[i] > 0.0).map(|i| squares[i]).collect();
    scales.sort_by(f64::total_cmp);
    (Degrees { shape: 0.5 * (3 - scales.len()) as f64, scales, depth: -trap.minimum / kt }, squares)
}

/// The energy above V_min (rad/µs) of the reach of a trap at temperature T_r (µK) that leaves out the share s (module
/// documentation, # The reach's level) -- the (1 − s) quantile of an atom's energy in the lowered isothermal as drawn at
/// T_r (`Degrees::level`) --, and the squares s_i² per axis; at T_r = ∞ its limit, the top (`top`), every s_i² one.
fn energy_at(trap: &Trap, temperature_uk: f64, share: f64) -> (f64, [f64; 3]) {
    if temperature_uk.is_infinite() {
        return (top(trap, share), [1.0; 3]);
    }
    let (degrees, squares) = degrees(trap, temperature_uk);
    (degrees.level(share) * KT_PER_UK * temperature_uk, squares)
}

/// The energy above V_min (rad/µs) that the reach's level rises to with its temperature (module documentation, # The
/// reach's level): at T_r → ∞ every s_i → 1 and the energy over the depth D has the density ∝ u^{k−1}(1 − u) on [0, 1],
/// whose share above q is 1 − (k + 1)q^k + k q^{k+1}, falling from one to nothing -- the q where it is s, by bisection to
/// the format's resolution, times D.
fn top(trap: &Trap, share: f64) -> f64 {
    let k = degrees(trap, 1.0).0.half();
    let above = |q: f64| 1.0 - (k + 1.0) * q.powf(k) + k * q.powf(k + 1.0) > share;
    let (mut lo, mut hi) = (0.0_f64, 1.0_f64);
    loop {
        let middle = 0.5 * (lo + hi);
        if !(lo < middle && middle < hi) {
            return -trap.minimum * hi;
        }
        if above(middle) { lo = middle } else { hi = middle }
    }
}

/// The least temperature (µK) at which the reach of a trap that leaves out the share s holds the level `level` (rad/µs,
/// the prepared energy; module documentation, # The reach's level), from a temperature `from` whose reach does not: the
/// reach's energy rises with its temperature towards its top (`top`), its limit at T_r → ∞, so on the inverse temperature
/// the bracket is [0, 1/from] -- 0 the top's, which holds every level below it -- and it is halved until its two ends are
/// neighbours in the format. None (∞) where the level is at the top or above it: no temperature's reach holds it -- an
/// atom at the trap's top or above its depth, leaving or lost, is no atom of a temperature's ensemble.
pub fn holding(trap: &Trap, level: f64, share: f64, from: f64) -> f64 {
    if level - trap.minimum >= top(trap, share) {
        return f64::INFINITY;
    }
    let holds = |beta: f64| trap.minimum + energy_at(trap, 1.0 / beta, share).0 >= level;
    let (mut lo, mut hi) = (0.0, 1.0 / from);
    if holds(hi) {
        return from;
    }
    loop {
        let middle = 0.5 * (lo + hi);
        if !(lo < middle && middle < hi) {
            return 1.0 / lo;
        }
        if holds(middle) { lo = middle } else { hi = middle }
    }
}

impl Model {
    /// The reach of a trap at temperature T_r (µK) that leaves out the share s (module documentation, # The reach's
    /// level, `energy_at`): the sublevel set of its level, where that is below the far value; else the set within the
    /// trap's region, and none outside it (# The trap's region; `outside_to` adds a level there).
    pub fn reach(&self, trap: &Trap, p: &[f64; GROUND], temperature_uk: f64, share: f64) -> Reach {
        let (energy, squares) = energy_at(trap, temperature_uk, share);
        let outside = if trap.minimum + energy < 0.0 { energy } else { f64::NEG_INFINITY };
        let mut reach = Reach {
            temperature_uk,
            kinetic_uk: squares.map(|square| square * temperature_uk),
            energy,
            outside,
            minimum: trap.minimum,
            cell: [[0.0; 2]; 3],
            speed: (2.0 * self.hbar_over_mass * energy).sqrt(),
            populations: *p,
        };
        reach.cell = self.reach_box(&reach);
        reach
    }

    /// The reach that holds, beside `reach`, the prepared energies up to `level` (rad/µs) within the trap's region (module
    /// documentation, # The trap's region): its level inside raised to it, its speeds with it; its temperature and its
    /// level outside kept. The same reach where it holds the level already.
    pub fn inside_to(&self, reach: &Reach, level: f64) -> Reach {
        if level - reach.minimum <= reach.energy {
            return reach.clone();
        }
        let energy = level - reach.minimum;
        let mut wider = Reach { energy, speed: (2.0 * self.hbar_over_mass * energy).sqrt(), ..reach.clone() };
        wider.cell = self.reach_box(&wider);
        wider
    }

    /// The reach that holds, beside `reach`, the prepared energies up to `level` (rad/µs) outside the trap's region (module
    /// documentation, # The trap's region): its outside level raised to it, no higher than its own level inside.
    pub fn outside_to(&self, reach: &Reach, level: f64) -> Reach {
        let mut wider = Reach { outside: reach.outside.max((level - reach.minimum).min(reach.energy)), ..reach.clone() };
        wider.cell = self.reach_box(&wider);
        wider
    }

    /// A box that holds a reach (`Reach::cell`; module documentation, # The reach, # The trap's region): the envelope's
    /// ellipsoids of its level where that is below the far value; else the trap light's ellipsoids of the region with
    /// the envelope's of the outside level. Unbounded where a level that bounds no set is asked, or a trap beam is a plane
    /// wave.
    fn reach_box(&self, reach: &Reach) -> [[f64; 2]; 3] {
        let envelope = self.envelope(&reach.populations);
        let (b, n) = (envelope.peak(), envelope.beams.len());
        // The sublevel set's extents at a level: of the envelope's beams, u = B/(−level) − 1 (none below every level).
        let below = |level: f64| -> [f64; 3] {
            let mut extent = [0.0_f64; 3];
            for (j, &l) in envelope.beams.iter().enumerate() {
                let weight: f64 = (0..n).map(|i| envelope.weights[j * n + i] + envelope.weights[i * n + j]).sum();
                if weight == 0.0 {
                    continue;
                }
                let beam = &self.light.beams[l];
                match (beam.waist, beam.rayleigh()) {
                    (Some(w), Some(z)) if level < 0.0 => {
                        let u = b / -level - 1.0;
                        for a in 0..3 {
                            let na = beam.direction[a];
                            extent[a] = extent[a].max((u.max(0.0) * (0.5 * w * w * (1.0 - na * na) + z * z * na * na)).sqrt());
                        }
                    }
                    _ => extent = [f64::INFINITY; 3],
                }
            }
            extent
        };
        let extent = if reach.outside == reach.energy {
            below(reach.minimum + reach.energy)
        } else {
            // The region: of every beam of the trap light, u ≤ 1/REGION_INTENSITY − 1.
            let (u, mut extent) = (1.0 / REGION_INTENSITY - 1.0, below(reach.minimum + reach.outside));
            for &l in &self.light.frequencies[self.light.trap].beams {
                let beam = &self.light.beams[l];
                match (beam.waist, beam.rayleigh()) {
                    (Some(w), Some(z)) => {
                        for a in 0..3 {
                            let na = beam.direction[a];
                            extent[a] = extent[a].max((u * (0.5 * w * w * (1.0 - na * na) + z * z * na * na)).sqrt());
                        }
                    }
                    _ => extent = [f64::INFINITY; 3],
                }
            }
            extent
        };
        std::array::from_fn(|a| [-extent[a], extent[a]])
    }

    /// Of a cell's beams (`Span`): whether the cell may meet the trap's region -- some beam of the trap light reaches
    /// `REGION_INTENSITY` at its largest over the cell -- and whether it lies within it -- one beam does at its least
    /// (field.rs, # The region).
    pub(crate) fn region_of(&self, spans: &[Span]) -> [bool; 2] {
        let beams = &self.light.frequencies[self.light.trap].beams;
        [beams.iter().any(|&l| spans[l].most * spans[l].most >= REGION_INTENSITY), beams.iter().any(|&l| spans[l].least * spans[l].least >= REGION_INTENSITY)]
    }

    /// Whether the model holds along a straight path r + v u, |u| ≤ |half|, at the speed `speed` (step.rs, # The domain;
    /// solve.rs `Domain`): every one of the units `units` has its fold (units.rs `Model::exist_over`) -- by the reach's
    /// proofs' own rule along the path (module documentation, # Provable extrema): a piece of the path is settled where
    /// every bound over its box is below ½, and the path fails where a piece's middle, a point of the path, attains ½;
    /// else the piece is halved, each half asked only what its whole left open. A bound over the path's whole box moves
    /// with the box: a step of 3.3 µs at 0.17 µm/µs spans half a waist of a tweezer, its trap's shift of a probe over it
    /// reads a fold at 0.78 that no point of the path has, and reaches grown for such boxes alone crept up in seven
    /// attempts.
    pub(crate) fn holds_along(&self, judged: &super::units::Judged, units: &[usize], r: [f64; 3], v: [f64; 3], half: f64, speed: f64) -> bool {
        let at = |u: f64| -> [f64; 3] { std::array::from_fn(|a| r[a] + v[a] * u) };
        // The units a region leaves open.
        let open = |region: &Region, asked: &[usize]| -> Vec<usize> { asked.iter().copied().filter(|&u| !self.exist_over(judged, &[u], region)).collect() };
        let mut pieces = vec![(-half.abs(), half.abs(), units.to_vec())];
        while let Some((a, b, asked)) = pieces.pop() {
            let (middle, width) = (0.5 * (a + b), 0.5 * (b - a));
            let over = self.region(self.spans(&Cell { centre: at(middle), half: v.map(|x| (x * width).abs()) }), speed);
            let left = open(&over, &asked);
            if left.is_empty() {
                continue;
            }
            let point = self.point_region(at(middle), speed);
            if !open(&point, &left).is_empty() || !(a < middle && middle < b) {
                return false;
            }
            pieces.push((a, middle, left.clone()));
            pieces.push((middle, b, left));
        }
        true
    }

    /// The level of a reach at r (rad/µs): the reach's own; where it holds only a lower level outside the trap's region
    /// (module documentation, # The trap's region) and r lies outside the region, that one.
    fn level_at(&self, reach: &Reach, r: [f64; 3]) -> f64 {
        reach.minimum + if reach.outside < reach.energy && self.light.outside(r) { reach.outside } else { reach.energy }
    }

    /// The kinetic energy a reach holds at r (rad/µs; module documentation, # The reach), under the prepared lights
    /// `prepared`: A(r) = V_min + E_r − V_prep(r; p), the reach's level at r (`level_at`) -- an atom at r is in the reach
    /// with (ħ/m) κ²/2 ≤ A(r), none where A is negative.
    pub fn room(&self, reach: &Reach, r: [f64; 3], prepared: &Lights) -> f64 {
        self.level_at(reach, r) - self.potential(r, &reach.populations, prepared)
    }

    /// Whether a reach holds an atom at r with the momentum κ: its kinetic energy (ħ/m) κ²/2 within the reach's `room`
    /// there.
    pub fn holds(&self, reach: &Reach, r: [f64; 3], kappa: [f64; 3]) -> bool {
        0.5 * self.hbar_over_mass * dot3(kappa, kappa) <= self.room(reach, r, &self.prepared())
    }

    /// What a reach leaves out of an initial ensemble (module documentation, # What the reach leaves out): the share of
    /// its atoms and the share of their kinetic energy beyond the reach, by the ensemble's quadrature of its own density
    /// with at most `most` evaluations of a component's potential a level (ensemble.rs `Ensemble::beyond`), each with
    /// the quadrature's difference between its last two levels added: the share as far as it is known.
    pub fn left_out(&self, reach: &Reach, ensemble: &crate::ensemble::Ensemble, most: u64) -> [f64; 2] {
        let prepared = self.prepared();
        ensemble.beyond(&|r| self.room(reach, r, &prepared), most).map(|share| share.value + share.difference)
    }

    /// The beams' largest moduli over a reach's box (`Span::most`, per request beam): what the kick's curvature over the
    /// reach takes the fringes' beams at (`kick_curvature`).
    pub fn moduli(&self, reach: &Reach) -> Vec<f64> {
        self.spans(&reach.whole()).iter().map(|span| span.most).collect()
    }

    /// What the model bounds of the curvature of the kick's potential, (ħ/m) λ_max ∂∂(Ĥ_h − V_ref) of any state held in
    /// its cluster's frames along any direction (rad²/µs²; module documentation, # The kick's curvature), every light
    /// on, the beams' moduli at most `most`: the largest over the clusters' states of how much stiffer than the
    /// reference's (of the reach's populations) the state's trap is at the trap's centre (`stiffer`) with the state's
    /// row of the constant terms' fringes (assemble.rs `Plan::fringes`) -- over the reach's box with its `moduli`, over a
    /// step's path outside it with the larger of those and the path's (solve.rs `Domain`). Negative where every state's
    /// well is softer than the reference's and no such light makes a fringe.
    pub fn kick_curvature(&self, reach: &Reach, most: &[f64], trap: &Trap) -> f64 {
        let stiffer = self.stiffer(trap.centre, &reach.populations);
        let mut largest = f64::NEG_INFINITY;
        for (c, cluster) in self.clusters.iter().enumerate() {
            let rows = self.plan(c, &self.running()).fringes(self, most);
            largest = cluster.states.iter().zip(&rows).fold(largest, |most, (&s, row)| most.max(stiffer[s] + row));
        }
        self.hbar_over_mass * largest
    }

    /// Every beam over a cell (`Span`).
    pub(crate) fn spans(&self, cell: &Cell) -> Vec<Span> {
        if !cell.bounded() {
            let nothing = Amplitude { value: C64::ZERO, gradient: [C64::ZERO; 3], hessian: [[C64::ZERO; 3]; 3] };
            return self.light.beams.iter().map(|beam| Span::over(beam, &nothing, cell)).collect();
        }
        let amplitudes = self.light.amplitudes(cell.centre);
        self.light.beams.iter().zip(&amplitudes).map(|(beam, a)| Span::over(beam, a, cell)).collect()
    }

    /// A bounded cell's region along its side `a` alone, from the beams' amplitudes at its centre: the same centre and speeds,
    /// its two other half-widths nothing (module documentation, # A cell's sides).
    fn along(&self, amplitudes: &[Amplitude], node: &Node, a: usize) -> Region {
        let mut side = Cell { centre: node.cell.centre, half: [0.0; 3] };
        side.half[a] = node.cell.half[a];
        self.region(self.light.beams.iter().zip(amplitudes).map(|(beam, at)| Span::over(beam, at, &side)).collect(), node.over)
    }

    /// A region of the beams `spans` at speeds up to `speed` (`Region`): its beam pairs made, the rest when asked.
    pub(crate) fn region(&self, spans: Vec<Span>, speed: f64) -> Region {
        self.region_paced(spans, speed, Own::constant(speed))
    }

    /// A region of the beams `spans` at speeds up to `speed`, their own form `pace` (`Region`).
    fn region_paced(&self, spans: Vec<Span>, speed: f64, pace: Own) -> Region {
        let pairs = self
            .light
            .frequencies
            .iter()
            .map(|f| f.beams.iter().flat_map(|&j| f.beams.iter().map(move |&i| (j, i))).map(|(j, i)| pair_span(&spans, j, i)).collect())
            .collect();
        Region {
            pairs,
            half: spans.first().map_or([0.0; 3], |s| s.half),
            rho: spans.first().map_or(0.0, |s| s.rho),
            speed,
            pace,
            spans,
            forms: self.forms.clone(),
            radii: OnceLock::new(),
            couplings: OnceLock::new(),
            resolvents: OnceLock::new(),
            dressings: OnceLock::new(),
            shifted: OnceLock::new(),
            forces: (0..self.clusters.len()).map(|_| OnceLock::new()).collect(),
            across: std::array::from_fn(|_| OnceLock::new()),
            tails: std::array::from_fn(|_| OnceLock::new()),
            folds: std::array::from_fn(|_| Mutex::new(vec![])),
            shifts: std::array::from_fn(|_| OnceLock::new()),
            diagonal: OnceLock::new(),
            gaps: Mutex::new(HashMap::new()),
            slopes: Mutex::new(HashMap::new()),
            terms: Mutex::new(HashMap::new()),
            kept: OnceLock::new(),
            moving: OnceLock::new(),
        }
    }

    /// The region of a point alone at one speed (a sample's: the errors measured there).
    pub(crate) fn point_region(&self, r: [f64; 3], speed: f64) -> Region {
        self.region(self.spans(&Cell { centre: r, half: [0.0; 3] }), speed)
    }

    /// Bounded cells' regions as a reach's proofs make them (`Cells::make`): over each cell at the reach's speeds there, and
    /// on its centre point at the reach's speed there; none where a cell cannot meet the reach.
    #[cfg(test)]
    pub(crate) fn proof_regions(&self, reach: &Reach, cells: &[Cell]) -> Vec<Option<(Region, Region)>> {
        let made = Cells { model: self, reach, prepared: self.prepared(), envelope: self.envelope(&reach.populations) };
        cells.iter().map(|&cell| made.make(cell).map(|(_, regions)| (*regions.over, *regions.at.expect("a bounded cell has its centre's region")))).collect()
    }

    /// The provable bounds over the reach of quantities given per region, each asked against `threshold` from the band's lower
    /// edge `attains` (module documentation, # Provable extrema): each one's largest upper bound of the cells left when it is
    /// decided -- below the threshold everywhere, or a value at or above `attains` at a centre in the reach --, or within the
    /// rounding of its centre's value; 0 where no cell meets the reach. The cells are halved one after another (`InTurn`).
    pub(crate) fn bounds_over(&self, reach: &Reach, threshold: f64, attains: f64, quantities: &[&dyn Fn(&Region) -> f64]) -> Vec<f64> {
        self.proven_over::<_, InTurn>(reach, Rule { settled: &decided(threshold, attains), scale: threshold.abs(), together: None }, quantities)
    }

    /// The bounds against `threshold` and `attains` (`bounds_over`) of quantities any thread may ask, proven on all the cores
    /// (`AtOnce`): the same bounds, to the bit.
    #[cfg(test)]
    pub(crate) fn bounds_over_in_parallel(&self, reach: &Reach, threshold: f64, attains: f64, quantities: &[&(dyn Fn(&Region) -> f64 + Sync)]) -> Vec<f64> {
        self.proven_over::<_, AtOnce>(reach, Rule { settled: &decided(threshold, attains), scale: threshold.abs(), together: None }, quantities)
    }

    /// The bounds of units' statements (`bounds_over_in_parallel`), consecutive quantities `together` at a time one unit's
    /// (module documentation, # Rounds): a unit one of whose statements is proven at the threshold or above is kept whatever
    /// its others read, and their proofs end there, each at the largest bound of its cells left -- a bound, if not the least.
    pub(crate) fn statements_over(&self, reach: &Reach, threshold: f64, attains: f64, together: usize, quantities: &[&(dyn Fn(&Region) -> f64 + Sync)]) -> Vec<f64> {
        self.proven_over::<_, AtOnce>(reach, Rule { settled: &decided(threshold, attains), scale: threshold.abs(), together: Some((together, threshold)) }, quantities)
    }

    /// The provable bounds over the reach of quantities given per region, each to `resolution`: each one's largest upper bound of
    /// the cells left once it is within the resolution of a value attained at a centre in the reach. Of quantities any
    /// thread may ask, proven on all the cores (`AtOnce`): an extent's cells are those of the whole place of its largest
    /// values, each to be halved whatever the order.
    pub(crate) fn extents_over(&self, reach: &Reach, resolution: f64, quantities: &[&(dyn Fn(&Region) -> f64 + Sync)]) -> Vec<f64> {
        self.proven_over::<_, AtOnce>(reach, Rule { settled: &|upper, attained| upper - attained <= resolution, scale: resolution, together: None }, quantities)
    }

    /// The proofs' subdivision (module documentation, # Provable extrema, # Rounds): per quantity the cell of its largest upper
    /// bound is halved, along the side that holds that bound up the most (# A cell's sides), until the bound is settled against
    /// the largest value attained at a centre in the reach so far, the cell is unbounded, or the bound is within the rounding
    /// of its centre's value (of the question's own size, where that is larger: `rule`). The proofs go on in rounds, at the
    /// pace `P`.
    fn proven_over<Q: Fn(&Region) -> f64 + ?Sized, P: Pace<Q>>(&self, reach: &Reach, rule: Rule, quantities: &[&Q]) -> Vec<f64> {
        let cells = Cells { model: self, reach, prepared: self.prepared(), envelope: self.envelope(&reach.populations) };
        let first = cells.make(reach.whole());
        let mut proofs: Vec<Proof> = quantities.iter().map(|q| Proof::new(*q, first.as_ref())).collect();
        let mut nodes: Vec<Node> = first.into_iter().map(|(node, _)| node).collect();
        // Per proof, the halves the round before made for it.
        let nothing = || -> Vec<Vec<(usize, [Option<Queued>; 2])>> { quantities.iter().map(|_| vec![]).collect() };
        let mut made = nothing();
        loop {
            let mut asking = P::proofs(&mut proofs, made, |proof, made| proof.advance(made, rule, P::AHEAD));
            // A unit kept by one statement: its other statements' proofs end at the bounds of their cells left.
            if let Some((together, threshold)) = rule.together {
                for (unit, asks) in proofs.chunks_mut(together).zip(asking.chunks_mut(together)) {
                    if unit.iter().any(|proof| proof.bound.is_some_and(|bound| bound >= threshold)) {
                        for (proof, asked) in unit.iter_mut().zip(asks).filter(|of| of.0.bound.is_none()) {
                            proof.bound = Some(proof.queue.peek().map_or(0.0, |top| top.upper));
                            asked.clear();
                        }
                    }
                }
            }
            // The cells asked for, each with the quantities asking.
            let mut by_cell: Vec<(usize, usize)> = asking.iter().enumerate().flat_map(|(q, cells)| cells.iter().map(move |&node| (node, q))).collect();
            if by_cell.is_empty() {
                break;
            }
            by_cell.sort_unstable();
            let mut asked: Vec<Asked> = vec![];
            for (node, q) in by_cell {
                match asked.last_mut() {
                    Some(cell) if cell.node == node => cell.quantities.push(q),
                    _ => asked.push(Asked { node, quantities: vec![q] }),
                }
            }
            let halved = P::cells(&Round { cells: &cells, nodes: &nodes, quantities }, &asked);
            made = nothing();
            for (cell, halved) in asked.iter().zip(halved) {
                for (a, new) in halved.made.into_iter().enumerate() {
                    if let Some(new) = new {
                        let at = new.map(|half| {
                            half.map(|half| {
                                nodes.push(half);
                                nodes.len() - 1
                            })
                        });
                        nodes[cell.node].halves[a] = Some(at);
                    }
                }
                for (&q, (a, inside, values)) in cell.quantities.iter().zip(halved.of) {
                    let halves = nodes[cell.node].halves[a].expect("the halves a quantity asked for are made");
                    let queued = std::array::from_fn(|x| halves[x].zip(values[x]).map(|(node, (centre, upper))| Queued { upper, node, centre, inside: inside[x] }));
                    made[q].push((cell.node, queued));
                }
            }
        }
        proofs.into_iter().map(|proof| proof.bound.expect("a proof left open asks for a cell")).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::Light;
    use crate::model::far::{pair, EXCITED};
    use crate::model::{Component, Unit};
    use crate::request::{Request, STATES};

    /// A reference scene's model with every drive of every co-rotating component but the trap's folded with the levels at
    /// the atom.
    fn at_the_atom(scene: &str) -> Model {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{scene}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let bare = Model::new(&request, &light, &[]).unwrap();
        let kept: Vec<Unit> = (0..light.frequencies.len()).filter(|&f| f != light.trap).flat_map(|frequency| bare.bare_units(Component { frequency, counter: false })).collect();
        Model::new(&request, &light, &kept).unwrap()
    }

    /// R8_R5051: a six-beam lattice of cooling light with its repump, in a tweezer.
    fn lattice() -> Model {
        at_the_atom("R8_R5051")
    }

    /// Points spread through a cell: its centre and a 5³ grid on and inside its faces.
    fn points(cell: &Cell) -> Vec<[f64; 3]> {
        let mut out = vec![cell.centre];
        for i in 0..5 {
            for j in 0..5 {
                for k in 0..5 {
                    let u = [i, j, k].map(|x| x as f64 / 2.0 - 1.0);
                    out.push(std::array::from_fn(|a| cell.centre[a] + u[a] * cell.half[a]));
                }
            }
        }
        out
    }

    /// The thermal draw's level (module documentation, # The reach's level), its ensemble not lowered: of a depth beyond
    /// every level asked, where e^{−d} is nothing and P(k + 1, d) one.
    fn thermal(shape: f64, scales: &[f64], share: f64) -> f64 {
        Degrees { shape, scales: scales.to_vec(), depth: 1e3 }.level(share)
    }

    /// The reach's level is the quantile of its ensemble's energy (module documentation, # The reach's level). Of the thermal
    /// draw, classical (every scale one): Q(n/2, L) = s to the format's resolution for the n = 3 to 6 degrees of a trap that
    /// holds along none to three of its axes; one oscillator's is ln(1/s); three axes' 12.89 at s = z²/(2¹⁴ + z²), where
    /// ln(1/s) = 8.32 left Q(3, 8.32) = 1.07 % above it; scales equal to nine digits are the same level. With the Wigner
    /// factor (two axes of one scale a, one of b): P(X > L) = e^{−L/a}(1 + L/a) + e^{−L/b}(1 − e^{−cL}(1 + cL))/(a c)²,
    /// c = 1/a − 1/b, is s at the level -- at s² = 1.66, 1.66, 1.04 and at R8_R5051's 7.2, 7.2, 30.6 (271 k_BT). On
    /// R2_RECOIL's tweezer at 30 µK the reach's level is that of its own three scales and its depth, a trap free along an
    /// axis has five degrees, and the ensemble's kinetic temperatures are s_i² T.
    #[test]
    fn the_reach_s_level_is_the_quantile_of_its_trap_s_degrees_of_freedom() {
        let share = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        for held in 0..=3 {
            let (shape, scales) = (0.5 * (3 - held) as f64, vec![1.0; held]);
            let level = thermal(shape, &scales, share);
            let above = crate::special::upper(shape + held as f64, level);
            assert!((above - share).abs() <= 1e-11 * share, "{held} axes held: Q({}, {level}) = {above} against s = {share}", shape + held as f64);
        }
        assert!((thermal(0.0, &[1.0], share) - (1.0 / share).ln()).abs() <= 1e-12 * (1.0 / share).ln(), "one oscillator's level is ln(1/s)");
        let classical = thermal(0.0, &[1.0; 3], share);
        assert!((classical - 12.89).abs() < 0.005 && (crate::special::upper(3.0, (1.0 / share).ln()) - 0.01072).abs() < 1e-5, "six degrees: {classical}");
        assert!((thermal(0.0, &[1.0, 1.0 + 1e-9, 1.0 + 2e-9], share) - classical).abs() <= 1e-8 * classical, "scales equal to nine digits");
        for (a, b, expected) in [(1.657, 1.041, None), (7.2, 30.6, Some(271.0))] {
            let level = thermal(0.0, &if a < b { [a, a, b] } else { [b, a, a] }, share);
            let c = 1.0 / a - 1.0 / b;
            let closed = (-level / a).exp() * (1.0 + level / a) + (-level / b).exp() * (1.0 - (-c * level).exp() * (1.0 + c * level)) / (a * c).powi(2);
            assert!((closed - share).abs() <= 1e-10 * share && expected.is_none_or(|l: f64| (level - l).abs() < 1.0), "scales {a}, {a}, {b}: the closed form's {closed} at L = {level} against s = {share}");
        }
        let model = at_the_atom("R2_RECOIL");
        let p = { let mut p = [0.0; GROUND]; p[7] = 1.0; p };
        let trap = model.trap(&p, &model.prepared()).unwrap();
        assert!(trap.frequencies.iter().all(|omega| *omega > 0.0), "the tweezer holds along its three axes");
        let kt = KT_PER_UK * 30.0;
        let held = model.reach(&trap, &p, 30.0, share);
        let free = model.reach(&Trap { frequencies: [trap.frequencies[0], trap.frequencies[1], 0.0], ..trap.clone() }, &p, 30.0, share);
        let squares = trap.frequencies.map(|omega| { let x = 0.5 * omega / kt; x / x.tanh() });
        let mut scales = squares.to_vec();
        scales.sort_by(f64::total_cmp);
        assert!(squares.iter().all(|s| *s > 1.0 && *s < 1.01) && held.kinetic_uk == squares.map(|s| s * 30.0), "a tweezer at 30 µK is nearly classical: s² = {squares:?}");
        let across: Vec<f64> = { let mut two = vec![squares[0], squares[1]]; two.sort_by(f64::total_cmp); two };
        let depth = -trap.minimum / kt;
        let (own, five) = (Degrees { shape: 0.0, scales, depth }.level(share), Degrees { shape: 0.5, scales: across, depth }.level(share));
        assert!((held.energy - own * kt).abs() <= 1e-12 * held.energy && (free.energy - five * kt).abs() <= 1e-12 * free.energy, "the degrees are the trap's: {} and {}", held.energy / kt, free.energy / kt);
        assert!(held.energy / kt > classical && held.energy / kt < 1.01 * classical && free.energy < held.energy, "the levels: {} and {} k_BT against the classical {classical} (depth {depth} k_BT)", held.energy / kt, free.energy / kt);
        assert!((held.speed / (2.0 * model.hbar_over_mass * (1.0 / share).ln() * kt).sqrt() - 1.245).abs() < 0.005, "the reach's speed, 1.245 of the Boltzmann factor's level's");
    }

    /// The reach's level is the quantile of the lowered isothermal, the initial ensemble's bound atoms alone, below the depth
    /// (module documentation, # The reach's level). A harmonic well of depth D = 1.5 mK, a tweezer's frequencies (100 kHz
    /// across, 15 kHz along: s² up to 1.002 at 30 µK), from 30 µK to 1 mK: the reach's level is below D, rises with T_r
    /// and at T_r → ∞ to its top (the Beta(3, 2) quantile, 0.9936 D), and is the level at which the share s of the
    /// lowered isothermal lies above -- that share integrated directly over the classical draw's energies, the radial
    /// two's sum r (of the measure r dr) and the axial one's e, of the weight (e^{−r−e} − e^{−d})₊ where a r + b e > L
    /// (the inner integral in closed form, the outer by Gauss–Legendre on the pieces between its kinks, a r = L and the
    /// depth's edge), and the level that share's bisection to 10⁻¹⁰ (D 1.5 mK at 30 µK: 12.912 k_BT, 387 µK; at 300 µK
    /// 4.935 k_BT, 1480 µK). Its holding temperature holds a level, and no temperature holds one at the top.
    #[test]
    fn a_reach_s_level_is_its_lowered_isothermal_s_quantile_below_the_depth() {
        let share = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        let per_khz = std::f64::consts::TAU * 1e-3;
        let depth_uk = 1500.0;
        let trap = Trap { centre: [0.0; 3], minimum: -depth_uk * KT_PER_UK, axes: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], frequencies: [100.0 * per_khz, 100.0 * per_khz, 15.0 * per_khz], lab: [0.0; 3] };
        let rule = crate::special::gauss_legendre(64);
        // ∫ f over [lo, hi] by the rule.
        let integral = |f: &dyn Fn(f64) -> f64, lo: f64, hi: f64| -> f64 { rule.iter().map(|&(x, w)| 0.5 * (hi - lo) * w * f(lo + 0.5 * (hi - lo) * (x + 1.0))).sum() };
        let mut last = 0.0;
        for t in [30.0, 100.0, 300.0, 1000.0] {
            let (degrees, _) = degrees(&trap, t);
            let (a, b, d) = (degrees.scales[1], degrees.scales[0], degrees.depth);
            assert!(degrees.scales[2] == a && b < a, "two radial scales and an axial one");
            // The share above L of the lowered isothermal, directly: per r, ∫ (e^{−r−e} − e^{−d}) de over e from max(0, (L − a r)/b)
            // to d − r.
            let share_above = |level: f64| -> f64 {
                let inner = |r: f64| -> f64 {
                    let (lo, hi) = (((level - a * r) / b).max(0.0), d - r);
                    if hi <= lo { 0.0 } else { r * ((-r - lo).exp() - (-r - hi).exp() - (-d).exp() * (hi - lo)) }
                };
                let all = |r: f64| r * ((-r).exp() - (-d).exp() - (-d).exp() * (d - r));
                let mut cuts: Vec<f64> = [level / a, (b * d - level) / (b - a)].into_iter().filter(|&c| c > 0.0 && c < d).collect();
                cuts.sort_by(f64::total_cmp);
                let ends: Vec<f64> = std::iter::once(0.0).chain(cuts).chain(std::iter::once(d)).collect();
                let part: f64 = ends.windows(2).map(|w| integral(&inner, w[0], w[1])).sum();
                part / integral(&all, 0.0, d)
            };
            let (mut lo, mut hi) = (0.0, a.min(b) * d);
            while hi - lo > 1e-10 * hi {
                let middle = 0.5 * (lo + hi);
                if share_above(middle) > share { lo = middle } else { hi = middle }
            }
            let level = degrees.level(share);
            let energy = energy_at(&trap, t, share).0;
            assert!((level - hi).abs() <= 1e-8 * hi, "{t} µK: the level {level} k_BT against the integrated quantile {hi} (depth {d} k_BT)");
            assert!(energy < -trap.minimum && energy > last, "{t} µK: the level {} µK below the depth and above the colder one's", energy / KT_PER_UK);
            last = energy;
            let warmer = holding(&trap, trap.minimum + energy, share, 0.5 * t);
            assert!((warmer - t).abs() <= 1e-9 * t, "{t} µK: its own level's holding temperature {warmer}");
        }
        let top = top(&trap, share);
        assert!((top / (-trap.minimum) - 0.9936).abs() < 5e-5 && last < top, "the top {} D above 1 mK's level {}", top / -trap.minimum, last / -trap.minimum);
        assert!(holding(&trap, trap.minimum + top, share, 1000.0).is_infinite() && holding(&trap, 0.0, share, 1000.0).is_infinite(), "nothing holds the top or the far value");
    }

    /// A path no temperature's reach holds is held by the region's level raised to it, the temperature kept (module
    /// documentation, # The trap's region): S2Q's tweezer (its sanity scene) at 30 µK, its reach's level the lowered
    /// isothermal's quantile, below the depth; a path's energy of 141.5 rad/µs above the bottom -- an atom of that scene's
    /// run, above the depth of 108.4 -- has no holding temperature; raised to it, the reach holds an atom of that energy at
    /// the trap's centre, which the temperature's does not, keeps its temperature, its level outside and its kinetic
    /// temperatures, and is the same reach where it is asked for a level it holds.
    #[test]
    fn a_path_above_the_depth_is_held_by_the_region_s_level_at_the_temperature() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("requests").join("S2Q.json");
        let request = Request::from_native(&serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()).unwrap();
        let light = Light::new(&request);
        let model = Model::new(&request, &light, &[]).unwrap();
        let state = (0..GROUND).max_by(|&a, &b| request.initial_density[(a, a)].re.total_cmp(&request.initial_density[(b, b)].re)).unwrap();
        let p: [f64; GROUND] = std::array::from_fn(|s| if s == state { 1.0 } else { 0.0 });
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let share = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        let reach = model.reach(&trap, &p, 30.0, share);
        let (degrees, _) = degrees(&trap, 30.0);
        assert!((reach.energy - degrees.level(share) * KT_PER_UK * 30.0).abs() <= 1e-12 * reach.energy && reach.energy < -trap.minimum, "the level {} below the depth {}", reach.energy, -trap.minimum);
        let level = trap.minimum + 141.5;
        assert!(141.5 > -trap.minimum && holding(&trap, level, share, 30.0).is_infinite(), "no temperature holds 141.5 above the bottom (depth {})", -trap.minimum);
        let raised = model.inside_to(&reach, level);
        let kappa = [0.0, 0.0, (2.0 * 141.5 / model.hbar_over_mass).sqrt() * (1.0 - 1e-12)];
        assert!(model.holds(&raised, trap.centre, kappa) && !model.holds(&reach, trap.centre, kappa), "the raised reach holds the path's energy at the centre");
        assert!(raised.temperature_uk == 30.0 && raised.outside == reach.outside && raised.kinetic_uk == reach.kinetic_uk && raised.energy == 141.5, "the temperature and the level outside kept");
        assert!((raised.speed - (2.0 * model.hbar_over_mass * 141.5).sqrt()).abs() <= 1e-12 * raised.speed && raised.cell.iter().zip(&reach.cell).all(|(a, b)| a[0] <= b[0] && a[1] >= b[1]), "its speeds and its box grown");
        let same = model.inside_to(&reach, trap.minimum + 0.5 * reach.energy);
        assert!(same.energy == reach.energy && same.cell == reach.cell && same.speed == reach.speed, "a level held already changes nothing");
    }

    /// A region's bounds hold at every point of its cell and are the values on its centre point (module documentation,
    /// # Provable extrema): on the six-beam lattice, each beam's modulus between its least and its largest, each beam pair's
    /// coefficient within its radius of the centre's and its
    /// gradient within its slope; every carrier's coherent coupling |Σ_l a_l X^f_l[e,s]|, its columns and its units' columns within
    /// their bounds; the far light's energies' differences within their slack (`Model::gap`) and their gradients within the gap's --
    /// and on a cell a thousand times smaller the couplings' bounds within that factor of the centre's values.
    #[test]
    fn a_region_s_bounds_hold_over_its_cell_and_are_the_values_at_its_centre() {
        let model = lattice();
        let within = |value: f64, bound: f64| value <= bound * (1.0 + 1e-12) + 1e-12;
        let norm = |v: &[C64]| v.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let cells = [
            Cell { centre: [0.13, -0.21, 0.07], half: [0.05, 0.08, 0.03] },
            Cell { centre: [-0.4, 0.3, 0.25], half: [0.3, 0.2, 0.25] },
            // Slabs and needles: thin along one or two axes, long along the others.
            Cell { centre: [1.0, 0.2, -0.1], half: [3.0, 0.3, 0.004] },
            Cell { centre: [0.1, -0.3, 2.0], half: [0.004, 0.25, 4.0] },
            Cell { centre: [0.02, 0.01, -0.03], half: [0.5, 0.01, 0.01] },
            Cell { centre: [0.0, 0.0, 0.0], half: [0.02, 2.0, 0.02] },
            // Far along each axis and off it: along a focused beam, off its axis, its modulus rises before it falls.
            Cell { centre: [6.0, 1.5, -1.0], half: [5.0, 1.0, 0.5] },
            Cell { centre: [-1.5, 6.0, 2.0], half: [1.0, 5.0, 0.5] },
            Cell { centre: [2.0, -1.0, 7.0], half: [0.5, 1.0, 6.0] },
        ];
        for cell in cells {
            let region = model.region(model.spans(&cell), 0.0);
            let couplings = model.couplings(&region);
            for r in points(&cell) {
                let a = model.light.amplitudes(r);
                for (l, span) in region.spans.iter().enumerate() {
                    assert!(within(a[l].value.abs(), span.most) && within(span.least, a[l].value.abs()), "beam {l} at {r:?}: |a| {} outside [{}, {}]", a[l].value.abs(), span.least, span.most);
                }
                let mut energy: Vec<f64> = (0..STATES).map(|s| model.base[(s, s)].re).collect();
                let mut gradient = vec![[0.0; 3]; STATES];
                for (w, f) in model.light.frequencies.iter().enumerate() {
                    let n = f.beams.len();
                    for x in 0..n * n {
                        let (c, bound) = (pair(&a[f.beams[x / n]], &a[f.beams[x % n]]), &region.pairs[w][x]);
                        assert!(within((c.value - bound.centre).abs(), bound.radius), "pair {w}/{x} at {r:?}: moved {} beyond {}", (c.value - bound.centre).abs(), bound.radius);
                        let g = c.gradient.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
                        assert!(within(g, bound.slope), "pair {w}/{x} at {r:?}: |∇c| {g} beyond {}", bound.slope);
                        let p = &model.tones[w].pairs[x];
                        for s in 0..STATES {
                            energy[s] += (c.value * p[(s, s)]).re;
                            for i in 0..3 {
                                gradient[s][i] += (c.gradient[i] * p[(s, s)]).re;
                            }
                        }
                    }
                }
                for x in 0..STATES {
                    for y in 0..x {
                        let (gap, here) = (model.gap(x, y, &region, 0.0).all, energy[x] - energy[y]);
                        assert!(within(here, gap.upper()) && within(gap.lower(), here), "E_{x} − E_{y} {here} at {r:?} outside [{}, {}]", gap.lower(), gap.upper());
                    }
                }
                for (x, y) in [(0, 3), (GROUND, 1), (GROUND + 5, 6), (STATES - 1, GROUND)] {
                    let g = (0..3).map(|i| (gradient[x][i] - gradient[y][i]).powi(2)).sum::<f64>().sqrt();
                    assert!(within(g, model.gap_slope(x, y, &region, false).norm().upper()), "|∇(E_{x} − E_{y})| {g} at {r:?}");
                }
                for (cc, bound) in model.carriers.iter().zip(couplings) {
                    let mut units: Vec<usize> = cc.units.clone();
                    units.sort();
                    units.dedup();
                    for s in 0..GROUND {
                        let field = |beams: &dyn Fn(usize) -> bool| -> Vec<C64> {
                            (0..EXCITED).map(|e| cc.beams.iter().enumerate().filter(|(l, _)| beams(*l)).map(|(l, &b)| a[b].value * cc.folded[l][(GROUND + e, s)]).sum()).collect()
                        };
                        let u = field(&|_| true);
                        for e in 0..EXCITED {
                            assert!(within(u[e].abs(), bound.entry[s][e].upper()), "|u_{s}[{e}]| {} beyond {} at {r:?}", u[e].abs(), bound.entry[s][e].upper());
                        }
                        assert!(within(norm(&u), bound.column[s].upper()), "‖u_{s}‖ at {r:?}");
                        for (ui, &unit) in units.iter().enumerate() {
                            assert!(within(norm(&field(&|l| cc.units[l] == unit)), bound.units[ui][s].upper()), "unit {ui}'s ‖u_{s}‖ at {r:?}");
                        }
                    }
                }
            }
            // On the centre point, the values; on a cell a thousand times smaller, within that factor of them.
            let point = model.region(region.spans.iter().map(Span::point).collect(), 0.0);
            let small = Cell { centre: cell.centre, half: cell.half.map(|h| h * 1e-3) };
            let tiny = model.region(model.spans(&small), 0.0);
            let a = model.light.amplitudes(cell.centre);
            for (ci, cc) in model.carriers.iter().enumerate() {
                for s in 0..GROUND {
                    for e in 0..EXCITED {
                        let exact = cc.beams.iter().enumerate().map(|(l, &b)| a[b].value * cc.folded[l][(GROUND + e, s)]).sum::<C64>().abs();
                        let (at, near) = (model.couplings(&point)[ci].entry[s][e].value, model.couplings(&tiny)[ci].entry[s][e].upper());
                        assert!((at - exact).abs() <= 1e-12 * exact + 1e-15, "the centre's |u| {at} against {exact}");
                        let scale = model.couplings(&region)[ci].entry[s][e].upper() - exact;
                        assert!(near - exact <= 2e-3 * scale + 1e-12 * exact.max(1e-3), "a thousandth of the cell: {} over the centre's, the cell's {scale}", near - exact);
                    }
                }
            }
        }
    }

    /// A focused beam's intensity over a cell far along its axis is read from its modulus's range there (`Span::moduli`,
    /// `pair_span`): R4_LIN_4's tweezer (waist 5 µm, Rayleigh range 92 µm) one and a half Rayleigh ranges from its focus, over
    /// a cell of 42 µm along the beam by 3 µm across it -- a cell of the gates' proofs over the reach at 120 µK. The pair's
    /// radius holds at every point of the cell and is within four times the intensity's largest movement over its points
    /// (bounded through the envelope's derivatives it was 1.16, seven times the intensity itself).
    #[test]
    fn a_beam_s_intensity_far_from_its_focus_is_within_its_modulus_s_range() {
        let model = at_the_atom("R4_LIN_4");
        let trap = model.light.frequencies[model.light.trap].beams[0];
        let cell = Cell { centre: [148.128, -1.623, 4.629], half: [21.1611, 1.6232, 0.0127] };
        let spans = model.spans(&cell);
        let pair = pair_span(&spans, trap, trap);
        let moved = points(&cell).iter().map(|&r| (model.light.beams[trap].amplitude(r).value.norm_sqr() - pair.centre.re).abs()).fold(0.0, f64::max);
        assert!(moved <= pair.radius, "the intensity moves by {moved} beyond its radius {}", pair.radius);
        assert!(pair.radius <= 4.0 * moved, "the radius {} against the intensity's movement {moved}", pair.radius);
        assert!(spans[trap].least > 0.0 && spans[trap].most < 1.0, "the modulus within [{}, {}]", spans[trap].least, spans[trap].most);
    }

    /// One tree of cells for many quantities proves each as its own proof would, and so do its rounds' cells halved on all the
    /// cores, the proofs asking ahead of their order: the shared cells and the cells made ahead change no bound (R3_DOPPLER's
    /// cooling pair's drives at the atom: the strongest drive's saturation and sweep from two ground states).
    #[test]
    fn one_tree_proves_each_quantity_as_its_own_proof_would() {
        let model = at_the_atom("R3_DOPPLER");
        let b = 7;
        let p = {
            let mut p = [0.0; GROUND];
            p[b] = 1.0;
            p
        };
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, 20.0, 2e-3);
        let s = |r: &Region| model.folds(b, 0, r)[0];
        let a = |r: &Region| model.folds(b, 0, r)[1];
        let other = |r: &Region| model.folds(3, 0, r)[0];
        let together = model.bounds_over(&reach, 0.5, 0.5, &[&s, &a, &other]);
        let alone: Vec<f64> = [&s as &dyn Fn(&Region) -> f64, &a, &other].iter().map(|q| model.bounds_over(&reach, 0.5, 0.5, &[*q])[0]).collect();
        assert_eq!(together, alone);
        assert_eq!(model.bounds_over_in_parallel(&reach, 0.5, 0.5, &[&s, &a, &other]), alone);
    }

    /// A gate's proof asks its band [¼, ½) (units.rs `KEEP`), not the margin of a value from ½: R3_DOPPLER's strongest
    /// cooling drive's saturation scaled to a largest value over the reach (its extent to 1e-3) of 0.45 is settled by
    /// keeping -- a centre attains ¼ -- in far fewer evaluations than the threshold question alone takes; scaled to 0.2,
    /// below the band, it is still proven below ½; scaled to 0.7 it is kept.
    #[test]
    fn a_gate_s_proof_asks_its_band_not_the_margin() {
        use crate::model::units::{EXISTS, KEEP};
        use std::sync::atomic::{AtomicUsize, Ordering};
        let model = at_the_atom("R3_DOPPLER");
        let b = 7;
        let p = {
            let mut p = [0.0; GROUND];
            p[b] = 1.0;
            p
        };
        let trap = model.trap(&p, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &p, 20.0, 2e-3);
        let s = |r: &Region| model.folds(b, 0, r)[0];
        let largest = model.extents_over(&reach, 1e-3, &[&s])[0];
        for (target, kept) in [(0.45, true), (0.2, false), (0.7, true)] {
            let (scale, asked) = (target / largest, AtomicUsize::new(0));
            let q = |r: &Region| {
                asked.fetch_add(1, Ordering::Relaxed);
                scale * s(r)
            };
            let banded = model.bounds_over(&reach, EXISTS, KEEP, &[&q])[0];
            let in_band = asked.swap(0, Ordering::Relaxed);
            let exact = model.bounds_over(&reach, EXISTS, EXISTS, &[&q])[0];
            let at_threshold = asked.load(Ordering::Relaxed);
            eprintln!("largest {target}: banded {banded:.4} in {in_band} evaluations, the threshold alone {exact:.4} in {at_threshold}");
            assert_eq!(banded >= EXISTS, kept, "largest {target}: the banded bound {banded}");
            assert!(in_band <= at_threshold, "largest {target}: {in_band} evaluations in the band, {at_threshold} at the threshold");
            if target < KEEP {
                assert_eq!(banded, exact, "below the band the two questions are one");
            }
        }
        // Its largest value at ½(1 − 1e-6), where the threshold question alone would resolve cells to a millionth of ½:
        // kept on the root's few halvings.
        let (scale, asked) = (0.5 * (1.0 - 1e-6) / largest, AtomicUsize::new(0));
        let q = |r: &Region| {
            asked.fetch_add(1, Ordering::Relaxed);
            scale * s(r)
        };
        let banded = model.bounds_over(&reach, EXISTS, KEEP, &[&q])[0];
        let asked = asked.load(Ordering::Relaxed);
        eprintln!("largest ½(1 − 1e-6): banded {banded:.4} in {asked} evaluations");
        assert!(banded >= EXISTS && asked <= 64, "a largest value at ½(1 − 1e-6): {banded} in {asked} evaluations");
    }

    /// A reference scene's request, its light and its model with every drive folded with the bare levels: a generator
    /// of constant terms alone.
    fn bare(scene: &str) -> (Request, Light, Model) {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("refs").join("scenes").join(format!("{scene}.json"));
        let line: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let request = Request::from_native(&line).unwrap();
        let light = Light::new(&request);
        let model = Model::new(&request, &light, &[]).unwrap();
        (request, light, model)
    }

    /// The kick's curvature (module documentation, # The kick's curvature) on models whose every drive is folded with
    /// the bare levels, where the bound holds all of the generator.
    /// - A σ⁺ tweezer alone, the atoms in the state it holds the least (R1_SIGMA_P, |2,+2⟩): every other ground state's
    ///   well is the same shape and deeper, so the bound is the largest ω_s² − ω_ref² of the states' own traps' largest
    ///   rates (potential.rs `Model::trap`), to rounding (printed: a tenth of ω_ref² and more -- the step's edge is
    ///   below 0.91 π). From the state it holds the most (R1_SIGMA_M, |2,−2⟩) every other well is the softer one: the
    ///   bound is the reference's own, none.
    /// - Six beams' cooling light with its repump in a retro-reflected tweezer (R8_R5051, every drive bare): at the
    ///   trap's centre the second difference of ⟨x|Ĥ_h|x⟩ − V_ref along any direction, of any state of a cluster, is
    ///   below the bound (16 directions and states a cluster; the largest found is printed beside it).
    #[test]
    fn the_kick_s_curvature_is_the_wells_difference_and_bounds_the_fringes() {
        use crate::step::Middle as _;
        let alone = |s: usize| -> [f64; GROUND] { std::array::from_fn(|b| if b == s { 1.0 } else { 0.0 }) };
        let share = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        for (scene, state) in [("R1_SIGMA_M", 3), ("R1_SIGMA_P", 7)] {
            let (request, _, model) = bare(scene);
            assert!(request.initial_density[(state, state)].re == 1.0, "{scene}: the atoms' state");
            let trap = model.trap(&alone(state), &model.prepared()).unwrap();
            let reach = model.reach(&trap, &alone(state), request.temperature_uk, share);
            let rate = |trap: &Trap| trap.frequencies.iter().copied().fold(0.0, f64::max);
            let stiffest = (0..GROUND).map(|s| rate(&model.trap(&alone(s), &model.prepared()).unwrap()).powi(2) - rate(&trap).powi(2)).fold(f64::NEG_INFINITY, f64::max);
            let (kick, own) = (model.kick_curvature(&reach, &model.moduli(&reach), &trap), rate(&trap).powi(2));
            eprintln!("{scene}: ω_k² = {:.4} ω²; the traps' {:.4} ω²", kick / own, stiffest / own);
            assert!((kick - stiffest).abs() <= 1e-9 * own, "{scene}: the bound {kick}, the wells' difference {stiffest}");
            assert!(if scene == "R1_SIGMA_P" { kick > 0.1 * own } else { kick.abs() <= 1e-9 * own }, "{scene}: {kick} of {own}");
        }
        let (request, _, model) = bare("R8_R5051");
        let reference = alone((0..GROUND).find(|&s| request.initial_density[(s, s)].re > 0.0).unwrap());
        let trap = model.trap(&reference, &model.prepared()).unwrap();
        let reach = model.reach(&trap, &reference, request.temperature_uk, share);
        let bound = model.kick_curvature(&reach, &model.moduli(&reach), &trap) / model.hbar_over_mass;
        let (lights, prepared) = (model.running(), model.prepared());
        let mut stream = crate::random::Stream::jump(11, 0, 0);
        let mut largest = f64::NEG_INFINITY;
        for cluster in 0..model.clusters.len() {
            let n = model.clusters[cluster].states.len();
            for _ in 0..16 {
                let direction = { let d = stream.normals(); let norm = norm3(d); d.map(|x| x / norm) };
                let x: Vec<C64> = (0..n).map(|_| { let z = stream.normals(); C64::new(z[0], z[1]) }).collect();
                let norm = crate::linalg::norm_sqr(&x);
                // ⟨x|Ĥ_h|x⟩ − V_ref along the direction, and its second difference over a fortieth of the light's
                // shortest length (its error (qh)²/12 of the curvature: 5·10⁻⁵).
                let step = model.shortest_length(&lights) / 40.0;
                let value = |s: f64| {
                    let r: [f64; 3] = std::array::from_fn(|a| trap.centre[a] + s * direction[a]);
                    let middle = model.assemble(cluster, r, 0.0, &lights);
                    let mut h = Mat::zeros(n, n);
                    middle.generator(&middle.path([0.0; 3], [0.0; 3]), 0.0, &mut h);
                    let form: C64 = (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).map(|(a, b)| x[a].conj() * h[(a, b)] * x[b]).sum();
                    form.re / norm - model.potential(r, &reference, &prepared)
                };
                let curvature = (value(step) - 2.0 * value(0.0) + value(-step)) / (step * step);
                largest = largest.max(curvature);
                assert!(curvature <= bound * (1.0 + 1e-3), "cluster {cluster}: a held state's curvature {curvature} above the bound {bound}");
            }
        }
        eprintln!("R8_R5051, every drive bare: the largest curvature found {largest:.3} rad/µs/µm², the bound {bound:.3}");
        assert!(largest > 0.0 && bound.is_finite(), "{largest} {bound}");
    }

    /// A reach above the potential's far value is held within the trap's region (module documentation, # The trap's
    /// region). R2_RECOIL's tweezer: at 30 µK its level lies below the far value, the sublevel set whole; no temperature
    /// holds a level half the trap's depth above it (`holding`), and raised there within the region (`inside_to`) the set
    /// is the region's -- its box the trap beam's ellipsoid u ≤ 19, the focus held at rest and at the speed the level leaves
    /// there, a point on the axis at six Rayleigh ranges (|a|² = 1/37, outside the region) not held even at rest, though
    /// bound, the level outside the region the temperature's. Holding that point's prepared energy outside the region too (`outside_to`) holds
    /// it, and everything the reach held before; its box is larger.
    #[test]
    fn a_reach_above_the_far_value_is_held_within_the_trap_s_region() {
        let share = crate::estimate::unseen(crate::units::MOST_TRAJECTORIES as f64);
        let model = at_the_atom("R2_RECOIL");
        let p = { let mut p = [0.0; GROUND]; p[7] = 1.0; p };
        let (trap, prepared) = (model.trap(&p, &model.prepared()).unwrap(), model.prepared());
        let cold = model.reach(&trap, &p, 30.0, share);
        assert!(cold.minimum + cold.energy < 0.0 && cold.outside == cold.energy && cold.cell.iter().all(|c| c[1].is_finite()), "{cold:?}");
        let level = -0.5 * trap.minimum;
        assert!(holding(&trap, level, share, 30.0).is_infinite(), "no temperature holds a level above the far value");
        let hot = model.inside_to(&cold, level);
        let beam = model.light.beams[model.light.frequencies[model.light.trap].beams[0]];
        let (w, z, n) = (beam.waist.unwrap(), beam.rayleigh().unwrap(), beam.direction);
        let region: [f64; 3] = std::array::from_fn(|a| (19.0 * (0.5 * w * w * (1.0 - n[a] * n[a]) + z * z * n[a] * n[a])).sqrt());
        assert!(hot.outside == cold.outside && hot.temperature_uk == 30.0 && (0..3).all(|a| (hot.cell[a][1] - region[a]).abs() <= 1e-12 * region[a]), "{:?} against {region:?}", hot.cell);
        let kinetic = |r: [f64; 3]| model.room(&hot, r, &prepared);
        let focus = [0.0; 3];
        let at_rest = |r: [f64; 3]| [0.0; 3].map(|x: f64| x + 0.0 * r[0]);
        let speed = (2.0 * kinetic(focus) / model.hbar_over_mass).sqrt() * (1.0 - 1e-9);
        assert!(model.holds(&hot, focus, at_rest(focus)) && model.holds(&hot, focus, [0.0, 0.0, speed]), "the focus");
        let far = n.map(|x| 6.0 * z * x);
        let bound = model.potential(far, &p, &prepared);
        assert!(model.light.outside(far) && bound < 0.0 && !model.holds(&hot, far, at_rest(far)), "six Rayleigh ranges out: V {bound}");
        let wider = model.outside_to(&hot, bound);
        assert!(model.holds(&wider, far, at_rest(far)) && model.holds(&wider, focus, [0.0, 0.0, speed]) && (0..3).all(|a| wider.cell[a][1] >= hot.cell[a][1]), "{wider:?}");
        assert!(wider.outside < wider.energy && model.outside_to(&wider, 0.5 * bound).outside > wider.outside, "the outside level grows, below the far value");
    }

    /// The model holds along a path where every point of it does, and fails where a point of it fails (`holds_along`):
    /// R2_RECOIL with its resonant beam's drives folded at the atom, whose folds the whole space does not settle. On 200
    /// paths of 0.5 µm about points across the trap's region at speeds up to 1 µm/µs, against their gates at 65 points
    /// each: a path with a point that fails is never held to hold; both outcomes occur.
    #[test]
    fn a_path_holds_where_its_points_do_and_fails_where_one_does_not() {
        let model = at_the_atom("R2_RECOIL");
        let units = model.units();
        let judged = model.judged(&units);
        let open = model.everywhere(&judged, 0.5).1;
        assert!(!open.is_empty(), "the folds at the atom are open over the whole space");
        let mut stream = crate::random::Stream::jump(11, 3, 0);
        let (mut held, mut failed) = (0, 0);
        for _ in 0..200 {
            let r: [f64; 3] = std::array::from_fn(|_| 6.0 * (stream.uniform() - 0.5));
            let v: [f64; 3] = std::array::from_fn(|_| 1.2 * (stream.uniform() - 0.5));
            let speed = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            let half = 0.25 / speed;
            let points_fail = (0..=64).any(|i| {
                let u = half * (i as f64 / 32.0 - 1.0);
                !model.exist_over(&judged, &open, &model.point_region(std::array::from_fn(|a| r[a] + v[a] * u), speed))
            });
            let along = model.holds_along(&judged, &open, r, v, half, speed);
            assert!(!(points_fail && along), "a path with a failing point held to hold: r {r:?} v {v:?}");
            if along { held += 1 } else { failed += 1 }
        }
        assert!(held > 0 && failed > 0, "{held} held, {failed} failed");
    }
}
