//! A generator along a step's path in the interaction picture of its static part at the step's middle (SPEC §2.1;
//! step.rs, # The kick): what the step's one propagation is made of.
//!
//! # The generator
//! Along the path Ĥ(u), u the time from the step's middle, is a static group and exact tones,
//!   Ĥ(u) = G₀(u) + Σ_m (e^{iν_m u} G_m⁺(u) + e^{−iν_m u} G_m⁻(u)),
//! each tone's beat ν_m exact (rad/µs: the beats of the lights and the residual wave vectors' turn along the path, as
//! the model gives them, step.rs `Middle::beats`) and its two members' envelopes G_m^± slow: the beams' profiles along
//! the path. A tone's two members turn at ±ν_m: a generator's beating part is such pairs, the Hermitian one's G⁻ = G⁺†
//! and its decay's alike (a member may be none). Ĥ₀ = G₀(0) holds every frequency of the atom's own -- detunings, light
//! shifts, decay rates, of any size against the step --, and D(u) = Ĥ(u) − Ĥ₀ the rest: the static group's change
//! along the path and every tone whole. Each envelope is taken through the step's three nodes, the middle and the two
//! ends ±h, as the parabola C + uA + u²B, C its middle's value, A = (G(h) − G(−h))/2h and B = (G(h) + G(−h) − 2G(0))/2h²
//! (the static group's C is none in D: it is Ĥ₀). Its error is the cubic's, (u³ − uh²)G‴/6: third order in the
//! envelope's slow rate at a time of the step, whatever Ĥ₀ is, and odd about the middle, so that the step's integrals
//! take it at the fourth; a half step takes its own three nodes on half the spacing. A tone's beat is not in the
//! parabola: it is a factor e^{±iν_m u}, integrated exactly with the pairs' own turns (# Tones), so that no node has to
//! resolve it.
//!
//! # Tones
//! In Ĥ₀'s eigenbasis the propagation's terms come in classes by their beat: the static group (ν = 0), each tone's
//! two members (±ν_m), and the products of two members' terms that the following's second order makes (# The second
//! order by tones). An entry (a, b) of a class of beat ν turns at its combined frequency
//!   Ω_ab = ω_ab + ν,  ω_ab = λ_a − λ_b,
//! and everything the propagation does to it -- whether it is followed or hopped (# The split), its following, its
//! hops' moments -- is of Ω alone: the static group's entries are the case ν = 0, and of one implementation the
//! propagation of the static group alone is the one it was before the tones, to the bit (tests,
//! `the_static_group_alone_is_propagated_as_before_to_the_bit`). An entry and its partner's -- (b, a) of the class of
//! the opposite beat, the static group its own partner -- turn at Ω and −Ω and are taken together.
//!
//! The combined frequencies are the gauge's invariants. A frame that turns a group of states at θ -- a group the static
//! group does not couple to the others -- moves their diagonal in the static group by θ, their eigenvalues with it, and
//! the beat of every tone between the group and the others by θ the other way: each Ω stays as it is, and so does the
//! state the propagation gives, taken back to the first frame, to rounding (tests,
//! `a_frame_on_a_group_of_states_leaves_the_propagated_state_as_it_is`: 10⁻¹⁵ at θτ up to 17). A beat the frame could
//! not take away -- a loop's, whose edges' beats sum to it whatever the frame -- costs the step nothing where the
//! loop's own detunings, the Ω of its entries, are slow: the step need not resolve the beat (tests,
//! `a_loop_s_weak_edge_beats_fast_and_is_taken_at_its_combined_frequency`: a loop closed at B = 26.4 rad/µs over a step
//! of Bτ = 7.9, its edge's Ω 0.64 rad/µs, within the hops' second Magnus term, 5.7·10⁻⁵ against 6.2·10⁻⁵; the beat
//! through the three nodes errs by 6.9·10⁻², the alias).
//!
//! # Which beats are tones
//! What goes into the static group and what is a tone is the model's to say (step.rs `Middle::beats`), by what each
//! leaves. A beat e^{iνu}C taken into the static group is the change e^{iνu}C − C through the nodes, of the slope
//! s_A ≈ |νC|h² over a half step, and its hops leave x s_A²/15 (# The split); taken as a tone it is the envelope C whole,
//! s_C = |C|h, and they leave x s_C²/3. The first is the less where νh < √5 -- a slow beat of a strong coupling, a 3D
//! lattice's residual q·v at the steps the scenes take --, the second beyond it -- a loop's weak closing edge at its
//! fast beat --; where the two are equal they swap, so nothing jumps there (as with the split, # The split). The
//! static group's change is also read through the nodes, which alias a beat beyond νh = π: a tone has no such edge.
//!
//! A tone is its Hermitian part, P = (G⁺ + G⁻†)/2 and P†; the decay's beats stay in the static group (`Picture::separate`).
//! The hop's exponent is the first Magnus term −i∫D̃ (# The state at any time of the step), and it keeps the flow's
//! dissipative structure only where its anti-Hermitian part is a decay: of D = H − (i/2)Γ it is −(1/2)∫Γ̃, negative
//! semidefinite where Γ̃ is at the times it is integrated over. A beating decay's part alone,
//! −(i/2)(e^{iνu}Γ_ν + e^{−iνu}Γ_ν†), is no decay: Γ_ν is not positive -- the interference term of two plane waves'
//! scattering --, its sign turns with its beat, and its integral over a hop is of either sign, so that as a tone it
//! raises m's norm by up to e^{|Γ_ν| w} over a hop of the width w, where only its sum with the static decay is positive
//! (measured: a random Lindblad's step, the survival test's case 68, overflowed). With the static group it is that sum,
//! as the nodes give it; the static group's parabola of it is the step's decay (step.rs, # Jumps), whose negative part
//! between the nodes is what the survival may rise by (step/tests.rs `negative_decay`), and its alias beyond νh = π the
//! decay rate's error, the check's to read, as before the tones. A Hermitian tone's hops are unitary.
//!
//! # The state at any time of the step
//! With Ĥ₀ = c + RΛR⁻¹ (its scalar part c apart, `linalg::Eigen`) and x = R d, i ḋ = (Λ + Dᵉ(u)) d, Dᵉ = R⁻¹DR, each
//! class's envelopes taken to the eigenbasis (two products of n³ each). What D does to an entry of a class depends on how
//! far it turns between the step's nodes against the size of what changes (# The split).
//!
//! **A fast entry** -- it turns far between two nodes for what changes: an explicit state's detuning, its decay, at
//! any step; a tone far from every pair it drives. The state follows the change: d = (1 + X(u)) g, X = Σ e^{iνu} Y(u)
//! over the classes, each Y on the class's fast entries the polynomial that solves Ω_ab Y_ab − i Ẏ_ab + E_ab = 0
//! exactly (E the class's envelope; ω_ab X_ab − iẊ_ab + Dᵉ_ab = 0 for the sum),
//!   Y = Y₀ + uY₁ + u²Y₂,  Y₂ = −B/Ω,  Y₁ = (2iY₂ − A)/Ω,  Y₀ = (iY₁ − C)/Ω  (element by element)
//! -- the generator's own eigenvectors along the path to the first order of the change, with the corrections of its
//! rate, each beat's at its own combined frequency. Then g obeys i ġ = (Λ + D₂(u)) g with (1 + X) D₂ = D_s + D X
//! exactly (D_s: D on the hopped entries). What that leaves on the fast entries in the second order,
//! F = [D X − X D_s]_f, is followed in its turn, class by class of its beats (# The second order by tones): F is made at
//! the nodes, taken through them like D, and its polynomials are added to X. With it (1 + X) D₂ = D_s + D X − F at the
//! nodes, and D₂'s fast entries are of the third order in the change. They are left out. No time of the step is
//! resolved for a fast entry and nothing of it is integrated: 1 + X(u) is a matrix at any time, at any |Ω| |h| however
//! large.
//!
//! **A slow entry**, and every state with itself in the static group, stays in D₂: from (1 + X) D₂ = D_s + D X − F by
//! the factors of 1 + X at the nodes -- exactly, so that the change's own light shift −Σ_e D_ge D_eg′/ω_eg′ is in it (a
//! hop through unresolved pairs left it out) and a generator without a decay keeps the state's norm (measured: within
//! 10⁻⁷ over a step where the following is 0.04, 10⁻¹⁰ where it is 0.01; taken to the second order alone it drifted by
//! a sixth of the following's square there) --, each class's part through the three nodes (# The second order by
//! tones).
//!
//! g = e^{−iΛ(u − u₀)} m, u₀ the time the state is launched at (the step's start, −h): Ĥ₀'s phases and decays are
//! factors, exact over any width, and i ṁ = D̃₂(u) m with D̃₂_ab = e^{iω_ab(u − u₀)} (D₂)_ab on the hopped entries. m is
//! hopped from its launch to the middle of what is left of the step and to the step's end -- from the step's start to
//! its middle and to its end: two hops, whatever Ĥ₀ --, each the first Magnus term, m ← e^{−iJ} m with J = ∫D̃₂ over the
//! hop, the integral over each entry's turn exact: for a hop from u over w, with each class's part of D₂
//! e^{iνu}(C + uA + u²B),
//!   J_ab = e^{iω_ab(u − u₀)} Σ e^{iνu} w [C_ab I₀ + A_ab (u I₀ + w I₁) + B_ab (u² I₀ + 2uw I₁ + w² I₂)],
//!   I_p = ∫₀¹ s^p e^{zs} ds,  z = iΩ_ab w,
//! the moments by their series Σ_j z^j/(j!(j + p + 1)) -- |z| is below the split for a hopped entry over a half step,
//! of the order of one: the terms' moduli sum to e^{|z|} at most, a digit of the rounding --, summed until its terms are
//! within the round-off of the state the exponent acts on or of the exponent's own first term (`series`,
//! `Picture::exponent`); an entry's and its partner's are the series' even and odd parts.
//!
//! Between its three stops m is their quintic: its values there and its rates ṁ = −iD̃₂m (zero at the step's middle
//! where there are no tones, as D₂ is), the Hermite interpolant. m changes by the hopped entries alone, so its sixth
//! derivative is of the size |Ω|⁵ |D₂| with |Ω| |h| below the split, and the quintic errs by no more than (4/27)/720
//! of w⁶ times it, w the stops' spacing. So the state at any time u from the launch on is
//! x(u) = R (1 + X(u)) e^{−iΛ(u − u₀)} m(u) -- n exponentials, the matrix R(1 + X(u)) from its three of each class
//! followed (R(1 + Y₀), RY₁ and RY₂ of the static group's, RY₀, RY₁, RY₂ of a beating class's at its phase, made once a
//! step) and its product of n² (`Picture::at`), no hop --, and at the three stops it is the hops' own.
//!
//! The propagation is linear in the state, and nothing of the picture but m knows the state: a state that changes at
//! a time of the step -- a quantum jump's (step.rs, # Jumps) -- is launched again at that time on the same picture,
//! m = (1 + X(u₀))⁻¹R⁻¹x there (one factorization of n³/3), its two hops over what is left of the step, each no wider
//! than a half step, and the quintic of its own three stops (`Picture::launch`).
//!
//! Where nothing moves (D = 0: an atom at rest, a generator still along every path, no tones) X = 0 and m is
//! constant: the state is e^{−i(c + Λ)(u − u₀)} on its launch, exact at any detuning times step.
//!
//! What is left out, all of the change alone. On one entry, with s_C = |C| |h|, s_A = |A| h² and s_B = |B| |h|³ the
//! sizes of its envelope's constant, slope and curvature over a half step and x = |Ω| |h| its turn between two nodes
//! (derived, and read against the flow: tests, `a_pair_hopped_and_followed_against_its_flow`), at the step's end:
//! - of a hopped entry, the hops' second Magnus term, −½∫∫[D̃₂(t₁), D̃₂(t₂)] -- of the constant
//!   |C|² ∫∫ sin Ω(t₁ − t₂) = |C|² (w − sin(Ωw)/Ω)/Ω a hop, (h⁵/60)[A, [Λ, A]] of the slope and terms of [A, B] --: over
//!   the step's two hops x s_C²/3, x s_A²/15 and x s_B²/42, and where the envelope's parts are out of phase
//!   Im(C A*) h³/3 and Im(A B*) h⁵/15. Of the static group's change the fifth order in the step, so the propagation is
//!   of the fourth order there like the three nodes of D (Simpson's rule's: their cubic is odd about the middle; what
//!   stays is τ⁵D⁗/2880 a step); of a tone's envelope the third, as of any coupling the first Magnus term takes whole;
//! - of a fast entry, in the second order what the three nodes' parabola of D₂ misses of the change's own light shift
//!   between them, the quartic u⁴|B|²/Ω: ∫(u⁴ − h²u²)du |B|²/Ω = (4/15) s_B²/x, and nothing else; and the third order
//!   of the following, D₂ on the entry itself, −X D X: for a slope −(|A|²A/Ω²) u (u + i/Ω)², s_A³ |I(x)|/x² with
//!   I(x) = ∫₋₁¹ e^{ixt} t (t + i/x)² dt = i(2/(3x) + x/15 + …), that is (2/3)(s_A/x)³ to the leading order; for a
//!   constant −(|C|²C/Ω²) at the entry's turn, (s_C/x)³ |∫₋₁¹ e^{ixt} dt| = 2 s_C³ |sin x|/x³ ≤ 2 s_C³ min(x, 1)/x³.
//!   They fall with the detuning, the slope's and the curvature's with the step. (A hop through an unresolved pair left
//!   the light shift of the change, (k·v τ)²/24 of the light shift's turn over the step at any detuning.)
//! (step/tests.rs, `a_step_is_of_the_fourth_order_in_the_path_s_change_where_its_hops_resolve_the_generator`.)
//!
//! # The split
//! Each entry takes one of three routes (`followed`), by what each leaves at the step's end. With the leading remainders
//!   F = (2/3)(S_A/x)³ + 2 S_C³ min(x, 1)/x³    (the following's third order),
//!   L = (4/15) S_B²/x                            (the parabola's miss of the change's own light shift),
//!   J = x (S_C²/3 + S_A²/15 + S_B²/42)           (the hops' second Magnus term),
//! and r the following's size on the entry, |Y| at the step's ends of what is followed there -- a first-order class's
//! change, a product's (# The second order by tones) own second order --, the larger of its two ways round: the
//! following's orders a series in r, their sum F/(1 − r). (An entry nothing changes on has r = 0: its following is
//! none, whatever the sizes the rule reads.)
//! - **followed** where the following converges, r < 1, and leaves no more than the hops: F/(1 − r) + L ≤ J;
//! - **hopped** where the hops leave less: J < F/(1 − r) + L with r < 1, or J < F + L with r ≥ 1 -- the hops hold
//!   where the entry turns little between the nodes, which is where J is the less --, and an entry that does not turn
//!   (x = 0);
//! - **neither**, the step by the strides (# The strides), where the following has no sum, r ≥ 1 -- 1 + X is no near
//!   identity and D₂'s fast entries left out are not of a higher order --, and the hops leave more than even its
//!   leading term, J ≥ F + L: such an entry turns far between the nodes, and hopped it turns by its decay rates'
//!   difference too, e^{|Im Ω| 2h} over the step, the hops' exponent without bound (a standing wave's two-level atom
//!   at a step of 2 µs, its pair 27 rad/µs apart and Γ/2 in decay: the hops' Chebyshev series of 10¹⁶ parts). The
//!   strides take the generator's exponentials at their nodes, bounded at any step: inaccurate, as every route is at
//!   such a step, and the check reads it.
//! Each choice is the route that leaves the least of those that exist: where r < 1 the two compared are both sums, and
//! where r ≥ 1 the hops are compared with the following's least -- its leading term -- so that the hops are taken
//! only where they leave less than any following could. (Without the convergence a tone of a coupling stronger than
//! its combined frequency, |C| > |Ω|, was followed at a step of |Ω||h| beyond a few radians, where F < J: the state
//! through it grew by 10⁶ over a step of 10 µs and overflowed at 90 µs, tests,
//! `the_survival_never_rises_whatever_the_generator_and_the_step`; a static group's slope is followed at r ≪ 1 at the
//! scenes' steps, where nothing changed, tests, `the_static_group_alone_is_propagated_as_before_to_the_bit`.)
//! F and L fall with x and J rises: the entries are split at one turn, and an entry that crosses it
//! with the step changes its treatment where the two leave the same. For a slope alone that turn is x⁴ = 10 S_A -- a
//! radian between two nodes where the change over a half step is a tenth, 0.56 rad where it is a hundredth --, for a
//! constant alone x³ = 6 S_C (x ≤ 1, S_C ≤ 1/6), x⁴ = 6 S_C beyond: no number. S_C, S_A and S_B are the sizes of all
//! the first-order classes' changes together -- the root of the sum of their squared constants |h||C_ab| (the tones'),
//! slopes h²|A_ab| and curvatures |h|³|B_ab|, each pair once, the mean of its two ways round --, one entry's own where
//! there is one. With several, either remainder is a sum over the states the entries share (the commutator's, X D X's),
//! and an entry's own slope does not bound what its following leaves (measured: with the pair's own size in the rule,
//! pairs of a small slope beside large ones were followed at a small turn, and the state erred up to twice more than
//! with every pair split at a radian). The products of the second order are split by the same rule at their own
//! combined frequencies.
//!
//! The rule holds the two leading remainders of the state at the step's end. It does not hold the hops' terms of
//! [C, A] and [A, B] (of either sign), the parts of an envelope out of phase with each other: an envelope that turns its
//! own phase turns at a frequency, which is the model's to give in its beat (a beam's profile along the path is real
//! but for its Gouy and curvature phases, which turn over the beam's Rayleigh length, not over a step), and its static
//! group's change is its parabola's. Nor does it hold the curvature's third order -- the next order in what the change
//! itself turns between two nodes --, or the state inside the step and the kick read on it (a hopped entry's state
//! between the stops is the quintic's, a followed one's the polynomial's). A half step takes the entries by its own
//! nodes.
//!
//! Measured on scenes' own paths against every pair split at a radian (step/tests.rs,
//! `a_scene_s_step_error_by_the_split`: the state over a span in one step against the path's flow, its fall on
//! halving the step, and the same of the path's impulse on those states; sixteen atoms, random states). At the
//! scenes' steps the two are within a tenth: PGC3D's five ground states at 0.625 µs 1.80·10⁻⁴ against 1.95·10⁻⁴,
//! GMD1C's thirteen states at 0.5 µs 7.3·10⁻⁵ against 6.5·10⁻⁵. At twice those steps, where the pairs' changes over a
//! half step reach 0.3 to 0.5 and a radian follows pairs a hop takes better: 4.3·10⁻³ against 8.4·10⁻³ (falling 14.7
//! against 28.7 on halving, where the order's fall is 16) and 1.9·10⁻³ against 2.7·10⁻³ (14.2 against 22.1); the
//! impulse 3.0·10⁻⁴ against 6.8·10⁻⁴ of the momentum (falling 17.6 against 40.4) and 9.6·10⁻⁵ against 2.0·10⁻⁴ (13.9
//! against 35.0). The propagation's own error is a hundredth to a seventh of what one step differs from its two halves
//! by, which is the splitting's. (The radian's numbers were read with the turn forced in the rule's place, a switch for
//! the tests that the engine no longer has: the measurement reads the rule's alone. They are of the static group alone.)
//!
//! # The second order by tones
//! The second order D X − X D_s is a sum of products of a first-order class's change and another's following, and the
//! beat of each is the sum of the two classes' beats: of the static group with itself the static group's; of a member
//! with the static group that member's; of a tone's two members the static group's (their beats cancel: a beam's own
//! light shift); of two members otherwise their sum's or their difference's, a class of its own -- the light shift's
//! beat between two beams that drive one far state, its lattice along the path. Each product is made at the nodes on
//! the classes' envelopes, its phase apart (two products of n³ a node: the static group's with itself at the two ends,
//! as before the tones, every other at the three), and its fast part F is followed at its class's combined frequency.
//! D₂ is made at the nodes whole, by the factors of 1 + X there, as before: the static group's all orders exact. Of it
//! each beating class keeps its part to the second order on its hopped entries -- its change and its second order --
//! through the three nodes at its own phase, and the static group the rest, the solve's less what the beating classes
//! hold at the nodes, through the three nodes (its middle's value, zero where there are no tones, its constant): what
//! is left there of the beats' third and higher orders is taken through the nodes as their parabola, at most the
//! fourth order's beats' size, (Σ|c|)⁴/Δ³ of couplings c a detuning Δ away, over the step. Following a product leaves
//! a third order on the entries it multiplies, the couplings times its following, c X⁽²⁾ at their frequency
//! (tests, `a_far_state_s_two_beams_are_followed_and_their_light_shift_beats_exactly`: two beams 600 rad/µs from a
//! decaying state, their beats and their light shift's beat 3.1, 4.7 and 7.8 rad over a step of 1 µs, 4.1·10⁻⁵ against
//! the bound's 1.3·10⁻⁴; through the three nodes 2.2·10⁻²).
//!
//! The work beside the static group's: per tone member its envelope to the eigenbasis (three changes, 6n³), per product
//! of two classes and node 2n³ (the products beyond the static group's with itself), per beating class at each node its
//! part of X, the right side and D (6n²) and its hops' envelopes and moments; the middle's solve where there are tones
//! (n³ + n³/3 + n³); and per beating class followed three products of n³ for a state at a time, 4n² more each.
//!
//! # What follows the generator's own frequencies, and what does not
//! The propagation's own work -- the decomposition, the following, two hops -- is the same whatever Ĥ₀'s frequencies
//! times the step are. The step's forms are read on the state at times of a quadrature (step.rs `propagate`), and
//! their number still follows the forms' bandwidth -- max|λ_a* − λ_b|; of a tone's forms, summed at its beat
//! (step.rs `Sums`), the combined frequencies max|λ_a* − λ_b ± ν| -- a tone far from every pair's frequency asks the
//! quadrature its combined frequency, not its beat: tilt2's loop's weak edge beats at 26 rad/µs and its pairs at
//! 0.4–1.2 --; and where a beating class is followed the beats' spread with it, the following moving the state's
//! frequencies by its beat: the kick is a quotient by the state's
//! norm, which with a decay is a sum of the pairs' exponentials of its own -- ringing wherever the state holds a fast
//! pair's two states --, and a quotient of two such sums has no closed form. (The forms on the unnormalized state have
//! one, pair by pair: the moments above. They would serve where the norm is one, which is no cluster that scatters.) A
//! state there costs n exponentials and two products of n², against a hop's series and its exponential before.
//!
//! # The sums
//! The picture's products -- the changes' change of basis (R⁻¹A)R, the following's D X, X D_s and its second pass, the
//! three matrices of a state at a time, the start's coordinates and a state from its coordinates -- are sums Σ_k c_k x_k
//! in the order of k, made by the lanes' kernel (`linalg::combine`: fused multiply-adds, a sum held in its place) on
//! matrices laid for it (`lay`: the rows over the lanes), a product's right factor column by column as the kernel's
//! weights. Each is the plain sum it stands for within the rounding of its operations, γ of their count times the
//! terms' moduli (tests: `the_kernel_s_products_are_their_plain_sums_within_their_rounding`); a sum of 3n terms in one
//! chain is avoided where its latency, not its work, would set the pace (a state's matrix is made first, then one sum
//! of n). The decomposition (`linalg::Eigen`), the factors of 1 + X and the hops' exponentials are not of that kind and
//! stay as they are.
//!
//! # The strides: the state without the decomposition
//! The state can also be taken without the decomposition (`Picture::through` false): from stop to stop of the caller's
//! (`Picture::advance`) by CF4 (Blanes & Moan 2006) of the same Ĥ₀ + D(u) -- the static group's parabola and each
//! tone's members at their phases at the two Gauss nodes --, two exponentials on the vector by Taylor's series
//! (`linalg::taylor_series`, the old engine's way: its terms through the first whose bound is within the share, not to
//! round-off). It is exact where nothing moves but for the series' truncation; with D it is of fourth order in the
//! stride's width, which is the caller's nodes' spacing -- a few radians of Ĥ₀'s bound at most --, so its error is a
//! share of D's own effect, not an error of Ĥ₀'s frequencies (`Picture::stridden`: the Magnus expansion's first terms a
//! fourth-order method leaves out, by the matrices' norms, a tone's derivatives with its beat's powers). Each
//! exponential's series is truncated where its remainder is within the kick's share (`units::QUADRATURE`, the share
//! the step's quadrature lays its nodes at, step.rs) of what the state forgets over its width: the state keeps an error
//! for 1/γ, γ the decay it is launched with -- a jump renews it --, so the exponentials' errors over that time add up to
//! the share whatever the step, as the old engine's truncation per waiting time did (NUMERICAL_SHARE x target x
//! dt/waiting); a state that forgets nothing is taken to round-off (`Picture::tolerance`). A step takes the strides
//! - where they cost less than the decomposition's route as it was last counted on this picture for as many states
//!   (its preparation and its states at its stops: `Picture::counted`), each with the states a jump's root is expected
//!   to read -- a stride each, against a state at a time through the decomposition (step.rs `pictured`) --, and the
//!   Magnus part of their error over the step is within the share: a small cluster -- a folded one, whose own
//!   frequencies over a step are a few radians -- on a few nodes -- the decomposition's QR sweeps are its n³ whatever
//!   the nodes, the strides a few n² a node --;
//! - towards a defective generator: a state taken through the eigenvectors rounds by κ(R) u (1 + |τ| × the decay
//!   rates' spread) relative to its norm (`linalg::Eigen`), and κ(R) grows without bound towards an exceptional point;
//!   the strides' rounding does not know κ. The rule is the rounding's, not a tuned number: the decomposition where a
//!   state through it rounds no more than through the exponential of Ĥ₀ over the step (`linalg::Chebyshev::rounding`),
//!   the strides otherwise;
//! - where the split takes an entry by no route of the decomposition (# The split: its following has no sum and its
//!   hops leave more than the following would): a step far too long for the change it holds, which the strides take
//!   bounded -- inaccurate, as any route is there; the check reads it.
//! Without the decomposition the forms' bandwidth is its bound 2(ρ + γ), a tone's members mixing the states'
//! frequencies by twice its beat besides, and the strides' nodes are laid at it.

use crate::linalg::{self, combine, C64, Mat, LANES};
use crate::model::assemble::laid;
use crate::units::{QUADRATURE, UNIT_ROUNDOFF};

/// The reals of one vector of rows of one column laid for the lanes: its real parts and its imaginary parts.
const PAIR: usize = 2 * LANES;

/// A matrix given row by row (`columns` numbers a row), laid for the lanes (`laid`: its rows over the lanes, as
/// `linalg::combine` takes its columns and gives its sums).
fn lay(matrix: &[C64], columns: usize, to: &mut [f64]) {
    for (row, numbers) in matrix.chunks_exact(columns).enumerate() {
        for (z, at) in numbers.iter().zip((laid(row, 0, columns)..).step_by(PAIR)) {
            (to[at], to[at + LANES]) = (z.re, z.im);
        }
    }
}

/// A laid matrix (`lay`) row by row.
fn unlay(from: &[f64], columns: usize, matrix: &mut [C64]) {
    for (row, numbers) in matrix.chunks_exact_mut(columns).enumerate() {
        for (z, at) in numbers.iter_mut().zip((laid(row, 0, columns)..).step_by(PAIR)) {
            *z = C64::new(from[at], from[at + LANES]);
        }
    }
}

/// `to` + p v of laid matrices (`lay`; their padding stays as it is: zero times anything adds nothing), v the
/// polynomial c + u a + u² b of three laid matrices one after another in `from` (`polynomial`).
fn add_turned(p: C64, from: &[f64], u: f64, to: &mut [f64]) {
    let length = to.len();
    for (at, chunk) in to.chunks_exact_mut(PAIR).enumerate() {
        for l in 0..LANES {
            let value = |part: usize| {
                let place = at * PAIR + part;
                (u * u).mul_add(from[2 * length + place], u.mul_add(from[length + place], from[place]))
            };
            let (re, im) = (value(l), value(LANES + l));
            chunk[l] += p.re * re - p.im * im;
            chunk[LANES + l] += p.re * im + p.im * re;
        }
    }
}

/// An entry of a laid matrix.
fn entry(matrix: &[f64], place: usize) -> C64 {
    C64::new(matrix[place], matrix[place + LANES])
}

/// An exact tone of the generator along a step's path (module documentation, # The generator; `Picture::separate`): its
/// beat ν (rad/µs), its place among the caller's candidates, and its two members' envelopes at the step's three nodes,
/// −h, 0 and h -- G⁺, at e^{iνu}, and G⁻, at e^{−iνu} (n × n each). After `prepare` each member's envelope as its nodes'
/// parabola C + uA + u²B, as given: C, A and B in the places of the nodes.
#[derive(Debug)]
pub struct Beat {
    pub beat: f64,
    pub candidate: usize,
    pub members: [[Mat; 3]; 2],
}

/// A class of the propagation's terms by their beat (module documentation, # Tones), in Ĥ₀'s eigenbasis: the static
/// group, a member of an exact tone, or a product of two members' terms that the following's second order makes. Its
/// memory is kept from step to step.
#[derive(Default)]
struct Class {
    /// Its beat ν (rad/µs; none of the static group's), and its partner: the class of the opposite beat, whose entry
    /// (b, a) turns as this one's (a, b) does, the other way round (the static group is its own).
    beat: f64,
    partner: usize,
    /// D's envelope C, A, B in the eigenbasis, row by row (n² each; of the first order's classes -- the static group's C
    /// is none, its middle being Ĥ₀), and laid for the lanes one after another.
    change: [Vec<C64>; 3],
    laid: Vec<f64>,
    /// The followed entries (n²), their 1/Ω, X's polynomial Y₀ + uY₁ + u²Y₂ on them (3n², each column by column),
    /// whether any entry of it is followed, and whether it is hopped at all (a product the second order makes nothing of
    /// is not).
    fast: Vec<bool>,
    inverses: Vec<C64>,
    follow: Vec<C64>,
    followed: bool,
    hops: bool,
    /// The following's second order at the nodes −h, h and 0 (laid, one after another).
    second: Vec<f64>,
    /// D₂ on the hopped entries: C, A, B row by row (n² each), and the hops' moments I_p (3n²).
    hop: [Vec<C64>; 3],
    moments: Vec<C64>,
    /// R(1 + Y₀) of the static group's, R Y₀ of a beating class's; R Y₁ and R Y₂: laid one after another, the matrices
    /// of a state at a time (`Picture::at`).
    nodes: Vec<f64>,
    /// Y at a node, column by column and laid: a product's right factor, and its left.
    now: Vec<C64>,
    across: Vec<f64>,
    /// Its phase at the time a hop, a rate or a launch is made at (`turn`).
    phase: C64,
}

impl Class {
    /// The class for n states, its memory zero.
    fn fit(&mut self, n: usize) {
        let (size, column) = (n * n, n.div_ceil(LANES) * PAIR);
        for (v, length) in self.change.iter_mut().chain(&mut self.hop).map(|v| (v, size)).chain([(&mut self.inverses, size), (&mut self.follow, 3 * size), (&mut self.moments, 3 * size), (&mut self.now, size)]) {
            v.clear();
            v.resize(length, C64::ZERO);
        }
        self.fast.clear();
        self.fast.resize(size, false);
        // Laid: zero beyond the rows, and kept so (every laid matrix is written on its rows alone, or as a sum of laid ones).
        for (v, columns) in [(&mut self.laid, 3 * n), (&mut self.second, 3 * n), (&mut self.nodes, 3 * n), (&mut self.across, n)] {
            v.clear();
            v.resize(columns * column, 0.0);
        }
    }

    /// Whether the class is fitted to n states.
    fn fits(&self, n: usize) -> bool {
        self.fast.len() == n * n
    }

    /// The class's phase e^{iνu} at the time u.
    fn turn(&self, u: f64) -> C64 {
        C64::cis(self.beat * u)
    }
}

/// A generator along a step's path (module documentation): filled by the caller with its static group -- Ĥ₀ at the
/// middle (`base`) and the group at the two ends (`ends`: at −h, at +h) -- and its exact tones (`beating`, `beats`),
/// then prepared (`prepare`) and launched from a state at a time of the step (`launch`: the step's start state, and a
/// jump's state at the jump); kept by a trajectory from step to step, its memory made once.
pub struct Picture {
    n: usize,
    /// Ĥ₀ (n × n; rad/µs), the caller's: kept as it is given.
    pub base: Mat,
    /// The static group at −h and at +h, the caller's; after `prepare` its slope A and curvature B as given.
    pub ends: [Mat; 2],
    /// The exact tones, the caller's (`beating`).
    pub beats: Vec<Beat>,
    beating: Vec<f64>,
    /// After `prepare` on a beating path, the static group's slope at the middle through the step's ends,
    /// (G₀(h) − G₀(−h))/2h, as the caller's matrices are laid out: the first order of the parabola it is taken for (the
    /// optical sector's along the path, step.rs).
    pub slope: Mat,
    /// A hop's exponent.
    matrix: Mat,
    /// Ĥ₀'s decomposition of its traceless part, and whether the state goes through it.
    basis: linalg::Eigen,
    through: bool,
    /// Whether the split took every entry by a route of the decomposition (`followed`): the step by the strides where not.
    routed: bool,
    /// Ĥ₀'s eigenvalues, the scalar part with them.
    lambda: Vec<C64>,
    /// The classes (module documentation, # Tones): the static group; each exact tone's two members, ⁺ and then ⁻; the
    /// products of two tones lo ≤ hi, their sum's members and, where lo < hi, their difference's. And the class of each
    /// product of a first-order class's change and another's following ((1 + 2T)², the change's class by row).
    classes: Vec<Class>,
    combined: Vec<usize>,
    /// The hops' moments' width, and the whole numbers' reciprocals 1/k their series take, as far as a series went (k
    /// its index; none at 0).
    width: f64,
    reciprocals: Vec<f64>,
    /// The exponential's work vectors (2n), a state's coordinates (n) and the pairs' turns (n).
    vectors: Vec<C64>,
    coordinates: Vec<C64>,
    turns: Vec<C64>,
    /// Work matrices (9n²): the static group's −D_s at the three nodes, column by column (the products' right factors);
    /// D₂ at the three nodes; X at a node, column by column; 1 + X's factors at a node; a beating class's −D_s.
    product: Vec<C64>,
    /// Matrices laid for the lanes (`lay`): R and R⁻¹; five work matrices (D at the three nodes -- the first a product's
    /// and R(1 + X(u))'s at a time besides --, a beating class's change at a node, and a spare); and a state.
    right: Vec<f64>,
    left: Vec<f64>,
    spare: [Vec<f64>; 5],
    lanes: Vec<f64>,
    /// Whether D is anything at all, and the step's half h.
    moving: bool,
    half: f64,
    /// The rate at which the state forgets its own errors (rad/µs): the decay it is launched with, at which a jump
    /// renews it (`prepare`; the strides' share, `tolerance`).
    forgets: f64,
    /// Whether any entry of any class is followed, and 1 + X factored (LU with partial pivoting, n² and its pivots) at
    /// the time `factored`: a launched state's way into g -- the following leaves the step's start's, a launch elsewhere
    /// makes its own.
    followed: bool,
    factors: Vec<C64>,
    pivots: Vec<usize>,
    factored: f64,
    /// The entries a hop takes (n²): those any class hops.
    hopped: Vec<bool>,
    /// The path: m and ṁ at its three stops (6n) -- the launch's time `origin`, `centre` = origin + `spacing` and the
    /// step's end, centre + spacing.
    path: Vec<C64>,
    origin: f64,
    centre: f64,
    spacing: f64,
    /// Without the decomposition: the frame's state at the launch and at the last two stops (3n) and their times.
    stops: Vec<C64>,
    times: [f64; 3],
    /// max|Ω| over the hopped entries of every class (rad/µs): what a hop's series is summed against.
    spread: f64,
    /// The bounds of Ĥ₀ less its scalar part (`linalg::bounds`), and the Frobenius norms of D's constants, slopes and
    /// curvatures as the hops take them, summed over the classes: what bounds a hop's exponent, and what a hop is priced
    /// by.
    bounds: [f64; 2],
    slopes: [f64; 3],
    /// The bandwidth of the forms ⟨x|·|x⟩ on a state propagated by Ĥ₀ and followed (rad/µs): max|λ_a* − λ_b| -- their
    /// frequencies and decay rates -- and the followed classes' beats' spread; without the decomposition its bound
    /// 2(ρ + γ) with the beats' twice.
    pub bandwidth: f64,
    /// The complex multiply-adds counted so far (`decide::Work` counts them four real ones each).
    pub work: f64,
    /// The decomposition's route as it was last counted on this picture, per count of states: its preparation and its
    /// price at the stops of its quadrature, and its price of a state at a time (`prepare`); none before it is taken.
    /// What the strides are weighed against -- the one thing a picture carries from one step to the next (`counts`).
    counted: Vec<Option<[f64; 2]>>,
}

impl Default for Picture {
    fn default() -> Picture {
        let none = || Mat::zeros(0, 0);
        Picture {
            n: 0,
            base: none(),
            ends: [none(), none()],
            beats: Vec::new(),
            beating: Vec::new(),
            slope: none(),
            matrix: none(),
            basis: linalg::Eigen::default(),
            through: false,
            routed: true,
            lambda: Vec::new(),
            classes: vec![Class::default()],
            combined: vec![0],
            width: f64::NAN,
            reciprocals: vec![f64::INFINITY],
            vectors: Vec::new(),
            coordinates: Vec::new(),
            turns: Vec::new(),
            product: Vec::new(),
            right: Vec::new(),
            left: Vec::new(),
            spare: Default::default(),
            lanes: Vec::new(),
            moving: false,
            half: 0.0,
            forgets: 0.0,
            followed: false,
            factors: Vec::new(),
            pivots: Vec::new(),
            factored: f64::NAN,
            hopped: Vec::new(),
            path: Vec::new(),
            origin: 0.0,
            centre: 0.0,
            spacing: 0.0,
            stops: Vec::new(),
            times: [0.0; 3],
            spread: 0.0,
            bounds: [0.0; 2],
            slopes: [0.0; 3],
            bandwidth: 0.0,
            work: 0.0,
            counted: Vec::new(),
        }
    }
}

/// How a term that turns x = |Ω| |h| between two of the step's nodes is taken (module documentation, # The split), the
/// changes over a half step being of the sizes `slope` = S_A, `curvature` = S_B and `constant` = S_C: followed (true)
/// where what its following leaves, (2/3)(S_A/x)³ + 2 S_C³ min(x, 1)/x³ -- the leading term of the following's orders,
/// a series in its size r, so over 1 − r -- and (4/15) S_B²/x, is no more than what its hops would,
/// x (S_C²/3 + S_A²/15 + S_B²/42) -- the two times x³ --, hopped (false) where it is more; a term that does not turn is
/// hopped. Where r ≥ 1 the following's series has no sum -- 1 + X no near identity --: the term is hopped where the
/// hops leave less than the following's leading term would, and taken by no route of the decomposition (None) where
/// they leave more -- the hops' own series is no better there: a pair that turns far between two nodes, hopped, turns
/// by its decay rates' difference too, e^{|Im Ω| 2h} over the step, its exponent without bound (`Picture::prepare`
/// takes the step by the strides then). r is the following's size on the entry, |Y| at the step's ends (`flag`).
fn followed(x: f64, [slope, curvature, constant]: [f64; 3], r: f64) -> Option<bool> {
    if !(x > 0.0) {
        return Some(false);
    }
    let (first, second, zeroth) = (slope * slope, curvature * curvature, constant * constant);
    let (leading, light, hops) = (2.0 / 3.0 * first * slope + 2.0 * zeroth * constant * x.min(1.0), 4.0 / 15.0 * second * (x * x), x.powi(4) * (zeroth / 3.0 + first / 15.0 + second / 42.0));
    if r < 1.0 {
        Some(leading / (1.0 - r) + light <= hops)
    } else {
        (leading + light > hops).then_some(false)
    }
}

/// The terms of ∫₀¹ t^p e^{zt} dt's series Σ_j z^j/(j!(j + p + 1)) at |z| ≤ x for an integral of the size `size` at
/// most: the least T with x^T/T! times the size within the unit round-off, the size no more than one -- the terms from T
/// on are within the round-off of the state the integral acts on, or of its own first term where that is more.
fn series(x: f64, size: f64) -> usize {
    let (mut terms, mut next) = (1, x * size.min(1.0));
    while next > UNIT_ROUNDOFF {
        terms += 1;
        next *= x / terms as f64;
    }
    terms
}

/// The quintic through three nodes at s = −1, 0, 1 (the Hermite interpolant of values and rates): the weights of the
/// values and of the rates d/ds at −1, 0, 1 -- with ℓ_i the nodes' Lagrange polynomials, (1 − 2ℓ_i′(s_i)(s − s_i)) ℓ_i²
/// and (s − s_i) ℓ_i².
fn quintic(s: f64) -> ([f64; 3], [f64; 3]) {
    let (minus, middle, plus) = (0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0));
    let (m2, p2) = (minus * minus, plus * plus);
    ([(1.0 + 3.0 * (s + 1.0)) * m2, middle * middle, (1.0 - 3.0 * (s - 1.0)) * p2], [(s + 1.0) * m2, s * middle * middle, (s - 1.0) * p2])
}

/// The Frobenius norm of a matrix row by row.
fn frobenius(m: &[C64]) -> f64 {
    m.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
}

/// A polynomial z₀ + u z₁ + u² z₂ of three matrices one after another (n² each, `size`), entry by entry.
fn polynomial(of: &[C64], size: usize, u: f64, out: &mut [C64]) {
    let (zero, rest) = of.split_at(size);
    let (one, two) = rest.split_at(size);
    for (((x, zero), one), two) in out.iter_mut().zip(zero).zip(one).zip(two) {
        *x = *zero + *one * u + *two * (u * u);
    }
}

/// Of a generator's two classes c ≤ p the first and the second, mutably (the same where c = p).
fn both(classes: &mut [Class], c: usize, p: usize) -> (&mut Class, Option<&mut Class>) {
    if c == p {
        (&mut classes[c], None)
    } else {
        let (low, high) = classes.split_at_mut(p);
        (&mut low[c], Some(&mut high[0]))
    }
}

/// The classes c and c + 1 a pair at ±ν, each the other's partner.
fn paired(classes: &mut [Class], c: usize, beat: f64) {
    (classes[c].beat, classes[c].partner) = (beat, c + 1);
    (classes[c + 1].beat, classes[c + 1].partner) = (-beat, c);
}

/// A term's following of its change C + uA + u²B at its combined frequency Ω, by its 1/Ω: the polynomial
/// Y₀ + uY₁ + u²Y₂ with ΩY − iẎ + C + uA + u²B = 0 (module documentation, # The state at any time of the step).
fn following(inverse: C64, c: C64, a: C64, b: C64) -> [C64; 3] {
    let two = -(b * inverse);
    let one = (C64::I * two * 2.0 - a) * inverse;
    [(C64::I * one - c) * inverse, one, two]
}

/// A class's entry k (its transpose's place t, X being held column by column) at the combined frequency Ω, `fast` or
/// hopped: its flag and, where followed, its 1/Ω and its following of the class's change on it (none of a product's).
fn take(class: &mut Class, k: usize, t: usize, omega: C64, fast: bool, size: usize) {
    class.fast[k] = fast;
    let mut x = [C64::ZERO; 3];
    if fast {
        class.inverses[k] = omega.recip();
        let [c, a, b] = [0, 1, 2].map(|j| class.change[j][k]);
        x = following(class.inverses[k], c, a, b);
        class.followed |= c != C64::ZERO || a != C64::ZERO || b != C64::ZERO;
    }
    for (j, x) in x.into_iter().enumerate() {
        class.follow[j * size + t] = x;
    }
}

/// The entries of the classes in `range` that are hopped at all, each with its partner's -- the entry (a, b) at
/// Ω = λ_a − λ_b + ν and its partner's (b, a) at −Ω (module documentation, # Tones) --, taken (`take`) by the split at
/// their turn |Ω| |h| and the changes' `sizes` (`followed`), each the following's convergence on its own -- r, the
/// larger of its two ways' |Y| at the step's ends, Y its following of the class's `envelope` on the entry (C, A, B:
/// a first-order class's change, a product's second order through the nodes) --; the largest |Ω| hopped into
/// `spread`. False where an entry is taken by no route of the decomposition (the rest is flagged as it would be, and
/// not used).
fn flag(classes: &mut [Class], range: std::ops::Range<usize>, lambda: &[C64], sizes: [f64; 3], envelope: &dyn Fn(&Class, usize) -> [C64; 3], h: f64, spread: &mut f64) -> bool {
    let size = |class: &Class, k: usize, omega: C64| {
        let [c, a, b] = envelope(class, k);
        let y = following(omega.recip(), c, a, b);
        [-h, h].into_iter().map(|u| (y[0] + y[1] * u + y[2] * (u * u)).abs()).fold(0.0, f64::max)
    };
    let mut routed = true;
    let n = lambda.len();
    for c in range {
        let (beat, partner) = (classes[c].beat, classes[c].partner);
        if partner < c || !classes[c].hops {
            continue;
        }
        for row in 0..n {
            for column in if partner == c { row } else { 0 }..n {
                let between = lambda[row] - lambda[column];
                let omega = if c == 0 { between } else { C64::new(between.re + beat, between.im) };
                let turn = omega.abs();
                let (k, t) = (row * n + column, column * n + row);
                let r = if turn > 0.0 { size(&classes[c], k, omega).max(size(&classes[partner], t, -omega)) } else { 0.0 };
                let split = followed(turn * h.abs(), sizes, r);
                routed &= split.is_some();
                let fast = split.unwrap_or(false);
                if !fast {
                    *spread = spread.max(turn);
                }
                let (one, other) = both(classes, c, partner);
                take(one, row * n + column, column * n + row, omega, fast, n * n);
                take(other.unwrap_or(one), column * n + row, row * n + column, -omega, fast, n * n);
            }
        }
    }
    routed
}

/// A hop's integral ∫₀ʷ e^{iΩt} E(u + t) dt of one entry over the width w from the time u (module documentation), the
/// envelope's constant (none where it has none), slope and curvature on the entry's moments I₀, I₁, I₂:
/// w [C I₀ + A (u I₀ + w I₁) + B (u² I₀ + 2uw I₁ + w² I₂)].
fn integral(constant: Option<C64>, a: C64, b: C64, [zero, one, two]: [C64; 3], u: f64, w: f64) -> C64 {
    let slope = a * (zero * u + one * w);
    (constant.map_or(slope, |c| c * zero + slope) + b * (zero * (u * u) + one * (2.0 * u * w) + two * (w * w))) * w
}

impl Picture {
    /// The picture for n states.
    pub fn fit(&mut self, n: usize) {
        if self.n != n {
            self.n = n;
            if self.counted.len() <= n {
                self.counted.resize(n + 1, None);
            }
            for m in [&mut self.base, &mut self.matrix, &mut self.slope].into_iter().chain(&mut self.ends) {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            let size = n * n;
            for (v, length) in [(&mut self.lambda, n), (&mut self.vectors, 2 * n), (&mut self.coordinates, n), (&mut self.turns, n), (&mut self.product, 9 * size), (&mut self.factors, size), (&mut self.path, 6 * n), (&mut self.stops, 3 * n)] {
                v.resize(length, C64::ZERO);
            }
            self.hopped.resize(size, false);
            self.pivots.resize(n, 0);
            for class in &mut self.classes {
                class.fit(n);
            }
            // A laid column: whole vectors of rows. Zero beyond the rows, and kept so: every laid matrix is written on
            // its rows alone, or as a sum of laid ones.
            let column = n.div_ceil(LANES) * PAIR;
            let matrices = [&mut self.right, &mut self.left].into_iter().chain(&mut self.spare).map(|v| (v, n));
            for (v, columns) in matrices.chain([(&mut self.lanes, 1)]) {
                v.clear();
                v.resize(columns * column, 0.0);
            }
        }
    }

    /// The step's exact tones out of the generator (module documentation, # Which beats are tones), the generator at the
    /// nodes given (`base`, `ends`): `candidates` gives the beats ν (rad/µs) of the caller's candidate tone pairs, each
    /// of members constant along the path; a candidate is a tone of the step of 2h where |ν| |h| > √5 -- its hops leave
    /// less than its change through the nodes would --, its members asked of `envelopes` (the candidate's place, G⁺ at
    /// e^{iνu}, G⁻ at e^{−iνu}), and its Hermitian part e^{iνu}P + e^{−iνu}P†, P = (G⁺ + G⁻†)/2, taken out of the
    /// generator at each node: what is left is the static group (`beats`: the tones). None where no candidate is so fast.
    /// The decay's beats stay with the static group: a hop's exponent is the first Magnus term, and a beating decay's
    /// part alone is no decay -- its sign turns with its beat --, so that over a hop it would grow the interaction
    /// picture's state by e^{|Γ_ν| w} where the static decay's factors hold the state (an overflow of m, not of x); a
    /// Hermitian tone's hops are unitary.
    pub fn separate(&mut self, half: f64, candidates: impl FnOnce(&mut Vec<f64>), mut envelopes: impl FnMut(usize, &mut Mat, &mut Mat)) {
        let n = self.n;
        self.beating.clear();
        candidates(&mut self.beating);
        let edge = 5.0_f64.sqrt() / half.abs();
        let count = self.beating.iter().filter(|beat| beat.abs() > edge).count();
        self.beats.truncate(count);
        while self.beats.len() < count {
            self.beats.push(Beat { beat: 0.0, candidate: 0, members: std::array::from_fn(|_| std::array::from_fn(|_| Mat::zeros(n, n))) });
        }
        let chosen = self.beating.iter().copied().enumerate().filter(|(_, beat)| beat.abs() > edge);
        for (tone, (candidate, beat)) in self.beats.iter_mut().zip(chosen) {
            (tone.beat, tone.candidate) = (beat, candidate);
            for m in tone.members.iter_mut().flatten() {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            let [plus, minus] = &mut tone.members;
            envelopes(candidate, &mut plus[1], &mut minus[1]);
            // Its Hermitian part: P = (G⁺ + G⁻†)/2 at e^{iνu}, P† at e^{−iνu}.
            let [given, middle, _] = &mut *plus;
            given.data.copy_from_slice(&middle.data);
            for a in 0..n {
                for b in 0..n {
                    middle.data[a * n + b] = (given.data[a * n + b] + minus[1].data[b * n + a].conj()) * 0.5;
                }
            }
            for a in 0..n {
                for b in 0..n {
                    minus[1].data[a * n + b] = plus[1].data[b * n + a].conj();
                }
            }
            for member in [&mut *plus, &mut *minus] {
                let [first, middle, last] = member;
                first.data.copy_from_slice(&middle.data);
                last.data.copy_from_slice(&middle.data);
            }
            let [start, end] = &mut self.ends;
            for (node, u) in [start, &mut self.base, end].into_iter().zip([-half, 0.0, half]) {
                let (p, q) = (C64::cis(beat * u), C64::cis(-beat * u));
                for ((z, g), h) in node.data.iter_mut().zip(&plus[1].data).zip(&minus[1].data) {
                    *z -= p * *g + q * *h;
                }
            }
        }
    }

    /// The decomposition's route as this picture last counted it, per count of states (`counted`): what it carries from
    /// one step to the next, the next step's route weighed against it (`prepare`).
    pub fn counts(&self) -> &[Option<[f64; 2]>] {
        &self.counted
    }

    /// A picture that has counted the decomposition's route so (`counts`), and holds nothing else: the next step's route
    /// is chosen as on the picture that counted it.
    pub fn counting(counted: &[Option<[f64; 2]>]) -> Picture {
        Picture { counted: counted.to_vec(), ..Picture::default() }
    }

    /// The states.
    pub fn states(&self) -> usize {
        self.n
    }

    /// Whether the state goes through the decomposition (module documentation, # The strides).
    pub fn decomposed(&self) -> bool {
        self.through
    }


    /// The classes of a step's T tones (module documentation, # Tones), their beats and partners: the static group; tone
    /// m's members 1 + 2m (⁺, at ν_m) and 2 + 2m (⁻, at −ν_m); and for the tones lo ≤ hi their sum's two members,
    /// ±(ν_lo + ν_hi), and where lo < hi their difference's, ±(ν_lo − ν_hi). And the class each product of a first-order
    /// class's change and another's following falls in: of the static group with itself, the static group; of a member
    /// with the static group, that member; of a tone's two members, the static group (their beats cancel); of two members
    /// otherwise, their sum's or their difference's.
    fn classify(&mut self) {
        let (n, tones) = (self.n, self.beats.len());
        let order = 1 + 2 * tones;
        self.classes.resize_with(order + 2 * tones * tones, Class::default);
        for class in self.classes.iter_mut().filter(|class| !class.fits(n)) {
            class.fit(n);
        }
        let (classes, combined) = (&mut self.classes, &mut self.combined);
        combined.clear();
        combined.resize(order * order, 0);
        (classes[0].beat, classes[0].partner) = (0.0, 0);
        let member = |m: usize, sign: usize| 1 + 2 * m + sign;
        for (m, tone) in self.beats.iter().enumerate() {
            paired(classes, member(m, 0), tone.beat);
            for t in [member(m, 0), member(m, 1)] {
                (combined[t], combined[t * order]) = (t, t);
            }
        }
        let mut next = order;
        for lo in 0..tones {
            for hi in lo..tones {
                let (a, b) = (self.beats[lo].beat, self.beats[hi].beat);
                paired(classes, next, a + b);
                for sign in 0..2 {
                    for (x, y) in [(member(lo, sign), member(hi, sign)), (member(hi, sign), member(lo, sign))] {
                        combined[x * order + y] = next + sign;
                    }
                }
                next += 2;
                if lo < hi {
                    paired(classes, next, a - b);
                }
                for sign in 0..2 {
                    let (x, y) = (member(lo, sign), member(hi, 1 - sign));
                    let class = if lo == hi { 0 } else { next + sign };
                    (combined[x * order + y], combined[y * order + x]) = (class, class);
                }
                if lo < hi {
                    next += 2;
                }
            }
        }
    }

    /// Prepares the picture of a step of 2h from `base` and, unless the generator is `still` along the path (then they
    /// are not read, and there are no tones), `ends` and the tones (`beats`): the changes, the route of the state
    /// (module documentation, # The strides) and on the decomposition's route Ĥ₀'s decomposition, the changes in its
    /// eigenbasis and their following (`follow`). `stops` gives the stops a step takes at a bandwidth of its forms (its
    /// quadrature's nodes, its middle and its end), `roots` the states a jump's root is expected to read besides
    /// (step.rs `pictured`): what each route is priced at -- a stride per state, or a state at a time through the
    /// decomposition; `forgets` the decay of the state the step is launched with (rad/µs; the strides' share,
    /// `tolerance`). `work` begins anew with what this counted.
    pub fn prepare(&mut self, half: f64, still: bool, stops: impl Fn(f64) -> usize, roots: f64, forgets: f64) {
        assert!(!still || self.beats.is_empty(), "exact tones on a still path");
        let n = self.n;
        (self.work, self.width, self.moving, self.forgets) = (0.0, f64::NAN, false, forgets.max(0.0));
        (self.half, self.followed, self.spread, self.factored, self.routed) = (half, false, 0.0, f64::NAN, true);
        self.classify();
        if !still {
            let [minus, plus] = &mut self.ends;
            let (first, second) = (0.5 / half, 0.5 / (half * half));
            for ((m, p), c) in minus.data.iter_mut().zip(plus.data.iter_mut()).zip(&self.base.data) {
                let (a, b) = ((*p - *m) * first, (*p + *m - *c * 2.0) * second);
                (*m, *p) = (a, b);
                self.moving |= a != C64::ZERO || b != C64::ZERO;
            }
            self.slope.data.copy_from_slice(&minus.data);
            self.work += (2 * n * n) as f64;
            // A tone's members: their nodes' parabola, C the middle's value.
            for [minus, middle, plus] in self.beats.iter_mut().flat_map(|tone| &mut tone.members) {
                for ((m, c), p) in minus.data.iter_mut().zip(middle.data.iter_mut()).zip(plus.data.iter_mut()) {
                    let (a, b) = ((*p - *m) * first, (*p + *m - *c * 2.0) * second);
                    (*m, *p) = (*c, b);
                    *c = a;
                    self.moving |= *m != C64::ZERO || a != C64::ZERO || b != C64::ZERO;
                }
                self.work += (2 * n * n) as f64;
            }
        }
        self.matrix.data.copy_from_slice(&self.base.data);
        let scalar = self.matrix.trace() / n as f64;
        for i in 0..n {
            self.matrix[(i, i)] -= scalar;
        }
        let linalg::Bounds { rho, gamma, .. } = linalg::bounds(&self.matrix, &mut self.vectors);
        self.bounds = [rho, gamma];
        // The strides' price at their nodes, where their error is within the share, against the decomposition's route
        // as last counted (the changes as given: the strides take them so). A tone's members mix the states' frequencies
        // by its beat: the strides' bandwidth with the beats' twice.
        self.slopes = self.given();
        let bound = 2.0 * (rho + gamma) + 2.0 * self.beats.iter().map(|tone| tone.beat.abs()).fold(0.0, f64::max);
        let strides = stops(bound);
        let strided = self.strided(strides) * (1.0 + roots / strides as f64);
        let cheaper = matches!((self.stridden(strides) <= QUADRATURE, self.counted[n]), (true, Some([made, state])) if strided < made + roots * state);
        let tau = 2.0 * half.abs();
        let exponential = linalg::Chebyshev::plan(rho * tau, gamma * tau).rounding(n);
        self.through = !cheaper && self.basis.decompose(&self.matrix) && {
            let decay = |pick: fn(f64, f64) -> f64, from: f64| self.basis.values.iter().map(|l| l.im).fold(from, pick);
            self.basis.condition * UNIT_ROUNDOFF * (1.0 + tau * (decay(f64::max, f64::NEG_INFINITY) - decay(f64::min, f64::INFINITY))) <= exponential
        };
        let e = &self.basis;
        match self.through {
            true => {
                for (l, value) in self.lambda.iter_mut().zip(&e.values) {
                    *l = *value + scalar;
                }
                // |λ_a* − λ_b| is the pair's whichever way round.
                self.bandwidth = self.lambda.iter().enumerate().flat_map(|(a, first)| self.lambda[a..].iter().map(move |second| (first.conj() - *second).abs())).fold(0.0, f64::max);
                self.work += e.work;
                lay(&e.vectors.data, n, &mut self.right);
                lay(&e.inverse.data, n, &mut self.left);
                if self.moving {
                    // Each change M ← (R⁻¹M)R: two products of n³, a product's right factor column by column -- the static
                    // group's A and B, each member's C, A and B.
                    let block = self.right.len();
                    let (left, columns, product) = (&self.left, &mut self.product[..n * n], &mut self.spare[0]);
                    let mut basis = |given: &Mat, laid: &mut [f64], change: &mut [C64]| {
                        for (column, numbers) in columns.chunks_exact_mut(n).enumerate() {
                            for (z, row) in numbers.iter_mut().zip(given.data.chunks_exact(n)) {
                                *z = row[column];
                            }
                        }
                        combine(false, columns, n, left, product, n, 0);
                        combine(false, e.columns(), n, product, laid, n, 0);
                        unlay(laid, n, change);
                    };
                    let group = &mut self.classes[0];
                    for (j, given) in self.ends.iter().enumerate() {
                        basis(given, &mut group.laid[(1 + j) * block..(2 + j) * block], &mut group.change[1 + j]);
                    }
                    for (m, tone) in self.beats.iter().enumerate() {
                        for (sign, member) in tone.members.iter().enumerate() {
                            let class = &mut self.classes[1 + 2 * m + sign];
                            for (j, given) in member.iter().enumerate() {
                                basis(given, &mut class.laid[j * block..(j + 1) * block], &mut class.change[j]);
                            }
                        }
                    }
                    self.work += ((4 + 12 * self.beats.len()) * n * n * n) as f64;
                }
            }
            false => self.bandwidth = bound,
        }
        if self.through && self.moving {
            self.follow();
            // An entry no route of the decomposition takes (`followed`): the step by the strides, at their bandwidth.
            if !self.routed {
                (self.through, self.bandwidth) = (false, bound);
            }
        }
        if self.through {
            if self.moving {
                self.slopes = self.hopping();
                // A followed class moves the state's frequencies by its beat: the forms' bandwidth with the beats' spread;
                // and a tone's forms are summed at its beat (step.rs `Sums`), e^{±iνu} on the state's pairs: their combined
                // frequencies, |λ_a* − λ_b ± ν|.
                let (low, high) = self.classes.iter().filter(|class| class.followed).fold((0.0_f64, 0.0_f64), |(low, high), class| (low.min(class.beat), high.max(class.beat)));
                let lambda = &self.lambda;
                let combined = self.beats.iter().flat_map(|tone| [tone.beat, -tone.beat]).flat_map(|beat| lambda.iter().enumerate().flat_map(move |(a, first)| lambda[a..].iter().map(move |second| (first.conj() - *second + C64::real(beat)).abs()))).fold(self.bandwidth, f64::max);
                self.bandwidth = combined + high - low;
            }
            self.counted[n] = Some([self.work + self.price(stops(self.bandwidth)), self.state_price()]);
        }
    }

    /// The Frobenius norms of D's constants, slopes and curvatures as given, summed over the static group and the tones'
    /// members (none where nothing moves): what bounds a stride's exponent.
    fn given(&self) -> [f64; 3] {
        if !self.moving {
            return [0.0; 3];
        }
        let mut norms = [0.0, frobenius(&self.ends[0].data), frobenius(&self.ends[1].data)];
        for member in self.beats.iter().flat_map(|tone| &tone.members) {
            for (norm, m) in norms.iter_mut().zip(member) {
                *norm += frobenius(&m.data);
            }
        }
        norms
    }

    /// The same of D₂ as the hops take it, summed over the classes hopped (the static group's constant is none without
    /// tones).
    fn hopping(&self) -> [f64; 3] {
        let group = &self.classes[0];
        let mut norms = [if self.beats.is_empty() { 0.0 } else { frobenius(&group.hop[0]) }, frobenius(&group.hop[1]), frobenius(&group.hop[2])];
        for class in self.classes[1..].iter().filter(|class| class.hops) {
            for (norm, m) in norms.iter_mut().zip(&class.hop) {
                *norm += frobenius(m);
            }
        }
        norms
    }

    /// The strides' error over the step at `stops` stops of its mean spacing w, relative to the state's norm (module
    /// documentation, # The strides): per stride the first terms the Magnus expansion of Ĥ(u_c + t) = H + tH′ + t²H″/2 +
    /// t³H‴/6 + t⁴H⁗/24 leaves out at the fourth order -- w⁵ ([H′, H″/2]/240, [H, [H, H″/2]]/360, [H′, [H, H′]]/240 and
    /// [H, [H, [H, H′]]]/720; Blanes, Casas, Oteo & Ros 2009, the expansion's sixth-order truncation), and where a tone
    /// beats the terms of H‴ and H⁗, [H, H‴/6]/80 and what the two Gauss nodes leave of ∫H, H⁗/4320 --, by the norms (a
    /// commutator's at most twice its factors') at the step's end, Ĥ₀'s bounds with D's there: of the static group its
    /// parabola's, of a member e^{iνu}(C + uA + u²B) its derivatives' bounds. Summed over the strides. None where nothing
    /// moves: the strides are then exact.
    fn stridden(&self, stops: usize) -> f64 {
        if !self.moving {
            return 0.0;
        }
        let h = self.half.abs();
        let w = 2.0 * h / stops as f64;
        let (a, b) = (frobenius(&self.ends[0].data), frobenius(&self.ends[1].data));
        let (mut size, mut first, mut curvature, mut third, mut fourth) = (self.bounds[0] + self.bounds[1] + h * a + h * h * b, a + 2.0 * h * b, b, 0.0, 0.0);
        for tone in &self.beats {
            let nu = tone.beat.abs();
            for member in &tone.members {
                let [c, slope, bend] = [0, 1, 2].map(|j| frobenius(&member[j].data));
                // The envelope's bound at the step's end and its first two derivatives'.
                let (e, e1, e2) = (c + h * slope + h * h * bend, slope + 2.0 * h * bend, 2.0 * bend);
                size += e;
                first += nu * e + e1;
                curvature += 0.5 * nu * nu * e + nu * e1 + 0.5 * e2;
                third += nu * nu * nu * e + 3.0 * nu * nu * e1 + 3.0 * nu * e2;
                fourth += nu.powi(4) * e + 4.0 * nu.powi(3) * e1 + 6.0 * nu * nu * e2;
            }
        }
        let terms = first * curvature / 120.0 + size * size * curvature / 90.0 + first * first * size / 60.0 + size * size * size * first / 90.0 + fourth / 4320.0 + size * third / 240.0;
        stops as f64 * w.powi(5) * terms
    }

    /// The following (module documentation, # The state at any time of the step; # Tones), class by class: the followed
    /// entries (`followed`, at the sizes of every class's changes together), their following X, and in the place of the
    /// changes D₂ on the hopped entries.
    /// - X's first order from each first-order class's change, entry by entry at its combined frequency.
    /// - At the nodes the second order D X − X D_s, product by product of a class's change and another's following, each
    ///   into the class of their beats' sum. Its fast part F is followed in its turn (X's second order: the same
    ///   polynomial of F through the three nodes -- the static group's alone is zero at the middle, as its D is).
    /// - At the nodes D₂ whole, (1 + X) D₂ = D_s + D X − F by the factors of 1 + X there (those at −h kept: the start
    ///   state's way into g). Of it each beating class's part to the second order -- its change and its second order on
    ///   its hopped entries, through the three nodes --, and the rest the static group's, through the three nodes (zero
    ///   at the middle where there are no tones).
    fn follow(&mut self) {
        let (n, h) = (self.n, self.half);
        let size = n * n;
        let tones = self.beats.len();
        let order = 1 + 2 * tones;
        let mut classes = std::mem::take(&mut self.classes);
        for (c, class) in classes.iter_mut().enumerate() {
            (class.followed, class.hops) = (false, c < order);
        }
        // The sizes of the changes over a half step, every first-order class's entries together, each pair once (the mean
        // of its two ways round): the slopes', the curvatures' and the tones' constants (a state with itself in the static
        // group is no pair).
        let mut sums = [0.0_f64; 3];
        for (c, class) in classes[..order].iter().enumerate() {
            for (j, sum) in sums.iter_mut().enumerate().skip(usize::from(c == 0)) {
                for (k, z) in class.change[j].iter().enumerate() {
                    if c > 0 || k / n != k % n {
                        *sum += z.norm_sqr();
                    }
                }
            }
        }
        let size_of = |sum: f64| (0.5 * sum).sqrt();
        let sizes = [size_of(sums[1]) * (h * h), size_of(sums[2]) * (h * h * h).abs(), size_of(sums[0]) * h.abs()];
        let mut spread = self.spread;
        if !flag(&mut classes, 0..order, &self.lambda, sizes, &|class, k| [0, 1, 2].map(|j| class.change[j][k]), h, &mut spread) {
            self.routed = false;
            return self.finish(classes, spread);
        }
        // Per entry the sizes' squares and the following: one and three of the static group's, three and four of a
        // member's.
        self.work += ((4 + 7 * (order - 1)) * size) as f64;
        self.followed = classes.iter().any(|class| class.followed);
        if !self.followed {
            for class in &mut classes[..order] {
                for (hop, change) in class.hop.iter_mut().zip(&class.change) {
                    hop.copy_from_slice(change);
                }
            }
            return self.finish(classes, spread);
        }
        let (first, second) = (0.5 / h, 0.5 / (h * h));
        // The nodes −h, h and, where tones put a constant into D, 0.
        let (count, times) = (if tones > 0 { 3 } else { 2 }, [-h, h, 0.0]);
        let block = self.right.len();
        for class in classes.iter_mut() {
            class.second[..count * block].fill(0.0);
        }
        let (slows, rest) = self.product.split_at_mut(3 * size);
        let (wholes, rest) = rest.split_at_mut(3 * size);
        let (x, rest) = rest.split_at_mut(size);
        let (factors, other) = rest.split_at_mut(size);
        let other = &mut other[..size];
        let (changes, rest) = self.spare.split_at_mut(3);
        let change = &mut rest[0];
        // The products beyond the static group's with itself.
        let mut made = 0;
        for (j, &u) in times[..count].iter().enumerate() {
            // Each followed first-order class's Y at u, column by column and laid.
            for class in classes[..order].iter_mut().filter(|class| class.followed) {
                polynomial(&class.follow, size, u, &mut class.now);
                for column in 0..n {
                    for row in 0..n {
                        let (t, place) = (column * n + row, laid(row, column, n));
                        (class.across[place], class.across[place + LANES]) = (class.now[t].re, class.now[t].im);
                    }
                }
            }
            // The static group's change at u, laid, and −D_s column by column: none at the middle.
            let (d, slow) = (&mut changes[j], &mut slows[j * size..(j + 1) * size]);
            if j < 2 {
                let group = &classes[0];
                for ((d, a), b) in d.iter_mut().zip(&group.laid[block..2 * block]).zip(&group.laid[2 * block..]) {
                    *d = *a * u + *b * (u * u);
                }
                for column in 0..n {
                    for row in 0..n {
                        let k = row * n + column;
                        slow[column * n + row] = if group.fast[k] { C64::ZERO } else { -(group.change[1][k] * u + group.change[2][k] * (u * u)) };
                    }
                }
            } else {
                d.fill(0.0);
                slow.fill(C64::ZERO);
            }
            // The products D_t Y_t′ − Y_t′ D_t^s, each into the class of its beats' sum.
            for t in (0..order).filter(|&t| t > 0 || j < 2) {
                if t > 0 {
                    let class = &classes[t];
                    for (i, z) in change.iter_mut().enumerate() {
                        *z = class.laid[i] + class.laid[block + i] * u + class.laid[2 * block + i] * (u * u);
                    }
                    for column in 0..n {
                        for row in 0..n {
                            let k = row * n + column;
                            other[column * n + row] = if class.fast[k] { C64::ZERO } else { -(class.change[0][k] + class.change[1][k] * u + class.change[2][k] * (u * u)) };
                        }
                    }
                }
                let (d, slow): (&[f64], &[C64]) = if t == 0 { (&changes[j], &slows[j * size..(j + 1) * size]) } else { (&*change, &*other) };
                for following in 0..order {
                    if !classes[following].followed {
                        continue;
                    }
                    let into = self.combined[t * order + following];
                    let mut out = std::mem::take(&mut classes[into].second);
                    let at = &mut out[j * block..(j + 1) * block];
                    combine(true, &classes[following].now, n, d, at, n, 0);
                    combine(true, slow, n, &classes[following].across, at, n, 0);
                    (classes[into].second, classes[into].hops) = (out, true);
                    made += usize::from(t > 0 || following > 0);
                }
            }
        }
        // F, each class's fast part of its second order through the three nodes, followed -- a product's entries taken
        // first at its own combined frequency, its following none before.
        let all = classes.len();
        // A product's envelope on an entry: its second order's constant, slope and curvature through the nodes.
        let product = |class: &Class, k: usize| {
            let place = laid(k / n, k % n, n);
            let (minus, plus) = (entry(&class.second, place), entry(&class.second, block + place));
            let middle = if tones > 0 { entry(&class.second, 2 * block + place) } else { C64::ZERO };
            [middle, (plus - minus) * first, (plus + minus - middle * 2.0) * second]
        };
        if !flag(&mut classes, order..all, &self.lambda, sizes, &product, h, &mut spread) {
            self.routed = false;
            return self.finish(classes, spread);
        }
        for class in classes.iter_mut().filter(|class| class.hops) {
            for row in 0..n {
                for column in 0..n {
                    let k = row * n + column;
                    if class.fast[k] {
                        let place = laid(row, column, n);
                        let (minus, plus) = (entry(&class.second, place), entry(&class.second, block + place));
                        let middle = if tones > 0 { entry(&class.second, 2 * block + place) } else { C64::ZERO };
                        for (j, x) in following(class.inverses[k], middle, (plus - minus) * first, (plus + minus - middle * 2.0) * second).into_iter().enumerate() {
                            class.follow[j * size + column * n + row] += x;
                        }
                        class.followed |= minus != C64::ZERO || plus != C64::ZERO || middle != C64::ZERO;
                    }
                }
            }
        }
        // D₂ at the nodes: (1 + X) D₂ = D_s + D X − F.
        let mut spare = [0; crate::model::STATES];
        for (j, &u) in times[..count].iter().enumerate() {
            // X at u, column by column: every class it follows, each at its phase.
            polynomial(&classes[0].follow, size, u, x);
            for class in classes[1..].iter().filter(|class| class.followed) {
                let p = class.turn(u);
                polynomial(&class.follow, size, u, other);
                for (x, y) in x.iter_mut().zip(&*other) {
                    *x += p * *y;
                }
            }
            // −F on the fast entries and D_s on the slow ones, each class at its phase; then D X onto them, D whole.
            let mut out = std::mem::take(&mut classes[0].second);
            let side = &mut out[j * block..(j + 1) * block];
            let slow = &slows[j * size..(j + 1) * size];
            for row in 0..n {
                for column in 0..n {
                    let place = laid(row, column, n);
                    (side[place], side[place + LANES]) = if classes[0].fast[row * n + column] { (-side[place], -side[place + LANES]) } else { (-slow[column * n + row].re, -slow[column * n + row].im) };
                }
            }
            let d = &mut changes[j];
            for (c, class) in classes.iter().enumerate().skip(1).filter(|(_, class)| class.hops) {
                let p = class.turn(u);
                for row in 0..n {
                    for column in 0..n {
                        let (k, place) = (row * n + column, laid(row, column, n));
                        let z = if class.fast[k] {
                            -(p * entry(&class.second, j * block + place))
                        } else if c < order {
                            p * (class.change[0][k] + class.change[1][k] * u + class.change[2][k] * (u * u))
                        } else {
                            continue;
                        };
                        side[place] += z.re;
                        side[place + LANES] += z.im;
                    }
                }
                if c < order {
                    add_turned(p, &class.laid, u, d);
                }
            }
            combine(true, x, n, d, side, n, 0);
            let whole = &mut wholes[j * size..(j + 1) * size];
            unlay(side, n, whole);
            classes[0].second = out;
            let (f, pivots) = if j == 0 { (&mut self.factors[..], &mut self.pivots[..]) } else { (&mut factors[..], &mut spare[..n]) };
            for (row, numbers) in f.chunks_exact_mut(n).enumerate() {
                for (z, column) in numbers.iter_mut().zip(x.chunks_exact(n)) {
                    *z = column[row];
                }
                numbers[row] += C64::ONE;
            }
            linalg::factor(f, pivots, n);
            linalg::solve(f, pivots, n, whole, n);
        }
        // Each beating class's D₂ to the second order on its hopped entries: its change, and its second order through the
        // three nodes.
        for (c, class) in classes.iter_mut().enumerate().skip(1).filter(|(_, class)| class.hops) {
            for row in 0..n {
                for column in 0..n {
                    let (k, place) = (row * n + column, laid(row, column, n));
                    let mut z = [C64::ZERO; 3];
                    if !class.fast[k] {
                        let (minus, plus, middle) = (entry(&class.second, place), entry(&class.second, block + place), entry(&class.second, 2 * block + place));
                        z = [middle, (plus - minus) * first, (plus + minus - middle * 2.0) * second];
                        if c < order {
                            for (z, change) in z.iter_mut().zip(&class.change) {
                                *z += change[k];
                            }
                        }
                    }
                    for (hop, z) in class.hop.iter_mut().zip(z) {
                        hop[k] = z;
                    }
                }
            }
        }
        // The static group's: what the solve leaves at the nodes less what the beating classes hold there, through the
        // three nodes.
        let (group, beating) = classes.split_first_mut().expect("the static group");
        for (j, &u) in times[..count].iter().enumerate() {
            for class in beating.iter().filter(|class| class.hops) {
                let p = class.turn(u);
                for (k, left) in wholes[j * size..(j + 1) * size].iter_mut().enumerate() {
                    if !class.fast[k] {
                        *left -= p * (class.hop[0][k] + class.hop[1][k] * u + class.hop[2][k] * (u * u));
                    }
                }
            }
        }
        for k in 0..size {
            let mut z = [C64::ZERO; 3];
            if !group.fast[k] {
                let (minus, plus) = (wholes[k], wholes[size + k]);
                z = if tones > 0 {
                    let middle = wholes[2 * size + k];
                    [middle, (plus - minus) * first, (plus + minus - middle * 2.0) * second]
                } else {
                    [C64::ZERO, (plus - minus) * first, (plus + minus) * second]
                };
            }
            for (hop, z) in group.hop.iter_mut().zip(z) {
                hop[k] = z;
            }
        }
        self.factored = -h;
        // A state at a time (`at`) is R(1 + X(u)) on its coordinates: R(1 + Y₀), RY₁ and RY₂ of the static group, RY₀, RY₁
        // and RY₂ of each beating class it follows.
        let matrix = self.right.len();
        group.nodes[..matrix].copy_from_slice(&self.right);
        for (j, (x, out)) in group.follow.chunks_exact(size).zip(group.nodes.chunks_exact_mut(matrix)).enumerate() {
            combine(j == 0, x, n, &self.right, out, n, 0);
        }
        for class in beating.iter_mut().filter(|class| class.followed) {
            for (x, out) in class.follow.chunks_exact(size).zip(class.nodes.chunks_exact_mut(matrix)) {
                combine(false, x, n, &self.right, out, n, 0);
            }
        }
        // Nine products of n³ (D X and X D_s at the two ends, D X again at the two ends, the three matrices of a state
        // at a time), two factorizations of 1 + X (n³/3) each with its n columns solved (n³), and per entry D at the
        // two ends (2 each), X at a time four times (2 each) and −D_s at the two ends (2 each): 16. Of the tones: the
        // products beyond the static group's with itself (two of n³ each), the middle's solve (D X, a factorization and
        // its columns), per beating class hopped and node X, the right side and D (six numbers an entry) and its hop's
        // envelope (eight), and three products of n³ a beating class followed.
        let (hopped, followers) = (beating.iter().filter(|class| class.hops).count(), beating.iter().filter(|class| class.followed).count());
        let middle = if tones > 0 { n * size + n * size / 3 + n * size } else { 0 };
        self.work += (9 * n * size + 2 * (n * size / 3 + n * size) + 16 * size) as f64;
        self.work += (2 * made * n * size + middle + hopped * (6 * count + 8) * size + 3 * followers * n * size) as f64;
        self.finish(classes, spread);
    }

    /// The following's classes into their place, the largest |Ω| hopped and the entries any class hops.
    fn finish(&mut self, classes: Vec<Class>, spread: f64) {
        self.spread = spread;
        for (k, hopped) in self.hopped.iter_mut().enumerate() {
            *hopped = classes.iter().any(|class| class.hops && !class.fast[k]);
        }
        self.classes = classes;
    }

    /// The bound of a hop's exponent from the time u over the width w: ∫‖D₂‖ at the norms of its constants, slopes and
    /// curvatures, D₂(t) = C + tA + t²B at the farther end of the hop from the middle.
    fn moved(&self, u: f64, w: f64) -> f64 {
        let far = u.abs().max((u + w).abs());
        (self.slopes[0] + far * self.slopes[1] + far * far * self.slopes[2]) * w.abs()
    }

    /// A hop's exponent ∫₀ʷ e^{iω_ab t}(D₂)_ab(u + t) dt into `matrix` (module documentation), from the time u over the
    /// width w: of each class hopped on the entry, at its phase e^{iνu}, w [C I₀ + A (u I₀ + w I₁) + B (u² I₀ + 2uw I₁ +
    /// w² I₂)] with the entry's moments I_p = ∫₀¹ s^p e^{zs} ds = Σ_j z^j/(j!(j + p + 1)), z = iΩw at its combined
    /// frequency Ω = ω_ab + ν, each entry's series as far as its own |z| and the exponent's bound ask (`series`). The
    /// moments are a width's: made once for the hops of one width.
    fn exponent(&mut self, u: f64, w: f64) {
        let n = self.n;
        let size = n * n;
        if w != self.width {
            // An entry and its partner's have z and −z: the series' even and odd parts, summed once for the two.
            let bound = self.moved(u, w);
            let mut summed = 0;
            let mut classes = std::mem::take(&mut self.classes);
            for c in 0..classes.len() {
                let (beat, partner) = (classes[c].beat, classes[c].partner);
                if partner < c || !classes[c].hops {
                    continue;
                }
                let (one, mut other) = both(&mut classes, c, partner);
                for a in 0..n {
                    for b in if partner == c { a } else { 0 }..n {
                        let k = a * n + b;
                        if one.fast[k] {
                            continue;
                        }
                        let between = self.lambda[a] - self.lambda[b];
                        let z = C64::I * (if c == 0 { between } else { C64::new(between.re + beat, between.im) }) * w;
                        let terms = series(z.abs(), bound);
                        while self.reciprocals.len() < terms + 3 {
                            self.reciprocals.push(1.0 / self.reciprocals.len() as f64);
                        }
                        let (mut power, mut parts) = (C64::ONE, [[C64::ZERO; 3]; 2]);
                        for j in 0..terms {
                            // 1/(j + p + 1) for the moment p.
                            let inverse = &self.reciprocals[j + 1..j + 4];
                            for (moment, inverse) in parts[j % 2].iter_mut().zip(inverse) {
                                *moment += power * *inverse;
                            }
                            power = power * z * inverse[0];
                        }
                        for p in 0..3 {
                            one.moments[p * size + k] = parts[0][p] + parts[1][p];
                            other.as_deref_mut().unwrap_or(&mut *one).moments[p * size + b * n + a] = parts[0][p] - parts[1][p];
                        }
                        summed += terms;
                    }
                }
            }
            self.classes = classes;
            self.width = w;
            self.work += (4 * summed) as f64;
        }
        for class in &mut self.classes {
            class.phase = class.turn(u);
        }
        let tones = !self.beats.is_empty();
        let (group, beating) = self.classes.split_first().expect("the static group");
        let moments = |class: &Class, k: usize| [0, 1, 2].map(|p| class.moments[p * size + k]);
        for (k, z) in self.matrix.data.iter_mut().enumerate() {
            *z = if group.fast[k] { C64::ZERO } else { integral(tones.then_some(group.hop[0][k]), group.hop[1][k], group.hop[2][k], moments(group, k), u, w) };
            for class in beating.iter().filter(|class| class.hops && !class.fast[k]) {
                *z += class.phase * integral(Some(class.hop[0][k]), class.hop[1][k], class.hop[2][k], moments(class, k), u, w);
            }
        }
        self.work += (4 * size * self.classes.iter().filter(|class| class.hops).count()) as f64;
    }

    /// `matrix`'s hopped entries turned by e^{iω_ab s}: from an entry's own time to the path's, s = u − u₀ from the
    /// launch. A hopped entry's combined frequency turns less than the split over |h| (module documentation, # The
    /// split), its decay rates' difference with it, so the factor is bounded whatever the states' own decays over s are:
    /// the turn by the two states' phases, the modulus by the rates' difference.
    fn turn(&mut self, s: f64) {
        if s == 0.0 {
            return;
        }
        let n = self.n;
        for (turn, l) in self.turns.iter_mut().zip(&self.lambda) {
            *turn = C64::cis(l.re * s);
        }
        for a in 0..n {
            for b in 0..n {
                if a != b && self.hopped[a * n + b] {
                    self.matrix.data[a * n + b] *= self.turns[a] * self.turns[b].conj() * ((self.lambda[b].im - self.lambda[a].im) * s).exp();
                }
            }
        }
        self.work += (2 * n * n) as f64;
    }

    /// One hop of m from the time u over the width w, either sign (module documentation): m ← e^{−iJ} m.
    fn hop(&mut self, m: &mut [C64], u: f64, w: f64) {
        let n = self.n;
        self.exponent(u, w);
        self.turn(u - self.origin);
        let plan = linalg::exp_apply(&mut self.matrix, C64::new(0.0, -1.0), m, &mut self.vectors);
        self.work += (plan.parts * plan.terms * n * n) as f64;
    }

    /// ṁ = −i D̃₂(u) m at the time u, into `rate`: each class hopped on an entry at its phase.
    fn rate(&mut self, u: f64, m: &[C64], rate: &mut [C64]) {
        let n = self.n;
        let tones = !self.beats.is_empty();
        for class in &mut self.classes {
            class.phase = class.turn(u);
        }
        let (group, beating) = self.classes.split_first().expect("the static group");
        for (k, z) in self.matrix.data.iter_mut().enumerate() {
            let (c, a, b) = (group.hop[0][k], group.hop[1][k], group.hop[2][k]);
            *z = if tones { c + a * u } else { a * u } + b * (u * u);
            for class in beating.iter().filter(|class| class.hops && !class.fast[k]) {
                *z += class.phase * (class.hop[0][k] + class.hop[1][k] * u + class.hop[2][k] * (u * u));
            }
        }
        let hopped = beating.iter().filter(|class| class.hops).count();
        self.turn(u - self.origin);
        for (row, out) in self.matrix.data.chunks(n).zip(rate.iter_mut()) {
            let z: C64 = row.iter().zip(m).map(|(d, m)| *d * *m).sum();
            *out = C64::new(z.im, -z.re);
        }
        self.work += ((2 + 4 * hopped) * n * n) as f64;
    }

    /// One stride of the frame's state without the decomposition, from the time u over the width w, either sign: CF4
    /// of Ĥ₀ + D(u) (module documentation, # The strides) -- the static group's parabola, and each tone's members at
    /// their phases --, each of its two exponentials on the vector by Taylor's series within its tolerance
    /// (`tolerance`) -- the two planned at once, at the larger of their bounds, and their scalar parts' factor taken once.
    fn stride(&mut self, state: &mut [C64], u: f64, w: f64) {
        let n = self.n;
        let size = n * n;
        // Ĥ at the stride's two Gauss nodes c = ½ ∓ √3/6, the weights ¼ ± √3/6: the two exponents' matrices.
        let r = 3.0_f64.sqrt() / 6.0;
        let (u1, u2) = (u + (0.5 - r) * w, u + (0.5 + r) * w);
        let (t, tolerance) = (C64::new(0.0, -w), self.tolerance(w));
        let (matrices, _) = self.product.split_at_mut(2 * size);
        let mut bounds = [(C64::ZERO, [0.0; 2]); 2];
        for (([w1, w2], matrix), bound) in [[0.25 + r, 0.25 - r], [0.25 - r, 0.25 + r]].into_iter().zip(matrices.chunks_exact_mut(size)).zip(&mut bounds) {
            if self.moving {
                let (first, second) = (w1 * u1 + w2 * u2, w1 * u1 * u1 + w2 * u2 * u2);
                for (((z, c), a), b) in matrix.iter_mut().zip(&self.base.data).zip(&self.ends[0].data).zip(&self.ends[1].data) {
                    *z = *c * (w1 + w2) + *a * first + *b * second;
                }
                for tone in &self.beats {
                    for (member, beat) in tone.members.iter().zip([tone.beat, -tone.beat]) {
                        let (p1, p2) = (C64::cis(beat * u1) * w1, C64::cis(beat * u2) * w2);
                        let q = [p1 + p2, p1 * u1 + p2 * u2, p1 * (u1 * u1) + p2 * (u2 * u2)];
                        for (((z, c), a), b) in matrix.iter_mut().zip(&member[0].data).zip(&member[1].data).zip(&member[2].data) {
                            *z += *c * q[0] + *a * q[1] + *b * q[2];
                        }
                    }
                }
            } else {
                for (z, c) in matrix.iter_mut().zip(&self.base.data) {
                    *z = *c * (w1 + w2);
                }
            }
            *bound = linalg::taylor_bound(matrix, n, t);
        }
        // An exponent of no finite bound makes no state: the stride's state is no number, and the step that took it is
        // refused at its end (step.rs `Refused::Overflow`; its generators are numbers, `pictured`, so only an
        // overflow gets here).
        let Some(plan) = linalg::Taylor::plan(bounds[0].1[0].max(bounds[1].1[0]), bounds[0].1[1].max(bounds[1].1[1]), tolerance, n) else {
            state.fill(C64::new(f64::NAN, f64::NAN));
            return;
        };
        for (matrix, (scalar, _)) in matrices.chunks_exact(size).zip(bounds) {
            linalg::taylor_series(matrix, n, scalar, t, state, plan);
        }
        let phase = (t * (bounds[0].0 + bounds[1].0)).exp();
        for x in state.iter_mut() {
            *x *= phase;
        }
        self.work += 2.0 * ((4 * size) as f64 + plan.work(n)) + (12 * size * self.beats.len()) as f64;
    }

    /// The tolerance of one of a stride's two exponentials over the width w, relative to the state's norm (module
    /// documentation, # The strides): the kick's share `QUADRATURE` of what the state forgets over it, the decay it is
    /// launched with over w/2 -- over the time the state keeps an error, 1/decay, the exponentials add up to the share
    /// whatever the step, as the old engine's truncation per waiting time did --; the unit round-off where the state
    /// forgets nothing (no decay: a unitary flow keeps every error).
    fn tolerance(&self, w: f64) -> f64 {
        (QUADRATURE * self.forgets * 0.5 * w.abs()).max(UNIT_ROUNDOFF)
    }

    /// Launches the picture from the frame's state x at the time `from` of the step (from the middle; the step's
    /// start −h, or a later time the state changes at). Through the decomposition: m there, (1 + X(from))⁻¹R⁻¹x, and
    /// where the generator moves m's two hops -- to the middle of what is left of the step and to the step's end --
    /// with its rates at the three stops: the state is then known at every time from the launch on (`at`). The middle
    /// stop's rate is none where it is the step's middle and there are no tones (D₂ is zero there). Without the
    /// decomposition: the launch's state, kept, and the first stop.
    pub fn launch(&mut self, from: f64, x: &[C64]) {
        let (n, h) = (self.n, self.half);
        self.origin = from;
        if !self.through {
            for stop in self.stops.chunks_exact_mut(n) {
                stop.copy_from_slice(x);
            }
            self.times = [from; 3];
            return;
        }
        let mut path = std::mem::take(&mut self.path);
        combine(false, x, n, &self.left, &mut self.lanes, 1, 0);
        unlay(&self.lanes, 1, &mut path[..n]);
        self.work += (n * n) as f64;
        if self.moving {
            if self.followed {
                // g = (1 + X(from))⁻¹ d by the factors of 1 + X there: X at the time row by row -- every class it
                // follows, each at its phase --, one more on its diagonal, factored -- unless they are the ones kept
                // (the following's, at the step's start).
                let size = n * n;
                if self.factored != from {
                    for class in &mut self.classes {
                        class.phase = class.turn(from);
                    }
                    let (group, beating) = self.classes.split_first().expect("the static group");
                    let (zero, rest) = group.follow.split_at(size);
                    let (one, two) = rest.split_at(size);
                    for (row, numbers) in self.factors.chunks_exact_mut(n).enumerate() {
                        for (column, z) in numbers.iter_mut().enumerate() {
                            let t = column * n + row;
                            *z = zero[t] + one[t] * from + two[t] * (from * from);
                            for class in beating.iter().filter(|class| class.followed) {
                                *z += class.phase * (class.follow[t] + class.follow[size + t] * from + class.follow[2 * size + t] * (from * from));
                            }
                        }
                        numbers[row] += C64::ONE;
                    }
                    let followers = beating.iter().filter(|class| class.followed).count();
                    linalg::factor(&mut self.factors, &mut self.pivots, n);
                    self.factored = from;
                    self.work += (n * size / 3 + 2 * size + 4 * followers * size) as f64;
                }
                linalg::solve(&self.factors, &self.pivots, n, &mut path[..n], 1);
                self.work += size as f64;
            }
            let spacing = 0.5 * (h - from);
            (self.centre, self.spacing) = (from + spacing, spacing);
            let centre = self.centre;
            let (start, rest) = path.split_at_mut(n);
            let (rate, rest) = rest.split_at_mut(n);
            self.rate(from, start, rate);
            let (middle, rest) = rest.split_at_mut(n);
            middle.copy_from_slice(start);
            self.hop(middle, from, spacing);
            let (rate, rest) = rest.split_at_mut(n);
            if centre == 0.0 && self.beats.is_empty() {
                rate.fill(C64::ZERO);
            } else {
                self.rate(centre, middle, rate);
            }
            let (end, rest) = rest.split_at_mut(n);
            end.copy_from_slice(middle);
            self.hop(end, centre, spacing);
            self.rate(h, end, &mut rest[..n]);
        }
        self.path = path;
    }

    /// The frame's state at the time u (from the middle; from the launch on) through the decomposition (module
    /// documentation): x = R (1 + X(u)) e^{−iΛ(u − u₀)} m(u), m its three stops' quintic.
    fn at(&mut self, u: f64, x: &mut [C64]) {
        let n = self.n;
        let size = n * n;
        assert!(self.through, "a state through the decomposition");
        let d = &mut self.coordinates[..];
        let path = &self.path;
        let node = |k: usize| &path[k * n..(k + 1) * n];
        if !self.moving {
            d.copy_from_slice(node(0));
        } else {
            let (values, rates) = quintic((u - self.centre) / self.spacing);
            let (start, first, middle, second, end, last) = (node(0), node(1), node(2), node(3), node(4), node(5));
            for a in 0..n {
                d[a] = start[a] * values[0] + middle[a] * values[1] + end[a] * values[2] + (first[a] * rates[0] + second[a] * rates[1] + last[a] * rates[2]) * self.spacing;
            }
        }
        for (z, l) in d.iter_mut().zip(&self.lambda) {
            *z *= (C64::new(0.0, -(u - self.origin)) * *l).exp();
        }
        // R(1 + X(u)) d: the matrix at the time from its three, R(1 + Y₀) + u RY₁ + u² RY₂ of the static group's and each
        // beating class's at its phase; R where nothing is followed.
        let (group, beating) = self.classes.split_first().expect("the static group");
        let followers = beating.iter().filter(|class| class.followed).count();
        let matrix = if self.followed {
            let (first, rest) = group.nodes.split_at(self.right.len());
            let (second, third) = rest.split_at(self.right.len());
            for (((m, first), second), third) in self.spare[0].iter_mut().zip(first).zip(second).zip(third) {
                *m = (u * u).mul_add(*third, u.mul_add(*second, *first));
            }
            for class in beating.iter().filter(|class| class.followed) {
                add_turned(class.turn(u), &class.nodes, u, &mut self.spare[0]);
            }
            &self.spare[0]
        } else {
            &self.right
        };
        combine(false, d, n, matrix, &mut self.lanes, 1, 0);
        unlay(&self.lanes, 1, x);
        self.work += ((if self.followed { 2 + 4 * followers } else { 1 }) * size + 4 * n) as f64;
    }

    /// The frame's state at the next stop, the time u from the middle, ahead of the last one on the path. Through
    /// the decomposition it is the state at that time (`at`); without it the stride from the last stop, which stays
    /// kept for `between`.
    pub fn advance(&mut self, u: f64, x: &mut [C64]) {
        let n = self.n;
        if self.through {
            return self.at(u, x);
        }
        let mut stops = std::mem::take(&mut self.stops);
        let (before, now) = stops[n..].split_at_mut(n);
        before.copy_from_slice(now);
        let from = self.times[2];
        self.stride(now, from, u - from);
        x.copy_from_slice(now);
        self.stops = stops;
        self.times = [self.origin, from, u];
    }

    /// The frame's state at a time u after the launch: the state there through the decomposition; without it the
    /// stride ahead from the latest state kept at or before u -- the last stop, the one before it, or the launch's
    /// (a stride ahead, as the flow goes: back from a later stop it would raise a decay's rounding by the decay).
    pub fn between(&mut self, u: f64, x: &mut [C64]) {
        let n = self.n;
        if self.through {
            return self.at(u, x);
        }
        let kept = (0..3).rev().find(|&k| (u - self.times[k]) * self.half >= 0.0).unwrap_or(0);
        x.copy_from_slice(&self.stops[kept * n..(kept + 1) * n]);
        let from = self.times[kept];
        self.stride(x, from, u - from);
    }

    /// The counted complex multiply-adds of a launch and of the states at `stops` stops over the step, priced without
    /// taking them: through the decomposition the start's coordinates, the two hops of m at the bound of D₂ over a
    /// half step (their series, their turn, their exponentials' plan, every class hopped) with the rates at the two
    /// ends, and a state per stop (`state_price`); without it a stride per stop at Ĥ₀'s bounds over the stops' mean
    /// spacing.
    pub fn price(&self, stops: usize) -> f64 {
        let (n, h) = (self.n, self.half.abs());
        let size = (n * n) as f64;
        if !self.through {
            return self.strided(stops);
        }
        let state = stops as f64 * self.state_price();
        if !self.moving {
            return size + state;
        }
        let moved = self.moved(0.0, h);
        let (plan, terms) = (linalg::Chebyshev::plan(moved, moved), series(self.spread * h, moved));
        let classes = self.classes.iter().filter(|class| class.hops).count();
        (2 + 4 * terms * classes + 2 * (4 * classes + 2 + plan.parts * plan.terms) + 2 + 2 * classes) as f64 * size + state
    }

    /// The counted complex multiply-adds of a state at a time through the decomposition (`at`).
    fn state_price(&self) -> f64 {
        let n = self.n;
        let followers = self.classes[1..].iter().filter(|class| class.followed).count();
        ((if self.followed { 2 + 4 * followers } else { 1 }) * n * n + 4 * n) as f64
    }

    /// The strides' price at `stops` stops: a stride per stop over the stops' mean spacing w, each exponential's series
    /// (`linalg::Taylor`) at the bound of its exponent, half of Ĥ₀'s bounds and D's farthest over w, within its tolerance
    /// (`tolerance`); and per exponential its matrix (three numbers an entry) and its bound's pass.
    fn strided(&self, stops: usize) -> f64 {
        let (n, h) = (self.n, self.half.abs());
        let w = 2.0 * h / stops as f64;
        let bound = 0.5 * (self.bounds[0] + self.bounds[1]) * w + self.moved(h - w, w);
        // A bound that is no number prices no route.
        linalg::Taylor::plan(bound, 0.5 * self.bounds[1] * w, self.tolerance(w), n).map_or(f64::INFINITY, |plan| stops as f64 * 2.0 * ((4 * n * n) as f64 + plan.work(n)))
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// What a split meets: every pair's turn |ω_ab| |h| between two nodes, and the sizes of its change over a half
    /// step, the slope's |A_ab| h² and the curvature's |B_ab| |h|³ (the larger of the pair's two ways round). None where
    /// nothing moves or the state does not go through the decomposition.
    pub(crate) fn pairs(picture: &Picture) -> Vec<[f64; 3]> {
        let (n, h) = (picture.n, picture.half.abs());
        if !(picture.through && picture.moving) {
            return Vec::new();
        }
        let slopes = &picture.classes[0].change[1..];
        let mut pairs = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                let size = |j: usize| slopes[j][a * n + b].abs().max(slopes[j][b * n + a].abs());
                pairs.push([(picture.lambda[a] - picture.lambda[b]).abs() * h, size(0) * h * h, size(1) * h * h * h]);
            }
        }
        pairs
    }

    /// Whether some pair's change is followed: the split's choice on a step.
    pub(crate) fn follows(picture: &Picture) -> bool {
        picture.followed
    }

    /// The largest row sum of |X| at the step's end: the size of the following there.
    fn following(picture: &Picture) -> f64 {
        let (n, h) = (picture.n, picture.half);
        let size = n * n;
        let follow = &picture.classes[0].follow;
        (0..n).map(|a| (0..n).map(|b| (follow[b * n + a] + follow[size + b * n + a] * h + follow[2 * size + b * n + a] * (h * h)).abs()).sum::<f64>()).fold(0.0, f64::max)
    }

    /// Dense RK4 of iẋ = (Ĥ₀ + uA + u²B)x from −h to h in `steps` steps: the tests' reference of a picture's state.
    fn flow(base: &Mat, a: &Mat, b: &Mat, half: f64, start: &[C64], steps: usize) -> Vec<C64> {
        let n = start.len();
        let at = |u: f64| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u));
        let slope = |u: f64, y: &[C64]| -> Vec<C64> { at(u).apply(y).into_iter().map(|z| C64::new(z.im, -z.re)).collect() };
        let add = |y: &[C64], k: &[C64], s: f64| -> Vec<C64> { y.iter().zip(k).map(|(y, k)| *y + *k * s).collect() };
        let dt = 2.0 * half / steps as f64;
        let mut x = start.to_vec();
        for i in 0..steps {
            let u = -half + i as f64 * dt;
            let k1 = slope(u, &x);
            let k2 = slope(u + 0.5 * dt, &add(&x, &k1, 0.5 * dt));
            let k3 = slope(u + 0.5 * dt, &add(&x, &k2, 0.5 * dt));
            let k4 = slope(u + dt, &add(&x, &k3, dt));
            x = (0..n).map(|s| x[s] + (k1[s] + k2[s] * 2.0 + k3[s] * 2.0 + k4[s]) * (dt / 6.0)).collect();
        }
        x
    }

    /// A picture of Ĥ₀ + uA + u²B over a step of 2h, prepared.
    fn pictured(base: &Mat, a: &Mat, b: &Mat, half: f64, moving: bool) -> Picture {
        let n = base.rows;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.base = base.clone();
        picture.ends = [-half, half].map(|u| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u)));
        picture.prepare(half, !moving, |_| 2, 0.0, 0.0);
        picture
    }

    /// The largest error of a launched picture's state at the times `at` (from the middle) against the flow of the
    /// same Ĥ₀ + uA + u²B from the step's start to each: that part of the step is a step of its own middle.
    fn errors(picture: &mut Picture, base: &Mat, a: &Mat, b: &Mat, half: f64, start: &[C64], at: &[f64], steps: usize) -> f64 {
        let n = start.len();
        let mut x = vec![C64::ZERO; n];
        let mut worst = 0.0_f64;
        for &u in at {
            picture.advance(u, &mut x);
            let (middle, part) = (0.5 * (u - half), 0.5 * (u + half));
            let shifted = Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * middle + b[(i, j)] * (middle * middle));
            let slope = Mat::from_fn(n, n, |i, j| a[(i, j)] + b[(i, j)] * (2.0 * middle));
            let exact = flow(&shifted, &slope, b, part, start, steps);
            worst = worst.max(x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max));
        }
        worst
    }

    /// The strides on a folded cluster's generator (module documentation, # The strides): five states, Ĥ₀ a random
    /// Hermitian of PGC3D's size about a large common light shift (its traceless part's bound 0.33 rad/µs, the trap's
    /// shift 50 rad/µs) with the Doppler frames' rates on its diagonal (±2.5 rad/µs) and the decay of a folded light
    /// (0.01 rad/µs), and slopes of a light pattern crossed at its speed (A, B of 0.3 and 0.6 rad/µs per µs, µs²); a
    /// random state, the strides' stops at a quadrature's spacing over steps of 0.5 and 1 µs, launched with the decay of
    /// a folded light's ground state (`DECAY`). At every stop the state is the parabola's flow within the kick's share
    /// `QUADRATURE` of its norm (the Magnus expansion's part, `stridden`) and of what it forgets over the time (the
    /// exponentials' truncation, `Picture::tolerance`); the error is printed beside the share.
    #[test]
    fn a_folded_cluster_s_strides_are_its_flow_within_the_share() {
        const DECAY: f64 = 0.3;
        let n = 5;
        let mut s = crate::random::Stream::new(41, crate::random::Purpose::Initial, 0, 0);
        let mut hermitian = |size: f64| {
            let m = Mat::from_fn(n, n, |_, _| C64::new(s.normal(), s.normal()));
            let h = Mat::from_fn(n, n, |i, j| (m[(i, j)] + m[(j, i)].conj()) * 0.5);
            let norm = h.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            Mat::from_fn(n, n, |i, j| h[(i, j)] * (size / norm))
        };
        let (light, a, b) = (hermitian(0.33), hermitian(0.3), hermitian(0.6));
        let base = Mat::from_fn(n, n, |i, j| light[(i, j)] + if i == j { C64::new(50.0 + 2.5 * (i as f64 - 2.0), -0.01) } else { C64::ZERO });
        let start: Vec<C64> = (0..n).map(|k| C64::new(1.0 / (1.0 + k as f64), 0.3 * k as f64 - 0.5)).collect();
        let norm = start.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        for half in [0.25, 0.5] {
            let mut picture = Picture::default();
            picture.fit(n);
            // The decomposition's route priced beyond the strides', so that they are taken where their Magnus part is
            // within its share.
            picture.counted[n] = Some([f64::MAX, 0.0]);
            picture.base = base.clone();
            picture.ends = [-half, half].map(|u| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u)));
            let stops = |bandwidth: f64| ((bandwidth * 2.0 * half).ceil() as usize).max(2) + 2;
            picture.prepare(half, false, stops, 0.0, DECAY);
            assert!(!picture.through, "half {half}: the strides taken");
            let count = stops(picture.bandwidth);
            picture.launch(-half, &start);
            let at: Vec<f64> = (1..=count).map(|k| -half + 2.0 * half * k as f64 / count as f64).collect();
            let error = errors(&mut picture, &base, &a, &b, half, &start, &at, 20_000) / norm;
            let share = QUADRATURE * (1.0 + DECAY * 2.0 * half);
            eprintln!("half {half}: {count} stops, the state's largest error {error:.2e} of its norm (the share {share:.2e})");
            assert!(error <= share, "half {half}: {error:e}");
        }
    }

    /// Towards a defective generator the state leaves the decomposition (module documentation): a resonant two-level
    /// atom whose coupling is a quarter of its decay rate, [[0, c], [c, −iΓ/2]] at c = Γ/4, has one eigenvector. At that
    /// point and a millionth of the coupling off it the picture takes the exponential's strides; a hundredth off it,
    /// where the eigenvectors' condition is some hundreds, it takes the decomposition's. Each way the state at the
    /// step's end, through sixteen stops, is the generator's flow: to the reference's own accuracy where nothing
    /// moves, and with slopes within the second order of what they move between two stops.
    #[test]
    fn a_defective_generator_s_strides_are_the_exponential_s() {
        let (gamma, half, stops) = (38.117, 0.03, 16);
        let start = [C64::ONE, C64::new(0.2, -0.1)];
        for (off, decomposed) in [(0.0, false), (1e-6, false), (1e-2, true)] {
            let c = 0.25 * gamma * (1.0 + off);
            let base = Mat::from_fn(2, 2, |i, j| if i != j { C64::real(c) } else if i == 1 { C64::new(0.0, -0.5 * gamma) } else { C64::ZERO });
            for moving in [false, true] {
                let slope = |size: f64| Mat::from_fn(2, 2, |i, j| if i != j && moving { C64::new(size, 0.3 * size * (i as f64 - j as f64)) } else { C64::ZERO });
                let (a, b) = (slope(2.0), slope(30.0));
                let exact = flow(&base, &a, &b, half, &start, 100_000);
                let mut picture = pictured(&base, &a, &b, half, moving);
                assert_eq!(picture.through, decomposed, "{off} off the exceptional point");
                let mut x = [C64::ZERO; 2];
                picture.launch(-half, &start);
                let w = 2.0 * half / stops as f64;
                for k in 1..=stops {
                    picture.advance(if k == stops { half } else { -half + k as f64 * w }, &mut x);
                }
                let error = x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max);
                // What the slopes move between two stops at the step's end: (h‖A‖ + h²‖B‖) w, its square the second order;
                // and the strides' exponentials' share of the kick's, relative to the state's norm (`Picture::exponential`).
                let moved = (half * 2.0 + half * half * 30.0) * 2.0_f64.sqrt() * w;
                // Launched with no decay (`pictured`), the exponentials are taken to round-off (`Picture::tolerance`).
                let (norm, share) = (start.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt(), 0.0);
                eprintln!("{off} off the exceptional point, moving {moving}: the state's error {error:e} (the slopes move {moved:e} between stops)");
                assert!(error < if moving { moved * moved } else { 1e-12 } + share * norm, "{off} off the exceptional point, moving {moving}: {error:e}");
            }
        }
    }

    /// The picture's state is the generator's flow where every pair is fast: Ĥ₀ of detunings up to 600 rad/µs with a
    /// decay and couplings of 40 rad/µs on four states, against RK4 at 4·10⁵ steps of the same Ĥ₀ + uA + u²B. Without
    /// slopes the state is exact, at the end and between the nodes. With slopes that turn the couplings by a tenth of
    /// a radian over the step it is the following alone (no pair is slow: two hops of nothing): it errs by the second
    /// order of the following, within the square of X's size at the step's end, and by a quarter of that at half the
    /// slopes.
    #[test]
    fn a_fast_pair_s_state_follows_the_change() {
        let (n, half) = (4, 0.05);
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        let mut base = hermitian(1.3, 40.0);
        for (i, shift) in [0.0, 5.0, -600.0, -620.0].into_iter().enumerate() {
            base[(i, i)] += C64::new(shift, if i >= 2 { -19.0 } else { 0.0 });
        }
        let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * 0.5).collect();
        let mut found = Vec::new();
        for scale in [0.0, 1.0, 0.5] {
            let (a, b) = (hermitian(2.1, 2.0 * scale), hermitian(0.4, 20.0 * scale));
            let moving = scale > 0.0;
            let mut picture = pictured(&base, &a, &b, half, moving);
            assert!(picture.through && picture.moving == moving);
            assert!(!moving || picture.classes[0].fast.iter().enumerate().all(|(k, fast)| *fast || k / n == k % n), "a slow pair");
            picture.launch(-half, &start);
            let error = errors(&mut picture, &base, &a, &b, half, &start, &[half, 0.37 * half, -0.6 * half], 400_000);
            eprintln!("slopes × {scale}: the state's error {error:e}, the following's size {:e}", following(&picture));
            found.push((error, following(&picture)));
        }
        assert!(found[0].0 < 1e-11, "{found:?}");
        assert!(found[1].0 < found[1].1 * found[1].1 && found[2].0 < 0.4 * found[1].0, "{found:?}");
    }

    /// A state launched inside the step goes on as the flow does (module documentation: the propagation is linear, and
    /// the picture knows the state by m alone). The step's start state is launched and read at a time of the step and
    /// at later ones; launched again at that time from the state read there, it is read at the later ones again: the
    /// two launches differ by what the hops and their quintics leave, each within its own bound -- slow pairs hopped
    /// (the fifth order of the step: 10⁻⁴ at the half 0.08, `a_slow_pair_s_hops_are_of_the_fourth_order`), fast pairs
    /// followed (the square of the following, `a_fast_pair_s_state_follows_the_change`) --, whether the launch lies
    /// before the step's middle or after it; and at its own time the launched state is the state given, to rounding.
    #[test]
    fn a_state_launched_inside_the_step_goes_on_as_the_flow_does() {
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |n: usize, seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        let slow = {
            let mut base = hermitian(3, 1.3, 2.0);
            for i in 0..3 {
                base[(i, i)] += C64::new(0.0, -0.05 * i as f64);
            }
            (base, hermitian(3, 2.1, 4.0), hermitian(3, 0.4, 20.0), 0.08)
        };
        let fast = {
            let mut base = hermitian(4, 1.3, 40.0);
            for (i, shift) in [0.0, 5.0, -600.0, -620.0].into_iter().enumerate() {
                base[(i, i)] += C64::new(shift, if i >= 2 { -19.0 } else { 0.0 });
            }
            (base, hermitian(4, 2.1, 2.0), hermitian(4, 0.4, 20.0), 0.05)
        };
        for (name, (base, a, b, half)) in [("slow pairs", slow), ("fast pairs", fast)] {
            let n = base.rows;
            let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * 0.5).collect();
            let mut picture = pictured(&base, &a, &b, half, true);
            assert!(picture.through && picture.followed == (name == "fast pairs"));
            let within = if picture.followed { following(&picture).powi(2) } else { 1e-4 };
            for from in [-0.55 * half, 0.0, 0.3 * half] {
                let later = [from + 0.21 * (half - from), from + 0.5 * (half - from), from + 0.83 * (half - from), half];
                picture.launch(-half, &start);
                let mut state = vec![C64::ZERO; n];
                picture.advance(from, &mut state);
                let first: Vec<Vec<C64>> = later.iter().map(|&u| {
                    let mut x = vec![C64::ZERO; n];
                    picture.advance(u, &mut x);
                    x
                }).collect();
                picture.launch(from, &state);
                let mut x = vec![C64::ZERO; n];
                picture.advance(from, &mut x);
                let own = x.iter().zip(&state).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max);
                assert!(own < 1e-13, "{name}, launched at {from}: the state there is {own:e} off the one given");
                let mut worst = 0.0_f64;
                for (&u, expected) in later.iter().zip(&first) {
                    picture.advance(u, &mut x);
                    worst = worst.max(x.iter().zip(expected).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max));
                }
                eprintln!("{name}, launched at {:.2} of the half step: the two launches differ by {worst:e} (within {within:e})", from / half);
                assert!(worst < within, "{name}, launched at {from}: {worst:e} against {within:e}");
            }
        }
    }

    /// The picture's products go through the lanes' kernel (`linalg::combine`: sums by fused multiply-adds, Σ_k c_k x_k
    /// in the order of k). Each is the plain sum it stands for within the rounding of the operations between them
    /// (`units::rounding`: γ of their count, times the sum of the terms' moduli):
    /// - the slopes in the eigenbasis, (R⁻¹A)R -- on a generator whose pairs are all slow, so that the slopes stay as
    ///   the change of basis leaves them --: two sums of n products, the kernel's and the plain ones' (2n operations
    ///   each of the four) and their difference: within γ(8n + 1) of |R⁻¹||A||R|, entry by entry;
    /// - a state at a time, R(1 + X(u)) on its coordinates d -- on a generator with fast pairs and slow ones --: the
    ///   kernel's matrices R(1 + X₀), RX₁, RX₂ (a sum of n products onto R: 2n + 1), their polynomial in u (2) and its
    ///   sum on d (n products: 2n) against the plain R(d + X(u)d) (X's polynomial: 4; two sums: 2n + 2n + 1) and their
    ///   difference: within γ(8n + 9) of |R|(1 + |X₀| + |u||X₁| + u²|X₂|)|d|, state by state.
    #[test]
    fn the_kernel_s_products_are_their_plain_sums_within_their_rounding() {
        use crate::units::rounding;
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |n: usize, seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        // The slopes: light shifts of five states, none far enough from another over the half step to be followed.
        let (n, half) = (5, 0.08);
        let mut base = hermitian(n, 1.3, 2.0);
        for i in 0..n {
            base[(i, i)] += C64::new(0.0, -0.05 * i as f64);
        }
        let (a, b) = (hermitian(n, 2.1, 4.0), hermitian(n, 0.4, 20.0));
        let picture = pictured(&base, &a, &b, half, true);
        assert!(picture.through && picture.moving && !picture.followed);
        let (r, inverse) = (&picture.basis.vectors, &picture.basis.inverse);
        let ends = [-half, half].map(|u| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u)));
        // The slopes as `prepare` takes them from the generators at the nodes.
        let (first, second) = (0.5 / half, 0.5 / (half * half));
        let slopes = [Mat::from_fn(n, n, |i, j| (ends[1][(i, j)] - ends[0][(i, j)]) * first), Mat::from_fn(n, n, |i, j| (ends[1][(i, j)] + ends[0][(i, j)] - base[(i, j)] * 2.0) * second)];
        let modulus = |m: &Mat| Mat::from_fn(m.rows, m.cols, |i, j| C64::real(m[(i, j)].abs()));
        let mut worst = 0.0_f64;
        for (slope, found) in slopes.iter().zip(&picture.classes[0].change[1..]) {
            let (plain, size) = (inverse.product(slope).product(r), modulus(inverse).product(&modulus(slope)).product(&modulus(r)));
            for k in 0..n * n {
                let (apart, bound) = ((found[k] - plain.data[k]).abs(), rounding(8 * n + 1) * size.data[k].re);
                assert!(apart <= bound, "a slope's entry {k}: {apart:e} against {bound:e}");
                worst = worst.max(apart / bound);
            }
        }
        eprintln!("the slopes in the eigenbasis: within {worst:.3} of their rounding's bound");
        // A state: two ground states and two explicit ones 600 rad/µs off, followed.
        let (n, half) = (4, 0.05);
        let mut base = hermitian(n, 1.3, 40.0);
        for (i, shift) in [0.0, 5.0, -600.0, -620.0].into_iter().enumerate() {
            base[(i, i)] += C64::new(shift, if i >= 2 { -19.0 } else { 0.0 });
        }
        let (a, b) = (hermitian(n, 2.1, 2.0), hermitian(n, 0.4, 20.0));
        let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * 0.5).collect();
        let mut picture = pictured(&base, &a, &b, half, true);
        assert!(picture.through && picture.followed && picture.classes[0].fast.iter().any(|fast| !fast));
        picture.launch(-half, &start);
        let mut worst = 0.0_f64;
        for u in [-half, -0.6 * half, 0.0, 0.37 * half, half] {
            let mut x = vec![C64::ZERO; n];
            picture.at(u, &mut x);
            let (d, r, size) = (&picture.coordinates[..n], &picture.basis.vectors, n * n);
            // X(u), row by row, and its terms' moduli.
            let x_at = |row: usize, column: usize| {
                let t = column * n + row;
                let follow = &picture.classes[0].follow;
                let terms = [follow[t], follow[size + t] * u, follow[2 * size + t] * (u * u)];
                (terms[0] + terms[1] + terms[2], terms[0].abs() + terms[1].abs() + terms[2].abs())
            };
            for row in 0..n {
                let (mut plain, mut scale) = (C64::ZERO, 0.0);
                for k in 0..n {
                    let (mut followed, mut moduli) = (d[k], d[k].abs());
                    for column in 0..n {
                        let (value, modulus) = x_at(k, column);
                        followed += value * d[column];
                        moduli += modulus * d[column].abs();
                    }
                    plain += r[(row, k)] * followed;
                    scale += r[(row, k)].abs() * moduli;
                }
                let (apart, bound) = ((x[row] - plain).abs(), rounding(8 * n + 9) * scale);
                assert!(apart <= bound, "the state {row} at {u}: {apart:e} against {bound:e}");
                worst = worst.max(apart / bound);
            }
        }
        eprintln!("a state at a time: within {worst:.3} of its rounding's bound");
    }

    /// Slow pairs are hopped (module documentation): Ĥ₀ of light shifts that turn less over a half step than their
    /// change's split,
    /// with a small decay, and slopes that change it by a tenth over the step, on three states -- no pair fast. The
    /// state at the end and between the nodes is the flow within the fifth order of the step -- the hop's second Magnus
    /// term, (h⁵/60)‖[A, [Λ, A]]‖ ≈ (|ω|h)(h²‖A‖)²/60 = 0.4 × 0.064²/60 = 3·10⁻⁵ at the half 0.08 --: at half the step
    /// it errs twenty-four times less and more.
    #[test]
    fn a_slow_pair_s_hops_are_of_the_fourth_order() {
        let n = 3;
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        let mut base = hermitian(1.3, 2.0);
        for i in 0..n {
            base[(i, i)] += C64::new(0.0, -0.05 * i as f64);
        }
        let (a, b) = (hermitian(2.1, 4.0), hermitian(0.4, 20.0));
        let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * 0.5).collect();
        let mut found = Vec::new();
        for half in [0.08, 0.04] {
            let mut picture = pictured(&base, &a, &b, half, true);
            assert!(picture.through && !picture.followed, "a fast pair at the half {half}");
            picture.launch(-half, &start);
            let error = errors(&mut picture, &base, &a, &b, half, &start, &[half, 0.0, -0.41 * half, 0.73 * half], 100_000);
            eprintln!("half {half}: the state's error {error:e}");
            found.push(error);
        }
        assert!(found[0] < 1e-4 && found[1] < found[0] / 24.0, "{found:?}");
    }

    /// One pair hopped, followed and under the rule: Ĥ₀ = diag(0, −ω) with the change uA + u²B on the pair alone,
    /// x = |ω| h from 0.5 to 4 and the change's size s = (|A| h + |B| h²) h from 0.01 to 0.3 -- A and B in equal parts
    /// (their phases 1.1 rad apart), A alone, B alone --, from three states; the state's largest error at the step's end
    /// and inside it (u = ±h/2) against the flow. A measurement of the two remainders.
    ///
    /// The treatment the rule does not take is reached by what the rule reads, and by nothing else: it splits a pair by
    /// the sizes of all the pairs' changes together (module documentation, # The split). So a second pair, of two
    /// further states with no coupling to the first two and no amplitude of the start, changes the first pair's
    /// treatment by its own slope and leaves that pair's state as it was. The slope is the least at which the rule
    /// decides so, of eight a decade from 10⁻³ to 10²; where the rule decides so at none (a pair too slow to be followed
    /// beside any other) that treatment is not read.
    fn a_pair_s_remainders(parts: [f64; 2], size: f64, x: f64) -> [Option<[f64; 2]>; 3] {
        let half = 1.0;
        let starts = [[C64::ONE, C64::ZERO], [C64::real(0.5_f64.sqrt()); 2], [C64::real(0.6), C64::new(0.0, 0.8)]];
        let own = [parts[0] * size, parts[1] * size];
        // The following's size on a pair of the turn x between two nodes (half = 1) and the change C + uA + u²B (`flag`).
        let across = |x: f64, a: C64, b: C64| {
            let y = super::following(C64::real(x).recip(), C64::ZERO, a, b);
            [-1.0, 1.0].into_iter().map(|u| (y[0] + y[1] * u + y[2] * (u * u)).abs()).fold(0.0, f64::max)
        };
        // A second pair's slope that takes the first to a route, and the second pair (3 rad between two nodes) to one.
        let beside = |fast: bool| (0..=40).map(|k| 10_f64.powf(-3.0 + k as f64 / 8.0)).find(|other| {
            let sizes = [own[0].hypot(*other), own[1], 0.0];
            followed(x, sizes, across(x, C64::cis(0.7) * own[0], C64::cis(-0.4) * own[1])) == Some(fast) && followed(3.0, sizes, across(3.0, C64::real(*other), C64::ZERO)).is_some()
        });
        let mut found = [None; 3];
        for (treatment, wanted) in [Some(false), Some(true), None].into_iter().enumerate() {
            let (n, other) = match wanted.map(beside) {
                None => (2, 0.0),
                Some(Some(other)) => (4, other),
                Some(None) => continue,
            };
            // The first pair's states and the second's: 0 and −ω; 7.3 and 3/h below it.
            let base = Mat::from_fn(n, n, |i, j| if i == j { C64::real([0.0, -x / half, 7.3, 7.3 - 3.0 / half][i]) } else { C64::ZERO });
            let change = |z: C64, other: f64| {
                Mat::from_fn(n, n, |i, j| match (i.min(j), i.max(j)) {
                    (0, 1) if i < j => z,
                    (0, 1) => z.conj(),
                    (2, 3) => C64::real(other),
                    _ => C64::ZERO,
                })
            };
            let (a, b) = (change(C64::cis(0.7) * (own[0] / half.powi(2)), other / half.powi(2)), change(C64::cis(-0.4) * (own[1] / half.powi(3)), 0.0));
            let mut picture = pictured(&base, &a, &b, half, true);
            assert!(picture.through && wanted.is_none_or(|fast| picture.classes[0].fast[1] == fast), "x {x}, size {size}: a second pair's slope {other} for {wanted:?}");
            let mut worst = [0.0_f64; 2];
            for start in &starts {
                let start: Vec<C64> = (0..n).map(|state| if state < 2 { start[state] } else { C64::ZERO }).collect();
                picture.launch(-half, &start);
                for (place, times) in [&[half][..], &[-0.5 * half, 0.5 * half][..]].into_iter().enumerate() {
                    worst[place] = worst[place].max(errors(&mut picture, &base, &a, &b, half, &start, times, 20_000));
                }
            }
            found[treatment] = Some(worst);
        }
        found
    }

    /// What the two treatments leave of one pair at the step's end, derived and read in the crate
    /// (`a_pair_s_remainders`; the table is printed). With s_A = |A| h², s_B = |B| h³ the sizes of the change's slope
    /// and curvature over a half step and x = |ω| h:
    /// - hopped: the second Magnus term of the two hops, −½∫∫[D̃(t₁), D̃(t₂)] -- on the pair a phase of
    ///   ∫∫ Im(e^{iω(t₁ − t₂)} D(t₁) D(t₂)*) over t₂ < t₁ of each hop --: x s_A²/15 for a slope alone and x s_B²/42 for a
    ///   curvature alone while the nodes resolve the pair (x ≤ 1: within a twentieth), and Im(A B*) h⁵/15 at any x
    ///   where the two are out of phase -- the second order of the change, the fifth of the step;
    /// - followed: of the second order the three nodes' parabola of D₂ misses the quartic of the change's own light
    ///   shift, u⁴|B|²/ω, and nothing else -- ∫(u⁴ − h²u²)du |B|²/ω = (4/15) s_B²/x, within a fiftieth at the size 0.01
    ///   from x = 1 on. A slope alone leaves the third order, D₂'s part on the pair itself that is left out,
    ///   −X D X = −(|A|² A/ω²) u (u + i/ω)²: s_A³ |I(x)|/x² with I(x) = ∫₋₁¹ e^{ixt} t (t + i/x)² dt = i(2/(3x) + x/15 + …),
    ///   that is (2/3)(s_A/x)³ to the leading order in x (within a thirtieth at the size 0.01 for x from 0.75 to 2.5);
    /// - so the two are equal at no number of x. For a slope alone (2/3)(s_A/x)³ = x s_A²/15 at x⁴ = 10 s_A: at
    ///   x = 1 for s_A = 0.1, at 0.56 for 0.01 and at 1.3 for 0.3; at x = 1 and the size 0.01 the followed pair errs ten
    ///   times less than the hopped one for a slope alone and ten times more for a curvature alone (which cross at 4);
    /// - the rule (`followed`: the leading remainders compared) takes the treatment that errs less, or one within a
    ///   half more than the other where the two are close -- at every x and size here where the other is read
    ///   (`a_pair_s_remainders`): the hops wherever the rule follows the pair, the following at some x where it hops.
    #[test]
    fn a_pair_hopped_and_followed_against_its_flow() {
        // |I(x)|, by Simpson's rule.
        let third = |x: f64| {
            let panels = 2000;
            let term = |k: usize| {
                let t = -1.0 + 2.0 * k as f64 / panels as f64;
                let weight = if k == 0 || k == panels { 1.0 } else if k % 2 == 1 { 4.0 } else { 2.0 };
                C64::cis(x * t) * (C64::new(t, 1.0 / x) * C64::new(t, 1.0 / x)) * (t * weight)
            };
            (0..=panels).map(term).sum::<C64>().abs() * (2.0 / (3.0 * panels as f64))
        };
        // The cells where both treatments are read.
        let mut both = 0;
        for (name, parts) in [("A and B", [0.5, 0.5]), ("A alone", [1.0, 0.0]), ("B alone", [0.0, 1.0])] {
            for size in [0.01, 0.1, 0.3] {
                for x in [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0] {
                    let [hop, follow, rule] = a_pair_s_remainders(parts, size, x);
                    let rule = rule.expect("the rule's own treatment");
                    // The hops are read wherever the rule follows the pair: a second pair's slope takes any pair to them.
                    assert!(hop.is_some() || follow.is_none(), "{name}, size {size}, x {x}: followed, and not hopped beside any slope");
                    both += usize::from(hop.is_some() && follow.is_some());
                    for (treatment, other) in [("hopped", hop), ("followed", follow)] {
                        assert!(other.is_none_or(|other| rule[0] <= 1.5 * other[0]), "{name}, size {size}, x {x}: the rule's {:e}, {treatment} {other:?}", rule[0]);
                    }
                    if size == 0.01 {
                        let (slope, curvature) = (parts[0] * size, parts[1] * size);
                        let mut held = vec![];
                        if let (Some(hop), true) = (hop, parts[0] * parts[1] == 0.0 && x <= 1.0) {
                            held.push(("hopped", hop[0], x * (slope * slope / 15.0 + curvature * curvature / 42.0), 0.05));
                        }
                        if let (Some(follow), true) = (follow, parts[1] > 0.0 && x >= 1.0) {
                            held.push(("followed", follow[0], 4.0 / 15.0 * curvature * curvature / x, 0.02));
                        }
                        if let (Some(follow), true) = (follow, parts[1] == 0.0 && (0.75..=2.5).contains(&x)) {
                            held.push(("followed", follow[0], slope.powi(3) * third(x) / (x * x), 0.03));
                        }
                        for (treatment, found, derived, within) in held {
                            assert!((found / derived - 1.0).abs() < within, "{name}, x {x}: {treatment} {found:e} against {derived:e}");
                        }
                        if let (Some(hop), Some(follow), true) = (hop, follow, x == 1.0 && parts[0] * parts[1] == 0.0) {
                            assert!(if parts[1] == 0.0 { follow[0] < 0.2 * hop[0] } else { follow[0] > 5.0 * hop[0] }, "{name}, x {x}: hopped {:e}, followed {:e}", hop[0], follow[0]);
                        }
                    }
                    let read = |found: Option<[f64; 2]>, place: usize| found.map_or("   not read".into(), |found| format!("{:.3e}", found[place]));
                    eprintln!(
                        "{name}, size {size:4}, x {x:4}: end hop {} follow {} | inside hop {} follow {} | the rule's: end {:.3e}, inside {:.3e}",
                        read(hop, 0),
                        read(follow, 0),
                        read(hop, 1),
                        read(follow, 1),
                        rule[0],
                        rule[1]
                    );
                }
            }
        }
        eprintln!("both treatments read at {both} of 81");
        assert!(both > 0);
    }

    /// The outputs of a picture of the static group alone, bit by bit: its route, its bandwidth, its counts and price,
    /// and the states it gives -- launched at the step's start and read at stops and between them, launched again inside
    /// the step and read on -- folded into one number (FNV-1a over their bits), with their count.
    fn static_outputs(picture: &mut Picture) -> (u64, usize) {
        let (n, h) = (picture.n, picture.half);
        let mut bits: Vec<u64> = vec![u64::from(picture.through), picture.bandwidth.to_bits(), picture.work.to_bits(), picture.price(5).to_bits()];
        bits.extend(picture.counts().iter().flatten().flat_map(|c| c.map(f64::to_bits)));
        let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * (0.5 + 0.1 * i as f64)).collect();
        let mut x = vec![C64::ZERO; n];
        let state = |x: &[C64], bits: &mut Vec<u64>| bits.extend(x.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]));
        picture.launch(-h, &start);
        for u in [-0.5 * h, 0.0, 0.3 * h] {
            picture.advance(u, &mut x);
            state(&x, &mut bits);
        }
        picture.between(0.1 * h, &mut x);
        state(&x, &mut bits);
        picture.advance(h, &mut x);
        state(&x, &mut bits);
        picture.between(0.2 * h, &mut x);
        let inside = x.clone();
        picture.launch(0.2 * h, &inside);
        for u in [0.6 * h, h] {
            picture.advance(u, &mut x);
            state(&x, &mut bits);
        }
        bits.push(picture.work.to_bits());
        let hash = bits.iter().flat_map(|b| b.to_le_bytes()).fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3));
        (hash, bits.len())
    }

    /// The static group alone is propagated as it was before the exact tones (module documentation, # Tones: the
    /// tones' propagation is one implementation, of which the static group alone is the case without tones): on the
    /// inputs of this module's tests -- slow pairs hopped, fast pairs followed, both, nothing moving, the strides of a
    /// folded cluster and of a defective generator, each route asserted --, every output of the picture
    /// (`static_outputs`) is the one the propagation gave before the tones (3640446e, read there by this test), to the
    /// bit.
    #[test]
    fn the_static_group_alone_is_propagated_as_before_to_the_bit() {
        let entry = |seed: f64, i: usize, j: usize| C64::new((seed * (1.0 + i as f64) + 0.7 * j as f64).sin(), (seed * (2.0 + j as f64) - 0.3 * i as f64).cos());
        let hermitian = |n: usize, seed: f64, size: f64| Mat::from_fn(n, n, |i, j| (entry(seed, i, j) + entry(seed, j, i).conj()) * (0.5 * size));
        let shifted = |m: Mat, shifts: &[(f64, f64)]| Mat::from_fn(m.rows, m.cols, |i, j| m[(i, j)] + if i == j { C64::new(shifts[i].0, shifts[i].1) } else { C64::ZERO });
        let mut found = Vec::new();
        // Slow pairs, moving; fast pairs, moving and still; fast and slow pairs together.
        let slow = shifted(hermitian(3, 1.3, 2.0), &[(0.0, 0.0), (0.0, -0.05), (0.0, -0.1)]);
        let fast = shifted(hermitian(4, 1.3, 40.0), &[(0.0, 0.0), (5.0, 0.0), (-600.0, -19.0), (-620.0, -19.0)]);
        let mixed = shifted(hermitian(5, 0.7, 10.0), &[(0.0, 0.0), (1.0, 0.0), (3.0, -0.02), (-300.0, -19.0), (-310.0, -19.0)]);
        for (name, base, a, b, half, moving) in [
            ("slow pairs", &slow, hermitian(3, 2.1, 4.0), hermitian(3, 0.4, 20.0), 0.08, true),
            ("fast pairs", &fast, hermitian(4, 2.1, 2.0), hermitian(4, 0.4, 20.0), 0.05, true),
            ("still", &fast, hermitian(4, 2.1, 2.0), hermitian(4, 0.4, 20.0), 0.05, false),
            ("fast and slow pairs", &mixed, hermitian(5, 2.1, 3.0), hermitian(5, 0.4, 30.0), 0.1, true),
        ] {
            let mut picture = pictured(base, &a, &b, half, moving);
            let n = base.rows;
            let hopped = (0..n * n).any(|k| k / n != k % n && !picture.classes[0].fast[k]);
            let route = [picture.through, picture.moving, picture.followed, picture.followed && hopped];
            let wanted = match name {
                "slow pairs" => [true, true, false, false],
                "still" => [true, false, false, false],
                "fast pairs" => [true, true, true, false],
                _ => [true, true, true, true],
            };
            assert_eq!(route, wanted, "{name}: the route (decomposed, moving, followed, a pair hopped beside)");
            found.push((name, static_outputs(&mut picture)));
        }
        // The strides: a folded cluster (`a_folded_cluster_s_strides_are_its_flow_within_the_share`) and a defective
        // generator (`a_defective_generator_s_strides_are_the_exponential_s`).
        let n = 5;
        let mut s = crate::random::Stream::new(41, crate::random::Purpose::Initial, 0, 0);
        let mut random = |size: f64| {
            let m = Mat::from_fn(n, n, |_, _| C64::new(s.normal(), s.normal()));
            let h = Mat::from_fn(n, n, |i, j| (m[(i, j)] + m[(j, i)].conj()) * 0.5);
            let norm = h.data.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            Mat::from_fn(n, n, |i, j| h[(i, j)] * (size / norm))
        };
        let (light, a, b) = (random(0.33), random(0.3), random(0.6));
        let base = Mat::from_fn(n, n, |i, j| light[(i, j)] + if i == j { C64::new(50.0 + 2.5 * (i as f64 - 2.0), -0.01) } else { C64::ZERO });
        let half = 0.25;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.counted[n] = Some([f64::MAX, 0.0]);
        picture.base = base.clone();
        picture.ends = [-half, half].map(|u| Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u)));
        picture.prepare(half, false, |bandwidth: f64| ((bandwidth * 2.0 * half).ceil() as usize).max(2) + 2, 0.0, 0.3);
        assert!(!picture.through, "a folded cluster's strides");
        found.push(("strides of a folded cluster", static_outputs(&mut picture)));
        let gamma = 38.117;
        let c = 0.25 * gamma;
        let base = Mat::from_fn(2, 2, |i, j| if i != j { C64::real(c) } else if i == 1 { C64::new(0.0, -0.5 * gamma) } else { C64::ZERO });
        let slope = |size: f64| Mat::from_fn(2, 2, |i, j| if i != j { C64::new(size, 0.3 * size * (i as f64 - j as f64)) } else { C64::ZERO });
        let mut picture = pictured(&base, &slope(2.0), &slope(30.0), 0.03, true);
        assert!(!picture.through, "a defective generator's strides");
        found.push(("strides of a defective generator", static_outputs(&mut picture)));
        // Read at 3640446e by this test.
        let before = [
            ("slow pairs", (0x5929_f2fe_4b8c_a975, 49)),
            ("fast pairs", (0xeb67_a5e6_163a_febe, 63)),
            ("still", (0x2063_56cf_0f1f_b516, 63)),
            ("fast and slow pairs", (0xbff6_10b9_9e9f_9cdd, 77)),
            ("strides of a folded cluster", (0x07d3_5ae0_fefe_35e7, 77)),
            ("strides of a defective generator", (0x3784_b559_9bc9_62a2, 33)),
        ];
        for ((name, (hash, count)), (_, (was, _))) in found.iter().zip(before) {
            eprintln!("{name}: {count} outputs, {hash:#018x} (before the tones {was:#018x})");
        }
        assert_eq!(found, before);
    }

    /// A candidate tone of a test: its beat ν and its members' envelopes G⁺ (at e^{iνu}) and G⁻ (at e^{−iνu}).
    struct Tone {
        beat: f64,
        plus: Mat,
        minus: Mat,
    }

    /// The generator of a static group G₀ + uA₀ and candidate tones at the time u: G₀ + uA₀ + Σ_m (e^{iν_m u} G_m⁺ +
    /// e^{−iν_m u} G_m⁻).
    fn toned(group: &[Mat; 2], tones: &[Tone], u: f64) -> Mat {
        let n = group[0].rows;
        Mat::from_fn(n, n, |i, j| group[0][(i, j)] + group[1][(i, j)] * u + tones.iter().map(|t| C64::cis(t.beat * u) * t.plus[(i, j)] + C64::cis(-t.beat * u) * t.minus[(i, j)]).sum::<C64>())
    }

    /// A picture of the whole generator at its three nodes and its candidate tones over a step of 2h (`separate`: those
    /// of |ν| h > √5 its tones), prepared on the decomposition's route (a fresh picture has counted no other).
    fn with_tones(group: &[Mat; 2], tones: &[Tone], half: f64) -> Picture {
        let n = group[0].rows;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.base = toned(group, tones, 0.0);
        picture.ends = [-half, half].map(|u| toned(group, tones, u));
        picture.separate(half, |beats| beats.extend(tones.iter().map(|t| t.beat)), |m, plus, minus| (*plus, *minus) = (tones[m].plus.clone(), tones[m].minus.clone()));
        assert_eq!(picture.beats.len(), tones.len(), "every candidate a tone");
        picture.prepare(half, false, |_| 2, 0.0, 0.0);
        picture
    }

    /// The same generator as a static group alone, its three nodes' values: every beat through the nodes, as the
    /// propagation took it before the tones.
    fn without_tones(group: &[Mat; 2], tones: &[Tone], half: f64) -> Picture {
        let n = group[0].rows;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.base = toned(group, tones, 0.0);
        picture.ends = [-half, half].map(|u| toned(group, tones, u));
        picture.prepare(half, false, |_| 2, 0.0, 0.0);
        picture
    }

    /// Dense RK4 of iẋ = Ĥ(u)x from −h to the time `to` in `steps` steps: the reference of a beating picture's state.
    fn flowing(at: &dyn Fn(f64) -> Mat, half: f64, to: f64, start: &[C64], steps: usize) -> Vec<C64> {
        let n = start.len();
        let slope = |u: f64, y: &[C64]| -> Vec<C64> { at(u).apply(y).into_iter().map(|z| C64::new(z.im, -z.re)).collect() };
        let add = |y: &[C64], k: &[C64], s: f64| -> Vec<C64> { y.iter().zip(k).map(|(y, k)| *y + *k * s).collect() };
        let dt = (to + half) / steps as f64;
        let mut x = start.to_vec();
        for i in 0..steps {
            let u = -half + i as f64 * dt;
            let k1 = slope(u, &x);
            let k2 = slope(u + 0.5 * dt, &add(&x, &k1, 0.5 * dt));
            let k3 = slope(u + 0.5 * dt, &add(&x, &k2, 0.5 * dt));
            let k4 = slope(u + dt, &add(&x, &k3, dt));
            x = (0..n).map(|s| x[s] + (k1[s] + k2[s] * 2.0 + k3[s] * 2.0 + k4[s]) * (dt / 6.0)).collect();
        }
        x
    }

    /// The largest error of a launched picture's state at the times `at` against the flow from the step's start.
    fn beating_errors(picture: &mut Picture, generator: &dyn Fn(f64) -> Mat, start: &[C64], at: &[f64], steps: usize) -> f64 {
        let (n, half) = (start.len(), picture.half);
        let mut x = vec![C64::ZERO; n];
        picture.launch(-half, start);
        at.iter().fold(0.0, |worst: f64, &u| {
            picture.advance(u, &mut x);
            let exact = flowing(generator, half, u, start, steps);
            worst.max(x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, f64::max))
        })
    }

    /// What the treatments of a picture's beating entries leave at the step's end (module documentation, # The split):
    /// of each exact tone's ⁺ member's entries -- each pair once, its ⁻ partner the other way round -- of the envelope
    /// C + uA over a half step, s = (|C| + |A| h) h, at its turn x = |Ω| h, a hopped entry's two hops' second Magnus term
    /// x s²/3 and a followed entry's third order 2 s³ min(x, 1)/x³.
    fn beating_remainders(picture: &Picture, tones: usize) -> f64 {
        let (n, h) = (picture.n, picture.half.abs());
        let mut sum = 0.0;
        for m in 0..tones {
            let class = &picture.classes[1 + 2 * m];
            for (k, (c, a)) in class.change[0].iter().zip(&class.change[1]).enumerate() {
                let between = picture.lambda[k / n] - picture.lambda[k % n];
                let (x, s) = (C64::new(between.re + class.beat, between.im).abs() * h, (c.abs() + a.abs() * h) * h);
                sum += if class.fast[k] { 2.0 * s.powi(3) * x.min(1.0) / x.powi(3) } else { x * s * s / 3.0 };
            }
        }
        sum
    }

    /// A loop whose closing edge beats fast and turns slowly (ELIMINATION.md v8; the tilted 1D PGC's): three states,
    /// a strong chain 0–1–2 in the static group (couplings 3 and 2.5 rad/µs, the states 0, 13.7 and B − 0.5 rad/µs
    /// apart) and the weak edge 0–2 (0.3 rad/µs) closing it at the loop's beat B = 26.4 rad/µs, an exact tone (its slope
    /// a tenth of it per µs), over a step of 0.3 µs: Bτ = 7.9, the nodes τ/2 apart read the beat as its alias. On Ĥ₀'s
    /// eigenstates the edge's combined frequencies λ_a − λ_b + B are the loop's own detunings -- the edge's on the pair
    /// that carries it 0.5 rad/µs and the chain's light shifts, the rest of the order of B/2.
    /// - Taken as an exact tone, each entry at its combined frequency, the state at the step's end and inside it is the
    ///   flow within what its treatments leave (`beating_remainders`: the hops' second Magnus term, the following's
    ///   third order), from three states.
    /// - Taken through the three nodes with the static group, as before the tones, the step errs by the beat's alias:
    ///   by far more. Both are printed.
    #[test]
    fn a_loop_s_weak_edge_beats_fast_and_is_taken_at_its_combined_frequency() {
        let (n, beat, half) = (3, 26.4, 0.15);
        let static_group = Mat::from_fn(n, n, |i, j| match (i, j) {
            (0, 0) => C64::ZERO,
            (1, 1) => C64::real(13.7),
            (2, 2) => C64::real(beat - 0.5),
            (0, 1) | (1, 0) => C64::real(3.0),
            (1, 2) => C64::cis(0.3) * 2.5,
            (2, 1) => C64::cis(-0.3) * 2.5,
            _ => C64::ZERO,
        });
        let group = [static_group, Mat::zeros(n, n)];
        let edge = C64::cis(0.7) * 0.3;
        let only = |i: usize, j: usize, z: C64| Mat::from_fn(n, n, |a, b| if (a, b) == (i, j) { z } else { C64::ZERO });
        let tones = [Tone { beat, plus: only(0, 2, edge), minus: only(2, 0, edge.conj()) }];
        let generator = |u: f64| toned(&group, &tones, u);
        let starts = [[C64::ONE, C64::ZERO, C64::ZERO], [C64::ZERO, C64::ZERO, C64::ONE], [C64::real(0.6), C64::new(0.0, 0.48), C64::real(0.64)]];
        let times = [-0.5 * half, 0.4 * half, half];
        let mut picture = with_tones(&group, &tones, half);
        let mut before = without_tones(&group, &tones, half);
        assert!(picture.through && before.through);
        let within = beating_remainders(&picture, tones.len());
        let combined: Vec<f64> = (0..n * n).map(|k| (picture.lambda[k / n] - picture.lambda[k % n] + C64::real(beat)).abs()).collect();
        let (mut tone, mut nodes) = (0.0_f64, 0.0_f64);
        for start in &starts {
            tone = tone.max(beating_errors(&mut picture, &generator, start, &times, 20_000));
            nodes = nodes.max(beating_errors(&mut before, &generator, start, &times, 20_000));
        }
        eprintln!("the edge's combined frequencies {combined:.2?} rad/µs; Bτ = {:.1}", beat * 2.0 * half);
        eprintln!("as an exact tone the state errs by {tone:.2e} (its treatments leave {within:.2e}); through the three nodes by {nodes:.2e}");
        assert!(tone <= within, "{tone:e} against {within:e}");
        assert!(nodes > 20.0 * tone, "through the nodes {nodes:e}, as a tone {tone:e}");
    }

    /// The tones' gauge (module documentation, # Tones): a frame that turns a group of states at θ moves their
    /// diagonal in the static group by θ and the beat of every tone between that group and the others by θ, and
    /// leaves each combined frequency as it is. Two groups of two states, each strongly coupled within itself (with
    /// its decay and a slope along the path), and two tones between them (each two entries, with their slopes), over
    /// a step of 0.2 µs: in the frame of every θ of five, from −61 to 87 rad/µs (θτ up to 17), the state at the step's
    /// end and inside it, taken back to the first frame, is the first frame's to rounding.
    #[test]
    fn a_frame_on_a_group_of_states_leaves_the_propagated_state_as_it_is() {
        let (n, half) = (4, 0.1);
        let block = |i: usize, j: usize| i / 2 == j / 2;
        let static_group = Mat::from_fn(n, n, |i, j| match (i, j) {
            (0, 0) => C64::ZERO,
            (1, 1) => C64::new(1.5, -0.1),
            (2, 2) => C64::new(30.0, -0.05),
            (3, 3) => C64::real(32.5),
            (0, 1) | (1, 0) => C64::real(2.0),
            (2, 3) => C64::cis(0.4) * 1.8,
            (3, 2) => C64::cis(-0.4) * 1.8,
            _ => C64::ZERO,
        });
        let slope = Mat::from_fn(n, n, |i, j| if block(i, j) && i != j { C64::new(0.7, 0.1 * (i as f64 - j as f64)) } else { C64::ZERO });
        let cross = |entries: &[(usize, usize, C64)], adjoint: bool, scale: f64| Mat::from_fn(n, n, |a, b| entries.iter().find(|e| if adjoint { (e.1, e.0) == (a, b) } else { (e.0, e.1) == (a, b) }).map_or(C64::ZERO, |e| if adjoint { e.2.conj() * scale } else { e.2 * scale }));
        let first = [(0, 2, C64::real(0.4)), (1, 3, C64::cis(0.2) * 0.3)];
        let second = [(0, 3, C64::new(0.0, 0.25)), (1, 2, C64::real(0.2))];
        let tones = |theta: f64| {
            [(30.3, &first[..]), (31.0, &second[..])].map(|(beat, entries)| Tone { beat: beat + theta, plus: cross(entries, false, 1.0), minus: cross(entries, true, 1.0) })
        };
        let framed = |theta: f64| [Mat::from_fn(n, n, |i, j| static_group[(i, j)] + if i == j && i >= 2 { C64::real(theta) } else { C64::ZERO }), slope.clone()];
        let start = [C64::real(0.5), C64::new(0.1, 0.4), C64::new(-0.3, 0.2), C64::new(0.0, -0.6)];
        let times = [-0.3 * half, 0.0, 0.55 * half, half];
        // The frame's state at the time u, and back.
        let turned = |x: &[C64], theta: f64, u: f64, to: bool| -> Vec<C64> { x.iter().enumerate().map(|(s, z)| if s >= 2 { *z * C64::cis(if to { -theta * u } else { theta * u }) } else { *z }).collect() };
        let states = |theta: f64| -> Vec<Vec<C64>> {
            let mut picture = with_tones(&framed(theta), &tones(theta), half);
            assert!(picture.through);
            picture.launch(-half, &turned(&start, theta, -half, true));
            times.iter().map(|&u| {
                let mut x = vec![C64::ZERO; n];
                picture.advance(u, &mut x);
                turned(&x, theta, u, false)
            }).collect()
        };
        let reference = states(0.0);
        let mut worst = 0.0_f64;
        for theta in [-61.0, -7.3, 13.9, 42.0, 87.0] {
            let found = states(theta);
            let apart = found.iter().zip(&reference).flat_map(|(x, y)| x.iter().zip(y).map(|(x, y)| (*x - *y).abs())).fold(0.0, f64::max);
            eprintln!("θ {theta:6.1} rad/µs (θτ {:5.1}): the state off the first frame's by {apart:.2e}", theta * 2.0 * half);
            worst = worst.max(apart);
        }
        assert!(worst < 1e-12, "{worst:e}");
    }

    /// A far state driven by two beams that beat (an explicit excited state of the old engine's kind): a ground state
    /// and an excited one 600 rad/µs off with its decay (Γ = 38 /µs), two couplings of 4 and 3 rad/µs at the beams'
    /// beats 3.1 and −4.7 rad/µs (each an exact tone), over a step of 1 µs -- the beats turn 3.1 and 4.7 rad over it, so
    /// that the nodes τ/2 apart do not resolve them, and the ground state's light shift beats at their difference, 7.8
    /// rad over the step, by 2·4·3/600 = 0.04 rad/µs. The pair is followed at each beam's combined frequency, and the
    /// light shift's beat is the second order's product of the two, at its own beat (module documentation, # The second
    /// order by tones): there its turn |ν₁ − ν₂| h = 3.9 has it followed too, X⁽²⁾ = 2|c₁c₂|/(|Δ| |ν₁ − ν₂|) on the
    /// ground state. From the ground state, the state at the step's end and inside it is the flow within what the
    /// treatments leave: of the beams' own entries (`beating_remainders`); the third order that following leaves on the
    /// pair, the couplings' (|c₁| + |c₂|) X⁽²⁾ at the pair's frequency, 2(|c₁| + |c₂|) X⁽²⁾/|Δ|; and what the static
    /// group's three nodes leave of the fourth order's beats, (|c₁| + |c₂|)⁴/|Δ|³ over the step. Through the three nodes
    /// with the static group, as before the tones, the step errs by far more. Both are printed.
    #[test]
    fn a_far_state_s_two_beams_are_followed_and_their_light_shift_beats_exactly() {
        let (n, half, detuning, gamma) = (2, 0.5, -600.0, 38.0);
        let static_group = Mat::from_fn(n, n, |i, j| if (i, j) == (1, 1) { C64::new(detuning, -0.5 * gamma) } else { C64::ZERO });
        let group = [static_group, Mat::zeros(n, n)];
        let only = |i: usize, j: usize, z: C64| Mat::from_fn(n, n, |a, b| if (a, b) == (i, j) { z } else { C64::ZERO });
        let beam = |beat: f64, c: C64| Tone { beat, plus: only(1, 0, c), minus: only(0, 1, c.conj()) };
        let (c1, c2) = (C64::real(4.0), C64::cis(0.4) * 3.0);
        let tones = [beam(5.1, c1), beam(-4.7, c2)];
        let generator = |u: f64| toned(&group, &tones, u);
        let start = [C64::ONE, C64::ZERO];
        let times = [-0.5 * half, 0.2 * half, half];
        let mut picture = with_tones(&group, &tones, half);
        let mut before = without_tones(&group, &tones, half);
        assert!(picture.through && picture.followed);
        let steps = 400_000;
        let (tone, nodes) = (beating_errors(&mut picture, &generator, &start, &times, steps), beating_errors(&mut before, &generator, &start, &times, steps));
        let (couplings, beat) = (c1.abs() + c2.abs(), 5.1_f64 + 4.7);
        let second = 2.0 * c1.abs() * c2.abs() / (detuning.abs() * beat);
        let within = beating_remainders(&picture, tones.len()) + 2.0 * couplings * second / detuning.abs() + couplings.powi(4) / detuning.abs().powi(3) * 2.0 * half;
        eprintln!("as exact tones the state errs by {tone:.2e} (within {within:.2e}); through the three nodes by {nodes:.2e}");
        assert!(tone <= within, "{tone:e} against {within:e}");
        assert!(nodes > 20.0 * tone, "through the nodes {nodes:e}, as tones {tone:e}");
    }
}
