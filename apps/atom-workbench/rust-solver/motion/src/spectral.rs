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
//! `a_frame_on_a_group_of_states_leaves_the_propagated_state_as_it_is`: 10⁻¹⁵ at θτ up to 17), where a candidate takes
//! the same route in either frame -- what the static group leaves of it moves with the frame, its beat and its turns
//! do (# Which beats are tones), as the static group's own change through the nodes does. A beat the frame could
//! not take away -- a loop's, whose edges' beats sum to it whatever the frame -- costs the step nothing where the
//! loop's own detunings, the Ω of its entries, are slow: the step need not resolve the beat (tests,
//! `a_loop_s_weak_edge_beats_fast_and_is_taken_at_its_combined_frequency`: a loop closed at B = 26.4 rad/µs over a step
//! of Bτ = 7.9, its edge's Ω 0.64 rad/µs, within the hops' second Magnus term, 5.7·10⁻⁵ against 6.2·10⁻⁵; the beat
//! through the three nodes errs by 6.9·10⁻², the alias).
//!
//! # Which beats are tones
//! Which of the model's candidates (step.rs `Middle::beats`) are tones is the propagation's choice, by what each route
//! leaves on the candidate's entries at the step's end (`toned`), before the decomposition: the two routes' Ĥ₀ differ by
//! the candidate's middle value, so the choice is the candidate's whole, not an entry's. A beat e^{iνu}C taken into the
//! static group puts C into Ĥ₀ and its change (e^{iνu} − 1)C through the three nodes, whose parabola has over a half step
//! the slope s_C|sin y| and the curvature s_C(1 − cos y) exactly (y = νh, s_C = |C| |h|): the split takes it by those at
//! the entry's turn x = |ω| |h| (# The split), and the parabola misses the beat in the first order of C -- of a hop by
//! s_C|β(ωh, y)|, β(x, y) = ∫₋₁¹ e^{ixt}[e^{iyt} − p_y(t)]dt of the hop's own moments, β(0, y) = −y⁴/90 + … (Simpson's
//! remainder: `aliased`); of a following by its polynomial against the beat's own following at ω + ν. Taken as a tone it
//! is the envelope C whole at its combined frequency, x′ = |ω + ν| |h|, the split's route on it. The turns are of the
//! generator's diagonal at the middle in its frames -- the zeroth order in the static coupling, no decomposition: a
//! strongly dressed pair's route is estimated, which costs accuracy alone, the check reading either route. A candidate
//! is a tone where the static group leaves more, and wherever |ν| |h| ≥ π: three nodes do not resolve a phase of more
//! than π between two of them, and beyond it the static group's miss is the alias that the step and its companion at
//! twice the step read alike (at νh = 2πk the same) -- a bound of the nodes, no number of the rule. Where the two leave
//! the same they swap, so nothing jumps there; their crossing is no number either: for a weak coupling of an entry that
//! does not turn it lies near y³ = 30 s_C (y = 0.67 at s_C = 0.01, 1.5 at 0.1), and from s_C ≈ 0.4 on the static group
//! leaves less up to π (DESIGN §12.5). A slow beat of a strong coupling -- a 3D lattice's residual q·v at the steps the
//! scenes take -- stays in the static group, a loop's weak closing edge at its fast beat is a tone. (Until 2026-11-16 the
//! rule was νh = √5: the slope taken as |νC|h², no curvature, no alias and one turn for both routes; with the exact slope
//! and curvature the static group's hops leave less at one turn whatever y -- sin²y/15 + (1 − cos y)²/42 ≤ 0.104 < 1/3
//! --, the routes crossing by the alias and by their turns' difference alone.)
//!
//! A tone is its candidate whole: both members as the model gives them, the Hermitian part and the decay's beat alike
//! (`Picture::separate`). The static group then holds the decay's mean over the beats, Γ_s (of Γ(u) = Σ_{o,o′}
//! e^{i(o − o′)u} L_o†L_o′ it is Σ_o L_o†L_o, positive), and no decay of a tone twice. The members take the routes the
//! Hermitian ones do: `take`, `following` and `exponent` are of an entry's complex numbers whatever G⁻ is to G⁺ -- a fast
//! entry's following solves for a non-Hermitian envelope as for a Hermitian one, no exponential in it; a slow entry's
//! hop is the first Magnus term −i∫D̃ (# The state at any time of the step), Λ exact. A beating decay's part alone,
//! −(i/2)(e^{iνu}Γ_ν + e^{−iνu}Γ_ν†), is no decay -- Γ_ν is not positive (the interference of two plane waves'
//! scattering), its sign turns with its beat --, so a hop's first Magnus term of it is not bounded by the static
//! decay's, and the propagation's norm may rise by it as it rises by the decomposition's change of basis: the
//! propagation is no flow that only lowers the norm (step.rs, # Jumps), and in no basis is it one while Λ keeps its
//! exact decays (DESIGN §12.4). It grows where the first Magnus term is itself wrong -- a strong beating decay at a
//! slow combined frequency, |Γ_ν| w ≳ 1 --, which the check reads. The survival is the law's, not the propagation's
//! norm: its hazard is the decay of the generator the propagation takes, each tone at its exact phase on the
//! quadrature's states (step.rs `Sums`), Σ L(u)†L(u) at each node where the envelopes are constant along the path --
//! positive semidefinite by construction -- and to the parabola's third order where they change, as the static group's.
//! (Through the three nodes with the static group, as the decay's beats were taken before 2026-11-16, the beat beyond
//! νh = π was its alias: the interference kept or dropped whole, and read the same by the step and by its companion
//! at twice the step where νh = 2πk -- a silent error of the scattering rate up to the interference term's size; and
//! three offsets' parabola went negative between the nodes, 7 % of the decay, raising the survival.)
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
//! step) and its product of n² (`Picture::at`), no hop --, and at the three stops it is the hops' own. The n
//! exponentials are four at a time (`linalg::times_exp`: within 4u of the library's, at a quarter of its time), where
//! one by one they were most of a state's time at the scenes' n (8 to 20 states: 55 to 65 % of it, 9 to 13 ns each).
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
//!   constant −(|C|²C/Ω²) at the entry's turn, (s_C/x)³ |∫₋₁¹ e^{ixt} dt| = 2 s_C³ |sin x|/x³ ≤ 2 s_C³ min(x, 1)/x³
//!   (2|sin x| = |∫₋₁¹ x e^{ixt} dt|: within its length's 2x and the lemma's 2x/x, model/classify.rs `by_parts`).
//!   They fall with the detuning, the slope's and the curvature's with the step. (A hop through an unresolved pair left
//!   the light shift of the change, (k·v τ)²/24 of the light shift's turn over the step at any detuning.)
//! (step/tests.rs, `a_step_is_of_the_fourth_order_in_the_path_s_change_where_its_hops_resolve_the_generator`.)
//!
//! # The split
//! Each entry takes one of three routes (`followed`): each route is a series, taken where it converges, and where both
//! do, by what each leaves at the step's end. The two series' radii -- conditions of convergence, no numbers to tune:
//! - the following's orders X⁽ᵏ⁾ ~ rᵏ (# The state at any time of the step) are the block diagonalization of Λ + D's
//!   fast entries by 1 + X, a series in the change over the gap -- Schrieffer–Wolff's, or Bloch's, which converges where
//!   the change is below the gap (Kato's sufficient condition ‖D‖ < gap/2; Bravyi, DiVincenzo & Loss 2011, Ann. Phys.
//!   326, 2793) --: on the entry r < 1, r the following's size there, |Y| at the step's ends of what is followed --
//!   a first-order class's change, a product's (# The second order by tones) own second order --, the larger of its two
//!   ways round. At r ≥ 1 1 + X is no near identity and D₂'s fast entries left out are not of a higher order;
//! - a hop's first Magnus term is the first of a series that converges where ∫‖Ã(t)‖₂ dt < π over the hop (Moan &
//!   Niesen 2008, Found. Comput. Math. 8, 291: the sufficient condition), Ã the whole hopped operator. A constant
//!   similarity S takes the series to S(·)S⁻¹ term by term, so it converges where the condition holds for one S
//!   Ã S⁻¹ -- of every hopped entry together, a state's own entry too, which commutes with itself but not with the
//!   others (`radius`, read on the hop's own envelope as the following leaves it, `Picture::hops_radius`). An entry's
//!   own share, of the entry and its partner alone, Ã = [[0, Ã_ab], [Ã_ba, 0]] balanced by S = diag(d, 1) at the hop's
//!   middle, ‖SÃ(t)S⁻¹‖₂ = √(|Ã_ab||Ã_ba|) e^{|Im Ω| |t − t_m|} within s = √(s_ab s_ba) e^{|Im Ω| |h|/2} over the hop
//!   (s_ab = S_C + S_A/2 + S_B/3 of its envelope's sizes over a half step, a hop's widest), is the route's rule on the
//!   entry below: s < π is the pair's series' radius, which the whole's holds only with every other entry beside it.
//!   (Until 2026-11-21 the pair's radius was taken for the whole's and a state's own entry was let through as
//!   commuting with itself: on R4 at 1.896 µs a hop's exponent of 2-norm 11 to 20, its diagonal's first-order shifts
//!   of two ill-conditioned modes growing by e^{3.7} to e^{11}, took the propagation's norm from 1 to 10^66 inside a
//!   piece -- each pair's s within π --, and the normalized state into that mode: tens of jumps a step.)
//! Where both converge the route is the one that leaves less, by the leading remainders
//!   F = (2/3)(S_A/x)³ + 2 S_C³ min(x, 1)/x³    (the following's third order),
//!   L = (4/15) S_B²/x                            (the parabola's miss of the change's own light shift),
//!   J = x (S_C²/3 + S_A²/15 + S_B²/42)           (the hops' second Magnus term),
//! the following's orders a series in r, their sum F/(1 − r). (An entry nothing changes on has r = 0: its following is
//! none, whatever the sizes the rule reads.)
//! - **followed** where the following converges, r < 1, and either the hops do not (s ≥ π) or the following leaves no
//!   more than they do: F/(1 − r) + L ≤ J -- of a product (# The second order by tones) F with the third order its
//!   following X⁽²⁾ leaves on the entries the first order's couplings carry it to, c X⁽²⁾, at most 2 S r over the step
//!   (S the couplings' size over a half step: measured where it was left out, 3.3·10⁻² of a share of 4.5·10⁻² on
//!   PGC3D's steps, up to 39 at a few);
//! - **hopped** where the hops converge, s < π, and either the following does not (r ≥ 1) or the hops leave less:
//!   J < F/(1 − r) + L -- the hops hold where the entry turns little between the nodes, which is where J is the less
//!   --, and an entry that does not turn (x = 0). J is at the entry's modulus in the interaction picture, which grows
//!   by e^{|Im Ω| 2|h|} over the step (`followed`): a pair apart in decay is hopped only where its hops leave less even
//!   so, and the hop's exponent is summed and priced at that growth (`Picture::moved`);
//! - **neither**, the step refused (`Refusal::Radius`; # Where the decomposition does not take the step), where neither
//!   series converges, r ≥ 1 and s ≥ π: a change over a half step beyond both the entry's gap and π -- S > max(x, π)
//!   in its sizes, as both radii fall with the step: S_A ~ h², x ~ h (stepcost.status.md). The advance takes such a
//!   step at its half (adapt.rs, # The step), which brings it within them. (Until 2026-11-20 neither was where r ≥ 1
//!   and the hops left more than the following's leading term, J ≥ F + L: a comparison of remainders, at all the
//!   first-order classes' sizes together, where the series may not converge at all.)
//! (Without the following's convergence a tone of a coupling stronger than its combined frequency, |C| > |Ω|, was
//! followed at a step of |Ω||h| beyond a few radians, where F < J: the state through it grew by 10⁶ over a step of 10 µs
//! and overflowed at 90 µs, tests, `the_survival_never_rises_whatever_the_generator_and_the_step`; a static group's
//! slope is followed at r ≪ 1 at the scenes' steps, where nothing changed, tests,
//! `the_static_group_alone_is_propagated_as_before_to_the_bit`.)
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
//! light shift); of two members otherwise their sum's or their difference's -- the light shift's beat between two beams
//! that drive one far state, its lattice along the path. The class of a product is its beat's, exactly: each tone comes
//! with an exact label (step.rs `Middle::beats`: the field's integers, its residual beat's and wave vector's), and two
//! products of one label sum are one class -- the static group where the sum is zero, a first-order class where it is
//! that class's label, else a pair of their own (`Picture::classify`). Their count is the field's: the distinct sums of
//! two of its components' labels, not the pairs of tones (GMD1C's 92 tones of a step, split by the frames' least squares
//! before, made 17 000 classes; a picture holds (1 + 2T + 2P) classes of n² each, T the tones and P the distinct sums).
//! Each product is made at the nodes on
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
//! How far the classes merge. The harmonics of one beat ν -- tilt2's four tones at ν, 2ν, 3ν and 4ν (ν = 13.19 rad/µs,
//! its tilt's Larmor beat) -- are one periodic envelope G(u) = Σ_k e^{ikνu} G_k(u), and its following is harmonic by
//! harmonic: Ω_ab Y_ab − iẎ_ab + E_ab = 0 at each class's combined frequency Ω = ω_ab + kν (# The state at any time of
//! the step), the polynomial of `following` -- X_k = −D_k/(ω + kν) at its leading order --, which nothing here changes.
//! With exact labels the products fall into the classes of their harmonics' sums: tilt2's first order ±1…±4 ν and its
//! products ±2…±8 ν, ±1…±3 ν and 0 make the classes 0, ±1…±8 ν -- 17, of which 16 beat and are followed. A state at a
//! time is then R(1 + X(u)) d assembled once from those 16 (`at`: (2 + 4·16)n² multiply-adds and 4n, against
//! (2 + 4·40)n² with the products of one beat apart); one class for the whole periodic envelope would not take less: each
//! harmonic is three matrices of n², the same in a sum over k as in a sum over classes, and of the harmonics' phases
//! z^k = e^{ikνu} only the exponentials would be shared, n a point against 16. Measured where the model labelled its
//! candidates by unit vectors (until 2026-11-21: a sum of two keys zero only for a candidate's two members, so that
//! 4ν − 3ν and ν were two classes of one beat): tilt2's 41 classes, 40 of them followed, at 17.2 ms a step (one thread,
//! τ 0.326 µs); labelled by the exact vectors 17 and 16 at 10.1 ms, the steps from the same 64 states within 3.7·10⁻⁸
//! (2.2–6·10⁻¹⁶ on the median; a few entries change their route with the merged envelope) (motion3d_status/rewrite/
//! r2prop2.status.md).
//!
//! # What follows the generator's own frequencies, and what does not
//! The propagation's own work -- the decomposition, the following, two hops -- is the same whatever Ĥ₀'s frequencies
//! times the step are. The step's forms are read on the state at times of a quadrature (step.rs `propagate`), and
//! their number still follows the forms' bandwidth -- max|λ_a* − λ_b| over the states the propagated state can hold
//! (`Picture::reach`: those its launch holds and those the generator couples them to; an excited state no light
//! drives stays empty, and no form holds its frequencies -- until 2026-11-17 an explicit cluster's six idle excited
//! states, a thousand rad/µs off and empty, set its bandwidth at 1010 rad/µs where its two live states' was 30); of a
//! tone's forms, summed at its beat
//! (step.rs `Sums`), the combined frequencies max|λ_a* − λ_b ± ν| -- a tone far from every pair's frequency asks the
//! quadrature its combined frequency, not its beat: tilt2's loop's weak edge beats at 26 rad/µs and its pairs at
//! 0.4–1.2 --; and where a beating class is followed the beats' spread with it, the following moving the state's
//! frequencies by its beat: the kick is a quotient by the state's
//! norm, which with a decay is a sum of the pairs' exponentials of its own -- ringing wherever the state holds a fast
//! pair's two states --, and a quotient of two such sums has no closed form. (The forms on the unnormalized state have
//! one, pair by pair: the moments above. They would serve where the norm is one, which is no cluster that scatters.) A
//! state there costs n exponentials and two products of n², against a hop's series and its exponential before. The
//! quotient's own bandwidth, the forms' with the norm's ringing and its harmonics to the round-off, is the launch's
//! (`Picture::quotient`, `quotient_from`): what the step's quadrature is laid at (step.rs, # The quadrature's tolerance).
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
//! # Where the decomposition does not take the step
//! The state goes through the decomposition alone. Where it cannot take a step, the step is refused (`Refusal`; step.rs
//! `Refused::Propagation`) and the advance takes it at its half (adapt.rs, # The step) -- no other route:
//! - Ĥ₀'s QR iteration does not converge (`Refusal::Decomposition`);
//! - towards a defective generator (`Refusal::Rounding`): a state taken through the eigenvectors rounds by
//!   κ(R) u (1 + |τ| × the decay rates' spread) relative to its norm (`linalg::Eigen`), and κ(R) grows without bound
//!   towards an exceptional point. The rule is the rounding's, not a tuned number: the decomposition takes the step
//!   where a state through it rounds no more than through the exponential of Ĥ₀ over the step
//!   (`linalg::Chebyshev::rounding`);
//! - an entry taken by no route of the decomposition, or the hops' series not converging on the whole hopped operator
//!   (`Refusal::Radius`; # The split): a step far too long for the change it holds;
//! - the propagation's norm above the launch's at its stops past the launch -- the piece's middle and end -- beyond
//!   the state's rounding (`Refusal::Growth`, `Picture::rounding`; read by the step, `Midway::stretch`): the flow of a
//!   dissipative generator is a contraction, d‖x‖²/dt = −⟨x|Γ|x⟩ ≤ 0, and the propagation has left it. A backstop
//!   beside the radii, not a route's rule: on the six scenes' walks at 0.474 and 0.948 µs no stop rose at all where the
//!   radius took the step; where it rises, the following's truncation at r near one (1 + X far from the identity) or
//!   a parabola whose decay dips between the nodes is what is left.
//! The half step is the cure of the last two; where the first two do not leave at any length of the step, the step
//! is refused down to the time's round-off and the trajectory ends with no outcome, which the solve counts (solve.rs
//! `Counts`). Forced through the decomposition the scenes' steps at their converged step and twice it -- 2.3·10⁵ of
//! them over R4, PGC3D, TET, GMD1C, S1, tilt0 and tilt2 -- met the first two on none, the third on R4 at 0.474 µs
//! (7 steps, 0.03 %) and GMD1C at 1.897 µs (4, 0.02 %) (motion3d_status/rewrite/stepcost.status.md).
//!
//! (Until 2026-11-20 such a step, and any step where they were priced below the decomposition, took the state without
//! it: CF4 from stop to stop of the quadrature, two exponentials by Taylor's series to the round-off. Removed with the
//! price that chose them: the price sent 97–99 % of tilt's steps to the strides where the decomposition was 6.7 to 13
//! times cheaper, and a state inside the step -- a jump's root, its time and its state -- was taken by one stride from
//! the latest stop before it, over up to a whole step: 10⁻³ to 1.6 off its flow on R4 at 1.9 µs, 3.6·10⁻² on tilt0 at
//! its converged step, the error growing with the span (stepcost.status.md).)

use crate::linalg::{self, combine, C64, Mat, LANES};
use crate::units::UNIT_ROUNDOFF;

/// Where row `row` of column `column` lies among `count` columns laid for the lanes -- each column's rows in blocks of
/// `LANES`, a block's real parts and then its imaginary parts --, from their start: its real part, its imaginary part
/// `LANES` after it.
pub(crate) fn laid(row: usize, column: usize, count: usize) -> usize {
    ((row / LANES) * count + column) * 2 * LANES + row % LANES
}

/// Why the decomposition does not take a step (module documentation, # Where the decomposition does not take the
/// step): the step is refused (step.rs `Refused::Propagation`), and the advance takes it at its half.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// Ĥ₀'s QR iteration did not converge (`linalg::Eigen::decompose`).
    Decomposition,
    /// A state through Ĥ₀'s eigenvectors rounds by more than through its exponential over the step: towards a
    /// defective generator.
    Rounding,
    /// An entry taken by no route of the decomposition, or the hops' series not converging on the whole hopped operator
    /// (# The split).
    Radius,
    /// The propagation's norm above the launch's beyond the state's rounding (`Picture::rounding`): a dissipative
    /// generator's flow is a contraction, d‖x‖²/dt = −⟨x|Γ|x⟩ ≤ 0, so the propagation has left it (step.rs
    /// `Midway::stretch` reads it at the propagation's stops).
    Growth,
}

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

/// to += p from, of laid matrices (`add_turned` of a matrix that does not change).
fn add_phased(p: C64, from: &[f64], to: &mut [f64]) {
    for (chunk, from) in to.chunks_exact_mut(PAIR).zip(from.chunks_exact(PAIR)) {
        for l in 0..LANES {
            let (re, im) = (from[l], from[LANES + l]);
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
    /// Its exact label (step.rs `Middle::beats`): its products' class is their labels' sum's.
    pub key: Vec<i64>,
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

/// The nodes' work an evaluation of the strip's V(y) costs at most (`Picture::strip`): its three points' two products by
/// R, 2n² complex multiply-adds a point, against a node's product by R and its sums, no fewer than 2n² (step.rs
/// `Sums`): three.
const EVALUATION: f64 = 3.0;

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
    keys: Vec<Vec<i64>>,
    /// The candidates `separate` weighed and left in the static group (their members read at the middle), and the
    /// complex multiply-adds of its weighing.
    pub weighed: Vec<usize>,
    pub weighing: f64,
    /// After `prepare` on a beating path, the static group's slope at the middle through the step's ends,
    /// (G₀(h) − G₀(−h))/2h, as the caller's matrices are laid out: the first order of the parabola it is taken for (the
    /// optical sector's along the path, step.rs).
    pub slope: Mat,
    /// A hop's exponent.
    matrix: Mat,
    /// Ĥ₀'s decomposition of its traceless part.
    basis: linalg::Eigen,
    /// The model's exact blocks, the caller's (step.rs from `Middle::blocks`; none: the states one block): every matrix
    /// the step is given is block-diagonal on them (model/selected.rs, # Exact blocks). Ĥ₀ is decomposed block by
    /// block (`linalg::Eigen::decompose_blocks`) and each eigencomponent is of one block (`part`): no pair of two blocks
    /// is a frequency of any form, of any class, or of the quadrature's bandwidth -- the forms are block-diagonal too.
    pub blocks: Vec<Vec<usize>>,
    part: Vec<usize>,
    scratch: Vec<(Mat, linalg::Eigen)>,
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
    /// Whether D is anything at all, and the step's half h; and whether every envelope is its constant over the step
    /// (`separate`'s `held` with every candidate a tone): the slopes and curvatures of every class, and so the
    /// following's Y₁ and Y₂, are zero, and nothing is made of them.
    moving: bool,
    half: f64,
    constant: bool,
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
    /// max|Ω| over the hopped entries of every class (rad/µs): what a hop's series is summed against.
    spread: f64,
    /// max e^{|Im Ω| 2|h|} over the hopped entries: their growth in the interaction picture over the step (`followed`),
    /// a factor of a hop's exponent's bound (`moved`).
    grown: f64,
    /// The hopped entries' moduli, the whole hopped operator's (`radius`): |C|, |A| and |B| at their growth, n² each.
    moduli: Vec<f64>,
    /// The bounds of Ĥ₀ less its scalar part (`linalg::bounds`): what the decomposition's rounding is weighed against
    /// (`prepare`). And the Frobenius norms of D₂'s constants, slopes and curvatures as the
    /// hops take them, summed over the classes: what bounds a hop's exponent (`moved`).
    bounds: [f64; 2],
    slopes: [f64; 3],
    /// The states the propagated state can hold (`reach`): every state a launched state holds and every state the
    /// generator couples them to, at any node or by any tone -- the others stay empty, and nothing of them is in the
    /// forms. Their indices; whether each of Ĥ₀'s eigenvectors is theirs (the larger part of it on them); and the
    /// generator's coupling of two states at any node or by any tone (n²).
    reached: Vec<bool>,
    within: Vec<usize>,
    held: Vec<bool>,
    coupled: Vec<bool>,
    /// The bandwidth of the forms ⟨x|·|x⟩ on a state propagated by Ĥ₀ and followed (rad/µs): max|λ_a* − λ_b| -- their
    /// frequencies and decay rates -- and the followed classes' beats' spread.
    pub bandwidth: f64,
    /// The bandwidth of the quotients ⟨x|·|x⟩/⟨x|x⟩ the step integrates over what is left of it from the last launch
    /// (rad/µs; `quotient`, # What follows the generator's own frequencies): the forms' with the norm's ringing and its
    /// harmonics to the round-off. And the eigenvectors' overlaps |(R†R)_ab| (R's columns of unit norm), n², what the
    /// norm rings by.
    pub quotient: f64,
    overlaps: Vec<f64>,
    /// The quotient's terms' items, (weight, frequency): the numerator's pairs and the norm's ringing (`quotient_from`),
    /// the ringing's pairs (a, b), and whether the last quotient made them (`ringing`).
    numerator: Vec<(f64, f64)>,
    ringing: Vec<(f64, f64)>,
    pairs: Vec<[u8; 2]>,
    rung: bool,
    /// The numerator's items with their pairs, as they were made (`strip` sorts the numerator's).
    numerated: Vec<([u8; 2], f64)>,
    /// Where the envelopes' last search for their bound ended (`envelope`; none before the first).
    warm: f64,
    /// The complex multiply-adds counted so far (`decide::Work` counts them four real ones each).
    pub work: f64,
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
            keys: Vec::new(),
            weighed: Vec::new(),
            weighing: 0.0,
            slope: none(),
            matrix: none(),
            basis: linalg::Eigen::default(),
            blocks: Vec::new(),
            part: Vec::new(),
            scratch: Vec::new(),
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
            constant: false,
            followed: false,
            factors: Vec::new(),
            pivots: Vec::new(),
            factored: f64::NAN,
            hopped: Vec::new(),
            path: Vec::new(),
            origin: 0.0,
            centre: 0.0,
            spacing: 0.0,
            spread: 0.0,
            grown: 1.0,
            moduli: Vec::new(),
            bounds: [0.0; 2],
            slopes: [0.0; 3],
            reached: Vec::new(),
            within: Vec::new(),
            held: Vec::new(),
            coupled: Vec::new(),
            bandwidth: 0.0,
            quotient: 0.0,
            overlaps: Vec::new(),
            numerator: Vec::new(),
            ringing: Vec::new(),
            pairs: Vec::new(),
            rung: false,
            numerated: Vec::new(),
            warm: f64::NAN,
            work: 0.0,
        }
    }
}

/// How a term that turns x = |Ω| |h| between two of the step's nodes is taken (module documentation, # The split), the
/// changes over a half step being of the sizes `slope` = S_A, `curvature` = S_B and `constant` = S_C, and `hop` the
/// term's own hop exponent's least 2-norm over a hop (module documentation, # The split: its two ways' geometric mean
/// with half the hop's growth): each route where its series converges -- the following where its size r < 1, the hops
/// where `hop` < π (Moan & Niesen's radius of the Magnus series) -- and where both do by what each leaves: followed (true) where
/// what its following leaves, (2/3)(S_A/x)³ + 2 S_C³ min(x, 1)/x³ -- the leading term of the following's orders, a
/// series in r, so over 1 − r -- and (4/15) S_B²/x, is no more than what its hops would, x (S_C²/3 + S_A²/15 +
/// S_B²/42) -- the two times x³ --, hopped (false) where it is more; a term that does not turn is hopped where its hops
/// converge. Taken by no route of the decomposition (None) where neither series converges (`Picture::prepare` refuses
/// the step then, `Refusal::Radius`). r is the following's size on the entry, |Y| at the step's ends (`flag`). A
/// product's following X⁽²⁾ (# The second order by tones) leaves besides the third order c X⁽²⁾ on the entries the
/// first order's couplings carry it to, `coupled` the couplings' size over a half step, S_C + S_A + S_B: at most 2 S r
/// over the step (the left-out term's turn there, min(x′, 1)/x′, at most one) -- the following's leading term with it.
/// With the route, what it leaves at the step's end (over x³): the route's own, the least of the two where both
/// converge; the following's leading term where neither does (the least any following could leave there); none where
/// the term does not turn (its hops' second term is x times the sizes').
///
/// A hopped entry is the hops' in the interaction picture of Ĥ₀ (`Picture::turn`): turned to the launch, its modulus is
/// e^{−Im Ω s} of the unturned one, s from the launch to its time -- at most the step, 2|h| --, and its partner's
/// e^{+Im Ω s}. Every order of the Magnus series carries on the entry (a, b) the factor of its two states (in a product
/// of the turned matrix the intermediate states' factors cancel but for their own decay over the hop's width), so the
/// hops' remainder on it is the one at |e^{iΩs}| = 1 times e^{|Im Ω| 2|h|}, `decay` = |Im Ω| |h| -- and so is the hop's
/// exponent itself, which its radius is read on and the hop's exponential is summed and priced by (`Picture::moved`).
/// The following takes Ω whole and grows by nothing. (Until 2026-11-18 the hops were read at a unit modulus: on S2Q an
/// F′ = 3 mode and a ground mode within a rad/µs of each other and Γ/2 apart in decay were hopped, their exponent 10² to
/// 2.6·10⁶, the first Magnus term far out of its range and the hop's Chebyshev series up to 1.3·10⁶ parts.)
fn followed(x: f64, decay: f64, [slope, curvature, constant]: [f64; 3], r: f64, coupled: f64, hop: f64) -> (Option<bool>, f64) {
    let hops_converge = hop < std::f64::consts::PI;
    if !(x > 0.0) {
        return (hops_converge.then_some(false), 0.0);
    }
    let (first, second, zeroth) = (slope * slope, curvature * curvature, constant * constant);
    // 2|sin x| = |∫₋₁¹ x e^{ixt} dt| within its length's 2x and the lemma's (module documentation, # The state at any time).
    let turned = linalg::smaller(2.0 * x, crate::model::classify::by_parts(2.0 * x, 0.0, C64::real(x)));
    let leading = 2.0 / 3.0 * first * slope + zeroth * constant * turned + 2.0 * coupled * r * x.powi(3);
    let hops = x.powi(4) * (zeroth / 3.0 + first / 15.0 + second / 42.0) * (2.0 * decay).exp();
    let (light, cube) = (4.0 / 15.0 * second * (x * x), x.powi(3));
    match (r < 1.0, hops_converge) {
        (true, true) => {
            let following = leading / (1.0 - r) + light;
            if following <= hops { (Some(true), following / cube) } else { (Some(false), hops / cube) }
        }
        (true, false) => (Some(true), (leading / (1.0 - r) + light) / cube),
        (false, true) => (Some(false), hops / cube),
        (false, false) => (None, (leading + light) / cube),
    }
}

/// An entry's hop exponent's bound over a hop (`followed`, module documentation # The split), of its envelope's own
/// sizes over a half step `own` = [S_A, S_B, S_C]: s = S_C + S_A/2 + S_B/3, the most ∫|C + uA + u²B| du over a hop of
/// at most h from the step's start or its middle.
fn hop_size([slope, curvature, constant]: [f64; 3]) -> f64 {
    constant + 0.5 * slope + curvature / 3.0
}

/// The terms of ∫₀¹ t^p e^{zt} dt's series Σ_j z^j/(j!(j + p + 1)) at |z| ≤ x for an integral of the size `size` at
/// most: the least T with x^T/T! times the size within the unit round-off, the size no more than one -- the terms from T
/// on are within the round-off of the state the integral acts on, or of its own first term where that is more. A hopped
/// entry's |z| is below the split (`followed`); one whose terms leave the numbers (|z| beyond some 700) has no sum here,
/// and is refused at once rather than summed without end.
fn series(x: f64, size: f64) -> usize {
    let (mut terms, mut next) = (1, x * linalg::smaller(size, 1.0));
    while next > UNIT_ROUNDOFF {
        assert!(next.is_finite(), "a hop's series at |z| = {x} leaves the numbers");
        terms += 1;
        next *= x / terms as f64;
    }
    terms
}

/// The moments I_p(z) = ∫₀¹ s^p e^{zs} ds, p = 0, 1, 2, by their series Σ_j z^j/(j!(j + p + 1)) to `terms` terms (`series`),
/// as the series' even and odd terms in j apart: I_p(z) is their sum, I_p(−z) their difference -- an entry's hop and its
/// partner's at once (module documentation). `reciprocals` holds the whole numbers' 1/k as far as a series went, and
/// is lengthened as this one asks.
fn moments(z: C64, terms: usize, reciprocals: &mut Vec<f64>) -> [[C64; 3]; 2] {
    while reciprocals.len() < terms + 3 {
        reciprocals.push(1.0 / reciprocals.len() as f64);
    }
    let (mut power, mut parts) = (C64::ONE, [[C64::ZERO; 3]; 2]);
    for j in 0..terms {
        // 1/(j + p + 1) for the moment p.
        let inverse = &reciprocals[j + 1..j + 4];
        for (moment, inverse) in parts[j % 2].iter_mut().zip(inverse) {
            *moment += power * *inverse;
        }
        power = power * z * inverse[0];
    }
    parts
}

/// What the three nodes' parabola of a beat leaves of a hop's integral over the step (module documentation, # Which
/// beats are tones): β(x, y) = ∫₋₁¹ e^{ixt}[e^{iyt} − p_y(t)] dt, p_y(t) = 1 + it sin y + t²(cos y − 1) the parabola
/// through e^{iyt} at t = −1, 0, 1, of an entry turning by x = ωh (complex: its decay rates' difference with it) and a beat
/// by y = νh between two nodes; first order in the beat's coupling C, |C| |h| |β| at the step's end. Of the hop's own
/// moments over the two half steps: M_p(a) = ∫₋₁¹ t^p e^{iat} dt = I_p(ia) + (−1)^p I_p(−ia), twice the series' even part
/// for p even and its odd part for p odd (`moments`), and β = M₀(x + y) − M₀(x) − i sin y M₁(x) − (cos y − 1) M₂(x);
/// β(0, y) = 2 sin y/y − 2 + (2/3)(1 − cos y) = −y⁴/90 + …, Simpson's rule's remainder. With the series' terms summed.
fn aliased(x: C64, y: f64, reciprocals: &mut Vec<f64>) -> (f64, usize) {
    let mut summed = 0;
    let mut at = |a: C64| {
        let z = C64::I * a;
        let terms = series(z.abs(), 1.0);
        summed += terms;
        moments(z, terms, reciprocals)
    };
    let ([even, odd], [turned, _]) = (at(x), at(x + C64::real(y)));
    (((turned[0] - even[0] - odd[1] * C64::new(0.0, y.sin()) - even[2] * (y.cos() - 1.0)) * 2.0).abs(), summed)
}

/// Whether a candidate e^{iνu}G⁺ + e^{−iνu}G⁻ is a tone of the step of 2h (module documentation, # Which beats are
/// tones), its members' envelopes at the middle `members` and the generator's diagonal at the middle `diagonal`: by what
/// each route leaves on its entries at the step's end, summed over its members' entries that are not zero -- each entry
/// (s, t) at its turn of the diagonal, ω = Ĥ_ss − Ĥ_tt (the zeroth order in the static coupling, no decomposition), and
/// its envelope's size over a half step s_C = |C| |h|; the beat at ±ν by its member, y = νh:
/// - as a tone, at its combined frequency ω + ν, x′ = |ω + ν| |h|: the split's route on its envelope C (`followed`, the
///   following's size r = |C|/|ω + ν|);
/// - in the static group, C in Ĥ₀ and the change (e^{iνu} − 1)C through the three nodes, at the turn x = |ω| |h|: the
///   split's route on the parabola's slope s_C|sin y| and curvature s_C(1 − cos y), and the beat the parabola misses --
///   of a hop s_C |β(ωh, y)| (`aliased`); of a following the parabola's following against the beat's own
///   at its combined frequency at the step's ends, |−C/ω + Y(±h) + e^{±iy}C/(ω + ν)|, Y the polynomial of the change
///   (`following`): first order in C, both.
/// An entry a route takes by no series of the decomposition (`followed`: neither converges) leaves all on that route.
/// A tone where the static group leaves more, and wherever |ν| |h| ≥ π: the three nodes do not resolve a phase of more than
/// π between two of them, and there the static group's miss of the beat is what the step and its companion at twice the
/// step read alike (at νh = 2πk the nodes of both hold the beat's same values) -- the bound is the nodes', no number of
/// the rule. Where the two leave the same (the routes' crossing, and a candidate of no envelope at the middle) the static
/// group: the routes swap where they leave the same, so nothing jumps there. Returns the weighing's complex multiply-adds
/// with it.
fn toned(diagonal: &[C64], beat: f64, half: f64, members: [&Mat; 2], reciprocals: &mut Vec<f64>) -> (bool, f64) {
    let h = half.abs();
    if !(beat.abs() * h < std::f64::consts::PI) {
        return (true, 0.0);
    }
    let n = diagonal.len();
    let (mut nodes, mut tone, mut work) = (0.0, 0.0, 0.0);
    for (member, nu) in members.into_iter().zip([beat, -beat]) {
        let y = nu * half;
        let (sin, cos) = y.sin_cos();
        for (k, &c) in member.data.iter().enumerate().filter(|(_, c)| **c != C64::ZERO) {
            let (s, t) = (k / n, k % n);
            let (omega, size) = (diagonal[s] - diagonal[t], c.abs() * h);
            let combined = omega + C64::real(nu);
            // As a tone: its envelope at its combined frequency.
            let r = if combined != C64::ZERO { c.abs() / combined.abs() } else { 0.0 };
            let (route, left) = followed(combined.abs() * h, combined.im.abs() * h, [0.0, 0.0, size], r, 0.0, size * (0.5 * combined.im.abs() * h).exp());
            tone += if route.is_some() { left } else { f64::INFINITY };
            // In the static group: the parabola's slope and curvature at the entry's turn, and the beat it misses.
            let (a, b) = (c * C64::new(0.0, sin / half), c * ((cos - 1.0) / (half * half)));
            let y_poly = if omega != C64::ZERO { following(omega.recip(), C64::ZERO, a, b) } else { [C64::ZERO; 3] };
            let at = |u: f64| y_poly[0] + y_poly[1] * u + y_poly[2] * (u * u);
            let r = [-half, half].into_iter().map(|u| at(u).abs()).fold(0.0, linalg::larger);
            let changed = [size * sin.abs(), size * (1.0 - cos), 0.0];
            let (route, left) = followed(omega.abs() * h, omega.im.abs() * h, changed, r, 0.0, hop_size(changed) * (0.5 * omega.im.abs() * h).exp());
            if route.is_none() {
                nodes = f64::INFINITY;
            }
            let missed = if route == Some(true) {
                [-half, half].into_iter().map(|u| (at(u) - c / omega + C64::cis(nu * u) * c / combined).abs()).fold(0.0, linalg::larger)
            } else {
                let (beta, terms) = aliased(omega * half, y, reciprocals);
                work += (4 * terms) as f64;
                size * beta
            };
            nodes += left + missed;
            // The entry's turns, sizes, following and remainders.
            work += 24.0;
        }
    }
    (!(nodes <= tone), work)
}

/// The quintic through three nodes at s = −1, 0, 1 (the Hermite interpolant of values and rates): the weights of the
/// values and of the rates d/ds at −1, 0, 1 -- with ℓ_i the nodes' Lagrange polynomials, (1 − 2ℓ_i′(s_i)(s − s_i)) ℓ_i²
/// and (s − s_i) ℓ_i².
fn quintic(s: f64) -> ([f64; 3], [f64; 3]) {
    let (minus, middle, plus) = (0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0));
    let (m2, p2) = (minus * minus, plus * plus);
    ([(1.0 + 3.0 * (s + 1.0)) * m2, middle * middle, (1.0 - 3.0 * (s - 1.0)) * p2], [(s + 1.0) * m2, s * middle * middle, (s - 1.0) * p2])
}

/// The 2-norm of a vector.
fn norm(x: &[C64]) -> f64 {
    x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt()
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
/// hopped: its flag and, where followed, its 1/Ω and -- of a first-order class (`first`) -- its following of the
/// class's change on it; none of a product's, whose following is its second order's alone (`Picture::follow`). (Until
/// 2026-11-20 a product's was read off its change too, which is none of a product's own: a class kept from a step of
/// more tones held a member's change there, and the state depended on the steps before -- hidden while such steps
/// mostly took the strides.)
fn take(class: &mut Class, k: usize, t: usize, omega: C64, fast: bool, size: usize, first: bool) {
    class.fast[k] = fast;
    let mut x = [C64::ZERO; 3];
    if fast {
        class.inverses[k] = omega.recip();
        if first {
            let [c, a, b] = [0, 1, 2].map(|j| class.change[j][k]);
            x = following(class.inverses[k], c, a, b);
            class.followed |= c != C64::ZERO || a != C64::ZERO || b != C64::ZERO;
        }
    }
    for (j, x) in x.into_iter().enumerate() {
        class.follow[j * size + t] = x;
    }
}

/// A class's entry k (its transpose's place t) between two exact blocks (`Picture::blocks`): none of the class's, as
/// every matrix the step is given is zero there -- taken as followed (`fast`) with nothing to follow, its inverse and its
/// following none, so that no route reads its frequency (the hops' series, their radius, the turns) and nothing of a
/// step before stays on it.
fn apart(class: &mut Class, k: usize, t: usize, size: usize) {
    class.fast[k] = true;
    class.inverses[k] = C64::ZERO;
    for j in 0..3 {
        class.follow[j * size + t] = C64::ZERO;
    }
}

/// The entries of the classes in `range` -- the first order's classes (`first`) or the products' -- that are hopped at
/// all, each with its partner's -- the entry (a, b) at
/// Ω = λ_a − λ_b + ν and its partner's (b, a) at −Ω (module documentation, # Tones) --, taken (`take`) by the split at
/// their turn |Ω| |h| and the changes' `sizes` (`followed`), each the following's convergence on its own -- r, the
/// larger of its two ways' |Y| at the step's ends, Y its following of the class's `envelope` on the entry (C, A, B:
/// a first-order class's change, a product's second order through the nodes) --; the largest |Ω| hopped into
/// `spread`, and the largest growth e^{|Im Ω| 2|h|} of a hopped entry in the interaction picture into `grown`
/// (`followed`), each route where its series converges: the following at r < 1, the hops where the entry's own
/// envelope's hop exponent is within π (its two ways' geometric mean, `hop_size` of its sizes over a half step, with
/// half the hop's growth; a state's own entry, which commutes with itself, has no radius). False where an entry is taken by no route of the decomposition
/// (the rest is flagged as it would be, and not used).
#[allow(clippy::too_many_arguments)]
fn flag(classes: &mut [Class], range: std::ops::Range<usize>, first: bool, lambda: &[C64], part: &[usize], sizes: [f64; 3], envelope: &dyn Fn(&Class, usize) -> [C64; 3], coupled: f64, h: f64, spread: &mut f64, grown: &mut f64) -> bool {
    let size = |class: &Class, k: usize, omega: C64| {
        let [c, a, b] = envelope(class, k);
        let y = following(omega.recip(), c, a, b);
        [-h, h].into_iter().map(|u| (y[0] + y[1] * u + y[2] * (u * u)).abs()).fold(0.0, linalg::larger)
    };
    // The entry's own hop exponent over a hop, of its envelope's sizes over a half step (`hop_size`): its hops' radius.
    let own = |class: &Class, k: usize| {
        let [c, a, b] = envelope(class, k);
        hop_size([a.abs() * (h * h), b.abs() * (h * h * h).abs(), c.abs() * h.abs()])
    };
    let mut routed = true;
    let n = lambda.len();
    for c in range {
        let (beat, partner) = (classes[c].beat, classes[c].partner);
        if partner < c || !classes[c].hops {
            continue;
        }
        for row in 0..n {
            // A pair of two blocks holds nothing of any class (every matrix the step is given is block-diagonal): it is
            // followed with nothing to follow (`apart`), so that no hop, no hop's series and no bound reads its frequency.
            for column in (if partner == c { row } else { 0 }..n).filter(|&column| part[column] != part[row]) {
                let (one, other) = both(classes, c, partner);
                apart(one, row * n + column, column * n + row, n * n);
                apart(other.unwrap_or(one), column * n + row, row * n + column, n * n);
            }
            for column in (if partner == c { row } else { 0 }..n).filter(|&column| part[column] == part[row]) {
                let between = lambda[row] - lambda[column];
                let omega = if c == 0 { between } else { C64::new(between.re + beat, between.im) };
                let turn = omega.abs();
                let (k, t) = (row * n + column, column * n + row);
                let r = if turn != 0.0 { linalg::larger(size(&classes[c], k, omega), size(&classes[partner], t, -omega)) } else { 0.0 };
                let decay = omega.im.abs() * h.abs();
                // The pair's hop exponent at its balancing similarity, its two ways' geometric mean with half the hop's
                // growth; a state's own entry commutes with itself: its hops' series is its first term.
                let hop = if c == 0 && row == column { 0.0 } else { (own(&classes[c], k) * own(&classes[partner], t)).sqrt() * (0.5 * decay).exp() };
                let (split, _) = followed(turn * h.abs(), decay, sizes, r, coupled, hop);
                routed &= split.is_some();
                let fast = split.unwrap_or(false);
                if !fast {
                    *spread = linalg::larger(*spread, turn);
                    *grown = linalg::larger(*grown, (2.0 * decay).exp());
                }
                let (one, other) = both(classes, c, partner);
                take(one, row * n + column, column * n + row, omega, fast, n * n, first);
                take(other.unwrap_or(one), column * n + row, row * n + column, -omega, fast, n * n, first);
            }
        }
    }
    routed
}

/// The hops' radius on the whole hopped operator (module documentation, # The split): Moan & Niesen's condition
/// ∫‖S Ã(t) S⁻¹‖₂ dt < π holds for one similarity S, of every hopped entry together -- a state's own entry too, which
/// commutes with itself but not with the others. Of `moduli` the hopped entries' |C|, |A| and |B| (row-major, each n²),
/// each times its largest modulus over a hop in the interaction picture, max(1, e^{−Im Ω 2h}) (`Picture::hops_radius`):
/// |Ã(t)| ≤ M⁽⁰⁾ + |t| M⁽¹⁾ + t² M⁽²⁾ entrywise at |t| ≤ |h| from the step's start or its middle, so at S = diag(d)
/// ∫‖SÃS⁻¹‖₂ ≤ Σ_k w_k ‖S M⁽ᵏ⁾ S⁻¹‖₂, w = (|h|, h²/2, |h|³/3) the moments' weights over a hop (the 2-norm is monotone
/// on nonnegative matrices). Each 2-norm within Collatz–Wielandt's bound √(maxᵢ (MᵀMv)ᵢ/vᵢ) at any positive v, its
/// power iterates from v = 1 while the bound falls by more than its rounding (it holds at every iterate). d by Osborne's
/// balancing of the weighted sum (each d_a equalizing its off-diagonal row and column, LAPACK's gebal), sweep by sweep
/// while the whole bound falls by more than its rounding: every d's bound holds, the least found is taken. Returns it
/// and its count (n² a sweep, 2n² a power iterate). Its vectors on the stack (until 2026-11-21 each iterate's on the
/// heap, some twenty allocations a step).
fn radius(moduli: &[f64], n: usize, h: f64) -> (f64, f64) {
    const STATES: usize = crate::model::STATES;
    let size = n * n;
    let weights = [h.abs(), 0.5 * h * h, h.abs().powi(3) / 3.0];
    let mut combined = [0.0_f64; STATES * STATES];
    for (k, c) in combined[..size].iter_mut().enumerate() {
        *c = (0..3).map(|j| weights[j] * moduli[j * size + k]).sum();
    }
    let mut count = 0.0;
    // ‖D M D⁻¹‖₂ of the nonnegative M, within Collatz–Wielandt's bound on (DMD⁻¹)ᵀ(DMD⁻¹)'s Perron root.
    let mut norm = |m: &[f64], d: &[f64]| -> f64 {
        let entry = |a: usize, b: usize| d[a] / d[b] * m[a * n + b];
        let (mut v, mut w, mut u) = ([1.0_f64; STATES], [0.0_f64; STATES], [0.0_f64; STATES]);
        let mut best = f64::INFINITY;
        loop {
            for (a, w) in w[..n].iter_mut().enumerate() {
                *w = (0..n).map(|b| entry(a, b) * v[b]).sum();
            }
            for (b, u) in u[..n].iter_mut().enumerate() {
                *u = (0..n).map(|a| entry(a, b) * w[a]).sum();
            }
            count += (4 * size) as f64;
            let bound = (0..n).fold(0.0_f64, |most, i| linalg::larger(most, if v[i] > 0.0 { u[i] / v[i] } else if u[i] > 0.0 { f64::INFINITY } else { 0.0 }));
            let top = u[..n].iter().fold(0.0_f64, |most, &x| linalg::larger(most, x));
            if !(bound < best - crate::units::rounding(2 * n) * best) || !(top > 0.0) {
                break linalg::smaller(best, bound).sqrt();
            }
            best = bound;
            for (v, u) in v[..n].iter_mut().zip(&u[..n]) {
                *v = u / top;
            }
        }
    };
    let mut bound = |d: &[f64]| -> f64 { (0..3).map(|j| weights[j] * norm(&moduli[j * size..(j + 1) * size], d)).sum() };
    let mut d = [1.0_f64; STATES];
    let d = &mut d[..n];
    let mut best = bound(d);
    loop {
        for a in 0..n {
            let row: f64 = (0..n).filter(|&b| b != a).map(|b| combined[a * n + b] / d[b]).sum();
            let column: f64 = (0..n).filter(|&b| b != a).map(|b| combined[b * n + a] * d[b]).sum();
            if row > 0.0 && column > 0.0 {
                d[a] = (column / row).sqrt();
            }
        }
        let next = bound(d);
        if !(next < best - crate::units::rounding(6 * size) * best) {
            best = linalg::smaller(best, next);
            break;
        }
        best = next;
    }
    count += (7 * size) as f64;
    (best, count)
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
            for m in [&mut self.base, &mut self.matrix, &mut self.slope].into_iter().chain(&mut self.ends) {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            let size = n * n;
            for (v, length) in [(&mut self.lambda, n), (&mut self.vectors, 2 * n), (&mut self.coordinates, n), (&mut self.turns, n), (&mut self.product, 9 * size), (&mut self.factors, size), (&mut self.path, 6 * n)] {
                v.resize(length, C64::ZERO);
            }
            self.hopped.resize(size, false);
            self.coupled.resize(size, false);
            self.overlaps.resize(size, 0.0);
            self.reached.resize(n, false);
            self.held.resize(n, false);
            self.part.resize(n, 0);
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
    /// nodes given (`base`, `ends`): `candidates` gives the beats ν (rad/µs) of the caller's candidate tone pairs, their
    /// members asked of `envelopes` (the candidate's place and a time u from the middle; G⁺ at e^{iνu}, G⁻ at e^{−iνu}) --
    /// first at the middle, where the routes are weighed (`toned`): a candidate is a tone of the step of 2h where its
    /// change through the three nodes would leave more than its own treatments, and wherever |ν| |h| ≥ π --, then at the
    /// outer nodes, and the candidate whole, e^{iνu}G⁺ + e^{−iνu}G⁻ -- its Hermitian part and its decay's beat alike --,
    /// taken out of the generator at each node: what is left is the static group (`beats`: the tones; `weighed`: the
    /// candidates weighed and left in the static group). None where there is no candidate, and the generator as it is.
    /// Each member's envelope is taken out at each of the three nodes (u = −h, 0, h) at its own value: an envelope that
    /// changes along the path (the light's profile at the path's points) goes through the nodes as the static group does.
    /// The routes are weighed on the generator's diagonal at the middle as given, before any candidate is taken out: a
    /// candidate's route does not depend on the others' order.
    ///
    /// `held`: the step takes its generator at the middle alone (step.rs, # One node a step) -- the static group and
    /// every envelope held at their middle's values, each beat at its exact phase: Ĥ(u) = G₀(0) + Σ_m (e^{iν_m u} G_m⁺(0)
    /// + e^{−iν_m u} G_m⁻(0)), known at every time from the middle's generator and envelopes alone (`ends` are not read
    /// before, the envelopes asked at the middle alone). The routes are weighed as on three nodes; a tone's members at
    /// the outer nodes are its middle's, and the static group at the outer nodes is the middle's with each candidate left
    /// in it at its exact phase there: the parabola the propagation takes of such a candidate is its beat's through the
    /// three nodes, as on three nodes, and of everything else its constant.
    pub fn separate(&mut self, half: f64, held: bool, candidates: impl FnOnce(&mut Vec<f64>, &mut Vec<Vec<i64>>), mut envelopes: impl FnMut(usize, f64, &mut Mat, &mut Mat)) {
        let n = self.n;
        self.beating.clear();
        self.keys.clear();
        self.weighed.clear();
        self.weighing = 0.0;
        candidates(&mut self.beating, &mut self.keys);
        let mut diagonal = [C64::ZERO; crate::model::STATES];
        for (i, d) in diagonal[..n].iter_mut().enumerate() {
            *d = self.base[(i, i)];
        }
        let mut count = 0;
        let Picture { beats, base, ends, beating, keys, reciprocals, weighed, weighing, .. } = self;
        if held {
            // The static group's change at the outer nodes: each candidate left in it, its phase there less its middle's.
            for end in ends.iter_mut() {
                end.data.fill(C64::ZERO);
            }
        }
        for (candidate, &beat) in beating.iter().enumerate() {
            if beats.len() == count {
                beats.push(Beat { beat: 0.0, key: vec![], candidate: 0, members: std::array::from_fn(|_| std::array::from_fn(|_| Mat::zeros(n, n))) });
            }
            let tone = &mut beats[count];
            for m in tone.members.iter_mut().flatten() {
                (m.rows, m.cols) = (n, n);
                m.data.resize(n * n, C64::ZERO);
            }
            let [plus, minus] = &mut tone.members;
            envelopes(candidate, 0.0, &mut plus[1], &mut minus[1]);
            let (taken, work) = toned(&diagonal[..n], beat, half, [&plus[1], &minus[1]], reciprocals);
            *weighing += work;
            if !taken {
                weighed.push(candidate);
                if held {
                    for (end, u) in ends.iter_mut().zip([-half, half]) {
                        let (p, q) = (C64::cis(beat * u) - C64::ONE, C64::cis(-beat * u) - C64::ONE);
                        for ((z, g), h) in end.data.iter_mut().zip(&plus[1].data).zip(&minus[1].data) {
                            *z += p * *g + q * *h;
                        }
                    }
                }
                continue;
            }
            (tone.beat, tone.candidate) = (beat, candidate);
            tone.key.clone_from(&keys[candidate]);
            if held {
                for member in [&mut *plus, &mut *minus] {
                    let [before, middle, after] = member;
                    before.data.copy_from_slice(&middle.data);
                    after.data.copy_from_slice(&middle.data);
                }
                for ((z, g), h) in base.data.iter_mut().zip(&plus[1].data).zip(&minus[1].data) {
                    *z -= *g + *h;
                }
                count += 1;
                continue;
            }
            for (j, u) in [(0, -half), (2, half)] {
                envelopes(candidate, u, &mut plus[j], &mut minus[j]);
            }
            let [start, end] = &mut *ends;
            for (j, (node, u)) in [start, &mut *base, end].into_iter().zip([-half, 0.0, half]).enumerate() {
                let (p, q) = (C64::cis(beat * u), C64::cis(-beat * u));
                for ((z, g), h) in node.data.iter_mut().zip(&plus[j].data).zip(&minus[j].data) {
                    *z -= p * *g + q * *h;
                }
            }
            count += 1;
        }
        beats.truncate(count);
        if held {
            for end in ends.iter_mut() {
                for (z, b) in end.data.iter_mut().zip(&base.data) {
                    *z += *b;
                }
            }
        }
        self.constant = held && self.weighed.is_empty();
    }

    /// The beat of the caller's candidate `m` as `separate` was given it (rad/µs).
    pub fn candidate_beat(&self, m: usize) -> f64 {
        self.beating[m]
    }

    /// The states.
    pub fn states(&self) -> usize {
        self.n
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
        // Each first-order class's label and beat: the static group's none, a tone's members ±its key.
        let length = self.beats.first().map_or(0, |tone| tone.key.len());
        let member = |m: usize, sign: usize| 1 + 2 * m + sign;
        let label = |c: usize| -> (Vec<i64>, f64) {
            if c == 0 {
                return (vec![0; length], 0.0);
            }
            let (tone, sign) = (&self.beats[(c - 1) / 2], if (c - 1) % 2 == 0 { 1 } else { -1 });
            (tone.key.iter().map(|k| sign * k).collect(), sign as f64 * tone.beat)
        };
        // Every product of two first-order classes into the class of its labels' sum: a first-order class where the sum
        // is one's label, else a pair of its own (the sum and its negation), made once -- their count the distinct sums.
        let mut index: std::collections::HashMap<Vec<i64>, usize> = (0..order).map(|c| (label(c).0, c)).collect();
        let mut beats: Vec<f64> = vec![];
        self.combined.clear();
        self.combined.resize(order * order, 0);
        for x in 0..order {
            let (kx, bx) = label(x);
            for y in 0..order {
                let (ky, by) = label(y);
                let sum: Vec<i64> = kx.iter().zip(&ky).map(|(a, b)| a + b).collect();
                let class = match index.get(&sum) {
                    Some(&c) => c,
                    None => {
                        let c = order + 2 * beats.len();
                        index.insert(sum.iter().map(|k| -k).collect(), c + 1);
                        index.insert(sum, c);
                        beats.push(bx + by);
                        c
                    }
                };
                self.combined[x * order + y] = class;
            }
        }
        self.classes.resize_with(order + 2 * beats.len(), Class::default);
        for class in self.classes.iter_mut().filter(|class| !class.fits(n)) {
            class.fit(n);
        }
        let classes = &mut self.classes;
        (classes[0].beat, classes[0].partner) = (0.0, 0);
        for (m, tone) in self.beats.iter().enumerate() {
            paired(classes, member(m, 0), tone.beat);
        }
        for (k, &beat) in beats.iter().enumerate() {
            paired(classes, order + 2 * k, beat);
        }
    }

    /// Prepares the picture of a step of 2h from `base` and, unless the generator is `still` along the path (then they
    /// are not read, and there are no tones), `ends` and the tones (`beats`): the changes, Ĥ₀'s decomposition, the
    /// changes in its eigenbasis and their following (`follow`). `start` is the state the step is launched with (none:
    /// any), whose reached states (`reach`) the forms' bandwidth is of. `work` begins anew with what this counted.
    /// Refused where the decomposition does not take the step (module documentation, # Where the decomposition does not
    /// take the step): the picture is then of no use until it is prepared again.
    pub fn prepare(&mut self, half: f64, still: bool, start: Option<&[C64]>) -> Result<(), Refusal> {
        assert!(!still || self.beats.is_empty(), "exact tones on a still path");
        let n = self.n;
        (self.work, self.width, self.moving) = (0.0, f64::NAN, false);
        (self.half, self.followed, self.spread, self.factored, self.grown, self.slopes) = (half, false, 0.0, f64::NAN, 1.0, [0.0; 3]);
        let classified = crate::prof::scope(84);
        self.classify();
        drop(classified);
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
        self.connect(still);
        self.reached.fill(false);
        self.within.clear();
        self.reach(start);
        // The decomposition where it converges and a state through it rounds no more than through Ĥ₀'s exponential over
        // the step (module documentation, # Where the decomposition does not take the step).
        let decomposing = crate::prof::scope(85);
        self.bounded();
        let [rho, gamma] = self.bounds;
        let tau = 2.0 * half.abs();
        let exponential = linalg::Chebyshev::plan(rho * tau, gamma * tau).rounding(n);
        let decomposed = if self.blocks.len() > 1 {
            self.basis.decompose_blocks(&self.matrix, &self.blocks, &mut self.scratch, &mut self.part)
        } else {
            self.part.clear();
            self.part.resize(n, 0);
            self.basis.decompose(&self.matrix)
        };
        if !decomposed {
            return Err(Refusal::Decomposition);
        }
        // Counted as made, before the rounding is weighed: a step refused here did the decomposition (step.rs
        // `Scratch::refused`).
        self.work += self.basis.work;
        let decay = |pick: fn(f64, f64) -> f64, from: f64| self.basis.values.iter().map(|l| l.im).fold(from, pick);
        if !(self.basis.condition * UNIT_ROUNDOFF * (1.0 + tau * (decay(linalg::larger, f64::NEG_INFINITY) - decay(linalg::smaller, f64::INFINITY))) <= exponential) {
            return Err(Refusal::Rounding);
        }
        let e = &self.basis;
        for (l, value) in self.lambda.iter_mut().zip(&e.values) {
            *l = *value + scalar;
        }
        // The eigenvectors' overlaps, what the norm rings by (`quotient`): R†R, n³.
        linalg::adjoint_product(&e.vectors.data, &e.vectors.data, (n, n, n), &mut self.product[..n * n]);
        for (o, z) in self.overlaps.iter_mut().zip(&self.product[..n * n]) {
            *o = z.abs();
        }
        self.work += (n * n * n) as f64;
        lay(&e.vectors.data, n, &mut self.right);
        lay(&e.inverse.data, n, &mut self.left);
        drop(decomposing);
        if self.moving {
            let changing = crate::prof::scope(86);
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
            // A constant picture's slopes and curvatures are zero: laid as zero, not changed.
            let constant = self.constant;
            let mut changed = 0;
            let group = &mut self.classes[0];
            for (j, given) in self.ends.iter().enumerate() {
                if constant {
                    group.laid[(1 + j) * block..(2 + j) * block].fill(0.0);
                    group.change[1 + j].fill(C64::ZERO);
                } else {
                    basis(given, &mut group.laid[(1 + j) * block..(2 + j) * block], &mut group.change[1 + j]);
                    changed += 1;
                }
            }
            for (m, tone) in self.beats.iter().enumerate() {
                for (sign, member) in tone.members.iter().enumerate() {
                    let class = &mut self.classes[1 + 2 * m + sign];
                    for (j, given) in member.iter().enumerate() {
                        if constant && j > 0 {
                            class.laid[j * block..(j + 1) * block].fill(0.0);
                            class.change[j].fill(C64::ZERO);
                        } else {
                            basis(given, &mut class.laid[j * block..(j + 1) * block], &mut class.change[j]);
                            changed += 1;
                        }
                    }
                }
            }
            self.work += (2 * changed * n * n * n) as f64;
            drop(changing);
            let following = crate::prof::scope(87);
            self.follow()?;
            drop(following);
            let _hopping = crate::prof::scope(88);
            // The hops' radius on the whole hopped operator (`hops_radius`), of the hop's own envelope as the following
            // leaves it: every entry's own within π is no convergence of the whole series.
            let reach = self.hops_radius();
            if !(reach < std::f64::consts::PI) {
                return Err(Refusal::Radius);
            }
        }
        self.spanned();
        if self.moving {
            self.slopes = self.hopping();
        }
        Ok(())
    }

    /// The generator's coupling (`coupled`): two states coupled by Ĥ₀, by the static group's change or by a tone's member
    /// at any node -- every matrix the propagation takes, as prepared.
    fn connect(&mut self, still: bool) {
        let n = self.n;
        let Picture { coupled, base, ends, beats, .. } = self;
        coupled.fill(false);
        let changes = ends.iter().filter(|_| !still);
        for m in std::iter::once(&*base).chain(changes).chain(beats.iter().flat_map(|tone| tone.members.iter().flatten())) {
            for (k, z) in m.data.iter().enumerate() {
                if *z != C64::ZERO && k / n != k % n {
                    coupled[k] = true;
                    coupled[(k % n) * n + k / n] = true;
                }
            }
        }
    }

    /// Adds to the reached states (`reached`, `within`) those a launched state x holds -- every state where there is no
    /// x -- and every state the generator couples them to (`coupled`), as far as it couples: the states the propagated
    /// state can hold until another launch. Whether any was added.
    fn reach(&mut self, x: Option<&[C64]>) -> bool {
        let (n, before) = (self.n, self.within.len());
        for a in 0..n {
            if !self.reached[a] && x.is_none_or(|x| x[a] != C64::ZERO) {
                self.reached[a] = true;
                self.within.push(a);
            }
        }
        let mut next = before;
        while next < self.within.len() {
            let a = self.within[next];
            next += 1;
            for b in 0..n {
                if self.coupled[a * n + b] && !self.reached[b] {
                    self.reached[b] = true;
                    self.within.push(b);
                }
            }
        }
        self.within.sort_unstable();
        self.within.len() > before
    }

    /// Ĥ₀'s bounds less its scalar part (`bounds`, `linalg::bounds`): the exponential's rounding over the step the
    /// decomposition's is weighed against (`prepare`) -- both of the whole Ĥ₀, as the decomposition is. (Until 2026-11-20
    /// they were of the reached states' block alone, the strides' exponents': a state launched on a block that holds no
    /// decaying state, its exponential exact, weighed the eigenvectors' rounding over every decay rate's spread against
    /// none, and no step of any length went through the decomposition -- the strides took it.)
    fn bounded(&mut self) {
        let linalg::Bounds { rho, gamma, .. } = linalg::bounds(&self.matrix, &mut self.vectors);
        self.bounds = [rho, gamma];
    }

    /// The bandwidth of the forms ⟨x|·|x⟩ on a state the step propagates (module documentation, # What follows the
    /// generator's own frequencies), on the reached states alone -- the others are empty, and no form holds their
    /// frequencies: through the decomposition max|λ_a* − λ_b| over the eigenvectors that are theirs (`held`: the larger
    /// part of an eigenvector on them; one of an eigenvalue two blocks share may mix them, at the same frequencies
    /// either way), and where the generator moves each tone's combined frequencies |λ_a* − λ_b ± ν| (its forms are
    /// summed at its beat, step.rs `Sums`) with the followed classes' beats' spread (a followed class moves the state's
    /// frequencies by its beat).
    fn spanned(&mut self) {
        let n = self.n;
        let all = self.within.len() == n;
        for a in 0..n {
            let (on, off) = (0..n).filter(|_| !all).fold((0.0, 0.0), |(on, off), i| {
                let p = self.basis.vectors[(i, a)].norm_sqr();
                if self.reached[i] { (on + p, off) } else { (on, off + p) }
            });
            self.held[a] = on >= off;
        }
        let (lambda, held, part) = (&self.lambda, &self.held, &self.part);
        // |λ_a* − λ_b + ν| is the pair's whichever way round; a pair of two blocks is no form's (`blocks`).
        let pairs = move |beat: f64| {
            (0..n).filter(move |&a| held[a]).flat_map(move |a| (a..n).filter(move |&b| held[b] && part[b] == part[a]).map(move |b| (lambda[a].conj() - lambda[b] + C64::real(beat)).abs()))
        };
        let mut bandwidth = pairs(0.0).fold(0.0, linalg::larger);
        if self.moving {
            let (low, high) = self.classes.iter().filter(|class| class.followed).fold((0.0_f64, 0.0_f64), |(low, high), class| (linalg::smaller(low, class.beat), linalg::larger(high, class.beat)));
            bandwidth = self.beats.iter().flat_map(|tone| [tone.beat, -tone.beat]).flat_map(pairs).fold(bandwidth, linalg::larger) + high - low;
        }
        self.bandwidth = bandwidth;
    }

    /// The Frobenius norms of D₂'s constants, slopes and curvatures as the hops take it, summed over the classes hopped
    /// (the static group's constant is none without tones): what bounds a hop's exponent (`moved`).
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
    /// Refused where the split takes an entry by no route (`flag`, `Refusal::Radius`).
    fn follow(&mut self) -> Result<(), Refusal> {
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
        if !flag(&mut classes, 0..order, true, &self.lambda, &self.part, sizes, &|class, k| [0, 1, 2].map(|j| class.change[j][k]), 0.0, h, &mut spread, &mut self.grown) {
            self.finish(classes, spread);
            return Err(Refusal::Radius);
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
            self.finish(classes, spread);
            return Ok(());
        }
        let (first, second) = (0.5 / h, 0.5 / (h * h));
        // The nodes −h, h and, where tones put a constant into D, 0. Where every envelope is constant the products are
        // the same at every node: made at the middle alone and laid at the others.
        let (count, times) = (if tones > 0 { 3 } else { 2 }, [-h, h, 0.0]);
        let constant = self.constant;
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
            if constant && j < 2 {
                // The static group's change none at the outer nodes too: its D and −D_s zero there, for D₂ below.
                changes[j].fill(0.0);
                slows[j * size..(j + 1) * size].fill(C64::ZERO);
                continue;
            }
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
            // The products D_t Y_t′ − Y_t′ D_t^s, each into the class of its beats' sum (of the static group's change
            // none at the middle, and none at all where every envelope is constant).
            for t in (0..order).filter(|&t| t > 0 || (j < 2 && !constant)) {
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
        if constant {
            for class in classes.iter_mut() {
                let (outer, middle) = class.second.split_at_mut(2 * block);
                outer[..block].copy_from_slice(&middle[..block]);
                outer[block..].copy_from_slice(&middle[..block]);
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
        if !flag(&mut classes, order..all, false, &self.lambda, &self.part, sizes, &product, sizes.iter().sum(), h, &mut spread, &mut self.grown) {
            self.finish(classes, spread);
            return Err(Refusal::Radius);
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
        Ok(())
    }

    /// The following's classes into their place, the largest |Ω| hopped and the entries any class hops.
    fn finish(&mut self, classes: Vec<Class>, spread: f64) {
        self.spread = spread;
        for (k, hopped) in self.hopped.iter_mut().enumerate() {
            *hopped = classes.iter().any(|class| class.hops && !class.fast[k]);
        }
        self.classes = classes;
    }

    /// The hops' radius on the whole hopped operator (`radius`): of each class that hops, each entry it does not follow
    /// at its envelope's |C|, |A| and |B| -- the hop's own (`Class::hop`), the following's second order on it in --, each
    /// at its largest modulus over a hop in the interaction picture, max(1, e^{−Im Ω 2h}) at its combined frequency
    /// Ω = λ_a − λ_b + ν (module documentation, # The split). Counts its work.
    fn hops_radius(&mut self) -> f64 {
        let (n, h) = (self.n, self.half);
        let (size, tones) = (n * n, !self.beats.is_empty());
        let mut moduli = std::mem::take(&mut self.moduli);
        moduli.clear();
        moduli.resize(3 * size, 0.0);
        for (c, class) in self.classes.iter().enumerate().filter(|(_, class)| class.hops) {
            for k in (0..size).filter(|&k| !class.fast[k]) {
                let between = self.lambda[k / n] - self.lambda[k % n];
                let omega = if c == 0 { between } else { C64::new(between.re + class.beat, between.im) };
                let growth = linalg::larger(1.0, (-omega.im * 2.0 * h).exp());
                // The static group's constant is a tone's alone (`exponent`).
                for j in usize::from(c == 0 && !tones)..3 {
                    moduli[j * size + k] += growth * class.hop[j][k].abs();
                }
            }
        }
        let (reach, work) = radius(&moduli, n, h);
        self.moduli = moduli;
        self.work += work + (4 * size * self.classes.len()) as f64;
        reach
    }

    /// The bound of a hop's exponent from the time u over the width w: ∫‖D₂‖ at the norms of its constants, slopes and
    /// curvatures, D₂(t) = C + tA + t²B at the farther end of the hop from the middle, times the hopped entries' largest
    /// growth in the interaction picture over the step (`grown`, `followed`).
    fn moved(&self, u: f64, w: f64) -> f64 {
        let far = linalg::larger(u.abs(), (u + w).abs());
        (self.slopes[0] + far * self.slopes[1] + far * far * self.slopes[2]) * w.abs() * self.grown
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
                        let parts = moments(z, terms, &mut self.reciprocals);
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

    /// The propagation's leading remainders on the state last launched (`launch`), relative to its norm, read a
    /// posteriori on its coordinates m at the launch (module documentation, # The split: what each route leaves):
    /// E = E_f + E_h + E_L, and the complex multiply-adds that made it. The picture's own count (`work`) is left as it
    /// stood: E is no part of the propagation.
    /// - E_f, the following's third order, D₂'s fast entries left out (# The state at any time of the step): made whole
    ///   at the nodes by the factors of 1 + X (`follow`), each entry of the static group's at its largest node value
    ///   times what its turning integral reaches over what is left of the step, min(h − u₀, 2/|Ω|) (the lemma's,
    ///   model/classify.rs `by_parts`), on |m|;
    /// - E_h, the hops' second Magnus term, −½∫∫[D̃₂(t₁), D̃₂(t₂)]: of each of the launch's two hops by its two halves'
    ///   exponents J_a, J_b, ½‖[J_b, J_a] m‖ -- the term of a linear envelope at three quarters of its size --, summed;
    /// - E_L, what the three nodes' parabola of D₂ misses of the change's own light shift between them on the hopped
    ///   entries of the static group: the quartic of D X, (4/15) |h|⁵ |Σ_k B_ak B_kc/Ω_kc| over the followed (k, c), on
    ///   |m_c|.
    /// Each is the leading term of what its route leaves, at the state's own components, where the split's rule
    /// (`followed`) takes the sizes of every pair together. Measured against the flow of the generator the step takes
    /// (RK4; stepcost.status.md, 304 steps of seven scenes at 0.237 to 1.897 µs whose error lay above RK4's floor): E
    /// within 0.25 and 11 times the state's error at the step's end, its median 0.77; the rule's own leading remainders
    /// 10³ to 10⁶ times it. None where nothing moves (the propagation is then exact).
    pub fn remainder(&mut self) -> (f64, f64) {
        let (n, h) = (self.n, self.half);
        if !self.moving {
            return (0.0, 0.0);
        }
        let size = n * n;
        let work = self.work;
        let mut m = [C64::ZERO; crate::model::STATES];
        m[..n].copy_from_slice(&self.path[..n]);
        let size_of = linalg::larger(norm(&m[..n]), f64::MIN_POSITIVE);
        let mut counted = 0.0;
        let (mut dropped, mut light) = (0.0, 0.0);
        if self.followed {
            // E_f: D₂ at the nodes as `follow` left it (the static group's part on its hopped entries taken out).
            let count = if self.beats.is_empty() { 2 } else { 3 };
            let wholes = &self.product[3 * size..(3 + count) * size];
            let width = (h - self.origin).abs();
            let group = &self.classes[0];
            let mut total = 0.0;
            for a in 0..n {
                let mut row = 0.0;
                for b in 0..n {
                    let k = a * n + b;
                    if group.fast[k] {
                        let omega = (self.lambda[a] - self.lambda[b]).abs();
                        let turned = linalg::smaller(width, crate::model::classify::by_parts(2.0, 0.0, C64::real(omega)));
                        row += (0..count).map(|j| wholes[j * size + k].abs()).fold(0.0, linalg::larger) * turned * m[b].abs();
                    }
                }
                total += row * row;
            }
            dropped = total.sqrt() / size_of;
            // E_L: the quartic u⁴|B|²/Ω of D X on the hopped entries, its parabola's miss (4/15)|h|⁵|Q| over the step.
            let b = &group.change[2];
            let mut total = 0.0;
            for a in 0..n {
                let mut row = 0.0;
                for c in (0..n).filter(|&c| !group.fast[a * n + c]) {
                    // Of c's own block alone: a pair of two blocks is no class's entry (`apart`), its B none.
                    let q: C64 = (0..n).filter(|&k| group.fast[k * n + c] && self.part[k] == self.part[c]).map(|k| b[a * n + k] * b[k * n + c] / (self.lambda[k] - self.lambda[c])).sum();
                    row += q.abs() * m[c].abs();
                }
                total += row * row;
            }
            light = 4.0 / 15.0 * h.abs().powi(5) * total.sqrt() / size_of;
            counted += (3 * size + size * n) as f64;
        }
        // E_h: each hop by its two halves, from the launch to the middle of what is left and on to the step's end, on m
        // where the hop meets it (the launch's and the middle stop's, `launch`).
        let mut hopped = 0.0;
        for (stop, u) in [(0, self.origin), (2, self.centre)] {
            let w = self.spacing;
            let mut x = [C64::ZERO; crate::model::STATES];
            x[..n].copy_from_slice(&self.path[stop * n..(stop + 1) * n]);
            let mut halves = std::mem::take(&mut self.product);
            for (j, at) in [u, u + 0.5 * w].into_iter().enumerate() {
                self.exponent(at, 0.5 * w);
                self.turn(at - self.origin);
                halves[j * size..(j + 1) * size].copy_from_slice(&self.matrix.data);
            }
            let apply = |matrix: &[C64], y: &[C64]| -> [C64; crate::model::STATES] {
                let mut out = [C64::ZERO; crate::model::STATES];
                for (row, z) in matrix.chunks_exact(n).zip(out.iter_mut()) {
                    *z = row.iter().zip(y).map(|(a, b)| *a * *b).sum();
                }
                out
            };
            let (first, second) = (&halves[..size], &halves[size..2 * size]);
            let (a, b) = (apply(first, &x[..n]), apply(second, &x[..n]));
            let (ba, ab) = (apply(second, &a[..n]), apply(first, &b[..n]));
            hopped += 0.5 * ba[..n].iter().zip(&ab[..n]).map(|(p, q)| (*p - *q).norm_sqr()).sum::<f64>().sqrt() / linalg::larger(norm(&x[..n]), f64::MIN_POSITIVE);
            self.product = halves;
            counted += (4 * size) as f64;
        }
        // The hops' moments are a width's (`exponent`): made anew for the next hop.
        self.width = f64::NAN;
        counted += self.work - work;
        self.work = work;
        (dropped + hopped + light, counted)
    }

    /// Launches the picture from the frame's state x at the time `from` of the step (from the middle; the step's
    /// start −h, or a later time the state changes at). Through the decomposition: m there, (1 + X(from))⁻¹R⁻¹x, and
    /// where the generator moves m's two hops -- to the middle of what is left of the step and to the step's end --
    /// with its rates at the three stops: the state is then known at every time from the launch on (`at`). The middle
    /// stop's rate is none where it is the step's middle and there are no tones (D₂ is zero there).
    pub fn launch(&mut self, from: f64, x: &[C64]) {
        let (n, h) = (self.n, self.half);
        self.origin = from;
        // A state that holds what the reached states do not (a jump's landing elsewhere): they grow with it, and the
        // forms' bandwidth with them.
        if self.reach(Some(x)) {
            self.spanned();
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
        self.quotient = self.quotient_from(from);
    }

    /// The bandwidth of the quotients a launch's piece integrates, from the time `from` to the step's end (module
    /// documentation, # What follows the generator's own frequencies): to the round-off u of the forms' scale, on the
    /// launch's coordinates m. In the eigenbasis the state is d(s) = e^{−iΛs} m (s from the launch, the following and
    /// the hops' slow change apart), its norm N = Σ_ab d_a* (R†R)_ab d_b, R's columns of unit norm: the diagonal
    /// D(s) = Σ_a |m_a|² e^{2 Im λ_a s} and the ringing O of the pairs a ≠ b. A form's quotient is
    /// (1/D) Σ_ab d_a* F_ab d_b Σ_k (−O/D)^k: a sum of terms, each a numerator pair's times k ringing pairs', of the
    /// weights p_ab = |m_a||m_b| e^{(Im λ_a + Im λ_b)s}/D(s) (the numerator's) and |(R†R)_ab| p_ab (the ringing's), at most
    /// their largest over the piece -- log p_ab is concave in s (a line less a log-sum-exp), so within its two ends'
    /// tangents --, and of the frequencies |λ_a* − λ_b| (the forms' bandwidth's pairs, their decays in) added. The terms
    /// beyond the frequency V weigh in all at most e^{−σV} N(σ)/(1 − O(σ)) for any σ > 0 with O(σ) < 1, N(σ) and O(σ) the
    /// weights' sums at e^{σν} (Chernoff's bound on the series' generating function): V the least over σ of
    /// (ln N(σ) − ln(1 − O(σ)) + ln 1/u)/σ, at the root of its derivative (Newton within a bracket: the root is one, the
    /// function's numerator convex). A tone's forms are summed at its beat and a followed class moves the frequencies by
    /// its beat: V with the largest beat and the followed beats' spread. Where the ringing weighs as much as the diagonal,
    /// O(0) ≥ 1, the series has no sum; there, and beside a series whose V is worth more nodes than it costs, the
    /// quotient as F times 1/N on the state's own continuation (`strip`), the less of the two. Its count: the
    /// pairs' weights and per Newton step the sums with their two derivatives, an exponential and six multiply-adds an
    /// item, counted as seven. The ringing's items are kept with their pairs (`ringing`: step.rs, # The quadrature by
    /// groups).
    fn quotient_from(&mut self, from: f64) -> f64 {
        self.quotient_by(from, true)
    }

    /// `quotient_from`, the strip taken where `strip` lets it (`envelope` does not: the strip's bound is of the norm's
    /// own continuation).
    fn quotient_by(&mut self, from: f64, strip: bool) -> f64 {
        let n = self.n;
        self.rung = false;
        let width = (self.half - from).abs();
        let (mut m, mut rates) = ([0.0; crate::model::STATES], [0.0; crate::model::STATES]);
        for a in 0..n {
            (m[a], rates[a]) = (self.path[a].abs(), self.lambda[a].im);
        }
        // D and D' at the piece's two ends.
        let ends = [0.0, width].map(|s| {
            let (mut d, mut slope) = (0.0, 0.0);
            for a in (0..n).filter(|&a| self.held[a]) {
                let term = m[a] * m[a] * (2.0 * rates[a] * s).exp();
                d += term;
                slope += 2.0 * rates[a] * term;
            }
            (d, slope)
        });
        if !(ends[0].0 > 0.0 && ends[1].0 > 0.0) {
            return self.bandwidth;
        }
        // The items: (weight, frequency), the numerator's and the ringing's, the ringing's with its pairs.
        let (mut numerator, mut ringing, mut pairs) = (std::mem::take(&mut self.numerator), std::mem::take(&mut self.ringing), std::mem::take(&mut self.pairs));
        numerator.clear();
        ringing.clear();
        pairs.clear();
        self.numerated.clear();
        // A weight that is not a number stays in: the quotient is not a number, and the stop it bounds is refused.
        for a in (0..n).filter(|&a| self.held[a] && !(m[a] <= 0.0)) {
            for b in (a..n).filter(|&b| self.held[b] && !(m[b] <= 0.0) && self.part[b] == self.part[a]) {
                let line = |s: f64, (d, slope): (f64, f64)| (m[a].ln() + m[b].ln() + (rates[a] + rates[b]) * s - d.ln(), rates[a] + rates[b] - slope / d);
                let ((f0, s0), (f1, s1)) = (line(0.0, ends[0]), line(width, ends[1]));
                let top = if s0 <= 0.0 {
                    f0
                } else if s1 >= 0.0 {
                    f1
                } else {
                    let t = (f1 - s1 * width - f0) / (s0 - s1);
                    f0 + s0 * t
                };
                let (weight, nu) = (linalg::smaller(top, 0.0).exp(), (self.lambda[a].conj() - self.lambda[b]).abs());
                let twice = if a == b { 1.0 } else { 2.0 };
                numerator.push((twice * weight, nu));
                self.numerated.push(([a as u8, b as u8], twice * weight));
                if a != b {
                    let o = self.overlaps[a * n + b];
                    if !(o <= 0.0) {
                        ringing.push((twice * o * weight, nu));
                        pairs.push([a as u8, b as u8]);
                    }
                }
            }
        }
        self.rung = true;
        let mut count = (numerator.len() * 12) as f64;
        let log = -UNIT_ROUNDOFF.ln();
        let sums = |items: &[(f64, f64)], sigma: f64| {
            items.iter().fold([0.0; 3], |[w, w1, w2], &(p, nu)| {
                let e = p * (sigma * nu).exp();
                [w + e, w1 + nu * e, w2 + nu * nu * e]
            })
        };
        let fastest = numerator.iter().map(|&(_, nu)| nu).fold(0.0, linalg::larger);
        let total: f64 = ringing.iter().map(|&(p, _)| p).sum();
        let shift = self.beats.iter().map(|tone| tone.beat.abs()).fold(0.0, linalg::larger) + {
            let (low, high) = self.classes.iter().filter(|class| class.followed && self.moving).fold((0.0_f64, 0.0_f64), |(low, high), class| (linalg::smaller(low, class.beat), linalg::larger(high, class.beat)));
            high - low
        };
        let v = if ringing.is_empty() {
            // No ringing: the frequency above which the numerator's terms weigh within the round-off.
            numerator.sort_by(|x, y| y.1.total_cmp(&x.1));
            let mut above = 0.0;
            numerator.iter().find(|&&(p, _)| {
                above += p;
                above > UNIT_ROUNDOFF
            }).map_or(0.0, |&(_, nu)| nu)
        } else {
            // The ringing's series where it has a sum (Chernoff's bound), and the quotient as F times 1/N on the
            // state's own continuation (`strip`) where it has none or where the series' V is worth more nodes than
            // the strip costs: the less of the two.
            // Items all at frequency zero (an envelope's, `envelope`, of groups of one component each and no decay): no
            // term beyond zero, and no σ to search (the bound falls as 1/σ for ever).
            let series = if total < 1.0 && fastest == 0.0 {
                0.0
            } else if total < 1.0 {
                // F(σ) = ln N − ln(1 − O) + ln 1/u, φ(σ) = σF′ − F, φ′ = σF″ > 0; V = F(σ*)/σ* at φ(σ*) = 0.
                let f = |sigma: f64| {
                    let ([w, w1, w2], [o, o1, o2]) = (sums(&numerator, sigma), sums(&ringing, sigma));
                    let rest = 1.0 - o;
                    if !(rest > 0.0) {
                        return None;
                    }
                    let value = w.ln() - rest.ln() + log;
                    let first = w1 / w + o1 / rest;
                    let second = w2 / w - (w1 / w).powi(2) + o2 / rest + (o1 / rest).powi(2);
                    Some((value, first, second))
                };
                // Newton within the bracket, a bisection where its step leaves it, doubling until a bracket stands; until the
                // step, once bracketed, stops shrinking (the round-off's floor). Every σ's value bounds V: the least is taken.
                let (mut lo, mut hi) = (0.0_f64, f64::INFINITY);
                // The envelopes' search (`envelope`, no strip) starts where its last one ended: a piece's envelopes keep
                // the step's spectrum less the same centres, and every σ bounds V alike.
                let mut sigma = if !strip && self.warm.is_finite() { self.warm } else { 1.0 / linalg::larger(fastest, f64::MIN_POSITIVE) };
                let (mut best, mut last, mut least) = (f64::INFINITY, f64::INFINITY, sigma);
                loop {
                    count += (7 * (numerator.len() + ringing.len())) as f64;
                    let next = match f(sigma) {
                        None => {
                            hi = sigma;
                            0.5 * (lo + hi)
                        }
                        Some((value, first, second)) => {
                            if value / sigma < best {
                                least = sigma;
                            }
                            best = linalg::smaller(best, value / sigma);
                            let phi = sigma * first - value;
                            if phi < 0.0 {
                                lo = sigma;
                            } else {
                                hi = sigma;
                            }
                            let next = sigma - phi / (sigma * second);
                            if next > lo && next < hi { next } else if hi.is_finite() { 0.5 * (lo + hi) } else { 2.0 * sigma }
                        }
                    };
                    let step = (next - sigma).abs();
                    if hi.is_finite() {
                        if !(step < last) {
                            break;
                        }
                        last = step;
                    }
                    sigma = next;
                    // A σ that left the numbers ends the search (every σ before it bounds V).
                    if !sigma.is_finite() {
                        break;
                    }
                }
                if !strip {
                    self.warm = least;
                }
                best
            } else {
                f64::INFINITY
            };
            let (strip, work) = if strip { self.strip(from, &mut numerator, fastest, log, series) } else { (f64::INFINITY, 0.0) };
            count += work;
            linalg::smaller(series, strip)
        };
        self.work += count;
        (self.numerator, self.ringing, self.pairs) = (numerator, ringing, pairs);
        linalg::larger(v + shift, 0.0)
    }

    /// The bandwidth of the quotient's envelopes over the last launch's piece where its eigencomponents are cut into
    /// groups (step.rs, # The quadrature by groups): the quotient's rule (`quotient_from`) on the eigenvalues less each
    /// one's group's centre, `shift[a]` -- the frequencies the envelopes keep, the groups' centres' phases taken out
    /// exactly --, by the ringing's series alone (the strip bounds the norm's own continuation, not the envelopes'): ∞
    /// where the series has no sum. Counted as `quotient_from` counts; the eigenvalues as they were after.
    pub(crate) fn envelope(&mut self, from: f64, shift: &[f64]) -> f64 {
        let n = self.n;
        let mut kept = [C64::ZERO; crate::model::STATES];
        kept[..n].copy_from_slice(&self.lambda[..n]);
        for (l, s) in self.lambda[..n].iter_mut().zip(shift) {
            *l -= C64::real(*s);
        }
        let v = self.quotient_by(from, false);
        self.lambda[..n].copy_from_slice(&kept[..n]);
        v
    }

    /// The last quotient's items (`quotient_from`) by a cut of the eigencomponents into groups (`group[a]` below `groups`;
    /// step.rs, # The quadrature by groups): per ordered pair of groups (g, q), g ≠ q, its pairs' ringing weights into
    /// `across` and its numerator's into `numerator` (G × G, each pair once in (g, q) and (q, g) alike, its weight the
    /// unordered pair's) -- at most Σ|N_gq|/D and the largest |y_g y_q†| over D anywhere on the piece --, and the ringing
    /// within the groups, Σ_g (N_gg − D_g)/D at the most. None where the quotient made no items (its diagonal no positive
    /// number).
    pub(crate) fn weighed_by(&self, group: &[u8], groups: usize, across: &mut [f64], numerator: &mut [f64]) -> Option<f64> {
        if !self.rung {
            return None;
        }
        across[..groups * groups].fill(0.0);
        numerator[..groups * groups].fill(0.0);
        let mut within = 0.0;
        for (&[a, b], &(p, _)) in self.pairs.iter().zip(&self.ringing) {
            let (g, q) = (usize::from(group[usize::from(a)]), usize::from(group[usize::from(b)]));
            if g == q {
                within += p;
            } else {
                across[g * groups + q] += p;
                across[q * groups + g] += p;
            }
        }
        for &([a, b], p) in &self.numerated {
            let (g, q) = (usize::from(group[usize::from(a)]), usize::from(group[usize::from(b)]));
            if g != q {
                numerator[g * groups + q] += p;
                numerator[q * groups + g] += p;
            }
        }
        Some(within)
    }

    /// The counted work of a state at a time (`at`), complex multiply-adds; and whether its matrix is assembled (where
    /// anything is followed), with the followed beating classes' count.
    pub(crate) fn state_work(&self) -> (f64, bool, usize) {
        let n = self.n;
        let followers = self.classes[1..].iter().filter(|class| class.followed).count();
        (((1 + self.assembly(followers)) * n * n + 4 * n) as f64, self.followed, self.assembly(followers))
    }

    /// Whether the generator moves along the path (D is anything at all: `prepare`).
    pub(crate) fn moving(&self) -> bool {
        self.moving
    }

    /// The eigenvalues (the scalar part in), the components the propagated state can hold (`reach`) and each one's exact
    /// block (step.rs, # The quadrature by groups).
    pub(crate) fn spectrum(&self) -> (&[C64], &[bool], &[usize]) {
        let n = self.n;
        (&self.lambda[..n], &self.held[..n], &self.part[..n])
    }

    /// The time of the last launch.
    pub(crate) fn origin(&self) -> f64 {
        self.origin
    }

    /// The quotient's bandwidth as F times 1/N (`quotient_from`): ∞ where it is not taken. q = F/N, F an exponential
    /// sum of the numerator's frequencies whose coefficients sum to at most P D(s) of the forms' scale at s (P the
    /// numerator's weights, D(s) the diagonal) and that holds nothing beyond B_F, the frequency above which its items
    /// weigh within the round-off of the quotient's scale (their weights times D/N); 1/N analytic where N has no zero.
    /// So q's spectrum beyond B_F + ν is within P D(s) times 1/N's beyond ν. N's continuation is N(s + iy) =
    /// ⟨x(s − iy), x(s + iy)⟩, x(s ± iy) = R e^{±yΛ}d(s) (d(s) = e^{−iΛs}m from the launch); on the strip where it stays
    /// within N(s) of N(s), |1/N(s + iy)| ≤ 1/(N(s) − |N(s + iy) − N(s)|), and 1/N's spectrum beyond ν is within that
    /// times e^{−yν}. Hence V(y) = B_F + (ln 1/u + ln max_s P D(s)/(N(s) − |N(s + iy) − N(s)|))/y, the least over the
    /// strip's y (V is unbounded at both its ends). N and its continuation are the state's own, exactly -- products by R,
    /// the sum's cancellation in, where a bound term by term weighs every pair at its own size (after a jump the
    /// eigen-coefficients cancel to a thousandth of their squares' sum, and the ringing's series has no sum) --, at the
    /// piece's start, middle and end (a point estimate, as the weights are), the eigenvalues taken about the middle of
    /// their frequencies (a common shift cancels in both).
    ///
    /// What it costs is weighed against what it can save, in nodes: V is worth |w|/4 nodes a unit (m ≈ κ/2, κ = V|w|/2),
    /// an evaluation of V(y) costs `EVALUATION`. Beside a `series` that holds (the ringing's series has a sum) it is
    /// taken only where that V over B_F is worth more than the search's first three evaluations; the search -- from
    /// y = 1/fastest, halved to where the strip holds and on to three points about the least, then golden sections --
    /// stops where V falls by less than an evaluation's worth. Returns V and its count: the three points' products,
    /// 4n² multiply-adds each, and per evaluation its points' two products and 2n exponentials.
    fn strip(&self, from: f64, numerator: &mut [(f64, f64)], fastest: f64, log: f64, series: f64) -> (f64, f64) {
        let n = self.n;
        let width = self.half - from;
        let worth = 0.25 * width.abs();
        let pays = |gain: f64| gain * worth > EVALUATION;
        if series.is_finite() && !pays(series / 3.0) {
            return (f64::INFINITY, 0.0);
        }
        // Each block about its own centre: a state's row is of its block's components alone (R block-diagonal), so a
        // pair's exponent is its own difference whatever the centre, and no block's offset from another's grows here.
        let blocks = self.part.iter().copied().fold(0, usize::max) + 1;
        let mut ends = vec![(f64::INFINITY, f64::NEG_INFINITY); blocks];
        for (a, &b) in self.part.iter().enumerate().take(n) {
            ends[b] = (linalg::smaller(ends[b].0, self.lambda[a].re), linalg::larger(ends[b].1, self.lambda[a].re));
        }
        let centre: Vec<C64> = self.part.iter().take(n).map(|&b| C64::real(0.5 * (ends[b].0 + ends[b].1))).collect();
        let r = &self.basis.vectors;
        let (lambda, path, held) = (&self.lambda, &self.path, &self.held);
        let points = [0.0, 0.5 * width, width].map(|s| {
            let mut d = [C64::ZERO; crate::model::STATES];
            for a in 0..n {
                d[a] = path[a] * (C64::new(0.0, -s) * lambda[a]).exp();
            }
            let diagonal: f64 = (0..n).filter(|&a| held[a]).map(|a| d[a].norm_sqr()).sum();
            let norm: f64 = (0..n).map(|i| (0..n).fold(C64::ZERO, |p, a| p + r[(i, a)] * d[a]).norm_sqr()).sum();
            (d, diagonal, norm)
        });
        let mut count = (3 * 4 * n * n) as f64;
        if !points.iter().all(|&(_, _, norm)| norm > 0.0) {
            return (f64::INFINITY, count);
        }
        let weight: f64 = numerator.iter().map(|&(p, _)| p).sum();
        let inflation = points.iter().fold(0.0_f64, |most, &(_, diagonal, norm)| linalg::larger(most, diagonal / norm));
        // B_F: the numerator's items from the highest frequency down, until they weigh more than the round-off.
        numerator.sort_by(|x, y| y.1.total_cmp(&x.1));
        let mut above = 0.0;
        let band = numerator.iter().find(|&&(p, _)| {
            above += p * inflation;
            above > UNIT_ROUNDOFF
        }).map_or(0.0, |&(_, nu)| nu);
        if series.is_finite() && !pays((series - band) / 3.0) {
            return (f64::INFINITY, count);
        }
        if !(fastest > 0.0 && fastest.is_finite()) {
            return (band, count);
        }
        let mut evaluations = 0usize;
        let mut at = |y: f64| -> f64 {
            evaluations += 1;
            let mut most = 0.0_f64;
            for (d, diagonal, norm) in &points {
                let (mut plus, mut minus) = ([C64::ZERO; crate::model::STATES], [C64::ZERO; crate::model::STATES]);
                for a in 0..n {
                    let e = ((lambda[a] - centre[a]) * y).exp();
                    (plus[a], minus[a]) = (d[a] * e, d[a] / e);
                }
                let mut continued = C64::ZERO;
                for i in 0..n {
                    let (mut p, mut q) = (C64::ZERO, C64::ZERO);
                    for a in 0..n {
                        (p, q) = (p + r[(i, a)] * plus[a], q + r[(i, a)] * minus[a]);
                    }
                    continued += q.conj() * p;
                }
                let within = norm - (continued - C64::real(*norm)).abs();
                most = linalg::larger(most, if within > 0.0 { weight * diagonal / within } else { f64::INFINITY });
            }
            band + (log + most.ln()) / y
        };
        let falls = |a: f64, b: f64| a < b && pays(b - a);
        let mut y = 1.0 / fastest;
        let mut fy = at(y);
        while fy == f64::INFINITY && y > 0.0 {
            y *= 0.5;
            fy = at(y);
        }
        let (mut lo, mut hi) = (0.5 * y, 2.0 * y);
        let mut f = [at(lo), fy, at(hi)];
        while falls(f[0], f[1]) {
            (hi, y) = (y, lo);
            lo = 0.5 * y;
            f = [at(lo), f[0], f[1]];
        }
        while falls(f[2], f[1]) {
            (lo, y) = (y, hi);
            hi = 2.0 * y;
            f = [f[1], f[2], at(hi)];
        }
        let golden = 0.5 * (5.0_f64.sqrt() - 1.0);
        let (mut c, mut e) = (hi - golden * (hi - lo), lo + golden * (hi - lo));
        let (mut fc, mut fe) = (at(c), at(e));
        let mut best = f.into_iter().chain([fc, fe]).fold(f64::INFINITY, linalg::smaller);
        while pays((fc - fe).abs()) && hi - lo > UNIT_ROUNDOFF * hi {
            if fc < fe {
                (hi, e, fe) = (e, c, fc);
                c = hi - golden * (hi - lo);
                fc = at(c);
            } else {
                (lo, c, fc) = (c, e, fe);
                e = lo + golden * (hi - lo);
                fe = at(e);
            }
            best = linalg::smaller(best, linalg::smaller(fc, fe));
        }
        count += (evaluations * 3 * (8 * n * n + 20 * n)) as f64;
        (best, count)
    }

    /// The rounding of a state's norm relative to it (`Refusal::Growth`): a component of R(1 + X)d carries the launch's
    /// product by R⁻¹, the following's by 1 + X and R's, 3n terms, and the exponential and the quintic's few, within
    /// rounding(3n + 4) of the components' sizes, which the eigenbasis' condition bounds by the state's; the norm twice
    /// that, and its own sum of n squares.
    pub fn rounding(&self) -> f64 {
        let n = self.n;
        2.0 * crate::units::rounding(3 * n + 4) * self.basis.condition + crate::units::rounding(n)
    }

    /// The frame's state at the time u (from the middle; from the launch on) through the decomposition (module
    /// documentation): x = R (1 + X(u)) e^{−iΛ(u − u₀)} m(u), m its three stops' quintic -- at any time, the same
    /// whatever was asked before it: the quadrature's nodes, a jump's root and its state, a time read.
    pub fn at(&mut self, u: f64, x: &mut [C64]) {
        let n = self.n;
        let size = n * n;
        self.coordinates_at(u, None);
        let followers = self.matrix_at(u);
        let matrix = if self.followed { &self.spare[0] } else { &self.right };
        combine(false, &self.coordinates, n, matrix, &mut self.lanes, 1, 0);
        unlay(&self.lanes, 1, x);
        self.work += ((1 + self.assembly(followers)) * size + 4 * n) as f64;
    }

    /// The state's matrix at a time made of the followed classes' (`matrix_at`), complex multiply-adds an entry:
    /// R(1 + Y₀) + uRY₁ + u²RY₂ and each class's three at its phase, or where every envelope is constant R(1 + Y₀) and
    /// each class's RY₀; none where nothing is followed.
    fn assembly(&self, followers: usize) -> usize {
        match (self.followed, self.constant) {
            (false, _) => 0,
            (true, false) => 1 + 4 * followers,
            (true, true) => followers,
        }
    }

    /// The frame's state at the time u by groups of its eigencomponents (step.rs, # The quadrature by groups): out[g] =
    /// R(1 + X(u)) e^{−i(Λ − ω)(u − u₀)} m(u) on the components a of `group[a]` = g alone, `shifted` = Λ less each
    /// component's group's centre ω -- so that x(u) = Σ_g e^{−iω_g(u − u₀)} out[g], `at`'s state --, for g below
    /// `groups`. R(1 + X(u)) is block-diagonal on the exact blocks (`blocks`), so a group of one block's components holds
    /// that block's rows alone: its product is the block's rows times its own components, Σ_a |block(a)| ≤ n² in all,
    /// with `at`'s matrix and exponentials.
    pub(crate) fn parts(&mut self, u: f64, shifted: &[C64], group: &[u8], groups: usize, out: &mut [[C64; crate::model::STATES]]) {
        let n = self.n;
        let size = n * n;
        self.coordinates_at(u, Some(shifted));
        let followers = self.matrix_at(u);
        let matrix = if self.followed { &self.spare[0] } else { &self.right };
        for out in out[..groups].iter_mut() {
            out[..n].fill(C64::ZERO);
        }
        let mut products = 0;
        for a in 0..n {
            let (d, out) = (self.coordinates[a], &mut out[usize::from(group[a])]);
            match self.blocks.get(self.part[a]) {
                Some(rows) if self.blocks.len() > 1 => {
                    for &i in rows {
                        out[i] += entry(matrix, laid(i, a, n)) * d;
                    }
                    products += rows.len();
                }
                _ => {
                    for (i, o) in out[..n].iter_mut().enumerate() {
                        *o += entry(matrix, laid(i, a, n)) * d;
                    }
                    products += n;
                }
            }
        }
        self.work += (self.assembly(followers) * size + products + 4 * n) as f64;
    }

    /// The exact blocks' states (`blocks`, none: one block) and each eigencomponent's block (`part`).
    pub(crate) fn blocked(&self) -> (&[Vec<usize>], &[usize]) {
        (&self.blocks, &self.part[..self.n])
    }

    /// The state's coordinates at the time u from the last launch (`at`): e^{−iΛ(u − u₀)} m(u), m its three stops'
    /// quintic, the eigenvalues Λ (or `shifted` in their place: `parts`), the n exponentials four at a time
    /// (`linalg::times_exp`).
    fn coordinates_at(&mut self, u: f64, shifted: Option<&[C64]>) {
        let n = self.n;
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
        linalg::times_exp(d, C64::new(0.0, -(u - self.origin)), shifted.unwrap_or(&self.lambda[..]));
    }

    /// The state's matrix at the time u (`at`): R(1 + X(u)) from its three, R(1 + Y₀) + u RY₁ + u² RY₂ of the static
    /// group's and each beating class's at its phase, into `spare[0]` where anything is followed (R itself where nothing
    /// is: `right`). The followed beating classes' count.
    fn matrix_at(&mut self, u: f64) -> usize {
        let (group, beating) = self.classes.split_first().expect("the static group");
        let followers = beating.iter().filter(|class| class.followed).count();
        if self.followed && self.constant {
            // Y₁ = Y₂ = 0: R(1 + Y₀) and each class's R Y₀ at its phase.
            let matrix = self.right.len();
            self.spare[0].copy_from_slice(&group.nodes[..matrix]);
            for class in beating.iter().filter(|class| class.followed) {
                add_phased(class.turn(u), &class.nodes[..matrix], &mut self.spare[0]);
            }
        } else if self.followed {
            let (first, rest) = group.nodes.split_at(self.right.len());
            let (second, third) = rest.split_at(self.right.len());
            for (((m, first), second), third) in self.spare[0].iter_mut().zip(first).zip(second).zip(third) {
                *m = (u * u).mul_add(*third, u.mul_add(*second, *first));
            }
            for class in beating.iter().filter(|class| class.followed) {
                add_turned(class.turn(u), &class.nodes, u, &mut self.spare[0]);
            }
        }
        followers
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// What a split meets: every pair's turn |ω_ab| |h| between two nodes, and the sizes of its change over a half
    /// step, the slope's |A_ab| h² and the curvature's |B_ab| |h|³ (the larger of the pair's two ways round). None where
    /// nothing moves.
    pub(crate) fn pairs(picture: &Picture) -> Vec<[f64; 3]> {
        let (n, h) = (picture.n, picture.half.abs());
        if !picture.moving {
            return Vec::new();
        }
        let slopes = &picture.classes[0].change[1..];
        let mut pairs = Vec::new();
        for a in 0..n {
            for b in a + 1..n {
                let size = |j: usize| linalg::larger(slopes[j][a * n + b].abs(), slopes[j][b * n + a].abs());
                pairs.push([(picture.lambda[a] - picture.lambda[b]).abs() * h, size(0) * h * h, size(1) * h * h * h]);
            }
        }
        pairs
    }

    /// The quotients' bandwidth of a piece launched at `from` again, and its counted work (`Picture::quotient_from`).
    pub(crate) fn quotient_of(picture: &mut Picture, from: f64) -> (f64, f64) {
        let work = picture.work;
        let nu = picture.quotient_from(from);
        let counted = picture.work - work;
        picture.work = work;
        (nu, counted)
    }

    /// Whether some pair's change is followed: the split's choice on a step.
    pub(crate) fn follows(picture: &Picture) -> bool {
        picture.followed
    }

    /// A picture's eigenvalues, the time of its last launch and the step's half (a measurement of its turns).
    pub(crate) fn turning(picture: &Picture) -> (Vec<C64>, f64, f64) {
        (picture.lambda.clone(), picture.origin, picture.half)
    }

    /// A picture's classes as prepared (a measurement): the states, the tones' beats, the classes (first order and in
    /// all), those followed and those hopped, and per class its beat, whether it is followed and hopped, and its
    /// followed entries' count.
    pub(crate) fn census(picture: &Picture) -> String {
        let order = 1 + 2 * picture.beats.len();
        let classes = &picture.classes;
        let followed = classes.iter().filter(|c| c.followed).count();
        let hopped = classes.iter().filter(|c| c.hops).count();
        let beats: Vec<String> = picture.beats.iter().map(|t| format!("{:.4}", t.beat)).collect();
        let each: Vec<String> = classes.iter().enumerate().map(|(i, c)| format!("{i}:{:.4}{}{}/{}", c.beat, if c.followed { "f" } else { "" }, if c.hops { "h" } else { "" }, c.fast.iter().filter(|&&f| f).count())).collect();
        format!("n {} tones {} [{}] classes {} of order {} followed {followed} hopped {hopped} | {}", picture.n, picture.beats.len(), beats.join(" "), classes.len(), order, each.join(" "))
    }

    /// The largest row sum of |X| at the step's end: the size of the following there.
    fn following(picture: &Picture) -> f64 {
        let (n, h) = (picture.n, picture.half);
        let size = n * n;
        let follow = &picture.classes[0].follow;
        (0..n).map(|a| (0..n).map(|b| (follow[b * n + a] + follow[size + b * n + a] * h + follow[2 * size + b * n + a] * (h * h)).abs()).sum::<f64>()).fold(0.0, linalg::larger)
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
        picture.prepare(half, !moving, None).expect("the decomposition takes the step");
        picture
    }

    /// The largest error of a launched picture's state at the times `at` (from the middle) against the flow of the
    /// same Ĥ₀ + uA + u²B from the step's start to each: that part of the step is a step of its own middle.
    fn errors(picture: &mut Picture, base: &Mat, a: &Mat, b: &Mat, half: f64, start: &[C64], at: &[f64], steps: usize) -> f64 {
        let n = start.len();
        let mut x = vec![C64::ZERO; n];
        let mut worst = 0.0_f64;
        for &u in at {
            picture.at(u, &mut x);
            let (middle, part) = (0.5 * (u - half), 0.5 * (u + half));
            let shifted = Mat::from_fn(n, n, |i, j| base[(i, j)] + a[(i, j)] * middle + b[(i, j)] * (middle * middle));
            let slope = Mat::from_fn(n, n, |i, j| a[(i, j)] + b[(i, j)] * (2.0 * middle));
            let exact = flow(&shifted, &slope, b, part, start, steps);
            worst = linalg::larger(worst, x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, linalg::larger));
        }
        worst
    }

    /// Towards a defective generator the decomposition refuses the step (module documentation, # Where the
    /// decomposition does not take the step): a resonant two-level atom whose coupling is a quarter of its decay rate,
    /// [[0, c], [c, −iΓ/2]] at c = Γ/4, has one eigenvector. At that point and a millionth of the coupling off it the
    /// eigenvectors would round the state beyond the exponential's rounding: the step is refused
    /// (`Refusal::Rounding`). A hundredth off it, where the eigenvectors' condition is some hundreds, the decomposition
    /// takes it, and the state at the step's end, read at sixteen times of the step, is the generator's flow: to the
    /// reference's own accuracy where nothing moves, and with slopes within the second order of what they move over a
    /// sixteenth of the step.
    #[test]
    fn a_defective_generator_s_step_is_refused_and_its_neighbour_s_is_the_flow() {
        let (gamma, half, stops) = (38.117, 0.03, 16);
        let start = [C64::ONE, C64::new(0.2, -0.1)];
        for (off, decomposed) in [(0.0, false), (1e-6, false), (1e-2, true)] {
            let c = 0.25 * gamma * (1.0 + off);
            let base = Mat::from_fn(2, 2, |i, j| if i != j { C64::real(c) } else if i == 1 { C64::new(0.0, -0.5 * gamma) } else { C64::ZERO });
            for moving in [false, true] {
                let slope = |size: f64| Mat::from_fn(2, 2, |i, j| if i != j && moving { C64::new(size, 0.3 * size * (i as f64 - j as f64)) } else { C64::ZERO });
                let (a, b) = (slope(2.0), slope(30.0));
                let mut picture = Picture::default();
                picture.fit(2);
                picture.base = base.clone();
                picture.ends = [-half, half].map(|u| Mat::from_fn(2, 2, |i, j| base[(i, j)] + a[(i, j)] * u + b[(i, j)] * (u * u)));
                let prepared = picture.prepare(half, !moving, None);
                eprintln!("{off} off the exceptional point, moving {moving}: {prepared:?}");
                if !decomposed {
                    assert_eq!(prepared, Err(Refusal::Rounding), "{off} off the exceptional point");
                    continue;
                }
                assert_eq!(prepared, Ok(()), "{off} off the exceptional point");
                let exact = flow(&base, &a, &b, half, &start, 100_000);
                let mut x = [C64::ZERO; 2];
                picture.launch(-half, &start);
                let w = 2.0 * half / stops as f64;
                for k in 1..=stops {
                    picture.at(if k == stops { half } else { -half + k as f64 * w }, &mut x);
                }
                let error = x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, linalg::larger);
                // What the slopes move over a sixteenth of the step at its end: (h‖A‖ + h²‖B‖) w, its square the second
                // order.
                let moved = (half * 2.0 + half * half * 30.0) * 2.0_f64.sqrt() * w;
                eprintln!("{off} off the exceptional point, moving {moving}: the state's error {error:e} (the slopes move {moved:e} over a sixteenth)");
                assert!(error < if moving { moved * moved } else { 1e-12 }, "{off} off the exceptional point, moving {moving}: {error:e}");
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
            assert!(picture.moving == moving);
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
            assert!(picture.followed == (name == "fast pairs"));
            let within = if picture.followed { following(&picture).powi(2) } else { 1e-4 };
            for from in [-0.55 * half, 0.0, 0.3 * half] {
                let later = [from + 0.21 * (half - from), from + 0.5 * (half - from), from + 0.83 * (half - from), half];
                picture.launch(-half, &start);
                let mut state = vec![C64::ZERO; n];
                picture.at(from, &mut state);
                let first: Vec<Vec<C64>> = later.iter().map(|&u| {
                    let mut x = vec![C64::ZERO; n];
                    picture.at(u, &mut x);
                    x
                }).collect();
                picture.launch(from, &state);
                let mut x = vec![C64::ZERO; n];
                picture.at(from, &mut x);
                let own = x.iter().zip(&state).map(|(x, e)| (*x - *e).abs()).fold(0.0, linalg::larger);
                assert!(own < 1e-13, "{name}, launched at {from}: the state there is {own:e} off the one given");
                let mut worst = 0.0_f64;
                for (&u, expected) in later.iter().zip(&first) {
                    picture.at(u, &mut x);
                    worst = linalg::larger(worst, x.iter().zip(expected).map(|(x, e)| (*x - *e).abs()).fold(0.0, linalg::larger));
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
        assert!(picture.moving && !picture.followed);
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
                worst = linalg::larger(worst, apart / bound);
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
        assert!(picture.followed && picture.classes[0].fast.iter().any(|fast| !fast));
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
                worst = linalg::larger(worst, apart / bound);
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
            assert!(!picture.followed, "a fast pair at the half {half}");
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
    /// treatment by its own slope and leaves that pair's state as it was -- the second pair 300 rad apart between two
    /// nodes, so that its own following converges at every slope read (r = slope/300, `followed`: each route where its
    /// series does; at 3 rad, as until 2026-11-20, its slopes of the scan left it no route). The slope is the least at which the rule
    /// decides so, of eight a decade from 10⁻³ to 10²; where the rule decides so at none (a pair too slow to be followed
    /// beside any other) that treatment is not read.
    fn a_pair_s_remainders(parts: [f64; 2], size: f64, x: f64) -> [Option<[f64; 2]>; 3] {
        let half = 1.0;
        let starts = [[C64::ONE, C64::ZERO], [C64::real(0.5_f64.sqrt()); 2], [C64::real(0.6), C64::new(0.0, 0.8)]];
        let own = [parts[0] * size, parts[1] * size];
        // The following's size on a pair of the turn x between two nodes (half = 1) and the change C + uA + u²B (`flag`).
        let across = |x: f64, a: C64, b: C64| {
            let y = super::following(C64::real(x).recip(), C64::ZERO, a, b);
            [-1.0, 1.0].into_iter().map(|u| (y[0] + y[1] * u + y[2] * (u * u)).abs()).fold(0.0, linalg::larger)
        };
        // A second pair's slope that takes the first to a route, and the second pair (300 rad between two nodes) to one.
        let beside = |fast: bool| (0..=40).map(|k| 10_f64.powf(-3.0 + k as f64 / 8.0)).find(|other| {
            let sizes = [own[0].hypot(*other), own[1], 0.0];
            followed(x, 0.0, sizes, across(x, C64::cis(0.7) * own[0], C64::cis(-0.4) * own[1]), 0.0, super::hop_size([own[0], own[1], 0.0])).0 == Some(fast) && followed(300.0, 0.0, sizes, across(300.0, C64::real(*other), C64::ZERO), 0.0, super::hop_size([*other, 0.0, 0.0])).0.is_some()
        });
        let mut found = [None; 3];
        for (treatment, wanted) in [Some(false), Some(true), None].into_iter().enumerate() {
            let (n, other) = match wanted.map(beside) {
                None => (2, 0.0),
                Some(Some(other)) => (4, other),
                Some(None) => continue,
            };
            // The first pair's states and the second's: 0 and −ω; 7.3 and 300/h below it.
            let base = Mat::from_fn(n, n, |i, j| if i == j { C64::real([0.0, -x / half, 7.3, 7.3 - 300.0 / half][i]) } else { C64::ZERO });
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
            assert!(wanted.is_none_or(|fast| picture.classes[0].fast[1] == fast), "x {x}, size {size}: a second pair's slope {other} for {wanted:?}");
            let mut worst = [0.0_f64; 2];
            for start in &starts {
                let start: Vec<C64> = (0..n).map(|state| if state < 2 { start[state] } else { C64::ZERO }).collect();
                picture.launch(-half, &start);
                for (place, times) in [&[half][..], &[-0.5 * half, 0.5 * half][..]].into_iter().enumerate() {
                    worst[place] = linalg::larger(worst[place], errors(&mut picture, &base, &a, &b, half, &start, times, 20_000));
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

    /// The outputs of a picture of the static group alone, bit by bit: its bandwidth and the states it gives
    /// -- launched at the step's start and read at times of the step, launched again inside the step and read on, each by
    /// `read` --, folded into one number (FNV-1a over their bits), with their count; and the states' components with each
    /// one's scale Σ_k |R(1 + X)_ik| |d_k| (`at_one_by_one`).
    fn static_outputs(picture: &mut Picture, read: fn(&mut Picture, f64, &mut [C64]) -> Vec<f64>) -> ((u64, usize), Vec<(C64, f64)>) {
        let (n, h) = (picture.n, picture.half);
        let mut bits: Vec<u64> = vec![picture.bandwidth.to_bits()];
        let start: Vec<C64> = (0..n).map(|i| C64::cis(0.9 * i as f64) * (0.5 + 0.1 * i as f64)).collect();
        let mut x = vec![C64::ZERO; n];
        let mut states = Vec::new();
        let mut state = |x: &[C64], scale: Vec<f64>, bits: &mut Vec<u64>| {
            bits.extend(x.iter().flat_map(|z| [z.re.to_bits(), z.im.to_bits()]));
            states.extend(x.iter().copied().zip(scale));
        };
        picture.launch(-h, &start);
        for u in [-0.5 * h, 0.0, 0.3 * h, 0.1 * h, h] {
            let scale = read(picture, u, &mut x);
            state(&x, scale, &mut bits);
        }
        read(picture, 0.2 * h, &mut x);
        let inside = x.clone();
        picture.launch(0.2 * h, &inside);
        for u in [0.6 * h, h] {
            let scale = read(picture, u, &mut x);
            state(&x, scale, &mut bits);
        }
        let hash = bits.iter().flat_map(|b| b.to_le_bytes()).fold(0xcbf2_9ce4_8422_2325_u64, |hash, byte| (hash ^ u64::from(byte)).wrapping_mul(0x0100_0000_01b3));
        ((hash, bits.len()), states)
    }

    /// `Picture::at` as it was before its exponentials went four at a time (until 2026-11-21, R2-PROP2): the n turns
    /// one by one by the library's `C64::exp`, the rest as it is -- the reference the engine's state is read against --;
    /// and each component's scale Σ_k |R(1 + X)_ik| |d_k|, d the turned coordinates.
    fn at_one_by_one(picture: &mut Picture, u: f64, x: &mut [C64]) -> Vec<f64> {
        let n = picture.n;
        let path = &picture.path;
        let node = |k: usize| &path[k * n..(k + 1) * n];
        let mut d = vec![C64::ZERO; n];
        if !picture.moving {
            d.copy_from_slice(node(0));
        } else {
            let (values, rates) = quintic((u - picture.centre) / picture.spacing);
            let (start, first, middle, second, end, last) = (node(0), node(1), node(2), node(3), node(4), node(5));
            for a in 0..n {
                d[a] = start[a] * values[0] + middle[a] * values[1] + end[a] * values[2] + (first[a] * rates[0] + second[a] * rates[1] + last[a] * rates[2]) * picture.spacing;
            }
        }
        for (z, l) in d.iter_mut().zip(&picture.lambda) {
            *z *= (C64::new(0.0, -(u - picture.origin)) * *l).exp();
        }
        let (group, beating) = picture.classes.split_first().expect("the static group");
        let mut matrix = picture.right.clone();
        if picture.followed {
            let (first, rest) = group.nodes.split_at(picture.right.len());
            let (second, third) = rest.split_at(picture.right.len());
            for (((m, first), second), third) in matrix.iter_mut().zip(first).zip(second).zip(third) {
                *m = (u * u).mul_add(*third, u.mul_add(*second, *first));
            }
            for class in beating.iter().filter(|class| class.followed) {
                add_turned(class.turn(u), &class.nodes, u, &mut matrix);
            }
        }
        let mut lanes = picture.lanes.clone();
        combine(false, &d, n, &matrix, &mut lanes, 1, 0);
        unlay(&lanes, 1, x);
        (0..n).map(|i| (0..n).map(|k| entry(&matrix, laid(i, k, n)).abs() * d[k].abs()).sum()).collect()
    }

    /// The hops' radius is the whole hopped operator's (# The split): a state coupled to two others by a slope of 500
    /// rad/µs², the states 1 rad/µs apart, at h = 0.1 µs -- each pair turning by 0.1 and its following's size 50, so
    /// hopped, its own radius s = 500 h²/2 = 2.5 < π --, the whole's 2-norm √2 times it, 3.54: refused (`Refusal::Radius`),
    /// where the pairs' radii alone took it until 2026-11-21. At half the step (s 0.625, the whole 0.88) taken.
    #[test]
    fn a_hop_whose_pairs_converge_but_whose_whole_does_not_is_refused() {
        let base = Mat::from_fn(3, 3, |i, j| if i == j { C64::real(i as f64) } else { C64::ZERO });
        let slope = Mat::from_fn(3, 3, |i, j| if (i == 0) != (j == 0) { C64::real(500.0) } else { C64::ZERO });
        let prepared = |half: f64| {
            let mut picture = Picture::default();
            picture.fit(3);
            picture.base = base.clone();
            picture.ends = [-half, half].map(|u| Mat::from_fn(3, 3, |i, j| base[(i, j)] + slope[(i, j)] * u));
            picture.prepare(half, false, None)
        };
        assert_eq!(prepared(0.1), Err(Refusal::Radius));
        assert_eq!(prepared(0.05), Ok(()));
    }

    /// The static group alone is propagated as it was before the exact tones (module documentation, # Tones: the
    /// tones' propagation is one implementation, of which the static group alone is the case without tones): on the
    /// inputs of this module's tests -- slow pairs hopped, fast pairs followed, both, nothing moving, each route
    /// asserted --, every output of the picture (`static_outputs`) is the one the propagation gave before the tones, to
    /// the bit (3640446e, read there by this test with the route and the price among the outputs). Since 2026-11-20 the
    /// picture has no other route and no price (module documentation, # Where the decomposition does not take the
    /// step): the outputs without them were read on 80b9782e, before the strides went, and are the same to the bit. And
    /// since the quotient's bandwidth (`quotient_from`) the picture's count holds its work too: read without the count,
    /// on the code of R1-PROP's fourth item, whose states are 80b9782e's. Since the exponentials of a state at a time go
    /// four at a time (`linalg::times_exp`, 2026-11-21) the outputs to the bit are those of the state read as before
    /// (`at_one_by_one`: the turns one by one by the library), and the engine's states are within 5u of each component's
    /// scale of them.
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
            let route = [picture.moving, picture.followed, picture.followed && hopped];
            let wanted = match name {
                "slow pairs" => [true, false, false],
                "still" => [false, false, false],
                "fast pairs" => [true, true, false],
                _ => [true, true, true],
            };
            assert_eq!(route, wanted, "{name}: the route (moving, followed, a pair hopped beside)");
            // The states as the propagation gave them before (`at_one_by_one`), and the engine's, read again on the same
            // picture: within 5u of each component's scale -- the turns' 4u, the rest the same operations.
            let (before, reference) = static_outputs(&mut picture, at_one_by_one);
            let (_, engine) = static_outputs(&mut picture, |picture, u, x| {
                let scale = at_one_by_one(picture, u, &mut x.to_vec());
                picture.at(u, x);
                scale
            });
            assert_eq!(engine.len(), reference.len());
            let worst = reference.iter().zip(&engine).map(|((r, scale), (e, _))| (*e - *r).abs() / (UNIT_ROUNDOFF * scale)).fold(0.0, linalg::larger);
            eprintln!("{name}: the engine's states within {worst:.2} u of their scale of the ones before");
            assert!(worst <= 5.0, "{name}: {worst} u");
            found.push((name, before));
        }
        // Read without the route, the price and the count (80b9782e's states: with the count 0x9468_5b73_6c10_bebe,
        // 0xb016_cf29_078f_a7be, 0x1a10_0fbd_7124_71ef and 0x0034_f20d_4180_ef5f there).
        let before = [
            ("slow pairs", (0x731a_2d41_9979_9f33, 43)),
            ("fast pairs", (0x0838_77ad_8ea4_eaa1, 57)),
            ("still", (0xce22_6de1_ce12_9a6f, 57)),
            ("fast and slow pairs", (0xbfde_6a7e_62be_5d20, 71)),
        ];
        for ((name, (hash, count)), (_, (was, _))) in found.iter().zip(before) {
            eprintln!("{name}: {count} outputs, {hash:#018x} (before {was:#018x})");
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
    /// its routes take as tones; every candidate here), prepared on the decomposition's route (a fresh picture has counted
    /// no other).
    fn with_tones(group: &[Mat; 2], tones: &[Tone], half: f64) -> Picture {
        let n = group[0].rows;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.base = toned(group, tones, 0.0);
        picture.ends = [-half, half].map(|u| toned(group, tones, u));
        picture.separate(
            half,
            false,
            |beats, keys| {
                beats.extend(tones.iter().map(|t| t.beat));
                keys.extend((0..tones.len()).map(|m| (0..tones.len()).map(|j| i64::from(j == m)).collect()));
            },
            |m, _, plus, minus| (*plus, *minus) = (tones[m].plus.clone(), tones[m].minus.clone()),
        );
        assert_eq!(picture.beats.len(), tones.len(), "every candidate a tone");
        picture.prepare(half, false, None).expect("the decomposition takes the step");
        picture
    }

    /// The same picture with every candidate a tone whatever its routes leave (`separate`'s taking out, the route not
    /// weighed): a route's own error, for the rule's to be read against.
    fn as_tones(group: &[Mat; 2], tones: &[Tone], half: f64) -> Picture {
        let n = group[0].rows;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.base = toned(group, &[], 0.0);
        picture.ends = [-half, half].map(|u| toned(group, &[], u));
        for (m, t) in tones.iter().enumerate() {
            let members = [-half, 0.0, half].map(|_| t.plus.clone());
            let partners = [-half, 0.0, half].map(|_| t.minus.clone());
            picture.beats.push(Beat { beat: t.beat, key: (0..tones.len()).map(|j| i64::from(j == m)).collect(), candidate: m, members: [members, partners] });
        }
        picture.prepare(half, false, None).expect("the decomposition takes the step");
        picture
    }

    /// The same generator as a static group alone, its three nodes' values: every beat through the nodes, as the
    /// propagation took it before the tones. None where the decomposition does not take the step so (`Refusal`).
    fn without_tones(group: &[Mat; 2], tones: &[Tone], half: f64) -> Option<Picture> {
        let n = group[0].rows;
        let mut picture = Picture::default();
        picture.fit(n);
        picture.base = toned(group, tones, 0.0);
        picture.ends = [-half, half].map(|u| toned(group, tones, u));
        picture.prepare(half, false, None).ok().map(|()| picture)
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
            picture.at(u, &mut x);
            let exact = flowing(generator, half, u, start, steps);
            linalg::larger(worst, x.iter().zip(&exact).map(|(x, e)| (*x - *e).abs()).fold(0.0, linalg::larger))
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
                sum += if class.fast[k] { 2.0 * s.powi(3) * linalg::smaller(x, 1.0) / x.powi(3) } else { x * s * s / 3.0 };
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
        let mut before = without_tones(&group, &tones, half).expect("the decomposition takes the step through the nodes");
                let within = beating_remainders(&picture, tones.len());
        let combined: Vec<f64> = (0..n * n).map(|k| (picture.lambda[k / n] - picture.lambda[k % n] + C64::real(beat)).abs()).collect();
        let (mut tone, mut nodes) = (0.0_f64, 0.0_f64);
        for start in &starts {
            tone = linalg::larger(tone, beating_errors(&mut picture, &generator, start, &times, 20_000));
            nodes = linalg::larger(nodes, beating_errors(&mut before, &generator, start, &times, 20_000));
        }
        eprintln!("the edge's combined frequencies {combined:.2?} rad/µs; Bτ = {:.1}", beat * 2.0 * half);
        eprintln!("as an exact tone the state errs by {tone:.2e} (its treatments leave {within:.2e}); through the three nodes by {nodes:.2e}");
        assert!(tone <= within, "{tone:e} against {within:e}");
        assert!(nodes > 20.0 * tone, "through the nodes {nodes:e}, as a tone {tone:e}");
    }

    /// A tone whose envelope changes along the path (the light's profile at the path's points; DESIGN §1.6, BRIEF_WIRE
    /// 2a): random Lindblad clusters -- n of 2 to 4 states, a static group of random light shifts and couplings with a
    /// slope along the path and the decay of random jumps, −(i/2)J†J --, and one Hermitian tone pair beating fast
    /// (|ν|h of 4 to 12), its envelope C + uA with a slope |A| of |C|/τ to 4|C|/τ. Over a step from three random states,
    /// against the dense flow of the whole generator: the envelope taken at each of the three nodes leaves the static
    /// group its own parabola, and the state errs within what the tones' treatments leave (`beating_remainders`, their
    /// leading terms) and the static group's own step (the same picture without the tone, against its own flow); the
    /// envelope held at its middle value -- as before -- leaves the slope's beat to the static group's nodes, and errs
    /// by far more, or leaves the decomposition no route at all (`Refusal::Radius`: the step refused).
    #[test]
    fn a_random_lindblad_s_changing_tone_is_its_flow_at_its_nodes_values() {
        let mut stream = crate::random::Stream::new(29, crate::random::Purpose::Initial, 0, 0);
        let mut s = || stream.uniform();
        let (mut worst, mut worst_held, mut cases, mut refused) = (0.0_f64, f64::INFINITY, 0, 0);
        for _ in 0..20 {
            let n = 2 + (3.0 * s()) as usize;
            let size = 1.0 + 4.0 * s();
            let half = 0.1 + 0.2 * s();
            let random = |s: &mut dyn FnMut() -> f64, scale: f64| Mat::from_fn(n, n, |_, _| C64::cis(std::f64::consts::TAU * s()) * (scale * s()));
            let hermitian = |m: &Mat| Mat::from_fn(n, n, |i, j| (m[(i, j)] + m[(j, i)].conj()) * 0.5);
            let jumps = random(&mut s, size.sqrt());
            let mut still = hermitian(&random(&mut s, size));
            still.add_scaled(C64::new(0.0, -0.5), &jumps.adjoint_product(&jumps));
            let group = [still, hermitian(&random(&mut s, size))];
            let beat = (4.0 + 8.0 * s()) / half;
            let c = random(&mut s, 0.3 * size);
            let a = random(&mut s, 0.3 * size).scaled(C64::real((1.0 + 3.0 * s()) / (2.0 * half)));
            let plus = |u: f64| Mat::from_fn(n, n, |i, j| c[(i, j)] + a[(i, j)] * u);
            let generator = |u: f64| {
                let mut g = Mat::from_fn(n, n, |i, j| group[0][(i, j)] + group[1][(i, j)] * u);
                g.add_scaled(C64::cis(beat * u), &plus(u));
                g.add_scaled(C64::cis(-beat * u), &plus(u).adjoint());
                g
            };
            let picture_of = |at_nodes: bool| {
                let mut picture = Picture::default();
                picture.fit(n);
                picture.base = generator(0.0);
                picture.ends = [-half, half].map(generator);
                picture.separate(
                    half,
                    false,
                    |beats, keys| {
                        beats.push(beat);
                        keys.push(vec![1]);
                    },
                    |_, u, p, m| {
                        let at = if at_nodes { u } else { 0.0 };
                        (*p, *m) = (plus(at), plus(at).adjoint());
                    },
                );
                assert_eq!(picture.beats.len(), 1, "the candidate a tone");
                picture.prepare(half, false, None).ok().map(|()| picture)
            };
            let (mut nodes, mut held) = (picture_of(true).expect("the decomposition takes the step at the nodes' values"), picture_of(false));
            let alone = |u: f64| Mat::from_fn(n, n, |i, j| group[0][(i, j)] + group[1][(i, j)] * u);
            let mut static_group = Picture::default();
            static_group.fit(n);
            static_group.base = alone(0.0);
            static_group.ends = [-half, half].map(alone);
            static_group.prepare(half, false, None).expect("the decomposition takes the step");
            let within = beating_remainders(&nodes, 1);
            let times = [-0.3 * half, 0.5 * half, half];
            let (mut e_nodes, mut e_held, mut e_static) = (0.0_f64, 0.0_f64, 0.0_f64);
            for _ in 0..3 {
                let start: Vec<C64> = (0..n).map(|_| C64::new(s() - 0.5, s() - 0.5)).collect();
                e_nodes = linalg::larger(e_nodes, beating_errors(&mut nodes, &generator, &start, &times, 20_000));
                if let Some(held) = held.as_mut() {
                    e_held = linalg::larger(e_held, beating_errors(held, &generator, &start, &times, 20_000));
                }
                e_static = linalg::larger(e_static, beating_errors(&mut static_group, &alone, &start, &times, 20_000));
            }
            let allowed = within + e_static;
            let read = if held.is_some() { format!("{e_held:.2e}") } else { "refused".to_string() };
            eprintln!("n {n}, νh {:.1}, |A|τ/|C| {:.1}: at the nodes' values {e_nodes:.2e} (the treatments leave {within:.2e}, the static group's own step {e_static:.2e}), held at the middle {read}", beat * half, 2.0 * half * a.max_abs() / c.max_abs());
            assert!(e_nodes <= allowed, "n {n}, νh {:.1}: at the nodes' values the state errs by {e_nodes:e}, its treatments and the static group's step leave {allowed:e}", beat * half);
            worst = linalg::larger(worst, e_nodes / allowed);
            match held {
                Some(_) => worst_held = linalg::smaller(worst_held, e_held / e_nodes),
                None => refused += 1,
            }
            cases += 1;
        }
        eprintln!("{cases} random Lindblad steps: at the nodes' values the error at most {worst:.2e} of what the treatments and the static group's step leave; held at the middle at least {worst_held:.1e} times it, or refused ({refused})");
        assert!(worst_held > 10.0, "the envelope held at its middle errs by {worst_held} times the nodes' values' at the least");
    }

    /// A decay's beats as exact tones (module documentation, # Which beats are tones; DESIGN §12.4): random Lindblad
    /// clusters -- n of 2 to 4 states, a static Hermitian part with a slope along the path, and the jumps
    /// L(u) = Σ_k e^{−io_k u} L_k at two to four offsets o_k, so that the decay −(i/2)L(u)†L(u) beats at every o_k − o_k′
    /// (each difference a candidate, −(i/2)L_k′†L_k at e^{i(o_k′ − o_k)u} and −(i/2)L_k†L_k′ at its opposite: no Hermitian
    /// part) -- over a step whose beats are near the multiples of 2π between two nodes, o_k h = k(2π + δ_k) with |δ_k| up to
    /// 0.3 (and exactly 2πk on every fourth case), the decay over a half step up to one. From three random states, against
    /// the dense flow of the whole generator:
    /// - through the three nodes with the static group, as before 2026-11-16, the beats are their alias: the state errs
    ///   by the interference's size over the step (3·10⁻² to 0.15 here); at νh = 2πk the nodes read e^{iνu} as one, the
    ///   interference whole, and a step of 2τ through its own nodes reads it so too, so that two steps through the nodes
    ///   and their companion at twice the step differ by less than the two err -- the check's blind spot (asserted where
    ///   νh = 2πk exactly);
    /// - taken as tones, whole, the state errs by the step's own error: twenty times less than through the nodes or
    ///   better, as the tones of the other tests (asserted); beside it are printed the static group's own step (the same
    ///   picture without the decay's beats, against its own flow) and Hermitian tones of the same members (G⁺ and G⁺†, the
    ///   same routes, against their own flow), which the decay's tones err about as much as -- and the treatments' leading
    ///   terms (`beating_remainders`), which the errors of either kind of tone exceed tenfold here: those terms do not hold
    ///   the following's next order (a product's c X⁽²⁾, # The split) nor the hops' cross terms of the tones with the
    ///   static group's slope.
    #[test]
    fn a_decay_s_beats_are_tones_and_their_flow_where_the_nodes_alias_them() {
        let mut stream = crate::random::Stream::new(31, crate::random::Purpose::Initial, 0, 0);
        let mut s = || stream.uniform();
        let (mut worst, mut nearest, mut blind, mut cases) = (0.0_f64, f64::INFINITY, f64::INFINITY, 0);
        for case in 0..16 {
            let n = 2 + (3.0 * s()) as usize;
            let offsets = 2 + (3.0 * s()) as usize;
            let half = 0.1 + 0.2 * s();
            let decay = (0.2 + 0.8 * s()) / half;
            let random = |s: &mut dyn FnMut() -> f64, scale: f64| Mat::from_fn(n, n, |_, _| C64::cis(std::f64::consts::TAU * s()) * (scale * s()));
            let hermitian = |m: &Mat| Mat::from_fn(n, n, |i, j| (m[(i, j)] + m[(j, i)].conj()) * 0.5);
            let exact = case % 4 == 0;
            let o: Vec<f64> = (0..offsets).map(|k| k as f64 * (std::f64::consts::TAU + if exact { 0.0 } else { 0.6 * (s() - 0.5) }) / half).collect();
            let jumps: Vec<Mat> = (0..offsets).map(|_| random(&mut s, (decay / (n * offsets) as f64).sqrt())).collect();
            let mut still = hermitian(&random(&mut s, 3.0));
            for l in &jumps {
                still.add_scaled(C64::new(0.0, -0.5), &l.adjoint_product(l));
            }
            let group = [still, hermitian(&random(&mut s, 3.0))];
            let mut tones = vec![];
            for k in 0..offsets {
                for j in k + 1..offsets {
                    let (plus, minus) = (jumps[j].adjoint_product(&jumps[k]).scaled(C64::new(0.0, -0.5)), jumps[k].adjoint_product(&jumps[j]).scaled(C64::new(0.0, -0.5)));
                    tones.push(Tone { beat: o[j] - o[k], plus, minus });
                }
            }
            let generator = |u: f64| toned(&group, &tones, u);
            let alone = |u: f64| toned(&group, &[], u);
            let mut picture = with_tones(&group, &tones, half);
            let mut before = without_tones(&group, &tones, half).expect("the decomposition takes the step through the nodes");
            let mut static_group = without_tones(&group, &[], half).expect("the decomposition takes the static group's step");
            let hermitian_tones: Vec<Tone> = tones.iter().map(|t| Tone { beat: t.beat, plus: t.plus.clone(), minus: t.plus.adjoint() }).collect();
            let unitary = |u: f64| toned(&group, &hermitian_tones, u);
            let mut hermitian_picture = with_tones(&group, &hermitian_tones, half);
            let within = beating_remainders(&picture, tones.len());
            let times = [-0.4 * half, 0.3 * half, half];
            // Through the nodes: two steps of 2h|... and the companion of 4h over the same span, from the span's start.
            let noded = |middle: f64, h: f64| {
                let shifted = |u: f64| generator(middle + u);
                let mut picture = Picture::default();
                picture.fit(n);
                picture.base = shifted(0.0);
                picture.ends = [-h, h].map(shifted);
                picture.prepare(h, false, None).expect("the decomposition takes the step");
                picture
            };
            let through = |picture: &mut Picture, x: &[C64]| {
                let mut y = vec![C64::ZERO; n];
                picture.launch(-picture.half, x);
                picture.at(picture.half, &mut y);
                y
            };
            let (mut first, mut second, mut companion) = (noded(-half, half), noded(half, half), noded(0.0, 2.0 * half));
            let (mut e_tone, mut e_nodes, mut e_static, mut e_hermitian, mut e_pair, mut apart) = (0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64);
            for _ in 0..3 {
                let start: Vec<C64> = (0..n).map(|_| C64::new(s() - 0.5, s() - 0.5)).collect();
                e_tone = linalg::larger(e_tone, beating_errors(&mut picture, &generator, &start, &times, 20_000));
                e_hermitian = linalg::larger(e_hermitian, beating_errors(&mut hermitian_picture, &unitary, &start, &times, 20_000));
                e_nodes = linalg::larger(e_nodes, beating_errors(&mut before, &generator, &start, &times, 20_000));
                e_static = linalg::larger(e_static, beating_errors(&mut static_group, &alone, &start, &times, 20_000));
                let halves = through(&mut second, &through(&mut first, &start));
                let whole = through(&mut companion, &start);
                let flow = flowing(&generator, 2.0 * half, 2.0 * half, &start, 40_000);
                e_pair = linalg::larger(e_pair, halves.iter().zip(&flow).map(|(x, e)| (*x - *e).abs()).fold(0.0, linalg::larger));
                apart = linalg::larger(apart, halves.iter().zip(&whole).map(|(x, y)| (*x - *y).abs()).fold(0.0, linalg::larger));
            }
            let beats: Vec<f64> = tones.iter().map(|t| t.beat * half / std::f64::consts::TAU).collect();
            eprintln!("n {n}, {offsets} offsets, νh/2π {beats:.3?}, Γh {:.2}: as tones {e_tone:.2e} (the static group's own step {e_static:.2e}, Hermitian tones of its members {e_hermitian:.2e}, the treatments' leading terms {within:.2e}); through the nodes {e_nodes:.2e}, two steps {e_pair:.2e} off the flow and {apart:.2e} off their companion", decay * half);
            assert!(e_nodes > 20.0 * e_tone, "n {n}, {offsets} offsets: as tones the state errs by {e_tone:e}, through the nodes by {e_nodes:e}");
            if exact {
                assert!(e_pair > apart, "at νh = 2πk two steps through the nodes err by {e_pair:e}, and differ from their companion by {apart:e}");
                blind = linalg::smaller(blind, e_pair / apart);
            }
            (worst, nearest) = (linalg::larger(worst, e_tone / e_hermitian), linalg::smaller(nearest, e_nodes / e_tone));
            cases += 1;
        }
        eprintln!("{cases} random decays: as tones at least {nearest:.0} times better than through the nodes, and at most {worst:.2} times the Hermitian tones of their members; at νh = 2πk two steps through the nodes erred at least {blind:.1} times their difference from the companion");
    }

    /// A candidate's route by what each leaves (module documentation, # Which beats are tones; DESIGN §12.5): small
    /// clusters -- two or three states, a static group of random light shifts and weak couplings with their decays, the
    /// pair (0, 1) at a turn between two nodes x = |ω|h of zero or one -- and one candidate on the pair, a Hermitian
    /// coupling C e^{iνu} with its partner, or a decay's beat, −(i/2)L₁†L₀ e^{iνu} and its opposite (two plane waves'
    /// jumps, their own decays in the static group), of the sizes s_C = |C|h of 0.01, 0.1 and 0.3 (the largest entry's),
    /// across the routes' crossing: y = νh from 0.3 to 3, and 3.5. Each step is taken both ways -- the candidate a tone
    /// (`as_tones`) and through the three nodes with the static group (`without_tones`) -- from three states against the
    /// dense flow, at the step's end and inside it. From y = π on the rule's route is the tone; both routes are taken
    /// across the scan; and the rule's routes err less in all than the rule they replaced did (νh = √5: the nodes below
    /// it, a tone above), summed over the cases. The table is printed, with each rule's count of cases where it took the
    /// route that erred more by over a half (the split's own measure, `a_pair_hopped_and_followed_against_its_flow`).
    /// Measured: the rule takes the worse route on 8 of the 96, by up to 83 times, each next to the routes' crossing in
    /// the scan; √5 on 42, by up to 5.5·10⁴ times; in all the rule's routes err a quarter of √5's. The static group's remainders (the alias β of a hop) are read
    /// within a factor two of what it errs by; a tone's following leaves 30 to 100 times less than its leading term
    /// 2s³ min(x, 1)/x³ says at a turn x′ of one to two, which moves the crossing the rule reads (DESIGN §12.5).
    #[test]
    fn a_candidate_s_route_is_the_one_that_leaves_less() {
        let mut stream = crate::random::Stream::new(37, crate::random::Purpose::Initial, 0, 0);
        let mut s = || stream.uniform();
        let (mut statics, mut cases, mut refused) = (0, 0, 0);
        let (mut sums, mut worse, mut most) = ([0.0_f64; 2], [0; 2], [0.0_f64; 2]);
        for decaying in [false, true] {
            for size in [0.01, 0.1, 0.3] {
                for x in [0.0, 1.0] {
                    for y in [0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] {
                        let (n, half) = (2 + (2.0 * s()) as usize, 0.2 + 0.2 * s());
                        let random = |s: &mut dyn FnMut() -> f64, scale: f64| Mat::from_fn(n, n, |_, _| C64::cis(std::f64::consts::TAU * s()) * (scale * s()));
                        let hermitian = |m: &Mat| Mat::from_fn(n, n, |i, j| (m[(i, j)] + m[(j, i)].conj()) * 0.5);
                        let sign = if s() < 0.5 { 1.0 } else { -1.0 };
                        let mut still = Mat::from_fn(n, n, |i, j| match (i, j) {
                            (0, 0) => C64::ZERO,
                            (1, 1) => C64::new(-sign * x / half, -0.1 * s()),
                            (i, j) if i == j => C64::new(3.0 * (s() - 0.5), -0.3 * s()),
                            _ => C64::ZERO,
                        });
                        still.add_scaled(C64::real(0.3), &hermitian(&random(&mut s, 1.0)));
                        let entry = C64::cis(std::f64::consts::TAU * s()) * (size / half);
                        let (plus, minus) = if decaying {
                            let (l0, l1) = (random(&mut s, 1.0), random(&mut s, 1.0));
                            let cross = l1.adjoint_product(&l0).scaled(C64::new(0.0, -0.5));
                            let scale = size / half / cross.max_abs();
                            still.add_scaled(C64::new(0.0, -0.5 * scale), &l0.adjoint_product(&l0));
                            still.add_scaled(C64::new(0.0, -0.5 * scale), &l1.adjoint_product(&l1));
                            (cross.scaled(C64::real(scale)), l0.adjoint_product(&l1).scaled(C64::new(0.0, -0.5 * scale)))
                        } else {
                            (Mat::from_fn(n, n, |i, j| if (i, j) == (0, 1) { entry } else { C64::ZERO }), Mat::from_fn(n, n, |i, j| if (i, j) == (1, 0) { entry.conj() } else { C64::ZERO }))
                        };
                        let group = [still, Mat::zeros(n, n)];
                        let tones = [Tone { beat: sign * y / half, plus, minus }];
                        let generator = |u: f64| toned(&group, &tones, u);
                        let mut rule = Picture::default();
                        rule.fit(n);
                        rule.base = generator(0.0);
                        rule.ends = [-half, half].map(generator);
                        let candidate = |beats: &mut Vec<f64>, keys: &mut Vec<Vec<i64>>| {
                            beats.push(tones[0].beat);
                            keys.push(vec![1]);
                        };
                        rule.separate(half, false, candidate, |_, _, plus, minus| (*plus, *minus) = (tones[0].plus.clone(), tones[0].minus.clone()));
                        let taken = rule.beats.len() == 1;
                        let (mut tone, mut nodes) = (as_tones(&group, &tones, half), without_tones(&group, &tones, half));
                        let times = [-0.3 * half, 0.4 * half, half];
                        let (mut e_tone, mut e_nodes) = (0.0_f64, 0.0_f64);
                        for _ in 0..3 {
                            let start: Vec<C64> = (0..n).map(|_| C64::new(s() - 0.5, s() - 0.5)).collect();
                            e_tone = linalg::larger(e_tone, beating_errors(&mut tone, &generator, &start, &times, 8_000));
                            if let Some(nodes) = nodes.as_mut() {
                                e_nodes = linalg::larger(e_nodes, beating_errors(nodes, &generator, &start, &times, 8_000));
                            }
                        }
                        // Where the decomposition takes no step through the nodes (`Refusal::Radius`), the rule takes
                        // the tone, and the case weighs neither rule's routes.
                        if nodes.is_none() {
                            eprintln!("{} s_C {size:4} x {x} y {y:3}: {:>6} | as a tone {e_tone:.2e}, through the nodes refused", if decaying { "decay" } else { "pair " }, if taken { "tone" } else { "nodes" });
                            assert!(taken, "y {y}: through the nodes, which the decomposition does not take");
                            refused += 1;
                            continue;
                        }
                        let chosen = if taken { e_tone } else { e_nodes };
                        let before = if y > 5.0_f64.sqrt() { e_tone } else { e_nodes };
                        eprintln!("{} s_C {size:4} x {x} y {y:3}: {:>6} | as a tone {e_tone:.2e}, through the nodes {e_nodes:.2e}", if decaying { "decay" } else { "pair " }, if taken { "tone" } else { "nodes" });
                        assert!(taken || y < std::f64::consts::PI, "y {y}: through the nodes beyond π");
                        for (k, found) in [chosen, before].into_iter().enumerate() {
                            sums[k] += found;
                            worse[k] += usize::from(found > 1.5 * linalg::smaller(e_tone, e_nodes) + 1e-13);
                            most[k] = linalg::larger(most[k], found / linalg::larger(linalg::smaller(e_tone, e_nodes), 1e-13));
                        }
                        statics += usize::from(!taken);
                        cases += 1;
                    }
                }
            }
        }
        eprintln!("{cases} candidates ({refused} more the nodes do not take), {statics} through the nodes: the rule's routes err {:.3e} in all, the worse route on {} (by up to {:.1} times); νh = √5's {:.3e}, the worse route on {} (by up to {:.1e} times)", sums[0], worse[0], most[0], sums[1], worse[1], most[1]);
        assert!(statics > 0 && statics < cases, "both routes taken");
        assert!(sums[0] < sums[1], "the rule's routes err {} in all, √5's {}", sums[0], sums[1]);
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
            picture.launch(-half, &turned(&start, theta, -half, true));
            times.iter().map(|&u| {
                let mut x = vec![C64::ZERO; n];
                picture.at(u, &mut x);
                turned(&x, theta, u, false)
            }).collect()
        };
        let reference = states(0.0);
        let mut worst = 0.0_f64;
        for theta in [-61.0, -7.3, 13.9, 42.0, 87.0] {
            let found = states(theta);
            let apart = found.iter().zip(&reference).flat_map(|(x, y)| x.iter().zip(y).map(|(x, y)| (*x - *y).abs())).fold(0.0, linalg::larger);
            eprintln!("θ {theta:6.1} rad/µs (θτ {:5.1}): the state off the first frame's by {apart:.2e}", theta * 2.0 * half);
            worst = linalg::larger(worst, apart);
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
        let mut before = without_tones(&group, &tones, half).expect("the decomposition takes the step through the nodes");
        assert!(picture.followed);
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
