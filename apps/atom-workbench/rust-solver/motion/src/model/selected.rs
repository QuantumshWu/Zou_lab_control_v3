//! The trajectory's model (RUSTPLAN §3.2 and its corrections of 2026-11-19, "模型只定一次", "一个折叠公式"): a P of the
//! selection's chain (select.rs) and its RS fold (model/rs.rs `Rs`) as the onefold integrator's
//! `step::Dynamics` / `step::Middle`. Nothing here decides or folds: P, the fold's formula and its coefficients are the
//! checkpoint's, `Rs::at` gives the model at a point, and this module puts it in the integrator's frame and terms.
//!
//! # One cluster, one frame
//! The trajectory's states are P, all of them one cluster (the ground states first: `Rs::keep` is sorted), in the frame
//! θ of floquet/frame.rs chosen at the checkpoint's centre at rest: ψ_a = e^{−iθ_a t} x_a with θ_a an exact vector over
//! the generators (the beams, the carrier) and α. At the step's outer nodes the model is taken at the path's own points,
//! so the beams' plane waves turn with the path there and the classical centre of mass gives each key its K·v; and the
//! states are in the Doppler frame that is the time frame's spatial twin, ψ_a = e^{iK_a·v s} x_a on the path s from the
//! middle, K_a = Σ_l (θ_a − θ_0)[l] k_l over the non-tweezer beams (the first state's K a phase of the whole state, which
//! nothing reads; model.rs `Cluster::waves`): an element a ← b of key c turns along the path at (q_c − K_a + K_b)·v, the
//! spatial twin of its slow remainder c − θ_a + θ_b, so that what is static in time is still along the path -- a running
//! wave's coupling of the frame's own beam --, and only the light's pattern, two beams' (k_l − k_m)·v, is left to the
//! nodes. The middle gives every matrix in the frame (`turns`: entry a ← b of the lab's times e^{−i(K_a − K_b)·v s},
//! the generator with diag(K_a·v)) and reads every density of it as the lab's there (step.rs, # The kick). Until
//! 2026-11-22 none (K = 0, as onefold): a running wave's phase along the path was the three nodes' parabola, which a step
//! of R9_PUSH's push turned by 3.3 rad over a half step -- with the end state's held forms (step.rs, # The path's
//! momentum) the window's momentum error 4.0 % at 2.84 µs; in the frame alone 0.45 %, with the mean state's forms
//! 0.018 ± 0.014 % (motion3d_status/rewrite/study/B.report.md §2.1). Its time phases are each beam's frequency at rest with its recoil,
//! f_g = ω_g − ħ|k_g|²/2m (`Scene::doppler` at κ = 0, the one Doppler shift per beam, model/rs.rs # The frequencies: the
//! frame is made at those, its lattice holding each laser's recoil exactly), so that a node at κ = 0 is `now(t)` at
//! every time. An element a ← b of a component of key c then holds e^{−iΩt}, Ω = (c − θ_a + θ_b)·(f, α)
//! (floquet/assemble.rs `to_frame`, the same vector), and x_a holds −θ_a on the diagonal. The fold keeps its slow elements alone, so every Ω is the slow remainder of that vector
//! (`Frame::split`): a beat of the lights against the frame -- a tilt's, a lattice's of two frequencies --, exact.
//!
//! # The generator by beats
//! Each node's model is grouped by its exact beats (`Beats`): Ĥ(t) = Σ_k G_k e^{−iΩ_k t}, G_k the H_eff elements of the
//! beat k with −θ on the static one's diagonal, and the decay −(i/2) Σ_ch L_ch(t)†L_ch(t) of the jump channels as they
//! are in the frame: a channel's elements at their slow beats -- its common fast index (n_φ, n_tw) is a phase of the
//! channel as a whole, its photon's frequency, no part of |L ψ|² --, each pair of its beats at their difference. So
//! i(Ĥ − Ĥ†) = Σ L†L at every time by construction, the photons' operators the decay's. The integrator's candidate tones
//! (`step::Middle::beats`) are the node's beats other than zero with their opposites, each pair one candidate at ν = |Ω|, in
//! the order of their exact vectors, its key the exact vector of its beat ν (its fast index n, and its slow remainder's
//! ratios over the candidates' common denominator: integers), so that the keys' sums are the beats' sums exactly --
//! equal sums the same beat, whichever candidates make them (4ν − 3ν is ν: one class of the propagation's second order,
//! spectral.rs, # The second order by tones). The static group and which beats are tones are the propagation's choice
//! (spectral.rs).
//!
//! # The nodes
//! A middle makes the model where the step asks for it -- its middle and its two outer nodes, at the path's momentum
//! κ_m (the fold's denominators at their Doppler frequencies, D2) --, once each, and every form of the step is read off
//! those three (`Node`). A node is the fold's numbers at the point (`Rs::evaluate`, model/rs/fold.rs `Local`), their
//! contraction by the lab's frequencies (`Out`), put in the node's structure: which beats, which elements, which
//! components of which photon class, which products of the decay -- all made once for the run from the fixed matrices'
//! structure (`Selected::layout`), so a node holds every structural beat whatever its value at the point. The numbers
//! are made in the thread's buffers and a middle's nodes go back to the thread's pool when it is dropped. A jump's photon
//! is the model at the jump's own place and time, the one node that holds the channels' photon forms (the photon's wave
//! number; `Making`). A jump component holds its elements alone (`Element`): a channel lands on the ground states,
//! its components have a few of P × P's elements each.
//!
//! # The photons
//! A class is a fast index of the channels (n_φ, n_tw): its photons land on P's ground states, its three Cartesian
//! axes the polarization's amplitudes u_a = L_(a,n)(t) ψ. A jump of the class carries its conditional mean b =
//! Σ_a Im⟨u_a|∇u_a⟩/W of the absorbed momentum as far as one absorbed photon's goes, s b with s = min(1, k_c/|b|), k_c
//! the largest wave number of the beams its channels' keys hold; the step carries the rest, Σ_c (1 − s_c) Σ_a
//! Im⟨u_a|∇u_a⟩ and 2D = Σ_c [Σ_a Re⟨∇u_a|⊗|∇u_a⟩ − s_c² W_c b_c b_cᵀ] (model.rs `Moments` and
//! `Scattering::of_moments`: the one rule). The photon's wave number is the class's rate-weighted mean of the elements' k_eg, read on the
//! channels' photon forms (`Channel::kicks`, made at a jump's node alone): Σ_a Re⟨u_a|K_aψ⟩/W. A jump's node is the
//! photon classes alone (`SelectedMiddle::photon`, `Making::PHOTONS`): the channels with their gradients and photon
//! forms, no H_eff, no generator -- the step goes on at its own nodes after the jump. Until 2026-11-23 a jump made the
//! whole node with the photon forms (and a middle's node again where the jump fell on it).
//!
//! # The rest of the model
//! The splitting's reference, the prepared wells, the loss energy and the reported trap are one potential
//! (`Scene::trap_potential`, RUSTPLAN §2.3): per ground state the tweezer's intensity times its unit fold's ground shift
//! and its own outside form's, weighed by ground populations -- the reference state's, an initial component's, or the
//! state's own (an explicit excited state's by its decay's branching). The unit fold's ground shift is the one fold's
//! (`Tweezer::reference` of the tweezer alone at unit amplitude). The displayed rows are P's populations with Q's lifted
//! through M_c (`ModelAt::lift`), each fast index of the lift apart (its terms do not interfere over time), over their
//! sum; a readout's contraction makes the lift alone.
//!
//! # The readings on the middle
//! A time read inside a step reads its rows on the step's own model there (step.rs `Middle::readout`, adapt.rs # The
//! reads), not by a fold of its own at the point: the middle's node holds the lift and its gradient along r (`Node::lift`,
//! what the contraction makes for H_eff's gradient anyway; only the middle's node, a copy), and an element q ← a of it
//! at the time s on the path is its first-order change there in the Doppler frame, M̃_qa(s) = M_qa + s (v·∇M_qa −
//! i(K_q − K_a)·v M_qa), K_q of the folded state as P's (`Selected::folded_waves`), the state framed by p_a = e^{−iK_a·v s}
//! (`turns`): the plane wave's turn of the element exact, the row's own a phase the population does not see. What is
//! left is the light's pattern and the beams' envelopes, s²/2 d²M̃/ds² -- of the second order in τ like the read state's
//! own (the step's constant middle along its stretch: (h² − u²)/2 of its change), read by the companion with the rows
//! (solve.rs, # The step). Measured (tests, `the_readings_on_the_middle_are_the_points_to_the_second_order`): on an atom
//! at 0.011 µm/µs, 0.2 µs from the middle, the rows 1.4·10⁻⁷ (PGC3D) to 1.3·10⁻⁵ (R4) from the fold's at the point,
//! a quarter of it at half the time; GMD1C's whole solve moved its populations by 3.8·10⁻⁵ against their sample errors
//! of 4.4·10⁻². The rows are one assembly either way (`Selected::lifted`: |(M ψ)_k|² of the pure state, each fast index
//! apart); a trajectory's start reads at its point. Until 2026-11-23 a read made the fold at the point (`READOUT`) and
//! took M ρ M†: 48 µs a read on PGC3D's G pinned, against a step's 310 (GMD1C's 1001 outputs 15 % of its
//! trajectories' work, R5_PUMP's 40 %, study C §2.5); on the middle 7 µs.
//!
//! # The direct entries spanned
//! A node's direct entries -- each piece's K × K entries of H_eff and of the channels, a term's factor at the point
//! times a fixed vector over an output's places -- are a sum over atoms (piece, term, coefficient) whose fixed vectors
//! span few directions: six beams of one frequency are three field components. The lab's grouping holds them spanned
//! (rs/fold.rs `Grouping::spanned`): per output an orthonormal basis of its atoms' vectors in the order of their size
//! over the reach (‖v‖ times |scale| times the coefficient's largest, every factor at most one), each atom's
//! coordinates on it, a residual left out where it is at most a tolerance of its kind's largest atom (H's outputs, the
//! channels'); the photon forms' entries one by one. What a span leaves out is a bound of the generator's change over
//! the reach -- an H output's left residuals' Σ ‖r‖b (its Frobenius norm, above its sup norm), a channel's (its
//! outputs' summed over its beats) 2‖L‖ times it and its square, `Selected::rate` -- booked with the dropped beats' Y
//! (`Judgment::dropped`, model/classify.rs # The drop tier: the solve books ½(e^{2YT} − 1)). Which tolerance is the
//! drop tier's rule (`Selected::span_directs`): of none and of 10⁻ᵏ (k = 2 … 16), the least W -- the node's counted
//! work times the account's factor at Y --, so the budget decides and a span that costs more than it saves is not
//! taken. (Until 2026-11-23 the entries one by one: PGC3D's 4134
//! direct entries, 13.5 µs of its 141 µs node; OLDNEW, motion3d_status/rewrite/impl/OLDNEW.report.md §3.1.)
//!
//! # The tables
//! A node may be read off tables of the fold's pair products in the tweezer's intensity s (rs/fold_tabled.rs: within
//! strict bounds of what it differs by from the fold's own) instead of folding at the point. The run buys them as it
//! rents (`Selected::buy_tables`): its trajectories' nodes off the fold count their work (the rent) and the least s they
//! meet; at a point no node is being made at -- before every hand-out of trajectories (solve.rs) -- where the rent has
//! reached what the tables were last priced at, their layout on [that s, 1] is made and priced exactly
//! (`Tables::layout`: the decompositions alone), and they are filled where the rent covers that price, else the price
//! is kept for the next time. Bought, they are read for the kinds of node they cost less for by count (`kind`, `KINDS`:
//! a step's node, the middle's with the lift, a jump's photon's, a whole one with the photon forms; `tables_pay`, at the
//! structure's point), and for none where none does -- then no tables at all. A point the tables do not read (its s
//! below them, a gap they cannot bound) is folded as before.
//!
//! The switch is a version of the model (step.rs `Dynamics::version`): a trajectory runs on the version it started on
//! through its life -- its checkpoints carry it, and whatever is made again of them (solve.rs `replay`, the selection's
//! walks) is made on it, to the bit --, and a hand-out's trajectories all start on one, the tables being bought between
//! hand-outs only. So a run is the same to the bit whatever its threads do. A node made outside a trajectory (a path, a
//! judgment) is the fold's own: the selection measures the model's error against its reference on its paths, and
//! the tables' own deviation is booked apart (solve.rs, # The tables' deviation) -- read off the tables there, it was
//! measured twice (2026-11-23: R4's selection asked 64 paths of P8 against 16, 54 s against 16 s).
//!
//! A step reads the lift only where it reads a time (`step::Middle::reading`): its middle's node is made with the lift
//! there alone (until 2026-11-23 every middle's node made it).
//!
//! # The tables' budget
//! A node off the tables takes the fewest of the Doppler series' terms O (of the tables' four) that hold its relative
//! errors within the run's budget: its rates' ρ^O/(1 − ρ) and its friction's ρ^{O−1}/(1 − ρ) (ρ its reading's largest
//! ratio of a member's shift to its gap, rs/fold_tabled.rs # The bounds); where none does, the fold's own. The budget is
//! weighed when the tables are bought (`Selected::budget`), as the drop tier weighs: of J = 1 … 4 terms at the run's
//! thermal speed, x(J) = [ρ^J, ρ^{J−1}]/(1 − ρ), the J of least W -- the node's counted work at J terms times the
//! account's factor with x(J) beside its errors --, its x the budget; the tables are laid and filled again to it where
//! a piece of their first layout (to the reach's tail) is beyond it (rs/fold_tabled.rs, # The pieces: the values to the
//! rates' budget, the slopes to the friction's). Each node's own errors -- the tail at its terms,
//! its interpolation's relative bound, ΔG's and the spans' parts (`Node::deviation`) -- are what its run books
//! (solve.rs, # The tables' deviation). One term leaves the friction out (x_T = 1/(1 − ρ): no factor bounds it); the
//! rule takes two on PGC3D, TET, tilt0 and tilt2 (ρ 3.1e-3 … 4.6e-3 thermal: x = 1e-5 … 2e-5 on the rates, 3e-3 …
//! 5e-3 on the friction). (Until 2026-11-23 every node's terms the tables' three, booked nowhere.)
//!
//! Measured (STEP4, `step::tests::stepcost::tabled_step_parts`, pinned, the fold's and the tables' versions on the same
//! states in turn; a step makes one node, its middle's): at three terms a step 0.92 (PGC3D; a step that reads 1.0, its
//! lifted node off the fold, dearer by count), 0.80 (TET), 0.79 (tilt2) of the fold's; at the budget's two terms 0.89
//! (a step that reads 0.97), 0.79, 0.80, the node 0.78, 0.67, 0.50, a jump's photon node 0.87, 0.79, 0.57; tilt0 none
//! bought at either (no kind cheaper by count: its fold's node 8.6e4 against 1.1e5 off the tables); the tables 6.2,
//! 3.5, 5.0 MB, made in 0.03–0.05 s.
//!
//! # The sideband parts dropped
//! K's sideband parts (rs/fold.rs `Block::sideband`: K's states by fast part, where a class has a fast element into
//! them) fold the fast elements within K; their pairs add to H_eff and to the channels what those elements make at
//! second order, on PGC3D and TET about 10⁻¹⁰ (OLDNEW §3.1), each a spectrum and its products a node. The drop tier
//! weighs them as it weighs beats (`Selected::drop_sidebands`): each part's Y -- its pairs' bounds over the reach
//! (`Out::bounds_of`), H's outputs and the channels' as a span's (`Selected::rate`) --; the parts in the order of their
//! Y, of every first few (none among them) the one of least W -- the node's counted work without them times the
//! account's factor at the dropped beats' Y and theirs --, left out of the fold (`Plan::drop_sideband`: its spectrum,
//! amplitudes and pairs, values, gradients and photon forms, not made), their Y booked with the dropped beats'.
//!
//! # The parts turned
//! K's fast parts (F = 1 and F = 2: `Plan::parts`) within an exact block are coupled by the fold's pairs through Q
//! (two-photon couplings, not elements of the fold) whose frequency in the frame's diagonal interaction picture -- the
//! diagonal's gap, about 1446 rad/µs on PGC3D, θ apart -- is far above them (V/μ ≈ 1e-4 … 6e-4). A node is turned by
//! their second order (Schrieffer–Wolff, `Selected::turned`): S between the parts solving [S, G_d] = −V (its diagonal
//! first, then `ROUNDS` of the blocks' own couplings), each block's static slot plus ½[S, V] (V's opposite beats' too,
//! each S_k = V_k/(d_a − d_b − ω_k)), its gradient's likewise, every slot's elements between the parts left out, and
//! every jump component turned to L + [S, L] (its value, gradient and photon form; the layout's places closed under
//! S). The state is then the turned one, e^{S(r)} of the whole model's. What is booked, at every node turned (`Seen`):
//! - once, on what is shown, 2‖S‖ (the frame at the start and at a reading, and 2‖S_k‖ of the opposite beats);
//! - on the motion, the frame's gradient: the frame moves with the atom, so the atom's momentum is the whole model's
//!   less ⟨−i∇S⟩, by at most ‖∇S‖ at every time (∇S = (∇V − S∇(d_a − d_b))/(d_a − d_b), the gap's own gradient in it) --
//!   the mean velocity by (ħ/m)‖∇S‖, the push by that over the window's half width, T by `estimate::frame_temperature`
//!   (solve.rs `Fixed::frame_motion`); not T's c_T·D, which takes a share D of the atoms anywhere in the trap
//!   (2026-11-23: on PGC3D 2.3 % of T against 5e-5);
//! - the remainder, of two kinds. Between the parts -- the third order 2‖S‖²‖V‖ of e^S G e^{−S}, the rounds' residual
//!   ‖V‖(2‖G_off‖/μ)^{ROUNDS+1}, each opposite beat's 2‖S_k‖(‖V‖ + ‖V_k‖), and the atom's motion through the frame,
//!   whose coupling v·(−i∇S) of size |v|‖∇S‖ the turned picture leaves out -- it turns at the gap and is booked by parts:
//!   its size over μ once (on the frame), its size times the rest's rate 2(‖G_off‖ + ‖Λ‖) over μ on the rate. Within the
//!   parts it grows: the fourth order 2‖S‖³‖V‖, the channels' second order 4‖S‖²‖Λ‖ (Σ_c ‖½[S,[S,L_c]]ρL_c† + h.c.‖₁ by
//!   Cauchy–Schwarz over the channels, Λ = Σ_c L_c†L_c the largest decay, `decay_bound`) and each opposite beat's
//!   channels 4‖S_k‖‖Λ‖ (its S_k is not turned into them) -- on the rate Y₃. Of V, S, μ (the frame's diagonal gap, θ
//!   apart) and the blocks' couplings G_off at the node. (Until 2026-11-23 the channels' term was 4‖S‖²Σ‖L‖², each
//!   channel's Frobenius norm summed, 6 times this on tilt0; and the opposite beats' sizes were read after their elements
//!   were taken out, none.) Measured against the whole model's Lindbladian at a point at rest, the jumps averaged
//!   (`the_turn_s_remainder_with_jumps`, 1 ms): the remainder does not grow -- 3e-6 to 1.3e-4 of trace distance at 1 ms
//!   against Y₃·1 ms of 1.7e-5 (PGC3D), 2.6e-4 (TET), 1.7e-2 (tilt0) --: strict, 30 to 500 times above what it bounds.
//! Dormant (2026-11-23, `turning`): a turned step takes 4–18 % longer on PGC3D, TET and tilt2 -- the turned node
//! 35–55 %, a jump's photon node made whole for its S 60–90 %, the scattered light on the places the turn adds 2.3–5.9
//! times, against 30–95 µs the integration saves --; tilt0 saves 25 % but its remainder's bound, 300 times what it
//! bounds, leaves no room; and a turned model keeps the trace to its second order alone, a mode the selection's memory
//! reads (paths.rs `memory`). What it lacks to pay: a jump's S from the H outputs between the parts and on the diagonal
//! alone (not a whole node), the places the turn adds laid in their rows (not after a component's), the trace kept (or
//! the memory telling its mode), and for tilt a bound of the remainder within reach of what it bounds.
//! The run books D = 2 max frame + ½(e^{2 max Y₃ T} − 1) on what is shown and the remainder's part alone on T, over the
//! nodes it made -- strict at every node walked, the time's integral taken as the largest rate times T.
//! Which parts are turned is the drop tier's rule (`Selected::turn_parts`), on the run made whole: in the order of
//! ‖V‖/μ at the structure's point, of none and every first few pairs of parts the cut of least W -- a step's counted
//! work a µs as the run takes it from the trap's centre at the physics prior's step (`priced_work`, solve.rs
//! `prior_step`; the count sees what the cut saves in the integration: the quadrature's nodes at the quotient's
//! bandwidth, the launch, the cut's choice, the decomposition by blocks) times `Account::factor_with` at the estimate's
//! Y₃, frame and gradient. Its pilot checks it (`Selected::turn_holds`): the same W at the largest sizes its nodes
//! noted; where the turn no longer pays the request is made unturned from its pilot again (solve.rs `attempt`,
//! `solve_with`). The reach's bounds (as the drop tier's, `Out::bounds`) cannot be used here: a pair's bound sums
//! every beam's modulus with no phase -- six beams two by two -- and the shifts' range takes the tweezer's intensity
//! over [0, 1]; on PGC3D, TET and tilt2 they put ‖V‖ at 57–97 rad/µs against 0.1–0.3 at the structure's point, and on
//! tilt0 the shifts above the gap: the factor unbounded, nothing ever turned (2026-11-23). Measured (STEP, 2026-11-23,
//! `the_split_s_walk`): over 100 µs jump-free the turned state stays 1.1–2.0e-4 apart from the whole model's (the
//! frame), the parts' elements left out (no turn) drift to 0.5–1e-3.
//!
//! # Exact blocks
//! P falls into blocks on which every matrix the integrator is handed is block-diagonal: the node's every beat of the
//! generator -- H_eff with its decay −(i/2)ΣL†L -- and of its gradient along each lab axis. They are the run's judgment's
//! (model/classify.rs, `Selected::judge`): P's connected parts under the kept beats' entries that are no rounding zeros,
//! read on the structure's node, the trap's centre at rest (the static beat with its frame's θ back on the diagonal).
//! What two keys of one lab frequency cancel at the centre alone would read as no element there; the tests read the
//! blocks at points off it as well (tests, `the_exact_blocks_hold_at_every_point`). The jumps are no part of it: a jump
//! may land a state of one block in several,
//! and the state is the direct sum of its blocks' parts -- the generator and every reading's operator (the force, the
//! hazard, the norm) block-diagonal, the jump acting on the whole state as it is. Every element between two blocks is
//! left out of the node (zero by construction), and the integrator decomposes Ĥ₀ block by block with no pair of two
//! blocks a frequency of any form (spectral.rs `Picture::blocks`). An element of more than the rounding between two such
//! states joins their blocks wherever the model holds it.
//!
//! # The structure's rounding
//! The structure -- which beats the node holds, which states its blocks join, and with the beats the tones'
//! candidates -- is the run's one judgment (model/classify.rs), made once on computed numbers, the node at the trap's
//! centre at rest at unit amplitude (`Selected::structure`). A computed entry's rounding is its own computation's bound,
//! not the result's ε·max: an entry x that is the sum of K terms of moduli summing to S is a rounding zero where
//! |x| ≤ γ_K S (Higham's γ_K = Ku/(1 − Ku), units.rs `rounding`; classify.rs `zero`), and nowhere else. S and K are the
//! computation's (`Selected::structure`, rs/fold.rs `Out::scales`, `Local::shadows`), the same sums and products on the
//! operands' moduli: a direct entry one term; a pair Σ_q |X_iq| |M_qj| over |B| + the leg sum's members, |X| the leg
//! sum's members' moduli summed and |M| the amplitudes' own scale through their resolvent (a Q block's
//! |V| Σ|w| |D| ∘ (|V⁻¹| |X|), where a decomposition that mixes nearly degenerate states would leave its rounding); the
//! decay's products Σ_g |l_g||r_g| over the landing states. A cancellation of terms larger than its result -- the
//! polarizations' axes, the emission's -- is its rounding, however far above ε times the result's largest it lies.
//! (2026-11-20: on every scene of the gate and of refs/scenes the structure read so was the one ε·max read; 3 to 34
//! single entries per scene are zeros by the sum alone, each within a block and in a beat that holds others. Until
//! 2026-11-21 an entry below ε times its matrix's largest was a zero as well, the largest of every beat's for the beats
//! and its own slot's for the blocks: one quantity of two definitions, now one.)
//!
//! # What is not here
//! No landing transient, no decision, no representation per trajectory: the integrator's plain landing L_qψ/‖·‖ is the
//! model's (D5), the trajectory holds nothing of the model.
//!
//! # The propagation's interface (FIXSTEP)
//! What a fixed-model propagation needs of the model is a `Node` at a point and momentum: the generator's components by
//! exact beat with their values (`Selected::components`), H_eff's gradient's, and the photon classes. A propagation that
//! takes the model once a step (STEPBENCH L2) asks for the middle's alone; the onefold integrator asks for three.

use crate::linalg::{larger, smaller};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, RwLock};

use crate::field::Light;
use crate::floquet::frame::{exact_sum, FrameKey, Ratio};
use crate::floquet::{Frame, Hilbert};
use crate::linalg::{inner, norm_sqr, C64, Mat, V3};
use crate::model::rs::fold::tabled::{Scratch as PairScratch, Tables as PairTables};
use crate::model::rs::fold::{Grouping, Local, Out, Plan};
use crate::model::rs::{Making, Rs};
use crate::model::classify::{Account, Judgment, Scales, Tier};
use crate::model::{Beat, Cluster, Moments, Point, Quadratic, Scattering};
use crate::request::{GROUND, STATES};
use crate::scene::{Key, Scene};
use crate::select::Selection;
use crate::step::{Emitted, Internal, Optical};
use crate::units::{hbar_over_mass, rounding, RAD_PER_US_PER_MHZ};
use crate::decide::{COMPLEX, DIVISION, ELEMENTARY, MOVED};
/// A real number's operation (a comparison), in real multiply-adds (decide.rs, # Work).
const REAL_OP: f64 = 1.0;

/// The selected model of a run (module documentation): the scene, the fold on P, the selection's report, the frame,
/// and what the potential and the photons read, made once.
pub struct Selected {
    pub scene: Scene,
    pub rs: Rs,
    pub report: Selection,
    frame: Frame,
    /// The frame's primitives' values (rad/µs): the request's frequency axes Φ_i, then each laser's recoil class δ_c; a
    /// beat's value Σ p_i (Φ, δ)_i.
    axes: Vec<f64>,
    /// θ_a of P's states (rad/µs).
    theta: Vec<f64>,
    cluster: Cluster,
    light: Light,
    hbar_over_mass: f64,
    /// The tweezer's unit fold's ground shift (rad/µs; `Scene::trap_potential`'s depth).
    depth: [f64; GROUND],
    /// The other lights' smooth shifts of the ground states (`Scene::smooth`; the reported trap's potential).
    smooth: Vec<(usize, [[f64; 2]; GROUND])>,
    /// The reference state's ground populations (V_ref's weights).
    reference: [f64; GROUND],
    /// Per state of P its decay's branching onto the ground states (a ground state's its own).
    branching: Vec<[f64; GROUND]>,
    /// γ_Q, the least decay over two of P's explicit excited states; None where P holds none.
    optical: Option<f64>,
    /// The non-tweezer beams' wave vectors (the light pattern's beats along a path).
    waves: Vec<V3>,
    /// Q's states' Doppler frame K_q = Σ_l (θ_q − θ_0)[l] k_l, as P's (`Cluster::waves`), in the fold's order of Q: a
    /// lift's element q ← a turns along the path at (K_q − K_a)·v but for its pattern (`SelectedMiddle::readout`).
    folded_waves: Vec<V3>,
    /// The light's pattern the Doppler frame leaves to the step's nodes (`Selected::pattern`, 1/µm).
    pattern: f64,
    /// The fold's grouping by the lab's frequencies (`Selected::at`), and the node's structure on it (`layout`).
    lab: Grouping,
    template: Node,
    tables: Tables,
    /// The run's tables of pair products as it buys them (# The tables).
    tabling: Tabling,
    /// The run's own number (the thread's buffers and pooled nodes are its).
    id: usize,
    /// The run's judgment of its node's beats and blocks (model/classify.rs, `judge`).
    judgment: Judgment,
    /// The exact blocks of P (module documentation, # Exact blocks): places in P, each sorted (its ground states first);
    /// and each place's block.
    blocks: Vec<Vec<usize>>,
    block_of: Vec<usize>,
    /// The parts turned (# The parts turned): the pairs of places (a, b) of one exact block in two turned ones, both
    /// ways; none where no part is turned.
    turn: Option<Vec<(usize, usize)>>,
    /// The turned nodes' sizes (`Seen`), shared with the run's accounts.
    seen: Arc<Seen>,
    /// The turn's decision as it was made (`turn_parts`), for the pilot's check (`turn_holds`); none where nothing is
    /// turned.
    turn_check: Option<TurnCheck>,
    /// Per jump component of the layout (its classes, axes and components in turn) the terms of [S, L] (`apply_cut`):
    /// (the element it adds to, the element of L it is of, S's place, whether it is the − L S term).
    turn_terms: Vec<Vec<(u32, u32, u32, bool)>>,
    beats: RwLock<Beats>,
    /// The candidate tones and their keys (module documentation, # The generator by beats): every node holds the
    /// layout's beats, so they are the run's, made once (`Selected::tones`).
    candidates: Vec<Candidate>,
    keys: Vec<Vec<i64>>,
    /// The lift's groups in the order of their keys, each with its elements' fast index (an index among the run's
    /// fast indices of the lift) and the value of its slow beat (`Selected::rows`).
    lifts: Vec<(usize, Vec<(usize, f64)>)>,
}

/// Which beats a model drops (`Selected::made`): those its judgment drops (model/classify.rs, # The drop tier), its
/// direct entries spanned where that is the cheaper (# The direct entries spanned), or (a check's) exactly the candidate
/// pairs of the listed |Ω| (rad/µs, to 10⁻⁹ of it), none of an empty list, and nothing else left out: the direct
/// entries one by one.
#[derive(Clone, Debug)]
pub(crate) enum Drops {
    Judged,
    #[cfg(test)]
    Only(Vec<f64>),
}

/// The rounds of the blocks' own couplings in S (# The parts turned): the residual ‖V‖(2‖G_off‖/μ)^{ROUNDS+1}.
const ROUNDS: usize = 3;

#[cfg(test)]
thread_local! {
    /// Whether a run turns its parts (# The parts turned; the tests': the node against the fold's components is the
    /// unturned run's).
    pub(crate) static TURNING: Cell<bool> = const { Cell::new(false) };
    /// A turning run's every pair of parts turned whatever its W (the tests' measurement of the turn itself).
    pub(crate) static FORCE_TURN: Cell<bool> = const { Cell::new(false) };
}

/// A bound of the largest decay ‖Λ‖ of a node, Λ(t) = Σ_c L_c†L_c at any time (# The parts turned): each slot's decay
/// part i(G_k − G_{−k}†) is within its entries' moduli |G_k| + |G_{−k}|, and Λ's operator norm within its largest row
/// sum, the slots' together: the static slot's Λ₀ = i(G₀ − G₀†) row by row, every other slot's |G_k| row sums twice
/// (its pair's and its own).
fn decay_bound(node: &Node) -> f64 {
    let Some((_, g)) = node.generator.iter().find(|x| x.0 == 0) else { return 0.0 };
    let n = g.rows;
    let rows = |f: &dyn Fn(usize, usize) -> f64| (0..n).map(|a| (0..n).map(|b| f(a, b)).sum::<f64>()).fold(0.0, larger);
    let mut bound = rows(&|a, b| (g[(a, b)] - g[(b, a)].conj()).abs());
    for (_, m) in node.generator.iter().filter(|x| x.0 != 0) {
        bound += 2.0 * rows(&|a, b| m[(a, b)].abs());
    }
    bound
}

/// Σ‖L‖² of a node's channels, each class's axis's components' Frobenius norms squared, summed (a bound of Σ_c ‖L_c‖²
/// over its jumps' components; the tests' comparison with `decay_bound`).
#[cfg(test)]
fn channels_size(node: &Node) -> f64 {
    node.classes.iter().flat_map(|c| c.axes.iter()).map(|axis| axis.iter().map(|j| j.elements.iter().map(|e| e.value.norm_sqr()).sum::<f64>().sqrt()).sum::<f64>().powi(2)).sum()
}

/// Whether a run turns its parts (`TURNING` in the tests; in the profiling build a measurement's ATOM_PROF_TURN=1;
/// never in the product). Dormant (2026-11-23, STEP3): measured on PGC3D, TET and tilt2 a turned step takes 4–18 %
/// longer than the whole one (the turned node 35–55 %, a jump's photon node made whole 60–90 %, the scattered light
/// on the places the turn adds 2.3–5.9 times, against 30–70 µs the integration saves), tilt0's 25 % less is not
/// affordable on its remainder's bound (strict, 300–500 times what it bounds); and a turned model's Liouvillian keeps
/// the trace to its second order alone (TET off the centre: 1.4e-5 over 1 ms, `the_turn_s_remainder_with_jumps`), so
/// its trace's mode is no rounding zero and the selection's memory (paths.rs `memory`) reads it as the slowest
/// relaxation: TET's τ_mem 72 µs → the run's 1000 µs, and its selection another.
fn turning() -> bool {
    #[cfg(test)]
    return TURNING.with(|t| t.get());
    #[cfg(all(not(test), feature = "prof"))]
    return std::env::var("ATOM_PROF_TURN").is_ok_and(|v| v == "1");
    #[cfg(all(not(test), not(feature = "prof")))]
    false
}

/// A turned run's sizes over the nodes it made (module documentation, # The parts turned): the largest frame
/// ‖S‖ + Σ‖S_k‖, the largest remainder's rate Y₃, and the largest gradient of the frame ‖∇S‖ (1/µm; the frame's
/// part of the momentum), as the bits of nonnegative f64s (their order the numbers').
#[derive(Debug, Default)]
pub struct Seen {
    frame: std::sync::atomic::AtomicU64,
    rate: std::sync::atomic::AtomicU64,
    gradient: std::sync::atomic::AtomicU64,
}

impl Seen {
    fn note(&self, frame: f64, rate: f64, gradient: f64) {
        use std::sync::atomic::Ordering::Relaxed;
        self.frame.fetch_max(frame.max(0.0).to_bits(), Relaxed);
        self.rate.fetch_max(rate.max(0.0).to_bits(), Relaxed);
        self.gradient.fetch_max(gradient.max(0.0).to_bits(), Relaxed);
    }

    /// The largest frame, rate and frame's gradient noted.
    pub fn get(&self) -> (f64, f64, f64) {
        use std::sync::atomic::Ordering::Relaxed;
        (f64::from_bits(self.frame.load(Relaxed)), f64::from_bits(self.rate.load(Relaxed)), f64::from_bits(self.gradient.load(Relaxed)))
    }
}

/// The tables' Chebyshev degree in s, and the Doppler series' terms they hold: the budget's candidates J = 1…4 (# The
/// tables' budget; rs/fold_tabled.rs).
const TABLE_DEGREE: usize = 4;
const TABLE_ORDERS: usize = 4;

/// A run's tables of pair products as it buys them (module documentation, # The tables): the work its trajectories'
/// nodes off the fold have taken (real multiply-adds, whole numbers: the same sum in any order), the least tweezer
/// intensity s at one of them (the bits of a nonnegative f64), what the tables were last priced at where that was more
/// (the bits of an f64), and the tables once bought with the kinds of node they are read for (`Kind`) -- `Some(None)`
/// where the run's plan has none or no kind costs less off them.
#[derive(Debug)]
struct Tabling {
    rent: std::sync::atomic::AtomicU64,
    low: std::sync::atomic::AtomicU64,
    price: std::sync::atomic::AtomicU64,
    tables: std::sync::OnceLock<Option<Bought>>,
}

/// A run's tables as bought (`Tabling`): the tables, the kinds of node they are read for, and the Doppler series'
/// budget a node off them is held to -- [its rates', its friction's] relative error (# The tables' budget).
#[derive(Debug)]
struct Bought {
    tables: Arc<PairTables>,
    uses: [bool; KINDS],
    budget: [f64; 2],
}

/// The kinds of node the tables are weighed for apart (# The tables): a step's node, the middle's with the lift (a
/// step that reads), a jump's photon's (the channels with their photon forms, no H_eff), a whole node with the photon
/// forms.
const KINDS: usize = 4;

/// A node's kind (`KINDS`) by what it makes.
fn kind(making: Making) -> usize {
    match (making.effective, making.lift, making.kicks) {
        (false, ..) => 2,
        (true, true, _) => 1,
        (true, false, true) => 3,
        (true, false, false) => 0,
    }
}

/// What each kind makes, in `KINDS`' order.
const MAKINGS: [Making; KINDS] = [Making::node(false), Making { lift: true, ..Making::node(false) }, Making::PHOTONS, Making::node(true)];

impl Default for Tabling {
    fn default() -> Tabling {
        Tabling { rent: Default::default(), low: f64::INFINITY.to_bits().into(), price: 0f64.to_bits().into(), tables: std::sync::OnceLock::new() }
    }
}

/// The turn's decision (`Selected::turn_parts`): the account it was weighed on, the dropped terms' rate and distance
/// beside the turn's, the unturned run's W, and the turned run's step work a µs.
#[derive(Debug, Clone, Copy)]
struct TurnCheck {
    account: Account,
    base: (f64, f64),
    unturned: f64,
    work: f64,
}

/// The requests whose turn did not hold on its pilot's nodes (`Selected::turn_holds`; solve.rs `attempt`), by the
/// selection's key (select.rs `key`): their runs are made unturned from then on, as a choice kept is the scene's.
static UNTURNED: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// Whether a request's turn was found not to hold (`UNTURNED`).
fn unturned(request: &crate::request::Request) -> bool {
    let key = crate::select::key(request);
    UNTURNED.lock().map_or(false, |u| u.contains(&key))
}

/// A request's turn found not to hold on its pilot (`UNTURNED`): its runs are made unturned from then on.
pub fn unturn(request: &crate::request::Request) {
    let key = crate::select::key(request);
    if let Ok(mut u) = UNTURNED.lock() {
        if !u.contains(&key) {
            u.push(key);
        }
    }
}

/// The runs' numbers (`Selected::id`).
static RUNS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// The nodes every run has made so far (`Selected::node_into`): the model at a point, which every trajectory's step,
/// every path of the selection and its reference, every scan makes -- the work counter the solve's supervisor reads
/// to tell a long computation from a stuck one (solve.rs `supervise`). Counted, never read by any result.
pub(crate) static NODES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The places of a contraction's outputs in a node (`Selected::layout`): per H output its elements (place, generator's
/// slot, gradient's slot); per channel output its elements (place, class, axis, component, element); the decay's
/// products.
#[derive(Clone, Debug, Default)]
struct Tables {
    h: Vec<Vec<(u32, u32, u32)>>,
    channels: Vec<Vec<(u32, u32, u8, u32, u32)>>,
    /// The decay's factors: every spherical jump component; where each one's numbers start in the thread's buffer.
    jumps: Vec<Spherical>,
    starts: Vec<usize>,
    decay: Vec<Decay>,
    /// The decay's products by their structural terms (`Selected::decay_terms`): per entry of a product (ℓ_l†ℓ_r at
    /// a place) where its terms end in `terms`, each term the places of its two factors in the components' buffer --
    /// a ground row the two share, the rows in their order --, and the entry's places in the generator.
    terms: Vec<(u32, u32)>,
    entries: Vec<Entry>,
}

/// An entry of the decay's products (`Tables::entries`): where its terms end; the forward slot and place a·n + b of
/// ℓ_l†ℓ_r, and where l ≠ r the back slot and place b·n + a of its adjoint (u32::MAX: none).
#[derive(Clone, Copy, Debug)]
struct Entry {
    end: u32,
    forward: (u32, u32),
    back: (u32, u32),
}

/// A jump component in the emission's spherical basis (`Selected::layout`): its class, channel q and beat, its columns
/// and rows, and per element its place among ground × columns (row-major) with the Cartesian elements it sums (axis,
/// component, element, U_qa).
#[derive(Clone, Debug)]
struct Spherical {
    class: usize,
    q: usize,
    beat: usize,
    cols: Vec<usize>,
    rows: Vec<usize>,
    elements: Vec<(usize, Vec<(usize, usize, usize, C64)>)>,
}

/// One product of the decay (`Selected::layout`): of the class's axis the components l ≤ r (indices into
/// `Tables::jumps`), the generator's slots of their beats r − l and l − r.
#[derive(Clone, Copy, Debug)]
struct Decay {
    left: usize,
    right: usize,
    forward: usize,
    back: usize,
}

/// OLDNEW's check (tests alone): every run of ground states alone split by K's fast parts as it is made
/// (`Selected::split_by_parts`), whole solves included.
#[cfg(test)]
pub(crate) static SPLIT_PARTS: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// A thread's buffers for the nodes and the readouts of one run (`Selected::node_into`, `Selected::rows`): the point,
/// the fold's numbers, the contraction.
#[derive(Default)]
struct Work {
    point: crate::scene::Point,
    local: Local,
    out: Out,
    /// The jump components on the ground rows (`GROUND` × their columns each).
    jumps: Vec<C64>,
    /// A readout's (`Selected::lifted`).
    rows: Rows,
    /// A node's off the tables (rs/fold_tabled.rs).
    pairs: PairScratch,
}

/// A readout's buffers (`Selected::lifted`): the state over its norm, a group's lift (Q × P), Q's amplitudes by fast
/// index (the fast indices in the order met, each's Q).
#[derive(Default)]
struct Rows {
    state: Vec<C64>,
    group: Vec<C64>,
    fasts: Vec<usize>,
    amplitudes: Vec<Vec<C64>>,
}

/// A thread's buffers for a middle's readings (`SelectedMiddle`): a node's beats' phases, a tone's density and its
/// adjoint, the gradient's chord and turn, every class's photons and their rates and moments.
#[derive(Default)]
struct Reads {
    phases: Vec<C64>,
    sigma: Vec<C64>,
    dagger: Vec<C64>,
    /// A density of the frame as the lab's (`SelectedMiddle::unframed`).
    lab: Vec<C64>,
    minus: [Vec<C64>; 3],
    turned: [Vec<C64>; 3],
    photons: Vec<Photons>,
    rates: Vec<f64>,
    moments: Vec<Moments>,
}

#[cfg(test)]
thread_local! {
    /// The nodes the middles made, without and with the photon forms (a check's count).
    pub(crate) static MADE: RefCell<[usize; 2]> = const { RefCell::new([0; 2]) };
}

thread_local! {
    /// The thread's buffers, for the run whose number it holds.
    static WORK: RefCell<Option<(usize, Work)>> = const { RefCell::new(None) };
    /// Nodes no middle holds any more, by their run's number: a middle's node is made in one of them
    /// (`Selected::pooled`).
    static POOL: RefCell<Vec<(usize, Rc<Node>)>> = const { RefCell::new(Vec::new()) };
    /// The middles' lists of their nodes, empty, for the next middles.
    static LISTS: RefCell<Vec<Vec<Rc<Node>>>> = const { RefCell::new(Vec::new()) };
    /// The middles' readings' buffers.
    static READS: RefCell<Reads> = RefCell::new(Reads::default());
}

// The run is one object the trajectories' threads share (solve.rs): Send + Sync, by the compiler.
const _: () = {
    const fn shared<T: Send + Sync>() {}
    shared::<Selected>();
};

/// A beat: the fast index n = (n_φ, n_tw, n_ν) and the slow remainder p of a frame vector (`Frame::split`, exact).
pub type Exact = ([i64; 3], Vec<Ratio>);

/// The run's beats (module documentation, # The generator by beats): every exact beat met, with its value, and per
/// component key the beat of each of its elements -- P × P, or Q × P for the lift; for the jumps their slow remainders
/// alone -- made where first asked for. Exact handles: a beat's value is its own vector's (`Selected::value`), whatever
/// order the threads meet it in.
#[derive(Default)]
struct Beats {
    ids: HashMap<Exact, usize>,
    exact: Vec<Exact>,
    value: Vec<f64>,
    elements: HashMap<(Key, u8), Arc<Vec<usize>>>,
    /// The beat k − k' of two jump beats (k, k'), made where first asked for.
    differences: HashMap<(usize, usize), usize>,
}

/// Which elements' beats (`Selected::elements`): H_eff's (P × P), the jumps' slow remainders (P × P), the lift's (Q × P).
#[derive(Clone, Copy, PartialEq)]
enum Of {
    Generator,
    Jump,
    Lift,
}

/// One jump component of a class's axis: its beat and its elements in P × P, in their places' order (row-major).
#[derive(Clone, Debug)]
pub struct Jump {
    pub beat: usize,
    pub elements: Vec<Element>,
}

/// An element of a jump component: its place x = row·n + column in P × P, L, ∇L and the photon form K = k∘L (zero in a
/// node made without the forms, `Node::kicked`).
#[derive(Clone, Copy, Debug)]
pub struct Element {
    pub at: usize,
    pub value: C64,
    pub gradient: [C64; 3],
    pub kick: C64,
}

/// A class of photons (module documentation, # The photons): its fast index, k_c, and per Cartesian axis its
/// components.
#[derive(Clone, Debug)]
pub struct Class {
    pub fast: [i64; 2],
    pub absorbed: f64,
    pub axes: [Vec<Jump>; 3],
}

/// The model at one point of a path in the frame (module documentation, # The nodes): the generator's components by
/// beat, H_eff's gradient's, the photon classes and whether they hold the photon forms (a jump's node); the point and
/// momentum it was made at; its counted work (real multiply-adds, decide.rs # Work: the point and the source's values,
/// the fold, its contraction, the filling and the turn, each loop as it runs, `Selected::node_work`); and at a middle
/// the lift by the lab's groups
/// with its gradient (`Out::lift`: the readings on the middle, `SelectedMiddle::readout`), elsewhere none. A jump's
/// photon's node holds the photon classes alone (`whole` false, `Making::PHOTONS`: its generator and gradient none).
#[derive(Clone, Debug)]
pub struct Node {
    pub kappa: V3,
    pub at: V3,
    pub generator: Vec<(usize, Mat)>,
    pub gradient: Vec<(usize, [Mat; 3])>,
    pub classes: Vec<Class>,
    pub kicked: bool,
    pub whole: bool,
    pub work: f64,
    pub lift: Vec<(Mat, [Mat; 3])>,
    /// Off the run's tables, the relative error of its rates and of its friction from the fold's own (# The tables'
    /// budget): its Doppler series' tail at the terms it took, its interpolation's, ΔG's and the spans' parts; none off
    /// the fold.
    pub deviation: [f64; 2],
}

/// A candidate tone (`step::Middle::beats`): its beat ν > 0 and the beats of its members at e^{+iνu} and e^{−iνu}.
#[derive(Clone, Copy, Debug)]
struct Candidate {
    nu: f64,
    plus: Option<usize>,
    minus: Option<usize>,
}

/// A path through a middle: the momentum the model is taken at and the velocity.
pub struct Path {
    kappa: V3,
    velocity: V3,
}

/// The cluster's generator at a point (`step::Middle`): the nodes made where asked for (`Node`), the middle's time.
pub struct SelectedMiddle<'a> {
    model: &'a Selected,
    r: V3,
    t: f64,
    nodes: RefCell<Vec<Rc<Node>>>,
    made: Cell<f64>,
    /// Whether its nodes are read off the run's tables (# The tables: its trajectory's version), and whether the step
    /// reads a time on it (`step::Middle::reading`: its middle's node with the lift).
    tabled: bool,
    reads: Cell<bool>,
    /// The largest deviation of the nodes it made (`Node::deviation`).
    deviation: Cell<[f64; 2]>,
}

fn zeros(n: usize, m: usize) -> [Mat; 3] {
    std::array::from_fn(|_| Mat::zeros(n, m))
}

/// p − q, componentwise, exact.
fn minus(p: &[Ratio], q: &[Ratio]) -> Vec<Ratio> {
    p.iter().zip(q).map(|(a, b)| Ratio::new(a.num * b.den - b.num * a.den, a.den * b.den)).collect()
}

/// −p, exact.
fn negated(p: &[Ratio]) -> Vec<Ratio> {
    p.iter().map(|a| Ratio::new(-a.num, a.den)).collect()
}

/// tr(A ρ) of an n × n A and ρ (row-major): Σ_ab A_ab ρ_ba.
fn trace(a: &Mat, rho: &[C64]) -> C64 {
    let n = a.rows;
    let mut sum = C64::ZERO;
    for x in 0..n {
        for y in 0..n {
            sum += a.data[x * n + y] * rho[y * n + x];
        }
    }
    sum
}

/// The place of a beat's component in a list by beats, made (zero) where it is not there yet.
fn slot<T>(list: &mut Vec<(usize, T)>, beat: usize, make: impl FnOnce() -> T) -> usize {
    match list.iter().position(|x| x.0 == beat) {
        Some(i) => i,
        None => {
            list.push((beat, make()));
            list.len() - 1
        }
    }
}

impl Selected {
    /// The run's model of a scene on the selection's P (module documentation), the splitting's reference state's ground
    /// populations `reference`. The frame is the checkpoint's centre's at rest (floquet/frame.rs), the fold's sectors
    /// its (`Rs::new`). Err where P does not hold every ground state (a jump lands on any of them).
    pub fn new(scene: Scene, report: Selection, reference: [f64; GROUND]) -> Result<Selected, String> {
        Selected::made(scene, report, reference, true, &Drops::Judged)
    }

    /// The model, its states in their exact blocks or (`split` false: what the tests check the blocks against) one, the
    /// beats `drops` says dropped.
    fn made(scene: Scene, report: Selection, reference: [f64; GROUND], split: bool, drops: &Drops) -> Result<Selected, String> {
        let run = Selected::made_model(scene, report, reference, split, drops)?;
        // (A measurement's, the profiling build's alone: ATOM_PROF_SPLIT=drop cuts the blocks by K's fast parts where
        // the elements between them are below a hundredth of their diagonal gap, leaving those elements out, on a run
        // made unturned (# The parts turned).)
        #[cfg(feature = "prof")]
        if let Ok(mode) = std::env::var("ATOM_PROF_SPLIT") {
            let pairs = run.across_pairs(1e-2);
            crate::prof::note(|| format!("split {mode}: the parts' pairs below 1e-2 of their gap {pairs:?}"));
            if !pairs.is_empty() && split {
                let mut run = run;
                run.cut_parts(&pairs);
                return Ok(run);
            }
        }
        Ok(run)
    }

    /// `made`'s model, before any measurement's switch.
    fn made_model(scene: Scene, report: Selection, reference: [f64; GROUND], split: bool, drops: &Drops) -> Result<Selected, String> {
        // The frame at rest with the beams' recoil shifts (module documentation, # One cluster, one frame).
        let recoil = scene.doppler([0.0; 3]);
        let centre = scene.components([0.0; 3], &recoil, 1.0);
        let frame = Frame::new(&scene, &Hilbert::full(&scene, &centre), &recoil);
        let rs = Rs::new(&scene, &report.keep, frame.sectors());
        let request = &scene.request;
        if rs.keep.iter().filter(|&&s| s < GROUND).count() != GROUND {
            return Err("选出的 P 必须含全部基态（跳跃落到任一基态）".into());
        }
        // The primitives' values: the request's axes Φ_i, then each Doppler class's δ_c (a laser's recoil): a generator
        // of the class has the coordinate −r on it, its shift r δ_c.
        let mut axes: Vec<f64> = request.frequencies_mhz.iter().map(|f| RAD_PER_US_PER_MHZ * f).collect();
        let count = scene.generators.len();
        let unit = |g: usize| -> FrameKey { (0..=count).map(|h| i64::from(h == g)).collect() };
        let width = frame.primitive(&unit(0)).len();
        for c in axes.len()..width {
            let (g, r) = (0..count)
                .find_map(|g| {
                    let p = frame.primitive(&unit(g));
                    (p[c].num != 0).then(|| (g, p[c]))
                })
                .expect("a generator of the Doppler class");
            let l = scene.generators[g].beam.expect("a beam's class");
            axes.push(recoil[l] / (-(r.num as f64) / r.den as f64));
        }
        let theta = rs.keep.iter().map(|&s| frame.theta[s].iter().zip(&frame.values).map(|(&k, w)| k as f64 * w).sum()).collect();
        let n = rs.keep.len();
        // The Doppler frame (module documentation, # One cluster, one frame): each state's K_a = Σ_l (θ_a − θ_0)[l] k_l
        // over the non-tweezer beams' generators l of its time frame, against P's first state's (a common K is a phase
        // of the whole state, which nothing reads); exactly, as integers on the beams, and their wave vectors.
        let beams: Vec<(usize, usize)> = scene.generators.iter().enumerate().filter_map(|(g, generator)| generator.beam.filter(|&l| l != scene.tweezer()).map(|l| (g, l))).collect();
        let on = |s: usize| -> Vec<i64> { beams.iter().map(|&(g, _)| frame.theta[s][g] - frame.theta[rs.keep[0]][g]).collect() };
        let wave = |on: &[i64]| -> V3 {
            let mut k = [0.0; 3];
            for (&c, &(_, l)) in on.iter().zip(&beams).filter(|(c, _)| **c != 0) {
                let q = scene.geometry[l].wave_vector();
                for a in 0..3 {
                    k[a] += c as f64 * q[a];
                }
            }
            k
        };
        let exact: Vec<Vec<i64>> = rs.keep.iter().map(|&s| on(s)).collect();
        let doppler: Vec<V3> = exact.iter().map(|on| wave(on)).collect();
        let folded_waves: Vec<V3> = rs.folded.iter().map(|&q| wave(&on(q))).collect();
        // The light's pattern the frame leaves (`Selected::pattern`): per structural element a ← b of the generator, of
        // the key c, its turn along the path in the frame, the spatial twin of c − θ_a + θ_b over the non-tweezer beams;
        // where that is no rounding zero (the sum's own, units.rs `rounding`), the beams it turns with -- those of
        // the twin's nonzero coordinates -- by their wave numbers, the largest of them over every element.
        let pattern = {
            let grouping = rs.folding.by_key();
            let mut most = 0.0_f64;
            for (key, list) in grouping.outputs.iter().map(|o| &o.0).zip(&grouping.lists) {
                for &x in list {
                    let (a, b) = (rs.keep[x as usize / n], rs.keep[x as usize % n]);
                    let (mut q, mut size, mut carried) = ([0.0; 3], 0.0, 0.0_f64);
                    for &(g, l) in &beams {
                        let r = key.get(g).map_or(0, |&k| i64::from(k)) - frame.theta[a][g] + frame.theta[b][g];
                        if r != 0 {
                            let k = scene.geometry[l].wave_vector();
                            for i in 0..3 {
                                q[i] += r as f64 * k[i];
                            }
                            size += (r as f64).abs() * scene.geometry[l].wave_number;
                            carried = larger(carried, scene.geometry[l].wave_number);
                        }
                    }
                    if (0..3).map(|i| q[i] * q[i]).sum::<f64>().sqrt() > rounding(3 * beams.len()) * size {
                        most = larger(most, carried);
                    }
                }
            }
            most
        };
        let cluster = Cluster { ground: GROUND, frames: vec![Beat::whole(vec![]); n], waves: doppler, waves_exact: exact.into_iter().map(Beat::whole).collect(), states: rs.keep.clone() };
        // The tweezer's unit fold's static part (`Tweezer::still`): its ground diagonal `Scene::trap_potential`'s depth, the
        // whole the other lights' smooth shifts' (`Scene::smooth`, the reported trap's).
        let unit = rs.tweezer().still();
        let depth = std::array::from_fn(|g| unit[(g, g)].re);
        let smooth = scene.smooth(&unit);
        let decay = |e: usize, g: usize| -> f64 { request.emission.iter().map(|c| c[(g, e)].norm_sqr()).sum() };
        let branching = rs
            .keep
            .iter()
            .map(|&s| {
                if s < GROUND {
                    std::array::from_fn(|g| f64::from(u8::from(g == s)))
                } else {
                    let total: f64 = (0..GROUND).map(|g| decay(s, g)).sum();
                    std::array::from_fn(|g| decay(s, g) / total)
                }
            })
            .collect();
        let optical = rs.keep.iter().filter(|&&s| s >= GROUND).map(|&e| 0.5 * (0..GROUND).map(|g| decay(e, g)).sum::<f64>()).reduce(smaller);
        let waves = scene.geometry.iter().enumerate().filter(|(l, _)| *l != scene.tweezer()).map(|(_, g)| g.wave_vector()).collect();
        // The fold's pieces by the lab's frequency at rest -- the frame's primitive vector of their keys --, each group
        // under the first of its keys met.
        let label = |keys: &[Key]| -> Vec<Key> {
            let mut first: Vec<(Vec<Ratio>, Key)> = Vec::new();
            keys.iter()
                .map(|key| {
                    let v: FrameKey = key.iter().map(|&k| i64::from(k)).chain(std::iter::once(0)).collect();
                    let p = frame.primitive(&v);
                    match first.iter().find(|f| f.0 == p) {
                        Some(f) => f.1.clone(),
                        None => {
                            first.push((p, key.clone()));
                            key.clone()
                        }
                    }
                })
                .collect()
        };
        let labels = label(rs.folding.keys());
        // The decay's structure: the same fold with its channels in the emission's spherical basis.
        let spherical_plan = rs.spherical();
        let spherical = Grouping::new(&spherical_plan, &label(spherical_plan.keys()));
        let lab = Grouping::new(rs.plan(), &labels);
        let n = rs.keep.len();
        let mut beats = Beats::default();
        let still: Exact = ([0; 3], vec![Ratio::ZERO; width]);
        beats.ids.insert(still.clone(), 0);
        beats.exact.push(still);
        beats.value.push(0.0);
        let mut run = Selected {
            light: Light::new(request),
            hbar_over_mass: hbar_over_mass(request.mass_kg),
            frame,
            axes,
            theta,
            cluster,
            depth,
            smooth,
            reference,
            branching,
            optical,
            waves,
            folded_waves,
            pattern,
            lab,
            template: Node { kappa: [0.0; 3], at: [0.0; 3], generator: vec![(0, Mat::zeros(n, n))], gradient: Vec::new(), classes: Vec::new(), kicked: false, whole: true, work: 0.0, lift: Vec::new(), deviation: [0.0; 2] },
            tables: Tables::default(),
            tabling: Tabling::default(),
            id: RUNS.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
            judgment: Judgment { tiers: Vec::new(), blocks: vec![(0..n).collect()], dropped: 0.0, pairs: Vec::new(), once: 0.0 },
            blocks: vec![(0..n).collect()],
            block_of: vec![0; n],
            turn: None,
            seen: Arc::new(Seen::default()),
            turn_check: None,
            turn_terms: Vec::new(),
            beats: RwLock::new(beats),
            candidates: Vec::new(),
            keys: Vec::new(),
            lifts: Vec::new(),
            rs,
            report,
            scene,
        };
        run.layout(&spherical);
        run.judge(split, (&spherical_plan, &spherical), drops);
        // (OLDNEW's check, tests alone: a model of ground states alone split by K's fast parts within its blocks.)
        #[cfg(test)]
        if SPLIT_PARTS.load(std::sync::atomic::Ordering::Relaxed) && run.rs.keep.iter().all(|&s| s < GROUND) {
            run.split_by_parts();
        }
        run.tones();
        run.lift_tables();
        // K's fast parts turned where that costs the run the less (module documentation, # The parts turned), on the run
        // made whole -- its steps are priced --, the decay's products laid on the exact blocks: the elements between the
        // parts are what S is made of.
        if matches!(drops, Drops::Judged) && split && turning() && !unturned(&run.scene.request) {
            run.turn_parts((&spherical_plan, &spherical));
        }
        Ok(run)
    }

    /// The pairs of K's fast parts within a block whose elements are below `ratio` of their gap (the profiling build's
    /// measurement): at the structure's node (`structure`), per pair the largest |G_ab| of its static slot between them over
    /// the least |Re G_aa − Re G_bb| of the diagonal in the frame (the structure's θ taken off it), H's and the decay's
    /// alike.
    #[cfg(feature = "prof")]
    fn across_pairs(&self, ratio: f64) -> Vec<[[i64; 2]; 2]> {
        let n = self.rs.keep.len();
        let parts = self.rs.plan().parts().to_vec();
        let (node, _) = self.structure();
        let Some(g) = node.generator.iter().find(|g| g.0 == 0).map(|g| &g.1) else { return Vec::new() };
        let mut pairs: Vec<([[i64; 2]; 2], f64, f64)> = Vec::new();
        for a in 0..n {
            for b in 0..n {
                if self.block_of[a] != self.block_of[b] || parts[a] >= parts[b] {
                    continue;
                }
                let key = [parts[a], parts[b]];
                let i = pairs.iter().position(|p| p.0 == key).unwrap_or_else(|| {
                    pairs.push((key, 0.0, f64::INFINITY));
                    pairs.len() - 1
                });
                pairs[i].1 = larger(pairs[i].1, larger(g[(a, b)].abs(), g[(b, a)].abs()));
                pairs[i].2 = smaller(pairs[i].2, ((g[(a, a)].re - self.theta[a]) - (g[(b, b)].re - self.theta[b])).abs());
            }
        }
        crate::prof::note(|| format!("split: the parts' pairs (pair, largest element, least gap) {pairs:?}"));
        pairs.into_iter().filter(|p| p.1 < ratio * p.2).map(|p| p.0).collect()
    }

    /// The blocks cut by K's fast parts (the profiling build's measurement, ATOM_PROF_SPLIT): within each block, the parts joined but where a pair
    /// is in `pairs`; the elements between blocks left out of the layout as `judge` leaves a judged block's, the decay's
    /// products recut. Said (the profiling build): the cut and the largest element left out of the static slot.
    #[cfg(feature = "prof")]
    fn cut_parts(&mut self, pairs: &[[[i64; 2]; 2]]) {
        let n = self.rs.keep.len();
        let parts = self.rs.plan().parts().to_vec();
        let apart = |x: [i64; 2], y: [i64; 2]| pairs.iter().any(|p| (p[0] == x && p[1] == y) || (p[0] == y && p[1] == x));
        // Union of the states of one block whose parts are not apart.
        fn find(root: &mut [usize], x: usize) -> usize {
            let mut x = x;
            while root[x] != x {
                root[x] = root[root[x]];
                x = root[x];
            }
            x
        }
        let mut root: Vec<usize> = (0..n).collect();
        for a in 0..n {
            for b in 0..n {
                if self.block_of[a] == self.block_of[b] && !apart(parts[a], parts[b]) {
                    let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                    root[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let mut of: Vec<usize> = Vec::new();
        for a in 0..n {
            let r = find(&mut root, a);
            match of.iter().position(|&x| x == r) {
                Some(i) => blocks[i].push(a),
                None => {
                    of.push(r);
                    blocks.push(vec![a]);
                }
            }
        }
        #[cfg(feature = "prof")]
        {
            let (node, _) = self.structure();
            let side = |a: usize| blocks.iter().position(|x| x.contains(&a));
            let left = node.generator.iter().find(|g| g.0 == 0).map_or(0.0, |(_, g)| {
                let mut most = 0.0_f64;
                for a in 0..n {
                    for b in 0..n {
                        if side(a) != side(b) {
                            most = larger(most, g[(a, b)].abs());
                        }
                    }
                }
                most
            });
            crate::prof::note(|| format!("split: blocks {:?} -> {:?}; the static slot's largest element between them {left:.3e}", self.blocks.iter().map(Vec::len).collect::<Vec<_>>(), blocks.iter().map(Vec::len).collect::<Vec<_>>()));
        }
        self.blocks.clone_from(&blocks);
        self.judgment.blocks.clone_from(&blocks);
        for (b, states) in blocks.iter().enumerate() {
            for &st in states {
                self.block_of[st] = b;
            }
        }
        let block_of = self.block_of.clone();
        for list in self.tables.h.iter_mut() {
            for e in list.iter_mut().filter(|e| block_of[e.0 as usize / n] != block_of[e.0 as usize % n]) {
                *e = (e.0, u32::MAX, u32::MAX);
            }
        }
        self.decay_terms();
        let id = self.id;
        POOL.with(|p| p.borrow_mut().retain(|x| x.0 != id));
    }

    /// The pairs of K's fast parts below `ratio` of their gap (`across_pairs`) and the blocks cut by them, the elements
    /// between left out (`cut_parts`; the tests').
    #[cfg(all(test, feature = "prof"))]
    pub(crate) fn split_for_test(&mut self) -> Vec<[[i64; 2]; 2]> {
        let pairs = self.across_pairs(1e-2);
        if !pairs.is_empty() {
            self.cut_parts(&pairs);
        }
        pairs
    }

    /// OLDNEW's check (tests alone): the blocks cut by K's fast parts within the judged ones (F = 1 and F = 2 apart, the
    /// static elements between them left out of the layout as `judge` leaves a judged block's), the decay's products recut.
    #[cfg(test)]
    pub(crate) fn split_by_parts(&mut self) -> Vec<Vec<usize>> {
        let n = self.rs.keep.len();
        let parts = self.rs.plan().parts().to_vec();
        let mut blocks: Vec<Vec<usize>> = Vec::new();
        let mut seen: Vec<(usize, [i64; 2])> = Vec::new();
        for a in 0..n {
            let key = (self.block_of[a], parts[a]);
            match seen.iter().position(|p| *p == key) {
                Some(i) => blocks[i].push(a),
                None => {
                    seen.push(key);
                    blocks.push(vec![a]);
                }
            }
        }
        self.blocks.clone_from(&blocks);
        for (b, states) in blocks.iter().enumerate() {
            for &st in states {
                self.block_of[st] = b;
            }
        }
        let block_of = self.block_of.clone();
        for list in self.tables.h.iter_mut() {
            for e in list.iter_mut().filter(|e| block_of[e.0 as usize / n] != block_of[e.0 as usize % n]) {
                *e = (e.0, u32::MAX, u32::MAX);
            }
        }
        self.decay_terms();
        let id = self.id;
        POOL.with(|p| p.borrow_mut().retain(|x| x.0 != id));
        blocks
    }

    /// The selection's report (stderr): P by levels and state by state, the states folded, the selection's notes (its
    /// chain, and the choice's lines where it was made: select.rs `Selection::notes`), the dropped beats.
    pub fn report(&self) -> String {
        let labels = &self.scene.request.states;
        let mut levels: Vec<String> = Vec::new();
        for &s in &self.rs.keep {
            // The level of a label manifold-F-m (`request::Level::parse`): its manifold and F, whatever m's sign.
            let label = labels[s].splitn(3, '-').take(2).collect::<Vec<_>>().join("-");
            if !levels.contains(&label) {
                levels.push(label);
            }
        }
        let names = |states: &[usize]| states.iter().map(|&s| labels[s].as_str()).collect::<Vec<_>>().join(", ");
        let mut out = format!("选态：P = {} 态 {:?}（结构链用时 {:.2} s）\n", self.rs.keep.len(), levels, self.report.seconds);
        out += &format!("  P 的态：{}\n", names(&self.rs.keep));
        out += &format!("  折叠 {} 态：{}\n", self.rs.folded.len(), if self.rs.folded.is_empty() { "无".to_string() } else { names(&self.rs.folded) });
        for note in self.report.notes() {
            out += &format!("  注：{note}\n");
        }
        let dropped: Vec<f64> = self.read(|b| self.judgment.tiers.iter().filter(|t| t.1 == Tier::Drop).map(|t| b.value[t.0]).filter(|v| *v > 0.0).collect());
        if !dropped.is_empty() {
            let length = self.scene.request.times_us.last().copied().unwrap_or(0.0);
            let beats = dropped.iter().map(|v| format!("±{v:.4}")).collect::<Vec<_>>().join("、");
            out += &format!("  删去拍 {} 对：{beats} rad/µs；迹距离增长率上界 Y = {:.2e} /µs（{length} µs 内 D ≤ {:.2e}）\n", dropped.len(), self.judgment.dropped, crate::model::classify::distance(self.judgment.dropped, length));
        }
        out
    }

    /// The run's light (the region, the beams' geometries).
    /// The largest wave number of the light's pattern the Doppler frame leaves to the step's nodes (1/µm; module
    /// documentation, # One cluster, one frame): of the non-tweezer beams that some element of the generator still turns
    /// with along a path in the frame -- the beams of the spatial twin of its slow remainder, where that twin is no rounding
    /// zero. None where the frame takes every plane wave away (a running wave's: R9_PUSH's push); every beam of a
    /// standing wave, a lattice and six beams' cooling, whose pattern no frame removes. What the step's prior asks the
    /// step to resolve (solve.rs `prior`).
    pub fn pattern(&self) -> f64 {
        self.pattern
    }

    pub fn light(&self) -> &Light {
        &self.light
    }

    /// The prepared potential of the ground populations p (module documentation, # The rest of the model): the trap
    /// light's (`Scene::trap_potential`).
    pub fn prepared(&self, p: &[f64; GROUND]) -> Potential<'_> {
        Potential { model: self, weights: *p, running: false }
    }

    /// The reported trap's potential of the ground populations p (SPEC §1.3): the trap light's with the other lights'
    /// smooth shifts (`Scene::running_potential`).
    pub fn running(&self, p: &[f64; GROUND]) -> Potential<'_> {
        Potential { model: self, weights: *p, running: true }
    }

    /// A beat's value (rad/µs): n·(φ, Φ_tw, Φ_ν) + Σ_i p_i Φ_i, its own vector's, correctly rounded (paths.rs: the
    /// recycling's beat differences' too).
    pub(crate) fn value_of(&self, (n, p): &Exact) -> f64 {
        let fast = n.iter().zip(&self.frame.fast).filter(|(k, _)| **k != 0).map(|(&k, f)| k as f64 * f);
        exact_sum(fast.chain(p.iter().zip(&self.axes).filter(|(c, _)| c.num != 0).map(|(c, x)| c.num as f64 / c.den as f64 * x)))
    }

    /// The handle of a beat, made where first met.
    fn intern(&self, beats: &mut Beats, exact: Exact) -> usize {
        if let Some(&id) = beats.ids.get(&exact) {
            return id;
        }
        let id = beats.exact.len();
        beats.value.push(self.value_of(&exact));
        beats.ids.insert(exact.clone(), id);
        beats.exact.push(exact);
        id
    }

    /// The beat of every element of a component of key c (module documentation, # One cluster, one frame), row-major:
    /// rows P, or Q for the lift, columns P; for the jumps the element's slow remainder alone, its fast index the
    /// channel's phase.
    fn elements(&self, key: &Key, of: Of) -> Arc<Vec<usize>> {
        let tag = (key.clone(), of as u8);
        if let Some(t) = self.beats.read().unwrap_or_else(|e| e.into_inner()).elements.get(&tag) {
            return t.clone();
        }
        let rows = if of == Of::Lift { &self.rs.folded } else { &self.rs.keep };
        let theta = &self.frame.theta;
        let mut exact: Vec<Exact> = Vec::with_capacity(rows.len() * self.rs.keep.len());
        for &a in rows {
            for &b in &self.rs.keep {
                let v: FrameKey = (0..theta[a].len()).map(|i| key.get(i).map_or(0, |&k| i64::from(k)) - theta[a][i] + theta[b][i]).collect();
                let (n, p) = self.frame.split(&v);
                exact.push((if of == Of::Jump { [0; 3] } else { n }, p));
            }
        }
        let mut beats = self.beats.write().unwrap_or_else(|e| e.into_inner());
        let table = Arc::new(exact.into_iter().map(|x| self.intern(&mut beats, x)).collect::<Vec<usize>>());
        beats.elements.entry(tag).or_insert(table).clone()
    }

    /// The handle of the beat k − k' of two jump beats (their slow remainders), made where first asked for.
    fn difference(&self, k: usize, k_: usize) -> usize {
        if let Some(&id) = self.read(|b| b.differences.get(&(k, k_)).copied()).as_ref() {
            return id;
        }
        let mut beats = self.beats.write().unwrap_or_else(|e| e.into_inner());
        let difference = minus(&beats.exact[k].1, &beats.exact[k_].1);
        let id = self.intern(&mut beats, ([0; 3], difference));
        beats.differences.insert((k, k_), id);
        id
    }

    /// What `then` reads of the beats as they stand.
    fn read<R>(&self, then: impl FnOnce(&Beats) -> R) -> R {
        then(&self.beats.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// The beats' values (rad/µs; a beat's handle its place) and exact vectors as they stand.
    pub fn components(&self) -> (Vec<f64>, Vec<Exact>) {
        self.read(|b| (b.value.clone(), b.exact.clone()))
    }

    /// The model at a point for the momentum κ (`Rs::local`), by the lab's frequencies (model/rs.rs, # Contractions):
    /// every component in the group of its key's frequency at rest -- the frame's primitive vector, so that the beams of
    /// one frequency are one group whatever their Doppler shifts (the fold's denominators keep those) --, each group under
    /// the first of its keys met (`Selected::elements` reads a key's beats by its primitive vector alone); what `making`
    /// asks for. The checks' (a readout is `rows`, on the thread's buffers).
    #[cfg(test)]
    fn at(&self, r: V3, kappa: V3, making: Making) -> crate::model::rs::ModelAt {
        let point = self.scene.point_at(r, kappa);
        let mut local = Local::default();
        self.rs.evaluate(&self.scene, &point, &mut local, Making { gradients: true, ..making });
        let mut out = Out::default();
        out.contract(self.rs.plan(), &self.lab, &local, None, making);
        crate::model::rs::model_of(self.rs.plan(), &self.lab, &out, &local.values.frequencies, making)
    }

    /// The node's structure (module documentation, # The nodes), made once from the fold's lab grouping: the beats of
    /// H_eff's and its gradient's structural elements, each output element's place, the photon classes with their
    /// channels' components by beat and their elements, and the decay's products between them.
    fn layout(&mut self, spherical: &Grouping) {
        let n = self.rs.keep.len();
        let lab = &self.lab;
        let mut template = Node { kappa: [0.0; 3], at: [0.0; 3], generator: vec![(0, Mat::zeros(n, n))], gradient: Vec::new(), classes: Vec::new(), kicked: false, whole: true, work: 0.0, lift: Vec::new(), deviation: [0.0; 2] };
        let mut tables = Tables::default();
        for (og, (key, _)) in lab.outputs.iter().enumerate() {
            let beats = self.elements(key, Of::Generator);
            let mut list = Vec::new();
            for &x in &lab.lists[og] {
                let beat = beats[x as usize];
                let g = slot(&mut template.generator, beat, || Mat::zeros(n, n));
                let d = slot(&mut template.gradient, beat, || zeros(n, n));
                list.push((x, g as u32, d as u32));
            }
            tables.h.push(list);
        }
        // The photon classes: per channel output its elements by (class, axis, beat), the elements in their places' order.
        let mut met: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut absorbed: Vec<([i64; 2], f64)> = Vec::new();
        for (co, (emission, fast, which, key, _)) in lab.channel_outputs.iter().enumerate() {
            let beats = self.elements(key, Of::Jump);
            let class = match template.classes.iter().position(|c| c.fast == *fast) {
                Some(i) => i,
                None => {
                    template.classes.push(Class { fast: *fast, absorbed: 0.0, axes: Default::default() });
                    template.classes.len() - 1
                }
            };
            if *which == 0 && !lab.channel_lists[co].is_empty() {
                let beams = key.iter().enumerate().filter(|(g, k)| **k != 0 && *g < self.scene.geometry.len()).map(|(g, _)| self.scene.geometry[g].wave_number);
                match absorbed.iter_mut().find(|a| a.0 == *fast) {
                    Some(a) => a.1 = beams.fold(a.1, larger),
                    None => absorbed.push((*fast, beams.fold(0.0, larger))),
                }
            }
            for x in lab.channel_lists[co].iter().map(|&x| x as usize) {
                let axis = &mut template.classes[class].axes[*emission];
                let jump = match axis.iter().position(|j| j.beat == beats[x]) {
                    Some(i) => i,
                    None => {
                        axis.push(Jump { beat: beats[x], elements: Vec::new() });
                        axis.len() - 1
                    }
                };
                met.push((class, *emission, jump, x));
            }
        }
        for c in template.classes.iter_mut() {
            c.absorbed = absorbed.iter().find(|a| a.0 == c.fast).map_or(0.0, |a| a.1);
        }
        met.sort_unstable();
        met.dedup();
        for &(class, axis, jump, x) in &met {
            template.classes[class].axes[axis][jump].elements.push(Element { at: x, value: C64::ZERO, gradient: [C64::ZERO; 3], kick: C64::ZERO });
        }
        for (co, (emission, fast, _, key, _)) in lab.channel_outputs.iter().enumerate() {
            let beats = self.elements(key, Of::Jump);
            let class = template.classes.iter().position(|c| c.fast == *fast).expect("its class");
            let mut list = Vec::new();
            for x in lab.channel_lists[co].iter().map(|&x| x as usize) {
                let axis = &template.classes[class].axes[*emission];
                let jump = axis.iter().position(|j| j.beat == beats[x]).expect("its jump");
                let element = axis[jump].elements.iter().position(|e| e.at == x).expect("its element");
                list.push((x as u32, class as u32, *emission as u8, jump as u32, element as u32));
            }
            tables.channels.push(list);
        }
        // The decay −(i/2) Σ_ch L(t)†L(t) = −(i/2) Σ_ch Σ_q ℓ_q†ℓ_q by beats, its channels in the emission's spherical
        // basis ℓ_q = Σ_a U_qa L_a (rs.rs `channel_basis`): their structure the spherical fold's (`spherical`), each element
        // of ℓ_q the Cartesian elements of its place and beat with their coefficients; per class and q each pair of its
        // components (l ≤ r) whose rows meet, ℓ_l†ℓ_r at the beat r − l and its adjoint at l − r (a jump lands on the
        // ground states: each component's rows are theirs).
        let basis = self.rs.tweezer().basis;
        let mut comps: Vec<(usize, usize, usize, Vec<usize>)> = Vec::new();
        for (co, (q, fast, which, key, _)) in spherical.channel_outputs.iter().enumerate() {
            if *which != 0 {
                continue;
            }
            let beats = self.elements(key, Of::Jump);
            let class = match template.classes.iter().position(|c| c.fast == *fast) {
                Some(i) => i,
                None => continue,
            };
            for x in spherical.channel_lists[co].iter().map(|&x| x as usize) {
                let beat = beats[x];
                match comps.iter_mut().find(|c| c.0 == class && c.1 == *q && c.2 == beat) {
                    Some(c) => c.3.push(x),
                    None => comps.push((class, *q, beat, vec![x])),
                }
            }
        }
        comps.sort_by_key(|c| (c.0, c.1, c.2));
        for (class, q, beat, mut places) in comps {
            places.sort_unstable();
            places.dedup();
            assert!(places.iter().all(|x| x / n < GROUND), "a jump lands on the ground states");
            let mut cols: Vec<usize> = places.iter().map(|x| x % n).collect();
            cols.sort_unstable();
            cols.dedup();
            let mut elements = Vec::new();
            for &x in &places {
                let at = (x / n) * cols.len() + cols.binary_search(&(x % n)).expect("its column");
                let mut sources = Vec::new();
                for (a, &u) in basis[q].iter().enumerate() {
                    if u == C64::ZERO {
                        continue;
                    }
                    let axis = &template.classes[class].axes[a];
                    if let Some(j) = axis.iter().position(|j| j.beat == beat) {
                        if let Some(e) = axis[j].elements.iter().position(|e| e.at == x) {
                            sources.push((a, j, e, u));
                        }
                    }
                }
                elements.push((at, sources));
            }
            let rows: Vec<usize> = {
                let mut r: Vec<usize> = places.iter().map(|x| x / n).collect();
                r.dedup();
                r
            };
            tables.jumps.push(Spherical { class, q, beat, cols, rows, elements });
        }
        let count = tables.jumps.len();
        for l in 0..count {
            for r in l..count {
                let (jl, jr) = (&tables.jumps[l], &tables.jumps[r]);
                if jl.class != jr.class || jl.q != jr.q || !jl.rows.iter().any(|x| jr.rows.contains(x)) {
                    continue;
                }
                let (bl, br) = (jl.beat, jr.beat);
                let forward = if bl == br { 0 } else { self.difference(br, bl) };
                let back = if bl == br { 0 } else { self.difference(bl, br) };
                let forward = slot(&mut template.generator, forward, || Mat::zeros(n, n));
                let back = slot(&mut template.generator, back, || Mat::zeros(n, n));
                tables.decay.push(Decay { left: l, right: r, forward, back });
            }
        }
        let mut at = 0;
        tables.starts = tables.jumps.iter().map(|j| {
            let here = at;
            at += GROUND * j.cols.len();
            here
        }).collect();
        self.template = template;
        self.tables = tables;
        self.decay_terms();
    }

    /// The node's beats and P's exact blocks by the run's judgment (model/classify.rs), read once on the structure's
    /// node (`structure`): every beat neither zero nor dropped is kept, the layout's places at the others go with them
    /// and the slots are renumbered (a place at a beat left out keeps its entry in the outputs' order, marked: nothing is
    /// written there); with `split` (what the tests check the blocks against: one block) the judgment's blocks, every
    /// place between two of them left out of the layout as well and the decay's products between them not made, zero by
    /// construction.
    fn judge(&mut self, split: bool, spherical: (&Plan, &Grouping), drops: &Drops) {
        let n = self.rs.keep.len();
        let (node, scales) = self.structure();
        let mut judgment = Judgment::new(&node, &scales);
        let candidates = judgment.candidates();
        if !candidates.is_empty() {
            let (pairs, account, nodes) = self.drop_sizes(&node, &candidates, spherical);
            let beats: Vec<usize> = node.generator.iter().map(|g| g.0).collect();
            let work = |left: &[usize]| -> f64 {
                let kept: Vec<usize> = candidates.iter().filter(|k| !left.contains(k)).map(|&k| beats[k]).collect();
                self.counted(&kept, nodes)
            };
            match drops {
                Drops::Judged => judgment.drop(&node, &scales, &pairs, &account, &work),
                #[cfg(test)]
                Drops::Only(listed) => {
                    let values: Vec<f64> = self.read(|b| beats.iter().map(|&k| b.value[k]).collect());
                    let chosen: Vec<usize> = (0..pairs.len()).filter(|&i| listed.iter().any(|v| (values[pairs[i].0[0]].abs() - v).abs() <= 1e-9 * v)).collect();
                    judgment.dropping(&node, &scales, &pairs, &chosen);
                }
            }
        }
        let keep: Vec<bool> = (0..judgment.tiers.len()).map(|k| judgment.kept(k)).collect();
        // The slots left, and every place of the layout renumbered (a place at a beat left out goes with it).
        let mut slot_of = vec![usize::MAX; keep.len()];
        let mut generator = Vec::new();
        for (k, g) in std::mem::take(&mut self.template.generator).into_iter().enumerate() {
            if keep[k] {
                slot_of[k] = generator.len();
                generator.push(g);
            }
        }
        self.template.generator = generator;
        let kept_beats: Vec<usize> = self.template.generator.iter().map(|g| g.0).collect();
        let mut gslot_of = vec![usize::MAX; self.template.gradient.len()];
        let mut gradient = Vec::new();
        for (k, g) in std::mem::take(&mut self.template.gradient).into_iter().enumerate() {
            if kept_beats.contains(&g.0) {
                gslot_of[k] = gradient.len();
                gradient.push(g);
            }
        }
        self.template.gradient = gradient;
        for list in self.tables.h.iter_mut() {
            for e in list.iter_mut() {
                *e = if slot_of[e.1 as usize] == usize::MAX { (e.0, u32::MAX, u32::MAX) } else { (e.0, slot_of[e.1 as usize] as u32, gslot_of[e.2 as usize] as u32) };
            }
        }
        self.tables.decay = self
            .tables
            .decay
            .iter()
            .filter(|d| slot_of[d.forward] != usize::MAX && slot_of[d.back] != usize::MAX)
            .map(|d| Decay { forward: slot_of[d.forward], back: slot_of[d.back], ..*d })
            .collect();
        if split && judgment.blocks.len() > 1 {
            self.blocks.clone_from(&judgment.blocks);
            for (b, states) in self.blocks.iter().enumerate() {
                for &s in states {
                    self.block_of[s] = b;
                }
            }
            let block_of = &self.block_of;
            for list in self.tables.h.iter_mut() {
                for e in list.iter_mut().filter(|e| block_of[e.0 as usize / n] != block_of[e.0 as usize % n]) {
                    *e = (e.0, u32::MAX, u32::MAX);
                }
            }
        }
        // K's sideband parts left out and the direct entries spanned where that costs the run the less (module
        // documentation, # The sideband parts dropped, # The direct entries spanned), what they leave out booked with
        // the dropped beats. The decay's products laid on the slots and blocks left first: the nodes these weigh are
        // made (`counted_node`).
        self.decay_terms();
        if matches!(drops, Drops::Judged) {
            let (_, account, _) = self.drop_sizes(&node, &[], spherical);
            judgment.dropped += self.drop_sidebands(&account, judgment.dropped);
            let nodes = self.counted_node();
            judgment.dropped += self.span_directs(&account, nodes, judgment.dropped);
        }
        self.judgment = judgment;
        self.decay_terms();
        // The pooled nodes were made on the layout before.
        POOL.with(|p| p.borrow_mut().retain(|x| x.0 != self.id));
    }

    /// K's fast parts turned (module documentation, # The parts turned): the pairs of parts within an exact block, by
    /// ‖V‖/μ at the structure's point; of none and every first few the cut of least W -- a step's counted work a µs as
    /// the run takes it (`priced_work`, the cut applied) times the account's factor at the estimate's remainder Y₃,
    /// frame 2‖S‖ and frame's gradient (`Account::factor_with`); the cut's blocks set, the layout's jump places closed
    /// under S; the decision's numbers kept for the pilot's check (`turn_holds`).
    fn turn_parts(&mut self, spherical: (&Plan, &Grouping)) {
        let n = self.rs.keep.len();
        let parts = self.rs.plan().parts().to_vec();
        let (node, _) = self.structure();
        let (_, account, _) = self.drop_sizes(&node, &[], spherical);
        let slots = self.template.generator.len();
        // The decision's estimate: the structure's node's own elements (its decay's products in them), per slot.
        let bound: Vec<Vec<f64>> = (0..slots).map(|k| node.generator[k].1.data.iter().map(|z| z.abs()).collect()).collect();
        let lambda = decay_bound(&node);
        let g = &node.generator[0].1;
        let theta = self.theta.clone();
        let gap = |a: usize, b: usize| -> f64 { ((g[(a, a)].re - theta[a]) - (g[(b, b)].re - theta[b])).abs() };
        // The static slot's gradient at the structure's point: the parts' couplings' and the gap's (the frame's gradient).
        let dg = node.gradient.iter().find(|x| x.0 == 0).map(|x| &x.1);
        let slope = |a: usize, b: usize| -> f64 { dg.map_or(0.0, |dg| (0..3).map(|i| dg[i][(a, b)].norm_sqr()).sum::<f64>().sqrt()) };
        let gap_slope = |a: usize, b: usize| -> f64 { dg.map_or(0.0, |dg| (0..3).map(|i| (dg[i][(a, a)] - dg[i][(b, b)]).norm_sqr()).sum::<f64>().sqrt()) };
        // The pairs of parts within an exact block, by ‖V‖/μ.
        let mut pairs: Vec<([[i64; 2]; 2], f64)> = Vec::new();
        for a in 0..n {
            for b in 0..n {
                if self.block_of[a] != self.block_of[b] || parts[a] >= parts[b] {
                    continue;
                }
                let key = [parts[a], parts[b]];
                let r = (bound[0][a * n + b].hypot(bound[0][b * n + a])) / gap(a, b).max(0.0);
                match pairs.iter_mut().find(|p| p.0 == key) {
                    Some(p) => p.1 = larger(p.1, r),
                    None => pairs.push((key, r)),
                }
            }
        }
        if pairs.is_empty() {
            return;
        }
        pairs.sort_by(|x, y| x.1.total_cmp(&y.1));
        let values: Vec<f64> = self.read(|b| self.template.generator.iter().map(|(beat, _)| b.value[*beat]).collect());
        // The step the run's ladder starts at or below (solve.rs `prior_step`) and the work a µs there as it stands.
        let tau = crate::solve::prior_step(self);
        let base = (self.judgment.dropped, self.judgment.once);
        // The steps are priced where a cut's factor leaves room (W unbounded else, whatever its work), the run as it
        // stands first.
        let mut unturned: Option<f64> = None;
        let mut least: (f64, Option<(Vec<Vec<usize>>, Vec<(usize, usize)>, f64)>) = (f64::INFINITY, None);
        crate::prof::note(|| format!("turn: the parts' pairs (pair, ‖V‖/μ) {pairs:?}; at τ {tau:.4} µs"));
        let mut last = self.blocks.clone();
        for k in 1..=pairs.len() {
            let apart = &pairs[..k];
            let blocks = self.cut_blocks(&parts, &|x, y| apart.iter().any(|p| (p.0[0] == x && p.0[1] == y) || (p.0[0] == y && p.0[1] == x)));
            if blocks == last {
                continue;
            }
            last.clone_from(&blocks);
            let mut of = vec![0; n];
            for (i, b) in blocks.iter().enumerate() {
                for &x in b {
                    of[x] = i;
                }
            }
            let cross: Vec<(usize, usize)> = (0..n).flat_map(|a| (0..n).map(move |b| (a, b))).filter(|&(a, b)| self.block_of[a] == self.block_of[b] && of[a] != of[b]).collect();
            let frob = |m: &[f64]| cross.iter().map(|&(a, b)| m[a * n + b] * m[a * n + b]).sum::<f64>().sqrt();
            let mu = cross.iter().map(|&(a, b)| gap(a, b)).fold(f64::INFINITY, smaller);
            let v = frob(&bound[0]);
            let s = cross.iter().map(|&(a, b)| (bound[0][a * n + b] / gap(a, b).max(0.0)).powi(2)).sum::<f64>().sqrt();
            let off = blocks
                .iter()
                .map(|b| b.iter().flat_map(|&x| b.iter().map(move |&y| (x, y))).filter(|(x, y)| x != y).map(|(x, y)| bound[0][x * n + y] * bound[0][x * n + y]).sum::<f64>().sqrt())
                .fold(0.0, larger);
            // The remainder as the nodes book it (`turned`): between the parts by parts, within them on the rate; the
            // motion's coupling at the request's thermal speed.
            let mut between = 2.0 * s * s * v + v * (2.0 * off / mu).powi(ROUNDS as i32 + 1);
            let mut within = 2.0 * s * s * s * v + 4.0 * s * s * lambda;
            let mut once = 2.0 * s;
            for (slot, w) in values.iter().enumerate().skip(1) {
                let vk = frob(&bound[slot]);
                if vk == 0.0 {
                    continue;
                }
                let sk = vk / (mu - w.abs()).max(0.0);
                between += 2.0 * sk * (v + vk);
                within += 4.0 * sk * lambda;
                once += 2.0 * sk;
            }
            // The frame's gradient at the structure's point (‖∇S‖ with the gap's own), on T and the motion.
            let gradient = cross.iter().map(|&(a, b)| ((slope(a, b) + bound[0][a * n + b] / gap(a, b).max(0.0) * gap_slope(a, b)) / gap(a, b).max(0.0)).powi(2)).sum::<f64>().sqrt();
            let velocity = self.hbar_over_mass * gradient;
            between += (crate::units::KT_PER_UK * self.scene.request.temperature_uk * self.hbar_over_mass).sqrt() * gradient;
            once += 2.0 * between / mu;
            let rate = within + between * 2.0 * (off + lambda) / mu;
            let factor = account.factor_with(base.0 + rate, base.1 + once, velocity);
            #[cfg(test)]
            let factor = if FORCE_TURN.with(|f| f.get()) { 1.0 } else { factor };
            if !factor.is_finite() {
                crate::prof::note(|| format!("turn: the first {k} pairs apart: blocks {:?}, ‖V‖ {v:.3e}, μ {mu:.4e}, ‖S‖ {s:.3e}, ‖∇S‖ {gradient:.3e} /µm; Y {rate:.3e} /µs, once {once:.3e}: W unbounded", blocks.iter().map(Vec::len).collect::<Vec<_>>()));
                continue;
            }
            if unturned.is_none() {
                let plain = self.priced_work(tau);
                let w = plain * account.factor_with(base.0, base.1, 0.0);
                crate::prof::note(|| format!("turn: none: work {plain:.4e} a µs, W {w:.4e}"));
                (unturned, least.0) = (Some(w), w);
            }
            // The cut applied, its steps priced, and taken off again.
            let kept = (self.template.classes.clone(), self.blocks.clone(), self.block_of.clone());
            self.apply_cut(&blocks, &cross);
            let work = self.priced_work(tau);
            (self.template.classes, self.blocks, self.block_of) = kept;
            self.turn = None;
            POOL.with(|p| p.borrow_mut().retain(|x| x.0 != self.id));
            let w = work * factor;
            crate::prof::note(|| format!("turn: the first {k} pairs apart: blocks {:?}, ‖V‖ {v:.3e}, μ {mu:.4e}, ‖S‖ {s:.3e}, ‖∇S‖ {gradient:.3e} /µm, off {off:.3e}; Y {rate:.3e} /µs, once {once:.3e}; work {work:.4e} a µs, W {w:.4e}", blocks.iter().map(Vec::len).collect::<Vec<_>>()));
            #[cfg(test)]
            let w = if FORCE_TURN.with(|f| f.get()) { 0.0 } else { w };
            if w < least.0 {
                least = (w, Some((blocks, cross, work)));
            }
        }
        let (Some((blocks, cross, work)), Some(unturned)) = (least.1, unturned) else { return };
        self.apply_cut(&blocks, &cross);
        self.judgment.blocks = blocks;
        self.turn_check = Some(TurnCheck { account, base, unturned, work });
        crate::prof::note(|| format!("turn: turned, W {:.4e} against {unturned:.4e} unturned", least.0));
    }

    /// A cut of K's fast parts applied (`turn_parts`): the layout's jump places closed under S -- (x, a) where (y, a)
    /// is, (c, y) where (c, x) is, x and y apart --, the blocks, the places apart turned; the nodes and sizes made on the
    /// layout before let go.
    fn apply_cut(&mut self, blocks: &[Vec<usize>], cross: &[(usize, usize)]) {
        let n = self.rs.keep.len();
        // The layout's jump places closed under S: (x, a) where (y, a) is, (c, y) where (c, x) is, x and y apart.
        // Each component's terms of [S, L] (`turned`): per place apart (x, y) of S, an element (y, a) of L adds S_xy L_ya at
        // (x, a) and an element (a, x) takes L_ax S_xy off (a, y) -- the places closed (the places added hold nothing of
        // L: no output fills them).
        self.turn_terms.clear();
        for class in self.template.classes.iter_mut() {
            for axis in class.axes.iter_mut() {
                for jump in axis.iter_mut() {
                    let mut have: Vec<bool> = vec![false; n * n];
                    for e in &jump.elements {
                        have[e.at] = true;
                    }
                    let mut more = Vec::new();
                    for &(x, y) in cross {
                        for a in 0..n {
                            if have[y * n + a] && !have[x * n + a] && !more.contains(&(x * n + a)) {
                                more.push(x * n + a);
                            }
                            if have[a * n + x] && !have[a * n + y] && !more.contains(&(a * n + y)) {
                                more.push(a * n + y);
                            }
                        }
                    }
                    let original = jump.elements.len();
                    jump.elements.extend(more.into_iter().map(|at| Element { at, value: C64::ZERO, gradient: [C64::ZERO; 3], kick: C64::ZERO }));
                    let index: HashMap<usize, u32> = jump.elements.iter().enumerate().map(|(i, e)| (e.at, i as u32)).collect();
                    let mut terms = Vec::new();
                    for &(x, y) in cross {
                        let place = (x * n + y) as u32;
                        for (i, e) in jump.elements[..original].iter().enumerate() {
                            let (row, col) = (e.at / n, e.at % n);
                            if row == y {
                                terms.push((index[&(x * n + col)], i as u32, place, false));
                            }
                            if col == x {
                                terms.push((index[&(row * n + y)], i as u32, place, true));
                            }
                        }
                    }
                    self.turn_terms.push(terms);
                }
            }
        }
        for (i, b) in blocks.iter().enumerate() {
            for &x in b {
                self.block_of[x] = i;
            }
        }
        self.blocks = blocks.to_vec();
        self.turn = Some(cross.to_vec());
        self.seen = Arc::new(Seen::default());
        POOL.with(|p| p.borrow_mut().retain(|x| x.0 != self.id));
    }

    /// The exact blocks cut by K's fast parts (# The parts turned): within each block the parts joined but where
    /// `apart` says so; each block's places sorted, the blocks in the order of their first place.
    fn cut_blocks(&self, parts: &[[i64; 2]], apart: &dyn Fn([i64; 2], [i64; 2]) -> bool) -> Vec<Vec<usize>> {
        let n = parts.len();
        let mut root: Vec<usize> = (0..n).collect();
        fn find(root: &mut [usize], x: usize) -> usize {
            let mut x = x;
            while root[x] != x {
                root[x] = root[root[x]];
                x = root[x];
            }
            x
        }
        for a in 0..n {
            for b in 0..n {
                if self.block_of[a] == self.block_of[b] && !apart(parts[a], parts[b]) {
                    let (ra, rb) = (find(&mut root, a), find(&mut root, b));
                    root[ra.max(rb)] = ra.min(rb);
                }
            }
        }
        let (mut blocks, mut of): (Vec<Vec<usize>>, Vec<usize>) = (Vec::new(), Vec::new());
        for a in 0..n {
            let r = find(&mut root, a);
            match of.iter().position(|&x| x == r) {
                Some(i) => blocks[i].push(a),
                None => {
                    of.push(r);
                    blocks.push(vec![a]);
                }
            }
        }
        blocks
    }

    /// A node turned (module documentation, # The parts turned): S between the turned blocks from its static slot (the
    /// frame's, θ on it), its diagonal d first and `ROUNDS` of the blocks' own couplings; within each block the static
    /// slot plus ½ Σ_b (S_ab V_bc − V_ab S_bc) and each opposite pair of beats' ½(S_k V_−k − V_k S_−k), S_k = V_k/(d_a
    /// − d_b − ω_k); the static gradient slot plus the same of ∇V and ∇S = ∇V/(d_a − d_b); every slot's elements between
    /// the blocks left out; each jump component to L + [S, L], its photon form likewise, its gradient + [S, ∇L] + [∇S, L].
    /// The frame S of a node's static slot `g` between the parts apart `cross` (# The parts turned): [S, G_d] = −V, its
    /// diagonal first, then `ROUNDS` of the blocks' own couplings; with its counted complex multiply-adds.
    fn frame_of(&self, g: &Mat, cross: &[(usize, usize)]) -> (Vec<C64>, f64) {
        let n = self.rs.keep.len();
        let (of, blocks) = (&self.block_of, &self.blocks);
        let (moved, square) = (MOVED / COMPLEX, (n * n) as f64);
        let within = |a: usize| blocks[of[a]].len() as f64;
        let gap = |a: usize, b: usize| g[(a, a)] - g[(b, b)];
        let mut work = moved * square + 2.0 * cross.len() as f64;
        let mut s = vec![C64::ZERO; n * n];
        for &(a, b) in cross {
            s[a * n + b] = g[(a, b)] / gap(a, b);
        }
        for _ in 0..ROUNDS {
            let prev = s.clone();
            work += moved * square + cross.iter().map(|&(a, b)| within(a) + within(b) + 2.0).sum::<f64>();
            for &(a, b) in cross {
                let mut x = g[(a, b)];
                for &q in blocks[of[b]].iter().filter(|&&q| q != b) {
                    x += prev[a * n + q] * g[(q, b)];
                }
                for &q in blocks[of[a]].iter().filter(|&&q| q != a) {
                    x -= g[(a, q)] * prev[q * n + b];
                }
                s[a * n + b] = x / gap(a, b);
            }
        }
        (s, work)
    }

    /// A turned run's node at a point unturned, and its frame S there (the tests': `frame_of` on its static slot), none
    /// where the run turns nothing.
    #[cfg(test)]
    pub(crate) fn unturned_at(&self, at: V3, kappa: V3) -> Option<(Node, Vec<C64>)> {
        let cross = self.turn.as_ref()?;
        let mut work = Work::default();
        let mut node = self.template.clone();
        let making = Making::node(false);
        self.scene.point_into(at, kappa, &mut work.point);
        self.rs.evaluate(&self.scene, &work.point, &mut work.local, making);
        work.out.contract(self.rs.plan(), &self.lab, &work.local, None, making);
        self.fill(&mut node, &work.out, (false, true), &mut work.jumps);
        let (s, _) = self.frame_of(&node.generator[0].1, cross);
        Some((node, s))
    }

    fn turned(&self, node: &mut Node, cross: &[(usize, usize)]) -> f64 {
        let n = self.rs.keep.len();
        let (of, blocks) = (&self.block_of, &self.blocks);
        // The counted complex multiply-adds (decide.rs, # Work): a complex division two, a matrix of n² made or copied
        // MOVED a number; per place apart its share of each pass below.
        let (moved, square, places) = (MOVED / COMPLEX, (n * n) as f64, cross.len() as f64);
        let within = |a: usize| blocks[of[a]].len() as f64;
        let g = node.generator[0].1.clone();
        let gap = |a: usize, b: usize| g[(a, a)] - g[(b, b)];
        let (s, made) = self.frame_of(&g, cross);
        let mut work = moved * square + made;
        // ½[S, V] within the blocks, of S and V given as dense n × n (the places apart alone set).
        let half = |s: &[C64], v: &Mat, out: &mut Mat| {
            for &(a, b) in cross {
                for &c in &blocks[of[a]] {
                    out[(a, c)] += (s[a * n + b] * v[(b, c)] - v[(a, b)] * s[b * n + c]) * 0.5;
                }
            }
        };
        let mut delta = Mat::zeros(n, n);
        half(&s, &g, &mut delta);
        // A half commutator: per place apart and state of its block two products added.
        let halving = cross.iter().map(|&(a, _)| 2.0 * within(a)).sum::<f64>();
        work += moved * square + halving;
        let values: Vec<f64> = self.read(|b| node.generator.iter().map(|(beat, _)| b.value[*beat]).collect());
        for (k, (_, m)) in node.generator.iter().enumerate().skip(1) {
            if let Some(j) = values.iter().position(|&w| w == -values[k]) {
                let opposite = &node.generator[j].1;
                let mut sk = vec![C64::ZERO; n * n];
                for &(a, b) in cross {
                    sk[a * n + b] = m[(a, b)] / (gap(a, b) - C64::real(values[k]));
                }
                half(&sk, opposite, &mut delta);
                work += moved * square + 2.0 * places + halving;
            }
        }
        // The gradient: ∇S = (∇V − S ∇(d_a − d_b))/(d_a − d_b) (the gap's own gradient, the light shifts' across the
        // atom's place), ½([∇S, V] + [S, ∇V]) within the blocks.
        let mut ds: [Vec<C64>; 3] = std::array::from_fn(|_| vec![C64::ZERO; n * n]);
        if let Some((_, dg)) = node.gradient.iter_mut().find(|g| g.0 == 0) {
            for i in 0..3 {
                for &(a, b) in cross {
                    ds[i][a * n + b] = (dg[i][(a, b)] - s[a * n + b] * (dg[i][(a, a)] - dg[i][(b, b)])) / gap(a, b);
                }
                let mut dd = Mat::zeros(n, n);
                half(&ds[i], &g, &mut dd);
                half(&s, &dg[i], &mut dd);
                dg[i].add_scaled(C64::ONE, &dd);
                work += 3.0 * moved * square + 2.0 * places + 2.0 * halving;
            }
        }
        // The node's own sizes (`Seen`; # The parts turned), read before the elements apart are taken out: ‖S‖, ‖V‖ over
        // the places apart, the blocks' couplings, the largest decay ‖Λ‖ of Λ = Σ_c L_c†L_c (`decay_bound`), each opposite
        // beat's S_k and the frame's gradient.
        {
            let norm = |m: &dyn Fn(usize, usize) -> f64| cross.iter().map(|&(a, b)| m(a, b).powi(2)).sum::<f64>().sqrt();
            let (sn, vn) = (norm(&|a, b| s[a * n + b].abs()), norm(&|a, b| g[(a, b)].abs()));
            let mu = cross.iter().map(|&(a, b)| gap(a, b).abs()).fold(f64::INFINITY, smaller);
            let off = blocks.iter().map(|b| b.iter().flat_map(|&x| b.iter().map(move |&y| (x, y))).filter(|(x, y)| x != y).map(|(x, y)| g[(x, y)].norm_sqr()).sum::<f64>().sqrt()).fold(0.0, larger);
            let lambda = decay_bound(node);
            let gap_gradient = |a: usize, b: usize| -> f64 {
                node.gradient.iter().find(|g| g.0 == 0).map_or(0.0, |(_, dg)| (0..3).map(|i| (dg[i][(a, a)] - dg[i][(b, b)]).norm_sqr()).sum::<f64>().sqrt())
            };
            let mut gradient = (0..3).map(|i| cross.iter().map(|&(a, b)| ds[i][a * n + b].norm_sqr()).sum::<f64>()).sum::<f64>().sqrt();
            // The remainder of the turn, two kinds (module documentation, # The parts turned): what lies between the parts
            // -- the third order 2‖S‖²‖V‖, the rounds' residual, each opposite beat's 2‖S_k‖(‖V‖ + ‖V_k‖) and the motion's
            // coupling a = |v|‖∇S‖ -- turns at their gap and is booked by parts (once its size over μ, on the rate its
            // size times the rest's rate 2(‖G_off‖ + ‖Λ‖) over μ); what lies within them grows: the fourth order
            // 2‖S‖³‖V‖, the channels' second order 4‖S‖²‖Λ‖ and each opposite beat's channels 4‖S_k‖‖Λ‖ (its S_k not
            // turned into them).
            let mut frame = sn;
            let mut between = 2.0 * sn * sn * vn + vn * (2.0 * off / mu).powi(ROUNDS as i32 + 1);
            let mut within = 2.0 * sn * sn * sn * vn + 4.0 * sn * sn * lambda;
            let beats: Vec<usize> = node.generator.iter().map(|g| g.0).collect();
            for (k, (_, m)) in node.generator.iter().enumerate().skip(1) {
                let vk = norm(&|a, b| m[(a, b)].abs());
                if vk != 0.0 {
                    let distance = (mu - values[k].abs()).max(f64::MIN_POSITIVE);
                    let sk = vk / distance;
                    frame += sk;
                    between += 2.0 * sk * (vn + vk);
                    within += 4.0 * sk * lambda;
                    let dvk = node.gradient.iter().find(|g| g.0 == beats[k]).map_or(0.0, |(_, dg)| (0..3).map(|i| cross.iter().map(|&(a, b)| dg[i][(a, b)].norm_sqr()).sum::<f64>()).sum::<f64>().sqrt());
                    let shifted = cross.iter().map(|&(a, b)| (m[(a, b)].abs() * gap_gradient(a, b)).powi(2)).sum::<f64>().sqrt() / distance;
                    gradient += (dvk + shifted) / distance;
                }
            }
            between += self.hbar_over_mass * (node.kappa.iter().map(|k| k * k).sum::<f64>()).sqrt() * gradient;
            frame += between / mu;
            let rate = within + between * 2.0 * (off + lambda) / mu;
            self.seen.note(frame, rate, gradient);
            // The norms: per place apart and slot a modulus and its square; the blocks' couplings; Λ's rows.
            work += 2.0 * places * node.generator.len() as f64 + 2.0 * square;
        }
        for (_, gs) in node.gradient.iter_mut() {
            for m in gs.iter_mut() {
                for &(a, b) in cross {
                    m[(a, b)] = C64::ZERO;
                }
            }
        }
        for (_, m) in node.generator.iter_mut() {
            for &(a, b) in cross {
                m[(a, b)] = C64::ZERO;
            }
        }
        work += moved * places * (node.generator.len() + 3 * node.gradient.len()) as f64;
        node.generator[0].1.add_scaled(C64::ONE, &delta);
        work += square;
        // The jump components: L + [S, L], [S, L]_xa = Σ_y S_xy L_ya − L_xy S_ya over the places apart, its gradient
        // [∇S, L] + [S, ∇L], by each component's own terms (`turn_terms`: a term an element of L and a place of S, made
        // once): of its five numbers the elements as they stood, each term's eight products.
        let mut component = 0;
        let mut before: Vec<[C64; 5]> = Vec::new();
        for class in node.classes.iter_mut() {
            for axis in class.axes.iter_mut() {
                for jump in axis.iter_mut() {
                    let terms = &self.turn_terms[component];
                    component += 1;
                    if terms.is_empty() {
                        continue;
                    }
                    before.clear();
                    before.extend(jump.elements.iter().map(|e| [e.value, e.kick, e.gradient[0], e.gradient[1], e.gradient[2]]));
                    for &(to, from, place, minus) in terms {
                        let (l, p) = (before[from as usize], place as usize);
                        let sign = if minus { -1.0 } else { 1.0 };
                        let (sxy, dxy) = (s[p] * sign, [ds[0][p] * sign, ds[1][p] * sign, ds[2][p] * sign]);
                        let e = &mut jump.elements[to as usize];
                        e.value += sxy * l[0];
                        e.kick += sxy * l[1];
                        for i in 0..3 {
                            e.gradient[i] += sxy * l[2 + i] + dxy[i] * l[0];
                        }
                    }
                    work += 5.0 * moved * jump.elements.len() as f64 + 8.0 * terms.len() as f64;
                }
            }
        }
        work
    }

    /// The direct entries spanned (module documentation, # The direct entries spanned): of the lab's grouping's spans
    /// at the tolerances 10⁻ᵏ of each kind's largest atom (k = 2 … 16) and of none, the one of least W -- the drop
    /// tier's rule, the node's counted work (complex multiply-adds, three nodes a step, the direct part as each does it:
    /// value and gradients) times the account's factor at the dropped beats' Y and the span's (`Account::factor`) --,
    /// set on the grouping. The span's Y: what it leaves out of each output bounded (`Selected::rate`). Returns it.
    fn span_directs(&mut self, account: &Account, nodes: f64, base: f64) -> f64 {
        let (_, coefficients, norms) = self.reach_bounds();
        let plan = self.rs.plan();
        let rate = |spanned: &crate::model::rs::fold::Spanned| -> f64 {
            let (mut h, mut channels) = (vec![0.0; self.lab.outputs.len()], vec![0.0; self.lab.channel_outputs.len()]);
            for &(is_h, o, d) in &spanned.dropped {
                if is_h {
                    h[o] += d;
                } else {
                    channels[o] += d;
                }
            }
            self.rate(&h, &channels, &norms)
        };
        // The candidates: none (the entries one by one, nothing left out), then the spans at 10⁻ᵏ.
        let plain = self.lab.spanned(plan, &coefficients, 0.0).work[0];
        let mut least = (nodes + 3.0 * plain) * account.factor(base);
        crate::prof::note(|| format!("span none: work {plain} W {least:.6e} (nodes {nodes:.0}, factor {:.6e})", account.factor(base)));
        let mut best: Option<(f64, crate::model::rs::fold::Spanned)> = None;
        for k in 2..=16 {
            let spanned = self.lab.spanned(plan, &coefficients, 10f64.powi(-k));
            let y = rate(&spanned);
            let w = (nodes + 3.0 * spanned.work[1]) * account.factor(base + y);
            crate::prof::note(|| format!("span 1e-{k}: directions {} work {} Y {y:.3e} W {w:.6e} (factor {:.6e})", spanned.spans.iter().map(|d| d.basis.len()).sum::<usize>(), spanned.work[1], account.factor(base + y)));
            if w < least {
                least = w;
                best = Some((y, spanned));
            }
        }
        match best {
            Some((y, spanned)) => {
                self.lab.spans = spanned.spans;
                y
            }
            None => {
                self.lab.spans.clear();
                0.0
            }
        }
    }

    /// K's sideband parts left out (module documentation, # The sideband parts dropped): each part's Y, its pairs' bounds
    /// over the reach (`Out::bounds_of`, the part's pairs alone) booked by `rate`; the parts in the order of their Y, of
    /// none and every first few the one of least W (`counted_node` without them times `Account::factor` at `base` plus
    /// their Y), left out of the fold (`Plan::drop_sideband`). Returns their Y.
    fn drop_sidebands(&mut self, account: &Account, base: f64) -> f64 {
        let sidebands = self.rs.plan().sidebands();
        if sidebands.is_empty() {
            return 0.0;
        }
        let (moduli, coefficients, norms) = self.reach_bounds();
        let frobenius = |b: &Vec<f64>| b.iter().map(|x| x * x).sum::<f64>().sqrt();
        let mut parts: Vec<(usize, f64)> = sidebands
            .iter()
            .map(|&b| {
                let (h, channels) = Out::bounds_of(self.rs.plan(), &self.lab, &moduli, &coefficients, &|x| x == Some(b));
                (b, self.rate(&h.iter().map(frobenius).collect::<Vec<_>>(), &channels.iter().map(frobenius).collect::<Vec<_>>(), &norms))
            })
            .collect();
        parts.sort_by(|a, b| a.1.total_cmp(&b.1));
        let sizes = self.rs.plan().blocks();
        let mut least = (0, self.counted_node() * account.factor(base));
        crate::prof::note(|| format!("sidebands none: W {:.6e}; parts (block, size, Y) {:?}", least.1, parts.iter().map(|&(b, y)| (b, sizes[b].0, y)).collect::<Vec<_>>()));
        let mut y = 0.0;
        for k in 0..parts.len() {
            self.rs.folding.plan.drop_sideband(parts[k].0, true);
            y += parts[k].1;
            let w = self.counted_node() * account.factor(base + y);
            crate::prof::note(|| format!("sidebands the first {}: Y {y:.3e} W {w:.6e} (factor {:.6e})", k + 1, account.factor(base + y)));
            if w < least.1 {
                least = (k + 1, w);
            }
        }
        for (k, &(b, _)) in parts.iter().enumerate() {
            self.rs.folding.plan.drop_sideband(b, k < least.0);
        }
        parts[..least.0].iter().map(|p| p.1).sum()
    }

    /// The three nodes' counted work of a step at the structure's point (complex multiply-adds): a node's own
    /// (`Node::work`, `node_work`).
    fn counted_node(&self) -> f64 {
        3.0 * self.node([0.0; 3], [0.0; 3], false).work / COMPLEX
    }

    /// A node's counted work (real multiply-adds; decide.rs, # Work), `Node::work`: the point (its beams' amplitudes,
    /// `point_work`) and the source's values there (`Tweezer::values_work`), the fold's numbers and their contraction
    /// (`Local::work`, `Out::work`), and `rest` -- the filling and the turn (`fill`, `turned`; complex multiply-adds). The
    /// one count of a node: a trajectory's, a path's and the judgment's.
    fn node_work(&self, local: &Local, out: &Out, rest: f64) -> f64 {
        COMPLEX * (local.work + out.work + rest) + self.point_work() + COMPLEX * self.rs.tweezer().values_work()
    }

    /// The point's counted work (real multiply-adds; `Scene::point_into`): per beam its amplitude and gradient -- a plane
    /// wave's cis and four products, a Gaussian beam's cis, exp and reciprocal with its twenty products --, per generator
    /// its rate (two dot products and their sum).
    fn point_work(&self) -> f64 {
        let beams: f64 = self.scene.geometry.iter().map(|g| if g.waist.is_some() { 2.0 * ELEMENTARY + DIVISION + 20.0 * COMPLEX } else { ELEMENTARY + 4.0 * COMPLEX }).sum();
        beams + 8.0 * self.scene.generators.len() as f64
    }

    /// The reach's bounds the judgment's later parts weigh (# The direct entries spanned): the folded blocks' amplitudes
    /// over the reach (`Plan::reach`, the structure's point's frequencies), the reach's coefficients' largest, and per
    /// channel output its channel's ‖L‖ bounded -- the channel (emission axis, fast index) its value outputs' Frobenius
    /// bounds summed over its beats (rs/fold.rs `Out::bounds`).
    fn reach_bounds(&self) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let reach = self.reach();
        let mut work = Work::default();
        self.scene.point_into([0.0; 3], [0.0; 3], &mut work.point);
        self.rs.evaluate(&self.scene, &work.point, &mut work.local, Making::node(false));
        let plan = self.rs.plan();
        let moduli = plan.reach(&work.local.values, &reach);
        let (_, channels) = Out::bounds(plan, &self.lab, &moduli, &reach.coefficients);
        let outputs = &self.lab.channel_outputs;
        let mut whole: HashMap<(usize, [i64; 2]), f64> = HashMap::new();
        for (o, (q, fast, which, ..)) in outputs.iter().enumerate() {
            if *which == 0 {
                *whole.entry((*q, *fast)).or_insert(0.0) += channels[o].iter().map(|x| x * x).sum::<f64>().sqrt();
            }
        }
        let norms = outputs.iter().map(|(q, fast, ..)| whole.get(&(*q, *fast)).copied().unwrap_or(0.0)).collect();
        (moduli, reach.coefficients, norms)
    }

    /// The run's reach (rs.rs `Rs::reach`): the momentum of an atom at the escape energy at the focus, √(2 U₀ m)/ħ with
    /// U₀ the deepest well of the trap's potential.
    fn reach(&self) -> crate::model::rs::fold::Reach {
        let wells = self.scene.trap_potential([0.0; 3], &self.depth);
        let depth = wells.iter().map(|q| -q.value).fold(0.0, larger);
        self.rs.reach(&self.scene, self.hbar_over_mass, (2.0 * depth / self.hbar_over_mass).sqrt())
    }

    /// The rate Y of a change of the node's outputs by at most `h` (per H output, a bound of its Frobenius norm) and
    /// `channels` (per channel output, likewise), the drop tier's (model/classify.rs, # The drop tier:
    /// ‖(𝓛′ − 𝓛)ρ‖₁ ≤ 2Y‖ρ‖₁): ‖δH_eff‖ at most Σ h over the outputs (one a beat); per channel (emission axis, fast index)
    /// δ its value outputs' sum and ‖L‖ its bound (`reach_bounds`), LρL† and ½{L†L, ρ} each moved by at most
    /// (2‖L‖δ + δ²)‖ρ‖₁. The photon forms' outputs add none: what a jump reads off them is the state's, which D bounds.
    fn rate(&self, h: &[f64], channels: &[f64], norms: &[f64]) -> f64 {
        let outputs = &self.lab.channel_outputs;
        let mut per: HashMap<(usize, [i64; 2]), (f64, f64)> = HashMap::new();
        for (o, (q, fast, which, ..)) in outputs.iter().enumerate() {
            if *which == 0 && channels[o] > 0.0 {
                let e = per.entry((*q, *fast)).or_insert((0.0, norms[o]));
                e.0 += channels[o];
            }
        }
        h.iter().sum::<f64>() + per.values().map(|&(d, l)| 2.0 * l * d + d * d).sum::<f64>()
    }

    /// The run's judgment of its node's beats (model/classify.rs): per slot of the structure's node, before the beats
    /// left out were taken from the layout, its beat and tier; the blocks; what the dropped beats cost.
    #[cfg(test)]
    pub fn judgment(&self) -> &Judgment {
        &self.judgment
    }

    /// Y (1/µs): the bound of the rate at which the model's dropped terms move the run's state in trace distance
    /// (model/classify.rs, # The drop tier) -- after a time t the distance is at most ½(e^{2Yt} − 1) = tY(1 + O(tY)), a row
    /// as shown and the loss move by at most that, T by at most c_T times it of its allowance. Zero where nothing is
    /// dropped. The run's error account takes it (the solve's).
    pub fn dropped(&self) -> f64 {
        self.judgment.dropped
    }

    /// The trace distance booked once beside the rate's (`Judgment::once`: the frame of the parts turned, # The parts
    /// turned), zero where no part is turned.
    pub fn dropped_once(&self) -> f64 {
        self.judgment.once
    }

    /// The reference state's ground populations the run's splitting was made at (`Selected::new`).
    pub(crate) fn reference_populations(&self) -> [f64; GROUND] {
        self.reference
    }

    /// Whether a turned run's turn still pays on the nodes it has made (module documentation, # The parts turned): the
    /// decision's W again (`turn_parts`) at the largest frame, remainder and frame's gradient the nodes noted (`Seen`) in
    /// place of the structure's estimate -- the turned run's step work a µs times the account's factor there -- against
    /// the unturned run's. True where nothing is turned or no node has been made; false where the frame or the
    /// remainder take the room the targets leave (the factor unbounded) or the turn no longer costs the less.
    pub fn turn_holds(&self) -> bool {
        let Some(check) = &self.turn_check else { return true };
        let (frame, rate, gradient) = self.seen.get();
        if frame == 0.0 && rate == 0.0 {
            return true;
        }
        let w = check.work * check.account.factor_with(check.base.0 + rate, check.base.1 + 2.0 * frame, self.hbar_over_mass * gradient);
        crate::prof::note(|| format!("turn: the pilot's nodes: frame {frame:.3e}, Y {rate:.3e} /µs, ‖∇S‖ {gradient:.3e} /µm: W {w:.4e} against {:.4e} unturned", check.unturned));
        w < check.unturned
    }

    /// A step's counted work a µs as the run takes it (`step::priced`, the work its stretches count, `Price::work`) at the
    /// step τ from the structure's point, the chief state at the thermal speed along each axis's sign of three starts
    /// (no jump: the threshold none); their mean. What the turn's decision weighs (`turn_parts`); unbounded where no
    /// step is taken.
    fn priced_work(&self, tau: f64) -> f64 {
        struct Anywhere;
        impl crate::step::Domain for Anywhere {
            fn runaway(&self, _: f64) -> bool {
                false
            }
        }
        let request = &self.scene.request;
        let speed = (crate::units::KT_PER_UK * request.temperature_uk / self.hbar_over_mass).sqrt();
        let chief = crate::solve::chief(request);
        let (mut total, mut taken) = (0.0, 0);
        for i in 0..3u64 {
            let sign = |b: u64| if (i >> b) & 1 == 1 { -1.0 } else { 1.0 };
            let mut from = crate::step::Trajectory::new(self, chief, [0.0; 3], [sign(0) * speed, sign(1) * speed, sign(2) * speed], 7, i);
            from.threshold = 0.0;
            if let Ok(prices) = crate::step::priced(self, &from, tau, &Anywhere) {
                total += prices.iter().map(|(_, p)| p.work(p.costs)).sum::<f64>();
                taken += 1;
            }
        }
        if taken == 0 { f64::INFINITY } else { total / (taken as f64 * tau) }
    }

    /// The turned nodes' sizes (`Seen`) the run books by, none where no part is turned.
    pub fn turn_seen(&self) -> Option<Arc<Seen>> {
        self.turn.as_ref().map(|_| self.seen.clone())
    }

    /// The candidate pairs of the drop tier with their Y (model/classify.rs, # The drop tier), the drop's account, and
    /// the three nodes' counted work of a step (complex multiply-adds): per candidate slot of the structure's node
    /// (`candidates`, before the layout drops anything) its pair -- it and the slot of its opposite beat -- and the sum
    /// of their sup norms' bounds over the reach, the H_eff part's √(‖S‖₁‖S‖_∞) of its entries' bounds S
    /// (rs/fold.rs `Out::bounds` on the lab's grouping) and the decay's share ½ Σ ‖ℓ_l‖‖ℓ_r‖ over its products, each
    /// spherical component's norm the Frobenius of its entries' bounds (the spherical fold's own, `spherical`); the
    /// reach's κ from the prepared well's depth U₀ (`Scene::trap_potential` at the focus: K ≤ s U₀ for a held atom),
    /// the account's hold of the held atoms by the same U₀ and T's scale at the emission's shortest wave number k
    /// (T_sc ≥ 30 k_BT_rec, estimate.rs `semiclassical`), its measured errors none (the model's own error is measured
    /// after the pilot, on the run's paths: select.rs).
    fn drop_sizes(&self, node: &Node, candidates: &[usize], spherical: (&Plan, &Grouping)) -> (Vec<(Vec<usize>, f64)>, Account, f64) {
        let n = self.rs.keep.len();
        let wells = self.scene.trap_potential([0.0; 3], &self.depth);
        let depth = wells.iter().map(|q| -q.value).fold(0.0, larger);
        let kappa = (2.0 * depth / self.hbar_over_mass).sqrt();
        let reach = self.rs.reach(&self.scene, self.hbar_over_mass, kappa);
        let mut work = Work::default();
        self.scene.point_into([0.0; 3], [0.0; 3], &mut work.point);
        self.rs.evaluate(&self.scene, &work.point, &mut work.local, Making::node(false));
        let nodes = self.counted_node();
        let values = &work.local.values;
        // The H_eff part per slot.
        let (h, _) = Out::bounds(self.rs.plan(), &self.lab, &self.rs.plan().reach(values, &reach), &reach.coefficients);
        let slots = node.generator.len();
        let mut parts = vec![vec![0.0_f64; n * n]; slots];
        for (list, b) in self.tables.h.iter().zip(&h) {
            for (k, &(x, g, _)) in list.iter().enumerate() {
                if g != u32::MAX {
                    parts[g as usize][x as usize] += b[k];
                }
            }
        }
        let norm = |m: &[f64]| -> f64 {
            let rows = (0..n).map(|a| (0..n).map(|b| m[a * n + b]).sum::<f64>()).fold(0.0, larger);
            let cols = (0..n).map(|b| (0..n).map(|a| m[a * n + b]).sum::<f64>()).fold(0.0, larger);
            (rows * cols).sqrt()
        };
        let mut size: Vec<f64> = parts.iter().map(|m| norm(m)).collect();
        // The decay's share per slot: each spherical component's entries' bounds (the spherical fold's channel outputs
        // at the component's class, axis and beat), its Frobenius norm, the products'.
        let (plan, grouping) = spherical;
        let (_, channels) = Out::bounds(plan, grouping, &plan.reach(values, &reach), &reach.coefficients);
        let mut entries: HashMap<(usize, u32), f64> = HashMap::new();
        for (co, (q, fast, which, key, _)) in grouping.channel_outputs.iter().enumerate() {
            let Some(class) = self.template.classes.iter().position(|c| c.fast == *fast).filter(|_| *which == 0) else { continue };
            let beats = self.elements(key, Of::Jump);
            for (k, &x) in grouping.channel_lists[co].iter().enumerate() {
                if let Some(j) = self.tables.jumps.iter().position(|j| j.class == class && j.q == *q && j.beat == beats[x as usize]) {
                    *entries.entry((j, x)).or_insert(0.0) += channels[co][k];
                }
            }
        }
        let mut components = vec![0.0_f64; self.tables.jumps.len()];
        for (&(j, _), v) in &entries {
            components[j] += v * v;
        }
        for d in &self.tables.decay {
            let v = 0.5 * components[d.left].sqrt() * components[d.right].sqrt();
            size[d.forward] += v;
            if d.left != d.right {
                size[d.back] += v;
            }
        }
        // The pairs: a slot and the slot of its opposite beat.
        let exact: Vec<Exact> = self.read(|b| node.generator.iter().map(|g| b.exact[g.0].clone()).collect());
        let mut pairs: Vec<(Vec<usize>, f64)> = Vec::new();
        for &k in candidates {
            if pairs.iter().any(|p| p.0.contains(&k)) {
                continue;
            }
            let opposite = (exact[k].0.map(|v| -v), negated(&exact[k].1));
            let mut slots = vec![k];
            slots.extend(candidates.iter().copied().filter(|&j| j != k && exact[j] == opposite));
            let y = slots.iter().map(|&j| size[j]).sum();
            pairs.push((slots, y));
        }
        (pairs, self.account(), nodes)
    }

    /// The account the model's choices are weighed on (model/classify.rs `Account`): the run's length, the held atoms'
    /// hold (the prepared well's depth) and T's semiclassical scale at its least -- its recoil the shortest emission's
    /// wave number. The model's own error is measured after the pilot, on the run's paths (select.rs): at the model's
    /// making it is not known, and a choice is weighed at none of it -- what it books is booked on top of whatever it is.
    fn account(&self) -> Account {
        let request = &self.scene.request;
        let wells = self.scene.trap_potential([0.0; 3], &self.depth);
        let depth = wells.iter().map(|q| -q.value).fold(0.0, larger);
        let shortest = (0..STATES)
            .flat_map(|i| (0..STATES).map(move |j| (i, j)))
            .filter(|&(i, j)| request.emission.iter().any(|c| c[(i, j)] != C64::ZERO))
            .map(|(i, j)| self.scene.jumps().wave_number[(i, j)].re.abs())
            .fold(f64::INFINITY, smaller);
        let scales = crate::estimate::Scales { hbar_over_mass: self.hbar_over_mass, frequencies: [0.0; 3] };
        Account {
            populations: 0.0,
            temperature: 0.0,
            duration: request.times_us.last().copied().unwrap_or(0.0),
            escape: crate::estimate::Escape::new(depth, self.hbar_over_mass),
            floor: crate::estimate::semiclassical([1.0, shortest * shortest], &scales),
        }
    }

    /// A step's counted work with the beats `kept` of the generator other than the static one (model/classify.rs, # The
    /// drop tier; complex multiply-adds): the three nodes' (`nodes`), and the propagation's part the tones change
    /// (spectral.rs, # The second order by tones) -- per tone member its envelope to the eigenbasis, 6n³; per product
    /// of two first-order classes beyond the static group's with itself 2n³ at each of three nodes; per beating class
    /// followed three products of n³; the middle's solve where there are tones, n³ + n³/3 + n³ --, the tones T the
    /// kept beats with their opposites, the classes the distinct sums of two of their exact beats and the static one.
    fn counted(&self, kept: &[usize], nodes: f64) -> f64 {
        self.counted_sized(kept, nodes, (self.rs.keep.len() as f64).powi(3))
    }

    /// `counted` with the cube of P's size in place of `cube` (the blocks' Σ n_b³, # The parts turned).
    fn counted_sized(&self, kept: &[usize], nodes: f64, cube: f64) -> f64 {
        let exact: Vec<Exact> = self.read(|b| kept.iter().map(|&k| b.exact[k].clone()).collect());
        let plus = |a: &Exact, b: &Exact| -> Exact {
            (std::array::from_fn(|i| a.0[i] + b.0[i]), a.1.iter().zip(&b.1).map(|(x, y)| Ratio::new(x.num * y.den + y.num * x.den, x.den * y.den)).collect())
        };
        let width = exact.first().map_or(0, |e| e.1.len());
        let zero: Exact = ([0; 3], vec![Ratio::ZERO; width]);
        let mut members: Vec<Exact> = vec![zero];
        for e in &exact {
            for x in [e.clone(), (e.0.map(|v| -v), negated(&e.1))] {
                if !members.contains(&x) {
                    members.push(x);
                }
            }
        }
        let tones = (members.len() - 1) / 2;
        let mut classes: Vec<Exact> = Vec::new();
        for a in &members {
            for b in &members {
                let c = plus(a, b);
                if !classes.contains(&c) {
                    classes.push(c);
                }
            }
        }
        let first = (1 + 2 * tones) as f64;
        let solve = if tones > 0 { 7.0 / 3.0 } else { 0.0 };
        nodes + cube * (12.0 * tones as f64 + 6.0 * (first * first - 1.0) + 3.0 * (classes.len() as f64 - 1.0) + solve)
    }

    /// The model at a point in the frame, by beats (`Node`; module documentation, # The generator by beats), with the
    /// channels' photon forms where `kicked` (a jump's node): the fold at the point (`Rs::evaluate`), its contraction by
    /// the lab's frequencies, put in the node's structure (`layout`), all in the thread's buffers.
    pub fn node(&self, at: V3, kappa: V3, kicked: bool) -> Node {
        let mut node = self.template.clone();
        self.node_into(&mut node, at, kappa, (kicked, true), false, self.tabled_here());
        node
    }

    /// Whether a node made here now is read off the run's tables (# The tables): the version of the trajectory being
    /// made on this thread (`step::on_version`); outside a trajectory (a path, a judgment) never.
    fn tabled_here(&self) -> bool {
        crate::step::version().is_some_and(|v| v >= 1)
    }

    /// The run's version (step.rs `Dynamics::version`; # The tables): 1 once it has bought its tables, else 0.
    pub(crate) fn version(&self) -> u32 {
        u32::from(matches!(self.tabling.tables.get(), Some(Some(_))))
    }

    /// The tables a node of the making `making` is read off where it is `tabled` (# The tables): the run's, where they
    /// are bought for its kind.
    fn tables_for(&self, making: Making, tabled: bool) -> Option<&Bought> {
        if !tabled {
            return None;
        }
        self.tabling.tables.get().and_then(|t| t.as_ref()).filter(|b| b.uses[kind(making)])
    }

    /// Per kind of node (`KINDS`) whether it costs less off the tables `t` than off the fold: the counted work of its
    /// fold and contraction (`Local::work`, `Out::work`) against the tables' reading and contraction (`Reading::work`,
    /// `Tables::contract`) at the structure's point (the trap's centre at rest, s = 1: in the tables) -- the rest of a
    /// node (the point, the source's values, the filling) the same either way. None where the tables do not read it.
    fn tables_pay(&self, t: &PairTables, orders: usize) -> Option<[bool; KINDS]> {
        let plan = self.rs.plan();
        let mut work = Work::default();
        self.scene.point_into([0.0; 3], [0.0; 3], &mut work.point);
        let mut uses = [false; KINDS];
        for (k, &making) in MAKINGS.iter().enumerate() {
            self.rs.evaluate(&self.scene, &work.point, &mut work.local, making);
            work.out.contract(plan, &self.lab, &work.local, None, making);
            let fold = work.local.work + work.out.work;
            let reading = t.read(plan, &self.lab, &work.local.values, &mut work.pairs)?;
            let tables = reading.work + t.contract(plan, &self.lab, &work.local.values, &reading, orders, making, &mut work.out, &mut work.pairs);
            crate::prof::note(|| format!("tables: kind {k} counted {:.4e} off the fold, {:.4e} off the tables", COMPLEX * fold, COMPLEX * tables));
            uses[k] = tables < fold;
        }
        Some(uses)
    }

    /// The run's tables bought where their price is paid (module documentation, # The tables), at a point no node of the
    /// run is being made at (solve.rs: before every hand-out of trajectories): where its trajectories' nodes off the fold
    /// have taken at least what the tables were last priced at, the layout on [the least s they met, 1] (rs/fold_tabled.rs
    /// `Tables::layout`, priced exactly before it is filled), and the tables filled where that layout's work and its fill's
    /// are within what they took; else the price kept for the next time. Once bought -- or found to be none for the plan
    /// -- never again.
    pub(crate) fn buy_tables(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        let tabling = &self.tabling;
        if tabling.tables.get().is_some() {
            return;
        }
        // (A measurement's, the profiling build's alone: ATOM_PROF_TABLES=0 buys none -- the same run on the fold's own
        // nodes.)
        #[cfg(feature = "prof")]
        if std::env::var("ATOM_PROF_TABLES").is_ok_and(|v| v == "0") {
            return;
        }
        let (rent, low, price) = (tabling.rent.load(Relaxed) as f64, f64::from_bits(tabling.low.load(Relaxed)), f64::from_bits(tabling.price.load(Relaxed)));
        if rent == 0.0 || rent < price || !(low < 1.0) {
            return;
        }
        let mut point = crate::scene::Point::default();
        self.scene.point_into([0.0; 3], [0.0; 3], &mut point);
        let rest = self.rs.tweezer().values(&self.scene, &point);
        let plan = self.rs.plan();
        let Some(layout) = PairTables::layout(plan, &self.lab, &rest, &self.reach(), TABLE_ORDERS, TABLE_DEGREE, low, None) else {
            crate::prof::note(|| "tables: none for the plan".to_string());
            let _ = tabling.tables.set(None);
            return;
        };
        let price = COMPLEX * (layout.work + layout.price);
        crate::prof::note(|| format!("tables: rent {rent:.4e}, price {price:.4e} (layout {:.3e}, fill {:.3e}), {} pieces on [{low:.4}, 1]", COMPLEX * layout.work, COMPLEX * layout.price, layout.pieces.len()));
        if rent < price {
            tabling.price.store(price.to_bits(), Relaxed);
            return;
        }
        // Filled, their budget weighed (`budget`) -- laid and filled again to it where a piece is beyond it (rs/
        // fold_tabled.rs, # The pieces) --, and kept for the kinds of node they cost less for at its terms; none where
        // no kind does.
        let tables = layout.fill(plan, &self.lab, &rest).and_then(|t| {
            let (budget, orders) = self.budget(&t)?;
            let t = if t.pieces.iter().all(|p| p.worst[0] <= budget[0] && p.worst[1] <= budget[1]) {
                t
            } else {
                let again = PairTables::layout(plan, &self.lab, &rest, &self.reach(), TABLE_ORDERS, TABLE_DEGREE, low, Some(budget))?;
                crate::prof::note(|| format!("tables laid again to the budget {budget:?}: {} pieces, priced {:.4e}", again.pieces.len(), COMPLEX * (again.work + again.price)));
                again.fill(plan, &self.lab, &rest)?
            };
            let uses = self.tables_pay(&t, orders).filter(|uses| uses.iter().any(|&u| u))?;
            Some(Bought { tables: Arc::new(t), uses, budget })
        });
        crate::prof::note(|| tables.as_ref().map_or("tables: none kept".to_string(), |b| format!("tables bought: {} pieces, {:.2} MB, {:.3} s, counted {:.4e}, for the kinds {:?}, budget {:?}", b.tables.pieces.len(), b.tables.bytes as f64 / 1e6, b.tables.build_seconds, COMPLEX * b.tables.work, b.uses, b.budget)));
        let _ = tabling.tables.set(tables);
    }

    /// The tables' budget (module documentation, # The tables' budget): of the Doppler series' terms J = 1 … the tables'
    /// at the run's thermal speed -- ρ the largest of the tables' readings at the trap's centre, the speed along each
    /// lab axis --, the relative errors x(J) = [ρ^J/(1 − ρ), ρ^{J−1}/(1 − ρ)] on the rates and on the friction, the one
    /// of least W: the node's counted work off the tables at J terms (the structure's point) times the account's factor
    /// (`Account::factor` at the dropped terms' Y) with x(J) beside the populations' and T's errors. Its x and its J.
    /// None where the tables do not read those points.
    fn budget(&self, t: &PairTables) -> Option<([f64; 2], usize)> {
        let plan = self.rs.plan();
        let mut work = Work::default();
        let request = &self.scene.request;
        // The thermal momentum: (ħ/m)κ² = k_B T.
        let kappa = (crate::units::KT_PER_UK * request.temperature_uk / self.hbar_over_mass).sqrt();
        let mut rho: f64 = 0.0;
        for a in 0..3 {
            self.scene.point_into([0.0; 3], std::array::from_fn(|x| if x == a { kappa } else { 0.0 }), &mut work.point);
            self.rs.tweezer().values_into(&self.scene, &work.point, &mut work.local.values);
            rho = rho.max(t.read(plan, &self.lab, &work.local.values, &mut work.pairs)?.rho);
        }
        self.scene.point_into([0.0; 3], [0.0; 3], &mut work.point);
        self.rs.tweezer().values_into(&self.scene, &work.point, &mut work.local.values);
        let reading = t.read(plan, &self.lab, &work.local.values, &mut work.pairs)?;
        let account = self.account();
        let mut best: Option<(f64, [f64; 2], usize)> = None;
        for j in 1..=t.orders {
            let x = [rho.powi(j as i32) / (1.0 - rho), rho.powi(j as i32 - 1) / (1.0 - rho)];
            let counted = reading.work + t.contract(plan, &self.lab, &work.local.values, &reading, j, Making::node(false), &mut work.out, &mut work.pairs);
            let w = counted * Account { populations: account.populations + x[0], temperature: account.temperature + x[1], ..account }.factor(self.judgment.dropped);
            crate::prof::note(|| format!("tables' budget: J {j}: x {x:?} at the thermal ρ {rho:.3e}, counted {:.4e}, W {w:.4e}", COMPLEX * counted));
            if best.is_none_or(|b| w < b.0) {
                best = Some((w, x, j));
            }
        }
        best.filter(|b| b.0.is_finite()).map(|b| (b.1, b.2))
    }

    /// The run's tables as bought (the tests'), none where it has none.
    #[cfg(test)]
    pub(crate) fn bought(&self) -> Option<&PairTables> {
        self.tabling.tables.get().and_then(|t| t.as_ref()).map(|b| b.tables.as_ref())
    }

    /// The kinds of node the run's tables are read for and their budget (the tests').
    #[cfg(test)]
    pub(crate) fn bought_for(&self) -> Option<([bool; KINDS], [f64; 2])> {
        self.tabling.tables.get().and_then(|t| t.as_ref()).map(|b| (b.uses, b.budget))
    }

    /// Whether the run's tables read the point (the tests').
    #[cfg(test)]
    pub(crate) fn tables_read(&self, at: V3, kappa: V3) -> bool {
        let Some(t) = self.bought() else { return false };
        let mut point = crate::scene::Point::default();
        self.scene.point_into(at, kappa, &mut point);
        let v = self.rs.tweezer().values(&self.scene, &point);
        t.read(self.rs.plan(), &self.lab, &v, &mut PairScratch::default()).is_some()
    }

    /// The run's tables bought now on [`low`, 1], whatever their price (the tests'); whether there are any.
    #[cfg(test)]
    pub(crate) fn buy_tables_now(&self, low: f64) -> bool {
        use std::sync::atomic::Ordering::Relaxed;
        self.tabling.rent.store(u64::MAX >> 1, Relaxed);
        self.tabling.low.store(low.to_bits(), Relaxed);
        self.buy_tables();
        self.bought().is_some()
    }

    /// The run with its direct entries one by one (the tests'): its spans taken off (# The direct entries spanned).
    #[cfg(test)]
    pub(crate) fn unspanned(&mut self) {
        self.lab.spans.clear();
    }

    /// The run with nothing left out but its dropped beats (the tests'): its spans taken off and K's sideband parts put
    /// back (# The direct entries spanned, # The sideband parts dropped); whether any part was left out.
    #[cfg(test)]
    pub(crate) fn whole(&mut self) -> bool {
        self.lab.spans.clear();
        self.sidebands_in()
    }

    /// The run with K's sideband parts put back (the tests', # The sideband parts dropped); whether any was left out.
    #[cfg(test)]
    pub(crate) fn sidebands_in(&mut self) -> bool {
        let parts = self.rs.plan().sidebands();
        let left = parts.iter().filter(|&&b| self.rs.plan().dropped(b)).count();
        for b in parts {
            self.rs.folding.plan.drop_sideband(b, false);
        }
        left > 0
    }

    /// A jump's photon's node at a point (`SelectedMiddle::photon`, the tests'): the photon classes alone.
    #[cfg(test)]
    pub(crate) fn photon_node(&self, at: V3, kappa: V3) -> Node {
        let mut node = self.template.clone();
        self.node_into(&mut node, at, kappa, (true, false), false, self.tabled_here());
        node
    }

    /// A node's parts at a point as `node_into` makes a plain one (a measurement's, the tests'): per part -- the point
    /// (`Scene::point_into`), the source's values there (`Tweezer::values_into`), the fold's numbers (`Local::evaluate`),
    /// the contraction (`Out::contract`), the filling (`fill`) -- its counted real multiply-adds as the node counts them
    /// (`Node::work`'s parts) and its time as `least` takes it (µs). Each part is made once first, untimed, so that each
    /// times its own operations on its inputs.
    #[cfg(test)]
    pub(crate) fn node_parts(&self, at: V3, kappa: V3, least: &mut dyn FnMut(&mut dyn FnMut()) -> f64) -> [(f64, f64); 5] {
        let mut work = Work::default();
        let mut node = self.template.clone();
        let making = Making { lift: false, ..Making::node(false) };
        self.scene.point_into(at, kappa, &mut work.point);
        self.rs.evaluate(&self.scene, &work.point, &mut work.local, making);
        work.out.contract(self.rs.plan(), &self.lab, &work.local, None, making);
        self.fill(&mut node, &work.out, (false, true), &mut work.jumps);
        let Work { point, local, out, jumps, .. } = &mut work;
        let t_point = least(&mut || self.scene.point_into(at, kappa, point));
        let t_values = least(&mut || {
            let mut v = std::mem::take(&mut local.values);
            self.rs.tweezer().values_into(&self.scene, point, &mut v);
            local.values = v;
        });
        let t_fold = least(&mut || local.evaluate(self.rs.plan(), making));
        let t_contract = least(&mut || out.contract(self.rs.plan(), &self.lab, local, None, making));
        let t_fill = least(&mut || {
            self.fill(&mut node, out, (false, true), jumps);
        });
        let filled = self.fill(&mut node, out, (false, true), jumps);
        [(self.point_work(), t_point), (COMPLEX * self.rs.tweezer().values_work(), t_values), (COMPLEX * local.work, t_fold), (COMPLEX * out.work, t_contract), (COMPLEX * filled, t_fill)]
    }

    /// The node at a point (`node`) into one of the run's layout: `kicked` with the photon forms, `whole` with the
    /// generator (else the photon classes alone, a jump's photon's: `Making::PHOTONS`); with the lift and its gradient
    /// where `lifted` (a middle's, `Node::lift`: copied from the contraction, which makes them for the model's gradient
    /// anyway); off the run's tables where `tabled` and they read the point (# The tables), else the fold's own -- a
    /// trajectory's such node's work the tables' rent, its s the least they would cover (`buy_tables`).
    fn node_into(&self, node: &mut Node, at: V3, kappa: V3, (kicked, whole): (bool, bool), lifted: bool, tabled: bool) {
        use std::sync::atomic::Ordering::Relaxed;
        NODES.fetch_add(1, Relaxed);
        let key = self.id;
        WORK.with(|w| {
            let mut w = w.borrow_mut();
            if w.as_ref().is_none_or(|x| x.0 != key) {
                *w = Some((key, Work::default()));
            }
            let work = &mut w.as_mut().expect("the thread's buffers").1;
            self.scene.point_into(at, kappa, &mut work.point);
            // A run of turned parts makes a jump's photon's node whole: its S is the static slot's at the point.
            let made = whole || self.turn.is_some();
            let making = if made { Making { lift: lifted, ..Making::node(kicked) } } else { Making { lift: lifted, ..Making::PHOTONS } };
            let plan = self.rs.plan();
            let tables = self.tables_for(making, tabled);
            // Off the tables where they read the point: the source's values there, the reading, the contraction.
            let mut read = None;
            node.deviation = [0.0; 2];
            if let Some(Bought { tables: t, budget, .. }) = tables {
                self.rs.tweezer().values_into(&self.scene, &work.point, &mut work.local.values);
                // The fewest of the series' terms that hold the node within the budget (# The tables' budget), none where
                // the tables' do not: the fold's own then.
                let reading = t.read(plan, &self.lab, &work.local.values, &mut work.pairs);
                if let Some((reading, orders)) = reading.and_then(|r| (1..=t.orders).find(|&o| r.tail(o) <= budget[0] && r.tail(o - 1) <= budget[1]).map(|o| (r, o))) {
                    let contracted = t.contract(plan, &self.lab, &work.local.values, &reading, orders, making, &mut work.out, &mut work.pairs);
                    let relative = work.pairs.interpolated.relative;
                    let whole = reading.moved + t.dropped;
                    node.deviation = [reading.tail(orders) + relative[0] + whole, reading.tail(orders - 1) + relative[1] + whole];
                    read = Some(reading.work + contracted);
                }
            }
            if read.is_none() {
                self.rs.evaluate(&self.scene, &work.point, &mut work.local, making);
                work.out.contract(plan, &self.lab, &work.local, None, making);
            }
            let filled = self.fill(node, &work.out, (kicked, made), &mut work.jumps);
            let turned = self.turn.as_ref().map_or(0.0, |cross| self.turned(node, cross));
            node.work = match read {
                Some(r) => self.point_work() + COMPLEX * (self.rs.tweezer().values_work() + r + filled + turned),
                None => self.node_work(&work.local, &work.out, filled + turned),
            };
            // A trajectory's node off the fold before the tables are bought: its work their rent, its s the least they
            // would cover.
            if read.is_none() && crate::step::version().is_some() && self.tabling.tables.get().is_none() {
                self.tabling.rent.fetch_add(node.work as u64, Relaxed);
                if let Some(s) = PairTables::intensity_of(plan, &work.local.values) {
                    self.tabling.low.fetch_min(s.max(0.0).to_bits(), Relaxed);
                }
            }
            node.lift.truncate(if lifted { work.out.lift.len() } else { 0 });
            if lifted {
                node.lift.resize_with(work.out.lift.len(), Default::default);
                for ((v, g), (from, from_g)) in node.lift.iter_mut().zip(&work.out.lift) {
                    for (m, f) in std::iter::once(v).chain(g.iter_mut()).zip(std::iter::once(from).chain(from_g)) {
                        (m.rows, m.cols) = (f.rows, f.cols);
                        m.data.clear();
                        m.data.extend_from_slice(&f.data);
                    }
                }
            }
        });
        node.kappa = kappa;
        node.at = at;
        node.kicked = kicked;
        node.whole = whole;
    }

    /// A middle's node at a point (`node`), made in one of the thread's pooled nodes of the run where there is one; whole
    /// or a photon's, with the lift where `lifted` (`node_into`).
    fn pooled(&self, at: V3, kappa: V3, made: (bool, bool), lifted: bool, tabled: bool) -> Rc<Node> {
        let key = self.id;
        let mut node = POOL.with(|p| {
            let mut p = p.borrow_mut();
            p.iter().position(|x| x.0 == key).map(|i| p.swap_remove(i).1)
        })
        .unwrap_or_else(|| Rc::new(self.template.clone()));
        self.node_into(Rc::get_mut(&mut node).expect("a pooled node is no one else's"), at, kappa, made, lifted, tabled);
        node
    }

    /// A middle's node back to the thread's pool, where no one else holds it.
    fn release(&self, node: Rc<Node>) {
        if Rc::strong_count(&node) == 1 && Rc::weak_count(&node) == 0 {
            POOL.with(|p| p.borrow_mut().push((self.id, node)));
        }
    }

    /// The candidate tones (module documentation, # The generator by beats), made once: the nonzero beats of the
    /// layout (every node's), each with its opposite, in the order of their exact vectors; each one's key the exact
    /// vector of the beat e^{+iνt} -- the negation of its plus member's (which beats at Ω = −ν), or its minus
    /// member's --, its slow remainder's ratios over the candidates' common denominator.
    fn tones(&mut self) {
        let met: Vec<usize> = self.template.generator.iter().map(|g| g.0).filter(|&beat| beat != 0).collect();
        let (candidates, keys) = self.read(|b| {
            let mut out: Vec<(Exact, Candidate)> = Vec::new();
            for &k in &met {
                let omega = b.value[k];
                let opposite = |x: &Exact| -> Exact { (x.0.map(|v| -v), negated(&x.1)) };
                // Oriented by ν = −Ω > 0: k is the plus member where Ω < 0.
                let (oriented, plus) = if omega < 0.0 { (b.exact[k].clone(), true) } else { (opposite(&b.exact[k]), false) };
                if out.iter().any(|c| c.0 == oriented) {
                    continue;
                }
                let other = b.ids.get(&if plus { opposite(&oriented) } else { oriented.clone() }).copied().filter(|i| met.contains(i));
                let (p, m) = if plus { (Some(k), other) } else { (other, Some(k)) };
                out.push((oriented, Candidate { nu: omega.abs(), plus: p, minus: m }));
            }
            out.sort_by(|a, b| a.0.cmp(&b.0));
            let candidates: Vec<Candidate> = out.into_iter().map(|c| c.1).collect();
            let exact: Vec<Exact> = candidates
                .iter()
                .map(|c| match (c.plus, c.minus) {
                    (Some(k), _) => (b.exact[k].0.map(|v| -v), negated(&b.exact[k].1)),
                    (None, Some(k)) => b.exact[k].clone(),
                    (None, None) => unreachable!("a candidate of no member"),
                })
                .collect();
            let gcd = |a: i64, b: i64| {
                let (mut a, mut b) = (a.abs(), b.abs());
                while b != 0 {
                    (a, b) = (b, a % b);
                }
                a
            };
            let common = exact.iter().flat_map(|e| e.1.iter().map(|r| r.den)).fold(1_i64, |l, d| l / gcd(l, d) * d);
            let keys = exact.iter().map(|(n, p)| n.iter().copied().chain(p.iter().map(|r| r.num * (common / r.den))).collect()).collect();
            (candidates, keys)
        });
        (self.candidates, self.keys) = (candidates, keys);
    }

    /// The lift's groups in the order of their keys (`rows`), each element's fast index (its place among the run's
    /// fast indices of the lift) and its slow beat's value, made once.
    fn lift_tables(&mut self) {
        let mut order: Vec<usize> = (0..self.lab.labels.len()).collect();
        order.sort_by(|&a, &b| self.lab.labels[a].0.cmp(&self.lab.labels[b].0));
        let mut fasts: Vec<[i64; 3]> = Vec::new();
        let mut lifts = Vec::with_capacity(order.len());
        for g in order {
            let table = self.elements(&self.lab.labels[g].0, Of::Lift);
            let beats: Vec<([i64; 3], f64)> = self.read(|b| table.iter().map(|&i| (b.exact[i].0, b.value[i])).collect());
            let elements = beats
                .into_iter()
                .map(|(fast, value)| {
                    let f = fasts.iter().position(|x| *x == fast).unwrap_or_else(|| {
                        fasts.push(fast);
                        fasts.len() - 1
                    });
                    // Within a fast index the slow beat's value (the fast part's is the index's, common).
                    (f, value - self.value_of(&(fast, vec![])))
                })
                .collect();
            lifts.push((g, elements));
        }
        self.lifts = lifts;
    }

    /// The node the model's structure is read on (module documentation, # The structure's rounding; model/classify.rs
    /// judges on it): at the trap's
    /// centre at rest, the static beat with θ back on its diagonal, and per place of every beat of the generator and of
    /// its gradient along each lab axis the rounding scale of its value -- the sum of the moduli of the terms it is the
    /// sum of and their count (`Out::scales`: H_eff's outputs; the decay's products ½ Σ_g |l_g| |r_g| over their landing
    /// states, each jump amplitude's |l| its sources' |u| times their channel elements' scales).
    fn structure(&self) -> (Node, Scales) {
        self.structure_at([0.0; 3], [0.0; 3])
    }

    /// The node and its rounding scales (`structure`) at a point and momentum: what a check of the node's entries holds
    /// them to there (tests).
    fn structure_at(&self, at: V3, kappa: V3) -> (Node, Scales) {
        let n = self.rs.keep.len();
        let mut node = self.template.clone();
        let mut scales = Scales {
            generator: node.generator.iter().map(|_| (vec![0.0; n * n], vec![0; n * n])).collect(),
            gradient: node.gradient.iter().map(|_| std::array::from_fn(|_| (vec![0.0; n * n], vec![0; n * n]))).collect(),
        };
        let mut work = Work::default();
        self.scene.point_into(at, kappa, &mut work.point);
        self.rs.evaluate(&self.scene, &work.point, &mut work.local, Making::node(false));
        work.out.contract(self.rs.plan(), &self.lab, &work.local, None, Making::node(false));
        work.out.scales(self.rs.plan(), &self.lab, &work.local, None);
        self.fill(&mut node, &work.out, (false, true), &mut work.jumps);
        for (a, t) in self.theta.iter().enumerate() {
            node.generator[0].1[(a, a)] += C64::real(*t);
        }
        // H_eff's outputs at their places.
        for (list, c) in self.tables.h.iter().zip(&work.out.h) {
            for (k, &(x, g, d)) in list.iter().enumerate() {
                if g == u32::MAX {
                    continue;
                }
                let (x, g, d) = (x as usize, g as usize, d as usize);
                scales.generator[g].0[x] += c.scale[k];
                scales.generator[g].1[x] += c.terms[k];
                for (axis, s) in scales.gradient[d].iter_mut().enumerate() {
                    s.0[x] += c.gradient_scale[axis][k];
                    s.1[x] += c.terms[k];
                }
            }
        }
        // The jumps' elements' scales, as `fill` sums their values; then each jump amplitude's and each product's.
        let mut elements: Vec<[Vec<Vec<(f64, usize)>>; 3]> = node.classes.iter().map(|c| std::array::from_fn(|a| c.axes[a].iter().map(|j| vec![(0.0, 0); j.elements.len()]).collect())).collect();
        for ((list, c), (_, _, which, _, _)) in self.tables.channels.iter().zip(&work.out.channels).zip(&self.lab.channel_outputs) {
            if *which != 0 {
                continue;
            }
            for (k, &(_, class, axis, jump, element)) in list.iter().enumerate() {
                let e = &mut elements[class as usize][axis as usize][jump as usize][element as usize];
                (e.0, e.1) = (e.0 + c.scale[k], e.1 + c.terms[k]);
            }
        }
        let mut amplitudes: Vec<(f64, usize)> = Vec::new();
        for j in &self.tables.jumps {
            let at = amplitudes.len();
            amplitudes.resize(at + GROUND * j.cols.len(), (0.0, 0));
            for (p, sources) in &j.elements {
                let (s, t) = sources.iter().fold((0.0, 0), |(s, t), &(a, c, e, u)| {
                    let x = elements[j.class][a][c][e];
                    (s + u.abs() * x.0, t.max(x.1))
                });
                amplitudes[at + p] = (s, t + sources.len());
            }
        }
        let block_of = &self.block_of;
        for d in &self.tables.decay {
            let (cl, cr) = (&self.tables.jumps[d.left].cols, &self.tables.jumps[d.right].cols);
            let (sl, sr) = (self.tables.starts[d.left], self.tables.starts[d.right]);
            for (i, &a) in cl.iter().enumerate() {
                for (j, &b) in cr.iter().enumerate().filter(|(_, &b)| block_of[b] == block_of[a]) {
                    let (mut s, mut t) = (0.0, 0);
                    for g in 0..GROUND {
                        let (l, r) = (amplitudes[sl + g * cl.len() + i], amplitudes[sr + g * cr.len() + j]);
                        s += l.0 * r.0;
                        t = t.max(l.1 + r.1);
                    }
                    let (s, t) = (0.5 * s, t + GROUND);
                    let forward = &mut scales.generator[d.forward];
                    (forward.0[a * n + b], forward.1[a * n + b]) = (forward.0[a * n + b] + s, forward.1[a * n + b] + t);
                    if d.left != d.right {
                        let back = &mut scales.generator[d.back];
                        (back.0[b * n + a], back.1[b * n + a]) = (back.0[b * n + a] + s, back.1[b * n + a] + t);
                    }
                }
            }
        }
        (node, scales)
    }

    /// The exact blocks of P (module documentation, # Exact blocks): places in P.
    pub fn blocks(&self) -> &[Vec<usize>] {
        &self.blocks
    }

    /// The decay's products by their structural terms (`Tables::entries`), made with the layout and again where its
    /// slots or its blocks change (`judge`): per product of the layout (`Tables::decay`, in its order) each
    /// place (a, b) of its two components' columns within one block, in their order, whose components share a ground
    /// row: the shared rows' terms in the rows' order, the forward and the back places. A place of no shared row is a
    /// structural zero of the product, no entry.
    fn decay_terms(&mut self) {
        let n = self.rs.keep.len();
        let t = &self.tables;
        let (mut terms, mut entries) = (Vec::new(), Vec::new());
        for d in &t.decay {
            let (jl, jr) = (&t.jumps[d.left], &t.jumps[d.right]);
            let (cl, cr) = (jl.cols.len(), jr.cols.len());
            let (sl, sr) = (t.starts[d.left], t.starts[d.right]);
            let held = |j: &Spherical, cols: usize| -> Vec<bool> {
                let mut held = vec![false; GROUND * cols];
                for (p, _) in &j.elements {
                    held[*p] = true;
                }
                held
            };
            let (hl, hr) = (held(jl, cl), held(jr, cr));
            for (i, &a) in jl.cols.iter().enumerate() {
                for (j, &b) in jr.cols.iter().enumerate().filter(|(_, &b)| self.block_of[b] == self.block_of[a]) {
                    let before = terms.len();
                    for g in (0..GROUND).filter(|&g| hl[g * cl + i] && hr[g * cr + j]) {
                        terms.push(((sl + g * cl + i) as u32, (sr + g * cr + j) as u32));
                    }
                    if terms.len() == before {
                        continue;
                    }
                    let back = if d.left != d.right { (d.back as u32, (b * n + a) as u32) } else { (u32::MAX, u32::MAX) };
                    entries.push(Entry { end: terms.len() as u32, forward: (d.forward as u32, (a * n + b) as u32), back });
                }
            }
        }
        (self.tables.terms, self.tables.entries) = (terms, entries);
    }

    /// A node's numbers from a contraction by the lab's frequencies (`layout`'s places): its photon classes, and where
    /// `whole` its generator and gradient.
    fn fill(&self, node: &mut Node, out: &Out, (kicked, whole): (bool, bool), jumps: &mut Vec<C64>) -> f64 {
        let n = self.rs.keep.len();
        // The counted complex multiply-adds (decide.rs, # Work): every number laid MOVED, every product one.
        let moved = MOVED / COMPLEX;
        let mut work = 0.0;
        for c in node.classes.iter_mut() {
            for axis in c.axes.iter_mut() {
                for j in axis.iter_mut() {
                    for e in j.elements.iter_mut() {
                        (e.value, e.gradient, e.kick) = (C64::ZERO, [C64::ZERO; 3], C64::ZERO);
                    }
                    work += 5.0 * moved * j.elements.len() as f64;
                }
            }
        }
        for ((list, c), (_, _, which, _, _)) in self.tables.channels.iter().zip(&out.channels).zip(&self.lab.channel_outputs) {
            if *which == 1 && !kicked {
                continue;
            }
            work += moved * (if *which == 0 { 4 } else { 1 } * list.len()) as f64;
            for (k, &(_, class, axis, jump, element)) in list.iter().enumerate() {
                let e = &mut node.classes[class as usize].axes[axis as usize][jump as usize].elements[element as usize];
                if *which == 0 {
                    e.value += c.value[k];
                    for (g, d) in e.gradient.iter_mut().zip(&c.gradient) {
                        *g += d[k];
                    }
                } else {
                    e.kick += c.value[k];
                }
            }
        }
        if !whole {
            return work;
        }
        work += moved * ((node.generator.len() + 3 * node.gradient.len()) * n * n + n) as f64;
        work += 4.0 * moved * self.tables.h.iter().map(Vec::len).sum::<usize>() as f64;
        for (_, m) in node.generator.iter_mut() {
            m.data.fill(C64::ZERO);
        }
        for (_, g) in node.gradient.iter_mut() {
            for m in g.iter_mut() {
                m.data.fill(C64::ZERO);
            }
        }
        for (a, t) in self.theta.iter().enumerate() {
            node.generator[0].1.data[a * n + a] = C64::real(-t);
        }
        for (list, c) in self.tables.h.iter().zip(&out.h) {
            for (k, &(x, g, d)) in list.iter().enumerate() {
                if g == u32::MAX {
                    continue;
                }
                let x = x as usize;
                node.generator[g as usize].1.data[x] += c.value[k];
                for (m, dg) in node.gradient[d as usize].1.iter_mut().zip(&c.gradient) {
                    m.data[x] += dg[k];
                }
            }
        }
        // The decay: the spherical components on the ground rows and their columns, then each pair's product.
        jumps.clear();
        for j in &self.tables.jumps {
            let at = jumps.len();
            jumps.resize(at + GROUND * j.cols.len(), C64::ZERO);
            let axes = &node.classes[j.class].axes;
            for (p, sources) in &j.elements {
                jumps[at + p] = sources.iter().map(|&(a, c, e, u)| u * axes[a][c].elements[e].value).sum();
                work += sources.len() as f64;
            }
            work += moved * (GROUND * j.cols.len()) as f64;
        }
        // Each product's terms (four real multiply-adds each), its −i/2 and its conjugate's at their places.
        work += self.tables.terms.len() as f64 + 2.0 * (1.0 + moved) * self.tables.entries.len() as f64;
        // Each product's entries by their structural terms (`decay_terms`; between two exact blocks nothing, `judge`):
        // Σ_g conj(ℓ_l[g, i]) ℓ_r[g, j] over the rows the two share, each term two fused multiply-adds a part as the
        // product kernel makes them (`linalg::adjoint_product`, whose other terms are the structural zeros'), then −i/2
        // of it at its forward place and of its conjugate at its back place.
        let half = C64::new(0.0, -0.5);
        let mut from = 0;
        for e in &self.tables.entries {
            let (mut re, mut im) = (0.0_f64, 0.0_f64);
            for &(l, r) in &self.tables.terms[from..e.end as usize] {
                let (x, y) = (jumps[l as usize].conj(), jumps[r as usize]);
                re = x.re.mul_add(y.re, (-x.im).mul_add(y.im, re));
                im = x.re.mul_add(y.im, x.im.mul_add(y.re, im));
            }
            from = e.end as usize;
            let p = C64::new(re, im);
            node.generator[e.forward.0 as usize].1.data[e.forward.1 as usize] += half * p;
            if e.back.0 != u32::MAX {
                node.generator[e.back.0 as usize].1.data[e.back.1 as usize] += half * p.conj();
            }
        }
        work
    }

    /// The ground populations of a state of P (an explicit excited state's by its decay's branching), over its norm.
    fn populations(&self, psi: &[C64]) -> [f64; GROUND] {
        let total = norm_sqr(psi);
        let mut p = [0.0; GROUND];
        for (z, branching) in psi.iter().zip(&self.branching) {
            let w = z.norm_sqr() / total;
            for (x, b) in p.iter_mut().zip(branching) {
                *x += w * b;
            }
        }
        p
    }

    /// The 32 states' populations of a state of P at a point and time (module documentation, # The rest of the model):
    /// P's, and Q's lifted through M_c (`ModelAt::lift`) of each fast index of the lift apart, each element at its slow
    /// beat's phase; over their sum. The lift is the fold's at the point (`READOUT`); a middle's readings take its node's
    /// instead (`SelectedMiddle::readout`), the one assembly (`lifted`).
    fn rows(&self, at: &Point, psi: &[C64]) -> [f64; STATES] {
        let key = self.id;
        WORK.with(|w| {
            let mut w = w.borrow_mut();
            if w.as_ref().is_none_or(|x| x.0 != key) {
                *w = Some((key, Work::default()));
            }
            let Work { point, local, out, rows, .. } = &mut w.as_mut().expect("the thread's buffers").1;
            self.scene.point_into(at.r, at.kappa, point);
            self.rs.evaluate(&self.scene, point, local, Making::READOUT);
            out.contract(self.rs.plan(), &self.lab, local, None, Making::READOUT);
            self.lifted(rows, psi, None, at.t, &mut |g, group| {
                let m = &out.lift[g].0;
                group.clear();
                group.extend_from_slice(&m.data);
                m.rows != 0
            })
        })
    }

    /// The rows (`rows`) of a state of P at the time t, the state framed by `turn` where the lift is the Doppler frame's
    /// (`SelectedMiddle::readout`), each lab group's lift put by `group` (false where it has none).
    fn lifted(&self, buffers: &mut Rows, psi: &[C64], turn: Option<&[C64; STATES]>, t: f64, group: &mut dyn FnMut(usize, &mut Vec<C64>) -> bool) -> [f64; STATES] {
        let Rows { state, group: values, fasts, amplitudes } = buffers;
        let n = self.rs.keep.len();
        // The state over its norm, framed where it is the frame's.
        let scale = 1.0 / norm_sqr(psi).sqrt();
        state.clear();
        state.extend(psi.iter().enumerate().map(|(a, z)| turn.map_or(*z * scale, |p| *z * scale * p[a])));
        let mut rows = [0.0; STATES];
        for (z, &s) in state.iter().zip(&self.rs.keep) {
            rows[s] = z.norm_sqr();
        }
        // Q's amplitudes M ψ by fast index, in the order met (the groups by their keys, each element in its place),
        // each element at its slow beat's phase at the time; a group of no amplitude none. The state is pure, so each
        // fast index's (M ρ M†)_kk is |(M ψ)_k|².
        fasts.clear();
        let q = self.rs.folded.len();
        for (g, elements) in &self.lifts {
            if !group(*g, values) || values.iter().all(|z| *z == C64::ZERO) {
                continue;
            }
            for (x, &(fast, slow)) in elements.iter().enumerate() {
                let z = values[x];
                if z == C64::ZERO {
                    continue;
                }
                let i = fasts.iter().position(|f| *f == fast).unwrap_or_else(|| {
                    fasts.push(fast);
                    if amplitudes.len() < fasts.len() {
                        amplitudes.push(Vec::new());
                    }
                    let u = &mut amplitudes[fasts.len() - 1];
                    u.clear();
                    u.resize(q, C64::ZERO);
                    fasts.len() - 1
                });
                let z = if slow == 0.0 { z } else { z * C64::cis(-slow * t) };
                amplitudes[i][x / n] += z * state[x % n];
            }
        }
        for u in &amplitudes[..fasts.len()] {
            for (z, &s) in u.iter().zip(&self.rs.folded) {
                rows[s] += z.norm_sqr();
            }
        }
        let total: f64 = rows.iter().sum();
        rows.map(|x| x / total)
    }
}

/// A potential of the run (module documentation, # The rest of the model): Σ_g p_g V_g(r) of `Scene::trap_potential`,
/// or, `running`, of `Scene::running_potential` (the reported trap's).
pub struct Potential<'a> {
    model: &'a Selected,
    weights: [f64; GROUND],
    running: bool,
}

impl Potential<'_> {
    /// Its second-order expansion at r.
    pub fn at(&self, r: V3) -> Quadratic {
        self.rounded(r).0
    }

    /// Its expansion at r with the rounding of its value: `rounding` of its ground terms times their moduli
    /// (model/potential.rs `descend`).
    pub fn rounded(&self, r: V3) -> (Quadratic, f64) {
        let (scene, depth) = (&self.model.scene, &self.model.depth);
        let wells = if self.running { scene.running_potential(r, depth, &self.model.smooth) } else { scene.trap_potential(r, depth) };
        let mut out = Quadratic { value: 0.0, gradient: [0.0; 3], hessian: [[0.0; 3]; 3] };
        let mut scale = 0.0;
        for (w, q) in self.weights.iter().zip(&wells).filter(|(w, _)| **w != 0.0) {
            out.value += w * q.value;
            scale += (w * q.value).abs();
            for x in 0..3 {
                out.gradient[x] += w * q.gradient[x];
                for y in 0..3 {
                    out.hessian[x][y] += w * q.hessian[x][y];
                }
            }
        }
        (out, rounding(GROUND) * scale)
    }

    /// B with −V(r) ≤ B |a_tw(r)|² (ensemble.rs `Well::envelope`): V is the tweezer's intensity -- a profile of one at
    /// the focus -- times one coefficient, so B = |V(0)|.
    pub fn peak(&self) -> f64 {
        self.at([0.0; 3]).value.abs()
    }
}

impl Drop for SelectedMiddle<'_> {
    /// The middle's nodes back to the thread's pool (`Selected::node`).
    fn drop(&mut self) {
        let mut nodes = std::mem::take(&mut *self.nodes.borrow_mut());
        for node in nodes.drain(..) {
            self.model.release(node);
        }
        LISTS.with(|l| l.borrow_mut().push(nodes));
    }
}

impl SelectedMiddle<'_> {
    /// The node at the time s from the middle along the path, made where first asked for, without the photon forms (a
    /// jump's are its own, `photon`); the middle's with the lift (`readout`).
    fn node(&self, path: &Path, s: f64) -> Rc<Node> {
        let at: V3 = std::array::from_fn(|i| self.r[i] + path.velocity[i] * s);
        if let Some(node) = self.nodes.borrow().iter().find(|x| x.kappa == path.kappa && x.at == at) {
            return node.clone();
        }
        let node = self.model.pooled(at, path.kappa, (false, true), s == 0.0 && self.reads.get(), self.tabled);
        self.made.set(self.made.get() + node.work);
        self.deviated(&node);
        #[cfg(test)]
        MADE.with(|m| m.borrow_mut()[0] += 1);
        self.nodes.borrow_mut().push(node.clone());
        node
    }

    /// A node made on the middle into its largest deviation (`step::Middle::deviation`).
    fn deviated(&self, node: &Node) {
        let d = self.deviation.get();
        self.deviation.set([d[0].max(node.deviation[0]), d[1].max(node.deviation[1])]);
    }

    /// The middle's node with its lift (`readout`): the middle's node, made again with the lift where it was made
    /// without (a step that told it reads nothing, `step::Middle::reading`) and put in its place.
    fn lifted(&self, path: &Path) -> Rc<Node> {
        let node = self.node(path, 0.0);
        if !node.lift.is_empty() {
            return node;
        }
        let made = self.model.pooled(node.at, path.kappa, (false, true), true, self.tabled);
        self.made.set(self.made.get() + made.work);
        self.deviated(&made);
        let mut nodes = self.nodes.borrow_mut();
        if let Some(at) = nodes.iter().position(|x| Rc::ptr_eq(x, &node)) {
            nodes[at] = made.clone();
        }
        made
    }

    /// The Doppler frame's turn at the time s from the middle along the path (module documentation, # One cluster, one
    /// frame): p_a = e^{−iK_a·v s} per state, an element a ← b of the lab's matrices the frame's times p_a p_b*; none
    /// where nothing turns -- at the middle, or where every K is zero.
    fn turns(&self, path: &Path, s: f64) -> Option<[C64; STATES]> {
        let waves = &self.model.cluster.waves;
        if s == 0.0 || waves.iter().all(|k| *k == [0.0; 3]) {
            return None;
        }
        let mut p = [C64::ONE; STATES];
        for (p, k) in p.iter_mut().zip(waves) {
            *p = C64::cis(-(0..3).map(|a| k[a] * path.velocity[a]).sum::<f64>() * s);
        }
        Some(p)
    }

    /// A matrix of the lab at a time into the frame there (`turns`): entry a ← b times p_a p_b*.
    fn framed(p: &[C64], m: &mut [C64]) {
        let n = m.len().isqrt();
        for a in 0..n {
            for b in 0..n {
                m[a * n + b] *= p[a] * p[b].conj();
            }
        }
    }

    /// A density of the frame at a time as the lab's there (`turns`), for the lab's matrices: tr(Ĝ_frame ρ) = Σ_ab
    /// G_ab p_a p_b* ρ_ba, so ρ_ab times p_a* p_b.
    fn unframed(p: &[C64], rho: &[C64], out: &mut Vec<C64>) {
        let n = rho.len().isqrt();
        out.clear();
        out.extend((0..n * n).map(|i| rho[i] * p[i / n].conj() * p[i % n]));
    }

    /// e^{−iΩ (t_m + s)} of a beat.
    fn phase(&self, values: &[f64], beat: usize, s: f64) -> C64 {
        if beat == 0 {
            C64::ONE
        } else {
            C64::cis(-values[beat] * (self.t + s))
        }
    }

    /// The candidates of the path (module documentation, # The generator by beats): the run's (`Selected::tones`)
    /// where a node is made at its momentum, none before.
    fn candidates(&self, path: &Path) -> &[Candidate] {
        if self.nodes.borrow().iter().any(|x| x.kappa == path.kappa) { &self.model.candidates } else { &[] }
    }

    /// tr(Ĝ_a σ) per axis of H_eff's gradient components at a node, each beat by its factor (none: left out). The paths'
    /// force reads it too (paths.rs `Form::read`).
    pub(crate) fn contract(node: &Node, each: impl Fn(usize) -> Option<C64>, rho: &[C64]) -> [C64; 3] {
        let mut out = [C64::ZERO; 3];
        for (beat, g) in &node.gradient {
            if let Some(f) = each(*beat) {
                for a in 0..3 {
                    out[a] += f * trace(&g[a], rho);
                }
            }
        }
        out
    }

    /// A jump's photon of the state ψ at the time s from the middle, at the jump's place (module documentation,
    /// # The photons): the classes' rates, the class `draw` picks and its photon.
    fn emitted(&self, path: &Path, s: f64, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        let node = self.photon(path, s);
        let emitted = READS.with(|r| {
            let mut r = r.borrow_mut();
            let Reads { photons: list, rates, .. } = &mut *r;
            // Every class's rate on its amplitudes alone; the photon of the class drawn whole (`class_photons`: the
            // same numbers as every class's whole, `photons`, to the bit).
            let norm = norm_sqr(psi);
            if list.len() < node.classes.len() {
                list.resize_with(node.classes.len(), Photons::default);
            }
            rates.clear();
            self.model.read(|b| {
                for (class, p) in node.classes.iter().zip(list.iter_mut()) {
                    rates.push(class_photons(class, node.kicked, (&b.value, self.t + s), psi, norm, p, false));
                }
            });
            let k = draw(rates)?;
            self.model.read(|b| class_photons(&node.classes[k], node.kicked, (&b.value, self.t + s), psi, norm, &mut list[k], true));
            let p = &list[k];
            let n = psi.len();
            let amplitudes = p.u.each_ref().map(|u| u.iter().copied().collect::<Internal>());
            // Every class's rate on its amplitudes alone and the drawn one's whole, as they were made.
            let work = node.classes.iter().map(|c| class_photons_work(c, node.kicked, false, n)).sum::<f64>() + class_photons_work(&node.classes[k], node.kicked, true, n);
            let wave_number = p.wave_number.expect("a jump's node holds the photon forms");
            Some(Emitted { landing: 0, wave_number, amplitudes, kick: p.moments.kick, work })
        });
        self.model.release(node);
        emitted
    }

    /// A jump's photon's node at the time s from the middle along the path (module documentation, # The photons): the
    /// photon classes alone (`Making::PHOTONS`: the channels with their gradients and photon forms, no H_eff and no
    /// generator), made for the jump, its work counted, and not kept -- nothing else reads the model at a jump's point.
    fn photon(&self, path: &Path, s: f64) -> Rc<Node> {
        let at: V3 = std::array::from_fn(|i| self.r[i] + path.velocity[i] * s);
        let node = self.model.pooled(at, path.kappa, (true, false), false, self.tabled);
        self.made.set(self.made.get() + node.work);
        self.deviated(&node);
        #[cfg(test)]
        MADE.with(|m| m.borrow_mut()[1] += 1);
        node
    }
}

/// A class's photons of a state (module documentation, # The photons): u_a, ∇u_a and the photon forms' K u_a, its
/// moments over ‖ψ‖² (model.rs `Moments`, the one rule: the rate W, Σ Im⟨u|∇u⟩, Σ Re⟨∇u|⊗|∇u⟩, the share s of b a jump
/// carries and s b), and the photon's wave number where the node holds the photon forms (`Node::kicked`). Kept in the
/// thread's buffers (`Reads`).
struct Photons {
    u: [Vec<C64>; 3],
    du: [[Vec<C64>; 3]; 3],
    ku: [Vec<C64>; 3],
    moments: Moments,
    wave_number: Option<f64>,
}

impl Default for Photons {
    fn default() -> Photons {
        Photons { u: Default::default(), du: Default::default(), ku: Default::default(), moments: Moments::new(0.0, [0.0; 3], [[0.0; 3]; 3], 0.0, 0), wave_number: None }
    }
}

/// Every class's photons of the state ψ at the time t on a node (module documentation, # The photons), into `list`
/// (as long as it needs, never shorter): the node's classes' in their order.
fn photons<'a>(node: &Node, values: &[f64], t: f64, psi: &[C64], list: &'a mut Vec<Photons>) -> &'a [Photons] {
    let norm = norm_sqr(psi);
    if list.len() < node.classes.len() {
        list.resize_with(node.classes.len(), Photons::default);
    }
    for (class, p) in node.classes.iter().zip(list.iter_mut()) {
        class_photons(class, node.kicked, (values, t), psi, norm, p, true);
    }
    &list[..node.classes.len()]
}

/// `class_photons`' counted real multiply-adds (decide.rs, # Work) of a class on n states: its fifteen vectors zeroed;
/// per jump component its beat's phase (a cis where it beats) and per element its products -- u's, and where `whole`
/// the three gradients' and the photon forms' where `kicked` --, each run of its elements in one row their sum added out
/// (a row's test an element); then
/// u's weight alone (its norm, over ‖ψ‖²), or where `whole` the moments (`Moments::of`: W, the three means' and the nine
/// second moments' inner products over the three axes) and the photon's wave number.
fn class_photons_work(class: &Class, kicked: bool, whole: bool, n: usize) -> f64 {
    let per = if whole { 4 + usize::from(kicked) } else { 1 } as f64;
    let parts: f64 = class
        .axes
        .iter()
        .flatten()
        .map(|jump| {
            // Its rows as `class_photons` takes them: each run of elements of one row a sum added out.
            let rows = 1 + jump.elements.windows(2).filter(|w| w[0].at / n != w[1].at / n).count();
            (if jump.beat == 0 { 0.0 } else { ELEMENTARY }) + COMPLEX * per * (jump.elements.len() + rows) as f64 + REAL_OP * jump.elements.len() as f64
        })
        .sum();
    let after = if whole { COMPLEX * ((3 + 9 + 27 + if kicked { 3 } else { 0 }) * n) as f64 } else { (6 * n) as f64 + DIVISION };
    15.0 * MOVED * n as f64 + parts + after
}

/// A class's photons of the state ψ at the time t (`photons`): where `whole`, u_a, ∇u_a, the photon forms' K u_a where
/// the node holds them, its moments over ‖ψ‖² = `norm` and its wave number; else u_a alone. Returns its rate W/‖ψ‖²,
/// the moments' weight (`Moments::of`, then `over`: the same operations either way).
///
/// One pass over each component's elements (`row_sums`): per row of P × P the sums of L, of ∇L and of K against ψ
/// side by side, each in column order from zero, then f times each into its output -- the same operations, in the same
/// order, as a pass per output, each beat's factor e^{−iνt} made once a class (until 2026-11-23: five passes a
/// component, each row found by a division, a factor a component, every buffer cleared and grown again a class), so
/// the same numbers to the bit. Where the class is not `whole` only u_a is
/// made, and only its buffers are zeroed (the gradients' and the forms' are read of a class made whole alone).
fn class_photons(class: &Class, kicked: bool, (values, t): (&[f64], f64), psi: &[C64], norm: f64, p: &mut Photons, whole: bool) -> f64 {
    let n = psi.len();
    let Photons { u, du, ku, .. } = &mut *p;
    let zeroed = |v: &mut Vec<C64>| {
        if v.len() == n {
            v.fill(C64::ZERO);
        } else {
            v.clear();
            v.resize(n, C64::ZERO);
        }
    };
    u.iter_mut().for_each(zeroed);
    if whole {
        ku.iter_mut().chain(du.iter_mut().flatten()).for_each(zeroed);
    }
    // Each beat's factor once: the components share a few beats (tilt2's 246 components, its tones' few).
    let mut turned: [(usize, C64); 32] = [(0, C64::ONE); 32];
    let mut held = 1;
    for (a, axis) in class.axes.iter().enumerate() {
        for jump in axis {
            let f = match turned[..held].iter().find(|x| x.0 == jump.beat) {
                Some(&(_, f)) => f,
                None => {
                    let f = C64::cis(-values[jump.beat] * t);
                    if held < turned.len() {
                        turned[held] = (jump.beat, f);
                        held += 1;
                    }
                    f
                }
            };
            row_sums(&jump.elements, psi, |row, [zv, zk, zg0, zg1, zg2]| {
                u[a][row] += f * zv;
                if whole {
                    if kicked {
                        ku[a][row] += f * zk;
                    }
                    for (d, z) in du.iter_mut().zip([zg0, zg1, zg2]) {
                        d[a][row] += f * z;
                    }
                }
            }, (whole, kicked));
        }
    }
    if !whole {
        let weight: f64 = (0..3).map(|a| norm_sqr(&u[a])).sum();
        return weight / norm;
    }
    let moments = Moments::of(u, du, class.absorbed);
    let weight = moments.weight;
    p.wave_number = kicked.then(|| if weight != 0.0 { (0..3).map(|a| inner(&u[a], &ku[a]).re).sum::<f64>() / weight } else { 0.0 });
    p.moments = moments.over(norm);
    p.moments.weight
}

/// A component's sums against ψ row by row (`class_photons`): its elements in their places' order (row-major), each
/// row's run of them the sums [L ψ, K ψ, ∂₀L ψ, ∂₁L ψ, ∂₂L ψ] of that row, each from zero in column order -- K's where
/// `kicked`, the gradients' and K's where `whole` (else zero) --, handed to `put` with the row. A row's run ends at the
/// first element past the row's last place: no division an element.
#[inline]
fn row_sums(elements: &[Element], psi: &[C64], mut put: impl FnMut(usize, [C64; 5]), (whole, kicked): (bool, bool)) {
    let n = psi.len();
    let mut i = 0;
    while i < elements.len() {
        let row = elements[i].at / n;
        let (base, end) = (row * n, (row + 1) * n);
        let mut z = [C64::ZERO; 5];
        if whole {
            while let Some(e) = elements.get(i).filter(|e| e.at < end) {
                let x = psi[e.at - base];
                z[0] += e.value * x;
                if kicked {
                    z[1] += e.kick * x;
                }
                z[2] += e.gradient[0] * x;
                z[3] += e.gradient[1] * x;
                z[4] += e.gradient[2] * x;
                i += 1;
            }
        } else {
            while let Some(e) = elements.get(i).filter(|e| e.at < end) {
                z[0] += e.value * psi[e.at - base];
                i += 1;
            }
        }
        put(row, z);
    }
}

#[cfg(test)]
impl SelectedMiddle<'_> {
    /// Every class's photon moments of the state ψ at the middle (model.rs `Moments`, over ‖ψ‖²; the tests').
    pub(crate) fn moments(&self, path: &Path, psi: &[C64]) -> Vec<Moments> {
        let node = self.node(path, 0.0);
        let mut list = Vec::new();
        self.model.read(|b| photons(&node, &b.value, self.t, psi, &mut list)).iter().map(|p| p.moments).collect()
    }
}

impl crate::step::Middle for SelectedMiddle<'_> {
    type Path = Path;

    fn states(&self) -> usize {
        self.model.rs.keep.len()
    }

    fn still(&self) -> bool {
        false
    }

    fn blocks(&self) -> &[Vec<usize>] {
        if self.model.blocks.len() > 1 { &self.model.blocks } else { &[] }
    }

    fn path(&self, kappa: [f64; 3], velocity: [f64; 3]) -> Path {
        Path { kappa, velocity }
    }

    /// K_a·v of each state's Doppler frame (module documentation, # One cluster, one frame).
    fn gauge(&self, path: &Path, rates: &mut [f64]) {
        for (rate, k) in rates.iter_mut().zip(&self.model.cluster.waves) {
            *rate = (0..3).map(|a| k[a] * path.velocity[a]).sum();
        }
    }

    /// Ĥ(s) = Σ_k G_k(s) e^{−iΩ_k(t_m + s)} (module documentation, # The generator by beats), in the Doppler frame with
    /// its diag(K_a·v).
    fn generator(&self, path: &Path, s: f64, h: &mut Mat) {
        let node = self.node(path, s);
        h.data.fill(C64::ZERO);
        self.model.read(|b| {
            for (beat, g) in &node.generator {
                let f = self.phase(&b.value, *beat, s);
                for (z, x) in h.data.iter_mut().zip(&g.data) {
                    *z += f * *x;
                }
            }
        });
        if let Some(p) = self.turns(path, s) {
            SelectedMiddle::framed(&p, &mut h.data);
        }
        let n = self.states();
        for (a, k) in self.model.cluster.waves.iter().enumerate() {
            h.data[a * n + a] += C64::real((0..3).map(|i| k[i] * path.velocity[i]).sum::<f64>());
        }
    }

    /// Each candidate's ν and its key (module documentation, # The generator by beats): the exact vector of the beat
    /// e^{+iνt} -- the negation of its plus member's (which beats at Ω = −ν), or its minus member's --, its slow
    /// remainder's ratios over the candidates' common denominator.
    fn beats(&self, path: &Path, beats: &mut Vec<f64>, keys: &mut Vec<Vec<i64>>) {
        let count = self.candidates(path).len();
        for (c, key) in self.model.candidates[..count].iter().zip(&self.model.keys) {
            beats.push(c.nu);
            keys.push(key.clone());
        }
    }

    /// Candidate m's members at s: G_plus(s) e^{iν t_m} and G_minus(s) e^{−iν t_m} (their parts of `generator`).
    fn envelopes(&self, path: &Path, m: usize, s: f64, plus: &mut Mat, minus: &mut Mat) {
        let c = self.candidates(path)[m];
        let node = self.node(path, s);
        self.model.read(|b| {
            for (member, out) in [(c.plus, plus), (c.minus, minus)] {
                out.data.fill(C64::ZERO);
                let Some(k) = member else { continue };
                if let Some((_, g)) = node.generator.iter().find(|g| g.0 == k) {
                    let f = self.phase(&b.value, k, 0.0);
                    for (z, x) in out.data.iter_mut().zip(&g.data) {
                        *z = f * *x;
                    }
                }
                if let Some(p) = self.turns(path, s) {
                    SelectedMiddle::framed(&p, &mut out.data);
                }
            }
        });
    }

    /// Σ_j Re tr(Ĝ⁺(s_j) σ_j + Ĝ⁻(s_j) σ_j†) of candidate m's members' H_eff gradients at the phases of `envelopes`.
    fn tone_force(&self, path: &Path, m: usize, sums: &[(f64, &[C64], &[C64])]) -> [f64; 3] {
        let c = self.candidates(path)[m];
        let n = self.states();
        let mut out = [0.0; 3];
        for &(s, re, im) in sums {
            let node = self.node(path, s);
            let forms = READS.with(|r| {
                let mut r = r.borrow_mut();
                let Reads { sigma, dagger, lab, .. } = &mut *r;
                sigma.clear();
                sigma.extend(re.iter().zip(im).map(|(r, i)| *r + C64::I * *i));
                dagger.clear();
                dagger.extend((0..n * n).map(|x| sigma[(x % n) * n + x / n].conj()));
                // The frame's sums as the lab's at the node (`turns`).
                if let Some(p) = self.turns(path, s) {
                    for m in [&mut *sigma, &mut *dagger] {
                        SelectedMiddle::unframed(&p, m, lab);
                        std::mem::swap(m, lab);
                    }
                }
                self.model.read(|b| {
                    let mut f = [C64::ZERO; 3];
                    for (member, rho) in [(c.plus, &*sigma), (c.minus, &*dagger)] {
                        let Some(k) = member else { continue };
                        let phase = self.phase(&b.value, k, 0.0);
                        let x = SelectedMiddle::contract(&node, |beat| (beat == k).then_some(phase), rho);
                        for a in 0..3 {
                            f[a] += x[a];
                        }
                    }
                    f
                })
            });
            for a in 0..3 {
                out[a] += forms[a].re;
            }
        }
        out
    }

    fn tone_work(&self, _m: usize) -> [f64; 2] {
        let n = self.states() as f64;
        [2.0 * n * n, 6.0 * n * n]
    }

    /// Re tr(Ĝ(s) ρ): H_eff's gradient at the time s, every beat at its phase, on the frame's ρ there.
    fn force(&self, path: &Path, s: f64, rho: &[C64]) -> [f64; 3] {
        let node = self.node(path, s);
        let contract = |rho: &[C64]| self.model.read(|b| SelectedMiddle::contract(&node, |beat| Some(self.phase(&b.value, beat, s)), rho));
        let f = match self.turns(path, s) {
            Some(p) => READS.with(|r| {
                let lab = &mut r.borrow_mut().lab;
                SelectedMiddle::unframed(&p, rho, lab);
                contract(lab)
            }),
            None => contract(rho),
        };
        f.map(|z| z.re)
    }

    /// The gradient at the middle by its beats: Ω_m = −Ω_k with Re tr(∂H_k e^{−iΩ_k t_m} ρ).
    fn tones(&self, path: &Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3])) {
        let node = self.node(path, 0.0);
        for (beat, _) in &node.gradient {
            let (value, phase) = self.model.read(|b| (b.value[*beat], self.phase(&b.value, *beat, 0.0)));
            let f = SelectedMiddle::contract(&node, |k| (k == *beat).then_some(phase), rho);
            each(if *beat == 0 { 0.0 } else { -value }, f.map(|z| z.re));
        }
    }

    /// The rows shown at the time s from the middle along the path (module documentation, # The readings on the
    /// middle): the middle node's lift M and its gradient (`Node::lift`) on the step's own model there, in the Doppler
    /// frame -- each element q ← a by its first-order change, M + s (v·∇M − i(K_q − K_a)·v M), the state framed --, the
    /// one assembly (`Selected::lifted`).
    fn readout(&self, path: &Path, s: f64, psi: &[C64]) -> Option<[f64; STATES]> {
        let node = self.lifted(path);
        let model = self.model;
        let n = self.states();
        let v = path.velocity;
        let along = |k: &V3| -> f64 { (0..3).map(|i| k[i] * v[i]).sum() };
        let (rows_turn, cols_turn): (Vec<f64>, Vec<f64>) = (model.folded_waves.iter().map(along).collect(), model.cluster.waves.iter().map(along).collect());
        let turn = self.turns(path, s);
        let key = model.id;
        Some(WORK.with(|w| {
            let mut w = w.borrow_mut();
            if w.as_ref().is_none_or(|x| x.0 != key) {
                *w = Some((key, Work::default()));
            }
            let Work { rows, .. } = &mut w.as_mut().expect("the thread's buffers").1;
            model.lifted(rows, psi, turn.as_ref(), self.t + s, &mut |g, group| {
                let (m, [dx, dy, dz]) = &node.lift[g];
                group.clear();
                group.extend(m.data.iter().zip(&dx.data).zip(&dy.data).zip(&dz.data).enumerate().map(|(x, (((&z, &gx), &gy), &gz))| {
                    if z == C64::ZERO && gx == C64::ZERO && gy == C64::ZERO && gz == C64::ZERO {
                        return C64::ZERO;
                    }
                    let slope = gx * v[0] + gy * v[1] + gz * v[2];
                    let turned = rows_turn[x / n] - cols_turn[x % n];
                    z + (slope - C64::new(0.0, turned) * z) * s
                }));
                m.rows != 0
            })
        }))
    }

    /// The light the state scatters at the middle beside the Ehrenfest force and the jumps (module documentation,
    /// # The photons).
    fn scattering(&self, path: &Path, psi: &[C64]) -> (Scattering, f64) {
        let node = self.node(path, 0.0);
        let n = self.states();
        let scattering = READS.with(|r| {
            let mut r = r.borrow_mut();
            let Reads { photons: list, moments, .. } = &mut *r;
            let classes = self.model.read(|b| photons(&node, &b.value, self.t, psi, list));
            moments.clear();
            moments.extend(classes.iter().map(|p| p.moments));
            Scattering::of_moments(moments)
        });
        let work: f64 = node.classes.iter().map(|c| class_photons_work(c, node.kicked, true, n)).sum();
        // The moments' sum over the classes (`Scattering::of_moments`): a class's sixteen numbers, the diffusion's factor's.
        (scattering, work + (16 * node.classes.len() + 64) as f64)
    }

    fn work(&self) -> [f64; 3] {
        let nodes = self.nodes.borrow();
        let Some(node) = nodes.first() else { return [0.0; 3] };
        let n2 = (self.states() * self.states()) as f64;
        [node.generator.len() as f64 * n2, 3.0 * node.gradient.len() as f64 * n2, 3.0 * node.gradient.len() as f64 * n2]
    }

    fn made(&self) -> f64 {
        self.made.get()
    }

    fn reading(&self, reads: bool) {
        self.reads.set(reads);
    }

    fn deviation(&self) -> [f64; 2] {
        self.deviation.get()
    }

    /// Ĝ_a(s) as matrices in the frame, and where asked their slope along the path at the middle: the envelopes' chord
    /// through the outer nodes ±h of the step it is asked with, each end in the frame there, and each beat's phase's
    /// derivative −iΩ.
    fn gradient(&self, path: &Path, s: f64, g: &mut [Vec<C64>; 3], slope: Option<(f64, &mut [Vec<C64>; 3])>) -> f64 {
        assert!(s == 0.0 || slope.is_none(), "the gradient's slope at the middle (the step's optical sector's)");
        let n = self.states();
        // Σ_k f_k ∂H_k of a node, f_k each beat's phase at s, or its turn −iΩ_k times it (`turned`).
        let at = |node: &Node, s: f64, turned: bool, phases: &mut Vec<C64>, out: &mut [Vec<C64>; 3]| {
            phases.clear();
            self.model.read(|b| {
                phases.extend(node.gradient.iter().map(|(beat, _)| {
                    let f = self.phase(&b.value, *beat, s);
                    if turned { f * C64::new(0.0, if *beat == 0 { 0.0 } else { -b.value[*beat] }) } else { f }
                }))
            });
            for (a, o) in out.iter_mut().enumerate() {
                o.clear();
                o.resize(n * n, C64::ZERO);
                for ((_, d), f) in node.gradient.iter().zip(phases.iter()) {
                    for (z, x) in o.iter_mut().zip(&d[a].data) {
                        *z += *f * *x;
                    }
                }
            }
        };
        let node = self.node(path, s);
        let (before, after) = match &slope {
            Some((h, _)) => (Some(self.node(path, -h)), Some(self.node(path, *h))),
            None => (None, None),
        };
        READS.with(|r| {
            let mut r = r.borrow_mut();
            let Reads { phases, minus, turned, .. } = &mut *r;
            at(&node, s, false, phases, g);
            if let Some(p) = self.turns(path, s) {
                g.iter_mut().for_each(|x| SelectedMiddle::framed(&p, x));
            }
            let mut counted = 3.0 * node.gradient.len() as f64 * (n * n) as f64;
            if let (Some((h, slope)), Some(before), Some(after)) = (slope, before, after) {
                // The envelopes' chord at the middle's phases, each end in the frame there, and the phases' turn on the
                // middle's envelopes.
                at(&before, 0.0, false, phases, minus);
                at(&after, 0.0, false, phases, slope);
                for (side, m) in [(-h, &mut *minus), (h, &mut *slope)] {
                    if let Some(p) = self.turns(path, side) {
                        m.iter_mut().for_each(|x| SelectedMiddle::framed(&p, x));
                    }
                }
                for (p, m) in slope.iter_mut().zip(minus.iter()) {
                    p.iter_mut().zip(m).for_each(|(p, m)| *p = (*p - *m) / (2.0 * h));
                }
                at(&node, 0.0, true, phases, turned);
                for (o, t) in slope.iter_mut().zip(turned.iter()) {
                    o.iter_mut().zip(t).for_each(|(o, t)| *o += *t);
                }
                counted *= 4.0;
            }
            counted
        })
    }

    /// P's explicit excited states' optical sector (step.rs, # The optical sector's noise): its ground states, γ_Q, and
    /// the light pattern's fastest beat along the path over the non-tweezer beams.
    /// The step's three nodes' forms (`step::Middle::nodes`).
    fn nodes(&self, path: &Path, h: f64, rho: [&[C64]; 3], held: Option<(&[C64], &dyn Fn(f64) -> f64)>, end: Option<(&[C64], &Path)>) -> ([[f64; 3]; 3], Option<crate::step::Forms>) {
        let forms = std::array::from_fn(|j| match held {
            Some((held, weight)) if j == 1 => self.centre(path, held, weight, rho[j]),
            _ => self.force(path, [-h, 0.0, h][j], rho[j]),
        });
        (forms, end.map(|(rho, fixed)| crate::step::forms_at(self, fixed, rho)))
    }

    fn optical(&self, path: &Path) -> Option<Optical> {
        let decay = self.model.optical?;
        let (most, least) = self.model.waves.iter().map(|k| (0..3).map(|a| k[a] * path.velocity[a]).sum::<f64>()).fold((f64::NEG_INFINITY, f64::INFINITY), |(most, least), x| (larger(most, x), smaller(least, x)));
        Some(Optical { ground: GROUND, decay, beat: if most < least { 0.0 } else { most - least } })
    }
}

impl crate::step::Dynamics for Selected {
    type Middle<'a>
        = SelectedMiddle<'a>
    where
        Self: 'a;

    fn hbar_over_mass(&self) -> f64 {
        self.hbar_over_mass
    }

    /// V_ref: the reference state's potential (module documentation, # The rest of the model).
    fn reference(&self, r: [f64; 3]) -> Quadratic {
        self.prepared(&self.reference).at(r)
    }

    fn cluster(&self, _cluster: usize) -> &Cluster {
        &self.cluster
    }

    fn cluster_of(&self, _ground: usize) -> usize {
        0
    }

    fn middle(&self, _cluster: usize, r: [f64; 3], t: f64) -> SelectedMiddle<'_> {
        let nodes = LISTS.with(|l| l.borrow_mut().pop()).unwrap_or_default();
        // The trajectory's version where one is being made on this thread; outside one the fold's own (# The tables).
        let tabled = self.tabled_here();
        SelectedMiddle { model: self, r, t, nodes: RefCell::new(nodes), made: Cell::new(0.0), tabled, reads: Cell::new(false), deviation: Cell::new([0.0; 2]) }
    }

    /// A jump's photon: on the step's middle at the jump's time and place, or at a point on a middle made there.
    fn emit<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&SelectedMiddle<'m>, &Path, f64)>, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted> {
        match on {
            Some((middle, path, s)) => middle.emitted(path, s, psi, draw),
            None => self.middle(cluster, at.r, at.t).emitted(&Path { kappa: at.kappa, velocity: [0.0; 3] }, 0.0, psi, draw),
        }
    }

    /// (ħ/m)κ²/2 + Σ_g p_g(ψ) V_g(r) (module documentation, # The rest of the model).
    fn energy(&self, _cluster: usize, at: &Point, psi: &[C64]) -> f64 {
        let kinetic = 0.5 * self.hbar_over_mass * at.kappa.iter().map(|k| k * k).sum::<f64>();
        kinetic + self.prepared(&self.populations(psi)).at(at.r).value
    }

    /// The loss energy above the escape value E_cap = 0, the potential's far value, wherever the atom is: the first
    /// passage out of the bound set H = {E ≤ 0} (SPEC §1.7). The trap's region is no part of it.
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool {
        self.energy(cluster, at, psi) > 0.0
    }

    fn readout(&self, _cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES] {
        self.rows(at, psi)
    }

    fn version(&self) -> u32 {
        Selected::version(self)
    }
}

#[cfg(test)]
mod tests;
