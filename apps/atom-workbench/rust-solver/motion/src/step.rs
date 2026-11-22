//! One step: A(τ/2) B(τ) A(τ/2) with the mid-step reference, the internal propagation along the path, the impulse, the
//! jump's root and the jump (SPEC §2.1, §1.6).
//!
//! # The motion and its splitting
//! A trajectory is (r, κ, ψ): position (µm), momentum κ = p/ħ (1/µm) and the cluster's unnormalized internal state between
//! jumps: ṙ = (ħ/m)κ, κ̇ = −⟨ψ|G|ψ⟩/⟨ψ|ψ⟩, iψ̇ = Hψ, H the cluster's generator and G = ∇H_h (`Dynamics::middle`).
//! Without the decay this is the Hamiltonian system 𝓗 = (ħ/m)κ²/2 + ⟨ψ|H_h|ψ⟩/⟨ψ|ψ⟩ on the product phase space. A step
//! splits it as 𝓗 = A + B, A = (ħ/m)κ²/2 + V_ref(r) and B the rest, and composes their flows A(τ/2) B(τ) A(τ/2):
//! symplectic in its conservative part, its own inverse, second order (propagation.md §2.1).
//!
//! # The reference (A)
//! V_ref is the second-order expansion of the fixed reference state's trap potential (`Dynamics::reference`) at the
//! step's middle p: V_ref(r) = V + g·(r − p) + ½(r − p)ᵀK(r − p). Its flow is exact along K's principal axes: with
//! y = e·(r − p), k = e·κ, curvature λ and gradient g along the axis e, ẏ = (ħ/m)k, k̇ = −(g + λy), solved over t by
//! y(t) = y c₀ + (ħ/m)k t c₁ − (ħ/m)g t² c₂ and k(t) = k c₀ − (λy + g) t c₁, the Stumpff functions c_k at x = (ħ/m)λt²
//! (`stumpff`): a rotation at ω² = (ħ/m)λ for λ > 0, a hyperbolic flow for λ < 0, free flight under the constant force for
//! λ = 0 -- one formula, entire in x (`Flow`). A step is therefore not bound to the trap's period. p is the fixed point
//! "the first half flow on V_ref(p) reaches p" (`middle`): the step's reverse takes the same reference, so the step is its
//! own inverse (Hut, Makino & McMillan 1995, the time-symmetrized leapfrog; propagation.md §2.2). Any geometry, one rule.
//!
//! # The kick (B)
//! The position stays at r_h; the internal state goes along the straight path r_h + v_m(t − t_m), the path's phases
//! exact: with ψ_s = e^{iK_s·v(t − t_m)} x_s in the cluster's Doppler frame (`Plan::waves`) x evolves under Ĥ(t), the
//! generator on the path in that frame with the frame's diag(K_s·v). The model gives it by its fields, not by a list of
//! beating matrices (`Middle`): at the middle point, once a step, everything that depends on neither the momentum nor the
//! time; per momentum the beams' Doppler weights and path rates (`Middle::path`); and at a time s from the middle Ĥ(s)
//! itself, each beam's plane wave turned by e^{ik_l·v s} (`Middle::generator`).
//!
//! One propagation, whatever the generator (`propagate`; spectral.rs). Ĥ(s) = Ĥ₀ + D(s): Ĥ₀ = G₀(0), the static group
//! of the generator at the step's middle (`Middle::generator`), holds the atom's own frequencies -- detunings, light
//! shifts, decay rates --, which may turn any number of times over the step (an explicit excited state twenty
//! linewidths off its light turns 48 rad in 0.0625 µs, and rings at that for 2/Γ after every jump); D(s) is what the
//! motion along the path and the lights' beats change: the static group's change, slow by the step's own choice, and the
//! model's exact tones (`Middle::beats`), each at its beat. The two are integrated apart:
//! - Ĥ₀ exactly: in its eigenbasis its phases and decays are factors over any width (the interaction picture,
//!   `spectral::Picture`), so the state is known at any time of the step for n exponentials and two products of n²;
//! - D through the step's three nodes alone, its middle and its two ends (two more generators of the model; none where
//!   Ĥ is the same at every time, `Middle::still`: every pair of states holds one wave vector): the static group's
//!   change sA + s²B, a tone's envelope C + sA + s²B at its exact phase e^{iνs}. An entry of Ĥ₀'s states that turns far
//!   between two nodes at its combined frequency ω_ab + ν for what changes -- where its following leaves no more than
//!   its hops would (`spectral::followed`; spectral.rs, # The split) -- follows the change (a polynomial in the time, to
//!   the second order: the generator's own eigenvectors along the path); the slower entries, with the change's own light
//!   shift, are two hops of the interaction picture a step, the first Magnus term with its integral over the entry's
//!   turn exact, and a quintic between the nodes (spectral.rs). The propagation's work is the same whatever Ĥ₀'s
//!   frequencies or the tones' beats times the step are. Which of the model's beats are tones is the propagation's
//!   choice (spectral.rs, # Which beats are tones: by what each route leaves), the rest of the generator its static group. The
//!   gradient's slow change is taken through the same three nodes (`propagate`), and so are the static group's decay
//!   (the survival's) and its gradient (the kick's); a tone is its candidate whole, its Hermitian part and its decay's
//!   beat (spectral.rs `Picture::separate`), and its forms -- its gradient's (the kick's) and its decay's (the
//!   survival's) -- are taken at its beat, as its part of the state is: on the quadrature's densities summed at
//!   e^{iνu} -- six Hermitian sums a tone, of the three nodes' Lagrange weights times cos νu and sin νu (`Sums`) -- so
//!   that what the step integrates of a tone resolves its combined frequencies, not its beat (the forms' bandwidth with
//!   the tones', spectral.rs).
//! The step's time integral -- the kick -- is read on that state at the nodes of a
//! Gauss–Legendre quadrature at the bandwidth of the forms ⟨x|·|x⟩ on a state Ĥ₀ propagates (`quadrature`: to the
//! round-off of each component's size times the step -- no credit taken for a fast component's cancelling over the
//! step, as the nodes follow the state; why not to a share, # The quadrature's tolerance), piece by piece between the step's jumps
//! (# Jumps), and the survival's with it. These nodes meet
//! no model and no hop. Their number still follows the
//! bandwidth: the kick is a quotient by the norm, whose sum of exponentials under the forms' has no closed form (below).
//! What is left is of the slow change alone (spectral.rs), and neither part of it grows with the detuning times the
//! step:
//! - the three nodes are Simpson's rule's: of the fourth order, for the generator and for the gradient alike (their
//!   cubic is odd about the middle and drops out of the step's integrals);
//! - a hop leaves the second Magnus term, of the fifth order in the step: where every pair is slow -- a folded
//!   cluster, whose own frequencies are its light shifts -- the propagation on its path is of the fourth order, state
//!   and kick;
//! - a followed pair -- an explicit far-detuned state -- leaves the third order of the following, within its square
//!   (k·v h V/ω)² at the coupling V and the pair's frequency ω: it falls with the step and with the detuning.
//! An atom at rest, or a generator still along the path, has D = 0: its state and its kick are exact.
//!
//! The step is of the second order all the same: its splitting's. The position is held at r_h through the kick, the
//! path is straight where the reference's flow bends it, and the path's momentum is predicted (below). Measured on six
//! beams' cooling in a tweezer (PGC3D, F = 2, atoms at 25 µK, τ = 0.625 µs: k·v τ = 0.25): the propagation's own error
//! on the step's path is 8 % of the step's difference from its two halves in the kick and three tenths in the state,
//! and at half the step it falls seventyfold and fiftyfold, where the difference falls fourteenfold and eightfold. The
//! pilot's companion at twice the step reads all of it (solve.rs, # The step).
//!
//! The impulse is exact in its decomposition about the state at the step's middle, held in the frame, x_m = x(t_m)
//! (= ψ(t_m), U(t_m) = 1):
//! - the shear: the conservative force at the fixed r_h on x_m, −∫⟨x_m|G(r_h, t)|x_m⟩dt/‖x_m‖². At a fixed point the
//!   gradient beats in time alone, G(r_h, t) = Σ_m G_m e^{iΩ_m(t − t_m)} over its tones (`Middle::tones`), each
//!   τ sinc(Ω_mτ/2) (`Midway::stretch`'s weights). Of a state that does not change it depends on r_h alone: the
//!   conservative kick is a shear and the step symplectic (propagation.md §2.3);
//! - what the state's change along the path adds: the evolved state's force along the path less x_m's held in the frame,
//!   −∫(⟨x(t)|Ĝ(t)|x(t)⟩/‖x(t)‖² − ⟨x_m|Ĝ(t)|x_m⟩/‖x_m‖²)dt, Ĝ(t) the gradient on the path, one quadrature of the
//!   difference on the propagation's nodes: it vanishes at every node where the state does not
//!   change -- a lone ground state in a standing wave's far light takes no error of the path's nodes, however long its
//!   step. The lag that friction and the Sisyphus effect live in;
//! - less the reference's force: + τ∇V_ref(r_h), since V_ref's force is the A flow's.
//!
//! The first two are read together (`propagate`): a mean ⟨x|G|x⟩/‖x‖² is tr(Gρ) of the state's density ρ, which the
//! model contracts without forming G (`Middle::force`). With ρ_m the middle state's, (s_k, w_k, ρ_k) the quadrature's
//! times from the middle, weights and densities, and ℓ_j the Lagrange polynomials of the step's three nodes s_j, they are
//! −Σ_m τ sinc(Ω_mτ/2) Re tr(G_m ρ_m) − Σ_j Re tr(Ĝ(s_j) Q_j), Q_j = Σ_k w_k ℓ_j(s_k)(ρ_k − ρ_m): the atom's own frequencies
//! are in the Q_j, integrated to the round-off, and the model's forms are three. Where Ĥ is still, Ĝ is one matrix and the
//! whole is its one form with ∫ρ = Σ_k w_k ρ_k. (With a decay the mean is a quotient by the norm, whose own ringing --
//! of the size Γ/Δ times the excited share -- puts parts of the quotient at twice the forms' bandwidth and beyond (# The
//! quadrature's tolerance). Taken to a share, the kick erred by a part of that ringing, measured 2·10⁻⁷ of the dipole
//! kick at twenty linewidths and Δτ = 50, 10⁻⁵ at Δτ = 200, and so did the survival's hazard, the same quotient
//! (# Jumps): at rest from the bare ground state 2.5·10⁻⁸, 3.4·10⁻⁸ and 3.5·10⁻⁷ of ln N over a step at Δτ = 10, 50, 200,
//! moving 1.6·10⁻⁵ of the hazard's integral at 200 -- a share that did not fall with the step, so that the doubling did
//! not read it (tests). Without a decay nothing does.)
//!
//! # The quadrature's tolerance
//! The nodes are laid to the round-off of each component, as the propagation takes Ĥ₀ (spectral.rs), not to a share of
//! the target. Until 2026-11-17 they were laid at the kick's share `QUADRATURE` (the old engine's rule): an error per unit
//! time that does not fall with the step and that the step's companion does not read -- a resonant two-level atom's
//! survival over 4 µs without a jump (two states, the forms' bandwidth its Rabi frequency) erred by 7.9·10⁻², 2.4·10⁻² and
//! 2.7·10⁻² of ln N = −76.6 at τ = 0.2, 0.1 and 0.05 µs, not falling, and by 1.2·10⁻⁵ at 0.025 µs, where the rule fell
//! to its least; to the round-off by 1.9·10⁻⁴, 9.6·10⁻⁷, 2.4·10⁻⁸ and 2.2·10⁻⁹. Nor is a tolerance at what the
//! propagation itself leaves enough: the kick's and the hazard's integrands are quotients by the norm, and a quotient of
//! two sums of the forms' bandwidth B holds the norm's ringing at 2B, 3B, … -- its parts at the multiples of B fall with
//! the ringing's relative size, not with the tolerance --, so that the Gauss rule's bound, which is of a component of
//! bandwidth B, does not hold for it: laid at the propagation's remainder (10⁻⁷ a step on that atom at τ = 0.2 µs), the
//! survival erred by 1.9·10⁻², the quotient's unresolved ringing; at 2B by 1.5·10⁻³; to the round-off by 1.9·10⁻⁴,
//! falling with the step as above -- the rule's margin there takes the multiples of B (its error grows as (νw)^{2m}).
//! So the rule is laid at the quotient's own bandwidth ν_Q (spectral.rs `Picture::quotient`): the forms' with the
//! norm's ringing and its harmonics to the round-off -- the quotient's terms, a numerator pair's times k ringing pairs'
//! of the eigenvectors' overlaps, their weights the launched state's and their frequencies added, cut by Chernoff's
//! bound on the series where what lies beyond weighs within the round-off --, over each piece of the step as many equal
//! Gauss–Legendre panels as leave the fewest nodes to take (`count`: one where no jump's root lies in the piece, more
//! where one does and the root's panel is taken again), each of the fewest nodes to the round-off (`fewest`), every
//! rule made in O(m) to the round-off once for the process (`special::gauss_legendre`, `rule`). Measured against brute
//! force on the six scenes' plain pieces, the pieces after a jump and the bare starts at τ = 0.3256 and 0.948 µs
//! (step/tests/r1prop.rs `r1prop_quadrature`; tilt2 at 0.3256): the sums, the hazard, the kick and the hazard to a time
//! inside the piece (a root's span) within 3·10⁻¹⁴, laid as one panel and as a root's piece's three to eight, where the
//! panels at the forms' bandwidth before left the norm's harmonics at 2B and 3B, up to 10⁻⁹ (motion3d_status/rewrite/
//! carrier.status.md); ν_Q 2.0 to 3.2 times the forms' bandwidth on the explicit clusters, 5.6 to 6.4 on GMD1C's ground
//! states, the nodes 1.3 to 2.0 times the panels' before (2.1 to 2.5 on GMD1C, of 9 to 27 nodes). The integration's
//! time -- the stretch less its picture's making, along a walk on one thread (`r1prop_integration_time`) -- 1.1 to 2.1
//! times the strides' before on PGC3D, TET, R4, GMD1C and tilt2, a fifth to nine tenths of it on tilt0.
//!
//! Why a quadrature and not a closed form. On the unnormalized state every form is a sum over the pairs of states of
//! a slow factor times e^{−i(λ_b − λ_a*)s}, whose integral over the step is a closed form pair by pair (the moments
//! ∫sᵖe^{zs}ds; spectral.rs). The kick is not that: it is the form over the norm ⟨x|x⟩(s), itself such a sum -- its
//! pairs' weights the eigenvectors' overlaps, which a decay makes non-zero, e^{−Γs} for each decaying state and a
//! ringing at the detuning wherever the state holds an explicit state's amplitude (after every jump; and at the
//! following's third order on every step of a moving atom). One over that sum is no sum of exponentials: the
//! quotient's integral has no closed form, and its series in the ringing costs n⁴ a term. Resolved on nodes it costs
//! n² a node -- so the nodes stay, as many as the bandwidth asks, each now a state at a time and no hop. A closed form
//! with the norm's ringing left out would err by that ringing times the fast pairs' share of the force -- by estimate
//! 10⁻³ of the dipole kick on the step after a jump at twenty linewidths, where the quadrature leaves 10⁻⁵ --, and
//! where the norm rings by tenths of itself (a light shift of a third of the detuning: the tests' Δτ = 10) no order
//! of that series is enough.
//!
//! A generator's scalar part, tr/n -- the trap's light shift and the decay every state shares --, is a phase and a
//! factor on the whole state: it goes with Ĥ₀'s eigenvalues, exactly, and the exponentials take it out
//! (`linalg::exp_apply`), so that their work and their rounding go with the states' differences alone.
//!
//! Held in the frame, not in the lab: a dressed state's coherences co-rotate with the light's phases along the path, so
//! its force along the path is the frame's, constant where Ĥ is static (a running wave's radiation pressure, steady on the
//! moving atom); held in the lab its coherence would slip against the light at K·v and its path force average away
//! (sinc(k·vτ/2)), counting the steady force twice -- a running wave's pressure came out doubled. Shear and path then differ
//! only by the path's beats, the light pattern's spatial modulation the atom crosses, which is what the shear holds at r_h.
//!
//! The middle state, not the start's: the step's reverse propagates back through the same x_m, so the decomposition is the
//! same both ways and the step its own inverse with the internal state changing too. About the start state the reverse
//! took its own start, x(t₀ + τ), and Q(x) = ∫[F(r_h, x) − F(path, x)]dt, the midpoint rule's O(τ³(v·∇)²F) on that state,
//! differed between the two: a two-level atom in a far-detuned standing wave missed its start by two hundred times the
//! step's own error (tests). A one-dimensional PGC read the same T on the middle state as on the start state
//! (propagation.md §2.3).
//!
//! # The path's momentum
//! v_m = (ħ/m)κ_m, κ_m the momentum at the step's middle time: the mean of the momenta before and after the kick,
//! κ_h + Δκ/2 (propagation.md §2.3). It depends on the kick the step makes, so it is predicted -- with no model
//! evaluated for it (r11.status.md, its recommendation E): the generator and the path are both taken at κ_m, one middle
//! and one evaluation of the model a step. κ_m = κ_h + K/2, K the kick of the start state held where the model last
//! evaluated it -- its held forms (`Forms`), the trajectory's: the static form over τ and each tone's at its phase over
//! the step, τ sinc(ντ/2) e^{iν(t_m − t₁)} (`predicted`, a function of what the trajectory carries). A form is what the
//! kick integrates, the model's force less the reference's gradient, P = Re tr(Ĝρ) − ∇V_ref (# The reference: the
//! reference's own flow is exact, the kick takes the rest), read at a middle point, where the model's generator and the
//! reference's expansion are at one point by construction -- whatever a model holds of its potential along a path (a
//! model may hold a trap the reference takes exactly as it is at the middle over the whole step: the kick integrates it
//! to the same τ∇V_ref(r_h) either way). The forms are those of the end state ψ₁ at the last step's middle r_h, its
//! density contracted with the step's own kick (`Middle::nodes`: one more density, its static form and each tone's
//! complex amplitude), less the step's τ∇V_ref(r_h); at the trajectory's start the model's at its own point
//! (`Dynamics::started`, the same contraction there); after a jump at a step's end into
//! another cluster, that cluster's at the jump's point (`Dynamics::forms`).
//! - The forms against the force the next step needs, its own average over its path about r_h′ = r_h + v_m τ + O(τ²):
//!   the state is the trajectory's own at t₁, the position O(τ) from r_h′, and the generator is taken at κ_m of the last
//!   step, O(τ) from the state's momentum: P_E = P(z₁) − τv·∇P − (τ/2)κ̇·∂_κP + O(τ²) against P(z₁) + (τ′/2)Ṗ + O(τ²).
//!   κ_m errs by O(τ²), the kick's sensitivity to it, (τ/2)|∂Δκ/∂κ_m|, is O(τ), so the kick and the state err by O(τ³),
//!   the scheme's own local order (r11.status.md §2(a); the pilot's companion at twice the step reads it with the rest of
//!   the step's error: each level carries its own forms). A tone is carried by its amplitude, so the next step weighs
//!   it by its own beat over its own width: the force at one time alone would put κ_m off by (τ/2)|F_tone| at ντ ≳ 1.
//!   What is read at one point against a reference at another holds the reference's own force between them, K·δr: a
//!   model's force at the end node against the next middle's reference put κ_m off by (τ/2)K·v_mτ, (ωτ)²/2 of κ in a
//!   trap of ω, which no step resolves (wire.status.md, wire4 item 1).
//! - The reverse step takes the forms the forward step left, at its own middle (a step and its reverse share r_h): its
//!   path's momentum is κ_h + Δκ − (τ/2)P_E(z₁) against the step's κ_h + (τ/2)P_E(z₀). They differ by the forms'
//!   trapezoid against the kick and by the start's forms' position, O(τ²), and by (τ²/2)κ̇∂_κP, the force's dependence
//!   on the velocity; the kick takes v_m by its path's odd moments: a step and its reverse miss the start beyond the
//!   scheme's order (tests: from a dressed state 2.6e-6, 3.0e-7, 2.1e-8 at τ = 0.2, 0.1, 0.05 µs), and by nothing where
//!   the kick does not depend on the path (one internal state: a conservative trap's step is its own inverse to
//!   round-off).
//! - Where the step does not resolve the internal state (a jump's bare state relaxing in 1/Γ ≪ τ) the force changes by
//!   O(1) over the step and the path's velocity errs in first order over that step; the companion reads that too. A
//!   jump inside a step: the step's κ_m was predicted from the state before it, as it was from the start state; the next
//!   step's forms are the jumped state's at the end.
//!
//! # Jumps (SPEC §1.6)
//! Between jumps ‖ψ‖² is the probability N that no jump happened, and the exact flow only lowers it, at the rate of the
//! generator's decay on the state's direction x̂: d ln N/du = −h(u), h = x̂†Γ(u)x̂ ≥ 0, Γ = i(Ĥ − Ĥ†) = ΣJ†J (the jumps'
//! product, by the model's construction). The propagation (spectral.rs) is no such flow: its state goes through a change
//! of basis that is not unitary, its hops' Magnus terms are taken in a basis that is not orthogonal, and its norm rises
//! where the flow's falls -- the map's largest singular value 1.0095 on R4_LIN_2's step at the engine's τ, up to 70 on
//! random long steps, 10²⁰⁰ compounded over a two-level atom's steps at 10 Γ (motion3d_status/rewrite/norm.status.md,
//! int.status.md) --, which made an atom wait too long for its next jump and a root taken as monotone miss the first
//! crossing. So the step takes the survival from the law, not from the propagation's norm: ln N(u) = ln N(u₀) − ∫h, h on
//! the propagation's direction, Γ(u) the decay of the generator the propagation takes -- the static group's through the
//! step's three nodes, its parabola Σ_j ℓ_j(u)Γ_j, and each tone's at its exact phase, its envelope's parabola times
//! e^{±iνu}; its one matrix where Ĥ is still --, and the trajectory's state is √N times the direction.
//! - The integral is the kick's quadrature's: each node's density is in the three nodes' Lagrange sums S_j already
//!   (`accumulate`) and in each tone's at its phase, C_j and S′_j of the weights ℓ_j(u) cos νu and ℓ_j(u) sin νu
//!   (`Sums`), and ∫h over what they hold is Σ_j tr(Γ_j S_j) + Σ_tones Σ_j [tr(Γ_j^c C_j) + tr(Γ_j^s S′_j)] (`hazard`),
//!   Γ_j^c and Γ_j^s the tone's decay at the phases 0 and π/2 at the node j -- the decays of G⁺ + G⁻ and of i(G⁺ − G⁻):
//!   e^{iθ}G⁺ + e^{−iθ}G⁻ is cos θ times the first and sin θ times the second -- no work a node; a trace of the laid
//!   decay at each panel's end (`panels`). At the quadrature's nodes this is x̂†Γ(u_k)x̂ of the generator's own decay
//!   where the envelopes are constant along the path, Σ L(u_k)†L(u_k), whatever the tones' beats: the decay's beats are
//!   neither aliased nor averaged by the three nodes (DESIGN §12.4).
//! - N does not rise where the decay is positive semidefinite at the quadrature's nodes: the weights are positive and
//!   each node's h a form of it. The model's decay is the jumps' product exactly, from its near tables to their error;
//!   along the measured scenes' steps -- PGC3D, GMD1C, S3, S3c, R4_LIN_2, S1, R3_DOPPLER, TET and a two-level atom at
//!   10 Γ, from the tables, exactly and as the step's parabola, at the nodes and at 21 times across -- it is positive
//!   definite everywhere, its least eigenvalue 6·10⁻⁷ of its largest at the least (GMD1C's dark ground states), so that
//!   N rises by nothing but a sum's rounding (step/tests.rs, `a_scene_s_decay_is_positive_along_its_steps`). (The
//!   parabola of a decay that interferes along the path -- a channel of several plane waves near its zero -- could dip
//!   below zero between the nodes by its own third-order error; no scene's does.)
//! - N reaches the trajectory's threshold θ first in the first panel whose end is at θ or below (N is monotone); a jump
//!   is made there, at the root u* of ln N(u) = ln θ to ln N's rounding, ln N inside the panel its start less ∫h over
//!   spans of it by their own quadratures on the propagation's states (`Picture::at`), stepped from below at the
//!   hazard's rate (`root`): the spans below the root are the kick's part of the panel as far as the root, so that the
//!   root costs the panel's nodes before it and a few more, as laying them again did. The states kept inside that
//!   panel -- the middle, a time read -- stand where they lie before the root; the others are met again after the
//!   jump.
//! - Where to look is the propagation's norm's to tell: it is the survival to the propagation's accuracy and costs n a
//!   stop. Once it is at θ at a stop, the panel's nodes past it are not taken and the root is sought from the panel's
//!   start with that stop's time as its first try -- the step does not finish a panel it jumps in, nor lay its nodes
//!   again; where the survival does not reach θ in the panel after all (a norm that fell too fast), the root's spans
//!   are the panel's kick to its end, and the step goes on. A crossing the norm misses (a norm that rose) the panel's
//!   end reads, as above: the survival alone decides.
//!
//! The jump is made from the state at the root, √θ times the propagation's direction x̂(u*) there, and **at the path's
//! own point there** -- the position r_h + v_m u*, the momentum κ_m the
//! generator is taken at: the photons, the kick and the landing (`jump`) are the model's at the point whose decay timed
//! the jump. The jump's time, its state and its point are one propagation's; rate × kick is then the scattering force
//! at one point of the step's path, the path the step's own force is read on, and what is left of it is the step's
//! error, the same both ways in time. (Read apart -- the hazard on one path, the kick at another place of the orbit,
//! always to one side --, a kick that depends on the position turns into a force along the velocity: timed by a step
//! of the nominal length and made at the end of a shorter one, a held atom at a standing wave's antinode at ω τ = 3
//! lost all its heating along the wave to such a friction, R10.) A jump draws from its own stream
//! (`random::Stream::jump`) in a fixed order: the next threshold, the class, the polarization, and the direction's
//! rejection loop last, so that the loop's length moves no other draw.
//!
//! The step goes on after the jump (`Midway::attempt`): the propagation is linear in the state, and the generator's
//! picture -- its middle, its decomposition, its following, its slopes -- knows the atom's place and velocity, not its
//! internal state. So the state the jump lands in is launched at u* on the same picture
//! (`spectral::Picture::launch`: a factorization and two hops over what is left of the step), and the step is one
//! propagation however many jumps it holds:
//! - the kick's quadrature is taken piece by piece between the jumps (the state is discontinuous there): a piece's
//!   nodes are laid from its start to the step's end, and where the survival reaches the threshold before that, the
//!   panel it reaches it in is taken again from its start up to the root by the root's own spans (`fallen`) -- states
//!   at nodes, no hop;
//! - a jump's kick δκ enters the reference's flow where it is made, exactly: a kick at the time u from the middle ends
//!   the step as the kick δκ cos ωu and the displacement −(ħ/m) δκ sin(ωu)/ω at the middle do, per principal axis (the
//!   identity of `spreading` and of `Segment`);
//! - a jump that lands in another cluster goes on in that cluster's generator at the same middle (`Dynamics::middle`),
//!   on the same path -- the middle the jump's landing was made on (`Places`) --: its picture is made there, its part
//!   of the kick is its own forms'.
//! What a jump inside a step leaves, each once a jump (the jumps' number is the physics', so each is an error of the
//! run of its own order in τ; h = τ/2, a the acceleration, f the force beside the reference's):
//! - the photons' Doppler weights are the path's, κ_m for κ_m + a u*: first order and odd in u*, so of zero mean over
//!   the jumps' places in their steps; the position is the path's, a(u*² + h²)/2 off: second order;
//! - the lights' amplitudes are the middle's: B is taken at r_h, and along the path the plane waves' phases alone
//!   turn. What the state does within 1/Γ of a jump -- its excited amplitudes rising to what they follow, a bare
//!   state's fast part going -- takes its kick K with every light's envelope and gradient at r_h for r_h + v_m u*:
//!   u* (v_m·∇)K off, first order and odd in u*, of zero mean. (At its largest on a state no trajectory holds, at a
//!   step's start: thirteen random amplitudes on GMD1C's eight ground and five excited states lose the excited 40 %
//!   in 1/Γ = 0.03 µs under the trap's force on the excited states, a kick of 1.5 % of the momentum whatever the
//!   step; a step of 0.5 µs then differs from its two halves by (τ/4)(v·∇)K = 9·10⁻⁴ of the momentum, falling 2.2
//!   and 2.7 on halving and linear in the speed, and by nothing of that at rest, without the decay, with the excited
//!   states' force the ground states', or on the states of a run -- an excited share of 3·10⁻⁴, a difference of
//!   8·10⁻⁵ of the momentum falling 7.7 to 7.9; tests, `a_scene_s_step_error_by_the_split`.)
//! - the state's errors inside a step -- ∂_κH·a(u² − h²)/2, zero at its ends (# The path between a step's ends) -- no
//!   longer cancel across the jump: their commutator with the jump, second order, of the sign of a·∂_κH;
//! - the force changes at the jump and is still one kick at the middle: the position at the step's end errs by
//!   (ħ/m)Δf(h² − u*²)/2, second order -- with the smooth part (ħ/m)(τ³/12)(f′ + γΔf) a step, γ the jumps' rate: the
//!   jumps' share of the force's rate of change;
//! - the scattered light's force and diffusion are the middle state's for the whole step, whichever side of the jump
//!   the middle lies: first order, of either sign by the side (zero mean);
//! - **the kick is not on the path until the step's end**: for the rest of the step the generator is read at κ_m, not
//!   κ_m + δκ -- a Doppler shift of k·δκ(ħ/m), at most 2ω_r, against the line's |Δ + iΓ/2| --, and the force's answer
//!   to the kick begins with the next step, half a step late on average. First order. The emission's direction has no
//!   mean; a fold's absorbed mean kick has one where the light is one-sided: (γτ/2)(2ω_r/|Δ + iΓ/2|) of the light's
//!   response. Through the friction every jump's kick stays undamped half a step longer: the energy is high by what
//!   the jumps' kicks put in over half a step, E_r/3 a jump a step as a temperature (0.06 µK for Rb; 0.13 % of PGC3D's
//!   13 µK at its 0.28 jumps a step). The Brownian part has had the same term since κ_m is predicted without the path's
//!   increment (# The absorbed light's momentum). The pilot's companion at twice the step reads all of these (solve.rs,
//!   # The step).
//!
//! # The absorbed light's momentum (SPEC §1.6)
//! Per unit time the light an atom scatters gives its momentum a mean and a second moment that are smooth in the
//! position and finite where the light interferes away -- the local Kramers–Moyal coefficients Σ Im⟨u|∇u⟩ and
//! Σ Re⟨∇u|⊗|∇u⟩ of the jumps' amplitudes u = Jψ, neither divided by the jumps' rate (model.rs `Moments`). They are
//! delivered so that no increment of the momentum is a quotient by a rate:
//! - a jump carries the emitted photon's recoil −k_φ n and its class's conditional mean as far as one absorbed photon's
//!   momentum goes (`Emitted::kick`: at most k_φ + k_c in all);
//! - a step carries the rest at its middle, on the state x_m there (`Middle::scattering`): the force F_s the jumps do
//!   not carry, over τ, and the diffusion 2D they do not carry, as the step's part of the trajectory's Brownian path
//!   (`Brownian`) through a factor Λ of it, Λ Λᵀ = 2D (`factor`): in free flight κ += τ F_s + Λ ΔW and r −= (ħ/m) Λ ΔZ,
//!   ΔW = ∫dW the path's increment over the step and ΔZ = ∫(s − τ/2)dW its first moment about the middle. With ΔZ the
//!   kick at the middle moves the step's end as the force Λ dW/ds spread over the step moves it, (ħ/m)∫(τ − s)Λ dW: a
//!   step and its two halves on the same path (`Brownian::split` at the middle: ΔW = ΔW₁ + ΔW₂, ΔZ = ΔZ₁ + ΔZ₂ +
//!   (τ/4)(ΔW₂ − ΔW₁)) then end together but for the scheme's own error, which is what the run's two levels differ by
//!   (adapt.rs, # The Brownian path: the same trajectory at twice the step is on the same path). Without it they would
//!   end (ħ/m)(τ/4)Λ(ΔW₂ − ΔW₁) apart, a difference of zero mean and of order τ^{3/2} that is no error of the scheme.
//!   In the reference's flow (`spreading`): along a principal axis of the rate ω (ω² = (ħ/m)λ) a force spread over
//!   the step ends as the kick ∫cos(ωu)Λ dW and the displacement −(ħ/m)∫(sin ωu/ω)Λ dW at the middle (u the time from
//!   it), uncorrelated, of the variances (τ/2)(1 + c₁(x)) and (τ/2)(1 − c₁(x))/ω² an axis of Λ Λᵀ, x = ω²τ² and c₁ the
//!   Stumpff function. The step takes ΔW and ΔZ at these variances, per principal axis: the energy the diffusion puts
//!   in is then (ħ/m) tr(2D) τ/2 in the mean at any ωτ, as a white force's is in a harmonic trap. (The free flight's
//!   lever puts (ωτ)²/12 of it more into the potential energy: three quarters more at the ladder's top rung.)
//!   The path's momentum κ_m is predicted without the path's increment (its mean is none): the propagation, the
//!   survival and so the jumps' times do not depend on the deviates.
//!   F_s and 2D are those of the state at the step's middle, and so is the optical sector's noise beside them (below).
//!
//! # The optical sector's noise (model/optical.rs; refs/diffusion/DESIGN.md)
//! Of the first-order correlation of the force, within(ψ) = Re Tr[𝓕 𝓡 X_ψ], the trajectory's state drops what the
//! Ehrenfest force it carries does not hold (D_x = D_E + E[within]). The part of an explicit cluster's optical sector --
//! every entry with an excited index, and the ground trace: it relaxes at Γ/2 or faster -- is local, and the trajectory
//! takes it as a white noise of its rate within_Q,ab = −Re⟨K_a x|Ĝ_b x⟩/‖x‖² + ⟨K_a⟩⟨Ĝ_b⟩ of its state (`within`;
//! model/optical.rs, # K_Q). The rest passes through the ground manifold's own state, which the trajectory keeps. A
//! folded cluster's fold holds that part as its photons' local second moment (above): nothing of this is asked of it.
//!
//! The momentum takes the noise's covariance over a step, C = 2∫sym within_Q dt, and of C nothing but its mean: the
//! increment Λ ΔW has E[Δκ Δκᵀ | C] = C, ΔW independent of all the step makes. A step takes the middle state's rate
//! for the whole step, C = |τ|(W + Wᵀ) of W = within_Q(x_m), as it takes the scattered light's 2D: the state at the
//! middle is the trajectory's state at that time -- its jumps are timed by its own thresholds, on no clock of the
//! steps --, so E[τ within_Q(x_m)] = τ m(t_m), m(t) = E[within_Q(ψ(t))] over the trajectories, and the steps' sum is
//! ∫m dt but for the midpoint rule's τ³m″/24 a step, m changing on the ensemble's slow time scales: the second order,
//! as K and Ĝ taken at the step's middle point are. What was not taken:
//! - the integral over the step's own quadrature, ∫within_Q(x(u))du on the kick's nodes. The product ⟨K_a⟩⟨Ĝ_b⟩ is
//!   quadratic in the state, no form of the kick's density sums, and asks its 3 + 9 forms at every node -- as many
//!   nodes as resolve the explicit states' own frequencies (S3: some 420 a step) --: it made S3's step three times its
//!   work (512 atoms × 60 µs at τ = 0.3: ×3.0). Its lesser variance changes no mean: over the same 78 451 steps the
//!   middle's sample read the integral's mean tr C to 0.6 % (2σ), its tr C negative on 4 % of the steps, the
//!   integral's on none (they are kept pending, below);
//! - the product of the step's mean forms, ⟨K⟩̄⟨Ĝ⟩̄ of the density sums: it drops ∫δK δĜ, the two forms' covariance
//!   inside the step. The fast ringing between the manifolds adds little to that (a pair of Ĥ₀'s states turning at
//!   ω ≫ γ holds K_kl ≈ −iĜ_kl/(ω + iγ), a quarter turn from Ĝ_kl: its product's mean is of γ/ω² -- for multilevel
//!   clusters a hypothesis), but a jump's transient over 1/Γ ≪ τ is no ringing: on S3 the mean forms' product read
//!   −15.6 for the integral's −57.0, of the whole 517: 8 % high.
//!
//! within_Q is no positive form of a state, and C's negative excursions are kept, not clipped: C with what was pending
//! is split by its eigenvectors (`split`), its positive part drawn with the scattered light's 2D through one factor on
//! the same Brownian path, the rest pending to the next step and across the jumps -- the drawn and the pending add up
//! to the sum of the C. What is pending at a trajectory's end is booked (ledger.rs `optical`), and so is the static K's
//! locality (below).
//!
//! A middle whose cluster makes no K -- a block's Schur form whose QR iteration does not converge, a singular ℳ, a
//! number that is none (model/optical.rs `Unmade`) -- has no noise to draw, and none is dropped in silence: the step is
//! refused (`Refused::Optical`), the trajectory ends there with no outcome, and the result tells how many did and why
//! (solve.rs `Counts`). It is not to happen -- Q relaxes at about Γ/2 or faster, so ℳ is regular -- and it did not on
//! the measured scenes: none in 25.2 million makes and 0.71 million prices' Schur forms over S1, S2, S2Q, S3, S3c, e038,
//! m20, R4_LIN_2 and a two-level atom at 10 Γ (motion3d_status/rewrite/int.status.md). A step taken shorter to step past
//! it would hide what it says about the model.
//!
//! K along the path. The static K of the middle -- the resolvent of the generator held still there -- is wrong at the
//! first order on a moving atom, whose couplings turn within the sector's memory 1/γ_Q: held at rest it misses a
//! running wave's Doppler shift (two-level, kv = Γ/10, Ω₀ = 2Γ, Δ = −3Γ: 1.044 of the master equation); held in the
//! cluster's Doppler frame, where a running wave is still and exact, it holds a second beam's coupling at the first's
//! frequency, 2kv off (a standing wave there: 0.888). So K is taken to the first order along the path, exactly: with
//! 𝓛(s) = 𝓛₀ + s𝓛₁ and F(s) = F₀ + sF₁ on the path in the frame (s the time after the middle),
//!
//! ```text
//!   K = ∫₀^∞ 𝒰_Q†(s)[F(s)] ds = K₀ + 𝓡_Q†𝓡_Q†(F₁ + 𝓛₁†K₀) + O(s²),   K₀ = 𝓡_Q†F₀,
//! ```
//! (∫ s e^{𝓛₀s} ds = 𝓡², and the propagator's first order ∫∫ e^{𝓛₀w} u𝓛₁ e^{𝓛₀u} = 𝓡𝓛₁𝓡²): two more solves an axis
//! on the same factorization, 𝓛₁†K₀ = i(H₁†K₀ − K₀H₁), and the generator's and the gradient's slopes H₁, G₁ at the
//! middle through the step's two ends -- the parabola the step's own propagation takes. A still cluster has none and its
//! K₀ is exact. What is left is of the second order: a Q mode of rate λ (Re λ ≥ γ_Q) answering a coupling that turns at
//! ω responds as 1/(λ − iω), and 1/λ + iω/λ² misses it by ω²/(λ²(λ − iω)), at most (ω/γ_Q)² of 1/λ -- for normal modes;
//! for a non-normal Q𝓛Q that bound is a hypothesis. It is booked: ε_Q = (ν/γ_Q)² of each step's C, ν = max |(k_l −
//! k_m)·v| the pattern's fastest beat over the cluster's beams (none where the cluster is still). On the two-level atom
//! at kv = Γ/10 (tests; σ 0.016): the running waves read 0.999–1.018 of the master equation (still in the frame: ε_Q
//! none, the error statistics alone); the standing waves at Ω₀ = 2, 5, 10Γ read 1.063, 1.053, 1.049 on resonance and
//! 0.998, 1.037, 1.084 at Δ = −3Γ -- the frame's static K 1.024, 1.034, 1.039 and 0.888, 0.921, 1.048 --, within 0.53 of
//! ε_Q = 0.16.
//!
//! The work is one K a step whose middle state lies in an explicit cluster (`optical`), in the trajectory's workspace
//! (`Scratch`): the generator at the middle and its slope through the step's ends are the picture's own, which the
//! propagation made (`Picture::slope`); the gradient's matrices and, where the cluster beats, their slope along the path
//! in one pass of the model's forms (`Middle::gradient`); the blocks' Schur forms, ℳ made at once and factored, and the
//! sides' solves (model/optical.rs, # The solve; S3: 36 reals where the whole sector is 96); and twelve forms
//! (`Sector::forms`) -- which the stretch's `Price` counts as made. It is made anew at every middle: a factorization
//! reused along the path as an iterative refinement's preconditioner contracts the error by the light pattern's change
//! over a step -- S3's standing wave turns 0.26 rad a step at 30 µK -- for an iteration that applies the operator and
//! solves, a third of the whole sector's factorization (measured on its 96 reals): more than the factorization to its
//! own accuracy. Nor is K a property of the field the trajectories could share (a table, as the near forms'): it is of
//! the cluster's generator at the point -- the near forms times every beam pair's product there, the light pattern's
//! phase within λ/2 --, of the time where tones beat, and of the velocity to all orders through the Doppler frames (a
//! still cluster's K is exact at any speed); and a table's read of its 12 n² numbers at 4³ nodes would cost more than
//! the solve (S3: some 93 thousand complex multiply-adds against 44).
//!
//! # The path between a step's ends
//! A trajectory is read at times that are no step's end (an output, a cut of T's window: adapt.rs, # The reads), and
//! the read is of the scheme's order there too. What a step knows between its ends:
//! - The internal state at any time: its propagation has x(u), u the time from the middle, on the straight path through
//!   the middle (spectral.rs), and ψ_s = e^{iK_s·v_m u} x_s. A time read is no stop of the propagation: the step's
//!   stops and nodes, its jumps' roots and its count are what the step resolves alone, and a read takes the state
//!   from the propagation as it passes -- the state there (`Picture::at`), after the jumps before it and before those
//!   after it (`looked`) --, its norm the propagation's there: a reading is read as
//!   ρ/tr ρ alone, and nothing of the survival is integrated for it. So a trajectory is the same to the bit whatever
//!   times it is read at, and a read is the same whatever else is read (adapt.rs, # The reads). That path lies ½au² + ½a_B h²
//!   from the atom's (a its
//!   acceleration, a_B the kick's part of it, h = τ/2) and has the velocity v_m for v_m + au: the state at u errs by
//!   ∫(∇H·d)du = O(τ³) and, where the generator depends on the momentum it is taken at (κ_m), by ∂_κH·a(u² − h²)/2 =
//!   O(τ²), zero at the ends.
//! - The position and the momentum: the splitting's own path -- the reference's flow, the kick at the middle, the flow
//!   again -- is right at the ends and wrong inside in the first order: before the middle the momentum lacks the
//!   kick's force over the time gone, after it it has the rest of the step's too soon, a sawtooth of half the kick.
//!   Over a whole step that drops out of ∫κ and, to τ², of ∫κ²; at a time inside it does not. So the kick is spread
//!   over the step (`Segment`), and a jump's own kick is where the jump is: the path goes on from it under the same
//!   force with the momentum the jump leaves (`Segment::after`), and ends at the step's end as the step does. The reference's flow is affine: a force f(u) acting through the step ends it as the
//!   kick ∫cos(ωu) f du and the displacement −(ħ/m)∫(sin ωu/ω) f du at the middle do, along each principal axis (ω² =
//!   (ħ/m)λ; the same identity as the diffusion's `spreading`). The force linear in the time with the step's kick and
//!   displacement for these two is the step's; the flow under it (`Flow::driven`) goes from the step's start to the
//!   step's end exactly, and between them it is
//!   - for the kick's deterministic part the force held constant over the step: the momentum at u errs by
//!     f′(h² − u²)/2, f′ the force's rate of change -- O(τ²), zero at the ends; the position by its integral;
//!   - for the Brownian part the path's mean given the two moments the step takes of it: in free flight
//!     E[W(s) | ΔW, ΔZ] = sΔW/τ − 6s(τ − s)ΔZ/τ³ is that flow exactly. What the two moments do not tell is the
//!     bridge's rest, of the variance τσ(1 − 3σ), σ = (s/τ)(1 − s/τ), through Λ: none at the ends, τ/12 at the most.
//!     A state read at a time takes it by a draw (`Segment::place`), a window's ∫κ² in the mean (`Segment::moments`).
//! One path for every time of the step; no other is kept. A read's error is the scheme's accumulated one at the
//! step's start and this one term of its own step, which is not accumulated: second order.
//!
//! What was not taken: the kick's quadrature up to the time read has the same f′(h² − u²)/2 (the shear is taken at the
//! fixed r_h: a held state's force in a standing wave is constant over the step in the scheme) for a third of a step's
//! work; a shortened step from the step's start as a side branch is of the third order and costs a whole step a read.
//!
//! # The step's momentum
//! What a step does to the momentum has one decomposition (`Step::momentum`), read off the momenta the step makes in
//! turn -- its start κ₀, the middle the first half flow reaches κ_h, the middle kicked without its jumps κ_k and with
//! them κ_j, its end κ₁: the reference's flow (κ_h − κ₀) + (κ₁ − κ_j), which carries V_ref's force −∇V_ref; the kick
//! κ_k − κ_h -- the propagation's impulse, the reference's +τ∇V_ref(r_h) (# The kick: the kick is the force less the
//! reference's), the scattered light's force and the Brownian push --; and the jumps κ_j − κ_k, each jump's kick as it
//! enters the flow at the middle (δκ cos ωu, # Jumps). The parts are differences of one chain of momenta, so they add up
//! to the whole, κ₁ − κ₀, to rounding, and a reader that takes the whole first and the parts as they are can miss no
//! term: the kick alone is not the force on the atom wherever the reference has a gradient (a held atom off a tweezer's
//! centre read its kick and jumps alone 46 % short of the master equation's mean force, the trap's pull being in the
//! flow; motion3d_status/rewrite/restforce.status.md).
//!
//! # The domain (SPEC §1.8)
//! Whether the model's folds hold where the atoms go is measured, not proven: the units' errors are read at the
//! pilot's states (solve.rs, # The model). A step asks its domain one cheap thing (`Domain`), from what it has anyway.
//! - The runaway (`Domain::runaway`): a fold far outside its validity is wrong without bound -- a drive folded at the
//!   atom whose Doppler series diverges scatters as (kv/Δ)², the released runaway's 10⁵ jumps/µs against Γ/2. Every
//!   stretch reads the hazard of its first state, x̂†Γx̂ at the decay laid for the step (`hazard`): the rate at which the
//!   state scatters, explicit excited states and folds alike. No state of the atom scatters faster than its fastest
//!   excited state does wholly excited, so a hazard above that is a fold beyond any state of the atom: the step is
//!   refused (`Refused::Runaway`), the trajectory stops there and is booked whole from then on (solve.rs `run`).
//! The step asks no edge of the light: the propagation integrates every beat of the generator at its combined
//! frequencies (spectral.rs, # Tones), so no step aliases one; a step long against the light's change is inaccurate,
//! not unstable -- the state goes on under the envelopes' parabolas, as bounded as the true generator, the survival
//! falls by the decay the step takes and never rises (# Jumps) --, and how inaccurate is the check's to read
//! (step/tests.rs: an atom past its standing wave's resolution).

use crate::decide::COMPLEX;
use crate::linalg::{self, inner, norm_sqr, C64, Mat, LANES};
use crate::model::optical::{Sector, Unmade};
use crate::model::{Cluster, Point, Quadratic, Scattering, STATES};
use crate::random::Stream;
use crate::spectral::{Beat, Picture, Refusal};
use crate::units::{rounding, UNIT_ROUNDOFF};

/// What one step needs from the model (SPEC §2.1, §1.6): defined here by its consumer, implemented by the model for its
/// `Model` with the run's lights and the reference state. Each method is one model function; the integrator defines
/// none of these quantities.
pub trait Dynamics {
    /// A cluster's generator at a position and a time (`Dynamics::middle`).
    type Middle<'a>: Middle
    where
        Self: 'a;

    /// ħ/m (µm²/µs): the velocity of κ is (ħ/m)κ, its kinetic energy (ħ/m)κ²/2 (rad/µs).
    fn hbar_over_mass(&self) -> f64;

    /// The reference potential's second-order expansion at r (SPEC §2.1): the fixed reference state's trap potential --
    /// the trajectory model's prepared potential of the reference state (model/selected.rs, # The rest of the model). The
    /// splitting is exact for any reference; this one moves the trap's harmonic part, and its
    /// gradient and curvature at the step's middle, exactly.
    fn reference(&self, r: [f64; 3]) -> Quadratic;

    /// A cluster: its states and its Doppler frame K_s (model.rs `Cluster`).
    fn cluster(&self, cluster: usize) -> &Cluster;

    /// The cluster a ground state belongs to.
    fn cluster_of(&self, ground: usize) -> usize;

    /// The cluster's generator at the position r and the time t with every light on, for every momentum and along every
    /// straight path through the point (`Middle`): made once a step, at its middle.
    fn middle(&self, cluster: usize, r: [f64; 3], t: f64) -> Self::Middle<'_>;

    /// A jump's photon of the cluster's state ψ (unnormalized) at a point (SPEC §1.6; model/selected.rs, # The
    /// photons): the model makes the rate W = Σ_a‖u_a‖² of every class the state emits there, in the
    /// classes' order; `draw` picks one of them by its place among those rates, or none; and the model makes the photon
    /// of the class drawn (`Emitted`) -- the one class whose amplitudes and kick a jump reads. None where `draw` picks
    /// none. Inside a step the point is on the step's path: `on` is the cluster's middle there with the path and the
    /// point's time s from the middle (`Places`), what the step made of the model for the jump to read (DESIGN §13.2);
    /// at a trajectory's start there is none.
    fn emit<'m>(&'m self, cluster: usize, at: &Point, on: Option<(&Self::Middle<'m>, &<Self::Middle<'m> as Middle>::Path, f64)>, psi: &[C64], draw: &mut dyn FnMut(&[f64]) -> Option<usize>) -> Option<Emitted>;
    /// Where a jump's bare state `f` lands in a cluster at a point, against the trajectory's threshold θ for its next jump
    /// (`Landing`, `Landed`): the state the trajectory goes on in and its threshold -- the state itself and θ for a model
    /// whose jumps land on its own states (the plain landing L_qψ/‖·‖, RUSTPLAN §2.2 D5) --; or nowhere where the landing
    /// does not exist in that cluster there. Inside a step `on` is the landing cluster's middle with the path (`emit`).
    fn landed<'m>(&'m self, _cluster: usize, _at: &Point, _on: Option<(&Self::Middle<'m>, &<Self::Middle<'m> as Middle>::Path)>, f: &[C64], threshold: f64) -> Landing {
        Landing::On(Landed::plain(f.iter().copied().collect(), threshold))
    }

    /// The loss energy E_s = (ħ/m)κ²/2 + V_run(r; p(ψ)) of the cluster's state at a point (rad/µs).
    fn energy(&self, cluster: usize, at: &Point, psi: &[C64]) -> f64;

    /// The loss test at a point: E_s above its escape value, the far value (SPEC §1.7).
    fn lost(&self, cluster: usize, at: &Point, psi: &[C64]) -> bool;

    /// The displayed rows of the cluster's state at a point.
    fn readout(&self, cluster: usize, at: &Point, psi: &[C64]) -> [f64; STATES];

    /// A cluster's state's held forms at a point (module documentation, # The path's momentum: `Forms`, at the point's
    /// time): the model at the point (`middle`, deciding nothing), the state on its path at the point's momentum, at the
    /// middle (`forms_at`), less the reference's gradient there. Where a jump at a step's end lands in another cluster,
    /// and after a trajectory's first jump.
    fn forms(&self, cluster: usize, at: &Point, psi: &[C64]) -> Forms
    where
        Self: Sized,
    {
        let middle = self.middle(cluster, at.r, at.t);
        let path = middle.path(at.kappa, [0.0; 3]);
        let mut rho = vec![C64::ZERO; psi.len() * psi.len()];
        density(psi, &mut rho);
        Forms { time: at.t, ..forms_at(&middle, &path, &rho) }.less(self.reference(at.r).gradient)
    }

    /// Where a trajectory starts (`Trajectory::new`): the bare state f on `cluster` at the point -- the cluster it lands
    /// in there, its landing against the threshold θ (`land`) and its held forms there (`forms`); no number where it
    /// lands nowhere.
    fn started(&self, cluster: usize, at: &Point, f: &[C64], threshold: f64) -> (usize, Landed, Forms)
    where
        Self: Sized,
    {
        let Some((cluster, _, landed)) = land(self, cluster, at, None, f, threshold) else {
            return (cluster, Landed::plain(f.iter().map(|_| C64::new(f64::NAN, f64::NAN)).collect(), threshold), Forms { time: at.t, held: [f64::NAN; 3], tones: vec![] });
        };
        let forms = self.forms(cluster, at, &landed.psi);
        (cluster, landed, forms)
    }

    /// The counted real multiply-adds of one jump's photons (`Dynamics::emit`), made once per run: `jump` adds it to the
    /// trajectory's work. The one count of a step's run that is not counted where it is made (`Price`): the model makes
    /// the photon (`Dynamics::emit`) and does not hand its count on; so it is priced at the dearest class's of every
    /// cluster at the focus, once a run.
    fn photon_work(&self) -> f64 {
        0.0
    }
}

/// A cluster's generator at a point (r_h, t_m) (module documentation, # The kick): what of it depends on neither the
/// momentum nor the time is made with it; a path through the point adds the momentum's part; and the generator and its
/// gradient's forms are asked at times along that path. In the cluster's Doppler frames throughout. The scattered
/// light of a state at the point is asked of it too (# The absorbed light's momentum): a step asks the model for its
/// middle point once.
pub trait Middle {
    /// A path through the point: the Doppler weights at the momentum the generator is taken at, and the rates at which
    /// the light's plane waves and the Doppler frames turn at the path's velocity. Plain numbers.
    type Path;

    /// The cluster's states.
    fn states(&self) -> usize;

    /// Whether Ĥ is the same at every time along every path: no term beats in the cluster's frames, in time or along a
    /// path. Exact (the model's integer frequencies and wave vectors), not a comparison of numbers.
    fn still(&self) -> bool;

    /// The path at the velocity v (µm/µs), the generator taken at the momentum κ (1/µm).
    fn path(&self, kappa: [f64; 3], velocity: [f64; 3]) -> Self::Path;

    /// The Doppler frames' rates K_s·v on a path, per state (rad/µs).
    fn gauge(&self, path: &Self::Path, rates: &mut [f64]);

    /// Ĥ(s) at the time s from the middle along the path, into `h` (n × n; rad/µs), the frames' diag(K_s·v) with it.
    fn generator(&self, path: &Self::Path, s: f64, h: &mut Mat);

    /// The candidate tones of Ĥ along the path (spectral.rs, # Which beats are tones): Ĥ(s) holds
    /// Σ_m (e^{iν_m s} G_m⁺ + e^{−iν_m s} G_m⁻) of terms whose beats ν_m are exact and whose envelopes G_m^± are constant
    /// along the path (`envelopes`); into `beats` each ν_m (rad/µs), and into `keys` each one's exact label: integers,
    /// one length for all, whose sums are the beats' sums exactly -- equal sums the same beat, the propagation's
    /// products of members of equal key sums one class (spectral.rs, # The second order by tones). The propagation
    /// takes as tones those it resolves better so and the rest with the generator's static group. None here; none on a
    /// path Ĥ is still along.
    fn beats(&self, _path: &Self::Path, _beats: &mut Vec<f64>, _keys: &mut Vec<Vec<i64>>) {}

    /// The model's exact blocks of the cluster's states (model/selected.rs, # Exact blocks): a partition on which every
    /// matrix the middle gives -- the generator at every time, its gradient, every tone's envelopes -- is block-diagonal
    /// (the jumps need not be). None: the states are one block.
    fn blocks(&self) -> &[Vec<usize>] {
        &[]
    }

    /// The m-th candidate's members' envelopes (`beats`) at the time s from the middle along the path (the step's nodes:
    /// spectral.rs `Picture::separate`) into `plus` (G_m⁺, at e^{iν_m s}) and `minus` (G_m⁻, at e^{−iν_m s}; n × n each,
    /// rad/µs): its part of `generator`'s at the time s is e^{iν_m s} G_m⁺(s) + e^{−iν_m s} G_m⁻(s).
    fn envelopes(&self, _path: &Self::Path, _m: usize, _s: f64, _plus: &mut Mat, _minus: &mut Mat) {}

    /// The m-th candidate's part of `force`'s forms at its beat, per lab axis: Σ_k w_k Re tr(Ĝ_m(s_k) ρ_k) of the
    /// tone's Hermitian part's gradient Ĝ_m(s) = e^{iν_m s} Ĝ_m⁺(s) + e^{−iν_m s} Ĝ_m⁻(s), its envelopes through the step's
    /// nodes as the propagation takes them (`envelopes`): Σ_j Re tr(Ĝ_m⁺(s_j) σ_j + Ĝ_m⁻(s_j) σ_j†), given per node s_j
    /// its sums σ_j = Σ_k w_k ℓ_j(s_k) e^{iν_m s_k} ρ_k = `re` + i `im` (n × n row by row each, Hermitian) as
    /// (s_j, re, im) -- the step's three nodes, or the middle alone (a fixed point's: `forms_at`).
    fn tone_force(&self, _path: &Self::Path, _m: usize, _sums: &[(f64, &[C64], &[C64])]) -> [f64; 3] {
        [0.0; 3]
    }

    /// The counted complex multiply-adds of the m-th candidate's `envelopes` and of its `tone_force`.
    fn tone_work(&self, _m: usize) -> [f64; 2] {
        [0.0; 2]
    }

    /// Re tr(Ĝ_a(s) ρ) per lab axis a (rad/µs/µm): the gradient of Ĥ's Hermitian part at the time s from the middle along
    /// the path, contracted with a Hermitian ρ on the states (n × n, row by row).
    fn force(&self, path: &Self::Path, s: f64, rho: &[C64]) -> [f64; 3];

    /// The gradient at the fixed point by its tones, G(r_h, t) = Σ_m G_m e^{iΩ_m(t − t_m)}: to `each` every Ω_m (rad/µs)
    /// with Re tr(G_m ρ) per lab axis, for a Hermitian ρ.
    fn tones(&self, path: &Self::Path, rho: &[C64], each: &mut dyn FnMut(f64, [f64; 3]));

    /// The middle's two forms at once (`Midway::stretch`): Σ_m w(Ω_m) Re tr(G_m σ) -- the fixed point's tones on σ, each
    /// weighted by w of its beat (`tones`) -- plus Re tr(Ĝ(0) ρ), the form at the middle along the path (`force` at s = 0),
    /// per lab axis.
    fn centre(&self, path: &Self::Path, held: &[C64], weight: &dyn Fn(f64) -> f64, rho: &[C64]) -> [f64; 3] {
        let mut out = self.force(path, 0.0, rho);
        self.tones(path, held, &mut |omega, form| {
            for a in 0..3 {
                out[a] += weight(omega) * form[a];
            }
        });
        out
    }

    /// What the light a state scatters at the point gives the momentum per unit time beside the Ehrenfest force and the
    /// jumps' kicks (model.rs `Scattering::of_moments`): the force and the second moment's rate the jumps do not carry, of the
    /// state ψ at the middle's time and the path's momentum; and the counted real multiply-adds that made it. Asked of
    /// the middle, which holds the point: the model makes nothing there a second time.
    fn scattering(&self, path: &Self::Path, psi: &[C64]) -> (Scattering, f64);

    /// The counted complex multiply-adds of one `generator`, one `force` and one `tones`.
    fn work(&self) -> [f64; 3];

    /// The counted real multiply-adds that made the middle (`Dynamics::middle`): what of the generator the point holds,
    /// whatever a step asks of it.
    fn made(&self) -> f64;

    /// Ĝ_a(s) as matrices at the time s from the middle along the path (n × n each, row by row, Hermitian): what
    /// `force` contracts; and where `slope` is asked -- with the half-width h of the step it is the slope of (a model
    /// may take it as its chord through the step's ends, ±h) --, their derivative along the path at s, dĜ_a/ds. With
    /// the counted complex multiply-adds that made them (`optical_work`: priced where they are made).
    fn gradient(&self, path: &Self::Path, s: f64, g: &mut [Vec<C64>; 3], slope: Option<(f64, &mut [Vec<C64>; 3])>) -> f64;

    /// The rows as shown (the model's readout, `Dynamics::readout`) of the cluster's state ψ at the time s from the middle
    /// along the path, on the step's own model there -- what the step propagates between its nodes --, read where the
    /// step stands (adapt.rs, # The reads); none where the model reads it at the point instead (`Dynamics::readout`).
    fn readout(&self, _path: &Self::Path, _s: f64, _psi: &[C64]) -> Option<[f64; STATES]> {
        None
    }

    /// The cluster's optical sector on the path (# The optical sector's noise; model/optical.rs): None where the
    /// cluster has no explicit excited state.
    fn optical(&self, _path: &Self::Path) -> Option<Optical> {
        None
    }

    /// The step's three nodes' forms at once (`Midway::stretch`): Re tr(Ĝ(s_j) ρ_j) per lab axis at s = −h, 0, h (`force` at
    /// each), the middle's with the fixed point's tones on a held state at their weights where one is held (`centre`); what
    /// the step takes is their sum, node by node. And where `end` is given -- the end state's density, and the path at
    /// rest through the middle (its fixed point's tones) -- its forms at the middle (`Forms`, its time the middle's, 0;
    /// the model's, the integrator takes the reference's gradient from them; module documentation, # The path's
    /// momentum). One call, so that a model whose forms are one derivative of its step contracts them at once (and gives
    /// their sum as the first node's, the end's in the same contraction); here the end's by the middle's forms
    /// (`forms_at`).
    fn nodes(&self, path: &Self::Path, h: f64, rho: [&[C64]; 3], held: Option<(&[C64], &dyn Fn(f64) -> f64)>, end: Option<(&[C64], &Self::Path)>) -> ([[f64; 3]; 3], Option<Forms>)
    where
        Self: Sized,
    {
        let forms = std::array::from_fn(|j| match held {
            Some((held, weight)) if j == 1 => self.centre(path, held, weight, rho[j]),
            _ => self.force(path, [-h, 0.0, h][j], rho[j]),
        });
        (forms, end.map(|(rho, fixed)| forms_at(self, fixed, rho)))
    }

}

/// The optical sector of a cluster on a path (# The optical sector's noise; model/optical.rs): its ground states (the
/// cluster's first), the slowest optical rate γ_Q -- the least of its excited states' decay over two (rad/µs) --, and
/// the light pattern's fastest beat on the path where the cluster beats, ν = max |(k_l − k_m)·v| over its beams (none
/// where it is still).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Optical {
    pub ground: usize,
    pub decay: f64,
    pub beat: f64,
}

/// The trajectory's workspace of the optical sector (`optical`): the sector, and the gradient's matrices with their
/// slope along the path.
#[derive(Default)]
struct Noise {
    sector: Sector,
    gradient: [Vec<C64>; 3],
    slope: [Vec<C64>; 3],
}

/// The optical sector's operators at a step's middle along its path (# The optical sector's noise), made in the
/// workspace `noise`: the gradient there and, where the cluster beats, its slope along the path over the step of the
/// half-width `h` (`Middle::gradient`);
/// K of the picture's generator at the middle, to the first order along the path with the picture's slope through the
/// step's ends (`Picture::slope`). Its locality ε_Q = (ν/γ_Q)², its ground states and its make's counted work with the
/// gradient's as the model counted it (real multiply-adds; the stretch's `Price` with the forms: `optical_work`); None
/// where the cluster has no optical sector; the step refused where its sector made no K (`Refused::Optical`).
fn optical<M: Middle>(along: &Along<M>, picture: &Picture, still: bool, h: f64, noise: &mut Noise) -> Result<Option<(f64, usize, f64)>, Refused> {
    let _p = crate::prof::scope(crate::prof::OPTICAL);
    let Some(shape) = along.middle.optical(&along.path) else { return Ok(None) };
    let Noise { sector, gradient, slope } = noise;
    let matrices = along.middle.gradient(&along.path, 0.0, gradient, (!still).then_some((h, &mut *slope)));
    sector.make(&picture.base, gradient, shape.ground, (!still).then_some((&picture.slope, &*slope))).map_err(Refused::Optical)?;
    Ok(Some(((shape.beat / shape.decay).powi(2), shape.ground, sector.work + COMPLEX * matrices)))
}

/// The counted work of the optical sector at a middle (`optical`): its make with the gradient's matrices as they were
/// counted (`work`: `Sector::cost`, `Middle::gradient`), and one `within`'s forms.
fn optical_work(n: usize, work: f64) -> f64 {
    COMPLEX * Sector::forms(n) + work
}

/// The optical sector's rate of a state x (unnormalized; model/optical.rs, # K_Q) from K_a and Ĝ_b on it (F = −Ĝ):
/// within_Q,ab = −Re⟨K_a x|Ĝ_b x⟩/‖x‖² + ⟨K_a⟩⟨Ĝ_b⟩, ½ d Var(κ_a, κ_b)/dt of the white noise it stands for in its
/// symmetric part (1/µm²/µs).
pub fn within(k: &[Vec<C64>; 3], g: &[Vec<C64>; 3], x: &[C64]) -> [[f64; 3]; 3] {
    let n = x.len();
    let apply = |m: &[C64]| -> [C64; STATES] { std::array::from_fn(|i| if i < n { m[i * n..(i + 1) * n].iter().zip(x).map(|(m, x)| *m * *x).sum() } else { C64::ZERO }) };
    let norm = norm_sqr(x);
    let (kx, gx): ([[C64; STATES]; 3], [[C64; STATES]; 3]) = (std::array::from_fn(|a| apply(&k[a])), std::array::from_fn(|b| apply(&g[b])));
    let (km, gm): ([f64; 3], [f64; 3]) = (std::array::from_fn(|a| inner(x, &kx[a][..n]).re / norm), std::array::from_fn(|b| inner(x, &gx[b][..n]).re / norm));
    std::array::from_fn(|a| std::array::from_fn(|b| -inner(&kx[a][..n], &gx[b][..n]).re / norm + km[a] * gm[b]))
}

/// A cluster's internal state: its amplitudes on the cluster's states -- at most the model's `STATES` --, held in place,
/// so that whatever holds one (a trajectory, a step's end, a jump's photon) copies without the heap. A slice of its amplitudes wherever one is asked; those beyond its length are zero.
#[derive(Clone, Copy, PartialEq)]
pub struct Internal {
    len: usize,
    amplitudes: [C64; STATES],
}

impl Internal {
    /// The zero state on `len` states.
    pub fn zeros(len: usize) -> Internal {
        assert!(len <= STATES, "an internal state on {len} states of the model's {STATES}");
        Internal { len, amplitudes: [C64::ZERO; STATES] }
    }
}

impl std::ops::Deref for Internal {
    type Target = [C64];
    fn deref(&self) -> &[C64] {
        &self.amplitudes[..self.len]
    }
}

impl std::ops::DerefMut for Internal {
    fn deref_mut(&mut self) -> &mut [C64] {
        &mut self.amplitudes[..self.len]
    }
}

impl std::fmt::Debug for Internal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.iter()).finish()
    }
}

impl FromIterator<C64> for Internal {
    fn from_iter<I: IntoIterator<Item = C64>>(amplitudes: I) -> Internal {
        let mut out = Internal::zeros(0);
        for z in amplitudes {
            assert!(out.len < STATES, "an internal state beyond the model's {STATES} states");
            out.amplitudes[out.len] = z;
            out.len += 1;
        }
        out
    }
}

/// The photon a jump emits (`Dynamics::emit`), of the class drawn among its state's classes: the cluster the jump lands
/// the atom in; |k_φ| (1/µm), 2π/c times the class's photon frequency -- the emission's recoil is −k_φ n --; the
/// amplitudes u_a = J_{a,φ}ψ (a = x, y, z) on the landing cluster's states -- the direction's density ∝ tr G − n·Re(G)·n
/// (G_ab = ⟨u_a|u_b⟩), the polarization ε across n by ‖Σ_a ε_a u_a‖², the state after f = Σ_a ε_a u_a --; and the absorbed
/// light's part of the jump's kick (1/µm): the class's conditional mean as far as one absorbed photon's momentum goes
/// (model.rs `Moments::kick`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Emitted {
    pub landing: usize,
    pub wave_number: f64,
    pub amplitudes: [Internal; 3],
    pub kick: [f64; 3],
    /// The counted real multiply-adds that made it where the model counts them there (its place and drawn photons; a
    /// model that prices a jump once a run says none: `Dynamics::photon_work`).
    pub work: f64,
}

/// Where a jump's bare state lands in a cluster (`Dynamics::landed`): the state the trajectory goes on in and its
/// threshold (`On`); or none, where the landing does not exist in that cluster at the point -- a number that is not
/// finite none (`Nowhere`).
#[derive(Clone, Debug, PartialEq)]
pub enum Landing {
    On(Landed),
    Nowhere,
}

/// A landing (`Dynamics::landed`): the state the trajectory goes on in, the threshold its next jump is at, the kick the
/// landing books (1/µm) and its counted work (real multiply-adds), the jump's own as its photon's is (`jump`): no part of a
/// step's. Where its norm is at or below its threshold already, the jump is at once.
#[derive(Clone, Debug, PartialEq)]
pub struct Landed {
    pub psi: Internal,
    pub threshold: f64,
    pub kick: [f64; 3],
    pub work: f64,
}

impl Landed {
    /// The plain landing: the state, the trajectory's own threshold, no kick, no count.
    pub fn plain(psi: Internal, threshold: f64) -> Landed {
        Landed { psi, threshold, kick: [0.0; 3], work: 0.0 }
    }
}

/// Where a jump's bare state f of the cluster `cluster` lands at a point against the threshold θ (`Dynamics::landed`): on
/// the middles `places` holds there (`Places::of`), or at the point. The cluster, f on its states and its landing; none
/// where it lands nowhere.
fn land<'m, D: Dynamics>(model: &'m D, cluster: usize, at: &Point, places: Option<&mut Places<'_, 'm, D>>, f: &[C64], threshold: f64) -> Option<(usize, Internal, Landed)> {
    let on = places.map(|p| p.of(cluster));
    match model.landed(cluster, at, on, f, threshold) {
        Landing::On(landed) => Some((cluster, f.iter().copied().collect(), landed)),
        Landing::Nowhere => None,
    }
}

/// A trajectory's state (SPEC §1.6, §1.7, §2.2).
#[derive(Clone, Debug, PartialEq)]
pub struct Trajectory {
    /// Position (µm).
    pub r: [f64; 3],
    /// Momentum over ħ (1/µm).
    pub kappa: [f64; 3],
    /// The internal state on the cluster's states, unnormalized between jumps: ‖ψ‖² is the probability of no jump since
    /// the last one.
    pub psi: Internal,
    /// The cluster ψ lives on.
    pub cluster: usize,
    /// Time (µs).
    pub time: f64,
    /// The waiting time's threshold: the next jump is where ‖ψ‖² reaches it -- θ ∈ (0, 1] drawn at the last jump, or its
    /// landing's (`Landed`).
    pub threshold: f64,
    /// The jumps made: the next one draws from `Stream::jump(seed, index, jumps + 1)`.
    pub jumps: u32,
    /// The photons its jumps emitted (a jump at its threshold on landing emits another) and their Σ k_φ² (1/µm²): the
    /// recoil of the light it scatters, which sets T's semiclassical error scale (solve.rs, # T's allowance).
    pub scattered: [f64; 2],
    /// The time the atom was lost (SPEC §1.7: checked at every step's end and after every jump); None while held.
    pub lost: Option<f64>,
    /// Whether a step of it did not resolve what it samples along its path at its speed (`Domain::proven`; module
    /// documentation, # The domain): from that step on the trajectory is not the model's to the step's accuracy, and
    /// what it shows is booked whole (solve.rs, # The reach's atoms).
    pub unresolved: bool,
    /// The run's seed and the trajectory's number: its jumps' streams.
    pub seed: u64,
    pub index: u64,
    /// The work counted (`decide::Work`, real multiply-adds) of the integrator's own algorithms over the trajectory --
    /// the generator's decomposition, its following and hops, the states at the kick's nodes and their sums, with the
    /// model's generators and forces at the step's nodes (`adapt` adds each step's) --, and its jumps' (`jumped`).
    pub work: f64,
    /// Of `work`, its jumps' own: their photons (`Emitted::work`, `Dynamics::photon_work`) and their landings
    /// (`Landed::work`) as they were made -- a jump's work is the jump's at whatever step it falls in, and the run prices
    /// it so, at the trajectory's own rate (solve.rs `work_at`).
    pub jumped: f64,
    /// The optical sector's pending covariance (1/µm², symmetric, negative semidefinite after every step: # The optical
    /// sector's noise): what of the noise's covariance its draws have not taken -- its negative excursions --, carried
    /// across steps and jumps.
    pub pending: [[f64; 3]; 3],
    /// The optical sector's booking over the trajectory (# The optical sector's noise; the ledger's): Σ ε_Q tr C of its
    /// steps' covariances C, Σ tr C, and the scattered light's Σ tr(2D)|τ| (1/µm²).
    pub booked: [f64; 3],
    /// Its state's held forms where the model last evaluated it (module documentation, # The path's momentum): its next
    /// step's path momentum.
    pub forms: Forms,
}

impl Trajectory {
    /// A trajectory at a basis state (a ground state of the model basis) at (r, κ) and time 0, the light switched on there
    /// (SPEC §2.4; ELIMINATION v3 4): the bare state -- the unit vector of that state on its cluster -- is a jump's
    /// landing, the same map (`Landed`), against the first threshold from jump 0's stream, with its held forms there
    /// (`Dynamics::started`); at its threshold already, the jump at once, as after a jump (`jump`), and the held forms of
    /// the state it lands in.
    pub fn new<D: Dynamics>(model: &D, state: usize, r: [f64; 3], kappa: [f64; 3], seed: u64, index: u64) -> Trajectory {
        let bare = |cluster: usize| -> Internal {
            let states = &model.cluster(cluster).states;
            let at = states.iter().position(|&s| s == state).expect("a ground state lies on its cluster");
            let mut psi = Internal::zeros(states.len());
            psi[at] = C64::ONE;
            psi
        };
        let threshold = Stream::jump(seed, index, 0).open_uniform();
        let (cluster, landed, forms) = model.started(model.cluster_of(state), &Point { r, kappa, t: 0.0 }, &bare(model.cluster_of(state)), threshold);
        let on = norm_sqr(&landed.psi) > landed.threshold;
        let mut trajectory = Trajectory {
            r,
            kappa: std::array::from_fn(|a| kappa[a] + landed.kick[a]),
            psi: landed.psi,
            cluster,
            time: 0.0,
            threshold: landed.threshold,
            jumps: 0,
            scattered: [0.0; 2],
            lost: None,
            unresolved: false,
            seed,
            index,
            work: 0.0,
            jumped: 0.0,
            pending: [[0.0; 3]; 3],
            booked: [0.0; 3],
            forms,
        };
        if !on {
            // Its jump at once; where the model has no photon to give, or its landing does not exist, the start is nowhere
            // as a start that lands nowhere is: no held forms (solve.rs `run` says it, # Exits).
            trajectory.forms = match jump(model, &mut trajectory, None) {
                Ok(()) => {
                    let t = &trajectory;
                    model.forms(t.cluster, &t.point(), &t.psi)
                }
                Err(_) => Forms { time: 0.0, held: [f64::NAN; 3], tones: vec![] },
            };
        }
        trajectory
    }
    /// The trajectory's point.
    pub fn point(&self) -> Point {
        Point { r: self.r, kappa: self.kappa, t: self.time }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The reference flow (A).

/// sin(s)/s, or sinh(s)/s when `hyperbolic`, of s ≥ 0: the series 1 ∓ s²/6 where its next term s⁴/120 is within the unit
/// round-off (derived: there the series is exact to round-off), the quotient beyond (accurate wherever s ≠ 0).
fn ratio(s: f64, hyperbolic: bool) -> f64 {
    if s * s * s * s <= 120.0 * UNIT_ROUNDOFF {
        if hyperbolic { 1.0 + s * s / 6.0 } else { 1.0 - s * s / 6.0 }
    } else if hyperbolic {
        s.sinh() / s
    } else {
        s.sin() / s
    }
}

/// The Stumpff functions (c₀, c₁, c₂) at x (Stumpff 1959): c_k(x) = Σ_j (−x)^j/(k + 2j)!, entire in x -- cos √x,
/// sin √x/√x and (1 − cos √x)/x for x > 0, their hyperbolic forms for x < 0, (1, 1, ½) at 0 --, without cancellation:
/// c₂(x) = ½ c₁(x/4)².
pub fn stumpff(x: f64) -> (f64, f64, f64) {
    let hyperbolic = x < 0.0;
    let s = x.abs().sqrt();
    let c0 = if hyperbolic { s.cosh() } else { s.cos() };
    let half = ratio(0.5 * s, hyperbolic);
    (c0, ratio(s, hyperbolic), 0.5 * half * half)
}

/// sin(x)/x of a real x: c₁(x²).
fn sinc(x: f64) -> f64 {
    stumpff(x * x).1
}

/// The reference's exact flow (A) of a step: the expansion point p (µm), the reference potential there (rad/µs), the
/// principal axes (unit rows), per axis the curvature λ (rad/µs/µm²) and the gradient g at p along it (rad/µs/µm), and ħ/m.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Flow {
    pub centre: [f64; 3],
    pub value: f64,
    pub axes: [[f64; 3]; 3],
    pub curvatures: [f64; 3],
    pub gradient: [f64; 3],
    pub hbar_over_mass: f64,
}

impl Flow {
    /// The flow of the expansion `q` at p.
    pub(crate) fn new(q: &Quadratic, centre: [f64; 3], hbar_over_mass: f64) -> Flow {
        let (curvatures, axes) = linalg::symmetric3_eigen(&q.hessian);
        let gradient = std::array::from_fn(|i| linalg::dot3(axes[i], q.gradient));
        Flow { centre, value: q.value, axes, curvatures, gradient, hbar_over_mass }
    }
    /// The reference potential's largest value on the straight path r + v u, |u| ≤ |half| (rad/µs), by its expansion
    /// about p -- the potential the step's flow moves in: V(p) + g·y + ½ yᵀΛy along the axes, y = (r − p) + v u, a parabola
    /// c + b u + ½ a u² in u, its largest at an end, or at its vertex where it bends down within the path.
    pub fn highest(&self, r: [f64; 3], v: [f64; 3], half: f64) -> f64 {
        let (mut a, mut b, mut c) = (0.0, 0.0, self.value);
        for i in 0..3 {
            let e = self.axes[i];
            let (y, w) = (linalg::dot3(e, [r[0] - self.centre[0], r[1] - self.centre[1], r[2] - self.centre[2]]), linalg::dot3(e, v));
            a += self.curvatures[i] * w * w;
            b += (self.gradient[i] + self.curvatures[i] * y) * w;
            c += self.gradient[i] * y + 0.5 * self.curvatures[i] * y * y;
        }
        let h = half.abs();
        let ends = linalg::larger(c + b * h + 0.5 * a * h * h, c - b * h + 0.5 * a * h * h);
        if a < 0.0 && b.abs() <= -a * h { linalg::larger(ends, c - 0.5 * b * b / a) } else { ends }
    }
    /// (r, κ) after the time t (either sign) from (r, κ) (module documentation, # The reference).
    pub fn advance(&self, from: ([f64; 3], [f64; 3]), t: f64) -> ([f64; 3], [f64; 3]) {
        self.reach(from, t).0
    }
    /// `advance`, and the sizes its position's and its momentum's coordinates are summed from (the centre's and every
    /// axis's terms' moduli): `units::rounding` of a coordinate's operations (`REACH`) times them bounds its rounding.
    fn reach(&self, from: ([f64; 3], [f64; 3]), t: f64) -> (([f64; 3], [f64; 3]), [f64; 2]) {
        self.driven(from, t, None)
    }
    /// `reach` under the reference and, where `drive` gives one, a force a + b s along each principal axis (s the
    /// time from the start; a in rad/µs/µm, b per µs more): the affine flow's variation of constants, k += a t c₁ +
    /// b t² c₂ and y += (ħ/m)(a t² c₂ + b t³ c₃) at x = (ħ/m)λt² -- the gradient lowered by a, and the ramp's own terms
    /// (module documentation, # The path between a step's ends).
    fn driven(&self, (r, kappa): ([f64; 3], [f64; 3]), t: f64, drive: Option<&[[f64; 2]; 3]>) -> (([f64; 3], [f64; 3]), [f64; 2]) {
        let hm = self.hbar_over_mass;
        let d = [r[0] - self.centre[0], r[1] - self.centre[1], r[2] - self.centre[2]];
        let (mut r1, mut k1) = (self.centre, [0.0; 3]);
        let mut sizes = [self.centre.iter().fold(0.0, |m: f64, x| linalg::larger(m, x.abs())), 0.0];
        for i in 0..3 {
            let e = self.axes[i];
            let (y, k, lambda) = (linalg::dot3(e, d), linalg::dot3(e, kappa), self.curvatures[i]);
            let x = hm * lambda * t * t;
            let (c0, c1, c2) = stumpff(x);
            let g = drive.map_or(self.gradient[i], |drive| self.gradient[i] - drive[i][0]);
            let position = [y * c0, hm * k * t * c1, -hm * g * t * t * c2];
            let momentum = [k * c0, -(lambda * y + g) * t * c1];
            let (mut y1, mut k_1) = (position[0] + position[1] + position[2], momentum[0] + momentum[1]);
            sizes[0] += position.iter().map(|x| x.abs()).sum::<f64>();
            sizes[1] += momentum.iter().map(|x| x.abs()).sum::<f64>();
            if let Some(drive) = drive {
                let ramp = [hm * drive[i][1] * t * t * t * stumpff3(x), drive[i][1] * t * t * c2];
                (y1, k_1) = (y1 + ramp[0], k_1 + ramp[1]);
                sizes = [sizes[0] + ramp[0].abs(), sizes[1] + ramp[1].abs()];
            }
            for a in 0..3 {
                r1[a] += y1 * e[a];
                k1[a] += k_1 * e[a];
            }
        }
        ((r1, k1), sizes)
    }
    /// ∇V_ref at r (rad/µs/µm): Σ_i e_i (g_i + λ_i e_i·(r − p)).
    fn gradient_at(&self, r: [f64; 3]) -> [f64; 3] {
        let d = [r[0] - self.centre[0], r[1] - self.centre[1], r[2] - self.centre[2]];
        let mut out = [0.0; 3];
        for i in 0..3 {
            let e = self.axes[i];
            let along = self.gradient[i] + self.curvatures[i] * linalg::dot3(e, d);
            for a in 0..3 {
                out[a] += along * e[a];
            }
        }
        out
    }
    /// The largest rate of the flow, max_i √((ħ/m)|λ_i|) (rad/µs): κ's frequency along its axes (a hyperbolic axis' rate).
    fn rate(&self) -> f64 {
        self.curvatures.iter().map(|l| (self.hbar_over_mass * l.abs()).sqrt()).fold(0.0, linalg::larger)
    }
}

/// Rounds of an iteration that halves its distance each round at least -- a bracket's bisection (`root`), and the
/// middle's rounds in a stable step, whose contraction is no less (propagation.md §2.2) --: the unit round-off is
/// 2⁻⁵³, the mantissa's bits, so after as many rounds the distance is the round-off's.
const ROUNDS: usize = f64::MANTISSA_DIGITS as usize;

/// The operations between a flow's inputs and a coordinate of its reach (`Flow::reach`): the coordinate adds three
/// axes' displacements to the centre (6), each the Stumpff combination of three terms (10) of the axis's coordinate (a
/// difference and a product of three: 6) and of its x (3) through its functions (3): 28. `units::rounding` of them
/// times the sizes the coordinate is summed from is the rounding of a reach.
const REACH: usize = 28;

/// The step's middle (module documentation, # The reference): the fixed point p of "the first half flow on V_ref(p)
/// reaches p" from (r, κ) over `half`, from the free flight's middle; the flow and (r_h, κ_h). A round moves p by the
/// reference's change over the move times the flow's response, (ħ/m)|∇K|·|r − p|·half²/3 of the last move: a contraction
/// while the step resolves the potential's third derivative, by a factor of 10⁻² to 10⁻⁴ a round at the steps a trap
/// takes (so that an accelerated iteration -- Broyden's, Anderson's -- gains less than one round of some six).
///
/// Converged is: the reach's move within its rounding (`REACH`) -- p is its own reach as far as the flow tells --, or
/// the next round's moves within it. The step's results depend on the centre p only through that third derivative:
/// taken at p with the reach (r_h, κ_h) they differ from the fixed point's by the next round's moves of the reach's
/// position and momentum, the last ones times the contraction, for which the largest measured stands (the centre itself
/// then lies a move, not a rounding, from its reach, which no result sees). Everything else is refused: a move that
/// does not shrink (the map is no contraction at this step: no fixed point is being reached), or `ROUNDS` rounds of a
/// contraction too slow for a stable step.
fn middle<D: Dynamics>(model: &D, r: [f64; 3], kappa: [f64; 3], half: f64) -> Result<(Flow, ([f64; 3], [f64; 3])), Refused> {
    let hm = model.hbar_over_mass();
    let apart = |a: [f64; 3], b: [f64; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).fold(0.0, linalg::larger);
    let mut p: [f64; 3] = std::array::from_fn(|a| r[a] + hm * kappa[a] * half);
    // The last move of the reach's position, the momentum it reached then, and the largest contraction measured (the
    // first round measures none: its move over no move before it is zero).
    let (mut last, mut before, mut contraction) = (f64::INFINITY, kappa, 0.0_f64);
    for _ in 0..ROUNDS {
        let flow = Flow::new(&model.reference(p), p, hm);
        let (reached, sizes) = flow.reach((r, kappa), half);
        let moved = apart(reached.0, p);
        if !moved.is_finite() || !reached.1.iter().all(|k| k.is_finite()) {
            return Err(Refused::Overflow);
        }
        contraction = linalg::larger(contraction, moved / last);
        let within = sizes.map(|size| rounding(REACH) * size);
        let next = [contraction * moved, contraction * apart(reached.1, before)];
        if moved <= within[0] || (last.is_finite() && next[0] <= within[0] && next[1] <= within[1]) {
            return Ok((flow, reached));
        }
        if moved >= last {
            return Err(Refused::Reference);
        }
        (last, before, p) = (moved, reached.1, reached.0);
    }
    Err(Refused::Reference)
}

// ---------------------------------------------------------------------------------------------------------------------
// Quadrature within a tolerance.

/// A Gauss–Legendre rule of m nodes on [−1, 1] (nodes from the largest down, weights).
struct Rule {
    nodes: Vec<f64>,
    weights: Vec<f64>,
}

/// The rule of m nodes (`special::gauss_legendre`: O(m) operations, to the round-off), made where it is asked and held by
/// the panels that take it (`panels`) while they run: nothing of it outlives its quadrature.
fn rule(m: usize) -> std::rc::Rc<Rule> {
    let (nodes, weights) = crate::special::gauss_legendre(m).into_iter().unzip();
    std::rc::Rc::new(Rule { nodes, weights })
}

thread_local! {
    /// The spherical Bessel functions' scratch (`fewest`).
    static BESSELS: std::cell::RefCell<Vec<f64>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// The index past `from` where the three-term recurrence's growing solution (p_{from−1} = 0, p_from = 1, upward) has
/// grown by the round-off's inverse: there j_l has fallen below the round-off of j_from, and Miller's recurrence started
/// there has j_from to the round-off (Gautschi 1967).
fn grown(x: f64, from: usize) -> usize {
    let (mut p0, mut p1, mut l) = (0.0f64, 1.0f64, from);
    while p1.abs() < 1.0 / UNIT_ROUNDOFF {
        let p2 = (2 * l + 1) as f64 / x * p1 - p0;
        (p0, p1, l) = (p1, p2, l + 1);
    }
    l
}

/// The fewest nodes of one Gauss–Legendre panel over the width w on an integrand that is entire at the bandwidth ν,
/// times a polynomial of `degree`, to the round-off of its size times the width -- every component of it, as the
/// propagation takes Ĥ₀ (# The quadrature's tolerance): on [−1, 1] a component is e^{iκt} p(t), κ = ν w/2, whose
/// Legendre series has the coefficients (2l + 1) i^l j_l(κ) (j_l the spherical Bessel functions), moved by p at most
/// `degree` places; the rule of m nodes integrates P_l exactly up to l = 2m − 1 and errs on a P_l beyond by no more
/// than 2 (|P_l| ≤ 1, its weights' sum 2), so within Σ_{l ≥ 2m − degree} (2l + 1)|j_l(κ)| over the integral's scale 2:
/// the least m that holds that within the unit round-off -- the phase over four and the Bessel functions' tail past
/// l ≈ κ, which falls faster than any power --, not fewer than `least`. Measured tight (CARRIER: at κ = 20, 150, 600
/// the rule of 27, 106, 349 nodes errs by 10⁻¹⁶–10⁻¹⁵, a tenth fewer nodes by 3·10⁻¹⁴, 6·10⁻¹⁰, 5·10⁻⁴). j_l by
/// Miller's backward recurrence from where it has fallen below the round-off (`grown`), normalized by
/// Σ(2l + 1)j_l² = 1.
fn fewest(nu: f64, width: f64, least: usize, degree: usize) -> usize {
    let kappa = 0.5 * nu * width.abs();
    // A beat that is not a number would size the rule by a recurrence that stops at once: no rule is sized for it.
    assert!(!kappa.is_nan(), "a tone's frequency is not a number");
    if kappa <= 0.0 {
        return least;
    }
    // No finite bandwidth -- ∞ or NaN, never read as none -- has a rule: its caller refuses it first (`Midway::stretch`).
    if !kappa.is_finite() {
        return usize::MAX;
    }
    BESSELS.with(|f| {
        let f = &mut *f.borrow_mut();
        let top = grown(kappa, grown(kappa, kappa.ceil() as usize + 1).max(2 * least));
        f.clear();
        f.resize(top + 2, 0.0);
        f[top] = 1.0;
        for l in (1..=top).rev() {
            f[l - 1] = (2 * l + 1) as f64 / kappa * f[l] - f[l + 1];
            if f[l - 1].abs() > 1e100 {
                f[l - 1..].iter_mut().for_each(|v| *v *= 1e-100);
            }
        }
        let norm = f.iter().enumerate().map(|(l, v)| (2 * l + 1) as f64 * v * v).sum::<f64>().sqrt();
        let mut tail = 0.0;
        let start = (0..=top).rev().find(|&l| {
            tail += (2 * l + 1) as f64 * f[l].abs() / norm;
            tail > UNIT_ROUNDOFF
        });
        start.map_or(least, |l| (l + 1 + degree).div_ceil(2).max(least))
    })
}

/// The panels' count over a piece of the width w at the bandwidth ν (`panels`; `fewest` of `nodes` nodes and more, on
/// an integrand times a polynomial of `degree`): the n that leaves the fewest nodes to take, each node a state and its
/// sums (`Midway::stretch`). The rules are made once for the process (`rule`): their making is no part of a piece's
/// cost. On n equal panels of m = m(νw/n) nodes each (m(κ) ≈ κ/2 and the Bessel tail's few, the tail each panel's own):
/// - a piece the survival does not end in takes them all: C(n) = n m;
/// - a piece a jump's root lies in takes the panels before the root's, the root's panel's nodes as far as the norm's
///   crossing -- which the root takes again, by its span from the panel's start (`root`, `fallen`) -- and that span,
///   and no node past the root. Over the root's place -- uniform in the piece on the trajectories' thresholds, which are
///   uniform in the survival, the survival taken linear between the piece's ends --, (n − 1)/2 panels before it, half
///   the root's panel's nodes (a rule's nodes lie symmetric in its panel: the share before a uniform point is ½ on the
///   mean) and the span at its mean, half the panel: C(n) = (n/2) m + m(νw/(2n)) -- the tail n times against what the
///   root takes again, a panel's.
/// Which one a piece is, `ends` tells (the propagation's norm at the piece's end at the trajectory's threshold or below:
/// the survival to the propagation's accuracy, a stop's cost), asked only where the root's n is not one. n is the first
/// past which C does not fall (each part of C is monotone in n, but for the node counts' steps of one). Returns n and
/// its panels' m, each m(n) made once (`fewest` is a recurrence of some κ terms; the two costs ask the same n again,
/// and until 2026-11-21 the panels made their m once more: 2 to 3 % of a step, PGC3D, R4, tilt0).
fn count(width: f64, bandwidth: f64, nodes: usize, degree: usize, ends: impl FnOnce() -> bool) -> (usize, usize) {
    let mut known = [0usize; 256];
    let mut m = |n: usize| -> usize {
        if let Some(&m) = known.get(n).filter(|&&m| m != 0) {
            return m;
        }
        let m = fewest(bandwidth, width / n as f64, nodes, degree);
        if let Some(slot) = known.get_mut(n) {
            *slot = m;
        }
        m
    };
    let least = |m: &mut dyn FnMut(usize) -> usize, cost: &dyn Fn(usize, &mut dyn FnMut(usize) -> usize) -> f64| {
        let (mut n, mut best) = (1, cost(1, &mut *m));
        loop {
            let next = cost(n + 1, &mut *m);
            if !(next < best) {
                break n;
            }
            (n, best) = (n + 1, next);
        }
    };
    let rooted = least(&mut m, &|n, m| 0.5 * n as f64 * m(n) as f64 + m(2 * n) as f64);
    let n = if rooted > 1 && ends() { rooted } else { least(&mut m, &|n, m| n as f64 * m(n) as f64) };
    (n, m(n))
}

/// The `count` equal panels over [a, b] of `nodes` nodes each (`count`: the fewest that integrate each to the
/// round-off), in their order from a to b: each panel's end (b itself for the last) and its nodes and weights in their
/// order.
fn panels(a: f64, b: f64, (count, nodes): (usize, usize)) -> impl Iterator<Item = (f64, impl Iterator<Item = (f64, f64)>)> {
    let w = (b - a) / count as f64;
    let rule = rule(nodes);
    (0..count).map(move |p| {
        let (centre, end, rule) = (a + (p as f64 + 0.5) * w, if p + 1 == count { b } else { a + (p + 1) as f64 * w }, rule.clone());
        (end, (0..rule.nodes.len()).rev().map(move |k| (centre + 0.5 * w * rule.nodes[k], 0.5 * w * rule.weights[k])))
    })
}

/// The nodes and weights over [a, b] of an integrand entire at the bandwidth ν times a polynomial of `degree`, to the
/// round-off, where no jump's root is sought (`count`).
fn quadrature(a: f64, b: f64, bandwidth: f64, nodes: usize, degree: usize) -> impl Iterator<Item = (f64, f64)> {
    panels(a, b, count(b - a, bandwidth, nodes, degree, || false)).flat_map(|(_, nodes)| nodes)
}

// ---------------------------------------------------------------------------------------------------------------------
// The step's path and its moments.

/// The fewest nodes of a path's moments (`Segment::moments`): in free flight under a force linear in the time κ is a
/// quadratic and κ² a quartic of the time, which the rule of three nodes integrates exactly (to the degree 5).
const MOMENTS: usize = 3;

/// A step's path (`Step`; module documentation, # The path between a step's ends): the reference's flow under the
/// step's kick spread over the step -- the force linear in the time that ends the step where the kick at the middle
/// ends it --, from the step's start, or from one of the step's jumps with the momentum the jump leaves (`after`).
/// Second order in τ at every time of the step, the step's own start and end at its two ends. T's window means are
/// its integrals of κ (`Segment::moments`, SPEC §1.7), an output inside the step its value (`Segment::at`,
/// `Segment::place`).
#[derive(Clone, Debug, PartialEq)]
pub struct Segment {
    /// The time the trajectory is on this path from (µs): the step's start, or a jump inside the step.
    pub start: f64,
    /// The step's own start time and its length (µs; negative for a reverse step): the stretch its kick and its part
    /// of the Brownian path are spread over.
    pub step: (f64, f64),
    /// The reference flow of the step.
    pub flow: Flow,
    /// (r, κ) at `start`.
    pub from: ([f64; 3], [f64; 3]),
    /// Per principal axis of the flow the kick's force a + b s, s the time from `start` (`Flow::driven`).
    drive: [[f64; 2]; 3],
    /// The factor Λ of the scattered light's diffusion the step put its part of the Brownian path through (`factor`;
    /// 1/µm/√µs): a state read inside the step draws through it what the step's moments do not tell (`place`).
    spread: [[f64; 3]; 3],
}

impl Segment {
    /// The path of a step of τ from the time `start`: its reference flow, its start, the kick at its middle beside
    /// its jumps' -- the displacement and the momentum, lab axes -- and the diffusion's factor Λ. Along a principal
    /// axis (x_h = (ħ/m)λh², h = τ/2) a force α + βu, u the time from the middle, ends the step as the kick
    /// ∫cos(ωu)(α + βu)du = α τ c₁(x_h) and the displacement −(ħ/m)∫(sin ωu/ω)(α + βu)du = −(ħ/m) β 2h³(c₂ − c₃)(x_h) at
    /// the middle do; so α and β from the kick, and a = α − βh, b = β from the start.
    fn new(start: f64, tau: f64, flow: Flow, from: ([f64; 3], [f64; 3]), kick: ([f64; 3], [f64; 3]), spread: [[f64; 3]; 3]) -> Segment {
        let (h, hm) = (0.5 * tau, flow.hbar_over_mass);
        let drive = std::array::from_fn(|i| {
            let x = hm * flow.curvatures[i] * h * h;
            let (_, c1, c2) = stumpff(x);
            let beta = -linalg::dot3(flow.axes[i], kick.0) / (hm * 2.0 * h * h * h * (c2 - stumpff3(x)));
            [linalg::dot3(flow.axes[i], kick.1) / (tau * c1) - beta * h, beta]
        });
        Segment { start, step: (start, tau), flow, from, drive, spread }
    }
    /// The path from a jump at a time of it on: from its place there with the momentum the jump's kick leaves, under
    /// the same force.
    pub fn after(&self, time: f64, kick: [f64; 3]) -> Segment {
        let ((r, kappa), s) = (self.at(time), time - self.start);
        Segment { start: time, from: (r, std::array::from_fn(|a| kappa[a] + kick[a])), drive: self.drive.map(|[a, b]| [a + b * s, b]), ..self.clone() }
    }
    /// (r, κ) at a time of the step, on its path.
    pub fn at(&self, time: f64) -> ([f64; 3], [f64; 3]) {
        self.flow.driven(self.from, time - self.start, Some(&self.drive)).0
    }
    /// The position and the momentum of a state of the trajectory at its time, on the path: the path's mean given the
    /// step's two moments of its Brownian path, and what they do not tell of it at this time (`untold`: none at the
    /// step's ends) drawn from the stream of the time (`random::Stream::bridge`) through the step's own factor Λ, so
    /// that the momentum read has the diffusion's spread up to the time and not only at the ends.
    pub fn place(&self, state: &mut Trajectory) {
        let (start, tau) = self.step;
        (state.r, state.kappa) = self.at(state.time);
        let untold = (tau.abs() * Segment::untold((state.time - start) / tau)).sqrt();
        let deviates = Stream::bridge(state.seed, state.index, state.time).normals();
        for a in 0..3 {
            state.kappa[a] += untold * (0..3).map(|c| self.spread[a][c] * deviates[c]).sum::<f64>();
        }
    }
    /// What the step's two moments of its Brownian path do not tell of it at the share θ of the step, over |τ|: the
    /// variance σ(1 − 3σ), σ = θ(1 − θ), of the bridge about its mean given its integral (the path is s ΔW/τ + B, B
    /// the bridge, Var B = τσ; ΔZ = −∫B, Cov(B(s), ∫B) = τ²σ/2, Var ∫B = τ³/12). Zero at the step's ends, 1/12 at the
    /// most; a time within the rounding of an end is the end.
    fn untold(theta: f64) -> f64 {
        let sigma = linalg::larger(theta * (1.0 - theta), 0.0);
        sigma * (1.0 - 3.0 * sigma)
    }
    /// ∫κ_a dt and ∫κ_a² dt (1/µm·µs, 1/µm²·µs) per lab axis over [from, to] within the path's part of the step:
    /// Gauss–Legendre to round-off on the path, κ² of bandwidth twice the flow's rate (`quadrature`). κ² in the mean
    /// over the Brownian path: the path is the mean given the step's two moments, and the square of what they do not
    /// tell is added, (Λ Λᵀ)_aa |τ| ∫σ(1 − 3σ)dt (`untold`; ∫₀^θ σ(1 − 3σ)dθ = θ²/2 − 4θ³/3 + 3θ⁴/2 − 3θ⁵/5, 1/15 over the
    /// step) -- so that a cut of a step has the diffusion's input up to it, as the whole step has. (The residual is the
    /// free flight's; the reference's rotation changes it by its (ωτ)² share, the third order.)
    pub fn moments(&self, from: f64, to: f64) -> ([f64; 3], [f64; 3]) {
        let (lo, hi) = (linalg::smaller(from, to), linalg::larger(from, to));
        let sign = if to >= from { 1.0 } else { -1.0 };
        let (mut one, mut two) = ([0.0; 3], [0.0; 3]);
        if !(hi > lo) {
            return (one, two);
        }
        for (t, w) in quadrature(lo, hi, 2.0 * self.flow.rate(), MOMENTS, 0) {
            let k = self.at(t).1;
            for x in 0..3 {
                one[x] += sign * w * k[x];
                two[x] += sign * w * k[x] * k[x];
            }
        }
        let (start, tau) = self.step;
        let integral = |time: f64| {
            let theta = (time - start) / tau;
            theta * theta * (0.5 + theta * (-4.0 / 3.0 + theta * (1.5 - 0.6 * theta)))
        };
        let untold = sign * tau * tau * (integral(hi) - integral(lo)).abs();
        for x in 0..3 {
            two[x] += self.spread[x].iter().map(|l| l * l).sum::<f64>() * untold;
        }
        (one, two)
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The generator along the path (B).

/// The cluster's generator along the step's path in its Doppler frame (module documentation, # The kick): the model's
/// generator at the middle (`Middle`) with the path through it, and per state the gauge's rate K_s·v -- at the time u
/// from the middle the lab's state is ψ_s = e^{iK_s·v u} x_s of the frame's.
struct Along<'a, M: Middle> {
    middle: &'a M,
    path: M::Path,
    gauge: [f64; STATES],
    n: usize,
}

impl<'a, M: Middle> Along<'a, M> {
    /// The path through a middle at the velocity v, its generator at the momentum κ.
    fn new(middle: &'a M, kappa: [f64; 3], velocity: [f64; 3]) -> Along<'a, M> {
        let (n, path) = (middle.states(), middle.path(kappa, velocity));
        let mut gauge = [0.0; STATES];
        middle.gauge(&path, &mut gauge[..n]);
        Along { middle, path, gauge, n }
    }
    /// A state of the lab at the time u from the middle in the frame (`to` the frame), or back.
    fn turned(&self, state: &[C64], u: f64, to: bool) -> [C64; STATES] {
        let mut out = [C64::ZERO; STATES];
        for ((out, z), rate) in out.iter_mut().zip(state).zip(&self.gauge) {
            let phase = C64::cis(rate * u);
            *out = *z * if to { phase.conj() } else { phase };
        }
        out
    }
}

/// The most sums `accumulate` takes at once: a beating path's three nodes'.
const SUMS: usize = 3;

/// An exact tone's sums (`Sums`): its density at each of the step's three nodes' Lagrange weights, times cos νu and sin νu.
const TONE: usize = 2 * SUMS;

/// The fewest nodes of the kick's quadrature (`propagate`): the smallest rule's -- the forms on a state the generator
/// propagates are entire in the time at their bandwidth, and nothing more is asked of the rule where it vanishes.
const FORMS: usize = 2;

/// The reals of a row of n states in whole vectors of `LANES` (zero beyond the states).
const fn row(n: usize) -> usize {
    n.div_ceil(LANES) * LANES
}

/// Hermitian sums, each with its weight times a state's density more: sums[j] += w_j |x⟩⟨x|/⟨x|x⟩ on the entries
/// a ≤ b (`whole` reads those and mirrors them). A sum is held row by row, a row's real parts and then its imaginary
/// parts, each of `row(n)` reals. Row a takes x_a/⟨x|x⟩ times the state's conjugate from the vector of its own column
/// on, lane by lane with fused multiply-adds (as the kernels of linalg.rs): two a real for the product, one a sum.
fn accumulate(x: &[C64], weights: &[f64], sums: &mut [f64]) {
    use std::arch::x86_64::{_mm256_fmadd_pd, _mm256_loadu_pd, _mm256_mul_pd, _mm256_set1_pd, _mm256_storeu_pd, _mm256_xor_pd};
    let (n, scale) = (x.len(), 1.0 / norm_sqr(x));
    let width = row(n);
    assert!(n <= STATES && sums.len() >= weights.len() * 2 * n * width);
    let (mut re, mut im) = ([0.0; row(STATES)], [0.0; row(STATES)]);
    for (z, (re, im)) in x.iter().zip(re.iter_mut().zip(&mut im)) {
        (*re, *im) = (z.re, z.im);
    }
    // SAFETY: every vector read and written lies within the row's `width` reals of `re`, `im` and each sum (above); the
    // build has AVX2 and FMA (linalg.rs).
    unsafe {
        let negative = _mm256_set1_pd(-0.0);
        for (a, left) in x.iter().enumerate() {
            let (left, from) = (*left * scale, a / LANES * LANES);
            let (real_part, imaginary_part) = (_mm256_set1_pd(left.re), _mm256_set1_pd(left.im));
            for column in (from..width).step_by(LANES) {
                // x_a x_b*/⟨x|x⟩ on a vector of columns b: (left.re re + left.im im, left.im re − left.re im).
                let (r, i) = (_mm256_loadu_pd(re.as_ptr().add(column)), _mm256_loadu_pd(im.as_ptr().add(column)));
                let real = _mm256_fmadd_pd(real_part, r, _mm256_mul_pd(imaginary_part, i));
                let imaginary = _mm256_fmadd_pd(imaginary_part, r, _mm256_xor_pd(_mm256_mul_pd(real_part, i), negative));
                let place = 2 * a * width + column;
                for (j, w) in weights.iter().enumerate() {
                    let (w, head) = (_mm256_set1_pd(*w), sums.as_mut_ptr().add(j * 2 * n * width + place));
                    _mm256_storeu_pd(head, _mm256_fmadd_pd(w, real, _mm256_loadu_pd(head)));
                    _mm256_storeu_pd(head.add(width), _mm256_fmadd_pd(w, imaginary, _mm256_loadu_pd(head.add(width))));
                }
            }
        }
    }
}

/// The Hermitian sum j of those `accumulate` holds, whole and row by row: its entries a ≤ b and their mirror.
fn whole(sums: &[f64], j: usize, n: usize, sum: &mut [C64]) {
    let width = row(n);
    for a in 0..n {
        for b in a..n {
            let at = (2 * (j * n + a)) * width + b;
            let z = C64::new(sums[at], sums[at + width]);
            sum[a * n + b] = z;
            if b > a {
                sum[b * n + a] = z.conj();
            }
        }
    }
}

/// A state's held forms (module documentation, # The path's momentum): where the model last evaluated it, at a middle
/// point and the time `time` (µs), the kick's integrand there -- the static form of the conservative force less the
/// reference's gradient, Re tr(Ĝρ) − ∇V_ref per lab axis (rad/µs/µm) -- and per tone its beat ν (rad/µs) and complex
/// amplitude a per lab axis, the tone's form at the time t being Re[a e^{iν(t − time)}]. What a trajectory carries to
/// predict its next step's path momentum (`predicted`).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Forms {
    pub time: f64,
    pub held: [f64; 3],
    pub tones: Vec<(f64, [C64; 3])>,
}

impl Forms {
    /// The model's forms less the reference's gradient at the point they were read at: the kick's integrand.
    pub fn less(mut self, gradient: [f64; 3]) -> Forms {
        for (f, g) in self.held.iter_mut().zip(gradient) {
            *f -= g;
        }
        self
    }
}

/// A state's forms at a middle on the path at rest through it, `fixed` (`Forms`, its time the middle's: 0; the model's,
/// the reference's gradient not taken), by the middle's forms, for a model that gives them no other way
/// (`Middle::nodes`, `Dynamics::forms`): the fixed point's tones -- its candidates, the light's beats in time alone, not
/// the pattern's turns along a path -- each by its complex amplitude c = F(ρ) − iF(iρ) of its forms on σ = ρ and σ = iρ
/// (`Middle::tone_force`), the static form the whole one at the middle (`Middle::force`) less the tones' there.
pub fn forms_at<M: Middle>(middle: &M, fixed: &M::Path, rho: &[C64]) -> Forms {
    let mut held = middle.force(fixed, 0.0, rho);
    let (mut beats, mut keys) = (vec![], vec![]);
    middle.beats(fixed, &mut beats, &mut keys);
    let none = vec![C64::ZERO; rho.len()];
    let tones = beats
        .iter()
        .enumerate()
        .map(|(m, &nu)| {
            let (re, im) = (middle.tone_force(fixed, m, &[(0.0, rho, &none)]), middle.tone_force(fixed, m, &[(0.0, &none, rho)]));
            let c: [C64; 3] = std::array::from_fn(|a| C64::new(re[a], -im[a]));
            for a in 0..3 {
                held[a] -= c[a].re;
            }
            (nu, c)
        })
        .collect();
    Forms { time: 0.0, held, tones }
}

/// The nodes' sums of a tone summed, Σ_j re_j and Σ_j im_j (`Middle::tone_force`): the whole σ, for a model whose
/// envelopes are the same at every time of the path.
pub fn summed(sums: &[(f64, &[C64], &[C64])]) -> (Vec<C64>, Vec<C64>) {
    let size = sums.first().map_or(0, |x| x.1.len());
    let (mut re, mut im) = (vec![C64::ZERO; size], vec![C64::ZERO; size]);
    for (_, r, i) in sums {
        re.iter_mut().zip(r.iter()).for_each(|(a, b)| *a += *b);
        im.iter_mut().zip(i.iter()).for_each(|(a, b)| *a += *b);
    }
    (re, im)
}

/// A state's density matrix ρ = |x⟩⟨x|/⟨x|x⟩, ρ_ab = x_a x_b*/‖x‖², row by row: a mean ⟨x|G|x⟩/⟨x|x⟩ is tr(Gρ), which the
/// model contracts (`Middle::force`, `Middle::tones`).
pub fn density(x: &[C64], rho: &mut [C64]) {
    let (n, scale) = (x.len(), 1.0 / norm_sqr(x));
    for (a, left) in x.iter().enumerate() {
        let left = *left * scale;
        for (b, right) in x.iter().enumerate() {
            rho[a * n + b] = left * right.conj();
        }
    }
}

/// A trajectory's scratch memory for its steps, made with it and fitted to each step's cluster (`Scratch::fit`): the
/// generator's picture along the step's path (`spectral::Picture`: the generator at the step's nodes, its decomposition,
/// its following, the state's path), one array carved (`Scratch::carve`) into the middle state's density and a
/// Hermitian sum the model takes, and four laid as `accumulate` holds its sums: the nodes' weighted sums of the states'
/// densities, the sums as they stood where the panel being propagated began (`Midway::stretch`), a span's own inside a
/// panel (`fallen`), and the generator's decay laid on the sums (`pictured`); and the states a panel keeps -- its
/// stops and its readings -- until its root is known (`Midway::stretch`). Every use writes what it reads: nothing of a
/// step is carried to the next one, and a trajectory taken up again with a fresh scratch goes on as itself, to the bit.
/// Beside, the counted work of the attempts refused on it (`refused`), which the solve's price reads and no step does.
/// (Until 2026-11-21 also the count of its steps refused for an entry of no route, which the solve's probe read before
/// the ladder's price took the refused attempts' work: read by nothing since, removed.)
/// (Until 2026-11-20 the scratch carried its picture's counts of the decomposition's route, which the next step's
/// route was priced against: the strides', spectral.rs, # Where the decomposition does not take the step.)
#[derive(Default)]
pub struct Scratch {
    picture: Picture,
    rest: Vec<C64>,
    sums: Vec<f64>,
    met: Vec<(Stop, f64, [C64; STATES])>,
    noise: Noise,
    /// The sums' count (`Sums`), and the weights' sums of the step and as they stood at a panel's start.
    slots: usize,
    weights: Vec<f64>,
    kept: Vec<f64>,
    /// The counted work of the attempts refused on the scratch (`refused`), and how far the picture of the stretch being
    /// taken has gone (`pictured`).
    refused: f64,
    stage: Stage,
}

/// How far a stretch's picture has gone (`pictured`), for the count of a stretch refused where it stood (`unfinished`):
/// nothing yet; the model's generator at the step's nodes made (a refusal there comes after all of them); the tones
/// separated (`spectral::Picture::separate`); the decay laid and the preparation begun -- the picture's own count
/// (`spectral::Picture::work`) is the stretch's from there on.
#[derive(Clone, Copy, Default, PartialEq, PartialOrd)]
enum Stage {
    #[default]
    None,
    Made,
    Separated,
    Prepared,
}

/// What a stretch has done as far as it went, beside its picture (`Midway::stretch`): its states and whether its
/// generator is still, the nodes gathered into the sums and the hazard's traces (`Price::nodes`, `Price::traces`), and
/// where it passed the middle the scattered light's count (`Price::scattering`) -- what a refused stretch's count reads
/// (`unfinished`).
#[derive(Clone, Copy, Default)]
struct Progress {
    states: usize,
    still: bool,
    counted: usize,
    traces: usize,
    scattering: f64,
}

struct Carved<'a> {
    picture: &'a mut Picture,
    noise: &'a mut Noise,
    held: &'a mut [C64],
    sum: &'a mut [C64],
    tones: &'a mut [C64],
    sums: &'a mut [f64],
    saved: &'a mut [f64],
    part: &'a mut [f64],
    decay: &'a mut [f64],
    met: &'a mut Vec<(Stop, f64, [C64; STATES])>,
    weights: &'a mut [f64],
    kept: &'a mut [f64],
}

impl Scratch {
    /// The counted work (real multiply-adds, `Step::work`'s) of every attempt on this scratch that was refused, since the
    /// scratch was made (`Midway::attempt`): what each did before its refusal, by the counts its price would have had
    /// (`Price::work`, `unfinished`) -- its stretches finished before the refused one, and of that one its middle's
    /// making, its generator at the nodes, its tones' envelopes, the picture's own count as far as it went, the decay laid
    /// with the hazard's traces taken, the nodes gathered, and the scattered light where it was made. None of a step's
    /// results reads it.
    pub fn refused(&self) -> f64 {
        self.refused
    }

    /// The scratch for a cluster of n states and `slots` sums (`Sums`): the weights' sums of the three nodes kept, a
    /// step's earlier stretches' (`Midway::stretch`).
    fn fit(&mut self, n: usize, slots: usize) {
        self.picture.fit(n);
        self.slots = slots;
        self.rest.resize((4 + TONE * (slots.saturating_sub(SUMS) / TONE)) * n * n, C64::ZERO);
        self.sums.resize(4 * slots * 2 * n * row(n), 0.0);
        self.weights.resize(slots.max(SUMS), 0.0);
        self.kept.resize(slots.max(SUMS), 0.0);
    }
    /// The scratch's parts.
    fn carve(&mut self) -> Carved<'_> {
        let n = self.picture.states();
        let (held, rest) = self.rest.split_at_mut(n * n);
        let (sum, tones) = rest.split_at_mut(3 * n * n);
        let size = self.slots * 2 * n * row(n);
        let (sums, rest) = self.sums.split_at_mut(size);
        let (saved, rest) = rest.split_at_mut(size);
        let (part, rest) = rest.split_at_mut(size);
        // The decay on every sum: the nodes' and the tones' (`pictured`).
        let decay = &mut rest[..size];
        let width = self.slots.max(SUMS);
        let (weights, kept) = (&mut self.weights[..width], &mut self.kept[..width]);
        Carved { picture: &mut self.picture, noise: &mut self.noise, held, sum, tones, sums, saved, part, decay, met: &mut self.met, weights, kept }
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// The propagations.

/// The counted work (`decide::Work`, real multiply-adds) of the integrator's operations on n states, one formula each,
/// the model's own counts of its generator, its force and its tones at a node with them (`Middle::work`): what a
/// stretch adds up (`Price::work`): its count and its price at once.
mod count {
    use super::COMPLEX;
    /// The propagation: the model's generator at the step's nodes -- three, or the one of a path it is still along --
    /// and the picture's own count of its complex multiply-adds (`spectral::Picture::work`: the decomposition, the
    /// slopes, the following, the hops, the states at the nodes).
    pub fn propagation(model: [f64; 3], still: bool, picture: f64, tones: f64) -> f64 {
        COMPLEX * (if still { 1.0 } else { 3.0 } * model[0] + picture + tones)
    }
    /// The kick's part of the state through `nodes` nodes: per node its density's upper triangle into the weighted
    /// sums (one on a still path, three on a beating one); then the model's one form of the sum, or on a beating path
    /// the middle state's density, laid with the middle node's sum, and the three nodes' forms -- where the stretch
    /// holds the step's middle (`centred`), the middle's with the fixed point's tones on the middle state
    /// (`Middle::centre`: a form and a `tones`). Of the exact tones (`tones` of them, their forms' counted work
    /// `forces`): per node and tone a density again and its six sums (`super::Sums`); their two matrices, each its three
    /// nodes' sums added, with the nodes' sums at their phases, and their forms.
    pub fn impulse(n: usize, model: [f64; 3], nodes: usize, still: bool, centred: bool, tones: usize, forces: f64) -> f64 {
        let forms = if still { model[1] } else { (2 * n * n) as f64 + 3.0 * model[1] + if centred { model[2] } else { 0.0 } };
        gathered(n, nodes, still, tones) + COMPLEX * (forms + (16 * tones * n * n) as f64 + forces)
    }
    /// The kick's nodes alone (`impulse`'s first part): per node its density's upper triangle into the weighted sums,
    /// and per node and tone a density again and its six sums.
    pub fn gathered(n: usize, nodes: usize, still: bool, tones: usize) -> f64 {
        let sums = if still { 1 } else { 3 };
        COMPLEX * (nodes * ((1 + sums) * n * (n + 1) / 2 + (1 + super::TONE) * tones * n * (n + 1) / 2)) as f64
    }
    /// The end state's held forms (# The path's momentum): its density and one more node's forms with its tones'.
    pub fn end(n: usize, model: [f64; 3], tones: f64) -> f64 {
        COMPLEX * ((n * n) as f64 + model[1] + tones)
    }
    /// The survival's hazard (# Jumps): the generator's decay laid on the sums -- at the step's nodes, one or three on a
    /// beating path, and each of the `tones` exact tones' six (`super::Sums`) -- once, and `traces` of it on the sums,
    /// each over the same upper triangles.
    pub fn hazard(n: usize, traces: usize, still: bool, tones: usize) -> f64 {
        let sums = if still { 1 } else { super::SUMS + super::TONE * tones };
        COMPLEX * ((1 + traces) * sums * n * (n + 1) / 2) as f64
    }
}

/// The node count of a quadrature over τ at a bandwidth (`fewest`, with the three nodes' Lagrange weights).
#[cfg(test)]
fn nodes(tau: f64, bandwidth: f64) -> usize {
    fewest(bandwidth, tau, FORMS, 2)
}

/// The picture of the generator along a step's path (`spectral::Picture`), prepared on `start`, the frame's state the
/// step is launched with (none: any; `Picture::prepare`'s reached states): the model's generator at the step's middle
/// and, unless it is still along every path, at the step's two ends -- its static group, and its exact tones' envelopes
/// at the three nodes (`Middle::beats`). The decay is laid into `decay` first in the order of the kick's sums
/// (`Sums`): the static group's node by node (−h, 0, h, or the middle alone where Ĥ is still), then each tone's at each
/// node at the phases 0 and π/2 -- the survival's hazard (# Jumps). Returns the largest sum over the nodes of the moduli
/// the decay's entries can reach (the static group's and each tone's two at once); a generator at a node that is no
/// number refuses the step (`Refused::Overflow`: nothing of it can be propagated), and so does a step the decomposition
/// does not take (`Refused::Propagation`).
fn pictured<M: Middle>(along: &Along<M>, scratch: &mut Scratch, tau: f64, start: Option<&[C64]>) -> Result<f64, Refused> {
    let _p = crate::prof::scope(crate::prof::PICTURED);
    let (h, still, n) = (0.5 * tau, along.middle.still(), along.n);
    scratch.stage = Stage::None;
    let picture = &mut scratch.picture;
    picture.fit(n);
    if picture.blocks.as_slice() != along.middle.blocks() {
        picture.blocks.clear();
        picture.blocks.extend_from_slice(along.middle.blocks());
    }
    let _g = crate::prof::scope(crate::prof::PIC_GENERATOR);
    along.middle.generator(&along.path, 0.0, &mut picture.base);
    if !still {
        along.middle.generator(&along.path, -h, &mut picture.ends[0]);
        along.middle.generator(&along.path, h, &mut picture.ends[1]);
    }
    drop(_g);
    scratch.stage = Stage::Made;
    let picture = &mut scratch.picture;
    let numbers = |m: &Mat| m.data.iter().all(|z| z.re.is_finite() && z.im.is_finite());
    if !(numbers(&picture.base) && (still || picture.ends.iter().all(numbers))) {
        return Err(Refused::Overflow);
    }
    let _s = crate::prof::scope(crate::prof::PIC_SEPARATE);
    picture.separate(
        h,
        |beats, keys| {
            if !still {
                along.middle.beats(&along.path, beats, keys);
            }
        },
        |m, s, plus, minus| along.middle.envelopes(&along.path, m, s, plus, minus),
    );
    drop(_s);
    scratch.stage = Stage::Separated;
    let picture = &mut scratch.picture;
    if !picture.beats.iter().flat_map(|tone| tone.members.iter().flatten()).all(numbers) {
        return Err(Refused::Overflow);
    }
    let slots = Sums { still, h, tones: &picture.beats }.count();
    scratch.fit(n, slots);
    let Carved { picture, decay, .. } = scratch.carve();
    decay.fill(0.0);
    let generators = if still { [Some(&picture.base), None, None] } else { [Some(&picture.ends[0]), Some(&picture.base), Some(&picture.ends[1])] };
    // The decay i(X − X†) of an n × n matrix X given entry by entry (row by row).
    fn of(x: impl Fn(usize) -> C64, n: usize) -> impl Fn(usize, usize) -> C64 {
        move |a, b| C64::I * (x(a * n + b) - x(b * n + a).conj())
    }
    // The static group's laid on the nodes' sums, and each tone's at each node on its two sums there: its decay at the
    // phases 0 and π/2, of G⁺ + G⁻ and of i(G⁺ − G⁻) (`Sums`). Per node the sum of the moduli its decay can reach.
    let mut sizes = [0.0; SUMS];
    for ((j, g), size) in generators.into_iter().flatten().enumerate().zip(&mut sizes) {
        *size = laid(of(|k| g.data[k], n), n, j, decay);
    }
    for (m, tone) in picture.beats.iter().enumerate() {
        for (j, size) in sizes.iter_mut().enumerate() {
            let [plus, minus] = [&tone.members[0][j].data, &tone.members[1][j].data];
            let at = SUMS + TONE * m + 2 * j;
            *size += laid(of(|k| plus[k] + minus[k], n), n, at, decay) + laid(of(|k| C64::I * (plus[k] - minus[k]), n), n, at + 1, decay);
        }
    }
    let scale = sizes.into_iter().fold(0.0, linalg::larger);
    let _r = crate::prof::scope(crate::prof::PIC_PREPARE);
    scratch.stage = Stage::Prepared;
    scratch.picture.prepare(h, still, start).map_err(Refused::Propagation)?;
    Ok(scale)
}

/// The counted work of a stretch refused where it stood (`Scratch::refused`), by its price's counts as far as it went
/// (`Price`, `count`): the middle's making (`Price::middle`); where its picture made the generator at the nodes, those
/// (a refusal in the picture comes after all of them), the tones' envelopes where they were separated, the picture's own
/// count where its preparation began (the decay laid with it, and the hazard's traces taken); the nodes gathered; the
/// scattered light where the stretch passed the middle. Not the middle's forms, the end state's held forms nor the
/// optical sector: each is made past the last place a stretch is refused before it.
fn unfinished<M: Middle>(generator: &M, progress: &Progress, scratch: &Scratch) -> f64 {
    let (n, still, picture) = (progress.states, progress.still, &scratch.picture);
    let costs = generator.work();
    let (tones, toned) = if scratch.stage >= Stage::Separated { (picture.beats.len(), toned(generator, picture, n)) } else { (0, [0.0; 2]) };
    let made = if scratch.stage >= Stage::Made { count::propagation(costs, still, if scratch.stage == Stage::Prepared { picture.work } else { 0.0 }, toned[0]) } else { 0.0 };
    let laid = if scratch.stage == Stage::Prepared { count::hazard(n, progress.traces, still, tones) } else { 0.0 };
    generator.made() + made + laid + count::gathered(n, progress.counted, still, tones) + progress.scattering
}

/// A stretch's tones' counts (`Price::toned`): their envelopes at the three nodes, the candidates weighed and left in the
/// static group at the middle alone, the weighing (`spectral::Picture::separate`) and the tones' sums; and their forms'.
fn toned<M: Middle>(generator: &M, picture: &Picture, n: usize) -> [f64; 2] {
    let [envelopes, forces] = picture.beats.iter().fold([0.0; 2], |sum, tone| {
        let work = generator.tone_work(tone.candidate);
        [sum[0] + work[0], sum[1] + work[1]]
    });
    let envelopes = envelopes + picture.weighed.iter().map(|&m| generator.tone_work(m)[0] / 3.0).sum::<f64>() + picture.weighing;
    [envelopes + (6 * picture.beats.len() * n * n) as f64, forces]
}

/// A Hermitian matrix Γ (by its entries `gamma`) laid as the sum j of those `accumulate` holds: on each entry a ≤ b its
/// real and imaginary parts, twice off the diagonal, so that `hazard` -- the sum of the two laid arrays' products -- is
/// tr(Γ S) = Σ_a Γ_aa S_aa + 2 Re Σ_{a<b} Γ_ab S_ab* of a Hermitian S held there. A generator's decay is
/// Γ = i(Ĥ − Ĥ†) (Γ_aa = −2 Im Ĥ_aa). Returns Σ_ab |Γ_ab|.
fn laid(gamma: impl Fn(usize, usize) -> C64, n: usize, j: usize, decay: &mut [f64]) -> f64 {
    let (width, mut size) = (row(n), 0.0);
    for a in 0..n {
        for b in a..n {
            let (gamma, twice) = (gamma(a, b), if a == b { 1.0 } else { 2.0 });
            let at = 2 * (j * n + a) * width + b;
            (decay[at], decay[at + width]) = (twice * gamma.re, twice * gamma.im);
            size += twice * gamma.abs();
        }
    }
    size
}

/// ∫h du over the nodes a laid sum holds (# Jumps): Σ_j tr(Γ_j S_j) over the laid decays and the sums `accumulate` made
/// (`pictured`, `Sums`) -- of the static group Γ_j its decay at the step's node j and S_j = Σ_k w_k ℓ_j(u_k) x̂_k x̂_k†, of
/// a tone its decay at the phases 0 and π/2 at the node j on its sums of ℓ_j(u_k) cos νu_k and ℓ_j(u_k) sin νu_k --:
/// Σ_k w_k x̂_k†Γ(u_k)x̂_k, Γ(u) the decay of the generator the propagation takes (its one matrix where Ĥ is still). On n
/// states, row by row from the vector of the row's own column on (the laid decay is zero below the diagonal), lane by lane.
fn hazard(decay: &[f64], sums: &[f64], n: usize) -> f64 {
    let width = row(n);
    let mut lanes = [0.0; LANES];
    for (at, (g, s)) in decay.chunks_exact(width).zip(sums.chunks_exact(width)).enumerate() {
        let from = at / 2 % n / LANES * LANES;
        for (g, s) in g[from..].chunks_exact(LANES).zip(s[from..].chunks_exact(LANES)) {
            for l in 0..LANES {
                lanes[l] = g[l].mul_add(s[l], lanes[l]);
            }
        }
    }
    lanes.iter().sum()
}

/// ∫h over [from, to] (# Jumps) on the propagation's states at the nodes of the span's own quadrature at the step's
/// bandwidth to the round-off (`Picture::at`: any time after the launch), their densities summed into
/// `part` as the kick's sums are, the weights' sums into `weights` (`gather`): the survival inside a panel, where a
/// jump's root lies -- and, a span before the root, the kick's part of it. Returns it and the nodes taken.
fn fallen(picture: &mut Picture, still: bool, h: f64, (from, to): (f64, f64), bandwidth: f64, decay: &[f64], part: &mut [f64], weights: &mut [f64]) -> (f64, usize) {
    let _p = crate::prof::scope(crate::prof::FALLEN);
    let n = picture.states();
    part.fill(0.0);
    let (mut x, mut nodes) = ([C64::ZERO; STATES], 0);
    for (u, w) in quadrature(from, to, bandwidth, FORMS, 2) {
        picture.at(u, &mut x[..n]);
        Sums { still, h, tones: &picture.beats }.gather(u, w, &x[..n], Some(&mut *weights), part);
        nodes += 1;
    }
    (hazard(decay, part, n), nodes)
}

/// A span's sums (`fallen`) into the kick's: the densities' and the weights'.
fn taken(part: &[f64], spanned: &[f64], sums: &mut [f64], weights: &mut [f64]) {
    for (sum, p) in sums.iter_mut().zip(part) {
        *sum += p;
    }
    for (sum, w) in weights.iter_mut().zip(spanned) {
        *sum += w;
    }
}

/// What the kick's and the survival's sums hold (`accumulate`): one where Ĥ is still along the path, the three nodes'
/// on a beating one (each node's weight times the nodes' Lagrange polynomials), and six of each exact tone of the step
/// (`spectral::Picture::separate`) -- per node j its weight times ℓ_j(u) cos νu and ℓ_j(u) sin νu, the density's sum at
/// e^{iνu} with the node's weight = the first + i the second (their sum over j the tone's whole, Σ_j ℓ_j = 1) --, in that
/// order: the tone's forms at its exact phase, its envelope through the nodes as the propagation takes it.
#[derive(Clone, Copy)]
struct Sums<'a> {
    still: bool,
    h: f64,
    tones: &'a [Beat],
}

impl Sums<'_> {
    /// The sums' count.
    fn count(&self) -> usize {
        if self.still { 1 } else { SUMS + TONE * self.tones.len() }
    }

    /// A node's state at the time u with its weight into the sums, the weights' sums into `weights` where asked.
    fn gather(&self, u: f64, weight: f64, x: &[C64], weights: Option<&mut [f64]>, sums: &mut [f64]) {
        if self.still {
            accumulate(x, &[weight], sums);
            return;
        }
        let s = u / self.h;
        let lagrange = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)].map(|l| weight * l);
        let mut weights = weights;
        if let Some(weights) = weights.as_deref_mut() {
            for (sum, l) in weights.iter_mut().zip(lagrange) {
                *sum += l;
            }
        }
        accumulate(x, &lagrange, sums);
        let size = 2 * x.len() * row(x.len());
        for (m, tone) in self.tones.iter().enumerate() {
            let (sin, cos) = (tone.beat * u).sin_cos();
            let parts: [f64; TONE] = std::array::from_fn(|k| lagrange[k / 2] * if k % 2 == 0 { cos } else { sin });
            accumulate(x, &parts, &mut sums[(SUMS + TONE * m) * size..]);
            if let Some(weights) = weights.as_deref_mut() {
                for (sum, part) in weights[SUMS + TONE * m..].iter_mut().zip(parts) {
                    *sum += part;
                }
            }
        }
    }
}

/// The hazard at a time u of the step (# Jumps): x̂†Γ(u)x̂ of the propagation's state there (`Picture::at`), its
/// density through `part` (`Sums::gather` at the weight one, `hazard`).
fn hazard_at(picture: &mut Picture, still: bool, h: f64, u: f64, decay: &[f64], part: &mut [f64]) -> f64 {
    let _p = crate::prof::scope(crate::prof::HAZARD_AT);
    let n = picture.states();
    part.fill(0.0);
    let mut x = [C64::ZERO; STATES];
    picture.at(u, &mut x[..n]);
    Sums { still, h, tones: &picture.beats }.gather(u, 1.0, &x[..n], None, part);
    hazard(decay, part, n)
}

/// The times read of `reads` (lab times; from the `read`-th on) that the propagation has passed at the time u (from the
/// middle of the step of 2h that starts at `start`), before the step's end -- a time read at the end is the end's own
/// (`Midway::attempt`) --, each the frame's state there (`Picture::at`), kept in `met` with its time until the panel's
/// root is known (`Midway::stretch`). A read is no stop
/// of the propagation and nothing of it is taken up by the step (# The path between a step's ends): the stops and the
/// nodes are the step's alone, and so is the picture's count of its work, which the reads leave as they found it.
/// Returns the next time to read.
fn looked(picture: &mut Picture, reads: &[f64], mut read: usize, (start, h): (f64, f64), u: f64, met: &mut Vec<(Stop, f64, [C64; STATES])>) -> usize {
    let _p = crate::prof::scope(crate::prof::LOOKED);
    let n = picture.states();
    let ahead = |v: f64| (v + h) / (2.0 * h);
    let work = picture.work;
    while let Some(t) = reads.get(read).map(|&time| (time - start) - h).filter(|&t| ahead(t) <= ahead(u) && ahead(t) < 1.0) {
        let mut x = [C64::ZERO; STATES];
        picture.at(t, &mut x[..n]);
        met.push((Stop::Read, t, x));
        read += 1;
    }
    picture.work = work;
    read
}

/// The root in the panel [a, b] of ln N(u) = ln θ (`target`; # Jumps) from the panel's start, ln N(a) = `log`, and
/// with it the kick's part of the panel as far as the root: from the panel's start on, spans of the panel (`fallen`),
/// each from the latest time below the root -- a span that stays below it stands, and its sums are the kick's (`taken`,
/// onto `sums` and `weights` as they stood at the panel's start). `past` is a time known at or past the root with its
/// ∫h from the panel's start -- the panel's end, where its survival read the crossing there --, or none where the
/// crossing is the propagation's norm's (`Midway::stretch`) and the panel's end was not read; `guess` the first time
/// tried (none: the first step's). A step goes from the latest time below to where what is left to fall, ln N − ln θ
/// there, is fallen at the larger of two rates: the hazard there (`hazard_at`) and its mean over what is left of the
/// bracket. At the larger rate a step falls short of the root rather than past it where the hazard does not rise, so
/// that the spans mostly stand; the steps are Newton's from below where it falls -- quadratic -- and the secant's where
/// the mean is the larger; a time outside the bracket is its middle (without a far end, the panel's end), and after
/// `ROUNDS` steps every step is. Ends where ln N is within its rounding of ln θ -- `floor` -- or where the bracket has
/// closed to neighbouring times. The root is then the latest time below and what is left, within that rounding, at the
/// bracket's mean rate -- not the latest time below itself, which lies on the one side of the root: over a run's
/// thousands of jumps that side's rounding adds up --, and the kick's sums are the panel's as far as the latest time
/// below, the sliver past it within the rounding. Without a far end the spans may reach the panel's end below ln θ: the
/// panel holds no root, and its sums are the spans'. Returns the root (none in the panel), ∫h from the panel's start to
/// the latest time below, the nodes taken and the hazard's traces.
fn root(picture: &mut Picture, (still, h, bandwidth): (bool, f64, f64), (decay, part): (&[f64], &mut [f64]), (sums, weights): (&mut [f64], &mut [f64]), (a, b): (f64, f64), log: f64, mut past: Option<(f64, f64)>, guess: f64, target: f64, floor: f64) -> (Option<f64>, f64, usize, usize) {
    let _p = crate::prof::scope(crate::prof::ROOT);
    let after = |u: f64, v: f64| (v - u) * (b - a) > 0.0;
    // (time, ∫h from the panel's start) at the latest time below the root.
    let (mut below, mut nodes, mut traces) = ((a, 0.0), 0, 0);
    for round in 0.. {
        let left = log - below.1 - target;
        if !(left > floor) {
            break;
        }
        let next = if round == 0 && guess.is_finite() {
            guess
        } else if round < ROUNDS {
            let mean = past.map_or(0.0, |(time, fell)| (fell - below.1) / (time - below.0));
            let rate = linalg::larger(hazard_at(picture, still, h, below.0, decay, part), mean);
            (nodes, traces) = (nodes + 1, traces + 1);
            below.0 + left / rate
        } else {
            f64::NAN
        };
        let u = match past {
            Some((far, _)) if after(below.0, next) && after(next, far) => next,
            Some((far, _)) => 0.5 * (below.0 + far),
            None if after(below.0, next) && after(next, b) => next,
            None => b,
        };
        if !(after(below.0, u) && past.is_none_or(|(far, _)| after(u, far))) {
            break;
        }
        let mut spanned = vec![0.0; weights.len()];
        let (fell, taken_nodes) = fallen(picture, still, h, (below.0, u), bandwidth, decay, part, &mut spanned);
        (nodes, traces) = (nodes + taken_nodes, traces + 1);
        // A span that ends within ln N's rounding past the root ends at it: a step aimed at the root lands there within
        // that rounding either way, and is taken.
        if log - (below.1 + fell) > target - floor {
            below = (u, below.1 + fell);
            taken(part, &spanned, sums, weights);
            if u == b && past.is_none() && log - below.1 - target > floor {
                return (None, below.1, nodes, traces);
            }
        } else {
            past = Some((u, below.1 + fell));
        }
    }
    let root = past.map_or(below.0, |(far, fell)| below.0 + (log - below.1 - target) * (far - below.0) / (fell - below.1));
    (Some(if after(below.0, root) && past.is_none_or(|(far, _)| after(root, far)) { root } else { below.0 }), below.1, nodes, traces)
}

// ---------------------------------------------------------------------------------------------------------------------
// The step.

/// One step of τ from a trajectory's state (`Midway::attempt`).
#[derive(Clone, Debug)]
pub struct Step {
    /// The state at the step's end; the loss is the caller's.
    pub end: Trajectory,
    /// The step's path from its start: as far as the step's first jump, and from each jump on the path `after` it.
    pub segment: Segment,
    /// The states made inside the step, in the order of their times: the trajectory's after each of its jumps, and the
    /// readings at the times read, which are no part of it (adapt.rs, # The reads).
    pub made: Vec<Made>,
    /// The step's counted work (real multiply-adds), its stretches' prices added up: its own, its nodes' generators,
    /// forces and tones, its middles' making and the scattered light at the middle (`Price`); its jumps' photons are
    /// counted on the states (`jump`).
    pub work: f64,
    /// Its stretches' prices, each with its cluster, in their order: what the step did, item by item (`Price`).
    pub prices: Vec<(usize, Price)>,
    /// The momenta the step makes in turn (1/µm, lab axes; module documentation, # The step's momentum): its start κ₀,
    /// the middle the first half flow reaches κ_h, the middle kicked without its jumps κ_k and with them κ_j, its end κ₁.
    momenta: [[f64; 3]; 5],
    /// The position the first half flow reaches, r_h: where the kick is made.
    reached: [f64; 3],
}

/// What a step does to the momentum (1/µm, lab axes; module documentation, # The step's momentum): the whole, end.κ −
/// start.κ, and its one decomposition, each part a difference of the step's own momenta -- the reference's two half
/// flows, the kick at the middle (the force less the reference's, # The kick) and the jumps' kicks as they enter the
/// flow at the middle --, which add up to the whole to rounding.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Momentum {
    pub whole: [f64; 3],
    pub flow: [f64; 3],
    pub kick: [f64; 3],
    pub jumps: [f64; 3],
}

impl Step {
    /// The step's momentum change and its decomposition (`Momentum`).
    pub fn momentum(&self) -> Momentum {
        let [start, middle, kicked, jumped, end] = self.momenta;
        Momentum {
            whole: std::array::from_fn(|a| end[a] - start[a]),
            flow: std::array::from_fn(|a| (middle[a] - start[a]) + (end[a] - jumped[a])),
            kick: std::array::from_fn(|a| kicked[a] - middle[a]),
            jumps: std::array::from_fn(|a| jumped[a] - kicked[a]),
        }
    }
    /// The middle the first half flow reaches, (r_h, κ_h): the kick's place and the momentum it is made at.
    pub fn middle(&self) -> ([f64; 3], [f64; 3]) {
        (self.reached, self.momenta[1])
    }
}

/// A state made inside a step: the trajectory's after a jump -- with the jump's kick (1/µm) --, or a reading at a time
/// read with its rows as shown, read on the step's middle where its model reads them there (`Middle::readout`). The
/// internal state, the cluster, the time and the trajectory's counters are the propagation's there; the position and the
/// momentum are the caller's to set on the step's path (`Segment::after`, `Segment::place`).
#[derive(Clone, Debug)]
pub struct Made {
    pub kick: Option<[f64; 3]>,
    pub state: Trajectory,
    pub rows: Option<[f64; STATES]>,
}

/// Why a step or a jump could not be made: the reference's fixed point is not a contraction at this step (`middle`); the
/// state left the floating-point range, or a piece's quotient has no finite bandwidth (`Midway::stretch`); the norm
/// reached its threshold with no photon to emit (`jump`: the model's decay and its photons contradict each other); a
/// state scatters faster than any state of the atom can (`Domain::runaway`: a fold far outside its validity); the
/// middle state's cluster made no K of its optical sector (`Unmade`: # The optical sector's noise); or the
/// decomposition of the generator does not take the step (spectral.rs `Refusal`, # Where the decomposition does not
/// take the step). The advance takes the step at its half at
/// the first two and the last (adapt.rs, # The step), and ends at the others.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Refused {
    Reference,
    Overflow,
    Emission,
    Runaway,
    Optical(Unmade),
    Propagation(Refusal),
}

/// What the integrator asks of the model's domain at every step (SPEC §1.8; module documentation, # The domain).
pub trait Domain {
    /// Whether a state scattering at the rate `hazard` (1/µs: x̂†Γx̂, explicit excited states and folds alike) is beyond
    /// every state of the atom -- a fold far outside its validity (module documentation, # The domain).
    fn runaway(&self, hazard: f64) -> bool;
}

/// The path's momentum of the step of τ about its middle time t_m (module documentation, # The path's momentum),
/// predicted from the start state's held forms: κ_h and half the kick they give over the step -- the static form (the
/// reference's gradient taken) over τ, each tone's over the step at its phase there, τ sinc(ντ/2) Re[a e^{iν(t_m − t₁)}].
pub fn predicted(forms: &Forms, kappa_h: [f64; 3], tau: f64, t_m: f64) -> [f64; 3] {
    let mut held = forms.held.map(|f| tau * f);
    for (nu, a) in &forms.tones {
        let (w, turn) = (tau * sinc(0.5 * nu * tau), C64::cis(nu * (t_m - forms.time)));
        for c in 0..3 {
            held[c] += w * (a[c] * turn).re;
        }
    }
    std::array::from_fn(|c| kappa_h[c] - 0.5 * held[c])
}

/// A step's middle (module documentation, # The reference, # The kick): the reference's flow of a step of τ and the
/// point (r_h, κ_h) its first half reaches from the step's start, the reference's gradient there, and the model's
/// generator there at the step's middle time (`Dynamics::middle`) -- all of a step that its internal state does not
/// enter. The trajectory's step goes through it.
pub struct Midway<'a, D: Dynamics + 'a> {
    model: &'a D,
    tau: f64,
    hbar_over_mass: f64,
    flow: Flow,
    reached: ([f64; 3], [f64; 3]),
    reference: [f64; 3],
    generator: D::Middle<'a>,
}

/// The states a panel keeps until its root is known (`Midway::stretch`): the stops of a step's propagation beside its
/// quadrature's nodes -- the step's middle and its end --, and the states at the times read, which are no stops of it
/// (`looked`).
#[derive(Clone, Copy)]
enum Stop {
    Middle,
    Read,
    End,
}

/// A step's propagation as it goes (`Midway::attempt`): the time it stands at (from the middle); the trajectory's
/// state on the path there -- its internal state, cluster, threshold and counters; its position and momentum the
/// path's at its last jump --; the next of the times to read; whether the middle is behind; the sums of the quadrature's
/// weights times the three nodes' Lagrange polynomials over what is behind; the kick's part of the state so far; the
/// scattered light at the middle, once it is passed, and the optical sector's rate of the middle state with its
/// locality ε_Q where its cluster has one (# The optical sector's noise); the counted work and its stretches' prices,
/// each with its cluster (`Price`); and the states made.
struct March {
    u: f64,
    state: Trajectory,
    read: usize,
    passed: bool,
    impulse: [f64; 3],
    scattered: Option<Scattering>,
    optical: Option<([[f64; 3]; 3], f64)>,
    work: f64,
    prices: Vec<(usize, Price)>,
    made: Vec<Made>,
    /// The end state's rows as shown on the middle of the stretch that reaches the step's end in its own cluster, where
    /// times are read at the end (`Middle::readout`).
    ended: Option<[f64; STATES]>,
    /// The stretch being taken, as far as it went (`Progress`).
    progress: Progress,
}

impl<'a, D: Dynamics> Midway<'a, D> {
    /// The middle of the step of τ (negative: the reverse step) from a trajectory's position, momentum, cluster and
    /// time: the reference's fixed point (`middle`; refused where it is not one) and the model's generator there.
    pub fn new(model: &'a D, from: &Trajectory, tau: f64) -> Result<Midway<'a, D>, Refused> {
        let _p = crate::prof::scope(crate::prof::STEP_NEW);
        let half = 0.5 * tau;
        let (flow, reached) = middle(model, from.r, from.kappa, half)?;
        let hm = model.hbar_over_mass();
        let reference = flow.gradient_at(reached.0);
        let generator = model.middle(from.cluster, reached.0, from.time + half);
        Ok(Midway { model, tau, hbar_over_mass: hm, flow, reached, reference, generator })
    }

    /// The step through this middle of a state at its start -- the middle's own position, momentum, cluster and time,
    /// any internal state of the cluster --, with its jumps (module documentation, # Jumps) and the readings at the
    /// times `reads` (increasing, after the step's start and no later than its end; no part of the step, `looked`):
    /// the path's momentum predicted from the state's held forms (`predicted`); the propagation at it and on its path,
    /// the end state's held forms with its kick (# The path's momentum), cluster by
    /// cluster as the jumps land (`stretch`); the kick -- the propagation's impulse, the reference's τ∇V_ref(r_h), the
    /// scattered light's force and its diffusion's part of the Brownian path `brownian` (module documentation, # The
    /// absorbed light's momentum), and each jump's own where it is made -- and the second half flow. `scratch` is the
    /// trajectory's. A jump the model gives no photon is refused (`jump`); so is a stretch whose first state scatters
    /// beyond every state of the atom (`Domain::runaway`; module documentation, # The domain).
    pub fn attempt(&self, from: &Trajectory, brownian: &Brownian, reads: &[f64], scratch: &mut Scratch, domain: &dyn Domain) -> Result<Step, Refused> {
        let _p = crate::prof::scope(crate::prof::ATTEMPT);
        let (tau, hm, n) = (self.tau, self.hbar_over_mass, from.psi.len());
        let h = 0.5 * tau;
        let (flow, (r_h, kappa_h), reference) = (self.flow, self.reached, self.reference);
        scratch.fit(n, SUMS);
        let kappa_m = predicted(&from.forms, kappa_h, tau, from.time + h);
        let mut march = March { u: -h, state: from.clone(), read: 0, passed: false, impulse: [0.0; 3], scattered: None, optical: None, work: 0.0, prices: Vec::new(), made: Vec::new(), ended: None, progress: Progress::default() };
        // The three nodes' weights' sums over the step, its stretches' together.
        scratch.weights.fill(0.0);
        // Cluster by cluster: the start's generator, then that of every cluster a jump lands in, at the same middle.
        let mut landed: Option<D::Middle<'a>> = None;
        loop {
            // A refused stretch: the attempt's count as far as it went, on the scratch (`Scratch::refused`).
            let generator = landed.as_ref().unwrap_or(&self.generator);
            let (done, next) = self.stretch(generator, kappa_m, from.time, reads, &mut march, scratch, domain).inspect_err(|_| scratch.refused += march.work + unfinished(generator, &march.progress, scratch))?;
            if done {
                break;
            }
            landed = Some(next.unwrap_or_else(|| {
                let _p = crate::prof::scope(crate::prof::STEP_OTHER_MIDDLE);
                self.model.middle(march.state.cluster, r_h, from.time + h)
            }));
        }
        let March { state: mut end, read, impulse, scattered, optical, work, prices, mut made, ended, .. } = march;
        // The scattered light at the middle, on the state there: its force over the step; and its diffusion with the
        // optical sector's drawn covariance -- the middle state's rate over the step with what was pending (# The
        // optical sector's noise) --, their factor on the Brownian path's increment (the momentum) and on its first
        // moment about the middle (the position).
        let scattered = scattered.expect("a step passes its middle");
        let force = scattered.force;
        let (mut rate, mut size, step) = (scattered.diffusion, scattered.size, tau.abs());
        let trace = |c: &[[f64; 3]; 3]| c[0][0] + c[1][1] + c[2][2];
        let added: [[f64; 3]; 3] = optical.map_or([[0.0; 3]; 3], |(w, _)| std::array::from_fn(|a| std::array::from_fn(|b| step * (w[a][b] + w[b][a]))));
        let covariance: [[f64; 3]; 3] = std::array::from_fn(|a| std::array::from_fn(|b| from.pending[a][b] + added[a][b]));
        if covariance.iter().flatten().any(|x| *x != 0.0) {
            let (drawn, pending) = split(&covariance);
            for a in 0..3 {
                for b in 0..3 {
                    rate[a][b] += drawn[a][b] / step;
                }
            }
            size += (0..3).map(|a| drawn[a][a]).fold(0.0, linalg::larger) / step;
            end.pending = pending;
        }
        if let Some((_, epsilon)) = optical {
            end.booked[0] += epsilon * trace(&added);
            end.booked[1] += trace(&added);
        }
        end.booked[2] += step * trace(&scattered.diffusion);
        let spread = factor(&rate, size, scattered.links);
        let through = |w: &[f64; 3]| -> [f64; 3] { std::array::from_fn(|a| (0..3).map(|c| spread[a][c] * w[c]).sum()) };
        let (push, shift) = spreading(&flow, tau, through(&brownian.increment), through(&brownian.moment));
        let mut kicked: ([f64; 3], [f64; 3]) = (
            std::array::from_fn(|a| r_h[a] - hm * shift[a]),
            std::array::from_fn(|a| kappa_h[a] + impulse[a] + tau * (reference[a] + force[a]) + push[a]),
        );
        // The kick itself, for the step's path: summed apart from the middle's momentum, which would round it.
        let kick = (shift.map(|x| -hm * x), std::array::from_fn(|a| impulse[a] + tau * (reference[a] + force[a]) + push[a]));
        // The jumps' kicks, each at its time u from the middle: in the reference's flow the kick δκ cos ωu and the
        // displacement −(ħ/m) δκ sin(ωu)/ω = −(ħ/m) δκ u c₁(ω²u²) at the middle, per principal axis.
        let unjumped = kicked.1;
        for (jumped, u) in made.iter().filter_map(|m| m.kick.map(|kick| (kick, (m.state.time - from.time) - h))) {
            for (e, lambda) in flow.axes.iter().zip(&flow.curvatures) {
                let ((c0, c1, _), along) = (stumpff(hm * lambda * u * u), linalg::dot3(*e, jumped));
                for a in 0..3 {
                    kicked.0[a] -= hm * along * u * c1 * e[a];
                    kicked.1[a] += along * c0 * e[a];
                }
            }
        }
        let (r1, kappa1) = flow.advance(kicked, 0.5 * tau);
        if !(r1.iter().chain(&kappa1).all(|x| x.is_finite()) && end.psi.iter().all(|z| z.is_finite())) {
            scratch.refused += work;
            return Err(Refused::Overflow);
        }
        (end.r, end.kappa, end.time) = (r1, kappa1, from.time + tau);
        // A time read at the step's end is the end itself.
        made.extend(reads[read..].iter().map(|&time| Made { kick: None, state: Trajectory { time, ..end.clone() }, rows: ended }));
        let segment = Segment::new(from.time, tau, flow, (from.r, from.kappa), kick, spread);
        Ok(Step { end, segment, made, work, prices, momenta: [from.kappa, kappa_h, unjumped, kicked.1, kappa1], reached: r_h })
    }

    /// The propagation of B on the step's path from where the march stands, in the cluster its state is in, as far as
    /// the step's end (true) or a jump that lands in another cluster (false): the cluster's generator `generator` at
    /// the middle with the path through it at κ_m, its picture (`spectral::Picture`), and piece by piece between the
    /// jumps
    /// - the picture launched from the piece's first state (`spectral::Picture::launch`: the state is then known at
    ///   any time from there on);
    /// - the frame's state read at the stops in the path's order, panel by panel: the nodes of a quadrature that
    ///   resolves the generator's own frequencies -- Gauss–Legendre from the piece's start to the step's end at the
    ///   bandwidth of the forms on a state it propagates, to the round-off (`panels`) --, each with its weight; the
    ///   step's middle and its end, where the state is kept -- a node within the time's round-off of one of them is
    ///   it --, standing once the panel's survival is known. The times read are no stops: each is the state there from
    ///   the latest stop at or before it (`looked`), kept beside the stops and standing with them -- so the step is
    ///   the same whatever is read of it;
    /// - the survival at each panel's end, ln N at the piece's start (‖ψ‖² there) less ∫h over the sums' nodes since
    ///   (`hazard`; module documentation, # Jumps). Where it is at the threshold or below -- or where the propagation's
    ///   norm is at the threshold at a stop, from which on the panel's nodes are not taken --: the sums as they stood at
    ///   the panel's start; the root inside the panel (`root`, ln N there from the panel's start by `fallen`), the
    ///   spans below it taken into the sums -- the kick's part of the panel as far as the root, its own nodes' --, the
    ///   stops met before it; and the jump at the root on the path's own point (module documentation, # Jumps; `jump`),
    ///   from √θ times the direction there, its state the next piece's first.
    ///
    /// The stretch's part of the kick is exact in its decomposition (module documentation, # The kick): the force
    /// along the path, −∫Re tr(Ĝ(u)ρ(u))du over the stretch, Ĝ through the step's three nodes u_j = −h, 0, h and their
    /// Lagrange polynomials ℓ_j -- Σ_j Re tr(Ĝ(u_j) S_j), S_j = Σ_k w_k ℓ_j(u_k) ρ_k over the stretch's nodes, three
    /// forms of the model --; and in the stretch the step's middle lies in, about the middle state held in the frame
    /// over the whole step: its shear at r_h, each tone at its beat alone (τ sinc(Ωτ/2)), and S_j less W_j ρ_m, W_j = ∫ℓ_j
    /// over the step -- the weights' sum over every piece of the step, and of what is left of it past this stretch.
    /// Of a state that does not change that difference is its rounding, and nothing where there is one state. Where
    /// Ĥ is still along every path Ĝ is one matrix, and the stretch's part its one form with ∫ρ(u)du. The middle's
    /// stretch reads the scattered light there and the optical sector's rate, on the middle state.
    #[allow(clippy::too_many_arguments)]
    fn stretch(&self, generator: &D::Middle<'a>, kappa_m: [f64; 3], start: f64, reads: &[f64], march: &mut March, scratch: &mut Scratch, domain: &dyn Domain) -> Result<(bool, Option<D::Middle<'a>>), Refused> {
        let _p = crate::prof::scope(crate::prof::STRETCH);
        let (tau, hm) = (self.tau, self.hbar_over_mass);
        let h = 0.5 * tau;
        let velocity = kappa_m.map(|k| hm * k);
        let along = Along::new(generator, kappa_m, velocity);
        let (n, still, cluster) = (along.n, generator.still(), march.state.cluster);
        march.progress = Progress { states: n, still, ..Progress::default() };
        let mut launched = Some(along.turned(&march.state.psi, march.u, true));
        // The nodes' sums (`Sums`): one where Ĥ is still, the three nodes' and the step's tones' on a beating path; the
        // decay laid alike.
        let scale = pictured(&along, scratch, tau, launched.as_ref().map(|x| &x[..n]))?;
        let Carved { picture, noise, held, sum, tones, sums, saved, part, decay, met, weights, kept } = scratch.carve();
        // What the picture's preparation counted (`Price::made`); what it counts on from here is its states'.
        let (bandwidth, size, prepared) = (picture.bandwidth, sums.len(), picture.work);
        sums.fill(0.0);
        // The tones' weights' sums, this stretch's; and where it begins.
        weights[SUMS..].fill(0.0);
        let begun = march.u;
        let ahead = |u: f64| (u + h) / tau;
        let mut middle = None;
        let mut x = [C64::ZERO; STATES];
        // A state of the frame at a time u in the lab at the survival ln N: the direction at the norm √N.
        let at = |x: &[C64], u: f64, log: f64| -> Internal {
            let scale = (0.5 * (log - norm_sqr(x).ln())).exp();
            along.turned(x, u, false)[..n].iter().map(|z| *z * scale).collect()
        };
        // The middle of the cluster the last jump landed the trajectory in, where it is another (`Places`).
        let mut next: Option<D::Middle<'a>> = None;
        let done = loop {
            // A piece: from the state the march stands at, the sums and the weights as they are, the survival from ‖ψ‖².
            let from = march.u;
            // The stretch's first piece is launched with the state its picture was prepared on, the next ones with a jump's.
            let launched = launched.take().unwrap_or_else(|| along.turned(&march.state.psi, from, true));
            {
                let _p = crate::prof::scope(crate::prof::LAUNCH);
                picture.launch(from, &launched[..n]);
            }
            // The quotients' bandwidth over this piece, on the states its state can hold (`spectral::Picture::launch`,
            // `quotient`): the forms' with the norm's ringing.
            let bandwidth = picture.quotient;
            // A quotient of no finite bandwidth lays no rule (`fewest`): the step is refused, as a state that left the
            // numbers is (`Refused::Overflow`).
            if !bandwidth.is_finite() {
                return Err(Refused::Overflow);
            }
            // The runaway's guard (module documentation, # The domain): the stretch's first state's hazard, x̂†Γx̂ of the
            // launched state's direction at the decay laid for the step (`hazard_at`'s, on the state itself: nothing of
            // the picture is moved or counted).
            if march.progress.traces == 0 {
                part.fill(0.0);
                Sums { still, h, tones: &picture.beats }.gather(from, 1.0, &launched[..n], None, part);
                if domain.runaway(hazard(decay, part, n)) {
                    return Err(Refused::Runaway);
                }
            }
            let (target, first, before) = (march.state.threshold.ln(), norm_sqr(&march.state.psi).ln(), hazard(decay, sums, n));
            // The norm does not rise (`Refusal::Growth`): the flow is a contraction, so at the propagation's stops past the
            // launch -- the piece's middle and end, which its hops reach -- a norm above the launch's beyond what the
            // propagation's own remainder E and the state's rounding allow, (1 + E)² − 1 (a state off by E of its norm),
            // is the propagation's leaving it. E is made only where a stop rises past the rounding.
            let (launch, rounded) = (norm_sqr(&launched[..n]), picture.rounding());
            let mut rise = 0.0_f64;
            for u in [0.5 * (from + h), h] {
                picture.at(u, &mut x[..n]);
                rise = linalg::larger(rise, norm_sqr(&x[..n]) / launch - 1.0);
            }
            if !(rise <= rounded) {
                let (estimate, work) = picture.remainder();
                picture.work += work;
                if !(rise <= estimate * (2.0 + estimate) + rounded) {
                    return Err(Refused::Propagation(Refusal::Growth));
                }
            }
            march.progress.traces += 1;
            // The panel's start and ln N there; the stops met, the middle and the times read as far as they stand.
            let (mut a, mut log, mut passed, mut read, mut crossed) = (from, first, march.passed, march.read, None);
            // The panels: as many as leave the fewest nodes, on whether the survival ends in the piece (`count`).
            let laid = count(h - from, bandwidth, FORMS, 2, || {
                picture.at(h, &mut x[..n]);
                norm_sqr(&x[..n]) <= march.state.threshold
            });
            for (b, nodes) in panels(from, h, laid) {
                let mut nodes = nodes.peekable();
                met.clear();
                saved.copy_from_slice(sums);
                kept.copy_from_slice(weights);
                // The latest time taken and, once the propagation's norm is at the threshold, that time: the crossing
                // the norm predicts lies past it (the norm is the survival to the propagation's accuracy, and costs n a
                // stop). From then on the panel's nodes are not taken -- its kick's part is the root's spans --, its kept
                // stops still are.
                let (mut last, mut predicted) = (a, None);
                loop {
                    // The next stop of the panel: a node, the next kept stop within it -- the middle, then the end -- or
                    // both at once. A time read is no stop (`looked`).
                    let fixed = if passed { (h, Stop::End) } else { (0.0, Stop::Middle) };
                    let kept = Some(fixed).filter(|&(u, _)| ahead(u) <= ahead(b));
                    let node = if predicted.is_none() { nodes.peek().copied() } else { None };
                    let (u, weight, keep) = match (node, kept) {
                        (Some((u, w)), Some((stop, kind))) if (u - stop).abs() <= f64::EPSILON * h.abs() => {
                            nodes.next();
                            (stop, Some(w), Some(kind))
                        }
                        (Some((u, w)), Some((stop, _))) if ahead(u) < ahead(stop) => {
                            nodes.next();
                            (u, Some(w), None)
                        }
                        (_, Some((stop, kind))) => (stop, None, Some(kind)),
                        (Some((u, w)), None) => {
                            nodes.next();
                            (u, Some(w), None)
                        }
                        (None, None) => break,
                    };
                    {
                        let _p = crate::prof::scope(crate::prof::AT_NODES);
                        picture.at(u, &mut x[..n]);
                    }
                    read = looked(picture, reads, read, (start, h), u, met);
                    if predicted.is_none() && norm_sqr(&x[..n]) <= march.state.threshold {
                        predicted = Some(last);
                    }
                    if let (Some(w), None) = (weight, predicted) {
                        march.progress.counted += 1;
                        let _p = crate::prof::scope(crate::prof::GATHER);
                        Sums { still, h, tones: &picture.beats }.gather(u, w, &x[..n], Some(&mut *weights), sums);
                    }
                    last = u;
                    if let Some(kind) = keep {
                        met.push((kind, u, x));
                        match kind {
                            Stop::End => break,
                            _ => passed = true,
                        }
                    }
                }
                read = looked(picture, reads, read, (start, h), b, met);
                // The panel's end: where the norm predicted a crossing, the root from the panel's start (`root`); else ln N
                // there by the sums, and where it is at the threshold the root inside the panel. The kick's part of the
                // panel as far as the root comes with it; a predicted crossing the survival does not reach in the panel
                // leaves the panel's whole part, ln N at its end (`end`).
                let mut end = f64::NAN;
                let search = match predicted {
                    Some(guess) => Some((None, guess)),
                    None => {
                        end = first - (hazard(decay, sums, n) - before);
                        march.progress.traces += 1;
                        (end <= target).then_some((Some((b, log - end)), f64::NAN))
                    }
                };
                let root = search.and_then(|(past, guess)| {
                    let floor = rounding(2 * march.progress.counted + size + n + 10) * 1.25 * scale * tau.abs() + 2.0 * UNIT_ROUNDOFF * target.abs();
                    sums.copy_from_slice(saved);
                    weights.copy_from_slice(kept);
                    let (crossed, fell, nodes, spans) = root(picture, (still, h, bandwidth), (decay, part), (sums, &mut *weights), (a, b), log, past, guess, target, floor);
                    (march.progress.counted, march.progress.traces) = (march.progress.counted + nodes, march.progress.traces + spans);
                    match crossed {
                        Some(crossed) => picture.at(crossed, &mut x[..n]),
                        None => end = log - fell,
                    }
                    crossed
                });
                // The states kept stand as far as the root: the middle state; a time read, its state as the propagation
                // passed it (`looked`), in the lab. A reading is read as ρ/tr ρ alone -- its rows, its place, a sample of
                // the model's error (`Dynamics::readout`) --, so its norm is
                // the propagation's there and nothing of the survival is integrated for it: the loss and the jumps are
                // the trajectory's own states' (adapt.rs, # The reads).
                for &(kind, u, state) in met.iter().filter(|m| root.is_none_or(|r| ahead(m.1) < ahead(r))) {
                    match kind {
                        Stop::Middle => (middle, march.passed) = (Some(state), true),
                        Stop::Read => {
                            let state = Trajectory { psi: along.turned(&state[..n], u, false)[..n].iter().copied().collect(), time: reads[march.read], ..march.state.clone() };
                            let rows = along.middle.readout(&along.path, u, &state.psi);
                            march.made.push(Made { kick: None, state, rows });
                            march.read += 1;
                        }
                        Stop::End => {}
                    }
                }
                if root.is_some() {
                    crossed = root;
                    break;
                }
                (a, log) = (b, end);
            }
            let Some(crossed) = crossed else {
                march.state.psi = at(&x[..n], h, log);
                march.u = h;
                break true;
            };
            // The jump, on the path's own point, from the state at the threshold, on the middles the step made there.
            let state = &mut march.state;
            state.psi = at(&x[..n], crossed, target);
            (state.r, state.kappa, state.time) = (std::array::from_fn(|a| self.reached.0[a] + velocity[a] * crossed), kappa_m, start + (crossed + h));
            let mut places = Places { model: self.model, point: (self.reached.0, start + h), path: (kappa_m, velocity), s: crossed, own: (cluster, generator, &along.path), other: None };
            {
                let _p = crate::prof::scope(crate::prof::JUMP);
                jump(self.model, state, Some(&mut places))?;
            }
            next = places.other.filter(|other| other.0 == state.cluster).map(|other| other.1);
            let kick = std::array::from_fn(|a| state.kappa[a] - kappa_m[a]);
            march.made.push(Made { kick: Some(kick), state: state.clone(), rows: None });
            march.u = crossed;
            if !(ahead(crossed) < 1.0) {
                break true;
            }
            if march.state.cluster != cluster {
                break false;
            }
        };
        // The end state's held forms (module documentation, # The path's momentum): where the stretch reaches the step's end
        // in its own cluster, its density at the middle (the frame's there is the lab's), with the kick's forms, less the
        // step's reference gradient; where a jump at the end left it in another, that cluster's at the jump's point.
        let ended = done && march.state.cluster == cluster;
        if ended && march.read < reads.len() {
            march.ended = along.middle.readout(&along.path, h, &march.state.psi);
        }
        let end: Option<Vec<C64>> = ended.then(|| {
            let mut rho = vec![C64::ZERO; n * n];
            density(&march.state.psi, &mut rho);
            rho
        });
        if done && !ended {
            let state = &march.state;
            let forms = self.model.forms(state.cluster, &state.point(), &state.psi);
            march.state.forms = forms;
        }
        let t_m = start + h;
        // The stretch's part of the kick; in the middle's stretch about the middle state, with the scattered light and the
        // optical sector's rate there.
        let part = if still {
            let sum = &mut sum[..n * n];
            whole(sums, 0, n, sum);
            if let Some(rho) = &end {
                let fixed = along.middle.path(kappa_m, [0.0; 3]);
                march.state.forms = Forms { time: t_m, ..forms_at(along.middle, &fixed, rho) }.less(self.reference);
            }
            along.middle.force(&along.path, 0.0, sum).map(|f| -f)
        } else {
            let mut out = [0.0; 3];
            let size = n * n;
            if let Some(middle) = &middle {
                if !done {
                    for (u, w) in quadrature(march.u, h, 0.0, FORMS, 2) {
                        let s = u / h;
                        for (sum, l) in weights.iter_mut().zip([0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)]) {
                            *sum += w * l;
                        }
                    }
                }
                density(&middle[..n], held);
            }
            // A tone's sums at its beat at each of the three nodes (`Sums`), less the middle state's at their weights: its
            // σ_j = Σ_k w_k ℓ_j(u_k) e^{iνu_k}(ρ_k − ρ_m) -- the weights' sums with what of the step lies outside this
            // stretch, ∫ℓ_j(u)e^{iνu}du over [−h, its start] and past its end by the kick's rule at the tone's beat.
            for ((m, node_sums), tone) in tones.chunks_exact_mut(TONE * size).enumerate().zip(&picture.beats) {
                let mut outside = [0.0; TONE];
                if middle.is_some() {
                    for (a, b) in [(-h, begun)].into_iter().chain((!done).then_some((march.u, h))) {
                        for (u, w) in quadrature(a, b, tone.beat.abs(), FORMS, 2) {
                            let (s, (sin, cos)) = (u / h, (tone.beat * u).sin_cos());
                            let lagrange = [0.5 * s * (s - 1.0), 1.0 - s * s, 0.5 * s * (s + 1.0)];
                            for (k, part) in outside.iter_mut().enumerate() {
                                *part += w * lagrange[k / 2] * if k % 2 == 0 { cos } else { sin };
                            }
                        }
                    }
                }
                for (k, node_sum) in node_sums.chunks_exact_mut(size).enumerate() {
                    let slot = SUMS + TONE * m + k;
                    whole(sums, slot, n, node_sum);
                    if middle.is_some() {
                        for (q, rho) in node_sum.iter_mut().zip(held.iter()) {
                            *q -= *rho * (weights[slot] + outside[k]);
                        }
                    }
                }
            }
            // The middle node's form and the shear of the middle state -- each tone at its beat alone, τ sinc(Ωτ/2) -- in
            // one pass of the model's forms (`Middle::centre`); the nodes' forms hold the exact tones' at the nodes, and
            // each tone's sums at its beat take them out node by node (`Middle::tone_force`).
            let shear = |omega: f64| tau * sinc(0.5 * omega * tau);
            for (j, (at, sum)) in [-h, 0.0, h].into_iter().zip(sum.chunks_exact_mut(size)).enumerate() {
                whole(sums, j, n, sum);
                if middle.is_some() {
                    for (q, rho) in sum.iter_mut().zip(held.iter()) {
                        *q -= *rho * weights[j];
                    }
                }
                for (node_sums, tone) in tones.chunks_exact_mut(TONE * size).zip(&picture.beats) {
                    let (sin, cos) = (tone.beat * at).sin_cos();
                    let (re, im) = node_sums[2 * j * size..2 * (j + 1) * size].split_at_mut(size);
                    for ((re, im), q) in re.iter_mut().zip(im.iter_mut()).zip(sum.iter()) {
                        *re -= *q * cos;
                        *im -= *q * sin;
                    }
                }
            }
            let (s0, rest) = sum.split_at(size);
            let (s1, s2) = rest.split_at(size);
            let fixed = end.as_ref().map(|_| along.middle.path(kappa_m, [0.0; 3]));
            let _pn = crate::prof::scope(crate::prof::NODES_CALL);
            let (forms, forms_end) = along.middle.nodes(&along.path, h, [s0, s1, &s2[..size]], middle.is_some().then_some((&held[..], &shear as &dyn Fn(f64) -> f64)), end.as_deref().zip(fixed.as_ref()));
            for form in forms {
                for a in 0..3 {
                    out[a] -= form[a];
                }
            }
            if let Some(forms) = forms_end {
                march.state.forms = Forms { time: t_m, ..forms }.less(self.reference);
            }
            drop(_pn);
            for (node_sums, tone) in tones.chunks_exact(TONE * size).zip(&picture.beats) {
                let at: Vec<(f64, &[C64], &[C64])> = node_sums.chunks_exact(2 * size).zip([-h, 0.0, h]).map(|(pair, s)| (s, &pair[..size], &pair[size..])).collect();
                let form = along.middle.tone_force(&along.path, tone.candidate, &at);
                for a in 0..3 {
                    out[a] -= form[a];
                }
            }
            out
        };
        for a in 0..3 {
            march.impulse[a] += part[a];
        }
        let (mut scattering, mut sector) = (0.0, None);
        if let Some(middle) = &middle {
            let _p = crate::prof::scope(crate::prof::MID_STEP_MIDDLE);
            let (scattered, made) = along.middle.scattering(&along.path, &middle[..n]);
            (march.scattered, scattering, march.progress.scattering) = (Some(scattered), made, made);
            if let Some((epsilon, ground, made)) = optical(&along, picture, still, h, noise)? {
                let k = noise.sector.operators().expect("a sector made");
                march.optical = Some((within(k, &noise.gradient, &middle[..n]), epsilon));
                sector = Some((ground, made));
            }
        }
        // The stretch's count, item by item as it was done (`Price`), the step's own and its price at once (its tones':
        // `toned`).
        let price = Price {
            states: n,
            still,
            tau,
            bandwidth,
            nodes: march.progress.counted,
            traces: march.progress.traces,
            made: prepared,
            stops: picture.work - prepared,
            sector,
            tones: picture.beats.len(),
            toned: toned(generator, picture, n),
            centred: middle.is_some(),
            ended,
            middle: generator.made(),
            scattering,
            costs: generator.work(),
        };
        march.work += price.work(price.costs);
        march.prices.push((cluster, price));
        Ok((done, next.filter(|_| !done)))
    }
}

/// The Stumpff function c₃(x) = (1 − c₁(x))/x = Σ_j (−x)^j/(3 + 2j)!, entire in x, by its series to round-off: the
/// quotient cancels where x is small, and the series' terms fall from the first while |x| is below its second term's
/// 20 (a step's x = ω²τ² is below π² on a trapping axis, SPEC §2.2; on a hyperbolic one every term is positive).
fn stumpff3(x: f64) -> f64 {
    let (mut term, mut sum, mut j): (f64, f64, f64) = (1.0 / 6.0, 0.0, 0.0);
    while sum + term != sum {
        // A term that is not a number is the sum's (the caller's step is refused, its end no number), not a loop's end.
        if term.is_nan() {
            return term;
        }
        sum += term;
        j += 1.0;
        term *= -x / ((2.0 * j + 2.0) * (2.0 * j + 3.0));
    }
    sum
}

/// A diffusion's part of a step in the reference's flow (module documentation, # The absorbed light's momentum): from
/// Λ ΔW and Λ ΔZ (lab axes) the kick and the displacement over ħ/m at the step's middle that end the step as the force
/// Λ dW/ds spread over it does -- per principal axis of the flow, x = (ħ/m)λτ², the increment at the variance
/// (τ/2)(1 + c₁(x)) of ∫cos(ωu)dW, by √((1 + c₁)/2), and the first moment at the variance (τ³/2) c₃(x) of ∫(sin ωu/ω)dW,
/// by √(6 c₃). Both are 1 in free flight, and hold on a hyperbolic axis as they stand (c₁ and c₃ entire in x).
fn spreading(flow: &Flow, tau: f64, increment: [f64; 3], moment: [f64; 3]) -> ([f64; 3], [f64; 3]) {
    let (mut push, mut shift) = ([0.0; 3], [0.0; 3]);
    for (e, lambda) in flow.axes.iter().zip(&flow.curvatures) {
        let x = flow.hbar_over_mass * lambda * tau * tau;
        let (kick, lever) = ((0.5 * (1.0 + stumpff(x).1)).sqrt() * linalg::dot3(*e, increment), (6.0 * stumpff3(x)).sqrt() * linalg::dot3(*e, moment));
        for a in 0..3 {
            push[a] += kick * e[a];
            shift[a] += lever * e[a];
        }
    }
    (push, shift)
}

/// A step's part of its trajectory's Brownian path W, three independent standard ones (module documentation, # The
/// absorbed light's momentum): its increment ΔW = ∫dW over the step (variance |τ| an axis, √µs) and its first moment
/// about the step's middle ΔZ = ∫(s − τ/2)dW (variance |τ|³/12; uncorrelated with ΔW, ∫(s − τ/2)ds = 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Brownian {
    pub increment: [f64; 3],
    pub moment: [f64; 3],
}

impl Brownian {
    /// A path that does not move.
    pub const STILL: Brownian = Brownian { increment: [0.0; 3], moment: [0.0; 3] };

    /// The part over a stretch of the length τ of the path two vectors of standard normal deviates make: ΔW = √|τ| g₀
    /// and ΔZ = √(|τ|³/12) g₁.
    pub fn over(tau: f64, deviates: &[[f64; 3]; 2]) -> Brownian {
        let (t, cube) = (tau.abs().sqrt(), (tau.abs().powi(3) / 12.0).sqrt());
        Brownian { increment: deviates[0].map(|g| t * g), moment: deviates[1].map(|g| cube * g) }
    }

    /// The path's parts over the two pieces of a stretch of the length h cut at s from its start, given its part over
    /// the stretch: the Brownian bridge between them, from three further vectors of deviates. With r = h − s, B the
    /// bridge's value at the cut (ΔW₁ = (s/h)ΔW + B, of variance s r/h) and ζ_i the pieces' own first moments about
    /// their own middles (variances s³/12 and r³/12), ΔZ = ζ₁ + ζ₂ − (h/2)B: three independent normals whose variances
    /// add up to h³/12. Given their sum, each is its variance's share of ΔZ and what a free draw y_i of its own
    /// variance leaves, y_i − (v_i/V)Σy. So ΔW₁ + ΔW₂ = ΔW and the pieces join to ΔZ exactly (`joined`), and the
    /// pieces' parts have the law of `over` at their lengths, independent of each other. At the middle: ΔZ = ΔZ₁ + ΔZ₂ +
    /// (h/4)(ΔW₂ − ΔW₁).
    pub fn split(&self, length: f64, cut: f64, deviates: &[[f64; 3]; 3]) -> [Brownian; 2] {
        let (h, s) = (length.abs(), cut.abs());
        let r = h - s;
        let variances = [s.powi(3) / 12.0, r.powi(3) / 12.0, 0.25 * h * s * r];
        let whole = h.powi(3) / 12.0;
        let mut out = [Brownian::STILL; 2];
        for a in 0..3 {
            let free: [f64; 3] = std::array::from_fn(|i| variances[i].sqrt() * deviates[i][a]);
            let rest = self.moment[a] - (free[0] + free[1] + free[2]);
            let parts: [f64; 3] = std::array::from_fn(|i| free[i] + variances[i] / whole * rest);
            out[0].increment[a] = s / h * self.increment[a] - 2.0 / h * parts[2];
            out[1].increment[a] = self.increment[a] - out[0].increment[a];
            out[0].moment[a] = parts[0];
            out[1].moment[a] = parts[1];
        }
        out
    }

    /// The path's part over a stretch made of this part, over the length s, and the next one, over the length r:
    /// ΔW = ΔW₁ + ΔW₂, and the first moment about the whole's middle from the pieces' about their own, ΔZ = ΔZ₁ + ΔZ₂ −
    /// (r/2)ΔW₁ + (s/2)ΔW₂.
    pub fn joined(&self, length: f64, next: &Brownian, further: f64) -> Brownian {
        let (s, r) = (length.abs(), further.abs());
        Brownian {
            increment: std::array::from_fn(|a| self.increment[a] + next.increment[a]),
            moment: std::array::from_fn(|a| self.moment[a] + next.moment[a] - 0.5 * r * self.increment[a] + 0.5 * s * next.increment[a]),
        }
    }
}

/// The counted work of a stretch of a step (`Midway::stretch`; real multiply-adds) item by item as the stretch did it --
/// the one count a step adds up (`Step::work`, `Step::prices`) and the one a model's step is priced by (`priced`; solve.rs
/// `Formula`): a step's price is the work its run does, counted where it is done, not a model of it. Apart from the
/// model's counts at a node (`Middle::work`), which `work` takes -- a kept model's are its own --: the generator's
/// picture along the path, its preparation (`made`: its decomposition, its slopes, its tones and their products, its
/// following, as `pictured` made them) and the states it made after (`stops`: the launches -- the stretch's first and
/// one after each of its jumps --, the states at the kick's nodes, at the middle and at the end, and those its jumps'
/// roots read: `spectral::Picture::launch`, `at`); the kick's nodes and the tones their states were
/// summed with (`nodes`, `tones`); the survival's traces (`traces`); where the stretch holds the step's middle
/// (`centred`) the middle's form there and the optical sector's make (`sector`: its ground states and the make's count);
/// the end state's held forms on the stretch that reaches the step's end (`ended`); and the model's own: the middle's
/// making (`Middle::made`) and the light scattered at the middle
/// (`Middle::scattering`). A stretch after a jump to another cluster is that cluster's (`Step::prices`). (Priced without
/// taking the step instead -- the picture prepared on the middle at the trap's centre, its stops at the quadrature's
/// nodes, no jump --, PGC3D's steps of 0.95, 1.9 and 3.8 µs were priced at 0.81, 1.52 and 0.38 times what its run's
/// steps counted, and the middle with its scattering at a constant five to nine times their cost at 0.95 µs: the
/// rungs' steps were weighed against one another wrong by a factor of four.)
///
/// A count is the operations done, deterministic -- a request's decisions the same on any machine and under any load,
/// and a run compared with another to the bit --, not their time: it holds no cache's effect. Where a long step's picture
/// leaves the model's numbers out of the caches the step takes longer than its count says (PGC3D at 3.8 µs: a jump's
/// photon and the middle 30 and 15 times their time at 0.95 µs, their counts the same), and a choice made by the counts
/// leans there toward the longer step. Timed instead, the rung a request takes would change with the machine and its
/// load.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Price {
    pub states: usize,
    pub still: bool,
    pub tau: f64,
    pub bandwidth: f64,
    pub nodes: usize,
    pub traces: usize,
    /// The picture's preparation, and its states after it (complex multiply-adds).
    pub made: f64,
    pub stops: f64,
    pub sector: Option<(usize, f64)>,
    /// The exact tones (spectral.rs, # Which beats are tones), and their envelopes' and forms' counted work with their
    /// separation from the generator at the nodes (`Middle::tone_work`).
    pub tones: usize,
    pub toned: [f64; 2],
    /// Whether the stretch holds the step's middle (its form and its scattering there), and whether it reaches the step's
    /// end in its cluster (the end state's held forms).
    pub centred: bool,
    pub ended: bool,
    /// The middle's making and the scattered light's (real multiply-adds); the model's counts at a node it was counted with
    /// (`Middle::work`).
    pub middle: f64,
    pub scattering: f64,
    pub costs: [f64; 3],
}

impl Price {
    /// The stretch's counted work with a model's counts at a node `costs` (`Middle::work`): what the stretch adds to
    /// `Step::work`, its integrator's part (`stepped`) and the model's own there (`modelled`).
    pub fn work(&self, costs: [f64; 3]) -> f64 {
        self.stepped(costs) + self.modelled()
    }

    /// The integrator's part with the model's calls at the nodes: the propagation, the kick's sums and forms, the
    /// survival's hazard, the optical sector (`optical_work`: its make, the gradient's matrices and one `within`'s
    /// forms), and on the stretch that reaches the step's end the end state's held forms.
    pub fn stepped(&self, costs: [f64; 3]) -> f64 {
        let (n, still) = (self.states, self.still);
        let optical = self.sector.map_or(0.0, |(_, make)| optical_work(n, make));
        let end = if self.ended { count::end(n, costs, self.toned[1]) } else { 0.0 };
        count::propagation(costs, still, self.made + self.stops, self.toned[0]) + count::impulse(n, costs, self.nodes, still, self.centred, self.tones, self.toned[1]) + count::hazard(n, self.traces, still, self.tones) + optical + end
    }

    /// The model's own there: the middle's making and the light scattered at the middle.
    pub fn modelled(&self) -> f64 {
        self.middle + self.scattering
    }

    /// `work`'s parts (a measurement): the propagation, the kick, the survival's hazard, the optical sector, the end
    /// state's held forms, the model's own.
    #[cfg(test)]
    pub fn parts(&self, costs: [f64; 3]) -> [f64; 6] {
        let (n, still) = (self.states, self.still);
        [
            count::propagation(costs, still, self.made + self.stops, self.toned[0]),
            count::impulse(n, costs, self.nodes, still, self.centred, self.tones, self.toned[1]),
            count::hazard(n, self.traces, still, self.tones),
            self.sector.map_or(0.0, |(_, make)| optical_work(n, make)),
            if self.ended { count::end(n, costs, self.toned[1]) } else { 0.0 },
            self.modelled(),
        ]
    }
}

/// The step of τ from a trajectory's state as its run takes it, for its count: its stretches' prices, each with its
/// cluster (`Step::prices`) -- a model's step priced where its run's states are (solve.rs `Formula::prices`). On the
/// route a trajectory's steps take once the decomposition's is counted (`spectral::Picture::prepare`: a step's route is
/// chosen against the counts its scratch carries): the step is taken twice on one scratch, the second counted. Its
/// jumps are the state's own -- its threshold, its streams --, the Brownian path none (no work is of it), no time
/// read. Refused as the run's step would be (`Midway::attempt`).
pub fn priced<D: Dynamics>(model: &D, from: &Trajectory, tau: f64, domain: &dyn Domain) -> Result<Vec<(usize, Price)>, Refused> {
    let (through, mut scratch) = (Midway::new(model, from, tau)?, Scratch::default());
    through.attempt(from, &Brownian::STILL, &[], &mut scratch, domain)?;
    Ok(through.attempt(from, &Brownian::STILL, &[], &mut scratch, domain)?.prices)
}

// ---------------------------------------------------------------------------------------------------------------------
// The jump.

/// A symmetric covariance's positive and negative parts (module documentation, # The optical sector's noise): C = V Λ Vᵀ
/// by its eigenvectors, the part of the positive eigenvalues -- drawn -- and of the others -- kept pending --, which add up
/// to C; nothing of it is clipped.
fn split(c: &[[f64; 3]; 3]) -> ([[f64; 3]; 3], [[f64; 3]; 3]) {
    let (values, vectors) = linalg::symmetric3_eigen(c);
    let (mut drawn, mut pending) = ([[0.0; 3]; 3], [[0.0; 3]; 3]);
    for (lambda, v) in values.iter().zip(&vectors) {
        // An eigenvalue that is not a number is drawn (the step's factor carries it to the momentum), not kept pending.
        let part = if !(*lambda <= 0.0) { &mut drawn } else { &mut pending };
        for a in 0..3 {
            for b in 0..3 {
                part[a][b] += lambda * v[a] * v[b];
            }
        }
    }
    (drawn, pending)
}

/// A factor Λ of a diffusion's second-moment rate C = 2D, Λ Λᵀ = C (positive semidefinite: the second moment's rate less
/// what the jumps carry of it, Cauchy–Schwarz; model.rs `Scattering::of_moments`): its Cholesky factor's columns, so that Λ on a Brownian
/// increment of unit rate has the covariance C τ.
///
/// In a null direction of C -- a plane wave's absorption leaves nothing beside its jumps' k -- a pivot is the rounding
/// of its differences, of either sign, and its root would make a kick of √u of the recoil out of the rounding u. So the
/// factor is taken with complete pivoting, the largest remaining pivot first, and ends at a pivot within its rounding.
/// C's entries err by at most γ_N S, S = `scale` the largest sum of a diagonal entry's terms' moduli and N = `links` the
/// operations behind an entry (`Scattering`), so ‖ΔC‖₂ ≤ 3γ_N S; and after r columns a pivot of the rest is moved by at
/// most ‖ΔC‖₂(1 + ‖W‖₂²), W the eliminated columns' multipliers on the rest, which the pivoting bounds, ‖W‖₂² ≤
/// (3 − r)(4^r − 1)/3 (Higham, Accuracy and Stability of Numerical Algorithms, Lemma 10.13, n = 3). A largest pivot
/// within that is no direction of C, nor are the rest.
fn factor(rate: &[[f64; 3]; 3], scale: f64, links: usize) -> [[f64; 3]; 3] {
    let mut c = *rate;
    let (mut order, mut l) = ([0, 1, 2], [[0.0; 3]; 3]);
    for r in 0..3 {
        let largest = (r..3).max_by(|&x, &y| c[order[x]][order[x]].total_cmp(&c[order[y]][order[y]])).expect("a remaining axis");
        order.swap(r, largest);
        let p = order[r];
        let moved = 3.0 * rounding(links) * scale * (1.0 + ((3 - r) * (4_usize.pow(r as u32) - 1)) as f64 / 3.0);
        // A pivot or a scale that is not a number is no direction to drop: the factor is not a number, and so is the kick.
        if c[p][p].is_nan() || moved.is_nan() {
            return [[f64::NAN; 3]; 3];
        }
        if !(c[p][p] > moved) {
            break;
        }
        let root = c[p][p].sqrt();
        l[p][r] = root;
        for &i in &order[r + 1..] {
            l[i][r] = c[i][p] / root;
        }
        for &i in &order[r + 1..] {
            for &k in &order[r + 1..] {
                c[i][k] -= l[i][r] * l[k][r];
            }
        }
    }
    l
}

/// The middles a jump inside a step is made on (`jump`; module documentation, # Jumps): the step's middle point and
/// time, the path's momentum and velocity and the jump's time s from the middle; the emitting cluster's middle with its
/// path, and the middle of a cluster a jump lands in where that is another, made there -- the one the step goes on in
/// after the jump (`Midway::stretch`).
pub struct Places<'p, 'm, D: Dynamics + 'm> {
    model: &'m D,
    point: ([f64; 3], f64),
    path: ([f64; 3], [f64; 3]),
    s: f64,
    own: (usize, &'p D::Middle<'m>, &'p <D::Middle<'m> as Middle>::Path),
    other: Option<(usize, D::Middle<'m>, <D::Middle<'m> as Middle>::Path)>,
}

impl<'m, D: Dynamics + 'm> Places<'_, 'm, D> {
    /// A cluster's middle with its path: the emitting cluster's, or another's, made where it is first asked for.
    fn of(&mut self, cluster: usize) -> (&D::Middle<'m>, &<D::Middle<'m> as Middle>::Path) {
        if cluster == self.own.0 {
            return (self.own.1, self.own.2);
        }
        if self.other.as_ref().is_none_or(|other| other.0 != cluster) {
            let middle = self.model.middle(cluster, self.point.0, self.point.1);
            let path = middle.path(self.path.0, self.path.1);
            self.other = Some((cluster, middle, path));
        }
        let (_, middle, path) = self.other.as_ref().expect("made");
        (middle, path)
    }
}

/// The jump of a trajectory at its state (SPEC §1.6): from `Stream::jump(seed, index, jumps + 1)` the next threshold, the
/// class ∝ Σ_a‖u_a‖² among the rates the model makes of the state's classes, whose photon alone the model then makes
/// (`Dynamics::emit`), the polarization across the direction by ‖Σ_a ε_a u_a‖², and the direction from the dipole pattern
/// tr G − n·Re(G)·n (at most tr G, as Re G ≥ 0) by exact rejection from the uniform sphere; then ψ ← f/‖f‖ on the landing
/// cluster and κ += −k_φ n + s b, the emitted photon's recoil and the class's conditional mean of the absorbed momentum
/// as far as one photon's goes (`Emitted::kick`; module documentation, # The absorbed light's momentum) -- at most
/// k_φ + k_c, whatever the light does at the point. The point is the trajectory's as it is given: a step makes its
/// jumps on its path's points (module documentation, # Jumps), on the middles it made there (`Places`; at a
/// trajectory's start, none: the model at the point), and the loss after a jump is its caller's to test, on the
/// trajectory's own place (adapt.rs). A norm that reached its threshold with no photon to emit -- every class's
/// weight zero, or not a number, or its amplitudes no finite numbers -- is refused (`Refused::Emission`) with the
/// trajectory as it was: the model's decay and its photons contradict each other, and no jump is made of that. So is a
/// landing that does not exist.
///
/// The landing (`Dynamics::landed`, `land`): the trajectory goes on in the state the bare state f/‖f‖ lands in, with that
/// state's norm, against the threshold its landing hands it (`Landed`; the plain landing, RUSTPLAN §2.2 D5: the state
/// itself and the threshold θ drawn here), with the kick the landing books. A landed norm at or below its threshold
/// already is that jump at once -- its class, polarization and direction drawn anew --, the next threshold drawn anew,
/// and it lands again. The waiting time after a plain landing is P(T > t) = min(1, n e^{−γt}) for the landing's norm n
/// and the state's rate γ.
pub fn jump<'m, D: Dynamics>(model: &'m D, trajectory: &mut Trajectory, mut places: Option<&mut Places<'_, 'm, D>>) -> Result<(), Refused> {
    let j = trajectory.jumps + 1;
    let mut stream = Stream::jump(trajectory.seed, trajectory.index, j);
    let mut threshold = stream.open_uniform();
    let (mut class, mut polarization) = (stream.uniform(), stream.uniform());
    let mut first = true;
    loop {
        let at = trajectory.point();
        // The class among the state's classes by their rates; none where they have no rate between them.
        let mut draw = |rates: &[f64]| {
            let total: f64 = rates.iter().sum();
            if !(total > 0.0) {
                return None;
            }
            let mut x = class * total;
            let mut pick = rates.len() - 1;
            for (i, w) in rates.iter().enumerate() {
                if x < *w {
                    pick = i;
                    break;
                }
                x -= w;
            }
            Some(pick)
        };
        let on = places.as_deref_mut().map(|p| {
            let s = p.s;
            let (middle, path) = p.of(trajectory.cluster);
            (middle, path, s)
        });
        let gram = |u: &[Internal; 3]| -> [[f64; 3]; 3] { std::array::from_fn(|a| std::array::from_fn(|b| inner(&u[a], &u[b]).re)) };
        // A photon whose amplitudes are no finite numbers is none, as a class's rate that is none (the one exit: the
        // direction's rejection would accept no direction of it).
        let emitted = model.emit(trajectory.cluster, &at, on, &trajectory.psi, &mut draw).map(|photon| (gram(&photon.amplitudes), photon));
        let Some((gram, photon)) = emitted.filter(|(g, _)| (g[0][0] + g[1][1] + g[2][2]).is_finite() && g[0][0] + g[1][1] + g[2][2] > 0.0) else {
            // A norm at its threshold with no photon to emit is refused; a landed state with none at once goes on as it is.
            if first {
                return Err(Refused::Emission);
            }
            break;
        };
        if first {
            trajectory.jumps = j;
        }
        let emitted = model.photon_work() + photon.work;
        trajectory.work += emitted;
        trajectory.jumped += emitted;
        let u = &photon.amplitudes;
        let trace = gram[0][0] + gram[1][1] + gram[2][2];
        let direction = loop {
            let n = stream.direction();
            let q: f64 = (0..3).map(|a| (0..3).map(|b| n[a] * gram[a][b] * n[b]).sum::<f64>()).sum();
            if stream.uniform() * trace <= trace - q {
                break n;
            }
        };
        let (e1, e2) = linalg::perpendicular(direction);
        let along = |e: [f64; 3]| -> Internal { (0..u[0].len()).map(|i| (0..3).map(|a| u[a][i] * e[a]).sum()).collect() };
        let (f1, f2) = (along(e1), along(e2));
        let (w1, w2) = (norm_sqr(&f1), norm_sqr(&f2));
        let (f, w) = if polarization * (w1 + w2) < w1 { (f1, w1) } else { (f2, w2) };
        for a in 0..3 {
            trajectory.kappa[a] += -photon.wave_number * direction[a] + photon.kick[a];
        }
        trajectory.scattered = [trajectory.scattered[0] + 1.0, trajectory.scattered[1] + photon.wave_number * photon.wave_number];
        let scale = 1.0 / w.sqrt();
        trajectory.psi = f.iter().map(|z| *z * scale).collect();
        trajectory.cluster = photon.landing;
        first = false;
        // The landing on the step's middles (`land`); none where it does not exist (refused, as a photon that is none);
        // against the next threshold.
        let at = trajectory.point();
        let Some((cluster, _, landed)) = land(model, trajectory.cluster, &at, places.as_deref_mut(), &trajectory.psi, threshold) else { return Err(Refused::Emission) };
        let at_once = norm_sqr(&landed.psi) <= landed.threshold;
        for a in 0..3 {
            trajectory.kappa[a] += landed.kick[a];
        }
        trajectory.work += landed.work;
        trajectory.jumped += landed.work;
        (trajectory.cluster, trajectory.psi, trajectory.threshold) = (cluster, landed.psi, landed.threshold);
        if !at_once {
            break;
        }
        threshold = stream.open_uniform();
        (class, polarization) = (stream.uniform(), stream.uniform());
    }
    Ok(())
}

/// The loss test at a trajectory's state (SPEC §1.7: at every step's end and after every jump -- the trajectory's own
/// states; a reading is none of them, adapt.rs # The reads), the one definition: its time once the model finds it
/// lost (`Dynamics::lost`).
pub fn test_loss<D: Dynamics>(model: &D, trajectory: &mut Trajectory) {
    if trajectory.lost.is_none() && model.lost(trajectory.cluster, &trajectory.point(), &trajectory.psi) {
        trajectory.lost = Some(trajectory.time);
    }
}

#[cfg(test)]
mod tests;
